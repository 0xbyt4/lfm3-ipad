import Foundation
import llama

/// Runs MUL_MAT and GET_ROWS on the backends of this device and compares them with a reference built from ggml's
/// own dequantization. The Mac Apple5 simulation cannot show errors of the real A12Z GPU compiler.
enum KernelTest {
    private static let types = [
        GGML_TYPE_Q6_K, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K, GGML_TYPE_Q3_K, GGML_TYPE_Q2_K,
        GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_F16,
    ]
    // (k, src0 rows, src1 rows): one short row, a tail that is not a multiple of 32, the tiled kernel.
    private static let shapes = [(256, 64, 1), (2048, 1000, 1), (2048, 256, 16)]

    /// The NMSE limit of test-backend-ops for MUL_MAT.
    private static let maxError = 5e-4

    /// One test case. `distinct` rows of real data repeat over all `m` rows, so a large matrix needs little
    /// reference work. `hostPtr` puts the weights in a buffer made from page-aligned host memory, as llama.cpp does
    /// for a memory-mapped model file.
    private struct Case {
        var op = "mul_mat"
        var type: ggml_type
        var k: Int
        var m: Int
        var n: Int
        var distinct: Int? = nil
        var hostPtr = false
        /// Weight magnitude. Small weights give block scales below the smallest normal half (6.1e-5).
        var scale: Float = 1
        /// Attention layout: `kv` src0 matrices, each shared by `r2` src1 heads (grouped-query attention).
        var kv = 1
        var r2 = 1
    }

    /// Returns one result per case: "backend", "op", "type", "shape", "nmse", "max_abs", "pass" or "skipped".
    static func run(report: (String) -> Void) -> [[String: Any]] {
        _ = LlamaRuntime.start
        guard let metal = ggml_backend_metal_init(), let cpu = ggml_backend_cpu_init() else {
            report("no backend")
            return []
        }
        defer {
            ggml_backend_free(metal)
            ggml_backend_free(cpu)
        }
        ggml_backend_cpu_set_n_threads(cpu, 4)

        var cases: [(ggml_backend_t, String, Case)] = []
        for type in types {
            for (k, m, n) in shapes {
                cases.append((metal, "metal", Case(type: type, k: k, m: m, n: n)))
            }
            cases.append((metal, "metal", Case(op: "get_rows", type: type, k: 2048, m: 64, n: 4)))
        }
        // The shapes and buffers of the LFM3 output layer (tied embedding, 128000 rows).
        for type in [GGML_TYPE_Q6_K, GGML_TYPE_Q8_0] {
            cases.append((metal, "metal", Case(type: type, k: 2048, m: 1000, n: 1, hostPtr: true)))
            cases.append((metal, "metal", Case(type: type, k: 2048, m: 128000, n: 1, distinct: 1000)))
            cases.append((metal, "metal", Case(type: type, k: 2048, m: 128000, n: 1, distinct: 1000, hostPtr: true)))
            cases.append((cpu, "cpu", Case(type: type, k: 2048, m: 1000, n: 1)))
            cases.append((cpu, "cpu", Case(op: "get_rows", type: type, k: 2048, m: 128000, n: 8, distinct: 1000)))
        }

        // Embedding-sized weights: the q6_K scales of the LFM3 token embedding are almost all subnormal halves.
        for type in types {
            cases.append((metal, "metal", Case(type: type, k: 2048, m: 1000, n: 1, scale: 0.004)))
            cases.append((metal, "metal", Case(type: type, k: 2048, m: 256, n: 16, scale: 0.004)))
            cases.append((metal, "metal", Case(op: "get_rows", type: type, k: 2048, m: 64, n: 4, scale: 0.004)))
        }

        // The LFM3 attention matmuls during generation: KQ (head_dim 128 x n_kv) and KQV (n_kv x head_dim),
        // 4 KV heads for 16 query heads. Odd row counts test the row tail of the K times Q kernel; compat needs
        // k % 32 == 0.
        for (k, m, kv, r2) in [(128, 4096, 4, 4), (4096, 128, 4, 4), (160, 1001, 4, 4), (64, 997, 2, 8), (96, 300, 3, 2)] {
            cases.append((metal, "metal", Case(type: GGML_TYPE_F16, k: k, m: m, n: 1, kv: kv, r2: r2)))
        }

        var results: [[String: Any]] = []
        var rng = SplitMix(seed: 1234)
        for (backend, name, c) in cases {
            var r = check(backend, c, rng: &rng)
            r["backend"] = name
            let status = r["skipped"] != nil ? "SKIP" : (r["pass"] as? Bool == true ? "ok  " : "FAIL")
            report(String(format: "%@ %@ %@ %@ %@%@%@ nmse %.2e", status, name, r["op"] as! String, r["type"] as! String,
                          r["shape"] as! String, c.hostPtr ? " host" : "", c.scale != 1 ? " small" : "",
                          r["nmse"] as? Double ?? 0))
            results.append(r)
        }
        return results
    }

    private static func check(_ backend: ggml_backend_t, _ c: Case, rng: inout SplitMix) -> [String: Any] {
        var result: [String: Any] = [
            "op": c.op, "type": String(cString: ggml_type_name(c.type)), "shape": "\(c.k)x\(c.m)x\(c.n)" + (c.kv*c.r2 > 1 ? " kv\(c.kv)r\(c.r2)" : ""),
            "host_ptr": c.hostPtr, "scale": c.scale,
        ]
        func skip(_ why: String) -> [String: Any] { result.merging(["skipped": why]) { $1 } }

        let params = ggml_init_params(mem_size: 4*ggml_tensor_overhead() + ggml_graph_overhead(), mem_buffer: nil,
                                      no_alloc: true)
        guard let ctx = ggml_init(params) else { return skip("ggml_init") }
        defer { ggml_free(ctx) }

        let heads = c.kv*c.r2
        let a = ggml_new_tensor_3d(ctx, c.type, Int64(c.k), Int64(c.m), Int64(c.kv))!
        let b = c.op == "mul_mat" ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Int64(c.k), Int64(c.n), Int64(heads))!
                                  : ggml_new_tensor_1d(ctx, GGML_TYPE_I32, Int64(c.n))!
        let out = c.op == "mul_mat" ? ggml_mul_mat(ctx, a, b)! : ggml_get_rows(ctx, a, b)!
        if !ggml_backend_supports_op(backend, out) { return skip("unsupported") }
        let graph = ggml_new_graph(ctx)
        ggml_build_forward_expand(graph, out)

        // The reference uses the dequantized weights, so it measures only the kernel, not the quantization.
        let total = c.m*c.kv
        let distinct = min(total, c.distinct ?? total)
        let rowSize = ggml_row_size(c.type, Int64(c.k))
        let weights = (0..<c.k*distinct).map { _ in rng.float()*c.scale }
        var rows = [UInt8](repeating: 0, count: rowSize*distinct)
        _ = ggml_quantize_chunk(c.type, weights, &rows, 0, Int64(distinct), Int64(c.k), nil)
        var dequant = [Float](repeating: 0, count: c.k*distinct)
        ggml_get_type_traits(c.type).pointee.to_float(rows, &dequant, Int64(c.k*distinct))
        var packed = [UInt8](repeating: 0, count: rowSize*total)
        for i in 0..<total {
            let src = (i % distinct)*rowSize
            packed.replaceSubrange(i*rowSize..<(i + 1)*rowSize, with: rows[src..<src + rowSize])
        }

        var hostMemory: UnsafeMutableRawPointer?
        var hostBuffer: ggml_backend_buffer_t?
        defer {
            if let hostBuffer { ggml_backend_buffer_free(hostBuffer) }
            free(hostMemory)
        }
        if c.hostPtr {
            let page = 16384
            let size = (packed.count + page - 1)/page*page
            guard posix_memalign(&hostMemory, page, size) == 0, let memory = hostMemory else { return skip("memalign") }
            packed.withUnsafeBytes { memory.copyMemory(from: $0.baseAddress!, byteCount: packed.count) }
            hostBuffer = ggml_backend_dev_buffer_from_host_ptr(ggml_backend_get_device(backend), memory, size, size)
            guard let hostBuffer, ggml_backend_tensor_alloc(hostBuffer, a, memory) == GGML_STATUS_SUCCESS else {
                return skip("host buffer")
            }
        }
        guard let buffer = ggml_backend_alloc_ctx_tensors(ctx, backend) else { return skip("alloc") }
        defer { ggml_backend_buffer_free(buffer) }
        if !c.hostPtr { ggml_backend_tensor_set(a, packed, 0, packed.count) }

        var expected = [Double]()
        if c.op == "mul_mat" {
            let x = (0..<c.k*c.n*heads).map { _ in rng.float() }
            ggml_backend_tensor_set(b, x, 0, x.count*4)
            // src1 head h uses src0 matrix h/r2, as ggml broadcasts it.
            for h in 0..<heads {
                for j in 0..<c.n {
                    let col = (h*c.n + j)*c.k
                    var dots = [Double](repeating: 0, count: distinct)
                    for i in 0..<distinct {
                        var s = 0.0
                        for l in 0..<c.k { s += Double(dequant[i*c.k + l])*Double(x[col + l]) }
                        dots[i] = s
                    }
                    for i in 0..<c.m { expected.append(dots[((h/c.r2)*c.m + i) % distinct]) }
                }
            }
        } else {
            let index = (0..<c.n).map { _ in Int32(rng.next() % UInt64(c.m)) }
            ggml_backend_tensor_set(b, index, 0, index.count*4)
            for r in index {
                let base = Int(r) % distinct
                expected += dequant[base*c.k..<(base + 1)*c.k].map(Double.init)
            }
        }

        guard ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS else { return skip("compute") }
        var got = [Float](repeating: 0, count: expected.count)
        ggml_backend_tensor_get(out, &got, 0, got.count*4)

        var err = 0.0, ref = 0.0, maxAbs = 0.0
        for (g, e) in zip(got, expected) {
            let d = Double(g) - e
            err += d*d
            ref += e*e
            maxAbs = max(maxAbs, abs(d))
        }
        let nmse = ref > 0 ? err/ref : err
        result["nmse"] = nmse.isFinite ? nmse : 1e30
        result["max_abs"] = maxAbs.isFinite ? maxAbs : 1e30
        result["pass"] = nmse.isFinite && nmse < maxError
        return result
    }
}

/// Small deterministic generator, so every run tests the same data.
struct SplitMix {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    /// Uniform in [-1, 1).
    mutating func float() -> Float { Float(next() >> 40)/Float(1 << 23) - 1 }
}
