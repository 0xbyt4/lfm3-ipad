import Foundation
import llama

/// Times the generation attention of the LFM3 full-attention layers on the Metal backend, without the rest of the
/// model. Thermal state changes the end-to-end tok/s by more than the attention kernels cost, so kernel changes are
/// compared here. The tensors have the shapes and strides of the non-flash-attention graph with one query token:
/// K is a strided view of a [n_embd_k_gqa, n_ctx] cache, V a transposed view of a [n_ctx, n_embd_v_gqa] cache.
enum AttentionBench {
    private static let headDim = 128
    private static let kvHeads = 4
    private static let heads = 16
    private static let context = 8192
    /// The full-attention layers of LFM3-1.7B. Each layer has its own cache, so no layer reads another's data.
    private static let layers = 6
    private static let depths = [1024, 4096, 8192]
    private static let warmup = 5
    private static let iterations = 40

    /// Returns one result per depth and part: "n_kv", "part" (kq, kq_softmax or full) and "ms_per_layer".
    static func run(report: (String) -> Void) -> [[String: Any]] {
        _ = LlamaRuntime.start
        guard let metal = ggml_backend_metal_init() else {
            report("no backend")
            return []
        }
        defer { ggml_backend_free(metal) }

        var results: [[String: Any]] = []
        for nkv in depths {
            for part in ["kq", "kq_softmax", "full"] {
                guard let ms = time(metal, nkv: nkv, part: part) else {
                    report("attention \(part) n_kv \(nkv) failed")
                    continue
                }
                report(String(format: "attention %@ n_kv %d: %.3f ms per layer", part, nkv, ms))
                results.append(["n_kv": nkv, "part": part, "ms_per_layer": ms])
            }
        }
        return results
    }

    private static func time(_ backend: ggml_backend_t, nkv: Int, part: String) -> Double? {
        let params = ggml_init_params(mem_size: 16*layers*ggml_tensor_overhead() + ggml_graph_overhead(),
                                      mem_buffer: nil, no_alloc: true)
        guard let ctx = ggml_init(params) else { return nil }
        defer { ggml_free(ctx) }

        let kvWidth = headDim*kvHeads
        let half = MemoryLayout<UInt16>.size
        let graph = ggml_new_graph(ctx)
        var caches: [UnsafeMutablePointer<ggml_tensor>] = []
        var queries: [UnsafeMutablePointer<ggml_tensor>] = []
        for _ in 0..<layers {
            let kCache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, Int64(kvWidth), Int64(context))!
            let vCache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, Int64(context), Int64(kvWidth))!
            let q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Int64(headDim), 1, Int64(heads))!
            caches += [kCache, vCache]
            queries.append(q)
            let k = ggml_view_3d(ctx, kCache, Int64(headDim), Int64(nkv), Int64(kvHeads),
                                 kvWidth*half, headDim*half, 0)!
            let v = ggml_view_3d(ctx, vCache, Int64(nkv), Int64(headDim), Int64(kvHeads),
                                 context*half, headDim*context*half, 0)!
            var out = ggml_mul_mat(ctx, k, q)!
            if part != "kq" { out = ggml_soft_max(ctx, out)! }
            if part == "full" { out = ggml_mul_mat(ctx, v, out)! }
            ggml_build_forward_expand(graph, out)
        }
        guard let buffer = ggml_backend_alloc_ctx_tensors(ctx, backend) else { return nil }
        defer { ggml_backend_buffer_free(buffer) }

        // Small values, so the softmax inputs stay in a normal range.
        var rng = SplitMix(seed: 99)
        let pattern = (0..<65536).map { _ in Float16(rng.float()*0.1).bitPattern }
        let cacheData = (0..<kvWidth*context).map { pattern[$0 & 65535] }
        for cache in caches { ggml_backend_tensor_set(cache, cacheData, 0, cacheData.count*half) }
        let qData = (0..<headDim*heads).map { _ in rng.float() }
        for q in queries { ggml_backend_tensor_set(q, qData, 0, qData.count*4) }

        for _ in 0..<warmup {
            guard ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS else { return nil }
        }
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0..<iterations {
            guard ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS else { return nil }
        }
        let ns = Double(DispatchTime.now().uptimeNanoseconds - start)
        return ns/1e6/Double(iterations*layers)
    }
}
