import Foundation
import llama

private var llamaLogHandle: FileHandle?

private func llamaLogCallback(level: ggml_log_level, text: UnsafePointer<CChar>?, user: UnsafeMutableRawPointer?) {
    guard let text, let data = String(cString: text).data(using: .utf8) else { return }
    llamaLogHandle?.write(data)
}

/// Lines of the current trace: one F32 tensor per line, "name<TAB>op<TAB>sum", in graph order.
var llamaTraceLines: [String] = []

/// Records the sum of every F32 tensor that the graph computes, for comparison with llama-eval-callback on the Mac.
private func llamaTraceCallback(t: UnsafeMutablePointer<ggml_tensor>?, ask: Bool, user: UnsafeMutableRawPointer?) -> Bool {
    guard let t else { return true }
    if ask { return t.pointee.type == GGML_TYPE_F32 }
    let name = withUnsafeBytes(of: t.pointee.name) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
    var bytes = [UInt8](repeating: 0, count: ggml_nbytes(t))
    ggml_backend_tensor_get(t, &bytes, 0, bytes.count)
    let ne = [t.pointee.ne.0, t.pointee.ne.1, t.pointee.ne.2, t.pointee.ne.3].map(Int.init)
    let nb = [t.pointee.nb.0, t.pointee.nb.1, t.pointee.nb.2, t.pointee.nb.3]
    var sum = 0.0
    bytes.withUnsafeBytes { raw in
        for i3 in 0..<ne[3] { for i2 in 0..<ne[2] { for i1 in 0..<ne[1] { for i0 in 0..<ne[0] {
            sum += Double(raw.load(fromByteOffset: i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3], as: Float.self))
        } } } }
    }
    llamaTraceLines.append("\(name)\t\(String(cString: ggml_op_desc(t)))\t\(sum)")
    return true
}

enum LlamaRuntime {
    /// Routes llama.cpp logs to Documents/llama-log.txt and initializes the backend once. A marker line starts each
    /// launch, so the lines before a crash survive the relaunch.
    static let start: Void = {
        let url = Util.documents.appendingPathComponent("llama-log.txt")
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int
        if (size ?? .max) > 8 << 20 {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        llamaLogHandle = try? FileHandle(forWritingTo: url)
        _ = try? llamaLogHandle?.seekToEnd()
        llamaLogHandle?.write(Data("=== launch \(ISO8601DateFormatter().string(from: Date()))\n".utf8))
        llama_log_set(llamaLogCallback, nil)
        llama_backend_init()
    }()
}

/// One loaded model with one context. Not thread-safe; use from a single queue.
final class LlamaSession {
    let model: OpaquePointer
    let ctx: OpaquePointer
    let vocab: OpaquePointer?
    private let batch: Int

    /// False with the launch argument `-mmap 0`: the weights are copied into Metal buffers instead of mapping the file.
    static var mapsFile: Bool {
        UserDefaults.standard.object(forKey: "mmap") == nil || UserDefaults.standard.bool(forKey: "mmap")
    }

    /// `batch` bounds the tokens per decode call and, as the micro-batch, sizes the compute buffer.
    /// `trace` records every F32 tensor into llamaTraceLines; it synchronizes after each node and is slow.
    /// `speculative` runs the LFM3 draft head of the file in every decode and keeps one short-conv state snapshot,
    /// so a rejected draft token can be removed (see `draft` and `removeFrom`).
    init?(path: String, gpuLayers: Int32, threads: Int32, context: UInt32, batch: UInt32, trace: Bool = false,
          speculative: Bool = false) {
        _ = LlamaRuntime.start
        var mparams = llama_model_default_params()
        mparams.n_gpu_layers = gpuLayers
        if !LlamaSession.mapsFile {
            mparams.load_mode = LLAMA_LOAD_MODE_NONE
        }
        guard let model = llama_model_load_from_file(path, mparams) else { return nil }
        var cparams = llama_context_default_params()
        cparams.n_ctx = context
        cparams.n_batch = batch
        cparams.n_ubatch = batch
        cparams.n_threads = threads
        cparams.n_threads_batch = threads
        if trace { cparams.cb_eval = llamaTraceCallback }
        if speculative { cparams.n_rs_seq = 1 }
        // The A12Z GPU has no flash attention kernels on the compat path.
        if gpuLayers > 0 { cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED }
        // Two encoder threads were faster than the default on the A12Z (2026-10-25).
        if gpuLayers > 0 { setenv("GGML_METAL_N_CB", "2", 0) }
        guard let ctx = llama_init_from_model(model, cparams) else {
            llama_model_free(model)
            return nil
        }
        if speculative { llama_set_draft_head(ctx, true) }
        // llama.cpp clamps the batch to the context; a decode call above the effective value aborts the process.
        self.batch = Int(llama_n_batch(ctx))
        self.model = model
        self.ctx = ctx
        self.vocab = llama_model_get_vocab(model)
    }

    deinit {
        llama_free(ctx)
        llama_model_free(model)
    }

    func tokenize(_ text: String, addSpecial: Bool, parseSpecial: Bool) -> [llama_token] {
        let byteCount = Int32(text.utf8.count)
        var tokens = [llama_token](repeating: 0, count: Int(byteCount) + 16)
        var n = llama_tokenize(vocab, text, byteCount, &tokens, Int32(tokens.count), addSpecial, parseSpecial)
        if n < 0 {
            tokens = [llama_token](repeating: 0, count: Int(-n))
            n = llama_tokenize(vocab, text, byteCount, &tokens, Int32(tokens.count), addSpecial, parseSpecial)
        }
        return Array(tokens.prefix(Int(max(n, 0))))
    }

    /// Feeds tokens in batches of the context's batch size. Returns false if llama_decode fails.
    func decode(_ tokens: [llama_token]) -> Bool {
        var start = 0
        while start < tokens.count {
            let count = min(batch, tokens.count - start)
            var chunk = Array(tokens[start..<(start + count)])
            let rc = chunk.withUnsafeMutableBufferPointer { buf in
                llama_decode(ctx, llama_batch_get_one(buf.baseAddress, Int32(count)))
            }
            if rc != 0 { return false }
            start += count
        }
        return true
    }

    /// Decodes one batch after the last position with logits for every token, as a speculative-decoding verify step
    /// needs. `decode` asks only for the logits of the last token.
    func decodeAll(_ tokens: [llama_token]) -> Bool {
        var b = llama_batch_init(Int32(tokens.count), 0, 1)
        defer { llama_batch_free(b) }
        let pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1
        for (i, token) in tokens.enumerated() {
            b.token[i] = token
            b.pos[i] = pos + Int32(i)
            b.n_seq_id[i] = 1
            b.seq_id[i]![0] = 0
            b.logits[i] = 1
        }
        b.n_tokens = Int32(tokens.count)
        return llama_decode(ctx, b) == 0
    }

    /// The draft head's guess for the token after `token`, from output row `i` of the last decode (as in
    /// llama_get_logits_ith). The head assumed the greedy token of that row; nil if `token` is another one.
    func draft(_ i: Int32, after token: llama_token) -> llama_token? {
        guard let d = llama_get_draft_ith(ctx, i), llama_token(d[1]) == token else { return nil }
        return llama_token(d[0])
    }

    /// Removes the tokens from position `pos` on: a rejected draft token. Needs `speculative`.
    func removeFrom(_ pos: Int) -> Bool {
        llama_memory_seq_rm(llama_get_memory(ctx), 0, llama_pos(pos), -1)
    }

    /// Removes all tokens from the context, so the next decode starts at position 0.
    func clear() {
        llama_memory_clear(llama_get_memory(ctx), true)
    }

    /// The UTF-8 bytes of one token, special tokens included.
    func piece(_ token: llama_token) -> [UInt8] {
        var buf = [CChar](repeating: 0, count: 64)
        var n = llama_token_to_piece(vocab, token, &buf, Int32(buf.count), 0, true)
        if n < 0 {
            buf = [CChar](repeating: 0, count: Int(-n))
            n = llama_token_to_piece(vocab, token, &buf, Int32(buf.count), 0, true)
        }
        return buf.prefix(Int(max(n, 0))).map { UInt8(bitPattern: $0) }
    }
}
