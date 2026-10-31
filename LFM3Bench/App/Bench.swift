import Foundation
import llama

struct BenchOptions {
    var seconds: Double = 180
    var waitCool = true
    var context: UInt32 = 4096
    var batch: UInt32 = 256
    var threads: Int32 = 4
    var gpuLayers: Int32 = 99
    /// Speculative decoding with the draft head of the file (d = 1); the output stays the greedy one.
    var speculative = false
    /// Generation prompts, one after another at each restart; empty: the essay prompt.
    var prompts: [String] = []
    /// Start position of the verify-cost run (at most ~6000: the run moves ~2000 tokens on in an 8192 context).
    var verifyAt = 1024
}

/// Measures one GGUF file: load time, prompt speed, and sustained greedy generation with heat, memory and main-thread
/// lag. Every measurement is one JSON line in Documents/results.jsonl.
final class Bench {
    private static let prefillTokens = 512
    private static let window = 32

    // The same prompt as the Mac reference run, so the first tokens can be compared.
    private static let prompt = """
        <|im_start|>user
        Write a long, detailed essay about the history of bridges, from ancient stone arches to modern \
        cable-stayed designs. Cover the engineering ideas, the materials and the famous failures.<|im_end|>
        <|im_start|>assistant
        <think>
        """

    private static let filler = """
        The river ran past the old mill and under the stone bridge at the edge of the town. In spring the water \
        rose and carried branches, mud and the occasional lost boat down to the lake. The miller kept a log of \
        every flood, with the date, the height of the water on the wall and the damage it did to the wheel. \
        Over forty years the log filled three books, and the town council used them to decide where to build \
        the new embankment.
        """

    private let out = OutputFile(name: "results.jsonl")
    private let stop: StopFlag
    private let lag: MainLagProbe
    private let report: (String) -> Void

    init(stop: StopFlag, lag: MainLagProbe, report: @escaping (String) -> Void) {
        self.stop = stop
        self.lag = lag
        self.report = report
    }

    func run(path: String, options: BenchOptions) {
        let name = URL(fileURLWithPath: path).lastPathComponent
        let runID = ISO8601DateFormatter().string(from: Date())
        func record(_ event: String, _ fields: [String: Any]) {
            var line = Util.state()
            line["event"] = event
            line["run"] = runID
            line["file"] = name
            line.merge(fields) { _, new in new }
            if let data = try? JSONSerialization.data(withJSONObject: line, options: [.sortedKeys]) {
                out.write(String(decoding: data, as: UTF8.self))
            }
        }

        record("start", [
            "seconds": options.seconds, "context": options.context, "batch": options.batch,
            "threads": options.threads, "gpu_layers": options.gpuLayers,
        ])
        report("== \(name)")
        if options.waitCool {
            let waited = waitForNominal()
            record("cool", ["waited_s": waited])
        }

        let t0 = Util.now()
        guard let session = LlamaSession(path: path, gpuLayers: options.gpuLayers, threads: options.threads,
                                         context: options.context, batch: options.batch,
                                         speculative: options.speculative) else {
            record("error", ["stage": "load"])
            report("load failed, see llama-log.txt")
            return
        }
        let loadSeconds = Util.now() - t0
        record("load", ["load_s": loadSeconds])
        report(String(format: "load %.1f s, footprint %d MiB", loadSeconds, Util.mib(Util.footprint())))

        guard prefill(session, record: record) else { return }
        generate(session, options: options, record: record)
    }

    /// Runs one forward pass of a short prompt and writes the sum of every F32 tensor to Documents/trace-<file>.txt.
    func trace(path: String, options: BenchOptions) {
        let name = URL(fileURLWithPath: path).lastPathComponent
        llamaTraceLines = []
        guard let session = LlamaSession(path: path, gpuLayers: options.gpuLayers, threads: options.threads,
                                         context: options.context, batch: options.batch, trace: true) else {
            report("load failed, see llama-log.txt")
            return
        }
        let tokens = session.tokenize("The capital of France is", addSpecial: true, parseSpecial: false)
        guard session.decode(tokens) else {
            report("trace decode failed")
            return
        }
        let url = Util.documents.appendingPathComponent("trace-\(name).txt")
        let text = "tokens\t\(tokens.map(String.init).joined(separator: ","))\n" + llamaTraceLines.joined(separator: "\n") + "\n"
        try? text.write(to: url, atomically: true, encoding: .utf8)
        report("trace: \(llamaTraceLines.count) tensors -> \(url.lastPathComponent)")
        llamaTraceLines = []
    }

    private static let depths = [256, 1024, 2048, 3072, 4096, 6144, 7936]
    private static let depthWarmup = 32
    private static let depthTokens = 128

    /// Generation speed at growing context depths: fills the cache with filler text up to each depth, generates
    /// 32 greedy tokens untimed (the first tokens after a prefill are slow), then times 128. One "depth" record
    /// per depth.
    func depth(path: String, options: BenchOptions) {
        let name = URL(fileURLWithPath: path).lastPathComponent
        let runID = ISO8601DateFormatter().string(from: Date())
        let context: UInt32 = 8192
        if options.waitCool { _ = waitForNominal() }
        guard let session = LlamaSession(path: path, gpuLayers: options.gpuLayers, threads: options.threads,
                                         context: context, batch: options.batch) else {
            report("load failed, see llama-log.txt")
            return
        }
        guard let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params()) else { return }
        defer { llama_sampler_free(sampler) }
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy())

        var filler = session.tokenize(Bench.filler, addSpecial: true, parseSpecial: false)
        while filler.count < Int(context) {
            filler += session.tokenize(" " + Bench.filler, addSpecial: false, parseSpecial: false)
        }
        var pos = 0
        for depth in Bench.depths where !stop.isSet {
            guard session.decode(Array(filler[pos..<depth])) else {
                report("fill decode failed at \(pos)")
                return
            }
            pos = depth
            var seconds = 0.0
            for i in 0..<(Bench.depthWarmup + Bench.depthTokens) {
                let t = Util.now()
                let token = llama_sampler_sample(sampler, session.ctx, -1)
                guard session.decode([token]) else {
                    report("decode failed at \(pos)")
                    return
                }
                if i >= Bench.depthWarmup { seconds += Util.now() - t }
                pos += 1
            }
            let rate = Double(Bench.depthTokens) / seconds
            var line = Util.state()
            line.merge(["event": "depth", "run": runID, "file": name, "pos": depth, "tok_s": rate,
                        "ms_per_token": 1000 * seconds / Double(Bench.depthTokens)]) { _, new in new }
            if let data = try? JSONSerialization.data(withJSONObject: line, options: [.sortedKeys]) {
                out.write(String(decoding: data, as: UTF8.self))
            }
            report(String(format: "depth %4d: %.1f tok/s", depth, rate))
        }
    }

    private static let verifyWarmup = 4
    private static let verifyIterations = 24

    /// Times one decode of n = 1...8 tokens from position options.verifyAt on, with logits for the last token only and
    /// for all tokens. A speculative-decoding verify step of a draft with n - 1 tokens costs the all-logits time; with
    /// options.speculative the draft head runs in each all-logits decode too.
    func verifyCost(path: String, options: BenchOptions) {
        let name = URL(fileURLWithPath: path).lastPathComponent
        let runID = ISO8601DateFormatter().string(from: Date())
        if options.waitCool { _ = waitForNominal() }
        guard let session = LlamaSession(path: path, gpuLayers: options.gpuLayers, threads: options.threads,
                                         context: 8192, batch: options.batch,
                                         speculative: options.speculative) else {
            report("load failed, see llama-log.txt")
            return
        }
        var filler = session.tokenize(Bench.filler, addSpecial: true, parseSpecial: false)
        while filler.count < 8192 {
            filler += session.tokenize(" " + Bench.filler, addSpecial: false, parseSpecial: false)
        }
        guard session.decode(Array(filler[0..<options.verifyAt])) else {
            report("fill decode failed")
            return
        }
        var pos = options.verifyAt
        for allLogits in [false, true] {
            for n in 1...8 where !stop.isSet {
                var seconds = 0.0
                for i in 0..<(Bench.verifyWarmup + Bench.verifyIterations) {
                    let tokens = Array(filler[pos..<(pos + n)])
                    let t = Util.now()
                    guard allLogits ? session.decodeAll(tokens) : session.decode(tokens) else {
                        report("decode failed at \(pos)")
                        return
                    }
                    llama_synchronize(session.ctx)
                    if i >= Bench.verifyWarmup { seconds += Util.now() - t }
                    pos += n
                }
                let ms = 1000 * seconds / Double(Bench.verifyIterations)
                var line = Util.state()
                line.merge(["event": "verify", "run": runID, "file": name, "n": n, "all_logits": allLogits,
                            "ms": ms, "pos": pos, "speculative": options.speculative]) { _, new in new }
                if let data = try? JSONSerialization.data(withJSONObject: line, options: [.sortedKeys]) {
                    out.write(String(decoding: data, as: UTF8.self))
                }
                report(String(format: "verify n %d %@: %.2f ms", n, allLogits ? "all logits" : "last logits", ms))
            }
        }
    }

    /// Waits until the device is cool again, so each run starts from the same state. Gives up after 10 minutes.
    private func waitForNominal() -> Double {
        let start = Util.now()
        while Util.thermal > 0 && !stop.isSet && Util.now() - start < 600 {
            report("waiting for nominal thermal: \(Util.thermalNames[Util.thermal])")
            Thread.sleep(forTimeInterval: 5)
        }
        return Util.now() - start
    }

    /// Feeds the same 512 tokens three times on an empty context.
    private func prefill(_ session: LlamaSession, record: (String, [String: Any]) -> Void) -> Bool {
        var tokens = session.tokenize(Bench.filler, addSpecial: true, parseSpecial: false)
        while tokens.count < Bench.prefillTokens {
            tokens += session.tokenize(" " + Bench.filler, addSpecial: false, parseSpecial: false)
        }
        tokens = Array(tokens.prefix(Bench.prefillTokens))

        _ = lag.takeMax()
        for iteration in 1...3 where !stop.isSet {
            session.clear()
            let t = Util.now()
            guard session.decode(tokens) else {
                record("error", ["stage": "prefill"])
                report("prefill decode failed")
                return false
            }
            llama_synchronize(session.ctx)
            let rate = Double(tokens.count) / (Util.now() - t)
            record("prefill", ["iteration": iteration, "tokens": tokens.count, "tok_s": rate,
                               "main_lag_ms": lag.takeMax()])
            report(String(format: "prefill %d: %.1f tok/s", iteration, rate))
        }
        return true
    }

    /// Greedy generation for the given time. The context restarts from the next prompt at an end-of-generation token
    /// or when it is full; the restart time does not count as generation time.
    private func generate(_ session: LlamaSession, options: BenchOptions, record: (String, [String: Any]) -> Void) {
        guard let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params()) else { return }
        defer { llama_sampler_free(sampler) }
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy())

        let prompts = (options.prompts.isEmpty ? [Bench.prompt] : options.prompts)
            .map { session.tokenize($0, addSpecial: true, parseSpecial: true) }
        let nctx = Int(llama_n_ctx(session.ctx))
        var pos = 0
        var turn = 0
        func restart() -> Bool {
            session.clear()
            llama_sampler_reset(sampler)
            let prompt = prompts[turn % prompts.count]
            turn += 1
            pos = prompt.count
            return session.decode(prompt)
        }
        guard restart() else {
            record("error", ["stage": "prompt"])
            report("prompt decode failed")
            return
        }

        var total = 0
        var restarts = 0
        var genTime = 0.0
        var maxLag = 0.0
        var seriousAt: Double?
        var maxThermal = Util.thermal
        var firstTokens: [Int] = []
        var firstBytes: [UInt8] = []
        var failed = false

        // speculative: verified drafts, verify steps
        var accepted = 0
        var verifies = 0

        let start = Util.now()
        var windowStart = start
        var windowTokens = 0
        var windowTime = 0.0
        _ = lag.takeMax()
        // `next` is sampled but not decoded yet; `draft` is the draft head's guess for the token after it
        var next: llama_token?
        var draft: llama_token?
        while !stop.isSet && Util.now() - start < options.seconds {
            let t = Util.now()
            let token = next ?? llama_sampler_sample(sampler, session.ctx, -1)
            next = nil
            if llama_vocab_is_eog(session.vocab, token) || pos + 2 >= nctx {
                restarts += 1
                draft = nil
                guard restart() else { failed = true; break }
                continue
            }
            var emitted = [token]
            if options.speculative, let d = draft, !llama_vocab_is_eog(session.vocab, d) {
                // verify step: decode [token, d]; d stays if the model picks it after token
                guard session.decodeAll([token, d]) else { failed = true; break }
                verifies += 1
                let y1 = llama_sampler_sample(sampler, session.ctx, 0)
                if y1 == d {
                    accepted += 1
                    emitted.append(d)
                    let y2 = llama_sampler_sample(sampler, session.ctx, 1)
                    draft = session.draft(1, after: y2)
                    next = y2
                } else {
                    guard session.removeFrom(pos + 1) else { failed = true; break }
                    draft = session.draft(0, after: y1)
                    next = y1
                }
            } else {
                guard session.decode([token]) else { failed = true; break }
                if options.speculative {
                    let y = llama_sampler_sample(sampler, session.ctx, -1)
                    draft = session.draft(-1, after: y)
                    next = y
                }
            }
            let dt = Util.now() - t
            pos += emitted.count
            total += emitted.count
            genTime += dt
            windowTime += dt
            windowTokens += emitted.count
            for e in emitted where restarts == 0 && firstTokens.count < 128 {
                firstTokens.append(Int(e))
                firstBytes += session.piece(e)
            }

            if windowTokens >= Bench.window {
                let now = Util.now()
                let thermal = Util.thermal
                maxThermal = max(maxThermal, thermal)
                if seriousAt == nil && thermal >= 2 { seriousAt = now - start }
                let lagMs = lag.takeMax()
                maxLag = max(maxLag, lagMs)
                let rate = Double(windowTokens) / windowTime
                record("decode", [
                    "elapsed_s": now - start, "tokens": total, "tok_s": rate,
                    "wall_tok_s": Double(windowTokens) / (now - windowStart), "main_lag_ms": lagMs,
                    "pos": pos, "restarts": restarts,
                ])
                report(String(format: "%4.0f s  %5.1f tok/s  %@  lag %.0f ms", now - start, rate,
                              Util.thermalNames[thermal], lagMs))
                windowStart = now
                windowTokens = 0
                windowTime = 0
            }
        }

        if failed {
            record("error", ["stage": "decode", "tokens": total])
            report("decode failed after \(total) tokens")
        }
        let elapsed = Util.now() - start
        let average = genTime > 0 ? Double(total) / genTime : 0
        var summary: [String: Any] = [
            "tokens": total, "elapsed_s": elapsed, "tok_s": average,
            "wall_tok_s": elapsed > 0 ? Double(total) / elapsed : 0, "restarts": restarts,
            "max_thermal": maxThermal, "max_main_lag_ms": maxLag, "stopped": stop.isSet,
            "first_tokens": firstTokens, "first_text": String(decoding: firstBytes, as: UTF8.self),
            "speculative": options.speculative, "verifies": verifies, "accepted": accepted,
        ]
        if let seriousAt { summary["serious_at_s"] = seriousAt }
        record("summary", summary)
        report(String(format: "summary: %d tokens, %.1f tok/s, max thermal %@, serious at %@", total, average,
                      Util.thermalNames[maxThermal], seriousAt.map { String(format: "%.0f s", $0) } ?? "never"))
    }
}
