import Foundation
import llama

/// One conversation on one model. The KV cache holds the whole conversation with the thinking of every turn (the
/// preserve_thinking form of the LFM3 chat template), so a turn decodes only its own new tokens. The short-conv
/// layers keep a recurrent state, so the cache can only grow or be cleared; it cannot drop old turns.
/// Not thread-safe; use from a single queue.
final class ChatEngine {
    static let context: UInt32 = 8192
    /// Tokens kept free for the reply when a turn starts.
    private static let replyReserve = 512

    struct Stats {
        var prefillRate = 0.0
        var generateRate = 0.0
        var tokens = 0
        var used = 0
        /// Speculative decoding: verify steps and the draft tokens they kept.
        var verifies = 0
        var accepted = 0
    }

    enum Outcome {
        case done
        case stopped
        case contextFull
        case failed
    }

    let session: LlamaSession
    /// Speculative decoding with the draft head of the file (d = 1). The sampler still picks every token, so the
    /// replies keep the same distribution; a draft token only saves a decode when the sampler picks it.
    let speculative: Bool
    private let sampler: UnsafeMutablePointer<llama_sampler>
    /// Tokens in the KV cache.
    private(set) var used = 0

    init?(path: String, speculative: Bool = false) {
        guard let session = LlamaSession(path: path, gpuLayers: 99, threads: 4, context: ChatEngine.context,
                                         batch: 256, speculative: speculative) else { return nil }
        self.session = session
        self.speculative = speculative
        // The sampling settings that the GGUF metadata recommends (general.sampling.*). The penalty comes after
        // top-k because it scans every candidate.
        sampler = llama_sampler_chain_init(llama_sampler_chain_default_params())
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(50))
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(llama_vocab_n_tokens(session.vocab), 64, 1.1, 0, 0))
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.1))
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(UInt32.random(in: 0...UInt32.max)))
    }

    deinit {
        llama_sampler_free(sampler)
    }

    func reset() {
        session.clear()
        llama_sampler_reset(sampler)
        used = 0
    }

    /// ASD-STE100 style rules for the answers, as a system prompt. The thinking is not addressed.
    static let steSystemPrompt = """
        Write your final answer in ASD-STE100 Simplified Technical English.
        - Give the result first, then the reason.
        - Write one idea per sentence. Keep each sentence under 20 words.
        - Use the active voice and simple, common words.
        - Use the same word for the same thing every time.
        - Write a procedure as numbered steps. Start each step with a verb.
        - Do not use idioms, metaphors or filler words.
        """

    /// BOS and the optional system message: the start of a conversation.
    private func start(system: String?) -> [llama_token] {
        var tokens = [llama_vocab_bos(session.vocab)]
        if let system {
            tokens += session.tokenize("<|im_start|>system\n", addSpecial: false, parseSpecial: true)
            tokens += session.tokenize(system, addSpecial: false, parseSpecial: false)
            tokens += session.tokenize("<|im_end|>\n", addSpecial: false, parseSpecial: true)
        }
        return tokens
    }

    /// One finished turn. `reply` is the generated text after the `<think>` or `<think></think>` prefix.
    struct Turn {
        var user: String
        var reply: String
        var think: Bool
    }

    /// Fills the cache with earlier turns of a conversation that is not the one in the cache. Returns false if
    /// the turns do not fit; the cache is then empty.
    func prime(_ turns: [Turn], system: String?) -> Bool {
        reset()
        guard !turns.isEmpty else { return true }
        var tokens = start(system: system)
        for turn in turns {
            tokens += session.tokenize("<|im_start|>user\n", addSpecial: false, parseSpecial: true)
            tokens += session.tokenize(turn.user, addSpecial: false, parseSpecial: false)
            tokens += session.tokenize("<|im_end|>\n<|im_start|>assistant\n" + (turn.think ? "<think>" : "<think></think>"),
                                       addSpecial: false, parseSpecial: true)
            tokens += session.tokenize(turn.reply + "<|im_end|>\n", addSpecial: false, parseSpecial: true)
        }
        guard tokens.count + ChatEngine.replyReserve <= Int(ChatEngine.context), session.decode(tokens) else {
            reset()
            return false
        }
        used = tokens.count
        return true
    }

    /// Answers one user message. `onText` gets the reply as it grows, `<think>` content included, and `onStats`
    /// the speed. Both run on the calling queue. `system` is used only when the cache is empty.
    func reply(to message: String, think: Bool, system: String?, stop: StopFlag, onText: (String) -> Void,
               onStats: (Stats) -> Void) -> Outcome {
        // The user text is tokenized without special tokens, so it cannot insert template markers.
        var turn = used == 0 ? start(system: system) : []
        turn += session.tokenize("<|im_start|>user\n", addSpecial: false, parseSpecial: true)
        turn += session.tokenize(message, addSpecial: false, parseSpecial: false)
        turn += session.tokenize("<|im_end|>\n<|im_start|>assistant\n" + (think ? "<think>" : "<think></think>"),
                                 addSpecial: false, parseSpecial: true)
        let limit = Int(ChatEngine.context)
        if used + turn.count + ChatEngine.replyReserve > limit { return .contextFull }

        var stats = Stats()
        let t0 = Util.now()
        guard session.decode(turn) else { return .failed }
        llama_synchronize(session.ctx)
        stats.prefillRate = Double(turn.count) / (Util.now() - t0)
        used += turn.count
        stats.used = used
        onStats(stats)

        var utf8 = UTF8Stream()
        var outcome = Outcome.done
        let start = Util.now()
        // speculative: `next` is sampled but not decoded yet; `draft` is the draft head's guess for the token after it
        var next: llama_token?
        var draft: llama_token?
        while true {
            if stop.isSet { outcome = .stopped; break }
            // two tokens stay free for the closing "<|im_end|>\n", one more for a draft token
            if used + (speculative ? 4 : 3) >= limit { outcome = .contextFull; break }
            let token = next ?? llama_sampler_sample(sampler, session.ctx, -1)
            next = nil
            if llama_vocab_is_eog(session.vocab, token) { break }
            var emitted = [token]
            if speculative, let d = draft, !llama_vocab_is_eog(session.vocab, d) {
                // verify step: decode [token, d]; d stays if the sampler picks it after token
                guard session.decodeAll([token, d]) else { outcome = .failed; break }
                stats.verifies += 1
                let y1 = llama_sampler_sample(sampler, session.ctx, 0)
                if y1 == d {
                    stats.accepted += 1
                    emitted.append(d)
                    let y2 = llama_sampler_sample(sampler, session.ctx, 1)
                    draft = session.draft(1, after: y2)
                    next = y2
                } else {
                    guard session.removeFrom(used + 1) else { outcome = .failed; break }
                    draft = session.draft(0, after: y1)
                    next = y1
                }
            } else {
                guard session.decode([token]) else { outcome = .failed; break }
                if speculative {
                    let y = llama_sampler_sample(sampler, session.ctx, -1)
                    draft = session.draft(-1, after: y)
                    next = y
                }
            }
            used += emitted.count
            for e in emitted {
                stats.tokens += 1
                let text = utf8.push(session.piece(e))
                if !text.isEmpty { onText(text) }
            }
            if stats.tokens % 8 < emitted.count {
                stats.generateRate = Double(stats.tokens) / (Util.now() - start)
                stats.used = used
                onStats(stats)
            }
        }
        if stats.tokens > 0 { stats.generateRate = Double(stats.tokens) / (Util.now() - start) }

        // Close the turn as the template does, also after a stop, so the next turn continues a valid conversation.
        let end = session.tokenize("<|im_end|>\n", addSpecial: false, parseSpecial: true)
        if outcome != .failed, used + end.count <= limit, session.decode(end) {
            used += end.count
        }
        stats.used = used
        onStats(stats)
        return outcome
    }
}

/// Joins token bytes into text. A token can end inside a multi-byte UTF-8 character; those bytes wait for the next
/// token.
struct UTF8Stream {
    private var pending: [UInt8] = []

    mutating func push(_ bytes: [UInt8]) -> String {
        pending += bytes
        var cut = pending.count
        var i = pending.count - 1
        var continuation = 0
        while i >= 0 && continuation < 3 && pending[i] & 0xC0 == 0x80 {
            i -= 1
            continuation += 1
        }
        if i >= 0 {
            let lead = pending[i]
            let length = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1
            if length > continuation + 1 { cut = i }
        }
        let text = String(decoding: pending[..<cut], as: UTF8.self)
        pending.removeFirst(cut)
        return text
    }
}
