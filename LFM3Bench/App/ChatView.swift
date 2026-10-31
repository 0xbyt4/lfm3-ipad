import SwiftUI
import UIKit

struct ChatMessage: Identifiable, Codable {
    var id = UUID()
    let user: Bool
    /// The reply started with an open `<think>`, so the text up to `</think>` is reasoning.
    let thinks: Bool
    var text = ""
    /// Generated tokens and speed of a finished reply.
    var tokens: Int?
    var rate: Double?

    var thinking: String? {
        guard thinks else { return nil }
        return text.components(separatedBy: "</think>").first?.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var answer: String {
        guard thinks else { return text.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard let range = text.range(of: "</think>") else { return "" }
        return text[range.upperBound...].trimmingCharacters(in: .whitespacesAndNewlines)
    }

    var thinkingDone: Bool { !thinks || text.contains("</think>") }
}

struct Conversation: Identifiable, Codable {
    var id = UUID()
    var title = "New chat"
    var messages: [ChatMessage] = []
    var updated = Date()
    /// The system prompt, fixed at the first message because the KV cache starts with it.
    var system: String?

    /// The finished turns: user messages with the reply after them.
    var turns: [ChatEngine.Turn] {
        stride(from: 0, to: messages.count - 1, by: 2).compactMap { i in
            let (user, reply) = (messages[i], messages[i + 1])
            guard user.user, !reply.user else { return nil }
            return ChatEngine.Turn(user: user.text, reply: reply.text, think: reply.thinks)
        }
    }
}

final class ChatModel: ObservableObject {
    static let shared = ChatModel()

    @Published var files: [String] = []
    @Published var selected = ""
    @Published private(set) var loaded: String?
    @Published private(set) var loading = false
    @Published private(set) var generating = false
    @Published var think = true
    /// New conversations answer in ASD-STE100 style. Off by default: with it, LFM3-1.7B spent most of its thinking
    /// on counting words and gave a worse answer (2026-10-29, 2118 against 814 thinking words).
    @Published var steStyle = UserDefaults.standard.object(forKey: "steStyle") as? Bool ?? false {
        didSet { UserDefaults.standard.set(steStyle, forKey: "steStyle") }
    }
    /// Speculative decoding with a draft-head copy of the model (scripts/add_draft_head.py). Off by default: on the
    /// iPad it was 0-10% faster with realistic prompts (2026-10-29).
    @Published private(set) var speculative = UserDefaults.standard.bool(forKey: "chatSpec") {
        didSet { UserDefaults.standard.set(speculative, forKey: "chatSpec") }
    }
    @Published private(set) var conversations: [Conversation] = []
    @Published private(set) var currentID: UUID?
    @Published private(set) var status = "No model loaded"
    @Published private(set) var stats = ChatEngine.Stats()
    @Published private(set) var remote = false
    /// The conversation list; closed at launch.
    @Published var showSidebar = false
    /// ProcessInfo.thermalState, updated when iOS reports a change.
    @Published private(set) var thermal = Util.thermal
    private var thermalObserver: NSObjectProtocol?

    private let queue = DispatchQueue(label: "chat", qos: .userInitiated)
    private let stopFlag = StopFlag()
    private var engine: ChatEngine?
    /// The conversation whose turns are in the KV cache. Main thread only.
    private var cachedID: UUID?
    private var remoteTimer: Timer?
    private static let store = Util.documents.appendingPathComponent("conversations.json")

    var busy: Bool { loading || generating }

    var messages: [ChatMessage] { current?.messages ?? [] }

    /// A model file with a draft head, for speculative decoding. The Q4_0 draft head was a little faster than the
    /// Q8_0 one with the same acceptance (2026-10-29).
    var draftFile: String? { files.first { $0.contains("draftq4") } ?? files.first { $0.contains("draft") } }

    /// The files the model menu offers: with speculative decoding the copies with a draft head, else the others.
    var menuFiles: [String] { files.filter { $0.contains("draft") == speculative } }

    /// The STE setting of the current conversation, or of the next one while it is empty.
    var steActive: Bool {
        guard let current, !current.messages.isEmpty else { return steStyle }
        return current.system != nil
    }

    private var current: Conversation? { conversations.first { $0.id == currentID } }

    init() {
        thermalObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: .main
        ) { [weak self] _ in self?.thermal = Util.thermal }
        if let data = try? Data(contentsOf: ChatModel.store),
           let saved = try? JSONDecoder().decode([Conversation].self, from: data) {
            conversations = saved
            currentID = saved.first?.id
        }
    }

    func refresh() {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: Util.documents.path)) ?? []
        files = names.filter { $0.hasSuffix(".gguf") }.sorted()
        // The Q8_0 embedding file was the fastest on the A12Z (2026-10-28); its copy with the 1-bit output sketch
        // generated 13% faster with the same tokens (2026-10-29).
        if speculative && draftFile == nil { speculative = false }
        if !menuFiles.contains(selected) {
            selected = speculative ? draftFile ?? ""
                : menuFiles.first { $0.contains("sketch") } ?? menuFiles.first { $0.contains("embQ8") } ?? menuFiles.first ?? ""
        }
    }

    func load() {
        guard !busy, !BenchModel.shared.running, !selected.isEmpty else { return }
        let name = selected
        // the draft head runs only with a file that has one
        let speculative = self.speculative && name.contains("draft")
        loading = true
        cachedID = nil
        status = "Loading \(name)"
        queue.async { [weak self] in
            self?.engine = nil
            let t0 = Util.now()
            let engine = ChatEngine(path: Util.documents.appendingPathComponent(name).path, speculative: speculative)
            let seconds = Util.now() - t0
            self?.engine = engine
            DispatchQueue.main.async {
                guard let self else { return }
                self.loading = false
                self.loaded = engine == nil ? nil : name
                self.stats = ChatEngine.Stats()
                self.status = engine == nil ? "Load failed, see llama-log.txt" : String(format: "Loaded in %.1f s", seconds)
            }
        }
    }

    /// Turns speculative decoding on or off. The model menu then offers the other set of files; a loaded model
    /// loads again from that set.
    func setSpeculative(_ on: Bool) {
        guard !busy, on != speculative, !on || draftFile != nil else { return }
        speculative = on
        refresh()
        if loaded != nil { load() }
    }

    func unload() {
        guard !busy else { return }
        queue.async { [weak self] in self?.engine = nil }
        loaded = nil
        cachedID = nil
        stats = ChatEngine.Stats()
        status = "No model loaded"
    }

    /// Selects an empty conversation, or creates one.
    func newChat() {
        guard !busy else { return }
        if let empty = conversations.first(where: { $0.messages.isEmpty }) {
            currentID = empty.id
            return
        }
        let conversation = Conversation()
        conversations.insert(conversation, at: 0)
        currentID = conversation.id
    }

    func select(_ id: UUID?) {
        guard !busy, let id else { return }
        currentID = id
    }

    func delete(_ id: UUID) {
        guard !busy else { return }
        conversations.removeAll { $0.id == id }
        if cachedID == id { cachedID = nil }
        if currentID == id { currentID = conversations.first?.id }
        save()
    }

    func send(_ text: String) {
        let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !busy, loaded != nil, !text.isEmpty else { return }
        if current == nil { newChat() }
        guard let id = currentID, let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        let think = self.think
        // A conversation that is not in the cache is read again from its saved turns.
        let prime = cachedID == id ? nil : conversations[index].turns
        cachedID = id
        if conversations[index].messages.isEmpty {
            conversations[index].title = String(text.prefix(40))
            conversations[index].system = steStyle ? ChatEngine.steSystemPrompt : nil
        }
        let system = conversations[index].system
        conversations[index].messages.append(ChatMessage(user: true, thinks: false, text: text))
        conversations[index].messages.append(ChatMessage(user: false, thinks: think))
        conversations[index].updated = Date()
        generating = true
        stats = ChatEngine.Stats()
        stopFlag.set(false)
        UIApplication.shared.isIdleTimerDisabled = true
        status = prime.map { $0.isEmpty ? "Reading the prompt" : "Reading \($0.count) earlier turns" } ?? "Reading the prompt"
        queue.async { [weak self, stopFlag] in
            guard let self else { return }
            guard let engine = self.engine else {
                DispatchQueue.main.async { self.finish(id, outcome: .failed) }
                return
            }
            if let prime, !engine.prime(prime, system: system) {
                DispatchQueue.main.async {
                    self.cachedID = nil
                    self.finish(id, outcome: .contextFull)
                }
                return
            }
            let outcome = engine.reply(to: text, think: think, system: system, stop: stopFlag, onText: { piece in
                DispatchQueue.main.async { self.append(piece, to: id) }
            }, onStats: { stats in
                DispatchQueue.main.async {
                    self.stats = stats
                    self.status = self.describe(stats)
                }
            })
            DispatchQueue.main.async { self.finish(id, outcome: outcome) }
        }
    }

    func stop() {
        stopFlag.set(true)
    }

    private func append(_ piece: String, to id: UUID) {
        guard let index = conversations.firstIndex(where: { $0.id == id }), !conversations[index].messages.isEmpty
        else { return }
        conversations[index].messages[conversations[index].messages.count - 1].text += piece
    }

    private func finish(_ id: UUID, outcome: ChatEngine.Outcome) {
        generating = false
        UIApplication.shared.isIdleTimerDisabled = false
        switch outcome {
        case .done: status = describe(stats)
        case .stopped: status = "Stopped. " + describe(stats)
        case .contextFull: status = "Context full (\(ChatEngine.context) tokens). Start a new chat."
        case .failed: status = "Decode failed, see llama-log.txt"
        }
        if let index = conversations.firstIndex(where: { $0.id == id }), stats.tokens > 0,
           let last = conversations[index].messages.indices.last, !conversations[index].messages[last].user {
            conversations[index].messages[last].tokens = stats.tokens
            conversations[index].messages[last].rate = stats.generateRate
        }
        // A turn that never started is not part of the conversation.
        if outcome == .contextFull || outcome == .failed,
           let index = conversations.firstIndex(where: { $0.id == id }),
           conversations[index].messages.last?.text.isEmpty == true {
            conversations[index].messages.removeLast(2)
            if outcome == .failed { cachedID = nil }
        }
        save()
    }

    private func save() {
        let kept = conversations.filter { !$0.messages.isEmpty }
        if let data = try? JSONEncoder().encode(kept) { try? data.write(to: ChatModel.store, options: .atomic) }
    }

    private func describe(_ s: ChatEngine.Stats) -> String {
        let draft = s.verifies > 0 ? String(format: ", draft %.0f%%", 100 * Double(s.accepted) / Double(s.verifies)) : ""
        return String(format: "%.1f tok/s, prompt %.0f tok/s, context %d/%d, %@", s.generateRate, s.prefillRate,
                      s.used, ChatEngine.context, Util.thermalNames[Util.thermal]) + draft
    }

    // MARK: Remote control

    private static let remoteDir = Util.documents.appendingPathComponent("remote")

    /// Debug builds only, with the launch argument `-remote 1`: each second the app reads remote/command.json,
    /// deletes it and runs it, then writes its state to remote/state.json. A Mac drives the chat with devicectl
    /// file copies while the screen shows it. Commands: {"cmd": "load", "file": name}, {"cmd": "new"},
    /// {"cmd": "send", "text": message, "think": bool, "ste": bool}, {"cmd": "stop"}, {"cmd": "unload"},
    /// {"cmd": "screenshot"}, {"cmd": "sidebar", "show": bool}, {"cmd": "spec", "on": bool}.
    func startRemote() {
        guard remoteTimer == nil else { return }
        try? FileManager.default.createDirectory(at: ChatModel.remoteDir, withIntermediateDirectories: true)
        remote = true
        remoteTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in self?.pollRemote() }
    }

    private func pollRemote() {
        let url = ChatModel.remoteDir.appendingPathComponent("command.json")
        if let data = try? Data(contentsOf: url) {
            try? FileManager.default.removeItem(at: url)
            if let command = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
                run(command)
            }
        }
        writeRemoteState()
    }

    private func run(_ command: [String: Any]) {
        switch command["cmd"] as? String {
        case "load":
            // only a model file of Documents, never a path
            if let file = command["file"] as? String, files.contains(file) { selected = file }
            load()
        case "new":
            newChat()
        case "send":
            if let value = command["think"] as? Bool { think = value }
            if let value = command["ste"] as? Bool { steStyle = value }
            send(command["text"] as? String ?? "")
        case "stop":
            stop()
        case "unload":
            unload()
        case "screenshot":
            screenshot()
        case "sidebar":
            showSidebar = command["show"] as? Bool ?? !showSidebar
        case "spec":
            setSpeculative(command["on"] as? Bool ?? !speculative)
        default:
            break
        }
    }

    /// Saves the key window as remote/screen.png.
    private func screenshot() {
        let windows = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.flatMap(\.windows)
        guard let window = windows.first(where: \.isKeyWindow) ?? windows.first else { return }
        let image = UIGraphicsImageRenderer(bounds: window.bounds).image { _ in
            window.drawHierarchy(in: window.bounds, afterScreenUpdates: false)
        }
        try? image.pngData()?.write(to: ChatModel.remoteDir.appendingPathComponent("screen.png"), options: .atomic)
    }

    private func writeRemoteState() {
        let messages = self.messages.map { m -> [String: Any] in
            ["user": m.user, "thinks": m.thinks, "thinking": m.thinking ?? "", "answer": m.answer]
        }
        let state: [String: Any] = [
            "status": status, "loaded": loaded ?? "", "loading": loading, "generating": generating, "think": think,
            "ste": steActive, "speculative": speculative, "verifies": stats.verifies, "accepted": stats.accepted,
            "tok_s": stats.generateRate, "prompt_tok_s": stats.prefillRate, "tokens": stats.tokens,
            "context_used": stats.used, "thermal": Util.thermal, "conversation": currentID?.uuidString ?? "",
            "messages": messages,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: state, options: [.sortedKeys]) {
            try? data.write(to: ChatModel.remoteDir.appendingPathComponent("state.json"), options: .atomic)
        }
    }
}

/// Type sizes of the chat, a little smaller than the system body text.
private enum Typo {
    static let body = Font.system(size: 15)
    static let heading = Font.system(size: 16, weight: .semibold)
    static let small = Font.system(size: 13)
    static let code = Font.system(size: 13, design: .monospaced)
}

struct ChatView: View {
    @ObservedObject private var model = ChatModel.shared
    @ObservedObject private var bench = BenchModel.shared
    @Environment(\.scenePhase) private var phase
    @State private var input = ""
    @State private var search = ""
    @FocusState private var focused: Bool

    private static let suggestions: [(icon: String, text: String)] = [
        ("bicycle", "Explain how a bicycle stays upright."),
        ("wrench.and.screwdriver", "Give the steps to replace a flat bicycle tire."),
        ("function", "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. What does the ball cost?"),
        ("doc.text", "Give three tips to write clear technical documentation."),
    ]

    var body: some View {
        NavigationSplitView(columnVisibility: Binding(get: { model.showSidebar ? .all : .detailOnly },
                                                      set: { model.showSidebar = $0 != .detailOnly })) {
            sidebar
        } detail: {
            VStack(spacing: 0) {
                statusStrip
                if model.messages.isEmpty {
                    emptyState
                } else {
                    transcript
                }
            }
            .safeAreaInset(edge: .bottom) { inputBar }
            .navigationTitle(model.conversations.first { $0.id == model.currentID }?.title ?? "LFM3 Chat")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button { model.newChat() } label: { Image(systemName: "square.and.pencil") }
                        .disabled(model.busy)
                }
                ToolbarItem(placement: .topBarLeading) {
                    // Speculative decoding: the model menu then offers the copies with a draft head.
                    Chip(title: "Spec", icon: model.speculative ? "hare.fill" : "hare", on: model.speculative) {
                        model.setSpeculative(!model.speculative)
                    }
                    .disabled(model.busy || (!model.speculative && model.draftFile == nil))
                }
                ToolbarItem(placement: .topBarTrailing) { modelMenu }
            }
        }
        .onAppear {
            model.refresh()
            #if DEBUG
            // development only: remote control writes the conversation and screenshots to Documents/remote
            if UserDefaults.standard.bool(forKey: "remote") { model.startRemote() }
            #endif
        }
        // iOS stops GPU work in the background.
        .onChange(of: phase) { _, newPhase in
            if newPhase == .background { model.stop() }
        }
    }

    // MARK: Sidebar

    /// Conversations that match the search, grouped by the day of their last message.
    private var sections: [(title: String, items: [Conversation])] {
        let calendar = Calendar.current
        let found = model.conversations.filter {
            search.isEmpty || $0.title.localizedCaseInsensitiveContains(search)
                || $0.messages.contains { $0.text.localizedCaseInsensitiveContains(search) }
        }
        let groups: [(String, (Date) -> Bool)] = [
            ("Today", { calendar.isDateInToday($0) }),
            ("Yesterday", { calendar.isDateInYesterday($0) }),
            ("Previous 7 days", { $0 > calendar.date(byAdding: .day, value: -7, to: Date())! }),
            ("Older", { _ in true }),
        ]
        var left = found
        var result: [(String, [Conversation])] = []
        for (title, test) in groups {
            let items = left.filter { test($0.updated) }
            left.removeAll { test($0.updated) }
            if !items.isEmpty { result.append((title, items)) }
        }
        return result
    }

    private var sidebar: some View {
        List(selection: Binding(get: { model.currentID }, set: { model.select($0) })) {
            Section {
                Button {
                    model.newChat()
                } label: {
                    Label("New chat", systemImage: "plus.bubble.fill")
                        .font(.system(size: 15, weight: .semibold))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.vertical, 10)
                        .padding(.horizontal, 12)
                        .foregroundStyle(.white)
                        .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 12))
                }
                .buttonStyle(.plain)
                .disabled(model.busy)
                .listRowInsets(EdgeInsets(top: 4, leading: 12, bottom: 8, trailing: 12))
                .listRowBackground(Color.clear)
            }
            ForEach(sections, id: \.title) { section in
                Section(section.title) {
                    ForEach(section.items) { conversation in
                        ConversationRow(conversation: conversation)
                            .tag(conversation.id)
                            .swipeActions {
                                Button("Delete", systemImage: "trash", role: .destructive) {
                                    model.delete(conversation.id)
                                }
                            }
                            .contextMenu {
                                Button("Delete", systemImage: "trash", role: .destructive) {
                                    model.delete(conversation.id)
                                }
                            }
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .searchable(text: $search, placement: .sidebar, prompt: "Search chats")
        .navigationTitle("Chats")
        .safeAreaInset(edge: .bottom) { deviceCard }
    }

    /// The model and the device state at the bottom of the sidebar.
    private var deviceCard: some View {
        HStack(spacing: 10) {
            Image(systemName: "cpu.fill")
                .font(.system(size: 16))
                .foregroundStyle(model.loaded == nil ? Color.secondary : Color.green)
                .frame(width: 34, height: 34)
                .background((model.loaded == nil ? Color.secondary : Color.green).opacity(0.15),
                            in: RoundedRectangle(cornerRadius: 9))
            VStack(alignment: .leading, spacing: 1) {
                Text(model.loaded.map(ModelName.title) ?? "No model loaded").font(.system(size: 14, weight: .semibold))
                Text("A12Z GPU · Metal Apple5").font(Typo.small).foregroundStyle(.secondary)
            }
            Spacer()
            if model.remote {
                Image(systemName: "dot.radiowaves.left.and.right")
                    .foregroundStyle(.orange)
                    .help("Remote control is on")
            }
        }
        .padding(12)
        .glass(RoundedRectangle(cornerRadius: 16))
        .padding(12)
    }

    // MARK: Header

    private var modelMenu: some View {
        Menu {
            Picker("Model file", selection: $model.selected) {
                ForEach(model.menuFiles, id: \.self) { file in
                    Text(ModelName.title(file) + " · " + ModelName.variant(file)).tag(file)
                }
            }
            .disabled(model.busy || model.loaded != nil)
            Divider()
            if model.loaded == nil {
                Button("Load", systemImage: "arrow.down.circle") { model.load() }
                    .disabled(model.busy || bench.running || model.selected.isEmpty)
            } else {
                Button("Unload", systemImage: "eject", role: .destructive) { model.unload() }
                    .disabled(model.busy)
            }
        } label: {
            HStack(spacing: 6) {
                Circle()
                    .fill(model.loading ? Color.yellow : model.loaded == nil ? Color.gray : Color.green)
                    .frame(width: 8, height: 8)
                Text(ModelName.title(model.loaded ?? model.selected)).font(.system(size: 14, weight: .semibold))
                Text(ModelName.variant(model.loaded ?? model.selected)).font(.caption).foregroundStyle(.secondary)
                Image(systemName: "chevron.down").font(.caption2.weight(.bold)).foregroundStyle(.secondary)
            }
        }
    }

    /// Speed, context and temperature of the loaded model.
    private var statusStrip: some View {
        HStack(spacing: 8) {
            if bench.running {
                Pill(icon: "speedometer", text: "Bench is running", tint: .orange)
            } else if model.loaded == nil || model.loading {
                Pill(icon: model.loading ? "hourglass" : "moon.zzz", text: model.status, tint: .secondary)
            } else if model.stats.used == 0 && !model.generating {
                Pill(icon: "checkmark.circle.fill", text: "Ready · " + model.status, tint: .green)
            } else {
                Pill(icon: "bolt.fill", text: String(format: "%.1f tok/s", model.stats.generateRate), tint: .green)
                if model.stats.verifies > 0 {
                    Pill(icon: "hare.fill", text: String(format: "draft %.0f%%",
                                                         100 * Double(model.stats.accepted) / Double(model.stats.verifies)),
                         tint: .purple)
                }
                Pill(icon: "text.cursor", text: String(format: "prompt %.0f tok/s", model.stats.prefillRate),
                     tint: .blue)
                contextPill
            }
            Spacer(minLength: 0)
            thermalPill
        }
        .font(.caption.monospacedDigit())
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .animation(.default, value: model.loaded)
    }

    private var contextPill: some View {
        let fraction = Double(model.stats.used) / Double(ChatEngine.context)
        return HStack(spacing: 6) {
            Image(systemName: "square.stack.3d.up")
            Text(String(format: "%.1fk / %dk", Double(model.stats.used) / 1000, ChatEngine.context / 1024))
            ProgressView(value: fraction).frame(width: 46).tint(fraction > 0.85 ? .orange : .accentColor)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(Color.secondary.opacity(0.12), in: Capsule())
    }

    private var thermalPill: some View {
        let state = min(model.thermal, 3)
        let tint: Color = [.green, .yellow, .orange, .red][state]
        return Pill(icon: "thermometer.medium", text: Util.thermalNames[state], tint: tint)
    }

    // MARK: Transcript

    private var emptyState: some View {
        ScrollView {
            VStack(spacing: 16) {
                Spacer(minLength: 70)
                Image(systemName: "sparkles")
                    .font(.system(size: 38))
                    .foregroundStyle(.tint)
                    .frame(width: 84, height: 84)
                    .background(Color.accentColor.opacity(0.12), in: Circle())
                Text("LFM3 1.7B on this iPad").font(.system(size: 22, weight: .semibold))
                Text(model.loaded == nil
                     ? "Load the model to start. It runs on the GPU. Nothing leaves the device."
                     : "Ask a question. Turn on Think for hard questions.")
                    .font(Typo.body)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                if model.loaded == nil {
                    Button { model.load() } label: {
                        Label(model.loading ? "Loading..." : "Load model", systemImage: "arrow.down.circle.fill")
                            .padding(.horizontal, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(model.busy || bench.running || model.selected.isEmpty)
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 260), spacing: 12)], spacing: 12) {
                    ForEach(ChatView.suggestions, id: \.text) { suggestion in
                        Button {
                            input = suggestion.text
                            focused = true
                        } label: {
                            HStack(alignment: .top, spacing: 10) {
                                Image(systemName: suggestion.icon)
                                    .font(.system(size: 14))
                                    .foregroundStyle(.tint)
                                    .frame(width: 20)
                                Text(suggestion.text)
                                    .font(Typo.body)
                                    .multilineTextAlignment(.leading)
                            }
                            .frame(maxWidth: .infinity, minHeight: 60, alignment: .topLeading)
                            .padding(14)
                            .background(Color.secondary.opacity(0.1), in: RoundedRectangle(cornerRadius: 14))
                        }
                        .buttonStyle(.plain)
                    }
                }
                .frame(maxWidth: 620)
                .padding(.top, 10)
            }
            .padding(24)
            .frame(maxWidth: .infinity)
        }
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 24) {
                    ForEach(model.messages) { message in
                        MessageView(message: message, live: model.generating && message.id == model.messages.last?.id)
                            .id(message.id)
                    }
                }
                .frame(maxWidth: 720)
                .frame(maxWidth: .infinity)
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
            }
            .scrollDismissesKeyboard(.interactively)
            .defaultScrollAnchor(.bottom, for: .initialOffset)
            .onChange(of: model.messages.last?.text) { _, _ in
                if let id = model.messages.last?.id { proxy.scrollTo(id, anchor: .bottom) }
            }
        }
    }

    // MARK: Input

    private var canSend: Bool {
        !model.busy && model.loaded != nil && !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var inputBar: some View {
        VStack(alignment: .leading, spacing: 10) {
            TextField(model.loaded == nil ? "Load a model first" : "Message LFM3", text: $input, axis: .vertical)
                .font(Typo.body)
                .lineLimit(1...8)
                .focused($focused)
                .disabled(model.loaded == nil)
                .onSubmit(submit)
            HStack(spacing: 8) {
                Chip(title: "Think", icon: model.think ? "brain.fill" : "brain", on: model.think) {
                    model.think.toggle()
                }
                .disabled(model.busy)
                // The system prompt is part of the KV cache, so it changes only for a new conversation.
                Chip(title: "STE", icon: "text.badge.checkmark", on: model.steActive) {
                    model.steStyle.toggle()
                }
                .disabled(model.busy || !model.messages.isEmpty)
                Spacer()
                if model.generating {
                    CircleButton(icon: "stop.fill", tint: .red, action: model.stop)
                } else {
                    CircleButton(icon: "arrow.up", tint: canSend ? .accentColor : Color.secondary.opacity(0.35),
                                 action: submit)
                        .disabled(!canSend)
                }
            }
        }
        .padding(14)
        .glass(RoundedRectangle(cornerRadius: 24))
        .frame(maxWidth: 720)
        .padding(.horizontal, 20)
        .padding(.bottom, 12)
    }

    private func submit() {
        guard canSend else { return }
        model.send(input)
        input = ""
    }
}

private struct ConversationRow: View {
    let conversation: Conversation

    var body: some View {
        let reply = conversation.messages.last { !$0.user }
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: reply?.thinks == true ? "brain" : "bubble.left")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.tint)
                .frame(width: 28, height: 28)
                .background(Color.accentColor.opacity(0.14), in: RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 2) {
                HStack(alignment: .firstTextBaseline) {
                    Text(conversation.title)
                        .font(.system(size: 14, weight: .medium))
                        .lineLimit(1)
                    Spacer(minLength: 4)
                    Text(conversation.updated, format: .relative(presentation: .numeric, unitsStyle: .narrow))
                        .font(.system(size: 11))
                        .foregroundStyle(.tertiary)
                }
                Text(reply.map { $0.answer.isEmpty ? "..." : Display.snippet($0.answer) } ?? "No messages")
                    .font(Typo.small)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                if conversation.system != nil {
                    Label("STE", systemImage: "text.badge.checkmark")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.tint)
                }
            }
        }
        .padding(.vertical, 3)
    }
}

private struct Chip: View {
    let title: String
    let icon: String
    let on: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            // not a Label: a toolbar would show only its icon
            HStack(spacing: 5) {
                Image(systemName: icon)
                Text(title)
            }
            .font(.system(size: 13, weight: .semibold))
            .padding(.horizontal, 11)
            .padding(.vertical, 6)
            .foregroundStyle(on ? Color.white : Color.secondary)
            .background(on ? Color.accentColor : Color.secondary.opacity(0.14), in: Capsule())
        }
        .buttonStyle(.plain)
        .animation(.snappy, value: on)
    }
}

private struct CircleButton: View {
    let icon: String
    let tint: Color
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 15, weight: .bold))
                .frame(width: 34, height: 34)
                .foregroundStyle(.white)
                .background(tint, in: Circle())
        }
        .buttonStyle(.plain)
    }
}

private struct Pill: View {
    let icon: String
    let text: String
    let tint: Color

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: icon).foregroundStyle(tint)
            Text(text).lineLimit(1)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(Color.secondary.opacity(0.12), in: Capsule())
    }
}

private struct MessageView: View {
    let message: ChatMessage
    let live: Bool
    @State private var showThinking = false
    @State private var copied = false

    var body: some View {
        if message.user {
            HStack {
                Spacer(minLength: 80)
                Text(message.text)
                    .font(Typo.body)
                    .textSelection(.enabled)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .foregroundStyle(.white)
                    .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 18))
            }
        } else {
            VStack(alignment: .leading, spacing: 12) {
                if let thinking = message.thinking, !thinking.isEmpty || live {
                    thinkingCard(thinking)
                }
                if !message.answer.isEmpty {
                    AnswerView(text: message.answer)
                } else if live && message.thinkingDone {
                    ProgressView().padding(.top, 4)
                }
                if !live && !message.answer.isEmpty {
                    footer
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var footer: some View {
        HStack(spacing: 12) {
            Button {
                UIPasteboard.general.string = Display.clean(message.answer)
                copied = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { copied = false }
            } label: {
                Image(systemName: copied ? "checkmark" : "doc.on.doc")
            }
            .buttonStyle(.plain)
            if let tokens = message.tokens, let rate = message.rate {
                Label(String(format: "%d tokens · %.1f tok/s", tokens, rate), systemImage: "bolt")
                    .monospacedDigit()
            }
        }
        .font(.system(size: 12))
        .foregroundStyle(.tertiary)
    }

    private func thinkingCard(_ thinking: String) -> some View {
        let running = live && !message.thinkingDone
        return VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(.snappy) { showThinking.toggle() }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "brain")
                        .symbolEffect(.pulse, isActive: running)
                    Text(running ? "Thinking..." : "Thought process")
                        .font(.system(size: 13, weight: .medium))
                    Text("\(thinking.split(whereSeparator: \.isWhitespace).count) words")
                        .font(.system(size: 12).monospacedDigit())
                        .foregroundStyle(.tertiary)
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.system(size: 11, weight: .semibold))
                        .rotationEffect(.degrees(showThinking ? 90 : 0))
                }
                .foregroundStyle(.secondary)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if showThinking {
                Text(Display.clean(thinking))
                    .font(Typo.small)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else if running {
                // The last lines while the model thinks.
                Text(Display.clean(String(thinking.suffix(240))))
                    .font(Typo.small)
                    .foregroundStyle(.tertiary)
                    .lineLimit(3)
                    .truncationMode(.head)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Color.secondary.opacity(0.14)))
    }
}

/// An answer as blocks: headings, lists with hanging indents, code, rules and paragraphs.
private struct AnswerView: View {
    let text: String

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(Display.blocks(text).enumerated()), id: \.offset) { _, block in
                view(for: block)
            }
        }
        .textSelection(.enabled)
    }

    @ViewBuilder private func view(for block: Display.Block) -> some View {
        switch block {
        case .heading(let text):
            Text(Display.inline(text)).font(Typo.heading).padding(.top, 4)
        case .paragraph(let text):
            Text(Display.inline(text)).font(Typo.body).lineSpacing(3)
        case .item(let marker, let text, let level):
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                if let marker {
                    Text(marker)
                        .font(.system(size: 13, weight: .semibold).monospacedDigit())
                        .foregroundStyle(.tint)
                        .frame(minWidth: 18, alignment: .trailing)
                } else {
                    Circle().fill(Color.accentColor).frame(width: 5, height: 5).frame(width: 18)
                        .alignmentGuide(.firstTextBaseline) { $0[.bottom] + 4 }
                }
                Text(Display.inline(text)).font(Typo.body).lineSpacing(3)
            }
            .padding(.leading, CGFloat(level) * 18)
        case .code(let text):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(text).font(Typo.code).padding(12)
            }
            .background(Color.secondary.opacity(0.1), in: RoundedRectangle(cornerRadius: 10))
        case .rule:
            Divider().padding(.vertical, 4)
        }
    }
}

/// Short names for the model files, for example "LFM3-1.7B-QAD-Q4_0-embQ8.gguf" is "LFM3-1.7B" and "QAD-Q4_0-embQ8".
private enum ModelName {
    static func title(_ file: String) -> String {
        let title = parts(file).prefix(2).joined(separator: "-")
        return title.isEmpty ? "No model" : title
    }

    static func variant(_ file: String) -> String {
        parts(file).dropFirst(2).joined(separator: "-")
    }

    private static func parts(_ file: String) -> [String] {
        let name = file.hasSuffix(".gguf") ? String(file.dropLast(5)) : file
        return name.split(separator: "-").map(String.init)
    }
}

/// Display-only text cleanup and layout. The model writes LaTeX and block markdown that SwiftUI Text does not
/// render. The stored text, and so the KV cache of the conversation, keeps the original form.
private enum Display {
    enum Block {
        case heading(String)
        case paragraph(String)
        /// A list item: the number ("1.") or nil for a bullet, the text, the indent level.
        case item(String?, String, Int)
        case code(String)
        case rule
    }

    private static let rules: [(Regex<AnyRegexOutput>, (Regex<AnyRegexOutput>.Match) -> String)] = [
        (try! Regex(#"\\boxed\{([^{}]*)\}"#), { "**\($0[1].substring ?? "")**" }),
        (try! Regex(#"\\(?:text|mathrm|mathbf)\{([^{}]*)\}"#), { String($0[1].substring ?? "") }),
        (try! Regex(#"\\frac\{([^{}]*)\}\{([^{}]*)\}"#), { "\($0[1].substring ?? "")/\($0[2].substring ?? "")" }),
        (try! Regex(#"\\\[|\\\]|\\\(|\\\)|\$\$"#), { _ in "" }),
        (try! Regex(#"\\(times|cdot|div|leq|geq|le|ge|neq|approx|rightarrow|to|pm)\b"#), { match in
            let symbols = ["times": "×", "cdot": "·", "div": "÷", "leq": "≤", "geq": "≥", "le": "≤", "ge": "≥",
                           "neq": "≠", "approx": "≈", "rightarrow": "→", "to": "→", "pm": "±"]
            return symbols[String(match[1].substring ?? "")] ?? ""
        }),
        (try! Regex(#"\\[,;! ]"#), { _ in " " }),
        (try! Regex(#"\{:\}"#), { _ in ":" }),
    ]

    static func clean(_ text: String) -> String {
        var text = text
        for (regex, replace) in rules { text = text.replacing(regex, with: replace) }
        return text
    }

    /// One line of plain text without markdown markers or think tags, for previews.
    static func snippet(_ text: String) -> String {
        clean(text).components(separatedBy: "\n")
            .map { $0.replacing(/^\s*(#{1,6}|[-*•]|\d{1,3}[.)])\s+/, with: "") }
            .joined(separator: " ")
            .replacing(/<\/?think>|\*\*|__|`/, with: "")
            .replacing(/\s+/, with: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    static func inline(_ text: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        return (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)
    }

    static func blocks(_ text: String) -> [Block] {
        var blocks: [Block] = []
        var paragraph: [String] = []
        var code: [String]?
        func flush() {
            if !paragraph.isEmpty { blocks.append(.paragraph(paragraph.joined(separator: "\n"))) }
            paragraph = []
        }
        for raw in clean(text).components(separatedBy: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("```") {
                if let lines = code {
                    blocks.append(.code(lines.joined(separator: "\n")))
                    code = nil
                } else {
                    flush()
                    code = []
                }
                continue
            }
            if code != nil {
                code?.append(raw)
                continue
            }
            if line.isEmpty {
                flush()
                continue
            }
            let indent = raw.prefix { $0 == " " }.count >= 2 ? 1 : 0
            if let match = line.firstMatch(of: /^#{1,6}\s+(.*)$/) {
                flush()
                blocks.append(.heading(String(match.1)))
            } else if line.firstMatch(of: /^(-{3,}|\*{3,}|_{3,})$/) != nil {
                flush()
                blocks.append(.rule)
            } else if let match = line.firstMatch(of: /^[-*•]\s+(.*)$/) {
                flush()
                blocks.append(.item(nil, String(match.1), indent))
            } else if let match = line.firstMatch(of: /^(\d{1,3})[.)]\s+(.*)$/) {
                flush()
                blocks.append(.item("\(match.1).", String(match.2), indent))
            } else if paragraph.isEmpty, indent > 0, case .item(let marker, let body, let level)? = blocks.last {
                // an indented line after a list item continues the item
                blocks[blocks.count - 1] = .item(marker, body + "\n" + line, level)
            } else {
                paragraph.append(line)
            }
        }
        if let lines = code { blocks.append(.code(lines.joined(separator: "\n"))) }
        flush()
        return blocks
    }
}

private extension View {
    /// Liquid Glass on iOS 26 and later, a material before.
    @ViewBuilder func glass(_ shape: some Shape) -> some View {
        if #available(iOS 26.0, *) {
            glassEffect(.regular, in: shape)
        } else {
            background(.ultraThinMaterial, in: shape)
        }
    }
}
