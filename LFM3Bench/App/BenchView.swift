import SwiftUI
import UIKit

final class BenchModel: ObservableObject {
    static let shared = BenchModel()

    @Published var files: [String] = []
    @Published var lines: [String] = []
    @Published var running = false
    @Published var seconds: Double = 180
    @Published var waitCool = true

    private let queue = DispatchQueue(label: "bench", qos: .userInitiated)
    private let stopFlag = StopFlag()
    private let lag = MainLagProbe()
    private var autoStarted = false

    func refresh() {
        files = Util.modelFiles()
    }

    /// Runs the given files one after another on the bench queue.
    func run(_ names: [String]) {
        guard !running, ChatModel.shared.loaded == nil, !ChatModel.shared.busy else { return }
        guard !names.isEmpty else {
            append("no model file to run in Documents")
            return
        }
        var options = BenchOptions(seconds: seconds, waitCool: waitCool)
        options.speculative = UserDefaults.standard.bool(forKey: "spec")
        if let file = UserDefaults.standard.string(forKey: "prompts") {
            guard let text = try? String(contentsOf: Util.documents.appendingPathComponent(file), encoding: .utf8)
            else {
                append("prompts file not found: \(file)")
                return
            }
            options.prompts = text.components(separatedBy: "\n%%\n")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
            guard !options.prompts.isEmpty else {
                append("no prompts in \(file)")
                return
            }
            options.promptsFile = file
        }
        running = true
        stopFlag.set(false)
        UIApplication.shared.isIdleTimerDisabled = true
        lag.start()
        let bench = Bench(stop: stopFlag, lag: lag) { [weak self] line in
            DispatchQueue.main.async { self?.append(line) }
        }
        queue.async { [stopFlag] in
            for name in names where !stopFlag.isSet {
                bench.run(path: Util.documents.appendingPathComponent(name).path, options: options)
            }
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.lag.stop()
                UIApplication.shared.isIdleTimerDisabled = false
                self.running = false
                self.append("done")
            }
        }
    }

    func stop() {
        stopFlag.set(true)
    }

    /// Writes the tensor sums of one forward pass; see Bench.trace.
    func trace(_ name: String) {
        guard !running, ChatModel.shared.loaded == nil, !ChatModel.shared.busy else { return }
        running = true
        stopFlag.set(false)
        let bench = Bench(stop: stopFlag, lag: lag) { [weak self] line in
            DispatchQueue.main.async { self?.append(line) }
        }
        queue.async {
            bench.trace(path: Util.documents.appendingPathComponent(name).path, options: BenchOptions())
            DispatchQueue.main.async { [weak self] in
                self?.running = false
                self?.append("done")
            }
        }
    }

    /// Generation speed at growing context depths; see Bench.depth.
    func depth(_ name: String) {
        guard !running, ChatModel.shared.loaded == nil, !ChatModel.shared.busy else { return }
        running = true
        stopFlag.set(false)
        let bench = Bench(stop: stopFlag, lag: lag) { [weak self] line in
            DispatchQueue.main.async { self?.append(line) }
        }
        var options = BenchOptions()
        options.waitCool = waitCool
        queue.async {
            bench.depth(path: Util.documents.appendingPathComponent(name).path, options: options)
            DispatchQueue.main.async { [weak self] in
                self?.running = false
                self?.append("depth done")
            }
        }
    }

    /// Times decode batches of 1...8 tokens; see Bench.verifyCost.
    func verifyCost(_ name: String) {
        guard !running, ChatModel.shared.loaded == nil, !ChatModel.shared.busy else { return }
        running = true
        stopFlag.set(false)
        let bench = Bench(stop: stopFlag, lag: lag) { [weak self] line in
            DispatchQueue.main.async { self?.append(line) }
        }
        var options = BenchOptions()
        options.waitCool = waitCool
        options.speculative = UserDefaults.standard.bool(forKey: "spec")
        if UserDefaults.standard.object(forKey: "at") != nil {
            options.verifyAt = UserDefaults.standard.integer(forKey: "at")
        }
        queue.async {
            bench.verifyCost(path: Util.documents.appendingPathComponent(name).path, options: options)
            DispatchQueue.main.async { [weak self] in
                self?.running = false
                self?.append("verify done")
            }
        }
    }

    /// Compares the Metal kernels with a CPU reference on this device; see KernelTest.
    func kernelTest() {
        guard !running, ChatModel.shared.loaded == nil, !ChatModel.shared.busy else { return }
        running = true
        let report: (String) -> Void = { [weak self] line in DispatchQueue.main.async { self?.append(line) } }
        queue.async {
            let out = OutputFile(name: "results.jsonl")
            let run = ISO8601DateFormatter().string(from: Date())
            for var r in KernelTest.run(report: report) {
                r["event"] = "kernel"
                r["run"] = run
                if let data = try? JSONSerialization.data(withJSONObject: r, options: [.sortedKeys]) {
                    out.write(String(decoding: data, as: UTF8.self))
                }
            }
            DispatchQueue.main.async { [weak self] in
                self?.running = false
                self?.append("kernel test done")
            }
        }
    }

    /// Times the attention of the full-attention layers on its own; see AttentionBench.
    func attentionBench() {
        guard !running, ChatModel.shared.loaded == nil, !ChatModel.shared.busy else { return }
        running = true
        let report: (String) -> Void = { [weak self] line in DispatchQueue.main.async { self?.append(line) } }
        queue.async {
            let out = OutputFile(name: "results.jsonl")
            let run = ISO8601DateFormatter().string(from: Date())
            for var r in AttentionBench.run(report: report) {
                r["event"] = "attention"
                r["run"] = run
                if let data = try? JSONSerialization.data(withJSONObject: r, options: [.sortedKeys]) {
                    out.write(String(decoding: data, as: UTF8.self))
                }
            }
            DispatchQueue.main.async { [weak self] in
                self?.running = false
                self?.append("attention bench done")
            }
        }
    }

    /// Starts a run from launch arguments, for example `-bench all -seconds 180 -cool 1`. `-bench` takes a file name
    /// in Documents or "all". `-kerneltest 1` runs the kernel test instead, `-trace <file>` the tensor trace,
    /// `-depth <file>` the speed at growing context depths, `-attention 1` the attention timing,
    /// `-verify <file>` the cost of decode batches of 1...8 tokens (`-at <pos>` start position, `-spec 1` with the
    /// draft head). With `-bench`, `-spec 1` turns on speculative
    /// decoding and `-prompts <file>` takes the generation prompts from a Documents file (separator: a `%%` line).
    func autoStart() {
        guard !autoStarted else { return }
        autoStarted = true
        let defaults = UserDefaults.standard
        if defaults.bool(forKey: "kerneltest") {
            kernelTest()
            return
        }
        if let file = defaults.string(forKey: "verify") {
            if defaults.object(forKey: "cool") != nil { waitCool = defaults.bool(forKey: "cool") }
            verifyCost(file)
            return
        }
        if defaults.bool(forKey: "attention") {
            attentionBench()
            return
        }
        if let file = defaults.string(forKey: "trace") {
            trace(file)
            return
        }
        if let file = defaults.string(forKey: "depth") {
            if defaults.object(forKey: "cool") != nil { waitCool = defaults.bool(forKey: "cool") }
            depth(file)
            return
        }
        guard let target = defaults.string(forKey: "bench") else { return }
        if defaults.object(forKey: "seconds") != nil { seconds = defaults.double(forKey: "seconds") }
        if defaults.object(forKey: "cool") != nil { waitCool = defaults.bool(forKey: "cool") }
        run(target == "all" ? files : files.filter { $0 == target })
    }

    private func append(_ line: String) {
        lines.append(line)
        if lines.count > 300 { lines.removeFirst(lines.count - 300) }
    }
}

struct BenchView: View {
    @ObservedObject private var model = BenchModel.shared
    @ObservedObject private var chat = ChatModel.shared
    @Environment(\.scenePhase) private var phase

    var body: some View {
        NavigationStack {
            List {
                Section("Settings") {
                    Stepper("Generation: \(Int(model.seconds)) s", value: $model.seconds, in: 30...900, step: 30)
                    Toggle("Wait for nominal thermal", isOn: $model.waitCool)
                }
                .disabled(model.running)
                if chat.loaded != nil || chat.busy {
                    Section {
                        Text("Unload the chat model first: two models would share the GPU and memory.")
                            .foregroundStyle(.secondary)
                    }
                }
                Section("Models in Documents") {
                    ForEach(model.files, id: \.self) { file in
                        HStack {
                            Text(file).font(.footnote.monospaced())
                            Spacer()
                            Button("Run") { model.run([file]) }
                                .disabled(model.running)
                        }
                    }
                    Button("Run all") { model.run(model.files) }
                        .disabled(model.running || model.files.isEmpty)
                    Button("Kernel test") { model.kernelTest() }
                        .disabled(model.running)
                }
                Section("Log") {
                    ForEach(Array(model.lines.enumerated()), id: \.offset) { _, line in
                        Text(line).font(.caption.monospaced())
                    }
                }
            }
            .navigationTitle("LFM3 Bench")
            .toolbar {
                if model.running {
                    Button("Stop", role: .destructive) { model.stop() }
                } else {
                    Button("Refresh") { model.refresh() }
                }
            }
        }
        .onAppear {
            model.refresh()
            model.autoStart()
        }
        // iOS stops GPU work in the background, so a run there would only record failures.
        .onChange(of: phase) { _, newPhase in
            if newPhase == .background { model.stop() }
        }
    }
}
