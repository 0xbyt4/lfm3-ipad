import Foundation
import os

enum Util {
    static var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }

    static func footprint() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int(info.phys_footprint) : -1
    }

    /// Free physical memory of the whole device. Mapped model pages do not count toward the app's footprint but
    /// take this memory.
    static func systemFree() -> Int {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &stats) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int(stats.free_count) * Int(vm_kernel_page_size) : -1
    }

    static func mib(_ bytes: Int) -> Int { bytes / 1_048_576 }
    static func now() -> Double { CFAbsoluteTimeGetCurrent() }
    static var thermal: Int { ProcessInfo.processInfo.thermalState.rawValue }
    static let thermalNames = ["nominal", "fair", "serious", "critical"]

    /// Memory and heat fields that every result line carries.
    static func state() -> [String: Any] {
        [
            "thermal": thermal,
            "footprint_mib": mib(footprint()),
            "available_mib": mib(Int(os_proc_available_memory())),
            "system_free_mib": mib(systemFree()),
        ]
    }
}

/// A stop request, set on the main thread and read by the bench loop.
final class StopFlag {
    private let lock = NSLock()
    private var value = false
    var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
    func set(_ v: Bool) { lock.lock(); value = v; lock.unlock() }
}

/// Measures how late a 100 ms timer on the main queue fires. A late timer means the UI could not respond: this is
/// the "freeze" the user sees while the model runs.
final class MainLagProbe {
    private let lock = NSLock()
    private var maxLag = 0.0
    private var timer: DispatchSourceTimer?
    private var expected = 0.0
    private let interval = 0.1

    func start() {
        let t = DispatchSource.makeTimerSource(queue: .main)
        expected = Util.now() + interval
        t.schedule(deadline: .now() + interval, repeating: interval)
        t.setEventHandler { [weak self] in self?.tick() }
        t.resume()
        timer = t
    }

    func stop() {
        timer?.cancel()
        timer = nil
    }

    /// The largest lag in milliseconds since the last call.
    func takeMax() -> Double {
        lock.lock()
        defer { maxLag = 0; lock.unlock() }
        return maxLag * 1000
    }

    private func tick() {
        let now = Util.now()
        lock.lock()
        maxLag = max(maxLag, now - expected)
        lock.unlock()
        expected = now + interval
    }
}

/// Append-only text file in Documents, flushed after every write so the lines before a crash survive.
final class OutputFile {
    private let handle: FileHandle?

    init(name: String) {
        let url = Util.documents.appendingPathComponent(name)
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        handle = try? FileHandle(forWritingTo: url)
        _ = try? handle?.seekToEnd()
    }

    deinit { try? handle?.close() }

    func write(_ line: String) {
        guard let data = (line + "\n").data(using: .utf8) else { return }
        handle?.write(data)
        try? handle?.synchronize()
    }
}
