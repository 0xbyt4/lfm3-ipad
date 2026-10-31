import SwiftUI

@main
struct LFM3BenchApp: App {
    /// A bench launch argument opens the Bench tab, whose onAppear starts the run.
    @State private var tab = ["bench", "kerneltest", "trace", "depth", "attention", "verify"]
        .contains { UserDefaults.standard.object(forKey: $0) != nil } ? 1 : 0

    var body: some Scene {
        WindowGroup {
            TabView(selection: $tab) {
                ChatView()
                    .tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right") }
                    .tag(0)
                BenchView()
                    .tabItem { Label("Bench", systemImage: "speedometer") }
                    .tag(1)
            }
        }
    }
}
