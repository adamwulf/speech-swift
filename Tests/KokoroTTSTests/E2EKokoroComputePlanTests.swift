import XCTest
import CoreML

/// Reports where Core ML runs each op of Kokoro end-to-end graphs under
/// `.cpuAndNeuralEngine`, from `MLComputePlan`: preferred device per op, the
/// share of estimated cost per device, and how many ops support the Neural
/// Engine at all. Not a pass/fail test: it skips unless `KOKORO_PLAN_DIR`
/// names a folder of `.mlmodelc` bundles; `KOKORO_PLAN_FILES` lists the
/// bundle names (comma-separated, default `kokoro_5s`). Planning an fp16
/// graph can take as long as its first Neural Engine compile.
///
/// Run with: KOKORO_PLAN_DIR=/path swift test --filter E2EKokoroComputePlanTests
final class E2EKokoroComputePlanTests: XCTestCase {

    func testComputePlans() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let dir = env["KOKORO_PLAN_DIR"], !dir.isEmpty else {
            throw XCTSkip("Set KOKORO_PLAN_DIR to a folder of Kokoro .mlmodelc bundles")
        }
        let bundle = URL(fileURLWithPath: dir, isDirectory: true)
        let files = (env["KOKORO_PLAN_FILES"] ?? "kokoro_5s").split(separator: ",").map(String.init)
        for name in files {
            let url = bundle.appendingPathComponent("\(name).mlmodelc", isDirectory: true)
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            let t0 = CFAbsoluteTimeGetCurrent()
            let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)
            let seconds = CFAbsoluteTimeGetCurrent() - t0
            guard case .program(let program) = plan.modelStructure,
                  let main = program.functions["main"] else {
                XCTFail("\(name): not an ML program")
                continue
            }
            var counts: [String: Int] = [:]
            var cost: [String: Double] = [:]
            var aneSupported = 0
            var total = 0
            for op in main.block.operations where op.operatorName != "const" {
                total += 1
                let usage = plan.deviceUsage(for: op)
                let device: String
                switch usage?.preferred {
                case .cpu: device = "cpu"
                case .gpu: device = "gpu"
                case .neuralEngine: device = "ane"
                default: device = "none"
                }
                counts[device, default: 0] += 1
                cost[device, default: 0] += plan.estimatedCost(of: op)?.weight ?? 0
                if usage?.supported.contains(where: { if case .neuralEngine = $0 { return true } else { return false } }) == true {
                    aneSupported += 1
                }
            }
            let costTotal = max(cost.values.reduce(0, +), 1e-12)
            let share = cost.map { "\($0.key) \(String(format: "%.1f", 100 * $0.value / costTotal))%" }.sorted()
            print("PLAN \(name): \(total) non-const ops, plan load \(String(format: "%.1f", seconds)) s | preferred ops \(counts.sorted { $0.key < $1.key }) | estimated cost share \(share) | ane-supported ops \(aneSupported)")
        }
    }
}
