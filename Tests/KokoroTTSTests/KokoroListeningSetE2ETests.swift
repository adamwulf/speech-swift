import XCTest
import CoreML
import Accelerate
import AudioCommon
@testable import KokoroTTS

/// Renders a Kokoro listening set as WAVs and prints, per configuration and
/// text, the render time and pitch stability (frame-to-frame F0 jitter and
/// wobble around a 50 ms average, in cents). Not a pass/fail test: it skips
/// unless `KOKORO_LISTENING_DIR` names the folder to write the WAVs to.
///
/// Three sets:
/// - Default: the warble set, `kokoro_5s` against FluidInference's
///   `kokoro_21_15s` / `kokoro_21_5s` under zero phases, CPU only, BOS/EOS 0
///   and the averaged voice (docs/research/2026-11-13-kokoro-long-window-and-exporters.md).
/// - With `KOKORO_AB_BUNDLE_DIR` set: the export A/B of
///   docs/plans/kokoro-e2e-5s-export-ab.md, loaded offline from that folder,
///   which holds `kokoro_5s.mlmodelc`, `kokoro_eliza_5s_fp32.mlmodelc`,
///   `kokoro_eliza_5s_fp16.mlmodelc` and the aufklarer vocabulary, G2P and voices.
/// - With `KOKORO_AB_BUNDLE_DIR` and `KOKORO_LISTENING_SET=long`: long
///   sentences rendered by the 5 s graphs in chunks and by the 10 s / 15 s
///   exports (`kokoro_eliza_{10,15}s_{fp32,fp16}.mlmodelc`) in one pass. The
///   fp16 graphs load with `.cpuOnly`, where they run fastest.
///
/// The A/B set also renders FluidInference's `kokoro_21_5s` and
/// `kokoro_21_15s` (downloaded) and the 10 s / 15 s fp16 exports for
/// comparison. `KOKORO_LISTENING_CONFIGS` (comma-separated config names)
/// limits a run to those configurations.
///
/// Run with: KOKORO_LISTENING_DIR=/path swift test --filter E2EKokoroListeningSetTests
final class E2EKokoroListeningSetTests: XCTestCase {

    static let warbleTexts = [
        ("short", "Sure, I can do that."),
        ("long", "I checked the build server this morning, and both jobs finished without errors, so the new release is ready whenever you want to ship it."),
    ]

    static let abTexts = warbleTexts + [
        ("yes", "Yes."),
        ("hello", "Hello there, how are you today?"),
        ("question", "Do you want me to move the meeting to Thursday afternoon instead?"),
        ("numbers", "Your flight leaves at 7:45 tomorrow morning from gate 23, and boarding starts thirty minutes earlier."),
        ("commas", "Bring the charger, the blue folder, two pens, and, if you have room, the spare headphones."),
        ("paragraph", "The meeting starts at nine tomorrow morning. Please bring the budget report and the travel plans for next month. We will finish with a short review of the new product launch."),
    ]

    /// Single sentences of about 120-200 phoneme tokens: several chunks on a
    /// 5 s graph, one pass on a 15 s graph.
    static let longTexts = [
        ("build", "I checked the build server this morning, and both jobs finished without errors, so the new release is ready whenever you want to ship it."),
        ("basement", "The team spent most of the afternoon looking through the old reports in the basement because nobody could remember where the original drawings for the north wing of the building had been stored after the move."),
        ("airport", "If you leave now and the traffic on the bridge stays light, you should reach the airport about forty minutes before boarding, which leaves enough time to check your bag and grab a coffee."),
        ("next-step", "So the next useful step is for you to run a few real conversations and tell me where the detection or the voice still feels off, and I'll send that straight to an agent."),
        ("saturday", "My plan for Saturday is to sleep in, take the dog for a long walk by the river, and then spend the afternoon finally fixing the squeaky hinge on the back door."),
        ("dentist", "Would you like me to move your dentist appointment to next Tuesday afternoon, or would you rather keep the original time and reschedule the team meeting instead?"),
    ]

    /// Parenthetical asides, each also with a comma typed before the `(`.
    static let parenTexts = [
        ("meeting", "The meeting (which was supposed to be short) ran for almost two hours."),
        ("meeting-comma", "The meeting, (which was supposed to be short) ran for almost two hours."),
        ("laptop", "Bring your laptop (and the charger) to the review tomorrow."),
        ("laptop-comma", "Bring your laptop, (and the charger) to the review tomorrow."),
        ("roommate", "I talked to Sam (my old roommate from college) about the trip."),
        ("roommate-comma", "I talked to Sam, (my old roommate from college) about the trip."),
        ("tip", "The total came to forty dollars (plus tip)."),
        ("tip-comma", "The total came to forty dollars, (plus tip)."),
    ]

    static func longConfigs(_ dir: URL) -> [Config] {
        let local = "local/kokoro-ab"
        return [
            Config(name: "A_aufklarer-5s", modelId: local, modelFile: "kokoro_5s", units: .cpuAndNeuralEngine, cacheDir: dir),
            Config(name: "X3c_eliza-5s-fp16_cpu", modelId: local, modelFile: "kokoro_eliza_5s_fp16", units: .cpuOnly, cacheDir: dir),
            Config(name: "L10_eliza-10s-fp32", modelId: local, modelFile: "kokoro_eliza_10s_fp32", units: .cpuAndNeuralEngine, cacheDir: dir),
            Config(name: "L10h_eliza-10s-fp16_cpu", modelId: local, modelFile: "kokoro_eliza_10s_fp16", units: .cpuOnly, cacheDir: dir),
            Config(name: "L15_eliza-15s-fp32", modelId: local, modelFile: "kokoro_eliza_15s_fp32", units: .cpuAndNeuralEngine, cacheDir: dir),
            Config(name: "L15h_eliza-15s-fp16_cpu", modelId: local, modelFile: "kokoro_eliza_15s_fp16", units: .cpuOnly, cacheDir: dir),
        ]
    }

    struct Config {
        let name: String
        let modelId: String
        let modelFile: String
        let units: MLComputeUnits
        var zeroPhases = false
        var boundary: Int? = nil
        var averaged = false
        /// A local bundle folder, loaded offline; nil downloads `modelId`.
        var cacheDir: URL? = nil
    }

    static let aufklarer = KokoroTTSModel.defaultModelId
    static let fluid = "FluidInference/kokoro-82m-coreml"

    static func abConfigs(_ dir: URL) -> [Config] {
        let local = "local/kokoro-ab"
        return [
            Config(name: "A_aufklarer-5s", modelId: local, modelFile: "kokoro_5s", units: .cpuAndNeuralEngine, cacheDir: dir),
            Config(name: "A0_aufklarer-5s_pad0", modelId: local, modelFile: "kokoro_5s", units: .cpuAndNeuralEngine, boundary: 0, cacheDir: dir),
            Config(name: "X1_eliza-5s-fp32", modelId: local, modelFile: "kokoro_eliza_5s_fp32", units: .cpuAndNeuralEngine, cacheDir: dir),
            Config(name: "X2_eliza-5s-fp32_pad0", modelId: local, modelFile: "kokoro_eliza_5s_fp32", units: .cpuAndNeuralEngine, boundary: 0, cacheDir: dir),
            Config(name: "X1c_eliza-5s-fp32_cpu", modelId: local, modelFile: "kokoro_eliza_5s_fp32", units: .cpuOnly, cacheDir: dir),
            Config(name: "X3_eliza-5s-fp16", modelId: local, modelFile: "kokoro_eliza_5s_fp16", units: .cpuAndNeuralEngine, cacheDir: dir),
            Config(name: "X3c_eliza-5s-fp16_cpu", modelId: local, modelFile: "kokoro_eliza_5s_fp16", units: .cpuOnly, cacheDir: dir),
            Config(name: "X4_eliza-5s-fp16_pad0", modelId: local, modelFile: "kokoro_eliza_5s_fp16", units: .cpuAndNeuralEngine, boundary: 0, cacheDir: dir),
            Config(name: "X4c_eliza-5s-fp16_cpu_pad0", modelId: local, modelFile: "kokoro_eliza_5s_fp16", units: .cpuOnly, boundary: 0, cacheDir: dir),
            Config(name: "H1_fluid-5s_ane", modelId: fluid, modelFile: "kokoro_21_5s", units: .cpuAndNeuralEngine),
            Config(name: "B_fluid-15s_ane", modelId: fluid, modelFile: "kokoro_21_15s", units: .cpuAndNeuralEngine),
            Config(name: "L10h_eliza-10s-fp16_cpu", modelId: local, modelFile: "kokoro_eliza_10s_fp16", units: .cpuOnly, cacheDir: dir),
            Config(name: "L15h_eliza-15s-fp16_cpu", modelId: local, modelFile: "kokoro_eliza_15s_fp16", units: .cpuOnly, cacheDir: dir),
        ]
    }

    static let warbleConfigs: [Config] = [
        Config(name: "A_aufklarer-5s_ane", modelId: aufklarer, modelFile: "kokoro_5s", units: .cpuAndNeuralEngine),
        Config(name: "B_fluid-15s_ane", modelId: fluid, modelFile: "kokoro_21_15s", units: .cpuAndNeuralEngine),
        Config(name: "C_fluid-15s_ane_zero-phases", modelId: fluid, modelFile: "kokoro_21_15s", units: .cpuAndNeuralEngine, zeroPhases: true),
        Config(name: "F_fluid-15s_ane_pad0", modelId: fluid, modelFile: "kokoro_21_15s", units: .cpuAndNeuralEngine, boundary: 0),
        Config(name: "G_fluid-15s_ane_avg-voice", modelId: fluid, modelFile: "kokoro_21_15s", units: .cpuAndNeuralEngine, averaged: true),
        Config(name: "D_fluid-15s_cpu", modelId: fluid, modelFile: "kokoro_21_15s", units: .cpuOnly),
        Config(name: "E_fluid-15s_cpu_zero-phases", modelId: fluid, modelFile: "kokoro_21_15s", units: .cpuOnly, zeroPhases: true),
        Config(name: "H1_fluid-5s_ane", modelId: fluid, modelFile: "kokoro_21_5s", units: .cpuAndNeuralEngine),
        Config(name: "H2_fluid-5s_ane_zero-phases", modelId: fluid, modelFile: "kokoro_21_5s", units: .cpuAndNeuralEngine, zeroPhases: true),
        Config(name: "H3_fluid-5s_cpu", modelId: fluid, modelFile: "kokoro_21_5s", units: .cpuOnly),
        Config(name: "H4_fluid-5s_cpu_zero-phases", modelId: fluid, modelFile: "kokoro_21_5s", units: .cpuOnly, zeroPhases: true),
    ]

    func testRenderListeningSet() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let out = env["KOKORO_LISTENING_DIR"], !out.isEmpty else {
            throw XCTSkip("Set KOKORO_LISTENING_DIR to render the Kokoro listening set")
        }
        let outDir = URL(fileURLWithPath: out, isDirectory: true)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let bundle = env["KOKORO_AB_BUNDLE_DIR"].flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0, isDirectory: true) }
        let set = env["KOKORO_LISTENING_SET"]
        let long = set == "long" || set == "parens"
        let allConfigs = bundle.map { long ? Self.longConfigs($0) : Self.abConfigs($0) } ?? Self.warbleConfigs
        // KOKORO_LISTENING_CONFIGS: comma-separated config names to render; default all.
        let only = Set((env["KOKORO_LISTENING_CONFIGS"] ?? "").split(separator: ",").map(String.init))
        let configs = only.isEmpty ? allConfigs : allConfigs.filter { only.contains($0.name) }
        let texts = bundle == nil ? Self.warbleTexts
            : set == "parens" ? Self.parenTexts : (long ? Self.longTexts : Self.abTexts)

        var loadedKey: String?
        var model: KokoroTTSModel?
        for config in configs {
            // A bundle folder need not hold every graph of the A/B set yet.
            if let dir = config.cacheDir,
               !FileManager.default.fileExists(atPath: dir.appendingPathComponent("\(config.modelFile).mlmodelc").path) {
                print("LISTEN skip \(config.name): no \(config.modelFile).mlmodelc in \(dir.path)")
                continue
            }
            let key = "\(config.modelFile)-\(config.units.rawValue)"
            if key != loadedKey {
                model = nil
                model = try await KokoroTTSModel.fromPretrained(
                    modelId: config.modelId, modelFile: config.modelFile, voiceFilter: ["af_heart"],
                    cacheDir: config.cacheDir, offlineMode: config.cacheDir != nil,
                    computeUnits: config.units)
                loadedKey = key
            }
            guard let m = model, let network = m.network else { return XCTFail("not loaded") }
            network.experimentZeroPhases = config.zeroPhases
            m.boundaryTokenId = config.boundary
            m.experimentAveragedVoice = config.averaged
            print("LISTEN \(config.name): slots \(network.inputSlots), output \(network.outputSamples), speed input \(network.acceptsSpeed), budget \(m.chunkBudget)")

            for (label, text) in texts {
                var times: [Double] = []
                var jitters: [Double] = []
                var wobbles: [Double] = []
                var voiced: [Int] = []
                var pieces = 0
                var seconds = 0.0
                for run in 0..<3 {
                    var audio: [Float] = []
                    var count = 0
                    let t0 = CFAbsoluteTimeGetCurrent()
                    try m.synthesizeChunks(text: text, voice: "af_heart", language: "en", speed: 1) { samples, _ in
                        audio.append(contentsOf: samples)
                        count += 1
                    }
                    times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                    pieces = count
                    seconds = Double(audio.count) / 24000
                    if run == 0 {
                        try WAVWriter.write(samples: audio, sampleRate: 24000,
                                            to: outDir.appendingPathComponent("\(config.name)_\(label).wav"))
                    }
                    let stats = Self.pitchStats(audio)
                    jitters.append(stats.jitterCents)
                    wobbles.append(stats.wobbleCents)
                    voiced.append(stats.voicedFrames)
                }
                func mean(_ v: [Double]) -> Double { v.reduce(0, +) / Double(max(v.count, 1)) }
                print(String(format: "LISTEN %-30@ %-9@ pieces %d audio %.2f s | ms %@ | jitter %.1f c (%@) | wobble %.1f c (%@) | voiced %d",
                             config.name as NSString, label as NSString, pieces, seconds,
                             times.map { String(format: "%.0f", $0) }.joined(separator: "/"),
                             mean(jitters), jitters.map { String(format: "%.1f", $0) }.joined(separator: "/"),
                             mean(wobbles), wobbles.map { String(format: "%.1f", $0) }.joined(separator: "/"),
                             voiced.first ?? 0))
            }
        }
    }

    /// Autocorrelation pitch track: 40 ms frames, 10 ms hop, 70-400 Hz.
    /// jitterCents: mean |frame-to-frame F0 change| over adjacent voiced
    /// frames (changes over 250 cents, tracking errors or voicing breaks,
    /// excluded). wobbleCents: RMS deviation of F0 from its 5-frame moving
    /// average inside voiced runs of at least 5 frames.
    static func pitchStats(_ audio: [Float]) -> (jitterCents: Double, wobbleCents: Double, voicedFrames: Int) {
        let sr = 24000, window = 960, hop = 240
        let minLag = sr / 400, maxLag = sr / 70
        guard audio.count > window + maxLag + 1 else { return (0, 0, 0) }
        var squares = [Float](repeating: 0, count: audio.count)
        vDSP_vsq(audio, 1, &squares, 1, vDSP_Length(audio.count))
        var prefix = [Double](repeating: 0, count: audio.count + 1)
        for i in 0..<audio.count { prefix[i + 1] = prefix[i] + Double(squares[i]) }
        func energy(_ start: Int) -> Double { prefix[start + window] - prefix[start] }

        var f0: [Double?] = []
        var corr = [Float](repeating: 0, count: maxLag + 1)
        var start = 0
        while start + window + maxLag + 1 <= audio.count {
            let e0 = energy(start)
            if e0 / Double(window) < 0.02 * 0.02 {
                f0.append(nil)
                start += hop
                continue
            }
            audio.withUnsafeBufferPointer { buf in
                let base = buf.baseAddress! + start
                vDSP_conv(base, 1, base, 1, &corr, 1, vDSP_Length(maxLag + 1), vDSP_Length(window))
            }
            var best = -1.0, bestLag = 0
            var norm = [Double](repeating: 0, count: maxLag + 2)
            for lag in max(1, minLag - 1)...min(maxLag + 1, maxLag) {
                let denom = (e0 * energy(start + lag)).squareRoot()
                norm[lag] = denom > 0 ? Double(corr[lag]) / denom : 0
            }
            for lag in minLag...(maxLag - 1) where norm[lag] > best {
                best = norm[lag]
                bestLag = lag
            }
            guard best > 0.75, bestLag > minLag, bestLag < maxLag - 1 else {
                f0.append(nil)
                start += hop
                continue
            }
            let a = norm[bestLag - 1], b = norm[bestLag], c = norm[bestLag + 1]
            let denom = a - 2 * b + c
            let shift = denom != 0 ? 0.5 * (a - c) / denom : 0
            f0.append(Double(sr) / (Double(bestLag) + max(-0.5, min(0.5, shift))))
            start += hop
        }

        var changes: [Double] = []
        for i in 1..<f0.count {
            guard let p = f0[i - 1], let q = f0[i] else { continue }
            let cents = abs(1200 * log2(q / p))
            if cents < 250 { changes.append(cents) }
        }
        var deviations: [Double] = []
        var run: [Double] = []
        func flush() {
            if run.count >= 5 {
                let cents = run.map { 1200 * log2($0 / 100) }
                for i in 2..<(cents.count - 2) {
                    let avg = cents[(i - 2)...(i + 2)].reduce(0, +) / 5
                    deviations.append((cents[i] - avg) * (cents[i] - avg))
                }
            }
            run = []
        }
        for (i, value) in f0.enumerated() {
            if let value, run.isEmpty || abs(1200 * log2(value / run.last!)) < 250 {
                run.append(value)
            } else {
                flush()
                if let value { run.append(value) }
            }
            if i == f0.count - 1 { flush() }
        }
        let jitter = changes.isEmpty ? 0 : changes.reduce(0, +) / Double(changes.count)
        let wobble = deviations.isEmpty ? 0 : (deviations.reduce(0, +) / Double(deviations.count)).squareRoot()
        return (jitter, wobble, f0.compactMap { $0 }.count)
    }
}
