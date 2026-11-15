import XCTest
import CoreML
@testable import KokoroTTS
@testable import Qwen3ASR

/// FluidAudio's 15 s end-to-end Kokoro bundle (`kokoro_21_15s.mlmodelc`):
/// 249 phoneme slots, a 372 600-sample window, no speed input, and voice
/// files with per-length rows.
/// Run with: swift test --filter E2EKokoroLongWindowTests
final class E2EKokoroLongWindowTests: XCTestCase {

    static let modelId = "FluidInference/kokoro-82m-coreml"
    static let modelFile = "kokoro_21_15s"

    private static var _sharedModel: KokoroTTSModel?

    private func model() async throws -> KokoroTTSModel {
        if let m = Self._sharedModel { return m }
        let m = try await KokoroTTSModel.fromPretrained(
            modelId: Self.modelId, modelFile: Self.modelFile,
            voiceFilter: ["af_heart", "bm_*"], computeUnits: .cpuAndNeuralEngine)
        Self._sharedModel = m
        return m
    }

    func testGraphShapeAndVoicesComeFromTheBundle() async throws {
        let m = try await model()
        let network = try XCTUnwrap(m.network)
        XCTAssertEqual(network.inputSlots, 249)
        XCTAssertEqual(network.outputSamples, 372_600)
        XCTAssertFalse(network.acceptsSpeed)
        XCTAssertEqual(m.config.maxPhonemeLength, 249)
        XCTAssertEqual(m.chunkBudget, KokoroChunkBudget(budget: 201, hardCap: 230))
        // The filter limits both the download and the load.
        XCTAssertEqual(m.availableVoices, ["af_heart", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"])
        for voice in m.voices.values { XCTAssertEqual(voice.rowCount, 510) }
    }

    func testLongSentenceRendersInOnePass() async throws {
        let m = try await model()
        let asr = try await Qwen3ASRModel.fromPretrained()
        let text = "The team spent most of the afternoon looking through the old reports in the basement "
            + "because nobody could remember where the original drawings for the north wing of the "
            + "building had been stored after the move."
        let tokens = m.phonemizer.tokenize(text, maxLength: .max, language: "en").count
        XCTAssertGreaterThan(tokens, KokoroChunkBudget.fiveSecond.hardCap * 2)

        var pieces: [[Float]] = []
        let t0 = CFAbsoluteTimeGetCurrent()
        try m.synthesizeChunks(text: text, voice: "af_heart", language: "en", speed: 1) { samples, _ in
            pieces.append(samples)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        let audio = Array(pieces.joined())
        let duration = Double(audio.count) / 24000
        print(String(format: "15 s graph: %d tokens -> %.2f s of audio in %.0f ms", tokens, duration, elapsed * 1000))
        // One chunk and no overflow retry.
        XCTAssertEqual(pieces.count, 1)
        XCTAssertGreaterThan(duration, 6)
        XCTAssertLessThan(duration, 15.3)

        let transcription = asr.transcribe(audio: audio, sampleRate: 24000).lowercased()
        print("Output: \"\(transcription)\"")
        let keywords = ["team", "afternoon", "reports", "basement", "remember", "drawings", "north", "building", "move"]
        let matched = keywords.filter { transcription.contains($0) }
        XCTAssertGreaterThanOrEqual(matched.count, 7, "Expected \(keywords) in \"\(transcription)\"")
    }

    func testSpeedIsIgnoredWithoutASpeedInput() async throws {
        // Durations are deterministic and a lone chunk is not trimmed, so
        // the same text renders the same length at any requested speed.
        let m = try await model()
        let normal = try m.synthesize(text: "Hello there, how are you today?", voice: "af_heart", speed: 1)
        let fast = try m.synthesize(text: "Hello there, how are you today?", voice: "af_heart", speed: 1.5)
        XCTAssertGreaterThan(normal.count, 12_000)
        XCTAssertEqual(fast.count, normal.count)
    }
}
