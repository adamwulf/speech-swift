import XCTest
@testable import KokoroTTS

/// A sentence rendered as several chunks must not pause at the seams longer
/// than Kokoro pauses inside one chunk, while a seam at a sentence end keeps
/// the model's own pause. Before seams were trimmed, every seam was 0.7-0.9 s
/// of silence.
/// Run with: swift test --filter E2EKokoroSeamTests
final class E2EKokoroSeamTests: XCTestCase {

    private static var _sharedModel: KokoroTTSModel?

    private func model() async throws -> KokoroTTSModel {
        if let m = Self._sharedModel { return m }
        let m = try await KokoroTTSModel.fromPretrained()
        Self._sharedModel = m
        return m
    }

    /// Inside one chunk, a comma pauses about 210-275 ms at speed 1 and most
    /// word boundaries have no silence; the longest seen were about 250 ms.
    private static let maxWordSeamMs = 80
    private static let maxClauseSeamMs = 250
    /// A sentence end keeps the model's own pause, about 0.7-0.9 s.
    private static let minSentenceSeamMs = 400

    func testSeamsInsideASentenceKeepShortGaps() async throws {
        let m = try await model()
        let texts = [
            "So the next useful step is for you to run a few real conversations and tell me where "
                + "the detection or the voice still feels off, and I'll send that straight to an agent.",
            "The team spent most of the afternoon looking through the old reports in the basement "
                + "because nobody could remember where the original drawings for the north wing of the "
                + "building had been stored after the move.",
            "The meeting starts at nine tomorrow morning. "
                + "Please bring the budget report and the travel plans for next month.",
        ]
        let count: (String) -> Int = { m.phonemizer.tokenize($0, maxLength: .max, language: "en").count }
        for text in texts {
            let plan = KokoroTTSModel.chunkPlan(text, countTokens: count, budget: m.chunkBudget)
            XCTAssertGreaterThan(plan.count, 1, "The sentence must span several chunks")
            for speed: Float in [1.0, 1.2] {
                var pieces: [[Float]] = []
                try m.synthesizeChunks(text: text, voice: "af_heart", language: "en", speed: speed) { samples, _ in
                    pieces.append(samples)
                }
                // An overflow retry would add pieces the plan does not list.
                guard pieces.count == plan.count else {
                    XCTFail("Expected \(plan.count) pieces, got \(pieces.count) at speed \(speed)")
                    continue
                }
                let audio = pieces.flatMap { $0 }
                var seam = 0
                for (index, chunk) in plan.dropLast().enumerated() {
                    seam += pieces[index].count
                    let gap = silenceMs(audio, around: seam)
                    let label = "seam after \"\(chunk)\" at speed \(speed): \(gap) ms"
                    print(label)
                    switch KokoroChunkSeams.kind(after: chunk) {
                    case .word: XCTAssertLessThanOrEqual(gap, Self.maxWordSeamMs, label)
                    case .clause: XCTAssertLessThanOrEqual(gap, Self.maxClauseSeamMs, label)
                    case .sentence: XCTAssertGreaterThan(gap, Self.minSentenceSeamMs, label)
                    }
                }
            }
        }
    }

    /// Length of the silence around sample `seam`: the run of 10 ms windows
    /// (5 ms hop) under 0.005 RMS that touches it.
    private func silenceMs(_ audio: [Float], around seam: Int) -> Int {
        let window = 240, hop = 120
        func silent(_ start: Int) -> Bool {
            guard start >= 0, start + window <= audio.count else { return false }
            var sum: Float = 0
            for i in start..<(start + window) { sum += audio[i] * audio[i] }
            return (sum / Float(window)).squareRoot() < 0.005
        }
        // The window grid is anchored at the seam, so the run is measured
        // from it outwards in whole hops.
        guard silent(seam - window / 2) || silent(seam - window) || silent(seam) else { return 0 }
        var before = seam
        while silent(before - window) { before -= hop }
        var after = seam
        while silent(after) { after += hop }
        return (max(after, seam) - min(before, seam)) * 1000 / 24000
    }
}
