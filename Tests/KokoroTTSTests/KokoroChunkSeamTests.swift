import XCTest
@testable import KokoroTTS

/// Seam trimming on synthetic signals; no model needed.
final class KokoroChunkSeamTests: XCTestCase {

    private let rate = 24000

    private func samples(ms: Int) -> Int { ms * rate / 1000 }

    private func silence(ms: Int) -> [Float] { [Float](repeating: 0, count: samples(ms: ms)) }

    /// A 220 Hz tone; the phase offset keeps its first and last samples nonzero.
    private func tone(ms: Int, amplitude: Float = 0.3) -> [Float] {
        (0..<samples(ms: ms)).map { amplitude * sin(2 * .pi * 220 * Float($0) / Float(rate) + 1) }
    }

    /// Deterministic noise of the given RMS: a stand-in for a soft fricative
    /// (h, f, s, th) or for the graph's low-level lead-in.
    private func noise(ms: Int, rms: Float, seed: UInt64 = 1) -> [Float] {
        var state = seed
        let peak = rms * Float(3).squareRoot()
        return (0..<samples(ms: ms)).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let unit = Float(state >> 40) / Float(1 << 24)
            return peak * (2 * unit - 1)
        }
    }

    private func leadingZeros(_ x: [Float]) -> Int { x.firstIndex { $0 != 0 } ?? x.count }
    private func trailingZeros(_ x: [Float]) -> Int { x.count - 1 - (x.lastIndex { $0 != 0 } ?? -1) }

    // MARK: - Seam kinds

    func testKindFollowsTheTextBeforeTheSeam() {
        let kind = KokoroChunkSeams.kind(after:)
        XCTAssertEqual(kind("So the next useful step is for you"), .word)
        XCTAssertEqual(kind("costs 3.5"), .word)
        XCTAssertEqual(kind("the voice still feels off,"), .clause)
        XCTAssertEqual(kind("one thing;"), .clause)
        XCTAssertEqual(kind("Note:"), .clause)
        XCTAssertEqual(kind("straight to an agent."), .sentence)
        XCTAssertEqual(kind("Really?\""), .sentence)
        XCTAssertEqual(kind("(wait!)"), .sentence)
        XCTAssertEqual(kind("你好，"), .clause)
        XCTAssertEqual(kind("这是一个测试。"), .sentence)
        XCTAssertEqual(kind(""), .sentence)
    }

    func testKeptSilenceByKindAndSpeed() {
        let kept = { KokoroChunkSeams.keptSilence(after: $0, speed: $1, sampleRate: self.rate) }
        // Half of each gap on either side of the seam.
        XCTAssertEqual(kept("run a few", 1.0), 240)
        XCTAssertEqual(kept("feels off,", 1.0), 1920)
        XCTAssertNil(kept("to an agent.", 1.0))
        XCTAssertEqual(kept("run a few", 1.2), 200)
        XCTAssertEqual(kept("feels off,", 1.2), 1600)
    }

    // MARK: - Trimming

    func testTrimCutsSilenceAroundSpeechToTheKeptAmount() {
        let speech = tone(ms: 400)
        let audio = silence(ms: 300) + speech + silence(ms: 500)
        let out = KokoroChunkSeams.trim(audio, keepLeading: 240, keepTrailing: 240)
        let lead = leadingZeros(out)
        // The cut lands on the 5 ms analysis grid, never inside the kept amount.
        XCTAssertGreaterThanOrEqual(lead, 240)
        XCTAssertLessThanOrEqual(lead, 240 + KokoroChunkSeams.hopSamples)
        XCTAssertGreaterThanOrEqual(trailingZeros(out), 240)
        XCTAssertLessThanOrEqual(trailingZeros(out), 240 + KokoroChunkSeams.windowSamples)
        XCTAssertEqual(Array(out[lead..<(lead + speech.count)]), speech)
        XCTAssertLessThan(out.count, samples(ms: 450))
    }

    func testTrimKeepsSoftOnsetAndEnding() {
        // A fricative-level onset and ending around a vowel must survive whole.
        let speech = noise(ms: 80, rms: 0.004, seed: 7) + tone(ms: 300) + noise(ms: 80, rms: 0.004, seed: 9)
        let audio = silence(ms: 300) + speech + silence(ms: 500)
        let out = KokoroChunkSeams.trim(audio, keepLeading: 240, keepTrailing: 240)
        let lead = leadingZeros(out)
        XCTAssertLessThanOrEqual(lead, 240 + KokoroChunkSeams.hopSamples)
        XCTAssertEqual(Array(out[lead..<(lead + speech.count)]), speech)
        XCTAssertGreaterThanOrEqual(trailingZeros(out), 240)
    }

    func testTrimTreatsLowLevelLeadInAsSilence() {
        // The graph's own lead-in hums under 0.001 RMS before speech starts.
        let audio = noise(ms: 250, rms: 0.0005) + tone(ms: 300) + silence(ms: 500)
        let out = KokoroChunkSeams.trim(audio, keepLeading: 240, keepTrailing: nil)
        let removed = audio.count - out.count
        XCTAssertGreaterThanOrEqual(removed, samples(ms: 250) - 240 - KokoroChunkSeams.hopSamples)
        XCTAssertLessThanOrEqual(removed, samples(ms: 250) - 240)
        // The trailing end is left alone.
        XCTAssertEqual(trailingZeros(out), samples(ms: 500))
    }

    func testNilLeavesThatEndAlone() {
        let audio = silence(ms: 300) + tone(ms: 400) + silence(ms: 500)
        XCTAssertEqual(KokoroChunkSeams.trim(audio, keepLeading: nil, keepTrailing: nil), audio)
        let headOnly = KokoroChunkSeams.trim(audio, keepLeading: 240, keepTrailing: nil)
        XCTAssertEqual(trailingZeros(headOnly), samples(ms: 500))
        let tailOnly = KokoroChunkSeams.trim(audio, keepLeading: nil, keepTrailing: 240)
        XCTAssertEqual(leadingZeros(tailOnly), samples(ms: 300))
    }

    func testKeptAmountLargerThanTheSilenceKeepsItAll() {
        let audio = silence(ms: 20) + tone(ms: 400) + silence(ms: 20)
        XCTAssertEqual(KokoroChunkSeams.trim(audio, keepLeading: 4800, keepTrailing: 4800), audio)
    }

    func testSilentOrTinyAudioIsReturnedUnchanged() {
        let quiet = silence(ms: 1000)
        XCTAssertEqual(KokoroChunkSeams.trim(quiet, keepLeading: 240, keepTrailing: 240), quiet)
        let tiny: [Float] = [0.5, -0.5, 0.5]
        XCTAssertEqual(KokoroChunkSeams.trim(tiny, keepLeading: 0, keepTrailing: 0), tiny)
    }

    func testCutEndFadesOut() {
        // A residual hum under the threshold is cut, and what is left of it
        // fades to zero so the seam does not click.
        let audio = silence(ms: 100) + tone(ms: 400) + noise(ms: 500, rms: 0.0015, seed: 3)
        let out = KokoroChunkSeams.trim(audio, keepLeading: nil, keepTrailing: 240)
        XCTAssertLessThan(out.count, audio.count)
        XCTAssertEqual(out.last, 0)
        let tail = out.suffix(KokoroChunkSeams.fadeOutSamples)
        let original = audio[(out.count - tail.count)..<out.count]
        for (faded, raw) in zip(tail, original) { XCTAssertLessThanOrEqual(abs(faded), abs(raw)) }
    }
}
