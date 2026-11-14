import XCTest
@testable import KokoroTTS

final class KokoroLeadInBurstTests: XCTestCase {

    let sr = 24_000

    /// Low noise of `count` samples at about `rms`, never exactly zero.
    func noise(_ count: Int, rms: Float) -> [Float] {
        (0..<count).map { i in (i % 2 == 0 ? rms : -rms) * (1 + 0.1 * Float(i % 7) / 7) }
    }

    func tone(_ count: Int, amplitude: Float) -> [Float] {
        (0..<count).map { amplitude * sin(2 * .pi * 200 * Float($0) / 24_000) + 1e-4 }
    }

    func testBurstBeforeExactSilenceIsZeroed() {
        // The 10 s fp16 shape: 27 ms of noise, exact zeros, then speech.
        let burst = noise(648, rms: 0.003)
        var audio = burst + [Float](repeating: 0, count: 2400) + tone(4800, amplitude: 0.3)
        let speech = Array(audio[(648 + 2400)...])
        XCTAssertTrue(KokoroLeadInBurst.silence(&audio, sampleRate: sr))
        XCTAssertTrue(audio[0..<648].allSatisfy { $0 == 0 })
        XCTAssertEqual(Array(audio[(648 + 2400)...]), speech, "speech is untouched")
        XCTAssertEqual(audio.count, 648 + 2400 + 4800, "timing is kept")
    }

    func testBurstBeforeSilenceUnder16BitResolutionIsZeroed() {
        // The silence after the burst is not always exact zeros: values
        // under 16-bit resolution count as silence too.
        let tiny = (0..<2400).map { Float($0 % 2 == 0 ? 2e-6 : -3e-6) }
        var audio = noise(648, rms: 0.003) + tiny + tone(4800, amplitude: 0.3)
        XCTAssertTrue(KokoroLeadInBurst.silence(&audio, sampleRate: sr))
        XCTAssertTrue(audio[0..<648].allSatisfy { $0 == 0 })
        XCTAssertEqual(audio[648], 2e-6, "the silence itself is left as rendered")
    }

    func testLongerQuietBurstIsZeroed() {
        // The 5 s fp16 shape: about 130 ms of quieter noise.
        var audio = noise(3120, rms: 0.0015) + [Float](repeating: 0, count: 1200) + tone(2400, amplitude: 0.3)
        XCTAssertTrue(KokoroLeadInBurst.silence(&audio, sampleRate: sr))
        XCTAssertTrue(audio[0..<3120].allSatisfy { $0 == 0 })
    }

    func testLowLevelLeadInIsUntouched() {
        // fp32 graphs: a lead-in around -74 dBFS, above 16-bit resolution.
        let original = noise(6000, rms: 0.0002) + tone(4800, amplitude: 0.3)
        var audio = original
        XCTAssertFalse(KokoroLeadInBurst.silence(&audio, sampleRate: sr))
        XCTAssertEqual(audio, original)
    }

    func testPassThatStartsSilentIsUntouched() {
        let original = [Float](repeating: 0, count: 2400) + tone(4800, amplitude: 0.3)
        var audio = original
        XCTAssertFalse(KokoroLeadInBurst.silence(&audio, sampleRate: sr))
        XCTAssertEqual(audio, original)
    }

    func testLoudAudioBeforeTheZerosIsKept() {
        // Speech-level sound before the zeros is never removed.
        let original = tone(1200, amplitude: 0.2) + [Float](repeating: 0, count: 2400) + tone(4800, amplitude: 0.3)
        var audio = original
        XCTAssertFalse(KokoroLeadInBurst.silence(&audio, sampleRate: sr))
        XCTAssertEqual(audio, original)
    }

    func testZerosAfterTheBurstWindowAreIgnored() {
        // A zero run that starts later than 200 ms is not a lead-in burst.
        let original = noise(6000, rms: 0.003) + [Float](repeating: 0, count: 2400) + tone(4800, amplitude: 0.3)
        var audio = original
        XCTAssertFalse(KokoroLeadInBurst.silence(&audio, sampleRate: sr))
        XCTAssertEqual(audio, original)
    }

    func testShortSilentRunIsNotEnough() {
        // 10 ms of silence is under the 20 ms run the signature needs.
        let original = noise(648, rms: 0.003) + [Float](repeating: 0, count: 240) + tone(4800, amplitude: 0.3)
        var audio = original
        XCTAssertFalse(KokoroLeadInBurst.silence(&audio, sampleRate: sr))
        XCTAssertEqual(audio, original)
    }

    func testEmptyAudio() {
        var audio: [Float] = []
        XCTAssertFalse(KokoroLeadInBurst.silence(&audio, sampleRate: sr))
    }
}
