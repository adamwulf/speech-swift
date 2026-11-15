import XCTest
@testable import KokoroTTS

final class KokoroTrailingArtifactsTests: XCTestCase {

    let sr = 24_000
    /// 300 ms of speech-level tone ends here in every case.
    let speechEnd = 7_200
    /// 50 ms, the walk-back window: the anchor lands less than one window
    /// past the tone's end.
    let anchorWindow = 1_200
    /// 10 ms, the run-on window and the fade.
    let tail = 240

    /// Noise of `count` samples at about `rms` (about 4 % above), never exactly zero.
    func noise(_ count: Int, rms: Float) -> [Float] {
        (0..<count).map { i in (i % 2 == 0 ? rms : -rms) * (1 + 0.1 * Float(i % 7) / 7) }
    }

    func tone(_ count: Int, amplitude: Float) -> [Float] {
        (0..<count).map { amplitude * sin(2 * .pi * 200 * Float($0) / 24_000) + 1e-4 }
    }

    func zeros(_ count: Int) -> [Float] { [Float](repeating: 0, count: count) }

    func testSoftFinalFricativeIsKept() {
        // The masked 5 s fp16 export's "voice." at speed 1.15: about 150 ms
        // of /s/ at 0.010 RMS after the vowel, under the 0.015 floor of the
        // 50 ms walk-back, then silence. It used to be zeroed whole.
        let fricative = 3_600
        let original = tone(speechEnd, amplitude: 0.1) + noise(fricative, rms: 0.010) + zeros(12_000)
        var audio = original
        let end = KokoroTrailingArtifacts.clear(&audio, speed: 1.15, sampleRate: sr)
        XCTAssertGreaterThanOrEqual(end, speechEnd + fricative, "the whole fricative is kept")
        XCTAssertLessThan(end, speechEnd + fricative + tail, "and the silence after it is not")
        XCTAssertEqual(Array(audio[..<(speechEnd + fricative - tail)]),
                       Array(original[..<(speechEnd + fricative - tail)]),
                       "nothing before the fade changes")
        XCTAssertTrue(audio[end...].allSatisfy { $0 == 0 })
    }

    func testIsolatedSpikeAfterSilenceIsZeroed() {
        // A 15 ms artifact 150 ms after the speech: 0.011 RMS over 50 ms,
        // so the walk-back passes it, and the run-on stops at the silence.
        var audio = tone(speechEnd, amplitude: 0.1) + zeros(3_600) + noise(360, rms: 0.020) + zeros(7_200)
        let end = KokoroTrailingArtifacts.clear(&audio, speed: 1, sampleRate: sr)
        XCTAssertGreaterThan(end, speechEnd)
        XCTAssertLessThanOrEqual(end, speechEnd + anchorWindow)
        XCTAssertTrue(audio[end...].allSatisfy { $0 == 0 }, "the spike is zeroed")
    }

    func testSpeechEndingInSilenceKeepsItsSpeech() {
        let original = tone(speechEnd, amplitude: 0.1) + zeros(12_000)
        var audio = original
        let end = KokoroTrailingArtifacts.clear(&audio, speed: 1, sampleRate: sr)
        XCTAssertGreaterThan(end, speechEnd)
        XCTAssertLessThanOrEqual(end, speechEnd + anchorWindow)
        XCTAssertEqual(Array(audio[..<(speechEnd - tail)]), Array(original[..<(speechEnd - tail)]))
    }

    func testRunOnStopsAtItsReachScaledBySpeed() {
        // Low noise that never stops is kept for at most 250 ms past the
        // last speech window at speed 1, and 125 ms at speed 2.
        let original = tone(speechEnd, amplitude: 0.1) + noise(24_000, rms: 0.005)
        for (speed, reach) in [(Float(1), 6_000), (Float(2), 3_000)] {
            var audio = original
            let end = KokoroTrailingArtifacts.clear(&audio, speed: speed, sampleRate: sr)
            XCTAssertGreaterThan(end, speechEnd + reach - tail, "speed \(speed)")
            XCTAssertLessThanOrEqual(end, speechEnd + anchorWindow + reach, "speed \(speed)")
            XCTAssertTrue(audio[end...].allSatisfy { $0 == 0 }, "speed \(speed)")
        }
    }

    func testPassWithNoSpeechLevelWindowKeepsItsLength() {
        // Nothing reaches the walk-back floor: nothing is zeroed, and only
        // the last 10 ms fade out.
        let original = noise(12_000, rms: 0.005)
        var audio = original
        XCTAssertEqual(KokoroTrailingArtifacts.clear(&audio, speed: 1, sampleRate: sr), original.count)
        XCTAssertEqual(Array(audio[..<(original.count - tail)]), Array(original[..<(original.count - tail)]))
        XCTAssertEqual(audio.last, 0)
    }

    func testShortAndEmptyAudio() {
        var short = tone(600, amplitude: 0.1)
        XCTAssertEqual(KokoroTrailingArtifacts.clear(&short, speed: 1, sampleRate: sr), 600)
        var empty: [Float] = []
        XCTAssertEqual(KokoroTrailingArtifacts.clear(&empty, speed: 1, sampleRate: sr), 0)
    }
}
