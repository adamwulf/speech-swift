import XCTest
@testable import KokoroTTS

final class KokoroLeadInBurstTests: XCTestCase {

    let sr = 24_000
    /// 25 ms and 5 ms at 24 kHz.
    let lead = 600
    let fade = 120

    /// Low noise of `count` samples at about `rms`, never exactly zero.
    func noise(_ count: Int, rms: Float) -> [Float] {
        (0..<count).map { i in (i % 2 == 0 ? rms : -rms) * (1 + 0.1 * Float(i % 7) / 7) }
    }

    func tone(_ count: Int, amplitude: Float) -> [Float] {
        (0..<count).map { amplitude * sin(2 * .pi * 200 * Float($0) / 24_000) + 1e-4 }
    }

    func testTenSecondBurstBeforeSilenceIsCleared() {
        // The 10 s fp16 shape: 27 ms of noise, digital silence, then speech.
        var audio = noise(648, rms: 0.005) + [Float](repeating: 0, count: 2400) + tone(4800, amplitude: 0.3)
        let speech = Array(audio[(648 + 2400)...])
        XCTAssertTrue(KokoroLeadInBurst.silence(&audio, sampleRate: sr))
        XCTAssertTrue(audio[0..<lead].allSatisfy { $0 == 0 })
        XCTAssertTrue(audio[lead..<648].allSatisfy { abs($0) <= 0.005 * 1.1 * 48 / 120 },
                      "the burst's last 2 ms lie in the start of the fade")
        XCTAssertEqual(Array(audio[(648 + 2400)...]), speech, "speech is untouched")
        XCTAssertEqual(audio.count, 648 + 2400 + 4800, "timing is kept")
    }

    func testBurstRightBeforeShortReplyIsCleared() {
        // The "Yes." shape: the burst runs into speech starting at 30 ms,
        // with no silence between.
        var audio = noise(lead, rms: 0.006) + noise(fade, rms: 0.0005) + tone(4800, amplitude: 0.3)
        let speech = Array(audio[(lead + fade)...])
        XCTAssertTrue(KokoroLeadInBurst.silence(&audio, sampleRate: sr))
        XCTAssertTrue(audio[0..<lead].allSatisfy { $0 == 0 })
        XCTAssertEqual(Array(audio[(lead + fade)...]), speech, "speech from 30 ms on is untouched")
    }

    func testFadeRampsFromZeroAfterTheLeadIn() {
        var audio = [Float](repeating: 0.001, count: lead + fade + 100)
        XCTAssertTrue(KokoroLeadInBurst.silence(&audio, sampleRate: sr))
        XCTAssertEqual(audio[lead], 0)
        XCTAssertEqual(audio[lead + fade / 2], 0.0005, accuracy: 1e-7)
        XCTAssertEqual(audio[lead + fade], 0.001, "full level after the fade")
    }

    func testSilentLeadInStaysSilentAndSpeechIsUntouched() {
        // fp32 graphs: the first 25 ms are already silence.
        let original = noise(2400, rms: 0.00002) + tone(4800, amplitude: 0.3)
        var audio = original
        KokoroLeadInBurst.silence(&audio, sampleRate: sr)
        XCTAssertEqual(Array(audio[(lead + fade)...]), Array(original[(lead + fade)...]))
    }

    func testSpeechLevelStartIsKept() {
        // Anything as loud as speech in the first 25 ms is never removed.
        let original = tone(4800, amplitude: 0.2)
        var audio = original
        XCTAssertFalse(KokoroLeadInBurst.silence(&audio, sampleRate: sr))
        XCTAssertEqual(audio, original)
    }

    func testTooShortAudioIsUntouched() {
        let original = noise(lead + fade - 1, rms: 0.005)
        var audio = original
        XCTAssertFalse(KokoroLeadInBurst.silence(&audio, sampleRate: sr))
        XCTAssertEqual(audio, original)
        var empty: [Float] = []
        XCTAssertFalse(KokoroLeadInBurst.silence(&empty, sampleRate: sr))
    }
}
