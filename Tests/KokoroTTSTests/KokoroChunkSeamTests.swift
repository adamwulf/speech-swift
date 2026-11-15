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

    // MARK: - Which ends of a piece are trimmed

    private func keeps(_ previous: String?, _ text: String, final isFinal: Bool = false,
                       speed: Float = 1.0) -> (leading: Int?, trailing: Int?) {
        KokoroChunkSeams.keeps(previousText: previous, text: text, isFinal: isFinal,
                               speed: speed, sampleRate: rate)
    }

    func testFirstPieceLeadInIsCutToAHundredMilliseconds() {
        let k = keeps(nil, "So the next useful step is for you")
        XCTAssertEqual(k.leading, 2400)
        XCTAssertEqual(k.trailing, 240)
        XCTAssertEqual(keeps(nil, "Yes.", final: true).leading, 2400)
        XCTAssertEqual(keeps(nil, "Yes.", final: true, speed: 1.25).leading, 1920)
    }

    func testShortLeadInIsLeftAlone() {
        // A lead-in already under 100 ms (BOS 1's usual) is not cut.
        let audio = silence(ms: 75) + tone(ms: 300) + silence(ms: 200)
        let k = keeps(nil, "Yes.", final: true)
        let out = KokoroChunkSeams.trim(audio, keepLeading: k.leading, keepTrailing: k.trailing)
        XCTAssertEqual(out, audio)
    }

    func testLongLeadInIsCut() {
        // BOS 0's ~450 ms lead-in before a short reply comes out at ~100 ms.
        let audio = noise(ms: 450, rms: 0.0005) + tone(ms: 300) + silence(ms: 200)
        let k = keeps(nil, "Yes.", final: true)
        let out = KokoroChunkSeams.trim(audio, keepLeading: k.leading, keepTrailing: k.trailing)
        let cut = audio.count - out.count
        let window = KokoroChunkSeams.windowSamples
        XCTAssertTrue(((samples(ms: 350) - window)...samples(ms: 350)).contains(cut), "\(cut)")
        XCTAssertEqual(Array(out.suffix(samples(ms: 500))), Array(audio.suffix(samples(ms: 500))),
                       "speech and tail are untouched")
    }

    func testMiddlePieceIsTrimmedByTheTextOnEachSide() {
        // Leading from the word cut before it, trailing from its own clause end.
        let k = keeps("So the next useful step is for you",
                      "to run a few real conversations and tell me where it feels off,")
        XCTAssertEqual(k.leading, 240)
        XCTAssertEqual(k.trailing, 1920)
        let faster = keeps("So the next useful step is for you", "to run a few", speed: 1.2)
        XCTAssertEqual(faster.leading, 200)
        XCTAssertEqual(faster.trailing, 200)
    }

    func testFinalPieceKeepsItsTail() {
        let k = keeps("the voice still feels off,", "and I'll send that straight to an agent", final: true)
        XCTAssertEqual(k.leading, 1920)
        XCTAssertNil(k.trailing)
    }

    func testSentenceSeamKeepsTheModelsPause() {
        let before = keeps("Please bring", "the budget report.")
        XCTAssertEqual(before.leading, 240)
        XCTAssertNil(before.trailing)
        let after = keeps("the budget report.", "We start at nine")
        XCTAssertNil(after.leading)
        XCTAssertEqual(after.trailing, 240)
    }

    func testRetryPiecesAreTrimmedByTheirOwnText() {
        // A chunk that overran is re-split; the cut between its pieces is a
        // seam like any other, here after a clause mark.
        let count: (String) -> Int = { 2 + $0.filter { $0 != " " }.count }
        let pieces = KokoroTextChunker.splitForRetry(
            "the old reports in the basement, because nobody could", countTokens: count,
            minTokens: 6, maxTokens: 40)
        XCTAssertEqual(pieces, ["the old reports in the basement,", "because nobody could"])
        guard pieces.count == 2 else { return }
        let first = keeps("They looked through", pieces[0])
        XCTAssertEqual(first.leading, 240)
        XCTAssertEqual(first.trailing, 1920)
        let second = keeps(pieces[0], pieces[1], final: true)
        XCTAssertEqual(second.leading, 1920)
        XCTAssertNil(second.trailing)
    }

    // MARK: - Emitting pieces

    func testEmitPiecesTrimsEachSeamByThePieceTexts() throws {
        let speech = tone(ms: 300)
        let framed = silence(ms: 300) + speech + silence(ms: 500)
        // Starts on sound, so the fade-in shows in its first sample.
        let open = speech + silence(ms: 500)
        let chunks = ["So the next step is for you", "to run a few real conversations,",
                      "and tell me what feels off."]
        // The middle and the last chunk come back as two retry pieces each.
        let rendered: [String: [(text: String, audio: [Float])]] = [
            chunks[0]: [(chunks[0], framed)],
            chunks[1]: [("to run a few", open), ("real conversations,", framed)],
            chunks[2]: [("and tell me", framed), ("what feels off.", framed)],
        ]
        var requested: [String] = []
        var emitted: [(audio: [Float], isFinal: Bool)] = []
        try KokoroTTSModel.emitPieces(
            of: chunks, speed: 1.0, sampleRate: rate,
            render: { chunk in
                requested.append(chunk)
                return rendered[chunk] ?? []
            },
            onChunk: { emitted.append(($0, $1)) })

        XCTAssertEqual(requested, chunks)
        XCTAssertEqual(emitted.map { $0.isFinal }, [false, false, false, false, true])
        guard emitted.count == 5 else { return }
        let word = 240, clause = 1920
        let hop = KokoroChunkSeams.hopSamples, window = KokoroChunkSeams.windowSamples
        let lead = emitted.map { leadingZeros($0.audio) }
        let tail = emitted.map { trailingZeros($0.audio) }
        // First piece: the utterance's 300 ms lead-in is cut to about 100 ms;
        // its tail before a word cut is trimmed.
        let leadIn = samples(ms: 100)
        XCTAssertTrue((leadIn...(leadIn + hop)).contains(lead[0]), "\(lead[0])")
        XCTAssertEqual(emitted[0].audio[lead[0]], framed[samples(ms: 300)])
        XCTAssertTrue((word...(word + window)).contains(tail[0]), "\(tail[0])")
        // First retry piece: faded in, and trimmed by its own text, a word
        // cut, not by its chunk's clause end.
        XCTAssertNotEqual(open.first, 0)
        XCTAssertEqual(emitted[1].audio.first, 0)
        XCTAssertTrue((word...(word + window)).contains(tail[1]), "\(tail[1])")
        // Second retry piece: leading from the piece before it, trailing from
        // its own clause end.
        XCTAssertTrue((word...(word + hop)).contains(lead[2]), "\(lead[2])")
        XCTAssertTrue((clause...(clause + window)).contains(tail[2]), "\(tail[2])")
        // The last chunk's first piece follows a clause and is not final.
        XCTAssertTrue((clause...(clause + hop)).contains(lead[3]), "\(lead[3])")
        XCTAssertTrue((word...(word + window)).contains(tail[3]), "\(tail[3])")
        // Final piece: its tail ends the utterance and is left alone.
        XCTAssertTrue((word...(word + hop)).contains(lead[4]), "\(lead[4])")
        XCTAssertEqual(tail[4], samples(ms: 500))
    }

    func testEmitPiecesDoesNotFadeTheFirstPiece() throws {
        // A first piece that starts on sound keeps its first samples.
        let open = tone(ms: 300) + silence(ms: 500)
        var emitted: [[Float]] = []
        try KokoroTTSModel.emitPieces(
            of: ["Hello there", "and welcome."], speed: 1.0, sampleRate: rate,
            render: { [($0, open)] },
            onChunk: { audio, _ in emitted.append(audio) })
        XCTAssertEqual(emitted.count, 2)
        XCTAssertNotEqual(open.first, 0)
        XCTAssertEqual(emitted.first?.prefix(KokoroChunkSeams.hopSamples), open.prefix(KokoroChunkSeams.hopSamples))
        XCTAssertEqual(emitted.last?.first, 0)
    }

    func testEmitPiecesWithNothingToSpeakYieldsOneEmptyFinalPiece() throws {
        var emitted: [(audio: [Float], isFinal: Bool)] = []
        try KokoroTTSModel.emitPieces(
            of: [], speed: 1.0, sampleRate: rate,
            render: { _ in
                XCTFail("Nothing to render")
                return []
            },
            onChunk: { emitted.append(($0, $1)) })
        XCTAssertEqual(emitted.count, 1)
        XCTAssertEqual(emitted.first?.audio, [])
        XCTAssertEqual(emitted.first?.isFinal, true)
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
