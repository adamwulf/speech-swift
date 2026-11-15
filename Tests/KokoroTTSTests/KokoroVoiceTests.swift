import XCTest
@testable import KokoroTTS

final class KokoroVoiceTests: XCTestCase {

    /// A 4-dim voice whose rows are filled with their own key: row "k" is
    /// `[k, k, k, k]`, so a picked row names itself.
    private func voiceJSON(rows: Int, styleDim: Int = 4) -> [String: Any] {
        var json: [String: Any] = ["embedding": Array(repeating: 0.5, count: styleDim)]
        for k in stride(from: 1, through: rows, by: 1) {
            json[String(k)] = Array(repeating: Double(k), count: styleDim)
        }
        return json
    }

    // MARK: - Row selection

    func testRowIndexIsPhonemeCountMinusOneClamped() {
        // Upstream Kokoro: ref_s = pack[len(phonemes) - 1], 510 rows.
        XCTAssertEqual(KokoroVoice.rowIndex(phonemeCount: 1, rowCount: 510), 0)
        XCTAssertEqual(KokoroVoice.rowIndex(phonemeCount: 2, rowCount: 510), 1)
        XCTAssertEqual(KokoroVoice.rowIndex(phonemeCount: 140, rowCount: 510), 139)
        XCTAssertEqual(KokoroVoice.rowIndex(phonemeCount: 510, rowCount: 510), 509)
        XCTAssertEqual(KokoroVoice.rowIndex(phonemeCount: 0, rowCount: 510), 0)
        XCTAssertEqual(KokoroVoice.rowIndex(phonemeCount: -3, rowCount: 510), 0)
        XCTAssertEqual(KokoroVoice.rowIndex(phonemeCount: 900, rowCount: 510), 509)
    }

    func testStylePicksTheRowForTheChunkLength() throws {
        let voice = try XCTUnwrap(KokoroVoice.parse(voiceJSON(rows: 510), styleDim: 4))
        XCTAssertEqual(voice.rowCount, 510)
        // A chunk of n phonemes uses key "n".
        XCTAssertEqual(voice.style(phonemeCount: 1), [1, 1, 1, 1])
        XCTAssertEqual(voice.style(phonemeCount: 54), [54, 54, 54, 54])
        XCTAssertEqual(voice.style(phonemeCount: 510), [510, 510, 510, 510])
        XCTAssertEqual(voice.style(phonemeCount: 0), [1, 1, 1, 1])
        XCTAssertEqual(voice.style(phonemeCount: 700), [510, 510, 510, 510])
    }

    func testVoiceWithoutRowsAlwaysUsesTheEmbedding() throws {
        // The aufklarer voice files carry only "embedding".
        let voice = try XCTUnwrap(KokoroVoice.parse(voiceJSON(rows: 0), styleDim: 4))
        XCTAssertEqual(voice.rowCount, 0)
        XCTAssertEqual(voice.style(phonemeCount: 1), [0.5, 0.5, 0.5, 0.5])
        XCTAssertEqual(voice.style(phonemeCount: 140), [0.5, 0.5, 0.5, 0.5])
    }

    // MARK: - Parsing

    func testEmbeddingIsTruncatedToStyleDim() throws {
        let voice = try XCTUnwrap(KokoroVoice.parse(["embedding": [1, 2, 3, 4, 5, 6]], styleDim: 4))
        XCTAssertEqual(voice.embedding, [1, 2, 3, 4])
    }

    func testMissingOrShortEmbeddingIsNotAVoice() {
        XCTAssertNil(KokoroVoice.parse(["1": [1, 1, 1, 1]], styleDim: 4))
        XCTAssertNil(KokoroVoice.parse(["embedding": [1, 2]], styleDim: 4))
        XCTAssertNil(KokoroVoice.parse(["embedding": "no"], styleDim: 4))
    }

    func testRowsStopAtTheFirstGap() throws {
        var json = voiceJSON(rows: 3)
        json["5"] = [5.0, 5.0, 5.0, 5.0]
        let voice = try XCTUnwrap(KokoroVoice.parse(json, styleDim: 4))
        XCTAssertEqual(voice.rowCount, 3)
        XCTAssertEqual(voice.style(phonemeCount: 10), [3, 3, 3, 3])
    }

    func testMalformedRowFallsBackToTheEmbedding() throws {
        var json = voiceJSON(rows: 3)
        json["2"] = [2.0, 2.0]
        let voice = try XCTUnwrap(KokoroVoice.parse(json, styleDim: 4))
        XCTAssertEqual(voice.rowCount, 0)
        XCTAssertEqual(voice.style(phonemeCount: 1), [0.5, 0.5, 0.5, 0.5])
    }

    func testLoadReadsAVoiceFile() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("kokoro-voice-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try JSONSerialization.data(withJSONObject: voiceJSON(rows: 2)).write(to: url)
        let voice = try XCTUnwrap(KokoroVoice.load(from: url, styleDim: 4))
        XCTAssertEqual(voice.rowCount, 2)
        XCTAssertEqual(voice.style(phonemeCount: 2), [2, 2, 2, 2])
    }

    // MARK: - Voice filter

    func testVoiceFilterMatchesShellPatterns() {
        XCTAssertTrue(KokoroVoice.matches("jf_alpha", filter: nil))
        let english = ["a*", "b*"]
        XCTAssertTrue(KokoroVoice.matches("af_heart", filter: english))
        XCTAssertTrue(KokoroVoice.matches("bm_lewis", filter: english))
        XCTAssertFalse(KokoroVoice.matches("jf_alpha", filter: english))
        XCTAssertFalse(KokoroVoice.matches("zf_xiaobei", filter: english))
        XCTAssertTrue(KokoroVoice.matches("bf_emma", filter: ["bf_emma"]))
        XCTAssertFalse(KokoroVoice.matches("bf_emma2", filter: ["bf_emma"]))
        XCTAssertFalse(KokoroVoice.matches("af_heart", filter: []))
    }
}
