import XCTest
@testable import KokoroTTS

/// Parentheses reach the model as Kokoro's own `(` and `)` tokens, so a
/// parenthetical aside is set off instead of read straight through.
final class KokoroParenthesesTests: XCTestCase {

    /// Letters the raw-letter fallback emits for the words below (no G2P
    /// models or dictionaries are loaded), plus the two bracket tokens at
    /// their ids in Kokoro's vocabulary.
    private func makeVocab() -> [String: Int] {
        ["<pad>": 0, "<bos>": 1, "<eos>": 2, "(": 12, ")": 13, " ": 16,
         "a": 43, "b": 44, "d": 46, "e": 47, "i": 51, "l": 54, "m": 55,
         "n": 56, "o": 57, "p": 58, "r": 60, "s": 61, "t": 62, "u": 63]
    }

    func testEnglishKeepsParentheses() {
        let phonemizer = KokoroPhonemizer(vocab: makeVocab())
        let phonemes = phonemizer.textToPhonemes("bring a laptop (and a pen) to us")
        XCTAssertTrue(phonemes.contains("("), "got \(phonemes)")
        XCTAssertTrue(phonemes.contains(")"), "got \(phonemes)")
        let ids = phonemizer.tokenize("bring a laptop (and a pen) to us")
        XCTAssertEqual(ids.filter { $0 == 12 }.count, 1)
        XCTAssertEqual(ids.filter { $0 == 13 }.count, 1)
        XCTAssertLessThan(ids.firstIndex(of: 12)!, ids.firstIndex(of: 13)!)
    }

    func testDictionaryLanguagesKeepParentheses() {
        let phonemizer = KokoroPhonemizer(vocab: makeVocab())
        for language in ["fr", "es", "pt", "it"] {
            let ids = phonemizer.tokenize("bonsoir (amis) merci", language: language)
            XCTAssertTrue(ids.contains(12) && ids.contains(13), "\(language): \(ids)")
        }
    }

    func testFullWidthParenthesesMapToTheSameTokens() {
        let phonemizer = KokoroPhonemizer(vocab: makeVocab())
        let french = phonemizer.tokenize("bonsoir （amis） merci", language: "fr")
        XCTAssertTrue(french.contains(12) && french.contains(13), "\(french)")
        let english = phonemizer.tokenize("bring a pen （to us）")
        XCTAssertTrue(english.contains(12) && english.contains(13), "\(english)")
    }
}
