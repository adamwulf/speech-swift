import XCTest
@testable import KokoroTTS

/// The sentence, clause and retry cases mirror speech-core's
/// `test_text_chunker.cpp`; cuts between words and the merging of short
/// chunks intentionally diverge from it (see `KokoroTextChunker`). Token
/// counts use the same stand-in: one per non-space character plus two for
/// BOS/EOS.
final class KokoroTextChunkerTests: XCTestCase {

    private func count(_ text: String) -> Int {
        2 + text.filter { $0 != " " && $0 != "\t" }.count
    }

    private func chunk(_ text: String, budget: Int, hardCap: Int = 128, minTail: Int = 12) -> [String] {
        KokoroTextChunker.chunk(text, countTokens: count, maxTokens: budget,
                                hardCapTokens: hardCap, minTailTokens: minTail)
    }

    private func retry(_ text: String, min: Int, max: Int) -> [String] {
        KokoroTextChunker.splitForRetry(text, countTokens: count, minTokens: min, maxTokens: max)
    }

    // MARK: - Synthesizer plan

    func testPlanKeepsShortTextWhole() {
        let text = "Hello there. How are you today?"
        XCTAssertEqual(KokoroTTSModel.chunkPlan(text, countTokens: count), [text])
    }

    func testPlanMergesShortTailWithinHardCap() {
        let a = "I moved the planning meeting with the design team to Thursday."
        XCTAssertEqual(count(a), 54)
        // 54 + "See you." is 61 tokens: past the 56 budget, within the 64 cap.
        XCTAssertEqual(KokoroTTSModel.chunkPlan(a + " See you.", countTokens: count), [a + " See you."])
    }

    func testPlanKeepsTailWhenMergeExceedsHardCap() {
        let a = "I moved the planning meetings with the design teams to Thursday."
        XCTAssertEqual(count(a), 56)
        // Merged with "Thank you." it would be 65 tokens, past the 64 cap.
        XCTAssertEqual(KokoroTTSModel.chunkPlan(a + " Thank you.", countTokens: count), [a, "Thank you."])
    }

    func testPlanForLongClauseCutsBeforeThePhrase() {
        // The first clause is 105 tokens. Greedy filling cut it wherever the
        // 56-token budget ran out, so a few words like "feels off," could be
        // rendered on their own.
        let text = "So the next useful step is for you to run a few real conversations and tell me "
            + "where the detection or the voice still feels off, and I'll send that straight to an agent."
        XCTAssertEqual(KokoroTTSModel.chunkPlan(text, countTokens: count), [
            "So the next useful step is for you to run a few real conversations",
            "and tell me where the detection or the voice still feels off,",
            "and I'll send that straight to an agent.",
        ])
    }

    func testPlanForLongSentenceWithoutCommas() {
        // 176 tokens need four chunks. Greedy filling cut after "the old",
        // "where the" and "had been", leaving "stored after the move." at
        // 21 tokens; the even split cuts before "through", "because", "for".
        let text = "The team spent most of the afternoon looking through the old reports in the basement "
            + "because nobody could remember where the original drawings for the north wing of the "
            + "building had been stored after the move."
        let plan = KokoroTTSModel.chunkPlan(text, countTokens: count)
        XCTAssertEqual(plan, [
            "The team spent most of the afternoon looking",
            "through the old reports in the basement",
            "because nobody could remember where the original drawings",
            "for the north wing of the building had been stored after the move.",
        ])
        for piece in plan { XCTAssertLessThanOrEqual(count(piece), KokoroTTSModel.chunkTokenBudget) }
    }

    // MARK: - Cuts between words

    func testWordCutUsesFewestEvenPieces() {
        // 38 tokens over a 30-token budget: greedy filling gave 30 + 10.
        let text = "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii"
        let chunks = chunk(text, budget: 30, hardCap: 30)
        XCTAssertEqual(chunks.count, 2)
        XCTAssertLessThanOrEqual(abs(count(chunks[0]) - count(chunks[1])), 4)
        XCTAssertEqual(chunks.joined(separator: " "), text)
    }

    func testWordCutPrefersPhraseStartWhenNearlyBalanced() {
        // The most even cuts (18 + 22 tokens) fall before "dddd" or "eeee";
        // 14 + 26 before "with" is close enough to win.
        XCTAssertEqual(chunk("aaaa bbbb cccc with dddd eeee ffff gggg hhhh", budget: 30, hardCap: 30),
                       ["aaaa bbbb cccc", "with dddd eeee ffff gggg hhhh"])
    }

    func testPhraseStartersIgnoreCaseAndLeadingQuotes() {
        XCTAssertTrue(KokoroTextChunker.startsPhrase("Because"))
        XCTAssertTrue(KokoroTextChunker.startsPhrase("\"which"))
        XCTAssertFalse(KokoroTextChunker.startsPhrase("the"))
        XCTAssertFalse(KokoroTextChunker.startsPhrase("android"))
    }

    func testWordCutKeepsPiecesAboveMinimumWhereTheWordsAllowIt() {
        let words = KokoroTextChunker.splitWords("aaaa bbbb cccc with dddd eeee ffff gggg hhhh")
        let split: (Int) -> [String]? = {
            KokoroTextChunker.balancedWordSplit(words, countTokens: self.count, maxTokens: 30, minTokens: $0)
        }
        // A 16-token floor rules out the 14-token piece the phrase cut makes.
        XCTAssertEqual(split(12), ["aaaa bbbb cccc", "with dddd eeee ffff gggg hhhh"])
        let pieces = split(16)
        XCTAssertEqual(pieces?.count, 2)
        for piece in pieces ?? [] { XCTAssertGreaterThanOrEqual(count(piece), 16) }
    }

    func testWordCutBelowMinimumWhenNothingElseFits() {
        // Every cut leaves the last piece under 12 tokens; the most even one wins.
        let words = ["xxxxxxxxxxxxxxxxxxxxxxxx", "ab", "cd", "ef", "gh"]
        XCTAssertEqual(
            KokoroTextChunker.balancedWordSplit(words, countTokens: count, maxTokens: 30, minTokens: 12),
            ["xxxxxxxxxxxxxxxxxxxxxxxx", "ab cd ef gh"])
    }

    func testWordCutLeavesOversizedWordToCharacterSplit() {
        XCTAssertNil(KokoroTextChunker.balancedWordSplit(
            ["ab", String(repeating: "x", count: 40)], countTokens: count, maxTokens: 30, minTokens: 12))
        XCTAssertNil(KokoroTextChunker.balancedWordSplit(
            ["alone"], countTokens: count, maxTokens: 30, minTokens: 12))
    }

    func testWordCutRecountsWhenEstimateIsLow() {
        // A counter where one word pair costs 3 extra tokens together, which
        // neither per-word counts nor the whole-run average capture. The
        // estimate's only two-piece cut (22 + 22) counts 25 + 22 exactly, so
        // the search must recount and use three pieces.
        let paired: (String) -> Int = { text in
            2 + text.filter { $0 != " " }.count + (text.contains("dddd eeee") ? 3 : 0)
        }
        let words = KokoroTextChunker.splitWords("aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj")
        let pieces = KokoroTextChunker.balancedWordSplit(words, countTokens: paired, maxTokens: 24, minTokens: 12)
        XCTAssertEqual(pieces?.count, 3)
        for piece in pieces ?? [] { XCTAssertLessThanOrEqual(paired(piece), 24) }
        XCTAssertEqual(pieces?.joined(separator: " "), words.joined(separator: " "))
    }

    func testSeparatorIsCalibratedOnTheWholeRun() {
        // A lone "a" costs two tokens but one in context, like Kokoro's "A"
        // (eI alone, ɐ in "A few"), and every space costs one. Calibrating on
        // the first pair alone read the space as free, so every run looked a
        // token short per space and the recounts settled on uneven pieces
        // (18, 26, 26, 26, 21).
        let contextual: (String) -> Int = { text in
            let words = text.split(separator: " ")
            let letters = words.reduce(0) { sum, word in
                sum + (word == "a" ? (words.count == 1 ? 2 : 1) : word.count)
            }
            return 2 + letters + max(0, words.count - 1)
        }
        // 23 words over a 30-token budget: 5 pieces at least.
        let words = ["a"] + (0..<22).map { String(repeating: Character(UnicodeScalar(98 + $0 % 24)!), count: 4) }
        let pieces = KokoroTextChunker.balancedWordSplit(words, countTokens: contextual, maxTokens: 30, minTokens: 12)
        XCTAssertEqual(pieces?.count, 5)
        for piece in pieces ?? [] {
            XCTAssertLessThanOrEqual(contextual(piece), 30)
            XCTAssertGreaterThanOrEqual(contextual(piece), 20)
        }
        XCTAssertEqual(pieces?.joined(separator: " "), words.joined(separator: " "))
    }

    func testCutAfterPunctuationIsFree() {
        // As with a phrase starter, a cut right after the text's own clause
        // or sentence punctuation (15 + 22 tokens) beats the most even cut
        // (19 + 18).
        for mark in [",", "."] {
            let words = KokoroTextChunker.splitWords("aaaa bbbb cccc\(mark) dddd eeee ffff gggg hhhh")
            XCTAssertEqual(
                KokoroTextChunker.balancedWordSplit(words, countTokens: count, maxTokens: 30, minTokens: 12),
                ["aaaa bbbb cccc\(mark)", "dddd eeee ffff gggg hhhh"])
        }
    }

    func testShortOpeningClauseIsCutWithTheLongClause() {
        // "Okay," is 7 tokens. The long clause alone splits 26 + 29, too full
        // for "Okay," to join either piece, so it used to render on its own.
        let text = "Okay, aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj kkkk llll mmm"
        XCTAssertEqual(chunk(text, budget: 30, hardCap: 30), [
            "Okay, aaaa bbbb cccc",
            "dddd eeee ffff gggg hhhh",
            "iiii jjjj kkkk llll mmm",
        ])
    }

    func testShortOpeningClauseJoinsAClauseThatFitsOnlyAlone() {
        // The clause after "Okay," fits alone (26 tokens) but not with it
        // (31), so "Okay," used to render as a 7-token chunk.
        XCTAssertEqual(chunk("Okay, aaaa bbbb cccc dddd eeee ffff", budget: 30, hardCap: 30),
                       ["Okay, aaaa bbbb", "cccc dddd eeee ffff"])
    }

    func testShortClauseJoinsTheWordPieceBeforeIt() {
        // The first clause is cut 18 + 19; "though," (9 tokens) fits on the
        // second piece, so the clause after it, which fits alone (24) but not
        // with "though," (31), stays whole instead of gaining a word cut.
        let text = "aaaa bbbb cccc dddd eeee ffff gggg hhhh, though, kkkk llll mmmm nnnn oooo pp"
        XCTAssertEqual(chunk(text, budget: 30, hardCap: 30), [
            "aaaa bbbb cccc dddd",
            "eeee ffff gggg hhhh, though,",
            "kkkk llll mmmm nnnn oooo pp",
        ])
    }

    func testShortSentenceJoinsThePieceBeforeIt() {
        // "Okay." fits after the first sentence's second piece; the sentence
        // after it fits alone and keeps the plan at three chunks.
        let text = "aaaa bbbb cccc dddd eeee ffff ggg archive. Okay. kkkk llll mmmm nnnn oooo ppp."
        XCTAssertEqual(chunk(text, budget: 30, hardCap: 30), [
            "aaaa bbbb cccc dddd",
            "eeee ffff ggg archive. Okay.",
            "kkkk llll mmmm nnnn oooo ppp.",
        ])
    }

    func testShortSentenceIsCutWithTheLongSentenceAfterIt() {
        let text = "Okay. aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj kkkk llll mmm"
        XCTAssertEqual(chunk(text, budget: 30, hardCap: 30), [
            "Okay. aaaa bbbb cccc",
            "dddd eeee ffff gggg hhhh",
            "iiii jjjj kkkk llll mmm",
        ])
    }

    func testShortChunkJoinsItsOwnSentenceFirst() {
        let merge: ([String]) -> [String] = {
            KokoroTextChunker.mergeShortChunks($0, countTokens: self.count, maxTokens: 30,
                                               hardCapTokens: 30, minTokens: 12)
        }
        // Both joins fit; the one inside the sentence keeps the seam at the
        // sentence end, where the pause belongs.
        XCTAssertEqual(merge(["aaaa bbbb cccc overall.", "Still,", "the marketing team wants"]),
                       ["aaaa bbbb cccc overall.", "Still, the marketing team wants"])
        // Only the join across the sentence end fits.
        XCTAssertEqual(merge(["aaaa bbbb overall.", "Still,", "the marketing team wants more"]),
                       ["aaaa bbbb overall. Still,", "the marketing team wants more"])
        // A short sentence end joins the start of its own sentence.
        XCTAssertEqual(merge(["aaaa bbbb cccc", "dd.", "Next sentence here."]),
                       ["aaaa bbbb cccc dd.", "Next sentence here."])
    }

    func testShortClauseJoinsTheNextChunk() {
        // "Well," alone would be 7 tokens; it joins the first word piece.
        XCTAssertEqual(chunk("Well, aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii", budget: 30, hardCap: 30),
                       ["Well, aaaa bbbb cccc dddd", "eeee ffff gggg hhhh iiii"])
    }

    // MARK: - Packing

    func testEmptyAndWhitespace() {
        XCTAssertEqual(chunk("", budget: 72), [])
        XCTAssertEqual(chunk("   \n\t  ", budget: 72), [])
    }

    func testShortTextIsOneChunk() {
        XCTAssertEqual(chunk("Music stopped.", budget: 72), ["Music stopped."])
    }

    func testSentencesPackTogether() {
        XCTAssertEqual(chunk("Calling Anna. Please wait.", budget: 72), ["Calling Anna. Please wait."])
    }

    func testSplitsOnSentenceBoundary() {
        let a = "The first sentence is fairly long already."
        let b = "The second sentence is also fairly long."
        XCTAssertEqual(chunk(a + " " + b, budget: count(a) + 4), [a, b])
    }

    func testBudgetRespected() {
        let text = "I can call your contacts, dial numbers, look up phone numbers, "
            + "play or stop music, and set the volume, all offline, on this "
            + "device. Ask me anything you like. There is more where that came from!"
        let chunks = chunk(text, budget: 72)
        XCTAssertGreaterThanOrEqual(chunks.count, 2)
        for piece in chunks {
            XCTAssertLessThanOrEqual(count(piece), 72)
            XCTAssertFalse(piece.isEmpty)
        }
    }

    func testClauseFallbackForRunOnSentence() {
        let text = "first clause goes here, second clause goes here, third clause goes here, fourth clause goes here"
        let chunks = chunk(text, budget: 30)
        XCTAssertGreaterThanOrEqual(chunks.count, 2)
        for piece in chunks { XCTAssertLessThanOrEqual(count(piece), 30) }
    }

    func testWordFallback() {
        let chunks = chunk("one two three four five six seven eight nine ten", budget: 12, minTail: 3)
        XCTAssertGreaterThanOrEqual(chunks.count, 3)
        for piece in chunks { XCTAssertLessThanOrEqual(count(piece), 12) }
    }

    func testLongSentencePreservesWholeWords() {
        let text = "The new short-turn graph is ready for quick replies."
        let chunks = chunk(text, budget: 14, hardCap: 14, minTail: 0)
        XCTAssertGreaterThanOrEqual(chunks.count, 2)
        for piece in chunks { XCTAssertLessThanOrEqual(count(piece), 14) }
        XCTAssertEqual(chunks.joined(separator: " "), text)
        XCTAssertTrue(chunks.contains("short-turn"))
    }

    func testMonsterWordCharacterSplit() {
        let word = String(repeating: "x", count: 50)
        let chunks = chunk(word, budget: 12, minTail: 3)
        XCTAssertGreaterThanOrEqual(chunks.count, 4)
        for piece in chunks { XCTAssertLessThanOrEqual(count(piece), 12) }
        XCTAssertEqual(chunks.joined(), word)
    }

    func testCombiningMarksStayWithTheirLetter() {
        // "é" spelled as e + U+0301 is one Character; a cut must never strand
        // the accent at the start of a chunk.
        let word = String(repeating: "e\u{301}中", count: 12)
        let chunks = chunk(word, budget: 10, minTail: 2)
        XCTAssertGreaterThanOrEqual(chunks.count, 2)
        for piece in chunks {
            XCTAssertNotEqual(piece.unicodeScalars.first, "\u{301}")
        }
        XCTAssertEqual(chunks.joined(), word)
    }

    func testTinyTailMerges() {
        let a = "This is a reasonably sized first sentence here."
        let b = "Ok."
        XCTAssertEqual(chunk(a + " " + b, budget: count(a) + 2), [a + " " + b])
    }

    func testTinyTailKeptWhenHardCapBlocks() {
        let a = "This is a reasonably sized first sentence here."
        let b = "Ok."
        let budget = count(a) + 2
        XCTAssertEqual(chunk(a + " " + b, budget: budget, hardCap: budget), [a, b])
    }

    func testNewlinesAreBoundaries() {
        XCTAssertEqual(chunk("Line one has words\nLine two has words", budget: 18).count, 2)
        XCTAssertEqual(chunk("Line one has words\r\nLine two has words", budget: 18).count, 2)
    }

    // MARK: - Inner punctuation

    func testInnerPunctuationIsNotABoundary() {
        // Decimals, versions, domains, times, dates and German number formats
        // must reach the phonemizer exactly as written.
        let texts = [
            "It costs 3.5 euros.",
            "Version 1.2.3 is live at soniqo.audio today.",
            "Der Zug kommt um 18:30 Uhr, nicht um 19.15 Uhr.",
            "Es kostet 1.000 Euro, z.B. heute.",
            "Am 03.10.2026 kostet es 12,80 Euro.",
        ]
        for text in texts {
            XCTAssertEqual(chunk(text, budget: 72), [text])
        }
    }

    func testClauseSplitKeepsNumbersWhole() {
        let chunks = chunk("Um 18:30 Uhr kostet das Ticket 12,80 Euro, danach kostet es 15,20 Euro", budget: 40)
        XCTAssertEqual(chunks, ["Um 18:30 Uhr kostet das Ticket 12,80 Euro,", "danach kostet es 15,20 Euro"])
    }

    // MARK: - Retry split

    func testRetrySplitProgressFor13To42Tokens() {
        for parent in 13...42 {
            let word = String(repeating: "x", count: parent - 2)
            let pieces = retry(word, min: 6, max: parent - 1)
            XCTAssertEqual(pieces.count, 2, "parent \(parent)")
            for piece in pieces {
                XCTAssertFalse(piece.isEmpty)
                XCTAssertGreaterThanOrEqual(count(piece), 6)
                XCTAssertLessThan(count(piece), parent)
            }
            XCTAssertEqual(pieces.joined(), word)
        }
    }

    func testRetrySplitPrefersWordBoundaryWhenBalanced() {
        XCTAssertEqual(retry("alpha bravo", min: 6, max: 12), ["alpha", "bravo"])
    }

    func testRetrySplitDoesNotTradeWordBoundaryForBalance() {
        XCTAssertEqual(retry("alpha bravo charlie", min: 6, max: 16), ["alpha bravo", "charlie"])
    }

    func testRetrySplitKeepsPunctuationOnLeft() {
        XCTAssertEqual(retry("aaaaa. bbbbb", min: 6, max: 12), ["aaaaa.", "bbbbb"])
    }

    func testRetrySplitKeepsClosingQuoteWithSentence() {
        XCTAssertEqual(retry("aaaaa.\" bbbbb", min: 6, max: 13), ["aaaaa.\"", "bbbbb"])
    }

    func testRetrySplitKeepsFullSentenceEnderRun() {
        XCTAssertEqual(retry("aaaaa...?!\" bbbbb", min: 6, max: 18), ["aaaaa...?!\"", "bbbbb"])
    }

    func testRetrySplitDoesNotCutInsideANumber() {
        let pieces = retry("aaaaa 3.5 bbbbb", min: 6, max: 13)
        XCTAssertEqual(pieces.count, 2)
        XCTAssertEqual(pieces.joined(separator: " "), "aaaaa 3.5 bbbbb")
    }

    func testRetrySplitImpossibleReturnsEmpty() {
        XCTAssertEqual(retry("tiny", min: 6, max: 6), [])
    }
}
