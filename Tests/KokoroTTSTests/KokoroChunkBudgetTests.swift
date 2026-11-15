import XCTest
@testable import KokoroTTS

final class KokoroChunkBudgetTests: XCTestCase {

    private func count(_ text: String) -> Int {
        2 + text.filter { $0 != " " && $0 != "\t" }.count
    }

    func testFiveSecondGraphKeepsSpeechCoreProfile() {
        // kokoro_5s: [1, 128] input_ids, [1, 1, 120000] audio.
        let budget = KokoroChunkBudget.forGraph(inputSlots: 128, outputSamples: 120_000)
        XCTAssertEqual(budget, KokoroChunkBudget(budget: 56, hardCap: 64))
        XCTAssertEqual(KokoroChunkBudget.fiveSecond, budget)
    }

    func testFifteenSecondGraphScalesWithItsSpeechTime() {
        // kokoro_21_15s: [1, 249] input_ids, [1, 1, 372600] audio. 349 800
        // samples of speech time against the 5 s graph's 97 200.
        let budget = KokoroChunkBudget.forGraph(inputSlots: 249, outputSamples: 372_600)
        XCTAssertEqual(budget, KokoroChunkBudget(budget: 201, hardCap: 230))
    }

    func testBudgetNeverExceedsTheInputSlots() {
        // A long window behind few slots: the slots bind both values.
        let budget = KokoroChunkBudget.forGraph(inputSlots: 128, outputSamples: 372_600)
        XCTAssertEqual(budget, KokoroChunkBudget(budget: 128, hardCap: 128))
    }

    func testBudgetGrowsWithTheWindow() {
        let tenSecond = KokoroChunkBudget.forGraph(inputSlots: 249, outputSamples: 240_000)
        XCTAssertGreaterThan(tenSecond.budget, KokoroChunkBudget.fiveSecond.budget)
        XCTAssertLessThan(tenSecond.budget, 201)
        XCTAssertLessThanOrEqual(tenSecond.budget, tenSecond.hardCap)
    }

    func testWindowWithNoSpeechTimeStillAllowsOneToken() {
        let budget = KokoroChunkBudget.forGraph(inputSlots: 128, outputSamples: 10_000)
        XCTAssertEqual(budget, KokoroChunkBudget(budget: 1, hardCap: 1))
    }

    func testFifteenSecondPlanKeepsALongSentenceWhole() {
        // 176 tokens: four chunks on the 5 s graph, one on the 15 s graph.
        let text = "The team spent most of the afternoon looking through the old reports in the basement "
            + "because nobody could remember where the original drawings for the north wing of the "
            + "building had been stored after the move."
        let fifteen = KokoroChunkBudget.forGraph(inputSlots: 249, outputSamples: 372_600)
        XCTAssertEqual(KokoroTTSModel.chunkPlan(text, countTokens: count, budget: fifteen), [text])
        XCTAssertEqual(KokoroTTSModel.chunkPlan(text, countTokens: count, budget: .fiveSecond).count, 4)
    }
}
