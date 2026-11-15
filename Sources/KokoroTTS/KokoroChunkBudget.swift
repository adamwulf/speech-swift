import Foundation

/// How many phoneme tokens one chunk may hold for a given end-to-end graph.
///
/// A graph renders a bounded output window, so the budget follows from how
/// much speech fits in it. Each pass spends part of the window on silence:
/// about 250 ms of lead-in before the first phoneme and about 500 ms of
/// trailing pause after the last, and the last 200 ms of the window are not
/// trusted. The rest is speech time, and the budget scales speech-core's
/// validated 5 s profile (56 tokens, hard cap 64, for 4.05 s of speech) to
/// it, capped by the graph's phoneme slots.
///
/// | Graph | Slots | Window | Budget | Hard cap |
/// |---|---|---|---|---|
/// | `kokoro_5s` | 128 | 120 000 (5 s) | 56 | 64 |
/// | `kokoro_21_15s` | 249 | 372 600 (15.5 s) | 201 | 230 |
struct KokoroChunkBudget: Equatable {
    /// Text is packed into chunks of at most this many tokens.
    let budget: Int
    /// No chunk grows past this many tokens: a short last chunk may join its
    /// predecessor up to here.
    let hardCap: Int

    /// The last 200 ms of the output window are not trusted, as in
    /// speech-core: speech reaching into them means the chunk was too long.
    /// All sample counts here are at Kokoro's 24 kHz.
    static let outputTailGuardSamples = 4800
    /// Silence each pass renders around its speech: ~250 ms of lead-in and
    /// ~500 ms of trailing pause.
    static let silenceSamples = 18_000
    /// The 5 s profile the budget scales: 56 tokens, hard cap 64, for the
    /// speech time of a 120 000-sample window.
    static let referenceBudget = 56
    static let referenceHardCap = 64
    static let referenceSpeechSamples = 120_000 - outputTailGuardSamples - silenceSamples

    /// The budget for a graph with `inputSlots` phoneme slots and an output
    /// window of `outputSamples`.
    static func forGraph(inputSlots: Int, outputSamples: Int) -> KokoroChunkBudget {
        let speech = max(0, outputSamples - outputTailGuardSamples - silenceSamples)
        let hardCap = max(1, min(speech * referenceHardCap / referenceSpeechSamples, inputSlots))
        let budget = max(1, min(speech * referenceBudget / referenceSpeechSamples, hardCap))
        return KokoroChunkBudget(budget: budget, hardCap: hardCap)
    }

    /// `kokoro_5s`: 128 input slots, a 5 s window.
    static let fiveSecond = forGraph(inputSlots: 128, outputSamples: 120_000)
}
