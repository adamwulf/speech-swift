import Foundation

/// Shortens the silence where two separately rendered pieces meet.
///
/// Each chunk renders as a complete utterance: the graph opens with about
/// 250 ms of its own lead-in silence, and the trailing artifacts the
/// synthesizer zeroes leave about 500 ms more at the end, so a cut in the
/// middle of a sentence would pause for most of a second. Where the text
/// before a seam does not end a sentence, the silence on both sides is cut
/// down to no more than Kokoro leaves inside one chunk: about a gap between
/// words after a cut between words, and less than its own comma pause after
/// `,` `;` `:`. Sentence ends keep the model's own pause. The utterance's
/// own lead-in, before the first piece, is cut to about 100 ms.
enum KokoroChunkSeams {

    /// What ends the text before a seam.
    enum Kind: Equatable {
        case sentence
        case clause
        case word
    }

    /// Silence kept at a cut between words at speed 1, both sides together.
    /// Inside one chunk most word boundaries have no silence at all and the
    /// rest about 15-45 ms.
    static let wordGapSeconds = 0.020
    /// Silence kept at a cut after `,` `;` `:` at speed 1, both sides
    /// together. It is kept shorter than Kokoro's own comma pause inside one
    /// chunk, measured at 210-275 ms at speed 1 (165-210 ms at speed 1.2).
    static let clauseGapSeconds = 0.160
    /// Lead-in silence kept before the first piece's speech at speed 1. With
    /// BOS/EOS 0 the start token's lead-in grows to about 450 ms on a short
    /// reply; cutting it keeps the model's onset without the wait. With BOS
    /// 1 the lead-in is already about 75-125 ms, so this rarely cuts.
    static let leadInSeconds = 0.100
    /// A 10 ms window below this RMS is silence. The graph's lead-in sits
    /// under 0.001; soft onsets and endings (h, f, s, th) rise above this
    /// within a window or two, and the kept silence covers the rest.
    static let silenceRMS: Float = 0.002
    /// RMS window and hop: 10 ms and 5 ms at 24 kHz.
    static let windowSamples = 240
    static let hopSamples = 120
    /// Linear fade over the end of a piece whose trailing silence was cut.
    static let fadeOutSamples = 120

    /// Sentence enders and clause marks, including the full-width forms.
    private static let sentenceEnders: Set<Character> = [".", "!", "?", "…", "。", "！", "？"]
    private static let clauseMarks: Set<Character> = [",", ";", ":", "，", "；", "：", "、"]
    private static let closers: Set<Character> = ["\"", "'", ")", "]", "}", "”", "’", "」", "』", "）"]

    /// The kind of seam that follows `text`: a sentence end, a clause mark,
    /// or a cut between words (also a cut inside a word, or a line break
    /// without punctuation). Closing quotes and brackets are skipped.
    static func kind(after text: String) -> Kind {
        guard let last = text.reversed().first(where: { !$0.isWhitespace && !closers.contains($0) }) else {
            return .sentence
        }
        if sentenceEnders.contains(last) { return .sentence }
        if clauseMarks.contains(last) { return .clause }
        return .word
    }

    /// Samples of silence each side of a seam keeps when it follows `text`,
    /// or nil after a sentence end, which keeps the model's own pause.
    /// Faster speech keeps proportionally less, as Kokoro's own pauses do.
    static func keptSilence(after text: String, speed: Float, sampleRate: Int) -> Int? {
        let gap: Double
        switch kind(after: text) {
        case .sentence: return nil
        case .clause: gap = clauseGapSeconds
        case .word: gap = wordGapSeconds
        }
        let rate = Double(max(speed, 0.1))
        return Int((gap / 2 / rate * Double(sampleRate)).rounded())
    }

    /// Samples of silence one rendered piece keeps at each end. `leading`
    /// follows from the text of the piece before it; for the first piece,
    /// whose lead-in starts the utterance, it is `leadInSeconds`. `trailing`
    /// follows from the piece's own text and is nil for the final piece,
    /// whose tail ends it. Either is nil after a sentence end. Pieces of an
    /// overflow retry are handled like any other piece, by their own text.
    /// Nil leaves that end as rendered. Faster speech keeps proportionally
    /// less.
    static func keeps(
        previousText: String?,
        text: String,
        isFinal: Bool,
        speed: Float,
        sampleRate: Int
    ) -> (leading: Int?, trailing: Int?) {
        let leading: Int?
        if let previousText {
            leading = keptSilence(after: previousText, speed: speed, sampleRate: sampleRate)
        } else {
            leading = Int((leadInSeconds / Double(max(speed, 0.1)) * Double(sampleRate)).rounded())
        }
        let trailing = isFinal ? nil : keptSilence(after: text, speed: speed, sampleRate: sampleRate)
        return (leading, trailing)
    }

    /// Cuts the silence before the first and after the last window at or
    /// above `threshold` down to `keepLeading` and `keepTrailing` samples;
    /// nil leaves that end as it is. Nothing between those windows is
    /// touched, so soft speech is never removed. A cut end gets a short
    /// fade-out. Audio with no window above the threshold is returned as is.
    static func trim(
        _ audio: [Float],
        keepLeading: Int?,
        keepTrailing: Int?,
        threshold: Float = silenceRMS,
        window: Int = windowSamples,
        hop: Int = hopSamples
    ) -> [Float] {
        guard keepLeading != nil || keepTrailing != nil,
              window > 0, hop > 0, audio.count >= window else { return audio }
        let limit = threshold * threshold * Float(window)
        func isSound(_ start: Int) -> Bool {
            var sum: Float = 0
            for i in start..<(start + window) { sum += audio[i] * audio[i] }
            return sum >= limit
        }
        var onset: Int?
        var start = 0
        while start + window <= audio.count {
            if isSound(start) {
                onset = start
                break
            }
            start += hop
        }
        guard let onset else { return audio }
        var offset = onset + window
        var end = audio.count - window
        while end > onset {
            if isSound(end) {
                offset = end + window
                break
            }
            end -= hop
        }

        let from = keepLeading.map { max(0, onset - max(0, $0)) } ?? 0
        let to = keepTrailing.map { min(audio.count, offset + max(0, $0)) } ?? audio.count
        guard from > 0 || to < audio.count else { return audio }
        var out = Array(audio[from..<to])
        if to < audio.count, let keep = keepTrailing {
            let n = min(fadeOutSamples, keep, out.count)
            if n >= 2 {
                let first = out.count - n
                for k in 0..<n { out[first + k] *= Float(n - 1 - k) / Float(n - 1) }
            }
        }
        return out
    }
}
