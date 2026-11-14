import Foundation

/// Splits text into pieces the fixed-capacity Kokoro graph can render.
///
/// Sentences and clauses are cut as speech-core's `chunk_text_for_synthesis`
/// and `split_text_for_synthesis_retry` cut them: sentences are packed
/// greedily up to a token budget, a sentence too long for one piece splits
/// at clause boundaries (`,` `;` `:`), and a single word too long for one
/// piece splits between characters. Punctuation is a boundary only when
/// whitespace or the end of the text follows it, so "3.5", "1.000", "z.B."
/// or "18:30" reach the phonemizer intact.
///
/// Two rules intentionally diverge from speech-core:
///
/// - Cuts between words. speech-core fills each piece to the budget and so
///   cuts wherever the budget runs out, often leaving a stub of a word or
///   two. Here a sentence or clause that has to be cut between words is
///   split into the fewest pieces the budget allows, of about even length;
///   among nearly balanced cuts, one just before a word that starts a phrase
///   ("and", "because", "where", "to", "with", ...) or just after clause or
///   sentence punctuation wins. A sentence or clause below the minimum size
///   that comes right before one too long for a piece is cut together with
///   it rather than left on its own. When the token estimate behind this
///   search cannot settle on pieces that fit, the words are packed greedily
///   as speech-core packs them.
/// - Short pieces. speech-core only merges a short last piece into the one
///   before it. Here any piece below the minimum size joins a neighbour
///   within the budget, one in the same sentence first.
///
/// Sentence and clause boundaries still outrank any word cut.
enum KokoroTextChunker {

    /// Packs `text` into chunks of at most `maxTokens`. A chunk below
    /// `minTailTokens` joins a neighbour when the result stays within
    /// `maxTokens`; the last chunk may join its predecessor up to
    /// `hardCapTokens`, the model's absolute input capacity.
    /// Whitespace-only text yields no chunks.
    static func chunk(
        _ text: String,
        countTokens: (String) -> Int,
        maxTokens: Int,
        hardCapTokens: Int,
        minTailTokens: Int
    ) -> [String] {
        guard maxTokens > 0 else { return [] }
        var chunks: [String] = []
        pack(splitUnits(text, isEnd: isSentenceEnd), level: .sentences, countTokens: countTokens,
             maxTokens: maxTokens, minTokens: minTailTokens, into: &chunks)
        return mergeShortChunks(chunks, countTokens: countTokens, maxTokens: maxTokens,
                                hardCapTokens: hardCapTokens, minTokens: minTailTokens)
    }

    /// Joins each chunk below `minTokens` to a neighbour while the joined
    /// chunk stays within `maxTokens` (`hardCapTokens` when the short chunk
    /// is the last one and joins its predecessor). A neighbour in the same
    /// sentence comes first, the previous one before the next; a join across
    /// a sentence end, where a seam pauses naturally, is the last resort.
    static func mergeShortChunks(
        _ chunks: [String],
        countTokens: (String) -> Int,
        maxTokens: Int,
        hardCapTokens: Int,
        minTokens: Int
    ) -> [String] {
        var chunks = chunks
        var i = 0
        while chunks.count >= 2, i < chunks.count {
            guard countTokens(chunks[i]) < minTokens else {
                i += 1
                continue
            }
            let isLast = i == chunks.count - 1
            let hasPrevious = i > 0
            let previousInSentence = hasPrevious && !endsSentence(chunks[i - 1])
            let nextInSentence = !isLast && !endsSentence(chunks[i])
            // true joins the previous chunk, false the next one.
            var order: [Bool] = []
            if previousInSentence { order.append(true) }
            if nextInSentence { order.append(false) }
            if hasPrevious && !previousInSentence { order.append(true) }
            if !isLast && !nextInSentence { order.append(false) }

            var joined = false
            for joinPrevious in order {
                if joinPrevious {
                    let merged = chunks[i - 1] + " " + chunks[i]
                    if countTokens(merged) <= (isLast ? hardCapTokens : maxTokens) {
                        chunks[i - 1] = merged
                        chunks.remove(at: i)
                        // Check the joined chunk again.
                        i -= 1
                        joined = true
                        break
                    }
                } else {
                    let merged = chunks[i] + " " + chunks[i + 1]
                    if countTokens(merged) <= maxTokens {
                        chunks[i] = merged
                        chunks.remove(at: i + 1)
                        joined = true
                        break
                    }
                }
            }
            if !joined { i += 1 }
        }
        return chunks
    }

    private static func endsSentence(_ text: String) -> Bool {
        KokoroChunkSeams.kind(after: text) == .sentence
    }

    /// Two strictly smaller pieces of a chunk the model could not render
    /// safely, or `[]` when no cut keeps both within `minTokens...maxTokens`.
    /// The boundary class (sentence, clause, word, character) outranks
    /// balance; a cut inside a punctuation run or before a closing quote is
    /// never taken.
    static func splitForRetry(
        _ text: String,
        countTokens: (String) -> Int,
        minTokens: Int,
        maxTokens: Int
    ) -> [String] {
        guard minTokens > 0, minTokens <= maxTokens else { return [] }
        let chars = Array(text)
        guard chars.count >= 2 else { return [] }
        // Every sentence, clause and word boundary touches whitespace, and any
        // of them beats a character cut, so character cuts are only tried
        // when no boundary works. This keeps the phonemizer calls per retry
        // proportional to the number of words.
        let boundaries = (1..<chars.count).filter {
            chars[$0 - 1].isWhitespace || chars[$0].isWhitespace
        }
        if let pieces = bestSplit(chars, cuts: boundaries, countTokens: countTokens,
                                  minTokens: minTokens, maxTokens: maxTokens) {
            return pieces
        }
        return bestSplit(chars, cuts: Array(1..<chars.count), countTokens: countTokens,
                         minTokens: minTokens, maxTokens: maxTokens) ?? []
    }

    // MARK: - Boundaries

    static func isSentenceEnd(_ c: Character) -> Bool { c == "." || c == "!" || c == "?" }
    static func isClauseEnd(_ c: Character) -> Bool { c == "," || c == ";" || c == ":" }
    private static func isQuote(_ c: Character) -> Bool { c == "\"" || c == "'" }

    /// Units ending at `isEnd` punctuation that whitespace or the end of the
    /// text follows; newlines always end a unit. A run of enders and closing
    /// quotes ("...", `?!"`) stays with its unit.
    static func splitUnits(_ text: String, isEnd: (Character) -> Bool) -> [String] {
        var units: [String] = []
        var current = ""
        func flush() {
            let unit = current.trimmingCharacters(in: .whitespacesAndNewlines)
            if !unit.isEmpty { units.append(unit) }
            current = ""
        }
        let chars = Array(text)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if c.isNewline {
                flush()
                i += 1
                continue
            }
            current.append(c)
            if isEnd(c) {
                while i + 1 < chars.count, isEnd(chars[i + 1]) || isQuote(chars[i + 1]) {
                    i += 1
                    current.append(chars[i])
                }
                if i + 1 == chars.count || chars[i + 1].isWhitespace { flush() }
            }
            i += 1
        }
        flush()
        return units
    }

    static func splitWords(_ text: String) -> [String] {
        text.split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
    }

    /// Last resort for one word the budget cannot hold: cut between
    /// characters, never inside one.
    static func splitCharacters(_ word: String, countTokens: (String) -> Int, maxTokens: Int) -> [String] {
        var parts: [String] = []
        var current = ""
        for c in word {
            let candidate = current + String(c)
            if !current.isEmpty && countTokens(candidate) > maxTokens {
                parts.append(current)
                current = String(c)
            } else {
                current = candidate
            }
        }
        if !current.isEmpty { parts.append(current) }
        return parts
    }

    private enum Level { case sentences, clauses, words }

    /// Greedily packs `units` into chunks of at most `maxTokens`; a unit too
    /// large on its own is refined: a sentence into clauses, a clause (or a
    /// sentence without clause punctuation) into balanced runs of words, a
    /// word into characters. Sentences or clauses packed so far that stay
    /// below `minTokens` are refined together with a too-large unit after
    /// them instead of becoming a chunk of their own.
    private static func pack(
        _ units: [String],
        level: Level,
        countTokens: (String) -> Int,
        maxTokens: Int,
        minTokens: Int,
        into out: inout [String]
    ) {
        var acc = ""
        for unit in units {
            let candidate = acc.isEmpty ? unit : acc + " " + unit
            if countTokens(candidate) <= maxTokens {
                acc = candidate
                continue
            }
            var unit = unit
            if !acc.isEmpty {
                if level != .words, countTokens(acc) < minTokens, countTokens(unit) > maxTokens {
                    unit = candidate
                } else {
                    out.append(acc)
                }
                acc = ""
            }
            if countTokens(unit) <= maxTokens {
                acc = unit
                continue
            }
            switch level {
            case .sentences:
                let clauses = splitUnits(unit, isEnd: isClauseEnd)
                if clauses.count > 1 {
                    pack(clauses, level: .clauses, countTokens: countTokens,
                         maxTokens: maxTokens, minTokens: minTokens, into: &out)
                } else {
                    splitBetweenWords(unit, countTokens: countTokens, maxTokens: maxTokens,
                                      minTokens: minTokens, into: &out)
                }
            case .clauses:
                splitBetweenWords(unit, countTokens: countTokens, maxTokens: maxTokens,
                                  minTokens: minTokens, into: &out)
            case .words:
                out.append(contentsOf: splitCharacters(unit, countTokens: countTokens, maxTokens: maxTokens))
            }
        }
        if !acc.isEmpty { out.append(acc) }
    }

    /// Cuts a unit too long for one chunk between words, as evenly as the
    /// words allow. When `balancedWordSplit` finds no cut (a word too long
    /// for any chunk, or recounts that never settle), the words are packed
    /// greedily instead, a word too long for any chunk cut between characters.
    private static func splitBetweenWords(
        _ unit: String,
        countTokens: (String) -> Int,
        maxTokens: Int,
        minTokens: Int,
        into out: inout [String]
    ) {
        let words = splitWords(unit)
        if let pieces = balancedWordSplit(words, countTokens: countTokens,
                                          maxTokens: maxTokens, minTokens: minTokens) {
            out.append(contentsOf: pieces)
        } else {
            pack(words, level: .words, countTokens: countTokens,
                 maxTokens: maxTokens, minTokens: minTokens, into: &out)
        }
    }

    // MARK: - Balanced word cuts

    /// Words that start a phrase, so a cut just before one falls where a
    /// speaker could pause: conjunctions, relative words and common
    /// prepositions. English only; other languages cut for balance alone.
    static let phraseStarters: Set<String> = [
        "and", "or", "but", "so", "because", "although", "though", "while",
        "unless", "until", "since", "if", "whether",
        "where", "when", "which", "that", "who", "whom", "whose", "what", "why", "how",
        "to", "for", "with", "about", "from", "into", "in", "on", "at", "by",
        "through", "after", "before", "without", "during",
    ]

    /// Cost of a cut that is neither just before a phrase starter nor just
    /// after clause or sentence punctuation, in the units of the balance cost
    /// (squared tokens per piece). Moving one cut `d` tokens off the most
    /// even position costs about `2 d^2`, so a free cut wins while it sits
    /// within about 9 tokens, one or two words, of the most even cut.
    static let nonPhraseCutCost = 160

    /// How often the search may re-run after an exact count shows that an
    /// estimated piece breaks a size limit.
    private static let maxRecounts = 8

    static func startsPhrase(_ word: String) -> Bool {
        let bare = word.drop(while: { !$0.isLetter }).lowercased()
        return phraseStarters.contains(bare)
    }

    /// Whether `word` ends in clause or sentence punctuation, closing quotes
    /// and brackets aside: a cut right after it keeps the text's own boundary.
    static func endsWithPunctuation(_ word: String) -> Bool {
        KokoroChunkSeams.kind(after: word) != .word
    }

    /// Cuts `words` into the fewest pieces of at most `maxTokens` and, where
    /// the words allow it, at least `minTokens`. Among those, the pieces are
    /// as even as possible, with a cut just before a phrase starter or just
    /// after clause or sentence punctuation preferred over a slightly more
    /// even one. Returns nil for fewer than two words, for a single word over
    /// `maxTokens`, and when the recounts run out before the exact counts
    /// confirm a cut; `splitBetweenWords` then packs the words greedily.
    static func balancedWordSplit(
        _ words: [String],
        countTokens: (String) -> Int,
        maxTokens: Int,
        minTokens: Int
    ) -> [String]? {
        let n = words.count
        guard n >= 2 else { return nil }
        // A run of words costs about the sum of its words plus a separator
        // token per space, so the search estimates runs from per-word counts
        // instead of phonemizing every candidate, then counts the pieces it
        // picks exactly. A piece the exact count rejects is remembered and
        // the search runs again.
        let overhead = countTokens("")
        let wordCounts = words.map(countTokens)
        guard wordCounts.allSatisfy({ $0 <= maxTokens }) else { return nil }
        var prefix = [0]
        for count in wordCounts { prefix.append(prefix[prefix.count - 1] + count - overhead) }
        // The separator is calibrated on the whole run, not on one pair: a
        // word alone can phonemize differently than in context ("A" alone is
        // two phonemes, one in "A few").
        let whole = countTokens(words.joined(separator: " "))
        let separator = max(0, Int((Double(whole - overhead - prefix[n]) / Double(n - 1)).rounded()))
        var exact: [Int: Int] = [:]
        let tokens: (Int, Int) -> Int = { i, j in
            exact[i * (n + 1) + j] ?? overhead + prefix[j] - prefix[i] + separator * (j - i - 1)
        }
        let cutCost: (Int) -> Int = {
            startsPhrase(words[$0]) || endsWithPunctuation(words[$0 - 1]) ? 0 : nonPhraseCutCost
        }

        for floor in [minTokens, 0] {
            for _ in 0..<maxRecounts {
                guard let starts = cheapestSplit(
                    count: n, tokens: tokens, cutCost: cutCost,
                    maxTokens: maxTokens, minTokens: floor) else { break }
                var pieces: [String] = []
                var fits = true
                for (start, end) in zip(starts, starts.dropFirst() + [n]) {
                    let piece = words[start..<end].joined(separator: " ")
                    let count = countTokens(piece)
                    exact[start * (n + 1) + end] = count
                    if count > maxTokens || count < floor { fits = false }
                    pieces.append(piece)
                }
                if fits { return pieces }
            }
        }
        return nil
    }

    /// The start index of each piece in the cheapest cut of `count` words:
    /// fewest pieces first, then the lowest sum of squared piece sizes plus
    /// `cutCost` for each cut. Nil when no cut keeps every piece within
    /// `minTokens...maxTokens`.
    private static func cheapestSplit(
        count n: Int,
        tokens: (Int, Int) -> Int,
        cutCost: (Int) -> Int,
        maxTokens: Int,
        minTokens: Int
    ) -> [Int]? {
        // best[j]: (pieces, cost) of the cheapest cut of words[0..<j].
        var pieces = [Int](repeating: .max, count: n + 1)
        var cost = [Int](repeating: .max, count: n + 1)
        var from = [Int](repeating: -1, count: n + 1)
        pieces[0] = 0
        cost[0] = 0
        for j in 1...n {
            for i in 0..<j where pieces[i] != .max {
                let size = tokens(i, j)
                guard size <= maxTokens, size >= minTokens else { continue }
                let p = pieces[i] + 1
                let c = cost[i] + size * size + (i > 0 ? cutCost(i) : 0)
                if p < pieces[j] || (p == pieces[j] && c < cost[j]) {
                    pieces[j] = p
                    cost[j] = c
                    from[j] = i
                }
            }
        }
        guard pieces[n] != .max else { return nil }
        var starts: [Int] = []
        var j = n
        while j > 0 {
            starts.append(from[j])
            j = from[j]
        }
        return starts.reversed()
    }

    // MARK: - Retry split

    private static func bestSplit(
        _ chars: [Character],
        cuts: [Int],
        countTokens: (String) -> Int,
        minTokens: Int,
        maxTokens: Int
    ) -> [String]? {
        var best: [String]?
        var bestScore = (Int.max, Int.max, Int.max)
        for cut in cuts {
            guard let rank = boundaryRank(chars, cut: cut) else { continue }
            let left = String(chars[..<cut]).trimmingCharacters(in: .whitespacesAndNewlines)
            let right = String(chars[cut...]).trimmingCharacters(in: .whitespacesAndNewlines)
            if left.isEmpty || right.isEmpty { continue }
            let leftTokens = countTokens(left)
            let rightTokens = countTokens(right)
            if leftTokens < minTokens || rightTokens < minTokens ||
                leftTokens > maxTokens || rightTokens > maxTokens {
                continue
            }
            let score = (rank, max(leftTokens, rightTokens), abs(leftTokens - rightTokens))
            if score < bestScore {
                bestScore = score
                best = [left, right]
            }
        }
        return best
    }

    /// 0 sentence, 1 clause, 2 word, 3 character; `nil` for a cut that would
    /// leave part of a punctuation or closing-quote run on the right.
    private static func boundaryRank(_ chars: [Character], cut: Int) -> Int? {
        let before = chars[cut - 1]
        let after = chars[cut]
        var sawSentenceEnd = false
        var sawClauseEnd = false
        var i = cut
        while i > 0 {
            let c = chars[i - 1]
            if isSentenceEnd(c) {
                sawSentenceEnd = true
            } else if isClauseEnd(c) {
                sawClauseEnd = true
            } else if !isQuote(c) {
                break
            }
            i -= 1
        }
        if sawSentenceEnd || sawClauseEnd {
            if isSentenceEnd(after) || isClauseEnd(after) || isQuote(after) { return nil }
            // "3.|5" or "18:|30" is a cut inside a token, not a boundary.
            if after.isWhitespace { return sawSentenceEnd ? 0 : 1 }
            return 3
        }
        return before.isWhitespace || after.isWhitespace ? 2 : 3
    }
}
