import Foundation

/// One voice preset as read from `voices/<id>.json`.
///
/// Every voice file has an `embedding` key: one 256-float style vector, the
/// mean of the voice's per-length rows. Some voice files also carry those
/// rows, under the keys `"1"` … `"510"`: upstream Kokoro's voice pack, where
/// the row for `n` phonemes is the style for an utterance of that length
/// (upstream picks `pack[len(phonemes) - 1]`). When the rows are present,
/// each chunk renders with the row for its own length; otherwise every chunk
/// uses `embedding`.
struct KokoroVoice {
    /// The `embedding` vector: the style used when the file has no rows.
    let embedding: [Float]
    /// The per-length rows, row-major, `rowCount × styleDim`; empty when the
    /// file has none.
    let rows: [Float]
    let rowCount: Int
    let styleDim: Int

    init(embedding: [Float], rows: [Float] = [], rowCount: Int = 0, styleDim: Int) {
        self.embedding = embedding
        self.rows = rows
        self.rowCount = rowCount
        self.styleDim = styleDim
    }

    /// The style vector for a chunk of `phonemeCount` phonemes, BOS and EOS
    /// not counted.
    func style(phonemeCount: Int) -> [Float] {
        guard rowCount > 0 else { return embedding }
        let start = Self.rowIndex(phonemeCount: phonemeCount, rowCount: rowCount) * styleDim
        return Array(rows[start..<(start + styleDim)])
    }

    /// Row for `phonemeCount` phonemes: `phonemeCount - 1`, clamped to the
    /// rows there are.
    static func rowIndex(phonemeCount: Int, rowCount: Int) -> Int {
        min(max(phonemeCount - 1, 0), rowCount - 1)
    }

    /// Reads a voice file's JSON object. Returns nil without a usable
    /// `embedding`. Rows are kept only when `"1"`, `"2"`, … run without a
    /// gap and each holds `styleDim` numbers; anything else falls back to
    /// `embedding` alone.
    static func parse(_ json: [String: Any], styleDim: Int) -> KokoroVoice? {
        guard let embedding = json["embedding"] as? [NSNumber], embedding.count >= styleDim else {
            return nil
        }
        let base = embedding.prefix(styleDim).map(\.floatValue)

        var rows: [Float] = []
        var rowCount = 0
        while let row = json[String(rowCount + 1)] {
            guard let values = row as? [NSNumber], values.count == styleDim else {
                return KokoroVoice(embedding: base, styleDim: styleDim)
            }
            rows.append(contentsOf: values.lazy.map(\.floatValue))
            rowCount += 1
        }
        return KokoroVoice(embedding: base, rows: rows, rowCount: rowCount, styleDim: styleDim)
    }

    /// Loads `voices/<id>.json`.
    static func load(from url: URL, styleDim: Int) throws -> KokoroVoice? {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return parse(json, styleDim: styleDim)
    }

    /// Whether voice id `name` matches any of the shell patterns in `filter`
    /// (`a*`, `bf_emma`); a nil filter matches every voice.
    static func matches(_ name: String, filter: [String]?) -> Bool {
        guard let filter else { return true }
        return filter.contains { fnmatch($0, name, 0) == 0 }
    }
}
