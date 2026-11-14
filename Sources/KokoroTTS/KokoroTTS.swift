import CoreML
import Foundation
import AudioCommon

/// Kokoro-82M text-to-speech — CoreML-based, runs on Neural Engine.
///
/// Lightweight (82M params) non-autoregressive TTS model.
/// Supports 10 languages with 54 preset voices. Designed for iOS/iPad deployment.
///
/// German is served by separate single-voice bundles (the kikiri-tts
/// fine-tunes): load one with `fromPretrained(modelId: "aufklarer/Kokoro-82M-German-Martin-CoreML")`
/// (or `…-Victoria-CoreML`) and synthesize with `language: "de"`.
///
/// Uses an end-to-end CoreML model (`kokoro_5s.mlmodelc` by default) that
/// runs the full pipeline (BERT → duration → alignment → prosody → decoder)
/// in one call. Other end-to-end bundles with the same inputs load through
/// `fromPretrained(modelId:modelFile:)`; their input and output sizes are
/// read from the model.
///
/// ```swift
/// let tts = try await KokoroTTSModel.fromPretrained()
/// let audio = try tts.synthesize(text: "Hello world", voice: "af_heart")
/// ```
public final class KokoroTTSModel {

    /// Default HuggingFace model ID.
    public static let defaultModelId = "aufklarer/Kokoro-82M-CoreML"
    /// German voice bundles: one Kokoro checkpoint per voice, each with its
    /// single `dm_martin` / `df_victoria` preset and `dict_de.json`.
    public static let germanMartinModelId = "aufklarer/Kokoro-82M-German-Martin-CoreML"
    public static let germanVictoriaModelId = "aufklarer/Kokoro-82M-German-Victoria-CoreML"
    /// Default end-to-end bundle, `kokoro_5s.mlmodelc`: 128 phoneme slots,
    /// a 5 s output window, and a `speed` input.
    public static let defaultModelFile = "kokoro_5s"
    /// Default voice preset.
    public static let defaultVoice = "af_heart"
    /// Output sample rate.
    public static let outputSampleRate = 24000

    let config: KokoroConfig
    var network: KokoroNetwork?
    let phonemizer: KokoroPhonemizer
    var voices: [String: KokoroVoice]
    /// Chunk sizes for the loaded graph's input slots and output window.
    let chunkBudget: KokoroChunkBudget
    /// Listening tests only: send this id in place of BOS and EOS (upstream
    /// Kokoro uses 0 for both; the phonemizer emits 1 and 2).
    var experimentBoundaryId: Int?
    /// Listening tests only: use a voice's averaged `embedding` even when
    /// its file has per-length rows.
    var experimentAveragedVoice = false

    var _isLoaded: Bool { network != nil }

    init(config: KokoroConfig, network: KokoroNetwork, phonemizer: KokoroPhonemizer, voices: [String: KokoroVoice]) {
        self.config = config
        self.network = network
        self.phonemizer = phonemizer
        self.voices = voices
        self.chunkBudget = .forGraph(
            inputSlots: min(config.maxPhonemeLength, network.inputSlots),
            outputSamples: network.outputSamples)
    }

    // MARK: - Synthesis

    /// A chunk shorter than this joins a neighbour when the budget allows it
    /// (the last chunk joins its predecessor up to the hard cap), and no cut
    /// between words makes one where the words allow it; very short
    /// utterances render unreliably.
    static let minTailTokens = 12
    /// Pieces of an overflow retry never go below this many tokens.
    static let minRetryTokens = 6
    /// How many times one chunk may be split again after an overflow.
    static let maxRetryDepth = 4

    /// Synthesize speech from text.
    ///
    /// The text is split into chunks of what the loaded graph's output window
    /// holds at the models' speech rate (56 phoneme tokens for the 5 s graph),
    /// preferring sentence ends, then clauses, then evenly sized runs of
    /// words. Each chunk renders in one call and the audio is concatenated,
    /// so short text is a single call. Where a chunk ends inside a sentence,
    /// the silence at the seam is cut to about a gap between words, or after
    /// clause punctuation to less than Kokoro's own comma pause. A chunk
    /// whose speech would still overrun the graph's output window is split
    /// again and rendered as two pieces.
    ///
    /// `speed` needs a graph with a `speed` input, as `kokoro_5s` has; a
    /// graph without one renders at speed 1 and logs that `speed` was
    /// ignored.
    public func synthesize(
        text: String,
        voice: String = "af_heart",
        language: String = "en",
        speed: Float = 1.0
    ) throws -> [Float] {
        var audio: [Float] = []
        try synthesizeChunks(text: text, voice: voice, language: language, speed: speed) { samples, _ in
            audio.append(contentsOf: samples)
        }
        return audio
    }

    /// Renders `text` chunk by chunk and hands each piece's audio to
    /// `onChunk` as soon as it is ready. Exactly one call, the last, has
    /// `isFinal` set; text with nothing to speak yields one empty final piece.
    func synthesizeChunks(
        text: String,
        voice: String,
        language: String,
        speed: Float,
        onChunk: ([Float], Bool) throws -> Void
    ) throws {
        guard _isLoaded, let network else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro-synthesize", reason: "Model not loaded")
        }
        guard let preset = voices[voice] else {
            let available = Array(voices.keys).sorted().prefix(5)
            throw AudioModelError.voiceNotFound(
                voice: voice,
                searchPath: "Available: \(available.joined(separator: ", "))...")
        }
        // A graph without a speed input speaks at speed 1; the seams follow
        // the speed the audio actually has.
        if !network.acceptsSpeed && speed != 1 {
            AudioLog.inference.warning("Kokoro: this model has no speed input; rendering at speed 1 instead of \(speed)")
        }
        let speed = network.acceptsSpeed ? speed : 1

        // Packing measures many overlapping candidates and rendering needs the
        // ids of the pieces it kept; phonemize each piece of text once.
        var tokenCache: [String: [Int]] = [:]
        let tokenIds: (String) -> [Int] = { piece in
            if let ids = tokenCache[piece] { return ids }
            let ids = self.phonemizer.tokenize(piece, maxLength: .max, language: language)
            tokenCache[piece] = ids
            return ids
        }
        // Chunk up front, as speech-core does. Text within the budget is one
        // chunk and renders in one call; rendering longer text whole first
        // would only overrun the output window and cost a discarded pass.
        let chunks = Self.chunkPlan(text, countTokens: { tokenIds($0).count }, budget: chunkBudget)

        try Self.emitPieces(
            of: chunks, speed: speed, sampleRate: config.sampleRate,
            render: { chunk in
                try self.renderChunk(
                    chunk, network: network, voice: preset, speed: speed,
                    tokenIds: tokenIds, depth: 0)
            },
            onChunk: onChunk)
    }

    /// Renders `chunks` in order with `render` and hands each piece to
    /// `onChunk`, trimming the silence at its seams and fading in every piece
    /// after the first. `render` returns one chunk's pieces, several after an
    /// overflow retry, each with its own text. Exactly one call, the last, has
    /// `isFinal` set; no chunks yield one empty final piece.
    static func emitPieces(
        of chunks: [String],
        speed: Float,
        sampleRate: Int,
        render: (String) throws -> [(text: String, audio: [Float])],
        onChunk: ([Float], Bool) throws -> Void
    ) throws {
        guard !chunks.isEmpty else {
            try onChunk([], true)
            return
        }
        var previousText: String?
        for (index, chunk) in chunks.enumerated() {
            // An overflow retry can turn one chunk into several pieces; they
            // come back together, so the final piece is known when emitted.
            let pieces = try render(chunk)
            for (pieceIndex, piece) in pieces.enumerated() {
                let isFinal = index == chunks.count - 1 && pieceIndex == pieces.count - 1
                // A seam inside a sentence keeps about a gap between words, or
                // after clause punctuation less than Kokoro's own comma pause.
                // Both sides follow from the text before the seam, so a
                // piece's tail is settled before it is emitted.
                let keep = KokoroChunkSeams.keeps(
                    previousText: previousText, text: piece.text, isFinal: isFinal,
                    speed: speed, sampleRate: sampleRate)
                var audio = KokoroChunkSeams.trim(
                    piece.audio, keepLeading: keep.leading, keepTrailing: keep.trailing)
                // A short fade-in keeps the seam between pieces click-free.
                if previousText != nil { fadeIn(&audio, sampleRate: sampleRate) }
                try onChunk(audio, isFinal)
                previousText = piece.text
            }
        }
    }

    /// The chunks `text` renders as: packed to `budget.budget`, with a short
    /// chunk merged into a neighbour within the budget, or, for the last
    /// chunk, within `budget.hardCap`.
    static func chunkPlan(
        _ text: String,
        countTokens: (String) -> Int,
        budget: KokoroChunkBudget = .fiveSecond
    ) -> [String] {
        KokoroTextChunker.chunk(
            text, countTokens: countTokens, maxTokens: budget.budget,
            hardCapTokens: budget.hardCap, minTailTokens: minTailTokens)
    }

    /// Renders one chunk, splitting it again while its speech would overrun
    /// the graph's output window. Each piece comes back with its text, which
    /// decides how the seams around it are trimmed.
    private func renderChunk(
        _ text: String,
        network: KokoroNetwork,
        voice: KokoroVoice,
        speed: Float,
        tokenIds: (String) -> [Int],
        depth: Int
    ) throws -> [(text: String, audio: [Float])] {
        let ids = tokenIds(text)
        let capacity = min(config.maxPhonemeLength, network.inputSlots)
        var unsplit: [Float]?
        if ids.count <= capacity {
            let pass = try runModel(tokenIds: ids, network: network, voice: voice, speed: speed)
            if !pass.overran { return [(text, pass.audio)] }
            unsplit = pass.audio
        }
        if depth < Self.maxRetryDepth, ids.count > Self.minRetryTokens * 2 {
            let pieces = KokoroTextChunker.splitForRetry(
                text, countTokens: { tokenIds($0).count },
                minTokens: Self.minRetryTokens, maxTokens: ids.count - 1)
            if pieces.count == 2 {
                AudioLog.inference.info("Kokoro: splitting a \(ids.count)-token chunk that overran the output window")
                return try pieces.flatMap {
                    try renderChunk(
                        $0, network: network, voice: voice, speed: speed,
                        tokenIds: tokenIds, depth: depth + 1)
                }
            }
        }
        // No safe split exists: keep what one bounded pass produces.
        if let unsplit { return [(text, unsplit)] }
        let clipped = Array(ids.prefix(capacity - 1)) + [phonemizer.eosId]
        return [(text, try runModel(tokenIds: clipped, network: network, voice: voice, speed: speed).audio)]
    }

    /// One pass of the end-to-end graph. `overran` is set when the predicted
    /// speech reaches the untrusted tail of the output window.
    private func runModel(
        tokenIds: [Int],
        network: KokoroNetwork,
        voice: KokoroVoice,
        speed: Float
    ) throws -> (audio: [Float], overran: Bool) {
        var tokenIds = tokenIds
        if let boundary = experimentBoundaryId, tokenIds.count >= 2 {
            tokenIds[0] = boundary
            tokenIds[tokenIds.count - 1] = boundary
        }
        let padTo = network.inputSlots
        let tokenCount = min(tokenIds.count, padTo)
        let paddedIds = phonemizer.pad(Array(tokenIds.prefix(padTo)), to: padTo)

        let inputIds = try createInt32Array(shape: [1, padTo], values: paddedIds.map { Int32($0) })
        let maskArray = try createInt32Array(shape: [1, padTo], values: (0..<padTo).map { Int32($0 < tokenCount ? 1 : 0) })
        // BOS and EOS are not phonemes; the voice's per-length row, when it
        // has them, is picked by the phonemes between them.
        let style = experimentAveragedVoice ? voice.embedding : voice.style(phonemeCount: tokenCount - 2)
        let refS = try createFloatArray(shape: [1, config.styleDim], values: style)
        let speedArray = network.acceptsSpeed ? try createFloatArray(shape: [1], values: [speed]) : nil

        let t0 = CFAbsoluteTimeGetCurrent()
        let result = try network.predictE2E(inputIds: inputIds, attentionMask: maskArray, refS: refS, speed: speedArray)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        let outputCapacity = result.audio.count
        let tailGuard = KokoroChunkBudget.outputTailGuardSamples
        let overran = outputCapacity > tailGuard
            && result.audioLengthSamples > outputCapacity - tailGuard
        let validSamples = min(result.audioLengthSamples, outputCapacity)
        guard validSamples > 0 else { return ([], overran) }

        var audio: [Float]
        if result.audio.dataType == .float16 {
            audio = floatsFromFloat16(result.audio.dataPointer, count: validSamples)
        } else {
            audio = [Float](repeating: 0, count: validSamples)
            let ptr = result.audio.dataPointer.bindMemory(to: Float.self, capacity: validSamples)
            for i in 0..<validSamples { audio[i] = ptr[i] }
        }
        // fp16 graphs can open with a faint noise burst before their lead-in
        // silence, heard as a click at the start of every piece.
        KokoroLeadInBurst.silence(&audio, sampleRate: config.sampleRate)

        // Kokoro's E2E model often emits 100–300 ms of trailing artifacts
        // past the real speech (random low-energy noise + an occasional
        // loud-spike click, observed on iPhone playback). Find where real
        // speech actually ended by walking backwards through 10 ms windows
        // and locating the last window above a silence-energy threshold —
        // then zero everything past that, plus a short ramp-down on the
        // last few ms of real speech to avoid a click at the new boundary.
        // Use a 50 ms window with a higher silence floor and require the
        // window to be SUSTAINED above threshold — Kokoro's trailing
        // artifacts are often a single 10–20 ms spike with low surrounding
        // energy, so a tight short window otherwise mistakes them for
        // speech. Real speech tails consistently above ~0.03 RMS over a
        // 50 ms span; isolated artifact spikes don't. Soft word endings
        // (German "Tisch", "Hamburg") fall under 0.030, so the floor is 0.015;
        // lower values only keep more tail.
        let win = max(1, Int(0.050 * Double(config.sampleRate)))
        let silenceRMS: Float = 0.015
        var speechEnd = validSamples
        var i = validSamples - win
        while i > 0 {
            var sumSq: Float = 0
            for j in 0..<win { let v = audio[i + j]; sumSq += v * v }
            let rms = sqrt(sumSq / Float(win))
            if rms > silenceRMS { speechEnd = i + win; break }
            i -= win / 2  // 50 % overlap so we don't miss a window boundary
        }
        // Zero the trailing artifact region.
        if speechEnd < validSamples {
            for k in speechEnd..<validSamples { audio[k] = 0 }
        }
        // Linear fade-out on the last ~10 ms of the kept signal so the
        // boundary between speech and the silenced region is also smooth.
        let fadeSamples = min(speechEnd, Int(0.010 * Double(config.sampleRate)))
        if fadeSamples >= 2 {
            let start = speechEnd - fadeSamples
            let denom = Float(fadeSamples - 1)
            for k in 0..<fadeSamples {
                let gain = Float(fadeSamples - 1 - k) / denom
                audio[start + k] *= gain
            }
        }
        // The vocoder is not bounded to +-1: the packed-LSTM models occasionally
        // overshoot by a few percent. Clamp so callers get valid float PCM.
        for k in 0..<speechEnd { audio[k] = min(1, max(-1, audio[k])) }

        let duration = Double(validSamples) / Double(config.sampleRate)
        let elapsedMs = elapsed * 1000
        AudioLog.inference.info("Kokoro E2E: \(tokenCount) tokens → \(validSamples) samples (\(String(format: "%.1f", duration))s) in \(String(format: "%.0f", elapsedMs))ms")
        return (audio, overran)
    }

    /// 5 ms linear fade-in at the start of a piece that follows another.
    static func fadeIn(_ audio: inout [Float], sampleRate: Int) {
        let count = min(audio.count, sampleRate / 200)
        guard count >= 2 else { return }
        for i in 0..<count { audio[i] *= Float(i) / Float(count) }
    }

    /// List available voice presets.
    public var availableVoices: [String] {
        Array(voices.keys).sorted()
    }

    /// Registers custom `word -> IPA` pronunciations for proper nouns missing from
    /// the shipped dictionaries. Resolved ahead of the bundled dictionaries, suffix
    /// stemming, and the neural G2P fallback.
    ///
    /// Keys are matched lowercased and must be single words — a key containing a
    /// space, hyphen, or apostrophe is never looked up, because text is split on
    /// punctuation before resolution. The IPA must use symbols from the model
    /// vocabulary; anything else is dropped at tokenization.
    ///
    /// Entries apply to English only. Passing `language:` zh, ja, it, fr, es, pt,
    /// or hi to `synthesize` selects a phonemizer that never sees them.
    ///
    /// The lookup is unlocked, so register entries before `synthesize` starts
    /// running concurrently on this instance.
    public func addPronunciations(_ entries: [String: String]) {
        phonemizer.addPronunciations(entries)
    }

    // MARK: - Helpers

    private func createInt32Array(shape: [Int], values: [Int32]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { $0 as NSNumber }, dataType: .int32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<values.count { ptr[i] = values[i] }
        return arr
    }

    private func createFloatArray(shape: [Int], values: [Float]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { $0 as NSNumber }, dataType: .float32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<values.count { ptr[i] = values[i] }
        return arr
    }

    // MARK: - Warmup

    /// Warm up CoreML model by running a dummy inference.
    ///
    /// A model's first prediction is several times slower than the ones after
    /// it. `fromPretrained` calls this before returning unless `warmUp` is
    /// false; a model loaded that way can be warmed up here later.
    public func warmUp() throws {
        _ = try? synthesize(text: "hello", voice: defaultBundleVoice)
    }

    // MARK: - Model Loading

    /// Load a pretrained Kokoro model from HuggingFace.
    ///
    /// - Parameter modelFile: The end-to-end bundle to download and load,
    ///   without `.mlmodelc`. Defaults to `kokoro_5s`, the bundle every
    ///   `aufklarer/Kokoro-82M-*` repo ships. A repo with other end-to-end
    ///   bundles can name one here; its phoneme slots, output window and
    ///   whether it takes a `speed` input are read from the model.
    /// - Parameter voiceFilter: Shell patterns on voice ids (`a*`, `bf_emma`)
    ///   that limit which `voices/<id>.json` files are downloaded and loaded.
    ///   `nil`, the default, takes every voice in the repo.
    /// - Parameter computeUnits: Which hardware the main CoreML model runs on.
    ///   Defaults to `.all` (Neural Engine preferred). Pass `.cpuAndGPU` to bypass
    ///   the Neural Engine — useful as a fallback on platforms where the ANE
    ///   compiler produces incorrect output for this model.
    /// - Parameter warmUp: Run one short synthesis before returning, so the
    ///   first call is as fast as later ones. Pass `false` to load without it,
    ///   for example when the model is only being downloaded.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        modelFile: String = defaultModelFile,
        voice: String = defaultVoice,
        voiceFilter: [String]? = nil,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        computeUnits: MLComputeUnits = .all,
        downloadOptions: DownloadOptions = .init(),
        warmUp: Bool = true,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> KokoroTTSModel {
        AudioLog.modelLoading.info("Loading Kokoro model: \(modelId) (\(modelFile))")

        let cacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        // Download E2E model + G2P + voices. A listed file the repo does not
        // have (dict_de.json outside the German bundles) matches nothing and
        // is skipped.
        progressHandler?(0.0, "Downloading model...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: [
                "\(modelFile).mlmodelc/**",
                "G2PEncoder.mlmodelc/**",
                "G2PDecoder.mlmodelc/**",
                "vocab_index.json",
                "g2p_vocab.json",
                "us_gold.json",
                "us_silver.json",
                "dict_de.json",
            ] + (voiceFilter ?? ["*"]).map { "voices/\($0).json" },
            offlineMode: offlineMode,
            downloadOptions: downloadOptions
        ) { fraction in
            progressHandler?(fraction * 0.7, "Downloading model...")
        }

        // Load vocabulary
        progressHandler?(0.72, "Loading vocabulary...")
        let vocabURL = cacheDir.appendingPathComponent("vocab_index.json")
        guard FileManager.default.fileExists(atPath: vocabURL.path) else {
            throw AudioModelError.modelLoadFailed(modelId: modelId, reason: "vocab_index.json not found")
        }
        let phonemizer = try KokoroPhonemizer.loadVocab(from: vocabURL)
        try phonemizer.loadDictionaries(from: cacheDir)
        try phonemizer.loadGermanDictionary(from: cacheDir)

        // Load G2P models
        progressHandler?(0.76, "Loading G2P models...")
        let g2pEncoderURL = cacheDir.appendingPathComponent("G2PEncoder.mlmodelc", isDirectory: true)
        let g2pDecoderURL = cacheDir.appendingPathComponent("G2PDecoder.mlmodelc", isDirectory: true)
        let g2pVocabURL = cacheDir.appendingPathComponent("g2p_vocab.json")
        if FileManager.default.fileExists(atPath: g2pEncoderURL.path) &&
           FileManager.default.fileExists(atPath: g2pDecoderURL.path) {
            try phonemizer.loadG2PModels(
                encoderURL: g2pEncoderURL, decoderURL: g2pDecoderURL, vocabURL: g2pVocabURL)
            AudioLog.modelLoading.debug("Loaded CoreML G2P encoder + decoder")
        }

        // Load voice presets
        progressHandler?(0.78, "Loading voice embeddings...")
        var voices = [String: KokoroVoice]()
        let voicesDir = cacheDir.appendingPathComponent("voices")
        if FileManager.default.fileExists(atPath: voicesDir.path) {
            let files = try FileManager.default.contentsOfDirectory(at: voicesDir, includingPropertiesForKeys: nil)
            for file in files where file.pathExtension == "json" {
                let voiceName = file.deletingPathExtension().lastPathComponent
                guard KokoroVoice.matches(voiceName, filter: voiceFilter) else { continue }
                if let preset = try? KokoroVoice.load(from: file, styleDim: KokoroConfig.default.styleDim) {
                    voices[voiceName] = preset
                }
            }
            AudioLog.modelLoading.debug("Loaded \(voices.count) voice presets")
        }

        // Load E2E CoreML model
        progressHandler?(0.85, "Loading CoreML model...")
        let network = try KokoroNetwork(directory: cacheDir, modelFile: modelFile, computeUnits: computeUnits)
        AudioLog.modelLoading.debug("Loaded Kokoro E2E model \(modelFile): \(network.inputSlots) phoneme slots, \(network.outputSamples) output samples, speed input: \(network.acceptsSpeed)")

        let model = KokoroTTSModel(
            config: KokoroConfig(maxPhonemeLength: network.inputSlots), network: network,
            phonemizer: phonemizer, voices: voices)

        // The first prediction pays CoreML's one-time setup; take it here
        // rather than on the caller's first reply.
        if warmUp {
            progressHandler?(0.95, "Warming up...")
            try model.warmUp()
        }

        progressHandler?(1.0, "Model loaded")
        AudioLog.modelLoading.info("Kokoro model loaded successfully")
        return model
    }
}
