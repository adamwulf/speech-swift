import Foundation

/// Configuration for Kokoro-82M TTS model.
public struct KokoroConfig: Codable, Sendable {
    /// Output audio sample rate in Hz.
    public let sampleRate: Int
    /// Maximum phoneme input length. `fromPretrained` sets it to the loaded
    /// E2E graph's `input_ids` slots: 128 for `kokoro_5s`.
    public let maxPhonemeLength: Int
    /// Style embedding dimension (ref_s input to CoreML model).
    public let styleDim: Int
    /// Supported languages.
    public let languages: [String]

    public init(
        sampleRate: Int = 24000,
        maxPhonemeLength: Int = 128,
        styleDim: Int = 256,
        languages: [String] = ["en", "fr", "es", "ja", "zh", "hi", "pt", "it", "de"]
    ) {
        self.sampleRate = sampleRate
        self.maxPhonemeLength = maxPhonemeLength
        self.styleDim = styleDim
        self.languages = languages
    }

    /// Default configuration matching Kokoro-82M.
    public static let `default` = KokoroConfig()
}
