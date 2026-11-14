import Foundation

/// Silences the faint noise burst some fp16 end-to-end graphs emit at the
/// very start of a pass.
///
/// An fp16 graph on the CPU can open its lead-in with a short burst of low
/// noise, followed by digital silence (under 16-bit resolution, often exact
/// zeros), then the speech. The masked 5 s fp16 export's burst is about
/// 130 ms at -55 to -59 dBFS RMS, and the 10 s fp16 export's about 27 ms at
/// -50 to -53 dBFS RMS, which is heard as a click or static at the start of
/// every piece. No fp32 graph has a sound-then-silence pattern like this in
/// its lead-in, so this signature (a quiet burst, then at least 20 ms of
/// digital silence, within the first 200 ms) leaves them, and any real
/// speech, untouched.
enum KokoroLeadInBurst {

    /// The burst must end within this time of the start of the pass.
    static let maxBurstSeconds = 0.200
    /// Digital silence that must follow the burst.
    static let minSilentRunSeconds = 0.020
    /// A sample under this magnitude is digital silence: below 16-bit
    /// resolution, about -90 dBFS.
    static let silentLevel: Float = 1.0 / 32768
    /// The burst must be quieter than this RMS (-40 dBFS); speech is far louder.
    static let maxBurstRMS: Float = 0.01

    /// Zeroes the samples before the first run of digital silence when that
    /// run starts within `maxBurstSeconds`, lasts at least
    /// `minSilentRunSeconds`, and what precedes it is below `maxBurstRMS`.
    /// Returns whether it did.
    @discardableResult
    static func silence(_ audio: inout [Float], sampleRate: Int) -> Bool {
        let maxBurst = Int(maxBurstSeconds * Double(sampleRate))
        let minRun = max(1, Int(minSilentRunSeconds * Double(sampleRate)))
        var runLength = 0
        var runStart: Int?
        for i in 0..<min(audio.count, maxBurst + minRun) {
            if abs(audio[i]) < silentLevel {
                runLength += 1
                if runLength == minRun {
                    runStart = i - minRun + 1
                    break
                }
            } else {
                runLength = 0
            }
        }
        guard let burstEnd = runStart, burstEnd > 0 else { return false }
        var sumSq: Float = 0
        for i in 0..<burstEnd { sumSq += audio[i] * audio[i] }
        guard (sumSq / Float(burstEnd)).squareRoot() < maxBurstRMS else { return false }
        for i in 0..<burstEnd { audio[i] = 0 }
        return true
    }
}
