import Foundation

/// Silences the faint noise burst some fp16 end-to-end graphs emit at the
/// very start of a pass.
///
/// Run on the CPU, the masked 5 s fp16 export opens every pass with about
/// 12 ms of noise at -39 to -50 dBFS, and the masked 10 s fp16 export with
/// about 27 ms at -42 to -60 dBFS. fp32 graphs, the 15 s fp16 export and
/// fp16 on the Neural Engine start in silence (-85 dBFS or less). Before a
/// silent lead-in the burst is heard as static at the start of every piece;
/// before a short reply such as "Yes.", whose speech starts about 30 ms in,
/// it is a soft click on the onset.
///
/// Kokoro gives the start token at least one 25 ms frame, so the first
/// 25 ms of a pass is lead-in, never speech. When it is quieter than
/// speech, it is zeroed and the next 5 ms fade in. On a graph without the
/// burst this only clears what is already silence.
enum KokoroLeadInBurst {

    /// One Kokoro alignment frame: the start token's shortest duration.
    static let leadInSeconds = 0.025
    /// Linear fade-in after the cleared lead-in.
    static let fadeSeconds = 0.005
    /// The cleared lead-in must be quieter than this RMS (-30 dBFS); the
    /// burst measures -44 dBFS at most, and speech is far louder.
    static let maxBurstRMS: Float = 0.0316

    /// Zeroes the first `leadInSeconds` of a pass and fades in the next
    /// `fadeSeconds` when that lead-in is below `maxBurstRMS`. Returns
    /// whether it did.
    @discardableResult
    static func silence(_ audio: inout [Float], sampleRate: Int) -> Bool {
        let lead = Int(leadInSeconds * Double(sampleRate))
        let fade = Int(fadeSeconds * Double(sampleRate))
        guard lead > 0, fade >= 2, audio.count >= lead + fade else { return false }
        var sumSq: Float = 0
        for i in 0..<lead { sumSq += audio[i] * audio[i] }
        guard (sumSq / Float(lead)).squareRoot() < maxBurstRMS else { return false }
        for i in 0..<lead { audio[i] = 0 }
        for k in 0..<fade { audio[lead + k] *= Float(k) / Float(fade) }
        return true
    }
}
