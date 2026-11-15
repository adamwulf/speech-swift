import Foundation

/// Zeroes the artifacts an end-to-end graph can leave after the speech of a
/// pass, without cutting a soft final consonant.
///
/// Kokoro's E2E graphs can emit 100-300 ms of trailing artifacts past the
/// real speech: random low-energy noise and an occasional 10-20 ms spike
/// with little energy around it, heard as a click on iPhone playback. The
/// speech is found by walking back from the end of the pass in 50 ms
/// windows, with 50 % overlap, to the last one above 0.015 RMS. A lone
/// spike rarely lifts a whole 50 ms window that high, so the walk passes
/// it. Real speech tails stay above about 0.03 RMS over 50 ms; soft word
/// endings (German "Tisch", "Hamburg") fall under that, which set the floor
/// at 0.015.
///
/// A final fricative can still sit under 0.015. The masked 5 s fp16 export
/// renders the closing /s/ of "the Heart voice." at 0.009-0.014 RMS. At
/// speed 1.15 all of it came after the last 50 ms window above the floor
/// and was zeroed, so the reply ended on "voi". The kept end therefore runs
/// on past that window, 10 ms at a time, while the sound continues at or
/// above the seam trim's silence floor (`KokoroChunkSeams.silenceRMS`), for
/// at most 250 ms at speed 1. An artifact after a silent window is still
/// zeroed.
enum KokoroTrailingArtifacts {

    /// Window and floor of the walk back to the last speech.
    static let anchorWindowSeconds = 0.050
    static let anchorRMS: Float = 0.015
    /// Window of the run-on past that speech, and its longest reach at
    /// speed 1. Faster speech reaches proportionally less far.
    static let tailWindowSeconds = 0.010
    static let maxTailSeconds = 0.250
    /// Linear fade over the last of the kept sound.
    static let fadeSeconds = 0.010

    /// Where the speech of `audio` ends: the end of the last 50 ms window
    /// above `anchorRMS`, run on through the sound that continues straight
    /// from it. `audio.count` when no window is above `anchorRMS`.
    static func speechEnd(in audio: [Float], speed: Float, sampleRate: Int) -> Int {
        func rms(_ start: Int, _ length: Int) -> Float {
            var sumSq: Float = 0
            for i in start..<(start + length) { sumSq += audio[i] * audio[i] }
            return (sumSq / Float(length)).squareRoot()
        }
        let window = max(1, Int(anchorWindowSeconds * Double(sampleRate)))
        var anchor: Int?
        var start = audio.count - window
        while start > 0 {
            if rms(start, window) > anchorRMS {
                anchor = start + window
                break
            }
            start -= max(1, window / 2)
        }
        guard let anchor else { return audio.count }

        let tail = max(1, Int(tailWindowSeconds * Double(sampleRate)))
        let reach = Int((maxTailSeconds / Double(max(speed, 0.1)) * Double(sampleRate)).rounded())
        let limit = min(audio.count, anchor + reach)
        var end = anchor
        while end + tail <= limit, rms(end, tail) >= KokoroChunkSeams.silenceRMS {
            end += tail
        }
        return end
    }

    /// Zeroes everything after `speechEnd(in:speed:sampleRate:)` and fades
    /// out the last 10 ms before it, so the cut does not click. Returns the
    /// speech end.
    @discardableResult
    static func clear(_ audio: inout [Float], speed: Float, sampleRate: Int) -> Int {
        let end = speechEnd(in: audio, speed: speed, sampleRate: sampleRate)
        for k in end..<audio.count { audio[k] = 0 }
        let fade = min(end, Int(fadeSeconds * Double(sampleRate)))
        if fade >= 2 {
            let first = end - fade
            let denom = Float(fade - 1)
            for k in 0..<fade { audio[first + k] *= Float(fade - 1 - k) / denom }
        }
        return end
    }
}
