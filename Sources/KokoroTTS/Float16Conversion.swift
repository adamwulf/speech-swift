import Accelerate

/// Reads `count` contiguous IEEE 754 half-precision values starting at `source`
/// and returns them as `Float`s.
///
/// CoreML can return `.float16` multi-arrays, but Swift's `Float16` type is
/// unavailable on x86_64 macOS, so this conversion goes through vImage, which
/// is available on every architecture. Half to single precision is exact,
/// including zeros, subnormals, infinities, and NaNs.
func floatsFromFloat16(_ source: UnsafeRawPointer, count: Int) -> [Float] {
    guard count > 0 else { return [] }
    return [Float](unsafeUninitializedCapacity: count) { buffer, initializedCount in
        var src = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: source),
            height: 1,
            width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<UInt16>.stride)
        var dst = vImage_Buffer(
            data: buffer.baseAddress,
            height: 1,
            width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<Float>.stride)
        let error = vImageConvert_Planar16FtoPlanarF(&src, &dst, vImage_Flags(kvImageNoFlags))
        precondition(error == kvImageNoError, "vImage half-to-float conversion failed: \(error)")
        initializedCount = count
    }
}
