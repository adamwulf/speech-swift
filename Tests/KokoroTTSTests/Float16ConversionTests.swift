import XCTest
@testable import KokoroTTS

/// Covers `floatsFromFloat16`, which replaces Swift `Float16` (unavailable on
/// x86_64 macOS) when reading `.float16` CoreML outputs. The tests avoid
/// `Float16` so they also compile for x86_64.
final class Float16ConversionTests: XCTestCase {

    /// Decodes an IEEE 754 binary16 bit pattern without using `Float16`.
    private func referenceFloat(fromHalfBits bits: UInt16) -> Float {
        let sign: FloatingPointSign = (bits & 0x8000) != 0 ? .minus : .plus
        let exponent = Int((bits >> 10) & 0x1F)
        let mantissa = Float(bits & 0x03FF)
        switch exponent {
        case 0:
            // Zero or subnormal: mantissa * 2^-24.
            return Float(sign: sign, exponent: -24, significand: mantissa)
        case 0x1F:
            return mantissa == 0 ? (sign == .minus ? -.infinity : .infinity) : .nan
        default:
            return Float(sign: sign, exponent: exponent - 15, significand: 1 + mantissa / 1024)
        }
    }

    private func convert(_ bits: [UInt16]) -> [Float] {
        bits.withUnsafeBytes { floatsFromFloat16($0.baseAddress!, count: bits.count) }
    }

    func testKnownValues() {
        let cases: [(bits: UInt16, expected: Float)] = [
            (0x0000, 0),
            (0x3C00, 1),
            (0xC000, -2),
            (0x3800, 0.5),
            (0x3555, 0.333251953125),
            (0x7BFF, 65504),               // largest finite half
            (0x0400, 0x1p-14),             // smallest normal half
            (0x0001, 0x1p-24),             // smallest subnormal half
            (0x7C00, .infinity),
            (0xFC00, -.infinity),
        ]
        let result = convert(cases.map(\.bits))
        for (i, c) in cases.enumerated() {
            XCTAssertEqual(result[i], c.expected, "bits 0x\(String(c.bits, radix: 16))")
        }
    }

    func testNegativeZeroKeepsSign() {
        let result = convert([0x8000])
        XCTAssertEqual(result[0], 0)
        XCTAssertEqual(result[0].sign, .minus)
    }

    func testNaN() {
        XCTAssertTrue(convert([0x7E00])[0].isNaN)
        XCTAssertTrue(convert([0xFE01])[0].isNaN)
    }

    /// Every one of the 65,536 half bit patterns converts exactly, in one call,
    /// so vector bodies and scalar tails are both exercised.
    func testAllBitPatternsMatchReference() {
        let bits = Array(0...UInt16.max)
        let result = convert(bits)
        XCTAssertEqual(result.count, bits.count)
        var mismatches = 0
        for (i, b) in bits.enumerated() {
            let expected = referenceFloat(fromHalfBits: b)
            let actual = result[i]
            let matches = expected.isNaN ? actual.isNaN : actual.bitPattern == expected.bitPattern
            if !matches {
                mismatches += 1
                if mismatches <= 5 {
                    XCTFail("bits 0x\(String(b, radix: 16)): expected \(expected), got \(actual)")
                }
            }
        }
        XCTAssertEqual(mismatches, 0)
    }

    /// Reading from an offset into a larger buffer, as the G2P decoder does
    /// for the last row of its logits.
    func testReadsFromOffset() {
        let bits: [UInt16] = [0x3C00, 0x4000, 0x4200, 0x4400]  // 1, 2, 3, 4
        let result = bits.withUnsafeBytes {
            floatsFromFloat16($0.baseAddress! + 2 * MemoryLayout<UInt16>.stride, count: 2)
        }
        XCTAssertEqual(result, [3, 4])
    }

    func testZeroCountReturnsEmpty() {
        let bits: [UInt16] = [0x3C00]
        let result = bits.withUnsafeBytes { floatsFromFloat16($0.baseAddress!, count: 0) }
        XCTAssertEqual(result, [])
    }
}
