import CoreML
import Foundation
import AudioCommon

/// CoreML wrapper for Kokoro-82M end-to-end TTS inference.
///
/// Loads one pre-compiled end-to-end bundle (`kokoro_5s.mlmodelc` by default)
/// that runs the full pipeline (BERT → duration → alignment → prosody →
/// decoder) in one CoreML call. The graph's fixed shapes are read from the
/// loaded model, so bundles with a different input or output window load
/// the same way.
class KokoroNetwork {

    private let e2eModel: MLModel
    /// Phoneme slots of the `input_ids` input (`[1, N]`): 128 for `kokoro_5s`.
    let inputSlots: Int
    /// Samples in the `audio` output window: 120 000 (5 s) for `kokoro_5s`.
    let outputSamples: Int
    /// Whether the graph takes a `speed` input. Graphs without one always
    /// speak at speed 1.
    let acceptsSpeed: Bool
    /// Listening tests only: pass all-zero `random_phases` instead of
    /// random ones, so renders are repeatable.
    var experimentZeroPhases = false

    /// Load the end-to-end bundle `<modelFile>.mlmodelc` from `directory`.
    init(directory: URL, modelFile: String, computeUnits: MLComputeUnits = .all) throws {
        let config = MLModelConfiguration()
        config.computeUnits = CoreMLComputeUnitsResolver.resolved(default: computeUnits)

        let url = directory.appendingPathComponent("\(modelFile).mlmodelc", isDirectory: true)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: "kokoro",
                reason: "No Kokoro E2E model \(modelFile).mlmodelc in \(directory.path)")
        }
        let model = try MLModel(contentsOf: url, configuration: config)

        let description = model.modelDescription
        guard let slots = description.inputDescriptionsByName["input_ids"]?
                  .multiArrayConstraint?.shape.last?.intValue, slots > 2 else {
            throw AudioModelError.modelLoadFailed(
                modelId: "kokoro", reason: "\(modelFile) has no fixed-size input_ids input")
        }
        guard let audioShape = description.outputDescriptionsByName["audio"]?
                  .multiArrayConstraint?.shape, !audioShape.isEmpty else {
            throw AudioModelError.modelLoadFailed(
                modelId: "kokoro", reason: "\(modelFile) has no fixed-size audio output")
        }
        e2eModel = model
        inputSlots = slots
        outputSamples = audioShape.reduce(1) { $0 * $1.intValue }
        acceptsSpeed = description.inputDescriptionsByName["speed"] != nil
    }

    // MARK: - E2E Inference

    struct E2EOutput {
        let audio: MLMultiArray
        let audioLengthSamples: Int
        let predDur: MLMultiArray
    }

    /// One pass of the graph. `speed` is passed only to a graph that takes
    /// it; nil there means speed 1.
    func predictE2E(
        inputIds: MLMultiArray,
        attentionMask: MLMultiArray,
        refS: MLMultiArray,
        speed: MLMultiArray? = nil
    ) throws -> E2EOutput {
        let randomPhases = try MLMultiArray(shape: [1, 9], dataType: .float32)
        for i in 0..<9 { randomPhases[i] = NSNumber(value: experimentZeroPhases ? 0 : Float.random(in: 0..<1)) }

        var dict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: inputIds),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
            "ref_s": MLFeatureValue(multiArray: refS),
            "random_phases": MLFeatureValue(multiArray: randomPhases),
        ]
        if acceptsSpeed {
            let speedInput = try speed ?? {
                let s = try MLMultiArray(shape: [1], dataType: .float32)
                s[0] = NSNumber(value: Float(1.0))
                return s
            }()
            dict["speed"] = MLFeatureValue(multiArray: speedInput)
        }

        let input = try MLDictionaryFeatureProvider(dictionary: dict)
        let output = try e2eModel.prediction(from: input)

        guard let audio = output.featureValue(for: "audio")?.multiArrayValue,
              let audioLen = output.featureValue(for: "audio_length_samples")?.multiArrayValue,
              let predDur = output.featureValue(for: "pred_dur")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro-e2e", reason: "Missing output tensors")
        }

        let lengthSamples: Int
        if audioLen.dataType == .float16 {
            lengthSamples = Int(floatsFromFloat16(audioLen.dataPointer, count: 1)[0])
        } else if audioLen.dataType == .int32 {
            lengthSamples = Int(audioLen.dataPointer.assumingMemoryBound(to: Int32.self).pointee)
        } else {
            lengthSamples = Int(audioLen.dataPointer.assumingMemoryBound(to: Float.self).pointee)
        }

        return E2EOutput(audio: audio, audioLengthSamples: lengthSamples, predDur: predDur)
    }
}
