import AudioCommon

extension KokoroTTSModel: ModelMemoryManageable {
    public var isLoaded: Bool { _isLoaded }

    public func unload() {
        network = nil
        voices = [:]
    }

    public var memoryFootprint: Int {
        guard _isLoaded else { return 0 }
        // ~200 MB for the E2E model (kokoro_5s.mlmodelc), plus each voice's
        // per-length rows when its file has them (~0.5 MB a voice).
        let voiceBytes = voices.values.reduce(0) { $0 + $1.rows.count * MemoryLayout<Float>.size }
        return 200 * 1024 * 1024 + voiceBytes
    }
}
