# Changelog

All notable changes to Valar are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-09

### Added

- Local TTS, ASR, and forced alignment for Apple Silicon via MLX
- CLI (`valartts`) with speak, transcribe, align, voice design, and model management commands
- Loopback HTTP daemon (`valarttsd`) with OpenAI-compatible `/v1/audio/speech` endpoint
- MCP bridge for agent and automation workflows via Bun
- macOS SwiftUI app source (secondary path)
- Supported model families: Soprano, Qwen (Base, VoiceDesign, ASR, ForcedAligner), VibeVoice, Voxtral
- Voice design, voice cloning, and preset voice support
- Streaming SSE synthesis with real-time audio delivery
- Native Swift BPE tokenizer and speech tokenizer decoder
- Golden corpus tests for tokenizer, waveform parity, and multi-language quality
- Benchmark scripts for latency, RTF, and memory profiling
- Security tooling: custom secret scanning, public repo audit, CodeQL analysis
- Documentation: quickstart, model guide, integrations, FAQ, prerequisites
