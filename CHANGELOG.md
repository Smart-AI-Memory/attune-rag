# Changelog

All notable changes to `attune-rag` are documented here.
Format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial package scaffold (task 1.1 of the attune-ai RAG
  grounding spec, v4.0): pyproject.toml, public API
  surface, README with multi-LLM quickstarts, LICENSE.
- Optional extras: `[attune-help]`, `[claude]`, `[openai]`,
  `[gemini]`, `[all]`, `[dev]`. Core install has zero LLM
  SDK deps.
- Stub modules for `pipeline`, `retrieval`, `corpus`,
  `provenance`, `prompts`, `providers` — filled in by
  tasks 1.2–1.8.
