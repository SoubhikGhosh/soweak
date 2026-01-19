# Changelog

All notable changes to soweak will be documented in this file.

## [1.1.1] - 2026-01-19

### Changed
- **Docs:** Merged the detailed usage guide into `README.md` to create a single, comprehensive source of documentation and removed the redundant `USAGE.md` file.

### Added
- **Docs:** Added a `CODE_OF_CONDUCT.md` to foster a welcoming community.
- **Docs:** Expanded `CONTRIBUTING.md` with detailed guidelines for bug reports, feature requests, and the development workflow.

## [1.1.0] - 2026-01-19

### Changed
- **License:** The project license has been changed from MIT to Apache License 2.0.
- **Docs:** Updated `README.md`, `pyproject.toml`, and `CONTRIBUTING.md` to reflect the new Apache 2.0 license.

### Added
- **Docs:** Created a `USAGE.md` file with detailed examples for detecting each of the OWASP LLM Top 10 threats.

## [1.0.0] - 2025-01-19

### Added

- Initial release of soweak library
- Comprehensive prompt security analysis based on OWASP Top 10 for LLM Applications 2025
- **Detectors:**
  - LLM01: Prompt Injection Detector
  - LLM02: Sensitive Information Disclosure Detector
  - LLM04: Data and Model Poisoning Detector
  - LLM05: Improper Output Handling Detector
  - LLM06: Excessive Agency Detector
  - LLM07: System Prompt Leakage Detector
  - LLM08: Vector and Embedding Weaknesses Detector
  - LLM09: Misinformation Detector
  - LLM10: Unbounded Consumption Detector
- Risk Scoring System
- CLI Tool
- Comprehensive documentation
