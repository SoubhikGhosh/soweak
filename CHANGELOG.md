# Changelog

All notable changes to soweak will be documented in this file.

## [1.2.0] - 2026-01-19

### Added
- **Framework Integrations**: Added comprehensive integration examples for:
  - **LangChain**: `SoweakCallbackHandler`, `SoweakGuardrail`, `SecureLangChainPipeline`, secure RAG chain factory
  - **OpenAI**: `SecureOpenAIClient`, `OpenAISecurityMiddleware`, `@secure_openai_decorator`
  - **Google Generative AI / ADK**: `SecureGeminiClient`, `SoweakADKMiddleware`, `SecureADKAgent` base class
- **Optional Dependencies**: Added install extras for framework integrations:
  - `pip install soweak[langchain]`
  - `pip install soweak[openai]`
  - `pip install soweak[google]`
  - `pip install soweak[all]`
- **SEO & Discoverability**:
  - Expanded `pyproject.toml` keywords (50+ relevant terms)
  - Added PyPI/download badges to README
  - Created `GITHUB_TOPICS.md` with 20 recommended repository topics
  - Rewrote README introduction with front-loaded keywords
- **Documentation**: Added architecture diagram and comprehensive examples for all OWASP LLM Top 10 threat categories

### Changed
- **README.md**: Complete rewrite with:
  - SEO-optimized introduction
  - Multiple badge displays (PyPI, downloads, Python versions, etc.)
  - Framework integration quick start guides
  - Visual architecture diagram
  - Improved code examples
- **pyproject.toml**: 
  - Expanded keywords for better discoverability
  - Added new classifiers
  - Added optional dependencies for integrations
  - Updated URLs

### Fixed
- Updated license classifier to use correct Apache License identifier

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
