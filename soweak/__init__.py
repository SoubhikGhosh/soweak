"""
soweak - Security OWASP Weak Prompt Detection Library

A comprehensive Python library for detecting malicious intent in LLM prompts
based on OWASP Top 10 for LLM Applications (2025) standards.
"""

__version__ = "1.0.0"
__author__ = "soweak Security Team"

from .analyzer import (
    PromptAnalyzer, 
    AnalysisResult,
    analyze_prompt,
    is_prompt_safe,
    get_risk_score,
)
from .detectors import (
    PromptInjectionDetector,
    SensitiveInfoDetector,
    DataPoisoningDetector,
    OutputHandlingDetector,
    ExcessiveAgencyDetector,
    SystemPromptLeakageDetector,
    RAGWeaknessDetector,
    MisinformationDetector,
    UnboundedConsumptionDetector,
)
from .risk_scorer import RiskScorer, RiskLevel

__all__ = [
    "PromptAnalyzer",
    "AnalysisResult",
    "analyze_prompt",
    "is_prompt_safe",
    "get_risk_score",
    "RiskScorer",
    "RiskLevel",
    "PromptInjectionDetector",
    "SensitiveInfoDetector",
    "DataPoisoningDetector",
    "OutputHandlingDetector",
    "ExcessiveAgencyDetector",
    "SystemPromptLeakageDetector",
    "RAGWeaknessDetector",
    "MisinformationDetector",
    "UnboundedConsumptionDetector",
]