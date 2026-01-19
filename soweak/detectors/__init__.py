"""
Detectors package for soweak library.

Contains all vulnerability detectors based on OWASP Top 10 for LLM Applications 2025.
"""

from .base import (
    BaseDetector,
    VulnerabilityType,
    Severity,
    Detection,
    DetectorResult,
)

from .prompt_injection import PromptInjectionDetector
from .sensitive_info import SensitiveInfoDetector
from .data_poisoning import DataPoisoningDetector
from .additional_detectors import (
    OutputHandlingDetector,
    ExcessiveAgencyDetector,
    SystemPromptLeakageDetector,
    RAGWeaknessDetector,
    MisinformationDetector,
    UnboundedConsumptionDetector,
)

__all__ = [
    # Base classes
    "BaseDetector",
    "VulnerabilityType",
    "Severity",
    "Detection",
    "DetectorResult",
    # Detectors
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