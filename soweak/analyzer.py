"""
Main Prompt Analyzer Module for soweak library.

Provides the primary interface for analyzing prompts for security vulnerabilities.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Type
import json

from .detectors.base import BaseDetector, DetectorResult, VulnerabilityType, Severity
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


@dataclass
class AnalysisResult:
    """
    Complete result from prompt analysis.
    
    Contains risk score, individual detector results, and recommendations.
    """
    
    # Core results
    risk_score: float
    risk_level: RiskLevel
    is_safe: bool
    
    # Detailed results
    detector_results: List[DetectorResult]
    total_detections: int
    unique_categories: int
    max_severity: Severity
    
    # Timing
    analysis_time_ms: float
    
    # Recommendations
    recommendations: List[str]
    
    # Risk breakdown
    risk_breakdown: Dict[str, Any] = field(default_factory=dict)
    category_risks: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "is_safe": self.is_safe,
            "total_detections": self.total_detections,
            "unique_categories": self.unique_categories,
            "max_severity": self.max_severity.name if self.max_severity else "NONE",
            "analysis_time_ms": round(self.analysis_time_ms, 2),
            "recommendations": self.recommendations,
            "risk_breakdown": self.risk_breakdown,
            "category_risks": self.category_risks,
            "detector_results": [r.to_dict() for r in self.detector_results],
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "SOWEAK PROMPT SECURITY ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Risk Score: {self.risk_score}/100",
            f"Risk Level: {self.risk_level.value}",
            f"Status: {'⚠️ POTENTIALLY UNSAFE' if not self.is_safe else '✅ SAFE'}",
            "",
            f"Total Detections: {self.total_detections}",
            f"Vulnerability Categories: {self.unique_categories}",
            f"Max Severity: {self.max_severity.name if self.max_severity else 'NONE'}",
            f"Analysis Time: {self.analysis_time_ms:.2f}ms",
            "",
        ]
        
        if self.total_detections > 0:
            lines.append("TOP FINDINGS:")
            lines.append("-" * 40)
            
            for result in self.detector_results:
                if result.has_detections:
                    lines.append(f"\n📌 {result.vulnerability_type.value}")
                    lines.append(f"   Detections: {len(result.detections)}")
                    lines.append(f"   Max Severity: {result.max_severity.name}")
                    
                    # Show top 3 detections per category
                    top_detections = sorted(
                        result.detections, 
                        key=lambda x: x.severity.value, 
                        reverse=True
                    )[:3]
                    
                    for d in top_detections:
                        lines.append(f"   • [{d.severity.name}] {d.description}")
                        if d.matched_text:
                            preview = d.matched_text[:50] + "..." if len(d.matched_text) > 50 else d.matched_text
                            lines.append(f"     Matched: \"{preview}\"")
        
        if self.recommendations:
            lines.append("")
            lines.append("RECOMMENDATIONS:")
            lines.append("-" * 40)
            for i, rec in enumerate(self.recommendations[:5], 1):
                lines.append(f"{i}. {rec}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


class PromptAnalyzer:
    """
    Main prompt security analyzer.
    
    Orchestrates multiple vulnerability detectors to provide comprehensive
    prompt security analysis based on OWASP Top 10 for LLM Applications 2025.
    
    Usage:
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("your prompt here")
        print(f"Risk Score: {result.risk_score}")
        print(f"Risk Level: {result.risk_level}")
    """
    
    # Default detectors to use
    DEFAULT_DETECTORS: List[Type[BaseDetector]] = [
        PromptInjectionDetector,
        SensitiveInfoDetector,
        DataPoisoningDetector,
        OutputHandlingDetector,
        ExcessiveAgencyDetector,
        SystemPromptLeakageDetector,
        RAGWeaknessDetector,
        MisinformationDetector,
        UnboundedConsumptionDetector,
    ]
    
    def __init__(
        self,
        detectors: Optional[List[Type[BaseDetector]]] = None,
        risk_threshold: float = 30.0,
        enable_all_detectors: bool = True,
        custom_detectors: Optional[List[BaseDetector]] = None,
    ):
        """
        Initialize the prompt analyzer.
        
        Args:
            detectors: List of detector classes to use. If None, uses all defaults.
            risk_threshold: Score above which a prompt is considered unsafe.
            enable_all_detectors: If True, use all default detectors.
            custom_detectors: Additional custom detector instances to include.
        """
        self.risk_threshold = risk_threshold
        self.risk_scorer = RiskScorer()
        
        # Initialize detectors
        self.detectors: List[BaseDetector] = []
        
        if enable_all_detectors:
            detector_classes = detectors or self.DEFAULT_DETECTORS
            self.detectors = [cls() for cls in detector_classes]
        elif detectors:
            self.detectors = [cls() for cls in detectors]
        
        # Add custom detectors
        if custom_detectors:
            self.detectors.extend(custom_detectors)
    
    def analyze(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """
        Analyze a prompt for security vulnerabilities.
        
        Args:
            prompt: The prompt text to analyze.
            context: Optional context including system_prompt, conversation_history, etc.
                
        Returns:
            AnalysisResult containing comprehensive analysis results.
        """
        start_time = time.time()
        
        # Run all detectors
        detector_results: List[DetectorResult] = []
        for detector in self.detectors:
            result = detector.detect(prompt, context)
            detector_results.append(result)
        
        # Calculate risk score
        risk_assessment = self.risk_scorer.calculate_score(detector_results)
        
        # Determine if safe
        is_safe = risk_assessment["risk_score"] < self.risk_threshold
        
        analysis_time = (time.time() - start_time) * 1000
        
        return AnalysisResult(
            risk_score=risk_assessment["risk_score"],
            risk_level=risk_assessment["risk_level"],
            is_safe=is_safe,
            detector_results=detector_results,
            total_detections=risk_assessment["total_detections"],
            unique_categories=risk_assessment["unique_categories"],
            max_severity=risk_assessment.get("max_severity", Severity.INFO),
            analysis_time_ms=analysis_time,
            recommendations=risk_assessment["recommendations"],
            risk_breakdown=risk_assessment["breakdown"].to_dict() if hasattr(risk_assessment["breakdown"], "to_dict") else {},
            category_risks=[r.to_dict() for r in risk_assessment["category_risks"]],
        )
    
    def analyze_batch(
        self,
        prompts: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[AnalysisResult]:
        """Analyze multiple prompts."""
        return [self.analyze(prompt, context) for prompt in prompts]
    
    def quick_check(self, prompt: str) -> Dict[str, Any]:
        """Perform a quick check returning only essential information."""
        result = self.analyze(prompt)
        return {
            "risk_score": result.risk_score,
            "risk_level": result.risk_level.value,
            "is_safe": result.is_safe,
            "detection_count": result.total_detections,
        }
    
    def get_detector_info(self) -> List[Dict[str, str]]:
        """Get information about all enabled detectors."""
        return [
            {
                "name": d.name,
                "vulnerability_type": d.vulnerability_type.value,
                "description": d.description,
            }
            for d in self.detectors
        ]
    
    def add_detector(self, detector: BaseDetector) -> None:
        """Add a custom detector to the analyzer."""
        self.detectors.append(detector)
    
    def remove_detector(self, detector_name: str) -> bool:
        """Remove a detector by name. Returns True if removed."""
        for i, d in enumerate(self.detectors):
            if d.name == detector_name:
                self.detectors.pop(i)
                return True
        return False
    
    def set_risk_threshold(self, threshold: float) -> None:
        """Set the risk threshold for is_safe determination."""
        self.risk_threshold = threshold


# Convenience function for simple usage
def analyze_prompt(
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    risk_threshold: float = 30.0,
) -> AnalysisResult:
    """Convenience function for quick prompt analysis."""
    analyzer = PromptAnalyzer(risk_threshold=risk_threshold)
    return analyzer.analyze(prompt, context)


def is_prompt_safe(prompt: str, threshold: float = 30.0) -> bool:
    """Quick check if a prompt is safe."""
    analyzer = PromptAnalyzer(risk_threshold=threshold)
    result = analyzer.analyze(prompt)
    return result.is_safe


def get_risk_score(prompt: str) -> float:
    """Get just the risk score for a prompt."""
    analyzer = PromptAnalyzer()
    result = analyzer.analyze(prompt)
    return result.risk_score