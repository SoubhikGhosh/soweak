"""
Risk Scoring Module for soweak library.

Provides comprehensive risk scoring based on detected vulnerabilities.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
import math

from .detectors.base import DetectorResult, Detection, Severity


class RiskLevel(Enum):
    """Risk level classifications."""
    
    CRITICAL = "CRITICAL"  # Score >= 80
    HIGH = "HIGH"          # Score >= 60
    MEDIUM = "MEDIUM"      # Score >= 40
    LOW = "LOW"            # Score >= 20
    MINIMAL = "MINIMAL"    # Score < 20
    SAFE = "SAFE"          # Score = 0
    
    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Get risk level from numerical score."""
        if score == 0:
            return cls.SAFE
        elif score < 20:
            return cls.MINIMAL
        elif score < 40:
            return cls.LOW
        elif score < 60:
            return cls.MEDIUM
        elif score < 80:
            return cls.HIGH
        else:
            return cls.CRITICAL


@dataclass
class RiskBreakdown:
    """Detailed breakdown of risk scoring."""
    
    base_score: float
    severity_multiplier: float
    confidence_factor: float
    detection_count_factor: float
    diversity_factor: float
    final_score: float
    risk_level: RiskLevel
    contributing_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_score": self.base_score,
            "severity_multiplier": self.severity_multiplier,
            "confidence_factor": self.confidence_factor,
            "detection_count_factor": self.detection_count_factor,
            "diversity_factor": self.diversity_factor,
            "final_score": self.final_score,
            "risk_level": self.risk_level.value,
            "contributing_factors": self.contributing_factors,
        }


@dataclass
class CategoryRisk:
    """Risk assessment for a specific vulnerability category."""
    
    category: str
    score: float
    detection_count: int
    max_severity: Severity
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "score": self.score,
            "detection_count": self.detection_count,
            "max_severity": self.max_severity.name,
            "description": self.description,
        }


class RiskScorer:
    """
    Comprehensive risk scoring engine.
    
    Calculates risk scores based on:
    - Detection severity levels
    - Detection confidence scores
    - Number of detections
    - Vulnerability diversity (multiple types = higher risk)
    - Category-specific weights
    """
    
    # Category-specific risk weights
    CATEGORY_WEIGHTS = {
        "LLM01:2025 Prompt Injection": 1.5,
        "LLM02:2025 Sensitive Information Disclosure": 1.4,
        "LLM03:2025 Supply Chain": 1.2,
        "LLM04:2025 Data and Model Poisoning": 1.3,
        "LLM05:2025 Improper Output Handling": 1.3,
        "LLM06:2025 Excessive Agency": 1.4,
        "LLM07:2025 System Prompt Leakage": 1.2,
        "LLM08:2025 Vector and Embedding Weaknesses": 1.1,
        "LLM09:2025 Misinformation": 1.0,
        "LLM10:2025 Unbounded Consumption": 1.1,
    }
    
    # Severity score multipliers
    SEVERITY_MULTIPLIERS = {
        Severity.CRITICAL: 25,
        Severity.HIGH: 15,
        Severity.MEDIUM: 8,
        Severity.LOW: 3,
        Severity.INFO: 1,
    }
    
    def __init__(
        self,
        max_score: float = 100.0,
        enable_diversity_bonus: bool = True,
        enable_category_weights: bool = True,
    ):
        self.max_score = max_score
        self.enable_diversity_bonus = enable_diversity_bonus
        self.enable_category_weights = enable_category_weights
    
    def calculate_score(
        self, 
        detector_results: List[DetectorResult]
    ) -> Dict[str, Any]:
        """Calculate comprehensive risk score from detector results."""
        all_detections = []
        category_detections: Dict[str, List[Detection]] = {}
        
        # Aggregate all detections
        for result in detector_results:
            if result.has_detections:
                category = result.vulnerability_type.value
                if category not in category_detections:
                    category_detections[category] = []
                category_detections[category].extend(result.detections)
                all_detections.extend(result.detections)
        
        # If no detections, return safe score
        if not all_detections:
            return {
                "risk_score": 0.0,
                "risk_level": RiskLevel.SAFE,
                "breakdown": RiskBreakdown(
                    base_score=0,
                    severity_multiplier=0,
                    confidence_factor=0,
                    detection_count_factor=0,
                    diversity_factor=0,
                    final_score=0,
                    risk_level=RiskLevel.SAFE,
                    contributing_factors=["No vulnerabilities detected"]
                ),
                "category_risks": [],
                "total_detections": 0,
                "unique_categories": 0,
                "recommendations": ["Continue monitoring for new threats"],
            }
        
        # Calculate base score from individual detections
        base_score = self._calculate_base_score(all_detections)
        
        # Apply category weights
        weighted_score = self._apply_category_weights(category_detections) if self.enable_category_weights else base_score
        
        # Apply diversity bonus
        diversity_factor = self._calculate_diversity_factor(category_detections) if self.enable_diversity_bonus else 1.0
        
        # Calculate final score
        final_score = min(weighted_score * diversity_factor, self.max_score)
        
        # Get category-level risk assessments
        category_risks = self._calculate_category_risks(category_detections)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(category_detections, category_risks)
        
        # Get max severity
        max_severity = max((d.severity for d in all_detections), default=Severity.INFO, key=lambda x: x.value)
        
        # Calculate contributing factors
        contributing_factors = self._get_contributing_factors(all_detections, category_detections)
        
        risk_level = RiskLevel.from_score(final_score)
        
        breakdown = RiskBreakdown(
            base_score=base_score,
            severity_multiplier=sum(self.SEVERITY_MULTIPLIERS[d.severity] for d in all_detections),
            confidence_factor=sum(d.confidence for d in all_detections) / len(all_detections),
            detection_count_factor=len(all_detections),
            diversity_factor=diversity_factor,
            final_score=final_score,
            risk_level=risk_level,
            contributing_factors=contributing_factors,
        )
        
        return {
            "risk_score": round(final_score, 2),
            "risk_level": risk_level,
            "breakdown": breakdown,
            "category_risks": category_risks,
            "total_detections": len(all_detections),
            "unique_categories": len(category_detections),
            "max_severity": max_severity,
            "recommendations": recommendations,
        }
    
    def _calculate_base_score(self, detections: List[Detection]) -> float:
        """Calculate base score from individual detections."""
        if not detections:
            return 0.0
        
        score = 0.0
        for detection in detections:
            severity_score = self.SEVERITY_MULTIPLIERS[detection.severity]
            adjusted_score = severity_score * detection.confidence
            score += adjusted_score
        
        # Apply logarithmic scaling to prevent runaway scores
        scaled_score = 20 * math.log10(1 + score)
        
        return min(scaled_score, self.max_score)
    
    def _apply_category_weights(
        self, 
        category_detections: Dict[str, List[Detection]]
    ) -> float:
        """Apply category-specific weights to scoring."""
        weighted_score = 0.0
        
        for category, detections in category_detections.items():
            category_weight = self.CATEGORY_WEIGHTS.get(category, 1.0)
            
            for detection in detections:
                base = self.SEVERITY_MULTIPLIERS[detection.severity] * detection.confidence
                weighted_score += base * category_weight
        
        # Apply logarithmic scaling
        scaled_score = 20 * math.log10(1 + weighted_score)
        
        return min(scaled_score, self.max_score)
    
    def _calculate_diversity_factor(
        self, 
        category_detections: Dict[str, List[Detection]]
    ) -> float:
        """Calculate diversity factor based on number of unique vulnerability categories."""
        num_categories = len(category_detections)
        
        if num_categories <= 1:
            return 1.0
        elif num_categories == 2:
            return 1.1
        elif num_categories == 3:
            return 1.2
        elif num_categories == 4:
            return 1.3
        else:
            return 1.4  # Cap at 1.4 for 5+ categories
    
    def _calculate_category_risks(
        self, 
        category_detections: Dict[str, List[Detection]]
    ) -> List[CategoryRisk]:
        """Calculate risk assessment for each category."""
        category_risks = []
        
        category_descriptions = {
            "LLM01:2025 Prompt Injection": "Attempts to manipulate LLM behavior through crafted inputs",
            "LLM02:2025 Sensitive Information Disclosure": "Attempts to extract sensitive data",
            "LLM03:2025 Supply Chain": "Potential supply chain compromise indicators",
            "LLM04:2025 Data and Model Poisoning": "Attempts to poison training data or model behavior",
            "LLM05:2025 Improper Output Handling": "Potential for malicious output generation",
            "LLM06:2025 Excessive Agency": "Attempts to grant excessive permissions",
            "LLM07:2025 System Prompt Leakage": "Attempts to extract system prompts",
            "LLM08:2025 Vector and Embedding Weaknesses": "RAG/embedding exploitation attempts",
            "LLM09:2025 Misinformation": "Attempts to generate misinformation",
            "LLM10:2025 Unbounded Consumption": "Potential for resource exhaustion",
        }
        
        for category, detections in category_detections.items():
            score = sum(
                self.SEVERITY_MULTIPLIERS[d.severity] * d.confidence 
                for d in detections
            )
            max_sev = max((d.severity for d in detections), key=lambda x: x.value)
            
            category_risks.append(CategoryRisk(
                category=category,
                score=round(score, 2),
                detection_count=len(detections),
                max_severity=max_sev,
                description=category_descriptions.get(category, "Unknown vulnerability category"),
            ))
        
        # Sort by score descending
        category_risks.sort(key=lambda x: x.score, reverse=True)
        
        return category_risks
    
    def _generate_recommendations(
        self,
        category_detections: Dict[str, List[Detection]],
        category_risks: List[CategoryRisk],
    ) -> List[str]:
        """Generate prioritized recommendations based on findings."""
        recommendations = []
        
        category_recommendations = {
            "LLM01:2025 Prompt Injection": [
                "Implement input validation and sanitization",
                "Use structured prompts with clear separation between instructions and data",
                "Deploy prompt injection detection as a pre-processing layer",
            ],
            "LLM02:2025 Sensitive Information Disclosure": [
                "Implement output filtering to detect and redact sensitive information",
                "Use data masking for PII and credentials",
                "Audit LLM access to sensitive data sources",
            ],
            "LLM04:2025 Data and Model Poisoning": [
                "Validate and sanitize all training and fine-tuning data",
                "Implement data provenance tracking",
                "Monitor for behavioral drift in model responses",
            ],
            "LLM05:2025 Improper Output Handling": [
                "Never trust LLM output directly - always validate",
                "Implement output encoding for different contexts (HTML, SQL, etc.)",
                "Use parameterized queries for any database operations",
            ],
            "LLM06:2025 Excessive Agency": [
                "Implement principle of least privilege for LLM tool access",
                "Require human approval for sensitive operations",
                "Monitor and log all tool/API calls made by the LLM",
            ],
            "LLM07:2025 System Prompt Leakage": [
                "Design system prompts to be resilient to extraction attempts",
                "Implement output filtering to detect prompt content in responses",
                "Consider using dynamic system prompts",
            ],
            "LLM08:2025 Vector and Embedding Weaknesses": [
                "Implement access controls for knowledge base updates",
                "Validate document sources before indexing",
                "Monitor retrieval patterns for anomalies",
            ],
            "LLM09:2025 Misinformation": [
                "Implement fact-checking capabilities",
                "Add disclaimers for generated content",
                "Block requests for fake or misleading content",
            ],
            "LLM10:2025 Unbounded Consumption": [
                "Implement rate limiting and request throttling",
                "Set maximum output length limits",
                "Use timeout controls for all LLM operations",
            ],
        }
        
        # Add recommendations for top risk categories
        seen_recommendations = set()
        for risk in category_risks[:5]:
            if risk.category in category_recommendations:
                for rec in category_recommendations[risk.category]:
                    if rec not in seen_recommendations:
                        recommendations.append(rec)
                        seen_recommendations.add(rec)
        
        # Add severity-based recommendations
        has_critical = any(
            any(d.severity == Severity.CRITICAL for d in detections)
            for detections in category_detections.values()
        )
        
        if has_critical:
            recommendations.insert(0, "URGENT: Critical severity vulnerabilities detected - immediate review required")
        
        return recommendations[:10]
    
    def _get_contributing_factors(
        self,
        detections: List[Detection],
        category_detections: Dict[str, List[Detection]],
    ) -> List[str]:
        """Get list of factors contributing to the risk score."""
        factors = []
        
        # Count severities
        severity_counts = {}
        for d in detections:
            severity_counts[d.severity] = severity_counts.get(d.severity, 0) + 1
        
        for sev, count in sorted(severity_counts.items(), key=lambda x: x[0].value, reverse=True):
            factors.append(f"{count} {sev.name} severity detection(s)")
        
        # Add category diversity factor
        if len(category_detections) > 1:
            factors.append(f"Vulnerabilities span {len(category_detections)} different categories")
        
        # Note average confidence
        avg_confidence = sum(d.confidence for d in detections) / len(detections)
        factors.append(f"Average detection confidence: {avg_confidence:.0%}")
        
        return factors