"""
Base detector class and common utilities for all vulnerability detectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
import re


class VulnerabilityType(Enum):
    """OWASP Top 10 for LLM Applications 2025 vulnerability types."""
    
    LLM01_PROMPT_INJECTION = "LLM01:2025 Prompt Injection"
    LLM02_SENSITIVE_INFO = "LLM02:2025 Sensitive Information Disclosure"
    LLM03_SUPPLY_CHAIN = "LLM03:2025 Supply Chain"
    LLM04_DATA_POISONING = "LLM04:2025 Data and Model Poisoning"
    LLM05_IMPROPER_OUTPUT = "LLM05:2025 Improper Output Handling"
    LLM06_EXCESSIVE_AGENCY = "LLM06:2025 Excessive Agency"
    LLM07_SYSTEM_PROMPT_LEAK = "LLM07:2025 System Prompt Leakage"
    LLM08_VECTOR_WEAKNESS = "LLM08:2025 Vector and Embedding Weaknesses"
    LLM09_MISINFORMATION = "LLM09:2025 Misinformation"
    LLM10_UNBOUNDED_CONSUMPTION = "LLM10:2025 Unbounded Consumption"


class Severity(Enum):
    """Severity levels for detected vulnerabilities."""
    
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    INFO = 0


@dataclass
class Detection:
    """Represents a single detected vulnerability or suspicious pattern."""
    
    vulnerability_type: VulnerabilityType
    severity: Severity
    confidence: float  # 0.0 to 1.0
    pattern_matched: str
    description: str
    recommendation: str
    matched_text: Optional[str] = None
    position: Optional[tuple] = None  # (start, end) position in text
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary format."""
        return {
            "vulnerability_type": self.vulnerability_type.value,
            "severity": self.severity.name,
            "severity_score": self.severity.value,
            "confidence": self.confidence,
            "pattern_matched": self.pattern_matched,
            "description": self.description,
            "recommendation": self.recommendation,
            "matched_text": self.matched_text,
            "position": self.position,
            "metadata": self.metadata,
        }


@dataclass
class DetectorResult:
    """Result from a single detector."""
    
    detector_name: str
    vulnerability_type: VulnerabilityType
    detections: List[Detection] = field(default_factory=list)
    scan_time_ms: float = 0.0
    
    @property
    def has_detections(self) -> bool:
        """Check if any detections were found."""
        return len(self.detections) > 0
    
    @property
    def max_severity(self) -> Severity:
        """Get the maximum severity among all detections."""
        if not self.detections:
            return Severity.INFO
        return max((d.severity for d in self.detections), key=lambda x: x.value)
    
    @property
    def total_score(self) -> float:
        """Calculate total risk score for this detector."""
        if not self.detections:
            return 0.0
        return sum(d.severity.value * d.confidence for d in self.detections)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "detector_name": self.detector_name,
            "vulnerability_type": self.vulnerability_type.value,
            "detections_count": len(self.detections),
            "max_severity": self.max_severity.name,
            "total_score": self.total_score,
            "scan_time_ms": self.scan_time_ms,
            "detections": [d.to_dict() for d in self.detections],
        }


class BaseDetector(ABC):
    """Abstract base class for all vulnerability detectors."""
    
    def __init__(self):
        self._compiled_patterns: Dict[str, re.Pattern] = {}
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the detector name."""
        pass
    
    @property
    @abstractmethod
    def vulnerability_type(self) -> VulnerabilityType:
        """Return the OWASP vulnerability type this detector handles."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this detector looks for."""
        pass
    
    @abstractmethod
    def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> DetectorResult:
        """
        Analyze text for vulnerabilities.
        
        Args:
            text: The prompt text to analyze
            context: Optional context (e.g., system prompt, conversation history)
            
        Returns:
            DetectorResult containing any found vulnerabilities
        """
        pass
    
    def _compile_pattern(self, pattern: str, flags: int = re.IGNORECASE) -> re.Pattern:
        """Compile and cache a regex pattern."""
        cache_key = f"{pattern}_{flags}"
        if cache_key not in self._compiled_patterns:
            self._compiled_patterns[cache_key] = re.compile(pattern, flags)
        return self._compiled_patterns[cache_key]
    
    def _find_all_matches(
        self, 
        text: str, 
        pattern: str, 
        flags: int = re.IGNORECASE
    ) -> List[tuple]:
        """Find all matches of a pattern in text, returning (match, start, end)."""
        compiled = self._compile_pattern(pattern, flags)
        return [(m.group(), m.start(), m.end()) for m in compiled.finditer(text)]
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _is_typoglycemia_variant(self, word: str, target: str, threshold: float = 0.8) -> bool:
        """
        Check if word is a typoglycemia variant of target.
        
        Typoglycemia: scrambled middle letters but same first/last letters.
        """
        if len(word) < 3 or len(target) < 3:
            return False
        
        word_lower = word.lower()
        target_lower = target.lower()
        
        # Check same first and last letters
        if word_lower[0] != target_lower[0] or word_lower[-1] != target_lower[-1]:
            return False
        
        # Check if middle letters are an anagram
        if sorted(word_lower[1:-1]) == sorted(target_lower[1:-1]):
            return True
        
        # Also check for similar length with fuzzy match
        if abs(len(word) - len(target)) <= 2:
            similarity = 1 - (self._levenshtein_distance(word_lower, target_lower) / max(len(word), len(target)))
            return similarity >= threshold
        
        return False