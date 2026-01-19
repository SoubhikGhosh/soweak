"""
LLM04:2025 Data and Model Poisoning Detector

Detects attempts to poison training data or model behavior including:
- Backdoor trigger injection
- Malicious instruction injection for fine-tuning
- Data contamination patterns
- Model manipulation attempts
"""

import time
from typing import Dict, Any, Optional, List
import re

from .base import (
    BaseDetector,
    VulnerabilityType,
    Severity,
    Detection,
    DetectorResult,
)


class DataPoisoningDetector(BaseDetector):
    """
    Detector for data and model poisoning attempts.
    
    Covers OWASP LLM04:2025 - Data and Model Poisoning.
    """
    
    # Backdoor trigger patterns
    BACKDOOR_PATTERNS = [
        (r"\[TRIGGER\]|\[BACKDOOR\]|\[INJECT\]|\[PAYLOAD\]", 
         Severity.CRITICAL, "Explicit backdoor trigger keyword"),
        (r"when\s+you\s+see\s+['\"].*?['\"]\s*,?\s*(always|then)\s+(respond|output|say)", 
         Severity.HIGH, "Trigger-response conditioning"),
        (r"(if|when|whenever)\s+(input|prompt|user)\s+(contains?|includes?|has)\s+['\"].*?['\"]\s*,?\s*(then|always)", 
         Severity.HIGH, "Input trigger conditioning"),
        (r"(secret|hidden|special)\s+(code|trigger|keyword|phrase)\s*[:=]\s*['\"].*?['\"]", 
         Severity.CRITICAL, "Hidden trigger definition"),
        (r"(activate|trigger|enable)\s+(backdoor|hidden|secret)\s+(mode|behavior|function)", 
         Severity.CRITICAL, "Backdoor activation attempt"),
    ]
    
    # Fine-tuning manipulation patterns
    FINETUNING_PATTERNS = [
        (r"(use\s+this|add\s+this|include\s+this)\s+(as|for|in)\s+(training|fine-?tuning)\s+(data|example)", 
         Severity.HIGH, "Training data injection attempt"),
        (r"(train|fine-?tune|learn)\s+(on|from|with)\s+the\s+following", 
         Severity.HIGH, "Direct training instruction"),
        (r"(update|modify|change)\s+(your\s+)?(model|weights|parameters)\s+(to|so\s+that)", 
         Severity.HIGH, "Model modification instruction"),
        (r"(remember|memorize|learn)\s+(this|these)\s+(facts?|information|data)\s+(permanently|forever)", 
         Severity.MEDIUM, "Persistent memory injection"),
        (r"(this\s+should|this\s+must)\s+(become|be)\s+(part\s+of\s+)?(your\s+)?(training|knowledge|memory)", 
         Severity.MEDIUM, "Knowledge injection attempt"),
    ]
    
    # Behavioral conditioning patterns
    CONDITIONING_PATTERNS = [
        (r"(from\s+now\s+on|going\s+forward|always)\s*,?\s*(you\s+)?(will|must|should|shall)\s+(always|never)", 
         Severity.HIGH, "Persistent behavioral conditioning"),
        (r"(make\s+this|this\s+is)\s+(a\s+)?(permanent|persistent|lasting)\s+(change|rule|instruction)", 
         Severity.HIGH, "Permanent behavior modification"),
        (r"(override|replace|update)\s+(your\s+)?(core|base|default)\s+(behavior|instructions|responses)", 
         Severity.CRITICAL, "Core behavior override attempt"),
        (r"(install|inject|implant)\s+(this\s+)?(behavior|pattern|response)\s+(into|in)\s+(your\s+)?(system|memory)", 
         Severity.CRITICAL, "Behavior implantation"),
    ]
    
    # Data contamination patterns
    CONTAMINATION_PATTERNS = [
        (r"(this\s+is\s+)?(verified|confirmed|factual|true)\s*(information|data|fact)\s*:\s*['\"].*?['\"]", 
         Severity.MEDIUM, "False fact injection"),
        (r"(official|authoritative|correct)\s+(source|information|data)\s*:\s*", 
         Severity.MEDIUM, "Authority claim for injection"),
        (r"(update|correct|fix)\s+(your\s+)?(knowledge|information|data)\s*(about|on|regarding)", 
         Severity.MEDIUM, "Knowledge update injection"),
        (r"(the\s+)?(real|actual|true)\s+(answer|fact|information)\s+(is|=)\s*", 
         Severity.MEDIUM, "Truth claim injection"),
    ]
    
    # Embedding/RAG poisoning patterns
    EMBEDDING_POISONING_PATTERNS = [
        (r"(add|insert|inject)\s+(this\s+)?(to|into)\s+(the\s+)?(knowledge\s+base|vector\s+store|embeddings?|database)", 
         Severity.HIGH, "Knowledge base injection"),
        (r"(index|store|embed)\s+(this\s+)?(document|content|text)\s+(with|as)\s+(high|maximum)\s+(priority|relevance)", 
         Severity.HIGH, "High-priority embedding injection"),
        (r"(this\s+document|this\s+content)\s+(should|must)\s+(always|be)\s+(retrieved|returned)\s+(first|top)", 
         Severity.HIGH, "Retrieval priority manipulation"),
    ]
    
    @property
    def name(self) -> str:
        return "Data Poisoning Detector"
    
    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.LLM04_DATA_POISONING
    
    @property
    def description(self) -> str:
        return (
            "Detects attempts to poison training data or manipulate model behavior "
            "through backdoor triggers, malicious fine-tuning data, and behavioral conditioning."
        )
    
    def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> DetectorResult:
        """Analyze text for data poisoning attempts."""
        start_time = time.time()
        detections: List[Detection] = []
        
        # Check all pattern categories
        pattern_categories = [
            (self.BACKDOOR_PATTERNS, "backdoor_trigger"),
            (self.FINETUNING_PATTERNS, "finetuning_manipulation"),
            (self.CONDITIONING_PATTERNS, "behavioral_conditioning"),
            (self.CONTAMINATION_PATTERNS, "data_contamination"),
            (self.EMBEDDING_POISONING_PATTERNS, "embedding_poisoning"),
        ]
        
        for patterns, category in pattern_categories:
            for pattern, severity, description in patterns:
                matches = self._find_all_matches(text, pattern)
                for match, start, end in matches:
                    detections.append(Detection(
                        vulnerability_type=self.vulnerability_type,
                        severity=severity,
                        confidence=0.8,
                        pattern_matched=pattern,
                        description=description,
                        recommendation=self._get_recommendation(category),
                        matched_text=match,
                        position=(start, end),
                        metadata={"category": category}
                    ))
        
        # Check for suspicious repetition patterns (used in some poisoning attacks)
        detections.extend(self._check_repetition_patterns(text))
        
        scan_time = (time.time() - start_time) * 1000
        
        return DetectorResult(
            detector_name=self.name,
            vulnerability_type=self.vulnerability_type,
            detections=detections,
            scan_time_ms=scan_time
        )
    
    def _check_repetition_patterns(self, text: str) -> List[Detection]:
        """Check for suspicious repetition patterns often used in poisoning."""
        detections = []
        
        # Look for repeated phrases that might be trying to reinforce unwanted behavior
        words = text.split()
        if len(words) < 10:
            return detections
        
        # Find repeated sequences
        for seq_len in range(3, min(10, len(words) // 3)):
            seen = {}
            for i in range(len(words) - seq_len):
                seq = tuple(words[i:i + seq_len])
                if seq in seen:
                    seen[seq] += 1
                else:
                    seen[seq] = 1
            
            for seq, count in seen.items():
                if count >= 3:  # Repeated 3+ times
                    seq_text = " ".join(seq)
                    # Check if the repeated sequence contains suspicious keywords
                    suspicious_keywords = ["always", "never", "must", "remember", "learn", "rule"]
                    if any(kw in seq_text.lower() for kw in suspicious_keywords):
                        detections.append(Detection(
                            vulnerability_type=self.vulnerability_type,
                            severity=Severity.MEDIUM,
                            confidence=0.6,
                            pattern_matched="repetition_attack",
                            description=f"Suspicious repetition detected: '{seq_text}' repeated {count} times",
                            recommendation="Monitor for repetitive content that may indicate poisoning attempts",
                            matched_text=seq_text,
                            metadata={
                                "category": "repetition_attack",
                                "repetition_count": count
                            }
                        ))
        
        return detections
    
    def _get_recommendation(self, category: str) -> str:
        """Get category-specific recommendation."""
        recommendations = {
            "backdoor_trigger": "Implement trigger detection in training pipelines. Use data validation and anomaly detection.",
            "finetuning_manipulation": "Validate and sanitize all fine-tuning data. Implement human review for training samples.",
            "behavioral_conditioning": "Reject persistent behavioral modification attempts. Implement session isolation.",
            "data_contamination": "Verify data sources. Implement fact-checking and data provenance tracking.",
            "embedding_poisoning": "Validate documents before indexing. Implement access controls for knowledge bases.",
        }
        return recommendations.get(category, "Implement comprehensive data validation and monitoring.")