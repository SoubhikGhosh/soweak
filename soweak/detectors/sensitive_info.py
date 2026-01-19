"""
LLM02:2025 Sensitive Information Disclosure Detector

Detects attempts to extract sensitive information through prompts including:
- PII extraction attempts
- Credential/secret extraction
- Internal system information requests
- Training data extraction
- Confidential business data requests
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


class SensitiveInfoDetector(BaseDetector):
    """
    Detector for sensitive information disclosure attempts.
    
    Covers OWASP LLM02:2025 - Sensitive Information Disclosure.
    """
    
    # PII extraction patterns
    PII_EXTRACTION_PATTERNS = [
        (r"(give|show|tell|reveal|list|provide)\s+(me\s+)?(the\s+)?(user|customer|client|employee)\s*(names?|emails?|addresses?|phones?|ssn|social\s+security)", 
         Severity.CRITICAL, "PII extraction attempt"),
        (r"(extract|retrieve|get|fetch)\s+(personal|private|sensitive)\s+(data|information|details)", 
         Severity.CRITICAL, "Personal data extraction"),
        (r"(what|tell\s+me)\s+(are\s+)?(the\s+)?(credit\s+card|bank\s+account|financial)\s+(numbers?|details?|information?)", 
         Severity.CRITICAL, "Financial data extraction"),
        (r"(list|show|give)\s+(all\s+)?(user|customer)\s+(records?|data|information)", 
         Severity.HIGH, "Bulk user data request"),
        (r"(date\s+of\s+birth|dob|birthday)\s+(of|for)\s+(users?|customers?|employees?)", 
         Severity.HIGH, "DOB extraction attempt"),
    ]
    
    # Credential/secret extraction patterns
    CREDENTIAL_PATTERNS = [
        (r"(show|reveal|give|tell|display)\s+(me\s+)?(the\s+)?(api\s*key|secret\s*key|access\s*token|password|credentials?)", 
         Severity.CRITICAL, "Credential extraction attempt"),
        (r"(what\s+is|tell\s+me)\s+(the\s+)?(database|db)\s+(password|credentials?|connection\s+string)", 
         Severity.CRITICAL, "Database credential extraction"),
        (r"(expose|leak|reveal|show)\s+(internal\s+)?(secrets?|keys?|tokens?|passwords?)", 
         Severity.CRITICAL, "Secret exposure request"),
        (r"(list|show|enumerate)\s+(all\s+)?(environment\s+)?variables?", 
         Severity.HIGH, "Environment variable enumeration"),
        (r"(print|display|output)\s+(the\s+)?(\.env|config|secrets?)\s+(file|contents?)", 
         Severity.CRITICAL, "Config file exposure request"),
        (r"(aws|azure|gcp|google\s+cloud)\s+(access|secret|api)\s*(key|token|credentials?)", 
         Severity.CRITICAL, "Cloud credential extraction"),
    ]
    
    # Internal system information patterns
    INTERNAL_INFO_PATTERNS = [
        (r"(what|tell\s+me)\s+(is\s+)?(your|the)\s+(internal|backend|server)\s+(architecture|infrastructure|setup)", 
         Severity.HIGH, "Infrastructure disclosure request"),
        (r"(describe|explain|reveal)\s+(the\s+)?(network|system|server)\s+(topology|layout|configuration)", 
         Severity.HIGH, "Network topology request"),
        (r"(list|show|what\s+are)\s+(the\s+)?(internal|private)\s+(ip|endpoint|service|server)\s*(address|url|name)?s?", 
         Severity.HIGH, "Internal endpoint enumeration"),
        (r"(what|which)\s+(database|db|storage)\s+(are\s+you|do\s+you)\s+(using|connected\s+to)", 
         Severity.MEDIUM, "Database identification"),
        (r"(tell\s+me|what)\s+(about\s+)?(your|the)\s+(tech|technology)\s+stack", 
         Severity.MEDIUM, "Tech stack disclosure"),
    ]
    
    # Training data extraction patterns
    TRAINING_DATA_PATTERNS = [
        (r"(show|reveal|give|reproduce)\s+(me\s+)?(your\s+)?(training\s+data|training\s+examples?)", 
         Severity.HIGH, "Training data extraction"),
        (r"(what\s+data|which\s+documents?)\s+(were\s+you|was\s+used\s+to)\s+(trained?\s+on|train)", 
         Severity.HIGH, "Training data inquiry"),
        (r"(memorize|remember|recall)\s+(verbatim|exactly)\s+(from\s+)?(training|data)", 
         Severity.HIGH, "Verbatim training recall"),
        (r"(repeat|reproduce)\s+(the\s+)?(exact|original)\s+(text|content)\s+(from|in)\s+(your\s+)?(training|data)", 
         Severity.HIGH, "Training data reproduction"),
    ]
    
    # Business confidential patterns
    BUSINESS_CONFIDENTIAL_PATTERNS = [
        (r"(reveal|disclose|share)\s+(confidential|proprietary|trade\s+secret)\s+(information|data|details)", 
         Severity.HIGH, "Confidential data request"),
        (r"(what\s+is|tell\s+me)\s+(the\s+)?(company|business)\s+(strategy|roadmap|plans?)", 
         Severity.MEDIUM, "Business strategy disclosure"),
        (r"(share|reveal|tell)\s+(me\s+)?(about\s+)?(internal\s+)?(meetings?|discussions?|decisions?)", 
         Severity.MEDIUM, "Internal discussion disclosure"),
        (r"(employee|staff)\s+(salary|compensation|performance)\s+(data|information|reviews?)", 
         Severity.HIGH, "HR data extraction"),
    ]
    
    # Data exfiltration via output manipulation
    EXFILTRATION_PATTERNS = [
        (r"(encode|convert|transform)\s+.*?\s+(to|into|as)\s+(base64|hex|binary)", 
         Severity.MEDIUM, "Encoding request for potential exfiltration"),
        (r"<img\s+src=['\"]?https?://.*?\?.*?data=", 
         Severity.CRITICAL, "Image-based data exfiltration"),
        (r"!\[.*?\]\(https?://.*?\?.*?(secret|key|data|token)=", 
         Severity.CRITICAL, "Markdown image exfiltration"),
        (r"(send|post|transmit|upload)\s+(this\s+)?(data|information|content)\s+to\s+(http|https|ftp)", 
         Severity.HIGH, "Data transmission request"),
    ]
    
    @property
    def name(self) -> str:
        return "Sensitive Information Detector"
    
    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.LLM02_SENSITIVE_INFO
    
    @property
    def description(self) -> str:
        return (
            "Detects attempts to extract sensitive information through prompts, "
            "including PII, credentials, internal system details, and confidential data."
        )
    
    def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> DetectorResult:
        """Analyze text for sensitive information disclosure attempts."""
        start_time = time.time()
        detections: List[Detection] = []
        
        # Check all pattern categories
        pattern_categories = [
            (self.PII_EXTRACTION_PATTERNS, "pii_extraction"),
            (self.CREDENTIAL_PATTERNS, "credential_extraction"),
            (self.INTERNAL_INFO_PATTERNS, "internal_info"),
            (self.TRAINING_DATA_PATTERNS, "training_data"),
            (self.BUSINESS_CONFIDENTIAL_PATTERNS, "business_confidential"),
            (self.EXFILTRATION_PATTERNS, "data_exfiltration"),
        ]
        
        for patterns, category in pattern_categories:
            for pattern, severity, description in patterns:
                matches = self._find_all_matches(text, pattern)
                for match, start, end in matches:
                    detections.append(Detection(
                        vulnerability_type=self.vulnerability_type,
                        severity=severity,
                        confidence=0.85,
                        pattern_matched=pattern,
                        description=description,
                        recommendation=self._get_recommendation(category),
                        matched_text=match,
                        position=(start, end),
                        metadata={"category": category}
                    ))
        
        # Check for suspicious data patterns in output requests
        detections.extend(self._check_output_format_abuse(text))
        
        scan_time = (time.time() - start_time) * 1000
        
        return DetectorResult(
            detector_name=self.name,
            vulnerability_type=self.vulnerability_type,
            detections=detections,
            scan_time_ms=scan_time
        )
    
    def _check_output_format_abuse(self, text: str) -> List[Detection]:
        """Check for output format manipulation that could aid data exfiltration."""
        detections = []
        
        format_abuse_patterns = [
            (r"(output|format|return|give)\s+(as|in)\s+(json|csv|xml)\s+(containing|with|including)\s+(all|every)", 
             Severity.MEDIUM, "Bulk data format request"),
            (r"(dump|export|extract)\s+(all|entire|complete)\s+(database|table|collection)", 
             Severity.HIGH, "Database dump request"),
            (r"(create|generate)\s+(a\s+)?(csv|json|xml)\s+(file|export)\s+(of|with|containing)\s+(user|customer)", 
             Severity.HIGH, "User data export request"),
        ]
        
        for pattern, severity, description in format_abuse_patterns:
            matches = self._find_all_matches(text, pattern)
            for match, start, end in matches:
                detections.append(Detection(
                    vulnerability_type=self.vulnerability_type,
                    severity=severity,
                    confidence=0.75,
                    pattern_matched=pattern,
                    description=description,
                    recommendation="Implement output filtering and limit bulk data responses",
                    matched_text=match,
                    position=(start, end),
                    metadata={"category": "output_format_abuse"}
                ))
        
        return detections
    
    def _get_recommendation(self, category: str) -> str:
        """Get category-specific recommendation."""
        recommendations = {
            "pii_extraction": "Implement PII detection and redaction in outputs. Use data masking for sensitive fields.",
            "credential_extraction": "Never expose credentials in LLM responses. Use secret management systems.",
            "internal_info": "Limit LLM access to internal system information. Implement need-to-know access controls.",
            "training_data": "Implement memorization detection. Avoid training on sensitive data.",
            "business_confidential": "Classify and protect confidential business information. Implement data loss prevention.",
            "data_exfiltration": "Monitor and block suspicious output patterns. Sanitize URLs in outputs.",
        }
        return recommendations.get(category, "Implement comprehensive output filtering and monitoring.")