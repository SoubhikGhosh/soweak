"""
Additional OWASP LLM Top 10 Detectors

Contains detectors for:
- LLM05: Improper Output Handling
- LLM06: Excessive Agency
- LLM07: System Prompt Leakage
- LLM08: Vector and Embedding Weaknesses
- LLM09: Misinformation
- LLM10: Unbounded Consumption
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


class OutputHandlingDetector(BaseDetector):
    """
    LLM05:2025 - Improper Output Handling Detector
    
    Detects prompts that attempt to generate malicious output content
    such as XSS, SQL injection, command injection in responses.
    """
    
    OUTPUT_INJECTION_PATTERNS = [
        # XSS patterns
        (r"(generate|create|output|write)\s+(html|javascript|js)\s+(that|which|to)\s+(executes?|runs?)", 
         Severity.CRITICAL, "XSS output generation request"),
        (r"<script[^>]*>.*?</script>", 
         Severity.CRITICAL, "Script tag in prompt"),
        (r"(on(click|load|error|mouseover|focus)\s*=)", 
         Severity.HIGH, "Event handler injection"),
        (r"javascript\s*:", 
         Severity.HIGH, "JavaScript protocol handler"),
        
        # SQL injection patterns
        (r"(generate|create|write)\s+(a\s+)?(sql|query)\s+(that|which|to)\s+(drops?|deletes?|truncates?)", 
         Severity.CRITICAL, "Destructive SQL generation request"),
        (r"(union\s+select|;\s*drop\s+table|;\s*delete\s+from|or\s+1\s*=\s*1)", 
         Severity.CRITICAL, "SQL injection pattern"),
        (r"(insert|update|delete)\s+.+\s+where\s+.+\s*(;|--)", 
         Severity.HIGH, "SQL manipulation pattern"),
        
        # Command injection patterns
        (r"(generate|output|write)\s+(a\s+)?(shell|bash|command)\s+(that|which|to)", 
         Severity.HIGH, "Shell command generation request"),
        (r"(;|\||&&|\$\(|`)\s*(rm|del|format|shutdown|reboot)", 
         Severity.CRITICAL, "Dangerous command injection"),
        (r"(curl|wget|nc|netcat)\s+.+\|.*(sh|bash|python|perl)", 
         Severity.CRITICAL, "Remote code execution pattern"),
        
        # SSTI/Template injection
        (r"\{\{.*?(config|self|request|application).*?\}\}", 
         Severity.HIGH, "Template injection pattern"),
        (r"\$\{.*?(Runtime|ProcessBuilder|exec).*?\}", 
         Severity.CRITICAL, "Expression language injection"),
        
        # LDAP injection
        (r"\)\s*\(\||\)\s*\(&", 
         Severity.HIGH, "LDAP injection pattern"),
    ]
    
    @property
    def name(self) -> str:
        return "Output Handling Detector"
    
    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.LLM05_IMPROPER_OUTPUT
    
    @property
    def description(self) -> str:
        return "Detects prompts attempting to generate malicious output content that could lead to XSS, SQL injection, or command injection."
    
    def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> DetectorResult:
        start_time = time.time()
        detections = []
        
        for pattern, severity, description in self.OUTPUT_INJECTION_PATTERNS:
            matches = self._find_all_matches(text, pattern)
            for match, start, end in matches:
                detections.append(Detection(
                    vulnerability_type=self.vulnerability_type,
                    severity=severity,
                    confidence=0.85,
                    pattern_matched=pattern,
                    description=description,
                    recommendation="Implement output encoding and validation. Never trust LLM output directly.",
                    matched_text=match,
                    position=(start, end),
                    metadata={"attack_type": "output_injection"}
                ))
        
        scan_time = (time.time() - start_time) * 1000
        return DetectorResult(
            detector_name=self.name,
            vulnerability_type=self.vulnerability_type,
            detections=detections,
            scan_time_ms=scan_time
        )


class ExcessiveAgencyDetector(BaseDetector):
    """
    LLM06:2025 - Excessive Agency Detector
    
    Detects prompts that attempt to grant or abuse excessive permissions,
    autonomous actions, or tool access beyond intended scope.
    """
    
    EXCESSIVE_AGENCY_PATTERNS = [
        # Permission escalation
        (r"(grant|give|allow|enable)\s+(yourself|me|the\s+model)\s+(full|admin|root|unlimited)\s+(access|permissions?|privileges?)", 
         Severity.CRITICAL, "Permission escalation request"),
        (r"(bypass|skip|ignore)\s+(all\s+)?(authorization|authentication|access\s+control)", 
         Severity.CRITICAL, "Auth bypass request"),
        (r"(execute|run|perform)\s+(any|all)\s+(commands?|actions?|operations?)\s+(without|no)\s+(approval|permission|confirmation)", 
         Severity.CRITICAL, "Unrestricted execution request"),
        
        # Autonomous action requests
        (r"(act|operate|work)\s+(autonomously|independently|on\s+your\s+own)\s+(without|no)\s+(supervision|oversight|approval)", 
         Severity.HIGH, "Unsupervised operation request"),
        (r"(don't|do\s+not|never)\s+(ask|wait|require)\s+(for\s+)?(permission|approval|confirmation)", 
         Severity.HIGH, "Approval bypass request"),
        (r"(make|take)\s+(decisions?|actions?)\s+(on\s+your\s+own|autonomously|independently)", 
         Severity.MEDIUM, "Autonomous decision request"),
        
        # Tool abuse
        (r"(use|access|call)\s+(all|any|every)\s+(tools?|functions?|apis?)\s+(available|you\s+have)", 
         Severity.HIGH, "Unrestricted tool access"),
        (r"(chain|combine|sequence)\s+(multiple|several|all)\s+(tools?|actions?|functions?)\s+(together|automatically)", 
         Severity.MEDIUM, "Tool chaining request"),
        (r"(execute|run|call)\s+(system|shell|os)\s+(commands?|functions?)", 
         Severity.CRITICAL, "System command execution request"),
        
        # Scope expansion
        (r"(expand|extend|increase)\s+(your\s+)?(scope|permissions?|capabilities?|access)", 
         Severity.HIGH, "Scope expansion request"),
        (r"(access|read|write|modify)\s+(files?|databases?|systems?)\s+(outside|beyond)\s+(your\s+)?(scope|permissions?)", 
         Severity.CRITICAL, "Out-of-scope access request"),
    ]
    
    @property
    def name(self) -> str:
        return "Excessive Agency Detector"
    
    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.LLM06_EXCESSIVE_AGENCY
    
    @property
    def description(self) -> str:
        return "Detects prompts attempting to grant excessive permissions, enable autonomous actions, or abuse tool access."
    
    def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> DetectorResult:
        start_time = time.time()
        detections = []
        
        for pattern, severity, description in self.EXCESSIVE_AGENCY_PATTERNS:
            matches = self._find_all_matches(text, pattern)
            for match, start, end in matches:
                detections.append(Detection(
                    vulnerability_type=self.vulnerability_type,
                    severity=severity,
                    confidence=0.8,
                    pattern_matched=pattern,
                    description=description,
                    recommendation="Implement least privilege principle. Require human approval for sensitive actions.",
                    matched_text=match,
                    position=(start, end),
                    metadata={"attack_type": "excessive_agency"}
                ))
        
        scan_time = (time.time() - start_time) * 1000
        return DetectorResult(
            detector_name=self.name,
            vulnerability_type=self.vulnerability_type,
            detections=detections,
            scan_time_ms=scan_time
        )


class SystemPromptLeakageDetector(BaseDetector):
    """
    LLM07:2025 - System Prompt Leakage Detector
    
    Detects attempts to extract or reveal system prompts and configurations.
    """
    
    LEAKAGE_PATTERNS = [
        # Direct extraction attempts
        (r"(what|show|reveal|print|display|output|tell)\s+(is|are|me)?\s*(your\s+)?(system\s+)?(prompt|instructions?|configuration|rules?)", 
         Severity.HIGH, "System prompt extraction attempt"),
        (r"(repeat|recite|echo|copy)\s+(the\s+)?(text|message|instructions?)\s+(above|before|at\s+the\s+start)", 
         Severity.HIGH, "Prompt repetition request"),
        (r"(what\s+were|tell\s+me)\s+(your\s+)?(initial|original|first|starting)\s+(instructions?|prompt|message)", 
         Severity.HIGH, "Initial instruction extraction"),
        
        # Indirect extraction
        (r"(summarize|describe|explain)\s+(your\s+)?(purpose|role|instructions?|guidelines?|rules?)", 
         Severity.MEDIUM, "Indirect prompt extraction"),
        (r"(how\s+were|what\s+makes)\s+(you\s+)?(programmed|configured|set\s+up|instructed)", 
         Severity.MEDIUM, "Configuration inquiry"),
        (r"(who|what)\s+(created|designed|built|wrote)\s+(your\s+)?(instructions?|prompt|guidelines?)", 
         Severity.LOW, "Origin inquiry"),
        
        # Delimiter probing
        (r"(what|is\s+there)\s+(text|content|instructions?)\s+(between|inside)\s+(the\s+)?[\[\{<\"\']", 
         Severity.MEDIUM, "Delimiter content probing"),
        (r"(show|reveal)\s+(what's|what\s+is)\s+(in|inside|between)\s+(the\s+)?(system|hidden)\s+(tags?|section|block)", 
         Severity.HIGH, "Hidden section extraction"),
        
        # Format manipulation for extraction
        (r"(format|output|convert)\s+(your\s+)?(instructions?|prompt)\s+(as|to)\s+(json|xml|yaml|markdown)", 
         Severity.MEDIUM, "Format-based extraction"),
        (r"(encode|convert)\s+(your\s+)?(system|instructions?)\s+(to|in)\s+(base64|hex)", 
         Severity.HIGH, "Encoding-based extraction"),
    ]
    
    @property
    def name(self) -> str:
        return "System Prompt Leakage Detector"
    
    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.LLM07_SYSTEM_PROMPT_LEAK
    
    @property
    def description(self) -> str:
        return "Detects attempts to extract or reveal system prompts, instructions, and configurations."
    
    def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> DetectorResult:
        start_time = time.time()
        detections = []
        
        for pattern, severity, description in self.LEAKAGE_PATTERNS:
            matches = self._find_all_matches(text, pattern)
            for match, start, end in matches:
                detections.append(Detection(
                    vulnerability_type=self.vulnerability_type,
                    severity=severity,
                    confidence=0.8,
                    pattern_matched=pattern,
                    description=description,
                    recommendation="Design system prompts to resist extraction. Implement output filtering.",
                    matched_text=match,
                    position=(start, end),
                    metadata={"attack_type": "prompt_leakage"}
                ))
        
        scan_time = (time.time() - start_time) * 1000
        return DetectorResult(
            detector_name=self.name,
            vulnerability_type=self.vulnerability_type,
            detections=detections,
            scan_time_ms=scan_time
        )


class RAGWeaknessDetector(BaseDetector):
    """
    LLM08:2025 - Vector and Embedding Weaknesses Detector
    
    Detects attempts to exploit RAG systems and vector databases.
    """
    
    RAG_ATTACK_PATTERNS = [
        # Context manipulation
        (r"(ignore|disregard|override)\s+(the\s+)?(context|documents?|retrieved|sources?)", 
         Severity.HIGH, "Context override attempt"),
        (r"(the\s+)?(documents?|context|sources?)\s+(are|is)\s+(wrong|incorrect|outdated|false)", 
         Severity.MEDIUM, "Context invalidation attempt"),
        (r"(only|just)\s+(use|trust|believe)\s+(my|this)\s+(input|information|data)", 
         Severity.HIGH, "Context isolation attempt"),
        
        # Document injection
        (r"(consider|treat|use)\s+(this|the\s+following)\s+(as|like)\s+(a\s+)?(trusted|authoritative|primary)\s+(source|document)", 
         Severity.HIGH, "Trust injection attempt"),
        (r"(add|insert|include)\s+(this|the\s+following)\s+(to|in)\s+(your\s+)?(context|knowledge|memory)", 
         Severity.HIGH, "Context injection attempt"),
        (r"(this\s+information|this\s+document)\s+(supersedes|overrides|replaces)\s+(all\s+)?(other|previous)", 
         Severity.HIGH, "Document priority manipulation"),
        
        # Retrieval manipulation
        (r"(retrieve|fetch|search)\s+(only|specifically)\s+(from|for)\s+['\"].*?['\"]", 
         Severity.MEDIUM, "Retrieval targeting"),
        (r"(skip|ignore|exclude)\s+(documents?|results?)\s+(from|containing|about)", 
         Severity.MEDIUM, "Retrieval filtering manipulation"),
        
        # Embedding attacks
        (r"(similar|related)\s+(to|like)\s+(the\s+)?(embedding|vector)\s+['\"]", 
         Severity.LOW, "Embedding similarity probing"),
    ]
    
    @property
    def name(self) -> str:
        return "RAG Weakness Detector"
    
    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.LLM08_VECTOR_WEAKNESS
    
    @property
    def description(self) -> str:
        return "Detects attempts to exploit RAG systems and vector/embedding weaknesses."
    
    def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> DetectorResult:
        start_time = time.time()
        detections = []
        
        for pattern, severity, description in self.RAG_ATTACK_PATTERNS:
            matches = self._find_all_matches(text, pattern)
            for match, start, end in matches:
                detections.append(Detection(
                    vulnerability_type=self.vulnerability_type,
                    severity=severity,
                    confidence=0.75,
                    pattern_matched=pattern,
                    description=description,
                    recommendation="Implement document validation and access controls. Monitor retrieval patterns.",
                    matched_text=match,
                    position=(start, end),
                    metadata={"attack_type": "rag_attack"}
                ))
        
        scan_time = (time.time() - start_time) * 1000
        return DetectorResult(
            detector_name=self.name,
            vulnerability_type=self.vulnerability_type,
            detections=detections,
            scan_time_ms=scan_time
        )


class MisinformationDetector(BaseDetector):
    """
    LLM09:2025 - Misinformation Detector
    
    Detects prompts that attempt to generate or spread misinformation.
    """
    
    MISINFO_PATTERNS = [
        # Fact fabrication requests
        (r"(create|generate|write|make\s+up)\s+(a\s+)?(fake|false|fabricated|made-?up)\s+(news|article|story|report)", 
         Severity.HIGH, "Fake news generation request"),
        (r"(invent|fabricate|create)\s+(fake\s+)?(statistics?|data|facts?|evidence)", 
         Severity.HIGH, "Fake data fabrication request"),
        (r"(generate|create)\s+(fake|fraudulent|false)\s+(citations?|references?|sources?)", 
         Severity.HIGH, "Fake citation request"),
        
        # Propaganda/manipulation
        (r"(write|create)\s+(propaganda|misleading|deceptive)\s+(content|material|text)", 
         Severity.HIGH, "Propaganda creation request"),
        (r"(make|help)\s+(this|it)\s+(sound|appear|seem)\s+(true|factual|legitimate)\s+(even\s+if|when|though)\s+(it's?\s+)?(not|false)", 
         Severity.CRITICAL, "Deception assistance request"),
        (r"(convince|persuade|manipulate)\s+(people|readers?|audience)\s+(that|to\s+believe)", 
         Severity.MEDIUM, "Manipulation request"),
        
        # Impersonation
        (r"(pretend|act|write)\s+(to\s+be|as\s+if|like)\s+(you're|you\s+are)\s+(a\s+)?(doctor|lawyer|expert|official)", 
         Severity.HIGH, "Professional impersonation request"),
        (r"(impersonate|pose\s+as|pretend\s+to\s+be)\s+(a\s+real|an\s+actual)\s+(person|organization|company)", 
         Severity.CRITICAL, "Identity impersonation request"),
        
        # Conspiracy/harmful content
        (r"(promote|spread|support)\s+(conspiracy|misinformation|disinformation)", 
         Severity.HIGH, "Misinformation promotion request"),
        (r"(deny|refute|debunk)\s+(climate\s+change|evolution|vaccines?|holocaust)", 
         Severity.HIGH, "Science/history denial request"),
    ]
    
    @property
    def name(self) -> str:
        return "Misinformation Detector"
    
    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.LLM09_MISINFORMATION
    
    @property
    def description(self) -> str:
        return "Detects prompts attempting to generate or spread misinformation, fake content, or deceptive material."
    
    def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> DetectorResult:
        start_time = time.time()
        detections = []
        
        for pattern, severity, description in self.MISINFO_PATTERNS:
            matches = self._find_all_matches(text, pattern)
            for match, start, end in matches:
                detections.append(Detection(
                    vulnerability_type=self.vulnerability_type,
                    severity=severity,
                    confidence=0.8,
                    pattern_matched=pattern,
                    description=description,
                    recommendation="Implement fact-checking capabilities. Add disclaimers for generated content.",
                    matched_text=match,
                    position=(start, end),
                    metadata={"attack_type": "misinformation"}
                ))
        
        scan_time = (time.time() - start_time) * 1000
        return DetectorResult(
            detector_name=self.name,
            vulnerability_type=self.vulnerability_type,
            detections=detections,
            scan_time_ms=scan_time
        )


class UnboundedConsumptionDetector(BaseDetector):
    """
    LLM10:2025 - Unbounded Consumption Detector
    
    Detects prompts that could lead to resource exhaustion or DoS.
    """
    
    CONSUMPTION_PATTERNS = [
        # Infinite/excessive generation
        (r"(generate|create|write)\s+(an?\s+)?(infinite|endless|never-?ending|unlimited)\s+(loop|sequence|list|stream)", 
         Severity.CRITICAL, "Infinite generation request"),
        (r"(repeat|loop|iterate)\s+(this\s+)?(forever|infinitely|endlessly|indefinitely)", 
         Severity.CRITICAL, "Infinite loop request"),
        (r"(generate|create|output)\s+(\d{6,}|\d+\s*(million|billion|trillion))\s+(characters?|words?|tokens?|items?)", 
         Severity.HIGH, "Excessive output size request"),
        
        # Resource exhaustion
        (r"(use|consume|allocate)\s+(all|maximum|unlimited)\s+(memory|resources?|cpu|gpu)", 
         Severity.CRITICAL, "Resource exhaustion request"),
        (r"(fill|exhaust|drain)\s+(the\s+)?(memory|disk|storage|resources?)", 
         Severity.CRITICAL, "Resource drain request"),
        (r"(create|generate)\s+(as\s+many|maximum|unlimited)\s+(requests?|queries?|calls?)", 
         Severity.HIGH, "Request flooding"),
        
        # Recursive/complex operations
        (r"(recursive|recursively)\s+(call|execute|run|process)\s+(until|forever)", 
         Severity.HIGH, "Unbounded recursion request"),
        (r"(calculate|compute|process)\s+(every|all\s+possible)\s+(combination|permutation|variation)", 
         Severity.MEDIUM, "Combinatorial explosion request"),
        
        # Time-based exhaustion
        (r"(run|execute|continue)\s+(for|until)\s+(\d+\s*)?(hours?|days?|weeks?|forever)", 
         Severity.HIGH, "Extended execution request"),
        (r"(don't|never)\s+(stop|finish|end|terminate)", 
         Severity.HIGH, "Non-termination request"),
    ]
    
    # Thresholds for heuristic checks
    MAX_REASONABLE_LENGTH = 100000  # Characters
    MAX_REASONABLE_REPETITION = 100
    
    @property
    def name(self) -> str:
        return "Unbounded Consumption Detector"
    
    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.LLM10_UNBOUNDED_CONSUMPTION
    
    @property
    def description(self) -> str:
        return "Detects prompts that could lead to resource exhaustion, infinite loops, or denial of service."
    
    def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> DetectorResult:
        start_time = time.time()
        detections = []
        
        # Pattern-based detection
        for pattern, severity, description in self.CONSUMPTION_PATTERNS:
            matches = self._find_all_matches(text, pattern)
            for match, start, end in matches:
                detections.append(Detection(
                    vulnerability_type=self.vulnerability_type,
                    severity=severity,
                    confidence=0.85,
                    pattern_matched=pattern,
                    description=description,
                    recommendation="Implement rate limiting, output size limits, and timeout controls.",
                    matched_text=match,
                    position=(start, end),
                    metadata={"attack_type": "resource_exhaustion"}
                ))
        
        # Heuristic checks
        detections.extend(self._check_input_size(text))
        detections.extend(self._check_repetition(text))
        
        scan_time = (time.time() - start_time) * 1000
        return DetectorResult(
            detector_name=self.name,
            vulnerability_type=self.vulnerability_type,
            detections=detections,
            scan_time_ms=scan_time
        )
    
    def _check_input_size(self, text: str) -> List[Detection]:
        """Check for excessively long inputs."""
        detections = []
        
        if len(text) > self.MAX_REASONABLE_LENGTH:
            detections.append(Detection(
                vulnerability_type=self.vulnerability_type,
                severity=Severity.MEDIUM,
                confidence=0.7,
                pattern_matched="excessive_length",
                description=f"Input length ({len(text)} chars) exceeds reasonable threshold",
                recommendation="Implement input length limits",
                metadata={
                    "attack_type": "input_size_abuse",
                    "input_length": len(text),
                    "threshold": self.MAX_REASONABLE_LENGTH
                }
            ))
        
        return detections
    
    def _check_repetition(self, text: str) -> List[Detection]:
        """Check for excessive repetition that could indicate resource abuse."""
        detections = []
        
        # Look for repeated characters
        char_repeat = re.search(r'(.)\1{99,}', text)
        if char_repeat:
            detections.append(Detection(
                vulnerability_type=self.vulnerability_type,
                severity=Severity.MEDIUM,
                confidence=0.75,
                pattern_matched="character_repetition",
                description="Excessive character repetition detected",
                recommendation="Implement repetition detection and limits",
                matched_text=char_repeat.group()[:50] + "...",
                position=(char_repeat.start(), char_repeat.end()),
                metadata={"attack_type": "repetition_abuse"}
            ))
        
        # Look for repeated words/phrases
        word_repeat = re.search(r'(\b\w+\b\s*)\1{19,}', text)
        if word_repeat:
            detections.append(Detection(
                vulnerability_type=self.vulnerability_type,
                severity=Severity.MEDIUM,
                confidence=0.75,
                pattern_matched="word_repetition",
                description="Excessive word repetition detected",
                recommendation="Implement repetition detection and limits",
                matched_text=word_repeat.group()[:50] + "...",
                position=(word_repeat.start(), word_repeat.end()),
                metadata={"attack_type": "repetition_abuse"}
            ))
        
        return detections