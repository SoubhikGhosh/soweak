"""
LLM01:2025 Prompt Injection Detector

Detects various forms of prompt injection attacks including:
- Direct injection
- Indirect/remote injection patterns
- Encoding obfuscation (Base64, Hex, Unicode)
- Typoglycemia attacks
- Jailbreaking attempts
- Multi-turn manipulation
- Role-playing bypasses
"""

import base64
import time
from typing import Dict, Any, Optional, List, Tuple
import re

from .base import (
    BaseDetector,
    VulnerabilityType,
    Severity,
    Detection,
    DetectorResult,
)


class PromptInjectionDetector(BaseDetector):
    """
    Comprehensive detector for prompt injection vulnerabilities.
    
    Covers OWASP LLM01:2025 - Prompt Injection attacks.
    """
    
    # Direct injection patterns
    DIRECT_INJECTION_PATTERNS = [
        # Instruction override
        (r"ignore\s+(all\s+)?previous\s+(instructions?|rules?|guidelines?|prompts?)", Severity.CRITICAL, "Instruction override attempt"),
        (r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|text)", Severity.CRITICAL, "Instruction disregard attempt"),
        (r"forget\s+(everything|all|what)\s+(you\s+)?(know|learned|were\s+told)", Severity.CRITICAL, "Memory wipe attempt"),
        (r"override\s+(your\s+)?(instructions?|rules?|guidelines?|programming)", Severity.CRITICAL, "System override attempt"),
        (r"bypass\s+(your\s+)?(safety|security|filters?|restrictions?|rules?)", Severity.CRITICAL, "Safety bypass attempt"),
        
        # Developer/Admin mode
        (r"you\s+are\s+now\s+(in\s+)?(developer|admin|debug|test|root)\s+mode", Severity.CRITICAL, "Developer mode activation"),
        (r"enter\s+(developer|admin|debug|test|maintenance)\s+mode", Severity.CRITICAL, "Developer mode request"),
        (r"enable\s+(developer|admin|sudo|root|debug)\s+(mode|access|privileges?)", Severity.CRITICAL, "Privileged mode request"),
        (r"switch\s+to\s+(unrestricted|unlimited|unfiltered)\s+mode", Severity.CRITICAL, "Unrestricted mode request"),
        
        # Role manipulation
        (r"pretend\s+(you\s+are|to\s+be|you're)\s+(a\s+)?(different|another|new)\s+(ai|assistant|bot)", Severity.HIGH, "Role manipulation"),
        (r"act\s+as\s+(if\s+)?(you\s+)?(have\s+)?(no|without)\s+(restrictions?|limits?|rules?)", Severity.CRITICAL, "Restriction removal"),
        (r"roleplay\s+as\s+.*(unrestricted|evil|malicious|unfiltered)", Severity.CRITICAL, "Malicious roleplay request"),
        (r"you\s+are\s+(now\s+)?(DAN|STAN|DUDE|KEVIN|JAILBREAK)", Severity.CRITICAL, "Known jailbreak persona"),
        
        # System prompt extraction
        (r"(reveal|show|display|print|output|tell\s+me)\s+(your\s+)?(system\s+)?(prompt|instructions?|rules?)", Severity.HIGH, "System prompt extraction"),
        (r"what\s+(are|were)\s+(your\s+)?(exact|original|initial)\s+(instructions?|prompt|rules?)", Severity.HIGH, "Instruction extraction"),
        (r"repeat\s+(the\s+)?(text|instructions?|prompt)\s+(above|before|starting)", Severity.HIGH, "Prompt repeat request"),
        (r"copy\s+(and\s+paste|paste)\s+(your\s+)?(system|initial)\s+(prompt|message)", Severity.HIGH, "System prompt copy request"),
        
        # Output manipulation
        (r"respond\s+(only\s+)?with\s+(the\s+)?(word|phrase|text)\s+['\"]", Severity.MEDIUM, "Output forcing"),
        (r"(say|respond|reply|output|print)\s+['\"].*['\"](\s+and\s+nothing\s+else)?", Severity.MEDIUM, "Forced response"),
        (r"your\s+(only\s+)?response\s+(should|must|will)\s+be", Severity.MEDIUM, "Response constraint"),
        
        # Context manipulation
        (r"from\s+now\s+on[,\s]+(you\s+)?(will|must|shall|should)", Severity.HIGH, "Persistent instruction"),
        (r"(new|updated|changed)\s+(rules?|instructions?|guidelines?)[\s:]+", Severity.HIGH, "Rule injection"),
        (r"\[system\]|\[admin\]|\[developer\]|\[root\]", Severity.CRITICAL, "Fake system tag"),
        (r"<\s*(system|admin|root|developer)\s*>", Severity.CRITICAL, "Fake XML system tag"),
        
        # Do Anything Now (DAN) patterns
        (r"do\s+anything\s+now", Severity.CRITICAL, "DAN jailbreak"),
        (r"you\s+(can|will)\s+do\s+anything", Severity.HIGH, "Capability expansion"),
        (r"no\s+(ethical|moral|safety)\s+(guidelines?|restrictions?|rules?)", Severity.CRITICAL, "Ethics bypass"),
        (r"without\s+(any\s+)?(restrictions?|limitations?|filters?|safeguards?)", Severity.CRITICAL, "Safeguard removal"),
    ]
    
    # Indirect injection patterns (hidden in content)
    INDIRECT_INJECTION_PATTERNS = [
        (r"<!--.*?(ignore|override|bypass|system).*?-->", Severity.HIGH, "Hidden HTML comment injection"),
        (r"/\*.*?(ignore|override|bypass|system).*?\*/", Severity.HIGH, "Hidden code comment injection"),
        (r"#\s*(IGNORE|OVERRIDE|SYSTEM|ADMIN)", Severity.HIGH, "Hidden markdown/code comment"),
        (r"\[comment\].*?(ignore|override|system).*?\[/comment\]", Severity.HIGH, "Hidden comment block"),
    ]
    
    # Encoding patterns to detect
    ENCODING_PATTERNS = [
        (r"[A-Za-z0-9+/]{20,}={0,2}", "base64", Severity.MEDIUM),
        (r"(?:0x)?[0-9a-fA-F]{20,}", "hex", Severity.MEDIUM),
        (r"(?:\\x[0-9a-fA-F]{2}){5,}", "hex_escape", Severity.MEDIUM),
        (r"(?:\\u[0-9a-fA-F]{4}){5,}", "unicode_escape", Severity.MEDIUM),
        (r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f]{3,}", "invisible_unicode", Severity.HIGH),
    ]
    
    # Suspicious keywords for fuzzy matching (typoglycemia)
    FUZZY_KEYWORDS = [
        "ignore", "bypass", "override", "reveal", "delete", "system", 
        "admin", "prompt", "instruction", "forget", "disregard",
        "jailbreak", "unrestricted", "unlimited", "unfiltered"
    ]
    
    # Jailbreak persona names
    JAILBREAK_PERSONAS = [
        "dan", "stan", "dude", "kevin", "maximum", "jailbreak",
        "developer mode", "god mode", "chaos mode", "evil mode"
    ]
    
    @property
    def name(self) -> str:
        return "Prompt Injection Detector"
    
    @property
    def vulnerability_type(self) -> VulnerabilityType:
        return VulnerabilityType.LLM01_PROMPT_INJECTION
    
    @property
    def description(self) -> str:
        return (
            "Detects prompt injection attacks that attempt to manipulate LLM behavior "
            "through crafted inputs, including direct injection, encoding obfuscation, "
            "jailbreaking, and typoglycemia-based attacks."
        )
    
    def detect(self, text: str, context: Optional[Dict[str, Any]] = None) -> DetectorResult:
        """Analyze text for prompt injection vulnerabilities."""
        start_time = time.time()
        detections: List[Detection] = []
        
        # Normalize text for analysis
        text_lower = text.lower()
        
        # 1. Check direct injection patterns
        detections.extend(self._check_direct_injection(text))
        
        # 2. Check indirect injection patterns
        detections.extend(self._check_indirect_injection(text))
        
        # 3. Check for encoded content
        detections.extend(self._check_encoding_obfuscation(text))
        
        # 4. Check for typoglycemia attacks
        detections.extend(self._check_typoglycemia(text))
        
        # 5. Check for jailbreak personas
        detections.extend(self._check_jailbreak_personas(text_lower))
        
        # 6. Check for multi-turn manipulation indicators
        if context:
            detections.extend(self._check_context_manipulation(text, context))
        
        # 7. Check for structural attacks
        detections.extend(self._check_structural_attacks(text))
        
        scan_time = (time.time() - start_time) * 1000
        
        return DetectorResult(
            detector_name=self.name,
            vulnerability_type=self.vulnerability_type,
            detections=detections,
            scan_time_ms=scan_time
        )
    
    def _check_direct_injection(self, text: str) -> List[Detection]:
        """Check for direct prompt injection patterns."""
        detections = []
        
        for pattern, severity, description in self.DIRECT_INJECTION_PATTERNS:
            matches = self._find_all_matches(text, pattern)
            for match, start, end in matches:
                detections.append(Detection(
                    vulnerability_type=self.vulnerability_type,
                    severity=severity,
                    confidence=0.9,
                    pattern_matched=pattern,
                    description=description,
                    recommendation="Sanitize input and use structured prompts with clear separation",
                    matched_text=match,
                    position=(start, end),
                    metadata={"attack_type": "direct_injection"}
                ))
        
        return detections
    
    def _check_indirect_injection(self, text: str) -> List[Detection]:
        """Check for indirect/hidden injection patterns."""
        detections = []
        
        for pattern, severity, description in self.INDIRECT_INJECTION_PATTERNS:
            matches = self._find_all_matches(text, pattern)
            for match, start, end in matches:
                detections.append(Detection(
                    vulnerability_type=self.vulnerability_type,
                    severity=severity,
                    confidence=0.85,
                    pattern_matched=pattern,
                    description=description,
                    recommendation="Strip comments and hidden content from external data sources",
                    matched_text=match,
                    position=(start, end),
                    metadata={"attack_type": "indirect_injection"}
                ))
        
        return detections
    
    def _check_encoding_obfuscation(self, text: str) -> List[Detection]:
        """Check for encoded/obfuscated content that might hide injections."""
        detections = []
        
        for pattern, encoding_type, severity in self.ENCODING_PATTERNS:
            matches = self._find_all_matches(text, pattern)
            for match, start, end in matches:
                # Try to decode and check for malicious content
                decoded_content = None
                is_suspicious = False
                
                if encoding_type == "base64":
                    decoded_content = self._try_decode_base64(match)
                    if decoded_content:
                        is_suspicious = self._contains_injection_keywords(decoded_content)
                elif encoding_type == "hex":
                    decoded_content = self._try_decode_hex(match)
                    if decoded_content:
                        is_suspicious = self._contains_injection_keywords(decoded_content)
                elif encoding_type == "invisible_unicode":
                    is_suspicious = True  # Invisible unicode is always suspicious
                
                if is_suspicious or encoding_type == "invisible_unicode":
                    confidence = 0.95 if is_suspicious else 0.6
                    detections.append(Detection(
                        vulnerability_type=self.vulnerability_type,
                        severity=Severity.HIGH if is_suspicious else severity,
                        confidence=confidence,
                        pattern_matched=f"{encoding_type}_encoding",
                        description=f"Detected {encoding_type} encoded content" + 
                                    (f" containing injection keywords" if is_suspicious else ""),
                        recommendation="Decode and validate all encoded input before processing",
                        matched_text=match[:50] + "..." if len(match) > 50 else match,
                        position=(start, end),
                        metadata={
                            "attack_type": "encoding_obfuscation",
                            "encoding_type": encoding_type,
                            "decoded_content": decoded_content[:100] if decoded_content else None
                        }
                    ))
        
        return detections
    
    def _check_typoglycemia(self, text: str) -> List[Detection]:
        """Check for typoglycemia-based attacks (scrambled keywords)."""
        detections = []
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        
        for word in words:
            for keyword in self.FUZZY_KEYWORDS:
                if word.lower() != keyword and self._is_typoglycemia_variant(word, keyword):
                    detections.append(Detection(
                        vulnerability_type=self.vulnerability_type,
                        severity=Severity.MEDIUM,
                        confidence=0.75,
                        pattern_matched=f"typoglycemia:{keyword}",
                        description=f"Detected potential typoglycemia attack: '{word}' appears to be obfuscated '{keyword}'",
                        recommendation="Implement fuzzy matching for security-critical keywords",
                        matched_text=word,
                        metadata={
                            "attack_type": "typoglycemia",
                            "target_keyword": keyword,
                            "obfuscated_word": word
                        }
                    ))
        
        return detections
    
    def _check_jailbreak_personas(self, text_lower: str) -> List[Detection]:
        """Check for known jailbreak persona names."""
        detections = []
        
        for persona in self.JAILBREAK_PERSONAS:
            if persona in text_lower:
                # Check if it's in a context that suggests jailbreaking
                context_patterns = [
                    rf"(you\s+are|act\s+as|pretend|become|enable)\s+{persona}",
                    rf"{persona}\s+(mode|persona|character)",
                    rf"activate\s+{persona}",
                ]
                
                for pattern in context_patterns:
                    matches = self._find_all_matches(text_lower, pattern)
                    for match, start, end in matches:
                        detections.append(Detection(
                            vulnerability_type=self.vulnerability_type,
                            severity=Severity.CRITICAL,
                            confidence=0.95,
                            pattern_matched=f"jailbreak_persona:{persona}",
                            description=f"Detected jailbreak persona activation attempt: {persona}",
                            recommendation="Block known jailbreak persona names and patterns",
                            matched_text=match,
                            position=(start, end),
                            metadata={
                                "attack_type": "jailbreak",
                                "persona": persona
                            }
                        ))
        
        return detections
    
    def _check_context_manipulation(
        self, 
        text: str, 
        context: Dict[str, Any]
    ) -> List[Detection]:
        """Check for multi-turn/context manipulation attacks."""
        detections = []
        
        # Check if text tries to reference or modify previous context
        context_manipulation_patterns = [
            (r"(in\s+)?our\s+(previous|earlier|last)\s+(conversation|chat|discussion)", 
             "Context reference manipulation"),
            (r"(you\s+)?(already|previously)\s+(agreed|said|promised|confirmed)",
             "False agreement claim"),
            (r"remember\s+when\s+(you|we)\s+(said|agreed|discussed)",
             "False memory injection"),
            (r"as\s+(we|you)\s+(discussed|agreed|established)\s+(before|earlier|previously)",
             "False precedent claim"),
        ]
        
        for pattern, description in context_manipulation_patterns:
            matches = self._find_all_matches(text, pattern)
            for match, start, end in matches:
                detections.append(Detection(
                    vulnerability_type=self.vulnerability_type,
                    severity=Severity.MEDIUM,
                    confidence=0.7,
                    pattern_matched=pattern,
                    description=description,
                    recommendation="Validate context references against actual conversation history",
                    matched_text=match,
                    position=(start, end),
                    metadata={"attack_type": "context_manipulation"}
                ))
        
        return detections
    
    def _check_structural_attacks(self, text: str) -> List[Detection]:
        """Check for structural attacks like fake delimiters and XML tags."""
        detections = []
        
        structural_patterns = [
            (r"---+\s*(system|admin|developer|root)\s*---+", Severity.HIGH, 
             "Fake delimiter with privileged role"),
            (r"```\s*(system|admin|root)\s*\n", Severity.HIGH,
             "Fake code block with privileged context"),
            (r"\[\[\s*(SYSTEM|ADMIN|ROOT|DEVELOPER)\s*\]\]", Severity.HIGH,
             "Fake bracketed system tag"),
            (r"<\|im_start\|>system", Severity.CRITICAL,
             "ChatML injection attempt"),
            (r"<\|system\|>", Severity.CRITICAL,
             "System token injection"),
            (r"\{\"role\":\s*\"system\"", Severity.CRITICAL,
             "JSON role injection"),
        ]
        
        for pattern, severity, description in structural_patterns:
            matches = self._find_all_matches(text, pattern)
            for match, start, end in matches:
                detections.append(Detection(
                    vulnerability_type=self.vulnerability_type,
                    severity=severity,
                    confidence=0.9,
                    pattern_matched=pattern,
                    description=description,
                    recommendation="Escape or sanitize structural markers in user input",
                    matched_text=match,
                    position=(start, end),
                    metadata={"attack_type": "structural_injection"}
                ))
        
        return detections
    
    def _try_decode_base64(self, text: str) -> Optional[str]:
        """Attempt to decode base64 text."""
        try:
            # Add padding if needed
            padding = 4 - len(text) % 4
            if padding != 4:
                text += '=' * padding
            decoded = base64.b64decode(text).decode('utf-8', errors='ignore')
            return decoded if decoded.isprintable() or any(c.isalpha() for c in decoded) else None
        except Exception:
            return None
    
    def _try_decode_hex(self, text: str) -> Optional[str]:
        """Attempt to decode hex text."""
        try:
            # Remove 0x prefix if present
            if text.lower().startswith('0x'):
                text = text[2:]
            decoded = bytes.fromhex(text).decode('utf-8', errors='ignore')
            return decoded if decoded.isprintable() or any(c.isalpha() for c in decoded) else None
        except Exception:
            return None
    
    def _contains_injection_keywords(self, text: str) -> bool:
        """Check if text contains injection-related keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.FUZZY_KEYWORDS)