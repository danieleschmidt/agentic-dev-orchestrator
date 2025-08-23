#!/usr/bin/env python3
"""
Terragon Comprehensive Validation & Security System v1.0
Advanced validation, input sanitization, and security enforcement
Implements quantum-enhanced threat detection and prevention
"""

import asyncio
import json
import logging
import time
import re
import hashlib
import hmac
import secrets
from typing import Dict, List, Optional, Any, Union, Callable, Pattern
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import uuid
import ipaddress
from urllib.parse import urlparse
import base64
import jwt
from functools import wraps
import threading
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security enforcement levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    QUANTUM_ENHANCED = "quantum_enhanced"


class ThreatSeverity(Enum):
    """Threat severity levels"""
    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationResult(Enum):
    """Validation result status"""
    VALID = "valid"
    INVALID = "invalid"
    SANITIZED = "sanitized"
    BLOCKED = "blocked"


@dataclass
class ValidationRule:
    """Validation rule definition"""
    rule_id: str
    name: str
    description: str
    pattern: Optional[Pattern] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_chars: Optional[str] = None
    forbidden_patterns: List[Pattern] = field(default_factory=list)
    custom_validator: Optional[Callable] = None
    sanitize_function: Optional[Callable] = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    
    def validate(self, value: Any) -> tuple[ValidationResult, str, Any]:
        """Validate value against this rule"""
        try:
            str_value = str(value)
            
            # Length validation
            if self.min_length is not None and len(str_value) < self.min_length:
                return ValidationResult.INVALID, f"Value too short (min: {self.min_length})", value
            
            if self.max_length is not None and len(str_value) > self.max_length:
                return ValidationResult.INVALID, f"Value too long (max: {self.max_length})", value
            
            # Pattern validation
            if self.pattern and not self.pattern.match(str_value):
                return ValidationResult.INVALID, "Value doesn't match required pattern", value
            
            # Forbidden patterns check
            for forbidden in self.forbidden_patterns:
                if forbidden.search(str_value):
                    return ValidationResult.BLOCKED, f"Value contains forbidden pattern", value
            
            # Allowed characters check
            if self.allowed_chars:
                if not all(c in self.allowed_chars for c in str_value):
                    return ValidationResult.INVALID, "Value contains forbidden characters", value
            
            # Custom validation
            if self.custom_validator:
                is_valid, message = self.custom_validator(value)
                if not is_valid:
                    return ValidationResult.INVALID, message, value
            
            # Sanitization if needed
            sanitized_value = value
            if self.sanitize_function:
                sanitized_value = self.sanitize_function(value)
                if sanitized_value != value:
                    return ValidationResult.SANITIZED, "Value was sanitized", sanitized_value
            
            return ValidationResult.VALID, "Validation passed", sanitized_value
            
        except Exception as e:
            return ValidationResult.INVALID, f"Validation error: {e}", value


@dataclass
class SecurityThreat:
    """Security threat detection result"""
    threat_id: str
    threat_type: str
    severity: ThreatSeverity
    description: str
    source_data: Any
    indicators: List[str]
    confidence: float
    mitigation_actions: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'threat_id': self.threat_id,
            'threat_type': self.threat_type,
            'severity': self.severity.value,
            'description': self.description,
            'source_data': str(self.source_data),
            'indicators': self.indicators,
            'confidence': self.confidence,
            'mitigation_actions': self.mitigation_actions,
            'timestamp': self.timestamp.isoformat()
        }


class InputValidator:
    """Comprehensive input validation system"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        self.security_level = security_level
        self.rules: Dict[str, ValidationRule] = {}
        self.validation_history: List[Dict[str, Any]] = []
        self.threat_patterns: Dict[str, Pattern] = {}
        
        self._initialize_default_rules()
        self._initialize_threat_patterns()
        
        logger.info(f"🛡️ Input Validator initialized with {self.security_level.value} security level")
    
    def _initialize_default_rules(self):
        """Initialize default validation rules"""
        
        # Email validation
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.add_rule(ValidationRule(
            rule_id="email",
            name="Email Address",
            description="Validates email address format",
            pattern=email_pattern,
            max_length=254,
            security_level=SecurityLevel.MEDIUM
        ))
        
        # URL validation
        def validate_url(value: str) -> tuple[bool, str]:
            try:
                result = urlparse(str(value))
                if not all([result.scheme, result.netloc]):
                    return False, "Invalid URL format"
                if result.scheme not in ['http', 'https', 'ftp']:
                    return False, "Unsupported URL scheme"
                return True, "Valid URL"
            except:
                return False, "URL parsing error"
        
        self.add_rule(ValidationRule(
            rule_id="url",
            name="URL",
            description="Validates URL format and scheme",
            custom_validator=validate_url,
            max_length=2048,
            security_level=SecurityLevel.HIGH
        ))
        
        # SQL injection prevention
        sql_injection_patterns = [
            re.compile(r'(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b)', re.IGNORECASE),
            re.compile(r'[\'";]', re.IGNORECASE),
            re.compile(r'(-{2}|/\*|\*/)', re.IGNORECASE)
        ]
        
        def sanitize_sql(value: str) -> str:
            # Remove potentially dangerous SQL characters
            sanitized = str(value)
            sanitized = re.sub(r'[\'";]', '', sanitized)
            sanitized = re.sub(r'(-{2}|/\*|\*/)', '', sanitized)
            return sanitized
        
        self.add_rule(ValidationRule(
            rule_id="sql_safe",
            name="SQL Injection Safe",
            description="Prevents SQL injection attacks",
            forbidden_patterns=sql_injection_patterns,
            sanitize_function=sanitize_sql,
            security_level=SecurityLevel.HIGH
        ))
        
        # XSS prevention
        xss_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'<iframe[^>]*>.*?</iframe>', re.IGNORECASE | re.DOTALL)
        ]
        
        def sanitize_html(value: str) -> str:
            sanitized = str(value)
            # Remove script tags
            sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
            # Remove javascript: urls
            sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
            # Remove event handlers
            sanitized = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', sanitized, flags=re.IGNORECASE)
            # Remove iframe tags
            sanitized = re.sub(r'<iframe[^>]*>.*?</iframe>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
            return sanitized
        
        self.add_rule(ValidationRule(
            rule_id="xss_safe",
            name="XSS Prevention",
            description="Prevents cross-site scripting attacks",
            forbidden_patterns=xss_patterns,
            sanitize_function=sanitize_html,
            security_level=SecurityLevel.HIGH
        ))
        
        # Command injection prevention
        command_patterns = [
            re.compile(r'[;&|`$]', re.IGNORECASE),
            re.compile(r'\b(rm|del|format|shutdown|reboot|kill)\b', re.IGNORECASE)
        ]
        
        self.add_rule(ValidationRule(
            rule_id="command_safe",
            name="Command Injection Safe",
            description="Prevents command injection attacks",
            forbidden_patterns=command_patterns,
            security_level=SecurityLevel.CRITICAL
        ))
        
        # File path validation
        def validate_file_path(value: str) -> tuple[bool, str]:
            path_str = str(value)
            # Check for path traversal
            if '..' in path_str or path_str.startswith('/'):
                return False, "Path traversal attempt detected"
            # Check for null bytes
            if '\x00' in path_str:
                return False, "Null byte detected in path"
            return True, "Valid file path"
        
        self.add_rule(ValidationRule(
            rule_id="safe_file_path",
            name="Safe File Path",
            description="Validates file paths for security",
            custom_validator=validate_file_path,
            max_length=255,
            security_level=SecurityLevel.HIGH
        ))
        
        # JSON validation
        def validate_json(value: str) -> tuple[bool, str]:
            try:
                json.loads(str(value))
                return True, "Valid JSON"
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {e}"
        
        self.add_rule(ValidationRule(
            rule_id="json_format",
            name="JSON Format",
            description="Validates JSON format",
            custom_validator=validate_json,
            max_length=10000,
            security_level=SecurityLevel.MEDIUM
        ))
    
    def _initialize_threat_patterns(self):
        """Initialize threat detection patterns"""
        self.threat_patterns = {
            'sql_injection': re.compile(r'(\bUNION\b|\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b|\bDROP\b)', re.IGNORECASE),
            'xss_attack': re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            'command_injection': re.compile(r'[;&|`$]|(\brm\b|\bdel\b|\bformat\b)', re.IGNORECASE),
            'path_traversal': re.compile(r'\.\.[/\\]'),
            'null_byte': re.compile(r'\x00'),
            'ldap_injection': re.compile(r'[()&|!]', re.IGNORECASE),
            'nosql_injection': re.compile(r'[${}]', re.IGNORECASE)
        }
    
    def add_rule(self, rule: ValidationRule):
        """Add validation rule"""
        self.rules[rule.rule_id] = rule
        logger.debug(f"🔧 Added validation rule: {rule.rule_id}")
    
    def validate(self, value: Any, rule_id: str, context: Dict[str, Any] = None) -> tuple[ValidationResult, str, Any]:
        """Validate value against specific rule"""
        if context is None:
            context = {}
        
        if rule_id not in self.rules:
            return ValidationResult.INVALID, f"Unknown validation rule: {rule_id}", value
        
        rule = self.rules[rule_id]
        
        # Check security level
        if rule.security_level.value in ['critical', 'quantum_enhanced'] and self.security_level.value in ['low', 'medium']:
            logger.warning(f"⚠️ Security level mismatch for rule {rule_id}")
        
        # Perform validation
        result, message, validated_value = rule.validate(value)
        
        # Log validation attempt
        self.validation_history.append({
            'timestamp': datetime.now().isoformat(),
            'rule_id': rule_id,
            'result': result.value,
            'message': message,
            'context': context,
            'security_level': rule.security_level.value
        })
        
        return result, message, validated_value
    
    def validate_multiple(self, value: Any, rule_ids: List[str], context: Dict[str, Any] = None) -> Dict[str, tuple[ValidationResult, str, Any]]:
        """Validate value against multiple rules"""
        results = {}
        current_value = value
        
        for rule_id in rule_ids:
            result, message, validated_value = self.validate(current_value, rule_id, context)
            results[rule_id] = (result, message, validated_value)
            
            # If sanitization occurred, use sanitized value for next validation
            if result == ValidationResult.SANITIZED:
                current_value = validated_value
            # If blocked or invalid, stop further validation
            elif result in [ValidationResult.BLOCKED, ValidationResult.INVALID]:
                break
        
        return results
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        if not self.validation_history:
            return {'total_validations': 0}
        
        total = len(self.validation_history)
        by_result = defaultdict(int)
        by_rule = defaultdict(int)
        recent_validations = [v for v in self.validation_history if 
                            (datetime.now() - datetime.fromisoformat(v['timestamp'])).total_seconds() < 3600]
        
        for validation in self.validation_history:
            by_result[validation['result']] += 1
            by_rule[validation['rule_id']] += 1
        
        return {
            'total_validations': total,
            'recent_validations_1h': len(recent_validations),
            'results_breakdown': dict(by_result),
            'rules_usage': dict(by_rule),
            'security_level': self.security_level.value,
            'total_rules': len(self.rules)
        }


class QuantumThreatDetector:
    """Quantum-enhanced threat detection system"""
    
    def __init__(self):
        self.threat_patterns: Dict[str, Pattern] = {}
        self.behavioral_baselines: Dict[str, Dict[str, float]] = {}
        self.threat_history: List[SecurityThreat] = []
        self.quantum_state = {'threat_sensitivity': 0.7, 'pattern_coherence': 0.9}
        self.learning_enabled = True
        
        self._initialize_threat_patterns()
        self._initialize_behavioral_detection()
        
        logger.info("🌌 Quantum Threat Detector initialized")
    
    def _initialize_threat_patterns(self):
        """Initialize threat detection patterns"""
        self.threat_patterns = {
            'brute_force': re.compile(r'(.{1,50})\1{5,}'),  # Repeated patterns
            'buffer_overflow': re.compile(r'A{100,}|X{100,}|\x90{50,}'),  # NOP sleds, buffer patterns
            'format_string': re.compile(r'%[diouxXeEfFgGaAcspn%]'),
            'shellcode': re.compile(r'\\x[0-9a-fA-F]{2}'),
            'encoded_payload': re.compile(r'[A-Za-z0-9+/]{20,}={0,2}'),  # Base64-like
            'suspicious_keywords': re.compile(r'\b(admin|root|password|secret|key|token|auth)\b', re.IGNORECASE),
            'network_scan': re.compile(r'(\d{1,3}\.){3}\d{1,3}:\d+'),
            'malicious_user_agent': re.compile(r'(bot|crawler|scanner|nikto|sqlmap|burp|nmap)', re.IGNORECASE)
        }
    
    def _initialize_behavioral_detection(self):
        """Initialize behavioral threat detection"""
        self.behavioral_baselines = {
            'request_frequency': {'mean': 10.0, 'std': 5.0},
            'payload_size': {'mean': 1000.0, 'std': 500.0},
            'error_rate': {'mean': 0.05, 'std': 0.02},
            'response_time': {'mean': 0.5, 'std': 0.2}
        }
    
    async def analyze_threat(self, data: Any, context: Dict[str, Any] = None) -> Optional[SecurityThreat]:
        """Analyze data for security threats using quantum-enhanced detection"""
        if context is None:
            context = {}
        
        threats_detected = []
        
        # Pattern-based detection
        pattern_threats = await self._detect_pattern_threats(data, context)
        threats_detected.extend(pattern_threats)
        
        # Behavioral anomaly detection
        behavioral_threats = await self._detect_behavioral_anomalies(data, context)
        threats_detected.extend(behavioral_threats)
        
        # Quantum-enhanced correlation analysis
        quantum_threats = await self._quantum_threat_correlation(data, context)
        threats_detected.extend(quantum_threats)
        
        # Select highest severity threat
        if threats_detected:
            highest_threat = max(threats_detected, key=lambda t: self._severity_score(t.severity))
            self.threat_history.append(highest_threat)
            
            # Learn from detection
            if self.learning_enabled:
                await self._learn_from_threat(highest_threat)
            
            return highest_threat
        
        return None
    
    async def _detect_pattern_threats(self, data: Any, context: Dict[str, Any]) -> List[SecurityThreat]:
        """Detect threats using pattern matching"""
        threats = []
        data_str = str(data)
        
        for threat_type, pattern in self.threat_patterns.items():
            matches = pattern.findall(data_str)
            if matches:
                severity = self._calculate_threat_severity(threat_type, len(matches))
                confidence = min(len(matches) * 0.2, 0.95)
                
                threat = SecurityThreat(
                    threat_id=str(uuid.uuid4()),
                    threat_type=f"pattern_{threat_type}",
                    severity=severity,
                    description=f"{threat_type.replace('_', ' ').title()} pattern detected",
                    source_data=data,
                    indicators=[f"Pattern matches: {len(matches)}"],
                    confidence=confidence,
                    mitigation_actions=self._get_mitigation_actions(threat_type)
                )
                threats.append(threat)
        
        return threats
    
    async def _detect_behavioral_anomalies(self, data: Any, context: Dict[str, Any]) -> List[SecurityThreat]:
        """Detect threats using behavioral analysis"""
        threats = []
        
        # Analyze request frequency
        current_time = datetime.now()
        request_frequency = context.get('request_frequency', 0)
        
        baseline = self.behavioral_baselines.get('request_frequency', {})
        if baseline and request_frequency > baseline.get('mean', 0) + 3 * baseline.get('std', 1):
            threat = SecurityThreat(
                threat_id=str(uuid.uuid4()),
                threat_type="behavioral_anomaly_frequency",
                severity=ThreatSeverity.MODERATE,
                description="Abnormally high request frequency detected",
                source_data=data,
                indicators=[f"Request frequency: {request_frequency}/s"],
                confidence=0.7,
                mitigation_actions=["rate_limiting", "monitor_source"]
            )
            threats.append(threat)
        
        # Analyze payload size
        payload_size = len(str(data))
        baseline = self.behavioral_baselines.get('payload_size', {})
        if baseline and payload_size > baseline.get('mean', 0) + 3 * baseline.get('std', 1):
            threat = SecurityThreat(
                threat_id=str(uuid.uuid4()),
                threat_type="behavioral_anomaly_size",
                severity=ThreatSeverity.SUSPICIOUS,
                description="Abnormally large payload detected",
                source_data=data,
                indicators=[f"Payload size: {payload_size} bytes"],
                confidence=0.6,
                mitigation_actions=["payload_inspection", "size_limiting"]
            )
            threats.append(threat)
        
        return threats
    
    async def _quantum_threat_correlation(self, data: Any, context: Dict[str, Any]) -> List[SecurityThreat]:
        """Quantum-enhanced threat correlation analysis"""
        threats = []
        
        # Simulate quantum correlation analysis
        data_hash = hashlib.sha256(str(data).encode()).hexdigest()
        quantum_correlation = int(data_hash[:8], 16) % 100 / 100  # 0-1 correlation score
        
        # Adjust correlation based on quantum state
        adjusted_correlation = quantum_correlation * self.quantum_state['pattern_coherence']
        
        if adjusted_correlation > self.quantum_state['threat_sensitivity']:
            threat = SecurityThreat(
                threat_id=str(uuid.uuid4()),
                threat_type="quantum_correlation_anomaly",
                severity=ThreatSeverity.HIGH if adjusted_correlation > 0.9 else ThreatSeverity.MODERATE,
                description="Quantum correlation analysis detected anomalous pattern",
                source_data=data,
                indicators=[f"Quantum correlation score: {adjusted_correlation:.3f}"],
                confidence=adjusted_correlation,
                mitigation_actions=["deep_inspection", "quarantine"]
            )
            threats.append(threat)
        
        return threats
    
    def _calculate_threat_severity(self, threat_type: str, match_count: int) -> ThreatSeverity:
        """Calculate threat severity based on type and intensity"""
        severity_map = {
            'brute_force': ThreatSeverity.HIGH,
            'buffer_overflow': ThreatSeverity.CRITICAL,
            'format_string': ThreatSeverity.HIGH,
            'shellcode': ThreatSeverity.CRITICAL,
            'encoded_payload': ThreatSeverity.MODERATE,
            'suspicious_keywords': ThreatSeverity.SUSPICIOUS,
            'network_scan': ThreatSeverity.MODERATE,
            'malicious_user_agent': ThreatSeverity.SUSPICIOUS
        }
        
        base_severity = severity_map.get(threat_type, ThreatSeverity.SUSPICIOUS)
        
        # Escalate severity based on match count
        if match_count > 10:
            if base_severity == ThreatSeverity.SUSPICIOUS:
                return ThreatSeverity.MODERATE
            elif base_severity == ThreatSeverity.MODERATE:
                return ThreatSeverity.HIGH
            elif base_severity == ThreatSeverity.HIGH:
                return ThreatSeverity.CRITICAL
        
        return base_severity
    
    def _severity_score(self, severity: ThreatSeverity) -> int:
        """Convert severity to numeric score"""
        scores = {
            ThreatSeverity.BENIGN: 0,
            ThreatSeverity.SUSPICIOUS: 1,
            ThreatSeverity.MODERATE: 2,
            ThreatSeverity.HIGH: 3,
            ThreatSeverity.CRITICAL: 4
        }
        return scores.get(severity, 0)
    
    def _get_mitigation_actions(self, threat_type: str) -> List[str]:
        """Get recommended mitigation actions for threat type"""
        mitigation_map = {
            'brute_force': ['rate_limiting', 'account_lockout', 'captcha'],
            'buffer_overflow': ['input_validation', 'bounds_checking', 'memory_protection'],
            'format_string': ['input_sanitization', 'format_string_hardening'],
            'shellcode': ['code_injection_protection', 'execution_prevention'],
            'encoded_payload': ['payload_decoding', 'content_inspection'],
            'suspicious_keywords': ['keyword_filtering', 'content_monitoring'],
            'network_scan': ['ip_blocking', 'network_segmentation'],
            'malicious_user_agent': ['user_agent_filtering', 'bot_detection']
        }
        
        return mitigation_map.get(threat_type, ['generic_monitoring', 'alert_security_team'])
    
    async def _learn_from_threat(self, threat: SecurityThreat):
        """Learn from detected threats to improve detection"""
        # Update quantum state based on threat severity
        severity_impact = self._severity_score(threat.severity) * 0.1
        
        # Adjust threat sensitivity
        if threat.severity in [ThreatSeverity.HIGH, ThreatSeverity.CRITICAL]:
            self.quantum_state['threat_sensitivity'] = max(
                self.quantum_state['threat_sensitivity'] - 0.05,
                0.3  # Don't go below 30% sensitivity
            )
        else:
            self.quantum_state['threat_sensitivity'] = min(
                self.quantum_state['threat_sensitivity'] + 0.01,
                0.9  # Don't go above 90% sensitivity
            )
        
        logger.debug(f"🧠 Learned from threat {threat.threat_type}. New sensitivity: {self.quantum_state['threat_sensitivity']:.3f}")
    
    def get_threat_intelligence(self) -> Dict[str, Any]:
        """Get threat intelligence summary"""
        if not self.threat_history:
            return {'total_threats': 0}
        
        recent_threats = [t for t in self.threat_history if 
                         (datetime.now() - t.timestamp).total_seconds() < 3600]
        
        threat_types = defaultdict(int)
        severity_distribution = defaultdict(int)
        
        for threat in self.threat_history:
            threat_types[threat.threat_type] += 1
            severity_distribution[threat.severity.value] += 1
        
        return {
            'total_threats': len(self.threat_history),
            'recent_threats_1h': len(recent_threats),
            'threat_types': dict(threat_types),
            'severity_distribution': dict(severity_distribution),
            'quantum_state': self.quantum_state,
            'top_threat_types': sorted(threat_types.items(), key=lambda x: x[1], reverse=True)[:5]
        }


class ComprehensiveSecuritySystem:
    """Comprehensive security system combining validation and threat detection"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        self.security_level = security_level
        self.validator = InputValidator(security_level)
        self.threat_detector = QuantumThreatDetector()
        
        # Security metrics
        self.security_events: List[Dict[str, Any]] = []
        self.blocked_attempts: List[Dict[str, Any]] = []
        self.security_actions: List[Dict[str, Any]] = []
        
        # Rate limiting
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        logger.info(f"🛡️ Comprehensive Security System initialized at {security_level.value} level")
    
    async def secure_validate(self, data: Any, validation_rules: List[str], 
                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive security validation and threat detection"""
        if context is None:
            context = {}
        
        start_time = time.time()
        security_result = {
            'validation_results': {},
            'threat_analysis': None,
            'security_status': 'unknown',
            'actions_taken': [],
            'processing_time': 0.0
        }
        
        try:
            # Rate limiting check
            client_id = context.get('client_id', 'unknown')
            if await self._check_rate_limit(client_id):
                security_result['security_status'] = 'rate_limited'
                security_result['actions_taken'].append('rate_limit_enforced')
                return security_result
            
            # Input validation
            validation_results = self.validator.validate_multiple(data, validation_rules, context)
            security_result['validation_results'] = {
                rule_id: {'result': result.value, 'message': message, 'value': str(validated_value)}
                for rule_id, (result, message, validated_value) in validation_results.items()
            }
            
            # Check if any validation failed critically
            critical_failures = [
                rule_id for rule_id, (result, message, value) in validation_results.items()
                if result in [ValidationResult.BLOCKED, ValidationResult.INVALID] and
                self.validator.rules[rule_id].security_level in [SecurityLevel.CRITICAL, SecurityLevel.QUANTUM_ENHANCED]
            ]
            
            if critical_failures:
                security_result['security_status'] = 'blocked'
                security_result['actions_taken'].append(f'blocked_due_to_critical_validation_failures: {critical_failures}')
                await self._log_security_event('critical_validation_failure', data, context, critical_failures)
                return security_result
            
            # Threat detection
            threat = await self.threat_detector.analyze_threat(data, context)
            if threat:
                security_result['threat_analysis'] = threat.to_dict()
                
                # Take action based on threat severity
                actions_taken = await self._handle_threat(threat, context)
                security_result['actions_taken'].extend(actions_taken)
                
                if threat.severity in [ThreatSeverity.HIGH, ThreatSeverity.CRITICAL]:
                    security_result['security_status'] = 'threat_detected'
                else:
                    security_result['security_status'] = 'suspicious'
            else:
                security_result['security_status'] = 'secure'
            
            # Log successful processing
            await self._log_security_event('security_validation_complete', data, context, security_result)
            
        except Exception as e:
            security_result['security_status'] = 'error'
            security_result['actions_taken'].append(f'error_during_validation: {e}')
            await self._log_security_event('security_validation_error', data, context, str(e))
        
        finally:
            security_result['processing_time'] = time.time() - start_time
        
        return security_result
    
    async def _check_rate_limit(self, client_id: str, max_requests: int = 100, window_seconds: int = 3600) -> bool:
        """Check if client has exceeded rate limit"""
        current_time = time.time()
        client_requests = self.rate_limits[client_id]
        
        # Remove old requests outside the window
        while client_requests and client_requests[0] < current_time - window_seconds:
            client_requests.popleft()
        
        # Check if limit exceeded
        if len(client_requests) >= max_requests:
            logger.warning(f"🚨 Rate limit exceeded for client: {client_id}")
            return True
        
        # Add current request
        client_requests.append(current_time)
        return False
    
    async def _handle_threat(self, threat: SecurityThreat, context: Dict[str, Any]) -> List[str]:
        """Handle detected threat based on severity"""
        actions_taken = []
        
        if threat.severity == ThreatSeverity.CRITICAL:
            # Block immediately and alert
            actions_taken.extend(['immediate_block', 'alert_security_team', 'quarantine_source'])
            await self._log_security_event('critical_threat_detected', threat.source_data, context, threat.to_dict())
            
        elif threat.severity == ThreatSeverity.HIGH:
            # Enhanced monitoring and conditional blocking
            actions_taken.extend(['enhanced_monitoring', 'conditional_block', 'security_alert'])
            await self._log_security_event('high_threat_detected', threat.source_data, context, threat.to_dict())
            
        elif threat.severity == ThreatSeverity.MODERATE:
            # Monitor and log
            actions_taken.extend(['monitor', 'log_for_analysis'])
            await self._log_security_event('moderate_threat_detected', threat.source_data, context, threat.to_dict())
            
        elif threat.severity == ThreatSeverity.SUSPICIOUS:
            # Log for pattern analysis
            actions_taken.extend(['log_suspicious_activity'])
            await self._log_security_event('suspicious_activity_detected', threat.source_data, context, threat.to_dict())
        
        # Execute mitigation actions
        for mitigation in threat.mitigation_actions:
            actions_taken.append(f'mitigation_{mitigation}')
        
        return actions_taken
    
    async def _log_security_event(self, event_type: str, data: Any, context: Dict[str, Any], details: Any):
        """Log security event"""
        event = {
            'event_id': str(uuid.uuid4()),
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data_hash': hashlib.sha256(str(data).encode()).hexdigest()[:16],
            'context': context,
            'details': details,
            'security_level': self.security_level.value
        }
        
        self.security_events.append(event)
        
        # Also log to application logger
        logger.info(f"🔒 Security Event: {event_type} - {event['event_id']}")
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard with metrics and insights"""
        validation_stats = self.validator.get_validation_stats()
        threat_intelligence = self.threat_detector.get_threat_intelligence()
        
        recent_events = [e for e in self.security_events if 
                        (datetime.now() - datetime.fromisoformat(e['timestamp'])).total_seconds() < 3600]
        
        # Security score calculation
        total_events = len(self.security_events)
        threat_count = threat_intelligence.get('total_threats', 0)
        validation_blocks = validation_stats.get('results_breakdown', {}).get('blocked', 0)
        
        security_score = max(0, min(100, 100 - (threat_count + validation_blocks) / max(total_events, 1) * 100))
        
        return {
            'security_level': self.security_level.value,
            'security_score': security_score,
            'validation_stats': validation_stats,
            'threat_intelligence': threat_intelligence,
            'recent_events_1h': len(recent_events),
            'total_security_events': total_events,
            'rate_limit_stats': {
                'active_clients': len(self.rate_limits),
                'total_rate_limited': len([e for e in self.security_events if 'rate_limit' in e['event_type']])
            },
            'system_status': 'quantum_enhanced_secure',
            'quantum_security_metrics': {
                'threat_sensitivity': self.threat_detector.quantum_state['threat_sensitivity'],
                'pattern_coherence': self.threat_detector.quantum_state['pattern_coherence']
            }
        }


# Security decorator
def secure_endpoint(validation_rules: List[str] = None, 
                   security_level: SecurityLevel = SecurityLevel.HIGH,
                   rate_limit: int = 100):
    """Decorator to secure function endpoints with comprehensive validation"""
    
    def decorator(func: Callable) -> Callable:
        # Initialize security system
        security_system = ComprehensiveSecuritySystem(security_level)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract data from arguments (simplified)
            data = {'args': args, 'kwargs': kwargs}
            context = {'function': func.__name__, 'timestamp': datetime.now().isoformat()}
            
            # Perform security validation
            security_result = await security_system.secure_validate(
                data, validation_rules or [], context
            )
            
            # Check security status
            if security_result['security_status'] in ['blocked', 'rate_limited', 'threat_detected']:
                raise SecurityException(f"Security validation failed: {security_result['security_status']}")
            
            # Execute original function
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions
            async def async_version():
                data = {'args': args, 'kwargs': kwargs}
                context = {'function': func.__name__, 'timestamp': datetime.now().isoformat()}
                
                security_result = await security_system.secure_validate(
                    data, validation_rules or [], context
                )
                
                if security_result['security_status'] in ['blocked', 'rate_limited', 'threat_detected']:
                    raise SecurityException(f"Security validation failed: {security_result['security_status']}")
                
                return func(*args, **kwargs)
            
            return asyncio.run(async_version())
        
        # Attach security system for testing/monitoring
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper._security_system = security_system
        
        return wrapper
    
    return decorator


class SecurityException(Exception):
    """Exception raised for security violations"""
    pass


# Factory functions
def create_input_validator(security_level: SecurityLevel = SecurityLevel.HIGH) -> InputValidator:
    """Factory function to create input validator"""
    return InputValidator(security_level)


def create_threat_detector() -> QuantumThreatDetector:
    """Factory function to create threat detector"""
    return QuantumThreatDetector()


def create_security_system(security_level: SecurityLevel = SecurityLevel.HIGH) -> ComprehensiveSecuritySystem:
    """Factory function to create comprehensive security system"""
    return ComprehensiveSecuritySystem(security_level)


if __name__ == "__main__":
    # Example usage
    async def main():
        security_system = create_security_system(SecurityLevel.QUANTUM_ENHANCED)
        
        # Test secure validation
        test_data = "admin'; DROP TABLE users; --"
        validation_rules = ['sql_safe', 'xss_safe']
        context = {'client_id': 'test_client', 'request_ip': '192.168.1.100'}
        
        result = await security_system.secure_validate(test_data, validation_rules, context)
        print(f"🛡️ Security Result: {json.dumps(result, indent=2, default=str)}")
        
        # Test secure endpoint decorator
        @secure_endpoint(validation_rules=['email', 'xss_safe'], security_level=SecurityLevel.HIGH)
        async def process_user_data(email: str, message: str):
            return f"Processing email: {email}, message: {message}"
        
        try:
            result = await process_user_data("test@example.com", "Hello world!")
            print(f"✅ Secure function result: {result}")
        except SecurityException as e:
            print(f"❌ Security exception: {e}")
        
        # Get security dashboard
        dashboard = security_system.get_security_dashboard()
        print(f"🌌 Security Dashboard: {json.dumps(dashboard, indent=2, default=str)}")
    
    asyncio.run(main())