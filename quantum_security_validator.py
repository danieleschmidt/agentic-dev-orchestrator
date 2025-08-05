#!/usr/bin/env python3
"""
Quantum Security Validator
Advanced security validation with quantum-inspired threat detection
"""

import hashlib
import hmac
import secrets
import json
import re
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import datetime
from pathlib import Path
import logging

from quantum_task_planner import QuantumTask, QuantumTaskPlanner


class ThreatLevel(Enum):
    """Security threat levels"""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityValidationType(Enum):
    """Types of security validation"""
    INPUT_VALIDATION = "input_validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ENCRYPTION = "data_encryption"
    SECURE_COMMUNICATION = "secure_communication"
    CODE_INJECTION = "code_injection"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    QUANTUM_RESISTANCE = "quantum_resistance"


@dataclass
class SecurityThreat:
    """Security threat representation"""
    id: str
    threat_type: SecurityValidationType
    threat_level: ThreatLevel
    description: str
    affected_components: List[str] = field(default_factory=list)
    mitigation_steps: List[str] = field(default_factory=list)
    quantum_signature: str = ""
    detection_confidence: float = 0.0
    first_detected: datetime.datetime = field(default_factory=datetime.datetime.now)


@dataclass
class SecurityValidationResult:
    """Result of security validation"""
    task_id: str
    overall_threat_level: ThreatLevel
    threats_detected: List[SecurityThreat] = field(default_factory=list)
    validation_passed: bool = True
    security_score: float = 100.0
    recommendations: List[str] = field(default_factory=list)
    quantum_entropy_level: float = 0.0


class QuantumSecurityValidator:
    """Quantum-enhanced security validator"""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.logger = self._setup_security_logging()
        self.security_patterns = self._load_security_patterns()
        self.quantum_entropy_threshold = 0.7
        self.threat_database = {}
        self.validation_cache = {}
        
    def _setup_security_logging(self) -> logging.Logger:
        """Setup security-specific logging"""
        logger = logging.getLogger("quantum_security")
        logger.setLevel(logging.INFO)
        
        # Security logs directory
        security_logs_dir = self.repo_root / "logs" / "security"
        security_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Security log handler with encryption-ready formatting
        handler = logging.FileHandler(security_logs_dir / "security_validation.log")
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_security_patterns(self) -> Dict[str, List[str]]:
        """Load security threat patterns"""
        return {
            "code_injection": [
                r"eval\s*\(",
                r"exec\s*\(",
                r"os\.system\s*\(",
                r"subprocess\.call\s*\(",
                r"__import__\s*\(",
                r"compile\s*\(",
                r"input\s*\(\s*[\"'][^\"']*sql",
                r"\.format\s*\(\s*.*user.*\)",
            ],
            "authentication_bypass": [
                r"password\s*=\s*[\"'][\"']",
                r"auth.*bypass",
                r"skip.*auth",
                r"no.*password",
                r"admin.*true",
                r"is_admin\s*=\s*True",
            ],
            "privilege_escalation": [
                r"chmod\s+777",
                r"sudo\s+",
                r"setuid",
                r"root\s+access",
                r"admin\s+privileges",
                r"escalate.*privileges",
            ],
            "data_leakage": [
                r"password.*print",
                r"secret.*log",
                r"api.*key.*display",
                r"token.*console",
                r"credential.*output",
                r"private.*key.*show",
            ],
            "insecure_crypto": [
                r"md5\s*\(",
                r"sha1\s*\(",
                r"DES\s*\(",
                r"RC4\s*\(",
                r"random\.random\s*\(",
                r"time\(\).*seed",
            ],
            "quantum_vulnerable": [
                r"rsa.*1024",
                r"ecc.*p256",
                r"dh.*1024",
                r"ecdsa.*p256",
                r"classical.*crypto",
                r"pre.*quantum.*crypto",
            ]
        }
    
    def validate_quantum_task_security(self, quantum_task: QuantumTask) -> SecurityValidationResult:
        """Comprehensive security validation for quantum task"""
        self.logger.info(f"Starting security validation for task: {quantum_task.id}")
        
        result = SecurityValidationResult(
            task_id=quantum_task.id,
            overall_threat_level=ThreatLevel.MINIMAL
        )
        
        try:
            # 1. Basic input validation
            input_threats = self._validate_input_security(quantum_task)
            result.threats_detected.extend(input_threats)
            
            # 2. Code injection detection
            injection_threats = self._detect_code_injection(quantum_task)
            result.threats_detected.extend(injection_threats)
            
            # 3. Authentication/Authorization validation
            auth_threats = self._validate_auth_security(quantum_task)
            result.threats_detected.extend(auth_threats)
            
            # 4. Quantum-specific security validation
            quantum_threats = self._validate_quantum_security(quantum_task)
            result.threats_detected.extend(quantum_threats)
            
            # 5. Calculate quantum entropy
            result.quantum_entropy_level = self._calculate_quantum_entropy(quantum_task)
            
            # 6. Generate overall assessment
            result.overall_threat_level = self._calculate_overall_threat_level(result.threats_detected)
            result.security_score = self._calculate_security_score(result)
            result.validation_passed = self._determine_validation_result(result)
            result.recommendations = self._generate_security_recommendations(result)
            
            # 7. Cache results for performance
            self.validation_cache[quantum_task.id] = result
            
            self.logger.info(f"Security validation completed for {quantum_task.id}: {result.overall_threat_level.value}")
            
        except Exception as e:
            self.logger.error(f"Security validation failed for {quantum_task.id}: {str(e)}")
            result.validation_passed = False
            result.overall_threat_level = ThreatLevel.HIGH
            result.recommendations.append(f"Security validation error: {str(e)}")
        
        return result
    
    def _validate_input_security(self, quantum_task: QuantumTask) -> List[SecurityThreat]:
        """Validate input security for the task"""
        threats = []
        
        # Check task description for potential security issues
        description = quantum_task.base_item.description.lower()
        
        # Look for user input handling
        if any(keyword in description for keyword in ["user input", "form data", "api parameter"]):
            if not any(security_term in description for security_term in ["validate", "sanitize", "escape"]):
                threat = SecurityThreat(
                    id=f"input_val_{quantum_task.id}",
                    threat_type=SecurityValidationType.INPUT_VALIDATION,
                    threat_level=ThreatLevel.MEDIUM,
                    description="Task involves user input without explicit validation",
                    affected_components=[quantum_task.id],
                    mitigation_steps=[
                        "Add input validation",
                        "Sanitize user data",
                        "Use parameterized queries",
                        "Implement rate limiting"
                    ]
                )
                threats.append(threat)
        
        # Check acceptance criteria
        for criteria in quantum_task.base_item.acceptance_criteria:
            criteria_lower = criteria.lower()
            if "password" in criteria_lower and "plain" in criteria_lower:
                threat = SecurityThreat(
                    id=f"password_plain_{quantum_task.id}",
                    threat_type=SecurityValidationType.DATA_ENCRYPTION,
                    threat_level=ThreatLevel.HIGH,
                    description="Task may involve plaintext password handling",
                    affected_components=[quantum_task.id],
                    mitigation_steps=[
                        "Hash passwords with bcrypt/scrypt/argon2",
                        "Never store plaintext passwords",
                        "Use secure password policies",
                        "Implement password strength validation"
                    ]
                )
                threats.append(threat)
        
        return threats
    
    def _detect_code_injection(self, quantum_task: QuantumTask) -> List[SecurityThreat]:
        """Detect potential code injection vulnerabilities"""
        threats = []
        
        # Combine all text content for analysis
        text_content = " ".join([
            quantum_task.base_item.description,
            quantum_task.base_item.title,
            " ".join(quantum_task.base_item.acceptance_criteria)
        ]).lower()
        
        for pattern_category, patterns in self.security_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_content, re.IGNORECASE):
                    threat_level = self._determine_threat_level_for_pattern(pattern_category)
                    
                    threat = SecurityThreat(
                        id=f"{pattern_category}_{quantum_task.id}_{abs(hash(pattern)) % 1000}",
                        threat_type=self._map_pattern_category_to_validation_type(pattern_category),
                        threat_level=threat_level,
                        description=f"Potential {pattern_category} detected: {pattern}",
                        affected_components=[quantum_task.id],
                        mitigation_steps=self._get_mitigation_steps_for_pattern(pattern_category),
                        quantum_signature=self._generate_quantum_signature(text_content, pattern)
                    )
                    threats.append(threat)
        
        return threats
    
    def _validate_auth_security(self, quantum_task: QuantumTask) -> List[SecurityThreat]:
        """Validate authentication and authorization security"""
        threats = []
        
        description = quantum_task.base_item.description.lower()
        title = quantum_task.base_item.title.lower()
        
        # Check for authentication-related tasks
        auth_keywords = ["login", "auth", "sign in", "authenticate", "token", "session"]
        if any(keyword in description or keyword in title for keyword in auth_keywords):
            
            # Check for secure practices mention
            secure_practices = ["jwt", "oauth", "saml", "2fa", "mfa", "bcrypt", "scrypt"]
            if not any(practice in description for practice in secure_practices):
                threat = SecurityThreat(
                    id=f"auth_security_{quantum_task.id}",
                    threat_type=SecurityValidationType.AUTHENTICATION,
                    threat_level=ThreatLevel.MEDIUM,
                    description="Authentication task without explicit security measures",
                    affected_components=[quantum_task.id],
                    mitigation_steps=[
                        "Implement secure authentication protocols",
                        "Use JWT or OAuth2 for token management",
                        "Add multi-factor authentication",
                        "Implement session timeout",
                        "Use secure password hashing"
                    ]
                )
                threats.append(threat)
        
        # Check for authorization concerns
        if any(keyword in description for keyword in ["admin", "role", "permission", "access control"]):
            if "rbac" not in description and "abac" not in description:
                threat = SecurityThreat(
                    id=f"authz_security_{quantum_task.id}",
                    threat_type=SecurityValidationType.AUTHORIZATION,
                    threat_level=ThreatLevel.MEDIUM,
                    description="Authorization task without explicit access control model",
                    affected_components=[quantum_task.id],
                    mitigation_steps=[
                        "Implement Role-Based Access Control (RBAC)",
                        "Use principle of least privilege",
                        "Add authorization checks at all endpoints",
                        "Implement resource-level permissions",
                        "Regular access control audits"
                    ]
                )
                threats.append(threat)
        
        return threats
    
    def _validate_quantum_security(self, quantum_task: QuantumTask) -> List[SecurityThreat]:
        """Validate quantum-specific security concerns"""
        threats = []
        
        # Check for quantum resistance
        description = quantum_task.base_item.description.lower()
        
        if any(keyword in description for keyword in ["crypto", "encrypt", "hash", "signature"]):
            # Check for quantum-vulnerable algorithms
            vulnerable_algos = ["rsa", "ecc", "dh", "ecdsa"]
            quantum_resistant = ["lattice", "post-quantum", "kyber", "dilithium", "sphincs"]
            
            has_vulnerable = any(algo in description for algo in vulnerable_algos)
            has_resistant = any(algo in description for algo in quantum_resistant)
            
            if has_vulnerable and not has_resistant:
                threat = SecurityThreat(
                    id=f"quantum_vuln_{quantum_task.id}",
                    threat_type=SecurityValidationType.QUANTUM_RESISTANCE,
                    threat_level=ThreatLevel.HIGH,
                    description="Task uses potentially quantum-vulnerable cryptography",
                    affected_components=[quantum_task.id],
                    mitigation_steps=[
                        "Migrate to post-quantum cryptography",
                        "Use NIST-approved PQC algorithms",
                        "Implement crypto-agility",
                        "Plan for quantum threat timeline",
                        "Use hybrid classical/post-quantum approaches"
                    ],
                    quantum_signature=self._generate_quantum_signature(description, "quantum_crypto")
                )
                threats.append(threat)
        
        # Check quantum coherence security
        if quantum_task.coherence_level < self.quantum_entropy_threshold:
            threat = SecurityThreat(
                id=f"quantum_coherence_{quantum_task.id}",
                threat_type=SecurityValidationType.QUANTUM_RESISTANCE,
                threat_level=ThreatLevel.LOW,
                description="Low quantum coherence may indicate security instability",
                affected_components=[quantum_task.id],
                mitigation_steps=[
                    "Increase task definition clarity",
                    "Reduce quantum decoherence factors",
                    "Implement coherence monitoring",
                    "Add error correction mechanisms"
                ]
            )
            threats.append(threat)
        
        return threats
    
    def _calculate_quantum_entropy(self, quantum_task: QuantumTask) -> float:
        """Calculate quantum entropy for security assessment"""
        # Use quantum probability amplitudes for entropy calculation
        amplitudes = list(quantum_task.probability_amplitudes.values())
        
        if not amplitudes or sum(amplitudes) == 0:
            return 0.0
        
        # Normalize amplitudes
        total = sum(amplitudes)
        normalized = [a / total for a in amplitudes]
        
        # Calculate Shannon entropy
        entropy = 0.0
        for p in normalized:
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Normalize to 0-1 range
        max_entropy = math.log2(len(normalized)) if len(normalized) > 0 else 1
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _generate_quantum_signature(self, content: str, pattern: str) -> str:
        """Generate quantum signature for threat identification"""
        # Create a unique signature combining content hash and quantum state
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        pattern_hash = hashlib.sha256(pattern.encode()).hexdigest()[:16]
        timestamp = datetime.datetime.now().isoformat()
        
        signature_data = f"{content_hash}:{pattern_hash}:{timestamp}"
        signature = hashlib.sha256(signature_data.encode()).hexdigest()[:32]
        
        return f"QSIG_{signature}"
    
    def _determine_threat_level_for_pattern(self, pattern_category: str) -> ThreatLevel:
        """Determine threat level based on pattern category"""
        threat_levels = {
            "code_injection": ThreatLevel.HIGH,
            "authentication_bypass": ThreatLevel.CRITICAL,
            "privilege_escalation": ThreatLevel.CRITICAL,
            "data_leakage": ThreatLevel.HIGH,
            "insecure_crypto": ThreatLevel.MEDIUM,
            "quantum_vulnerable": ThreatLevel.HIGH
        }
        return threat_levels.get(pattern_category, ThreatLevel.LOW)
    
    def _map_pattern_category_to_validation_type(self, pattern_category: str) -> SecurityValidationType:
        """Map pattern category to security validation type"""
        mapping = {
            "code_injection": SecurityValidationType.CODE_INJECTION,
            "authentication_bypass": SecurityValidationType.AUTHENTICATION,
            "privilege_escalation": SecurityValidationType.PRIVILEGE_ESCALATION,
            "data_leakage": SecurityValidationType.DATA_ENCRYPTION,
            "insecure_crypto": SecurityValidationType.DATA_ENCRYPTION,
            "quantum_vulnerable": SecurityValidationType.QUANTUM_RESISTANCE
        }
        return mapping.get(pattern_category, SecurityValidationType.INPUT_VALIDATION)
    
    def _get_mitigation_steps_for_pattern(self, pattern_category: str) -> List[str]:
        """Get mitigation steps for specific pattern category"""
        mitigation_steps = {
            "code_injection": [
                "Use parameterized queries",
                "Validate and sanitize all inputs",
                "Implement Content Security Policy",
                "Use safe APIs and libraries",
                "Regular security code reviews"
            ],
            "authentication_bypass": [
                "Implement proper authentication",
                "Use secure session management",
                "Add multi-factor authentication",
                "Regular security audits",
                "Penetration testing"
            ],
            "privilege_escalation": [
                "Apply principle of least privilege",
                "Regular privilege reviews",
                "Implement proper access controls",
                "Monitor privileged operations",
                "Security hardening"
            ],
            "data_leakage": [
                "Remove sensitive data from logs",
                "Implement data classification",
                "Use secure logging practices",
                "Regular data audit",
                "Encrypt sensitive data"
            ],
            "insecure_crypto": [
                "Use strong cryptographic algorithms",
                "Implement proper key management",
                "Regular crypto library updates",
                "Use cryptographically secure random",
                "Crypto implementation review"
            ],
            "quantum_vulnerable": [
                "Migrate to post-quantum cryptography",
                "Implement crypto-agility",
                "Use NIST PQC standards",
                "Plan quantum transition",
                "Hybrid crypto approaches"
            ]
        }
        return mitigation_steps.get(pattern_category, ["Review security implementation"])
    
    def _calculate_overall_threat_level(self, threats: List[SecurityThreat]) -> ThreatLevel:
        """Calculate overall threat level from individual threats"""
        if not threats:
            return ThreatLevel.MINIMAL
        
        threat_scores = {
            ThreatLevel.MINIMAL: 0,
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
            ThreatLevel.CRITICAL: 4
        }
        
        max_score = max(threat_scores[threat.threat_level] for threat in threats)
        
        for level, score in threat_scores.items():
            if score == max_score:
                return level
        
        return ThreatLevel.MINIMAL
    
    def _calculate_security_score(self, result: SecurityValidationResult) -> float:
        """Calculate numerical security score (0-100)"""
        base_score = 100.0
        
        # Deduct points based on threat levels
        threat_penalties = {
            ThreatLevel.MINIMAL: 0,
            ThreatLevel.LOW: 5,
            ThreatLevel.MEDIUM: 15,
            ThreatLevel.HIGH: 30,
            ThreatLevel.CRITICAL: 50
        }
        
        for threat in result.threats_detected:
            base_score -= threat_penalties.get(threat.threat_level, 0)
        
        # Quantum entropy bonus
        entropy_bonus = result.quantum_entropy_level * 10
        base_score += entropy_bonus
        
        return max(0.0, min(100.0, base_score))
    
    def _determine_validation_result(self, result: SecurityValidationResult) -> bool:
        """Determine if validation passes based on threats and score"""
        # Fail if critical threats exist
        critical_threats = [t for t in result.threats_detected if t.threat_level == ThreatLevel.CRITICAL]
        if critical_threats:
            return False
        
        # Fail if too many high-level threats
        high_threats = [t for t in result.threats_detected if t.threat_level == ThreatLevel.HIGH]
        if len(high_threats) > 2:
            return False
        
        # Fail if security score is too low
        if result.security_score < 60.0:
            return False
        
        return True
    
    def _generate_security_recommendations(self, result: SecurityValidationResult) -> List[str]:
        """Generate security recommendations based on validation result"""
        recommendations = []
        
        # General recommendations based on overall threat level
        if result.overall_threat_level == ThreatLevel.CRITICAL:
            recommendations.append("URGENT: Address critical security issues before proceeding")
            recommendations.append("Consider security review with expert consultation")
        elif result.overall_threat_level == ThreatLevel.HIGH:
            recommendations.append("Address high-priority security concerns")
            recommendations.append("Implement additional security controls")
        
        # Quantum-specific recommendations
        if result.quantum_entropy_level < 0.5:
            recommendations.append("Increase quantum entropy through better task definition")
        
        # Score-based recommendations
        if result.security_score < 80:
            recommendations.append("Improve overall security posture")
            recommendations.append("Implement security testing in CI/CD pipeline")
        
        # Threat-specific recommendations
        threat_types = set(threat.threat_type for threat in result.threats_detected)
        if SecurityValidationType.CODE_INJECTION in threat_types:
            recommendations.append("Implement comprehensive input validation")
        if SecurityValidationType.AUTHENTICATION in threat_types:
            recommendations.append("Strengthen authentication mechanisms")
        if SecurityValidationType.QUANTUM_RESISTANCE in threat_types:
            recommendations.append("Plan migration to post-quantum cryptography")
        
        return recommendations
    
    def bulk_validate_tasks(self, quantum_tasks: List[QuantumTask]) -> Dict[str, SecurityValidationResult]:
        """Perform bulk security validation of multiple tasks"""
        results = {}
        
        self.logger.info(f"Starting bulk security validation for {len(quantum_tasks)} tasks")
        
        for task in quantum_tasks:
            try:
                result = self.validate_quantum_task_security(task)
                results[task.id] = result
            except Exception as e:
                self.logger.error(f"Bulk validation failed for task {task.id}: {str(e)}")
                # Create error result
                error_result = SecurityValidationResult(
                    task_id=task.id,
                    overall_threat_level=ThreatLevel.HIGH,
                    validation_passed=False,
                    security_score=0.0,
                    recommendations=[f"Validation error: {str(e)}"]
                )
                results[task.id] = error_result
        
        self.logger.info(f"Bulk security validation completed for {len(results)} tasks")
        return results
    
    def generate_security_report(self, validation_results: Dict[str, SecurityValidationResult]) -> Dict:
        """Generate comprehensive security report"""
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_tasks_validated": len(validation_results),
            "overall_security_status": "UNKNOWN",
            "threat_summary": {},
            "score_distribution": {},
            "top_recommendations": [],
            "quantum_security_metrics": {},
            "task_details": {}
        }
        
        # Calculate summary statistics
        passed_count = sum(1 for result in validation_results.values() if result.validation_passed)
        failed_count = len(validation_results) - passed_count
        
        # Threat level distribution
        threat_levels = [result.overall_threat_level for result in validation_results.values()]
        for level in ThreatLevel:
            report["threat_summary"][level.value] = threat_levels.count(level)
        
        # Score distribution
        scores = [result.security_score for result in validation_results.values()]
        report["score_distribution"] = {
            "mean": sum(scores) / len(scores) if scores else 0,
            "min": min(scores) if scores else 0,
            "max": max(scores) if scores else 0,
            "below_60": sum(1 for s in scores if s < 60),
            "above_80": sum(1 for s in scores if s >= 80)
        }
        
        # Overall status
        critical_count = report["threat_summary"].get("critical", 0)
        high_count = report["threat_summary"].get("high", 0)
        
        if critical_count > 0:
            report["overall_security_status"] = "CRITICAL"
        elif high_count > 3:
            report["overall_security_status"] = "HIGH_RISK"
        elif failed_count > len(validation_results) * 0.3:
            report["overall_security_status"] = "MEDIUM_RISK"
        else:
            report["overall_security_status"] = "ACCEPTABLE"
        
        # Top recommendations
        all_recommendations = []
        for result in validation_results.values():
            all_recommendations.extend(result.recommendations)
        
        # Count recommendation frequency
        rec_counts = {}
        for rec in all_recommendations:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        report["top_recommendations"] = sorted(
            rec_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]
        
        # Quantum security metrics
        entropy_levels = [result.quantum_entropy_level for result in validation_results.values()]
        report["quantum_security_metrics"] = {
            "average_entropy": sum(entropy_levels) / len(entropy_levels) if entropy_levels else 0,
            "low_entropy_tasks": sum(1 for e in entropy_levels if e < 0.5),
            "high_entropy_tasks": sum(1 for e in entropy_levels if e > 0.8)
        }
        
        # Task details (summary only)
        for task_id, result in validation_results.items():
            report["task_details"][task_id] = {
                "threat_level": result.overall_threat_level.value,
                "security_score": result.security_score,
                "validation_passed": result.validation_passed,
                "threats_count": len(result.threats_detected),
                "quantum_entropy": result.quantum_entropy_level
            }
        
        return report
    
    def save_security_report(self, report: Dict) -> Path:
        """Save security report to file"""
        reports_dir = self.repo_root / "security_reports"
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f"quantum_security_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save as latest
        latest_file = reports_dir / "latest_quantum_security_report.json"
        with open(latest_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Security report saved to: {report_file}")
        return report_file


def main():
    """CLI entry point for quantum security validator"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quantum_security_validator.py <command>")
        print("Commands: validate-all, report, check <task_id>")
        return
    
    command = sys.argv[1]
    validator = QuantumSecurityValidator()
    
    if command == "validate-all":
        print("🔒 Running comprehensive security validation...")
        planner = QuantumTaskPlanner()
        planner.initialize_quantum_system()
        
        tasks = list(planner.quantum_tasks.values())
        results = validator.bulk_validate_tasks(tasks)
        
        report = validator.generate_security_report(results)
        report_file = validator.save_security_report(report)
        
        print(f"✅ Security validation completed")
        print(f"📊 Report saved to: {report_file}")
        print(f"🛡️  Overall status: {report['overall_security_status']}")
        
    elif command == "report":
        print("📊 Generating security report from cached results...")
        # This would load previous results and generate report
        print("Feature not implemented - run 'validate-all' first")
        
    elif command == "check" and len(sys.argv) > 2:
        task_id = sys.argv[2]
        print(f"🔍 Checking security for task: {task_id}")
        
        planner = QuantumTaskPlanner()
        planner.initialize_quantum_system()
        
        if task_id in planner.quantum_tasks:
            task = planner.quantum_tasks[task_id]
            result = validator.validate_quantum_task_security(task)
            print(json.dumps({
                "task_id": result.task_id,
                "threat_level": result.overall_threat_level.value,
                "security_score": result.security_score,
                "validation_passed": result.validation_passed,
                "threats_detected": len(result.threats_detected),
                "recommendations": result.recommendations
            }, indent=2))
        else:
            print(f"❌ Task not found: {task_id}")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()