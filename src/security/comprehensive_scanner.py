#!/usr/bin/env python3
"""
Comprehensive Security Scanner
Advanced security scanning and vulnerability detection system
"""

import json
import yaml
import subprocess
import time
import asyncio
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import re
import os
import sys
import tempfile
from abc import ABC, abstractmethod


class VulnerabilitySeverity(Enum):
    """Security vulnerability severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VulnerabilityCategory(Enum):
    """Categories of security vulnerabilities"""
    DEPENDENCY = "dependency"
    CODE_INJECTION = "code_injection"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CRYPTOGRAPHY = "cryptography"
    CONFIGURATION = "configuration"
    SECRETS = "secrets"
    XSS = "xss"
    CSRF = "csrf"
    PATH_TRAVERSAL = "path_traversal"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    SECURITY_MISCONFIGURATION = "security_misconfiguration"


@dataclass
class Vulnerability:
    """Security vulnerability finding"""
    id: str
    title: str
    description: str
    severity: VulnerabilitySeverity
    category: VulnerabilityCategory
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    remediation: Optional[str] = None
    references: List[str] = field(default_factory=list)
    confidence: float = 1.0
    scanner_source: str = "unknown"
    discovered_at: datetime = field(default_factory=datetime.now)
    
    @property
    def risk_score(self) -> float:
        """Calculate risk score based on severity and confidence"""
        severity_scores = {
            VulnerabilitySeverity.INFO: 1.0,
            VulnerabilitySeverity.LOW: 2.5,
            VulnerabilitySeverity.MEDIUM: 5.0,
            VulnerabilitySeverity.HIGH: 7.5,
            VulnerabilitySeverity.CRITICAL: 10.0
        }
        return severity_scores[self.severity] * self.confidence


@dataclass
class ScanResult:
    """Result of security scan"""
    scan_id: str
    scan_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    scan_status: str = "running"
    error_message: Optional[str] = None
    files_scanned: int = 0
    total_execution_time: float = 0.0
    
    @property
    def is_complete(self) -> bool:
        """Check if scan is complete"""
        return self.scan_status in ["completed", "failed"]
    
    @property
    def critical_count(self) -> int:
        """Count of critical vulnerabilities"""
        return sum(1 for v in self.vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL)
    
    @property
    def high_count(self) -> int:
        """Count of high severity vulnerabilities"""
        return sum(1 for v in self.vulnerabilities if v.severity == VulnerabilitySeverity.HIGH)
    
    @property
    def total_risk_score(self) -> float:
        """Total risk score for all vulnerabilities"""
        return sum(v.risk_score for v in self.vulnerabilities)


class SecurityScanner(ABC):
    """Abstract base class for security scanners"""
    
    @abstractmethod
    async def scan(self, target_path: Path, config: Dict[str, Any]) -> ScanResult:
        """Perform security scan on target"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if scanner tool is available"""
        pass


class BanditScanner(SecurityScanner):
    """Python security linter using Bandit"""
    
    def __init__(self):
        self.name = "bandit"
    
    def is_available(self) -> bool:
        """Check if Bandit is available"""
        try:
            result = subprocess.run(["bandit", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    async def scan(self, target_path: Path, config: Dict[str, Any]) -> ScanResult:
        """Run Bandit security scan"""
        scan_id = f"bandit_{int(time.time())}"
        start_time = datetime.now()
        
        result = ScanResult(
            scan_id=scan_id,
            scan_type="bandit",
            start_time=start_time
        )
        
        try:
            # Run Bandit
            cmd = [
                "bandit", "-r", str(target_path),
                "-f", "json",
                "--skip", "B101",  # Skip assert_used test
            ]
            
            # Add configuration options
            severity = config.get("bandit", {}).get("severity", "medium")
            if severity == "low":
                cmd.extend(["-l"])
            elif severity == "high":
                cmd.extend(["-i", "high"])
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0 or (stdout and len(stdout) > 0):
                # Parse Bandit JSON output
                try:
                    bandit_data = json.loads(stdout.decode())
                    result.vulnerabilities = self._parse_bandit_results(bandit_data)
                    result.files_scanned = len(bandit_data.get("metrics", {}).get("_totals", {}).get("loc", 0))
                    result.scan_status = "completed"
                except json.JSONDecodeError as e:
                    result.scan_status = "failed"
                    result.error_message = f"Failed to parse Bandit output: {e}"
            else:
                result.scan_status = "failed"
                result.error_message = stderr.decode() if stderr else "Bandit scan failed"
        
        except Exception as e:
            result.scan_status = "failed"
            result.error_message = str(e)
        
        result.end_time = datetime.now()
        result.total_execution_time = (result.end_time - start_time).total_seconds()
        
        return result
    
    def _parse_bandit_results(self, bandit_data: Dict) -> List[Vulnerability]:
        """Parse Bandit JSON output into Vulnerability objects"""
        vulnerabilities = []
        
        for issue in bandit_data.get("results", []):
            severity_map = {
                "LOW": VulnerabilitySeverity.LOW,
                "MEDIUM": VulnerabilitySeverity.MEDIUM,
                "HIGH": VulnerabilitySeverity.HIGH
            }
            
            category_map = {
                "B102": VulnerabilityCategory.CODE_INJECTION,
                "B103": VulnerabilityCategory.PATH_TRAVERSAL,
                "B301": VulnerabilityCategory.INSECURE_DESERIALIZATION,
                "B506": VulnerabilityCategory.AUTHENTICATION,
                "B601": VulnerabilityCategory.CODE_INJECTION,
            }
            
            vuln = Vulnerability(
                id=f"bandit_{issue.get('test_id', 'unknown')}_{hash(issue.get('filename', ''))}",
                title=issue.get("test_name", "Security Issue"),
                description=issue.get("issue_text", ""),
                severity=severity_map.get(issue.get("issue_severity", "MEDIUM"), VulnerabilitySeverity.MEDIUM),
                category=category_map.get(issue.get("test_id"), VulnerabilityCategory.SECURITY_MISCONFIGURATION),
                file_path=issue.get("filename"),
                line_number=issue.get("line_number"),
                code_snippet=issue.get("code"),
                confidence={"HIGH": 0.9, "MEDIUM": 0.7, "LOW": 0.5}.get(
                    issue.get("issue_confidence", "MEDIUM"), 0.7
                ),
                cwe_id=issue.get("cwe", {}).get("id"),
                scanner_source="bandit"
            )
            vulnerabilities.append(vuln)
        
        return vulnerabilities


class SafetyScanner(SecurityScanner):
    """Dependency vulnerability scanner using Safety"""
    
    def __init__(self):
        self.name = "safety"
    
    def is_available(self) -> bool:
        """Check if Safety is available"""
        try:
            result = subprocess.run(["safety", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False
    
    async def scan(self, target_path: Path, config: Dict[str, Any]) -> ScanResult:
        """Run Safety dependency scan"""
        scan_id = f"safety_{int(time.time())}"
        start_time = datetime.now()
        
        result = ScanResult(
            scan_id=scan_id,
            scan_type="safety",
            start_time=start_time
        )
        
        try:
            # Check for requirements files
            req_files = list(target_path.glob("*requirements*.txt"))
            req_files.extend(list(target_path.glob("pyproject.toml")))
            
            if not req_files:
                result.scan_status = "completed"
                result.end_time = datetime.now()
                result.total_execution_time = (result.end_time - start_time).total_seconds()
                return result
            
            # Run Safety check
            cmd = ["safety", "check", "--json"]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=target_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # Safety returns non-zero exit code when vulnerabilities are found
            if stdout:
                try:
                    safety_data = json.loads(stdout.decode())
                    result.vulnerabilities = self._parse_safety_results(safety_data)
                    result.scan_status = "completed"
                except json.JSONDecodeError:
                    # Safety might output plain text in some cases
                    result.vulnerabilities = self._parse_safety_text(stdout.decode())
                    result.scan_status = "completed"
            else:
                result.scan_status = "completed"  # No vulnerabilities found
        
        except Exception as e:
            result.scan_status = "failed"
            result.error_message = str(e)
        
        result.end_time = datetime.now()
        result.total_execution_time = (result.end_time - start_time).total_seconds()
        
        return result
    
    def _parse_safety_results(self, safety_data: List) -> List[Vulnerability]:
        """Parse Safety JSON output into Vulnerability objects"""
        vulnerabilities = []
        
        for item in safety_data:
            vuln = Vulnerability(
                id=f"safety_{item.get('id', 'unknown')}",
                title=f"Vulnerable dependency: {item.get('package', 'unknown')}",
                description=item.get("advisory", "Dependency vulnerability"),
                severity=VulnerabilitySeverity.HIGH,  # Default for dependency vulns
                category=VulnerabilityCategory.DEPENDENCY,
                remediation=f"Update {item.get('package')} to version {item.get('safe_versions', ['latest'])[0] if item.get('safe_versions') else 'latest'}",
                cvss_score=item.get("cvss"),
                scanner_source="safety"
            )
            vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _parse_safety_text(self, safety_text: str) -> List[Vulnerability]:
        """Parse Safety text output"""
        vulnerabilities = []
        
        # Simple text parsing for Safety output
        lines = safety_text.split('\n')
        for line in lines:
            if 'vulnerability' in line.lower() or 'cve' in line.lower():
                vuln = Vulnerability(
                    id=f"safety_text_{hash(line)}",
                    title="Dependency Vulnerability",
                    description=line.strip(),
                    severity=VulnerabilitySeverity.MEDIUM,
                    category=VulnerabilityCategory.DEPENDENCY,
                    scanner_source="safety"
                )
                vulnerabilities.append(vuln)
        
        return vulnerabilities


class SecretsScanner(SecurityScanner):
    """Scanner for secrets and API keys in code"""
    
    def __init__(self):
        self.name = "secrets"
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for secret detection"""
        return {
            'api_key': re.compile(r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?([a-zA-Z0-9]{20,})["\']?'),
            'aws_access_key': re.compile(r'AKIA[0-9A-Z]{16}'),
            'aws_secret_key': re.compile(r'(?i)aws[_-]?secret[_-]?access[_-]?key.*[:=]\s*["\']?([a-zA-Z0-9+/]{40})["\']?'),
            'github_token': re.compile(r'ghp_[a-zA-Z0-9]{36}'),
            'slack_token': re.compile(r'xox[baprs]-[a-zA-Z0-9-]{10,48}'),
            'jwt_token': re.compile(r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*'),
            'password': re.compile(r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']?([^"\'\s]{6,})["\']?'),
            'private_key': re.compile(r'-----BEGIN [A-Z]+ PRIVATE KEY-----'),
            'database_url': re.compile(r'(?i)(database_url|db_url)\s*[:=]\s*["\']?([^"\'\s]+)["\']?'),
            'credit_card': re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b')
        }
    
    def is_available(self) -> bool:
        """Secrets scanner is always available (built-in)"""
        return True
    
    async def scan(self, target_path: Path, config: Dict[str, Any]) -> ScanResult:
        """Scan for secrets in code files"""
        scan_id = f"secrets_{int(time.time())}"
        start_time = datetime.now()
        
        result = ScanResult(
            scan_id=scan_id,
            scan_type="secrets",
            start_time=start_time
        )
        
        try:
            # Get list of files to scan
            extensions = config.get("secrets", {}).get("extensions", [".py", ".js", ".ts", ".json", ".yaml", ".yml", ".env"])
            exclude_patterns = config.get("secrets", {}).get("exclude_patterns", ["test_", "mock_", "example_"])
            
            files_to_scan = []
            for ext in extensions:
                files_to_scan.extend(target_path.glob(f"**/*{ext}"))
            
            # Filter out excluded files
            filtered_files = []
            for file_path in files_to_scan:
                if not any(pattern in file_path.name for pattern in exclude_patterns):
                    filtered_files.append(file_path)
            
            result.files_scanned = len(filtered_files)
            
            # Scan files for secrets
            vulnerabilities = []
            for file_path in filtered_files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        file_vulns = self._scan_file_content(content, file_path)
                        vulnerabilities.extend(file_vulns)
                except Exception as e:
                    logging.warning(f"Could not scan file {file_path}: {e}")
            
            result.vulnerabilities = vulnerabilities
            result.scan_status = "completed"
        
        except Exception as e:
            result.scan_status = "failed"
            result.error_message = str(e)
        
        result.end_time = datetime.now()
        result.total_execution_time = (result.end_time - start_time).total_seconds()
        
        return result
    
    def _scan_file_content(self, content: str, file_path: Path) -> List[Vulnerability]:
        """Scan file content for secrets"""
        vulnerabilities = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for secret_type, pattern in self.patterns.items():
                matches = pattern.finditer(line)
                for match in matches:
                    # Calculate confidence based on context
                    confidence = self._calculate_confidence(secret_type, line, match)
                    
                    if confidence > 0.3:  # Minimum confidence threshold
                        vuln = Vulnerability(
                            id=f"secrets_{secret_type}_{file_path}_{line_num}",
                            title=f"Potential {secret_type.replace('_', ' ').title()} Exposed",
                            description=f"Potential {secret_type} found in code",
                            severity=self._get_severity(secret_type),
                            category=VulnerabilityCategory.SECRETS,
                            file_path=str(file_path),
                            line_number=line_num,
                            code_snippet=line.strip(),
                            confidence=confidence,
                            remediation=f"Remove or encrypt the {secret_type} and use environment variables or secure configuration",
                            scanner_source="secrets"
                        )
                        vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _calculate_confidence(self, secret_type: str, line: str, match: re.Match) -> float:
        """Calculate confidence level for secret detection"""
        confidence = 0.7  # Base confidence
        
        # Reduce confidence for test/example patterns
        if any(keyword in line.lower() for keyword in ['test', 'example', 'sample', 'demo', 'fake']):
            confidence *= 0.3
        
        # Reduce confidence for placeholder values
        matched_value = match.group(0).lower()
        if any(placeholder in matched_value for placeholder in ['xxx', '***', 'placeholder', 'your_key_here']):
            confidence *= 0.1
        
        # Increase confidence for specific patterns
        if secret_type == 'aws_access_key' and line.count('AKIA') == 1:
            confidence = 0.95
        elif secret_type == 'github_token' and 'ghp_' in matched_value:
            confidence = 0.9
        
        return min(confidence, 1.0)
    
    def _get_severity(self, secret_type: str) -> VulnerabilitySeverity:
        """Get severity level for different secret types"""
        high_risk_secrets = ['aws_access_key', 'aws_secret_key', 'github_token', 'private_key']
        medium_risk_secrets = ['api_key', 'database_url', 'jwt_token']
        
        if secret_type in high_risk_secrets:
            return VulnerabilitySeverity.HIGH
        elif secret_type in medium_risk_secrets:
            return VulnerabilitySeverity.MEDIUM
        else:
            return VulnerabilitySeverity.LOW


class ComprehensiveSecurityScanner:
    """Orchestrates multiple security scanners"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.scanners = self._initialize_scanners()
        self.scan_history: List[ScanResult] = []
        
        # Setup logging
        self.logger = logging.getLogger("security_scanner")
        
        # Metrics
        self.metrics = {
            "total_scans": 0,
            "total_vulnerabilities": 0,
            "critical_vulnerabilities": 0,
            "scan_execution_time": 0.0
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load scanner configuration"""
        default_config = {
            "security_scanning": {
                "enabled": True,
                "concurrent_scanners": 3,
                "timeout_seconds": 300,
                "scanners": {
                    "bandit": {
                        "enabled": True,
                        "severity": "medium"
                    },
                    "safety": {
                        "enabled": True
                    },
                    "secrets": {
                        "enabled": True,
                        "extensions": [".py", ".js", ".ts", ".json", ".yaml", ".yml", ".env"],
                        "exclude_patterns": ["test_", "mock_", "example_"]
                    }
                },
                "reporting": {
                    "save_reports": True,
                    "report_format": ["json", "markdown"],
                    "report_directory": ".terragon/security-reports"
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    if "security_scanning" in loaded_config:
                        default_config["security_scanning"].update(loaded_config["security_scanning"])
            except Exception as e:
                self.logger.warning(f"Could not load security scanning config: {e}")
        
        return default_config
    
    def _initialize_scanners(self) -> Dict[str, SecurityScanner]:
        """Initialize available security scanners"""
        scanners = {}
        scanner_config = self.config.get("security_scanning", {}).get("scanners", {})
        
        # Initialize Bandit scanner
        if scanner_config.get("bandit", {}).get("enabled", True):
            bandit = BanditScanner()
            if bandit.is_available():
                scanners["bandit"] = bandit
                self.logger.info("✅ Bandit scanner initialized")
            else:
                self.logger.warning("⚠️ Bandit not available")
        
        # Initialize Safety scanner
        if scanner_config.get("safety", {}).get("enabled", True):
            safety = SafetyScanner()
            if safety.is_available():
                scanners["safety"] = safety
                self.logger.info("✅ Safety scanner initialized")
            else:
                self.logger.warning("⚠️ Safety not available")
        
        # Initialize Secrets scanner
        if scanner_config.get("secrets", {}).get("enabled", True):
            secrets = SecretsScanner()
            scanners["secrets"] = secrets
            self.logger.info("✅ Secrets scanner initialized")
        
        return scanners
    
    async def scan_repository(self, target_path: Path) -> Dict[str, ScanResult]:
        """Run comprehensive security scan on repository"""
        self.logger.info(f"🔍 Starting comprehensive security scan of {target_path}")
        
        start_time = datetime.now()
        results = {}
        
        # Run scanners concurrently
        max_concurrent = self.config.get("security_scanning", {}).get("concurrent_scanners", 3)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_scanner(name: str, scanner: SecurityScanner) -> Tuple[str, ScanResult]:
            async with semaphore:
                self.logger.info(f"   Running {name} scanner...")
                result = await scanner.scan(target_path, self.config["security_scanning"]["scanners"])
                self.logger.info(f"   ✅ {name} completed: {len(result.vulnerabilities)} vulnerabilities found")
                return name, result
        
        # Create tasks for all enabled scanners
        tasks = [
            run_scanner(name, scanner) 
            for name, scanner in self.scanners.items()
        ]
        
        # Execute all scanners
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for task_result in completed_tasks:
            if isinstance(task_result, Exception):
                self.logger.error(f"Scanner failed: {task_result}")
            else:
                name, result = task_result
                results[name] = result
                self.scan_history.append(result)
        
        # Update metrics
        total_vulns = sum(len(result.vulnerabilities) for result in results.values())
        critical_vulns = sum(result.critical_count for result in results.values())
        
        self.metrics["total_scans"] += len(results)
        self.metrics["total_vulnerabilities"] += total_vulns
        self.metrics["critical_vulnerabilities"] += critical_vulns
        self.metrics["scan_execution_time"] += (datetime.now() - start_time).total_seconds()
        
        # Save reports if configured
        if self.config.get("security_scanning", {}).get("reporting", {}).get("save_reports", True):
            await self._save_scan_reports(results, target_path)
        
        self.logger.info(f"🔍 Security scan completed: {total_vulns} vulnerabilities found ({critical_vulns} critical)")
        
        return results
    
    async def _save_scan_reports(self, results: Dict[str, ScanResult], target_path: Path):
        """Save scan reports to disk"""
        report_config = self.config.get("security_scanning", {}).get("reporting", {})
        report_dir = Path(report_config.get("report_directory", ".terragon/security-reports"))
        report_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Aggregate all results
        all_vulnerabilities = []
        for result in results.values():
            all_vulnerabilities.extend(result.vulnerabilities)
        
        report_formats = report_config.get("report_format", ["json", "markdown"])
        
        # Save JSON report
        if "json" in report_formats:
            json_report = {
                "scan_timestamp": datetime.now().isoformat(),
                "target_path": str(target_path),
                "total_vulnerabilities": len(all_vulnerabilities),
                "critical_vulnerabilities": sum(1 for v in all_vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL),
                "high_vulnerabilities": sum(1 for v in all_vulnerabilities if v.severity == VulnerabilitySeverity.HIGH),
                "scanner_results": {
                    name: {
                        "scan_status": result.scan_status,
                        "vulnerabilities_found": len(result.vulnerabilities),
                        "execution_time": result.total_execution_time,
                        "files_scanned": result.files_scanned
                    }
                    for name, result in results.items()
                },
                "vulnerabilities": [
                    {
                        "id": v.id,
                        "title": v.title,
                        "description": v.description,
                        "severity": v.severity.value,
                        "category": v.category.value,
                        "file_path": v.file_path,
                        "line_number": v.line_number,
                        "risk_score": v.risk_score,
                        "scanner_source": v.scanner_source
                    }
                    for v in all_vulnerabilities
                ]
            }
            
            json_path = report_dir / f"security_scan_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(json_report, f, indent=2)
            
            # Also save as latest
            latest_path = report_dir / "security_scan_latest.json"
            with open(latest_path, 'w') as f:
                json.dump(json_report, f, indent=2)
        
        # Save Markdown report
        if "markdown" in report_formats:
            md_path = report_dir / f"security_scan_{timestamp}.md"
            await self._generate_markdown_report(results, all_vulnerabilities, md_path)
    
    async def _generate_markdown_report(self, results: Dict[str, ScanResult], 
                                      all_vulnerabilities: List[Vulnerability], 
                                      output_path: Path):
        """Generate comprehensive markdown security report"""
        
        # Count vulnerabilities by severity
        severity_counts = {
            VulnerabilitySeverity.CRITICAL: 0,
            VulnerabilitySeverity.HIGH: 0,
            VulnerabilitySeverity.MEDIUM: 0,
            VulnerabilitySeverity.LOW: 0,
            VulnerabilitySeverity.INFO: 0
        }
        
        for vuln in all_vulnerabilities:
            severity_counts[vuln.severity] += 1
        
        with open(output_path, 'w') as f:
            f.write(f"""# 🔒 Comprehensive Security Scan Report

**Generated**: {datetime.now().isoformat()}  
**Total Vulnerabilities**: {len(all_vulnerabilities)}  
**Total Risk Score**: {sum(v.risk_score for v in all_vulnerabilities):.1f}

## 📊 Executive Summary

| Severity | Count | Percentage |
|----------|-------|------------|
| 🔴 Critical | {severity_counts[VulnerabilitySeverity.CRITICAL]} | {(severity_counts[VulnerabilitySeverity.CRITICAL] / max(len(all_vulnerabilities), 1)) * 100:.1f}% |
| 🟠 High | {severity_counts[VulnerabilitySeverity.HIGH]} | {(severity_counts[VulnerabilitySeverity.HIGH] / max(len(all_vulnerabilities), 1)) * 100:.1f}% |
| 🟡 Medium | {severity_counts[VulnerabilitySeverity.MEDIUM]} | {(severity_counts[VulnerabilitySeverity.MEDIUM] / max(len(all_vulnerabilities), 1)) * 100:.1f}% |
| 🟢 Low | {severity_counts[VulnerabilitySeverity.LOW]} | {(severity_counts[VulnerabilitySeverity.LOW] / max(len(all_vulnerabilities), 1)) * 100:.1f}% |
| 🔵 Info | {severity_counts[VulnerabilitySeverity.INFO]} | {(severity_counts[VulnerabilitySeverity.INFO] / max(len(all_vulnerabilities), 1)) * 100:.1f}% |

## 🔍 Scanner Results

""")
            
            for name, result in results.items():
                status_emoji = "✅" if result.scan_status == "completed" else "❌"
                f.write(f"""### {status_emoji} {name.title()} Scanner

- **Status**: {result.scan_status}
- **Vulnerabilities Found**: {len(result.vulnerabilities)}
- **Files Scanned**: {result.files_scanned}
- **Execution Time**: {result.total_execution_time:.2f}s

""")
            
            # Top vulnerabilities
            if all_vulnerabilities:
                f.write("## 🚨 Top Vulnerabilities\n\n")
                
                # Sort by risk score
                sorted_vulns = sorted(all_vulnerabilities, key=lambda v: v.risk_score, reverse=True)
                
                for i, vuln in enumerate(sorted_vulns[:10], 1):  # Top 10
                    severity_emoji = {
                        VulnerabilitySeverity.CRITICAL: "🔴",
                        VulnerabilitySeverity.HIGH: "🟠",
                        VulnerabilitySeverity.MEDIUM: "🟡",
                        VulnerabilitySeverity.LOW: "🟢",
                        VulnerabilitySeverity.INFO: "🔵"
                    }[vuln.severity]
                    
                    f.write(f"""### {i}. {severity_emoji} {vuln.title}

- **Severity**: {vuln.severity.value.title()}
- **Category**: {vuln.category.value.replace('_', ' ').title()}
- **Risk Score**: {vuln.risk_score:.1f}
- **Scanner**: {vuln.scanner_source}
""")
                    
                    if vuln.file_path:
                        f.write(f"- **File**: {vuln.file_path}")
                        if vuln.line_number:
                            f.write(f":{vuln.line_number}")
                        f.write("\n")
                    
                    if vuln.description:
                        f.write(f"- **Description**: {vuln.description}\n")
                    
                    if vuln.remediation:
                        f.write(f"- **Remediation**: {vuln.remediation}\n")
                    
                    f.write("\n")
            
            f.write(f"""## 💡 Recommendations

""")
            
            if severity_counts[VulnerabilitySeverity.CRITICAL] > 0:
                f.write("- 🚨 **URGENT**: Address critical vulnerabilities immediately\n")
            
            if severity_counts[VulnerabilitySeverity.HIGH] > 0:
                f.write("- ⚠️ **HIGH PRIORITY**: Review and fix high severity vulnerabilities\n")
            
            if len(all_vulnerabilities) > 20:
                f.write("- 📊 **HIGH VOLUME**: Large number of vulnerabilities detected - consider implementing security-first development practices\n")
            
            f.write(f"""
- 🔄 **Regular Scanning**: Implement automated security scanning in CI/CD pipeline
- 🛡️ **Security Training**: Provide security awareness training for development team
- 📝 **Security Policies**: Establish and enforce secure coding guidelines

---

*Generated by Terragon Comprehensive Security Scanner*
""")
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics"""
        recent_scans = [s for s in self.scan_history if 
                       (datetime.now() - s.start_time).total_seconds() < 86400]  # Last 24 hours
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_scans_performed": self.metrics["total_scans"],
            "total_vulnerabilities_found": self.metrics["total_vulnerabilities"],
            "critical_vulnerabilities": self.metrics["critical_vulnerabilities"],
            "average_scan_time": self.metrics["scan_execution_time"] / max(self.metrics["total_scans"], 1),
            "recent_scans_24h": len(recent_scans),
            "available_scanners": list(self.scanners.keys()),
            "scanner_availability": {
                name: scanner.is_available() 
                for name, scanner in self.scanners.items()
            }
        }


async def main():
    """CLI entry point for security scanning"""
    import sys
    
    print("🔒 Comprehensive Security Scanner")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage: python comprehensive_scanner.py <target_path>")
        return
    
    target_path = Path(sys.argv[1])
    if not target_path.exists():
        print(f"❌ Target path does not exist: {target_path}")
        return
    
    # Initialize scanner
    scanner = ComprehensiveSecurityScanner()
    
    print(f"🔍 Scanning: {target_path}")
    print(f"📊 Available scanners: {', '.join(scanner.scanners.keys())}")
    print()
    
    # Run scan
    results = await scanner.scan_repository(target_path)
    
    # Display summary
    total_vulns = sum(len(result.vulnerabilities) for result in results.values())
    critical_vulns = sum(result.critical_count for result in results.values())
    
    print(f"\n📋 Scan Summary:")
    print(f"   Total Vulnerabilities: {total_vulns}")
    print(f"   Critical Vulnerabilities: {critical_vulns}")
    
    for name, result in results.items():
        print(f"   {name.title()}: {len(result.vulnerabilities)} vulnerabilities ({result.total_execution_time:.1f}s)")
    
    # Show metrics
    print(f"\n📊 Security Metrics:")
    metrics = scanner.get_security_metrics()
    for key, value in metrics.items():
        if key not in ["timestamp", "scanner_availability"]:
            print(f"   {key}: {value}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())