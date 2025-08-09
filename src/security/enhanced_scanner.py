#!/usr/bin/env python3
"""
Enhanced Security Scanner for ADO
Comprehensive security analysis with multiple scan types and reporting
"""

import json
import logging
import subprocess
import tempfile
import hashlib
import re
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from enum import Enum


class SeverityLevel(Enum):
    """Security issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ScanType(Enum):
    """Types of security scans"""
    STATIC_ANALYSIS = "static_analysis"
    DEPENDENCY_CHECK = "dependency_check"
    SECRET_SCAN = "secret_scan"
    CONTAINER_SCAN = "container_scan"
    CONFIGURATION_CHECK = "configuration_check"
    LICENSE_CHECK = "license_check"


@dataclass
class SecurityFinding:
    """Individual security finding"""
    id: str
    title: str
    description: str
    severity: SeverityLevel
    scan_type: ScanType
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    rule_id: Optional[str] = None
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None
    remediation: Optional[str] = None
    confidence: float = 1.0
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ScanResult:
    """Results from a security scan"""
    scan_type: ScanType
    timestamp: str
    duration_seconds: float
    findings: List[SecurityFinding]
    total_files_scanned: int = 0
    scan_successful: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EnhancedSecurityScanner:
    """Comprehensive security scanner with multiple scan engines"""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.logger = logging.getLogger("enhanced_security_scanner")
        self.scan_results: List[ScanResult] = []
        self.lock = threading.RLock()
        
        # Secret patterns for detection
        self.secret_patterns = {
            'aws_access_key': r'AKIA[0-9A-Z]{16}',
            'aws_secret_key': r'[0-9a-zA-Z/+=]{40}',
            'github_token': r'ghp_[0-9a-zA-Z]{36}',
            'openai_key': r'sk-[0-9a-zA-Z]{48}',
            'jwt_token': r'eyJ[0-9a-zA-Z_-]*\.[0-9a-zA-Z_-]*\.[0-9a-zA-Z_-]*',
            'private_key': r'-----BEGIN [A-Z]+ PRIVATE KEY-----',
            'password_in_url': r'[a-zA-Z]{3,10}://[^/\s:@]{3,20}:[^/\s:@]{3,20}@.{1,100}',
            'api_key_generic': r'[aA][pP][iI][_-]?[kK][eE][yY][\s]*[=:][\s]*[\"\']?[0-9a-zA-Z]{16,}[\"\']?',
        }
        
        # File extensions to scan
        self.scannable_extensions = {
            '.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c', '.h',
            '.php', '.rb', '.cs', '.scala', '.kt', '.swift', '.dart',
            '.yaml', '.yml', '.json', '.xml', '.ini', '.cfg', '.conf',
            '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
            '.sql', '.md', '.txt', '.env', '.properties'
        }
        
    def run_comprehensive_scan(self) -> Dict[str, ScanResult]:
        """Run comprehensive security scan with all available scanners"""
        scan_results = {}
        
        # Define scan tasks
        scan_tasks = [
            (self._run_static_analysis, ScanType.STATIC_ANALYSIS),
            (self._run_dependency_check, ScanType.DEPENDENCY_CHECK),
            (self._run_secret_scan, ScanType.SECRET_SCAN),
            (self._run_configuration_check, ScanType.CONFIGURATION_CHECK),
            (self._run_license_check, ScanType.LICENSE_CHECK),
        ]
        
        # Check for Docker files and add container scan if found
        if self._has_docker_files():
            scan_tasks.append((self._run_container_scan, ScanType.CONTAINER_SCAN))
            
        # Run scans in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_scan = {executor.submit(scan_func): scan_type 
                             for scan_func, scan_type in scan_tasks}
            
            for future in as_completed(future_to_scan):
                scan_type = future_to_scan[future]
                try:
                    result = future.result()
                    scan_results[scan_type.value] = result
                    self.logger.info(f"Completed {scan_type.value} scan")
                except Exception as e:
                    self.logger.error(f"Failed to run {scan_type.value} scan: {e}")
                    scan_results[scan_type.value] = ScanResult(
                        scan_type=scan_type,
                        timestamp=datetime.now().isoformat(),
                        duration_seconds=0.0,
                        findings=[],
                        scan_successful=False,
                        error_message=str(e)
                    )
                    
        with self.lock:
            self.scan_results.extend(scan_results.values())
            
        return scan_results
        
    def _run_static_analysis(self) -> ScanResult:
        """Run static code analysis using bandit for Python files"""
        start_time = datetime.now()
        findings = []
        files_scanned = 0
        
        try:
            # Find Python files
            python_files = list(self.repo_root.rglob("*.py"))
            if not python_files:
                return ScanResult(
                    scan_type=ScanType.STATIC_ANALYSIS,
                    timestamp=start_time.isoformat(),
                    duration_seconds=0.0,
                    findings=[],
                    total_files_scanned=0
                )
                
            files_scanned = len(python_files)
            
            # Run bandit if available
            try:
                cmd = ['bandit', '-r', str(self.repo_root), '-f', 'json', '--quiet']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode in [0, 1]:  # 0 = no issues, 1 = issues found
                    try:
                        bandit_output = json.loads(result.stdout)
                        
                        for issue in bandit_output.get('results', []):
                            finding = SecurityFinding(
                                id=f"bandit_{issue['test_id']}_{hashlib.md5(issue['filename'].encode()).hexdigest()[:8]}",
                                title=issue['test_name'],
                                description=issue['issue_text'],
                                severity=self._map_bandit_severity(issue['issue_severity']),
                                scan_type=ScanType.STATIC_ANALYSIS,
                                file_path=str(Path(issue['filename']).relative_to(self.repo_root)),
                                line_number=issue['line_number'],
                                rule_id=issue['test_id'],
                                cwe_id=issue.get('cwe', {}).get('id'),
                                confidence=self._map_bandit_confidence(issue['issue_confidence']),
                                remediation=self._get_bandit_remediation(issue['test_id'])
                            )
                            findings.append(finding)
                            
                    except json.JSONDecodeError:
                        self.logger.warning("Could not parse bandit JSON output")
                        
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Bandit not available or timed out, run basic pattern analysis
                findings.extend(self._run_basic_static_analysis(python_files))
                
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return ScanResult(
                scan_type=ScanType.STATIC_ANALYSIS,
                timestamp=start_time.isoformat(),
                duration_seconds=duration,
                findings=[],
                total_files_scanned=files_scanned,
                scan_successful=False,
                error_message=str(e)
            )
            
        duration = (datetime.now() - start_time).total_seconds()
        return ScanResult(
            scan_type=ScanType.STATIC_ANALYSIS,
            timestamp=start_time.isoformat(),
            duration_seconds=duration,
            findings=findings,
            total_files_scanned=files_scanned
        )
        
    def _run_basic_static_analysis(self, files: List[Path]) -> List[SecurityFinding]:
        """Run basic static analysis using pattern matching"""
        findings = []
        
        dangerous_patterns = {
            'eval_usage': (r'\beval\s*\(', 'Use of eval() function is dangerous'),
            'exec_usage': (r'\bexec\s*\(', 'Use of exec() function is dangerous'),
            'shell_true': (r'shell\s*=\s*True', 'shell=True in subprocess is risky'),
            'hardcoded_secret': (r'(password|secret|key)\s*=\s*[\"\'][^\"\'{\}]+[\"\']', 'Possible hardcoded secret'),
            'sql_injection': (r'(SELECT|INSERT|UPDATE|DELETE).*%s', 'Possible SQL injection vulnerability'),
        }
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = content.splitlines()
                    
                for line_num, line in enumerate(lines, 1):
                    for pattern_id, (pattern, description) in dangerous_patterns.items():
                        if re.search(pattern, line, re.IGNORECASE):
                            finding = SecurityFinding(
                                id=f"static_{pattern_id}_{hashlib.md5(f'{file_path}:{line_num}'.encode()).hexdigest()[:8]}",
                                title=f"Potential security issue: {pattern_id}",
                                description=description,
                                severity=SeverityLevel.MEDIUM,
                                scan_type=ScanType.STATIC_ANALYSIS,
                                file_path=str(file_path.relative_to(self.repo_root)),
                                line_number=line_num,
                                confidence=0.7
                            )
                            findings.append(finding)
                            
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {e}")
                
        return findings
        
    def _run_dependency_check(self) -> ScanResult:
        """Run dependency vulnerability check"""
        start_time = datetime.now()
        findings = []
        files_scanned = 0
        
        try:
            # Check for requirements files
            req_files = list(self.repo_root.glob("*requirements*.txt"))
            req_files.extend(self.repo_root.glob("pyproject.toml"))
            req_files.extend(self.repo_root.glob("package.json"))
            req_files.extend(self.repo_root.glob("Gemfile"))
            
            files_scanned = len(req_files)
            
            # Try safety check for Python dependencies
            if any(f.name.endswith(('.txt', '.toml')) for f in req_files):
                try:
                    cmd = ['safety', 'check', '--json']
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=self.repo_root)
                    
                    if result.returncode != 0 and result.stdout:
                        try:
                            safety_output = json.loads(result.stdout)
                            
                            for vuln in safety_output:
                                finding = SecurityFinding(
                                    id=f"safety_{vuln['id']}_{hashlib.md5(vuln['package_name'].encode()).hexdigest()[:8]}",
                                    title=f"Vulnerable dependency: {vuln['package_name']}",
                                    description=f"Vulnerability in {vuln['package_name']} {vuln['installed_version']}: {vuln['vulnerability_description']}",
                                    severity=SeverityLevel.HIGH,
                                    scan_type=ScanType.DEPENDENCY_CHECK,
                                    cwe_id=vuln.get('cwe'),
                                    remediation=f"Upgrade to version {vuln.get('safe_version', 'latest')}"
                                )
                                findings.append(finding)
                                
                        except json.JSONDecodeError:
                            pass
                            
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    # Safety not available, run basic dependency analysis
                    findings.extend(self._run_basic_dependency_check(req_files))
                    
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return ScanResult(
                scan_type=ScanType.DEPENDENCY_CHECK,
                timestamp=start_time.isoformat(),
                duration_seconds=duration,
                findings=[],
                total_files_scanned=files_scanned,
                scan_successful=False,
                error_message=str(e)
            )
            
        duration = (datetime.now() - start_time).total_seconds()
        return ScanResult(
            scan_type=ScanType.DEPENDENCY_CHECK,
            timestamp=start_time.isoformat(),
            duration_seconds=duration,
            findings=findings,
            total_files_scanned=files_scanned
        )
        
    def _run_secret_scan(self) -> ScanResult:
        """Run secret detection scan"""
        start_time = datetime.now()
        findings = []
        files_scanned = 0
        
        try:
            # Get all scannable files
            scannable_files = []
            for ext in self.scannable_extensions:
                scannable_files.extend(self.repo_root.rglob(f"*{ext}"))
                
            # Remove files in common ignore directories
            ignore_dirs = {'.git', '__pycache__', 'node_modules', '.pytest_cache', 'venv', 'env', '.venv'}
            scannable_files = [f for f in scannable_files 
                             if not any(part in ignore_dirs for part in f.parts)]
                             
            files_scanned = len(scannable_files)
            
            for file_path in scannable_files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = content.splitlines()
                        
                    for line_num, line in enumerate(lines, 1):
                        for secret_type, pattern in self.secret_patterns.items():
                            matches = re.finditer(pattern, line, re.IGNORECASE)
                            for match in matches:
                                # Skip likely false positives
                                if self._is_likely_false_positive(line, match.group()):
                                    continue
                                    
                                finding = SecurityFinding(
                                    id=f"secret_{secret_type}_{hashlib.md5(f'{file_path}:{line_num}:{match.start()}'.encode()).hexdigest()[:8]}",
                                    title=f"Potential {secret_type.replace('_', ' ')} detected",
                                    description=f"Potential {secret_type.replace('_', ' ')} found in {file_path.name}",
                                    severity=SeverityLevel.HIGH,
                                    scan_type=ScanType.SECRET_SCAN,
                                    file_path=str(file_path.relative_to(self.repo_root)),
                                    line_number=line_num,
                                    confidence=0.8,
                                    remediation="Remove hardcoded secrets and use environment variables or secure vault"
                                )
                                findings.append(finding)
                                
                except Exception as e:
                    self.logger.warning(f"Could not scan {file_path} for secrets: {e}")
                    
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return ScanResult(
                scan_type=ScanType.SECRET_SCAN,
                timestamp=start_time.isoformat(),
                duration_seconds=duration,
                findings=[],
                total_files_scanned=files_scanned,
                scan_successful=False,
                error_message=str(e)
            )
            
        duration = (datetime.now() - start_time).total_seconds()
        return ScanResult(
            scan_type=ScanType.SECRET_SCAN,
            timestamp=start_time.isoformat(),
            duration_seconds=duration,
            findings=findings,
            total_files_scanned=files_scanned
        )
        
    def _run_configuration_check(self) -> ScanResult:
        """Run configuration security check"""
        start_time = datetime.now()
        findings = []
        files_scanned = 0
        
        try:
            config_files = []
            config_patterns = ['*.yaml', '*.yml', '*.json', '*.ini', '*.cfg', '*.conf', '*.env']
            
            for pattern in config_patterns:
                config_files.extend(self.repo_root.rglob(pattern))
                
            files_scanned = len(config_files)
            
            insecure_configs = {
                'debug_true': (r'debug\s*[=:]\s*true', 'Debug mode enabled in configuration'),
                'allow_all_origins': (r'allow_origins?\s*[=:]\s*[\"\']\*[\"\']', 'CORS allows all origins'),
                'insecure_ssl': (r'ssl_verify\s*[=:]\s*false|verify_ssl\s*[=:]\s*false', 'SSL verification disabled'),
                'weak_secret_key': (r'secret_key\s*[=:]\s*[\"\'][a-z]{1,8}[\"\']', 'Weak secret key detected'),
            }
            
            for config_file in config_files:
                try:
                    with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = content.splitlines()
                        
                    for line_num, line in enumerate(lines, 1):
                        for config_id, (pattern, description) in insecure_configs.items():
                            if re.search(pattern, line, re.IGNORECASE):
                                finding = SecurityFinding(
                                    id=f"config_{config_id}_{hashlib.md5(f'{config_file}:{line_num}'.encode()).hexdigest()[:8]}",
                                    title=f"Insecure configuration: {config_id}",
                                    description=description,
                                    severity=SeverityLevel.MEDIUM,
                                    scan_type=ScanType.CONFIGURATION_CHECK,
                                    file_path=str(config_file.relative_to(self.repo_root)),
                                    line_number=line_num,
                                    confidence=0.8
                                )
                                findings.append(finding)
                                
                except Exception as e:
                    self.logger.warning(f"Could not scan {config_file}: {e}")
                    
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            return ScanResult(
                scan_type=ScanType.CONFIGURATION_CHECK,
                timestamp=start_time.isoformat(),
                duration_seconds=duration,
                findings=[],
                total_files_scanned=files_scanned,
                scan_successful=False,
                error_message=str(e)
            )
            
        duration = (datetime.now() - start_time).total_seconds()
        return ScanResult(
            scan_type=ScanType.CONFIGURATION_CHECK,
            timestamp=start_time.isoformat(),
            duration_seconds=duration,
            findings=findings,
            total_files_scanned=files_scanned
        )
        
    def _run_container_scan(self) -> ScanResult:
        """Run container security scan"""
        start_time = datetime.now()
        findings = []
        
        docker_files = list(self.repo_root.rglob("Dockerfile*"))
        docker_compose_files = list(self.repo_root.rglob("docker-compose*.yml"))
        docker_compose_files.extend(self.repo_root.rglob("docker-compose*.yaml"))
        
        all_files = docker_files + docker_compose_files
        files_scanned = len(all_files)
        
        # Basic Docker security checks
        docker_issues = {
            'run_as_root': (r'^USER\s+root|^USER\s+0', 'Container runs as root user'),
            'privileged_mode': (r'privileged:\s*true', 'Privileged mode enabled'),
            'host_network': (r'network_mode:\s*[\"\']host[\"\']', 'Host network mode used'),
            'latest_tag': (r'FROM\s+[^:\s]+:latest', 'Using latest tag is not recommended'),
        }
        
        for docker_file in all_files:
            try:
                with open(docker_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for line_num, line in enumerate(lines, 1):
                    for issue_id, (pattern, description) in docker_issues.items():
                        if re.search(pattern, line, re.IGNORECASE):
                            finding = SecurityFinding(
                                id=f"docker_{issue_id}_{hashlib.md5(f'{docker_file}:{line_num}'.encode()).hexdigest()[:8]}",
                                title=f"Docker security issue: {issue_id}",
                                description=description,
                                severity=SeverityLevel.MEDIUM,
                                scan_type=ScanType.CONTAINER_SCAN,
                                file_path=str(docker_file.relative_to(self.repo_root)),
                                line_number=line_num,
                                confidence=0.9
                            )
                            findings.append(finding)
                            
            except Exception as e:
                self.logger.warning(f"Could not scan {docker_file}: {e}")
                
        duration = (datetime.now() - start_time).total_seconds()
        return ScanResult(
            scan_type=ScanType.CONTAINER_SCAN,
            timestamp=start_time.isoformat(),
            duration_seconds=duration,
            findings=findings,
            total_files_scanned=files_scanned
        )
        
    def _run_license_check(self) -> ScanResult:
        """Run license compliance check"""
        start_time = datetime.now()
        findings = []
        
        # Check for license files
        license_files = list(self.repo_root.glob("LICENSE*"))
        license_files.extend(self.repo_root.glob("COPYING*"))
        
        if not license_files:
            finding = SecurityFinding(
                id="license_missing",
                title="Missing license file",
                description="No license file found in repository",
                severity=SeverityLevel.LOW,
                scan_type=ScanType.LICENSE_CHECK,
                confidence=1.0
            )
            findings.append(finding)
            
        duration = (datetime.now() - start_time).total_seconds()
        return ScanResult(
            scan_type=ScanType.LICENSE_CHECK,
            timestamp=start_time.isoformat(),
            duration_seconds=duration,
            findings=findings,
            total_files_scanned=len(license_files)
        )
        
    def _has_docker_files(self) -> bool:
        """Check if repository has Docker files"""
        docker_files = list(self.repo_root.rglob("Dockerfile*"))
        docker_compose_files = list(self.repo_root.rglob("docker-compose*.yml"))
        docker_compose_files.extend(self.repo_root.rglob("docker-compose*.yaml"))
        return len(docker_files + docker_compose_files) > 0
        
    def _is_likely_false_positive(self, line: str, match: str) -> bool:
        """Check if a secret match is likely a false positive"""
        # Skip if it looks like a template or example
        if any(word in line.lower() for word in ['example', 'template', 'placeholder', 'xxx', 'todo']):
            return True
            
        # Skip if it's in a comment
        if re.search(r'^\s*[#//]', line.strip()):
            return True
            
        # Skip very short matches
        if len(match.strip('"\'\'`')) < 8:
            return True
            
        return False
        
    def _map_bandit_severity(self, severity: str) -> SeverityLevel:
        """Map bandit severity to our severity levels"""
        mapping = {
            'LOW': SeverityLevel.LOW,
            'MEDIUM': SeverityLevel.MEDIUM,
            'HIGH': SeverityLevel.HIGH
        }
        return mapping.get(severity.upper(), SeverityLevel.MEDIUM)
        
    def _map_bandit_confidence(self, confidence: str) -> float:
        """Map bandit confidence to numeric value"""
        mapping = {
            'LOW': 0.3,
            'MEDIUM': 0.6,
            'HIGH': 0.9
        }
        return mapping.get(confidence.upper(), 0.5)
        
    def _get_bandit_remediation(self, test_id: str) -> str:
        """Get remediation advice for bandit test"""
        remediations = {
            'B101': 'Remove assert statements or use proper error handling',
            'B102': 'Use specific exception types instead of bare except',
            'B108': 'Use secure temporary file creation methods',
            'B110': 'Avoid using try/except/pass without logging',
            'B201': 'Use flask.escape() instead of Markup',
            'B301': 'Use safer alternatives to pickle',
            'B501': 'Remove or secure requests with verify=False',
            'B506': 'Use yaml.safe_load() instead of yaml.load()',
            'B601': 'Avoid shell=True in subprocess calls',
            'B602': 'Use subprocess with shell=False and list arguments',
            'B608': 'Use parameterized queries to prevent SQL injection'
        }
        return remediations.get(test_id, 'Review code for security implications')
        
    def _run_basic_dependency_check(self, req_files: List[Path]) -> List[SecurityFinding]:
        """Run basic dependency vulnerability check"""
        findings = []
        
        # Known vulnerable package patterns (simplified)
        vulnerable_packages = {
            'pillow': ('< 8.3.2', 'PIL library vulnerability'),
            'urllib3': ('< 1.26.5', 'urllib3 vulnerability'),
            'requests': ('< 2.25.1', 'Requests library vulnerability'),
        }
        
        for req_file in req_files:
            if req_file.name.endswith('.txt'):
                try:
                    with open(req_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if line and not line.startswith('#'):
                                package_name = line.split('==')[0].split('>=')[0].split('<=')[0].lower()
                                if package_name in vulnerable_packages:
                                    version_constraint, description = vulnerable_packages[package_name]
                                    finding = SecurityFinding(
                                        id=f"dep_{package_name}_{hashlib.md5(line.encode()).hexdigest()[:8]}",
                                        title=f"Potentially vulnerable dependency: {package_name}",
                                        description=f"{description}. Check if version satisfies {version_constraint}",
                                        severity=SeverityLevel.MEDIUM,
                                        scan_type=ScanType.DEPENDENCY_CHECK,
                                        file_path=str(req_file.relative_to(self.repo_root)),
                                        line_number=line_num,
                                        confidence=0.6
                                    )
                                    findings.append(finding)
                except Exception as e:
                    self.logger.warning(f"Could not analyze {req_file}: {e}")
                    
        return findings
        
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        all_findings = []
        for result in self.scan_results:
            all_findings.extend(result.findings)
            
        # Categorize by severity
        severity_counts = {severity.value: 0 for severity in SeverityLevel}
        for finding in all_findings:
            severity_counts[finding.severity.value] += 1
            
        # Categorize by scan type
        scan_type_counts = {}
        for result in self.scan_results:
            scan_type_counts[result.scan_type.value] = len(result.findings)
            
        # Calculate risk score (weighted by severity)
        severity_weights = {
            SeverityLevel.CRITICAL: 10,
            SeverityLevel.HIGH: 7,
            SeverityLevel.MEDIUM: 4,
            SeverityLevel.LOW: 2,
            SeverityLevel.INFO: 1
        }
        
        risk_score = sum(severity_weights[finding.severity] for finding in all_findings)
        
        # Generate recommendations
        recommendations = self._generate_security_recommendations(all_findings)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_findings': len(all_findings),
            'severity_distribution': severity_counts,
            'scan_type_distribution': scan_type_counts,
            'risk_score': risk_score,
            'scan_results': [asdict(result) for result in self.scan_results],
            'top_findings': [asdict(f) for f in sorted(all_findings, 
                           key=lambda x: severity_weights[x.severity], reverse=True)[:10]],
            'recommendations': recommendations
        }
        
    def _generate_security_recommendations(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate security recommendations based on findings"""
        recommendations = set()
        
        # Count finding types
        finding_types = {}
        for finding in findings:
            key = f"{finding.scan_type.value}_{finding.severity.value}"
            finding_types[key] = finding_types.get(key, 0) + 1
            
        # High-level recommendations based on findings
        if any('secret_' in f.id for f in findings):
            recommendations.add("Implement secret management using environment variables or secure vaults")
            recommendations.add("Add pre-commit hooks to prevent secret commits")
            
        if any(f.severity == SeverityLevel.CRITICAL for f in findings):
            recommendations.add("Address critical security issues immediately")
            
        if any(f.scan_type == ScanType.DEPENDENCY_CHECK for f in findings):
            recommendations.add("Regularly update dependencies and monitor for vulnerabilities")
            recommendations.add("Consider using automated dependency scanning in CI/CD")
            
        if any(f.scan_type == ScanType.STATIC_ANALYSIS for f in findings):
            recommendations.add("Implement static code analysis in development workflow")
            
        if any(f.scan_type == ScanType.CONTAINER_SCAN for f in findings):
            recommendations.add("Follow Docker security best practices")
            recommendations.add("Use minimal base images and non-root users")
            
        return list(recommendations)
        
    def save_scan_results(self, output_file: Optional[Path] = None) -> Path:
        """Save scan results to file"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.repo_root / "security_reports" / f"security_scan_{timestamp}.json"
            
        output_file.parent.mkdir(exist_ok=True)
        
        report = self.generate_security_report()
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Save as latest
        latest_file = output_file.parent / "security_scan_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        return output_file


def main():
    """CLI entry point for security scanner"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_scanner.py <command> [options]")
        print("Commands:")
        print("  scan - Run comprehensive security scan")
        print("  report - Generate security report from existing results")
        return
        
    command = sys.argv[1]
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                       
    scanner = EnhancedSecurityScanner()
    
    if command == "scan":
        print("🔍 Running comprehensive security scan...")
        results = scanner.run_comprehensive_scan()
        
        output_file = scanner.save_scan_results()
        print(f"📊 Scan complete. Results saved to: {output_file}")
        
        # Print summary
        report = scanner.generate_security_report()
        print(f"\n📈 Security Summary:")
        print(f"  Total findings: {report['total_findings']}")
        print(f"  Risk score: {report['risk_score']}")
        
        for severity, count in report['severity_distribution'].items():
            if count > 0:
                print(f"  {severity.capitalize()}: {count}")
                
    elif command == "report":
        report = scanner.generate_security_report()
        print(json.dumps(report, indent=2, default=str))
        
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
