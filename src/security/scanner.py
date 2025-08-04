#!/usr/bin/env python3
"""
Security scanner for autonomous execution
"""

import re
import json
import hashlib
import subprocess
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import datetime
import logging


@dataclass
class SecurityFinding:
    """Security scan finding"""
    severity: str  # critical, high, medium, low, info
    rule_id: str
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    fix_suggestion: Optional[str] = None


@dataclass
class SecurityReport:
    """Security scan report"""
    timestamp: str
    scan_duration: float
    total_findings: int
    critical_findings: int
    high_findings: int
    medium_findings: int
    low_findings: int
    info_findings: int
    findings: List[SecurityFinding]
    scanned_files: List[str]


class SecurityScanner:
    """Comprehensive security scanner"""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.logger = logging.getLogger("security_scanner")
        self.rules = self._load_security_rules()
        
    def _load_security_rules(self) -> Dict:
        """Load security scanning rules"""
        return {
            "secrets": [
                {
                    "id": "SECRET_001",
                    "pattern": r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{8,}['\"]",
                    "severity": "high",
                    "title": "Hardcoded password detected",
                    "description": "Found potential hardcoded password in source code"
                },
                {
                    "id": "SECRET_002", 
                    "pattern": r"(?i)(api[_-]?key|apikey|access[_-]?token)\s*[=:]\s*['\"][^'\"]{20,}['\"]",
                    "severity": "critical",
                    "title": "API key or token detected",
                    "description": "Found potential API key or access token in source code"
                },
                {
                    "id": "SECRET_003",
                    "pattern": r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----",
                    "severity": "critical", 
                    "title": "Private key detected",
                    "description": "Found private key in source code"
                }
            ],
            "injection": [
                {
                    "id": "INJECT_001",
                    "pattern": r"os\.system\s*\([^)]*\+",
                    "severity": "high",
                    "title": "Potential command injection",
                    "description": "Using os.system with string concatenation"
                },
                {
                    "id": "INJECT_002",
                    "pattern": r"subprocess\.(call|run)\s*\([^)]*\+",
                    "severity": "medium",
                    "title": "Potential command injection in subprocess",
                    "description": "Using subprocess with string concatenation"
                },
                {
                    "id": "INJECT_003",
                    "pattern": r"eval\s*\([^)]*input",
                    "severity": "critical",
                    "title": "Code injection via eval",
                    "description": "Using eval() with user input"
                }
            ],
            "crypto": [
                {
                    "id": "CRYPTO_001",
                    "pattern": r"hashlib\.(md5|sha1)\(",
                    "severity": "medium",
                    "title": "Weak cryptographic hash",
                    "description": "Using weak MD5 or SHA1 hash function"
                },
                {
                    "id": "CRYPTO_002",
                    "pattern": r"random\.random\(\)",
                    "severity": "medium",
                    "title": "Weak random number generation",
                    "description": "Using random.random() for security purposes"
                }
            ],
            "files": [
                {
                    "id": "FILE_001",
                    "pattern": r"open\s*\([^)]*['\"]\/[^'\"]*['\"][^)]*['\"]w",
                    "severity": "medium",
                    "title": "Potential path traversal",
                    "description": "Writing to absolute path without validation"
                }
            ]
        }
    
    def scan_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan a single file for security issues"""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
            # Scan with all rule categories
            for category, rules in self.rules.items():
                for rule in rules:
                    pattern = re.compile(rule["pattern"], re.MULTILINE | re.IGNORECASE)
                    
                    for match in pattern.finditer(content):
                        # Find line number
                        line_start = content[:match.start()].count('\n') + 1
                        
                        # Get code snippet
                        if line_start <= len(lines):
                            code_snippet = lines[line_start - 1].strip()
                        else:
                            code_snippet = match.group(0)
                        
                        finding = SecurityFinding(
                            severity=rule["severity"],
                            rule_id=rule["id"],
                            title=rule["title"],
                            description=rule["description"],
                            file_path=str(file_path.relative_to(self.repo_root)),
                            line_number=line_start,
                            code_snippet=code_snippet
                        )
                        findings.append(finding)
                        
        except Exception as e:
            self.logger.warning(f"Failed to scan {file_path}: {e}")
            
        return findings
    
    def scan_directory(self, extensions: List[str] = None) -> List[SecurityFinding]:
        """Scan directory for security issues"""
        if extensions is None:
            extensions = ['.py', '.yml', '.yaml', '.json', '.sh', '.env']
        
        findings = []
        
        for ext in extensions:
            for file_path in self.repo_root.rglob(f"*{ext}"):
                # Skip hidden directories and common ignore patterns
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                if 'node_modules' in file_path.parts or '__pycache__' in file_path.parts:
                    continue
                
                file_findings = self.scan_file(file_path)
                findings.extend(file_findings)
        
        return findings
    
    def run_external_scanners(self) -> List[SecurityFinding]:
        """Run external security scanners if available"""
        findings = []
        
        # Try to run bandit if available
        try:
            result = subprocess.run(
                ['bandit', '-r', '.', '-f', 'json'],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                bandit_data = json.loads(result.stdout)
                for issue in bandit_data.get('results', []):
                    finding = SecurityFinding(
                        severity=issue['issue_severity'].lower(),
                        rule_id=f"BANDIT_{issue['test_id']}",
                        title=issue['issue_text'],
                        description=f"Bandit: {issue['issue_text']}",
                        file_path=issue['filename'],
                        line_number=issue['line_number'],
                        code_snippet=issue['code']
                    )
                    findings.append(finding)
                    
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            # Bandit not available or failed
            pass
        
        return findings
    
    def generate_report(self) -> SecurityReport:
        """Generate comprehensive security report"""
        import time
        start_time = time.time()
        
        self.logger.info("Starting security scan")
        
        # Collect findings
        findings = []
        findings.extend(self.scan_directory())
        findings.extend(self.run_external_scanners())
        
        # Count by severity
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        for finding in findings:
            severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
        
        # Get scanned files
        scanned_files = []
        for ext in ['.py', '.yml', '.yaml', '.json', '.sh', '.env']:
            for file_path in self.repo_root.rglob(f"*{ext}"):
                if not any(part.startswith('.') for part in file_path.parts):
                    scanned_files.append(str(file_path.relative_to(self.repo_root)))
        
        scan_duration = time.time() - start_time
        
        report = SecurityReport(
            timestamp=datetime.datetime.now().isoformat(),
            scan_duration=scan_duration,
            total_findings=len(findings),
            critical_findings=severity_counts["critical"],
            high_findings=severity_counts["high"],
            medium_findings=severity_counts["medium"],
            low_findings=severity_counts["low"],
            info_findings=severity_counts["info"],
            findings=findings,
            scanned_files=scanned_files
        )
        
        self.logger.info(f"Security scan completed in {scan_duration:.2f}s with {len(findings)} findings")
        return report
    
    def save_report(self, report: SecurityReport) -> Path:
        """Save security report to file"""
        reports_dir = self.repo_root / "security_reports"
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f"security_scan_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        # Save as latest
        latest_file = reports_dir / "security_scan_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        return report_file
    
    def check_security_gates(self, report: SecurityReport) -> Tuple[bool, List[str]]:
        """Check if security gates pass"""
        failures = []
        
        # Critical findings block deployment
        if report.critical_findings > 0:
            failures.append(f"Critical security findings: {report.critical_findings}")
        
        # High findings limit (configurable)
        max_high_findings = 5  # Could be configuration
        if report.high_findings > max_high_findings:
            failures.append(f"Too many high severity findings: {report.high_findings} > {max_high_findings}")
        
        return len(failures) == 0, failures


def main():
    """CLI entry point for security scanner"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python scanner.py <command>")
        print("Commands: scan, report, gate")
        return
    
    command = sys.argv[1]
    scanner = SecurityScanner()
    
    if command == "scan":
        report = scanner.generate_report()
        report_file = scanner.save_report(report)
        print(f"Security scan completed. Report saved to: {report_file}")
        print(f"Total findings: {report.total_findings}")
        print(f"Critical: {report.critical_findings}, High: {report.high_findings}, Medium: {report.medium_findings}")
        
    elif command == "report":
        report = scanner.generate_report()
        print(json.dumps(asdict(report), indent=2, default=str))
        
    elif command == "gate":
        report = scanner.generate_report()
        passed, failures = scanner.check_security_gates(report)
        
        if passed:
            print("✅ Security gates passed")
            sys.exit(0)
        else:
            print("❌ Security gates failed:")
            for failure in failures:
                print(f"  - {failure}")
            sys.exit(1)
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()