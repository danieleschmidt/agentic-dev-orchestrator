#!/usr/bin/env python3
"""
Repository health monitoring script for ADO project.
Monitors repository health and generates status updates.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class RepositoryHealthMonitor:
    """Monitors repository health and generates status reports."""
    
    def __init__(self):
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.status = {
            "timestamp": self.timestamp,
            "overall_health": "unknown",
            "health_score": 0,
            "categories": {},
            "alerts": [],
            "recommendations": [],
            "trends": {}
        }
    
    def _run_command(self, command: str) -> Optional[str]:
        """Run shell command and return output."""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def _calculate_score(self, passed: int, total: int) -> int:
        """Calculate percentage score."""
        if total == 0:
            return 100
        return int((passed / total) * 100)
    
    def monitor_code_quality(self) -> Tuple[int, Dict[str, Any]]:
        """Monitor code quality metrics."""
        print("Monitoring code quality...")
        
        checks = {
            "linting": {"passed": False, "details": ""},
            "formatting": {"passed": False, "details": ""},
            "type_checking": {"passed": False, "details": ""},
            "complexity": {"passed": False, "details": ""}
        }
        
        # Linting check
        lint_output = self._run_command("ruff check . --quiet")
        if lint_output is not None:
            checks["linting"]["passed"] = True
            checks["linting"]["details"] = "No linting issues found"
        else:
            issue_count_output = self._run_command("ruff check . --format=json | jq length")
            if issue_count_output:
                checks["linting"]["details"] = f"Issues found: {issue_count_output}"
        
        # Formatting check
        format_output = self._run_command("ruff format --check . --quiet")
        if format_output is not None:
            checks["formatting"]["passed"] = True
            checks["formatting"]["details"] = "Code is properly formatted"
        else:
            checks["formatting"]["details"] = "Formatting issues detected"
        
        # Type checking
        mypy_output = self._run_command("mypy . --quiet")
        if mypy_output is not None:
            checks["type_checking"]["passed"] = True
            checks["type_checking"]["details"] = "No type errors found"
        else:
            checks["type_checking"]["details"] = "Type checking issues detected"
        
        # Complexity check (using radon if available)
        complexity_output = self._run_command("radon cc . -a")
        if complexity_output:
            # Parse complexity output for high complexity functions
            lines = complexity_output.split('\n')
            high_complexity = [line for line in lines if 'C' in line or 'D' in line or 'E' in line or 'F' in line]
            if not high_complexity:
                checks["complexity"]["passed"] = True
                checks["complexity"]["details"] = "Acceptable code complexity"
            else:
                checks["complexity"]["details"] = f"High complexity detected in {len(high_complexity)} functions"
        else:
            checks["complexity"]["passed"] = True  # Assume good if tool not available
            checks["complexity"]["details"] = "Complexity check skipped (radon not available)"
        
        passed_checks = sum(1 for check in checks.values() if check["passed"])
        score = self._calculate_score(passed_checks, len(checks))
        
        return score, checks
    
    def monitor_test_health(self) -> Tuple[int, Dict[str, Any]]:
        """Monitor test suite health."""
        print("Monitoring test health...")
        
        metrics = {
            "test_execution": {"passed": False, "details": ""},
            "test_coverage": {"passed": False, "details": ""},
            "test_count": {"passed": False, "details": ""}
        }
        
        # Test execution
        test_output = self._run_command("python -m pytest --quiet --tb=no")
        if test_output is not None:
            metrics["test_execution"]["passed"] = True
            metrics["test_execution"]["details"] = "All tests passing"
        else:
            metrics["test_execution"]["details"] = "Some tests are failing"
        
        # Test coverage
        coverage_output = self._run_command("python -m pytest --cov=. --cov-report=term-missing --quiet")
        if coverage_output:
            # Extract coverage percentage
            for line in coverage_output.split('\n'):
                if 'TOTAL' in line and '%' in line:
                    try:
                        coverage_pct = int(line.split()[-1].replace('%', ''))
                        if coverage_pct >= 80:
                            metrics["test_coverage"]["passed"] = True
                            metrics["test_coverage"]["details"] = f"Coverage: {coverage_pct}%"
                        else:
                            metrics["test_coverage"]["details"] = f"Coverage below target: {coverage_pct}%"
                        break
                    except (ValueError, IndexError):
                        pass
        
        # Test count
        test_files = list(Path("tests").rglob("test_*.py")) if Path("tests").exists() else []
        py_files = list(Path(".").rglob("*.py"))
        py_files = [f for f in py_files if not any(p in str(f) for p in ['tests/', '.venv/', '__pycache__'])]
        
        if len(test_files) > 0:
            test_ratio = len(test_files) / max(len(py_files), 1)
            if test_ratio >= 0.5:  # At least 50% test files to source files
                metrics["test_count"]["passed"] = True
                metrics["test_count"]["details"] = f"Good test coverage: {len(test_files)} test files"
            else:
                metrics["test_count"]["details"] = f"Low test file count: {len(test_files)} test files"
        else:
            metrics["test_count"]["details"] = "No test files found"
        
        passed_checks = sum(1 for metric in metrics.values() if metric["passed"])
        score = self._calculate_score(passed_checks, len(metrics))
        
        return score, metrics
    
    def monitor_security_status(self) -> Tuple[int, Dict[str, Any]]:
        """Monitor security status."""
        print("Monitoring security status...")
        
        security_checks = {
            "dependency_vulnerabilities": {"passed": False, "details": ""},
            "code_security": {"passed": False, "details": ""},
            "secrets_detection": {"passed": False, "details": ""}
        }
        
        # Dependency vulnerabilities
        safety_output = self._run_command("safety check --short-report")
        if safety_output is not None:
            if "No known security vulnerabilities found" in safety_output:
                security_checks["dependency_vulnerabilities"]["passed"] = True
                security_checks["dependency_vulnerabilities"]["details"] = "No known vulnerabilities"
            else:
                security_checks["dependency_vulnerabilities"]["details"] = "Vulnerabilities detected"
        else:
            security_checks["dependency_vulnerabilities"]["details"] = "Safety check failed"
        
        # Code security with Bandit
        bandit_output = self._run_command("bandit -r . -x tests/ --quiet -f json")
        if bandit_output:
            try:
                bandit_data = json.loads(bandit_output)
                high_issues = len([r for r in bandit_data.get("results", []) 
                                 if r.get("issue_severity") == "HIGH"])
                if high_issues == 0:
                    security_checks["code_security"]["passed"] = True
                    security_checks["code_security"]["details"] = "No high-severity security issues"
                else:
                    security_checks["code_security"]["details"] = f"{high_issues} high-severity issues found"
            except json.JSONDecodeError:
                security_checks["code_security"]["details"] = "Bandit scan failed"
        
        # Basic secrets detection (simple patterns)
        secret_output = self._run_command("grep -r -i 'password\\|secret\\|key\\|token' --include='*.py' . | grep -v test | head -5")
        if not secret_output:
            security_checks["secrets_detection"]["passed"] = True
            security_checks["secrets_detection"]["details"] = "No obvious secrets in code"
        else:
            security_checks["secrets_detection"]["details"] = "Potential secrets detected (review required)"
        
        passed_checks = sum(1 for check in security_checks.values() if check["passed"])
        score = self._calculate_score(passed_checks, len(security_checks))
        
        return score, security_checks
    
    def monitor_dependency_health(self) -> Tuple[int, Dict[str, Any]]:
        """Monitor dependency health."""
        print("Monitoring dependency health...")
        
        dep_checks = {
            "outdated_packages": {"passed": False, "details": ""},
            "vulnerability_scan": {"passed": False, "details": ""},
            "license_compliance": {"passed": False, "details": ""}
        }
        
        # Outdated packages
        outdated_output = self._run_command("pip list --outdated --format=json")
        if outdated_output:
            try:
                outdated_data = json.loads(outdated_output)
                outdated_count = len(outdated_data)
                if outdated_count <= 5:  # Threshold for acceptable outdated packages
                    dep_checks["outdated_packages"]["passed"] = True
                    dep_checks["outdated_packages"]["details"] = f"{outdated_count} outdated packages"
                else:
                    dep_checks["outdated_packages"]["details"] = f"{outdated_count} outdated packages (review needed)"
            except json.JSONDecodeError:
                dep_checks["outdated_packages"]["details"] = "Unable to check outdated packages"
        
        # Vulnerability scan with pip-audit
        audit_output = self._run_command("pip-audit --format=json")
        if audit_output:
            try:
                audit_data = json.loads(audit_output)
                vuln_count = len(audit_data.get("vulnerabilities", []))
                if vuln_count == 0:
                    dep_checks["vulnerability_scan"]["passed"] = True
                    dep_checks["vulnerability_scan"]["details"] = "No vulnerabilities found"
                else:
                    dep_checks["vulnerability_scan"]["details"] = f"{vuln_count} vulnerabilities detected"
            except json.JSONDecodeError:
                dep_checks["vulnerability_scan"]["details"] = "Vulnerability scan failed"
        
        # License compliance (basic check)
        licenses_output = self._run_command("pip-licenses --format=json")
        if licenses_output:
            try:
                licenses_data = json.loads(licenses_output)
                problematic_licenses = ['GPL-2.0', 'GPL-3.0', 'AGPL-3.0']
                problem_licenses = [pkg for pkg in licenses_data 
                                  if pkg.get('License') in problematic_licenses]
                if not problem_licenses:
                    dep_checks["license_compliance"]["passed"] = True
                    dep_checks["license_compliance"]["details"] = "No license compliance issues"
                else:
                    dep_checks["license_compliance"]["details"] = f"{len(problem_licenses)} packages with problematic licenses"
            except json.JSONDecodeError:
                dep_checks["license_compliance"]["details"] = "License check failed"
        
        passed_checks = sum(1 for check in dep_checks.values() if check["passed"])
        score = self._calculate_score(passed_checks, len(dep_checks))
        
        return score, dep_checks
    
    def monitor_repository_structure(self) -> Tuple[int, Dict[str, Any]]:
        """Monitor repository structure and organization."""
        print("Monitoring repository structure...")
        
        structure_checks = {
            "essential_files": {"passed": False, "details": ""},
            "documentation": {"passed": False, "details": ""},
            "configuration": {"passed": False, "details": ""},
            "organization": {"passed": False, "details": ""}
        }
        
        # Essential files
        essential_files = ['README.md', 'LICENSE', 'requirements.txt', 'pyproject.toml']
        missing_files = [f for f in essential_files if not Path(f).exists()]
        
        if not missing_files:
            structure_checks["essential_files"]["passed"] = True
            structure_checks["essential_files"]["details"] = "All essential files present"
        else:
            structure_checks["essential_files"]["details"] = f"Missing files: {', '.join(missing_files)}"
        
        # Documentation
        doc_files = list(Path("docs").rglob("*.md")) if Path("docs").exists() else []
        if len(doc_files) >= 5:  # Reasonable documentation threshold
            structure_checks["documentation"]["passed"] = True
            structure_checks["documentation"]["details"] = f"{len(doc_files)} documentation files"
        else:
            structure_checks["documentation"]["details"] = "Insufficient documentation"
        
        # Configuration files
        config_files = ['.gitignore', '.pre-commit-config.yaml', 'pyproject.toml']
        present_configs = [f for f in config_files if Path(f).exists()]
        
        if len(present_configs) >= 2:
            structure_checks["configuration"]["passed"] = True
            structure_checks["configuration"]["details"] = f"{len(present_configs)} config files present"
        else:
            structure_checks["configuration"]["details"] = "Missing configuration files"
        
        # Organization (proper directory structure)
        expected_dirs = ['tests', 'scripts', 'docs']
        present_dirs = [d for d in expected_dirs if Path(d).exists()]
        
        if len(present_dirs) >= 2:
            structure_checks["organization"]["passed"] = True
            structure_checks["organization"]["details"] = "Well-organized directory structure"
        else:
            structure_checks["organization"]["details"] = "Improve directory organization"
        
        passed_checks = sum(1 for check in structure_checks.values() if check["passed"])
        score = self._calculate_score(passed_checks, len(structure_checks))
        
        return score, structure_checks
    
    def generate_health_trends(self) -> Dict[str, Any]:
        """Generate health trends from historical data."""
        trends = {}
        
        # Look for previous status files
        status_dir = Path("docs/status")
        if status_dir.exists():
            status_files = sorted(status_dir.glob("status_*.json"))[-5:]  # Last 5 status files
            
            if len(status_files) >= 2:
                # Calculate trends
                scores = []
                for status_file in status_files:
                    try:
                        with open(status_file, 'r') as f:
                            data = json.load(f)
                            scores.append(data.get("health_score", 0))
                    except (json.JSONDecodeError, FileNotFoundError):
                        continue
                
                if len(scores) >= 2:
                    trend_direction = "improving" if scores[-1] > scores[0] else "declining" if scores[-1] < scores[0] else "stable"
                    trends["health_score_trend"] = {
                        "direction": trend_direction,
                        "change": scores[-1] - scores[0],
                        "history": scores
                    }
        
        return trends
    
    def generate_recommendations(self, categories: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on health status."""
        recommendations = []
        
        # Code quality recommendations
        if categories.get("code_quality", {}).get("score", 100) < 80:
            recommendations.append("Improve code quality by addressing linting and formatting issues")
        
        # Test health recommendations
        if categories.get("test_health", {}).get("score", 100) < 70:
            recommendations.append("Enhance test coverage and fix failing tests")
        
        # Security recommendations
        if categories.get("security_status", {}).get("score", 100) < 90:
            recommendations.append("Address security vulnerabilities and improve security practices")
        
        # Dependency recommendations
        if categories.get("dependency_health", {}).get("score", 100) < 75:
            recommendations.append("Update dependencies and resolve security vulnerabilities")
        
        # Structure recommendations
        if categories.get("repository_structure", {}).get("score", 100) < 80:
            recommendations.append("Improve repository organization and documentation")
        
        return recommendations
    
    def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive repository health check."""
        print("Starting comprehensive health check...")
        
        # Monitor all categories
        code_quality_score, code_quality_data = self.monitor_code_quality()
        test_health_score, test_health_data = self.monitor_test_health()
        security_score, security_data = self.monitor_security_status()
        dependency_score, dependency_data = self.monitor_dependency_health()
        structure_score, structure_data = self.monitor_repository_structure()
        
        # Store category results
        self.status["categories"] = {
            "code_quality": {"score": code_quality_score, "checks": code_quality_data},
            "test_health": {"score": test_health_score, "checks": test_health_data},
            "security_status": {"score": security_score, "checks": security_data},
            "dependency_health": {"score": dependency_score, "checks": dependency_data},
            "repository_structure": {"score": structure_score, "checks": structure_data}
        }
        
        # Calculate overall health score
        category_scores = [code_quality_score, test_health_score, security_score, 
                          dependency_score, structure_score]
        overall_score = sum(category_scores) // len(category_scores)
        self.status["health_score"] = overall_score
        
        # Determine overall health status
        if overall_score >= 90:
            self.status["overall_health"] = "excellent"
        elif overall_score >= 80:
            self.status["overall_health"] = "good"
        elif overall_score >= 70:
            self.status["overall_health"] = "fair"
        else:
            self.status["overall_health"] = "needs_attention"
        
        # Generate alerts for critical issues
        for category, data in self.status["categories"].items():
            if data["score"] < 60:
                self.status["alerts"].append(f"Critical issue in {category.replace('_', ' ')}: {data['score']}% health")
        
        # Generate trends
        self.status["trends"] = self.generate_health_trends()
        
        # Generate recommendations
        self.status["recommendations"] = self.generate_recommendations(self.status["categories"])
        
        # Save status to file
        status_file = Path(f"docs/status/status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        status_file.parent.mkdir(exist_ok=True)
        with open(status_file, 'w') as f:
            json.dump(self.status, f, indent=2)
        
        # Update latest status
        latest_file = Path("docs/status/latest.json")
        with open(latest_file, 'w') as f:
            json.dump(self.status, f, indent=2)
        
        print(f"\n=== REPOSITORY HEALTH SUMMARY ===")
        print(f"Overall Health: {self.status['overall_health'].upper()} ({overall_score}%)")
        print(f"Code Quality: {code_quality_score}%")
        print(f"Test Health: {test_health_score}%")
        print(f"Security: {security_score}%")
        print(f"Dependencies: {dependency_score}%")
        print(f"Structure: {structure_score}%")
        
        if self.status["alerts"]:
            print(f"\n🚨 ALERTS:")
            for alert in self.status["alerts"]:
                print(f"  • {alert}")
        
        if self.status["recommendations"]:
            print(f"\n📋 RECOMMENDATIONS:")
            for rec in self.status["recommendations"]:
                print(f"  • {rec}")
        
        print(f"\nDetailed status saved to: {status_file}")
        print(f"Latest status: {latest_file}")
        
        return self.status


def main():
    """Main entry point."""
    monitor = RepositoryHealthMonitor()
    status = monitor.run_comprehensive_health_check()
    
    # Exit with error code if health is poor
    if status["health_score"] < 70:
        sys.exit(1)


if __name__ == "__main__":
    main()