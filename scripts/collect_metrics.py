#!/usr/bin/env python3
"""
Automated metrics collection script for ADO project.
Collects various metrics and updates the project-metrics.json file.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class MetricsCollector:
    """Collects and updates project metrics."""
    
    def __init__(self, metrics_file: Path):
        self.metrics_file = metrics_file
        self.metrics = self._load_metrics()
        self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load existing metrics from file."""
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Metrics file not found: {self.metrics_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in metrics file: {e}")
            sys.exit(1)
    
    def _save_metrics(self) -> None:
        """Save updated metrics to file."""
        self.metrics['last_updated'] = self.timestamp
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics updated and saved to {self.metrics_file}")
    
    def _run_command(self, command: str) -> Optional[str]:
        """Run shell command and return output."""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {command}")
            print(f"Error: {e.stderr}")
            return None
    
    def collect_code_quality_metrics(self) -> None:
        """Collect code quality metrics."""
        print("Collecting code quality metrics...")
        
        # Test coverage
        coverage_output = self._run_command(
            "python -m pytest --cov=. --cov-report=json --quiet"
        )
        if coverage_output and Path("coverage.json").exists():
            with open("coverage.json", 'r') as f:
                coverage_data = json.load(f)
                coverage_percent = coverage_data.get("totals", {}).get("percent_covered", 0)
                self.metrics["metrics"]["code_quality"]["test_coverage"]["current"] = round(coverage_percent, 2)
                self.metrics["metrics"]["code_quality"]["test_coverage"]["last_updated"] = self.timestamp
        
        # Linting issues
        lint_output = self._run_command("ruff check . --format=json")
        if lint_output:
            try:
                lint_data = json.loads(lint_output)
                issue_count = len(lint_data)
                self.metrics["metrics"]["code_quality"]["linting"]["issues"] = issue_count
                self.metrics["metrics"]["code_quality"]["linting"]["last_scan"] = self.timestamp
            except json.JSONDecodeError:
                pass
        
        # Type coverage
        mypy_output = self._run_command("mypy . --json-report mypy-report")
        if mypy_output and Path("mypy-report/index.txt").exists():
            # Parse mypy report for type coverage
            self.metrics["metrics"]["code_quality"]["type_coverage"]["last_scan"] = self.timestamp
    
    def collect_security_metrics(self) -> None:
        """Collect security metrics."""
        print("Collecting security metrics...")
        
        # Bandit security scan
        bandit_output = self._run_command("bandit -r . -x tests/ -f json")
        if bandit_output:
            try:
                bandit_data = json.loads(bandit_output)
                results = bandit_data.get("results", [])
                security_metrics = self.metrics["metrics"]["security"]["vulnerability_scan"]
                security_metrics["critical_issues"] = len([r for r in results if r.get("issue_severity") == "HIGH"])
                security_metrics["high_issues"] = len([r for r in results if r.get("issue_severity") == "MEDIUM"])
                security_metrics["medium_issues"] = len([r for r in results if r.get("issue_severity") == "LOW"])
                security_metrics["last_scan"] = self.timestamp
            except json.JSONDecodeError:
                pass
        
        # Safety dependency check
        safety_output = self._run_command("safety check --json")
        if safety_output:
            try:
                safety_data = json.loads(safety_output)
                vuln_count = len(safety_data)
                self.metrics["metrics"]["security"]["dependency_vulnerabilities"]["vulnerable_packages"] = vuln_count
                self.metrics["metrics"]["security"]["dependency_vulnerabilities"]["last_scan"] = self.timestamp
            except json.JSONDecodeError:
                pass
    
    def collect_performance_metrics(self) -> None:
        """Collect performance metrics."""
        print("Collecting performance metrics...")
        
        # Build time
        build_start = datetime.now()
        build_output = self._run_command("python -m build")
        if build_output:
            build_duration = (datetime.now() - build_start).total_seconds()
            self.metrics["metrics"]["performance"]["build_time"]["average_seconds"] = round(build_duration, 2)
            self.metrics["metrics"]["performance"]["build_time"]["last_measured"] = self.timestamp
        
        # Test execution time
        test_start = datetime.now()
        test_output = self._run_command("python -m pytest --quiet")
        if test_output:
            test_duration = (datetime.now() - test_start).total_seconds()
            self.metrics["metrics"]["performance"]["test_execution_time"]["average_seconds"] = round(test_duration, 2)
            self.metrics["metrics"]["performance"]["test_execution_time"]["last_measured"] = self.timestamp
        
        # Docker image size
        docker_output = self._run_command("docker images ado:latest --format 'table {{.Size}}'")
        if docker_output:
            # Parse docker image size
            size_lines = docker_output.split('\n')[1:]  # Skip header
            if size_lines:
                size_str = size_lines[0].strip()
                self.metrics["metrics"]["performance"]["docker_image_size"]["last_measured"] = self.timestamp
    
    def collect_development_metrics(self) -> None:
        """Collect development process metrics."""
        print("Collecting development metrics...")
        
        # Commit frequency (last 7 days)
        commit_output = self._run_command("git log --since='7 days ago' --oneline")
        if commit_output:
            commit_count = len(commit_output.split('\n')) if commit_output else 0
            self.metrics["metrics"]["development"]["commit_frequency"]["commits_per_week"] = commit_count
            self.metrics["metrics"]["development"]["commit_frequency"]["last_calculated"] = self.timestamp
        
        # Contributors
        contributors_output = self._run_command("git log --since='30 days ago' --format='%ae' | sort | uniq")
        if contributors_output:
            contributor_count = len(contributors_output.split('\n')) if contributors_output else 0
            self.metrics["metrics"]["development"]["commit_frequency"]["contributors"] = contributor_count
    
    def collect_maintenance_metrics(self) -> None:
        """Collect maintenance metrics."""
        print("Collecting maintenance metrics...")
        
        # Outdated packages
        outdated_output = self._run_command("pip list --outdated --format=json")
        if outdated_output:
            try:
                outdated_data = json.loads(outdated_output)
                outdated_count = len(outdated_data)
                self.metrics["metrics"]["maintenance"]["dependency_freshness"]["outdated_packages"] = outdated_count
                self.metrics["metrics"]["maintenance"]["dependency_freshness"]["last_checked"] = self.timestamp
            except json.JSONDecodeError:
                pass
        
        # Documentation coverage (count Python files vs docstrings)
        py_files_output = self._run_command("find . -name '*.py' -not -path './tests/*' | wc -l")
        if py_files_output:
            try:
                py_files_count = int(py_files_output)
                # Simplified docstring check
                docstring_output = self._run_command("grep -r '\"\"\"' --include='*.py' . | wc -l")
                if docstring_output:
                    docstring_count = int(docstring_output)
                    doc_coverage = (docstring_count / max(py_files_count, 1)) * 100
                    self.metrics["metrics"]["maintenance"]["documentation_coverage"]["percentage"] = round(doc_coverage, 2)
                    self.metrics["metrics"]["maintenance"]["documentation_coverage"]["last_calculated"] = self.timestamp
            except ValueError:
                pass
    
    def collect_all_metrics(self) -> None:
        """Collect all available metrics."""
        print("Starting metrics collection...")
        
        try:
            self.collect_code_quality_metrics()
            self.collect_security_metrics()
            self.collect_performance_metrics()
            self.collect_development_metrics()
            self.collect_maintenance_metrics()
            
            self._save_metrics()
            print("Metrics collection completed successfully!")
            
        except Exception as e:
            print(f"Error during metrics collection: {e}")
            sys.exit(1)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a summary report of current metrics."""
        report = {
            "timestamp": self.timestamp,
            "summary": {},
            "alerts": [],
            "recommendations": []
        }
        
        # Code quality summary
        test_coverage = self.metrics["metrics"]["code_quality"]["test_coverage"]["current"]
        if test_coverage is not None:
            target_coverage = self.metrics["thresholds"]["code_quality"]["min_test_coverage"]
            report["summary"]["test_coverage"] = f"{test_coverage}% (target: {target_coverage}%)"
            
            if test_coverage < target_coverage:
                report["alerts"].append(f"Test coverage ({test_coverage}%) below target ({target_coverage}%)")
                report["recommendations"].append("Increase test coverage by adding more unit tests")
        
        # Security summary
        vuln_scan = self.metrics["metrics"]["security"]["vulnerability_scan"]
        if vuln_scan.get("critical_issues") is not None:
            critical = vuln_scan["critical_issues"]
            high = vuln_scan["high_issues"]
            
            if critical > 0:
                report["alerts"].append(f"Critical security issues found: {critical}")
                report["recommendations"].append("Address critical security vulnerabilities immediately")
            
            if high > self.metrics["thresholds"]["security"]["max_high_vulnerabilities"]:
                report["alerts"].append(f"High security issues exceed threshold: {high}")
        
        # Dependency freshness
        outdated = self.metrics["metrics"]["maintenance"]["dependency_freshness"]["outdated_packages"]
        if outdated is not None and outdated > 0:
            report["summary"]["outdated_dependencies"] = f"{outdated} packages"
            if outdated > 10:
                report["recommendations"].append("Update outdated dependencies to improve security and performance")
        
        return report


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--report-only":
        # Generate report only
        metrics_file = Path(".github/project-metrics.json")
        collector = MetricsCollector(metrics_file)
        report = collector.generate_report()
        print(json.dumps(report, indent=2))
        return
    
    # Full metrics collection
    metrics_file = Path(".github/project-metrics.json")
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        print("Please ensure the project-metrics.json file exists in .github/")
        sys.exit(1)
    
    collector = MetricsCollector(metrics_file)
    collector.collect_all_metrics()
    
    # Generate and display report
    report = collector.generate_report()
    print("\n=== METRICS REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()