#!/usr/bin/env python3
"""
Automated maintenance script for ADO project.
Performs routine maintenance tasks and generates reports.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class MaintenanceAutomator:
    """Automates routine maintenance tasks."""
    
    def __init__(self):
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.results = {
            "timestamp": self.timestamp,
            "tasks_completed": [],
            "tasks_failed": [],
            "recommendations": [],
            "summary": {}
        }
    
    def _run_command(self, command: str, capture_output: bool = True) -> Optional[str]:
        """Run shell command and return output."""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=capture_output, text=True, check=True
            )
            return result.stdout.strip() if capture_output else None
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {command}")
            if capture_output:
                print(f"Error: {e.stderr}")
            return None
    
    def cleanup_cache_files(self) -> bool:
        """Clean up cache files and temporary directories."""
        print("Cleaning up cache files...")
        
        cleanup_commands = [
            "find . -type d -name '__pycache__' -delete",
            "find . -name '*.pyc' -delete",
            "rm -rf .pytest_cache",
            "rm -rf .mypy_cache",
            "rm -rf .coverage",
            "rm -rf htmlcov",
            "rm -rf build/",
            "rm -rf dist/",
            "rm -rf *.egg-info/"
        ]
        
        success = True
        for command in cleanup_commands:
            if self._run_command(command, capture_output=False) is None:
                success = False
        
        if success:
            self.results["tasks_completed"].append("Cache cleanup")
            print("✅ Cache files cleaned up")
        else:
            self.results["tasks_failed"].append("Cache cleanup")
            print("❌ Some cache cleanup operations failed")
        
        return success
    
    def update_dependencies(self, update_type: str = "patch") -> bool:
        """Update project dependencies."""
        print(f"Updating dependencies ({update_type})...")
        
        # Backup current requirements
        backup_commands = [
            "cp requirements.txt requirements.txt.bak",
            "cp requirements-dev.txt requirements-dev.txt.bak"
        ]
        
        for command in backup_commands:
            if self._run_command(command, capture_output=False) is None:
                self.results["tasks_failed"].append("Dependency backup")
                return False
        
        # Update based on type
        if update_type == "patch":
            update_command = "pip-compile --upgrade-package '*' requirements.in"
            update_dev_command = "pip-compile --upgrade-package '*' requirements-dev.in"
        else:
            update_command = "pip-compile --upgrade requirements.in"
            update_dev_command = "pip-compile --upgrade requirements-dev.in"
        
        success = True
        if self._run_command(update_command) is None:
            success = False
        if self._run_command(update_dev_command) is None:
            success = False
        
        if success:
            # Check for changes
            diff_output = self._run_command("diff requirements.txt requirements.txt.bak")
            if diff_output:
                self.results["tasks_completed"].append(f"Dependencies updated ({update_type})")
                self.results["summary"]["dependency_changes"] = "Changes detected"
                print("✅ Dependencies updated successfully")
            else:
                self.results["tasks_completed"].append("Dependencies checked (no updates needed)")
                print("✅ Dependencies are up to date")
        else:
            self.results["tasks_failed"].append("Dependency update")
            print("❌ Dependency update failed")
        
        return success
    
    def run_security_audit(self) -> bool:
        """Run comprehensive security audit."""
        print("Running security audit...")
        
        audit_results = {}
        
        # Safety check
        safety_output = self._run_command("safety check --json")
        if safety_output:
            try:
                safety_data = json.loads(safety_output)
                audit_results["safety"] = {
                    "vulnerabilities": len(safety_data),
                    "status": "passed" if len(safety_data) == 0 else "failed"
                }
            except json.JSONDecodeError:
                audit_results["safety"] = {"status": "error"}
        
        # pip-audit
        pip_audit_output = self._run_command("pip-audit --format=json")
        if pip_audit_output:
            try:
                pip_audit_data = json.loads(pip_audit_output)
                vuln_count = len(pip_audit_data.get("vulnerabilities", []))
                audit_results["pip_audit"] = {
                    "vulnerabilities": vuln_count,
                    "status": "passed" if vuln_count == 0 else "failed"
                }
            except json.JSONDecodeError:
                audit_results["pip_audit"] = {"status": "error"}
        
        # Bandit
        bandit_output = self._run_command("bandit -r . -x tests/ -f json")
        if bandit_output:
            try:
                bandit_data = json.loads(bandit_output)
                high_severity = len([r for r in bandit_data.get("results", []) 
                                   if r.get("issue_severity") == "HIGH"])
                audit_results["bandit"] = {
                    "high_severity_issues": high_severity,
                    "status": "passed" if high_severity == 0 else "failed"
                }
            except json.JSONDecodeError:
                audit_results["bandit"] = {"status": "error"}
        
        # Determine overall status
        failed_audits = [k for k, v in audit_results.items() if v.get("status") == "failed"]
        
        if failed_audits:
            self.results["tasks_failed"].append("Security audit")
            self.results["recommendations"].append("Address security vulnerabilities found in audit")
            print(f"❌ Security audit failed: {', '.join(failed_audits)}")
        else:
            self.results["tasks_completed"].append("Security audit")
            print("✅ Security audit passed")
        
        self.results["summary"]["security_audit"] = audit_results
        return len(failed_audits) == 0
    
    def validate_code_quality(self) -> bool:
        """Validate code quality standards."""
        print("Validating code quality...")
        
        quality_checks = {}
        
        # Linting
        lint_output = self._run_command("ruff check . --format=json")
        if lint_output:
            try:
                lint_data = json.loads(lint_output)
                issue_count = len(lint_data)
                quality_checks["linting"] = {
                    "issues": issue_count,
                    "status": "passed" if issue_count == 0 else "warning"
                }
            except json.JSONDecodeError:
                quality_checks["linting"] = {"status": "error"}
        
        # Formatting
        format_output = self._run_command("ruff format --check .")
        quality_checks["formatting"] = {
            "status": "passed" if format_output is not None else "failed"
        }
        
        # Type checking
        mypy_output = self._run_command("mypy .")
        quality_checks["type_checking"] = {
            "status": "passed" if mypy_output is not None else "failed"
        }
        
        # Test execution
        test_output = self._run_command("python -m pytest --quiet")
        quality_checks["tests"] = {
            "status": "passed" if test_output is not None else "failed"
        }
        
        # Determine overall status
        failed_checks = [k for k, v in quality_checks.items() if v.get("status") == "failed"]
        
        if failed_checks:
            self.results["tasks_failed"].append("Code quality validation")
            self.results["recommendations"].append(f"Fix code quality issues: {', '.join(failed_checks)}")
            print(f"❌ Code quality validation failed: {', '.join(failed_checks)}")
        else:
            self.results["tasks_completed"].append("Code quality validation")
            print("✅ Code quality validation passed")
        
        self.results["summary"]["code_quality"] = quality_checks
        return len(failed_checks) == 0
    
    def optimize_repository(self) -> bool:
        """Optimize repository structure and performance."""
        print("Optimizing repository...")
        
        optimization_tasks = []
        
        # Git cleanup
        git_commands = [
            "git gc --aggressive --prune=now",
            "git remote prune origin"
        ]
        
        git_success = True
        for command in git_commands:
            if self._run_command(command, capture_output=False) is None:
                git_success = False
        
        if git_success:
            optimization_tasks.append("Git optimization")
        
        # Remove empty directories
        if self._run_command("find . -type d -empty -delete", capture_output=False) is not None:
            optimization_tasks.append("Empty directory cleanup")
        
        # Update file permissions
        script_files = Path("scripts").glob("*.py") if Path("scripts").exists() else []
        for script in script_files:
            self._run_command(f"chmod +x {script}", capture_output=False)
        
        if optimization_tasks:
            self.results["tasks_completed"].append("Repository optimization")
            self.results["summary"]["optimization"] = optimization_tasks
            print("✅ Repository optimization completed")
            return True
        else:
            self.results["tasks_failed"].append("Repository optimization")
            print("❌ Repository optimization failed")
            return False
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive repository health report."""
        print("Generating health report...")
        
        # File counts and structure
        py_files = len(list(Path(".").rglob("*.py")))
        test_files = len(list(Path("tests").rglob("*.py"))) if Path("tests").exists() else 0
        doc_files = len(list(Path(".").rglob("*.md")))
        
        # Git statistics
        commit_count_output = self._run_command("git rev-list --count HEAD")
        commit_count = int(commit_count_output) if commit_count_output else 0
        
        branch_count_output = self._run_command("git branch -r | wc -l")
        branch_count = int(branch_count_output) if branch_count_output else 0
        
        # Size information
        repo_size_output = self._run_command("du -sh . | cut -f1")
        
        health_report = {
            "repository_structure": {
                "python_files": py_files,
                "test_files": test_files,
                "documentation_files": doc_files,
                "test_to_code_ratio": round(test_files / max(py_files, 1), 2)
            },
            "git_statistics": {
                "total_commits": commit_count,
                "total_branches": branch_count,
                "repository_size": repo_size_output
            },
            "maintenance_status": {
                "last_maintenance": self.timestamp,
                "tasks_completed": len(self.results["tasks_completed"]),
                "tasks_failed": len(self.results["tasks_failed"]),
                "overall_health": "good" if len(self.results["tasks_failed"]) == 0 else "needs_attention"
            }
        }
        
        self.results["summary"]["health_report"] = health_report
        return health_report
    
    def run_all_maintenance_tasks(self, update_deps: bool = False) -> Dict[str, Any]:
        """Run all maintenance tasks."""
        print("Starting automated maintenance...")
        
        try:
            # Core maintenance tasks
            self.cleanup_cache_files()
            
            if update_deps:
                self.update_dependencies()
            
            self.run_security_audit()
            self.validate_code_quality()
            self.optimize_repository()
            
            # Generate comprehensive report
            health_report = self.generate_health_report()
            
            # Final summary
            total_tasks = len(self.results["tasks_completed"]) + len(self.results["tasks_failed"])
            success_rate = len(self.results["tasks_completed"]) / max(total_tasks, 1) * 100
            
            print(f"\n=== MAINTENANCE SUMMARY ===")
            print(f"✅ Tasks completed: {len(self.results['tasks_completed'])}")
            print(f"❌ Tasks failed: {len(self.results['tasks_failed'])}")
            print(f"📊 Success rate: {success_rate:.1f}%")
            
            if self.results["recommendations"]:
                print(f"\n📋 Recommendations:")
                for rec in self.results["recommendations"]:
                    print(f"  • {rec}")
            
            print("\nMaintenance completed!")
            
        except Exception as e:
            print(f"Error during maintenance: {e}")
            self.results["tasks_failed"].append(f"Maintenance automation: {str(e)}")
        
        return self.results


def main():
    """Main entry point."""
    update_deps = "--update-deps" in sys.argv
    report_only = "--report-only" in sys.argv
    
    automator = MaintenanceAutomator()
    
    if report_only:
        health_report = automator.generate_health_report()
        print(json.dumps(health_report, indent=2))
    else:
        results = automator.run_all_maintenance_tasks(update_deps=update_deps)
        
        # Save results to file
        results_file = Path("docs/status/maintenance_results.json")
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    main()