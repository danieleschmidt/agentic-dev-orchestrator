#!/usr/bin/env python3
"""
Quantum SDLC Deployment Guide
Production deployment preparation and validation
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
import datetime

class QuantumDeploymentGuide:
    """Production deployment guide for quantum SDLC system"""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.deployment_config = self._load_deployment_config()
        
    def _load_deployment_config(self) -> Dict:
        """Load deployment configuration"""
        config_file = self.repo_root / "deployment_config.json"
        default_config = {
            "environment": "production",
            "python_version": "3.8+",
            "required_packages": [
                "PyYAML>=6.0",
                "requests>=2.28.0",
                "click>=8.0.0",
                "rich>=12.0.0",
                "pydantic>=1.10.0",
                "python-dotenv>=0.19.0",
                "gitpython>=3.1.0",
                "jinja2>=3.0.0",
                "python-dateutil>=2.8.0",
                "packaging>=21.0"
            ],
            "optional_packages": [
                "numpy>=1.21.0",
                "psutil>=5.8.0",
                "pytest>=7.0.0",
                "pytest-cov>=4.0.0"
            ],
            "system_requirements": {
                "min_memory": "2GB",
                "min_cpu_cores": "2",
                "min_disk_space": "1GB",
                "supported_os": ["Linux", "macOS", "Windows"]
            },
            "security_requirements": {
                "encrypt_config": True,
                "secure_logging": True,
                "api_key_management": True,
                "input_validation": True
            }
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception:
                pass
        
        return default_config
    
    def validate_deployment_readiness(self) -> Dict[str, Any]:
        """Validate deployment readiness"""
        print("🚀 Validating Deployment Readiness...")
        
        validation_results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "ready_for_deployment": False,
            "checks": {},
            "recommendations": [],
            "deployment_score": 0.0
        }
        
        # 1. Check Python version
        python_check = self._check_python_version()
        validation_results["checks"]["python_version"] = python_check
        
        # 2. Check required files
        files_check = self._check_required_files()
        validation_results["checks"]["required_files"] = files_check
        
        # 3. Check configuration
        config_check = self._check_configuration()
        validation_results["checks"]["configuration"] = config_check
        
        # 4. Check security setup
        security_check = self._check_security_setup()
        validation_results["checks"]["security"] = security_check
        
        # 5. Check quality gates
        quality_check = self._check_quality_gates()
        validation_results["checks"]["quality_gates"] = quality_check
        
        # 6. Check documentation
        docs_check = self._check_documentation()
        validation_results["checks"]["documentation"] = docs_check
        
        # Calculate deployment score
        checks = validation_results["checks"]
        total_score = sum(check["score"] for check in checks.values())
        max_score = len(checks) * 100
        validation_results["deployment_score"] = (total_score / max_score) * 100
        
        # Determine readiness
        validation_results["ready_for_deployment"] = (
            validation_results["deployment_score"] >= 80.0 and
            all(check["passed"] for check in checks.values())
        )
        
        # Generate recommendations
        for check_name, check_result in checks.items():
            validation_results["recommendations"].extend(check_result.get("recommendations", []))
        
        return validation_results
    
    def _check_python_version(self) -> Dict[str, Any]:
        """Check Python version compatibility"""
        try:
            version_info = sys.version_info
            current_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
            
            # Check if version meets requirements
            required_major, required_minor = 3, 8
            version_ok = (version_info.major > required_major or 
                         (version_info.major == required_major and version_info.minor >= required_minor))
            
            return {
                "passed": version_ok,
                "score": 100.0 if version_ok else 0.0,
                "message": f"Python {current_version} ({'✅ Compatible' if version_ok else '❌ Incompatible'})",
                "details": {"current": current_version, "required": f"{required_major}.{required_minor}+"},
                "recommendations": [] if version_ok else [f"Upgrade to Python {required_major}.{required_minor} or higher"]
            }
        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "message": f"Failed to check Python version: {e}",
                "recommendations": ["Fix Python version detection issues"]
            }
    
    def _check_required_files(self) -> Dict[str, Any]:
        """Check for required deployment files"""
        required_files = [
            "ado.py",
            "backlog_manager.py",
            "autonomous_executor.py",
            "quantum_task_planner.py",
            "quantum_security_validator.py",
            "quantum_error_recovery.py",
            "quantum_performance_optimizer.py",
            "run_quality_gates.py",
            "README.md",
            "requirements.txt"
        ]
        
        existing_files = []
        missing_files = []
        
        for file_name in required_files:
            file_path = self.repo_root / file_name
            if file_path.exists():
                existing_files.append(file_name)
            else:
                missing_files.append(file_name)
        
        files_score = (len(existing_files) / len(required_files)) * 100
        files_passed = len(missing_files) == 0
        
        message = f"Files: {len(existing_files)}/{len(required_files)} present"
        recommendations = []
        if missing_files:
            recommendations.append(f"Add missing files: {', '.join(missing_files)}")
        
        return {
            "passed": files_passed,
            "score": files_score,
            "message": message,
            "details": {"existing": existing_files, "missing": missing_files},
            "recommendations": recommendations
        }
    
    def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration completeness"""
        config_files = [
            "pyproject.toml",
            "setup.py",
            "requirements.txt"
        ]
        
        config_score = 0.0
        config_passed = True
        recommendations = []
        existing_configs = []
        
        for config_file in config_files:
            if (self.repo_root / config_file).exists():
                existing_configs.append(config_file)
                config_score += 33.33
        
        # Check if at least one packaging config exists
        packaging_configs = ["pyproject.toml", "setup.py"]
        has_packaging = any((self.repo_root / config).exists() for config in packaging_configs)
        
        if not has_packaging:
            config_passed = False
            recommendations.append("Add packaging configuration (pyproject.toml or setup.py)")
        
        if not (self.repo_root / "requirements.txt").exists():
            recommendations.append("Add requirements.txt for dependency management")
        
        return {
            "passed": config_passed,
            "score": config_score,
            "message": f"Configuration: {len(existing_configs)}/{len(config_files)} files present",
            "details": {"existing_configs": existing_configs},
            "recommendations": recommendations
        }
    
    def _check_security_setup(self) -> Dict[str, Any]:
        """Check security configuration"""
        security_checks = {
            "has_gitignore": (self.repo_root / ".gitignore").exists(),
            "has_security_config": (self.repo_root / ".ado_security.json").exists(),
            "has_env_template": (self.repo_root / ".env.template").exists() or (self.repo_root / ".env.example").exists()
        }
        
        passed_checks = sum(security_checks.values())
        security_score = (passed_checks / len(security_checks)) * 100
        security_passed = passed_checks >= 2  # At least 2/3 checks should pass
        
        recommendations = []
        if not security_checks["has_gitignore"]:
            recommendations.append("Add .gitignore to exclude sensitive files")
        if not security_checks["has_security_config"]:
            recommendations.append("Add .ado_security.json for security configuration")
        if not security_checks["has_env_template"]:
            recommendations.append("Add .env.template for environment variable documentation")
        
        return {
            "passed": security_passed,
            "score": security_score,
            "message": f"Security: {passed_checks}/{len(security_checks)} checks passed",
            "details": security_checks,
            "recommendations": recommendations
        }
    
    def _check_quality_gates(self) -> Dict[str, Any]:
        """Check quality gates status"""
        try:
            # Check if quality gates report exists
            quality_report_file = self.repo_root / "quality_reports" / "quality_gates_latest.json"
            
            if not quality_report_file.exists():
                return {
                    "passed": False,
                    "score": 0.0,
                    "message": "No quality gates report found",
                    "recommendations": ["Run quality gates validation: python3 run_quality_gates.py"]
                }
            
            # Load and check quality gates report
            with open(quality_report_file, 'r') as f:
                quality_report = json.load(f)
            
            overall_score = quality_report.get("overall_score", 0.0)
            quality_passed = overall_score >= 80.0 and quality_report.get("failed", 0) == 0
            
            return {
                "passed": quality_passed,
                "score": overall_score,
                "message": f"Quality Gates: {overall_score:.1f}/100 ({'✅ Passed' if quality_passed else '❌ Failed'})",
                "details": {
                    "overall_score": overall_score,
                    "passed_gates": quality_report.get("passed", 0),
                    "failed_gates": quality_report.get("failed", 0),
                    "warnings": quality_report.get("warnings", 0)
                },
                "recommendations": [] if quality_passed else ["Address quality gate failures before deployment"]
            }
            
        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "message": f"Quality gates check failed: {e}",
                "recommendations": ["Fix quality gates validation system"]
            }
    
    def _check_documentation(self) -> Dict[str, Any]:
        """Check documentation completeness"""
        doc_files = {
            "README.md": "Project overview and setup instructions",
            "ARCHITECTURE.md": "System architecture documentation",
            "CONTRIBUTING.md": "Contribution guidelines",
            "CHANGELOG.md": "Version history and changes"
        }
        
        existing_docs = []
        missing_docs = []
        
        for doc_file, description in doc_files.items():
            if (self.repo_root / doc_file).exists():
                existing_docs.append(doc_file)
            else:
                missing_docs.append((doc_file, description))
        
        docs_score = (len(existing_docs) / len(doc_files)) * 100
        docs_passed = len(existing_docs) >= 2  # At least README and one other doc
        
        recommendations = []
        for doc_file, description in missing_docs:
            recommendations.append(f"Add {doc_file}: {description}")
        
        return {
            "passed": docs_passed,
            "score": docs_score,
            "message": f"Documentation: {len(existing_docs)}/{len(doc_files)} files present",
            "details": {"existing": existing_docs, "missing": [doc[0] for doc in missing_docs]},
            "recommendations": recommendations
        }
    
    def generate_deployment_checklist(self) -> List[str]:
        """Generate pre-deployment checklist"""
        checklist = [
            "✅ Validate deployment readiness",
            "✅ Run comprehensive quality gates",
            "✅ Update version in pyproject.toml",
            "✅ Update CHANGELOG.md with release notes",
            "✅ Create release branch",
            "✅ Tag release version",
            "✅ Build distribution packages",
            "✅ Test installation in clean environment",
            "✅ Validate all CLI commands work",
            "✅ Run security scan",
            "✅ Backup production data (if applicable)",
            "✅ Deploy to staging environment",
            "✅ Run smoke tests in staging",
            "✅ Deploy to production",
            "✅ Verify production deployment",
            "✅ Monitor system health post-deployment"
        ]
        return checklist
    
    def create_deployment_artifacts(self) -> Dict[str, str]:
        """Create deployment artifacts"""
        artifacts = {}
        
        # 1. Create .env.template
        env_template = """# Quantum SDLC Environment Configuration
# Copy this file to .env and fill in your values

# GitHub Integration
GITHUB_TOKEN=your_github_personal_access_token_here

# OpenAI Integration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Security Scanning
SEMGREP_APP_TOKEN=your_semgrep_token_here
SNYK_TOKEN=your_snyk_token_here

# Optional: Monitoring
CODECOV_TOKEN=your_codecov_token_here

# System Configuration
ADO_LOG_LEVEL=INFO
ADO_MAX_WORKERS=4
ADO_CACHE_SIZE=1000
"""
        env_template_file = self.repo_root / ".env.template"
        with open(env_template_file, 'w') as f:
            f.write(env_template)
        artifacts[".env.template"] = str(env_template_file)
        
        # 2. Create .gitignore
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
.venv/
.env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Environment variables
.env
.env.local
.env.production

# Cache and temporary files
.cache/
.tmp/
temp/
tmp/

# Coverage and testing
.coverage
.pytest_cache/
htmlcov/
.tox/

# Security
*.key
*.pem
*.crt
secrets/

# Quantum SDLC specific
escalations/
docs/status/
quality_reports/
security_reports/
error_reports/
performance_reports/
quantum_state/
backlog.json
"""
        gitignore_file = self.repo_root / ".gitignore"
        if not gitignore_file.exists():
            with open(gitignore_file, 'w') as f:
                f.write(gitignore_content)
            artifacts[".gitignore"] = str(gitignore_file)
        
        # 3. Create security configuration
        security_config = {
            "allowed_commands": [
                "git", "python3", "pytest", "black", "ruff", "mypy"
            ],
            "blocked_patterns": [
                "rm -rf", "sudo", "chmod 777", "password", "secret", "api_key"
            ],
            "max_file_size": 10485760,
            "allowed_extensions": [".py", ".yml", ".yaml", ".json", ".md", ".txt"],
            "quantum_security": {
                "enable_threat_detection": True,
                "entropy_threshold": 0.7,
                "coherence_validation": True
            }
        }
        
        security_config_file = self.repo_root / ".ado_security.json"
        if not security_config_file.exists():
            with open(security_config_file, 'w') as f:
                json.dump(security_config, f, indent=2)
            artifacts[".ado_security.json"] = str(security_config_file)
        
        return artifacts
    
    def save_deployment_report(self, validation_results: Dict[str, Any]) -> Path:
        """Save deployment readiness report"""
        reports_dir = self.repo_root / "deployment_reports"
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = reports_dir / f"deployment_readiness_{timestamp}.json"
        
        # Add deployment checklist and artifacts info
        validation_results["deployment_checklist"] = self.generate_deployment_checklist()
        validation_results["deployment_config"] = self.deployment_config
        
        with open(report_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        # Also save as latest
        latest_file = reports_dir / "deployment_readiness_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        return report_file

def main():
    """CLI entry point for deployment guide"""
    deployment_guide = QuantumDeploymentGuide()
    
    print("🚀 Quantum SDLC Deployment Guide")
    print("=" * 50)
    
    # Validate deployment readiness
    validation_results = deployment_guide.validate_deployment_readiness()
    
    # Create deployment artifacts
    print("\n📦 Creating deployment artifacts...")
    artifacts = deployment_guide.create_deployment_artifacts()
    for artifact_name, artifact_path in artifacts.items():
        print(f"   ✅ Created: {artifact_name}")
    
    # Print validation results
    print(f"\n📊 Deployment Readiness Assessment")
    print(f"Overall Score: {validation_results['deployment_score']:.1f}/100")
    print(f"Ready for Deployment: {'✅ YES' if validation_results['ready_for_deployment'] else '❌ NO'}")
    
    print(f"\n🔍 Detailed Checks:")
    for check_name, check_result in validation_results["checks"].items():
        status = "✅" if check_result["passed"] else "❌"
        print(f"   {status} {check_name}: {check_result['message']}")
    
    # Show recommendations
    if validation_results["recommendations"]:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(validation_results["recommendations"], 1):
            print(f"   {i}. {rec}")
    
    # Show deployment checklist
    print(f"\n📋 Deployment Checklist:")
    checklist = deployment_guide.generate_deployment_checklist()
    for item in checklist:
        print(f"   {item}")
    
    # Save report
    report_file = deployment_guide.save_deployment_report(validation_results)
    print(f"\n📄 Deployment report saved to: {report_file}")
    
    # Exit with appropriate code
    if validation_results["ready_for_deployment"]:
        print(f"\n🎉 System is ready for production deployment!")
        return 0
    else:
        print(f"\n⚠️  System requires attention before deployment.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)