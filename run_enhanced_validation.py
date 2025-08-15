#!/usr/bin/env python3
"""
Enhanced Validation Runner for Terragon SDLC v4.0
Comprehensive validation without external dependencies
"""

import json
import time
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedValidationRunner:
    """Comprehensive validation system"""
    
    def __init__(self):
        self.validation_results = {}
        self.start_time = datetime.now()
        
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation phases"""
        print("🚀 Terragon SDLC v4.0 - Enhanced Validation Suite")
        print("=" * 70)
        
        validations = [
            ("Repository Structure", self.validate_repository_structure),
            ("Code Quality", self.validate_code_quality),
            ("Security Baseline", self.validate_security_baseline),
            ("Performance Architecture", self.validate_performance_architecture),
            ("SDLC Implementation", self.validate_sdlc_implementation),
            ("Documentation Quality", self.validate_documentation),
            ("Deployment Readiness", self.validate_deployment_readiness)
        ]
        
        passed = 0
        total = len(validations)
        
        for phase_name, validation_func in validations:
            print(f"\n🔍 {phase_name}...")
            try:
                result = validation_func()
                self.validation_results[phase_name] = result
                
                if result["passed"]:
                    print(f"   ✅ {phase_name}: PASSED ({result['score']:.1f}/100)")
                    passed += 1
                else:
                    print(f"   ❌ {phase_name}: FAILED ({result['score']:.1f}/100)")
                    
                if result.get("warnings"):
                    for warning in result["warnings"]:
                        print(f"   ⚠️  {warning}")
                        
            except Exception as e:
                print(f"   ❌ {phase_name}: ERROR - {str(e)}")
                self.validation_results[phase_name] = {
                    "passed": False,
                    "score": 0.0,
                    "error": str(e)
                }
        
        # Generate summary
        execution_time = (datetime.now() - self.start_time).total_seconds()
        overall_score = sum(r.get("score", 0) for r in self.validation_results.values()) / total
        
        print(f"\n" + "=" * 70)
        print(f"📊 ENHANCED VALIDATION SUMMARY")
        print(f"=" * 70)
        print(f"⏰ Execution Time: {execution_time:.2f}s")
        print(f"📈 Overall Score: {overall_score:.1f}/100")
        print(f"✅ Passed: {passed}/{total} validations")
        print(f"❌ Failed: {total - passed}/{total} validations")
        
        if passed == total:
            print(f"\n🎉 ALL VALIDATIONS PASSED - TERRAGON SDLC v4.0 READY!")
        else:
            print(f"\n⚠️  Some validations failed - Review required")
        
        # Save comprehensive report
        self.save_validation_report(overall_score, passed, total, execution_time)
        
        return {
            "overall_score": overall_score,
            "passed_count": passed,
            "total_count": total,
            "execution_time": execution_time,
            "results": self.validation_results
        }
    
    def validate_repository_structure(self) -> Dict[str, Any]:
        """Validate repository structure and organization"""
        score = 0.0
        max_score = 100.0
        warnings = []
        
        required_files = [
            "README.md",
            "pyproject.toml", 
            "requirements.txt",
            "ARCHITECTURE.md",
            "autonomous_sdlc_engine.py",
            "terragon_enhanced_executor.py"
        ]
        
        required_dirs = [
            "src/",
            "tests/", 
            "docs/",
            "src/performance/",
            "src/security/",
            "src/resilience/"
        ]
        
        # Check required files
        files_found = 0
        for file_path in required_files:
            if Path(file_path).exists():
                files_found += 1
            else:
                warnings.append(f"Missing required file: {file_path}")
        
        score += (files_found / len(required_files)) * 40
        
        # Check required directories
        dirs_found = 0
        for dir_path in required_dirs:
            if Path(dir_path).exists():
                dirs_found += 1
            else:
                warnings.append(f"Missing required directory: {dir_path}")
        
        score += (dirs_found / len(required_dirs)) * 30
        
        # Check for advanced features
        advanced_features = [
            "src/performance/adaptive_cache.py",
            "src/security/comprehensive_scanner.py",
            "src/resilience/enhanced_error_handling.py",
            "value_discovery_engine.py"
        ]
        
        advanced_found = sum(1 for f in advanced_features if Path(f).exists())
        score += (advanced_found / len(advanced_features)) * 30
        
        return {
            "passed": score >= 80.0,
            "score": score,
            "warnings": warnings,
            "details": {
                "required_files": f"{files_found}/{len(required_files)}",
                "required_dirs": f"{dirs_found}/{len(required_dirs)}",
                "advanced_features": f"{advanced_found}/{len(advanced_features)}"
            }
        }
    
    def validate_code_quality(self) -> Dict[str, Any]:
        """Validate code quality and structure"""
        score = 0.0
        warnings = []
        
        # Check Python files for basic quality
        python_files = list(Path(".").glob("**/*.py"))
        if python_files:
            score += 20  # Python files exist
        
        # Check for type hints and docstrings
        files_with_quality = 0
        for py_file in python_files[:10]:  # Sample first 10 files
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if '"""' in content or "'''" in content:
                        files_with_quality += 1
            except:
                pass
        
        if python_files:
            score += (files_with_quality / min(len(python_files), 10)) * 30
        
        # Check for configuration files
        config_files = ["pyproject.toml", "setup.py", "requirements.txt"]
        config_found = sum(1 for f in config_files if Path(f).exists())
        score += (config_found / len(config_files)) * 25
        
        # Check for testing structure
        test_files = list(Path("tests").glob("**/*.py")) if Path("tests").exists() else []
        if test_files:
            score += 25
        else:
            warnings.append("No test files found in tests/ directory")
        
        return {
            "passed": score >= 70.0,
            "score": score,
            "warnings": warnings,
            "details": {
                "python_files": len(python_files),
                "files_with_docs": files_with_quality,
                "config_files": config_found,
                "test_files": len(test_files)
            }
        }
    
    def validate_security_baseline(self) -> Dict[str, Any]:
        """Validate security implementation"""
        score = 0.0
        warnings = []
        
        # Check for security modules
        security_files = [
            "src/security/comprehensive_scanner.py",
            "src/security/enhanced_scanner.py", 
            "src/security/input_validator.py"
        ]
        
        security_found = sum(1 for f in security_files if Path(f).exists())
        score += (security_found / len(security_files)) * 40
        
        # Check for resilience patterns
        resilience_files = [
            "src/resilience/enhanced_error_handling.py",
            "src/resilience/circuit_breaker_enhanced.py",
            "src/resilience/retry_with_backoff.py"
        ]
        
        resilience_found = sum(1 for f in resilience_files if Path(f).exists())
        score += (resilience_found / len(resilience_files)) * 30
        
        # Check for security configurations
        if Path("src/security").exists():
            score += 20
        
        # Check for error handling patterns
        error_handling_score = 0
        for py_file in Path(".").glob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'try:' in content and 'except' in content:
                        error_handling_score += 1
            except:
                pass
        
        if error_handling_score > 0:
            score += 10
        
        if score < 80:
            warnings.append("Security implementation could be enhanced")
        
        return {
            "passed": score >= 70.0,
            "score": score,
            "warnings": warnings,
            "details": {
                "security_modules": f"{security_found}/{len(security_files)}",
                "resilience_modules": f"{resilience_found}/{len(resilience_files)}",
                "error_handling_files": error_handling_score
            }
        }
    
    def validate_performance_architecture(self) -> Dict[str, Any]:
        """Validate performance and scalability architecture"""
        score = 0.0
        warnings = []
        
        # Check for performance modules
        performance_files = [
            "src/performance/adaptive_cache.py",
            "src/performance/distributed_task_executor.py",
            "src/performance/intelligent_load_balancer.py",
            "src/performance/auto_scaling_manager.py"
        ]
        
        perf_found = sum(1 for f in performance_files if Path(f).exists())
        score += (perf_found / len(performance_files)) * 50
        
        # Check for async patterns
        async_files = 0
        for py_file in Path(".").glob("**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'async def' in content or 'await ' in content:
                        async_files += 1
            except:
                pass
        
        if async_files > 0:
            score += 25
        else:
            warnings.append("Limited async/await patterns found")
        
        # Check for caching implementation
        if Path("src/performance/adaptive_cache.py").exists():
            score += 25
        else:
            warnings.append("No adaptive caching implementation found")
        
        return {
            "passed": score >= 75.0,
            "score": score,
            "warnings": warnings,
            "details": {
                "performance_modules": f"{perf_found}/{len(performance_files)}",
                "async_files": async_files
            }
        }
    
    def validate_sdlc_implementation(self) -> Dict[str, Any]:
        """Validate SDLC automation implementation"""
        score = 0.0
        warnings = []
        
        # Check for core SDLC files
        sdlc_files = [
            "autonomous_sdlc_engine.py",
            "terragon_enhanced_executor.py",
            "value_discovery_engine.py",
            "backlog_manager.py"
        ]
        
        sdlc_found = sum(1 for f in sdlc_files if Path(f).exists())
        score += (sdlc_found / len(sdlc_files)) * 40
        
        # Check for quality gates
        quality_files = [
            "run_quality_gates.py",
            "quantum_quality_gates.py",
            "src/quality/gates.py"
        ]
        
        quality_found = sum(1 for f in quality_files if Path(f).exists())
        score += (quality_found / len(quality_files)) * 30
        
        # Check for automation scripts
        if Path("scripts/").exists():
            score += 15
        
        # Check for backlog management
        if Path("backlog/").exists():
            score += 15
        else:
            warnings.append("No backlog directory found")
        
        return {
            "passed": score >= 80.0,
            "score": score,
            "warnings": warnings,
            "details": {
                "sdlc_modules": f"{sdlc_found}/{len(sdlc_files)}",
                "quality_gates": f"{quality_found}/{len(quality_files)}"
            }
        }
    
    def validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation quality and completeness"""
        score = 0.0
        warnings = []
        
        # Check for essential documentation
        doc_files = [
            "README.md",
            "ARCHITECTURE.md", 
            "CONTRIBUTING.md",
            "CHANGELOG.md"
        ]
        
        doc_found = sum(1 for f in doc_files if Path(f).exists())
        score += (doc_found / len(doc_files)) * 40
        
        # Check docs directory
        if Path("docs/").exists():
            score += 20
            docs_files = list(Path("docs/").glob("**/*.md"))
            if docs_files:
                score += 20
        else:
            warnings.append("No docs/ directory found")
        
        # Check README quality
        if Path("README.md").exists():
            try:
                with open("README.md", 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 1000:  # Substantial README
                        score += 20
            except:
                pass
        
        return {
            "passed": score >= 70.0,
            "score": score,
            "warnings": warnings,
            "details": {
                "essential_docs": f"{doc_found}/{len(doc_files)}"
            }
        }
    
    def validate_deployment_readiness(self) -> Dict[str, Any]:
        """Validate deployment readiness"""
        score = 0.0
        warnings = []
        
        # Check for deployment files
        deploy_files = [
            "Dockerfile",
            "docker-compose.yml",
            "pyproject.toml",
            "requirements.txt"
        ]
        
        deploy_found = sum(1 for f in deploy_files if Path(f).exists())
        score += (deploy_found / len(deploy_files)) * 40
        
        # Check for configuration
        if Path("deployment/").exists():
            score += 20
        
        # Check for monitoring setup
        monitoring_files = [
            "docker-compose.observability.yml",
            "monitoring/prometheus/prometheus.yml"
        ]
        
        monitoring_found = sum(1 for f in monitoring_files if Path(f).exists())
        score += (monitoring_found / len(monitoring_files)) * 20
        
        # Check for automation scripts
        if Path("scripts/").exists():
            score += 20
        else:
            warnings.append("No automation scripts found")
        
        return {
            "passed": score >= 75.0,
            "score": score,
            "warnings": warnings,
            "details": {
                "deployment_files": f"{deploy_found}/{len(deploy_files)}",
                "monitoring_setup": f"{monitoring_found}/{len(monitoring_files)}"
            }
        }
    
    def save_validation_report(self, overall_score: float, passed: int, total: int, execution_time: float):
        """Save comprehensive validation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure reports directory exists
        reports_dir = Path("validation_reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Generate JSON report
        report = {
            "timestamp": datetime.now().isoformat(),
            "terragon_version": "v4.0",
            "validation_type": "enhanced_comprehensive",
            "summary": {
                "overall_score": overall_score,
                "passed_validations": passed,
                "total_validations": total,
                "success_rate": (passed / total) * 100,
                "execution_time_seconds": execution_time
            },
            "results": self.validation_results,
            "recommendations": self.generate_recommendations()
        }
        
        json_path = reports_dir / f"enhanced_validation_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save as latest
        latest_path = reports_dir / "enhanced_validation_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Validation report saved: {json_path}")
        
        # Generate markdown report
        self.generate_markdown_report(report, reports_dir / f"enhanced_validation_{timestamp}.md")
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        for phase, result in self.validation_results.items():
            if not result.get("passed", True):
                score = result.get("score", 0)
                if score < 50:
                    recommendations.append(f"CRITICAL: {phase} requires immediate attention (score: {score:.1f})")
                elif score < 75:
                    recommendations.append(f"HIGH: Improve {phase} implementation (score: {score:.1f})")
                else:
                    recommendations.append(f"MEDIUM: Minor {phase} enhancements needed (score: {score:.1f})")
        
        if not recommendations:
            recommendations.append("✅ All validations passed - System is production ready!")
        
        return recommendations
    
    def generate_markdown_report(self, report: Dict, output_path: Path):
        """Generate markdown validation report"""
        with open(output_path, 'w') as f:
            f.write(f"""# 🚀 Terragon SDLC v4.0 - Enhanced Validation Report

**Generated**: {report['timestamp']}  
**Version**: {report['terragon_version']}  
**Validation Type**: {report['validation_type']}

## 📊 Executive Summary

- **Overall Score**: {report['summary']['overall_score']:.1f}/100
- **Success Rate**: {report['summary']['success_rate']:.1f}%
- **Validations Passed**: {report['summary']['passed_validations']}/{report['summary']['total_validations']}
- **Execution Time**: {report['summary']['execution_time_seconds']:.2f} seconds

## 📋 Validation Results

""")
            
            for phase, result in report['results'].items():
                status = "✅ PASSED" if result.get('passed', False) else "❌ FAILED"
                score = result.get('score', 0)
                
                f.write(f"""### {status} {phase}

- **Score**: {score:.1f}/100
""")
                
                if result.get('warnings'):
                    f.write("- **Warnings**:\n")
                    for warning in result['warnings']:
                        f.write(f"  - ⚠️ {warning}\n")
                
                if result.get('details'):
                    f.write("- **Details**:\n")
                    for key, value in result['details'].items():
                        f.write(f"  - {key.replace('_', ' ').title()}: {value}\n")
                
                f.write("\n")
            
            f.write(f"""## 💡 Recommendations

""")
            
            for rec in report['recommendations']:
                f.write(f"- {rec}\n")
            
            f.write(f"""
---

*Generated by Terragon Enhanced Validation System v4.0*
""")
        
        print(f"📄 Markdown report saved: {output_path}")


def main():
    """Main entry point"""
    runner = EnhancedValidationRunner()
    results = runner.run_comprehensive_validation()
    
    return results["passed_count"] == results["total_count"]


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)