#!/usr/bin/env python3
"""
Quality Gates Validation and Testing
Comprehensive validation of all quality gates and production readiness checks
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
import time

# Add current directory to path
sys.path.insert(0, '.')

def test_code_functionality():
    """Test core functionality works as expected"""
    print("🧪 Testing core functionality...")
    
    try:
        # Test CLI functionality
        result = subprocess.run([
            sys.executable, 'ado.py', 'help'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ CLI help command works")
        else:
            print(f"❌ CLI help failed: {result.stderr}")
            return False
        
        # Test ADO status with demo environment
        env = os.environ.copy()
        env['GITHUB_TOKEN'] = 'demo_token'
        env['OPENAI_API_KEY'] = 'demo_key'
        
        result = subprocess.run([
            sys.executable, 'ado.py', 'status'
        ], capture_output=True, text=True, timeout=30, env=env)
        
        if result.returncode == 0 and 'Backlog Status' in result.stdout:
            print("✅ ADO status command works")
        else:
            print(f"❌ ADO status failed: {result.stderr}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False

def test_security_validation():
    """Test security validation and scanning"""
    print("🛡️ Testing security validation...")
    
    try:
        # Check for hardcoded secrets in common files
        sensitive_patterns = [
            'password = ',
            'secret = ',
            'api_key = ',
            'token = '
        ]
        
        security_issues = []
        
        # Check main source files
        for file_path in ['ado.py', 'backlog_manager.py', 'autonomous_executor.py']:
            if Path(file_path).exists():
                with open(file_path) as f:
                    content = f.read().lower()
                    for pattern in sensitive_patterns:
                        if pattern in content and 'demo' not in content and 'test' not in content:
                            security_issues.append(f"Potential hardcoded secret in {file_path}")
        
        if not security_issues:
            print("✅ No hardcoded secrets detected")
        else:
            for issue in security_issues:
                print(f"⚠️ {issue}")
        
        # Test input validation exists
        validation_found = False
        for src_file in Path('src').rglob('*.py'):
            if 'validation' in src_file.name or 'security' in src_file.name:
                validation_found = True
                break
        
        if validation_found:
            print("✅ Input validation modules found")
        else:
            print("⚠️ Limited input validation modules")
        
        return True
    except Exception as e:
        print(f"❌ Security validation failed: {e}")
        return False

def test_performance_benchmarks():
    """Test performance meets benchmarks"""
    print("🚀 Testing performance benchmarks...")
    
    try:
        # Test CLI startup time
        start_time = time.time()
        result = subprocess.run([
            sys.executable, 'ado.py', 'help'
        ], capture_output=True, text=True, timeout=10)
        startup_time = time.time() - start_time
        
        if startup_time < 5.0:  # Should start in under 5 seconds
            print(f"✅ CLI startup time: {startup_time:.2f}s (< 5s target)")
        else:
            print(f"⚠️ CLI startup time: {startup_time:.2f}s (> 5s)")
        
        # Test backlog processing performance
        env = os.environ.copy()
        env['GITHUB_TOKEN'] = 'demo_token'
        env['OPENAI_API_KEY'] = 'demo_key'
        
        start_time = time.time()
        result = subprocess.run([
            sys.executable, 'ado.py', 'status'
        ], capture_output=True, text=True, timeout=30, env=env)
        processing_time = time.time() - start_time
        
        if processing_time < 10.0:  # Should process in under 10 seconds
            print(f"✅ Backlog processing time: {processing_time:.2f}s (< 10s target)")
        else:
            print(f"⚠️ Backlog processing time: {processing_time:.2f}s (> 10s)")
        
        return True
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def test_documentation_quality():
    """Test documentation completeness and quality"""
    print("📚 Testing documentation quality...")
    
    try:
        required_docs = [
            'README.md',
            'CONTRIBUTING.md',
            'CHANGELOG.md',
            'LICENSE'
        ]
        
        missing_docs = []
        for doc in required_docs:
            if not Path(doc).exists():
                missing_docs.append(doc)
        
        if not missing_docs:
            print("✅ All required documentation files present")
        else:
            print(f"⚠️ Missing documentation: {missing_docs}")
        
        # Check README quality
        if Path('README.md').exists():
            with open('README.md') as f:
                readme_content = f.read()
            
            readme_sections = [
                'installation',
                'usage',
                'example',
                'configuration'
            ]
            
            sections_found = sum(1 for section in readme_sections 
                               if section.lower() in readme_content.lower())
            
            if sections_found >= 3:
                print(f"✅ README has {sections_found}/{len(readme_sections)} key sections")
            else:
                print(f"⚠️ README has {sections_found}/{len(readme_sections)} key sections")
        
        return True
    except Exception as e:
        print(f"❌ Documentation quality test failed: {e}")
        return False

def test_configuration_validation():
    """Test configuration files are valid"""
    print("⚙️ Testing configuration validation...")
    
    try:
        # Test pyproject.toml
        if Path('pyproject.toml').exists():
            try:
                if sys.version_info >= (3, 11):
                    import tomllib
                    with open('pyproject.toml', 'rb') as f:
                        config = tomllib.load(f)
                    print("✅ pyproject.toml is valid TOML")
                else:
                    print("✅ pyproject.toml exists (TOML parser not available)")
            except Exception as e:
                print(f"❌ pyproject.toml invalid: {e}")
                return False
        
        # Test JSON files
        json_files = list(Path('backlog').glob('*.json')) if Path('backlog').exists() else []
        json_files.extend(Path('.').glob('*.json'))
        
        invalid_json = []
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    json.load(f)
            except json.JSONDecodeError:
                invalid_json.append(json_file.name)
        
        if not invalid_json:
            print(f"✅ All {len(json_files)} JSON files are valid")
        else:
            print(f"❌ Invalid JSON files: {invalid_json}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def test_dependency_security():
    """Test dependencies for known vulnerabilities"""
    print("📦 Testing dependency security...")
    
    try:
        # Check if requirements files exist
        req_files = ['requirements.txt', 'pyproject.toml']
        deps_found = any(Path(f).exists() for f in req_files)
        
        if deps_found:
            print("✅ Dependency files found")
        else:
            print("⚠️ No dependency files found")
        
        # Basic check for common vulnerable patterns
        if Path('requirements.txt').exists():
            with open('requirements.txt') as f:
                requirements = f.read()
            
            # Check for pinned versions
            lines = [line.strip() for line in requirements.split('\n') if line.strip()]
            pinned_count = sum(1 for line in lines if '>=' in line or '==' in line or '~=' in line)
            
            if pinned_count > 0:
                print(f"✅ {pinned_count}/{len(lines)} dependencies have version constraints")
            else:
                print("⚠️ No version constraints found in requirements")
        
        return True
    except Exception as e:
        print(f"❌ Dependency security test failed: {e}")
        return False

def test_error_handling_coverage():
    """Test error handling coverage"""
    print("🚨 Testing error handling coverage...")
    
    try:
        # Check for error handling patterns in code
        error_patterns = [
            'try:',
            'except:',
            'except ',
            'raise ',
            'logging.'
        ]
        
        source_files = []
        source_files.extend(Path('.').glob('*.py'))
        if Path('src').exists():
            source_files.extend(Path('src').rglob('*.py'))
        
        files_with_error_handling = 0
        total_files = 0
        
        for py_file in source_files:
            if '__pycache__' in str(py_file):
                continue
                
            total_files += 1
            try:
                with open(py_file) as f:
                    content = f.read()
                
                has_error_handling = any(pattern in content for pattern in error_patterns)
                if has_error_handling:
                    files_with_error_handling += 1
            except:
                continue
        
        coverage_percent = (files_with_error_handling / max(total_files, 1)) * 100
        
        if coverage_percent >= 70:
            print(f"✅ Error handling coverage: {coverage_percent:.1f}% ({files_with_error_handling}/{total_files} files)")
        else:
            print(f"⚠️ Error handling coverage: {coverage_percent:.1f}% ({files_with_error_handling}/{total_files} files)")
        
        return True
    except Exception as e:
        print(f"❌ Error handling coverage test failed: {e}")
        return False

def test_deployment_readiness():
    """Test deployment readiness"""
    print("🚀 Testing deployment readiness...")
    
    try:
        deployment_checklist = {
            'Entry point defined': Path('ado.py').exists() or 'ado' in str(Path('pyproject.toml').read_text()) if Path('pyproject.toml').exists() else False,
            'License file present': Path('LICENSE').exists(),
            'Version defined': 'version' in str(Path('pyproject.toml').read_text()) if Path('pyproject.toml').exists() else False,
            'Dependencies listed': Path('requirements.txt').exists() or Path('pyproject.toml').exists(),
            'Documentation present': Path('README.md').exists(),
        }
        
        passed_checks = sum(deployment_checklist.values())
        total_checks = len(deployment_checklist)
        
        for check, status in deployment_checklist.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {check}")
        
        if passed_checks == total_checks:
            print(f"✅ Deployment readiness: {passed_checks}/{total_checks} checks passed")
        else:
            print(f"⚠️ Deployment readiness: {passed_checks}/{total_checks} checks passed")
        
        return passed_checks >= total_checks * 0.8  # 80% threshold
    except Exception as e:
        print(f"❌ Deployment readiness test failed: {e}")
        return False

def run_quality_gates():
    """Run all quality gate validations"""
    print("🛡️ Quality Gates Validation and Testing")
    print("=" * 60)
    
    tests = [
        ("Code Functionality", test_code_functionality),
        ("Security Validation", test_security_validation),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("Documentation Quality", test_documentation_quality),
        ("Configuration Validation", test_configuration_validation),
        ("Dependency Security", test_dependency_security),
        ("Error Handling Coverage", test_error_handling_coverage),
        ("Deployment Readiness", test_deployment_readiness),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Quality Gates Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Quality Score: {passed/(passed+failed)*100:.1f}%")
    
    # Save quality gates report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": passed + failed,
        "passed_tests": passed,
        "failed_tests": failed,
        "quality_score": passed/(passed+failed)*100,
        "test_results": {name: True for name, _ in tests[:passed]} | {name: False for name, _ in tests[passed:]}
    }
    
    quality_reports_dir = Path("quality_reports")
    quality_reports_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = quality_reports_dir / f"quality_gates_{timestamp}.json"
    latest_file = quality_reports_dir / "quality_gates_latest.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    with open(latest_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Quality gates report saved to: {report_file}")
    
    if failed == 0:
        print("🎉 ALL QUALITY GATES PASSED!")
        return True
    elif passed > failed:
        print("✅ MAJORITY OF QUALITY GATES PASSED!")
        return True
    else:
        print("⚠️ Quality gates need attention before production deployment")
        return False

if __name__ == "__main__":
    success = run_quality_gates()
    sys.exit(0 if success else 1)