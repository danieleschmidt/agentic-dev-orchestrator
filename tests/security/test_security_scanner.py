"""Security scanner tests for ADO.

Tests for security scanning functionality including:
- Dependency vulnerability scanning
- Code security analysis
- Secret detection
- SBOM generation
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ado.security import SecurityScanner, VulnerabilityLevel


class TestSecurityScanner:
    """Test security scanning functionality."""

    @pytest.fixture
    def security_scanner(self):
        """Create a security scanner instance."""
        return SecurityScanner(workspace_path=".")

    @pytest.fixture
    def sample_requirements(self):
        """Create sample requirements file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("requests==2.25.1\n")
            f.write("urllib3==1.26.0\n")
            yield f.name
        os.unlink(f.name)

    def test_scan_dependencies_with_vulnerabilities(self, security_scanner, sample_requirements):
        """Test dependency scanning with known vulnerabilities."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = json.dumps([
                {
                    "package": "urllib3",
                    "version": "1.26.0",
                    "vulnerability": {
                        "id": "CVE-2021-33503",
                        "severity": "HIGH",
                        "summary": "urllib3 before 1.26.5 allows CRLF injection"
                    }
                }
            ])

            results = security_scanner.scan_dependencies(sample_requirements)
            
            assert len(results) == 1
            assert results[0]["package"] == "urllib3"
            assert results[0]["vulnerability"]["severity"] == "HIGH"

    def test_scan_dependencies_no_vulnerabilities(self, security_scanner, sample_requirements):
        """Test dependency scanning with no vulnerabilities."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "[]"

            results = security_scanner.scan_dependencies(sample_requirements)
            
            assert len(results) == 0

    def test_scan_code_for_secrets(self, security_scanner):
        """Test code scanning for hardcoded secrets."""
        test_code = '''
        import os
        
        # This should be detected as a potential secret
        API_KEY = "sk-1234567890abcdef"
        PASSWORD = "super_secret_password"
        
        # This should be okay
        API_KEY = os.environ.get("API_KEY")
        '''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            f.flush()
            
            results = security_scanner.scan_for_secrets(f.name)
            
            # Should detect hardcoded secrets
            assert len(results) >= 1
            secret_types = [r["type"] for r in results]
            assert any("api" in t.lower() or "key" in t.lower() for t in secret_types)
        
        os.unlink(f.name)

    def test_scan_code_quality(self, security_scanner):
        """Test code quality security scanning."""
        test_code = '''
        import subprocess
        import os
        
        def unsafe_function(user_input):
            # This should trigger security warnings
            command = f"ls {user_input}"
            subprocess.call(command, shell=True)  # B602: subprocess with shell=True
            
            # Another security issue
            eval(user_input)  # B307: Use of eval()
        '''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            f.flush()
            
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 1
                mock_run.return_value.stdout = json.dumps({
                    "results": [
                        {
                            "filename": f.name,
                            "issue_severity": "HIGH",
                            "issue_confidence": "HIGH",
                            "issue_text": "subprocess call with shell=True identified",
                            "test_id": "B602",
                            "line_number": 7
                        },
                        {
                            "filename": f.name,
                            "issue_severity": "HIGH", 
                            "issue_confidence": "HIGH",
                            "issue_text": "Use of eval() is dangerous",
                            "test_id": "B307",
                            "line_number": 10
                        }
                    ]
                })
                
                results = security_scanner.scan_code_quality(f.name)
                
                assert len(results) == 2
                severities = [r["issue_severity"] for r in results]
                assert all(s == "HIGH" for s in severities)
        
        os.unlink(f.name)

    def test_generate_sbom(self, security_scanner):
        """Test SBOM (Software Bill of Materials) generation."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = json.dumps({
                "bomFormat": "CycloneDX",
                "specVersion": "1.4",
                "components": [
                    {
                        "type": "library",
                        "name": "requests",
                        "version": "2.28.1",
                        "purl": "pkg:pypi/requests@2.28.1"
                    }
                ]
            })
            
            sbom = security_scanner.generate_sbom()
            
            assert sbom["bomFormat"] == "CycloneDX"
            assert len(sbom["components"]) == 1
            assert sbom["components"][0]["name"] == "requests"

    def test_vulnerability_level_comparison(self):
        """Test vulnerability level enum comparison."""
        assert VulnerabilityLevel.CRITICAL > VulnerabilityLevel.HIGH
        assert VulnerabilityLevel.HIGH > VulnerabilityLevel.MEDIUM
        assert VulnerabilityLevel.MEDIUM > VulnerabilityLevel.LOW
        assert VulnerabilityLevel.LOW > VulnerabilityLevel.INFO

    def test_security_report_generation(self, security_scanner):
        """Test comprehensive security report generation."""
        with patch.object(security_scanner, 'scan_dependencies') as mock_deps:
            with patch.object(security_scanner, 'scan_for_secrets') as mock_secrets:
                with patch.object(security_scanner, 'scan_code_quality') as mock_quality:
                    with patch.object(security_scanner, 'generate_sbom') as mock_sbom:
                        
                        # Mock return values
                        mock_deps.return_value = [{"vulnerability": {"severity": "HIGH"}}]
                        mock_secrets.return_value = [{"type": "api_key"}]
                        mock_quality.return_value = [{"issue_severity": "MEDIUM"}]
                        mock_sbom.return_value = {"components": []}
                        
                        report = security_scanner.generate_security_report()
                        
                        assert "dependencies" in report
                        assert "secrets" in report
                        assert "code_quality" in report
                        assert "sbom" in report
                        assert "summary" in report
                        
                        # Check summary counts
                        assert report["summary"]["total_vulnerabilities"] == 1
                        assert report["summary"]["secrets_found"] == 1
                        assert report["summary"]["quality_issues"] == 1

    def test_security_threshold_checking(self, security_scanner):
        """Test security threshold validation."""
        vulnerabilities = [
            {"vulnerability": {"severity": "CRITICAL"}},
            {"vulnerability": {"severity": "HIGH"}},
            {"vulnerability": {"severity": "MEDIUM"}},
        ]
        
        # Should fail with CRITICAL threshold
        assert not security_scanner.check_security_threshold(
            vulnerabilities, VulnerabilityLevel.CRITICAL
        )
        
        # Should pass with LOW threshold
        assert security_scanner.check_security_threshold(
            vulnerabilities, VulnerabilityLevel.LOW
        )
        
        # Should fail with MEDIUM threshold (has HIGH and CRITICAL)
        assert not security_scanner.check_security_threshold(
            vulnerabilities, VulnerabilityLevel.MEDIUM
        )

    @pytest.mark.parametrize("file_extension,expected_scanner", [
        (".py", "bandit"),
        (".js", "eslint"),
        (".ts", "tslint"),
        (".java", "spotbugs"),
        (".go", "gosec"),
    ])
    def test_language_specific_scanning(self, security_scanner, file_extension, expected_scanner):
        """Test language-specific security scanner selection."""
        scanner = security_scanner.get_scanner_for_file(f"test{file_extension}")
        assert expected_scanner in scanner.lower()

    def test_exclude_patterns(self, security_scanner):
        """Test file exclusion patterns for security scanning."""
        test_files = [
            "src/main.py",
            "tests/test_main.py",
            ".env",
            "secrets.txt",
            "node_modules/package/index.js",
            "venv/lib/python3.9/site-packages/requests.py"
        ]
        
        filtered_files = security_scanner.filter_files(test_files)
        
        # Should exclude .env, secrets.txt, node_modules, and venv
        assert "src/main.py" in filtered_files
        assert "tests/test_main.py" in filtered_files
        assert ".env" not in filtered_files
        assert "secrets.txt" not in filtered_files
        assert not any("node_modules" in f for f in filtered_files)
        assert not any("venv" in f for f in filtered_files)