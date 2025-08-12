#!/usr/bin/env python3
"""
Enhanced Input Validation and Sanitization System
Comprehensive validation, sanitization, and security checks for robust operation
"""

import re
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Type, get_type_hints
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import inspect
from pathlib import Path
import urllib.parse
import html
import bleach
from email.utils import parseaddr
import ipaddress
import base64


class ValidationSeverity(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationCategory(Enum):
    """Categories of validation checks"""
    FORMAT = "format"
    LENGTH = "length"
    RANGE = "range"
    PATTERN = "pattern"
    SECURITY = "security"
    BUSINESS_LOGIC = "business_logic"
    TYPE = "type"
    ENCODING = "encoding"


@dataclass
class ValidationRule:
    """Individual validation rule"""
    name: str
    category: ValidationCategory
    severity: ValidationSeverity
    validator_func: Callable[[Any], bool]
    error_message: str
    sanitizer_func: Optional[Callable[[Any], Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validation check"""
    is_valid: bool
    original_value: Any
    sanitized_value: Optional[Any] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rule_results: Dict[str, bool] = field(default_factory=dict)
    execution_time: float = 0.0
    
    @property
    def has_errors(self) -> bool:
        """Check if validation has any errors"""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if validation has any warnings"""
        return len(self.warnings) > 0
    
    @property
    def safe_value(self) -> Any:
        """Get safe value (sanitized if available, original if clean)"""
        return self.sanitized_value if self.sanitized_value is not None else self.original_value


class SecurityValidator:
    """Security-focused validation methods"""
    
    # Common malicious patterns
    XSS_PATTERNS = [
        r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',
        r'javascript:',
        r'on\w+\s*=',
        r'data:text/html',
        r'vbscript:',
    ]
    
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(\b(OR|AND)\b\s+\d+\s*=\s*\d+)",
        r"['\";]",
        r"--",
        r"/\*.*\*/",
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$()]",
        r"\b(cat|ls|rm|chmod|sudo|su|passwd)\b",
        r"\.\.\/",
        r"\$\{.*\}",
    ]
    
    @staticmethod
    def is_safe_string(value: str) -> bool:
        """Check if string is safe from XSS attacks"""
        if not isinstance(value, str):
            return False
        
        for pattern in SecurityValidator.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        return True
    
    @staticmethod
    def is_safe_sql_input(value: str) -> bool:
        """Check if string is safe from SQL injection"""
        if not isinstance(value, str):
            return False
        
        for pattern in SecurityValidator.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        return True
    
    @staticmethod
    def is_safe_command_input(value: str) -> bool:
        """Check if string is safe from command injection"""
        if not isinstance(value, str):
            return False
        
        for pattern in SecurityValidator.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return False
        
        return True
    
    @staticmethod
    def sanitize_html(value: str) -> str:
        """Sanitize HTML content"""
        if not isinstance(value, str):
            return str(value)
        
        # Define allowed tags and attributes
        allowed_tags = ['p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        allowed_attributes = {
            '*': ['class'],
            'a': ['href', 'title'],
        }
        
        return bleach.clean(value, tags=allowed_tags, attributes=allowed_attributes, strip=True)
    
    @staticmethod
    def sanitize_filename(value: str) -> str:
        """Sanitize filename to prevent path traversal"""
        if not isinstance(value, str):
            return str(value)
        
        # Remove path traversal attempts
        sanitized = re.sub(r'\.\.\/|\.\.\\', '', value)
        
        # Remove or replace dangerous characters
        sanitized = re.sub(r'[<>:"|?*]', '_', sanitized)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Limit length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:255-len(ext)] + ext
        
        return sanitized.strip()


class EnhancedValidator:
    """Comprehensive validation and sanitization system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.rules: Dict[str, List[ValidationRule]] = {}
        self.global_rules: List[ValidationRule] = []
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.security_validator = SecurityValidator()
        
        # Setup logging
        self.logger = logging.getLogger("enhanced_validator")
        
        # Initialize built-in validation rules
        self._initialize_builtin_rules()
        
        # Validation metrics
        self.metrics = {
            "total_validations": 0,
            "validation_failures": 0,
            "security_violations": 0,
            "sanitizations_applied": 0,
            "cache_hits": 0
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load validation configuration"""
        default_config = {
            "validation": {
                "enabled": True,
                "cache_enabled": True,
                "cache_ttl": 300,  # 5 minutes
                "strict_mode": False,
                "security_checks": True,
                "sanitization": True,
                "max_string_length": 10000,
                "max_list_length": 1000,
                "max_dict_depth": 10
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    if "validation" in loaded_config:
                        default_config["validation"].update(loaded_config["validation"])
            except Exception as e:
                self.logger.warning(f"Could not load validation config: {e}")
        
        return default_config
    
    def _initialize_builtin_rules(self):
        """Initialize built-in validation rules"""
        
        # String validation rules
        self.add_rule("string_length", ValidationRule(
            name="string_length",
            category=ValidationCategory.LENGTH,
            severity=ValidationSeverity.ERROR,
            validator_func=lambda x: isinstance(x, str) and len(x) <= self.config["validation"]["max_string_length"],
            error_message=f"String length exceeds maximum of {self.config['validation']['max_string_length']} characters"
        ))
        
        self.add_rule("xss_check", ValidationRule(
            name="xss_check",
            category=ValidationCategory.SECURITY,
            severity=ValidationSeverity.CRITICAL,
            validator_func=lambda x: not isinstance(x, str) or self.security_validator.is_safe_string(x),
            error_message="Input contains potentially malicious XSS content",
            sanitizer_func=lambda x: self.security_validator.sanitize_html(x) if isinstance(x, str) else x
        ))
        
        self.add_rule("sql_injection_check", ValidationRule(
            name="sql_injection_check",
            category=ValidationCategory.SECURITY,
            severity=ValidationSeverity.CRITICAL,
            validator_func=lambda x: not isinstance(x, str) or self.security_validator.is_safe_sql_input(x),
            error_message="Input contains potential SQL injection patterns"
        ))
        
        self.add_rule("command_injection_check", ValidationRule(
            name="command_injection_check",
            category=ValidationCategory.SECURITY,
            severity=ValidationSeverity.CRITICAL,
            validator_func=lambda x: not isinstance(x, str) or self.security_validator.is_safe_command_input(x),
            error_message="Input contains potential command injection patterns"
        ))
        
        # Email validation rule
        self.add_rule("email_format", ValidationRule(
            name="email_format",
            category=ValidationCategory.FORMAT,
            severity=ValidationSeverity.ERROR,
            validator_func=self._is_valid_email,
            error_message="Invalid email format"
        ))
        
        # URL validation rule
        self.add_rule("url_format", ValidationRule(
            name="url_format",
            category=ValidationCategory.FORMAT,
            severity=ValidationSeverity.ERROR,
            validator_func=self._is_valid_url,
            error_message="Invalid URL format"
        ))
        
        # IP address validation rule
        self.add_rule("ip_address", ValidationRule(
            name="ip_address",
            category=ValidationCategory.FORMAT,
            severity=ValidationSeverity.ERROR,
            validator_func=self._is_valid_ip,
            error_message="Invalid IP address format"
        ))
        
        # JSON validation rule
        self.add_rule("json_format", ValidationRule(
            name="json_format",
            category=ValidationCategory.FORMAT,
            severity=ValidationSeverity.ERROR,
            validator_func=self._is_valid_json,
            error_message="Invalid JSON format"
        ))
    
    def _is_valid_email(self, value: Any) -> bool:
        """Validate email format"""
        if not isinstance(value, str):
            return False
        
        try:
            name, addr = parseaddr(value)
            if '@' not in addr or '.' not in addr.split('@')[1]:
                return False
            
            # Basic regex check
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return re.match(pattern, addr) is not None
        except:
            return False
    
    def _is_valid_url(self, value: Any) -> bool:
        """Validate URL format"""
        if not isinstance(value, str):
            return False
        
        try:
            result = urllib.parse.urlparse(value)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _is_valid_ip(self, value: Any) -> bool:
        """Validate IP address format"""
        if not isinstance(value, str):
            return False
        
        try:
            ipaddress.ip_address(value)
            return True
        except:
            return False
    
    def _is_valid_json(self, value: Any) -> bool:
        """Validate JSON format"""
        if not isinstance(value, str):
            return False
        
        try:
            json.loads(value)
            return True
        except:
            return False
    
    def add_rule(self, field_name: str, rule: ValidationRule):
        """Add validation rule for specific field"""
        if field_name not in self.rules:
            self.rules[field_name] = []
        self.rules[field_name].append(rule)
    
    def add_global_rule(self, rule: ValidationRule):
        """Add global validation rule applied to all fields"""
        self.global_rules.append(rule)
    
    def remove_rule(self, field_name: str, rule_name: str) -> bool:
        """Remove specific validation rule"""
        if field_name in self.rules:
            self.rules[field_name] = [r for r in self.rules[field_name] if r.name != rule_name]
            return True
        return False
    
    def _get_cache_key(self, field_name: str, value: Any) -> str:
        """Generate cache key for validation result"""
        value_str = str(value)[:100]  # Limit cache key size
        return hashlib.sha256(f"{field_name}:{value_str}".encode()).hexdigest()[:16]
    
    def validate_field(self, field_name: str, value: Any, 
                      additional_rules: Optional[List[ValidationRule]] = None) -> ValidationResult:
        """Validate individual field with all applicable rules"""
        
        # Check cache first
        if self.config["validation"]["cache_enabled"]:
            cache_key = self._get_cache_key(field_name, value)
            if cache_key in self.validation_cache:
                self.metrics["cache_hits"] += 1
                return self.validation_cache[cache_key]
        
        start_time = time.time()
        
        result = ValidationResult(
            is_valid=True,
            original_value=value
        )
        
        # Collect all applicable rules
        applicable_rules = []
        applicable_rules.extend(self.global_rules)
        applicable_rules.extend(self.rules.get(field_name, []))
        if additional_rules:
            applicable_rules.extend(additional_rules)
        
        # Apply each rule
        sanitized_value = value
        
        for rule in applicable_rules:
            try:
                is_valid = rule.validator_func(sanitized_value)
                result.rule_results[rule.name] = is_valid
                
                if not is_valid:
                    result.is_valid = False
                    
                    if rule.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                        result.errors.append(rule.error_message)
                        
                        if rule.category == ValidationCategory.SECURITY:
                            self.metrics["security_violations"] += 1
                    else:
                        result.warnings.append(rule.error_message)
                    
                    # Apply sanitization if available
                    if rule.sanitizer_func and self.config["validation"]["sanitization"]:
                        try:
                            sanitized_value = rule.sanitizer_func(sanitized_value)
                            self.metrics["sanitizations_applied"] += 1
                        except Exception as e:
                            self.logger.warning(f"Sanitization failed for rule {rule.name}: {e}")
            
            except Exception as e:
                self.logger.error(f"Validation rule {rule.name} failed: {e}")
                result.errors.append(f"Validation rule {rule.name} failed: {str(e)}")
                result.is_valid = False
        
        # Set sanitized value if any sanitization was applied
        if sanitized_value != value:
            result.sanitized_value = sanitized_value
        
        result.execution_time = time.time() - start_time
        
        # Update metrics
        self.metrics["total_validations"] += 1
        if not result.is_valid:
            self.metrics["validation_failures"] += 1
        
        # Cache result
        if self.config["validation"]["cache_enabled"]:
            self.validation_cache[cache_key] = result
        
        return result
    
    def validate_dict(self, data: Dict[str, Any], 
                     field_rules: Optional[Dict[str, List[ValidationRule]]] = None) -> Dict[str, ValidationResult]:
        """Validate dictionary of fields"""
        results = {}
        
        for field_name, value in data.items():
            additional_rules = field_rules.get(field_name, []) if field_rules else None
            results[field_name] = self.validate_field(field_name, value, additional_rules)
        
        return results
    
    def validate_function_args(self, func: Callable, *args, **kwargs) -> Dict[str, ValidationResult]:
        """Validate function arguments based on type hints and rules"""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        results = {}
        
        # Validate positional arguments
        for i, (param_name, param) in enumerate(sig.parameters.items()):
            if i < len(args):
                value = args[i]
                results[param_name] = self.validate_field(param_name, value)
        
        # Validate keyword arguments
        for param_name, value in kwargs.items():
            if param_name in sig.parameters:
                results[param_name] = self.validate_field(param_name, value)
        
        return results
    
    def create_type_validator(self, expected_type: Type) -> ValidationRule:
        """Create validation rule for specific type"""
        return ValidationRule(
            name=f"type_{expected_type.__name__}",
            category=ValidationCategory.TYPE,
            severity=ValidationSeverity.ERROR,
            validator_func=lambda x: isinstance(x, expected_type),
            error_message=f"Expected type {expected_type.__name__}"
        )
    
    def create_range_validator(self, min_val: Optional[Union[int, float]] = None,
                             max_val: Optional[Union[int, float]] = None) -> ValidationRule:
        """Create validation rule for numeric range"""
        def range_check(value):
            if not isinstance(value, (int, float)):
                return False
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
            return True
        
        range_desc = []
        if min_val is not None:
            range_desc.append(f"min: {min_val}")
        if max_val is not None:
            range_desc.append(f"max: {max_val}")
        
        return ValidationRule(
            name=f"range_{min_val}_{max_val}",
            category=ValidationCategory.RANGE,
            severity=ValidationSeverity.ERROR,
            validator_func=range_check,
            error_message=f"Value out of range ({', '.join(range_desc)})"
        )
    
    def create_pattern_validator(self, pattern: str, flags: int = 0) -> ValidationRule:
        """Create validation rule for regex pattern"""
        compiled_pattern = re.compile(pattern, flags)
        
        return ValidationRule(
            name=f"pattern_{hash(pattern)}",
            category=ValidationCategory.PATTERN,
            severity=ValidationSeverity.ERROR,
            validator_func=lambda x: isinstance(x, str) and compiled_pattern.match(x) is not None,
            error_message=f"Value does not match required pattern"
        )
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation metrics"""
        total = max(self.metrics["total_validations"], 1)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_validations": self.metrics["total_validations"],
            "validation_failures": self.metrics["validation_failures"],
            "failure_rate": self.metrics["validation_failures"] / total,
            "security_violations": self.metrics["security_violations"],
            "sanitizations_applied": self.metrics["sanitizations_applied"],
            "cache_hits": self.metrics["cache_hits"],
            "cache_hit_rate": self.metrics["cache_hits"] / total if total > 0 else 0,
            "total_rules": len(self.global_rules) + sum(len(rules) for rules in self.rules.values())
        }
    
    def export_validation_report(self, output_path: str = "validation-report.md"):
        """Export comprehensive validation report"""
        metrics = self.get_validation_metrics()
        
        with open(output_path, 'w') as f:
            f.write(f"""# 🛡️ Input Validation Analysis Report

Generated: {metrics['timestamp']}
Total Validations: {metrics['total_validations']}
Validation Failure Rate: {metrics['failure_rate']:.1%}

## 📊 Validation Metrics

- **Total Validations**: {metrics['total_validations']}
- **Validation Failures**: {metrics['validation_failures']}
- **Security Violations**: {metrics['security_violations']}
- **Sanitizations Applied**: {metrics['sanitizations_applied']}
- **Cache Hit Rate**: {metrics['cache_hit_rate']:.1%}
- **Total Active Rules**: {metrics['total_rules']}

## 🔒 Security Assessment

- Security violations detected: {metrics['security_violations']}
- Sanitizations automatically applied: {metrics['sanitizations_applied']}
- Security rule coverage: {'✅ Comprehensive' if 'xss_check' in [r.name for rules in self.rules.values() for r in rules] else '⚠️ Limited'}

## 📋 Active Validation Rules

### Global Rules (Applied to all fields)
""")
            
            for rule in self.global_rules:
                f.write(f"- **{rule.name}** ({rule.category.value}): {rule.error_message}\n")
            
            f.write("\n### Field-Specific Rules\n")
            
            for field_name, field_rules in self.rules.items():
                f.write(f"\n**{field_name}**:\n")
                for rule in field_rules:
                    f.write(f"  - {rule.name} ({rule.severity.value}): {rule.error_message}\n")
            
            f.write(f"""
## 💡 Recommendations

""")
            
            if metrics['failure_rate'] > 0.1:
                f.write("- High validation failure rate detected. Review input validation rules.\n")
            
            if metrics['security_violations'] > 0:
                f.write("- Security violations detected. Review and strengthen security validation rules.\n")
            
            if metrics['cache_hit_rate'] < 0.3:
                f.write("- Low cache hit rate. Consider optimizing validation caching.\n")
            
            f.write("\n---\n*Generated by Enhanced Validation System*\n")


def validate_input(**field_rules):
    """
    Decorator for automatic input validation based on function type hints
    """
    def decorator(func: Callable) -> Callable:
        validator = EnhancedValidator()
        
        # Add custom rules for fields
        for field_name, rules in field_rules.items():
            if isinstance(rules, ValidationRule):
                validator.add_rule(field_name, rules)
            elif isinstance(rules, list):
                for rule in rules:
                    validator.add_rule(field_name, rule)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate arguments
            validation_results = validator.validate_function_args(func, *args, **kwargs)
            
            # Check for validation failures
            has_errors = any(result.has_errors for result in validation_results.values())
            
            if has_errors:
                error_messages = []
                for field, result in validation_results.items():
                    if result.has_errors:
                        error_messages.extend([f"{field}: {error}" for error in result.errors])
                
                raise ValueError(f"Validation failed: {'; '.join(error_messages)}")
            
            # Apply sanitization to kwargs
            sanitized_kwargs = {}
            param_names = list(inspect.signature(func).parameters.keys())
            
            for field, result in validation_results.items():
                if field in param_names:
                    sanitized_kwargs[field] = result.safe_value
            
            # Update kwargs with sanitized values
            for field, value in sanitized_kwargs.items():
                if field in kwargs:
                    kwargs[field] = value
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def main():
    """Demonstration of enhanced validation system"""
    print("🛡️ Enhanced Input Validation System Demo")
    print("=" * 50)
    
    # Create validator
    validator = EnhancedValidator()
    
    # Add custom validation rules
    validator.add_rule("username", validator.create_pattern_validator(r'^[a-zA-Z0-9_]{3,20}$'))
    validator.add_rule("age", validator.create_range_validator(min_val=0, max_val=150))
    validator.add_rule("score", validator.create_range_validator(min_val=0.0, max_val=100.0))
    
    # Test data with various validation scenarios
    test_data = {
        "username": "valid_user123",
        "email": "user@example.com",
        "age": 25,
        "score": 95.5,
        "website": "https://example.com",
        "description": "<script>alert('xss')</script>Normal text",
        "ip_address": "192.168.1.1",
        "json_data": '{"key": "value"}',
        "malicious_sql": "'; DROP TABLE users; --",
        "command_injection": "test; rm -rf /"
    }
    
    print("\n🧪 Testing validation on sample data...")
    
    # Validate all fields
    results = validator.validate_dict(test_data)
    
    # Display results
    for field, result in results.items():
        status = "✅" if result.is_valid else "❌"
        value_display = str(result.original_value)[:50] + "..." if len(str(result.original_value)) > 50 else str(result.original_value)
        
        print(f"\n{status} {field}: {value_display}")
        
        if result.has_errors:
            for error in result.errors:
                print(f"     🚨 Error: {error}")
        
        if result.has_warnings:
            for warning in result.warnings:
                print(f"     ⚠️  Warning: {warning}")
        
        if result.sanitized_value is not None and result.sanitized_value != result.original_value:
            sanitized_display = str(result.sanitized_value)[:50] + "..." if len(str(result.sanitized_value)) > 50 else str(result.sanitized_value)
            print(f"     🧽 Sanitized: {sanitized_display}")
    
    # Test decorator
    print("\n🎯 Testing validation decorator...")
    
    @validate_input(
        name=validator.create_pattern_validator(r'^[a-zA-Z\s]{2,50}$'),
        age=validator.create_range_validator(min_val=0, max_val=120)
    )
    def create_user(name: str, age: int, email: str):
        return f"Created user: {name}, {age}, {email}"
    
    # Test valid input
    try:
        result = create_user("John Doe", 30, "john@example.com")
        print(f"✅ Valid input: {result}")
    except Exception as e:
        print(f"❌ Validation error: {e}")
    
    # Test invalid input
    try:
        result = create_user("J", 200, "invalid-email")
        print(f"✅ Should not reach here: {result}")
    except Exception as e:
        print(f"❌ Expected validation error: {e}")
    
    # Show validation metrics
    print("\n📊 Validation Metrics:")
    metrics = validator.get_validation_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # Export validation report
    validator.export_validation_report("demo-validation-report.md")
    print("\n📄 Validation report saved to: demo-validation-report.md")


if __name__ == "__main__":
    import time
    main()