#!/usr/bin/env python3
"""
Advanced Input Validation and Security Hardening
Comprehensive input validation, sanitization, and security checks for the Terragon SDLC
"""

import re
import json
import hashlib
import hmac
import secrets
import time
import ipaddress
from typing import Any, Dict, List, Optional, Union, Type, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import urllib.parse
from datetime import datetime, timedelta
import base64
import logging
from functools import wraps
import inspect


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised for validation failures"""
    def __init__(self, message: str, field: Optional[str] = None, 
                 error_code: Optional[str] = None):
        super().__init__(message)
        self.field = field
        self.error_code = error_code
        self.timestamp = datetime.utcnow()


class SecurityLevel(Enum):
    """Security levels for validation"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InputType(Enum):
    """Types of input data"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    EMAIL = "email"
    URL = "url"
    IP_ADDRESS = "ip_address"
    UUID = "uuid"
    JSON = "json"
    PATH = "path"
    FILENAME = "filename"
    SQL_QUERY = "sql_query"
    COMMAND = "command"
    REGEX = "regex"


@dataclass
class ValidationRule:
    """Validation rule definition"""
    name: str
    input_type: InputType
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable] = None
    sanitizer: Optional[Callable] = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    description: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of validation operation"""
    is_valid: bool
    sanitized_value: Any = None
    errors: List[str] = None
    warnings: List[str] = None
    security_issues: List[str] = None
    metadata: Dict[str, Any] = None


class SecurityPatterns:
    """Common security patterns and checks"""
    
    # SQL Injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(?i)(union\s+select)",
        r"(?i)(drop\s+table)",
        r"(?i)(delete\s+from)",
        r"(?i)(insert\s+into)",
        r"(?i)(update\s+.+\s+set)",
        r"(?i)(exec\s*\()",
        r"(?i)(execute\s*\()",
        r"(?i)(sp_executesql)",
        r"(?i)(/\*.*\*/)",
        r"(?i)(;.*--)",
        r"(?i)(--.*$)",
        r"(?i)('.*;\s*--)",
        r"(?i)(\b(or|and)\b\s+\w+\s*[=<>])",
        r"(?i)(1\s*=\s*1)",
        r"(?i)('.*'.*=.*')"
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"(?i)(<script[^>]*>)",
        r"(?i)(javascript:)",
        r"(?i)(on\w+\s*=)",
        r"(?i)(<iframe[^>]*>)",
        r"(?i)(<object[^>]*>)",
        r"(?i)(<embed[^>]*>)",
        r"(?i)(<applet[^>]*>)",
        r"(?i)(<meta[^>]*>)",
        r"(?i)(<link[^>]*>)",
        r"(?i)(eval\s*\()",
        r"(?i)(expression\s*\()",
        r"(?i)(vbscript:)",
        r"(?i)(mocha:)",
        r"(?i)(livescript:)"
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$\(\)\[\]\{\}]",
        r"(?i)(\|\s*nc\s)",
        r"(?i)(\|\s*netcat)",
        r"(?i)(>\s*/dev/)",
        r"(?i)(<\s*/dev/)",
        r"(?i)(wget\s+)",
        r"(?i)(curl\s+)",
        r"(?i)(chmod\s+)",
        r"(?i)(rm\s+-)",
        r"(?i)(sudo\s+)",
        r"(?i)(su\s+-)",
        r"(?i)(\$\(.*\))",
        r"(?i)(`.*`)",
        r"(?i)(exec\s*\()"
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\.[\\/]",
        r"[\\/]\.\.[\\/]",
        r"\.\.%2f",
        r"\.\.%5c",
        r"%2e%2e%2f",
        r"%2e%2e%5c",
        r"(?i)(etc[\\/]passwd)",
        r"(?i)(boot\.ini)",
        r"(?i)(windows[\\/]system32)"
    ]
    
    # Dangerous file extensions
    DANGEROUS_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs', '.js', '.jar',
        '.php', '.asp', '.aspx', '.jsp', '.py', '.rb', '.pl', '.sh', '.bash'
    }


class AdvancedInputValidator:
    """Advanced input validation and sanitization system"""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM,
                 rate_limit_window: int = 3600, rate_limit_max: int = 1000):
        self.security_level = security_level
        self.rate_limit_window = rate_limit_window
        self.rate_limit_max = rate_limit_max
        
        # Rate limiting storage
        self.rate_limit_store: Dict[str, List[float]] = {}
        
        # Security patterns
        self.patterns = SecurityPatterns()
        
        # Compiled regex patterns for performance
        self._compile_patterns()
        
        # Validation statistics
        self.stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'security_violations': 0,
            'blocked_ips': set(),
            'start_time': time.time()
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for better performance"""
        self.compiled_sql_patterns = [re.compile(p) for p in self.patterns.SQL_INJECTION_PATTERNS]
        self.compiled_xss_patterns = [re.compile(p) for p in self.patterns.XSS_PATTERNS]
        self.compiled_cmd_patterns = [re.compile(p) for p in self.patterns.COMMAND_INJECTION_PATTERNS]
        self.compiled_path_patterns = [re.compile(p) for p in self.patterns.PATH_TRAVERSAL_PATTERNS]
    
    def validate_input(self, value: Any, rules: Union[ValidationRule, List[ValidationRule]],
                      client_ip: Optional[str] = None) -> ValidationResult:
        """Validate input against rules"""
        
        # Rate limiting check
        if client_ip and not self._check_rate_limit(client_ip):
            self.stats['security_violations'] += 1
            raise ValidationError("Rate limit exceeded", error_code="RATE_LIMIT_EXCEEDED")
        
        # Convert single rule to list
        if isinstance(rules, ValidationRule):
            rules = [rules]
        
        result = ValidationResult(is_valid=True, sanitized_value=value, 
                                errors=[], warnings=[], security_issues=[], metadata={})
        
        self.stats['total_validations'] += 1
        
        try:
            # Apply each validation rule
            for rule in rules:
                rule_result = self._apply_rule(value, rule)
                
                # Merge results
                if not rule_result.is_valid:
                    result.is_valid = False
                
                result.errors.extend(rule_result.errors or [])
                result.warnings.extend(rule_result.warnings or [])
                result.security_issues.extend(rule_result.security_issues or [])
                
                # Update sanitized value
                if rule_result.sanitized_value is not None:
                    result.sanitized_value = rule_result.sanitized_value
                
                # Merge metadata
                if rule_result.metadata:
                    result.metadata.update(rule_result.metadata)
            
            # Security checks
            security_result = self._perform_security_checks(result.sanitized_value)
            result.security_issues.extend(security_result)
            
            if security_result:
                self.stats['security_violations'] += 1
                logger.warning(f"Security issues detected: {security_result}")
            
            if not result.is_valid:
                self.stats['failed_validations'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            result.is_valid = False
            result.errors.append(f"Validation system error: {str(e)}")
            return result
    
    def _apply_rule(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Apply a single validation rule"""
        result = ValidationResult(is_valid=True, sanitized_value=value, 
                                errors=[], warnings=[], security_issues=[], metadata={})
        
        # Required check
        if rule.required and (value is None or value == ""):
            result.is_valid = False
            result.errors.append(f"Field '{rule.name}' is required")
            return result
        
        # Skip further validation if value is empty and not required
        if not rule.required and (value is None or value == ""):
            return result
        
        # Type-specific validation
        try:
            if rule.input_type == InputType.STRING:
                result = self._validate_string(value, rule)
            elif rule.input_type == InputType.INTEGER:
                result = self._validate_integer(value, rule)
            elif rule.input_type == InputType.FLOAT:
                result = self._validate_float(value, rule)
            elif rule.input_type == InputType.BOOLEAN:
                result = self._validate_boolean(value, rule)
            elif rule.input_type == InputType.EMAIL:
                result = self._validate_email(value, rule)
            elif rule.input_type == InputType.URL:
                result = self._validate_url(value, rule)
            elif rule.input_type == InputType.IP_ADDRESS:
                result = self._validate_ip_address(value, rule)
            elif rule.input_type == InputType.UUID:
                result = self._validate_uuid(value, rule)
            elif rule.input_type == InputType.JSON:
                result = self._validate_json(value, rule)
            elif rule.input_type == InputType.PATH:
                result = self._validate_path(value, rule)
            elif rule.input_type == InputType.FILENAME:
                result = self._validate_filename(value, rule)
            elif rule.input_type == InputType.REGEX:
                result = self._validate_regex(value, rule)
            
            # Custom validator
            if result.is_valid and rule.custom_validator:
                custom_result = rule.custom_validator(result.sanitized_value)
                if not custom_result:
                    result.is_valid = False
                    result.errors.append(f"Custom validation failed for '{rule.name}'")
            
            # Sanitization
            if result.is_valid and rule.sanitizer:
                result.sanitized_value = rule.sanitizer(result.sanitized_value)
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation error for '{rule.name}': {str(e)}")
        
        return result
    
    def _validate_string(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate string input"""
        result = ValidationResult(is_valid=True, sanitized_value=str(value), 
                                errors=[], warnings=[], security_issues=[])
        
        str_value = str(value)
        
        # Length checks
        if rule.min_length is not None and len(str_value) < rule.min_length:
            result.is_valid = False
            result.errors.append(f"'{rule.name}' must be at least {rule.min_length} characters")
        
        if rule.max_length is not None and len(str_value) > rule.max_length:
            result.is_valid = False
            result.errors.append(f"'{rule.name}' must not exceed {rule.max_length} characters")
        
        # Pattern check
        if rule.pattern and not re.match(rule.pattern, str_value):
            result.is_valid = False
            result.errors.append(f"'{rule.name}' does not match required pattern")
        
        # Allowed values check
        if rule.allowed_values and str_value not in rule.allowed_values:
            result.is_valid = False
            result.errors.append(f"'{rule.name}' must be one of: {rule.allowed_values}")
        
        # Security sanitization
        if rule.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            result.sanitized_value = self._sanitize_string(str_value)
        
        return result
    
    def _validate_integer(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate integer input"""
        result = ValidationResult(is_valid=True, errors=[])
        
        try:
            int_value = int(value)
            result.sanitized_value = int_value
            
            # Range checks
            if rule.min_value is not None and int_value < rule.min_value:
                result.is_valid = False
                result.errors.append(f"'{rule.name}' must be at least {rule.min_value}")
            
            if rule.max_value is not None and int_value > rule.max_value:
                result.is_valid = False
                result.errors.append(f"'{rule.name}' must not exceed {rule.max_value}")
            
            # Allowed values check
            if rule.allowed_values and int_value not in rule.allowed_values:
                result.is_valid = False
                result.errors.append(f"'{rule.name}' must be one of: {rule.allowed_values}")
                
        except (ValueError, TypeError):
            result.is_valid = False
            result.errors.append(f"'{rule.name}' must be a valid integer")
        
        return result
    
    def _validate_float(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate float input"""
        result = ValidationResult(is_valid=True, errors=[])
        
        try:
            float_value = float(value)
            result.sanitized_value = float_value
            
            # Range checks
            if rule.min_value is not None and float_value < rule.min_value:
                result.is_valid = False
                result.errors.append(f"'{rule.name}' must be at least {rule.min_value}")
            
            if rule.max_value is not None and float_value > rule.max_value:
                result.is_valid = False
                result.errors.append(f"'{rule.name}' must not exceed {rule.max_value}")
                
        except (ValueError, TypeError):
            result.is_valid = False
            result.errors.append(f"'{rule.name}' must be a valid number")
        
        return result
    
    def _validate_boolean(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate boolean input"""
        result = ValidationResult(is_valid=True, errors=[])
        
        if isinstance(value, bool):
            result.sanitized_value = value
        elif isinstance(value, str):
            lower_value = value.lower()
            if lower_value in ['true', '1', 'yes', 'on']:
                result.sanitized_value = True
            elif lower_value in ['false', '0', 'no', 'off']:
                result.sanitized_value = False
            else:
                result.is_valid = False
                result.errors.append(f"'{rule.name}' must be a valid boolean")
        else:
            result.is_valid = False
            result.errors.append(f"'{rule.name}' must be a valid boolean")
        
        return result
    
    def _validate_email(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate email input"""
        result = ValidationResult(is_valid=True, sanitized_value=str(value).lower(), errors=[])
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, str(value)):
            result.is_valid = False
            result.errors.append(f"'{rule.name}' must be a valid email address")
        
        return result
    
    def _validate_url(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate URL input"""
        result = ValidationResult(is_valid=True, sanitized_value=str(value), errors=[], security_issues=[])
        
        try:
            parsed = urllib.parse.urlparse(str(value))
            
            # Basic URL structure check
            if not parsed.scheme or not parsed.netloc:
                result.is_valid = False
                result.errors.append(f"'{rule.name}' must be a valid URL")
                return result
            
            # Security checks for URLs
            if rule.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                # Check for dangerous schemes
                dangerous_schemes = ['javascript', 'data', 'vbscript', 'file']
                if parsed.scheme.lower() in dangerous_schemes:
                    result.security_issues.append(f"Dangerous URL scheme: {parsed.scheme}")
                
                # Check for suspicious patterns
                url_str = str(value).lower()
                if any(pattern in url_str for pattern in ['../', '..\\', '%2e%2e']):
                    result.security_issues.append("Potential path traversal in URL")
            
        except Exception:
            result.is_valid = False
            result.errors.append(f"'{rule.name}' must be a valid URL")
        
        return result
    
    def _validate_ip_address(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate IP address input"""
        result = ValidationResult(is_valid=True, sanitized_value=str(value), errors=[])
        
        try:
            ip = ipaddress.ip_address(str(value))
            result.sanitized_value = str(ip)
            
            # Additional checks for private/reserved IPs if needed
            if rule.security_level == SecurityLevel.CRITICAL:
                if ip.is_private or ip.is_loopback or ip.is_reserved:
                    result.warnings.append(f"'{rule.name}' is a private/reserved IP address")
                    
        except ValueError:
            result.is_valid = False
            result.errors.append(f"'{rule.name}' must be a valid IP address")
        
        return result
    
    def _validate_uuid(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate UUID input"""
        result = ValidationResult(is_valid=True, sanitized_value=str(value), errors=[])
        
        uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        if not re.match(uuid_pattern, str(value)):
            result.is_valid = False
            result.errors.append(f"'{rule.name}' must be a valid UUID")
        
        return result
    
    def _validate_json(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate JSON input"""
        result = ValidationResult(is_valid=True, errors=[])
        
        try:
            if isinstance(value, str):
                json_data = json.loads(value)
                result.sanitized_value = json_data
            elif isinstance(value, (dict, list)):
                result.sanitized_value = value
            else:
                result.is_valid = False
                result.errors.append(f"'{rule.name}' must be valid JSON")
                
        except json.JSONDecodeError:
            result.is_valid = False
            result.errors.append(f"'{rule.name}' must be valid JSON")
        
        return result
    
    def _validate_path(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate file path input"""
        result = ValidationResult(is_valid=True, sanitized_value=str(value), 
                                errors=[], security_issues=[])
        
        path_str = str(value)
        
        # Basic path validation
        try:
            path = Path(path_str)
            result.sanitized_value = str(path.resolve())
        except Exception:
            result.is_valid = False
            result.errors.append(f"'{rule.name}' must be a valid path")
            return result
        
        # Security checks
        if rule.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            # Check for path traversal
            for pattern in self.compiled_path_patterns:
                if pattern.search(path_str):
                    result.security_issues.append("Potential path traversal detected")
                    break
            
            # Check for absolute paths if not allowed
            if path.is_absolute() and rule.metadata and not rule.metadata.get('allow_absolute', False):
                result.security_issues.append("Absolute paths not allowed")
        
        return result
    
    def _validate_filename(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate filename input"""
        result = ValidationResult(is_valid=True, sanitized_value=str(value),
                                errors=[], security_issues=[])
        
        filename = str(value)
        
        # Basic filename checks
        invalid_chars = '<>:"/\\|?*'
        if any(char in filename for char in invalid_chars):
            result.is_valid = False
            result.errors.append(f"'{rule.name}' contains invalid characters")
        
        # Security checks
        if rule.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            # Check file extension
            path = Path(filename)
            if path.suffix.lower() in self.patterns.DANGEROUS_EXTENSIONS:
                result.security_issues.append(f"Potentially dangerous file extension: {path.suffix}")
            
            # Check for hidden files
            if filename.startswith('.') and rule.metadata and not rule.metadata.get('allow_hidden', False):
                result.warnings.append("Hidden file detected")
        
        return result
    
    def _validate_regex(self, value: Any, rule: ValidationRule) -> ValidationResult:
        """Validate regex pattern input"""
        result = ValidationResult(is_valid=True, sanitized_value=str(value), errors=[])
        
        try:
            re.compile(str(value))
        except re.error as e:
            result.is_valid = False
            result.errors.append(f"'{rule.name}' must be a valid regex pattern: {str(e)}")
        
        return result
    
    def _perform_security_checks(self, value: Any) -> List[str]:
        """Perform comprehensive security checks"""
        security_issues = []
        
        if not isinstance(value, str):
            return security_issues
        
        value_str = str(value).lower()
        
        # SQL Injection check
        for pattern in self.compiled_sql_patterns:
            if pattern.search(value_str):
                security_issues.append("Potential SQL injection detected")
                break
        
        # XSS check
        for pattern in self.compiled_xss_patterns:
            if pattern.search(value_str):
                security_issues.append("Potential XSS detected")
                break
        
        # Command injection check
        for pattern in self.compiled_cmd_patterns:
            if pattern.search(value_str):
                security_issues.append("Potential command injection detected")
                break
        
        # Path traversal check
        for pattern in self.compiled_path_patterns:
            if pattern.search(value_str):
                security_issues.append("Potential path traversal detected")
                break
        
        return security_issues
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input"""
        # HTML entity encoding
        value = value.replace('&', '&amp;')
        value = value.replace('<', '&lt;')
        value = value.replace('>', '&gt;')
        value = value.replace('"', '&quot;')
        value = value.replace("'", '&#x27;')
        
        # Remove control characters
        value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        
        # Normalize whitespace
        value = re.sub(r'\s+', ' ', value).strip()
        
        return value
    
    def _check_rate_limit(self, client_ip: str) -> bool:
        """Check rate limiting for client IP"""
        current_time = time.time()
        
        # Clean old entries
        if client_ip in self.rate_limit_store:
            self.rate_limit_store[client_ip] = [
                t for t in self.rate_limit_store[client_ip]
                if current_time - t < self.rate_limit_window
            ]
        else:
            self.rate_limit_store[client_ip] = []
        
        # Check limit
        if len(self.rate_limit_store[client_ip]) >= self.rate_limit_max:
            self.stats['blocked_ips'].add(client_ip)
            return False
        
        # Add current request
        self.rate_limit_store[client_ip].append(current_time)
        return True
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        uptime = time.time() - self.stats['start_time']
        
        return {
            'total_validations': self.stats['total_validations'],
            'failed_validations': self.stats['failed_validations'],
            'security_violations': self.stats['security_violations'],
            'success_rate': (
                (self.stats['total_validations'] - self.stats['failed_validations']) / 
                max(self.stats['total_validations'], 1)
            ),
            'blocked_ips_count': len(self.stats['blocked_ips']),
            'uptime_seconds': uptime,
            'validations_per_second': self.stats['total_validations'] / max(uptime, 1)
        }


def validate_input(func):
    """Decorator for automatic input validation"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Check for validation rules in function annotations
        validator = AdvancedInputValidator()
        
        for param_name, param_value in bound_args.arguments.items():
            param = sig.parameters[param_name]
            if hasattr(param.annotation, '__metadata__'):
                # Extract validation rules from type annotations
                for metadata in param.annotation.__metadata__:
                    if isinstance(metadata, ValidationRule):
                        result = validator.validate_input(param_value, metadata)
                        if not result.is_valid:
                            raise ValidationError(
                                f"Validation failed for parameter '{param_name}': {', '.join(result.errors)}",
                                field=param_name
                            )
                        # Update with sanitized value
                        bound_args.arguments[param_name] = result.sanitized_value
        
        return func(*bound_args.args, **bound_args.kwargs)
    
    return wrapper


# Global validator instance
global_validator = AdvancedInputValidator()


def create_validation_rule(name: str, input_type: InputType, **kwargs) -> ValidationRule:
    """Helper function to create validation rules"""
    return ValidationRule(name=name, input_type=input_type, **kwargs)


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    validator = AdvancedInputValidator(security_level=SecurityLevel.HIGH)
    
    # Test various validations
    test_cases = [
        # Valid cases
        ("John Doe", create_validation_rule("name", InputType.STRING, max_length=50)),
        (25, create_validation_rule("age", InputType.INTEGER, min_value=0, max_value=120)),
        ("john@example.com", create_validation_rule("email", InputType.EMAIL)),
        ("https://example.com", create_validation_rule("website", InputType.URL)),
        
        # Invalid cases
        ("", create_validation_rule("required_field", InputType.STRING, required=True)),
        (-5, create_validation_rule("positive_number", InputType.INTEGER, min_value=0)),
        ("not_an_email", create_validation_rule("email", InputType.EMAIL)),
        
        # Security tests
        ("'; DROP TABLE users; --", create_validation_rule("user_input", InputType.STRING)),
        ("<script>alert('xss')</script>", create_validation_rule("comment", InputType.STRING)),
        ("../../etc/passwd", create_validation_rule("filename", InputType.FILENAME)),
    ]
    
    print("Input Validation Test Results:")
    print("=" * 50)
    
    for i, (test_value, rule) in enumerate(test_cases, 1):
        try:
            result = validator.validate_input(test_value, rule)
            status = "✅ VALID" if result.is_valid else "❌ INVALID"
            print(f"Test {i}: {status}")
            print(f"  Input: {repr(test_value)}")
            print(f"  Rule: {rule.name} ({rule.input_type.value})")
            
            if not result.is_valid:
                print(f"  Errors: {result.errors}")
            
            if result.security_issues:
                print(f"  Security Issues: {result.security_issues}")
            
            if result.sanitized_value != test_value:
                print(f"  Sanitized: {repr(result.sanitized_value)}")
            
            print()
            
        except ValidationError as e:
            print(f"Test {i}: ❌ VALIDATION ERROR")
            print(f"  Input: {repr(test_value)}")
            print(f"  Error: {e}")
            print()
    
    # Print statistics
    stats = validator.get_validation_stats()
    print("Validation Statistics:")
    print("=" * 30)
    for key, value in stats.items():
        print(f"{key}: {value}")