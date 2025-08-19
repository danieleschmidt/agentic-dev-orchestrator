#!/usr/bin/env python3
"""
Enhanced Input Validation System
Provides comprehensive security validation for all user inputs and file operations
"""

import re
import json
from pathlib import Path
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    # Create a mock YAMLError for exception handling
    class YAMLError(Exception):
        pass
    yaml = type('MockYAML', (), {'YAMLError': YAMLError})()
from typing import Any, Dict, List, Optional, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_value: Optional[Any] = None


class SecurityInputValidator:
    """Comprehensive security validator for all ADO inputs"""
    
    # Dangerous patterns that should be blocked
    DANGEROUS_PATTERNS = [
        r'__import__',
        r'eval\s*\(',
        r'exec\s*\(',
        r'subprocess\.',
        r'os\.system',
        r'open\s*\(',
        r'file\s*\(',
        r'input\s*\(',
        r'raw_input\s*\(',
        r'\.\./',
        r'[;&|`$]',  # Shell injection
        r'<script',   # XSS
        r'javascript:',
        r'data:',
        r'vbscript:',
    ]
    
    # Safe file extensions
    SAFE_EXTENSIONS = {'.json', '.yml', '.yaml', '.md', '.txt', '.py'}
    
    # Maximum sizes
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_STRING_LENGTH = 100000
    MAX_LIST_LENGTH = 10000
    
    def __init__(self):
        self.pattern_cache = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for performance"""
        for pattern in self.DANGEROUS_PATTERNS:
            try:
                self.pattern_cache[pattern] = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                logger.warning(f"Invalid regex pattern {pattern}: {e}")
    
    def validate_string(self, value: str, max_length: Optional[int] = None) -> ValidationResult:
        """Validate string input for security issues"""
        errors = []
        warnings = []
        
        if not isinstance(value, str):
            errors.append("Value must be a string")
            return ValidationResult(False, errors, warnings)
        
        # Check length
        max_len = max_length or self.MAX_STRING_LENGTH
        if len(value) > max_len:
            errors.append(f"String too long: {len(value)} > {max_len}")
        
        # Check for dangerous patterns
        for pattern_str, compiled_pattern in self.pattern_cache.items():
            if compiled_pattern.search(value):
                errors.append(f"Dangerous pattern detected: {pattern_str}")
        
        # Check for non-printable characters (except common whitespace)
        non_printable = [c for c in value if ord(c) < 32 and c not in '\t\n\r']
        if non_printable:
            warnings.append(f"Non-printable characters found: {len(non_printable)}")
        
        # Sanitize value if no critical errors
        sanitized = value
        if not errors:
            # Basic HTML entity encoding for safety
            sanitized = (value
                        .replace('&', '&amp;')
                        .replace('<', '&lt;')
                        .replace('>', '&gt;')
                        .replace('"', '&quot;')
                        .replace("'", '&#x27;'))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=sanitized
        )
    
    def validate_file_path(self, path: Union[str, Path]) -> ValidationResult:
        """Validate file path for security"""
        errors = []
        warnings = []
        
        path_str = str(path)
        path_obj = Path(path)
        
        # Basic string validation
        string_result = self.validate_string(path_str, max_length=4096)
        if not string_result.is_valid:
            return string_result
        
        # Path traversal checks
        if '..' in path_str:
            errors.append("Path traversal detected")
        
        if path_str.startswith('/'):
            warnings.append("Absolute path detected")
        
        # Check extension
        if path_obj.suffix and path_obj.suffix.lower() not in self.SAFE_EXTENSIONS:
            warnings.append(f"Potentially unsafe file extension: {path_obj.suffix}")
        
        # Resolve path safely
        try:
            resolved_path = path_obj.resolve()
            # Ensure resolved path is within current working directory or subdirs
            cwd = Path.cwd().resolve()
            if not str(resolved_path).startswith(str(cwd)):
                errors.append("Path resolves outside working directory")
        except (OSError, ValueError) as e:
            errors.append(f"Path resolution failed: {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=path_obj
        )
    
    def validate_json_data(self, data: str) -> ValidationResult:
        """Validate and parse JSON data"""
        errors = []
        warnings = []
        
        # Basic string validation
        string_result = self.validate_string(data)
        if not string_result.is_valid:
            return string_result
        
        # JSON parsing
        try:
            parsed_data = json.loads(data)
            
            # Deep validation of parsed structure
            validation_result = self._validate_dict_recursively(parsed_data, max_depth=10)
            errors.extend(validation_result.errors)
            warnings.extend(validation_result.warnings)
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_value=parsed_data
            )
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")
            return ValidationResult(False, errors, warnings)
    
    def validate_yaml_data(self, data: str) -> ValidationResult:
        """Validate and parse YAML data"""
        errors = []
        warnings = []
        
        # Basic string validation
        string_result = self.validate_string(data)
        if not string_result.is_valid:
            return string_result
        
        # YAML parsing with safe loader
        if not YAML_AVAILABLE:
            errors.append("YAML parsing not available - install PyYAML")
            return ValidationResult(False, errors, warnings)
            
        try:
            parsed_data = yaml.safe_load(data)
            
            # Deep validation of parsed structure
            if parsed_data is not None:
                validation_result = self._validate_dict_recursively(parsed_data, max_depth=10)
                errors.extend(validation_result.errors)
                warnings.extend(validation_result.warnings)
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_value=parsed_data
            )
            
        except yaml.YAMLError as e:
            errors.append(f"Invalid YAML: {e}")
            return ValidationResult(False, errors, warnings)
    
    def validate_environment_variable(self, value: str) -> ValidationResult:
        """Validate environment variable value"""
        errors = []
        warnings = []
        
        # Environment variables should be non-empty strings
        if not value or not isinstance(value, str):
            errors.append("Environment variable must be non-empty string")
            return ValidationResult(False, errors, warnings)
        
        # Check for common security issues
        if len(value) < 10:
            warnings.append("Environment variable seems too short for a secure token")
        
        if value.lower() in ['password', 'secret', 'token', 'key']:
            errors.append("Environment variable appears to be a placeholder")
        
        # No dangerous patterns in env vars
        string_result = self.validate_string(value, max_length=8192)
        if not string_result.is_valid:
            return string_result
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_value=value
        )
    
    def _validate_dict_recursively(self, data: Any, max_depth: int, current_depth: int = 0) -> ValidationResult:
        """Recursively validate dictionary/list structures"""
        errors = []
        warnings = []
        
        if current_depth > max_depth:
            errors.append(f"Data structure too deeply nested (> {max_depth})")
            return ValidationResult(False, errors, warnings)
        
        if isinstance(data, dict):
            if len(data) > 1000:
                warnings.append(f"Large dictionary: {len(data)} keys")
            
            for key, value in data.items():
                # Validate keys
                if isinstance(key, str):
                    key_result = self.validate_string(key, max_length=1000)
                    errors.extend([f"Dict key '{key}': {err}" for err in key_result.errors])
                    warnings.extend([f"Dict key '{key}': {warn}" for warn in key_result.warnings])
                
                # Recursively validate values
                value_result = self._validate_dict_recursively(value, max_depth, current_depth + 1)
                errors.extend(value_result.errors)
                warnings.extend(value_result.warnings)
        
        elif isinstance(data, list):
            if len(data) > self.MAX_LIST_LENGTH:
                errors.append(f"List too long: {len(data)} > {self.MAX_LIST_LENGTH}")
            
            for i, item in enumerate(data):
                item_result = self._validate_dict_recursively(item, max_depth, current_depth + 1)
                errors.extend([f"List item [{i}]: {err}" for err in item_result.errors])
                warnings.extend([f"List item [{i}]: {warn}" for warn in item_result.warnings])
        
        elif isinstance(data, str):
            string_result = self.validate_string(data)
            errors.extend(string_result.errors)
            warnings.extend(string_result.warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_file_content(self, file_path: Union[str, Path]) -> ValidationResult:
        """Validate file exists and content is safe"""
        errors = []
        warnings = []
        
        # First validate the path
        path_result = self.validate_file_path(file_path)
        if not path_result.is_valid:
            return path_result
        
        path_obj = Path(file_path)
        
        # Check file exists
        if not path_obj.exists():
            errors.append(f"File does not exist: {file_path}")
            return ValidationResult(False, errors, warnings)
        
        # Check file size
        try:
            file_size = path_obj.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                errors.append(f"File too large: {file_size} > {self.MAX_FILE_SIZE}")
            elif file_size == 0:
                warnings.append("File is empty")
        except OSError as e:
            errors.append(f"Cannot access file: {e}")
        
        # Read and validate content
        if not errors:
            try:
                with open(path_obj, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                content_result = self.validate_string(content)
                errors.extend(content_result.errors)
                warnings.extend(content_result.warnings)
                
                return ValidationResult(
                    is_valid=len(errors) == 0,
                    errors=errors,
                    warnings=warnings,
                    sanitized_value=content
                )
                
            except (UnicodeDecodeError, IOError) as e:
                errors.append(f"Cannot read file: {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


# Global validator instance
validator = SecurityInputValidator()


def validate_input(value: Any, input_type: str = "string", **kwargs) -> ValidationResult:
    """Convenience function for input validation"""
    if input_type == "string":
        return validator.validate_string(str(value), **kwargs)
    elif input_type == "path":
        return validator.validate_file_path(value)
    elif input_type == "json":
        return validator.validate_json_data(str(value))
    elif input_type == "yaml":
        return validator.validate_yaml_data(str(value))
    elif input_type == "env":
        return validator.validate_environment_variable(str(value))
    elif input_type == "file":
        return validator.validate_file_content(value)
    else:
        return ValidationResult(
            is_valid=False,
            errors=[f"Unknown validation type: {input_type}"],
            warnings=[]
        )