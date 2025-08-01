#!/usr/bin/env python3
"""
Schema validation for ADO backlog items and configuration files.
Used by pre-commit hooks to ensure data integrity.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

BACKLOG_SCHEMA = {
    "type": "object",
    "required": ["title", "wsjf", "description"],
    "properties": {
        "title": {"type": "string", "minLength": 10, "maxLength": 200},
        "description": {"type": "string", "minLength": 20},
        "wsjf": {
            "type": "object",
            "required": ["user_business_value", "time_criticality", "risk_reduction_opportunity_enablement", "job_size"],
            "properties": {
                "user_business_value": {"type": "integer", "minimum": 1, "maximum": 10},
                "time_criticality": {"type": "integer", "minimum": 1, "maximum": 10},
                "risk_reduction_opportunity_enablement": {"type": "integer", "minimum": 1, "maximum": 10},
                "job_size": {"type": "integer", "minimum": 1, "maximum": 20}
            }
        },
        "status": {"type": "string", "enum": ["NEW", "IN_PROGRESS", "DONE", "BLOCKED"]},
        "assignee": {"type": "string"},
        "labels": {"type": "array", "items": {"type": "string"}},
        "dependencies": {"type": "array", "items": {"type": "string"}},
        "acceptance_criteria": {"type": "array", "items": {"type": "string"}}
    }
}

def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any], file_path: str) -> List[str]:
    """Simple schema validation without external dependencies"""
    errors = []
    
    # Check required fields
    for field in schema.get("required", []):
        if field not in data:
            errors.append(f"{file_path}: Missing required field '{field}'")
    
    # Check field types and constraints
    for field, constraints in schema.get("properties", {}).items():
        if field in data:
            value = data[field]
            field_type = constraints.get("type")
            
            if field_type == "string":
                if not isinstance(value, str):
                    errors.append(f"{file_path}: Field '{field}' must be a string")
                elif "minLength" in constraints and len(value) < constraints["minLength"]:
                    errors.append(f"{file_path}: Field '{field}' must be at least {constraints['minLength']} characters")
                elif "maxLength" in constraints and len(value) > constraints["maxLength"]:
                    errors.append(f"{file_path}: Field '{field}' must be at most {constraints['maxLength']} characters")
            
            elif field_type == "integer":
                if not isinstance(value, int):
                    errors.append(f"{file_path}: Field '{field}' must be an integer")
                elif "minimum" in constraints and value < constraints["minimum"]:
                    errors.append(f"{file_path}: Field '{field}' must be at least {constraints['minimum']}")
                elif "maximum" in constraints and value > constraints["maximum"]:
                    errors.append(f"{file_path}: Field '{field}' must be at most {constraints['maximum']}")
            
            elif field_type == "object" and isinstance(value, dict):
                # Recursively validate nested objects
                nested_errors = validate_json_schema(value, constraints, f"{file_path}.{field}")
                errors.extend(nested_errors)
            
            elif field_type == "array":
                if not isinstance(value, list):
                    errors.append(f"{file_path}: Field '{field}' must be an array")
                elif "enum" in constraints.get("items", {}):
                    for item in value:
                        if item not in constraints["items"]["enum"]:
                            errors.append(f"{file_path}: Invalid value '{item}' in field '{field}'")
    
    return errors

def validate_backlog_file(file_path: Path) -> List[str]:
    """Validate a backlog JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return validate_json_schema(data, BACKLOG_SCHEMA, str(file_path))
    
    except json.JSONDecodeError as e:
        return [f"{file_path}: Invalid JSON format: {e}"]
    except Exception as e:
        return [f"{file_path}: Error reading file: {e}"]

def main():
    parser = argparse.ArgumentParser(description='Validate ADO schema files')
    parser.add_argument('files', nargs='*', help='Files to validate')
    args = parser.parse_args()
    
    if not args.files:
        print("No files provided for validation")
        return 0
    
    all_errors = []
    
    for file_path in args.files:
        path = Path(file_path)
        
        if path.suffix == '.json' and 'backlog' in str(path):
            errors = validate_backlog_file(path)
            all_errors.extend(errors)
    
    if all_errors:
        print("Schema validation errors found:")
        for error in all_errors:
            print(f"  ❌ {error}")
        return 1
    else:
        print(f"✅ All {len(args.files)} files passed schema validation")
        return 0

if __name__ == '__main__':
    sys.exit(main())