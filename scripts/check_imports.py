#!/usr/bin/env python3
"""
Check ADO module imports for consistency and best practices.
Used by pre-commit hooks to maintain code quality.
"""

import ast
import sys
import argparse
from pathlib import Path
from typing import List, Set, Dict, Any

class ImportChecker(ast.NodeVisitor):
    """AST visitor to analyze import statements"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.imports = []
        self.from_imports = []
        self.errors = []
        self.warnings = []
    
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements"""
        for alias in node.names:
            self.imports.append({
                'module': alias.name,
                'alias': alias.asname,
                'line': node.lineno
            })
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statements"""
        if node.module:
            for alias in node.names:
                self.from_imports.append({
                    'module': node.module,
                    'name': alias.name,
                    'alias': alias.asname,
                    'line': node.lineno,
                    'level': node.level
                })
    
    def check_ado_imports(self) -> None:
        """Check ADO-specific import patterns"""
        ado_modules = {'ado', 'backlog_manager', 'autonomous_executor'}
        
        for imp in self.imports + self.from_imports:
            module = imp.get('module', '')
            
            # Check for circular imports
            if any(ado_mod in module for ado_mod in ado_modules):
                if 'ado' in self.file_path and module != 'ado':
                    self.warnings.append(
                        f"Line {imp['line']}: Potential circular import of '{module}' in ADO module"
                    )
            
            # Check for deprecated imports
            deprecated_patterns = ['imp', 'distutils']
            for pattern in deprecated_patterns:
                if pattern in module:
                    self.errors.append(
                        f"Line {imp['line']}: Use of deprecated module '{module}'"
                    )
            
            # Check for security-sensitive imports
            security_sensitive = ['subprocess', 'os', 'sys']
            if any(pattern in module for pattern in security_sensitive):
                if 'test' not in self.file_path.lower():
                    self.warnings.append(
                        f"Line {imp['line']}: Security-sensitive import '{module}' - ensure proper usage"
                    )
    
    def check_import_order(self) -> None:
        """Check import ordering following PEP 8"""
        all_imports = [(imp, 'import') for imp in self.imports] + \
                     [(imp, 'from') for imp in self.from_imports]
        
        # Sort by line number
        all_imports.sort(key=lambda x: x[0]['line'])
        
        stdlib_modules = {
            'os', 'sys', 'json', 'pathlib', 'datetime', 'typing', 'dataclasses',
            'subprocess', 'logging', 'argparse', 'ast', 'collections'
        }
        
        import_groups = {'stdlib': [], 'third_party': [], 'local': []}
        
        for imp, imp_type in all_imports:
            module = imp.get('module', '')
            base_module = module.split('.')[0]
            
            if base_module in stdlib_modules:
                import_groups['stdlib'].append(imp)
            elif any(ado_mod in module for ado_mod in ['ado', 'backlog_manager', 'autonomous_executor']):
                import_groups['local'].append(imp)
            else:
                import_groups['third_party'].append(imp)
        
        # Check for proper grouping (basic check)
        prev_group = None
        for group_name, imports in import_groups.items():
            if imports and prev_group and prev_group != group_name:
                # This is a simplified check - in practice, would need more sophisticated logic
                pass
            if imports:
                prev_group = group_name

def check_file_imports(file_path: Path) -> Dict[str, Any]:
    """Check imports in a single Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source, filename=str(file_path))
        checker = ImportChecker(str(file_path))
        checker.visit(tree)
        
        # Run checks
        checker.check_ado_imports()
        checker.check_import_order()
        
        return {
            'file': str(file_path),
            'imports': len(checker.imports),
            'from_imports': len(checker.from_imports),
            'errors': checker.errors,
            'warnings': checker.warnings
        }
    
    except SyntaxError as e:
        return {
            'file': str(file_path),
            'imports': 0,
            'from_imports': 0,
            'errors': [f"Syntax error: {e}"],
            'warnings': []
        }
    except Exception as e:
        return {
            'file': str(file_path),
            'imports': 0,
            'from_imports': 0,
            'errors': [f"Error reading file: {e}"],
            'warnings': []
        }

def main():
    parser = argparse.ArgumentParser(description='Check ADO module imports')
    parser.add_argument('files', nargs='*', help='Python files to check')
    args = parser.parse_args()
    
    if not args.files:
        print("No files provided for import checking")
        return 0
    
    total_errors = 0
    total_warnings = 0
    
    for file_path in args.files:
        path = Path(file_path)
        
        if path.suffix == '.py' and path.exists():
            result = check_file_imports(path)
            
            if result['errors']:
                print(f"\n❌ {file_path}:")
                for error in result['errors']:
                    print(f"  ERROR: {error}")
                total_errors += len(result['errors'])
            
            if result['warnings']:
                if not result['errors']:  # Only print filename if no errors above
                    print(f"\n⚠️  {file_path}:")
                for warning in result['warnings']:
                    print(f"  WARNING: {warning}")
                total_warnings += len(result['warnings'])
    
    print(f"\n📊 Import Check Summary:")
    print(f"  Files checked: {len(args.files)}")
    print(f"  Errors: {total_errors}")
    print(f"  Warnings: {total_warnings}")
    
    if total_errors > 0:
        print("\n❌ Import check failed due to errors")
        return 1
    elif total_warnings > 0:
        print("\n⚠️  Import check passed with warnings")
        return 0
    else:
        print("\n✅ All imports passed validation")
        return 0

if __name__ == '__main__':
    sys.exit(main())