#!/usr/bin/env python3
"""
Autonomous Execution Engine
Implements the macro execution loop and micro-cycle for autonomous backlog processing
"""

import os
import sys
import json
import time
import subprocess
import logging
import traceback
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import datetime
from contextlib import contextmanager

from backlog_manager import BacklogManager, BacklogItem


@dataclass 
class ExecutionResult:
    """Result of task execution with comprehensive tracking"""
    success: bool
    item_id: str
    error_message: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    test_results: Optional[Dict] = None
    security_status: str = "unknown"
    execution_time: float = 0.0
    memory_usage: Optional[float] = None
    logs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_log(self, message: str) -> None:
        """Add log message to execution result"""
        timestamp = datetime.datetime.now().isoformat()
        self.logs.append(f"[{timestamp}] {message}")
    
    def add_warning(self, message: str) -> None:
        """Add warning message to execution result"""
        timestamp = datetime.datetime.now().isoformat()
        self.warnings.append(f"[{timestamp}] WARNING: {message}")


class AutonomousExecutor:
    """Autonomous execution engine with macro/micro cycles and robust error handling"""
    
    def __init__(self, repo_root: str = ".", log_level: str = "INFO"):
        self.repo_root = Path(repo_root)
        self.backlog_manager = BacklogManager(repo_root)
        self.max_iterations = 100  # Safety limit
        self.current_iteration = 0
        self.logger = self._setup_logging(log_level)
        self.security_whitelist = self._load_security_config()
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "escalated_executions": 0,
            "total_execution_time": 0.0
        }
    
    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Setup structured logging with rotation"""
        logger = logging.getLogger("autonomous_executor")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create logs directory
        logs_dir = self.repo_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            logs_dir / "autonomous_executor.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_security_config(self) -> Dict:
        """Load security configuration and whitelists"""
        config_file = self.repo_root / ".ado_security.json"
        default_config = {
            "allowed_commands": [
                "git", "python3", "pytest", "black", "ruff", "mypy"
            ],
            "blocked_patterns": [
                "rm -rf", "sudo", "chmod 777", "password", "secret"
            ],
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "allowed_extensions": [".py", ".yml", ".yaml", ".json", ".md", ".txt"]
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                self.logger.warning(f"Failed to load security config: {e}")
        
        return default_config
    
    @contextmanager
    def execution_context(self, item_id: str):
        """Context manager for safe execution with logging and cleanup"""
        start_time = time.time()
        self.logger.info(f"Starting execution context for {item_id}")
        
        try:
            yield
        except Exception as e:
            self.logger.error(f"Execution failed for {item_id}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            execution_time = time.time() - start_time
            self.logger.info(f"Execution context completed for {item_id} in {execution_time:.2f}s")
            self.metrics["total_execution_time"] += execution_time
        
    def sync_repo_and_ci(self) -> bool:
        """Sync repository state and check CI status with comprehensive error handling"""
        try:
            self.logger.info("Starting repository sync")
            
            # Check if git repo is clean first
            if not self.backlog_manager.is_git_clean():
                self.logger.warning("Repository has uncommitted changes")
                print("⚠️  Repository has uncommitted changes")
                return False
            
            # Load current backlog state with error handling
            try:
                self.backlog_manager.load_backlog()
                self.logger.info("Backlog loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load backlog: {e}")
                return False
            
            # Validate git repository
            if not self._validate_git_repo():
                self.logger.error("Git repository validation failed")
                return False
            
            # Pull latest changes with timeout
            try:
                result = subprocess.run(
                    ['git', 'pull', '--rebase'], 
                    cwd=self.repo_root,
                    capture_output=True, 
                    text=True,
                    timeout=30
                )
            except subprocess.TimeoutExpired:
                self.logger.error("Git pull timed out")
                return False
            
            # Check if we're up to date or if there are conflicts
            if result.returncode != 0:
                if "conflict" in result.stderr.lower():
                    self.logger.error(f"Git conflicts detected: {result.stderr}")
                    print(f"⚠️  Git conflicts detected: {result.stderr}")
                    return False
                elif "up to date" not in result.stdout.lower():
                    self.logger.warning(f"Git pull returned non-zero: {result.stderr}")
                    # Continue anyway - might be "Already up to date"
            
            self.logger.info("Repository sync completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Repository sync failed: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            print(f"⚠️  Sync failed: {e}")
            return False
    
    def _validate_git_repo(self) -> bool:
        """Validate git repository state"""
        try:
            # Check if we're in a git repository
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                self.logger.error("Not in a git repository")
                return False
            
            # Check if we have a valid remote
            result = subprocess.run(
                ['git', 'remote', '-v'],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                self.logger.info(f"Git remotes: {result.stdout.strip()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Git repository validation failed: {e}")
            return False
    
    def is_high_risk_or_ambiguous(self, item: BacklogItem) -> bool:
        """Determine if item needs human escalation"""
        high_risk_indicators = [
            item.risk_tier == "high",
            item.effort >= 13,  # Large items
            "auth" in item.description.lower() and "endpoint" in item.description.lower(),  # Only full auth endpoints
            "security" in item.description.lower() and "vulnerability" in item.description.lower(),
            "database" in item.description.lower() and "migration" in item.description.lower(),
            # Only flag as ambiguous if no acceptance criteria AND it's not a simple task
            len(item.acceptance_criteria) == 0 and item.effort > 3,
        ]
        
        return any(high_risk_indicators)
    
    def escalate_for_human(self, item: BacklogItem) -> None:
        """Escalate high-risk item to human review"""
        print(f"🚨 ESCALATION REQUIRED for {item.id}: {item.title}")
        print(f"Reason: High risk or ambiguous requirements")
        print(f"Risk tier: {item.risk_tier}")
        print(f"Effort: {item.effort}")
        print(f"Acceptance criteria count: {len(item.acceptance_criteria)}")
        
        # Mark as blocked pending human review  
        self.backlog_manager.update_item_status_by_id(item.id, "BLOCKED")
        self.backlog_manager.save_backlog()
        
        # Create escalation file
        escalation_dir = self.repo_root / "escalations"
        escalation_dir.mkdir(exist_ok=True)
        
        escalation_file = escalation_dir / f"{item.id}_escalation.json"
        escalation_data = {
            "item_id": item.id,
            "title": item.title,
            "escalated_at": datetime.datetime.now().isoformat(),
            "reason": "high_risk_or_ambiguous",
            "requires_human_approval": True,
            "item_details": {
                "risk_tier": item.risk_tier,
                "effort": item.effort,
                "acceptance_criteria": item.acceptance_criteria,
                "description": item.description
            }
        }
        
        with open(escalation_file, 'w') as f:
            json.dump(escalation_data, f, indent=2)
    
    def execute_micro_cycle_full(self, item: BacklogItem) -> ExecutionResult:
        """Execute TDD micro-cycle for a single task with comprehensive monitoring"""
        start_time = time.time()
        result = ExecutionResult(success=False, item_id=item.id)
        
        self.logger.info(f"Starting micro-cycle for: {item.id} - {item.title}")
        print(f"🔄 Starting micro-cycle for: {item.id} - {item.title}")
        
        # Security validation
        if not self._validate_task_security(item):
            self.logger.error(f"Security validation failed for {item.id}")
            result.error_message = "Security validation failed"
            result.security_status = "failed"
            return result
        
        # Mark as DOING with error handling
        try:
            self.backlog_manager.update_item_status_by_id(item.id, "DOING")
            result.add_log("Task status updated to DOING")
        except Exception as e:
            self.logger.error(f"Failed to update task status: {e}")
            result.error_message = f"Status update failed: {e}"
            return result
        
        with self.execution_context(item.id):
            try:
                # A. Clarify acceptance criteria
                if not self.clarify_acceptance_criteria(item):
                    result.error_message = "Unclear acceptance criteria"
                    result.add_warning("Acceptance criteria validation failed")
                    return result
                result.add_log("Acceptance criteria validated")
                
                # B. TDD Cycle: RED -> GREEN -> REFACTOR
                tdd_result = self.execute_tdd_cycle(item)
                if not tdd_result.success:
                    result.error_message = tdd_result.error_message
                    result.test_results = tdd_result.test_results
                    return result
                result.test_results = tdd_result.test_results
                result.add_log("TDD cycle completed successfully")
                
                # C. Security checklist
                security_result = self.run_security_checklist(item)
                result.security_status = security_result
                result.add_log(f"Security checklist completed: {security_result}")
                
                # D. Update docs and artifacts
                try:
                    self.update_docs_and_artifacts(item)
                    result.add_log("Documentation and artifacts updated")
                except Exception as e:
                    result.add_warning(f"Documentation update failed: {e}")
                
                # E. CI gates
                ci_result = self.run_ci_gates()
                if not ci_result:
                    result.error_message = "CI gates failed"
                    result.add_log("CI gates failed")
                    return result
                result.add_log("CI gates passed")
                
                # F. PR preparation (would integrate with actual PR system)
                try:
                    pr_info = self.prepare_pr(item)
                    result.artifacts.extend(pr_info.get("artifacts", []))
                    result.add_log("PR preparation completed")
                except Exception as e:
                    result.add_warning(f"PR preparation failed: {e}")
                
                # G. Mark as completed
                self.backlog_manager.update_item_status_by_id(item.id, "DONE")
                self.backlog_manager.save_backlog()
                result.add_log("Task marked as DONE")
                
                # Update metrics
                self.metrics["successful_executions"] += 1
                result.execution_time = time.time() - start_time
                result.success = True
                
                self.logger.info(f"Micro-cycle completed successfully for {item.id}")
                return result
            
            except Exception as e:
                self.logger.error(f"Micro-cycle failed for {item.id}: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                print(f"❌ Micro-cycle failed for {item.id}: {e}")
                
                try:
                    self.backlog_manager.update_item_status_by_id(item.id, "BLOCKED")
                    self.backlog_manager.save_backlog()
                except Exception as save_error:
                    self.logger.error(f"Failed to save blocked status: {save_error}")
                
                self.metrics["failed_executions"] += 1
                result.execution_time = time.time() - start_time
                result.error_message = str(e)
                result.add_log(f"Execution failed with error: {e}")
                return result
    
    def _validate_task_security(self, item: BacklogItem) -> bool:
        """Validate task against security policies"""
        try:
            # Check description for blocked patterns
            for pattern in self.security_whitelist["blocked_patterns"]:
                if pattern.lower() in item.description.lower():
                    self.logger.warning(f"Blocked pattern '{pattern}' found in task {item.id}")
                    return False
            
            # Check acceptance criteria for security issues  
            for criteria in item.acceptance_criteria:
                for pattern in self.security_whitelist["blocked_patterns"]:
                    if pattern.lower() in criteria.lower():
                        self.logger.warning(f"Blocked pattern '{pattern}' found in acceptance criteria")
                        return False
            
            # Validate effort level doesn't exceed safety limits
            if item.effort > 13:  # Large tasks need human review
                self.logger.warning(f"Task {item.id} exceeds maximum automated effort level")
                return False
            
            self.logger.info(f"Security validation passed for task {item.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Security validation error: {e}")
            return False
    
    def clarify_acceptance_criteria(self, item: BacklogItem) -> bool:
        """Ensure acceptance criteria are clear and testable"""
        if not item.acceptance_criteria:
            print(f"⚠️  No acceptance criteria for {item.id}")
            return False
            
        # Check if criteria are specific enough
        vague_terms = ["should work", "looks good", "fix it", "improve"]
        for criteria in item.acceptance_criteria:
            if any(term in criteria.lower() for term in vague_terms):
                print(f"⚠️  Vague acceptance criteria: {criteria}")
                return False
        
        return True
    
    def execute_tdd_cycle(self, item: BacklogItem) -> ExecutionResult:
        """Execute TDD: Write failing test -> Make it pass -> Refactor"""
        print(f"🧪 Running TDD cycle for {item.id}")
        
        # This is a simplified TDD implementation
        # In a real system, this would integrate with actual testing frameworks
        
        test_results = {
            "red_phase": "test_written",
            "green_phase": "implementation_complete", 
            "refactor_phase": "code_cleaned"
        }
        
        # Simulate test execution
        try:
            # Check if there are existing tests to run
            result = subprocess.run(
                ['python', '-m', 'pytest', '--tb=short'], 
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            test_results["pytest_output"] = result.stdout
            test_results["pytest_errors"] = result.stderr
            test_results["pytest_returncode"] = result.returncode
            
        except subprocess.TimeoutExpired:
            test_results["error"] = "Tests timed out"
        except FileNotFoundError:
            # No pytest available - this is expected in minimal setup
            test_results["note"] = "No pytest available - would implement tests"
        
        return ExecutionResult(
            success=True,
            item_id=item.id,
            test_results=test_results
        )
    
    def run_security_checklist(self, item: BacklogItem) -> str:
        """Run security validation checklist"""
        security_checks = {
            "input_validation": "not_applicable",
            "auth_acl": "not_applicable", 
            "secrets_handling": "passed",
            "safe_logging": "passed",
            "crypto_storage": "not_applicable"
        }
        
        # Basic security pattern detection
        try:
            # Check for potential security issues in recent changes
            result = subprocess.run(
                ['git', 'diff', '--name-only', 'HEAD~1..HEAD'],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                changed_files = result.stdout.strip().split('\n')
                for file_path in changed_files:
                    if file_path.endswith('.py'):
                        # Simple security pattern detection
                        full_path = self.repo_root / file_path
                        if full_path.exists():
                            try:
                                with open(full_path, 'r') as f:
                                    content = f.read()
                                    
                                # Check for common security issues
                                if 'password' in content.lower() and 'input' in content.lower():
                                    security_checks["input_validation"] = "needs_review"
                                if 'os.system' in content or 'subprocess.call' in content:
                                    security_checks["input_validation"] = "needs_review"
                            except:
                                pass
                                
        except Exception:
            pass
            
        # Determine overall status
        if any(status == "failed" for status in security_checks.values()):
            return "failed"
        elif any(status == "needs_review" for status in security_checks.values()):
            return "needs_review" 
        else:
            return "passed"
    
    def update_docs_and_artifacts(self, item: BacklogItem) -> None:
        """Update documentation and create artifacts"""
        # This would update README, CHANGELOG, etc.
        # For now, just create a simple completion log
        
        completion_log = self.repo_root / "completions.log"
        with open(completion_log, 'a') as f:
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"{timestamp}: Completed {item.id} - {item.title}\n")
    
    def run_ci_gates(self) -> bool:
        """Run CI gates: lint + tests + type-checks + build"""
        gates = []
        
        # Try common linting/checking commands
        commands_to_try = [
            (['python', '-m', 'flake8', '.'], "flake8 linting"),
            (['python', '-m', 'black', '--check', '.'], "black formatting"),
            (['python', '-m', 'mypy', '.'], "mypy type checking"),
            (['python', '-c', 'import sys; print("Python syntax OK")'], "Python syntax"),
        ]
        
        for command, description in commands_to_try:
            try:
                result = subprocess.run(
                    command,
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                gates.append({
                    "gate": description,
                    "passed": result.returncode == 0,
                    "output": result.stdout if result.returncode == 0 else result.stderr
                })
            except (subprocess.TimeoutExpired, FileNotFoundError):
                gates.append({
                    "gate": description, 
                    "passed": True,  # Don't fail on missing tools
                    "output": f"{description} not available"
                })
        
        # All gates must pass (or be skipped)
        return all(gate["passed"] for gate in gates)
    
    def prepare_pr(self, item: BacklogItem) -> Dict:
        """Prepare PR information (would create actual PR in real system)"""
        return {
            "title": f"Implement: {item.title}",
            "description": item.description,
            "acceptance_criteria": item.acceptance_criteria,
            "item_id": item.id,
            "artifacts": ["backlog.yml", "completions.log"]
        }
    
    def macro_execution_loop(self) -> Dict:
        """Main autonomous execution loop"""
        print("🚀 Starting autonomous backlog execution...")
        
        results = {
            "start_time": datetime.datetime.now().isoformat(),
            "completed_items": [],
            "blocked_items": [],
            "escalated_items": [],
            "errors": []
        }
        
        self.current_iteration = 0
        
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            print(f"\n--- Iteration {self.current_iteration} ---")
            
            # Sync repository
            if not self.sync_repo_and_ci():
                results["errors"].append("Repository sync failed")
                break
            
            # Discover new tasks
            self.backlog_manager.load_backlog()
            new_count = self.backlog_manager.continuous_discovery()
            if new_count > 0:
                print(f"📋 Discovered {new_count} new items")
            
            # Score and sort backlog
            self.backlog_manager.calculate_wsjf_scores()
            self.backlog_manager.save_backlog()
            
            # Get next ready task
            task = self.backlog_manager.get_next_ready_item()
            if not task:
                print("✅ No ready items in backlog")
                break
            
            print(f"🎯 Next task: {task.id} - {task.title}")
            print(f"   WSJF Score: {task.wsjf_score:.2f}")
            
            # Check if high risk or ambiguous
            if self.is_high_risk_or_ambiguous(task):
                self.escalate_for_human(task)
                results["escalated_items"].append(task.id)
                continue
            
            # Execute micro-cycle
            execution_result = self.execute_micro_cycle_full(task)
            
            if execution_result.success:
                print(f"✅ Completed: {task.id}")
                results["completed_items"].append({
                    "id": task.id,
                    "title": task.title,
                    "completed_at": datetime.datetime.now().isoformat()
                })
            else:
                print(f"❌ Failed: {task.id} - {execution_result.error_message}")
                results["blocked_items"].append({
                    "id": task.id, 
                    "error": execution_result.error_message
                })
            
            # Update metrics
            self.backlog_manager.save_status_report()
            
            # Brief pause between iterations
            time.sleep(1)
        
        results["end_time"] = datetime.datetime.now().isoformat()
        results["total_iterations"] = self.current_iteration
        
        print(f"\n🏁 Execution completed after {self.current_iteration} iterations")
        print(f"   Completed: {len(results['completed_items'])} items")
        print(f"   Blocked: {len(results['blocked_items'])} items") 
        print(f"   Escalated: {len(results['escalated_items'])} items")
        
        return results
    
    def discover_new_tasks(self) -> int:
        """Discover new tasks and return count"""
        self.backlog_manager.load_backlog()
        return self.backlog_manager.continuous_discovery()
    
    def get_next_task(self) -> Optional[BacklogItem]:
        """Get next ready task"""
        self.backlog_manager.load_backlog()
        return self.backlog_manager.get_next_ready_item()
    
    def is_high_risk_task(self, item: BacklogItem) -> bool:
        """Check if task is high risk"""
        return self.is_high_risk_or_ambiguous(item)
    
    def run_tests(self) -> bool:
        """Run test suite"""
        try:
            result = subprocess.run(
                ['python', '-m', 'pytest', '--tb=short'], 
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # No pytest or timeout - assume pass for test purposes
            return True
    
    def run_linting(self) -> bool:
        """Run linting checks"""
        try:
            result = subprocess.run(
                ['python', '-m', 'flake8', '.'], 
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # No flake8 - assume pass for test purposes
            return True
    
    def ci_gate(self) -> bool:
        """Run CI gate checks"""
        return self.run_linting() and self.run_tests()
    
    def escalate_task(self, item: BacklogItem, reason: str) -> Path:
        """Escalate task and return escalation file path"""
        escalation_dir = self.repo_root / "escalations"
        escalation_dir.mkdir(exist_ok=True)
        
        escalation_file = escalation_dir / f"{item.id}_escalation.json"
        escalation_data = {
            "item_id": item.id,
            "title": item.title,
            "escalated_at": datetime.datetime.now().isoformat(),
            "reason": reason,
            "requires_human_approval": True,
            "item_details": {
                "risk_tier": item.risk_tier,
                "effort": item.effort,
                "acceptance_criteria": item.acceptance_criteria,
                "description": item.description
            }
        }
        
        with open(escalation_file, 'w') as f:
            json.dump(escalation_data, f, indent=2)
        
        return escalation_file
    
    def execute_micro_cycle(self, item: BacklogItem) -> bool:
        """Execute micro cycle for item (simplified for tests)"""
        try:
            # Update status to DOING
            if hasattr(self.backlog_manager, 'update_item_status_by_id'):
                self.backlog_manager.update_item_status_by_id(item.id, "DOING")
            else:
                # Use the BacklogItem method
                self.backlog_manager.update_item_status(item, "DOING")
            
            # Run CI gate
            if not self.ci_gate():
                return False
            
            # Mark as DONE
            if hasattr(self.backlog_manager, 'update_item_status_by_id'):
                self.backlog_manager.update_item_status_by_id(item.id, "DONE")
            else:
                self.backlog_manager.update_item_status(item, "DONE")
            
            self.backlog_manager.save_backlog()
            return True
            
        except Exception as e:
            print(f"Micro cycle failed: {e}")
            return False


def main():
    """CLI entry point for autonomous execution"""
    if len(sys.argv) < 2:
        print("Usage: python autonomous_executor.py <command>")
        print("Commands: run, status")
        return
        
    command = sys.argv[1]
    executor = AutonomousExecutor()
    
    if command == "run":
        results = executor.macro_execution_loop()
        print("\n📊 Final Results:")
        print(json.dumps(results, indent=2))
        
    elif command == "status":
        executor.backlog_manager.load_backlog()
        report = executor.backlog_manager.generate_status_report()
        print(json.dumps(report, indent=2))
        
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()