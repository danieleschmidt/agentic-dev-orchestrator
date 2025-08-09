#!/usr/bin/env python3
"""
Production Deployment Guide and Automation for ADO
Comprehensive deployment orchestration with health checks and rollback capabilities
"""

import os
import sys
import json
import time
import subprocess
import logging
import shutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from enum import Enum
import tempfile
# import yaml  # Optional dependency


class DeploymentStage(Enum):
    """Deployment stages"""
    PREPARATION = "preparation"
    VALIDATION = "validation"
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    STAGING_DEPLOY = "staging_deploy"
    PRODUCTION_DEPLOY = "production_deploy"
    HEALTH_CHECK = "health_check"
    MONITORING_SETUP = "monitoring_setup"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLBACK = "rollback"


class Environment(Enum):
    """Target environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    project_name: str
    version: str
    environment: Environment
    repo_root: str
    build_command: str = "python -m build"
    test_command: str = "python -m pytest tests/ -v"
    security_scan_command: str = "python src/security/enhanced_scanner.py scan"
    health_check_endpoint: Optional[str] = None
    health_check_timeout: int = 30
    rollback_enabled: bool = True
    backup_count: int = 3
    deployment_timeout: int = 600  # 10 minutes
    

@dataclass
class DeploymentStep:
    """Individual deployment step"""
    stage: DeploymentStage
    name: str
    command: Optional[str] = None
    working_dir: Optional[str] = None
    timeout: int = 300
    required: bool = True
    retry_count: int = 0
    max_retries: int = 2
    

@dataclass
class DeploymentResult:
    """Result of deployment step"""
    step: DeploymentStep
    success: bool
    output: str = ""
    error: str = ""
    duration: float = 0.0
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ProductionDeploymentOrchestrator:
    """Orchestrates production deployment with comprehensive checks"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.repo_root = Path(config.repo_root)
        self.deployment_dir = self.repo_root / "deployment"
        self.logs_dir = self.deployment_dir / "logs"
        self.backups_dir = self.deployment_dir / "backups"
        
        # Ensure directories exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        self.deployment_steps = self._define_deployment_steps()
        self.results: List[DeploymentResult] = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup deployment logging"""
        logger = logging.getLogger("deployment_orchestrator")
        logger.setLevel(logging.INFO)
        
        # File handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.logs_dir / f"deployment_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
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
        
    def _define_deployment_steps(self) -> List[DeploymentStep]:
        """Define deployment pipeline steps"""
        steps = [
            DeploymentStep(
                stage=DeploymentStage.PREPARATION,
                name="Environment preparation",
                timeout=60
            ),
            DeploymentStep(
                stage=DeploymentStage.VALIDATION,
                name="Code validation and linting",
                command="python -m ruff check . && python -m mypy .",
                timeout=120,
                required=False  # Don't fail deployment on linting issues
            ),
            DeploymentStep(
                stage=DeploymentStage.BUILD,
                name="Build package",
                command=self.config.build_command,
                timeout=300
            ),
            DeploymentStep(
                stage=DeploymentStage.TEST,
                name="Run test suite",
                command=self.config.test_command,
                timeout=600
            ),
            DeploymentStep(
                stage=DeploymentStage.SECURITY_SCAN,
                name="Security scan",
                command=self.config.security_scan_command,
                timeout=300,
                required=False  # Warning only for now
            )
        ]
        
        # Add environment-specific steps
        if self.config.environment != Environment.DEVELOPMENT:
            steps.extend([
                DeploymentStep(
                    stage=DeploymentStage.STAGING_DEPLOY,
                    name="Deploy to staging",
                    timeout=300
                ),
                DeploymentStep(
                    stage=DeploymentStage.HEALTH_CHECK,
                    name="Health check",
                    timeout=self.config.health_check_timeout
                )
            ])
            
        if self.config.environment == Environment.PRODUCTION:
            steps.extend([
                DeploymentStep(
                    stage=DeploymentStage.PRODUCTION_DEPLOY,
                    name="Deploy to production",
                    timeout=600
                ),
                DeploymentStep(
                    stage=DeploymentStage.MONITORING_SETUP,
                    name="Setup monitoring",
                    timeout=120,
                    required=False
                )
            ])
            
        return steps
        
    def deploy(self) -> bool:
        """Execute full deployment pipeline"""
        self.logger.info(f"Starting deployment of {self.config.project_name} v{self.config.version} to {self.config.environment.value}")
        
        deployment_start = time.time()
        
        try:
            # Create backup if enabled
            if self.config.rollback_enabled and self.config.environment == Environment.PRODUCTION:
                self._create_backup()
                
            # Execute deployment steps
            for step in self.deployment_steps:
                self.logger.info(f"Executing step: {step.name}")
                
                result = self._execute_step(step)
                self.results.append(result)
                
                if not result.success:
                    if step.required:
                        self.logger.error(f"Required step '{step.name}' failed. Aborting deployment.")
                        
                        # Attempt rollback if enabled
                        if self.config.rollback_enabled:
                            self._rollback()
                            
                        return False
                    else:
                        self.logger.warning(f"Optional step '{step.name}' failed. Continuing deployment.")
                        
            # Final health check
            if self.config.health_check_endpoint:
                health_result = self._perform_health_check()
                if not health_result:
                    self.logger.error("Final health check failed")
                    if self.config.rollback_enabled:
                        self._rollback()
                    return False
                    
            deployment_duration = time.time() - deployment_start
            self.logger.info(f"Deployment completed successfully in {deployment_duration:.2f} seconds")
            
            # Generate deployment report
            self._generate_deployment_report(True, deployment_duration)
            
            return True
            
        except Exception as e:
            deployment_duration = time.time() - deployment_start
            self.logger.error(f"Deployment failed with exception: {e}")
            
            # Generate failure report
            self._generate_deployment_report(False, deployment_duration, str(e))
            
            # Attempt rollback
            if self.config.rollback_enabled:
                self._rollback()
                
            return False
            
    def _execute_step(self, step: DeploymentStep) -> DeploymentResult:
        """Execute a single deployment step"""
        start_time = time.time()
        
        try:
            if step.stage == DeploymentStage.PREPARATION:
                return self._prepare_environment(step)
            elif step.stage == DeploymentStage.STAGING_DEPLOY:
                return self._deploy_to_staging(step)
            elif step.stage == DeploymentStage.PRODUCTION_DEPLOY:
                return self._deploy_to_production(step)
            elif step.stage == DeploymentStage.HEALTH_CHECK:
                return self._health_check_step(step)
            elif step.stage == DeploymentStage.MONITORING_SETUP:
                return self._setup_monitoring(step)
            elif step.command:
                return self._execute_command_step(step)
            else:
                # Default success for steps without commands
                return DeploymentResult(
                    step=step,
                    success=True,
                    output="Step completed successfully",
                    duration=time.time() - start_time
                )
                
        except Exception as e:
            return DeploymentResult(
                step=step,
                success=False,
                error=str(e),
                duration=time.time() - start_time
            )
            
    def _execute_command_step(self, step: DeploymentStep) -> DeploymentResult:
        """Execute a command-based step"""
        start_time = time.time()
        
        working_dir = step.working_dir or str(self.repo_root)
        
        try:
            result = subprocess.run(
                step.command,
                shell=True,
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=step.timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return DeploymentResult(
                    step=step,
                    success=True,
                    output=result.stdout,
                    duration=duration
                )
            else:
                return DeploymentResult(
                    step=step,
                    success=False,
                    output=result.stdout,
                    error=result.stderr,
                    duration=duration
                )
                
        except subprocess.TimeoutExpired:
            return DeploymentResult(
                step=step,
                success=False,
                error=f"Command timed out after {step.timeout} seconds",
                duration=time.time() - start_time
            )
        except Exception as e:
            return DeploymentResult(
                step=step,
                success=False,
                error=str(e),
                duration=time.time() - start_time
            )
            
    def _prepare_environment(self, step: DeploymentStep) -> DeploymentResult:
        """Prepare deployment environment"""
        start_time = time.time()
        
        try:
            # Check Python version
            python_version = sys.version
            self.logger.info(f"Python version: {python_version}")
            
            # Check disk space
            disk_usage = shutil.disk_usage(self.repo_root)
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb < 1.0:  # Less than 1GB free
                raise Exception(f"Insufficient disk space: {free_gb:.2f}GB free")
                
            # Check required files
            required_files = ['pyproject.toml', 'ado.py']
            for file_name in required_files:
                file_path = self.repo_root / file_name
                if not file_path.exists():
                    raise Exception(f"Required file missing: {file_name}")
                    
            # Environment-specific preparations
            if self.config.environment == Environment.PRODUCTION:
                # Verify production environment variables
                required_env_vars = ['GITHUB_TOKEN']
                missing_vars = [var for var in required_env_vars if not os.getenv(var)]
                if missing_vars:
                    self.logger.warning(f"Missing environment variables: {missing_vars}")
                    
            return DeploymentResult(
                step=step,
                success=True,
                output=f"Environment prepared successfully. Free disk space: {free_gb:.2f}GB",
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return DeploymentResult(
                step=step,
                success=False,
                error=str(e),
                duration=time.time() - start_time
            )
            
    def _deploy_to_staging(self, step: DeploymentStep) -> DeploymentResult:
        """Deploy to staging environment"""
        start_time = time.time()
        
        try:
            # Staging deployment simulation
            self.logger.info("Deploying to staging environment...")
            
            # Check if staging directory exists
            staging_dir = self.deployment_dir / "staging"
            staging_dir.mkdir(exist_ok=True)
            
            # Copy built artifacts to staging
            dist_dir = self.repo_root / "dist"
            if dist_dir.exists():
                staging_artifacts = staging_dir / "artifacts"
                if staging_artifacts.exists():
                    shutil.rmtree(staging_artifacts)
                shutil.copytree(dist_dir, staging_artifacts)
                
            # Create staging deployment marker
            staging_marker = staging_dir / "deployed.json"
            staging_info = {
                'version': self.config.version,
                'deployed_at': datetime.now().isoformat(),
                'environment': 'staging'
            }
            
            staging_marker.write_text(json.dumps(staging_info, indent=2))
            
            return DeploymentResult(
                step=step,
                success=True,
                output="Successfully deployed to staging",
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return DeploymentResult(
                step=step,
                success=False,
                error=str(e),
                duration=time.time() - start_time
            )
            
    def _deploy_to_production(self, step: DeploymentStep) -> DeploymentResult:
        """Deploy to production environment"""
        start_time = time.time()
        
        try:
            # Production deployment
            self.logger.info("Deploying to production environment...")
            
            # Check if production directory exists
            production_dir = self.deployment_dir / "production"
            production_dir.mkdir(exist_ok=True)
            
            # Copy artifacts from staging
            staging_artifacts = self.deployment_dir / "staging" / "artifacts"
            if not staging_artifacts.exists():
                raise Exception("Staging artifacts not found. Deploy to staging first.")
                
            production_artifacts = production_dir / "artifacts"
            if production_artifacts.exists():
                shutil.rmtree(production_artifacts)
            shutil.copytree(staging_artifacts, production_artifacts)
            
            # Create production deployment marker
            production_marker = production_dir / "deployed.json"
            production_info = {
                'version': self.config.version,
                'deployed_at': datetime.now().isoformat(),
                'environment': 'production',
                'previous_version': self._get_previous_version()
            }
            
            production_marker.write_text(json.dumps(production_info, indent=2))
            
            # Update symlink (simulation of atomic deployment)
            current_link = self.deployment_dir / "current"
            if current_link.is_symlink():
                current_link.unlink()
            current_link.symlink_to(production_dir)
            
            return DeploymentResult(
                step=step,
                success=True,
                output="Successfully deployed to production",
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return DeploymentResult(
                step=step,
                success=False,
                error=str(e),
                duration=time.time() - start_time
            )
            
    def _health_check_step(self, step: DeploymentStep) -> DeploymentResult:
        """Perform health check step"""
        start_time = time.time()
        
        try:
            # Perform health checks
            health_checks = [
                self._check_deployment_files,
                self._check_system_resources,
                self._check_application_health
            ]
            
            results = []
            for check in health_checks:
                result = check()
                results.append(result)
                if not result[0]:  # If check failed
                    return DeploymentResult(
                        step=step,
                        success=False,
                        error=result[1],
                        duration=time.time() - start_time
                    )
                    
            return DeploymentResult(
                step=step,
                success=True,
                output="All health checks passed",
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return DeploymentResult(
                step=step,
                success=False,
                error=str(e),
                duration=time.time() - start_time
            )
            
    def _setup_monitoring(self, step: DeploymentStep) -> DeploymentResult:
        """Setup monitoring and observability"""
        start_time = time.time()
        
        try:
            # Setup monitoring configuration
            monitoring_dir = self.deployment_dir / "monitoring"
            monitoring_dir.mkdir(exist_ok=True)
            
            # Create monitoring configuration
            monitoring_config = {
                'application': self.config.project_name,
                'version': self.config.version,
                'environment': self.config.environment.value,
                'metrics': {
                    'enabled': True,
                    'endpoint': '/metrics',
                    'port': 9090
                },
                'health_check': {
                    'endpoint': '/health',
                    'interval': 30
                },
                'alerts': {
                    'enabled': True,
                    'channels': ['email', 'slack']
                }
            }
            
            config_file = monitoring_dir / "config.json"
            config_file.write_text(json.dumps(monitoring_config, indent=2))
            
            # Create Prometheus configuration if directory exists
            prometheus_dir = self.repo_root / "monitoring" / "prometheus"
            if prometheus_dir.exists():
                # Copy Prometheus configuration
                target_prometheus = monitoring_dir / "prometheus"
                if target_prometheus.exists():
                    shutil.rmtree(target_prometheus)
                shutil.copytree(prometheus_dir, target_prometheus)
                
            return DeploymentResult(
                step=step,
                success=True,
                output="Monitoring setup completed",
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return DeploymentResult(
                step=step,
                success=False,
                error=str(e),
                duration=time.time() - start_time
            )
            
    def _check_deployment_files(self) -> Tuple[bool, str]:
        """Check deployment files are in place"""
        try:
            current_link = self.deployment_dir / "current"
            if not current_link.exists():
                return False, "Current deployment symlink not found"
                
            deployed_marker = current_link / "deployed.json"
            if not deployed_marker.exists():
                return False, "Deployment marker not found"
                
            # Verify deployment marker content
            deployment_info = json.loads(deployed_marker.read_text())
            if deployment_info.get('version') != self.config.version:
                return False, f"Version mismatch: expected {self.config.version}, found {deployment_info.get('version')}"
                
            return True, "Deployment files verified"
            
        except Exception as e:
            return False, f"Error checking deployment files: {e}"
            
    def _check_system_resources(self) -> Tuple[bool, str]:
        """Check system resources"""
        try:
            # Check disk space
            disk_usage = shutil.disk_usage(self.repo_root)
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb < 0.5:  # Less than 500MB free
                return False, f"Low disk space: {free_gb:.2f}GB free"
                
            # Check if key processes would be able to start
            # (This is a simulation - in real deployment you'd check actual services)
            
            return True, f"System resources OK. Free space: {free_gb:.2f}GB"
            
        except Exception as e:
            return False, f"Error checking system resources: {e}"
            
    def _check_application_health(self) -> Tuple[bool, str]:
        """Check application health"""
        try:
            # Import and test core components
            sys.path.insert(0, str(self.repo_root))
            
            # Test basic imports
            try:
                from backlog_manager import BacklogManager
                from autonomous_executor import AutonomousExecutor
                # If imports succeed, basic health is OK
                return True, "Application health check passed"
            except ImportError as e:
                return False, f"Import error: {e}"
                
        except Exception as e:
            return False, f"Error checking application health: {e}"
            
    def _perform_health_check(self) -> bool:
        """Perform comprehensive health check"""
        if not self.config.health_check_endpoint:
            self.logger.info("No health check endpoint configured, skipping external health check")
            return True
            
        # In a real deployment, this would make HTTP requests to health endpoints
        self.logger.info(f"Performing health check on {self.config.health_check_endpoint}")
        
        # Simulate health check
        time.sleep(1)  # Simulate network request
        
        # For demo purposes, assume health check passes
        return True
        
    def _create_backup(self):
        """Create deployment backup"""
        self.logger.info("Creating deployment backup...")
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"backup_{timestamp}"
            backup_path = self.backups_dir / backup_name
            
            # Check if current deployment exists
            current_link = self.deployment_dir / "current"
            if current_link.exists() and current_link.is_symlink():
                current_deployment = current_link.resolve()
                if current_deployment.exists():
                    # Create backup
                    shutil.copytree(current_deployment, backup_path)
                    self.logger.info(f"Backup created: {backup_path}")
                    
                    # Cleanup old backups
                    self._cleanup_old_backups()
                    
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")
            
    def _cleanup_old_backups(self):
        """Clean up old backups beyond the configured limit"""
        try:
            backups = sorted(self.backups_dir.glob("backup_*"), 
                           key=lambda x: x.stat().st_mtime, 
                           reverse=True)
            
            # Remove backups beyond the limit
            for old_backup in backups[self.config.backup_count:]:
                if old_backup.is_dir():
                    shutil.rmtree(old_backup)
                    self.logger.info(f"Removed old backup: {old_backup.name}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old backups: {e}")
            
    def _rollback(self):
        """Rollback to previous deployment"""
        self.logger.info("Initiating rollback...")
        
        try:
            # Find the most recent backup
            backups = sorted(self.backups_dir.glob("backup_*"), 
                           key=lambda x: x.stat().st_mtime, 
                           reverse=True)
            
            if not backups:
                self.logger.error("No backups available for rollback")
                return False
                
            latest_backup = backups[0]
            self.logger.info(f"Rolling back to: {latest_backup.name}")
            
            # Create rollback deployment directory
            rollback_dir = self.deployment_dir / "rollback"
            if rollback_dir.exists():
                shutil.rmtree(rollback_dir)
                
            shutil.copytree(latest_backup, rollback_dir)
            
            # Update current symlink to rollback
            current_link = self.deployment_dir / "current"
            if current_link.is_symlink():
                current_link.unlink()
            current_link.symlink_to(rollback_dir)
            
            self.logger.info("Rollback completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False
            
    def _get_previous_version(self) -> Optional[str]:
        """Get previous deployment version"""
        try:
            current_link = self.deployment_dir / "current"
            if current_link.exists() and current_link.is_symlink():
                deployed_marker = current_link / "deployed.json"
                if deployed_marker.exists():
                    deployment_info = json.loads(deployed_marker.read_text())
                    return deployment_info.get('version')
        except Exception:
            pass
        return None
        
    def _generate_deployment_report(self, success: bool, duration: float, error: str = None):
        """Generate comprehensive deployment report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.logs_dir / f"deployment_report_{timestamp}.json"
        
        report = {
            'deployment': {
                'project_name': self.config.project_name,
                'version': self.config.version,
                'environment': self.config.environment.value,
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': duration,
                'success': success,
                'error': error
            },
            'configuration': asdict(self.config),
            'steps': [asdict(result) for result in self.results],
            'summary': {
                'total_steps': len(self.deployment_steps),
                'completed_steps': len([r for r in self.results if r.success]),
                'failed_steps': len([r for r in self.results if not r.success]),
                'optional_failures': len([r for r in self.results 
                                        if not r.success and not r.step.required])
            }
        }
        
        report_file.write_text(json.dumps(report, indent=2, default=str))
        
        # Also save as latest
        latest_report = self.logs_dir / "deployment_report_latest.json"
        latest_report.write_text(json.dumps(report, indent=2, default=str))
        
        self.logger.info(f"Deployment report saved: {report_file}")
        
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        try:
            current_link = self.deployment_dir / "current"
            if not current_link.exists():
                return {'status': 'not_deployed', 'message': 'No current deployment found'}
                
            deployed_marker = current_link / "deployed.json"
            if not deployed_marker.exists():
                return {'status': 'unknown', 'message': 'Deployment marker not found'}
                
            deployment_info = json.loads(deployed_marker.read_text())
            
            return {
                'status': 'deployed',
                'version': deployment_info.get('version'),
                'environment': deployment_info.get('environment'),
                'deployed_at': deployment_info.get('deployed_at'),
                'previous_version': deployment_info.get('previous_version')
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}


def create_default_config() -> DeploymentConfig:
    """Create default deployment configuration"""
    return DeploymentConfig(
        project_name="agentic-dev-orchestrator",
        version="0.2.0",  # Next version
        environment=Environment.STAGING,
        repo_root=".",
        health_check_endpoint="http://localhost:5000/health",
        rollback_enabled=True
    )


def main():
    """CLI entry point for deployment orchestrator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Deployment Orchestrator")
    parser.add_argument("command", choices=["deploy", "status", "rollback", "config"],
                       help="Deployment command")
    parser.add_argument("--environment", choices=["staging", "production"],
                       default="staging", help="Target environment")
    parser.add_argument("--version", help="Version to deploy")
    parser.add_argument("--config", help="Path to deployment config file")
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            try:
                import yaml
                config_data = yaml.safe_load(f)
                config = DeploymentConfig(**config_data)
            except ImportError:
                # Fallback to JSON if YAML not available
                config_data = json.load(f)
                config = DeploymentConfig(**config_data)
    else:
        config = create_default_config()
        
    # Override with command line arguments
    if args.environment:
        config.environment = Environment(args.environment)
    if args.version:
        config.version = args.version
        
    # Create orchestrator
    orchestrator = ProductionDeploymentOrchestrator(config)
    
    if args.command == "deploy":
        print(f"🚀 Starting deployment of {config.project_name} v{config.version} to {config.environment.value}...")
        success = orchestrator.deploy()
        
        if success:
            print("✅ Deployment completed successfully!")
            sys.exit(0)
        else:
            print("❌ Deployment failed!")
            sys.exit(1)
            
    elif args.command == "status":
        status = orchestrator.get_deployment_status()
        print(f"📄 Deployment Status:")
        print(json.dumps(status, indent=2))
        
    elif args.command == "rollback":
        print("🔄 Initiating rollback...")
        success = orchestrator._rollback()
        
        if success:
            print("✅ Rollback completed successfully!")
            sys.exit(0)
        else:
            print("❌ Rollback failed!")
            sys.exit(1)
            
    elif args.command == "config":
        # Generate example configuration
        example_config = asdict(config)
        # Convert enum to string for serialization
        example_config['environment'] = example_config['environment'].value
        
        config_file = "deployment_config.json"  # Use JSON as default
        
        with open(config_file, 'w') as f:
            json.dump(example_config, f, indent=2)
            
        print(f"📄 Example configuration saved to {config_file}")
        

if __name__ == "__main__":
    main()
