#!/usr/bin/env python3
"""
Comprehensive Monitoring and Observability System
Real-time system health monitoring, metrics collection, and alerting for the Terragon SDLC
"""

import asyncio
import json
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque, defaultdict
from threading import Thread, Event
import statistics
import traceback
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    load_average: float
    temperature_celsius: Optional[float] = None


@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    timestamp: float
    requests_per_second: float
    response_time_avg: float
    response_time_p95: float
    response_time_p99: float
    active_tasks: int
    queued_tasks: int
    error_rate: float
    success_rate: float
    throughput_ops_per_sec: float
    cache_hit_rate: float
    memory_usage_mb: float
    open_file_descriptors: int


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    timestamp: datetime
    level: AlertLevel
    component: str
    metric: str
    current_value: Any
    threshold: Any
    message: str
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class HealthCheck:
    """Health check definition"""
    
    def __init__(self, name: str, check_function: Callable, interval: float = 30.0,
                 timeout: float = 10.0, critical: bool = True):
        self.name = name
        self.check_function = check_function
        self.interval = interval
        self.timeout = timeout
        self.critical = critical
        self.last_run: Optional[datetime] = None
        self.last_result: Optional[bool] = None
        self.last_error: Optional[str] = None
        self.consecutive_failures = 0


class ComprehensiveMonitoringSystem:
    """Comprehensive monitoring and observability system"""
    
    def __init__(self, metrics_retention_hours: int = 24, 
                 alert_history_hours: int = 168):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.metrics_retention_hours = metrics_retention_hours
        self.alert_history_hours = alert_history_hours
        self.monitoring_interval = 5.0  # seconds
        
        # Storage
        self.system_metrics: deque = deque(maxlen=int(metrics_retention_hours * 3600 / self.monitoring_interval))
        self.app_metrics: deque = deque(maxlen=int(metrics_retention_hours * 3600 / self.monitoring_interval))
        self.alerts: List[Alert] = []
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # State
        self.monitoring_active = False
        self.shutdown_event = Event()
        self.monitoring_thread: Optional[Thread] = None
        self.health_check_thread: Optional[Thread] = None
        
        # Thresholds (configurable)
        self.thresholds = {
            'cpu_percent': {'warning': 75, 'critical': 90},
            'memory_percent': {'warning': 80, 'critical': 95},
            'disk_usage_percent': {'warning': 85, 'critical': 95},
            'response_time_p95': {'warning': 1000, 'critical': 5000},  # ms
            'error_rate': {'warning': 0.05, 'critical': 0.15},  # 5%, 15%
            'load_average': {'warning': 2.0, 'critical': 4.0}
        }
        
        # Metric collectors
        self.metric_collectors = {
            'requests_processed': 0,
            'requests_failed': 0,
            'response_times': deque(maxlen=1000),
            'start_time': time.time()
        }
        
        # Initialize reports directory
        self.reports_dir = Path("monitoring_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
        self.logger.info("Comprehensive monitoring system initialized")
    
    def start_monitoring(self):
        """Start the monitoring system"""
        if self.monitoring_active:
            self.logger.warning("Monitoring system is already active")
            return
        
        self.monitoring_active = True
        self.shutdown_event.clear()
        
        # Start monitoring thread
        self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start health check thread
        self.health_check_thread = Thread(target=self._health_check_loop, daemon=True)
        self.health_check_thread.start()
        
        self.logger.info("Monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.shutdown_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=5.0)
        
        self.logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active and not self.shutdown_event.is_set():
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Collect application metrics
                app_metrics = self._collect_application_metrics()
                self.app_metrics.append(app_metrics)
                
                # Check thresholds and generate alerts
                self._check_thresholds(system_metrics, app_metrics)
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Save metrics periodically (every 10 cycles)
                if len(self.system_metrics) % 10 == 0:
                    self._save_metrics_report()
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
            
            time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network
            network = psutil.net_io_counters()
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            except (AttributeError, OSError):
                load_avg = 0.0
            
            # Temperature (if available)
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get first available temperature
                    for sensor_name, sensors in temps.items():
                        if sensors:
                            temperature = sensors[0].current
                            break
            except (AttributeError, OSError):
                pass
            
            # Active connections
            try:
                connections = len(psutil.net_connections())
            except (psutil.AccessDenied, OSError):
                connections = 0
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=(disk.used / disk.total) * 100,
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                active_connections=connections,
                load_average=load_avg,
                temperature_celsius=temperature
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            # Return default metrics
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                active_connections=0,
                load_average=0.0
            )
    
    def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics"""
        try:
            current_time = time.time()
            uptime = current_time - self.metric_collectors['start_time']
            
            # Calculate rates
            total_requests = self.metric_collectors['requests_processed']
            failed_requests = self.metric_collectors['requests_failed']
            
            requests_per_second = total_requests / max(uptime, 1) if uptime > 0 else 0
            error_rate = failed_requests / max(total_requests, 1) if total_requests > 0 else 0
            success_rate = 1.0 - error_rate
            
            # Response time statistics
            response_times = list(self.metric_collectors['response_times'])
            if response_times:
                response_time_avg = statistics.mean(response_times)
                response_time_p95 = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times)
                response_time_p99 = statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max(response_times)
            else:
                response_time_avg = response_time_p95 = response_time_p99 = 0.0
            
            # Process-specific metrics
            try:
                process = psutil.Process()
                memory_usage_mb = process.memory_info().rss / (1024 * 1024)
                open_fds = process.num_fds() if hasattr(process, 'num_fds') else 0
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                memory_usage_mb = 0.0
                open_fds = 0
            
            return ApplicationMetrics(
                timestamp=current_time,
                requests_per_second=requests_per_second,
                response_time_avg=response_time_avg,
                response_time_p95=response_time_p95,
                response_time_p99=response_time_p99,
                active_tasks=0,  # Would be set by application
                queued_tasks=0,  # Would be set by application
                error_rate=error_rate,
                success_rate=success_rate,
                throughput_ops_per_sec=requests_per_second,
                cache_hit_rate=0.95,  # Mock value - would be real in production
                memory_usage_mb=memory_usage_mb,
                open_file_descriptors=open_fds
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {e}")
            # Return default metrics
            return ApplicationMetrics(
                timestamp=time.time(),
                requests_per_second=0.0,
                response_time_avg=0.0,
                response_time_p95=0.0,
                response_time_p99=0.0,
                active_tasks=0,
                queued_tasks=0,
                error_rate=0.0,
                success_rate=1.0,
                throughput_ops_per_sec=0.0,
                cache_hit_rate=0.0,
                memory_usage_mb=0.0,
                open_file_descriptors=0
            )
    
    def _check_thresholds(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """Check metrics against thresholds and generate alerts"""
        current_time = datetime.now()
        
        # System metric checks
        checks = [
            ('cpu_percent', system_metrics.cpu_percent, 'System CPU'),
            ('memory_percent', system_metrics.memory_percent, 'System Memory'),
            ('disk_usage_percent', system_metrics.disk_usage_percent, 'Disk Usage'),
            ('load_average', system_metrics.load_average, 'System Load'),
            ('response_time_p95', app_metrics.response_time_p95, 'Response Time P95'),
            ('error_rate', app_metrics.error_rate, 'Error Rate')
        ]
        
        for metric_name, current_value, component in checks:
            if metric_name not in self.thresholds:
                continue
                
            thresholds = self.thresholds[metric_name]
            
            # Check critical threshold
            if current_value >= thresholds.get('critical', float('inf')):
                self._create_alert(
                    level=AlertLevel.CRITICAL,
                    component=component,
                    metric=metric_name,
                    current_value=current_value,
                    threshold=thresholds['critical'],
                    message=f"{component} {metric_name} is critically high: {current_value:.2f} >= {thresholds['critical']}"
                )
            
            # Check warning threshold
            elif current_value >= thresholds.get('warning', float('inf')):
                self._create_alert(
                    level=AlertLevel.WARNING,
                    component=component,
                    metric=metric_name,
                    current_value=current_value,
                    threshold=thresholds['warning'],
                    message=f"{component} {metric_name} is elevated: {current_value:.2f} >= {thresholds['warning']}"
                )
    
    def _create_alert(self, level: AlertLevel, component: str, metric: str, 
                      current_value: Any, threshold: Any, message: str):
        """Create and store an alert"""
        alert_id = f"{component}_{metric}_{int(time.time())}"
        
        # Check if we already have a recent alert for this metric
        recent_alerts = [
            a for a in self.alerts[-10:] 
            if a.component == component and a.metric == metric 
            and not a.resolved 
            and (datetime.now() - a.timestamp).total_seconds() < 300  # 5 minutes
        ]
        
        if recent_alerts:
            return  # Don't spam alerts
        
        alert = Alert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            level=level,
            component=component,
            metric=metric,
            current_value=current_value,
            threshold=threshold,
            message=message
        )
        
        self.alerts.append(alert)
        self.logger.warning(f"Alert created: {message}")
        
        # Auto-resolve old alerts for the same metric if current value is below threshold
        if current_value < threshold:
            for old_alert in self.alerts:
                if (old_alert.component == component and 
                    old_alert.metric == metric and 
                    not old_alert.resolved):
                    old_alert.resolved = True
                    old_alert.resolution_time = datetime.now()
    
    def _cleanup_old_data(self):
        """Clean up old metrics and alerts"""
        cutoff_time = datetime.now() - timedelta(hours=self.alert_history_hours)
        
        # Clean up old alerts
        self.alerts = [
            alert for alert in self.alerts 
            if alert.timestamp > cutoff_time
        ]
    
    def add_health_check(self, name: str, check_function: Callable, 
                        interval: float = 30.0, timeout: float = 10.0, 
                        critical: bool = True):
        """Add a health check"""
        self.health_checks[name] = HealthCheck(
            name=name,
            check_function=check_function,
            interval=interval,
            timeout=timeout,
            critical=critical
        )
        self.logger.info(f"Added health check: {name}")
    
    def _health_check_loop(self):
        """Health check execution loop"""
        while self.monitoring_active and not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()
                
                for health_check in self.health_checks.values():
                    # Check if it's time to run this health check
                    if (health_check.last_run is None or 
                        (current_time - health_check.last_run).total_seconds() >= health_check.interval):
                        
                        self._run_health_check(health_check)
                        
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}", exc_info=True)
            
            time.sleep(5.0)  # Check every 5 seconds
    
    def _run_health_check(self, health_check: HealthCheck):
        """Execute a single health check"""
        try:
            start_time = time.time()
            
            # Run the check with timeout
            try:
                result = health_check.check_function()
                if asyncio.iscoroutine(result):
                    # Handle async functions
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            asyncio.wait_for(result, timeout=health_check.timeout)
                        )
                    finally:
                        loop.close()
                
                health_check.last_result = bool(result)
                health_check.last_error = None
                health_check.consecutive_failures = 0
                
            except Exception as e:
                health_check.last_result = False
                health_check.last_error = str(e)
                health_check.consecutive_failures += 1
                
                # Create alert for failed health check
                if health_check.critical and health_check.consecutive_failures >= 3:
                    self._create_alert(
                        level=AlertLevel.CRITICAL,
                        component="Health Check",
                        metric=health_check.name,
                        current_value="FAILED",
                        threshold="PASS",
                        message=f"Critical health check '{health_check.name}' failed {health_check.consecutive_failures} times: {health_check.last_error}"
                    )
            
            health_check.last_run = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error running health check {health_check.name}: {e}")
    
    def record_request(self, response_time: float, success: bool = True):
        """Record a request for metrics"""
        self.metric_collectors['requests_processed'] += 1
        if not success:
            self.metric_collectors['requests_failed'] += 1
        
        self.metric_collectors['response_times'].append(response_time)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system state"""
        try:
            latest_system = self.system_metrics[-1] if self.system_metrics else None
            latest_app = self.app_metrics[-1] if self.app_metrics else None
            
            # Recent alerts (last 24 hours)
            recent_alerts = [
                alert for alert in self.alerts
                if (datetime.now() - alert.timestamp).total_seconds() < 86400
            ]
            
            # Health check status
            health_status = {}
            for name, check in self.health_checks.items():
                health_status[name] = {
                    'status': 'PASS' if check.last_result else 'FAIL',
                    'last_run': check.last_run.isoformat() if check.last_run else None,
                    'consecutive_failures': check.consecutive_failures,
                    'last_error': check.last_error
                }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'monitoring_active': self.monitoring_active,
                'system_metrics': asdict(latest_system) if latest_system else None,
                'application_metrics': asdict(latest_app) if latest_app else None,
                'alerts': {
                    'total_active': len([a for a in recent_alerts if not a.resolved]),
                    'critical': len([a for a in recent_alerts if a.level == AlertLevel.CRITICAL and not a.resolved]),
                    'warning': len([a for a in recent_alerts if a.level == AlertLevel.WARNING and not a.resolved]),
                    'recent': [asdict(a) for a in recent_alerts[-5:]]
                },
                'health_checks': health_status,
                'uptime_seconds': time.time() - self.metric_collectors['start_time']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current metrics: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'monitoring_active': self.monitoring_active
            }
    
    def _save_metrics_report(self):
        """Save metrics report to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.reports_dir / f"monitoring_report_{timestamp}.json"
            
            # Prepare report data
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'system_metrics': [asdict(m) for m in list(self.system_metrics)[-100:]],  # Last 100 samples
                'application_metrics': [asdict(m) for m in list(self.app_metrics)[-100:]],
                'recent_alerts': [asdict(a) for a in self.alerts[-50:]],  # Last 50 alerts
                'health_checks': {
                    name: {
                        'status': 'PASS' if check.last_result else 'FAIL',
                        'last_run': check.last_run.isoformat() if check.last_run else None,
                        'consecutive_failures': check.consecutive_failures,
                        'last_error': check.last_error
                    }
                    for name, check in self.health_checks.items()
                }
            }
            
            # Save detailed report
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            # Save latest report
            latest_file = self.reports_dir / "latest_monitoring_report.json"
            with open(latest_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
        except Exception as e:
            self.logger.error(f"Error saving metrics report: {e}")
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        try:
            current_metrics = self.get_current_metrics()
            
            # System health assessment
            system_health = "HEALTHY"
            critical_issues = []
            warnings = []
            
            # Check system metrics
            if current_metrics.get('system_metrics'):
                sys_metrics = current_metrics['system_metrics']
                
                if sys_metrics['cpu_percent'] > 90:
                    critical_issues.append(f"Critical CPU usage: {sys_metrics['cpu_percent']:.1f}%")
                    system_health = "CRITICAL"
                elif sys_metrics['cpu_percent'] > 75:
                    warnings.append(f"High CPU usage: {sys_metrics['cpu_percent']:.1f}%")
                    if system_health == "HEALTHY":
                        system_health = "WARNING"
                
                if sys_metrics['memory_percent'] > 95:
                    critical_issues.append(f"Critical memory usage: {sys_metrics['memory_percent']:.1f}%")
                    system_health = "CRITICAL"
                elif sys_metrics['memory_percent'] > 80:
                    warnings.append(f"High memory usage: {sys_metrics['memory_percent']:.1f}%")
                    if system_health == "HEALTHY":
                        system_health = "WARNING"
            
            # Check active alerts
            active_alerts = current_metrics.get('alerts', {})
            if active_alerts.get('critical', 0) > 0:
                system_health = "CRITICAL"
                critical_issues.append(f"{active_alerts['critical']} critical alerts active")
            elif active_alerts.get('warning', 0) > 0:
                if system_health == "HEALTHY":
                    system_health = "WARNING"
                warnings.append(f"{active_alerts['warning']} warning alerts active")
            
            # Check health checks
            failed_health_checks = [
                name for name, status in current_metrics.get('health_checks', {}).items()
                if status['status'] == 'FAIL'
            ]
            
            if failed_health_checks:
                system_health = "CRITICAL"
                critical_issues.extend([f"Health check failed: {name}" for name in failed_health_checks])
            
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_health': system_health,
                'critical_issues': critical_issues,
                'warnings': warnings,
                'metrics_summary': current_metrics,
                'uptime_hours': (time.time() - self.metric_collectors['start_time']) / 3600,
                'monitoring_status': 'ACTIVE' if self.monitoring_active else 'INACTIVE'
            }
            
        except Exception as e:
            self.logger.error(f"Error generating health report: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_health': 'ERROR',
                'error': str(e)
            }


# Global monitoring instance
monitoring_system = ComprehensiveMonitoringSystem()


# Decorator for monitoring function calls
def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        success = True
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            raise
        finally:
            duration = time.time() - start_time
            monitoring_system.record_request(duration * 1000, success)  # Convert to milliseconds
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        success = True
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            raise
        finally:
            duration = time.time() - start_time
            monitoring_system.record_request(duration * 1000, success)  # Convert to milliseconds
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return wrapper


def setup_default_health_checks():
    """Setup default health checks"""
    
    def check_disk_space():
        """Check if disk space is available"""
        disk = psutil.disk_usage('/')
        free_percent = (disk.free / disk.total) * 100
        return free_percent > 10  # At least 10% free space
    
    def check_memory():
        """Check if memory is not critically low"""
        memory = psutil.virtual_memory()
        return memory.percent < 95  # Less than 95% memory usage
    
    def check_file_descriptors():
        """Check if file descriptor limit is not reached"""
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            current = len(psutil.Process().open_files())
            return current < (soft * 0.8)  # Less than 80% of limit
        except (ImportError, psutil.AccessDenied):
            return True  # Assume OK if we can't check
    
    monitoring_system.add_health_check("disk_space", check_disk_space, interval=60.0)
    monitoring_system.add_health_check("memory_usage", check_memory, interval=30.0)
    monitoring_system.add_health_check("file_descriptors", check_file_descriptors, interval=60.0)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Setup monitoring
    setup_default_health_checks()
    monitoring_system.start_monitoring()
    
    try:
        # Keep running for demo
        time.sleep(60)
        
        # Print health report
        health_report = monitoring_system.generate_health_report()
        print(json.dumps(health_report, indent=2))
        
    finally:
        monitoring_system.stop_monitoring()