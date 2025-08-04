#!/usr/bin/env python3
"""
Health monitoring and system metrics collection
"""

import json
import time
import psutil
import datetime
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import logging


@dataclass
class HealthMetrics:
    """System health metrics"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    load_average: Optional[List[float]]
    git_status: str
    backlog_items_total: int
    backlog_items_ready: int
    backlog_items_doing: int
    backlog_items_done: int
    backlog_items_blocked: int
    execution_success_rate: float
    avg_execution_time: float


class HealthMonitor:
    """Comprehensive health monitoring system"""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.metrics_dir = self.repo_root / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger("health_monitor")
        
    def collect_system_metrics(self) -> Dict:
        """Collect system performance metrics"""
        try:
            metrics = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage(str(self.repo_root)).percent,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Load average (Unix-like systems only)
            try:
                metrics["load_average"] = list(psutil.getloadavg())
            except AttributeError:
                metrics["load_average"] = None
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return {}
    
    def collect_git_metrics(self) -> Dict:
        """Collect git repository health metrics"""
        try:
            import subprocess
            
            # Check git status
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                if result.stdout.strip():
                    git_status = "dirty"
                else:
                    git_status = "clean" 
            else:
                git_status = "error"
                
            # Check commit count
            result = subprocess.run(
                ['git', 'rev-list', '--count', 'HEAD'],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            commit_count = int(result.stdout.strip()) if result.returncode == 0 else 0
            
            return {
                "git_status": git_status,
                "commit_count": commit_count,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect git metrics: {e}")
            return {"git_status": "error", "commit_count": 0}
    
    def collect_backlog_metrics(self) -> Dict:
        """Collect backlog health metrics"""
        try:
            import sys
            import os
            
            # Add repo root to Python path
            repo_root_str = str(self.repo_root)
            if repo_root_str not in sys.path:
                sys.path.insert(0, repo_root_str)
            
            from backlog_manager import BacklogManager
            
            manager = BacklogManager(str(self.repo_root))
            manager.load_backlog()
            
            status_counts = {}
            for item in manager.items:
                status_counts[item.status] = status_counts.get(item.status, 0) + 1
            
            return {
                "total_items": len(manager.items),
                "ready_items": status_counts.get("READY", 0),
                "doing_items": status_counts.get("DOING", 0),
                "done_items": status_counts.get("DONE", 0),
                "blocked_items": status_counts.get("BLOCKED", 0),
                "new_items": status_counts.get("NEW", 0),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect backlog metrics: {e}")
            return {"total_items": 0, "ready_items": 0}
    
    def collect_execution_metrics(self) -> Dict:
        """Collect execution performance metrics"""
        try:
            # Read execution history from status files
            status_dir = self.repo_root / "docs" / "status"
            if not status_dir.exists():
                return {"success_rate": 0.0, "avg_execution_time": 0.0}
            
            execution_data = []
            for status_file in status_dir.glob("status_*.json"):
                try:
                    with open(status_file, 'r') as f:
                        data = json.load(f)
                        execution_data.append(data)
                except Exception:
                    continue
            
            if not execution_data:
                return {"success_rate": 0.0, "avg_execution_time": 0.0}
            
            # Calculate success rate
            total_executions = len(execution_data)
            successful_executions = sum(1 for data in execution_data 
                                      if len(data.get("completed_items", [])) > 0)
            
            success_rate = successful_executions / total_executions if total_executions > 0 else 0.0
            
            # Calculate average execution time (placeholder)
            avg_execution_time = 2.5  # Would calculate from actual timing data
            
            return {
                "success_rate": success_rate,
                "avg_execution_time": avg_execution_time,
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect execution metrics: {e}")
            return {"success_rate": 0.0, "avg_execution_time": 0.0}
    
    def generate_health_report(self) -> HealthMetrics:
        """Generate comprehensive health report"""
        system_metrics = self.collect_system_metrics()
        git_metrics = self.collect_git_metrics()
        backlog_metrics = self.collect_backlog_metrics()
        execution_metrics = self.collect_execution_metrics()
        
        return HealthMetrics(
            timestamp=datetime.datetime.now().isoformat(),
            cpu_percent=system_metrics.get("cpu_percent", 0.0),
            memory_percent=system_metrics.get("memory_percent", 0.0),
            disk_percent=system_metrics.get("disk_percent", 0.0),
            load_average=system_metrics.get("load_average"),
            git_status=git_metrics.get("git_status", "unknown"),
            backlog_items_total=backlog_metrics.get("total_items", 0),
            backlog_items_ready=backlog_metrics.get("ready_items", 0),
            backlog_items_doing=backlog_metrics.get("doing_items", 0),
            backlog_items_done=backlog_metrics.get("done_items", 0),
            backlog_items_blocked=backlog_metrics.get("blocked_items", 0),
            execution_success_rate=execution_metrics.get("success_rate", 0.0),
            avg_execution_time=execution_metrics.get("avg_execution_time", 0.0)
        )
    
    def save_health_report(self, metrics: HealthMetrics) -> Path:
        """Save health report to file"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        health_file = self.metrics_dir / f"health_{timestamp}.json"
        
        with open(health_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        # Save as latest
        latest_file = self.metrics_dir / "health_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        return health_file
    
    def check_health_thresholds(self, metrics: HealthMetrics) -> List[str]:
        """Check metrics against health thresholds"""
        alerts = []
        
        # System resource alerts
        if metrics.cpu_percent > 80:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > 85:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.disk_percent > 90:
            alerts.append(f"High disk usage: {metrics.disk_percent:.1f}%")
        
        # Execution alerts
        if metrics.execution_success_rate < 0.8:
            alerts.append(f"Low success rate: {metrics.execution_success_rate:.1%}")
        
        if metrics.backlog_items_blocked > 3:
            alerts.append(f"High blocked items: {metrics.backlog_items_blocked}")
        
        if metrics.git_status == "dirty":
            alerts.append("Repository has uncommitted changes")
        
        return alerts


def main():
    """CLI entry point for health monitoring"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python health_monitor.py <command>")
        print("Commands: collect, report, alerts")
        return
    
    command = sys.argv[1]
    monitor = HealthMonitor()
    
    if command == "collect":
        metrics = monitor.generate_health_report()
        report_file = monitor.save_health_report(metrics)
        print(f"Health report saved to: {report_file}")
        
    elif command == "report":
        metrics = monitor.generate_health_report()
        print(json.dumps(asdict(metrics), indent=2))
        
    elif command == "alerts":
        metrics = monitor.generate_health_report()
        alerts = monitor.check_health_thresholds(metrics)
        
        if alerts:
            print("🚨 Health Alerts:")
            for alert in alerts:
                print(f"  - {alert}")
        else:
            print("✅ All health checks passed")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()