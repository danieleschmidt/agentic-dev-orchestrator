#!/usr/bin/env python3
"""
Self-Healing Infrastructure Manager v5.0
Autonomous infrastructure management with predictive healing,
adaptive resource optimization, and intelligent failure recovery
"""

import asyncio
import json
import logging
import os
import psutil
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import uuid
import threading
from collections import defaultdict, deque
import statistics
import subprocess
import signal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels"""
    OPTIMAL = "optimal"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILING = "failing"
    RECOVERY = "recovery"


class HealingStrategy(Enum):
    """Self-healing strategy types"""
    PREVENTIVE = "preventive"
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"
    EVOLUTIONARY = "evolutionary"


class ResourceType(Enum):
    """Infrastructure resource types"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESS = "process"
    SERVICE = "service"
    DATABASE = "database"
    CACHE = "cache"


@dataclass
class HealthMetric:
    """Individual health metric measurement"""
    metric_id: str
    resource_type: ResourceType
    current_value: float
    threshold_warning: float
    threshold_critical: float
    threshold_optimal: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_velocity: float
    prediction_accuracy: float
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def health_status(self) -> HealthStatus:
        """Calculate health status based on current value and thresholds"""
        if self.current_value <= self.threshold_optimal:
            return HealthStatus.OPTIMAL
        elif self.current_value <= self.threshold_warning:
            return HealthStatus.HEALTHY
        elif self.current_value <= self.threshold_critical:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.CRITICAL
    
    @property
    def time_to_critical(self) -> Optional[float]:
        """Estimate time until critical threshold (in minutes)"""
        if self.trend_direction != "increasing" or self.trend_velocity <= 0:
            return None
        
        remaining_capacity = self.threshold_critical - self.current_value
        if remaining_capacity <= 0:
            return 0.0
        
        return remaining_capacity / (self.trend_velocity / 60.0)  # Convert to minutes


@dataclass
class HealingAction:
    """Self-healing action definition"""
    action_id: str
    action_type: str
    target_resource: ResourceType
    healing_strategy: HealingStrategy
    command: str
    parameters: Dict[str, Any]
    expected_impact: Dict[str, float]
    execution_timeout: int
    rollback_command: Optional[str]
    success_criteria: List[Dict[str, Any]]
    risk_level: str  # "low", "medium", "high"
    requires_approval: bool
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class HealingEvent:
    """Record of healing action execution"""
    event_id: str
    action: HealingAction
    trigger_metrics: List[HealthMetric]
    execution_start: datetime
    execution_end: Optional[datetime]
    success: bool
    impact_metrics: Dict[str, float]
    error_message: Optional[str]
    rollback_performed: bool
    learning_data: Dict[str, Any]


class PredictiveHealthMonitor:
    """Predictive health monitoring with trend analysis"""
    
    def __init__(self, monitoring_interval: int = 30):
        self.monitoring_interval = monitoring_interval
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.prediction_models: Dict[str, Any] = {}
        self.anomaly_detectors: Dict[str, Any] = {}
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        
    async def collect_system_metrics(self) -> List[HealthMetric]:
        """Collect comprehensive system health metrics"""
        metrics = []
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_metric = HealthMetric(
            metric_id="cpu_utilization",
            resource_type=ResourceType.CPU,
            current_value=cpu_percent,
            threshold_warning=70.0,
            threshold_critical=85.0,
            threshold_optimal=50.0,
            trend_direction=await self._calculate_trend("cpu_utilization", cpu_percent),
            trend_velocity=await self._calculate_trend_velocity("cpu_utilization", cpu_percent),
            prediction_accuracy=0.85
        )
        metrics.append(cpu_metric)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_metric = HealthMetric(
            metric_id="memory_utilization",
            resource_type=ResourceType.MEMORY,
            current_value=memory_percent,
            threshold_warning=75.0,
            threshold_critical=90.0,
            threshold_optimal=60.0,
            trend_direction=await self._calculate_trend("memory_utilization", memory_percent),
            trend_velocity=await self._calculate_trend_velocity("memory_utilization", memory_percent),
            prediction_accuracy=0.82
        )
        metrics.append(memory_metric)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_metric = HealthMetric(
            metric_id="disk_utilization",
            resource_type=ResourceType.DISK,
            current_value=disk_percent,
            threshold_warning=80.0,
            threshold_critical=95.0,
            threshold_optimal=70.0,
            trend_direction=await self._calculate_trend("disk_utilization", disk_percent),
            trend_velocity=await self._calculate_trend_velocity("disk_utilization", disk_percent),
            prediction_accuracy=0.90
        )
        metrics.append(disk_metric)
        
        # Network metrics (basic)
        network_stats = psutil.net_io_counters()
        network_errors = network_stats.errin + network_stats.errout
        network_metric = HealthMetric(
            metric_id="network_errors",
            resource_type=ResourceType.NETWORK,
            current_value=float(network_errors),
            threshold_warning=10.0,
            threshold_critical=50.0,
            threshold_optimal=0.0,
            trend_direction=await self._calculate_trend("network_errors", float(network_errors)),
            trend_velocity=await self._calculate_trend_velocity("network_errors", float(network_errors)),
            prediction_accuracy=0.75
        )
        metrics.append(network_metric)
        
        # Process count metric
        process_count = len(psutil.pids())
        process_metric = HealthMetric(
            metric_id="process_count",
            resource_type=ResourceType.PROCESS,
            current_value=float(process_count),
            threshold_warning=500.0,
            threshold_critical=800.0,
            threshold_optimal=200.0,
            trend_direction=await self._calculate_trend("process_count", float(process_count)),
            trend_velocity=await self._calculate_trend_velocity("process_count", float(process_count)),
            prediction_accuracy=0.70
        )
        metrics.append(process_metric)
        
        # Store metrics in history
        for metric in metrics:
            self.metric_history[metric.metric_id].append({
                'timestamp': metric.last_updated.isoformat(),
                'value': metric.current_value
            })
        
        return metrics
    
    async def _calculate_trend(self, metric_id: str, current_value: float) -> str:
        """Calculate trend direction for metric"""
        history = self.metric_history[metric_id]
        
        if len(history) < 3:
            return "stable"
        
        recent_values = [h['value'] for h in list(history)[-3:]]
        
        # Simple trend calculation
        if recent_values[-1] > recent_values[-2] > recent_values[-3]:
            return "increasing"
        elif recent_values[-1] < recent_values[-2] < recent_values[-3]:
            return "decreasing"
        else:
            return "stable"
    
    async def _calculate_trend_velocity(self, metric_id: str, current_value: float) -> float:
        """Calculate trend velocity (rate of change per minute)"""
        history = self.metric_history[metric_id]
        
        if len(history) < 2:
            return 0.0
        
        # Calculate velocity over last few measurements
        recent_history = list(history)[-5:]  # Last 5 measurements
        
        if len(recent_history) < 2:
            return 0.0
        
        time_deltas = []
        value_deltas = []
        
        for i in range(1, len(recent_history)):
            prev_time = datetime.fromisoformat(recent_history[i-1]['timestamp'])
            curr_time = datetime.fromisoformat(recent_history[i]['timestamp'])
            time_delta = (curr_time - prev_time).total_seconds() / 60.0  # Convert to minutes
            
            value_delta = recent_history[i]['value'] - recent_history[i-1]['value']
            
            if time_delta > 0:
                time_deltas.append(time_delta)
                value_deltas.append(value_delta)
        
        if not time_deltas:
            return 0.0
        
        # Average velocity
        avg_velocity = sum(v / t for v, t in zip(value_deltas, time_deltas)) / len(time_deltas)
        return avg_velocity
    
    async def predict_future_metrics(self, metrics: List[HealthMetric], 
                                   prediction_horizon_minutes: int = 30) -> List[HealthMetric]:
        """Predict future metric values using trend analysis"""
        predicted_metrics = []
        
        for metric in metrics:
            if metric.trend_direction == "stable":
                predicted_value = metric.current_value
            else:
                # Linear prediction based on trend velocity
                velocity_per_minute = metric.trend_velocity
                predicted_change = velocity_per_minute * prediction_horizon_minutes
                predicted_value = max(0, metric.current_value + predicted_change)
            
            predicted_metric = HealthMetric(
                metric_id=f"predicted_{metric.metric_id}",
                resource_type=metric.resource_type,
                current_value=predicted_value,
                threshold_warning=metric.threshold_warning,
                threshold_critical=metric.threshold_critical,
                threshold_optimal=metric.threshold_optimal,
                trend_direction=metric.trend_direction,
                trend_velocity=metric.trend_velocity,
                prediction_accuracy=metric.prediction_accuracy * 0.8,  # Reduced accuracy for predictions
                last_updated=datetime.now() + timedelta(minutes=prediction_horizon_minutes)
            )
            predicted_metrics.append(predicted_metric)
        
        return predicted_metrics


class AdaptiveHealingEngine:
    """Adaptive healing engine with learning capabilities"""
    
    def __init__(self):
        self.healing_strategies: Dict[str, List[HealingAction]] = {}
        self.healing_history: List[HealingEvent] = []
        self.success_rates: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.learning_models: Dict[str, Any] = {}
        self.adaptive_thresholds: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Initialize default healing actions
        self._initialize_default_healing_actions()
    
    def _initialize_default_healing_actions(self):
        """Initialize default set of healing actions"""
        
        # CPU healing actions
        cpu_actions = [
            HealingAction(
                action_id="cpu_kill_high_usage_processes",
                action_type="process_management",
                target_resource=ResourceType.CPU,
                healing_strategy=HealingStrategy.REACTIVE,
                command="pkill -f 'high_cpu_process_pattern'",
                parameters={"cpu_threshold": 90, "grace_period": 30},
                expected_impact={"cpu_reduction": 20.0},
                execution_timeout=60,
                rollback_command=None,
                success_criteria=[{"metric": "cpu_utilization", "target": "<75"}],
                risk_level="medium",
                requires_approval=False
            ),
            HealingAction(
                action_id="cpu_nice_adjustment",
                action_type="priority_adjustment",
                target_resource=ResourceType.CPU,
                healing_strategy=HealingStrategy.PREVENTIVE,
                command="renice -n 10 -p {pid}",
                parameters={"nice_value": 10},
                expected_impact={"cpu_reduction": 10.0},
                execution_timeout=30,
                rollback_command="renice -n 0 -p {pid}",
                success_criteria=[{"metric": "cpu_utilization", "target": "<80"}],
                risk_level="low",
                requires_approval=False
            )
        ]
        self.healing_strategies[ResourceType.CPU.value] = cpu_actions
        
        # Memory healing actions
        memory_actions = [
            HealingAction(
                action_id="memory_clear_cache",
                action_type="cache_management",
                target_resource=ResourceType.MEMORY,
                healing_strategy=HealingStrategy.REACTIVE,
                command="sync && echo 3 > /proc/sys/vm/drop_caches",
                parameters={},
                expected_impact={"memory_reduction": 15.0},
                execution_timeout=60,
                rollback_command=None,
                success_criteria=[{"metric": "memory_utilization", "target": "<80"}],
                risk_level="low",
                requires_approval=False
            ),
            HealingAction(
                action_id="memory_restart_high_memory_service",
                action_type="service_restart",
                target_resource=ResourceType.MEMORY,
                healing_strategy=HealingStrategy.REACTIVE,
                command="systemctl restart {service_name}",
                parameters={"memory_threshold": 85},
                expected_impact={"memory_reduction": 30.0},
                execution_timeout=120,
                rollback_command=None,
                success_criteria=[{"metric": "memory_utilization", "target": "<75"}],
                risk_level="medium",
                requires_approval=True
            )
        ]
        self.healing_strategies[ResourceType.MEMORY.value] = memory_actions
        
        # Disk healing actions
        disk_actions = [
            HealingAction(
                action_id="disk_cleanup_temp_files",
                action_type="cleanup",
                target_resource=ResourceType.DISK,
                healing_strategy=HealingStrategy.PREVENTIVE,
                command="find /tmp -type f -atime +7 -delete",
                parameters={},
                expected_impact={"disk_reduction": 5.0},
                execution_timeout=300,
                rollback_command=None,
                success_criteria=[{"metric": "disk_utilization", "target": "<85"}],
                risk_level="low",
                requires_approval=False
            ),
            HealingAction(
                action_id="disk_log_rotation",
                action_type="log_management",
                target_resource=ResourceType.DISK,
                healing_strategy=HealingStrategy.PREVENTIVE,
                command="logrotate -f /etc/logrotate.conf",
                parameters={},
                expected_impact={"disk_reduction": 10.0},
                execution_timeout=180,
                rollback_command=None,
                success_criteria=[{"metric": "disk_utilization", "target": "<80"}],
                risk_level="low",
                requires_approval=False
            )
        ]
        self.healing_strategies[ResourceType.DISK.value] = disk_actions
    
    async def select_optimal_healing_actions(self, 
                                           problematic_metrics: List[HealthMetric]) -> List[HealingAction]:
        """Select optimal healing actions based on current metrics and historical success"""
        
        selected_actions = []
        
        for metric in problematic_metrics:
            if metric.health_status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]:
                # Get available actions for resource type
                available_actions = self.healing_strategies.get(metric.resource_type.value, [])
                
                # Score actions based on effectiveness and risk
                scored_actions = []
                for action in available_actions:
                    score = await self._score_healing_action(action, metric)
                    scored_actions.append((action, score))
                
                # Sort by score and select best action
                scored_actions.sort(key=lambda x: x[1], reverse=True)
                
                if scored_actions and scored_actions[0][1] > 0.5:
                    selected_actions.append(scored_actions[0][0])
        
        return selected_actions
    
    async def _score_healing_action(self, action: HealingAction, metric: HealthMetric) -> float:
        """Score healing action based on effectiveness, risk, and historical success"""
        score = 0.0
        
        # Base effectiveness score
        expected_impact = action.expected_impact.get(f"{metric.resource_type.value}_reduction", 0.0)
        effectiveness = min(expected_impact / 50.0, 1.0)  # Normalize to 0-1
        score += effectiveness * 0.4
        
        # Historical success rate
        action_key = f"{action.action_type}_{action.target_resource.value}"
        success_rate = self.success_rates[action_key].get("success_rate", 0.5)
        score += success_rate * 0.3
        
        # Risk factor (lower risk = higher score)
        risk_scores = {"low": 1.0, "medium": 0.7, "high": 0.4}
        risk_score = risk_scores.get(action.risk_level, 0.5)
        score += risk_score * 0.2
        
        # Urgency factor based on metric severity
        if metric.health_status == HealthStatus.CRITICAL:
            score += 0.1
        elif metric.health_status == HealthStatus.DEGRADED:
            score += 0.05
        
        return score
    
    async def execute_healing_action(self, action: HealingAction, 
                                   trigger_metrics: List[HealthMetric]) -> HealingEvent:
        """Execute healing action and monitor results"""
        
        event = HealingEvent(
            event_id=str(uuid.uuid4()),
            action=action,
            trigger_metrics=trigger_metrics,
            execution_start=datetime.now(),
            execution_end=None,
            success=False,
            impact_metrics={},
            error_message=None,
            rollback_performed=False,
            learning_data={}
        )
        
        logger.info(f"Executing healing action: {action.action_id}")
        
        try:
            # Execute command with timeout
            if action.command:
                process = await asyncio.create_subprocess_shell(
                    action.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), 
                        timeout=action.execution_timeout
                    )
                    
                    if process.returncode == 0:
                        event.success = True
                        logger.info(f"Healing action {action.action_id} completed successfully")
                    else:
                        event.error_message = stderr.decode() if stderr else "Command failed"
                        logger.error(f"Healing action {action.action_id} failed: {event.error_message}")
                
                except asyncio.TimeoutError:
                    process.kill()
                    event.error_message = "Command timed out"
                    logger.error(f"Healing action {action.action_id} timed out")
        
        except Exception as e:
            event.error_message = str(e)
            logger.error(f"Error executing healing action {action.action_id}: {e}")
        
        event.execution_end = datetime.now()
        
        # Update success rates for learning
        await self._update_success_rates(action, event.success)
        
        # Store healing event
        self.healing_history.append(event)
        
        return event
    
    async def _update_success_rates(self, action: HealingAction, success: bool):
        """Update success rates for learning"""
        action_key = f"{action.action_type}_{action.target_resource.value}"
        
        # Get current stats
        current_success_rate = self.success_rates[action_key].get("success_rate", 0.5)
        current_count = self.success_rates[action_key].get("count", 0)
        
        # Update with new result
        new_count = current_count + 1
        if success:
            new_success_rate = (current_success_rate * current_count + 1.0) / new_count
        else:
            new_success_rate = (current_success_rate * current_count) / new_count
        
        self.success_rates[action_key] = {
            "success_rate": new_success_rate,
            "count": new_count
        }


class SelfHealingInfrastructureManager:
    """Main self-healing infrastructure management system"""
    
    def __init__(self, monitoring_interval: int = 30):
        self.health_monitor = PredictiveHealthMonitor(monitoring_interval)
        self.healing_engine = AdaptiveHealingEngine()
        self.monitoring_interval = monitoring_interval
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.system_state = {
            "last_health_check": None,
            "current_health_status": HealthStatus.HEALTHY,
            "active_healing_actions": [],
            "system_metrics": {},
            "uptime_start": datetime.now()
        }
        
    async def start_monitoring(self):
        """Start continuous health monitoring and self-healing"""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        logger.info("Starting self-healing infrastructure monitoring")
        
        # Start monitoring loop
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        try:
            await self.monitoring_task
        except asyncio.CancelledError:
            logger.info("Monitoring task cancelled")
        finally:
            self.is_running = False
    
    async def stop_monitoring(self):
        """Stop monitoring and healing"""
        if not self.is_running:
            return
        
        logger.info("Stopping self-healing infrastructure monitoring")
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self):
        """Main monitoring and healing loop"""
        while self.is_running:
            try:
                # Collect current metrics
                current_metrics = await self.health_monitor.collect_system_metrics()
                
                # Update system state
                self.system_state["last_health_check"] = datetime.now()
                self.system_state["system_metrics"] = {
                    m.metric_id: m.current_value for m in current_metrics
                }
                
                # Determine overall system health
                overall_health = await self._calculate_overall_health(current_metrics)
                self.system_state["current_health_status"] = overall_health
                
                # Identify problematic metrics
                problematic_metrics = [
                    m for m in current_metrics 
                    if m.health_status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]
                ]
                
                if problematic_metrics:
                    logger.warning(f"Found {len(problematic_metrics)} problematic metrics")
                    
                    # Predictive analysis
                    predicted_metrics = await self.health_monitor.predict_future_metrics(current_metrics)
                    future_problems = [
                        m for m in predicted_metrics
                        if m.health_status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]
                    ]
                    
                    # Select and execute healing actions
                    healing_actions = await self.healing_engine.select_optimal_healing_actions(
                        problematic_metrics + future_problems
                    )
                    
                    for action in healing_actions:
                        if not action.requires_approval:  # Auto-execute safe actions
                            healing_event = await self.healing_engine.execute_healing_action(
                                action, problematic_metrics
                            )
                            logger.info(f"Executed healing action: {action.action_id}, Success: {healing_event.success}")
                        else:
                            logger.info(f"Healing action {action.action_id} requires approval")
                
                else:
                    logger.debug("All metrics healthy")
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    async def _calculate_overall_health(self, metrics: List[HealthMetric]) -> HealthStatus:
        """Calculate overall system health from individual metrics"""
        if not metrics:
            return HealthStatus.HEALTHY
        
        health_scores = {
            HealthStatus.OPTIMAL: 5,
            HealthStatus.HEALTHY: 4,
            HealthStatus.DEGRADED: 2,
            HealthStatus.CRITICAL: 1,
            HealthStatus.FAILING: 0
        }
        
        total_score = sum(health_scores[m.health_status] for m in metrics)
        avg_score = total_score / len(metrics)
        
        if avg_score >= 4.5:
            return HealthStatus.OPTIMAL
        elif avg_score >= 3.5:
            return HealthStatus.HEALTHY
        elif avg_score >= 2.0:
            return HealthStatus.DEGRADED
        elif avg_score >= 1.0:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.FAILING
    
    async def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report"""
        current_metrics = await self.health_monitor.collect_system_metrics()
        predicted_metrics = await self.health_monitor.predict_future_metrics(current_metrics)
        
        # Calculate system uptime
        uptime = datetime.now() - self.system_state["uptime_start"]
        
        # Recent healing events
        recent_events = [
            asdict(event) for event in self.healing_engine.healing_history[-10:]
        ]
        
        # Success rates summary
        success_summary = {}
        for action_key, stats in self.healing_engine.success_rates.items():
            success_summary[action_key] = {
                "success_rate": stats["success_rate"],
                "execution_count": stats["count"]
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": {
                "overall_status": self.system_state["current_health_status"].value,
                "uptime_hours": uptime.total_seconds() / 3600,
                "last_check": self.system_state["last_health_check"].isoformat() if self.system_state["last_health_check"] else None
            },
            "current_metrics": [asdict(m) for m in current_metrics],
            "predicted_metrics": [asdict(m) for m in predicted_metrics],
            "problematic_metrics": [
                asdict(m) for m in current_metrics 
                if m.health_status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]
            ],
            "healing_statistics": {
                "total_healing_events": len(self.healing_engine.healing_history),
                "recent_events": recent_events,
                "success_rates": success_summary
            },
            "predictions": {
                "metrics_approaching_thresholds": [
                    {
                        "metric": m.metric_id,
                        "current_value": m.current_value,
                        "threshold": m.threshold_critical,
                        "time_to_critical": m.time_to_critical
                    }
                    for m in current_metrics 
                    if m.time_to_critical and m.time_to_critical < 60  # Within 1 hour
                ]
            }
        }


async def main():
    """Main execution for self-healing infrastructure manager"""
    
    # Create and start infrastructure manager
    manager = SelfHealingInfrastructureManager(monitoring_interval=15)
    
    print(f"\n{'='*70}")
    print("SELF-HEALING INFRASTRUCTURE MANAGER")
    print(f"{'='*70}")
    
    # Generate initial health report
    initial_report = await manager.get_system_health_report()
    print(f"\nInitial System Health: {initial_report['system_health']['overall_status'].upper()}")
    
    print(f"\nCurrent Metrics:")
    for metric in initial_report['current_metrics']:
        status = metric['health_status'].upper()
        value = metric['current_value']
        threshold = metric['threshold_critical']
        print(f"  {metric['metric_id']}: {value:.1f} ({status}) [Critical: {threshold}]")
    
    # Show predictions
    if initial_report['predictions']['metrics_approaching_thresholds']:
        print(f"\n⚠️  Metrics Approaching Critical Thresholds:")
        for pred in initial_report['predictions']['metrics_approaching_thresholds']:
            print(f"  {pred['metric']}: {pred['time_to_critical']:.1f} minutes until critical")
    
    # Demonstrate healing capabilities (run for limited time)
    print(f"\nStarting self-healing monitoring (will run for 60 seconds)...")
    
    # Start monitoring in background
    monitoring_task = asyncio.create_task(manager.start_monitoring())
    
    # Let it run for demonstration
    await asyncio.sleep(60)
    
    # Stop monitoring
    await manager.stop_monitoring()
    
    # Final health report
    final_report = await manager.get_system_health_report()
    print(f"\n{'='*70}")
    print("FINAL HEALTH REPORT")
    print(f"{'='*70}")
    
    print(f"System Health: {final_report['system_health']['overall_status'].upper()}")
    print(f"Total Healing Events: {final_report['healing_statistics']['total_healing_events']}")
    
    if final_report['healing_statistics']['recent_events']:
        print(f"\nRecent Healing Actions:")
        for event in final_report['healing_statistics']['recent_events'][-3:]:
            print(f"  {event['action']['action_id']}: {'✅' if event['success'] else '❌'}")
    
    print(f"\nSystem Uptime: {final_report['system_health']['uptime_hours']:.2f} hours")


if __name__ == "__main__":
    asyncio.run(main())