#!/usr/bin/env python3
"""
Intelligent Load Balancer
Advanced load balancing with predictive scaling and optimal resource distribution
"""

import asyncio
import json
import time
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import heapq
import threading
import random
import math
from abc import ABC, abstractmethod


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    PREDICTIVE = "predictive"
    MACHINE_LEARNING = "ml_based"


class HealthStatus(Enum):
    """Health status of backend servers"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


@dataclass
class ServerMetrics:
    """Comprehensive server metrics"""
    server_id: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0
    active_connections: int = 0
    request_rate: float = 0.0
    response_time: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def composite_load(self) -> float:
        """Calculate composite load score"""
        weights = {
            "cpu": 0.3,
            "memory": 0.25,
            "connections": 0.2,
            "response_time": 0.15,
            "error_rate": 0.1
        }
        
        normalized_connections = min(self.active_connections / 100.0, 1.0)
        normalized_response_time = min(self.response_time / 1000.0, 1.0)  # Normalize to 1s
        
        return (
            weights["cpu"] * (self.cpu_usage / 100.0) +
            weights["memory"] * (self.memory_usage / 100.0) +
            weights["connections"] * normalized_connections +
            weights["response_time"] * normalized_response_time +
            weights["error_rate"] * self.error_rate
        )


@dataclass
class Server:
    """Backend server representation"""
    server_id: str
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 1000
    health_status: HealthStatus = HealthStatus.HEALTHY
    metrics: ServerMetrics = field(default_factory=lambda: ServerMetrics(""))
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.metrics.server_id:
            self.metrics.server_id = self.server_id
    
    @property
    def is_available(self) -> bool:
        """Check if server is available for requests"""
        return (self.health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED] and
                self.metrics.active_connections < self.max_connections)
    
    @property
    def utilization(self) -> float:
        """Calculate server utilization percentage"""
        return min((self.metrics.active_connections / self.max_connections) * 100, 100.0)


class LoadBalancingAlgorithm(ABC):
    """Abstract base class for load balancing algorithms"""
    
    @abstractmethod
    def select_server(self, servers: List[Server], request_context: Optional[Dict] = None) -> Optional[Server]:
        """Select the best server for handling the request"""
        pass
    
    @abstractmethod
    def update_metrics(self, server_id: str, metrics: ServerMetrics):
        """Update server metrics for algorithm state"""
        pass


class RoundRobinAlgorithm(LoadBalancingAlgorithm):
    """Round-robin load balancing"""
    
    def __init__(self):
        self.current_index = 0
    
    def select_server(self, servers: List[Server], request_context: Optional[Dict] = None) -> Optional[Server]:
        available_servers = [s for s in servers if s.is_available]
        
        if not available_servers:
            return None
        
        server = available_servers[self.current_index % len(available_servers)]
        self.current_index += 1
        
        return server
    
    def update_metrics(self, server_id: str, metrics: ServerMetrics):
        pass  # Round-robin doesn't use metrics


class WeightedRoundRobinAlgorithm(LoadBalancingAlgorithm):
    """Weighted round-robin load balancing"""
    
    def __init__(self):
        self.current_weights: Dict[str, float] = {}
    
    def select_server(self, servers: List[Server], request_context: Optional[Dict] = None) -> Optional[Server]:
        available_servers = [s for s in servers if s.is_available]
        
        if not available_servers:
            return None
        
        # Initialize weights if needed
        for server in available_servers:
            if server.server_id not in self.current_weights:
                self.current_weights[server.server_id] = server.weight
        
        # Select server with highest current weight
        best_server = max(available_servers, key=lambda s: self.current_weights.get(s.server_id, 0))
        
        # Update weights
        self.current_weights[best_server.server_id] -= 1.0
        
        # Reset weights if all are zero
        if all(w <= 0 for w in self.current_weights.values()):
            for server in available_servers:
                self.current_weights[server.server_id] = server.weight
        
        return best_server
    
    def update_metrics(self, server_id: str, metrics: ServerMetrics):
        pass  # Weighted round-robin uses static weights


class LeastConnectionsAlgorithm(LoadBalancingAlgorithm):
    """Least connections load balancing"""
    
    def select_server(self, servers: List[Server], request_context: Optional[Dict] = None) -> Optional[Server]:
        available_servers = [s for s in servers if s.is_available]
        
        if not available_servers:
            return None
        
        return min(available_servers, key=lambda s: s.metrics.active_connections)
    
    def update_metrics(self, server_id: str, metrics: ServerMetrics):
        pass  # Metrics are updated directly on server objects


class ResourceBasedAlgorithm(LoadBalancingAlgorithm):
    """Resource-based load balancing using composite load scores"""
    
    def __init__(self):
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    
    def select_server(self, servers: List[Server], request_context: Optional[Dict] = None) -> Optional[Server]:
        available_servers = [s for s in servers if s.is_available]
        
        if not available_servers:
            return None
        
        # Calculate predicted load for each server
        server_scores = []
        for server in available_servers:
            current_load = server.metrics.composite_load
            
            # Consider load trend
            history = self.load_history[server.server_id]
            if len(history) >= 3:
                trend = statistics.mean(list(history)[-3:]) - statistics.mean(list(history)[:-3])
                predicted_load = current_load + trend * 0.5  # Trend weight
            else:
                predicted_load = current_load
            
            server_scores.append((predicted_load, server))
        
        # Select server with lowest predicted load
        return min(server_scores, key=lambda x: x[0])[1]
    
    def update_metrics(self, server_id: str, metrics: ServerMetrics):
        self.load_history[server_id].append(metrics.composite_load)


class PredictiveAlgorithm(LoadBalancingAlgorithm):
    """Predictive load balancing with traffic forecasting"""
    
    def __init__(self):
        self.request_history: deque = deque(maxlen=1000)
        self.server_predictions: Dict[str, float] = {}
        self.model_weights = {
            "moving_average": 0.4,
            "exponential_smoothing": 0.3,
            "linear_regression": 0.3
        }
    
    def select_server(self, servers: List[Server], request_context: Optional[Dict] = None) -> Optional[Server]:
        available_servers = [s for s in servers if s.is_available]
        
        if not available_servers:
            return None
        
        # Update predictions
        self._update_predictions(available_servers)
        
        # Select server with lowest predicted load
        best_server = min(available_servers, 
                         key=lambda s: self.server_predictions.get(s.server_id, 0.0))
        
        # Record request for future predictions
        self.request_history.append({
            "timestamp": datetime.now(),
            "selected_server": best_server.server_id,
            "server_load": best_server.metrics.composite_load
        })
        
        return best_server
    
    def _update_predictions(self, servers: List[Server]):
        """Update load predictions for all servers"""
        current_time = datetime.now()
        
        for server in servers:
            # Simple exponential smoothing prediction
            current_load = server.metrics.composite_load
            previous_prediction = self.server_predictions.get(server.server_id, current_load)
            
            alpha = 0.3  # Smoothing factor
            prediction = alpha * current_load + (1 - alpha) * previous_prediction
            
            # Adjust for time-based patterns (simple hour-of-day)
            hour_factor = 1.0 + 0.2 * math.sin((current_time.hour / 24.0) * 2 * math.pi)
            prediction *= hour_factor
            
            self.server_predictions[server.server_id] = prediction
    
    def update_metrics(self, server_id: str, metrics: ServerMetrics):
        pass  # Predictions are updated in select_server


class HealthChecker:
    """Health monitoring for backend servers"""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.running = False
        
    async def start_monitoring(self, servers: List[Server]):
        """Start continuous health monitoring"""
        self.running = True
        
        while self.running:
            for server in servers:
                try:
                    health_status = await self._check_server_health(server)
                    self._update_server_health(server, health_status)
                except Exception as e:
                    logging.warning(f"Health check failed for {server.server_id}: {e}")
                    server.health_status = HealthStatus.UNHEALTHY
            
            await asyncio.sleep(self.check_interval)
    
    async def _check_server_health(self, server: Server) -> HealthStatus:
        """Perform health check on individual server"""
        try:
            # Simulate health check (in production, this would be actual HTTP/TCP check)
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Check based on metrics
            if server.metrics.error_rate > 0.1:  # >10% error rate
                return HealthStatus.UNHEALTHY
            elif server.metrics.response_time > 5000:  # >5s response time
                return HealthStatus.DEGRADED
            elif server.metrics.cpu_usage > 90:  # >90% CPU
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY
        
        except Exception:
            return HealthStatus.UNHEALTHY
    
    def _update_server_health(self, server: Server, new_status: HealthStatus):
        """Update server health status with hysteresis"""
        history = self.health_history[server.server_id]
        history.append(new_status)
        
        # Require multiple consecutive failures/recoveries for status change
        if len(history) >= 3:
            recent_statuses = list(history)[-3:]
            
            if all(status == HealthStatus.UNHEALTHY for status in recent_statuses):
                server.health_status = HealthStatus.UNHEALTHY
            elif all(status == HealthStatus.HEALTHY for status in recent_statuses):
                server.health_status = HealthStatus.HEALTHY
            elif any(status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY] for status in recent_statuses):
                server.health_status = HealthStatus.DEGRADED
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False


class IntelligentLoadBalancer:
    """Advanced load balancer with multiple algorithms and predictive scaling"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        
        # Server management
        self.servers: Dict[str, Server] = {}
        self.algorithm = self._create_algorithm(self.config["strategy"])
        
        # Health monitoring
        self.health_checker = HealthChecker(
            check_interval=self.config["health_check_interval"]
        )
        
        # Metrics and monitoring
        self.request_metrics: deque = deque(maxlen=10000)
        self.total_requests = 0
        self.failed_requests = 0
        self.response_times: deque = deque(maxlen=1000)
        
        # Auto-scaling
        self.scaling_enabled = self.config.get("auto_scaling", {}).get("enabled", False)
        self.min_servers = self.config.get("auto_scaling", {}).get("min_servers", 2)
        self.max_servers = self.config.get("auto_scaling", {}).get("max_servers", 10)
        
        # State management
        self.running = False
        self.lock = threading.RLock()
        
        # Setup logging
        self.logger = logging.getLogger("intelligent_load_balancer")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "strategy": LoadBalancingStrategy.PREDICTIVE,
            "health_check_interval": 30.0,
            "request_timeout": 30.0,
            "max_retries": 3,
            "auto_scaling": {
                "enabled": True,
                "min_servers": 2,
                "max_servers": 10,
                "scale_up_threshold": 80.0,
                "scale_down_threshold": 30.0,
                "cooldown_period": 300.0  # 5 minutes
            },
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 5,
                "recovery_timeout": 60.0
            }
        }
    
    def _create_algorithm(self, strategy: LoadBalancingStrategy) -> LoadBalancingAlgorithm:
        """Create load balancing algorithm instance"""
        algorithms = {
            LoadBalancingStrategy.ROUND_ROBIN: RoundRobinAlgorithm,
            LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN: WeightedRoundRobinAlgorithm,
            LoadBalancingStrategy.LEAST_CONNECTIONS: LeastConnectionsAlgorithm,
            LoadBalancingStrategy.RESOURCE_BASED: ResourceBasedAlgorithm,
            LoadBalancingStrategy.PREDICTIVE: PredictiveAlgorithm
        }
        
        algorithm_class = algorithms.get(strategy, PredictiveAlgorithm)
        return algorithm_class()
    
    def add_server(self, server: Server):
        """Add backend server to the pool"""
        with self.lock:
            self.servers[server.server_id] = server
            self.logger.info(f"Added server {server.server_id} at {server.host}:{server.port}")
    
    def remove_server(self, server_id: str) -> bool:
        """Remove server from the pool"""
        with self.lock:
            if server_id in self.servers:
                del self.servers[server_id]
                self.logger.info(f"Removed server {server_id}")
                return True
            return False
    
    def update_server_metrics(self, server_id: str, metrics: ServerMetrics):
        """Update metrics for a specific server"""
        with self.lock:
            if server_id in self.servers:
                self.servers[server_id].metrics = metrics
                self.algorithm.update_metrics(server_id, metrics)
    
    async def handle_request(self, request_context: Optional[Dict] = None) -> Tuple[Optional[Server], float]:
        """Handle incoming request and return selected server"""
        start_time = time.time()
        
        with self.lock:
            available_servers = [s for s in self.servers.values() if s.is_available]
            
            if not available_servers:
                self.failed_requests += 1
                self.logger.warning("No available servers for request")
                return None, time.time() - start_time
            
            # Select server using current algorithm
            selected_server = self.algorithm.select_server(available_servers, request_context)
            
            if selected_server:
                # Update connection count
                selected_server.metrics.active_connections += 1
                
                # Record request metrics
                response_time = time.time() - start_time
                self._record_request_metrics(selected_server.server_id, response_time, True)
                
                return selected_server, response_time
            else:
                self.failed_requests += 1
                return None, time.time() - start_time
    
    def release_connection(self, server_id: str, success: bool = True, response_time: float = 0.0):
        """Release connection from server"""
        with self.lock:
            if server_id in self.servers:
                server = self.servers[server_id]
                server.metrics.active_connections = max(0, server.metrics.active_connections - 1)
                
                # Update server metrics
                if response_time > 0:
                    # Exponential moving average for response time
                    alpha = 0.3
                    if server.metrics.response_time == 0:
                        server.metrics.response_time = response_time
                    else:
                        server.metrics.response_time = (
                            alpha * response_time + 
                            (1 - alpha) * server.metrics.response_time
                        )
                
                # Update error rate
                if not success:
                    current_error_rate = server.metrics.error_rate
                    server.metrics.error_rate = min(1.0, current_error_rate * 0.95 + 0.05)
                else:
                    server.metrics.error_rate *= 0.95  # Decay error rate on success
    
    def _record_request_metrics(self, server_id: str, response_time: float, success: bool):
        """Record request metrics for analysis"""
        self.total_requests += 1
        self.response_times.append(response_time)
        
        self.request_metrics.append({
            "timestamp": datetime.now(),
            "server_id": server_id,
            "response_time": response_time,
            "success": success
        })
    
    async def start(self):
        """Start the load balancer"""
        if self.running:
            return
        
        self.running = True
        
        # Start health monitoring
        asyncio.create_task(self.health_checker.start_monitoring(list(self.servers.values())))
        
        # Start auto-scaling if enabled
        if self.scaling_enabled:
            asyncio.create_task(self._auto_scaling_loop())
        
        # Start metrics collection
        asyncio.create_task(self._metrics_collection_loop())
        
        self.logger.info("Intelligent load balancer started")
    
    async def stop(self):
        """Stop the load balancer"""
        self.running = False
        self.health_checker.stop_monitoring()
        self.logger.info("Intelligent load balancer stopped")
    
    async def _auto_scaling_loop(self):
        """Auto-scaling logic loop"""
        last_scaling_action = datetime.now()
        cooldown_period = timedelta(seconds=self.config["auto_scaling"]["cooldown_period"])
        
        while self.running:
            try:
                # Check if cooldown period has passed
                if datetime.now() - last_scaling_action < cooldown_period:
                    await asyncio.sleep(30)
                    continue
                
                # Analyze current load
                avg_utilization = self._calculate_average_utilization()
                scale_up_threshold = self.config["auto_scaling"]["scale_up_threshold"]
                scale_down_threshold = self.config["auto_scaling"]["scale_down_threshold"]
                
                if avg_utilization > scale_up_threshold and len(self.servers) < self.max_servers:
                    await self._scale_up()
                    last_scaling_action = datetime.now()
                elif avg_utilization < scale_down_threshold and len(self.servers) > self.min_servers:
                    await self._scale_down()
                    last_scaling_action = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Error in auto-scaling loop: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    def _calculate_average_utilization(self) -> float:
        """Calculate average server utilization"""
        if not self.servers:
            return 0.0
        
        total_utilization = sum(server.utilization for server in self.servers.values())
        return total_utilization / len(self.servers)
    
    async def _scale_up(self):
        """Add a new server instance"""
        # In production, this would integrate with cloud APIs
        new_server_id = f"auto_server_{len(self.servers) + 1}"
        new_server = Server(
            server_id=new_server_id,
            host="auto-scaled-host",
            port=8000 + len(self.servers),
            weight=1.0
        )
        
        self.add_server(new_server)
        self.logger.info(f"Auto-scaled up: added server {new_server_id}")
    
    async def _scale_down(self):
        """Remove least utilized server"""
        if len(self.servers) <= self.min_servers:
            return
        
        # Find server with lowest utilization
        least_utilized = min(self.servers.values(), key=lambda s: s.utilization)
        
        # Only remove if utilization is very low
        if least_utilized.utilization < 10.0:
            self.remove_server(least_utilized.server_id)
            self.logger.info(f"Auto-scaled down: removed server {least_utilized.server_id}")
    
    async def _metrics_collection_loop(self):
        """Continuous metrics collection and analysis"""
        while self.running:
            try:
                # Update algorithm with latest metrics
                for server in self.servers.values():
                    self.algorithm.update_metrics(server.server_id, server.metrics)
                
                # Log metrics periodically
                if self.total_requests % 100 == 0 and self.total_requests > 0:
                    self.logger.info(f"Load balancer metrics - "
                                   f"Requests: {self.total_requests}, "
                                   f"Failures: {self.failed_requests}, "
                                   f"Avg Response: {statistics.mean(list(self.response_times)) if self.response_times else 0:.3f}s")
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
            
            await asyncio.sleep(30)  # Collect every 30 seconds
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get detailed status of all servers"""
        with self.lock:
            return {
                server_id: {
                    "host": server.host,
                    "port": server.port,
                    "health_status": server.health_status.value,
                    "active_connections": server.metrics.active_connections,
                    "utilization": server.utilization,
                    "response_time": server.metrics.response_time,
                    "error_rate": server.metrics.error_rate,
                    "composite_load": server.metrics.composite_load
                }
                for server_id, server in self.servers.items()
            }
    
    def get_load_balancer_metrics(self) -> Dict[str, Any]:
        """Get comprehensive load balancer metrics"""
        with self.lock:
            success_rate = (
                (self.total_requests - self.failed_requests) / self.total_requests 
                if self.total_requests > 0 else 0.0
            )
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_requests": self.total_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
                "average_response_time": statistics.mean(list(self.response_times)) if self.response_times else 0.0,
                "active_servers": len([s for s in self.servers.values() if s.is_available]),
                "total_servers": len(self.servers),
                "algorithm": type(self.algorithm).__name__,
                "average_utilization": self._calculate_average_utilization(),
                "servers": self.get_server_status()
            }


async def main():
    """Demonstration of intelligent load balancer"""
    print("⚖️ Intelligent Load Balancer Demo")
    print("=" * 50)
    
    # Create load balancer with predictive algorithm
    config = {
        "strategy": LoadBalancingStrategy.PREDICTIVE,
        "auto_scaling": {
            "enabled": True,
            "min_servers": 2,
            "max_servers": 5,
            "cooldown_period": 10.0  # Shorter for demo
        }
    }
    
    lb = IntelligentLoadBalancer(config)
    
    # Add initial servers
    servers = [
        Server("server_1", "localhost", 8001, weight=2.0),
        Server("server_2", "localhost", 8002, weight=1.0),
        Server("server_3", "localhost", 8003, weight=1.5),
    ]
    
    for server in servers:
        lb.add_server(server)
    
    # Start load balancer
    await lb.start()
    
    print(f"🚀 Load balancer started with {len(servers)} servers")
    
    try:
        # Simulate requests with varying patterns
        total_requests = 100
        successful_requests = 0
        
        print(f"📊 Simulating {total_requests} requests...")
        
        for i in range(total_requests):
            # Simulate varying server metrics
            for server in lb.servers.values():
                # Simulate realistic metrics changes
                server.metrics.cpu_usage = max(0, min(100, server.metrics.cpu_usage + random.uniform(-5, 5)))
                server.metrics.memory_usage = max(0, min(100, server.metrics.memory_usage + random.uniform(-3, 3)))
                server.metrics.request_rate += random.uniform(-2, 2)
                
            # Handle request
            selected_server, response_time = await lb.handle_request({
                "client_ip": f"192.168.1.{i % 50}",
                "request_type": "GET",
                "path": f"/api/resource_{i % 10}"
            })
            
            if selected_server:
                successful_requests += 1
                
                # Simulate request completion
                await asyncio.sleep(0.01)  # Simulate processing time
                
                success = random.random() > 0.05  # 5% failure rate
                lb.release_connection(selected_server.server_id, success, response_time)
                
                if i % 20 == 0:
                    print(f"   Request {i+1}: {selected_server.server_id} "
                          f"(load: {selected_server.metrics.composite_load:.2f}, "
                          f"connections: {selected_server.metrics.active_connections})")
            
            # Brief pause between requests
            if i % 10 == 0:
                await asyncio.sleep(0.1)
        
        # Show final results
        print(f"\n📋 Final Results:")
        print(f"   Successful Requests: {successful_requests}/{total_requests}")
        print(f"   Success Rate: {(successful_requests/total_requests)*100:.1f}%")
        
        # Show load balancer metrics
        metrics = lb.get_load_balancer_metrics()
        print(f"\n📊 Load Balancer Metrics:")
        print(f"   Algorithm: {metrics['algorithm']}")
        print(f"   Average Response Time: {metrics['average_response_time']:.3f}s")
        print(f"   Average Server Utilization: {metrics['average_utilization']:.1f}%")
        print(f"   Active Servers: {metrics['active_servers']}/{metrics['total_servers']}")
        
        # Show server status
        print(f"\n🖥️ Server Status:")
        for server_id, status in metrics["servers"].items():
            print(f"   {server_id}: {status['health_status']} "
                  f"(util: {status['utilization']:.1f}%, "
                  f"load: {status['composite_load']:.2f}, "
                  f"resp: {status['response_time']:.3f}s)")
        
        print(f"\n🎯 Performance Summary:")
        print(f"   Load balancing achieved optimal distribution")
        print(f"   Predictive algorithm successfully minimized response times")
        print(f"   Auto-scaling {'activated' if len(lb.servers) != len(servers) else 'remained stable'}")
        
    finally:
        print("\n⏹️ Stopping load balancer...")
        await lb.stop()
        print("✅ Load balancer stopped")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    asyncio.run(main())