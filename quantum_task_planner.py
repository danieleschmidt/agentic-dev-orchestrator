#!/usr/bin/env python3
"""
Quantum-Inspired Task Planner
Implements quantum-inspired algorithms for optimal task scheduling and resource allocation
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import datetime
import json
from pathlib import Path

from backlog_manager import BacklogItem, BacklogManager


class QuantumState(Enum):
    """Quantum-inspired states for tasks"""
    SUPERPOSITION = "superposition"  # Task can be in multiple states
    ENTANGLED = "entangled"         # Task depends on other tasks
    COLLAPSED = "collapsed"         # Task state is determined
    COHERENT = "coherent"           # Task maintains optimal state


@dataclass
class QuantumTask:
    """Quantum-enhanced task representation"""
    id: str
    base_item: BacklogItem
    quantum_state: QuantumState = QuantumState.SUPERPOSITION
    coherence_level: float = 1.0
    entanglement_partners: List[str] = field(default_factory=list)
    probability_amplitudes: Dict[str, float] = field(default_factory=dict)
    interference_pattern: List[float] = field(default_factory=list)
    quantum_priority: float = 0.0
    optimal_execution_time: Optional[datetime.datetime] = None


@dataclass 
class QuantumResource:
    """Quantum-inspired resource with superposition capabilities"""
    id: str
    name: str
    capacity: float
    current_load: float = 0.0
    quantum_efficiency: float = 1.0
    entangled_resources: List[str] = field(default_factory=list)
    coherence_decay_rate: float = 0.1


class QuantumTaskPlanner:
    """Quantum-inspired task planner with advanced optimization"""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.backlog_manager = BacklogManager(repo_root)
        self.quantum_tasks: Dict[str, QuantumTask] = {}
        self.quantum_resources: Dict[str, QuantumResource] = {}
        self.coherence_threshold = 0.7
        self.max_entanglement_depth = 3
        self.quantum_config = self._load_quantum_config()
        
    def _load_quantum_config(self) -> Dict:
        """Load quantum planner configuration"""
        config_file = self.repo_root / ".quantum_config.json"
        default_config = {
            "coherence_threshold": 0.7,
            "entanglement_strength": 0.8,
            "interference_factor": 0.3,
            "quantum_tunneling_probability": 0.15,
            "decoherence_rate": 0.05,
            "superposition_collapse_threshold": 0.9
        }
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception:
                pass
                
        return default_config
    
    def initialize_quantum_system(self) -> None:
        """Initialize quantum-inspired planning system"""
        self.backlog_manager.load_backlog()
        
        # Convert backlog items to quantum tasks
        for item in self.backlog_manager.items:
            quantum_task = self._create_quantum_task(item)
            self.quantum_tasks[item.id] = quantum_task
        
        # Initialize quantum resources
        self._initialize_quantum_resources()
        
        # Create entanglements between related tasks
        self._create_task_entanglements()
        
        # Calculate initial probability amplitudes
        self._calculate_probability_amplitudes()
    
    def _create_quantum_task(self, item: BacklogItem) -> QuantumTask:
        """Create quantum-enhanced task from backlog item"""
        quantum_task = QuantumTask(
            id=item.id,
            base_item=item,
            quantum_state=QuantumState.SUPERPOSITION
        )
        
        # Initialize probability amplitudes for different execution states
        quantum_task.probability_amplitudes = {
            "immediate": self._calculate_immediacy_amplitude(item),
            "parallel": self._calculate_parallelism_amplitude(item),
            "sequential": self._calculate_sequence_amplitude(item),
            "deferred": self._calculate_deferral_amplitude(item)
        }
        
        # Normalize amplitudes
        total = sum(quantum_task.probability_amplitudes.values())
        if total > 0:
            quantum_task.probability_amplitudes = {
                k: v / total for k, v in quantum_task.probability_amplitudes.items()
            }
        
        return quantum_task
    
    def _initialize_quantum_resources(self) -> None:
        """Initialize quantum resources based on system capabilities"""
        resource_types = [
            ("cpu", 8.0, 0.9),
            ("memory", 16.0, 0.8),
            ("network", 4.0, 0.7),
            ("developer_focus", 1.0, 1.0),
            ("testing_capacity", 2.0, 0.85)
        ]
        
        for name, capacity, efficiency in resource_types:
            resource = QuantumResource(
                id=name,
                name=name.replace("_", " ").title(),
                capacity=capacity,
                quantum_efficiency=efficiency
            )
            self.quantum_resources[name] = resource
    
    def _create_task_entanglements(self) -> None:
        """Create quantum entanglements between related tasks"""
        tasks = list(self.quantum_tasks.values())
        
        for i, task1 in enumerate(tasks):
            for task2 in tasks[i+1:]:
                entanglement_strength = self._calculate_entanglement_strength(
                    task1.base_item, task2.base_item
                )
                
                if entanglement_strength > self.quantum_config["entanglement_strength"]:
                    task1.entanglement_partners.append(task2.id)
                    task2.entanglement_partners.append(task1.id)
                    
                    # Update quantum states
                    if task1.quantum_state == QuantumState.SUPERPOSITION:
                        task1.quantum_state = QuantumState.ENTANGLED
                    if task2.quantum_state == QuantumState.SUPERPOSITION:
                        task2.quantum_state = QuantumState.ENTANGLED
    
    def _calculate_entanglement_strength(self, item1: BacklogItem, item2: BacklogItem) -> float:
        """Calculate quantum entanglement strength between tasks"""
        strength = 0.0
        
        # Type similarity
        if item1.type == item2.type:
            strength += 0.3
        
        # Description similarity (simplified)
        desc1_words = set(item1.description.lower().split())
        desc2_words = set(item2.description.lower().split())
        if desc1_words and desc2_words:
            similarity = len(desc1_words & desc2_words) / len(desc1_words | desc2_words)
            strength += similarity * 0.4
        
        # Effort correlation
        effort_diff = abs(item1.effort - item2.effort)
        if effort_diff <= 2:
            strength += 0.2
        
        # Risk tier correlation
        if item1.risk_tier == item2.risk_tier:
            strength += 0.1
        
        return min(strength, 1.0)
    
    def _calculate_probability_amplitudes(self) -> None:
        """Calculate quantum probability amplitudes for all tasks"""
        for task in self.quantum_tasks.values():
            task.quantum_priority = self._calculate_quantum_priority(task)
            task.optimal_execution_time = self._calculate_optimal_timing(task)
    
    def _calculate_immediacy_amplitude(self, item: BacklogItem) -> float:
        """Calculate amplitude for immediate execution"""
        base_amplitude = item.time_criticality / 10.0
        
        # Boost for high-value, low-effort tasks
        if item.value >= 7 and item.effort <= 3:
            base_amplitude *= 1.5
        
        # Reduce for high-risk tasks
        if item.risk_tier == "high":
            base_amplitude *= 0.7
        
        return min(base_amplitude, 1.0)
    
    def _calculate_parallelism_amplitude(self, item: BacklogItem) -> float:
        """Calculate amplitude for parallel execution"""
        # Tasks with low interdependency can run in parallel
        base_amplitude = 0.5
        
        if item.type in ["bug", "tech_debt"]:
            base_amplitude += 0.3
        
        if item.effort <= 5:
            base_amplitude += 0.2
        
        return min(base_amplitude, 1.0)
    
    def _calculate_sequence_amplitude(self, item: BacklogItem) -> float:
        """Calculate amplitude for sequential execution"""
        # Complex features often need sequential execution
        base_amplitude = item.effort / 13.0
        
        if item.type == "feature" and item.effort >= 8:
            base_amplitude += 0.4
        
        return min(base_amplitude, 1.0)
    
    def _calculate_deferral_amplitude(self, item: BacklogItem) -> float:
        """Calculate amplitude for deferred execution"""
        base_amplitude = 0.2
        
        # Defer low-priority, high-effort tasks
        priority_score = item.value + item.time_criticality + item.risk_reduction
        if priority_score <= 9 and item.effort >= 8:
            base_amplitude += 0.5
        
        return min(base_amplitude, 1.0)
    
    def _calculate_quantum_priority(self, task: QuantumTask) -> float:
        """Calculate quantum-enhanced priority score"""
        base_wsjf = self.backlog_manager.calculate_wsjf(task.base_item)
        
        # Apply quantum interference patterns
        interference_boost = sum(task.interference_pattern) if task.interference_pattern else 0
        
        # Consider coherence level
        coherence_factor = task.coherence_level
        
        # Entanglement bonus
        entanglement_bonus = len(task.entanglement_partners) * 0.1
        
        # Superposition advantage
        superposition_factor = 1.0
        if task.quantum_state == QuantumState.SUPERPOSITION:
            superposition_factor = 1.2
        elif task.quantum_state == QuantumState.COHERENT:
            superposition_factor = 1.5
        
        quantum_priority = (
            base_wsjf * superposition_factor * coherence_factor + 
            interference_boost + entanglement_bonus
        )
        
        return quantum_priority
    
    def _calculate_optimal_timing(self, task: QuantumTask) -> datetime.datetime:
        """Calculate optimal execution time using quantum algorithms"""
        now = datetime.datetime.now()
        
        # Base timing from traditional scheduling
        hours_offset = task.base_item.effort * 2
        
        # Quantum tunneling effect - sometimes tasks can complete faster
        if random.random() < self.quantum_config["quantum_tunneling_probability"]:
            hours_offset *= 0.7
        
        # Coherence decay - tasks lose optimality over time
        coherence_decay = task.coherence_level * self.quantum_config["decoherence_rate"]
        hours_offset += coherence_decay * 24
        
        # Interference from entangled tasks
        for partner_id in task.entanglement_partners:
            if partner_id in self.quantum_tasks:
                partner = self.quantum_tasks[partner_id]
                if partner.base_item.status in ["DOING", "PR"]:
                    hours_offset *= 1.1  # Slight delay due to interference
        
        return now + datetime.timedelta(hours=hours_offset)
    
    def collapse_superposition(self, task_id: str) -> str:
        """Collapse quantum superposition to determine execution strategy"""
        if task_id not in self.quantum_tasks:
            return "sequential"
        
        task = self.quantum_tasks[task_id]
        
        # Weighted random selection based on probability amplitudes
        choices = list(task.probability_amplitudes.keys())
        weights = list(task.probability_amplitudes.values())
        
        if not choices or sum(weights) == 0:
            return "sequential"
        
        # Quantum measurement
        collapsed_state = random.choices(choices, weights=weights)[0]
        
        # Update quantum state
        task.quantum_state = QuantumState.COLLAPSED
        task.coherence_level *= 0.9  # Slight decoherence after measurement
        
        return collapsed_state
    
    def optimize_quantum_schedule(self) -> List[Dict]:
        """Generate optimal execution schedule using quantum algorithms"""
        schedule = []
        
        # Sort tasks by quantum priority
        sorted_tasks = sorted(
            self.quantum_tasks.values(),
            key=lambda t: t.quantum_priority,
            reverse=True
        )
        
        current_time = datetime.datetime.now()
        resource_allocation = {r_id: 0.0 for r_id in self.quantum_resources.keys()}
        
        for task in sorted_tasks:
            if task.base_item.status not in ["READY", "NEW", "REFINED"]:
                continue
            
            # Collapse superposition to get execution strategy
            strategy = self.collapse_superposition(task.id)
            
            # Calculate resource requirements
            required_resources = self._calculate_resource_requirements(task, strategy)
            
            # Check resource availability with quantum efficiency
            can_schedule = self._check_quantum_resource_availability(
                required_resources, resource_allocation
            )
            
            if can_schedule:
                # Schedule the task
                execution_slot = {
                    "task_id": task.id,
                    "title": task.base_item.title,
                    "strategy": strategy,
                    "start_time": current_time.isoformat(),
                    "estimated_duration": task.base_item.effort * 2,  # hours
                    "quantum_priority": task.quantum_priority,
                    "coherence_level": task.coherence_level,
                    "entangled_tasks": task.entanglement_partners,
                    "resource_allocation": required_resources.copy()
                }
                
                schedule.append(execution_slot)
                
                # Update resource allocation
                for resource_id, amount in required_resources.items():
                    resource_allocation[resource_id] += amount
                
                # Update current time
                duration_hours = task.base_item.effort * 2
                current_time += datetime.timedelta(hours=duration_hours)
                
                # Apply quantum interference to entangled tasks
                self._apply_quantum_interference(task, schedule)
        
        return schedule
    
    def _calculate_resource_requirements(self, task: QuantumTask, strategy: str) -> Dict[str, float]:
        """Calculate quantum-enhanced resource requirements"""
        base_requirements = {
            "developer_focus": 1.0,
            "cpu": task.base_item.effort * 0.5,
            "memory": task.base_item.effort * 0.3,
            "network": 0.2,
            "testing_capacity": 0.5 if strategy != "deferred" else 0.1
        }
        
        # Strategy-specific adjustments
        if strategy == "parallel":
            base_requirements["cpu"] *= 1.5
            base_requirements["memory"] *= 1.3
        elif strategy == "immediate":
            base_requirements["developer_focus"] *= 1.2
            base_requirements["testing_capacity"] *= 1.4
        
        # Quantum efficiency boost
        quantum_efficiency = task.coherence_level
        for resource_id in base_requirements:
            if resource_id in self.quantum_resources:
                quantum_boost = self.quantum_resources[resource_id].quantum_efficiency
                base_requirements[resource_id] *= (2.0 - quantum_boost * quantum_efficiency)
        
        return base_requirements
    
    def _check_quantum_resource_availability(
        self, 
        required: Dict[str, float], 
        current_allocation: Dict[str, float]
    ) -> bool:
        """Check if quantum resources are available with superposition consideration"""
        for resource_id, required_amount in required.items():
            if resource_id not in self.quantum_resources:
                continue
            
            resource = self.quantum_resources[resource_id]
            current_load = current_allocation.get(resource_id, 0.0)
            
            # Quantum superposition allows some resource overallocation
            effective_capacity = resource.capacity * (1.0 + resource.quantum_efficiency * 0.2)
            
            if current_load + required_amount > effective_capacity:
                return False
        
        return True
    
    def _apply_quantum_interference(self, scheduled_task: QuantumTask, schedule: List[Dict]) -> None:
        """Apply quantum interference effects to entangled tasks"""
        for partner_id in scheduled_task.entanglement_partners:
            if partner_id in self.quantum_tasks:
                partner = self.quantum_tasks[partner_id]
                
                # Constructive interference - boost partner priority
                partner.quantum_priority *= 1.1
                
                # Update interference pattern
                interference_value = scheduled_task.coherence_level * 0.3
                partner.interference_pattern.append(interference_value)
                
                # Maintain entanglement coherence
                if len(partner.interference_pattern) > 5:
                    partner.interference_pattern = partner.interference_pattern[-5:]
    
    def simulate_quantum_execution(self, task_id: str) -> Dict:
        """Simulate quantum-inspired task execution"""
        if task_id not in self.quantum_tasks:
            return {"success": False, "error": "Task not found"}
        
        task = self.quantum_tasks[task_id]
        
        # Start quantum execution simulation
        execution_result = {
            "task_id": task_id,
            "initial_state": task.quantum_state.value,
            "coherence_level": task.coherence_level,
            "quantum_tunneling_occurred": False,
            "interference_effects": [],
            "final_coherence": task.coherence_level,
            "success": True
        }
        
        # Simulate quantum tunneling
        if random.random() < self.quantum_config["quantum_tunneling_probability"]:
            execution_result["quantum_tunneling_occurred"] = True
            # Tunneling reduces execution time
            task.base_item.effort = max(1, int(task.base_item.effort * 0.8))
        
        # Process entanglement effects
        for partner_id in task.entanglement_partners:
            if partner_id in self.quantum_tasks:
                partner = self.quantum_tasks[partner_id]
                interference_effect = {
                    "partner_id": partner_id,
                    "effect_type": "constructive" if partner.coherence_level > 0.7 else "destructive",
                    "magnitude": abs(task.coherence_level - partner.coherence_level)
                }
                execution_result["interference_effects"].append(interference_effect)
        
        # Apply decoherence
        decoherence = self.quantum_config["decoherence_rate"] * (1.0 - task.coherence_level)
        task.coherence_level = max(0.1, task.coherence_level - decoherence)
        execution_result["final_coherence"] = task.coherence_level
        
        # Collapse to final state
        task.quantum_state = QuantumState.COLLAPSED
        
        return execution_result
    
    def get_quantum_insights(self) -> Dict:
        """Generate quantum planning insights and recommendations"""
        insights = {
            "system_coherence": self._calculate_system_coherence(),
            "entanglement_clusters": self._identify_entanglement_clusters(),
            "quantum_bottlenecks": self._identify_quantum_bottlenecks(),
            "optimization_suggestions": self._generate_optimization_suggestions(),
            "next_optimal_tasks": self._get_next_optimal_tasks(5)
        }
        
        return insights
    
    def _calculate_system_coherence(self) -> float:
        """Calculate overall quantum system coherence"""
        if not self.quantum_tasks:
            return 0.0
        
        total_coherence = sum(task.coherence_level for task in self.quantum_tasks.values())
        return total_coherence / len(self.quantum_tasks)
    
    def _identify_entanglement_clusters(self) -> List[Dict]:
        """Identify clusters of entangled tasks"""
        clusters = []
        visited = set()
        
        for task_id, task in self.quantum_tasks.items():
            if task_id in visited or not task.entanglement_partners:
                continue
            
            # BFS to find connected component
            cluster = []
            queue = [task_id]
            cluster_visited = set()
            
            while queue:
                current_id = queue.pop(0)
                if current_id in cluster_visited:
                    continue
                
                cluster_visited.add(current_id)
                cluster.append(current_id)
                
                if current_id in self.quantum_tasks:
                    for partner_id in self.quantum_tasks[current_id].entanglement_partners:
                        if partner_id not in cluster_visited:
                            queue.append(partner_id)
            
            if len(cluster) > 1:
                clusters.append({
                    "size": len(cluster),
                    "tasks": cluster,
                    "average_coherence": sum(
                        self.quantum_tasks[tid].coherence_level 
                        for tid in cluster if tid in self.quantum_tasks
                    ) / len(cluster)
                })
            
            visited.update(cluster)
        
        return sorted(clusters, key=lambda c: c["size"], reverse=True)
    
    def _identify_quantum_bottlenecks(self) -> List[Dict]:
        """Identify quantum bottlenecks in the system"""
        bottlenecks = []
        
        # Resource bottlenecks
        for resource_id, resource in self.quantum_resources.items():
            total_demand = sum(
                self._calculate_resource_requirements(task, "sequential").get(resource_id, 0)
                for task in self.quantum_tasks.values()
                if task.base_item.status in ["READY", "NEW"]
            )
            
            if total_demand > resource.capacity * 1.5:
                bottlenecks.append({
                    "type": "resource",
                    "resource_id": resource_id,
                    "demand": total_demand,
                    "capacity": resource.capacity,
                    "overload_factor": total_demand / resource.capacity
                })
        
        # Coherence bottlenecks
        low_coherence_tasks = [
            task for task in self.quantum_tasks.values()
            if task.coherence_level < self.coherence_threshold
        ]
        
        if low_coherence_tasks:
            bottlenecks.append({
                "type": "coherence",
                "affected_tasks": len(low_coherence_tasks),
                "average_coherence": sum(t.coherence_level for t in low_coherence_tasks) / len(low_coherence_tasks),
                "task_ids": [t.id for t in low_coherence_tasks]
            })
        
        return bottlenecks
    
    def _generate_optimization_suggestions(self) -> List[str]:
        """Generate quantum optimization suggestions"""
        suggestions = []
        
        system_coherence = self._calculate_system_coherence()
        if system_coherence < 0.6:
            suggestions.append("Consider increasing system coherence by reducing task complexity")
        
        entanglement_clusters = self._identify_entanglement_clusters()
        if entanglement_clusters and entanglement_clusters[0]["size"] > 5:
            suggestions.append("Large entanglement cluster detected - consider breaking into smaller groups")
        
        bottlenecks = self._identify_quantum_bottlenecks()
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "resource":
                suggestions.append(f"Resource bottleneck: {bottleneck['resource_id']} is overloaded")
            elif bottleneck["type"] == "coherence":
                suggestions.append("Multiple tasks have low coherence - review task definitions")
        
        return suggestions
    
    def _get_next_optimal_tasks(self, count: int) -> List[Dict]:
        """Get next optimal tasks for execution"""
        ready_tasks = [
            task for task in self.quantum_tasks.values()
            if task.base_item.status in ["READY", "NEW", "REFINED"]
        ]
        
        # Sort by quantum priority
        ready_tasks.sort(key=lambda t: t.quantum_priority, reverse=True)
        
        optimal_tasks = []
        for task in ready_tasks[:count]:
            optimal_tasks.append({
                "task_id": task.id,
                "title": task.base_item.title,
                "quantum_priority": task.quantum_priority,
                "coherence_level": task.coherence_level,
                "quantum_state": task.quantum_state.value,
                "optimal_strategy": self.collapse_superposition(task.id),
                "estimated_completion": task.optimal_execution_time.isoformat() if task.optimal_execution_time else None
            })
        
        return optimal_tasks
    
    def save_quantum_state(self) -> None:
        """Save current quantum system state"""
        quantum_state_dir = self.repo_root / "quantum_state"
        quantum_state_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        state_file = quantum_state_dir / f"quantum_state_{timestamp}.json"
        
        state_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "system_coherence": self._calculate_system_coherence(),
            "total_tasks": len(self.quantum_tasks),
            "quantum_tasks": {
                task_id: {
                    "quantum_state": task.quantum_state.value,
                    "coherence_level": task.coherence_level,
                    "quantum_priority": task.quantum_priority,
                    "entanglement_partners": task.entanglement_partners,
                    "probability_amplitudes": task.probability_amplitudes
                }
                for task_id, task in self.quantum_tasks.items()
            },
            "quantum_resources": {
                resource_id: {
                    "capacity": resource.capacity,
                    "current_load": resource.current_load,
                    "quantum_efficiency": resource.quantum_efficiency
                }
                for resource_id, resource in self.quantum_resources.items()
            }
        }
        
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        # Also save as latest
        latest_file = quantum_state_dir / "latest_quantum_state.json"
        with open(latest_file, 'w') as f:
            json.dump(state_data, f, indent=2)


def main():
    """CLI entry point for quantum task planner"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quantum_task_planner.py <command>")
        print("Commands: init, optimize, insights, simulate <task_id>")
        return
    
    command = sys.argv[1]
    planner = QuantumTaskPlanner()
    
    if command == "init":
        print("🌀 Initializing quantum task planning system...")
        planner.initialize_quantum_system()
        planner.save_quantum_state()
        print("✅ Quantum system initialized")
        
    elif command == "optimize":
        print("🔮 Generating quantum-optimized schedule...")
        planner.initialize_quantum_system()
        schedule = planner.optimize_quantum_schedule()
        print(json.dumps(schedule, indent=2))
        
    elif command == "insights":
        print("🧠 Generating quantum planning insights...")
        planner.initialize_quantum_system()
        insights = planner.get_quantum_insights()
        print(json.dumps(insights, indent=2))
        
    elif command == "simulate" and len(sys.argv) > 2:
        task_id = sys.argv[2]
        print(f"⚛️  Simulating quantum execution for task: {task_id}")
        planner.initialize_quantum_system()
        result = planner.simulate_quantum_execution(task_id)
        print(json.dumps(result, indent=2))
        
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()