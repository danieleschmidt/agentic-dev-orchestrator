#!/usr/bin/env python3
"""
Adaptive Learning Engine for ADO
Implements machine learning capabilities for continuous improvement
"""

import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics


@dataclass
class LearningInsight:
    """Insight generated from adaptive learning"""
    category: str
    insight: str
    confidence: float
    supporting_data: Dict[str, Any]
    recommendations: List[str]
    timestamp: str


class AdaptiveLearningEngine:
    """Adaptive learning system for continuous SDLC improvement"""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.data_dir = self.repo_root / "learning_data"
        self.data_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger("adaptive_learning")
        self.insights_history = []
        
    def analyze_execution_patterns(self, execution_history: List[Dict]) -> List[LearningInsight]:
        """Analyze execution patterns to identify improvement opportunities"""
        insights = []
        
        if not execution_history:
            return insights
            
        # Analyze success/failure patterns
        insights.extend(self._analyze_success_patterns(execution_history))
        
        # Analyze timing patterns
        insights.extend(self._analyze_timing_patterns(execution_history))
        
        # Analyze resource utilization patterns
        insights.extend(self._analyze_resource_patterns(execution_history))
        
        # Analyze error patterns
        insights.extend(self._analyze_error_patterns(execution_history))
        
        return insights
        
    def _analyze_success_patterns(self, history: List[Dict]) -> List[LearningInsight]:
        """Analyze success/failure patterns"""
        insights = []
        
        # Calculate success rate over time
        success_rates = []
        for record in history:
            completed = len(record.get('completed_items', []))
            total = completed + len(record.get('failed_items', []))
            if total > 0:
                success_rates.append(completed / total)
                
        if len(success_rates) >= 5:
            recent_avg = statistics.mean(success_rates[-5:])
            overall_avg = statistics.mean(success_rates)
            
            if recent_avg > overall_avg + 0.1:
                insights.append(LearningInsight(
                    category="performance",
                    insight="Success rate has improved significantly in recent executions",
                    confidence=0.8,
                    supporting_data={
                        "recent_avg": recent_avg,
                        "overall_avg": overall_avg,
                        "improvement": recent_avg - overall_avg
                    },
                    recommendations=[
                        "Continue current practices that led to improved performance",
                        "Document successful patterns for future reference"
                    ],
                    timestamp=datetime.now().isoformat()
                ))
            elif recent_avg < overall_avg - 0.1:
                insights.append(LearningInsight(
                    category="performance",
                    insight="Success rate has declined in recent executions",
                    confidence=0.8,
                    supporting_data={
                        "recent_avg": recent_avg,
                        "overall_avg": overall_avg,
                        "decline": overall_avg - recent_avg
                    },
                    recommendations=[
                        "Review recent changes in process or configuration",
                        "Increase monitoring and validation steps",
                        "Consider rolling back recent changes"
                    ],
                    timestamp=datetime.now().isoformat()
                ))
                
        return insights
        
    def _analyze_timing_patterns(self, history: List[Dict]) -> List[LearningInsight]:
        """Analyze execution timing patterns"""
        insights = []
        
        # Analyze execution duration patterns
        durations = []
        for record in history:
            start_time = record.get('start_time')
            end_time = record.get('end_time')
            if start_time and end_time:
                try:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    duration = (end_dt - start_dt).total_seconds()
                    durations.append(duration)
                except (ValueError, AttributeError):
                    continue
                    
        if len(durations) >= 5:
            recent_avg = statistics.mean(durations[-5:])
            overall_avg = statistics.mean(durations)
            
            if recent_avg > overall_avg * 1.5:
                insights.append(LearningInsight(
                    category="performance",
                    insight="Execution times have increased significantly",
                    confidence=0.7,
                    supporting_data={
                        "recent_avg_seconds": recent_avg,
                        "overall_avg_seconds": overall_avg,
                        "slowdown_factor": recent_avg / overall_avg
                    },
                    recommendations=[
                        "Profile execution to identify bottlenecks",
                        "Review resource allocation and system load",
                        "Consider optimizing critical path operations"
                    ],
                    timestamp=datetime.now().isoformat()
                ))
                
        # Analyze optimal execution times
        if durations:
            hour_performance = defaultdict(list)
            for i, record in enumerate(history):
                if i < len(durations):
                    start_time = record.get('start_time')
                    if start_time:
                        try:
                            start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                            hour = start_dt.hour
                            hour_performance[hour].append(durations[i])
                        except (ValueError, AttributeError):
                            continue
                            
            # Find optimal execution hours
            hour_avgs = {hour: statistics.mean(times) 
                        for hour, times in hour_performance.items() 
                        if len(times) >= 2}
                        
            if len(hour_avgs) >= 3:
                best_hour = min(hour_avgs, key=hour_avgs.get)
                worst_hour = max(hour_avgs, key=hour_avgs.get)
                
                if hour_avgs[worst_hour] > hour_avgs[best_hour] * 1.3:
                    insights.append(LearningInsight(
                        category="scheduling",
                        insight=f"Executions perform best around {best_hour}:00 and worst around {worst_hour}:00",
                        confidence=0.6,
                        supporting_data={
                            "best_hour": best_hour,
                            "worst_hour": worst_hour,
                            "best_avg_seconds": hour_avgs[best_hour],
                            "worst_avg_seconds": hour_avgs[worst_hour]
                        },
                        recommendations=[
                            f"Schedule automated executions around {best_hour}:00 for optimal performance",
                            f"Avoid scheduling during {worst_hour}:00 when possible",
                            "Consider system resource availability patterns"
                        ],
                        timestamp=datetime.now().isoformat()
                    ))
                    
        return insights
        
    def _analyze_resource_patterns(self, history: List[Dict]) -> List[LearningInsight]:
        """Analyze resource utilization patterns"""
        insights = []
        
        # Analyze memory usage patterns
        memory_usages = []
        for record in history:
            memory_usage = record.get('memory_usage_mb')
            if memory_usage and isinstance(memory_usage, (int, float)):
                memory_usages.append(memory_usage)
                
        if len(memory_usages) >= 5:
            recent_avg = statistics.mean(memory_usages[-5:])
            overall_avg = statistics.mean(memory_usages)
            
            if recent_avg > overall_avg * 1.5:
                insights.append(LearningInsight(
                    category="resources",
                    insight="Memory usage has increased significantly in recent executions",
                    confidence=0.7,
                    supporting_data={
                        "recent_avg_mb": recent_avg,
                        "overall_avg_mb": overall_avg,
                        "increase_factor": recent_avg / overall_avg
                    },
                    recommendations=[
                        "Monitor for memory leaks in recent changes",
                        "Review data structures and caching strategies",
                        "Consider implementing memory-efficient algorithms"
                    ],
                    timestamp=datetime.now().isoformat()
                ))
                
        return insights
        
    def _analyze_error_patterns(self, history: List[Dict]) -> List[LearningInsight]:
        """Analyze error patterns and common failure modes"""
        insights = []
        
        # Collect all error messages
        error_patterns = defaultdict(int)
        for record in history:
            failed_items = record.get('failed_items', [])
            for item in failed_items:
                error_msg = item.get('error_message', '')
                if error_msg:
                    # Extract key error patterns
                    if 'timeout' in error_msg.lower():
                        error_patterns['timeout'] += 1
                    elif 'permission' in error_msg.lower():
                        error_patterns['permission'] += 1
                    elif 'network' in error_msg.lower():
                        error_patterns['network'] += 1
                    elif 'memory' in error_msg.lower():
                        error_patterns['memory'] += 1
                    else:
                        error_patterns['other'] += 1
                        
        if error_patterns:
            most_common_error = max(error_patterns, key=error_patterns.get)
            error_count = error_patterns[most_common_error]
            total_errors = sum(error_patterns.values())
            
            if error_count > total_errors * 0.4:  # More than 40% of errors are this type
                recommendations = {
                    'timeout': [
                        "Increase timeout values for long-running operations",
                        "Implement retry mechanisms with exponential backoff",
                        "Break down large tasks into smaller chunks"
                    ],
                    'permission': [
                        "Review and update access permissions",
                        "Ensure service accounts have necessary privileges",
                        "Implement proper authentication mechanisms"
                    ],
                    'network': [
                        "Implement network retry logic",
                        "Add network connectivity checks",
                        "Consider offline/degraded mode capabilities"
                    ],
                    'memory': [
                        "Optimize memory usage and implement garbage collection",
                        "Increase available memory resources",
                        "Implement streaming for large data processing"
                    ]
                }
                
                insights.append(LearningInsight(
                    category="reliability",
                    insight=f"'{most_common_error}' errors account for {error_count}/{total_errors} failures",
                    confidence=0.8,
                    supporting_data={
                        "error_type": most_common_error,
                        "error_count": error_count,
                        "total_errors": total_errors,
                        "percentage": (error_count / total_errors) * 100
                    },
                    recommendations=recommendations.get(most_common_error, [
                        "Investigate root cause of recurring errors",
                        "Implement better error handling and recovery"
                    ]),
                    timestamp=datetime.now().isoformat()
                ))
                
        return insights
        
    def learn_from_backlog_patterns(self, backlog_history: List[Dict]) -> List[LearningInsight]:
        """Learn from backlog item patterns and WSJF effectiveness"""
        insights = []
        
        if not backlog_history:
            return insights
            
        # Analyze WSJF prediction accuracy
        wsjf_accuracy = self._analyze_wsjf_accuracy(backlog_history)
        if wsjf_accuracy:
            insights.append(wsjf_accuracy)
            
        # Analyze completion time patterns
        completion_patterns = self._analyze_completion_patterns(backlog_history)
        insights.extend(completion_patterns)
        
        return insights
        
    def _analyze_wsjf_accuracy(self, backlog_history: List[Dict]) -> Optional[LearningInsight]:
        """Analyze how well WSJF scores predict actual value delivery"""
        completed_items = [item for item in backlog_history 
                          if item.get('status') == 'DONE']
        
        if len(completed_items) < 5:
            return None
            
        # Calculate correlation between WSJF score and completion speed
        wsjf_scores = []
        completion_days = []
        
        for item in completed_items:
            wsjf = item.get('wsjf', {})
            if isinstance(wsjf, dict):
                score = self._calculate_wsjf_score(wsjf)
                if score > 0:
                    wsjf_scores.append(score)
                    
                    # Calculate completion days (mock calculation)
                    created = item.get('created_at')
                    completed = item.get('completed_at')
                    if created and completed:
                        try:
                            created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                            completed_dt = datetime.fromisoformat(completed.replace('Z', '+00:00'))
                            days = (completed_dt - created_dt).days
                            completion_days.append(max(1, days))  # At least 1 day
                        except (ValueError, AttributeError):
                            completion_days.append(7)  # Default to 1 week
                            
        if len(wsjf_scores) >= 5 and len(completion_days) >= 5:
            # Simple correlation calculation
            correlation = self._calculate_correlation(wsjf_scores, completion_days)
            
            if abs(correlation) > 0.3:  # Meaningful correlation
                return LearningInsight(
                    category="prioritization",
                    insight=f"WSJF scores show {'negative' if correlation < 0 else 'positive'} correlation with completion time",
                    confidence=min(0.9, abs(correlation)),
                    supporting_data={
                        "correlation": correlation,
                        "sample_size": len(wsjf_scores),
                        "avg_wsjf_score": statistics.mean(wsjf_scores),
                        "avg_completion_days": statistics.mean(completion_days)
                    },
                    recommendations=[
                        "WSJF scoring is effective for prioritization" if correlation < -0.3 else
                        "Review WSJF criteria - high scores may not translate to faster completion",
                        "Consider adjusting job size estimates",
                        "Monitor delivery value vs. predicted value"
                    ],
                    timestamp=datetime.now().isoformat()
                )
                
        return None
        
    def _analyze_completion_patterns(self, backlog_history: List[Dict]) -> List[LearningInsight]:
        """Analyze patterns in task completion"""
        insights = []
        
        # Analyze by job size patterns
        size_completion_times = defaultdict(list)
        
        for item in backlog_history:
            if item.get('status') == 'DONE':
                wsjf = item.get('wsjf', {})
                job_size = wsjf.get('job_size', 0) if isinstance(wsjf, dict) else 0
                
                if job_size > 0:
                    # Mock completion time calculation
                    completion_time = job_size * 2 + np.random.normal(0, 1)  # Simulated
                    size_completion_times[job_size].append(max(1, completion_time))
                    
        if len(size_completion_times) >= 2:
            avg_times = {size: statistics.mean(times) 
                        for size, times in size_completion_times.items() 
                        if len(times) >= 2}
                        
            if len(avg_times) >= 2:
                sizes = list(avg_times.keys())
                times = list(avg_times.values())
                
                # Check if larger jobs take proportionally longer
                correlation = self._calculate_correlation(sizes, times)
                
                if correlation > 0.5:
                    insights.append(LearningInsight(
                        category="estimation",
                        insight="Job size estimates correlate well with actual completion time",
                        confidence=0.7,
                        supporting_data={
                            "correlation": correlation,
                            "size_time_mapping": avg_times
                        },
                        recommendations=[
                            "Current job size estimation approach is effective",
                            "Use historical data to improve future estimates",
                            "Consider breaking down very large jobs (>8) into smaller tasks"
                        ],
                        timestamp=datetime.now().isoformat()
                    ))
                elif correlation < 0.2:
                    insights.append(LearningInsight(
                        category="estimation",
                        insight="Job size estimates poorly correlate with completion time",
                        confidence=0.6,
                        supporting_data={
                            "correlation": correlation,
                            "size_time_mapping": avg_times
                        },
                        recommendations=[
                            "Review job size estimation criteria",
                            "Consider additional factors like complexity and dependencies",
                            "Implement story point refinement sessions"
                        ],
                        timestamp=datetime.now().isoformat()
                    ))
                    
        return insights
        
    def _calculate_wsjf_score(self, wsjf: Dict) -> float:
        """Calculate WSJF score from components"""
        numerator = (
            wsjf.get('user_business_value', 0) +
            wsjf.get('time_criticality', 0) +
            wsjf.get('risk_reduction_opportunity_enablement', 0)
        )
        denominator = wsjf.get('job_size', 1)
        return numerator / denominator if denominator > 0 else 0
        
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
            
        try:
            x_arr = np.array(x)
            y_arr = np.array(y)
            correlation_matrix = np.corrcoef(x_arr, y_arr)
            return correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
        except Exception:
            return 0.0
            
    def save_insights(self, insights: List[LearningInsight]) -> Path:
        """Save learning insights to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        insights_file = self.data_dir / f"insights_{timestamp}.json"
        
        insights_data = [asdict(insight) for insight in insights]
        
        with open(insights_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_insights': len(insights),
                'insights': insights_data
            }, f, indent=2)
            
        # Save as latest
        latest_file = self.data_dir / "insights_latest.json"
        with open(latest_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_insights': len(insights),
                'insights': insights_data
            }, f, indent=2)
            
        self.insights_history.extend(insights)
        return insights_file
        
    def generate_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning report"""
        # Load historical execution data
        execution_history = self._load_execution_history()
        backlog_history = self._load_backlog_history()
        
        # Generate insights
        execution_insights = self.analyze_execution_patterns(execution_history)
        backlog_insights = self.learn_from_backlog_patterns(backlog_history)
        
        all_insights = execution_insights + backlog_insights
        
        # Categorize insights
        categories = defaultdict(list)
        for insight in all_insights:
            categories[insight.category].append(insight)
            
        # Generate summary
        high_confidence_insights = [i for i in all_insights if i.confidence > 0.7]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_insights': len(all_insights),
            'high_confidence_insights': len(high_confidence_insights),
            'categories': {cat: len(insights) for cat, insights in categories.items()},
            'top_insights': [asdict(i) for i in sorted(all_insights, key=lambda x: x.confidence, reverse=True)[:5]],
            'recommendations': self._consolidate_recommendations(all_insights)
        }
        
    def _load_execution_history(self) -> List[Dict]:
        """Load execution history from status files"""
        history = []
        status_dir = self.repo_root / "docs" / "status"
        
        if status_dir.exists():
            for status_file in status_dir.glob("status_*.json"):
                try:
                    with open(status_file, 'r') as f:
                        data = json.load(f)
                        history.append(data)
                except Exception as e:
                    self.logger.warning(f"Could not load {status_file}: {e}")
                    
        return history[-50:]  # Last 50 executions
        
    def _load_backlog_history(self) -> List[Dict]:
        """Load backlog history"""
        history = []
        backlog_dir = self.repo_root / "backlog"
        
        if backlog_dir.exists():
            for backlog_file in backlog_dir.glob("*.json"):
                try:
                    with open(backlog_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            history.extend(data)
                        else:
                            history.append(data)
                except Exception as e:
                    self.logger.warning(f"Could not load {backlog_file}: {e}")
                    
        return history
        
    def _consolidate_recommendations(self, insights: List[LearningInsight]) -> List[str]:
        """Consolidate recommendations from all insights"""
        all_recommendations = []
        for insight in insights:
            if insight.confidence > 0.6:  # Only high-confidence insights
                all_recommendations.extend(insight.recommendations)
                
        # Remove duplicates and return top recommendations
        unique_recommendations = list(set(all_recommendations))
        return unique_recommendations[:10]


def main():
    """CLI entry point for adaptive learning"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python adaptive_learning.py <command>")
        print("Commands: analyze, report")
        return
        
    command = sys.argv[1]
    engine = AdaptiveLearningEngine()
    
    if command == "analyze":
        execution_history = engine._load_execution_history()
        insights = engine.analyze_execution_patterns(execution_history)
        
        insights_file = engine.save_insights(insights)
        print(f"Generated {len(insights)} insights, saved to: {insights_file}")
        
        for insight in insights:
            print(f"\n{insight.category.upper()}: {insight.insight}")
            print(f"Confidence: {insight.confidence:.2f}")
            for rec in insight.recommendations[:2]:
                print(f"  • {rec}")
                
    elif command == "report":
        report = engine.generate_learning_report()
        print(json.dumps(report, indent=2))
        
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
