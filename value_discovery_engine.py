#!/usr/bin/env python3
"""
Terragon Value Discovery Engine
Autonomous SDLC value discovery and prioritization system
"""

import json
import yaml
import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskCategory(Enum):
    SECURITY = "security"
    TECHNICAL_DEBT = "technical_debt" 
    PERFORMANCE = "performance"
    FEATURE = "feature"
    DOCUMENTATION = "documentation"
    DEPENDENCY = "dependency"
    INFRASTRUCTURE = "infrastructure"
    COMPLIANCE = "compliance"


@dataclass
class ValueItem:
    """Represents a discovered value item with scoring"""
    id: str
    title: str
    description: str
    category: TaskCategory
    
    # WSJF Components
    user_business_value: float  # 1-10
    time_criticality: float     # 1-10
    risk_reduction: float       # 1-10
    opportunity_enablement: float # 1-10
    job_size: float            # Story points or hours
    
    # ICE Components  
    impact: float              # 1-10
    confidence: float          # 1-10
    ease: float               # 1-10
    
    # Technical Debt Components
    debt_impact: float         # Hours saved annually
    debt_interest: float       # Future cost if not addressed
    hotspot_multiplier: float  # Based on file churn/complexity
    
    # Metadata
    source: str               # Discovery source
    file_paths: List[str]     # Affected files
    estimated_hours: float    # Effort estimate
    discovered_at: datetime
    
    # Calculated scores
    wsjf_score: Optional[float] = None
    ice_score: Optional[float] = None
    technical_debt_score: Optional[float] = None
    composite_score: Optional[float] = None


class ValueDiscoveryEngine:
    """Main engine for discovering and scoring value items"""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.discovered_items: List[ValueItem] = []
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def discover_all_value_items(self) -> List[ValueItem]:
        """Run all discovery methods and return prioritized list"""
        logger.info("Starting comprehensive value discovery...")
        
        # Clear previous discoveries
        self.discovered_items = []
        
        # Run discovery methods based on configuration
        if self._is_source_enabled("gitHistory"):
            self._discover_from_git_history()
            
        if self._is_source_enabled("staticAnalysis"):
            self._discover_from_static_analysis()
            
        if self._is_source_enabled("vulnerabilityDatabases"):
            self._discover_from_security_scans()
            
        if self._is_source_enabled("dependencyTracking"):
            self._discover_from_dependency_analysis()
            
        if self._is_source_enabled("performanceMonitoring"):
            self._discover_from_performance_analysis()
            
        if self._is_source_enabled("architecturalAnalysis"):
            self._discover_from_architectural_analysis()
        
        # Calculate composite scores for all items
        self._calculate_composite_scores()
        
        # Sort by composite score descending
        self.discovered_items.sort(key=lambda x: x.composite_score or 0, reverse=True)
        
        logger.info(f"Discovered {len(self.discovered_items)} value items")
        return self.discovered_items
    
    def _is_source_enabled(self, source: str) -> bool:
        """Check if discovery source is enabled in config"""
        enabled_sources = self.config.get("discovery", {}).get("sources", {}).get("enabled", [])
        return source in enabled_sources
    
    def _discover_from_git_history(self):
        """Discover items from git commit history analysis"""
        logger.info("Analyzing git history for value opportunities...")
        
        try:
            # Get recent commits with TODO/FIXME patterns
            result = subprocess.run([
                'git', 'log', '--grep=TODO\\|FIXME\\|HACK\\|DEPRECATED', 
                '--oneline', '--since=30 days ago'
            ], capture_output=True, text=True, check=True)
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    commit_hash = line.split(' ')[0]
                    commit_msg = ' '.join(line.split(' ')[1:])
                    
                    item = ValueItem(
                        id=f"git-{commit_hash[:8]}",
                        title=f"Address technical debt: {commit_msg[:50]}...",
                        description=f"Technical debt identified in commit: {commit_msg}",
                        category=TaskCategory.TECHNICAL_DEBT,
                        user_business_value=5.0,
                        time_criticality=3.0,
                        risk_reduction=6.0,
                        opportunity_enablement=4.0,
                        job_size=3.0,
                        impact=6.0,
                        confidence=8.0,
                        ease=7.0,
                        debt_impact=8.0,
                        debt_interest=5.0,
                        hotspot_multiplier=1.0,
                        source="git_history",
                        file_paths=[],
                        estimated_hours=3.0,
                        discovered_at=datetime.now()
                    )
                    self.discovered_items.append(item)
                    
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git history analysis failed: {e}")
    
    def _discover_from_static_analysis(self):
        """Discover items from static code analysis"""
        logger.info("Running static analysis for value discovery...")
        
        # Fallback to basic code analysis if ruff not available
        try:
            result = subprocess.run(['ruff', 'check', '.', '--format=json'], 
                                  capture_output=True, text=True)
            
            if result.stdout:
                issues = json.loads(result.stdout)
                
                # Group issues by file for hotspot analysis
                file_issues = {}
                for issue in issues:
                    file_path = issue.get('filename', 'unknown')
                    if file_path not in file_issues:
                        file_issues[file_path] = []
                    file_issues[file_path].append(issue)
                
                # Create value items for files with multiple issues (hotspots)
                for file_path, file_issue_list in file_issues.items():
                    if len(file_issue_list) >= 3:  # Hotspot threshold
                        item = ValueItem(
                            id=f"static-{hash(file_path) % 10000}",
                            title=f"Refactor code quality hotspot: {Path(file_path).name}",
                            description=f"File has {len(file_issue_list)} code quality issues",
                            category=TaskCategory.TECHNICAL_DEBT,
                            user_business_value=4.0,
                            time_criticality=2.0,
                            risk_reduction=7.0,
                            opportunity_enablement=5.0,
                            job_size=len(file_issue_list) * 0.5,  # Scale with issue count
                            impact=7.0,
                            confidence=9.0,
                            ease=6.0,
                            debt_impact=len(file_issue_list) * 2.0,
                            debt_interest=len(file_issue_list) * 1.0,
                            hotspot_multiplier=min(len(file_issue_list) / 3.0, 5.0),
                            source="static_analysis",
                            file_paths=[file_path],
                            estimated_hours=len(file_issue_list) * 0.5,
                            discovered_at=datetime.now()
                        )
                        self.discovered_items.append(item)
                        
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Ruff not available, using basic code analysis: {e}")
            
            # Fallback: analyze Python files for basic patterns
            try:
                python_files = list(Path('.').glob('**/*.py'))
                for py_file in python_files[:10]:  # Limit to avoid overwhelming
                    if py_file.stat().st_size > 5000:  # Files > 5KB might need refactoring
                        item = ValueItem(
                            id=f"large-file-{hash(str(py_file)) % 10000}",
                            title=f"Consider refactoring large file: {py_file.name}",
                            description=f"Large Python file ({py_file.stat().st_size} bytes) may need refactoring",
                            category=TaskCategory.TECHNICAL_DEBT,
                            user_business_value=3.0,
                            time_criticality=1.0,
                            risk_reduction=5.0,
                            opportunity_enablement=4.0,
                            job_size=4.0,
                            impact=5.0,
                            confidence=6.0,
                            ease=4.0,
                            debt_impact=8.0,
                            debt_interest=3.0,
                            hotspot_multiplier=1.2,
                            source="basic_file_analysis",
                            file_paths=[str(py_file)],
                            estimated_hours=4.0,
                            discovered_at=datetime.now()
                        )
                        self.discovered_items.append(item)
            except Exception as e:
                logger.warning(f"Basic file analysis failed: {e}")
    
    def _discover_from_security_scans(self):
        """Discover security-related value items"""
        logger.info("Scanning for security vulnerabilities...")
        
        # Run safety check for dependency vulnerabilities
        try:
            result = subprocess.run(['safety', 'check', '--json'], 
                                  capture_output=True, text=True)
            
            if result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    vulnerabilities = safety_data.get('vulnerabilities', [])
                    
                    for vuln in vulnerabilities:
                        item = ValueItem(
                            id=f"sec-{vuln.get('id', 'unknown')}",
                            title=f"Fix security vulnerability: {vuln.get('package_name', 'unknown')}",
                            description=f"Security vulnerability: {vuln.get('advisory', 'No description')}",
                            category=TaskCategory.SECURITY,
                            user_business_value=9.0,  # High for security
                            time_criticality=8.0,
                            risk_reduction=10.0,
                            opportunity_enablement=3.0,
                            job_size=2.0,  # Usually quick fixes
                            impact=9.0,
                            confidence=9.0,
                            ease=8.0,
                            debt_impact=5.0,
                            debt_interest=20.0,  # High future cost
                            hotspot_multiplier=2.0,  # Security boost
                            source="security_scan",
                            file_paths=["requirements.txt", "pyproject.toml"],
                            estimated_hours=2.0,
                            discovered_at=datetime.now()
                        )
                        self.discovered_items.append(item)
                        
                except json.JSONDecodeError:
                    pass  # Safety output might not be JSON in all cases
                    
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Safety check failed or not installed")
            
            # Fallback: create generic security improvement item
            item = ValueItem(
                id="sec-general-001",
                title="Implement comprehensive security scanning pipeline",
                description="Set up automated security scanning with tools like bandit, safety, and pip-audit",
                category=TaskCategory.SECURITY,
                user_business_value=8.0,
                time_criticality=6.0,
                risk_reduction=9.0,
                opportunity_enablement=5.0,
                job_size=4.0,
                impact=8.0,
                confidence=8.0,
                ease=6.0,
                debt_impact=10.0,
                debt_interest=15.0,
                hotspot_multiplier=1.5,
                source="security_baseline",
                file_paths=["pyproject.toml", "requirements.txt"],
                estimated_hours=4.0,
                discovered_at=datetime.now()
            )
            self.discovered_items.append(item)
    
    def _discover_from_dependency_analysis(self):
        """Discover dependency update opportunities"""
        logger.info("Analyzing dependencies for updates...")
        
        # Check for outdated packages
        try:
            result = subprocess.run(['pip3', 'list', '--outdated', '--format=json'], 
                                  capture_output=True, text=True, check=True)
            
            if result.stdout:
                outdated_packages = json.loads(result.stdout)
                
                for package in outdated_packages:
                    # Prioritize security-related packages
                    security_packages = ['cryptography', 'requests', 'urllib3', 'pillow', 'django']
                    is_security = package['name'].lower() in security_packages
                    
                    item = ValueItem(
                        id=f"dep-{package['name']}",
                        title=f"Update dependency: {package['name']}",
                        description=f"Update {package['name']} from {package['version']} to {package['latest_version']}",
                        category=TaskCategory.SECURITY if is_security else TaskCategory.DEPENDENCY,
                        user_business_value=7.0 if is_security else 3.0,
                        time_criticality=6.0 if is_security else 2.0,
                        risk_reduction=8.0 if is_security else 4.0,
                        opportunity_enablement=5.0,
                        job_size=1.0,  # Usually straightforward
                        impact=6.0 if is_security else 4.0,
                        confidence=8.0,
                        ease=9.0,
                        debt_impact=3.0,
                        debt_interest=10.0 if is_security else 2.0,
                        hotspot_multiplier=2.0 if is_security else 0.8,
                        source="dependency_analysis",
                        file_paths=["requirements.txt", "pyproject.toml"],
                        estimated_hours=1.0,
                        discovered_at=datetime.now()
                    )
                    self.discovered_items.append(item)
                    
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Dependency analysis failed: {e}")
            
            # Fallback: Create generic dependency modernization item
            item = ValueItem(
                id="dep-modernization-001",
                title="Modernize dependency management system",
                description="Review and update dependency management, consolidate requirements files, add automated dependency scanning",
                category=TaskCategory.DEPENDENCY,
                user_business_value=5.0,
                time_criticality=3.0,
                risk_reduction=6.0,
                opportunity_enablement=7.0,
                job_size=6.0,
                impact=6.0,
                confidence=7.0,
                ease=5.0,
                debt_impact=12.0,
                debt_interest=8.0,
                hotspot_multiplier=1.3,
                source="dependency_baseline",
                file_paths=["requirements.txt", "pyproject.toml", "requirements-dev.txt"],
                estimated_hours=6.0,
                discovered_at=datetime.now()
            )
            self.discovered_items.append(item)
    
    def _discover_from_performance_analysis(self):
        """Analyze performance opportunities"""
        logger.info("Analyzing performance optimization opportunities...")
        
        # Look for performance-related TODOs in code
        try:
            result = subprocess.run([
                'grep', '-r', '-n', '--include=*.py', 
                'performance\\|optimize\\|slow\\|bottleneck', '.'
            ], capture_output=True, text=True)
            
            matches = result.stdout.strip().split('\n') if result.stdout else []
            
            for match in matches[:5]:  # Limit to top 5 findings
                if ':' in match:
                    file_path, line_num, content = match.split(':', 2)
                    
                    item = ValueItem(
                        id=f"perf-{hash(match) % 10000}",
                        title=f"Performance optimization opportunity",
                        description=f"Performance issue found: {content.strip()[:100]}",
                        category=TaskCategory.PERFORMANCE,
                        user_business_value=6.0,
                        time_criticality=4.0,
                        risk_reduction=3.0,
                        opportunity_enablement=8.0,
                        job_size=4.0,
                        impact=7.0,
                        confidence=6.0,
                        ease=5.0,
                        debt_impact=10.0,
                        debt_interest=8.0,
                        hotspot_multiplier=1.5,
                        source="performance_analysis",
                        file_paths=[file_path],
                        estimated_hours=4.0,
                        discovered_at=datetime.now()
                    )
                    self.discovered_items.append(item)
                    
        except subprocess.CalledProcessError:
            logger.info("No performance optimization opportunities found")
    
    def _discover_from_architectural_analysis(self):
        """Analyze architectural improvement opportunities"""
        logger.info("Analyzing architectural patterns...")
        
        # Look for architectural debt indicators
        patterns = [
            ("circular imports", r"from\s+\w+\s+import.*\n.*from\s+\w+\s+import", "Circular import detected"),
            ("large functions", r"def\s+\w+.*\n(\s+.*\n){50,}", "Large function (>50 lines)"),
            ("deep nesting", r"(\s{12,})", "Deep nesting detected")
        ]
        
        for pattern_name, regex, description in patterns:
            try:
                result = subprocess.run([
                    'grep', '-r', '-E', regex, '--include=*.py', '.'
                ], capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout:
                    matches = result.stdout.strip().split('\n')
                    
                    if len(matches) >= 3:  # Multiple instances indicate architectural debt
                        item = ValueItem(
                            id=f"arch-{pattern_name.replace(' ', '-')}",
                            title=f"Architectural refactoring: {pattern_name}",
                            description=f"Multiple instances of {pattern_name} detected ({len(matches)} files)",
                            category=TaskCategory.TECHNICAL_DEBT,
                            user_business_value=5.0,
                            time_criticality=2.0,
                            risk_reduction=8.0,
                            opportunity_enablement=7.0,
                            job_size=8.0,  # Architectural changes are larger
                            impact=8.0,
                            confidence=7.0,
                            ease=3.0,  # Architectural changes are harder
                            debt_impact=15.0,
                            debt_interest=12.0,
                            hotspot_multiplier=1.8,
                            source="architectural_analysis", 
                            file_paths=[],
                            estimated_hours=8.0,
                            discovered_at=datetime.now()
                        )
                        self.discovered_items.append(item)
                        
            except subprocess.CalledProcessError:
                continue
    
    def _calculate_composite_scores(self):
        """Calculate composite scores for all discovered items"""
        logger.info("Calculating composite scores...")
        
        weights = self.config.get("scoring", {}).get("weights", {}).get("advanced", {})
        wsjf_weight = weights.get("wsjf", 0.5)
        ice_weight = weights.get("ice", 0.1) 
        debt_weight = weights.get("technicalDebt", 0.3)
        security_weight = weights.get("security", 0.1)
        
        for item in self.discovered_items:
            # Calculate WSJF score
            cost_of_delay = (item.user_business_value + item.time_criticality + 
                           item.risk_reduction + item.opportunity_enablement)
            item.wsjf_score = cost_of_delay / item.job_size if item.job_size > 0 else 0
            
            # Calculate ICE score
            item.ice_score = item.impact * item.confidence * item.ease
            
            # Calculate Technical Debt score
            item.technical_debt_score = ((item.debt_impact + item.debt_interest) * 
                                       item.hotspot_multiplier)
            
            # Normalize scores (0-100 scale)
            normalized_wsjf = min(item.wsjf_score * 10, 100)
            normalized_ice = min(item.ice_score / 10, 100)  
            normalized_debt = min(item.technical_debt_score, 100)
            
            # Apply security boost
            security_multiplier = 2.0 if item.category == TaskCategory.SECURITY else 1.0
            
            # Calculate composite score
            item.composite_score = (
                wsjf_weight * normalized_wsjf +
                ice_weight * normalized_ice +
                debt_weight * normalized_debt +
                security_weight * 10  # Base security score
            ) * security_multiplier
    
    def get_next_best_value_item(self) -> Optional[ValueItem]:
        """Get the highest-value item that meets execution criteria"""
        if not self.discovered_items:
            return None
            
        min_score = self.config.get("scoring", {}).get("thresholds", {}).get("minScore", 15.0)
        
        for item in self.discovered_items:
            if item.composite_score and item.composite_score >= min_score:
                return item
                
        return None
    
    def export_backlog(self, output_path: str = "AUTONOMOUS_VALUE_BACKLOG.md"):
        """Export discovered items to markdown backlog"""
        logger.info(f"Exporting backlog to {output_path}")
        
        with open(output_path, 'w') as f:
            f.write("# 📊 Autonomous Value Backlog\n\n")
            f.write(f"Last Updated: {datetime.now().isoformat()}\n")
            f.write(f"Repository: agentic-dev-orchestrator\n")
            f.write(f"Maturity Level: ADVANCED\n\n")
            
            if not self.discovered_items:
                f.write("No value items discovered.\n")
                return
            
            # Next best value item
            next_item = self.get_next_best_value_item()
            if next_item:
                f.write("## 🎯 Next Best Value Item\n\n")
                f.write(f"**[{next_item.id.upper()}] {next_item.title}**\n")
                f.write(f"- **Composite Score**: {next_item.composite_score:.1f}\n")
                f.write(f"- **WSJF**: {next_item.wsjf_score:.1f} | **ICE**: {next_item.ice_score:.1f} | **Tech Debt**: {next_item.technical_debt_score:.1f}\n")
                f.write(f"- **Estimated Effort**: {next_item.estimated_hours} hours\n")
                f.write(f"- **Category**: {next_item.category.value.title()}\n")
                f.write(f"- **Description**: {next_item.description}\n\n")
            
            # Top 10 backlog items
            f.write("## 📋 Top 10 Value Items\n\n")
            f.write("| Rank | ID | Title | Score | Category | Hours |\n")
            f.write("|------|-----|--------|---------|----------|-------|\n")
            
            for i, item in enumerate(self.discovered_items[:10], 1):
                score = f"{item.composite_score:.1f}" if item.composite_score else "N/A"
                f.write(f"| {i} | {item.id.upper()} | {item.title[:40]}... | {score} | {item.category.value.title()} | {item.estimated_hours} |\n")
            
            # Discovery statistics
            f.write("\n## 📈 Discovery Statistics\n\n")
            
            total_items = len(self.discovered_items)
            by_category = {}
            total_hours = 0
            
            for item in self.discovered_items:
                category = item.category.value
                by_category[category] = by_category.get(category, 0) + 1
                total_hours += item.estimated_hours
            
            f.write(f"- **Total Items Discovered**: {total_items}\n")
            f.write(f"- **Total Estimated Effort**: {total_hours:.1f} hours\n")
            f.write(f"- **Average Item Score**: {sum(item.composite_score or 0 for item in self.discovered_items) / max(total_items, 1):.1f}\n\n")
            
            f.write("### Items by Category\n\n")
            for category, count in sorted(by_category.items()):
                f.write(f"- **{category.title()}**: {count} items\n")
            
            f.write("\n### Discovery Sources\n\n")
            sources = {}
            for item in self.discovered_items:
                source = item.source
                sources[source] = sources.get(source, 0) + 1
            
            for source, count in sorted(sources.items()):
                percentage = (count / total_items) * 100
                f.write(f"- **{source.replace('_', ' ').title()}**: {count} items ({percentage:.1f}%)\n")


def main():
    """Main entry point for value discovery"""
    engine = ValueDiscoveryEngine()
    
    # Discover all value items
    items = engine.discover_all_value_items()
    
    if items:
        print(f"\n✅ Discovered {len(items)} value items")
        
        # Show next best item
        next_item = engine.get_next_best_value_item()
        if next_item:
            print(f"\n🎯 Next Best Value Item:")
            print(f"   [{next_item.id.upper()}] {next_item.title}")
            print(f"   Score: {next_item.composite_score:.1f} | Hours: {next_item.estimated_hours}")
            print(f"   Category: {next_item.category.value.title()}")
        
        # Export backlog
        engine.export_backlog()
        print(f"\n📊 Backlog exported to AUTONOMOUS_VALUE_BACKLOG.md")
        
    else:
        print("\n✅ No value items discovered. Repository is in excellent state!")


if __name__ == "__main__":
    main()