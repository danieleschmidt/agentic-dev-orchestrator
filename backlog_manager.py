#!/usr/bin/env python3
"""
Autonomous Backlog Management System
Implements WSJF scoring, continuous discovery, and execution orchestration
"""

import json
import re
import os
import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess


@dataclass
class BacklogItem:
    """Normalized backlog item with WSJF scoring"""
    id: str
    title: str
    type: str
    description: str
    acceptance_criteria: List[str]
    effort: int  # 1-2-3-5-8-13 scale
    value: int
    time_criticality: int
    risk_reduction: int
    status: str  # NEW → REFINED → READY → DOING → PR → DONE/BLOCKED
    risk_tier: str  # low/medium/high
    created_at: str
    links: List[str]
    wsjf_score: Optional[float] = None
    aging_multiplier: float = 1.0


class BacklogManager:
    """Core backlog management with WSJF prioritization"""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.backlog_file = self.repo_root / "backlog.yml"
        self.backlog_dir = self.repo_root / "backlog"
        self.status_dir = self.repo_root / "docs" / "status"
        self.items: List[BacklogItem] = []
        self.config = {"name": "ADO Backlog", "version": "1.0"}
        
    def load_backlog(self) -> None:
        """Load backlog from YAML/JSON file and discover items from directory"""
        # First, load configuration from main backlog file
        if self.backlog_file.exists():
            # Try YAML first, fallback to JSON
            try:
                import yaml
                try:
                    with open(self.backlog_file, 'r') as f:
                        data = yaml.safe_load(f) or {}
                except yaml.YAMLError:
                    # If YAML parsing fails, try JSON
                    with open(self.backlog_file, 'r') as f:
                        data = json.load(f)
            except ImportError:
                # YAML not available, try JSON directly
                try:
                    with open(self.backlog_file, 'r') as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}
            
        # Set up config with defaults
        data = data or {
            'version': '1.0',
            'name': 'ADO Backlog',
            'metadata': {
                'created_at': datetime.datetime.now().isoformat() + 'Z',
                'last_updated': datetime.datetime.now().isoformat() + 'Z',
                'scoring_method': 'wsjf',
                'aging_multiplier_max': 2.0,
            },
            'items': [],
            'scoring_weights': {
                'value': 1.0,
                'time_criticality': 1.0,
                'risk_reduction': 1.0,
                'effort_divisor': 1.0,
                'aging_factor': 0.1
            }
        }
            
        self.config = {
            'name': data.get('name', 'ADO Backlog'),
            'version': data.get('version', '1.0'),
            'aging_multiplier_max': data.get('metadata', {}).get('aging_multiplier_max', 2.0),
            'scoring_weights': data.get('scoring_weights', {}),
        }
        
        self.items = []
        
        # Load items from main backlog file
        items_data = data.get('items', []) or data.get('backlog', [])
        for item_data in items_data:
            if isinstance(item_data, dict) and all(key in item_data for key in ['id', 'title', 'type', 'description']):
                # Ensure all required fields exist with defaults
                item_data.setdefault('acceptance_criteria', [])
                item_data.setdefault('effort', 5)
                item_data.setdefault('value', 5)
                item_data.setdefault('time_criticality', 3)
                item_data.setdefault('risk_reduction', 3)
                item_data.setdefault('status', 'NEW')
                item_data.setdefault('risk_tier', 'low')
                item_data.setdefault('created_at', '2025-01-27T00:00:00Z')
                item_data.setdefault('links', [])
                item_data.setdefault('wsjf_score', None)
                item_data.setdefault('aging_multiplier', 1.0)
                
                item = BacklogItem(**item_data)
                self.items.append(item)
        
        # Also discover items from JSON files in backlog directory
        discovered_items = self.discover_from_json_files()
        self.merge_and_dedupe_items(discovered_items)
    
    def save_backlog(self) -> None:
        """Save backlog to YAML file (with fallback to JSON)"""
        data = {
            **self.config,
            'metadata': {
                'created_at': datetime.datetime.now().isoformat() + 'Z',
                'last_updated': datetime.datetime.now().isoformat() + 'Z',
                'scoring_method': 'wsjf',
                'aging_multiplier_max': self.config.get('aging_multiplier_max', 2.0),
            },
            'items': [asdict(item) for item in self.items],
            'scoring_weights': self.config.get('scoring_weights', {
                'value': 1.0,
                'time_criticality': 1.0,
                'risk_reduction': 1.0,
                'effort_divisor': 1.0,
                'aging_factor': 0.1
            })
        }
        
        # Try YAML first, fallback to JSON
        try:
            import yaml
            with open(self.backlog_file, 'w') as f:
                yaml.dump(data, f, indent=2)
        except ImportError:
            # Fallback to JSON if YAML not available
            with open(self.backlog_file, 'w') as f:
                json.dump(data, f, indent=2)
    
    def discover_from_json_files(self) -> List[BacklogItem]:
        """Discover backlog items from backlog/*.json files"""
        items = []
        if not self.backlog_dir.exists():
            return items
            
        for json_file in self.backlog_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Check if it's already in the correct format
                if all(key in data for key in ['id', 'title', 'type', 'description']):
                    # Ensure all required fields exist with defaults
                    data.setdefault('acceptance_criteria', [])
                    data.setdefault('effort', 5)
                    data.setdefault('value', 5)
                    data.setdefault('time_criticality', 3)
                    data.setdefault('risk_reduction', 3)
                    data.setdefault('status', 'NEW')
                    data.setdefault('risk_tier', 'low')
                    data.setdefault('created_at', datetime.datetime.now().isoformat() + 'Z')
                    data.setdefault('links', [])
                    data.setdefault('wsjf_score', None)
                    data.setdefault('aging_multiplier', 1.0)
                    
                    item = BacklogItem(**data)
                    items.append(item)
                else:
                    # Convert from README schema to normalized schema
                    wsjf_data = data.get('wsjf', {})
                    item = BacklogItem(
                        id=f"json-{json_file.stem}",
                        title=data.get('title', ''),
                        type='feature',
                        description=data.get('description', ''),
                        acceptance_criteria=data.get('acceptance_criteria', []),
                        effort=wsjf_data.get('job_size', 5),
                        value=wsjf_data.get('user_business_value', 5),
                        time_criticality=wsjf_data.get('time_criticality', 3),
                        risk_reduction=wsjf_data.get('risk_reduction_opportunity_enablement', 3),
                        status='NEW',
                        risk_tier='low',
                        created_at=datetime.datetime.now().isoformat() + 'Z',
                        links=[]
                    )
                    items.append(item)
            except Exception as e:
                print(f"Error parsing {json_file}: {e}")
                
        return items
    
    def discover_from_code_comments(self) -> List[BacklogItem]:
        """Discover TODO/FIXME/HACK/BUG comments in codebase"""
        items = []
        patterns = [
            r'#\s*TODO:?\s*(.+)',
            r'#\s*FIXME:?\s*(.+)',
            r'#\s*HACK:?\s*(.+)',  
            r'#\s*BUG:?\s*(.+)',
            r'//\s*TODO:?\s*(.+)',
            r'//\s*FIXME:?\s*(.+)',
            r'/\*\s*TODO:?\s*(.+?)\s*\*/',
        ]
        
        try:
            # Use grep to find TODO/FIXME comments
            result = subprocess.run(
                ['grep', '-rn', '-E', 'TODO|FIXME|HACK|BUG', '.'],
                cwd=self.repo_root,
                capture_output=True, 
                text=True
            )
            
            # Process grep output even if return code != 0 (no matches)
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                seen_comments = set()
                
                for line in lines:
                    if ':' not in line:
                        continue
                        
                    parts = line.split(':', 2)
                    if len(parts) < 3:
                        continue
                        
                    file_path = parts[0]
                    comment_text = parts[2].strip()
                    
                    # Skip binary files, git directories, etc.
                    if '.git/' in file_path or file_path.endswith('.pyc'):
                        continue
                    
                    # Extract actual comment text
                    for pattern in patterns:
                        match = re.search(pattern, comment_text, re.IGNORECASE)
                        if match:
                            comment_content = match.group(1).strip()
                            if len(comment_content) < 10:  # Skip trivial comments
                                continue
                            
                            # Deduplicate by content
                            comment_key = f"{file_path}:{comment_content}"
                            if comment_key in seen_comments:
                                continue
                            seen_comments.add(comment_key)
                                
                            item_id = f"code-{abs(hash(comment_key)) % 10000:04d}"
                            item = BacklogItem(
                                id=item_id,
                                title=f"Address code comment: {comment_content[:50]}{'...' if len(comment_content) > 50 else ''}",
                                type='tech_debt',
                                description=f"Found in {file_path}: {comment_content}",
                                acceptance_criteria=[f"Resolve comment in {file_path}"],
                                effort=2,  # Default small effort
                                value=2,
                                time_criticality=1,
                                risk_reduction=3,
                                status='NEW',
                                risk_tier='low',
                                created_at=datetime.datetime.now().isoformat() + 'Z',
                                links=[f"file://{file_path}"]
                            )
                            items.append(item)
                            break  # Only match first pattern per line
                    
        except Exception as e:
            # Fallback: return empty list if grep fails
            pass
            
        return items
    
    def calculate_wsjf_scores(self) -> None:
        """Calculate WSJF scores with aging multiplier"""
        now = datetime.datetime.now()
        
        for item in self.items:
            # Calculate base WSJF score
            cost_of_delay = item.value + item.time_criticality + item.risk_reduction
            base_score = cost_of_delay / max(item.effort, 1)
            
            # Apply aging multiplier
            try:
                created = datetime.datetime.fromisoformat(item.created_at.replace('Z', '+00:00'))
                age_days = (now - created).days
                aging_factor = min(1 + (age_days * 0.01), self.config.get('aging_multiplier_max', 2.0))
                item.aging_multiplier = aging_factor
            except:
                item.aging_multiplier = 1.0
                
            item.wsjf_score = base_score * item.aging_multiplier
    
    def get_prioritized_backlog(self) -> List[BacklogItem]:
        """Get backlog sorted by WSJF score (highest first)"""
        self.calculate_wsjf_scores()
        return sorted(self.items, key=lambda x: x.wsjf_score or 0, reverse=True)
    
    def get_next_ready_item(self) -> Optional[BacklogItem]:
        """Get next READY item with highest WSJF score"""
        ready_items = [item for item in self.items if item.status == 'READY']
        if not ready_items:
            return None
            
        # Calculate scores for ready items
        for item in ready_items:
            cost_of_delay = item.value + item.time_criticality + item.risk_reduction
            item.wsjf_score = cost_of_delay / max(item.effort, 1) * item.aging_multiplier
            
        return max(ready_items, key=lambda x: x.wsjf_score or 0)
    
    def merge_and_dedupe_items(self, new_items: List[BacklogItem]) -> None:
        """Merge new items with existing, handling deduplication"""
        existing_ids = {item.id for item in self.items}
        existing_titles = {item.title.lower().strip() for item in self.items}
        
        for new_item in new_items:
            # Skip if ID already exists
            if new_item.id in existing_ids:
                continue
                
            # Check for similar titles (simple deduplication)
            if new_item.title.lower().strip() in existing_titles:
                continue
                
            self.items.append(new_item)
    
    def continuous_discovery(self) -> int:
        """Run continuous discovery from all sources"""
        new_items = []
        
        # Discover from JSON files
        new_items.extend(self.discover_from_json_files())
        
        # Discover from code comments (use _discover_from_code for test compatibility)
        code_items_data = self._discover_from_code()
        for item_data in code_items_data:
            if isinstance(item_data, dict):
                # Ensure all required fields exist with defaults
                item_data.setdefault('acceptance_criteria', [])
                item_data.setdefault('effort', 5)
                item_data.setdefault('value', 5)
                item_data.setdefault('time_criticality', 3)
                item_data.setdefault('risk_reduction', 3)
                item_data.setdefault('status', 'NEW')
                item_data.setdefault('risk_tier', 'low')
                item_data.setdefault('created_at', datetime.datetime.now().isoformat() + 'Z')
                item_data.setdefault('links', [])
                item_data.setdefault('wsjf_score', None)
                item_data.setdefault('aging_multiplier', 1.0)
                
                item = BacklogItem(**item_data)
                new_items.append(item)
        
        # Merge with existing items
        initial_count = len(self.items)
        self.merge_and_dedupe_items(new_items)
        
        return len(self.items) - initial_count
    
    def update_item_status_by_id(self, item_id: str, new_status: str) -> bool:
        """Update status of a backlog item by ID"""
        valid_statuses = ['NEW', 'REFINED', 'READY', 'DOING', 'PR', 'DONE', 'BLOCKED']
        if new_status not in valid_statuses:
            return False
            
        for item in self.items:
            if item.id == item_id:
                if self.is_valid_transition(item.status, new_status):
                    item.status = new_status
                    return True
                return False
        return False
    
    def generate_status_report(self) -> Dict:
        """Generate status report for metrics"""
        status_counts = {}
        for item in self.items:
            status_counts[item.status] = status_counts.get(item.status, 0) + 1
            
        completed_today = [
            item for item in self.items 
            if item.status == 'DONE' and 
            item.created_at.startswith(datetime.date.today().isoformat())
        ]
        
        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'backlog_size_by_status': status_counts,
            'completed_ids': [item.id for item in completed_today],
            'total_items': len(self.items),
            'ready_items': len([item for item in self.items if item.status == 'READY']),
            'wsjf_snapshot': {
                'top_3_ready': [
                    {'id': item.id, 'title': item.title, 'wsjf_score': item.wsjf_score}
                    for item in self.get_prioritized_backlog()[:3]
                    if item.status == 'READY'
                ]
            }
        }
    
    def save_status_report(self) -> None:
        """Save status report to docs/status/"""
        self.status_dir.mkdir(parents=True, exist_ok=True)
        
        report = self.generate_status_report()
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_file = self.status_dir / f"status_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Save latest report
        latest_file = self.status_dir / "latest.json"
        with open(latest_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    def calculate_wsjf(self, item: BacklogItem) -> float:
        """Calculate WSJF score for a single item"""
        if item.effort == 0:
            return float('inf')  # Handle division by zero
        
        cost_of_delay = item.value + item.time_criticality + item.risk_reduction
        base_score = cost_of_delay / item.effort
        return base_score * item.aging_multiplier
    
    def deduplicate_items(self, items: List[Dict]) -> List[Dict]:
        """Deduplicate items by title and description"""
        seen = set()
        deduplicated = []
        
        for item in items:
            key = (item.get('title', ''), item.get('description', ''))
            if key not in seen:
                seen.add(key)
                deduplicated.append(item)
        
        return deduplicated
    
    def is_valid_transition(self, from_status: str, to_status: str) -> bool:
        """Check if status transition is valid"""
        valid_transitions = {
            'NEW': ['REFINED', 'BLOCKED'],
            'REFINED': ['READY', 'NEW', 'BLOCKED'],
            'READY': ['DOING', 'REFINED', 'BLOCKED'],
            'DOING': ['PR', 'READY', 'BLOCKED'],
            'PR': ['DONE', 'DOING', 'BLOCKED'],
            'DONE': [],  # Terminal state
            'BLOCKED': ['NEW', 'REFINED', 'READY', 'DOING', 'PR']  # Can return to any state
        }
        
        return to_status in valid_transitions.get(from_status, [])
    
    def update_item_status(self, item: BacklogItem, new_status: str) -> bool:
        """Update item status with validation"""
        if not self.is_valid_transition(item.status, new_status):
            return False
        
        item.status = new_status
        return True
    
    def get_prioritized_items(self) -> List[BacklogItem]:
        """Alias for get_prioritized_backlog for test compatibility"""
        return self.get_prioritized_backlog()
    
    def get_ready_items(self) -> List[BacklogItem]:
        """Get all items with READY status"""
        return [item for item in self.items if item.status == 'READY']
    
    @staticmethod
    def _calculate_wsjf_score(user_business_value: int, time_criticality: int, 
                             risk_reduction: int, job_size: int) -> float:
        """Calculate WSJF score from components"""
        if job_size == 0:
            return float('inf')
        cost_of_delay = user_business_value + time_criticality + risk_reduction
        return cost_of_delay / job_size
    
    def _discover_from_code(self) -> List[Dict]:
        """Discover items from code and return as dicts for test compatibility"""
        items = self.discover_from_code_comments()
        return [asdict(item) for item in items]
    
    def is_git_clean(self) -> bool:
        """Check if git working directory is clean"""
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.repo_root,
                capture_output=True,
                text=True
            )
            return result.returncode == 0 and not result.stdout.strip()
        except Exception:
            return False
    
    def create_commit(self, message: str) -> bool:
        """Create git commit with message"""
        try:
            # Add all changes
            add_result = subprocess.run(
                ['git', 'add', '.'],
                cwd=self.repo_root,
                capture_output=True
            )
            if add_result.returncode != 0:
                return False
            
            # Commit changes
            commit_result = subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=self.repo_root,
                capture_output=True
            )
            return commit_result.returncode == 0
        except Exception:
            return False


def main():
    """CLI entry point for backlog management"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python backlog_manager.py <command>")
        print("Commands: discover, score, status, next")
        return
        
    command = sys.argv[1]
    manager = BacklogManager()
    
    if command == "discover":
        manager.load_backlog()
        new_count = manager.continuous_discovery()
        manager.save_backlog()
        print(f"Discovered {new_count} new items")
        
    elif command == "score":
        manager.load_backlog()
        manager.calculate_wsjf_scores()
        manager.save_backlog()
        print("WSJF scores calculated")
        
    elif command == "status":
        manager.load_backlog()
        report = manager.generate_status_report()
        print(json.dumps(report, indent=2))
        
    elif command == "next":
        manager.load_backlog()
        next_item = manager.get_next_ready_item()
        if next_item:
            print(f"Next item: {next_item.id} - {next_item.title}")
            print(f"WSJF Score: {next_item.wsjf_score:.2f}")
        else:
            print("No READY items in backlog")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()