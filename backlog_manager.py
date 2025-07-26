#!/usr/bin/env python3
"""
Autonomous Backlog Management System
Implements WSJF scoring, continuous discovery, and execution orchestration
"""

import yaml
import json
import re
import os
import datetime
from typing import Dict, List, Optional, Tuple
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
        self.config = {}
        
    def load_backlog(self) -> None:
        """Load backlog from YAML file"""
        if not self.backlog_file.exists():
            self.items = []
            return
            
        with open(self.backlog_file, 'r') as f:
            data = yaml.safe_load(f)
            
        self.config = {
            'aging_multiplier_max': data.get('metadata', {}).get('aging_multiplier_max', 2.0),
            'scoring_weights': data.get('scoring_weights', {}),
        }
        
        self.items = []
        for item_data in data.get('items', []):
            item = BacklogItem(**item_data)
            self.items.append(item)
    
    def save_backlog(self) -> None:
        """Save backlog to YAML file"""
        data = {
            'version': '1.0',
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
        
        with open(self.backlog_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def discover_from_json_files(self) -> List[BacklogItem]:
        """Discover backlog items from backlog/*.json files"""
        items = []
        if not self.backlog_dir.exists():
            return items
            
        for json_file in self.backlog_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
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
            r'# TODO:?\s*(.+)',
            r'# FIXME:?\s*(.+)',
            r'# HACK:?\s*(.+)',  
            r'# BUG:?\s*(.+)',
            r'//\s*TODO:?\s*(.+)',
            r'//\s*FIXME:?\s*(.+)',
            r'/\*\s*TODO:?\s*(.+?)\s*\*/',
        ]
        
        try:
            # Use git to find tracked files to avoid .git directories
            result = subprocess.run(
                ['git', 'ls-files'], 
                cwd=self.repo_root,
                capture_output=True, 
                text=True
            )
            if result.returncode != 0:
                return items
                
            files = result.stdout.strip().split('\n')
            
            for file_path in files:
                full_path = self.repo_root / file_path
                if not full_path.is_file():
                    continue
                    
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    for pattern in patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                        for match in matches:
                            comment_text = match.group(1).strip()
                            if len(comment_text) < 10:  # Skip trivial comments
                                continue
                                
                            item_id = f"code-{hash(f'{file_path}:{comment_text}') % 10000:04d}"
                            item = BacklogItem(
                                id=item_id,
                                title=f"Address code comment: {comment_text[:50]}...",
                                type='tech_debt',
                                description=f"Found in {file_path}: {comment_text}",
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
                            
                except Exception as e:
                    continue  # Skip files that can't be read
                    
        except Exception as e:
            print(f"Error discovering code comments: {e}")
            
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
        
        # Discover from code comments
        new_items.extend(self.discover_from_code_comments())
        
        # Merge with existing items
        initial_count = len(self.items)
        self.merge_and_dedupe_items(new_items)
        
        return len(self.items) - initial_count
    
    def update_item_status(self, item_id: str, new_status: str) -> bool:
        """Update status of a backlog item"""
        valid_statuses = ['NEW', 'REFINED', 'READY', 'DOING', 'PR', 'DONE', 'BLOCKED']
        if new_status not in valid_statuses:
            return False
            
        for item in self.items:
            if item.id == item_id:
                item.status = new_status
                return True
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