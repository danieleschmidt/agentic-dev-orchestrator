#!/usr/bin/env python3
"""
Autonomous Development Orchestrator (ADO) CLI
Main entry point for the autonomous backlog management system
"""

import sys
import json
from pathlib import Path

from backlog_manager import BacklogManager
from autonomous_executor import AutonomousExecutor


def cmd_init():
    """Initialize ADO in current directory"""
    print("🔧 Initializing Autonomous Development Orchestrator...")
    
    # Create basic directory structure
    dirs_to_create = [
        "backlog",
        "docs/status",
        "escalations"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    # Create sample backlog file if it doesn't exist
    backlog_file = Path("backlog.yml")
    if not backlog_file.exists():
        manager = BacklogManager()
        manager.save_backlog()
        print("✅ Created backlog.yml")
    
    print("🎉 ADO initialization complete!")
    print("\nNext steps:")
    print("1. Set environment variables: GITHUB_TOKEN, OPENAI_API_KEY")
    print("2. Add backlog items to backlog.yml or backlog/*.json")
    print("3. Run: python ado.py run")


def cmd_run():
    """Run the autonomous execution loop"""
    print("🚀 Starting autonomous backlog execution...")
    
    executor = AutonomousExecutor()
    results = executor.macro_execution_loop()
    
    print("\n📊 Execution Summary:")
    print(f"Completed items: {len(results['completed_items'])}")
    print(f"Blocked items: {len(results['blocked_items'])}")
    print(f"Escalated items: {len(results['escalated_items'])}")
    
    # Save results
    results_dir = Path("docs/status")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "last_execution.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to: {results_file}")


def cmd_status():
    """Show current backlog status"""
    manager = BacklogManager()
    manager.load_backlog()
    
    report = manager.generate_status_report()
    
    print("📊 Backlog Status:")
    print(f"Total items: {report['total_items']}")
    print(f"Ready items: {report['ready_items']}")
    
    print("\nStatus breakdown:")
    for status, count in report['backlog_size_by_status'].items():
        print(f"  {status}: {count}")
    
    print("\nTop 3 ready items by WSJF:")
    for item in report['wsjf_snapshot']['top_3_ready']:
        print(f"  {item['id']}: {item['title']} (WSJF: {item['wsjf_score']:.2f})")


def cmd_discover():
    """Run backlog discovery"""
    print("🔍 Running backlog discovery...")
    
    manager = BacklogManager()
    manager.load_backlog()
    
    new_count = manager.continuous_discovery()
    manager.save_backlog()
    
    print(f"📋 Discovered {new_count} new items")
    
    if new_count > 0:
        print("\nRun 'python ado.py status' to see updated backlog")


def cmd_help():
    """Show help information"""
    print("Autonomous Development Orchestrator (ADO)")
    print("=========================================")
    print()
    print("Commands:")
    print("  init      Initialize ADO in current directory")
    print("  run       Execute autonomous backlog processing")
    print("  status    Show current backlog status")
    print("  discover  Discover new backlog items from code")
    print("  help      Show this help message")
    print()
    print("Environment Variables:")
    print("  GITHUB_TOKEN    - GitHub Personal Access Token")
    print("  OPENAI_API_KEY  - OpenAI API key for LLM agents")
    print()
    print("Files:")
    print("  backlog.yml     - Main backlog configuration")
    print("  backlog/*.json  - Individual backlog items")
    print("  docs/status/    - Execution reports and metrics")


def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        cmd_help()
        return
    
    command = sys.argv[1].lower()
    
    commands = {
        'init': cmd_init,
        'run': cmd_run, 
        'status': cmd_status,
        'discover': cmd_discover,
        'help': cmd_help,
        '--help': cmd_help,
        '-h': cmd_help,
    }
    
    if command in commands:
        try:
            commands[command]()
        except KeyboardInterrupt:
            print("\n⏹️  Operation cancelled")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"❌ Unknown command: {command}")
        print("Run 'python ado.py help' for available commands")


if __name__ == "__main__":
    main()