#!/usr/bin/env python3
"""
Autonomous Development Orchestrator (ADO) CLI v4.0
Main entry point for the autonomous backlog management system
Enhanced with global-first internationalization support
"""

import sys
import json
import os
import datetime
from pathlib import Path

from backlog_manager import BacklogManager
from autonomous_executor import AutonomousExecutor

# Enhanced internationalization support
try:
    from src.globalization.i18n_manager import i18n, _, _n, SupportedLocale, set_locale
    I18N_AVAILABLE = True
    
    # Auto-detect system locale
    import locale as system_locale
    try:
        system_lang = system_locale.getdefaultlocale()[0]
        if system_lang:
            set_locale(system_lang)
    except Exception:
        pass  # Use default locale
        
except ImportError:
    I18N_AVAILABLE = False
    # Fallback translation functions
    def _(key: str, **kwargs) -> str:
        return key.split('.')[-1].replace('_', ' ').title()
    
    def _n(key: str, count: int, **kwargs) -> str:
        return _(key, **kwargs)


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
    """Run the autonomous execution loop with enhanced error handling"""
    print(f"🤖 {_('cli.welcome')}")
    print(f"🚀 {_('cli.starting_execution')}")
    
    # Validate environment setup
    required_env = ['GITHUB_TOKEN', 'OPENAI_API_KEY']
    missing_env = [var for var in required_env if not os.getenv(var)]
    if missing_env:
        print(f"❌ {_('cli.missing_env_vars', vars=', '.join(missing_env))}")
        print(_('cli.please_set_variables'))
        return
    
    # Initialize components with error handling
    components = {}
    
    # Initialize sentiment analysis
    try:
        from src.ai.sentiment_analyzer import SentimentAnalyzer
        components['sentiment'] = SentimentAnalyzer()
        print("✨ Sentiment analysis engine initialized")
    except ImportError as e:
        print(f"⚠️  Sentiment analysis unavailable: {e}")
        components['sentiment'] = None
    
    # Initialize adaptive learning
    try:
        from src.intelligence.adaptive_learning import AdaptiveLearningEngine
        components['learning'] = AdaptiveLearningEngine()
        print("🧠 Adaptive learning engine initialized")
    except ImportError as e:
        print(f"⚠️  Learning engine unavailable: {e}")
        components['learning'] = None
    
    # Initialize performance monitoring
    try:
        from src.performance.metrics_collector import MetricsCollector
        components['metrics'] = MetricsCollector()
        print("📊 Performance monitoring initialized")
    except ImportError:
        components['metrics'] = None
    
    try:
        executor = AutonomousExecutor()
        
        # Run pre-execution analysis
        if components['sentiment']:
            print("🔍 Analyzing team sentiment from backlog...")
            
        if components['learning']:
            print("📊 Loading historical insights...")
            try:
                report = components['learning'].generate_learning_report()
                if report and 'high_confidence_insights' in report and report['high_confidence_insights'] > 0:
                    print(f"💡 Found {report['high_confidence_insights']} high-confidence insights")
            except Exception as e:
                print(f"⚠️  Learning report generation failed: {e}")
        
        print("\n🏗️  Starting 3-generation progressive enhancement...")
        results = executor.macro_execution_loop()
        
        # Enhanced execution summary
        print(f"\n📊 {_('cli.execution_summary')}")
        completed_count = len(results.get('completed_items', []))
        print(f"✅ {_('cli.completed_items', count=completed_count)}")
        print(f"🔄 {_('cli.in_progress_items', count=len(results.get('in_progress_items', [])))}")
        print(f"🚫 {_('cli.blocked_items', count=len(results.get('blocked_items', [])))}")
        print(f"🔺 {_('cli.escalated_items', count=len(results.get('escalated_items', [])))}")
        
        # Performance metrics
        if components['metrics']:
            try:
                metrics = components['metrics'].collect_metrics()
                print(f"⏱️  Average execution time: {metrics.get('avg_execution_time', 'N/A')}s")
                print(f"💾 Memory usage: {metrics.get('memory_usage_mb', 'N/A')} MB")
            except Exception as e:
                print(f"⚠️  Metrics collection failed: {e}")
        
        # Save enhanced results
        results_dir = Path("docs/status")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"execution_{timestamp}.json"
        latest_file = results_dir / "latest.json"
        
        enhanced_results = {
            **results,
            'execution_timestamp': timestamp,
            'environment_status': {var: 'SET' if os.getenv(var) else 'MISSING' for var in required_env},
            'components_loaded': {k: v is not None for k, v in components.items()}
        }
        
        with open(results_file, 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        with open(latest_file, 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        print(f"📄 Results saved to: {results_file}")
        print(f"📄 Latest results: {latest_file}")
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        # Save error report
        error_report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'error': str(e),
            'components_status': {k: v is not None for k, v in components.items()}
        }
        
        error_file = Path("docs/status/error_report.json")
        error_file.parent.mkdir(parents=True, exist_ok=True)
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        print(f"🔍 Error report saved to: {error_file}")
        raise


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


def cmd_validate():
    """Validate ADO environment and configuration"""
    print("🔍 Validating ADO Environment...")
    
    issues = []
    
    # Check environment variables
    required_env = ['GITHUB_TOKEN', 'OPENAI_API_KEY']
    for var in required_env:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"❌ {var} is not set")
            issues.append(f"Missing environment variable: {var}")
    
    # Check directory structure
    required_dirs = [
        "backlog",
        "docs/status", 
        "escalations"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"⚠️  Directory missing: {dir_path}")
            issues.append(f"Missing directory: {dir_path}")
    
    # Check backlog files
    backlog_yml = Path("backlog.yml")
    if backlog_yml.exists():
        print("✅ backlog.yml exists")
    else:
        print("⚠️  backlog.yml not found")
        issues.append("Missing backlog.yml")
    
    # Check for backlog items
    backlog_dir = Path("backlog")
    if backlog_dir.exists():
        json_files = list(backlog_dir.glob("*.json"))
        if json_files:
            print(f"✅ Found {len(json_files)} backlog JSON files")
        else:
            print("⚠️  No backlog JSON files found")
            issues.append("No backlog items found")
    
    # Check Python dependencies
    try:
        import yaml
        print("✅ PyYAML available")
    except ImportError:
        print("❌ PyYAML not available")
        issues.append("Missing dependency: PyYAML")
    
    try:
        import requests
        print("✅ Requests available")
    except ImportError:
        print("❌ Requests not available")  
        issues.append("Missing dependency: requests")
    
    if not issues:
        print("\n🎉 Environment validation passed!")
        return True
    else:
        print("\n⚠️  Validation issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False


def cmd_metrics():
    """Show performance and execution metrics"""
    print("📊 ADO Performance Metrics")
    print("=" * 40)
    
    # Load latest execution results
    latest_file = Path("docs/status/latest.json")
    if latest_file.exists():
        with open(latest_file) as f:
            results = json.load(f)
        
        print(f"📅 Last execution: {results.get('execution_timestamp', 'N/A')}")
        print(f"✅ Completed: {len(results.get('completed_items', []))}")
        print(f"🔄 In progress: {len(results.get('in_progress_items', []))}")
        print(f"🚫 Blocked: {len(results.get('blocked_items', []))}")
        print(f"🔺 Escalated: {len(results.get('escalated_items', []))}")
        
        components = results.get('components_loaded', {})
        print("\n🔧 Components Status:")
        for comp, loaded in components.items():
            status = "✅ Active" if loaded else "❌ Inactive"
            print(f"  {comp}: {status}")
    else:
        print("⚠️  No execution data found. Run 'ado run' first.")
    
    # Show historical data if available
    status_dir = Path("docs/status")
    if status_dir.exists():
        execution_files = list(status_dir.glob("execution_*.json"))
        if execution_files:
            print(f"\n📈 Historical executions: {len(execution_files)}")
            recent_files = sorted(execution_files, key=lambda x: x.stat().st_mtime)[-5:]
            print("Recent executions:")
            for file in recent_files:
                mtime = datetime.datetime.fromtimestamp(file.stat().st_mtime)
                print(f"  - {file.name}: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")


def cmd_help():
    """Show help information"""
    print("Terragon ADO v4.0 - Autonomous Development Orchestrator")
    print("=" * 60)
    print()
    print("Commands:")
    print("  init       Initialize ADO in current directory")
    print("  run        Execute autonomous backlog processing")
    print("  status     Show current backlog status")
    print("  discover   Discover new backlog items from code")
    print("  validate   Validate environment and configuration")
    print("  metrics    Show performance and execution metrics")  
    print("  help       Show this help message")
    print()
    print("Environment Variables:")
    print("  GITHUB_TOKEN    - GitHub Personal Access Token (required)")
    print("  OPENAI_API_KEY  - OpenAI API key for LLM agents (required)")
    print("  ADO_LOG_LEVEL   - Logging level (DEBUG, INFO, WARN, ERROR)")
    print("  ADO_CONFIG_PATH - Custom configuration file path")
    print()
    print("Files & Directories:")
    print("  backlog.yml       - Main backlog configuration")
    print("  backlog/*.json    - Individual backlog items")  
    print("  docs/status/      - Execution reports and metrics")
    print("  escalations/      - Human intervention logs")
    print()
    print("Examples:")
    print("  ado init          # Setup new project")
    print("  ado validate      # Check environment")
    print("  ado run           # Start autonomous execution")
    print("  ado status        # View backlog status")
    print("  ado metrics       # View performance data")


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
        'validate': cmd_validate,
        'metrics': cmd_metrics,
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