#!/usr/bin/env python3
"""
ADO REST API Server
Provides REST endpoints for backlog management and execution monitoring
"""

import os
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from flask import Flask, jsonify, request, abort
    from flask_cors import CORS
except ImportError:
    print("Flask not available - REST API disabled")
    Flask = None

from backlog_manager import BacklogManager, BacklogItem
from autonomous_executor import AutonomousExecutor
from src.database.connection import get_connection
from src.repositories.backlog_repository import BacklogRepository
from src.cache.cache_manager import get_cache_manager
from src.sentiment.analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)


class ADOAPIServer:
    """REST API Server for ADO"""
    
    def __init__(self, repo_root: str = ".", host: str = "0.0.0.0", port: int = 5000):
        if Flask is None:
            raise ImportError("Flask is required for REST API. Install with: pip install flask flask-cors")
        
        self.repo_root = Path(repo_root)
        self.host = host
        self.port = port
        
        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Initialize components
        self.backlog_manager = BacklogManager(str(self.repo_root))
        self.executor = AutonomousExecutor(str(self.repo_root))
        self.connection = get_connection(str(self.repo_root))
        self.repository = BacklogRepository(self.connection)
        self.cache = get_cache_manager(str(self.repo_root / ".ado" / "cache"))
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Setup routes
        self._setup_routes()
        self._setup_error_handlers()
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '0.1.0',
                'service': 'ado-api'
            })
        
        @self.app.route('/api/v1/backlog', methods=['GET'])
        def get_backlog():
            """Get all backlog items"""
            try:
                # Check cache first
                cached = self.cache.get('backlog_items')
                if cached:
                    return jsonify(cached)
                
                items = self.repository.load_all()
                response = {
                    'items': [item.__dict__ for item in items],
                    'total': len(items),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Cache for 5 minutes
                self.cache.set('backlog_items', response, ttl=300)
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Failed to get backlog: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/backlog/<item_id>', methods=['GET'])
        def get_backlog_item(item_id: str):
            """Get specific backlog item"""
            try:
                item = self.repository.load(item_id)
                if not item:
                    return jsonify({'error': 'Item not found'}), 404
                
                return jsonify(item.__dict__)
                
            except Exception as e:
                logger.error(f"Failed to get item {item_id}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/backlog', methods=['POST'])
        def create_backlog_item():
            """Create new backlog item"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
                # Validate required fields
                required_fields = ['id', 'title', 'type', 'description']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing required field: {field}'}), 400
                
                # Set defaults
                data.setdefault('acceptance_criteria', [])
                data.setdefault('effort', 5)
                data.setdefault('value', 5)
                data.setdefault('time_criticality', 3)
                data.setdefault('risk_reduction', 3)
                data.setdefault('status', 'NEW')
                data.setdefault('risk_tier', 'low')
                data.setdefault('created_at', datetime.now().isoformat() + 'Z')
                data.setdefault('links', [])
                data.setdefault('aging_multiplier', 1.0)
                
                # Create item
                item = BacklogItem(**data)
                
                # Check if item already exists
                if self.repository.exists(item.id):
                    return jsonify({'error': 'Item already exists'}), 409
                
                # Save item
                if self.repository.save(item):
                    # Clear cache
                    self.cache.delete('backlog_items')
                    return jsonify(item.__dict__), 201
                else:
                    return jsonify({'error': 'Failed to save item'}), 500
                    
            except Exception as e:
                logger.error(f"Failed to create item: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/backlog/<item_id>', methods=['PUT'])
        def update_backlog_item(item_id: str):
            """Update backlog item"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
                # Load existing item
                item = self.repository.load(item_id)
                if not item:
                    return jsonify({'error': 'Item not found'}), 404
                
                # Update fields
                for field, value in data.items():
                    if hasattr(item, field):
                        setattr(item, field, value)
                
                # Save updated item
                if self.repository.save(item):
                    # Clear cache
                    self.cache.delete('backlog_items')
                    return jsonify(item.__dict__)
                else:
                    return jsonify({'error': 'Failed to update item'}), 500
                    
            except Exception as e:
                logger.error(f"Failed to update item {item_id}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/backlog/<item_id>/status', methods=['PATCH'])
        def update_item_status(item_id: str):
            """Update item status"""
            try:
                data = request.get_json()
                if not data or 'status' not in data:
                    return jsonify({'error': 'Status required'}), 400
                
                new_status = data['status']
                if self.repository.update_status(item_id, new_status):
                    # Clear cache
                    self.cache.delete('backlog_items')
                    return jsonify({'message': 'Status updated', 'item_id': item_id, 'status': new_status})
                else:
                    return jsonify({'error': 'Failed to update status or invalid transition'}), 400
                    
            except Exception as e:
                logger.error(f"Failed to update status for {item_id}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/backlog/<item_id>', methods=['DELETE'])
        def delete_backlog_item(item_id: str):
            """Delete backlog item"""
            try:
                if self.repository.delete(item_id):
                    # Clear cache
                    self.cache.delete('backlog_items')
                    return jsonify({'message': 'Item deleted', 'item_id': item_id})
                else:
                    return jsonify({'error': 'Item not found'}), 404
                    
            except Exception as e:
                logger.error(f"Failed to delete item {item_id}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/backlog/prioritized', methods=['GET'])
        def get_prioritized_backlog():
            """Get backlog sorted by WSJF priority"""
            try:
                items = self.repository.get_prioritized_items()
                return jsonify({
                    'items': [item.__dict__ for item in items],
                    'total': len(items),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Failed to get prioritized backlog: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/backlog/ready', methods=['GET'])
        def get_ready_items():
            """Get items with READY status"""
            try:
                items = self.repository.get_ready_items()
                return jsonify({
                    'items': [item.__dict__ for item in items],
                    'total': len(items),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Failed to get ready items: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/backlog/next', methods=['GET'])
        def get_next_item():
            """Get next highest priority READY item"""
            try:
                item = self.repository.get_next_ready_item()
                if item:
                    return jsonify(item.__dict__)
                else:
                    return jsonify({'message': 'No ready items available'}), 404
                    
            except Exception as e:
                logger.error(f"Failed to get next item: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/backlog/discover', methods=['POST'])
        def discover_items():
            """Discover new backlog items"""
            try:
                self.backlog_manager.load_backlog()
                new_count = self.backlog_manager.continuous_discovery()
                self.backlog_manager.save_backlog()
                
                # Clear cache
                self.cache.delete('backlog_items')
                
                return jsonify({
                    'discovered_count': new_count,
                    'message': f'Discovered {new_count} new items',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Failed to discover items: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/execution/start', methods=['POST'])
        def start_execution():
            """Start autonomous execution"""
            try:
                # Run in background - in production, use celery or similar
                results = self.executor.macro_execution_loop()
                
                return jsonify({
                    'message': 'Execution completed',
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Failed to start execution: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/metrics', methods=['GET'])
        def get_metrics():
            """Get backlog metrics"""
            try:
                # Check cache first
                cached = self.cache.get_backlog_metrics()
                if cached:
                    return jsonify(cached)
                
                metrics = self.repository.get_metrics_summary()
                
                # Cache for 5 minutes
                self.cache.cache_backlog_metrics(metrics, ttl=300)
                
                return jsonify(metrics)
                
            except Exception as e:
                logger.error(f"Failed to get metrics: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/wsjf/update', methods=['POST'])
        def update_wsjf_scores():
            """Update WSJF scores for all items"""
            try:
                updated_count = self.repository.update_wsjf_scores()
                
                # Clear caches
                self.cache.delete('backlog_items')
                self.cache.delete('wsjf_scores')
                
                return jsonify({
                    'updated_count': updated_count,
                    'message': f'Updated WSJF scores for {updated_count} items',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Failed to update WSJF scores: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/cache/clear', methods=['POST'])
        def clear_cache():
            """Clear all caches"""
            try:
                results = self.cache.clear()
                return jsonify({
                    'message': 'Cache cleared',
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Sentiment Analysis Endpoints
        @self.app.route('/api/v1/sentiment/analyze', methods=['POST'])
        def analyze_sentiment():
            """Analyze sentiment of text"""
            try:
                data = request.get_json()
                if not data or 'text' not in data:
                    return jsonify({'error': 'Text required'}), 400
                
                text = data['text']
                metadata = data.get('metadata', {})
                
                result = self.sentiment_analyzer.analyze(text, metadata)
                
                return jsonify(result.to_dict())
                
            except Exception as e:
                logger.error(f"Failed to analyze sentiment: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/sentiment/batch', methods=['POST'])
        def analyze_sentiment_batch():
            """Analyze sentiment of multiple texts"""
            try:
                data = request.get_json()
                if not data or 'texts' not in data:
                    return jsonify({'error': 'Texts array required'}), 400
                
                texts = data['texts']
                if not isinstance(texts, list):
                    return jsonify({'error': 'Texts must be an array'}), 400
                
                results = self.sentiment_analyzer.analyze_batch(texts)
                
                return jsonify({
                    'results': [result.to_dict() for result in results],
                    'total': len(results),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Failed to analyze batch sentiment: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/sentiment/backlog/<item_id>', methods=['POST'])
        def analyze_backlog_sentiment(item_id: str):
            """Analyze sentiment of backlog item"""
            try:
                item = self.repository.load(item_id)
                if not item:
                    return jsonify({'error': 'Item not found'}), 404
                
                # Analyze sentiment of title and description
                title_result = self.sentiment_analyzer.analyze(item.title, {'field': 'title'})
                desc_result = self.sentiment_analyzer.analyze(item.description, {'field': 'description'})
                
                return jsonify({
                    'item_id': item_id,
                    'title_sentiment': title_result.to_dict(),
                    'description_sentiment': desc_result.to_dict(),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Failed to analyze backlog sentiment: {e}")
                return jsonify({'error': str(e)}), 500
    
    def _setup_error_handlers(self):
        """Setup error handlers"""
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Not found'}), 404
        
        @self.app.errorhandler(400)
        def bad_request(error):
            return jsonify({'error': 'Bad request'}), 400
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
    
    def run(self, debug: bool = False):
        """Run the API server"""
        logger.info(f"Starting ADO API server on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=debug)
    
    def get_app(self):
        """Get Flask app instance for testing"""
        return self.app


def create_app(repo_root: str = ".") -> Flask:
    """Factory function to create Flask app"""
    server = ADOAPIServer(repo_root)
    return server.get_app()


if __name__ == "__main__":
    # CLI for running the server
    import argparse
    
    parser = argparse.ArgumentParser(description="ADO REST API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        server = ADOAPIServer(args.repo_root, args.host, args.port)
        server.run(debug=args.debug)
    except ImportError as e:
        print(f"Error: {e}")
        print("Install required dependencies: pip install flask flask-cors")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)