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
import jwt
import hashlib
import time
from functools import wraps

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from flask import Flask, jsonify, request, abort, g
    from flask_cors import CORS
except ImportError:
    print("Flask not available - REST API disabled")
    Flask = None

from backlog_manager import BacklogManager, BacklogItem
from autonomous_executor import AutonomousExecutor
from src.database.connection import get_connection
from src.repositories.backlog_repository import BacklogRepository
from src.cache.cache_manager import get_cache_manager

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
        self.app.config['SECRET_KEY'] = os.environ.get('ADO_SECRET_KEY', 'dev-secret-change-in-production')
        CORS(self.app)
        
        # Initialize components
        self.backlog_manager = BacklogManager(str(self.repo_root))
        self.executor = AutonomousExecutor(str(self.repo_root))
        self.connection = get_connection(str(self.repo_root))
        self.repository = BacklogRepository(self.connection)
        self.cache = get_cache_manager(str(self.repo_root / ".ado" / "cache"))
        
        # Setup middleware, routes, and error handlers
        self._setup_middleware()
        self._setup_routes()
        self._setup_error_handlers()
    
    def _setup_middleware(self):
        """Setup authentication and validation middleware"""
        
        def require_auth(f):
            """Authentication decorator"""
            @wraps(f)
            def decorated(*args, **kwargs):
                # Skip auth for health check and development
                if request.endpoint == 'health_check' or os.environ.get('ADO_DISABLE_AUTH') == 'true':
                    return f(*args, **kwargs)
                
                # Check API key
                api_key = request.headers.get('X-API-Key')
                if not api_key:
                    return jsonify({'error': 'API key required'}), 401
                
                # Validate API key (simple implementation for Gen1)
                expected_key = os.environ.get('ADO_API_KEY')
                if expected_key and api_key != expected_key:
                    return jsonify({'error': 'Invalid API key'}), 401
                
                # Set authenticated user context
                g.authenticated = True
                g.api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:8]
                
                return f(*args, **kwargs)
            return decorated
        
        self.require_auth = require_auth
        
        @self.app.before_request
        def before_request():
            """Pre-request validation and logging"""
            g.request_start_time = time.time()
            g.request_id = hashlib.md5(f"{time.time()}{request.remote_addr}".encode()).hexdigest()[:8]
            
            # Log incoming requests
            logger.info(f"[{g.request_id}] {request.method} {request.path} from {request.remote_addr}")
        
        @self.app.after_request
        def after_request(response):
            """Post-request logging and metrics"""
            duration = time.time() - g.request_start_time
            logger.info(f"[{g.request_id}] Response: {response.status_code} ({duration:.3f}s)")
            
            # Add security headers
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['X-Request-ID'] = g.request_id
            
            return response
    
    def _validate_backlog_item_data(self, data: Dict) -> tuple[bool, str]:
        """Validate backlog item data"""
        required_fields = ['id', 'title', 'type', 'description']
        for field in required_fields:
            if field not in data or not data[field]:
                return False, f'Missing required field: {field}'
        
        # Validate types
        if not isinstance(data.get('effort', 5), int) or data.get('effort', 5) <= 0:
            return False, 'Effort must be a positive integer'
        
        if data.get('type') not in ['feature', 'bug', 'chore', 'tech_debt', 'epic']:
            return False, 'Invalid item type'
        
        if data.get('status', 'NEW') not in ['NEW', 'REFINED', 'READY', 'DOING', 'PR', 'DONE', 'BLOCKED']:
            return False, 'Invalid status'
        
        return True, ''
    
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
        @self.require_auth
        def get_backlog():
            """Get all backlog items"""
            try:
                # Get pagination parameters
                page = request.args.get('page', 1, type=int)
                limit = min(request.args.get('limit', 50, type=int), 100)  # Max 100 items
                status_filter = request.args.get('status')
                
                # Check cache first
                cache_key = f'backlog_items_{page}_{limit}_{status_filter}'
                cached = self.cache.get(cache_key)
                if cached:
                    return jsonify(cached)
                
                items = self.repository.load_all()
                
                # Apply filters
                if status_filter:
                    items = [item for item in items if item.status == status_filter]
                
                # Apply pagination
                total = len(items)
                start_idx = (page - 1) * limit
                end_idx = start_idx + limit
                paginated_items = items[start_idx:end_idx]
                
                response = {
                    'items': [item.__dict__ for item in paginated_items],
                    'total': total,
                    'page': page,
                    'limit': limit,
                    'pages': (total + limit - 1) // limit,
                    'timestamp': datetime.now().isoformat(),
                    'request_id': g.request_id
                }
                
                # Cache for 2 minutes
                self.cache.set(cache_key, response, ttl=120)
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"[{g.request_id}] Failed to get backlog: {e}")
                return jsonify({'error': 'Failed to retrieve backlog', 'request_id': g.request_id}), 500
        
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
        @self.require_auth
        def create_backlog_item():
            """Create new backlog item"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided', 'request_id': g.request_id}), 400
                
                # Validate input data
                is_valid, error_msg = self._validate_backlog_item_data(data)
                if not is_valid:
                    return jsonify({'error': error_msg, 'request_id': g.request_id}), 400
                
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
        @self.require_auth
        def start_execution():
            """Start autonomous execution"""
            try:
                # Get execution parameters
                data = request.get_json() or {}
                max_items = min(data.get('max_items', 5), 10)  # Limit concurrent items
                dry_run = data.get('dry_run', False)
                
                if dry_run:
                    # Simulate execution for testing
                    results = {
                        'mode': 'dry_run',
                        'would_process': max_items,
                        'message': 'Dry run completed successfully'
                    }
                else:
                    # Run actual execution - in production, use celery or similar
                    results = self.executor.macro_execution_loop(max_items=max_items)
                
                return jsonify({
                    'message': 'Execution completed' if not dry_run else 'Dry run completed',
                    'results': results,
                    'timestamp': datetime.now().isoformat(),
                    'request_id': g.request_id
                })
                
            except Exception as e:
                logger.error(f"[{g.request_id}] Failed to start execution: {e}")
                return jsonify({'error': 'Execution failed', 'request_id': g.request_id}), 500
        
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