#!/usr/bin/env python3
"""
Resource pooling and connection management for scalable operations
"""

import threading
import time
import subprocess
import concurrent.futures
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from queue import Queue, Empty
import logging
from contextlib import contextmanager


@dataclass
class PooledResource:
    """Represents a pooled resource"""
    resource_id: str
    resource: Any
    created_at: float
    last_used: float
    usage_count: int = 0
    is_healthy: bool = True


class ResourcePool:
    """Generic resource pool with health checking and auto-cleanup"""
    
    def __init__(self, factory: Callable, max_size: int = 10, 
                 max_idle_time: float = 300, health_check: Optional[Callable] = None):
        self.factory = factory
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.health_check = health_check
        self.pool: Queue[PooledResource] = Queue(maxsize=max_size)
        self.active_resources: Dict[str, PooledResource] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger("resource_pool")
        self.stats = {
            "created": 0,
            "borrowed": 0,
            "returned": 0,
            "destroyed": 0,
            "health_check_failures": 0
        }
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    @contextmanager
    def acquire(self):
        """Context manager for acquiring and releasing resources"""
        resource = self.borrow()
        try:
            yield resource.resource
        finally:
            self.return_resource(resource)
    
    def borrow(self) -> PooledResource:
        """Borrow a resource from the pool"""
        with self.lock:
            # Try to get existing resource
            try:
                resource = self.pool.get_nowait()
                if self._is_resource_healthy(resource):
                    resource.last_used = time.time()
                    resource.usage_count += 1
                    self.active_resources[resource.resource_id] = resource
                    self.stats["borrowed"] += 1
                    return resource
                else:
                    # Resource is unhealthy, destroy it
                    self._destroy_resource(resource)
            except Empty:
                pass
            
            # Create new resource if pool is not at max capacity
            if len(self.active_resources) < self.max_size:
                resource = self._create_resource()
                self.active_resources[resource.resource_id] = resource
                self.stats["borrowed"] += 1
                return resource
            
            # Wait for resource to become available
            self.logger.warning("Resource pool at capacity, waiting...")
            raise Exception("Resource pool exhausted")
    
    def return_resource(self, resource: PooledResource):
        """Return a resource to the pool"""
        with self.lock:
            if resource.resource_id in self.active_resources:
                del self.active_resources[resource.resource_id]
                
                if self._is_resource_healthy(resource):
                    try:
                        self.pool.put_nowait(resource)
                        self.stats["returned"] += 1
                    except:
                        # Pool is full, destroy excess resource
                        self._destroy_resource(resource)
                else:
                    self._destroy_resource(resource)
    
    def _create_resource(self) -> PooledResource:
        """Create a new pooled resource"""
        resource_id = f"resource_{int(time.time() * 1000)}_{threading.get_ident()}"
        resource = PooledResource(
            resource_id=resource_id,
            resource=self.factory(),
            created_at=time.time(),
            last_used=time.time()
        )
        self.stats["created"] += 1
        self.logger.debug(f"Created resource {resource_id}")
        return resource
    
    def _is_resource_healthy(self, resource: PooledResource) -> bool:
        """Check if resource is healthy"""
        try:
            # Check age
            if time.time() - resource.created_at > self.max_idle_time * 2:
                return False
            
            # Run health check if provided
            if self.health_check:
                is_healthy = self.health_check(resource.resource)
                if not is_healthy:
                    self.stats["health_check_failures"] += 1
                return is_healthy
            
            return resource.is_healthy
            
        except Exception as e:
            self.logger.warning(f"Health check failed for {resource.resource_id}: {e}")
            self.stats["health_check_failures"] += 1
            return False
    
    def _destroy_resource(self, resource: PooledResource):
        """Destroy a resource"""
        try:
            if hasattr(resource.resource, 'close'):
                resource.resource.close()
            elif hasattr(resource.resource, 'cleanup'):
                resource.resource.cleanup()
        except Exception as e:
            self.logger.warning(f"Error destroying resource {resource.resource_id}: {e}")
        
        self.stats["destroyed"] += 1
        self.logger.debug(f"Destroyed resource {resource.resource_id}")
    
    def _cleanup_loop(self):
        """Background cleanup of idle resources"""
        while True:
            try:
                with self.lock:
                    current_time = time.time()
                    resources_to_cleanup = []
                    
                    # Check pool for idle resources
                    temp_resources = []
                    while not self.pool.empty():
                        try:
                            resource = self.pool.get_nowait()
                            if current_time - resource.last_used > self.max_idle_time:
                                resources_to_cleanup.append(resource)
                            else:
                                temp_resources.append(resource)
                        except Empty:
                            break
                    
                    # Put back non-idle resources
                    for resource in temp_resources:
                        try:
                            self.pool.put_nowait(resource)
                        except:
                            resources_to_cleanup.append(resource)
                    
                    # Cleanup idle resources
                    for resource in resources_to_cleanup:
                        self._destroy_resource(resource)
                
                time.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                time.sleep(60)
    
    def get_stats(self) -> Dict:
        """Get pool statistics"""
        with self.lock:
            return {
                **self.stats,
                "pool_size": self.pool.qsize(),
                "active_resources": len(self.active_resources),
                "max_size": self.max_size,
                "timestamp": time.time()
            }


class ProcessPool:
    """Process pool for executing subprocess operations safely"""
    
    def __init__(self, max_processes: int = 5):
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_processes)
        self.logger = logging.getLogger("process_pool")
        self.active_processes = 0
        self.max_processes = max_processes
        self.lock = threading.Lock()
    
    async def execute_command(self, command: List[str], cwd: str = None, 
                            timeout: int = 30) -> subprocess.CompletedProcess:
        """Execute command in process pool"""
        with self.lock:
            if self.active_processes >= self.max_processes:
                raise Exception("Process pool at capacity")
            self.active_processes += 1
        
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                self.executor,
                self._run_command,
                command, cwd, timeout
            )
            return result
            
        finally:
            with self.lock:
                self.active_processes -= 1
    
    def _run_command(self, command: List[str], cwd: str, timeout: int) -> subprocess.CompletedProcess:
        """Run command synchronously"""
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            return result
        except subprocess.TimeoutExpired as e:
            self.logger.error(f"Command timed out: {' '.join(command)}")
            raise e
        except Exception as e:
            self.logger.error(f"Command failed: {' '.join(command)}: {e}")
            raise e
    
    def shutdown(self):
        """Shutdown process pool"""
        self.executor.shutdown(wait=True)
        self.logger.info("Process pool shutdown complete")


class ConnectionPool:
    """Connection pool for database or network connections"""
    
    def __init__(self, connection_factory: Callable, max_connections: int = 20):
        self.connection_factory = connection_factory
        self.pool = ResourcePool(
            factory=connection_factory,
            max_size=max_connections,
            health_check=self._check_connection_health
        )
        self.logger = logging.getLogger("connection_pool")
    
    def _check_connection_health(self, connection) -> bool:
        """Check if connection is healthy"""
        try:
            # Try a simple operation
            if hasattr(connection, 'ping'):
                connection.ping()
            elif hasattr(connection, 'execute'):
                connection.execute('SELECT 1')
            return True
        except Exception:
            return False
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool"""
        with self.pool.acquire() as connection:
            yield connection
    
    def get_stats(self) -> Dict:
        """Get connection pool statistics"""
        return self.pool.get_stats()


class CachePool:
    """Memory cache pool with LRU eviction"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, tuple] = {}  # key -> (value, timestamp)
        self.access_order: List[str] = []  # LRU tracking
        self.lock = threading.RLock()
        self.logger = logging.getLogger("cache_pool")
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                # Check TTL
                if time.time() - timestamp > self.ttl:
                    del self.cache[key]
                    self.access_order.remove(key)
                    self.stats["expired"] += 1
                    self.stats["misses"] += 1
                    return None
                
                # Update access order (LRU)
                self.access_order.remove(key)
                self.access_order.append(key)
                self.stats["hits"] += 1
                return value
            
            self.stats["misses"] += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put value in cache"""
        with self.lock:
            # Remove if already exists
            if key in self.cache:
                self.access_order.remove(key)
            
            # Evict if at capacity
            while len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new item
            self.cache[key] = (value, time.time())
            self.access_order.append(key)
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if self.access_order:
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
            self.stats["evictions"] += 1
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            hit_rate = self.stats["hits"] / max(self.stats["hits"] + self.stats["misses"], 1)
            return {
                **self.stats,
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
                "timestamp": time.time()
            }


def main():
    """Test resource pools"""
    def create_test_resource():
        return {"id": time.time(), "data": "test"}
    
    # Test resource pool
    pool = ResourcePool(create_test_resource, max_size=5)
    
    with pool.acquire() as resource:
        print(f"Using resource: {resource}")
    
    print("Pool stats:", pool.get_stats())
    
    # Test cache pool
    cache = CachePool(max_size=10)
    cache.put("key1", "value1")
    print("Cache get:", cache.get("key1"))
    print("Cache stats:", cache.get_stats())


if __name__ == "__main__":
    main()