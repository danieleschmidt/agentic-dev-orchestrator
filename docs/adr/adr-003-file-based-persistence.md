# ADR-003: File-based Persistence Strategy

## Status
Accepted

## Context

The agentic development orchestrator needs to persist various types of data:
- Backlog items and their metadata
- Execution state and progress tracking
- Agent artifacts and work products
- Configuration and preferences
- Metrics and historical data

Several persistence strategies were evaluated:

1. **File-based Storage**: JSON/YAML files on filesystem
2. **Embedded Database**: SQLite or similar
3. **External Database**: PostgreSQL, MySQL, etc.
4. **Object Storage**: S3, GCS, Azure Blob
5. **Hybrid Approach**: Combination of the above

Key requirements:
- Simple deployment and maintenance
- Version control friendly
- Human readable/editable
- Suitable for single-node operation
- Minimal external dependencies

## Decision

We will implement a **file-based persistence strategy** using structured JSON and YAML files organized in a predictable directory hierarchy.

**File Organization:**
```
project/
├── backlog.yml              # Main backlog configuration
├── backlog/                 # Individual backlog items
│   ├── item-001.json
│   ├── item-002.json
│   └── ...
├── docs/
│   ├── status/              # Execution reports and metrics
│   │   ├── latest.json
│   │   └── status_YYYYMMDD_HHMMSS.json
│   └── adr/                 # Architecture decisions
├── escalations/             # Human intervention logs
│   ├── escalation-001.json
│   └── ...
└── .ado/                    # Runtime state and cache
    ├── config.json
    ├── locks/
    └── cache/
```

**Data Formats:**
- **YAML**: Human-editable configuration files
- **JSON**: Structured data and API responses
- **Markdown**: Documentation and reports
- **Plain Text**: Log files and simple data

## Consequences

### Positive
- **Simplicity**: No database installation or management required
- **Version Control**: All data can be tracked in Git
- **Transparency**: Files are human-readable and editable
- **Portability**: Easy to backup, migrate, and share
- **Debugging**: Direct file inspection for troubleshooting
- **Low Overhead**: Minimal resource usage
- **Development Friendly**: Easy to modify during development
- **Offline Capability**: Works without network connectivity

### Negative
- **Scalability Limits**: Poor performance with large datasets
- **Concurrency Issues**: File locking required for concurrent access
- **Query Limitations**: No SQL-like query capabilities
- **Data Integrity**: Manual validation required
- **Atomic Operations**: Complex to implement transactions
- **Search Performance**: Linear search through files
- **Type Safety**: No schema enforcement at storage level

## Alternatives Considered

### 1. SQLite Embedded Database
- **Pros**: SQL queries, ACID transactions, good performance
- **Cons**: Binary format, not version-control friendly, requires SQL knowledge

### 2. PostgreSQL/MySQL External Database
- **Pros**: Full SQL features, excellent performance, mature ecosystem
- **Cons**: Complex deployment, requires database administration, overkill for single-node

### 3. NoSQL Database (MongoDB, CouchDB)
- **Pros**: Schema flexibility, good for document storage
- **Cons**: Additional infrastructure, learning curve, complex deployment

### 4. Cloud Object Storage (S3, GCS)
- **Pros**: Highly scalable, durable, managed service
- **Cons**: Requires cloud account, network dependency, API complexity

### 5. Git as Database
- **Pros**: Version control built-in, distributed, conflict resolution
- **Cons**: Not designed for frequent updates, complex merge scenarios

## Implementation Guidelines

### File Naming Conventions
- Use lowercase with hyphens: `backlog-item-123.json`
- Include timestamps for historical data: `status_20250127_143022.json`
- Use descriptive names: `authentication-feature.json` vs `item-001.json`

### Data Validation
```python
def validate_backlog_item(data: dict) -> bool:
    """Validate backlog item structure"""
    required_fields = ['id', 'title', 'type', 'description']
    return all(field in data for field in required_fields)
```

### Concurrency Control
```python
import fcntl
import json

def atomic_write(filepath: str, data: dict) -> None:
    """Write data atomically to prevent corruption"""
    temp_file = f"{filepath}.tmp"
    with open(temp_file, 'w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        json.dump(data, f, indent=2)
    
    os.rename(temp_file, filepath)
```

### Backup Strategy
- **Git Integration**: All configuration files tracked in version control
- **Timestamped Copies**: Status files include timestamps for history
- **Export Functionality**: Ability to export data to other formats
- **Recovery Procedures**: Documented steps for data recovery

### Performance Optimizations
- **Lazy Loading**: Load files only when needed
- **Caching**: In-memory cache for frequently accessed data
- **Indexing**: Maintain simple index files for quick lookups
- **Compression**: Gzip large files to save space

### Migration Path
Future migration to database storage is supported by:
- **Abstraction Layer**: Data access through repository pattern
- **Export Tools**: Convert file data to database format
- **Hybrid Mode**: Gradual migration with both systems running
- **Schema Documentation**: Clear data structure documentation

### Example Implementation
```python
class FileRepository:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
    
    def save_backlog_item(self, item: BacklogItem) -> None:
        filepath = self.base_path / "backlog" / f"{item.id}.json"
        atomic_write(filepath, asdict(item))
    
    def load_backlog_items(self) -> List[BacklogItem]:
        items = []
        backlog_dir = self.base_path / "backlog"
        for file_path in backlog_dir.glob("*.json"):
            with open(file_path) as f:
                data = json.load(f)
                items.append(BacklogItem(**data))
        return items
```

This file-based approach provides an excellent foundation for the initial version while maintaining flexibility for future enhancements.