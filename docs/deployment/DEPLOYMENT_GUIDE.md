# Deployment Guide for Agentic Dev Orchestrator

This guide covers different deployment strategies and environments for ADO.

## 📋 Deployment Options

### Local Development

#### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- Git
- GitHub CLI (optional)

#### Quick Start
```bash
# Clone repository
git clone https://github.com/terragon-labs/agentic-dev-orchestrator.git
cd agentic-dev-orchestrator

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Install dependencies
make install

# Run tests
make test

# Start development server
make dev
```

#### Docker Development
```bash
# Build and start development environment
docker-compose up -d ado-dev

# Access development container
docker-compose exec ado-dev bash

# Run ADO commands
ado --help
```

### Production Deployment

#### Container-Based Deployment

##### Docker
```bash
# Build production image
docker build -t ado:latest .

# Run production container
docker run -d \
  --name ado-prod \
  -e GITHUB_TOKEN=your_token \
  -e OPENAI_API_KEY=your_key \
  -v $(pwd)/backlog:/app/backlog \
  -v $(pwd)/.ado:/app/.ado \
  ado:latest
```

##### Docker Compose (Production)
```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d

# Scale workers
docker-compose -f docker-compose.prod.yml up -d --scale ado-worker=3
```

#### Kubernetes Deployment

##### Prerequisites
- Kubernetes cluster (1.19+)
- kubectl configured
- Helm 3.x (optional)

##### Basic Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ado
  namespace: ado-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ado
  template:
    metadata:
      labels:
        app: ado
    spec:
      containers:
      - name: ado
        image: ghcr.io/terragon-labs/ado:latest
        ports:
        - containerPort: 8000
        env:
        - name: GITHUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: ado-secrets
              key: github-token
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ado-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ado-service
  namespace: ado-system
spec:
  selector:
    app: ado
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

##### Deploy to Kubernetes
```bash
# Create namespace
kubectl create namespace ado-system

# Create secrets
kubectl create secret generic ado-secrets \
  --from-literal=github-token=your_token \
  --from-literal=openai-api-key=your_key \
  -n ado-system

# Deploy application
kubectl apply -f k8s/

# Check deployment
kubectl get pods -n ado-system
kubectl logs -f deployment/ado -n ado-system
```

#### Cloud Platform Deployments

##### AWS ECS
```json
{
  "family": "ado",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/adoTaskRole",
  "containerDefinitions": [
    {
      "name": "ado",
      "image": "ghcr.io/terragon-labs/ado:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "GITHUB_TOKEN",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:ado/github-token"
        },
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:ado/openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ado",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

##### Google Cloud Run
```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ado
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 10
      containers:
      - image: gcr.io/project-id/ado:latest
        ports:
        - containerPort: 8000
        env:
        - name: GITHUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: ado-secrets
              key: github-token
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ado-secrets
              key: openai-key
        resources:
          limits:
            cpu: "1"
            memory: "1Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
```

##### Azure Container Instances
```yaml
# azure-container-instance.yaml
apiVersion: 2019-12-01
location: eastus
name: ado-container-group
properties:
  containers:
  - name: ado
    properties:
      image: ghcr.io/terragon-labs/ado:latest
      resources:
        requests:
          cpu: 0.5
          memoryInGb: 1
      ports:
      - port: 8000
      environmentVariables:
      - name: ENVIRONMENT
        value: production
      - name: GITHUB_TOKEN
        secureValue: your_github_token
      - name: OPENAI_API_KEY
        secureValue: your_openai_key
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8000
tags:
  environment: production
  application: ado
type: Microsoft.ContainerInstance/containerGroups
```

## 🔒 Security Configuration

### Environment Variables

**Required Secrets:**
```bash
# API Keys (NEVER commit these)
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional but recommended
SENTRY_DSN=https://xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx@sentry.io/project
DATADOG_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Security Best Practices:**
1. Use secret management services (AWS Secrets Manager, Azure Key Vault, etc.)
2. Rotate API keys regularly
3. Use least privilege access for service accounts
4. Enable audit logging
5. Implement network security policies

### Docker Security

```dockerfile
# Security-hardened Dockerfile example
FROM python:3.11-slim-bullseye AS base

# Create non-root user
RUN groupadd -r ado && useradd -r -g ado -d /app -s /bin/bash ado

# Set security-focused environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Copy application
COPY --chown=ado:ado . /app/
WORKDIR /app

# Switch to non-root user
USER ado

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["python", "-m", "ado", "serve"]
```

## 📋 Environment Configuration

### Development
```bash
# .env.development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
ADO_LOG_LEVEL=DEBUG

# Mock external services for development
MOCK_GITHUB_API=true
MOCK_LLM_API=true

# Development database
DATABASE_URL=sqlite:///dev.db

# Disable security features for development
SECURITY_SCAN_REQUIRED=false
HUMAN_REVIEW_REQUIRED=false
```

### Staging
```bash
# .env.staging
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO
ADO_LOG_LEVEL=INFO

# Use real services but with test credentials
GITHUB_TOKEN=${STAGING_GITHUB_TOKEN}
OPENAI_API_KEY=${STAGING_OPENAI_KEY}

# Staging database
DATABASE_URL=postgresql://user:pass@staging-db:5432/ado

# Enable most security features
SECURITY_SCAN_REQUIRED=true
HUMAN_REVIEW_REQUIRED=true
```

### Production
```bash
# .env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
ADO_LOG_LEVEL=INFO

# Production credentials (from secret management)
GITHUB_TOKEN=${GITHUB_TOKEN}
OPENAI_API_KEY=${OPENAI_API_KEY}

# Production database
DATABASE_URL=postgresql://user:pass@prod-db:5432/ado

# All security features enabled
SECURITY_SCAN_REQUIRED=true
HUMAN_REVIEW_REQUIRED=true
AUDIT_LOGGING_ENABLED=true

# Performance settings
AGENT_CONCURRENCY=5
MAX_ITEMS_PER_RUN=10
```

## 📋 Monitoring and Health Checks

### Health Check Endpoints

```python
# Health check implementation
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": __version__
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check with dependency validation."""
    checks = {
        "database": check_database_connection(),
        "github": check_github_api(),
        "llm": check_llm_api()
    }
    
    all_healthy = all(checks.values())
    status_code = 200 if all_healthy else 503
    
    return Response(
        content=json.dumps({
            "status": "ready" if all_healthy else "not_ready",
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }),
        status_code=status_code,
        media_type="application/json"
    )
```

### Monitoring Stack

#### Prometheus Metrics
```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  grafana-storage:
```

## 🚀 CI/CD Pipelines

### GitHub Actions (Production)

```yaml
# .github/workflows/deploy-production.yml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=ref,event=tag
            type=sha,prefix={{branch}}-
            type=raw,value=latest,enable={{is_default_branch}}
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      - name: Deploy to Kubernetes
        run: |
          echo "$KUBE_CONFIG" | base64 -d > kubeconfig
          export KUBECONFIG=kubeconfig
          kubectl set image deployment/ado ado=${{ steps.meta.outputs.tags }} -n ado-system
          kubectl rollout status deployment/ado -n ado-system --timeout=300s
        env:
          KUBE_CONFIG: ${{ secrets.KUBE_CONFIG }}
```

## 🔄 Backup and Recovery

### Data Backup Strategy

```bash
#!/bin/bash
# backup.sh - Automated backup script

BACKUP_DIR="/backups/ado/$(date +%Y-%m-%d)"
DATABASE_URL=${DATABASE_URL}
S3_BUCKET=${BACKUP_S3_BUCKET}

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup database
if [[ $DATABASE_URL == postgresql* ]]; then
    pg_dump "$DATABASE_URL" > "$BACKUP_DIR/database.sql"
elif [[ $DATABASE_URL == sqlite* ]]; then
    cp "${DATABASE_URL#sqlite:///}" "$BACKUP_DIR/database.db"
fi

# Backup configuration and data
tar -czf "$BACKUP_DIR/ado-data.tar.gz" \
    .ado/ \
    backlog/ \
    docs/status/ \
    escalations/

# Upload to S3 (if configured)
if [[ -n "$S3_BUCKET" ]]; then
    aws s3 sync "$BACKUP_DIR" "s3://$S3_BUCKET/ado/$(date +%Y-%m-%d)/"
fi

# Cleanup old backups (keep last 30 days)
find /backups/ado/ -type d -mtime +30 -exec rm -rf {} +

echo "Backup completed: $BACKUP_DIR"
```

### Disaster Recovery

```bash
#!/bin/bash
# restore.sh - Disaster recovery script

BACKUP_DATE=${1:-$(date +%Y-%m-%d)}
BACKUP_DIR="/backups/ado/$BACKUP_DATE"
S3_BUCKET=${BACKUP_S3_BUCKET}

# Download backup from S3 if needed
if [[ -n "$S3_BUCKET" ]] && [[ ! -d "$BACKUP_DIR" ]]; then
    mkdir -p "$BACKUP_DIR"
    aws s3 sync "s3://$S3_BUCKET/ado/$BACKUP_DATE/" "$BACKUP_DIR/"
fi

# Restore database
if [[ -f "$BACKUP_DIR/database.sql" ]]; then
    psql "$DATABASE_URL" < "$BACKUP_DIR/database.sql"
elif [[ -f "$BACKUP_DIR/database.db" ]]; then
    cp "$BACKUP_DIR/database.db" "${DATABASE_URL#sqlite:///}"
fi

# Restore data
if [[ -f "$BACKUP_DIR/ado-data.tar.gz" ]]; then
    tar -xzf "$BACKUP_DIR/ado-data.tar.gz"
fi

echo "Restore completed from backup: $BACKUP_DATE"
```

## 🔍 Troubleshooting Deployment Issues

### Common Issues

#### Container Won't Start
```bash
# Check container logs
docker logs ado-container

# Inspect container configuration
docker inspect ado-container

# Test with interactive shell
docker run -it --entrypoint=/bin/bash ado:latest
```

#### Permission Issues
```bash
# Fix file permissions
chown -R ado:ado /app
chmod -R 755 /app

# Check SELinux context (if applicable)
ls -Z /app
restorecon -R /app
```

#### Network Connectivity
```bash
# Test external API connectivity
curl -I https://api.github.com
curl -I https://api.openai.com

# Check DNS resolution
nslookup api.github.com

# Test internal service communication
telnet database-service 5432
```

#### Resource Constraints
```bash
# Check resource usage
docker stats ado-container

# Kubernetes resource monitoring
kubectl top pods -n ado-system
kubectl describe pod ado-xxx -n ado-system
```

### Performance Tuning

#### Container Resources
```yaml
# Kubernetes resource configuration
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

#### Application Configuration
```bash
# Performance environment variables
AGENT_CONCURRENCY=3
MAX_ITEMS_PER_RUN=5
CACHE_ENABLED=true
CACHE_TTL=3600
HTTP_TIMEOUT=30
AGENT_TIMEOUT=300
```

## 📚 Additional Resources

### Documentation
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)
- [Kubernetes Production Guide](https://kubernetes.io/docs/setup/production-environment/)
- [AWS ECS Best Practices](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/)

### Tools
- [Hadolint](https://github.com/hadolint/hadolint) - Dockerfile linter
- [dive](https://github.com/wagoodman/dive) - Docker image analyzer
- [kube-score](https://github.com/zegl/kube-score) - Kubernetes object analysis

### Internal Resources
- [Security Guide](../security/) - Security best practices
- [Monitoring Guide](../monitoring/) - Observability setup
- [Troubleshooting Guide](../troubleshooting/) - Common issues

---

**Need Help?** Check our [troubleshooting guide](../troubleshooting/) or create an [issue](https://github.com/terragon-labs/agentic-dev-orchestrator/issues) for deployment-specific problems.