#!/bin/bash
# Production deployment script for Sentiment Analysis Service

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="ado-sentiment"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.sentiment.yml}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    local deps=("docker" "docker-compose")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "$dep is required but not installed."
            exit 1
        fi
    done
}

# Health check function
health_check() {
    local service_url="$1"
    local max_attempts=30
    local attempt=1
    
    log_info "Performing health check on $service_url"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$service_url/health" > /dev/null 2>&1; then
            log_info "Health check passed!"
            return 0
        fi
        
        log_info "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 10
        ((attempt++))
    done
    
    log_error "Health check failed after $max_attempts attempts"
    return 1
}

# Pre-deployment checks
pre_deploy_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check if .env file exists
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        log_warn "No .env file found. Creating template..."
        cat > "$PROJECT_ROOT/.env" << EOF
# Sentiment Analysis Configuration
LOG_LEVEL=INFO
SENTIMENT_MAX_WORKERS=4
SENTIMENT_BATCH_SIZE=20
SENTIMENT_CACHE_TTL=3600

# Optional API keys
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Security
ALLOWED_HOSTS=*
CORS_ORIGINS=*

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
EOF
        log_warn "Please configure the .env file before proceeding"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Create SSL certificates if they don't exist
    if [ ! -d "$PROJECT_ROOT/nginx/ssl" ]; then
        log_info "Creating self-signed SSL certificates for development..."
        mkdir -p "$PROJECT_ROOT/nginx/ssl"
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$PROJECT_ROOT/nginx/ssl/key.pem" \
            -out "$PROJECT_ROOT/nginx/ssl/cert.pem" \
            -subj "/C=US/ST=CA/L=SF/O=ADO/CN=localhost" 2>/dev/null || true
    fi
    
    log_info "Pre-deployment checks completed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build main sentiment service
    docker-compose -f "$COMPOSE_FILE" build --no-cache sentiment-api
    
    # Tag with version
    docker tag "${SERVICE_NAME}_sentiment-api:latest" "${SERVICE_NAME}_sentiment-api:$VERSION"
    
    log_info "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log_info "Deploying services with Docker Compose..."
    
    cd "$PROJECT_ROOT"
    
    # Stop existing services
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans || true
    
    # Pull latest images for third-party services
    docker-compose -f "$COMPOSE_FILE" pull redis nginx prometheus grafana
    
    # Start services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    log_info "Services deployed successfully"
}

# Post-deployment verification
post_deploy_verification() {
    log_info "Running post-deployment verification..."
    
    # Wait for services to be ready
    sleep 30
    
    # Check service health
    local services=("http://localhost:5000" "http://localhost:80")
    
    for service in "${services[@]}"; do
        if health_check "$service"; then
            log_info "Service $service is healthy"
        else
            log_error "Service $service failed health check"
            return 1
        fi
    done
    
    # Test sentiment analysis endpoint
    log_info "Testing sentiment analysis functionality..."
    response=$(curl -s -X POST "http://localhost/api/v1/sentiment/analyze" \
        -H "Content-Type: application/json" \
        -d '{"text": "This is a test message"}' || echo "ERROR")
    
    if [[ "$response" == *"label"* ]]; then
        log_info "Sentiment analysis endpoint is working correctly"
    else
        log_error "Sentiment analysis endpoint test failed"
        log_error "Response: $response"
        return 1
    fi
    
    log_info "Post-deployment verification completed successfully"
}

# Performance benchmark
run_benchmark() {
    log_info "Running performance benchmark..."
    
    # Simple load test using curl
    log_info "Testing single analysis performance..."
    time curl -s -X POST "http://localhost/api/v1/sentiment/analyze" \
        -H "Content-Type: application/json" \
        -d '{"text": "This is a performance test message"}' > /dev/null
    
    # Batch analysis test
    log_info "Testing batch analysis performance..."
    batch_data='{"texts": ["Great service!", "Terrible experience", "Average quality", "Outstanding work!", "Poor implementation"]}'
    time curl -s -X POST "http://localhost/api/v1/sentiment/batch" \
        -H "Content-Type: application/json" \
        -d "$batch_data" > /dev/null
    
    log_info "Benchmark completed"
}

# Show service status
show_status() {
    log_info "Service Status:"
    docker-compose -f "$PROJECT_ROOT/$COMPOSE_FILE" ps
    
    log_info "\nService URLs:"
    echo "  • Sentiment API: http://localhost:5000"
    echo "  • Nginx Proxy: http://localhost (HTTPS: https://localhost)"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • Grafana: http://localhost:3000 (admin/admin123)"
    echo "  • Redis: localhost:6379"
    
    log_info "\nHealth Check:"
    curl -s "http://localhost:5000/health" | python3 -m json.tool || echo "Health check failed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up deployment..."
    cd "$PROJECT_ROOT"
    docker-compose -f "$COMPOSE_FILE" down --volumes --remove-orphans
    docker system prune -f
    log_info "Cleanup completed"
}

# Main deployment function
deploy() {
    log_info "Starting deployment of Sentiment Analysis Service..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Version: $VERSION"
    
    check_dependencies
    pre_deploy_checks
    build_images
    deploy_services
    post_deploy_verification
    run_benchmark
    show_status
    
    log_info "Deployment completed successfully! 🚀"
    log_info "Access the service at: http://localhost"
}

# Script usage
usage() {
    cat << EOF
Usage: $0 [COMMAND]

Commands:
    deploy      Deploy the sentiment analysis service (default)
    build       Build Docker images only
    start       Start existing services
    stop        Stop all services
    restart     Restart all services
    status      Show service status
    logs        Show service logs
    test        Run functionality tests
    benchmark   Run performance benchmark
    cleanup     Stop services and clean up resources
    help        Show this help message

Environment Variables:
    VERSION     Image version tag (default: latest)
    ENVIRONMENT Deployment environment (default: production)
    COMPOSE_FILE Docker compose file (default: docker-compose.sentiment.yml)

Examples:
    $0 deploy
    VERSION=v1.0.0 $0 deploy
    ENVIRONMENT=staging $0 build
EOF
}

# Handle commands
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    build)
        check_dependencies
        pre_deploy_checks
        build_images
        ;;
    start)
        cd "$PROJECT_ROOT"
        docker-compose -f "$COMPOSE_FILE" up -d
        show_status
        ;;
    stop)
        cd "$PROJECT_ROOT"
        docker-compose -f "$COMPOSE_FILE" down
        ;;
    restart)
        cd "$PROJECT_ROOT"
        docker-compose -f "$COMPOSE_FILE" restart
        show_status
        ;;
    status)
        show_status
        ;;
    logs)
        cd "$PROJECT_ROOT"
        docker-compose -f "$COMPOSE_FILE" logs -f
        ;;
    test)
        post_deploy_verification
        ;;
    benchmark)
        run_benchmark
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        log_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac