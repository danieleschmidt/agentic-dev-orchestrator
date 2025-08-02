#!/bin/bash
# Automation runner script for ADO project
# Runs various automation tasks based on command line arguments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  metrics          Collect project metrics"
    echo "  maintenance      Run maintenance tasks"
    echo "  health           Check repository health"
    echo "  all              Run all automation tasks"
    echo ""
    echo "Options:"
    echo "  --update-deps    Update dependencies (for maintenance)"
    echo "  --report-only    Generate report only (no changes)"
    echo "  --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 metrics"
    echo "  $0 maintenance --update-deps"
    echo "  $0 health"
    echo "  $0 all"
}

# Function to run metrics collection
run_metrics() {
    print_status "Running metrics collection..."
    cd "$PROJECT_ROOT"
    
    if [[ "$1" == "--report-only" ]]; then
        python scripts/collect_metrics.py --report-only
    else
        python scripts/collect_metrics.py
    fi
    
    if [[ $? -eq 0 ]]; then
        print_success "Metrics collection completed"
    else
        print_error "Metrics collection failed"
        return 1
    fi
}

# Function to run maintenance
run_maintenance() {
    print_status "Running maintenance tasks..."
    cd "$PROJECT_ROOT"
    
    local args=""
    if [[ "$1" == "--update-deps" ]]; then
        args="--update-deps"
    elif [[ "$1" == "--report-only" ]]; then
        args="--report-only"
    fi
    
    python scripts/automate_maintenance.py $args
    
    if [[ $? -eq 0 ]]; then
        print_success "Maintenance completed"
    else
        print_error "Maintenance failed"
        return 1
    fi
}

# Function to run health check
run_health_check() {
    print_status "Running repository health check..."
    cd "$PROJECT_ROOT"
    
    python scripts/repository_health_monitor.py
    
    if [[ $? -eq 0 ]]; then
        print_success "Health check completed - Repository is healthy"
    else
        print_warning "Health check completed - Issues detected"
        return 1
    fi
}

# Function to run all automation tasks
run_all() {
    print_status "Running all automation tasks..."
    
    local failed=0
    
    # Run health check first
    if ! run_health_check; then
        failed=$((failed + 1))
    fi
    
    # Run metrics collection
    if ! run_metrics; then
        failed=$((failed + 1))
    fi
    
    # Run maintenance (without dependency updates by default)
    if ! run_maintenance; then
        failed=$((failed + 1))
    fi
    
    if [[ $failed -eq 0 ]]; then
        print_success "All automation tasks completed successfully"
    else
        print_warning "$failed automation task(s) failed"
        return 1
    fi
}

# Function to setup automation (install dependencies)
setup_automation() {
    print_status "Setting up automation dependencies..."
    cd "$PROJECT_ROOT"
    
    # Install required Python packages
    pip install -q pip-tools pip-audit radon bandit safety mypy ruff pytest pytest-cov
    
    # Make sure scripts are executable
    chmod +x scripts/*.py scripts/*.sh
    
    print_success "Automation setup completed"
}

# Main execution
main() {
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Parse command line arguments
    case "$1" in
        "metrics")
            run_metrics "$2"
            ;;
        "maintenance")
            run_maintenance "$2"
            ;;
        "health")
            run_health_check
            ;;
        "all")
            run_all
            ;;
        "setup")
            setup_automation
            ;;
        "--help"|"-h"|"help")
            show_usage
            ;;
        "")
            print_error "No command specified"
            show_usage
            exit 1
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Check if script is being executed (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi