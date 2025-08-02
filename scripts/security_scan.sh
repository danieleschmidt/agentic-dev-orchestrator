#!/bin/bash
# Security scanning script for ADO
# Runs various security scans and generates reports

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCAN_DIR="${1:-.}"
REPORT_DIR="security-reports"
FAIL_ON_HIGH="${FAIL_ON_HIGH:-true}"
FAIL_ON_MEDIUM="${FAIL_ON_MEDIUM:-false}"
DOCKER_IMAGE="${DOCKER_IMAGE:-ado:latest}"

# Create report directory
mkdir -p "$REPORT_DIR"

echo -e "${BLUE}🔒 Starting security scan for ADO${NC}"
echo -e "${BLUE}Scan directory: $SCAN_DIR${NC}"
echo -e "${BLUE}Report directory: $REPORT_DIR${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to run scan with error handling
run_scan() {
    local scan_name="$1"
    local scan_command="$2"
    local report_file="$3"
    
    echo -e "${YELLOW}Running $scan_name...${NC}"
    
    if eval "$scan_command" > "$REPORT_DIR/$report_file" 2>&1; then
        echo -e "${GREEN}✅ $scan_name completed successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ $scan_name failed${NC}"
        return 1
    fi
}

# Track scan results
SCAN_RESULTS=()
HIGH_ISSUES=0
MEDIUM_ISSUES=0
LOW_ISSUES=0

# 1. Python dependency vulnerability scan
if command_exists safety; then
    echo -e "${YELLOW}🔍 Scanning Python dependencies for vulnerabilities...${NC}"
    
    # Run safety check
    if safety check --json > "$REPORT_DIR/python-vulnerabilities.json" 2>/dev/null; then
        VULN_COUNT=$(jq length "$REPORT_DIR/python-vulnerabilities.json" 2>/dev/null || echo "0")
        if [ "$VULN_COUNT" -gt 0 ]; then
            echo -e "${RED}❌ Found $VULN_COUNT Python vulnerabilities${NC}"
            HIGH_ISSUES=$((HIGH_ISSUES + VULN_COUNT))
            SCAN_RESULTS+=("Python vulnerabilities: $VULN_COUNT")
        else
            echo -e "${GREEN}✅ No Python vulnerabilities found${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Safety check completed with warnings${NC}"
        # Try to get human-readable output
        safety check --output text > "$REPORT_DIR/python-vulnerabilities.txt" 2>&1 || true
    fi
else
    echo -e "${YELLOW}⚠️  Safety not installed, skipping Python vulnerability scan${NC}"
fi

# 2. Python code security scan with Bandit
if command_exists bandit; then
    echo -e "${YELLOW}🔍 Scanning Python code for security issues...${NC}"
    
    bandit -r "$SCAN_DIR" -f json -o "$REPORT_DIR/bandit-report.json" -x "*/tests/*,*/venv/*" 2>/dev/null || true
    
    if [ -f "$REPORT_DIR/bandit-report.json" ]; then
        HIGH_COUNT=$(jq '[.results[] | select(.issue_severity == "HIGH")] | length' "$REPORT_DIR/bandit-report.json" 2>/dev/null || echo "0")
        MEDIUM_COUNT=$(jq '[.results[] | select(.issue_severity == "MEDIUM")] | length' "$REPORT_DIR/bandit-report.json" 2>/dev/null || echo "0")
        LOW_COUNT=$(jq '[.results[] | select(.issue_severity == "LOW")] | length' "$REPORT_DIR/bandit-report.json" 2>/dev/null || echo "0")
        
        HIGH_ISSUES=$((HIGH_ISSUES + HIGH_COUNT))
        MEDIUM_ISSUES=$((MEDIUM_ISSUES + MEDIUM_COUNT))
        LOW_ISSUES=$((LOW_ISSUES + LOW_COUNT))
        
        if [ "$HIGH_COUNT" -gt 0 ] || [ "$MEDIUM_COUNT" -gt 0 ] || [ "$LOW_COUNT" -gt 0 ]; then
            echo -e "${RED}❌ Bandit found security issues: High=$HIGH_COUNT, Medium=$MEDIUM_COUNT, Low=$LOW_COUNT${NC}"
            SCAN_RESULTS+=("Bandit: High=$HIGH_COUNT, Medium=$MEDIUM_COUNT, Low=$LOW_COUNT")
        else
            echo -e "${GREEN}✅ No security issues found by Bandit${NC}"
        fi
        
        # Generate human-readable report
        bandit -r "$SCAN_DIR" -f txt -o "$REPORT_DIR/bandit-report.txt" -x "*/tests/*,*/venv/*" 2>/dev/null || true
    fi
else
    echo -e "${YELLOW}⚠️  Bandit not installed, skipping Python security scan${NC}"
fi

# 3. Secrets detection
if command_exists detect-secrets; then
    echo -e "${YELLOW}🔍 Scanning for secrets...${NC}"
    
    # Create baseline first
    detect-secrets scan --baseline "$REPORT_DIR/secrets-baseline.json" "$SCAN_DIR" 2>/dev/null || true
    
    if [ -f "$REPORT_DIR/secrets-baseline.json" ]; then
        SECRET_COUNT=$(jq '.results | to_entries | map(.value | length) | add // 0' "$REPORT_DIR/secrets-baseline.json" 2>/dev/null || echo "0")
        
        if [ "$SECRET_COUNT" -gt 0 ]; then
            echo -e "${RED}❌ Found $SECRET_COUNT potential secrets${NC}"
            HIGH_ISSUES=$((HIGH_ISSUES + SECRET_COUNT))
            SCAN_RESULTS+=("Secrets detected: $SECRET_COUNT")
            
            # Generate human-readable report
            detect-secrets audit "$REPORT_DIR/secrets-baseline.json" --report > "$REPORT_DIR/secrets-report.txt" 2>/dev/null || true
        else
            echo -e "${GREEN}✅ No secrets detected${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠️  detect-secrets not installed, skipping secrets scan${NC}"
fi

# 4. Docker image security scan (if Docker image exists)
if command_exists docker && docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    echo -e "${YELLOW}🔍 Scanning Docker image for vulnerabilities...${NC}"
    
    # Try Trivy first
    if command_exists trivy; then
        trivy image --format json --output "$REPORT_DIR/docker-trivy.json" "$DOCKER_IMAGE" 2>/dev/null || true
        
        if [ -f "$REPORT_DIR/docker-trivy.json" ]; then
            DOCKER_HIGH=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH")] | length' "$REPORT_DIR/docker-trivy.json" 2>/dev/null || echo "0")
            DOCKER_MEDIUM=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "MEDIUM")] | length' "$REPORT_DIR/docker-trivy.json" 2>/dev/null || echo "0")
            DOCKER_LOW=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "LOW")] | length' "$REPORT_DIR/docker-trivy.json" 2>/dev/null || echo "0")
            
            HIGH_ISSUES=$((HIGH_ISSUES + DOCKER_HIGH))
            MEDIUM_ISSUES=$((MEDIUM_ISSUES + DOCKER_MEDIUM))
            LOW_ISSUES=$((LOW_ISSUES + DOCKER_LOW))
            
            if [ "$DOCKER_HIGH" -gt 0 ] || [ "$DOCKER_MEDIUM" -gt 0 ] || [ "$DOCKER_LOW" -gt 0 ]; then
                echo -e "${RED}❌ Docker vulnerabilities: High=$DOCKER_HIGH, Medium=$DOCKER_MEDIUM, Low=$DOCKER_LOW${NC}"
                SCAN_RESULTS+=("Docker: High=$DOCKER_HIGH, Medium=$DOCKER_MEDIUM, Low=$DOCKER_LOW")
            else
                echo -e "${GREEN}✅ No Docker vulnerabilities found${NC}"
            fi
            
            # Generate human-readable report
            trivy image --format table --output "$REPORT_DIR/docker-trivy.txt" "$DOCKER_IMAGE" 2>/dev/null || true
        fi
    # Try Snyk if available
    elif command_exists snyk; then
        snyk container test "$DOCKER_IMAGE" --json > "$REPORT_DIR/docker-snyk.json" 2>/dev/null || true
        
        if [ -f "$REPORT_DIR/docker-snyk.json" ]; then
            DOCKER_ISSUES=$(jq '.vulnerabilities | length' "$REPORT_DIR/docker-snyk.json" 2>/dev/null || echo "0")
            
            if [ "$DOCKER_ISSUES" -gt 0 ]; then
                echo -e "${RED}❌ Found $DOCKER_ISSUES Docker vulnerabilities${NC}"
                HIGH_ISSUES=$((HIGH_ISSUES + DOCKER_ISSUES))
                SCAN_RESULTS+=("Docker vulnerabilities: $DOCKER_ISSUES")
            else
                echo -e "${GREEN}✅ No Docker vulnerabilities found${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}⚠️  No Docker security scanner available (trivy, snyk)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Docker image '$DOCKER_IMAGE' not found, skipping Docker scan${NC}"
fi

# 5. Dockerfile security scan
if [ -f "$SCAN_DIR/Dockerfile" ] && command_exists hadolint; then
    echo -e "${YELLOW}🔍 Scanning Dockerfile for security issues...${NC}"
    
    hadolint "$SCAN_DIR/Dockerfile" --format json > "$REPORT_DIR/dockerfile-scan.json" 2>/dev/null || true
    
    if [ -f "$REPORT_DIR/dockerfile-scan.json" ]; then
        DOCKERFILE_ISSUES=$(jq length "$REPORT_DIR/dockerfile-scan.json" 2>/dev/null || echo "0")
        
        if [ "$DOCKERFILE_ISSUES" -gt 0 ]; then
            echo -e "${YELLOW}⚠️  Found $DOCKERFILE_ISSUES Dockerfile issues${NC}"
            MEDIUM_ISSUES=$((MEDIUM_ISSUES + DOCKERFILE_ISSUES))
            SCAN_RESULTS+=("Dockerfile issues: $DOCKERFILE_ISSUES")
        else
            echo -e "${GREEN}✅ No Dockerfile issues found${NC}"
        fi
        
        # Generate human-readable report
        hadolint "$SCAN_DIR/Dockerfile" > "$REPORT_DIR/dockerfile-scan.txt" 2>/dev/null || true
    fi
else
    echo -e "${YELLOW}⚠️  Hadolint not installed or Dockerfile not found, skipping Dockerfile scan${NC}"
fi

# 6. License compliance scan
if command_exists pip-licenses; then
    echo -e "${YELLOW}🔍 Scanning license compliance...${NC}"
    
    pip-licenses --format json --output-file "$REPORT_DIR/licenses.json" 2>/dev/null || true
    pip-licenses --format plain --output-file "$REPORT_DIR/licenses.txt" 2>/dev/null || true
    
    echo -e "${GREEN}✅ License scan completed${NC}"
else
    echo -e "${YELLOW}⚠️  pip-licenses not installed, skipping license scan${NC}"
fi

# 7. Generate SBOM
if [ -f "$SCAN_DIR/scripts/generate_sbom.py" ]; then
    echo -e "${YELLOW}🔍 Generating Software Bill of Materials (SBOM)...${NC}"
    
    python "$SCAN_DIR/scripts/generate_sbom.py" --output "$REPORT_DIR/sbom.json" --project-root "$SCAN_DIR" 2>/dev/null || true
    
    if [ -f "$REPORT_DIR/sbom.json" ]; then
        echo -e "${GREEN}✅ SBOM generated successfully${NC}"
    else
        echo -e "${YELLOW}⚠️  SBOM generation failed${NC}"
    fi
fi

# Generate summary report
echo -e "${BLUE}📊 Generating security scan summary...${NC}"

cat > "$REPORT_DIR/security-summary.json" << EOF
{
  "scan_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "scan_directory": "$SCAN_DIR",
  "total_issues": {
    "high": $HIGH_ISSUES,
    "medium": $MEDIUM_ISSUES,
    "low": $LOW_ISSUES
  },
  "scan_results": $(printf '%s\n' "${SCAN_RESULTS[@]}" | jq -R . | jq -s .),
  "reports_generated": [
    $(find "$REPORT_DIR" -name "*.json" -o -name "*.txt" | jq -R . | paste -sd, -)
  ]
}
EOF

# Generate human-readable summary
cat > "$REPORT_DIR/security-summary.txt" << EOF
Security Scan Summary
====================

Scan Date: $(date)
Scan Directory: $SCAN_DIR

Issues Found:
- High Severity: $HIGH_ISSUES
- Medium Severity: $MEDIUM_ISSUES  
- Low Severity: $LOW_ISSUES

Detailed Results:
EOF

for result in "${SCAN_RESULTS[@]}"; do
    echo "- $result" >> "$REPORT_DIR/security-summary.txt"
done

echo "" >> "$REPORT_DIR/security-summary.txt"
echo "Reports Generated:" >> "$REPORT_DIR/security-summary.txt"
find "$REPORT_DIR" -name "*.json" -o -name "*.txt" | sed 's|^|- |' >> "$REPORT_DIR/security-summary.txt"

# Display summary
echo ""
echo -e "${BLUE}📊 Security Scan Summary${NC}"
echo -e "${BLUE}========================${NC}"
echo -e "High Severity Issues: ${RED}$HIGH_ISSUES${NC}"
echo -e "Medium Severity Issues: ${YELLOW}$MEDIUM_ISSUES${NC}"
echo -e "Low Severity Issues: ${BLUE}$LOW_ISSUES${NC}"
echo ""

if [ ${#SCAN_RESULTS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Detailed Results:${NC}"
    for result in "${SCAN_RESULTS[@]}"; do
        echo -e "  • $result"
    done
    echo ""
fi

echo -e "${GREEN}Reports saved to: $REPORT_DIR/${NC}"
echo ""

# Determine exit code based on findings
EXIT_CODE=0

if [ "$FAIL_ON_HIGH" = "true" ] && [ "$HIGH_ISSUES" -gt 0 ]; then
    echo -e "${RED}❌ Failing due to high severity issues (FAIL_ON_HIGH=true)${NC}"
    EXIT_CODE=1
fi

if [ "$FAIL_ON_MEDIUM" = "true" ] && [ "$MEDIUM_ISSUES" -gt 0 ]; then
    echo -e "${RED}❌ Failing due to medium severity issues (FAIL_ON_MEDIUM=true)${NC}"
    EXIT_CODE=1
fi

if [ "$EXIT_CODE" -eq 0 ]; then
    if [ "$HIGH_ISSUES" -eq 0 ] && [ "$MEDIUM_ISSUES" -eq 0 ] && [ "$LOW_ISSUES" -eq 0 ]; then
        echo -e "${GREEN}🎉 No security issues found!${NC}"
    else
        echo -e "${YELLOW}⚠️  Security issues found but not failing build${NC}"
    fi
fi

exit $EXIT_CODE