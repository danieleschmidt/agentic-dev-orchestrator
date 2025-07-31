# SLSA Compliance Implementation Guide

## Overview

This document outlines the implementation of SLSA (Supply-chain Levels for Software Artifacts) compliance for the agentic-dev-orchestrator project.

## SLSA Level 2 Implementation

### Build Integrity Requirements

#### 1. Provenance Generation

**GitHub Actions Integration:**
```yaml
# .github/workflows/slsa-build.yml
name: SLSA Build and Publish

on:
  push:
    tags: ["v*"]

permissions:
  contents: read
  id-token: write
  packages: write

jobs:
  build:
    name: Build and Generate Provenance
    runs-on: ubuntu-latest
    outputs:
      hashes: ${{ steps.hash.outputs.hashes }}
      version: ${{ steps.version.outputs.version }}
    
    steps:
    - name: Checkout
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Generate hashes
      id: hash
      run: |
        cd dist && echo "hashes=$(sha256sum * | base64 -w0)" >> "$GITHUB_OUTPUT"
    
    - name: Extract version
      id: version
      run: echo "version=${GITHUB_REF#refs/tags/v}" >> "$GITHUB_OUTPUT"
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/

  provenance:
    name: Generate Provenance
    needs: [build]
    permissions:
      actions: read
      id-token: write
      contents: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.7.0
    with:
      base64-subjects: "${{ needs.build.outputs.hashes }}"
      upload-assets: true
      upload-tag-name: "v${{ needs.build.outputs.version }}"

  publish:
    name: Publish to PyPI
    needs: [build, provenance]
    runs-on: ubuntu-latest
    environment: release
    
    steps:
    - name: Download artifacts
      uses: actions/download-artifact@v3
      with:
        name: dist
        path: dist/
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        attestations: true
```

#### 2. Dependency Pinning Strategy

**requirements-lock.txt Generation:**
```bash
# Generate locked requirements
pip-compile requirements.in --generate-hashes --output-file requirements-lock.txt
pip-compile requirements-dev.in --generate-hashes --output-file requirements-dev-lock.txt
```

**Hash Verification in Dockerfile:**
```dockerfile
COPY requirements-lock.txt .
RUN pip install --require-hashes --no-deps -r requirements-lock.txt
```

### Source Integrity

#### 1. Signed Commits Configuration

```bash
# Enable signed commits
git config --global commit.gpgsign true
git config --global user.signingkey [key-id]
git config --global gpg.program gpg
```

#### 2. Branch Protection Rules

**Required Configuration:**
```json
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "continuous-integration",
      "security-scan",
      "dependency-review"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true,
    "required_approving_review_count": 2
  },
  "restrictions": {
    "users": [],
    "teams": ["maintainers"]
  }
}
```

## SLSA Level 3 Enhancements

### Hermetic Builds

#### 1. Container-based Build Environment

```dockerfile
# build.Dockerfile
FROM python:3.11-slim as builder

# Install build dependencies
RUN pip install --no-cache-dir build wheel

# Create non-root user
RUN useradd --create-home --shell /bin/bash builder
USER builder
WORKDIR /home/builder

# Copy source
COPY --chown=builder:builder . .

# Build in isolated environment
RUN python -m build --wheel --outdir /tmp/dist
```

#### 2. Reproducible Build Configuration

```yaml
# reproducible-build.yml
name: Reproducible Build

on:
  push:
    tags: ["v*"]

jobs:
  build:
    runs-on: ubuntu-latest
    container:
      image: python:3.11-slim
      options: --read-only --tmpfs /tmp
    
    steps:
    - name: Install build tools
      run: |
        pip install --user build
        export PATH=$HOME/.local/bin:$PATH
    
    - name: Checkout
      uses: actions/checkout@v4
    
    - name: Set reproducible timestamp
      run: |
        export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
        echo "SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH" >> $GITHUB_ENV
    
    - name: Build package
      run: python -m build
```

## Supply Chain Security Monitoring

### 1. SBOM Enhancement

```python
# scripts/generate_sbom.py
import json
from cyclonedx.builder import Builder
from cyclonedx.model import Component, ComponentType
from cyclonedx.output.json import JsonV1Dot4

def generate_enhanced_sbom():
    builder = Builder()
    
    # Add main component
    main_component = Component(
        name="agentic-dev-orchestrator",
        version="0.1.0",
        component_type=ComponentType.APPLICATION
    )
    builder.add_component(main_component)
    
    # Add dependencies with vulnerability data
    # ... implementation
    
    # Output with SLSA attestation
    json_output = JsonV1Dot4(builder.build())
    with open("sbom-enhanced.json", "w") as f:
        f.write(json_output.output_as_string())
```

### 2. Continuous Vulnerability Monitoring

```yaml
# .github/workflows/security-monitoring.yml
name: Security Monitoring

on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:

jobs:
  vulnerability-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run OSV Scanner
      uses: google/osv-scanner-action@v1
      with:
        scan-args: |-
          --output=json
          --format=sarif
          ./
    
    - name: Upload SARIF results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: results.sarif
    
    - name: Notify security team
      if: failure()
      uses: 8398a7/action-slack@v3
      with:
        status: failure
        channel: '#security-alerts'
        webhook_url: ${{ secrets.SECURITY_SLACK_WEBHOOK }}
```

## Compliance Verification

### 1. SLSA Verifier Integration

```bash
# Install SLSA verifier
go install github.com/slsa-framework/slsa-verifier/v2/cmd/slsa-verifier@latest

# Verify package
slsa-verifier verify-artifact \
  --provenance-path agentic-dev-orchestrator.intoto.jsonl \
  --source-uri github.com/terragon-labs/agentic-dev-orchestrator \
  --source-tag v0.1.0 \
  agentic-dev-orchestrator-0.1.0.tar.gz
```

### 2. Policy as Code

```yaml
# .github/policies/supply-chain.yml
name: Supply Chain Policy
on:
  pull_request:
    paths:
      - 'requirements*.txt'
      - 'pyproject.toml'
      - 'setup.py'

jobs:
  policy-check:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Check dependency licenses
      uses: fossa-contrib/fossa-action@v2
      with:
        api-key: ${{ secrets.FOSSA_API_KEY }}
        
    - name: Validate SBOM
      run: |
        python scripts/validate_sbom.py
        
    - name: Check for known vulnerabilities
      run: |
        safety check --json --output vulns.json
        [ $(jq '.vulnerabilities | length' vulns.json) -eq 0 ]
```

## Attestation Framework

### 1. Build Attestation

```python
# scripts/create_attestation.py
import json
import hashlib
from datetime import datetime, timezone

def create_build_attestation(artifacts):
    attestation = {
        "_type": "https://in-toto.io/Statement/v0.1",
        "predicateType": "https://slsa.dev/provenance/v0.2",
        "subject": [
            {
                "name": artifact["name"],
                "digest": {"sha256": artifact["sha256"]}
            }
            for artifact in artifacts
        ],
        "predicate": {
            "builder": {
                "id": "https://github.com/terragon-labs/agentic-dev-orchestrator/.github/workflows/build.yml"
            },
            "buildType": "https://github.com/Attestations/GitHubActionsWorkflow@v1",
            "invocation": {
                "configSource": {
                    "uri": "git+https://github.com/terragon-labs/agentic-dev-orchestrator",
                    "digest": {"sha1": os.environ.get("GITHUB_SHA")},
                    "entryPoint": ".github/workflows/build.yml"
                }
            },
            "metadata": {
                "buildInvocationId": os.environ.get("GITHUB_RUN_ID"),
                "buildStartedOn": datetime.now(timezone.utc).isoformat(),
                "completeness": {
                    "parameters": True,
                    "environment": False,
                    "materials": True
                },
                "reproducible": True
            }
        }
    }
    
    return json.dumps(attestation, indent=2)
```

## Implementation Roadmap

### Phase 1: Basic SLSA Level 1 (Week 1)
- [x] Documented build process
- [ ] Version controlled source
- [ ] Generated provenance

### Phase 2: SLSA Level 2 (Week 2-3)
- [ ] Hosted build service (GitHub Actions)
- [ ] Authenticated provenance
- [ ] Service-generated provenance

### Phase 3: SLSA Level 3 (Week 4-6)
- [ ] Hermetic builds
- [ ] Signed provenance
- [ ] Isolated build environment

### Phase 4: Continuous Monitoring (Ongoing)
- [ ] Automated vulnerability scanning
- [ ] Policy enforcement
- [ ] Compliance reporting

## Verification Commands

```bash
# Verify build reproducibility
docker build -t ado-verify -f build.Dockerfile .
docker run --rm ado-verify python -m build
sha256sum dist/*

# Verify SLSA provenance
slsa-verifier verify-artifact \
  --provenance-path provenance.intoto.jsonl \
  --source-uri github.com/terragon-labs/agentic-dev-orchestrator \
  dist/agentic-dev-orchestrator-*.whl

# Verify SBOM integrity
cyclonedx-cli validate --input-file sbom.json
```

## References

- [SLSA Specification](https://slsa.dev/spec/)
- [GitHub SLSA Generator](https://github.com/slsa-framework/slsa-github-generator)
- [Supply Chain Security Best Practices](https://cloud.google.com/software-supply-chain-security/docs/safeguarding-packages)