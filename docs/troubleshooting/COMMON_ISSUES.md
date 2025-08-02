# Troubleshooting Common Issues

This guide helps you resolve common issues when using the Agentic Dev Orchestrator.

## 🚨 Quick Diagnostics

### Health Check

Run the built-in health check to identify common problems:

```bash
# Basic health check
ado health

# Detailed diagnostics
ado health --verbose

# Check specific components
ado health --check github,llm,config
```

### Common Error Patterns

| Error Pattern | Likely Cause | Quick Fix |
|--------------|-------------|----------|
| `Authentication failed` | Invalid tokens | Check API keys |
| `Rate limit exceeded` | Too many API calls | Reduce concurrency |
| `No backlog items found` | Empty/invalid backlog | Check JSON syntax |
| `Agent timeout` | Long-running operations | Increase timeout |
| `Permission denied` | Insufficient GitHub permissions | Update token scope |

## 🔑 Authentication Issues

### GitHub Authentication

#### Problem: `GitHub API authentication failed`

**Symptoms:**
```bash
ERROR: Authentication failed for GitHub API
HTTP 401: Bad credentials
```

**Solutions:**

1. **Check Token Validity**
   ```bash
   # Test your GitHub token
   curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user
   ```

2. **Verify Token Permissions**
   Required scopes:
   - `repo` (for repository access)
   - `workflow` (for GitHub Actions)
   - `read:user` (for user information)

3. **Regenerate Token**
   - Go to [GitHub Settings > Tokens](https://github.com/settings/tokens)
   - Click "Regenerate token"
   - Update `.env` file

4. **Check Token Format**
   ```bash
   # Classic tokens start with ghp_
   GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   
   # Fine-grained tokens start with github_pat_
   GITHUB_TOKEN=github_pat_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

#### Problem: `Repository not found or access denied`

**Solutions:**

1. **Verify Repository Format**
   ```bash
   # Correct format
   GITHUB_REPO=owner/repository-name
   
   # Not URLs
   # WRONG: GITHUB_REPO=https://github.com/owner/repo
   ```

2. **Check Repository Permissions**
   ```bash
   # Test repository access
   curl -H "Authorization: token $GITHUB_TOKEN" \
        https://api.github.com/repos/$GITHUB_REPO
   ```

### LLM Provider Authentication

#### Problem: `OpenAI API request failed`

**Symptoms:**
```bash
ERROR: OpenAI API authentication failed
HTTP 401: Incorrect API key provided
```

**Solutions:**

1. **Verify API Key Format**
   ```bash
   # OpenAI keys start with sk-
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

2. **Check API Key Status**
   - Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
   - Verify key is active and not revoked
   - Check usage limits and billing status

3. **Test API Access**
   ```bash
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
        -H "Content-Type: application/json" \
        -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"test"}],"max_tokens":5}' \
        https://api.openai.com/v1/chat/completions
   ```

#### Problem: `Anthropic API authentication failed`

**Solutions:**

1. **Check API Key Format**
   ```bash
   # Anthropic keys start with sk-ant-
   ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

2. **Verify Model Access**
   - Some models require special access
   - Check your Anthropic console for available models

## 🔧 Configuration Issues

### Problem: `Configuration file not found`

**Solutions:**

1. **Initialize ADO**
   ```bash
   ado init
   ```

2. **Check File Permissions**
   ```bash
   ls -la .ado/
   chmod 644 .ado/config.yml
   ```

3. **Validate Configuration**
   ```bash
   ado config validate
   ```

### Problem: `Invalid environment configuration`

**Solutions:**

1. **Check Required Variables**
   ```bash
   # Minimum required
   echo $GITHUB_TOKEN
   echo $OPENAI_API_KEY
   echo $GITHUB_REPO
   ```

2. **Validate .env File**
   ```bash
   # Check for syntax errors
   cat .env | grep -E '^[A-Z_]+='
   
   # Look for missing quotes around values with spaces
   cat .env | grep -E '= .* '
   ```

3. **Reset to Defaults**
   ```bash
   cp .env.example .env
   # Edit with your values
   ```

## 📝 Backlog and Task Issues

### Problem: `No backlog items found`

**Solutions:**

1. **Check Directory Structure**
   ```bash
   ls -la backlog/
   # Should contain .json files
   ```

2. **Validate JSON Syntax**
   ```bash
   # Check each JSON file
   for file in backlog/*.json; do
     echo "Checking $file"
     python -m json.tool "$file" > /dev/null || echo "Invalid JSON in $file"
   done
   ```

3. **Check File Permissions**
   ```bash
   chmod 644 backlog/*.json
   ```

### Problem: `Invalid WSJF values`

**Symptoms:**
```bash
ERROR: WSJF values must be between 1 and 10
```

**Solutions:**

1. **Validate WSJF Ranges**
   ```json
   {
     "wsjf": {
       "user_business_value": 8,     // 1-10
       "time_criticality": 6,        // 1-10
       "risk_reduction_opportunity_enablement": 5,  // 1-10
       "job_size": 3                 // 1-10
     }
   }
   ```

2. **Check for Missing Fields**
   ```bash
   # Validate required WSJF fields
   ado backlog validate
   ```

## 🚀 Agent Execution Issues

### Problem: `Agent timeout`

**Symptoms:**
```bash
ERROR: Agent execution timed out after 300 seconds
```

**Solutions:**

1. **Increase Timeout**
   ```bash
   # In .env
   AGENT_TIMEOUT=600
   
   # Or command line
   ado run --timeout 600
   ```

2. **Reduce Task Complexity**
   - Break large tasks into smaller ones
   - Simplify acceptance criteria
   - Reduce scope of changes

3. **Check System Resources**
   ```bash
   # Monitor during execution
   top -p $(pgrep -f ado)
   ```

### Problem: `Agent pipeline failure`

**Solutions:**

1. **Check Agent Logs**
   ```bash
   # Enable debug logging
   ADO_LOG_LEVEL=DEBUG ado run
   ```

2. **Test Individual Agents**
   ```bash
   # Run specific agent only
   ado run --agent planner --dry-run
   ```

3. **Reset Agent State**
   ```bash
   # Clear agent cache
   rm -rf .ado/cache/agents/
   ```

## 📋 API Rate Limiting

### Problem: `Rate limit exceeded`

**Symptoms:**
```bash
ERROR: API rate limit exceeded
Retry after: 60 seconds
```

**Solutions:**

1. **Reduce Concurrency**
   ```bash
   # In .env
   AGENT_CONCURRENCY=1
   API_RATE_LIMIT_RPM=30
   ```

2. **Implement Backoff Strategy**
   ```bash
   # Enable automatic retries
   AGENT_MAX_RETRIES=5
   RETRY_DELAY=10
   ```

3. **Upgrade API Limits**
   - OpenAI: Upgrade to higher tier
   - GitHub: Use GitHub App instead of PAT
   - Anthropic: Contact support for higher limits

## 🗁️ Logging and Debugging

### Enable Debug Logging

```bash
# Environment variable
ADO_LOG_LEVEL=DEBUG

# Command line
ado run --log-level DEBUG

# Specific components
ado run --debug github,agents,wsjf
```

### Log File Locations

```bash
# Default log locations
.ado/logs/ado.log           # Main application log
.ado/logs/agents.log        # Agent execution logs
.ado/logs/github.log        # GitHub API interactions
.ado/logs/errors.log        # Error-only log
```

### Common Log Patterns

```bash
# Search for specific errors
grep -r "ERROR" .ado/logs/

# Find authentication issues
grep -r "401\|403\|authentication" .ado/logs/

# Check API rate limiting
grep -r "rate.limit\|429" .ado/logs/

# Monitor agent execution
tail -f .ado/logs/agents.log
```

## 🔍 Performance Issues

### Problem: `Slow execution`

**Solutions:**

1. **Profile Execution**
   ```bash
   # Enable profiling
   ado run --profile
   
   # Check profile results
   cat .ado/profiles/latest.json
   ```

2. **Optimize Agent Configuration**
   ```yaml
   # .ado/agents.yml
   planner:
     max_tokens: 1000  # Reduce from 2000
     temperature: 0.3  # Increase for faster responses
   ```

3. **Enable Caching**
   ```bash
   # In .env
   CACHE_ENABLED=true
   CACHE_TTL=3600
   ```

### Problem: `High memory usage`

**Solutions:**

1. **Limit Concurrent Agents**
   ```bash
   AGENT_CONCURRENCY=1
   MAX_CONCURRENT_TASKS=3
   ```

2. **Process Fewer Items**
   ```bash
   ado run --max-items 1
   ```

3. **Clear Cache Regularly**
   ```bash
   # Add to cron job
   0 0 * * * rm -rf /path/to/project/.ado/cache/*
   ```

## 🚫 Permission Issues

### Problem: `File permission denied`

**Solutions:**

```bash
# Fix common permission issues
chmod 755 .ado/
chmod 644 .ado/*.yml
chmod 644 backlog/*.json
chmod 600 .env

# Fix ownership
sudo chown -R $USER:$USER .ado/
```

### Problem: `Git permission denied`

**Solutions:**

1. **Configure Git Credentials**
   ```bash
   git config --global user.name "ADO Bot"
   git config --global user.email "ado@yourcompany.com"
   ```

2. **Check SSH Keys**
   ```bash
   ssh -T git@github.com
   ```

3. **Use HTTPS with Token**
   ```bash
   git remote set-url origin https://$GITHUB_TOKEN@github.com/owner/repo.git
   ```

## 🌐 Network Issues

### Problem: `Connection timeout`

**Solutions:**

1. **Check Network Connectivity**
   ```bash
   # Test GitHub API
   curl -I https://api.github.com
   
   # Test OpenAI API
   curl -I https://api.openai.com
   ```

2. **Configure Proxy**
   ```bash
   # In .env
   HTTP_PROXY=http://proxy.company.com:8080
   HTTPS_PROXY=http://proxy.company.com:8080
   ```

3. **Increase Timeouts**
   ```bash
   # In .env
   HTTP_TIMEOUT=60
   AGENT_TIMEOUT=600
   ```

## 🔄 Recovery Procedures

### Complete Reset

```bash
# Backup current state
cp -r .ado .ado.backup

# Reset configuration
rm -rf .ado/
ado init

# Restore custom settings
cp .ado.backup/config.yml .ado/
```

### Partial Reset

```bash
# Clear cache only
rm -rf .ado/cache/

# Reset agent state
rm -rf .ado/agents/state/

# Clear logs
rm -rf .ado/logs/*
```

## 🔍 Getting Help

If you can't resolve the issue:

1. **Collect Diagnostics**
   ```bash
   # Generate diagnostic report
   ado diagnose > ado-diagnostics.txt
   ```

2. **Check Documentation**
   - [User Guide](../guides/)
   - [Configuration Reference](../configuration/)
   - [API Documentation](../api/)

3. **Search Issues**
   - [GitHub Issues](https://github.com/terragon-labs/agentic-dev-orchestrator/issues)
   - [Community Discussions](https://github.com/terragon-labs/agentic-dev-orchestrator/discussions)

4. **Create a Bug Report**
   - Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.yml)
   - Include diagnostic output
   - Provide minimal reproduction steps

5. **Get Community Support**
   - Join our [Discord](https://discord.gg/terragon-labs)
   - Post in [GitHub Discussions](https://github.com/terragon-labs/agentic-dev-orchestrator/discussions)

---

**Still stuck?** Don't hesitate to reach out! The ADO community is here to help. 🚀