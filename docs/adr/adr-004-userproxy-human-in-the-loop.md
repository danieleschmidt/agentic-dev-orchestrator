# ADR-004: UserProxy Human-in-the-Loop Pattern

## Status
Accepted

## Context

Autonomous development systems must handle scenarios where human intervention is necessary or beneficial:

1. **Complex Decision Making**: Ambiguous requirements or multiple valid solutions
2. **Safety Concerns**: Potentially risky changes that need human approval
3. **Policy Violations**: Changes that conflict with business rules or practices
4. **Technical Escalations**: Implementation challenges beyond agent capabilities
5. **Quality Gates**: Critical reviews that require human judgment

Several patterns were considered for implementing human-in-the-loop capabilities:

1. **Synchronous Interruption**: Stop processing and wait for human input
2. **Asynchronous Escalation**: Continue with other work while awaiting human response
3. **Approval Workflows**: Predetermined checkpoints requiring human approval
4. **AI Confidence Thresholds**: Escalate when agent confidence is low
5. **UserProxy Agent Pattern**: Dedicated agent for human interaction management

The system must balance autonomous operation with appropriate human oversight while maintaining development velocity.

## Decision

We will implement the **UserProxy Agent Pattern** from the AutoGen framework, which provides a structured approach to human-in-the-loop interactions.

**Core Components:**

1. **UserProxy Agent**: Specialized agent that manages human interactions
2. **Escalation Framework**: Systematic criteria for when to escalate
3. **Async Processing**: Non-blocking escalation handling
4. **Escalation Queue**: Persistent storage for pending human interventions
5. **Resolution Tracking**: Monitor and learn from human decisions

**Escalation Triggers:**
- Code quality violations above threshold
- Security scan failures
- Test coverage below minimum
- Large-scale changes (LOC > threshold)
- Policy rule violations
- Agent confidence below threshold
- Explicit escalation requests

## Consequences

### Positive
- **Safety Net**: Prevents autonomous system from making poor decisions
- **Human Expertise**: Leverages human judgment for complex scenarios
- **Learning Opportunity**: Human decisions improve future agent behavior
- **Risk Mitigation**: Reduces risk of autonomous system causing issues
- **Stakeholder Confidence**: Provides oversight and control mechanisms
- **Flexible Boundaries**: Escalation thresholds can be adjusted over time
- **Audit Trail**: Clear record of human interventions and decisions

### Negative
- **Velocity Impact**: Human interventions slow down autonomous processing
- **Availability Dependency**: Requires human availability for escalations
- **Inconsistent Decisions**: Different humans may make different decisions
- **Context Loss**: Human reviewers may lack full context of agent decisions
- **Escalation Fatigue**: Too many escalations reduce human engagement
- **Process Overhead**: Additional tools and processes required

## Alternatives Considered

### 1. Fully Autonomous (No Human Loop)
- **Pros**: Maximum velocity, no human dependency
- **Cons**: High risk, no safety net, stakeholder concerns

### 2. Manual Approval for All Changes
- **Pros**: Maximum oversight, human control
- **Cons**: Eliminates autonomous benefits, bottleneck

### 3. Time-based Approval Windows
- **Pros**: Automatic approval after timeout
- **Cons**: May approve changes during off-hours, inconsistent oversight

### 4. Voting-based Decisions
- **Pros**: Multiple perspectives, democratic decisions
- **Cons**: Complex coordination, slow decisions

### 5. External Workflow Tools (JIRA, ServiceNow)
- **Pros**: Integration with existing processes
- **Cons**: Complex integration, external dependencies

## Implementation Details

### UserProxy Agent Architecture
```python
class UserProxyAgent:
    def __init__(self):
        self.escalation_queue = EscalationQueue()
        self.notification_service = NotificationService()
        self.decision_tracker = DecisionTracker()
    
    def escalate(self, context: EscalationContext) -> EscalationResult:
        """Create escalation and await human decision"""
        escalation = self.escalation_queue.create(context)
        self.notification_service.notify_humans(escalation)
        return self.await_decision(escalation.id)
    
    def record_decision(self, escalation_id: str, decision: Decision) -> None:
        """Record human decision for learning"""
        self.decision_tracker.record(escalation_id, decision)
        self.update_thresholds_if_needed()
```

### Escalation Context Structure
```python
@dataclass
class EscalationContext:
    trigger_type: str  # quality|security|policy|complexity
    severity: str      # low|medium|high|critical
    item_id: str
    agent_id: str
    description: str
    code_changes: List[str]
    risk_assessment: Dict[str, Any]
    suggested_actions: List[str]
    deadline: Optional[datetime]
```

### Escalation Triggers

#### 1. Code Quality Thresholds
```python
def check_quality_escalation(changes: CodeChanges) -> bool:
    return (
        changes.complexity_score > COMPLEXITY_THRESHOLD or
        changes.lines_changed > LOC_THRESHOLD or
        changes.test_coverage < COVERAGE_THRESHOLD
    )
```

#### 2. Security Scan Results
```python
def check_security_escalation(scan_results: SecurityScan) -> bool:
    critical_vulnerabilities = scan_results.get_critical_count()
    return critical_vulnerabilities > 0
```

#### 3. Policy Violations
```python
def check_policy_escalation(changes: CodeChanges) -> bool:
    violations = PolicyEngine.check(changes)
    return len(violations) > 0
```

### Notification Mechanisms
- **Slack Integration**: Real-time notifications to development channels
- **Email Alerts**: Detailed escalation summaries to stakeholders
- **GitHub Issues**: Create tracked issues for complex escalations
- **Dashboard Updates**: Web interface showing pending escalations
- **Mobile Notifications**: Critical escalations via push notifications

### Decision Tracking and Learning
```python
class DecisionTracker:
    def record(self, escalation_id: str, decision: Decision) -> None:
        """Record human decision with context"""
        record = {
            'escalation_id': escalation_id,
            'decision': decision.action,
            'rationale': decision.rationale,
            'timestamp': datetime.now(),
            'reviewer_id': decision.reviewer_id
        }
        self.storage.save(record)
    
    def analyze_patterns(self) -> ThresholdRecommendations:
        """Analyze decisions to recommend threshold adjustments"""
        # ML analysis of historical decisions
        pass
```

### Async Processing Model
```python
async def process_with_escalation(item: BacklogItem) -> ExecutionResult:
    """Process item with potential escalation"""
    try:
        result = await agent_pipeline.process(item)
        if should_escalate(result):
            escalation = await userproxy.escalate(result.context)
            if escalation.approved:
                return await finalize_changes(result)
            else:
                return ExecutionResult(success=False, escalated=True)
        return result
    except Exception as e:
        # Automatic escalation for unexpected errors
        await userproxy.escalate(error_context(e))
        raise
```

### Configuration Management
```yaml
escalation:
  thresholds:
    complexity_score: 8
    lines_changed: 100
    test_coverage: 0.8
  
  notifications:
    slack:
      channel: "#dev-escalations"
      mention_users: ["@tech-lead", "@senior-dev"]
    
    email:
      recipients: ["team-lead@company.com"]
      template: "escalation_template.html"
  
  timeouts:
    default_response_time: "4h"
    critical_response_time: "1h"
    auto_approve_timeout: "24h"
```

### Human Interface Design
The escalation interface provides:
- **Context Summary**: Clear description of what needs review
- **Visual Diff**: Side-by-side comparison of changes
- **Risk Assessment**: Automated analysis of potential impacts
- **Recommended Actions**: Suggested next steps with rationale
- **Quick Actions**: Approve/Reject/Modify buttons
- **Feedback Collection**: Capture human rationale for decisions

This UserProxy pattern provides essential human oversight while maintaining the autonomous benefits of the system.