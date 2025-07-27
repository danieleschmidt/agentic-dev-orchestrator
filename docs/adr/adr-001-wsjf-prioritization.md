# ADR-001: WSJF Prioritization Methodology

## Status
Accepted

## Context

The agentic development orchestrator needs a systematic way to prioritize backlog items for autonomous execution. Different prioritization methods were considered:

1. **First-In-First-Out (FIFO)**: Simple but ignores business value and urgency
2. **Manual Priority Assignment**: Requires constant human intervention
3. **Story Points Only**: Focuses on effort but ignores business value
4. **Weighted Shortest Job First (WSJF)**: Balances value, urgency, and effort

The system is designed to operate autonomously for extended periods, making manual prioritization impractical. The prioritization method must be objective, consistent, and align with business value delivery.

## Decision

We will implement the Weighted Shortest Job First (WSJF) prioritization methodology from the Scaled Agile Framework (SAFe).

**WSJF Formula:**
```
WSJF = (User-Business Value + Time Criticality + Risk Reduction & Opportunity Enablement) / Job Size
```

**Scoring Scale:**
- All factors use a 1-10 scale for consistency
- Job Size uses Fibonacci sequence (1, 2, 3, 5, 8, 13) for effort estimation
- Higher WSJF scores indicate higher priority

**Implementation Details:**
- Each backlog item must include all four WSJF factors
- Scores are calculated automatically by the BacklogManager
- Items are sorted by WSJF score for execution order
- Aging multiplier can be applied for items that remain unaddressed

## Consequences

### Positive
- **Objective Prioritization**: Removes subjective bias from task ordering
- **Business Value Alignment**: Ensures high-value work is prioritized
- **Autonomous Operation**: System can prioritize without human intervention
- **Industry Standard**: WSJF is a proven methodology in enterprise environments
- **Transparent Logic**: Stakeholders can understand why items are prioritized

### Negative
- **Initial Overhead**: Requires scoring all four factors for each backlog item
- **Learning Curve**: Teams need to understand WSJF methodology
- **Scoring Consistency**: Different people may score items differently
- **Regular Calibration**: Scores may need periodic review and adjustment

## Alternatives Considered

### 1. Simple Priority Levels (High/Medium/Low)
- **Pros**: Easy to understand and implement
- **Cons**: Too simplistic for complex prioritization needs, prone to everything being "high priority"

### 2. MoSCoW Method (Must/Should/Could/Won't)
- **Pros**: Good for feature categorization
- **Cons**: Doesn't provide fine-grained ordering within categories

### 3. Cost of Delay (CoD) Only
- **Pros**: Strong business focus
- **Cons**: Doesn't account for implementation effort, harder to calculate

### 4. Kano Model
- **Pros**: Excellent for feature classification
- **Cons**: Complex to implement, requires customer research for each item

### 5. Stack Ranking
- **Pros**: Provides clear ordering
- **Cons**: Difficult to maintain, doesn't scale well, requires manual updates

## Implementation Notes

The WSJF calculation is implemented in the `BacklogManager.calculate_wsjf()` method with the following features:

- **Input Validation**: Ensures all scores are within valid ranges
- **Aging Factor**: Optional multiplier for items that have been waiting
- **Tie Breaking**: Uses creation timestamp for items with identical WSJF scores
- **Recalculation**: WSJF scores are recalculated when backlog is loaded

Example backlog item with WSJF factors:
```json
{
  "id": "feature-123",
  "title": "Implement user authentication",
  "wsjf": {
    "user_business_value": 8,
    "time_criticality": 6,
    "risk_reduction_opportunity_enablement": 5,
    "job_size": 3
  }
}
```

This results in: WSJF = (8 + 6 + 5) / 3 = 6.33