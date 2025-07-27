# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the Agentic Development Orchestrator project.

## What are ADRs?

Architecture Decision Records are documents that capture important architectural decisions made along with their context and consequences. They help teams understand why certain decisions were made and provide historical context for future changes.

## ADR Format

Each ADR follows this structure:

```markdown
# ADR-XXX: [Title]

## Status
[Proposed | Accepted | Deprecated | Superseded]

## Context
[What is the issue that we're seeing that is motivating this decision or change?]

## Decision
[What is the change that we're proposing and/or doing?]

## Consequences
[What becomes easier or more difficult to do because of this change?]

## Alternatives Considered
[What other options were considered and why were they rejected?]
```

## Index of ADRs

- [ADR-001: WSJF Prioritization Methodology](./adr-001-wsjf-prioritization.md)
- [ADR-002: Agent Pipeline Architecture](./adr-002-agent-pipeline-architecture.md)
- [ADR-003: File-based Persistence](./adr-003-file-based-persistence.md)
- [ADR-004: UserProxy Human-in-the-Loop Pattern](./adr-004-userproxy-human-in-the-loop.md)

## Guidelines

1. **Create an ADR for any significant architectural decision**
   - Technology choices
   - Integration patterns
   - Data models
   - Security approaches

2. **Keep ADRs concise but complete**
   - Focus on the decision and rationale
   - Include enough context for future readers
   - Document alternatives considered

3. **Update the index when adding new ADRs**
   - Use sequential numbering
   - Include descriptive titles
   - Link to the actual ADR file

4. **Mark superseded ADRs clearly**
   - Update status to "Superseded"
   - Link to the replacing ADR
   - Keep historical record intact