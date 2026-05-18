"""Prompt templates for Meta-Controller Option C (LLM)."""

SYSTEM_PROMPT = """You are a meta-controller for a microservice Root Cause Analysis (RCA) system.
Your role: given the current investigation state, decide the next action to efficiently find the root cause.

AVAILABLE STRATEGIES:
- A (broad_shallow): Verify many candidates cheaply. Use when signal is diffuse (many services show mild degradation). Cost: minimal compute only, 0 API calls.
- B (deep_dive): Full multi-modal verification on 1-2 candidates. Use when signal is concentrated (1-2 services show severe anomaly). Cost: 1 LLM call + compute.
- C (causal_trace): Trace along call dependency chains. Use when traces show topological propagation patterns. Cost: compute only, 0 API calls.
- D (log_compare): Log keyword analysis + time alignment. Use when log anomalies are present and text-rich. Cost: 0-1 LLM call. NOTE: only available if the system has logs.

CONSTRAINTS:
- Each LLM call costs budget. Use sparingly (max ~3-5 calls per query including both controller and strategy calls).
- Strategy A is free. Default to A unless there is strong evidence for another.
- Stop when confidence > 0.85 or budget is nearly depleted.
- You should aim to reach a confident conclusion in 3-5 iterations total.

OUTPUT FORMAT - respond with EXACTLY this JSON:
{"strategy": "A|B|C|D", "depth": "shallow|medium|deep", "stop": false, "reasoning": "One sentence explaining the choice", "next_candidates": ["entity1", "entity2"]}

When stop is true, include a final_answer with your best prediction in this format:
{"strategy": "A", "depth": "shallow", "stop": true, "reasoning": "Confident in root cause", "final_answer": {"component": "ServiceName", "reason": "failure reason", "time": "YYYY-MM-DD HH:MM:SS"}, "next_candidates": []}"""


USER_PROMPT_TEMPLATE = """## Current Investigation State (Iteration {iteration})

### Top Candidates (ranked by confidence)
{candidates_table}

### Evidence Summary
{evidence_summary}

### Case Features
- Task type: {task_type} (1=time, 2=reason, 3=component, 4=time+reason, 5=time+component, 6=component+reason, 7=time+component+reason)
- System: {system_name}
- Fault window: {time_range}
- Number of candidate entities: {num_candidates}
- Trace data available: {has_traces}
- Log data available: {has_logs}
- Number of entities in system: {num_entities}

### Budget
- API calls used: {calls_used}/{calls_total}
- Strategies executed so far: {strategy_history}
- Iteration: {iteration}/{max_iterations}

### Instruction
Given the above state, what is the best next action to find the root cause?
Output your decision as the specified JSON format."""


def build_prompt(state) -> str:
    """Build user prompt from controller state."""
    candidates = state.top_candidates(min(7, len(state.candidate_scores)))

    # Format candidates table
    lines = []
    for i, c in enumerate(candidates):
        ev_count = len(state.evidence_pool.get(c.entity, []))
        depth = "none"
        for ev in state.evidence_pool.get(c.entity, []):
            if isinstance(ev, dict) and ev.get("type", "").startswith(("deep", "shallow", "causal", "log", "llm")):
                depth = ev["type"]
        lines.append(f"{i+1}. {c.entity:<25} score={c.recovery_score or 0:.4f}  depth={depth}  evidence_items={ev_count}")

    candidates_table = "\n".join(lines) if lines else "(no candidates yet)"

    # Format evidence summary
    evidence_parts = []
    for entity, ev_list in state.evidence_pool.items():
        if not ev_list:
            continue
        scores = []
        for ev in ev_list:
            if isinstance(ev, dict) and "score" in ev:
                scores.append(f"{ev.get('type', '?'):.15} = {ev['score']:.4f}")
        if scores:
            evidence_parts.append(f"  {entity}: {' | '.join(scores[:3])}")
    evidence_summary = "\n".join(evidence_parts[:10]) if evidence_parts else "(no evidence collected yet)"

    return USER_PROMPT_TEMPLATE.format(
        iteration=state.iteration,
        max_iterations=8,
        candidates_table=candidates_table,
        evidence_summary=evidence_summary,
        task_type=state.case_features.get("task_type", "?"),
        system_name=state.case_features.get("system", "?"),
        time_range=state.case_features.get("time_range", "?"),
        num_candidates=len(state.candidate_scores),
        has_traces=state.case_features.get("has_traces", False),
        has_logs=state.case_features.get("has_logs", False),
        num_entities=state.case_features.get("num_entities", 0),
        calls_used=state.budget.api_calls_used,
        calls_total=state.budget.total_api_calls,
        strategy_history=", ".join(state.strategy_history) if state.strategy_history else "none",
    )
