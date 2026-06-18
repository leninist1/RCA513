"""Structured LLM policy implementations for PRISM-CHT.

Provides ``StructuredLLMLeadPolicy`` and ``StructuredLLMChallengerPolicy``
that satisfy the ``LeadPolicy`` and ``ChallengerPolicy`` protocols
respectively, backed by a ``StructuredModelRequester`` that implements
limited retry on parse failures.

The policies compose a ``ModelClient`` (real or fake), deterministic
prompt builders, and strict JSON parsers.  They do not modify the
EvidenceGraph, execute tools, implement Gate logic, compute scores,
or automatically retry Gate rejections.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .action_schema import DiscriminativeAction
from .evidence_graph import EvidenceAtom, EvidenceGraph
from .tournament_types import (
    EvidenceAssessment,
    LeadNomination,
    LeadTournamentSnapshot,
)
from .challenge_types import (
    ChallengeProposal,
    ChallengeResolution,
    ChallengeSnapshot,
)
from .lead_policy import LeadPolicy, PolicyDecision
from .challenger_policy import ChallengerPolicy
from .llm_types import (
    ModelClient,
    ModelRequest,
    StructuredOutputError,
)
from .llm_json import (
    parse_evidence_assessment,
    parse_lead_policy_decision,
    parse_challenge_proposal,
    parse_challenge_resolution,
)
from .llm_prompts import (
    DEFAULT_POLICY_TOOL_NAMES,
    build_evidence_catalog,
    build_lead_decision_request,
    build_lead_assessment_request,
    build_challenge_proposal_request,
    build_challenge_resolution_request,
)
from .hypothesis import CausalHypothesis


# ===========================================================================
# StructuredModelRequester
# ===========================================================================


class StructuredModelRequester:
    """Wraps a ``ModelClient`` with limited retry on parse failures.

    Does NOT retry Gate rejections, tool exceptions, or any non-parse
    errors.  Only retries when the JSON parser raises
    ``StructuredOutputError``.
    """

    def __init__(
        self,
        *,
        client: ModelClient,
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._client = client
        self._max_attempts = max_attempts

    def request_parsed(
        self,
        *,
        request_builder: Callable[..., ModelRequest],
        parser: Callable[[str], Any],
        builder_kwargs: Mapping[str, Any],
    ) -> Any:
        """Build a request, call the model, and parse the response.

        On parse failure, retries up to ``max_attempts - 1`` additional
        times with a ``repair_error`` message.  Does not include the full
        previous output in the repair message.
        """
        last_error: StructuredOutputError | None = None

        for attempt in range(self._max_attempts):
            repair_error = None
            if attempt > 0 and last_error is not None:
                # Extract only a brief error message, not the full output
                repair_error = str(last_error)

            request = request_builder(
                attempt_index=attempt,
                repair_error=repair_error,
                **builder_kwargs,
            )

            response = self._client.complete(request=request)

            try:
                return parser(response.content)
            except StructuredOutputError as e:
                last_error = e
                continue

        # Exhausted all attempts
        assert last_error is not None
        raise StructuredOutputError(
            f"request_parsed failed after {self._max_attempts} "
            f"attempt(s); last error: {last_error}"
        ) from last_error


# ===========================================================================
# StructuredLLMLeadPolicy
# ===========================================================================


class StructuredLLMLeadPolicy:
    """LLM-backed Lead Policy satisfying the ``LeadPolicy`` protocol.

    Uses deterministic prompt builders and strict JSON parsers.
    Does not modify the EvidenceGraph, execute tools, compute scores,
    or retry Gate rejections.
    """

    def __init__(
        self,
        *,
        client: ModelClient,
        graph: EvidenceGraph,
        allowed_tool_names: Sequence[str] = DEFAULT_POLICY_TOOL_NAMES,
        max_attempts: int = 2,
    ) -> None:
        self._requester = StructuredModelRequester(
            client=client, max_attempts=max_attempts
        )
        self._graph = graph
        self._allowed_tool_names = tuple(allowed_tool_names)

    def decide_next(
        self,
        *,
        snapshot: LeadTournamentSnapshot,
    ) -> PolicyDecision:
        decision = self._requester.request_parsed(
            request_builder=build_lead_decision_request,
            parser=parse_lead_policy_decision,
            builder_kwargs={
                "snapshot": snapshot,
                "graph": self._graph,
                "allowed_tool_names": self._allowed_tool_names,
            },
        )
        assert isinstance(decision, (DiscriminativeAction, LeadNomination))
        return decision

    def assess_evidence(
        self,
        *,
        snapshot: LeadTournamentSnapshot,
        action: DiscriminativeAction,
        evidence: EvidenceAtom,
    ) -> EvidenceAssessment:
        assessment = self._requester.request_parsed(
            request_builder=build_lead_assessment_request,
            parser=parse_evidence_assessment,
            builder_kwargs={
                "snapshot": snapshot,
                "action": action,
                "evidence": evidence,
            },
        )
        assert isinstance(assessment, EvidenceAssessment)
        return assessment


# ===========================================================================
# StructuredLLMChallengerPolicy
# ===========================================================================


class StructuredLLMChallengerPolicy:
    """LLM-backed Challenger Policy satisfying the ``ChallengerPolicy`` protocol.

    Uses deterministic prompt builders and strict JSON parsers.
    Does not modify the EvidenceGraph, execute tools, compute scores,
    or retry Gate rejections.
    """

    def __init__(
        self,
        *,
        client: ModelClient,
        graph: EvidenceGraph,
        allowed_tool_names: Sequence[str] = DEFAULT_POLICY_TOOL_NAMES,
        max_attempts: int = 2,
    ) -> None:
        self._requester = StructuredModelRequester(
            client=client, max_attempts=max_attempts
        )
        self._graph = graph
        self._allowed_tool_names = tuple(allowed_tool_names)

    def propose_challenge(
        self,
        *,
        snapshot: ChallengeSnapshot,
    ) -> ChallengeProposal:
        proposal = self._requester.request_parsed(
            request_builder=build_challenge_proposal_request,
            parser=parse_challenge_proposal,
            builder_kwargs={
                "snapshot": snapshot,
                "graph": self._graph,
                "allowed_tool_names": self._allowed_tool_names,
            },
        )
        assert isinstance(proposal, ChallengeProposal)
        return proposal

    def assess_challenge(
        self,
        *,
        snapshot: ChallengeSnapshot,
        proposal: ChallengeProposal,
        evidence: EvidenceAtom,
    ) -> ChallengeResolution:
        resolution = self._requester.request_parsed(
            request_builder=build_challenge_resolution_request,
            parser=parse_challenge_resolution,
            builder_kwargs={
                "snapshot": snapshot,
                "proposal": proposal,
                "evidence": evidence,
                "graph": self._graph,
            },
        )
        assert isinstance(resolution, ChallengeResolution)
        return resolution
