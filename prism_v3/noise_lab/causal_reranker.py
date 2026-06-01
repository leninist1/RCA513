"""Semantic causal chain reranker for Noise Lab top-K candidates.

Reframes RCA from "score each node" to "verify which causal chain is real":
1. Pass query instruction + top-K candidates + their reasons to an LLM.
2. LLM emits 3-5 causal chain hypotheses (e.g. ["Tomcat02:OOM", "MG02:thread pool full", "MySQL01:connection wait"]).
3. Each hypothesis is graph-and-temporal-validated WITHOUT a second LLM call:
   - Adjacency: do consecutive nodes appear as caller→callee in the graph?
   - Temporal monotonicity: does earliest_timestamp grow along the chain?
   - Mechanism compatibility: do reasons follow plausible cause→effect?
4. Each candidate inherits a chain-origin vote; top-K is reranked by combining
   the original score with the chain vote.

The validator catches LLM hallucinations: a fabricated chain that disagrees with
the graph or timestamps gets a near-zero validation score and contributes nothing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..mace.graph import ObjectGraph, _canonical_object_name
from ..utils.llm_client import LLMClient


_MECH_AFFINITY: Dict[Tuple[str, str], float] = {
    ("memory", "memory"): 1.0,
    ("memory", "latency"): 0.7,
    ("memory", "process"): 0.7,
    ("cpu", "cpu"): 1.0,
    ("cpu", "latency"): 0.7,
    ("cpu", "process"): 0.7,
    ("network", "network"): 1.0,
    ("network", "latency"): 0.9,
    ("network", "db"): 0.6,
    ("disk", "disk"): 1.0,
    ("disk", "db"): 0.7,
    ("disk", "latency"): 0.5,
    ("db", "db"): 1.0,
    ("db", "latency"): 0.8,
    ("db", "network"): 0.4,
    ("process", "process"): 1.0,
    ("process", "latency"): 0.7,
    ("change", "change"): 1.0,
}


def _classify_reason(reason: str) -> str:
    text = (reason or "").lower()
    if "oom" in text or "memory" in text:
        return "memory"
    if "cpu" in text:
        return "cpu"
    if "network" in text or "packet" in text or "loss" in text or "latency" in text or "delay" in text:
        return "network" if "packet" in text or "loss" in text else "latency"
    if "disk" in text or "i/o" in text or "io" in text:
        return "disk"
    if "db" in text or "database" in text or "connection limit" in text or "db close" in text:
        return "db"
    if "process" in text or "termination" in text or "thread" in text:
        return "process"
    if "change" in text:
        return "change"
    return "memory"


@dataclass
class CausalChain:
    nodes: List[str]
    explanation: str
    raw_score: float = 0.0
    graph_score: float = 0.0
    temporal_score: float = 0.0
    mechanism_score: float = 0.0
    validation_score: float = 0.0


class SemanticCausalChainReranker:
    """LLM-proposed chains, graph-validated, no chain-level LLM calls beyond proposal."""

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "deepseek-chat",
        top_k: int = 15,
        n_chains: int = 5,
        max_chain_len: int = 4,
        chain_weight: float = 0.25,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> None:
        self.client = llm_client
        self.model = model
        self.top_k = top_k
        self.n_chains = n_chains
        self.max_chain_len = max_chain_len
        self.chain_weight = chain_weight
        self.temperature = temperature
        self.max_tokens = max_tokens

    def rerank(
        self,
        instruction: str,
        graph: ObjectGraph,
        ranking: List[Dict[str, Any]],
        entity_types: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        debug: Dict[str, Any] = {
            "n_candidates": min(len(ranking), self.top_k),
            "chains_proposed": 0,
            "chains_validated": 0,
        }
        if not ranking or self.client is None or not self.client.api_key:
            return ranking, debug

        # Confidence gate: if rank-1 dominates rank-2 by a wide margin, the
        # original scorer is already very confident — reranking adds noise.
        if len(ranking) >= 2:
            s1 = float(ranking[0].get("score", 0.0))
            s2 = float(ranking[1].get("score", 0.0))
            denom = max(abs(s1), 1e-6)
            if (s1 - s2) / denom >= 0.40:
                debug["skipped_confident_top1"] = True
                debug["margin"] = round((s1 - s2) / denom, 3)
                return ranking, debug

        head = ranking[: self.top_k]
        tail = ranking[self.top_k :]

        prompt = self._build_prompt(instruction, head, graph, entity_types or {})
        chains = self._call_llm(prompt)
        debug["chains_proposed"] = len(chains)
        if not chains:
            return ranking, debug

        validated = []
        for chain in chains:
            self._validate_chain(chain, graph)
            validated.append(chain)
        debug["chains_validated"] = sum(1 for c in validated if c.validation_score > 0)
        debug["chain_summaries"] = [
            {
                "nodes": c.nodes,
                "raw": round(c.raw_score, 3),
                "graph": round(c.graph_score, 3),
                "temporal": round(c.temporal_score, 3),
                "mechanism": round(c.mechanism_score, 3),
                "validated": round(c.validation_score, 3),
                "explanation": c.explanation[:200],
            }
            for c in validated
        ]

        # Adaptive gate: if no chain has meaningful evidence (typically a fully
        # disconnected graph with no timestamps), skip reranking. The LLM
        # cannot help here; trust the original scorer.
        max_validation = max((c.validation_score for c in validated), default=0.0)
        if max_validation < 0.30:
            debug["skipped_low_evidence"] = True
            debug["max_validation"] = round(max_validation, 3)
            return ranking, debug

        votes = self._aggregate_votes(validated, head)
        head_reranked = self._merge_scores(head, votes)
        ranking_new = head_reranked + tail
        ranking_new.sort(key=lambda x: x["score"], reverse=True)
        return ranking_new, debug

    def _build_prompt(
        self,
        instruction: str,
        candidates: List[Dict[str, Any]],
        graph: ObjectGraph,
        entity_types: Dict[str, str],
    ) -> str:
        cand_lines = []
        for i, c in enumerate(candidates, 1):
            obj_id = str(c.get("object_id", ""))
            entity = str(c.get("entity", ""))
            reason = str(c.get("reason", ""))
            etype = entity_types.get(entity, entity_types.get(obj_id, "?"))
            score = float(c.get("score", 0.0))
            n = c.get("noise", {}) or {}
            s = c.get("structure", {}) or {}
            cand_lines.append(
                f"  {i:>2}. id={obj_id} type={etype} reason='{reason}' "
                f"score={score:.3f} root_src={n.get('root_source_score',0):.2f} "
                f"struct={s.get('structural_score',0):.2f} hotspot={n.get('hotspot_bias',0):.2f}"
            )

        edges = []
        cand_ids = {str(c.get("object_id", "")) for c in candidates}
        for src, dests in graph.adjacency.items():
            if src not in cand_ids:
                continue
            for dst in dests:
                if dst in cand_ids:
                    edges.append(f"  {src} -> {dst}")
        edge_text = "\n".join(edges[:40]) if edges else "  (no edges among candidates — graph is disconnected)"

        return (
            f"You are an SRE doing root cause analysis. Below is a fault investigation.\n\n"
            f"INSTRUCTION:\n{instruction}\n\n"
            f"TOP CANDIDATES (already pre-ranked by an anomaly+structural scorer; rank 1 may be a hub artifact):\n"
            f"{chr(10).join(cand_lines)}\n\n"
            f"GRAPH EDGES AMONG CANDIDATES (caller -> callee, fault propagates upstream from callee to caller):\n"
            f"{edge_text}\n\n"
            f"TASK: identify the SINGLE most likely causal chain explaining this fault, and at most "
            f"{self.n_chains - 2} alternative hypotheses you have lower but real confidence in. "
            f"Each chain is an ordered list of 2 to {self.max_chain_len} candidate IDs from the list above, "
            f"where the FIRST element is the suspected ROOT CAUSE and subsequent elements are downstream EFFECTS. "
            f"\n\nGuidance:\n"
            f"- Only include chains where you have GRAPH or TEMPORAL evidence — no speculation\n"
            f"- DO NOT mechanically start every chain with the highest-scored candidate; the scorer is biased toward hubs\n"
            f"- Real root causes often appear at rank 3-10, not rank 1\n"
            f"- Quality over quantity: 2 confident chains beat 5 speculative ones\n\n"
            f"Output STRICT JSON only, no prose:\n"
            f'{{"chains":[{{"nodes":["id1","id2",...],"explanation":"..."}},...]}}'
        )

    def _call_llm(self, prompt: str) -> List[CausalChain]:
        try:
            resp = self.client.call(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception:
            return []
        if not resp or resp.content is None:
            return []
        # LLMClient returns parsed dict when response_format is json_object;
        # fall back to manual extraction if a string came back.
        if isinstance(resp.content, dict):
            data = resp.content
        else:
            text = str(resp.content).strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return []
            try:
                data = json.loads(match.group(0))
            except Exception:
                return []
        chains_raw = data.get("chains", [])
        out: List[CausalChain] = []
        for entry in chains_raw[: self.n_chains]:
            nodes = entry.get("nodes") or []
            if not isinstance(nodes, list) or len(nodes) < 2:
                continue
            nodes = [str(n).strip() for n in nodes if n][: self.max_chain_len]
            if len(nodes) < 2:
                continue
            explanation = str(entry.get("explanation", "")).strip()
            out.append(CausalChain(nodes=nodes, explanation=explanation))
        return out

    def _validate_chain(self, chain: CausalChain, graph: ObjectGraph) -> None:
        if len(chain.nodes) < 2:
            chain.validation_score = 0.0
            return

        ids: List[Optional[str]] = []
        for n in chain.nodes:
            ids.append(self._resolve_node(n, graph))

        present = sum(1 for x in ids if x is not None)
        chain.raw_score = present / len(ids)
        if present < 2:
            chain.validation_score = 0.0
            return

        # Graph adjacency is caller -> callee, while chains are root -> effect.
        # For upstream propagation, effect -> root is the strongest evidence.
        # A root -> effect edge is kept as weak evidence because some resource
        # faults can propagate downstream, but it should not validate a chain
        # as strongly as the expected reverse call direction.
        edge_score_sum = 0.0
        edge_total = 0
        for a, b in zip(ids, ids[1:]):
            if a is None or b is None:
                continue
            edge_total += 1
            root_calls_effect = b in graph.adjacency.get(a, {})
            effect_calls_root = a in graph.adjacency.get(b, {})
            if effect_calls_root:
                edge_score_sum += 1.0
            elif root_calls_effect:
                edge_score_sum += 0.35
        chain.graph_score = (edge_score_sum / edge_total) if edge_total else 0.0

        # Temporal monotonicity: earliest timestamp should be earliest at root.
        # Pairs with missing timestamps contribute 0, not a 0.5 default — there
        # is nothing to validate against, so the LLM's claim earns no credit.
        ts = [
            graph.nodes[i].earliest_timestamp if i and i in graph.nodes else None
            for i in ids
        ]
        valid_pairs = 0
        mono_pairs = 0
        for a, b in zip(ts, ts[1:]):
            if a is None or b is None:
                continue
            valid_pairs += 1
            if a <= b:
                mono_pairs += 1
        chain.temporal_score = (mono_pairs / valid_pairs) if valid_pairs else 0.0

        # Mechanism compatibility along the chain.
        reasons = [
            _classify_reason(graph.nodes[i].best_reason()) if i and i in graph.nodes else None
            for i in ids
        ]
        affinity_sum = 0.0
        affinity_count = 0
        for a, b in zip(reasons, reasons[1:]):
            if a is None or b is None:
                continue
            affinity_count += 1
            affinity_sum += _MECH_AFFINITY.get((a, b), 0.3)
        chain.mechanism_score = (affinity_sum / affinity_count) if affinity_count else 0.5

        # Evidence-coverage gate: if neither graph edges nor timestamps are
        # available, mechanism alone is too weak to act on — refuse the chain.
        evidence_count = (1 if edge_total > 0 else 0) + (1 if valid_pairs > 0 else 0)
        if evidence_count == 0:
            chain.validation_score = 0.0
            return

        chain.validation_score = (
            0.55 * chain.graph_score
            + 0.30 * chain.temporal_score
            + 0.15 * chain.mechanism_score
        ) * chain.raw_score

    def _resolve_node(self, name: str, graph: ObjectGraph) -> Optional[str]:
        if not name:
            return None
        if name in graph.nodes:
            return name
        canonical = _canonical_object_name(name)
        if canonical in graph.nodes:
            return canonical
        # Best-effort: substring/canonical match across nodes.
        for nid in graph.nodes:
            if nid == canonical or _canonical_object_name(nid) == canonical:
                return nid
        for nid in graph.nodes:
            if canonical and (canonical in nid or nid in canonical):
                return nid
        return None

    def _aggregate_votes(
        self,
        chains: List[CausalChain],
        head: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        votes: Dict[str, float] = {str(c.get("object_id", "")): 0.0 for c in head}
        if not chains:
            return votes
        # Use only the K-best chains (by validation), and require a meaningful
        # spread to avoid "every chain wins, every node wins" averaging.
        scored = sorted(chains, key=lambda c: c.validation_score, reverse=True)
        if not scored or scored[0].validation_score <= 0:
            return votes
        # Take chains within 0.85x of the best, capped at 3, to keep votes sharp.
        cutoff = scored[0].validation_score * 0.85
        elite = [c for c in scored if c.validation_score >= cutoff][:3]
        total_strength = sum(c.validation_score for c in elite)
        if total_strength <= 0:
            return votes
        head_id_set = {str(c.get("object_id", "")) for c in head}
        head_canon_to_id = {_canonical_object_name(cid): cid for cid in head_id_set}
        head_lower_to_id = {cid.lower(): cid for cid in head_id_set}

        def resolve(raw: str) -> Optional[str]:
            if raw in head_id_set:
                return raw
            lo = str(raw).lower()
            if lo in head_lower_to_id:
                return head_lower_to_id[lo]
            return head_canon_to_id.get(_canonical_object_name(raw))

        # First pass: collect which candidates ever appear as a chain head.
        ever_root: set = set()
        for chain in elite:
            head_node = resolve(chain.nodes[0]) if chain.nodes else None
            if head_node is not None:
                ever_root.add(head_node)

        # Second pass: distribute positive vote to chain heads, symptom penalty
        # only to nodes that NEVER appear as a chain head in the elite set —
        # this way a candidate cited as root by *any* chain is protected from
        # being demoted by other chains that happen to mention it downstream.
        for chain in elite:
            for position, raw in enumerate(chain.nodes):
                candidate_id = resolve(raw)
                if candidate_id is None:
                    continue
                if position == 0:
                    votes[candidate_id] = votes.get(candidate_id, 0.0) + (
                        chain.validation_score / total_strength
                    )
                else:
                    if candidate_id in ever_root:
                        continue
                    symptom_penalty = 0.12 * (chain.validation_score / total_strength)
                    votes[candidate_id] = votes.get(candidate_id, 0.0) - symptom_penalty
        # Normalize positive votes to [0, 1] (negatives kept as-is for symptoms).
        positive_max = max((v for v in votes.values() if v > 0), default=0.0)
        if positive_max > 0:
            votes = {
                k: (v / positive_max if v > 0 else max(-1.0, v))
                for k, v in votes.items()
            }
        return votes

    def _merge_scores(
        self,
        head: List[Dict[str, Any]],
        votes: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        if not head:
            return head
        scores = [float(c.get("score", 0.0)) for c in head]
        smin, smax = min(scores), max(scores)
        norm_range = smax - smin if smax > smin else 1.0
        out = []
        for c in head:
            obj_id = str(c.get("object_id", ""))
            base_norm = (float(c.get("score", 0.0)) - smin) / norm_range
            vote = votes.get(obj_id, 0.0)
            new_score = (1.0 - self.chain_weight) * base_norm + self.chain_weight * vote
            new_c = dict(c)
            new_c["chain_vote"] = round(vote, 4)
            new_c["pre_chain_score"] = round(float(c.get("score", 0.0)), 4)
            new_c["score"] = round(new_score, 6)
            out.append(new_c)
        return out
