"""Tests for fact tools, FactToolResult invariants, and ToolRegistry."""

import pytest

from prismv4.prism_cht.tool_types import (
    FactToolResult,
    validate_fact_only_payload,
)
from prismv4.prism_cht.telemetry_store import (
    MockTelemetryStore,
    OnsetObservation,
    TraceHop,
    TracePath,
)
from prismv4.prism_cht.tools.compare_onset_order import CompareOnsetOrderTool
from prismv4.prism_cht.tools.inspect_trace_path import InspectTracePathTool
from prismv4.prism_cht.tools.retrieve_raw_evidence import RetrieveRawEvidenceTool
from prismv4.prism_cht.tool_registry import (
    ToolRegistry,
    build_default_tool_registry,
)


def _make_fact_result(**overrides):
    defaults = {
        "modality": "metric",
        "component_scope": ("payment-svc",),
        "time_window": (1000.0, 2000.0),
        "observation": {"cpu_pct": 95.0},
        "provenance": {"source": "prometheus"},
    }
    defaults.update(overrides)
    return FactToolResult(**defaults)


def _make_store(
    onset_observations=(),
    trace_paths=(),
    raw_records=(),
):
    return MockTelemetryStore(
        onset_observations=onset_observations,
        trace_paths=trace_paths,
        raw_records=raw_records,
    )


# ===========================================================================
# 1-5. FactToolResult invariants
# ===========================================================================


class TestFactToolResultImmutability:
    """FactToolResult is deeply immutable."""

    def test_observation_cannot_be_mutated(self):
        r = _make_fact_result(observation={"x": 1})
        with pytest.raises(TypeError):
            r.observation["x"] = 2

    def test_provenance_cannot_be_mutated(self):
        r = _make_fact_result(provenance={"y": 1})
        with pytest.raises(TypeError):
            r.provenance["y"] = 2

    def test_nested_observation_cannot_be_mutated(self):
        r = _make_fact_result(observation={"nested": {"k": "v"}})
        with pytest.raises(TypeError):
            r.observation["nested"]["k"] = "changed"

    def test_external_dict_mutation_no_effect(self):
        obs = {"cpu_pct": 95.0, "nested": {"a": 1}}
        r = _make_fact_result(observation=obs)
        obs["cpu_pct"] = 0.0
        obs["nested"]["a"] = 999
        obs["new"] = "injected"
        assert r.observation["cpu_pct"] == 95.0
        assert r.observation["nested"]["a"] == 1
        assert "new" not in r.observation

    def test_external_provenance_mutation_no_effect(self):
        prov = {"source": "prometheus", "extra": {"b": 2}}
        r = _make_fact_result(provenance=prov)
        prov["source"] = "hacked"
        prov["extra"]["b"] = 888
        assert r.provenance["source"] == "prometheus"
        assert r.provenance["extra"]["b"] == 2


class TestFactToolResultForbiddenKeys:
    """FactToolResult rejects forbidden score/verdict keys."""

    def test_rejects_score_in_observation(self):
        with pytest.raises(ValueError, match="Forbidden key"):
            _make_fact_result(observation={"score": 0.9})

    def test_rejects_confidence_in_observation(self):
        with pytest.raises(ValueError, match="Forbidden key"):
            _make_fact_result(observation={"confidence": 0.5})

    def test_rejects_posterior_in_provenance(self):
        with pytest.raises(ValueError, match="Forbidden key"):
            _make_fact_result(provenance={"posterior": 0.8})

    def test_rejects_root_cause_in_observation(self):
        with pytest.raises(ValueError, match="Forbidden key"):
            _make_fact_result(observation={"root_cause": "payment-svc"})

    def test_rejects_nested_score(self):
        with pytest.raises(ValueError, match="Forbidden key"):
            _make_fact_result(
                observation={"metrics": {"nested": {"score": 0.5}}}
            )

    def test_rejects_nested_posterior(self):
        with pytest.raises(ValueError, match="Forbidden key"):
            _make_fact_result(
                observation={"results": [{"posterior": 0.7}]}
            )

    def test_rejects_component_score(self):
        with pytest.raises(ValueError, match="Forbidden key"):
            _make_fact_result(observation={"component_score": 10})

    def test_allows_error_rate(self):
        r = _make_fact_result(observation={"error_rate": 0.05})
        assert r.observation["error_rate"] == 0.05

    def test_allows_latency_ms(self):
        r = _make_fact_result(observation={"latency_ms": 120.0})
        assert r.observation["latency_ms"] == 120.0

    def test_allows_onset_time(self):
        r = _make_fact_result(observation={"onset_time": 1005.0})
        assert r.observation["onset_time"] == 1005.0

    def test_allows_source_component(self):
        r = _make_fact_result(observation={"source_component": "svc-a"})
        assert r.observation["source_component"] == "svc-a"

    def test_allows_target_component(self):
        r = _make_fact_result(observation={"target_component": "svc-b"})
        assert r.observation["target_component"] == "svc-b"

    def test_case_insensitive_forbidden_match(self):
        # "Score" should be caught
        with pytest.raises(ValueError, match="Forbidden key"):
            _make_fact_result(observation={"Score": 0.5})

    def test_whitespace_stripped_match(self):
        with pytest.raises(ValueError, match="Forbidden key"):
            _make_fact_result(observation={"  score  ": 0.5})


class TestFactToolResultValidation:
    """FactToolResult basic validation invariants."""

    def test_empty_modality_raises(self):
        with pytest.raises(ValueError, match="modality"):
            _make_fact_result(modality="")

    def test_empty_component_scope_raises(self):
        with pytest.raises(ValueError, match="component_scope"):
            _make_fact_result(component_scope=())

    def test_reversed_time_window_raises(self):
        with pytest.raises(ValueError, match="must be <= end"):
            _make_fact_result(time_window=(2000.0, 1000.0))

    def test_none_observation_raises(self):
        with pytest.raises(ValueError, match="observation"):
            _make_fact_result(observation=None)

    def test_none_provenance_raises(self):
        with pytest.raises(ValueError, match="provenance"):
            _make_fact_result(provenance=None)

    def test_observation_not_mapping_raises(self):
        with pytest.raises(ValueError, match="Mapping"):
            _make_fact_result(observation="not_a_map")

    def test_component_scope_dedup_sorted(self):
        r = _make_fact_result(component_scope=("C", "A", "B", "A"))
        assert r.component_scope == ("A", "B", "C")

    def test_missing_fields_dedup_sorted(self):
        r = _make_fact_result(
            missing_fields=("z", "a", "z"), observation={"x": 1}
        )
        assert r.missing_fields == ("a", "z")


# ===========================================================================
# 6-8. CompareOnsetOrderTool
# ===========================================================================


class TestCompareOnsetOrderTool:
    """CompareOnsetOrderTool returns facts, not scores."""

    def test_returns_sorted_by_onset_time(self):
        store = _make_store(
            onset_observations=[
                OnsetObservation("B", "cpu", 1200.0, "prom"),
                OnsetObservation("A", "cpu", 1000.0, "prom"),
                OnsetObservation("A", "mem", 1100.0, "prom"),
            ]
        )
        tool = CompareOnsetOrderTool()
        result = tool.execute(
            args={
                "component_scope": ["A", "B"],
                "signal_scope": ["cpu", "mem"],
                "time_window": [500.0, 2000.0],
            },
            store=store,
        )
        onsets = result.observation["observed_onsets"]
        times = [o["onset_time"] for o in onsets]
        assert times == [1000.0, 1100.0, 1200.0]

    def test_no_winner_field(self):
        store = _make_store(
            onset_observations=[
                OnsetObservation("A", "cpu", 1000.0, "prom"),
                OnsetObservation("B", "cpu", 1200.0, "prom"),
            ]
        )
        tool = CompareOnsetOrderTool()
        result = tool.execute(
            args={
                "component_scope": ["A", "B"],
                "signal_scope": ["cpu"],
                "time_window": [500.0, 2000.0],
            },
            store=store,
        )
        obs = result.observation
        assert "winner" not in obs
        assert "rank" not in obs
        assert "score" not in obs

    def test_no_score_in_keys(self):
        store = _make_store(
            onset_observations=[
                OnsetObservation("A", "cpu", 1000.0, "prom"),
            ]
        )
        tool = CompareOnsetOrderTool()
        result = tool.execute(
            args={
                "component_scope": ["A", "B"],
                "signal_scope": ["cpu"],
                "time_window": [500.0, 2000.0],
            },
            store=store,
        )
        # Validate the entire observation has no forbidden keys
        validate_fact_only_payload(result.observation)

    def test_missing_components_listed(self):
        store = _make_store(
            onset_observations=[
                OnsetObservation("A", "cpu", 1000.0, "prom"),
            ]
        )
        tool = CompareOnsetOrderTool()
        result = tool.execute(
            args={
                "component_scope": ["A", "B", "C"],
                "signal_scope": ["cpu"],
                "time_window": [500.0, 2000.0],
            },
            store=store,
        )
        assert "B" in result.observation["missing_components"]
        assert "C" in result.observation["missing_components"]
        assert "A" not in result.observation["missing_components"]

    def test_all_components_present_no_missing(self):
        store = _make_store(
            onset_observations=[
                OnsetObservation("A", "cpu", 1000.0, "prom"),
                OnsetObservation("B", "cpu", 1100.0, "prom"),
            ]
        )
        tool = CompareOnsetOrderTool()
        result = tool.execute(
            args={
                "component_scope": ["A", "B"],
                "signal_scope": ["cpu"],
                "time_window": [500.0, 2000.0],
            },
            store=store,
        )
        assert result.observation["missing_components"] == ()

    def test_requires_at_least_two_components(self):
        tool = CompareOnsetOrderTool()
        with pytest.raises(ValueError, match="at least two"):
            tool.execute(
                args={
                    "component_scope": ["A"],
                    "signal_scope": ["cpu"],
                    "time_window": [500.0, 2000.0],
                },
                store=_make_store(),
            )

    def test_requires_at_least_one_signal(self):
        tool = CompareOnsetOrderTool()
        with pytest.raises(ValueError, match="at least one signal"):
            tool.execute(
                args={
                    "component_scope": ["A", "B"],
                    "signal_scope": [],
                    "time_window": [500.0, 2000.0],
                },
                store=_make_store(),
            )


# ===========================================================================
# 9-10. InspectTracePathTool
# ===========================================================================


class TestInspectTracePathTool:
    """InspectTracePathTool returns only explicit paths."""

    def test_returns_explicit_paths_only(self):
        store = _make_store(
            trace_paths=[
                TracePath(
                    hops=(
                        TraceHop("A", "B", 1000.0, 5.0, "ok"),
                        TraceHop("B", "C", 1005.0, 3.0, "ok"),
                    )
                ),
            ]
        )
        tool = InspectTracePathTool()
        result = tool.execute(
            args={
                "source_component": "A",
                "target_component": "C",
                "time_window": [500.0, 2000.0],
                "max_hops": 4,
                "max_paths": 10,
            },
            store=store,
        )
        paths = result.observation["paths"]
        assert len(paths) == 1
        assert len(paths[0]["hops"]) == 2
        assert paths[0]["hops"][0]["source_component"] == "A"
        assert paths[0]["hops"][1]["target_component"] == "C"

    def test_no_fake_edges_when_no_match(self):
        store = _make_store(
            trace_paths=[
                TracePath(
                    hops=(TraceHop("X", "Y", 1000.0, 1.0, "ok"),)
                ),
            ]
        )
        tool = InspectTracePathTool()
        result = tool.execute(
            args={
                "source_component": "A",
                "target_component": "B",
                "time_window": [500.0, 2000.0],
                "max_hops": 4,
                "max_paths": 10,
            },
            store=store,
        )
        assert result.observation["paths"] == ()

    def test_no_propagation_score_field(self):
        store = _make_store(
            trace_paths=[
                TracePath(
                    hops=(TraceHop("A", "B", 1000.0, 5.0, "ok"),)
                ),
            ]
        )
        tool = InspectTracePathTool()
        result = tool.execute(
            args={
                "source_component": "A",
                "target_component": "B",
                "time_window": [500.0, 2000.0],
                "max_hops": 4,
                "max_paths": 10,
            },
            store=store,
        )
        validate_fact_only_payload(result.observation)

    def test_empty_source_raises(self):
        tool = InspectTracePathTool()
        with pytest.raises(ValueError, match="source_component"):
            tool.execute(
                args={
                    "source_component": "",
                    "target_component": "B",
                    "time_window": [500.0, 2000.0],
                },
                store=_make_store(),
            )

    def test_empty_target_raises(self):
        tool = InspectTracePathTool()
        with pytest.raises(ValueError, match="target_component"):
            tool.execute(
                args={
                    "source_component": "A",
                    "target_component": "",
                    "time_window": [500.0, 2000.0],
                },
                store=_make_store(),
            )

    def test_default_max_hops(self):
        store = _make_store(
            trace_paths=[
                TracePath(
                    hops=(
                        TraceHop("A", "B", 1000.0, 1.0, "ok"),
                        TraceHop("B", "C", 1001.0, 1.0, "ok"),
                        TraceHop("C", "D", 1002.0, 1.0, "ok"),
                    )
                ),
            ]
        )
        tool = InspectTracePathTool()
        result = tool.execute(
            args={
                "source_component": "A",
                "target_component": "D",
                "time_window": [500.0, 2000.0],
            },
            store=store,
        )
        # 3 hops <= default max_hops=4, should be found
        assert len(result.observation["paths"]) == 1


# ===========================================================================
# 11-12. RetrieveRawEvidenceTool
# ===========================================================================


class TestRetrieveRawEvidenceTool:
    """RetrieveRawEvidenceTool filters correctly and respects limits."""

    def test_filters_by_modality(self):
        store = _make_store(
            raw_records=[
                {"modality": "metric", "component": "A", "timestamp": 1000.0, "payload": {"v": 1}},
                {"modality": "log", "component": "A", "timestamp": 1001.0, "payload": {"msg": "err"}},
            ]
        )
        tool = RetrieveRawEvidenceTool()
        result = tool.execute(
            args={
                "modality": "metric",
                "component_scope": ["A"],
                "time_window": [500.0, 2000.0],
                "limit": 10,
            },
            store=store,
        )
        records = result.observation["records"]
        assert len(records) == 1
        assert records[0]["modality"] == "metric"

    def test_filters_by_component_scope(self):
        store = _make_store(
            raw_records=[
                {"modality": "metric", "component": "A", "timestamp": 1000.0, "payload": {"v": 1}},
                {"modality": "metric", "component": "B", "timestamp": 1001.0, "payload": {"v": 2}},
                {"modality": "metric", "component": "C", "timestamp": 1002.0, "payload": {"v": 3}},
            ]
        )
        tool = RetrieveRawEvidenceTool()
        result = tool.execute(
            args={
                "modality": "metric",
                "component_scope": ["A", "B"],
                "time_window": [500.0, 2000.0],
                "limit": 10,
            },
            store=store,
        )
        records = result.observation["records"]
        components = {r["component"] for r in records}
        assert components == {"A", "B"}

    def test_filters_by_time_window(self):
        store = _make_store(
            raw_records=[
                {"modality": "metric", "component": "A", "timestamp": 500.0, "payload": {"v": 1}},
                {"modality": "metric", "component": "A", "timestamp": 1000.0, "payload": {"v": 2}},
                {"modality": "metric", "component": "A", "timestamp": 1500.0, "payload": {"v": 3}},
            ]
        )
        tool = RetrieveRawEvidenceTool()
        result = tool.execute(
            args={
                "modality": "metric",
                "component_scope": ["A"],
                "time_window": [501.0, 1499.0],
                "limit": 10,
            },
            store=store,
        )
        records = result.observation["records"]
        assert len(records) == 1
        assert records[0]["timestamp"] == 1000.0

    def test_respects_limit(self):
        store = _make_store(
            raw_records=[
                {"modality": "metric", "component": "A", "timestamp": float(t), "payload": {"v": t}}
                for t in range(100)
            ]
        )
        tool = RetrieveRawEvidenceTool()
        result = tool.execute(
            args={
                "modality": "metric",
                "component_scope": ["A"],
                "time_window": [0.0, 200.0],
                "limit": 5,
            },
            store=store,
        )
        assert len(result.observation["records"]) == 5

    def test_rejects_limit_over_50(self):
        tool = RetrieveRawEvidenceTool()
        with pytest.raises(ValueError, match="limit"):
            tool.execute(
                args={
                    "modality": "metric",
                    "component_scope": ["A"],
                    "time_window": [0.0, 100.0],
                    "limit": 51,
                },
                store=_make_store(),
            )

    def test_no_score_in_observation(self):
        store = _make_store(
            raw_records=[
                {"modality": "metric", "component": "A", "timestamp": 1000.0, "payload": {"v": 1}},
            ]
        )
        tool = RetrieveRawEvidenceTool()
        result = tool.execute(
            args={
                "modality": "metric",
                "component_scope": ["A"],
                "time_window": [500.0, 2000.0],
                "limit": 10,
            },
            store=store,
        )
        validate_fact_only_payload(result.observation)


# ===========================================================================
# 13-15. ToolRegistry
# ===========================================================================


class TestToolRegistry:
    """ToolRegistry registration and execution."""

    def test_rejects_duplicate_registration(self):
        reg = ToolRegistry()
        tool = CompareOnsetOrderTool()
        reg.register(tool)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(CompareOnsetOrderTool())

    def test_get_unregistered_raises(self):
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="not registered"):
            reg.get("nonexistent")

    def test_execute_unregistered_raises(self):
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="not registered"):
            reg.execute(
                tool_name="nonexistent",
                args={},
                store=_make_store(),
            )

    def test_has_returns_correctly(self):
        reg = ToolRegistry()
        assert not reg.has("compare_onset_order")
        reg.register(CompareOnsetOrderTool())
        assert reg.has("compare_onset_order")

    def test_build_default_has_fact_tools(self):
        reg = build_default_tool_registry()
        assert reg.has("compare_onset_order")
        assert reg.has("inspect_trace_path")
        assert reg.has("retrieve_raw_evidence")
        assert reg.has("inspect_reason_signature")
        assert reg.has("check_propagation_consistency")
        assert reg.has("find_unexplained_symptoms")

    def test_registry_execute_returns_fact_result(self):
        reg = ToolRegistry()
        reg.register(CompareOnsetOrderTool())
        store = _make_store(
            onset_observations=[
                OnsetObservation("A", "cpu", 1000.0, "prom"),
                OnsetObservation("B", "cpu", 1100.0, "prom"),
            ]
        )
        result = reg.execute(
            tool_name="compare_onset_order",
            args={
                "component_scope": ["A", "B"],
                "signal_scope": ["cpu"],
                "time_window": [500.0, 2000.0],
            },
            store=store,
        )
        assert isinstance(result, FactToolResult)
