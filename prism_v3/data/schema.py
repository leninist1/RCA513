"""Schema registry: column mappings for Bank, Telecom, and Market systems.

Each schema maps raw CSV column names to unified names used by the engine.
"""

SYSTEM_SCHEMAS = {
    "Bank": {
        "metric_app": {
            "time_col": "timestamp",
            "entity_col": "tc",
            "value_cols": ["rr", "sr", "cnt", "mrt"],
            "type": "service",
        },
        "metric_container": {
            "time_col": "timestamp",
            "entity_col": "cmdb_id",
            "metric_col": "kpi_name",
            "value_col": "value",
            "type": "container",
        },
        "log_service": {
            "time_col": "timestamp",
            "entity_col": "cmdb_id",
            "message_col": "value",
            "log_name_col": "log_name",
            "type": "service",
        },
        "trace_span": {
            "time_col": "timestamp",
            "entity_col": "cmdb_id",
            "trace_id_col": "trace_id",
            "span_id_col": "span_id",
            "parent_id_col": "parent_id",
            "duration_col": "duration",
            "type": "service",
        },
    },
    "Telecom": {
        "metric_app": {
            "time_col": "startTime",
            "entity_col": "serviceName",
            "value_cols": ["avg_time", "num", "succee_num", "succee_rate"],
            "time_unit": "millis",
            "type": "service",
        },
        "metric_node": {
            "time_col": "timestamp",
            "entity_col": "cmdb_id",
            "metric_col": "name",
            "value_col": "value",
            "time_unit": "millis",
            "type": "node",
        },
        "metric_service": {
            "time_col": "timestamp",
            "entity_col": "cmdb_id",
            "metric_col": "name",
            "value_col": "value",
            "time_unit": "millis",
            "type": "service",
        },
        "metric_container": {
            "time_col": "timestamp",
            "entity_col": "cmdb_id",
            "metric_col": "name",
            "value_col": "value",
            "time_unit": "millis",
            "type": "container",
        },
        "metric_middleware": {
            "time_col": "timestamp",
            "entity_col": "cmdb_id",
            "metric_col": "name",
            "value_col": "value",
            "time_unit": "millis",
            "type": "middleware",
        },
        "trace_span": {
            "time_col": "startTime",
            # Telecom serviceName is empty for JDBC spans; cmdb_id carries
            # docker/container identities that align with infra GT labels.
            "entity_col": "cmdb_id",
            "trace_id_col": "traceId",
            "span_id_col": "id",
            "parent_id_col": "pid",
            "duration_col": "elapsedTime",
            "success_col": "success",
            "time_unit": "millis",
            "type": "service",
        },
    },
    "Market": {
        "metric_service": {
            "time_col": "timestamp",
            "entity_col": "service",
            "value_cols": ["rr", "sr", "mrt", "count"],
            "type": "service",
        },
        "metric_container": {
            "time_col": "timestamp",
            "entity_col": "cmdb_id",
            "metric_col": "kpi_name",
            "value_col": "value",
            "type": "container",
        },
        "metric_node": {
            "time_col": "timestamp",
            "entity_col": "cmdb_id",
            "metric_col": "kpi_name",
            "value_col": "value",
            "type": "node",
        },
        "metric_mesh": {
            "time_col": "timestamp",
            "entity_col": "cmdb_id",
            "metric_col": "kpi_name",
            "value_col": "value",
            "type": "mesh",
        },
        "metric_runtime": {
            "time_col": "timestamp",
            "entity_col": "cmdb_id",
            "metric_col": "kpi_name",
            "value_col": "value",
            "type": "runtime",
        },
        "log_service": {
            "time_col": "timestamp",
            "entity_col": "cmdb_id",
            "message_col": "value",
            "log_name_col": "log_name",
            "type": "service",
        },
        "log_proxy": {
            "time_col": "timestamp",
            "entity_col": "cmdb_id",
            "message_col": "value",
            "log_name_col": "log_name",
            "type": "proxy",
        },
        "trace_span": {
            "time_col": "timestamp",
            "entity_col": "cmdb_id",
            "trace_id_col": "trace_id",
            "span_id_col": "span_id",
            "parent_id_col": "parent_span",
            "duration_col": "duration",
            "status_code_col": "status_code",
            "type_col": "type",
            "operation_col": "operation_name",
            "type": "service",
        },
    },
}


def get_available_metric_files(system: str) -> list:
    """List metric CSV types available for a system."""
    schema = SYSTEM_SCHEMAS[system]
    return [k for k in schema if k.startswith("metric_")]


def get_available_log_files(system: str) -> list:
    """List log CSV types available for a system."""
    schema = SYSTEM_SCHEMAS[system]
    return [k for k in schema if k.startswith("log_")]


def get_available_trace_files(system: str) -> list:
    """List trace CSV types available for a system."""
    schema = SYSTEM_SCHEMAS[system]
    return [k for k in schema if k.startswith("trace_")]
