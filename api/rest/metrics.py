try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST  # type: ignore[import-not-found]

    DELIB_REQUESTS = Counter(
        "eleanor_deliberate_requests_total",
        "Total deliberate requests",
        ["outcome"],
    )
    DELIB_LATENCY = Histogram(
        "eleanor_deliberate_duration_seconds",
        "Deliberation duration in seconds",
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
    )
    OPA_CALLS = Counter(
        "eleanor_opa_calls_total",
        "Total OPA evaluations",
        ["result"],
    )
except Exception:
    DELIB_REQUESTS = None
    DELIB_LATENCY = None
    OPA_CALLS = None
    generate_latest = None
    CONTENT_TYPE_LATEST = None
