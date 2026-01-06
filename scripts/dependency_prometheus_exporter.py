"""
Lightweight exporter that polls /admin/dependencies and exposes
the same metrics via Prometheus for sidecar deployments.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from urllib import error, request

from prometheus_client import Counter, Gauge, start_http_server  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


FAILURE_COUNTER = Counter(
    "eleanor_dependency_failures_total",
    "Count of dependency load failures",
    ["dependency"],
)

LAST_FAILURE_TS = Gauge(
    "eleanor_dependency_failure_last_timestamp_seconds",
    "Timestamp of the last failure for a dependency",
    ["dependency"],
)

SCRAPED_SUCCESS = Counter(
    "eleanor_dependency_failure_exporter_scrapes_total",
    "Exporter scrape successes",
)

SCRAPED_FAILURE = Counter(
    "eleanor_dependency_failure_exporter_errors_total",
    "Exporter scrape failures",
)


def fetch_metrics(endpoint: str, timeout: float):
    try:
        with request.urlopen(endpoint, timeout=timeout) as response:
            return json.loads(response.read())
    except error.HTTPError as exc:
        logger.warning("dependency exporter got HTTP error %s", exc)
    except error.URLError as exc:
        logger.warning("dependency exporter could not reach %s: %s", endpoint, exc)
    except Exception:
        logger.exception("dependency exporter fetch failed")
    return None


def update_prometheus(metrics: dict):
    failures = metrics.get("failures") or {}
    for dependency, info in failures.items():
        try:
            FAILURE_COUNTER.labels(dependency=dependency).inc(info.get("count", 0))
        except Exception:
            logger.warning("failed to update counter for %s", dependency)
        last_failure = info.get("last_failure")
        if last_failure:
            try:
                ts = datetime.fromisoformat(last_failure).timestamp()
            except ValueError:
                logger.debug("parsing timestamp failed; using now")
                ts = time.time()
            LAST_FAILURE_TS.labels(dependency=dependency).set(ts)


def run_exporter(args):
    start_http_server(args.listen_port, addr=args.listen_addr)
    logger.info("dependency exporter listening on %s:%s", args.listen_addr, args.listen_port)

    while True:
        payload = fetch_metrics(args.admin_endpoint, args.timeout)
        if payload:
            SCRAPED_SUCCESS.inc()
            update_prometheus(payload)
        else:
            SCRAPED_FAILURE.inc()
        time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description="Export dependency health to Prometheus")
    parser.add_argument(
        "--admin-endpoint",
        default=os.getenv("DEP_EXPORT_ADMIN_ENDPOINT", "http://localhost:8000/admin/dependencies"),
        help="Eleanor admin endpoint returning dependency metrics",
    )
    parser.add_argument(
        "--listen-addr",
        default=os.getenv("DEP_EXPORT_LISTEN_ADDR", "0.0.0.0"),
        help="Address the exporter binds to",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=int(os.getenv("DEP_EXPORT_LISTEN_PORT", "9105")),
        help="Prometheus scrape port",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.getenv("DEP_EXPORT_INTERVAL", "10")),
        help="Polling interval in seconds",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("DEP_EXPORT_TIMEOUT", "5")),
        help="Request timeout for the admin endpoint",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_exporter(args)


if __name__ == "__main__":
    main()
