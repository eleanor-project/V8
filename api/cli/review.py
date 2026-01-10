import json
import os

from engine.replay_store import load_review_packet, load_human_reviews
from governance.human_review.metrics import review_metrics
from governance.human_review.analytics import (
    severity_drift,
    dissent_suppression_rate,
)
from precedent.quarantine import list_quarantined_cases


def list_reviews():
    packet_dir = "logs/review_packets"
    if not os.path.exists(packet_dir):
        return []
    return sorted([f for f in os.listdir(packet_dir) if f.endswith(".json")])


def replay_case(case_id: str):
    packet = load_review_packet(case_id)
    reviews = load_human_reviews(case_id)

    print("\n=== REVIEW PACKET ===")
    if packet:
        print(json.dumps(packet, indent=2))
    else:
        print("No review packet found.")

    print("\n=== HUMAN REVIEWS ===")
    if not reviews:
        print("No human reviews found.")
    else:
        for r in reviews:
            print(json.dumps(r, indent=2))
            print("-" * 20)


def show_metrics(case_id: str):
    metrics = review_metrics(case_id)
    drift = severity_drift(case_id)
    dissent = dissent_suppression_rate(case_id)

    print("\n=== REVIEW METRICS ===")
    if metrics:
        print(json.dumps(metrics, indent=2))
    else:
        print("No review metrics found.")

    print("\n=== SEVERITY DRIFT ===")
    print(json.dumps(drift, indent=2) if drift else "No severity drift data.")

    print("\n=== DISSENT SUPPRESSION ===")
    print(json.dumps(dissent, indent=2) if dissent else "No dissent data.")


def list_quarantine():
    cases = list_quarantined_cases()
    print("\n=== QUARANTINED CASES ===")
    if not cases:
        print("None")
        return
    for case in cases:
        print(json.dumps(case, indent=2))
        print("-" * 20)
