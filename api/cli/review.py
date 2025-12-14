import json
import os

from replay_store import load_review_packet, load_human_reviews


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
