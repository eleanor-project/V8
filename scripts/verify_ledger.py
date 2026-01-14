#!/usr/bin/env python3
import argparse
import json
import sys

from engine.security.ledger import verify_ledger


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify audit ledger hash chain integrity.")
    parser.add_argument(
        "--path",
        help="Override ledger path (defaults to ELEANOR_LEDGER_PATH for stone_tablet_ledger).",
    )
    args = parser.parse_args()

    result = verify_ledger(path=args.path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
