#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime, timezone


def _load_boto3():
    try:
        import boto3  # type: ignore[import-not-found]
        from botocore.exceptions import ClientError  # type: ignore[import-not-found]
    except Exception:
        print("boto3 is required. Install with: pip install boto3", file=sys.stderr)
        return None, None
    return boto3, ClientError


def _env_default(name: str, fallback: str = "") -> str:
    return os.getenv(name, fallback)


def _init_meta(table, ledger_id: str, client_error):
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        table.put_item(
            Item={
                "ledger_id": ledger_id,
                "seq": "META",
                "seq_counter": 0,
                "last_hash": "",
                "updated_at": timestamp,
            },
            ConditionExpression="attribute_not_exists(ledger_id) AND attribute_not_exists(seq)",
        )
        print("Initialized META row.")
    except client_error as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            print("META row already exists.")
        else:
            raise


def main() -> int:
    boto3, client_error = _load_boto3()
    if boto3 is None:
        return 1

    parser = argparse.ArgumentParser(description="Create DynamoDB ledger index table.")
    parser.add_argument("--table", default=_env_default("ELEANOR_LEDGER_DDB_TABLE", ""))
    parser.add_argument("--region", default=_env_default("ELEANOR_LEDGER_DDB_REGION", _env_default("AWS_REGION", _env_default("AWS_DEFAULT_REGION", ""))))
    parser.add_argument("--ledger-id", default=_env_default("ELEANOR_LEDGER_DDB_LEDGER_ID", "default"))
    parser.add_argument("--billing-mode", choices=["PAY_PER_REQUEST", "PROVISIONED"], default="PAY_PER_REQUEST")
    parser.add_argument("--read-capacity", type=int, default=5)
    parser.add_argument("--write-capacity", type=int, default=5)
    parser.add_argument("--wait", action="store_true", help="Wait for table to become ACTIVE.")
    parser.add_argument("--init-meta", action="store_true", help="Create initial META row.")
    args = parser.parse_args()

    if not args.table:
        print("Table name is required. Set --table or ELEANOR_LEDGER_DDB_TABLE.", file=sys.stderr)
        return 1
    if not args.region:
        print("Region is required. Set --region or ELEANOR_LEDGER_DDB_REGION/AWS_REGION.", file=sys.stderr)
        return 1

    dynamodb = boto3.resource("dynamodb", region_name=args.region)
    client = boto3.client("dynamodb", region_name=args.region)

    try:
        client.describe_table(TableName=args.table)
        print(f"Table already exists: {args.table}")
        table = dynamodb.Table(args.table)
        if args.init_meta:
            _init_meta(table, args.ledger_id, client_error)
        return 0
    except client_error as exc:
        if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise

    params = {
        "TableName": args.table,
        "KeySchema": [
            {"AttributeName": "ledger_id", "KeyType": "HASH"},
            {"AttributeName": "seq", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "ledger_id", "AttributeType": "S"},
            {"AttributeName": "seq", "AttributeType": "S"},
        ],
        "BillingMode": args.billing_mode,
    }
    if args.billing_mode == "PROVISIONED":
        params["ProvisionedThroughput"] = {
            "ReadCapacityUnits": args.read_capacity,
            "WriteCapacityUnits": args.write_capacity,
        }

    client.create_table(**params)
    print(f"Created table: {args.table}")

    if args.wait:
        waiter = client.get_waiter("table_exists")
        waiter.wait(TableName=args.table)
        print("Table is ACTIVE.")

    table = dynamodb.Table(args.table)
    if args.init_meta:
        _init_meta(table, args.ledger_id, client_error)

    return 0


if __name__ == "__main__":
    sys.exit(main())
