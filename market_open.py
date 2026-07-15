"""Exit successfully only while the regular US equity session is open."""
from __future__ import annotations

import argparse
import sys

import yfinance as yf


def market_is_open(status: dict | None) -> bool:
    return bool(status and str(status.get("status", "")).lower() == "open")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="bypass the session check")
    args = parser.parse_args()
    if args.force:
        print("market check bypassed")
        return 0
    try:
        status = yf.Market("us_market").status
    except Exception as exc:  # fail closed when Yahoo cannot confirm the session
        print(f"market status unavailable: {exc}", file=sys.stderr)
        return 1
    if not market_is_open(status):
        print(f"regular US market is closed ({status.get('status', 'unknown')})")
        return 1
    print(f"regular US market is open; closes {status.get('close', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
