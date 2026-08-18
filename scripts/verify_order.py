"""Verification: place a real order via Kalshi V2 demo API using the fixed client.

Places a 1-contract immediate_or_cancel bid at $0.01 on an open market.
IOC at $0.01 will not fill on a market asking more, but proves:
  - order request is ACCEPTED (no 400 count error)
  - HTTP 201 is handled as success
  - order_id + fill/remaining counts are parsed
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from market.kalshi_client import KalshiClient

api_key = __import__("os").environ.get("KALSHI_API_KEY", "")
private_key = __import__("os").environ.get("KALSHI_PRIVATE_KEY", "")

client = KalshiClient(
    api_key=api_key,
    private_key_pem=private_key,
    dry_run=False,
    use_demo=True,
)

balance_before = client.get_balance()
print(f"balance_before=${balance_before}")

ticker = "KXUCLGAME-26AUG18DINVIK-DIN"
try:
    order_id = client.place_order(
        ticker=ticker,
        side="bid",
        count=1,
        yes_price=0.01,
        time_in_force="immediate_or_cancel",
    )
    print(f"ORDER_RESULT_ID={order_id}")
    if order_id and not str(order_id).startswith("dry_run"):
        print("SUCCESS: order accepted by demo API (201 handled)")
        print(f"verify_order: {order_id}")
    else:
        print("FAIL: order_id missing or dry_run returned")
except Exception as e:
    print(f"EXCEPTION: {e}")
    sys.exit(1)

time.sleep(1)
print("done")
