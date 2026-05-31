"""Fetch multi-asset, multi-year daily USD prices for out-of-sample validation.

Uses the Coinbase Exchange public candles API (no key) and paginates backward in
300-day chunks. Saves one CSV per asset under data/oos/ in the engine's
normalized format (block_number, timestamp, asset_symbol, price_usd).

These assets/years are distinct from the ETH series the policy was tuned on, so
evaluating the FIXED best candidate on them is genuinely out-of-sample.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "oos"
PRODUCTS = ["BTC-USD", "ETH-USD", "LTC-USD", "BCH-USD", "ETC-USD",
            "LINK-USD", "XLM-USD", "EOS-USD", "ZEC-USD", "XTZ-USD",
            "ADA-USD", "SOL-USD", "DOT-USD", "AVAX-USD", "MATIC-USD"]
START = datetime(2019, 1, 1, tzinfo=timezone.utc)
END = datetime(2024, 12, 31, tzinfo=timezone.utc)
GRAN = 86400  # daily
UA = {"User-Agent": "Mozilla/5.0 (research; daily candles)"}


def fetch_product(product: str) -> list[tuple[int, float]]:
    """Return [(unix_ts, close)] ascending, paginating 300 days at a time."""
    rows: dict[int, float] = {}
    cur = START
    while cur < END:
        chunk_end = min(cur + timedelta(days=300), END)
        url = (f"https://api.exchange.coinbase.com/products/{product}/candles"
               f"?granularity={GRAN}&start={cur.isoformat()}&end={chunk_end.isoformat()}")
        try:
            with urlopen(Request(url, headers=UA), timeout=30) as r:
                data = json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            print(f"  {product} {cur.date()} ERR {e}")
            data = []
        # candle = [time, low, high, open, close, volume]
        for c in data:
            rows[int(c[0])] = float(c[4])
        cur = chunk_end
        time.sleep(0.35)  # be polite to the public endpoint
    return sorted(rows.items())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for product in PRODUCTS:
        series = fetch_product(product)
        if len(series) < 400:  # need a few years of daily data
            print(f"{product}: only {len(series)} points, skipping")
            continue
        sym = product.split("-")[0]
        lines = ["block_number,timestamp,asset_symbol,price_usd"]
        for i, (ts, close) in enumerate(series):
            iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            lines.append(f"{i},{iso},{sym},{close}")
        (OUT / f"{sym}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        first = datetime.fromtimestamp(series[0][0], tz=timezone.utc).date()
        last = datetime.fromtimestamp(series[-1][0], tz=timezone.utc).date()
        manifest.append({"symbol": sym, "points": len(series), "first": str(first), "last": str(last)})
        print(f"{sym}: {len(series)} days, {first} -> {last}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n{len(manifest)} assets saved to {OUT}")


if __name__ == "__main__":
    main()
