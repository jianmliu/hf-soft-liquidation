from __future__ import annotations

import argparse
import hashlib
import json
import itertools
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

REQUIRED_PRICES_COLUMNS = ["block_number", "timestamp", "asset_symbol", "price_usd"]
REQUIRED_POSITIONS_COLUMNS = [
    "account",
    "asset_symbol",
    "collateral_amount",
    "debt_amount",
    "liquidation_threshold",
]
REQUIRED_LIQUIDATION_COLUMNS = [
    "block_number",
    "timestamp",
    "account",
    "asset_symbol",
    "debt_asset_symbol",
    "collateral_amount",
    "debt_repaid",
    "liquidator",
]


@dataclass
class TriggerTier:
    trigger_tier: str
    hf_down: float
    hf_up: float
    close_factor: float
    liquidation_bonus: float
    buyback_ratio: float


@dataclass
class Scenario:
    name: str
    tiers: list[TriggerTier]
    dynamic_policy: "DynamicPolicy | None" = None


@dataclass
class DynamicPolicy:
    lltv: float
    min_close_factor: float
    max_close_factor: float
    cf_slope: float
    liquidation_bonus: float
    buyback_ratio: float
    buyback_funding: str
    recovery_ltv_gap: float
    sell_cooldown_steps: int
    buy_cooldown_steps: int
    enable_buyback: bool
    target_hf: float | None = None
    # Buyback re-leverage guard: cap reborrow so post-buy HF stays >= this floor
    # (prevents buyback from pushing the position back toward liquidation). If
    # None, fall back to borrowing up to the LLTV limit (legacy behaviour).
    buyback_hf_floor: float | None = None
    # Require the matched sell price to exceed the current price by at least this
    # fraction (a stricter positive-spread guard than P_t < P_sell).
    min_buyback_spread: float = 0.0
    # WHEN to buy: require a confirmed bounce off a local bottom -- the price must
    # have recovered at least `buyback_min_bounce` above the lowest price in the
    # last `buyback_uptrend_lookback` steps -- to avoid catching a falling knife
    # (or buying a dead-cat bounce) during multi-leg crashes. lookback 0 disables.
    buyback_uptrend_lookback: int = 0
    buyback_min_bounce: float = 0.0
    # HOW MUCH to buy: size the reborrow so the post-buy health factor stays >= 1
    # even after a further `buyback_stress_drawdown` fractional price drop
    # (stress-tested quantity). 0 disables; equivalent to LT -> LT*(1-d) in B*.
    buyback_stress_drawdown: float = 0.0


@dataclass
class AavePreset:
    name: str
    endpoint: str
    market: str
    default_symbols: list[str]
    coingecko_ids: dict[str, str]


AAVE_PRESETS: dict[str, AavePreset] = {
    "ethereum-v3": AavePreset(
        name="ethereum-v3",
        endpoint="https://api.thegraph.com/subgraphs/name/aave/protocol-v3",
        market="aave-v3-ethereum",
        default_symbols=["WETH", "WBTC", "USDC", "USDT", "DAI"],
        coingecko_ids={
            "WETH": "ethereum",
            "WBTC": "wrapped-bitcoin",
            "USDC": "usd-coin",
            "USDT": "tether",
            "DAI": "dai",
        },
    ),
    "arbitrum-v3": AavePreset(
        name="arbitrum-v3",
        endpoint="https://api.thegraph.com/subgraphs/name/aave/protocol-v3-arbitrum",
        market="aave-v3-arbitrum",
        default_symbols=["WETH", "WBTC", "USDC", "USDT", "ARB"],
        coingecko_ids={
            "WETH": "ethereum",
            "WBTC": "wrapped-bitcoin",
            "USDC": "usd-coin",
            "USDT": "tether",
            "ARB": "arbitrum",
        },
    ),
    "polygon-v3": AavePreset(
        name="polygon-v3",
        endpoint="https://api.thegraph.com/subgraphs/name/aave/protocol-v3-polygon",
        market="aave-v3-polygon",
        default_symbols=["WMATIC", "WETH", "WBTC", "USDC", "USDT"],
        coingecko_ids={
            "WMATIC": "matic-network",
            "WETH": "ethereum",
            "WBTC": "wrapped-bitcoin",
            "USDC": "usd-coin",
            "USDT": "tether",
        },
    ),
}

DUNE_DEFAULT_COINGECKO_MAP: dict[str, str] = {
    "WETH": "ethereum",
    "WBTC": "wrapped-bitcoin",
    "USDC": "usd-coin",
    "USDT": "tether",
    "DAI": "dai",
    "AAVE": "aave",
    "GHO": "gho-token",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "CRV": "curve-dao-token",
    "LDO": "lido-dao",
    "MKR": "maker",
    "ENS": "ethereum-name-service",
    "FRAX": "frax",
    "SNX": "havven",
    "KNC": "kyber-network-crystal",
    "RPL": "rocket-pool",
}

DUNE_DEFAULT_TOKEN_DECIMALS: dict[str, int] = {
    "USDC": 6,
    "USDT": 6,
    "DAI": 18,
    "WETH": 18,
    "WBTC": 8,
    "AAVE": 18,
    "GHO": 18,
    "LINK": 18,
    "UNI": 18,
    "CRV": 18,
    "LDO": 18,
    "MKR": 18,
    "ENS": 18,
    "FRAX": 18,
    "SNX": 18,
    "KNC": 18,
    "RPL": 18,
    "RETH": 18,
    "CBETH": 18,
    "STETH": 18,
    "RLUSD": 18,
}


class DatasetError(ValueError):
    pass


def render_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def parse_symbols(raw: str) -> list[str]:
    return [part.strip().upper() for part in raw.split(",") if part.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def normalize_token_amount(raw_amount: Any, decimals: int) -> float:
    amount = to_float(raw_amount, default=0.0)
    if decimals < 0:
        decimals = 0
    text = str(raw_amount).strip()
    is_integer_like = bool(re.fullmatch(r"[-+]?\d+", text))
    if decimals > 0 and (is_integer_like or abs(amount) >= 1e12):
        return amount / (10**decimals)
    return amount


def infer_token_decimals(symbol: Any, fallback: int = 18) -> int:
    key = str(symbol or "").strip().upper()
    if not key:
        return fallback
    return int(DUNE_DEFAULT_TOKEN_DECIMALS.get(key, fallback))


def normalize_lt(raw_lt: Any) -> float:
    lt = to_float(raw_lt, default=0.0)
    if lt > 1:
        lt = lt / 10000.0
    return lt


def find_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {col.lower(): col for col in df.columns}
    for key in candidates:
        if key.lower() in lower_map:
            return lower_map[key.lower()]
    return None


def dune_api_get(url: str, api_key: str) -> dict[str, Any]:
    headers = {
        "x-dune-api-key": api_key,
        "Content-Type": "application/json",
    }
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")[:800]
        except Exception:
            body = ""
        raise DatasetError(f"Dune API HTTP error {exc.code} {exc.reason}: {body or '<empty>'}") from exc
    except URLError as exc:
        raise DatasetError(f"Dune API connection error: {exc}") from exc


def dune_api_post(url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {
        "x-dune-api-key": api_key,
        "Content-Type": "application/json",
    }
    request = Request(url, headers=headers, data=json.dumps(payload).encode("utf-8"), method="POST")
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")[:800]
        except Exception:
            body = ""
        raise DatasetError(f"Dune API HTTP error {exc.code} {exc.reason}: {body or '<empty>'}") from exc
    except URLError as exc:
        raise DatasetError(f"Dune API connection error: {exc}") from exc


def dune_fetch_rows(query_id: int, api_key: str, max_wait_seconds: int = 180, poll_seconds: int = 3) -> pd.DataFrame:
    latest_url = f"https://api.dune.com/api/v1/query/{query_id}/results"
    latest = dune_api_get(latest_url, api_key=api_key)
    rows = latest.get("result", {}).get("rows", [])
    if rows:
        return pd.DataFrame(rows)

    exec_url = f"https://api.dune.com/api/v1/query/{query_id}/execute"
    execution = dune_api_post(exec_url, api_key=api_key, payload={})
    execution_id = execution.get("execution_id")
    if not execution_id:
        raise DatasetError(f"Unable to obtain execution_id for Dune query {query_id}")

    status_url = f"https://api.dune.com/api/v1/execution/{execution_id}/status"
    result_url = f"https://api.dune.com/api/v1/execution/{execution_id}/results"
    start_time = time.time()
    while True:
        status_payload = dune_api_get(status_url, api_key=api_key)
        state = (status_payload.get("state") or "").upper()
        if state == "QUERY_STATE_COMPLETED":
            break
        if state in {"QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED", "QUERY_STATE_EXPIRED"}:
            raise DatasetError(f"Dune query {query_id} execution failed with state {state}")
        if time.time() - start_time > max_wait_seconds:
            raise DatasetError(f"Timed out waiting for Dune query {query_id} execution")
        time.sleep(max(1, poll_seconds))

    result_payload = dune_api_get(result_url, api_key=api_key)
    rows = result_payload.get("result", {}).get("rows", [])
    return pd.DataFrame(rows)


def normalize_dune_liquidations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_LIQUIDATION_COLUMNS)

    account_col = find_first_column(df, ["user", "user_address", "borrower", "account", "on_behalf_of"])
    liquidator_col = find_first_column(df, ["liquidator", "liquidator_address", "liquidator_addr"])
    collateral_symbol_col = find_first_column(
        df,
        ["collateral_symbol", "collateralsymbol", "collateral", "symbol", "asset_symbol"],
    )
    debt_symbol_col = find_first_column(
        df,
        [
            "debt_symbol",
            "debtsymbol",
            "debt_asset_symbol",
            "debt_asset",
            "principal_symbol",
            "borrow_symbol",
            "borrowsymbol",
        ],
    )
    collateral_amount_col = find_first_column(
        df,
        [
            "collateral_amount",
            "liquidated_collateral",
            "liquidated_collateral_amount",
            "liquidatedcollateralamount",
            "collateral_qty",
            "collateralamount",
        ],
    )
    debt_repaid_col = find_first_column(
        df,
        [
            "debt_repaid",
            "debt_covered",
            "debt_to_cover",
            "debttocover",
            "principal_amount",
            "debtamount",
            "debt_amount",
        ],
    )
    timestamp_col = find_first_column(df, ["timestamp", "block_time", "time", "evt_block_time"])
    block_col = find_first_column(df, ["block_number", "block", "evt_block_number"])
    collateral_decimals_col = find_first_column(
        df,
        ["collateral_decimals", "collateraldecimals", "asset_decimals", "decimals"],
    )
    debt_decimals_col = find_first_column(
        df,
        ["debt_decimals", "debtdecimals", "debt_asset_decimals", "debtassetdecimals"],
    )

    required = [account_col, collateral_symbol_col, collateral_amount_col, debt_repaid_col, timestamp_col]
    if any(col is None for col in required):
        raise DatasetError(
            "Dune liquidation query result missing required columns for normalization. "
            f"Columns seen: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["account"] = df[account_col].astype(str).str.lower()
    out["asset_symbol"] = df[collateral_symbol_col].astype(str).str.upper()
    out["debt_asset_symbol"] = df[debt_symbol_col].astype(str).str.upper() if debt_symbol_col else "UNKNOWN"

    def row_decimals(row: pd.Series, explicit_col: str | None, symbol_col: str | None) -> int:
        if explicit_col:
            explicit = to_float(row.get(explicit_col), default=-1)
            if explicit >= 0:
                return int(explicit)
        if symbol_col:
            return infer_token_decimals(row.get(symbol_col), fallback=18)
        return 18

    out["collateral_amount"] = df.apply(
        lambda row: normalize_token_amount(
            row.get(collateral_amount_col),
            row_decimals(row, collateral_decimals_col, collateral_symbol_col),
        ),
        axis=1,
    )
    out["debt_repaid"] = df.apply(
        lambda row: normalize_token_amount(
            row.get(debt_repaid_col),
            row_decimals(row, debt_decimals_col, debt_symbol_col),
        ),
        axis=1,
    )
    out["liquidator"] = df[liquidator_col].astype(str).str.lower() if liquidator_col else ""

    ts = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
    out["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    out["timestamp"] = out["timestamp"].str.replace(r"(\+|\-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)

    if block_col:
        out["block_number"] = df[block_col].map(lambda x: int(to_float(x, default=0)))
    else:
        out = out.sort_values("timestamp").reset_index(drop=True)
        out["block_number"] = np.arange(len(out), dtype=int)

    out = out[
        [
            "block_number",
            "timestamp",
            "account",
            "asset_symbol",
            "debt_asset_symbol",
            "collateral_amount",
            "debt_repaid",
            "liquidator",
        ]
    ]
    out = out.dropna(subset=["account", "asset_symbol", "timestamp"]).reset_index(drop=True)
    return out


def normalize_dune_positions(df: pd.DataFrame, liquidation_threshold: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=REQUIRED_POSITIONS_COLUMNS)

    account_col = find_first_column(df, ["user", "user_address", "account", "borrower", "wallet"])
    symbol_col = find_first_column(df, ["asset_symbol", "symbol", "asset", "token", "reserve_symbol"])
    collateral_col = find_first_column(df, ["collateral_amount", "supplied_amount", "deposit_amount", "a_token_balance", "collateral"])
    debt_col = find_first_column(df, ["debt_amount", "borrow_amount", "outstanding_debt", "total_debt", "debt"])

    required = [account_col, symbol_col, collateral_col, debt_col]
    if any(col is None for col in required):
        raise DatasetError(
            "Dune positions query result missing required columns for normalization. "
            f"Columns seen: {list(df.columns)}"
        )

    out = pd.DataFrame()
    out["account"] = df[account_col].astype(str).str.lower()
    out["asset_symbol"] = df[symbol_col].astype(str).str.upper()
    out["collateral_amount"] = df[collateral_col].map(to_float)
    out["debt_amount"] = df[debt_col].map(to_float)
    out["liquidation_threshold"] = liquidation_threshold

    out = out[(out["collateral_amount"] > 0) & (out["debt_amount"] > 0)]
    out = (
        out.groupby(["account", "asset_symbol"], as_index=False)
        .agg(
            collateral_amount=("collateral_amount", "sum"),
            debt_amount=("debt_amount", "sum"),
            liquidation_threshold=("liquidation_threshold", "max"),
        )
        .sort_values(["asset_symbol", "account"])
        .reset_index(drop=True)
    )
    return out


def collect_dune_channel(
    dataset_dir: Path,
    dune_api_key: str,
    liquidation_query_id: int,
    positions_query_id: int | None,
    liquidation_threshold: float,
    collect_prices: bool,
    symbol_map: dict[str, str],
) -> dict[str, Any]:
    raw_dir = dataset_dir / "raw"
    normalized_dir = dataset_dir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    liq_raw = dune_fetch_rows(liquidation_query_id, api_key=dune_api_key)
    liq_raw.to_csv(raw_dir / "dune_liquidations_raw.csv", index=False)
    liq_norm = normalize_dune_liquidations(liq_raw)
    liq_norm.to_csv(normalized_dir / "liquidation_events.csv", index=False)

    pos_count = 0
    positions_written = False
    if positions_query_id is not None:
        pos_raw = dune_fetch_rows(positions_query_id, api_key=dune_api_key)
        pos_raw.to_csv(raw_dir / "dune_positions_raw.csv", index=False)
        pos_norm = normalize_dune_positions(pos_raw, liquidation_threshold=liquidation_threshold)
        pos_norm.to_csv(normalized_dir / "positions_initial.csv", index=False)
        pos_count = len(pos_norm)
        positions_written = True

    price_rows_by_symbol: dict[str, int] = {}
    if collect_prices:
        symbols = sorted(set(liq_norm["asset_symbol"].dropna().astype(str).tolist()))
        symbol_to_id = {symbol: symbol_map[symbol] for symbol in symbols if symbol in symbol_map}
        if symbol_to_id:
            start, end = default_time_window(30)
            price_rows_by_symbol = collect_prices_for_symbols(
                symbol_to_asset_id=symbol_to_id,
                start=start,
                end=end,
                out_csv=normalized_dir / "prices.csv",
            )

    meta = {
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        "liquidation_query_id": liquidation_query_id,
        "positions_query_id": positions_query_id,
        "liquidation_events_count": int(len(liq_norm)),
        "positions_count": int(pos_count),
        "positions_written": positions_written,
        "price_rows_by_symbol": price_rows_by_symbol,
    }
    (raw_dir / "dune_collection_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def graphql_query(endpoint: str, query: str, variables: dict[str, Any], api_key: str | None) -> dict[str, Any]:
    effective_endpoint = quote(endpoint, safe=':/?&=%')
    if api_key and "gateway.thegraph.com/api/subgraphs/id/" in endpoint and "/api/" in endpoint and "/subgraphs/id/" in endpoint:
        effective_endpoint = quote(
            endpoint.replace("/api/subgraphs/id/", f"/api/{api_key}/subgraphs/id/"),
            safe=':/?&=%',
        )

    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(effective_endpoint, data=payload, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")[:800]
        except Exception:
            body = ""
        raise DatasetError(
            f"GraphQL HTTP error from {effective_endpoint}: {exc.code} {exc.reason}. "
            f"Response: {body or '<empty>'}. "
            "If this endpoint requires an API key, pass --api-key; "
            "if it is not a subgraph endpoint, use a compatible Aave subgraph URL."
        ) from exc
    except URLError as exc:
        raise DatasetError(f"GraphQL connection error for {effective_endpoint}: {exc}") from exc
    if "errors" in data:
        raise DatasetError(f"GraphQL returned errors: {data['errors']}")
    if "data" not in data:
        raise DatasetError("GraphQL response missing 'data'")
    return data["data"]


def detect_query_fields(endpoint: str, api_key: str | None) -> set[str]:
    data = graphql_query(
        endpoint=endpoint,
        query="query { __schema { queryType { fields { name } } } }",
        variables={},
        api_key=api_key,
    )
    fields = data.get("__schema", {}).get("queryType", {}).get("fields", [])
    return {str(item.get("name")) for item in fields if item.get("name")}


def fetch_user_reserves(
    endpoint: str,
    symbols: list[str],
    page_size: int,
    max_pages: int,
    api_key: str | None,
) -> list[dict[str, Any]]:
    query = """
    query GetUserReserves($first: Int!, $lastId: String!, $symbols: [String!]) {
      userReserves(
        first: $first,
        orderBy: id,
        orderDirection: asc,
        where: {
          id_gt: $lastId,
          currentATokenBalance_gt: "0",
          reserve_: { symbol_in: $symbols }
        }
      ) {
        id
        currentATokenBalance
        currentTotalDebt
        currentVariableDebt
        currentStableDebt
        user { id }
        reserve {
          id
          symbol
          decimals
          liquidationThreshold
        }
      }
    }
    """
    records: list[dict[str, Any]] = []
    last_id = ""
    for _ in range(max_pages):
        data = graphql_query(
            endpoint,
            query,
            {"first": page_size, "lastId": last_id, "symbols": symbols},
            api_key=api_key,
        )
        rows = data.get("userReserves", [])
        if not rows:
            break
        records.extend(rows)
        last_id = str(rows[-1].get("id", ""))
        if len(rows) < page_size:
            break
    return records


def fetch_liquidation_calls(
    endpoint: str,
    symbols: list[str],
    start_ts: int | None,
    end_ts: int | None,
    page_size: int,
    max_pages: int,
    api_key: str | None,
) -> list[dict[str, Any]]:
    query = """
    query GetLiquidationCalls($first: Int!, $lastId: String!, $startTs: BigInt, $endTs: BigInt) {
      liquidationCalls(
        first: $first,
        orderBy: id,
        orderDirection: asc,
        where: { id_gt: $lastId, timestamp_gte: $startTs, timestamp_lte: $endTs }
      ) {
        id
        blockNumber
        timestamp
        txHash
        collateralAmount
        principalAmount
        user { id }
        liquidator { id }
        collateralReserve { symbol decimals }
        principalReserve { symbol decimals }
      }
    }
    """
    records: list[dict[str, Any]] = []
    last_id = ""
    symbol_set = set(symbols)
    for _ in range(max_pages):
        data = graphql_query(
            endpoint,
            query,
            {
                "first": page_size,
                "lastId": last_id,
                "startTs": start_ts,
                "endTs": end_ts,
            },
            api_key=api_key,
        )
        rows = data.get("liquidationCalls", [])
        if not rows:
            break

        for row in rows:
            collateral_symbol = str(row.get("collateralReserve", {}).get("symbol", "")).upper()
            debt_symbol = str(row.get("principalReserve", {}).get("symbol", "")).upper()
            if symbol_set and collateral_symbol not in symbol_set and debt_symbol not in symbol_set:
                continue
            records.append(row)

        last_id = str(rows[-1].get("id", ""))
        if len(rows) < page_size:
            break
    return records


def collect_aave_subgraph(
    endpoint: str,
    market: str,
    symbols: list[str],
    dataset_dir: Path,
    start: str | None,
    end: str | None,
    page_size: int,
    max_pages: int,
    api_key: str | None,
) -> dict[str, int]:
    if not symbols:
        raise DatasetError("symbols cannot be empty")

    raw_dir = dataset_dir / "raw"
    normalized_dir = dataset_dir / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    start_ts = int(parse_float_date(start)) if start else None
    end_ts = int(parse_float_date(end)) if end else None
    if start_ts and end_ts and end_ts <= start_ts:
        raise DatasetError("end must be greater than start")

    query_fields = detect_query_fields(endpoint=endpoint, api_key=api_key)
    has_subgraph_shape = "userReserves" in query_fields and "liquidationCalls" in query_fields
    if not has_subgraph_shape:
        hint = ""
        if "userSupplies" in query_fields and "userBorrows" in query_fields:
            hint = (
                "Detected Aave API user-centric schema (e.g. api.v3.aave.com/graphql), "
                "which cannot directly scan market-wide positions/events without a user-address universe."
            )
        raise DatasetError(
            "Endpoint is not compatible with market-wide subgraph collection. "
            "Required query fields: userReserves and liquidationCalls. "
            f"{hint}"
        )

    user_reserves = fetch_user_reserves(
        endpoint=endpoint,
        symbols=symbols,
        page_size=page_size,
        max_pages=max_pages,
        api_key=api_key,
    )
    liquidation_calls = fetch_liquidation_calls(
        endpoint=endpoint,
        symbols=symbols,
        start_ts=start_ts,
        end_ts=end_ts,
        page_size=page_size,
        max_pages=max_pages,
        api_key=api_key,
    )

    positions_raw_rows: list[dict[str, Any]] = []
    positions_normalized_rows: list[dict[str, Any]] = []
    fetched_at = datetime.now(tz=timezone.utc).isoformat()
    for item in user_reserves:
        reserve = item.get("reserve") or {}
        user = item.get("user") or {}
        symbol = str(reserve.get("symbol", "")).upper()
        decimals = int(to_float(reserve.get("decimals", 18), default=18))
        collateral_amount = normalize_token_amount(item.get("currentATokenBalance", 0), decimals)
        debt_total = to_float(item.get("currentTotalDebt"), default=0.0)
        if debt_total <= 0:
            debt_total = to_float(item.get("currentVariableDebt"), default=0.0) + to_float(
                item.get("currentStableDebt"), default=0.0
            )
        debt_amount = normalize_token_amount(debt_total, decimals)
        lt = normalize_lt(reserve.get("liquidationThreshold"))
        account = str(user.get("id", "")).lower()
        if not account or not symbol:
            continue

        positions_raw_rows.append(
            {
                "market": market,
                "fetched_at": fetched_at,
                "id": item.get("id"),
                "account": account,
                "asset_symbol": symbol,
                "collateral_amount": collateral_amount,
                "debt_amount": debt_amount,
                "liquidation_threshold": lt,
                "reserve_id": reserve.get("id"),
                "raw_currentATokenBalance": item.get("currentATokenBalance"),
                "raw_currentTotalDebt": item.get("currentTotalDebt"),
                "raw_currentVariableDebt": item.get("currentVariableDebt"),
                "raw_currentStableDebt": item.get("currentStableDebt"),
                "raw_decimals": decimals,
            }
        )

        if collateral_amount > 0 and debt_amount > 0:
            positions_normalized_rows.append(
                {
                    "account": account,
                    "asset_symbol": symbol,
                    "collateral_amount": collateral_amount,
                    "debt_amount": debt_amount,
                    "liquidation_threshold": lt,
                }
            )

    positions_raw_df = pd.DataFrame(positions_raw_rows)
    if not positions_raw_df.empty:
        positions_raw_df = positions_raw_df.sort_values(["asset_symbol", "account"]).reset_index(drop=True)
    positions_raw_df.to_csv(raw_dir / "positions_raw.csv", index=False)

    positions_df = pd.DataFrame(positions_normalized_rows)
    if not positions_df.empty:
        positions_df = (
            positions_df.groupby(["account", "asset_symbol"], as_index=False)
            .agg(
                collateral_amount=("collateral_amount", "sum"),
                debt_amount=("debt_amount", "sum"),
                liquidation_threshold=("liquidation_threshold", "max"),
            )
            .sort_values(["asset_symbol", "account"])
            .reset_index(drop=True)
        )
    positions_df.to_csv(normalized_dir / "positions_initial.csv", index=False)

    liquidation_raw_rows: list[dict[str, Any]] = []
    liquidation_normalized_rows: list[dict[str, Any]] = []
    for item in liquidation_calls:
        collateral_reserve = item.get("collateralReserve") or {}
        principal_reserve = item.get("principalReserve") or {}
        collateral_symbol = str(collateral_reserve.get("symbol", "")).upper()
        debt_symbol = str(principal_reserve.get("symbol", "")).upper()
        collateral_decimals = int(to_float(collateral_reserve.get("decimals", 18), default=18))
        debt_decimals = int(to_float(principal_reserve.get("decimals", 18), default=18))
        collateral_amount = normalize_token_amount(item.get("collateralAmount", 0), collateral_decimals)
        debt_repaid = normalize_token_amount(item.get("principalAmount", 0), debt_decimals)
        account = str((item.get("user") or {}).get("id", "")).lower()
        liquidator = str((item.get("liquidator") or {}).get("id", "")).lower()
        timestamp_value = int(to_float(item.get("timestamp"), default=0))
        block_number = int(to_float(item.get("blockNumber"), default=0))

        liquidation_raw_rows.append(
            {
                "market": market,
                "fetched_at": fetched_at,
                "id": item.get("id"),
                "tx_hash": item.get("txHash"),
                "block_number": block_number,
                "timestamp": timestamp_value,
                "account": account,
                "liquidator": liquidator,
                "asset_symbol": collateral_symbol,
                "debt_asset_symbol": debt_symbol,
                "collateral_amount": collateral_amount,
                "debt_repaid": debt_repaid,
                "raw_collateralAmount": item.get("collateralAmount"),
                "raw_principalAmount": item.get("principalAmount"),
            }
        )

        if account and collateral_symbol:
            liquidation_normalized_rows.append(
                {
                    "block_number": block_number,
                    "timestamp": datetime.fromtimestamp(timestamp_value, tz=timezone.utc).isoformat()
                    if timestamp_value > 0
                    else "",
                    "account": account,
                    "asset_symbol": collateral_symbol,
                    "debt_asset_symbol": debt_symbol,
                    "collateral_amount": collateral_amount,
                    "debt_repaid": debt_repaid,
                    "liquidator": liquidator,
                }
            )

    liquidation_raw_df = pd.DataFrame(liquidation_raw_rows)
    if not liquidation_raw_df.empty:
        liquidation_raw_df = liquidation_raw_df.sort_values(["block_number", "asset_symbol"]).reset_index(drop=True)
    liquidation_raw_df.to_csv(raw_dir / "liquidations_raw.csv", index=False)

    liquidation_df = pd.DataFrame(liquidation_normalized_rows)
    if not liquidation_df.empty:
        liquidation_df = liquidation_df.sort_values(["block_number", "asset_symbol"]).reset_index(drop=True)
    liquidation_df.to_csv(normalized_dir / "liquidation_events.csv", index=False)

    meta_path = raw_dir / "collection_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "market": market,
                "endpoint": endpoint,
                "symbols": symbols,
                "start": start,
                "end": end,
                "fetched_at": fetched_at,
                "positions_count": int(len(positions_df)),
                "liquidation_events_count": int(len(liquidation_df)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "positions_count": int(len(positions_df)),
        "liquidation_events_count": int(len(liquidation_df)),
    }


def parse_float_date(date_str: str) -> float:
    dt = datetime.fromisoformat(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def default_time_window(days: int) -> tuple[str, str]:
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def resolve_preset(preset_name: str) -> AavePreset:
    key = preset_name.strip().lower()
    if key not in AAVE_PRESETS:
        options = ", ".join(sorted(AAVE_PRESETS.keys()))
        raise DatasetError(f"Unknown preset '{preset_name}'. Available: {options}")
    return AAVE_PRESETS[key]


def build_tiers(raw_tiers: list[dict[str, Any]], buyback_bandwidth: float) -> list[TriggerTier]:
    tiers: list[TriggerTier] = []
    for idx, tier in enumerate(raw_tiers):
        hf_down = float(tier["hf_down"])
        close_factor = float(tier["close_factor"])
        liquidation_bonus = float(tier["liquidation_bonus"])
        buyback_ratio = float(tier["buyback_ratio"])
        name = str(tier.get("name", f"Tier {idx + 1}"))
        if hf_down <= 0:
            raise DatasetError(f"Invalid hf_down in tier {name}")
        if not (0 < close_factor <= 1):
            raise DatasetError(f"Invalid close_factor in tier {name}")
        if liquidation_bonus < 0:
            raise DatasetError(f"Invalid liquidation_bonus in tier {name}")
        if not (0 < buyback_ratio <= 1):
            raise DatasetError(f"Invalid buyback_ratio in tier {name}")
        tiers.append(
            TriggerTier(
                trigger_tier=name,
                hf_down=hf_down,
                hf_up=hf_down + buyback_bandwidth,
                close_factor=close_factor,
                liquidation_bonus=liquidation_bonus,
                buyback_ratio=buyback_ratio,
            )
        )
    return sorted(tiers, key=lambda item: item.hf_down, reverse=True)


def build_dynamic_policy(raw: dict[str, Any]) -> DynamicPolicy:
    lltv = float(raw["lltv"])
    min_close_factor = float(raw["min_close_factor"])
    max_close_factor = float(raw["max_close_factor"])
    cf_slope = float(raw["cf_slope"])
    liquidation_bonus = float(raw["liquidation_bonus"])
    buyback_ratio = float(raw["buyback_ratio"])
    buyback_funding = str(raw.get("buyback_funding", "reborrow")).strip().lower()
    recovery_ltv_gap = float(raw.get("recovery_ltv_gap", 0.03))
    sell_cooldown_steps = int(raw.get("sell_cooldown_steps", 1))
    buy_cooldown_steps = int(raw.get("buy_cooldown_steps", 1))
    enable_buyback = bool(raw.get("enable_buyback", True))
    target_hf_raw = raw.get("target_hf", None)
    target_hf = float(target_hf_raw) if target_hf_raw is not None else None
    buyback_hf_floor_raw = raw.get("buyback_hf_floor", None)
    buyback_hf_floor = float(buyback_hf_floor_raw) if buyback_hf_floor_raw is not None else None
    min_buyback_spread = float(raw.get("min_buyback_spread", 0.0))
    buyback_uptrend_lookback = int(raw.get("buyback_uptrend_lookback", 0))
    buyback_min_bounce = float(raw.get("buyback_min_bounce", 0.0))
    buyback_stress_drawdown = float(raw.get("buyback_stress_drawdown", 0.0))

    if not (0 < lltv < 1):
        raise DatasetError("dynamic.lltv must be in (0, 1)")
    if not (0 < min_close_factor <= 1):
        raise DatasetError("dynamic.min_close_factor must be in (0, 1]")
    if not (0 < max_close_factor <= 1):
        raise DatasetError("dynamic.max_close_factor must be in (0, 1]")
    if min_close_factor > max_close_factor:
        raise DatasetError("dynamic.min_close_factor must be <= dynamic.max_close_factor")
    if cf_slope < 0:
        raise DatasetError("dynamic.cf_slope must be >= 0")
    if liquidation_bonus < 0:
        raise DatasetError("dynamic.liquidation_bonus must be >= 0")
    if not (0 < buyback_ratio <= 1):
        raise DatasetError("dynamic.buyback_ratio must be in (0, 1]")
    if buyback_funding != "reborrow":
        raise DatasetError("dynamic.buyback_funding must be reborrow")
    if recovery_ltv_gap < 0:
        raise DatasetError("dynamic.recovery_ltv_gap must be >= 0")
    if sell_cooldown_steps < 1 or buy_cooldown_steps < 1:
        raise DatasetError("dynamic cooldown steps must be >= 1")
    if target_hf is not None and target_hf <= 0:
        raise DatasetError("dynamic.target_hf must be > 0 when provided")
    if buyback_hf_floor is not None and buyback_hf_floor <= 0:
        raise DatasetError("dynamic.buyback_hf_floor must be > 0 when provided")
    if min_buyback_spread < 0:
        raise DatasetError("dynamic.min_buyback_spread must be >= 0")
    if buyback_uptrend_lookback < 0:
        raise DatasetError("dynamic.buyback_uptrend_lookback must be >= 0")
    if not (0.0 <= buyback_stress_drawdown < 1.0):
        raise DatasetError("dynamic.buyback_stress_drawdown must be in [0, 1)")

    return DynamicPolicy(
        lltv=lltv,
        min_close_factor=min_close_factor,
        max_close_factor=max_close_factor,
        cf_slope=cf_slope,
        liquidation_bonus=liquidation_bonus,
        buyback_ratio=buyback_ratio,
        buyback_funding=buyback_funding,
        recovery_ltv_gap=recovery_ltv_gap,
        sell_cooldown_steps=sell_cooldown_steps,
        buy_cooldown_steps=buy_cooldown_steps,
        enable_buyback=enable_buyback,
        target_hf=target_hf,
        buyback_hf_floor=buyback_hf_floor,
        min_buyback_spread=min_buyback_spread,
        buyback_uptrend_lookback=buyback_uptrend_lookback,
        buyback_min_bounce=buyback_min_bounce,
        buyback_stress_drawdown=buyback_stress_drawdown,
    )


def dynamic_close_factor(ltv: float, policy: DynamicPolicy) -> float:
    exceed_ratio = max(0.0, (ltv / policy.lltv) - 1.0)
    close_factor = policy.min_close_factor + policy.cf_slope * exceed_ratio
    return float(np.clip(close_factor, policy.min_close_factor, policy.max_close_factor))


def target_hf_debt_repaid(
    collateral: float,
    debt: float,
    price: float,
    liquidation_threshold: float,
    policy: DynamicPolicy,
) -> float:
    """Target-HF partial-liquidation sizing (paper eq. 56).

    Solve the debt reduction Delta D that moves the post-trade health factor to
    HF*, accounting for the liquidation bonus on collateral sold:

        Delta D* = (HF* * D - LT * C * P) / (HF* - LT * (1 + b)).

    Equivalently Delta D* = D * (HF* - HF_t) / (HF* - LT * (1 + b)), so the size
    scales with the health-factor gap. Clipped to [0, D]; if the denominator is
    non-positive (HF* <= LT*(1+b)) the target is infeasible and we fall back to
    full repayment, matching the close-factor==full behaviour.
    """
    target_hf = policy.target_hf
    denominator = target_hf - liquidation_threshold * (1.0 + policy.liquidation_bonus)
    if denominator <= 0:
        return debt
    delta_d = (target_hf * debt - liquidation_threshold * collateral * price) / denominator
    return float(min(max(0.0, delta_d), debt))


def load_scenarios(path: Path) -> list[Scenario]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    scenarios: list[Scenario] = []
    for obj in payload:
        name = str(obj["name"])
        if "dynamic" in obj:
            dynamic = build_dynamic_policy(obj["dynamic"])
            scenarios.append(Scenario(name=name, tiers=[], dynamic_policy=dynamic))
            continue

        if "tiers" in obj:
            buyback_bandwidth = float(obj.get("buyback_bandwidth", 0.05))
            tiers = build_tiers(obj["tiers"], buyback_bandwidth)
            scenarios.append(Scenario(name=name, tiers=tiers, dynamic_policy=None))
            continue

        raise DatasetError(f"Scenario '{name}' must define either 'dynamic' or 'tiers'")
    if not scenarios:
        raise DatasetError("No scenarios found")
    return scenarios


def ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DatasetError(f"{label} missing columns: {missing}")


def init_dataset(dataset_dir: Path, with_sample_data: bool, seed: int) -> None:
    raw_dir = dataset_dir / "raw"
    normalized_dir = dataset_dir / "normalized"
    config_dir = dataset_dir / "config"
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    prices_template = pd.DataFrame(columns=REQUIRED_PRICES_COLUMNS)
    positions_template = pd.DataFrame(columns=REQUIRED_POSITIONS_COLUMNS)

    (raw_dir / "prices_raw.csv").write_text(prices_template.to_csv(index=False), encoding="utf-8")
    (raw_dir / "positions_raw.csv").write_text(positions_template.to_csv(index=False), encoding="utf-8")

    sample_scenarios = [
        {
            "name": "baseline_dynamic",
            "dynamic": {
                "lltv": 0.82,
                "min_close_factor": 0.12,
                "max_close_factor": 0.55,
                "cf_slope": 1.8,
                "liquidation_bonus": 0.06,
                "buyback_ratio": 0.70,
                "buyback_funding": "reborrow",
                "recovery_ltv_gap": 0.04,
                "sell_cooldown_steps": 1,
                "buy_cooldown_steps": 1,
            },
        }
    ]
    (config_dir / "scenarios.json").write_text(json.dumps(sample_scenarios, indent=2), encoding="utf-8")

    if with_sample_data:
        rng = np.random.default_rng(seed)
        steps = 360
        times = pd.date_range("2024-01-01", periods=steps, freq="h", tz="UTC")
        base = 2500 + 260 * np.sin(np.arange(steps) / 22) + 120 * np.sin(np.arange(steps) / 7)
        shock = -280 * np.maximum(0, 1 - np.abs(np.arange(steps) - 190) / 65)
        noise = rng.normal(0, 28, size=steps)
        prices = np.clip(base + shock + noise, 300, None)
        price_df = pd.DataFrame(
            {
                "block_number": np.arange(steps),
                "timestamp": times.astype(str),
                "asset_symbol": "WETH",
                "price_usd": prices,
            }
        )

        positions_df = pd.DataFrame(
            [
                {
                    "account": "0xsample001",
                    "asset_symbol": "WETH",
                    "collateral_amount": 80.0,
                    "debt_amount": 120000.0,
                    "liquidation_threshold": 0.83,
                },
                {
                    "account": "0xsample002",
                    "asset_symbol": "WETH",
                    "collateral_amount": 52.0,
                    "debt_amount": 76000.0,
                    "liquidation_threshold": 0.83,
                },
                {
                    "account": "0xsample003",
                    "asset_symbol": "WETH",
                    "collateral_amount": 35.0,
                    "debt_amount": 54000.0,
                    "liquidation_threshold": 0.83,
                },
            ]
        )

        (normalized_dir / "prices.csv").write_text(price_df.to_csv(index=False), encoding="utf-8")
        (normalized_dir / "positions_initial.csv").write_text(positions_df.to_csv(index=False), encoding="utf-8")


def collect_prices_coingecko(asset_id: str, symbol: str, start: str, end: str, out_csv: Path) -> None:
    start_ts = int(parse_float_date(start))
    end_ts = int(parse_float_date(end))
    if end_ts <= start_ts:
        raise DatasetError("end must be greater than start")

    query = urlencode({"vs_currency": "usd", "from": start_ts, "to": end_ts})
    url = f"https://api.coingecko.com/api/v3/coins/{asset_id}/market_chart/range?{query}"
    with urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    prices = payload.get("prices", [])
    if not prices:
        raise DatasetError("No price data returned from CoinGecko")

    rows = []
    for idx, point in enumerate(prices):
        ts_ms, price = point
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
        rows.append(
            {
                "block_number": idx,
                "timestamp": ts,
                "asset_symbol": symbol,
                "price_usd": float(price),
            }
        )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(df.to_csv(index=False), encoding="utf-8")


def fetch_prices_coingecko_df(asset_id: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    start_ts = int(parse_float_date(start))
    end_ts = int(parse_float_date(end))
    if end_ts <= start_ts:
        raise DatasetError("end must be greater than start")

    query = urlencode({"vs_currency": "usd", "from": start_ts, "to": end_ts})
    url = f"https://api.coingecko.com/api/v3/coins/{asset_id}/market_chart/range?{query}"
    with urlopen(url, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    prices = payload.get("prices", [])
    if not prices:
        raise DatasetError(f"No price data returned from CoinGecko for {symbol} ({asset_id})")

    rows = []
    for idx, point in enumerate(prices):
        ts_ms, price = point
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
        rows.append(
            {
                "block_number": idx,
                "timestamp": ts,
                "asset_symbol": symbol,
                "price_usd": float(price),
            }
        )
    return pd.DataFrame(rows)


def collect_prices_for_symbols(
    symbol_to_asset_id: dict[str, str],
    start: str,
    end: str,
    out_csv: Path,
) -> dict[str, int]:
    frames: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    for symbol, asset_id in symbol_to_asset_id.items():
        frame = fetch_prices_coingecko_df(asset_id=asset_id, symbol=symbol, start=start, end=end)
        frames.append(frame)
        counts[symbol] = int(len(frame))

    if not frames:
        raise DatasetError("No symbols to collect prices for")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["asset_symbol", "timestamp"]).reset_index(drop=True)
    merged["block_number"] = merged.groupby("asset_symbol").cumcount()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(merged.to_csv(index=False), encoding="utf-8")
    return counts


def bootstrap_aave_market(
    preset_name: str,
    dataset_dir: Path,
    symbols: list[str] | None,
    start: str | None,
    end: str | None,
    endpoint_override: str | None,
    page_size: int,
    max_pages: int,
    api_key: str | None,
) -> dict[str, Any]:
    preset = resolve_preset(preset_name)
    selected_symbols = symbols or preset.default_symbols
    unknown_symbols = [symbol for symbol in selected_symbols if symbol not in preset.coingecko_ids]
    if unknown_symbols:
        raise DatasetError(f"Missing CoinGecko mapping for symbols: {unknown_symbols}")

    if not start or not end:
        default_start, default_end = default_time_window(30)
        start = start or default_start
        end = end or default_end

    endpoint = endpoint_override or preset.endpoint
    subgraph_result = collect_aave_subgraph(
        endpoint=endpoint,
        market=preset.market,
        symbols=selected_symbols,
        dataset_dir=dataset_dir,
        start=start,
        end=end,
        page_size=page_size,
        max_pages=max_pages,
        api_key=api_key,
    )

    symbol_to_asset_id = {symbol: preset.coingecko_ids[symbol] for symbol in selected_symbols}
    price_counts = collect_prices_for_symbols(
        symbol_to_asset_id=symbol_to_asset_id,
        start=start,
        end=end,
        out_csv=dataset_dir / "normalized" / "prices.csv",
    )

    validate_dataset(dataset_dir)
    return {
        "preset": preset.name,
        "market": preset.market,
        "endpoint": endpoint,
        "symbols": selected_symbols,
        "start": start,
        "end": end,
        "positions_count": subgraph_result["positions_count"],
        "liquidation_events_count": subgraph_result["liquidation_events_count"],
        "price_rows_by_symbol": price_counts,
    }


def validate_dataset(dataset_dir: Path) -> None:
    prices_path = dataset_dir / "normalized" / "prices.csv"
    positions_path = dataset_dir / "normalized" / "positions_initial.csv"
    liquidation_path = dataset_dir / "normalized" / "liquidation_events.csv"

    if not prices_path.exists():
        raise DatasetError(f"Missing {prices_path}")
    if not positions_path.exists():
        raise DatasetError(f"Missing {positions_path}")

    prices_df = pd.read_csv(prices_path)
    positions_df = pd.read_csv(positions_path)

    ensure_columns(prices_df, REQUIRED_PRICES_COLUMNS, "prices.csv")
    ensure_columns(positions_df, REQUIRED_POSITIONS_COLUMNS, "positions_initial.csv")

    prices_df["block_number"] = prices_df["block_number"].astype(int)
    prices_df["price_usd"] = prices_df["price_usd"].astype(float)
    if (prices_df["price_usd"] <= 0).any():
        raise DatasetError("prices.csv has non-positive prices")

    positions_df["collateral_amount"] = positions_df["collateral_amount"].astype(float)
    positions_df["debt_amount"] = positions_df["debt_amount"].astype(float)
    positions_df["liquidation_threshold"] = positions_df["liquidation_threshold"].astype(float)
    if (positions_df["collateral_amount"] <= 0).any():
        raise DatasetError("positions_initial.csv has non-positive collateral amounts")
    if (positions_df["debt_amount"] <= 0).any():
        raise DatasetError("positions_initial.csv has non-positive debt amounts")
    if ((positions_df["liquidation_threshold"] <= 0) | (positions_df["liquidation_threshold"] > 1)).any():
        raise DatasetError("positions_initial.csv has invalid liquidation_threshold values")

    if liquidation_path.exists():
        liquidation_df = pd.read_csv(liquidation_path)
        ensure_columns(liquidation_df, REQUIRED_LIQUIDATION_COLUMNS, "liquidation_events.csv")
        if not liquidation_df.empty:
            liquidation_df["block_number"] = liquidation_df["block_number"].astype(int)
            liquidation_df["collateral_amount"] = liquidation_df["collateral_amount"].astype(float)
            liquidation_df["debt_repaid"] = liquidation_df["debt_repaid"].astype(float)


def risk_metrics(collateral: float, debt: float, price: float, liquidation_threshold: float) -> tuple[float, float, float]:
    collateral_value = collateral * price
    if debt <= 0:
        return float("inf"), 0.0, float("inf")
    if collateral_value <= 0:
        return 0.0, float("inf"), 0.0
    hf = collateral_value * liquidation_threshold / debt
    ltv = debt / collateral_value
    cr = collateral_value / debt
    return hf, ltv, cr


def run_counterfactual(dataset_dir: Path, scenario_path: Path, output_dir: Path, run_id: str | None) -> Path:
    validate_dataset(dataset_dir)
    scenarios = load_scenarios(scenario_path)

    prices_df = pd.read_csv(dataset_dir / "normalized" / "prices.csv")
    positions_df = pd.read_csv(dataset_dir / "normalized" / "positions_initial.csv")
    prices_df = prices_df.sort_values("block_number").reset_index(drop=True)

    if run_id is None:
        run_id = datetime.now(tz=timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    event_rows: list[dict[str, Any]] = []
    account_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        for _, pos in positions_df.iterrows():
            account = str(pos["account"])
            symbol = str(pos["asset_symbol"])
            collateral = float(pos["collateral_amount"])
            debt = float(pos["debt_amount"])
            lt = float(pos["liquidation_threshold"])
            reserve_usd = 0.0

            start_block_raw = pos.get("start_block") if "start_block" in positions_df.columns else None
            end_block_raw = pos.get("end_block") if "end_block" in positions_df.columns else None
            start_block = int(start_block_raw) if pd.notna(start_block_raw) else None
            end_block = int(end_block_raw) if pd.notna(end_block_raw) else None

            asset_prices = prices_df[prices_df["asset_symbol"] == symbol]
            if start_block is not None:
                asset_prices = asset_prices[asset_prices["block_number"] >= start_block]
            if end_block is not None:
                asset_prices = asset_prices[asset_prices["block_number"] <= end_block]
            if asset_prices.empty:
                continue

            tier_state: dict[str, dict[str, float]] = {
                tier.trigger_tier: {"sold_qty": 0.0, "bought_qty": 0.0, "outstanding_qty": 0.0}
                for tier in scenario.tiers
            }
            dynamic_state = {
                "sold_qty": 0.0,
                "bought_qty": 0.0,
                "outstanding_qty": 0.0,
                "sell_lots": [],
                "last_sell_block": -10**12,
                "last_buy_block": -10**12,
            }

            first_price = float(asset_prices.iloc[0]["price_usd"])
            # Optional: initialize collateral from a target initial CR at the loan's
            # start price (C_0 = D * CR_0 / P_0), so each loan starts at a controlled
            # health regardless of when it begins. Backward compatible: if no
            # initial_cr column is provided, use the explicit collateral amount.
            initial_cr_raw = pos.get("initial_cr") if "initial_cr" in positions_df.columns else None
            if initial_cr_raw is not None and pd.notna(initial_cr_raw) and float(initial_cr_raw) > 0 and debt > 0:
                collateral = debt * float(initial_cr_raw) / first_price
            initial_collateral = collateral
            initial_debt = debt
            min_hf = float("inf")
            max_ltv = 0.0
            min_cr = float("inf")
            max_bad_debt = 0.0
            total_sell_usd = 0.0
            total_buy_usd = 0.0
            total_debt_repaid = 0.0
            sell_events = 0
            buy_events = 0

            prev_hf, _, _ = risk_metrics(collateral, debt, first_price, lt)
            price_history: list[float] = []

            for _, row in asset_prices.iterrows():
                block_number = int(row["block_number"])
                timestamp = str(row["timestamp"])
                price = float(row["price_usd"])
                price_history.append(price)
                hf_now, ltv_now, cr_now = risk_metrics(collateral, debt, price, lt)

                if scenario.dynamic_policy is not None:
                    policy = scenario.dynamic_policy
                    should_sell = (
                        ltv_now >= policy.lltv
                        and debt > 0
                        and collateral > 0
                        and (block_number - int(dynamic_state["last_sell_block"])) >= policy.sell_cooldown_steps
                    )
                    if should_sell:
                        if policy.target_hf is not None:
                            # Target-HF repair sizing (paper eq. 56).
                            debt_repaid = target_hf_debt_repaid(collateral, debt, price, lt, policy)
                        else:
                            # LTV-proportional close-factor sizing.
                            close_factor = dynamic_close_factor(ltv_now, policy)
                            debt_repaid = min(close_factor * debt, debt)
                        collateral_sold = (1.0 + policy.liquidation_bonus) * debt_repaid / price
                        if collateral_sold > collateral:
                            collateral_sold = collateral
                            debt_repaid = min(debt, collateral_sold * price / (1.0 + policy.liquidation_bonus))

                        if debt_repaid > 0 and collateral_sold > 0:
                            sell_value = collateral_sold * price
                            reserve_gain = max(0.0, sell_value - debt_repaid)

                            collateral -= collateral_sold
                            debt -= debt_repaid
                            reserve_usd += reserve_gain

                            dynamic_state["sold_qty"] += collateral_sold
                            dynamic_state["outstanding_qty"] += collateral_sold
                            dynamic_state["sell_lots"].append(
                                {
                                    "qty": float(collateral_sold),
                                    "sell_price": float(price),
                                }
                            )
                            dynamic_state["last_sell_block"] = block_number

                            total_sell_usd += sell_value
                            total_debt_repaid += debt_repaid
                            sell_events += 1

                            event_rows.append(
                                {
                                    "run_id": run_id,
                                    "scenario": scenario.name,
                                    "account": account,
                                    "asset_symbol": symbol,
                                    "block_number": block_number,
                                    "timestamp": timestamp,
                                    "event": "SELL",
                                    "trigger_tier": "Dynamic",
                                    "hf": hf_now,
                                    "ltv": ltv_now,
                                    "price_usd": price,
                                    "collateral_amount": collateral_sold,
                                    "debt_repaid": debt_repaid,
                                    "reserve_change_usd": reserve_gain,
                                }
                            )
                            hf_now, ltv_now, cr_now = risk_metrics(collateral, debt, price, lt)

                    buy_risk_guard_ltv = min(1.0, policy.lltv + policy.recovery_ltv_gap)
                    # WHEN to buy: optional confirmed-upturn gate. Only buy if the
                    # price has risen versus `lookback` steps ago, so the recovery
                    # leg does not catch a falling knife during multi-leg crashes.
                    uptrend_ok = True
                    if policy.buyback_uptrend_lookback > 0:
                        L = policy.buyback_uptrend_lookback
                        if len(price_history) > L:
                            recent_low = min(price_history[-(L + 1):])
                            uptrend_ok = price >= recent_low * (1.0 + policy.buyback_min_bounce)
                        else:
                            uptrend_ok = False
                    should_buy = (
                        policy.enable_buyback
                        and policy.buyback_ratio > 0
                        and uptrend_ok
                        and ltv_now <= buy_risk_guard_ltv
                        and dynamic_state["outstanding_qty"] > 0
                        and (block_number - int(dynamic_state["last_buy_block"])) >= policy.buy_cooldown_steps
                    )
                    if should_buy:
                        target_buy = policy.buyback_ratio * dynamic_state["sold_qty"]
                        remaining_target = max(0.0, target_buy - dynamic_state["bought_qty"])
                        # Band-bottom buy: trigger when the price is below the band
                        # (an open sell lot priced above (1+s)*P_t exists). The recovery
                        # quantity is NOT a 1:1 match to any single sell lot; it is sized
                        # by solvency (the HF-floor reborrow cap) and the aggregate band
                        # inventory, and consumed across lots from the band top down.
                        eligible: list[dict] = []
                        if remaining_target > 0:
                            spread_gate = price * (1.0 + policy.min_buyback_spread)
                            eligible = sorted(
                                (lot for lot in dynamic_state["sell_lots"]
                                 if lot["qty"] > 0 and lot["sell_price"] > spread_gate),
                                key=lambda lot: lot["sell_price"],
                                reverse=True,  # band top (highest sell price) first
                            )

                        buy_qty = 0.0
                        if eligible:
                            if policy.buyback_hf_floor is not None and policy.buyback_hf_floor > lt:
                                # Solvency cap: keep post-buy HF >= floor. Buying adds equal
                                # USD of collateral and debt, lowering HF, so this cap (not a
                                # lot match) is what sizes the recovery and blocks ping-pong.
                                floor = policy.buyback_hf_floor
                                borrow_capacity_usd = max(
                                    0.0, (collateral * price * lt - floor * debt) / (floor - lt)
                                )
                            else:
                                borrow_capacity_usd = max(0.0, collateral * price * policy.lltv - debt)
                            # HOW MUCH to buy: stress-tested cap. Size the reborrow so the
                            # post-buy HF stays >= 1 even after a further `d` price drop
                            # (same B* form with LT -> LT*(1-d)), so a buy made mid-decline
                            # cannot re-arm the trigger on the next down-leg.
                            d = policy.buyback_stress_drawdown
                            if d > 0.0:
                                lt_s = lt * (1.0 - d)
                                if lt_s < 1.0:  # 1.0 = stress health floor
                                    stress_capacity_usd = max(
                                        0.0, (collateral * price * lt_s - 1.0 * debt) / (1.0 - lt_s)
                                    )
                                    borrow_capacity_usd = min(borrow_capacity_usd, stress_capacity_usd)
                            affordable_qty = max(0.0, borrow_capacity_usd / price)
                            eligible_qty = sum(float(lot["qty"]) for lot in eligible)
                            buy_qty = max(0.0, min(remaining_target, affordable_qty, eligible_qty))

                        if buy_qty > 0:
                            buy_cost = buy_qty * price
                            reserve_change = 0.0
                            debt_change = buy_cost

                            collateral += buy_qty
                            reserve_usd += reserve_change
                            debt += debt_change
                            dynamic_state["bought_qty"] += buy_qty
                            dynamic_state["outstanding_qty"] -= buy_qty
                            # Consume the bought quantity across the band, top first.
                            remaining_buy = buy_qty
                            for lot in eligible:
                                if remaining_buy <= 1e-15:
                                    break
                                take = min(float(lot["qty"]), remaining_buy)
                                lot["qty"] = max(0.0, float(lot["qty"]) - take)
                                remaining_buy -= take
                            dynamic_state["last_buy_block"] = block_number

                            total_buy_usd += buy_cost
                            buy_events += 1

                            event_rows.append(
                                {
                                    "run_id": run_id,
                                    "scenario": scenario.name,
                                    "account": account,
                                    "asset_symbol": symbol,
                                    "block_number": block_number,
                                    "timestamp": timestamp,
                                    "event": "BUY",
                                    "trigger_tier": "Dynamic",
                                    "hf": hf_now,
                                    "ltv": ltv_now,
                                    "price_usd": price,
                                    "collateral_amount": buy_qty,
                                    "debt_repaid": 0.0,
                                    "reserve_change_usd": reserve_change,
                                }
                            )
                            hf_now, ltv_now, cr_now = risk_metrics(collateral, debt, price, lt)
                else:
                    for tier in scenario.tiers:
                        if prev_hf > tier.hf_down and hf_now <= tier.hf_down and debt > 0 and collateral > 0:
                            debt_repaid = min(tier.close_factor * debt, debt)
                            collateral_sold = (1.0 + tier.liquidation_bonus) * debt_repaid / price
                            if collateral_sold > collateral:
                                collateral_sold = collateral
                                debt_repaid = min(debt, collateral_sold * price / (1.0 + tier.liquidation_bonus))

                            sell_value = collateral_sold * price
                            reserve_gain = max(0.0, sell_value - debt_repaid)

                            collateral -= collateral_sold
                            debt -= debt_repaid
                            reserve_usd += reserve_gain

                            state = tier_state[tier.trigger_tier]
                            state["sold_qty"] += collateral_sold
                            state["outstanding_qty"] += collateral_sold

                            total_sell_usd += sell_value
                            total_debt_repaid += debt_repaid
                            sell_events += 1

                            event_rows.append(
                                {
                                    "run_id": run_id,
                                    "scenario": scenario.name,
                                    "account": account,
                                    "asset_symbol": symbol,
                                    "block_number": block_number,
                                    "timestamp": timestamp,
                                    "event": "SELL",
                                    "trigger_tier": tier.trigger_tier,
                                    "hf": hf_now,
                                    "ltv": ltv_now,
                                    "price_usd": price,
                                    "collateral_amount": collateral_sold,
                                    "debt_repaid": debt_repaid,
                                    "reserve_change_usd": reserve_gain,
                                }
                            )
                            hf_now, ltv_now, cr_now = risk_metrics(collateral, debt, price, lt)

                    for tier in sorted(scenario.tiers, key=lambda item: item.hf_up):
                        state = tier_state[tier.trigger_tier]
                        if prev_hf < tier.hf_up and hf_now >= tier.hf_up and state["outstanding_qty"] > 0 and reserve_usd > 0:
                            target_buy = tier.buyback_ratio * state["sold_qty"]
                            target_incremental = max(0.0, target_buy - state["bought_qty"])
                            if target_incremental <= 0:
                                continue

                            affordable = min(target_incremental, state["outstanding_qty"], reserve_usd / price)
                            if affordable <= 0:
                                continue

                            buy_cost = affordable * price
                            collateral += affordable
                            reserve_usd -= buy_cost
                            state["bought_qty"] += affordable
                            state["outstanding_qty"] -= affordable

                            total_buy_usd += buy_cost
                            buy_events += 1

                            event_rows.append(
                                {
                                    "run_id": run_id,
                                    "scenario": scenario.name,
                                    "account": account,
                                    "asset_symbol": symbol,
                                    "block_number": block_number,
                                    "timestamp": timestamp,
                                    "event": "BUY",
                                    "trigger_tier": tier.trigger_tier,
                                    "hf": hf_now,
                                    "ltv": ltv_now,
                                    "price_usd": price,
                                    "collateral_amount": affordable,
                                    "debt_repaid": 0.0,
                                    "reserve_change_usd": -buy_cost,
                                }
                            )
                            hf_now, ltv_now, cr_now = risk_metrics(collateral, debt, price, lt)

                bad_debt = max(0.0, debt - collateral * price)
                min_hf = min(min_hf, hf_now)
                max_ltv = max(max_ltv, ltv_now if np.isfinite(ltv_now) else 0.0)
                min_cr = min(min_cr, cr_now)
                max_bad_debt = max(max_bad_debt, bad_debt)
                prev_hf = hf_now

            final_price = float(asset_prices.iloc[-1]["price_usd"])
            restoration_ratio = collateral / initial_collateral if initial_collateral > 0 else np.nan
            impermanent_loss_pct = max(0.0, (1 - restoration_ratio) * 100)
            initial_equity_usd = initial_collateral * final_price - initial_debt
            final_equity_usd = collateral * final_price - debt
            borrower_final_loss_usd = max(0.0, initial_equity_usd - final_equity_usd)

            account_rows.append(
                {
                    "run_id": run_id,
                    "scenario": scenario.name,
                    "account": account,
                    "asset_symbol": symbol,
                    "initial_price_usd": first_price,
                    "final_price_usd": final_price,
                    "initial_collateral": initial_collateral,
                    "final_collateral": collateral,
                    "initial_debt_usd": initial_debt,
                    "final_debt_usd": debt,
                    "protocol_profit_usd": reserve_usd,
                    "total_sell_usd": total_sell_usd,
                    "total_buy_usd": total_buy_usd,
                    "total_debt_repaid_usd": total_debt_repaid,
                    "restoration_ratio": restoration_ratio,
                    "impermanent_loss_pct": impermanent_loss_pct,
                    "borrower_final_loss_usd": borrower_final_loss_usd,
                    "min_hf": min_hf,
                    "max_ltv": max_ltv,
                    "min_cr": min_cr,
                    "max_bad_debt_usd": max_bad_debt,
                    "sell_events": sell_events,
                    "buy_events": buy_events,
                }
            )

    events_df = pd.DataFrame(event_rows)
    account_df = pd.DataFrame(account_rows)
    metrics_df = (
        account_df.groupby("scenario", as_index=False)
        .agg(
            accounts=("account", "count"),
            avg_protocol_profit_usd=("protocol_profit_usd", "mean"),
            avg_impermanent_loss_pct=("impermanent_loss_pct", "mean"),
            avg_restoration_ratio=("restoration_ratio", "mean"),
            avg_borrower_final_loss_usd=("borrower_final_loss_usd", "mean"),
            avg_min_hf=("min_hf", "mean"),
            avg_max_ltv=("max_ltv", "mean"),
            max_bad_debt_usd=("max_bad_debt_usd", "max"),
            total_sell_events=("sell_events", "sum"),
            total_buy_events=("buy_events", "sum"),
        )
        .sort_values("scenario")
        .reset_index(drop=True)
    )

    events_df.to_csv(run_dir / "event_log.csv", index=False)
    account_df.to_csv(run_dir / "account_outcomes.csv", index=False)
    metrics_df.to_csv(run_dir / "scenario_metrics.csv", index=False)
    (run_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
                "dataset_dir": str(dataset_dir),
                "scenario_file": str(scenario_path),
                "scenarios": [scenario.name for scenario in scenarios],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run_dir


def write_batch_report(batch_dir: Path, summary_df: pd.DataFrame) -> Path:
    report_lines = [
        "# Batch Counterfactual Simulation Report",
        "",
        f"- Batch directory: {batch_dir}",
        f"- Generated at: {datetime.now(tz=timezone.utc).isoformat()}",
        f"- Total batch runs: {summary_df['batch_run_id'].nunique() if not summary_df.empty else 0}",
        f"- Total scenario rows: {len(summary_df)}",
        "",
        "## Per-run Scenario Metrics",
        "",
        render_table(summary_df),
    ]

    if not summary_df.empty:
        agg = (
            summary_df.groupby("scenario", as_index=False)
            .agg(
                runs=("batch_run_id", "count"),
                mean_avg_protocol_profit_usd=("avg_protocol_profit_usd", "mean"),
                mean_avg_impermanent_loss_pct=("avg_impermanent_loss_pct", "mean"),
                mean_avg_borrower_final_loss_usd=("avg_borrower_final_loss_usd", "mean"),
                mean_avg_min_hf=("avg_min_hf", "mean"),
                worst_max_bad_debt_usd=("max_bad_debt_usd", "max"),
            )
            .sort_values("scenario")
            .reset_index(drop=True)
        )
        report_lines.extend(["", "## Scenario Aggregate Across Batch", "", render_table(agg)])

    report_path = batch_dir / "batch_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def generate_batch_topn(batch_dir: Path, top_n: int) -> tuple[Path, Path, Path]:
    summary_path = batch_dir / "batch_summary.csv"
    if not summary_path.exists():
        raise DatasetError(f"Missing batch summary file: {summary_path}")

    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        raise DatasetError(f"Batch summary is empty: {summary_path}")

    required_cols = [
        "batch_run_id",
        "scenario",
        "seed",
        "price_shock",
        "debt_scale",
        "noise_scale",
        "avg_protocol_profit_usd",
        "avg_impermanent_loss_pct",
        "avg_borrower_final_loss_usd",
        "avg_min_hf",
        "max_bad_debt_usd",
        "total_sell_events",
        "total_buy_events",
    ]
    ensure_columns(summary_df, required_cols, "batch_summary.csv")

    top_n = max(1, int(top_n))

    best_df = (
        summary_df.sort_values(
            [
                "avg_borrower_final_loss_usd",
                "max_bad_debt_usd",
                "avg_min_hf",
                "avg_protocol_profit_usd",
            ],
            ascending=[True, True, False, False],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    worst_df = (
        summary_df.sort_values(
            [
                "avg_borrower_final_loss_usd",
                "max_bad_debt_usd",
                "avg_min_hf",
                "avg_protocol_profit_usd",
            ],
            ascending=[False, False, True, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    best_path = batch_dir / f"topn_best_{top_n}.csv"
    worst_path = batch_dir / f"topn_worst_{top_n}.csv"
    best_df.to_csv(best_path, index=False)
    worst_df.to_csv(worst_path, index=False)

    md_lines = [
        f"# Batch Top-{top_n} Parameter Combination Summary",
        "",
        f"- Batch directory: {batch_dir}",
        f"- Generated at: {datetime.now(tz=timezone.utc).isoformat()}",
        "",
        "## Best Combinations (borrower_loss↓, bad_debt↓, min_hf↑)",
        "",
        render_table(best_df),
        "",
        "## Worst Combinations (borrower_loss↑, bad_debt↑, min_hf↓)",
        "",
        render_table(worst_df),
    ]
    md_path = batch_dir / f"topn_summary_{top_n}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return best_path, worst_path, md_path


def export_batch_buy_closures(batch_dir: Path, output_csv: Path | None, scenario: str | None) -> Path:
    runs_dir = batch_dir / "runs"
    if not runs_dir.exists():
        raise DatasetError(f"Missing runs directory in batch: {runs_dir}")

    rows: list[dict[str, Any]] = []
    run_dirs = sorted(path for path in runs_dir.iterdir() if path.is_dir() and path.name.startswith("run_"))
    if not run_dirs:
        raise DatasetError(f"No run directories found under {runs_dir}")

    for run_dir in run_dirs:
        event_path = run_dir / "event_log.csv"
        if not event_path.exists():
            continue
        try:
            events = pd.read_csv(event_path)
        except EmptyDataError:
            continue
        if events.empty:
            continue

        required_event_cols = [
            "scenario",
            "account",
            "asset_symbol",
            "trigger_tier",
            "event",
            "block_number",
            "timestamp",
            "collateral_amount",
            "reserve_change_usd",
        ]
        ensure_columns(events, required_event_cols, str(event_path))

        if scenario:
            events = events[events["scenario"] == scenario]
        if events.empty:
            continue

        events = events.copy().reset_index(drop=True)
        events["event_id"] = events.index
        events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce", utc=True)
        events["block_number"] = events["block_number"].astype(int)
        events["collateral_amount"] = events["collateral_amount"].astype(float)
        events["reserve_change_usd"] = events["reserve_change_usd"].astype(float)
        events = events.sort_values(
            ["scenario", "account", "asset_symbol", "trigger_tier", "block_number", "timestamp", "event_id"]
        )

        group_cols = ["scenario", "account", "asset_symbol", "trigger_tier"]
        for (scenario_name, account, asset_symbol, trigger_tier), group in events.groupby(group_cols):
            open_sells: list[dict[str, Any]] = []
            for _, event in group.iterrows():
                event_type = str(event["event"])
                collateral_amount = float(event["collateral_amount"])
                reserve_change = float(event["reserve_change_usd"])

                if event_type == "SELL" and collateral_amount > 0:
                    open_sells.append(
                        {
                            "event_id": int(event["event_id"]),
                            "block_number": int(event["block_number"]),
                            "timestamp": event["timestamp"],
                            "original_qty": collateral_amount,
                            "remaining_qty": collateral_amount,
                            "remaining_reserve": reserve_change,
                            "price_usd": float(event.get("price_usd", np.nan)),
                        }
                    )
                    continue

                if event_type != "BUY" or collateral_amount <= 0 or reserve_change >= 0:
                    continue

                buy_remaining_qty = collateral_amount
                buy_remaining_reserve = reserve_change
                buy_block = int(event["block_number"])
                buy_ts = event["timestamp"]
                buy_event_id = int(event["event_id"])
                buy_price = float(event.get("price_usd", np.nan))

                while buy_remaining_qty > 1e-12 and open_sells:
                    sell = open_sells[0]
                    matched_qty = min(buy_remaining_qty, sell["remaining_qty"])
                    if matched_qty <= 1e-12:
                        open_sells.pop(0)
                        continue

                    sell_reserve_per_qty = sell["remaining_reserve"] / sell["remaining_qty"] if sell["remaining_qty"] > 0 else 0.0
                    buy_reserve_per_qty = buy_remaining_reserve / buy_remaining_qty if buy_remaining_qty > 0 else 0.0
                    allocated_sell_reserve = sell_reserve_per_qty * matched_qty
                    allocated_buy_reserve = buy_reserve_per_qty * matched_qty

                    sell_ts = sell["timestamp"]
                    lag_hours = np.nan
                    if pd.notna(sell_ts) and pd.notna(buy_ts):
                        lag_hours = (buy_ts - sell_ts).total_seconds() / 3600.0

                    rows.append(
                        {
                            "batch_id": batch_dir.name,
                            "run_id": run_dir.name,
                            "scenario": scenario_name,
                            "account": account,
                            "asset_symbol": asset_symbol,
                            "trigger_tier": trigger_tier,
                            "sell_event_id": sell["event_id"],
                            "buy_event_id": buy_event_id,
                            "sell_block": sell["block_number"],
                            "buy_block": buy_block,
                            "sell_timestamp": sell_ts.isoformat() if pd.notna(sell_ts) else "",
                            "buy_timestamp": buy_ts.isoformat() if pd.notna(buy_ts) else "",
                            "sell_to_buy_lag_hours": lag_hours,
                            "matched_buy_qty": matched_qty,
                            "sell_original_qty": sell["original_qty"],
                            "buy_replenishment_ratio": matched_qty / sell["original_qty"] if sell["original_qty"] > 0 else np.nan,
                            "sell_price_usd": sell["price_usd"],
                            "buy_price_usd": buy_price,
                            "allocated_sell_reserve_usd": allocated_sell_reserve,
                            "allocated_buy_reserve_usd": allocated_buy_reserve,
                            "net_reserve_change_usd": allocated_sell_reserve + allocated_buy_reserve,
                        }
                    )

                    sell["remaining_qty"] -= matched_qty
                    sell["remaining_reserve"] -= allocated_sell_reserve
                    buy_remaining_qty -= matched_qty
                    buy_remaining_reserve -= allocated_buy_reserve

                    if sell["remaining_qty"] <= 1e-12:
                        open_sells.pop(0)

    closure_df = pd.DataFrame(rows)
    if output_csv is None:
        scenario_suffix = f"_{slugify(scenario)}" if scenario else ""
        output_csv = batch_dir / f"buy_closure_cases{scenario_suffix}.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    closure_df.to_csv(output_csv, index=False)
    return output_csv


def batch_simulate(
    dataset_dir: Path,
    scenario_path: Path,
    output_dir: Path,
    seeds: list[int],
    price_shocks: list[float],
    debt_scales: list[float],
    noise_scales: list[float],
    max_runs: int,
) -> tuple[Path, Path]:
    validate_dataset(dataset_dir)
    base_prices = pd.read_csv(dataset_dir / "normalized" / "prices.csv")
    base_positions = pd.read_csv(dataset_dir / "normalized" / "positions_initial.csv")

    combos = list(itertools.product(seeds, price_shocks, debt_scales, noise_scales))
    if len(combos) > max_runs:
        combos = combos[:max_runs]

    batch_id = datetime.now(tz=timezone.utc).strftime("batch_%Y%m%d_%H%M%S_%f")
    batch_dir = output_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for index, (seed, price_shock, debt_scale, noise_scale) in enumerate(combos, start=1):
        rng = np.random.default_rng(seed)

        prices = base_prices.copy()
        prices["price_usd"] = prices["price_usd"].astype(float)
        noise = rng.normal(loc=0.0, scale=noise_scale, size=len(prices))
        prices["price_usd"] = np.clip(prices["price_usd"] * (1.0 + price_shock + noise), 1e-8, None)

        positions = base_positions.copy()
        positions["debt_amount"] = positions["debt_amount"].astype(float) * debt_scale
        positions["debt_amount"] = np.clip(positions["debt_amount"], 1e-8, None)

        run_tag = f"{index:03d}_seed{seed}_ps{price_shock:+.3f}_ds{debt_scale:.3f}_ns{noise_scale:.3f}"
        run_slug = slugify(run_tag)
        run_dataset_dir = batch_dir / "datasets" / run_slug
        (run_dataset_dir / "normalized").mkdir(parents=True, exist_ok=True)
        prices.to_csv(run_dataset_dir / "normalized" / "prices.csv", index=False)
        positions.to_csv(run_dataset_dir / "normalized" / "positions_initial.csv", index=False)

        run_output_dir = batch_dir / "runs"
        run_id = f"run_{run_slug}"
        run_dir = run_counterfactual(
            dataset_dir=run_dataset_dir,
            scenario_path=scenario_path,
            output_dir=run_output_dir,
            run_id=run_id,
        )

        metrics = pd.read_csv(run_dir / "scenario_metrics.csv")
        metrics.insert(0, "batch_run_id", run_slug)
        metrics.insert(1, "seed", seed)
        metrics.insert(2, "price_shock", price_shock)
        metrics.insert(3, "debt_scale", debt_scale)
        metrics.insert(4, "noise_scale", noise_scale)
        summary_rows.extend(metrics.to_dict(orient="records"))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = batch_dir / "batch_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    report_path = write_batch_report(batch_dir=batch_dir, summary_df=summary_df)
    return summary_path, report_path


def historical_backtest(
    dataset_dir: Path,
    scenario_path: Path,
    output_dir: Path,
    window_size: int,
    window_step: int,
    max_windows: int,
    loan_mid_starts: bool,
    loan_min_duration_blocks: int,
) -> tuple[Path, Path, Path]:
    validate_dataset(dataset_dir)
    if window_size < 2:
        raise DatasetError("window_size must be >= 2")
    if window_step < 1:
        raise DatasetError("window_step must be >= 1")
    if max_windows < 1:
        raise DatasetError("max_windows must be >= 1")
    if loan_min_duration_blocks < 1:
        raise DatasetError("loan_min_duration_blocks must be >= 1")

    prices = pd.read_csv(dataset_dir / "normalized" / "prices.csv")
    positions = pd.read_csv(dataset_dir / "normalized" / "positions_initial.csv")
    prices = prices.sort_values(["block_number", "asset_symbol"]).reset_index(drop=True)

    unique_blocks = sorted(prices["block_number"].astype(int).unique().tolist())
    if len(unique_blocks) < window_size:
        raise DatasetError(
            f"Not enough price points for historical backtest: blocks={len(unique_blocks)}, window_size={window_size}"
        )

    starts = list(range(0, len(unique_blocks) - window_size + 1, window_step))
    if not starts:
        raise DatasetError("No valid windows generated")
    starts = starts[:max_windows]

    backtest_id = datetime.now(tz=timezone.utc).strftime("histbt_%Y%m%d_%H%M%S_%f")
    backtest_dir = output_dir / backtest_id
    backtest_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    runs_dir = backtest_dir / "runs"

    for idx, start in enumerate(starts, start=1):
        window_blocks = unique_blocks[start : start + window_size]
        start_block = int(window_blocks[0])
        end_block = int(window_blocks[-1])

        window_prices = prices[(prices["block_number"] >= start_block) & (prices["block_number"] <= end_block)].copy()
        if window_prices.empty:
            continue

        ts_sorted = pd.to_datetime(window_prices["timestamp"], errors="coerce", utc=True).sort_values()
        window_start_ts = ts_sorted.iloc[0].isoformat() if not ts_sorted.empty and pd.notna(ts_sorted.iloc[0]) else ""
        window_end_ts = ts_sorted.iloc[-1].isoformat() if not ts_sorted.empty and pd.notna(ts_sorted.iloc[-1]) else ""

        window_slug = f"window_{idx:03d}_b{start_block}_to_{end_block}"
        run_dataset_dir = backtest_dir / "datasets" / window_slug / "normalized"
        run_dataset_dir.mkdir(parents=True, exist_ok=True)
        window_prices.to_csv(run_dataset_dir / "prices.csv", index=False)

        positions_for_window = positions.copy()
        if loan_mid_starts:
            block_count = len(window_blocks)
            start_low_idx = max(1, int(0.15 * block_count))
            start_high_idx = max(start_low_idx, int(0.55 * block_count))
            end_high_idx = max(start_high_idx + 1, int(0.95 * block_count))

            span = max(1, start_high_idx - start_low_idx + 1)

            start_values: list[int] = []
            end_values: list[int] = []
            for _, pos_row in positions_for_window.iterrows():
                account = str(pos_row.get("account", ""))
                symbol = str(pos_row.get("asset_symbol", ""))
                key = f"{window_slug}|{account}|{symbol}"
                # Deterministic digest (Python's built-in hash() is salted per
                # process, which would make loan start/end sampling — and hence
                # the whole backtest — irreproducible across runs).
                digest = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
                start_idx = start_low_idx + (digest % span)
                min_end_idx = min(block_count - 1, start_idx + loan_min_duration_blocks)
                if min_end_idx >= end_high_idx:
                    end_idx = block_count - 1
                else:
                    end_span = max(1, end_high_idx - min_end_idx + 1)
                    end_idx = min_end_idx + ((digest // 97) % end_span)
                start_values.append(int(window_blocks[start_idx]))
                end_values.append(int(window_blocks[end_idx]))

            positions_for_window["start_block"] = start_values
            positions_for_window["end_block"] = end_values

        positions_for_window.to_csv(run_dataset_dir / "positions_initial.csv", index=False)

        run_dir = run_counterfactual(
            dataset_dir=run_dataset_dir.parent,
            scenario_path=scenario_path,
            output_dir=runs_dir,
            run_id=f"run_{window_slug}",
        )

        metrics = pd.read_csv(run_dir / "scenario_metrics.csv")
        metrics.insert(0, "window_id", window_slug)
        metrics.insert(1, "window_index", idx)
        metrics.insert(2, "start_block", start_block)
        metrics.insert(3, "end_block", end_block)
        metrics.insert(4, "start_timestamp", window_start_ts)
        metrics.insert(5, "end_timestamp", window_end_ts)
        summary_rows.extend(metrics.to_dict(orient="records"))

        window_rows.append(
            {
                "window_id": window_slug,
                "window_index": idx,
                "start_block": start_block,
                "end_block": end_block,
                "start_timestamp": window_start_ts,
                "end_timestamp": window_end_ts,
                "price_rows": int(len(window_prices)),
                "asset_count": int(window_prices["asset_symbol"].nunique()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    windows_df = pd.DataFrame(window_rows)
    if summary_df.empty:
        raise DatasetError("Historical backtest produced no scenario metrics")

    summary_path = backtest_dir / "backtest_summary.csv"
    windows_path = backtest_dir / "backtest_windows.csv"
    summary_df.to_csv(summary_path, index=False)
    windows_df.to_csv(windows_path, index=False)

    agg_df = (
        summary_df.groupby("scenario", as_index=False)
        .agg(
            windows=("window_id", "count"),
            mean_avg_protocol_profit_usd=("avg_protocol_profit_usd", "mean"),
            mean_avg_impermanent_loss_pct=("avg_impermanent_loss_pct", "mean"),
            mean_avg_borrower_final_loss_usd=("avg_borrower_final_loss_usd", "mean"),
            p50_avg_borrower_final_loss_usd=("avg_borrower_final_loss_usd", "median"),
            p90_avg_borrower_final_loss_usd=("avg_borrower_final_loss_usd", lambda s: float(s.quantile(0.90))),
            mean_avg_min_hf=("avg_min_hf", "mean"),
            worst_max_bad_debt_usd=("max_bad_debt_usd", "max"),
            total_sell_events=("total_sell_events", "sum"),
            total_buy_events=("total_buy_events", "sum"),
        )
        .sort_values("scenario")
        .reset_index(drop=True)
    )
    aggregate_path = backtest_dir / "backtest_aggregate.csv"
    agg_df.to_csv(aggregate_path, index=False)

    lines = [
        "# Historical Backtest Report",
        "",
        f"- Backtest directory: {backtest_dir}",
        f"- Generated at: {datetime.now(tz=timezone.utc).isoformat()}",
        f"- Window size (blocks): {window_size}",
        f"- Window step (blocks): {window_step}",
        f"- Windows executed: {len(windows_df)}",
        "",
        "## Scenario Aggregate Across Windows",
        "",
        render_table(agg_df),
    ]
    report_path = backtest_dir / "backtest_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    meta_path = backtest_dir / "backtest_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "backtest_id": backtest_id,
                "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
                "dataset_dir": str(dataset_dir),
                "scenario_file": str(scenario_path),
                "window_size": window_size,
                "window_step": window_step,
                "max_windows": max_windows,
                "loan_mid_starts": loan_mid_starts,
                "loan_min_duration_blocks": loan_min_duration_blocks,
                "windows_executed": int(len(windows_df)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return summary_path, aggregate_path, report_path


def write_report(run_dir: Path) -> Path:
    metrics_path = run_dir / "scenario_metrics.csv"
    accounts_path = run_dir / "account_outcomes.csv"
    if not metrics_path.exists() or not accounts_path.exists():
        raise DatasetError(f"Missing outputs in {run_dir}")

    metrics = pd.read_csv(metrics_path)
    accounts = pd.read_csv(accounts_path)

    baseline_name = metrics.iloc[0]["scenario"] if not metrics.empty else None
    lines: list[str] = [
        "# Counterfactual Simulation Report",
        "",
        f"- Run directory: {run_dir}",
        f"- Generated at: {datetime.now(tz=timezone.utc).isoformat()}",
        f"- Scenarios: {', '.join(metrics['scenario'].tolist())}",
        "",
        "## Scenario Metrics",
        "",
        render_table(metrics),
        "",
        "## Baseline Delta",
        "",
    ]

    if baseline_name and len(metrics) > 1:
        base_row = metrics[metrics["scenario"] == baseline_name].iloc[0]
        delta_rows = []
        for _, row in metrics.iterrows():
            if row["scenario"] == baseline_name:
                continue
            delta_rows.append(
                {
                    "scenario": row["scenario"],
                    "delta_avg_protocol_profit_usd": row["avg_protocol_profit_usd"] - base_row["avg_protocol_profit_usd"],
                    "delta_avg_impermanent_loss_pct": row["avg_impermanent_loss_pct"] - base_row["avg_impermanent_loss_pct"],
                    "delta_avg_borrower_final_loss_usd": row["avg_borrower_final_loss_usd"]
                    - base_row["avg_borrower_final_loss_usd"],
                    "delta_avg_min_hf": row["avg_min_hf"] - base_row["avg_min_hf"],
                    "delta_max_bad_debt_usd": row["max_bad_debt_usd"] - base_row["max_bad_debt_usd"],
                }
            )
        if delta_rows:
            delta_df = pd.DataFrame(delta_rows)
            lines.append(render_table(delta_df))
        else:
            lines.append("Only one scenario present; no baseline delta.")
    else:
        lines.append("Only one scenario present; no baseline delta.")

    lines.extend(
        [
            "",
            "## Data Coverage",
            "",
            f"- Accounts simulated: {accounts['account'].nunique() if not accounts.empty else 0}",
            f"- Asset symbols: {', '.join(sorted(accounts['asset_symbol'].unique().tolist())) if not accounts.empty else '-'}",
        ]
    )

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aave data collection and counterfactual simulation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init-dataset", help="Initialize dataset folders and templates")
    p_init.add_argument("--dataset-dir", type=Path, default=Path("data/aave"))
    p_init.add_argument("--no-sample-data", action="store_true")
    p_init.add_argument("--seed", type=int, default=42)

    p_collect = subparsers.add_parser("collect-prices", help="Collect price history from CoinGecko")
    p_collect.add_argument("--asset-id", type=str, required=True)
    p_collect.add_argument("--symbol", type=str, required=True)
    p_collect.add_argument("--start", type=str, required=True, help="ISO datetime, e.g. 2024-01-01T00:00:00+00:00")
    p_collect.add_argument("--end", type=str, required=True, help="ISO datetime, e.g. 2024-03-01T00:00:00+00:00")
    p_collect.add_argument("--out-csv", type=Path, default=Path("data/aave/normalized/prices.csv"))

    p_collect_subgraph = subparsers.add_parser(
        "collect-aave-subgraph",
        help="Collect positions and liquidation events from Aave-compatible subgraph",
    )
    p_collect_subgraph.add_argument("--endpoint", type=str, required=True, help="GraphQL endpoint URL")
    p_collect_subgraph.add_argument("--market", type=str, default="aave-v3")
    p_collect_subgraph.add_argument("--symbols", type=str, required=True, help="Comma-separated symbols, e.g. WETH,WBTC")
    p_collect_subgraph.add_argument("--dataset-dir", type=Path, default=Path("data/aave"))
    p_collect_subgraph.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional start ISO datetime for liquidation events window",
    )
    p_collect_subgraph.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional end ISO datetime for liquidation events window",
    )
    p_collect_subgraph.add_argument("--page-size", type=int, default=1000)
    p_collect_subgraph.add_argument("--max-pages", type=int, default=25)
    p_collect_subgraph.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional bearer token for GraphQL gateway",
    )

    p_collect_dune = subparsers.add_parser(
        "collect-dune",
        help="Collect Aave-related data from Dune query results and map to normalized tables",
    )
    p_collect_dune.add_argument("--dataset-dir", type=Path, default=Path("data/aave"))
    p_collect_dune.add_argument("--dune-api-key", type=str, required=True)
    p_collect_dune.add_argument("--liquidation-query-id", type=int, default=2104473)
    p_collect_dune.add_argument("--positions-query-id", type=int, default=None)
    p_collect_dune.add_argument("--liquidation-threshold", type=float, default=0.83)
    p_collect_dune.add_argument("--collect-prices", action="store_true")
    p_collect_dune.add_argument(
        "--symbol-map",
        type=str,
        default=None,
        help="Optional symbol to CoinGecko mapping, format: WETH=ethereum,WBTC=wrapped-bitcoin",
    )

    p_list_presets = subparsers.add_parser("list-presets", help="List built-in Aave market presets")
    p_list_presets.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    p_bootstrap = subparsers.add_parser(
        "bootstrap-aave-market",
        help="Near-zero-parameter bootstrap: collect Aave positions/events + mapped prices",
    )
    p_bootstrap.add_argument("--preset", type=str, default="ethereum-v3")
    p_bootstrap.add_argument("--dataset-dir", type=Path, default=Path("data/aave"))
    p_bootstrap.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Optional comma-separated subset of preset symbols",
    )
    p_bootstrap.add_argument("--start", type=str, default=None, help="Optional ISO datetime, default now-30d")
    p_bootstrap.add_argument("--end", type=str, default=None, help="Optional ISO datetime, default now")
    p_bootstrap.add_argument("--endpoint", type=str, default=None, help="Optional endpoint override")
    p_bootstrap.add_argument("--page-size", type=int, default=1000)
    p_bootstrap.add_argument("--max-pages", type=int, default=25)
    p_bootstrap.add_argument("--api-key", type=str, default=None)

    p_validate = subparsers.add_parser("validate-dataset", help="Validate normalized dataset files")
    p_validate.add_argument("--dataset-dir", type=Path, default=Path("data/aave"))

    p_simulate = subparsers.add_parser("simulate", help="Run counterfactual replay for all scenarios")
    p_simulate.add_argument("--dataset-dir", type=Path, default=Path("data/aave"))
    p_simulate.add_argument("--scenario-file", type=Path, default=Path("data/aave/config/scenarios.json"))
    p_simulate.add_argument("--output-dir", type=Path, default=Path("runs"))
    p_simulate.add_argument("--run-id", type=str, default=None)

    p_report = subparsers.add_parser("report", help="Generate markdown summary from run outputs")
    p_report.add_argument("--run-dir", type=Path, required=True)

    p_batch = subparsers.add_parser("batch-simulate", help="Run a batch of synthetic stress simulations")
    p_batch.add_argument("--dataset-dir", type=Path, default=Path("data/aave"))
    p_batch.add_argument("--scenario-file", type=Path, default=Path("data/aave/config/scenarios.json"))
    p_batch.add_argument("--output-dir", type=Path, default=Path("runs"))
    p_batch.add_argument("--seeds", type=str, default="41,42,43")
    p_batch.add_argument("--price-shocks", type=str, default="-0.15,0.00,0.10")
    p_batch.add_argument("--debt-scales", type=str, default="0.95,1.00,1.05")
    p_batch.add_argument("--noise-scales", type=str, default="0.00,0.01")
    p_batch.add_argument("--max-runs", type=int, default=18)

    p_hist = subparsers.add_parser(
        "historical-backtest",
        help="Run rolling-window historical path backtests under fixed scenario parameters",
    )
    p_hist.add_argument("--dataset-dir", type=Path, default=Path("data/aave"))
    p_hist.add_argument("--scenario-file", type=Path, default=Path("data/aave/config/scenarios.json"))
    p_hist.add_argument("--output-dir", type=Path, default=Path("runs"))
    p_hist.add_argument("--window-size", type=int, default=120, help="Window size in unique price blocks")
    p_hist.add_argument("--window-step", type=int, default=24, help="Rolling step in unique price blocks")
    p_hist.add_argument("--max-windows", type=int, default=24)
    p_hist.add_argument(
        "--loan-mid-starts",
        action="store_true",
        help="When set, each loan starts/ends inside the window rather than always at window boundaries",
    )
    p_hist.add_argument(
        "--loan-min-duration-blocks",
        type=int,
        default=36,
        help="Minimum active duration (in blocks) for sampled mid-window loans",
    )

    p_batch_topn = subparsers.add_parser("batch-topn", help="Generate best/worst Top-N summary tables from batch outputs")
    p_batch_topn.add_argument("--batch-dir", type=Path, required=True)
    p_batch_topn.add_argument("--top-n", type=int, default=5)

    p_batch_closures = subparsers.add_parser(
        "batch-buy-closures",
        help="Export SELL->BUY closure cases from batch runs",
    )
    p_batch_closures.add_argument("--batch-dir", type=Path, required=True)
    p_batch_closures.add_argument("--output-csv", type=Path, default=None)
    p_batch_closures.add_argument("--scenario", type=str, default=None, help="Optional scenario filter, e.g. conservative")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "init-dataset":
        init_dataset(args.dataset_dir, with_sample_data=not args.no_sample_data, seed=args.seed)
        print(f"Initialized dataset at: {args.dataset_dir}")
        return

    if args.command == "collect-prices":
        collect_prices_coingecko(
            asset_id=args.asset_id,
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            out_csv=args.out_csv,
        )
        print(f"Collected prices to: {args.out_csv}")
        return

    if args.command == "collect-aave-subgraph":
        symbols = parse_symbols(args.symbols)
        result = collect_aave_subgraph(
            endpoint=args.endpoint,
            market=args.market,
            symbols=symbols,
            dataset_dir=args.dataset_dir,
            start=args.start,
            end=args.end,
            page_size=args.page_size,
            max_pages=args.max_pages,
            api_key=args.api_key,
        )
        print(f"Collected normalized positions: {result['positions_count']}")
        print(f"Collected normalized liquidation events: {result['liquidation_events_count']}")
        print(f"Updated dataset directory: {args.dataset_dir}")
        return

    if args.command == "collect-dune":
        symbol_map = dict(DUNE_DEFAULT_COINGECKO_MAP)
        if args.symbol_map:
            pairs = [item.strip() for item in args.symbol_map.split(",") if item.strip()]
            for pair in pairs:
                if "=" not in pair:
                    raise DatasetError(f"Invalid symbol-map entry: {pair}")
                symbol, asset_id = pair.split("=", 1)
                symbol_map[symbol.strip().upper()] = asset_id.strip()

        result = collect_dune_channel(
            dataset_dir=args.dataset_dir,
            dune_api_key=args.dune_api_key,
            liquidation_query_id=args.liquidation_query_id,
            positions_query_id=args.positions_query_id,
            liquidation_threshold=args.liquidation_threshold,
            collect_prices=args.collect_prices,
            symbol_map=symbol_map,
        )
        print(f"Dune liquidation events: {result['liquidation_events_count']}")
        print(f"Dune positions written: {result['positions_written']} ({result['positions_count']})")
        print(f"Price rows by symbol: {json.dumps(result['price_rows_by_symbol'], ensure_ascii=False)}")
        print(f"Updated dataset directory: {args.dataset_dir}")
        return

    if args.command == "list-presets":
        rows = []
        for preset in AAVE_PRESETS.values():
            rows.append(
                {
                    "preset": preset.name,
                    "market": preset.market,
                    "endpoint": preset.endpoint,
                    "default_symbols": ",".join(preset.default_symbols),
                }
            )
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print(render_table(pd.DataFrame(rows)))
        return

    if args.command == "bootstrap-aave-market":
        selected_symbols = parse_symbols(args.symbols) if args.symbols else None
        result = bootstrap_aave_market(
            preset_name=args.preset,
            dataset_dir=args.dataset_dir,
            symbols=selected_symbols,
            start=args.start,
            end=args.end,
            endpoint_override=args.endpoint,
            page_size=args.page_size,
            max_pages=args.max_pages,
            api_key=args.api_key,
        )
        print(f"Preset: {result['preset']} ({result['market']})")
        print(f"Endpoint: {result['endpoint']}")
        print(f"Time window: {result['start']} -> {result['end']}")
        print(f"Symbols: {', '.join(result['symbols'])}")
        print(f"Normalized positions: {result['positions_count']}")
        print(f"Normalized liquidation events: {result['liquidation_events_count']}")
        print(f"Price rows by symbol: {json.dumps(result['price_rows_by_symbol'], ensure_ascii=False)}")
        print(f"Updated dataset directory: {args.dataset_dir}")
        return

    if args.command == "validate-dataset":
        validate_dataset(args.dataset_dir)
        print(f"Dataset validated: {args.dataset_dir}")
        return

    if args.command == "simulate":
        run_dir = run_counterfactual(
            dataset_dir=args.dataset_dir,
            scenario_path=args.scenario_file,
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
        print(f"Simulation completed: {run_dir}")
        return

    if args.command == "batch-simulate":
        summary_path, report_path = batch_simulate(
            dataset_dir=args.dataset_dir,
            scenario_path=args.scenario_file,
            output_dir=args.output_dir,
            seeds=parse_int_list(args.seeds),
            price_shocks=parse_float_list(args.price_shocks),
            debt_scales=parse_float_list(args.debt_scales),
            noise_scales=parse_float_list(args.noise_scales),
            max_runs=args.max_runs,
        )
        print(f"Batch simulation summary: {summary_path}")
        print(f"Batch simulation report: {report_path}")
        return

    if args.command == "historical-backtest":
        summary_path, aggregate_path, report_path = historical_backtest(
            dataset_dir=args.dataset_dir,
            scenario_path=args.scenario_file,
            output_dir=args.output_dir,
            window_size=args.window_size,
            window_step=args.window_step,
            max_windows=args.max_windows,
            loan_mid_starts=args.loan_mid_starts,
            loan_min_duration_blocks=args.loan_min_duration_blocks,
        )
        print(f"Historical backtest summary: {summary_path}")
        print(f"Historical backtest aggregate: {aggregate_path}")
        print(f"Historical backtest report: {report_path}")
        return

    if args.command == "batch-topn":
        best_path, worst_path, md_path = generate_batch_topn(batch_dir=args.batch_dir, top_n=args.top_n)
        print(f"Top-N best table: {best_path}")
        print(f"Top-N worst table: {worst_path}")
        print(f"Top-N markdown summary: {md_path}")
        return

    if args.command == "batch-buy-closures":
        output_csv = export_batch_buy_closures(batch_dir=args.batch_dir, output_csv=args.output_csv, scenario=args.scenario)
        print(f"BUY closure cases CSV: {output_csv}")
        return

    if args.command == "report":
        report_path = write_report(args.run_dir)
        print(f"Report generated: {report_path}")
        return

    raise DatasetError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
