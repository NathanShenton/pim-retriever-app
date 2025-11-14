#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit app: Paste an internal PIM URL up to & including attribute_codes=...,
then the app appends &skus=... (repeated params) in safe batches and compiles results.

Security & Ops:
- URL input is masked (password field) and never cached/logged by default.
- Host allow-listed (default: *.hbi.systems); adjust ALLOWED_SUFFIXES if needed.
- Bearer token is read from env var PIM_TOKEN (optional).
- Caching of responses is disabled by default (toggleable).
- Raw JSON download is off by default (toggleable).

Requirements:
    streamlit>=1.36
    pandas>=2.2
    httpx>=0.27
    pyarrow>=17 (optional, for Parquet download)
"""

import io
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import httpx
import pandas as pd
import streamlit as st
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

# =========================
# Config / Security
# =========================

# Only allow requests to hosts ending with these suffixes
ALLOWED_SUFFIXES = (".hbi.systems",)

# Reasonable safety default for max SKUs per request to avoid long URLs (414)
DEFAULT_CHUNK_SIZE = 900

# =========================
# Helpers
# =========================

def auth_headers() -> Dict[str, str]:
    """Attach Authorization header if PIM_TOKEN is set."""
    token = os.getenv("PIM_TOKEN")
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def get_client(timeout_s: int = 60) -> httpx.Client:
    """Create an httpx client; only pass 'proxies' if they’re explicitly set."""
    proxies_env = {
        "http://": os.getenv("HTTP_PROXY"),
        "https://": os.getenv("HTTPS_PROXY"),
    }
    # keep only non-empty
    proxies_env = {k: v for k, v in proxies_env.items() if v}

    kwargs = dict(timeout=timeout_s, headers=auth_headers())
    if proxies_env:
        kwargs["proxies"] = proxies_env  # only include if present

    return httpx.Client(**kwargs)

def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]

def clean_skus(raw: str) -> List[str]:
    """
    Accept newline/comma/semicolon/tab separated SKUs, trim, de-dup (keep order).
    """
    if not raw:
        return []
    raw = raw.replace(";", "\n").replace(",", "\n").replace("\t", "\n")
    skus = [s.strip() for s in raw.splitlines() if s.strip()]
    seen, out = set(), []
    for s in skus:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def pad_sku(s: str, width: int = 6) -> str:
    """
    Left-pad purely numeric SKUs to a fixed width with zeros; leave others untouched.
    '42' -> '000042', 'ABC123' -> 'ABC123'
    """
    return s.zfill(width) if s.isdigit() and len(s) < width else s

def normalize_response(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Flatten common API shapes into a SKU-wide table. Falls back to json_normalize.
    Supported patterns:
      1) {"data":[{"sku":"...","attributes":{"code":"val",...}}, ...]}
      2) {"data":[{"sku":"...","attribute_code":"...","value":...}, ...]}  (long-form)
      3) {"skus":{"SKU1":{"code":"val",...},"SKU2":{...}}}
      4) other list/dict -> json_normalize
    """
    if not payload:
        return pd.DataFrame()

    data = payload.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        # case 1: attributes dict
        if "sku" in data[0] and isinstance(data[0].get("attributes"), dict):
            rows = []
            for item in data:
                sku = item.get("sku")
                attrs = item.get("attributes") or {}
                row = {"sku": sku}
                for k, v in attrs.items():
                    row[str(k)] = v
                rows.append(row)
            return pd.DataFrame(rows)

        # case 2: long-form per-attribute
        if "sku" in data[0] and ("attribute_code" in data[0] or "code" in data[0]):
            df = pd.DataFrame(data)
            code_col = "attribute_code" if "attribute_code" in df.columns else "code"
            val_col = "value" if "value" in df.columns else (
                "attribute_value" if "attribute_value" in df.columns else None
            )
            if val_col:
                pivot = (
                    df.pivot_table(index="sku", columns=code_col, values=val_col, aggfunc="first")
                      .reset_index()
                )
                pivot.columns.name = None
                return pivot
            return pd.json_normalize(data)

        # unknown 'data' list
        return pd.json_normalize(data)

    # case 3: map of skus -> attr dicts
    skus_map = payload.get("skus")
    if isinstance(skus_map, dict):
        rows = []
        for sku, attrs in skus_map.items():
            row = {"sku": sku}
            if isinstance(attrs, dict):
                for k, v in attrs.items():
                    row[str(k)] = v
            else:
                row["value"] = attrs
            rows.append(row)
        return pd.DataFrame(rows)

    # fallback(s)
    if isinstance(payload, list):
        return pd.json_normalize(payload)
    return pd.json_normalize(payload)

def parse_user_url(user_url: str) -> Tuple[str, Dict[str, List[str]]]:
    """
    Split the user's URL into (base_url_without_query, params_dict_of_lists).
    - Validates host against ALLOWED_SUFFIXES
    - Requires 'attribute_codes' present
    - Requires that no 'skus' are present (we append them)
    """
    if not user_url:
        raise ValueError("URL is required.")

    u = urlparse(user_url)
    host = u.hostname or ""
    if not any(host.endswith(suf) for suf in ALLOWED_SUFFIXES):
        raise ValueError("Unapproved host. Connect to VPN and use an internal *.hbi.systems URL.")

    q_items = parse_qsl(u.query, keep_blank_values=True)
    params: Dict[str, List[str]] = {}
    for k, v in q_items:
        params.setdefault(k, []).append(v)

    if "attribute_codes" not in params:
        raise ValueError("URL must include attribute_codes=...")

    if "skus" in params:
        raise ValueError("URL should NOT include skus; the app will append them.")

    base_no_query = urlunparse((u.scheme, u.netloc, u.path, "", "", ""))
    return base_no_query, params

def build_params_with_skus(params_base: Dict[str, List[str]], skus: List[str]) -> Dict[str, List[str]]:
    """
    Clone the base params and append SKUs as repeated keys (&skus=A&skus=B...).
    httpx will automatically encode list values as repeated query params.
    """
    params = {k: list(v) for k, v in params_base.items()}
    params["skus"] = skus
    return params

def backoff_sleep(attempt: int, base: float = 0.5, cap: float = 20.0):
    """Exponential backoff with jitter-ish increment."""
    delay = min(cap, base * (2 ** attempt)) + (0.1 * (attempt + 1))
    time.sleep(delay)

def fetch_batch(
    client: httpx.Client,
    base_url: str,
    params: Dict[str, List[str]],
    max_retries: int = 6
) -> Dict[str, Any]:
    attempt = 0
    while True:
        try:
            resp = client.get(base_url, params=params)
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt >= max_retries:
                    resp.raise_for_status()
                backoff_sleep(attempt)
                attempt += 1
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.RequestError:
            if attempt >= max_retries:
                raise
            backoff_sleep(attempt)
            attempt += 1

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="PIM SKU Attribute Fetcher", layout="wide")
st.title("PIM SKU Attribute Fetcher")

with st.expander("Connection, Auth & Safety", expanded=False):
    st.markdown(
        """
- Paste a **full URL** up to and including `attribute_codes=...` (no `skus`).
- The app will append `&skus=...` as **repeated params** in batches (default 900).
- **Auth**: set `PIM_TOKEN` as an environment variable if required by the API.
- **Network**: run locally on VPN or on an internal host. Host allowlist enforced.
- **Privacy**: URL field is masked; by default we don't cache responses or keep raw JSON.
        """
    )

# Secure URL input
st.subheader("Endpoint (paste up to attribute_codes; no SKUs)")
user_url = st.text_input(
    "Internal PIM URL (kept in memory only)",
    value="",
    placeholder="https://.../v4/description/skus?attribute_codes=returnable",
    type="password",   # hides on screen
)
show_url = st.checkbox("Temporarily reveal URL", value=False)
if show_url and user_url:
    st.code(user_url)

# SKU input
st.subheader("SKUs")
sku_text = st.text_area(
    "Paste SKUs (any delimiter: newline, comma, semicolon, tab)",
    height=160,
    placeholder="064037\n123456\n42\nABC123",
)

# Options
colA, colB, colC = st.columns([1, 1, 1])

with colA:
    pad_to_six = st.toggle("Pad numeric SKUs to 6 chars (000042)", value=True)

with colB:
    chunk_size = st.number_input(
        "Batch size (max SKUs per request)",
        value=DEFAULT_CHUNK_SIZE, min_value=50, max_value=2000, step=50
    )

with colC:
    disable_cache = st.toggle("Disable response caching (recommended)", value=True)

save_raw = st.toggle("Enable raw JSONL download (responses)", value=False)

# Load SKUs
skus = clean_skus(sku_text)
st.caption(f"Total SKUs (before padding): {len(skus)}")

# File upload (optional)
uploaded = st.file_uploader("...or upload a CSV with a 'sku' column", type=["csv"])
if uploaded:
    try:
        df_up = pd.read_csv(uploaded)
        sku_col = None
        for c in df_up.columns:
            if c.lower() in ("sku", "skus"):
                sku_col = c
                break
        if not sku_col:
            sku_col = df_up.columns[0]
        file_skus = [str(x).strip() for x in df_up[sku_col].tolist() if str(x).strip()]
        # merge with pasted list, de-dup, keep order
        seen = set()
        merged = []
        for s in skus + file_skus:
            if s not in seen:
                seen.add(s)
                merged.append(s)
        skus = merged
        st.success(f"Loaded {len(file_skus)} SKUs from file. Total unique SKUs: {len(skus)}")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

# Run button
run_btn = st.button("Run fetch", type="primary", disabled=not (user_url and skus))

if run_btn:
    try:
        base_no_query, params_base = parse_user_url(user_url)

        # padding
        if pad_to_six:
            skus = [pad_sku(s) for s in skus]

        batches = list(chunked(skus, int(chunk_size)))
        client = get_client()

        compiled_frames: List[pd.DataFrame] = []
        raw_lines: List[Dict[str, Any]] = []

        with st.spinner(f"Calling API in {len(batches)} batch(es)..."):
            for i, batch in enumerate(batches, start=1):
                params = build_params_with_skus(params_base, batch)  # -> repeated &skus=
                payload = fetch_batch(client, base_no_query, params)
                if save_raw and not disable_cache:
                    raw_lines.append({
                        "batch_index": i, "batch_size": len(batch),
                        "skus": batch, "response": payload
                    })

                df = normalize_response(payload)
                if not df.empty:
                    cols = list(df.columns)
                    if "sku" in cols:
                        df = df[["sku"] + [c for c in cols if c != "sku"]]
                    compiled_frames.append(df)

        client.close()

        if not compiled_frames:
            st.warning("No rows parsed from responses. Verify URL/attribute_codes and try again.")
        else:
            compiled = pd.concat(compiled_frames, ignore_index=True).drop_duplicates()
            st.success(f"Got {len(compiled)} rows.")
            st.dataframe(compiled, use_container_width=True)

            # Downloads
            st.download_button(
                "Download compiled CSV",
                data=compiled.to_csv(index=False).encode("utf-8"),
                file_name="compiled.csv",
                mime="text/csv",
            )

            # Parquet (optional)
            try:
                buf = io.BytesIO()
                compiled.to_parquet(buf, index=False)
                st.download_button(
                    "Download compiled Parquet",
                    data=buf.getvalue(),
                    file_name="compiled.parquet",
                    mime="application/octet-stream",
                )
            except Exception:
                st.info("Parquet download requires pyarrow. Add it to requirements if needed.")

            # Raw JSONL (optional)
            if save_raw and raw_lines:
                st.download_button(
                    "Download raw JSONL (per-batch)",
                    data="\n".join(json.dumps(x, ensure_ascii=False) for x in raw_lines).encode("utf-8"),
                    file_name="raw.jsonl",
                    mime="application/json",
                )

    except Exception as e:
        st.error(f"{e}")
