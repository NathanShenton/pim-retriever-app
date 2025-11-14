#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit app: Paste an internal PIM URL up to & including attribute_codes=...,
then the app appends &skus=... (repeated params) in safe batches and compiles results.

Visibility:
- Progress bar + running ETA
- Status feed (success/retry per batch)
- Live per-batch log table (rows, retries, time)
- Preflight panel to preview request counts and max query length

Security & Ops:
- URL input is masked (password field)
- Host allow-listed (default: *.hbi.systems)
- Bearer token read from env var PIM_TOKEN (optional)
- Response caching disabled by default (toggleable)
- Raw JSON download is opt-in

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
from time import perf_counter
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlparse, parse_qsl, urlunparse, urlencode

import httpx
import pandas as pd
import streamlit as st

from typing import Any, Dict, Iterable, List, Tuple, Callable, Optional


# =========================
# Config / Security
# =========================

ALLOWED_SUFFIXES = (".hbi.systems",)  # tighten/extend as needed
DEFAULT_CHUNK_SIZE = 900              # keeps URLs under typical 414 limits

# =========================
# Helpers
# =========================

def auth_headers() -> Dict[str, str]:
    token = os.getenv("PIM_TOKEN")
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def get_client(timeout_s: int = 60) -> httpx.Client:
    """
    Create an httpx client; only pass 'proxies' if they’re explicitly set to
    avoid version mismatches. TLS verification remains ON by default.
    """
    proxies_env = {
        "http://": os.getenv("HTTP_PROXY"),
        "https://": os.getenv("HTTPS_PROXY"),
    }
    proxies_env = {k: v for k, v in proxies_env.items() if v}

    kwargs = dict(timeout=timeout_s, headers=auth_headers())
    if proxies_env:
        kwargs["proxies"] = proxies_env  # httpx will honor NO_PROXY if set

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
    """Left-pad numeric SKUs to fixed width with zeros; leave alphanumerics unchanged."""
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
    Clone base params and append SKUs as repeated keys (&skus=A&skus=B...).
    httpx will encode list values as repeated query params when doseq=True internally.
    """
    params = {k: list(v) for k, v in params_base.items()}
    params["skus"] = skus
    return params

def backoff_sleep(attempt: int, base: float = 0.5, cap: float = 20.0):
    """Exponential backoff with a small additive nudge."""
    delay = min(cap, base * (2 ** attempt)) + (0.1 * (attempt + 1))
    time.sleep(delay)

def fetch_batch(
    client: httpx.Client,
    base_url: str,
    params: Dict[str, List[str]],
    max_retries: int = 6,
    on_retry: Optional[Callable[[int, Any], None]] = None,  # (attempt_index, http_status_or_exc) -> None
) -> Tuple[Dict[str, Any], int, float]:
    attempt = 0
    start = perf_counter()
    while True:
        try:
            resp = client.get(base_url, params=params)
            if resp.status_code in (429, 500, 502, 503, 504):
                if attempt >= max_retries:
                    resp.raise_for_status()
                if on_retry:
                    on_retry(attempt + 1, resp.status_code)
                backoff_sleep(attempt)
                attempt += 1
                continue
            resp.raise_for_status()
            return resp.json(), attempt, perf_counter() - start
        except httpx.RequestError as e:
            if attempt >= max_retries:
                raise
            if on_retry:
                on_retry(attempt + 1, type(e).__name__)
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
    type="password",
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

# Optional CSV upload
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

# -------------------------
# Preflight (optional)
# -------------------------
if user_url and skus:
    try:
        base_no_query_pf, params_base_pf = parse_user_url(user_url)
        # simulate a single batch to estimate max query length
        probe_batch = [pad_sku(s) for s in skus[: min(len(skus), int(chunk_size))]] if pad_to_six else skus[: min(len(skus), int(chunk_size))]
        qs = urlencode(build_params_with_skus(params_base_pf, probe_batch), doseq=True)
        with st.expander("Preflight", expanded=False):
            st.write(f"Planned requests: **{(len(skus) - 1) // int(chunk_size) + 1}**")
            st.write(f"Largest single-batch querystring length (estimate): **~{len(qs)}** characters")
    except Exception as e:
        st.info(f"Preflight skipped: {e}")

# Run button
run_btn = st.button("Run fetch", type="primary", disabled=not (user_url and skus))

if run_btn:
    try:
        base_no_query, params_base = parse_user_url(user_url)

        # padding
        if pad_to_six:
            skus = [pad_sku(s) for s in skus]

        batches = list(chunked(skus, int(chunk_size)))
        total_batches = len(batches)
        client = get_client()

        compiled_frames: List[pd.DataFrame] = []
        raw_lines: List[Dict[str, Any]] = []

        # --- Live visibility UI ---
        prog = st.progress(0.0)
        stat = st.status("Starting…", expanded=True)
        log_ph = st.empty()
        kpi1, kpi2, kpi3 = st.columns(3)

        log_rows: List[Dict[str, Any]] = []
        total_elapsed = 0.0
        total_retries = 0

        with stat:
            st.write(f"Prepared **{total_batches}** batch(es).")

        for i, batch in enumerate(batches, start=1):
            def _on_retry(attempt_i: int, code_or_exc: Any):
                with stat:
                    st.write(f"🔁 Retry {attempt_i} on batch {i}/{total_batches} (reason: {code_or_exc})")

            params = build_params_with_skus(params_base, batch)

            payload, attempts, duration = fetch_batch(
                client, base_no_query, params, on_retry=_on_retry
            )
            total_elapsed += duration
            total_retries += attempts

            df = normalize_response(payload)
            rows = 0
            if not df.empty:
                cols = list(df.columns)
                if "sku" in cols:
                    df = df[["sku"] + [c for c in cols if c != "sku"]]
                compiled_frames.append(df)
                rows = len(df)

            if save_raw and not disable_cache:
                raw_lines.append({
                    "batch_index": i, "batch_size": len(batch),
                    "skus": batch, "response": payload
                })

            # Update progress + ETA + KPIs
            prog.progress(i / total_batches)
            avg = (total_elapsed / i) if i else 0.0
            remaining = total_batches - i
            eta = remaining * avg

            with stat:
                st.write(f"✅ Batch {i}/{total_batches}: {len(batch)} SKUs | {rows} rows | {duration:.2f}s | retries: {attempts}")

            # live log table
            log_rows.append({
                "batch": i,
                "batch_size": len(batch),
                "rows": rows,
                "retries": attempts,
                "duration_s": round(duration, 3),
                "avg_s": round(avg, 3),
                "eta_s": round(eta, 1),
            })
            log_df = pd.DataFrame(log_rows)
            log_ph.dataframe(log_df, use_container_width=True, height=260)

            kpi1.metric("Batches done", f"{i}/{total_batches}")
            kpi2.metric("Total retries", f"{total_retries}")
            kpi3.metric("ETA (s)", f"{int(eta)}")

            st.toast(f"Batch {i}/{total_batches} done in {duration:.2f}s (retries: {attempts})")

        client.close()

        # Compile + downloads
        if not compiled_frames:
            stat.update(label="Finished (no rows parsed).", state="warning", expanded=False)
            st.warning("No rows parsed from responses. Verify URL/attribute_codes and try again.")
        else:
            compiled = pd.concat(compiled_frames, ignore_index=True).drop_duplicates()
            stat.update(label="Finished successfully.", state="complete", expanded=False)
            st.success(f"Got {len(compiled)} rows total.")
            st.dataframe(compiled, use_container_width=True)

            st.download_button(
                "Download compiled CSV",
                data=compiled.to_csv(index=False).encode("utf-8"),
                file_name="compiled.csv",
                mime="text/csv",
            )

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

            if save_raw and raw_lines:
                st.download_button(
                    "Download raw JSONL (per-batch)",
                    data="\n".join(json.dumps(x, ensure_ascii=False) for x in raw_lines).encode("utf-8"),
                    file_name="raw.jsonl",
                    mime="application/json",
                )

    except Exception as e:
        st.error(f"{e}")
