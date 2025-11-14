#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import json
import os
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlparse, parse_qsl, urlunparse

import httpx
import pandas as pd
import streamlit as st

# ------------------------------
# Basics
# ------------------------------
DEFAULT_CHUNK_SIZE = 900

def chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i:i+size]

def clean_skus(raw: str) -> List[str]:
    if not raw:
        return []
    raw = raw.replace(";", "\n").replace(",", "\n").replace("\t", "\n")
    skus = [s.strip() for s in raw.splitlines() if s.strip()]
    # de-dup keep order
    seen, out = set(), []
    for s in skus:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def pad_sku(s: str, width: int = 6) -> str:
    return s.zfill(width) if s.isdigit() and len(s) < width else s

def parse_user_url(user_url: str) -> Tuple[str, Dict[str, List[str]]]:
    if not user_url:
        raise ValueError("URL is required.")
    u = urlparse(user_url)
    q_items = parse_qsl(u.query, keep_blank_values=True)
    params: Dict[str, List[str]] = {}
    for k, v in q_items:
        params.setdefault(k, []).append(v)
    if "attribute_codes" not in params:
        raise ValueError("URL must include attribute_codes=...")
    # we will append 'skus' ourselves
    params.pop("skus", None)
    base_no_query = urlunparse((u.scheme, u.netloc, u.path, "", "", ""))
    return base_no_query, params

def build_params_with_skus(params_base: Dict[str, List[str]], batch_skus: List[str]) -> Dict[str, List[str]]:
    params = {k: list(v) for k, v in params_base.items()}
    params["skus"] = batch_skus  # httpx encodes list as repeated &skus=...
    return params

def get_client() -> httpx.Client:
    """
    Smallest possible client:
    - trust_env=True so it behaves like other CLI tools (honours HTTP(S)_PROXY/NO_PROXY)
    - reasonable timeouts to avoid hanging forever
    - optional bearer via PIM_TOKEN
    """
    headers = {"Accept": "application/json"}
    token = os.getenv("PIM_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    timeouts = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=60.0)
    return httpx.Client(timeout=timeouts, headers=headers, trust_env=True)

# ------------------------------
# Normaliser for YOUR JSON shape
# (as shown in your browser output)
# ------------------------------
def flatten_variants_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Expected shape (your example):
    {
      "data": {
        "variants": [
          {
            "skus": [
              {"id":"069220","barcodes":["5060..."],"attributes": null}
            ],
            "variant_id": 88132,
            "variant_name": {"en_UK":"...", "fr_FR":"...", "nl_NL":"..."},
            "attributes": [],
            "complexes": []
          }
        ],
        "meta": {}
      },
      "success": true
    }

    We’ll output one row per SKU with columns:
      sku, barcode(s), variant_id, variant_name_en_UK, variant_name_fr_FR, variant_name_nl_NL
    """
    d = payload.get("data") or {}
    variants = d.get("variants") or []
    rows = []
    for v in variants:
        variant_id = v.get("variant_id")
        vname = v.get("variant_name") or {}
        name_en = vname.get("en_UK")
        name_fr = vname.get("fr_FR")
        name_nl = vname.get("nl_NL")
        for sku_obj in v.get("skus") or []:
            sku = sku_obj.get("id")
            barcodes = sku_obj.get("barcodes") or []
            rows.append({
                "sku": sku,
                "barcodes": ", ".join(map(str, barcodes)) if isinstance(barcodes, list) else str(barcodes),
                "variant_id": variant_id,
                "variant_name_en_UK": name_en,
                "variant_name_fr_FR": name_fr,
                "variant_name_nl_NL": name_nl,
            })
    return pd.DataFrame(rows)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="PIM Attribute Fetcher – Minimal", layout="wide")
st.title("PIM Attribute Fetcher – Minimal")

st.markdown(
    "Paste your **full URL up to `attribute_codes=...`** (no `skus`). "
    "The app will append `&skus=` per item, in batches."
)

user_url = st.text_input(
    "Endpoint (paste up to attribute_codes; no SKUs)",
    value="",
    placeholder="https://.../v4/description/skus?attribute_codes=product_age_retail",
)

col1, col2 = st.columns([2,1])
with col1:
    sku_text = st.text_area("SKUs (any delimiter: newline/comma/semicolon/tab)", height=160)
with col2:
    pad_flag = st.toggle("Pad numeric SKUs to 6 chars (000042)", value=True)
    chunk_size = st.number_input("Batch size", value=DEFAULT_CHUNK_SIZE, min_value=50, max_value=2000, step=50)

skus = clean_skus(sku_text)
if pad_flag:
    skus = [pad_sku(s) for s in skus]

st.caption(f"Total SKUs: {len(skus)}")
go = st.button("Fetch", type="primary", disabled=not (user_url and skus))

if go:
    try:
        base_no_query, params_base = parse_user_url(user_url)

        batches = list(chunked(skus, int(chunk_size)))
        client = get_client()

        prog = st.progress(0.0)
        stat = st.status("Starting…", expanded=True)
        frames: List[pd.DataFrame] = []
        log_rows = []

        with stat:
            st.write(f"Prepared **{len(batches)}** batch(es).")

        for i, batch in enumerate(batches, start=1):
            params = build_params_with_skus(params_base, batch)
            with stat:
                st.write(f"🚀 Batch {i}/{len(batches)} — {len(batch)} SKUs")

            r = client.get(base_no_query, params=params)
            r.raise_for_status()
            payload = r.json()

            # Try “variants” flattener first; fallback to generic
            df = flatten_variants_payload(payload)
            if df.empty:
                # generic flatten; you can uncomment to inspect
                # st.write(payload)
                df = pd.json_normalize(payload)

            if not df.empty:
                frames.append(df)

            prog.progress(i / len(batches))

        client.close()

        if not frames:
            st.warning("No rows parsed from responses. (Tip: paste one of the full URLs including a single `&skus=` into your browser and share the JSON if shape differs.)")
        else:
            compiled = pd.concat(frames, ignore_index=True).drop_duplicates()
            st.success(f"Got {len(compiled)} rows total.")
            st.dataframe(compiled, use_container_width=True)

            st.download_button(
                "Download CSV",
                data=compiled.to_csv(index=False).encode("utf-8"),
                file_name="compiled.csv",
                mime="text/csv",
            )
            try:
                buf = io.BytesIO()
                compiled.to_parquet(buf, index=False)
                st.download_button(
                    "Download Parquet",
                    data=buf.getvalue(),
                    file_name="compiled.parquet",
                    mime="application/octet-stream",
                )
            except Exception:
                st.info("Parquet needs pyarrow; install if you want that button to work.")

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
