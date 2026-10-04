import numpy as np
import pandas as pd
import json
import re
from pathlib import Path
from scipy.interpolate import interp1d

# === Calibrator Detection ===

def is_calibrator(request_no, attribute=None, mode="id_pattern", keywords=None) -> bool:
    """
    Check if a given request number belongs to a calibrator.
    """
    if keywords is None:
        keywords = ["CAL", "cal", "キャリブ"]
        
    is_id_match = False
    if mode in ("id_pattern", "both"):
        is_id_match = bool(re.match(r"^C\d+$", str(request_no), re.IGNORECASE))
        
    is_attr_match = False
    if mode in ("attribute", "both") and attribute is not None:
        is_attr_match = any(kw in str(attribute) for kw in keywords)
        
    if mode == "id_pattern":
        return is_id_match
    elif mode == "attribute":
        return is_attr_match
    elif mode == "both":
        return is_id_match or is_attr_match
    return False

def detect_calibrators(measurement_df, profile_df, item_name, mode="id_pattern", keywords=None) -> list[str]:
    """
    Detect all calibrator IDs for a given item.
    """
    req_nos = set()
    
    # Extract IDs from profile_df
    if profile_df is not None and not profile_df.empty:
        if '項目名' in profile_df.columns and '依頼No.' in profile_df.columns:
            subset = profile_df[profile_df['項目名'] == item_name]
            req_nos.update(subset['依頼No.'].dropna().astype(str).tolist())
            
    # Extract IDs from measurement_df
    attr_col_name = None
    if measurement_df is not None and not measurement_df.empty:
        if '属性' in measurement_df.columns:
            attr_col_name = '属性'
        elif len(measurement_df.columns) >= 5:
            attr_col_name = measurement_df.columns[4]
            
        found_item_col = None
        for col in measurement_df.columns:
            if item_name in col and 'FLAG' not in col:
                found_item_col = col
                break
                
        if found_item_col:
            subset = measurement_df[measurement_df[found_item_col].notna()]
            req_nos.update(subset['依頼No.'].dropna().astype(str).tolist())
            
    # Filter using is_calibrator
    cal_ids = []
    for req in req_nos:
        attr_val = None
        if measurement_df is not None and not measurement_df.empty and attr_col_name is not None:
            row = measurement_df[measurement_df['依頼No.'].astype(str) == str(req)]
            if not row.empty:
                attr_val = row.iloc[0][attr_col_name]
                
        if is_calibrator(req, attribute=attr_val, mode=mode, keywords=keywords):
            cal_ids.append(req)
            
    return sorted(list(cal_ids))

# === Calibration Table Construction ===

def build_cal_level_table(cal_ids, n_levels, n_replicates):
    """
    Build a 2D table of calibrator IDs.
    """
    table = []
    idx = 0
    warning = None
    
    if len(cal_ids) != n_levels * n_replicates:
        warning = f"Expected {n_levels * n_replicates} IDs, but got {len(cal_ids)}."
        
    for i in range(n_levels):
        row = []
        for j in range(n_replicates):
            if idx < len(cal_ids):
                row.append(str(cal_ids[idx]))
            else:
                row.append("")
            idx += 1
        table.append(row)
        
    return table, warning

# === Batch Auto-Assignment (一括自動割り当て) ===
# UI非依存の純粋関数群。Phase 2 で FastAPI (POST /api/calibrations/auto-assign) へ移植する想定。

_CAL_ID_RE = re.compile(r"^C(\d+)$", re.IGNORECASE)


def _natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def parse_measure_time(value):
    """'2026/07/28 9:51:47 9:59:49' のような測定日文字列から、日付 + 末尾時刻を datetime として返す。"""
    parts = str(value).split()
    if len(parts) >= 2:
        return pd.to_datetime(f"{parts[0]} {parts[-1]}", errors="coerce")
    return pd.to_datetime(value, errors="coerce")


def split_id_blocks(cal_ids):
    """C+数字 のIDを番号順に並べ、番号が連続する区間ごとにブロック分割する。"""
    nums = []
    for cid in cal_ids:
        m = _CAL_ID_RE.match(str(cid))
        if m:
            nums.append((int(m.group(1)), str(cid)))
    nums.sort()
    blocks, cur, prev = [], [], None
    for n, cid in nums:
        if prev is not None and n != prev + 1:
            blocks.append(cur)
            cur = []
        cur.append(cid)
        prev = n
    if cur:
        blocks.append(cur)
    return blocks


def assign_block_to_levels(block, values, n_levels):
    """
    ブロックをレベルへ割り当てる。
    - ブロック長が n_levels で割り切れる: ID順に n 個ずつ (装置の登録順 = レベル昇順 × n回)
    - 割り切れない: 装置測定値の大きなギャップ上位 (n_levels-1) 箇所で分割 (フォールバック)
    Returns: (level_table, method, warnings)
    """
    n_levels = max(1, int(n_levels))
    if len(block) % n_levels == 0 and len(block) >= n_levels:
        n = len(block) // n_levels
        return [block[i * n:(i + 1) * n] for i in range(n_levels)], "ID順", []

    warns = [f"ID数{len(block)}がCal点数{n_levels}で割り切れないため測定値ギャップで分割"]
    pairs = sorted(((values.get(cid, np.nan), cid) for cid in block),
                   key=lambda x: (np.inf if not np.isfinite(x[0]) else x[0]))
    if len(pairs) <= n_levels:
        return [[cid] for _, cid in pairs], "値ギャップ", warns
    v = np.array([p[0] for p in pairs], dtype=float)
    gaps = np.nan_to_num(np.diff(v), nan=0.0)
    cuts = sorted((np.argsort(gaps)[::-1][: n_levels - 1] + 1).tolist())
    ids = [p[1] for p in pairs]
    table, start = [], 0
    for c in cuts + [len(ids)]:
        table.append(ids[start:c])
        start = c
    return table, "値ギャップ", warns


def validate_level_table(level_table, values, cv_limit=15.0):
    """レベル代表値(中央値)の単調増加とレベル内CVをチェック。Returns: (medians, warnings)"""
    warns = []
    meds = []
    for lv in level_table:
        vs = [values.get(cid, np.nan) for cid in lv]
        vs = [x for x in vs if x is not None and np.isfinite(x)]
        meds.append(float(np.median(vs)) if vs else np.nan)
    finite = [m for m in meds if np.isfinite(m)]
    if len(finite) >= 2 and any(np.diff(finite) <= 0):
        warns.append("レベル代表値が単調増加でない")
    for k, lv in enumerate(level_table):
        vs = [values.get(cid, np.nan) for cid in lv]
        vs = [x for x in vs if x is not None and np.isfinite(x)]
        if len(vs) >= 2 and np.mean(vs) > 0:
            cv = float(np.std(vs, ddof=1) / np.mean(vs) * 100.0)
            if cv > cv_limit:
                warns.append(f"Cal{k} CV={cv:.1f}%")
    return meds, warns


def auto_assign_calibrators(profile_df, measurement_df, port_to_reagent, reagent_master,
                            set_gap_minutes=30, default_points=6):
    """
    全項目のキャリブレーターを一括で自動割り当てする。

    1. 項目ごとに C+数字 のIDを抽出し、番号の連続性でブロック分割
    2. 試薬マスターの calibration_points をレベル数として各ブロックをレベル割り当て
    3. 全ブロックを開始時刻でクラスタリングし Calセット (=Calロット) を推定
    4. 妥当性チェック (単調増加 / CV)

    Returns: list[dict]  (1要素 = 1本の検量線)
    """
    if profile_df is None or profile_df.empty:
        return []

    times = {}
    if measurement_df is not None and "測定日" in measurement_df.columns:
        for rid, t in zip(measurement_df["依頼No."].astype(str), measurement_df["測定日"]):
            times[rid] = parse_measure_time(t)

    cal_rows = profile_df[profile_df["依頼No."].astype(str).str.match(_CAL_ID_RE)]
    cal_rows = cal_rows.drop_duplicates(["依頼No.", "項目名"])

    entries = []
    for item, sub in cal_rows.groupby("項目名"):
        reagent = port_to_reagent.get(item, "") or ""
        n_levels = int(reagent_master.get(reagent, {}).get("calibration_points", default_points))
        if "処理値" in sub.columns:
            values = {str(k): float(v) if pd.notna(v) else np.nan
                      for k, v in zip(sub["依頼No."], sub["処理値"])}
        else:
            values = {}
        for block in split_id_blocks(sub["依頼No."].astype(str).tolist()):
            table, method, w1 = assign_block_to_levels(block, values, n_levels)
            meds, w2 = validate_level_table(table, values)
            ts = [times[c] for c in block if c in times and pd.notna(times[c])]
            n_reps = len(block) // len(table) if table and len(block) % len(table) == 0 else None
            entries.append({
                "item": item,
                "reagent": reagent,
                "id_range": f"{block[0]}–{block[-1]}",
                "n_ids": len(block),
                "n_levels": len(table),
                "n_reps": n_reps,
                "method": method,
                "level_table": table,
                "level_medians": [round(m, 2) if np.isfinite(m) else None for m in meds],
                "t_start": min(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else None,
                "warnings": w1 + w2,
                "enabled": True,
            })

    # Calセット推定: 開始時刻順に並べ、set_gap_minutes 以上空いたら別セット
    def _t(e):
        return pd.to_datetime(e["t_start"]) if e["t_start"] else pd.Timestamp.max

    set_idx, prev_t = 0, None
    for e in sorted(entries, key=_t):
        t = _t(e)
        if prev_t is not None and t is not pd.Timestamp.max and prev_t is not pd.Timestamp.max:
            if (t - prev_t).total_seconds() / 60.0 > set_gap_minutes:
                set_idx += 1
        e["cal_set"] = f"CalSet-{set_idx + 1}"
        prev_t = t

    entries.sort(key=lambda e: (_natural_key(e["cal_set"]), _natural_key(e["item"])))
    return entries


def build_cal_config_from_registry(registry, item, cal_set):
    """一括登録レジストリから、従来形式の cal_config (Step 2 以降で使用) を組み立てる。"""
    entry = next((e for e in registry.get("entries", [])
                  if e["item"] == item and e["cal_set"] == cal_set and e.get("enabled", True)), None)
    if entry is None:
        return None
    set_info = registry.get("sets", {}).get(cal_set, {})
    concs = set_info.get("concentrations", {}).get(entry["reagent"] or "-", [])
    n_levels = len(entry["level_table"])
    concs = [float(c) if c is not None else 0.0 for c in concs[:n_levels]]
    concs += [0.0] * (n_levels - len(concs))
    return {
        "lot_name": set_info.get("lot_name", cal_set),
        "cal_set": cal_set,
        "item_name": item,
        "n_levels": n_levels,
        "n_replicates": entry.get("n_reps") or 1,
        "concentrations": concs,
        "level_table": [list(lv) for lv in entry["level_table"]],
        "aggregation": registry.get("aggregation", "median"),
    }


def save_cal_registry(registry, parsed_dir):
    """
    一括登録結果を履歴として保存する。
    - <parsed_dir>/cal_registry/cal_registry_<timestamp>.json (履歴; Phase 2 で calibrations テーブルへ移行)
    - <parsed_dir>/cal_config_<item>_<lot>.json (従来形式; ロット差検討 Step 4 との互換)
    Returns: 保存したレジストリファイルのPath
    """
    parsed_dir = Path(parsed_dir)
    reg_dir = parsed_dir / "cal_registry"
    reg_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    registry = dict(registry)
    registry.setdefault("created_at", ts)
    reg_path = reg_dir / f"cal_registry_{ts}.json"
    save_cal_config(registry, reg_path)

    for e in registry.get("entries", []):
        if not e.get("enabled", True):
            continue
        cfg = build_cal_config_from_registry(registry, e["item"], e["cal_set"])
        if cfg is None:
            continue
        legacy = {
            "lot_name": cfg["lot_name"],
            "item_name": cfg["item_name"],
            "n_levels": cfg["n_levels"],
            "n_replicates": cfg["n_replicates"],
            "concentrations": cfg["concentrations"],
            "levels": [{"level": i, "ids": ids} for i, ids in enumerate(cfg["level_table"])],
            "aggregation": cfg["aggregation"],
            "detection_mode": "auto_batch",
        }
        safe_lot = re.sub(r"[\\/:*?\"<>|\s]", "_", str(cfg["lot_name"]))
        save_cal_config(legacy, parsed_dir / f"cal_config_{e['item']}_{safe_lot}.json")
    return reg_path


def list_cal_registries(parsed_dir):
    reg_dir = Path(parsed_dir) / "cal_registry"
    if not reg_dir.exists():
        return []
    return sorted(reg_dir.glob("cal_registry_*.json"), reverse=True)

# === Rate Calculations ===

def calc_rate(profile_df, request_no, item_name, time_start, time_end) -> float:
    """
    Calculate the rate (mAbs/min) for a single sample.
    """
    subset = profile_df[(profile_df['依頼No.'].astype(str) == str(request_no)) & 
                        (profile_df['項目名'] == item_name)]
    if subset.empty:
        return np.nan
        
    times = subset['時間'].values
    abss = subset['吸光度'].values
    
    if len(times) == 0:
        return np.nan
        
    idx_start = np.argmin(np.abs(times - time_start))
    idx_end = np.argmin(np.abs(times - time_end))
    
    t_start_actual = times[idx_start]
    t_end_actual = times[idx_end]
    a_start = abss[idx_start]
    a_end = abss[idx_end]
    
    if t_end_actual == t_start_actual:
        return np.nan
        
    rate = ((a_end - a_start) * 0.1) / ((t_end_actual - t_start_actual) / 60.0)
    return rate

def calc_rates_batch(profile_df, item_name, time_start, time_end) -> dict:
    """
    全サンプルについて指定項目の処理値(Rate)を一括算出する。
    """
    subset = profile_df[profile_df['項目名'] == item_name]
    req_nos = subset['依頼No.'].dropna().unique()
    
    rates = {}
    for req in req_nos:
        rate = calc_rate(profile_df, req, item_name, time_start, time_end)
        rates[str(req)] = rate
    return rates

def aggregate_cal_rates(rates_dict, level_table, method="median") -> list[float]:
    """
    Compute representative rate per calibration level.
    """
    agg_rates = []
    for row in level_table:
        vals = []
        for req in row:
            if req and req in rates_dict and not np.isnan(rates_dict[req]):
                vals.append(rates_dict[req])
                
        if len(vals) == 0:
            agg_rates.append(np.nan)
        elif len(vals) == 1:
            agg_rates.append(vals[0])
        elif len(vals) == 2:
            agg_rates.append(float(np.mean(vals)))
        else:
            if method == "mean":
                agg_rates.append(float(np.mean(vals)))
            else:
                agg_rates.append(float(np.median(vals)))
    return agg_rates

# === Curve Construction and Prediction ===

def build_calibration_curve(cal_rates, concentrations, curve_mode="piecewise_linear") -> dict:
    """
    Build a calibration curve dictionary.
    """
    valid_rates = []
    valid_concs = []
    for r, c in zip(cal_rates, concentrations):
        if not np.isnan(r) and not np.isnan(c):
            valid_rates.append(r)
            valid_concs.append(c)
            
    if len(valid_rates) < 2:
        return None
        
    rates = np.array(valid_rates)
    concs = np.array(valid_concs)
    
    sort_idx = np.argsort(rates)
    rates = rates[sort_idx]
    concs = concs[sort_idx]
    
    interp_func = None
    if curve_mode == "spline":
        if len(rates) >= 4:
            interp_func = interp1d(rates, concs, kind='cubic', fill_value='extrapolate')
        else:
            curve_mode = "piecewise_linear"
            
    return {
        "rates": rates,
        "concentrations": concs,
        "curve_mode": curve_mode,
        "interp_func": interp_func
    }

def predict_concentration(cal_curve, rate_value) -> float:
    """
    Predict concentration from a rate value using the calibration curve.
    """
    if cal_curve is None or np.isnan(rate_value):
        return np.nan
        
    if cal_curve["curve_mode"] == "spline" and cal_curve["interp_func"] is not None:
        return float(cal_curve["interp_func"](rate_value))
    else: # piecewise_linear
        rates = cal_curve["rates"]
        concs = cal_curve["concentrations"]
        
        if rate_value < rates[0]:
            slope = (concs[1] - concs[0]) / (rates[1] - rates[0]) if rates[1] != rates[0] else 0
            return float(concs[0] + slope * (rate_value - rates[0]))
        elif rate_value > rates[-1]:
            slope = (concs[-1] - concs[-2]) / (rates[-1] - rates[-2]) if rates[-1] != rates[-2] else 0
            return float(concs[-1] + slope * (rate_value - rates[-1]))
        else:
            return float(np.interp(rate_value, rates, concs))

def recalculate_all_samples(profile_df, measurement_df, item_name, cal_curve, time_start, time_end, cal_id_to_conc=None) -> pd.DataFrame:
    """
    Recalculate concentrations for all samples.
    """
    rates_dict = calc_rates_batch(profile_df, item_name, time_start, time_end)
    
    attr_col_name = None
    if measurement_df is not None and not measurement_df.empty:
        if '属性' in measurement_df.columns:
            attr_col_name = '属性'
        elif len(measurement_df.columns) >= 5:
            attr_col_name = measurement_df.columns[4]
            
    meas_col = None
    if measurement_df is not None and not measurement_df.empty:
        for col in measurement_df.columns:
            if item_name in col and 'FLAG' not in col:
                meas_col = col
                break
                
    results = []
    
    for req_no, rate_val in rates_dict.items():
        attr_val = None
        orig_conc = np.nan
        
        if measurement_df is not None and not measurement_df.empty:
            row = measurement_df[measurement_df['依頼No.'].astype(str) == req_no]
            if not row.empty:
                if attr_col_name:
                    attr_val = row.iloc[0][attr_col_name]
                if meas_col:
                    orig_conc = row.iloc[0][meas_col]
                    
        is_cal = is_calibrator(req_no, attribute=attr_val, mode="both")
        sample_type = "キャリブレーター" if is_cal else "一般検体"
        
        if is_cal and cal_id_to_conc and req_no in cal_id_to_conc:
            orig_conc = cal_id_to_conc[req_no]
            
        recalc_conc = predict_concentration(cal_curve, rate_val)
        
        results.append({
            "依頼No.": req_no,
            "サンプル区分": sample_type,
            "処理値": rate_val,
            "装置測定値": orig_conc,
            "再計算濃度": recalc_conc
        })
        
    return pd.DataFrame(results)

# === Configuration & UI Helpers ===

def save_cal_config(config, path):
    """Save calibration config to JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def load_cal_config(path) -> dict:
    """Load calibration config from JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_cal_level_detail_table(rates_dict, level_table, concentrations, item_name) -> pd.DataFrame:
    """
    Build a detail DataFrame showing each individual calibrator measurement.
    """
    records = []
    for level_idx, row in enumerate(level_table):
        conc = concentrations[level_idx] if level_idx < len(concentrations) else np.nan
        for req in row:
            if req:
                rate = rates_dict.get(req, np.nan)
                records.append({
                    "レベル": f"Cal {level_idx}",
                    "依頼No.": req,
                    "表示値濃度": conc,
                    "処理値(Rate)": rate
                })
    return pd.DataFrame(records)


def get_cal_level_summary_table(rates_dict, level_table, concentrations, agg_method="median") -> pd.DataFrame:
    """
    各レベルの代表値、n数、Mean、SD、CV%(n>=2)をまとめたサマリーテーブルを作成。
    キャリブレーションの安定性・バラつき（CV%）確認に使用。
    """
    rows = []
    for level_idx, row in enumerate(level_table):
        conc = concentrations[level_idx] if level_idx < len(concentrations) else np.nan
        vals = [rates_dict[req] for req in row if req and req in rates_dict and np.isfinite(rates_dict[req])]
        n = len(vals)
        mean_val = float(np.mean(vals)) if n > 0 else np.nan
        sd_val = float(np.std(vals, ddof=1)) if n >= 2 else np.nan
        cv_val = float((sd_val / mean_val) * 100.0) if (np.isfinite(sd_val) and mean_val != 0) else np.nan
        rep_val = float(np.median(vals)) if agg_method == "median" and n > 0 else mean_val

        rows.append({
            "レベル": f"Cal {level_idx}",
            "表示値濃度": conc,
            "測定点数(n)": n,
            "代表値(Rate)": rep_val,
            "平均値(Mean)": mean_val,
            "標準偏差(SD)": sd_val,
            "CV(%)": cv_val
        })
    return pd.DataFrame(rows)


def compare_two_recalc_results(df_a, df_b, label_a="Lot-A", label_b="Lot-B") -> pd.DataFrame:
    """
    2つの再計算結果（ロット間または条件間）をマージして差や比率を算出。
    実検体・コントロール検体におけるロット間測定値差の検討に使用。
    """
    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        return pd.DataFrame()

    sub_a = df_a[["依頼No.", "サンプル区分", "再計算濃度"]].rename(columns={"再計算濃度": f"濃度_{label_a}"})
    sub_b = df_b[["依頼No.", "再計算濃度"]].rename(columns={"再計算濃度": f"濃度_{label_b}"})

    merged = sub_a.merge(sub_b, on="依頼No.", how="inner")
    ca = pd.to_numeric(merged[f"濃度_{label_a}"], errors="coerce")
    cb = pd.to_numeric(merged[f"濃度_{label_b}"], errors="coerce")

    merged[f"差({label_b}-{label_a})"] = cb - ca
    merged[f"相対比({label_b}/{label_a})"] = np.where(ca != 0, cb / ca, np.nan)
    merged[f"乖離率(%)"] = np.where(ca != 0, (cb - ca) / np.abs(ca) * 100.0, np.nan)

    return merged
