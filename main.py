import sys
from pathlib import Path
import json
import traceback
import itertools
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.csv_loader import load_parsed_for_analysis
from common.analysis_utils import (
    setup_japanese_font, pick_col, normalize_group_col, detect_value_cols,
    safe_name, make_output_dirs, apply_value_range_filter,
    pearson_r, regression_fit_info, plot_suite, write_df_to_sheet,
    compute_pair_sample_metrics, classify_outlier_level, insert_images_into_excel,
    REGRESSION_METHODS_ALL, SHEET_PLOTS, SHEET_SUMMARY,
    SHEET_OUTLIERS, SHEET_SAMPLE_METRICS,
    OUT_SUFFIX, ID_COL_CANDIDATES, GROUP_COL_CANDIDATES, VALUE_PREFIXES
)
from common.calibration_manager import (
    is_calibrator as cm_is_calibrator,
    detect_calibrators, build_cal_level_table,
    calc_rate, calc_rates_batch, aggregate_cal_rates,
    build_calibration_curve, predict_concentration,
    recalculate_all_samples, save_cal_config, load_cal_config,
    get_cal_level_detail_table, get_cal_level_summary_table,
    compare_two_recalc_results,
    auto_assign_calibrators, build_cal_config_from_registry,
    save_cal_registry, list_cal_registries,
)

setup_japanese_font()

OUTPUT_ROOT = PROJECT_ROOT / "data" / "export"

st.set_page_config(page_title="相関解析 & タイムコース表示", layout="wide")

# ============================================================
# Session State Initialization
# ============================================================
if "df" not in st.session_state:
    st.session_state.update({
        "df": None,
        "id_col": None,
        "group_col": None,
        "value_cols": None,
        "parsed_dir": None,
        "metadata": None,
        "profile_df": None,
        "ref_outlier_map": None,
        "analysis_results": None,
        "metadata_enhanced": None
    })

# ============================================================
# Calibrator Detection
# ============================================================
def is_calibrator(request_no: str) -> bool:
    """依頼No.が 'C + 数字' パターンならキャリブレーターと判定する。
    例: 'C001', 'C1', 'c002' -> True
        '0001', '001'       -> False
    """
    return bool(re.match(r'^C\d+$', str(request_no).strip(), re.IGNORECASE))


def find_calibrator_groups(measurement_df, item_col):
    """指定項目に対して、連続するキャリブレーターIDグループを検出する。"""
    if measurement_df is None or item_col not in measurement_df.columns:
        return []
    
    groups = []
    current_group = []
    current_values = []
    
    id_col = "依頼No." if "依頼No." in measurement_df.columns else measurement_df.columns[0]
    
    for _, row in measurement_df.iterrows():
        rid = str(row[id_col])
        is_cal = is_calibrator(rid)
        val = row.get(item_col)
        has_val = pd.notna(val)
        
        if is_cal and has_val:
            current_group.append(rid)
            current_values.append(float(val))
        else:
            if current_group:
                groups.append({'ids': current_group, 'values': current_values})
                current_group, current_values = [], []
    if current_group:
        groups.append({'ids': current_group, 'values': current_values})
    return groups


# ============================================================
# Helper Functions
# ============================================================
def discover_latest_parsed_dir(parsed_root=None):
    root = PROJECT_ROOT
    parsed_root = Path(parsed_root) if parsed_root is not None else root / "data" / "parsed-data"
    if not parsed_root.is_absolute():
        parsed_root = root / parsed_root

    if parsed_root.is_dir() and (parsed_root / "measurement.parquet").exists() and (parsed_root / "metadata.json").exists():
        return [parsed_root]

    if not parsed_root.exists():
        return None

    candidates = [
        p for p in parsed_root.iterdir()
        if p.is_dir() and (p / "measurement.parquet").exists() and (p / "metadata.json").exists()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates

def load_parsed_data_for_app(parsed_dir):
    parsed_dir = Path(parsed_dir)
    measurement_df, profile_df, metadata, prescription_columns = load_parsed_for_analysis(parsed_dir)

    measurement_df = measurement_df.copy()

    # SID列が存在しても中身が全てNone/nanの場合は他の候補や依頼No.で上書きする
    def _sid_col_is_empty(df):
        if "SID" not in df.columns:
            return True
        s = df["SID"].astype(str).str.strip()
        return s.isin(["None", "nan", "NaN", "", "<NA>"]).all()

    if _sid_col_is_empty(measurement_df):
        if "依頼No." in measurement_df.columns:
            measurement_df["SID"] = measurement_df["依頼No."].astype(str)
        else:
            measurement_df["SID"] = measurement_df.index.astype(str)

    id_col_candidates_ext = ["SID"] + list(ID_COL_CANDIDATES) + ["依頼No."]
    id_col = pick_col(measurement_df, id_col_candidates_ext, default="SID")
    group_col_raw = pick_col(measurement_df, GROUP_COL_CANDIDATES, default=None)
    group_col = normalize_group_col(measurement_df, group_col_raw)

    value_cols = [c for c in prescription_columns if c in measurement_df.columns]
    if not value_cols:
        value_cols = detect_value_cols(measurement_df, id_col, group_col, prefixes=VALUE_PREFIXES)
    value_cols = [c for c in value_cols if c in measurement_df.columns and not str(c).endswith("_FLAG")]

    return measurement_df, profile_df, metadata, parsed_dir, id_col, group_col, value_cols

def load_action(target_dir):
    try:
        df, profile_df, metadata, parsed_dir, id_col, group_col, value_cols = load_parsed_data_for_app(target_dir)
        st.session_state.update({
            "df": df,
            "profile_df": profile_df,
            "metadata": metadata,
            "parsed_dir": parsed_dir,
            "id_col": id_col,
            "group_col": group_col,
            "value_cols": value_cols,
            "analysis_results": None,
            "metadata_enhanced": None,
            "ref_outlier_map": None
        })
        st.success(f"読み込み完了: {parsed_dir.name}\nID列: {id_col}, 比較列数: {len(value_cols)}")
    except Exception as e:
        st.error(f"データ読み込みエラー:\n{traceback.format_exc()}")

# ============================================================
# Main UI
# ============================================================
st.title("相関解析 & タイムコース表示")
st.markdown("""
<div style='border:1px solid #ccc; padding:12px; border-radius:8px; background:#fafafa; line-height:1.6;'>
<b>このツールの目的</b><br>
解析済みデータ（parsed-data）から相関・回帰・Bland–Altman・残差・乖離候補を確認し、Excelに出力します。<br>
また、検体ごとのタイムコース反応（吸光度変化）を確認できます。
</div>
""", unsafe_allow_html=True)

# 1. Directory Selection & Loading
st.header("① データ読み込み")
parsed_dirs = discover_latest_parsed_dir()
if parsed_dirs:
    dir_options = {p.name: str(p) for p in parsed_dirs}
    selected_dir_name = st.selectbox("解析対象", options=list(dir_options.keys()))
    selected_dir_path = dir_options[selected_dir_name]
    if st.button("①データ読み込み", type="primary"):
        with st.spinner("読み込み中..."):
            load_action(selected_dir_path)
else:
    st.warning("対象データが見つかりません。")

st.divider()

if st.session_state["df"] is not None:
    value_cols = st.session_state["value_cols"]

    tab_reagent, tab_calib, tab_measurement, tab_timecourse, tab_correl_old, tab_excel_old = st.tabs([
        "試薬登録", "キャリブレーション", "検体測定", "タイムコース", "相関解析(旧)", "Excel出力(旧)"
    ])

    # ----------------------------------------------------
    # TAB 0: 試薬登録
    # ----------------------------------------------------
    with tab_reagent:
        st.header("試薬登録")
        st.write("データ内の各測定ポート（項目名）に対して、試薬マスターから適切なパラメーターを割り当てます。")
        
        # 試薬マスターの読み込み
        reagent_master = {}
        try:
            with open(PROJECT_ROOT / "data" / "reagent_master.json", "r", encoding="utf-8") as f:
                reagent_master = json.load(f)
        except Exception as e:
            st.warning("試薬マスター(reagent_master.json)が見つかりません。")

        if st.session_state.get("profile_df") is not None and reagent_master:
            profile_df = st.session_state["profile_df"]
            unique_items = sorted(list(profile_df["項目名"].dropna().unique()))
            
            # デフォルトの割り当てを推論 (項目名がTAT1ならTATを割り当て)
            port_mapping = []
            for item in unique_items:
                assigned_reagent = ""
                for master_key in reagent_master.keys():
                    if item.startswith(master_key):
                        assigned_reagent = master_key
                        break
                port_mapping.append({"測定ポート (項目名)": item, "割り当て試薬": assigned_reagent})
            
            df_mapping = pd.DataFrame(port_mapping)
            
            st.markdown("#### ポート割り当て設定")
            edited_mapping = st.data_editor(
                df_mapping,
                column_config={
                    "測定ポート (項目名)": st.column_config.TextColumn(disabled=True),
                    "割り当て試薬": st.column_config.SelectboxColumn(
                        options=[""] + list(reagent_master.keys()),
                        required=True
                    )
                },
                use_container_width=True,
                key="reagent_mapping_editor"
            )
            st.session_state["port_mapping"] = edited_mapping
            
            st.markdown("#### 登録されている試薬パラメーター一覧")
            st.json(reagent_master)
        else:
            st.info("データ（CSV/Parquet）を読み込むと、ここにポート割り当て設定が表示されます。")

    # ----------------------------------------------------
    # TAB 1: 相関解析(旧)
    # ----------------------------------------------------
    with tab_correl_old:
        st.header("② 解析設定 & 実行")
        col1, col2, col3 = st.columns(3)
        with col1:
            mode = st.selectbox("モード", options=["all", "adjacent", "baseline"],
                                format_func=lambda x: {"all":"全組合せ", "adjacent":"隣同士のみ", "baseline":"基準処方 vs その他"}[x], index=2)
            if mode == "baseline":
                baseline_sel = st.multiselect("基準処方", options=value_cols, default=[value_cols[0]] if value_cols else [])
            else:
                baseline_sel = []
        with col2:
            reg_method = st.selectbox("回帰法", options=["OLS", "Deming", "TheilSen", "PassingBablok"],
                                      format_func=lambda x: {"OLS":"OLS（最小二乗）", "Deming":"Deming（両軸誤差）", "TheilSen":"Theil-Sen（ロバスト）", "PassingBablok":"Passing-Bablok（ノンパラメトリック）"}[x], index=3)
            all_reg_ck = st.checkbox("全回帰法で出力", value=False)
            deming_lambda_val = st.number_input("λ(Deming)", value=1.0)
        with col3:
            outlier_mode_dd = st.selectbox("乖離判定基準", options=["zMAD", "error"], format_func=lambda x: "zMAD (標準化残差)" if x == "zMAD" else "error (臨床的許容誤差)")
            if outlier_mode_dd == "zMAD":
                z_thresh_val = st.slider("乖離z(MAD)", min_value=1.5, max_value=8.0, value=3.5, step=0.1)
                pct_thresh_val, abs_thresh_val = 10.0, 2.0
            else:
                pct_thresh_val = st.number_input("許容誤差(%)", value=10.0)
                abs_thresh_val = st.number_input("許容誤差(絶対値)", value=2.0)
                z_thresh_val = 3.5
            label_top_val = st.slider("ラベル数", min_value=0, max_value=30, value=8, step=1)

        st.markdown("**範囲絞り設定**")
        col4, col5, col6 = st.columns(3)
        with col4:
            use_range_ck = st.checkbox("対象範囲絞り", value=False)
        with col5:
            range_min_txt = st.number_input("下限", value=0.0)
        with col6:
            range_max_txt = st.number_input("上限", value=100.0)

        st.markdown("**乖離判定の基準（このペアで赤い検体を他でも赤く表示）**")
        col7, col8 = st.columns(2)
        with col7:
            ref_pair_x = st.selectbox("乖離基準X", options=["(未選択)"] + value_cols, index=1 if len(value_cols) >= 2 else 0)
        with col8:
            ref_pair_y = st.selectbox("乖離基準Y", options=["(未選択)"] + value_cols, index=2 if len(value_cols) >= 2 else 0)

        col9, col10 = st.columns(2)
        with col9:
            show_py_ck = st.checkbox("画面上に表示", value=True)
        with col10:
            max_show_val = st.slider("表示上限", min_value=0, max_value=30, value=30, step=1)

        if st.button("②解析実行", type="primary", key="run_analysis"):
            with st.spinner("解析実行中..."):
                df = st.session_state["df"]
                id_col = st.session_state["id_col"]
                group_col = st.session_state["group_col"]
                parsed_dir = st.session_state["parsed_dir"]

                if mode == "adjacent":
                    pairs = list(zip(value_cols[:-1], value_cols[1:]))
                elif mode == "all":
                    pairs = list(itertools.combinations(value_cols, 2))
                else:
                    bases = list(baseline_sel) if baseline_sel else [value_cols[0]]
                    pairs = []
                    for b in bases:
                        for c in value_cols:
                            if c != b and (b, c) not in pairs and (c, b) not in pairs:
                                pairs.append((b, c))

                methods = REGRESSION_METHODS_ALL if all_reg_ck else [reg_method]
                lam = deming_lambda_val
                z_thresh = z_thresh_val

                summary_rows, outlier_tables, sample_metric_tables, figures = [], [], [], []
                shown = 0
                run_metadata = st.session_state["metadata"].copy() if st.session_state["metadata"] else {}

                # --- Pre-calculate Reference Outliers for Coloring ---
                ref_outlier_map = {}
                ref_x, ref_y = ref_pair_x, ref_pair_y
                if ref_x != "(未選択)" and ref_y != "(未選択)" and ref_x != ref_y:
                    df_ref = apply_value_range_filter(df, ref_x, ref_y, use_range=use_range_ck, lo=range_min_txt, hi=range_max_txt)
                    if id_col in df_ref.columns:
                        df_ref[id_col] = df_ref[id_col].astype(str).replace(["nan", "None", "<NA>", "NaN"], "Unknown")
                        df_ref[id_col] = df_ref[id_col].fillna("Unknown")
                    sub_ref = df_ref[[ref_x, ref_y]].dropna()
                    if len(sub_ref) >= 2:
                        xr, yr = sub_ref[ref_x].astype(float).values, sub_ref[ref_y].astype(float).values
                        ar, br, _ = regression_fit_info(xr, yr, method=reg_method, deming_lambda=lam)
                        try:
                            metrics_ref = compute_pair_sample_metrics(df_ref, id_col, group_col, ref_x, ref_y, ar, br, z_thresh=z_thresh_val, outlier_mode=outlier_mode_dd, pct_thresh=pct_thresh_val, abs_thresh=abs_thresh_val)
                        except TypeError:
                            metrics_ref = compute_pair_sample_metrics(df_ref, id_col, group_col, ref_x, ref_y, ar, br, z_thresh=z_thresh_val)
                        for _, row in metrics_ref.iterrows():
                            ref_outlier_map[str(row[id_col])] = row["outlier_level"]

                for method in methods:
                    for xcol, ycol in pairs:
                        try:
                            df_pair = apply_value_range_filter(df, xcol, ycol, use_range=use_range_ck, lo=range_min_txt, hi=range_max_txt)
                            if id_col in df_pair.columns:
                                df_pair[id_col] = df_pair[id_col].astype(str).replace(["nan", "None", "<NA>", "NaN"], "Unknown")
                                df_pair[id_col] = df_pair[id_col].fillna("Unknown")
                            sub = df_pair[[xcol, ycol]].dropna()
                            if len(sub) < 2:
                                st.warning(f"警告: {xcol} vs {ycol} の有効データが2件未満のためスキップします。")
                                continue

                            x, y = sub[xcol].astype(float).values, sub[ycol].astype(float).values
                            r = pearson_r(x, y)
                            a, b, fit_info = regression_fit_info(x, y, method=method, deming_lambda=lam)

                            pair_key = f"{xcol}_vs_{ycol}"
                            try:
                                metrics_df = compute_pair_sample_metrics(df_pair, id_col, group_col, xcol, ycol, a, b, z_thresh=z_thresh_val, outlier_mode=outlier_mode_dd, pct_thresh=pct_thresh_val, abs_thresh=abs_thresh_val)
                            except TypeError:
                                metrics_df = compute_pair_sample_metrics(df_pair, id_col, group_col, xcol, ycol, a, b, z_thresh=z_thresh_val)

                            color_list = []
                            for _, row in metrics_df.iterrows():
                                sid = str(row[id_col])
                                sample_meta = run_metadata.setdefault(sid, {})
                                outliers_meta = sample_meta.setdefault("outliers", {})

                                if outlier_mode_dd == "error":
                                    level = row.get("outlier_level", "none")
                                    outliers_meta[pair_key] = {"level": level, "abs_residual": float(row.get("abs_residual", np.nan)), "rel_diff_yx_pct": float(row.get("rel_diff_yx_pct", np.nan))}
                                else:
                                    z_mad = row.get("z_MAD", np.nan)
                                    level = classify_outlier_level(abs(z_mad), thresh=z_thresh_val) if np.isfinite(z_mad) else "none"
                                    outliers_meta[pair_key] = {"level": level, "z_MAD": float(z_mad) if np.isfinite(z_mad) else None}

                                target_level = ref_outlier_map.get(sid, "none") if ref_outlier_map else level

                                if target_level == "strong_candidate": color_list.append("red")
                                elif target_level == "candidate": color_list.append("orange")
                                elif target_level == "mild_candidate": color_list.append("yellow")
                                else: color_list.append("#1f77b4")

                            ref_keys = [k for k, v in ref_outlier_map.items() if v in ("strong_candidate", "candidate", "mild_candidate")] if ref_outlier_map else None
                            try:
                                fig, used_sub, flagged, bias, loa = plot_suite(
                                    df=df_pair, id_col=id_col, group_col=group_col, xcol=xcol, ycol=ycol,
                                    method=method, lam=lam, a=a, b=b, r=r, fit_info=fit_info,
                                    z_thresh=z_thresh_val, outlier_label_top=label_top_val,
                                    fig_width=16, fig_height=10, dpi=100, external_colors=color_list,
                                    force_flagged_ids=ref_keys, outlier_mode=outlier_mode_dd, pct_thresh=pct_thresh_val, abs_thresh=abs_thresh_val
                                )
                            except TypeError:
                                fig, used_sub, flagged, bias, loa = plot_suite(
                                    df=df_pair, id_col=id_col, group_col=group_col, xcol=xcol, ycol=ycol,
                                    method=method, lam=lam, a=a, b=b, r=r, fit_info=fit_info,
                                    z_thresh=z_thresh_val, outlier_label_top=label_top_val,
                                    fig_width=16, fig_height=10, dpi=100, external_colors=color_list,
                                    force_flagged_ids=ref_keys
                                )

                            if fig is not None:
                                figures.append((fig, method, xcol, ycol))
                            else:
                                # Fallback generation if plot_suite unexpectedly returned None
                                import matplotlib.pyplot as plt
                                fallback_fig, fallback_ax = plt.subplots(figsize=(8, 6))
                                fallback_ax.scatter(x, y, alpha=0.7)
                                fallback_ax.set_title(f"Fallback Plot: {xcol} vs {ycol}")
                                fallback_ax.set_xlabel(xcol)
                                fallback_ax.set_ylabel(ycol)
                                figures.append((fallback_fig, method, xcol, ycol))
                                if bias is None: bias = float(np.nanmean(y - x))
                                if flagged is None: flagged = pd.DataFrame()
                                st.warning(f"警告: {xcol} vs {ycol} の描画処理で予期せぬ空データが返されたため、フォールバック描画を行いました。")

                            summary_rows.append({"regression": method, "X": xcol, "Y": ycol, "n": len(x), "r": r,
                                                 "slope": a, "intercept": b, "BA_bias": bias if bias is not None and not np.isnan(bias) else None, "n_outliers": len(flagged) if flagged is not None else 0})
                            if not metrics_df.empty: sample_metric_tables.append(metrics_df.assign(regression=method, X=xcol, Y=ycol))
                            if flagged is not None and not flagged.empty: outlier_tables.append(flagged.assign(regression=method, X=xcol, Y=ycol))
                        except Exception as e:
                            st.error(f"{xcol} vs {ycol} の解析中にエラーが発生しました: {e}")

                # 解析設定をまとめて保存（Excel出力時の情報表示用）
                run_settings = {
                    "parsed_dir": str(parsed_dir),
                    "input_stem": parsed_dir.name,
                    "mode": mode,
                    "regression_method": reg_method,
                    "all_regression": all_reg_ck,
                    "deming_lambda": deming_lambda_val,
                    "outlier_mode": outlier_mode_dd,
                    "z_thresh": z_thresh_val,
                    "pct_thresh": pct_thresh_val,
                    "abs_thresh": abs_thresh_val,
                    "outlier_label_top": label_top_val,
                    "use_range": use_range_ck,
                    "range_min": range_min_txt,
                    "range_max": range_max_txt,
                    "ref_pair_x": ref_pair_x,
                    "ref_pair_y": ref_pair_y,
                    "n_pairs": len(pairs),
                    "pairs": [(x, y) for x, y in pairs],
                }

                st.session_state["analysis_results"] = {
                    "run_metadata": run_metadata,
                    "ref_outlier_map": ref_outlier_map,
                    "summary_rows": summary_rows,
                    "sample_metric_tables": sample_metric_tables,
                    "outlier_tables": outlier_tables,
                    "figures": figures,
                    "run_settings": run_settings,
                }
                st.session_state["metadata_enhanced"] = run_metadata
                st.session_state["ref_outlier_map"] = ref_outlier_map

                st.success(f"解析完了! {len(summary_rows)}件のペアを処理しました。内容を確認後、必要であれば『Excel出力』タブへ進んでください。")

        if st.session_state.get("analysis_results") and show_py_ck:
            st.markdown("### 解析結果のグラフ")
            shown = 0
            for fig, method, xcol, ycol in st.session_state["analysis_results"]["figures"]:
                if shown < max_show_val:
                    st.pyplot(fig)
                    shown += 1
                else:
                    break

    # ----------------------------------------------------
    # TAB 2: Excel出力(旧)
    # ----------------------------------------------------
    with tab_excel_old:
        st.header("③ Excel出力")
        if st.session_state["analysis_results"] is None:
            st.warning("先に『相関解析』タブで『②解析実行』を行ってください。")
        else:
            # 解析設定の概要を常に表示
            res = st.session_state["analysis_results"]
            s = res.get("run_settings", {})
            if s:
                _mode_labels = {"adjacent": "隣接ペア", "all": "全ペア", "baseline": "基準ペア"}
                _outlier_labels = {"zMAD": "zMAD (標準化残差)", "error": "error (臨床的許容誤差)"}
                _reg_labels = {"OLS": "OLS（最小二乗）", "Deming": "Deming（両軸誤差）",
                               "TheilSen": "Theil-Sen（ロバスト）", "PassingBablok": "Passing-Bablok（ノンパラメトリック）"}
                _ref_x = s.get("ref_pair_x", "(未選択)")
                _ref_y = s.get("ref_pair_y", "(未選択)")
                _ref_str = f"{_ref_x} vs {_ref_y}" if _ref_x != "(未選択)" and _ref_y != "(未選択)" else "なし"

                if s.get("outlier_mode") == "zMAD":
                    _outlier_detail = f"z(MAD) ≥ {s.get('z_thresh', 3.5)}"
                else:
                    _outlier_detail = f"許容誤差 {s.get('pct_thresh', 10.0)}% / 絶対値 {s.get('abs_thresh', 2.0)}"

                st.info(
                    f"📋 **解析設定の確認**\n\n"
                    f"| 項目 | 値 |\n"
                    f"|---|---|\n"
                    f"| 📂 解析対象ファイル | `{s.get('input_stem', '?')}` |\n"
                    f"| 🔗 ペアモード | {_mode_labels.get(s.get('mode',''), s.get('mode','?'))} ({s.get('n_pairs', '?')} ペア) |\n"
                    f"| 📐 回帰法 | {_reg_labels.get(s.get('regression_method',''), s.get('regression_method','?'))}{' (全回帰法)' if s.get('all_regression') else ''} |\n"
                    f"| 🔍 乖離判定基準 | {_outlier_labels.get(s.get('outlier_mode',''), '?')} ({_outlier_detail}) |\n"
                    f"| 🎯 乖離基準ペア | {_ref_str} |\n"
                    f"| 📏 対象範囲絞り | {'あり: ' + str(s.get('range_min','')) + ' ～ ' + str(s.get('range_max','')) if s.get('use_range') else 'なし'} |"
                )

            if st.button("③Excel出力", type="primary"):
                with st.spinner("Excelファイル作成中..."):
                    try:
                        parsed_dir = st.session_state["parsed_dir"]

                        dirs = make_output_dirs(OUTPUT_ROOT, input_stem=parsed_dir.name)
                        img_paths = []

                        for fig, method, xcol, ycol in res["figures"]:
                            png = dirs["plots"] / f"QC_{safe_name(method)}_{safe_name(ycol)}_vs_{safe_name(xcol)}.png"
                            fig.savefig(png, bbox_inches="tight")
                            img_paths.append(png)

                        with open(parsed_dir / "metadata.json", "w", encoding="utf-8") as f:
                            json.dump(res["run_metadata"], f, indent=2, ensure_ascii=False)

                        output_xlsx = dirs["excel"] / f"{parsed_dir.name}{OUT_SUFFIX}.xlsx"
                        import openpyxl
                        wb = openpyxl.Workbook()
                        wb.save(output_xlsx)

                        if img_paths:
                            try:
                                insert_images_into_excel(input_xlsx=output_xlsx, output_xlsx=output_xlsx, image_paths=img_paths, plot_sheet=SHEET_PLOTS)
                            except Exception as e:
                                st.warning(f"Plots could not be inserted into Excel: {e}")

                        if res["summary_rows"]: write_df_to_sheet(output_xlsx, pd.DataFrame(res["summary_rows"]), SHEET_SUMMARY)
                        if res["sample_metric_tables"]: write_df_to_sheet(output_xlsx, pd.concat(res["sample_metric_tables"]), SHEET_SAMPLE_METRICS)
                        if res["outlier_tables"]: write_df_to_sheet(output_xlsx, pd.concat(res["outlier_tables"]), SHEET_OUTLIERS)

                        # 出力完了メッセージ（解析設定 + 出力先をまとめて表示）
                        st.success(f"✅ Excel保存完了!")
                        st.info(
                            f"📁 **出力先情報**\n\n"
                            f"| 項目 | パス |\n"
                            f"|---|---|\n"
                            f"| 📊 Excelファイル | `{output_xlsx}` |\n"
                            f"| 🖼️ プロット保存先 | `{dirs['plots']}` |\n"
                            f"| 📂 出力フォルダ | `{dirs['plots'].parent}` |"
                        )

                    except Exception as e:
                        st.error(f"出力エラー:\n{traceback.format_exc()}")

    # ----------------------------------------------------
    # TAB 3: タイムコース表示
    # ----------------------------------------------------
    with tab_timecourse:
        st.header("タイムコース反応表示")
        profile_df = st.session_state["profile_df"]
        df = st.session_state["df"]

        if profile_df is not None and df is not None:
            items = list(profile_df["項目名"].unique())
            if not items:
                st.warning("プロファイルデータに項目名がありません。")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    tc_item = st.selectbox("表示項目", options=items)

                # Dynamic range for the selected item
                vmin, vmax = 0.0, 1000.0
                if tc_item in df.columns:
                    vals = pd.to_numeric(df[tc_item], errors="coerce").dropna()
                    if not vals.empty:
                        vmin, vmax = float(vals.min()), float(vals.max())

                with col2:
                    tc_conc_range = st.slider("濃度範囲", min_value=float(vals.min()) if not vals.empty else 0.0,
                                              max_value=float(vals.max()) if not vals.empty else 1000.0,
                                              value=(vmin, vmax), step=0.1)

                times = sorted(profile_df["時間"].unique())
                baseline_time_options = {"(生データ表示)": None}
                baseline_time_options.update({f"{t:.1f}s": t for t in times})

                col3, col4 = st.columns(2)
                with col3:
                    tc_outlier = st.selectbox("乖離選択（一般検体のみ適用）", options=["all", "outlier", "normal"],
                                              format_func=lambda x: {"all":"全て", "outlier":"乖離のみ", "normal":"非乖離のみ"}[x])
                with col4:
                    tc_baseline_time_name = st.selectbox("基準時間(秒)", options=list(baseline_time_options.keys()))
                    tc_baseline_time = baseline_time_options[tc_baseline_time_name]

                col5, col6 = st.columns(2)
                with col5:
                    show_calibrators = st.checkbox("キャリブレーター表示", value=True,
                                                   help="C+数字パターン（例: C001）の検体を表示します。緑色で描画されます。")
                with col6:
                    st.markdown("""<div style='padding-top:8px; font-size:0.85em; color:#666;'>
                        🟢 キャリブレーターは乖離フィルタの対象外です
                    </div>""", unsafe_allow_html=True)

                if st.button("タイムコース表示", type="primary"):
                    metadata = st.session_state.get("metadata_enhanced") or st.session_state.get("metadata")
                    ref_outlier_map = st.session_state.get("ref_outlier_map")
                    id_col = st.session_state.get("id_col")

                    cmin, cmax = tc_conc_range
                    df_filtered = df[(pd.to_numeric(df[tc_item], errors='coerce') >= cmin) &
                                     (pd.to_numeric(df[tc_item], errors='coerce') <= cmax)]
                    allowed_sids = set(df_filtered[id_col].astype(str).tolist())

                    id_mapping = {}
                    if "依頼No." in df.columns and id_col in df.columns:
                        id_mapping = dict(zip(df["依頼No."].astype(str), df[id_col].astype(str)))

                    df_item = profile_df[profile_df["項目名"] == tc_item]
                    fig, ax = plt.subplots(figsize=(12, 7))

                    from matplotlib.lines import Line2D

                    plotted_count = 0
                    calib_count = 0
                    for sid, gdf in df_item.groupby("依頼No."):
                        sid_str = str(sid)
                        mapped_id = id_mapping.get(sid_str, sid_str)
                        if mapped_id not in allowed_sids: continue

                        calib = is_calibrator(sid_str)

                        # ── キャリブレーター ──────────────────────────────────
                        if calib:
                            if not show_calibrators:
                                continue
                            # キャリブレーターは乖離判定と独立した固有カテゴリ
                            color, lw, alpha = "#2ca02c", 1.5, 0.55  # 緑
                            calib_count += 1

                        # ── 一般検体 ─────────────────────────────────────────
                        else:
                            level = "none"
                            if ref_outlier_map and mapped_id in ref_outlier_map:
                                level = ref_outlier_map[mapped_id]
                            elif metadata and mapped_id in metadata and "outliers" in metadata[mapped_id]:
                                levels = [v["level"] for v in metadata[mapped_id]["outliers"].values()]
                                if "strong_candidate" in levels: level = "strong_candidate"
                                elif "candidate" in levels: level = "candidate"
                                elif "mild_candidate" in levels: level = "mild_candidate"

                            is_outlier = level in ["strong_candidate", "candidate", "mild_candidate"]
                            if tc_outlier == "outlier" and not is_outlier: continue
                            if tc_outlier == "normal" and is_outlier: continue

                            if level == "strong_candidate":   color, lw, alpha = "red",     1.5, 0.9
                            elif level == "candidate":        color, lw, alpha = "orange",  1.5, 0.8
                            elif level == "mild_candidate":   color, lw, alpha = "yellow",  1.5, 0.7
                            else:                             color, lw, alpha = "#1f77b4", 1.5, 0.3

                        time_vals = gdf["時間"].values
                        abs_vals = gdf["吸光度"].values

                        if tc_baseline_time is not None:
                            idx = np.argmin(np.abs(time_vals - tc_baseline_time))
                            base_abs = abs_vals[idx]
                            abs_vals = abs_vals - base_abs

                        ax.plot(time_vals, abs_vals, color=color, linewidth=lw, alpha=alpha)
                        plotted_count += 1

                    ax.set_title(
                        f"タイムコース反応: {tc_item}  "
                        f"（一般検体: {plotted_count - calib_count}件"
                        + (f", キャリブレーター: {calib_count}件" if show_calibrators else "")
                        + f", 基準時間: {tc_baseline_time if tc_baseline_time is not None else 'None'}s)"
                    )
                    ax.set_xlabel("時間(秒)")
                    ax.set_ylabel("吸光度")
                    ax.grid(True, alpha=0.3)
                    if tc_baseline_time is not None:
                        ax.axvline(tc_baseline_time, color='black', linestyle='--', alpha=0.5)

                    # 凡例
                    legend_elements = [
                        Line2D([0], [0], color="#1f77b4", lw=1.5, alpha=0.7, label="一般検体（正常）"),
                        Line2D([0], [0], color="yellow",  lw=1.5, alpha=0.9, label="一般検体（軽度乖離）"),
                        Line2D([0], [0], color="orange",  lw=1.5, alpha=0.9, label="一般検体（乖離）"),
                        Line2D([0], [0], color="red",     lw=1.5, alpha=0.9, label="一般検体（強乖離）"),
                    ]
                    if show_calibrators:
                        legend_elements.append(
                            Line2D([0], [0], color="#2ca02c", lw=1.5, alpha=0.8, label="キャリブレーター")
                        )
                    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
                              framealpha=0.85, edgecolor="#cccccc")

                    st.pyplot(fig)
                    plt.close(fig)
    # ----------------------------------------------------
    # TAB X: 検体測定
    # ----------------------------------------------------
    with tab_measurement:
        st.header("検体測定")
        st.write("試薬パラメーターの測光区間を用いて処理値を計算し、キャリブレーション結果を用いて濃度を算出します。")

    # ----------------------------------------------------
    # TAB 4: キャリブレーション解析
    # ----------------------------------------------------
    with tab_calib:
        st.header("キャリブレーション解析（検量線構築 & 濃度再計算）")
        profile_df = st.session_state["profile_df"]
        measurement_df = st.session_state["df"]

        if profile_df is not None and measurement_df is not None:
            items = list(profile_df["項目名"].unique())
            if not items:
                st.warning("プロファイルデータに項目名がありません。")
            else:
                # === Session state for Cal config ===
                for _k in ("cal_config", "cal_patterns", "cal_results", "cal_registry", "cal_registry_draft", "cal_selection_key"):
                    if _k not in st.session_state:
                        st.session_state[_k] = None

                # 試薬マスター & ポート割り当て (試薬登録タブの結果を利用)
                try:
                    with open(PROJECT_ROOT / "data" / "reagent_master.json", "r", encoding="utf-8") as f:
                        _rmaster = json.load(f)
                except Exception:
                    _rmaster = {}
                _pm = st.session_state.get("port_mapping")
                if isinstance(_pm, pd.DataFrame) and not _pm.empty:
                    port_to_reagent = dict(zip(_pm["測定ポート (項目名)"], _pm["割り当て試薬"].fillna("")))
                else:
                    port_to_reagent = {it: next((k for k in _rmaster if str(it).startswith(k)), "") for it in items}

                # ============================================================
                # Step 1: キャリブレーター一括登録
                # ============================================================
                st.subheader("Step 1: キャリブレーター一括登録")
                st.caption("全項目のC+数字IDを自動検出し、ID連番ブロック × 試薬マスターのCal点数でレベル割り当て、"
                           "測定時刻でCalセット（Calロット）を推定します。濃度はCalセット×試薬ごとに1回入力するだけです。")

                col_det, col_gap, col_hist = st.columns([1, 1, 2])
                with col_gap:
                    set_gap = st.number_input("Calセット区切り (分)", min_value=1, max_value=600, value=30,
                                              help="Calブロックの開始時刻がこの時間以上離れていたら別のCalセットとみなします。",
                                              key="cal_set_gap")
                with col_det:
                    st.write("")
                    if st.button("🔍 自動検出", type="primary", key="btn_cal_autodetect", use_container_width=True):
                        entries = auto_assign_calibrators(profile_df, measurement_df, port_to_reagent, _rmaster,
                                                          set_gap_minutes=int(set_gap))
                        st.session_state["cal_registry_draft"] = {"entries": entries, "sets": {}, "aggregation": "median"}
                        st.session_state.pop("cal_entries_editor", None)
                        st.session_state.pop("cal_conc_editor", None)
                with col_hist:
                    _pdir = st.session_state.get("parsed_dir")
                    _hist = list_cal_registries(_pdir) if _pdir else []
                    if _hist:
                        _sel_hist = st.selectbox("登録履歴から読込", [p.name for p in _hist], key="cal_hist_sel")
                        if st.button("📂 履歴を読み込む", key="btn_cal_hist_load"):
                            st.session_state["cal_registry_draft"] = load_cal_config(Path(_pdir) / "cal_registry" / _sel_hist)
                            st.session_state.pop("cal_entries_editor", None)
                            st.session_state.pop("cal_conc_editor", None)
                            st.rerun()
                    else:
                        st.caption("登録履歴はまだありません。")

                draft = st.session_state.get("cal_registry_draft")
                if draft and draft.get("entries"):
                    entries = draft["entries"]

                    # --- 1-a: 自動割り当て結果の確認・編集 ---
                    st.markdown("**① 自動割り当て結果** — Calセット名の修正・使用有無を編集できます")
                    ent_df = pd.DataFrame([{
                        "使用": e.get("enabled", True),
                        "Calセット": e["cal_set"],
                        "項目": e["item"],
                        "試薬": e["reagent"] or "-",
                        "ID範囲": e["id_range"],
                        "レベル×n": f"{e['n_levels']}×{e['n_reps'] if e['n_reps'] else '?'}",
                        "割当方法": e["method"],
                        "開始時刻": (e["t_start"] or "")[-8:],
                        "装置測定値(レベル中央値)": ", ".join("-" if m is None else f"{m:g}" for m in e["level_medians"]),
                        "判定": "✅ OK" if not e["warnings"] else "⚠ " + " / ".join(e["warnings"]),
                    } for e in entries])
                    edited_ent = st.data_editor(
                        ent_df,
                        column_config={c: st.column_config.Column(disabled=True) for c in ent_df.columns if c not in ("使用", "Calセット")},
                        use_container_width=True, hide_index=True, num_rows="fixed", key="cal_entries_editor",
                    )
                    for e, (_, row) in zip(entries, edited_ent.iterrows()):
                        e["enabled"] = bool(row["使用"])
                        e["cal_set"] = str(row["Calセット"]).strip() or e["cal_set"]

                    n_warn = sum(1 for e in entries if e["warnings"] and e.get("enabled", True))
                    if n_warn:
                        st.warning(f"⚠ {n_warn}件に警告があります。割り当てを確認してください（低濃度レベルのCV警告は多くの場合問題ありません）。")

                    # --- 1-b: Calセット × 試薬ごとの表示値濃度 & ロット名 ---
                    st.markdown("**② Calセット別 表示値濃度・ロット名** — セット×試薬ごとに1回入力すれば全項目へ反映されます")
                    keys = sorted({(e["cal_set"], e["reagent"] or "-") for e in entries if e.get("enabled", True)})
                    max_lv = max([e["n_levels"] for e in entries] + [1])
                    old_sets = draft.get("sets", {})
                    conc_rows = []
                    for cs, rg in keys:
                        prev = old_sets.get(cs, {})
                        prev_c = prev.get("concentrations", {}).get(rg, [])
                        n_items = sum(1 for e in entries if e["cal_set"] == cs and (e["reagent"] or "-") == rg and e.get("enabled", True))
                        row = {"Calセット": cs, "試薬": rg, "ロット名": prev.get("lot_name", cs), "対象項目数": n_items}
                        for i in range(max_lv):
                            row[f"Cal{i}"] = float(prev_c[i]) if i < len(prev_c) and prev_c[i] is not None else 0.0
                        conc_rows.append(row)
                    conc_df = pd.DataFrame(conc_rows)
                    conc_cfg = {"Calセット": st.column_config.TextColumn(disabled=True),
                                "試薬": st.column_config.TextColumn(disabled=True),
                                "対象項目数": st.column_config.NumberColumn(disabled=True)}
                    for i in range(max_lv):
                        conc_cfg[f"Cal{i}"] = st.column_config.NumberColumn(min_value=0.0, format="%.3f")
                    edited_conc = st.data_editor(conc_df, column_config=conc_cfg, use_container_width=True,
                                                 hide_index=True, num_rows="fixed", key="cal_conc_editor")

                    new_sets = {}
                    for _, r in edited_conc.iterrows():
                        s = new_sets.setdefault(r["Calセット"], {"lot_name": r["ロット名"], "concentrations": {}})
                        s["lot_name"] = str(r["ロット名"]).strip() or r["Calセット"]
                        s["concentrations"][r["試薬"]] = [float(r[f"Cal{i}"]) for i in range(max_lv)]
                    draft["sets"] = new_sets

                    default_agg = draft.get("aggregation", "median")
                    draft["aggregation"] = st.selectbox("代表値算出法", ["median", "mean"], index=0 if default_agg == "median" else 1,
                                                        format_func=lambda x: "中央値" if x == "median" else "平均値", key="agg_method")

                    if st.button("▶ Step 1 完了: 全項目を一括登録", type="primary", key="btn_cal_register"):
                        zero_sets = [f"{cs}/{rg}" for cs, rg in keys
                                     if all(c == 0.0 for c in new_sets.get(cs, {}).get("concentrations", {}).get(rg, [0.0]))]
                        if zero_sets:
                            st.error(f"表示値濃度が未入力（全て0）のCalセットがあります: {', '.join(zero_sets)}")
                        else:
                            registry = json.loads(json.dumps(draft, default=str))
                            st.session_state["cal_registry"] = registry
                            st.session_state["cal_selection_key"] = None
                            _pdir = st.session_state.get("parsed_dir")
                            msg = ""
                            if _pdir:
                                try:
                                    reg_path = save_cal_registry(registry, _pdir)
                                    msg = f"（履歴保存: cal_registry/{reg_path.name}）"
                                except Exception as ex:
                                    msg = f"（履歴保存に失敗: {ex}）"
                            n_ok = sum(1 for e in registry["entries"] if e.get("enabled", True))
                            st.success(f"✅ {n_ok}本の検量線定義を一括登録しました {msg}")
                elif draft is not None:
                    st.info("キャリブレーター（C+数字のID）が検出されませんでした。")
                else:
                    st.info("「🔍 自動検出」を押すか、登録履歴を読み込んでください。")

                st.divider()

                # ============================================================
                # 解析対象の選択 (項目 × Calセット)
                # ============================================================
                st.subheader("解析対象の選択")
                registry = st.session_state.get("cal_registry")
                if not registry:
                    st.info("Step 1 で一括登録を完了すると、ここで項目とCalセットを選択できます。")
                    st.session_state["cal_config"] = None
                    tc_item_tab4 = items[0]
                else:
                    reg_entries = [e for e in registry["entries"] if e.get("enabled", True)]
                    reg_items = sorted({e["item"] for e in reg_entries}, key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)])
                    col_si, col_ss = st.columns(2)
                    with col_si:
                        tc_item_tab4 = st.selectbox("解析項目", options=reg_items, key="tc_item_tab4")
                    with col_ss:
                        set_opts = [e["cal_set"] for e in reg_entries if e["item"] == tc_item_tab4]
                        sel_set = st.selectbox(
                            "Calセット", options=set_opts, key="tc_calset_tab4",
                            format_func=lambda s: f"{s} ({registry['sets'].get(s, {}).get('lot_name', s)})",
                        )

                    base_cfg = build_cal_config_from_registry(registry, tc_item_tab4, sel_set)
                    sel_key = f"{tc_item_tab4}|{sel_set}"
                    if st.session_state.get("cal_selection_key") != sel_key:
                        st.session_state["cal_selection_key"] = sel_key
                        st.session_state["cal_results"] = None
                        st.session_state["cal_patterns"] = None
                        st.session_state["lot_comparison_data"] = None

                    # 個別修正 (例外時のみ)
                    with st.expander("✏️ この項目のレベル表を個別に修正する（例外時のみ）", expanded=False):
                        lv_df = pd.DataFrame([{
                            "レベル": f"Cal {i}",
                            "表示値濃度": base_cfg["concentrations"][i],
                            "依頼No. (n回分)": ", ".join(ids),
                        } for i, ids in enumerate(base_cfg["level_table"])])
                        edited_lv = st.data_editor(
                            lv_df,
                            column_config={
                                "レベル": st.column_config.TextColumn(disabled=True),
                                "表示値濃度": st.column_config.NumberColumn(min_value=0.0, format="%.3f"),
                            },
                            use_container_width=True, hide_index=True, num_rows="fixed",
                            key=f"cal_editor_{sel_key}",
                        )
                        base_cfg["concentrations"] = [float(v) for v in edited_lv["表示値濃度"]]
                        base_cfg["level_table"] = [[s.strip() for s in str(v).split(",") if s.strip()]
                                                   for v in edited_lv["依頼No. (n回分)"]]

                    st.session_state["cal_config"] = base_cfg
                    lv_desc = " / ".join(f"Cal{i}={c:g}" for i, c in enumerate(base_cfg["concentrations"]))
                    st.caption(f"選択中: **{tc_item_tab4}** × **{base_cfg['lot_name']}** "
                               f"({base_cfg['n_levels']}レベル × n={base_cfg['n_replicates']}) — {lv_desc}")

                st.divider()

                # ============================================================
                # Step 2: 測光区間 & 検量線モード設定
                # ============================================================
                st.subheader("Step 2: 測光区間 & 検量線モード設定")

                if st.session_state["cal_config"] is None:
                    st.info("先に Step 1 でキャリブレーターを登録してください。")
                else:
                    cal_cfg = st.session_state["cal_config"]
                    times = sorted(profile_df[profile_df["項目名"] == tc_item_tab4]["時間"].unique())
                    time_opts = {f"{t:.1f}s": t for t in times}
                    time_keys = list(time_opts.keys())

                    st.info("計算式: 処理値(mAbs/min) = {(Abs_end - Abs_start) × 0.1} / {(Time_end - Time_start) / 60}")

                    n_patterns = st.number_input("比較パターン数", min_value=1, max_value=6, value=2, key="n_patterns")

                    pattern_defs = []
                    cols = st.columns(min(int(n_patterns), 3))
                    for i in range(int(n_patterns)):
                        with cols[i % len(cols)]:
                            st.markdown(f"**パターン {i+1}**")
                            pname = st.text_input("名称", value="Base" if i == 0 else f"New-{chr(64+i)}", key=f"pname_{i}")
                            pstart = st.selectbox("開始時間", options=time_keys, index=min(9, len(time_keys)-1), key=f"pstart_{i}")
                            pend = st.selectbox("終了時間", options=time_keys, index=min(19, len(time_keys)-1), key=f"pend_{i}")
                            pcurve = st.selectbox("検量線モード", ["piecewise_linear", "spline"],
                                                  format_func=lambda x: "折れ線" if x == "piecewise_linear" else "スプライン", key=f"pcurve_{i}")
                            pattern_defs.append({
                                "name": pname,
                                "time_start": time_opts[pstart],
                                "time_end": time_opts[pend],
                                "curve_mode": pcurve,
                            })

                    if st.button("▶ Step 2 完了: 検量線を構築 & 全検体再計算", type="primary", key="btn_build_curves"):
                        with st.spinner("検量線構築 & 全検体再計算中..."):
                            cal_cfg = st.session_state["cal_config"]
                            level_table = cal_cfg["level_table"]
                            concentrations = cal_cfg["concentrations"]
                            agg = cal_cfg["aggregation"]

                            all_pattern_results = []

                            for pat in pattern_defs:
                                # 全サンプルの処理値(Rate)算出
                                rates = calc_rates_batch(profile_df, tc_item_tab4, pat["time_start"], pat["time_end"])

                                # Calレベル代表値の集約
                                cal_agg_rates = aggregate_cal_rates(rates, level_table, method=agg)

                                # 検量線構築
                                curve = build_calibration_curve(cal_agg_rates, concentrations, curve_mode=pat["curve_mode"])

                                # Cal ID → 表示値濃度 マッピング
                                cal_id_to_conc = {}
                                for lv_idx, ids in enumerate(level_table):
                                    for cid in ids:
                                        if cid and lv_idx < len(concentrations):
                                            cal_id_to_conc[cid] = concentrations[lv_idx]

                                # 全検体再計算
                                if curve is not None:
                                    recalc_df = recalculate_all_samples(
                                        profile_df, measurement_df, tc_item_tab4,
                                        curve, pat["time_start"], pat["time_end"],
                                        cal_id_to_conc=cal_id_to_conc
                                    )
                                else:
                                    recalc_df = pd.DataFrame()

                                # Calの詳細テーブル & サマリーテーブル
                                detail_df = get_cal_level_detail_table(rates, level_table, concentrations, tc_item_tab4)
                                summary_df = get_cal_level_summary_table(rates, level_table, concentrations, agg_method=agg)

                                all_pattern_results.append({
                                    "pattern": pat,
                                    "rates": rates,
                                    "cal_agg_rates": cal_agg_rates,
                                    "curve": curve,
                                    "recalc_df": recalc_df,
                                    "detail_df": detail_df,
                                    "summary_df": summary_df,
                                    "cal_id_to_conc": cal_id_to_conc,
                                })

                            st.session_state["cal_results"] = all_pattern_results
                            st.session_state["cal_patterns"] = pattern_defs
                            st.success(f"✅ {len(pattern_defs)}パターンの検量線構築 & 再計算が完了しました。")

                st.divider()

                # ============================================================
                # Step 3: 結果表示 (方式検討: 区間・検量線比較)
                # ============================================================
                st.subheader("Step 3: 結果表示 (測光区間 & 検量線方式の検討)")

                if st.session_state["cal_results"] is None:
                    st.info("先に Step 2 で検量線構築 & 再計算を実行してください。")
                else:
                    cal_cfg = st.session_state["cal_config"]
                    results = st.session_state["cal_results"]
                    patterns = st.session_state["cal_patterns"]

                    # 単位取得
                    unit = ""
                    parsed_dir = st.session_state.get("parsed_dir")
                    if parsed_dir and (Path(parsed_dir) / "metadata.json").exists():
                        try:
                            with open(Path(parsed_dir) / "metadata.json", "r", encoding="utf-8") as f:
                                disk_metadata = json.load(f)
                            if "measurement_units" in disk_metadata and tc_item_tab4 in disk_metadata["measurement_units"]:
                                unit = f" ({disk_metadata['measurement_units'][tc_item_tab4]})"
                        except Exception:
                            pass

                    # --- 3a: 検量線プロット（全パターン重ね描き）---
                    st.markdown("#### ① 検量線プロット比較")
                    fig_cal, ax_cal = plt.subplots(figsize=(10, 6))
                    pat_colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 1)))

                    for idx, res in enumerate(results):
                        pat = res["pattern"]
                        curve = res["curve"]
                        cal_agg = res["cal_agg_rates"]
                        concs = cal_cfg["concentrations"]
                        color = pat_colors[idx]
                        label_prefix = pat["name"]
                        mode_str = "折れ線" if pat["curve_mode"] == "piecewise_linear" else "スプライン"

                        # Cal代表値の散布
                        valid_pairs = [(r, c) for r, c in zip(cal_agg, concs) if np.isfinite(r) and np.isfinite(c)]
                        if valid_pairs:
                            vr, vc = zip(*valid_pairs)
                            ax_cal.scatter(vc, vr, color=color, marker="o", s=60, zorder=5,
                                           label=f"{label_prefix} Cal点")

                        # 検量線カーブ
                        if curve is not None:
                            c_arr = curve["concentrations"]
                            r_arr = curve["rates"]
                            if curve["curve_mode"] == "piecewise_linear":
                                ax_cal.plot(c_arr, r_arr, color=color, linewidth=1.8, alpha=0.8,
                                            label=f"{label_prefix} ({mode_str})")
                            else:
                                dense_r = np.linspace(r_arr.min(), r_arr.max(), 200)
                                dense_c = [predict_concentration(curve, rv) for rv in dense_r]
                                ax_cal.plot(dense_c, dense_r, color=color, linewidth=1.8, linestyle="--", alpha=0.8,
                                            label=f"{label_prefix} ({mode_str})")

                    ax_cal.set_xlabel(f"濃度{unit}")
                    ax_cal.set_ylabel("処理値 (mAbs/min)")
                    ax_cal.set_title(f"検量線比較: {tc_item_tab4} [{cal_cfg['lot_name']}]")
                    ax_cal.grid(True, alpha=0.3)
                    ax_cal.legend(fontsize=8, loc="best")
                    st.pyplot(fig_cal)
                    plt.close(fig_cal)

                    # --- 3b: Calレベル詳細テーブル & CV% サマリー ---
                    st.markdown("#### ② キャリブレーター処理値 & ばらつき (CV%)")
                    for idx, res in enumerate(results):
                        pat = res["pattern"]
                        detail_df = res["detail_df"]
                        summary_df = res.get("summary_df", pd.DataFrame())

                        with st.expander(f"📋 パターン: {pat['name']} ({pat['time_start']:.1f}s ~ {pat['time_end']:.1f}s)", expanded=(idx == 0)):
                            st.markdown("**【レベル別サマリー（代表値・平均値・標準偏差・CV%）】**")
                            st.dataframe(summary_df, use_container_width=True)

                            if not detail_df.empty:
                                st.markdown("**【個別測定値一覧（Replicates）】**")
                                st.dataframe(detail_df, use_container_width=True)

                    # --- 3c: 全検体再計算結果テーブル ---
                    st.markdown("#### ③ 全検体再計算結果マトリクス")

                    # Merge all pattern results into a single wide table
                    if len(results) > 0 and not results[0]["recalc_df"].empty:
                        base_df = results[0]["recalc_df"][["依頼No.", "サンプル区分", "装置測定値"]].copy()

                        for idx, res in enumerate(results):
                            pat = res["pattern"]
                            rdf = res["recalc_df"]
                            if not rdf.empty:
                                base_df = base_df.merge(
                                    rdf[["依頼No.", "処理値", "再計算濃度"]].rename(columns={
                                        "処理値": f"処理値_{pat['name']}",
                                        "再計算濃度": f"再計算濃度_{pat['name']}",
                                    }),
                                    on="依頼No.", how="left"
                                )

                        # 濃度差列の追加（Baseとの差）
                        if len(results) >= 2:
                            base_col = f"再計算濃度_{results[0]['pattern']['name']}"
                            for idx in range(1, len(results)):
                                new_col = f"再計算濃度_{results[idx]['pattern']['name']}"
                                diff_col = f"濃度差_{results[idx]['pattern']['name']}-{results[0]['pattern']['name']}"
                                if base_col in base_df.columns and new_col in base_df.columns:
                                    base_df[diff_col] = base_df[new_col] - base_df[base_col]

                        st.dataframe(base_df, use_container_width=True)
                        st.session_state["recalc_matrix_df"] = base_df

                    # --- 3d: 相関プロット（パターン間比較） ---
                    if len(results) >= 2:
                        st.markdown("#### ④ 相関プロット: 測光パターン間比較")

                        base_res = results[0]
                        base_pat_name = base_res["pattern"]["name"]

                        for cmp_idx in range(1, len(results)):
                            cmp_res = results[cmp_idx]
                            cmp_pat_name = cmp_res["pattern"]["name"]

                            base_rdf = base_res["recalc_df"]
                            cmp_rdf = cmp_res["recalc_df"]

                            if base_rdf.empty or cmp_rdf.empty:
                                continue

                            # 一般検体のみ（キャリブレーター除外）
                            merged = base_rdf[base_rdf["サンプル区分"] == "一般検体"][["依頼No.", "再計算濃度"]].rename(
                                columns={"再計算濃度": "base_conc"}
                            ).merge(
                                cmp_rdf[cmp_rdf["サンプル区分"] == "一般検体"][["依頼No.", "再計算濃度"]].rename(
                                    columns={"再計算濃度": "cmp_conc"}
                                ),
                                on="依頼No.", how="inner"
                            ).dropna(subset=["base_conc", "cmp_conc"])

                            if len(merged) < 2:
                                st.warning(f"{base_pat_name} vs {cmp_pat_name}: 有効なデータが不足しています。")
                                continue

                            corr_x = merged["base_conc"].values.astype(float)
                            corr_y = merged["cmp_conc"].values.astype(float)

                            corr_r = pearson_r(corr_x, corr_y)
                            corr_a, corr_b, corr_fi = regression_fit_info(corr_x, corr_y, method="PassingBablok")

                            fig_corr, ax_corr = plt.subplots(figsize=(8, 8))
                            ax_corr.scatter(corr_x, corr_y, color="steelblue", s=30, alpha=0.7, zorder=5)

                            # y=x 線
                            lo = min(float(np.nanmin(corr_x)), float(np.nanmin(corr_y)))
                            hi = max(float(np.nanmax(corr_x)), float(np.nanmax(corr_y)))
                            margin = (hi - lo) * 0.05 if hi > lo else 1.0
                            ax_corr.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                                         "--", lw=1, alpha=0.6, color="gray", label="y=x")

                            # 回帰直線
                            if np.isfinite(corr_a) and np.isfinite(corr_b):
                                xx_line = np.array([lo - margin, hi + margin])
                                ax_corr.plot(xx_line, corr_a * xx_line + corr_b,
                                             lw=1.8, alpha=0.85, color="darkorange", label="回帰直線")

                            # 統計情報テキスト
                            stat_lines = [
                                f"n={len(merged)}",
                                f"Pearson r={corr_r:.4f}",
                                f"Passing-Bablok",
                                f"y={corr_a:.4f}x+{corr_b:.4f}",
                            ]
                            sl_ci_lo = corr_fi.get("slope_ci_low", np.nan)
                            sl_ci_hi = corr_fi.get("slope_ci_high", np.nan)
                            ic_ci_lo = corr_fi.get("intercept_ci_low", np.nan)
                            ic_ci_hi = corr_fi.get("intercept_ci_high", np.nan)
                            if np.isfinite(sl_ci_lo) and np.isfinite(sl_ci_hi):
                                stat_lines.append(f"slope 95%CI [{sl_ci_lo:.4f}, {sl_ci_hi:.4f}]")
                            if np.isfinite(ic_ci_lo) and np.isfinite(ic_ci_hi):
                                stat_lines.append(f"intercept 95%CI [{ic_ci_lo:.4f}, {ic_ci_hi:.4f}]")

                            ax_corr.text(0.03, 0.97, "\n".join(stat_lines), transform=ax_corr.transAxes,
                                         va="top", fontsize=9,
                                         bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))

                            ax_corr.set_xlabel(f"{base_pat_name} 再計算濃度{unit}")
                            ax_corr.set_ylabel(f"{cmp_pat_name} 再計算濃度{unit}")
                            ax_corr.set_title(f"相関プロット: {base_pat_name} vs {cmp_pat_name}")
                            ax_corr.set_aspect("equal", adjustable="datalim")
                            ax_corr.grid(True, alpha=0.25)
                            ax_corr.legend(fontsize=8, loc="lower right")
                            st.pyplot(fig_corr)
                            plt.close(fig_corr)

                    st.divider()

                    # ============================================================
                    # Step 4: ロット差検討 (異なるCalロット間での検量線・実検体比較)
                    # ============================================================
                    st.subheader("Step 4: ロット差検討（異なるCalロット間での実検体・コントロール比較）")
                    st.markdown("""
                    保存された別のCalロット設定ファイル（JSON）を選択し、同一の測光区間を用いて検量線・実検体濃度のロット間測定値差を比較します。
                    """)

                    parsed_dir = st.session_state.get("parsed_dir")
                    available_lot_jsons = []
                    if parsed_dir:
                        available_lot_jsons = sorted(list(Path(parsed_dir).glob(f"cal_config_{tc_item_tab4}_*.json")))
                        if not available_lot_jsons:
                            available_lot_jsons = sorted(list(Path(parsed_dir).glob("cal_config_*.json")))

                    if len(available_lot_jsons) < 1 and not st.session_state.get("cal_config"):
                        st.info("※ ロット差検討を行うには、Step 1でCal設定を保存して、少なくとも1つ以上のCal設定JSONを作成してください。")
                    else:
                        col_lota, col_lotb = st.columns(2)
                        with col_lota:
                            st.markdown(f"**基準ロット (Lot-A)**: `{cal_cfg['lot_name']}` (現在の設定)")
                        with col_lotb:
                            lot_options = {p.name: p for p in available_lot_jsons}
                            selected_compare_json = st.selectbox(
                                "比較対照ロット (Lot-B) の設定JSONを選択",
                                options=list(lot_options.keys()),
                                key="sel_compare_lot_json"
                            )

                        # 比較に用いる測光区間 (パターン1を使用)
                        base_pat = patterns[0]
                        st.caption(f"※ 比較に使用する測光区間: `{base_pat['name']}` ({base_pat['time_start']:.1f}s ~ {base_pat['time_end']:.1f}s, {base_pat['curve_mode']})")

                        if st.button("🔬 ロット差を比較解析", type="primary", key="btn_compare_lots"):
                            with st.spinner("ロット比較解析中..."):
                                compare_cfg = load_cal_config(lot_options[selected_compare_json])
                                lot_b_name = compare_cfg.get("lot_name", "Lot-B")

                                # Lot A (Current) の検量線 & 再計算結果 (パターン1)
                                res_a = results[0]
                                curve_a = res_a["curve"]
                                recalc_a = res_a["recalc_df"]

                                # Lot B の検量線 & 再計算結果
                                rates_b = calc_rates_batch(profile_df, tc_item_tab4, base_pat["time_start"], base_pat["time_end"])
                                b_level_table = [[cid for cid in lv.get("ids", [])] for lv in compare_cfg.get("levels", [])]
                                b_concs = compare_cfg.get("concentrations", [])
                                b_agg_rates = aggregate_cal_rates(rates_b, b_level_table, method=compare_cfg.get("aggregation", "median"))
                                curve_b = build_calibration_curve(b_agg_rates, b_concs, curve_mode=base_pat["curve_mode"])

                                cal_b_map = {}
                                for lv_i, cids in enumerate(b_level_table):
                                    for cid in cids:
                                        if cid and lv_i < len(b_concs):
                                            cal_b_map[cid] = b_concs[lv_i]

                                recalc_b = recalculate_all_samples(
                                    profile_df, measurement_df, tc_item_tab4,
                                    curve_b, base_pat["time_start"], base_pat["time_end"],
                                    cal_id_to_conc=cal_b_map
                                )

                                # マージ & 差分算出
                                lot_cmp_df = compare_two_recalc_results(recalc_a, recalc_b, label_a=cal_cfg['lot_name'], label_b=lot_b_name)
                                st.session_state["lot_comparison_data"] = {
                                    "lot_a_name": cal_cfg['lot_name'],
                                    "lot_b_name": lot_b_name,
                                    "curve_a": curve_a,
                                    "curve_b": curve_b,
                                    "b_concs": b_concs,
                                    "b_agg_rates": b_agg_rates,
                                    "lot_cmp_df": lot_cmp_df,
                                }
                                st.success(f"✅ ロット比較完了: {cal_cfg['lot_name']} vs {lot_b_name}")

                        # ロット比較結果の表示
                        if st.session_state.get("lot_comparison_data"):
                            lcd = st.session_state["lot_comparison_data"]
                            la_name = lcd["lot_a_name"]
                            lb_name = lcd["lot_b_name"]
                            cmp_df = lcd["lot_cmp_df"]

                            st.markdown(f"#### ① 検量線の重ね描き比較 ({la_name} vs {lb_name})")
                            fig_lot, ax_lot = plt.subplots(figsize=(10, 6))

                            # Lot A
                            c_a = lcd["curve_a"]
                            if c_a:
                                ax_lot.scatter(c_a["concentrations"], c_a["rates"], color="blue", marker="o", s=60, label=f"{la_name} Cal点")
                                ax_lot.plot(c_a["concentrations"], c_a["rates"], color="blue", lw=1.8, label=f"{la_name} 検量線")

                            # Lot B
                            c_b = lcd["curve_b"]
                            if c_b:
                                ax_lot.scatter(c_b["concentrations"], c_b["rates"], color="red", marker="^", s=60, label=f"{lb_name} Cal点")
                                ax_lot.plot(c_b["concentrations"], c_b["rates"], color="red", lw=1.8, linestyle="--", label=f"{lb_name} 検量線")

                            ax_lot.set_xlabel(f"濃度{unit}")
                            ax_lot.set_ylabel("処理値 (mAbs/min)")
                            ax_lot.set_title(f"検量線ロット間比較: {la_name} vs {lb_name}")
                            ax_lot.grid(True, alpha=0.3)
                            ax_lot.legend(fontsize=8, loc="best")
                            st.pyplot(fig_lot)
                            plt.close(fig_lot)

                            # 相関 & Bland-Altman
                            st.markdown(f"#### ② 実検体・コントロール測定値のロット間相関 & Bland-Altman")
                            sample_cmp = cmp_df[cmp_df["サンプル区分"] == "一般検体"].dropna(subset=[f"濃度_{la_name}", f"濃度_{lb_name}"])

                            if len(sample_cmp) >= 2:
                                x_lot = sample_cmp[f"濃度_{la_name}"].values.astype(float)
                                y_lot = sample_cmp[f"濃度_{lb_name}"].values.astype(float)

                                col_c1, col_c2 = st.columns(2)

                                with col_c1:
                                    # 相関プロット
                                    r_lot = pearson_r(x_lot, y_lot)
                                    a_lot, b_lot, fi_lot = regression_fit_info(x_lot, y_lot, method="PassingBablok")

                                    fig_lc, ax_lc = plt.subplots(figsize=(7, 7))
                                    ax_lc.scatter(x_lot, y_lot, color="purple", s=35, alpha=0.75, zorder=5)

                                    lo_l = min(float(np.nanmin(x_lot)), float(np.nanmin(y_lot)))
                                    hi_l = max(float(np.nanmax(x_lot)), float(np.nanmax(y_lot)))
                                    mar_l = (hi_l - lo_l) * 0.05 if hi_l > lo_l else 1.0
                                    ax_lc.plot([lo_l - mar_l, hi_l + mar_l], [lo_l - mar_l, hi_l + mar_l], "--", lw=1, color="gray", label="y=x")

                                    if np.isfinite(a_lot) and np.isfinite(b_lot):
                                        xx_l = np.array([lo_l - mar_l, hi_l + mar_l])
                                        ax_lc.plot(xx_l, a_lot * xx_l + b_lot, lw=1.8, color="darkorange", label="Passing-Bablok")

                                    stat_t = f"n={len(sample_cmp)}\nr={r_lot:.4f}\ny={a_lot:.4f}x+{b_lot:.4f}"
                                    ax_lc.text(0.03, 0.97, stat_t, transform=ax_lc.transAxes, va="top", fontsize=9,
                                               bbox=dict(boxstyle="round", facecolor="white", alpha=0.75))
                                    ax_lc.set_xlabel(f"{la_name} 濃度{unit}")
                                    ax_lc.set_ylabel(f"{lb_name} 濃度{unit}")
                                    ax_lc.set_title(f"相関: {la_name} vs {lb_name}")
                                    ax_lc.grid(True, alpha=0.25)
                                    ax_lc.legend(fontsize=8, loc="lower right")
                                    st.pyplot(fig_lc)
                                    plt.close(fig_lc)

                                with col_c2:
                                    # Bland-Altman プロット
                                    mean_lot = (x_lot + y_lot) / 2.0
                                    diff_lot = y_lot - x_lot
                                    bias_lot = float(np.nanmean(diff_lot))
                                    sd_lot = float(np.nanstd(diff_lot, ddof=1)) if len(diff_lot) > 1 else np.nan
                                    loa_hi_l = bias_lot + 1.96 * sd_lot if np.isfinite(sd_lot) else np.nan
                                    loa_lo_l = bias_lot - 1.96 * sd_lot if np.isfinite(sd_lot) else np.nan

                                    fig_ba, ax_ba = plt.subplots(figsize=(7, 7))
                                    ax_ba.scatter(mean_lot, diff_lot, color="teal", s=35, alpha=0.75, zorder=5)
                                    ax_ba.axhline(bias_lot, color="black", lw=1.5, label=f"平均差={bias_lot:.3f}")
                                    if np.isfinite(loa_hi_l) and np.isfinite(loa_lo_l):
                                        ax_ba.axhline(loa_hi_l, color="gray", lw=1.2, ls="--", label=f"+1.96SD={loa_hi_l:.3f}")
                                        ax_ba.axhline(loa_lo_l, color="gray", lw=1.2, ls="--", label=f"-1.96SD={loa_lo_l:.3f}")

                                    ax_ba.set_xlabel(f"平均濃度 (({la_name}+{lb_name})/2){unit}")
                                    ax_ba.set_ylabel(f"差 ({lb_name} - {la_name}){unit}")
                                    ax_ba.set_title(f"Bland–Altman: {lb_name} - {la_name}")
                                    ax_ba.grid(True, alpha=0.25)
                                    ax_ba.legend(fontsize=8, loc="best")
                                    st.pyplot(fig_ba)
                                    plt.close(fig_ba)

                            st.markdown("#### ③ ロット間濃度差テーブル")
                            st.dataframe(cmp_df, use_container_width=True)

                    st.divider()

                    # ============================================================
                    # Step 5: Excelレポート出力
                    # ============================================================
                    st.subheader("Step 5: Excelレポート出力")
                    st.markdown("再計算結果マトリクス、Calサマリー、Cal詳細、ロット差検討結果をまとめたExcelファイルを生成します。")

                    if st.button("📊 Excelレポートを生成", type="primary", key="btn_export_cal_excel"):
                        with st.spinner("Excelファイル作成中..."):
                            try:
                                parsed_dir = st.session_state.get("parsed_dir")
                                dirs = make_output_dirs(OUTPUT_ROOT, input_stem=f"CalAnalysis_{tc_item_tab4}_{cal_cfg['lot_name']}")
                                out_xlsx = dirs["excel"] / f"CalAnalysis_{tc_item_tab4}_{cal_cfg['lot_name']}.xlsx"

                                with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
                                    # Sheet 1: 再計算マトリクス
                                    if "recalc_matrix_df" in st.session_state:
                                        st.session_state["recalc_matrix_df"].to_excel(writer, sheet_name="再計算マトリクス", index=False)

                                    # Sheet 2: Calサマリー
                                    cal_sum_list = []
                                    for res in results:
                                        if "summary_df" in res:
                                            cal_sum_list.append(res["summary_df"].assign(パターン=res["pattern"]["name"]))
                                    if cal_sum_list:
                                        pd.concat(cal_sum_list, ignore_index=True).to_excel(writer, sheet_name="Calサマリー", index=False)

                                    # Sheet 3: Cal詳細 (Replicates)
                                    cal_det_list = []
                                    for res in results:
                                        if "detail_df" in res:
                                            cal_det_list.append(res["detail_df"].assign(パターン=res["pattern"]["name"]))
                                    if cal_det_list:
                                        pd.concat(cal_det_list, ignore_index=True).to_excel(writer, sheet_name="Cal個別測定値", index=False)

                                    # Sheet 4: ロット比較 (あれば)
                                    if st.session_state.get("lot_comparison_data"):
                                        st.session_state["lot_comparison_data"]["lot_cmp_df"].to_excel(writer, sheet_name="ロット間比較", index=False)

                                st.success(f"✅ Excel保存完了: `{out_xlsx}`")

                                # ダウンロードボタン
                                with open(out_xlsx, "rb") as f_x:
                                    st.download_button(
                                        label="📥 Excelファイルをダウンロード",
                                        data=f_x.read(),
                                        file_name=out_xlsx.name,
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key="dl_cal_excel"
                                    )
                            except Exception as e:
                                st.error(f"Excel出力エラー: {e}")

