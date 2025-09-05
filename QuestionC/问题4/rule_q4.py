import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
)


SEED = 42


def load_data(base_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(base_dir, "female_q4_dataset.csv"))
    return df


def apply_qc_flags(df: pd.DataFrame) -> pd.Series:
    """放宽质控：缺失不否决、阈值更宽，仅作为建议复检标记，不用于剔除样本。"""
    gc13 = pd.to_numeric(df.get("gc13"), errors="coerce")
    gc18 = pd.to_numeric(df.get("gc18"), errors="coerce")
    gc21 = pd.to_numeric(df.get("gc21"), errors="coerce")
    good_gc = sum([
        gc13.between(0.38, 0.62, inclusive="both"),
        gc18.between(0.38, 0.62, inclusive="both"),
        gc21.between(0.38, 0.62, inclusive="both"),
    ])
    # 至少两条GC在范围内，或三者都缺失（不否决）
    all_nan_gc = gc13.isna() & gc18.isna() & gc21.isna()
    ok_gc = (good_gc >= 2) | all_nan_gc

    reads_total = pd.to_numeric(df.get("reads_total"), errors="coerce")
    align_ratio = pd.to_numeric(df.get("align_ratio"), errors="coerce")
    duplicate_ratio = pd.to_numeric(df.get("duplicate_ratio"), errors="coerce")

    reads_ok = reads_total.isna() | (reads_total > 2e5)
    align_ok = align_ratio.isna() | align_ratio.between(0.35, 0.999, inclusive="both")
    dup_ok = duplicate_ratio.isna() | (duplicate_ratio <= 0.95)

    ok = ok_gc & reads_ok & align_ok & dup_ok
    return ok.fillna(True)


def build_z_features(df: pd.DataFrame) -> pd.DataFrame:
    zq = pd.to_numeric(df.get("q_z13"), errors="coerce")
    zr = pd.to_numeric(df.get("r_z18"), errors="coerce")
    zs = pd.to_numeric(df.get("s_z21"), errors="coerce")
    t = pd.to_numeric(df.get("t_zx"), errors="coerce")
    bmi = pd.to_numeric(df.get("bmi"), errors="coerce")
    gc21 = pd.to_numeric(df.get("gc21"), errors="coerce")

    z_stack = np.vstack([zq, zr, zs])
    zmax = np.nanmax(z_stack, axis=0)
    zabsmax = np.nanmax(np.abs(z_stack), axis=0)
    zmean = np.nanmean(z_stack, axis=0)
    zstd = np.nanstd(z_stack, axis=0)

    # 安全除法：bmi<=0 或 NaN 时结果设为NaN
    def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
        out = a.copy()
        denom = b.copy()
        mask = denom <= 0
        denom[mask] = np.nan
        return out / denom

    feats = pd.DataFrame({
        # 原始Z
        "z13": zq, "z18": zr, "z21": zs, "zx": t,
        # 聚合
        "zmax": zmax,
        "zabsmax": zabsmax,
        "zmean": zmean,
        "zstd": zstd,
        "zsum": zq + zr + zs,
        # 布尔阈值（2.5 与 3）
        "z21_gt25": (zs > 2.5).astype(int),
        "z18_gt25": (zr > 2.5).astype(int),
        "z13_gt25": (zq > 2.5).astype(int),
        "z21_gt3": (zs > 3).astype(int),
        "z18_gt3": (zr > 3).astype(int),
        "z13_gt3": (zq > 3).astype(int),
        "zmax_gt3": (zmax > 3).astype(int),
        # 绝对值与平方
        "z21_abs": np.abs(zs),
        "z18_abs": np.abs(zr),
        "z13_abs": np.abs(zq),
        "z21_sq": zs * zs,
        "z18_sq": zr * zr,
        "z13_sq": zq * zq,
        # 交互/比值
        "z21_bmi": zs * bmi,
        "zsum_bmi": (zq + zr + zs) * bmi,
        "z21_over_bmi": safe_div(zs, bmi),
        "z21_gc21": zs * gc21,
    }, index=df.index)
    return feats


def rule_decision(df: pd.DataFrame, qc_ok: pd.Series) -> pd.Series:
    feats = build_z_features(df)
    zmax = feats["zmax"]
    # 规则（放宽，不剔除，仅分档）：
    # 1) maxZ >= 3 → 高风险异常
    # 2) 2.6<=maxZ<3 且 质控OK 且 孕周>=12 → 可疑（计为1用于高召回筛查）
    gest = pd.to_numeric(df.get("gestational_week"), errors="coerce")
    pred = pd.Series(0, index=df.index)
    pred[zmax >= 3] = 1
    pred[(zmax >= 2.6) & (zmax < 3) & qc_ok & (gest >= 12.0)] = 1
    return pred


def calibrate_probability(df: pd.DataFrame) -> Pipeline:
    feats = build_z_features(df)
    y = df["is_abnormal"].astype(int)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000, C=1.0)),
    ])
    clf = CalibratedClassifierCV(pipe, method="isotonic", cv=5)
    clf.fit(feats, y)
    return clf


def evaluate(rule_pred: pd.Series, proba: np.ndarray, y: pd.Series, out_md: str, qc_ok: pd.Series, base_dir: str):
    # 使用全体样本评估概率模型；同时报告规则法与分层(qc_ok子集)结果
    auc = roc_auc_score(y, proba)
    ap = average_precision_score(y, proba)
    p_curve, r_curve, thr = precision_recall_curve(y, proba)
    # F1最优
    f1s = (2 * p_curve[1:] * r_curve[1:]) / (p_curve[1:] + r_curve[1:] + 1e-12)
    best_idx = int(np.nanargmax(f1s)) if len(thr) > 0 else 0
    thr_f1 = float(thr[best_idx]) if len(thr) > 0 else 0.5
    # 固定阈值（用户指定）
    thr_fixed = 0.1086
    # 在PR曲线中选择在Recall>=target时Precision最高的阈值（避免退化）
    def _select_for_recall(target: float):
        if len(thr) == 0:
            return None
        best = None  # (precision, threshold, recall)
        for p, r, t in zip(p_curve[1:], r_curve[1:], thr):
            if r >= target:
                if best is None or p > best[0]:
                    best = (float(p), float(t), float(r))
        return best
    sel_r70 = _select_for_recall(0.7)
    sel_r80 = _select_for_recall(0.8)
    sel_r90 = _select_for_recall(0.9)
    # Top-k 策略（按概率排序取前5%/10%判阳）
    order = np.argsort(proba)[::-1]
    def _topk_threshold(frac: float) -> float:
        k = max(1, int(len(proba) * frac))
        return float(proba[order[k-1]])
    thr_top5 = _topk_threshold(0.05)
    thr_top10 = _topk_threshold(0.10)

    def _report_at(th: float) -> tuple[pd.DataFrame, str]:
        yp = (proba >= th).astype(int)
        cm = confusion_matrix(y, yp)
        rep = classification_report(y, yp, zero_division=0)
        return pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]), rep

    cm_f1, rep_f1 = _report_at(thr_f1)
    cm_fix, rep_fix = _report_at(thr_fixed)
    cm_top5, rep_top5 = _report_at(thr_top5)
    cm_top10, rep_top10 = _report_at(thr_top10)

    # 规则直判
    cm_rule = confusion_matrix(y, rule_pred)
    rep_rule = classification_report(y, rule_pred, zero_division=0)

    # 分层：仅qc_ok子集
    sub = qc_ok.fillna(True)
    y_sub = y[sub]
    proba_sub = proba[sub]
    auc_sub = roc_auc_score(y_sub, proba_sub) if len(np.unique(y_sub)) > 1 else np.nan
    ap_sub = average_precision_score(y_sub, proba_sub) if len(np.unique(y_sub)) > 1 else np.nan

    # 绘制ROC/PR
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve(y, proba)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0,1],[0,1],'k--',alpha=0.5)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (Z-only Calibrated)"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "q4_rule_roc.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(5,4))
        plt.plot(r_curve, p_curve, label=f"AP={ap:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (Z-only Calibrated)"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "q4_rule_pr.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    # 错误样本分布图（FN/FP箱线图）：Zmax、BMI、reads_total 对比（基于F1最优阈值）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns  # type: ignore
        y_pred_f1 = (proba >= thr_f1).astype(int)
        feats_full = build_z_features(df)
        zmax_full = feats_full["zmax"]
        dfp = pd.DataFrame({
            "err_type": np.select([
                (y == 1) & (y_pred_f1 == 0),
                (y == 0) & (y_pred_f1 == 1),
            ], ["FN", "FP"], default="TN/TP"),
            "zmax": zmax_full,
            "bmi": pd.to_numeric(df.get("bmi"), errors="coerce"),
            "reads_total": pd.to_numeric(df.get("reads_total"), errors="coerce"),
        })
        for col, fname, title in [("zmax","q4_err_box_zmax.png","Zmax distribution in FN vs FP"),
                                   ("bmi","q4_err_box_bmi.png","BMI in FN vs FP"),
                                   ("reads_total","q4_err_box_reads.png","Reads total in FN vs FP")]:
            plt.figure(figsize=(8,4))
            sns.boxplot(data=dfp[dfp["err_type"].isin(["FN","FP"])], x="err_type", y=col)
            plt.title(title)
            plt.tight_layout(); plt.savefig(os.path.join(base_dir, fname), dpi=150); plt.close()
    except Exception:
        pass

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Q4 规则+校准概率判定报告\n\n")
        f.write("## 概率模型（仅Z系+衍生，Isotonic校准）\n")
        f.write(f"AUC: {auc:.4f}, AP: {ap:.4f}\n\n")
        f.write(f"F1最优阈值: {thr_f1:.4f}\n\n")
        f.write(cm_f1.to_string()); f.write("\n\n"); f.write(rep_f1); f.write("\n\n")
        f.write(f"固定阈值(p=0.1086)\n\n"); f.write(cm_fix.to_string()); f.write("\n\n"); f.write(rep_fix); f.write("\n\n")
        # 高召回模式（PR曲线选点）
        f.write("### 高召回模式（PR曲线选点）\n\n")
        def _dump(sel, label):
            if sel is None:
                f.write(f"{label}: 当前模型在非退化阈值下难以达到目标召回\n\n")
                return
            prec, th, rec = sel
            yp = (proba >= th).astype(int)
            cm = confusion_matrix(y, yp)
            rep = classification_report(y, yp, zero_division=0)
            f.write(f"{label}: 阈值={th:.4f}, Precision≈{prec:.4f}, Recall≈{rec:.4f}\n\n")
            f.write(pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_string())
            f.write("\n\n"); f.write(rep); f.write("\n\n")
        _dump(sel_r70, "Recall≈0.70")
        _dump(sel_r80, "Recall≈0.80")
        _dump(sel_r90, "Recall≈0.90")

        # Top-K 模式
        f.write("### Top-K 模式\n\n")
        f.write(f"Top-5% 阈值≈{thr_top5:.4f}\n\n"); f.write(cm_top5.to_string()); f.write("\n\n"); f.write(rep_top5); f.write("\n\n")
        f.write(f"Top-10% 阈值≈{thr_top10:.4f}\n\n"); f.write(cm_top10.to_string()); f.write("\n\n"); f.write(rep_top10); f.write("\n\n")

        # 模式总览
        f.write("## 阈值分模式建议\n\n")
        f.write("- 高召回模式：采用PR曲线选点（Recall≈0.7/0.8/0.9），用于疾病初筛，需人工复核。\n")
        f.write("- 平衡模式：F1最优阈值（本次≈{:.4f}）。\n".format(thr_f1))
        f.write("- 高精度模式：可在PR曲线中寻找Precision≥0.30的最大Recall点；若当前不存在，则提示受限。\n\n")

        f.write("## 规则直判（Z阈值+孕周、质控标记不剔除）\n")
        f.write(pd.DataFrame(cm_rule, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_string()); f.write("\n\n")
        f.write(rep_rule); f.write("\n\n")

        f.write("## 质控统计（不剔除，仅提示）\n")
        f.write(f"通过质控: {int(qc_ok.sum())}, 未通过: {int((~qc_ok).sum())}\n\n")
        f.write("若需更严格评估，可仅在质控通过子集上统计：\n")
        f.write(f"AUC_sub: {auc_sub}, AP_sub: {ap_sub}\n")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = load_data(base_dir)
    y = df["is_abnormal"].astype(int)
    qc_ok = apply_qc_flags(df)

    # 规则判定
    rule_pred = rule_decision(df, qc_ok)

    # 概率校准（仅Z特征）
    clf = calibrate_probability(df)
    proba = clf.predict_proba(build_z_features(df))[:, 1]

    out_md = os.path.join(base_dir, "q4_rule_report.md")
    evaluate(rule_pred, proba, y, out_md, qc_ok, base_dir)

    # 追加：z_max 与分染色体门限扫描
    try:
        with open(out_md, "a", encoding="utf-8") as f:
            f.write("\n\n## 阈值扫描：z_max 与 分染色体\n\n")

            # z_max 扫描
            feats = build_z_features(df)
            zmax = feats["zmax"].fillna(-1e9).values
            grid = np.round(np.arange(1.5, 3.21, 0.05), 2)
            def _metrics_at_th(scores: np.ndarray, thr: float) -> tuple[float, float, float, np.ndarray]:
                yp = (scores >= thr).astype(int)
                cm = confusion_matrix(y, yp)
                tn, fp, fn, tp = cm.ravel()
                prec = tp / (tp + fp + 1e-12)
                rec = tp / (tp + fn + 1e-12)
                f1 = 2 * prec * rec / (prec + rec + 1e-12)
                return prec, rec, f1, cm
            def _find_thr_for_recall(target: float) -> tuple[float, float, float, float, np.ndarray]:
                chosen = None
                chosen_vals = (0.0, 0.0, 0.0, np.zeros((2,2), dtype=int))
                for th in grid:
                    prec, rec, f1, cm = _metrics_at_th(zmax, th)
                    if rec >= target:
                        chosen = th
                        chosen_vals = (prec, rec, f1, cm)
                        break
                if chosen is None:
                    # 取最小阈值
                    th = grid[0]
                    prec, rec, f1, cm = _metrics_at_th(zmax, th)
                    chosen = th
                    chosen_vals = (prec, rec, f1, cm)
                return chosen, *chosen_vals

            for tgt in [0.7, 0.8]:
                th, prec, rec, f1, cm = _find_thr_for_recall(tgt)
                f.write(f"z_max: 达到Recall≥{tgt:.1f}的最小阈值 τ={th:.2f}\n\n")
                f.write(pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_string())
                f.write(f"\nPrecision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}\n\n")

            # 分染色体一对多：独立找阈值并合并预测
            f.write("### 分染色体一对多阈值（合并规则：任一达标即判阳）\n\n")
            ab = df.get("ab_type").astype(str).str.upper().fillna("")
            z13 = pd.to_numeric(df.get("q_z13"), errors="coerce").fillna(-1e9).values
            z18 = pd.to_numeric(df.get("r_z18"), errors="coerce").fillna(-1e9).values
            z21 = pd.to_numeric(df.get("s_z21"), errors="coerce").fillna(-1e9).values
            def _thr_for_class(scores: np.ndarray, mask_pos: np.ndarray, target: float) -> float:
                # 在该类正样本上达成目标召回的最小阈值
                pos_idx = np.where(mask_pos)[0]
                if len(pos_idx) == 0:
                    return 3.0
                for th in grid:
                    yp = (scores[pos_idx] >= th).astype(int)
                    rec = yp.mean() if len(yp) > 0 else 0.0
                    if rec >= target:
                        return th
                return grid[-1]

            for tgt in [0.7, 0.8]:
                m13 = (ab == "T13").values
                m18 = (ab == "T18").values
                m21 = (ab == "T21").values
                t13 = _thr_for_class(z13, m13, tgt)
                t18 = _thr_for_class(z18, m18, tgt)
                t21 = _thr_for_class(z21, m21, tgt)
                y_pred = ((z13 >= t13) | (z18 >= t18) | (z21 >= t21)).astype(int)
                cm = confusion_matrix(y, y_pred)
                tn, fp, fn, tp = cm.ravel()
                prec = tp / (tp + fp + 1e-12)
                rec = tp / (tp + fn + 1e-12)
                f1 = 2 * prec * rec / (prec + rec + 1e-12)
                f.write(f"目标Recall≥{tgt:.1f}: τ13={t13:.2f}, τ18={t18:.2f}, τ21={t21:.2f}\n\n")
                f.write(pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_string())
                f.write(f"\nPrecision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}\n\n")
    except Exception:
        pass
    print(f"Saved report: {out_md}")


if __name__ == "__main__":
    main()


