import os
import sys
from typing import List, Tuple, Optional, Dict

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score, roc_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Optional deps
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE  # type: ignore
    from imblearn.under_sampling import RandomUnderSampler  # type: ignore
    from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier  # type: ignore
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False


SEED = 42


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_feature_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    # 衍生特征
    df = df.copy()
    for c in ["q_z13", "r_z18", "s_z21", "t_zx"]:
        if c not in df.columns:
            df[c] = np.nan
    # 派生：Z相关强特征
    df["z_max"] = df[["q_z13", "r_z18", "s_z21"]].max(axis=1)
    df["z_sum"] = df[["q_z13", "r_z18", "s_z21"]].sum(axis=1)
    df["z_abs_max"] = df[["q_z13", "r_z18", "s_z21"]].abs().max(axis=1)
    df["z_pos_count"] = ((df[["q_z13", "r_z18", "s_z21"]] > 0).sum(axis=1)).astype(float)
    # 阈值布尔特征与交互
    df["z21_gt3"] = (df["s_z21"] > 3).astype(int)
    df["z18_gt3"] = (df["r_z18"] > 3).astype(int)
    df["z13_gt3"] = (df["q_z13"] > 3).astype(int)
    df["zmax_gt3"] = (df["z_max"] > 3).astype(int)
    df["z21_bmi"] = df["s_z21"] * df.get("bmi", 0)
    df["zsum_bmi"] = df["z_sum"] * df.get("bmi", 0)

    # 使用问题描述中的特征集 + 衍生特征
    feature_cols = [
        "q_z13", "r_z18", "s_z21", "t_zx",  # Z值
        "gc13", "gc18", "gc21",               # GC 含量
        "reads_total", "align_ratio", "duplicate_ratio", "unique_reads", "filtered_ratio",  # 读段及比率
        "bmi", "maternal_age", "gestational_week", "ivf_flag",  # 生理与控制变量
        # 衍生
        "z_max", "z_sum", "z_abs_max", "z_pos_count",
        "z21_gt3", "z18_gt3", "z13_gt3", "zmax_gt3",
        "z21_bmi", "zsum_bmi",
    ]
    # 可能存在的缺列用占位补充
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    X = df[feature_cols].copy()
    y = df["is_abnormal"].astype(int)
    groups = df["subject_id"].astype(str)
    return X, y, groups


def get_z_only_columns() -> List[str]:
    return [
        "q_z13", "r_z18", "s_z21", "t_zx",
        "z_max", "z_sum", "z_abs_max", "z_pos_count",
        "z21_gt3", "z18_gt3", "z13_gt3", "zmax_gt3",
        "z21_bmi", "zsum_bmi",
    ]


def baseline_rule(df: pd.DataFrame) -> np.ndarray:
    # 规则：任一 Q/R/S > 3 判为 1
    q = pd.to_numeric(df.get("q_z13"), errors="coerce")
    r = pd.to_numeric(df.get("r_z18"), errors="coerce")
    s = pd.to_numeric(df.get("s_z21"), errors="coerce")
    return ((q > 3) | (r > 3) | (s > 3)).astype(int).values


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "female_q4_dataset.csv")
    if not os.path.exists(data_path):
        print(f"未找到数据集: {data_path}")
        sys.exit(1)

    df = load_dataset(data_path)
    X, y, groups = build_feature_target(df)

    # 基线规则评估（使用全量y做参考，不作为模型）
    base_pred = baseline_rule(df)
    try:
        base_auc = roc_auc_score(y, base_pred)
    except Exception:
        base_auc = float("nan")

    # 按孕妇分组划分，避免同一孕妇泄漏
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    def eval_with_thresholds(y_true: pd.Series, y_scores: np.ndarray) -> Dict[str, object]:
        # 默认
        auc_val = roc_auc_score(y_true, y_scores)
        ap_val = average_precision_score(y_true, y_scores)
        pred_default = (y_scores >= 0.5).astype(int)
        cm_default = confusion_matrix(y_true, pred_default)
        rep_default = classification_report(y_true, pred_default, zero_division=0)

        # 最佳F1阈值
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        if len(thresholds) > 0:
            f1s = (2 * precisions[1:] * recalls[1:]) / (precisions[1:] + recalls[1:] + 1e-12)
            best_idx = int(np.nanargmax(f1s))
            thr_f1 = float(thresholds[best_idx])
        else:
            thr_f1 = 0.5
        pred_f1 = (y_scores >= thr_f1).astype(int)
        cm_f1 = confusion_matrix(y_true, pred_f1)
        rep_f1 = classification_report(y_true, pred_f1, zero_division=0)
        prec_f1 = precision_score(y_true, pred_f1, zero_division=0)
        rec_f1 = recall_score(y_true, pred_f1, zero_division=0)
        f1_f1 = f1_score(y_true, pred_f1, zero_division=0)

        # 精度约束下最高召回（如 precision >= 0.30）
        target_p = 0.30
        thr_pc = 0.5
        best_rec = -1.0
        for p, r, t in zip(precisions[1:], recalls[1:], thresholds):
            if p >= target_p and r > best_rec:
                best_rec = r
                thr_pc = float(t)
        pred_pc = (y_scores >= thr_pc).astype(int)
        cm_pc = confusion_matrix(y_true, pred_pc)
        rep_pc = classification_report(y_true, pred_pc, zero_division=0)
        prec_pc = precision_score(y_true, pred_pc, zero_division=0)
        rec_pc = recall_score(y_true, pred_pc, zero_division=0)
        f1_pc = f1_score(y_true, pred_pc, zero_division=0)

        # 召回≥0.9 的最小阈值（若可行）
        thr_r90 = 0.0
        found_r90 = False
        for p, r, t in zip(precisions[1:], recalls[1:], thresholds):
            if r >= 0.9:
                thr_r90 = float(t)
                found_r90 = True
                break
        pred_r90 = (y_scores >= thr_r90).astype(int) if found_r90 else pred_f1
        cm_r90 = confusion_matrix(y_true, pred_r90)
        rep_r90 = classification_report(y_true, pred_r90, zero_division=0)
        prec_r90 = precision_score(y_true, pred_r90, zero_division=0)
        rec_r90 = recall_score(y_true, pred_r90, zero_division=0)
        f1_r90 = f1_score(y_true, pred_r90, zero_division=0)

        return {
            "auc": auc_val,
            "ap": ap_val,
            "default": {"thr": 0.5, "cm": cm_default, "report": rep_default},
            "best_f1": {"thr": thr_f1, "cm": cm_f1, "report": rep_f1, "prec": prec_f1, "rec": rec_f1, "f1": f1_f1},
            "prec_constrained": {"thr": thr_pc, "cm": cm_pc, "report": rep_pc, "prec": prec_pc, "rec": rec_pc, "f1": f1_pc},
            "recall90": {"thr": thr_r90, "found": found_r90, "cm": cm_r90, "report": rep_r90, "prec": prec_r90, "rec": rec_r90, "f1": f1_r90},
        }

    # 1) 随机森林 baseline（类权重平衡）
    rf_clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=500,
            random_state=SEED,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])
    rf_clf.fit(X_train, y_train)
    rf_prob = rf_clf.predict_proba(X_test)[:, 1]
    rf_eval = eval_with_thresholds(y_test, rf_prob)

    # 决策树（加权、浅层降低过拟合）
    dt_clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("dt", DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=SEED)),
    ])
    dt_clf.fit(X_train, y_train)
    dt_prob = dt_clf.predict_proba(X_test)[:, 1]
    dt_eval = eval_with_thresholds(y_test, dt_prob)

    # 2) 随机森林 + SMOTE（若可用）/ BalancedRF / EasyEnsemble
    smote_eval: Optional[Dict[str, object]] = None
    brf_eval: Optional[Dict[str, object]] = None
    eec_eval: Optional[Dict[str, object]] = None
    if HAS_SMOTE:
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        sm = SMOTE(random_state=SEED)
        X_res, y_res = sm.fit_resample(X_train_imp, y_train)
        rf_sm = RandomForestClassifier(
            n_estimators=600,
            random_state=SEED,
            class_weight=None,
            n_jobs=-1,
        )
        rf_sm.fit(X_res, y_res)
        prob_sm = rf_sm.predict_proba(X_test_imp)[:, 1]
        smote_eval = eval_with_thresholds(y_test, prob_sm)

        # Balanced Random Forest（每棵树对多数类下采样）
        try:
            brf = BalancedRandomForestClassifier(n_estimators=600, random_state=SEED, sampling_strategy="auto")
            brf.fit(X_train_imp, y_train)
            prob_brf = brf.predict_proba(X_test_imp)[:, 1]
            brf_eval = eval_with_thresholds(y_test, prob_brf)
        except Exception:
            brf_eval = None

        # EasyEnsemble（多次下采样训练AdaBoost集成）
        try:
            eec = EasyEnsembleClassifier(n_estimators=10, random_state=SEED)
            eec.fit(X_train_imp, y_train)
            prob_eec = eec.predict_proba(X_test_imp)[:, 1]
            eec_eval = eval_with_thresholds(y_test, prob_eec)
        except Exception:
            eec_eval = None

    # 3) 逻辑回归（L2, 平衡权重）
    lr_clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(solver="liblinear", C=1.0, class_weight="balanced", max_iter=1000)),
    ])
    lr_clf.fit(X_train, y_train)
    lr_prob = lr_clf.predict_proba(X_test)[:, 1]
    lr_eval = eval_with_thresholds(y_test, lr_prob)

    # 3b) 逻辑回归（Z-only）
    z_cols = [c for c in get_z_only_columns() if c in X.columns]
    X_train_z = X_train[z_cols].copy()
    X_test_z = X_test[z_cols].copy()
    lr_z = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(solver="liblinear", C=1.0, class_weight="balanced", max_iter=1000)),
    ])
    lr_z.fit(X_train_z, y_train)
    lr_z_prob = lr_z.predict_proba(X_test_z)[:, 1]
    lr_z_eval = eval_with_thresholds(y_test, lr_z_prob)

    # 4) XGBoost（若可用）
    xgb_eval: Optional[Dict[str, object]] = None
    xgb_importances: Optional[pd.Series] = None
    if HAS_XGB:
        # scale_pos_weight = neg/pos
        pos = max(1, int((y_train == 1).sum()))
        neg = max(1, int((y_train == 0).sum()))
        spw = float(neg / pos)
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)
        xgb = XGBClassifier(
            n_estimators=600,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            learning_rate=0.05,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
            scale_pos_weight=spw,
        )
        xgb.fit(X_train_imp, y_train)
        prob_xgb = xgb.predict_proba(X_test_imp)[:, 1]
        xgb_eval = eval_with_thresholds(y_test, prob_xgb)
        try:
            xgb_importances = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
        except Exception:
            xgb_importances = None

    # 特征重要性（RF baseline）
    rf: RandomForestClassifier = rf_clf.named_steps["rf"]
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    # 输出报告与曲线
    out_md = os.path.join(base_dir, "q4_model_report.md")
    roc_png = os.path.join(base_dir, "q4_roc.png")
    pr_png = os.path.join(base_dir, "q4_pr.png")
    # 以 RF 概率绘图（也可换为最优模型）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve(y_test, rf_prob)
        plt.figure(figsize=(5,4))
        plt.plot(fpr, tpr, label=f"RF AUC={rf_eval['auc']:.3f}")
        plt.plot([0,1],[0,1],'k--',alpha=0.5)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
        plt.savefig(roc_png, dpi=150)
        plt.close()

        precisions, recalls, _ = precision_recall_curve(y_test, rf_prob)
        plt.figure(figsize=(5,4))
        plt.plot(recalls, precisions, label=f"RF AP={rf_eval['ap']:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve"); plt.legend(); plt.tight_layout()
        plt.savefig(pr_png, dpi=150)
        plt.close()
    except Exception:
        roc_png = ""
        pr_png = ""
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# 问题4 女胎异常判定模型报告\n\n")
        f.write("## 数据集规模\n")
        f.write(f"训练集: {len(X_train)}, 测试集: {len(X_test)}\n\n")
        f.write("## 基线规则 (Q/R/S > 3)\n")
        f.write(f"AUC(全量参考): {base_auc:.4f}\n\n")
        f.write("## 模型对比\n\n")

        # RF baseline
        f.write("### 随机森林 Baseline (class_weight='balanced')\n")
        f.write(f"AUC: {rf_eval['auc']:.4f}, AP: {rf_eval['ap']:.4f}\n\n")
        f.write("- 默认阈值(0.5)\n\n")
        f.write(pd.DataFrame(rf_eval["default"]["cm"], index=["TN", "FP"], columns=["FN", "TP"]).to_string())
        f.write("\n\n")
        f.write(rf_eval["default"]["report"])  # type: ignore
        f.write("\n\n")
        bf = rf_eval["best_f1"]  # type: ignore
        f.write(f"- F1最优阈值: {bf['thr']:.4f}\n\n")
        f.write(pd.DataFrame(bf["cm"], index=["TN", "FP"], columns=["FN", "TP"]).to_string())  # type: ignore
        f.write("\n\n")
        f.write(bf["report"])  # type: ignore
        f.write(f"\nPrecision={bf['prec']:.4f}, Recall={bf['rec']:.4f}, F1={bf['f1']:.4f}\n\n")  # type: ignore
        pc = rf_eval["prec_constrained"]  # type: ignore
        f.write(f"- 精度≥0.30 最大召回 阈值: {pc['thr']:.4f}\n\n")
        r90 = rf_eval["recall90"]  # type: ignore
        if r90.get("found", False):  # type: ignore
            f.write(f"- 召回≥0.90 阈值: {r90['thr']:.4f}\n\n")
            f.write(pd.DataFrame(r90["cm"], index=["TN", "FP"], columns=["FN", "TP"]).to_string())  # type: ignore
            f.write("\n\n")
            f.write(r90["report"])  # type: ignore
            f.write(f"\nPrecision={r90['prec']:.4f}, Recall={r90['rec']:.4f}, F1={r90['f1']:.4f}\n\n")  # type: ignore
        else:
            f.write("- 召回≥0.90: 在当前模型概率下不可达\n\n")
        f.write(pd.DataFrame(pc["cm"], index=["TN", "FP"], columns=["FN", "TP"]).to_string())  # type: ignore
        f.write("\n\n")
        f.write(pc["report"])  # type: ignore
        f.write(f"\nPrecision={pc['prec']:.4f}, Recall={pc['rec']:.4f}, F1={pc['f1']:.4f}\n\n")  # type: ignore

        # 决策树
        f.write("### 决策树 (max_depth=4, class_weight='balanced')\n")
        f.write(f"AUC: {dt_eval['auc']:.4f}, AP: {dt_eval['ap']:.4f}\n\n")
        bf = dt_eval["best_f1"]  # type: ignore
        f.write(f"- F1最优阈值: {bf['thr']:.4f}\n\n")
        f.write(pd.DataFrame(bf["cm"], index=["TN", "FP"], columns=["FN", "TP"]).to_string())  # type: ignore
        f.write("\n\n")
        f.write(bf["report"])  # type: ignore
        f.write(f"\nPrecision={bf['prec']:.4f}, Recall={bf['rec']:.4f}, F1={bf['f1']:.4f}\n\n")  # type: ignore

        # RF + SMOTE
        if smote_eval is not None:
            f.write("### 随机森林 + SMOTE\n")
            f.write(f"AUC: {smote_eval['auc']:.4f}\n\n")
            bf = smote_eval["best_f1"]  # type: ignore
            f.write(f"- F1最优阈值: {bf['thr']:.4f}\n\n")
            f.write(pd.DataFrame(bf["cm"], index=["TN", "FP"], columns=["FN", "TP"]).to_string())  # type: ignore
            f.write("\n\n")
            f.write(bf["report"])  # type: ignore
            f.write(f"\nPrecision={bf['prec']:.4f}, Recall={bf['rec']:.4f}, F1={bf['f1']:.4f}\n\n")  # type: ignore

        if brf_eval is not None:
            f.write("### Balanced Random Forest\n")
            f.write(f"AUC: {brf_eval['auc']:.4f}\n\n")
            bf = brf_eval["best_f1"]  # type: ignore
            f.write(f"- F1最优阈值: {bf['thr']:.4f}\n\n")
            f.write(pd.DataFrame(bf["cm"], index=["TN", "FP"], columns=["FN", "TP"]).to_string())  # type: ignore
            f.write("\n\n")
            f.write(bf["report"])  # type: ignore
            f.write(f"\nPrecision={bf['prec']:.4f}, Recall={bf['rec']:.4f}, F1={bf['f1']:.4f}\n\n")  # type: ignore

        if eec_eval is not None:
            f.write("### EasyEnsembleClassifier\n")
            f.write(f"AUC: {eec_eval['auc']:.4f}\n\n")
            bf = eec_eval["best_f1"]  # type: ignore
            f.write(f"- F1最优阈值: {bf['thr']:.4f}\n\n")
            f.write(pd.DataFrame(bf["cm"], index=["TN", "FP"], columns=["FN", "TP"]).to_string())  # type: ignore
            f.write("\n\n")
            f.write(bf["report"])  # type: ignore
            f.write(f"\nPrecision={bf['prec']:.4f}, Recall={bf['rec']:.4f}, F1={bf['f1']:.4f}\n\n")  # type: ignore

        # Logistic
        f.write("### Logistic 回归 (L2, class_weight='balanced')\n")
        f.write(f"AUC: {lr_eval['auc']:.4f}, AP: {lr_eval['ap']:.4f}\n\n")
        bf = lr_eval["best_f1"]  # type: ignore
        f.write(f"- F1最优阈值: {bf['thr']:.4f}\n\n")
        f.write(pd.DataFrame(bf["cm"], index=["TN", "FP"], columns=["FN", "TP"]).to_string())  # type: ignore
        f.write("\n\n")
        f.write(bf["report"])  # type: ignore
        f.write(f"\nPrecision={bf['prec']:.4f}, Recall={bf['rec']:.4f}, F1={bf['f1']:.4f}\n\n")  # type: ignore

        f.write("### Logistic Z-only (L2, class_weight='balanced')\n")
        f.write(f"AUC: {lr_z_eval['auc']:.4f}, AP: {lr_z_eval['ap']:.4f}\n\n")
        bf = lr_z_eval["best_f1"]  # type: ignore
        f.write(f"- F1最优阈值: {bf['thr']:.4f}\n\n")
        f.write(pd.DataFrame(bf["cm"], index=["TN", "FP"], columns=["FN", "TP"]).to_string())  # type: ignore
        f.write("\n\n")
        f.write(bf["report"])  # type: ignore
        f.write(f"\nPrecision={bf['prec']:.4f}, Recall={bf['rec']:.4f}, F1={bf['f1']:.4f}\n\n")  # type: ignore

        # XGB
        if xgb_eval is not None:
            f.write("### XGBoost (scale_pos_weight)\n")
            f.write(f"AUC: {xgb_eval['auc']:.4f}\n\n")
            bf = xgb_eval["best_f1"]  # type: ignore
            f.write(f"- F1最优阈值: {bf['thr']:.4f}\n\n")
            f.write(pd.DataFrame(bf["cm"], index=["TN", "FP"], columns=["FN", "TP"]).to_string())  # type: ignore
            f.write("\n\n")
            f.write(bf["report"])  # type: ignore
            f.write(f"\nPrecision={bf['prec']:.4f}, Recall={bf['rec']:.4f}, F1={bf['f1']:.4f}\n\n")  # type: ignore

        if roc_png:
            f.write(f"![ROC]({os.path.basename(roc_png)})\n\n")
        if pr_png:
            f.write(f"![PR]({os.path.basename(pr_png)})\n\n")
        f.write("### 特征重要性(Top 15)\n\n")
        f.write(feat_imp.head(15).to_string())
        if xgb_importances is not None:
            f.write("\n\n### XGBoost 特征重要性(Top 15)\n\n")
            f.write(xgb_importances.head(15).to_string())

    print(f"Saved report: {out_md}")


if __name__ == "__main__":
    main()


