"""
Machine Learning-Based Key Driver Analysis of Training Satisfaction

Purpose
-------
Identify the key drivers of training satisfaction using
CatBoost regression and SHAP explainability.

Methodology
-----------
1. Data preprocessing
2. Nested cross-validation
3. Hyperparameter optimisation
4. Final model fitting
5. SHAP analysis
6. Automated reporting

Notes
-----
The repository uses synthetic data replicating the structure
of the original dataset.
"""

RANDOM_STATE = 42

# Imports
import os
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)
import numpy as np
import pandas as pd
import random
import logging
from datetime import datetime
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from catboost import CatBoostRegressor, Pool
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import randint, uniform
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import ast
import re
from pathlib import Path
import warnings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

logging.info("Libraries loaded successfully")

warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib")
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# --------------------------------
# Configuration
# --------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

DATA_PATH = PROJECT_ROOT / "data" / "synthetic_training_data.xlsx"

BEST_PARAMS_INPUT = PROJECT_ROOT / "data" / "best_params.txt"
LOAD_PRETUNED_PARAMS = False  # Set to True to skip tuning and use parameters from best_params.txt

TARGET_COL = "satisfaction_level"

CATEGORICAL_FEATURES = ["session_delivery", "session_framework", "session_organiser",
                        "session_language", "modules_topic", "level", "year"]

NUMERICAL_FEATURES = ["enrolments", "trainers",  "duration_days"]

OUT_CV_SPLITS, IN_CV_SPLITS = 5, 5  # nested cv splits
RAND_SEARCH_ITER = 50 # iterations for randomized search

# Control random seeds for reproducibility
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

CATBOOST_PARAMS = dict(
    random_seed = RANDOM_STATE,
    iterations = 100,
    early_stopping_rounds = 10,
    verbose = False,
    eval_metric='RMSE',
    bootstrap_type = "Bayesian"
)

# Folder Creation
OUTPUT_ROOT = PROJECT_ROOT / "Output"

OUT_DIR = OUTPUT_ROOT / "CatBoost" / f"{TARGET_COL}_run_{RUN_ID}"

EXCEL_PATH = OUT_DIR / f"results_{RUN_ID}.xlsx"
BEST_PARAMS_FILE = OUT_DIR / "best_params.txt"

WATERFALL_DIR = OUT_DIR / "Waterfall Plots"
DEPENDENCE_DIR = OUT_DIR / "Dependence Plots"
VAL_DIR = OUT_DIR / "Train and Validation Convergence Plot"

SUBCAT_PLOT_DIR = OUT_DIR / "Subcategory Contribution Plots"
SUBCAT_REL_CONTRIB_PLOT_DIR = (SUBCAT_PLOT_DIR / "Relative Contribution Plots")
SUBCAT_DIRECTIONAL_PLOT_DIR = (SUBCAT_PLOT_DIR / "Global Directional Contribution Plots")
SUBCAT_DUAL_PLOT_DIR = (SUBCAT_PLOT_DIR / "Dual Plots")

for dir in [OUT_DIR, SUBCAT_PLOT_DIR, SUBCAT_REL_CONTRIB_PLOT_DIR, SUBCAT_DIRECTIONAL_PLOT_DIR,
            SUBCAT_DUAL_PLOT_DIR, WATERFALL_DIR, DEPENDENCE_DIR, VAL_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

# --------------------------------
# Utility Functions
# --------------------------------

def load_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {ext}")

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero = y_true != 0
    return np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

def load_best_params(path):
    with open(path, "r") as f:
        text = f.read()

    # Extract the part inside the curly braces
    match = re.search(r"\{.*\}", text)
    if not match:
        raise ValueError("No dictionary found in best_params.txt")
    
    dict_text = match.group(0)

    # Replace np.float64(...) with just the number
    dict_text = re.sub(r"np\.float64\((.*?)\)", r"\1", dict_text)

    # Safely parse the dict
    params_dict = ast.literal_eval(dict_text)

    return params_dict

def residual_plot(X, y, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title(f"Observed vs Predicted ({TARGET_COL})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"Observed_vs_Predicted_{TARGET_COL}.png"), dpi=200)
    plt.close()

# --------------------------------
# CatBoost Hyperparameter Tuning
# --------------------------------

def tune_catboost(X: pd.DataFrame, y: pd.Series, cat_cols: list, manual_params=None):
    """
    Nested CV CatBoost tuning with averaged train/validation curves:
    - If manual_params is provided, skip hyperparameter tuning and return those.
    - Otherwise, perform randomized search tuning using inner CV and evaluate on outer CV folds.
    - Returns best parameters and CV metrics (mean ± std).
    - Plots train/validation curves averaged across all outer folds for each loss function.
    """
    if manual_params is not None:
        logging.info("Using manually provided best parameters, skipping hyperparameter tuning.")
        return manual_params, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    logging.info("Running CatBoost hyperparameter tuning with nested CV...")

    # Parameter distributions
    param_dist = {
        "depth": randint(6, 10),
        "learning_rate": uniform(0.01, 0.1),
        "l2_leaf_reg": randint(5, 30),
        "random_strength": randint(2, 8),
        "min_child_samples": randint(20, 50),
        "border_count": randint(32, 254),
        "bagging_temperature": uniform(0.5, 1.5),
    }

    # Outer CV for nested evaluation
    outer_cv = KFold(n_splits=OUT_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    loss_functions = ["RMSE"]
    # removed loss functions: "Huber:delta=10.0", "Huber:delta=50.0",

    best_r2 = -np.inf
    best_params = None
    best_metrics = None
    results = []
    models_trained_avg = {}  # Will store averaged train/val curves

    for loss_fn in loss_functions:
        logging.info(f"Testing loss function = {loss_fn}")
        rmses_outer, r2s_outer, maes_outer, mapes_outer = [], [], [], []
        train_curves, val_curves = [], []

        # Outer CV loop
        for outer_tr_idx, outer_va_idx in outer_cv.split(X):
            X_outer_tr, X_outer_va = X.iloc[outer_tr_idx], X.iloc[outer_va_idx]
            y_outer_tr, y_outer_va = y.iloc[outer_tr_idx], y.iloc[outer_va_idx]

            # Inner CV for hyperparameter tuning
            inner_cv = KFold(n_splits=IN_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            base_model = CatBoostRegressor(**CATBOOST_PARAMS, loss_function=loss_fn)
            random_search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_dist,
                n_iter=RAND_SEARCH_ITER,
                scoring="r2",
                cv=inner_cv,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=0,
            )
            random_search.fit(X_outer_tr, y_outer_tr, cat_features=cat_cols)

            # Train outer model with best inner params
            best_params_fold = {**CATBOOST_PARAMS, **random_search.best_params_, "loss_function": loss_fn}
            model = CatBoostRegressor(**best_params_fold)
            pool_tr = Pool(X_outer_tr, y_outer_tr, cat_features=cat_cols)
            pool_va = Pool(X_outer_va, y_outer_va, cat_features=cat_cols)
            model.fit(pool_tr, eval_set=pool_va, use_best_model=True, verbose=False)

            # Collect train/val curves
            evals_result = model.get_evals_result()
            train_curves.append(evals_result['learn']['RMSE'])
            val_curves.append(evals_result['validation']['RMSE'])

            # Evaluate metrics
            preds = model.predict(pool_va)
            rmses_outer.append(root_mean_squared_error(y_outer_va, preds))
            r2s_outer.append(r2_score(y_outer_va, preds))
            maes_outer.append(mean_absolute_error(y_outer_va, preds))
            mapes_outer.append(mean_absolute_percentage_error(y_outer_va, preds))

        # Average curves across outer folds
        max_iters = max(len(c) for c in train_curves)
        train_avg = np.mean([np.pad(c, (0, max_iters - len(c)), 'edge') for c in train_curves], axis=0)
        val_avg = np.mean([np.pad(c, (0, max_iters - len(c)), 'edge') for c in val_curves], axis=0)
        models_trained_avg[loss_fn] = {"train": train_avg, "validation": val_avg}

        # Aggregate outer CV metrics
        rmse_mean, rmse_std = np.mean(rmses_outer), np.std(rmses_outer)
        r2_mean, r2_std = np.mean(r2s_outer), np.std(r2s_outer)
        mae_mean, mae_std = np.mean(maes_outer), np.std(maes_outer)
        mape_mean, mape_std = np.mean(mapes_outer), np.std(mapes_outer)
        logging.info(f"{loss_fn} | RMSE={rmse_mean:.4f}±{rmse_std:.4f} | R2={r2_mean:.4f}±{r2_std:.4f} | "
            f"MAE={mae_mean:.4f}±{mae_std:.4f} | MAPE={mape_mean:.2f}±{mape_std:.2f}")

        # Save results
        results.append({
            "loss_function": loss_fn,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "r2_mean": r2_mean,
            "r2_std": r2_std,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "mape_mean": mape_mean,
            "mape_std": mape_std,
            "best_params": random_search.best_params_
        })

        # Track best performing loss function
        if r2_mean > best_r2:
            best_r2 = r2_mean
            best_params = best_params_fold.copy()
            best_metrics = (rmse_mean, rmse_std, r2_mean, r2_std, mae_mean, mae_std, mape_mean, mape_std)
    #-------------------------------

    # Plot averaged train/val curves
    plt.figure(figsize=(10, 6))
    colors = ["blue", "orange", "green", "red", "purple", "brown", "cyan", "magenta"]
    for i, loss_fn in enumerate(loss_functions):
        train_curve = models_trained_avg[loss_fn]["train"]
        val_curve = models_trained_avg[loss_fn]["validation"]
        color = colors[i % len(colors)]
        plt.plot(train_curve, linestyle='--', color=color, label=f"{loss_fn} Train")
        plt.plot(val_curve, linestyle='-', color=color, label=f"{loss_fn} Validation")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("CatBoost Train vs Validation Curves Across Loss Functions (Averaged)")
    plt.legend(fontsize=9, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VAL_DIR, "train_val_curve_all_loss_functions_overlay_avg.png"),
                 dpi=200, bbox_inches="tight")
    plt.close()

    for loss_fn in loss_functions:
        train_curve = models_trained_avg[loss_fn]["train"]
        val_curve = models_trained_avg[loss_fn]["validation"]

        plt.figure(figsize=(8, 5))
        plt.plot(train_curve, linestyle='--', color="blue", label="Train RMSE")
        plt.plot(val_curve, linestyle='-', color="orange", label="Validation RMSE")

        plt.xlabel("Iteration")
        plt.ylabel("RMSE")
        plt.title(f"Train vs Validation RMSE ({loss_fn})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(VAL_DIR, f"train_val_curve_{loss_fn.replace(':', '_')}_avg.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()
    
    # Export comparison table
    results_df = pd.DataFrame(results)
    results_df.to_excel(os.path.join(OUT_DIR, "catboost_loss_comparison.xlsx"), index=False)
    logging.info(f"Loss function comparison exported to catboost_loss_comparison.xlsx")

    # Final reporting
    logging.info(f"Best parameters: {best_params} | CV R2={best_metrics[2]:.4f}±{best_metrics[3]:.4f}")
    with open(BEST_PARAMS_FILE, "w") as f:
        f.write(f"Best parameters: {best_params} | CV R2={best_metrics[2]:.4f}±{best_metrics[3]:.4f}")

    return best_params, *best_metrics

# --------------------------------
# SHAP Helper functions
# --------------------------------

def compute_shap_importance(X, sv):
    imp = pd.DataFrame({
        "Feature": X.columns,
        "MeanAbsSHAP": np.abs(sv).mean(axis=0)
    }).sort_values("MeanAbsSHAP", ascending=False).reset_index(drop=True)
    imp["Importance_%"] = 100 * imp["MeanAbsSHAP"] / imp["MeanAbsSHAP"].sum()
    return imp

def plot_shap_bar(imp):
    fig, ax = plt.subplots(figsize = (max(8, len(imp) * 0.5), max(6, len(imp) * 0.4)))
    imp_plot = imp.sort_values("MeanAbsSHAP", ascending=True)
    bars = ax.barh(imp_plot["Feature"], imp_plot["Importance_%"], color="#87CEFA")

    for bar, pct in zip(bars, imp_plot["Importance_%"]):
        y = bar.get_y() + bar.get_height()/2
        if pct < 5:
            x, ha = bar.get_width() + 0.5, "left"
        else:
            x, ha = bar.get_width() - 0.5, "right"
        ax.text(x, y, f"{pct:.1f}%", va = "center", ha = ha, fontsize = 10, color = "black")

    ax.set_xlabel("Contribution (% of total)")
    ax.set_title("SHAP Feature Importance (% of total)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"Shap_bar.png"), dpi = 200, bbox_inches = "tight")
    plt.close(fig)

def plot_shap_directional_bar_raw(X, sv):
    """Plot average directional SHAP values per feature using raw SHAP values."""
    shap_means = np.mean(sv, axis=0)
    imp_dir = pd.DataFrame({"Feature": X.columns, "MeanSHAP": shap_means})
    imp_dir = imp_dir.sort_values("MeanSHAP", ascending=True)

    threshold = imp_dir["MeanSHAP"].abs().max() / 2  # threshold for label placement

    fig, ax = plt.subplots(figsize=(max(8, len(X.columns)*0.5), max(6, len(X.columns)*0.4)))
    bars = ax.barh(
        imp_dir["Feature"],
        imp_dir["MeanSHAP"],
        color=["#87CEFA" if v > 0 else "#F08080" for v in imp_dir["MeanSHAP"]]
    )
    ax.axvline(0, color="black", linewidth=0.8)

    xmin, xmax = ax.get_xlim()
    offset = 0.01 * (xmax - xmin)
    
    for bar, val in zip(bars, imp_dir["MeanSHAP"]):
        y = bar.get_y() + bar.get_height()/2
        abs_val = abs(val)

        if abs_val >= threshold:
            # Large contribution → put text inside bar
            x_text = bar.get_width() / 2
            ha = "center"
            
        else:
            # Small contribution → put text outside bar
            x_text = bar.get_width() + offset if val > 0 else bar.get_width() - offset
            ha = "left" if val > 0 else "right"
            
        ax.text(x_text, y, f"{val:.2f}", va="center", ha=ha, fontsize=10, color="black")

    ax.set_xlabel("Average SHAP Value (same units as model output)")
    ax.set_title("Average Directional Impact of Features")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"Shap_directional_bar_raw.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_shap_waterfalls(explainer, sv, X, n=3, method="max_shap"):

    if method == "max_shap":
        row_idx = np.argsort(np.abs(sv).sum(axis=1))[-n:][::-1]
    elif method == "random":
        row_idx = np.random.choice(X.shape[0], n, replace = False)
    else:
        raise ValueError("Invalid method")

    for idx in row_idx:
        shap_exp = shap.Explanation(
            values = sv[idx],
            base_values = float(explainer.expected_value),
            data = X.iloc[idx],
            feature_names = X.columns
        )
        fig = plt.figure(figsize = (14, 10))
        shap.plots.waterfall(shap_exp, max_display = 15, show = False)
        fig.tight_layout()
        fig.savefig(os.path.join(WATERFALL_DIR, f"Shap_waterfall_{method}_row{idx}.png"),
                    dpi = 200, bbox_inches = "tight")
        plt.close(fig)

def plot_shap_beeswarm(X, sv, subgroup_tables, categorical_features):
    X_num, X_cat = pd.DataFrame(index = X.index), pd.DataFrame(index = X.index)
    sv_num, sv_cat = [], []

    for cat in categorical_features or []:
        subcat_summary = subgroup_tables.get(cat, pd.DataFrame())
        if subcat_summary.empty:
            continue
        top_subcat = subcat_summary.iloc[0]["Category"]
        col_name = f"{cat}:\n{top_subcat}"
        X_cat[col_name] = (X[cat] == top_subcat).astype(int)
        sv_cat.append(sv[:, X.columns.get_loc(cat)])

    num_features = [c for c in X.columns if c not in categorical_features]
    for num in num_features:
        X_num[num] = X[num]
        sv_num.append(sv[:, X.columns.get_loc(num)])

    sv_num = np.column_stack(sv_num) if sv_num else np.empty((X.shape[0], 0))
    sv_cat = np.column_stack(sv_cat) if sv_cat else np.empty((X.shape[0], 0))

    if sv_num.shape[1] > 0:
        fig_num = plt.figure(figsize=(12, 8))
        shap.summary_plot(sv_num, X_num, plot_type = "dot", show=False, color=plt.get_cmap("coolwarm"))
        plt.title("Numerical Features", fontsize = 10)
        fig_num.tight_layout()
        fig_num.savefig(os.path.join(OUT_DIR, f"Shap_beeswarm_numerical.png"),
                         dpi = 200, bbox_inches = "tight")
        plt.close(fig_num)

    if sv_cat.shape[1] > 0:
        fig_cat = plt.figure(figsize=(12, 8))

        # Create a color matrix same shape as sv_cat (rows x features)
        # 1 -> in top subcategory (red), 0 -> other (blue)
        colors = np.where(X_cat.values == 1, "#FF2020", "#2079FF")

        # SHAP expects colors per value; flatten row-wise
        shap.summary_plot(
            sv_cat, X_cat,
            plot_type = "dot",
            show = False,
            color = colors
        )

        # Remove default gradient colorbar
        if fig_cat.axes[-1].get_legend() is None:  # precaution
            try:
                fig_cat.axes[-1].remove()
            except Exception:
                pass

        # Add custom legend
        patch_in = mpatches.Patch(color = "#FF2020", label = "In subcategory")
        patch_out = mpatches.Patch(color = "#2079FF", label = "Other")
        plt.legend(handles=[patch_in, patch_out], title = "", loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.title("Top Subcategory Features", fontsize = 10)
        fig_cat.tight_layout()
        fig_cat.savefig(os.path.join(OUT_DIR, f"Shap_beeswarm_categorical.png"),
                         dpi = 200, bbox_inches = "tight")
        plt.close(fig_cat)

def plot_shap_dependence_numerical(X, sv, numerical_features):
    # Create SHAP dependence plots for each numerical feature.

    # Define zoom regions here (customize per feature)
    zoom_regions = {
        "enrolments": {"xlim": (0, 40)},
        "duration_days": {"xlim": (0, 50)},
        "trainers": {"xlim": (0, 14)}
    }

    for num_feat in numerical_features:
        # SHAP dependence_plot internally creates its own figure
        shap.dependence_plot(
            num_feat,
            sv,
            X,
            display_features=X,
            interaction_index=None,
            show=False
        )
        
        fig = plt.gcf()
        ax = plt.gca()

        # Add grid and SHAP=0 reference line
        ax.minorticks_on()
        ax.grid(True, color="lightgrey", linestyle="-", linewidth=0.5, alpha=0.7)
        ax.grid(which="minor", color="lightgrey", linestyle="-", linewidth=0.3, alpha=0.5)
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.8)
        ax.set_title(f"Dependence Plot: {num_feat}")

        # Detect potential outlier
        values = X[num_feat]
        q98 = values.quantile(0.98)
        max_val = values.max()
        idx_max = values.values.argmax()  # <- position in array
        shap_val_max = sv[idx_max, X.columns.get_loc(num_feat)]

        if max_val > q98 * 5:  # heuristic: extreme outlier
            xlim_outlier = (0, 210)  # only enrolments has an extreme outlier
            ax.set_xlim(*xlim_outlier) 

            # Compute mask for points within the xlim
            mask_outlier = (values >= xlim_outlier[0]) & (values <= xlim_outlier[1])
            if mask_outlier.any():
                ymin = sv[mask_outlier, X.columns.get_loc(num_feat)].min()
                ymax = sv[mask_outlier, X.columns.get_loc(num_feat)].max()
                ax.set_ylim(ymin - 0.5, ymax + 0.5)

            # Add inset axis
            ax_inset = inset_axes(ax, width="20%", height="20%", loc="upper right")

            shap.dependence_plot(
                num_feat,
                sv,
                X,
                display_features=X,
                interaction_index=None,
                show=False,
                ax=ax_inset
            )

            # Focus inset on extreme tail only
            ax_inset.set_xlim(max_val - 10, max_val + 10)
            ax_inset.set_ylim(shap_val_max - 1, shap_val_max + 1)
            ax_inset.minorticks_on()
            ax_inset.grid(True, color="lightgrey", linestyle="-", linewidth=0.5, alpha=0.7)
            ax_inset.grid(which="minor", color="lightgrey", linestyle="-", linewidth=0.3, alpha=0.5)
            ax_inset.set_xlabel("")
            ax_inset.set_ylabel("")
            ax_inset.set_title("Outlier region", fontsize=8)
        
        fig.tight_layout()
        fig.savefig(os.path.join(DEPENDENCE_DIR, f"Dependence_{num_feat}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

        # If zoom region is defined for this feature, plot it
        if num_feat in zoom_regions:
            xlim = zoom_regions[num_feat].get("xlim", None)

            # Mask SHAP values within the xlim
            mask = (X[num_feat] >= xlim[0]) & (X[num_feat] <= xlim[1])
            ymin = sv[mask, X.columns.get_loc(num_feat)].min()
            ymax = sv[mask, X.columns.get_loc(num_feat)].max()
            zoom_regions[num_feat]["ylim"] = (ymin - 0.2, ymax + 0.2)

            shap.dependence_plot(
                num_feat,
                sv,
                X,
                display_features=X,
                interaction_index=None,
                show=False
            )

            fig_zoom = plt.gcf()
            ax_zoom = plt.gca()
            ax_zoom.minorticks_on()
            ax_zoom.grid(True, color="lightgrey", linestyle="-", linewidth=0.5, alpha=0.7)
            ax_zoom.grid(which="minor", color="lightgrey", linestyle="-", linewidth=0.3, alpha=0.5)
            ax_zoom.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.8)
            ax_zoom.set_title(f"Zoomed Dependence: {num_feat}")

            ax_zoom.set_xlim(*zoom_regions[num_feat]["xlim"])
            ax_zoom.set_ylim(*zoom_regions[num_feat]["ylim"])

            fig_zoom.tight_layout()
            fig_zoom.savefig(os.path.join(DEPENDENCE_DIR, f"Dependence_{num_feat}_zoom.png"),
                             dpi=200, bbox_inches="tight")
            plt.close(fig_zoom)

# Subcategory plotting functions
def plot_subcat_contrib(subgroup_summary, cat):
    subgroup_summary = subgroup_summary.sort_values("Contribution_within%", ascending=True)
    fig, ax = plt.subplots(figsize=(max(8, len(subgroup_summary) * 0.6), max(6, len(subgroup_summary) * 0.4)))
    bars = ax.barh(subgroup_summary["Category"], subgroup_summary["Contribution_within%"], color = "#87CEFA")

    for bar, pct in zip(bars, subgroup_summary["Contribution_within%"]):
        y = bar.get_y() + bar.get_height()/2
        if pct < 5:
            x, ha = bar.get_width() + 0.5, "left"
        else:
            x, ha = bar.get_width() - 0.5, "right"
        ax.text(x, y, f"{pct:.1f}%", va = "center", ha = ha, fontsize = 10)

    ax.set_xlabel("Contribution (% within feature)")
    ax.set_title(f"Subcategory SHAP Contribution (within {cat})")
    fig.tight_layout()
    fig.savefig(os.path.join(SUBCAT_REL_CONTRIB_PLOT_DIR, f"Shap_subcats_{cat}.png"), dpi = 200, bbox_inches = "tight")
    plt.close(fig)

def plot_subcat_dir_raw(subgroup_summary, cat):
    """Plot subcategory directional SHAP using raw SHAP values."""
    subgroup_summary = subgroup_summary.sort_values("MeanSHAP", ascending=True)
    colors = ['#87CEFA' if v > 0 else '#F08080' for v in subgroup_summary["MeanSHAP"]]

    # Direction-specific label placement thresholds
    max_pos = subgroup_summary["MeanSHAP"].clip(lower=0).max()
    min_neg = subgroup_summary["MeanSHAP"].clip(upper=0).min()
    threshold_pos = max_pos / 2 if max_pos > 0 else 0
    threshold_neg = abs(min_neg) / 2 if min_neg < 0 else 0

    fig, ax = plt.subplots(figsize=(max(8, len(subgroup_summary)*0.6), max(6, len(subgroup_summary)*0.4)))
    bars = ax.barh(subgroup_summary["Category"], subgroup_summary["MeanSHAP"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)

    xmin, xmax = ax.get_xlim()
    offset = 0.01 * (xmax - xmin)

    for bar, val in zip(bars, subgroup_summary["MeanSHAP"]):
        y = bar.get_y() + bar.get_height()/2

        if val > 0:  # positive bar
            if val >= threshold_pos:
                x_text = bar.get_width() - offset  # inside, near end
                ha = "right"
            else:
                x_text = bar.get_width() + offset  # outside
                ha = "left"
        else:  # negative bar
            if abs(val) >= threshold_neg:
                x_text = bar.get_width() + offset  # inside, near end
                ha = "left"
            else:  
                x_text = bar.get_width() - offset  # outside
                ha = "right"

        ax.text(x_text, y, f"{val:.2f}", va="center", ha=ha, fontsize=10, color="black")

    ax.set_xlabel("Mean SHAP Value")
    ax.set_title(f"Directional Impact of {cat} Subcategories")
    fig.tight_layout()
    fig.savefig(os.path.join(SUBCAT_DIRECTIONAL_PLOT_DIR, f"Shap_subcats_dir_raw_{cat}.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_subcat_dual_raw(subgroup_summary, cat):
    """
    Dual-panel plot:
    - Panel A: Within-feature contribution (%)
    - Panel B: Directional impact (raw mean SHAP values)
    """
    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(12, len(subgroup_summary) * 0.8), max(6, len(subgroup_summary) * 0.5)),
        sharey=True
    )

    # Panel A: Within-feature contribution (as %)
    ax = axes[0]
    sf = subgroup_summary.sort_values("Contribution_within%", ascending=True)
    bars = ax.barh(sf["Category"], sf["Contribution_within%"], color="#87CEFA")
    for bar, pct in zip(bars, sf["Contribution_within%"]):
        y = bar.get_y() + bar.get_height()/2
        if pct < 5:
            x, ha = bar.get_width() + 0.5, "left"
        else:
            x, ha = bar.get_width() - 0.5, "right"
        ax.text(x, y, f"{pct:.1f}%", va="center", ha=ha, fontsize=8, color="black")
    ax.set_xlabel("Contribution (% within feature)")
    ax.set_title(f"{cat} — Relative Share")

    # Panel B: Directional impact (raw mean SHAP)
    ax = axes[1]
    sf = subgroup_summary.sort_values("MeanSHAP", ascending=True)
    colors = ['#87CEFA' if v > 0 else '#F08080' for v in sf["MeanSHAP"]]

    # Direction-specific label placement thresholds
    max_pos = subgroup_summary["MeanSHAP"].clip(lower=0).max()
    min_neg = subgroup_summary["MeanSHAP"].clip(upper=0).min()
    threshold_pos = max_pos / 2 if max_pos > 0 else 0
    threshold_neg = abs(min_neg) / 2 if min_neg < 0 else 0

    bars = ax.barh(sf["Category"], sf["MeanSHAP"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)

    xmin, xmax = ax.get_xlim()
    offset = 0.01 * (xmax - xmin)

    for bar, val in zip(bars, sf["MeanSHAP"]):
        y = bar.get_y() + bar.get_height()/2

        if val > 0:  # positive bar
            if val >= threshold_pos:
                x_text = bar.get_width() - offset  # inside, near end
                ha = "right"
            else:
                x_text = bar.get_width() + offset  # outside
                ha = "left"
        else:  # negative bar
            if abs(val) >= threshold_neg:
                x_text = bar.get_width() + offset  # inside, near end
                ha = "left"
            else:  
                x_text = bar.get_width() - offset  # outside
                ha = "right"

        ax.text(x_text, y, f"{val:.2f}", va="center", ha=ha, fontsize=8, color="black")

    ax.set_xlabel("Mean SHAP Value (same units as model output)")
    ax.set_title(f"{cat} — Directional Impact")

    fig.tight_layout()
    fig.savefig(
        os.path.join(SUBCAT_DUAL_PLOT_DIR, f"Shap_subcats_{cat}_dual_raw.png"),
        dpi=200, bbox_inches="tight"
    )
    plt.close(fig)

def compute_and_plot_subcategories_raw(X, sv, categorical_features):
    """Compute subcategory contributions and directional SHAP using raw SHAP values."""
    subgroup_tables = {}
    if not categorical_features:
        return subgroup_tables

    for cat in categorical_features:
        vals = pd.DataFrame({
            "Category": X[cat].values,
            "SHAP": sv[:, X.columns.get_loc(cat)]
        })
        grouped = vals.groupby("Category")
        contrib_abs = grouped["SHAP"].apply(lambda x: np.abs(x).sum())
        contrib_within = 100 * contrib_abs / contrib_abs.sum()

        mean_shap = grouped["SHAP"].mean()
        # Keep raw mean as directional value
        freq = grouped.size() / len(X) * 100

        subgroup_summary = pd.DataFrame({
            "Category": contrib_abs.index,
            "Contribution_within%": contrib_within.values,
            "MeanSHAP": mean_shap.values,
            "Frequency_%": freq.values
        }).sort_values("Contribution_within%", ascending=False)

        subgroup_tables[cat] = subgroup_summary

        # Call plotting helpers
        plot_subcat_contrib(subgroup_summary, cat)
        plot_subcat_dir_raw(subgroup_summary, cat)
        plot_subcat_dual_raw(subgroup_summary, cat)  # within-feature contribution can stay as % 

    return subgroup_tables

# Main SHAP function
def run_shap_and_save(model, X, categorical_features=None):
    """
    Compute SHAP values for a trained CatBoost, generate plots, and return importance summaries.
    Returns
    -------
    imp (pd.DataFrame) : Overall SHAP importance (% of total)
    subgroup_tables (dict) :  Subcategory-level contributions per categorical feature
    """

    # Prepare SHAP input
    if categorical_features:
        for cat in categorical_features:
            X[cat] = X[cat].astype(str)

    # Compute SHAP values
    logging.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)

    # Feature importance summary
    imp = compute_shap_importance(X, sv)

    # Plots
    logging.info("Generating SHAP plots...")
    plot_shap_bar(imp)
    plot_shap_directional_bar_raw(X, sv)
    plot_shap_dependence_numerical(X, sv, NUMERICAL_FEATURES)
    for method in ["max_shap", "random"]:
        plot_shap_waterfalls(explainer, sv, X, n = 5, method = method)

    # Subcategories
    logging.info("Computing and plotting subcategory contributions...")
    subgroup_tables = compute_and_plot_subcategories_raw(X, sv, categorical_features)

    # Beeswarm plots
    plot_shap_beeswarm(X, sv, subgroup_tables, categorical_features)

    return imp, subgroup_tables

# -----------------------------------
# Exporting subcategory data to Excel
# -----------------------------------

def write_grouped_subcats(writer, subgroup_tables):
    """
    Write grouped subcategory contributions into one Excel sheet
    """
    all_blocks = []

    for cat, df in subgroup_tables.items():
        # Sort subcategories by within-feature contribution descending
        df_sorted = df.sort_values("Contribution_within%", ascending = False)

        # Select relevant columns and add Mean SHAP
        block = df_sorted[["Category", "Contribution_within%",
                            "MeanSHAP", "Frequency_%"]].copy()
        block.columns = ["Category", "Contribution (% within feature)",
                            "Mean SHAP", "Frequency (%)"]

        # Add a header row for the feature
        header_row = pd.DataFrame([[cat] + [None]*(len(block.columns)-1) ], columns=block.columns)
        all_blocks.append(header_row)
        all_blocks.append(block)

        # Add a blank row for spacing
        spacer = pd.DataFrame([[""] + [None]*(len(block.columns)-1)], columns=block.columns)
        all_blocks.append(spacer)

    all_blocks = [df for df in all_blocks if df is not None and not df.empty]
    final_df = pd.concat(all_blocks, ignore_index = True)
    sheet_name = "Subcats_Summary"
    final_df.to_excel(writer, sheet_name = sheet_name, index = False)

    # Conditional formatting for negative directional contributions
    try:
        import openpyxl
        wb = writer.book
        ws = wb[sheet_name]
        for row in range(2, ws.max_row + 1):
            cell = ws.cell(row = row, column = 3)  # "Directional Contribution (% global)"
            if isinstance(cell.value, (int, float)) and cell.value < 0:
                cell.font = openpyxl.styles.Font(color = "FF0000")  # red font
    except ImportError:
        pass  # skip formatting if openpyxl is not installed

# --------------------------------
# Main Pipeline
# --------------------------------

def main():
    # Load global data
    df = load_dataframe(DATA_PATH)

    needed = list(dict.fromkeys(CATEGORICAL_FEATURES + NUMERICAL_FEATURES + [TARGET_COL]))
    df = df.dropna(subset = needed).reset_index(drop = True)
    logging.info(f"Dataframe size after dropping rows with missing feature info: {df.shape}")

    # Split dataframe into predictors (X), target (y), and track categorical features for CatBoost
    X = df[[c for c in CATEGORICAL_FEATURES + NUMERICAL_FEATURES if c in df.columns]].copy()
    y = df[TARGET_COL].copy()
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    # --- enforce numeric consistency for plotting/model stability ---
    for col in X.columns:
        if X[col].dtype == "object":
            try:
                X[col] = X[col].astype(float)
            except:
                pass

    # CatBoost Nested CV & tuning (with option to skip tuning and load best params from file)       
    manual_params = load_best_params(BEST_PARAMS_INPUT) if LOAD_PRETUNED_PARAMS else None
    (best_params, cv_rmse_mean, cv_rmse_std, cv_r2_mean, cv_r2_std, cv_mae_mean, cv_mae_std,
        cv_mape_mean, cv_mape_std) = tune_catboost(X, y, cat_cols, manual_params=manual_params)

    # Final CatBoost regression on full dataset using the best hyperparameters
    logging.info(
    "Refitting final CatBoost model on the full dataset using best hyperparameters "
    "for SHAP analysis and deployment."
    )
    final_model = CatBoostRegressor(**best_params)
    final_model.fit(Pool(X, y, cat_features=cat_cols), verbose=False)
    final_model.save_model(os.path.join(OUT_DIR, "final_catboost_model.cbm"))
    y_pred = final_model.predict(X)

    # === Observed vs Predicted Plot ===
    residual_plot(X, y, y_pred)

    # SHAP
    shap_imp_df, subgroup_tables = run_shap_and_save(final_model, X, cat_cols)

    # Save to Excel
    with pd.ExcelWriter(EXCEL_PATH, engine = "openpyxl") as writer:

        pd.DataFrame({
            "Metric": [
                "CV_RMSE_Mean", "CV_RMSE_Std",
                "CV_R2_Mean", "CV_R2_Std",
                "CV_MAE_Mean", "CV_MAE_Std",
                "CV_MAPE_Mean", "CV_MAPE_Std"
            ],
            "Value": [
                cv_rmse_mean, cv_rmse_std,
                cv_r2_mean, cv_r2_std,
                cv_mae_mean, cv_mae_std,
                cv_mape_mean, cv_mape_std
            ]
        }).to_excel(writer, sheet_name="Model_Performance", index=False)
        
        shap_imp_df.to_excel(writer, sheet_name = "SHAP_Importance", index = False)
        write_grouped_subcats(writer, subgroup_tables)

    logging.info("=== Analysis Complete ===")
    logging.info(
        f"CV RMSE: {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f} | "
        f"CV R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f} | "
        f"CV MAE: {cv_mae_mean:.4f} ± {cv_mae_std:.4f} | "
        f"CV MAPE: {cv_mape_mean:.2f} ± {cv_mape_std:.2f}"
    )
    logging.info(f"Excel saved at: {EXCEL_PATH}")
    logging.info(f"Plots saved in: {OUT_DIR}")

if __name__=="__main__":
    main()
