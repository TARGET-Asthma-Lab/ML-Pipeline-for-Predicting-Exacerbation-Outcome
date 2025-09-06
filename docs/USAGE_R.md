# R Models — Essential & Composite Regression

This guide explains how to run the R implementations (Essential Regression and Composite Regression) with the **same flags** as the Python models. Keep the R scripts generic—no hard-coded paths; use CLI flags instead.

# R Workflows: Essential Regression (ER) & Composite Regression (CR)

This guide explains how to run the R pipelines for **Essential Regression (ER)** and **Composite Regression (CR)** in this repo. The instructions match your two scripts:
- `R/ER_Apr13-v2.R`
- `R/ER_CompositeRegression_Apr29.R`

Keep scripts generic—no hard-coded paths; use relative paths in the repo.

---

## 1) Environment

### Option A — Use `renv` (recommended)
```r
install.packages("renv")
renv::init()

# Core packages used by the ER/CR scripts
install.packages(c(
  "ggplot2","pROC","dplyr","tidyr","randomForest","glmnet","Matrix",
  "igraph","reshape2","gridExtra","patchwork","data.table","readr","stringr"
))

# Bioconductor deps
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("graph","RBGL","pcalg","Rgraphviz"))

# rCausalMGM (if used in your ER analysis)
install.packages("devtools")
devtools::install_github("tyler-lovelace1/rCausalMGM")

renv::snapshot()
```

### Option B — Base R (no `renv`)
```r
install.packages(c(
  "ggplot2","pROC","dplyr","tidyr","randomForest","glmnet","Matrix",
  "igraph","reshape2","gridExtra","patchwork","data.table","readr","stringr","devtools"
))
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("graph","RBGL","pcalg","Rgraphviz"))
devtools::install_github("tyler-lovelace1/rCausalMGM")
```

---

## 2) File placement

Place your R scripts under `R/` and (optionally) any upstream ER sources in `R/ER/`:

```
R/
├─ ER_Apr13-v2.R                    # Essential Regression driver
├─ ER_CompositeRegression_Apr29.R   # Composite Regression driver
└─ ER/                              # (optional) upstream ER helpers: Utilities.R, EstPure.R, ...
```

If you keep upstream ER helper files (e.g., `Utilities.R`, `EstPure.R`, `ER-inference.R`) in `R/ER/`, **source them relative to the repo** (avoid absolute `C:\...` paths). Example:

```r
# in ER_Apr13-v2.R (if you need to source upstream ER helpers):
er_dir <- file.path(getwd(), "R", "ER")
setwd(er_dir)
source("Utilities.R")
source("EstPure.R")
source("Est_beta_dz.R")
source("SupLOVE.R")
source("EstOmega.R")
source("ER-inference.R")
```

---

## 3) Inputs

- CSVs (5 imputations) using a **pattern with `{i}`** so it expands to files `..._1.csv` … `..._5.csv`.  
  Example: `data/clinical/clinical_Oct30_imputed_rf_{i}.csv`

---

## 4) Quickstart — Composite Regression (CR)

Run from an R session (RStudio or terminal `R`):

```r
# Load the CR script
source("R/ER_CompositeRegression_Apr29.R")

# Choose the main dataset pattern (5 imputations)
base_file_path <- "data/clinical/clinical_Oct30_imputed_rf_{i}.csv"

# Set analysis parameters (edit as needed)
params <- list(
  delta = 0.05,            # ER delta
  beta_est = "lasso",      # ER beta estimation method
  var_threshold = 0.01,    # drop near-constant features
  cor_threshold = 0.95,    # high-correlation threshold
  n_preselected = 60,      # pre-screen top-K features (use 60 or 200)
  preserve_correlated = TRUE,
  use_multi_dimensions = TRUE
)

# Analyze all 5 imputations
results <- analyze_datasets(base_file_path, params, n_datasets = 5)

# After the run, metrics & plots are saved; see 'output/' and 'figures/'
```

### Outputs

- Per-run and comparison **metrics** written under `output/` (e.g., `comprehensive_model_comparison.csv`).
- ROC curves (per-dataset + averaged) saved under `figures/`.
- The CR workflow selects the **best AUC model** and saves the average ROC plot across datasets.

---

## 5) Quickstart — Essential Regression (ER)

Run from an R session:

```r
# Load the ER script
source("R/ER_Apr13-v2.R")

# (Optional) if you use upstream ER helpers, source them relative to the repo (see Section 2)

# Prepare your matrix/data:
# X <- as.matrix(
#   readr::read_csv("data/clinical/clinical_Oct30_imputed_rf_1.csv") |>
#   dplyr::select(-Exacerbation.Outcome, -subject_id)
# )
# Y <- readr::read_csv("data/clinical/clinical_Oct30_imputed_rf_1.csv")$Exacerbation.Outcome

# Example: make a scree plot & causal graph after fitting
ensure_output_dir()
# create_scree_plot(X, output_dir = "output")  # saves output/scree_plot.png

# ... run ER (latent factors), then:
# perform_causal_mgm(Z, Y, selected_features_list, feature_loadings_list)
```

**What it produces:**

- **Scree plot** to guide latent dimension choice (`output/scree_plot.png`).  
- **FCI causal graph** and diagnostics (e.g., `output/fci_causal_graph.png`, `output/fci_graph.rds`).  
- **Allocation matrix** (feature→latent loadings) and heatmap in `output/`.  
- **Regression summaries** for latent variables and coefficient plot.

---

## 6) Parameters you can tune (CR)

The CR driver uses a `params` list:

- `delta` (e.g., `0.05`)  
- `beta_est` (e.g., `"lasso"`)  
- `var_threshold` (drop near-constant features)  
- `cor_threshold` (high-correlation pruning threshold)  
- `n_preselected` (`60` or `200`)  
- `preserve_correlated` (`TRUE/FALSE`)  
- `use_multi_dimensions` (`TRUE/FALSE`)

These are passed into the dataset processing and downstream training/evaluation.

---

## 7) Tips

- Use **relative paths** inside the repo (avoid absolute `C:\...` paths).
- The pipelines create `output/` and `figures/` automatically; you can commit key CSVs/PNGs for reproducibility.
- If your CSVs use a different outcome column name, update the loader / preprocessing in the script.

---

## 8) Minimal “one-liner” (non-interactive) run

From a shell/terminal:

```bash
# Composite Regression (one-liner)
Rscript -e 'source("R/ER_CompositeRegression_Apr29.R"); params<-list(delta=0.05,beta_est="lasso",var_threshold=0.01,cor_threshold=0.95,n_preselected=60,preserve_correlated=TRUE,use_multi_dimensions=TRUE); analyze_datasets("data/clinical/clinical_Oct30_imputed_rf_{i}.csv", params, n_datasets=5)'
```

(For ER, prefer an interactive R session so you can inspect intermediate objects and plots.)
