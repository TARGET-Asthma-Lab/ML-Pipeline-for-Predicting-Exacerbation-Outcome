## R Models — Essential & Composite Regression

This guide shows how to run the R implementations (Essential Regression and Composite Regression) with the **same flags** as the Python models.

R scripts. Use the CLI flags below.

---

## Environment

### Option A — R + renv
``r
install.packages("renv")
renv::init()
install.packages(c(
  "optparse","data.table","readr","dplyr","stringr","ggplot2",
  "pROC","ROCR","caret","glmnet","Matrix","patchwork"
))
renv::snapshot()``


### Common flags
`-data main dataset (pattern with {i} for 5 imputations)`
`-alpha / --beta optional diversity datasets (patterns with {i})`
`-raw-div single file (optional); --raw-div-pca-var 0.95 to PCA it`
`-runs 5 expands {i} = 1..5`
`-target Exacerbation.Outcome, --id-col subject_id`
`-top-k 60 (or 200)`
`-folds 10`
`-tag my_run_name (names output files)`
`Outputs: metrics CSV → results/, plots → figures/`

##!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse); library(data.table); library(readr); library(dplyr); library(stringr)
  library(glmnet); library(Matrix); library(pROC); library(ggplot2); library(patchwork)
})

### ----- Flags -----
`option_list <- list(
  make_option("--data", type="character", action="append", help="Main CSV(s) or pattern with {i}"),
  make_option("--runs", type="integer", default=5),
  make_option("--alpha", type="character", action="append", default=NULL),
  make_option("--beta",  type="character", action="append", default=NULL),
  make_option("--raw-div", type="character", default=NULL),
  make_option("--raw-div-pca-var", type="double", default=NA_real_),
  make_option("--target", type="character", default="Exacerbation.Outcome"),
  make_option("--id-col", type="character", default="subject_id"),
  make_option("--top-k",  type="integer", default=60),
  make_option("--folds",  type="integer", default=10),
  make_option("--tag",    type="character", default="ereg_run"),
  make_option("--results", type="character", default="results"),
  make_option("--figures", type="character", default="figures")
)
opt <- parse_args(OptionParser(option_list=option_list))`

dir.create(opt$results, showWarnings=FALSE, recursive=TRUE)
dir.create(opt$figures, showWarnings=FALSE, recursive=TRUE)

### ----- Helpers -----
expand_paths <- function(pats, runs) {
  if (is.null(pats)) return(NULL)
  if (length(pats)==1 && grepl("\\{i\\}", pats[1])) {
    return(vapply(1:runs, function(i) gsub("\\{i\\}", i, pats[1]), FUN.VALUE=character(1)))
  }
  pats
}

read_csvs <- function(paths) lapply(paths, readr::read_csv, show_col_types = FALSE)

merge_sets <- function(main_list, alpha_list=NULL, beta_list=NULL, raw_df=NULL, id_col="subject_id") {
  out <- vector("list", length(main_list))
  for (i in seq_along(main_list)) {
    d <- main_list[[i]]
    if (!is.null(alpha_list)) d <- d %>% inner_join(alpha_list[[i]], by=id_col)
    if (!is.null(beta_list))  d <- d %>% inner_join(beta_list[[i]],  by=id_col)
    if (!is.null(raw_df))     d <- d %>% inner_join(raw_df,         by=id_col)
    out[[i]] <- d
  }
  out
}

pca_raw <- function(raw_path, id_col, var_keep=NA_real_) {
  raw <- readr::read_csv(raw_path, show_col_types = FALSE)
  if (is.na(var_keep)) return(raw)
  feats <- setdiff(names(raw), id_col)
  X <- scale(as.matrix(raw[,feats]))
  p <- prcomp(X)
  var_expl <- cumsum(p$sdev^2)/sum(p$sdev^2)
  k <- which(var_expl >= var_keep)[1]
  score <- as.data.frame(p$x[,1:k,drop=FALSE])
  names(score) <- paste0("PCA_", seq_len(k))
  cbind(raw[id_col], score)
}

### ----- Load data -----
main_paths  <- expand_paths(opt$data,  opt$runs)
alpha_paths <- expand_paths(opt$alpha, opt$runs)
beta_paths  <- expand_paths(opt$beta,  opt$runs)

main_list  <- read_csvs(main_paths)
alpha_list <- if (!is.null(alpha_paths)) read_csvs(alpha_paths) else NULL
beta_list  <- if (!is.null(beta_paths))  read_csvs(beta_paths)  else NULL
raw_df     <- if (!is.null(opt$`raw-div`)) pca_raw(opt$`raw-div`, opt$`id-col`, opt$`raw-div-pca-var`) else NULL

datasets <- merge_sets(main_list, alpha_list, beta_list, raw_df, id_col=opt$`id-col`)

### ----- Feature selection (lasso as proxy for SHAP screening) -----
topk_select <- function(df, target, id_col, k) {
  X <- df %>% select(-all_of(c(target, id_col))) %>% as.matrix()
  y <- df[[target]]
  cv <- glmnet::cv.glmnet(X, y, family="binomial", alpha=1, nfolds=5, type.measure="deviance")
  fit <- glmnet::glmnet(X, y, family="binomial", alpha=1, lambda=cv$lambda.1se)
  coefs <- as.matrix(coef(fit))[-1,1]         # drop intercept
  ord <- order(abs(coefs), decreasing=TRUE)
  idx <- ord[seq_len(min(k, length(ord)))]
  colnames(X)[idx]
}

### ----- TODO: Train/Eval your model -----
For Essential / Composite Regression, take each 'df' in 'datasets',
do K-fold CV, use 'topk_select(df, opt$target, opt$`id-col`, opt$`top-k`)' to reduce features,
compute metrics (AUC via pROC), and save JSON/CSV in 'results/' and plots in 'figures/'.
message("Loaded ", length(datasets), " dataset(s); ready to train.")
