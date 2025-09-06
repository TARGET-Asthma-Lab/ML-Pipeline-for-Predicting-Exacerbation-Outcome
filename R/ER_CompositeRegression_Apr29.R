# One-time package installation (run these commands only once when setting up):
# For CRAN packages:
# install.packages("devtools")
# install.packages(c("huge", "ggplot2", "pROC", "dplyr", "tidyr", 
#                   "randomForest", "glmnet", "Matrix", "igraph"))
#
# For Bioconductor packages:
# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")
# BiocManager::install(c("graph", "RBGL", "pcalg", "Rgraphviz"))
#
# For rCausalMGM:
# devtools::install_github("tyler-lovelace1/rCausalMGM")

# Add to the installation section at the top:
# install.packages("gridExtra")

# APR 24
# Train test split and LOOCV. Final best implementation: CR 0.87 all other around 0.7


library(BiocManager)
library(graph)
library(RBGL)
library(pcalg)
library(huge)
library(ggplot2)
library(pROC)
library(dplyr)
library(tidyr)
library(randomForest)
library(glmnet)
library(Matrix)
library(igraph)
library(devtools)
library(rCausalMGM)
library(Rgraphviz)
library(reshape2)
library(gridExtra)


# Set working directory and source ER functions
setwd("C:\\Users\\omatu\\OneDrive\\Desktop\\R_UdeM\\UdeM\\ER")
source("ER_Apr13-v2.R")


# Function to split data into training and test sets
split_dataset <- function(X, Y, Z, test_size = 0.2, seed = 42) {
 set.seed(seed)
 n <- nrow(X)
 test_idx <- sample(1:n, size = floor(n * test_size))
 train_idx <- setdiff(1:n, test_idx)
 
 list(
  train = list(X = X[train_idx,], Y = Y[train_idx], Z = Z[train_idx,]),
  test = list(X = X[test_idx,], Y = Y[test_idx], Z = Z[test_idx,])
 )
}

# Function to calculate model metrics
calculate_model_metrics <- function(actual, predicted, threshold = 0.5) {
 # Print diagnostics
 cat("\nDiagnostic information:")
 cat("\nActual class distribution:", table(actual))
 cat("\nPredicted probabilities summary:", summary(predicted))
 
 # Convert predictions to binary
 predicted_class <- ifelse(predicted > threshold, 1, 0)
 cat("\nPredicted class distribution:", table(predicted_class))
 
 # Create confusion matrix with error handling
 confusion <- tryCatch({
  table(factor(actual, levels=c(0,1)), 
        factor(predicted_class, levels=c(0,1)))
 }, error = function(e) {
  cat("\nError creating confusion matrix:", e$message)
  return(matrix(0, nrow=2, ncol=2))
 })
 
 # Print confusion matrix
 cat("\nConfusion matrix:\n")
 print(confusion)
 
 # Calculate metrics with error handling
 metrics <- tryCatch({
  # Basic accuracy can always be calculated
  accuracy <- mean(actual == predicted_class)
  
  # Other metrics need valid confusion matrix
  if(nrow(confusion) == 2 && ncol(confusion) == 2) {
   precision <- confusion[2,2] / sum(confusion[,2])
   recall <- confusion[2,2] / sum(confusion[2,])
   f1_score <- 2 * (precision * recall) / (precision + recall)
  } else {
   precision <- NA
   recall <- NA
   f1_score <- NA
  }
  
  # MSE can always be calculated
  mse <- mean((actual - predicted)^2)
  
  list(
   accuracy = accuracy,
   precision = precision,
   recall = recall,
   f1_score = f1_score,
   mse = mse
  )
 }, error = function(e) {
  cat("\nError calculating metrics:", e$message)
  list(
   accuracy = NA,
   precision = NA,
   recall = NA,
   f1_score = NA,
   mse = NA
  )
 })
 
 return(metrics)
}

# Function to standardize column names
standardize_datasets <- function(datasets) {
 # Clean column names - replace hyphens with underscores
 datasets <- lapply(datasets, function(df) {
  names(df) <- gsub("-", "_", names(df))
  return(df)
 })
 
 # Get common columns
 common_cols <- Reduce(intersect, lapply(datasets, colnames))
 
 # Keep only common columns in all datasets
 datasets <- lapply(datasets, function(df) {
  df[, common_cols, drop = FALSE]
 })
 
 return(datasets)
}

# Function to create Z factor summary
create_z_factor_summary <- function(selected_features, X_scaled_train, model_dir, dataset_num) {
 # Create summary dataframe
 z_summary <- data.frame(
  Z_Factor = character(),
  N_Contributing = numeric(),
  Top_Features = character(),
  stringsAsFactors = FALSE
 )
 
 for(z_idx in names(selected_features)) {
  features <- selected_features[[z_idx]]
  # Get top 2 features with their correlations
  top_features <- head(colnames(X_scaled_train)[features], 2)  # Get top 2 features
  
  z_summary <- rbind(z_summary,
                     data.frame(
                      Z_Factor = paste0("Z", z_idx),
                      N_Contributing = length(features),
                      Top_Features = paste(top_features, collapse=", "),
                      stringsAsFactors = FALSE
                     )
  )
 }
 
 # Save summary
 write.csv(z_summary, 
           file.path(model_dir, sprintf("z_factor_summary_dataset_%d.csv", dataset_num)), 
           row.names=FALSE)
 
 # Print summary in a formatted way
 cat(sprintf("\nZ Factor Summary for Dataset %d:\n", dataset_num))
 cat("\nZ_Factor  N_Contributing  Top_Features\n")
 cat("----------------------------------------\n")
 for(i in 1:nrow(z_summary)) {
  cat(sprintf("%-8s  %-13d  %s\n", 
              z_summary$Z_Factor[i], 
              z_summary$N_Contributing[i], 
              z_summary$Top_Features[i]))
 }
 cat("\n")
 
 return(z_summary)
}

# Move this function before analyze_datasets
create_top_features_summary <- function(all_ensemble_models, model_dir) {
 # 1. Overall top features across all Z factors
 all_feature_stats <- data.frame(
  Feature = character(),
  Total_Frequency = numeric(),
  Mean_Importance = numeric(),
  Contributing_Z_Factors = character(),
  Best_AUC = numeric(),
  stringsAsFactors = FALSE
 )
 
 # Collect statistics for each feature
 for(z_idx in names(all_ensemble_models)) {
  z_models <- all_ensemble_models[[z_idx]]
  
  # Get all unique features for this Z factor
  features <- unique(unlist(lapply(z_models, function(m) 
   colnames(split_data$train$X)[m$features])))
  
  for(feature in features) {
   # Calculate statistics for this feature
   feature_models <- z_models[sapply(z_models, function(m) 
    feature %in% colnames(split_data$train$X)[m$features])]
   
   if(length(feature_models) > 0) {
    best_auc <- max(sapply(feature_models, function(m) m$auc))
    mean_imp <- mean(sapply(feature_models, function(m) {
     if(feature %in% colnames(split_data$train$X)[m$features]) {
      abs(coef(m$model, s=m$lambda)[
       which(colnames(split_data$train$X)[m$features] == feature)])
     } else 0
    }))
    
    # Update or add to all_feature_stats
    if(feature %in% all_feature_stats$Feature) {
     idx <- which(all_feature_stats$Feature == feature)
     all_feature_stats$Total_Frequency[idx] <- 
      all_feature_stats$Total_Frequency[idx] + length(feature_models)
     all_feature_stats$Mean_Importance[idx] <- 
      (all_feature_stats$Mean_Importance[idx] + mean_imp) / 2
     all_feature_stats$Contributing_Z_Factors[idx] <- 
      paste(all_feature_stats$Contributing_Z_Factors[idx], 
            paste0("Z", z_idx), sep=",")
     all_feature_stats$Best_AUC[idx] <- 
      max(all_feature_stats$Best_AUC[idx], best_auc)
    } else {
     all_feature_stats <- rbind(all_feature_stats,
                                data.frame(
                                 Feature = feature,
                                 Total_Frequency = length(feature_models),
                                 Mean_Importance = mean_imp,
                                 Contributing_Z_Factors = paste0("Z", z_idx),
                                 Best_AUC = best_auc,
                                 stringsAsFactors = FALSE
                                )
     )
    }
   }
  }
 }
 
 # Sort by total frequency and mean importance
 all_feature_stats$Score <- all_feature_stats$Total_Frequency * 
  all_feature_stats$Mean_Importance * 
  all_feature_stats$Best_AUC
 all_feature_stats <- all_feature_stats[order(-all_feature_stats$Score),]
 
 # Save top 20 overall features
 write.csv(head(all_feature_stats, 20), 
           file.path(model_dir, "cr_top20_overall_features.csv"), 
           row.names=FALSE)
 
 # 2. Top features by Z factor
 z_factor_features <- list()
 for(z_idx in names(all_ensemble_models)) {
  z_models <- all_ensemble_models[[z_idx]]
  
  # Get feature statistics for this Z factor
  z_features <- data.frame(
   Feature = character(),
   Frequency = numeric(),
   Mean_Importance = numeric(),
   Best_AUC = numeric(),
   Best_Model_Details = character(),
   stringsAsFactors = FALSE
  )
  
  features <- unique(unlist(lapply(z_models, function(m) 
   colnames(split_data$train$X)[m$features])))
  
  for(feature in features) {
   feature_models <- z_models[sapply(z_models, function(m) 
    feature %in% colnames(split_data$train$X)[m$features])]
   
   if(length(feature_models) > 0) {
    best_model <- feature_models[[which.max(sapply(feature_models, function(m) m$auc))]]
    
    z_features <- rbind(z_features,
                        data.frame(
                         Feature = feature,
                         Frequency = length(feature_models),
                         Mean_Importance = mean(sapply(feature_models, function(m) {
                          if(feature %in% colnames(split_data$train$X)[m$features]) {
                           abs(coef(m$model, s=m$lambda)[
                            which(colnames(split_data$train$X)[m$features] == feature)])
                          } else 0
                         })),
                         Best_AUC = best_model$auc,
                         Best_Model_Details = sprintf("threshold=%s,alpha=%.2f,lambda=%s",
                                                      best_model$threshold, best_model$alpha, best_model$lambda),
                         stringsAsFactors = FALSE
                        )
    )
   }
  }
  
  # Sort by frequency and importance
  z_features$Score <- z_features$Frequency * z_features$Mean_Importance * z_features$Best_AUC
  z_features <- z_features[order(-z_features$Score),]
  
  # Save top 10 features for this Z factor
  write.csv(head(z_features, 10), 
            file.path(model_dir, sprintf("cr_top10_features_Z%s.csv", z_idx)), 
            row.names=FALSE)
  
  z_factor_features[[paste0("Z", z_idx)]] <- z_features
 }
 
 # 3. Create a summary report
 sink(file.path(model_dir, "cr_features_summary.txt"))
 cat("=== CR Feature Selection Summary ===\n\n")
 
 cat("Overall Top 10 Features:\n")
 print(head(all_feature_stats[,c("Feature", "Total_Frequency", "Mean_Importance", 
                                 "Contributing_Z_Factors", "Best_AUC")], 10))
 
 for(z_idx in names(z_factor_features)) {
  cat(sprintf("\n\nTop 5 Features for %s:\n", z_idx))
  print(head(z_factor_features[[z_idx]][,c("Feature", "Frequency", 
                                           "Mean_Importance", "Best_AUC")], 5))
 }
 sink()
 
 return(list(
  overall_features = all_feature_stats,
  z_factor_features = z_factor_features
 ))
}

# Function to create and save ROC curves with correct indexing
create_roc_curves <- function(model_results, dataset_num, output_dir) {
 library(pROC)
 library(ggplot2)
 
 # Define models to include and their colors
 models <- c("ER (all dimensions)", "Causal ER", "Lasso with interactions", "CR")
 colors <- c("#1f78b4", "#e31a1c", "#33a02c", "#6a3d9a")  # Added purple for CR
 
 # Create plot data frame
 plot_data <- data.frame()
 
 # Add ROC curve data for each model
 if(!is.null(model_results$er) && !is.null(model_results$er$roc)) {
  roc_obj <- model_results$er$roc
  plot_data <- rbind(plot_data, data.frame(
   FPR = 1 - roc_obj$specificities,
   TPR = roc_obj$sensitivities,
   Model = "ER (all dimensions)",
   AUC = round(auc(roc_obj), 3)
  ))
 }
 
 if(!is.null(model_results$causal) && !is.null(model_results$causal$roc)) {
  roc_obj <- model_results$causal$roc
  plot_data <- rbind(plot_data, data.frame(
   FPR = 1 - roc_obj$specificities,
   TPR = roc_obj$sensitivities,
   Model = "Causal ER",
   AUC = round(auc(roc_obj), 3)
  ))
 }
 
 if(!is.null(model_results$lasso) && !is.null(model_results$lasso$roc)) {
  roc_obj <- model_results$lasso$roc
  plot_data <- rbind(plot_data, data.frame(
   FPR = 1 - roc_obj$specificities,
   TPR = roc_obj$sensitivities,
   Model = "Lasso with interactions",
   AUC = round(auc(roc_obj), 3)
  ))
 }
 
 # Add CR model
 if(!is.null(model_results$cr) && !is.null(model_results$cr$roc)) {
  roc_obj <- model_results$cr$roc
  plot_data <- rbind(plot_data, data.frame(
   FPR = 1 - roc_obj$specificities,
   TPR = roc_obj$sensitivities,
   Model = "CR",
   AUC = round(auc(roc_obj), 3)
  ))
 }
 
 # Create legend labels with AUC values and thresholds
 legend_labels <- sapply(unique(plot_data$Model), function(m) {
  auc_val <- unique(plot_data$AUC[plot_data$Model == m])
  threshold <- if(!is.null(model_results[[tolower(m)]]$threshold)) {
   sprintf(", threshold = %.2f", model_results[[tolower(m)]]$threshold)
  } else {
   ""
  }
  sprintf("%s [AUC = %0.3f%s]", m, auc_val, threshold)
 })
 
 # Create the plot
 p <- ggplot(plot_data, aes(x = FPR, y = TPR, color = Model)) +
  geom_abline(linetype = "dashed", color = "gray70") +
  geom_path(size = 1.2) +
  coord_equal() +
  scale_color_manual(values = setNames(colors, models),
                     labels = legend_labels) +
  labs(title = sprintf("ROC Curves: Model Comparison (Dataset %d)", dataset_num),
       x = "False Positive Rate (1 – Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal(base_size = 14) +
  theme(
   plot.title = element_text(hjust = 0.5, size = 16),
   legend.position = c(0.7, 0.2),
   legend.background = element_rect(fill = "white", color = "gray80"),
   legend.title = element_blank(),
   panel.grid.minor = element_blank(),
   panel.border = element_rect(fill = NA, color = "black"),
   panel.background = element_rect(fill = "white")
  )
 
 # Save plots
 ggsave(file.path(output_dir, sprintf("roc_curves_dataset_%d.pdf", dataset_num)),
        plot = p, width = 10, height = 8, dpi = 300)
 ggsave(file.path(output_dir, sprintf("roc_curves_dataset_%d.png", dataset_num)),
        plot = p, width = 10, height = 8, dpi = 300)
 
 return(list(plot = p, data = plot_data))
}

# Function to create average ROC curve across all datasets
create_average_roc <- function(all_results) {
 # Initialize data frames for each model
 model_data <- list()
 
 # Define models and their display names
 models <- c("er", "causal", "lasso", "cr")
 model_names <- c("ER", "Causal ER", "Lasso", "CR")
 colors <- c("#1f78b4", "#e31a1c", "#33a02c", "#6a3d9a")
 
 # Common FPR grid for interpolation
 fpr_grid <- seq(0, 1, length.out = 100)
 
 # Collect and interpolate ROC curve data
 for(i in seq_along(all_results)) {
  if(!is.null(all_results[[i]])) {
   for(name in models) {
    if(!is.null(all_results[[i]][[name]]) && 
       !is.null(all_results[[i]][[name]]$roc)) {
     
     roc_obj <- all_results[[i]][[name]]$roc
     
     # Skip if we don't have enough points for interpolation
     if(length(roc_obj$sensitivities) < 2) {
      warning(sprintf("Skipping dataset %d, model %s: only one ROC point", i, name))
      next
     }
     
     # Get ROC curve points and ensure proper ordering
     fpr <- 1 - roc_obj$specificities
     tpr <- roc_obj$sensitivities
     ord <- order(fpr)
     fpr <- fpr[ord]
     tpr <- tpr[ord]
     
     # Interpolate
     tpr_interp <- approx(x = fpr, y = tpr, xout = fpr_grid, yleft = 0, yright = 1)$y
     
     if(is.null(model_data[[name]])) {
      model_data[[name]] <- list()
     }
     
     model_data[[name]] <- c(
      model_data[[name]],
      list(data.frame(
       FPR = fpr_grid,
       TPR = tpr_interp,
       AUC = as.numeric(auc(roc_obj))
      ))
     )
    }
   }
  }
 }
 
 # Create average ROC curves
 avg_roc_data <- data.frame()
 
 for(j in seq_along(models)) {
  model_name <- models[j]
  if(!is.null(model_data[[model_name]]) && length(model_data[[model_name]]) > 0) {
   model_curves <- do.call(rbind, model_data[[model_name]])
   
   # Calculate mean TPR for each FPR point
   avg_curve <- aggregate(TPR ~ FPR, data = model_curves, FUN = mean)
   mean_auc <- mean(sapply(model_data[[model_name]], function(x) x$AUC[1]))
   
   avg_roc_data <- rbind(avg_roc_data,
                         data.frame(
                          FPR = avg_curve$FPR,
                          TPR = avg_curve$TPR,
                          Model = model_names[j],
                          AUC = mean_auc
                         ))
  }
 }
 
 # Ensure sorted for proper step plotting
 avg_roc_data <- avg_roc_data[order(avg_roc_data$Model, avg_roc_data$FPR),]
 
 # Create the plot
 p <- ggplot(avg_roc_data, aes(x = FPR, y = TPR, color = Model)) +
  geom_abline(linetype = "dashed", color = "gray70") +
  geom_step(size = 1.2, direction = "vh") +
  scale_color_manual(values = setNames(colors, model_names),
                     labels = sprintf("%s [AUC = %.3f]",
                                      unique(avg_roc_data$Model),
                                      unique(avg_roc_data$AUC))) +
  labs(title = "Average ROC Curves Across All Datasets",
       x = "False Positive Rate (1–Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal(base_size = 14) +
  theme(
   plot.title = element_text(hjust = 0.5, size = 16),
   legend.position = c(0.7, 0.2),
   legend.background = element_rect(fill = "white", color = "gray80"),
   legend.title = element_blank(),
   panel.grid.minor = element_blank(),
   panel.border = element_rect(fill = NA, color = "black"),
   panel.background = element_rect(fill = "white")
  ) +
  coord_equal()
 
 # Save plots
 ggsave("output/average_roc_curves.pdf", plot = p, width = 10, height = 8, dpi = 300)
 ggsave("output/average_roc_curves.png", plot = p, width = 10, height = 8, dpi = 300)
 
 return(list(plot = p, data = avg_roc_data))
}

# Function to select features based on correlation with Z factors
select_features <- function(X_scaled_train, current_split_data, significant_z) {
 selected_features <- list()
 
 # For each significant Z factor
 for(z_idx in significant_z) {
  # Calculate correlations between X features and Z factor
  correlations <- cor(X_scaled_train, current_split_data$train$Z[,z_idx])
  
  # Handle NA values
  correlations[is.na(correlations)] <- 0
  
  # Select features with absolute correlation above threshold
  # Using mean correlation as adaptive threshold
  threshold <- mean(abs(correlations), na.rm = TRUE)
  selected <- which(abs(correlations) > threshold)
  
  if(length(selected) > 0) {
   selected_features[[as.character(z_idx)]] <- selected
  }
 }
 
 return(selected_features)
}

# Function to create composite matrices for training and testing
create_composite_matrices <- function(X_scaled_train, X_scaled_test, selected_features) {
 # Initialize composite matrices
 composite_train <- matrix(0, nrow = nrow(X_scaled_train), ncol = 0)
 composite_test <- matrix(0, nrow = nrow(X_scaled_test), ncol = 0)
 
 # For each Z factor
 for(z_idx in names(selected_features)) {
  features <- selected_features[[z_idx]]
  
  if(length(features) > 0) {
   # Add selected features to composite matrices
   composite_train <- cbind(composite_train, X_scaled_train[, features, drop = FALSE])
   composite_test <- cbind(composite_test, X_scaled_test[, features, drop = FALSE])
  }
 }
 
 # Convert to data frames
 composite_train <- as.data.frame(composite_train)
 composite_test <- as.data.frame(composite_test)
 
 # Add column names if missing
 if(ncol(composite_train) > 0 && is.null(colnames(composite_train))) {
  colnames(composite_train) <- paste0("Feature_", 1:ncol(composite_train))
  colnames(composite_test) <- colnames(composite_train)
 }
 
 return(list(
  train = composite_train,
  test = composite_test
 ))
}

# Function to fit final CR model
fit_final_cr_model <- function(X_scaled_train, X_scaled_test,
                               train_y, test_y,
                               selected_features,
                               model_dir,      # new parameter
                               dataset_idx     # new parameter
) {
 # Create composite matrices
 composite_matrices <- create_composite_matrices(X_scaled_train, X_scaled_test, selected_features)
 
 if(ncol(composite_matrices$train) == 0) {
  cat("\nWarning: No features selected for CR model")
  return(list(roc = NULL, metrics = NULL, probs = NULL))
 }
 
 # Alpha tuning grid
 alpha_grid <- c(0.1, 0.5, 0.9, 1.0)
 best_auc <- -Inf
 best_model <- NULL
 best_alpha <- NULL
 best_probs <- NULL
 
 # Try different alpha values
 for(alpha in alpha_grid) {
  # Fit elastic net model with LOOCV
  cv_model <- cv.glmnet(
   x = as.matrix(composite_matrices$train),
   y = train_y,
   family = "binomial",
   alpha = alpha,
   nfolds = nrow(composite_matrices$train)  # True LOOCV
  )
  
  # Get predictions
  probs <- predict(
   cv_model,
   newx = as.matrix(composite_matrices$test),
   s = "lambda.min",
   type = "response"
  )
  
  probs <- as.numeric(probs)
  
  # Calculate AUC
  current_roc <- tryCatch({
   roc(test_y, probs, quiet = TRUE)
  }, error = function(e) NULL)
  
  if(!is.null(current_roc)) {
   current_auc <- as.numeric(auc(current_roc))
   if(current_auc > best_auc) {
    best_auc <- current_auc
    best_model <- cv_model
    best_alpha <- alpha
    best_probs <- probs
   }
  }
 }
 
 if(is.null(best_model)) {
  return(list(roc = NULL, metrics = NULL, probs = NULL))
 }
 
 # Create ROC curve with best model
 cr_roc <- tryCatch({
  roc(test_y, best_probs, quiet = TRUE)
 }, error = function(e) {
  cat("\nError creating ROC curve for CR model:", e$message)
  NULL
 })
 
 # Calculate metrics with multiple threshold options
 thresholds <- c(0.3, 0.4, 0.5, 0.6, 0.7)
 metrics_list <- lapply(thresholds, function(t) {
  calculate_model_metrics(test_y, best_probs, threshold = t)
 })
 
 # Find best threshold based on F1 score
 f1_scores <- sapply(metrics_list, function(m) m$f1_score)
 best_threshold_idx <- which.max(f1_scores)
 optimal_threshold <- thresholds[best_threshold_idx]
 
 cr_metrics <- metrics_list[[best_threshold_idx]]
 
 cat(sprintf("\nBest alpha: %.2f, Best AUC: %.3f\n", best_alpha, best_auc))
 
 # Save coefficients
 if(!is.null(best_model)) {
  # pull out the lambda.min coefficients as a dense matrix
  coef_mat <- as.matrix(coef(best_model, s = "lambda.min"))
  
  # turn into a data.frame
  all_coefs <- data.frame(
   Feature     = rownames(coef_mat),
   Coefficient = coef_mat[,1],
   row.names   = NULL,
   stringsAsFactors = FALSE
  )
  
  # save 'em all
  write.csv(all_coefs,
            file = file.path(model_dir, sprintf("cr_all_coefficients_dataset_%d.csv", dataset_idx)),
            row.names = FALSE)
  
  # now pick only the nonzero ones and sort by absolute strength
  top_coefs <- subset(all_coefs, Coefficient != 0)
  top_coefs <- top_coefs[order(-abs(top_coefs$Coefficient)), ]
  
  # save the tops
  write.csv(top_coefs,
            file = file.path(model_dir, sprintf("cr_top_coefficients_dataset_%d.csv", dataset_idx)),
            row.names = FALSE)
  
  # optionally print the top 10
  cat("\nTop CR features by coefficient magnitude:\n")
  print(head(top_coefs, 10))
 }
 
 return(list(
  roc = cr_roc,
  metrics = cr_metrics,
  probs = best_probs,
  model = best_model,
  threshold = optimal_threshold,
  alpha = best_alpha,
  coefficients = list(all = all_coefs, top = top_coefs)  # Add coefficients to return value
 ))
}

# Function to run all models and return their results
run_models <- function(current_split_data, results, model_dir, dataset_idx) {
 # Initialize variables for model results
 er_roc <- er_metrics <- er_probs <- NULL
 causal_roc <- causal_metrics <- causal_probs <- NULL
 lasso_roc <- lasso_metrics <- lasso_probs <- NULL
 cr_roc <- cr_metrics <- cr_probs <- NULL
 
 # 1. ER Model
 er_model <- tryCatch({
  if(ncol(current_split_data$train$Z) == 0) {
   cat("\nWarning: No Z factors available for ER model")
   return(NULL)
  }
  
  # Fit on training data
  model <- glm(current_split_data$train$Y ~ ., 
               data = as.data.frame(current_split_data$train$Z), 
               family = "binomial")
  
  # Get predictions on test data
  probs <- predict(model, 
                   newdata = as.data.frame(current_split_data$test$Z), 
                   type = "response")
  
  if(any(is.na(probs)) || all(probs == probs[1])) {
   cat("\nWarning: Invalid predictions from ER model")
   return(NULL)
  }
  
  probs <- pmax(pmin(probs, 1), 0)
  list(model = model, probs = probs)
 }, error = function(e) {
  cat("\nError in ER model:", e$message)
  return(NULL)
 })
 
 if(!is.null(er_model)) {
  er_probs <- er_model$probs
  er_roc <- tryCatch({
   roc(current_split_data$test$Y, er_probs)
  }, error = function(e) {
   cat("\nError creating ROC curve for ER model:", e$message)
   NULL
  })
  er_metrics <- calculate_model_metrics(current_split_data$test$Y, er_probs)
 }
 
 # 2. Causal ER (dimensions 1,2,3,5)
 causal_dims <- c(1, 2, 3, 5)
 Z_causal_train <- current_split_data$train$Z[, causal_dims, drop = FALSE]
 Z_causal_test <- current_split_data$test$Z[, causal_dims, drop = FALSE]
 
 causal_model <- tryCatch({
  if(ncol(Z_causal_train) == 0) {
   cat("\nWarning: No causal dimensions available")
   return(NULL)
  }
  
  model <- glm(current_split_data$train$Y ~ ., 
               data = as.data.frame(Z_causal_train), 
               family = "binomial")
  probs <- predict(model, 
                   newdata = as.data.frame(Z_causal_test), 
                   type = "response")
  
  if(any(is.na(probs)) || all(probs == probs[1])) {
   cat("\nWarning: Invalid predictions from Causal ER model")
   return(NULL)
  }
  
  probs <- pmax(pmin(probs, 1), 0)
  list(model = model, probs = probs)
 }, error = function(e) {
  cat("\nError in Causal ER model:", e$message)
  return(NULL)
 })
 
 if(!is.null(causal_model)) {
  causal_probs <- causal_model$probs
  causal_roc <- tryCatch({
   roc(current_split_data$test$Y, causal_probs)
  }, error = function(e) {
   cat("\nError creating ROC curve for Causal ER model:", e$message)
   NULL
  })
  causal_metrics <- calculate_model_metrics(current_split_data$test$Y, causal_probs)
 }
 
 # 3. Lasso with interactions
 Z_interact_train <- as.data.frame(Z_causal_train)
 Z_interact_test <- as.data.frame(Z_causal_test)
 
 # Create interaction terms
 for(j in 1:(length(causal_dims)-1)) {
  for(k in (j+1):length(causal_dims)) {
   Z_interact_train[,paste0("Z", causal_dims[j], "_Z", causal_dims[k])] <- 
    Z_causal_train[,j] * Z_causal_train[,k]
   Z_interact_test[,paste0("Z", causal_dims[j], "_Z", causal_dims[k])] <- 
    Z_causal_test[,j] * Z_causal_test[,k]
  }
 }
 
 cv_lasso <- cv.glmnet(as.matrix(Z_interact_train), 
                       current_split_data$train$Y, 
                       family = "binomial", 
                       alpha = 1)
 lasso_probs <- predict(cv_lasso, 
                        newx = as.matrix(Z_interact_test), 
                        s = "lambda.min", 
                        type = "response")
 
 if(!all(lasso_probs == lasso_probs[1])) {
  lasso_roc <- tryCatch({
   roc(current_split_data$test$Y, lasso_probs)
  }, error = function(e) {
   cat("\nError creating ROC curve for Lasso model:", e$message)
   NULL
  })
  lasso_metrics <- calculate_model_metrics(current_split_data$test$Y, lasso_probs)
 }
 
 # 4. Composite Regression
 cr_results <- fit_cr_model(current_split_data, results, causal_model$model, dataset_idx, model_dir)
 
 # Return all model results
 return(list(
  er = list(roc = er_roc, metrics = er_metrics, probs = er_probs),
  causal = list(roc = causal_roc, metrics = causal_metrics, probs = causal_probs),
  lasso = list(roc = lasso_roc, metrics = lasso_metrics, probs = lasso_probs),
  cr = cr_results
 ))
}

# Helper function for CR model fitting
fit_cr_model <- function(split_data, results, causal_model, dataset_num, model_dir) {
 tryCatch({
  # Get significant causal Z factors
  causal_coef <- coef(causal_model)[-1]  # Remove intercept
  significant_z <- which(!is.na(causal_coef) & abs(causal_coef) > mean(abs(causal_coef), na.rm = TRUE))
  
  if(length(significant_z) == 0) {
   cat("\nNo significant Z factors found")
   return(NULL)
  }
  
  # Scale features
  X_scaled_train <- scale(split_data$train$X)
  X_scaled_test <- scale(split_data$test$X)
  X_scaled_train[is.na(X_scaled_train)] <- 0
  X_scaled_test[is.na(X_scaled_test)] <- 0
  
  # Feature selection
  selected_features <- select_features(X_scaled_train, split_data, significant_z)
  
  if(length(selected_features) == 0) {
   cat("\nNo features selected")
   return(NULL)
  }
  
  # Create summary of selected features
  create_z_factor_summary(selected_features, X_scaled_train, model_dir, dataset_num)
  
  # LOOCV for model evaluation
  n <- nrow(X_scaled_train)
  loocv_aucs <- numeric(n)
  loocv_predictions <- numeric(n)
  
  cat("\nPerforming LOOCV for dataset", dataset_num, "...\n")
  
  for(i in 1:n) {
   if(i %% 10 == 0) cat("Processing fold", i, "of", n, "\n")
   
   # Create LOOCV split
   train_idx <- setdiff(1:n, i)
   
   # Create LOOCV matrices
   X_loocv_train <- X_scaled_train[train_idx, ]
   X_loocv_test <- X_scaled_train[i, , drop=FALSE]
   y_loocv_train <- split_data$train$Y[train_idx]
   y_loocv_test <- split_data$train$Y[i]
   
   # Feature selection for this fold
   loocv_features <- select_features(X_loocv_train, 
                                     list(train=list(Z=split_data$train$Z[train_idx,])), 
                                     significant_z)
   
   # Create composite matrices
   composite_loocv <- create_composite_matrices(X_loocv_train, X_loocv_test, loocv_features)
   
   if(ncol(composite_loocv$train) > 0) {
    # Fit model
    cv_model <- cv.glmnet(
     x = as.matrix(composite_loocv$train),
     y = y_loocv_train,
     family = "binomial",
     alpha = 0.5,
     nfolds = 5
    )
    
    # Get prediction
    pred <- predict(cv_model, 
                    newx = as.matrix(composite_loocv$test),
                    s = "lambda.min",
                    type = "response")
    
    pred <- as.numeric(pred)
    loocv_predictions[i] <- pred
    
    # Calculate AUC for this fold
    if(!is.na(pred) && !is.na(y_loocv_test)) {
     loocv_aucs[i] <- tryCatch({
      roc_obj <- roc(y_loocv_test, pred, quiet = TRUE)
      as.numeric(auc(roc_obj))
     }, error = function(e) NA)
    }
   }
  }
  
  # Calculate and save LOOCV results
  mean_loocv_auc <- mean(loocv_aucs, na.rm = TRUE)
  sd_loocv_auc <- sd(loocv_aucs, na.rm = TRUE)
  
  cat(sprintf("\nLOOCV Results for dataset %d:", dataset_num))
  cat(sprintf("\nMean AUC: %.3f (SD: %.3f)\n", mean_loocv_auc, sd_loocv_auc))
  
  # Save LOOCV results
  loocv_results <- data.frame(
   Fold = 1:n,
   True_Label = split_data$train$Y,
   Predicted_Prob = loocv_predictions,
   AUC = loocv_aucs
  )
  
  write.csv(
   loocv_results,
   file = file.path(model_dir, sprintf("loocv_results_dataset_%d.csv", dataset_num)),
   row.names = FALSE
  )
  
  # Use LOOCV AUC for the final model metrics
  cr_model <- fit_final_cr_model(
   X_scaled_train, X_scaled_test,
   split_data$train$Y, split_data$test$Y,
   selected_features,
   model_dir,       # pass in model_dir
   dataset_num      # pass in dataset_num
  )
  
  if(!is.null(cr_model)) {
   cr_model$metrics$auc <- mean_loocv_auc  # Use LOOCV AUC
   cr_model$loocv_results <- list(
    predictions = loocv_predictions,
    aucs = loocv_aucs,
    mean_auc = mean_loocv_auc,
    sd_auc = sd_loocv_auc
   )
  }
  
  return(cr_model)
  
 }, error = function(e) {
  cat(sprintf("\nError in CR model fitting for dataset %d: %s\n", dataset_num, e$message))
  return(NULL)
 })
}

# Function to create metrics comparison
create_metrics_comparison <- function(all_results) {
 tryCatch({
  # Create the base data frame
  metrics_comparison <- data.frame(
   Dataset = rep(1:length(all_results), each = 4),
   Model = rep(c("ER", "Causal ER", "Lasso", "CR"), length(all_results)),
   Test_AUC = NA,
   Test_Accuracy = NA,
   Test_Precision = NA,
   Test_Recall = NA,
   Test_F1_Score = NA,
   Test_MSE = NA,
   stringsAsFactors = FALSE
  )
  
  # Process each dataset's results
  for(i in seq_along(all_results)) {
   if(is.null(all_results[[i]])) {
    cat(sprintf("\nSkipping dataset %d - no results available", i))
    next
   }
   
   base_idx <- (i-1)*4
   model_names <- c("er", "causal", "lasso", "cr")
   
   # Process each model's results
   for(j in seq_along(model_names)) {
    tryCatch({
     model_results <- all_results[[i]][[model_names[j]]]
     
     if(!is.null(model_results) && !is.null(model_results$probs)) {
      idx <- base_idx + j
      
      # Calculate AUC if ROC curve exists
      auc_value <- NA
      if(!is.null(model_results$roc)) {
       auc_value <- tryCatch({
        as.numeric(auc(model_results$roc))
       }, error = function(e) {
        cat(sprintf("\nError calculating AUC for dataset %d, model %s: %s", 
                    i, model_names[j], e$message))
        NA
       })
      }
      
      # Get other metrics
      metrics <- c(
       auc_value,
       if(!is.null(model_results$metrics$accuracy)) model_results$metrics$accuracy else NA,
       if(!is.null(model_results$metrics$precision)) model_results$metrics$precision else NA,
       if(!is.null(model_results$metrics$recall)) model_results$metrics$recall else NA,
       if(!is.null(model_results$metrics$f1_score)) model_results$metrics$f1_score else NA,
       if(!is.null(model_results$metrics$mse)) model_results$metrics$mse else NA
      )
      
      # Update metrics in the data frame
      metrics_comparison[idx, c("Test_AUC", "Test_Accuracy", "Test_Precision", 
                                "Test_Recall", "Test_F1_Score", "Test_MSE")] <- metrics
     }
    }, error = function(e) {
     cat(sprintf("\nError processing metrics for dataset %d, model %s: %s\n", 
                 i, model_names[j], e$message))
    })
   }
  }
  
  return(metrics_comparison)
  
 }, error = function(e) {
  cat("\nError in create_metrics_comparison:", e$message, "\n")
  return(NULL)
 })
}

# Function to compare and analyze multiple datasets
analyze_datasets <- function(base_file_path, params, n_datasets = 5) {
 # Create output directory
 dir.create("output", showWarnings = FALSE)
 
 # Initialize results storage
 all_results <- list()
 
 # Generate file paths
 file_paths <- sapply(1:n_datasets, function(i) {
  file_path <- sprintf("%s_%d.csv", base_file_path, i)
  if(!file.exists(file_path)) {
   stop(sprintf("Dataset file not found: %s", file_path))
  }
  return(file_path)
 })
 
 # Process each dataset
 for(dataset_idx in 1:n_datasets) {
  cat(sprintf("\n\n========= Processing dataset %d of %d =========\n", dataset_idx, n_datasets))
  
  # Create directories first
  dataset_dir <- file.path("output", sprintf("dataset_%d", dataset_idx))
  if(dir.exists(dataset_dir)) {
   unlink(dataset_dir, recursive = TRUE)  # Clean up existing directory
  }
  dir.create(dataset_dir, showWarnings = FALSE)
  model_dir <- file.path(dataset_dir, "model_features")
  dir.create(model_dir, showWarnings = FALSE)
  
  # Process the dataset
  tryCatch({
   # Process dataset
   cat(sprintf("\nReading and processing dataset %d...\n", dataset_idx))
   results <- process_dataset(
    file_path = file_paths[dataset_idx],
    delta = params$delta,
    beta_est = params$beta_est,
    var_threshold = params$var_threshold,
    cor_threshold = params$cor_threshold,
    n_preselected = params$n_preselected,
    preserve_correlated = params$preserve_correlated,
    use_multi_dimensions = params$use_multi_dimensions
   )
   
   # Create a new split_data for each dataset
   cat(sprintf("\nSplitting dataset %d into train/test sets...\n", dataset_idx))
   current_split_data <- split_dataset(results$X, results$Y, results$Z)
   
   # Run models and get results
   model_results <- run_models(current_split_data, results, model_dir, dataset_idx)
   
   if(!is.null(model_results)) {
    cat(sprintf("\nSuccessfully processed dataset %d\n", dataset_idx))
    all_results[[dataset_idx]] <- model_results
    
    # Create ROC curves
    tryCatch({
     roc_results <- create_roc_curves(model_results, dataset_idx, dataset_dir)
     all_results[[dataset_idx]]$roc_results <- roc_results
    }, error = function(e) {
     cat(sprintf("\nError creating ROC curves for dataset %d: %s\n", dataset_idx, e$message))
    })
   }
   
  }, error = function(e) {
   cat(sprintf("\nError processing dataset %d: %s\n", dataset_idx, e$message))
  })
  
  cat(sprintf("\n========= Completed dataset %d of %d =========\n", dataset_idx, n_datasets))
 }
 
 # Create metrics comparison
 metrics_comparison <- create_metrics_comparison(all_results)
 
 # Create comprehensive performance summary
 cat("\n=== Model Performance Summary ===\n")
 
 # Calculate mean performance across datasets
 mean_performance <- aggregate(
  cbind(Test_AUC, Test_Accuracy, Test_Precision, Test_Recall, Test_F1_Score, Test_MSE) ~ Model,
  data = metrics_comparison,
  FUN = function(x) round(mean(x, na.rm = TRUE), 3)
 )
 
 # Calculate standard deviations
 sd_performance <- aggregate(
  cbind(Test_AUC, Test_Accuracy, Test_Precision, Test_Recall, Test_F1_Score, Test_MSE) ~ Model,
  data = metrics_comparison,
  FUN = function(x) {
   if(sum(!is.na(x)) <= 1) {
    return(0)  # Return 0 instead of NA for single dataset
   } else {
    return(round(sd(x, na.rm = TRUE), 3))
   }
  }
 )
 
 # Print summary
 cat("\nMean Performance Metrics:\n")
 print(mean_performance)
 cat("\nStandard Deviations:\n")
 print(sd_performance)
 
 # Find best performing model for each metric
 metrics <- c("Test_AUC", "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1_Score")
 cat("\nBest Performing Models:\n")
 for(metric in metrics) {
  best_idx <- which.max(mean_performance[[metric]])
  cat(sprintf("%s: %s (%.3f ± %.3f)\n",
              gsub("Test_", "", metric),  # Remove "Test_" prefix for display
              mean_performance$Model[best_idx],
              mean_performance[[metric]][best_idx],
              sd_performance[[metric]][best_idx]))
 }
 
 # Save summary to file
 summary_df <- data.frame(
  Model = mean_performance$Model,
  Metric = rep(names(mean_performance)[-1], each = nrow(mean_performance)),
  Mean = unlist(mean_performance[,-1]),
  SD = unlist(sd_performance[,-1])
 )
 
 write.csv(summary_df, "output/model_performance_summary.csv", row.names = FALSE)
 
 # Create summary plots
 pdf("output/performance_plots.pdf")
 for(metric in metrics) {
  boxplot(as.formula(paste(metric, "~ Model")), 
          data = metrics_comparison,
          main = paste("Distribution of", gsub("Test_", "", metric)),
          ylab = gsub("Test_", "", metric))
 }
 dev.off()
 
 # Create and save average ROC curve
 cat("\nCreating average ROC curves...\n")
 average_roc_results <- create_average_roc(all_results)
 
 # Save average ROC data with more details
 avg_roc_data <- average_roc_results$data %>%
  group_by(Model) %>%
  summarise(
   Mean_AUC = mean(AUC, na.rm = TRUE),
   SD_AUC = sd(AUC, na.rm = TRUE),
   Mean_TPR = mean(TPR, na.rm = TRUE),
   SD_TPR = sd(TPR, na.rm = TRUE)
  )
 
 write.csv(avg_roc_data, "output/average_roc_summary.csv", row.names = FALSE)
 write.csv(average_roc_results$data, "output/average_roc_detailed.csv", row.names = FALSE)
 
 # Print average AUC values
 cat("\nAverage AUC values across datasets:\n")
 print(avg_roc_data)
 
 # Return results
 return(list(
  all_results = all_results,
  metrics_comparison = metrics_comparison,
  mean_performance = mean_performance,
  sd_performance = sd_performance
 ))
}

# Set analysis parameters
params <- list(
 delta = 0.1,
 beta_est = "Lasso",
 var_threshold = 0.01,
 cor_threshold = 0.95,
 n_preselected = 1000,
 preserve_correlated = TRUE,
 use_multi_dimensions = TRUE
)

# Define base file path

## ClinGeneDiv
# base_file_path <- "C:/Users/omatu/OneDrive/Desktop/R_UdeM/UdeM/ER/gene_RF_imputed_Oct30_sklearnImp_Asthma_treatment_modified_Apr10_merged_with_div"

## ClinGene
# base_file_path <- "C:/Users/omatu/OneDrive/Desktop/R_UdeM/UdeM/ER/gene_RF_imputed_Oct30_sklearnImp_Asthma_treatment_modified_Apr10"

# #Gene
# base_file_path <- "C:/Users/omatu/OneDrive/Desktop/R_UdeM/UdeM/ER/gene_RF_imputed_Oct30_sklearnImp_NoClinical"

## Clin
base_file_path <- "C:/Users/omatu/OneDrive/Desktop/R_UdeM/UdeM/ER/ClinDiv_May6_cleaned"

##GenDiv
# base_file_path <- "C:/Users/omatu/OneDrive/Desktop/R_UdeM/UdeM/ER/GenDiv_May6"



# Run the analysis
results <- analyze_datasets(base_file_path, params)



# ROC
# — after your analysis run — 
# res <- analyze_datasets(base_file_path, params)

library(pROC)
library(ggplot2)

all_results <- results$all_results


models     <- c("er", "causal", "lasso", "cr")
model_lbl  <- c("ER", "Causal ER", "Lasso", "CR")

avg_rocs <- lapply(seq_along(models), function(i) {
 m     <- models[i]
 label <- model_lbl[i]
 
 # pull out only the valid ROC objects for this model
 rocs <- lapply(all_results, function(x) {
  if (!is.null(x[[m]]) && !is.null(x[[m]]$roc)) x[[m]]$roc else NULL
 })
 rocs <- Filter(function(r) !is.null(r) && length(r$sensitivities) > 1, rocs)
 if (length(rocs)==0) return(NULL)
 
 # build the union of all FPR breakpoints
 all_fpr <- sort(unique(unlist(lapply(rocs, function(roc) 1-roc$specificities))))
 
 # at each of those FPRs, get each curve's step‐value (last carried‐forward TPR)
 tpr_mat <- sapply(rocs, function(roc) {
  fpr <- 1-roc$specificities
  tpr <- roc$sensitivities
  sapply(all_fpr, function(x0) {
   ii <- which(fpr <= x0)
   if (length(ii)==0) return(0)
   max(tpr[ii])
  })
 })
 
 # average across columns
 data.frame(
  FPR   = all_fpr,
  TPR   = rowMeans(tpr_mat),
  Model = label,
  stringsAsFactors = FALSE
 )
})
avg_rocs <- avg_rocs[!sapply(avg_rocs, is.null)]
avg_df   <- do.call(rbind, avg_rocs)

# now plot with real staircase style
ggplot(avg_df, aes(x=FPR, y=TPR, color=Model)) +
 geom_abline(linetype="dashed", color="gray70") +
 geom_step(direction="vh", size=1.2) +
 coord_equal() +
 labs(
  title = "Average ROC Curves Across All Datasets",
  x = "False Positive Rate (1 – Specificity)",
  y = "True Positive Rate (Sensitivity)"
 ) +
 theme_minimal(base_size=14) +
 theme(
  legend.position = c(0.7, 0.2),
  legend.title    = element_blank()
 )


# ggsave("output/average_roc_curve_ClinDiv_final_wo_cohort_columns.png", width=10, height=8, dpi=300)

