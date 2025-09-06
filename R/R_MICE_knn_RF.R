# Install MICE package
#install.packages("mice")

# Explanation of parameters:
# m = 5: Number of multiple imputations (5 different imputed datasets)
# maxit = 50: Maximum number of iterations per imputation
# method = 'pmm': Use predictive mean matching for numeric values
# seed = 111: Set a seed for reproducibility

# m: Number of imputed datasets to create. Typically set between 5-10.
# maxit: Number of iterations to run for each imputation.
# method: Specifies the imputation method for different types of variables.
# pmm (Predictive Mean Matching): Used for numeric variables.
# logreg: Used for binary variables.
# polyreg: Used for polytomous (multiclass) variables.
# norm: Linear regression for continuous variables.

# Run each code block (if separeted as #-----------------------) separately

# Load MICE library
library(mice)

# install.packages("VIM")
library(VIM)


data_clinical_sel_var_all_gx_ubiop_Oct22 <- read.csv("sel_var_all_gx_ubiop_class_exacerb_data_clinical_for_imputation_Oct22.csv", header = TRUE, check.names = FALSE)

sum(is.na(data_clinical_sel_var_all_gx_ubiop_Oct22))

# Convert Exacerbation.Outcome to a factor
data_clinical_sel_var_all_gx_ubiop_Oct22$Exacerbation.Outcome <- as.factor(data_clinical_sel_var_all_gx_ubiop_Oct22$Exacerbation.Outcome)


# Define the columns to be scaled (only those you plan to impute)
columns_to_rf <- c("Asthma.Control.Test.ACT.Childhood.Total", 
                      "Screening.eosinophils", 
                      "Screening.neutrophils", 
                      "PAQLQ.all.questions.average",
                      "Total.lung.capacity.TLC.Predicted.persentage"
                      )

# Simplify all column names (in case there are special characters or spaces)
colnames(data_clinical_sel_var_all_gx_ubiop_Oct22) <- make.names(colnames(data_clinical_sel_var_all_gx_ubiop_Oct22), unique = TRUE)

# Set up MICE imputation methods
methods <- make.method(data_clinical_sel_var_all_gx_ubiop_Oct22)
methods[] <- ""  # Set all columns to "" (no imputation by default)

# Specify random forest imputation for selected columns
for (col in columns_to_rf) {
 methods[col] <- "rf"
}

# # Define the variables to impute using kNN
# variables_to_impute <- c("Asthma.Control.Test.ACT.Childhood.Total", 
#                          "Screening.eosinophils", 
#                          "Screening.neutrophils", 
#                          "PAQLQ.all.questions.average")
# 
# # Define the range of k values to test
# k_values <- c(3, 5, 7, 10)
# 
# 
# # Loop through each variable and each k value
# for (variable in variables_to_impute) {
#  for (k_val in k_values) {
#   
#   # Perform kNN imputation for the specific variable with current k value
#   data_clinical_imputed <- kNN(data_clinical_sel_var_all_gx_ubiop_Oct22,
#                                variable = variable, 
#                                k = k_val)
#   
#   # Plot density of imputed vs. original data for comparison
#   plot(density(data_clinical_imputed[[variable]], na.rm = TRUE), 
#        main = paste("Density Plot for", variable, "with k =", k_val),
#        col = "red", lwd = 2)
#   lines(density(data_clinical_sel_var_all_gx_ubiop_Oct22[[variable]], na.rm = TRUE), 
#         col = "black", lty = 2, lwd = 2)
#   legend("topright", legend = c("Imputed", "Observed"), col = c("red", "black"), lty = 1:2, lwd = 2)
#  }
# }


# # Simplify all column names
# colnames(data_clinical_sel_var_all_gx_ubiop_Oct22) <- make.names(colnames(data_clinical_sel_var_all_gx_ubiop_Oct22), unique = TRUE)
# 
# # Set up MICE imputation for remaining variables
# methods <- make.method(data_clinical_sel_var_all_gx_ubiop_Oct22)
# methods[] <- ""  # Set all columns to "" (no imputation by default)

# Set the imputation methods for specific columns
methods["Standard.Flow.Rate"] <- "pmm"
methods["Total.Lung.capacity.TLC.Predicted.L"] <- "pmm"
methods["FEV1.Post.Salbutamol...L"] <- "pmm"
methods["FEV1.Pre.Salbutamol...L"] <- "pmm"
methods["Scale.MARS.Total"] <- "pmm"
methods["Oral.CS.dose"] <- "pmm"
methods["Exacerbation.Outcome"] <- "logreg"


# # Ensure that these columns are NOT re-imputed by MICE
# methods["Asthma.Control.Test.ACT.Childhood.Total"] <- ""  # Exclude from MICE since it was already imputed by rf
# methods["Screening.eosinophils"] <- ""
# methods["Screening.neutrophils"] <- ""
# methods["PAQLQ.all.questions.average"] <- ""

# Double-check length consistency
print(length(methods))  # Should match ncol(data_clinical_sel_var_all_gx_ubiop_Oct22)
print(ncol(data_clinical_sel_var_all_gx_ubiop_Oct22))  # Should match length(methods)


# Perform the MICE imputation for the remaining variables
imputed_data <- mice(data_clinical_sel_var_all_gx_ubiop_Oct22, 
                     m = 5, 
                     maxit = 100, 
                     method = methods, 
                     seed = 123, 
                     printFlag = TRUE)




# Extract and inspect the imputed data
complete_data <- complete(imputed_data, 1)

# Check if the missing values were imputed
sum(is.na(complete_data))

write.csv(complete_data, "data_clinical_sel_var_all_gx_ubiop_Oct30_imputed_rf_1_vv.csv", row.names = TRUE)


#------------------------------------------------------------------------------------------------------------

# Jan 30, 2025 
# imputation of beta diversity metrics for 28 patients


beta_div <- read.csv("beta_div_metrics_MICE_agg_filtered.csv", header = TRUE, check.names = FALSE)

sum(is.na(beta_div))

# Define the columns to be scaled (only those you plan to impute)
columns_to_pmm <- c("braycurtis_mean", 
                   "braycurtis_median", 
                   "braycurtis_max", 
                   "jaccard_mean",
                   "jaccard_median",
                   "jaccard_max",
                   "canberra_mean", 
                   "canberra_median", 
                   "canberra_max",
                   "aitchison_mean", 
                   "aitchison_median", 
                   "aitchison_max"
                   
)


# Simplify all column names (in case there are special characters or spaces)
colnames(beta_div) <- make.names(colnames(beta_div), unique = TRUE)


# Set up MICE imputation methods
methods <- make.method(beta_div)
methods[] <- ""  # Set all columns to "" (no imputation by default)

# Specify random forest imputation for selected columns
# Specify PMM (Predictive Mean Matching) for selected columns
for (col in columns_to_pmm) {
 methods[col] <- "pmm"
}

# Double-check length consistency
print(length(methods))  # Should match ncol(data_clinical_sel_var_all_gx_ubiop_Oct22)
print(ncol(beta_div))  # Should match length(methods)


# Perform the MICE imputation for the remaining variables
imputed_data <- mice(beta_div, 
                     m = 5, 
                     maxit = 100, 
                     method = methods, 
                     seed = 123, 
                     printFlag = TRUE)




# Extract and inspect the imputed data
complete_data <- complete(imputed_data, 5)

# Check if the missing values were imputed
sum(is.na(complete_data))

write.csv(complete_data, "beta_div_imputed_pmm5_Jan30.csv", row.names = TRUE)



#-----------------------------------------------------------------------------
# alpha diversity

# Load MICE library
library(mice)

# install.packages("VIM")
library(VIM)


alpha_div <- read.csv("alpha_div_MICE_filtered.csv", header = TRUE, check.names = FALSE)

sum(is.na(alpha_div))


# Define the columns to be scaled (only those you plan to impute)
columns_to_pmm <- c("chao1", 
                    "shannon", 
                    "observed_otus"
                    
)


# Simplify all column names (in case there are special characters or spaces)
colnames(alpha_div) <- make.names(colnames(alpha_div), unique = TRUE)


# Set up MICE imputation methods
methods <- make.method(alpha_div)
methods[] <- ""  # Set all columns to "" (no imputation by default)

# Specify random forest imputation for selected columns
# Specify PMM (Predictive Mean Matching) for selected columns
for (col in columns_to_pmm) {
 methods[col] <- "pmm"
}

# Double-check length consistency
print(length(methods))  # Should match ncol(data_clinical_sel_var_all_gx_ubiop_Oct22)
print(ncol(alpha_div))  # Should match length(methods)


# Perform the MICE imputation for the remaining variables
imputed_data <- mice(alpha_div, 
                     m = 5, 
                     maxit = 100, 
                     method = methods, 
                     seed = 123, 
                     printFlag = TRUE)




# Extract and inspect the imputed data
complete_data <- complete(imputed_data, 5)

# Check if the missing values were imputed
sum(is.na(complete_data))

write.csv(complete_data, "alpha_div_imputed_pmm5_Jan30.csv", row.names = TRUE)




# Jan 30, 2025 
# imputation of beta diversity metrics for 28 patients


beta_div <- read.csv("beta_div_metrics_MICE_agg_filtered.csv", header = TRUE, check.names = FALSE)

sum(is.na(beta_div))

# Define the columns to be scaled (only those you plan to impute)
columns_to_pmm <- c("braycurtis_mean", 
                    "braycurtis_median", 
                    "braycurtis_max", 
                    "jaccard_mean",
                    "jaccard_median",
                    "jaccard_max",
                    "canberra_mean", 
                    "canberra_median", 
                    "canberra_max",
                    "aitchison_mean", 
                    "aitchison_median", 
                    "aitchison_max"
                    
)


# Simplify all column names (in case there are special characters or spaces)
colnames(beta_div) <- make.names(colnames(beta_div), unique = TRUE)


# Set up MICE imputation methods
methods <- make.method(beta_div)
methods[] <- ""  # Set all columns to "" (no imputation by default)

# Specify random forest imputation for selected columns
# Specify PMM (Predictive Mean Matching) for selected columns
for (col in columns_to_pmm) {
 methods[col] <- "pmm"
}

# Double-check length consistency
print(length(methods))  # Should match ncol(data_clinical_sel_var_all_gx_ubiop_Oct22)
print(ncol(beta_div))  # Should match length(methods)


# Perform the MICE imputation for the remaining variables
imputed_data <- mice(beta_div, 
                     m = 5, 
                     maxit = 100, 
                     method = methods, 
                     seed = 123, 
                     printFlag = TRUE)




# Extract and inspect the imputed data
complete_data <- complete(imputed_data, 5)

# Check if the missing values were imputed
sum(is.na(complete_data))

write.csv(complete_data, "beta_div_imputed_pmm5_Jan30.csv", row.names = TRUE)



#-----------------------------------------------------------------------------
#Feb 13, 2025
# otutab_transpose

install.packages("car")
install.packages("VIM") 
install.packages("future")  # Install future package
install.packages("missForest")  # Fastest method for large datasets


library(missForest)
library(future)  
# Load MICE library
library(mice)
# install.packages("VIM")
library(VIM)


otutab_transp_div <- read.csv("otutab_transposed_missing_patients_MICE.csv", header = TRUE, check.names = FALSE)

# Convert character columns to numeric (if applicable)
otutab_transp_div[] <- lapply(otutab_transp_div, function(x) 
 if(is.character(x)) as.numeric(as.character(x)) else x)

# Check structure and missing values
str(otutab_transp_div)
cat("Total missing values before imputation:", sum(is.na(otutab_transp_div)), "\n")

# 🚀 **Option 1: Fastest Alternative → Use `missForest()`**
cat("Using missForest for fast imputation...\n")
imputed_data <- missForest(otutab_transp_div)  # Automatically imputes all missing values

# Extract imputed dataset
complete_data <- imputed_data$ximp


# Check for missing values after imputation
cat("Total missing values after imputation:", sum(is.na(complete_data)), "\n")

# Save the imputed dataset
write.csv(complete_data, "otutab_transp_div_imputed_fast.csv", row.names = FALSE)

cat("Imputation completed successfully and saved as 'otutab_transp_div_imputed_fast.csv'\n")

#------------------------------------------------------------------------------

# Feb 25 kNN MICE for ASV
# Not worthy as well as RF or pmm, all are too slow,
#becuase we have 34205 values for imputation
# Let's try MICE fast

# Install required packages
install.packages("mice")  # MICE package for imputation
install.packages("VIM")   # VIM contains kNN imputation functions

# Load libraries
library(mice)
library(VIM)

# Install miceFast if not already installed
install.packages("miceFast")

# Load the package
library(miceFast)

ls("package:miceFast")


# Verify if the package is correctly loaded
sessionInfo()


# Load the dataset
otutab_transp_div <- read.csv("otutab_transposed_missing_patients_MICE.csv", header = TRUE, check.names = FALSE)


# Remove non-numeric columns (e.g., subject_id)
numeric_cols <- which(sapply(otutab_transp_div, is.numeric))  # Get indices of numeric columns

if (length(numeric_cols) < 2) {
 stop("🚨 Not enough numeric columns available for imputation!")
}

# Subset only numeric columns and convert to matrix
otutab_matrix <- as.matrix(otutab_transp_div[, numeric_cols])

# Dynamically select the **actual** first two columns from the filtered matrix
columns_to_impute <- seq_len(min(2, ncol(otutab_matrix)))  # First two available numeric columns

# Count missing values before imputation
total_missing_before <- sum(is.na(otutab_matrix))
cat("🔹 Total missing values BEFORE imputation:", total_missing_before, "\n")

# Run `fill_NA_N()` on each column separately
for (col_idx in columns_to_impute) {
 # Confirm `col_idx` is within matrix dimensions
 if (col_idx > ncol(otutab_matrix)) {
  cat("Skipping column", col_idx, "- index is out of bounds!\n")
  next  # Skip to next iteration
 }
 
 missing_values <- sum(is.na(otutab_matrix[, col_idx]))  # Count missing values in the column
 
 if (missing_values > 0) {
  cat("\n Imputing column index:", col_idx, "with", missing_values, "missing values...\n")
  
  # Recalculate independent variables dynamically
  remaining_cols <- which(colSums(!is.na(otutab_matrix)) > 0)  # Recalculate non-empty columns
  posit_x <- setdiff(remaining_cols, col_idx)  # Ensure predictors exclude target column
  
  # Print Debugging Information
  cat("Debugging - Column being imputed (posit_y):", col_idx, "\n")
  cat("Debugging - Predictors (posit_x):", paste(posit_x, collapse = ", "), "\n")
  cat("Debugging - Matrix dimensions:", dim(otutab_matrix), "\n")
  
  if (length(posit_x) == 0) {
   cat("🚨 Not enough independent variables (posit_x) to impute column:", col_idx, "\n")
   next  # Skip this iteration if no valid predictors exist
  }
  
  # Ensure `otutab_matrix` is still a matrix
  otutab_matrix <- as.matrix(otutab_matrix)
  
  # Run imputation using numeric indices
  otutab_matrix <- fill_NA_N(
   x = otutab_matrix,  # Data matrix
   model = "lm_bayes", # Imputation model
   posit_y = col_idx,  # Single column index to impute
   posit_x = posit_x,  # Dynamically updated predictor columns
   w = NULL            # No weighting
  )
  
  # Ensure matrix structure is maintained
  otutab_matrix <- as.matrix(otutab_matrix)
 }
}

# Count missing values after imputation
total_missing_after <- sum(is.na(otutab_matrix))
cat("\n🔹 Total missing values AFTER imputation:", total_missing_after, "\n")

# Convert back to data frame
complete_data_miceFast <- as.data.frame(otutab_matrix)

# Save the dataset
write.csv(complete_data_miceFast, "otutab_transp_div_imputed_miceFast.csv", row.names = FALSE)

cat("GPU-Accelerated MICE Imputation completed and saved as 'otutab_transp_div_imputed_miceFast.csv'\n")

#-------------------------------------------------------------------------------------

# Ensure column names are valid
colnames(otutab_transp_div) <- make.names(colnames(otutab_transp_div), unique = TRUE)

# Select only numeric columns
numeric_columns <- names(otutab_transp_div)[sapply(otutab_transp_div, is.numeric)]

sum(is.na(otutab_transp_div))


# Simplify all column names (in case there are special characters or spaces)
colnames(otutab_transp_div) <- make.names(colnames(otutab_transp_div), unique = TRUE)

# Automatically detect numeric columns for PMM imputation
numeric_columns <- names(otutab_transp_div)[sapply(otutab_transp_div, is.numeric)]

# Set up MICE imputation methods
methods <- make.method(otutab_transp_div)
methods[] <- ""  # Set all columns to "" (no imputation by default)

methods[numeric_columns] <- "pmm" 

# Specify PMM (Predictive Mean Matching) for selected columns
for (col in numeric_columns) {
 methods[col] <- "cart"
}

# Double-check length consistency
print(length(methods))  # Should match ncol(data_clinical_sel_var_all_gx_ubiop_Oct22)
print(ncol(otutab_transp_div))  # Should match length(methods)

# Enable parallel processing
plan(multisession)

# Perform the MICE imputation for the remaining variables
imputed_data <- mice(otutab_transp_div, 
                     m = 1, 
                     maxit = 1, 
                     method = methods, 
                     seed = 123, 
                     printFlag = TRUE)




# Extract and inspect the imputed data
complete_data <- complete(imputed_data, 1)

# Check if the missing values were imputed
sum(is.na(complete_data))

write.csv(complete_data, "otutab_transp_div_imputed_pmm5_Feb13.csv", row.names = TRUE)

#------------------------------------------------------------------------------


library(VIM)

# Perform kNN imputation
complete_data <- kNN(otutab_transp_div, k = 5)  # k = 5 means it finds 5 closest neighbors

# Save imputed dataset
write.csv(complete_data, "otutab_transp_div_imputed_kNN.csv", row.names = FALSE)

cat("Imputation completed using kNN! 🚀 Saved as 'otutab_transp_div_imputed_kNN.csv'\n")


#-----------------------------------------------------------------------------
#Feb 13, 2025
# otutab_transpose

# Load MICE library
library(mice)

# install.packages("VIM")
library(VIM)


otutab_transp_div <- read.csv("otutab_transposed_missing_patients_MICE.csv", header = TRUE, check.names = FALSE)

# Check for missing values before imputation
cat("Total missing values before imputation:", sum(is.na(otutab_transp_div)), "\n")

# Simplify all column names to avoid special characters
colnames(otutab_transp_div) <- make.names(colnames(otutab_transp_div), unique = TRUE)

# Automatically detect numeric columns for PMM imputation
numeric_columns <- names(otutab_transp_div)[sapply(otutab_transp_div, is.numeric)]

# Set up MICE imputation methods
methods <- make.method(otutab_transp_div)
methods[] <- ""  # Default: no imputation for all columns
methods[numeric_columns] <- "pmm"  # Apply PMM to numeric columns

# Verify that methods match column count
cat("Length of methods:", length(methods), "\n")
cat("Number of columns:", ncol(otutab_transp_div), "\n")

# Perform MICE imputation
imputed_data <- mice(otutab_transp_div, 
                     m = 5, 
                     maxit = 100, 
                     method = methods, 
                     seed = 123, 
                     printFlag = TRUE)

# Extract the complete imputed dataset (using last imputation)
complete_data <- complete(imputed_data)

# Check for remaining missing values
cat("Total missing values after imputation:", sum(is.na(complete_data)), "\n")

# Save the imputed dataset
write.csv(complete_data, "otutab_transp_div_imputed_pmm5_Feb13.csv", row.names = FALSE)

