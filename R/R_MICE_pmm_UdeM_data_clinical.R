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


# Load MICE library
library(mice)

# install.packages("ranger")
library(ranger)

# install.packages("VIM")
library(VIM)


data_clinical_sel_var_all_gx_ubiop_Oct22 <- read.csv("sel_var_all_gx_ubiop_class_exacerb_data_clinical_for_imputation_Oct22.csv", header = TRUE, check.names = FALSE)

# Check number of columns
print(ncol(data_clinical_sel_var_all_gx_ubiop_Oct22))

sum(is.na(data_clinical_sel_var_all_gx_ubiop_Oct22))

# Simplify all column names
colnames(data_clinical_sel_var_all_gx_ubiop_Oct22) <- make.names(colnames(data_clinical_sel_var_all_gx_ubiop_Oct22), unique = TRUE)

# Verify the new column names
colnames(data_clinical_sel_var_all_gx_ubiop_Oct22)

# Convert Exacerbation.Outcome to a factor
data_clinical_sel_var_all_gx_ubiop_Oct22$Exacerbation.Outcome <- as.factor(data_clinical_sel_var_all_gx_ubiop_Oct22$Exacerbation.Outcome)

# # Check the structure of all variables in the dataset
str(data_clinical_sel_var_all_gx_ubiop_Oct22)
# 
# # Impute using KNN for 'Asthma.Control.Test.ACT.Children.over.12.Total'
# data_clinical_imputed_knn <- kNN(data_clinical_sel_var_all_gx_ubiop_Oct22, variable = "Asthma.Control.Test.ACT.Children.over.12.Total", k = 5)

# # Check if the missing values for the KNN-imputed variable were filled
# sum(is.na(data_clinical_imputed_knn$Asthma.Control.Test.ACT.Children.over.12.Total))

# Set up MICE imputation for remaining variables
methods <- make.method(data_clinical_sel_var_all_gx_ubiop_Oct22)
methods[] <- ""  # Set all columns to "" (no imputation by default)

# Set the imputation methods for specific columns
methods["Standard.Flow.Rate"] <- "pmm"
methods["Screening.eosinophils"] <- "pmm"
methods["Screening.neutrophils"] <- "pmm"
methods["Total.lung.capacity.TLC.Predicted.."] <- "pmm"
methods["Total.lung.capacity.TLC.Predicted.L"] <- "pmm"
methods["FEV1.Post.Salbutamol...L"] <- "pmm"
methods["FEV1.Pre.Salbutamol...L"] <- "pmm"
methods["Asthma.Control.Test.ACT.Childhood.Total"] <- "pmm"
methods["Medication.Adherence.Rating.Scale.MARS.Total"] <- "pmm"
methods["Paediatric.Asthma.Quality.of.Life.Questionnaire.PAQLQ.all.questions.average"] <- "pmm"
methods["Oral.CS.dose"] <- "pmm"
methods["Exacerbation.Outcome"] <- "logreg"

# Check length of the methods vector and compare with the number of columns in the data
length(methods)
ncol(data_clinical_sel_var_all_gx_ubiop_Oct22)
# 
# # List of columns and corresponding methods you want to apply
# columns_to_impute <- c("Standard.Flow.Rate", "Screening.eosinophils", "Screening.neutrophils",
#                        "Total.lung.capacity.TLC.Predicted..", "Total.lung.capacity.TLC.Predicted.L",
#                        "FEV1.Post.Salbutamol...L", "FEV1.Pre.Salbutamol...L",
#                        "Asthma.Control.Test.ACT.Childhood.Total",
#                        "Medication.Adherence.Rating.Scale.MARS.Total",
#                        "Paediatric.Asthma.Quality.of.Life.Questionnaire.PAQLQ.all.questions.average",
#                        "Oral.CS.dose", "Exacerbation.Outcome")
# 
# # Corresponding methods for each column
# imputation_methods <- c("pmm", "pmm", "pmm", "pmm", "pmm", "pmm", "pmm", "pmm", "pmm", "pmm", "pmm", "logreg")
# 
# # Check if columns exist and apply methods only for existing columns
# for (i in seq_along(columns_to_impute)) {
#  column_name <- columns_to_impute[i]
#  method <- imputation_methods[i]
# 
#  if (column_name %in% colnames(data_clinical_sel_var_all_gx_ubiop_Oct22)) {
#   methods[column_name] <- method
#  } else {
#   warning(paste("Column", column_name, "does not exist in the dataset. Skipping imputation method assignment for this column."))
#  }
# }
# 
# # Verify the number of methods matches the number of columns
# length(methods)  # Should be equal to ncol(data_clinical_sel_var_all_gx_ubiop_Oct22)
# ncol(data_clinical_sel_var_all_gx_ubiop_Oct22)  # Check number of columns in the dataset
# 
# 
# 

# Perform the imputation
imputed_data <- mice(data_clinical_sel_var_all_gx_ubiop_Oct22, 
                     m = 5, 
                     maxit = 100, 
                     method = methods, 
                     seed = 123, 
                     printFlag = TRUE)


# # Ensure that the 'Asthma.Control.Test.ACT.Children.over.12.Total' column is NOT re-imputed by MICE
# methods["Asthma.Control.Test.ACT.Children.over.12.Total"] <- ""  # Exclude from MICE since it was already imputed by KNN


# # Set all other columns to "none"
# methods[!(names(methods) %in% c("Standard.Flow.Rate", "Screening.eosinophils", "Screening.neutrophils", "Total.lung.capacity.TLC.Predicted..", "Total.lung.capacity.TLC.Predicted.L", "FEV1.Post.Salbutamol...L", "FEV1.Pre.Salbutamol...L", "Asthma.Control.Test.ACT.Childhood.Total", "Asthma.Control.Test.ACT.Children.over.12.Total","Medication.Adherence.Rating.Scale.MARS.Total","Paediatric.Asthma.Quality.of.Life.Questionnaire.PAQLQ.all.questions.average", "Oral.CS.dose"))] <- "none"
# 

# 
# # Check the column names in the imputed data
# colnames(data_clinical_sel_var_all_gx_ubiop_Oct22)
# 
# # Then use the exact column name in the plot function
# plot(imputed_data, c("Standard.Flow.Rate"))



# Extract and inspect the imputed data
complete_data <- complete(imputed_data, 5)

# Check if the missing values were imputed
sum(is.na(complete_data))

write.csv(complete_data, "data_clinical_sel_var_all_gx_ubiop_Oct23_imputed_5.csv", row.names = TRUE)
