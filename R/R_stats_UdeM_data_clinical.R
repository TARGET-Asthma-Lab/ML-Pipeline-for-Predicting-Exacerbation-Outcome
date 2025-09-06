
# Load the data from the CSV file
data_clinical <- read.csv("R_UdeM/UdeM/data_clinical.csv", header = TRUE, check.names = FALSE)

# # Convert the variable "\IC supplementary oxygen\" to a factor
# data_clinical[["\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Demographic Data\\IC supplementary oxygen\\"]] <- as.factor(data_clinical[["\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Demographic Data\\IC supplementary oxygen\\"]])
# 
# # Check the structure to confirm the change
# str(data_clinical[["\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Demographic Data\\IC supplementary oxygen\\"]])
# 
# # Produce a frequency table for the categorical variable
# frequency_table <- table(data_clinical[["\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Demographic Data\\IC supplementary oxygen\\"]])
# print(frequency_table)

library(summarytools)
freq_table_detailed <- freq(data_clinical[, c("\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Anti IgE Therapy\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Antibiotic Therapy\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Cromones\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Inhaled Combinations\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Inhaled Corticosteroids\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Injectable Corticosteroids\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Leukotriene Modifiers\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Long Acting Anticholinergics\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Long Acting Beta Agonist\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Oral Corticosteroids\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Short Acting Beta Agonist\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Short Acting Combination\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\OCS Normalised Dose (mg)\\")])

print(freq_table_detailed)

# Save the frequency table detailed output to a text file
capture.output(freq_table_detailed, file = "freq_table_Medication_Baseline-v2.txt")

# Get descriptive stats for continuous variables
continuous_stats <- describe(data_clinical[, c("\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\OCS Normalised Dose (mg)\\")])
print(continuous_stats)
# -------------------------------------------------------------------------------

# Install and load psych package
# install.packages("psych")
library(psych)

# Get descriptive stats for continuous variables
continuous_stats <- describe(data_clinical[["\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Anti IgE Therapy\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Antibiotic Therapy\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Cromones\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Inhaled Combinations\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Inhaled Corticosteroids\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Injectable Corticosteroids\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Leukotriene Modifiers\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Long Acting Anticholinergics\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Long Acting Beta Agonist\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Oral Corticosteroids\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Short Acting Beta Agonist\\","\\UBIOPRED\\Paediatric_Cohort_(Jan_2019)\\Medication\\Baseline\\Short Acting Combination\\"]])

# Check for categorical variables and calculate frequency counts
categorical_vars <- sapply(data_clinical, is.factor) | sapply(data_clinical, is.character)

# Calculate frequencies for all categorical variables
categorical_stats <- lapply(data_clinical[, categorical_vars], table)

# Print out the frequency tables
categorical_stats


# Install and load summarytools
install.packages("summarytools")
library(summarytools)

# Frequency stats for categorical variables
categorical_stats <- freq(data_clinical[, categorical_vars])

# Print frequency tables
print(categorical_stats, method = "pander")

options(max.print = 10000)

# Combine continuous and categorical descriptive stats
list(continuous_stats, categorical_stats)

# Combine continuous and categorical descriptive stats
combined_stats <- list(continuous_stats, categorical_stats)

# Save the output to a text file
sink("combined_stats.txt")
print(combined_stats)
sink()


# Create a mapping of column numbers (vars) to column names
column_mapping <- data.frame(
 vars = 1:length(colnames(data_clinical)),   # Create a sequence from 1 to the number of columns
 column_name = colnames(data_clinical)       # Corresponding column names
)

# Save the mapping to a CSV file
write.csv(column_mapping, "column_mapping.csv", row.names = FALSE)

# Capture the output of summary into a text file
capture.output(summary_stats, file = "summary_stats.txt")


# Save the summary stats to a CSV file
write.csv(summary_stats, "summary_stats_final.csv", row.names = TRUE)


# Calculate the number of missing values for each column in the dataset
missing_counts <- sapply(data_clinical, function(x) sum(is.na(x)))

# Get descriptive stats using psych::describe
continuous_stats <- describe(data_clinical)

# Calculate the number of missing values for each variable
missing_counts <- sapply(data_clinical, function(x) sum(is.na(x)))

# Add the missing counts to the continuous stats dataframe
continuous_stats$missing <- missing_counts

# Save the updated continuous stats to a CSV file
write.csv(continuous_stats, "continuous_stats_with_missing.csv", row.names = TRUE)




