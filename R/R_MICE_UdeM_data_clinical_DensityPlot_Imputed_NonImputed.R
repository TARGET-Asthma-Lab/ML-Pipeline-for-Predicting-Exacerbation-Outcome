# Load necessary library
library(mice)


non_imputed_data_clinical_Oct22 <- read.csv("sel_var_all_gx_ubiop_class_exacerb_data_clinical_for_imputation_Oct22.csv", header = TRUE, check.names = FALSE)
imputed_data_5 <- read.csv("data_clinical_sel_var_all_gx_ubiop_Oct30_imputed_rf_1_vv.csv") # then add "data_clinical_sel_var_all_gx_ubiop_Oct30_imputed_rf_2_vv.csv", "...._rf_3_vv.csv", etc

# Specify the variables to compare
selected_variables <- c("Standard.Flow.Rate", "Screening.eosinophils", "Screening.neutrophils",
                        "Total.lung.capacity.TLC.Predicted.persentage","FEV1.Post.Salbutamol...L",
                        "FEV1.Pre.Salbutamol...L", "Asthma.Control.Test.ACT.Childhood.Total",
                        "PAQLQ.all.questions.average", "Scale.MARS.Total", "Exacerbation.Outcome")



variables_to_plot <- intersect(selected_variables, intersect(names(non_imputed_data_clinical_Oct22), names(imputed_data_5)))

custom_ylim <- list(
 "Screening.neutrophils" = c(0, 0.5),  # Adjust based on observed/imputed data distribution
 "Asthma.Control.Test.ACT.Childhood.Total" = c(0, 0.13),
 "Scale.MARS.Total" = c(0, 0.25),
 "PAQLQ.all.questions.average" = c(0, 0.35),
 "Screening.eosinophils" =c(0, 1.9),
 "Total.lung.capacity.TLC.Predicted.persentage" = c(0, 0.03)
)

png(filename = "density_plots_dataset_Oct30_rf_1_v3_vv.png", width = 1200, height = 1600, res = 150)

# Set up a 4x3 grid for plotting
par(mfrow = c(4, 3), mar = c(4, 4, 2, 1))

for (i in seq_along(variables_to_plot)) {
 variable <- variables_to_plot[i]

 # Determine y-axis limits
 ylim_val <- if (variable %in% names(custom_ylim)) custom_ylim[[variable]] else NULL
 
 # Plot density for the imputed data
 plot(density(imputed_data_5[[variable]], na.rm = TRUE), col = "red", lwd = 2,
      main = paste("Density Plot for", variable),
      xlab = variable, ylab = "Density", ylim = ylim_val)
 
 # Overlay the density for the observed (non-imputed) data from non_imputed_data
 lines(density(non_imputed_data_clinical_Oct22[[variable]], na.rm = TRUE), col = "black", lwd = 2, lty = 2)
 
#  # Add legend to distinguish imputed vs observed
#  legend("topright", legend = c("Imputed", "Observed"), col = c("red", "black"), lty = 1:2, lwd = 2)
# }
 
 # Add legend only for the first plot
 if (i == 1) {
  legend("topright", legend = c("Imputed", "Observed"), col = c("red", "black"), lty = 1:2, lwd = 2, cex = 0.8)
 }
}

# Close the PNG device to save the file
dev.off()

# Filter imputed_data to include only selected variables that are numeric
selected_numeric_data <- imputed_data_1[ , selected_variables]
selected_numeric_data <- selected_numeric_data[sapply(selected_numeric_data, is.numeric)]

# Save Boxplot of Selected Numeric Variables as a PNG
png(filename = "boxplot_selected_imputed_data_5.png", width = 1200, height = 800, res = 150)
boxplot(selected_numeric_data, main = "Boxplot of Selected Imputed Numeric Variables", las = 2, col = "lightblue")
dev.off()




#----------------------------------------------------------------------------------
# Jan 30, 2025

# Load necessary library
library(mice)


non_imputed_beta_div <- read.csv("beta_div_metrics_MICE_agg_filtered.csv", header = TRUE, check.names = FALSE)
imputed_data_1 <- read.csv("beta_div_imputed_pmm1_Jan30.csv") # "...._pmm2_Jan30.csv", "...pmm3_Jan30.csv", etc

# Specify the variables to compare
selected_variables <- c("braycurtis_mean", 
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



variables_to_plot <- intersect(selected_variables, intersect(names(non_imputed_beta_div), names(imputed_data_1)))

png(filename = "beta_div_density_Jant30_pmm1.png", width = 1200, height = 1600, res = 150)

# Set up a 4x3 grid for plotting
par(mfrow = c(4, 3), mar = c(4, 4, 2, 1))

for (i in seq_along(variables_to_plot)) {
 variable <- variables_to_plot[i]
 
 # Calculate densities
 density_imputed <- density(imputed_data_5[[variable]], na.rm = TRUE)
 density_observed <- density(non_imputed_beta_div[[variable]], na.rm = TRUE)
 
 # Combine y-values to set a proper ylim
 ylim_val <- range(c(density_imputed$y, density_observed$y))
 
 # Plot density for the imputed data
 plot(density_imputed, col = "red", lwd = 2,
      main = paste("Density Plot for", variable),
      xlab = variable, ylab = "Density", ylim = ylim_val)
 
 # Overlay the density for the observed (non-imputed) data
 lines(density_observed, col = "black", lwd = 2, lty = 2)
 
 # Add legend only for the first plot
 if (i == 1) {
  legend("topright", legend = c("Imputed", "Observed"), col = c("red", "black"), lty = 1:2, lwd = 2, cex = 0.8)
 }
}

# Close the PNG device to save the file
dev.off()



#--------------------------------------------------------------------------

# alpha diversity

library(mice)


non_imputed_div <- read.csv("alpha_div_MICE_filtered.csv", header = TRUE, check.names = FALSE)
imputed_data_1 <- read.csv("alpha_div_imputed_pmm1_Jan30.csv") # "alpha_div_imputed_pmm2_Jan30.csv", "alpha_div_imputed_pmm3_Jan30.csv", etc

# Specify the variables to compare
selected_variables <- c("chao1", 
                        "shannon", 
                        "observed_otus"
                        
)



variables_to_plot <- intersect(selected_variables, intersect(names(non_imputed_alpha_div), names(imputed_data_1)))

png(filename = "alpha_div_density_Jant30_pmm1.png", width = 1500, height = 600, res = 150)

# Set up the grid layout
par(mfrow = c(1, 3), mar = c(4, 4, 2, 1))

# Loop through each variable and plot densities
for (variable in variables_to_plot) {
 # Calculate densities
 density_imputed <- density(imputed_data[[variable]], na.rm = TRUE)
 density_observed <- density(non_imputed[[variable]], na.rm = TRUE)
 
 # Combine y-values to determine proper ylim
 ylim_val <- range(c(density_imputed$y, density_observed$y))
 
 # Plot density for the imputed data
 plot(density_imputed, col = "red", lwd = 2,
      main = paste("Density Plot for", variable),
      xlab = variable, ylab = "Density", ylim = ylim_val)
 
 # Overlay the density for the observed (non-imputed) data
 lines(density_observed, col = "black", lwd = 2, lty = 2)
 
 # Add legend only for the first plot
 if (variable == variables_to_plot[1]) {
  legend("topright", legend = c("Imputed", "Observed"), col = c("red", "black"), lty = 1:2, lwd = 2, cex = 0.8)
 }
}
# Close the PNG device to save the file
dev.off()

#--------------------------------------------------------------------------


# Filter imputed_data to include only selected variables that are numeric
selected_numeric_data <- imputed_data_1[ , selected_variables]
selected_numeric_data <- selected_numeric_data[sapply(selected_numeric_data, is.numeric)]

# Save Boxplot of Selected Numeric Variables as a PNG
png(filename = "boxplot_selected_imputed_data_5.png", width = 1200, height = 800, res = 150)
boxplot(selected_numeric_data, main = "Boxplot of Selected Imputed Numeric Variables", las = 2, col = "lightblue")
dev.off()



# Feb 13, 2025

# Load necessary library
library(mice)


non_imputed <- read.csv("otutab_transposed_missing_patients_MICE.csv", header = TRUE, check.names = FALSE)
imputed_data <- read.csv("otutab_transp_div_imputed_fast.csv", header = TRUE, check.names = FALSE)

# Specify the variables to compare
selected_variables <- c("ASV1", 
                        "ASV2", 
                        "ASV3", 
                        "ASV4",
                        "ASV5",
                        "ASV6",
                        "ASV7", 
                        "ASV8", 
                        "ASV9",
                        "ASV10", 
                        "ASV11", 
                        "ASV12"
                        
)



variables_to_plot <- intersect(selected_variables, intersect(names(non_imputed), names(imputed_data)))


png(filename = "otutab_transposed_missing_patients_fast.png", width = 1200, height = 1600, res = 150)

# Set up a 4x3 grid for plotting
par(mfrow = c(4, 3), mar = c(4, 4, 2, 1))

for (i in seq_along(variables_to_plot)) {
 variable <- variables_to_plot[i]
 
 # Calculate densities
 density_imputed <- density(imputed_data[[variable]], na.rm = TRUE)
 density_observed <- density(non_imputed[[variable]], na.rm = TRUE)
 
 # Combine y-values to set a proper ylim
 ylim_val <- range(c(density_imputed$y, density_observed$y))
 
 # Plot density for the imputed data
 plot(density_imputed, col = "red", lwd = 2,
      main = paste("Density Plot for", variable),
      xlab = variable, ylab = "Density", ylim = ylim_val)
 
 # Overlay the density for the observed (non-imputed) data
 lines(density_observed, col = "black", lwd = 2, lty = 2)
 
 # Add legend only for the first plot
 if (i == 1) {
  legend("topright", legend = c("Imputed", "Observed"), col = c("red", "black"), lty = 1:2, lwd = 2, cex = 0.8)
 }
}

# Close the PNG device to save the file
dev.off()
