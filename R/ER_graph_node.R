library(pcalg)

setwd("ER")
source("ER_Apr13-v2.R")

# List of your dataset file paths, here is an example for Clinical+Diversity
dataset_paths <- list(
 "ClinDiv_May6_cleaned_1.csv",
 "ClinDiv_May6_cleaned_2.csv",
 "ClinDiv_May6_cleaned_3.csv",
 "ClinDiv_May6_cleaned_4.csv",
 "ClinDiv_May6_cleaned_5.csv"
)

# Loop over datasets and recalculate Zs
for (i in seq_along(dataset_paths)) {
 cat(sprintf("Processing dataset %d: %s\n", i, dataset_paths[[i]]))
 results <- process_dataset(
  file_path = dataset_paths[[i]],
  delta = 0.1,
  beta_est = "Lasso",
  var_threshold = 0.01,
  cor_threshold = 0.95,
  n_preselected = 1000,
  preserve_correlated = TRUE,
  use_multi_dimensions = TRUE
 )
 # Create output directory if it doesn't exist
 outdir <- sprintf("output/dataset_%d", i)
 dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
 # Save Zs for this dataset
 write.csv(results$Z, file.path(outdir, "Z_factors.csv"), row.names = FALSE)
}


# Read latent variables (Zs)
latent <- read.csv("output/dataset_5/Z_factors.csv")

# Read the original dataset to get the outcome
orig <- read.csv("ClinDiv_May6_cleaned_5.csv")

# Add the outcome column to the latent variable data frame
latent$Exacerbation.Outcome <- orig$Exacerbation.Outcome

# Save the combined data frame (optional)
write.csv(latent, "output/dataset_5/Z_factors_with_outcome.csv", row.names = FALSE)




# run the pc algorithm (casual) for Z_factors+Exacerbation.Outcome
latent <- read.csv("output/dataset_1/Z_factors_with_outcome.csv")

vars <- colnames(latent)

# For continuous data, use the correlation matrix and sample size
suffStat <- list(C = cor(latent), n = nrow(latent))

# Run the PC algorithm
pc.fit <- pc(suffStat, indepTest = gaussCItest, alpha = 0.05, labels = vars)


# Convert the graphNEL object to igraph for better plotting
g <- igraph.from.graphNEL(pc.fit@graph)

# Optional: highlight the outcome node
V(g)$color <- ifelse(V(g)$name == "Exacerbation.Outcome", "deeppink3", "skyblue")
V(g)$frame.color <- "black"
V(g)$size <- 30
V(g)$label.cex <- 1.2

plot(g, main = "Causal Graph: Z Factors and Outcome (Dataset 1)")


library(igraph)

plot_er_causal_graph <- function(mapping_csv, outcome, output_png = "er_causal_graph.png") {
 # Read mapping
 mapping <- read.csv(mapping_csv, stringsAsFactors = FALSE)
 
 # Get unique Zs and features
 z_factors <- unique(mapping$z_factor)
 features  <- unique(mapping$feature_name)
 
 # Identify Markov blanket Zs
 mb_zs <- unique(mapping$z_factor[mapping$markov_blanket == TRUE])
 
 # Edges: Z -> feature
 edges <- data.frame(from = mapping$z_factor, to = mapping$feature_name, stringsAsFactors = FALSE)
 
 # Edges: Z -> outcome (for all Zs present)
 edges_outcome <- data.frame(from = z_factors, to = outcome, stringsAsFactors = FALSE)
 
 # Combine all edges
 all_edges <- rbind(edges, edges_outcome)
 
 # Create graph
 g <- graph_from_data_frame(all_edges, directed = TRUE)
 
 # Node types for coloring
 V(g)$type <- ifelse(V(g)$name %in% z_factors, "Z",
                     ifelse(V(g)$name == outcome, "Outcome", "Feature"))
 
 # Node colors
 node_colors <- c(Z = "white", Outcome = "deeppink3", Feature = "darkseagreen3")
 V(g)$color <- node_colors[V(g)$type]
 
 # Node border colors
 V(g)$frame.color <- ifelse(V(g)$type == "Z", "blue",
                            ifelse(V(g)$type == "Outcome", "deeppink3", "darkgreen"))
 
 # Node shapes
 V(g)$shape <- ifelse(V(g)$type == "Z", "rectangle",
                      ifelse(V(g)$type == "Outcome", "rectangle", "square"))
 
 # Node label size
 V(g)$label.cex <- ifelse(V(g)$type == "Z", 1.2,
                          ifelse(V(g)$type == "Outcome", 1.2, 0.9))
 
 # Node label font
 V(g)$label.font <- ifelse(V(g)$type == "Z", 2,
                           ifelse(V(g)$type == "Outcome", 2, 1))
 
 # Highlight Markov blanket Zs
 V(g)$mb_z <- V(g)$name %in% mb_zs
 
 # Node border width
 V(g)$frame.width <- ifelse(V(g)$mb_z, 4, 2)
 
 mb_features <- unique(mapping$feature_name[mapping$markov_blanket == TRUE])
 V(g)$mb_feature <- V(g)$name %in% mb_features
 V(g)$frame.color <- ifelse(V(g)$mb_z | V(g)$mb_feature, "blue",
                            ifelse(V(g)$type == "Z", "blue",
                                   ifelse(V(g)$type == "Outcome", "deeppink3", "darkgreen")))
 V(g)$frame.width <- ifelse(V(g)$mb_z | V(g)$mb_feature, 4, 2)
 V(g)$label.font  <- ifelse(V(g)$mb_z | V(g)$mb_feature, 2,
                            ifelse(V(g)$type == "Z", 2,
                                   ifelse(V(g)$type == "Outcome", 2, 1)))
 
 # Save to PNG
 png(output_png, width = 900, height = 900)
 plot(g,
      vertex.label.color = "black",
      vertex.size = 20,
      edge.arrow.size = 0.3,
      layout = layout_with_fr)
 dev.off()
 cat(sprintf("Saved graph to %s\n", output_png))
}


# Example usage:
# plot_er_causal_graph("C:/Users/omatu/OneDrive/Desktop/R_UdeM/UdeM/ER/causal_feature_mapping_ClinGenDiv.csv", outcome = "Exacerbation.Outcome", output_png = "er_causal_graph_ClinGenDiv.png")

plot_er_causal_graph("C:/Users/omatu/OneDrive/Desktop/R_UdeM/UdeM/ER/causal_feature_mapping_Clin.csv", outcome = "Exacerbation.Outcome", output_png = "er_causal_graph_Clin.png")
