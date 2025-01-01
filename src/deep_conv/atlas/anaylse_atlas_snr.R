library(data.table)
library(ggplot2)
library(gridExtra)

# R_LIBS_USER=~/R/library Rscript /users/zetzioni/sharedscratch/deepconv/src/deep_conv/atlas/anaylse_atlas_snr.R \
# -a /users/zetzioni/sharedscratch/atlas/atlas_dmr_by_read.blood+gi+tum.U250.l4.bed \
# -o /users/zetzioni/sharedscratch/atlas/

calculate_and_visualize_marker_metrics <- function(atlas_file, blood_columns = c("B-cells", "CD34-erythroblasts", 
                                                                "CD34-megakaryocytes", "CD4-T-cells",
                                                                "CD8-T-cells", "Eosinophils", 
                                                                "Monocytes", "NK-cells", "Neutrophils")) {
   # Calculate metrics as before
   atlas <- fread(atlas_file)
   dir_col <- which(names(atlas) == "direction")
   value_cols <- names(atlas)[(dir_col + 1):ncol(atlas)]
   
   # Calculate metrics for blood/immune cells
   blood_metrics <- atlas[, {
       blood_values <- unlist(.SD[, ..blood_columns])
       blood_values <- blood_values[!is.na(blood_values)]
       
       blood_var <- var(blood_values)
       blood_mean <- mean(blood_values)
       blood_snr <- blood_var / (blood_mean + 1e-10)
       
       list(blood_variance = blood_var,
            blood_mean = blood_mean,
            blood_snr = blood_snr)
   }, by = .(chr, start, end, target)]
   
   # Calculate metrics for all cell types
   all_metrics <- atlas[, {
       all_values <- unlist(.SD[, ..value_cols])
       all_values <- all_values[!is.na(all_values)]
       
       all_var <- var(all_values)
       all_mean <- mean(all_values)
       all_snr <- all_var / (all_mean + 1e-10)
       
       list(all_variance = all_var,
            all_mean = all_mean,
            all_snr = all_snr)
   }, by = .(chr, start, end, target)]
   
   results <- merge(blood_metrics, all_metrics, by = c("chr", "start", "end", "target"))
   
   plots <- list()
   
   plots$snr_dist <- ggplot(results, aes(x = blood_snr, fill = target)) +
       geom_histogram(bins = 50, alpha = 0.7) +
       facet_wrap(~target, scales = "free_y") +
       theme_minimal() +
       labs(title = "Distribution of Blood/Immune SNR by Target Cell Type",
            x = "SNR", y = "Count") +
       theme(legend.position = "none")
   
   plots$snr_scatter <- ggplot(results, aes(x = blood_snr, y = all_snr, color = target)) +
       geom_point(alpha = 0.5) +
       geom_abline(linetype = "dashed") +
       theme_minimal() +
       labs(title = "Blood/Immune SNR vs All Cell Types SNR",
            x = "Blood/Immune SNR", y = "All Cell Types SNR")
   
   plots$snr_box <- ggplot(results, aes(x = reorder(target, blood_snr, FUN = median), 
                                       y = blood_snr, fill = target)) +
       geom_boxplot() +
       theme_minimal() +
       theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
       labs(title = "Blood/Immune SNR Distribution by Target",
            x = "Target Cell Type", y = "SNR") +
       theme(legend.position = "none")
   
   plots$var_dist <- ggplot(results) +
       geom_density(aes(x = blood_variance, fill = "Blood"), alpha = 0.5) +
       geom_density(aes(x = all_variance, fill = "All"), alpha = 0.5) +
       theme_minimal() +
       labs(title = "Distribution of Variance",
            x = "Variance", y = "Density")
   
   snr_thresholds <- seq(0, max(results$blood_snr), length.out = 100)
   markers_above_threshold <- sapply(snr_thresholds, function(t) {
       results[blood_snr >= t, .N, by = target][, sum(N)]
   })
   threshold_df <- data.table(threshold = snr_thresholds, 
                            markers = markers_above_threshold)
   
   plots$threshold_curve <- ggplot(threshold_df, aes(x = threshold, y = markers)) +
       geom_line() +
       theme_minimal() +
       labs(title = "Number of Markers vs SNR Threshold",
            x = "SNR Threshold", y = "Total Markers")
   
   blood_markers <- atlas[target %in% blood_columns]
   mean_values <- blood_markers[, lapply(.SD, mean), 
                               .SDcols = blood_columns, 
                               by = target]
   
   mean_values_long <- melt(mean_values, 
                           id.vars = "target",
                           variable.name = "cell_type",
                           value.name = "mean_uxm")
   
   plots$heatmap <- ggplot(mean_values_long, 
                          aes(x = target, y = cell_type, fill = mean_uxm)) +
       geom_tile() +
       scale_fill_gradient2(low = "blue", mid = "white", high = "red", 
                          midpoint = 0.5) +
       theme_minimal() +
       theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
       labs(title = "Mean UXM Values for Blood Cell Markers",
            x = "Target Cell Type", y = "Cell Type",
            fill = "Mean UXM")
   
   return(list(metrics = results, plots = plots))
}

main <- function() {
   option_list <- list(
       make_option(c("-a", "--atlas_file"), type="character",
                   help="Input atlas file [REQUIRED]",
                   metavar="atlas.txt"),
       make_option(c("-o", "--output_dir"), type="character", default=".",
                   help="Output directory for analysis files [default %default]",
                   metavar="path")
   )
   
   parser <- OptionParser(option_list=option_list)
   params <- parse_args(parser)
   
   if (is.null(params$atlas_file)) {
       print_help(parser)
       stop("Missing required argument: atlas_file")
   }
   
   message(sprintf("Analyzing atlas file: %s", params$atlas_file))
   message(sprintf("Output directory: %s", params$output_dir))
   
   # Create output directory if it doesn't exist
   dir.create(params$output_dir, showWarnings = FALSE, recursive = TRUE)
   
   # Run analysis
   results <- calculate_and_visualize_marker_metrics(params$atlas_file)
   
   # Save plots
   pdf_path <- file.path(params$output_dir, "marker_analysis.pdf")
   pdf(pdf_path, width = 12, height = 10)
   do.call(grid.arrange, c(results$plots, ncol = 2))
   dev.off()
   
   message(sprintf("Analysis complete. Plots saved to: %s", pdf_path))
   
   # Save metrics to CSV
   metrics_path <- file.path(params$output_dir, "marker_metrics.csv")
   fwrite(results$metrics, metrics_path)
   message(sprintf("Metrics saved to: %s", metrics_path))
}

# Run main function if script is run directly
if (sys.nframe() == 0) {
   suppressPackageStartupMessages({
       library(optparse)
       library(data.table)
       library(ggplot2)
       library(gridExtra)
   })
   main()
}