required_packages <- c("data.table", "ggplot2", "gridExtra", "optparse")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) {
    install.packages(new_packages, 
                    lib = Sys.getenv("R_LIBS_USER"), 
                    repos = "https://cloud.r-project.org")
}


# R_LIBS_USER=~/R/library Rscript /users/zetzioni/sharedscratch/deepconv/src/deep_conv/atlas/anaylse_atlas_snr.R \
# -a /users/zetzioni/sharedscratch/atlas/atlas_dmr_by_read.blood+gi+tum.U250.l4.bed \
# -o /users/zetzioni/sharedscratch/atlas/

library(data.table)
library(ggplot2)

calculate_and_visualize_marker_metrics <- function(atlas_file, blood_columns = c("B-cells", "CD34-erythroblasts", 
                                                                 "CD34-megakaryocytes", "CD4-T-cells",
                                                                 "CD8-T-cells", "Eosinophils", 
                                                                 "Monocytes", "NK-cells", "Neutrophils")) {
    # Read atlas
    atlas <- fread(atlas_file)
    dir_col <- which(names(atlas) == "direction")
    value_cols <- names(atlas)[(dir_col + 1):ncol(atlas)]
    
    # Calculate blood-specific metrics
    blood_metrics <- atlas[, {
        # Get values for blood cells only
        blood_vals <- as.numeric(unlist(.SD[, ..blood_columns]))
        blood_vals <- blood_vals[!is.na(blood_vals)]
        
        # Calculate key metrics
        blood_var <- var(blood_vals)
        blood_range <- max(blood_vals) - min(blood_vals)
        blood_min <- min(blood_vals)
        blood_max <- max(blood_vals)
        
        list(blood_variance = blood_var,
             blood_range = blood_range,
             blood_min = blood_min,
             blood_max = blood_max,
             blood_mean = mean(blood_vals))
    }, by = .(chr, start, end, target)]
    
    # Create visualizations
    plots <- list()
    
    # 1. Distribution of variance
    plots$variance_dist <- ggplot(blood_metrics, aes(x = blood_variance, fill = target)) +
        geom_histogram(bins = 50, alpha = 0.7) +
        facet_wrap(~target, scales = "free_y") +
        theme_minimal() +
        labs(title = "Distribution of Blood/Immune Variance by Target Cell Type",
             x = "Variance", y = "Count") +
        theme(legend.position = "none")
    
    # 2. Range vs Minimum value
    plots$range_vs_min <- ggplot(blood_metrics, aes(x = blood_min, y = blood_range, color = target)) +
        geom_point(alpha = 0.5) +
        theme_minimal() +
        labs(title = "Range vs Minimum UXM Value",
             x = "Minimum UXM Value", y = "Range of UXM Values")
    
    # 3. Box plots of variance by target
    plots$var_box <- ggplot(blood_metrics, aes(x = reorder(target, blood_variance, FUN = median), 
                                        y = blood_variance, fill = target)) +
        geom_boxplot() +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(title = "Blood/Immune Variance Distribution by Target",
             x = "Target Cell Type", y = "Variance") +
        theme(legend.position = "none")
    
    # 4. Number of markers above thresholds
    var_thresholds <- seq(0, max(blood_metrics$blood_variance), length.out = 100)
    markers_above_var <- sapply(var_thresholds, function(t) {
        blood_metrics[blood_variance >= t, .N]
    })
    min_thresholds <- seq(0, 0.2, length.out = 100)  # Adjust range as needed
    markers_above_min <- sapply(min_thresholds, function(t) {
        blood_metrics[blood_min >= t, .N]
    })
    
    threshold_df <- data.table(
        var_threshold = var_thresholds,
        min_threshold = min_thresholds,
        var_markers = markers_above_var,
        min_markers = markers_above_min
    )
    
    plots$threshold_curves <- ggplot(threshold_df) +
        geom_line(aes(x = var_threshold, y = var_markers, color = "Variance")) +
        geom_line(aes(x = min_threshold, y = min_markers, color = "Minimum Value")) +
        theme_minimal() +
        labs(title = "Number of Markers vs Thresholds",
             x = "Threshold", y = "Total Markers",
             color = "Threshold Type")

    # Print summary statistics
    message("\nSummary Statistics:")
    message("\nMarkers by minimum UXM value:")
    print(blood_metrics[, .N, by=.(min_bin = cut(blood_min, breaks=seq(0, 1, by=0.1)))])
    
    message("\nMarkers by variance:")
    print(blood_metrics[, .N, by=.(var_bin = cut(blood_variance, breaks=seq(0, max(blood_variance), by=0.01)))])
    
    message("\nTop markers by variance:")
    print(blood_metrics[order(-blood_variance)][1:10])
    
    # Add to the existing code
    message("\nMarker counts for different threshold combinations:")
    min_thresholds <- seq(0, 0.2, by=0.05)
    var_thresholds <- seq(0, 0.1, by=0.02)

    counts_matrix <- sapply(min_thresholds, function(min_t) {
        sapply(var_thresholds, function(var_t) {
            blood_metrics[blood_min >= min_t & blood_variance >= var_t, .N]
        })
    })

    # Create a table of results
    result_dt <- data.table(
        min_threshold = rep(min_thresholds, each=length(var_thresholds)),
        var_threshold = rep(var_thresholds, times=length(min_thresholds)),
        count = as.vector(counts_matrix)
    )

    # Print results in a readable format
    message("\nNumber of markers meeting both thresholds:")
    for(min_t in min_thresholds) {
        message(sprintf("\nMinimum UXM >= %.2f:", min_t))
        for(var_t in var_thresholds) {
            count <- result_dt[min_threshold == min_t & var_threshold == var_t, count]
            message(sprintf("  Variance >= %.2f: %d markers", var_t, count))
        }
    }

    # Look at top markers by variance
    message("\nTop 20 markers by variance:")
    print(blood_metrics[order(-blood_variance)][1:20,
        .(chr, start, end, target, blood_variance, blood_min, blood_max, blood_range)])


    # For each combination of thresholds, count markers per target cell type
    message("\nMarkers per cell type for different threshold combinations:")
    for(min_t in c(0, 0.05, 0.1)) {
        message(sprintf("\nMinimum UXM >= %.2f:", min_t))
        for(var_t in c(0, 0.02, 0.04, 0.06)) {
            message(sprintf("\n  Variance >= %.2f:", var_t))
            counts_by_target <- blood_metrics[blood_min >= min_t & 
                                            blood_variance >= var_t, 
                                            .N, by=target][order(-N)]
            print(counts_by_target)
        }
    }

    # Also look at specific interesting thresholds:
    message("\nDetailed look at potential threshold combination:")
    message("\nMinimum UXM >= 0.1 and Variance >= 0.04:")
    print(blood_metrics[blood_min >= 0.1 & blood_variance >= 0.04, 
                    .N, by=target][order(-N)])

    return(list(metrics = blood_metrics, plots = plots))
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
    
    dir.create(params$output_dir, showWarnings = FALSE, recursive = TRUE)
    
    results <- calculate_and_visualize_marker_metrics(params$atlas_file)
    
    pdf_path <- file.path(params$output_dir, "marker_analysis.pdf")
    pdf(pdf_path, width = 12, height = 10)
    do.call(grid.arrange, c(results$plots, ncol = 2))
    dev.off()
    
    metrics_path <- file.path(params$output_dir, "marker_metrics.csv")
    fwrite(results$metrics, metrics_path)
    
    message(sprintf("Analysis complete. Plots saved to: %s", pdf_path))
    message(sprintf("Metrics saved to: %s", metrics_path))
}


if (sys.nframe() == 0) {
    suppressPackageStartupMessages({
        library(optparse)
        library(data.table)
        library(ggplot2)
        library(gridExtra)
    })
    main()
}