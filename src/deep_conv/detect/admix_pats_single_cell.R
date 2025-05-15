#!/usr/bin/env Rscript

# Add a simple logging function
log_info <- function(message) {
  cat(sprintf("[INFO] %s: %s\n", format(Sys.time(), "%H:%M:%S"), message))
}

log_debug <- function(message, debug_mode=FALSE) {
  if (debug_mode) {
    cat(sprintf("[DEBUG] %s: %s\n", format(Sys.time(), "%H:%M:%S"), message))
  }
}

print_debug <- function(label, obj, debug_mode=FALSE) {
  if (debug_mode) {
    cat(sprintf("[DEBUG] %s: ", format(Sys.time(), "%H:%M:%S")))
    cat(label, " (", typeof(obj), "/", class(obj), ")\n")
    print(str(obj))
  }
}

# Progress tracking variables
progress_file <- NULL
progress_total <- 0

# Initialize progress tracking
init_progress <- function(output_dir, total_samples) {
  progress_file <<- file.path(output_dir, "progress.txt")
  progress_total <<- total_samples
  # Create initial progress file
  cat(sprintf("Started processing at %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")), 
      file=progress_file)
  cat(sprintf("Total samples to process: %d\n", total_samples), 
      file=progress_file, append=TRUE)
  cat("0% complete (0 samples processed)\n", 
      file=progress_file, append=TRUE)
}

# Update progress
update_progress <- function(completed) {
  if (!is.null(progress_file) && progress_total > 0) {
    percent <- round(completed / progress_total * 100)
    cat(sprintf("%d%% complete (%d samples processed) - %s\n", 
               percent, completed, format(Sys.time(), "%H:%M:%S")), 
        file=progress_file, append=TRUE)
    
    # Also log to console if it's a significant milestone
    if (percent %% 10 == 0 || completed == progress_total) {
      log_info(sprintf("%d%% complete (%d/%d samples processed)", 
                       percent, completed, progress_total))
    }
  }
}

suppressPackageStartupMessages({
    library(data.table)
    library(optparse)
    library(doParallel)
    library(jsonlite)
    library(gtools)  # For rdirichlet
})

# Parse command line arguments
option_list <- list(
    make_option(c("-p", "--pat_dir"), type="character",
                help="Directory containing input pat files [REQUIRED]"),
    make_option(c("-o", "--output_dir"), type="character",
                help="Output directory for mixed samples [REQUIRED]"),
    make_option(c("--tmp_dir"), type="character",
                help="Temporary directory for intermediate files [default: output_dir/tmp]"),
    make_option(c("-t", "--threads"), type="integer", default=1,
                help="Number of threads to use [default %default]"),
    make_option(c("-c", "--concentrations"), type="character",
                help="JSON file specifying cell type of interest and concentration distribution [REQUIRED]"),
    make_option(c("--num_samples"), type="integer", default=50000,
                help="Number of synthetic samples to generate [default %default]"),
    make_option(c("--min_depth"), type="integer", default=40000,
                help="Minimum total coverage [default %default]"),
    make_option(c("--max_depth"), type="integer", default=100000,
                help="Maximum total coverage [default %default]"),
    make_option(c("--overwrite"), action="store_true", default=FALSE,
                help="Overwrite existing files [default %default]"),
    make_option(c("--prefix"), type="character", default="mix",
                help="Prefix for output files [default %default]"),
    make_option(c("--debug"), action="store_true", default=FALSE,
                help="Run in debug mode [default %default]"),
    make_option(c("--batch_size"), type="integer", default=500,
                help="Batch size for progress reporting [default %default]")
)

# Helper functions
make_target_table <- function(cell_type_order, concentrations, pat_dir=".", suffix=".pat.gz") {
    # Create data table for a single mixture
    conc_table <- data.table(
        celltype = cell_type_order,
        fraction = concentrations,
        filename = paste0(pat_dir, '/', cell_type_order, suffix),
        dilution = 1
    )
    return(conc_table)
}

read_count_table <- function(patdir, cell_type_order) {
    # Read total read counts from pat files for each cell type
    if (!dir.exists(patdir)) {
        stop(sprintf("Directory does not exist: %s", patdir))
    }
    files <- paste0(patdir, "/", cell_type_order, ".pat.gz")
    missing_files <- files[!file.exists(files)]
    if (length(missing_files) > 0) {
        stop(sprintf("Missing .pat.gz files: %s", paste(basename(missing_files), collapse=", ")))
    }
    names(files) <- cell_type_order
    all_frags_list <- lapply(files, function(file_name) {
        file_info <- file.info(file_name)
        if (file_info$size == 0) {
            cat(sprintf("Skipping empty file: %s\n", file_name))
            return(NULL)
        }
        cmd <- sprintf("zcat %s", shQuote(file_name))
        data <- tryCatch({
            fread(cmd = cmd, stringsAsFactors = TRUE, header = FALSE, select = 4)
        }, error = function(e) {
            warning(sprintf("Error reading file %s: %s", file_name, e$message))
            return(NULL)
        })
        if (is.null(data) || ncol(data) == 0) {
            warning(sprintf("Failed to read data from: %s", file_name))
            return(NULL)
        }
        setnames(data, "V4", "counts")
        total_counts <- sum(data$counts, na.rm = TRUE)
        return(total_counts)
    })
    valid_indices <- !sapply(all_frags_list, is.null)
    all_frags_list <- all_frags_list[valid_indices]
    if (length(all_frags_list) == 0) {
        stop("No valid .pat.gz files were processed")
    }
    all_frags <- data.table(
        celltype = names(all_frags_list),
        fragments = unlist(all_frags_list)
    )
    return(all_frags)
}

process_single_sample <- function(sample_id, concentrations, pat_dir, output_dir, tmp_dir, min_depth, max_depth,
                                 overwrite, prefix, cell_type_order, reads_by_celltype_df) {
    # Function to process a single sample (to be run in parallel)
    
    # Create targets table
    targets <- make_target_table(
        cell_type_order = cell_type_order,
        concentrations = concentrations,
        pat_dir = pat_dir
    )
    
    # Sample current_depth
    current_depth <- round(runif(1, min_depth, max_depth))
    
    # Setup directories
    if (is.null(tmp_dir)) {
        tmp_dir <- paste0(output_dir, "/tmp")
    }
    
    # Create main directories
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE, mode = "0755")
    dir.create(tmp_dir, showWarnings = FALSE, recursive = TRUE, mode = "0755")
    
    # Generate mix prefix
    mix_prefix <- sprintf("%s_sample%d", prefix, sample_id)
    
    # Create a worker-specific temporary directory for this sample
    worker_tmp_dir <- file.path(tmp_dir, paste0("worker_", sample_id))
    dir.create(worker_tmp_dir, showWarnings = FALSE, recursive = TRUE, mode = "0755")
    
    # Process each cell type
    for (ct in cell_type_order) {
        out_file <- paste0(output_dir, '/', mix_prefix, '.pat.gz')
        if (overwrite || !file.exists(out_file)) {
            # Get the reads data for this cell type
            ct_row_idx <- which(reads_by_celltype_df[,"celltype"] == ct)
            if (length(ct_row_idx) == 0) {
                warning(sprintf("No data found for cell type: %s", ct))
                next
            }
            
            # Get necessary values
            fraction <- concentrations[ct]
            filename <- paste0(pat_dir, '/', ct, '.pat.gz')
            fragments <- reads_by_celltype_df[ct_row_idx, "fragments"]
            
            # Adjust fraction based on current_depth vs total fragments
            adjusted_fraction <- fraction * (current_depth / fragments)
            
            # Generate sampled pat file in the worker-specific temporary directory
            tmp_file <- sprintf("%s/%s_%s.pat.gz", worker_tmp_dir, mix_prefix, ct)
            cmd <- sprintf('"/users/zetzioni/sharedscratch/pattools sample -s %.8f %s | bgzip -c > %s"', 
                        adjusted_fraction, filename, tmp_file)
            result <- system2("sh", c("-c", cmd))
            if (result != 0) {
                warning(sprintf("Failed to sample reads for %s in sample %d", ct, sample_id))
            }
        }
    }
    
    # Merge pat files from the worker-specific temporary directory
    out_file <- paste0(output_dir, '/', mix_prefix, '.pat.gz')
    if (!file.exists(out_file) || overwrite) {
        merge_cmd <- paste0(
            '"zcat ', worker_tmp_dir, '/', mix_prefix, '_*.pat.gz | ',
            'sort -k1,1V -k2,2n -k3,3 | ',
            'perl -n /users/zetzioni/sharedscratch/atlas/deduplicate_pat.pl | ',
            'bgzip -c > ', out_file,
            '; tabix -s 1 -b 2 -e 2 -C ', out_file, '"'
        )
        result <- system2("sh", c("-c", merge_cmd))
    }
    
    # Calculate and save true concentrations
    calculate_true_concentrations(worker_tmp_dir, output_dir, mix_prefix, cell_type_order)
    
    # Cleanup worker-specific temporary files and directory
    tmp_files <- list.files(worker_tmp_dir, 
                           pattern = paste0(mix_prefix, "_.*\\.pat\\.gz$"), 
                           full.names = TRUE)
    file.remove(tmp_files)
    unlink(worker_tmp_dir, recursive = TRUE)
    
    return(sample_id)
}

calculate_true_concentrations <- function(tmp_dir, target_dir, mix_prefix, cell_type_order) {
    # Create a named vector to store counts in the specified order
    counts <- numeric(length(cell_type_order))
    names(counts) <- cell_type_order

    # Calculate counts for each cell type
    for(ct in cell_type_order) {
        # Construct exact filename instead of searching
        tmp_file <- file.path(tmp_dir, paste0(mix_prefix, "_", ct, ".pat.gz"))
        
        if(file.exists(tmp_file)) {
            counts[ct] <- tryCatch({
                con <- gzfile(tmp_file, open = "r")
                lines <- readLines(con, warn = FALSE)
                close(con)
                
                data <- strsplit(lines, "\t")
                total <- sum(sapply(data, function(row) {
                    if (length(row) >= 4 && grepl("^[0-9]+$", row[[4]])) {
                        as.numeric(row[[4]])
                    } else {
                        0
                    }
                }), na.rm = TRUE)
                total
            }, error = function(e) {
                warning(sprintf("Error reading counts for %s: %s", ct, e$message))
                0
            })
        }
    }

    # Calculate proportions
    total_counts <- sum(counts)
    concentrations <- if(total_counts > 0) counts / total_counts else counts

    # Create sorted output
    result <- data.frame(matrix(concentrations[sort(names(concentrations))], 
                               nrow=1,
                               dimnames=list(NULL, sort(names(concentrations)))),
                        check.names=FALSE)
    
    out_file <- file.path(target_dir, 
                         paste0(mix_prefix, "_true_concentrations.csv"))
    
    write.table(result, 
                out_file, 
                row.names=FALSE, 
                sep=",", 
                col.names=TRUE, 
                quote=FALSE)
                
    return(concentrations)
}

process_batch <- function(batch_indices, all_concentrations, args, cell_type_order, reads_by_celltype_df) {
    # Process a batch of samples
    registerDoParallel(cores=args$threads)
    
    results <- foreach(i=batch_indices, 
                      .packages=c("data.table", "gtools"),
                      .export=c("make_target_table", "calculate_true_concentrations", 
                                "process_single_sample")) %dopar% {
        # Process one sample
        process_single_sample(
            sample_id = i,
            concentrations = all_concentrations[[i]],
            pat_dir = args$pat_dir,
            output_dir = args$output_dir,
            tmp_dir = args$tmp_dir,
            min_depth = args$min_depth,
            max_depth = args$max_depth,
            overwrite = args$overwrite,
            prefix = args$prefix,
            cell_type_order = cell_type_order,
            reads_by_celltype_df = reads_by_celltype_df
        )
    }
    
    stopImplicitCluster()
    return(results)
}

main <- function() {
    # Parse command-line arguments
    parser <- OptionParser(option_list=option_list)
    args <- parse_args(parser)
    debug_mode <- !is.null(args$debug) && args$debug
    
    if (debug_mode) {
        log_info("Running in DEBUG mode")
    }
    
    if (is.null(args$pat_dir) || is.null(args$output_dir) || is.null(args$concentrations)) {
        print_help(parser)
        stop("Missing required arguments")
    }
    if (is.null(args$tmp_dir)) {
        args$tmp_dir <- paste0(args$output_dir, "/tmp")
    }
    
    # Read JSON configuration
    log_info("Reading JSON config file")
    json_text <- readLines(args$concentrations, warn = FALSE)
    json_text <- paste(json_text, collapse = "")
    
    # Parse JSON with simplifyVector=FALSE to prevent data.frame conversion
    json_data <- fromJSON(json_text, simplifyVector = FALSE)
    
    if (debug_mode) {
        print_debug("json_data", json_data, debug_mode)
    }
    
    # Validate JSON structure
    if (is.null(json_data$cell_type_order) || is.null(json_data$target_cell_type) || 
        is.null(json_data$distribution)) {
        stop("JSON must contain 'cell_type_order', 'target_cell_type', and 'distribution'")
    }
    
    # Extract configuration
    cell_type_order <- unlist(json_data$cell_type_order)
    target_cell_type <- json_data$target_cell_type
    distribution <- json_data$distribution
    
    if (debug_mode) {
        print_debug("cell_type_order", cell_type_order, debug_mode)
        print_debug("target_cell_type", target_cell_type, debug_mode)
        print_debug("distribution", distribution, debug_mode)
    }
    
    # Validate distribution probabilities
    log_info("Validating distribution probabilities")
    
    # Access probability values safely
    probs <- numeric(length(distribution))
    for (i in 1:length(distribution)) {
        if (debug_mode) {
            print_debug(paste0("distribution[[", i, "]]"), distribution[[i]], debug_mode)
        }
        probs[i] <- distribution[[i]]$probability
    }
    
    if (debug_mode) {
        print_debug("probs", probs, debug_mode)
    }
    
    if (abs(sum(probs) - 1) > 1e-6) {
        stop("Distribution probabilities must sum to 1")
    }
    
    # Read total read counts for each cell type
    log_info("Reading fragment counts from PAT files")
    reads_by_celltype <- read_count_table(args$pat_dir, cell_type_order)
    
    # Generate bin indices for all samples
    log_info(sprintf("Generating %d samples", args$num_samples))
    set.seed(42)  # For reproducibility
    bin_indices <- sample(1:length(distribution), args$num_samples, replace=TRUE, prob=probs)
    
    # Create all sample concentration specs
    all_concentrations <- list()
    
    log_info("Creating concentration specifications")
    for (i in 1:args$num_samples) {
        # Get the selected distribution bin
        bin <- distribution[[bin_indices[i]]]
        
        if (debug_mode && i == 1) {
            print_debug("selected bin", bin, debug_mode)
            print_debug("bin$range", bin$range, debug_mode)
        }
        
        # Get the target concentration from the range
        range_values <- unlist(bin$range)
        
        if (range_values[1] == range_values[2]) {
            c <- range_values[1]
        } else {
            c <- runif(1, min(range_values), max(range_values))
        }
        
        # Get other cell types
        other_cell_types <- setdiff(cell_type_order, target_cell_type)
        k <- length(other_cell_types)
        
        # Create concentration vector
        if (c >= 1) {
            concentrations <- setNames(rep(0, length(cell_type_order)), cell_type_order)
            concentrations[target_cell_type] <- 1
        } else {
            d <- rdirichlet(1, rep(1, k))
            d_scaled <- d * (1 - c)
            concentrations <- setNames(c(d_scaled[1,], c), c(other_cell_types, target_cell_type))
        }
        
        # Re-order to match cell_type_order
        all_concentrations[[i]] <- concentrations[cell_type_order]
        
        if (debug_mode && i == 1) {
            print_debug("concentrations for sample 1", all_concentrations[[i]], debug_mode)
        }
    }
    
    # Initialize progress tracking
    init_progress(args$output_dir, args$num_samples)
    
    # If in debug mode, just process the first sample
    if (debug_mode) {
        log_info("DEBUG mode: Processing only first sample")
        reads_by_celltype_df <- as.data.frame(reads_by_celltype)
        
        result <- process_single_sample(
            sample_id = 1,
            concentrations = all_concentrations[[1]],
            pat_dir = args$pat_dir,
            output_dir = args$output_dir,
            tmp_dir = args$tmp_dir,
            min_depth = args$min_depth,
            max_depth = args$max_depth,
            overwrite = args$overwrite,
            prefix = args$prefix,
            cell_type_order = cell_type_order,
            reads_by_celltype_df = reads_by_celltype_df
        )
        
        log_info("Debug sample processed successfully")
        update_progress(1)
    } else {
        # Convert to data frame for easier parallel processing
        reads_by_celltype_df <- as.data.frame(reads_by_celltype)
        
        # Process in batches for better progress reporting
        log_info(sprintf("Processing %d samples in batches of %d", args$num_samples, args$batch_size))
        total_processed <- 0
        
        batch_size <- min(args$batch_size, args$num_samples)
        num_batches <- ceiling(args$num_samples / batch_size)
        
        for (batch in 1:num_batches) {
            start_idx <- (batch - 1) * batch_size + 1
            end_idx <- min(batch * batch_size, args$num_samples)
            batch_indices <- start_idx:end_idx
            
            log_info(sprintf("Processing batch %d/%d (samples %d-%d)", 
                            batch, num_batches, start_idx, end_idx))
            
            # Process this batch
            batch_results <- process_batch(
                batch_indices, 
                all_concentrations, 
                args, 
                cell_type_order, 
                reads_by_celltype_df
            )
            
            # Update progress
            total_processed <- total_processed + length(batch_results)
            update_progress(total_processed)
        }
        
        log_info(sprintf("All batches complete. Processed %d/%d samples.", 
                        total_processed, args$num_samples))
    }
    
    log_info("Script completed successfully")
}

# Wrap with error handling
tryCatch({
    main()
}, error = function(e) {
    cat(sprintf("ERROR: %s\n", e$message))
    cat("Error call:\n")
    print(e$call)
    cat("Traceback:\n")
    print(traceback())
    quit(status = 1)
})