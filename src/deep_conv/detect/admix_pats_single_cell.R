#!/usr/bin/env Rscript

# Add a custom logging function
log_info <- function(message) {
  cat(sprintf("[INFO] %s: %s\n", format(Sys.time(), "%H:%M:%S"), message))
}

log_debug <- function(message) {
  cat(sprintf("[DEBUG] %s: %s\n", format(Sys.time(), "%H:%M:%S"), message))
}

log_error <- function(message) {
  cat(sprintf("[ERROR] %s: %s\n", format(Sys.time(), "%H:%M:%S"), message))
}

log_object <- function(name, obj) {
  cat(sprintf("[DEBUG] %s: %s is %s, class: %s\n", 
              format(Sys.time(), "%H:%M:%S"), 
              name, 
              typeof(obj),
              paste(class(obj), collapse=", ")))
}

# Start with logging what packages we're loading
log_info("Starting script - loading packages")

suppressPackageStartupMessages({
    library(data.table)
    library(optparse)
    library(doParallel)
    library(jsonlite)
    library(gtools)  # For rdirichlet
})

log_info("All packages loaded")

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
                help="Run in debug mode [default %default]")
)

# Helper functions
make_target_table <- function(cell_type_order, concentrations, pat_dir=".", suffix=".pat.gz") {
    log_debug("make_target_table: Starting")
    log_object("cell_type_order", cell_type_order)
    log_object("concentrations", concentrations)
    
    # Create data table for a single mixture
    conc_table <- data.table(
        celltype = cell_type_order,
        fraction = concentrations,
        filename = paste0(pat_dir, '/', cell_type_order, suffix),
        dilution = 1
    )
    
    log_debug("make_target_table: Created conc_table")
    log_object("conc_table", conc_table)
    
    return(conc_table)
}

read_count_table <- function(patdir, cell_type_order) {
    log_debug("read_count_table: Starting")
    log_debug(paste("patdir:", patdir))
    log_object("cell_type_order", cell_type_order)
    
    # Read total read counts from pat files for each cell type
    if (!dir.exists(patdir)) {
        log_error(sprintf("Directory does not exist: %s", patdir))
        stop(sprintf("Directory does not exist: %s", patdir))
    }
    
    log_debug("read_count_table: Directory exists, continuing")
    files <- paste0(patdir, "/", cell_type_order, ".pat.gz")
    log_debug(paste("Files to check:", paste(files, collapse=", ")))
    
    missing_files <- files[!file.exists(files)]
    if (length(missing_files) > 0) {
        log_error(sprintf("Missing .pat.gz files: %s", paste(basename(missing_files), collapse=", ")))
        stop(sprintf("Missing .pat.gz files: %s", paste(basename(missing_files), collapse=", ")))
    }
    
    log_debug("read_count_table: All files exist, continuing")
    names(files) <- cell_type_order
    
    all_frags_list <- list()
    for (i in 1:length(files)) {
        file_name <- files[i]
        ct <- names(files)[i]
        log_debug(sprintf("Processing file: %s for cell type: %s", file_name, ct))
        
        file_info <- file.info(file_name)
        if (file_info$size == 0) {
            log_debug(sprintf("Skipping empty file: %s", file_name))
            next
        }
        
        cmd <- sprintf("zcat %s", shQuote(file_name))
        log_debug(sprintf("Running command: %s", cmd))
        
        tryCatch({
            data <- fread(cmd = cmd, stringsAsFactors = TRUE, header = FALSE, select = 4)
            setnames(data, "V4", "counts")
            total_counts <- sum(data$counts, na.rm = TRUE)
            log_debug(sprintf("Total counts for %s: %d", ct, total_counts))
            all_frags_list[[ct]] <- total_counts
        }, error = function(e) {
            log_error(sprintf("Error reading file %s: %s", file_name, e$message))
        })
    }
    
    # Check if we successfully processed any files
    valid_indices <- !sapply(all_frags_list, is.null)
    
    if (sum(valid_indices) == 0) {
        log_error("No valid .pat.gz files were processed")
        stop("No valid .pat.gz files were processed")
    }
    
    log_debug("read_count_table: Creating final data table")
    
    # Create data frame with celltype and fragments columns
    all_frags <- data.frame(
        celltype = names(all_frags_list),
        fragments = unlist(all_frags_list)
    )
    # Convert to data.table
    all_frags <- as.data.table(all_frags)
    
    log_debug("read_count_table: Final results")
    log_object("all_frags", all_frags)
    print(all_frags)
    
    return(all_frags)
}

process_single_sample <- function(sample_id, concentrations, pat_dir, output_dir, tmp_dir, min_depth, max_depth,
                                 overwrite, prefix, cell_type_order, reads_by_celltype_df) {
    # Function to process a single sample (to be run in parallel)
    log_debug(sprintf("process_single_sample: Starting for sample %d", sample_id))
    log_object("concentrations", concentrations)
    log_object("reads_by_celltype_df", reads_by_celltype_df)
    
    # Convert reads_by_celltype_df to data.frame if it's not already
    if (!is.data.frame(reads_by_celltype_df)) {
        log_debug("Converting reads_by_celltype_df to data.frame")
        reads_by_celltype_df <- as.data.frame(reads_by_celltype_df)
    }
    
    # Create targets table
    log_debug("Creating targets table")
    targets <- make_target_table(
        cell_type_order = cell_type_order,
        concentrations = concentrations,
        pat_dir = pat_dir
    )
    
    # Sample current_depth
    current_depth <- round(runif(1, min_depth, max_depth))
    log_debug(sprintf("Selected current_depth: %d", current_depth))
    
    # Setup directories
    if (is.null(tmp_dir)) {
        tmp_dir <- paste0(output_dir, "/tmp")
    }
    
    # Create main directories
    log_debug(sprintf("Creating directories: %s and %s", output_dir, tmp_dir))
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE, mode = "0755")
    dir.create(tmp_dir, showWarnings = FALSE, recursive = TRUE, mode = "0755")
    
    # Generate mix prefix
    mix_prefix <- sprintf("%s_sample%d", prefix, sample_id)
    
    # Create a worker-specific temporary directory for this sample
    worker_tmp_dir <- file.path(tmp_dir, paste0("worker_", sample_id))
    log_debug(sprintf("Creating worker-specific tmp dir: %s", worker_tmp_dir))
    dir.create(worker_tmp_dir, showWarnings = FALSE, recursive = TRUE, mode = "0755")
    
    # Process each cell type
    for (ct in cell_type_order) {
        log_debug(sprintf("Processing cell type: %s", ct))
        
        out_file <- paste0(output_dir, '/', mix_prefix, '.pat.gz')
        if (overwrite || !file.exists(out_file)) {
            # Get the reads data for this cell type
            log_debug(sprintf("Finding data for cell type: %s", ct))
            
            # Print out the structure to debug
            log_debug("Structure of reads_by_celltype_df:")
            str(reads_by_celltype_df)
            
            # Find the row for this cell type
            ct_row_idx <- which(reads_by_celltype_df$celltype == ct)
            log_debug(sprintf("Row index for %s: %s", ct, paste(ct_row_idx, collapse=", ")))
            
            if (length(ct_row_idx) == 0) {
                log_error(sprintf("No data found for cell type: %s", ct))
                next
            }
            
            # Get necessary values
            log_debug("Getting necessary values")
            fraction <- concentrations[ct]
            log_debug(sprintf("Fraction for %s: %f", ct, fraction))
            
            filename <- paste0(pat_dir, '/', ct, '.pat.gz')
            log_debug(sprintf("Filename: %s", filename))
            
            # Access fragments directly from the data frame
            fragments <- reads_by_celltype_df$fragments[ct_row_idx]
            log_debug(sprintf("Fragments: %s", paste(fragments, collapse=", ")))
            
            # Adjust fraction based on current_depth vs total fragments
            adjusted_fraction <- fraction * (current_depth / fragments)
            log_debug(sprintf("Adjusted fraction: %f", adjusted_fraction))
            
            # Generate sampled pat file in the worker-specific temporary directory
            tmp_file <- sprintf("%s/%s_%s.pat.gz", worker_tmp_dir, mix_prefix, ct)
            log_debug(sprintf("Temporary file: %s", tmp_file))
            
            cmd <- sprintf('"/users/zetzioni/sharedscratch/pattools sample -s %.8f %s | bgzip -c > %s"', 
                        adjusted_fraction, filename, tmp_file)
            log_debug(sprintf("Running command: %s", cmd))
            
            result <- system2("sh", c("-c", cmd))
            if (result != 0) {
                log_error(sprintf("Failed to sample reads for %s in sample %d, return code: %d", 
                                 ct, sample_id, result))
            }
            
            # Verify the tmp file was created and has content
            if (file.exists(tmp_file)) {
                file_info <- file.info(tmp_file)
                log_debug(sprintf("File created, size: %d bytes", file_info$size))
                if (file_info$size == 0) {
                    log_error(sprintf("Empty tmp file created: %s", tmp_file))
                }
            } else {
                log_error(sprintf("Temp file not created: %s", tmp_file))
            }
        }
    }
    
    # Merge pat files from the worker-specific temporary directory
    out_file <- paste0(output_dir, '/', mix_prefix, '.pat.gz')
    log_debug(sprintf("Merging files to: %s", out_file))
    
    if (!file.exists(out_file) || overwrite) {
        merge_cmd <- paste0(
            '"zcat ', worker_tmp_dir, '/', mix_prefix, '_*.pat.gz | ',
            'sort -k1,1V -k2,2n -k3,3 | ',
            'perl -n /users/zetzioni/sharedscratch/atlas/deduplicate_pat.pl | ',
            'bgzip -c > ', out_file,
            '; tabix -s 1 -b 2 -e 2 -C ', out_file, '"'
        )
        log_debug(sprintf("Running merge command: %s", merge_cmd))
        
        result <- system2("sh", c("-c", merge_cmd))
        if (result != 0) {
            log_error(sprintf("Merge command failed for sample %d, return code: %d", 
                             sample_id, result))
        }
        
        if (file.exists(out_file)) {
            file_info <- file.info(out_file)
            log_debug(sprintf("Merged file created, size: %d bytes", file_info$size))
            if (file_info$size == 0) {
                log_error(sprintf("Empty merged file created: %s", out_file))
            }
        } else {
            log_error(sprintf("Merged file not created: %s", out_file))
        }
    }
    
    # Calculate and save true concentrations
    log_debug("Calculating true concentrations")
    calculate_true_concentrations(worker_tmp_dir, output_dir, mix_prefix, cell_type_order)
    
    # Cleanup worker-specific temporary files and directory
    log_debug("Cleaning up temporary files")
    tmp_files <- list.files(worker_tmp_dir, 
                           pattern = paste0(mix_prefix, "_.*\\.pat\\.gz$"), 
                           full.names = TRUE)
    file.remove(tmp_files)
    unlink(worker_tmp_dir, recursive = TRUE)
    
    log_info(sprintf("Finished processing sample: %d", sample_id))
    return(sample_id)
}

calculate_true_concentrations <- function(tmp_dir, target_dir, mix_prefix, cell_type_order) {
    log_debug("calculate_true_concentrations: Starting")
    log_object("cell_type_order", cell_type_order)
    
    # Create a named vector to store counts in the specified order
    counts <- numeric(length(cell_type_order))
    names(counts) <- cell_type_order
    log_debug("Created empty counts vector")

    # Calculate counts for each cell type
    for(ct in cell_type_order) {
        log_debug(sprintf("Processing cell type: %s", ct))
        
        # Construct exact filename instead of searching
        tmp_file <- file.path(tmp_dir, paste0(mix_prefix, "_", ct, ".pat.gz"))
        log_debug(sprintf("Looking for file: %s", tmp_file))
        
        if(file.exists(tmp_file)) {
            log_debug(sprintf("File exists: %s", tmp_file))
            tryCatch({
                con <- gzfile(tmp_file, open = "r")
                lines <- readLines(con, warn = FALSE)
                close(con)
                
                log_debug(sprintf("Read %d lines from file", length(lines)))
                
                data <- strsplit(lines, "\t")
                total <- sum(sapply(data, function(row) {
                    if (length(row) >= 4 && grepl("^[0-9]+$", row[[4]])) {
                        as.numeric(row[[4]])
                    } else {
                        0
                    }
                }), na.rm = TRUE)
                
                log_debug(sprintf("Total count for %s: %d", ct, total))
                counts[ct] <- total
            }, error = function(e) {
                log_error(sprintf("Error reading counts for %s: %s", ct, e$message))
                counts[ct] <- 0
            })
        } else {
            log_error(sprintf("File not found: %s", tmp_file))
        }
    }

    # Calculate proportions
    total_counts <- sum(counts)
    log_debug(sprintf("Total counts across all cell types: %d", total_counts))
    
    concentrations <- if(total_counts > 0) counts / total_counts else counts
    log_debug("Calculated proportions")
    log_debug(paste(names(concentrations), concentrations, sep=": ", collapse=", "))

    # Create sorted output
    result <- data.frame(matrix(concentrations[sort(names(concentrations))], 
                               nrow=1,
                               dimnames=list(NULL, sort(names(concentrations)))),
                        check.names=FALSE)
    
    out_file <- file.path(target_dir, 
                         paste0(mix_prefix, "_true_concentrations.csv"))
    
    log_debug(sprintf("Writing results to: %s", out_file))
    write.table(result, 
                out_file, 
                row.names=FALSE, 
                sep=",", 
                col.names=TRUE, 
                quote=FALSE)
    
    log_debug("Finished calculating true concentrations")
    return(concentrations)
}

main <- function() {
    log_info("Starting main function")
    
    # Parse command-line arguments
    log_debug("Parsing command-line arguments")
    parser <- OptionParser(option_list=option_list)
    args <- parse_args(parser)
    
    log_debug("Command-line arguments:")
    for (arg_name in names(args)) {
        log_debug(sprintf("  %s: %s", arg_name, as.character(args[[arg_name]])))
    }
    
    if (is.null(args$pat_dir) || is.null(args$output_dir) || is.null(args$concentrations)) {
        log_error("Missing required arguments")
        print_help(parser)
        stop("Missing required arguments")
    }
    
    if (is.null(args$tmp_dir)) {
        args$tmp_dir <- paste0(args$output_dir, "/tmp")
        log_debug(sprintf("tmp_dir not provided, using: %s", args$tmp_dir))
    }
    
    # Read JSON configuration
    log_debug(sprintf("Reading JSON configuration from: %s", args$concentrations))
    json_data <- tryCatch({
        if (file.exists(args$concentrations)) {
            log_debug("JSON file exists, reading directly")
            fromJSON(args$concentrations)
        } else {
            log_debug("JSON file doesn't exist, trying to parse as JSON string")
            fromJSON(args$concentrations)
        }
    }, error=function(e) {
        log_error(sprintf("Failed to parse concentrations JSON: %s", e$message))
        stop("Failed to parse concentrations JSON: ", e$message)
    })
    
    # Log the JSON structure
    log_debug("JSON structure:")
    log_debug(paste("Keys:", paste(names(json_data), collapse=", ")))
    
    # Validate JSON structure
    if (is.null(json_data$cell_type_order) || is.null(json_data$target_cell_type) || 
        is.null(json_data$distribution)) {
        log_error("JSON must contain 'cell_type_order', 'target_cell_type', and 'distribution'")
        stop("JSON must contain 'cell_type_order', 'target_cell_type', and 'distribution'")
    }
    
    # Extract configuration
    cell_type_order <- json_data$cell_type_order
    target_cell_type <- json_data$target_cell_type
    distribution <- json_data$distribution
    
    log_debug(sprintf("Cell type order: %s", paste(cell_type_order, collapse=", ")))
    log_debug(sprintf("Target cell type: %s", target_cell_type))
    log_debug(sprintf("Distribution has %d bins", length(distribution)))
    
    # Validate distribution probabilities
    probs <- sapply(distribution, function(b) b$probability)
    log_debug(sprintf("Distribution probabilities: %s", paste(probs, collapse=", ")))
    
    if (abs(sum(probs) - 1) > 1e-6) {
        log_error(sprintf("Distribution probabilities sum to %f, not 1", sum(probs)))
        stop("Distribution probabilities must sum to 1")
    }
    
    # Read total read counts for each cell type
    log_debug("Reading fragment counts from PAT files")
    reads_by_celltype <- read_count_table(args$pat_dir, cell_type_order)
    log_debug("Fragment counts read successfully")
    
    # Generate bin indices for all samples
    log_debug(sprintf("Sampling %d bin indices according to probabilities", args$num_samples))
    set.seed(42)  # For reproducibility
    bin_indices <- sample(1:length(distribution), args$num_samples, replace=TRUE, prob=probs)
    log_debug("Bin indices sampled")
    
    # Create all sample concentration specs 
    log_debug("Creating concentration specifications for all samples")
    all_concentrations <- list()
    
    # Just process one sample for testing if in debug mode
    num_samples_to_process <- if(args$debug) min(1, args$num_samples) else args$num_samples
    log_debug(sprintf("Will process %d samples", num_samples_to_process))
    
    for (i in 1:num_samples_to_process) {
        log_debug(sprintf("Creating concentration for sample %d", i))
        bin <- distribution[[bin_indices[i]]]
        log_debug(sprintf("Using bin with range [%f, %f]", bin$range[1], bin$range[2]))
        
        if (bin$range[1] == bin$range[2]) {
            c <- bin$range[1]
            log_debug(sprintf("Using exact value: %f", c))
        } else {
            c <- runif(1, bin$range[1], bin$range[2])
            log_debug(sprintf("Sampled value from range: %f", c))
        }
        
        other_cell_types <- setdiff(cell_type_order, target_cell_type)
        k <- length(other_cell_types)
        log_debug(sprintf("Number of other cell types: %d", k))
        
        # Create concentration vector
        if (c >= 1) {
            log_debug("Target concentration >= 1, setting target to 1 and others to 0")
            concentrations <- setNames(rep(0, length(cell_type_order)), cell_type_order)
            concentrations[target_cell_type] <- 1
        } else {
            log_debug("Sampling Dirichlet distribution for other cell types")
            set.seed(i + 100)  # Different seed for each sample
            d <- rdirichlet(1, rep(1, k))
            d_scaled <- d * (1 - c)
            log_debug("Creating combined concentration vector")
            concentrations <- setNames(c(d_scaled[1,], c), c(other_cell_types, target_cell_type))
        }
        
        # Re-order to match cell_type_order
        log_debug("Re-ordering concentrations to match cell_type_order")
        all_concentrations[[i]] <- concentrations[cell_type_order]
        
        # Log the final concentrations
        log_debug("Final concentrations:")
        for (ct in cell_type_order) {
            log_debug(sprintf("  %s: %f", ct, all_concentrations[[i]][ct]))
        }
    }
    
    # If in debug mode, just process the first sample without parallelism
    if (args$debug) {
        log_info("Debug mode: Processing sample 1 without parallelism")
        process_single_sample(
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
            reads_by_celltype_df = reads_by_celltype
        )
        log_info("Debug processing complete")
    } else {
        # Set up parallel processing
        log_info(sprintf("Setting up parallel processing with %d threads", args$threads))
        registerDoParallel(cores=args$threads)
        
        # Make sure reads_by_celltype is a data frame for parallel processing
        reads_by_celltype_df <- as.data.frame(reads_by_celltype)
        log_debug("Converted reads_by_celltype to data.frame for parallel processing")
        log_object("reads_by_celltype_df", reads_by_celltype_df)
        
        # Generate mixtures in parallel
        log_info("Starting parallel processing")
        results <- foreach(i=1:num_samples_to_process, 
                          .packages=c("data.table", "gtools"),
                          .export=c("make_target_table", "calculate_true_concentrations", 
                                    "process_single_sample", "log_debug", "log_info", 
                                    "log_error", "log_object")) %dopar% {
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
        
        log_info(sprintf("Parallel processing complete. Processed %d samples.", length(results)))
        stopImplicitCluster()
    }
    
    log_info("Script execution complete")
}

# Check if this script is being run directly
if (sys.nframe() == 0) {
    tryCatch({
        log_info("Script starting")
        main()
    }, error = function(e) {
        # Print detailed error info
        log_error(sprintf("ERROR OCCURRED: %s", e$message))
        log_error(sprintf("In: %s", e$call))
        print(traceback())
        # Re-throw the error
        stop(e)
    })
}