#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(data.table)
    library(optparse)
    library(jsonlite)
    library(doParallel)
})

# Helper function for Dirichlet sampling
rdirichlet <- function(n, alpha) {
    # Generate gamma samples and normalize
    gamma_samples <- matrix(0, nrow=n, ncol=length(alpha))
    for(i in 1:length(alpha)) {
        gamma_samples[,i] <- rgamma(n, shape=alpha[i], rate=1)
    }
    # Normalize rows to get Dirichlet samples
    gamma_samples / rowSums(gamma_samples)
}

# Parse command line arguments
option_list <- list(
    make_option(c("-p", "--pat_dir"), type="character",
                help="Directory containing input pat files [REQUIRED]"),
    make_option(c("-o", "--output_dir"), type="character",
                help="Output directory for mixed samples [REQUIRED]"),
    make_option(c("--tmp_dir"), type="character",
                help="Temporary directory for intermediate files [default: output_dir/tmp]"),
    make_option(c("-d", "--target_depth"), type="integer", default=50000,
                help="Target depth for generated mixtures [default %default]"),
    make_option(c("-t", "--threads"), type="integer", default=1,
                help="Number of threads to use [default %default]"),
    make_option(c("--n_train"), type="integer", default=100000,
                help="Number of training samples to generate [default %default]"),
    make_option(c("--n_eval"), type="integer", default=20000,
                help="Number of evaluation samples to generate [default %default]"),
    make_option(c("--reps_per_combo"), type="integer", default=10,
                help="Number of repetitions per combination [default %default]")
)

# Function to read count table from pat files
read_count_table <- function(patdir) {
    # First check the directory exists
    if (!dir.exists(patdir)) {
        stop(sprintf("Directory does not exist: %s", patdir))
    }

    # List and check files
    files <- list.files(patdir, pattern = ".*\\.pat\\.gz$", full.names = TRUE)
    if (length(files) == 0) {
        stop(sprintf("No .pat.gz files found in: %s", patdir))
    }
    
    cat("Found files:\n")
    print(files)
    
    # Assign names to files
    names(files) <- gsub(".pat.gz", "", basename(files), fixed = TRUE)
    
    # Initialize list to store fragment counts
    all_frags_list <- lapply(files, function(file_name) {
        cat(sprintf("\nProcessing file: %s\n", file_name))
        
        # Check if file exists
        if (!file.exists(file_name)) {
            stop(sprintf("File does not exist: %s", file_name))
        }
        
        # Check file size
        file_info <- file.info(file_name)
        if (file_info$size == 0) {
            cat(sprintf("Skipping empty file: %s\n", file_name))
            return(NULL)
        }
        
        # Construct and run the command
        cmd <- sprintf("zcat %s", shQuote(file_name))
        cat(sprintf("Running command: %s\n", cmd))
        
        # Read the data using fread
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
        cat(sprintf("Total counts for %s: %d\n", file_name, total_counts))
        return(total_counts)
    })
    
    # Remove NULL entries
    valid_indices <- !sapply(all_frags_list, is.null)
    all_frags_list <- all_frags_list[valid_indices]
    
    if (length(all_frags_list) == 0) {
        stop("No valid .pat.gz files were processed")
    }
    
    all_frags <- data.table(
        sample = names(all_frags_list),
        fragments = unlist(all_frags_list)
    )
    
    return(all_frags)
}

# Function to generate random concentrations
generate_stratified_concentrations <- function(n_samples, cell_types) {
    # Define proportion ranges similar to the Python script
    ranges <- list(
        list(min=0, max=1e-4, name="Ultra-low", frac=0.3),
        list(min=1e-4, max=1e-3, name="Very-low", frac=0.25),
        list(min=1e-3, max=1e-2, name="Low", frac=0.2),
        list(min=1e-2, max=1e-1, name="Medium", frac=0.15),
        list(min=1e-1, max=1.0, name="High", frac=0.1)
    )
    
    n_cell_types <- length(cell_types)
    concentrations <- data.table()
    
    # Calculate samples per range
    for (range in ranges) {
        n_range_samples <- ceiling(n_samples * range$frac)
        for (i in 1:n_range_samples) {
            # Randomly select number of cell types for this range
            n_types_in_range <- sample(1:max(2, n_cell_types %/% 2), 1)
            
            # Select cell types for this range
            selected_types <- sample(cell_types, n_types_in_range)
            
            # Generate concentrations
            conc <- numeric(n_cell_types)
            names(conc) <- cell_types
            
            # For selected types, generate values in range
            selected_conc <- runif(n_types_in_range, range$min, range$max)
            conc[selected_types] <- selected_conc
            
            # Distribute remaining proportion
            remaining <- 1 - sum(selected_conc)
            if (remaining > 0) {
                other_types <- setdiff(cell_types, selected_types)
                if (length(other_types) > 0) {
                    other_conc <- remaining * rdirichlet(1, rep(1, length(other_types)))
                    conc[other_types] <- other_conc
                }
            }
            
            # Normalize to sum to 1
            conc <- conc / sum(conc)
            
            # Add to results
            concentrations <- rbindlist(list(
                concentrations,
                as.data.table(as.list(conc))
            ))
        }
    }
    
    # Shuffle and take exact number needed
    concentrations <- concentrations[sample(.N, min(.N, n_samples))]
    return(concentrations)
}

generate_samples <- function(conc_table, reads_by_celltype, target_depth, prefix, tmp_dir, out_dir, pat_dir, rep) {
    setkey(conc_table, celltype)
    setkey(reads_by_celltype, sample)
    
    target_dilutions <- conc_table[reads_by_celltype, nomatch=NULL][
        , list(celltype, filename=file.path(pat_dir, paste0(celltype, ".pat.gz")), 
              fraction=target_depth * fraction/fragments)
    ]
    
    rep_prefix <- paste0(prefix, "_", rep)
    
    # Sample from each cell type
    for (ct in target_dilutions$celltype) {
        sub.dt <- target_dilutions[celltype == ct]
        cmd <- sprintf('"/users/zetzioni/sharedscratch/pattools sample -s %.8f %s | bgzip -c > %s/%s_%s.pat.gz"',
                      sub.dt$fraction, sub.dt$filename, tmp_dir, rep_prefix, ct)
        cat(sprintf("Running sample command for %s\n", ct))
        result <- system2("sh", c("-c", cmd))
        if (result != 0) {
            stop(sprintf("Failed to sample reads for %s in replica %d", ct, rep))
        }
    }
}

merge_pat_files <- function(prefix, rep, tmp_dir, out_dir) {
    rep_prefix <- paste0(prefix, "_", rep)
    out_file <- file.path(out_dir, paste0(rep_prefix, ".pat.gz"))
    merged_temp <- file.path(tmp_dir, paste0("merged_", rep, ".tmp"))
    sorted_temp <- file.path(tmp_dir, paste0("sorted_", rep, ".tmp"))
    
    # List and check files
    pat_files <- sort(list.files(tmp_dir, pattern=paste0(rep_prefix, "_.*\\.pat\\.gz$"), full.names=TRUE))
    valid_files <- character()
    for (f in pat_files) {
        size <- file.info(f)$size
        if (size > 28) {
            valid_files <- c(valid_files, f)
        }
    }
    
    if (length(valid_files) == 0) {
        stop("No valid files to merge")
    }
    
    # Single command merge
    merge_cmd <- sprintf('cat %s | gunzip -c > %s', 
                        paste(valid_files, collapse=" "), 
                        merged_temp)
    result <- system2("sh", c("-c", merge_cmd))
    if (result != 0) stop("Merge failed")
    
    # Sort
    sort_cmd <- sprintf('sort -k1,1V -k2,2n -k3,3 %s > %s', merged_temp, sorted_temp)
    result <- system2("sh", c("-c", sort_cmd))
    if (result != 0) stop("Sort failed")
    
    # Deduplicate and compress
    dedup_cmd <- sprintf('cat %s | perl -n /users/zetzioni/sharedscratch/atlas/deduplicate_pat.pl | bgzip -c > %s',
                        sorted_temp, out_file)
    result <- system2("sh", c("-c", dedup_cmd))
    if (result != 0) stop("Deduplicate failed")
    
    # Index
    result <- system2("sh", c("-c", sprintf('tabix -s 1 -b 2 -e 2 -C %s', out_file)))
    if (result != 0) stop("Index failed")
    
    # Calculate and save true concentrations
    counts <- lapply(pat_files, function(f) {
        cmd <- sprintf("zcat %s | wc -l", f)
        count <- as.numeric(system2("sh", c("-c", cmd), stdout=TRUE))
        data.table(celltype=sub(paste0(rep_prefix, "_(.*)\\.pat\\.gz"), "\\1", basename(f)), 
                  count=count)
    })
    counts <- rbindlist(counts)
    
    total_reads <- sum(counts$count)
    counts[, true_concentration := count/total_reads]
    
    wide_counts <- dcast(counts, . ~ celltype, value.var = "true_concentration")
    wide_counts[, `:=`(. = NULL, sample = rep_prefix)]
    fwrite(wide_counts, file.path(out_dir, paste0(rep_prefix, "_true_concentrations.csv")))
    
    # Cleanup
    unlink(c(merged_temp, sorted_temp))
}

# Function to process a batch of samples in parallel
# Inside process_batch:
process_batch <- function(concentrations, prefix, out_dir, threads, tmp_base_dir, pat_dir, target_depth, reads_by_celltype, reps_per_combo) {
    # Do all sampling in parallel
    registerDoParallel(cores=threads)
    
    # First phase: generate all pat files in parallel
    foreach(i=1:nrow(concentrations)) %dopar% {
        tmp_dir <- file.path(tmp_base_dir, sprintf("tmp_%s_%d", prefix, i))
        dir.create(tmp_dir, recursive=TRUE, showWarnings=FALSE)
        
        conc_table <- melt(concentrations[i], 
                          measure.vars = names(concentrations),
                          variable.name = "celltype",
                          value.name = "fraction")
        
        # Sample for each replica
        for (rep in 1:reps_per_combo) {
            generate_samples(conc_table, reads_by_celltype, target_depth, 
                           paste0(prefix, "_", i), tmp_dir, out_dir, pat_dir, rep)
        }
    }
    
    # Second phase: merge files serially for each combination and replica
    stopImplicitCluster()  # Stop parallel processing
    
    for (i in 1:nrow(concentrations)) {
        tmp_dir <- file.path(tmp_base_dir, sprintf("tmp_%s_%d", prefix, i))
        for (rep in 1:reps_per_combo) {
            merge_pat_files(paste0(prefix, "_", i), rep, tmp_dir, out_dir)
        }
        unlink(tmp_dir, recursive=TRUE)
    }
}

main <- function() {
   # Parse arguments
   parser <- OptionParser(option_list=option_list)
   args <- parse_args(parser)
   
   # Check required arguments
   if (is.null(args$pat_dir) || is.null(args$output_dir)) {
       print_help(parser)
       stop("Missing required arguments")
   }
   
   if(is.null(args$tmp_dir)) {
       args$tmp_dir <- paste0(args$output_dir, "/tmp")
   }
   
   # Get list of cell types from pat_dir
   cell_types <- gsub("\\.pat\\.gz$", "", list.files(args$pat_dir, pattern="\\.pat\\.gz$"))
   cat("Found cell types:", paste(cell_types, collapse=", "), "\n")
   
   # Check for cached read counts or calculate them
   counts_cache_file <- file.path(args$pat_dir, "read_counts.rds")
   if (file.exists(counts_cache_file)) {
       cat("Loading cached read counts...\n")
       reads_by_celltype <- readRDS(counts_cache_file)
   } else {
       cat("Calculating read counts...\n")
       reads_by_celltype <- read_count_table(args$pat_dir)
       saveRDS(reads_by_celltype, counts_cache_file)
   }
   # Generate training concentrations
   cat("Generating training concentrations...\n")
   train_concentrations <- generate_stratified_concentrations(args$n_train, cell_types)
   cat("Generating evaluation concentrations...\n")
   eval_concentrations <- generate_stratified_concentrations(args$n_eval, cell_types)
   
   # Create output directories
   train_dir <- file.path(args$output_dir, "train")
   eval_dir <- file.path(args$output_dir, "eval")
   dir.create(train_dir, recursive=TRUE, showWarnings=FALSE)
   dir.create(eval_dir, recursive=TRUE, showWarnings=FALSE)
   
   # Save initial concentration tables
   cat("Saving initial concentration tables...\n")
   fwrite(train_concentrations, file.path(args$output_dir, "train_concentrations.csv"))
   fwrite(eval_concentrations, file.path(args$output_dir, "eval_concentrations.csv"))
   
   # Create base tmp directory
   tmp_base_dir <- if(is.null(args$tmp_dir)) file.path(args$output_dir, "tmp") else args$tmp_dir
   dir.create(tmp_base_dir, recursive=TRUE, showWarnings=FALSE)
   
   # Process training and evaluation sets
   cat(sprintf("Processing %d training samples with %d threads...\n", args$n_train, args$threads))
   process_batch(train_concentrations, "train", train_dir, args$threads, 
                tmp_base_dir, args$pat_dir, args$target_depth, reads_by_celltype, 
                args$reps_per_combo)
   
   cat(sprintf("Processing %d evaluation samples with %d threads...\n", args$n_eval, args$threads))
   process_batch(eval_concentrations, "eval", eval_dir, args$threads, 
                tmp_base_dir, args$pat_dir, args$target_depth, reads_by_celltype,
                args$reps_per_combo)
   
   # Final cleanup
   unlink(tmp_base_dir, recursive=TRUE)
   cat("Processing completed.\n")
}

if(sys.nframe() == 0) {
    main()
}