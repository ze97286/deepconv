#!/usr/bin/env Rscript
suppressPackageStartupMessages({
    library(data.table)
    library(optparse)
    library(jsonlite)
    library(doParallel)
})

# R_LIBS_USER=~/R/library Rscript /users/zetzioni/sharedscratch/deepconv/src/deep_conv/atlas/generate_training_data.R --pat_dir /users/zetzioni/sharedscratch/pat/snr_fixed.blood+gi+tum.l4/Song/celltypes --output_dir /users/zetzioni/sharedscratch/atlas/training --tmp_dir /users/zetzioni/sharedscratch/atlas/tmp --threads 5 --n_train 10 --n_eval 2

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

# Function to process one batch of samples in parallel
process_batch <- function(concentrations, prefix, out_dir, threads, tmp_base_dir, pat_dir, target_depth, reps_per_combo) {
    registerDoParallel(cores=threads)
    
    foreach(i=1:nrow(concentrations)) %dopar% {
        # Create unique temp dir for this combination
        tmp_dir <- file.path(tmp_base_dir, sprintf("tmp_%s_%d", prefix, i))
        dir.create(tmp_dir, recursive=TRUE, showWarnings=FALSE)
        
        conc_json <- toJSON(as.list(concentrations[i]))
        
        # Call the admixing script
        system2("Rscript", c(
            "/users/zetzioni/sharedscratch/deepconv/src/deep_conv/atlas/admix_pats.R",
            "--pat_dir", pat_dir,
            "--output_dir", out_dir,
            "--tmp_dir", tmp_dir,
            "--target_depth", target_depth,
            "--threads", "1",  # Single thread here since we're parallel at combination level
            "--concentrations", shQuote(conc_json),
            "--prefix", paste0(prefix, "_", i),
            "--repeats", reps_per_combo
        ))
        
        # Cleanup temp directory after processing
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
    
    # Save concentration ground truth
    cat("Saving initial concentration tables...\n")
    fwrite(train_concentrations, file.path(args$output_dir, "train_concentrations.csv"))
    fwrite(eval_concentrations, file.path(args$output_dir, "eval_concentrations.csv"))
    
    # Create base tmp directory
    tmp_base_dir <- if(is.null(args$tmp_dir)) file.path(args$output_dir, "tmp") else args$tmp_dir
    dir.create(tmp_base_dir, recursive=TRUE, showWarnings=FALSE)
    
    # Process training and evaluation sets
    cat(sprintf("Processing %d training samples with %d threads...\n", args$n_train, args$threads))
    process_batch(
        train_concentrations, "train", train_dir, args$threads, tmp_base_dir,
        args$pat_dir, args$target_depth, args$reps_per_combo
    )
    
    cat(sprintf("Processing %d evaluation samples with %d threads...\n", args$n_eval, args$threads))
    process_batch(
        eval_concentrations, "eval", eval_dir, args$threads, tmp_base_dir,
        args$pat_dir, args$target_depth, args$reps_per_combo
    )
    
    # Final cleanup
    unlink(tmp_base_dir, recursive=TRUE)
    cat("Processing completed.\n")
}

if(sys.nframe() == 0) {
    main()
}