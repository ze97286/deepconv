#!/usr/bin/env Rscript
suppressPackageStartupMessages({
    library(data.table)
    library(optparse)
    library(jsonlite)
})

# R_LIBS_USER=~/R/library Rscript /users/zetzioni/sharedscratch/deepconv/src/deep_conv/atlas/generate_training_data.R --pat_dir /users/zetzioni/sharedscratch/pat/snr_fixed.blood+gi+tum.l4/Song/celltypes --output_dir /users/zetzioni/sharedscratch/atlas/training --tmp_dir /users/zetzioni/sharedscratch/atlas/tmp --threads 5 --n_train 10 --n_eval 2

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
                help="Number of evaluation samples to generate [default %default]")
)

# Function to generate random concentrations following the python script's strategy
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
    
    # Generate training concentrations
    train_concentrations <- generate_stratified_concentrations(args$n_train, cell_types)
    eval_concentrations <- generate_stratified_concentrations(args$n_eval, cell_types)
    
    # Create output directories
    train_dir <- file.path(args$output_dir, "train")
    eval_dir <- file.path(args$output_dir, "eval")
    dir.create(train_dir, recursive=TRUE, showWarnings=FALSE)
    dir.create(eval_dir, recursive=TRUE, showWarnings=FALSE)
    
    # Save concentration ground truth
    fwrite(train_concentrations, file.path(args$output_dir, "train_concentrations.csv"))
    fwrite(eval_concentrations, file.path(args$output_dir, "eval_concentrations.csv"))
    
    # Function to process one batch of samples
    process_batch <- function(concentrations, prefix, out_dir) {
        for (i in 1:nrow(concentrations)) {
            conc_json <- toJSON(as.list(concentrations[i]))
            
            # Call the admixing script
            system2("Rscript", c(
                "/Users/zoharetzioni/Downloads/deepconv/src/deep_conv/atlas/admix_pats.R",
                "--pat_dir", args$pat_dir,
                "--output_dir", out_dir,
                "--tmp_dir", args$tmp_dir,
                "--target_depth", args$target_depth,
                "--threads", args$threads,
                "--concentrations", shQuote(conc_json),
                "--prefix", paste0(prefix, "_", i)
            ))
        }
    }
    
    # Process training and evaluation sets
    process_batch(train_concentrations, "train", train_dir)
    process_batch(eval_concentrations, "eval", eval_dir)
}

if(sys.nframe() == 0) {
    main()
}