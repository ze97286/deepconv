#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(data.table)
    library(optparse)
    library(doParallel)
    library(jsonlite)
})

# Parse command line arguments
option_list <- list(
    make_option(c("-p", "--pat_dir"), type="character",
                help="Directory containing input pat files [REQUIRED]"),
    make_option(c("-o", "--output_dir"), type="character",
                help="Output directory for mixed samples [REQUIRED]"),
    make_option(c("--tmp_dir"), type="character",
                help="Temporary directory for intermediate files [default: output_dir/tmp]"),
    make_option(c("-d", "--target_depth"), type="integer", default=22000,
                help="Target depth for generated mixtures [default %default]"),
    make_option(c("-r", "--repeats"), type="integer", default=100,
                help="Number of replicates to generate [default %default]"),
    make_option(c("-t", "--threads"), type="integer", default=1,
                help="Number of threads to use [default %default]"),
    make_option(c("-s", "--skew"), type="numeric", default=0,
                help="Skew parameter for sampling [default %default]"),
    make_option(c("--overwrite"), action="store_true", default=FALSE,
                help="Overwrite existing files [default %default]"),
    make_option(c("-c", "--concentrations"), type="character",
                help="JSON file or JSON string containing cell type concentrations [REQUIRED]")
)

# Helper function to read and validate concentrations
read_concentrations <- function(json_input) {
    # Try to parse as direct JSON string first
    concentrations <- tryCatch({
        fromJSON(json_input)
    }, error = function(e) {
        # If that fails, try reading as a file
        if (file.exists(json_input)) {
            fromJSON(json_input)
        } else {
            stop("Input is neither valid JSON nor an existing file path")
        }
    })
    
    # Validate that concentrations sum to 1 (with small tolerance for floating point)
    total <- sum(unlist(concentrations))
    tolerance <- 1e-10
    if (abs(total - 1) > tolerance) {
        stop(sprintf("Concentrations must sum to 1. Current sum is %f", total))
    }
    
    # Convert to data.table
    conc.dt <- data.table(
        celltype = names(concentrations),
        concentration = as.numeric(unlist(concentrations))
    )
    
    return(conc.dt)
}

# Modified make_target_table function
make_target_table <- function(concentrations.dt, pat_dir=".", suffix=".pat.gz") {
    # Normalize concentrations to fractions
    concentrations.dt[, fraction := concentration / sum(concentration)]
    setkey(concentrations.dt, celltype)
    
    # Create output data.table
    out.dt <- concentrations.dt[, .(
        celltype,
        filename = paste0(pat_dir, '/', celltype, suffix),
        fraction
    )]
    
    # Add dilution identifier
    out.dt[, dilution := 1]
    
    return(out.dt)
}

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

generate_mix_from_pat <- function(targets, target_dir, repeats=1, target_depth=22000, 
                                threads=1, skew=0, tmp_dir=NULL, overwrite=FALSE) {
    if(is.null(tmp_dir)) {
        tmp_dir <- paste0(target_dir,"/tmp")
    }
    
    registerDoParallel(cores=threads)
    
    dir.create(target_dir, showWarnings = FALSE, recursive = TRUE, mode = "0755")
    dir.create(tmp_dir, showWarnings = FALSE, recursive = TRUE, mode = "0755")
    
    setkey(targets, dilution, celltype)
    
    for (dil in unique(targets$dilution)) {
        foreach(r=1:repeats) %dopar% {
            for (ct in unique(targets$celltype)) {
                out_file <- paste0(target_dir,'/mix_',r,'.pat.gz')
                if (overwrite || !file.exists(out_file)) {
                    sub.dt <- targets[list(dilution=dil, celltype=ct)]
                    fraction <- sub.dt$fraction
                    filename <- sub.dt$filename
                    
                    skew_param <- ifelse(is.numeric(skew) && skew != 0, 
                                       paste0("-n ", skew), "")
                    
                    # Sample and count reads
                    tmp_file <- sprintf("%s/mix_%d_%s.pat.gz", tmp_dir, r, ct)
                    system2("sh", c("-c", sprintf(
                        '"/users/zetzioni/sharedscratch/pattools sample -s %.8f %s %s | tee >(wc -l > %s.count) | bgzip -c > %s"', 
                        fraction, skew_param, filename, tmp_file, tmp_file
                    )))
                    
                    # Read the count
                    count <- as.numeric(readLines(sprintf("%s.count", tmp_file))[1])
                    file.remove(sprintf("%s.count", tmp_file))
                    
                    # Store count in a tracking file
                    write.table(
                        data.frame(celltype=ct, count=count),
                        file=sprintf("%s/mix_%d_counts.txt", tmp_dir, r),
                        append=TRUE,
                        row.names=FALSE,
                        col.names=!file.exists(sprintf("%s/mix_%d_counts.txt", tmp_dir, r))
                    )
                }
            }
            
            # merge pat files
            if (!file.exists(out_file) || overwrite) {
                system2("sh", c("-c", paste0(
                    '"zcat ', tmp_dir, '/mix_',r,'_*.pat.gz | ',
                    'sort -k1,1V -k2,2n -k3,3 | ',
                    'perl -n /users/zetzioni/sharedscratch/atlas/deduplicate_pat.pl | ',
                    'bgzip -c > ', out_file,
                    '; tabix -s 1 -b 2 -e 2 -C ', out_file, '"'
                )))
            }
            
            # Read the counts and calculate true concentrations
            counts <- fread(sprintf("%s/mix_%d_counts.txt", tmp_dir, r))
            total_reads <- sum(counts$count)
            counts[, true_concentration := count/total_reads]
            
            # Reshape the data to have cell types as columns
            wide_counts <- dcast(counts, . ~ celltype, value.var = "true_concentration")
            wide_counts[, `:=`(. = NULL, sample = sprintf("mix_%d", r))]
            
            # Save the true concentrations
            fwrite(wide_counts, sprintf("%s/mix_%d_true_concentrations.csv", args$output_dir, r))
            
            # Cleanup temp files
            file.remove(list.files(tmp_dir, 
                                 paste0("mix_",r,"_.*\\.(pat\\.gz|count)$"), 
                                 full.names = TRUE))
            file.remove(sprintf("%s/mix_%d_counts.txt", tmp_dir, r))
        }
    }
}

main <- function() {
    # Parse arguments
    parser <- OptionParser(option_list=option_list)
    args <- parse_args(parser)
    
    # Check required arguments
    if (is.null(args$pat_dir) || is.null(args$output_dir) || is.null(args$concentrations)) {
        print_help(parser)
        stop("Missing required arguments")
    }
    
    if(is.null(args$tmp_dir)) {
        args$tmp_dir <- paste0(args$output_dir, "/tmp")
    }
    
    # Read concentrations from JSON file
    concentrations <- read_concentrations(args$concentrations)
    
    # Generate target table
    target_table <- make_target_table(concentrations, pat_dir = args$pat_dir)
    reads_by_celltype <- read_count_table(args$pat_dir)
    target_table[, target_fragments := ceiling(args$target_depth * fraction)]
    
    setkey(target_table, celltype)
    setkey(reads_by_celltype, sample)
    
    target_dilutions <- target_table[reads_by_celltype, nomatch=NULL][
        , list(celltype, dilution, filename, fraction=target_fragments/fragments)
    ]
    
    # Generate mixtures
    generate_mix_from_pat(
        target_dilutions, 
        args$output_dir,
        repeats = args$repeats,
        target_depth = args$target_depth,
        threads = args$threads,
        skew = args$skew,
        tmp_dir = args$tmp_dir,
        overwrite = args$overwrite
    )
}

if(sys.nframe() == 0) {
    main()
}