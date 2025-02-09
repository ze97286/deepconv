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
                help="JSON file or JSON string containing cell type concentrations [REQUIRED]"),
    make_option(c("--prefix"), type="character", default="mix",
            help="Prefix for output files [default %default]")
)

# Helper functions
make_target_table <- function(target_fraction, dilutions, pat_dir=".", suffix=".pat.gz") {
    target_fraction.dt <- as.data.table(target_fraction, keep.rownames = TRUE)
    setnames(target_fraction.dt, c("celltype", "fraction"))
    setkey(target_fraction.dt, celltype)
    
    out.dt <- lapply(dilutions, function(dilution) {
        total_dil <- sum(dilution)
        dilution.dt <- as.data.table(dilution, keep.rownames = TRUE)
        setnames(dilution.dt, c("celltype", "fraction"))
        return(dilution.dt[target_fraction.dt, on="celltype"][, .(
            celltype, 
            filename=paste0(pat_dir,'/',celltype,suffix), 
            fraction=ifelse(is.na(fraction), 
                          i.fraction-total_dil/length(unique(celltype[is.na(fraction)])), 
                          fraction)
        )])
    })
    
    return(rbindlist(out.dt, idcol="dilution"))
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
                               threads=1, skew=0, tmp_dir=NULL, overwrite=FALSE, prefix="mix") {
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
               out_file <- paste0(target_dir,'/', prefix, '_',r,'.pat.gz')
               if (overwrite || !file.exists(out_file)) {
                   sub.dt <- targets[list(dilution=dil, celltype=ct)]
                   fraction <- sub.dt$fraction
                   filename <- sub.dt$filename
                   
                   skew_param <- ifelse(is.numeric(skew) && skew != 0, 
                                      paste0("-n ", skew), "")
                   
                   # Generate sampled pat file
                   tmp_file <- sprintf("%s/%s_%d_%s.pat.gz", tmp_dir, prefix, r, ct)
                   cmd <- sprintf('"/users/zetzioni/sharedscratch/pattools sample -s %.8f %s %s | bgzip -c > %s"', 
                                fraction, skew_param, filename, tmp_file)
                   result <- system2("sh", c("-c", cmd))
                   if (result != 0) {
                       stop(sprintf("Failed to sample reads for %s", ct))
                   }
               }
           }
           
           # merge pat files
           if (!file.exists(out_file) || overwrite) {
               system2("sh", c("-c", paste0(
                   '"zcat ', tmp_dir, '/', prefix, '_',r,'_*.pat.gz | ',
                   'sort -k1,1V -k2,2n -k3,3 | ',
                   'perl -n /users/zetzioni/sharedscratch/atlas/deduplicate_pat.pl | ',
                   'bgzip -c > ', out_file,
                   '; tabix -s 1 -b 2 -e 2 -C ', out_file, '"'
               )))
           }
           
            # Calculate true concentrations using proper read counts
            pat_files <- list.files(tmp_dir, pattern=paste0(prefix, "_", r, "_.*\\.pat\\.gz$"), full.names=TRUE)

            # Create a named vector to store counts, using the cell types from targets
            cell_types <- unique(targets$celltype)
            counts <- numeric(length(cell_types))
            names(counts) <- cell_types

            # Fill in the counts, matching files to cell types
            for(ct in cell_types) {
                f <- grep(ct, pat_files, value=TRUE)
                if(length(f) == 1) {
                    counts[ct] <- tryCatch({
                        con <- gzfile(f, open = "r")
                        lines <- readLines(con, warn = FALSE)
                        close(con)
                        
                        data <- strsplit(lines, "\t")
                        sum(sapply(data, function(row) {
                            if (length(row) >= 4 && grepl("^[0-9]+$", row[[4]])) {
                                as.numeric(row[[4]])
                            } else {
                                0
                            }
                        }), na.rm = TRUE)
                    }, error = function(e) {
                        0
                    })
                }
            }

            total_counts <- sum(counts)
            concentrations <- counts / total_counts

            # Create data frame and write CSV ensuring column names are preserved correctly
            result <- data.frame(matrix(concentrations, nrow=1,
                                    dimnames=list(NULL, names(concentrations))))
            # Use check.names=FALSE in write.table to preserve exact column names
            write.table(result, 
                    file.path(target_dir, paste0(prefix, "_", r, "_true_concentrations.csv")), 
                    row.names=FALSE, sep=",", col.names=TRUE, quote=FALSE)
           
           # Cleanup temp files
           file.remove(list.files(tmp_dir, 
                                pattern=paste0(prefix, "_", r, "_.*\\.pat\\.gz$"), 
                                full.names=TRUE))
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
    
    cat("Input concentrations string:\n")
    print(args$concentrations)
    print(class(args$concentrations))
    print(nchar(args$concentrations))
    writeLines(args$concentrations, "debug_json_input.txt")

    concentrations <- tryCatch({
        jsonlite::validate(args$concentrations)  # First validate the JSON
        cat("JSON validation passed\n")
        fromJSON(args$concentrations)
    }, error = function(e) {
        # If that fails, try reading as a file
        if (file.exists(args$concentrations)) {
            fromJSON(args$concentrations)
        } else {
            cat("JSON validation error:", e$message, "\n")
            stop("Input is neither valid JSON nor an existing file path")
        }
    })
    
    # Create table with desired concentrations
    conc_table <- data.table(
        celltype = names(concentrations),
        fraction = unlist(concentrations),
        filename = paste0(args$pat_dir, '/', names(concentrations), '.pat.gz')
    )
    
    # Add dilution identifier (we only have one mixture now)
    conc_table[, dilution := 1]
    
    # Get read counts from source files
    reads_by_celltype <- read_count_table(args$pat_dir)
    print("\nSource file read counts:")
    print(reads_by_celltype)
    
    # Calculate target fragments based on desired depth
    conc_table[, target_fragments := ceiling(args$target_depth * fraction)]
    print("Concentration table with target fragments:")
    print(conc_table[, .(celltype, fraction, target_fragments)])

    setkey(conc_table, celltype)
    setkey(reads_by_celltype, sample)
    
    # Calculate sampling fractions
    target_dilutions <- conc_table[reads_by_celltype, nomatch=NULL][
        , list(celltype, dilution, filename, fraction=target_fragments/fragments)
    ]
    print("\nFinal sampling fractions:")
    print(target_dilutions)

    
    # Generate mixtures
    generate_mix_from_pat(
        target_dilutions, 
        args$output_dir,
        repeats = args$repeats,
        target_depth = args$target_depth,
        threads = args$threads,
        skew = args$skew,
        tmp_dir = args$tmp_dir,
        overwrite = args$overwrite,
        prefix = args$prefix
    )
}

if(sys.nframe() == 0) {
    main()
}