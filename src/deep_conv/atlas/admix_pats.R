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
    make_option(c("-t", "--threads"), type="integer", default=1,
                help="Number of threads to use [default %default]"),
    make_option(c("-c", "--concentrations"), type="character",
                help="JSON file containing cell type concentrations and repeats [REQUIRED]"),
    make_option(c("--min_depth"), type="integer", default=40000,
                help="Minimum total coverage [default %default]"),
    make_option(c("--max_depth"), type="integer", default=100000,
                help="Maximum total coverage [default %default]"),
    make_option(c("--overwrite"), action="store_true", default=FALSE,
                help="Overwrite existing files [default %default]"),
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

read_count_table <- function(patdir, cell_type_order) {
    # First check the directory exists
    if (!dir.exists(patdir)) {
        stop(sprintf("Directory does not exist: %s", patdir))
    }

    # Create the full file paths in the specified order
    files <- paste0(patdir, "/", cell_type_order, ".pat.gz")
    
    # Verify all files exist
    missing_files <- files[!file.exists(files)]
    if (length(missing_files) > 0) {
        stop(sprintf("Missing .pat.gz files: %s", 
                    paste(basename(missing_files), collapse=", ")))
    }
    
    # Assign names to files based on cell_type_order
    names(files) <- cell_type_order
    
    # Initialize list to store fragment counts
    all_frags_list <- lapply(files, function(file_name) {
        # Check file size
        file_info <- file.info(file_name)
        if (file_info$size == 0) {
            cat(sprintf("Skipping empty file: %s\n", file_name))
            return(NULL)
        }
        
        # Construct and run the command
        cmd <- sprintf("zcat %s", shQuote(file_name))
        
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
        return(total_counts)
    })
    
    # Remove NULL entries
    valid_indices <- !sapply(all_frags_list, is.null)
    all_frags_list <- all_frags_list[valid_indices]
    
    if (length(all_frags_list) == 0) {
        stop("No valid .pat.gz files were processed")
    }
    
    # Create data table in the specified order
    all_frags <- data.table(
        sample = names(all_frags_list),
        fragments = unlist(all_frags_list)
    )[match(cell_type_order, sample)]  # Ensure order matches cell_type_order
    
    return(all_frags)
}

generate_mix_from_pat <- function(targets, target_dir, repeats=1, min_depth=20000, max_depth=80000,
                                  threads=1, tmp_dir=NULL, overwrite=FALSE, prefix="mix", mixture_id=1) {
    if (is.null(tmp_dir)) {
        tmp_dir <- paste0(target_dir, "/tmp")
    }
    
    print(targets)
   
    # Create main directories
    dir.create(target_dir, showWarnings = FALSE, recursive = TRUE, mode = "0755")
    dir.create(tmp_dir, showWarnings = FALSE, recursive = TRUE, mode = "0755")
    
    setkey(targets, dilution, celltype)
    
    # Generate depths for all repeats upfront
    all_depths <- round(runif(repeats, min_depth, max_depth))
    
    # Process all repeats (this loop can be converted to %dopar% for parallel processing)
    foreach(r = 1:repeats, 
        .export = c("calculate_true_concentrations", "generate_mix_from_pat"),
        .packages = c("data.table")) %dopar% { 
        current_depth <- all_depths[r]
        mix_prefix <- sprintf("%s_mix%d", prefix, mixture_id)
        
        # Create a worker-specific temporary directory for this repeat
        worker_tmp_dir <- file.path(tmp_dir, paste0("worker_", r))
        dir.create(worker_tmp_dir, showWarnings = FALSE, recursive = TRUE, mode = "0755")
        
        for (ct in unique(targets$celltype)) {
            out_file <- paste0(target_dir, '/', mix_prefix, '_', r, '.pat.gz')
            if (overwrite || !file.exists(out_file)) {
                sub.dt <- targets[list(dilution = 1, celltype = ct)]
                fraction <- sub.dt$fraction
                filename <- sub.dt$filename
                
                # Adjust fraction based on current_depth vs target_depth
                adjusted_fraction <- fraction * (current_depth / sub.dt$target_depth)
                
                # Generate sampled pat file in the worker-specific temporary directory
                tmp_file <- sprintf("%s/%s_%d_%s.pat.gz", worker_tmp_dir, mix_prefix, r, ct)
                cmd <- sprintf('"/users/zetzioni/sharedscratch/pattools sample -s %.8f %s | bgzip -c > %s"', 
                            adjusted_fraction, filename, tmp_file)
                result <- system2("sh", c("-c", cmd))
                if (result != 0) {
                    stop(sprintf("Failed to sample reads for %s in mixture %d, repeat %d", 
                                 ct, mixture_id, r))
                }
                
                # Verify the tmp file was created and has content
                if (file.exists(tmp_file)) {
                    file_info <- file.info(tmp_file)
                } else {
                    cat(sprintf("WARNING: tmp file not created: %s\n", tmp_file))
                }
            }
        }
        
        # Merge pat files from the worker-specific temporary directory
        out_file <- paste0(target_dir, '/', mix_prefix, '_', r, '.pat.gz')
        if (!file.exists(out_file) || overwrite) {
            merge_cmd <- paste0(
                '"zcat ', worker_tmp_dir, '/', mix_prefix, '_',r,'_*.pat.gz | ',
                'sort -k1,1V -k2,2n -k3,3 | ',
                'perl -n /users/zetzioni/sharedscratch/atlas/deduplicate_pat.pl | ',
                'bgzip -c > ', out_file,
                '; tabix -s 1 -b 2 -e 2 -C ', out_file, '"'
            )
            system2("sh", c("-c", merge_cmd))
            
            if (file.exists(out_file)) {
                file_info <- file.info(out_file)
            } else {
                cat(sprintf("WARNING: merged file not created: %s\n", out_file))
            }
        }
        
        # Calculate and save true concentrations (using the worker-specific temporary directory)
        calculate_true_concentrations(worker_tmp_dir, target_dir, mix_prefix, r, targets$celltype)
        
        # Cleanup worker-specific temporary files and directory
        tmp_files <- list.files(worker_tmp_dir, 
                                pattern = paste0(mix_prefix, "_", r, "_.*\\.pat\\.gz$"), 
                                full.names = TRUE)
        file.remove(tmp_files)
        unlink(worker_tmp_dir, recursive = TRUE)
        
        cat(sprintf("\nfinished processing repeat: %s\n", r))
    }
}

calculate_true_concentrations <- function(tmp_dir, target_dir, mix_prefix, repeat_num, cell_type_order) {
    # Create a named vector to store counts in the specified order
    counts <- numeric(length(cell_type_order))
    names(counts) <- cell_type_order

    # Calculate counts for each cell type
    for(ct in cell_type_order) {
        # Construct exact filename instead of searching
        tmp_file <- file.path(tmp_dir, paste0(mix_prefix, "_", repeat_num, "_", ct, ".pat.gz"))
        
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
        } else {
            warning(sprintf("File not found: %s", tmp_file))
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
                         paste0(mix_prefix, "_", repeat_num, "_true_concentrations.csv"))
    
    write.table(result, 
                out_file, 
                row.names=FALSE, 
                sep=",", 
                col.names=TRUE, 
                quote=FALSE)
                
    return(concentrations)
}


process_concentration_set <- function(reads_by_celltype, concentration_set, args, mixture_id, cell_type_order) {
    mixture <- concentration_set$mixture
    repeats <- concentration_set$repeats
    # Create concentration table
    conc_table <- data.table(
        celltype = names(mixture),
        fraction = as.numeric(mixture),
        filename = paste0(args$pat_dir, '/', names(mixture), '.pat.gz')
    )
    median_depth <- (args$min_depth + args$max_depth) / 2
    conc_table[, `:=`(
        dilution = 1,
        target_depth = median_depth,
        target_fragments = ceiling(median_depth * fraction)
    )]
    setnames(reads_by_celltype, c("sample", "fragments"))
    
    # Show intermediate calculation steps
    merged_table <- merge(
        conc_table,
        reads_by_celltype,
        by.x = "celltype",
        by.y = "sample"
    )
    target_dilutions <- merged_table[, list(
        celltype = celltype,
        dilution = dilution,
        filename = filename,
        target_depth = target_depth,
        fraction = target_fragments/fragments
    )]
    oac_row <- merged_table[celltype == "OAC"]
    # Rest of the function remains the same...
    generate_mix_from_pat(
        target_dilutions, 
        args$output_dir,
        repeats = repeats,
        min_depth = args$min_depth,
        max_depth = args$max_depth,
        threads = args$threads,
        tmp_dir = args$tmp_dir,
        overwrite = args$overwrite,
        prefix = args$prefix,
        mixture_id = mixture_id
    )
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
    
    # Read and validate JSON
    json_data <- tryCatch({
        if (file.exists(args$concentrations)) {
            fromJSON(args$concentrations)
        } else {
            fromJSON(args$concentrations)
        }
    }, error = function(e) {
        stop("Failed to parse concentrations JSON: ", e$message)
    })
    
    # Validate JSON structure
    if (is.null(json_data$cell_type_order)) {
        stop("JSON must contain 'cell_type_order' field")
    }
    
    # Extract cell type order
    cell_type_order <- json_data$cell_type_order
    
    # Get read counts from source files (now with ordered cell types)
    reads_by_celltype <- read_count_table(args$pat_dir, cell_type_order)
    
    # Get the mixtures data frame and repeats
    mixtures_df <- json_data$concentrations$mixture
    repeats <- json_data$concentrations$repeats[1]
    registerDoParallel(cores=args$threads)
    
    for (i in 1:nrow(mixtures_df)) {
        mixture <- as.numeric(mixtures_df[i,])
        names(mixture) <- names(mixtures_df)
        process_concentration_set(reads_by_celltype, 
                                list(mixture=mixture, repeats=repeats), 
                                args, i, cell_type_order)
    }
    
    stopImplicitCluster()
}


if(sys.nframe() == 0) {
    main()
}