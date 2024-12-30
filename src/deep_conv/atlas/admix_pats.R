#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(data.table)
    library(optparse)
    library(doParallel)
})

# R_LIBS_USER=~/R/library Rscript /users/zetzioni/sharedscratch/deepconv/src/deep_conv/atlas/admix_pats.R \
#  --pat_dir /users/zetzioni/sharedscratch/pat/dmr_by_read.blood+gi+tum.250/Song/celltypes \
#   --output_dir /users/zetzioni/sharedscratch/pat/dmr_by_read.blood+gi+tum.250/Song/mixed/CD4 \
#   --tmp_dir /path/to/tmp \
#   --target_depth 10 \
#   --repeats 100 \
#   --threads 32

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
                help="Overwrite existing files [default %default]")
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
    files <- list.files(patdir, pattern=".*\\.pat\\.gz$", full.names=TRUE)
    if (length(files) == 0) {
        stop(sprintf("No .pat.gz files found in: %s", patdir))
    }
    
    cat("Found files:\n")
    print(files)
    
    names(files) <- gsub(".pat.gz", "", basename(files), fixed=TRUE)
    
    all_frags <- lapply(files, function(file_name) {
        cat(sprintf("\nProcessing file: %s\n", file_name))
        
        # Check file exists and has size
        if (!file.exists(file_name)) {
            stop(sprintf("File does not exist: %s", file_name))
        }
        file_info <- file.info(file_name)
        if (file_info$size == 0) {
            stop(sprintf("File is empty: %s", file_name))
        }
        
        cmd <- sprintf("zcat %s", file_name)
        cat(sprintf("Running command: %s\n", cmd))
        
        data <- fread(cmd=cmd, stringsAsFactors=TRUE, header=FALSE, select=4)
        if (is.null(data) || ncol(data) == 0) {
            stop(sprintf("Failed to read data from: %s", file_name))
        }
        
        setnames(data, c("counts"))
        return(sum(data$counts))
    })
    
    all_frags <- as.data.table(all_frags)
    all_frags <- data.table(sample=names(all_frags), fragments=t(all_frags)[,1])
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
                out_file <- paste0(target_dir,'/d',dil,'_',r,'.pat.gz')
                if (overwrite || !file.exists(out_file)) {
                    sub.dt <- targets[list(dilution=dil, celltype=ct)]
                    fraction <- sub.dt$fraction
                    filename <- sub.dt$filename
                    
                    skew_param <- ifelse(is.numeric(skew) && skew != 0, 
                                       paste0("-n ", skew), "")
                    
                    system2("sh", c("-c", sprintf(
                        '"pattools sample -s %.8f %s %s | bgzip -c > %s/d%s_%d_%s.pat.gz"', 
                        fraction, skew_param, filename, tmp_dir, dil, r, ct
                    )))
                }
            }
            
            # merge pat files
            if (!file.exists(out_file) || overwrite) {
                system2("sh", c("-c", paste0(
                    '"zcat ', tmp_dir, '/d',dil, '_',r,'_*.pat.gz | ',
                    'sort -k1,1V -k2,2n -k3,3 | ',
                    'perl -n deduplicate_pat.pl | ',
                    'bgzip -c > ', out_file,
                    '; tabix -s 1 -b 2 -e 2 -C ', out_file, '"'
                )))
            }
            
            # Cleanup temp files
            file.remove(list.files(tmp_dir, 
                                 paste0("d",dil,"_",r,"_.*\\.pat\\.gz$"), 
                                 full.names = TRUE))
        }
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
    
    # Define mixture composition
    targetRatio <- c(
        `CD34-megakaryocytes`=3, 
        Monocytes=2, 
        Neutrophils=3, 
        `CD34-erythroblasts`=0.5, 
        `CD4-T-cells`=NA
    )
    target_fraction <- targetRatio/sum(targetRatio, na.rm = TRUE)
    
    # Define dilutions
    dilutions <- list(
        c("CD4-T-cells"=0.00001),
        c("CD4-T-cells"=0.0001),
        c("CD4-T-cells"=0.001),
        c("CD4-T-cells"=0.005),
        c("CD4-T-cells"=0.01),
        c("CD4-T-cells"=0.05),
        c("CD4-T-cells"=0.1)
    )
    names(dilutions) <- c("1e-05", "1e-04", "0.001", "0.005", "0.01", "0.05", "0.1")
    
    # Generate dilution table
    dil_table <- make_target_table(target_fraction, dilutions, pat_dir = args$pat_dir)
    reads_by_celltype <- read_count_table(args$pat_dir)
    dil_table[, target_fragments := ceiling(args$target_depth * fraction)]
    
    setkey(dil_table, celltype)
    setkey(reads_by_celltype, sample)
    
    target_dilutions <- dil_table[reads_by_celltype, nomatch=NULL][
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