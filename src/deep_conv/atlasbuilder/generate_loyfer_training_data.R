#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(data.table)
    library(optparse)
    library(jsonlite)
    library(doParallel)
    library(tictoc)  
})

##
## 1. Command-Line Options
##
option_list <- list(
    make_option(c("-p", "--pat_dir"), type="character",
                help="Directory containing input pat files [REQUIRED]"),
    make_option(c("-o", "--output_dir"), type="character",
                help="Output directory for mixed samples [REQUIRED]"),
    make_option(c("--tmp_dir"), type="character",
                help="Temporary directory for intermediate files [default: output_dir/tmp]"),
    make_option(c("-t", "--threads"), type="integer", default=1,
                help="Number of threads to use [default %default]"),
    make_option(c("--n_train"), type="integer", default=100000,
                help="Number of training samples to generate [default %default]"),
    make_option(c("--n_eval"), type="integer", default=20000,
                help="Number of evaluation samples to generate [default %default]"),
    make_option(c("--reps_per_combo"), type="integer", default=10,
                help="Number of repetitions per concentration combo [default %default]"),
    make_option(c("--min_depth"), type="integer", default=40000,
                help="Minimum total coverage for each synthetic mixture [default %default]"),
    make_option(c("--max_depth"), type="integer", default=100000,
                help="Maximum total coverage for each synthetic mixture [default %default]"),
    make_option(c("--zero_fraction"), type="double", default=0.3,
                help="Fraction of samples forcing at least one cell type to 0% [default %default]"),
    make_option(c("--coverage_fuzz"), type="double", default=0.0,
                help="If > 0, random partial coverage dropping after merge (range 0-1) [default %default]"),
    make_option(c("--tier3"), action="store_true", default=FALSE,
                help="Use tier3 concentration generation method [default %default]")                
)

##
## 2. Helper Functions
##

# (a) Dirichlet sampling (not strictly used in this script, but can be handy)
rdirichlet <- function(n, alpha) {
    gamma_samples <- matrix(0, nrow=n, ncol=length(alpha))
    for(i in 1:length(alpha)) {
        gamma_samples[,i] <- rgamma(n, shape=alpha[i], rate=1)
    }
    gamma_samples / rowSums(gamma_samples)
}

# (b) Function to read total fragment counts from each .pat.gz file
read_count_table <- function(patdir) {
    if (!dir.exists(patdir)) {
        stop(sprintf("Directory does not exist: %s", patdir))
    }
    files <- list.files(patdir, pattern = ".*\\.pat\\.gz$", full.names = TRUE)
    if (length(files) == 0) {
        stop(sprintf("No .pat.gz files found in: %s", patdir))
    }
    cat("Found .pat.gz files:\n")
    print(files)
    names(files) <- gsub(".pat.gz", "", basename(files), fixed = TRUE)
    
    all_frags_list <- lapply(files, function(file_name) {
        cat(sprintf("\nProcessing file: %s\n", file_name))
        if (!file.exists(file_name)) {
            stop(sprintf("File does not exist: %s", file_name))
        }
        file_info <- file.info(file_name)
        if (file_info$size == 0) {
            cat(sprintf("Skipping empty file: %s\n", file_name))
            return(NULL)
        }
        cmd <- sprintf("zcat %s", shQuote(file_name))
        cat(sprintf("Running command: %s\n", cmd))
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

##
## 3. Modified "generate_stratified_concentrations" with zeros, near-pure, random depth
##
generate_stratified_concentrations <- function(n_samples,
                                             cell_types,
                                             mode_weights = c(A=0.2, B=0.3, C=0.2, D=0.3),
                                             zero_fraction = 0.3,
                                             min_depth = 20000,
                                             max_depth = 80000) {
  
  stopifnot(abs(sum(mode_weights) - 1) < 1e-6)
  
  # Maximum allowed concentration per cell type
  max_concentrations <- c(
    "Granulocytes"       = 0.45,
    "Monocytes"          = 0.35,
    "T-cells"            = 0.30,  # Combined max for T-cells
    "CD34-erythroblasts" = 0.15,
    "CD34-megakaryocytes"= 0.35,
    "OAC"                = 0.45,
    "B-cells"            = 0.20,
    "NK-cells"           = 0.20,
    "Esophagus"          = 0.10,
    "Colon"              = 0.10,
    "Small-intestine"    = 0.10,
    "Gastric"            = 0.10
  )
  
  # Cell type correlations for Mode B
  blood_correlations <- list(
    gran_mono = c("Granulocytes", "Monocytes"),
    adaptive = c("T-cells", "B-cells"),
    cytotoxic = c("NK-cells", "T-cells")
  )

  # Correlation strengths
  correlation_strengths <- list(
    gran_mono = 0.2,
    adaptive = 0.3,
    cytotoxic = 0.25
  )

  # Example typical means for mode B (normal blood)
  typical_blood_means <- c(
    "Granulocytes"        = 0.30,
    "Monocytes"           = 0.25,
    "T-cells"             = 0.10,
    "CD34-erythroblasts"  = 0.05,
    "CD34-megakaryocytes" = 0.30
  )

  # Add placeholders for missing types
  for (ct in setdiff(cell_types, names(typical_blood_means))) {
    typical_blood_means[ct] <- 0.01
  }
  typical_blood_means <- typical_blood_means / sum(typical_blood_means)
  
  # OAC concentration ranges
  oac_ranges <- list(
    ultra_low = c(0.0005, 0.005),   # 0.05% to 0.5%
    low = c(0.005, 0.05),           # 0.5% to 5%
    medium = c(0.05, 0.15)          # 5% to 15%
  )

  # T-cell concentration ranges
  tcell_ranges <- list(
    ultra_low = c(0.00005, 0.001),  # 0.005% to 0.1%
    very_low = c(0.0005, 0.005),    # 0.05% to 0.5%
    low = c(0.003, 0.01)          # 0.3% to 1%
  )
  
  # Track zero counts for each cell type to ensure balanced training data
  zero_counts <- setNames(rep(0, length(cell_types)), cell_types)
  target_zeros_per_type <- ceiling(n_samples * 0.15)  # Aim for ~15% of samples to have each type zeroed
  
  concentrations <- data.table()
  for (i in seq_len(n_samples)) {
    mode_pick <- sample(names(mode_weights), size=1, prob=mode_weights)

    if (mode_pick == "A") {
      # Mode A: Broad Dirichlet with max constraints
      alpha <- rep(0.8, length(cell_types))  # Slightly spikier than uniform
      repeat {
        d <- rdirichlet(1, alpha)
        conc <- as.numeric(d[1, ])
        names(conc) <- cell_types
        if (all(conc <= max_concentrations[cell_types])) break
      }
      
    } else if (mode_pick == "B") {
        # Mode B: "Typical" distribution with correlations
        repeat {
            # Base values with smaller CV for more stable correlations
            base <- sapply(cell_types, function(ct) {
                m <- typical_blood_means[ct]
                val <- rnorm(1, mean=m, sd=m * 0.25)
                pmax(val, 0) 
            })

            for (corr_group in names(blood_correlations)) {
                group_cells <- blood_correlations[[corr_group]]
                strength <- correlation_strengths[[corr_group]]
                
                if (all(group_cells %in% names(base))) {
                    # Generate more strongly correlated noise
                    n_cells <- length(group_cells)
                    sigma <- matrix(strength, n_cells, n_cells)
                    diag(sigma) <- 1
                    noise <- MASS::mvrnorm(1, rep(0, n_cells), sigma)
                    # Modified noise application
                    group_total <- sum(base[group_cells])
                    noise_scaled <- noise * 0.15
                    # More direct correlation application
                    adjustments <- (1 + noise_scaled)  # Using additive rather than exponential
                    adjustments <- pmax(adjustments, 0)  # Ensure non-negative
                    adjustments <- adjustments / sum(adjustments)
                    base[group_cells] <- group_total * adjustments
                }
            }
            
            if (sum(base) == 0) base[] <- 1.0
            conc <- base / sum(base)
            if (all(conc <= max_concentrations[cell_types])) break
        }
      
    } else if (mode_pick == "C") {
      # Mode C: OAC focused with sub-ranges
      alpha <- rep(1, length(cell_types))
      repeat {
        d <- rdirichlet(1, alpha)
        conc <- as.numeric(d[1, ])
        names(conc) <- cell_types
        # Pick OAC range
        range_pick <- sample(names(oac_ranges), 1, prob=c(0.3, 0.4, 0.3))
        range <- oac_ranges[[range_pick]]
        tumor_target <- runif(1, range[1], range[2])
        sum_others <- sum(conc[names(conc) != "OAC"])
        if (sum_others > 0) {
          scaling <- (1 - tumor_target) / sum_others
          conc[names(conc) != "OAC"] <- conc[names(conc) != "OAC"] * scaling
          conc["OAC"] <- tumor_target
        }
        if (all(conc <= max_concentrations[cell_types])) break
      }
      
    } else if (mode_pick == "D") {
      # Mode D: T-cell focused with sub-ranges
      alpha <- rep(1, length(cell_types))
      repeat {
        d <- rdirichlet(1, alpha)
        conc <- as.numeric(d[1, ])
        names(conc) <- cell_types
        # Pick concentration sub-range
        range_pick <- sample(names(tcell_ranges), 1, prob=c(0.3, 0.4, 0.3))
        range <- tcell_ranges[[range_pick]]
        # Set T-cell concentration
        t_cell_target <- runif(1, range[1], range[2])
        sum_others <- sum(conc[names(conc) != "T-cells"])
        if (sum_others > 0) {
          scaling <- (1 - t_cell_target) / sum_others
          conc[names(conc) != "T-cells"] <- conc[names(conc) != "T-cells"] * scaling
          conc["T-cells"] <- t_cell_target
        }
        if (all(conc <= max_concentrations[cell_types])) break
      }
    }

    # NEW: Enhanced zero handling - prioritise cell types that haven't been zeroed enough
    if (runif(1) < zero_fraction || any(zero_counts < target_zeros_per_type)) { 
        # Split cell types into groups
        immune_cells <- c("T-cells", "B-cells", "NK-cells")
        myeloid_cells <- c("Granulocytes", "Monocytes")
        tissue_cells <- c("Colon", "Small-intestine", "Esophagus", "Gastric")
        special_cells <- c("CD34-erythroblasts", "CD34-megakaryocytes", "OAC")
        
        # NEW: Prioritize under-represented zeros
        under_represented <- names(which(zero_counts < target_zeros_per_type))
        
        # If we have under-represented cell types, prioritise those
        if (length(under_represented) > 0) {
            # Choose 1-3 types to zero out
            n_zero <- min(length(under_represented), sample(1:3, 1))
            zero_cts <- sample(under_represented, n_zero)
        } else {
            # Traditional approach with 2-4 zeros if we've met all targets
            n_zero <- sample(2:4, 1)
            # Ensure at least one zero from tissue cells (they're often absent)
            zero_cts <- sample(tissue_cells, 1)  
            # Add remaining zeros, avoiding zeroing all cells of any group
            remaining_zeros <- n_zero - 1
            if (remaining_zeros > 0) {
                # Create pool of possible cells to zero
                pool <- setdiff(cell_types, zero_cts)
                # Add more zeros
                additional_zeros <- sample(pool, remaining_zeros)
                zero_cts <- c(zero_cts, additional_zeros)
            }
        }

        # For each zeroed cell type, either true zero or very small value
        for (ct in zero_cts) {
            conc[ct] <- if(runif(1) < 0.7) 0 else runif(1, 1e-6, 1e-4)
            # Track which cell types have been zeroed
            if (conc[ct] == 0) zero_counts[ct] <- zero_counts[ct] + 1
        }
        s2 <- sum(conc)
        if (s2 > 0) conc <- conc / s2  
    }

    concentrations <- rbindlist(list(
      concentrations,
      as.data.table(as.list(conc))
    ))
  }

  depths <- round(runif(n_samples, min_depth, max_depth))
  concentrations[, depth := depths]
  
  # Validate distribution of concentrations
  validate_concentrations <- function(conc_table) {
    # Count samples in each important range
    n_low_tcell <- sum(conc_table$`T-cells` < 0.01)
    n_low_oac <- sum(conc_table$OAC < 0.01)
    n_zeros <- sum(apply(conc_table, 1, function(x) any(x == 0)))
    
    # Print summary
    cat(sprintf("\nConcentration Distribution Summary:"))
    cat(sprintf("\n - Low T-cell samples (<1%%): %d (%.1f%%)", 
                n_low_tcell, 100 * n_low_tcell/nrow(conc_table)))
    cat(sprintf("\n - Low OAC samples (<1%%): %d (%.1f%%)", 
                n_low_oac, 100 * n_low_oac/nrow(conc_table)))
    cat(sprintf("\n - Samples with zeros: %d (%.1f%%)", 
                n_zeros, 100 * n_zeros/nrow(conc_table)))
    
    # NEW: Print zeros per cell type
    cat("\n\nZeros per cell type:")
    for (ct in cell_types) {
        n_ct_zeros <- sum(conc_table[[ct]] == 0)
        cat(sprintf("\n - %s: %d (%.1f%%)", 
                    ct, n_ct_zeros, 100 * n_ct_zeros/nrow(conc_table)))
    }
    
    # Cell type correlations
    cor_matrix <- cor(as.matrix(conc_table[, !c("depth")]))
    cat("\n\nKey Correlations:")
    cat(sprintf("\n - NK-T-cells: %.3f", cor_matrix["NK-cells", "T-cells"]))
    cat(sprintf("\n - Granulocytes-Monocytes: %.3f", cor_matrix["Granulocytes", "Monocytes"]))
  }
  
  validate_concentrations(concentrations)
  return(concentrations)
}

generate_stratified_concentrations_tier3 <- function(n_samples, cell_types,
                                                    min_depth = 20000, max_depth = 80000) {
    
    max_concentrations <- c(
        "Granulocytes"       = 0.45,
        "Monocytes"          = 0.35,
        "T-cells"            = 0.30,  # Combined max for T-cells
        "CD34-erythroblasts" = 0.15,
        "CD34-megakaryocytes"= 0.35,
        "OAC"                = 0.45,
        "B-cells"            = 0.20,
        "NK-cells"           = 0.20,
        "Esophagus"          = 0.10,
        "Colon"              = 0.10,
        "Small-intestine"    = 0.10,
        "Gastric"            = 0.10
    )
    
    categories <- c("immunosuppression", "inflammation", "tissue_dominance", 
                   "t_cell_anomaly", "low_depth", "cancer")
    category_probs <- c(0.2, 0.2, 0.2, 0.15, 0.1, 0.15)
    
    check_max_concentrations <- function(conc, max_concentrations) {
        for (cell_type in names(max_concentrations)) {
            if (conc[cell_type] > max_concentrations[cell_type]) {
                return(FALSE)
            }
        }
        return(TRUE)
    }
    
    concentrations <- data.table()
    
    for (i in seq_len(n_samples)) {
        repeat {
            category <- sample(categories, 1, prob = category_probs)
            conc <- numeric(length(cell_types))
            names(conc) <- cell_types
            
            if (category == "immunosuppression") {
                # Ensure >80% granulocytes + monocytes
                neut_mono_total <- runif(1, 0.8, 0.95)
                neut_frac <- runif(1, 0.6, 0.8)
                conc["Granulocytes"] <- neut_mono_total * neut_frac
                conc["Monocytes"] <- neut_mono_total * (1 - neut_frac)
                conc["T-cells"] <- runif(1, 0.0001, 0.001)
                
            } else if (category == "inflammation") {
                # Ensure >90% blood cells, minimal tissue
                blood_total <- runif(1, 0.9, 0.98)
                conc["Granulocytes"] <- blood_total * runif(1, 0.6, 0.8)
                conc["Monocytes"] <- blood_total * runif(1, 0.2, 0.4)
                # Set tissue cells to very low values
                tissue_types <- c("Colon", "Small-intestine", "Esophagus", "Gastric")
                conc[tissue_types] <- runif(length(tissue_types), 0.001, 0.005)
                
            } else if (category == "tissue_dominance") {
                tissue_types <- c("Colon", "Small-intestine", "Esophagus", "Gastric")
                dominant_tissue <- sample(tissue_types, 1)
                conc[dominant_tissue] <- runif(1, 0.8, 0.9)
                # Set blood cells to very low values
                blood_cells <- c("Granulocytes", "Monocytes", "T-cells", 
                               "B-cells", "NK-cells")
                conc[blood_cells] <- runif(length(blood_cells), 0.001, 0.01)
                
            } else if (category == "t_cell_anomaly") {
                # High T-cell fraction
                conc["T-cells"] <- runif(1, 0.1, 0.2)
                # Normal granulocytes
                conc["Granulocytes"] <- runif(1, 0.3, 0.5)
                
            } else if (category == "low_depth") {
                # Generate random concentrations
                conc[] <- runif(length(cell_types), 0.01, 0.1)
                
            } else if (category == "cancer") {
                conc["OAC"] <- runif(1, 0.3, 0.5)
                # Suppressed adaptive immunity
                immune_cells <- c("T-cells", "B-cells")
                conc[immune_cells] <- runif(length(immune_cells), 0.0001, 0.005)
            }
            
            # Fill remaining cells with small values
            remaining_cells <- setdiff(cell_types, names(conc[conc > 0]))
            if (length(remaining_cells) > 0) {
                conc[remaining_cells] <- runif(length(remaining_cells), 0.001, 0.05)
            }
            
            # Normalise
            conc <- conc / sum(conc)
            
            # Check if concentrations are within limits
            if (check_max_concentrations(conc, max_concentrations)) {
                break  # Valid concentrations found
            }
        }
        
        concentrations <- rbindlist(list(concentrations, as.data.table(as.list(conc))))
    }
    
    # Handle depths, with special case for low_depth category
    depths <- ifelse(category == "low_depth",
                    round(runif(n_samples, min_depth/4, min_depth/2)),  # Lower depth range
                    round(runif(n_samples, min_depth, max_depth)))
    concentrations[, depth := depths]
    
    return(concentrations)
}

##
## 4. generate_samples() function
##    Here we read "conc_table" (with fractions), compute how many fragments to sample
##
generate_samples <- function(conc_table, reads_by_celltype, target_depth, prefix, tmp_dir, out_dir, pat_dir, rep) {
    # conc_table: data.table with columns: celltype, fraction
    # reads_by_celltype: data.table with columns: sample, fragments
    # target_depth: integer (e.g. 50k) or random from the new approach
    # rep: replicate number

    setkey(conc_table, celltype)
    setkey(reads_by_celltype, sample)

    # For each cell type, figure out how many fragments to sample
    # fraction_of_depth = target_depth * fraction
    # Then scale by 1 / fragments_of_that_cell_type (the pattools -s param).
    target_dilutions <- conc_table[reads_by_celltype, nomatch=NULL][
        , list(celltype,
               filename=file.path(pat_dir, paste0(celltype, ".pat.gz")), 
               fraction = target_depth * fraction / fragments)
    ]
    
    rep_prefix <- paste0(prefix, "_", rep)
    
    # Sample from each cell type
    for (ct in target_dilutions$celltype) {
        sub.dt <- target_dilutions[celltype == ct]
        if (sub.dt$fraction <= 0) {
            # skip
            next
        }
        cat(sprintf("Running sample command for %s with fraction %.6f\n", ct, sub.dt$fraction))
        cmd <- sprintf('"/users/zetzioni/sharedscratch/pattools sample -s %.8f %s | bgzip -c > %s/%s_%s.pat.gz"',
                       sub.dt$fraction, sub.dt$filename, tmp_dir, rep_prefix, ct)
        result <- system2("sh", c("-c", cmd))
        if (result != 0) {
            stop(sprintf("Failed to sample reads for %s in replica %d", ct, rep))
        }
    }
}


##
## 5. merge_pat_files() function with coverage_fuzz
##
merge_pat_files <- function(prefix, rep, tmp_dir, out_dir, coverage_fuzz=0.0) {
    rep_prefix <- paste0(prefix, "_", rep)
    out_file <- file.path(out_dir, paste0(rep_prefix, ".pat.gz"))
    
    # Merge
    merge_cmd <- paste0(
        '"zcat ', tmp_dir, '/', rep_prefix, '_*.pat.gz | ',
        'sort -k1,1V -k2,2n -k3,3 | ',
        'perl -n /users/zetzioni/sharedscratch/atlas/deduplicate_pat.pl | ',
        'bgzip -c > ', out_file, '; ',
        'tabix -s 1 -b 2 -e 2 -C ', out_file, '"'
    )
    system2("sh", c("-c", merge_cmd))
    
    # Optionally fuzz coverage
    if (coverage_fuzz > 0) {
        # e.g. random factor in [1-coverage_fuzz, 1+coverage_fuzz]
        factor <- runif(1, 1 - coverage_fuzz, 1 + coverage_fuzz)
        
        if (factor < 1) {
            # Drop some fraction
            drop_prob <- 1 - factor
            fuzzed_file <- file.path(out_dir, paste0(rep_prefix, "_fuzzed.pat.gz"))
            
            fuzz_cmd <- paste0(
                '"zcat ', out_file, 
                ' | awk \'BEGIN{srand()} {if(rand()>', drop_prob, ') print}\' | ',
                'bgzip -c > ', fuzzed_file, '; ',
                'tabix -s 1 -b 2 -e 2 -C ', fuzzed_file, '"'
            )
            system2("sh", c("-c", fuzz_cmd))
            
            # Overwrite the original
            file.copy(fuzzed_file, out_file, overwrite=TRUE)
            file.remove(fuzzed_file)
        } else {
            # factor > 1 => replicating lines is more complicated.
            # For simplicity, we skip replication.
            # Or just do nothing. The user can implement replication if desired.
            cat(sprintf("Coverage fuzz factor >1 (%.2f), skipping replication.\n", factor))
        }
    }
    
    # Summarise final read counts by cell type
    pat_files <- list.files(tmp_dir, pattern=paste0(rep_prefix, "_.*\\.pat\\.gz$"), full.names=TRUE)
    if (length(pat_files) == 0) {
        warning("No partial .pat.gz files found in tmp_dir for merging.")
        return(NULL)
    }

    counts <- sapply(pat_files, function(f) {
        tryCatch({
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
    })
    
    cell_types <- sapply(basename(pat_files), function(x) {
        # remove prefix and .pat.gz
        clean_name <- gsub(paste0(rep_prefix, "_"), "", x)
        clean_name <- gsub("\\.pat\\.gz$", "", clean_name)
        # dots to hyphens
        clean_name <- gsub("\\.", "-", clean_name)
        return(clean_name)
    })
    names(counts) <- cell_types
    total_counts <- sum(counts)
    if (total_counts > 0) {
        concentrations <- counts / total_counts
    } else {
        concentrations <- counts
    }
    
    # Write result
    col_names <- gsub("\\.", "-", sort(names(concentrations)))
    result <- data.frame(matrix(concentrations[sort(names(concentrations))], 
                                nrow=1,
                                dimnames=list(NULL, col_names)),
                         check.names=FALSE)
    out_csv <- file.path(out_dir, paste0(rep_prefix, "_true_concentrations.csv"))
    write.csv(result, out_csv, row.names = FALSE, quote = FALSE)
    cat(sprintf("Saved mixture concentrations to: %s\n", out_csv))
}


##
## 6. process_batch() for generating each mixture & merging
##
process_batch <- function(concentrations, prefix, out_dir, threads, tmp_base_dir, 
                         pat_dir, reads_by_celltype, reps_per_combo,
                         coverage_fuzz=0.0) {
    registerDoParallel(cores=threads)
    
    # Generate and merge in parallel
    foreach(i = 1:nrow(concentrations), 
        .export = c("generate_samples", "merge_pat_files"), 
        .packages = c("data.table")) %dopar% {
        tmp_dir <- file.path(tmp_base_dir, sprintf("tmp_%s_%d", prefix, i))
        dir.create(tmp_dir, recursive=TRUE, showWarnings=FALSE)
        
        # Extract the proportions for the i-th mixture
        cell_types_cols <- setdiff(names(concentrations), "depth")
        row_conc <- concentrations[i, ..cell_types_cols]
        
        # "melt" so we have (celltype, fraction)
        conc_table <- melt(
            row_conc,
            measure.vars = cell_types_cols,
            variable.name = "celltype",
            value.name = "fraction"
        )
        
        # The chosen total depth for this mixture
        this_depth <- concentrations[i, depth]
        
        # Process each replicate
        for (rep in 1:reps_per_combo) {
            # Generate samples
            generate_samples(conc_table,
                           reads_by_celltype,
                           target_depth = this_depth,
                           prefix = paste0(prefix, "_", i),
                           tmp_dir = tmp_dir,
                           out_dir = out_dir,
                           pat_dir = pat_dir,
                           rep = rep)
            
            # Merge immediately after generating each replicate
            merge_pat_files(prefix=paste0(prefix, "_", i),
                          rep=rep,
                          tmp_dir=tmp_dir,
                          out_dir=out_dir,
                          coverage_fuzz=coverage_fuzz)
        }
        
        # Clean up temporary directory after processing all replicates for this mixture
        unlink(tmp_dir, recursive=TRUE)
    }
    stopImplicitCluster()
}


##
## 7. main() function
##
main <- function() {
   tic.clearlog()  # Clear any previous timings
   tic("Total Runtime")
   
   # Parse arguments
   tic("Argument Parsing")
   parser <- OptionParser(option_list=option_list)
   args <- parse_args(parser)
   toc(log=TRUE)
   
   # Check required
   if (is.null(args$pat_dir) || is.null(args$output_dir)) {
       print_help(parser)
       stop("Missing required arguments: --pat_dir or --output_dir")
   }
   
   if (is.null(args$tmp_dir)) {
       args$tmp_dir <- file.path(args$output_dir, "tmp")
   }
   
   # Read or load read counts
   tic("Reading Count Data")
   counts_cache_file <- file.path(args$pat_dir, "read_counts.rds")
   if (file.exists(counts_cache_file)) {
       cat("Loading cached read counts...\n")
       reads_by_celltype <- readRDS(counts_cache_file)
   } else {
       cat("Calculating read counts from .pat.gz files...\n")
       reads_by_celltype <- read_count_table(args$pat_dir)
       saveRDS(reads_by_celltype, counts_cache_file)
   }
   toc(log=TRUE)
   
   # Identify all cell types
   cell_types <- c("B-cells", "T-cells", "NK-cells", "Granulocytes", "Monocytes", 
                   "CD34-megakaryocytes", "CD34-erythroblasts", "Esophagus", 
                   "Colon", "Small-intestine", "Gastric", "OAC")
   cat("Using cell types:", paste(cell_types, collapse=", "), "\n")
   
   # Choose concentration generation function based on tier3 flag
   concentration_generator <- if(args$tier3) {
       cat("Using tier3 concentration generation...\n")
       generate_stratified_concentrations_tier3
   } else {
       cat("Using standard concentration generation...\n")
       generate_stratified_concentrations
   }
   
   # Generate training concentrations
   tic("Generating Training Data")
   cat("Generating training concentrations...\n")
   train_concentrations <- if(args$tier3) {
       concentration_generator(
           n_samples = args$n_train,
           cell_types = cell_types,
           min_depth = args$min_depth,
           max_depth = args$max_depth
       )
   } else {
       concentration_generator(
           n_samples = args$n_train,
           cell_types = cell_types,
           zero_fraction = args$zero_fraction,
           min_depth = args$min_depth,
           max_depth = args$max_depth
       )
   }
   toc(log=TRUE)
   
   tic("Generating Evaluation Data")
   cat("Generating evaluation concentrations...\n")
   eval_concentrations <- if(args$tier3) {
       concentration_generator(
           n_samples = args$n_eval,
           cell_types = cell_types,
           min_depth = args$min_depth,
           max_depth = args$max_depth
       )
   } else {
       concentration_generator(
           n_samples = args$n_eval,
           cell_types = cell_types,
           zero_fraction = args$zero_fraction,
           min_depth = args$min_depth,
           max_depth = args$max_depth
       )
   }
   toc(log=TRUE)
   
   # Save concentrations
   tic("Saving Concentration Tables")
   dir.create(args$output_dir, showWarnings=FALSE, recursive=TRUE)
   train_dir <- file.path(args$output_dir, "train")
   eval_dir <- file.path(args$output_dir, "eval")
   dir.create(train_dir, showWarnings=FALSE, recursive=TRUE)
   dir.create(eval_dir, showWarnings=FALSE, recursive=TRUE)
   
   fwrite(train_concentrations, file.path(args$output_dir, "train_concentrations.csv"))
   fwrite(eval_concentrations, file.path(args$output_dir, "eval_concentrations.csv"))
   toc(log=TRUE)
   
   # Create tmp dir
   tmp_base_dir <- args$tmp_dir
   dir.create(tmp_base_dir, showWarnings=FALSE, recursive=TRUE)
   
   # Process training
   tic("Processing Training Samples")
   cat(sprintf("Processing %d training samples with %d threads...\n", args$n_train, args$threads))
   process_batch(
       concentrations = train_concentrations,
       prefix = "train",
       out_dir = train_dir,
       threads = args$threads,
       tmp_base_dir = tmp_base_dir,
       pat_dir = args$pat_dir,
       reads_by_celltype = reads_by_celltype,
       reps_per_combo = args$reps_per_combo,
       coverage_fuzz = args$coverage_fuzz
   )
   toc(log=TRUE)
   
   # Process evaluation
   tic("Processing Evaluation Samples")
   cat(sprintf("Processing %d evaluation samples with %d threads...\n", args$n_eval, args$threads))
   process_batch(
       concentrations = eval_concentrations,
       prefix = "eval",
       out_dir = eval_dir,
       threads = args$threads,
       tmp_base_dir = tmp_base_dir,
       pat_dir = args$pat_dir,
       reads_by_celltype = reads_by_celltype,
       reps_per_combo = args$reps_per_combo,
       coverage_fuzz = args$coverage_fuzz
   )
   toc(log=TRUE)
   
   # Cleanup
   unlink(tmp_base_dir, recursive=TRUE)
   
   toc(log=TRUE)  # Total Runtime
   
   # Print timing summary
   cat("\nTiming Summary:\n")
   print(tic.log(format=TRUE))
}

if(sys.nframe() == 0) {
    main()
}