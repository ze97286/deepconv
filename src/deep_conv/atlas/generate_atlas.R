#!/usr/bin/env Rscript


# R_LIBS_USER=~/R/library Rscript /users/zetzioni/sharedscratch/deepconv/src/deep_conv/atlas/generate_atlas.R \
#   --cpg_file /users/zetzioni/sharedscratch/wgbs_tools/references/hg38/CpG.bed.gz \
#   --map_file /users/zetzioni/sharedscratch/atlas/per-read-bed2class.csv \
#   --base_dir /users/zetzioni/sharedscratch/atlas/ \
#   --out_file /users/zetzioni/sharedscratch/atlas/dmr_by_read.blood+gi+tum.25_500.l4.bed \
#   --index_file /users/zetzioni/sharedscratch/atlas/pats/cell_type_pat_index_l4.csv.gz \
#   --top_n 25 \
#   --threads 32 \
#   --verbose


# Required libraries
suppressPackageStartupMessages({
  library(data.table)
  library(optparse)
  library(progressr)
  library(furrr)
  library(parallel)
})

# Data loading helper functions
load_sample2group <- function(map_file) {
  bam2type <- fread(map_file, header=TRUE, stringsAsFactors = TRUE)
  bam2type[, sample := gsub(".*/", "", name)]
  bam2type[, sample := as.factor(gsub("_md.*", "", sample))]
  setkey(bam2type, sample)
  return(bam2type)  
}

load_cpg_info <- function(cpg_file) {
  cpg_info <- fread(cpg_file, header=FALSE, stringsAsFactors = TRUE)
  setnames(cpg_info, c("chr", "start", "index"))
  cpg_info[, end:=start]
  setkey(cpg_info, chr, start, end)
  return(cpg_info)
}

verify_region_coverage <- function(regions, coverage_index, min_coverage=3, min_cpgs=4, verbose=FALSE) {
    if(verbose) {
        message("Starting batch coverage verification...")
    }
    
    # Process each chromosome separately to manage memory
    results <- vector("logical", nrow(regions))
    
    for(current_chr in unique(regions$chr)) {
        if(verbose) {
            message(sprintf("Processing chromosome %s...", current_chr))
        }
        
        # Get regions for current chromosome
        chr_idx <- which(regions$chr == current_chr)
        chr_regions <- regions[chr_idx]
        
        # Create start positions table
        positions_dt <- data.table(
            region_id = rep(chr_idx, 
                          times = pmax(0, chr_regions$endCpG - chr_regions$startCpG - min_cpgs + 1)),
            start_idx = unlist(mapply(function(start, end) {
                seq(start, end - min_cpgs)
            }, chr_regions$startCpG, chr_regions$endCpG, 
            SIMPLIFY = FALSE))
        )
        
        if(verbose && nrow(positions_dt) == 0) {
            message(sprintf("  No valid positions for %s", current_chr))
            next
        }
        
        # Get coverage for these positions
        chr_coverage <- coverage_index[chr == current_chr & 
                                     start_idx %in% unique(positions_dt$start_idx)]
        
        if(nrow(chr_coverage) > 0) {
            # Join positions with coverage
            region_coverage <- positions_dt[chr_coverage, 
                                         on = .(start_idx), 
                                         allow.cartesian=TRUE]
            
            # Calculate coverage per region and group
            coverage_summary <- region_coverage[
                !is.na(read_count), 
                .(total_reads = sum(read_count)),
                by = .(region_id, group)
            ]
            
            # Reshape to wide format
            coverage_matrix <- dcast(coverage_summary, 
                                   region_id ~ group, 
                                   value.var = "total_reads",
                                   fill = 0)
            
            # Check coverage requirements
            results[coverage_matrix$region_id] <- 
                apply(coverage_matrix[, -1, with=FALSE], 1, 
                     function(x) all(x >= min_coverage))
        }
        
        if(verbose) {
            n_processed <- length(chr_idx)
            n_passing <- sum(results[chr_idx])
            message(sprintf("  %s: %d/%d regions pass coverage criteria", 
                          current_chr, n_passing, n_processed))
        }
    }
    
    if(verbose) {
        message(sprintf("Found %d regions with sufficient coverage (≥%d reads with ≥%d CpGs)", 
                       sum(results), min_coverage, min_cpgs))
    }
    
    # Add debug info if verbose
    if(verbose) {
        # Randomly sample a few passing regions to verify ranges
        passing_regions <- which(results)
        if(length(passing_regions) > 0) {
            sample_size <- min(5, length(passing_regions))
            sample_idx <- sample(passing_regions, sample_size)
            message("\nSample of passing regions for verification:")
            for(idx in sample_idx) {
                region <- regions[idx]
                message(sprintf("Region %s:%d-%d [startCpG=%d, endCpG=%d]", 
                              region$chr, region$start, region$end, 
                              region$startCpG, region$endCpG))
                message(sprintf("Valid coverage range: %d-%d", 
                              region$startCpG, region$endCpG - min_cpgs))
            }
        }
    }
    
    return(results)
}

# Region processing functions
collapse_to_regions <- function(dmrs, cpg_info, mixture_cell_types, max_gap=1, max_dist=1e3, 
                             min_logp=-20, min_length=100, mad=0.1, verbose=TRUE) {
   n_groups = length(unique(dmrs$group))
   
   if (!identical(key(dmrs), c("chr", "pos"))) {
       setkey(dmrs, chr, pos)
   }
   
   if (!identical(key(cpg_info), c("chr", "start", "end"))) {
       setkey(cpg_info, chr, start, end)
   }
   
   if(verbose) {
       message("Finding significant positions...")
   }
   
   sign.positions <- dmrs[fifelse(pval.med==0, -308.0, log10(pval.med)) < min_logp][
       cpg_info, nomatch=NULL][, 
       .(pos, region_index=cumsum(c(TRUE, diff(index)>max_gap | diff(pos)>max_dist))), 
       by=.(chr, group)]
   
   if(verbose) {
       message("Creating initial regions...")
   }
   
   sign.regions <- sign.positions[, 
       .(start=min(pos), end=max(pos), total_positions=.N), 
       by=.(chr, group, region_index)]
   
   setkey(sign.regions, chr, start, end)
   
   if(verbose) {
       message("Processing overlapping regions...")
   }
   
   overlapping.regions <- foverlaps(sign.regions, sign.regions)[group != i.group]
   setkey(overlapping.regions, group, chr, start, end, region_index)
   setkey(sign.regions, group, chr, start, end, region_index)
   
   overlapping.regions <- overlapping.regions[sign.regions, nomatch=NA][
       , .(chr, group, region_index, start, end, i.total_positions, 
           i.group, i.region_index, i.start, i.end, i.total_positions)]
   
   split_regions  <- function(ref_start, ref_end, starts, ends) {
       if (is.na(starts) || is.na(ends) || is.null(starts) || is.null(ends) || 
           length(starts)==0 || length(ends)==0) {
           return(list(start=ref_start, end=ref_end))
       }
       
       rel_start <- starts-ref_start + 1
       rel_end <- ends-ref_start + 1
       rel_start[rel_start < 1] <- 1
       rel_end[rel_end > ref_end-ref_start + 1] <- ref_end-ref_start + 1
       mask <- rep(TRUE, ref_end-ref_start + 1)
       
       for (i in 1:length(rel_end)) {
           mask[rel_start[i]:rel_end[i]] <- FALSE
       }
       
       remaining_indices <- which(mask)
       clusters <- cumsum(c(TRUE, diff(remaining_indices)!=1))
       output <- list(start=c(), end=c())
       
       for (cluster in unique(clusters)) {
           output$start <- c(output$start, ref_start+min(remaining_indices[clusters==cluster])-1)
           output$end <- c(output$end, ref_start+max(remaining_indices[clusters==cluster])-1)
       }
       
       output$start <- as.integer(output$start)
       output$end <- as.integer(output$end)
       
       return(output)
   }
   
   if(verbose) {
       message("Splitting overlapping regions...")
   }
   
   unique.sign.regions <- overlapping.regions[
       , split_regions(start[1], end[1], i.start, i.end), 
       by=.(chr, group, region_index)]
   
   unique.sign.regions <- unique.sign.regions[!is.na(end) & !is.na(start) & 
                                            end-start >= min_length]
   
   setkey(unique.sign.regions, chr, start, end)
   unique.sign.regions <- foverlaps(cpg_info, unique.sign.regions, nomatch = NULL)
   setnames(unique.sign.regions, "i.start", "pos")
   unique.sign.regions[, i.end:=NULL]
   setkey(unique.sign.regions, group, chr, pos)
   
   if(verbose) {
       message("Joining with methylation data...")
   }
   
   unique.sign.regions <- dmrs[unique.sign.regions, on=list(group, chr, pos), nomatch=NULL]
   
   # First stage - calculate per group stats
   if(verbose) {
       message("Calculating per-group region statistics...")
   }
   
   unique.sign.regions.stats <- unique.sign.regions[, {
       if(verbose) {
           message(sprintf("  Processing region: chr=%s, start=%d, end=%d, group=%s", 
                         chr[1], start[1], end[1], group[1]))
       }
       
       list(
           r_len=length(unique(pos)), 
           p_len=min(end-start+1),
           avg_logp=mean(fifelse(pval.mean==0,-308,log10(pval.mean))),
           med_logp=median(fifelse(pval.med==0,-308,log10(pval.med))),
           min_logp=min(fifelse(pval.min==0,-308,log10(pval.min))),
           max_logp=max(fifelse(pval.max==0,-308,log10(pval.max))),
           fully_covered=all(outgroup_count==n_groups-1),
           avg_min_alpha_dist = mean(min_alpha_dist),
           avg_min_ci = mean(ci.min),
           n_low_alpha_dist=sum(min_alpha_dist < mad),
           n_total=.N,
           region_alpha = mean(mean_alpha_dist)
       )
   }, by=.(chr, group, start, end, region_index)]
   
   # Second stage - calculate between group metrics
   if(verbose) {
       message("Calculating between-group metrics...")
   }
   
  # Second stage - calculate between group metrics
  final_regions <- unique.sign.regions.stats[, {
      if(verbose) {
          message(sprintf("\nCalculating between-group metrics for region: chr=%s, start=%d, end=%d", 
                        chr[1], start[1], end[1]))
      }
      
      # Get mixture values
      mixture_vals <- region_alpha[group %in% mixture_cell_types]
      
      if(verbose) {
          message("Number of mixture values: ", length(mixture_vals))
          message("Mixture values:")
          print(data.table(group=group[group %in% mixture_cell_types], 
                          alpha=mixture_vals))
      }
      
      # Only calculate metrics if we have enough values
      if(length(mixture_vals) > 1) {
          mpd <- mean(abs(outer(mixture_vals, mixture_vals, "-")))
          snr <- var(mixture_vals) / (var(rowMeans(matrix(mixture_vals))) + 1e-10)
          mixture_var <- var(mixture_vals)
      } else {
          mpd <- 0
          snr <- 0
          mixture_var <- 0
      }
      
      if(verbose) {
          message(sprintf("MPD: %.4f, SNR: %.4f, Mixture variance: %.4f", 
                        mpd, snr, mixture_var))
      }
      
      # Create new columns instead of trying to include .SD
      .(
          MPD = mpd,
          SNR = snr,
          mixture_variance = mixture_var
      )
  }, by=.(chr, start, end, region_index)]

  # Now merge back with original stats
  final_regions <- merge(unique.sign.regions.stats, final_regions, 
                        by=c("chr", "start", "end", "region_index"))
    
   return(final_regions)
}

write_marker_file <- function(regions, outfile) {
    fwrite(regions[, .(
        `#chr`=chr, 
        start, 
        end, 
        startCpG, 
        endCpG, 
        target, 
        region=paste0(chr,":",start+1,"-",end), 
        lenCpG=paste0(r_len, "CpGs"), 
        bp=paste0(p_len,"bp"), 
        tg_mean, 
        bg_mean=0.0, 
        dela_means=0.0, 
        delta_quants=0.0, 
        delta_maxmin=0.0, 
        ttest, 
        direction)], 
        outfile, 
        sep="\t", 
        quote=FALSE, 
        col.names=TRUE, 
        row.names=FALSE)
}

calculate_within_mixture_variance <- function(regions, mixture_cell_types, variance_threshold) {
    # Do it all in one operation
    setkey(regions, chr, start, end)  # Ensure keys are set
    
    filtered_regions <- regions[
        target %in% mixture_cell_types,  # First filter to only mixture cell types
        {
            var_val = var(mean_alpha_dist)  # Calculate variance directly
            if(var_val > variance_threshold) .SD  # Only return data if variance exceeds threshold
        },
        by=.(chr, start, end)  # Group by region
    ]
    
    return(filtered_regions)
}


main <- function() {
  option_list <- list(
    make_option(c("-c", "--cpg_file"), type="character",
                help="List of all cpG positions with pos and unique index [REQUIRED]",
                metavar="cpg_map.txt"),
    make_option(c("-m", "--map_file"), type="character",
                help="Map of per-read .bed files and their associated group [REQUIRED]",
                metavar="bed2group.txt"),
    make_option(c("-d", "--base_dir"), type="character", default=".",
                help="Base directory containing input files and for output [default %default]",
                metavar="path"),
    make_option(c("-i", "--index_file"), type="character",
                help="Coverage index file [REQUIRED]",
                metavar="coverage_index.csv"),
    make_option(c("-o", "--out_file"), type="character",
                help="Output file path for markers [REQUIRED]",
                metavar="markers.bed"),
    make_option(c("-n", "--top_n"), type="integer", default=50,
                help="Minimum number of top markers to keep per target [default %default]"),
    make_option(c("--max_per_cell_type"), type="integer", default=500,
                help="Maximum number of top markers to keep per target [default %default]"),
    make_option(c("--mpd_threshold"), type="integer", default=0.001,
                help="Minimum pairwise difference [default %default]"),
    make_option(c("--mdd_threshold"), type="integer", default=0.0001,
                help="Minimum detectable threshold [default %default]"),
    make_option(c("-r", "--min_reads"), type="integer", default=3,
                help="Minimum number of qualifying reads per group [default %default]"),
    make_option(c("-g", "--min_cpgs"), type="integer", default=4,
                help="Minimum CpGs per read [default %default]"),
    make_option(c("-t", "--threads"), type="integer", default=8,
                help="Number of threads [default %default]"),
    make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
                help="Print progress information")
  )
  
  parser <- OptionParser(option_list=option_list)
  params <- parse_args(parser)
  
  # Check required arguments
  if (is.null(params$cpg_file) || is.null(params$map_file) || 
      is.null(params$out_file) || is.null(params$index_file)) {
    print_help(parser)
    stop("Missing required arguments")
  }
  
  if (params$verbose) {
    message(Sys.time(), " Starting DMR detection...")
  }
  
  # Load coverage index with details
  if (params$verbose) {
    message(Sys.time(), " Loading coverage index...")
    start_time <- Sys.time()
  }
  
  coverage_index <- fread(params$index_file, 
                         col.names=c("chr", "start_idx", "group", "read_count"))
  setkey(coverage_index, chr, start_idx, group)
  
  if (params$verbose) {
    end_time <- Sys.time()
    message(sprintf("Loaded index with %d positions across %d groups (%.1f seconds)", 
                   nrow(coverage_index), 
                   uniqueN(coverage_index$group),
                   difftime(end_time, start_time, units="secs")))
    
    message("\nCoverage index statistics:")
    message(sprintf("Positions per chromosome:"))
    print(coverage_index[, .N, by=chr])
    message(sprintf("\nPositions per group:"))
    print(coverage_index[, .N, by=group])
  }
  
  # Load required data
  if (params$verbose) {
    message(Sys.time(), " Loading sample and CpG mappings...")
    start_time <- Sys.time()
  }
  
  bed2type <- load_sample2group(params$map_file)
  cpg_info <- load_cpg_info(params$cpg_file)
  
  if (params$verbose) {
    end_time <- Sys.time()
    message(sprintf("Loaded mappings (%.1f seconds)", 
                   difftime(end_time, start_time, units="secs")))
    message(sprintf("Found %d CpG positions", nrow(cpg_info)))
  }
  
  # Set up parallel processing
  plan(multisession, workers = params$threads)
  
  if (params$verbose) {
    message(Sys.time(), " Loading methylation data...")
    message("Processing chromosomes:")
    start_time <- Sys.time()
  }
  
  # Load and process methylation data for all chromosomes
  chrom_sizes <- numeric(22)
  with_progress({
    p <- progressor(steps=22)
    pval.all <- future_map(paste0("chr", 1:22), function(chrom) {
      if (params$verbose) message(sprintf("  Reading %s...", chrom))
      res <- fread(paste0(params$base_dir, "/dmr_by_read/blood+tum+gi_scores-by-position_", 
                         chrom, ".txt.gz"), 
                   header=TRUE, stringsAsFactors = TRUE)
      chrom_sizes[as.numeric(sub("chr", "", chrom))] <<- nrow(res)
      p()
      return(res)
    }) 
  })
  pval.all <- rbindlist(pval.all)
  
  if (params$verbose) {
    end_time <- Sys.time()
    message(sprintf("Loaded %d methylation positions in %.1f seconds", 
                   nrow(pval.all), 
                   difftime(end_time, start_time, units="secs")))
    message("\nPositions per chromosome:")
    names(chrom_sizes) <- paste0("chr", 1:22)
    print(chrom_sizes)
  }
  
  if (params$verbose) {
    message(Sys.time(), " Finding unique regions...")
    start_time <- Sys.time()
  }
  
  # Find candidate regions
  unique.regions <- collapse_to_regions(pval.all, cpg_info, mixture_cell_types = c("B-cells", "CD34-erythroblasts", "CD34-megakaryocytes", "CD4-T-cells","CD8-T-cells","Eosinophils", "Monocytes", "NK-cells","Neutrophils"),verbose = params$verbose)
  
  if (params$verbose) {
    end_time <- Sys.time()
    message(sprintf("Found %d initial regions in %.1f seconds", 
                   nrow(unique.regions),
                   difftime(end_time, start_time, units="secs")))
  }
  
  # Calculate region statistics
  if (params$verbose) {
    message(Sys.time(), " Calculating region statistics...")
    start_time <- Sys.time()
  }
  
  setkey(unique.regions, chr, start, end)
  setkey(cpg_info, chr, start, end)
  
  unique.regions.stat <- foverlaps(
    cpg_info, 
    unique.regions[fully_covered==TRUE], 
    nomatch=NULL)[
      , .(startCpG=min(index), 
          endCpG=max(index) + 1,
          r_len=mean(r_len), 
          p_len=mean(p_len), 
          tg_mean=mean(avg_min_ci), 
          ttest=mean(med_logp), 
          delta_means=mean(avg_min_alpha_dist)), 
      by=.(chr, start=start-1, end=end, target=group)]
  
  unique.regions.stat[, direction := fifelse(tg_mean>0, "M", "U")]
  
  if (params$verbose) {
    end_time <- Sys.time()
    message(sprintf("Calculated statistics for %d regions in %.1f seconds", 
                   nrow(unique.regions.stat),
                   difftime(end_time, start_time, units="secs")))
  }
  
  # Get candidate regions
  if (params$verbose) {
    message(Sys.time(), " Selecting candidate regions...")
    start_time <- Sys.time()
  }
  
  candidate_regions <- unique.regions.stat[direction=="U" & r_len>5, 
                                         .SD[order(delta_means, -ttest, decreasing = TRUE)], 
                                         by=target]
  
  if (params$verbose) {
    end_time <- Sys.time()
    message(sprintf("Found %d candidate regions in %.1f seconds", 
                   nrow(candidate_regions),
                   difftime(end_time, start_time, units="secs")))
    message("\nCandidates per target:")
    print(candidate_regions[, .N, by=target])
  }
  
  if (params$verbose) {
    message("\nCandidate regions structure:")
    print(str(candidate_regions))
    message("\nFirst few rows of candidate regions:")
    print(head(candidate_regions))
    message("\nColumn names:")
    print(names(candidate_regions))
  }

  # Check coverage
  if (params$verbose) {
    message(Sys.time(), " Checking coverage...")
    start_time <- Sys.time()
  }
  
  candidate_regions[, has_coverage := verify_region_coverage(.SD, 
                                                             coverage_index,
                                                             min_coverage=params$min_reads,
                                                             min_cpgs=params$min_cpgs,
                                                             verbose=params$verbose)]
    
    if (params$verbose) {
        end_time <- Sys.time()
        message(sprintf("\nFound %d regions with sufficient coverage in %.1f seconds", 
                       sum(candidate_regions$has_coverage), 
                       difftime(end_time, start_time, units="secs")))
        
        message("\nPassing regions per target:")
        print(candidate_regions[has_coverage==TRUE, .N, by=target])
    }
  
  # Select final regions
  if (params$verbose) {
    message(Sys.time(), " Selecting final regions...")
  }

  # Apply thresholds
  filtered_regions <- candidate_regions[has_coverage == TRUE & 
                                    SNR > 1]

  # Log after filtering
  if (params$verbose) {
      message("Number of markers per cell type after SNR filtering:")
      print(filtered_regions[, .N, by = target])
  }

  # Check for minimum markers per cell type
  marker_counts <- filtered_regions[, .N, by = target]
  if (any(marker_counts$N < params$top_n)) {
      stop(sprintf("Failed to find at least %d markers for all cell types. Missing: %s", 
                  params$top_n, paste(marker_counts[target %in% marker_counts[N < params$top_n, target]]$target, collapse = ", ")))
  }

  # Sort by combination of metrics including mixture variance
  filtered_regions <- filtered_regions[order(-delta_means, -MPD, -mixture_variance)]

  # Take top N per cell type
  top_regions <- filtered_regions[, head(.SD, params$max_per_cell_type), by = target]

  # Log final number of markers
  if (params$verbose) {
    message(sprintf("Final number of top regions: %d", nrow(top_regions)))
    message("Final distribution of markers by cell type:")
    print(top_regions[, .N, by = target])
  }
  
  # Write output
  if (params$verbose) {
    message(Sys.time(), " Writing output...")
  }
  
  write_marker_file(top_regions, params$out_file)
  
  if (params$verbose) {
    message(Sys.time(), " Done. Marker file written to ", params$out_file)
  }
}

# Run main function if script is run directly
if (sys.nframe() == 0) {
  main()
}