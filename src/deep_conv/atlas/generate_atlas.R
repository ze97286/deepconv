#!/usr/bin/env Rscript


# R_LIBS_USER=~/R/library Rscript /users/zetzioni/sharedscratch/deepconv/src/deep_conv/atlas/generate_atlas.R \
#   --cpg_file /users/zetzioni/sharedscratch/wgbs_tools/references/hg38/CpG.bed.gz \
#   --map_file /users/zetzioni/sharedscratch/atlas/per-read-bed2class.csv \
#   --base_dir /users/zetzioni/sharedscratch/atlas/ \
#   --out_file /users/zetzioni/sharedscratch/atlas/dmr_by_read.blood+gi+tum.100.l4.bed \
#   --class_file /users/zetzioni/sharedscratch/atlas/taps_atlas_class.csv \
#   --top_n 100 \
#   --threads 32 \
#   --verbose


# Required libraries
suppressPackageStartupMessages({
  library(data.table)
  library(optparse)
  library(progressr)
  library(furrr)
})

# Helper functions for data loading
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

# Core functions for region detection
check_region_coverage <- function(region, reads, min_coverage=3, min_cpg_per_read=4, verbose=TRUE) {
  if(verbose) {
    cat(sprintf("\nChecking region %s:%d-%d\n", region$chr[1], region$start[1], region$end[1]))
  }
  
  # Get reads overlapping this region
  region_reads <- reads[chr == region$chr & 
                       pos >= region$start & 
                       pos <= region$end]
  
  if(nrow(region_reads) == 0) {
    if(verbose) cat("No reads found in region\n")
    return(FALSE)
  }
  
  # Count CpGs per read - this is the key change
  reads_with_cpgs <- region_reads[, .(
    cpg_count = uniqueN(pos)  # Count unique CpG positions per read
  ), by=.(group, read_id)]
  
  # Count reads with sufficient CpGs per group
  qualifying_reads <- reads_with_cpgs[cpg_count >= min_cpg_per_read, 
                                    .N, 
                                    by=group]
  
  if(verbose) {
    cat("\nQualifying reads (≥4 CpGs) by group:\n")
    print(qualifying_reads)
  }
  
  # Check we have data for all groups
  all_groups <- unique(reads$group)
  coverage_status <- data.table(group = all_groups)[
    qualifying_reads, 
    on="group"
  ][, N := ifelse(is.na(N), 0, N)]
  
  if(verbose) {
    cat("\nCoverage status for all groups:\n")
    print(coverage_status)
  }
  
  # All groups must have sufficient coverage
  result <- all(coverage_status$N >= min_coverage)
  
  if(verbose) {
    if(!result) {
      cat("\nFailing groups:\n")
      print(coverage_status[N < min_coverage])
    }
    cat(sprintf("\nFinal decision: %s\n", ifelse(result, "PASS", "FAIL")))
  }
  
  return(result)
}

# Get pat files from class file using same logic as UXM command
get_pat_files <- function(class_file, base_dir, verbose=FALSE) {
  # Read the class file - it has no header and comma separator
  classes <- fread(class_file, header=FALSE, sep=",")
  if (verbose) {
    message(sprintf("Read %d lines from class file", nrow(classes)))
  }
  
  # Get file paths and convert to pat paths
  pat_files <- classes$V1
  pat_files <- gsub("\\.beta$", ".pat.gz", pat_files)  # replace .beta at end
  pat_files <- gsub("/beta/", "/pat/Tissue/", pat_files)  # replace /beta/ with /pat/Tissue/
  pat_files <- gsub("\\.calls\\.all", "", pat_files)
  
  # Create named vector mapping to groups
  names(pat_files) <- classes$V2
  
  # Verify each file exists
  if (verbose) {
    for (f in pat_files) {
      message(sprintf("Looking for file: %s", f))
      message(sprintf("File exists: %s", file.exists(f)))
    }
  }
  
  return(pat_files)
}

# Check if a region has sufficient coverage in all groups
verify_region_coverage <- function(region, pat_files, min_cpg_per_read=4, min_coverage=3, verbose=TRUE) {
    region_str <- sprintf("%s:%d-%d", region$chr[1], region$start[1], region$end[1])
    if(verbose) {
        cat(sprintf("\n====== Checking region %s (target: %s) ======\n", 
                   region_str, region$target[1]))
    }
    
    # Store all results
    coverage_data <- list()
    
    # Check each cell type
    for(group in names(pat_files)) {
        file <- pat_files[group]
        if(verbose) {
            cat(sprintf("\n--- Group: %s ---\n", group))
            cat(sprintf("File: %s\n", file))
        }
        
        # Read region from pat file - fixed tabix command
        cmd <- sprintf("tabix '%s' '%s'", file, region_str)
        if(verbose) {
            cat("Running command:", cmd, "\n")
        }
        
        reads <- try(fread(cmd=cmd))
        
        if(inherits(reads, "try-error")) {
            if(verbose) cat("Error reading file!\n")
            coverage_data[[group]] <- list(reads=0, cpgs_per_read=numeric(0), qualifying_reads=0)
            next
        }
        
        if(nrow(reads) == 0) {
            if(verbose) cat("No reads found\n")
            coverage_data[[group]] <- list(reads=0, cpgs_per_read=numeric(0), qualifying_reads=0)
            next
        }
        
        # Count CpGs per read
        read_cpgs <- reads[, .N, by=V4]  # V4 should be read ID
        qualifying_reads <- sum(read_cpgs$N >= min_cpg_per_read)
        
        coverage_data[[group]] <- list(
            reads=nrow(read_cpgs),
            cpgs_per_read=read_cpgs$N,
            qualifying_reads=qualifying_reads
        )
        
        if(verbose) {
            cat(sprintf("Total reads: %d\n", nrow(read_cpgs)))
            cat("CpGs per read: ", paste(read_cpgs$N, collapse=", "), "\n")
            cat(sprintf("Qualifying reads (>=%d CpGs): %d\n", min_cpg_per_read, qualifying_reads))
            cat(sprintf("Meets coverage requirement (>=%d reads): %s\n", 
                       min_coverage, qualifying_reads >= min_coverage))
        }
    }
    
    # Summarize results
    coverage_summary <- data.table(
        group = names(coverage_data),
        reads = sapply(coverage_data, function(x) x$reads),
        qualifying_reads = sapply(coverage_data, function(x) x$qualifying_reads)
    )
    
    if(verbose) {
        cat("\n=== Summary ===\n")
        print(coverage_summary)
    }
    
    # Check if all groups pass
    has_coverage <- all(coverage_summary$qualifying_reads >= min_coverage)
    
    if(verbose) {
        cat(sprintf("\nFinal decision: %s\n", ifelse(has_coverage, "PASS", "FAIL")))
        if(!has_coverage) {
            cat("Failed groups:\n")
            print(coverage_summary[qualifying_reads < min_coverage])
        }
        cat("======================\n\n")
    }
    
    return(has_coverage)
}

# Filter regions based on coverage requirements
filter_regions_by_coverage <- function(regions, pat_files, min_cpg_per_read=4, 
                                     min_coverage=3, verbose=FALSE) {
  if(verbose) {
    message("Checking coverage for ", nrow(regions), " candidate regions...")
  }
  
  regions[, has_coverage := verify_region_coverage(.BY, pat_files, min_cpg_per_read, 
                                                 min_coverage, verbose), 
          by=.(chr, start, end)]
  
  passing_regions <- regions[has_coverage == TRUE]
  
  if(verbose) {
    message(sprintf("Found %d regions with sufficient coverage out of %d candidates", 
                   nrow(passing_regions), nrow(regions)))
  }
  
  return(passing_regions)
}


collapse_to_regions <- function(dmrs, cpg_info, max_gap=1, max_dist=1e3, 
                              min_logp=-20, min_length=100, mad=0.1) {
  n_groups = length(unique(dmrs$group))
  
  if (!identical(key(dmrs), c("chr", "pos"))) {
    setkey(dmrs, chr, pos)
  }
  
  if (!identical(key(cpg_info), c("chr", "start", "end"))) {
    setkey(cpg_info, chr, start, end)
  }
  
  sign.positions <- dmrs[fifelse(pval.med==0, -308.0, log10(pval.med)) < min_logp][
    cpg_info, nomatch=NULL][, 
    .(pos, region_index=cumsum(c(TRUE, diff(index)>max_gap | diff(pos)>max_dist))), 
    by=.(chr, group)]
  
  sign.regions <- sign.positions[, 
    .(start=min(pos), end=max(pos), total_positions=.N), 
    by=.(chr, group, region_index)]
  
  setkey(sign.regions, chr, start, end)
  
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
  
  unique.sign.regions <- dmrs[unique.sign.regions, on=list(group, chr, pos), nomatch=NULL]
  
  unique.sign.regions.stats <- suppressWarnings(
    unique.sign.regions[, .(
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
      n_total=.N
    ), by=.(chr, group, start, end, region_index)])
  
  return(unique.sign.regions.stats)
}

write_marker_file <- function(regions, cpg_info, outfile, top_n = 100) {
  setkey(regions, chr, start, end)
  setkey(cpg_info, chr, start, end)
  
  # Calculate statistics for unique regions
  unique.regions.stat <- foverlaps(
    cpg_info, 
    regions[fully_covered==TRUE], 
    nomatch=NULL)[
      , .(startCpG=min(index), 
          endCpG=max(index) + 1,  # Add 1 to make endCpG exclusive 
          r_len=mean(r_len), 
          p_len=mean(p_len), 
          tg_mean=mean(avg_min_ci), 
          ttest=mean(med_logp), 
          delta_means=mean(avg_min_alpha_dist)), 
      by=.(chr, start=start-1, end=end, target=group)]
  
  # Add direction and select top N regions per target
  unique.regions.stat[, direction := fifelse(tg_mean>0, "M", "U")]
  top_n_regions <- unique.regions.stat[
    direction=="U" & r_len>5, 
    .SD[order(delta_means, -ttest, decreasing = TRUE)][1:top_n], 
    by=target]
  
  # Write the output file
  fwrite(top_n_regions[, .(
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
    make_option(c("-l", "--class_file"), type="character",
                help="Class file for pat files [REQUIRED]",
                metavar="taps_atlas_class.csv"),
    make_option(c("-o", "--out_file"), type="character",
                help="Output file path for markers [REQUIRED]",
                metavar="markers.bed"),
    make_option(c("-n", "--top_n"), type="integer", default=100,
                help="Number of top markers to keep per target [default %default]"),
    make_option(c("-t", "--threads"), type="integer", default=8,
                help="Number of threads [default %default]"),
    make_option(c("-r", "--min_reads"), type="integer", default=3,
                help="Minimum number of qualifying reads per group [default %default]"),
    make_option(c("-g", "--min_cpgs"), type="integer", default=4,
                help="Minimum CpGs per read [default %default]"),
    make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
                help="Print progress information")
  )
  
  parser <- OptionParser(option_list=option_list)
  params <- parse_args(parser)
  
  # Check required arguments
  if (is.null(params$cpg_file) || is.null(params$map_file) || 
      is.null(params$out_file) || is.null(params$class_file)) {
    print_help(parser)
    stop("Missing required arguments")
  }
  
  if (params$verbose) {
    message(Sys.time(), " Starting DMR detection...")
  }
  
  # Get pat files
  pat_files <- get_pat_files(params$class_file, params$base_dir)
  
  if(params$verbose) {
    message(sprintf("Found %d pat files to process", length(pat_files)))
    for(group in names(pat_files)) {
        message(sprintf("%s: %s", group, pat_files[group]))
    }
}
  
  # Load required data
  bed2type <- load_sample2group(params$map_file)
  cpg_info <- load_cpg_info(params$cpg_file)
  
  # Set up parallel processing
  plan(multisession, workers = params$threads)
  
  # Load and process methylation data for all chromosomes
  if (params$verbose) {
    message(Sys.time(), " Loading methylation data...")
  }
  
  with_progress({
    p <- progressor(steps=22)
    pval.all <- future_map(paste0("chr", 1:22), function(chrom) {
      res <- fread(paste0(params$base_dir, "/dmr_by_read/blood+tum+gi_scores-by-position_", 
                         chrom, ".txt.gz"), 
                   header=TRUE, stringsAsFactors = TRUE)
      p()
      return(res)
    }) 
  })
  pval.all <- rbindlist(pval.all)
  
  if (params$verbose) {
    message(Sys.time(), " Finding unique regions...")
  }
  
  # Find candidate regions
  unique.regions <- collapse_to_regions(pval.all, cpg_info)

  # Set keys before foverlaps
  setkey(unique.regions, chr, start, end)
  setkey(cpg_info, chr, start, end)
  
  # Calculate initial stats and add direction
  unique.regions.stat <- foverlaps(
    cpg_info, 
    unique.regions[fully_covered==TRUE], 
    nomatch=NULL)[
      , .(startCpG=min(index), 
          endCpG=max(index) + 1,  # Make endCpG exclusive
          r_len=mean(r_len), 
          p_len=mean(p_len), 
          tg_mean=mean(avg_min_ci), 
          ttest=mean(med_logp), 
          delta_means=mean(avg_min_alpha_dist)), 
      by=.(chr, start=start-1, end=end, target=group)]
  
  unique.regions.stat[, direction := fifelse(tg_mean>0, "M", "U")]
  
  # Get candidate hypomethylated regions for each target
  candidate_regions <- unique.regions.stat[direction=="U" & r_len>5, 
                                         .SD[order(delta_means, -ttest, decreasing = TRUE)], 
                                         by=target]
  
  if (params$verbose) {
    message(sprintf("Found %d candidate regions before coverage check", nrow(candidate_regions)))
  }
  
  # Check read coverage for each region using specified parameters
  candidate_regions[, has_coverage := verify_region_coverage(.SD, pat_files, 
                                                           min_cpg_per_read=params$min_cpgs,
                                                           min_coverage=params$min_reads,
                                                           verbose=params$verbose), 
                   by=.(chr, start, end)]
  
  if (params$verbose) {
    message(sprintf("Found %d regions with sufficient coverage (≥%d reads with ≥%d CpGs)", 
                   sum(candidate_regions$has_coverage), 
                   params$min_reads, 
                   params$min_cpgs))
  }
  
  # Get top N passing regions per target
  top_regions <- candidate_regions[has_coverage==TRUE, 
                                 head(.SD[order(delta_means, -ttest, decreasing=TRUE)], 
                                      n=params$top_n), 
                                 by=target]
  
  if (params$verbose) {
    message(sprintf("Selected top %d regions per target (total %d regions)", 
                   params$top_n, nrow(top_regions)))
    # Show distribution of regions per target
    print(top_regions[, .N, by=target])
  }
  
  # Write output file
  fwrite(top_regions[, .(
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
    params$out_file, 
    sep="\t", 
    quote=FALSE, 
    col.names=TRUE, 
    row.names=FALSE)
  
  if (params$verbose) {
    message(Sys.time(), " Done. Marker file written to ", params$out_file)
  }
}

# Run main function if script is run directly
if (sys.nframe() == 0) {
  main()
}
