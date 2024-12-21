#!/usr/bin/env Rscript


# R_LIBS_USER=~/R/library Rscript /users/zetzioni/sharedscratch/deepconv/src/deep_conv/atlas/generate_atlas.R \
#   --cpg_file /users/zetzioni/sharedscratch/wgbs_tools/references/hg38/CpG.bed.gz \
#   --map_file /users/zetzioni/sharedscratch/atlas/per-read-bed2class.csv \
#   --base_dir /users/zetzioni/sharedscratch/atlas/ \
#   --out_file /users/zetzioni/sharedscratch/atlas/dmr_by_read.blood+gi+tum.100.l4.bed \
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
check_region_coverage <- function(region, reads, min_coverage=3, min_cpg_per_read=4) {
  region_reads <- reads[chr == region$chr & 
                       pos >= region$start & 
                       pos <= region$end]
  
  if(nrow(region_reads) == 0) {
    return(FALSE)
  }
  
  reads_by_group <- region_reads[, .(
    n_qualifying_reads = sum(.N >= min_cpg_per_read)
  ), by=.(group, read_id)]
  
  sufficient_coverage <- reads_by_group[, .(
    has_coverage = sum(n_qualifying_reads > 0) >= min_coverage
  ), by=group]
  
  return(all(sufficient_coverage$has_coverage))
}

scores_per_pos <- function(groups, successes, totals, p=NULL) {
  if (length(groups)==0) {
    return(NULL)
  }
  else if (length(groups)==1) {
    return(list(group=as.character(groups[1]), pval.mean=1.0, pval.med=1.0, pval.min=1.0, pval.max=1.0, pval.iqr=0.0, ci.min=0.0, ci.max=1.0, outgroup_count=as.integer(0), coverage.in=totals, coverage.out=as.integer(0), mean_alpha_dist=0.0, min_alpha_dist=0.0))
  }
  
  cache_names <- as.data.table(expand.grid(groups, groups, stringsAsFactors = FALSE))[Var1!=Var2, paste0(Var1,':',Var2)]
  is_cached <- vector(mode="logical", length=length(cache_names))
  names(is_cached) <- cache_names
  pval_cache <- vector(mode="double", length=length(cache_names))
  names(pval_cache) <- cache_names
  ci_cache <- vector(mode="double", length=length(cache_names))
  names(ci_cache) <- cache_names
  alpha_dist_cache <- vector(mode="double", length=length(cache_names))
  names(alpha_dist_cache) <- cache_names
  
  scores <- lapply(groups, function(g) {
    pvals <- vector(mode="double", length=length(groups)-1)
    names(pvals) <- groups[groups != g]
    cis <- vector(mode="double", length=length(groups)-1)
    names(cis) <- groups[groups != g]
    alpha_dist <- vector(mode="double", length=length(groups)-1)
    names(alpha_dist) <- groups[groups != g]
    
    for (g2 in groups[groups!=g]) {
      id_str <- paste0(g, ":", g2)
      if (is_cached[id_str]) {
        p <- pval_cache[id_str]
        ci <- ci_cache[id_str]
        alpha_delta <- alpha_dist_cache[id_str]
      } else {
        x <- c(successes[groups==g], successes[groups == g2])
        n <- c(totals[groups==g], totals[groups == g2])
        test.p <- suppressWarnings(prop.test(x, n))
        p <- ifelse(is.na(test.p$p.value), 1, test.p$p.value)
        ci <- ifelse(test.p$conf.int[1]>=0, test.p$conf.int[1], test.p$conf.int[2])
        pval_cache[id_str] <- p
        ci_cache[id_str] <- ci
        rev_str <- paste0(g2, ":", g)
        pval_cache[rev_str] <- pval_cache[id_str]
        ci_cache[rev_str] <- ci_cache[id_str]
        alpha_delta <- abs(x[1]/n[1]-x[2]/n[2])
        alpha_dist_cache[id_str] <- alpha_delta
        alpha_dist_cache[rev_str] <- alpha_delta
      }
      pvals[g2] <- p
      cis[g2] <- ci
      alpha_dist[g2] <- alpha_delta
    }
    
    return(list(pval.mean=mean(pvals), pval.med=median(pvals), pval.min=min(pvals), 
                pval.max=min(pvals), pval.iqr=IQR(pvals), ci.min=min(cis), ci.max=max(cis),
                outgroup_count=sum(groups != g), coverage.in=totals[groups==g], 
                coverage.out=sum(totals[groups != g]), mean_alpha_dist=mean(alpha_dist),
                min_alpha_dist=min(alpha_dist)))
  })
  names(scores) <- as.character(groups)
  scores <- rbindlist(scores, idcol = "group")
  
  if (!is.null(p)) {
    p()
  }
  return(scores)
}

collapse_to_regions <- function(dmrs, cpg_info, reads=NULL, max_gap=1, max_dist=1e3, 
                              min_logp=-20, min_length=100, mad=0.1, 
                              min_coverage=3, min_cpg_per_read=3) {
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
  
  # Check read coverage only if reads are provided
  if (!is.null(reads)) {
    sign.regions[, has_coverage := check_region_coverage(.SD, reads, 
                                                       min_coverage, 
                                                       min_cpg_per_read), 
                 by=.(chr, start, end)]
    
    # Filter regions without sufficient coverage
    sign.regions <- sign.regions[has_coverage == TRUE]
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
  
  # Add direction and select top n hypomethylated regions per target
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
    make_option(c("-o", "--out_file"), type="character",
                help="Output file path for markers [REQUIRED]",
                metavar="markers.bed"),
    make_option(c("-n", "--top_n"), type="integer", default=100,
                help="Number of top markers to keep per target [default %default]"),
    make_option(c("-t", "--threads"), type="integer", default=8,
                help="Number of threads [default %default]"),
    make_option(c("-v", "--verbose"), action="store_true", default=FALSE,
                help="Print progress information")
  )
  
  parser <- OptionParser(option_list=option_list)
  params <- parse_args(parser)
  
  # Check required arguments
  if (is.null(params$cpg_file) || is.null(params$map_file) || is.null(params$out_file)) {
    print_help(parser)
    stop("Missing required arguments")
  }
  
  if (params$verbose) {
    message(Sys.time(), " Starting DMR detection...")
  }
  
  # Load required data
  bed2type <- load_sample2group(params$map_file)
  cpg_info <- load_cpg_info(params$cpg_file)
  
  # Set up parallel processing
  plan(multisession, workers = params$threads)
  
  # Load and process methylation data for all chromosomes
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
  
  # Find unique regions
  unique.regions <- collapse_to_regions(pval.all, cpg_info, 
                                      max_gap = 1, max_dist = 1000, 
                                      min_logp = -30, min_length = 150)
  
  if (params$verbose) {
    message(Sys.time(), " Writing marker file...")
  }
  
  # Write marker file for uxm build
  write_marker_file(unique.regions, cpg_info, params$out_file, params$top_n)
  
  if (params$verbose) {
    message(Sys.time(), " Done.")
  }
}

# Run main function if script is run directly
if (sys.nframe() == 0) {
  main()
}