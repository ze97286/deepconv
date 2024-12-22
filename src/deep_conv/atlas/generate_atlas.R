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

# Get pat files from class file using same logic as UXM command
get_pat_files <- function(class_file, base_dir, verbose=FALSE) {
  # Read the class file with header
  classes <- fread(class_file, sep=",", header=TRUE)
  if (verbose) {
    message(sprintf("Read %d lines from class file", nrow(classes)))
  }
  
  # Get file paths and convert to pat paths
  pat_files <- classes$name
  pat_files <- gsub("\\.beta$", ".pat.gz", pat_files)  # replace .beta at end
  pat_files <- gsub("/beta/", "/pat/Tissue/", pat_files)  # replace /beta/ with /pat/Tissue/
  pat_files <- gsub("\\.calls\\.all", "", pat_files)
  
  # Create list where each group has all its files
  result <- split(pat_files, classes$group)
  
  if(verbose) {
    message("\nPat files by group:")
    for(group in names(result)) {
      message(sprintf("\n%s (%d files):", group, length(result[[group]])))
      message(paste("  ", result[[group]], collapse="\n"))
    }
  }
  
  return(result)
}

build_coverage_index <- function(pat_files, min_cpgs=4, threads=8, verbose=FALSE) {
    if(verbose) {
        message("Building coverage index...")
    }
    
    plan(multisession, workers = threads)
    
    with_progress({
        p <- progressor(steps=length(names(pat_files)))
        coverage_by_group <- future_map(names(pat_files), function(group) {
            if(verbose) {
                message(sprintf("Processing group %s (%d files)...", 
                              group, length(pat_files[[group]])))
            }
            
            group_coverage <- list()
            for(file in pat_files[[group]]) {
                reads <- fread(file, select=1:4, 
                             col.names=c("chr", "start_idx", "pattern", "count"))
                
                for(i in 1:nrow(reads)) {
                    pattern <- reads$pattern[i]
                    chars <- strsplit(pattern, "")[[1]]
                    ct_positions <- which(chars %in% c("C", "T"))
                    
                    for(j in 1:length(ct_positions)) {
                        if(length(ct_positions) - j + 1 >= min_cpgs) {
                            pos <- reads$start_idx[i] + ct_positions[j] - 1
                            key <- sprintf("%s_%d", reads$chr[i], pos)
                            group_coverage[[key]] <- (if(is.null(group_coverage[[key]])) 0 else group_coverage[[key]]) + 
                                                   reads$count[i]
                        }
                    }
                }
            }
            
            if(length(group_coverage) > 0) {
                dt <- data.table(
                    key = names(group_coverage),
                    count = unlist(group_coverage)
                )
                dt[, c("chr", "pos") := tstrsplit(key, "_")]
                dt[, pos := as.integer(pos)]
                dt[, key := NULL]
                dt[, group := group]
                p()
                return(dt)
            }
            p()
            return(NULL)
        })
    })
    
    coverage_dt <- rbindlist(Filter(Negate(is.null), coverage_by_group))
    setkey(coverage_dt, chr, pos, group)
    
    if(verbose) {
        message(sprintf("Built index with %d positions across %d groups", 
                       uniqueN(coverage_dt[, .(chr, pos)]), 
                       uniqueN(coverage_dt$group)))
    }
    
    return(coverage_dt)
}

save_coverage_index <- function(coverage_index, params, outfile, verbose=FALSE) {
    # Save both the index and the parameters used to create it
    metadata <- list(
        min_cpgs = params$min_cpgs,
        pat_files_hash = digest::digest(params$class_file, file=TRUE),
        creation_time = Sys.time(),
        version = "1.0"
    )
    
    to_save <- list(
        metadata = metadata,
        coverage_index = coverage_index
    )
    
    saveRDS(to_save, file=outfile)
    
    if(verbose) {
        message(sprintf("Saved coverage index to %s", outfile))
    }
}

load_coverage_index <- function(infile, params, verbose=FALSE) {
    if(!file.exists(infile)) {
        if(verbose) message("No existing index file found")
        return(NULL)
    }
    
    if(verbose) message(sprintf("Loading coverage index from %s", infile))
    
    data <- readRDS(infile)
    
    # Verify metadata matches current parameters
    current_hash <- digest::digest(params$class_file, file=TRUE)
    if(data$metadata$min_cpgs != params$min_cpgs || 
       data$metadata$pat_files_hash != current_hash) {
        if(verbose) {
            message("Existing index was built with different parameters or input files")
        }
        return(NULL)
    }
    
    return(data$coverage_index)
}

get_coverage_index <- function(pat_files, params, verbose=FALSE) {
    # Construct index filename based on input files
    index_file <- file.path(params$base_dir, 
                           sprintf("coverage_index_minCpg%d.rds", params$min_cpgs))
    
    # Try to load existing index
    coverage_index <- load_coverage_index(index_file, params, verbose)
    
    if(is.null(coverage_index)) {
        if(verbose) message("Building new coverage index...")
        
        # Build new index
        coverage_index <- build_coverage_index(pat_files, 
                                             min_cpgs=params$min_cpgs,
                                             threads=params$threads,
                                             verbose=verbose)
        
        # Save for future use
        save_coverage_index(coverage_index, params, index_file, verbose)
    }
    
    return(coverage_index)
}


verify_region_coverage <- function(region, coverage_index, min_coverage=3, verbose=FALSE) {
    if(verbose) {
        cat(sprintf("\nChecking coverage for region %s:%d-%d\n", 
                   region$chr[1], region$startCpG, region$endCpG))
    }
    
    # Get coverage for all positions in the region
    region_coverage <- coverage_index[chr == region$chr[1] & 
                                    pos >= region$startCpG & 
                                    pos <= region$endCpG]
    
    # Check if all groups have sufficient coverage
    group_coverage <- region_coverage[, .(min_count = min(count)), by=group]
    has_coverage <- nrow(group_coverage) == uniqueN(coverage_index$group) && 
                   all(group_coverage$min_count >= min_coverage)
    
    if(verbose) {
        if(nrow(region_coverage) == 0) {
            cat("No coverage data found for region\n")
        } else {
            cat("Coverage by group:\n")
            print(group_coverage)
            cat(sprintf("Has sufficient coverage: %s\n", has_coverage))
        }
    }
    
    return(has_coverage)
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
  
  if (params$verbose) {
    total_files <- sum(sapply(pat_files, length))
    message(sprintf("Found %d cell types with total of %d pat files:", 
                   length(pat_files), total_files))
    for(group in names(pat_files)) {
      message(sprintf("  %s: %d files", group, length(pat_files[[group]])))
    }
  }
  
  # Get or build coverage index
  coverage_index <- get_coverage_index(pat_files, params, verbose=params$verbose)
  
  # Load required data
  bed2type <- load_sample2group(params$map_file)
  cpg_info <- load_cpg_info(params$cpg_file)
  
  # Set up parallel processing
  plan(multisession, workers = params$threads)
  
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
    
    # Calculate region statistics once
    setkey(unique.regions, chr, start, end)
    setkey(cpg_info, chr, start, end)
    
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
    
    # Get candidate hypomethylated regions
    candidate_regions <- unique.regions.stat[direction=="U" & r_len>5, 
                                           .SD[order(delta_means, -ttest, decreasing = TRUE)], 
                                           by=target]
    
    if (params$verbose) {
        message(sprintf("Found %d candidate regions before coverage check", 
                       nrow(candidate_regions)))
    }
    
    # Check coverage
    candidate_regions[, has_coverage := verify_region_coverage(.SD, coverage_index,
                                                             min_coverage=params$min_reads,
                                                             verbose=params$verbose), 
                     by=.(chr, startCpG, endCpG)]
    
    if (params$verbose) {
        message(sprintf("Found %d regions with sufficient coverage (≥%d reads with ≥%d CpGs)", 
                       sum(candidate_regions$has_coverage), 
                       params$min_reads, 
                       params$min_cpgs))
    }
    
    # Select top N regions that pass coverage
    top_regions <- candidate_regions[has_coverage==TRUE, 
                                   head(.SD[order(delta_means, -ttest, decreasing=TRUE)], 
                                        n=params$top_n), 
                                   by=target]
    
    if (params$verbose) {
        message(sprintf("Selected top %d regions per target (total %d regions)", 
                       params$top_n, nrow(top_regions)))
        message("\nDistribution by target:")
        print(top_regions[, .N, by=target])
    }
    
    # Write final marker file
    write_marker_file(top_regions, params$out_file)
    
    if (params$verbose) {
        message(Sys.time(), " Done. Marker file written to ", params$out_file)
    }
}

# Run main function if script is run directly
if (sys.nframe() == 0) {
  main()
}
