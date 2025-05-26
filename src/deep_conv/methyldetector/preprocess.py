
def extract_taps_methylation(read, reference_genome):
    """
    Extract methylation patterns from TAPS data
    
    In TAPS:
    - Methylated C → G-to-A in original top strand
    - Unmethylated C → C remains C
    """
    methylation_pattern = []
    
    read_seq = read.query_sequence
    ref_positions = read.get_reference_positions()
    
    for i, ref_pos in enumerate(ref_positions):
        ref_base = reference_genome.fetch(read.reference_name, ref_pos, ref_pos+1).upper()
        
        # Check if this is a CpG site
        if ref_base == 'C' and reference_genome.fetch(
                read.reference_name, ref_pos+1, ref_pos+2).upper() == 'G':
            
            read_base = read_seq[i]
            
            if read_base == 'T':  # G-to-A on original top strand = methylated
                methylation_pattern.append(1)  # Methylated
            elif read_base == 'C':  # Unmethylated
                methylation_pattern.append(0)  # Unmethylated
            else:
                methylation_pattern.append(2)  # Unknown
    
    return methylation_pattern


def process_with_read_aggregation(methylation_data, model):
    """
    Process methylation data with two levels:
    1. Read-level classification
    2. Region-level aggregation
    
    This preserves both the individual read information and provides
    a way to estimate tumor fraction.
    """
    # First, classify individual reads
    read_classifications = []
    
    for region, reads in methylation_data.items():
        for read_pattern in reads:
            # Format for MethylBERT
            inputs = format_single_read(read_pattern)
            
            # Get read classification
            with torch.no_grad():
                tumor_prob = model.classify_read(**inputs)
            
            read_classifications.append({
                "region": region,
                "pattern": read_pattern,
                "tumor_prob": tumor_prob
            })
    
    # Calculate tumor fraction from read classifications
    tumor_reads = sum(1 for r in read_classifications if r["tumor_prob"] > 0.5)
    total_reads = len(read_classifications)
    
    return tumor_reads / total_reads if total_reads > 0 else 0.0

