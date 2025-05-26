
def train_methylbert_tf_estimator(train_bams, val_bams, known_tf_values, marker_regions):
    """
    Train the MethylBERT TF estimator
    
    Args:
        train_bams: List of BAM files for training
        val_bams: List of BAM files for validation
        known_tf_values: Dictionary mapping BAM files to known TF values
        marker_regions: List of tumor-specific marker regions
    """
    # Initialize model
    model = TumorFractionMethylBERT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = TumorFractionLoss()
    
    # Training loop
    for epoch in range(10):
        model.train()
        for bam_file in train_bams:
            # Extract methylation data
            methylation_data = extract_methylation_from_bam(bam_file)
            
            # Prepare inputs
            inputs = prepare_methylbert_input(methylation_data, marker_regions)
            true_tf = torch.tensor([known_tf_values[bam_file]])
            
            # Forward pass
            outputs = model(**inputs)
            
            # Calculate loss
            loss = criterion(
                outputs["tumor_fraction"], 
                true_tf,
                outputs.get("uncertainty")
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_losses = []
        for bam_file in val_bams:
            methylation_data = extract_methylation_from_bam(bam_file)
            inputs = prepare_methylbert_input(methylation_data, marker_regions)
            true_tf = torch.tensor([known_tf_values[bam_file]])
            
            with torch.no_grad():
                outputs = model(**inputs)
                loss = criterion(
                    outputs["tumor_fraction"], 
                    true_tf,
                    outputs.get("uncertainty")
                )
                val_losses.append(loss.item())
        
        print(f"Epoch {epoch}, Validation Loss: {sum(val_losses)/len(val_losses)}")
    
    return model

def predict_tf_from_bam(model, bam_file, marker_regions):
    """
    Predict tumor fraction from a BAM file
    
    Args:
        model: Trained TumorFractionMethylBERT model
        bam_file: Path to BAM file
        marker_regions: List of tumor-specific marker regions
    
    Returns:
        Estimated tumor fraction and uncertainty
    """
    model.eval()
    
    # Extract methylation data
    methylation_data = extract_methylation_from_bam(bam_file)
    
    # Prepare inputs
    inputs = prepare_methylbert_input(methylation_data, marker_regions)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    return {
        "tumor_fraction": outputs["tumor_fraction"].item(),
        "uncertainty": outputs["uncertainty"].item(),
        "confidence_interval": calculate_confidence_interval(
            outputs["tumor_fraction"].item(),
            outputs["uncertainty"].item()
        )
    }