import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F

class MethylBERTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=0
        )
        
        # Position embeddings for CpG positions within reads
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        
        # Region embeddings to distinguish different genomic regions
        self.region_embeddings = nn.Embedding(
            config.max_regions,
            config.hidden_size
        )
        
        # Layer normalisation and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            MethylBERTLayer(config) for _ in range(config.num_hidden_layers)
        ])
    
    def forward(self, input_ids, attention_mask, position_ids, region_ids):
        # Create embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        region_embeds = self.region_embeddings(region_ids.unsqueeze(-1).expand(-1, -1, input_ids.size(-1)))
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds + region_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Flatten for transformer processing
        batch_size, num_reads, seq_len, hidden_size = embeddings.shape
        embeddings = embeddings.view(batch_size, num_reads * seq_len, hidden_size)
        
        # Adjust attention mask
        attention_mask = attention_mask.view(batch_size, num_reads * seq_len)
        
        # Pass through transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Reshape back
        hidden_states = hidden_states.view(batch_size, num_reads, seq_len, hidden_size)
        
        return hidden_states

class MethylBERTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        
        self.LayerNorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.LayerNorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, attention_mask):
        # Self-attention
        attention_mask_bool = attention_mask.bool()
        attn_output, _ = self.attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~attention_mask_bool
        )
        
        # Add & Norm
        hidden_states = self.LayerNorm1(hidden_states + self.dropout(attn_output))
        
        # Feed-forward
        intermediate_output = F.gelu(self.intermediate(hidden_states))
        layer_output = self.output(intermediate_output)
        
        # Add & Norm
        hidden_states = self.LayerNorm2(hidden_states + self.dropout(layer_output))
        
        return hidden_states

class ReadLevelProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Per-read classification head
        self.read_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Read importance weighting
        self.read_attention = nn.Linear(config.hidden_size, 1)
        
        # Coverage-based read reliability
        self.coverage_processor = nn.Sequential(
            nn.Linear(1, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, read_embeddings, coverage_info, region_ids):
        """
        Process individual reads to get tumour probability scores
        
        Args:
            read_embeddings: [batch_size, num_reads, hidden_size]
            coverage_info: [batch_size, num_regions]
            region_ids: [batch_size, num_reads]
        
        Returns:
            read_scores: [batch_size, num_reads] - tumour probability per read
            read_weights: [batch_size, num_reads] - importance weights
        """
        batch_size, num_reads, hidden_size = read_embeddings.shape
        
        # Get tumour probability for each read
        read_scores = self.read_classifier(read_embeddings).squeeze(-1)  # [batch_size, num_reads]
        
        # Calculate read importance weights
        read_importance = self.read_attention(read_embeddings).squeeze(-1)  # [batch_size, num_reads]
        
        # Adjust weights based on coverage of the region each read belongs to
        coverage_weights = torch.zeros_like(read_importance)
        for batch_idx in range(batch_size):
            for read_idx in range(num_reads):
                region_id = region_ids[batch_idx, read_idx]
                if region_id < coverage_info.shape[1]:  # Valid region
                    coverage_weight = self.coverage_processor(
                        coverage_info[batch_idx, region_id].unsqueeze(0)
                    )
                    coverage_weights[batch_idx, read_idx] = coverage_weight.squeeze()
        
        # Combine importance and coverage weights
        read_weights = F.softmax(read_importance * coverage_weights, dim=1)
        
        return read_scores, read_weights

class RegionLevelAggregator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Region-specific processing
        self.region_processor = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2)
        )
        
        # Region attention for final aggregation
        self.region_attention = nn.Sequential(
            nn.Linear(config.hidden_size // 2 + 1, config.hidden_size // 4),  # +1 for coverage
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1)
        )
        
    def forward(self, read_embeddings, read_scores, read_weights, region_ids, coverage_info):
        """
        Aggregate reads by region and then aggregate regions
        
        Returns:
            sample_embedding: [batch_size, hidden_size] - final sample representation
            region_weights: [batch_size, num_regions] - attention weights for regions
        """
        batch_size, num_reads, hidden_size = read_embeddings.shape
        max_regions = coverage_info.shape[1]
        
        # Aggregate reads within each region
        region_embeddings = []
        region_tumour_scores = []
        region_coverage = []
        
        for batch_idx in range(batch_size):
            batch_region_embeddings = []
            batch_region_scores = []
            batch_region_coverage = []
            
            for region_id in range(max_regions):
                # Find reads belonging to this region
                region_mask = (region_ids[batch_idx] == region_id)
                
                if region_mask.sum() == 0:
                    # No reads for this region
                    batch_region_embeddings.append(torch.zeros(hidden_size, device=read_embeddings.device))
                    batch_region_scores.append(torch.tensor(0.0, device=read_embeddings.device))
                    batch_region_coverage.append(torch.tensor(0.0, device=read_embeddings.device))
                else:
                    # Aggregate reads for this region
                    region_read_embeddings = read_embeddings[batch_idx][region_mask]
                    region_read_weights = read_weights[batch_idx][region_mask]
                    region_read_scores = read_scores[batch_idx][region_mask]
                    
                    # Weighted average of read embeddings
                    region_weights_norm = region_read_weights / (region_read_weights.sum() + 1e-8)
                    region_embedding = torch.sum(
                        region_weights_norm.unsqueeze(-1) * region_read_embeddings, 
                        dim=0
                    )
                    
                    # Weighted average of read tumour scores
                    region_tumour_score = torch.sum(region_weights_norm * region_read_scores)
                    
                    batch_region_embeddings.append(region_embedding)
                    batch_region_scores.append(region_tumour_score)
                    batch_region_coverage.append(coverage_info[batch_idx, region_id])
            
            region_embeddings.append(torch.stack(batch_region_embeddings))
            region_tumour_scores.append(torch.stack(batch_region_scores))
            region_coverage.append(torch.stack(batch_region_coverage))
        
        region_embeddings = torch.stack(region_embeddings)  # [batch_size, num_regions, hidden_size]
        region_tumour_scores = torch.stack(region_tumour_scores)  # [batch_size, num_regions]
        region_coverage = torch.stack(region_coverage)  # [batch_size, num_regions]
        
        # Process region embeddings
        processed_regions = self.region_processor(region_embeddings)  # [batch_size, num_regions, hidden_size//2]
        
        # Calculate region attention weights
        region_attention_input = torch.cat([
            processed_regions, 
            region_coverage.unsqueeze(-1)
        ], dim=-1)  # [batch_size, num_regions, hidden_size//2 + 1]
        
        region_attention_scores = self.region_attention(region_attention_input).squeeze(-1)  # [batch_size, num_regions]
        
        # Mask out regions with no coverage
        coverage_mask = (region_coverage > 0).float()
        region_attention_scores = region_attention_scores * coverage_mask - 1e9 * (1 - coverage_mask)
        
        region_weights = F.softmax(region_attention_scores, dim=1)  # [batch_size, num_regions]
        
        # Final sample embedding
        sample_embedding = torch.sum(
            region_weights.unsqueeze(-1) * processed_regions, 
            dim=1
        )  # [batch_size, hidden_size//2]
        
        return sample_embedding, region_weights

class TumourFractionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Multi-scale TF estimation (similar to your mixture of experts)
        self.high_tf_head = nn.Sequential(
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size // 4, 1)
        )
        
        self.medium_tf_head = nn.Sequential(
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size // 4, 1)
        )
        
        self.low_tf_head = nn.Sequential(
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.GELU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.hidden_size // 4, 1)
        )
        
        # Gating mechanism to choose between heads
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 3),
            nn.Softmax(dim=-1)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Softplus()
        )
        
        # Zero detection (similar to your zero anchoring)
        self.zero_detector = nn.Sequential(
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, sample_embedding):
        """
        Estimate tumour fraction from sample embedding
        
        Args:
            sample_embedding: [batch_size, hidden_size//2]
            
        Returns:
            tumour_fraction: [batch_size, 1]
            uncertainty: [batch_size, 1] 
            zero_prob: [batch_size, 1]
        """
        # Get predictions from each head
        high_tf = torch.sigmoid(self.high_tf_head(sample_embedding)) * 0.5 + 0.5  # [0.5, 1.0]
        medium_tf = torch.sigmoid(self.medium_tf_head(sample_embedding)) * 0.45 + 0.05  # [0.05, 0.5]
        low_tf = torch.sigmoid(self.low_tf_head(sample_embedding)) * 0.049 + 0.001  # [0.001, 0.05]
        
        # Gate weights
        gate_weights = self.gate(sample_embedding)  # [batch_size, 3]
        
        # Weighted combination
        tumour_fraction = (
            gate_weights[:, 0:1] * high_tf +
            gate_weights[:, 1:2] * medium_tf +
            gate_weights[:, 2:3] * low_tf
        )
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(sample_embedding)
        
        # Zero detection
        zero_prob = self.zero_detector(sample_embedding)
        
        # Apply zero masking
        tumour_fraction = tumour_fraction * (1 - zero_prob)
        
        # Ensure valid range
        tumour_fraction = torch.clamp(tumour_fraction, 0.0, 1.0)
        
        return tumour_fraction, uncertainty, zero_prob

class TFMethylBERTOutput:
    def __init__(self):
        self.tumor_fraction: torch.FloatTensor    # [batch_size, 1] - main prediction
        self.uncertainty: torch.FloatTensor       # [batch_size, 1] - epistemic uncertainty
        self.read_scores: torch.FloatTensor       # [batch_size, max_reads] - per-read tumor probability
        self.region_weights: torch.FloatTensor    # [batch_size, num_regions] - region attention weights
        self.confidence_interval: torch.FloatTensor  # [batch_size, 2] - [lower, upper] bounds

class TumourFractionMethylBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Core components
        self.methylbert_encoder = MethylBERTEncoder(config)
        self.read_processor = ReadLevelProcessor(config)
        self.region_aggregator = RegionLevelAggregator(config)
        self.tf_head = TumourFractionHead(config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask, position_ids, region_ids, coverage_info):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, max_reads, max_cpg_sites]
            attention_mask: [batch_size, max_reads, max_cpg_sites]
            position_ids: [batch_size, max_reads, max_cpg_sites]
            region_ids: [batch_size, max_reads]
            coverage_info: [batch_size, num_regions]
            
        Returns:
            TFMethylBERTOutput object
        """
        # Encode reads with MethylBERT
        hidden_states = self.methylbert_encoder(
            input_ids, attention_mask, position_ids, region_ids
        )
        
        # Get read-level embeddings (use CLS token embedding)
        read_embeddings = hidden_states[:, :, 0, :]  # [batch_size, max_reads, hidden_size]
        
        # Process reads
        read_scores, read_weights = self.read_processor(
            read_embeddings, coverage_info, region_ids
        )
        
        # Aggregate by regions
        sample_embedding, region_weights = self.region_aggregator(
            read_embeddings, read_scores, read_weights, region_ids, coverage_info
        )
        
        # Estimate tumour fraction
        tumour_fraction, uncertainty, zero_prob = self.tf_head(sample_embedding)
        
        # Calculate confidence intervals
        confidence_interval = self._calculate_confidence_interval(tumour_fraction, uncertainty)
        
        return TFMethylBERTOutput(
            tumour_fraction=tumour_fraction,
            uncertainty=uncertainty,
            read_scores=read_scores,
            region_weights=region_weights,
            confidence_interval=confidence_interval
        )
    
    def _calculate_confidence_interval(self, mean, std, confidence=0.95):
        """Calculate confidence interval"""
        from scipy.stats import norm
        z_score = norm.ppf((1 + confidence) / 2)
        
        lower = torch.clamp(mean - z_score * std, 0.0, 1.0)
        upper = torch.clamp(mean + z_score * std, 0.0, 1.0)
        
        return torch.cat([lower, upper], dim=-1)

class TFMethylBERTConfig:
    def __init__(self):
        # Model architecture
        self.vocab_size = 6  # [PAD, UNK, UNMETH, METH, CLS, SEP]
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.max_position_embeddings = 512
        self.max_regions = 1000
        
        # Regularization
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.classifier_dropout = 0.1
        self.layer_norm_eps = 1e-12
        
        # Initialization
        self.initializer_range = 0.02

class TumorFractionLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Loss component weights
        self.mse_weight = 1.0
        self.log_mse_weight = 3.0
        self.uncertainty_weight = 0.2
        self.zero_classification_weight = 2.0
        self.read_consistency_weight = 0.5
        
    def forward(self, outputs, targets):
        """
        Calculate comprehensive loss
        
        Args:
            outputs: TFMethylBERTOutput
            targets: dict with 'tumor_fraction', 'is_zero', etc.
        """
        pred_tf = outputs.tumor_fraction
        true_tf = targets['tumor_fraction']
        
        losses = {}
        
        # 1. Basic MSE loss
        mse_loss = F.mse_loss(pred_tf, true_tf)
        losses['mse'] = mse_loss
        
        # 2. Log-space MSE for better handling of low TF values
        epsilon = 1e-6
        log_pred = torch.log10(pred_tf + epsilon)
        log_true = torch.log10(true_tf + epsilon)
        log_mse = F.mse_loss(log_pred, log_true)
        losses['log_mse'] = log_mse
        
        # 3. Uncertainty calibration loss
        if outputs.uncertainty is not None:
            # Encourage uncertainty to match actual error
            errors = torch.abs(pred_tf - true_tf)
            uncertainty_loss = F.mse_loss(outputs.uncertainty.squeeze(), errors.squeeze())
            losses['uncertainty'] = uncertainty_loss
        
        # 4. Zero classification loss
        if 'is_zero' in targets:
            zero_targets = targets['is_zero'].float()
            zero_loss = F.binary_cross_entropy(
                outputs.zero_prob.squeeze() if hasattr(outputs, 'zero_prob') else torch.zeros_like(zero_targets),
                zero_targets
            )
            losses['zero_classification'] = zero_loss
        
        # 5. Read consistency loss (reads should be consistent with final prediction)
        if outputs.read_scores is not None:
            # Higher TF should correlate with more reads having high tumor probability
            read_consistency_loss = self._calculate_read_consistency_loss(
                outputs.read_scores, pred_tf
            )
            losses['read_consistency'] = read_consistency_loss
        
        # Total loss
        total_loss = (
            self.mse_weight * losses['mse'] +
            self.log_mse_weight * losses['log_mse'] +
            self.uncertainty_weight * losses.get('uncertainty', 0) +
            self.zero_classification_weight * losses.get('zero_classification', 0) +
            self.read_consistency_weight * losses.get('read_consistency', 0)
        )
        
        losses['total'] = total_loss
        return losses
    
    def _calculate_read_consistency_loss(self, read_scores, sample_tf):
        """
        Ensure read-level scores are consistent with sample-level TF
        """
        # Average read score should roughly correlate with sample TF
        avg_read_score = read_scores.mean(dim=1, keepdim=True)
        consistency_loss = F.mse_loss(avg_read_score, sample_tf)
        return consistency_loss


