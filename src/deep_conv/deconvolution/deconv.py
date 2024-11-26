import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from deep_conv.deconvolution.prepare_training import generate_dataset_with_coverage
from deep_conv.deconvolution.train_visualiser import DeconvolutionVisualiser, save_visualisations

class MethylationDataset(Dataset):
    """Dataset for methylation data with coverage information"""
    def __init__(self, methylation_values, coverage, proportions):
        self.methylation = torch.FloatTensor(methylation_values)
        self.coverage = torch.FloatTensor(coverage)
        self.proportions = torch.FloatTensor(proportions)
    def __len__(self):
        return len(self.methylation)
    def __getitem__(self, idx):
        return {
            'methylation': self.methylation[idx],
            'coverage': self.coverage[idx],
            'proportions': self.proportions[idx]
        }


class DeconvolutionModel(nn.Module):
    def __init__(self, n_regions, n_cell_types, hidden_dims=[512, 256]):
        super().__init__()
        self.methylation_encoder = nn.Sequential(
            nn.Linear(n_regions, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.4),  
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.coverage_encoder = nn.Sequential(
            nn.Linear(n_regions, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.combined_layers = nn.Sequential(
            nn.Linear(hidden_dims[1] * 2, n_cell_types)
        )
    def forward(self, methylation, coverage):
        meth_features = self.methylation_encoder(methylation)
        cov_features = self.coverage_encoder(coverage)
        combined = torch.cat([meth_features, cov_features], dim=1)
        logits = self.combined_layers(combined)
        proportions = F.softmax(logits, dim=1)
        return proportions


def train_model(model, train_loader, val_loader, cell_types, output_dir, n_epochs=100, lr=1e-3, 
                weight_decay=1e-4,
                early_stopping_patience=10,
                device='cuda'):
    """Train the deconvolution model with detailed validation logging and visualisations"""
    visualiser = DeconvolutionVisualiser()
    
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    running_train_loss = 0
    beta = 0.98
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        n_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            methylation = batch['methylation'].to(device)
            coverage = batch['coverage'].to(device)
            true_proportions = batch['proportions'].to(device)
            
            optimiser.zero_grad()
            pred_proportions = model(methylation, coverage)
            loss = F.kl_div(pred_proportions.log(), true_proportions, reduction='batchmean')
            loss.backward()
            optimiser.step()
            
            current_loss = loss.item()
            epoch_train_loss += current_loss
            running_train_loss = beta * running_train_loss + (1 - beta) * current_loss
            
            if (batch_idx + 1) % 10 == 0:
                print(f'\rEpoch {epoch+1}/{n_epochs} '
                      f'[{batch_idx+1}/{n_batches}] '
                      f'Current Loss: {current_loss:.4f} | '
                      f'Running Avg: {running_train_loss:.4f} | '
                      f'Epoch Avg: {epoch_train_loss/(batch_idx+1):.4f}', 
                      end='')
                
        epoch_train_loss /= n_batches
        
        # Validation
        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                methylation = batch['methylation'].to(device)
                coverage = batch['coverage'].to(device)
                true_proportions = batch['proportions']
                pred_proportions = model(methylation, coverage).cpu()
                loss = F.kl_div(pred_proportions.log(), true_proportions.to(device), reduction='batchmean')
                val_loss += loss.item()
                all_val_preds.append(pred_proportions.numpy())
                all_val_true.append(true_proportions.numpy())
                
        val_loss /= len(val_loader)
        
        # Update visualiser with epoch metrics
        visualiser.update_training_metrics(
            epoch + 1,
            epoch_train_loss,
            val_loss,
            optimiser.param_groups[0]['lr']
        )
        
        # Calculate validation metrics
        val_preds = np.vstack(all_val_preds)
        val_true = np.vstack(all_val_true)
        val_metrics = evaluate_predictions(val_true, val_preds, cell_types)
        
        # Update visualiser with cell type metrics
        visualiser.update_cell_type_metrics(val_metrics, cell_types)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
            print('✓ New best model saved')
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
            
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{n_epochs} Summary:')
        print(f'Train Loss: {epoch_train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Learning Rate: {optimiser.param_groups[0]["lr"]:.2e}')
        print_validation_results(val_metrics, cell_types)
        print('---\n')
        
        # Save current visualisations
        # if (epoch + 1) % 5 == 0:  # Save every 5 epochs
        save_visualisations(visualiser, output_dir)
    
    # Save final visualisations
    save_visualisations(visualiser, output_dir)
    
    return best_model, visualiser


def evaluate_predictions(y_true, y_pred, cell_types):
    """Calculate comprehensive metrics for predictions"""
    # Overall metrics
    overall_mae = np.mean(np.abs(y_true - y_pred))
    overall_rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    # Per cell type metrics
    cell_type_metrics = {}
    for i, cell_type in enumerate(cell_types):
        true_props = y_true[:, i]
        pred_props = y_pred[:, i]
        mae = np.mean(np.abs(true_props - pred_props))
        rmse = np.sqrt(mean_squared_error(true_props, pred_props))
        r2 = r2_score(true_props, pred_props)
        # Calculate correlation for non-zero true proportions
        mask = true_props > 0.01  # Focus on cell types with >1% proportion
        if mask.sum() > 0:
            corr = np.corrcoef(true_props[mask], pred_props[mask])[0, 1]
        else:
            corr = np.nan
        cell_type_metrics[cell_type] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'corr': corr,
            'mean_true': np.mean(true_props),
            'mean_pred': np.mean(pred_props)
        }
    return {
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'cell_type_metrics': cell_type_metrics
    }


def print_validation_results(metrics, cell_types, top_k=40):
    """Print formatted validation results"""
    print("\nValidation Performance:")
    print(f"Overall MAE: {metrics['overall_mae']:.4f}")
    print(f"Overall RMSE: {metrics['overall_rmse']:.4f}")
    print("\nPer Cell Type Performance:")
    print(f"{'Cell Type':<20} {'True %':<8} {'Pred %':<8} {'MAE':<8} {'R²':<8} {'Corr':<8}")
    print("-" * 70)
    # Sort cell types by true proportion for meaningful ordering
    cell_metrics = metrics['cell_type_metrics']
    sorted_types = sorted(cell_types, 
                         key=lambda x: cell_metrics[x]['mean_true'],
                         reverse=True)
    # Print top K most abundant cell types
    for cell_type in sorted_types[:top_k]:
        m = cell_metrics[cell_type]
        print(f"{cell_type[:20]:<20} "
              f"{m['mean_true']*100:>7.2f} "
              f"{m['mean_pred']*100:>7.2f} "
              f"{m['mae']:>7.4f} "
              f"{m['r2']:>7.4f} "
              f"{m['corr']:>7.4f}")
    if len(sorted_types) > top_k:
        print("...")


def train_deconvolution_model(dataset_dict, model_save_path, batch_size=32):
    """Setup and train the deconvolution model"""
    # Create datasets
    print("before train set")
    train_dataset = MethylationDataset(
        dataset_dict['X_train'],
        dataset_dict['coverage_train'],
        dataset_dict['y_train']
    )
    print("before val set")
    val_dataset = MethylationDataset(
        dataset_dict['X_val'],
        dataset_dict['coverage_val'],
        dataset_dict['y_val']
    )
    # Create dataloaders
    print("before train loader")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    print("before val loader")
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size
    )
    # Initialize model
    print("before model init")
    n_regions = dataset_dict['X_train'].shape[1]
    n_cell_types = dataset_dict['y_train'].shape[1]
    model = DeconvolutionModel(n_regions, n_cell_types)
    print("after model init")
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("starting training with device",device)
    best_model = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        cell_types=dataset_dict['cell_types'], 
        output_dir=model_save_path,        
        device=device
    )
    return model, best_model


def train(atlas_path,model_save_path, n_train: int = 100000, n_val: int = 20000, batch_size=32):
    dataset_dict = generate_dataset_with_coverage(atlas_path, n_train, n_val)
    model, _ = train_deconvolution_model(dataset_dict, model_save_path, batch_size)  
    torch.save(model.state_dict(), model_save_path+"model.pt")