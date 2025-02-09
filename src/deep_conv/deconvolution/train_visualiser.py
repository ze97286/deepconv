import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Any
import wandb
import os


class DeconvolutionVisualiser:
    def __init__(self):
        """Initialize the visualizer with all required attributes."""
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epochs = []
        
        # Range and uncertainty metrics
        self.range_metrics = {}  # Current range metrics
        self.range_metrics_history = defaultdict(lambda: defaultdict(list))  # Historical range metrics
        self.uncertainty_metrics = defaultdict(list)  # Historical uncertainty metrics
        self.uncertainty_data = defaultdict(lambda: defaultdict(list))
        
        # Loss components
        self.loss_components = {'train': {}, 'val': {}}
        
        # Cell type metrics
        self.cell_type_metrics = defaultdict(lambda: defaultdict(list))
        
        # Overall metrics
        self.overall_metrics = defaultdict(list)
        
        # Calibration history
        self.calibration_history = defaultdict(list)


    def update_proportion_range_metrics(self, epoch: int, range_metrics: Dict[str, Dict[str, Dict[str, float]]]):
        """Update range-specific metrics."""
        self.range_metrics = range_metrics  # Store the latest range metrics


    def update_loss_components(self, train_components: Dict[str, float], val_components: Dict[str, float]):
        """Update loss component history."""
        for name, value in train_components.items():
            if name not in self.loss_components['train']:
                self.loss_components['train'][name] = []
            self.loss_components['train'][name].append(value)
            
        for name, value in val_components.items():
            if name not in self.loss_components['val']:
                self.loss_components['val'][name] = []
            self.loss_components['val'][name].append(value)


    def update_uncertainty_metrics(self, epoch: int, uncertainties: np.ndarray, true_proportions: np.ndarray):
        """Update uncertainty metrics for each cell type."""
        if not hasattr(self, 'range_metrics') or not self.range_metrics:
            print("Warning: No range metrics available for uncertainty update")
            return
            
        for cell_type in self.range_metrics.keys():
            cell_idx = list(self.range_metrics.keys()).index(cell_type)
            
            # Extract uncertainties and true proportions for this cell type
            cell_uncertainties = uncertainties[:, cell_idx]
            cell_true_props = true_proportions[:, cell_idx]
            
            # Calculate metrics
            mean_uncertainty = np.mean(cell_uncertainties)
            median_uncertainty = np.median(cell_uncertainties)
            
            # Store metrics
            if cell_type not in self.uncertainty_data:
                self.uncertainty_data[cell_type] = defaultdict(list)
                
            self.uncertainty_data[cell_type]['epoch'].append(epoch)
            self.uncertainty_data[cell_type]['mean'].append(mean_uncertainty)
            self.uncertainty_data[cell_type]['median'].append(median_uncertainty)
            
            # Calculate uncertainty vs true proportion correlation
            if np.std(cell_uncertainties) > 0 and np.std(cell_true_props) > 0:
                correlation = np.corrcoef(cell_uncertainties, cell_true_props)[0, 1]
                self.uncertainty_data[cell_type]['correlation'].append(correlation)
            else:
                self.uncertainty_data[cell_type]['correlation'].append(np.nan)


    def update_training_metrics(self, epoch: int, train_loss: float, val_loss: float, lr: float):
            """Update basic training metrics."""
            self.epochs.append(epoch)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(lr)


    def update_cell_type_metrics(self, metrics: Dict[str, Any], cell_types: List[str]):
        """Update per-cell-type metrics with proper list handling."""
        cell_type_data = metrics.get('cell_type_metrics', {})
        for cell_type in cell_types:
            if cell_type in cell_type_data:
                current_metrics = cell_type_data[cell_type]
                for metric_name, value in current_metrics.items():
                    if isinstance(value, (int, float, np.number)):
                        self.cell_type_metrics[cell_type][metric_name].append(float(value))
                    elif isinstance(value, (list, np.ndarray)):
                        self.cell_type_metrics[cell_type][metric_name].append(value.tolist())



    def plot_loss_components(self) -> go.Figure:
        """
        Create visualization of loss components.
        
        Returns:
            go.Figure: Plotly figure object.
        """
        components = [
            ('focal_loss', 'Focal Loss'),
            ('weighted_mse', 'Weighted MSE'),
            ('uncertainty_reg', 'Uncertainty Regularization'),
            ('recon_loss', 'Reconstruction Loss'),
            ('sparsity_loss', 'Sparsity Loss')
        ]
        
        n_components = len(components)
        n_cols = 2
        n_rows = (n_components + 1) // n_cols
        
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[title for _, title in components])
        
        for idx, (comp_name, title) in enumerate(components, start=1):
            row = (idx - 1) // n_cols + 1
            col = (idx - 1) % n_cols + 1
            
            train_key = f'train_{comp_name}'
            val_key = f'val_{comp_name}'
            
            if train_key in self.loss_components:
                fig.add_trace(
                    go.Scatter(
                        y=self.loss_components[train_key],
                        name=f'Train {title}',
                        line=dict(color='blue')
                    ),
                    row=row, col=col
                )
            
            if val_key in self.loss_components:
                fig.add_trace(
                    go.Scatter(
                        y=self.loss_components[val_key],
                        name=f'Val {title}',
                        line=dict(color='red')
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(height=400 * n_rows, showlegend=True, title_text="Loss Components Over Epochs")
        return fig

    def plot_uncertainty_distributions(self) -> go.Figure:
        """
        Plot uncertainty distributions across proportion ranges.
        
        Returns:
            go.Figure: Plotly figure object.
        """
        ranges = ['ultra_low', 'very_low', 'low', 'medium', 'high']
        metrics = ['mean', 'std', 'calibration_90']
        
        fig = make_subplots(
            rows=len(metrics), cols=1,
            subplot_titles=['Mean Uncertainty', 'Uncertainty Std Dev', 'Calibration (90%)']
        )
        
        colors = px.colors.qualitative.Set3
        
        for i, metric in enumerate(metrics, 1):
            for range_name, color in zip(ranges, colors):
                metric_key = f'{range_name}_{metric}'
                if metric_key in self.uncertainty_metrics and len(self.uncertainty_metrics[metric_key]) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=self.epochs,
                            y=self.uncertainty_metrics[metric_key],
                            name=f'{range_name}',
                            line=dict(color=color)
                        ),
                        row=i, col=1
                    )
        
        fig.update_layout(
            height=300 * len(metrics),
            showlegend=True,
            title_text="Uncertainty Metrics Over Epochs"
        )
        
        return fig

    def plot_training_history(self) -> go.Figure:
        """
        Create an interactive plot of training and validation loss and learning rate.
        
        Returns:
            go.Figure: Plotly figure object.
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Training and Validation Loss', 'Learning Rate'),
            vertical_spacing=0.15
        )
        
        # Loss curves
        fig.add_trace(
            go.Scatter(
                x=self.epochs, 
                y=self.train_losses, 
                name="Training Loss",
                mode='lines',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=self.epochs, 
                y=self.val_losses, 
                name="Validation Loss",
                mode='lines',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # Learning rate curve
        fig.add_trace(
            go.Scatter(
                x=self.epochs, 
                y=self.learning_rates, 
                name="Learning Rate",
                mode='lines',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            title_text="Training History",
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Learning Rate", type="log", row=2, col=1)
        
        return fig

    def plot_cell_type_metrics(self, top_k: int = 20) -> go.Figure:
        """
        Create visualization of cell type-specific metrics with uncertainty information.
        
        Args:
            top_k (int, optional): Number of top cell types to display. Defaults to 20.
        
        Returns:
            go.Figure: Plotly figure object.
        """
        if not hasattr(self, 'cell_type_metrics') or not self.cell_type_metrics:
            print("Warning: No cell type metrics available for plotting")
            return None
            
        # Prepare data
        metrics_data = []
        for cell_type, metrics_dict in self.cell_type_metrics.items():
            data = {
                'Cell Type': cell_type,
                'True %': np.mean(metrics_dict.get('mean_true', [0])) * 100,
                'Predicted %': np.mean(metrics_dict.get('mean_pred', [0])) * 100,
                'MAE': np.mean(metrics_dict.get('mae', [0])),
                'R²': np.mean(metrics_dict.get('r2', [0])),
                'Correlation': np.mean(metrics_dict.get('corr', [0]))
            }
            # Add uncertainty metrics if available
            if 'mean_uncertainty' in metrics_dict:
                data.update({
                    'Uncertainty': np.mean(metrics_dict['mean_uncertainty']),
                    'Calibration': np.mean(metrics_dict.get('calibration_90', [0]))
                })
            metrics_data.append(data)
        
        if not metrics_data:
            print("Warning: No metrics data to plot")
            return None
            
        df = pd.DataFrame(metrics_data)
        df = df.sort_values('MAE', ascending=True).head(top_k)
        
        # Create subplot figure
        n_rows = 2
        n_cols = 3 if 'Uncertainty' in df.columns else 2
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=(
                'Cell Type Proportions',
                'Performance Metrics',
                'Uncertainty Analysis' if 'Uncertainty' in df.columns else None,
                'True vs Predicted',
                'Error Distribution',
                'Calibration Plot' if 'Uncertainty' in df.columns else None
            ),
            vertical_spacing=0.2,
            horizontal_spacing=0.1
        )
        
        try:
            # Proportions comparison
            fig.add_trace(
                go.Bar(
                    name='True %',
                    x=df['Cell Type'],
                    y=df['True %'],
                    marker_color='blue',
                    opacity=0.6
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(
                    name='Predicted %',
                    x=df['Cell Type'],
                    y=df['Predicted %'],
                    marker_color='red',
                    opacity=0.6
                ),
                row=1, col=1
            )
            
            # Performance metrics
            fig.add_trace(
                go.Bar(
                    name='R²',
                    x=df['Cell Type'],
                    y=df['R²'],
                    marker_color='green'
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(
                    name='MAE',
                    x=df['Cell Type'],
                    y=df['MAE'],
                    marker_color='orange'
                ),
                row=1, col=2
            )
            
            # True vs Predicted scatter
            fig.add_trace(
                go.Scatter(
                    x=df['True %'],
                    y=df['Predicted %'],
                    mode='markers',
                    text=df['Cell Type'],
                    marker=dict(
                        size=10,
                        color=df['R²'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Predictions'
                ),
                row=2, col=1
            )
            
            # Error distribution
            errors = df['True %'] - df['Predicted %']
            fig.add_trace(
                go.Histogram(
                    x=errors,
                    nbinsx=30,
                    name='Error Distribution',
                    marker_color='purple'
                ),
                row=2, col=2
            )
            
            # Add uncertainty plots if available
            if 'Uncertainty' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['True %'],
                        y=df['Uncertainty'],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=df['MAE'],
                            colorscale='RdYlBu_r',
                            showscale=True
                        ),
                        name='Uncertainty'
                    ),
                    row=1, col=3
                )
                
                if 'Calibration' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['Uncertainty'],
                            y=df['Calibration'],
                            mode='markers',
                            name='Calibration (90%)',
                            marker=dict(
                                size=10,
                                color='cyan'
                            )
                        ),
                        row=2, col=3
                    )
            
            # Update layout
            fig.update_layout(
                height=900,
                width=1500,
                showlegend=True,
                title_text="Cell Type Metrics Visualization",
                template='plotly_white'
            )
            
            # Update axis labels
            fig.update_xaxes(title_text="Cell Type", row=1, col=1)
            fig.update_xaxes(title_text="Cell Type", row=1, col=2)
            fig.update_xaxes(title_text="True %", row=2, col=1)
            fig.update_xaxes(title_text="Error", row=2, col=2)
            
            fig.update_yaxes(title_text="Proportion %", row=1, col=1)
            fig.update_yaxes(title_text="Value", row=1, col=2)
            fig.update_yaxes(title_text="Predicted %", row=2, col=1)
            fig.update_yaxes(title_text="Count", row=2, col=2)
            
            return fig
        except Exception as e:
            print(f"Error in plot_cell_type_metrics: {str(e)}")
            return None   

    def plot_correlation_heatmap(self) -> go.Figure:
        """
        Create correlation heatmap between true and predicted proportions with uncertainty.
        
        Returns:
            go.Figure: Plotly figure object.
        """
        # Prepare correlation data
        cell_types = list(self.cell_type_metrics.keys())
        
        # Create correlation matrix
        corr_data = []
        for metric in ['corr']:
            row = []
            for cell_type in cell_types:
                value = self.cell_type_metrics[cell_type].get(metric, np.nan)
                row.append(value)
            corr_data.append(row)
        
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=('Prediction Correlation',)
        )
        
        fig.add_trace(
            go.Heatmap(
                z=corr_data,
                x=cell_types,
                y=['Correlation'],
                colorscale='RdBu',
                zmin=-1, zmax=1,
                colorbar=dict(title='Correlation')
            ),
            row=1, col=1
        )
        
        fig.update_layout(
            height=400,
            title='Cell Type Prediction Correlations',
            xaxis_tickangle=45,
            template='plotly_white'
        )
        
        return fig

    def plot_range_metrics(self) -> go.Figure:
        """
        Create figures showing metrics over epochs for each proportion range.
        
        Returns:
            go.Figure: Plotly figure object.
        """
        metrics_to_plot = [
            ('mae', 'MAE'),
            ('rmse', 'RMSE'),
            ('r2', 'R²'),
            ('mean_uncertainty', 'Mean Uncertainty'),
            ('calibration_90', 'Calibration (90%)')
        ]
        
        n_metrics = len(metrics_to_plot)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[title for _, title in metrics_to_plot],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        colors = px.colors.qualitative.Set3
        
        for idx, (metric, title) in enumerate(metrics_to_plot, start=1):
            row = (idx - 1) // n_cols + 1
            col = (idx - 1) % n_cols + 1
            
            for range_name, color in zip(self.range_metrics_history.keys(), colors):
                metric_values = self.range_metrics_history[range_name].get(metric, [])
                epochs = self.range_metrics_history[range_name].get('epochs', [])
                if metric_values and epochs:
                    fig.add_trace(
                        go.Scatter(
                            x=epochs,
                            y=metric_values,
                            name=range_name,
                            line=dict(color=color)
                        ),
                        row=row, col=col
                    )
        
        fig.update_layout(
            height=600 * n_rows,
            showlegend=True,
            title_text="Proportion Range Metrics Over Epochs",
            template='plotly_white'
        )
        
        return fig

    def plot_calibration_metrics(self) -> go.Figure:
        """
        Create calibration plots to assess the alignment between predicted uncertainties and actual errors.
        
        Returns:
            go.Figure: Plotly figure object.
        """
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=('Calibration (90%) Across Ranges',)
        )
        
        for range_name, values in self.uncertainty_metrics.items():
            if 'calibration_90' in range_name:
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(1, len(values) + 1),
                        y=values,
                        mode='lines+markers',
                        name=range_name.replace('_calibration_90', '').capitalize()
                    )
                )
        
        fig.update_layout(
            height=400,
            title='Calibration (90%) Across Proportion Ranges',
            xaxis_title='Epoch',
            yaxis_title='Calibration Rate',
            yaxis=dict(range=[0, 1]),
            template='plotly_white'
        )
        
        return fig

    def save_to_wandb(self, wandb_run, global_step: int):
        """Save visualizations to wandb with consistent step counting."""
        if wandb_run is None:
            return
        
        # Create wandb table for training curves
        n_epochs = len(self.epochs)
        steps_per_epoch = global_step // max(n_epochs, 1)
        
        train_table = wandb.Table(
            columns=["step", "train_loss", "val_loss"],
            data=[[e * steps_per_epoch, t, v] for e, t, v in zip(self.epochs, self.train_losses, self.val_losses)]
        )
        
        # Create wandb table for learning rate
        lr_table = wandb.Table(
            columns=["step", "learning_rate"],
            data=[[e * steps_per_epoch, lr] for e, lr in zip(self.epochs, self.learning_rates)]
        )
        
        # Log training curves using global_step
        wandb_run.log({
            'Visualizations/Training_Loss': wandb.plot.line(
                train_table,
                "step",
                ["train_loss", "val_loss"],
                title='Training and Validation Loss'
            )
        }, step=global_step)
        
        # Log learning rate
        wandb_run.log({
            'Visualizations/Learning_Rate': wandb.plot.line(
                lr_table,
                "step",
                "learning_rate",
                title='Learning Rate Over Time'
            )
        }, step=global_step)
        
        # Log uncertainty metrics
        for cell_type, data in self.uncertainty_data.items():
            if data['epoch']:  # Check if we have data
                # Create table for uncertainty metrics
                uncertainty_table = wandb.Table(
                    columns=["step", "mean", "median", "correlation"],
                    data=[[e * steps_per_epoch, m, med, c] for e, m, med, c in zip(
                        data['epoch'],
                        data['mean'],
                        data['median'],
                        data['correlation']
                    )]
                )
                wandb_run.log({
                    f'Uncertainty/{cell_type}/Metrics': wandb.plot.line(
                        uncertainty_table,
                        "step",
                        ["mean", "median", "correlation"],
                        title=f'Uncertainty Metrics for {cell_type}'
                    )
                }, step=global_step)
                
                # Also log current values
                wandb_run.log({
                    f'Uncertainty/{cell_type}/Mean': data['mean'][-1],
                    f'Uncertainty/{cell_type}/Median': data['median'][-1],
                    f'Uncertainty/{cell_type}/Correlation': data['correlation'][-1]
                }, step=global_step)


def save_visualisations(visualiser: DeconvolutionVisualiser, output_dir: str):
    """
    Save all visualizations to HTML and PNG files with error handling.
    
    Args:
        visualiser (DeconvolutionVisualiser): The visualizer instance containing all metrics and plots.
        output_dir (str): Directory to save the visualization files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define all visualisations to save
    visualisations = [
        ('training_history', visualiser.plot_training_history, 'Training History'),
        ('cell_type_metrics', visualiser.plot_cell_type_metrics, 'Cell Type Metrics'),
        ('correlation_heatmap', visualiser.plot_correlation_heatmap, 'Correlation Heatmap'),
        ('loss_components', visualiser.plot_loss_components, 'Loss Components'),
        ('uncertainty_distributions', visualiser.plot_uncertainty_distributions, 'Uncertainty Distributions'),
        ('range_metrics', visualiser.plot_range_metrics, 'Range Metrics'),
        ('calibration_metrics', visualiser.plot_calibration_metrics, 'Calibration Metrics')
    ]
    
    # Save each visualisation
    for filename_prefix, plot_func, title in visualisations:
        try:
            # Create the plot
            fig = plot_func()
            
            # Update layout with consistent styling
            fig.update_layout(
                title_text=title,
                template='plotly_white',
                font=dict(size=12)
            )
            
            # Save as HTML for interactivity
            html_path = os.path.join(output_dir, f'{filename_prefix}.html')
            fig.write_html(html_path)
            
            # Save as PNG for quick viewing
            png_path = os.path.join(output_dir, f'{filename_prefix}.png')
            try:
                fig.write_image(png_path, scale=2)  # Higher resolution
            except Exception as e:
                print(f"Warning: Could not save PNG for {filename_prefix}: {e}")
            
            print(f"✓ Saved {filename_prefix}")
            
        except Exception as e:
            print(f"Error saving {filename_prefix}: {e}")
    
    # Save a summary HTML file that links to all visualisations
    summary_path = os.path.join(output_dir, 'summary.html')
    try:
        with open(summary_path, 'w') as f:
            f.write('<html><body>\n')
            f.write('<h1>Training Visualizations</h1>\n')
            f.write('<ul>\n')
            for filename_prefix, _, title in visualisations:
                f.write(f'<li><a href="{filename_prefix}.html">{title}</a></li>\n')
            f.write('</ul>\n')
            f.write('</body></html>')
        print("✓ Saved visualization summary")
        
    except Exception as e:
        print(f"Error saving summary: {e}")
    
    # Save metrics as JSON for later analysis
    try:
        import json
        metrics_path = os.path.join(output_dir, 'metrics.json')
        metrics = {
            'overall_metrics': visualiser.overall_metrics,
            'cell_type_metrics': visualiser.cell_type_metrics,
            'range_metrics_history': visualiser.range_metrics_history,
            'uncertainty_metrics': dict(visualiser.uncertainty_metrics),
            'loss_components': dict(visualiser.loss_components),
            'calibration_history': dict(visualiser.calibration_history)
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print("✓ Saved metrics JSON")
        
    except Exception as e:
        print(f"Error saving metrics JSON: {e}")