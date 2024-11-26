import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


class DeconvolutionVisualiser:
    def __init__(self):
        # Store training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epochs = []
        self.cell_type_metrics = {}
        
    def update_training_metrics(self, epoch, train_loss, val_loss, lr):
        """Update training metrics after each epoch"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
        
    def update_cell_type_metrics(self, metrics, cell_types):
        """Update cell type-specific metrics"""
        self.cell_type_metrics = metrics['cell_type_metrics']
        self.overall_metrics = {
            'overall_mae': metrics['overall_mae'],
            'overall_rmse': metrics['overall_rmse']
        }
        self.cell_types = cell_types

    def plot_training_history(self):
        """Create an interactive plot of training history"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Training and Validation Loss', 'Learning Rate'),
            vertical_spacing=0.15
        )
        
        # Loss curves
        fig.add_trace(
            go.Scatter(x=self.epochs, y=self.train_losses, name="Training Loss",
                      mode='lines', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.epochs, y=self.val_losses, name="Validation Loss",
                      mode='lines', line=dict(color='red')),
            row=1, col=1
        )
        
        # Learning rate curve
        fig.add_trace(
            go.Scatter(x=self.epochs, y=self.learning_rates, name="Learning Rate",
                      mode='lines', line=dict(color='green')),
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

    def plot_cell_type_metrics(self, top_k=20):
        """Create visualisation of cell type-specific metrics with non-overlapping legends"""
        # Prepare data
        metrics_data = []
        for cell_type, metrics in self.cell_type_metrics.items():
            metrics_data.append({
                'Cell Type': cell_type,
                'True %': metrics['mean_true'] * 100,
                'Predicted %': metrics['mean_pred'] * 100,
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                'Correlation': metrics['corr']
            })
        
        df = pd.DataFrame(metrics_data)
        df = df.sort_values('True %', ascending=False).head(top_k)
        
        # Create subplot figure with adjusted spacing
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cell Type Proportions (True vs Predicted)',
                'Performance Metrics by Cell Type',
                'True vs Predicted Scatter',
                'Error Distribution'
            ),
            vertical_spacing=0.25,  # Increased spacing
            horizontal_spacing=0.2   # Increased spacing
        )
        
        # Bar chart comparing true vs predicted proportions
        fig.add_trace(
            go.Bar(name='True %', x=df['Cell Type'], y=df['True %'],
                  marker_color='blue', opacity=0.6),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Predicted %', x=df['Cell Type'], y=df['Predicted %'],
                  marker_color='red', opacity=0.6),
            row=1, col=1
        )
        
        # Performance metrics
        fig.add_trace(
            go.Bar(name='R²', x=df['Cell Type'], y=df['R²'],
                  marker_color='green'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name='MAE', x=df['Cell Type'], y=df['MAE'],
                  marker_color='orange'),
            row=1, col=2
        )
        
        # Scatter plot of true vs predicted with adjusted colorbar position
        scatter_trace = go.Scatter(
            x=df['True %'], 
            y=df['Predicted %'],
            mode='markers',
            text=df['Cell Type'],
            marker=dict(
                size=10,
                color=df['R²'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title='R²',
                    x=0.45,  # Adjusted position
                    y=0.5,   # Centered vertically
                    len=0.4  # Reduced length
                )
            ),
            name='Predictions'
        )
        fig.add_trace(scatter_trace, row=2, col=1)
        
        # Error distribution
        fig.add_trace(
            go.Histogram(
                x=df['True %'] - df['Predicted %'],
                nbinsx=20,
                marker_color='purple',
                opacity=0.7,
                name='Error Distribution'
            ),
            row=2, col=2
        )
        
        # Update layout with adjusted legend position
        fig.update_layout(
            height=1200,  # Increased height
            width=1400,   # Increased width
            title_text="Cell Type Deconvolution Performance",
            showlegend=True,
            template='plotly_white',
            legend=dict(
                orientation="h",     # Horizontal legend
                yanchor="bottom",    # Anchor to bottom
                y=1.02,             # Position above the plots
                xanchor="center",    # Center horizontally
                x=0.5,              # Center position
                bgcolor='rgba(255,255,255,0.8)'  # Semi-transparent background
            )
        )
        
        # Update axes
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text="Percentage", row=1, col=1)
        fig.update_yaxes(title_text="Metric Value", row=1, col=2)
        fig.update_xaxes(title_text="True Percentage", row=2, col=1)
        fig.update_yaxes(title_text="Predicted Percentage", row=2, col=1)
        fig.update_xaxes(title_text="Prediction Error", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        # Ensure proper subplot spacing
        fig.update_layout(
            margin=dict(t=150),  # Increased top margin for legend
        )
        
        return fig


    def plot_correlation_heatmap(self):
        """Create correlation heatmap between true and predicted proportions"""
        # Prepare correlation data
        cell_types = list(self.cell_type_metrics.keys())
        correlations = [self.cell_type_metrics[ct]['corr'] for ct in cell_types]
        
        fig = go.Figure(data=go.Heatmap(
            z=[correlations],
            y=['Correlation'],
            x=cell_types,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title='Cell Type Prediction Correlations',
            xaxis_tickangle=45,
            height=400,
            template='plotly_white'
        )
        
        return fig


def save_visualisations(visualiser, output_dir):
    """Save all visualisations to HTML files"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training history
    training_fig = visualiser.plot_training_history()
    training_fig.write_html(os.path.join(output_dir, 'training_history.html'))
    
    # Save cell type metrics
    metrics_fig = visualiser.plot_cell_type_metrics()
    metrics_fig.write_html(os.path.join(output_dir, 'cell_type_metrics.html'))
    
    # Save correlation heatmap
    corr_fig = visualiser.plot_correlation_heatmap()
    corr_fig.write_html(os.path.join(output_dir, 'correlation_heatmap.html'))