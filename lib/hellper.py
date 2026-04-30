import os
import numpy as np
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import os


def plot_nse_cd(stage_true, stage_pred, mode, save_path="../plots"):
    """
    Compute and plot the cumulative distribution of NSE scores, and save the NSE data.
    
    Args:
        stage_true: numpy array of shape [nb, nx, ny, nt]
        stage_pred: numpy array of shape [nb, nx, ny, nt]
        mode: String identifier for the run (used for saving the file).
        save_path: Path to save the NSE values and plot.
    
    Returns:
        Figure object.
    """
    def calculate_nse(y_true, y_pred):
        """
        Calculate Nash-Sutcliffe Efficiency for each spatial point and time series.
        
        Args:
            y_true: numpy array of shape [nb, nx, ny, nt]
            y_pred: numpy array of shape [nb, nx, ny, nt]
        
        Returns:
            numpy array of shape [nb, nx, ny] containing NSE values for each spatial point.
        """
        numerator = np.sum((y_true - y_pred) ** 2, axis=-1)
        denominator = np.sum((y_true - np.mean(y_true, axis=-1, keepdims=True)) ** 2, axis=-1)

        # Handle division by zero
        denominator = np.where(denominator == 0, np.ones_like(denominator), denominator)

        nse = 1 - (numerator / denominator)
        return nse

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Calculate NSE for all spatial points and batches
    nse_values = calculate_nse(stage_true, stage_pred)
    
    # Save NSE values for future analysis
    nse_filename = os.path.join(save_path, f"{mode}.npy")
    np.save(nse_filename, nse_values)
    print(f"NSE values saved at: {nse_filename}")


import os
import numpy as np
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

def plot_cumulative_distributions(path):
    """
    Load all .npy files in the specified path, compute their cumulative distributions,
    and plot them interactively using Plotly.

    Args:
        path (str): Path to the directory containing .npy files.
    """
    # Get all .npy files in the directory
    npy_files = [f for f in os.listdir(path) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"No .npy files found in the directory: {path}")
        return
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Generate 100 unique colors using matplotlib colormap
    color_sequence = plt.cm.tab20.colors  # 20 unique colors
    color_sequence += plt.cm.tab20b.colors  # 20 more unique colors
    color_sequence += plt.cm.tab20c.colors  # 20 more unique colors
    color_sequence += plt.cm.Set3.colors  # 12 more unique colors
    color_sequence += plt.cm.Pastel1.colors  # 9 more unique colors
    color_sequence += plt.cm.Pastel2.colors  # 8 more unique colors
    color_sequence += plt.cm.Accent.colors  # 8 more unique colors
    color_sequence += plt.cm.Dark2.colors  # 8 more unique colors
    color_sequence += plt.cm.Paired.colors  # 12 more unique colors
    
    # Convert RGB tuples to hex color codes
    color_sequence = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b, *_ in color_sequence]
    
    # Function to process a single file
    def process_file(file):
        data = np.load(os.path.join(path, file))
        data_flat = data.reshape(-1)
        data_flat = data_flat[~np.isnan(data_flat)]
        data_flat = data_flat[~np.isinf(data_flat)]
        sorted_data = np.sort(data_flat)
        p = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Downsample the data
        step = max(1, len(sorted_data) // 1000)  # Adjust for desired resolution
        sorted_data = sorted_data[::step]
        p = p[::step]
        
        return sorted_data, p, file
    
    # Process files in parallel
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, npy_files))
    
    # Add traces to the figure with unique colors
    for i, (sorted_data, p, file) in enumerate(results):
        # Remove the file extension and add a '.' at the end
        file_name = os.path.splitext(file)[0] + '.'
        color = color_sequence[i % len(color_sequence)]  # Cycle through the color sequence
        fig.add_trace(go.Scatter(
            x=sorted_data,
            y=p,
            mode='lines',
            name=f'Model {i+1}: {file_name}',  # Use the modified filename
            line=dict(width=2, color=color),  # Assign a unique color
            visible=True  # Initially visible
        ))
    
    # Add "Clear All" and "Show All" buttons
    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "Clear All",
                        "method": "update",
                        "args": [{"visible": ["legendonly"] * len(results)}],  # Keep legend, deselect lines
                    },
                    {
                        "label": "Show All",
                        "method": "update",
                        "args": [{"visible": [True] * len(results)}],  # Show all lines
                    },
                ],
                "direction": "down",
                "showactive": True,
                "x": 1.15,
                "y": 1.15
            }
        ],
        title='Cumulative Distribution of NSE Values',
        xaxis_title='Nash-Sutcliffe Efficiency (NSE)',
        yaxis_title='Cumulative Distribution',
        xaxis_range=[0.0, 1.0],
        yaxis_range=[0, 1.0],
        legend_title='Models',
        hovermode='x unified',
        width=1450,  # Set the width of the plot
        height=500  # Set the height of the plot
    )
    
    # Show the plot
    fig.show()


# import os
# import numpy as np
# import plotly.graph_objects as go
# from concurrent.futures import ThreadPoolExecutor
# import matplotlib.pyplot as plt

# def plot_cumulative_distributions(path):
#     """
#     Load all .npy files in the specified path, compute their cumulative distributions,
#     and plot them interactively using Plotly.

#     Args:
#         path (str): Path to the directory containing .npy files.
#     """
#     # Get all .npy files in the directory
#     npy_files = [f for f in os.listdir(path) if f.endswith('.npy')]
    
#     if not npy_files:
#         print(f"No .npy files found in the directory: {path}")
#         return
    
#     # Create a Plotly figure
#     fig = go.Figure()
    
#     # Generate 100 unique colors using matplotlib colormap
#     color_sequence = plt.cm.tab20.colors  # 20 unique colors
#     color_sequence += plt.cm.tab20b.colors  # 20 more unique colors
#     color_sequence += plt.cm.tab20c.colors  # 20 more unique colors
#     color_sequence += plt.cm.Set3.colors  # 12 more unique colors
#     color_sequence += plt.cm.Pastel1.colors  # 9 more unique colors
#     color_sequence += plt.cm.Pastel2.colors  # 8 more unique colors
#     color_sequence += plt.cm.Accent.colors  # 8 more unique colors
#     color_sequence += plt.cm.Dark2.colors  # 8 more unique colors
#     color_sequence += plt.cm.Paired.colors  # 12 more unique colors
    
#     # Convert RGB tuples to hex color codes
#     color_sequence = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b, *_ in color_sequence]
    
#     # Function to process a single file
#     def process_file(file):
#         data = np.load(os.path.join(path, file))
#         data_flat = data.reshape(-1)
#         data_flat = data_flat[~np.isnan(data_flat)]
#         data_flat = data_flat[~np.isinf(data_flat)]
#         sorted_data = np.sort(data_flat)
#         p = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
#         # Downsample the data
#         step = max(1, len(sorted_data) // 1000)  # Adjust for desired resolution
#         sorted_data = sorted_data[::step]
#         p = p[::step]
        
#         return sorted_data, p, file
    
#     # Process files in parallel
#     with ThreadPoolExecutor() as executor:
#         results = list(executor.map(process_file, npy_files))
    
#     # Add traces to the figure with unique colors
#     for i, (sorted_data, p, file) in enumerate(results):
#         # Remove the file extension and add a '.' at the end
#         file_name = os.path.splitext(file)[0] + '.'
#         color = color_sequence[i % len(color_sequence)]  # Cycle through the color sequence
#         fig.add_trace(go.Scatter(
#             x=sorted_data,
#             y=p,
#             mode='lines',
#             name=file_name,  # Use the modified filename
#             line=dict(width=2, color=color)  # Assign a unique color
#         ))
    
#     # Update layout
#     fig.update_layout(
#         title='Cumulative Distribution of NSE Values',
#         xaxis_title='Nash-Sutcliffe Efficiency (NSE)',
#         yaxis_title='Cumulative Distribution',
#         xaxis_range=[0.5, 1.0],
#         yaxis_range=[0, 1.0],
#         legend_title='Models',
#         hovermode='x unified',
#         width=1450,  # Set the width of the plot
#         height=500  # Set the height of the plot
#     )
    
#     # Show the plot
#     fig.show()

# import os
# import pickle
# import pandas as pd
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt

# def plot_error_metrics_interactively(error_path):
#     """
#     Load error metrics from .pkl files in the specified path and plot them interactively using Plotly.

#     Args:
#         error_path (str): Path to the directory containing .pkl files with error metrics.
#     """
#     # Initialize a dictionary to store all error metrics
#     all_error_metrics = {}

#     # Check if the directory exists before proceeding
#     if not os.path.exists(error_path):
#         print(f"Error: Directory {error_path} does not exist or is not accessible.")
#         return

#     # Loop through all .pkl files in the directory
#     for file in os.listdir(error_path):
#         if file.endswith(".pkl"):
#             file_path = os.path.join(error_path, file)
#             with open(file_path, "rb") as f:
#                 error_data = pickle.load(f)
#                 all_error_metrics.update(error_data)  # Merge all stored dictionaries

#     # Convert to DataFrame for better visualization
#     df_errors = pd.DataFrame.from_dict(all_error_metrics, orient='index')

#     # Assign model names (m1, m2, m3, etc.)
#     model_labels = [f"Model {i+1}" for i in range(len(df_errors))]
#     model_mapping = {model_labels[i]: model_name for i, model_name in enumerate(df_errors.index)}
#     df_errors.index = model_labels

#     # Ensure all numerical values are correctly formatted
#     df_errors = df_errors.astype(float)

#     # Generate a large set of unique colors using matplotlib colormaps
#     color_sequence = plt.cm.tab20.colors  # 20 unique colors
#     color_sequence += plt.cm.tab20b.colors  # 20 more unique colors
#     color_sequence += plt.cm.tab20c.colors  # 20 more unique colors
#     color_sequence += plt.cm.Set3.colors  # 12 more unique colors
#     color_sequence += plt.cm.Pastel1.colors  # 9 more unique colors
#     color_sequence += plt.cm.Pastel2.colors  # 8 more unique colors
#     color_sequence += plt.cm.Accent.colors  # 8 more unique colors
#     color_sequence += plt.cm.Dark2.colors  # 8 more unique colors
#     color_sequence += plt.cm.Paired.colors  # 12 more unique colors

#     # Convert RGB tuples to hex color codes
#     color_sequence = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b, *_ in color_sequence]

#     # Plot each metric separately using Plotly
#     metrics = df_errors.columns
#     for metric in metrics:
#         if metric =="R2 Score":
#             continue
#         # Create a Plotly figure
#         fig = go.Figure()

#         # Add bar traces for each model with unique colors
#         for i, (label, value) in enumerate(zip(df_errors.index, df_errors[metric])):
#             color = color_sequence[i % len(color_sequence)]  # Cycle through the color sequence
#             fig.add_trace(go.Bar(
#                 x=[label],
#                 y=[value],
#                 name=f"{label}: {model_mapping[label]}",  # Include model label and full name in legend
#                 marker_color=color,  # Assign unique color
#                 text=f"{value:.3f}",  # Display value on hover
#                 textposition='auto',  # Automatically position text
#                 hovertemplate=f"<b>{label}</b><br>{model_mapping[label]}<br>{metric}: {value:.3f}<extra></extra>"
#             ))

#         # Update layout
#         fig.update_layout(
#             title=f"{metric} Comparison Across Models",
#             xaxis_title="Models",
#             yaxis_title=metric,
#             showlegend=True,  # Show legend for toggling models
#             legend_title='Models',
#             template="plotly_white",  # Use a clean template
#             hovermode="x unified",  # Show hover info for all bars at once
#             barmode="group"  # Group bars for better visibility
#         )

#         # Show the plot
#         fig.show()
import os
import pickle
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def plot_error_metrics_interactively(error_path):
    """
    Load error metrics from .pkl files in the specified path and plot them interactively using Plotly.

    Args:
        error_path (str): Path to the directory containing .pkl files with error metrics.
    """
    # Initialize a dictionary to store all error metrics
    all_error_metrics = {}

    # Check if the directory exists before proceeding
    if not os.path.exists(error_path):
        print(f"Error: Directory {error_path} does not exist or is not accessible.")
        return

    # Loop through all .pkl files in the directory
    for file in os.listdir(error_path):
        if file.endswith(".pkl"):
            file_path = os.path.join(error_path, file)
            with open(file_path, "rb") as f:
                error_data = pickle.load(f)
                all_error_metrics.update(error_data)  # Merge all stored dictionaries

    # Convert to DataFrame for better visualization
    df_errors = pd.DataFrame.from_dict(all_error_metrics, orient='index')

    # Assign model names (m1, m2, m3, etc.)
    model_labels = [f"Model {i+1}" for i in range(len(df_errors))]
    model_mapping = {model_labels[i]: model_name for i, model_name in enumerate(df_errors.index)}
    df_errors.index = model_labels

    # Ensure all numerical values are correctly formatted
    df_errors = df_errors.astype(float)

    # Generate a large set of unique colors using matplotlib colormaps
    color_sequence = plt.cm.tab20.colors  # 20 unique colors
    color_sequence += plt.cm.tab20b.colors  # 20 more unique colors
    color_sequence += plt.cm.tab20c.colors  # 20 more unique colors
    color_sequence += plt.cm.Set3.colors  # 12 more unique colors
    color_sequence += plt.cm.Pastel1.colors  # 9 more unique colors
    color_sequence += plt.cm.Pastel2.colors  # 8 more unique colors
    color_sequence += plt.cm.Accent.colors  # 8 more unique colors
    color_sequence += plt.cm.Dark2.colors  # 8 more unique colors
    color_sequence += plt.cm.Paired.colors  # 12 more unique colors

    # Convert RGB tuples to hex color codes
    color_sequence = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' for r, g, b, *_ in color_sequence]

    # Plot each metric separately using Plotly
    metrics = df_errors.columns
    for metric in metrics:
        if metric == "R2 Score":
            continue

        # Create a Plotly figure
        fig = go.Figure()

        # Add bar traces for each model with unique colors
        trace_visibility = []  # Store visibility states
        for i, (label, value) in enumerate(zip(df_errors.index, df_errors[metric])):
            color = color_sequence[i % len(color_sequence)]  # Cycle through the color sequence
            fig.add_trace(go.Bar(
                x=[label],
                y=[value],
                name=f"{label}: {model_mapping[label]}",  # Include model label and full name in legend
                marker_color=color,  # Assign unique color
                text=f"{value:.3f}",  # Display value on hover
                textposition='auto',  # Automatically position text
                hovertemplate=f"<b>{label}</b><br>{model_mapping[label]}<br>{metric}: {value:.3f}<extra></extra>",
                visible=True  # Initially visible
            ))
            trace_visibility.append(True)  # Store visibility state

        # Add buttons for "Clear All" (legendonly) and "Show All"
        fig.update_layout(
            updatemenus=[
                {
                    "buttons": [
                        {
                            "label": "Clear All",
                            "method": "update",
                            "args": [{"visible": ["legendonly"] * len(df_errors)}],  # Deselect all but keep legend
                        },
                        {
                            "label": "Show All",
                            "method": "update",
                            "args": [{"visible": [True] * len(df_errors)}],  # Show all bars
                        },
                    ],
                    "direction": "down",
                    "showactive": True,
                    "x": 1.15,
                    "y": 1.15
                }
            ],
            title=f"{metric} Comparison Across Models",
            xaxis_title="Models",
            yaxis_title=metric,
            showlegend=True,  # Show legend for toggling models
            legend_title='Models',
            template="plotly_white",  # Use a clean template
            hovermode="x unified",  # Show hover info for all bars at once
            barmode="group"  # Group bars for better visibility
        )

        # Show the plot
        fig.show()
