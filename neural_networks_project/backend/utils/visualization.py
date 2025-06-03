import numpy as np
import base64
import io
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for Docker
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def create_layer_visualization(layer_outputs, model):
    """Create visualization data for each layer of the neural network"""
    visualization_data = {"layers": [], "layer_names": [], "activations": []}

    for i, (layer, output) in enumerate(
        zip(model.layers[1:], layer_outputs)
    ):  # Skip input layer
        layer_info = {
            "name": layer.name,
            "type": layer.__class__.__name__,
            "shape": output.shape,
            "activation_stats": {
                "mean": float(np.mean(output)),
                "std": float(np.std(output)),
                "min": float(np.min(output)),
                "max": float(np.max(output)),
                "zeros": int(np.sum(output == 0)),
                "total": int(np.prod(output.shape)),
            },
        }  # Create full visualization with plots based on layer type
        try:
            if layer.__class__.__name__ in ["Conv2D"]:
                layer_info["visualization"] = create_conv_visualization(
                    output, layer.name
                )
            elif layer.__class__.__name__ in ["Dense"]:
                layer_info["visualization"] = create_dense_visualization(
                    output, layer.name
                )
            elif layer.__class__.__name__ in ["Flatten"]:
                layer_info["visualization"] = create_flatten_visualization(
                    output, layer.name
                )
            else:
                layer_info["visualization"] = {
                    "type": "generic",
                    "stats": layer_info["activation_stats"],
                }
        except Exception as e:
            # Fallback to simplified stats if full visualization fails
            print(f"Full visualization failed for {layer.name}: {str(e)}")
            try:
                if layer.__class__.__name__ in ["Conv2D"]:
                    layer_info["visualization"] = create_conv_visualization_simple(
                        output, layer.name
                    )
                elif layer.__class__.__name__ in ["Dense"]:
                    layer_info["visualization"] = create_dense_visualization_simple(
                        output, layer.name
                    )
                elif layer.__class__.__name__ in ["Flatten"]:
                    layer_info["visualization"] = create_flatten_visualization_simple(
                        output, layer.name
                    )
                else:
                    layer_info["visualization"] = {
                        "type": "generic",
                        "stats": layer_info["activation_stats"],
                    }
            except Exception as e2:
                layer_info["visualization"] = {
                    "type": "error",
                    "message": f"All visualization failed: {str(e2)}",
                    "stats": layer_info["activation_stats"],
                }

        visualization_data["layers"].append(layer_info)
        visualization_data["layer_names"].append(layer.name)

    return visualization_data


def create_conv_visualization(output, layer_name):
    """Create visualization for convolutional layers"""
    # Get first sample from batch
    activation = output[0]  # Shape: (height, width, channels)

    if len(activation.shape) == 3:
        height, width, channels = activation.shape

        # Create a grid of feature maps
        n_cols = min(8, channels)  # Show max 8 feature maps
        n_rows = min(4, (channels + n_cols - 1) // n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
        fig.suptitle(
            f"{layer_name} - Feature Maps", fontsize=14
        )  # Handle different subplot configurations
        if n_rows == 1 and n_cols == 1:
            axes_list = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
        else:
            axes_list = axes.flatten()

        # Ensure we have a list to work with
        if not isinstance(axes_list, (list, np.ndarray)):
            axes_list = [axes_list]

        for i in range(min(channels, n_rows * n_cols)):
            if len(axes_list) > 1:
                ax = axes_list[i]
            else:
                ax = axes_list[0]

            im = ax.imshow(activation[:, :, i], cmap="viridis", aspect="auto")
            ax.set_title(f"Filter {i+1}", fontsize=10)
            ax.axis("off")  # Hide unused subplots
        for i in range(min(channels, n_rows * n_cols), n_rows * n_cols):
            if len(axes_list) > i:
                axes_list[i].axis("off")

        plt.tight_layout()

        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)

        return {
            "type": "feature_maps",
            "image": f"data:image/png;base64,{plot_data}",
            "stats": {
                "feature_maps": channels,
                "spatial_size": f"{height}x{width}",
                "active_neurons": int(np.sum(activation > 0.1)),
                "total_neurons": int(np.prod(activation.shape)),
            },
        }

    return {"type": "unknown", "message": "Could not visualize this layer"}


def create_dense_visualization(output, layer_name):
    """Create visualization for dense layers"""
    activation = output[0]  # Shape: (units,)

    # Create bar plot of neuron activations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{layer_name} - Neuron Activations", fontsize=14)

    # Bar plot of all activations
    x_pos = np.arange(len(activation))
    bars = ax1.bar(x_pos, activation, alpha=0.7, color="steelblue")
    ax1.set_title("All Neurons")
    ax1.set_xlabel("Neuron Index")
    ax1.set_ylabel("Activation Value")
    ax1.grid(True, alpha=0.3)

    # Highlight top activated neurons
    top_indices = np.argsort(activation)[-10:]  # Top 10
    for i in top_indices:
        bars[i].set_color("orange")

    # Histogram of activation values
    ax2.hist(activation, bins=20, alpha=0.7, color="green", edgecolor="black")
    ax2.set_title("Activation Distribution")
    ax2.set_xlabel("Activation Value")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return {
        "type": "dense_activations",
        "image": f"data:image/png;base64,{plot_data}",
        "stats": {
            "neurons": len(activation),
            "active_neurons": int(np.sum(activation > 0.1)),
            "top_activation": float(np.max(activation)),
            "avg_activation": float(np.mean(activation)),
        },
    }


def create_flatten_visualization(output, layer_name):
    """Create visualization for flatten layer"""
    activation = output[0]  # Shape: (flattened_size,)

    # Reshape to approximate square for visualization
    size = len(activation)
    side = int(np.sqrt(size))
    if side * side == size:
        reshaped = activation.reshape(side, side)
    else:
        # Pad to make square
        padded_size = side * side
        if padded_size < size:
            side += 1
            padded_size = side * side
        padded = np.zeros(padded_size)
        padded[:size] = activation
        reshaped = padded.reshape(side, side)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{layer_name} - Flattened Features", fontsize=14)

    # Heatmap visualization
    im1 = ax1.imshow(reshaped, cmap="viridis", aspect="auto")
    ax1.set_title("Feature Map")
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Line plot of activations
    ax2.plot(activation, alpha=0.7, color="blue", linewidth=1)
    ax2.set_title("Activation Values")
    ax2.set_xlabel("Feature Index")
    ax2.set_ylabel("Activation Value")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return {
        "type": "flattened_features",
        "image": f"data:image/png;base64,{plot_data}",
        "stats": {
            "total_features": len(activation),
            "active_features": int(np.sum(activation > 0.1)),
            "sparsity": float(np.sum(activation == 0) / len(activation)),
        },
    }


def create_conv_visualization_simple(output, layer_name):
    """Create simple visualization for convolutional layers without complex plots"""
    activation = output[0]  # Get first sample from batch

    if len(activation.shape) == 3:
        height, width, channels = activation.shape

        # Calculate basic statistics
        active_neurons = int(np.sum(activation > 0.1))
        total_neurons = int(np.prod(activation.shape))

        # Find most active filters
        filter_means = [
            float(np.mean(activation[:, :, i])) for i in range(min(channels, 8))
        ]
        filter_maxes = [
            float(np.max(activation[:, :, i])) for i in range(min(channels, 8))
        ]

        return {
            "type": "feature_maps",
            "stats": {
                "feature_maps": channels,
                "spatial_size": f"{height}x{width}",
                "active_neurons": active_neurons,
                "total_neurons": total_neurons,
                "filter_activations": {"means": filter_means, "maxes": filter_maxes},
            },
        }

    return {"type": "unknown", "message": "Could not visualize this layer"}


def create_dense_visualization_simple(output, layer_name):
    """Create simple visualization for dense layers without plots"""
    activation = output[0]  # Shape: (units,)

    # Calculate statistics
    active_neurons = int(np.sum(activation > 0.1))
    total_neurons = len(activation)

    # Get top activated neurons
    top_indices = np.argsort(activation)[-5:]  # Top 5
    top_activations = [float(activation[i]) for i in top_indices]

    return {
        "type": "dense_activations",
        "stats": {
            "neurons": total_neurons,
            "active_neurons": active_neurons,
            "activation_percentage": float(active_neurons / total_neurons * 100),
            "mean_activation": float(np.mean(activation)),
            "std_activation": float(np.std(activation)),
            "top_neurons": {
                "indices": top_indices.tolist(),
                "activations": top_activations,
            },
        },
    }


def create_flatten_visualization_simple(output, layer_name):
    """Create simple visualization for flatten layers"""
    activation = output[0]  # Shape: (flattened_size,)

    # Calculate basic statistics
    active_neurons = int(np.sum(activation > 0.1))
    total_neurons = len(activation)

    return {
        "type": "flattened",
        "stats": {
            "total_neurons": total_neurons,
            "active_neurons": active_neurons,
            "activation_percentage": float(active_neurons / total_neurons * 100),
            "mean_activation": float(np.mean(activation)),
            "std_activation": float(np.std(activation)),
            "shape": activation.shape,
        },
    }
