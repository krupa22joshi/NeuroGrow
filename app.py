import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from io import StringIO, BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification, make_moons, make_circles

# Inbuilt datasets
def generate_dataset(dataset_name, n_samples=100, noise=0.1):
    if dataset_name == "Linear":
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                                  n_clusters_per_class=1, flip_y=noise, random_state=42)
    elif dataset_name == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_name == "Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, random_state=42, factor=0.5)
    else:
        raise ValueError("Unknown dataset type")
    
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df['target'] = y
    return df

# Hebbian learning rule
def hebbian_learning(X, y, learning_rate=0.01, epochs=100):
    n_samples, n_features = X.shape
    weights = np.random.normal(0, 0.1, n_features)
    
    for _ in range(epochs):
        for i in range(n_samples):
            weights += learning_rate * X[i] * y[i]
        weights /= np.linalg.norm(weights) + 1e-8
    
    return weights

# STDP learning rule
def stdp_learning(X, y, learning_rate=0.01, epochs=100, tau_plus=10, tau_minus=10):
    n_samples, n_features = X.shape
    weights = np.random.normal(0, 0.1, n_features)
    
    for _ in range(epochs):
        for i in range(n_samples):
            activation_order = np.argsort(X[i])
            time_diffs = np.arange(n_features) - activation_order
            
            if y[i] > 0:
                weights += learning_rate * X[i] * np.exp(-np.abs(time_diffs)/tau_plus)
            else:
                weights -= learning_rate * X[i] * np.exp(-np.abs(time_diffs)/tau_minus)
        
        weights = weights / (np.linalg.norm(weights) + 1e-8)
    
    return weights

# Hybrid learning rule
def hybrid_learning(X, y, learning_rate=0.01, epochs=100, alpha=0.5):
    hebb_weights = hebbian_learning(X, y, learning_rate, epochs)
    stdp_weights = stdp_learning(X, y, learning_rate, epochs)
    return alpha * hebb_weights + (1 - alpha) * stdp_weights

def predict(X, weights):
    return np.sign(np.dot(X, weights))

def process_data(data_source, dataset_name=None, csv_file=None, 
                learning_rate=0.01, epochs=100, test_size=0.2, 
                n_samples=100, noise=0.1):
    try:
        if data_source == "inbuilt":
            df = generate_dataset(dataset_name, n_samples, noise)
        else:
            # Handle file upload
            if hasattr(csv_file, 'read'):
                content = csv_file.read()
            elif isinstance(csv_file, bytes):
                content = csv_file.decode('utf-8')
            else:
                with open(csv_file, 'r') as f:
                    content = f.read()
            df = pd.read_csv(StringIO(content))
        
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Convert to binary -1/1
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("Target must have exactly 2 classes")
        y = np.where(y == unique_classes[0], -1, 1)
        
        # Preprocess and split data
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        # Train models
        hebb_weights = hebbian_learning(X_train, y_train, learning_rate, epochs)
        stdp_weights = stdp_learning(X_train, y_train, learning_rate, epochs)
        hybrid_weights = hybrid_learning(X_train, y_train, learning_rate, epochs)
        
        # Predictions and accuracies
        models = {
            'Hebbian': hebb_weights,
            'STDP': stdp_weights,
            'Hybrid': hybrid_weights
        }
        
        accuracies = {}
        for name, weights in models.items():
            pred = predict(X_test, weights)
            accuracies[name] = accuracy_score(y_test, pred)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Weight comparison
        x_axis = np.arange(len(hebb_weights))
        ax1.plot(x_axis, hebb_weights, label='Hebbian', marker='o')
        ax1.plot(x_axis, stdp_weights, label='STDP', marker='s')
        ax1.plot(x_axis, hybrid_weights, label='Hybrid', marker='^')
        ax1.set_title('Learned Weights Comparison')
        ax1.set_xlabel('Feature Index')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy comparison
        ax2.bar(accuracies.keys(), accuracies.values(), color=['blue', 'orange', 'green'])
        ax2.set_title('Model Accuracy Comparison')
        ax2.set_ylim(0, 1.1)
        for i, acc in enumerate(accuracies.values()):
            ax2.text(i, acc + 0.02, f"{acc:.3f}", ha='center')
        
        plt.tight_layout()
        
        info = f"""Results:
        Data Info: {X.shape[0]} samples, {X.shape[1]} features
        Classes: {unique_classes[0]} → -1, {unique_classes[1]} → 1
        Accuracies:
        - Hebbian: {accuracies['Hebbian']:.3f}
        - STDP: {accuracies['STDP']:.3f}
        - Hybrid: {accuracies['Hybrid']:.3f}"""
        
        return fig, info
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, error_msg, ha='center', va='center', color='red')
        ax.axis('off')
        return fig, error_msg

# Gradio interface
with gr.Blocks(title="Learning Rules Visualizer") as app:
    gr.Markdown("# Neural Learning Rules Visualizer")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Data Source")
            data_source = gr.Radio(["inbuilt", "upload"], label="Select data source", value="inbuilt")
            
            with gr.Group(visible=True) as inbuilt_group:
                dataset_name = gr.Dropdown(["Linear", "Moons", "Circles"], label="Dataset", value="Linear")
                n_samples = gr.Slider(50, 500, value=100, step=10, label="Samples")
                noise = gr.Slider(0, 0.5, value=0.1, step=0.01, label="Noise level")
            
            with gr.Group(visible=False) as upload_group:
                csv_file = gr.File(label="Upload CSV", file_types=[".csv"])
            
            gr.Markdown("### Learning Parameters")
            learning_rate = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Learning Rate")
            epochs = gr.Slider(10, 500, value=100, step=10, label="Epochs")
            test_size = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="Test Size")
            
            submit_btn = gr.Button("Run Experiment", variant="primary")
        
        with gr.Column(scale=2):
            gr.Markdown("### Results")
            plot_output = gr.Plot()
            text_output = gr.Textbox(label="Details", lines=10)
    
    # Show/hide appropriate controls based on data source
    def toggle_data_source(source):
        return {
            inbuilt_group: gr.Group(visible=(source == "inbuilt")),
            upload_group: gr.Group(visible=(source == "upload"))
        }
    data_source.change(toggle_data_source, data_source, [inbuilt_group, upload_group])
    
    submit_btn.click(
        process_data,
        inputs=[data_source, dataset_name, csv_file, learning_rate, epochs, test_size, n_samples, noise],
        outputs=[plot_output, text_output]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8080)
