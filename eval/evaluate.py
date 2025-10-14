import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch.nn import Linear
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import yaml
import copy
import sys
from matplotlib.ticker import PercentFormatter

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_dims_from_filename(filename):
    """
    Parse hidden dimensions from filenames like 'gcn_224_128.pt'
    Returns list of dimensions
    """
    numbers = re.findall(r'\d+', filename)
    return [int(num) for num in numbers] if numbers else [128, 128]

def parse_architecture_from_filename(filename):
    """Parse architecture from filename"""
    if 'gcn' in filename:
        return 'gcn'
    elif 'gat' in filename:
        return 'gat'
    elif 'sage' in filename:
        return 'sage'
    return 'gcn'  # default

# ------------------- Flexible Model Definitions -------------------
class FlexibleGCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims):
        super(FlexibleGCN, self).__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        self.layers = torch.nn.ModuleList()
        dims = [in_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.layers.append(GCNConv(dims[i], dims[i + 1]))
        
        self.fc = Linear(dims[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        embedding = x
        x = self.fc(x)
        return embedding, x

class FlexibleGAT(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, heads=8):
        super(FlexibleGAT, self).__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        self.layers = torch.nn.ModuleList()
        dims = [in_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            if i == 0:
                # First layer
                self.layers.append(GATConv(dims[i], dims[i + 1] // heads, heads=heads, concat=True))
            elif i == len(dims) - 2:
                # Last layer
                self.layers.append(GATConv(dims[i], dims[i + 1], heads=1, concat=False))
            else:
                # Middle layers
                self.layers.append(GATConv(dims[i], dims[i + 1] // heads, heads=heads, concat=True))
        
        self.fc = Linear(dims[-1], out_dim)
        self.heads = heads

    def forward(self, data):
        x, edge_index = data
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        embedding = x
        x = self.fc(x)
        return embedding, x

class FlexibleGraphSage(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims):
        super(FlexibleGraphSage, self).__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        self.layers = torch.nn.ModuleList()
        dims = [in_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.layers.append(SAGEConv(dims[i], dims[i + 1]))
        
        self.fc = Linear(dims[-1], out_dim)

    def forward(self, data):
        x, edge_index = data
        for layer in self.layers:
            x = F.relu(layer(x, edge_index))
        embedding = x
        x = self.fc(x)
        return embedding, x

def create_flexible_model(arch, in_dim, out_dim, filename):
    """
    Create a model with flexible architecture based on filename
    """
    dims = parse_dims_from_filename(filename)
    hidden_dims = dims[:-1] if len(dims) > 1 else dims
    final_dim = dims[-1] if dims else out_dim
    
    print(f"Creating {arch} model with hidden_dims={hidden_dims}, final_dim={final_dim}")
    
    if arch == 'gcn':
        return FlexibleGCN(in_dim, out_dim, hidden_dims)
    elif arch == 'gat':
        return FlexibleGAT(in_dim, out_dim, hidden_dims, heads=8)
    elif arch == 'sage':
        return FlexibleGraphSage(in_dim, out_dim, hidden_dims)

def evaluate_gnn(model, x, edge_index, test_mask, data):
    model.eval()
    with torch.no_grad():
        _, out = model((x, edge_index))
        pred = out.argmax(dim=1)
        return accuracy_score(data.y[test_mask].cpu(), pred[test_mask].cpu())

def get_posteriors(model, x, edge_index, mask):
    model.eval()
    with torch.no_grad():
        _, out = model((x, edge_index))
        selected = out[mask]
        return selected.detach().cpu()

def train_classifier(X, y, input_dim, hidden_layers=None, epochs=100, lr=0.01):
    if hidden_layers is None:
        hidden_layers = [64]
    
    class MLPClassifier(nn.Module):
        def __init__(self, input_dim, hidden_layers):
            super(MLPClassifier, self).__init__()
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 2))
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    classifier = MLPClassifier(input_dim, hidden_layers)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    X = X.float()
    y = y.long()
    
    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
    
    return classifier

def evaluate_classifier(classifier, X, y):
    classifier.eval()
    with torch.no_grad():
        logits = classifier(X.float())
        preds = logits.argmax(dim=1)
        y_long = y.long()
        acc = accuracy_score(y_long.cpu(), preds.cpu())
        neg_mask = (y_long == 0)
        pos_mask = (y_long == 1)
        fp = ((preds == 1) & neg_mask).sum().float() / neg_mask.sum().float() if neg_mask.sum().item() > 0 else torch.tensor(0.0)
        fn = ((preds == 0) & pos_mask).sum().float() / pos_mask.sum().float() if pos_mask.sum().item() > 0 else torch.tensor(0.0)
    return float(acc), float(fp), float(fn)

def safe_load_state(model, path):
    """
    Load state_dict safely, handling pickle issues
    """
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return False
    
    try:
        # Try to load with weights_only first (safer)
        try:
            state = torch.load(path, map_location='cpu', weights_only=True)
        except:
            # Fallback to non-weights_only
            state = torch.load(path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(state, torch.nn.Module):
            state = state.state_dict()
        elif isinstance(state, dict):
            for key in ['state_dict', 'model_state', 'model']:
                if key in state:
                    state = state[key]
                    break
        
        # Load compatible parameters
        model_state = model.state_dict()
        filtered_state = {}
        
        for key, value in state.items():
            if key in model_state and value.shape == model_state[key].shape:
                filtered_state[key] = value
            else:
                # Try to find matching parameter by pattern
                for model_key in model_state.keys():
                    if (key.endswith('.weight') and model_key.endswith('.weight') and 
                        value.shape == model_state[model_key].shape):
                        filtered_state[model_key] = value
                        break
                    elif (key.endswith('.bias') and model_key.endswith('.bias') and 
                          value.shape == model_state[model_key].shape):
                        filtered_state[model_key] = value
                        break
        
        model.load_state_dict(filtered_state, strict=False)
        print(f"Loaded {len(filtered_state)}/{len(state)} parameters from {path}")
        return True
        
    except Exception as e:
        print(f"Failed to load {path}: {str(e)}")
        return False

def simple_mask_graph_data(args, data):
    """
    Simple masking function for evaluation
    """
    masked_data = copy.deepcopy(data)
    if args.mask_feat_ratio > 0:
        mask = torch.rand_like(masked_data.x) < args.mask_feat_ratio
        masked_data.x[mask] = 0
    return masked_data

def pad_posteriors(posteriors, target_size):
    """Pad or truncate posteriors to target size"""
    padded = []
    for p in posteriors:
        if p.numel() < target_size:
            # Pad with zeros
            pad_size = target_size - p.numel()
            padded.append(torch.cat([p, torch.zeros(pad_size)]))
        elif p.numel() > target_size:
            # Truncate
            padded.append(p[:target_size])
        else:
            padded.append(p)
    return torch.stack(padded)

# ------------------- Plotting Functions -------------------
def plot_verification_accuracy_by_architecture(df, dataset_name):
    """Plot verification accuracy by architecture type"""
    plt.figure(figsize=(10, 6))
    
    # Group by architecture and calculate mean verification accuracy
    arch_results = df.groupby(['model', 'mode'])['ver_acc'].mean().reset_index()
    
    # Create bar plot
    ax = sns.barplot(x='model', y='ver_acc', hue='mode', data=arch_results)
    
    plt.title(f'Verification Accuracy by Architecture on {dataset_name}')
    plt.xlabel('Model Architecture')
    plt.ylabel('Verification Accuracy')
    plt.ylim(0, 1)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1%}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), 
                   textcoords='offset points')
    
    plt.legend(title='Mode')
    plt.tight_layout()
    plt.savefig(f'experiments/results/verification_accuracy_by_architecture.png')
    plt.close()
    print("Plot saved: verification_accuracy_by_architecture.png")

def plot_target_accuracy_vs_mask_ratio(df, dataset_name):
    """Plot target accuracy vs mask ratio"""
    plt.figure(figsize=(10, 6))
    
    # Extract mask ratio from setting and add to dataframe
    df['mask_ratio'] = df['mode'].map({'inductive': 0.05, 'transductive': 0.1})
    
    # Group by architecture and mask ratio
    accuracy_results = df.groupby(['model', 'mask_ratio'])['target_acc'].mean().reset_index()
    
    # Create line plot
    sns.lineplot(x='mask_ratio', y='target_acc', hue='model', 
                 style='model', markers=True, data=accuracy_results)
    
    plt.title(f'Target Accuracy vs Mask Ratio on {dataset_name}')
    plt.xlabel('Mask Ratio')
    plt.ylabel('Target Accuracy')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.legend(title='Architecture')
    plt.tight_layout()
    plt.savefig(f'experiments/results/target_accuracy_vs_mask_ratio.png')
    plt.close()
    print("Plot saved: target_accuracy_vs_mask_ratio.png")

def plot_verification_performance_by_setting(df, dataset_name):
    """Plot verification performance across different settings"""
    plt.figure(figsize=(12, 7))
    
    # Extract configuration from filename
    df['config'] = df['setting'].str.extract(r'(\d+_\d+)')
    
    # Group by architecture and configuration
    config_results = df.groupby(['model', 'config'])['ver_acc'].mean().reset_index()
    
    # Create grouped bar chart
    ax = sns.barplot(x='config', y='ver_acc', hue='model', data=config_results)
    
    plt.title(f'Verification Performance by Model Configuration on {dataset_name}')
    plt.xlabel('Model Configuration (Hidden_Output dimensions)')
    plt.ylabel('Verification Accuracy')
    plt.ylim(0, 1)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.legend(title='Architecture')
    plt.tight_layout()
    plt.savefig(f'experiments/results/verification_performance_by_setting.png')
    plt.close()
    print("Plot saved: verification_performance_by_setting.png")

def plot_false_rates(df, dataset_name):
    """Plot false positive and false negative rates"""
    plt.figure(figsize=(10, 6))
    
    # Calculate mean false rates by architecture
    false_rates = df.groupby('model')[['fpr', 'fnr']].mean().reset_index()
    
    # Convert to long format for plotting
    false_rates_long = false_rates.melt(id_vars='model', 
                                       value_vars=['fpr', 'fnr'],
                                       var_name='rate_type', 
                                       value_name='rate')
    
    # Create stacked bar chart
    ax = sns.barplot(x='model', y='rate', hue='rate_type', data=false_rates_long)
    
    plt.title(f'False Positive/Negative Rates by Architecture on {dataset_name}')
    plt.xlabel('Model Architecture')
    plt.ylabel('Rate')
    plt.ylim(0, 0.5)  # Assuming rates are between 0-0.5
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1%}', 
                   (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                   ha='center', va='center', xytext=(0, 0), 
                   textcoords='offset points', color='white', weight='bold')
    
    plt.legend(title='Rate Type')
    plt.tight_layout()
    plt.savefig(f'experiments/results/false_rates.png')
    plt.close()
    print("Plot saved: false_rates.png")

def plot_comparative_performance(df, dataset_name):
    """Plot comparative performance: original vs verification accuracy"""
    plt.figure(figsize=(12, 6))
    
    # Calculate mean accuracy by architecture
    performance = df.groupby('model')[['target_acc', 'ver_acc']].mean().reset_index()
    
    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Bar width
    x = np.arange(len(performance['model']))
    width = 0.35
    
    # Plot target accuracy
    bars1 = ax1.bar(x - width/2, performance['target_acc'], width, 
                   label='Target Accuracy', alpha=0.7, color='skyblue')
    ax1.set_xlabel('Model Architecture')
    ax1.set_ylabel('Target Accuracy', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Create second y-axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, performance['ver_acc'], width, 
                   label='Verification Accuracy', alpha=0.7, color='salmon')
    ax2.set_ylabel('Verification Accuracy', color='salmon')
    ax2.tick_params(axis='y', labelcolor='salmon')
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    # Set x-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(performance['model'])
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f'Comparative Performance: Target vs Verification Accuracy on {dataset_name}')
    plt.tight_layout()
    plt.savefig(f'experiments/results/comparative_performance.png')
    plt.close()
    print("Plot saved: comparative_performance.png")

def plot_all_results(df, dataset_name):
    """Generate all plots"""
    if df.empty:
        print("No data to plot")
        return
    
    print("\nGenerating all plots...")
    
    # Create all plots
    plot_verification_accuracy_by_architecture(df, dataset_name)
    plot_target_accuracy_vs_mask_ratio(df, dataset_name)
    plot_verification_performance_by_setting(df, dataset_name)
    plot_false_rates(df, dataset_name)
    plot_comparative_performance(df, dataset_name)
    
    print("All plots generated successfully!")

def main():
    global_cfg = load_config('config/global_cfg.yaml')
    dataset_name = global_cfg['dataset']
    architectures = ['gcn', 'gat', 'sage']
    mask_ratios = {'inductive': 0.05, 'transductive': 0.1}

    dataset = Planetoid(root='./data', name=dataset_name)
    data = dataset[0]
    num_classes = dataset.num_classes
    num_features = dataset.num_features

    results = []

    for mode in ['inductive', 'transductive']:
        mask_ratio = mask_ratios[mode]
        masks = {'train': data.train_mask, 'val': data.val_mask, 'test': data.test_mask}

        class MaskArgs:
            mask_node_ratio = 0
            mask_feat_ratio = mask_ratio
            mask_feat_type = 'random_mask'
            mask_method = 'fix'
            mask_node_type = 'overall'
            feature_random_seed = 42
            task_type = mode

        for arch in architectures:
            # Get all target model files for the current architecture
            target_dir = f"temp_results/diff/model_states/{dataset_name}/{mode}/mask_models/random_mask/1.0_{mask_ratio}"
            if not os.path.exists(target_dir):
                print(f"Directory not found: {target_dir}")
                continue
                
            target_files = [f for f in os.listdir(target_dir) if f.startswith(arch) and f.endswith('.pt')]

            for target_file in target_files:
                target_path = os.path.join(target_dir, target_file)
                print(f"\nProcessing: {target_path}")

                # Create flexible model
                target_model = create_flexible_model(arch, num_features, num_classes, target_file)

                if not safe_load_state(target_model, target_path):
                    print(f"Skipping {arch} {mode} {target_file}: Failed to load")
                    continue

                # Use simple masking
                masked_data = simple_mask_graph_data(MaskArgs(), data)
                masked_x = masked_data.x

                target_acc = evaluate_gnn(target_model, masked_x, data.edge_index, masks['test'], data)
                print(f"Target accuracy: {target_acc:.4f}")

                posteriors = []
                labels = []

                # Independent models
                ind_dir = f"temp_results/diff/model_states/{dataset_name}/{mode}/independent_models"
                if os.path.exists(ind_dir):
                    for fname in os.listdir(ind_dir):
                        if arch in fname and fname.endswith('.pt'):
                            file_arch = parse_architecture_from_filename(fname)
                            ind_model = create_flexible_model(file_arch, num_features, num_classes, fname)
                            if safe_load_state(ind_model, os.path.join(ind_dir, fname)):
                                post = get_posteriors(ind_model, masked_x, data.edge_index, masks['train'])
                                if post.numel() > 0:
                                    posteriors.append(post.flatten())
                                    labels.append(0)
                                    print(f"Loaded independent model: {fname}")

                # Surrogate models - skip for now due to import issues
                # hidden_dims = parse_dims_from_filename(target_file)
                # if len(hidden_dims) >= 2:
                #     surr_dir = f"temp_results/diff/model_states/{dataset_name}/{mode}/extraction_models/random_mask/{arch}_{hidden_dims[0]}_{hidden_dims[-1]}/1.0_{mask_ratio}"
                #     if os.path.exists(surr_dir):
                #         for fname in os.listdir(surr_dir):
                #             if fname.endswith('.pt'):
                #                 file_arch = parse_architecture_from_filename(fname)
                #                 surr_model = create_flexible_model(file_arch, num_features, num_classes, fname)
                #                 if safe_load_state(surr_model, os.path.join(surr_dir, fname)):
                #                     post = get_posteriors(surr_model, masked_x, data.edge_index, masks['train'])
                #                     if post.numel() > 0:
                #                         posteriors.append(post.flatten())
                #                         labels.append(1)
                #                         print(f"Loaded surrogate model: {fname}")

                if len(posteriors) < 2:
                    print(f"Skipping {arch} {mode} {target_file}: Insufficient models ({len(posteriors)})")
                    continue

                # Find minimum size and pad all posteriors
                min_size = min(p.numel() for p in posteriors)
                X = pad_posteriors(posteriors, min_size)
                y = torch.tensor(labels, dtype=torch.long)
                
                if X.shape[0] != y.shape[0]:
                    print(f"Shape mismatch: X={X.shape}, y={y.shape}")
                    continue
                    
                try:
                    classifier = train_classifier(X, y, X.shape[1], hidden_layers=[64], epochs=100, lr=0.01)
                    ver_acc, fpr, fnr = evaluate_classifier(classifier, X, y)

                    results.append({
                        'dataset': dataset_name,
                        'mode': mode,
                        'model': arch,
                        'setting': target_file,
                        'target_acc': target_acc,
                        'ver_acc': ver_acc,
                        'fpr': fpr,
                        'fnr': fnr
                    })
                    print(f"Result: {arch}, {mode}, {target_file}, Target Acc: {target_acc:.4f}, Ver Acc: {ver_acc:.4f}")
                except Exception as e:
                    print(f"Error training classifier: {e}")

    # Save results and generate plots
    if results:
        df = pd.DataFrame(results)
        os.makedirs('experiments/results', exist_ok=True)
        df.to_csv('experiments/results/results.csv', index=False)
        print(f"Results saved to experiments/results/results.csv")
        
        # Generate all comprehensive plots
        plot_all_results(df, dataset_name)
    else:
        print("No results to save or plot")

if __name__ == '__main__':
    main()