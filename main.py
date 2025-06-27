import yaml
from train import train_model
from models.simple_cnn import SimpleCNN
from models.better_cnn import BetterCNN
from datasets.cifar10 import load_dataset
from datasets.cifar100 import load_dataset as load_cifar100
from activations.custom_activations import (
    custom_relu, custom_sigmoid, custom_tanh, custom_softmax,
    custom_leaky_relu, custom_swish, custom_gelu, custom_softplus
)

def get_activation_function(activation_type):
    """Map activation type string to actual function"""
    activation_map = {
        'custom_relu': custom_relu,
        'custom_sigmoid': custom_sigmoid,
        'custom_tanh': custom_tanh,
        'custom_softmax': custom_softmax,
        'custom_leaky_relu': custom_leaky_relu,
        'custom_swish': custom_swish,
        'custom_gelu': custom_gelu,
        'custom_softplus': custom_softplus,
    }
    
    if activation_type not in activation_map:
        raise ValueError(f"Unknown activation type: {activation_type}")
    
    return activation_map[activation_type]

def get_model(model_name, activation_fn, precision):
    """Factory function to create models based on config"""
    model_map = {
        'simple_cnn': SimpleCNN,
        'better_cnn': BetterCNN,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model_map[model_name](activation_fn, precision)

def get_dataset_loader(dataset_name, batch_size):
    """Load dataset based on config"""
    if dataset_name == 'cifar10':
        return load_dataset(batch_size)
    elif dataset_name == 'cifar100':
        return load_cifar100(batch_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def main():
    # Load configuration
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    # Get activation function from config
    activation_fn = get_activation_function(cfg["activation"]["type"])
    
    # Load dataset from config
    train_loader, test_loader = get_dataset_loader(
        cfg["dataset"], 
        cfg["training"]["batch_size"]
    )
    
    # Create model from config
    model = get_model(
        cfg["model"], 
        activation_fn, 
        cfg["activation"]["precision"]
    )
    
    print(f"Using model: {cfg['model']}")
    print(f"Using dataset: {cfg['dataset']}")
    print(f"Using activation: {cfg['activation']['type']} with precision: {cfg['activation']['precision']}")
    
    # Train the model
    train_model(model, train_loader, cfg)

if __name__ == "__main__":
    main()