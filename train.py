import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from datetime import datetime

def get_optimizer(model_parameters, optimizer_name, learning_rate):
    """Factory function for optimizers"""
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model_parameters, lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model_parameters, lr=learning_rate)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model_parameters, lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def save_training_results(config, training_history, start_time, end_time):
    """Save training results to a timestamped JSON file"""
    if not config.get("results", {}).get("save_results", True):
        return
    
    # Create results folder if it doesn't exist
    results_folder = config.get("results", {}).get("results_folder", "results")
    os.makedirs(results_folder, exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_{timestamp}.json"
    filepath = os.path.join(results_folder, filename)
    
    # Prepare results data
    results_data = {
        "training_info": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "model": config["model"],
            "dataset": config["dataset"],
            "device": config.get("device", "cuda"),
            "activation": config["activation"],
            "training_config": config["training"]
        },
        "training_history": training_history
    }
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Training results saved to: {filepath}")

def train_model(model, train_loader, config):
    # Record start time
    start_time = datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine device
    device_name = config.get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print(f"Training on device: {device}")
    
    # Get precision settings
    precision = config["activation"]["precision"]
    use_amp = precision in ["fp16", "fp8"]
    
    # Setup model and training components
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Use optimizer from config
    optimizer = get_optimizer(
        model.parameters(), 
        config["training"]["optimizer"], 
        config["training"]["learning_rate"]
    )
    
    # Setup mixed precision training
    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    # Get printing interval from config
    print_every_n_epochs = config["training"].get("print_every_n_epochs", 1)
    
    # Training history storage
    training_history = []
    
    # Training loop
    epochs = config["training"]["epochs"]
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Use autocast for mixed precision
            with torch.amp.autocast(device_type=str(device).split(':')[0], enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress every 100 batches (optional, can be removed if too verbose)
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # Store epoch data
        epoch_data = {
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "timestamp": datetime.now().isoformat()
        }
        training_history.append(epoch_data)
        
        # Print epoch summary based on configured interval
        if (epoch + 1) % print_every_n_epochs == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}] Summary - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    # Record end time
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training duration: {duration}")
    
    # Save training results
    save_training_results(config, training_history, start_time, end_time)
    
    print("Training completed!")