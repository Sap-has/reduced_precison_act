import torch
import torch.nn as nn
import torch.optim as optim

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

def train_model(model, train_loader, config):
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
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Print epoch summary
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Summary - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    print("Training completed!")