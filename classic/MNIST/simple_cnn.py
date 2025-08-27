from typing import Sequence, Mapping, Any, Text
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from mnist_fechter import get_mnist_data_loader

class CustomCNN(nn.Module):
    def __init__(
        self,
        conv_layers: Sequence, 
        fc_layers: Sequence,
        loss_function: nn.Module,
        optimizer_class: type = optim.Adam,
        optimizer_params: Mapping[Text, Any] = None,
        model_path: Text = None,
        device: Text = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        # Save configuration
        self.device = device
        self.loss_function = loss_function
        self.to(device) # Move to specified device
        self.model_path = model_path

        # Build network
        self.conv_layers = nn.Sequential(*conv_layers)
        self.fc_layers = nn.Sequential(*fc_layers)

        # Initialize optimizer
        if optimizer_params is None:
            optimizer_params = {'lr': 0.001}
        self.optimizer = optimizer_class(self.parameters(), **optimizer_params)

        # Training history
        self.train_history = {'loss': [], 'accuracy': []}

    def forward(self, x):
        """Forward propagation"""
        # Device transfer is handled in train_step/evaluate_step
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def backward(self, loss: torch.Tensor):
        """Backward propagation"""
        self.optimizer.zero_grad()  # Zero gradients
        loss.backward()             # Backward propagation
        self.optimizer.step()       # Update parameters

    def train_step(self, images, labels):
        """Single training step"""
        self.train()
        
        # Forward pass
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        
        # Backward pass
        self.backward(loss)
        
        # Calculate accuracy
        accuracy = self.get_accuracy(outputs, labels)
        
        return loss.item(), accuracy

    def evaluate_step(self, images, labels):
        """Single evaluation step"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images)
            loss = self.loss_function(outputs, labels)
            accuracy = self.get_accuracy(outputs, labels)
            
        return loss.item(), accuracy

    def fit(self, train_loader, val_loader=None, epochs=10):
        """Train the model"""
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            train_acc = 0.0
            train_batches = 0
            
            print(f"\nEpoch {epoch+1}/{epochs}")
            train_pbar = tqdm(train_loader, desc="Training")
 
            for images, labels in train_pbar:
                loss, acc = self.train_step(images, labels)
                train_loss += loss
                train_acc += acc
                train_batches += 1
                
                # Update progress bar
                train_pbar.set_postfix({
                    'Loss': f'{loss:.4f}',
                    'Acc': f'{acc:.2f}%'
                })
            
            # Calculate averages
            avg_train_loss = train_loss / train_batches
            avg_train_acc = train_acc / train_batches

            # Validation phase
            if val_loader is not None:
                val_loss = 0.0
                val_acc = 0.0
                val_batches = 0
                
                val_pbar = tqdm(val_loader, desc="Validating")
                for images, labels in val_pbar:
                    loss, acc = self.evaluate_step(images, labels)
                    val_loss += loss
                    val_acc += acc
                    val_batches += 1
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss:.4f}',
                        'Acc': f'{acc:.2f}%'
                    })
                
                avg_val_loss = val_loss / val_batches
                avg_val_acc = val_acc / val_batches
                
                print(f"Train - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.2f}%")
                print(f"Val - Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.2f}%")
            else:
                print(f"Train - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.2f}%")
            
            # Save history
            self.train_history['loss'].append(avg_train_loss)
            self.train_history['accuracy'].append(avg_train_acc)
        
        print("Training completed!")

    def predict(self, x):
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = torch.argmax(outputs, dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            
        return predictions, probabilities

    def get_accuracy(self, predictions, labels):
        """Calculate accuracy"""
        pred_labels = torch.argmax(predictions, dim=1)
        correct = (pred_labels == labels).sum().item()
        total = labels.size(0)
        accuracy = (correct / total) * 100
        return accuracy
    
    def model_save(self, filepath: str, save_optimizer: bool = True):
        """Save model"""
        save_dict = {
            'model_state_dict': self.state_dict(),
            'train_history': self.train_history,
            'device': self.device
        }
        
        if save_optimizer:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(save_dict, filepath)
        print(f"Model saved to: {filepath}")
    
    def model_load(self, filepath: str, load_optimizer: bool = True):
        """Load model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Load model parameters
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        
        # Load optimizer parameters
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self):
        """Get model summary"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("Model Summary:")
        print("=" * 50)
        print(f"Device: {self.device}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Loss function: {type(self.loss_function).__name__}")
        print(f"Optimizer: {type(self.optimizer).__name__}")
        print("=" * 50)
        print("Network structure:")
        print(self)    


# Usage example
if __name__ == "__main__":
    # 1. Custom convolutional layers
    custom_conv_layers = [
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten()
    ]
    
    # 2. Custom fully connected layers
    custom_fc_layers = [
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 10)
    ]
    
    # 3. Custom loss function
    custom_loss = nn.CrossEntropyLoss()
    
    # 4. Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CustomCNN(
        conv_layers=custom_conv_layers,
        fc_layers=custom_fc_layers,
        loss_function=custom_loss,
        optimizer_class=optim.Adam,
        optimizer_params={'lr': 0.001, 'weight_decay': 1e-4},
        device=device
    )
    
    # 5. Load data
    train_loader = get_mnist_data_loader(batch_size=64, shuffle=True, is_train=True)
    test_loader = get_mnist_data_loader(batch_size=64, shuffle=False, is_train=False)
    
    # 6. Show model summary
    model.get_model_summary()
    
    # 7. Train model
    model.fit(train_loader, test_loader, epochs=3) # if need to load model, this line is unnecessary
    
    # 8. Save model
    model.model_save('custom_mnist_model.pth')
    model.model_load('custom_mnist_model.pth', load_optimizer=True)

    # 9. Test predictions
    data_iter = iter(test_loader)
    test_images, test_labels = next(data_iter)
    predictions, probabilities = model.predict(test_images[:5])
    
    print(f"\nPrediction results:")
    print(f"True labels: {test_labels[:5].tolist()}")
    print(f"Predicted labels: {predictions.cpu().tolist()}")
    
    print("All functions tested successfully!")
