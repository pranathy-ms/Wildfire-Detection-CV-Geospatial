"""
Transformer-based wildfire prediction model
Uses attention mechanisms to predict fire confidence levels
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class WildfireDataset(Dataset):
    """PyTorch Dataset for wildfire features"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: (N, F) array of features
            labels: (N,) array of labels
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor (batch_size, seq_len, d_model)
        """
        return x + self.pe[:x.size(1)]


class WildfireTransformer(nn.Module):
    """Transformer model for wildfire confidence prediction"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 3
    ):
        """
        Args:
            input_dim: Number of input features
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            num_classes: Number of output classes (3 for l/n/h confidence)
        """
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: Tensor (batch_size, input_dim)
        Returns:
            Tensor (batch_size, num_classes)
        """
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # Project to d_model
        x = self.input_projection(x)  # (batch_size, 1, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch_size, 1, d_model)
        
        # Take the output of the single token
        x = x.squeeze(1)  # (batch_size, d_model)
        
        # Classification head
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class WildfireModelTrainer:
    """Training pipeline for wildfire transformer model"""
    
    def __init__(
        self,
        model: WildfireTransformer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.scaler = StandardScaler()
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
    def prepare_data(
        self,
        features_df: pd.DataFrame,
        feature_cols: list,
        target_col: str = 'confidence_num',
        test_size: float = 0.2,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders
        
        Args:
            features_df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Target column name
            test_size: Fraction for test set
            batch_size: Batch size for training
            
        Returns:
            train_loader, val_loader
        """
        # Extract features and labels
        X = features_df[feature_cols].values
        y = features_df[target_col].values
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train/test split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Create datasets
        train_dataset = WildfireDataset(X_train, y_train)
        val_dataset = WildfireDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: L2 regularization parameter
        """
        # Setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        logger.info(f"Starting training for {num_epochs} epochs on {self.device}")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Logging
            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
                )
        
        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            features: (N, F) array of features
            
        Returns:
            (N,) array of predicted class labels
        """
        self.model.eval()
        
        # Standardize features
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features_tensor)
            _, predicted = outputs.max(1)
        
        return predicted.cpu().numpy()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'history': self.history
        }, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.history = checkpoint['history']
        logger.info(f"Checkpoint loaded from {path}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load data (assuming you have features_df from data extraction)
    features_df = pd.read_csv('wildfire_features.csv')
    
    # Initialize model
    model = WildfireTransformer(
        input_dim=5,  # frp, u10, v10, elevation, slope
        d_model=128,
        nhead=8,
        num_layers=4,
        num_classes=3
    )
    
    # Initialize trainer
    trainer = WildfireModelTrainer(model)
    
    # Prepare data
    feature_cols = ['frp', 'u10', 'v10', 'elevation', 'slope']
    train_loader, val_loader = trainer.prepare_data(features_df, feature_cols, batch_size=32)
    
    # Train model
    trainer.train(train_loader, val_loader, num_epochs=50, learning_rate=0.001)
    
    # Make predictions
    predictions = trainer.predict(features_df[feature_cols].values)
    features_df['predicted_confidence'] = predictions
    
    # Save results
    features_df.to_csv('wildfire_predictions.csv', index=False)
    print(f"Predictions saved. Validation accuracy: {trainer.history['val_acc'][-1]:.2f}%")
