"""
Model Trainer - Trains spatial transformer on fire spread data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pickle
import config
from pathlib import Path

# ============================================================================
# DATASET
# ============================================================================

class FireSpreadDataset(Dataset):
    """Dataset for 5x5 spatial patches"""
    def __init__(self, patches, labels):
        # patches: (N, 175) → reshape to (N, 25, 7) for transformer
        self.patches = torch.FloatTensor(np.array(patches.tolist())).reshape(-1, config.PATCH_SIZE**2, 7)
        self.labels = torch.LongTensor(labels.astype(np.int64))
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class PositionalEncoding(nn.Module):
    """Add positional information to patches"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class SpatialTransformer(nn.Module):
    """Spatial Transformer - predicts fire spread from 5x5 patches"""
    def __init__(self, feature_dim=7, d_model=128, nhead=8, num_layers=4, dropout=0.1, patch_size=5):
        super().__init__()
        
        self.patch_size = patch_size
        
        # Project each cell's 7 features to d_model dimensions
        self.input_proj = nn.Linear(feature_dim, d_model)
        
        # Positional encoding for spatial positions
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder - learns spatial relationships
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: (batch, 25 cells, 7 features)
        
        # Project to d_model
        x = self.input_proj(x)  # (batch, 25, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer learns spatial relationships
        x = self.transformer(x)  # (batch, 25, d_model)
        
        # Use CENTER cell (index 12) for prediction
        center_idx = (self.patch_size ** 2) // 2
        x = x[:, center_idx, :]  # (batch, d_model)
        
        # Classification
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ============================================================================
# TRAINING
# ============================================================================

def train_model():
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    # Load dataset
    print("\n📥 Loading dataset...")
    dataset_file = config.DATA_DIR / "processed" / "training_dataset.pkl"
    df = pd.read_pickle(dataset_file)
    
    patches = df['patch_features'].values
    labels = df['spread'].values.astype(np.int64)
    
    print(f"  ✓ Loaded {len(df)} examples")
    print(f"  ✓ Positive class: {(labels==1).sum()} ({(labels==1).mean()*100:.1f}%)")
    
    # Class weights for imbalanced data
    pos_weight = (labels == 0).sum() / (labels == 1).sum()
    print(f"  ✓ Class weight: {pos_weight:.2f}:1 (negative:positive)")
    
    # Standardize features
    print("\n🔧 Standardizing features...")
    all_features = np.array(patches.tolist()).reshape(-1, 7)
    scaler = StandardScaler()
    all_features_scaled = scaler.fit_transform(all_features)
    patches_scaled = all_features_scaled.reshape(len(patches), -1)
    
    print(f"  ✓ Feature means: {scaler.mean_[:3]}")
    print(f"  ✓ Feature stds: {scaler.scale_[:3]}")
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        patches_scaled, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_dataset = FireSpreadDataset(X_train, y_train)
    val_dataset = FireSpreadDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"\n📊 Data split:")
    print(f"  Training:   {len(train_dataset)} examples")
    print(f"  Validation: {len(val_dataset)} examples")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n💻 Device: {device}")
    
    model = SpatialTransformer(
        feature_dim=7,
        d_model=128,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        patch_size=config.PATCH_SIZE
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"📐 Model parameters: {num_params:,}")
    
    # Loss with class weighting (moderate to balance precision/recall)
    weights = torch.FloatTensor([1.0, pos_weight * 0.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    # Training loop
    print("\n🏋️  Training...")
    print("-" * 60)
    
    best_f1 = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}
    
    num_epochs = 50
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        
        for patches_batch, labels_batch in train_loader:
            patches_batch = patches_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(patches_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for patches_batch, labels_batch in val_loader:
                patches_batch = patches_batch.to(device)
                labels_batch = labels_batch.to(device)
                
                outputs = model(patches_batch)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
        
        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        
        history['train_loss'].append(train_loss)
        history['val_f1'].append(f1)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        
        scheduler.step(f1)
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            
            model_path = config.DATA_DIR / "models" / "trained_model.pth"
            model_path.parent.mkdir(exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'history': history,
                'config': {
                    'patch_size': config.PATCH_SIZE,
                    'd_model': 128,
                    'nhead': 8,
                    'num_layers': 4
                }
            }, model_path)
            
            # Also save scaler separately
            with open(config.DATA_DIR / "models" / "scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 10:
            print(f"\n⏹️  Early stopping at epoch {epoch + 1}")
            break
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    print(f"\n🎯 Best Performance:")
    print(f"  F1 Score:  {best_f1:.3f}")
    print(f"  Precision: {history['val_precision'][-1]:.3f}")
    print(f"  Recall:    {history['val_recall'][-1]:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\n Confusion Matrix:")
    print(f"  True Negatives:  {cm[0,0]:4d}")
    print(f"  False Positives: {cm[0,1]:4d}")
    print(f"  False Negatives: {cm[1,0]:4d} ⚠️")
    print(f"  True Positives:  {cm[1,1]:4d}")
    
    print(f"\n Model saved: data/models/trained_model.pth")
    print("=" * 60)
    
    return model, scaler, history

if __name__ == "__main__":
    train_model()