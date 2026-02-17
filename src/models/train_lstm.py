import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.lstm_model import create_model
from src.utils.logging import log

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "num_posts", "num_pos", "num_neg", "num_neu", "mean_sentiment_score",
    "rsi", "sma_20", "sma_50", "momentum", "volume_ratio"
]
TARGET_COL = "target_direction"

def load_dataset(path: str, fit_scaler: bool = True, scaler_params: dict = None) -> tuple:
    df = pd.read_csv(path)
    
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    
    features = df[available_features].values
    targets = df[TARGET_COL].values
    
    if fit_scaler:
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1
    else:
        if scaler_params is None:
            raise ValueError("scaler_params must be provided when fit_scaler=False")
        mean = scaler_params['mean']
        std = scaler_params['std']
    
    features = (features - mean) / std
    
    features_tensor = torch.tensor(features, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    
    return features_tensor, targets_tensor, mean, std

def create_sequences(
    features: torch.Tensor,
    targets: torch.Tensor,
    seq_len: int,
    step: int = 1,
) -> tuple:
    X = []
    y = []
    for i in range(0, len(features) - seq_len, step):
        X.append(features[i : i + seq_len])
        y.append(targets[i + seq_len])
    X = torch.stack(X)
    y = torch.stack(y)
    return X, y

def train_lstm_model(
    epochs: int = 25,
    seq_len: int = 20,
    lr: float = 1e-3,
    dropout: float = 0.375,
    train_sequence_step: int = 7,
) -> None:
    full_dataset_path = project_root / "data" / "processed" / "datasets" / "full_dataset.csv"
    
    if not full_dataset_path.exists():
        log("Full dataset not found. Building from sentiment and price data...")
        from src.features.dataset_builder import build_full_dataset
        build_full_dataset()
    
    df = pd.read_csv(full_dataset_path, low_memory=False)
    
    total_samples = len(df)
    train_end = int(total_samples * 0.7)
    val_end = int(total_samples * 0.85)
    
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    
    log(f"Raw data split - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    
    train_features = df_train[available_features].values
    train_targets = df_train[TARGET_COL].values
    
    val_features = df_val[available_features].values
    val_targets = df_val[TARGET_COL].values
    
    test_features = df_test[available_features].values
    test_targets = df_test[TARGET_COL].values
    
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std == 0] = 1
    
    train_features = (train_features - mean) / std
    val_features = (val_features - mean) / std
    test_features = (test_features - mean) / std
    
    # Replace NaN/Inf so LSTM doesn't crash
    train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
    val_features = np.nan_to_num(val_features, nan=0.0, posinf=0.0, neginf=0.0)
    test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.long)
    
    val_features_tensor = torch.tensor(val_features, dtype=torch.float32)
    val_targets_tensor = torch.tensor(val_targets, dtype=torch.long)
    
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
    test_targets_tensor = torch.tensor(test_targets, dtype=torch.long)
    
    # Strided sequences for train to reduce overlap/overfitting; full sequences for val/test
    X_train, y_train = create_sequences(
        train_features_tensor, train_targets_tensor, seq_len, step=train_sequence_step
    )

    val_context = train_features_tensor[-seq_len:]
    val_features_with_context = torch.cat([val_context, val_features_tensor], dim=0)
    val_targets_with_context = torch.cat([train_targets_tensor[-seq_len:], val_targets_tensor], dim=0)
    X_val, y_val = create_sequences(val_features_with_context, val_targets_with_context, seq_len)
    
    test_context = val_features_tensor[-seq_len:]
    test_features_with_context = torch.cat([test_context, test_features_tensor], dim=0)
    test_targets_with_context = torch.cat([val_targets_tensor[-seq_len:], test_targets_tensor], dim=0)
    X_test, y_test = create_sequences(test_features_with_context, test_targets_with_context, seq_len)
    
    log(f"Sequence split - Train: {len(X_train)} (step={train_sequence_step}), Val: {len(X_val)}, Test: {len(X_test)}")
    
    train_class_counts = torch.bincount(y_train)
    total_train_samples = len(y_train)
    class_weights = total_train_samples / (len(train_class_counts) * train_class_counts.float())
    
    log(f"Class distribution (train): {train_class_counts.tolist()}")
    log(f"Class weights: {class_weights.tolist()}")
    
    input_dim = X_train.shape[2]
    output_dim = 2
    
    model = create_model(input_dim, output_dim, hidden_dim=128, num_layers=3, dropout=dropout)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    batch_size = 32
    best_val_loss = float('inf')
    patience = 4
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_indices = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_indices = train_indices[i:i + batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / (len(X_train) // batch_size + 1)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                X_batch = X_val[i:i + batch_size]
                y_batch = y_val[i:i + batch_size]
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / (len(X_val) // batch_size + 1)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        log(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            

            model_dir = project_root / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "lstm_volatility.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'mean': mean,
                'std': std,
                'input_dim': input_dim,
                'output_dim': output_dim,
                'hidden_dim': 128,
                'num_layers': 3,
                'dropout': dropout,
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log(f"Early stopping at epoch {epoch + 1}")
                break
    

    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i + batch_size]
            y_batch = y_test[i:i + batch_size]

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            all_predicted.append(predicted)
            all_labels.append(y_batch)

    all_predicted = torch.cat(all_predicted).numpy()
    all_labels = torch.cat(all_labels).numpy()

    avg_test_loss = test_loss / (len(X_test) // batch_size + 1)
    test_accuracy = 100 * correct / total

    # Balanced accuracy (average of per-class recall) — meaningful with imbalanced classes
    n_classes = 2
    recall_sum = 0.0
    for c in range(n_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            recall_sum += (all_predicted[mask] == c).sum().item() / mask.sum().item()
    balanced_accuracy = 100 * (recall_sum / n_classes)

    log(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Balanced Accuracy: {balanced_accuracy:.2f}%")
    log(f"Model saved to {model_path}")
    
    history_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    
    history_path = model_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    log(f"Training history saved to {history_path}")
    
    import json
    test_metrics = {
        'test_loss': float(avg_test_loss),
        'test_accuracy': float(test_accuracy),
        'test_balanced_accuracy': float(balanced_accuracy),
        'num_train_samples': int(len(X_train)),
        'num_val_samples': int(len(X_val)),
        'num_test_samples': int(len(X_test)),
        'final_epoch': int(len(train_losses)),
        'best_val_loss': float(best_val_loss),
        'timestamp': pd.Timestamp.now(tz='UTC').isoformat()
    }
    
    metrics_path = model_dir / "test_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    log(f"Test metrics saved to {metrics_path}")
