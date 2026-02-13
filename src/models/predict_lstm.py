import torch
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.lstm_model import TinyLSTM
from src.utils.logging import log

def load_trained_model() -> tuple:
    
    model_path = project_root / "models" / "lstm_volatility.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")
    
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    input_dim = checkpoint.get('input_dim', 10)
    output_dim = checkpoint.get('output_dim', 3)
    mean = checkpoint.get('mean', None)
    std = checkpoint.get('std', None)
    
    model = TinyLSTM(input_dim, hidden_dim=64, num_layers=2, output_dim=output_dim, dropout=0.2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, mean, std, input_dim, output_dim

def predict_sequences(model: TinyLSTM, X: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs

def predict_direction_labels(model: TinyLSTM, X: torch.Tensor) -> np.ndarray:
   
    probs = predict_sequences(model, X)
    predictions = torch.argmax(probs, dim=1)
    return predictions.numpy()

def predict_action(prediction: int) -> str:
    
    action_map = {0: "short", 1: "stay", 2: "buy"}
    return action_map.get(prediction, "stay")