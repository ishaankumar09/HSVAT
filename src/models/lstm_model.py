import torch
import torch.nn as nn

class TinyLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.2):
        super(TinyLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        
        last_hidden = self.dropout(last_hidden)
        return self.fc(last_hidden)

def create_model(input_dim: int, output_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2) -> TinyLSTM:
    
    return TinyLSTM(input_dim, hidden_dim, num_layers, output_dim, dropout)
