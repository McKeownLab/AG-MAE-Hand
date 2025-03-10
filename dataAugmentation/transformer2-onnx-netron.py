import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx  # Import ONNX module
import netron  # Import Netron
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import wandb


BATCH_SIZE      = 32
LEARNING_RATE   = 1e-4
NUM_EPOCHS      = 50
NHEAD           = 4
NUM_LAYERS      = 3
FF_DIM          = 128
DROPOUT         = 0.2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=400):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

# Define the Transformer classifier model
class TransformerClassifier(nn.Module):
    def __init__(self, d_model=48, nhead=NHEAD, num_layers=NUM_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT, seq_length=400):
        super(TransformerClassifier, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, 128)
        self.ln1 = nn.LayerNorm(128)  
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)  
        self.fc3 = nn.Linear(64, 1)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)      
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x  # raw logits

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier().to(device)

# **Function to Export to ONNX**
def export_to_onnx(model, filename="transformer_model.onnx"):
    model.eval()
    dummy_input = torch.randn(1, 400, 48).to(device)  # Example input (batch_size=1, seq_length=400, feature_dim=48)
    torch.onnx.export(
        model, 
        dummy_input, 
        filename, 
        input_names=["input"], 
        output_names=["output"], 
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=14  # Change this from 11 to 13
    )
    print(f"Model exported to {filename}")

    # Open in Netron
    netron.start(filename)


export_to_onnx(model)
