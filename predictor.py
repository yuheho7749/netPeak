import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class ThroughputPredictor(nn.Module):
  def __init__(self, num_features, hidden_dim=64):
    super().__init__()
    self.stack = nn.Sequential(
      nn.Linear(num_features, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, 1),
    )

  def forward(self, x):
    return self.stack(x)

def load_dataset(file_path):
  df = pd.read_csv(file_path).dropna()
  df.sort_values(by=['TestID', 'ElapsedTime'], inplace=True)
  
  df['DeltaTime'] = df['ElapsedTime'] - df.groupby('TestID')['ElapsedTime'].shift(1).fillna(0)
  df['DeltaBytesSent'] = df['BytesSent'] - df.groupby('TestID')['BytesSent'].shift(1).fillna(0)
  df['DeltaBytesAcked'] = df['BytesAcked'] - df.groupby('TestID')['BytesAcked'].shift(1).fillna(0)
  df['DeltaBytesRetrans'] = df['BytesRetrans'] - df.groupby('TestID')['BytesRetrans'].shift(1).fillna(0)
  
  df.drop(columns=['TestID'], inplace=True)
  labels = df.pop('FinalSpeed')
  return df, labels

def train_model(model: ThroughputPredictor, dataset, epochs=10, batch_size=32, learning_rate=0.001):
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for features, labels in dataloader:
      optimizer.zero_grad()
      outputs = model(features).squeeze()
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}')
  
  return model

features, labels = load_dataset('./dataset.csv')
print(features)
model = ThroughputPredictor(num_features=len(features.columns))