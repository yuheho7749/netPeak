import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
  
  df['CurrentEstimate'] = df['BytesAcked'] / df['ElapsedTime'] * 8
  df['DeltaTime'] = df['ElapsedTime'] - df.groupby('TestID')['ElapsedTime'].shift(1).fillna(0)
  df['DeltaBytesSent'] = df['BytesSent'] - df.groupby('TestID')['BytesSent'].shift(1).fillna(0)
  df['DeltaBytesAcked'] = df['BytesAcked'] - df.groupby('TestID')['BytesAcked'].shift(1).fillna(0)
  df['DeltaBytesRetrans'] = df['BytesRetrans'] - df.groupby('TestID')['BytesRetrans'].shift(1).fillna(0)
  
  df.drop(columns=['TestID'], inplace=True)
  labels = df.pop('FinalSpeed')
  
  return df, labels

def train_model(model: ThroughputPredictor, dataset, epochs=100, batch_size=16, learning_rate=0.01):
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
model = ThroughputPredictor(num_features=len(features.columns))
print(features)

X, y = features.values, labels.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_model(model, torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)))

model.eval()
with torch.no_grad():
  preds = model(torch.tensor(scaler.transform(X), dtype=torch.float32)).view(-1)
  features['PredictedSpeed'] = preds.numpy()
  print("Mean prediction:", preds.mean().item())
  print("Std prediction:", preds.std().item())
  features.to_csv('./predictions.csv', index=False)