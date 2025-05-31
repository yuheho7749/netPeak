import torch
import torch.nn as nn
import pandas as pd

class ThroughputPredictor(nn.Module):
  def __init__(self, num_features, hidden_dim=64):
    super().__init__()
    self.stack = nn.Sequential(
      nn.Linear(num_features, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, 1),
    )

  def forward(self, x):
    return self.stack(x)
  
  def to(self, device):
    super().to(device)
    self.device = device
    return self
  
  def fit(self, X, y, epochs=100, batch_size=16, learning_rate=0.001):
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32, device=self.device), torch.tensor(y, dtype=torch.float32, device=self.device))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    for epoch in range(epochs):
      self.train()
      total_loss = 0.0
      for features, labels in dataloader:
        optimizer.zero_grad()
        outputs = self(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

      print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}')

    return self
  
  def predict(self, X):
    self.eval()
    with torch.no_grad():
      return self(torch.tensor(X, dtype=torch.float32, device=self.device)).cpu().view(-1).numpy()
    
  def predict_trustee(self, X: pd.DataFrame):
    predictions = self.predict(X.to_numpy())
    return pd.Series(predictions, index=X.index if isinstance(X, pd.DataFrame) else None)

  def fit_predict(self, X, y, epochs=100, batch_size=16, learning_rate=0.01):
    self.fit(X, y, epochs, batch_size, learning_rate)
    return self.predict(X)