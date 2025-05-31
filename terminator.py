import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Terminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(Terminator, self).__init__()
        # NOTE: Simple nn
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train(model, device, dataloader, optimizer, criterion, tolerance, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}')

def test(model, device, dataloader, criterion, tolerance):
    model.eval()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for features, labels in dataloader:
            outputs = model(features).squeeze()
            loss += criterion(outputs, labels)
            relative_error = (torch.abs(outputs - labels) / labels) <= tolerance
            correct += (relative_error).sum().item()
            total += labels.numel()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, total, 100. * correct / total))
    return correct, total

def predict(model, device, dataset):
    model.eval()
    with torch.no_grad():
      return model(torch.tensor(dataset, dtype=torch.float32, device=device)).cpu().view(-1).numpy()

def generate_dataloaders(file_path, args, device):
    df = pd.read_csv(file_path).dropna()
    ground_truth = df.pop('FinalSpeed')
    labels = df.pop('StopPredict')
    features = df
    print(features)
    print(labels)
    X, y = features.values, labels.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32, device=device), torch.tensor(y_train, dtype=torch.float32, device=device))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32, device=device), torch.tensor(y_test, dtype=torch.float32, device=device))
    test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    return df, labels, X_train, X_test, y_train, y_test, train_dataset, train_dataloader, test_dataset, test_dataloader

def main():
    parser = argparse.ArgumentParser(description='Terminator')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='disables dGPU training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--speed-tolerance', type=float, default=0.05, metavar='T',
                        help='acceptable tolerance (percent) between predicted and actual download speed (default: 0.05')

    args = parser.parse_args()
    use_gpu = not args.no_gpu and torch.cuda.is_available
    torch.manual_seed(args.seed)
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # NOTE:
    # Input is the same as input to predictor plus the predicted download speed
    # Ground truth is generated with 1 - upper_clamp(percent_error)
    # Output is sigmoid output continue (0) or terminate (1) the speed test

    # Load data and create model
    features, labels, train_dataset, X_train, X_test, y_train, y_test, train_dataloader, test_dataset, test_dataloader = generate_dataloaders('./terminator_dataset.csv', args, device)
    print(len(features.columns))
    model = Terminator(len(features.columns)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Train and test model
    tolerance = args.speed_tolerance
    train(model, device, train_dataloader, optimizer, criterion, tolerance, args.epochs)

    test(model, device, test_dataloader, criterion, tolerance)
    # pred = predict(model, device, test_dataset)


if __name__ == "__main__":
    main()
