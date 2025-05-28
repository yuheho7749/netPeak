import argparse
import torch
import torch.nn as nn
import torch.optim as optim

class Terminator(nn.Module):
    def __init__(self, input_dim):
        super(Terminator, self).__init__()
        # TEMP: Testing stub
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def train(model, device, dataset, optimizer, epoch, batch_size, learning_rate):
    model.train()
    # TODO:


def test(model, device, dataset):
    model.eval()
    test_loss = 0
    correct = 0
    # TODO:

def predict(model):
    model.eval()
    # TODO:

def load_data():
    # TODO:
    pass

def main():
    parser = argparse.ArgumentParser(description='Terminator')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-gpu', action='store_true', default=False,
                        help='disables dGPU training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--speed-tolerance', type=float, default=0.05, metavar='T',
                        help='acceptable tolerance between predicted and actual download speed (default: 0.05')

    args = parser.parse_args()
    use_gpu = not args.no_gpu and torch.cuda.is_available
    torch.manual_seed(args.seed)
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # NOTE:
    # Input is the estimated download speed and duration of speed test (aka cost of speed test)
    # Output is binary output 0 or 1 to continue or terminate the speed test
    # Ground truth: abs(estimated - actual) <= tolerance, the tolerance is set by the human for now

    # TODO: Load data and create model
    input_dim = 42 # TEMP: STUB
    model = Terminator(input_dim).to(device)



    # TODO: Train and test model


if __name__ == "__main__":
    main()
