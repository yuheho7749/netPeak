import argparse
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

def main():
    pass

if __name__ == "__main__":
    main()
