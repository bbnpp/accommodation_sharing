import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class AccommodationDataset(Dataset):
    def __init__(self, fp, types='train'):
        self.fp = fp
        self.type = types
        if self.type == 'train':
            with open(f'{fp}/X_train.pkl', 'rb') as f:
                self.X = pickle.load(f)
            with open(f'{fp}/y_train.pkl', 'rb') as f:
                self.y = pickle.load(f)
        else:
            with open(f'{fp}/X_test.pkl', 'rb') as f:
                self.X = pickle.load(f)
            with open(f'{fp}/y_test.pkl', 'rb') as f:
                self.y = pickle.load(f)
        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.float32)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, 9),
            nn.BatchNorm1d(9),
            nn.ReLU(),
            nn.Linear(9, 9),
            nn.BatchNorm1d(9),
            nn.ReLU(),
            nn.Linear(9, 9),
            nn.BatchNorm1d(9),
            nn.ReLU(),
            nn.Linear(9, 9),
            nn.BatchNorm1d(9),
            nn.ReLU(),
            nn.Linear(9, 1),
            nn.ReLU()
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    train = AccommodationDataset(fp='./preprocessed', types='train')
    test = AccommodationDataset(fp='./preprocessed', types='test')
    train_dataloader = DataLoader(dataset=train, batch_size=32, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dataset=test, batch_size=16, shuffle=True, num_workers=0)

    model = NeuralNetwork()
    learning_rate = 1e-5
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    with open('./result/nn_prediction.npy', 'wb') as f:
        np.save(f, model(torch.Tensor(test.X)).detach().numpy())
