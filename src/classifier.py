import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

try:
    from src.utils import mk_dir
except ModuleNotFoundError:
    from utils import mk_dir
import pickle
import numpy as np

from enum import Enum

try:
    import seaborn as sns
    plt.style.use('seaborn-paper')
except ImportError:
    pass

class weight_norm(Enum):
    NONE = 0
    L1 = 1
    L2 = 2

class Classifier():
    '''Trains a network on image classification tasks.

   Args:
       network: The network to be trained.
       train_loader: A pytorch DataLoader for the training data.
       test_loader: A pytorch DataLoader for the test data.
       learning_rate: The learning rate for the network training.
       optimizer: "adam" or "sgd" for ADAM or SGD optimizer.
       loss: "nll" or "mse" for negative log-likelihood or mean-squared error loss.
       weight_range: A tuple of limits (lower, upper) to clip the weights in each linear layer to.
       weight_normalisation: A weight_norm enum for normalising the weights during training.
       init_weight_mean: Initial mean of the weights.
       init_weight_std: Initial standard deviation of linear layer weights.
       init_conv_weight_std: Initial standard deviation of convolutional layer weights.
       n_epochs: How many epochs to train for.
       n_test_per_epoch: How many times to test the network performance during a single epoch.
       log_interval: How often to log the loss during training.
       save_path: Where to save the network and training information.
   '''

    def __init__(self,
                 network,
                 train_loader,
                 test_loader,

                 learning_rate=0.001,
                 optimizer="adam",
                 loss = "nll",
                 weight_range=(0,1), # None denotes no limit.
                 weight_normalisation=weight_norm.NONE,

                 init_weight_mean=0.0,
                 init_weight_std=0.01,

                 init_conv_weight_std=0.01,

                 n_epochs=10,
                 n_test_per_epoch=0,
                 log_interval=25,

                 save_path="classifier"
                 ):

        self.network = network
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.learning_rate = learning_rate
        self.optimizer = None
        if callable(optimizer):
            self.optimizer = optimizer(self.network.parameters(), lr=self.learning_rate)
        elif type(optimizer) is str:
            if optimizer.lower() == "adam":
                self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
            elif optimizer.lower() == "sgd":
                self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)
        if self.optimizer is None:
            raise NotImplementedError("Unrecognised optimizer :", optimizer)

        if callable(loss):
            self.loss = loss
        elif loss == "nll":
            self.loss = F.nll_loss
        elif loss == "mse":
            self.loss = lambda probs, target, *args, **kwargs: F.mse_loss(probs, torch.zeros(probs.shape, device=target.device).scatter_(1, target.unsqueeze(-1), 1).to(target.device), *args, **kwargs)
        else:
            raise Exception()

        self.weight_range = weight_range

        if self.weight_range is not None:

            if len(self.weight_range)!=2:
                raise Exception("weight range must be of length 2.")

            self.clamp_weight_args = {}
            if self.weight_range[0] is not None:
                self.clamp_weight_args['min']=self.weight_range[0]
            if self.weight_range[1] is not None:
                self.clamp_weight_args['max']=self.weight_range[1]

        if (init_weight_mean is not None) and (init_weight_std is not None):
            def init_weights(m):
                # if type(m) in [torch.nn.Linear, torch.nn.Conv2d]:
                if type(m) in [torch.nn.Linear]:
                    print("Setting weights for", m)
                    m.weight.normal_(init_weight_mean, init_weight_std)
                    m.weight *= (2*torch.randint_like(m.weight,0,2)-1)
                    if self.weight_range is not None:
                        m.weight.clamp_(**self.clamp_weight_args)
                elif type(m) in [torch.nn.Conv2d]:
                    print("Setting weights for", m)
                    m.weight.normal_(init_weight_mean, init_conv_weight_std)
                    if self.weight_range is not None:
                        m.weight.clamp_(**self.clamp_weight_args)
            with torch.no_grad():
                self.network.apply(init_weights)

        self.weight_normalisation = weight_normalisation

        self.n_epochs = n_epochs
        self.n_test_per_epoch = n_test_per_epoch
        self.log_interval = log_interval

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.network.to(self.device)

        print("Prepared classifier with network:\n\n", self.network)

        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_correct = []

        self.save_path = save_path
        mk_dir(self.save_path)
        self.network_save_path = os.path.join(self.save_path, "network.pth")
        self.scores_save_path = os.path.join(self.save_path, "scores.pkl")
        self.loss_save_path = os.path.join(self.save_path, "loss.pkl")

    def train(self):

        n_test_in_epoch = max(self.n_test_per_epoch - 1,0)

        for i_epoch in range(1, self.n_epochs + 1):
            self.train_epoch(i_epoch, n_test_in_epoch)
            self.test(i_epoch)

        for data, f, lab in zip([self.train_losses, self.test_correct],
                                [self.loss_save_path, self.scores_save_path],
                                ["Losses", "Scores"]):
            with open(f, 'wb+') as output:
                pickle.dump(np.array(data), output, pickle.HIGHEST_PROTOCOL)
                print('{} saved to {}'.format(lab, f))

        return np.array(self.train_losses), np.array(self.test_correct)

    def train_epoch(self, i_epoch, n_test_in_epoch=0):
        len_loader = len(self.train_loader)
        len_loader_dataset = len(self.train_loader.dataset)

        if n_test_in_epoch>=1:
            test_interval = int(len_loader / n_test_in_epoch)
        else:
            test_interval = 2*len_loader

        self.network.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            probs = self.network(data.to(self.device)).squeeze(1)
            loss = self.loss(probs, target.to(self.device))
            loss.backward()
            self.optimizer.step()

            if self.weight_range is not None:
                for p in self.network.parameters():
                    p.data.clamp_(**self.clamp_weight_args)

            if self.weight_normalisation != weight_norm.NONE:
                def norm_weights(m):
                    if type(m) is torch.nn.Linear:
                        if self.weight_normalisation == weight_norm.L1:
                            norm = m.weight.abs().sum().item()
                            if norm > 1:
                                m.weight.div_(norm)
                        elif self.weight_normalisation == weight_norm.L2:
                            norm = torch.sqrt(torch.pow(m.weight.abs(), 2).sum()).item()
                            if norm > 1:
                                m.weight.div_(norm)

                with torch.no_grad():
                    self.network.apply(norm_weights)

            if batch_idx>0 and (batch_idx % test_interval == 0):
                self.test()

            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i_epoch, batch_idx * len(data), len_loader_dataset,
                             100. * batch_idx / len_loader, loss.item()))
                self.train_losses.append([(i_epoch - 1) + batch_idx/len_loader, loss.item()])
                self.train_counter.append(
                    (batch_idx * 64) + ((i_epoch - 1) * len_loader_dataset))

    def test(self, i_epoch):

        test_loss, correct = self.validate(self.test_loader)

        if all([correct > score for _,score in self.test_correct]):
            self.save()

        self.test_losses.append([i_epoch, test_loss])
        self.test_correct.append([i_epoch, correct])

        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

    def validate(self, data_loader):
        self.network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                probs = self.network(data.to(self.device)).squeeze(1).to('cpu')
                test_loss += self.loss(probs, target, reduction='sum').item()
                pred = probs.max(-1)[-1]
                correct += pred.eq(target).sum().item()
        test_loss /= len(self.test_loader.dataset)

        return test_loss, correct

    def save(self):
        torch.save(self.network.state_dict(), self.network_save_path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))