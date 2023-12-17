import torch #pytorch
import torch.nn as nn
import pytorch_lightning as pl
from torch.autograd import Variable

class LSTM(nn.Module):
    def __init__(self, n_features, n_classes, n_hidden, n_layers, dropout, device):
        super(LSTM, self).__init__()
        self.n_feature = n_features
        self.n_classes = n_classes
        self.n_hidden =  n_hidden
        self.n_layers = n_layers
        self.dropout = dropout

        self.device = device

        self.lstm = nn.LSTM(input_size=self.n_feature, hidden_size=self.n_hidden,
                            num_layers=self.n_layers, batch_first=True, dropout=dropout)  # lstm
        self.classifier = nn.Linear(n_hidden, n_classes)

        self.relu = nn.ReLU()


    def forward(self, x):
        self.lstm.flatten_parameters()
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x)  # lstm with input, hidden, and internal state
        out = hn[-1]
        out = self.classifier(out)  # Final Output

        return out