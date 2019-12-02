import torch
import torch.nn as nn
import torch.optim as optim


class Autoencoder(nn.Module):
    """
    Adapated from https://stackoverflow.com/questions/54411662/lstm-autoencoder-always-returns-the-average-of-the-input-sequence?fbclid=IwAR3f-qQq7_AvvwxKZV_k5jGbJBDNxOkMX4yfrF0DtCo3CnKItZ1lgAF6Oks
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.decoder = nn.LSTM(self.hidden_size, self.input_size, self.num_layers)

    def forward(self, input):
        # Encode
        _, (last_hidden, _) = self.encoder(input)
        # It is way more general that way

        # last_hideen = last_hidden.repeat(input.shape)

        # Decode
        y, _ = self.decoder(last_hidden)
        return torch.squeeze(y)
