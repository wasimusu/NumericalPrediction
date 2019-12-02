import torch
import torch.nn as nn
import torch.optim as optim


class LSTM(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(self.input_dim, self.latent_dim, self.num_layers)
        self.dense = nn.Linear(self.latent_dim, 128)
        self.decoder = nn.LSTM(128, self.input_dim, self.num_layers)

    def forward(self, input):
        # Encode
        _, (last_hidden, _) = self.encoder(input)
        # It is way more general that way

        last_hidden = last_hidden.repeat(input.shape)
        last_hidden = self.dense(last_hidden)
        # Decode
        y, _ = self.decoder(last_hidden)
        return torch.squeeze(y)


model = LSTM(input_dim=1, latent_dim=512, num_layers=1)
loss_function = nn.L1Loss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)

y = torch.Tensor([0, 1, 2, 3, 4])
# y = torch.Tensor([0.0, 0.10, 0.200, 0.3000, 0.40000])
x = y.view(len(y), 1, -1)

while True:
    y_pred = model(x)
    optimizer.zero_grad()
    loss = loss_function(y_pred, y)
    loss.backward()
    optimizer.step()
    # print(y_pred)
    print(loss, y_pred)
