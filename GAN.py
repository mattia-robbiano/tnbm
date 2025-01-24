import torch
import torch.nn as nn
import torch.optim as optim
from functions import real_batch, fake_batch_maker, Generator
import quimb as qu
import quimb.tensor as qtn

numberOpenIndex = 2**3
BondDimension = 2
dtype = 'float32'
Psi = qtn.MPS_rand_state(L=numberOpenIndex, bond_dim=2)
OpenIndex = [f'k{i}' for i in range(numberOpenIndex)]
Psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)

# Hyperparameters
input_size = 8
hidden_size = 256
batch_size = 20
num_epochs = 100
Glearning_rate = 0.0002
Dlearning_rate = 0.00001

# Initialize models
#G = Generator(Psi, OpenIndex)
G = Generator(Psi)
D = Discriminator(input_size, hidden_size, 1)

# Loss and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr=Dlearning_rate)
optimizerG = optim.Adam(G.parameters(), lr=Glearning_rate)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(D.parameters(), 1)
torch.nn.utils.clip_grad_norm_(G.parameters(), 1)

# Training loop
print('Starting Training Loop...')
for epoch in range(num_epochs):
    for i in range(30):

        real_images = real_batch(batch_size)
        # Train Discriminator
        D.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        outputs = D(real_images)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        fake_images = fake_batch_maker(G, batch_size)
        outputs = D(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        optimizerD.step()

        # Train Generator
        G.zero_grad()
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizerG.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss_real.item() + d_loss_fake.item()}, g_loss: {g_loss.item()}')
    if epoch % 5 == 0:
        print(f'Generated vector at epoch {epoch}: {fake_images[0]}')

print(f'fake_images: {fake_images[0]}')
