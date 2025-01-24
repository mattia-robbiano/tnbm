import numpy as np
import matplotlib.pyplot as plt
import quimb as qu
import quimb.tensor as qtn
from quimb.experimental.merabuilder import MERA
import torch
import torch.nn as nn
from functions import *

# Making a simple synthetic dataset of 8x8 images
def real_batch_maker(batch_size, n=4, noise_level=0.1):
    digit_1 = torch.tensor([
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 0]
    ], dtype=torch.float32)

    # Add noise
    noise = torch.rand(batch_size, n, n) < noise_level
    noisy_matrices = torch.where(noise, 1 - digit_1, digit_1).view(batch_size, -1)
    return noisy_matrices

# Parameters
numberOpenIndex = 2**4
BondDimension = 2
dtype = 'float32'

# Initialize a random MERA
Psi = qtn.MERA.rand_invar(numberOpenIndex, BondDimension, dtype=dtype)
OpenIndex = [f'k{i}' for i in range(numberOpenIndex)]

# Setting hyperparameters
num_epochs = 100
batch_number = 938
batch_size = 64

# Initialize models
Psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
generator = Generator(Psi, OpenIndex)
discriminator = Discriminator(input_size=numberOpenIndex)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=0.0002)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.00001)

# Lists to store losses
losses_D = []
losses_G = []

# Training loop
print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i in range(batch_number):

        # Create batches
        real_data = real_batch_maker(batch_size)
        fake_data = fake_batch_maker(generator, batch_size)

        # Create labels
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # === Train Discriminator ===
        for _ in range(2):
            discriminator.train()
            outputs_real = discriminator(real_data)
            loss_real = criterion(outputs_real, real_labels)

            outputs_fake = discriminator(fake_data.detach())
            loss_fake = criterion(outputs_fake, fake_labels)

        # Total loss and optimization for discriminator
        loss_D = loss_real + loss_fake
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # === Train Generator ===
        outputs_fake = discriminator(fake_data)
        loss_G = criterion(outputs_fake, real_labels)  # Labels are real for generator training

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Logging epoch progress
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

    # Store losses
    losses_D.append(loss_D.item())
    losses_G.append(loss_G.item())

# Plotting the losses
plt.figure(figsize=(10, 5))
plt.plot(losses_D, label='Discriminator Loss')
plt.plot(losses_G, label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses')
plt.savefig('/media/mattia/DATA_LINUX/documents/tesi/TN-QML/loss_plot.png')
plt.show()
