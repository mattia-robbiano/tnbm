import numpy as np
import matplotlib.pyplot as plt
import quimb as qu
import quimb.tensor as qtn
from quimb.experimental.merabuilder import MERA
import torch
import torch.nn as nn
from functions import *
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def main():

    # Load the digits dataset
    digits = datasets.load_digits()
    data   = digits.data
    target = digits.target

    # Print the shape of the data
    print("Data shape:", data.shape)
    print("Target shape:", target.shape)

    # split in train and test
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create TensorDataset and DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize tensor network
    numberOpenIndex = 2**6
    BondDimension = 2
    dtype = 'float32'
    Psi = qtn.MPS_rand_state(L=numberOpenIndex, bond_dim=2)
    OpenIndex = [f'k{i}' for i in range(numberOpenIndex)]

    # Setting hyperparameters
    num_epochs = 100

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
        for batch_idx, (data, target) in enumerate(train_loader):

            batch_size = data.size(0)

            # Create batches
            real_data = data
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

if __name__ == '__main__':
    main()
