import numpy as np
import matplotlib.pyplot as plt
import quimb as qu
import quimb.tensor as qtn
from quimb.experimental.merabuilder import MERA
import torch
import torch.nn as nn


# parameters
numberOpenIndex = 2**6
BondDimension = 2
dtype = 'float32'

Psi = qtn.MERA.rand_invar(numberOpenIndex, BondDimension, dtype=dtype)
OpenIndex = [f'k{i}' for i in range(numberOpenIndex)]

def perfect_sampling(TN,SamplingIndex):
    
    # inizizlize arrays
    numberSamplingIndex = len(SamplingIndex)
    s_hat = torch.zeros((numberSamplingIndex, 2), dtype=torch.float32)
    probability = np.zeros(numberSamplingIndex)

    #reindexing open indexes to be sampled to k_i
    TN.reindex({SamplingIndex[i]: f'k{i}' for i in range(numberSamplingIndex)})

    for i in range(numberSamplingIndex):
        stepTN = TN
        for j in range(i): 
            # connecting original TN with all extracted samples (tensors)
            v = qtn.Tensor(data=s_hat[j].clone().detach().requires_grad_(True), inds=[f'k{j}'], tags=[f'v{int(s_hat[j][0])}'])
            stepTN = (stepTN & v)/np.sqrt(probability[j])

        # computing marginal
        reducedTN = (stepTN.H.reindex({f'k{i}':f'k{i}*'}) & stepTN).contract(all, optimize='auto-hq')
        v0 = qtn.Tensor(data=torch.tensor([0, 1], dtype=torch.float32), inds=[f'k{i}'])
        v1 = qtn.Tensor(data=torch.tensor([1, 0], dtype=torch.float32), inds=[f'k{i}'])
        ps0 = v0 @ reducedTN @ v0.reindex({f'k{i}':f'k{i}*'})
        ps1 = v1 @ reducedTN @ v1.reindex({f'k{i}':f'k{i}*'})

        # check if the probability is normalized
        if ps0+ps1<0.999 or ps0+ps1>1.001:
            print("errore al ciclo: ",i)
            print('ps0: ',ps0)
            print('ps1: ',ps1)
        

        #extracting new element
        r = np.random.uniform(0, 1)
        if r < ps0:
            s_hat[i] = v0.data
            probability[i] = ps0
        else:
            s_hat[i] = v1.data
            probability[i] = ps1
    return s_hat

def norm_fn(TN):
    # reconduct all tensors to closest isometric/unitary to stay in MERA space
    return TN.isometrize(method="cayley")

class Generator(torch.nn.Module):
    def __init__(self, TN, SamplingIndex):
        super().__init__()
        # Extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(TN)
        self.torch_params = torch.nn.ParameterDict({
            str(i): torch.nn.Parameter(initial.clone().detach().requires_grad_(True))
            for i, initial in params.items()
        })
        self.SamplingIndex = SamplingIndex

    def forward(self):
        params = {int(i): p for i, p in self.torch_params.items()}
        # Reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        return perfect_sampling(norm_fn(psi), self.SamplingIndex)


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        data = data.view(data.size(0), -1)
        out = self.model(data)
        return out


def real_batch_maker(batch_size, n=8, noise_level=0.1):
    digit_1 = torch.tensor([
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0]
    ], dtype=torch.float32)

    # Add noise
    noise = torch.rand(batch_size, n, n) < noise_level
    noisy_matrices = torch.where(noise, 1 - digit_1, digit_1).view(batch_size, -1)
    return noisy_matrices


def fake_batch_maker(generator, batch_size):
    fake_batch = []
    for _ in range(batch_size):
        fake_sample = generator.forward()
        fake_column_vector = torch.tensor(
            [0 if torch.equal(row, torch.tensor([0, 1])) else 1 for row in fake_sample],
            dtype=torch.float32
        )
        fake_batch.append(fake_column_vector)
    return torch.stack(fake_batch)


# Setting hyperparameters
num_epochs = 100
batch_number = 2
batch_size = 5

# Initialize models
Psi.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
generator = Generator(Psi, OpenIndex)
discriminator = Discriminator(input_size=numberOpenIndex)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00001)

print("Starting Training Loop...")
for epoch in range(num_epochs):
    for i in range(batch_number):
        # Generate real data
        real_data = real_batch_maker(batch_size)

        # Generate fake data
        fake_data = fake_batch_maker(generator, batch_size)

        # Create labels
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # === Train Discriminator ===
        discriminator.train()  # Ensure batchnorm/dropout behaves correctly
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
