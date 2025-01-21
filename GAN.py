import numpy as np
import matplotlib.pyplot as plt
import quimb as qu
import quimb.tensor as qtn
from quimb.experimental.merabuilder import MERA
import torch
import torch.nn as nn


# number of sites
n = 2**6

# max bond dimension
D = 2

# use single precision for quick GPU optimization
dtype = 'float32'
mera = qtn.MERA.rand_invar(n, D, dtype=dtype)

from math import cos, sin, pi

fix = {
    f'k{i}': (sin(2 * pi * i / n), cos(2 * pi * i / n))
    for i in range(n)
}

# reduce the 'spring constant' k as well
draw_opts = dict(fix=fix, k=0.01)

#mera.draw(color=['_UNI', '_ISO'], **draw_opts)

def perfect_sampling_ttn(ttn,number_indexes):
    s_hat = np.zeros([number_indexes,2])
    probability = np.zeros(number_indexes)

    # cycle 0: the ttn is connected to the conjugate, except for the ph0 index, we obtain a matrix, multiplied by the basis vectors
    # gives me the probability of extracting the first element. We extract the first element
    # cycle 1: the vector extracted in the previous cycle is connected to the ph0 index of the ttn. The procedure repeats identically

    for i in range(number_indexes):
        step_tensor_network = ttn

        for j in range(i): 
            # for each cycle connect a tensor to an index (in sequence from ph0 to ph14) up to the index before 
            # the one I want to sample
            v = qtn.Tensor(data=torch.tensor(s_hat[j], dtype=torch.float32), inds=[f'k{j}'], tags=[f'v{int(s_hat[j][0])}'])
            step_tensor_network = step_tensor_network & v
            step_tensor_network = step_tensor_network / np.sqrt(probability[j])

        # take the complex conjugate of the new network with the same indices as the first one except for the one I want to sample
        step_tensor_network_full = step_tensor_network.H.reindex({f'k{i}':f'k{i}*'}) & step_tensor_network

        # contraction of the network, I get the probability matrix of extracting the new element
        reduced_tensor_network = step_tensor_network_full.contract(all)

        # calculate the probability of extracting the two elements
        v0 = qtn.Tensor(data=torch.tensor([0, 1], dtype=torch.float32), inds=[f'k{i}'])
        v1 = qtn.Tensor(data=torch.tensor([1, 0], dtype=torch.float32), inds=[f'k{i}'])
        
        ps0 = v0 @ reduced_tensor_network @ v0.reindex({f'k{i}':f'k{i}*'})
        ps1 = v1 @ reduced_tensor_network @ v1.reindex({f'k{i}':f'k{i}*'})

        ##########NORMALIZATION TEST############
        if ps0+ps1<0.999 or ps0+ps1>1.001:
            print("errore al ciclo: ",i)
            print('ps0: ',ps0)
            print('ps1: ',ps1)
        ########################################

        #extracting new element
        r = np.random.uniform(0, 1)
        if r < ps0:
            s_hat[i] = v0.data
            probability[i] = ps0
        else:
            s_hat[i] = v1.data
            probability[i] = ps1
    return s_hat

def norm_fn(mera):
    # parametrize our tensors as isometric/unitary
    return mera.isometrize(method="cayley")

#mera.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))

class Generator(torch.nn.Module):

    def __init__(self, tn):
        super().__init__()
        # extract the raw arrays and a skeleton of the TN
        params, self.skeleton = qtn.pack(tn)
        # n.b. you might want to do extra processing here to e.g. store each
        # parameter as a reshaped matrix (from left_inds -> right_inds), for
        # some optimizers, and for some torch parametrizations
        self.torch_params = torch.nn.ParameterDict({
            # torch requires strings as keys
            str(i): torch.nn.Parameter(initial)
            for i, initial in params.items()
        })

    def forward(self):
        # convert back to original int key format
        params = {int(i): p for i, p in self.torch_params.items()}
        # reconstruct the TN with the new parameters
        psi = qtn.unpack(params, self.skeleton)
        # isometrize and then return the energy
        return perfect_sampling_ttn(norm_fn(psi), 64)
    
class Discriminator(nn.Module):
    def __init__(self, input_size=64):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        data = data.view(data.size(0), -1)  # Flatten the input tensor
        out = self.model(data)
        return out
def real_batch_maker(batch_size, n=8, noise_level=0.1):
    # Define 8x8 matrices for digits 1
    digit_1 = np.array([
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0]
    ], dtype=np.float32)

    matrices = []
    for i in range(batch_size):
        # Add noise to digit 1
        noise = np.random.rand(n, n) < noise_level
        noisy_digit_1 = np.where(noise, 1 - digit_1, digit_1)
        matrices.append(noisy_digit_1.flatten())
    return np.array(matrices).T
def fake_batch_maker(generator, batch_size):
    fake_batch=[]
    for _ in range(batch_size):
        fake_sample = generator.forward()
        # Convert each row ([0,1] or [1,0] states) of fake_sample to a single value: 0 and 1
        fake_column_vector = np.array([0 if np.array_equal(row, [0, 1]) else 1 for row in fake_sample])
        fake_batch.append(fake_column_vector)
    return fake_batch

# setting hyperparameters, since i am using synthetic dataset i am making up the size of the dataset
# for each batch number iteration i will generate a new fake batch
num_epochs = 100
batch_number = 2
batch_size = 5

# initialize models
mera.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float32))
generator = Generator(mera)
discriminator = Discriminator()

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.1)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.1)

# Training loop
for epoch in range(num_epochs):
    for i in range(batch_number):
        
        # Generate real data
        real_data = torch.tensor(np.array(real_batch_maker(batch_size)), dtype=torch.float32)
        #batch_size = real_data.size(0)

        # Generate fake data
        fake_data = torch.tensor(np.array(fake_batch_maker(generator, batch_size)), dtype=torch.float32)

        # Create labels
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Train discriminator on real data
        outputs = discriminator(real_data.T)
        loss_real = criterion(outputs, real_labels)

        # Train discriminator on fake data
        outputs = discriminator(fake_data.detach())
        loss_fake = criterion(outputs, fake_labels)
        
        # Total loss for discriminator
        loss_D = loss_real + loss_fake
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        outputs = discriminator(fake_data)
        loss_G = criterion(outputs, real_labels)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")