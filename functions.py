import numpy as np
import matplotlib.pyplot as plt
import quimb as qu
import quimb.tensor as qtn
from quimb.experimental.merabuilder import MERA
import torch
import torch.nn as nn


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
