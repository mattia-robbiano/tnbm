import numpy as np
import matplotlib.pyplot as plt
import quimb as qu
import quimb.tensor as qtn
import torch
import torch.nn as nn

#def perfect_sampling(TN,SamplingIndex):
def perfect_sampling(TN):

    # inizizlize arrays
    # numberSamplingIndex = len(SamplingIndex)
    numberSamplingIndex = 8
    s_hat = torch.zeros((numberSamplingIndex, 2), dtype=torch.float32)
    probability = np.zeros(numberSamplingIndex)

    #reindexing open indexes to be sampled to k_i
    # TN.reindex({SamplingIndex[i]: f'k{i}' for i in range(numberSamplingIndex)})

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
        ps0 = (v0 & reducedTN & v0.reindex({f'k{i}':f'k{i}*'})).contract(all, optimize='auto-hq')
        ps1 = (v1 & reducedTN & v1.reindex({f'k{i}':f'k{i}*'})).contract(all, optimize='auto-hq')

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
    
    # post-processing
    for i in range(numberSamplingIndex):
        if torch.equal(s_hat[i], torch.tensor([0, 1], dtype=torch.float32)):
            s_hat[i] = torch.tensor([0], dtype=torch.float32)
        elif torch.equal(s_hat[i], torch.tensor([1, 0], dtype=torch.float32)):
            s_hat[i] = torch.tensor([1], dtype=torch.float32)
        else:
            print("Error: wrong value")
            raise ValueError
    s_hat = s_hat.T
    x= s_hat[0]
    return x

def norm_fn(TN):
    # reconduct all tensors to closest isometric/unitary to stay in MERA space (used for MERA)
    return TN.isometrize(method="cayley")

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
        return perfect_sampling(psi)

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
    fake_batch = torch.empty((0, 8), dtype=torch.float32)
    for i in range(batch_size):
        fake_sample = generator.forward().unsqueeze(0)
        fake_batch = torch.cat((fake_batch, fake_sample), dim=0)
    return fake_batch

def real_batch(num_vectors):
    tensor1 = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float32)
    tensor2 = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1], dtype=torch.float32)
    tensor_vectors = []
    for _ in range(num_vectors):
        if np.random.rand() < 0.5:
            tensor_vectors.append(tensor1)
        else:
            tensor_vectors.append(tensor2)
    tensor_vectors = torch.stack(tensor_vectors)
    return tensor_vectors
