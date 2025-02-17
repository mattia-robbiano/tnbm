import torch
import torch.nn as nn
import torch.optim as optim
from functions import perfect_sampling
import quimb as qu
import quimb.tensor as qtn

numberOpenIndex = 9
BondDimension = 30
Psi = qtn.MPS_rand_state(L=numberOpenIndex, bond_dim=2)

print(perfect_sampling(Psi))