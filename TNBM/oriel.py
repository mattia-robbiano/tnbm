import qulacs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from time import time
import cma

def run():

    # region Parameters
    nbr = 213
    n_shots = 1000
    loss_function = 'MMD'
    batch = 1
    #batch = int(10000/n_shots)
    rep = 10
    lr0 = 0.1
    epochs = 1000
    depths = [1,4,16]
    qubits = 16
    genetic = False
    alpha = 0.01
    # alpha = 0.5
    log_transform = False
    log_train = True
    exact = True
    sigmas = [1] + list(np.array([0.1,0.25,0.5,1])*qubits)
    sigmas = list(np.array([0.01,0.1,0.25,0.5,1,10])*qubits)
    sigmas = [1]
    connectivity = 'linear'
    real_amp = False
    ZZ = True
    grad_batch = 5

    if loss_function =='MMD':
        grad_loss_function = grad_MMD
    elif loss_function == 'KLD':
        grad_loss_function = grad_KLD
    elif loss_function =='LFD':
        local = True
        exact = True
    else:
        print('wrong loss function')

    if n_shots == 0:
        grad_batch = 1
    
    # endregion Parameters

    # region Data and initialization
    values_bits = [format(i, "0" + str(qubits) + "b")[::-1] for i in range(2**qubits)]
    values  = np.array([tuple(int(b) for b in bs) for bs in [format(i, "0" + str(qubits) + "b")[::-1] for i in range(2**qubits)]])

    data = select_energy(data_ECAL['ECAL'],data_ECAL['target'],energy = [0,1000])
    size_data = np.shape(data)[0]
    data_ecal = cut_data(qubits,data)
    data_thres = apply_threshold(data_ecal,alpha)
    target = data_thres.reshape(size_data,qubits)

    target_prob = exact_data_ecal(target[:])
    batch_size = int(size_data/batch)
    np.save('results/p_target{}.npy'.format(qubits),target_prob)

    if log_transform:
        target_prob, norm =  transform_p(target_prob)
        target = sample_from_p(qubits, size_data, target_prob)

    signal = sigmas
    projectors  = None
    kern_matrices = None
    if loss_function =='MMD' and qubits>12:
        exact = False
        if n_shots==0:
            print('we need finit shots')
        batch = 10
        n_shots = 1000

    if loss_function == 'MMD' and exact:
        kern_matrices = compute_kernel_matrix(qubits,sigmas)
        signal = kern_matrices
    if loss_function == 'LFD':
        projectors = get_projectors(qubits)

    if genetic:
        epochs = 100000

    save_samples = np.zeros((rep,len(depths),2**qubits))
    TV = np.zeros((rep,len(depths),epochs))
    if genetic:
        epochs = 1

    file = open('results/configuration.txt','a')
    file.write('run :'+str(nbr)+'\n')
    file.write('exact: ' +str(exact)+'\n')
    file.write('batch: ' +str(batch)+'\n')
    file.write('shots: ' +str(n_shots)+'\n')
    file.write('alpha: ' +str(alpha)+'\n')
    file.write('loss: ' +str(loss_function)+'\n')
    file.write('depth: ' +str(depths)+'\n')
    file.write('sigmas: '+str(sigmas)+'\n')
    file.write('qubits: ' +str(qubits)+'\n')
    file.write('genetic: ' +str(genetic)+'\n')
    file.write(100*'-'+'\n')
    file.close()
    print(nbr,qubits, loss_function, n_shots)
    if loss_function == 'KLD' or loss_function =='MMD':
        projectors = None

    # endregion Data


########### TRAINING  ############

    for d_,depth in enumerate(depths):
        ansatz = lambda s,p : ansatz_qc(s, qubits, depth , p, connectivity, real_amp, ZZ)[0]
        ansatz_dagger = lambda s,p : ansatz_qc_dagger(s, qubits, depth , p, connectivity, real_amp, ZZ)[0]

        #parameter initilisation ####
        parameters = np.random.normal(0, 1, size = 10000)
        state = qulacs.QuantumState(qubits)

        state, n_param = ansatz_qc(state.copy(), qubits, depth , parameters,
                        connectivity = connectivity, real = real_amp, ZZ = ZZ)

        get_TV = lambda p: compute_TV(ansatz, p, qubits,  target_prob.copy(), values, projectors)

        for r in range(rep):
            optimizer = ADAM(n_param, lr = lr0)
            seed = 12+42 + 53*(1+r)
            np.random.seed(seed)
            parameters = np.random.normal(0, np.pi, size  = n_param)
            parameters[1:] = 0

            best_tv = 1

            for ep in range(epochs):

                lr = lr0*np.exp(-0.005*ep)
                lr = max(10**-6,lr)
                optimizer._lr = lr

                np.random.shuffle(target)
                target_train = target[:batch_size,...]

                if exact:
                    if batch>1:

                        target_train = exact_data_ecal(target_train)
                        if log_transform:
                            target_train, norm_ =  transform_p(target_train)
                    else:
                        target_train = target_prob



                # TV
                tv, samples_full = get_TV(parameters)
                print(tv)
                TV[r,d_,ep] = tv

                if tv<best_tv:
                    best_tv = tv
                    save_samples[r,d_,:] = samples_full


                #early stopping
                if ep>100 and False:
                    if abs(TV[r,d_,ep]-TV[r,d_,ep-10])<10**-10:

                        TV[r,d_,ep:] = tv
                        break

                #gradients
                median_grad = []

                global intermediary_tv
                intermediary_tv = []
                def loss(x):

                    tv, samples = get_TV(x)
                    intermediary_tv.append(tv)
                    if tv == min(intermediary_tv):
                        save_samples[r,d_,:] = samples

                    if loss_function =='MMD':
                        state_raw = qulacs.QuantumState(qubits)
                        state_raw = ansatz(state_raw, x)
                        samples = compute_samples(state_raw, exact = exact, n_shots=n_shots,
                                            n_qubits=qubits, projectors=projectors,values=values)
                        loss =  MMD(samples,target_train,signal,exact)
                        loss = np.mean(loss)
                    elif loss_function == 'KLD':
                        state_raw = qulacs.QuantumState(qubits)
                        state_raw = ansatz(state_raw, x)
                        samples = compute_samples(state_raw, exact=True, n_shots=n_shots,
                                            n_qubits=qubits, projectors=projectors,values=values)
                        loss = KLD(samples, target_train)
                        loss = np.mean(loss)
                    else:
                        state = qulacs.QuantumState(qubits)
                        wf = np.array(np.sum(np.array([np.sqrt(target_train[x])*projectors[x]
                                for x in range(2**qubits)]),axis = 0))

                        state.load(wf)
                        state = ansatz_dagger(state.copy(),x)

                        loss = compute_samples_fidelity(state.copy(), n_shots,qubits, values, local)

                    return float(loss)


                if genetic:
                    intermediary_tv = []
                    es = cma.CMAEvolutionStrategy(list(parameters), 1)
                    es.optimize(loss)

                    intermediary_tv = intermediary_tv[::10]
                    it = len(intermediary_tv)
                    if it<len(TV[r,d_,:]):
                        TV[r,d_,:it] = intermediary_tv
                        TV[r,d_,it:] = intermediary_tv[-1]
                    else:
                        a = intermediary_tv[::int(it/len(TV[0,0,:]))]
                        TV[r,d_,:len(a)] = a
                    continue

                for _ in range(grad_batch):
                    if loss_function =='LFD':
                        gradients = compute_gradients_fidelity(ansatz_dagger,parameters,target_train,
                                                               qubits, n_shots, values, local,projectors)
                    else:
                        gradients = compute_gradient(ansatz, parameters, target_train, qubits, n_shots,
                                 grad_loss_function, signal = signal, exact = exact, values = values)

                    median_grad.append(gradients)

                temp_grad = np.mean(np.mean(np.array(median_grad), axis=0), axis=-1).reshape(-1)

                #clipping

                threshold = 0.1
                # temp_grad[temp_grad>threshold]  =  threshold
                # temp_grad[temp_grad<-threshold] = -threshold
#                 parameters = parameters - lr*temp_grad

                parameters = optimizer.update(parameters,temp_grad)



#             np.save('results/loss_{}.npy'.format(run),loss)


            np.save('results/TV_{}.npy'.format(nbr),TV)
            np.save('results/prob_{}.npy'.format(nbr),save_samples)


# region functions

# region DATA  
import h5py
data_ECAL =h5py.File('Electron2D_data.h5')

def prime_factors(n):
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors

def get_pixel(n):
    factor = prime_factors(n)

    if len(factor)==1:
        return [1,factor[0]]
    elif len(factor) ==2:
        return factor
    elif len(factor) ==3:
        return [factor[0]*factor[1],factor[2]]

    elif len(factor) ==4:
        return [factor[0]*factor[3],factor[2]*factor[1]]

    else:
        print('factor too large')

def select_energy(data,target,energy = [125,175]):

    new_data = []
    for i in range(np.shape(data)[0]):
        if target[i]>=energy[0] and target[i]<=energy[1]:
            new_data.append(data[i,...])
    return np.array(new_data)

def get_indices(pi):
    indices = []
    it = np.floor(25/pi)
    for i in range(pi):
        indices.append(range(int(it*i),int(it*(i+1))))
    return indices

def cut_data(n,data_full):
    pixel = get_pixel(n)
    indices_x = np.array(get_indices(pixel[0]))
    indices_y = np.array(get_indices(pixel[1]))
    size = np.shape(data_full)[0]

    data = np.zeros((size,pixel[0],pixel[1]))
    for i in range(pixel[0]):
         for j in range(pixel[1]):

            temp = np.sum(data_full[:,0,indices_x[i],:],axis = -2).reshape(size,-1)
            data[:,i,j] = np.sum(temp[:,indices_y[j]],axis=-1)
    return data

def apply_threshold(data,alpha):
    mean = np.mean(data.reshape(-1))

    threshold_indices = data < alpha*mean
    new_data = np.ones_like(data)
    new_data[threshold_indices] = 0

    return new_data

def order(data):
    size = np.shape(data)[0]
    new_data = np.zeros_like(data).reshape(size,-1)
    even_line = True
    counter = 0
    for i in range(np.shape(data)[1]):
        for j in range(np.shape(data)[2]):
            if i%2==0:
                k=j
            else:
                k=-1-j
            new_data[:,counter] = data[:,i,k]

            counter +=1
    return new_data

def count_unique(data):
    integer_data = []
    for d in data:
        d = d.reshape(-1)

        b = np.sum([2**i*d[i] for i in range(len(d))])
        integer_data.append(b)
#     print(integer_data)
    return integer_data, len(np.unique(integer_data))

def exact_data_ecal(data):
    N = np.shape(data)[1]
    p = np.zeros((2**N))
    for d in data:
        b = int(np.sum([d[i]*2**i for i in range(len(d))]))
        p[b]+=1
    p = p/np.shape(data)[0]
    return p

def transform_p(prob, inverse = False, eps = 4, normalization = None):
    if inverse:
        prob *= normalization
        log_prob = 10**(-prob)-10**-eps
        log_prob = log_prob/np.sum(log_prob)

    else:
        log_prob = -np.log10(prob+10**-eps)
        normalization = np.sum(log_prob)
        log_prob/=normalization

    return np.array(log_prob), normalization

def sample_from_p(n_qubits,size,p):

    s = np.random.choice(range(2**n_qubits),size=size,replace = True,p = p)
    values_bits = [format(i, "0" + str(n_qubits) + "b")[::-1] for i in range(2**n_qubits)]
    values = np.array([tuple(int(b) for b in bs) for bs in [format(i, "0" + str(n_qubits) + "b")[::-1] for i in range(2**n_qubits)]])
    samples = values[s]

    return samples

# endregion DATA

#region ANSATZ

def ansatz_qc(state, n_qubits, depth , param, connectivity = 'linear', real = 'True', ZZ = False):
    counter = 0

    for d in range(depth+1):
        for i in range(n_qubits):
            if real:

                qulacs.gate.RY(i,param[counter]).update_quantum_state(state)
                counter +=1
            else:
                qulacs.gate.RZ(i,param[counter]).update_quantum_state(state)
                qulacs.gate.RY(i,param[counter+1]).update_quantum_state(state)
                qulacs.gate.RZ(i,param[counter+2]).update_quantum_state(state)
                counter +=3

        if d<depth:
            if connectivity =='full':
                for i in range(n_qubits-1):
                    for j in range(i+1,n_qubits):
                        qulacs.gate.CNOT(j,i).update_quantum_state(state)
                        if ZZ:
                            qulacs.gate.RY((j)%n_qubits,param[counter]).update_quantum_state(state)
                            qulacs.gate.CNOT(j,i).update_quantum_state(state)
                            counter +=1
            else:
                for i in range(n_qubits):
                    qulacs.gate.CNOT((i+1)%n_qubits,i).update_quantum_state(state)
                    if ZZ:
                        qulacs.gate.RY((i+1)%n_qubits,param[counter]).update_quantum_state(state)
                        qulacs.gate.CNOT((i+1)%n_qubits,i).update_quantum_state(state)
                        counter +=1

    return state, counter

def ansatz_qc_dagger(state, n_qubits, depth , param, connectivity = 'linear', real = 'True', ZZ = False):
    counter = len(param)-1
    param = - param.copy()


    for d in range(depth+1):
        for i in range(n_qubits-1,-1,-1):
            if real:

                qulacs.gate.RY(i,param[counter]).update_quantum_state(state)
                counter -=1
            else:
                qulacs.gate.RZ(i,param[counter]).update_quantum_state(state)
                qulacs.gate.RY(i,param[counter-1]).update_quantum_state(state)
                qulacs.gate.RZ(i,param[counter-2]).update_quantum_state(state)
                counter -=3

        if d<depth:
            pair = []
            if connectivity =='full':
                for i in range(n_qubits-1):
                    for j in range(i+1,n_qubits):
                        pair.append((i,j))
                for p in pair[::-1]:
                    qulacs.gate.CNOT(p[1]%n_qubits,p[0]%n_qubits).update_quantum_state(state)
                    if ZZ:
                        qulacs.gate.RY((p[1])%n_qubits,param[counter]).update_quantum_state(state)
                        qulacs.gate.CNOT(p[1]%n_qubits,p[0]%n_qubits).update_quantum_state(state)
                        counter -=1
            else:
                for i in range(n_qubits):
                    pair.append((i,i+1))

                for p in pair[::-1]:
                    qulacs.gate.CNOT(p[1]%n_qubits,p[0]).update_quantum_state(state)
                    if ZZ:
                        qulacs.gate.RY(p[1]%n_qubits,param[counter]).update_quantum_state(state)
                        qulacs.gate.CNOT(p[1]%n_qubits,p[0]).update_quantum_state(state)
                        counter -=1


    return state, counter

# endregion ANSATZ

# region LOSS_FUNCTIONS

def get_projectors(n):
    projectors = []
    for i in range(2**n):
        b = bin(i)[2:]
        while len(b)<n:
            b = '0'+b
        b = np.array([int(j) for j in b])[::-1]
        array = 1

        for bi in b:

            array = np.kron(array,(1+(-1)**bi)/2*np.array([1,0])+(1-(-1)**bi)/2*np.array([0,1]))

#         proj = np.outer(array,array)
        proj = array
        projectors.append(proj)

    return projectors

def kernel(X,Y,sigma):


    distances = pairwise_distances(X,Y,n_jobs=-1).reshape(-1)
    kern      = np.array([np.exp(-distances**2/(2*s)) for s in sigma])

    return np.mean(kern, axis=-1)

def exact_kernel(pa,pb,matrices):

    kern = np.array([np.dot(pa.transpose(),np.dot(mat,pb)) for mat in matrices])
    return kern

def compute_kernel_matrix(N,sigma):
    matrices = []
    for s in sigma:
        mat = np.zeros((2**N,2**N))
        matrices.append(mat)

    for n in range(2**N):
        b = bin(n)[2:]
        while len(b)<N:
            b= '0'+b

        b = [int(i) for i in b][::-1]
        for m in range(n,2**N):
            c = bin(m)[2:]
            while len(c)<N:
                c= '0'+c

            c = [int(i) for i in c][::-1]

            for _,s in enumerate(sigma):
                matrices[_][n,m] = np.exp(-0.5*np.sum(np.abs(np.array(b)-np.array(c)))/s)
                matrices[_][m,n] = matrices[_][n,m]
    return matrices

def MMD(samples, target, signal, exact):
    """
     exact: whetever the samples are samples or probabilities, in that case the expectation value is
        computed exactly
    """

    if exact:
        return exact_kernel(samples,samples,signal) - 2*exact_kernel(samples,target,signal) + exact_kernel(target,target,signal)
    else:
        return kernel(samples,samples,signal) - 2*kernel(samples,target,signal) + kernel(target,target,signal)

def KLD(samples, target):
    logp = np.log(samples + 10**-8)
    logq = np.log(target  + 10**-8)

    return np.array(np.dot(samples,logp-logq)).reshape(-1,1)

def grad_KLD(samples, samples_plus, samples_minus, target, signal = None, exact=None):
    grad_p = (samples_plus-samples_minus)/2
    return -np.array(np.dot(target,grad_p/(samples+10**-8))).reshape(-1,1)

def compute_TV(ansatz, parameters, qubits,  q, values, projectors, log_train = False, target_log = None, norm_log = None):
    state_raw = qulacs.QuantumState(qubits)
    state_raw = ansatz(state_raw.copy(), parameters)
    p = compute_samples(state_raw.copy(),  shots = 0,
        n_qubits=qubits, values = values, projectors = projectors)

    return 0.5*np.sum(np.abs(p-q)), p

def grad_MMD(samples, samples_plus, samples_minus, target, signal, exact):

    if exact:
        a =     exact_kernel(samples_plus,samples,signal) - exact_kernel(samples_minus,samples,signal)
        a = a - exact_kernel(samples_plus,target,signal)  + exact_kernel(samples_minus,target,signal)
    else:

        a =     kernel(samples_plus,samples,signal)  -  kernel(samples_minus,samples,signal)
        a = a - kernel(samples_plus,target,signal)   +  kernel(samples_minus,target,signal)

    return a

def compute_gradient(ansatz, parameters, target_train, n_qubits, n_shots,
                     grad_loss_function, signal = None, exact = True, values = None):
    gradients = []
    state_raw = qulacs.QuantumState(n_qubits)
    state     = ansatz(state_raw.copy(), parameters)
    samples   = compute_samples(state.copy(), n_shots, n_qubits, exact,values)


    for i in range(len(parameters)):
        eps  = np.pi/2

        v    = np.zeros_like(parameters)
        v[i] = 1

        param_plus  = parameters + eps*v
        param_minus = parameters - eps*v


        state_plus  = ansatz(state_raw.copy(),param_plus )
        state_minus = ansatz(state_raw.copy(),param_minus)


        samples_plus  = compute_samples(state_plus.copy() , n_shots,n_qubits, exact, values)
        samples_minus = compute_samples(state_minus.copy(), n_shots,n_qubits,exact, values)

        jac = grad_loss_function(samples.copy(), samples_plus, samples_minus,
                                 target_train, signal,  exact)
        gradients.append(jac)


    return gradients

def compute_samples(state, shots, n_qubits, exact = True,values = None, projectors = None):


    if shots  == 0:

        vector  = state.get_vector()
        if projectors is None:
            samples = (vector*vector.conjugate()).real
        else:
            samples = np.array([np.linalg.norm(np.dot(proj,vector))**2 for proj in projectors],dtype=np.float32).real

        return samples

    else:
        new_state = state.copy()
        s  = new_state.sampling(shots)


        if not exact:
            s = values[s]
            return s

        samples = np.zeros((2**n_qubits))
        for _ in s:
            samples[_]+=1
        samples = samples/shots


        return samples

# endregion LOSS_FUNCTIONS

# region FIDELITY 

def vectorized_reverse_fidelity(reverse_wavefunction: np.ndarray, local=True):

    n_qubits = round(np.log2(len(reverse_wavefunction)))

    if local:
        fid = np.mean(
            jax_get_local_probs(reverse_wavefunction.reshape((2,) * n_qubits))
        )
    else:

        fid = np.abs(reverse_wavefunction[0]) ** 2

    return 1 - fid


import tensornetwork as tn
from jax import jit

@jit
def jax_get_local_probs(wf: np.ndarray):
    print("COMPILING")
    n_qubits = len(wf.shape)

    expects = [0] * n_qubits

    n = tn.Node(wf, backend="jax")
    m = tn.Node(wf.conj(), backend="jax")

    for q_ind in range(n_qubits):

        for ii in range(0, q_ind):
            tn.connect(n[ii], m[ii])

        for ii in range(q_ind + 1, n_qubits):
            tn.connect(n[ii], m[ii])

        expects[q_ind] = abs(tn.contract_between(n, m).tensor[0, 0])

    return expects

def local_sampled_reverse_fidelity(reversed_model_dict: dict):
    n_qubits = len(list(reversed_model_dict.keys())[0])

    expectations = [0] * n_qubits
    for x, x_prob in reversed_model_dict.items():

        for ii in range(n_qubits):

            if x[ii] == '0':
                expectations[ii] += x_prob

    return 1 - np.mean(expectations)

def compute_gradients_fidelity(ansatz_dagger,parameters,target_train, n_qubits, n_shots,
                              values, local, projectors):
    gradients = []
    state = qulacs.QuantumState(n_qubits)
    wf = np.array(np.sum(np.array([np.sqrt(target_train[x])*projectors[x]
                                for x in range(2**n_qubits)]),axis = 0))

    state.load(wf)


    for i in range(len(parameters)):
        eps  = np.pi/2

        v    = np.zeros_like(parameters)
        v[i] = 1

        param_plus  = parameters + eps*v
        param_minus = parameters - eps*v



        state_plus = ansatz_dagger(state.copy(),param_plus)
        state_minus = ansatz_dagger(state.copy(),param_minus)


        fid_plus  = compute_samples_fidelity(state_plus.copy(), n_shots,n_qubits, values, local)
        fid_minus = compute_samples_fidelity(state_minus.copy(), n_shots,n_qubits, values, local)
#         print(fid_plus,fid_minus)
        jac = (fid_plus-fid_minus)/2

        gradients.append(np.array(jac).reshape(1,1))
    return gradients

def compute_samples_fidelity(state, n_shots, n_qubits, values, local):

    if n_shots  == 0:
        vector  = state.get_vector()
        return vectorized_reverse_fidelity(vector, local)



    s   = state.sampling(n_shots)
    samples = values[s]

    vector = {}
    h      = np.histogram(s.copy(), bins = 2**n_qubits)
    prob = h[0]/n_shots

    for _, p in enumerate(prob):
        key = format(_, "0" + str(n_qubits) + "b")

        vector[key]=p


    return local_sampled_reverse_fidelity(vector)

# endregion FIDELITY

class ADAM:
    def __init__(
        self,
        n_param,
        tol: float = 1e-6,
        lr: float = 1e-1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        noise_factor: float = 1e-8,
        eps: float = 1e-10,
        amsgrad: bool = False,
    ) -> None:

        self._tol = tol
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._noise_factor = noise_factor
        self._eps = eps
        self._amsgrad = amsgrad


        self._t = 1
        self._m = np.zeros((n_param))
        self._v = np.zeros((n_param))
        if self._amsgrad:
            self._v_eff = np.zeros((n_param))

    def update(self,params,derivative):

        self._m = self._beta_1 * self._m + (1 - self._beta_1) * derivative
        self._v = self._beta_2 * self._v + (1 - self._beta_2) * derivative * derivative
        lr_eff = self._lr * np.sqrt(1 - self._beta_2**self._t) / (1 - self._beta_1**self._t)
        if not self._amsgrad:
            params_new = params - lr_eff * self._m.flatten() / (
                    np.sqrt(self._v.flatten()) + self._noise_factor
                )
        else:
            self._v_eff = np.maximum(self._v_eff, self._v)
            params_new = params - lr_eff * self._m.flatten() / (
                    np.sqrt(self._v_eff.flatten()) + self._noise_factor
                )
        self._t +=1
        return params_new

# endregion functions

if __name__ == '__main__':
    run()
