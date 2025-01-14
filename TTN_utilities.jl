module TTN_utilities

export build_random_ttn
export rename_tensor_indices!
export print_tensor_vector
export squared_norm

using ITensors

function build_random_ttn(n_physical_indexes::Int, bond_dimension::Int, layers::Int)
    # Input validation
    if layers < 1
        error("The number of layers must be at least 1.")
    end
    
    # Define physical indices
    physical_indexes = [Index(2, "site $i") for i in 1:n_physical_indexes]
    
    # Initialize variables to store bond indices and tensors for each layer
    bond_indices = Vector{Vector{Index}}()
    tensors = Vector{Vector{ITensor}}()
    
    # Compute the number of bond indices and construct tensors layer by layer
    current_physical_indexes = physical_indexes
    for layer in 1:layers
        n_bond_indexes = div(length(current_physical_indexes), 2)
        if n_bond_indexes < 1
            error("Too many layers for the given number of physical indices.")
        end

        # Define bond indices for this layer
        bond_indexes = [Index(bond_dimension, "layer$(layer)_bond $i") for i in 1:n_bond_indexes]
        push!(bond_indices, bond_indexes)

        # Define tensors for this layer
        layer_tensors = [randomITensor(current_physical_indexes[2*i-1], current_physical_indexes[2*i], bond_indexes[i])
                         for i in 1:n_bond_indexes]
        layer_tensors = [normalize!(t) for t in layer_tensors]
        push!(tensors, layer_tensors)

        # Update current physical indices to bond indices for the next layer
        current_physical_indexes = bond_indexes
    end
    
    # Define the root tensor
    if length(current_physical_indexes) == 2
        root_tensor = randomITensor(current_physical_indexes[1], current_physical_indexes[2])
    else
        root_tensor = randomITensor(current_physical_indexes[1]) # Fallback for 1 bond index
    end
    root_tensor = normalize!(root_tensor)
    

    return tensors, root_tensor
end



function rename_tensor_indices!(tensors, old_indexes)
    """
    Rename the indices of a collection of tensors or a single tensor.
    
    Parameters:
        tensors: Either a vector of ITensors or a single ITensor.
        old_indexes: A corresponding array of old index sets to replace.
    """
    if isa(tensors, Vector{ITensor})
        # Case: A collection of tensors
        for i in 1:length(tensors)
            A = tensors[i]
            new_indexes = inds(A)
            for j in 1:length(old_indexes[i])
                replaceind!(A, new_indexes[j], old_indexes[i][j])
            end
            tensors[i] = A
        end
    else
        # Case: A single tensor
        new_indexes = inds(tensors)
        for j in 1:length(old_indexes[1])
            replaceind!(tensors, new_indexes[j], old_indexes[1][j])
        end
    end
end

function print_tensor_vector(tensors, name="Tensor Network")
    """
    Print a collection of tensors.

    Parameters:
        tensors: A vector of ITensors.
        name: A name for the tensor network.
    """
    println("=== $name ===")
    for (n, tensor) in enumerate(tensors)
        println("Tensor $n:")
        println(tensor)
        println()
    end
end

function squared_norm(tensors::Vector{ITensor})
    """
    Compute the squared norm of a tensor network.
    
    Parameters:
        tensors: A vector of ITensors representing the tensor network.
    
    Returns:
        The squared norm as a scalar value.
    """

    # Conjugate each tensor
    conj_tensors = [dag(tensor) for tensor in tensors]

    # Contract the original and conjugate tensor networks
    # Assuming the network is properly connected with shared indices
    norm_squared = 1.0
    for (tensor, conj_tensor) in zip(tensors, conj_tensors)
        norm_squared *= tensor * conj_tensor
    end

    return norm_squared
end


end
