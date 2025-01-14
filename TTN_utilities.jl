module TTN_utilities

export rename_tensor_indices!
export print_tensor_vector
export squared_norm

using ITensors

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
