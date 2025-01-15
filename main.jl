using ITensors
include("TTN_utilities.jl")
using .TTN_utilities

# Parameters
# Last layer contain only root tensor
number_physical_indexes = 8
bond_dimension = 2
number_layers = 3

# Define indexes
number_bound_1_indexes = div(number_physical_indexes,2)
number_bound_2_indexes = div(number_physical_indexes,4)

index_physical = [Index(2, "physical $i") for i in 1:number_physical_indexes]
index_bound_1 = [Index(bond_dimension, "bound 1 $i") for i in 1:number_bound_1_indexes]
index_bound_2 = [Index(bond_dimension, "bound 2 $i") for i in 1:number_bound_2_indexes]

# Build TTN
TTN = Vector{Vector{ITensor}}(undef, number_layers)
TTN[1] = [randomITensor(index_physical[2*i-1], index_physical[2*i], index_bound_1[i]) for i in 1:number_bound_1_indexes]
TTN[2] = [randomITensor(index_bound_1[2*i-1], index_bound_1[2*i], index_bound_2[i]) for i in 1:number_bound_2_indexes]
TTN[3] = [randomITensor(index_bound_2[1], index_bound_2[2])]

##############################################################################################################################
contracted_initial_bottom = (TTN[1][1]*TTN[2][1]*TTN[1][2])*TTN[3][1]*(TTN[1][3]*TTN[2][2]*TTN[1][4])
println("contracted_initial_top: ",contracted_initial_bottom*dag(contracted_initial_bottom))
contracted_initial_top = vcat(TTN[1], TTN[2], TTN[3])
println("contracted_initial_bottom: ",squared_norm(contracted_initial_top))
##############################################################################################################################

# Reduction to canonical form
println("Starting reduction to canonical form")
for i in 1:number_bound_1_indexes
    Q, R = qr(TTN[1][i], [index_physical[2*i-1],index_physical[2*i]], [index_bound_1[i]])
    norm_Q = sqrt((Q*dag(Q))[][])
    R = R*norm_Q
    Q = Q/norm_Q
    TTN[1][i] = Q
    TTN[2][Int(ceil(i/2))] = R*TTN[2][Int(ceil(i/2))]
end

# Restoring indexes names
for i in 1:number_bound_1_indexes
    replaceind!(TTN[1][i], inds(TTN[1][i])[3], index_bound_1[i])
end
for i in 1:number_bound_2_indexes
    replaceind!(TTN[2][i], inds(TTN[2][i])[1], index_bound_1[2*i-1])
    replaceind!(TTN[2][i], inds(TTN[2][i])[2], index_bound_1[2*i])
end

for i in 1:number_bound_2_indexes
    Q, R = qr(TTN[2][i], [index_bound_1[2*i-1],index_bound_1[2*i]], [index_bound_2[i]])
    norm_Q = sqrt((Q*dag(Q))[][])
    R = R*norm_Q
    Q = Q/norm_Q
    TTN[2][i] = Q
    TTN[3][1] = R * TTN[3][1]
end
for i in 1:number_bound_2_indexes
    replaceind!(TTN[2][i], inds(TTN[2][i])[3], index_bound_2[i])
end
replaceind!(TTN[3][1], inds(TTN[3][1])[1], index_bound_2[1])
replaceind!(TTN[3][1], inds(TTN[3][1])[2], index_bound_2[2])

##############################################################################################################################
contracted_final_top = (TTN[1][1]*TTN[2][1]*TTN[1][2])*TTN[3][1]*(TTN[1][3]*TTN[2][2]*TTN[1][4])
println("contracted_final_top: ",contracted_final_top*dag(contracted_final_top))
contracted_final_bottom = vcat(TTN[1], TTN[2], TTN[3])
println("contracted_final_bottom: ",squared_norm(contracted_final_bottom))
println("contracted_final_root: ",TTN[3][1]*dag(TTN[3][1]))
##############################################################################################################################

