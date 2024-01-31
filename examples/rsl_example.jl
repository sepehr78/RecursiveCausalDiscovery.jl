using Graphs
using Random
using LinearAlgebra
using DataFrames
using Plots
using GraphRecipes
using Statistics
using Distributions

using .RecursiveCausalDiscovery

"""
    gen_er_dag_adj_mat(num_vars::Int, edge_prob::Float64)

Generate an Erdos-Renyi DAG with a given number of variables and edge probability.

# Arguments
- `num_vars::Int`: Number of variables.
- `edge_prob::Float64`: Probability of an edge between any two variables.

# Returns
- `Array{Int, 2}`: Adjacency matrix of the generated DAG.
"""
function gen_er_dag_adj_mat(num_vars::Int, edge_prob::Float64)
    # Generate a random upper triangular matrix
    arr = triu(rand(num_vars, num_vars), 1)

    # Convert to adjacency matrix with probability edge_prob
    adj_mat = Int.(arr .> 1 - edge_prob)

    # Generate a random permutation
    perm = randperm(num_vars)

    # Apply the permutation to rows and the corresponding columns
    adj_mat_perm = adj_mat[perm, perm]

    return adj_mat_perm
end

"""
    gen_gaussian_data(dag_adj_mat::Matrix{Int}, num_samples::Int)

Generate random Gaussian samples for each variable from a given DAG.

# Arguments
- `dag_adj_mat::Matrix{Int}`: The adjacency matrix of the DAG.
- `num_samples::Int`: Number of samples to generate for each variable.

# Returns
- `DataFrame`: A DataFrame with the generated samples.
"""
function gen_gaussian_data(dag_adj_mat::Matrix{Int}, num_samples::Int)
    n = size(dag_adj_mat, 2)
    noise = randn(num_samples, n) * Diagonal(0.7 .+ 0.5 .* rand(n))
    B = transpose(dag_adj_mat) .* ((1 .+ 0.5 .* rand(n)) .* ((-1) .^ (rand(n) .> 0.5)))
    D = noise / (I - transpose(B))
    return DataFrame(D, :auto)
end

"""
    fisher_z(x_name::String, y_name::String, s::Vector{String}, data_df::DataFrame, significance_level::Float64=0.01)

Test for conditional independence between variables X and Y given a set Z in dataset D.

# Arguments
- `x_name::String`: Name of the first variable.
- `y_name::String`: Name of the second variable.
- `s::Vector{String}`: A list of names for variables in the conditioning set.
- `data_df::DataFrame`: A DataFrame of data.
- `significance_level::Float64`: The significance level for the test.

# Returns
- `Bool`: `true` if conditionally independent, `false` otherwise.
"""
function fisher_z(x_name::String, y_name::String, s::Vector{String}, data_df::DataFrame, significance_level::Float64=0.01)
    # Number of samples
    n = size(data_df, 1)

    # Select columns corresponding to X, Y, and Z from the dataset
    data_mat = data_df[!, [x_name, y_name, s...]]

    # Compute the precision matrix
    R = cor(Matrix(data_mat))
    P = inv(R)

    # Calculate the partial correlation coefficient and Fisher Z-transform
    ro = -P[1, 2] / sqrt(P[1, 1] * P[2, 2])
    zro = 0.5 * log((1 + ro) / (1 - ro))

    # Test for conditional independence
    c = quantile(Normal(), 1 - significance_level / 2)
    return abs(zro) < c / sqrt(n - length(s) - 3)
end


adj_mat = gen_er_dag_adj_mat(10, 0.3)
df = gen_gaussian_data(adj_mat, 1000)
ci_test = (x, y, z, data) -> fisher_z(x, y, z, data, 2 / size(data, 1)^2)

rsl = RSL(df, ci_test)

# Learn the skeleton
skeleton = learn_and_get_skeleton!(rsl, df)  # potentially not working
