using Revise
using Graphs
using Random
using LinearAlgebra
using DataFrames
using Plots
using GraphRecipes
using Statistics
using Distributions
using CSV
using DelimitedFiles
using RecursiveCausalDiscovery
using CausalInference

function oy()
    println("sdfsdfef")
end

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
function fisher_z(x_idx::Int, y_idx::Int, s::Vector{Int}, data_mat_all::Matrix{Float64}, significance_level::Float64=0.01)
    # Number of samples
    n = size(data_mat_all, 1)

    # Select columns corresponding to X, Y, and Z from the dataset
    data_mat = @view data_mat_all[:, [x_idx, y_idx, s...]]

    # Compute the precision matrix
    R = cor(data_mat)
    P = inv(R)

    # Calculate the partial correlation coefficient and Fisher Z-transform
    ro = -P[1, 2] / sqrt(P[1, 1] * P[2, 2])
    zro = 0.5 * log((1 + ro) / (1 - ro))

    # Test for conditional independence
    c = quantile(Normal(), 1 - significance_level / 2)
    return abs(zro) < c / sqrt(n - length(s) - 3)
end

function opt_fisher_z(x_idx::Int, y_idx::Int, s::Vector{Int}, num_samples::Int, precision_mat::Matrix{Float64}, significance_level::Float64=0.01)
    # Select rows and columns from the precision matrix corresponding to X, Y, and Z from the dataset
    P = precision_mat[[x_idx, y_idx, s...], :][:, [x_idx, y_idx, s...]]

    # Calculate the partial correlation coefficient and Fisher Z-transform
    ro = -P[1, 2] / sqrt(P[1, 1] * P[2, 2])
    zro = 0.5 * log((1 + ro) / (1 - ro))

    # Test for conditional independence
    c = quantile(Normal(), 1 - significance_level / 2)
    return abs(zro) < c / sqrt(num_samples - length(s) - 3)
end

function get_pc_skeleton(t, p::Float64, test::typeof(gausscitest))
    Tables.istable(t) || throw(ArgumentError("Argument does not support Tables.jl"))
    X = Tables.matrix(t)
    N, n = size(X)
    C = Statistics.cor(X, dims = 1)
    g_skeleton, _ = CausalInference.skeleton(n, test, (C, N), quantile(Normal(), 1 - p / 2))
    return g_skeleton
end

function f1_score(true_graph::Graph, predicted_graph::Graph)::Float64
    true_edges = Set(Graphs.edges(true_graph))
    predicted_edges = Set(Graphs.edges(predicted_graph))
    
    true_positives = length(intersect(true_edges, predicted_edges))
    false_positives = length(setdiff(predicted_edges, true_edges))
    false_negatives = length(setdiff(true_edges, predicted_edges))
    
    if true_positives == 0
        return 0.0
    end
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    return 2 * (precision * recall) / (precision + recall)
end



# adj_mat = gen_er_dag_adj_mat(10, 0.3)
# df = gen_gaussian_data(adj_mat, 1000)
# ix = [1, 2, 3]
# A = [1, 1, 1]

# deforder = Base.Order.ord(isless, identity, nothing, Base.Order.Forward)

# ix = [1 4; 2 5; 3 6]
# A = [2 1; 4 3; 6 5]
# dims = nothing
# Base.Order.Perm(deforder, vec(A))
# sort!(ix; Base.Sort.DEFAULT_UNSTABLE, order = Base.Order.Perm(deforder, vec(A)), nothing, dims=2)

# sort!(ix, dims=1)

# sortperm([1,2;3,4;5,6], dims=1)
# sortperm([1], dims=1)

# load csv as matrix
data = CSV.read("data.csv", DataFrame)
matrix_data = Matrix(data)
table_data = Tables.table(matrix_data)

sig_level = 2 / size(matrix_data, 2)^2
ci_test = (x::Int, y::Int, z::Vector{Int}, data::Matrix{Float64}) -> fisher_z(x, y, z, data, sig_level)

precision_mat = inv(cor(matrix_data))
opt_ci_test = (x::Int, y::Int, z::Vector{Int}, data::Matrix{Float64}) -> opt_fisher_z(x, y, z, size(matrix_data, 1), precision_mat, sig_level)

# learn the skeleton using RSL
rsl_skeleton = learn_and_get_skeleton(table_data, ci_test, mkbd_ci_test=opt_ci_test)
rsl_adj_mat = adjacency_matrix(rsl_skeleton)

# learn the skeleton using PC (CausalInference.jl)
pc_skeleton = get_pc_skeleton(table_data, sig_level, gausscitest)
pc_adj_mat = adjacency_matrix(pc_skeleton)

# load true adjacency matrix from csv
true_adj_mat = Int.(readdlm("learned_adj_mat.csv", ','))
true_skeleton = Graph(true_adj_mat)

# calculate f1 score
rsl_f1 = f1_score(true_skeleton, rsl_skeleton)
pc_f1 = f1_score(true_skeleton, pc_skeleton)

println("RSL F1 Score: ", rsl_f1)
println("PC F1 Score: ", pc_f1)

@profview learn_and_get_skeleton(table_data, ci_test)

# THE NEW OPTIMIZED CALL TO LEARN THE SKELETON
rsl_skeleton = learn_and_get_skeleton(table_data, ci_test, mkbd_ci_test=opt_ci_test)

