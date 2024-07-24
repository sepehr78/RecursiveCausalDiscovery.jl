using Random
using LinearAlgebra
using Statistics
using Graphs
using Distributions

export gen_er_dag_adj_mat, gen_gaussian_data, fisher_z, opt_fisher_z, f1_score

"""
    gen_er_dag_adj_mat(num_vars::Int, edge_prob::Float64)

Generate an Erdos-Renyi Directed Acyclic Graph (DAG) with a given number of variables and edge probability.

# Arguments
- `num_vars::Int`: Number of variables.
- `edge_prob::Float64`: Probability of an edge between any two variables.

# Returns
- `Matrix{Int}`: Adjacency matrix of the generated DAG.
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
    gen_gaussian_data(dag_adj_mat::Matrix{Int}, num_samples::Int)::Matrix{Float64}

Generate random Gaussian samples for each variable from a given Directed Acyclic Graph (DAG).

# Arguments
- `dag_adj_mat::Matrix{Int}`: The adjacency matrix of the DAG.
- `num_samples::Int`: Number of samples to generate for each variable.

# Returns
- `Matrix{Float64}`: A matrix with the generated samples. Each row represents a sample, and each column represents a variable.
"""
function gen_gaussian_data(dag_adj_mat::AbstractMatrix{Int}, num_samples::Int)::AbstractMatrix{Float64}
    n = size(dag_adj_mat, 2)
    noise = randn(num_samples, n) * Diagonal(0.7 .+ 0.5 .* rand(n))
    B = dag_adj_mat' .* ((1 .+ 0.5 .* rand(n)) .* ((-1) .^ (rand(n) .> 0.5)))
    D = noise * pinv(I - B')
    return D
end

"""
    fisher_z(x_idx::Int, y_idx::Int, s::Vector{Int}, data_mat_all::Matrix{Float64}, significance_level::Float64=0.01)

Test for conditional independence between variables X and Y given a set Z in dataset D using Fisher's Z-test.

# Arguments
- `x_idx::Int`: Index of the first variable.
- `y_idx::Int`: Index of the second variable.
- `s::Vector{Int}`: A list of indices for variables in the conditioning set.
- `data_mat_all::Matrix{Float64}`: The full dataset matrix with rows as samples and columns as variables.
- `significance_level::Float64`: The significance level for the test (default: 0.01).

# Returns
- `Bool`: `true` if `x_idx` conditionally independent from `y_idx` given `s`, `false` otherwise.
"""
function fisher_z(x_idx::Int, y_idx::Int, s::Vector{Int},
        data_mat_all::Matrix{Float64}, significance_level::Float64 = 0.01)
    # Number of samples
    n = size(data_mat_all, 1)

    # Select columns corresponding to X, Y, and Z from the dataset
    data_mat = @view data_mat_all[:, [x_idx, y_idx, s...]]

    # Compute the precision matrix
    R = cor(data_mat)
    P = inv(R)

    # R_sep = compute_correlation_matrix(data_mat)
    # P_sep = inv(R_sep)

    # Calculate the partial correlation coefficient and Fisher Z-transform
    ro = -P[1, 2] / sqrt(P[1, 1] * P[2, 2])
    zro = 0.5 * log((1 + ro) / (1 - ro))

    # Test for conditional independence
    c = quantile(Normal(), 1 - significance_level / 2)
    return abs(zro) < c / sqrt(n - length(s) - 3)
end


"""
    opt_fisher_z(x_idx::Int, y_idx::Int, s::Vector{Int}, num_samples::Int, precision_mat::Matrix{Float64}, significance_level::Float64=0.01)

Optimized version of Fisher's Z-test for conditional independence using a pre-computed precision matrix.

# Arguments
- `x_idx::Int`: Index of the first variable.
- `y_idx::Int`: Index of the second variable.
- `s::Vector{Int}`: A list of indices for variables in the conditioning set.
- `num_samples::Int`: Number of samples in the dataset.
- `precision_mat::Matrix{Float64}`: Pre-computed precision matrix.
- `significance_level::Float64`: The significance level for the test (default: 0.01).

# Returns
- `Bool`: `true` if `x_idx` conditionally independent from `y_idx` given `s`, `false` otherwise.
"""
function opt_fisher_z(x_idx::Int, y_idx::Int, s::Vector{Int}, num_samples::Int,
    precision_mat::Matrix{Float64}, significance_level::Float64 = 0.01)
    # Select rows and columns from the precision matrix corresponding to X, Y, and Z from the dataset
    P = precision_mat[[x_idx, y_idx, s...], :][:, [x_idx, y_idx, s...]]

    # Calculate the partial correlation coefficient and Fisher Z-transform
    ro = -P[1, 2] / sqrt(P[1, 1] * P[2, 2])
    zro = 0.5 * log((1 + ro) / (1 - ro))

    # Test for conditional independence
    c = quantile(Normal(), 1 - significance_level / 2)
    return abs(zro) < c / sqrt(num_samples - length(s) - 3)
end

"""
    f1_score(true_graph::AbstractGraph, predicted_graph::AbstractGraph)::Float64

Calculate the F1 score between the true graph and the predicted graph with respect to the edges.

# Arguments
- `true_graph::AbstractGraph`: The true graph.
- `predicted_graph::AbstractGraph`: The predicted graph.

# Returns
- `Float64`: The F1 score.
"""
function f1_score(true_graph::AbstractGraph, predicted_graph::AbstractGraph)::Float64
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
