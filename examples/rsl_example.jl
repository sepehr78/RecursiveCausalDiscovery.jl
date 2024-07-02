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
using BenchmarkTools


function get_pc_skeleton(t, p::Float64, test::typeof(gausscitest))
    Tables.istable(t) || throw(ArgumentError("Argument does not support Tables.jl"))
    X = Tables.matrix(t)
    N, n = size(X)
    C = Statistics.cor(X, dims = 1)
    g_skeleton, _ = CausalInference.skeleton(n, test, (C, N), quantile(Normal(), 1 - p / 2))
    return g_skeleton
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
folder_name = "../test/n_200/"
matrix_data = CSV.read(joinpath(folder_name, "data.csv"), Tables.matrix)
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
true_adj_mat = Int.(CSV.read(joinpath(folder_name, "true_adj_mat.csv"), Tables.matrix, header=false))
true_skeleton = Graph(true_adj_mat)

# calculate f1 score
rsl_f1 = f1_score(true_skeleton, rsl_skeleton)
pc_f1 = f1_score(true_skeleton, pc_skeleton)

println("RSL F1 Score: ", rsl_f1)
println("PC F1 Score: ", pc_f1)

#@benchmark learn_and_get_skeleton(table_data, ci_test, mkbd_ci_test=opt_ci_test)
#@benchmark get_pc_skeleton(table_data, sig_level, gausscitest)

