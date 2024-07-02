using RecursiveCausalDiscovery
using Test
using Statistics
using Tables
using CSV
using Graphs

# navigate with your shell in the main directory of the package
# First run: using Pkg; Pkg.activate("."); using TestEnv; TestEnv.activate(); include("test/runtests.jl")
# Second run: include("test/runtests.jl")

function test_data(folder_name)
    matrix_data = CSV.read(joinpath(folder_name, "data.csv"), Tables.matrix)
    table_data = Tables.table(matrix_data)

    sig_level = 2 / size(matrix_data, 2)^2
    ci_test = (x::Int, y::Int, z::Vector{Int}, data::Matrix{Float64}) -> fisher_z(x, y, z, data, sig_level)

    precision_mat = inv(cor(matrix_data))
    opt_ci_test = (x::Int, y::Int, z::Vector{Int}, data::Matrix{Float64}) -> opt_fisher_z(x, y, z, size(matrix_data, 1), precision_mat, sig_level)

    # learn the skeleton using RSL
    rsl_skeleton = learn_and_get_skeleton(table_data, ci_test, mkbd_ci_test=opt_ci_test)

    # load true adjacency matrix from csv
    true_adj_mat = Int.(CSV.read(joinpath(folder_name, "true_adj_mat.csv"), Tables.matrix, header=false))
    true_skeleton = Graph(true_adj_mat)

    # calculate f1 score
    rsl_f1 = f1_score(true_skeleton, rsl_skeleton)
    return rsl_f1
end


@testset "RSL test n=10" begin
    folder_name = "n_10"
    @test test_data(folder_name) == 1.0
end

@testset "RSL test n=20" begin
    folder_name = "n_20"
    @test test_data(folder_name) == 1.0
end

@testset "RSL test n=50" begin
    folder_name = "n_50"
    @test test_data(folder_name) == 1.0
end

@testset "RSL test n=100" begin
    folder_name = "n_100"
    @test test_data(folder_name) == 1.0
end

@testset "RSL test n=200" begin
    folder_name = "n_200"
    @test test_data(folder_name) == 0.9698375870069604
end