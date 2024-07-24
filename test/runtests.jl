using RecursiveCausalDiscovery
using Test
using Statistics
using Tables
using CSV
using Graphs
using Random

function test_data(folder_name)
    matrix_data = CSV.read(joinpath(folder_name, "data.csv"), Tables.matrix)
    table_data = Tables.table(matrix_data)

    sig_level = 2 / size(matrix_data, 2)^2
    ci_test = (x::Int, y::Int, z::Vector{Int}, data::Matrix{Float64}) -> fisher_z(x, y, z, data, sig_level)

    precision_mat = inv(cor(matrix_data))
    opt_ci_test = (x::Int, y::Int, z::Vector{Int}, data::Matrix{Float64}) -> opt_fisher_z(x, y, z, size(matrix_data, 1), precision_mat, sig_level)

    # learn the skeleton using RSL
    rsl_pdag = rsld(table_data, ci_test, false, mkbd_ci_test=opt_ci_test)
    rsl_skeleton = Graph(rsl_pdag)

    # load true adjacency matrix from csv
    true_adj_mat = Int.(CSV.read(joinpath(folder_name, "true_adj_mat.csv"), Tables.matrix, header=false))
    true_skeleton = Graph(true_adj_mat)

    # calculate f1 score
    rsl_f1 = f1_score(true_skeleton, rsl_skeleton)
    return rsl_f1
end

function test_orienting(true_dag)
    # generate data
    num_samples = 1000
    Random.seed!(123)
    matrix_data = gen_gaussian_data(Matrix(adjacency_matrix(true_dag)), num_samples)

    sig_level = 2 / size(matrix_data, 2)^2
    ci_test = (x::Int, y::Int, z::Vector{Int}, data::Matrix{Float64}) -> fisher_z(x, y, z, data, sig_level)

    precision_mat = inv(cor(matrix_data))
    opt_ci_test = (x::Int, y::Int, z::Vector{Int}, data::Matrix{Float64}) -> opt_fisher_z(x, y, z, size(matrix_data, 1), precision_mat, sig_level)

    # learn the skeleton using RSL
    rsl_pdag = rsld(matrix_data, ci_test, true, mkbd_ci_test=opt_ci_test)

    # calculate f1 score
    rsl_f1 = f1_score(true_dag, rsl_pdag)
    return rsl_f1
end


@testset "RSL n=10 test" begin
    folder_name = "n_10"
    @test test_data(folder_name) == 1.0
end

@testset "RSL n=20 test" begin
    folder_name = "n_20"
    @test test_data(folder_name) == 1.0
end

@testset "RSL n=50 test" begin
    folder_name = "n_50"
    @test test_data(folder_name) == 1.0
end

@testset "RSL n=100 test" begin
    folder_name = "n_100"
    @test test_data(folder_name) == 1.0
end

@testset "RSL n=200 test" begin
    folder_name = "n_200"
    @test test_data(folder_name) == 0.9698375870069604
end

@testset "RSL v-structure test" begin
    # test that RSL can learn v-structures as expected

    # TEST 1: 1 -> 3 <- 2
    true_dag = DiGraph(3)
    add_edge!(true_dag, 1, 3)
    add_edge!(true_dag, 2, 3)
    @test test_orienting(true_dag) == 1.0

    # TEST 2: 1 -> 3 <- 2 and 1-> 4 <- 2
    true_dag = DiGraph(4)
    add_edge!(true_dag, 1, 3)
    add_edge!(true_dag, 2, 3)
    add_edge!(true_dag, 1, 4)
    add_edge!(true_dag, 2, 4)
    @test test_orienting(true_dag) == 1.0

    # TEST 3: 1 -> 3 <- 2 and ... and 1 -> k + 2 <- 2
    k = 10
    true_dag = DiGraph(k + 2)
    for i in 1:k
        add_edge!(true_dag, 1, i + 2)
        add_edge!(true_dag, 2, i + 2)
    end
    @test test_orienting(true_dag) == 1.0
end
