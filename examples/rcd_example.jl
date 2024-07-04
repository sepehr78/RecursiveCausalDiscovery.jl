using RecursiveCausalDiscovery
using Graphs
using CSV
using Tables

# load data (columns are variables and rows are samples)
data = CSV.read("data.csv", Tables.matrix)

# use a Gaussian conditional independence test
sig_level = 0.01
ci_test = (x, y, cond_vec, data) -> fisher_z(x, y, cond_vec, data, sig_level)

# learn the skeleton of causal graph using RSL
learned_skeleton = learn_and_get_skeleton(data, ci_test)

# load true adjacency matrix from csv
true_adj_mat = Int.(CSV.read("dag_adj_mat.csv", Tables.matrix, header=false))
true_dag = SimpleDiGraph(true_adj_mat)
true_skeleton = SimpleGraph(true_dag)

# calculate the f1 score
rsl_f1 = f1_score(true_skeleton, learned_skeleton)
println("RSL F1 Score: ", rsl_f1)
