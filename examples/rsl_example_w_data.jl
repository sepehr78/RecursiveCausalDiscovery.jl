using RecursiveCausalDiscovery
using Graphs
using CSV
using Tables

# load data (columns are variables and rows are samples)
data = CSV.read("data.csv", Tables.matrix)

# use a Gaussian conditional independence test
sig_level = 0.01
ci_test = (x, y, cond_vec, data) -> fisher_z(x, y, cond_vec, data, sig_level)

# learn the the causal graph using RSL
learned_pdag = rsld(data, ci_test, true)
learned_skeleton = SimpleGraph(learned_pdag)

# load true adjacency matrix from csv
true_adj_mat = Int.(CSV.read("dag_adj_mat.csv", Tables.matrix, header=false))
true_dag = SimpleDiGraph(true_adj_mat)
true_skeleton = SimpleGraph(true_dag)

# calculate the f1 score with respect to dag
rsl_f1 = f1_score(true_dag, learned_pdag)
println("RSL F1 Score: ", rsl_f1)

# calculate the f1 score with respect to the skeleton
rsl_skeleton_f1 = f1_score(true_skeleton, learned_skeleton)
println("RSL Skeleton F1 Score: ", rsl_skeleton_f1)
