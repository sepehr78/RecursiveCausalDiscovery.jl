using RecursiveCausalDiscovery
using Graphs
using CSV
using Tables

# generate a random DAG
num_vars = 50
edge_prob = 0.1
adj_mat = gen_er_dag_adj_mat(num_vars, edge_prob)

# generate random Gaussian data
num_samples = 1000
data = gen_gaussian_data(adj_mat, num_samples)

# use a Gaussian conditional independence test
sig_level = 0.01
ci_test = (x, y, cond_vec, data) -> fisher_z(x, y, cond_vec, data, sig_level)

# learn the skeleton of causal graph using RSL
learned_pdag = rsld(data, ci_test, true)
learned_skeleton = SimpleGraph(learned_pdag)

# load true adjacency matrix from csv
true_dag = SimpleDiGraph(adj_mat)
true_skeleton = SimpleGraph(true_dag)

# calculate the f1 score with respect to dag (we don't expect this to be high because many edges cannot be oriented solely from data)
rsl_f1 = f1_score(true_dag, learned_pdag)
println("RSL F1 Score: ", rsl_f1)

# calculate the f1 score with respect to the skeleton
rsl_skeleton_f1 = f1_score(true_skeleton, learned_skeleton)
println("RSL Skeleton F1 Score: ", rsl_skeleton_f1)

