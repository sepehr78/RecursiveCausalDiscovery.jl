module RecursiveCausalDiscovery

using Graphs
using DataFrames
using LinearAlgebra


"""
    find_markov_boundary_matrix(data::DataFrame, ci_test)

Computes the Markov boundary matrix for all variables.

# Arguments
- `data::DataFrame`: DataFrame where each column is a variable.
- `ci_test`: Conditional independence test to use.

# Returns
- `Matrix{Int}`: A matrix indicating whether variable i is in the Markov boundary of j.
"""
function find_markov_boundary_matrix(data::DataFrame, ci_test)
    num_vars = size(data, 2)
    var_names = names(data)
    markov_boundary_matrix = zeros(Int, num_vars, num_vars)

    for i in 1:(num_vars - 1)
        var_name = var_names[i]
        for j in (i + 1):num_vars
            var_name2 = var_names[j]
            cond_set = setdiff(var_names, [var_name, var_name2])
            if !ci_test(var_name, var_name2, cond_set, data)
                markov_boundary_matrix[i, j] = 1
                markov_boundary_matrix[j, i] = 1
            end
        end
    end

    return markov_boundary_matrix
end

"""
RSLBase Class for learning graph structure.

# Fields
- `data`: The dataset.
- `ci_test`: The conditional independence test function.
- Other fields for storing intermediate results and the learned graph.
"""
mutable struct RSL
    data::DataFrame
    ci_test
    var_names::Vector{String}
    markov_boundary_matrix::Matrix{Int}
    flag_arr::Vector{Bool}

    function RSL(data::DataFrame, ci_test)
        num_vars = size(data, 2)
        new(data, ci_test, names(data), zeros(Int, num_vars, num_vars), fill(true, num_vars))
        # Initialize other fields as needed
    end
end


"""
    learn_and_get_skeleton!(rsl::RSLBase, data::DataFrame)

Runs the algorithm on the data to learn and return the learned skeleton graph.

# Arguments
- `rsl::RSLBase`: The RSLBase instance.
- `data::DataFrame`: The data on which to run the algorithm.

# Returns
- `Graph`: The learned graph skeleton.
"""
function reset_rsl!(rsl::RSL, data::DataFrame)
    rsl.data = data
    num_vars = size(data, 2)
    rsl.var_names = names(data)
    rsl.flag_arr = fill(true, num_vars)
    rsl.markov_boundary_matrix = zeros(Int, num_vars, num_vars)
end

function learn_and_get_skeleton!(rsl::RSL, data::DataFrame)
    reset_rsl!(rsl, data)

    # Initialize graph with node names corresponding to variables
    skeleton = SimpleGraph(length(rsl.var_names))

    rsl.markov_boundary_matrix = find_markov_boundary_matrix(rsl.data, rsl.ci_test)

    var_idx_left_set = Set(1:length(rsl.var_names))
    # repeat for the number of variables
    for i in 1:length(rsl.var_names)
        # Find a removable variable
        removable_var_idx = find_removable(rsl, collect(var_idx_left_set))

        # Find the neighbors of the removable variable
        neighbors = find_neighborhood(rsl, removable_var_idx)

        # Update the Markov boundary matrix
        update_markov_boundary_matrix(rsl, removable_var_idx, neighbors)

        # Add edges between the removable variable and its neighbors
        for neighbor_idx in neighbors
            add_edge!(skeleton, removable_var_idx, neighbor_idx)
        end

        # Remove the removable variable from the set of variables left
        delete!(var_idx_left_set, removable_var_idx)
    end

    rsl.learned_skeleton = skeleton
    return skeleton
end


function find_neighborhood(rsl, var_idx::Int)::Vector{Int}
    """
    Find the neighborhood of a variable using Lemma 4 of the rsl paper.

    # Arguments
    - `var_idx::Int`: Index of the variable in the data.

    # Returns
    - `Vector{Int}`: Array containing the indices of the variables in the neighborhood.
    """

    var_name = rsl.var_names[var_idx]
    var_mk_arr = rsl.markov_boundary_matrix[var_idx, :]
    var_mk_idxs = findall(x -> x != 0, var_mk_arr)

    # Assume all variables are neighbors initially
    neighbors = copy(var_mk_arr)

    for mb_idx_y in 1:length(var_mk_idxs)
        for mb_idx_z in 1:length(var_mk_idxs)
            if mb_idx_y == mb_idx_z
                continue
            end
            var_y_idx = var_mk_idxs[mb_idx_y]
            var_z_idx = var_mk_idxs[mb_idx_z]
            var_y_name = rsl.var_names[var_y_idx]
            cond_set = [rsl.var_names[idx] for idx in setdiff(var_mk_idxs, [var_y_idx, var_z_idx])]

            if rsl.ci_test(var_name, var_y_name, cond_set, rsl.data)
                # var2 is a co-parent and thus NOT a neighbor
                neighbors[var_y_idx] = 0
                break
            end
        end
    end

    # Remove all variables that are not neighbors
    neighbors_idx_arr = findall(x -> x != 0, neighbors)
    return neighbors_idx_arr
end


function is_removable(rsl, var_idx::Int)::Bool
    """
    Check whether a variable is removable using Lemma 3 of the rsl paper.

    # Arguments
    - `var_idx::Int`: Index of the variable.

    # Returns
    - `Bool`: True if the variable is removable, False otherwise.
    """

    var_mk_arr = rsl.markov_boundary_matrix[var_idx, :]
    var_mk_idxs = findall(x -> x != 0, var_mk_arr)

    # Use Lemma 3 of rsl paper: var_y_idx is Y and var_z_idx is Z. cond_set is Mb(X) + {X} - {Y, Z}
    for mb_idx_y in 1:(length(var_mk_idxs) - 1)  # -1 because no need to check last variable and also symmetry
        for mb_idx_z in (mb_idx_y + 1):length(var_mk_idxs)
            var_y_idx = var_mk_idxs[mb_idx_y]
            var_z_idx = var_mk_idxs[mb_idx_z]
            var_y_name = rsl.var_names[var_y_idx]
            var_z_name = rsl.var_names[var_z_idx]
            cond_set = [rsl.var_names[idx] for idx in setdiff(var_mk_idxs, [var_y_idx, var_z_idx])] 
            push!(cond_set, rsl.var_names[var_idx])

            if rsl.ci_test(var_y_name, var_z_name, cond_set, rsl.data)
                return false
            end
        end
    end

    return true
end

function find_removable(rsl::RSL, var_idx_list::Vector{Int})::Int
    """
    Find a removable variable in the given list of variables.

    # Arguments
    - `var_idx_list::Vector{Int}`: List of variable indices.

    # Returns
    - `Int`: Index of the removable variable.
    """

    # Sort variables by the size of their Markov boundary
    mb_size = sum(rsl.markov_boundary_matrix[var_idx_list, :], dims=2)
    sort_indices = sortperm(mb_size, alg=QuickSort, by=x->-x) # Sorting in descending order
    sorted_var_idx = var_idx_list[sort_indices]

    for var_idx in sorted_var_idx
        if rsl.flag_arr[var_idx]
            rsl.flag_arr[var_idx] = false
            if is_removable(rsl, var_idx)
                return var_idx
            end
        end
    end

    # If no removable found, return the first variable
    return sorted_var_idx[1]
end

function update_markov_boundary_matrix(rsl, var_idx::Int, var_neighbors::Vector{Int})
    """
    Update the Markov boundary matrix after removing a variable.

    # Arguments
    - `var_idx::Int`: Index of the variable to remove.
    - `var_neighbors::Vector{Int}`: Array containing the indices of the neighbors of var_idx.
    """

    var_markov_boundary = findall(x -> x != 0, rsl.markov_boundary_matrix[var_idx, :])

    # For every variable in the Markov boundary of var_idx, remove it from the Markov boundary and update flag
    for mb_var_idx in var_markov_boundary
        rsl.markov_boundary_matrix[mb_var_idx, var_idx] = 0
        rsl.markov_boundary_matrix[var_idx, mb_var_idx] = 0
        rsl.flag_arr[mb_var_idx] = true
    end

    # Find nodes whose co-parent status changes
    # Only remove Y from mkvb of Z if X is their ONLY common child and they are NOT neighbors
    for ne_idx_y in 1:(length(var_neighbors) - 1)
        for ne_idx_z in (ne_idx_y + 1):length(var_neighbors)
            var_y_idx = var_neighbors[ne_idx_y]
            var_z_idx = var_neighbors[ne_idx_z]
            var_y_name = rsl.var_names[var_y_idx]
            var_z_name = rsl.var_names[var_z_idx]

            var_y_markov_boundary = findall(x -> x != 0, rsl.markov_boundary_matrix[var_y_idx, :])
            var_z_markov_boundary = findall(x -> x != 0, rsl.markov_boundary_matrix[var_z_idx, :])

            if sum(rsl.markov_boundary_matrix[var_y_idx, :]) < sum(rsl.markov_boundary_matrix[var_z_idx, :])
                cond_set = [rsl.var_names[idx] for idx in setdiff(var_y_markov_boundary, [var_z_idx])]
            else
                cond_set = [rsl.var_names[idx] for idx in setdiff(var_z_markov_boundary, [var_y_idx])]
            end

            if rsl.ci_test(var_y_name, var_z_name, cond_set, rsl.data)
                rsl.markov_boundary_matrix[var_y_idx, var_z_idx] = 0
                rsl.markov_boundary_matrix[var_z_idx, var_y_idx] = 0
                rsl.flag_arr[var_y_idx] = true
                rsl.flag_arr[var_z_idx] = true
            end
        end
    end
end

end # module RecursiveCausalDiscovery
