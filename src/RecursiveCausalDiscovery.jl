module RecursiveCausalDiscovery
include("utils.jl")
include("meek.jl")
using Graphs
using Tables
using LinearAlgebra
using Base.Threads

export rsld

const global REMOVABLE_NOT_FOUND = -1

"""
    find_markov_boundary_matrix!(markov_boundary_matrix::BitMatrix, data::AbstractMatrix, ci_test::Function)

Compute the Markov boundary matrix for all variables in-place.

# Arguments
- `markov_boundary_matrix::BitMatrix`: Pre-allocated matrix to be updated.
- `data::AbstractMatrix`: Data matrix where each column is a variable and rows are samples.
- `ci_test::Function`: Conditional independence test to use.
"""
function find_markov_boundary_matrix!(
        markov_boundary_matrix::BitMatrix, data::AbstractMatrix, ci_test::Function)
    num_vars = size(data, 2)
    @threads for i in 1:(num_vars - 1)
        for j in (i + 1):num_vars
            cond_set = setdiff(1:num_vars, (i, j))
            @inbounds markov_boundary_matrix[i, j] = markov_boundary_matrix[j, i] = !ci_test(
                i, j, cond_set, data)
        end
    end
end

"""
    RSL{T}

Recursive Structure Learner (RSL) for learning graph structure.

# Fields
- `data::Matrix{T}`: The data from which to learn the graph.
- `ci_test::Function`: The conditional independence test function to use.
- `markov_boundary_matrix::BitMatrix`: Matrix indicating whether variable i is in the Markov boundary of j.
- `skip_rem_check_vec::BitVector`: Used to keep track of which variables to skip when checking for removability.
"""
struct RSL{T, F}
    data::Matrix{T}
    ci_test::F
    markov_boundary_matrix::BitMatrix
    skip_rem_check_vec::BitVector

    function RSL(data::Matrix{T}, ci_test::Function) where {T}
        num_vars = size(data, 2)
        return new{T, typeof(ci_test)}(
            data, ci_test, falses(num_vars, num_vars), falses(num_vars)
        )
    end
end

"""
    rsld(data::AbstractMatrix, ci_test::Function; mkbd_ci_test::Function=ci_test)::SimpleDiGraph
    rsld(data, ci_test::Function; mkbd_ci_test::Function=ci_test)::SimpleDiGraph

Run the RSL-D algorithm on the data to learn and return the learned partially directed acyclic graph (PDAG).

# Arguments
- `data`: The data on which to run the algorithm. Can be an AbstractMatrix or any type supporting Tables.jl interface. Columns should be variables and rows samples.
- `ci_test::Function`: The conditional independence test function to use. Should have signature `ci_test(x::Int, y::Int, cond_set::Vector{Int}, data::Matrix{Float64}) -> Bool`.
- `orient_edges::Bool`: Whether to orient the edges further using Meek rules (default: `true`).
- `mkbd_ci_test::Function`: The conditional independence test function to use for Markov boundary discovery (default: `ci_test`).

# Returns
- `SimpleDiGraph`: The learned PDAG.
"""
function rsld(data::AbstractMatrix, ci_test::Function, orient_edges = true;
        mkbd_ci_test::Function = ci_test)::SimpleDiGraph
    return rsld(Tables.table(data), ci_test, orient_edges; mkbd_ci_test = ci_test)
end

function rsld(data, ci_test::Function, orient_edges = true;
        mkbd_ci_test::Function = ci_test)::SimpleDiGraph
    Tables.istable(data) || throw(ArgumentError("Argument does not support Tables.jl"))
    data_mat = Tables.matrix(data)
    num_vars = size(data_mat, 2)

    rsl = RSL(data_mat, ci_test)

    # Initialize graph with node names corresponding to variables
    pc_dag = SimpleDiGraph(num_vars)

    # auxilary vector used to store extra info in find_neighbors and update_markov_boundary_matrix
    aux_vec = zeros(Int, num_vars)

    # Compute the Markov boundary matrix
    find_markov_boundary_matrix!(rsl.markov_boundary_matrix, rsl.data, mkbd_ci_test)

    var_arr = 1:num_vars
    var_left_bool_arr = trues(num_vars)  # if ith position is True, indicates that i is left
    var_removed_bool_arr = falses(num_vars)  # if ith position is True, indicates that i is removed

    for i in 1:(num_vars - 1)
        # only consider variables that are left and have skip check set to False
        var_to_check_arr = @views var_arr[var_left_bool_arr .& .!rsl.skip_rem_check_vec]

        # sort the variables by the size of their markov boundary
        mb_size = @views sum(rsl.markov_boundary_matrix[:, var_to_check_arr]; dims = 1)[
            1, :]
        sort_indices = sortperm(mb_size)
        sorted_var_arr = @views var_to_check_arr[sort_indices]

        # find a removable variable
        removable_var = find_removable!(rsl, sorted_var_arr)

        if removable_var == REMOVABLE_NOT_FOUND
            # if no removable found, then pick the variable with the smallest markov boundary from var_left_bool_arr
            var_left_arr = findall(var_left_bool_arr)
            mb_size_all = @views sum(rsl.markov_boundary_matrix[var_left_arr, :], dims = 2)
            removable_var = var_left_arr[argmin(mb_size_all)]

            rsl.skip_rem_check_vec .= false
        end

        # find the neighbors of the removable variable
        # aux_vec[i] is a common child of removable_var and i (removable_var -> aux_vec[i] <- i)
        aux_vec .= 0
        neighbors = find_neighbors!(rsl, removable_var, aux_vec)
        
        if orient_edges
        # add directed edges between co-parents and child and update v-structures
            for (coparent_idx, child_idx) in enumerate(aux_vec)
                if child_idx != 0
                    add_edge_pdag!(pc_dag, coparent_idx, child_idx, true)
                    add_edge_pdag!(pc_dag, removable_var, child_idx, true)

                    # since coparent and removable_var are NOT neighbors, check for previous v-structures
                    update_v_structures!(pc_dag, var_removed_bool_arr, coparent_idx, removable_var)
                end
            end
        end

        # update the markov boundary matrix
        # aux_vec[i] and i are parents of removable_var (aux_vec[i] -> removable_var <- i)
        aux_vec .= 0
        update_markov_boundary_matrix!(rsl, removable_var, neighbors, aux_vec)

        if orient_edges
            # add directed edges between the parents and the removable variable, and update v-structures
            for (coparent1, coparent2) in enumerate(aux_vec)
                if coparent2 != 0
                    add_edge_pdag!(pc_dag, coparent1, removable_var, true)
                    add_edge_pdag!(pc_dag, coparent2, removable_var, true)
                    
                    # since coparent1 and coparent2 are NOT neighbors, check for previous v-structures
                    update_v_structures!(pc_dag, var_removed_bool_arr, coparent1, coparent2)
                end
            end
        end

        # add undirected edges between the removable variable and its neighbors
        for neighbor_idx in neighbors
            add_edge_pdag!(pc_dag, removable_var, neighbor_idx, false)
        end

        # remove the removable variable from the set of variables left
        var_left_bool_arr[removable_var] = false
        var_removed_bool_arr[removable_var] = true
    end

    # apply meek rules to orient edges furhter
    if orient_edges
        meek_rules!(pc_dag)
    end

    return pc_dag
end


"""
    update_v_structures!(pc_dag::DiGraph, removed_bool_vec::BitVector, var_y::Int, var_z::Int)

Goes through removed variables and checks whether there are any v-structures given by var_y -> removed_var <- var_z. If so, orients the edge var_y -> removed_var and var_z -> removed_var. Note that var_y and var_z MUST NOT be adjacent.

# Arguments
- `pc_dag::DiGraph`: The graph to update.
- `removed_bool_vec::BitVector`: Vector containing the removed variables.
- `var_y::Int`: Index of variable Y, NOT a neighbor of `var_z`.
- `var_z::Int`: Index of variable Z, NOT a neighbor of `var_y`.
"""

function update_v_structures!(pc_dag::DiGraph, removed_bool_vec::BitVector, var_y::Int, var_z::Int)
    # for each removed variable, check whether BOTH var_y and var_z are neighbors of the removed variable
    # if so, we have a v-structure and we need to orient the edge var_y -> removed_var <- var_z
    for (removed_var, is_removed) in enumerate(removed_bool_vec)
        if is_removed && isundirected(pc_dag, removed_var, var_y) && isundirected(pc_dag, removed_var, var_z)
            orientedge!(pc_dag, var_y, removed_var)
            orientedge!(pc_dag, var_z, removed_var)
        end
    end

end


"""
    add_edge_pdag!(pc_dag::SimpleDiGraph, src::Int, dst::Int, is_directed::Bool)

Add an edge to the graph. Directed edge takes precedence over undirected edge. If a directed edge is being added, the undirected edge (i.e, reverse direction) is removed. If an undirected edge is being added and a directed edge exists, the directed edge is kept.

# Arguments
- `pc_dag::SimpleDiGraph`: The graph to add the edge to.
- `src::Int`: Source node.
- `dst::Int`: Destination node.
- `is_directed::Bool`: Whether the edge is directed.
"""
function add_edge_pdag!(pc_dag::SimpleDiGraph, src::Int, dst::Int, is_directed::Bool)
    if is_directed
        # check if a directed edge exists in the opposite direction. If so, throw error
        if has_edge(pc_dag, dst, src) && !has_edge(pc_dag, src, dst)
            # this should not happen, so it means CI test at some point gave an incorrect result
            return
        else
            add_edge!(pc_dag, src, dst)
            rem_edge!(pc_dag, dst, src)
        end
    else  # we are adding an undirected edge
        # check if a directed edge exists
        if !isoriented(pc_dag, src, dst)
            add_edge!(pc_dag, src, dst)
            add_edge!(pc_dag, dst, src)
        end
    end
end

"""
    find_neighbors(rsl::RSL, var_idx::Int, common_child_arr::AbstractVector{Int})::Vector{Int}

Find the neighbors of a variable. Uses Lemma 4 of the RSL paper.

# Arguments
- `rsl::RSL`: The RSL object.
- `var_idx::Int`: Index of the variable in the data (i.e., column index in the data matrix)
- `common_child_arr::AbstractVector{Int}`: Vector to store the common child of the variable and its co-parents (used for orientation).

# Returns
- `Vector{Int}`: Vector containing the variables in the neighbors.
"""
function find_neighbors!(rsl::RSL, var_idx::Int, common_child_arr::AbstractVector{Int})::Vector{Int}
    var_mk_arr = @view rsl.markov_boundary_matrix[:, var_idx]
    var_mk_idxs = findall(var_mk_arr)

    # Assume all variables are neighbors initially
    neighbors = copy(var_mk_arr)

    for var_y in var_mk_idxs
        for var_z in var_mk_idxs
            if var_y == var_z
                continue
            end
            cond_set = setdiff(var_mk_idxs, [var_y, var_z])

            if rsl.ci_test(var_idx, var_y, cond_set, rsl.data)
                # var_y is a co-parent and thus NOT a neighbor
                neighbors[var_y] = 0

                # var_z is a common child of var_idx and var_y (var_idx -> var_z <- var_y)
                # var_i at index i is a common child of var_idx and i (var_idx -> var_i <- i)
                common_child_arr[var_y] = var_z
                break
            end
        end
    end

    # Remove all variables that are not neighbors
    neighbors_idx_arr = findall(neighbors)
    return neighbors_idx_arr
end

"""
    is_removable(rsl::RSL, var_idx::Int)::Bool

Check whether a variable is removable using Lemma 3 of the RSL paper.

# Arguments
- `rsl::RSL`: The RSL object.
- `var_idx::Int`: Index of the variable to check.

# Returns
- `Bool`: `true` if the variable is removable, `false` otherwise.
"""
function is_removable(rsl, var_idx::Int)::Bool
    var_mk_arr = @view rsl.markov_boundary_matrix[:, var_idx]
    var_mk_idxs = findall(var_mk_arr)

    # Use Lemma 3 of rsl paper: var_y_idx is Y and var_z_idx is Z. cond_set is Mb(X) + {X} - {Y, Z}
    @inbounds for mb_idx_y in 1:(length(var_mk_idxs) - 1)  # -1 because no need to check last variable and also symmetry
        for mb_idx_z in (mb_idx_y + 1):length(var_mk_idxs)
            var_y_idx = var_mk_idxs[mb_idx_y]
            var_z_idx = var_mk_idxs[mb_idx_z]
            cond_set = setdiff(var_mk_idxs, [var_y_idx, var_z_idx])
            push!(cond_set, var_idx)

            if rsl.ci_test(var_y_idx, var_z_idx, cond_set, rsl.data)
                return false
            end
        end
    end
    return true
end

"""
    find_removable!(rsl::RSL, var_idx_list::AbstractVector{Int})::Int

Find a removable variable in the given list of variables.

# Arguments
- `rsl::RSL`: The RSL object.
- `var_idx_list::AbstractVector{Int}`: List of variable indices to check.

# Returns
- `Int`: Index of the removable variable, or `REMOVABLE_NOT_FOUND` if no removable variable is found.
"""
function find_removable!(rsl::RSL, var_idx_list::AbstractVector{Int})::Int
    for var_idx in var_idx_list
        if is_removable(rsl, var_idx)
            return var_idx
        end
        rsl.skip_rem_check_vec[var_idx] = true
    end
    return REMOVABLE_NOT_FOUND
end

"""
    update_markov_boundary_matrix!(rsl::RSL, var_idx::Int, var_neighbors::AbstractVector{Int}, coparent_vec::AbstractVector{Int})

Update the Markov boundary matrix after removing a variable.

# Arguments
- `rsl::RSL`: The RSL object.
- `var_idx::Int`: Index of the variable to remove.
- `var_neighbors::AbstractVector{Int}`: Array containing the the neighbors of `var_idx`.
- `coparent_vec::AbstractVector{Int}`: Vector to store which nodes are parents of the variable.
"""
function update_markov_boundary_matrix!(
        rsl::RSL, var_idx::Int, var_neighbors::AbstractVector{Int}, coparent_vec::AbstractVector{Int})
    var_markov_boundary = findall(@view rsl.markov_boundary_matrix[:, var_idx])

    # For every variable in the Markov boundary of var_idx, remove it from the Markov boundary and update flag
    rsl.markov_boundary_matrix[var_markov_boundary, var_idx] .= false
    rsl.markov_boundary_matrix[var_idx, var_markov_boundary] .= false
    rsl.skip_rem_check_vec[var_markov_boundary] .= false

    if length(var_markov_boundary) > length(var_neighbors)
        # Sufficient condition for diamond-free graphs
        return nothing
    end

    # Find nodes whose co-parent status changes
    # Only remove Y from mkvb of Z if X is their ONLY common child and they are NOT neighbors
    @inbounds for ne_idx_y in 1:(length(var_neighbors) - 1)
        for ne_idx_z in (ne_idx_y + 1):length(var_neighbors)
            var_y_idx = var_neighbors[ne_idx_y]
            var_z_idx = var_neighbors[ne_idx_z]

            var_y_markov_boundary = findall(@view rsl.markov_boundary_matrix[:, var_y_idx])
            var_z_markov_boundary = findall(@view rsl.markov_boundary_matrix[:, var_z_idx])

            if @views sum(rsl.markov_boundary_matrix[:, var_y_idx]) <
                      sum(rsl.markov_boundary_matrix[:, var_z_idx])
                cond_set = setdiff(var_y_markov_boundary, [var_z_idx])
            else
                cond_set = setdiff(var_z_markov_boundary, [var_y_idx])
            end

            if rsl.ci_test(var_y_idx, var_z_idx, cond_set, rsl.data)
                # we know that Y and Z are co-parents of X and thus NOT neighbors
                coparent_vec[var_y_idx] = var_z_idx
                
                # remove Y and Z from each other's markov boundary
                rsl.markov_boundary_matrix[var_y_idx, var_z_idx] = false
                rsl.markov_boundary_matrix[var_z_idx, var_y_idx] = false
                rsl.skip_rem_check_vec[var_y_idx] = false
                rsl.skip_rem_check_vec[var_z_idx] = false
            end
        end
    end
end

end # module RecursiveCausalDiscovery
