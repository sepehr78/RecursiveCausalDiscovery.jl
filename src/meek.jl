# code adapted from CausalInference.jl
# https://github.com/mschauer/CausalInference.jl/blob/master/src/meek.jl
include("pdag.jl")
using Graphs
using Combinatorics

# http://proceedings.mlr.press/v89/katz19a/katz19a-supp.pdf
# https://arxiv.org/pdf/1302.4972.pdf
"""
    meek_rules!(g; rule4=false)

Apply Meek's rules 1-3 or 1-4 with rule4=true to orient edges in a 
partially directed graph without creating cycles or new v-structures.
Rule 4 is needed if edges are compelled/preoriented using external knowledge.
"""
function meek_rules!(g; rule4=false)
    done = false
    while !done 
        done = true
        # Loop through all the edges in the graph (u-v)
        for e in collect(edges(g)) # collect iterator as we are deleting
            u, v = Pair(e)
            # We only need to update (still) undirected edges
            if isundirected(g, u, v)
                # check only case u->v, we'll check v->u later
                if meek_rule1(g, u, v) || meek_rule2(g, u, v) || meek_rule3(g, u, v) || (rule4 && meek_rule4(g, u, v))
                    # Make u→v
                    remove!(g, v => u)
                    done = false
                end
            end
        end
    end
    return g
end

"""
    meek_rule1(g, v, w)

Rule 1: Orient v-w into v->w whenever there is u->v
such that u and w are not adjacent
(otherwise a new v-structure is created.)
"""
function meek_rule1(g, v, w)
    for u in inneighbors(g, v)
        if u != w
            if !has_edge(g, v => u) && !isadjacent(g, u, w)
                return true  # v-structure is created, so orient v->w
            end
        end
    end
    return false
end

"""
    meek_rule2(g, v, w)

Rule 2: Orient v-w into v->w whenever there is a chain v->k->w
(otherwise a directed cycle is created.)
"""
function meek_rule2(g, v, w)
    outs = Int[]
    for k in outneighbors(g, v)
        if k != w && !has_edge(g, k => v)
            push!(outs, k)
        end
    end
    ins = Int[]
    for k in inneighbors(g, w)
        if k != v && !has_edge(g, w => k)
            push!(ins, k)
        end
    end
    if !is_disjoint_sorted(ins, outs)
        return true
    end
    return false 
end

"""
    meek_rule3(g, v, w)

Rule 3 (Diagonal): Orient v-w into v->w whenever there are two chains
v-k->w and v-l->w such that k and l are nonadjacent
(otherwise a new v-structure or a directed cycle is created.)
"""
function meek_rule3(g, v, w)
    fulls = Int[] # Find nodes k where v-k
    for k in outneighbors(g, v)
        if has_edge(g, k => v)  # ensure that v-k
            # Skip if not k->w (or if not l->w)
            if has_edge(g, w => k) || !has_edge(g, k => w)
                continue
            end
            push!(fulls, k)
        end
    end
    for (k, l) in combinations(fulls, 2) # FIXME: 
        if !isadjacent(g, k, l)
            return true
        end
    end
    return false
end

"""
    meek_rule4(g, v, w)

Rule 4: Orient v-w into v→w if v-k→l→w where adj(v,l) and not adj(k,w) [check].
"""
function meek_rule4(g, v, w)
    for l in inneighbors(g, w)
        has_edge(g, w => l) && continue # undirected
        !isadjacent(g, v, l) && continue # not adjacent to v      
        for k in inneighbors(g, l)
            has_edge(g, l => k) && continue # undirected
            !isundirected(g, v, k) && continue # not undirected to v
            isadjacent(g, k, w) && continue # adjacent to w
            return true
        end
    end
    return false
end

"""
    is_disjoint_sorted(u, v)

Check if the intersection of sorted collections is empty. The intersection
of empty collectios is empty.
"""
function is_disjoint_sorted(u, v)
    xa = iterate(u)
    xa === nothing && return true
    yb = iterate(v)
    yb === nothing && return true

    x, a = xa
    y, b = yb

    while true
        x == y && return false
        if x > y
            yb = iterate(v, b)
            yb === nothing && return true
            y, b = yb
        else
            xa = iterate(u, a)
            xa === nothing && return true
            x, a = xa
        end
    end
end