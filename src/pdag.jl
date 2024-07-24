# code adapted from CausalInference.jl
# https://github.com/mschauer/CausalInference.jl/blob/master/src/pdag.jl

using Graphs

"""
    isadjacent(g, x, y)

Test if `x` and `y` are connected by a any edge in the graph `g`
(i.e. x --- y, x --> y, or x <-- y .)
"""
isadjacent(dg, v, w) = has_edge(dg, v, w) || has_edge(dg, w, v)

"""
    remove!(dg::DiGraph, x => y)

Removes directed edge.
"""
remove!(dg::Graph, e::Edge) = rem_edge!(dg, e)
remove!(dg::Graph, e::Tuple) = rem_edge!(dg, Edge(e))
remove!(dg::DiGraph, e::Pair) = rem_edge!(dg, Edge(e))

"""
    isundirected(g, edge::Edge)
    isundirected(g, x, y)

Test if `x` and `y` are connected by a undirected edge in the graph `g`.
"""
isundirected(g, x, y) = has_edge(g, x, y) && has_edge(g, y, x)
isundirected(g, edge) = has_edge(g, edge) && has_edge(g, reverse(edge))

"""
    isparent(g, x, y)

Test if `x` is a parent of `y` in the graph `g`, meaning x→y.
"""
isparent(g, x, y) = has_edge(g, x, y) && !has_edge(g, y, x)


"""
    ischild(g, x, y)

Test if `x` is a child of `y` in the graph `g`, meaning x←y.
"""
ischild(g, x, y) = !has_edge(g, x, y) && has_edge(g, y, x)

"""
    isoriented(g, edge::Edge)
    isoriented(g, x, y)

Test if `x` and `y` are connected by a directed edge in the graph `g`, either x←y OR x→y.
Can also perform the same test given an `edge`.
"""
isoriented(g, edge) = has_edge(g,edge) ⊻ has_edge(g, reverse(edge)) # xor
isoriented(g, x, y) = has_edge(g,x,y) ⊻ has_edge(g,y,x)

"""
    orientedge!(g, x, y)

Update the edge `x`-`y` to `x`→`y` in the graph `g`. 
Throws error if not `x - y`
"""
orientedge!(g, edge) = @assert isundirected(g, edge) && rem_edge!(g, reverse(edge))
orientedge!(g, x, y) = @assert isundirected(g, x, y) && rem_edge!(g, y, x)

"""
    parents(g, x)

Parents are vertices y such that there is a directed edge y --> x.
Returns sorted array.
"""
parents(g, x) = setdiff(inneighbors(g, x), outneighbors(g, x))

"""
    children(g, x)

Children of x in g are vertices y such that there is a directed edge y <-- x.
Returns sorted array.
"""
children(g, x) = setdiff(outneighbors(g, x), inneighbors(g, x))