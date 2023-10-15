using Graphs
using Printf
using DataFrames 
using Distributions
using SpecialFunctions

# ========= Variable Class ========================================================================

struct Variable 
    name::Symbol
    r::Int
end

const Assignment = Dict{Symbol, Int}
const FactorTable = Dict{Assignment, Float64}

struct Factor 
    vars::Vector{Variable}
    table::FactorTable 
end 

variablenames(φ::Factor) = [v.name for v in φ.vars]

select(a::Assignment, varnames::Vector{Symbol}) = 
    Assignment(n=>a[n] for n in varnames)

function assignments(vars::AbstractVector{Variable})
    names = [var.name for var in vars]
    return vec([Assignment(n=>v for (n,v) in zip(names, values)) 
            for varlues in product((1:v.r for v in vars))])
end

# normalize based on L1-norm 
function normalize!(φ::Factor)
    z = sum(p for (a,p) in φ.table)
    for (a,p) in φ.table 
        φ.table[a] = p/z 
    end
    return φ
end 


# ========= Convenience Functions ================================================

Base.Dict{Symbol, V}(a::NamedTuple) where V = 
    Dict{Symbol, V}(n=>v for (n,v) in zip(keys(a), values(a)))
Base.convert(::Type{Dict{Symbol, V}}, a::NamedTuple) where V =
    Dict{Symbol, V}(a)
Base.isequal(a::Dict{Symbol,<:Any}, nt::NamedTuple) = 
    length(a) == length(nt) && 
    all(all(a[n] == v for (n,v) in zip(keys(nt), values(nt))))
        
struct SetCategorical{S}
    elements::Vector{S}
    distr::Categorical 
    
    function SetCategorical(elements::AbstractVector{S}) where S 
        weights = ones(lengths(elements))
        return new{S}(elements, Categorical(normalize(weights, 1)))
    end 
   
    function SetCategorical(elements::AbstractVector{S}, weights::AbstractVector{Float64}) where S 
        l1 = norm(weights,1) 
        if l1 < 1e-6 || isinf(l1) 
            return SetCategorical(elements) 
        end 
        distr = Categorical(normalize(weights, 1)) 
        return new{S}(elements, distr)
    end 
end 

Distributions.rand(D::SetCategorical) = D.elements[rand(D.distr)]
Distributions.rand(D::SetCategorical, n::Int) = D.elements[rand(D.distr, n)]
function Distributions.pdf(D::SetCategorical, x)
    sum(e==x ? w : 0.0 for (e,w) in zip(D.elements, D.distr.p))   # conditional ? if so do : else do 
end

# ==================================================================================================

struct BayesianNetwork
    vars::Vector{Variable}
    factors::Vector{Factor}
    graph::SimpleDiGraph{Int64}    
end

function probability(bn::BayesianNetwork, assignment) 
    subassignment(ϕ) = select(assignment, variablenames(ϕ)) 
    probability(ϕ) = get(ϕ.table, subassignment(ϕ), 0.0)
    return prod(probability(ϕ) for ϕ in bn.factors)
end 

# ================================================================================================

"""
    sub2ind(siz, x)
    Identification of which parental instantiation is relevant to a particular sample and variable 
    # Arguments
    - `siz`: a vector of size (# of instantiations) of each parents
    - `x`: a vector of the parent's instantiation 
    # Return
    - `k`: a linear index (scalar) 
"""
function sub2ind(siz, x)
    k = vcat(1, cumprod(siz[1:end-1]))
    return dot(k, x.-1) + 1
end 

"""
    statistics(vars, G, D)
    # Arguments 
    - `vars`: Variable Struct 
    - `G`: a DiGraph
    - `D`: a matrix of discrete data set (n x m, n: # of var, m: # of samples)
    # Return 
    - `M`: an array of count matrices (length n, each is q x r; q: # of instantiation of parents, r: # of instantiation of the variable)
"""
function statistics(vars, G, D::Matrix{Int})
    n = size(D,1) 
    r = [vars[i].r for i in 1:n]  # list the # of instantiation of nodes 
    q = [prod([r[j] for j in inneighbors(G, i)]) for i in 1:n]  # parent's instantiation
    M = [zeros(q[i], r[i]) for i in 1:n]
    
#     println(r)

    for o in eachcol(D)   # each data sample 
        for i in 1:n      # each node 
            k = o[i]      # r_i
            parents = inneighbors(G, i)  # parents = incoming neighbors of node i
            j = 1 
            if !isempty(parents)
                j = sub2ind(r[parents], o[parents])  # q_i; get the index of the instantiation of parents
#                 println(parents, r[parents], o[parents], j)
            end 
            M[i][j,k] += 1.0 
        end 
    end 
    return M 
end 


"""
    prior(vars, G)
    # Arguments 
    - `vars`: Variable struct
    - `G`: a DiGraph
    # Return 
    - a priori matrix of the pseudocounts (alpha_{ijk}). All elements are 1
"""
function prior(vars, G)
    n = length(vars)
    r = [vars[i].r for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G, i)]) for i in 1:n]
    return [ones(q[i], r[i]) for i in 1:n]
end 

"""
    bayesian_score_component(M, α)
    # Arguments 
    - `M`: a count matrix (m_{ijk}), i is fixed 
    - `α`: pseudocounts (alpha_{ijk}), i is fixed
    # Return 
    - `p`: component of bayseian score (summed up through q_i and r_i, i.e., each node)
"""
function bayesian_score_component(M, α) 
    p = sum(loggamma.(α + M)) 
    p -= sum(loggamma.(α)) 
    p += sum(loggamma.(sum(α,dims=2))) 
    p -= sum(loggamma.(sum(α,dims=2) + sum(M,dims=2))) 
    return p 
end

"""
    bayesian_score(vars, G, D)
    # Arguments 
    - `vars`: Variable struct
    - `G`: a DiGraph
    - `D`: a matrix of discrete data set (n x m, n: # of var, m: # of samples)
    # Return 
    - `p`: bayesian score
"""
function bayesian_score(vars, G, D) 
    n = length(vars)            # number of nodes
    M = statistics(vars, G, D) 
    α = prior(vars, G) 
    return sum(bayesian_score_component(M[i], α[i]) for i in 1:n) 
end

