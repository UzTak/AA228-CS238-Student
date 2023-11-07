using Printf
using DataFrames
using CSV
using LinearAlgebra


struct MDP 
    γ   # discount rate 
    S   # state space
    A   # action space
    T   # transition function
    R   # reward function
    TR  # transition function 
end 


""" Update scheme """

# Incremental Estimate (expectation)
mutable struct IncrementalEstimate
    μ  # mean estimate 
    α  # learning rate (function) 
    m  # num of updates 
end 

function update!(model::IncrementalEstimate, x)
    model.m += 1 
    model.μ += model.α(model.m) * (x - model.μ) 
    return model 
end 

# Q-learning 
mutable struct QLearning
    S 
    A
    γ
    Q
    α
end 

lookahead(model::QLearning, s,a) = model.Q[s,a]

function update!(model::QLearning, s,a,r,s_)
    γ, Q, α = model.γ, model.Q, model.α
    Q[s,a] += α*(r + maximum(Q[s_, :]) - Q[s,a])  # update of Q-function
    return model 
end 


# SARSA
mutable struct Sarsa
    S
    A
    γ
    Q  # action value function (initial)
    α  # learning rate 
    l  # most recent experience tuble(s,a,r)
end

lookahead(model::Sarsa, s,a) = model.Q[s,a]

function update!(model::Sarsa, s,a,r,s_)
    if model.l != nothing 
        γ, Q, α, l = model.γ, model.Q, model.α, model.l 
        model.Q[l.s, l.a] += α * (l.r + γ*Q[s,a] - Q[l.s, l.a]) 
    end 
    model.l = (s=s, a=a, r=r, s_=s_) 
end 


# Sarsa(λ)
mutable struct SarsaLambda
    S 
    A 
    γ
    Q
    N  # trace (visit count)
    α  # lr 
    λ  # trace decay rate 
    l  # most recent experinece tuple (s,a,r)
end

lookahead(model::SarsaLambda, s, a) = model.Q[s,a]

function update!(model::SarsaLambda, s, a, r, s_)
    if !isnothing(model.l)  
        γ, λ, Q, α, l = model.γ, model.λ, model.Q, model.α, model.l 
        # println(l.s, " ", l.a)
        model.N[l.s, l.a] += 1 
        δ = l.r + γ * Q[s,a] - Q[l.s, l.a]  # temporal differece update
        for s in model.S
            for a in model.A
                model.Q[s,a] += α*δ*model.N[s,a]
                model.N[s,a] *= γ*λ  # decay visit counts 
            end 
        end 
    else 
        model.N[:,:] .= 0  # initialize the visit vount 
    end 
    model.l = (s=s, a=a, r=r, s_=s_)
    return model
end 


# Gradient Q-learning (continuous space) -> can we connect to the DQN with this? 
mutable struct GradientQLearning 
    # S is continuous so not defined here
    A 
    γ
    Q 
    ∇Q   # gradient dQ/dθ 
    θ    # parameter for the Q function 
    α    # learning rate 
end 

function lookahead(model::GradientQLearning, s, a)
    return model.Q(model.θ, s, a)
end 

function update!(model::GradientQLearning, s,a,r,s_, i=nothing, h=nothing)
    A, γ, Q, θ, α = model.A, model.γ, model.Q, model.θ, model.α
    Qvec = [Q(θ, s_, a_) for a_ in A]
    # println("Qvec: ", Qvec)
    
    u = maximum(Qvec)
    # println("u: ", u)
    # println("Q(θ,s,a): ", Q(θ, s, a))
    # println("∇Q: ", model.∇Q(θ,s,a))
    Δ = (r + γ*u - Q(θ, s, a)) * model.∇Q(θ,s,a)
    # println("Δ: ", Δ)
    delta =  α * scale_gradient(Δ, 1)
    println(round.(delta; sigdigits=4))
    θ[:] += delta   # scale gradient to prevent too catastrophically big gradient 
    
    model.α *= 0.999  # learning rate decay
    return model 
end 

# gradient Q learning with Experince Replay, "batch" learning
struct ReplayGradientQLearning 
    # S is continuous so not defined here
    A 
    γ
    Q 
    ∇Q   # gradient dQ/dθ 
    θ    # parameter for the Q function 
    α    # learning rate
    buffer  # circular memory buffer
    m       # number of samples to be discarded at the gradient updates 
    m_grad  # batch size 
end 

function lookahead(model::ReplayGradientQLearning, s, a)
    return model.Q(model.θ, s, a)
end  

function update!(model::ReplayGradientQLearning, s,a,r,s_)
    A, γ, Q, θ, α = model.A, model.γ, model.Q, model.θ, model.α
    buffer, m, m_grad = model.buffer, model.m, model.m_grad
    
    if isfull(buffer)
        U(s) = maximum(Q(θ,s,a) for a in A)
        ∇Q(s,a,r,s_) = (r + r*U(s_) - Q(θ,s,a)) * model.∇Q(θ,s,a)
        Δ = mean(∇Q(s,a,r,s_) for (s,a,r,s_) in rand(bugger, m_grad))
        θ[:] += α * scale_gradient(Δ, 1)
        for i in 1:m 
            popfirst!(buffer)  # discard the experience 
        end
    else 
        push!(buffer, (s,a,r,s_))
    end 
    return model 
end 


""" Training """

# algorithm 15.9
function train_online(𝒫::MDP, model, π, h, s) 
    for i in 1:h 
        a = π(model, s) 
        s′, r = 𝒫.TR(s, a) 
        update!(model, s, a, r, s′)  # update model from the sample (s,a,r,s_)
        s = s′ 
    end 
end

function train_offline(𝒫::MDP, model::SarsaLambda, df, h) 
    for i in 1:h 
        s, a, r, s_ = sample_data(model, df)   # TODO: add ϵ after implementing the ϵ-greedy in sample_data()
        update!(model, s, a, r, s_)  # update model from the sample (s,a,r,s_)
    end 
end


function train_offline(𝒫::MDP, model::GradientQLearning, df, h) 
    for i in 1:h 
        s, a, r, s_ = sample_data(model, df)   # TODO: add ϵ after implementing the ϵ-greedy in sample_data()
        update!(model, s, a, r, s_, i, h)  # update model from the sample (s,a,r,s_)
    end 
end




""" Prevention of gradient overshooting """

scale_gradient(∇, L2_max) = min(L2_max/norm(∇), 1)*∇ 
clip_gradient(∇, a, b)    = clamp.(∇, a, b)


"""  Sampling technique """
# for offline RL, sample a tuple (s,a,r,s_) to update a model
function sample_data(model::GradientQLearning, df, ϵ1=nothing, ϵ2 = 0.2)
    row = size(df,1)
    if !isnothing(ϵ1)   # ϵ-greedy? 
        if rand() < ϵ1 
            i = rand(1:row)
            s, a, r, s_ = df.s[i], df.a[i], df.r[i], df.sp[i] 
        else 
            # TODO: greedy Q function search (discrete: look up Q table's row / continuous: ??)
            # how to define the initial state if so? 
            Qtable = model.Q

        end 
    else  # random sampling with reward bias 
        # println("random sampling")
        if rand() < ϵ2
            idx = findall(df.r .> 0)  # find all the positive rewards
            i = rand(idx) 
        else
            i = rand(1:row)
        end 
        # s, a, r, s_ = [df.s_i[i], df.s_j[i]], df.a[i], df.r[i], [df.sp_i[i], df.sp_j[i]]  # small.CSV
        s, a, r, s_ = [df.pos[i], df.vel[i]], df.a[i], df.r[i], [df.pos_[i], df.vel_[i]]  # medium.CSV
        
        # println("sampled: ", s, " ", a, " ", r, " ", s_)
        # s, a, r, s_ = df.s[i], df.a[i], df.r[i], df.sp[i] 
    end
        return s, a, r, s_
end 

# for offline RL, sample a tuple (s,a,r,s_) to update a model
function sample_data(model::SarsaLambda, df, ϵ1=nothing, ϵ2 = 0.3)
    row = size(df,1)
    if !isnothing(ϵ1)   # ϵ-greedy? 
        if rand() < ϵ1 
            i = rand(1:row)
            s, a, r, s_ = df.s[i], df.a[i], df.r[i], df.sp[i] 
        else 
            # TODO: greedy Q function search (discrete: look up Q table's row / continuous: ??)
            # how to define the initial state if so? 
            Qtable = model.Q

        end 
    else  # random sampling with reward bias 
        # println("random sampling")
        l = model.l

        if !isnothing(l)
            s,a,r,s_ = l.s, l.a, l.r, l.s_
            # s_ = gridworld_step(s,a)
        else 
            s_ = NaN  
        end 

        if isnan(s_)
            # random sampling 
            if rand() < ϵ2
                idx = findall(df.r .> 0)  # find all the positive rewards
                i = rand(idx) 
            else
                i = rand(1:row)
            end 
        else 
            # find the index of the succeeding sample 
            idx = findall(df.s .== s_) 
            i = rand(idx)
        end
        
        # s, a, r, s_ = [df.s_i[i], df.s_j[i]], df.a[i], df.r[i], [df.sp_i[i], df.sp_j[i]] 
        s, a, r, s_ = df.s[i], df.a[i], df.r[i], df.sp[i] 
        # println("sampled: ", s, " ", a, " ", r, " ", s_)

    end
    
    return s, a, r, s_
end 


# ϵ-greedy 
struct EpsilonGreedyExploration
    ϵ # probability of random arm
end
   
function (π::EpsilonGreedyExploration)(model, s)   # FIXME: this is for online stragety, we need offline version of this
    A, ϵ = π.A, π.ϵ
    if rand() < ϵ
        return rand(A)  # exploration  # FIXME: random sampling from the data 
    end 
    Q(s,a) = lookahead(model, s, a)
    return argmax(a->Q(s,a), A)   # greedy (exploitation)  # FIXME: find the max Q 
end



""" Gridworld """
function gridworld_step(s::Int,a::Int)

    if a == 1 # left 
        if s <= 10 
            s_ = NaN
        else
            s_ = s - 10 
        end
    elseif a == 2 # right
        if s >= 91 
            s_ = NaN
        else
            s_ = s + 10
        end
    elseif a == 3 # up
        if mod(s,10) == 1
            s_ = NaN 
        else 
            s_ = s - 1 
        end 
    elseif a == 4 # down
        if mod(s,10) == 0 
            s_ = NaN
        else
            s_ = s + 1
        end
    end

    return s_
end 


""" Medium """

function extract_state(s)
    vel = ( s - 1 ) ÷ 500
    pos = ( s - 1 ) % 500
    return [pos, vel] 
end 

function pack_state(pos,vel)
    return 1 + pos + 500*vel
end 

