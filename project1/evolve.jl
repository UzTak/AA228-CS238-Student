using Random
using ProgressMeter

struct GA_params
    population_size::Int
    selection_rate::Float64
    crossover_rate::Float64
    mutation_rate::Float64
end

function mutation(indiv)
    i, j = rand(1:length(indiv), 2)
    indiv[i], indiv[j] = indiv[j], indiv[i]
    return indiv
end

# order crossover OX1
function order_crossover(indiv1, indiv2)
    # Randomly select two distinct crossover points
    n = length(indiv1)
    start, stop = sort(sample(1:n, 2, replace = false))
    
    # Create a child with the genetic material between the crossover points
    child = zeros(Int, n)
    child[start:stop] .= indiv1[start:stop]
    
    # Fill the remaining positions with genes from the second parent in order
    pos = 1
    for i in 1:n
        if pos == start
            pos = stop + 1
        end
        
        if pos > n
            pos = 1  # Reset pos if it exceeds n
        end

        if indiv2[i] != child
            # println(pos)
            child[pos] = indiv2[i]
            pos += 1
        end

    end
    return child
end


function init_ga(vars, N)
    population = []
    for i in 1:N
        population_ = randperm(length(vars))
        push!(population, population_)
    end
    return population
end


function evaluate_population(population, vars, D)
    fitness = zeros(length(population))
    for i in 1:length(population) 
        indiv = population[i]
        method = K2Search(indiv)
        G, y = fit(method, vars, D)  
        fitness[i] = y
    end
    return fitness
end



function run_ga(vars, D, max_gen::Int, params::GA_params)

    population_size = params.population_size
    selection_rate = params.selection_rate
    crossover_rate = params.crossover_rate
    mutation_rate = params.mutation_rate

    population = init_ga(vars, population_size)

    pop_best = population[1]
    y_best = -Inf

    @showprogress dt=1 desc="Computing..." for i in 1:max_gen 
        fitness = evaluate_population(population, vars, D)

        # selection 
        num_to_keep = round(Int, population_size * selection_rate)
        # println(fitness)
        sorted_indices = sortperm(fitness, rev=true)
        sorted_population = population[sorted_indices]
        selected_population = sorted_population[1:num_to_keep]

        # at least the best 3 ones are kept
        new_population = sorted_population[1:3]

        while length(new_population) < population_size
            parent1, parent2 = selected_population[rand(1:length(selected_population), 2)]
            
            # crossover
            if rand() < crossover_rate
                child = [(order_crossover(parent1, parent2)...)...]
                # println("mutation: ", child)
            else 
                child = [(copy(parent1)...)...]
                # println("no crossover: ", child)
            end


            # mutation 
            if rand() < mutation_rate
                child = [(mutation(child)...)...]
                # println("mutation: ", child)
            # else 
            #     child = [(copy(parent)...)...]
            #     # println("no mutation: ", child)
            end

            push!(new_population, child)            
        end

        population = new_population
        
        println(population)
        println("gen $i: best BIS = $(maximum(fitness))")

        if maximum(fitness) > y_best
            pop_best = population[argmax(fitness)]
            y_best = maximum(fitness)
        end

    end

    return pop_best, y_best
end 




