import numpy as np

def grow_generation(hypotheses, fitness_function, generation_size = 100):
    # hypotheses is a list of hypotheses
    # fitness_function is a function to determine the fitness of a hypothesis

    curr_gen_fitness = np.ones(generation_size)
    # hypotheses are referred to by their index into the list of hypotheses
    for h in range(0, generation_size):
        curr_gen_fitness[h] = fitness_function(hypotheses[h])
    
    # selection using remainder stochastic sampling
    avg_fitness = curr_gen_fitness.sum() / generation_size
    curr_gen_fitness = curr_gen_fitness / avg_fitness

    int_selection =  np.floor(curr_gen_fitness)



