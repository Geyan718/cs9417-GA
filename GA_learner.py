import numpy as np

class GA_learner:
    def __init__(self, epochs_size=1000, generation_size=100, mutation_probability= 0.1,
        fitness_function=None, one_point_crossover=None, two_point_crossover=None, mutation=None):
        self.epochs_size = epochs_size
        self.generation_size = generation_size
        self.mutation_probability = mutation_probability
        self.fitness_function = fitness_function
        self.one_point_crossover = one_point_crossover
        self.two_point_crossover = two_point_crossover
        self.mutation = mutation

    def grow_generation(self, hypotheses):
        # hypotheses is a list of hypotheses
        # fitness_function is a function to determine the fitness of a hypothesis

        curr_gen_fitness = np.ones(self.generation_size)
        # hypotheses are referred to by their index into the list of hypotheses
        for h in range(0, self.generation_size):
            curr_gen_fitness[h] = self.fitness_function(hypotheses[h])
    
        elite_id = np.argmax(curr_gen_fitness)[0]
        elite = (elite_id, curr_gen_fitness[elite_id])

        # selection using remainder stochastic sampling
        avg_fitness = curr_gen_fitness.sum() / generation_size
        curr_gen_fitness = curr_gen_fitness / avg_fitness

        selection = []
        int_selection = np.floor(curr_gen_fitness)

        #TODO: rest of the method

        return avg_fitness, elite, next_generation

    def ga_learn(self, initial_hypotheses):
        curr_generation = initial_hypotheses
        for e in range(0, epochs_size):
            curr_avg_fitness, curr_elite, next_generation = grow_generation(curr_generation)
            
            if (e % 50 == 0):
                print('Epoch {}: avg_fitness {}, best_fitness {}\n'.format(e, curr_avg_fitness, elite[1]))
            
            curr_generation = next_generation
        
        return curr_generation

