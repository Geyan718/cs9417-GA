import numpy as np

class GA_learner:
    # crossover functions take in a list of chromosomes, and return a list of offspring
    # mutation function takes a single chromosome, and returns a single chromosome
    def __init__(self, epochs_size=1000, generation_size=100, crossover_probability=0.9, crossover_type=1, mutation_probability= 0.1,
        fitness_function=None, one_point_crossover=None, two_point_crossover=None, mutation=None):
        self.epochs_size = epochs_size
        self.generation_size = generation_size
        self.crossover_probability = crossover_probability
        self.crossover_type = crossover_type
        self.mutation_probability = mutation_probability
        self.fitness_function = fitness_function
        self.one_point_crossover = one_point_crossover
        self.two_point_crossover = two_point_crossover
        self.mutation = mutation

    def grow_generation(self, chromosomes):
        # chromosomes is a list of chromosomes, representing hypotheses

        curr_gen_fitness = np.ones(self.generation_size)
        # chromosomes are referred to by their index into the list of chromosomes
        for c in range(0, self.generation_size):
            curr_gen_fitness[c] = self.fitness_function(chromosomes[c])
    
        elite_id = np.argmax(curr_gen_fitness)[0]
        elite = (chromosomes[elite_id], curr_gen_fitness[elite_id])

        # selection of intermediate generation using remainder stochastic sampling
        intermediate_gen = []
        avg_fitness = curr_gen_fitness.sum() / self.generation_size
        curr_gen_fitness = curr_gen_fitness / avg_fitness

        # integer portion of fitness 
        # represents number of times chromosome is copied
        selection_prob_split = np.modf(curr_gen_fitness)
        int_selection = selection_prob_split[1]
        int_selection.astype(int)

        if (int_selection.sum() > 100):
            # TODO: Sort by fitness, pick best ones
            # should be a rare if not non-occuring case
            pass
        else:
            for id in range(0, self.generation_size):
                intermediate_gen.extend([id] * int_selection[id])

            # fractional portion of fitness
            # represents probability a chromosome adds additional copies
            frac_selection = selection_prob_split[0]
            while(len(intermediate_gen) < self.generation_size):
                rand_chosen_id = np.rand.randint(0, self.generation_size)
                if frac_selection[rand_chosen_id] == 0:
                    continue
                if np.random.random_sample() < frac_selection[rand_chosen_id]:
                    intermediate_gen.append(rand_chosen_id)

        # creating next generation
        next_gen = []
        while (len(next_gen) < self.generation_size - 2):
            rand_pairing = np.random.randint(0, self.generation_size, size=2)
            # crossover is applied with a probability
            # otherwise pair is copied unchanged
            if np.random.random_sample() < self.crossover_probability:
                if self.crossover_type == 1:
                    offspring = self.one_point_crossover([chromosomes[intermediate_gen[rand_pairing[0]]],
                                                        chromosomes[intermediate_gen[rand_pairing[1]]]])
                else:
                    offspring = self.two_point_crossover([chromosomes[intermediate_gen[rand_pairing[0]]],
                                                        chromosomes[intermediate_gen[rand_pairing[1]]]])
                next_gen.extend(offspring)
            else:
                next_gen.extend([chromosomes[intermediate_gen[rand_pairing[0]]], chromosomes[intermediate_gen[rand_pairing[1]]]])
        
        # apply mutation according to mutation probability
        mutation_chance = np.random.random_sample(self.generation_size - 2)
        for m in [np.where(mutation_chance < self.mutation_probability)]:
            next_gen[m] = self.mutation(next_gen[m])

        next_gen.extend([chromosomes[elite_id]] * 2)

        return avg_fitness, elite, next_gen

    def ga_learn(self, initial_chromosomes):
        curr_generation = initial_chromosomes
        for e in range(0, epochs_size):
            curr_avg_fitness, curr_elite, next_generation = grow_generation(curr_generation)
            
            if (e % 50 == 0):
                print('Epoch {}: avg_fitness {}, best_fitness {}\n'.format(e, curr_avg_fitness, elite[1]))
            
            curr_generation = next_generation
        
        return curr_generation

