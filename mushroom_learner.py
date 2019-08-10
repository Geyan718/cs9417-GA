import arff
import numpy as np
import time
import GA_learner

class mushroom_learner:
    def __init__(self):
        self.attributes, self.data = self.parse_arff_file('mushroom.arff')
        self.ga_trainer = GA_learner(1000, 100, 0.9, 1, 0.1,
                                     self.mushroom_fitness, self.mushroom_one_point_crossover,
                                     self.mushroom_two_point_crossover, self.mushroom_mutation)

    def parse_arff_file(self, file='mushroom.arff'):
        with open(file, 'r') as f:
            file_contents = arff.load(f)
        attributes = file_contents['attributes']
        data = file_contents['data']
        return attributes, data

    def mushroom_fitness(self, chromosome):
        pass

    def mushroom_one_point_crossover(self, parents):
        return parents

    def mushroom_two_point_crossover(self, parents):
        return parents

    def mushroom_mutation(self, chromosome):
        return chromosome

    def create_hypothesis(self, chromosome):
        pass

    def print_hypothesis(self, h):
        # return hypothesis in readable string form
        pass

    def test_hypothesis(self, hypothesis):
        pass

    def evaluate_hypothesis(self, hypothesis, training_instance):
        pass

    def create_random_chromosome(self):
        return []

    def run_simulation(self, file_name, epochs, generation_size, 
                       crossover_probability, crossover_type, mutation_probability):
        self.ga_trainer.epochs_size = epochs
        self.ga_trainer.generation_size = generation_size
        self.ga_trainer.crossover_probability = crossover_probability
        self.ga_trainer.crossover_type = crossover_type
        self.ga_trainer.mutation_probability = mutation_probability
        
        results = open(file_name, 'w')
        self.write_information(results)
        # TODO: time logging
        results.write('Data format:\n==== or ++++ for last solution\nEpoch\ncurr_avg_fitness\ncurr_best_chromosome\ncurr_best_fitness\n')

        initial_chromosomes = []
        initial_chromosomes.extend([create_random_chromosome()] * generation_size)

        curr_generation = initial_chromosomes
        # final_generation = self.ga_trainer.ga_learn(initial_chromosomes)
        for e in range(0, epochs):
            curr_avg_fitness, curr_elite, next_generation = self.ga_trainer.grow_generation(curr_generation)
            
            if e % 10 == 0:
                results.write('====\n')
                results.write('{}\n{}\n{}\n{}\n'.format(str(e), str(curr_avg_fitness), 
                                                        self.print_hypothesis(curr_elite[0]), str(curr_elite[1])))
                print('Epoch {}: avg_fitness {}, best_fitness {}\n'.format(e, curr_avg_fitness, elite[1]))
            curr_generation = next_generation

        final_fitness = self.ga_trainer.find_gen_fitness(curr_generation)
        best_id = np.argmax(final_fitness)[0]
        best_fitness = final_fitness[best_id]
        best_chromosome = curr_generation[best_id]
        best_hypothesis = self.create_hypothesis(best_chromosome)
        best_readable = self.print_hypothesis(best_hypothesis)
        
        results.write('++++\n')
        results.write('{}\n{}'.format(best_readable, str(best_fitness)))
        results.close()


    def write_information(self, file_handle):
        file_handle.write('Epochs: {}\n'.format(str(self.ga_trainer.epochs_size)))
        file_handle.write('Generation size: {}\n'.format(str(self.ga_trainer.generation_size)))
        file_handle.write('Crossover probability: {}\n'.format(str(self.ga_trainer.crossover_probability)))
        file_handle.write('Crossover type: {}\n'.format(str(self.ga_trainer.crossover_type)))
        file_handle.write('Mutation probability: {}\n\n'.format(str(self.ga_trainer.mutation_probability)))