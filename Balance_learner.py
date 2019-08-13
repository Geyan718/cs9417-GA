import arff
import numpy as np
import time
import bitstring
from GA_learner import GA_learner
from NeuralNetwork import NeuralNetwork as nn 


class balance_learner:

    def __init__(self):
        self.attributes, self.data = self.parse_arff_file('balance-scale.arff')
        self.ga_trainer = GA_learner(1000, 100, 0.9, 1, 0.1,
                                     self.balance_fitness, self.balance_one_point_crossover,
                                     self.balance_two_point_crossover, self.balance_mutation)
        self.attributes_encoding = []
        self.read_format_string = ''
        total_val_length = 0
        for a in self.attributes:
            value_length = len(a[1])
            self.attributes_encoding.append((a[0], value_length, [r for r in range(0, value_length)]))
            total_val_length += value_length
            self.read_format_string += 'bits:{}, '.format(value_length*2)
        
        # final attribute only has 1 binary value 
        # remove extra space and comma at the end as well  
        self.read_format_string = self.read_format_string[:-3]
        self.read_format_string += '1'
        self.chromosome_length = 35
        self.chromosome = create_random_chromosome(chromosome_length)

    def parse_arff_file(self, file='balance-scale.arff'):
        with open(file, 'r') as f:
            file_contents = arff.load(f,encode_nominal=True)
        attributes = file_contents['attributes']
        data = file_contents['data']
        return attributes, data

    def create_random_chromosome(length):
        initial = numpy.random.normal(0,0.1,length)
        return initial

    def import_weights(self,chromosome):
        chrom = self.chromosome.copy()
        first_layer_weights = chrom[0:15].reshape(4,4)
        first_layer_bias = chrom[16:19].reshape(1,4)
        second_layer_weights = chrom[20:32].reshape(4,3)
        second_layer_bias = chrom[33:35].reshape(1,3)
        
        #pass it to feed forward 
        nn.weight_list = first_layer_weights
        output = nn.forward(nn,)


    
   

    def balance_one_point_crossover(self, parents):
        pass

    def balance_two_point_crossover(self, parents):
        pass

    def mutation(self, chromosome):
        pass

    def create_hypothesis(self, chromosome):
        
        # a hypothesis looks like:
        # LW LD RW RD B
        # 00111 10000 11111 10101 010
        # 12345 12345 12345 12345 LBR    
         
        pass    

    def run_simulation(self, file_name, epochs, generation_size, 
                       crossover_probability, crossover_type, mutation_probability):
        self.ga_trainer.epochs_size = epochs
        self.ga_trainer.generation_size = generation_size
        self.ga_trainer.crossover_probability = crossover_probability
        self.ga_trainer.crossover_type = crossover_type
        self.ga_trainer.mutation_probability = mutation_probability
        
        with open(file_name, 'w') as results:
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

    def write_information(self, file_handle):
        file_handle.write('Epochs: {}\n'.format(str(self.ga_trainer.epochs_size)))
        file_handle.write('Generation size: {}\n'.format(str(self.ga_trainer.generation_size)))
        file_handle.write('Crossover probability: {}\n'.format(str(self.ga_trainer.crossover_probability)))
        file_handle.write('Crossover type: {}\n'.format(str(self.ga_trainer.crossover_type)))
        file_handle.write('Mutation probability: {}\n\n'.format(str(self.ga_trainer.mutation_probability)))    
