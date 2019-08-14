import arff
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from GA_learner import GA_learner
import NeuralNetwork as nn


class balance_learner:

    def __init__(self):
        self.attributes, self.data = self.parse_arff_file('balance-scale.arff')
        self.ga_trainer = GA_learner(1000, 100, 0.9, 1, 0.1,
                                     self.balance_fitness, self.balance_one_point_crossover,
                                     self.balance_two_point_crossover, self.balance_mutation)
        self.chromosome_length = 35

        # one hot encoding for data
        ct = ColumnTransformer([('one_hot', OneHotEncoder(), [4])],remainder='passthrough',sparse_threshold=0)
        tf_d = ct.fit_transform(self.data)
        self.training_instances = tf_d[:,3:]
        self.training_labels = tf_d[:,0:3]

    def parse_arff_file(self, file='balance-scale.arff'):
        with open(file, 'r') as f:
            file_contents = arff.load(f, encode_nominal = True)
        attributes = file_contents['attributes']
        data = file_contents['data']
        return attributes, data

    def create_random_chromosome(self):
        initial = np.random.normal(0,0.1,self.chromosome_length)
        return initial

    def get_weights_and_biases(self, chromosome):
        first_layer_weights = chromosome[:16].reshape(4,4)
        first_layer_bias = chromosome[16:20].reshape(1,4)
        second_layer_weights = chromosome[20:32].reshape(4,3)
        second_layer_bias = chromosome[32:].reshape(1,3)
        weights_list = [first_layer_weights, second_layer_weights]
        biases_list = [first_layer_bias, second_layer_bias]
        return weights_list, biases_list

    def balance_fitness(self, chromosome):
        weights, biases = self.get_weights_and_biases(chromosome)
        predictions = nn.forward(self.training_instances, weights, biases)
        loss = nn.cross_entropy(predictions, self.training_labels)
        return -loss

    def balance_one_point_crossover(self, parents):
        crossover_point = np.random.randint(0, self.chromosome_length)
        parent_1 = parents[0]
        parent_2 = parents[1]
        offspring_1 = np.concatenate((parent_1[:crossover_point], parent_2[crossover_point:]))
        offspring_2 = np.concatenate((parent_2[:crossover_point], parent_1[crossover_point:]))
        return [offspring_1, offspring_2]

    def balance_two_point_crossover(self, parents):
        crossover_points = np.random.randint(0, self.chromosome_length, 2)
        if crossover_points[0] == crossover_points[1]:
            offspring_1 = parents[0].copy()
            offspring_2 = parents[1].copy()
        else:
            parent_1 = parents[0]
            parent_2 = parents[1]
            high = np.amax(crossover_points)
            low = np.amin(crossover_points)
            offspring_1 = np.concatenate((parent_1[:low], parent_2[low:high], parent_1[high:]))
            offspring_2 = np.concatenate((parent_2[:low], parent_1[low:high], parent_2[high:]))
        return [offspring_1, offspring_2]

    def balance_mutation(self, chromosome):

        mutation_chance = np.random.random_sample(self.chromosome_length)

        mutation_list = np.where(mutation_chance < self.ga_trainer.mutation_probability)[0].tolist()
        for index in mutation_list:
            increment = 1 if np.random.random() < 0.5 else -1
            chromosome[index] += increment*0.1
        
        return chromosome
    
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
            results.write('Data format:\n++++ before last solution\nEpoch,curr_avg_fitness,curr_best_fitness\n\n')

            initial_chromosomes = []
            for i in range(0, generation_size):
                initial_chromosomes.append(self.create_random_chromosome())

            curr_generation = initial_chromosomes
            # final_generation = self.ga_trainer.ga_learn(initial_chromosomes)
            for e in range(0, epochs):
                curr_avg_fitness, curr_elite, next_generation = self.ga_trainer.grow_generation(curr_generation)
            
                if e % 10 == 0:
                    results.write('{},{},{}\n'.format(e, curr_avg_fitness, curr_elite[1]))
                    print('Epoch {}: avg_fitness {}, best_fitness {}'.format(e, curr_avg_fitness, curr_elite[1]))
                curr_generation = next_generation

            final_fitness = self.ga_trainer.find_gen_fitness(curr_generation)
            best_id = np.argmax(final_fitness)
            best_fitness = final_fitness[best_id]
            best_chromosome = curr_generation[best_id]
        
            results.write('++++\n')
            results.write('{}\n{}'.format(best_fitness, best_chromosome))

    def write_information(self, file_handle):
        file_handle.write('Epochs: {}\n'.format(str(self.ga_trainer.epochs_size)))
        file_handle.write('Generation size: {}\n'.format(str(self.ga_trainer.generation_size)))
        file_handle.write('Crossover probability: {}\n'.format(str(self.ga_trainer.crossover_probability)))
        file_handle.write('Crossover type: {}\n'.format(str(self.ga_trainer.crossover_type)))
        file_handle.write('Mutation probability: {}\n\n'.format(str(self.ga_trainer.mutation_probability)))    
