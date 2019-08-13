import arff
import numpy as np
import time
import bitstring
import GA_learner

class mushroom_learner:
    # chromosomes are represented by a bitstring
    # 2 bits for each value of each attribute, except for classification
    # 1 bit for classification as either poisonous or edible

    def __init__(self):
        self.attributes, self.data = self.parse_arff_file('mushroom.arff')
        self.num_data = len(self.data)
        self.ga_trainer = GA_learner(1000, 100, 0.9, 1, 0.1,
                                     self.mushroom_fitness, self.mushroom_one_point_crossover,
                                     self.mushroom_two_point_crossover, self.mushroom_mutation)
        self.attributes_encoding = []
        self.read_format_string = ''
        total_val_length = 0
        for a in self.attributes:
            value_length = len(a[1])
            self.attributes_encoding.append((a[0], value_length, [r for r in range(1, value_length+1)]))
            total_val_length += value_length
            self.read_format_string += 'bits:{}, '.format(value_length*2)
        
        # final attribute only has 1 binary value 
        # remove extra space and comma at the end as well  
        self.read_format_string = self.read_format_string[:-3]
        self.read_format_string += '1'
        self.chromosome_length = total_val_length * 2 - 1
        self.num_attributes = len(self.attributes_encoding)

        # add 1 to all encoding values, which by default in arff start from 0
        for instance in self.data:
            for a in instance:
                a += 1

    def parse_arff_file(self, file='mushroom.arff'):
        with open(file, 'r') as f:
            file_contents = arff.load(f,encode_nominal=True)
        attributes = file_contents['attributes']
        data = file_contents['data']
        return attributes, data

    def mushroom_fitness(self, chromosome):
        hypothesis = self.create_hypothesis(chromosome)
        return self.test_hypothesis(hypothesis)

    def mushroom_one_point_crossover(self, parents):
        crossover_point = np.random.randint(0, self.chromosome_length)
        parent_1 = parents[0]
        parent_2 = parents[1]
        offspring_1 = parent_1[:crossover_point] + parent_2[crossover_point:]
        offspring_2 = parent_2[:crossover_point] + parent_1[crossover_point:]
        return [offspring_1, offspring_2]

    def mushroom_two_point_crossover(self, parents):
        crossover_points = np.random.randint(0, self.chromosome_length, 2)
        if crossover_points[0] == crossover_points[1]:
            offspring_1 = parents[0].copy()
            offspring_2 = parents[1].copy()
        else:
            parent_1 = parents[0]
            parent_2 = parents[1]
            high = np.amax(crossover_points)
            low = np.amin(crossover_points)
            offspring_1 = parent_1[:low] + parent_2[low:high] + parent_1[high:]
            offspring_2 = parent_2[:low] + parent_1[low:high] + parent_2[high:]
        return [offspring_1, offspring_2]

    def mushroom_mutation(self, chromosome):
        mutation_chance = np.random.random_sample(self.chromosome_length)
        chromosome.invert(np.where(mutation_chance < self.ga_trainer.mutation_probability)[0].tolist())
        return chromosome

    def create_hypothesis(self, chromosome):
        # hypothesis is a CNF logical boolean rule
        # where rule <==> class{edible, poisonous}
        # i.e. rule iff classification / classification iff rule
        
        # hypothesis is represented as a list of disjunctive clauses
        # clauses are represented as a list of encoding values for attribute values
        # each encoding value represents a literal, meaning attribute = attribute_value
        # for the attribute value represented by the encoding value
        # positive encoding value means that the literal is in the clause
        # negative encoding value means that NOT_literal is in the clause
        # if an encoding value does not appear in the clause list, it is not in the clause at all

        # each attribute value has two bits
        # first bit is whether the literal will be in the clause
        # 0 means not in clause, 1 is in clause
        # second bit represents whether positive or negation of literal is in clause
        # 0 is negation, 1 is positive
        # final class is represented by one bit only
        # bit value corresponds directly to encoding value
        
        # clauses and encoding values are all in the same order as parsed and defined in the arff file

        attribute_blocks = chromosome.readlist(self.read_format_string)
        hypothesis = []
        for a in range(0, self.num_attributes - 1):
            clause = []
            for encoding_value in self.attributes_encoding[a][2]:
                attribute_value_code = attribute_blocks[a].read('bin:2')
                if attribute_value_code[0] == '0':
                    # value is not inlcuded
                    continue
                else:
                    if attribute_value_code[1] == '0':
                        # negation of value is in clause
                        # notated as a negative of the encoding value
                        clause.append(-encoding_value)
                    else:
                        # encoding value is added to clause
                        clause.append(encoding_value)
            hypothesis.append(clause)

        # add final classification to hypothesis
        # encoding value of classification corresponds to bit value + 1
        hypothesis.append(int(attribute_blocks[-1].read('bin:1')) + 1)
        
        return hypothesis

    def print_hypothesis(self, hypothesis):
        # return hypothesis in readable string form
        h_string = ''
        for c in range(0, self.num_attributes - 1):
            if not hypothesis[c]:
                h_string += '{}=() AND '.format(self.attributes[c][0])
            else:
                h_string += '{}=('.format(self.attributes[c][0])
                for v in hypothesis[c]:
                    # index in attributes is encoding value - 1
                    if v > 0:
                        h_string += '{} OR '.format(self.attributes[c][1][v-1])
                    else: # v < 0, negative literal
                        h_string += 'NOT {} OR '.format(self.attributes[c][1][-v-1])
                h_string = h_string[:-4]
                h_string += ') AND '
        h_string = h_string[:-4]
        h_string += self.attributes[self.num_attributes - 1][0]

        return h_string

    def test_hypothesis(self, hypothesis):
        num_pass = 0
        for instance in self.data:
            consistent = evaluate_hypothesis(hypothesis, instance)
            if consistent:
                num_pass += 1
        percentage_pass = num_pass / self.num_data * 100
        return percentage_pass

    def evaluate_hypothesis(self, hypothesis, training_instance):
        # for each attribute value in the training instance
        # check if satisfies hypothesis clause for that attribute
        # must satisfy all clauses to satisfy rule

        satisfied = True
        for a in range(0, self.num_attributes - 1):
            if not hypothesis[a]:
                # empty disjunctive clause evaluates to false
                # this is different to no clause appearing, such as given in the baseline rules
                # no clause appearing is equivalent to disjunctive clause with all literals
                satisfied = False
                break
            if training_instance[a] in hypothesis[a]:
                continue
            else:
                neg_literals = [-l for l in hypothesis[a] if l < 0]
                if not neg_literals:
                    if training_instance[a] not in neg_literals:
                        continue
                    else:
                        satisfied = False
                        break
                else:
                    # neg_literals is empty
                    # training instance value is not mentioned at all in positive
                    # hence value does not satisfy clause
                    satisfied = False
                    break

        # if rule has been satisfied, we expect classes to match
        # if rule has not been satisfied, classes should not match
        # consistent is a bool representing whether the rule and classification
        # is consistent with the training instance
        if satisfied:
            if training_instance[-1] == hypothesis[-1]:
                consistent = True
            else:
                consistent = False
        else:
            if training_instance[-1] != hypothesis[-1]:
                consistent = True
            else:
                consistent = False

        return consistent

    def create_random_chromosome(self):
        initial = bitstring.BitStream(bin='0'*self.chromosome_length)
        random_init = np.random.random_sample(self.chromosome_length)
        intial.invert(np.where(random_init < 0.5)[0].tolist())
        return initial

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
            results.write('Data format:\n ++++ for last solution\nEpoch,curr_avg_fitness,curr_best_chromosome,curr_best_fitness\n')

            initial_chromosomes = []
            initial_chromosomes.extend([create_random_chromosome()] * generation_size)

            curr_generation = initial_chromosomes
            # final_generation = self.ga_trainer.ga_learn(initial_chromosomes)
            for e in range(0, epochs):
                curr_avg_fitness, curr_elite, next_generation = self.ga_trainer.grow_generation(curr_generation)
            
                if e % 10 == 0:
                    results.write('{},{},{},{}\n'.format(str(e), str(curr_avg_fitness),
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
            results.write('{},{}'.format(best_readable, str(best_fitness)))

    def write_information(self, file_handle):
        file_handle.write('Epochs: {}\n'.format(str(self.ga_trainer.epochs_size)))
        file_handle.write('Generation size: {}\n'.format(str(self.ga_trainer.generation_size)))
        file_handle.write('Crossover probability: {}\n'.format(str(self.ga_trainer.crossover_probability)))
        file_handle.write('Crossover type: {}\n'.format(str(self.ga_trainer.crossover_type)))
        file_handle.write('Mutation probability: {}\n\n'.format(str(self.ga_trainer.mutation_probability)))