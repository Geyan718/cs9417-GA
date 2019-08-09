import arff
import GA_learner

class mushroom_learner:
    def __init__(self):
        self.attributes, self.data = self.parse_arff_file('mushroom.arff')
        self.ga_trainer = GA_learner(1000, 100, 0.9, 1, 0.1,
                                     self.mushroom_fitness, self.mushroom_one_point_crossover,
                                     self.mushroom_two_point_crossover, self.mushroom_mutation)

    def parse_arff_file(self, file='mushroom.arff'):
        file_contents = arff.load(open(file, 'r'))
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

    def test_hypothesis(self, hypothesis):
        pass

    def evaluate_hypothesis(self, hypothesis, training_instance):
        pass

    def create_random_chromosome(self):
        return []

    def run_simulation(self, epochs, generation_size, 
                       crossover_probability, crossover_type, mutation_probability):
        self.ga_trainer.epochs_size = epochs
        self.ga_trainer.generation_size = generation_size
        self.ga_trainer.crossover_probability = crossover_probability
        self.ga_trainer.crossover_type = crossover_type
        self.ga_trainer.mutation_probability = mutation_probability

        initial_chromosomes = []
        initial_chromosomes.extend([create_random_chromosome()] * generation_size)

        final_generation = self.ga_trainer.ga_learn(initial_chromosomes)