from mushroom_learner import mushroom_learner

m = mushroom_learner()
epochs = 500
generation_size = 100
crossover_prob = 0.9
crossover_type = 2
mutation_prob = 0.05
m.run_simulation('mushroom-{}-{}-{}-{}-{}.txt'.format(epochs, generation_size, crossover_prob,
                                                     crossover_type, mutation_prob),
                 epochs, generation_size, crossover_prob, crossover_type, mutation_prob)

