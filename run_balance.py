from Balance_learner import balance_learner

b = balance_learner()
epochs = 1000
generation_size = 100
crossover_prob = 0.9
crossover_type = 2
mutation_prob = 0.05
b.run_simulation('balance-{}-{}-{}-{}-{}.txt'.format(epochs, generation_size, crossover_prob,
                                                     crossover_type, mutation_prob),
                 epochs, generation_size, crossover_prob, crossover_type, mutation_prob)
