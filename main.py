import numpy as np
import matplotlib.pyplot as plt
import life

# Size of the population in each generation
POPULATION_SIZE = 20

# Maximum generations to run before giving up
MAX_GENERATIONS = 500

# Maximum generations to run without improvement in fitness
MAX_GENERATIONS_WITHOUT_IMPROVEMENT = 40

# Number of the best individuals to pass to the next generation
NUM_ELITISTS = 1

# Probabilities for crossover and mutations
CROSSOVER_PROBABILITY = 0.95
MUTATION_PROBABILITY = 0.2


def initialize_population():
    """ Initializes a population of POPULATION_SIZE individuals. """
    new_pop = []

    while len(new_pop) < POPULATION_SIZE:
        individual = life.GameOfLife()
        individual.loop()
        new_pop.append(individual)

    return new_pop


def calculate_fitness(individual):
    """ Calculates fitness of the individual. """
    fitness = 0
    if individual.is_methuselah:
        fitness = 2 * individual.lifespan + individual.max_population

    return fitness


def calculate_fitness_of_individuals(pop):
    """ Calculates fitness for each individual in the population. """
    for i in range(len(pop)):
        individual = pop[i]
        individual.fitness = calculate_fitness(individual)


def sum_fitness(pop):
    """ Calculates sum of fitness of all individuals in the population. """
    total_fitness = 0
    for individual in pop:
        total_fitness += individual.fitness
    return total_fitness


def select(pop):
    """ Selects an individual from the population using the roulette method. """
    total_fitness = sum_fitness(pop)
    wheel_location = np.random.uniform(0, 1) * total_fitness

    index = 0
    current_sum = calculate_fitness(pop[index])

    while current_sum < wheel_location and index < POPULATION_SIZE - 1:
        index += 1
        current_sum += calculate_fitness(pop[index])

    return pop[index]


def crossover(parent1, parent2):
    """ Performs a crossover between two individuals at probability CROSSOVER_PROBABILITY. """
    if np.random.uniform(0, 1) > CROSSOVER_PROBABILITY:
        # no crossover, choose one of the parents randomly
        if np.random.choice(['parent1', 'parent2']) == 'parent1':
            return parent1.initial_config
        else:
            return parent2.initial_config

    if np.random.choice(['horizontal', 'vertical']) == 'horizontal':
        # choose randomly crossover line within the configuration and horizontally split each of the parents
        cut = np.random.randint(0, life.CONFIG_SIZE)
        parent1_split = np.hsplit(parent1.initial_config, [cut])
        parent2_split = np.hsplit(parent2.initial_config, [cut])

        # do the crossover
        offspring1_config = np.hstack((parent1_split[0], parent2_split[1]))
        offspring2_config = np.hstack((parent2_split[0], parent1_split[1]))
    else:
        # choose randomly crossover line within the configuration and vertically split each of the parents
        cut = np.random.randint(0, life.CONFIG_SIZE)
        parent1_split = np.vsplit(parent1.initial_config, [cut])
        parent2_split = np.vsplit(parent2.initial_config, [cut])

        # do the crossover
        offspring1_config = np.vstack((parent1_split[0], parent2_split[1]))
        offspring2_config = np.vstack((parent2_split[0], parent1_split[1]))

    # randomly choose one of the offsprings and return it
    if np.random.choice(['offspring1', 'offspring2']) == 'offspring1':
        return offspring1_config
    else:
        return offspring2_config


def mutate(config):
    """ Mutates an individual's configuration at probability MUTATION_PROBABILITY."""
    if np.random.uniform(0, 1) > MUTATION_PROBABILITY:
        # no mutation, return the individual as is
        return config

    # choose randomly one of the cells within the configuration
    cell = life.random_cell()

    # flip the state of the chosen cell
    if config[cell] == life.LIVE:
        config[cell] = life.DEAD
    else:
        config[cell] = life.LIVE

    return config


def next_generation(pop):
    """ Generates the next generation of the specified population. """
    calculate_fitness_of_individuals(pop)

    # First, we pass to the next generation the best solutions so far (Elitism Method)
    pop.sort(key=lambda individual: individual.fitness, reverse=True)
    new_pop = pop[:NUM_ELITISTS]

    # Now we grow the next generation by reproduction
    while len(new_pop) < POPULATION_SIZE:
        parent1 = select(pop)
        parent2 = select(pop)
        offspring_config = crossover(parent1, parent2)
        offspring_config = mutate(offspring_config)
        offspring = life.GameOfLife(offspring_config)
        offspring.loop()
        if offspring.is_methuselah:
            offspring.fitness = calculate_fitness(offspring)
            new_pop.append(offspring)

    return new_pop


def report_pop(pop, generation, max_fitness_per_generation, mean_fitness_per_generation):
    """ Shows some graphs, then plays animation of the best solution found. """
    print(f"Stopped after {generation} generations.")
    pop.sort(key=lambda individual: individual.fitness, reverse=True)
    best = pop[0]

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    ax1.set_title("max. fitness")
    ax1.set_xlabel("generations")
    ax1.set_ylabel("fitness")
    ax1.plot(np.arange(generation), max_fitness_per_generation, linestyle='solid')
    ax2.set_title("mean fitness")
    ax2.set_xlabel("generations")
    ax2.plot(np.arange(generation), mean_fitness_per_generation, linestyle='solid')
    ax3.set_title(f"best solution\nlifespan: {best.lifespan}\nmax. pop.: {best.max_population}")
    ax3.imshow(best.initial_config, interpolation='nearest')
    plt.show()

    g = life.GameOfLife(best.initial_config)
    g.run_animation()


if __name__ == '__main__':
    print("Please wait...")

    max_fitness_per_generation = []
    mean_fitness_per_generation = []

    pop = initialize_population()

    generation = 0
    max_fitness = 0
    generations_without_improvement = 0
    while generation < MAX_GENERATIONS and generations_without_improvement < MAX_GENERATIONS_WITHOUT_IMPROVEMENT:
        new_pop = next_generation(pop)
        pop = new_pop
        generation += 1

        # check if maximal fitness was improved in this generation
        fitness = max([i.fitness for i in pop])
        if fitness > max_fitness:
            max_fitness = fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        max_fitness_per_generation.append(max_fitness)
        mean_fitness = np.average([x.fitness for x in pop])
        mean_fitness_per_generation.append(mean_fitness)
        num_methuselahs = sum([x.is_methuselah for x in pop])

    report_pop(pop, generation, max_fitness_per_generation, mean_fitness_per_generation)

