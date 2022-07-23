#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 15:18
# @Author  : Xavier Ma
# @Email   : xavier_mayiming@163.com
# @File    : NSGA_II4MOPOP.py
# @Statement :The NSGA-II for the multi-objective shortest path problem
# @Reference : Deb K, Pratap A, Agarwal S, et al. A fast and elitist multiobjective genetic algorithm: NSGA-II[J]. IEEE Transactions on Evolutionary Computation, 2002, 6(2): 182-197.
# @Reference : Ahn C W, Ramakrishna R S. A genetic algorithm for shortest path routing problem and the sizing of populations[J]. IEEE Transactions on Evolutionary Computation, 2002, 6(6): 566-579.
import copy
import random


def find_neighbor(network):
    """
    find the neighbor of each node
    :param network:
    :return: {node 1: [the neighbor nodes of node 1], ...}
    """
    nn = len(network)
    neighbor = []
    for i in range(nn):
        neighbor.append(list(network[i].keys()))
    return neighbor


def random_path_generator(source, destination, neighbor):
    """
    generate random path
    :param source: source node
    :param destination: destination node
    :param neighbor: neighbor
    :return:
    """
    path = [source]
    while path[-1] != destination:
        temp_node = path[-1]
        neighbors = neighbor[temp_node]
        node_set = []
        for node in neighbors:
            if node not in path:
                node_set.append(node)
        if node_set:
            path.append(random.choice(node_set))
        else:
            path = [source]
    return path


def cal_obj(network, path, nw):
    """
    calculate the fitness of an individual
    :param network:
    :param path:
    :param nw:
    :return:
    """
    obj = [0 for i in range(nw)]
    for i in range(len(path) - 1):
        for j in range(nw):
            obj[j] += network[path[i]][path[i + 1]][j]
    return obj


def tournament_selection(population, p_crossover):
    """
    tournament selection
    :param population:
    :param p_crossover:
    :return:
    """
    mating_pool = []
    for i in range(int(len(population) * p_crossover / 2)):
        [pop1, pop2] = random.sample(population, 2)
        if pop1['pareto rank'] < pop2['pareto rank']:
            mating_pool.append(pop1)
        elif pop1['pareto rank'] > pop2['pareto rank']:
            mating_pool.append(pop2)
        elif pop1['distance'] > pop2['distance']:
            mating_pool.append(pop1)
        else:
            mating_pool.append(pop2)
    return mating_pool


def pareto_dominated(obj1, obj2):
    """
    judge whether individual 1 is Pareto dominated by individual 2
    :param obj1: the objective of individual 1
    :param obj2: the objective of individual 2
    :return:
    """
    sum_less = 0
    for i in range(len(obj1)):
        if obj1[i] < obj2[i]:
            return False
        elif obj1[i] > obj2[i]:
            sum_less += 1
    if sum_less != 0:
        return True
    return False


def non_domination_sort(population):
    """
    non domination sort
    :param population:
    :return:
    """
    pop = len(population)
    index = 1
    pareto_rank = {index: []}
    for i in range(pop):
        population[i]['n'] = 0  # domination counter
        population[i]['s'] = []  # the set of solutions dominated by population[i]
        temp_obj = population[i]['objective']
        for j in range(pop):
            if i != j:
                temp_population = population[j]
                if pareto_dominated(temp_obj, temp_population['objective']):
                    population[i]['n'] += 1
                elif pareto_dominated(temp_population['objective'], temp_obj):
                    population[i]['s'].append(j)
        if population[i]['n'] == 0:
            pareto_rank[index].append(i)
            population[i]['pareto rank'] = index
    while pareto_rank[index]:
        pareto_rank[index + 1] = []
        q_index = index + 1
        for p in pareto_rank[index]:
            for q in population[p]['s']:
                population[q]['n'] -= 1
                if population[q]['n'] == 0:
                    pareto_rank[q_index].append(q)
                    population[q]['pareto rank'] = q_index
        index += 1
    return population


def crowding_distance_assignment(population):
    """
    crowding distance assignment
    :param population:
    :return:
    """
    pop = len(population)
    for item in population:
        item['distance'] = 0
    n_obj = len(population[0]['objective'])
    for index in range(n_obj):
        population = sorted(population, key=lambda i: i['objective'][index])
        population[0]['distance'] = 1e6
        population[-1]['distance'] = 1e6
        max_min = max(population[-1]['objective'][index] - population[0]['objective'][index], 0.01)
        for i in range(1, pop - 1):
            population[i]['distance'] += (population[i + 1]['objective'][index] - population[i]['objective'][
                index]) / max_min
    return population


def pop_sort(population, children, pop):
    """
    sort the population based on the crowding distance and Pareto rank
    :param population:
    :param children:
    :param pop:
    :return:
    """
    temp_population = copy.deepcopy(population)
    temp_population.extend(children)
    temp_population = non_domination_sort(temp_population)
    temp_population = crowding_distance_assignment(temp_population)
    temp_population = sorted(temp_population, key=lambda i: (i['pareto rank'], -i['distance']))
    return temp_population[: pop]


def cal_fitness(network, population, nw):
    """
    calculate the fitness of a population
    :param network:
    :param population:
    :param nw:
    :return:
    """
    for item in population:
        item['objective'] = cal_obj(network, item['chromosome'], nw)
    return population


def crossover(chromosome1, chromosome2):
    """
    the crossover operation of two individuals
    :param chromosome1:
    :param chromosome2:
    :return:
    """
    potential_crossing_site = []
    for i in range(1, len(chromosome1) - 1):
        for j in range(1, len(chromosome2) - 1):
            if chromosome1[i] == chromosome2[j]:
                potential_crossing_site.append([i, j])
    if potential_crossing_site:
        crossing_site = random.choice(potential_crossing_site)
        offspring1 = chromosome1[0: crossing_site[0]]
        offspring2 = chromosome2[0: crossing_site[1]]
        offspring1.extend(chromosome2[crossing_site[1]:])
        offspring2.extend(chromosome1[crossing_site[0]:])
        return offspring1, offspring2
    else:
        return chromosome1, chromosome2


def crossover_pop(mothers, fathers):
    """
    the crossover operation of the populations
    :param mothers:
    :param fathers:
    :return:
    """
    children = []
    for i in range(len(mothers)):
        offspring1, offspring2 = crossover(mothers[i]['chromosome'], fathers[i]['chromosome'])
        children.append({'chromosome': offspring1})
        children.append({'chromosome': offspring2})
    return children


def mutation(chromosome, destination, neighbor):
    """
    the mutation operation of an individual
    :param chromosome:
    :param destination:
    :param neighbor:
    :return:
    """
    temp_index = random.randint(1, len(chromosome) - 1)
    new_chromosome = chromosome[: temp_index]
    while True:
        temp_node = new_chromosome[-1]
        if temp_node == destination:
            break
        neighbors = neighbor[temp_node]
        node_set = []
        for node in neighbors:
            if node not in new_chromosome:
                node_set.append(node)
        if node_set:
            new_chromosome.append(random.choice(node_set))
        else:
            temp_index = random.randint(1, len(chromosome) - 1)
            new_chromosome = chromosome[: temp_index]
    return new_chromosome


def mutation_pop(population, p_mutation, neighbor, destination):
    """
    执行群体的变异操作
    :param population: population
    :param p_mutation: mutation probability
    :param neighbor: neighbor set
    :param destination: destination node
    :return:
    """
    for i in range(len(population)):
        if random.random() < p_mutation:
            population[i]['chromosome'] = mutation(population[i]['chromosome'], destination, neighbor)
    return population


def repair(chromosome):
    """
    the repair operation of an individual to eliminate loops
    :param chromosome:
    :return:
    """
    length = len(chromosome)
    new_chromosome = copy.deepcopy(chromosome)
    for i in range(length):
        for j in range(length):
            if i < j and chromosome[i] == chromosome[j]:
                new_chromosome = chromosome[: i]
                new_chromosome.extend(chromosome[j:])
    return new_chromosome


def repair_pop(population):
    """
    the repair operation of the population
    :param population:
    :return:
    """
    for i in range(len(population)):
        population[i]['chromosome'] = repair(population[i]['chromosome'])
    return population


def main(network, source, destination):
    """
    the main function
    :param network: {node 1: {node 2: [weight1, weight2, ...], ...}, ...}
    :param source: the source node
    :param destination: the destination node
    :return:
    """
    gen = 100  # the maximum number of generations (iterations)
    pop = 10  # population number
    p_mutation = 0.15  # mutation probability
    p_crossover = 1  # crossover probability
    neighbor = find_neighbor(network)
    nw = len(network[source][neighbor[source][0]])  # the number of objectives
    population = []
    for i in range(pop):
        temp_path = random_path_generator(source, destination, neighbor)
        population.append({
            'chromosome': temp_path,
        })
    population = cal_fitness(network, population, nw)
    population = pop_sort(population, [], pop)
    best_pop = []
    best_path = []

    # The main loop
    for iteration in range(gen):
        fathers = tournament_selection(population, p_crossover)
        mothers = tournament_selection(population, p_crossover)
        children = crossover_pop(mothers, fathers)
        children = mutation_pop(children, p_mutation, neighbor, destination)
        children = repair_pop(children)
        children = cal_fitness(network, children, nw)
        population = pop_sort(population, children, pop)
        for item in population:
            if item['pareto rank'] == 1 and item['chromosome'] not in best_path:
                best_path.append(item['chromosome'])
                best_pop.append(item)

    # Sort the results
    population = non_domination_sort(best_pop)
    result = []
    best_path = []
    for item in population:
        if item['pareto rank'] == 1 and item['chromosome'] not in best_path:
            best_path.append(item['chromosome'])
            result.append({
                'path': item['chromosome'],
                'objective': item['objective'],
            })
    return result


if __name__ == '__main__':
    test_network = {
        0: {1: [62, 50], 2: [44, 90], 3: [67, 10]},
        1: {0: [62, 50], 2: [33, 25], 4: [52, 90]},
        2: {0: [44, 90], 1: [33, 25], 3: [32, 10], 4: [52, 40]},
        3: {0: [67, 10], 2: [32, 10], 4: [54, 100]},
        4: {1: [52, 90], 2: [52, 40], 3: [54, 100]},
    }
    source_node = 0
    destination_node = 4
    print(main(test_network, source_node, destination_node))
