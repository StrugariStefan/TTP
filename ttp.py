from reader import get_problem_instances, ProblemInstance
from itertools import permutations
from math import factorial, pow, ceil, log2
from typing import Callable
from functools import reduce

import numpy as np

def city_traversal(n: int) -> (Callable[[int], list], Callable[[list], int], Callable[[int], list], Callable[[list], int]) :
    if n < 1 :
        raise Exception('Invalid Permutation Length')

    no_of_bits = ceil(log2(factorial(n)))

    def integer_to_boolean_vector(k: int) -> list :
        if k >= factorial(n) :
            raise Exception('This does not reprezint a valid permutation')

        boolean_vector = [False for _ in range(no_of_bits)]
        for i in reversed(range(no_of_bits)) :
            boolean_vector[i] = k % 2 == 1
            k = k // 2

        return boolean_vector

    def boolean_vector_to_integer(boolean_vector: list) -> int :
        if len(boolean_vector) != no_of_bits :
            raise Exception('Invalid boolean vector') 
        
        k = 0
        for i in range(no_of_bits) :
            if boolean_vector[n - i - 1] :
                k += 2 ** i

        return k

    def integer_to_permutation(k: int) -> list:
        if k >= factorial(n) :
            raise Exception('Permutation does not exist')

        permutation = [0 for _ in range(n)]
        elements = [i for i in range(n)]
        
        m = k

        for i in range(n) :
            ind = m % (n - i)
            m = m // (n - i)
            permutation[i] = elements[ind]
            elements[ind] = elements[n - i - 1]

        permutation = [x + 1 for x in permutation]
        permutation.insert(0, 0)
        return permutation

    def permutation_to_integer(permutation: list) -> int :
        if len(permutation) is not n + 1 :
            raise Exception('Invalid permutation')

        permutation.pop(0)
        permutation = [x - 1 for x in permutation]

        if any([index not in permutation for index in range(n)]) :
            raise Exception('Invalid permutation elements')

        positions = [i for i in range(n)]
        elements = [i for i in range(n)]
        k = 0
        m = 1

        for i in range(n):
            k += m * positions[permutation[i]]
            m = m * (n - i)
            positions[elements[n - i - 1]] = positions[permutation[i]]
            elements[positions[permutation[i]]] = elements[n - i - 1]

        return k

    return integer_to_permutation, permutation_to_integer, integer_to_boolean_vector, boolean_vector_to_integer

def items_values(n: int) -> (Callable[[int], list], Callable[[list], int]) :
    if n < 1 :
        raise Exception('Invalid Length')

    def integer_to_value_vector(k: int) -> list :
        if k >= 2 ** n :
            raise Exception('Value does not exist')

        value_vector = [False for _ in range(n)]
        for i in reversed(range(n)) :
            value_vector[i] = k % 2 == 1
            k = k // 2

        return value_vector

    def value_vector_to_integer(value_vector: list) -> int :
        if len(value_vector) is not n :
            raise Exception('Invalid value vector') 
        
        k = 0
        for i in range(n) :
            if value_vector[n - i - 1] :
                k += 2 ** i

        return k

    return integer_to_value_vector, value_vector_to_integer

def get_objective_functions(instance: ProblemInstance) :

    def get_distances(instance: ProblemInstance) -> list :
        n = instance.no_cities
        k = 0
        distances = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(0, n) :
            for j in range(i + 1, n) :
                distances[i][j] = instance.distances[k]
                distances[j][i] = instance.distances[k]
                k += 1
        return np.array(distances)

    distances = get_distances(instance)
    assignments = np.array(instance.item_city_assignment)


    def knapsack_objective(value_vector: list) -> int :
        if len(instance.values) != len(value_vector) :
            raise Exception('Incompatible values with value vector length')

        if get_weights(value_vector) > instance.capacity:
            return 0

        return - reduce(lambda cumulated, current : cumulated + (current[1] if value_vector[current[0]] else 0), enumerate(instance.values), 0)

    def traveling_salesman_objective(permutation: list, value_vector: list) -> float :
        n = instance.no_cities

        # print ('-----------')
        # print (get_weights(value_vector))
        # print (instance.capacity)
        # print (sum(instance.distances))
        # print (instance.v_min)

        if get_weights(value_vector) > instance.capacity:
            return sum(instance.distances) / instance.v_min

        def velocity(quantity: int) -> float :
            return instance.v_max - quantity / instance.capacity * (instance.v_max - instance.v_min)

        def get_weights_until_current_city(partial_permutation: list) -> int :
            i = len(partial_permutation)
            # print (partial_permutation)
            total_weight = 0
            for k in range(i) :
                for j in range(n) :
                    total_weight += value_vector[j] * instance.weights[j] * assignments.item((j, partial_permutation[k]))

            return total_weight

        distance_cost = 0
        for i in range(n - 1) :
            distance_cost += distances[permutation[i], permutation[i + 1]] / velocity(get_weights_until_current_city(permutation[:i]))

        distance_cost += distances[permutation[n - 1], permutation[0]] / velocity(get_weights_until_current_city(permutation))

        return distance_cost

    def get_weights(value_vector: list) -> int :
        if len(instance.weights) != len(value_vector) :
            raise Exception('Incompatible weights with value vector length')

        return reduce(lambda cumulated, current : cumulated + (current[1] if value_vector[current[0]] else 0), enumerate(instance.weights), 0)

    return knapsack_objective, traveling_salesman_objective, get_weights, distances, assignments

def get_random_population(size: int, n: int, m: int) -> list :
    population = []
    for i in range(size) :
        permutation_repr = np.random.randint(low = 0, high = factorial(n - 1), size = None)
        value_vector_repr = np.random.randint(low = 0, high = 2 ** m, size = None)
        population.append((permutation_repr, value_vector_repr))

    return population

def crossover(individuals: tuple) -> tuple:
    o1 = individuals[0][:]
    o2 = individuals[1][:]

    if (len(o1) != len(o2)) :
        raise Exception("Parents do not have same reprezentation")

    n = len(o1)
    slices = np.random.choice(range(n), size = 2, replace = False)
    slices = np.sort(slices)

    aux = o1[slices[0] : slices[1]]
    o1[slices[0] : slices[1]] = o2[slices[0] : slices[1]]
    o2[slices[0] : slices[1]] = aux

    return o1, o2

def mutation(individual: list, mutation_rate: float) -> list :
    mutated_individual = [False for _ in range(len(individual))]
    for index, gene in enumerate(individual) :
        if np.random.uniform(low = 0.0, high = 1.0, size = None) <= mutation_rate :
            mutated_individual[index] = not gene
        else :
            mutated_individual[index] = gene
    
    return mutated_individual

def apply_operators(population: list, crossover: Callable[[tuple], tuple], mutation: Callable[[list], list]) -> list :
    n = len(population)
    offspring = []
    
    np.random.shuffle(population)

    for index in range(0, n, 2) :
        o1, o2 = crossover((population[index], population[index + 1]))
        offspring.append(o1)
        offspring.append(o2)

    return offspring


def NSGA_II(instance: ProblemInstance) :
    n = instance.no_cities
    m = instance.no_items

    ko, tso, w, d, a = get_objective_functions(instance)
    dist_encoder, dist_decoder, dist_into_boolean_vector, boolean_vector_to_dist = city_traversal(n - 1)
    item_encoder, item_decoder = items_values(m)

    size = int(5e+1)

    p = get_random_population(size, n, m)

    encoded_population = list(map(lambda element: dist_into_boolean_vector(element[0]) + item_encoder(element[1]), p))
    encoded_offspring = apply_operators(encoded_population, crossover, mutation)

    print (encoded_offspring[0])
    t = 0
    while t < 10:
        t += 1
    # for e in p:
    #     break

    # integer = e[1]
    # print (integer)
    # value_vector = item_encoder(integer)
    # print (value_vector)
    # new_integer = item_decoder(value_vector)
    # print (new_integer)

    # o1, o2 = crossover((value_vector, item_encoder(21)))
    # print (o1)
    # print (o2)

    # mi = mutation(value_vector, 0.5)
    # print (mi)

    # integer = e[0]
    # print (integer)
    # boolean_vector = dist_into_boolean_vector(integer)
    # print (boolean_vector)
    # decoded_integer = boolean_vector_to_dist(boolean_vector)
    # print (decoded_integer)



def multi_objective_genetic_algorithm(instance: ProblemInstance) :
    pass

if __name__ == '__main__':
    print ('Deserialization started')
    instances = get_problem_instances('input_example.json')

    for instance in instances :

        NSGA_II(instance)
        # encoder, decoder = city_traversal(instance.no_cities - 1)
        # encoder2, decoder2 = items_values(instance.no_items)

        # ko, tso, gw, dist, ica = get_objective_functions(instance)

        # for i in range(factorial(instance.no_cities - 1)) :
        #     for j in range(2 ** instance.no_items) :
        #         perm = encoder(i)
        #         val_vec = encoder2(j)
                
        #         tso(perm, val_vec)
        #         ko(val_vec)
                # print ('--------------')
                # print (perm)
                # print (val_vec)
                # print (tso(perm, val_vec))
                # print (ko(val_vec))


        # print (dist)
        # print (ica)

        # for i in range(2 ** instance.no_items) :
        #     print (val_vec)
        #     print (ko(val_vec))


