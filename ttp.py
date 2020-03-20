from reader import get_problem_instances, ProblemInstance
from itertools import permutations
from math import factorial, pow
from typing import Callable
from functools import reduce

import numpy as np

def city_traversal(n: int) -> (Callable[[int], list], Callable[[list], int]) :
    if n < 1 :
        raise Exception('Invalid Permutation Length')

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

    return integer_to_permutation, permutation_to_integer

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

        return reduce(lambda cumulated, current : cumulated + (current[1] if value_vector[current[0]] else 0), enumerate(instance.values), 0)

    def traveling_salesman_objective(permutation: list, value_vector: list) -> float :
        n = instance.no_cities

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



def multi_objective_genetic_algorithm(instance: ProblemInstance) :
    pass

if __name__ == '__main__':
    print ('Deserialization started')
    instances = get_problem_instances('input_example.json')

    for instance in instances :

        encoder, decoder = city_traversal(instance.no_cities - 1)
        encoder2, decoder2 = items_values(instance.no_items)

        ko, tso, gw, dist, ica = get_objective_functions(instance)

        for i in range(factorial(instance.no_cities - 1)) :
            for j in range(2 ** instance.no_items) :
                perm = encoder(i)
                val_vec = encoder2(j)
                
                print (perm)
                print (val_vec)
                print (tso(perm, val_vec))

        # print (dist)
        # print (ica)

        # for i in range(2 ** instance.no_items) :
        #     print (val_vec)
        #     print (ko(val_vec))


