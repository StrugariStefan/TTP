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
        k %= factorial(n)

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
        k %= factorial(n)

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
        m = instance.no_items

        if get_weights(value_vector) > instance.capacity:
            return sum(instance.distances) / instance.v_min

        def velocity(quantity: int) -> float :
            # print ('velocity')
            # print (quantity)
            # print (instance.capacity)
            # print (instance.v_max - quantity / instance.capacity * (instance.v_max - instance.v_min))
            return instance.v_max - quantity / instance.capacity * (instance.v_max - instance.v_min)

        def get_weights_until_current_city(partial_permutation: list) -> int :
            i = len(partial_permutation)
            total_weight = 0
            # print ('Permutation')
            # print (partial_permutation)
            for k in range(i) :
                for j in range(m) :
                    total_weight += value_vector[j] * instance.weights[j] * assignments.item((j, partial_permutation[k]))
                    # print ('Weight:' + str(total_weight))

            return total_weight

        distance_cost = 0
        for i in range(n - 1) :
            distance_cost += distances[permutation[i], permutation[i + 1]] / velocity(get_weights_until_current_city(permutation[:i+1]))
            # print (distance_cost)

        distance_cost += distances[permutation[n - 1], permutation[0]] / velocity(get_weights_until_current_city(permutation))

        return round(distance_cost, 2)

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

def NSGA_II(instance: ProblemInstance) :
    n = instance.no_cities
    m = instance.no_items

    ko, tso, w, d, a = get_objective_functions(instance)
    int_to_perm, perm_to_int, dist_into_boolean_vector, boolean_vector_to_dist = city_traversal(n - 1)
    int_to_decision_vector, decision_vector_to_int = items_values(m)

    inf_tso = sum(instance.distances) / instance.v_min
    inf_ko = 0
    
    def apply_operators(population: list, crossover: Callable[[tuple], tuple], mutation: Callable[[list], list]) -> list :
        encoded_population = list(map(lambda element: dist_into_boolean_vector(element[0]) + int_to_decision_vector(element[1]), population))

        k = len(encoded_population)
        rate = 1e-1
        encoded_offspring = []
        
        np.random.shuffle(encoded_population)

        for index in range(0, k, 2) :
            o1, o2 = crossover((encoded_population[index], encoded_population[index + 1]))
            o1 = mutation(o1, rate)
            o2 = mutation(o2, rate)
            encoded_offspring.append(o1)
            encoded_offspring.append(o2)

        offspring = list(map(lambda element: (boolean_vector_to_dist(element[:n - 1]), decision_vector_to_int(element[n - 1:])), encoded_offspring))

        return offspring

    size = int(ceil(log2(factorial(n - 1) * 2 ** m)) * 2)
    population = get_random_population(size, n, m)
    offspring = apply_operators(population, crossover, mutation)

    def get_objective_limits(population: list) -> tuple :
        ztso = list(map(lambda s: tso(int_to_perm(s[0]), int_to_decision_vector(s[1])), population))
        zko = list(map(lambda  s: ko(int_to_decision_vector(s[1])), population))

        max_tso = max(ztso)
        min_tso = min(ztso)

        max_ko = max(zko)
        min_ko = min(zko)            

        return (max_tso, min_tso, max_ko, min_ko)

    def compute_crowding_distance(pareto_front: set) -> dict :
        tso_sorted_solutions = list(map(lambda s: (s, tso(int_to_perm(s[0]), int_to_decision_vector(s[1]))), pareto_front.copy())) 
        tso_sorted_solutions.sort(key = lambda so : so[1])

        ko_sorted_solutions = list(map(lambda s : (s, ko(int_to_decision_vector(s[1]))), pareto_front.copy()))
        ko_sorted_solutions.sort(key = lambda so : so[1])
            
        cd_tso = [inf_tso for _ in tso_sorted_solutions]
        cd_ko = [inf_ko for _ in ko_sorted_solutions]

        l = len(cd_tso)
        cd = { p : 0.0 for p in pareto_front }

        cd[ko_sorted_solutions[0][0]] += cd_ko[0]
        cd[tso_sorted_solutions[0][0]] += cd_tso[0]
        cd[ko_sorted_solutions[l - 1][0]] += cd_ko[l - 1]
        cd[tso_sorted_solutions[l - 1][0]] += cd_tso[l - 1]

        for i in range(1, l - 1) :
            cd_ko[i] = (ko_sorted_solutions[i + 1][1] - ko_sorted_solutions[i - 1][1]) / (ko_max - ko_min)
            cd_tso[i] = (tso_sorted_solutions[i + 1][1] - tso_sorted_solutions[i - 1][1]) / (tso_max - tso_min)

            cd[tso_sorted_solutions[i][0]] += cd_tso[i]
            cd[ko_sorted_solutions[i][0]] += cd_ko[i]

        sorted_by_crowding_distance_pf = sorted(cd.keys(), key = lambda k : cd[k], reverse = True)

        return sorted_by_crowding_distance_pf

    def fast_non_dominated_sort(population: list) -> list :
        def dominates(p: tuple, q: tuple) -> bool :
            pr1 = int_to_perm(p[0])
            pr2 = int_to_decision_vector(p[1])
            qr1 = int_to_perm(q[0])
            qr2 = int_to_decision_vector(q[1])

            pz1 = tso(pr1, pr2)
            pz2 = ko(pr2)
            qz1 = tso(qr1, qr2)
            qz2 = ko(qr2)

            if pz1 <= qz1 and pz2 <= qz2 :
                if pz1 < qz1 or pz2 < qz2 :
                    return True
            
            return False

        F1 = set()    
        S = {}
        n = {}

        for p in population :
            S[p] = set()
            n[p] = 0
            for q in population :
                if dominates(p, q) :
                    S[p].add(q)
                elif dominates(q, p) :
                    n[p] += 1
            if n[p] == 0 :
                F1.add(p)

        i = 0
        F = F1
        pareto_fronts = []

        while len(F) != 0 :
            pareto_fronts.append(F.copy())
            H = set()
            for p in F :
                for q in S[p] :
                    n[q] -= 1
                    if n[q] == 0 :
                        H.add(q)
            F = H

        return pareto_fronts

    t = 0
    p = population
    q = offspring

    while t < 10:
        # print ('Epoch:' + str(t))
        p = set(p)
        q = set(q)
        r = list(p.union(q))

        (tso_max, tso_min, ko_max, ko_min) = get_objective_limits(r)
        ordered_pareto_fronts = fast_non_dominated_sort(r)
        p = set()

        for pf in ordered_pareto_fronts :
            if len(p) + len(pf) <= size :
                p |= pf
            else :
                sorted_by_crowding_distance_pf = compute_crowding_distance(pf)

                p |= set(sorted_by_crowding_distance_pf[: size - len(p)])
                break

        q = apply_operators(list(p), crossover, mutation)
        t += 1

    return p

if __name__ == '__main__':
    instances = get_problem_instances('input_example.json')

    for instance in instances[:1] :
        p = NSGA_II(instance)


