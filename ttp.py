from reader import get_problem_instances, ProblemInstance, Problem, read_txt_instance
from itertools import permutations
from math import factorial, pow, ceil, log2, sqrt
from typing import Callable
from functools import reduce

import numpy as np
import random
import time

tso_solutions = {}
ko_solutions = {}
weights = {}
profit = {}

assigned_items = {}

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
        if len(permutation) != n + 1 :
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

def items_values(capacity: int, items: list) -> (Callable[[int], list], Callable[[list], int]) :
    m = len(items)
    
    if m < 1 :
        raise Exception('Invalid Length')

    memoized_combinations = {}

    def combinations(w: int, item_weight_list: tuple) -> list :
        if (w, item_weight_list) not in memoized_combinations :
            if len(item_weight_list) == 0 :
                return [[]]
            
            x = item_weight_list[0]
            xs = item_weight_list[1:]
            if w >= x[1] :
                result = [ [x] + ys for ys in combinations(w - x[1], xs)] + combinations(w, xs)
            else : 
                result = combinations(w, xs)
            memoized_combinations[(w, item_weight_list)] = result
        
        return memoized_combinations[(w, item_weight_list)]

    def get_profit_vector(item_combination: list) -> list :
        profit_vector = [False for _ in range(m)]
        for item_index in item_combination :
            profit_vector[item_index] = True

        return profit_vector

    def get_combination(value_vector: list) -> list :
        k = 0
        for i in range(m) :
            if value_vector[n - i - 1] :
                k += 2 ** i

        if k >= number_of_combinations :
            k %= number_of_combinations

        return valid_item_combinations[k]

    valid_item_combinations = combinations(capacity, tuple(map(lambda x: (x[0], x[1][1]), enumerate(items))))
    valid_item_combinations = list(map(lambda x: list(map(lambda y: y[0], x)), valid_item_combinations))

    number_of_combinations = len(valid_item_combinations)

    n = ceil(log2(number_of_combinations))
    
    def integer_to_value_vector(k: int) -> list :
        k %= number_of_combinations
        value_vector = get_profit_vector(valid_item_combinations[k])

        return value_vector

    def value_vector_to_integer(value_vector: list) -> int :
        if len(value_vector) != m :
            raise Exception('Invalid value vector') 
        
        combination = get_combination(value_vector)
        integer_reprezentation = valid_item_combinations.index(combination)        

        return integer_reprezentation

    return integer_to_value_vector, value_vector_to_integer

def get_objective_functions(instance: Problem) :

    def compare_nodes(n1: int, n2: int) -> list :
        x = instance.coordinates[n1]
        y = instance.coordinates[n2]
        
        dist = ceil(sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2))
        return dist

    def get_item(item_index: int) -> tuple :
        return instance.items[item_index]

    def get_assigned_items(node: int) -> list :
        global assigned_items

        if node not in assigned_items :
            assigned_items[node] = list(filter(lambda ex: ex[1][2] == node, enumerate(instance.items)))

        return assigned_items[node]

    def isItemAssigned(item_index: int, node: int) -> bool :
        return instance.items[item_index][2] == node


    def knapsack_objective(value_vector: list) -> int :
        global profit
        value_vector = tuple(value_vector[:])

        if len(instance.items) != len(value_vector) :
            raise Exception('Incompatible values with value vector length')

        if get_weights(value_vector) > instance.capacity:
            return 0
        
        if value_vector not in profit :
            total_profit = - reduce(lambda cumulated, current : cumulated + (get_item(current[0])[0] if current[1] else 0), enumerate(value_vector), 0)
            profit[value_vector] = total_profit

        return profit[value_vector]

    def traveling_salesman_objective(permutation: list, value_vector: list) -> float :
        n = instance.no_cities
        m = instance.no_items

        global tso_solutions

        permutation = tuple(permutation[:])
        value_vector = tuple(value_vector[:])

        if (permutation, value_vector) not in tso_solutions :

            if get_weights(value_vector) > instance.capacity:
                return float("inf")

            def velocity(quantity: int) -> float :
                return instance.v_max - quantity / instance.capacity * (instance.v_max - instance.v_min)

            def get_weights_until_current_city(partial_permutation: list) -> int :
                i = len(partial_permutation)
                total_weight = 0
                for k in range(i) :
                    for j in range(m) :
                        total_weight += value_vector[j] * get_item(j)[1] * isItemAssigned(j, partial_permutation[k])

                return total_weight

            def get_cumulated_capacity(val_vec: list, perm: list) -> list :
                s = 0
                cummulated_capacity = []
                for c in perm :
                    items = get_assigned_items(c)
                    for index, item in items :
                        s += value_vector[index] * item[1]
                    cummulated_capacity.append(s)
                return cummulated_capacity

            ccap = get_cumulated_capacity(value_vector, permutation)

            distance_cost = 0
            for i in range(n - 1) :
                distance_cost += compare_nodes(permutation[i], permutation[i + 1]) / velocity(ccap[i])

            distance_cost += compare_nodes(permutation[n - 1], permutation[0]) / velocity(ccap[n - 1])

            tso_obj = round(distance_cost, 2)
            tso_solutions[(permutation, value_vector)] = tso_obj

        return tso_solutions[(permutation, value_vector)]
        

    def get_weights(value_vector: list) -> int :
        global weights
        value_vector = tuple(value_vector[:])

        if len(instance.items) != len(value_vector) :
            raise Exception('Incompatible weights with value vector length')

        if value_vector not in weights :
            total_weight = reduce(lambda cumulated, current : cumulated + (get_item(current[0])[1] if current[1] else 0), enumerate(value_vector), 0)
            weights[value_vector] = total_weight

        return weights[value_vector]

    return knapsack_objective, traveling_salesman_objective, get_weights, compare_nodes, get_item, get_assigned_items

def get_random_population(size: int, n: int, m: int) -> list :
    population = []
    for i in range(size) :
        permutation_repr = random.randrange(start = 0, stop = factorial(n - 1), step = 1)
        value_vector_repr = random.randrange(start = 0, stop = 2 ** m, step = 1)
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

def NSGA_II(instance: Problem) :
    n = instance.no_cities
    m = instance.no_items

    ko, tso, w, d, a, get_items_for_node = get_objective_functions(instance)
    int_to_perm, perm_to_int, dist_into_boolean_vector, boolean_vector_to_dist = city_traversal(n - 1)
    int_to_decision_vector, decision_vector_to_int = items_values(instance.capacity, instance.items)

    inf_tso = float("inf")
    inf_ko = 0
    
    def apply_operators(population: list, crossover: Callable[[tuple], tuple], mutation: Callable[[list], list]) -> list :
        encoded_population = list(map(lambda element: dist_into_boolean_vector(element[0]) + int_to_decision_vector(element[1]), population))

        ai = ceil(log2(factorial(n - 1)))

        k = len(encoded_population)
        rate = 1e-3
        encoded_offspring = []
        
        np.random.shuffle(encoded_population)

        for index in range(0, k, 2) :
            o1, o2 = crossover((encoded_population[index], encoded_population[index + 1]))
            o1 = mutation(o1, rate)
            o2 = mutation(o2, rate)
            encoded_offspring.append(o1)
            encoded_offspring.append(o2)

        offspring = list(map(lambda element: (boolean_vector_to_dist(element[:ai]), decision_vector_to_int(element[ai:])), encoded_offspring))

        return offspring

    size = min(int(ceil(log2(factorial(n - 1) * 2 ** m)) * 2), 500)

    print ('Population size: ' + str(size))
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
    T_MAX = 5
    p = population
    q = offspring

    while t < T_MAX:
        print ('Epoch:' + str(t))
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
        yield p
        t += 1

from reader import read_txt_instance

if __name__ == '__main__':
    from utils import run_ndga, plot

    # output_file_generator = run_ndga('test-example-n4')
    output_file_generator = run_ndga('a20-n20')

    plot(list(output_file_generator))




