from reader import get_problem_instances, ProblemInstance
from itertools import permutations
from math import factorial, pow
from typing import Callable
from functools import reduce

def can_item_be_stolen_for_city(city) :
    pass

def get_weight_for_city(city_index, city_traversal, items_decision_vector) :
    pass

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
            
        

class TTP :
    def __init__(self, instance: ProblemInstance) :
        self.instance: ProblemInstance = instance

    def encode_traversal_into_genotype(permutation: list) :
        pass

        


    def decode_genotypy_into_traversal() :
        pass

    def encode() : 
        pass


    def get_decision_vector() :
        pass


if __name__ == '__main__':
    print ('Deserialization started')
    instaces = get_problem_instances('input_example.json')
    print (instaces)

    encoder, decoder = city_traversal(3)

    for i in range(factorial(3)) :
        perm = encoder(i)
        print (perm)
        print (decoder(perm))

    encoder2, decoder2 = items_values(3)
    for i in range(2 ** 3) :
        val_vec = encoder2(i)
        print (val_vec)
        print (decoder2(val_vec))

