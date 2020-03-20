import json

class ProblemInstance :
    def __init__(self, capacity: float, no_cities: int, no_items: int, distances: list, values: list, weights: list, v_min: float, v_max: float, item_city_assignment: list) :
        self.capacity = capacity
        self.no_cities = no_cities
        self.distances = distances
        self.values = values
        self.weights = weights
        self.v_min = v_min
        self.v_max = v_max
        self.no_items = no_items
        self.item_city_assignment = item_city_assignment
    def __repr__(self) -> str:
        return str(self.__dict__)

def object_decoder(obj) -> ProblemInstance:
    return ProblemInstance(obj['capacity'], obj['no_cities'], obj['no_items'], obj['distances'], obj['values'], obj['weights'], obj['v_min'], obj['v_max'], obj['item_city_assignment'])

def get_problem_instances(json_file_path: str) -> list :
    with open(json_file_path) as f :
        d = json.load(f)
    return [object_decoder(obj) for obj in list(d)]