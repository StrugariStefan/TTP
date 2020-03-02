import json

class ProblemInstance :
    def __init__(self, capacity: float, no_cities: int, distances: list, values: list):
        self.capacity = capacity
        self.no_cities = no_cities
        self.distances = distances
        self.values = values

def object_decoder(obj):
    return ProblemInstance(obj['capacity'], obj['no_cities'], obj['distances'], obj['values'])

def get_problem_instances(json_file_path: str) :
    with open(json_file_path) as f:
        d = json.load(f)
    return [object_decoder(obj) for obj in list(d)]