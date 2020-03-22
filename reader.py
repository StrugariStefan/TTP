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

def read_txt_instance(txt_file_path: str) -> ProblemInstance :
    f = open(txt_file_path, 'r') 

    count = 0
    coordinates = []
    items = []
    while True: 
        count += 1
  
        line = f.readline() 
  
        if not line: 
            break

        if line.find("PROBLEM NAME") >= 0 :
            pass
        elif line.find("KNAPSACK DATA TYPE") >= 0 :
            pass
        elif line.find("DIMENSION") >= 0 :
            numOfCities = int(line.split(":")[1].strip())
            distances = [0 for _ in range(numOfCities * (numOfCities - 1) // 2)]
        elif line.find("NUMBER OF ITEMS")>= 0 :
            numOfItems = int(line.split(":")[1].strip())
            weight = [0 for _ in range(numOfItems)]
            profit = [0 for _ in range(numOfItems)]
        elif line.find("RENTING RATIO")>= 0 :
            R = float(line.split(":")[1].strip())
        elif line.find("CAPACITY OF KNAPSACK")>= 0 :
            capacity = int(line.split(":")[1].strip())
        elif line.find("MIN SPEED")>= 0 :
            v_min = float(line.split(":")[1].strip())
        elif line.find("MAX SPEED")>= 0 :
            v_max = float(line.split(":")[1].strip())
        elif line.find("EDGE_WEIGHT_TYPE")>= 0 :
            edgeWeightType = line.split(":")[1].strip()
        elif line.find("NODE_COORD_SECTION")>= 0 :
            for i in range(numOfCities) :
                line = f.readline()
                a = line.split()
                x = int(float(a[1]))
                y = int(float(a[2]))
                coordinates.append((x, y))

        elif line.find("ITEMS SECTION")>= 0 :

            for i in range(numOfItems) :
                line = f.readline()
                a = line.split()
                p = int(a[1])
                w = int(a[2])
                am = int(a[3]) - 1

                items.append((p, w, am))

    print (items)
    print (coordinates)
    print (numOfCities)
    print (numOfItems)
    print (capacity)
    print (v_min)
    print (v_max)
            