import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from os import path
from ttp import NSGA_II, city_traversal, items_values, get_objective_functions
from reader import get_problem_instances, ProblemInstance, Problem, read_txt_instance

def dominates(x: tuple, y: tuple) -> bool :
    if x[0] <= y[0] and -x[1] <= -y[1] :
        if x[0] < y[0] or -x[1] < -y[1] :
            return True
    
    return False

def plot(path_list: list) :
    ovdfs = []
    for i, path in enumerate(path_list) :
        ovdf = pd.read_pickle(path)
        ovdf['Epoch'] = i
        ovdfs.append(ovdf)
    df = pd.concat(ovdfs)

    display(ovdf)

    g = sns.FacetGrid(df,  row="Epoch", hue="ParetoFront", height = 4.5, aspect = 1.7)
    g = (g.map(plt.scatter, "Time", "Profit", edgecolor="w").add_legend())

    plt.show()

def run_ndga(filename: str) :

    instance = read_txt_instance('.\\resources\\' + filename + '.txt')
    population_generator = NSGA_II(instance)

    n = instance.no_cities
    m = instance.no_items

    ko, tso, w, d, a, get_assigned_items = get_objective_functions(instance)

    int_to_perm, perm_to_int, dist_into_boolean_vector, boolean_vector_to_dist = city_traversal(n - 1)
    int_to_decision_vector, decision_vector_to_int = items_values(instance.capacity, instance.items)

    for i, population in enumerate(list(population_generator)) :
        decoded_population = list(map(lambda x : (int_to_perm(x[0]), int_to_decision_vector(x[1])), population))
        objective_values = list(map(lambda x: (tso(x[0], x[1]), -ko(x[1])), decoded_population))

        is_in_pareto_front = [True for _ in objective_values]

        for ov1 in objective_values :
            for index, ov2 in enumerate(objective_values) :
                if dominates(ov1, ov2) :
                    is_in_pareto_front[index] = False

        ovdf = pd.DataFrame(objective_values, columns = ['Time', 'Profit'])

        ovdf['ParetoFront'] = is_in_pareto_front
        ovdf['Tour'] = decoded_population

        instance_output = '.\\output\\' + filename + '-' + 'capacity' + str(instance.capacity) + '-' + 'epoch' + str(i) + '.pkl'

        if path.exists(instance_output) == False :
            ovdf.to_pickle(instance_output)

        yield instance_output