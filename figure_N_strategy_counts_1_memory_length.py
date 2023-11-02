import itertools
import globals 
import merging
import hyperparameters as hp
from datetime import date
import os
import json

def run():
    '''
    ---------
    Figure N:
    (Will be placed in the section where we talk about changes of strategy through time)
    --------- 
    - Show that agents in the population tend to evolve to be TfT when constrained to 1 memory length
    
    - NOTE: This is different than success! We are just looking at sub-population sizes by strategy. 

    '''
    # General data storage setup (should be extracted to a helper function)
    path_number = 0
    while os.path.exists("figure_N_" + str(path_number) + "_" + str(date.today())):
        path_number += 1
    path = "figure_N_" + str(path_number) + "_" + str(date.today())
    os.makedirs(path)

    # Set Hyperparameters

    hp.GENERATIONS_TO_PLOT = [25000]
    hp.SHOULD_CALCULATE_SINGLE_MEMORY_STRATEGIES = True

    items_to_sweep = {
        "hp.ROWS": [15],
        "hp.COLS": [15],
        "hp.AGENTS_CAN_MERGE": [False],
        "hp.NUM_TIMES_AGENT_MUTATED_IN_A_ROW_CAP": [5],
        "hp.MEMORY_LENGTH_CAN_EVOLVE": [False],
        "hp.GENOME_MUTATION_RATE": [.01, .025, .05],
        "hp.STARTING_MEMORY_LENGTH": [1], # -1 = random between 1 and max
    }
    permutation = list(itertools.product(*items_to_sweep.values()))
    logs = {}
    for file_counter, values in enumerate(permutation):
        for i, hp_str in enumerate(items_to_sweep.keys()):
            logs[hp_str] = values[i]
            exec(hp_str + "=" + str(values[i])) # Sets the HPs with the appropriate values
        
        globals.CURRENT_ROUND = 0
        sim = merging.Animate(display_animation=False)

        # data gathering (plots)
        def plot_override():
            sim.single_memory_length_strategy_plot()
            sim.single_memory_color_map()

        sim.plot = plot_override

        sim_path = os.path.join(path, "MutationRate_" + str(hp.GENOME_MUTATION_RATE) + "__Harshness_" + str(hp.NUM_TIMES_AGENT_MUTATED_IN_A_ROW_CAP)) 
        for generation in hp.GENERATIONS_TO_PLOT:
            os.makedirs(os.path.join(sim_path, "generation_" + str(generation)))
        sim.set_path(sim_path)
        sim.run()
        