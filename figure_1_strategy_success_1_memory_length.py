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
    Figure 1:
    --------- 
    Sanity check: TfT is indeed a winner for 1-length memory
        - Show individual graphs for different parameter values:  harshness 
            (threshold for when an agent dies and is replaced), mutation rate: 
            4x4 grid of plots of each combo of (harshness, mutation rate)
        - Use results to set the harshness and mutation rate hyperparameters for the rest of the experiments.

    NOTE: We are defining harshness as the number of rounds before an agent dies and
            is replaced (copies) by its best performing neighbor. 

    TODO: Build a way to analyze the metrics across the sweep without having to 
          look at each file by hand. 
    '''
    # General data storage setup (should be extracted to a helper function)
    path_number = 0
    while os.path.exists("figure_1_" + str(path_number) + "_" + str(date.today())):
        path_number += 1
    path = "figure_1_" + str(path_number) + "_" + str(date.today())
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
            sim.plot_success_per_policy_for_one_memory()

        sim.plot = plot_override

        sim_path = os.path.join(path, "MutationRate_" + str(hp.GENOME_MUTATION_RATE) + "__Harshness_" + str(hp.NUM_TIMES_AGENT_MUTATED_IN_A_ROW_CAP)) 
        for generation in hp.GENERATIONS_TO_PLOT:
            os.makedirs(os.path.join(sim_path, "generation_" + str(generation)))
        sim.set_path(sim_path)
        sim.run()
        