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
    Figure 2: Memory length impact
    --------- 
    - Make a graph of memory length vs. health - any correlation? [ with selection off (harshness = 0]
        - make bins (on X axis = bins of memory size) and do a histogram (Y axis of bars = avg health of everyone with that memory size)
        - this would be looking at it at one timestamp (toward the very end) 

    - Make a graph of memory max, min, avg vs. generation [ with selection on (try different levels of harshness)]


    BUG: Right now we have to manually run this by hand twice because we want the histogram
         when selection is turned off (harshness = 0) and we want the line graph when 
         selection is turned on (harshness = [3, 5, 10, 20])

    BUG: Issues come up when trying to multiple levels of harshness on the same run. 
    '''
    # General data storage setup (should be extracted to a helper function)
    path_number = 0
    while os.path.exists("figure_2_" + str(path_number) + "_" + str(date.today())):
        path_number += 1
    path = "figure_2_" + str(path_number) + "_" + str(date.today())
    os.makedirs(path)

    # Set Hyperparameters

    hp.GENERATIONS_TO_PLOT = [200]
    hp.MAX_MEMORY_LENGTH = 10000

    items_to_sweep = {
        "hp.ROWS": [15],
        "hp.COLS": [15],
        "hp.AGENTS_CAN_MERGE": [False],
        # "hp.NUM_TIMES_AGENT_MUTATED_IN_A_ROW_CAP": [3], # NOTE: THE DIFFERENT GRAPHS DEPEND ON THIS!!
        "hp.MEMORY_LENGTH_CAN_EVOLVE": [True],
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
            sim.memory_length_vs_health_histogram() # make sure that selection is off for this
            sim.memory_lengths_over_time() # make sure that selection is turned on for this 


        sim.plot = plot_override

        sim_path = os.path.join(path, "StartingMemoryLength" + str(hp.STARTING_MEMORY_LENGTH)) 
        for generation in hp.GENERATIONS_TO_PLOT:
            os.makedirs(os.path.join(sim_path, "generation_" + str(generation)))
        sim.set_path(sim_path)
        sim.run()
        