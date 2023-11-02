import TfT_reproducibility_exp
import hyper_parameter_sweep
import memory_exp
import figure_1_strategy_success_1_memory_length
import figure_2_evolution_select_for_memory
import figure_N_strategy_counts_1_memory_length
import hyperparameters as hp

if __name__ == '__main__':
    # figure_N_strategy_counts_1_memory_length.run()
    # figure_1_strategy_success_1_memory_length.run()
    # hyper_parameter_sweep.run()
    hp.NUM_TIMES_AGENT_MUTATED_IN_A_ROW_CAP = -1
    figure_2_evolution_select_for_memory.run()

    # hp.NUM_TIMES_AGENT_MUTATED_IN_A_ROW_CAP = 3
    # figure_2_evolution_select_for_memory.run()

    # hp.NUM_TIMES_AGENT_MUTATED_IN_A_ROW_CAP = 10
    # figure_2_evolution_select_for_memory.run()

    # hp.NUM_TIMES_AGENT_MUTATED_IN_A_ROW_CAP = 20
    # figure_2_evolution_select_for_memory.run()

