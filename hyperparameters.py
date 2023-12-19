# TODO: Hyperparameters that need work
AGENTS_CAN_MERGE = False

# Environment
ROWS = 15
COLS = 15

# Genome / Evolution
# NOTE: Mutations only impact the worst performing agent in a neighborhood. 
MAX_MEMORY_LENGTH = 5
STARTING_MEMORY_LENGTH = -1 # If set to -1, it will be a random value between 1 and MAX_MEMORY_LENGTH
NUM_TIMES_AGENT_MUTATED_IN_A_ROW_CAP = 3 # If set to -1, then we don't have neighbor cloning. 
GENOME_MUTATION_RATE = 0.20 # [.05 = 5%], [.2 = 20%], etc.
MEMORY_LENGTH_CAN_EVOLVE = False
MEMORY_LENGTH_INCREASE_MUTATION_RATE = .05
MEMORY_LENGTH_DECREASE_MUTATION_RATE = .05

# Prisoner's Dilemma Scoring
# NOTE: The reason why there are no point values for merge against
# cooperate or defect is because the merging agent will opt to use its
# backup strategy if it merges against a non merger. 
MERGE_AGAINST_MERGE_POINTS = 0
COOPERATE_AGAINST_COOPERATE_POINTS = 8
COOPERATE_AGAINST_DEFECT_POINTS = 0
DEFECT_AGAINST_DEFECT_POINTS = 5
DEFECT_AGAINST_COOPERATE_POINTS = 10

# Merge related hyperparameters
PERCENT_OF_MERGE_ACTIONS_IN_POLICY = .05 # % of the outputs in the policy table will be merged on average during initial generation and mutation. 

MUTATION_STRATEGY = "incremental" # "incremental = anchored in previous policy, "complete" = rerolling the output without regard for previous

# Data Acquisition
GENERATIONS_TO_PLOT = [1, 5, 10] # the total # of rounds that will run is the last element
NUM_ROUNDS_TO_TRACK_PHENOTYPE = 3
# Should be on if using single_memory_length_strategy_plot()
SHOULD_CALCULATE_SINGLE_MEMORY_STRATEGIES = True
# These two should only be on if plotting individuality_plot()
# or plot_cooperability_ratio_graph()
# BUG: When these are on, we see a progressive slowdown of the simulation.
# NOTE: They should be excluded from the hyperparameter sweep until the bug is fixed.
SHOULD_CALCULATE_HETEROGENEITY = False
SHOULD_CALCULATE_INDIVIDUALITY = False

# Parameter Tuning DBScan
PHENOTYPE_EPS = 1.05
PHENOTYPE_MIN_SAMPLES = 2
RELATIVE_HEALTH_EPS = 1.015
RELATIVE_HEALTH_MIN_SAMPLES = 2
'''
Original payoff matrix (for reference):
    COOPERATE_AGAINST_COOPERATE_POINTS = 8
    COOPERATE_AGAINST_DEFECT_POINTS = 0
    DEFECT_AGAINST_DEFECT_POINTS = 5
    DEFECT_AGAINST_COOPERATE_POINTS = 10
'''