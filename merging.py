import io
from pstats import SortKey
import pstats
import random
import sys
from sklearn.cluster import DBSCAN
from datetime import date
import os
from scipy import stats
import seaborn as sn
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.animation as animation
import numpy as np
import cProfile
import copy
import collections
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import hyperparameters as hp
import globals
import time


sys.setrecursionlimit(5000)

plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\pdsmi\\Desktop\\IPD\\ffmpeg-5.1.2-essentials_build\\bin\\ffmpeg.exe'

# rows, cols = (hp.ROWS, hp.COLS)
rows, cols = (3, 3)
x_max, y_max = (500, 500)

DB = []

def remove_first_encounters_from_memory(genome):
    # Helper function that removes the first memory_length encounters for each opponent in each agent.
    # This is because those memories are mostly random and likely won't provide meaningful data.
    inputs_to_delete = []
    for input in genome:
        if '0' in input:
            inputs_to_delete.append(input)
    for input in inputs_to_delete:
        del genome[input]

class Agent(object):
    instances = [] # A list of every agent in the game (does not include super agents, but does include their sub agents)

    # Data collection
    heterogeneity_per_round = []
    cooperability_per_round = []
    average_memory_length_per_round = []
    min_memory_length_per_round = []
    max_memory_length_per_round = []
    single_memory_strategy_info = {"TfT": {"counts": [], "avg_scores": []}, 
                                   "AC": {"counts": [], "avg_scores": []}, 
                                   "AD": {"counts": [], "avg_scores": []}, 
                                   "X": {"counts": [], "avg_scores": []}}
    individuality_per_round = {}
    for i in range(1, hp.MAX_MEMORY_LENGTH + 1):
        individuality_per_round[i] = []

    @classmethod
    def reset(cls):
        Agent.instances = []

        # Data collection
        Agent.heterogeneity_per_round = []
        Agent.cooperability_per_round = []
        Agent.average_memory_length_per_round = []
        Agent.min_memory_length_per_round = []
        Agent.max_memory_length_per_round = []
        Agent.single_memory_strategy_info = {"TfT": {"counts": [], "avg_scores": []}, 
                                               "AC": {"counts": [], "avg_scores": []}, 
                                               "AD": {"counts": [], "avg_scores": []}, 
                                               "X": {"counts": [], "avg_scores": []}}
        Agent.individuality_per_round = {}
        for i in range(1, hp.MAX_MEMORY_LENGTH + 1):
            Agent.individuality_per_round[i] = []

    @classmethod
    def populate(cls):
        matrix = []
        # Create all the agents
        for i in range(rows):
            matrix.append([])
            for j in range(cols):
                a = Agent(row=i, col=j)
                Agent.instances.append(a)
                matrix[-1].append(a)

        # Add their neighbors
        for a in Agent.instances:
            # Sides
            a.neighbors.add(matrix[(a.row - 1) % len(matrix)][a.col])
            a.neighbors.add(matrix[(a.row + 1) % len(matrix)][a.col])
            a.neighbors.add(matrix[a.row][(a.col - 1) % len(matrix[0])])
            a.neighbors.add(matrix[a.row][(a.col + 1) % len(matrix[0])])

            # Corners
            a.neighbors.add(matrix[(a.row - 1) % len(matrix)]
                            [(a.col - 1) % len(matrix[0])])
            a.neighbors.add(matrix[(a.row - 1) % len(matrix)]
                            [(a.col + 1) % len(matrix[0])])
            a.neighbors.add(matrix[(a.row + 1) % len(matrix)]
                            [(a.col + 1) % len(matrix[0])])
            a.neighbors.add(matrix[(a.row + 1) % len(matrix)]
                            [(a.col - 1) % len(matrix[0])])

    @classmethod
    def average_health(cls):
        return float(np.mean(list(a.health for a in Agent.instances)))

    @classmethod
    def worst_health(cls):
        return float(np.min(list(a.health for a in Agent.instances)))

    @classmethod
    def best_health(cls):
        return float(np.max(list(a.health for a in Agent.instances)))

    @classmethod
    def calculate_heterogeneity_and_cooperability(cls, include_first_encounters=False):
        num_defects = 0
        num_cooperates = 0

        unique_genomes = {}
        copied_agents = copy.deepcopy(Agent.instances)
        for a in copied_agents:
            sorted_genome = collections.OrderedDict(
                sorted(a.policy_table.items()))
            
            # Clears out random initial first encounters that only happened once.
            if not include_first_encounters:
                remove_first_encounters_from_memory(sorted_genome)

            # Records the number of unique genomes across the population.
            genome_string = json.dumps(sorted_genome)
            if genome_string in unique_genomes:
                unique_genomes[genome_string] += 1
            else:
                unique_genomes[genome_string] = 1

            # Counts the number of cooperates and defects across the population.
            for action in sorted_genome.values():
                if action == "d":
                    num_defects += 1
                elif action == "c":
                    num_cooperates += 1
                    
        Agent.heterogeneity_per_round.append(unique_genomes)
        if num_cooperates + num_defects == 0:
            # protect against divide by 0 error.
            Agent.cooperability_per_round.append(0)
        else:
            Agent.cooperability_per_round.append(
                num_cooperates / (num_cooperates + num_defects))
            
    @classmethod
    def assert_all_agents_in_super_agent_have_same_memory_length(self):
        super_agents = set()
        for agent in Agent.instances:
            if agent.super_agent is not None and agent.super_agent not in super_agents:
                super_agents.add(agent.super_agent)

        for super_agent in super_agents:
            for sub_agent in super_agent.sub_agents:
                for policy_input in list(sub_agent.policy_table.keys()):
                    if len(policy_input) != super_agent.memory_length:
                        raise Exception("Sub agent has a policy table with length != the super agent's memory length")
        
        for super_agent in super_agents:
            for sub_agent in super_agent.sub_agents:
                for opponent_past_actions in list(sub_agent.memory.values()):
                    if len(opponent_past_actions) != sub_agent.memory_length:
                        raise Exception("There exist an opponent for which the number of memories != sub_agent.memory_length")
                    if len(opponent_past_actions) != super_agent.memory_length:
                        raise Exception("There exist an opponent for which the number of memories != super_agent.memory_length")

    @classmethod
    def mutate_population(cls):
        already_viewed_super_agents = set()
        for agent in Agent.instances:
            if agent.super_agent is not None:
                if agent.super_agent in already_viewed_super_agents:
                    continue  # we have already checked this super agent
                already_viewed_super_agents.add(agent.super_agent)
                agent = agent.super_agent  # Temporarily setting the SuperAgent to the agent variable just for this loop iteration.

            # Decide if current agent is the worst agent in the neighborhood.
            worst_performing_agent_in_neighborhood = True
            for neighbor in list(agent.get_neighbors()):
                if agent.health_gained_this_round > neighbor.health_gained_this_round:
                    worst_performing_agent_in_neighborhood = False
                    break

            if worst_performing_agent_in_neighborhood:
                # If an agent performs the worst of its neighbors NUM_TIMES_AGENT_MUTATED_IN_A_ROW_CAP times
                # then it will just copy the genome of its most successful neighbor.
                agent.num_times_mutated_in_a_row += 1
                if agent.num_times_mutated_in_a_row == hp.NUM_TIMES_AGENT_MUTATED_IN_A_ROW_CAP:
                    # Finds the best neighbor and copies it
                    best_neighbor = agent
                    for neighbor in list(agent.get_neighbors()):
                        if neighbor.health_gained_this_round > best_neighbor.health_gained_this_round:
                            best_neighbor = neighbor
                    agent.copy_policy_table(best_neighbor)
                    # Adjust the memory length to match its new policy table (keep original memory)
                    agent.update_memory_length_to(best_neighbor.memory_length)
                    agent.copy_threshold(best_neighbor)
                else:
                    agent.mutate_policy_table()
                    agent.maybe_mutate_split_threshold()
                    if hp.MEMORY_LENGTH_CAN_EVOLVE:
                        agent.maybe_mutate_memory_length()
            else:
                agent.num_times_mutated_in_a_row = 0
 
    def __init__(self, row, col):
        self.super_agent = None  # points to the agent's SuperAgent

        self.memory = {}  # key = agent ID, value = array of past moves
        self.policy_table = {}  # key = memory (newest [left] -> oldest [right]), value = ("d" -> w1, "c" -> w2, "m" -> w3)
        self.neighbors = set()
        
        self.health = 0
        self.num_times_mutated_in_a_row = 0
        self.memory_length = 0; self.set_memory_length()  # how many actions back an agent remembers of its opponents
        self.split_threshold = 0; self.initialize_split_threshold()

        self.row = row
        self.col = col

        # Data collection:
        self.health_data = []
        self.phenotype_memory = []
        self.health_gained_this_round = 0

    # Updates health based on points gained or lost this round
    # Get's overloaded for Super Agent
    def update_health(self, points):
        number_of_neighbors = len(self.get_neighbors())
        if number_of_neighbors  == 0:
            raise Exception("All agents have merged together. TODO: Trigger end state")

        points_scaled_to_number_of_neighbors = points / number_of_neighbors
        self.health_gained_this_round += points_scaled_to_number_of_neighbors
        self.health += points_scaled_to_number_of_neighbors

    # Updates the memory based on the action performed by the opponent this round
    def update_memory(self, opponent, new_memory):
        for i in range(self.memory_length - 1):
            self.memory[opponent][i] = self.memory[opponent][i + 1]
        self.memory[opponent][-1] = new_memory

        if type(self) == SuperAgent:
            for sub_agent in self.sub_agents:
                sub_agent.memory = self.memory

    def initialize_split_threshold(self):
        # the lower threshold at which the agent will split from the super agent
        # if its super agent's performance is in the threshold_percentage bottom
        # percent of all of the agents in the simulation. 
        threshold_percentage = np.random.normal(loc=0.25, scale = .1) # .25 is the center and .1 is the standard deviation. 
        
        # make sure that the percentage is in [0, 1]
        return min(1, max(0, threshold_percentage))
       
    def set_memory_length(self):
        self.memory_length = hp.STARTING_MEMORY_LENGTH
        if self.memory_length == -1:
            # if STARTING_MEMORY_LENGTH is -1, then we set it randomly for each agent
            self.memory_length = np.random.randint(
                1, hp.MAX_MEMORY_LENGTH + 1)

    def get_neighbors(self, force=False):
        if self.super_agent and not force:
            return self.super_agent.get_neighbors()
        
        # self.neighbors stores the individual neighbors even if they are sub agents;
        # return only unmerged or super agent neighbors. Not sub agents.
        unique_neighbors = set()
        for neighbor in self.neighbors:
            if not neighbor.super_agent:
                unique_neighbors.add(neighbor)
            elif neighbor.super_agent not in unique_neighbors:
                unique_neighbors.add(neighbor.super_agent)

        # Otherwise if no super-agent or force it:
        return unique_neighbors
    
    def maybe_mutate_memory_length(self): 
        # Randomly decides whether or not to increase or shrink memory length by one.
        random_value = np.random.random()
        if random_value <= hp.MEMORY_LENGTH_INCREASE_MUTATION_RATE and self.memory_length < hp.MAX_MEMORY_LENGTH:
            if self.memory_length < hp.MAX_MEMORY_LENGTH:
                self.memory_length += 1
                self.grow_memory_by_one()
                self.grow_policy_keys_by_one()
        elif random_value <= hp.MEMORY_LENGTH_DECREASE_MUTATION_RATE + hp.MEMORY_LENGTH_INCREASE_MUTATION_RATE and self.memory_length > 1:
            if self.memory_length > 1:
                self.memory_length -= 1
                self.shrink_memory_by_one()
                self.shrink_policy_keys_by_one()

    def maybe_mutate_split_threshold(self):
        random_value = np.random.random()
        if random_value <= hp.SPLIT_THRESHOLD_MUTATION_RATE:
            threshold_percentage_change = np.random.normal(loc=0.0, scale = .05) # 0.0 is the center and .05 is the standard deviation. 
            self.split_threshold = min(1, max(0, self.split_threshold + threshold_percentage_change)) # make sure that the percentage is in [0, 1]

    def mutate_policy_table(self):
        for key in self.policy_table.keys():
            if np.random.random() < hp.GENOME_MUTATION_RATE:
                self.initialize_or_mutate_policy_output(key)
        
    def initialize_or_mutate_policy_output(self, policy_key):
        # This path represents a complete re-randomization of action weights
        if policy_key not in self.policy_table or hp.MUTATION_STRATEGY == "complete":
            # Sets cooperate and defect probabilities based of merge probability hyperparameter 
            remaining_probability = 1 - hp.PERCENT_OF_MERGE_ACTIONS_IN_POLICY
            defect_probability = random.uniform(0, remaining_probability)
            cooperate_probability = remaining_probability - defect_probability
            self.policy_table[policy_key] = {
                "c": cooperate_probability,
                "d": defect_probability,
                "m": hp.PERCENT_OF_MERGE_ACTIONS_IN_POLICY,
            }

            if hp.DEBUG:
                sum_of_weights = sum([cooperate_probability, defect_probability, hp.PERCENT_OF_MERGE_ACTIONS_IN_POLICY])
                if (sum_of_weights < 0.99 or sum_of_weights > 1.01):
                    raise Exception("Action weights do not sum to 1 [complete]: Sum = ", sum([cooperate_probability, defect_probability, hp.PERCENT_OF_MERGE_ACTIONS_IN_POLICY]))
            
        # This path bases the change in weights off of the previous weights 
        elif hp.MUTATION_STRATEGY == "incremental":
            # Chooses random values to nudge each weight such that the sum of the nudges adds to zero
            # which keeps the sum of the weights equal to 1.0
            nudges = {'c': np.random.normal(0, .1), 'd': np.random.normal(0, .1), 'm': np.random.normal(0, .1)}
            nudges_mean = np.mean(list(nudges.values()))
            for action in list(nudges.keys()):
                nudges[action] = nudges[action] - nudges_mean

            for action, nudge in nudges.items():
                new_weight = nudge + self.policy_table[policy_key][action]
                if new_weight < 0 or new_weight > 1:
                    # If anything is outside of [0, 1], reroll the nudges and try again
                    return self.initialize_or_mutate_policy_output(policy_key)
            
            for action, nudge in nudges.items():
                self.policy_table[policy_key][action] += nudge

            if hp.DEBUG:
                sum_of_nudges = sum(list(nudges.values()))
                if (sum_of_nudges < -0.001 or sum_of_nudges > .001):
                    raise Exception("Sum of nudges != 0: ", sum_of_nudges)

                sum_of_new_weights = sum(list(self.policy_table[policy_key].values()))
                if (sum_of_new_weights > 1.001 or sum_of_new_weights < .999):
                    raise Exception("Sum of new weights != 1: ", sum_of_new_weights)

        else:
            raise Exception("Hyperparameter was not found. MUTATION_STRATEGY = ", hp.MUTATION_STRATEGY)

    def get_policy_output(self, memory_of_opp):
        policy_key = "".join(memory_of_opp)
        if policy_key not in self.policy_table:
            # If the agent has never seen this memory combination before, it will
            # build an output for it and store it in its policy table
            self.initialize_or_mutate_policy_output(policy_key)

        # Queries the policy table with the memory of the current opponent and randomly chooses
        # an action based on the associated weights
        (actions, weights) = self.policy_table[policy_key].keys(), self.policy_table[policy_key].values()
        return random.choices(list(actions), list(weights))[0]
    
    def copy_policy_table(self, other):
        if type(other) == SuperAgent:
            # Single cell copying a single cell
            self.policy_table = other.consolidate_sub_agents_policies_into_one_policy_table()
        else:
            # Single cell copying a single cell
            self.policy_table = copy.deepcopy(other.policy_table)

    def copy_threshold(self, other):
        if type(other) == SuperAgent:
            # Single cell copying super agent
            self.split_threshold = np.mean([sub_agent.split_threshold for sub_agent in other.sub_agents])
        else:
            # Single cell copying a single cell
            self.split_threshold = other.split_threshold

    def maybe_split(self):
        if self.super_agent is None:
            return
        
        if self.super_agent.health <= self.split_threshold:
            # NOTE: Neighbors should get handled out of the box. So should rendering. 
            #       But make sure to validate this.
            self.num_times_mutated_in_a_row = 0
            self.super_agent.sub_agents.remove(self)
            self.super_agent.update_super_agent_health_from_sub_agents()
            self.memory_length = self.super_agent.memory_length
            new_memory = {}
            for opp, history in self.super_agent.memory.items():
                new_memory[opp] = copy.deepcopy(history)
            self.memory = new_memory
            self.super_agent = None
            
    def play_against(self, opp):
        # time.sleep(.25)
        # print("sleeping")

        # If both are part of super-agents
        if self.super_agent and opp.super_agent:
            # And both are part of the *same* super-agent, it's a bug
            if self.super_agent == opp.super_agent: 
                raise Exception("Agents from the same super agent are playing against each other.")
            else:
                return self.super_agent.play_against(opp.super_agent)
        # If I'm part of a super agent and my opponent isn't:
        elif self.super_agent and not opp.super_agent:
            return self.super_agent.play_against(opp)
        # If my opponent is part of a super agent and I'm not:
        elif (not self.super_agent) and opp.super_agent:
            return self.play_against(opp.super_agent)
        # Else play as below:

        # Sets up an empty memory of a never before seen opponent
        if opp not in self.memory:
            self.memory[opp] = ['0'] * self.memory_length
        if self not in opp.memory:
            opp.memory[self] = ['0'] * opp.memory_length

        # Gets the action based on the memory of the current opponent
        my_action = self.get_policy_output(self.memory[opp])
        opp_action = opp.get_policy_output(opp.memory[self])

        # Awards correct points based on the agents' actions
        if my_action == 'm' and opp_action == 'm':
            my_point_change = hp.MERGE_AGAINST_MERGE_POINTS
            opp_point_change = hp.MERGE_AGAINST_MERGE_POINTS
            self.merge_with(opp)
        elif my_action == 'm' and opp_action == 'c':
            my_point_change = hp.COOPERATE_AGAINST_COOPERATE_POINTS
            opp_point_change = hp.COOPERATE_AGAINST_COOPERATE_POINTS
        elif my_action == 'c' and opp_action == 'm':
            my_point_change = hp.COOPERATE_AGAINST_COOPERATE_POINTS
            opp_point_change = hp.COOPERATE_AGAINST_COOPERATE_POINTS
        elif my_action == 'd' and opp_action == 'm':
            my_point_change = hp.DEFECT_AGAINST_COOPERATE_POINTS
            opp_point_change = hp.COOPERATE_AGAINST_DEFECT_POINTS
        elif my_action == 'm' and opp_action == 'd':
            my_point_change = hp.COOPERATE_AGAINST_DEFECT_POINTS
            opp_point_change = hp.COOPERATE_AGAINST_DEFECT_POINTS
        elif my_action == 'c' and opp_action == 'c':
            my_point_change = hp.COOPERATE_AGAINST_COOPERATE_POINTS
            opp_point_change = hp.COOPERATE_AGAINST_COOPERATE_POINTS
        elif my_action == 'c' and opp_action == 'd':
            my_point_change = hp.COOPERATE_AGAINST_DEFECT_POINTS
            opp_point_change = hp.DEFECT_AGAINST_COOPERATE_POINTS
        elif my_action == 'd' and opp_action == 'c':
            my_point_change = hp.DEFECT_AGAINST_COOPERATE_POINTS
            opp_point_change = hp.COOPERATE_AGAINST_DEFECT_POINTS
        elif my_action == 'd' and opp_action == 'd':
            my_point_change = hp.DEFECT_AGAINST_DEFECT_POINTS
            opp_point_change = hp.DEFECT_AGAINST_DEFECT_POINTS
        else:
            raise Exception("Unknown action taken by an agent.")

        # skip updating memory if just merged
        if not (my_action == 'm' and opp_action == 'm'):
            # Update memories based on other's action
            self.update_memory(opp, opp_action)
            opp.update_memory(self, my_action)

        # Add or subtract the change in points from that interaction
        self.update_health(my_point_change)
        opp.update_health(opp_point_change)
        
        return (my_action, opp_action)

    def merge_with(self, opp):
        if type(opp) == SuperAgent:
            opp.merge_with(self)
        elif self.super_agent == None and opp.super_agent == None:
            # Creates a new super agent with two individual agents
            SuperAgent(sub_agents=[self, opp])
        elif self.super_agent and opp.super_agent:
            self.super_agent.merge_with(opp.super_agent)
        elif self.super_agent and (not opp.super_agent):
            self.super_agent.merge_with(opp)
        elif (not self.super_agent) and opp.super_agent:
            opp.super_agent.merge_with(self)
        else:
            raise Exception("Merging not supported for this combination of agents.")

    def shrink_memory_by_one(self):
        for other in self.memory.keys():
            self.memory[other] = self.memory[other][:-1]
    
    def shrink_policy_table_keys_by_one(self):
        # Cuts off the last memory of each opponent and sets that new memory
        # equal to the average of the weights for the two memories that will conflict.
        # Ex: D,M,D and D,M,C --[last memory removed]--> D,M and D,M (newly overlapping memories need to be merged)
        new_policy_table = {}
        for memory_string, action_weights in self.policy_table.items():
            new_key = memory_string[:-1]
            if new_key in new_policy_table:
                averaged_weights = {}
                for action, weight in new_policy_table[new_key].items():
                    averaged_weights[action] = (weight + action_weights[action]) / 2
                new_policy_table[new_key] = averaged_weights
            else:
                new_policy_table[new_key] = action_weights

        self.policy_table = new_policy_table

    def grow_memory_by_one(self):
        for other in self.memory.keys():
            self.memory[other].append('0')

    def grow_policy_table_keys_by_one(self):
        # The action weights for each memory combination get duplicated for each new
        # (longer) memory input that gets created in the policy table. 
        new_policy_table = {}
        for memory_string, action_weights in self.policy_table.items():
            new_policy_table[memory_string + "c"] = action_weights
            new_policy_table[memory_string + "d"] = action_weights
            new_policy_table[memory_string + "m"] = action_weights
            new_policy_table[memory_string + "0"] = action_weights

        self.policy_table = new_policy_table

    def shrink_memory_by_n(self, n):
        for _ in range(n):
            self.shrink_memory_by_one()

    def shrink_policy_keys_by_n(self, n):
        for _ in range(n):
            self.shrink_policy_keys_by_one()
        
    def grow_memory_by_n(self, n):
        for _ in range(n):
            self.grow_memory_by_one()

    def grow_policy_keys_by_n(self, n):
        for _ in range(n):
            self.grow_policy_table_keys_by_one()

    def update_memory_length_to(self, new_length):
        if self.memory_length > new_length:
            amount_to_shrink = self.memory_length - new_length
            # Cut down the memory of each opponent
            for opponent, history in self.memory.items():
                self.memory[opponent] = history[:(0 - amount_to_shrink)]
        elif self.memory_length < new_length:
            amount_to_grow = new_length - self.memory_length
            # Add '0's (AKA unknown interaction placeholders) to the back of the memory for each
            # opponent. 
            for opponent, history in self.memory.items():
                self.memory[opponent] = history + (amount_to_grow * ['0'])

        self.memory_length = new_length

        if hp.DEBUG:
            for opponent_past_actions in self.memory.values():
                if len(opponent_past_actions) != self.memory_length:
                    raise Exception("There exists an opponent for which self.memory[opponent] != self.memory_length")

    # Return the health of this agent on a scale from 0 (worst) to 1 (best)
    # relative to its peers
    def relative_health(self):
        spread = Agent.best_health() - Agent.worst_health()
        if spread == 0:
            return 0.5
        me_above_average = self.health - Agent.average_health()
        return 0.5 + (me_above_average / spread)*0.5

    def _merger_list(self):
        if not self.super_agent:
            return [False, False, False, False]
        else:
            merger_to_left = False
            merger_to_right = False
            merger_to_top = False
            merger_to_bottom = False
            for sa in self.super_agent.sub_agents:
                if sa.col == (self.col - 1) and sa.row == self.row:
                    merger_to_left = True
                elif sa.col == (self.col + 1) and sa.row == self.row:
                    merger_to_right = True
                elif sa.col == self.col and sa.row == (self.row + 1):
                    merger_to_top = True
                elif sa.col == self.col and sa.row == (self.row - 1):
                    merger_to_bottom = True
            return [merger_to_left,
                    merger_to_right,
                    merger_to_top,
                    merger_to_bottom]

    # Return [x, y] coordinates for the corners of a rectangle to draw
    def corners(self):
        merger_to_left, merger_to_right, merger_to_top, merger_to_bottom = self._merger_list()
        # How much space to leave, in percent
        spacer = 0.3
        unspaced_height = float(x_max) / cols
        unspaced_width = float(y_max) / rows
        left_spacer = 0 if merger_to_left else spacer/2
        right_spacer = 0 if merger_to_right else spacer/2
        top_spacer = 0 if merger_to_top else spacer/2
        bottom_spacer = 0 if merger_to_bottom else spacer/2
        x_start = (self.col + left_spacer)*unspaced_width
        y_start = (self.row + bottom_spacer)*unspaced_height
        x_end = (self.col + 1 - right_spacer)*unspaced_width
        y_end = (self.row + 1 - top_spacer)*unspaced_height
        return np.array([[x_start, y_start],
                         [x_start, y_end],
                         [x_end, y_end],
                         [x_end, y_start]])


class SuperAgent(Agent):
    def __init__(self, sub_agents):
        self.super_agent = None
        if any(type(sa) == SuperAgent for sa in sub_agents):
            raise Exception("SuperAgent is a sub agent of a SuperAgent")
        self.sub_agents = sub_agents
        for sa in sub_agents:
            sa.super_agent = self
        self.health = 0
        self.num_times_mutated_in_a_row = 0
        self.health_gained_this_round = 0
        self.update_super_agent_health_from_sub_agents()
        self.memory = {}
        self.memory_length = 0

        self.merge_memories()

    # Forces all sub agents to have the the same memory length
    def make_memories_equal_size(self):
        average_memory_length = int(np.round(np.mean([sa.memory_length for sa in self.sub_agents])))
        self.memory_length = average_memory_length

        # Need to get the unique memories because many of them will be shallow copies of each other.
        # No doing this would cause memory shrinks and grows to effect multiple sub agents at once.
        # We point the sub agents' memories at the super agent's memory.
        sub_agents_with_unique_memories = {}
        for sub_agent in self.sub_agents:
            if id(sub_agent.memory) not in sub_agents_with_unique_memories:
                sub_agents_with_unique_memories[id(sub_agent.memory)] = sub_agent

        for sub_agent in list(sub_agents_with_unique_memories.values()):
            if sub_agent.memory_length < average_memory_length:
                sub_agent.grow_memory_by_n(average_memory_length - sub_agent.memory_length)
            elif sub_agent.memory_length > average_memory_length:
                sub_agent.shrink_memory_by_n(sub_agent.memory_length - average_memory_length)

        for sub_agent in self.sub_agents:
            if sub_agent.memory_length < average_memory_length:
                sub_agent.grow_policy_keys_by_n(average_memory_length - sub_agent.memory_length)
            elif sub_agent.memory_length > average_memory_length:
                sub_agent.shrink_policy_keys_by_n(average_memory_length - sub_agent.memory_length)
            sub_agent.memory_length = average_memory_length

        if hp.DEBUG:
            for sub_agent in self.sub_agents:
                if sub_agent.memory_length != average_memory_length:
                    raise Exception("Agent's memory length does not match the super agent's memory length")
                
                for opponent_actions in self.memory.values():
                    if len(opponent_actions) != self.memory_length:
                        raise Exception("There exists an opponent for which the length of past actions is different from the super agent's memory length")

                for policy_input in list(sub_agent.policy_table.keys()):
                    if len(policy_input) != average_memory_length:
                        raise Exception("Policy table's memory input does not match the agent's memory length")

    # Combines the memories of all sub agents
    def merge_memories(self):
        self.make_memories_equal_size()
    
        opponent_to_memories_map = {}
        # Builds a map of {opponent -> list[memory1, memory2, ...]}
        for sub_agent in self.sub_agents:
            for (opponent, memory_of_opponent) in sub_agent.memory.items():
                if opponent in opponent_to_memories_map:
                    opponent_to_memories_map[opponent].append(memory_of_opponent)
                else:
                    opponent_to_memories_map[opponent] = [memory_of_opponent]

        # takes a list of memories and returns a single array with the most common action for each index
        def get_most_common_action_per_memory_idx(memories):
            condensed_memories = []
            t = np.array(memories).transpose()
            for row in t:
                vals, counts = np.unique(row, return_counts=True)
                most_common_actions = []
                max_count = max(counts)
                for (v, c) in zip(vals, counts):
                    if c == max_count:
                        most_common_actions.append(v)
                condensed_memories.append(random.choice(most_common_actions))
            return condensed_memories

        self.memory = {}
        # NOTE: If we want each sub agent to adopt this memory, we should loop through them and
        # set their memories equal to the super agent's memory. 
        for (opponent, redundant_memories) in opponent_to_memories_map.items():
            self.memory[opponent] = get_most_common_action_per_memory_idx(redundant_memories)

        for sub_agent in self.sub_agents:
            sub_agent.memory = self.memory  # Note that this is making a shallow copy

    # Get the policy output for each sub agent return the most common one.
    def get_policy_output(self, memory_of_opp):
        super_agent_policy = {
            "m": 0,
            "c": 0,
            "d": 0
        }
        for sub_agent in self.sub_agents:
            super_agent_policy[sub_agent.get_policy_output(memory_of_opp)] += 1

        return max(super_agent_policy, key=super_agent_policy.get)

    # Consolidates all of the neighbors for all of the sub agents.
    def get_neighbors(self, force=False):
        unique_neighbors = set()
        for sub_agent in self.sub_agents:
            for neighbor in sub_agent.neighbors:
                if not neighbor.super_agent:
                    # Add if it's a single celled agent
                    unique_neighbors.add(neighbor)
                elif neighbor.super_agent not in unique_neighbors and neighbor.super_agent is not self:
                    # Add the super agent if it hasn't been added yet AND make sure that super agent isn't ourself.
                    unique_neighbors.add(neighbor.super_agent)

        return unique_neighbors

    # Sets the SuperAgent's health to the average of all the sub agents' healths.
    def update_super_agent_health_from_sub_agents(self):
        avg_health = np.mean([sa.health for sa in self.sub_agents])
        for sa in self.sub_agents:
            sa.health = avg_health
        self.health = avg_health

    # Merges with other Agent or SuperAgent
    def merge_with(self, other):
        if isinstance(other, SuperAgent):
            self.sub_agents += other.sub_agents
            for osa in other.sub_agents:
                osa.super_agent = self
            del (other)
        elif isinstance(other, Agent):
            self.sub_agents.append(other)
            other.super_agent = self
        else:
            raise Exception("Agent is of unknown type")

        self.update_super_agent_health_from_sub_agents()
        self.merge_memories()
        

    def maybe_mutate_memory_length(self):     
        # Randomly decides whether or not to increase or shrink memory length by one.
        random_value = np.random.random()
        direction = ""
        if random_value <= hp.MEMORY_LENGTH_INCREASE_MUTATION_RATE:
            direction = "grow"
        elif random_value <= hp.MEMORY_LENGTH_DECREASE_MUTATION_RATE + hp.MEMORY_LENGTH_INCREASE_MUTATION_RATE:
            direction = "shrink"
        else:
            return 

        # Make the same memory length change to each sub_agent
        for sub_agent in self.sub_agents:
            if direction == "grow" and sub_agent.memory_length < hp.MAX_MEMORY_LENGTH:
                if self.memory_length < hp.MAX_MEMORY_LENGTH:
                    sub_agent.memory_length += 1
                    sub_agent.grow_memory_by_one()
                    sub_agent.grow_policy_keys_by_one()
            elif direction == "shrink" and sub_agent.memory_length > 1:
                if self.memory_length > 1:
                    sub_agent.memory_length -= 1
                    sub_agent.shrink_memory_by_one()
                    sub_agent.shrink_policy_keys_by_one()

    def maybe_mutate_split_threshold(self):
        for sub_agent in self.sub_agents:
            sub_agent.maybe_mutate_split_threshold()

    def mutate_policy_table(self):
        for sub_agent in self.sub_agents:
            sub_agent.mutate_policy_table()

    # Returns a deep copied version of the super agent's policy table
    def consolidate_sub_agents_policies_into_one_policy_table(self):
        # Populate a map of all of the of the sub agents' weights for each policy input
        policy_input_to_action_weight_lists_map = {} # Ex: {'cdmd' -> {m: [.2, .2, .1], c: [.3, .5, .2], d: [.5, .3, .7]...}, ...}
        for sub_agent in self.sub_agents:
            for policy_input, weight_map in sub_agent.policy_table.items():
                if policy_input in policy_input_to_action_weight_lists_map:
                    for w_key, w_val in weight_map.items():
                        policy_input_to_action_weight_lists_map[policy_input][w_key].append(w_val)
                else:
                    for w_key, w_val in weight_map.items():
                        policy_input_to_action_weight_lists_map[policy_input][w_key] = list(w_val)
        
        # Go through the filled map and average the weights
        for policy_input, action_weight_lists in policy_input_to_action_weight_lists_map.items():
            for w_key, w_val_list in action_weight_lists.items():
                policy_input_to_action_weight_lists_map[policy_input][w_key] = np.mean(w_val_list)

        if hp.DEBUG:
            # Validate that the weights add up to 1 for each row of each sub_agents' policy table
            for policy_input, averaged_action_weights_map in policy_input_to_action_weight_lists_map.items():
                sum_of_weights = sum(averaged_action_weights_map.values())
                if  sum_of_weights > 1.001 or sum_of_weights < 9.999:
                    raise Exception("Consolidated table has at least one output where weights don't sum to 1")

        return policy_input_to_action_weight_lists_map

    def copy_policy_table(self, other):
        if type(other) == SuperAgent:
            # Super agent copying another super agent
            other_super_agent_consolidated_policy_table = other.consolidate_sub_agents_policies_into_one_policy_table()
            for sub_agent in self.sub_agents:
                sub_agent.policy_table = other_super_agent_consolidated_policy_table
        else:
            # Super agent copying a single agent
            for sub_agent in self.sub_agents:
                sub_agent.policy_table = copy.deepcopy(other.policy_table)

    def copy_threshold(self, other):
        if type(other) == SuperAgent:
            # Super agent copying super agent
            mean_of_other_sub_agents_split_thresholds = np.mean([sub_agent.split_threshold for sub_agent in other.sub_agents])
            for sub_agent in self.sub_agents:
                sub_agent.split_threshold = np.random.normal(loc=mean_of_other_sub_agents_split_thresholds, scale = .15)
        else:
            # Super agent copying a single cell
            for sub_agent in self.sub_agents:
                sub_agent.split_threshold = np.random.normal(loc=other.split_threshold, scale = .15)

class Animate:
    def __init__(self, display_animation):
        global R, S, T, P
        R = np.random.uniform(-100.0, 100.0)
        S = np.random.uniform(-100.0, 100.0)
        T = np.random.uniform(-100.0, 100.0)
        P = np.random.uniform(-100.0, 100.0)
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, x_max)
        self.ax.set_ylim(0, y_max)

        self.display_animation = display_animation

        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        Agent.reset()
        Agent.populate()

        self.players = []
        self.current_player_idx = 0
        self.current_opponent_idx = 0
        self.already_played = set()

        self.patches = []

        self.max_scores = []
        self.avg_scores = []
        self.min_scores = []

        self.path = ""
        
    def set_path(self, path):
        self.path = path

    def init(self):
        # agents = Agent.instances
        # return self.patches
        pass

    def display_matchup_animation(self, player0, player1, my_policy, opp_policy):
        if my_policy == "m":
            fc = "yellow"
        elif my_policy == "c":
            fc = "green"
        elif my_policy == "d":
            fc = "red"
        else:
            raise Exception("Tried to animate unknown policy: ", my_policy)
        p1 = Polygon(player0.corners(), facecolor=fc, linewidth=0)
        self.patches.append(p1)

        if opp_policy == "m":
            fc = "yellow"
        elif opp_policy == "c":
            fc = "green"
        elif opp_policy == "d":
            fc = "red"
        else:
            raise Exception("Tried to animate unknown policy: ", opp_policy)

        p2 = Polygon(player1.corners(), facecolor=fc, linewidth=0)
        self.patches.append(p2)

        for a in Agent.instances:
            if a is not player0 and a is not player1:
                rh = a.relative_health()
                fc = [rh, rh, rh]
                pg = Polygon(a.corners(), facecolor=fc, linewidth=0)
                self.patches.append(pg)

    def initialize_round(self):
        # Only allow individual agents and super agents (no sub agents)
        unique_players = set()
        for agent in Agent.instances:
            if agent.super_agent and agent.super_agent not in unique_players:
                unique_players.add(agent.super_agent)
            elif not agent.super_agent:
                unique_players.add(agent)

        # Decide the opponents for each player and shuffle them
        for player in unique_players:
            opponents = list(player.get_neighbors())
            np.random.shuffle(opponents)
            self.players.append((player, opponents))

        # Shuffle the order in which the players go this round
        np.random.shuffle(self.players)

    def animate(self, step):
        for p in self.ax.patches:
            p.remove()

        if self.current_player_idx == 0 and self.current_opponent_idx == 0:
            self.initialize_round()

        player, opponents = self.players[self.current_player_idx]
        current_opponent = opponents[self.current_opponent_idx]

        '''
        case if there is a merge mid round and two of the sub agents were set to play against each other
        or a super agent was set to play against an individual agent that is now part of it
        '''
        if not ((player.super_agent is not None and 
                 current_opponent.super_agent is not None and 
                 player.super_agent == current_opponent.super_agent)
                or (type(player) == SuperAgent and current_opponent.super_agent == player)
                or (type(current_opponent) == SuperAgent and player.super_agent == current_opponent)):
            my_policy, opp_policy = player.play_against(current_opponent)
            if self.display_animation:
                self.display_matchup_animation(player, current_opponent, my_policy, opp_policy)

        # if len(player0.phenotype_memory) == (len(self.current_neighbors) * hp.NUM_ROUNDS_TO_TRACK_PHENOTYPE):
        #     player0.phenotype_memory = player0.phenotype_memory[len(self.current_neighbors):]
        # player0.phenotype_memory.append(my_policy)

        

        self.current_opponent_idx += 1
        if self.current_opponent_idx == len(opponents):
            # New Agent
            self.current_player_idx += 1
            self.current_opponent_idx = 0
            if self.current_player_idx == len(self.players):
                # New Round
                self.current_player_idx = 0
                print("Current round: ", globals.CURRENT_ROUND)
                if hp.SHOULD_CALCULATE_INDIVIDUALITY:
                    self.calculate_individuality()

                # if hp.SHOULD_CALCULATE_SINGLE_MEMORY_STRATEGIES:
                #     self.calculate_single_memory_info()

                Agent.assert_all_agents_in_super_agent_have_same_memory_length
                Agent.mutate_population() 

                if hp.SHOULD_CALCULATE_HETEROGENEITY:
                    # print("calculating heterogeneity")
                    Agent.calculate_heterogeneity_and_cooperability()

                max_memory_length = 1 
                min_memory_length = hp.MAX_MEMORY_LENGTH
                memory_lengths = []
                for a in Agent.instances:
                    a.maybe_split()
                    a.health_data.append(a.health)
                    a.health_gained_this_round = 0
                    if a.memory_length > max_memory_length:
                        max_memory_length = a.memory_length
                    if a.memory_length < min_memory_length:
                        min_memory_length = a.memory_length
                    memory_lengths.append(a.memory_length)
                Agent.average_memory_length_per_round.append(np.average(memory_lengths))
                Agent.max_memory_length_per_round.append(max_memory_length)
                Agent.min_memory_length_per_round.append(min_memory_length)

                if globals.CURRENT_ROUND in hp.GENERATIONS_TO_PLOT:
                    pass
                    # self.plot()

                globals.CURRENT_ROUND += 1
                self.current_agent_num = 0

                # Data collection at the end of each round
                self.max_scores.append(Agent.best_health())
                self.avg_scores.append(Agent.average_health())
                self.min_scores.append(Agent.worst_health())

        if self.display_animation:
            for p in self.patches:
                self.ax.add_patch(p)

            self.patches.clear()

    def plot_max_min_avg(self):
        plt.plot([y / (x + 1) for x, y in enumerate(self.max_scores)])
        plt.plot([y / (x + 1) for x, y in enumerate(self.avg_scores)])
        plt.plot([y / (x + 1) for x, y in enumerate(self.min_scores)])
        ax = plt.gca()
        ax.set(xlabel='Round', ylabel='Health',
               title="Agent Health Throughout Simulation")
        ax.grid(True)
        # ax.set_ylim([-1, 101])
        ax.legend(["Max Health", "Average Health", "Min Health"])
        plt.savefig(os.path.join(self.path, "generation_" + str(globals.CURRENT_ROUND),
                    "Max_Min_Avg_Scores" + str(globals.CURRENT_ROUND)))
        plt.clf()

    def plot_all_healths(self):
        plt.clf()
        self.ax.get_xaxis().set_visible(True)
        self.ax.get_yaxis().set_visible(True)
        
        for agent in Agent.instances:
            # print(len(agent.health_data))
            plt.plot([y / (x + 1) for x, y in enumerate(agent.health_data)])

        ax = plt.gca()
        ax.set(xlabel='Round', ylabel='Health',
               title="Agent Health Throughout Simulation")
        ax.grid(True)
        ax.set_xlim([0, len(agent.health_data) - 1])
        # ax.set_ylim([-1, 101])
        print("PRINTING PATH: ", self.path)
        plt.savefig(os.path.join(self.path, "generation_" +
                    str(globals.CURRENT_ROUND), "Individual_Scores" + str(globals.CURRENT_ROUND)))
        plt.clf()
        plt.cla()
        plt.close('all')
        

    def print_policy_tables(self):
        self.agents.sort(key=lambda x: x.health, reverse=True)
        for agent in self.agents:
            print("R: ", agent.row, "C: ", agent.col, " Health: ",
                  agent.health, " -- Table: ", agent.policy_table)

    def print_heterogeneity_per_round(self):
        print("Heterogeneity:")
        for i, entry in enumerate(Agent.heterogeneity_per_round):
            print(i, ": ", entry)

    def plot_cooperability_ratio_graph(self):
        print(Agent.cooperability_per_round)
        plt.plot(Agent.cooperability_per_round)
        ax = plt.gca()
        ax.set(xlabel="Round", ylabel="Cooperability Ratio",
               title="Cooperability Ratios Throughout Simulation")
        ax.grid(True)
        plt.savefig(os.path.join(self.path, "generation_" + str(globals.CURRENT_ROUND),
                    "Cooperability_Ratio_Graph" + str(globals.CURRENT_ROUND)))
        # for i, entry in enumerate(Agent.cooperability_per_round):
        #     print(i, ": ", entry)
        plt.clf()

    def genotype_color_map(self, include_first_encounters=False):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, x_max)
        self.ax.set_ylim(0, y_max)

        # self.graph, self.graph_ax = plt.subplots()
        # x_ticks = np.linspace(0, x_max, cols + 1)
        # y_ticks = np.linspace(0, y_max, rows + 1)
        # ax.set_xticks(x_ticks)
        # ax.set_yticks(y_ticks)
        # ax.xaxis.grid(True)
        # ax.yaxis.grid(True)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_title("Genotype color map: Red=D, Yellow=C")

        cmap = LinearSegmentedColormap.from_list('rg', ["r", "y"], N=256)

        for a in Agent.instances:
            genome = a.policy_table
            if not include_first_encounters:
                remove_first_encounters_from_memory(genome)

            defect_counter = 0
            cooperate_counter = 0
            for action in genome.values():
                if action == "d":
                    defect_counter += 1
                elif action == "c":
                    cooperate_counter += 1

            if cooperate_counter + defect_counter == 0:
                p = Polygon(a.corners(), facecolor=cmap(0), linewidth=0)
            else:
                p = Polygon(a.corners(), facecolor=cmap(
                    cooperate_counter / (cooperate_counter + defect_counter)), linewidth=0)

            self.ax.add_patch(p)

        plt.savefig(os.path.join(self.path, "generation_" + str(globals.CURRENT_ROUND),
                    "Genotype_Color_Map" + str(globals.CURRENT_ROUND)))
        plt.clf()

    def phenotype_color_map(self, include_first_encounters=False):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, x_max)
        self.ax.set_ylim(0, y_max)

        # self.graph, self.graph_ax = plt.subplots()
        # x_ticks = np.linspace(0, x_max, cols + 1)
        # y_ticks = np.linspace(0, y_max, rows + 1)
        # ax.set_xticks(x_ticks)
        # ax.set_yticks(y_ticks)
        # ax.xaxis.grid(True)
        # ax.yaxis.grid(True)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_title("Phenotype color map: Red=D, Yellow=C")

        cmap = LinearSegmentedColormap.from_list('rg', ["r", "y"], N=256)

        for a in Agent.instances:
            defect_counter = 0
            cooperate_counter = 0
            for pt in a.phenotype_memory:
                if pt == "c":
                    cooperate_counter += 1
                elif pt == "d":
                    defect_counter += 1

            p = Polygon(a.corners(), facecolor=cmap(
                cooperate_counter / (cooperate_counter + defect_counter)), linewidth=0)

            self.ax.add_patch(p)

        # plt.show()
        plt.savefig(os.path.join(self.path, "generation_" + str(globals.CURRENT_ROUND),
                    "Phenotype_Color_Map" + str(globals.CURRENT_ROUND)))
        plt.clf()

    def phenotype_DBScan(self, include_first_encounters=False):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, x_max)
        self.ax.set_ylim(0, y_max)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_title("Phenotype DBScan Map")

        data_points = []
        for a in Agent.instances:
            defect_counter = 0
            cooperate_counter = 0
            for pt in a.phenotype_memory:
                if pt == "c":
                    cooperate_counter += 1
                elif pt == "d":
                    defect_counter += 1

            phenotype_value = 2 * (cooperate_counter /
                                   (cooperate_counter + defect_counter))
            data_points.append([a.row, a.col, phenotype_value])
        
        # max = data_points[0][2]
        # min = data_points[0][2]
        # for dp in data_points:
        #     if dp[2] < min:
        #         min = dp[2]
        #     if dp[2] > max:
        #         max = dp[2]
        # print(min)
        # print(max)

        clustering = DBSCAN(eps=hp.PHENOTYPE_EPS, min_samples=hp.PHENOTYPE_MIN_SAMPLES).fit(data_points)
        colors = {-1: "black"}
        for i, a in enumerate(Agent.instances):
            label = clustering.labels_[i]
            color = ""
            if label in colors:
                color = colors[label]
            else:
                colors[label] = "#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
                color = colors[label]

            p = Polygon(a.corners(), facecolor=color, linewidth=0)

            self.ax.add_patch(p)

        # plt.show()
        plt.savefig(os.path.join(self.path, "generation_" + str(globals.CURRENT_ROUND),
                    "Phenotype_DBScan_Color_Map" + str(globals.CURRENT_ROUND)))
        plt.clf()


    def general_DBScan_info(self, version="random"):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, x_max)
        self.ax.set_ylim(0, y_max)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_title("Phenotype DBScan Map")

        data_points = []
        for a in Agent.instances:
            data_points.append([a.row, a.col, random.random() * 2])

        clustering = DBSCAN(eps=1.05, min_samples=2).fit(data_points)
        colors = {-1: "black"}
        for i, a in enumerate(Agent.instances):
            label = clustering.labels_[i]
            if version == "all_islands":
                label = -1 # for all islands
            elif version == "all_same":
                label = 0 # for all the same
            color = ""
            if label in colors:
                color = colors[label]
            else:
                colors[label] = "#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
                color = colors[label]

            p = Polygon(a.corners(), facecolor=color, linewidth=0)

            self.ax.add_patch(p)

        # plt.show()
        plt.savefig(os.path.join(self.path, "generation_" + str(globals.CURRENT_ROUND),
                    "General_DBScan_Color_Map" + str(globals.CURRENT_ROUND)))
 

    def individuality_color_map(self, include_first_encounters=False):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, x_max)
        self.ax.set_ylim(0, y_max)

        # self.graph, self.graph_ax = plt.subplots()
        # x_ticks = np.linspace(0, x_max, cols + 1)
        # y_ticks = np.linspace(0, y_max, rows + 1)
        # ax.set_xticks(x_ticks)
        # ax.set_yticks(y_ticks)
        # ax.xaxis.grid(True)
        # ax.yaxis.grid(True)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_title(
            "Individuality: White=No difference, Black=Always different")

        cmap = LinearSegmentedColormap.from_list(
            'rg', ["white", "black"], N=256)

        genotype_ratios = []

        for a in Agent.instances:
            genome = a.policy_table
            if not include_first_encounters:
                remove_first_encounters_from_memory(genome)

            defect_counter = 0
            cooperate_counter = 0
            for action in genome.values():
                if action == "d":
                    defect_counter += 1
                elif action == "c":
                    cooperate_counter += 1

            if cooperate_counter + defect_counter == 0:
                genotype_ratios.append(0)
            else:
                genotype_ratios.append(cooperate_counter /
                                    (cooperate_counter + defect_counter))

        for i, a in enumerate(Agent.instances):
            defect_counter = 0
            cooperate_counter = 0
            for pt in a.phenotype_memory:
                if pt == "c":
                    cooperate_counter += 1
                elif pt == "d":
                    defect_counter += 1

            phenotype_ratio = cooperate_counter / \
                (cooperate_counter + defect_counter)
            p = Polygon(a.corners(), facecolor=cmap(
                abs(genotype_ratios[i] - phenotype_ratio)), linewidth=0)

            self.ax.add_patch(p)
            # x_text_position = np.mean([a.corners()[0][0], a.corners()[2][0]])
            # y_text_position = np.mean([a.corners()[0][1], a.corners()[1][1]])
            # self.ax.text(x_text_position, y_text_position, round(
            #     abs(genotype_ratios[i] - phenotype_ratio), 3), fontsize=6)

        plt.savefig(os.path.join(self.path, "generation_" + str(globals.CURRENT_ROUND),
                    "Individuality_Color_Map" + str(globals.CURRENT_ROUND)))
        plt.clf()

    def single_memory_color_map(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, x_max)
        self.ax.set_ylim(0, y_max)

        # self.graph, self.graph_ax = plt.subplots()
        # x_ticks = np.linspace(0, x_max, cols + 1)
        # y_ticks = np.linspace(0, y_max, rows + 1)
        # ax.set_xticks(x_ticks)
        # ax.set_yticks(y_ticks)
        # ax.xaxis.grid(True)
        # ax.yaxis.grid(True)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_title("Green=TfT, Red=AD, Yellow=AC, Blue=X")

        for a in Agent.instances:
            x_text_position = a.corners()[0][0]
            y_text_position = np.mean([a.corners()[0][1], a.corners()[1][1]])

            color = ""
            if a.memory_length == 1:
                try:
                    if a.policy_table['c'] == 'c' and a.policy_table['d'] == 'd':
                        info = "TfT"
                        color = "green"
                    elif a.policy_table['c'] == 'c' and a.policy_table['d'] == 'c':
                        info = "AC"
                        color = "yellow"
                    elif a.policy_table['c'] == 'd' and a.policy_table['d'] == 'd':
                        info = "AD"
                        color = "red"
                    elif a.policy_table['c'] == 'd' and a.policy_table['d'] == 'c':
                        info = "X"
                        color = "blue"
                except:
                    info += "N/a"
            self.ax.text(x_text_position, y_text_position, info, fontsize=4)
            p = Polygon(a.corners(), facecolor=color, linewidth=0)

            self.ax.add_patch(p)

        plt.savefig(os.path.join(self.path, "generation_" +
                    str(globals.CURRENT_ROUND), "Single_Memory_Color_Map" + str(globals.CURRENT_ROUND)))
        plt.clf()
        plt.cla()
        plt.close('all')

    def memory_color_map(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, x_max)
        self.ax.set_ylim(0, y_max)

        # self.graph, self.graph_ax = plt.subplots()
        # x_ticks = np.linspace(0, x_max, cols + 1)
        # y_ticks = np.linspace(0, y_max, rows + 1)
        # ax.set_xticks(x_ticks)
        # ax.set_yticks(y_ticks)
        # ax.xaxis.grid(True)
        # ax.yaxis.grid(True)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_title("Memory: Light=Short, Dark=Long")

        cmap = LinearSegmentedColormap.from_list('rg', ["w", "b"], N=256)

        for a in Agent.instances:
            x_text_position = a.corners()[0][0]
            y_text_position = np.mean([a.corners()[0][1], a.corners()[1][1]])

            info = str(a.memory_length)
            if a.memory_length == 1:
                try:
                    if a.policy_table['c'] == 'c' and a.policy_table['d'] == 'd':
                        info += "-TfT"
                    elif a.policy_table['c'] == 'c' and a.policy_table['d'] == 'c':
                        info += "-AC"
                    elif a.policy_table['c'] == 'd' and a.policy_table['d'] == 'd':
                        info += "-AD"
                    elif a.policy_table['c'] == 'd' and a.policy_table['d'] == 'c':
                        info += "-X"
                except:
                    info += "N/a"
            self.ax.text(x_text_position, y_text_position, info, fontsize=4)
            p = Polygon(a.corners(), facecolor=cmap(
                a.memory_length / hp.MAX_MEMORY_LENGTH), linewidth=0)

            self.ax.add_patch(p)

        plt.savefig(os.path.join(self.path, "generation_" +
                    str(globals.CURRENT_ROUND), "Memory_Color_Map" + str(globals.CURRENT_ROUND)))
        plt.clf()
        plt.cla()
        plt.close('all')

    def plot_relative_health(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, x_max)
        self.ax.set_ylim(0, y_max)

        # self.graph, self.graph_ax = plt.subplots()
        # x_ticks = np.linspace(0, x_max, cols + 1)
        # y_ticks = np.linspace(0, y_max, rows + 1)
        # ax.set_xticks(x_ticks)
        # ax.set_yticks(y_ticks)
        # ax.xaxis.grid(True)
        # ax.yaxis.grid(True)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_title("Relative Score: Light=High, Dark=Low")

        # cmap = LinearSegmentedColormap.from_list('rg',["w", "b"], N=256)

        for a in Agent.instances:
            rh = a.relative_health()
            fc = [rh, rh, rh]
            p = Polygon(a.corners(), facecolor=fc, linewidth=0)

            self.ax.add_patch(p)

        plt.savefig(os.path.join(self.path, "generation_" +
                    str(globals.CURRENT_ROUND), "Relative_Health" + str(globals.CURRENT_ROUND)))
        plt.clf()

    def relative_health_DBScan(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, x_max)
        self.ax.set_ylim(0, y_max)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_title("Relative Score DBScan Map")

        data_points = []
        for a in Agent.instances:
            rh = a.relative_health() * 5
            data_points.append([a.row, a.col, rh])

        colors = {-1: [0,0,0]}
        clustering = DBSCAN(eps=hp.RELATIVE_HEALTH_EPS, min_samples=hp.RELATIVE_HEALTH_MIN_SAMPLES).fit(data_points)

        for i, a in enumerate(Agent.instances):
            label = clustering.labels_[i]
            color = ""
            if label in colors:
                color = colors[label]
            else:
                colors[label] = "#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
                color = colors[label]

            p = Polygon(a.corners(), facecolor=color, linewidth=0)

            self.ax.add_patch(p)

        # plt.show()
        plt.savefig(os.path.join(self.path, "generation_" + str(globals.CURRENT_ROUND),
                    "Relative_Health_DBScan_Color_Map" + str(globals.CURRENT_ROUND)))
        plt.clf()

    def plot_correlations(self, include_first_encounters=False):
        memory_lengths = []
        individuality = []
        success = []
        phenotype_cooperability = []
        genotype_cooperability = []

        for a in Agent.instances:
            memory_lengths.append(a.memory_length)
            success.append(a.health)

            genome = a.policy_table
            if not include_first_encounters:
                remove_first_encounters_from_memory(genome)

            g_defect_counter = 0
            g_cooperate_counter = 0
            for action in genome.values():
                if action == "d":
                    g_defect_counter += 1
                elif action == "c":
                    g_cooperate_counter += 1

            if g_cooperate_counter + g_defect_counter == 0:
                genotype_ratio = 0
            else:
                genotype_ratio = g_cooperate_counter / \
                    (g_cooperate_counter + g_defect_counter)
            genotype_cooperability.append(genotype_ratio)

            p_defect_counter = 0
            p_cooperate_counter = 0
            for pt in a.phenotype_memory:
                if pt == "c":
                    p_cooperate_counter += 1
                elif pt == "d":
                    p_defect_counter += 1

            phenotype_ratio = p_cooperate_counter / \
                (p_cooperate_counter + p_defect_counter)
            phenotype_cooperability.append(phenotype_ratio)

            individuality.append(abs(genotype_ratio - phenotype_ratio))

        sn.set()
        df = pd.DataFrame({
            "memory": memory_lengths,
            "individuality": individuality,
            "success": success,
            "geno_c": genotype_cooperability,
            "pheno_c": phenotype_cooperability
        })
        self.fig, self.ax = plt.subplots()
        self.ax.get_xaxis().set_visible(True)
        self.ax.get_yaxis().set_visible(True)
        corr = df.corr()
        corr.style.format(precision=2)
        corr.to_csv('correlation_grid.csv')

        def corr_sig(df=None):
            p_matrix = np.zeros(shape=(df.shape[1], df.shape[1]))
            for col in df.columns:
                for col2 in df.drop(col, axis=1).columns:
                    _, p = stats.pearsonr(df[col], df[col2])
                    p_matrix[df.columns.to_list().index(
                        col), df.columns.to_list().index(col2)] = p
            return p_matrix
        p_values = corr_sig(df=df)
        mask = np.invert(np.tril(p_values < 0.05))
        for i in range(len(mask)):
            mask[i][i] = True
        self.ax = sn.heatmap(corr, ax=None, annot=True,
                             cmap='coolwarm', vmin=-1, vmax=1, mask=mask)
        # self.ax.set_yticks(y_ticks)
        self.ax.set_title("Correlation Matrix")
        plt.savefig(os.path.join(self.path, "generation_" + str(globals.CURRENT_ROUND),
                    "Correlation_Matrix" + str(globals.CURRENT_ROUND)))
        plt.clf()

    def individuality_plot(self):
        for hl, individuality_per_round_lists in Agent.individuality_per_round.items():
            averages = [np.average(sub_list) for sub_list in individuality_per_round_lists]
            # print(hl, averages)
            plt.plot(averages, label="Memory length: " + str(hl))

        # plt.plot([1] * len(Agent.individuality_per_round))

        self.ax = plt.gca()
        self.ax.set_xlim(left=5)
        self.ax.set(xlabel='Round', ylabel='Individuality',
                    title="Individuality Over Time")
        self.ax.grid(True)
        # self.ax.legend(["Population's Average Individuality", "Max"])
        self.ax.legend()
        plt.savefig(os.path.join(self.path, "generation_" + str(globals.CURRENT_ROUND),
                    "Individuality Plot" + str(globals.CURRENT_ROUND)))
        plt.clf()
        plt.close('all')

    def calculate_individuality(self, include_first_encounters=False):
        individuality_by_memory_length = {}
        for i in range(1, Agent.max_memory_length + 1):
            individuality_by_memory_length[i] = []
        unique_genomes = {}
        copied_agents = copy.deepcopy(Agent.instances)
        for a in copied_agents:
            genome = a.policy_table
            if not include_first_encounters:
                remove_first_encounters_from_memory(genome)

            g_defect_counter = 0
            g_cooperate_counter = 0
            for action in genome.values():
                if action == "d":
                    g_defect_counter += 1
                elif action == "c":
                    g_cooperate_counter += 1

            try:
                genotype_ratio = g_cooperate_counter / \
                    (g_cooperate_counter + g_defect_counter)
            except:
                continue

            p_defect_counter = 0
            p_cooperate_counter = 0
            for pt in a.phenotype_memory:
                if pt == "c":
                    p_cooperate_counter += 1
                elif pt == "d":
                    p_defect_counter += 1

            try:
                phenotype_ratio = p_cooperate_counter / \
                    (p_cooperate_counter + p_defect_counter)
            except:
                continue

            individuality_by_memory_length[a.memory_length].append(abs(genotype_ratio - phenotype_ratio))

        for i in range(1, Agent.max_memory_length + 1):
            Agent.individuality_per_round[i].append(individuality_by_memory_length[i])


    def memory_lengths_over_time(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(auto=True)
        self.ax.set_ylim(auto=True)
        self.ax.get_xaxis().set_visible(True)
        self.ax.get_yaxis().set_visible(True)
        self.ax.grid(True)

        plt.plot(Agent.average_memory_length_per_round)
        plt.plot(Agent.max_memory_length_per_round)
        plt.plot(Agent.min_memory_length_per_round)
        # plt.plot([hp.MAX_MEMORY_LENGTH] * len(Agent.max_memory_length_per_round))

        self.ax = plt.gca()
        self.ax.set_xlim(left=5)
        self.ax.set(xlabel='Round', ylabel='Memory Length',
                    title="Evolution of Memory Length")
        self.ax.legend(["Average Memory Length Per Round", "Max Memory Length Per Round", "Min Memory Length Per Round", "Memory Length Cap"])
        self.ax.grid(True)
        plt.savefig(os.path.join(self.path, "generation_" + str(globals.CURRENT_ROUND),
                    "Memory Length Plot" + str(globals.CURRENT_ROUND)))
        plt.clf()
        plt.close('all')

    def single_memory_length_strategy_plot(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(auto=True)
        self.ax.set_ylim(auto=True)
        self.ax.get_xaxis().set_visible(True)
        self.ax.get_yaxis().set_visible(True)
        self.ax.grid(True)

        TfT_counts = Agent.single_memory_strategy_info["TfT"]["counts"]
        AC_counts = Agent.single_memory_strategy_info["AC"]["counts"]
        AD_counts = Agent.single_memory_strategy_info["AD"]["counts"]
        X_counts = Agent.single_memory_strategy_info["X"]["counts"]
        total_pops = [sum([a, b, c, d]) for a, b, c, d in zip(TfT_counts, AC_counts, AD_counts, X_counts)]


        plt.plot([100 * sub / tot for sub, tot in zip(TfT_counts, total_pops)])
        plt.plot([100 * sub / tot for sub, tot in zip(AC_counts, total_pops)])
        plt.plot([100 * sub / tot for sub, tot in zip(AD_counts, total_pops)])
        plt.plot([100 * sub / tot for sub, tot in zip(X_counts, total_pops)])

        self.ax = plt.gca()
        self.ax.set_xlim(left=5)
        self.ax.set(xlabel='Round', ylabel='% of agents with strategy',
                    title="Different Single Memory Strategies")
        self.ax.legend(["TfT", "Always Cooperate", "Always Defect", "Opposite of TfT"])
        self.ax.grid(True)
        plt.savefig(os.path.join(self.path, "generation_" + str(globals.CURRENT_ROUND),
                    "single_memory_strategies_plot_" + str(globals.CURRENT_ROUND)))
        plt.clf()
        plt.close('all')


    def plot_success_per_policy_for_one_memory(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(auto=True)
        self.ax.set_ylim(auto=True)
        self.ax.get_xaxis().set_visible(True)
        self.ax.get_yaxis().set_visible(True)
        self.ax.grid(True)

        y1 = [sum(avg_scores) / max(1, counts) for avg_scores, counts in zip(Agent.single_memory_strategy_info["TfT"]["avg_scores"], Agent.single_memory_strategy_info["TfT"]["counts"])]
        x1 = np.linspace(0, len(y1), len(y1))
        std_1 = [np.std(ss) for ss in Agent.single_memory_strategy_info["TfT"]["avg_scores"]]
        plt.fill_between(x1, [y - e for y, e in zip(y1, std_1)], [y + e for y, e in zip(y1, std_1)], alpha=0.2)
        plt.plot(y1)
       
        y2 = [sum(avg_scores) / max(1, counts) for avg_scores, counts in zip(Agent.single_memory_strategy_info["AD"]["avg_scores"], Agent.single_memory_strategy_info["AD"]["counts"])]
        x2 = np.linspace(0, len(y2), len(y2))
        std_2 = [np.std(ss) for ss in Agent.single_memory_strategy_info["AD"]["avg_scores"]]
        plt.fill_between(x2, [y - e for y, e in zip(y2, std_2)], [y + e for y, e in zip(y2, std_2)], alpha=0.2)
        plt.plot(y2)

        y3 = [sum(avg_scores) / max(1, counts) for avg_scores, counts in zip(Agent.single_memory_strategy_info["AC"]["avg_scores"], Agent.single_memory_strategy_info["AC"]["counts"])]
        x3 = np.linspace(0, len(y3), len(y3))
        std_3 = [np.std(ss) for ss in Agent.single_memory_strategy_info["AC"]["avg_scores"]]
        plt.fill_between(x3, [y - e for y, e in zip(y3, std_3)], [y + e for y, e in zip(y3, std_3)], alpha=0.2)
        plt.plot(y3)

        y4 = [sum(avg_scores) / max(1, counts) for avg_scores, counts in zip(Agent.single_memory_strategy_info["X"]["avg_scores"], Agent.single_memory_strategy_info["X"]["counts"])]
        x4 = np.linspace(0, len(y4), len(y4))
        std_4 = [np.std(ss) for ss in Agent.single_memory_strategy_info["X"]["avg_scores"]]
        plt.fill_between(x4, [y - e for y, e in zip(y4, std_4)], [y + e for y, e in zip(y4, std_4)], alpha=0.2)
        plt.plot(y4)

        self.ax = plt.gca()
        self.ax.set_xlim(left=5)
        self.ax.set(xlabel='Round', ylabel='Average Score',
                    title="Average Success of Different Single Memory Strategies")
        self.ax.legend(["TfT", "Always Cooperate", "Always Defect", "Opposite of TfT"])
        plt.savefig(os.path.join(self.path, "generation_" + str(globals.CURRENT_ROUND),
                    "success_per_policy_for_one_memory" + str(globals.CURRENT_ROUND)))
        plt.clf()
        plt.close('all')

    def calculate_single_memory_info(self):
        TfT_count = 0
        AD_count = 0
        AC_count = 0
        X_count = 0

        TfT_scores = []
        AC_scores = []
        AD_scores = []
        X_scores = []


        for a in Agent.instances:
            try:
                if a.policy_table['c'] == 'c' and a.policy_table['d'] == 'd':
                    TfT_count += 1
                    TfT_scores.append(a.health / (globals.CURRENT_ROUND + 1))
                elif a.policy_table['c'] == 'c' and a.policy_table['d'] == 'c':
                    AC_count += 1
                    AC_scores.append(a.health / (globals.CURRENT_ROUND + 1))
                elif a.policy_table['c'] == 'd' and a.policy_table['d'] == 'd':
                    AD_count += 1
                    AD_scores.append(a.health / (globals.CURRENT_ROUND + 1))
                elif a.policy_table['c'] == 'd' and a.policy_table['d'] == 'c':
                    X_count += 1
                    X_scores.append(a.health / (globals.CURRENT_ROUND + 1))
            except:
                continue
        
        # print(TfT_score_sum / (max(1, TfT_count) * (globals.CURRENT_ROUND + 1)))
        Agent.single_memory_strategy_info["TfT"]["counts"].append(TfT_count)
        Agent.single_memory_strategy_info["AC"]["counts"].append(AC_count)
        Agent.single_memory_strategy_info["AD"]["counts"].append(AD_count)
        Agent.single_memory_strategy_info["X"]["counts"].append(X_count)
        # print(TfT_score_sum, TfT_count)
        Agent.single_memory_strategy_info["TfT"]["avg_scores"].append(TfT_scores)
        Agent.single_memory_strategy_info["AC"]["avg_scores"].append(AC_scores)
        Agent.single_memory_strategy_info["AD"]["avg_scores"].append(AD_scores)
        Agent.single_memory_strategy_info["X"]["avg_scores"].append(X_scores)
    
    def memory_length_vs_health_histogram(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(auto=True)
        self.ax.set_ylim(auto=True)
        self.ax.get_xaxis().set_visible(True)
        self.ax.get_yaxis().set_visible(True)
        self.ax.grid(True)

        highest_memory_length = 0
        memory_length_to_average_health_map = {}
        for a in Agent.instances:
            highest_memory_length = max(highest_memory_length, a.memory_length) + 1
            try:
                memory_length_to_average_health_map[a.memory_length].append(a.health / (globals.CURRENT_ROUND + 1))
            except:
                memory_length_to_average_health_map[a.memory_length] = a.health / (globals.CURRENT_ROUND + 1)

        
        agent_counts_in_each_memory_length_bins = [0] * highest_memory_length
        for a in Agent.instances:
            agent_counts_in_each_memory_length_bins[a.memory_length] += 1
        cmap = LinearSegmentedColormap.from_list(
            'wb', ["white", "black"], N=highest_memory_length)
        
        memory_length_to_average_health_of_agents_in_corresponding_bin = {}
        for mem_len, healths in memory_length_to_average_health_map.items():
            memory_length_to_average_health_of_agents_in_corresponding_bin[mem_len] = np.mean(healths)
        
        y = [0] * (len(memory_length_to_average_health_of_agents_in_corresponding_bin) + 2)
        for mem_len, avg_health in memory_length_to_average_health_of_agents_in_corresponding_bin.items():
            y[mem_len] = avg_health

        y = y[1:] # knock off memory of 0 which can't exist
        x = range(1, len(y) + 1)

        ax = sn.barplot(x=list(x), y=y, palette=cmap(agent_counts_in_each_memory_length_bins[1:]), dodge=False)
        
        # plt.bar(x, y, align='center')

        self.ax.set(xlabel='Memory Length Bins', ylabel='Average Agent Health',
                    title="Memory Length vs. Agent Health")
        self.ax.grid(True)
        plt.plot()
        plt.savefig(os.path.join(self.path, "generation_" + str(globals.CURRENT_ROUND),
                    "memory_length_vs_health_histogram_" + str(globals.CURRENT_ROUND)))
        plt.clf()
        plt.close('all')

    def plot(self):
        # NOTE: This function will get overridden in experiments.py
        #       so this function will only run if merging.py is the
        #       file that got executed!

        self.plot_all_healths()
        # self.print_policy_tables()
        # self.print_heterogeneity_per_round()
        self.genotype_color_map()
        self.phenotype_color_map()
        self.individuality_color_map() # difference between genotype and phenotype
        self.memory_color_map()
        self.single_memory_color_map()
        self.plot_relative_health()
        self.plot_max_min_avg()
        self.plot_correlations() # BUG: Fail to allocate bitmap issue has something to do with this or individuality plot
        self.plot_success_per_policy_for_one_memory()
        
        # Make sure to turn on hp.should_calculate_individuality:
        self.individuality_plot()

        # Make sure to turn on hp.should_calculate_heterogeneity:
        self.plot_cooperability_ratio_graph()
        
        self.phenotype_DBScan()
        self.relative_health_DBScan()
        self.memory_lengths_over_time()
        self.general_DBScan_info(version="all_islands")
        pass


    def run(self):
        if self.display_animation:
            anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                           frames=500000,
                                           repeat=False)

            if len(sys.argv) > 1 and sys.argv[1] == '--save':
                writervideo = animation.FFMpegWriter(fps=1)
                anim.save('merge_sim.mp4', writer=writervideo)
                single_agents = filter(
                    lambda a: not a.super_agent, Agent.instances)
                multi_agents = set([a.super_agent for a in Agent.instances])
                multi_agents.remove(None)
                avg_single_agent_health = np.mean(
                    [sa.health for sa in single_agents])
                avg_multi_agent_health = np.mean(
                    [ma.health for ma in list(multi_agents)])
                new_DB_entry = [R, S, T, P, avg_single_agent_health,
                                avg_multi_agent_health]
                print(new_DB_entry)
                DB.append(new_DB_entry)
                plt.close('all')
            else:
                plt.show()
        else:
            for step in range(5000000000):
                self.animate(step)
                step += 1
                if globals.CURRENT_ROUND == hp.GENERATIONS_TO_PLOT[-1] + 1:
                    plt.cla()
                    plt.close('all')
                    break


if __name__ == '__main__':
    ob = cProfile.Profile()
    ob.enable()

    anim = Animate(display_animation=True)

    path_number = 0
    while os.path.exists("data_" + str(path_number) + "_" + str(date.today())):
        path_number += 1
    anim.path = "data_" + str(path_number) + "_" + str(date.today())
    # os.makedirs(anim.path)

    # for generation in hp.GENERATIONS_TO_PLOT:
    #     os.makedirs(os.path.join(anim.path, "generation_" + str(generation)))

    anim.run()

    # os.makedirs(os.path.join(
    #     anim.path, "generation_" + str(globals.CURRENT_ROUND)))
    # anim.plot()  # will automatically plot at the end.

    # ob.disable()
    # sec = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(ob, stream=sec).sort_stats(sortby)
    # ps.print_stats()

    # # print(sec.getvalue())
    # with open('profile.txt', 'a') as f:
    #     f.write(sec.getvalue())
