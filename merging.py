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


sys.setrecursionlimit(5000)

plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\pdsmi\\Desktop\\IPD\\ffmpeg-5.1.2-essentials_build\\bin\\ffmpeg.exe'

rows, cols = (hp.ROWS, hp.COLS)
x_max, y_max = (500, 500)

DB = []

# TODO: Read about cooperative game theory and social niche construction by Powers

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
    def mutate_population(cls):
        already_viewed_super_agents = {}
        for agent in Agent.instances:
            # Decide if current agent is the worst agent in the neighborhood.
            worst_performing_agent_in_neighborhood = True
            for neighbor in list(agent.neighbors):
                if agent.health_gained_this_round > neighbor.health_gained_this_round:
                    worst_performing_agent_in_neighborhood = False
                    break

            if worst_performing_agent_in_neighborhood:
                # If an agent performs the worst of its neighbors NUM_TIMES_AGENT_MUTATED_IN_A_ROW_CAP times
                # then it will just copy the genome of its most successful neighbor.
                agent.num_times_mutated_in_a_row += 1
                if agent.num_times_mutated_in_a_row == hp.NUM_TIMES_AGENT_MUTATED_IN_A_ROW_CAP:
                    # Finds the best neighbor and copies its policy table
                    best_neighbor = agent
                    for neighbor in list(agent.neighbors):
                        if neighbor.health_gained_this_round > best_neighbor.health_gained_this_round:
                            best_neighbor = neighbor
                    agent.policy_table = copy.deepcopy(best_neighbor.policy_table)
                    # Adjust the memory length to match its new policy table
                    agent.update_memory_length_to(best_neighbor.memory_length)
                else:
                    agent.mutate_policy_table()
                    if hp.MEMORY_LENGTH_CAN_EVOLVE:
                        # if the super agent hasn't been checked yet, maybe change its length
                        if agent.super_agent and not already_viewed_super_agents[agent.super_agent]:
                            agent.super_agent.maybe_mutate_memory_length()
                        # if the agent is not part of a super agent, maybe change its length
                        if not agent.super_agent:
                            agent.maybe_mutate_memory_length()
            else:
                agent.num_times_mutated_in_a_row = 0
            
            # Keep track of which super agents we have seen so that we are not 
            # repeatedly doing super agent targeted operations that should happen once.
            if agent.super_agent:
                already_viewed_super_agents[agent.super_agent] = True
 
    def __init__(self, row, col):
        self.super_agent = None  # points to the agent's SuperAgent

        self.memory = {}  # key = agent ID, value = array of past moves
        self.policy_table = {}  # key = memory (newest [left] -> oldest [right]), value = ("d" -> w1, "c" -> w2, "m" -> w3)
        self.neighbors = set()
        
        self.health = 0
        self.num_times_mutated_in_a_row = 0
        self.memory_length = 0; self.set_memory_length()  # how many actions back an agent remembers of its opponents

        self.row = row
        self.col = col

        # Data collection:
        self.health_data = []
        self.phenotype_memory = []
        self.health_gained_this_round = 0

       
    def set_memory_length(self):
        self.memory_length = hp.STARTING_MEMORY_LENGTH
        if self.memory_length == -1:
            # if STARTING_MEMORY_LENGTH is -1, then we set it randomly for each agent
            self.memory_length = np.random.randint(
                1, hp.MAX_MEMORY_LENGTH + 1)

    def get_neighbors(self, force=False):
        if self.super_agent and not force:
            return self.super_agent.get_neighbors()
        # Otherwise if no super-agent or force it:
        return self.neighbors
    
    def maybe_mutate_memory_length(self):     
        # Randomly decides whether or not to increase or shrink memory length by one.
        random_value = np.random.random()
        if random_value <= hp.MEMORY_LENGTH_INCREASE_MUTATION_RATE and self.memory_length < hp.MAX_MEMORY_LENGTH:
            self.grow_memory_by_one()
        elif random_value <= hp.MEMORY_LENGTH_DECREASE_MUTATION_RATE + hp.MEMORY_LENGTH_INCREASE_MUTATION_RATE and self.memory_length > 1:
            self.shrink_memory_by_one()
    
    def mutate(self):
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
            if (sum([cooperate_probability, defect_probability, hp.PERCENT_OF_MERGE_ACTIONS_IN_POLICY]) != 1):
                raise Exception("Action weights to not sum to 1 [complete]: Sum = ", sum([cooperate_probability, defect_probability, hp.PERCENT_OF_MERGE_ACTIONS_IN_POLICY]))
            self.policy_table[policy_key] = {
                "c": cooperate_probability,
                "d": defect_probability,
                "m": hp.PERCENT_OF_MERGE_ACTIONS_IN_POLICY,
            }
        # This path bases the change in weights off of the previous weights 
        elif hp.MUTATION_STRATEGY == "incremental":
            # Chooses random values to nudge each weight such that the sum of the nudges adds to zero
            # which keeps the sum of the weights equal to 1.0
            nudges = {'c': 0, 'd': 0, 'm': 0}
            for action in nudges.keys():
                nudges[action] = np.random.normal(.1, .1)
            mean = np.mean(nudges.values())
            for action in nudges.keys():
                nudges[action] -= mean

            for action in nudges.keys():
                if nudges[action] + self.policy_table[policy_key][action] < 0:
                    # retry if we are trying to shift the action weight below 0 
                    return self.initialize_or_mutate_policy_output(policy_key)
                else:
                    self.policy_table[policy_key][action] += nudges[action]
            if (sum(list(self.policy_table[policy_key].values())) != 1):
                raise Exception("Action weights to not sum to 1 [incremental]: Sum = ", sum(list(self.policy_table[policy_key].values())))
        else:
            print("LOG(DEBUG): MUTATION_STRATEGY hyperparameter was not found: ", hp.MUTATION_STRATEGY)

    def get_policy_output(self, memory_of_opp):
        policy_key = "".join(memory_of_opp)
        if policy_key not in self.policy_table:
            # If the agent has never seen this memory combination before, it will
            # build an output for it and store it in its policy table
            self.initialize_or_mutate_policy_output(policy_key)

        # Queries the policy table with the memory of the current opponent and randomly chooses
        # an action based on the associated weights
        (actions, weights) = self.policy_table[policy_key].keys(), self.policy_table[policy_key].values()
        return random.choice(list(actions), list(weights))

    def play_against(self, opp):
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
            my_inc = hp.MERGE_AGAINST_MERGE_POINTS
            opp_inc = hp.MERGE_AGAINST_MERGE_POINTS
            self.merge_with(opp)
        elif my_action == 'm' and opp_action == 'c':
            my_inc = hp.COOPERATE_AGAINST_COOPERATE_POINTS
            opp_inc = hp.COOPERATE_AGAINST_COOPERATE_POINTS
        elif my_action == 'c' and opp_action == 'm':
            my_inc = hp.COOPERATE_AGAINST_COOPERATE_POINTS
            opp_inc = hp.COOPERATE_AGAINST_COOPERATE_POINTS
        elif my_action == 'd' and opp_action == 'm':
            my_inc = hp.DEFECT_AGAINST_COOPERATE_POINTS
            opp_inc = hp.COOPERATE_AGAINST_DEFECT_POINTS
        elif my_action == 'm' and opp_action == 'd':
            my_inc = hp.COOPERATE_AGAINST_DEFECT_POINTS
            opp_inc = hp.COOPERATE_AGAINST_DEFECT_POINTS
        elif my_action == 'c' and opp_action == 'c':
            my_inc = hp.COOPERATE_AGAINST_COOPERATE_POINTS
            opp_inc = hp.COOPERATE_AGAINST_COOPERATE_POINTS
        elif my_action == 'c' and opp_action == 'd':
            my_inc = hp.COOPERATE_AGAINST_DEFECT_POINTS
            opp_inc = hp.DEFECT_AGAINST_COOPERATE_POINTS
        elif my_action == 'd' and opp_action == 'c':
            my_inc = hp.DEFECT_AGAINST_COOPERATE_POINTS
            opp_inc = hp.COOPERATE_AGAINST_DEFECT_POINTS
        elif my_action == 'd' and opp_action == 'd':
            my_inc = hp.DEFECT_AGAINST_DEFECT_POINTS
            opp_inc = hp.DEFECT_AGAINST_DEFECT_POINTS
        else:
            raise Exception("Unknown action taken by an agent.")

        self.health_gained_this_round += my_inc
        opp.health_gained_this_round += opp_inc
        self.increment_health(my_inc)
        opp.increment_health(opp_inc)

        # Update memories based on opponents previous move
        for i in range(self.memory_length - 1):
            self.memory[opp][i] = self.memory[opp][i + 1]
        self.memory[opp][-1] = opp_action

        if type(self) == SuperAgent:
            for sa in self.sub_agents:
                for i in range(sa.memory_length - 1):
                    self.memory[opp][i] = sa.memory[opp][i + 1]
                sa.memory[opp][-1] = opp_action
        
        for i in range(opp.memory_length - 1):
            opp.memory[self][i] = opp.memory[self][i + 1]
        opp.memory[self][-1] = my_action

        if type(opp) == SuperAgent:
            for sa in opp.sub_agents:
                for i in range(sa.memory_length - 1):
                    opp.memory[self][i] = sa.memory[self][i + 1]
                sa.memory[self][-1] = my_action

        return (my_action, my_inc, opp_action, opp_inc)

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

    def increment_health(self, inc):
        self.health += inc

    def shrink_memory_by_one(self):
        # Need at least a memory length of 1
        if self.memory_length <= 1:
            return
        
        self.memory_length -= 1
        self.memory = self.memory[:-1]

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

    def grow_memory_by_one(self):
        if self.memory >= hp.MAX_MEMORY_LENGTH:
            return

        self.memory_length += 1
        self.memory.append('0')

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
        
    def grow_memory_by_n(self, n):
        for _ in range(n):
            self.grow_memory_by_one()

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
            raise Exception("SuperAgent is a sub Agent of a SuperAgent")
        self.sub_agents = sub_agents
        for sa in sub_agents:
            sa.super_agent = self
        self.health = 0
        self.health_gained_this_round = 0
        self.update_health()
        self.merge_memories()
        self.memory = {}

        self.make_memories_equal_size()

    # Forces all sub agents to have the the same memory length
    def make_memories_equal_size(self):
        average_memory_length = np.round(np.mean([sa.memory_length for sa in self.sub_agents]))

        for agent in self.sub_agents:
            if agent.memory_length < average_memory_length:
                agent.grow_memory_by_n(average_memory_length - agent.memory_length)
            elif agent.memory_length > average_memory_length:
                agent.shrink_memory_by_n(agent.memory_length - average_memory_length)

        if hp.DEV:
            for agent in self.sub_agents:
                if agent.memory_length != average_memory_length:
                    raise Exception("Agent's memory length does not match the super agent's memory length")
                if len(agent.policy_table[0]) != average_memory_length:
                    raise Exception("Policy table's memory input does not match the agent's memory length")

    # 
    def merge_memories(self):
        opponent_to_memories_map = {}
        # Builds a map of {opponent -> list[memory1, memory2, ...]}
        for sub_agent in self.sub_agents:
            for (opponent, memory) in sub_agent.memory.items():
                if opponent in opponent_to_memories_map:
                    opponent_to_memories_map[opponent].append(memory)
                else:
                    opponent_to_memories_map[opponent] = list(memory)

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
        all_neighbors = set.union(
            *[sa.get_neighbors(force=True) for sa in self.sub_agents])
        my_members = set(self.sub_agents)
        # Return all my neighbors that aren't in me:
        return all_neighbors - my_members

    # Sets the SuperAgent's health to the average of all the sub agents' healths.
    def update_health(self):
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

        self.update_health()
        self.make_memories_equal_size()
        self.merge_memories()

    
    def increment_health(self, inc):
        divided_health = inc / len(self.sub_agents)
        for sa in self.sub_agents:
            sa.health += divided_health


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
                sub_agent.grow_memory_by_one()
            elif direction == "shrink" and sub_agent.memory_length > 1:
                sub_agent.shrink_memory_by_one()


class Animate:
    def __init__(self, display_animation):
        # , num_rounds, agents, current_round, current_agent_num, current_neighbor_num, neighbors
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
        self.agents = Agent.instances
        np.random.shuffle(self.agents)
        self.current_agent_num = 0
        self.neighbors = list(
            self.agents[self.current_agent_num].get_neighbors())
        np.random.shuffle(self.neighbors)
        self.current_neighbor_num = 0
        self.patches = []

        self.max_scores = []
        self.avg_scores = []
        self.min_scores = []

        self.path = ""
        
    def set_path(self, path):
        self.path = path
        # print("SET PATH: ", self.path)

    def init(self):
        # agents = Agent.instances
        # return self.patches
        pass


    def animate(self, step):
        # print(step)
        for p in self.ax.patches:
            p.remove()

        player0 = self.agents[self.current_agent_num]
        player1 = self.neighbors[self.current_neighbor_num]
        my_policy, inc0, opp_policy, inc1 = player0.play_against(player1)

        # TODO: Change when we introduce merging
        if len(player0.phenotype_memory) == (8 * hp.NUM_ROUNDS_TO_TRACK_PHENOTYPE):
            player0.phenotype_memory = player0.phenotype_memory[8:]

        player0.phenotype_memory.append(my_policy)
        if self.display_animation:
            if my_policy == "merge":
                fc = "yellow"
            elif my_policy == "c":
                fc = "green"
            elif my_policy == "d":
                fc = "red"
            else:
                fc = "purple"
            p1 = Polygon(player0.corners(), facecolor=fc, linewidth=0)
            self.patches.append(p1)

            if opp_policy == "merge":
                fc = "yellow"
            elif opp_policy == "c":
                fc = "green"
            elif opp_policy == "d":
                fc = "red"
            else:
                fc = "purple"

            p2 = Polygon(player1.corners(), facecolor=fc, linewidth=0)
            self.patches.append(p2)

            for a in Agent.instances:
                if a is not player0 and a is not player1:
                    rh = a.relative_health()
                    fc = [rh, rh, rh]
                    pg = Polygon(a.corners(), facecolor=fc, linewidth=0)
                    self.patches.append(pg)

        self.current_neighbor_num += 1

        if self.current_neighbor_num == len(self.neighbors):
            # New Agent
            self.current_agent_num += 1
            self.current_neighbor_num = 0
            if self.current_agent_num == len(self.agents):
                # New Round
                print("Current round: ", globals.CURRENT_ROUND)
                if hp.SHOULD_CALCULATE_INDIVIDUALITY:
                    self.calculate_individuality()

                if hp.SHOULD_CALCULATE_SINGLE_MEMORY_STRATEGIES:
                    self.calculate_single_memory_info()

                Agent.mutate_population() 

                if hp.SHOULD_CALCULATE_HETEROGENEITY:
                    # print("calculating heterogeneity")
                    Agent.calculate_heterogeneity_and_cooperability()

                max_memory_length = 1 
                min_memory_length = hp.MAX_MEMORY_LENGTH
                memory_lengths = []
                for a in Agent.instances:
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
                    self.plot()

                globals.CURRENT_ROUND += 1
                self.current_agent_num = 0
                np.random.shuffle(self.agents)

                # Data collection at the end of each round
                self.max_scores.append(Agent.best_health())
                self.avg_scores.append(Agent.average_health())
                self.min_scores.append(Agent.worst_health())

            self.neighbors = list(
                self.agents[self.current_agent_num].get_neighbors())
            np.random.shuffle(self.neighbors)

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
            for step in range(50000000):
                self.animate(step)
                step += 1
                if globals.CURRENT_ROUND == hp.GENERATIONS_TO_PLOT[-1] + 1:
                    plt.cla()
                    plt.close('all')
                    break


if __name__ == '__main__':
    # ob = cProfile.Profile()
    # ob.enable()

    anim = Animate(display_animation=False)

    path_number = 0
    while os.path.exists("data_" + str(path_number) + "_" + str(date.today())):
        path_number += 1
    anim.path = "data_" + str(path_number) + "_" + str(date.today())
    os.makedirs(anim.path)

    for generation in hp.GENERATIONS_TO_PLOT:
        os.makedirs(os.path.join(anim.path, "generation_" + str(generation)))

    anim.run()

    os.makedirs(os.path.join(
        anim.path, "generation_" + str(globals.CURRENT_ROUND)))
    anim.plot()  # will automatically plot at the end.

    # ob.disable()
    # sec = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(ob, stream=sec).sort_stats(sortby)
    # ps.print_stats()

    # # print(sec.getvalue())
    # with open('profile.txt', 'a') as f:
    #     f.write(sec.getvalue())
