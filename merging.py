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

plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\pdsmi\\Desktop\\IPD\\ffmpeg-5.1.2-essentials_build\\bin\\ffmpeg.exe'

# rows, cols = (30, 30)
# rows, cols = (20, 20)
rows, cols = (10, 10)
# rows, cols = (5, 5)
# rows, cols = (3, 3)
x_max, y_max = (500, 500)
# R = 3
# S = 0
# T = 5
# P = 1
M = 0.0
# The evolved values from "Multiagent Reinforcement Learning in the Iterated Prisoner's Dilemma: Fast Cooperation through Evolved Payoffs" are below:
# R = 3.01
# S = -44.05
# T = 6.37
# P = -41.04
R = S = T = P = 1

DB = []

if M == S:
    # To prevent confusion
    raise

# TODO: Read about cooperative game theory and social niche construction by Powers

# Perhaps worth normalizing health in each step of the animation so we see who's doing better or worse relatively?


class Agent(object):
    # instances should only have sub-agents:
    instances = []
    heterogeneity_per_round = []
    cooperability_per_round = []
    individuality_per_round = []
    average_memory_length_per_round = []
    min_history_length_per_round = []
    max_history_length_per_round = []
    max_history_length = 5

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
    def calculate_heterogeneity(cls, include_first_encounters=False):
        num_defects = 0
        num_cooperates = 0

        unique_genomes = {}
        copied_agents = copy.deepcopy(Agent.instances)
        for a in copied_agents:
            sorted_genome = collections.OrderedDict(
                sorted(a.policy_table.items()))
            if not include_first_encounters:
                inputs_to_delete = []
                for input in sorted_genome:
                    if '0' in input:
                        inputs_to_delete.append(input)
                for input in inputs_to_delete:
                    del sorted_genome[input]
            genome_string = json.dumps(sorted_genome)
            if genome_string in unique_genomes:
                unique_genomes[genome_string] += 1
            else:
                unique_genomes[genome_string] = 1
            for action in sorted_genome.values():
                if action == "d":
                    num_defects += 1
                elif action == "c":
                    num_cooperates += 1

        Agent.heterogeneity_per_round.append(unique_genomes)
        if num_cooperates + num_defects == 0:
            Agent.cooperability_per_round.append(0)
        else:
            Agent.cooperability_per_round.append(
                num_cooperates / (num_cooperates + num_defects))

    @classmethod
    def mutate_population(cls):
        for a in Agent.instances:
            lowest = True
            for n in list(a.neighbors):
                if a.health_gained_this_round > n.health_gained_this_round:
                    lowest = False
                    break
            if lowest:
                a.num_times_mutated_in_a_row += 1
                if a.num_times_mutated_in_a_row == 3:
                    print("GETTING COPIED")
                    best_neighbor = a
                    for n in list(a.neighbors):
                        if n.health_gained_this_round > best_neighbor.health_gained_this_round:
                            best_neighbor = n
                    a.policy_table = copy.deepcopy(best_neighbor.policy_table)
                    if a.history_length > best_neighbor.history_length:
                        diff = a.history_length - best_neighbor.history_length
                        for k, v in a.history.items():
                            a.history[k] = v[diff:]
                    elif a.history_length < best_neighbor.history_length:
                        diff = best_neighbor.history_length - a.history_length
                        for k, v in a.history.items():
                            a.history[k] = diff * ['0'] + v
                    a.history_length = best_neighbor.history_length
                else:
                    for key in a.policy_table.keys():
                        if np.random.random() < .2:  # 20% mutation rate
                            print("MUTATING")
                            a.policy_table[key] = np.random.choice(["d", "c"])
                            
                            memory_length_mutation = np.random.random()
                            if memory_length_mutation <= .25:
                                if a.history_length < Agent.max_history_length:
                                    print("history length increasing")
                                    for history_list in a.history.values():
                                        history_list.insert(0, '0') # insert unknown history at the beginning
                                    
                                    a.history_length += 1
                                    new_policy_table = {}
                                    for k, v in a.policy_table.items():
                                        new_policy_table["c" + k] = v
                                        new_policy_table["d" + k] = v
                                    
                            elif memory_length_mutation <= .5:
                                if a.history_length > 1:
                                    a.history_length -= 1
                                    print("history length decreasing")
                                    new_policy_table = {}
                                    for k, v in a.policy_table.items():
                                        new_key = k[1:]
                                        if new_key in new_policy_table:
                                            if new_policy_table[new_key] == v:
                                                continue
                                            else:
                                                new_policy_table[new_key] = random.choice([v, new_policy_table[new_key]])
                                        else:
                                            new_policy_table[new_key] = v
                            # 50 % chance of no change in memory length
            else:
                a.num_times_mutated_in_a_row = 0
 
    def __init__(self, row, col):

        self.super_agent = None
        self.row = row
        self.col = col
        self.history_length = np.random.randint(
            1, Agent.max_history_length + 1)
        self.policy_table = {}
        self.neighbors = set()
        self.health = 0
        self.health_gained_this_round = 0
        self.health_data = []
        self.num_times_mutated_in_a_row = 0
        self.history = {}  # key = agent ID, value = array of past moves
        self.phenotype_history = []

    def get_neighbors(self, force=False):
        if self.super_agent and not force:
            return self.super_agent.get_neighbors()
        # Otherwise if no super-agent or force it:
        return self.neighbors

    def ranked_strategies(self, force=False):
        if self.super_agent and not force:
            return self.super_agent.ranked_strategies()
        # Otherwise if no super-agent or force it
        sw = self.strategy_weights.items()
        ranked_strategies = []
        while len(sw) > 0:
            strategies, weights = zip(*sw)
            choice = np.random.choice(strategies, p=weights)
            ranked_strategies.append(choice)
            sw = list(filter(lambda q: q[0] != choice, sw))
            # Update weights to sum to 1:
            total_weight = sum(q[-1] for q in sw)
            new_sw = []
            for q in sw:
                new_sw.append((q[0], q[-1] / total_weight))
            sw = new_sw
        return ranked_strategies

    def get_policy_output(self, opp_history):
        policy_key = "".join(opp_history)
        if policy_key not in self.policy_table:
            # TODO: If policy is 'merge', we are going to need a backup policy
            self.policy_table[policy_key] = np.random.choice(["c", "d"])

        return self.policy_table[policy_key]

    def play_against(self, opp):
        # If both are part of super-agents
        if self.super_agent and opp.super_agent:
            # And both are part of the *same* super-agent, it's a bug
            if self.super_agent == opp.super_agent:
                print("NOOOOO")
                raise
            else:
                return self.super_agent.play_against(opp.super_agent)
        # If I'm part of a group and my opponent isn't:
        elif self.super_agent and not opp.super_agent:
            return self.super_agent.play_against(opp)
        # If my opponent is part of a group and I'm not:
        elif (not self.super_agent) and opp.super_agent:
            return self.play_against(opp.super_agent)
        # Else play as below:

        if opp not in self.history:
            self.history[opp] = ['0'] * self.history_length
        if self not in opp.history:
            opp.history[self] = ['0'] * opp.history_length

        my_policy = self.get_policy_output(self.history[opp])
        opp_policy = opp.get_policy_output(opp.history[self])
        # if my_ranked_strats[0] == 'merge' and opp_ranked_strats[0] == 'merge':
        #     self.merge_with(opp)
        # For now, remove 'merge's
        if my_policy == 'merge' and opp_policy == 'merge':
            my_inc = 0
            opp_inc = 0
            self.merge_with(opp)
        elif my_policy == 'c' and opp_policy == 'c':
            my_inc = 8
            opp_inc = 8
        elif my_policy == 'c' and opp_policy == 'd':
            my_inc = 0
            opp_inc = 10
        elif my_policy == 'd' and opp_policy == 'c':
            my_inc = 10
            opp_inc = 0
        elif my_policy == 'd' and opp_policy == 'd':
            my_inc = 5
            opp_inc = 5
        else:
            # The only reason we should be here is if both voted to merge:
            if not (my_policy == 'merge' and opp_policy == 'merge'):
                raise

        self.health_gained_this_round += my_inc
        opp.health_gained_this_round += opp_inc
        self.increment_health(my_inc)
        opp.increment_health(opp_inc)

        for i in range(self.history_length - 1):
            self.history[opp][i] = self.history[opp][i + 1]
        self.history[opp][-1] = opp_policy

        
        for i in range(opp.history_length - 1):
            print(opp.history_length)
            try:
                opp.history[self][i] = opp.history[self][i + 1]
            except:
                print(opp.policy_table)
                print(opp.history)
                print("here")
                quit()
        opp.history[self][-1] = my_policy

        return (my_policy, my_inc, opp_policy, opp_inc)

    def merge_with(self, opp):
        if type(opp) == SuperAgent:
            opp.merge_with(self)
        elif self.super_agent == None and opp.super_agent == None:
            sa = SuperAgent(sub_agents=[self, opp])
        elif self.super_agent and opp.super_agent:
            self.super_agent.merge_with(opp.super_agent)
        elif self.super_agent and (not opp.super_agent):
            self.super_agent.merge_with(opp)
        elif (not self.super_agent) and opp.super_agent:
            opp.super_agent.merge_with(self)
        else:
            # This should never happen
            raise

    def increment_health(self, inc):
        self.health += inc

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
            raise
        self.sub_agents = sub_agents
        for sa in sub_agents:
            sa.super_agent = self
        self.health = 0
        self.health_gained_this_round = 0
        self.update_health()
        self.history = {}

    def get_neighbors(self, force=False):
        all_neighbors = set.union(
            *[sa.get_neighbors(force=True) for sa in self.sub_agents])
        my_members = set(self.sub_agents)
        # Return all my neighbors that aren't in me:
        return all_neighbors - my_members

    def update_health(self):
        avg_health = np.mean([sa.health for sa in self.sub_agents])
        for sa in self.sub_agents:
            sa.health = avg_health
        self.health = avg_health

    def merge_with(self, other):
        if isinstance(other, SuperAgent):
            self.sub_agents += other.sub_agents
            for osa in other.sub_agents:
                osa.super_agent = self
            del (other)
            self.update_health()
        elif isinstance(other, Agent):
            self.sub_agents.append(other)
            other.super_agent = self
            self.update_health()
        else:
            raise

    # TODO: Look at other algorithms that have been used for learning in
    # iterated PD, e.g. for agents with memory look at The Evolution of
    # Cooperation. Eventually it would be good to say, "what emerges resembles
    # paper XYZ."
    def ranked_strategies(self, force=False):
        sub_strats = [sa.ranked_strategies(force=True)
                      for sa in self.sub_agents]
        ranked_strats = []

        # While there any ranked votes left to be counted:
        while max([len(ss) for ss in sub_strats]) > 0:
            # Compute the first choice votes:
            first_choice_dict = {}
            for ss in sub_strats:
                first_choice = ss[0]
                if first_choice in first_choice_dict:
                    first_choice_dict[first_choice] += 1
                else:
                    first_choice_dict[first_choice] = 1

            # Tally up the first choice votes:
            max_first_choice_votes = max(first_choice_dict.values())
            winners = []
            for candidate in first_choice_dict.keys():
                if first_choice_dict[candidate] == max_first_choice_votes:
                    winners.append(candidate)
            # In case there are tied winners:
            winner = np.random.choice(winners)

            # Put the winner next in the list
            ranked_strats.append(winner)

            # Remove the winner from the next round:
            for ss in sub_strats:
                if winner in ss:
                    ss.remove(winner)

        return ranked_strats

    def increment_health(self, inc):
        divided_health = inc / len(self.sub_agents)
        for sa in self.sub_agents:
            sa.health += divided_health


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

        # self.graph, self.graph_ax = plt.subplots()
        # x_ticks = np.linspace(0, x_max, cols + 1)
        # y_ticks = np.linspace(0, y_max, rows + 1)
        # ax.set_xticks(x_ticks)
        # ax.set_yticks(y_ticks)
        # ax.xaxis.grid(True)
        # ax.yaxis.grid(True)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        self.num_rounds = 10
        Agent.populate()
        self.agents = Agent.instances
        np.random.shuffle(self.agents)
        self.current_round = 0
        self.current_agent_num = 0
        self.neighbors = list(
            self.agents[self.current_agent_num].get_neighbors())
        np.random.shuffle(self.neighbors)
        self.current_neighbor_num = 0
        self.patches = []

        self.max_scores = []
        self.avg_scores = []
        self.min_scores = []

        self.generations_to_plot = []
        self.path = ""

    def init(self):
        # agents = Agent.instances
        # return self.patches
        pass

    def animate(self, step):
        print(step)
        for p in self.ax.patches:
            p.remove()

        player0 = self.agents[self.current_agent_num]
        player1 = self.neighbors[self.current_neighbor_num]
        my_policy, inc0, opp_policy, inc1 = player0.play_against(player1)

        num_rounds_to_track_phenotype = 3
        # TODO: Change when we introduce merging
        if len(player0.phenotype_history) == (8 * num_rounds_to_track_phenotype):
            player0.phenotype_history = player0.phenotype_history[8:]

        player0.phenotype_history.append(my_policy)
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
                self.calculate_individuality()
                Agent.mutate_population()
                if self.current_round % 1 == 0:
                    Agent.calculate_heterogeneity()

                max_history_length = 1 
                min_history_length = Agent.max_history_length
                history_lengths = []
                for a in Agent.instances:
                    a.health_data.append(a.health)
                    a.health_gained_this_round = 0
                    if a.history_length > max_history_length:
                        max_history_length = a.history_length
                    if a.history_length < min_history_length:
                        min_history_length = a.history_length
                    history_lengths.append(a.history_length)
                Agent.average_memory_length_per_round.append(np.average(history_lengths))
                Agent.max_history_length_per_round.append(max_history_length)
                Agent.min_history_length_per_round.append(min_history_length)

                if self.current_round in self.generations_to_plot:
                    self.plot()

                self.current_round += 1
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
        plt.savefig(os.path.join(self.path, "generation_" + str(self.current_round),
                    "Max_Min_Avg_Scores" + str(self.current_round)))
        plt.clf()

    def plot_all_healths(self):
        for agent in Agent.instances:
            plt.plot([y / (x + 1) for x, y in enumerate(agent.health_data)])

        ax = plt.gca()
        ax.set(xlabel='Round', ylabel='Health',
               title="Agent Health Throughout Simulation")
        ax.grid(True)
        # ax.set_ylim([-1, 101])
        plt.savefig(os.path.join(self.path, "generation_" +
                    str(self.current_round), "Individual_Scores" + str(self.current_round)))
        plt.clf()

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
        plt.plot(Agent.cooperability_per_round)
        ax = plt.gca()
        ax.set(xlabel="Round", ylabel="Cooperability Ratio",
               title="Cooperability Ratios Throughout Simulation")
        ax.grid(True)
        plt.savefig(os.path.join(self.path, "generation_" + str(self.current_round),
                    "Cooperability_Ratio_Graph" + str(self.current_round)))
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
                inputs_to_delete = []
                for input in genome:
                    if '0' in input:
                        inputs_to_delete.append(input)
                for input in inputs_to_delete:
                    del genome[input]

            defect_counter = 0
            cooperate_counter = 0
            for action in genome.values():
                if action == "d":
                    defect_counter += 1
                elif action == "c":
                    cooperate_counter += 1

            p = Polygon(a.corners(), facecolor=cmap(
                cooperate_counter / (cooperate_counter + defect_counter)), linewidth=0)

            self.ax.add_patch(p)

        plt.savefig(os.path.join(self.path, "generation_" + str(self.current_round),
                    "Genotype_Color_Map" + str(self.current_round)))
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
            for pt in a.phenotype_history:
                if pt == "c":
                    cooperate_counter += 1
                elif pt == "d":
                    defect_counter += 1

            p = Polygon(a.corners(), facecolor=cmap(
                cooperate_counter / (cooperate_counter + defect_counter)), linewidth=0)

            self.ax.add_patch(p)

        # plt.show()
        plt.savefig(os.path.join(self.path, "generation_" + str(self.current_round),
                    "Phenotype_Color_Map" + str(self.current_round)))
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
            for pt in a.phenotype_history:
                if pt == "c":
                    cooperate_counter += 1
                elif pt == "d":
                    defect_counter += 1

            phenotype_value = 2 * (cooperate_counter /
                                   (cooperate_counter + defect_counter))
            data_points.append([a.row, a.col, phenotype_value])

        clustering = DBSCAN(eps=1.05, min_samples=2).fit(data_points)
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
        plt.savefig(os.path.join(self.path, "generation_" + str(self.current_round),
                    "Phenotype_DBScan_Color_Map" + str(self.current_round)))
        plt.clf()

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
                inputs_to_delete = []
                for input in genome:
                    if '0' in input:
                        inputs_to_delete.append(input)
                for input in inputs_to_delete:
                    del genome[input]

            defect_counter = 0
            cooperate_counter = 0
            for action in genome.values():
                if action == "d":
                    defect_counter += 1
                elif action == "c":
                    cooperate_counter += 1

            genotype_ratios.append(cooperate_counter /
                                   (cooperate_counter + defect_counter))

        for i, a in enumerate(Agent.instances):
            defect_counter = 0
            cooperate_counter = 0
            for pt in a.phenotype_history:
                if pt == "c":
                    cooperate_counter += 1
                elif pt == "d":
                    defect_counter += 1

            phenotype_ratio = cooperate_counter / \
                (cooperate_counter + defect_counter)
            p = Polygon(a.corners(), facecolor=cmap(
                abs(genotype_ratios[i] - phenotype_ratio)), linewidth=0)

            self.ax.add_patch(p)
            x_text_position = np.mean([a.corners()[0][0], a.corners()[2][0]])
            y_text_position = np.mean([a.corners()[0][1], a.corners()[1][1]])
            self.ax.text(x_text_position, y_text_position, round(
                abs(genotype_ratios[i] - phenotype_ratio), 3), fontsize=6)

        plt.savefig(os.path.join(self.path, "generation_" + str(self.current_round),
                    "Individuality_Color_Map" + str(self.current_round)))
        plt.clf()

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

            info = str(a.history_length)
            if a.history_length == 1:
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
                a.history_length / Agent.max_history_length), linewidth=0)

            self.ax.add_patch(p)

        plt.savefig(os.path.join(self.path, "generation_" +
                    str(self.current_round), "Memory_Color_Map" + str(self.current_round)))
        plt.clf()

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
                    str(self.current_round), "Relative_Health" + str(self.current_round)))
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
        clustering = DBSCAN(eps=1.015, min_samples=2).fit(data_points)

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
        plt.savefig(os.path.join(self.path, "generation_" + str(self.current_round),
                    "Relative_Health_DBScan_Color_Map" + str(self.current_round)))
        plt.clf()

    def plot_correlations(self, include_first_encounters=False):
        memory_lengths = []
        individuality = []
        success = []
        phenotype_cooperability = []
        genotype_cooperability = []

        for a in Agent.instances:
            memory_lengths.append(a.history_length)
            success.append(a.health)

            genome = a.policy_table
            if not include_first_encounters:
                inputs_to_delete = []
                for input in genome:
                    if '0' in input:
                        inputs_to_delete.append(input)
                for input in inputs_to_delete:
                    del genome[input]

            g_defect_counter = 0
            g_cooperate_counter = 0
            for action in genome.values():
                if action == "d":
                    g_defect_counter += 1
                elif action == "c":
                    g_cooperate_counter += 1

            genotype_ratio = g_cooperate_counter / \
                (g_cooperate_counter + g_defect_counter)
            genotype_cooperability.append(genotype_ratio)

            p_defect_counter = 0
            p_cooperate_counter = 0
            for pt in a.phenotype_history:
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
        plt.savefig(os.path.join(self.path, "generation_" + str(self.current_round),
                    "Correlation_Matrix" + str(self.current_round)))
        plt.clf()

    def individuality_plot(self):
        plt.plot(Agent.individuality_per_round, label="individuality")
        plt.plot([1] * len(Agent.individuality_per_round))

        self.ax = plt.gca()
        self.ax.set_xlim(left=5)
        self.ax.set(xlabel='Round', ylabel='Individuality',
                    title="Individuality Over Time")
        self.ax.grid(True)
        self.ax.legend(["Population's Average Individuality", "Max"])
        plt.savefig(os.path.join(self.path, "generation_" + str(self.current_round),
                    "Individuality Plot" + str(self.current_round)))
        plt.clf()
        plt.close('all')

    def calculate_individuality(self, include_first_encounters=False):
        individuality = []
        unique_genomes = {}
        copied_agents = copy.deepcopy(Agent.instances)
        for a in copied_agents:
            genome = a.policy_table
            if not include_first_encounters:
                inputs_to_delete = []
                for input in genome:
                    if '0' in input:
                        inputs_to_delete.append(input)
                for input in inputs_to_delete:
                    del genome[input]

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
            for pt in a.phenotype_history:
                if pt == "c":
                    p_cooperate_counter += 1
                elif pt == "d":
                    p_defect_counter += 1

            try:
                phenotype_ratio = p_cooperate_counter / \
                    (p_cooperate_counter + p_defect_counter)
            except:
                continue

            individuality.append(abs(genotype_ratio - phenotype_ratio))
        Agent.individuality_per_round.append(np.average(individuality))

    
    def memory_lengths_over_time(self):
        plt.plot(Agent.average_memory_length_per_round)
        plt.plot(Agent.max_history_length_per_round)
        plt.plot(Agent.min_history_length_per_round)
        plt.plot([Agent.max_history_length] * len(Agent.max_history_length_per_round))

        self.ax = plt.gca()
        self.ax.set_xlim(left=5)
        self.ax.set(xlabel='Round', ylabel='History Length',
                    title="Evolution of History Length")
        self.ax.legend(["Average History Length Per Round", "Max History Length Per Round", "Min History Length Per Round", "History Length Cap"])
        self.ax.grid(True)
        plt.savefig(os.path.join(self.path, "generation_" + str(self.current_round),
                    "History Length Plot" + str(self.current_round)))
        plt.clf()
        plt.close('all')




    def plot(self):
        # self.plot_all_healths()
        # self.plot_cooperability_ratio_graph()
        # self.print_policy_tables()
        # self.print_heterogeneity_per_round()
        # self.genotype_color_map()
        # self.phenotype_color_map()
        # self.individuality_color_map() # difference between genotype and phenotype
        # self.memory_color_map()
        # self.plot_relative_health()
        # self.plot_max_min_avg()
        # self.plot_correlations() # BUG: Fail to allocate bitmap issue has something to do with this or individuality plot
        # self.individuality_plot()
        # self.phenotype_DBScan()
        # self.relative_health_DBScan()
        self.memory_lengths_over_time()
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
                plt.close()
            else:
                plt.show()
        else:
            for step in range(50000000):
                self.animate(step)
                step += 1


if __name__ == '__main__':
    # ob = cProfile.Profile()
    # ob.enable()

    anim = Animate(display_animation=False)

    path_number = 0
    while os.path.exists("data_" + str(path_number) + "_" + str(date.today())):
        path_number += 1
    anim.path = "data_" + str(path_number) + "_" + str(date.today())
    os.makedirs(anim.path)

    anim.generations_to_plot = [1, 50, 100, 150, 200,
                                250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 800, 900, 1000, 1100, 1200, 1300]
    for generation in anim.generations_to_plot:
        os.makedirs(os.path.join(anim.path, "generation_" + str(generation)))

    anim.run()

    os.makedirs(os.path.join(
        anim.path, "generation_" + str(anim.current_round)))
    anim.plot()  # will automatically plot at the end.

    # ob.disable()
    # sec = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(ob, stream=sec).sort_stats(sortby)
    # ps.print_stats()

    # # print(sec.getvalue())
    # with open('profile.txt', 'a') as f:
    #     f.write(sec.getvalue())
