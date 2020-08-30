import random
import itertools
import numpy as np
from itertools import combinations
import os

SAFE = 0
MINE = 1


class Statement():
    def __init__(self, undiscovered_cells, mines_count):
        self.mines_count = mines_count  # constrain - sum (undiscovered variables==count)
        self.variables = set(undiscovered_cells)

    def get_length(self):
        return len(self.variables)

    def __eq__(self, other):
        return self.variables == other.variables and self.mines_count == other.mines_count

    def __str__(self):
        return str(self.variables) + "=" + str(self.mines_count)

    def get_known_mines(self):

        if len(self.variables) == self.mines_count:
            return self.variables
        return None

    def get_safe_cells(self):
        """
        The following
        :return:
        """
        if self.mines_count == 0:
            return self.variables
        return None

    def register_cell_is_mine(self, cell):

        newCells = set()
        for item in self.variables:
            if item != cell:
                newCells.add(item)
            else:
                self.mines_count -= 1
        self.variables = newCells

    def register_cell_is_safe(self, cell):

        cells = set()
        for item in self.variables:
            if item != cell:
                cells.add(item)
        self.variables = cells

    def evaluate(self, assignments):
        """
        The following evaluates whether the assignment satisfies the
        statement
        :param assignments:
        :return: False,True
        """
        count_mines = sum([assignments[cell] for cell in self.variables])
        if count_mines != self.mines_count: return False
        return True


class SAT_AI:
    def __init__(self, height=8, width=8, no_of_mines_in_board=0, prob_mode=(1, 1)):
        if os.path.isfile('PDB.npy'):
            self._pattern_db = np.load('PDB.npy', allow_pickle=True).item()
        else:
            self._pattern_db = dict()
        self.height = height
        self.width = width
        self.no_of_mines_in_board = no_of_mines_in_board
        self.moves_made = set()
        self.mines = set()
        self.safes = set()
        self.knowledge_base = []
        self._first_move = True
        self.open_information = dict()
        self._prob_mode = prob_mode

    def get_configuration(self, top_left=(0, 0)):
        """
        The following returns a 4X4 configuration on the board as (mines,open cells)
        :param top_left: The top left point of the 4x4 configuration square
        :return:(mines,open cells)
        """
        start_h, start_w = top_left
        stop_h, stop_w = min(self.height, start_h + 4), min(self.width, start_w + 4)
        mines, opens = frozenset(), frozenset()
        for h in range(start_h, stop_h):
            for w in range(start_w, stop_w):
                if (h, w) in self.mines:
                    mines = mines.union({(h - start_h, w - start_w)})
                elif (h, w) in self.open_information:
                    opens = opens.union({((h - start_h, w - start_w), self.open_information[(h, w)])})
        return mines, opens

    def calculate_probabilities(self, sentences):
        if len(sentences) == 0:
            return
        satisfying_assignments = None
        # combined_variables = set()
        assignment = dict()
        all_ones = [combinations(range(len(sentence.variables)), sentence.mines_count) for sentence in sentences]
        for combination in itertools.product(*all_ones, repeat=1):
            assignment = dict()
            for combo, sentence in zip(combination, sentences):
                variables = sentence.variables
                for index, variable in enumerate(variables):
                    value = MINE if index in combo else SAFE
                    if variable not in assignment.keys():
                        assignment[variable] = value
                    elif value != assignment[variable]:
                        continue
            if all([sentence.evaluate(assignment) for sentence in sentences]):
                satisfying_assignments = np.array(list(assignment.values())) \
                    if satisfying_assignments is None else np.vstack([satisfying_assignments,
                                                                      np.array(list(assignment.values()))])
        probabilities = np.sum(satisfying_assignments, axis=0) / satisfying_assignments.shape[0]
        return probabilities, list(assignment.keys())

    def find_closest_k_sentences(self, sentence, k):
        intersections = []
        for other_sentence in self.knowledge_base:
            res = len(other_sentence.variables.intersection(sentence.variables))
            intersections.append((other_sentence, res))
        intersections = sorted(intersections, key=lambda a: a[1], reverse=True)

        return [inter[0] for inter in intersections[:k] if inter[1] > 0]

    def annotate_mine_cell(self, cell):

        self.mines.add(cell)
        for statement in self.knowledge_base:
            statement.register_cell_is_mine(cell)

    def annotate_safe_cell(self, cell):
        self.safes.add(cell)
        for statement in self.knowledge_base:
            statement.register_cell_is_safe(cell)

    def extend_knowledge_base(self, cell, count):

        self.annotate_safe_cell(cell)
        self.moves_made.add(cell)
        self.open_information[cell] = count
        neighbors, count = self.get_cell_neighbors(cell, count)
        sentence = Statement(neighbors, count)
        self.knowledge_base.append(sentence)
        if (len(self.safes - self.moves_made) > 0):
            return
        new_knowledge = []
        for other_sentence in self.knowledge_base:
            if other_sentence == sentence:
                continue
            elif other_sentence.variables.issuperset(sentence.variables):
                self.subset_resolution(new_knowledge, other_sentence, sentence)
            elif sentence.variables.issuperset(other_sentence.variables):
                self.subset_resolution(new_knowledge, sentence, other_sentence)
        self.knowledge_base.extend(new_knowledge)
        if (len(self.safes - self.moves_made) > 0):
            return
        for sentence in self.knowledge_base:
            closest_k = self.find_closest_k_sentences(sentence, 6)
            if len(closest_k) == 0: continue
            probabilities, values = self.calculate_probabilities(closest_k)
            safe_mines = np.where(probabilities == 1)
            safe_opening = np.where(probabilities == 0)
            for index in safe_mines[0]:
                self.annotate_mine_cell(values[index])
            for index in safe_opening[0]:
                self.annotate_safe_cell(values[index])
        self.duplication_removal()
        self.infer_from_complete_sentences()

    def subset_resolution(self, new_knowledge, other_sentence, sentence):
        difference = other_sentence.variables - sentence.variables
        if other_sentence.mines_count == sentence.mines_count:
            for safeFound in difference:
                self.annotate_safe_cell(safeFound)
        elif len(difference) == other_sentence.mines_count - sentence.mines_count:
            for mineFound in difference:
                self.annotate_mine_cell(mineFound)
        else:
            new_knowledge.append(
                Statement(difference, other_sentence.mines_count - sentence.mines_count)
            )

    def make_safe_move(self):
        safeCells = self.safes - self.moves_made
        if not safeCells:
            return None
        move = safeCells.pop()
        return move

    def use_pattern_db(self):
        moves_and_probabilites = dict()
        for h in range(self.height - 4):
            for w in range(self.width - 4):
                current_conf = self.get_configuration((h, w))
                if current_conf in self._pattern_db:
                    relative_move, probability = self._pattern_db[current_conf]
                    absolute_move = relative_move[0] + h, relative_move[1] + w
                    moves_and_probabilites[absolute_move] = probability  # how to update ??
        if len(moves_and_probabilites.keys()) > 0:
            winner = max(moves_and_probabilites, key=moves_and_probabilites.get)
            prob = moves_and_probabilites[winner]
            print("DB is Used with probability:", prob)
            return winner, prob
        return None, None

    def make_random_move(self):
        """
        1,1 - average assignments+DB
        0,1 - average assignments
        1,0 - pattern db ->random
        0,0 - random

        """
        if self._first_move:
            self._first_move = False
            return self.get_move_uniform_at_random()

        if self._prob_mode == (1, 1):
            pattern_winner, pattern_prob = self.use_pattern_db()
            if pattern_winner: return pattern_winner
            return self.average_assignments()
        elif self._prob_mode == (0, 1):
            return self.average_assignments()
        elif self._prob_mode == (1, 0):
            pattern_winner, pattern_prob = self.use_pattern_db()
            if pattern_winner:
                return pattern_winner
            else:
                return self.get_move_uniform_at_random()
        else:
            return self.get_move_uniform_at_random()

    def average_assignments(self):
        min_move, min_value = None, 1
        max_move, max_value = None, 0
        for sentence in self.knowledge_base:
            closest_k = self.find_closest_k_sentences(sentence, 6)
            if len(closest_k) != 0:
                probabilities, values = self.calculate_probabilities(closest_k)
                min_move_index, min_move_value = np.argmin(probabilities), np.min(probabilities)
                max_move_index, max_move_value = np.argmax(probabilities), np.max(probabilities)
                if min_move_value <= min_value:
                    min_move, min_value = values[min_move_index], min_move_value
                if max_move_value >= max_value:
                    max_move, max_value = values[max_move_index], max_move_value
        uniform_at_random_probability = self.calculate_mine_prob()
        # debug
        if uniform_at_random_probability <= 0:
            print("WEIRD (or game ended)")
            pass
        best_safe = min(min_value, uniform_at_random_probability)
        best_mine = max_value
        if 1 - best_safe < best_mine:
            print("Flagged unsafely", max_move, "with probability", max_value)
            self.annotate_mine_cell(max_move)
            return self.make_random_move()
        if uniform_at_random_probability < min_value:
            print("choosing uniform with probability", uniform_at_random_probability)
            return self.get_move_uniform_at_random()
        print("choosing smartly with probability", min_value)
        return min_move

    def calculate_mine_prob(self):
        uniform_at_random_probability = (self.no_of_mines_in_board - len(self.mines)) / (
                self.width * self.height - len(self.mines) - len(self.moves_made)) if (
                                                                                              self.width * self.height - len(
                                                                                          self.mines) - len(
                                                                                          self.moves_made)) > 0 else 0
        return uniform_at_random_probability

    def get_move_uniform_at_random(self):
        all_possible_moves = set()
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) not in self.mines and (i, j) not in self.moves_made:
                    all_possible_moves.add((i, j))
        if len(all_possible_moves) == 0:
            return None
        move = random.choice(tuple(all_possible_moves))
        return move

    def get_cell_neighbors(self, cell, count):
        i, j = cell
        neighbors = []
        for row in range(i - 1, i + 2):
            for col in range(j - 1, j + 2):
                if (row >= 0 and row < self.height) \
                        and (col >= 0 and col < self.width) \
                        and (row, col) != cell \
                        and (row, col) not in self.safes \
                        and (row, col) not in self.mines:
                    neighbors.append((row, col))
                if (row, col) in self.mines:
                    count -= 1
        return neighbors, count

    def duplication_removal(self):
        """
        The following function clears duplications from the knowledge base
        (2 or more instances of the same statement)
        :return:None
        """
        unique_knowledge_base = []
        for statement in self.knowledge_base:
            if statement not in unique_knowledge_base and statement.get_length() > 0:
                unique_knowledge_base.append(statement)
        self.knowledge_base = unique_knowledge_base

    def infer_from_complete_sentences(self):
        """
        The following checks whether there is a sentence
        which the assignment of all it's variables clear,
        i.e case #mines=#variables it's clear the all the
        variables are known mines,
        and in case that #mines=0 it's known
         that all the variables are safes.
        :return: None
        """
        new_KB = []
        for statement in self.knowledge_base:
            new_KB.append(statement)
            if statement.get_known_mines():
                for mineFound in statement.get_known_mines():
                    self.annotate_mine_cell(mineFound)
                new_KB.pop(-1)
            elif statement.get_safe_cells():
                for safeFound in statement.get_safe_cells():
                    self.annotate_safe_cell(safeFound)
                new_KB.pop(-1)
        self.knowledge_base = new_KB
