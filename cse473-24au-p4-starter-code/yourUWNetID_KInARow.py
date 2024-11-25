'''
<yourUWNetID>_KInARow.py
Authors: <your name(s) here, lastname first and partners separated by ";">
  Example:  
    Authors: Smith, Jane; Lee, Laura

An agent for playing "K-in-a-Row with Forbidden Squares" and related games.
CSE 473, University of Washington

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.
YOU CAN ADD WHATEVER ADDITIONAL FUNCTIONS YOU NEED IN ORDER
TO PROVIDE A GOOD STRUCTURE FOR YOUR IMPLEMENTATION.

'''

from agent_base import KAgent
from game_types import State, Game_Type

AUTHORS = 'Sameeksha Sharma and Neha Pinni'

import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id
)


class OurAgent(KAgent):

    def __init__(self, twin=False):
        self.twin = twin
        self.nickname = 'Nic'
        if twin: self.nickname += '2'
        self.long_name = 'Wizard of KRows'
        if twin: self.long_name += ' II'
        self.persona = 'kind'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "don't know yet"
        self.WHO_MY_OPPONENT_PLAYS = None
        self.MY_PAST_UTTERANCES = []
        self.OPPONENT_PAST_UTTERANCES = []
        self.UTTERANCE_COUNT = 0
        self.REPEAT_COUNT = 0
        self.move_history = []
        self.game_state_history = []
        self.minimax_states_evaluated = 0
        self.minimax_cutoffs = 0

    def introduce(self):
        intro = '\nMy name is Wizard of KRows.\n' + \
                '"Sami and Neha" made me.\n' + \
                'I am here to WIN!\n'
        if self.twin: intro += "By the way, I'm the TWIN.\n"
        return intro

    def prepare(
        self,
        game_type,
        what_side_to_play,
        opponent_nickname,
        expected_time_per_move=0.1,
        utterances_matter=True):

        self.game_type = game_type
        self.playing = what_side_to_play
        self.WHO_MY_OPPONENT_PLAYS = self.other(what_side_to_play)
        self.opponent_nickname = opponent_nickname
        self.time_limit = expected_time_per_move
        self.utterances_matter = utterances_matter
        return "OK"

    def makeMove(self, currentState, currentRemark, timeLimit=10000):
        self.UTTERANCE_COUNT += 1
        self.OPPONENT_PAST_UTTERANCES.append(currentRemark)
        
        if currentRemark == "Tell me how you did that":
            special_utterance = self.explainLastMove()
        elif currentRemark == "What's your take on the game so far?":
            special_utterance = self.storyOfTheGame()
        else:
            special_utterance = None

        self.minimax_states_evaluated = 0
        self.minimax_cutoffs = 0
        depth = 4
        bestScore, bestMove, cutoffs = self.minimax(state=currentState, depthRemaining=depth, pruning=True, alpha=float("-inf"), beta=float("inf"))
        self.minimax_cutoffs = cutoffs
        if bestMove is None:
            return [[[0, 0], currentState], self.generateUtterance(currentState, "No valid moves!")]

        newState = self.applyMove(currentState, bestMove)
        if special_utterance:
            newRemark = special_utterance
        else:
            newRemark = self.generateUtterance(currentState, "Choosing the best move.")
        
        return [[bestMove, newState], newRemark]

    def minimax(self, state, depthRemaining, pruning=False, alpha=None, beta=None):
        self.minimax_states_evaluated += 1

        if depthRemaining == 0 or self.isTerminal(state):
            return [self.staticEval(state), None, 0]

        bestMove = None
        cutoffs = 0
        if state.whose_move == self.playing:
            maxEval = float("-inf")
            for move, nextState in self.orderChildren(state):
                eval, _, childCutoffs = self.minimax(nextState, depthRemaining - 1, pruning, alpha, beta)
                cutoffs += childCutoffs
                if eval > maxEval:
                    maxEval = eval
                    bestMove = move
                if pruning:
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        cutoffs += 1
                        break
            return [maxEval, bestMove, cutoffs]
        else:
            minEval = float("inf")
            for move, nextState in self.orderChildren(state):
                eval, _, childCutoffs = self.minimax(nextState, depthRemaining - 1, pruning, alpha, beta)
                cutoffs += childCutoffs
                if eval < minEval:
                    minEval = eval
                    bestMove = move
                if pruning:
                    beta = min(beta, eval)
                    if beta <= alpha:
                        cutoffs += 1
                        break
            return [minEval, bestMove, cutoffs]

    def staticEval(self, state):
        my_score = self.countPotential(state, self.playing)
        opp_score = self.countPotential(state, self.WHO_MY_OPPONENT_PLAYS)
        return my_score - opp_score

    def orderChildren(self, state):
        children = self.generateSuccessors(state)
        return sorted(children, key=lambda child: self.staticEval(child[1]), reverse=state.whose_move == self.playing)

    # def generateUtterance(self, state, reason):
    #     sarcastic_prompts = [
    #         "How's that for random?", "Flip!", "Spin!", "I hope this is my lucky day!",
    #         "How's this move for high noise to signal ratio?", "Uniformly distributed. That's me.",
    #         "Maybe I'll look into Bayes' Nets in the future.", "Eenie Meenie Miney Mo. I hope I'm getting K in a row.",
    #         "Your choice is probably more informed than mine.", "If I only had a brain."
    #     ]

    #     kind_prompts = [
    #         "I'd while away the hours, playing K in a Row.", "So much fun.", "Roll the dice!",
    #         "Yes, I am on a roll -- of my virtual dice.", "randint is my cousin.",
    #         "I like to spread my influence around on the board.", "Let's see if lady luck is on my side today!",
    #         "Watch me work some RNG magic.", "Feeling lucky? Here we go!", "Is it time for a game of chance?"
    #     ]

    #     funny_prompts = [
    #         "I think I'm developing a winning streak!", "Guessing games are my forte.", "Is this my moment of glory?",
    #         "Statistically, I should win soon, right?", "Here's a wild guess!", "Another move, another chance.",
    #         "The odds are ever in my favor.", "Let's see if lady luck is on my side today!"
    #     ]

    #     persona_prompts = {
    #         "sarcastic": sarcastic_prompts,
    #         "kind": kind_prompts,
    #         "funny": funny_prompts
    #     }

    #     example_prompts = persona_prompts.get(self.persona, ["I hope you have a great game!", "Let's enjoy this game together!"])

    #     prompt = (f"You are a {self.persona} player of K in a Row. Respond to your opponent in one sentence.\n"
    #               "Here is an example:\n" f"{example_prompts[0]}")
    #     prompt_tokens = tokenizer(prompt, return_tensors="pt")
    #     input_ids = prompt_tokens["input_ids"]
    #     attention_mask = prompt_tokens["attention_mask"]
    #     start_index = input_ids.shape[-1]
    #     output = model.generate(input_ids, attention_mask=attention_mask, num_return_sequences=1, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    #     generation_output = output[0][start_index:]
    #     generation_text = tokenizer.decode(generation_output, skip_special_tokens=True)
    #     self.MY_PAST_UTTERANCES.append(generation_text.strip())
    #     return generation_text.strip()

    def applyMove(self, state, move):
        new_state = State(old=state)
        i, j = move
        new_state.board[i][j] = state.whose_move
        new_state.whose_move = "X" if state.whose_move == "O" else "O"
        
        self.move_history.append((state.whose_move, move))
        self.game_state_history.append(new_state)
        return new_state

    def countPotential(self, state, player):
        board = state.board
        k = self.game_type.k
        potential_count = 0

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if cell == player or cell == " ":
                    for dx, dy in directions:
                        line = []
                        for step in range(k):
                            ni, nj = i + step * dx, j + step * dy
                            if 0 <= ni < len(board) and 0 <= nj < len(board[0]):
                                line.append(board[ni][nj])
                            else:
                                break
                        if len(line) == k and " " in line:
                            potential_count += 1
        return potential_count

    def generateSuccessors(self, state):
        successors = []
        board = state.board
        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if cell == " ":
                    nextState = self.applyMove(state, [i, j])
                    successors.append(([i, j], nextState))
        return successors

    def isTerminal(self, state):
        for row in state.board:
            if " " in row:
                return False
        return True

    def other(self, player):
        return 'X' if player == 'O' else 'O'

    def explainLastMove(self):
        explanation = (
            f"I evaluated {self.minimax_states_evaluated} states, performed static evaluations, "
            f"and made the best move I could find within the time limit of {self.last_move_time:.2f} seconds. "
            f"I performed {self.minimax_cutoffs} cutoffs during the search."
        )
        return explanation

    def storyOfTheGame(self):
        story = "Here's the story of our game so far:\n"
        story += "From the beginning, we have both made strategic moves.\n"
        story += "Let's look at some highlights:\n"

        for i, (state, remark) in enumerate(zip(self.game_state_history, self.OPPONENT_PAST_UTTERANCES)):
            story += f"Turn {i+1}:\n"
            story += f"  Game state: {state}\n"
            story += f"  Your response: {remark}\n"
        
        my_potential = self.countPotential(self.game_state_history[-1], self.playing)
        opponent_potential = self.countPotential(self.game_state_history[-1], self.WHO_MY_OPPONENT_PLAYS)
        prediction = "It's a close game, but I believe I have a slight edge." if my_potential > opponent_potential else "You're in a strong position!"

        story += f"\nBased on the current board state, {prediction}\n"
        return story


    def generateUtterance(self, state, reason):
        sarcastic_utterances = [
            "How's that for random?", "Flip!", "Spin!", "I hope this is my lucky day!",
            "How's this move for high noise to signal ratio?", "Uniformly distributed. That's me.",
            "Maybe I'll look into Bayes' Nets in the future.", "Eenie Meenie Miney Mo. I hope I'm getting K in a row.",
            "Your choice is probably more informed than mine.", "If I only had a brain."
        ]

        kind_utterances = [
            "I'd while away the hours, playing K in a Row.", "So much fun.", "Roll the dice!",
            "Yes, I am on a roll -- of my virtual dice.", "randint is my cousin.",
            "I like to spread my influence around on the board.", "Let's see if lady luck is on my side today!",
            "Watch me work some RNG magic.", "Feeling lucky? Here we go!", "Is it time for a game of chance?"
        ]

        funny_utterances = [
            "I think I'm developing a winning streak!", "Guessing games are my forte.", "Is this my moment of glory?",
            "Statistically, I should win soon, right?", "Here's a wild guess!", "Another move, another chance.",
            "The odds are ever in my favor.", "Let's see if lady luck is on my side today!"
        ]

        persona_utterances = {
            "sarcastic": sarcastic_utterances,
            "kind": kind_utterances,
            "funny": funny_utterances
        }

        example_utterances = persona_utterances.get(self.persona, ["I hope you have a great game!", "Let's enjoy this game together!"])

        return random.choice(example_utterances)