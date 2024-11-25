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

import time # You'll probably need this to avoid losing a
 # game due to exceeding a time limit.
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

import os
import random;
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_id = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

pipe = pipeline(
    "text-generation", 
    model=model_id,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id  # Set the pad_token_id to eos_token_id
)


# Create your own type of agent by subclassing KAgent:

class OurAgent(KAgent):  # Keep the class name "OurAgent" so a game master
    # knows how to instantiate your agent class.

    def __init__(self, twin=False):
        self.twin=twin
        self.nickname = 'Nic'
        if twin: self.nickname += '2'
        self.long_name = 'Templatus Skeletus'
        if twin: self.long_name += ' II'
        self.persona = 'strategic'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "don't know yet" # e.g., "X" or "O".

    def introduce(self):
        intro = '\nMy name is Templatus Skeletus.\n'+\
            '"An instructor" made me.\n'+\
            'Somebody please turn me into a real game-playing agent!\n'
        if self.twin: intro += "By the way, I'm the TWIN.\n"
        return intro

    # Receive and acknowledge information about the game from
    # the game master:
    def prepare(
        self,
        game_type,
        what_side_to_play,
        opponent_nickname,
        expected_time_per_move = 0.1, # Time limits can be
                                      # changed mid-game by the game master.
        utterances_matter=True):      # If False, just return 'OK' for each utterance.

       # Write code to save the relevant information in variables
       # local to this instance of the agent.
       # Game-type info can be in global variables.
       self.game_type = game_type
       self.playing = what_side_to_play
       self.opponent_nickname = opponent_nickname
       self.time_limit = expected_time_per_move
       self.utterances_matter = utterances_matter
       return "OK"
       #print("Change this to return 'OK' when ready to test the method.")
       #return "Not-OK"
   
    # The core of your agent's ability should be implemented here:             
    def makeMove(self, currentState, currentRemark, timeLimit=10000):
        # Existing decision-making code...
        depth = 3
        bestScore, bestMove, _ = self.minimax(state=currentState, depthRemaining=depth, pruning=True, alpha=float("-inf"), beta=float("inf"))

        # Case where no valid move is found 
        if bestMove is None: 
            return [[[0,0], currentState], self.generateUtterance(currentState, "No valid moves!")]

        newState = self.applyMove(currentState, bestMove)
        newRemark = self.generateUtterance(currentState, "Choosing the best move.")
        return [[bestMove, newState], newRemark]

        # print("code to compute a good move should go here.")
        # # Here's a placeholder:
        # a_default_move = [0, 0] # This might be legal ONCE in a game,
        # # if the square is not forbidden or already occupied.
    
        # newState = currentState # This is not allowed, and even if
        # # it were allowed, the newState should be a deep COPY of the old.
    
        # newRemark = "I need to think of something appropriate.\n" +\
        # "Well, I guess I can say that this move is probably illegal."

        # print("Returning from makeMove")
        # return [[a_default_move, newState], newRemark]
    

    # The main adversarial search function:
    def minimax(self,
            state,
            depthRemaining,
            pruning=False,
            alpha=None,
            beta=None,
            zHashing=None):
        #print("Calling minimax. We need to implement its body.")
        if depthRemaining == 0 or self.isTerminal(state): 
            return [self.staticEval(state), None, None]
        bestMove = None
        if state.whose_move == self.playing: # maximizing player
            maxEval = float("-inf")
            for move, nextState in self.generateSuccessors(state):
                eval, _, _ = self.minimax(nextState, depthRemaining - 1, pruning, alpha, beta)
                if eval > maxEval: 
                    maxEval = eval
                    bestMove = move
                if pruning: 
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return [maxEval, bestMove, None]
        else: # minimizing player 
            minEval = float("inf")
            for move, nextState in self.generateSuccessors(state): 
                eval, _, _ = self.minimax(nextState, depthRemaining - 1, pruning, alpha, beta)
                if eval < minEval: 
                    minEval = eval 
                    bestMove = move
                if pruning: 
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return [minEval, bestMove, None]

        #default_score = 0 # Value of the passed-in state. Needs to be computed.
    
        #return [default_score, "my own optional stuff", "more of my stuff"]
        # Only the score is required here but other stuff can be returned
        # in the list, after the score, in case you want to pass info
        # back from recursive calls that might be used in your utterances,
        # etc. 
 
    def staticEval(self, state):
        #print('calling staticEval. Its value needs to be computed!')
        # Values should be higher when the states are better for X,
        # lower when better for O.
        #board = state.board
        my_score = self.countPotential(state, self.playing)
        opp_score = self.countPotential(state, "X" if self.playing == "O" else "O")
        return my_score - opp_score
        #return 0


    # HELPER FUNCTIONS: 

    # Generate an intelligent, persona-specific utterance using the LLM
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
        
        if self.persona == "sarcastic":
            example_utterances = sarcastic_utterances
        elif self.persona == "kind":
            example_utterances = kind_utterances
        elif self.persona == "funny":
            example_utterances = funny_utterances
        else:
            example_utterances = ["I hope you have a great game!", "Let's enjoy this game together!"]
        
        random_comment = random.choice(example_utterances)

        prompt = f"You are playing an opponent in Tic Tac Toe. Respond with a kind sentence to your opponent. This is an example {random_comment}"
        prompt_tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = prompt_tokens["input_ids"]
        attention_mask = prompt_tokens["attention_mask"]
        start_index = input_ids.shape[-1]
        output = model.generate(input_ids, attention_mask=attention_mask, num_return_sequences=1, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        generation_output = output[0][start_index:]
        generation_text = tokenizer.decode(generation_output, skip_special_tokens=True)
        
        return generation_text.strip()


    # def generateUtterance(self, state, reason):
    #     # Generate persona-specific utterance
    #     if self.persona == 'bland':
    #         return f"{self.long_name}: {reason}. {self.gameStateComment(state)}"
    #     elif self.persona == 'cheerful':
    #         return f"{self.long_name} says: Yay! {reason}! {self.gameStateComment(state)}"
    #     elif self.persona == 'serious':
    #         return f"{self.long_name}: {reason}. Here are my thoughts on the current state: {self.gameStateComment(state)}"
    #     # Add more persona types as needed
    #     else:
    #         return f"{self.long_name}: {reason}. {self.gameStateComment(state)}"
        
    # def gameStateComment(self, state):
    #     # Generate game-state-specific comment
    #     # This is a very basic example and should be improved
    #     if self.isTerminal(state):
    #         return "The game is almost over!"
    #     elif self.countPotential(state, self.playing) > self.countPotential(state, self.other(self.playing)):
    #         return "I have the upper hand right now."
    #     else:
    #         return "I need to make a strong move to gain an advantage."

    def applyMove(self, state, move):
        new_state = State(old=state) # making a deep copy
        i, j = move
        new_state.board[i][j] = state.whose_move
        new_state.whose_move = "X" if state.whose_move == "O" else "O"
        return new_state
    
    # to count the potential lines of length 'k' that the player can form
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

        # NEED TO FINISH IMPLEMENTING

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
        # check if there's a win or the board is full 
        for row in state.board: 
            if " " in row: 
                return False
        return True
    
    def other(self, player):
        return 'X' if player == 'O' else 'O'
 
# OPTIONAL THINGS TO KEEP TRACK OF:

#  WHO_MY_OPPONENT_PLAYS = other(WHO_I_PLAY)
#  MY_PAST_UTTERANCES = []
#  OPPONENT_PAST_UTTERANCES = []
#  UTTERANCE_COUNT = 0
#  REPEAT_COUNT = 0 or a table of these if you are reusing different utterances

