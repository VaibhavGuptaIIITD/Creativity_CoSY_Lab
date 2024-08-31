import numpy as np  
np.__version__ 
import math 

class TicTacToe:
    def __init__(self):
        self.row_count = 3  # This declares the number of rows in tic tac toe i.e. 3
        self.column_count = 3  # This declares the number of columns in tic tac toe i.e. 3
        self.action_size = self.row_count * self.column_count  # total possible actions id equal to r*c(row*column)
        
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count)) # Here we are initialising an empty tic tac toe board with all positions initially set to 0
    
    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        #In the above two lines we are calculating the row and column indices corresponding to the action.
        state[row, column] = player # This simulates the move being made by the player.
        return state
    
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8) # This function is used to return an array which contains the positions which are valid for the current state of the board.
                                                        # Basically it shows the empty positions on the board.
    
    def check_win(self, state, action):
        if action == None:
            return False
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column] # this line of code determines that which player has made the action.
        
        #The code below does all the checks to check if any of the player has won by making that move.
        return (
            np.sum(state[row, :]) == player * self.column_count  # This one checks row
            or np.sum(state[:, column]) == player * self.row_count  # This one checks diagonal
            or np.sum(np.diag(state)) == player * self.row_count  # This one checks main diagonal
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count # This one checks anti-diagonal
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True  #This returns 1 if the game has been won by some player and returns a win for that player.
        
        #The line of code below checks if there are any more valid moves left , or if the board is full.
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True  #This returns draw
        
        return 0, False  #This is the condition which tells that the game is not over and they should continue playing.
    
    def get_opponent(self, player):
        return -player # This returns the opponent player, -1 for first player and 1 for second.
    
    def get_opponent_value(self, value):
        return -value # This returns the value of the opponents move and just reverses it to make the move valid for the next move of the player.
    
    def change_perspective(self, state, player):
        # Change the perspective of the board by multiplying it by the player (invert the board for the opponent)
        return state * player

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None):
        self.game = game  
        self.args = args  # Store the arguments for MCTS
        self.state = state  
        self.parent = parent  # References to the parent node in the MCTS tree.
        self.action_taken = action_taken  # Stores the action that has been taken to reach this node.
        
        self.children = []  # It stores the child nodes ,basically it stores the future states.
        self.expandable_moves = game.get_valid_moves(state)  # Gives valid moves for expanding the given node.
        
        self.visit_count = 0  # This initialises the visit count of this node
        self.value_sum = 0  # This initialises the value sum of this node
        
    def is_fully_expanded(self):
        #This will check if the node has been expanded fully and there are no more valid moves and it has at least one child.
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0
    
    def select(self):
        best_child = None  # This variable has been created to store the best child of the node.
        best_ucb = -np.inf  #Initialising the Upper Confidence Bound to negative infinity.
        
        for child in self.children:
            ucb = self.get_ucb(child)  # Calculates the UCB value for the child node.
            if ucb > best_ucb:
                best_child = child  # The best child node is updated.
                best_ucb = ucb  # The UCB value is also updated.
                
        return best_child  # Return the child node with the highest UCB value
    
    def get_ucb(self, child):
        # Calculates the Q-value of the child node
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        # Calculates the UCB value using the Q value and the exploration term 
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)
    
    def expand(self):
        # Random valid action to expand this node 
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0  # Marking the selected action as no more expandable.
        
        # Create the new child state by applying the action
        child_state = self.state.copy()
        child_state = self.game.get_next_state(child_state, action, 1)
        child_state = self.game.change_perspective(child_state, player=-1)  # Change perspective for the opponent
        
        # Create a new child node and add it to the list of children
        child = Node(self.game, self.args, child_state, self, action)
        self.children.append(child)
        return child
    
    def simulate(self):
        # Perform a simulation to the end of the game from the current state
        value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
        value = self.game.get_opponent_value(value)  # Get the value from the opponent's perspective
        
        if is_terminal:
            return value  # Return the value if the game has ended
        
        # Simulate the game by randomly playing until it reaches a terminal state
        rollout_state = self.state.copy()
        rollout_player = 1
        while True:
            valid_moves = self.game.get_valid_moves(rollout_state)  # Get valid moves
            action = np.random.choice(np.where(valid_moves == 1)[0])  # Choose a random valid action
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)  # Apply the action
            value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)  # Check for terminal state
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.get_opponent_value(value)  # Invert value if it's the opponent's turn
                return value  # Return the value if the game has ended
            
            rollout_player = self.game.get_opponent(rollout_player)  # Switch players
            
    def backpropagate(self, value):
        # Update the current node's value and visit count based on the simulation result
        self.value_sum += value
        self.visit_count += 1
        
        # Propagate the value back to the parent node
        value = self.game.get_opponent_value(value)  # Invert the value for the parent's perspective
        if self.parent is not None:
            self.parent.backpropagate(value)  # Recursively backpropagate the value up the tree

# MCTS (Monte Carlo Tree Search) class handles the search process.
class MCTS:
    def __init__(self, game, args):
        self.game = game  # Reference to the TicTacToe game.
        self.args = args  # Arguments for controlling MCTS behavior.
        
    def search(self, state):
        # Initialize the root node of the MCTS tree with the current state.
        root = Node(self.game, self.args, state)
        
        # Perform a fixed number of MCTS simulations.
        for search in range(self.args['num_searches']):
            node = root  # Start from the root node for each simulation.
            
            # Traverse the tree until a node is found that is not fully expanded.
            while node.is_fully_expanded():
                node = node.select()  # Select the best child node based on UCB.
                
            # Check if the node is terminal and get its value.
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            # If the node is not terminal, expand it and simulate a random playout.
            if not is_terminal:
                node = node.expand()
                value = node.simulate()
                
            # Backpropagate the value up the tree to update the statistics.
            node.backpropagate(value)    
            
        # Calculate the action probabilities from the root node's children.
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        # Normalize the action probabilities to sum to 1.
        action_probs /= np.sum(action_probs)
        return action_probs  # Return the calculated action probabilities.
        
        
# Initialize the TicTacToe game.
tictactoe = TicTacToe()
player = 1  # Start with player 1.

# Define the MCTS parameters.
args = {
    'C': 1.41,  # Exploration parameter for UCB.
    'num_searches': 1000  # Number of MCTS simulations to perform.
}

# Initialize the MCTS with the game and parameters.
mcts = MCTS(tictactoe, args)

# Get the initial state of the board.
state = tictactoe.get_initial_state()


# Main loop to alternate between human and AI player moves.
while True:
    print(state)  # Print the current state of the board.
    
    if player == 1:
        # If it's the human player's turn, get valid moves.
        valid_moves = tictactoe.get_valid_moves(state)
        print("valid_moves", [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
        action = int(input(f"{player}:"))  # Get the action from the human player.

        # Check if the selected action is valid.
        if valid_moves[action] == 0:
            print("action not valid")
            continue  # If the action is invalid, prompt again.
            
    else:
        # If it's the AI player's turn, change the state perspective.
        neutral_state = tictactoe.change_perspective(state, player)
        # Perform MCTS search to get the best move.
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)  # Select the action with the highest probability.
        
    # Update the state with the selected action.
    state = tictactoe.get_next_state(state, action, player)
    
    # Check if the game has ended and get the value.
    value, is_terminal = tictactoe.get_value_and_terminated(state, action)
    
    # If the game is over, print the final state and the result.
    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")  # Print the winner if someone won.
        else:
            print("draw")  # Print draw if the game ended in a draw.
        break  # Exit the loop since the game is over.
        
    # Switch to the opponent player for the next turn.
    player = tictactoe.get_opponent(player)
