import numpy as np
print(np.__version__)  # Print the version of NumPy library being used

import torch
print(torch.__version__)  # Print the version of PyTorch library being used

import torch.nn as nn  # Import PyTorch's neural network module
import torch.nn.functional as F  # Import PyTorch's functional module for activation functions

torch.manual_seed(0)  # Set a manual seed for reproducibility in PyTorch

from tqdm.notebook import trange  # Import a progress bar for iterations

import random  # Import random for shuffling data
import math  # Import math module for mathematical operations

# Define TicTacToe class to handle the game logic
class TicTacToe:
    def __init__(self):
        self.row_count = 3  # Number of rows in the game board
        self.column_count = 3  # Number of columns in the game board
        self.action_size = self.row_count * self.column_count  # Total number of possible actions (cells)

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))  # Return an empty 3x3 game board (all zeros)

    def get_next_state(self, state, action, player):
        row = action // self.column_count  # Calculate the row based on the action
        column = action % self.column_count  # Calculate the column based on the action
        state[row, column] = player  # Set the player's move on the board
        return state  # Return the updated game state

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)  # Return a 1D array of valid moves (where the board is empty)

    # Check if the current player has won the game
    def check_win(self, state, action):
        if action == None:
            return False

        row = action // self.column_count  # Get the row of the move
        column = action % self.column_count  # Get the column of the move
        player = state[row, column]  # Get the current player at that position

        # Check if the player has a winning row, column, or diagonal
        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )

    # Get the game outcome (win, draw, ongoing) and if the game is terminated
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):  # Check if the last action was a win
            return 1, True  # Return 1 (win) and True (game over)
        if np.sum(self.get_valid_moves(state)) == 0:  # If there are no more valid moves (draw)
            return 0, True  # Return 0 (draw) and True (game over)
        return 0, False  # If game is still ongoing, return 0 and False

    # Get the opponent of the current player (player = 1, opponent = -1)
    def get_opponent(self, player):
        return -player

    # Get the opposite value (for backpropagating outcomes)
    def get_opponent_value(self, value):
        return -value

    # Change the perspective of the game state to reflect the current player
    def change_perspective(self, state, player):
        return state * player  # Flip the state for the opponent (-1 to 1)

    # Encode the state into 3 layers: one for -1 (opponent), one for 0 (empty), and one for 1 (current player)
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        return encoded_state

# Residual Network (ResNet) model for policy and value head
class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()

        self.device = device  # Device to run the model on (GPU or CPU)
        
        # First convolutional block
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),  # 3 input channels (game state), num_hidden filters
            nn.BatchNorm2d(num_hidden),  # Batch normalization
            nn.ReLU()  # Activation function
        )
        
        # Backbone consisting of several residual blocks
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]  # A list of residual blocks
        )
        
        # Policy head for choosing the next action
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),  # Conv layer with 32 output channels
            nn.BatchNorm2d(32),  # Batch normalization
            nn.ReLU(),  # Activation
            nn.Flatten(),  # Flatten the output
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)  # Fully connected layer for actions
        )
        
        # Value head for predicting the game outcome
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),  # Conv layer with 3 output channels
            nn.BatchNorm2d(3),  # Batch normalization
            nn.ReLU(),  # Activation
            nn.Flatten(),  # Flatten the output
            nn.Linear(3 * game.row_count * game.column_count, 1),  # Fully connected layer for value prediction
            nn.Tanh()  # Output value in range [-1, 1]
        )
        
        self.to(device)  # Move the model to the specified device (GPU or CPU)
        
    def forward(self, x):
        x = self.startBlock(x)  # Pass input through the first convolution block
        for resBlock in self.backBone:  # Pass through each residual block
            x = resBlock(x)
        policy = self.policyHead(x)  # Pass through the policy head
        value = self.valueHead(x)  # Pass through the value head
        return policy, value  # Return both policy and value predictions

# Residual Block class used in the ResNet
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        # First convolution + batch norm
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        # Second convolution + batch norm
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x  # Save the input (residual connection)
        x = F.relu(self.bn1(self.conv1(x)))  # First convolution and activation
        x = self.bn2(self.conv2(x))  # Second convolution
        x += residual  # Add the residual connection
        x = F.relu(x)  # Apply ReLU after the residual connection
        return x

# Node class for Monte Carlo Tree Search (MCTS)
class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game  # Game instance
        self.args = args  # Arguments for MCTS (such as exploration constant)
        self.state = state  # The game state
        self.parent = parent  # Parent node in the tree
        self.action_taken = action_taken  # Action taken to reach this node
        self.prior = prior  # Prior probability of selecting this node (from neural network)
        self.children = []  # List of child nodes
        self.visit_count = visit_count  # Number of visits to this node
        self.value_sum = 0  # Sum of values backpropagated to this node
        
    # Check if this node has been fully expanded (children nodes created)
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    # Select the best child node based on UCB score
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)  # Get the UCB score for each child
            if ucb > best_ucb:
                best_child = child  # Select the child with the highest UCB score
                best_ucb = ucb
                
        return best_child  # Return the best child node
    
    # Calculate the UCB (Upper Confidence Bound) score for a child node
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0  # If the node hasn't been visited, set Q-value to 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2  # Calculate Q-value based on value sum
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior  # UCB formula
    
    # Expand the node by creating child nodes based on the policy
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:  # Only expand valid moves (non-zero probability)
                child_state = self.state.copy()  # Copy the current state
                child_state = self.game.get_next_state(child_state, action, 1)  # Simulate the action
                child_state = self.game.change_perspective(child_state, player=-1)  # Switch perspective to opponent

                child = Node(self.game, self.args, child_state, self, action, prob)  # Create the child node
                self.children.append(child)  # Add child to the list of children
                
        return child  # Return the last expanded child node
            
    # Backpropagate the value up the tree
    def backpropagate(self, value):
        self.value_sum += value  # Add value to this node's value sum
        self.visit_count += 1  # Increment visit count
        
        value = self.game.get_opponent_value(value)  # Change the value to the opponent's perspective
        if self.parent is not None:  # If there's a parent node, recursively backpropagate
            self.parent.backpropagate(value)  


# Monte Carlo Tree Search (MCTS) class
class MCTS:
    def __init__(self, game, args, model):
        self.game = game  # Game instance
        self.args = args  # Arguments for MCTS
        self.model = model  # Neural network model
        
    @torch.no_grad()  # Disable gradient calculation during search
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)  # Create root node with visit count of 1
        
        # Get the policy and value from the neural network for the root node
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()  # Apply softmax to get probabilities
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)  # Add exploration noise
        
        valid_moves = self.game.get_valid_moves(state)  # Get valid moves for the current state
        policy *= valid_moves  # Mask invalid moves
        policy /= np.sum(policy)  # Normalize the policy
        root.expand(policy)  # Expand the root node using the policy
        
        # Perform multiple searches
        for search in range(self.args['num_searches']):
            node = root  # Start from the root
            
            while node.is_fully_expanded():  # Traverse the tree until a non-expanded node is found
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)  # Check game outcome
            value = self.game.get_opponent_value(value)  # Get value from the opponent's perspective
            
            if not is_terminal:  # If the game isn't over, get the policy and value from the model
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()  # Apply softmax to get probabilities
                valid_moves = self.game.get_valid_moves(node.state)  # Get valid moves for the state
                policy *= valid_moves  # Mask invalid moves
                policy /= np.sum(policy)  # Normalize the policy
                
                value = value.item()  # Convert the tensor to a Python scalar
                
                node.expand(policy)  # Expand the node using the policy
                
            node.backpropagate(value)  # Backpropagate the value up the tree
            
        # Calculate the visit counts for each action at the root
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count  # Assign visit count to the corresponding action
        action_probs /= np.sum(action_probs)  # Normalize the visit counts to get probabilities
        return action_probs  # Return the action probabilities


# AlphaZero class combining the model, optimizer, and MCTS for training
class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model  # Neural network model
        self.optimizer = optimizer  # Optimizer for training
        self.game = game  # Game instance
        self.args = args  # Arguments for AlphaZero
        self.mcts = MCTS(game, args, model)  # MCTS instance
        
    # Self-play function for generating training data
    def selfPlay(self):
        memory = []  # Initialize memory for storing game data
        player = 1  # Start with player 1
        state = self.game.get_initial_state()  # Get the initial empty game state
        
        while True:
            neutral_state = self.game.change_perspective(state, player)  # Get the game state from current player's perspective
            action_probs = self.mcts.search(neutral_state)  # Use MCTS to get action probabilities
            
            memory.append((neutral_state, action_probs, player))  # Store the state, action probabilities, and player
            
            temperature_action_probs = action_probs ** (1 / self.args['temperature'])  # Adjust action probabilities based on temperature
            action = np.random.choice(self.game.action_size, p=action_probs)  # Sample an action based on the probabilities
            
            state = self.game.get_next_state(state, action, player)  # Apply the action and get the next state
            
            value, is_terminal = self.game.get_value_and_terminated(state, action)  # Check if the game is over
            
            if is_terminal:  # If the game is over, process and return the memory
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)  # Determine the game outcome
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),  # Encode the game state
                        hist_action_probs,  # Store the action probabilities
                        hist_outcome  # Store the game outcome
                    ))
                return returnMemory  # Return the collected data
            
            player = self.game.get_opponent(player)  # Switch to the opponent's turn
                
    # Train the model on a batch of self-play data
    def train(self, memory):
        random.shuffle(memory)  # Shuffle the memory to randomize training data
        for batchIdx in range(0, len(memory), self.args['batch_size']):  # Iterate over mini-batches
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]  # Sample a mini-batch
            state, policy_targets, value_targets = zip(*sample)  # Extract states, policies, and values
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)  # Convert to NumPy arrays
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)  # Convert states to PyTorch tensors
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)  # Convert policies to PyTorch tensors
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)  # Convert values to PyTorch tensors
            
            out_policy, out_value = self.model(state)  # Get model predictions (policy and value)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)  # Compute cross-entropy loss for policy head
            value_loss = F.mse_loss(out_value, value_targets)  # Compute mean squared error loss for value head
            loss = policy_loss + value_loss  # Total loss
            
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the model parameters
    
    # Main learning loop
    def learn(self):
        for iteration in range(self.args['num_iterations']):  # Iterate for a specified number of training iterations
            memory = []  # Initialize memory for each iteration
            
            self.model.eval()  # Set the model to evaluation mode
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):  # Perform self-play multiple times
                memory += self.selfPlay()  # Accumulate self-play data
                
            self.model.train()  # Set the model to training mode
            for epoch in trange(self.args['num_epochs']):  # Train for multiple epochs on the collected data
                self.train(memory)  # Train the model on the memory
            
            # Save the model and optimizer states after each iteration
            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")

# Instantiate the TicTacToe game
tictactoe = TicTacToe()

# Set the device for training (use GPU if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the ResNet model with 4 residual blocks and 64 hidden units
model = ResNet(tictactoe, 4, 64, device)

# Set up the optimizer (Adam) with learning rate and weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Set the training arguments
args = {
    'C': 2,  # Exploration parameter for MCTS
    'num_searches': 60,  # Number of searches per MCTS
    'num_iterations': 3,  # Number of AlphaZero iterations
    'num_selfPlay_iterations': 500,  # Number of self-play games per iteration
    'num_epochs': 4,  # Number of training epochs per iteration
    'batch_size': 64,  # Batch size for training
    'temperature': 1.25,  # Temperature parameter for action selection
    'dirichlet_epsilon': 0.25,  # Weight for Dirichlet noise
    'dirichlet_alpha': 0.3  # Dirichlet alpha parameter for noise
}

# Instantiate AlphaZero with the model, optimizer, game, and arguments
alphaZero = AlphaZero(model, optimizer, tictactoe, args)

# Start the AlphaZero learning process
alphaZero.learn()  # Perform training using self-play and MCTS
