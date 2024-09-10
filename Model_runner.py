import torch
import numpy as np

# Assuming TicTacToe and ResNet classes and other necessary classes are already defined

# Initialize the TicTacToe game
tictactoe = TicTacToe()

# Player 1 will be the human, and Player -1 will be the AI
player = 1

# Define the MCTS arguments
args = {
    'C': 2,                              # Exploration constant for MCTS
    'num_searches': 1000,                # Number of MCTS simulations
    'dirichlet_epsilon': 0.25,           # Dirichlet noise factor in MCTS
    'dirichlet_alpha': 0.3               # Alpha value for Dirichlet distribution
}

# Load the existing model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU
model = ResNet(tictactoe, 4, 64, device)  # Initialize model with same structure as the saved one

# Load the model's pre-trained weights
checkpoint_path = "/content/Tweaked.pt"  # Change this to the actual path of the saved model
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

model.eval()  # Set the model to evaluation mode (important for inference)

# Initialize MCTS with the loaded model
mcts = MCTS(tictactoe, args, model)

# Get the initial game state
state = tictactoe.get_initial_state()

# Game loop
while True:
    print(state)  # Display the current game state
    
    if player == 1:  # Human player's turn
        valid_moves = tictactoe.get_valid_moves(state)  # Get valid moves
        print("Valid moves:", [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
        action = int(input(f"Player {player}, enter your move (0-{tictactoe.action_size - 1}): "))

        if valid_moves[action] == 0:  # Check if the move is valid
            print("Action not valid. Try again.")
            continue
            
    else:  # AI's turn
        neutral_state = tictactoe.change_perspective(state, player)  # Change perspective for the AI
        mcts_probs = mcts.search(neutral_state)  # Get MCTS search probabilities
        action = np.argmax(mcts_probs)  # AI selects the action with the highest probability
        
    # Apply the action and get the next game state
    state = tictactoe.get_next_state(state, action, player)
    
    # Check if the game is over
    value, is_terminal = tictactoe.get_value_and_terminated(state, action)
    
    if is_terminal:  # If the game is over, display the final state and result
        print(state)
        if value == 1:
            print(f"Player {player} won!")
        else:
            print("It's a draw!")
        break  # Exit the loop and end the game
        
    # Switch to the next player
    player = tictactoe.get_opponent(player)
