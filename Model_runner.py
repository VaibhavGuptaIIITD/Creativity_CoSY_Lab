import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from nltk.translate.bleu_score import sentence_bleu

# Initialize the TicTacToe game
tictactoe = TicTacToe()

# Define the MCTS arguments
args = {
    'C': 2,                              
    'num_searches': 1000,               
    'dirichlet_epsilon': 0.25,          
    'dirichlet_alpha': 0.3               
}

# Load the existing model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Opponent AI model
model = ResNet(tictactoe, 4, 64, device)
checkpoint_path = "/content/5by5_3.pt"  # Update with your model path
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

#AI that will monitor human's moves
monitoring_ai = ResNet(tictactoe, 4, 64, device)
monitoring_checkpoint_path = "/content/5by5_3.pt"  # Path to the monitoring AI model
monitoring_ai.load_state_dict(torch.load(monitoring_checkpoint_path, map_location=device))
monitoring_ai.eval()


# Initialize MCTS with the loaded model
mcts = MCTS(tictactoe, args, model)
monitoring_mcts = MCTS(tictactoe, args, monitoring_ai)

# Metrics initialization
num_games = 1  # Reduced for interactive play
ai_wins = 0
human_wins = 0
draws = 0
total_moves = 0
game_lengths = []
move_frequency_ai = np.zeros(25)
move_frequency_human = np.zeros(25)

# Elo rating tracking
ai_rating = 1500
human_rating = 1500

# For ROC curve and BLEU score
true_labels = []  # Store whether a move results in a win (1) or not (0)
predicted_scores = []  # Store AI's prediction score for the move being a win (probability)
human_moves = []  # Store human moves to calculate BLEU
ai_moves = []  # Store AI moves to calculate BLEU

def update_elo(winner_rating, loser_rating, k=32):
    expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
    expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))
    
    winner_new_rating = winner_rating + k * (1 - expected_winner)
    loser_new_rating = loser_rating + k * (0 - expected_loser)
    
    return winner_new_rating, loser_new_rating

# Human Player Initialization
class HumanPlayer:
    def get_move(self, state, valid_moves):
        while True:
            try:
                # Find indices of valid moves
                available_moves = np.where(valid_moves == 1)[0]
                print(f"Available moves: {available_moves}")
                
                move = int(input("Enter your move: "))
                if valid_moves[move] == 1:
                    return move
                else:
                    print(f"Move {move} is invalid. Please choose from available moves.")
            except (ValueError, IndexError):
                print("Please enter a valid move.")

human_player = HumanPlayer()

print("Welcome to 5x5 Tic Tac Toe against AI!")
print("The board is numbered 0-24, row by row from top left.")

# Game Loop
for game in range(num_games):
    # print(f"\n--- Game {game + 1} ---")
    # state = tictactoe.get_initial_state()
    # player = -1  # Human starts first
    # moves_in_game = 0
    # game_winner = 0
    # human_move_analysis = []
    print(f"\n--- Game {game + 1} ---")
    state = tictactoe.get_initial_state()
    player = -1  # Human starts first
    moves_in_game = 0
    game_winner = 0
    human_move_analysis = []

    while True:
        # Human's turn
        print(state)
        if player == -1:
            print("\nYour turn:")
            valid_moves = tictactoe.get_valid_moves(state)
            action = human_player.get_move(state, valid_moves)
            move_frequency_human[action] += 1
            human_moves.append(action)  # Track human moves

            # Monitoring AI's evaluation of human move
            neutral_state = tictactoe.change_perspective(state, 1)
            monitoring_probs = monitoring_mcts.search(neutral_state)
            optimal_move = np.argmax(monitoring_probs)

            # Compare human's move with the optimal move
            human_move_analysis.append({
                "turn": moves_in_game + 1,
                 "human_move": action,
                 "optimal_move": optimal_move,
                 "probability_of_human_move": mcts_probs[action],
                 "probability_of_optimal_move": mcts_probs[optimal_move]
                })

        # AI's turn
        else:
            # print("\nAI's turn:")
            # neutral_state = tictactoe.change_perspective(state, 1)
            # mcts_probs = mcts.search(neutral_state)
            # action = np.argmax(mcts_probs)
            # move_frequency_ai[action] += 1
            # ai_moves.append(action)  # Track AI moves
            # print(f"AI chooses move: {action}")
            print("\nAI's turn:")
            neutral_state = tictactoe.change_perspective(state, 1)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)
            move_frequency_ai[action] += 1
            ai_moves.append(action)  # Track AI moves
            print(f"AI chooses move: {action}")
            plt.figure(figsize=(6, 4))
            valid_move_indices = np.where(tictactoe.get_valid_moves(state) == 1)[0]
            valid_probs = mcts_probs[valid_move_indices]
            
            plt.bar(valid_move_indices, valid_probs, color='blue', alpha=0.7)
            plt.title("Move Probability Distribution after AI Move")
            plt.xlabel("Board Position")
            plt.ylabel("Probability")
            plt.xticks(np.arange(25))
            plt.show(block=False)  # Display non-blocking plot
            time.sleep(2)  # Pause to allow visualization before proceeding
            plt.close()  # Close the plot to free resources
        
        # Update game state
        state = tictactoe.get_next_state(state, action, player)
        moves_in_game += 1
        
        # Check game termination
        value, is_terminal = tictactoe.get_value_and_terminated(state, action)
        
        if is_terminal:
            if value == 1 and player == -1:
                true_labels.append(1)  # Human wins
                predicted_scores.append(mcts_probs[action])  # AI's predicted probability of winning
                print("Human wins!")
                human_wins += 1
                game_winner = -1
                human_rating, ai_rating = update_elo(human_rating, ai_rating)
            elif value == 1 and player == 1:
                true_labels.append(1)  # AI wins
                predicted_scores.append(mcts_probs[action])  # AI's predicted probability of winning
                print("AI wins!")
                ai_wins += 1
                game_winner = 1
                ai_rating, human_rating = update_elo(ai_rating, human_rating)
            else:
                true_labels.append(0)  # Draw
                predicted_scores.append(mcts_probs[action])  # AI's predicted probability of winning
                print("It's a draw!")
                draws += 1
                game_winner = 0
            
            break
        
        # Switch players
        player = -player

    # # Print the human move analysis at the end of the game
    # print("\n--- Human Move Analysis ---")
    # for analysis in human_move_analysis:
    #     print(f"Turn {analysis['turn']}:")
    #     print(f"  Human Move: {analysis['human_move']} (Probability: {analysis['probability_of_human_move']:.2f})")
    #     print(f"  Optimal Move: {analysis['optimal_move']} (Probability: {analysis['probability_of_optimal_move']:.2f})")
    #     if analysis['human_move'] != analysis['optimal_move']:
    #         print("  The human made a suboptimal move.")
    #     else:
    #         print("  The human made the optimal move.")
    print("\n--- Human Move Analysis by Monitoring AI ---")
    for analysis in human_move_analysis:
        print(f"Turn {analysis['turn']}:")
        print(f"  Human Move: {analysis['human_move']} (Probability: {analysis['probability_of_human_move']:.2f})")
        print(f"  Monitoring AI Optimal Move: {analysis['optimal_move']} (Probability: {analysis['probability_of_optimal_move']:.2f})")
        if analysis['human_move'] != analysis['optimal_move']:
            print("  The human made a suboptimal move.")
        else:
            print("  The human made the optimal move.")

    # Track game metrics
    game_lengths.append(moves_in_game)

# Metrics Calculations
print("\n--- Game Statistics ---")
print(f"AI Wins: {ai_wins}")
print(f"Human Wins: {human_wins}")
print(f"Draws: {draws}")
print(f"Average Game Length: {np.mean(game_lengths):.2f} moves")
print(f"Final AI Elo Rating: {ai_rating:.2f}")
print(f"Final Human Elo Rating: {human_rating:.2f}")

# Visualization Section (Optional, can be commented out if needed)
# Heatmap for Move Preferences
plt.figure(figsize=(10, 6))
sns.heatmap(move_frequency_ai.reshape((5, 5)), annot=True, cmap='Blues', cbar=True)
plt.title("AI Move Frequency Heatmap")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(move_frequency_human.reshape((5, 5)), annot=True, cmap='Reds', cbar=True)
plt.title("Human Move Frequency Heatmap")
plt.show()

# Game Length Distribution
plt.figure(figsize=(10, 6))
plt.hist(game_lengths, bins=20, color='green', edgecolor='black')
plt.title("Game Length Distribution")
plt.xlabel("Number of Moves")
plt.ylabel("Frequency")
plt.show()


