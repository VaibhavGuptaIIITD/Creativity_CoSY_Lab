Real-world problems cannot just function with a tree-based planning method, as the dynamics are complex and unknown. Until now, all the models know the environment’s dynamics or have a perfect simulator. Model-free RL estimates the optimal policy and/or value function directly from interactions with the environment but isn’t accurate at all.
 
**Model-based Reinforcement Learning** learns a model of the environment’s dynamics, that is it first constructs the state representation that the model should predict and then plans with respect to the learned model. Represented by a Markov’s Decison Process / MDP, it has next state prediction and expected reward prediction, MCTS is applied to compute the optimal value or optimal policy. All of this is an issue since the agent is not able to optimize its representation or model for the purpose of effective planning (modeling errors may compound during planning). To improve this,  predicting the value function is useful as they construct an abstract MDP model that is equivalent to planning in the real environment (using value equivalence, starting from the same real state, the cumulative reward of a trajectory through the abstract MDP matches the cumulative reward of a trajectory in the real environment). MDP model is viewed as a hidden layer of the network, and unrolled MDP is trained such that the expected cumulative sum of rewards matches the expected value with respect to the real environment (as there is no requirement for its transition model to match real states in the environment).

// image consisting of connected arrows of muzero //

**MuZero Algorithm** combines tree-based search with a learned model and achieves results without any knowledge of the game's dynamics. MuZero builds upon AlphaZero’s search and search-based policy iteration algorithms and incorporates a learned model into the training procedure. It extends AlphaZero to broader environments, including single-agent domains and non-zero rewards at intermediate time steps.
The algorithm receives observation as input and transforms it to a hidden state which is updated iteratively using its next action. At each step, the model predicts the policy (the move to play), value function (the predicted winner), and immediate reward (the points scored by playing a move); and the objective is to estimate these quantities accurately to match the ones found by the MCTS. No constraint for the hidden state  capture all information necessary to reconstruct the original observation or match the unknown / true state of environment which reduces the information that it has to maintain. The hidden state is free to represent state in whatever way is relevant to predicting current and future values and policies meaning it can invent the rules or dynamics that lead to most accurate planning.

// image of MuZero //

The model consists of three interconnected parts:

Representation function (h)
which encodes the environment observation into a hidden state.

Dynamics function (g) 
that predicts the next hidden state and immediate reward given the current state and action.

Prediction function (f)
which computes policy and value estimates from the hidden state.

**Planning**
Starting from an initial hidden state s^0, MuZero plans by iteratively applying the dynamics function g. For each action a^k, it predicts the next state s^k and immediate reward r^k. The prediction function f then estimates the policy p^k and value v^k for each new state. This creates a tree of possible future states and actions.

**Acting**
MuZero uses Monte Carlo Tree Search (MCTS) at each timestep t. The search tree is built using the learned model, starting from the current state.
Actions are sampled based on the visit count of each node, forming a search policy π_t. The final action a_t+1 is chosen based on this search policy. The environment then provides a new observation o_t+1 and reward u_t+1.

**Training**
MuZero learns from experience stored in a replay buffer. For training, it samples trajectories from past games or episodes. The representation function h processes the initial observation to get s^0. The model is then unrolled for K steps, using the dynamics function g to predict subsequent states and rewards. At each step, the prediction function f estimates policy, value, and reward. The model is trained end-to-end using backpropagation through time to minimize the difference between predicted and actual outcomes.

// image tic-tac-toe //

Loss Function of MuZero

MuZero has three primary learning objectives for every hypothetical step K

**Policy Objective**
Minimize the error between the predicted policy pt^k and the search policy π(t+k) (ensures that the model learns to take actions that align with the search's recommendations)

**Value Objective**
Minimize the error between the predicted value vt^k and the value target z(t+k) (value target is computed using bootstrapping n steps into the future search value results)

**Reward Objective**
Minimize the error between the predicted reward rt^k and the observed reward u(t+k) (ensures that the model learns to accurately predict the rewards it will receive)

// vibz wali image //

The overall loss function for MuZero is a combination of the policy, value, and reward losses, along with an L2 regularization term.

// overall loss function //
