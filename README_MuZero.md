**MUZERO** https://www.nature.com/articles/s41586-020-03051-4.epdf?sharing_token=kTk-xTZpQOF8Ym8nTQK6EdRgN0jAjWel9jnR3ZoTv0PMSWGj38iNIyNOw_ooNp2BvzZ4nIcedo7GEXD7UmLqb0M_V_fop31mMY9VBBLNmGbm0K9jETKkZnJ9SgJ8Rwhp3ySvLuTcUr888puIYbngQ0fiMf45ZGDAQ7fUI66-u7Y%3D (MuZero Paper)

Real-world problems cannot just function with a tree-based planning method, as the dynamics are complex and unknown. Until now, all the models know the environment’s dynamics or have a perfect simulator. Model-free RL estimates the optimal policy and/or value function directly from interactions with the environment but isn’t accurate at all.

// model based RL reference //
 
**Model-based Reinforcement Learning** learns a model of the environment’s dynamics; that is, it first constructs the state representation that the model should predict and then plans with respect to the learned model. Represented by Markov’s Decision Process / MDP, it has next-state prediction and expected reward prediction; MCTS is applied to compute the optimal value or optimal policy. All of this is an issue since the agent cannot optimize its representation or model for effective planning (modeling errors may compound during planning). To improve this,  predicting the value function is useful as they construct an abstract MDP model that is equivalent to planning in the real environment (using value equivalence, starting from the same real state, the cumulative reward of a trajectory through the abstract MDP matches the cumulative reward of a trajectory in the real environment). The MDP model is viewed as a hidden layer of the network, and unrolled MDP is trained such that the expected cumulative sum of rewards matches the expected value with respect to the real environment (as there is no requirement for its transition model to match real states in the environment).

**-------------------------------------------------------------------------------------------------------------**

// image consisting of connected arrows of muzero //

**MuZero Algorithm** combines tree-based search with a learned model and achieves results without any knowledge of the game's dynamics. MuZero builds upon AlphaZero’s search and search-based policy iteration algorithms and incorporates a learned model into the training procedure. It extends AlphaZero to broader environments, including single-agent domains and non-zero rewards at intermediate time steps.

The algorithm receives observation as input and transforms it to a hidden state which is updated iteratively using its next action. At each step, the model predicts the policy (the move to play), value function (the predicted winner), and immediate reward (the points scored by playing a move); and the objective is to estimate these quantities accurately to match the ones found by the MCTS.

// image of the tree //

**MuZero MCTS**

**Simulation** always starts at the root of the tree (light blue circle at the top of the figure), the current position in the environment or game. At each node (**state s**), it uses a scoring function **U(s,a)** to compare different actions **a** and choose the most promising one. The scoring function used in MuZero would combine a prior estimate **p(s,a)** with the value estimate for **v(s,a)**.

**U(s,a)=v(s,a)+c⋅p(s,a)**

where c is a scaling factor ( that ensures that the influence of the prior diminishes as our value estimate becomes more accurate )

Each time an action is selected, we increment its associated visit count **n(s,a)** for use in the UCB scaling factor c and later action selection. Simulation proceeds down the tree until it reaches a leaf that has not yet been expanded; at this point, the neural network is used to evaluate the node. Evaluation results (prior and value estimates) are stored in the node.

**Expansion** occurs once a node has reached a certain number of evaluations. Being expanded means that children can be added to a node; this allows the search to proceed deeper. In MuZero, the expansion threshold is 1, i.e. every node is expanded immediately after it is evaluated for the first time.

**Backpropagation** occurs as the value estimate from the neural network evaluation is propagated back up the search tree; each node keeps a running mean of all value estimates below it. This averaging process allows the UCB formula to make increasingly accurate decisions over time, ensuring that the MCTS eventually converges to the best move.

**-------------------------------------------------------------------------------------------------------------**

There is no constraint for the hidden state to capture all information necessary to reconstruct the original observation or match the unknown/true state of the environment, reducing the information it must maintain. The hidden state is free to represent the state in whatever way is relevant to predicting current and future values and policies, meaning it can invent the rules or dynamics that lead to the most accurate planning.

**Immediate Rewards** also have to be considered as frequent feedback; generally, a **reward r** is observed after every transition from one state to the next.

**U(s,a)=r(s,a)+γ⋅v(s')+c⋅p(s,a)**

where r(s,a) is the reward observed in transitioning from state s by choosing action a, and γ is a discount factor that describes how much we care about future rewards.

We further normalize the combined reward/value estimate to lie in the interval [0,1] before combining it with the prior:

**U(s,a) = [ ( r(s,a) + γ⋅v(s') − qmin ) / ( qmax − qmin ) ] + c⋅p(s,a)**

where qmin and qmax are the minimum and maximum **r(s,a)+γ⋅v(s')** estimates observed across the search tree.

**-------------------------------------------------------------------------------------------------------------**

**MuZero Algorithm**

**1. Planning and Episode Generation**

The MCTS procedure described above can be applied repeatedly to play entire episodes:

- Run a search in the environment's current state s(t).
- Select an action a(t+1) according to the statistics π(t) of the search.
- Apply the action to the environment to advance to the next state s(t+1) and observe reward u(t+1).
- Repeat until the environment terminates.

**-------------------------------------------------------------------------------------------------------------**

**2. Training**

We sample a trajectory and a position within it from our dataset, and then we unroll the MuZero model alongside the trajectory.

// image of training //

// image of MuZero ka 3 parts //

- the representation function h maps from a set of observations to the hidden state s used by the neural network
- the dynamics function g maps from a state s(t) to the next state s(t+1), it also estimates the reward r(t) observed in this transition (this allows the learned model to roll forward inside the search)
- the prediction function f makes estimates for policy p(t) and value v(t) based on a state s(t). These estimates are used by the UCB formula and aggregated in the MCTS.

The observations and actions used as input to the network are taken from this trajectory; similarly, the prediction targets for policy, value, and reward are stored with the trajectory when generated. 

MuZero has three primary learning objectives for every hypothetical **step K**

**Policy Objective**
Minimize the error between the predicted policy pt^k and the search policy π(t+k) (ensures that the model learns to take actions that align with the search's recommendations)

**Value Objective**
Minimize the error between the predicted value vt^k and the value target z(t+k) (value target is computed using bootstrapping n steps into the future search value results)

**Reward Objective**
Minimize the error between the predicted reward rt^k and the observed reward u(t+k) (ensures that the model learns to predict the rewards it will receive accurately)

The overall **Loss Function for MuZero** is a combination of the policy, value, and reward losses, along with an **L2 regularization term**.

// overall loss function //

**-------------------------------------------------------------------------------------------------------------**

**3. Reanalyzing**

While training, we generated many trajectories and stored them in the replay buffer, but as this is stored already, we cannot change states, actions, or received rewards as this would reset the environment. 
What we can do is use existing inputs with fresh, improved labels is enough for continued learning using MuZero's learned model and the MuZero MCTS. We keep the saved trajectory (observations, actions, and rewards) as is and instead only re-run the MCTS. This generates fresh search statistics, providing us with new targets for the policy and value prediction.

// reanalyze image //

Two sets of jobs that communicate with each other asynchronously:

- a learner that receives the latest trajectories keeps the most recent of these in a replay buffer and uses them to perform the training algorithm described above.
- multiple actors that periodically fetch the latest network checkpoint from the learner use the network in MCTS to select actions and interact with the environment to generate trajectories.

Two jobs are introduced to implement reanalyze:

- a reanalyze buffer that receives all trajectories generated by the actors and keeps the most recent ones.
- multiple reanalyze actors that sample stored trajectories from the reanalyze buffer, re-run MCTS using the latest network checkpoints from the learner, and send the resulting trajectories with updated search statistics to the learner.

// image tic-tac-toe //

**-------------------------------------------------------------------------------------------------------------**
