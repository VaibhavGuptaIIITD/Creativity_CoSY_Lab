**ALPHAZERO IMPLEMENTATION**

AlphaZero works with the modified version of **Monte Carlo Tree Search** which aids in choice-making and game playing. It follows multiple steps and computes decisions to make sure the steps taken, whether from existing options or from expanded steps ultimately leads to the best possible output, which it improves upon by playing against itself over time. It follows the exploration-exploitation trade-off, which is making sure that choosing options which have given the best outputs till now as well as taking chances with options which haven't been chosen much as of now, in the hope that it improves later.

In Monte Carlo Tree Search, we look at the current situation that the game is in right now, and the action that we can take which looks the most promising for winning the game. Consider a tree with a root node which would be considered State 0, and it has 2 children, State 1 and State 2 which can be reached by undergoing Action 1 and Action 2 respectively. For each State of the game or each node of the tree, there are 2 parameters, the number of wins that has been recorded from that node and the total number of times that node has been traversed. The overall process is divided into 4 major parts, which in sequence are as follows :

1. **Selection**

This refers to selecting the child we would like to move to from the parent node, or the action we would like to perform which would guarantee better returns in the future. The direction we choose to traverse downwards depends on the computation of the Upper Confidence Bound (UCB) formula that takes into consideration both the winning ratio of the node as well as giving opportunity to the nodes which were traversed less often. 

![Upper Confidence Bound](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/WhatsApp%20Image%202024-08-31%20at%2013.52.31.jpeg?raw=true)

2. **Expansion**

This phase refers to creation of a new node in the tree or new state in the game, which can be done from a pre-existing parent state by taking a new action. On creation of a node, the number of wins and number of visits are initially set as 0.

3. **Simulation**

This state of the game refers to playing randomly from a state until the game ends, that is a draw, loss or win is recorded in the process. At the end of this phase, we reach the terminal node and record the outcome of the game which is helpful for the next phase.

4. **Backpropagation**

At this state, we start traversing backwards / upwards from the terminal node all the way to the root node encountering all the states and actions we took earlier to reach the conclusion of the game. While traversing backwards, we increase the number of visits count by 1 for all the traversed nodes / states, as well as increment the number of wins count by either 1 or 0.5 in case of a win or draw respectively.

![Monte Carlo Tree Search]()

Now, to make sure that AlphaZero could work based on this, we modify the existing Monte Carlo Tree Search to **Alpha MCTS** by incorporating a few key changes. One of the key changes is that the process of Simulation, that is randomly playing, has been eliminated and a parameter Value has been added which was computed by the Neural Network when it evaluated a certain State / Node in the tree. The other change is that a new parameter, Policy has been added which is an estimation of likelihood of selection of the child node from the parent node. A higher policy means that it is more likely that particular child node will now be more preferred. So, now the policy for selection, as well as value for backpropagation are imperative. This also means that the formula for Upper Confidence Bound (UCB) has been updated. The updated formula is as follows :

![Updated UCB](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/WhatsApp%20Image%202024-08-31%20at%2013.52.32%20(1).jpeg?raw=true)

Let’s say State 0 exists as the root node in the tree with no children whatsoever. It would then have the number of visits as 0, the number of wins as 0 and the policy of state as 1. The neural network finds out that the State 0 has policy distribution of 0.6 and 0.4 across its two children nodes during Expansion phase ( State 1 and State 2 which can be reached by committing Action 1 and Action 2 respectively ), as well as a Value of 0.4. Now, the backpropagation phase would increment State 0’s number of visits as 1 and the number of wins as 0.4. That is the Value that the Neural Network found out from the State before. The next time selection of the nodes is done using the updated UCB formula above and the cycle repeats until the game finishes.

![Alpha MCTS](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/WhatsApp%20Image%202024-08-31%20at%2013.52.30%20(2).jpeg?raw=true)

To train the model, it plays with itself in order to identify the best states and their respective actions for the future. From each position, the model plays against itself on the basis of the Monte Carlo Tree Search distribution until there's an outcome to the game. For each given state, the reward is equal to the final outcome of the player; that is the chance that the player might be in the game from that position onwards.

![Self Play MCTS Distribution](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/WhatsApp%20Image%202024-08-31%20at%2013.52.30%20(1).jpeg?raw=true)

The model is trained on the basis of the dataset present, that is S (state), Pi (MCTS distribution), Z (reward) are fed in as sample data to the model, which then provides the policy and value for a given state as output. To improve the model, loss is minimized using the loss formula, using backpropagation.

![Training and Loss](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/WhatsApp%20Image%202024-08-31%20at%2013.52.30.jpeg?raw=true)
