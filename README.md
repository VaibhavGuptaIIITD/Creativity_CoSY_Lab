**ALPHAZERO IMPLEMENTATION** https://arxiv.org/pdf/1712.01815 (AlphaZero Paper Link)

[1]https://youtu.be/wuSQpLinRB4?si=TUlC5nD7B-gQaD-f (AlphaZero Video Reference 1)

[2]https://youtu.be/62nq4Zsn8vc (AlphaZero Video Reference 2)

AlphaZero works with the modified version of **Monte Carlo Tree Search** which aids in choice-making and game playing. It follows multiple steps and computes decisions to make sure the steps taken, whether from existing options or from expanded steps ultimately leads to the best possible output, which it improves upon by playing against itself over time. It follows the exploration-exploitation trade-off, which is making sure that choosing options which have given the best outputs till now as well as taking chances with options which haven't been chosen much as of now, in the hope that it improves later.

[3]https://www.geeksforgeeks.org/ml-monte-carlo-tree-search-mcts/ (MCTS Article Reference)

[4]https://youtu.be/UXW2yZndl7U (MCTS Video Reference)

In Monte Carlo Tree Search, we look at the current situation that the game is in right now, and the action that we can take which looks the most promising for winning the game. Consider a tree with a root node which would be considered State 0, and it has 2 children, State 1 and State 2 which can be reached by undergoing Action 1 and Action 2 respectively. For each State of the game or each node of the tree, there are 2 parameters, the number of wins that has been recorded from that node and the total number of times that node has been traversed. The overall process is divided into 4 major parts, which in sequence are as follows :

1. **Selection**

This refers to selecting the child we would like to move to from the parent node, or the action we would like to perform which would guarantee better returns in the future. The direction we choose to traverse downwards depends on the computation of the Upper Confidence Bound (UCB) formula that takes into consideration both the winning ratio of the node as well as giving opportunity to the nodes which were traversed less often. 

![Upper Confidence Bound](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/WhatsApp%20Image%202024-08-31%20at%2013.52.31.jpeg?raw=true)

[5]https://www.geeksforgeeks.org/upper-confidence-bound-algorithm-in-reinforcement-learning/ (Upper Confidence Bound)

2. **Expansion**

This phase refers to creation of a new node in the tree or new state in the game, which can be done from a pre-existing parent state by taking a new action. On creation of a node, the number of wins and number of visits are initially set as 0.

3. **Simulation**

This state of the game refers to playing randomly from a state until the game ends, that is a draw, loss or win is recorded in the process. At the end of this phase, we reach the terminal node and record the outcome of the game which is helpful for the next phase.

4. **Backpropagation**

At this state, we start traversing backwards / upwards from the terminal node all the way to the root node encountering all the states and actions we took earlier to reach the conclusion of the game. While traversing backwards, we increase the number of visits count by 1 for all the traversed nodes / states, as well as increment the number of wins count by either 1 or 0.5 in case of a win or draw respectively.

![Monte Carlo Tree Search](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/WhatsApp%20Image%202024-08-31%20at%2013.52.32.jpeg?raw=true)

Now, to make sure that AlphaZero could work based on this, we modify the existing Monte Carlo Tree Search to **Alpha MCTS** by incorporating a few key changes. One of the key changes is that the process of Simulation, that is randomly playing, has been eliminated and a parameter Value has been added which was computed by the Neural Network when it evaluated a certain State / Node in the tree. The other change is that a new parameter, Policy has been added which is an estimation of likelihood of selection of the child node from the parent node. A higher policy means that it is more likely that particular child node will now be more preferred. So, now the policy for selection, as well as value for backpropagation are imperative. This also means that the formula for Upper Confidence Bound (UCB) has been updated. The updated formula is as follows :

![Updated UCB](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/WhatsApp%20Image%202024-08-31%20at%2013.52.32%20(1).jpeg?raw=true)

Let’s say State 0 exists as the root node in the tree with no children whatsoever. It would then have the number of visits as 0, the number of wins as 0 and the policy of state as 1. The neural network finds out that the State 0 has policy distribution of 0.6 and 0.4 across its two children nodes during Expansion phase ( State 1 and State 2 which can be reached by committing Action 1 and Action 2 respectively ), as well as a Value of 0.4. Now, the backpropagation phase would increment State 0’s number of visits as 1 and the number of wins as 0.4. That is the Value that the Neural Network found out from the State before. The next time selection of the nodes is done using the updated UCB formula above and the cycle repeats until the game finishes.

![Alpha MCTS](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/WhatsApp%20Image%202024-08-31%20at%2013.52.30%20(2).jpeg?raw=true)

[6]https://youtu.be/0slFo1rV0EM?si=Su3XllFRajJxVP4J (AlphaMCTS and Neural Network Working Video Reference)

To train the model, it plays with itself in order to identify the best states and their respective actions for the future. From each position, the model plays against itself on the basis of the Monte Carlo Tree Search distribution until there's an outcome to the game. For each given state, the reward is equal to the final outcome of the player; that is the chance that the player might be in the game from that position onwards.

![Self Play MCTS Distribution](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/WhatsApp%20Image%202024-08-31%20at%2013.52.30%20(1).jpeg?raw=true)

Now, onto the neural network model that is used to predict the policy and value from a particular state at a particular time of the game. Say, there's a State at which the tic-tac-toe game is currently on, this could be represented by 3 matrices ( stored as linear lists ) of 9 elements ( 3x3 as in a tic-tac-toe ) consisting of 1s and 0s. One matrix is for player 1s moves, the position they played on is marked as 1 and the rest is 0, another matrix does the same for player -1, and the third matrix stores the empty and occupied positions currently in the game ( occupied being 1 and empty being 0 ). 

![Neural Network Architecture](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/PHOTO-2024-09-06-14-53-53.jpg?raw=true)

This is fed into a Convolutional Neural Network, specifically a Residual Network / ResNet which is the backbone of the neural network computation here. It helps process the states first which are fed in from the above step. So, in the neural network, the inputs are fed in which in this case is the states of the tic-tac-toe game which the model processes with weights and biases as usual to compute two types of outputs, policies and values. 

[7]https://www.geeksforgeeks.org/vanishing-and-exploding-gradients-problems-in-deep-learning/ (Vanishing and Exploding Gradients Reference)

Now, for a model to compute outputs where there are a high possibility of solution steps, the convolutional neural network thus created is a deep one with a large number of layers to accurately predict the output. This is where the problem of vanishing and exploding gradients occur. When the neural network becomes more and more complex with a large number of layers, either the slopes of the activation functions become progressively smaller or larger as we move backward through the layers of a neural network in backpropagation. This could result in either updates becoming exponentially small, prolonging the training time / halt the training process altogether, or the gradient failing to converge and can lead to the network oscillating around local minima, making it challenging to reach the global minimum point.

[8]https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/ (Residual Network Architecture)

To come up with a solution for this problem, ResNet architecture is introduced where a technique called skip connections is used, which connects activations of a layer to further layers by skipping some layers in between forming a residual block which when stacked together forms a ResNet. The approach behind this network is instead of layers learning the underlying mapping, we allow the network to fit the residual mapping. So, instead of say H(x), initial mapping, let the network fit, F(x) := H(x) - x , which gives H(x) := F(x) + x.
The advantage of adding this type of skip connection is that any layer that hurts the performance of architecture, will be skipped by regularization which results in training a very deep neural network without the problems caused by vanishing/exploding gradient. 

[9]https://www.analyticsvidhya.com/blog/2021/04/introduction-to-softmax-for-neural-network/ (Softmax Function)

[10]https://www.ml-science.com/tanh-activation-function (Tan h function)

Finally, the model provides two types of outputs, one being the policies for the given node and the other being the value. For the policy, the output layer is fed into a softmax function which outputs the same dimensional layer but as a probability distribution ( adding up to 1 ). For the value, the output is fed to the Tan function to make sure that the output flattens out to a number between -1 and 1. This forms the basics of how neural-network architecture works in this case.

The model is trained on the basis of the dataset present, that is S (state), Pi (MCTS distribution), Z (reward) are fed in as sample data to the model, which then provides the policy and value for a given state as output. The optimal policy is then decided using these arguments by calculating the loss incurred on each state and thus changing the parameters and tuning them to minimise this loss. 

![Training and Loss](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/WhatsApp%20Image%202024-08-31%20at%2013.52.30.jpeg?raw=true)

1. **(z-v)^2**: Here z is the reward value that will be given by the MCTS tree by performing simulations of the game from a given state and v is the reward value given by the raw network. This term is minimising the difference between required reward, and the reward provided by the raw network, thus decreasing the loss. Here, the difference has been squared to keep the value positive.
2. **pi^(T).log(p)**: In this term, pi is the probability distribution that has been given by the MCTS distribution and P is the probability distribution given by the raw network. Here in this term we are taking the dot product of both of these terms which forces the value of p to come close to the value of pi, as because more closer the value of pi and p, greater will be the value of dot product, thus decreasing the overall loss value.
3. **c|theta|^2**: This term is reponsible for l2 regularisation. It is a form of ridge regression that is done to prevent overfitting of data by preventing the model to learn the noise and random fluctuations in the training data. This is done by modifying the training data according to the formula, where c is the regularisation coefficient.
