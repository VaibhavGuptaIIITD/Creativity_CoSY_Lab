# RL Based Protein Folding Prototype 

## Simplified Option: Hydrophobic-Polar (HP) Lattice Model  
For **initial proof-of-concept testing** with reinforcement learning (RL), the HP model offers unparalleled simplicity:  

### HP Model Overview 

​The Hydrophobic-Polar (HP) model is a simplified computational framework designed to study the fundamental principles of protein folding. Introduced by Ken Dill in 1985, this model abstracts the complex nature of proteins by categorizing amino acids into two types: hydrophobic (H) and polar (P). By focusing on the interactions between these two types, the HP model aims to capture the essential driving forces behind protein folding.​ This ignores real-world side-chain complexity and focuses on hydrophobic collapse, a core folding principle [^1][^2].  


![HP Model](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/Screenshot%202025-04-17%20at%2023.38.03.png?raw=true)


1. **Simplified Representation**: In the HP model, a protein is represented as a chain of monomers placed on a lattice (either 2D or 3D). Each monomer corresponds to an amino acid and is classified as either hydrophobic (H) or polar (P). The chain forms a self-avoiding walk on the lattice, ensuring that no two monomers occupy the same position.

2. **Hydrophobic Interactions**: The model emphasizes the tendency of hydrophobic residues to avoid contact with the aqueous environment. Therefore, non-covalent contacts between adjacent hydrophobic monomers (not sequentially connected) are energetically favorable. These interactions drive the folding process, leading to the formation of a compact hydrophobic core surrounded by polar residues.

3. **Tractable Conformational Space**: Constrained to 2D/3D lattices, short sequences (e.g., `HPPHHP`) have limited valid folds, avoiding Levinthal’s paradox [^3]. ​Levinthal's Paradox highlights the discrepancy between the astronomical number of possible protein conformations and the rapid folding observed in nature. If a protein sampled all conformations randomly, folding would take longer than the universe's age. Yet, proteins fold within milliseconds to seconds. This suggests that folding is guided by specific pathways and energy landscapes, not random sampling. 

4. **Energy Minimization**: The stability of a folded conformation is assessed by its energy, calculated based on the number of favorable H-H contacts. The native state of the protein corresponds to the conformation with the lowest possible energy, representing the most stable structure. Reward = Number of non-consecutive H-H contacts. Easy to compute and ideal for sparse reward RL [^4].

5. **Benchmarked in RL Studies**: Used in frameworks like [Gym-Lattice](https://ljvmiranda921.github.io/projects/2018/05/13/gym-lattice/) and [FoldingZero](https://arxiv.org/abs/1812.00967) [^5][^6].


[^1]: Dill, K. A. (1985). "Theory for the folding and stability of globular proteins." *Biochemistry*.  
[^2]: Lau, K. F. & Dill, K. A. (1989). "A lattice statistical mechanics model of the conformational and sequence spaces of proteins." *Macromolecules*.  
[^3]: Levinthal, C. (1969). "How to fold graciously." *Mossbauer Spectroscopy in Biological Systems*.  
[^4]: Černý, V. (1985). "Thermodynamical approach to the traveling salesman problem." *Journal of Optimization Theory and Applications*.  
[^5]: Gym-Lattice repository (2022). [Link](https://ljvmiranda921.github.io/projects/2018/05/13/gym-lattice/).  
[^6]: FoldingZero paper (2023). [DOI]([https://doi.org/yourlink](https://arxiv.org/abs/1812.00967)).  

---

### Gym-Lattice Environment for Protein Folding

The Gym-Lattice project by Lester James Miranda is a reinforcement learning (RL) environment designed to simulate the protein folding problem using the 2D Hydrophobic-Polar (HP) lattice model. This environment formulates protein folding as a Markov Decision Process (MDP), enabling RL agents to learn optimal folding strategies.

In this framework, protein folding is modeled as a sequential decision-making process

- **States ($s_t$)**: Represent the current configuration of the protein on a 2D lattice.
- **Actions ($a_t$)**: Choices to place the next amino acid in one of four directions—left, down, up, or right.
- **Rewards ($r_t$)**: Feedback based on the stability of the resulting structure, with penalties for invalid moves. The agent aims to learn a policy $\pi(a_t | s_t)$ that maximizes the expected cumulative reward over an episode.


**Mathematical Formulation**

The energy function $E(\mathcal{C})$ for a given conformation $\mathcal{C}$ is defined as:

$$
E(C) = Σ I(i, j)
$$

where the interaction function $I(i,j)$ i:

$$
I(i, j) = 
  -1, if pᵢ = pⱼ = H and |xᵢ - xⱼ| + |yᵢ - yⱼ| = 1  
   0, otherwise
$$

Here:
$p_i$, $p_j$ denote the types (H or P) of the ith and jth amino acids.
$(x_i, y_i)$ and $(x_j, y_j)$ are their positions on the lattice.
This rewards non-consecutive hydrophobic amino acids that are adjacent on the lattice.

**State Representation**

The lattice is a 2D grid $S \in {-1, 0, +1}^{(2n+1) \times (2n+1)}$ where:

- +1: Hydrophobic (H)
- -1: Polar (P)
- 0: Empty

**Action Space**

The agent can choose from four discrete actios:

- 0: Left
- 1: Down
- 2: Up
- 3: Right

**Reward Function**

At each timestep $t$, the reward $r_t$ is calculated as:

$$
r_t = state_reward + collision_penalty + trap_penalty
$$

- **state_reward*: Calculated at the end of the episode as the total number of non-consecutive adjacent H-H pais.
- **collision_penalty*: Applied when the agent attempts to place an amino acid on an already occupied space (default: -2).
- **trap_penalty*: Applied if the agent traps itself, preventing the completion of the sequene.

**Workflow Overview**

1. **Initialization*: The environment is initialized with a given HP sequece.
2. **Agent Decision*: At each step, the agent selects an action based on the current stte.
3. **Environment Update*: The environment updates the lattice configuration based on the acton.
4. **Reward Calculation*: The environment computes the reward and checks for termination conditins.
5. **Iteration*: Steps 2–4 are repeated until the sequence is fully placed or the agent is traped.


![Gym 1](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/Screenshot%202025-04-18%20at%2000.31.51.png?raw=true)

![Gym 2](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/Screenshot%202025-04-18%20at%2000.32.07.png?raw=true)


The Gym-Lattice environment provides a simplified yet insightful platform for exploring protein folding through reinforcement learig. By abstracting the complex nature of proteins into a 2D lattice model, it allows researchers and enthusiasts to experiment with RL algorithms in a controlled seting.


## Biologically Relevant Option: Real Protein Families 
For **real protein sequences** with minimal complexity, consider these experimentally validated families:  

### Recommended Families  
1. **Villin Headpiece (HP35/HP36)**  
   - **Size**: 35–36 residues.  
   - **Features**: Microsecond-folding three-helix bundle, abundant NMR/X-ray data [^7][^8].  
   - **Example PDB**: [1VII](https://www.rcsb.org/structure/1VII).  

2. **Chignolin**  
   - **Size**: 10 residues (synthetic).  
   - **Features**: Ultra-fast β-hairpin fold, ideal for RL validation [^9].  
   - **Example PDB**: [1UAO](https://www.rcsb.org/structure/1UAO).  

3. **B1 Domain of Protein G (GB1)**  
   - **Size**: 56 residues.  
   - **Features**: Cooperative folding, CASP benchmark [^10].  
   - **Example PDB**: [1PGA](https://www.rcsb.org/structure/1PGA).  

### Why These Proteins?  
- **Small Size**: Reduces computational load (e.g., Chignolin has only 10 residues).  
- **Rich Data**: Experimentally solved structures enable reward function design and validation.  
- **RL Compatibility**: Used in AlphaFold/Rosetta training pipelines [^11][^12].  
 
[^7]: Kubelka, J. et al. (2004). "The folding kinetics of the villin headpiece." *Journal of Molecular Biology*.  
[^8]: AlphaFold Dataset (2021). [DOI](https://doi.org/10.1038/s41586-021-03819-2).  
[^9]: Honda, S. et al. (2008). "Ultra-fast folding of a β-hairpin in aqueous solution." *Journal of the American Chemical Society*.  
[^10]: CASP Competition (2020). [Link](https://predictioncenter.org/casp14/).  
[^11]: DeepMind AlphaFold (2021). [Nature](https://www.nature.com/articles/s41586-021-03819-2).  
[^12]: Rosetta@Home (2023). [Link](https://boinc.bakerlab.org/rosetta/).  

---

# Composite Reward Function for RL-Based Protein Folding

This is a composite reward function for a reinforcement learning (RL) agent applied to protein folding. The overall reward is given as a weighted sum of several components, each reflecting a biophysical or evolutionary criterion that guides the system toward a native-like folded structure. Formally:

$$
R(s,a) = \sum_{k \in \{\,c,\,cl,\,b,\,r,\,hb,\,hp,\,e,\,t,\,p,\,d,\,rec,\,int\}} w_k\, R_k(s,a).
$$

Here:
- \(s\) is the current state (e.g., coordinates, bond lengths, torsion angles),
- \(a\) is the action taken,
- \(w_k\) are positive hyperparameters (weights),
- \(R_k(s,a)\) are the individual reward (or penalty) components.

Below, each reward component is described in detail.

---

## 1. Contact Map Agreement \(R_c(s)\)

Rewards agreement with AlphaFold-predicted contacts [^8][^11]:

### Mathematical Formula

$$
R_c(s) = \sum_{i,j}\,\mathbf{1}\bigl(d_{ij}(s) \le \tau\bigr)\,\mathbf{1}\bigl(d_{ij}^{AF} \le \tau\bigr),
$$

where:
- \(d_{ij}(s)\) is the Euclidean distance between residues \(i\) and \(j\),
- \(d_{ij}^{AF}\) is the predicted contact from AlphaFold,
- \(\tau\) is a threshold (e.g., 8\,\AA),
- \(\mathbf{1}(\cdot)\) is the indicator function (1 if true, 0 if false).

### Intuition and Rationale

Residue--residue contacts are often inferred from evolutionary data (MSAs). This term rewards the RL agent for reproducing the contacts predicted by AlphaFold. Conformations that satisfy these “evolutionary couplings” are likelier to be near the native fold.

---

## 2. Steric Clash Penalty \(R_{cl}(s)\)

Penalizes atom overlaps, inspired by force fields like AMBER/CHARMM [^13]:

### Mathematical Formula

$$
R_{cl}(s) = -\sum_{i < j}\,\mathbf{1}\bigl(d_{ij}(s) < d_{\min}\bigr),
$$

where:
- \(d_{ij}(s)\) is the distance between residues \(i\) and \(j\),
- \(d_{\min}\) is the minimum allowable distance,
- \(\mathbf{1}\) is the indicator function.

### Intuition and Rationale

Steric clashes occur when atoms come too close, leading to physically impossible overlaps. This term penalizes each clash with a fixed negative contribution. By enforcing a minimal distance \(d_{\min}\), the agent is guided to produce physically valid conformations.

---

## 3. Bond Geometry Reward \(R_b(s)\)

Enforces ideal bond lengths (3.8 Å) [^14]: 

### Mathematical Formula

$$
R_b(s) = -\frac{1}{N-1}\,\sum_{i=1}^{N-1}\,\Bigl(\,\|x_{i+1} - x_i\| - L_{\mathrm{ideal}}\Bigr)^2,
$$

where:
- \(x_i\) is the coordinate of residue \(i\),
- \(N\) is the number of residues,
- \(L_{\mathrm{ideal}}\) is the ideal bond length (e.g., 3.8\,\AA).

### Intuition and Rationale

Local geometry must be correct for a realistic fold. This term penalizes deviations from the ideal bond length. Quadratic punishment ensures small errors remain small, while large deviations are heavily penalized. This helps maintain a coherent backbone structure.

---

## 4. Ramachandran (Torsion Angle) Reward \(R_r(s)\)

Uses Ramachandran distributions from MolProbity [^15]: 

### Mathematical Formula

$$
R_r(s) = \sum_{i=1}^N\,\log\,P_{\mathrm{rama}}\bigl(\phi_i,\,\psi_i\bigr),
$$

where \(P_{\mathrm{rama}}(\phi_i,\psi_i)\) is the probability of dihedral angles \(\phi_i,\psi_i\) (from Ramachandran distributions).

### Intuition and Rationale

Protein backbone angles are constrained by sterics and energetics. The Ramachandran plot indicates favored regions. Summing the log-probabilities rewards angles in these favored zones, promoting realistic local conformations.

---

## 5. Hydrogen Bond Reward \(R_{hb}(s)\)

Rewards H-bond formation (β-sheets/α-helices) [^16]:  

### Mathematical Formula

$$
R_{hb}(s) = \sum_{(i,j)\in\mathcal{HB}} h_{ij}(s),
$$

where:
- \(\mathcal{HB}\) is the set of residue pairs that can form hydrogen bonds,
- \(h_{ij}(s)\) measures the presence or strength of a hydrogen bond.

### Intuition and Rationale

Hydrogen bonds stabilize secondary structures (e.g., helices, sheets). Rewarding their formation encourages the agent to produce locally stable, biologically realistic motifs.

---

## 6. Hydrophobic Packing Reward \(R_{hp}(s)\)

Minimizes SASA of hydrophobic residues [^1][^2]:  

### Mathematical Formula

$$
R_{hp}(s) = -\frac{1}{|\mathcal{H}|}\,\sum_{i \in \mathcal{H}}\,\mathrm{SASA}\bigl(x_i\bigr),
$$

where:
- \(\mathcal{H}\) is the set of hydrophobic residues,
- \(\mathrm{SASA}(x_i)\) is the solvent-accessible surface area of residue \(i\).

### Intuition and Rationale

Hydrophobic residues typically cluster inside the protein core. This term penalizes large solvent-exposed areas for these residues, thus rewarding a well-packed hydrophobic core essential for overall stability.

---

## 7. Electrostatic Energy Penalty \(R_e(s)\)

Coulombic interactions from MD force fields [^13]: 

### Mathematical Formula

$$
R_e(s) = -\sum_{i < j}\,\frac{q_i\,q_j}{\|\,x_i - x_j\,\| + \epsilon},
$$

where:
- \(q_i\) is the charge of residue \(i\),
- \(\|x_i - x_j\|\) is the distance between residues \(i\) and \(j\),
- \(\epsilon\) is a small constant.

### Intuition and Rationale

Electrostatic interactions can stabilize or destabilize a fold. Like charges (positive product) yield a penalty; opposite charges (negative product) contribute favorably. By dividing by distance, closer interactions are emphasized. The constant \(\epsilon\) prevents division by zero.

---

## 8. Template Similarity Reward \(R_t(s)\)

Uses TM-score for template alignment [^17]:  

### Mathematical Formula

A common choice is inverse RMSD:

$$
R_t(s) = \frac{1}{\mathrm{RMSD}\bigl(s,\,s^{\mathrm{template}}\bigr) + \epsilon}.
$$

Alternatively, a TM-score (0--1) can be used.

### Intuition and Rationale

If a known template structure is available, this term rewards similarity to that template. Lower RMSD means the predicted fold is closer to the template, which can accelerate finding a realistic structure.

---

## 9. pLDDT-Weighted Conformational Accuracy Reward \(R_p(s)\)

Leverages AlphaFold’s confidence scores [^11]: 

### Mathematical Formula

$$
R_p(s) = \frac{1}{\tfrac1N \sum_{i=1}^N p_i\,d\bigl(x_i,\,x_i^{\mathrm{true}}\bigr) + \epsilon},
$$

where:
- \(p_i\) is the pLDDT (per-residue confidence) score,
- \(d(x_i,x_i^{\mathrm{true}})\) is the deviation from the true position,
- \(\epsilon\) is a small constant.

### Intuition and Rationale

AlphaFold’s pLDDT score reflects how confident it is in each residue’s position. Weighting the positional error by pLDDT and taking the inverse rewards the agent for achieving higher accuracy in the most confident regions. This is especially useful if partial ground-truth data or high-confidence predictions are known.

---

## 10. Distogram Consistency Reward \(R_d(s)\)

Matches AlphaFold’s distogram predictions [^11]:  

### Mathematical Formula

$$
R_d(s) = -\,D_{\mathrm{KL}}\Bigl(P_{\mathrm{agent}}(\cdot \mid s)\,\Big\|\,P_{AF}(\cdot)\Bigr),
$$

where \(D_{\mathrm{KL}}\) is the Kullback--Leibler divergence between the agent’s distance distribution \(P_{\mathrm{agent}}\) and AlphaFold’s predicted distogram \(P_{AF}\).

### Intuition and Rationale

AlphaFold predicts a probability distribution over inter-residue distances (the distogram). This term penalizes divergence from that distribution. A lower KL divergence indicates better agreement with AlphaFold’s learned statistics, guiding the agent toward conformations that match evolutionary and structural patterns.

---

## 11. Recycling Consistency Reward \(R_{rec}(s)\)

Stabilizes iterative refinement (AlphaFold recycling) [^11]:  

### Mathematical Formula

$$
R_{rec}(s) = -\,\frac{1}{N}\,\sum_{i=1}^N \bigl\|\,x_i^{(t)} - x_i^{(t-1)}\bigr\|^2,
$$

where \(x_i^{(t)}\) and \(x_i^{(t-1)}\) are positions of residue \(i\) in consecutive recycling iterations.

### Intuition and Rationale

AlphaFold uses iterative refinement (“recycling”). A stable structure will change little between iterations. This term penalizes large changes, indicating that once the conformation is close to stable, further big adjustments are undesirable. This helps ensure convergence.

---

## 12. Intrinsic (Curiosity) Reward \(R_{int}(s,a)\)

Encourages exploration via prediction error [^18]:  

### Mathematical Formula

$$
R_{int}(s,a) = \bigl\|\,f(s) \;-\; \hat{f}(s,a)\bigr\|^2,
$$

where:
- \(f(s)\) is the predicted next state from a forward model,
- \(\hat{f}(s,a)\) is the observed next state after action \(a\).

### Intuition and Rationale

In complex or sparse-reward tasks like protein folding, intrinsic rewards encourage exploration of novel states. This term rewards the agent for visiting states that produce high prediction error in the forward model, prompting exploration beyond familiar conformations.

---

## Composite Reward Summary

Bringing all components together:

$$
\begin{aligned}
R(s,a) \;=\;& w_c\,R_c(s) \;+\; w_{cl}\,R_{cl}(s) \;+\; w_b\,R_b(s) \;+\; w_r\,R_r(s) \\
&+\; w_{hb}\,R_{hb}(s) \;+\; w_{hp}\,R_{hp}(s) \;+\; w_e\,R_e(s) \;+\; w_t\,R_t(s) \\
&+\; w_p\,R_p(s) \;+\; w_d\,R_d(s) \;+\; w_{rec}\,R_{rec}(s) \;+\; w_{int}\,R_{int}(s,a).
\end{aligned}
$$

Each \(w_k\) can be tuned to balance its contribution. Together, these terms guide the RL agent to produce protein conformations that are physically realistic, evolutionarily consistent, and optimized under known biophysical principles.

---

## Proposed Model

The proposed model is a deep reinforcement learning framework that learns to fold proteins by optimizing a composite reward function. The model uses a protein's amino acid sequence as input and produces three-dimensional coordinates representing its folded structure.


![proposed model](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/Screenshot%202025-04-18%20at%2000.53.12.png?raw=true)


**Key Components**

1. **State Space**: The state represents the current conformation of the protein, including:
   - 3D coordinates of each residue
   - Torsion angles (φ, ψ) along the backbone
   - Current bond lengths
   - Partial folding information

2. **Action Space**: Actions modify the protein conformation through:
   - Torsion angle adjustments
   - Local refinement moves
   - Fragment replacements
   - Global topology modifications

3. **Policy Network**: A neural network that predicts the optimal folding actions given the current state.

4. **Value Network**: Estimates the expected future reward from the current state.

5. **Environment**: Simulates protein physics and calculates the composite reward.

6. **External Knowledge Integration**: Incorporates AlphaFold predictions, templates, and biophysical constraints.

**Training Methodology**

The model employs a variant of Proximal Policy Optimization (PPO) with these key features:

1. **Curriculum Learning**: Training begins with small protein fragments and gradually increases complexity.

2. **Multi-level Representation**: The model uses both coarse-grained (Cα backbone) and fine-grained (all-atom) representations at different stages.

3. **Hybrid Exploration Strategy**:
   - Intrinsic curiosity reward to discover novel conformations
   - Temperature-scaled sampling during early training
   - Gradually decreasing stochasticity as training progresses

4. **Reward Shaping**: Dynamic adjustment of reward component weights based on training progress:
   - Early stages: Emphasize basic physics (bond geometry, clash avoidance)
   - Mid-stages: Increase importance of evolutionary information
   - Late stages: Refinement with all components

**State Representation**

The state is encoded using:
- Residue-level features (amino acid identity, hydrophobicity, charge)
- Structural features (distances, angles, relative orientations)
- Graph-based representation of residue interactions
- External predictions from AlphaFold (pLDDT scores, predicted contacts)

**Neural Network Architecture**

1. **Encoder**:
   - Graph Neural Network (GNN) to capture residue relationships
   - Transformer encoder to process sequential information

2. **Policy Network**:
   - Actor-critic architecture
   - Outputs probability distributions over actions
   - Recurrent layers to maintain structural coherence

3. **Forward Model**:
   - Predicts next state after action execution
   - Used for intrinsic reward calculation and planning

**Environment Physics**

The environment implements:
- Fast differentiable simulation of protein backbone kinematics
- Calculation of all reward components from the provided composite function
- Enforcement of biophysical constraints through soft penalties

**Algorithmic Overview**

1. **Input Processing**:
   - Parse amino acid sequence
   - Generate initial random conformation
   - Load AlphaFold predictions or templates if available

2. **Training Loop**:
   - Sample action from policy network
   - Apply action to current state
   - Calculate composite reward
   - Update policy via PPO
   - Update forward model to improve prediction accuracy

3. **Refinement Phase**:
   - Fine-tune structure with emphasis on local quality metrics
   - Implement recycling consistency for convergence
   - Apply side-chain packing and energy minimization

---

## Advice 1

## Applying Deep Reinforcement Learning to the HP Model    

**Paper**: [Applying Deep Reinforcement Learning to the HP Model for Protein Structure Prediction](https://arxiv.org/pdf/2211.14939)

**GitHub**: [Applying Deep Reinforcement Learning to the HP Model for Protein Structure Prediction](https://github.com/CompSoftMatterBiophysics-CityU-HK/Applying-DRL-to-HP-Model-for-Protein-Structure-Prediction)  

---

**1. HP Model on a 2D Square Lattice**

A protein of length N is a self‑avoiding walk (SAW) on the integer lattice.  
- **Sequence**: S = (p1, …, pN), each pi ∈ {H, P}  
- **Backbone adjacency** (neighboring monomers must sit on adjacent grid points):  

if abs(i - j) == 1:
    assert abs(x[i] - x[j]) + abs(y[i] - y[j]) == 1

- **Self‑avoidance** (no two monomers occupy the same site): 

if i != j:
    assert (x[i], y[i]) != (x[j], y[j])

- **Energy of conformation C** (minus number of non‑bonded H–H contacts):  

E = 0
for i in range(N):
    for j in range(i+1, N):
        if abs(i - j) > 1 and p[i] == 'H' and p[j] == 'H':
            if abs(x[i] - x[j]) + abs(y[i] - y[j]) == 1:
                E -= 1

  The agent seeks to **minimize** E(C).

**2. MDP Formulation**

- **State** `s_t`: current partial SAW positions + residue types.  
- **Action** `a_t ∈ {L, F, R}`: turn Left, Forward, or Right to place the next monomer.  
- **Transition** enforces the adjacency and self‑avoidance rules above.  
- **Reward** (sparse—only at end of walk):  

  **if t = T (i.e. all N placed) then
    r_t = |E(s_T)|
  else
    r_t = 0**


**3. Deep Q‑Network (DQN)**

Approximate the optimal value Q*(s,a) using a neural net Q(s,a;θ).

1. **Bellman update**  

   **Q*(s, a) = E[ r_{t+1}
                  + γ * max_{a'} Q*(s_{t+1}, a')
                | s_t = s, a_t = a ]**

2. **Target network** Q̂(s,a;θ̂) is a periodic copy of Q(s,a;θ) to stabilize training.  
3. **TD‑error**  

   **δ = (r + γ * max_{a'} Q̂(s',a';θ̂)) – Q(s,a;θ)**

4. **Huber loss** on δ (average over a minibatch B):  

   **L(δ) = { 0.5*δ²      if |δ| ≤ 1
            |δ| – 0.5  otherwise }
   loss = (1/|B|) * ∑_{(s,a,r,s')∈B} L(δ)**


**4. Exploration vs. Exploitation**

- **ε‑greedy policy**  

  with prob. ε: pick a_t uniformly at random  
  else       : pick a_t = argmax_a Q(s_t, a; θ)

- **Decay schedule for ε over episode index i**  

 **ε_i = ε_min
        + (ε_max – ε_min) * exp( – i * λ / total_episodes )**

  where  

  **ε_max = 1.0,  ε_min = 0.01,  λ = 5**


**5. State Encoding & Network**

- **One‑hot encoding**: represent the length‑N sequence of (action_so_far + residue_type) as an N×6 binary array.  
- **LSTM‑DQN**: stack of LSTM layers + final fully connected head → 3 real‑valued Q outputs.  
  - For N ≤ 36: use 2 LSTM layers of 256 units each.  
  - For N > 36: use 3 LSTM layers of 512 units each.

**6. Training Hyperparameters**

| Parameter              | Value                       |
|------------------------|-----------------------------|
| Discount factor γ      | 0.98                        |
| Replay memory size     | min(50 000, total_eps/10)   |
| Minibatch size         | 32                          |
| Target update period   | every 100 gradient steps    |
| Optimizer              | Adam, learning rate = 0.0005|
| Episodes per sequence  | 100 K–600 K (↑ with N)      |

**7. Training Loop**

```
Initialize Q(θ), Q̂(θ̂), replay buffer D.
for episode = 1 to total_episodes:
  ε = compute_epsilon(episode)
  Reset env → state s_0
  for t = 0 to N–1:
    choose action a_t via ε‑greedy on Q(s_t, ·; θ)
    step → observe (s_{t+1}, r_{t+1})
    store (s_t, a_t, r_{t+1}, s_{t+1}) in D
    sample minibatch B from D
    compute TD error δ and Huber loss over B
    update θ via Adam gradient step
    every 100 steps: set θ̂ ← θ
    s_t ← s_{t+1}
  endfor
endfor
```

**Improvements Done in the Model**

1. **Electrostatic Potential**  
   - Add a **distance cutoff** `r_c` to ignore far‐apart pairs for speed.  
   - Include a **dielectric constant** ε_r:  

     **E_elec = k_e * q_i * q_j / (ε_r * r_ij)**

   - Allow **variable charges** (e.g. partial charges) rather than only {0, –1}.  

2. **Van der Waals (Lennard–Jones)**  
   - Use a **shifted or truncated LJ** so that `E_LJ(r_c) = 0` for better stability.  
   - Maintain a **Verlet neighbor list** to reduce O(N²) to O(N).  
   - Exclude both **i±1** and **i±2** bonded interactions to avoid overlaps.

3. **Reward Function**  
   - Combine terms with weights:  

**r = α * |E_HP|  +  β * E_elec_total  +  γ * E_LJ_total**

   - **Normalize** each energy term so none dominates.  
   - Give **small stepwise rewards/penalties** (e.g. for collisions) to smooth the learning signal.

4. **Training & Logging**  
   - Implement **early stopping** when validation energy plateaus over M episodes.  
   - Log to **TensorBoard**: rewards, losses, ε, and sample folded‐path videos.  
   - Run **hyperparameter sweeps** (learning rate, batch size, γ, network size).  
   - Fix all random seeds (Python/Numpy/PyTorch) and report mean ± std over ≥ 5 runs.

![training](https://github.com/VoHunMain/Creativity_CoSY_Lab/blob/main/readme_images2/moving_avg-500.png?raw=true)

The above image provides a visual summary of the learning progress by smoothing the episode rewards over a window of 500 episodes, helping track whether the agent’s performance is improving steadily over time. 

**X-Axis (Episode Index)**: The x-axis represents the training episodes, usually scaled in thousands (K) if using a large number of episodes.

**Y-Axis (Moving Average of Rewards)**: The y-axis shows the average reward computed over the past 500 episodes. Since the rewards are negative (energy minimization), a more negative moving average indicates that the agent has found lower-energy (better) conformations.

5. **Moving‑Average Plot**  
   - **Window size**: experiment with shorter (100) vs. longer (1000) windows to balance smoothness vs. responsiveness.  
   - **Variance shading**: plot ± one standard deviation around the moving average for stability insight.  
   - **Alternative metric**: show min/median/max rewards in each window to highlight outlier successes.

[^13]: Cornell, W. D. et al. (1995). *JACS*, 117(19), 5179–5197. [DOI](https://doi.org/10.1021/ja00124a002) (AMBER)  
[^14]: Engh, R. A. & Huber, R. (1991). *Acta Cryst.*, A47(4), 392–400. [DOI](https://doi.org/10.1107/S0108767391001071)  
[^15]: Chen, V. B. et al. (2010). *Acta Cryst.*, D66(1), 12–21. [DOI](https://doi.org/10.1107/S0907444909042073) (MolProbity)  
[^16]: Baker, E. N. & Hubbard, R. E. (1984). *Prog. Biophys. Mol. Biol.*, 44(2), 97–179. [DOI](https://doi.org/10.1016/0079-6107(84)90007-3)  
[^17]: Zhang, Y. & Skolnick, J. (2004). *Proteins*, 57(4), 702–710. [DOI](https://doi.org/10.1002/prot.20264) (TM-score)  
[^18]: Pathak, D. et al. (2017). *ICML*. [arXiv](https://arxiv.org/abs/1705.05363) (Curiosity-driven RL)  

---

## Advice 2

## Biological Concepts Implementation

Protein folding is the process by which a linear sequence of amino acids assumes its native three-dimensional structure, which is essential for biological function. Traditional computational approaches like molecular dynamics simulations are computationally intensive and may struggle with the vast conformational space. Reinforcement learning offers a promising alternative by allowing an agent to learn optimal folding pathways through trial and error.

**Key Components**

**1. Force Fields in Protein Folding**

Force fields are mathematical models used in molecular dynamics simulations to describe the potential energy of a molecular system as a function of atomic positions. Two prominent force fields are analyzed:

**AMBER**
- Developed by Peter Kollman's group at UCSF
- Includes parameter sets like ff99 for peptides and proteins, GAFF for small organic molecules
- Uses RESP (Restricted Electrostatic Potential) method to fit charges
- Often performs better for DNA simulations

**CHARMM**
- Developed by Martin Karplus's group at Harvard
- Includes parameter sets like CHARMM22 (all-atom proteins) and CHARMM27 (DNA, RNA, lipids)
- Uses scaled Mulliken charges with focus on experimental validation
- Often performs better for protein simulations

The total potential energy calculation in force fields typically includes:

```
E = E_bonded + E_non-bonded
```

where:
- `E_bonded = E_bonds + E_angles + E_dihedrals`
- `E_non-bonded = E_vdW + E_electrostatic`

For reinforcement learning, this energy function can be used to create a reward function: `r = -E`, encouraging the RL agent to find stable (low-energy) protein conformations.

**2. Dihedral Angles and Protein Structure**

Dihedral angles, such as φ (phi) and ψ (psi), are critical internal angles that describe protein backbone conformations:
- φ is defined by C_(i-1) - N_i - C_(αi) - C_i
- ψ is defined by N_i - C_(αi) - C_i - N_(i+1)

In force fields, dihedral angles are scored using the potential energy function:

```
V(φ) = Σ(1/2)V_n(1 + cos(nφ - γ_n))
```

where V_n is the energy barrier, n is the multiplicity, and γ_n is the phase shift.

**3. Reinforcement Learning Approach: DeepFoldit**

DeepFoldit applies reinforcement learning to protein folding by formulating it as a Markov Decision Process (MDP):

**State Representation**
- Images of protein structures (160x160 pixels)
- Preprocessed using log-polar transformation for rotation and scaling invariance:
  ```
  ρ = log(√((x - x_c)² + (y - y_c)²))
  θ = arctan((y - y_c)/(x - x_c))
  ```

**Action Space**
- 15 discrete moves (e.g., "Up," "Down," "Left," "Right," "Zoom In," "Zoom Out")
- Implemented via tools like xdotool for interaction with Foldit Standalone

**Reward Function**
- Based on improvement in folding score calculated by the Rosetta algorithm:
  ```
  r = score(s_t) - score(s_t+1)
  ```
- Typically scaled (e.g., by a factor of 0.175)

**Neural Network Architecture**
- Input: 160x160 RGB image
- Convolutional Layers: 2-3 layers with ReLU activations
- Fully Connected Layers: One or more hidden layers (e.g., 256 neurons)
- Output Layer: 15 neurons (one per discrete action)

**Training Methodology**
- Deep Q-Network (DQN) with Q-learning update:
  ```
  Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
  ```
- Experience replay with memory size of 8000
- ε-greedy exploration strategy (decaying from 1.0 to 0.1)
- Learning rate: ~1×10⁻⁴
- Discount factor: ~0.95

**Experimental Results**

DeepFoldit was evaluated on 40 small proteins from the Protein Data Bank (PDB):
- 20 proteins for training, 20 for testing
- After ~30,000 training steps:
  - Training set improvement: ~5.8%
  - Test set improvement: ~4.5%

These results suggest the model can generalize to unseen protein structures.

**Optimized Hyperparameters**

| Hyperparameter | Optimal Value |
|----------------|---------------|
| First Conv Layer Depth | 30 |
| First Layer Receptive Field | 15 |
| First Layer Stride | 1 |
| Second Conv Layer Depth | 15 |
| Second Layer Receptive Field | 20 |
| Second Layer Stride | 1 |
| Dense Layer Size | 200 |
| Replay Memory Size | 8000 |
| Learning Rate | 1×10⁻⁴ |
| Momentum | 0.95 |
| Decay | 2×10⁻⁶ |
| Input Image Size | 160 |
| Score Scaling | 0.175 |
| Discount Factor (γ) | 0.95 |
| Mini Batch Size | 50 |
| Initial Epsilon | 0.25 |
| Final Epsilon | 0.05 |

**Proposed Novel Architecture**

We propose a novel model architecture integrating:
1. Reinforcement learning for decision-making
2. Graph Neural Networks (GNNs) for capturing the relational structure of proteins
3. Long Short-Term Memory (LSTM) networks for sequential aspects of protein folding

This hybrid approach aims to better capture both the sequential and relational nature of protein folding.

**Future Directions**

1. **Extended Training**: Increase training durations to millions of timesteps
2. **Continuous Action Space**: Move from discrete to continuous action spaces for finer control (possibly using DDPG)
3. **Integration with Physics-Based Models**: Combine RL with AMBER or CHARMM force fields for better energy estimation
4. **Hybrid Approaches**: Combine reinforcement learning with supervised methods (similar to AlphaFold)
5. **Partial Charges Optimization**: Incorporate amino acid partial charges for more accurate electrostatic interaction modeling

**References**

[^19]: Smith et al., Neural Network Energy Functions for Protein Folding, 2020. [DOI](https://doi.org/10.1021/acs.jctc.2c00069)
[^20]: Lee et al., Machine Learning in Protein Structure Prediction, 2021. [DOI](https://doi.org/10.1016/j.cbpa.2021.04.005)
[^21]: Jones and Wang, Comparison of AMBER and CHARMM for Molecular Simulations, Journal of Computational Chemistry, 2019. [DOI](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.sciencedirect.com/science/article/abs/pii/S0950061819330296&ved=2ahUKEwil_qn2-d-MAxXKZSoJHVdWO9YQFnoECBcQAQ&usg=AOvVaw0WYuQACoERgZQD9ZxMrLcx)
[^22]: DeepFoldit – A Deep Reinforcement Learning Neural Network Folding Proteins. [DOI](https://arxiv.org/abs/2011.03442)
[^23]: Jumper, J. et al., Highly Accurate Protein Structure Prediction with AlphaFold. [DOI](https://www.nature.com/articles/s41586-021-03819-2)
[^24]: Proteopedia: Dihedral Angles. [DOI](https://proteopedia.org/wiki/index.php/Dihedral_angles)
[^25]: Schmid et al., Optimization of Dihedral Angle Parameters in GROMOS, JCTC, 2011. [DOI](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://abstracts.boku.ac.at/download.php%3Fdataset_id%3D19374%26property_id%3D107%26role_id%3DNONE&ved=2ahUKEwiw_-jc-d-MAxUl3TgGHUK7FTgQFnoECBYQAQ&usg=AOvVaw0aVeUJuQ15aaIx5kjh0Uag)
[^26]: Foldit: An Online Puzzle Game for Protein Structure Prediction. [DOI](https://fold.it/portal/)
[^27]: The Protein Data Bank. [DOI](https://www.rcsb.org/)
[^28]: The Amber Biomolecular Simulation Programs. [DOI](https://onlinelibrary.wiley.com/doi/10.1002/jcc.20253)






