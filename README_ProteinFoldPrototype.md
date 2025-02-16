# Composite Reward Function for RL-Based Protein Folding

**Author:** Your Name  
**Date:** \today

This document describes a composite reward function for a reinforcement learning (RL) agent applied to protein folding. The overall reward is given as a weighted sum of several components, each reflecting a biophysical or evolutionary criterion that guides the system toward a native-like folded structure. In general, we write the total reward as:

$$
R(s,a) = \sum_{k \in \{c,\, cl,\, b,\, r,\, hb,\, hp,\, e,\, t,\, p,\, d,\, rec,\, int\}} w_k \, R_k(s,a)
$$

where:
- **\(s\)** is the current state (e.g., the positions, bond lengths, torsion angles, etc.),
- **\(a\)** is the action taken,
- **\(w_k\)** are positive hyperparameters (weights),
- **\(R_k(s,a)\)** are the individual reward (or penalty) components.

Below, each reward component is described in detail.

---

## 1. Contact Map Agreement \(R_c(s)\)

### Mathematical Formula
$$
R_c(s) = \sum_{i,j} \mathbb{1}\{d_{ij}(s) \le \tau\} \cdot \mathbb{1}\{d_{ij}^{AF} \le \tau\}
$$
where:
- \(d_{ij}(s)\) is the Euclidean distance between residues \(i\) and \(j\) in the current state \(s\),
- \(d_{ij}^{AF}\) is the corresponding predicted contact (or binary indicator) from AlphaFold,
- \(\tau\) is a distance threshold (typically around 8\,\AA),
- \(\mathbb{1}\{\cdot\}\) is the indicator function.

### Intuition and Rationale
The contact map agreement term is central to integrating evolutionary information into the folding process. In many protein structure prediction methods, evolutionary data—extracted from multiple sequence alignments (MSAs)—is used to infer which pairs of residues are likely to be in close proximity. Residue contacts that are conserved across homologous proteins indicate critical interactions that maintain the protein's structure.

By comparing the contacts observed in the current conformation \(s\) with those predicted by a state-of-the-art model such as AlphaFold, the RL agent is rewarded for producing structures that satisfy these evolutionary constraints. This term acts as a guidepost: if the conformation respects the “blueprint” provided by evolutionary couplings, it is likely to be closer to a native fold.

Moreover, the use of the indicator function ensures that only pairs within a threshold distance contribute to the reward. This discrete measure simplifies the reward calculation and reinforces the idea that, beyond a certain distance, interactions are not significant. In practice, this term helps the RL agent focus on the global topology and long-range interactions that are crucial for proper folding.

---

## 2. Steric Clash Penalty \(R_{cl}(s)\)

### Mathematical Formula
$$
R_{cl}(s) = -\sum_{i<j} \mathbb{1}\{d_{ij}(s) < d_{\min}\}
$$
where \(d_{\min}\) is the minimum allowable distance between nonbonded atoms.

### Intuition and Rationale
Proteins are physical objects that obey the laws of steric hindrance: atoms cannot occupy the same space. Steric clashes occur when atoms come too close, violating physical constraints and indicating an unrealistic or strained conformation. This penalty term explicitly discourages such clashes by subtracting a fixed penalty for each pair of atoms whose distance falls below a predefined threshold \(d_{\min}\).

In essence, this term enforces a form of spatial exclusion and ensures that the RL agent learns to produce conformations that are physically plausible. Without such a penalty, the agent might converge on solutions that are energetically favorable in the abstract (e.g., many contacts) but are physically impossible because of overlapping atoms.

The simplicity of the indicator function in this formulation provides a clear, binary signal: if a clash is detected, a penalty is applied; if not, no penalty is incurred. This direct feedback is crucial in the early stages of training when the agent is learning to respect the basic rules of molecular geometry.

---

## 3. Bond Geometry Reward \(R_b(s)\)

### Mathematical Formula
$$
R_b(s) = -\frac{1}{N-1}\sum_{i=1}^{N-1} \Bigl( \|x_{i+1} - x_i\| - L_{\mathrm{ideal}} \Bigr)^2
$$
where:
- \(x_i\) is the position of residue \(i\),
- \(N\) is the total number of residues,
- \(L_{\mathrm{ideal}}\) is the ideal bond length (approximately 3.8\,\AA\ for backbone \(\mathrm{C}\alpha\) atoms).

### Intuition and Rationale
Proper bond geometry is a fundamental requirement for a physically realistic protein structure. The bond lengths between consecutive residues should remain close to an ideal value, which is derived from empirical observations of protein structures. Deviations from this ideal bond length are energetically unfavorable and can lead to unrealistic structures.

The reward term \(R_b(s)\) is defined as the negative mean squared deviation of the observed bond lengths from the ideal bond length. This quadratic penalty means that small deviations incur only a minor penalty, whereas larger deviations are heavily penalized. Such a formulation not only enforces local geometric consistency but also helps the agent to avoid accumulating small errors that could distort the overall structure.

By including this term, the RL agent is encouraged to maintain the correct spacing between residues, ensuring that the backbone is built accurately. This, in turn, provides a reliable scaffold on which long-range interactions and higher-order structures can be formed.

---

## 4. Ramachandran (Torsion Angle) Reward \(R_r(s)\)

### Mathematical Formula
$$
R_r(s) = \sum_{i=1}^{N} \log P_{\mathrm{rama}}(\phi_i, \psi_i)
$$
where \(P_{\mathrm{rama}}(\phi_i, \psi_i)\) is the empirical probability of the dihedral angles \((\phi_i, \psi_i)\) based on the Ramachandran distribution.

### Intuition and Rationale
The backbone conformation of a protein is largely determined by its dihedral angles, \(\phi\) and \(\psi\), which are subject to steric and electronic constraints. The Ramachandran plot provides a statistical distribution of these angles in known protein structures, highlighting regions of high probability (favorable conformations) and regions that are rarely observed.

By taking the logarithm of the probability \(P_{\mathrm{rama}}(\phi_i, \psi_i)\) for each residue, the reward term \(R_r(s)\) assigns a higher (less negative) value to conformations with torsion angles in the favored regions. In contrast, conformations with angles outside these regions receive a lower reward. The logarithm is used because probabilities are typically small and the log function transforms multiplicative effects into additive ones, making the overall reward easier to interpret and combine.

This term is crucial for guiding the RL agent not only to achieve a correct global fold but also to maintain realistic local backbone conformations. It ensures that even if the overall topology is correct, the detailed local structure is also physically plausible.

---

## 5. Hydrogen Bond Reward \(R_{hb}(s)\)

### Mathematical Formula
$$
R_{hb}(s) = \sum_{(i,j)\in \mathcal{HB}} h_{ij}(s)
$$
where:
- \(\mathcal{HB}\) is the set of residue pairs capable of forming hydrogen bonds,
- \(h_{ij}(s)\) quantifies the quality or presence (often binary or weighted) of a hydrogen bond between residues \(i\) and \(j\).

### Intuition and Rationale
Hydrogen bonds are critical for stabilizing secondary structures like \(\alpha\)-helices and \(\beta\)-sheets in proteins. They represent directional, non-covalent interactions that contribute substantially to the overall stability of the protein fold. In our reward function, the hydrogen bond term \(R_{hb}(s)\) rewards conformations where appropriate hydrogen bonds are formed.

The formulation sums contributions from all potential hydrogen-bonding pairs. In practice, the function \(h_{ij}(s)\) may be designed to return a value of 1 when a bond is properly formed (based on criteria such as distance and angle) and 0 otherwise, or it may return a continuous score representing the strength of the interaction.

This reward guides the RL agent toward conformations that not only have a favorable global topology but also exhibit the correct local interactions. By encouraging the formation of hydrogen bonds, the model is nudged toward biologically realistic structures, reflecting the well-established principles of protein chemistry.

---

## 6. Hydrophobic Packing Reward \(R_{hp}(s)\)

### Mathematical Formula
$$
R_{hp}(s) = -\frac{1}{|\mathcal{H}|}\sum_{i\in \mathcal{H}} \mathrm{SASA}(x_i)
$$
where:
- \(\mathcal{H}\) is the set of hydrophobic residues,
- \(\mathrm{SASA}(x_i)\) is the solvent-accessible surface area of residue \(i\).

### Intuition and Rationale
A hallmark of protein folding is the tendency of hydrophobic (water-repelling) residues to be buried in the protein’s core. This effect minimizes the exposure of these residues to the aqueous environment and is a key driving force in the folding process.

The hydrophobic packing reward term \(R_{hp}(s)\) is designed to capture this phenomenon by penalizing high solvent-accessible surface area (SASA) for hydrophobic residues. The idea is that if hydrophobic residues are well buried, their SASA will be low, leading to a less negative (or higher) reward.

By averaging the SASA over all hydrophobic residues, we obtain a measure of the overall “packedness” of the hydrophobic core. Lower values indicate a more tightly packed, and thus more stable, core structure. This term therefore guides the RL agent toward conformations where the hydrophobic effect is properly realized—a fundamental aspect of native protein structure.

---

## 7. Electrostatic Energy Penalty \(R_e(s)\)

### Mathematical Formula
$$
R_e(s) = -\sum_{i<j} \frac{q_i\,q_j}{\|x_i - x_j\| + \epsilon}
$$
where:
- \(q_i\) is the charge of residue \(i\),
- \(\|x_i - x_j\|\) is the Euclidean distance between residues \(i\) and \(j\),
- \(\epsilon\) is a small constant to avoid division by zero.

### Intuition and Rationale
Electrostatic interactions—both attractive and repulsive—play a significant role in stabilizing or destabilizing protein structures. The term \(R_e(s)\) captures these effects by computing a pairwise sum of Coulomb-like interactions between residues.

In this formulation, residues with like charges (both positive or both negative) will contribute a positive value (since \(q_i\,q_j > 0\)), which, when negated, leads to a penalty. Conversely, opposite charges yield a negative product and thus a favorable contribution. The division by the distance ensures that these interactions are stronger when residues are closer together.

The constant \(\epsilon\) guarantees numerical stability. This energy-based term ensures that the RL agent considers long-range electrostatic interactions, which are essential for maintaining the correct tertiary structure of proteins. Overall, it helps the agent avoid configurations that would result in unrealistically high repulsion or insufficient attraction.

---

## 8. Template Similarity Reward \(R_t(s)\)

### Mathematical Formula
One common formulation is:
$$
R_t(s) = \frac{1}{\mathrm{RMSD}(s, s^{\mathrm{template}}) + \epsilon}
$$
Alternatively, one might use a TM-score that scales between 0 and 1.

### Intuition and Rationale
When a structural template is available—i.e., a known structure from a homologous protein—the RL agent can be guided toward conformations that resemble this template. Structural similarity is often measured by the root-mean-square deviation (RMSD) between the predicted structure \(s\) and the template \(s^{\mathrm{template}}\).

Using the inverse RMSD as the reward means that smaller deviations (i.e., higher similarity) yield larger rewards. This term encourages the agent to produce folds that are consistent with known structural motifs and homologous models. Even if a template is not strictly available for all residues, regions with high similarity provide strong signals that help ensure the overall fold is biologically plausible.

---

## 9. pLDDT-Weighted Conformational Accuracy Reward \(R_p(s)\)

### Mathematical Formula
$$
R_p(s) = \frac{1}{\frac{1}{N}\sum_{i=1}^{N} p_i\, d(x_i, x_i^{\mathrm{true}}) + \epsilon}
$$
where:
- \(p_i\) is the pLDDT score (confidence) for residue \(i\),
- \(d(x_i, x_i^{\mathrm{true}})\) is the deviation between the predicted and true positions,
- \(N\) is the number of residues,
- \(\epsilon\) is a small constant.

### Intuition and Rationale
The pLDDT (predicted Local Distance Difference Test) score is a measure of per-residue confidence output by models like AlphaFold. It reflects the reliability of the predicted position for each residue. In regions where the model is highly confident, even small deviations from the true structure are significant.

By weighting the positional deviation by the pLDDT score and then taking the inverse, \(R_p(s)\) rewards conformations that are close to the true structure, particularly in regions of high confidence. This encourages the RL agent to achieve a low overall RMSD while focusing on critical regions where accurate prediction is most important. This term is especially valuable when experimental data or high-confidence predictions are available as a benchmark.

---

## 10. Distogram Consistency Reward \(R_d(s)\)

### Mathematical Formula
$$
R_d(s) = -D_{KL}\Bigl(P_{\mathrm{agent}}(\cdot\,|\,s) \,\Big\|\, P_{AF}(\cdot)\Bigr)
$$
where \(D_{KL}\) denotes the Kullback--Leibler divergence between the agent’s observed inter-residue distance distribution \(P_{\mathrm{agent}}(\cdot\,|\,s)\) and AlphaFold's predicted distogram \(P_{AF}(\cdot)\).

### Intuition and Rationale
AlphaFold predicts a distogram—a probability distribution over possible inter-residue distances—based on both evolutionary and structural features. The distogram consistency reward encourages the RL agent to produce a conformation whose inter-residue distance distribution aligns with this prediction.

By computing the KL divergence between the distribution derived from the agent's conformation and the target distogram, we obtain a measure of similarity. A lower divergence indicates better agreement. The negative sign ensures that lower divergence results in a higher reward. This term integrates the probabilistic insights from AlphaFold into the RL framework, guiding the exploration of conformational space based on statistical trends observed in large sequence alignments.

---

## 11. Recycling Consistency Reward \(R_{rec}(s)\)

### Mathematical Formula
$$
R_{rec}(s) = -\frac{1}{N}\sum_{i=1}^{N} \|x_i^{(t)} - x_i^{(t-1)}\|^2
$$
where \(x_i^{(t)}\) and \(x_i^{(t-1)}\) are the positions of residue \(i\) at the current and previous recycling iterations, respectively.

### Intuition and Rationale
In many protein structure prediction pipelines, such as AlphaFold, iterative refinement (recycling) is used to improve the conformation progressively. A well-behaved recycling process shows diminishing changes between iterations as the structure converges toward a stable state.

The recycling consistency reward \(R_{rec}(s)\) penalizes large changes between consecutive iterations. When the differences \(\|x_i^{(t)} - x_i^{(t-1)}\|\) are small, it indicates convergence toward a stable, high-quality structure. This term thus acts as a convergence measure, encouraging the RL agent to refine the structure until further adjustments yield only marginal improvements.

---

## 12. Intrinsic (Curiosity) Reward \(R_{int}(s,a)\)

### Mathematical Formula
$$
R_{int}(s,a) = \| f(s) - \hat{f}(s,a) \|^2
$$
where:
- \(f(s)\) is the predicted next state from a learned forward model,
- \(\hat{f}(s,a)\) is the observed next state after taking action \(a\).

### Intuition and Rationale
In reinforcement learning, intrinsic rewards are used to encourage exploration, particularly in environments where extrinsic rewards are sparse or delayed. The intrinsic (or curiosity) reward in this formulation measures the prediction error of a forward model. A large error suggests that the state resulting from action \(a\) is novel or that the dynamics in that region of the state space are not well understood.

By rewarding high prediction error, the RL agent is incentivized to explore such regions, which is crucial in the complex landscape of protein folding. This term helps prevent the agent from prematurely converging on suboptimal solutions and promotes the discovery of new, potentially better conformations.

---

## Composite Reward Summary

The overall reward function is defined as:
$$
R(s,a) = w_c\, R_c(s) + w_{cl}\, R_{cl}(s) + w_b\, R_b(s) + w_r\, R_r(s) + w_{hb}\, R_{hb}(s) + w_{hp}\, R_{hp}(s) + w_e\, R_e(s) + w_t\, R_t(s) + w_p\, R_p(s) + w_d\, R_d(s) + w_{rec}\, R_{rec}(s) + w_{int}\, R_{int}(s,a)
$$
Each component is tuned via its corresponding weight \(w_k\) to balance its influence on the final reward. These components collectively guide the RL agent to generate protein conformations that are physically realistic, evolutionarily consistent, and optimized according to known biophysical principles.
