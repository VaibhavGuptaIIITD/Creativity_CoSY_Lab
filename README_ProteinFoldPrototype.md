# Composite Reward Function for RL-Based Protein Folding

**Author:** Your Name  
**Date:** [Insert Date]

This document describes a composite reward function for a reinforcement learning (RL) agent applied to protein folding. The overall reward is given as a weighted sum of several components, each reflecting a biophysical or evolutionary criterion that guides the system toward a native-like folded structure. In general, we write the total reward as:

$$
R(s,a) = \sum_{k \in \{c,\, cl,\, b,\, r,\, hb,\, hp,\, e,\, t,\, p,\, d,\, rec,\, int\}} w_k \, R_k(s,a)
$$

where:
- **\(s\)** is the current state (e.g., positions, bond lengths, torsion angles, etc.),
- **\(a\)** is the action taken,
- **\(w_k\)** are positive hyperparameters (weights),
- **\(R_k(s,a)\)** are the individual reward (or penalty) components.

Below, each reward component is described in detail.

---

## 1. Contact Map Agreement \(R_c(s)\)

### Mathematical Formula

$$
R_c(s) = \sum_{i,j} \mathbf{1}\{ d_{ij}(s) \le \tau \} \cdot \mathbf{1}\{ d_{ij}^{AF} \le \tau \}
$$

where:
- \(d_{ij}(s)\) is the Euclidean distance between residues \(i\) and \(j\) in the current state \(s\),
- \(d_{ij}^{AF}\) is the predicted contact (or binary indicator) from AlphaFold,
- \(\tau\) is a distance threshold (typically around 8\,\AA),
- \(\mathbf{1}\{\cdot\}\) is the indicator function (equal to 1 if the condition holds, 0 otherwise).

### Intuition and Rationale

This term rewards the RL agent for producing conformations in which residue pairs predicted to be in contact actually are close in space. Evolutionary data from multiple sequence alignments suggest which residues are likely to interact. By matching the agent’s contacts to these predictions, the agent is guided toward a fold that respects the evolutionary “blueprint.”

---

## 2. Steric Clash Penalty \(R_{cl}(s)\)

### Mathematical Formula

$$
R_{cl}(s) = -\sum_{i<j} \mathbf{1}\{ d_{ij}(s) < d_{\min} \
$$

where:
- \(d_{ij}(s)\) is the Euclidean distance between residues \(i\) and \(j\),
- \(d_{\min}\) is the minimum allowable distance between nonbonded atoms,
- \(\mathbf{1}\{\cdot\}\) is the indicator function.

### Intuition and Rationale

Steric clashes occur when atoms are too close, resulting in unrealistic, physically strained conformations. This term imposes a fixed penalty for every pair of atoms violating the minimum distance \(d_{\min}\). Such a binary penalty enforces spatial exclusion and ensures that the RL agent learns to generate physically plausible structures.

---

## 3. Bond Geometry Reward \(R_b(s)\)

### Mathematical Formula

$$
R_b(s) = -\frac{1}{N-1} \sum_{i=1}^{N-1} \Bigl( \|x_{i+1} - x_i\| - L_{\mathrm{ideal}} \Bigr)^2
$$

where:
- \(x_i\) is the position of residue \(i\),
- \(N\) is the total number of residues,
- \(L_{\mathrm{ideal}}\) is the ideal bond length (approximately 3.8\,\AA\ for backbone \(\mathrm{C}\alpha\) atoms).

### Intuition and Rationale

Maintaining ideal bond lengths is essential for accurate protein geometry. This term penalizes deviations from the ideal bond length quadratically, ensuring that even small deviations are corrected and larger errors are heavily penalized. It helps the agent maintain a correct backbone structure, which is crucial for building higher-order structures.

---

## 4. Ramachandran (Torsion Angle) Reward \(R_r(s)\)

### Mathematical Formula

$$
R_r(s) = \sum_{i=1}^{N} \log P_{\mathrm{rama}}(\phi_i, \psi_i)
$$

where \(P_{\mathrm{rama}}(\phi_i, \psi_i)\) is the empirical probability (from the Ramachandran distribution) of the dihedral angles \((\phi_i, \psi_i)\).

### Intuition and Rationale

Backbone dihedral angles determine the local conformation of proteins. The Ramachandran plot indicates regions of favorable angles. By summing the log-probabilities, the agent is rewarded for adopting torsion angles common in native proteins. This term helps ensure that local geometries are realistic.

---

## 5. Hydrogen Bond Reward \(R_{hb}(s)\)

### Mathematical Formula

$$
R_{hb}(s) = \sum_{(i,j) \in \mathcal{HB}} h_{ij}(s)
$$

where:
- \(\mathcal{HB}\) is the set of residue pairs capable of forming hydrogen bonds,
- \(h_{ij}(s)\) quantifies the presence or quality of a hydrogen bond between residues \(i\) and \(j\).

### Intuition and Rationale

Hydrogen bonds stabilize secondary structures like \(\alpha\)-helices and \(\beta\)-sheets. Rewarding their formation encourages the agent to create locally stable, biologically realistic structures. This term directs the agent toward conformations where these critical interactions are present.

---

## 6. Hydrophobic Packing Reward \(R_{hp}(s)\)

### Mathematical Formula

$$
R_{hp}(s) = -\frac{1}{|\mathcal{H}|} \sum_{i \in \mathcal{H}} \mathrm{SASA}(x_i)
$$

where:
- \(\mathcal{H}\) is the set of hydrophobic residues,
- \(\mathrm{SASA}(x_i)\) is the solvent-accessible surface area of residue \(i\).

### Intuition and Rationale

Hydrophobic residues are typically buried in the protein core. This term penalizes high solvent-accessible surface area for hydrophobic residues, thereby rewarding conformations where these residues are well-packed. A well-formed hydrophobic core is crucial for the overall stability of the protein.

---

## 7. Electrostatic Energy Penalty \(R_e(s)\)

### Mathematical Formula

$$
R_e(s) = -\sum_{i<j} \frac{q_i\,q_j}{\left\| x_i - x_j \right\| + \epsilon}
$$

where:
- \(q_i\) is the charge of residue \(i\),
- \(\left\| x_i - x_j \right\|\) is the Euclidean distance between residues \(i\) and \(j\),
- \(\epsilon\) is a small constant to avoid division by zero.

### Intuition and Rationale

Electrostatic interactions are key to protein stability. This term calculates a Coulomb-like interaction between residue pairs. Residues with like charges produce a penalty, while those with opposite charges provide a favorable contribution. The division by the distance emphasizes that interactions are stronger at close range. This term guides the agent to consider the effects of charge interactions when optimizing the structure.

---

## 8. Template Similarity Reward \(R_t(s)\)

### Mathematical Formula

One common formulation is:

$$
R_t(s) = \frac{1}{\mathrm{RMSD}(s, s^{\mathrm{template}}) + \epsilon}
$$

Alternatively, a TM-score (ranging from 0 to 1) could be used.

### Intuition and Rationale

When a structural template is available, this term rewards the agent for producing a conformation similar to the template. A lower RMSD (or higher TM-score) means that the predicted structure is close to a known homologous structure. This leverages known structural information to guide the folding process toward biologically plausible conformations.

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

The pLDDT score indicates the confidence in the predicted positions of residues. By weighting the positional error with pLDDT and taking the inverse, this term rewards conformations that are closer to the true structure, especially in high-confidence regions. It helps ensure that the agent focuses on achieving high accuracy where it matters most.

---

## 10. Distogram Consistency Reward \(R_d(s)\)

### Mathematical Formula

$$
R_d(s) = -D_{KL}\Bigl(P_{\mathrm{agent}}(\cdot\,|\,s) \,\Big\|\, P_{AF}(\cdot)\Bigr)
$$

where \(D_{KL}\) denotes the Kullback--Leibler divergence between the agent’s observed inter-residue distance distribution \(P_{\mathrm{agent}}(\cdot\,|\,s)\) and AlphaFold's predicted distogram \(P_{AF}(\cdot)\).

### Intuition and Rationale

AlphaFold predicts a distogram, a probability distribution over inter-residue distances, which encapsulates both evolutionary and structural information. This term penalizes the divergence between the distribution obtained from the current conformation and the predicted distogram. A lower divergence indicates better agreement, thus higher reward. This term helps align the RL agent’s predictions with statistical trends from large-scale sequence data.

---

## 11. Recycling Consistency Reward \(R_{rec}(s)\)

### Mathematical Formula

$$
R_{rec}(s) = -\frac{1}{N}\sum_{i=1}^{N} \|x_i^{(t)} - x_i^{(t-1)}\|^2
$$

where \(x_i^{(t)}\) and \(x_i^{(t-1)}\) are the positions of residue \(i\) at the current and previous recycling iterations, respectively.

### Intuition and Rationale

Iterative refinement (or recycling) is used to progressively improve the conformation. This term penalizes large changes between successive iterations, encouraging the structure to converge to a stable state. When changes are minimal, it indicates that the structure is close to converging, which is desirable for a high-quality prediction.

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

Intrinsic rewards encourage exploration, especially when extrinsic rewards are sparse. Here, the RL agent is motivated to explore states where the forward model's prediction error is high. A large error suggests that the state is novel or not well understood, thereby incentivizing the agent to explore such regions in the conformational space. This is critical for avoiding premature convergence on suboptimal folds.

---

## Composite Reward Summary

The overall reward function is defined as:

$$
R(s,a) = w_c\, R_c(s) + w_{cl}\, R_{cl}(s) + w_b\, R_b(s) + w_r\, R_r(s) + w_{hb}\, R_{hb}(s) + w_{hp}\, R_{hp}(s) + w_e\, R_e(s) + w_t\, R_t(s) + w_p\, R_p(s) + w_d\, R_d(s) + w_{rec}\, R_{rec}(s) + w_{int}\, R_{int}(s,a)
$$

Each component is tuned via its corresponding weight \(w_k\) to balance its influence on the final reward. Collectively, these components guide the RL agent to generate protein conformations that are physically realistic, evolutionarily consistent, and optimized according to known biophysical principles.
