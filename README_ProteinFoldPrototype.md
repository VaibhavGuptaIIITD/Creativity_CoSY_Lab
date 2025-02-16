# Composite Reward Function for RL-Based Protein Folding

**Author:** Your Name  
**Date:** 2025-01-01

This document describes a composite reward function for a reinforcement learning (RL) agent applied to protein folding. The overall reward is given as a weighted sum of several components, each reflecting a biophysical or evolutionary criterion that guides the system toward a native-like folded structure. Formally:

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

### Mathematical Formula

$$
R_r(s) = \sum_{i=1}^N\,\log\,P_{\mathrm{rama}}\bigl(\phi_i,\,\psi_i\bigr),
$$

where \(P_{\mathrm{rama}}(\phi_i,\psi_i)\) is the probability of dihedral angles \(\phi_i,\psi_i\) (from Ramachandran distributions).

### Intuition and Rationale

Protein backbone angles are constrained by sterics and energetics. The Ramachandran plot indicates favored regions. Summing the log-probabilities rewards angles in these favored zones, promoting realistic local conformations.

---

## 5. Hydrogen Bond Reward \(R_{hb}(s)\)

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

### Mathematical Formula

$$
R_d(s) = -\,D_{\mathrm{KL}}\Bigl(P_{\mathrm{agent}}(\cdot \mid s)\,\Big\|\,P_{AF}(\cdot)\Bigr),
$$

where \(D_{\mathrm{KL}}\) is the Kullback--Leibler divergence between the agent’s distance distribution \(P_{\mathrm{agent}}\) and AlphaFold’s predicted distogram \(P_{AF}\).

### Intuition and Rationale

AlphaFold predicts a probability distribution over inter-residue distances (the distogram). This term penalizes divergence from that distribution. A lower KL divergence indicates better agreement with AlphaFold’s learned statistics, guiding the agent toward conformations that match evolutionary and structural patterns.

---

## 11. Recycling Consistency Reward \(R_{rec}(s)\)

### Mathematical Formula

$$
R_{rec}(s) = -\,\frac{1}{N}\,\sum_{i=1}^N \bigl\|\,x_i^{(t)} - x_i^{(t-1)}\bigr\|^2,
$$

where \(x_i^{(t)}\) and \(x_i^{(t-1)}\) are positions of residue \(i\) in consecutive recycling iterations.

### Intuition and Rationale

AlphaFold uses iterative refinement (“recycling”). A stable structure will change little between iterations. This term penalizes large changes, indicating that once the conformation is close to stable, further big adjustments are undesirable. This helps ensure convergence.

---

## 12. Intrinsic (Curiosity) Reward \(R_{int}(s,a)\)

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
