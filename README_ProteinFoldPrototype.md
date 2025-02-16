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

Proteins have specific residue–residue contacts that are evolutionarily conserved. Evolutionary data (from MSAs) suggest which residues are likely to be close together. This term rewards the agent when the conformation \(s\) exhibits contacts that match those predicted by AlphaFold. By enforcing that only pairs within the threshold \(\tau\) contribute, the term focuses on critical long-range interactions. Essentially, if the conformation respects these evolutionary “blueprints,” it is likely to be closer to the native fold.

---

## 2. Steric Clash Penalty \(R_{cl}(s)\)

### Mathematical Formula

$$
R_{cl}(s) = -\sum_{i<j} \mathbb{1}\{d_{ij}(s) < d_{\min}\}
$$

where \(d_{\min}\) is the minimum allowable distance between nonbonded atoms.

### Intuition and Rationale

Steric clashes occur when atoms come too close, violating physical constraints. This term penalizes such unrealistic overlaps. It enforces spatial exclusion by subtracting a penalty for every atom pair that violates the minimum distance \(d_{\min}\). This binary signal is crucial for teaching the agent to respect basic molecular geometry.

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

This term ensures that the bond lengths between consecutive residues remain near the ideal value. Deviations from \(L_{\mathrm{ideal}}\) are penalized quadratically, meaning larger errors incur disproportionately higher penalties. This helps maintain local geometric consistency and prevents the accumulation of errors that could distort the entire structure.

---

## 4. Ramachandran (Torsion Angle) Reward \(R_r(s)\)

### Mathematical Formula

$$
R_r(s) = \sum_{i=1}^{N} \log P_{\mathrm{rama}}(\phi_i, \psi_i)
$$

where \(P_{\mathrm{rama}}(\phi_i, \psi_i)\) is the empirical probability (from the Ramachandran distribution) of the dihedral angles \((\phi_i, \psi_i)\).

### Intuition and Rationale

The backbone dihedral angles \(\phi\) and \(\psi\) determine local protein conformation. The Ramachandran plot shows regions where these angles are statistically favorable. By summing the logarithm of these probabilities, the RL agent is rewarded for adopting favorable angles. This guides the agent toward realistic local conformations even if the overall fold is correct.

---

## 5. Hydrogen Bond Reward \(R_{hb}(s)\)

### Mathematical Formula

$$
R_{hb}(s) = \sum_{(i,j)\in \mathcal{HB}} h_{ij}(s)
$$

where:
- \(\mathcal{HB}\) is the set of residue pairs capable of forming hydrogen bonds,
- \(h_{ij}(s)\) quantifies the presence or quality of a hydrogen bond between residues \(i\) and \(j\).

### Intuition and Rationale

Hydrogen bonds stabilize secondary structures like \(\alpha\)-helices and \(\beta\)-sheets. Rewarding the formation of these bonds encourages the agent to build locally stable, realistic structures. Depending on implementation, \(h_{ij}(s)\) may be a binary value or a continuous measure of bond strength.

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

Hydrophobic residues tend to be buried in the protein core. This term penalizes high solvent-accessible surface area (SASA) for these residues, thereby rewarding conformations where they are properly buried. A well-packed hydrophobic core is essential for structural stability.

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

Electrostatic interactions play a key role in protein stability. This term calculates a Coulomb-like interaction between residues. Like charges (positive product) lead to a penalty, while opposite charges contribute favorably. Dividing by the distance emphasizes short-range interactions. This ensures the RL agent considers long-range electrostatic effects when optimizing the fold.

---

## 8. Template Similarity Reward \(R_t(s)\)

### Mathematical Formula

One common formulation is:

$$
R_t(s) = \frac{1}{\mathrm{RMSD}(s, s^{\mathrm{template}}) + \epsilon}
$$

Alternatively, one might use a TM-score that scales between 0 and 1.

### Intuition and Rationale

When a structural template is available, this term rewards the agent for producing a conformation similar to the known template. The inverse RMSD formulation means that smaller deviations yield higher rewards. This guides the agent towards biologically plausible structures by leveraging homologous information.

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

The pLDDT score reflects the model’s confidence in the predicted position of each residue. By weighting the positional error by these scores and taking the inverse, the term rewards conformations that are closer to the true structure, especially in high-confidence regions. This term is useful when high-quality experimental data or predictions are available for comparison.

---

## 10. Distogram Consistency Reward \(R_d(s)\)

### Mathematical Formula

$$
R_d(s) = -D_{KL}\Bigl(P_{\mathrm{agent}}(\cdot\,|\,s) \,\Big\|\, P_{AF}(\cdot)\Bigr)
$$

where \(D_{KL}\) denotes the Kullback–Leibler divergence between the agent’s observed inter-residue distance distribution \(P_{\mathrm{agent}}(\cdot\,|\,s)\) and AlphaFold's predicted distogram \(P_{AF}(\cdot)\).

### Intuition and Rationale

AlphaFold provides a predicted distribution over inter-residue distances (a distogram). This term penalizes deviations between the distribution derived from the current conformation and the predicted one. A smaller KL divergence (i.e., a closer match) results in a higher reward. This probabilistic term helps integrate statistical trends from large-scale sequence data into the RL optimization.

---

## 11. Recycling Consistency Reward \(R_{rec}(s)\)

### Mathematical Formula

$$
R_{rec}(s) = -\frac{1}{N}\sum_{i=1}^{N} \|x_i^{(t)} - x_i^{(t-1)}\|^2
$$

where \(x_i^{(t)}\) and \(x_i^{(t-1)}\) are the positions of residue \(i\) at the current and previous recycling iterations, respectively.

### Intuition and Rationale

Many protein folding pipelines use iterative refinement (recycling) to improve the structure. A well-converged structure will show small changes between iterations. This term penalizes large differences between successive iterations, encouraging the agent to refine the fold until it reaches a stable configuration. It effectively serves as a convergence indicator.

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

In complex environments like protein folding, extrinsic rewards may be sparse. Intrinsic rewards provide motivation for exploration by rewarding novelty. Here, the RL agent is incentivized to explore states where the forward model's prediction error is high. This encourages exploration of less familiar regions in the conformational space, potentially leading to the discovery of better folding pathways.

---

## Composite Reward Summary

The overall reward function is defined as:

$$
R(s,a) = w_c\, R_c(s) + w_{cl}\, R_{cl}(s) + w_b\, R_b(s) + w_r\, R_r(s) + w_{hb}\, R_{hb}(s) + w_{hp}\, R_{hp}(s) + w_e\, R_e(s) + w_t\, R_t(s) + w_p\, R_p(s) + w_d\, R_d(s) + w_{rec}\, R_{rec}(s) + w_{int}\, R_{int}(s,a)
$$

Each component is tuned via its corresponding weight \(w_k\) to balance its influence on the final reward. Collectively, these components guide the RL agent to generate protein conformations that are physically realistic, evolutionarily consistent, and optimized according to known biophysical principles.
