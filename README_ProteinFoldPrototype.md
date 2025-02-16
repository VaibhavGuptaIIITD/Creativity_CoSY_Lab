# RL-Based Protein Folding Prototype

## Option 1: Hydrophobic-Polar (HP) Lattice Model (Simplified)  
For **initial proof-of-concept testing** with reinforcement learning (RL), the HP model offers unparalleled simplicity.  

### Why HP Models?  
- **Simplified Abstraction**  
  Sequences consist of only two residue types: hydrophobic (H) and polar (P). This ignores real-world side-chain complexity and focuses on hydrophobic collapse, a core folding principle [^1][^2].  
- **Tractable Conformational Space**  
  Constrained to 2D/3D lattices, short sequences (e.g., `HPPHHP`) have limited valid folds, avoiding Levinthal’s paradox [^3].  
- **Straightforward Energy Function**  
  Reward = Number of non-consecutive H-H contacts. Easy to compute and ideal for sparse reward RL [^4].  
- **Benchmarked in RL Studies**  
  Used in frameworks like [Gym-Lattice](https://github.com/yourlink/gym-lattice) and [FoldingZero](https://github.com/yourlink/foldingzero) [^5][^6].  

[^1]: Dill, K. A. (1985). *Biochemistry*, 24(6), 1501–1509. [DOI](https://doi.org/10.1021/bi00328a032)  
[^2]: Lau, K. F. & Dill, K. A. (1989). *Macromolecules*, 22(10), 3986–3997. [DOI](https://doi.org/10.1021/ma00200a030)  
[^3]: Levinthal, C. (1969). *Mossbauer Spectroscopy in Biological Systems*, 22–24.  
[^4]: Černý, V. (1985). *Journal of Optimization Theory and Applications*, 45(1), 41–51. [DOI](https://doi.org/10.1007/BF00940812)  
[^5]: Gym-Lattice repository (2022). [Link](https://github.com/yourlink/gym-lattice)  
[^6]: FoldingZero paper (2023). [DOI](https://doi.org/yourlink)  

---

## Option 2: Real Protein Families (Biologically Relevant)  
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

[^7]: Kubelka, J. et al. (2004). *J. Mol. Biol.*, 336(3), 731–744. [DOI](https://doi.org/10.1016/j.jmb.2003.12.013)  
[^8]: AlphaFold Dataset (2021). *Nature*, 596(7873), 583–589. [DOI](https://doi.org/10.1038/s41586-021-03819-2)  
[^9]: Honda, S. et al. (2008). *JACS*, 130(11), 3304–3305. [DOI](https://doi.org/10.1021/ja711278f)  
[^10]: CASP Competition (2020). [Link](https://predictioncenter.org/casp14/)  
[^11]: Jumper, J. et al. (2021). *Nature*, 596(7873), 583–589. [DOI](https://doi.org/10.1038/s41586-021-03819-2)  
[^12]: Rosetta@Home (2023). [Link](https://boinc.bakerlab.org/rosetta/)  

---

# Composite Reward Function for RL-Based Protein Folding

The composite reward function combines biophysical, evolutionary, and exploratory components:

$$
R(s,a) = \sum_{k} w_k\, R_k(s,a)
$$

### 1. **Contact Map Agreement** \(R_c(s)\)  
Rewards agreement with AlphaFold-predicted contacts [^8][^11]:  
$$
R_c(s) = \sum_{i,j}\,\mathbf{1}\bigl(d_{ij}(s) \le \tau\bigr)\,\mathbf{1}\bigl(d_{ij}^{AF} \le \tau\bigr)
$$

### 2. **Steric Clash Penalty** \(R_{cl}(s)\)  
Penalizes atom overlaps, inspired by force fields like AMBER/CHARMM [^13]:  
$$
R_{cl}(s) = -\sum_{i < j}\,\mathbf{1}\bigl(d_{ij}(s) < d_{\min}\bigr)
$$

### 3. **Bond Geometry Reward** \(R_b(s)\)  
Enforces ideal bond lengths (3.8 Å) [^14]:  
$$
R_b(s) = -\frac{1}{N-1}\,\sum_{i=1}^{N-1}\,\Bigl(\,\|x_{i+1} - x_i\| - L_{\mathrm{ideal}}\Bigr)^2
$$

### 4. **Ramachandran Reward** \(R_r(s)\)  
Uses Ramachandran distributions from MolProbity [^15]:  
$$
R_r(s) = \sum_{i=1}^N\,\log\,P_{\mathrm{rama}}\bigl(\phi_i,\,\psi_i\bigr)
$$

### 5. **Hydrogen Bond Reward** \(R_{hb}(s)\)  
Rewards H-bond formation (β-sheets/α-helices) [^16]:  
$$
R_{hb}(s) = \sum_{(i,j)\in\mathcal{HB}} h_{ij}(s)
$$

### 6. **Hydrophobic Packing Reward** \(R_{hp}(s)\)  
Minimizes SASA of hydrophobic residues [^1][^2]:  
$$
R_{hp}(s) = -\frac{1}{|\mathcal{H}|}\,\sum_{i \in \mathcal{H}}\,\mathrm{SASA}\bigl(x_i\bigr)
$$

### 7. **Electrostatic Energy Penalty** \(R_e(s)\)  
Coulombic interactions from MD force fields [^13]:  
$$
R_e(s) = -\sum_{i < j}\,\frac{q_i\,q_j}{\|\,x_i - x_j\,\| + \epsilon}
$$

### 8. **Template Similarity Reward** \(R_t(s)\)  
Uses TM-score for template alignment [^17]:  
$$
R_t(s) = \frac{1}{\mathrm{RMSD}\bigl(s,\,s^{\mathrm{template}}\bigr) + \epsilon}
$$

### 9. **pLDDT-Weighted Reward** \(R_p(s)\)  
Leverages AlphaFold’s confidence scores [^11]:  
$$
R_p(s) = \frac{1}{\tfrac1N \sum_{i=1}^N p_i\,d\bigl(x_i,\,x_i^{\mathrm{true}}\bigr) + \epsilon}
$$

### 10. **Distogram Consistency Reward** \(R_d(s)\)  
Matches AlphaFold’s distogram predictions [^11]:  
$$
R_d(s) = -\,D_{\mathrm{KL}}\Bigl(P_{\mathrm{agent}}(\cdot \mid s)\,\Big\|\,P_{AF}(\cdot)\Bigr)
$$

### 11. **Recycling Consistency Reward** \(R_{rec}(s)\)  
Stabilizes iterative refinement (AlphaFold recycling) [^11]:  
$$
R_{rec}(s) = -\,\frac{1}{N}\,\sum_{i=1}^N \bigl\|\,x_i^{(t)} - x_i^{(t-1)}\bigr\|^2
$$

### 12. **Intrinsic (Curiosity) Reward** \(R_{int}(s,a)\)  
Encourages exploration via prediction error [^18]:  
$$
R_{int}(s,a) = \bigl\|\,f(s) \;-\; \hat{f}(s,a)\bigr\|^2
$$

### Composite Reward Summary  
$$
\begin{aligned}
R(s,a) \;=\;& w_c\,R_c(s) \;+\; w_{cl}\,R_{cl}(s) \;+\; w_b\,R_b(s) \;+\; w_r\,R_r(s) \\
&+\; w_{hb}\,R_{hb}(s) \;+\; w_{hp}\,R_{hp}(s) \;+\; w_e\,R_e(s) \;+\; w_t\,R_t(s) \\
&+\; w_p\,R_p(s) \;+\; w_d\,R_d(s) \;+\; w_{rec}\,R_{rec}(s) \;+\; w_{int}\,R_{int}(s,a)
\end{aligned}
$$

### References for Reward Components  
[^13]: Cornell, W. D. et al. (1995). *JACS*, 117(19), 5179–5197. [DOI](https://doi.org/10.1021/ja00124a002) (AMBER)  
[^14]: Engh, R. A. & Huber, R. (1991). *Acta Cryst.*, A47(4), 392–400. [DOI](https://doi.org/10.1107/S0108767391001071)  
[^15]: Chen, V. B. et al. (2010). *Acta Cryst.*, D66(1), 12–21. [DOI](https://doi.org/10.1107/S0907444909042073) (MolProbity)  
[^16]: Baker, E. N. & Hubbard, R. E. (1984). *Prog. Biophys. Mol. Biol.*, 44(2), 97–179. [DOI](https://doi.org/10.1016/0079-6107(84)90007-3)  
[^17]: Zhang, Y. & Skolnick, J. (2004). *Proteins*, 57(4), 702–710. [DOI](https://doi.org/10.1002/prot.20264) (TM-score)  
[^18]: Pathak, D. et al. (2017). *ICML*. [arXiv](https://arxiv.org/abs/1705.05363) (Curiosity-driven RL)  

---

## Usage Notes  
- Replace `yourlink` placeholders with actual URLs/DOIs.  
- Tune weights \(w_k\) empirically (start with equal weights).  
- Validate folds against experimental PDB structures using RMSD/TM-score.  
