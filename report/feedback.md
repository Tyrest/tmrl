# Report Feedback: Offline Reinforcement Learning for High-Frequency Control

## Overall Assessment
This is a well-structured and rigorous report that makes meaningful contributions to understanding algorithm performance in vision-based, high-frequency control domains. The writing is clear, experiments are comprehensive, and the findings are insightful. Below is detailed feedback organized by the rubric criteria.

---

## A. Well Motivated Topic Intro and Background

### Strengths
- **Clear problem statement**: The introduction effectively establishes the data collection bottleneck in robotics and the motivation for sample-efficient off-policy RL.
- **Good real-world connection**: The analogy between TrackMania's non-parallelizable nature and physical robotics data collection is compelling.
- **Concrete research question**: The paper clearly asks which algorithm family (distributional RL, ensemble methods, or standard actor-critic) works best under these constraints.

### Areas for Improvement
- **Title mismatch**: The title mentions "Offline Reinforcement Learning," but the methodology describes "semi-offline" or high-UTD off-policy learning with ongoing data collection. This could confuse readers—consider revising to "Sample-Efficient Off-Policy Reinforcement Learning" or clarifying the distinction earlier.
- **Missing sentence fragment**: In Section 1.1, the sentence "This motivates off-policy reinforcement learning, ratios that perform many gradient updates per collected transition" is incomplete/grammatically awkward. Should likely read: "This motivates off-policy reinforcement learning with high update-to-data (UTD) ratios that perform many gradient updates per collected transition."
- **Contributions could be stronger**: The second contribution ("Extensions to the open-source tmrl framework") is vague. Specify what extensions were made (e.g., TQC/REDQ implementations, DINOv3 integration, etc.).

---

## B. Relevant Related Work

### Strengths
- **Comprehensive coverage**: The related work spans off-policy RL, distributional/ensemble methods, offline RL, and vision-based racing—all directly relevant.
- **Clear positioning**: The paper effectively distinguishes its "semi-offline" setting from pure offline RL.
- **Appropriate citations**: Key foundational works (DDPG, TD3, SAC, TQC, REDQ, D4RL, CQL, CARLA, TMRL) are all cited.

### Areas for Improvement
- **Missing key delta statement**: While the related work mentions that TQC/REDQ were primarily evaluated on MuJoCo, it would help to explicitly state upfront: "Our key contribution is the first systematic evaluation of these methods in a vision-based, high-frequency control domain with real-time data constraints."
- **Consider citing DROQ**: Since you mention DROQ in your config files (config_droq.json), you might want to mention Dropout Q-functions as another relevant ensemble approach if you experimented with it.
- **DINOv3 reference missing**: The ablation section mentions DINOv3 but there's no citation for DINOv2/v3 in the related work or bibliography.

---

## C. Methodological Rigor

### Strengths
- **Detailed POMDP formulation**: The problem formulation is complete with state, observation, action, reward, and dynamics clearly defined.
- **Algorithm descriptions are thorough**: The mathematical formulations for SAC, TQC, and REDQ are accurate and well-presented.
- **Reproducible setup**: Hyperparameters are tabulated (Table 1), training budget is specified (30 hours), and track descriptions are detailed.
- **Appropriate ablations**: Testing encoder architectures and temporal representations addresses key design choices.

### Areas for Improvement

#### Hyperparameter Concerns
- **SAC has 1 critic?**: Table 1 shows SAC with 1 critic, but standard SAC uses twin critics (2). This is inconsistent with the algorithm description in Section 3.2.1 which correctly describes twin critics. Please clarify—is this a typo, or was a single-critic variant used?
- **REDQ UTD discrepancy**: The REDQ paper recommends UTD=20, but you use UTD=2.0. This significantly undercuts REDQ's intended advantage. Was this due to computational constraints? This should be explicitly discussed as it may explain REDQ's failure.
- **Missing hyperparameters**: What is the discount factor γ? Target network update rate (τ)? Number of quantiles (K) and truncation amount (d) for TQC?

#### Experimental Design
- **Only 2 tracks**: While the tracks represent different styles (fullspeed vs. technical), the generalization of findings to other environments is limited. Acknowledging this limitation would strengthen the paper.
- **No multiple random seeds reported**: Are these results from single runs? Standard practice is 3-5 seeds per configuration with confidence intervals.
- **Ablations on different track**: The ablations are on "Power Source" track but main experiments are on Fall 2020-01 and Eaux. Why the different track? This makes it harder to connect ablation findings to main results.
- **Standard error vs. standard deviation**: Table 2 reports "mean ± SE" but Section 3.3.2 mentions reporting "standard deviations." Clarify which is used and be consistent.

---

## D. Accurate Results and Analysis

### Strengths
- **Clear performance comparison**: Tables 2 and 3 effectively summarize final performance and relative gains.
- **Learning dynamics analysis**: The discussion of learning curves (Figure 2) provides insight beyond final performance.
- **Statistical significance testing**: Welch's t-tests (Table 6) properly validate that differences are not due to chance.
- **REDQ failure analysis**: The paper doesn't hide the negative result and provides reasonable hypotheses for REDQ's failure.
- **Ablation insights**: The finding that temporal context dominates encoder architecture is valuable and clearly stated.

### Areas for Improvement

#### Missing Details
- **How is "epoch" defined?**: The paper mentions epochs but never defines what constitutes an epoch in this asynchronous training setup.
- **Gradient steps vs. environment steps**: Table 7 uses "Steps to Reach Threshold"—are these gradient steps or environment steps? The distinction matters for sample efficiency claims.
- **Episode length/termination conditions**: What happens when the car crashes? Is there a timeout? This affects return interpretability.

#### Analysis Gaps
- **TQC collapse on Eaux unexplored**: The observation that TQC collapses around epoch 5000 on Eaux is interesting but not deeply analyzed. What specifically about technical tracks causes this? Is it the higher penalty for conservative braking?
- **No qualitative analysis**: Including video frames or trajectory visualizations would help readers understand behavioral differences between algorithms.
- **REDQ hyperparameter sensitivity unexplored**: You hypothesize that REDQ needs domain-specific tuning, but didn't test alternative hyperparameters. Even one additional configuration would strengthen this claim.

#### Potential Inconsistency
- **Check Table 2 vs Table 7**: TQC on Fall 2020-01 reaches 150 at step 3,252 (Table 7), and final performance is 154.4 (Table 2). But SAC never reaches 150 per Table 7, yet Table 2 shows SAC at 128.3. This is consistent, but worth double-checking the data.

---

## E. Significance / Impact

### Strengths
- **Practical implications clearly stated**: The discussion section connects findings to autonomous vehicles, drones, and industrial control.
- **Honest limitations**: The paper acknowledges environmental, algorithmic, and evaluation limitations.
- **Clear takeaways**: The main message—that algorithm performance is environment-dependent and no single approach universally dominates—is well-supported and valuable.

### Areas for Improvement
- **Quantify "high-frequency"**: The paper repeatedly mentions "high-frequency control" but never specifies the control rate (Hz). Is it 20Hz? 60Hz? This matters for transferability claims.
- **Missing comparison to baselines**: How do these returns compare to human performance or scripted baselines? Is 154.4 return good? Without reference points, it's hard to assess absolute performance.
- **Code release details**: The footnote mentions a GitHub link, but is the code actually released? Verify the link works and consider mentioning what's included (trained models, training scripts, etc.).

---

## F. Clarity of Writing

### Strengths
- **Well-organized structure**: The paper follows a logical flow from motivation through methodology, results, and discussion.
- **Professional tables**: Tables are well-formatted with clear headers and appropriate precision.
- **Mathematical notation is consistent**: Algorithm formulations use consistent notation throughout.

### Areas for Improvement

#### Grammar/Typos
- **Section 1.1**: "This motivates off-policy reinforcement learning, ratios that perform..." → incomplete sentence (mentioned above)
- **Section 3.3.1 Track descriptions**: These read more like gaming reviews than academic descriptions. Consider trimming and focusing on the control-relevant characteristics.
- **Unused LaTeX command**: `\newcommand{\tyler}[1]{\textcolor{red}{#1}}` suggests there may be unresolved comments—check if any `\tyler{}` commands remain.

#### Figures
- **Figure 1**: Consider adding subfigure labels in the caption (a) and (b) for easier reference.
- **Figure 2**: Would benefit from clearer legend placement and possibly log-scale for y-axis to show early learning dynamics.
- **Missing Power Source track description**: The ablations use "Power Source" track but it's never described. Add a brief description or justify why this track was chosen.

---

## Summary of Key Recommendations

### Must Fix
1. Correct the sentence fragment in Section 1.1 about UTD ratios
2. Clarify SAC's number of critics (Table 1 says 1, algorithm description says 2)
3. Explain why REDQ uses UTD=2 instead of the paper's recommended UTD=20
4. Specify control frequency (Hz) somewhere in the paper
5. Add missing hyperparameters (γ, τ, K, d for TQC)

### Strongly Recommended
1. Revise title to better reflect the "semi-offline" or "sample-efficient off-policy" nature of the work
2. Strengthen contributions list with specific framework extensions
3. Define epochs and clarify step units in tables
4. Add brief Power Source track description for ablation context
5. Report number of random seeds used

### Nice to Have
1. Add human or scripted baseline for absolute performance reference
2. Include trajectory visualizations
3. Test at least one alternative REDQ configuration
4. Add DINOv2/v3 citation

---

## Grade Estimate by Criterion

| Criterion | Rating | Notes |
|-----------|--------|-------|
| A. Topic Intro & Background | Good | Clear motivation, minor clarity issues |
| B. Related Work | Good | Comprehensive, could strengthen novelty statement |
| C. Methodological Rigor | Good- | Solid overall, missing some reproducibility details |
| D. Results & Analysis | Good | Thorough analysis, some unexplored phenomena |
| E. Significance/Impact | Good | Clear takeaways, honest limitations |
| F. Clarity of Writing | Good | Well-organized, minor grammar issues |

**Overall**: This is a solid project report with meaningful contributions. Addressing the "Must Fix" items above will significantly strengthen the paper.
