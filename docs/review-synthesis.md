Combined scientific review, 14 September 2026.

Revision scope: the [current fix plan](/Users/edeno/Documents/GitHub/statespacecheck-paper/docs/manuscript-fix-plan.md) retains fitting and checking on the full recording. Reusing the recording for retrospective model checking is legitimate; the remaining qualification concerns calibration of the fitted procedure and the interpretation of flag reduction. The recommendations below use full-recording broadening controls and simulation validation.

I evaluated the supplied agent review against the manuscript, implementation, canonical Figure 4 cache, and additional simulation runs. Its strongest criticisms are supported, especially the sensitivity of HPD overlap to broadening and the distinction between mark rarity and spatial disagreement. Several absolute statements and numerical interpretations are overstated or incorrect. The synthesis below supersedes the priority ordering in [my original review](/Users/edeno/Documents/GitHub/statespacecheck-paper/docs/manuscript-code-review.md), whose detailed findings remain applicable. Numerical results and verification metadata are saved in [review-synthesis-evidence.json](/Users/edeno/Documents/GitHub/statespacecheck-paper/docs/review-synthesis-evidence.json).

The core diagnostic equations are implemented consistently. That conclusion does not establish that their proposed interpretation or validation is correct. The most consequential remaining question is whether the diagnostics support useful model refinement beyond detecting selected forms of incompatibility. The current evidence does not yet establish that claim.

1. **Highest priority: the HPD “rescue” result can be reproduced by modest broadening alone.**

   I independently checked all 17,289 events flagged by Continuous but rescued by Continuous–Fragmented. Their predictive 95% HPD regions contain a median of **8 versus 110 of 248 valid position bins**, respectively. The other review's 109-bin figure is approximately right; the current implementation gives 110.

   I then performed a direct counterfactual on all 18,790 original Continuous HPD flags. Holding each event's likelihood and the Continuous prediction's spatial structure fixed, I replaced the prediction by

   \[
   P'_k=(1-w)P_k+wU,
   \]

   where `U` is uniform over the valid track bins, and recomputed the actual HPD-overlap function at the manuscript's 0.05 cutoff.

   | Added uniform weight | Original flags removed | Fraction removed |
   | --- | ---: | ---: |
   | 2% | 4,643 / 18,790 | 24.7% |
   | 5% | 14,777 / 18,790 | 78.6% |
   | **6%** | **17,947 / 18,790** | **95.5%** |
   | 8.8% | 18,755 / 18,790 | 99.8% |

   **Adding only 6% uniform mass removes more of the original flags than the actual model refinement's approximately 92%.** This is a diagnostic counterfactual, not a refitted alternative decoder or a comparison of overall flag rates. It does not prove that the fragmented model fails to improve inference. It demonstrates that removing these flags, by itself, cannot distinguish improvement from broadening.

   The other review's mathematical bound is valid: any set `A` containing at least probability `alpha` under this mixture must satisfy

   \[
   \frac{|A|}{N}\geq\max\left(0,\frac{\alpha-1+w}{w}\right),\qquad w>0.
   \]

   This follows from `P'(A) <= 1-w+w|A|/N`. At 95% coverage, uniform mass just above 5% forces inclusion of additional track area. This is an inherent consequence of the selected region coverage and nesting criterion, not an arithmetic error.

   Two qualifications correct the supplied review. First, the predictive fragmented probability is not confined to 4–9% “almost everywhere”: it lies in that range for **45.1% of time bins**, and its 10th–90th percentiles are **4.03–35.19%**. Its time median is indeed 8.83%. Second, fragmented-mode probability is slightly different from the total uniform component in the position prediction. For the installed transition structure, with previous filtered fragmented probability `f`,

   \[
   \Pr(F_k\mid\text{past})=0.02+0.96f,
   \qquad w_k=0.02+0.98f.
   \]

   The latter also includes the uniform Fragmented-to-Continuous transition. Its time median is 8.98%, giving a lower bound of about 44.3% of the track for a 95% region. The first prediction has a separate uniform initial condition. I checked the mode-probability recurrence against 500 cached rows; the maximum difference was below `5.4e-7`.

   **Revision needed:** report predictive-region sizes and compare Continuous, Continuous–Fragmented, uniform prediction, and a simple uniform-mixture control using the same full recording. Show flag removal versus added uniform mass and sensitivity to HPD coverage. Assess state accuracy and uncertainty calibration in simulations where ground truth is available. Revise [main.tex:346](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:346) to distinguish alleviating observed discrepancies from establishing more accurate state inference. Coverage alone cannot penalize excessive breadth; region sizes make that limitation visible.

2. **The predictive rank measures observation-space surprise, including neuron rarity. The stronger claim that it only flags rare units is unsupported.**

   The other review correctly identifies a substantial rate association. Using session event counts divided by duration, and excluding 21 units with no analyzed events, I obtained:

   | Quantity | Continuous | Continuous–Fragmented |
   | --- | ---: | ---: |
   | Spearman correlation: unit rate versus unit flag fraction, 182 active units | −0.864 | −0.876 |
   | Active units with every spike flagged | 88 | 87 |
   | All flags contributed by those always-flagged units | 16.6% | 20.9% |
   | All flags contributed by units below 0.1 Hz | **5.3%** | **6.9%** |

   There are 106 units below 0.1 Hz, but that includes the 21 units with no events; **85 active units** meet this threshold. Those units collectively contribute only 2,120 of the 870,018 events. Counting units without counting their events gives a misleading impression of what dominates the results.

   A constant mark-frequency baseline flags many of the same events: 58.1% of actual Continuous flags and 68.2% of Continuous–Fragmented flags are also flagged by ranks based only on each unit's session count. This supports a substantial marginal-rate contribution; it is an exploratory comparison using the same session, not a causal decomposition.

   The claimed explanation for the 28% p-value rescue and newly flagged spikes is also unproven. Units below 0.1 Hz account for only **20 of 9,373 rescued events** and **20 of 1,706 newly flagged events**. Rate dependence more generally may matter, but these very low-rate units do not explain those changes.

   The statement that a cell with low overall rate must rank near the bottom “whatever the predictive distribution looks like” is mathematically false. In a two-position example with cell rate functions `(0.01, 0)` and `(1e-8, 200)`, the first cell has a much smaller spatial-average rate, but a state prediction concentrated at the first position makes it the most probable mark. The implemented rank p-value for its spike is **1**.

   The supplied uniform-prediction toy also needs a complete rate vector and distinction between mark probability and rank p-value. For one constant-rate 1 Hz cell and ten constant-rate 200 Hz cells, the rare cell has `p=1/2001`, while **every 200 Hz cell has rank p=1**, because the inclusive tail contains all equally probable high-rate marks. Their individual mark probabilities are approximately 0.1; those are not their p-values. The reported toy numbers are not a general property of the statistic.

   Nevertheless, the underlying interpretation problem is real. My original counterexample makes the normalized event likelihood *exactly equal* to the predictive state distribution, while giving the event rank p-value 0.01. Thus a small mark p-value need not indicate conflicting state locations. This is compatible with a correctly calibrated categorical p-value: rare outcomes can be flagged under a correct model.

   At a **4 cm/s** running cutoff, I reproduced the supplied Continuous-model percentages: p-value flags **4.10% immobile versus 3.82% moving**, and HPD flags **4.41% versus 1.21%**. These show stronger immobility enrichment for HPD at that cutoff. They do not establish that p-values never localize to replay: immobility is not an independent replay label, the percentages depend on the speed cutoff, and firing-unit composition changes between behavioral conditions.

   **Revision needed:** distinguish mark surprise from state-distribution overlap, report both unit-weighted and event-weighted summaries, and stratify by firing rate and independently identified replay. Curated-unit and controlled split/merge sensitivity checks would address the dependence on mark definitions. Removing units changes the mark distribution and requires recomputation and calibration.

   I would not adopt the suggested “rate correction” without a new derivation. Conditioning on cell identity removes the observed categorical variable being ranked; latent position is not an additional observed outcome available for an ordinary predictive check. Dividing a predictive probability by a baseline rate produces a score, not automatically a p-value. A defensible new statistic could rank a prespecified rate-relative score under the original predictive mark distribution, but it would answer a different question and require validation. The present statistic itself does not need its correct probability formula repaired.

3. **The sparse-control sensitivity is verified, with important denominator and model qualifications.**

   I changed `sparse_control_ordinary_rate_scale` from zero to 0.05 in **both generation and decoding**, then reran the complete simulation. At seed 1:

   | Sparse-window quantity | Original control | Ordinary scale 0.05 |
   | --- | ---: | ---: |
   | Sparse-cell events | 10 | 10 |
   | Ordinary-cell events | 0 | 48 |
   | Sparse-cell p-value flags | 0 / 10 | **7 / 10** |
   | All-event p-value flags | 0 / 10 | **8 / 58** |

   Across seeds 1–10 with ordinary scale 0.05, the median sparse-cell flag fraction was **68.3%**, while the median all-event fraction was **10.7%**. This confirms that the zero result depends strongly on the composition of the active population. It is not evidence of a 70% overall false-positive rate. Even an exactly calibrated unconditional mark p-value need not flag only 5% within every chosen cell subgroup.

   Under the original seed-1 control, every active mark already has predictive probability greater than 0.05; a rank p-value below the cutoff is impossible. The zero-flag result therefore has limited evidential value for distinguishing benign sparsity from misfit.

   The “6–14 spikes per realization” range holds for seeds 1–10, not all 100 manuscript seeds. Regenerating the original spike trains for seeds 1–100 gave **1–18 sparse events**, median **9**, IQR **7–11**. I omitted decoding for that count-only check. The supplied review is right that percentages based on these counts are coarse; show counts and realization distributions.

   Supplying known observation rates to a controlled simulation is legitimate when explicitly described. Calling that alone an invalid “oracle” overstates the objection. The unresolved issues are the limited parameter setting and the transition mismatch: the sparse trajectory is stationary while the decoder assumes a random walk. It is not a fully matched null, and the KL flags cannot automatically be called errors against a correct generative model.

   **Revision needed:** retain an observation-matched sparse example with clear labeling, add a genuinely matched latent-state null, vary residual ordinary activity and sparse-population composition, and compare calibration and power at matched operating points.

4. **The original review's objective errors and broader validation concerns remain.**

   Agreement between equations and diagnostic implementation does not clear the simulation generator, mathematical exposition, or observation indexing. The supplied review does not rebut these independently verified findings:

   | Finding | Status and implication |
   | --- | --- |
   | Burst window implemented at 3–11 ms instead of the stated 2–10 ms | Confirmed off-by-one error in [simulation.py:559](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/simulation.py:559). Correct and regenerate the affected results. |
   | Within-bin Poisson multiplicities despite a hard refractory interpretation | Confirmed: 31% of seed-1 history-phase spikes occur in cell/time bins containing multiple spikes. Clarify the count-process model or enforce the stated process. |
   | History modulation leaves marginal firing rates unchanged | Incorrect: a fixed-position check increases the mean approximately 1.68-fold. Use a rate-matched control if isolating dependence. |
   | Reflected continuous generator versus truncated, renormalized grid transition | Confirmed difference; the nominal baseline is an approximation, not an exact discrete null. |
   | Reversing KL avoids sensitivity to a broad likelihood | Mathematically false: `KL(P || Q)` also diverges as Gaussian `Q` becomes arbitrarily broad. Its implementation is correct. |
   | Replacing prediction with an ordinary smoother increases power | No general guarantee; reusing the checked observation can erase the discrepancy. A smoothing-based diagnostic requires its own interpretation and validation. |
   | First-bin observations excluded and unassimilated | Confirmed general decoder contract issue. No effect on the displayed seed's empty first bin; require a dummy-row contract or assimilate the data. |
   | Real-data models fitted and assessed on the same session | Confirmed; fixed-parameter predictive calibration does not automatically apply to this fitted procedure. |

   The original report also documents unequal diagnostic thresholds, uncertainty hidden by posterior-mean error, joint-bin incompatibilities missed by per-event checks, coherent wrong remappings, coordinate/reference-measure dependence, one-session generalization limits, and incomplete raw-data export reproducibility. These remain relevant reviewer concerns. The prior finding that the canonical diagnostic cache agrees with the current functions remains intact.

5. **Other claims in the supplied review: verification and corrections.**

   | Claim | Evaluation |
   | --- | --- |
   | Figure 3 HPD threshold is zero, with ties producing more than 1% baseline flags | Confirmed. The recovery median is approximately 1.77%. However, overlaps are not universally `k/11`; the denominator is the smaller HPD-set size and varies. Report the threshold and coverage sensitivity. |
   | Broad place fields make zero overlap a demanding condition | Supported. A central Gaussian field with standard deviation 10 has a roughly 39-bin 95% region on this grid; boundary truncation and predictive width change the geometry. This alone does not establish the causal explanation for every missed drift event. |
   | Replay uses fourfold rates, supplied to the decoder | Confirmed: approximately 800 versus 200 Hz peak. Disclose the gain and test robustness to rates and rate uncertainty. |
   | Replay sweep speed equals the decoder's step standard deviation | **Incorrect.** The configured speed cap and step standard deviation are both 0.5, but the seed-1 sweep's actual maximum step is **0.0775**. The finite out-and-back construction leaves the cap inactive; for this track and duration, the maximum is at most approximately 0.1001. |
   | Drift is an extreme condition | The unconstrained velocity process has stationary standard deviation **1.053 a.u./ms**. The arithmetic is correct; biological realism depends on the intended mapping of these arbitrary spatial units. Present it as a stress test and include milder regimes. |
   | History misspecification is undetectable by construction | Too absolute. A cell-specific history multiplier can cancel from its normalized spatial likelihood, but history affects the prediction and relative mark probabilities. Weak response in this benchmark does not establish complete mathematical blindness. |
   | Predictive ranks are cheaper than HPD | Supported in a focused implementation benchmark, not a universal theorem. For 10,000 random events with 248 bins and 203 cells, median rank time was **6.60 ms**, versus **86.69 ms** for HPD. At 101 bins and 16 cells, times were **0.78 versus 30.64 ms**. These exclude upstream likelihood construction and are not end-to-end decoder timings. Revise the unsupported general cost ordering in [main.tex:342](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:342). |
   | Viewer parquet cache is stale | Confirmed against identical event times and cell IDs: Continuous HPD flags are **0.756% in the viewer versus 2.160% in the canonical cache**. The corresponding fragmented figures are **0.0541% versus 0.1928%**. This affects the existing viewer artifacts, not the manuscript's canonical figure summaries. Refresh them with provenance checks. |
   | Show realization variability, improve unit-quality checks, substantiate “lower error rates” | Agree. These are missing evidence and presentation issues, rather than demonstrated arithmetic errors. |

The most useful revision sequence is: correct the generator and mathematical claims; add exactly matched nulls and sensitivity analyses; compare real-data diagnostics and uniform-mixture controls on the full recording; distinguish mark surprise from spatial disagreement; then present uncertainty, operating-point comparisons, and broader recording replication where available. This would support a narrower but defensible contribution: local complementary diagnostics whose passing values do not certify accurate latent-state inference.

Verification scope: reviewed revision `8a72a4369e37a380e8df4f19771f7a95d3e65035`; the prior suite returned **571 passed, 1 slow test deselected**. The follow-up added direct cache calculations, counterfactual metric evaluations, simulation reruns, analytic counterexamples, and focused timings. I did not refit the real-data models, regenerate the full 100-seed diagnostic summary, or modify scientific source code, manuscript text, or existing figure/viewer artifacts. The broader sensitivity analyses and matched-null validation remain work needed to resolve the scientific concerns.
