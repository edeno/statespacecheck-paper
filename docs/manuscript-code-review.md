Scientific review of the manuscript and analysis code, 14 September 2026.

Reviewed revision: `8a72a4369e37a380e8df4f19771f7a95d3e65035`. The working tree was clean when the review began. This report adds no changes to the manuscript, analysis implementations, or published artifacts.

Follow-up: [the combined review](/Users/edeno/Documents/GitHub/statespacecheck-paper/docs/review-synthesis.md) evaluates the other agent's review against additional experiments and revises the priorities below. In particular, adding 6% uniform mass to the Continuous predictions removes 95.5% of their original HPD flags. It also verifies the firing-rate association while correcting its interpretation, tests the sparse-control sensitivity, and confirms a stale interactive-viewer artifact. Read the combined review first; this report retains the detailed original findings and reproductions.

Revision scope: the [current fix plan](/Users/edeno/Documents/GitHub/statespacecheck-paper/docs/manuscript-fix-plan.md) retains full-recording fitting and retrospective model checking. Fitting and checking the same observations is not itself an error; formal calibration must account for the fitted procedure, and fewer flags alone do not establish better latent-state inference. The recommendations below have been aligned with that scope.

The central construction is mathematically defensible as a set of **local diagnostics of selected aspects of agreement**. However, the current evidence does not establish that passing these diagnostics implies reliable state inference, that fewer flags establishes a better model, or that the controls are fully specified by the decoder. There are also concrete implementation and mathematical exposition errors. These are addressable, but several affect the scientific argument rather than just presentation.

The main equations that checked out are the Poisson event factorization, normalization of the selected event's intensity, the ratio-of-integrated-intensities formula for the predictive mark distribution, the overlap coefficient, and the stated direction of KL divergence in the implementation. In particular, event-rate weighting in the predictive mark distribution is appropriate for the explicitly stated expected-event sampling construction; replacing it with an unweighted average of cell-identity probabilities would introduce an error. The code also correctly retains repeated events for spike counts greater than one, uses the decoder's actual overridden rate tables, and marginalizes dynamics mode before comparing the real-data models over position.

1. **Confirmed code error: the burst window is delayed by one time step. The generator also does not enforce a within-bin hard refractory period.**

   The manuscript specifies a 2–10 ms burst window ([main.tex:267](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:267)). After a spike, the implementation resets `steps_since_spike` to zero. At the beginning of the next time step that counter is still zero, so checking it against `(2, 10)` applies the burst multiplier at actual offsets **3–11 ms** ([simulation.py:559](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/simulation.py:559)).

   I forced one spike at time zero and recorded subsequent Poisson means. Relative to the baseline mean, the code produced:

   | Offset after spike, ms | 0 | 1 | 2 | 3–11 | 12 |
   | --- | --- | --- | --- | --- | --- |
   | Rate multiplier | 1 | 0 | 1 | 3 | 1 |

   The one subsequent refractory bin is suppressed as documented; the burst offsets are wrong. Use actual elapsed offsets consistently for both conditions and regenerate the affected Figure 3 analysis.

   Separately, `rng.poisson(rate)` can generate several spikes from the same neuron in a 1 ms bin. In the displayed seed-1 history phase, **1,051 of 3,393 spikes (31%)** occurred in cell/time bins with more than one spike; one bin contained five. Those events cannot all obey a 1 ms minimum interspike interval. Either describe this as a binned history-modulated count process with suppression of the following bin, or implement a process that actually enforces the claimed refractory interval. This is consequential at the chosen rates, not an edge case that never occurs.

   The perturbation also changes marginal firing rates. With position fixed at a field center, 100,000 simulated bins at seed 314 gave a mean count of **0.33557**, versus the unmodulated mean **0.199471**. Thus the code comment that the change is not in the per-step marginal ([simulation.py:467](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/simulation.py:467)) is incorrect. The conditional single-event likelihood can retain its spatial shape while the marginal firing rate changes. If the intended experiment isolates temporal dependence, add a rate-matched control.

2. **Confirmed model mismatch: the simulation's “well-specified” reference and controls are not fully matched to the decoder.**

   The generator folds a continuous Gaussian random walk at the track boundaries ([simulation.py:378](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/simulation.py:378)). The decoder instead evaluates a Gaussian at grid points and renormalizes each column after truncation ([simulation.py:238](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/simulation.py:238)). Reflection and truncation/renormalization are different transition laws.

   There is also appreciable discretization at the default grid spacing of 1 and step standard deviation of 0.5. At an interior integer position, the decoder assigns probability **0.78657** to staying in that grid cell. Integrating the true Gaussian over the corresponding nearest-grid-point cell gives **0.68269**. At the left boundary the corresponding probabilities are **0.88054** and **0.68269**. The decoder's interior step second moment is **0.21501**, versus **0.25** for the underlying continuous increment. These comparisons use cells halfway between grid centers and distinguish an approximation from an exactly matched discrete null.

   This does not make a grid approximation unacceptable. It does mean that the reference should not be presented as an exact well-specified null without checking the approximation. Generate from the decoder's discrete transition matrix for one genuinely matched benchmark, or use a reflected, cell-integrated transition and demonstrate grid convergence.

   Other mismatches are explicit in the construction:

   - The last second of recovery is a deterministic approach to the sparse location, but is included in the “Well-specified” comparison ([figure03_simulation.py:435](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/figure03_simulation.py:435); [figure03_summary.py:127](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/figure03_summary.py:127)).
   - The sparse control has constant true position and a random-walk decoder transition ([figure03_simulation.py:710](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/figure03_simulation.py:710)). It is observation-matched and can satisfy the operational overlap criterion; it is still transition-misspecified.
   - The replay trajectory is a deterministic out-and-back sweep, not a draw from the decoder's random walk ([figure03_simulation.py:339](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/figure03_simulation.py:339)). Tracking it well is different from correctly specifying its dynamics.

   Consequently, the sparse result cannot by itself establish that KL is flagging a fully correct model. Preserve these useful demonstrations, label their exact purpose, and add a fully matched sparse generative model. The distinction matters especially in [main.tex:271](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:271), [main.tex:275](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:275), and the “known component” row in Figure 3.

3. **Calibration qualification: the real-data encoding models are fitted using the same session whose spikes are checked.**

   Both `.fit(...)` calls receive the full supplied position and spike arrays, without a training mask ([figure04_decoder.py:450](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/figure04_decoder.py:450)). The installed decoder defaults to using every position time for training when the mask is absent. The workflow then predicts and computes diagnostics on those same spike trains ([figure04_workflow.py:347](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/figure04_workflow.py:347)).

   Thus these are retrospective checks with plug-in parameters estimated from the assessed observations, including future observations. Conditional on fixed fitted parameters the filtering recursion uses the past correctly, but the complete fitted procedure is not a prediction based only on past data. The known-model predictive p-value calibration argument does not automatically apply.

   There is also no running-only training restriction in this path. Spikes in putative replay periods can contribute to place fields at the animal's physical location even though the scientific interpretation is that they represent another location. This can contaminate the observation model. I verified the absence of the restriction; the magnitude and direction of its effect require a reanalysis.

   Report the fitting procedure explicitly and retain full-recording checks as specified in the current plan. If formal fitted-model calibration is claimed, assess the complete procedure, for example through a simulate-and-refit study. Retrospective checking on the observations used for fitting is legitimate and does not itself require a holdout.

4. **Mathematical counterexample: the predictive rank checks mark rarity, which is not equivalent to disagreement between state distributions.**

   The manuscript acknowledges that a high p-value does not prove state overlap, but its positive interpretation still substantially links rank p-values to the same state consistency targeted by HPD ([main.tex:219](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:219)). The converse failure also matters: a low p-value can occur with *exactly equal* state distributions and highly informative tuning.

   Take the predictive state mass `P = (0.1, 0.8, 0.1)` and two cells with intensities

   \[
   \lambda_1(x)=0.01P(x),\qquad \lambda_2(x)=0.99P(x).
   \]

   The normalized event likelihood for either cell is exactly `P`. Therefore HPD overlap is 1 and KL is 0. Nevertheless, the predictive identity probabilities are `(0.01, 0.99)`, so a spike from cell 1 has **p = 0.01** and is flagged. Running the actual diagnostic code gives those results to floating-point precision.

   This is a valid rare-mark result under a correct model, not a faulty p-value computation. It disproves an interpretation that every small rank p-value reflects conflicting locations. Make the distinction explicit: HPD measures a chosen geometric relation; rank p measures observation-space surprise, which also depends on relative firing rates and unit definitions.

   This bears on the claim that imperfect automated curation does not affect the purpose of the example ([main.tex:302](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:302)). Unequal splitting or merging changes the mark categories, their probabilities, and potentially their tuning. A sensitivity analysis using curated units or controlled split/merge perturbations would substantiate robustness.

5. **Major interpretation problem: making a prediction diffuse can remove HPD flags without improving predictive quality.**

   With the installed HPD tie convention, a uniform distribution's HPD region contains every state bin. Its overlap with every nonempty likelihood HPD region is therefore **1**. A decoder that discards all temporal information and predicts uniformly can receive perfect HPD overlap everywhere.

   This behavior is intentional under the manuscript's nesting criterion. It also means the **92% HPD “rescue”** result is insufficient evidence of better state inference or better prediction on its own ([main.tex:330](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:330); [main.tex:346](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:346)). The added fragmented state specifically broadens predictions, so this alternative explanation is directly relevant.

   Add an always-uniform prediction and uniform-mixture control on the same recording. Report overlap, region sizes, and flag removal versus added uniform mass. Distinguishing calibration from concentration is established practice in evaluating predictive distributions; see [Gneiting, Balabdaoui, and Raftery (2007)](https://doi.org/10.1111/j.1467-9868.2007.00587.x). The proposed application of that distinction here follows from the uniform counterexample.

   The sparse p-value result also has limited discriminatory content at the chosen cutoff. Along the entire seed-1 sparse window, the smallest positive predicted identity probability was **0.13354**; observed rank p-values ranged from **0.16074 to 1**. With only the five similarly probable sparse cells active, a 0.05 flag was unattainable for any possible active-cell mark at those predictive distributions. Thus zero flags in this example does not independently demonstrate a powerful ability to distinguish benign sparse activity from faults.

6. **The proposed smoothing extension does not preserve predictive checking and has no guaranteed power improvement.**

   [main.tex:348](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:348) proposes replacing the prediction with any posterior, particularly a smoothing distribution, to increase power. A full smoother already incorporates the observation being checked. It can pull the state distribution toward that observation and erase precisely the conflict the diagnostic was designed to find.

   A two-state counterexample uses `P = (0.999, 0.001)` and cell intensity vectors `(0.001, 1)` and `(1, 0.001)`. For an event from the first cell, the actual code gives predictive HPD overlap **0** and rank p **0.001997**. Replacing `P` by the posterior after that event changes the values to **1** and **1**, without any independent new evidence. The total intensity is constant, so this example does not rely on changing event-rate weights.

   State that this creates a different retrospective diagnostic with different calibration. Derive its observation reference and assess its behavior separately if developing that extension. Ordinary posterior predictive checking is possible, but its calibration and power are not inherited by substitution. The current revision can retain the one-step prediction and remove the unsubstantiated smoothing-power claim.

7. **Confirmed mathematical exposition error: reversing KL does not eliminate sensitivity to a broad likelihood.**

   [main.tex:239](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:239) says choosing `D_KL(P || Q)` avoids the broad-likelihood behavior of the reverse direction. For centered Gaussian distributions with standard deviations `s_P` and `s_Q`,

   \[
   D_{\rm KL}(P\|Q)=\log(s_Q/s_P)+\frac{s_P^2}{2s_Q^2}-\frac12.
   \]

   This diverges when **either** `s_Q/s_P` tends to infinity or to zero. The growth is logarithmic for a broad `Q` and quadratic in the opposite ratio. The chosen direction reduces one form of sensitivity; it does not avoid it. The KL implementation agrees with its equation—the error is in the rationale.

   The extrapolation to the entire f-divergence family ([main.tex:342](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:342)) should also be narrowed. These measures do distinguish unequal nested distributions, but their actual flagging depends on the divergence and calibration. In particular, a raw divergence threshold is different from comparing a divergence with its own model-implied reference distribution. The latter is an established approach to conflict checking: [Nott et al., “Checking for prior-data conflict using prior to posterior divergences”](https://arxiv.org/abs/1611.00113). That paper uses a different distribution pair, so it is relevant context and a possible comparator, not an interchangeable implementation of this manuscript's statistic.

8. **The threshold comparison does not isolate comparative sensitivity or error rates.**

   HPD and KL use nominal 1% empirical baseline tails; the rank statistic uses a 5% cutoff ([diagnostics.py:49](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/diagnostics.py:49)). The committed HPD threshold is exactly **0**, and the rule includes ties. Consequently, its flagged baseline fraction need not equal 1%. In the reported recovery comparison the median percentages are about **1.77% HPD, 2.76% rank, and 1.18% KL**. Comparing their misfit flag fractions therefore mixes sensitivity with different operating points.

   Provide threshold curves or comparisons at matched realized false-positive rates, assessed on independently simulated, truly matched null data. Include sensitivity to HPD coverage (for example 80%, 90%, 95%, 99%) and explain the fixed real-data overlap cutoff of 0.05; it is a geometric cutoff, not a significance level.

   For a mark genuinely drawn from a specified categorical distribution `q`, the exact statistic `sum(q[c] for q[c] <= q[observed])` is **super-uniform**, generally not uniform because of discreteness and ties. Large values do not constitute affirmative evidence that the model is true. Several events in one bin share a prediction, and sequential events are dependent, so the displayed cutoff alone supplies no session-wide error control. If the flags are descriptive, say so; if inferential claims are intended, define the target error rate and account for fitting and dependence.

   The realization medians also need their distributions or paired uncertainty, already recognized in `TODO.md`. The current approximate median SEs do not capture the full uncertainty from re-estimating a shared pooled threshold; a bootstrap of the entire calibration-and-evaluation procedure would address that if uncertainty in the full procedure is to be reported.

9. **Point-estimate error is insufficient to establish that unflagged misspecification is harmless.**

   [main.tex:294](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:294) uses median absolute error of the posterior mean to interpret history dependence and the sparse control. History dependence can leave the mean accurate while making posterior uncertainty wrong. Symmetric diffuse or multimodal posteriors can also have accurate means.

   I reran seeds 1–10 and evaluated coverage using the code's 95% HPD masks at the grid cell nearest the true continuous position. The median across these ten realizations was **95.20% in the opening baseline** and **93.50% during history dependence**. Seed 4 had **96.30%** and **89.08%**, respectively. These are exploratory results from ten seeds, not a replacement for the manuscript's 100-realization analysis or a formal significance test. They show that the omitted property is not automatically unchanged.

   In seed 1's sparse control, median absolute posterior-mean error was **0.817 a.u.**, but median posterior standard deviation was **6.384 a.u.**, median MAP error was **2 a.u.**, and maximum MAP error was **7 a.u.**. The mean-error statistic is computed correctly; its interpretation needs limits.

   Add posterior coverage, uncertainty width, and a distributional score across the full realization set. In replay, also report accuracy against the represented trajectory, separately from the deliberately large gap to physical position. Currently the returned simulation result retains physical position, so retaining represented position would make that check straightforward.

10. **The diagnostics have important joint-observation and identifiability blind spots that limit the broad goodness-of-fit claims.**

    The omission of exposure and total count is clearly disclosed in Methods and is not a factorization bug. However, the broader claims in the Introduction and Discussion should retain that restriction. A particularly simple example illustrates what can be missed within a bin: let `P=(1/2,1/2)`, with one cell firing only in the first state and another only in the second. Observe one spike from each cell in the same bin. The joint observation has probability zero at every possible state, but both events have **HPD overlap 1 and rank p 1** against the shared prediction. KL is infinite in this example. The individual nesting criterion does not guarantee joint compatibility with one latent state.

    With very small positive off-location rates, the same example becomes extremely unlikely rather than impossible. This is relevant to a binned population decoder that evaluates simultaneous events separately against the same `P_k`. Add a count/time-rescaling or joint-bin predictive check, or explicitly restrict what passing the event diagnostics establishes.

    Likewise, a coherent wrong remapping can give accurate internal agreement and incorrect state interpretation. The code comments acknowledge reflection as such an undetectable remapping, but the manuscript mainly demonstrates a single spatially incoherent scramble. A coherent wrong-map control would make the distinction between consistency and truth concrete. Disagreement also cannot identify the faulty model component by itself: the Figure 3 component labels come from experimental design, not from an identifiable decomposition supplied by the diagnostics.

11. **The extension to arbitrary latent coordinates or continuous marks needs a reference-measure qualification.**

    The normalized likelihood is a distribution only after choosing the state-space measure in its denominator ([main.tex:145](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:145)). If `z=g(x)` is a nonlinear reparameterization, normalizing the scalar likelihood over `dz` generally does not give the pushforward of the likelihood normalized over `dx`. HPD sets and their volume overlap are also coordinate-dependent. The equal-grid counting implementation implicitly chooses a spatial measure and equal bin volumes.

    This is manageable for physical track position in centimeters, but is material to the discussion of generic latent states. Specify the measure, finite-normalization requirement, and grid approximation. Use cell volumes for genuinely unequal cells, and check sensitivity to resolution and HPD ties.

    The continuous-mark rank statistic has a related issue: a nonlinear waveform-feature transformation changes density values through its Jacobian and can change their ordering. An extreme example transforms a one-dimensional observation by its model CDF: the transformed density is uniform, making all density ranks tie. This issue and invariant alternatives are treated by [Evans and Jang (2010), “Invariant P-values for model checking”](https://arxiv.org/abs/1001.1886). It does not invalidate the discrete identity calculations in this paper, but the claimed natural clusterless extension needs this qualification and ideally an actual clusterless example.

12. **Likely reviewer requests: broaden the benchmark and position the contribution more precisely.**

    The 100 realizations vary trajectories and spikes under one parameter setting and one fixed remapping; they do not sample the diversity of remapping structures. The permutation is constrained to displace all fields by at least three spacings ([figure03_protocol.py:206](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/figure03_protocol.py:206)), so describing the remapped locations as statistically independent is not literally correct. A constrained permutation is dependent and preserves the ensemble's set of field centers.

    The baseline peak is about **200 Hz**, and the replay rate scale of 20 rather than 5 makes its peak about **800 Hz**. That fourfold replay gain, which the decoder is told, is in the code but is not clearly specified in the replay Methods paragraph. Report it. Test lower rates, heterogeneous fields and gains, different neuron counts, bin widths, transition strengths, and several remappings. Vary perturbation order or use independent segments so recovery transients do not determine the comparison. No particular biological upper firing-rate limit is assumed in this criticism; the issue is robustness to the chosen regime and its substantial within-bin multiplicities.

    The real example is one session from one animal. It contains 203 units, about 23.64 minutes, and 870,018 analyzed events. These are many observations of one recording, not independent biological replications. Either add sessions/animals or frame the real data as an illustrative case study. The displayed window was selected around a KL spike during immobility according to [figure-pipeline.md:239](/Users/edeno/Documents/GitHub/statespacecheck-paper/docs/figure-pipeline.md:239); state that selection rule and distinguish it from independently identified replay. Stratifying whole-session results by running, immobility, and independently detected replay would be informative.

    The assertion that existing cross-validation and related tools only give a single recording-wide score is too broad ([main.tex:99](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:99)). Held-out predictive losses and residuals can be inspected locally. Marked time-rescaling methods already operate on individual events; [Tao et al. (2018)](https://link.springer.com/article/10.1007/s10827-018-0698-4), already cited in the manuscript, explicitly develops this construction. Predictive density-rank checks and prior–data conflict checking also have an established literature, including [Evans and Moshonov (2006)](https://utstat.utoronto.ca/mikevans/papers.html) and the Evans–Jang and Nott papers above. The defensible contribution is the selected consistency criterion, its neural-decoding application and interpretation, and the implementation—not the invention of local predictive model criticism as a whole.

    Finally, the Discussion's claims of robustly lower error rates and greater predictive-check computational cost ([main.tex:342](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:342)) need appropriate operating-point comparisons and timings. The exact categorical implementation makes the latter model- and implementation-dependent.

13. **Reproducibility gaps remain despite strong internal artifact checks.**

    The repository documents that raw recording to the five derived input exports is external ([figure-pipeline.md:261](/Users/edeno/Documents/GitHub/statespacecheck-paper/docs/figure-pipeline.md:261)). Consequently, the DANDI link and figure scripts do not by themselves provide end-to-end reproduction. Archive the five exact exports if permitted, or supply an executable export procedure tied to exact DANDI assets, sorting/curation settings, and track definitions. Complete the archival DOI placeholder. This is a publication/reproduction requirement, not evidence that the existing numbers are wrong.

    The decode cache includes diagnostic arrays, but its fingerprint hashes data, decoder configuration, schema, and the `non_local_detector` version—not the diagnostic source code or `statespacecheck` version ([figure04_cache.py:190](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/figure04_cache.py:190)). A diagnostic implementation change requires a manual schema bump or forced recomputation. Otherwise old diagnostic values can accompany a new source hash in the summary. Automating that dependency, or separating cached decoder distributions from recomputable diagnostics, would improve the guarantee. The sampled checks below found no such staleness in this checkout.

14. **Two smaller confirmed code/contract discrepancies should be recorded separately from the main scientific concerns.**

    The general simulation decoder never assimilates `spike_counts[0]` and excludes all first-bin events ([decoding.py:692](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/decoding.py:692); [decoding.py:715](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/decoding.py:715)). Adding ten spikes to the first bin of an otherwise identical input leaves every returned posterior unchanged and returns no events for those spikes. A flat initial prior is valid before seeing data; it does not justify skipping the first observation. Either make that row an explicit dummy initial condition in the input contract, with zero counts required, or assimilate it. The displayed simulation's first bin happened to contain no spikes. Also, `simulate_walk` takes a first random increment before returning its first sample, so the manuscript's literal `x_0=0` claim does not match the returned seed-1 first position, **0.172792**.

    Exact last-timestamp events have a separate boundary mismatch in the real-data path: the paper maps an event at `time[-1]` to the last row ([figure04_diagnostics.py:202](/Users/edeno/Documents/GitHub/statespacecheck-paper/src/statespacecheck_paper/figure04_diagnostics.py:202)), while the installed decoder's `digitize(spike_times, time[1:-1])` assigns it to the penultimate row. I checked the actual recording: **zero spikes equal the final timestamp**, so this has no effect on the reported real-data results. Align the conventions for future inputs.

The changes with the highest scientific return are: correct the history generator; add a genuinely matched null and sparse control; compare the real-data diagnostics with uninformative and uniform-mixture predictions on the full recording; separate mark rarity from state consistency; and revise the smoothing and KL claims. Follow those with matched-error-rate comparisons, uncertainty calibration, and sensitivity across rates, maps, and recordings.

Verification performed:

- Read the manuscript and traced the scientific Figure 1–4 pipeline, focusing on simulation, filtering, per-event metrics, mode marginalization, real-data fitting, summaries, and the installed `statespacecheck` and `non_local_detector` implementations. Inspected the rendered main simulation and real-data figures. This was not an exhaustive independent audit of every GUI or plotting utility.
- Ran `uv run --frozen --no-sync pytest --no-cov -q -o addopts='' -m 'not slow'` with a writable temporary uv cache: **571 passed, 1 slow test deselected**, two non-failing warnings. Passing tests support implementation consistency; they do not establish the statistical validity of the benchmark.
- Reran the full default simulation at seed 1 for the detailed counterchecks, and seeds 1–10 for the coverage assessment. Did not regenerate the full 100-seed figure summary or refit the real-data decoders.
- Read the real-data cache using memory mapping. Its schema and fingerprint match the committed summary. Recomputed all three diagnostics at 500 evenly spaced event indices for each model from the cached predictive distributions and shared place fields; maximum absolute discrepancy was **0** for each metric and model.
- Confirmed full cached flag totals at cutoff 0.05: Continuous HPD **18,790**, Continuous–Fragmented HPD **1,677**; Continuous rank **33,954**, Continuous–Fragmented rank **26,287**. These agree with the summary's paired counts. These checks validate cache/implementation agreement, not the scientific adequacy of the fitted models.
- Tested the rare-mark, uniform-prediction, posterior-reuse, impossible-co-spike, burst-offset, and ignored-first-bin counterexamples directly against the current functions. The Gaussian KL limit is an analytic counterexample.

Two compact reproductions of the central findings, runnable from the repository environment:

```python
import numpy as np
from statespacecheck_paper.diagnostics import (
    compute_spike_event_diagnostics_from_rates,
)

p = np.array([0.1, 0.8, 0.1])
rates = p[:, None] * np.array([0.01, 0.99])[None, :]
d = compute_spike_event_diagnostics_from_rates(
    p[None, :], rates, np.array([0]), np.array([0])
)
print(d.event_hpd_overlap, d.event_kl_divergence, d.event_predictive_pvalue)
# [1.], approximately [0.], [0.01]
```

```python
import numpy as np
from statespacecheck_paper.simulation import simulate_spikes_history_dependent

class Recorder:
    def __init__(self):
        self.rates = []

    def poisson(self, rate):
        self.rates.append(rate.copy())
        # Force exactly one initial spike, then record without further spikes.
        return np.full_like(rate, int(len(self.rates) == 1), dtype=int)

recorder = Recorder()
simulate_spikes_history_dependent(
    np.zeros(14), np.zeros(1), 10.0, 5.0, recorder
)
print(np.array(recorder.rates).ravel() / recorder.rates[0][0])
# [1, 0, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1]
```
