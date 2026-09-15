Recommended manuscript revision plan, 14 September 2026.

This plan follows the [combined review](/Users/edeno/Documents/GitHub/statespacecheck-paper/docs/review-synthesis.md). Its purpose is to correct demonstrated errors and establish which scientific claims survive stronger validation. It does not assume that the results must favor HPD overlap, predictive ranks, or the Continuous–Fragmented model. This document proposes work; the scientific implementation and manuscript have not yet been changed.

Scope: fit and check the real-data models using the full recording. No training/validation/test split is required for these retrospective diagnostics. This supersedes the earlier holdout recommendation. Model checking on the data used for fitting is legitimate; posterior predictive checking is an established example ([Gelman, Meng, and Stern, 1996](https://www3.stat.sinica.edu.tw/statistica/j6n4/j6n41/j6n41.htm)). The present plug-in mark-rank statistic is not automatically a posterior predictive check or a calibrated fitted-model test merely by analogy. Keep its mathematical definition and state which calibration claims are supported.

Keep the current diagnostic formulas for this revision. Define their roles precisely: HPD overlap describes a selected geometric compatibility relation, the predictive rank measures mark surprise under the predictive model, and KL measures distributional difference. Passing these checks does not certify accurate or calibrated state inference. Developing a new rate-corrected statistic, coordinate-invariant overlap, or smoothing diagnostic should be a separate methods project.

Plan review: the core priorities are appropriate, but the manuscript revision must cover every affected section and figure caption. Implement the text changes in two passes: correct existing mathematical and interpretive statements first; integrate changed methods, numbers, figures, and conclusions once the supporting code and analyses are complete. The detailed section-by-section scope is in item 4. Review evidence from the current revision remains evidence about that revision; promote new numerical claims into the manuscript through reproducible summaries rather than copying exploratory review results directly into prose.

The required analysis scope is the broadening/rarity assessment on the full recording, an exactly matched simulation reference and sparse regime, threshold and coverage sensitivity, uncertainty assessment, and characterization of the history perturbation. Additional transition models, a broad parameter sweep, a full runtime study, and more recordings are conditional extensions when the corresponding stronger claims are retained. The paper can present a single-recording illustration with appropriately scoped conclusions.

1. **Correct the demonstrated errors and make the simulation contracts explicit.**

   Start in `simulation.py`, `decoding.py`, `figure03_protocol.py`, and `figure04_diagnostics.py`.

   - Correct the burst multiplier to operate at actual elapsed offsets 2–10 ms. Add a regression check that forces an initial event and verifies the subsequent conditional rates, including the boundaries.
   - Retain the history-modulated binned Poisson process and describe it accurately as suppressing the following bin. Remove the claim of a hard within-bin refractory interval. This is the recommended limited revision; enforcing a true refractory point process would require a different generator and validation.
   - Assimilate the first observation using the initial prior and the applicable observation model, then include first-bin events in diagnostics. Make initial-state and transition indexing explicit. Check a small example against a direct Bayesian calculation, including an informative first event and a zero-count bin with spatially varying total rate.
   - Align diagnostic event binning with the actual decoder convention at the terminal timestamp. Test boundaries against the decoder's bin assignments; the current data contain no terminal-timestamp event, but future inputs should agree.
   - Resolve the `simulate_walk` initial-position contract. Its current implementation takes an increment before returning the first sample. Prefer documenting that pre-step convention if it is consistent with the intended model and callers; change the implementation only if the intended contract is that the first returned value equals the supplied initial position. Make the exact-null generator's initial-state law explicit separately. Avoid changing random-number consumption solely to preserve an inaccurate sentence.
   - Correct the KL width argument, the claim of unchanged marginal rates under history modulation, the description of the fixed constrained remapping, and the undisclosed replay gain. Remove the promised power gain from substituting an ordinary smoother and the unsupported universal runtime ordering.

   Completion criterion: targeted regression checks and relevant implementation tests pass, and every stated time offset, initial condition, and rate schedule agrees with the implementation. Update result-dependent artifact checks from regenerated analyses at the final integration step. Existing figure values are expected to change; preserving them is not a correctness criterion. Describe corrected timing or new decoder behavior in the manuscript only after the implementation actually provides it.

2. **Characterize Figure 4's broadening effect using the full recording.**

   Fit both models using the full recording and compute their diagnostics on that same recording. This step consists of the following analyses:

   - Report predictive HPD-region sizes for Continuous and Continuous–Fragmented, both across the recording and at rescued events.
   - Recompute HPD overlap after adding fixed amounts of uniform mass to the existing Continuous predictions: `P'_k = (1-w)P_k + wU`, for example at weights 0.02, 0.05, 0.06, and 0.10. Show the fraction of original flags removed versus weight. Include the fully uniform prediction as a limiting example.
   - Repeat the comparison at the selected HPD coverage levels. Use the same observed events throughout, so changes have an interpretable denominator.
   - Describe where the discrepancies occur using behavior labels and an explicit selection rule for any candidate replay event.
   - Explain how much flag removal can result from broadening alone. The diagnostics can identify observed discrepancies and show how a model modification alleviates them; the rescue count alone does not certify more accurate state inference.

   The uniform-mixture analysis modifies the diagnostic input and measures the metric's response. Label it accordingly. A model with occasional uniform jumps, `T_w = (1-w)T_C + w U 1^T`, is an optional additional control if distinguishing transition mechanisms becomes necessary; it would use the same full-recording fitting and checking procedure.

   Completion criterion: predictive-region sizes, broadening curves, and coverage sensitivity computed on the full recording, with manuscript claims matching what those checks establish.

   Complete the firing-rate analyses in item 4 alongside these calculations, so the Figure 4 revision has both geometric and mark-space interpretations. Use the canonical cache with memory mapping and bounded chunks; the existing cache is large, and these additions do not by themselves require refitting its models. Save analysis code, parameters, and summary outputs for every new reported result.

3. **Rebuild Figure 3 around exact nulls and independent calibration.**

   Work in `figure03_simulation.py`, `figure03_protocol.py`, and `figure03_summary.py`.

   - Add a discrete state generator that samples from the decoder's transition matrix and initial-state law. Use the same observation model for a truly matched reference. Retain the reflected continuous walk only as an explicitly approximate benchmark, with grid sensitivity if it remains central.
   - Add an exactly matched low-information regime. Match both the initial law and transition; the existing stationary sparse trajectory with a random-walk decoder cannot serve as a fully matched null. Preserve the existing demonstration with accurate labeling if it remains useful.
   - Use independent simulated realizations to set thresholds and assess baseline error rates. This is Monte Carlo validation of the method, not withholding observations from the recorded data used for fitting and checking. Do not calibrate the null on deterministic recovery ramps.
   - Compare attainable operating points and threshold curves. HPD discreteness may prevent an exact 1% or 5% flag rate; report its realized rate and ties instead of claiming nominal equality. Show predictive-rank calibration against the super-uniform bound under the specified event-sampling law, rather than expecting uniform p-values or 5% within every cell subgroup. Keep per-realization fractions and pooled event fractions distinct; a random denominator and serial dependence require an explicit interpretation of any empirical error-rate summary.
   - Repeat HPD coverage at 50%, 80%, and 95%, using independent calibration at each coverage. Keep the chosen primary setting fixed before examining evaluation outcomes.
   - Vary residual ordinary activity in the sparse control, including scales 0, 0.005, 0.01, and 0.05. Report all-event and sparse-cell denominators separately. Include heterogeneous gains and a lower-rate setting, rather than relying on five similarly probable active marks.
   - Add a history control with approximately matched marginal rates. Estimate any gain adjustment on separate calibration simulations, and report remaining rate differences. Include a coherent wrong mapping as a counterexample to equating internal agreement with correct state interpretation. Treat several additional remappings and milder drift levels as targeted sensitivity extensions if retaining broad claims about robustness across those perturbations.
   - Retain the represented replay trajectory as an output. Report accuracy against it separately from physical-position discrepancy.
   - Report filtered-posterior coverage and region size alongside point error, and retain predictive-region size for interpreting the diagnostic itself. Distinguish these two distributions in the summaries and captions. A distributional score against known simulated states can supplement these quantities if it informs a retained accuracy claim. For continuous trajectories, state how truth maps to grid cells. Display paired realization distributions and event counts, not medians alone.

   Use ten seeds to validate the revised pipeline, then regenerate the main 100-realization analysis and the selected sensitivity settings. The pilot chooses whether the pipeline works, not which settings produce favorable findings. Do not launch a large Cartesian parameter sweep before the principal comparisons are stable.

   Completion criterion: independent matched-null calibration and evaluation, clear control labels, reproducible sensitivity results, and uncertainty summaries sufficient to assess any claimed advantage. If formal false-positive control for the fitted real-data procedure is claimed, assess it by simulating data under a specified null, refitting parameters in each replicate, and rerunning filtering and diagnostics. The simulated state/covariate protocol must be consistent with the fitting procedure, and any relevant data-dependent selection must be reproduced. Such a parametric-bootstrap assessment is approximate and needs no split of the observed recording. Refitting simulated samples is standard when calibrating goodness-of-fit statistics with estimated parameters; see the [SciPy algorithm description](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html) for the generic principle, not a ready-made state-space implementation. If these flags remain descriptive, state that the exact rank calibration result assumes the specified predictive law and do not claim an automatic nominal error rate for the complete fitted procedure.

4. **Revise the complete manuscript, including equations, figures, and supporting text.**

   Add a compact Figure 4 supplement showing per-unit firing rates versus flag fractions, event contributions by rate group, a descriptive constant mark-frequency baseline estimated from the same recording, and comparison across running and independently identified replay where such labels exist. Report both the fraction of units affected and the fraction of events contributed. Define any unit-quality criterion independently of diagnostic results; rerun fitting and prediction when changing the analyzed population.

   Keep the predictive probability and rank formulas. Do not divide by a marginal rate and call the result a p-value, or remove rare units merely to improve the plot. If quality metrics are unavailable, state the limitation and use transparent rate and split/merge sensitivity checks rather than claiming curated-unit validation.

   Use this manuscript revision map. The line links identify the current source; retain the corresponding LaTeX labels as locations shift. "Text now" means the correction follows from the existing audit. "After code/analysis" means the final wording or numerical result depends on completed implementation and regenerated artifacts.

   | Area | Required manuscript changes | When |
   | --- | --- | --- |
   | Title, keywords, and [Abstract](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:84) | Keep the neural-decoding focus. Qualify goodness-of-fit as local diagnostics of selected model–data relationships. Identify the distinct roles of HPD, predictive ranks, and KL. Describe the empirical example as a recording; avoid promising detection of every fault, reliable inference whenever flags are absent, or guaranteed improvement from smoothing. Make any headline outcome agree with the revised Results. Review the title for fit to this scope; a rename is not required. | Scope wording now; final outcome sentences after analysis. |
   | [Introduction](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:91), `sec:intro` | Preserve the scientific motivation and explain the selected nesting criterion as a useful target. Correct the claim that cross-validation, residuals, and other checks necessarily yield only a session-wide number. Distinguish the contribution from existing local predictive checking. Remove the implication that diagnostics identify which component failed without contextual evidence, and that divergences flag every nonzero difference regardless of threshold. | Text now. |
   | [State-space Methods](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:107), `sec:methods` and `sec:statespace` | Define full-bin observations separately from individual marks. Make initial prior, first observation, and time indexing agree with the decoder. Distinguish predictive, filtered, and smoothed distributions. State that parameter estimates are treated as fixed in the filtering equations and are estimated from the full recording in the real example. Preserve full-recording model checking as the intended use. | Notation now; first-bin details after code. |
   | [Event factorization and likelihood](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:128), `sec:metrics` | Preserve the correct Poisson factorization and rate/count convention `m=lambda*dt`. Define `Q` as a normalized single-event intensity factor, distinguish it from the full observation likelihood and posterior, and specify the spatial reference measure and positive finite normalization requirement. Explain the common prediction for same-bin events and the omitted count/exposure information. State the equal-bin grid approximation. | Text now. |
   | [HPD definition](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:154), `sec:hpd` | Define region coverage separately from the diagnostic flag cutoff. Specify density ties, discrete region sizes, and the overlap coefficient. Explain that nesting and a uniform prediction can produce overlap 1 without sharp state inference. Retain 95% as the main setting if that remains the protocol, and describe the selected coverage sensitivity. | Definitions now; sensitivity results after analysis. |
   | [Predictive ranks](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:168), `sec:predcheck` | Preserve event weighting and the ratio of integrated intensities. Separate mark probability from its inclusive rank-tail p-value. Replace the paragraph claiming broad/nested state distributions generally produce high p-values with the precise mark-surprise interpretation and a rare-mark counterexample. State super-uniformity under the specified mark law, discrete ties, and the qualification for fitted parameters. Distinguish a descriptive real-data cutoff from established false-positive control. Continuous-mark ranks require a specified feature coordinate/reference measure. | Text now; fitted-procedure calibration only if that inferential claim is retained. |
   | [KL direction and interpretation](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:221), `sec:kl` | Keep the implemented KL direction from `P` to `Q`. Correct the assertion that this direction avoids a broad likelihood: for centered Gaussians it equals `log(s_Q/s_P) + s_P^2/(2*s_Q^2) - 1/2` and grows with width mismatch in either direction. Explain direction-dependent weighting and possible infinite values when support is missing. Separate raw divergence size from a null-calibrated decision rule. | Text now. |
   | [Simulation design](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:242), `sec:simstudy` and `sec:generative` | Describe the exact matched reference, initialization, transition discretization, boundaries, and observation rates. Correct hard-refractory language, actual burst offsets, changed marginal rates, and the fixed constrained permutation. Disclose replay's fourfold gain and supplied rate overrides; distinguish its speed cap from realized speed. Label the deterministic replay and sparse trajectories by their actual role rather than declaring their dynamics exactly matched. Exclude the deterministic approach from a claimed matched-null comparison. Define interval endpoints and phase counts consistently. | Existing misdescriptions now; final protocol after code and analysis. |
   | [Simulation Results](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:277), `sec:simresults` | Replace broad conclusions drawn from low mean error or zero flags with results on error, coverage, and region size. Report realized baseline rates, thresholds and ties, sparse-event counts and denominators, paired realization distributions, and selected sensitivities. Distinguish physical position from represented replay position. A component label identifies the experimental intervention, not an inference performed by the metric. Avoid calling KL flags false positives against a fully correct model until the matched-null comparison supports that description. | Remove unsupported interpretations now; quantitative rewrite after analysis. |
   | [Real-data Methods](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:296), `sec:realdatamethods` | Explicitly describe fitting and checking on the full recording, including immobility, unit eligibility, zero-event units, observation-model sharing, mode marginalization, and identical valid position bins. Report recording duration and event counts from summaries. Give the displayed-window selection rule and any behavior thresholds. Explain that automatic curation and mark splitting/merging can affect the diagnostics. State both cutoffs separately even when numerically equal, and avoid equating the HPD cutoff with a significance level. | Procedure and interpretation now; added summaries after analysis. |
   | [Real-data Results](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:322), `sec:realdataresults` | Describe changes in the observed diagnostics rather than claiming the true trajectory is known during replay. Retain exact rescue/new-flag counts with clear denominators. Add predictive-region sizes, uniform-mixture curves, coverage sensitivity, and rate-stratified event contributions. Distinguish a candidate replay window from an independently labeled replay set. Define a larger p-value as a less extreme density rank; it does not by itself compare absolute mark probabilities across models. | Interpretation now; new numerical comparisons after analysis. |
   | [Discussion](/Users/edeno/Documents/GitHub/statespacecheck-paper/manuscript/main.tex:338), `sec:discussion` | Explain complementary diagnostic targets and use simulation evidence to describe strengths and limitations. Remove unsupported claims of uniformly lower error rates, universal computational cost ordering, automatic smoothing power gains, and inevitable failure of all f-divergences. Explain broad-prediction tolerance, rarity, joint-bin and temporal limitations, coherent wrong-state interpretations, and coordinate dependence without claiming complete blindness beyond the counterexamples. Keep model refinement as a hypothesis informed by localized discrepancies and scientific context; fewer flags alone cannot certify accurate inference. Qualify clusterless and longer-window extensions as requiring further development and validation. End with the demonstrated contribution and its scope. | Text now; final synthesis after analysis. |
   | All four figure captions and schematic text | Figure 1 must distinguish the full Bayesian observation update from the selected event factor being checked. Figure 2 must match HPD normalization, inclusive rank direction, event weighting, exact finite summation, and KL direction. Figure 3 must state the actual calibration samples, thresholds, phase labels, denominators, uncertainty summaries, and which trajectory defines error. Figure 4 must state full-recording scope, selection of the detail window, mode marginalization, actual cutoffs, and the descriptive meaning of rescue. Correct caption references whenever panels or supplements change; review generated diagram labels as well as LaTeX captions. | Definitions now; result-dependent captions with regenerated figures. |
   | Bibliography, availability, and supporting material | Verify primary sources for the overlap coefficient, predictive ranks/model criticism, divergence comparisons, and coordinate dependence; correct any weak or mismatched citation. Review source metadata before adding references. Document the five input exports and raw-data processing gap, exact code/data versions, and final release DOI. Align `README.md` and `manuscript/README.md` with the manuscript's diagnostic scope. Check author metadata, contributions, acknowledgements, and ethics text for consistency with verified project information. | Reference and documentation pass now; archival details at release. |

   Add a compact mathematical explanation in Methods or a supplement for the inclusive categorical rank bound and the effect of a uniform component on 95% HPD size. A short counterexample table should cover equal state distributions with a rare mark, uniform prediction, and incompatible joint-bin events. These are explanations of the existing methods, not additional metric proposals. Keep the exposition proportional: state the operational definitions and limitations clearly, then use the Results to show their practical consequences.

   Apply one editorial pass across the complete manuscript after the scientific rewrite: consistent terminology, state/observation notation, units, temporal indexing, figure references, and distinction between a descriptive flag and a formal test decision. Remove repeated motivation and unsupported absolutes; explain each result before interpreting it. Preserve the motivation for local checks so the revision remains a coherent methods paper rather than a list of limitations.

   Route every new headline quantity through analysis summaries and the reporting emitter in `reported_values.py`. Update `reported_values.tex` by running the emitter, including changed thresholds, phase summaries, and any new real-data quantities. Maintain the repository's rounding policy and exact event denominators. A pilot result or counterexample must not be described as a completed 100-realization benchmark.

   Completion criterion: every manuscript section, equation explanation, figure caption, and relevant supporting document has been reviewed. Every statement of superiority, calibration, harmless misspecification, or successful refinement is supported by an appropriate analysis or replaced by a descriptive statement. Each planned revision must have a recorded location and status, and every data-dependent statement must agree with the final artifacts.

5. **Regenerate and make the revised evidence reproducible.**

   Implement cache dependency changes before the expensive reruns in items 2 and 3: include the fitting and diagnostic configuration, any data-selection masks, diagnostic implementation/version, and source/data provenance. Prefer separately caching fitted predictions and derived diagnostics where practical, so a diagnostic change does not require unnecessary decoder fitting. This infrastructure work precedes final regeneration even though release integration is listed here.

   After the scientific configuration is settled, regenerate figures, summary JSON, manuscript macros, and viewer artifacts from that configuration. Verify event alignment and flag totals between the canonical cache, summaries, and viewer; the current viewer mismatch must disappear. Follow the repository build chain: run `uv run python scripts/emit_reported_values.py`, then `make -C manuscript`. Check missing references/citations, equation layout, figure readability, supplementary links, and unwanted page breaks in the rendered PDF. Run required repository checks and document any omitted long-running check. For an earlier text-only pass, compile and inspect the manuscript without treating that as validation of unimplemented analysis changes.

   Provide an executable procedure for the five input exports or archive the exact permitted inputs with checksums. Record environment versions, seeds, fitting and data-selection definitions, and commands. Finish the release DOI when the archival release exists. Add sessions or animals if retaining general empirical claims; otherwise explicitly retain a single-recording case-study scope.

   Completion criterion: one documented revision reproduces all reported results, with no unexplained artifact disagreement or unresolved placeholder presented as completed work.

Suggested implementation units are: (1) correctness, contracts, cache dependencies, and the first manuscript text pass; (2) full-recording Figure 4 broadening and rarity checks; (3) calibrated Figure 3 benchmarks and uncertainty; (4) the complete manuscript rewrite with regenerated figures and captions; (5) reproducible release. Run the Figure 4 broadening analysis early. Broad new metric development, smoothing implementations, and clusterless validation are follow-up projects unless those claims remain central to this manuscript. The plan is complete only when the manuscript-wide revision is integrated, rather than when the code changes alone are finished.

---

## Status (branch `manuscript/scientific-revision`, 14 September 2026)

Everything below was executed on this branch. Every number quoted here is
emitted by `uv run python scripts/emit_reported_values.py` into
`manuscript/reported_values.tex` from the four summary JSON files named in each
item; the manuscript quotes none of them literally.

### Item 1 — implementation errors and contracts: **done**

| Fix | Location | Regression evidence |
| --- | --- | --- |
| Burst multiplier applied at elapsed offsets 2–10 (was 3–11); suppression at offset 1; per-cell elapsed counter | `simulation.py::simulate_history_dependent_spikes` | `tests/test_simulation.py` forces an initial event and checks conditional means at every offset including both boundaries |
| First observation assimilated against the initial law; `t = 0` events enter the diagnostics | `decoding.py::update_step`, `decode_with_diagnostics` | `tests/test_decoding.py` compares one informative first event and a zero-count bin with spatially varying total rate against a direct Bayes calculation |
| Real-data event binning uses the decoder's `np.digitize(times, time[1:-1])` convention (terminal timestamp lands in the last bin) | `figure04_diagnostics.py::_get_spike_events_from_spike_times` | `tests/test_figure04_diagnostics.py` boundary cases; the current recording has no terminal-timestamp event, so no reported value changed |
| `simulate_walk` pre-step convention documented; exact-null initial law made explicit (`simulate_discrete_walk`, uniform initial law) | `simulation.py`, `figure03_simulation.py::_TrajectorySampler` | `tests/test_simulation.py`, `tests/test_figure03_simulation.py` |
| Place-field rates floored at `np.finfo(float).tiny` so narrow fields do not underflow to zero and produce infinite KL | `simulation.py::place_field_rates` | `tests/test_simulation.py`; surfaced by the continuous-walk sensitivity calibration |
| Manuscript: KL-width argument, history marginal-rate claim, constrained-remap description, replay gain (fourfold, disclosed), smoother power claim and universal runtime ordering removed | `manuscript/main.tex` | editorial pass, see item 4 |

Commands: `uv run pytest` (full suite), `uv run ruff format . && uv run ruff check . && uv run mypy src/`.

### Item 2 — Figure 4 broadening on the full recording: **done**

Both models are fitted on, decoded on, and checked against the full epoch
(`event_selection = "all_spikes_in_recording"`, 870,018 events). The decode
bundle (`data/intermediates/j1620210710_02_r1_fig4_cache.joblib`, 19 GB,
fingerprint `c30cead6…`) was reused memory-mapped; diagnostics were recomputed
into the new sidecar `…_fig4_diagnostics.joblib` (83 MB) and verified
bit-identical to the embedded legacy diagnostics.

New module `figure04_broadening.py`, recipe `figure04_supplement_generation.py`,
CLI `scripts/generate_figure04_supplement.py` (~6 min, bounded 50,000-event
chunks) → `manuscript/figures/supplementary/figure04_supplement_summary.json`
and `figure04_supplement.{pdf,png}` (Fig S2); `tests/test_figure04_broadening.py`
covers the analyses on a synthetic decode whose cached diagnostics are
computed from its own predictions. Results:

- 95% predictive region sizes (median bins of 248): Continuous 13 all / 8 at
  rescued events; Continuous–Fragmented 144 / 110; likelihood 95% region 228 (8% of the track outside it).
- Uniform mixture `(1-w)P + wU` applied to the Continuous prediction removes
  25% (w=0.02), 79% (0.05), 95% (0.06; more than the fitted alternative's
  92%), 100% (0.10) of the original HPD flags with at most 24 new flags. The
  manuscript labels this as a change to the diagnostic input, not a refit.
- Coverage sensitivity at 50/80/95% on the same events (Table
  `tab:realdata_broadening`).
- Behavior strata (4 cm/s): HPD flags enriched during immobility under the
  Continuous model; rank flags only slightly.
- Rate stratification: Spearman(unit rate, fraction of its spikes rank-flagged)
  reported per model; units <0.1 Hz contribute a small share of flags; 1–5 Hz
  units contribute most rank flags; a constant session-frequency rank baseline
  reproduces a stated fraction of the rank flags.

### Item 3 — Figure 3 exact nulls and independent calibration: **done**

`figure03_protocol.py` / `figure03_simulation.py` / `figure03_summary.py`:
10-phase ladder (matched-null baseline, remap, recovery, history with
rate-matching gain 0.512 estimated on seeds 5001+, recovery, replay (fourfold
gain, represented trajectory retained), recovery, drift, recovery, reflected
coherent wrong map, recovery, sparse heterogeneous low-information regime).
Trajectories in unperturbed phases are drawn from the decoder's own
column-stochastic transition matrix (`trajectory_model="discrete_matched"`);
thresholds come from 100 separate matched-null sessions (seeds 1001–1100),
evaluation from seeds 1–100. Pilot: 10 + 10 seeds; main: 100 + 100 (~4 min,
joblib). Sensitivity (`figure03_sensitivity.py`, ~22 min): HPD coverage 0.5 and
0.8 with their own calibration, sparse ordinary-activity scales 0.005/0.01/0.05,
sparse cells at half rate, reflected continuous walk.

Headline values (`figure03_summary.json`, schema 6): HPD threshold is exactly 0
(all tied events flagged, realized null rate 1.6%); KL threshold 4.2 (null
1.0%); rank cutoff 0.05 (null 2.6%, tails 0.35/2.6/4.7% at α = 0.01/0.05/0.10,
i.e. super-uniform). Median flag rates: matched null 1–3%, recovery 1–3%,
remap 19–24% (HPD IQR 12–52% across realizations; pooled 28–33%), history
1–2%, replay 1–3%, drift 8–14%, reflected map 1–3% (passes despite being
wrong), sparse regime 0/0/56% (HPD/rank/KL) by median, 0.99/1.4% pooled.
Continuous-walk sensitivity gives remap HPD 41% (vs 23% on the matched grid).

### Item 4 — manuscript revision: **done (first and integration passes)**

`manuscript/main.tex` rewritten section by section: Abstract, Introduction,
Methods (operational definitions; new subsection "What passing the diagnostics
does and does not establish" with counterexample Table `tab:counterexamples`,
each row reproduced by `tests/test_diagnostics.py::TestManuscriptCounterexamples`),
simulation design and results (one paragraph per condition), real-data methods
and results (full-recording scope, tables `tab:realdata_broadening`,
`tab:sensitivity_coverage`, `tab:sensitivity_sparse`), Discussion, all captions,
supplementary figures S1–S2 in their own section. Bibliography: weak
TV/Pearson citations removed; Liese & Vajda 2006, Simpson 1943, Gneiting et
al. 2007 added. 200 macros emitted; `tests/test_reported_values.py` verifies
every macro is used and the committed macro file matches the summaries.

### Item 5 — regeneration and reproducibility: **done except release items**

- Cache split (decode bundle + diagnostics sidecar keyed by decode fingerprint
  and a diagnostics fingerprint over config, `statespacecheck` version, and the
  docstring-stripped AST of the diagnostic modules) landed before the reruns
  (`figure04_cache.py`, schema 5 + diagnostics schema 1).
- Regenerated in order after the last source edit: `generate_figure04.py`,
  `generate_figure04_supplement.py`, `generate_figure03.py`,
  `generate_figure03_sensitivity.py`, then `emit_reported_values.py` and
  `make -C manuscript`.
- Viewer cache rebuilt (`python -m statespacecheck_paper.interactive.cache
  build --data-dir data --cache-dir data/cache --model both --force`). Flag
  totals in the parquet event tables equal the canonical
  `figure04_summary.json` confusions exactly: HPD 18,790 (17,289 + 1,501)
  Continuous and 1,677 (176 + 1,501) fragmented; rank 33,954 and 26,287. The
  earlier viewer/canonical disagreement is resolved.
- PDF inspected page by page (44 pages): no undefined references or citations,
  no float-too-large or overfull warnings, supplementary figures isolated
  before the back matter.

**Remaining (not done on this branch):** archival DOI placeholder; the raw
recording → five-export step remains external (archive the exact exports with
the checksums in `figure04_summary.json` or provide an executable export
procedure); the conditional extensions listed in `TODO.md` (uniform-jump
transition control, parameter sweep, runtime study, additional recordings,
parametric-bootstrap calibration of the fitted real-data procedure) were not
started and the manuscript claims none of them.

### Post-review fixes (14 September 2026, second pass)

Five issues raised on the branch, all fixed here:

1. **Reflected map did not mirror the sparse cells** — the reflected rate
   table is now the entire baseline table with its rows reversed (exact
   reflection on the symmetric grid), sparse cells and exposure term
   included; `tests/test_figure03_phases.py` checks the identity for both
   blocks. Seed 4's sparse spike at 32,129 ms goes from HPD 0 / KL 467 to
   HPD 1 / KL 0.30. Figure 3 and its sensitivity summaries were regenerated.
2. **Rate clip under-states KL** — `place_field_rates` now documents the
   clip as the observation model in use and states that the clipped KL is a
   lower bound on the exact-Gaussian value (test: width-2 field at 0,
   prediction at 100 → exact 1251.10 vs clipped 707.89). Rather than move
   the whole decoder to log-domain tables, the pipeline measures the clip's
   effect: `unclipped_event_kl_divergence` recomputes every event's exact KL
   and the summary records `kl_clip_impact` for the evaluated realizations
   and the calibration sessions (macros `SimKlClipEvents`,
   `SimKlClipDiffering`, `SimKlClipMaxDifference`, `SimKlClipFlagChanges`,
   `SimKlClipNullDiffering`, quoted in the simulation Methods). With the
   reflection fix in place, 54 of 2,126,394 evaluated events differ (max
   1.02 nats, all with clipped KL ≥ 7.35 > threshold 4.22), no flag decision
   changes, and no calibration event differs.
3. **Rank check described as nesting-tolerant** — the Introduction and
   Discussion now reserve the nesting-tolerant geometric criterion for HPD
   overlap and describe the rank check as predictive mark surprise, which
   includes relative-rate effects.
4. **"Rate errors cannot trigger flags"** — the Methods and Discussion now
   say the diagnostics do not *directly* assess the exposure term and are not
   comprehensive rate checks, and note that a common gain error reaches the
   prediction through the filtering updates (HPD overlap 1 → 0 after a silent
   bin in a two-state example).
5. **Continuous-walk sensitivity explanation** — the text now states that
   the alternative generator changes both the effective step and the
   initialization (endpoint vs uniform), presents the larger-step account as
   a hypothesis, and replaces "within about a percentage point" with the
   actual values (null and drift within a fraction of a point; sparse KL
   quoted via `\SensContinuousSparseKl`).

The comparability caveat for the 50%/80% coverage rows (null HPD rates of
about 38% and 11%) stands as written: the manuscript makes no claim of
improved sensitivity at comparable false-positive rates.

