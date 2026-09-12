# Agent implementation guideline: controlled KG reasoning playground and idea verification

Version: 2.0 · 10 September 2026  
Status: design specification; no implementation, experiment, or hypothesis is represented as verified.

## 1. Mission and revision scope

Build a reproducible playground for GNN-based knowledge graph completion that separates three places where information can disappear:

1. **Path composition:** different ordered relation histories become indistinguishable while messages are constructed.
2. **Neighbourhood compression:** distinguishable incoming evidence loses query-relevant distinctions when merged into a state.
3. **Neighbour selection:** sampling, pruning, or budget allocation omits evidence required for an answer.

Primary question: **How do branching and reconvergence affect the relational distinctions preserved by a GNN, and can adaptive aggregation preserve those distinctions at a lower measured computational cost?**

Atomic-score quality, decoder restrictions, optimization, depth and input access are additional controls. They must not be silently assigned to one of the three mechanisms. The mechanisms can interact; accuracy differences are not an additive causal decomposition.

This revision promotes composition from an optional extension to a core diagnostic, adds compression and neighbour-selection experiments, and makes expressiveness under computational budgets the main study. The version 1 shared-witness/max-product benchmark is retained in Appendix A as a supporting executor track. Candidate-witness retrieval in that track is not equivalent to selecting GNN neighbours, and its certification does not transfer to a GNN.

Success is a trustworthy experiment that can reject a proposed method. Clifford networks are an optional, controlled parameterization whose value must follow from the diagnosed failure.

## 2. Agent operating rules

- Inspect the project and applicable AGENTS.md before editing; preserve unrelated work. If no repository is supplied, use a clearly named project in an authorized working directory.
- Implement in stages and pass each acceptance gate before advancing. This document authorizes a design, not a claim of completed code.
- Develop on CPU first. Follow the user's existing resource authorization; do not initiate unapproved paid resources, long GPU runs, publishing or pushes.
- Prefer available dependencies and record versions actually tested. Record Python, framework, hardware, dtype and determinism settings; never invent version pins.
- Separate train, validation and final evaluation. Do not tune labels, thresholds, seeds, difficulty or budgets on final test outcomes.
- Keep truth, witnesses and diagnostic oracle policies outside model inputs. An oracle intervention runs only in an explicitly labelled diagnostic mode.
- Log failures and null results. Distinguish planned, implemented, tested and reproduced work in every handoff.
- Compare relevant prior work before novelty claims. Low training accuracy alone is not evidence of non-representability.

## 3. Architecture scope and claim levels

| Family | Study location | Required control |
| --- | --- | --- |
| TransE, DistMult, ComplEx, RotatE and other triple scorers | Atomic scores, relation operators or decoder | Their usual forms do not perform neighbourhood aggregation; do not report them as complete GNN/query executors |
| R-GCN and CompGCN with a decoder | Entity representations, relation-dependent messages and aggregation | Fix the decoder when attributing an effect to the encoder |
| Source/query-conditioned GNN, initially an NBFNet-style harness | Propagation, local merging and selection for a source/query | Fix initialization, depth, reverse-edge policy and decoder across component interventions |
| A*Net/AdaProp-style adaptive propagation | Which nodes/edges receive computation | Include selection overhead and compare with full propagation |

Use the conditional GNN as the main integration harness. A minimal harness is not a reproduction of NBFNet; use that name for results only after reproducing its relevant configuration. Add R-GCN/CompGCN as architecture comparisons after the diagnostic is stable.

Label every claim as one of:

- **Formal distinction:** a proof or exhaustive counterexample for an explicitly specified model family, input features and task.
- **Fixed-budget learnability:** repeatable learning outcomes at stated width, depth, training budget and precision.
- **Computational benefit:** a measured accuracy/error versus runtime/memory improvement, including all required work.

Relational node encoders and conditional link predictors have different expressiveness characterizations. Choose the relevant relational WL framework; do not state that all KG GNNs have an identical 1-WL bound. Merely replacing real vectors with finite Clifford multivectors does not escape the applicable message-passing bound. See [relational link-prediction theory](https://arxiv.org/abs/2302.02209).

## 4. Freeze graph and task semantics

### 4.1 Primary diagnostic: ordered relational support

Use directed relation-labelled graphs. A query specifies a source x, a query relation q and a candidate y. Start with a registered synthetic rule:

    q(x,y) := exists z: r(x,z) AND s(z,y)

The assignment defines the complete label rule in this diagnostic, not an assumption that real KG rules are biconditionals. Query/derived q edges are excluded from model-visible support, together with inverses that reveal them. For this track, labels come from the exact support graph: absence of a required bounded route is false by construction.

Generalize using a registered finite language of relation sequences and bounded walks of length at most L. In acyclic fixtures, these are paths. If testing simple paths, repeated vertices or cycle rules, implement and name those semantics separately. A symbolic evaluator must check the declared rule independently of neural representations.

Distinguish the ordered sequence along a path from the unordered collection of neighbours and from the unordered atoms of a logical conjunction. Reversing r then s can change the answer; permuting the enumeration of neighbours must not.

### 4.2 Shared-witness supporting track

Retain the one-variable conjunctive query benchmark and frozen max-product executor from version 1 in Appendix A. It tests identity alignment and omitted witnesses, providing a separate diagnostic for lost dependencies. A shared-witness join is not automatically a path-order task or a GNN compression theorem.

### 4.3 Incomplete-graph extension

Only after exact-input diagnostics pass, introduce learnable missing facts via a documented generative rule or allowed attributes. Score quality is a separate axis. Missing observed facts are unknown in this extension; labels use an evaluator-only truth graph. Freeze the scorer for paired executor studies. Do not expect independently hidden random edges to be predictably recoverable.

## 5. Playground structure and interfaces

Suggested package: `kg_reasoning_lab`. These paths and commands are implementation contracts.

| Path | Responsibility |
| --- | --- |
| pyproject.toml and lockfile | Tested dependencies and entry point |
| configs/smoke.yaml, pilot.yaml | Registered stages, fixtures, budgets and seeds |
| src/kg_reasoning_lab/data/ | Graphs, queries, matched instances, splits and audits |
| src/kg_reasoning_lab/oracles/ | Symbolic bounded-walk rules, joins and soft reference executor |
| src/kg_reasoning_lab/composition/ | Ordered message operators and exact sequence reference |
| src/kg_reasoning_lab/compression/ | Sum, mean, nonlinear sum, attention and grouped aggregation |
| src/kg_reasoning_lab/selection/ | Full, random, fixed-budget, adaptive and diagnostic oracle policies |
| src/kg_reasoning_lab/models/ | Conditional GNN harness and architecture adapters |
| src/kg_reasoning_lab/scorers/ | Exact, synthetic and learned atomic sources |
| src/kg_reasoning_lab/executors/ | Appendix A shared-witness executors |
| src/kg_reasoning_lab/traces/ | Boundary snapshots, replay and intervention checks |
| src/kg_reasoning_lab/evaluation/ | Quality, information probes, paired comparisons and costs |
| tests/, manifests/, reports/ | Correctness checks, provenance and evidence-linked reports |

```text
generate(config, seed) -> GraphBundle
make_queries(bundle, config, seed) -> QueryBatch
symbolic_oracle(truth_graph, queries, semantics) -> LabelsAndEvidence
select(frontier, visible_graph, query, policy, budget) -> SelectionResult
compose(selected_edges, source_states, relations, query) -> MessageBatch
compress(messages, receiver_index, query, budget) -> StateBatch
predict(states, query, decoder) -> Predictions
capture_boundary(stage, inputs, outputs, manifest) -> Trace
replay(trace, replacement_component, config) -> DiagnosticResult
evaluate(predictions, evaluator_only_truth, costs) -> Metrics
```

This execution order is illustrative: a selector may require cheap provisional messages before selection. Declare the actual order and charge all provisional computation. The three mechanisms are analytical locations, not a mandatory global sequence.

Every message carries receiver/source mapping, relation and layer. Full path histories are optional debug data for small graphs, not automatically supplied to a learned model. Exact symbolic sequence references are bounded diagnostic controls; disclose their additional state and computation.

SelectionResult records inspected and retained edges/nodes, rejection reason if available, selection cost and budget exhaustion. DiagnosticResult records intervention location, score, pre/post state hashes, component hashes, cost, termination reason and optional bounds. Unsupported bounds are null.

### Data separation

- Model-visible: permitted support triples, local entity keys, relation map, query, registered attributes and source marking.
- Evaluator-only: hidden truth, labels, supporting routes, witness sets, corruption masks, matched-pair IDs and oracle selection decisions.
- Manifest: schema/generator version, independent graph/query/corruption seeds, split, rule semantics, feature policy and hashes.

Entity IDs align arrays but are not cross-graph learned identity features in inductive tests. Do not leak labels through filenames, ordering or IDs. Preserve the same permitted inputs across learned comparisons.

## 6. Topological fixtures and controlled sweeps

| Fixture | Main intervention | What must be checked |
| --- | --- | --- |
| Ordered chains | Swap relation order at fixed length | Both labels represented; simple order fixtures are not evidence of difficulty for every GNN |
| Branching then reconvergence | Vary branch count or merge depth | Distinct versus repeated relation histories; useful versus redundant evidence |
| Rare support among distractors | Vary distractor count and selection rank | Useful path exists before pruning; no hidden oracle signal enters selector |
| Repeated merge/split structures | Vary number of merges at fixed or reported depth | Information loss is localized; do not silently change path length and degree together |
| Long routes through narrow cuts | Vary distance or bottleneck separately | Task requires distant information; high degree alone is not over-squashing |
| Shared-witness joins | Vary overlap and witness rank | Entity alignment and common-witness truth follow Appendix A |

Distinguish topology from relation labelling. Use paired interventions on each while fixing the other where feasible. Include redundant branches and genuinely different histories as separate conditions. Sweep one factor first; add registered interaction experiments later.

Start with short chains and tiny reconverging graphs. Simple r-s versus s-r examples may be solved from the last relation alone. Before claiming a multi-hop difficulty, construct harder matched pairs, audit endpoint degrees, incident relation counts, path lengths and other permitted nuisance statistics, and report residual differences. Do not claim all statistics or WL colours match without verification.

Split by generated graph family instance; keep renamed variants and matched examples together. Hold out entity identities, graph sizes and, in separate experiments, motif arrangements. An unseen-ID split requires an inductive model. Train nuisance-feature baselines and freeze final test rules before model selection.

## 7. Idea verification matrix

| ID | Falsifiable hypothesis | Isolation experiment | Evidence against the hypothesis or proposed remedy |
| --- | --- | --- | --- |
| H-P | A specified composition family loses an ordered distinction needed by the task | Full evidence, no lossy merging, fixed initialization and readout; replace only the composer | Distinction already survives; failure is in readout/training; ordinary matched operators recover equally well |
| H-C | At fixed available inputs and budget, a specified compressor loses a useful distinction present in incoming messages | Freeze and replay identical MessageBatch objects; full selection; replace only compression | Incoming evidence was already indistinguishable; a suitable nonlinear sum solves it at comparable cost; improvement comes only from extra state |
| H-S | A specified selection policy drops required support, explaining errors relative to full propagation | Frozen weights/scorer and depth; change only selection; log support coverage | Support was retained but answer still fails; full propagation also fails; gains vanish when selection overhead is charged |
| H-A | Adaptive capacity/grouping gives better quality–cost trade-offs on diverse evidence than fixed allocation | Compare at matched measured budgets on held-out graphs and topologies | Fixed allocation or existing adaptive propagation matches it; overhead erases benefit |
| H-G | Clifford composition supplies a useful inductive bias for the established mechanism | Cross composer and compressor choices; compare real, quaternion and Clifford variants under matched resources | Gain vanishes with coefficient/compute matching or is entirely due to grouping, conditioning or optimization |

Before running each study, record: claim level, exact model family, assumptions, intervention, fixed inputs, expected observable, competing explanations, falsification criterion and resource budget. Register a meaningful target effect or decision rule from pilot/validation data; never choose it using the final test.

### 7.1 Composition diagnostic

For tied linear operators acting on column states, no intervening nonlinearity and identical h0:

    h_rs = T_s T_r h0
    h_sr = T_r T_s h0

Commuting T_r and T_s force equality. This is an algebraic statement about this restricted family. It is not a theorem about all elementwise-message GNNs: layer-specific operators and intermediate updates change the argument.

Compare exact sequence tracking, commuting diagonal operators, unconstrained matrices and suitable structured noncommutative operators. Include a noncommuting case whose action on the chosen h0 actually differs; a nonzero commutator alone does not ensure it acts nontrivially on that state. Test tied versus layer-specific parameters and nonlinear updates separately.

Use a path-only readout with no alternative endpoint/ID shortcut for the restricted diagnostic. Then test the full GNN under its declared features and readout; do not transfer the path-only theorem without checking those assumptions.

### 7.2 Compression diagnostic

Save incoming messages before merging and replay them into each compressor. Establish that the task-relevant distinction is available before claiming it is lost afterward.

A deterministic scalar fixture is M+={1,3} and M-={2,2}, with a declared diagnostic target of detecting a message equal to 1. Both sum to 4 and have mean 2. A second moment or suitable nonlinear feature map can distinguish them. This is a fixed-message unit fixture, not a proof about trained KG encodings or a claim of real-graph difficulty.

Compare raw sum/mean, learned sum of transformed messages, attention, and a budgeted grouped variant. A downstream MLP alone cannot separate exactly equal pooled states, whereas transforming individual messages first can change the result. Include an explicit evidence-retaining control on tiny instances, with its larger memory disclosed.

Do not call sum aggregation universally lossy: suitable learned multiset encodings can make it injective under specified assumptions. Do not assert that more paths than dimensions forces integer path-multiset collisions. Separate exact collisions, numerical near-collisions and probe failure. See [GIN expressiveness](https://arxiv.org/abs/1810.00826).

### 7.3 Selection diagnostic

Start with full-neighbour propagation, uniform sampling and a fully specified fixed-budget policy. Define whether the budget is per receiver, per layer or per query; specify replacement sampling, tie handling and whether pruned states persist.

For small positive fixtures, retain all edges needed by at least one valid supporting route with an evaluator-only oracle policy. Compare with a same-budget random policy; pad the oracle to equal retained work where feasible. Record cases where the budget cannot retain a full route. This is a privileged diagnostic control, not a deployable competitor.

Run two modes: frozen candidate/message replay to isolate immediate omission, and frozen-parameter GNN rollout to observe downstream state changes. Their costs and outputs answer different questions. Removing edges changes later messages even when model weights are fixed; do not claim all incoming tensors are identical in rollout mode.

A full-graph model is a reference, not an oracle for truth. If retained support does not yield a correct answer, investigate composition/compression/readout. If full propagation fails, omitted evidence cannot by itself explain the error. Actual attention/pruning weights are not automatically causal explanations.

## 8. Interventions and baseline discipline

Run one-component interventions before training combined variants. Freeze upstream values and register downstream training/readout access for replay studies. If a replacement changes dimensionality, use and disclose an adapter, or compare separately trained decoders with equal protocols; do not call a changed downstream function a pure intervention.

Then run a small crossed study of composer × compressor × selector. Report interactions and paired outcomes, not a decomposition summing to total error. End-to-end retrained variants belong in a separate learnability study.

Required initial controls are an exact symbolic evaluator, full evidence, raw sum, nonlinear sum, random selection and a noncommutative real-matrix composer. Add quaternion/Clifford after these work. Published comparisons should include suitable R-GCN/CompGCN and NBFNet configurations, and A*Net/AdaProp/TAGNet for efficiency where tasks and access match. Consider PN-GNN for claims about richer path-neighbour expressiveness. Unsupported tasks must be stated, not silently adapted and attributed to a paper.

## 9. Candidate intervention: allocate capacity to useful diversity

Hypothesis: a fixed state budget may be spent more effectively by retaining a few distinct query-relevant message groups and pooling redundant evidence within groups. Degree alone need not predict the required capacity.

Begin with a fixed number of groups, then compare adaptive allocation under the same total per-query budget. A learned routing function may use only permitted states, relation types and query information. No witness labels or future hidden edges enter it. Define empty-group handling, count preservation, allocation overhead and behaviour when the budget is exhausted.

Use permutation-invariant set aggregation within groups and a final representation insensitive to arbitrary group naming, or define stable semantic slots with permutation-equivariant routing. Do not multiply an arbitrary neighbour ordering merely to use a noncommutative algebra.

Separate the effect of grouping from the choice of algebra. There is no speed claim until routing, preliminary scoring, group state and all inspected messages are included. A selector that scores every edge may reduce later work without reducing its initial scan.

## 10. Clifford evaluation and previous-paper bridge

Specify the algebra signature, retained grades, total real coefficients, multiplication convention, action on states, activation, initialization and decoder. Full Cl(p,q) has 2^(p+q) basis coefficients; a truncated representation needs its own closure/projection specification.

Compare real matrix, diagonal/complex and quaternion alternatives appropriate to the claimed benefit. Match total real coefficients and parameter counts in one comparison, and report accuracy at matched measured compute/memory in another. These resource constraints need not be simultaneously attainable; disclose the trade-offs.

Clifford multiplication may distinguish ordered compositions, but addition still pools coefficient vectors. No intrinsic guarantee prevents cancellation, preserves separate witnesses, solves pruning or improves runtime. Small-dimensional rotor constructions can overlap complex/quaternion alternatives; noncommutativity alone is not novel. Restrict norm-preservation claims to the exact action and signature implemented. Invertibility alone does not represent general many-to-many relations.

For the earlier planar/ladder paper, obtain the actual manuscript, generator and result definitions before reproducing its diagnosis. Connect the thesis through architecture-specific restrictions exposed by graph structure. Keep cycle-pressure and message-merging experiments distinct; do not assume every KG cycle requires a relation-operator product equal to identity.

## 11. Correctness and verification gates

| Check | Acceptance criterion |
| --- | --- |
| Independent truth | Symbolic walk/rule evaluator agrees with independent brute force on tiny graphs |
| Ordered actions | Hand-calculated products, tied commuting equality and a noncommuting separating fixture agree |
| Compression fixtures | Known raw-sum collision reproduced; separating positive control succeeds |
| Boundary replay | Unchanged components reproduce deterministic reference outputs and trace hashes |
| Full selection | Unlimited policy retains all eligible neighbours and matches full propagation |
| Omission fixture | Registered small example loses all support under fixed selection and retains support under feasible oracle selection |
| Permutations | Entity relabelling remaps outputs; neighbour/edge enumeration leaves outputs invariant within tolerance |
| Meaningful order | Relation order is preserved along walks; conjunction remains atom-order invariant in Appendix A |
| Budget accounting | Inspected, scored, retained and repeatedly visited edges are counted separately; overflow is explicit |
| Leakage/splits | Oracle data inaccessible to ordinary training; no paired/renamed instance crosses splits |
| Numeric cases | Empty neighbourhoods, zeros, ties, isolated nodes and near-threshold decisions are specified |
| Supporting executor | Appendix A bound, binary-equivalence and full-enumeration tests pass when that track is enabled |

Correctness checks target mathematical or implementation invariants, not whether the proposed method wins. Keep trivial diagnostic fixtures separate from held-out learning benchmarks. Log and stop dependent stages when an oracle or semantic check fails.

## 12. Pilot configuration and cost accounting

Suggested starting settings, not validated resource estimates:

```yaml
stage: diagnostic_smoke
tracks: [composition, compression, selection]
graph_sizes: [16, 32]
max_walk_lengths: [2, 3]
branch_counts: [1, 2, 4]
reconvergence_counts: [0, 1]
distractor_counts: [0, 4]
queries_per_condition: 8
seed_count: 2
dtype_reference: float64
device: cpu
sweep: one_factor_at_a_time
clifford: disabled
shared_witness_appendix: optional
final_test_tuning: forbidden
```

After the smoke gate, register a pilot with enough independent graphs and seeds for uncertainty estimates. Sweep width, depth, retained edges, active groups and dtype separately before a manageable crossed design. No mandatory large Cartesian grid or GPU run is implied.

Report two cost tracks: component replay on cached inputs, and end-to-end graph/query processing. Include graph preprocessing, reverse-edge construction, indexes, scoring, sorting, routing, data transfer, message composition, aggregation, decoder and candidate ranking as applicable. Disclose caching and amortization across queries. Full path enumeration is limited to tiny diagnostic references; ordinary message passing need not enumerate exponentially many paths.

Measure wall-clock latency/throughput, peak process/device memory, real coefficient count, parameters, inspected and retained edges, message evaluations, routing evaluations and active states. Separate training and inference. Record hardware, threads, batch size, warmup, synchronization and dtype. Extra logging/oracle instrumentation is reported separately from normal inference timing.

## 13. Metrics and decision rules

| Stage | Primary measurements | Interpretation limit |
| --- | --- | --- |
| Composition | Ordered-pair score margin, exact/numerical state separation, task accuracy | A successful probe demonstrates accessibility; a failed probe does not prove no decoder can recover information |
| Compression | Pre/post task-predicate probe performance, registered collisions, accuracy versus width/groups | State reconstruction is a diagnostic, not equivalent to task correctness |
| Selection | Fraction of positives with at least one complete support route retained; error versus full propagation | Edge recall alone may miss that no complete route survives |
| Integrated GNN | Boolean accuracy, false positives/negatives, margin, quality–cost frontier | Accuracy gains are not a universal expressiveness theorem |
| Real KG extension | Filtered ranking metrics with declared protocol plus motif-stratified results | Observational topology/error correlation is not causal attribution |
| Shared-witness track | Exact soft error, witness recall, coverage and unresolved rate | Certification is only for the Appendix A objective and fixed scores |

Include probe complexity and training protocol; all probes use training/validation data only. Log paired per-query outputs, but calculate uncertainty at graph/seed level because queries are correlated. Report all seeds and null/failed runs.

Required report figures are: composition accuracy versus length; compression error versus incoming evidence diversity at fixed budget; selection quality versus retained and inspected work; integrated quality versus runtime and peak memory. Show held-out topology results separately. Use measured axes; do not precommit to collapse at paths/dimension = 1.

## 14. Build order and handoff

1. **Scaffold and truth:** schemas, manifests, CPU CLI, deterministic fixtures and independent symbolic oracles. Gate: semantics, permutation and leakage checks pass.
2. **Three diagnostic modules:** restricted composer, cached-message compression replay, full/fixed/random/oracle selection, and boundary traces. Gate: known fixtures and accounting pass for all three mechanisms.
3. **Conditional GNN smoke integration:** fixed initialization/readout, full propagation and one-component interventions on tiny graphs. Gate: deterministic reproduction, full-selection equivalence and a report separating diagnostic controls from learned results. No method must win.
4. **Frozen pilot:** registered splits, nuisance audits, component and crossed comparisons, uncertainty and costs. Gate: attributable evidence or clearly reported null results.
5. **Architecture and adaptive study:** stronger published controls, grouped/adaptive aggregation; optional Clifford factorial comparisons. Gate: training/access parity and complete resource accounting.
6. **External relevance:** incomplete graphs, real KG protocol and previous-paper reproduction as separate extensions. Gate: atomic quality, leakage and task compatibility checked before interpreting gains.

Appendix A can be implemented as a parallel supporting track after the truth layer; its use does not replace stages 2–3. Here parallel means an independent workstream, not authorization to spawn agents.

```text
python -m kg_reasoning_lab generate --config configs/smoke.yaml
python -m kg_reasoning_lab audit --config configs/smoke.yaml
python -m pytest
python -m kg_reasoning_lab run --config configs/smoke.yaml
python -m kg_reasoning_lab report --run-id <actual-run-id>
```

At the first implementation handoff deliver stages 1–3 and a small CPU smoke report unless broader execution is requested. Include runnable source, actual environment versions, reproduction commands, frozen configs, tests, raw outputs, dataset/trace/component hashes and source revision or snapshot hash. Identify unimplemented components. Do not claim the playground exists merely because this design has been written.

Stop dependent work and report if oracles fail, shortcuts explain difficulty, available evidence already lacks the distinction, atomic quality prevents diagnosis, or an intervention is being credited for a changed decoder/input budget. Reject the proposed remedy if stronger matched controls explain gains or end-to-end costs erase them. A negative result is valid; do not redesign final tests to force Clifford success.

## 15. Research grounding and boundaries

Starting references, not an exhaustive novelty review. Record exact versions and supported tasks before implementation/publication.

- [Relational WL for link prediction](https://arxiv.org/abs/2302.02209): architecture-specific expressiveness framework.
- [Weisfeiler and Leman Go Relational](https://proceedings.mlr.press/v198/barcelo22a.html): relational GNN analysis.
- [How Powerful Are Graph Neural Networks?](https://arxiv.org/abs/1810.00826): learned multiset aggregation and expressiveness.
- [GNN bottlenecks](https://arxiv.org/abs/2006.05205): long-range information propagation and over-squashing.
- [R-GCN](https://arxiv.org/abs/1703.06103) and [CompGCN](https://arxiv.org/abs/1911.03082): relational encoders and composition.
- [NBFNet](https://arxiv.org/abs/2106.06935): conditional path-based link prediction; not restricted to raw sum.
- [A*Net](https://arxiv.org/abs/2206.04798), [AdaProp](https://arxiv.org/abs/2205.15319) and [TAGNet](https://aclanthology.org/2023.emnlp-main.908/): existing efficient propagation approaches.
- [PN-GNN](https://arxiv.org/abs/2511.07994): path-neighbour aggregation and logical expressiveness; relevant to novelty checks.
- [GeomE](https://aclanthology.org/2020.coling-main.46/) and [Keci](https://papers.dice-research.org/2023/ECML_Clifford/public.pdf): existing geometric/Clifford KGE.
- [CQD](https://openreview.net/pdf?id=Mos9F9kDwkz), [GNN-QE](https://arxiv.org/abs/2205.10128), [AnyCQ](https://arxiv.org/abs/2409.13959) and [CQD adaptation](https://proceedings.neurips.cc/paper_files/paper/2023/hash/55c518a17bd17dcb69aa14d69d085994-Abstract-Conference.html): supporting query-answering comparisons from version 1.
- [Fagin, Lotem and Naor](https://arxiv.org/abs/cs/0204046): threshold aggregation is prior art; Appendix A is a correctness reference.

Never substitute a reconstruction bound for a task-specific reasoning bound, an empirical failure for impossibility, a privileged oracle for an equal-input learned baseline, or certified inference relative to scores for certified factual truth.

## 16. Ready-to-use implementing-agent instruction

> Implement stages 1–3 of this specification. Inspect the project and report runtime/blockers first. Build deterministic ordered-chain, reconverging-branch and rare-support fixtures with independent symbolic truth. Implement separate composition, compression and selection interfaces plus boundary replay. Verify a restricted commuting-composition collision, a fixed-message sum collision with a separating nonlinear control, and a selection miss with feasible oracle support retention. Integrate a small source/query-conditioned GNN harness, freeze non-intervened components, and record all inspected as well as retained work. Preserve entity alignment and neighbour permutation invariance; keep truth and oracle policies evaluator-only. Run only a small CPU smoke study and deliver code, commands, manifests, checks and a report separating proven fixture properties from learned outcomes. Retain Appendix A as an optional supporting executor track. Do not add Clifford layers, claim novelty, tune on final tests or launch large runs in this first handoff. A null result is a valid outcome.


## Appendix A. Retained shared-witness verification track

The following contracts retain the original guide's exact-input, frozen-score and bounded-search design. IDs A–E, certificates, cost axes and pilot defaults in this appendix apply only to this supporting track. They are not guarantees or primary pilot settings for the GNN experiments. Implement this track when enabled in the configuration.

### A.1. Freeze the semantics before coding models

Use Boolean answer classification first:

    Q(x,z,w) = exists y: r(x,y) AND s(y,z) AND t(y,w)

All anchors are fixed. The output is whether any entity y satisfies every atom. Define candidate sets A, B and C for the three atoms; truth is whether their intersection is nonempty.

For J atoms, define a score tensor b[j,v] in [0,1] over a common, explicitly aligned entity universe. The first soft executor uses:

    S(Q) = max_v product_j b[j,v]

For binary b this equals existential conjunction. For soft b it is a chosen fuzzy score, not automatically the probability that Q is true. Use product conjunction and max existential aggregation consistently across executor comparisons. Other semantics are separate ablations.

Use log-domain calculations for products, preserving exact zero as negative infinity. An epsilon floor changes semantics and must not be inserted silently.

Define one entity universe per graph. Entity IDs are local keys for alignment, not a cross-graph learned identity feature. Keep relations consistent across splits unless explicitly testing unseen relations.

### A.2. Build the benchmark in three layers

#### Layer 1: exact Boolean fixtures

Construct small graphs directly from candidate sets. For example:

    Positive: A={0,1}, B={0,2}, C={0,3}
    Negative: A={0,1}, B={1,2}, C={0,2}

Both have set sizes (2,2,2), but only the positive has a common witness. This fixture matches marginal sizes only; it does not claim to match all pairwise overlaps or degrees.

Include empty sets, no witness, one witness, multiple witnesses, all entities valid, duplicated atoms and permuted entity IDs. Duplicated atoms must be idempotent under Boolean semantics. Product fuzzy semantics is not generally idempotent for fractional scores; test and document the distinction.

#### Layer 2: score-controlled executor stress tests

Generate score tensors independently of a learned KGE to isolate inference behavior. Include:

- One shared moderate-scoring entity versus distinct high-scoring distractors for each atom.
- Ties, zeros, all-ones tensors and near-threshold answers.
- A best joint witness ranked below fixed K in every individual atom list.
- Cases where pruning changes the soft maximum without changing Boolean truth, and vice versa when atomic scores are wrong.

Persist the tensor itself so every executor receives identical values. Report these as controlled score experiments, not learned-model results.

#### Layer 3: learnable incomplete graphs

Do not randomly hide independent random edges and expect their recovery to be learnable. Define an explicit generative mechanism that provides observable predictive structure, such as noisy relation-specific rules or permitted entity attributes. Document what information makes a hidden edge predictable.

Construct the full truth graph, then hide edges using a registered corruption process. Labels come from the full graph; the scorer sees only the observed graph and allowed features. Missing observed edges mean unknown, not false. Never hide an edge but leave an automatic inverse that trivially reveals it.

Train the scorer on training graphs, calibrate on validation graphs and freeze it for executor comparisons. Include an atomic prediction sanity check before interpreting complex-query errors. An unseen-ID split requires an inductive scorer; ordinary fixed entity tables are not eligible without a defined adaptation protocol.

#### Splits and shortcuts

- Split by generated graph instance, not by query alone. Keep matched examples and identity-permuted variants in the same split.
- Use independent train/validation/test seeds; freeze final test generation rules before model selection.
- Evaluate larger candidate universes and more conjuncts separately, then optionally jointly.
- Record label distributions by every sweep variable.
- Audit relation frequencies, set sizes, graph size, anchor degrees and witness-degree histograms. Train simple nuisance-feature baselines using only the statistics intended to be matched.
- Report any residual shortcut. Do not claim relational color-refinement indistinguishability without an actual construction and check.
- Do not suppress the intended signal: identifying that all queried edges touch the same entity is legitimate reasoning, not leakage.

### A.3. Required baselines and causal comparisons

| ID | Atomic source | Executor | Role |
| --- | --- | --- | --- |
| A | Exact Boolean truth | Exact set join | Ground-truth implementation oracle |
| B | Same exact truth | Learned compressed executor | Representation/optimization diagnostic |
| C | Frozen soft scores | Exhaustive max-product | Exact reference for the chosen soft semantics |
| D | Same frozen scores | Fixed-K candidates | Pruning baseline |
| E | Same frozen scores | Adaptive bounded search | Proposed inference intervention |

Implement a deliberately weak marginal-count or independent-existence baseline as a sanity check. It should fail the matched examples; this is not evidence against stronger models.

For fixed-K, specify the exact policy: retrieve top-K per atom, take the union, evaluate all atoms for every entity in that union, then maximize the product. Top-K intersections or beams are different baselines and must be separately named.

A learned compressed baseline needs a documented architecture, input access and training objective. Include an entity-aligned conjunction-then-pool positive control. Give all baselines the same permitted inputs; do not force one to operate without the identity alignment provided to another.

Interpret paired differences carefully:

- B versus A tests the learned executor with uncertainty removed, not a universal capacity theorem.
- D versus C isolates approximation of the same soft objective.
- E versus D tests budget allocation; match measured cost rather than nominal K alone.
- C versus graph truth reflects atomic scoring and the chosen score-combination semantics; it is not pruning error.

Before any paper-level claim, add or reproduce relevant baselines such as CQD/CQD adaptation, GNN-QE and AnyCQ on tasks they support. Record unsupported tasks instead of silently changing a published method. Do not benchmark a bare triple scorer as if it were a complete query executor.

### A.4. Adaptive executor with a checkable bound

For the first implementation use fixed, cached scores in [0,1], a finite entity universe and exact sorted lists. This isolates executor correctness. Charge list creation and sorting separately; do not claim end-to-end score savings from a fully precomputed tensor.

Algorithm contract:

1. Read a prefix of each atom's sorted entity list.
2. Let C be the union of all retrieved entities.
3. Obtain all J scores for each v in C, respecting a shared cache.
4. Compute L = max over C of the product of its atom scores; use L=0 for an empty C.
5. For each atom j, obtain tau[j], a valid upper bound on that atom's score for every entity outside C. The largest unread score in its exact sorted list is a conservative choice; use 1 for an untouched list.
6. Set U = max(L, product_j tau[j]). If C is the full universe, set U=L directly.
7. Stop on sufficient accuracy, certified classification, full enumeration or budget exhaustion; otherwise expand and repeat.

Correctness: every omitted entity has each score at most tau[j], so its product is at most their product. Therefore L <= S(Q) <= U. This is a reference guarantee, not a new theorem claim.

Termination conventions:

- Approximate score: U-L <= epsilon.
- Classification with positive defined by S >= theta: positive if L >= theta; negative if U < theta.
- Budget exhausted without either: return unresolved with interval [L,U]. A forced point prediction may be reported separately but is not certified.
- Do not call approximate nearest-neighbor retrieval certified unless it supplies valid bounds for every omitted entity.

Begin with round-robin expansion. Only later test a heuristic that expands the least selective/highest-threshold list or allocates work using cost estimates. Correctness comes from validated bounds, not the expansion heuristic.

This guarantee applies only to fixed atomic scores under this one-witness max-product objective. It does not certify the KG's truth, arbitrary multi-variable joins, or an executor whose scores change as candidates are added.

### A.5. Non-negotiable tests

| Test | Acceptance criterion |
| --- | --- |
| Boolean oracle | 100% agreement with independent brute-force evaluation |
| Boolean/soft equivalence | Max-product agrees with joins on all binary fixtures |
| Bound soundness | Every intermediate interval contains exhaustive S, within registered numerical tolerance |
| Full enumeration | C, D with K=universe size, and E agree |
| Budget exhaustion | Unresolved outcomes never labeled certified |
| Adversarial ranks | Fixture demonstrates a genuine fixed-K miss and adaptive recovery if budget permits |
| Entity permutation | Oracle outputs invariant; scores equivariant after index remapping |
| Atom order | Max-product invariant to atom reordering, up to numerical tolerance |
| No leakage | Training loader cannot expose truth labels, witness files or hidden-edge metadata as features |
| Split integrity | No paired/renamed instance crosses splits |
| Score freezing | C/D/E reference the same tensor/checkpoint hash |
| Reproducibility | Same config and seeds reproduce data hashes and deterministic reference outputs |
| Numeric edge cases | Zeros, ties, singleton universe and empty candidates handled explicitly |

For J<=3 and universe size <=4, exhaustively enumerate binary score matrices. Add randomized floating-point tests and adversarial near-threshold cases. Test implementation invariants, not only final accuracy.

### A.6. Pilot configuration and cost accounting

Suggested defaults are starting points, not experimentally validated budgets. Begin with 64 queries per condition on CPU. Scale only after measuring runtime and memory.

    stage: executor_pilot
    universe_sizes: [32, 128, 512]
    conjunct_counts: [2, 3, 4]
    fixed_k: [1, 4, 16]
    epsilon: 0.000001
    classification_threshold: 0.5
    seed_count: 5
    dtype_reference: float64
    final_test_tuning: forbidden

Run one-factor sweeps before a full Cartesian grid. Keep the same instances and score tensors paired across executors. Register the training/validation procedure for any non-default classification threshold.

Report two cost tracks:

1. Executor-only on cached scores: list accesses, join work, candidates, memory and runtime.
2. End-to-end: atomic scoring, retrieval/index construction, cache memory, query execution and amortization assumptions.

Report unique score evaluations and repeated calls separately. If every entity was scored to create sorted lists, say so. More materialized witness states can use substantially more memory than a small neural vector; do not call equal embedding width equal total budget.

### A.7. Metrics and interpretation

Primary metrics:

- Soft score error relative to exhaustive C: mean and worst-case absolute error.
- Boolean accuracy and false-positive/false-negative rates relative to graph truth.
- Certification coverage, unresolved rate and conditional accuracy on resolved queries. Always show coverage beside conditional accuracy.
- Valid-witness retrieval recall on positive queries.
- Best-soft-witness retention, allowing score ties; distinguish this from true-witness recall.
- Runtime, peak memory, unique atomic scores, list reads and candidate evaluations.

For learned scores also report atomic prediction quality. Do not report a fuzzy score as a calibrated probability without validation.

Use paired comparisons on the same graph/query cases. Report per-seed results and graph-level uncertainty intervals; correlated queries from one graph are not independent samples. Do not interpret accuracy gaps as an additive decomposition of causal errors.

Headline figure: approximation error or resolved coverage versus measured computation, stratified by witness rank and conjunct count. Do not precommit to collapse at paths/dimension = 1.

