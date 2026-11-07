# OrchSLM: Probing the Dynamics of Small Language Model Orchestration

Источник PDF: https://arxiv.org/pdf/2609.13470v1

Извлечённый текст; формулы и таблицы следует сверять с PDF.

```text
PAGE 1
OrchSLM: Probing the Dynamics of Small Language
Model Orchestration
Chengxi Zhang∗
Harvard
chengxizhang@g.harvard.edu
Yu Yao∗
MIT
yu_yao@mit.edu
Abstract
Although large language models (LLMs) have demonstrated remarkable capabil-
ities, their reliance on cloud-scale infrastructure poses fundamental challenges
for deployment in agentic pipelines, including latency, privacy, connectivity, and
substantial computational cost. Small language models (SLMs) offer a compelling
alternative: recent studies suggest that many repetitive and narrowly scoped sub-
tasks in agentic workloads may be better served by specialized SLMs than by
monolithic LLMs. However, the limited capacity and context windows of SLMs
can constrain long-horizon reasoning and interaction-heavy orchestration strate-
gies such as iterative verification and debate. This motivates a complementary,
non-interactive paradigm in which heterogeneous SLMs independently generate
candidate solutions and a router orchestrates their cached samples without further
model interaction. To further understand the mechanisms of such orchestration,
we introduce OrchSLM, a routing framework that unifies existing non-interactive
orchestration methods and exposes their underlying design choices as controllable
parameters. Using OrchSLM as a systematic probe, we reveal how orchestration
behavior emerges from diverse knobs, including the task structure, model-pool
composition, and multi-agent consensus.
1 Introduction
Large language models have demonstrated remarkable progress across reasoning, coding, tool use,
and increasingly complex agentic tasks [OpenAI, 2023, Touvron et al., 2023, Grattafiori et al., 2024,
DeepSeek-AI, 2025, Snell et al., 2024b]. Scaling model capacity and inference-time computation has
further extended their ability to solve challenging problems [Kaplan et al., 2020, Hoffmann et al.,
2022, Lightman et al., 2024, Brown et al., 2024, Snell et al., 2024a]. However, these capabilities
come with substantial computational and deployment costs, particularly when language models must
be invoked repeatedly as components of agentic pipelines. As an alternative, small language models
are increasingly emerging as low-cost, deployable, and heterogeneous reasoning engines [Wang
et al., 2024a, Belcak et al., 2025]. Their modest computational footprint makes it practical to query
multiple models at inference time, raising the possibility of treating a collection of SLMs as a pool of
complementary specialists. However, most orchestration strategies developed for larger language
models rely heavily on interaction: models exchange natural-language messages, critique intermediate
solutions, synthesize one another’s responses, or verify candidate answers before a final decision is
made [Wang et al., 2024b, Du et al., 2023, Lifshitz et al., 2025]. For SLMs, such interaction can be
costly relative to the models themselves and relies on deliberation and self-correction capabilities
that remain limited [Taubenfeld et al., 2024, Huang et al., 2024, Liu et al., 2023, Fu et al., 2025].
∗Equal contribution; order decided by a coin flip.
Preprint.
arXiv:2609.13470v1  [cs.AI]  11 Sep 2026

PAGE 2
This motivates a complementary,non-interactive 2 setting: each model independently generates a set
of cached samples, and final answers are obtained by combining the resulting answer distributions at
inference time. Despite its simplicity, a systematic understanding of such non-interactive orchestration
still remain unknown. Existing methods make fundamentally different implicit assumptions about
what constitutes useful evidence. Query-level routers and cascades choose a model before generation,
typically based on predicted expertise, cost, or preference [Chen et al., 2023, Ong et al., 2025,
Chen et al., 2024]. V oting methods instead treat agreement across independently generated answers
as evidence [Wang et al., 2023, Li et al., 2024]. Routing-style routing estimates within-model
consistency and favors answers from models that repeatedly support the same prediction [Wang et al.,
2025]. These approaches expose distinct signals (expertise,self-consistency, andmutual agreement)
but entangle them through different routing rules.
To understand these signals is important because each can be informative in one regime and misleading
in another. A strong expert may provide a reliable default answer, yet expert selection discards
corroborating evidence from the rest of the pool. Majority agreement can reveal shared evidence,
yet a collection of weaker models may reinforce the same plausible error [Li et al., 2024, Wang
et al., 2025]. High self-consistency can indicate confidence, but it can also arise from a model that
repeatedly makes the same mistake [Wang et al., 2023, 2025]. Even the composition of the model
pool introduces a nontrivial tradeoff: adding another model may increase the probability that some
model produces the correct answer while simultaneously introducing additional incorrect support
that makes the answer harder to identify. Consequently, orchestration quality is governed not by the
number of available models or the amount of agreement alone, but by how expertise, agreement, and
pool composition interact with the structure of the task. This raises the central question of our work:
what governs the dynamics of small language model orchestration?Rather than searching for a single
universally optimal router, we seek to isolate the mechanisms that determine when heterogeneous
SLMs complement one another, when collective agreement provides useful evidence, and when
adding more models instead makes routing harder.
To probe these tradeoffs systematically, we decompose non-interactive multi-SLM orchestration along
three research axis.(1) Whom should we trust?How should authority be distributed across models?
(2) How should evidence combine?How should multiple beliefs form a collective decision?(3)
When does agreement matter?When does consensus provide evidence beyond individual confidence?
Existing routing methods implicitly make different choices along these dimensions. Majority voting
weights models equally, accumulates support across the pool, and relies on mutual agreement. Single-
best routing places all trust in a fixed expert. Router-style favors the model exhibiting the strongest
self-consistency, while cross-support treats probability mass assigned by other models to a candidate
answer as question-specific evidence. We make these choices explicit as controllable axes, allowing
existing routers and their intermediate variants to be studied within a common design space.
We instantiate this design space as a generalized routing family in which common orchestration
methods emerge as exact parameter settings, and use it as a controlled probe of orchestration
dynamics. Across MATH-500, GSM8K, GPQA, and MMLU-Pro with a heterogeneous model pool,
our systematic analysis reveals distinct dynamics of multi-SLM orchestration: effective routing
emerges from the interplay between task structure, model expertise, pool composition, and agreement
patterns, rather than from any single source of evidence.
Contributions.(1) We formulate non-interactive multi-SLM orchestration as a controlled design
space organized around three coupled mechanisms:whom to trust,how to aggregate evidence, and
what form of consistency should constitute confidence. (2) We introduce ORCHSLM, a generalized
routing family that exposes these mechanisms as controllable axes and recovers majority voting,
self-consistency routing, cross-support, expert-weighted routing, and the single-best limit as specific
configurations within a common framework. (3) We use ORCHSLM to systematically probe the
dynamics of multi-SLM orchestration across tasks and model pools. Our experiments reveal how
expertise, agreement, and pool composition jointly determine routing behavior, including a persistent
gap betweenoracle coverageandroutable support: more models can make the correct answer
available without making it easier to identify.
2We usenon-interactiveto mean that models do not communicate with one another or condition their generations on other
models’ outputs. Each model generates independently, and cross-model information is combined only by the router after
generation, without consuming additional context tokens for inter-model communication.
2

PAGE 3
2 Preliminaries and Methodology
Question: In QCD, the strong coupling becomes weaker at very short distances mainly because...
A: quark screening B: gluon antiscreening C: chiral breaking D: confinement Gold: B
Generated answers, trials = 5
Model Expert prior 1 2 3 4 5
Qwen expert B B B A C
Gemma strong B B A B D
Phi confident wrong A A A A B
Llama mixed A A B C D
Weak SLM weak A A A C D
Top votes favor A,
but stronger models
support B,
C/D are distractors.
Routing decision phase
▼ ▼ ▼
The same generated answers can support different decisions; the router decides which evidence counts.
1. Majority vote
Collects: Each model's top answer
Decision: A wins 3-2 -> Wrong
Failure case: Weak models share a distractor
and outvote stronger support
A x
2. Self-consistency routing
Collects: Most repeated answer
within one model
Decision: Phi repeats A -> Wrong
Failure case: Repeated wrong answers
look overconfident
A x
3. Single-best expert
Collects: Only the validation-best model
Decision: Qwen selects B -> Correct
Failure case: Fixed expert has no backup
when it misses this question
B ✓
4. Expert-weighted vote
Collects: Model votes weighted by
expert prior
Decision: Qwen and Gemma weights lift B
-> Correct
Failure case: Prior can dominate evidence
when experts are mismatched
B ✓
5. Cross-consistency routing
Collects: All answer chips equally
Decision: A has more total support -> Wrong
Failure case: Weak support accumulates
and overwhelms useful signals
A x
6. Generalized routing
Collects: Expert prior + top support +
pooled support
Decision: Varies with routing assumptions
Failure case: No fixed setting transfers
across tasks and pools
Assumption-dependent answer
Figure 1: Different routers can pick different answers from the same cached outputs.
Motivation.Orchestration is a problem of collective decision-making [Arrow, 1951]: given multiple
imperfect and heterogeneous reasoners, how should their individual beliefs be transformed into
a collective judgment? More voices do not necessarily imply more knowledge, agreement does
not necessarily imply truth, and expertise does not necessarily imply correctness on every instance.
The challenge is therefore not merely to combine model outputs, but to determine when individual
authority should prevail, when evidence should accumulate, and when agreement should be trusted.
We study this problem through three fundamental questions.Whom should we trust?Models differ
in expertise, raising a tension between democratic aggregation and expert authority.How should
evidence combine?Support may derive its strength from accumulation across many independent
sources, or from the conviction of a particularly reliable one.When does agreement matter?Repeated
belief can reflect genuine confidence, but consensus can also amplify shared errors; its value therefore
depends on who agrees and how their evidence is formed. These questions expose three distinct
but coupled mechanisms underlying multi-SLM orchestration:trust,aggregation, andconsistency.
We next formalize them as controllable axes of a generalized routing family, allowing different
orchestration strategies to be studied within a common framework.
2.1 Evaluation of Orchestration
How should collective evidence become a decision, and whose evidence actually shapes that decision?
We evaluate orchestration through these two complementary perspectives. The first concernsdecision:
we formalize how trust, aggregation, and agreement determine which answer is selected. The
second concernsinfluence: we distinguish between models that are heard by the router and models
whose presence actually improves its decision. Together, these views characterize not only what an
orchestration system decides, but also how that collective decision emerges. Additional derivations
and implementation details are provided in Appendix §A.
From samples to beliefs.For a question x, each model i independently generates K samples. After
answer extraction and normalization, these samples induce an empirical distribution over candidate
answers:
Pia(x) = 1K
KX
k=1
1[yik =a].(1)
3

PAGE 4
Here, Pia(x) represents the empirical support that model i assigns to answer a. Let a⋆i =
arg maxa Pia(x)denote modeli’s most frequent answer, with ties broken deterministically, and de-
fine Mia =1[a=a ⋆i ]. Thus, Pia captures the model’s full answer distribution, while Mia identifies
the answer on which its own belief concentrates most strongly.
Table 2: Common routers mapped to(α, κ,Agg)in the generalized family.
Method Examples Trust Aggregation Position in family
Expert selection Query routing, cascades, model routers Expert Single(1,∞,−)
Majority voting Self-consistency voting and multi-agent majority Average Pooled sum(0,0,−)
Weighted voting Cross-validation weighted probabilistic ensemble Expert Pooled sum(0, κ >0,−)
Self-consistency SLM-MUX-style answer-level routing Average Max(1,0,Max)
Weighted consistency Validation-aware consistency routing Expert Max(1, κ >0,Max)
Cross consistency Answer-level cross-confidence and support variants Average Max + Pooled(0< α <1,0,Max)
Generalized routing This generalized family Tunable Tunable([0,1],[0,∞],{Sum,Max})
Generalized routing family
Expert-weighted
Equal-weight
Mutual
support
Self-
consistency
Consistency sourceα
Expert trustκ
Mutual Mixed Self
Expert
Average
Single-best
Expert-weighted Vote
Majority-Vote
Cross-consistency Self-consistency
Figure 2: Existing routers as fixed corners of the
(α, κ,Agg)generalized family; axes are defined in Sec-
tion 2.1.
Trust: whose evidence should count?
Models need not be equally reliable. Let
acci denote an expert prior for model i, es-
timated from a validation split or supplied
as a fixed prior score. We translate these
priors into normalized trust weights:
ri(κ) = max(acci, ϵ)κP
j max(accj, ϵ)κ .(2)
The parameter κ determines how concen-
trated authority is across the model pool.
At κ= 0 , every model receives equal
weight, corresponding to a democratic prior
over expertise. As κ increases, authority
shifts toward models with stronger prior
performance; in the limit κ→ ∞ , it
concentrates on the validation-best model.
Thus, κ continuously interpolates between
collective equality and expert authority.
Aggregation and agreement: how should
evidence combine?For each candidate answera, we define the routing score
W(a) =αAgg i [Miari(κ)Pia(x)] + (1−α)
X
i
ri(κ)Pia(x),(3)
and select ˆa= arg maxa W(a) . Here, Agg controls how evidence combines across models: Sum
accumulates support from multiple models, whereas Max retains the strongest supporter. The
parameter α controls the form of agreement, interpolating between support for each model’s preferred
answer and pooled support over its full answer distribution. Together, (κ,Agg, α)specify whom to
trust, how to combine their evidence, and which form of agreement to use.
Influence: being heard versus being helpful.The routing score determines which answer is selected,
but does not tell us whether each model helps make that selection correct. We therefore measure
model influence through attribution and contribution.
For a datasetD, letˆat be the routed answer for questiont. We define the attribution of modelias
Atr(i) = 1|D| Xt∈D P (t)i,ˆat .(4)
Atr measures how strongly a model supports the answers selected by the router. A model with high
Atr is frequentlyheard, but its support need not improve accuracy.
To measure whether a model is helpful, we remove it and rerun the router. We define its counterfactual
contribution as
Ctr(i) = Acc(full pool)−Acc(pool withouti).(5)
4

PAGE 5
Equivalently, ifˆa(−i)t is the routed answer without modeli, then
Ctr(i) = 1|D| Xt∈D 1[ˆat =y t]−1[ˆa(−i)t =y t] .(6)
3 Experiments
Setup.All routers operate on the same cached generations from a seven-model pool (Qwen2.5-
7B-Instruct, Meta-Llama-3.1-8B-Instruct, Llama-3.1-Tulu-3-8B-DPO, DeepSeek-Math-7B-Instruct,
Gemma-2-9B-IT, Phi-3.5-mini-instruct, Ministral-8B-Instruct-2410) across four benchmarks: MATH-
500 (competition math, one dominant expert), GSM8K (saturated arithmetic), GPQA (hard, balanced
science), and MMLU-Pro (broad multi-topic science). Implementation details are at Appendix §A.
3.1 Scaling up Task Structures
0
.5
1
2
4
8
Sum
κ (log)
MATH
 GSM8K
 GPQA
 MMLU-Pro
0.0 0.5 1.0
α
0
.5
1
2
4
8
Max
κ (log)
0.0 0.5 1.0
α
0.0 0.5 1.0
α
0.0 0.5 1.0
α
0.72 0.76 0.79
 0.90 0.91 0.92
 0.34 0.38 0.43
 0.57 0.61 0.64
Figure 3: Accuracy heatmaps over (α, κ)for Sum and Max aggregation on the full pool. Dashed
contour: cells within1pp of the best score; stars: best per task.
MATH-500 GSM8K GPQA MMLU-Pro
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy
0.794
0.917
0.433
0.640
Single-best
Majority
Self-consistency
Cross-support
Best generalized
Oracle
Figure 4: Router generalizations.
Table 3: Optimal-point behavior under
the OrchSLM parameter scan. The se-
lected optima emphasize expert trust
on the two math benchmarks, while
the science benchmarks rely more on
mixed or mutual support.
Task Trust Consistency Aggregation Regime
MATH-500 Expert Mixed Tie Expert
GSM8K Expert Self Max Saturated
GPQA Expert Mixed Tie Support
MMLU-Pro Mixed Mutual Tie Pooled
The best routing strategy varies across tasks. Figure 4 compares four cache-only routing strategies
with the oracle upper bound. On MATH-500, single-best routing is near-optimal, indicating that
routing is largely determined by the strongest expert. On GSM8K, several strategies perform similarly.
In contrast, GPQA and MMLU-Pro retain substantial gaps to the oracle, suggesting that the presence
of a correct answer does not make it easy to identify. These differences motivate us to study
routing as a task-dependent decision rather than fixing a single strategy across tasks. We vary the
5

PAGE 6
three components of the generalized router: the trust concentration κ, the aggregation operator
Agg∈ {Sum,Max}, and the balance α between top-answer and pooled support. Figure 3 shows the
resulting routing landscape, and Table 3 summarizes the best regime for each task. We next isolate
the effect of each component.
Table 4: Experts vs average models. Rel.
∆ is the relative change from the best
equal-weight cell ( κ= 0 ) to the best
full-sweep cell.
Task Equal Expert Rel.∆
MATH-500 0.7338 0.7938 +8.2%
GSM8K 0.9052 0.9166 +1.3%
GPQA 0.3909 0.4328 +10.7%
MMLU-Pro 0.6304 0.6402 +1.6%
Trust concentration.Expert weighting has the largest
effect on routing performance. Table 4 compares the best
setting with κ >0 against the best equal-weight setting
with κ= 0 . Concentrating trust consistently improves
routing across all four tasks. On math, it prevents weaker
models from diluting the strongest expert. On the harder
science tasks, it reduces the influence of unreliable support.
Thus, deciding which models to trust is the primary routing
decision.
Aggregation.The choice between Sum and Max has a
smaller effect. As shown in Table 5, GSM8K clearly favors
Max, while the best Sum and Max settings are close on the
other three tasks. This is consistent with GSM8K being a
saturated task, where selecting the strongest expert-supported answer is sufficient. In the remaining
tasks, changing how support is aggregated matters less than changing which models receive weight.
Table 5: Majority vs coalition support,
vs strongest supporter. Rel. ∆ is the
relative change from Sum to Max aggre-
gation.
Task Sum Max Rel.∆
MATH-500 0.7938 0.7936 0.0%
GSM8K 0.9089 0.9166 +0.8%
GPQA 0.4318 0.4328 +0.2%
MMLU-Pro 0.6398 0.6402 +0.1%
Agreement.The usefulness of self-consistency and mu-
tual support depends on which models the router trusts.
Table 6 compares the two signals by setting α= 1 for
top-answer support and α= 0 for pooled support under
a fixed κ. With equal model weights, the two signals are
similar and neither consistently dominates. After expert
weighting, their roles separate: math favors self-consistent
expert predictions, whereas the broad science task ben-
efits more from pooled support. Agreement is therefore
informative only relative to the models providing it.
3.2 Scaling up Routers
Table 6: Self-consistency vs mutual agreement.
Pooled-only uses α= 0 ; top-only uses α= 1 .
Rel. ∆ is the relative change from pooled-only to
top-only routing.
Allκ κ= 0
Task Pooled Top Rel.∆Pooled Top Rel.∆
MATH-500 0.7934 0.7936 0.0% 0.7246 0.7242 -0.1%
GSM8K 0.9086 0.9166 +0.9% 0.9032 0.9050 +0.2%
GPQA 0.4318 0.4308 -0.2% 0.3864 0.3879 +0.4%
MMLU-Pro 0.6398 0.6350 -0.8% 0.6282 0.6270 -0.2%
Table 7: Oracle coverage is not routable support.
Benchmark Oracle Best generalized Gap
MATH-500 0.8536 0.7938 0.0598
GSM8K 0.9760 0.9166 0.0594
GPQA 0.7854 0.4328 0.3525
MMLU-Pro 0.8420 0.6402 0.2018
0-.2 .2-.4 .4-.6 .6-.8 .8-1
-20
-10
0
10
20
Rate (%)
MATH
0-.2 .2-.4 .4-.6 .6-.8 .8-1
GSM8K
0-.2 .2-.4 .4-.6 .6-.8 .8-1
Question disagreement
-20
-10
0
10
20
Rate (%)
GPQA
0-.2 .2-.4 .4-.6 .6-.8 .8-1
Question disagreement
MMLU-Pro
Net Fix Break
Figure 5: Per-bin fixes (green) and breaks
(red) when switching self-consistency to the
best generalized cell, with the net rate (line).
Oracle coverage and routability.Higher oracle coverage
does not necessarily translate into higher routed accuracy.
Table 7 compares the oracle upper bound with the best generalized routing result. The gap is small
on the math tasks but substantially larger on GPQA and MMLU-Pro. On these science benchmarks,
a correct answer is often present in the model pool but cannot be reliably identified from the cached
predictions. Increasing coverage alone is therefore insufficient; the router must also distinguish useful
evidence from competing support.
6

PAGE 7
Disagreement.Disagreement creates both opportunities and risks for routing. Figure 5 groups
questions by the level of disagreement among the seven models and measures how generalized
routing changes the predictions of self-consistency. As disagreement increases, the router corrects
more self-consistency errors, but it also changes more correct predictions into incorrect ones. The
largest net benefit occurs at moderate disagreement, where alternative evidence is available without
being dominated by conflicting support.
0.5
0.6
0.7
0.8
0.9Accuracy
MATH
0.825
0.850
0.875
0.900
0.925
0.950
0.975
1.000
GSM8K
1 2 3 4 5 6 7
Pool size
0.2
0.3
0.4
0.5
0.6
0.7
0.8Accuracy
GPQA
1 2 3 4 5 6 7
Pool size
0.3
0.4
0.5
0.6
0.7
0.8
0.9
MMLU-Pro
All subsets Oracle Single-best Best generalized
0.0
0.2
0.4
0.6
0.8
1.0Best
MATH
0.0
0.2
0.4
0.6
0.8
1.0
GSM8K
1 2 3 4 5 6 7
Pool size
0.0
0.2
0.4
0.6
0.8
1.0Best
GPQA
1 2 3 4 5 6 7
Pool size
0.0
0.2
0.4
0.6
0.8
1.0
MMLU-Pro
0
1
2
4
8
Best
0
1
2
4
8
Best
Figure 6: Pool-size scan. Left: routed accuracy vs. size (gray: each subset; lines: oracle, single-best,
best generalized). Right: best-cell(α, κ)shifts with pool size.
Pool composition.Adding more models does not necessarily improve routed accuracy. As shown in
Figure 6 (left), oracle coverage continues to increase with pool size, while routed accuracy typically
peaks with three or four models. Additional models can introduce correct answers that increase oracle
coverage, but they can also introduce stable incorrect support that makes those answers harder to
identify. Consequently, the quality of the pool depends not only on the answers it contains, but also on
the support patterns it creates. Pool composition also changes the preferred routing strategy. Figure 6
(right) shows that the optimal (α, κ)shifts as models are added or removed. Smaller curated pools
generally require less trust concentration because the remaining models are already more reliable.
Model selection and answer routing are therefore coupled decisions rather than independent stages.
3.3 Being Heard Is Not Being Helpful
Attribution and contribution capture different aspects of model influence. Atr measures how much
probability mass a model assigns to the router’s selected answer, whereas Ctr measures how routed
accuracy changes when that model is removed. A model can therefore receive high attribution without
making the router more accurate. Figure 7 shows that the two measures behave differently across
tasks. On math, the strongest expert has both high Atr and positive Ctr: the model that most strongly
supports the routed answers is also the one that improves them. On GPQA and MMLU-Pro, however,
several models receive substantial Atr while having near-zero or negative Ctr. Their predictions
frequently agree with the router’s decisions, but removing them does not hurt accuracy and can
sometimes improve it. Pool curation reduces this mismatch. After removing models with unhelpful
support, attribution becomes more concentrated on models with positive counterfactual contribution.
This explains why expert weighting and pool selection can improve routing on the harder science
tasks: effective orchestration depends not only on how much support a model provides, but also on
whether that support is useful for selecting the correct answer. Per-model results are reported in
Table 33, with an accepted-support decomposition in Table 34.
7

PAGE 8
qwenllamatuludsmathgemmaphiministral
0.0
0.2
0.4
0.6
0.8
1.0
Atr
MATH
Full
qwenllamatuludsmathgemmaphiministral
GSM8K
Full
qwenllamatuludsmathgemmaphiministral
GPQA
Full
qwenllamatuludsmathgemmaphiministral
MMLU-Pro
Full
qwen llama dsmath gemma
0.0
0.2
0.4
0.6
0.8
1.0
Atr
MATH
Curated
qwen llama dsmath
GSM8K
Curated
qwen llama gemmaministral
GPQA
Curated
qwen llama dsmath
phi
MMLU-Pro
Curated
0.000
0.025
0.050
0.075
0.100
0.125
Ctr
0.00
0.05
0.10
0.15
Ctr
Atr (left axis) Ctr (right axis)
Figure 7: Per-model Atr (heard) vs. Ctr (helpful) under each canonical router.
4 Related Work
Small Language Models.Small language models (SLMs) trade capacity for lower latency, memory,
and deployment cost [Abdin et al., 2024, Liu et al., 2024]. Recent gains at this scale come from
curated and synthetic data, knowledge distillation, scale-specific architectures, and targeted post-
training [Wang et al., 2024a, Abdin et al., 2024, Allal et al., 2025, Gemma Team, 2024, Liu et al.,
2024]. This efficiency supports repeated or specialized calls on devices and in agentic systems [Abdin
et al., 2024, Liu et al., 2024, Belcak et al., 2025, Wang et al., 2025], while chain-of-thought, repeated
sampling, and test-time scaling improve reasoning at inference time [Wei et al., 2022, Wang et al.,
2023, Snell et al., 2024b, Muennighoff et al., 2025]. However, weak self-correction and long-context
use constrain interaction-heavy orchestration [Huang et al., 2024, Liu et al., 2023]. SLM-MUX
instead uses independent sampling, confidence-based selection, and model-subset search [Wang et al.,
2025]; OrchSLM studies routing within this non-interactive setting.
Multi-Agent and Compound AI Systems.Multi-agent and compound AI systems organize multiple
model calls into a single solution or decision. At the system level, routing and allocation determine
which calls are made, while prompt and workflow search configure how they are used [Chen et al.,
2023, Ong et al., 2025, Chen et al., 2024, 2025, Poon et al., 2025, Khattab et al., 2023, Opsahl-Ong
et al., 2024, Saad-Falcon et al., 2025, Zhang et al., 2025a]. Given a call configuration, systems differ in
whether outputs interact during generation. Interactive systems condition later calls on earlier outputs
through debate, critique, synthesis, or verification [Du et al., 2023, Wang et al., 2024b, Lifshitz et al.,
2025]. MoA variants refine this pattern through proposer quality and diversity, sparse communication,
and residual, attention, or memory-based flow [Li et al., 2025b,a, Xie et al., 2025, Wen et al., 2026,
Ping et al., 2026]. Non-interactive systems instead sample independently and aggregate only after
generation [Wang et al., 2025]. OrchSLM focuses on this post-generation decision.
Ensembling and Test-Time Inference.Test-time inference improves fixed models or model pools
through additional sampling, search, verification, and aggregation [Snell et al., 2024b, Zhang et al.,
2025b]. In its non-interactive branch, the remaining problem is how to select or combine indepen-
dently generated candidates. Existing rules range from sample voting and verifier-based selection
to confidence-weighted consensus, multi-generator and reward-model ensembles, and reasoning-
structure aggregation [Wang et al., 2023, Li et al., 2024, Lightman et al., 2024, Yao et al., 2026, Song
et al., 2025, Parulekar et al., 2026, Wang et al., 2025]. These rules can be organized along three
axes: model authority (equal or quality-weighted), evidence aggregation (accumulated support or
the strongest supporter), and confidence source (self-consistency or mutual agreement). Attribution
methods add a separate question: whether a model merely supports a decision or changes it when
removed [Ribeiro et al., 2016, Lundberg and Lee, 2017, Koh and Liang, 2017]. OrchSLM parame-
8

PAGE 9
terizes these choices and separates answer availability from routability and received support from
counterfactual contribution.
5 Conclusion
We study SLM orchestration through three decisions: whom to trust, how to aggregate evidence,
and which form of agreement to use. The best choice depends on both the task and the model
pool. Expert weighting is the strongest routing factor, larger pools do not necessarily improve
accuracy, and agreement is useful only when it comes from helpful models. These results suggest
that effective orchestration requires jointly selecting the models and deciding how their evidence
should be combined.
Limitations
Our study is limited to cache-only, non-interactive orchestration and benchmarks with extractable
answers. Extending the framework to interactive or open-ended settings would require additional
generations and semantic equivalence mechanisms. Expert priors also require validation data or an
estimate of model quality. Finally, while our all-subset analysis explores the pool design space, we
use split-selected evaluation to test pool selection on held-out data.
A natural next step is to use routing signals to decide when additional interaction, verification, or
debate is worth the compute. The same signals could support adaptive pool construction, selecting
models and routing strategies at the question or domain level. Extending OrchSLM to open-ended
generation would further require semantic matching and partial-credit evaluation while preserving
interpretable influence diagnostics.
Disclosure of Generative AI Usage
We used large language models to assist with writing clarity, grammar, and experimental code
development. All final wording, technical claims, code, results, figures’ scientific content, and
conclusions were reviewed, validated, and produced by the authors.
References
Marah Abdin et al. Phi-3 technical report: A highly capable language model locally on your phone.
arXiv preprint arXiv:2404.14219, 2024.
Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo,
Lewis Tunstall, Andrés Marafioti, Hynek Kydlíˇcek, Agustín Piqueres Lajarín, Vaibhav Srivastav,
et al. Smollm2: When smol goes big–data-centric training of a small language model.arXiv
preprint arXiv:2502.02737, 2025.
Kenneth J. Arrow.Social Choice and Individual Values. John Wiley & Sons, New York, 1951.
Peter Belcak, Greg Heinrich, Shizhe Diao, Yonggan Fu, Xin Dong, Saurav Muralidharan, Yingyan Ce-
line Lin, and Pavlo Molchanov. Small language models are the future of agentic ai.arXiv preprint
arXiv:2506.02153, 2025.
Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V . Le, Christopher Ré, and
Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling.
arXiv preprint arXiv:2407.21787, 2024.
Lingjiao Chen, Matei Zaharia, and James Zou. Frugalgpt: How to use large language models while
reducing cost and improving performance, 2023. URL https://arxiv.org/abs/2305.05176.
Lingjiao Chen, Jared Quincy Davis, Boris Hanin, Peter Bailis, Matei Zaharia, James Zou, and Ion
Stoica. Optimizing model selection for compound AI systems.arXiv preprint arXiv:2502.14815,
2025.
9

PAGE 10
Shuhao Chen, Weisen Jiang, Baijiong Lin, James T. Kwok, and Yu Zhang. Routerdc: Query-
based router by dual contrastive learning for assembling large language models, 2024. URL
https://arxiv.org/abs/2409.19886.
DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning.
arXiv preprint arXiv:2501.12948, 2025.
Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving
factuality and reasoning in language models through multiagent debate, 2023. URL https:
//arxiv.org/abs/2305.14325.
Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, and Yu Wang.
Cache-to-cache: Direct semantic communication between large language models.arXiv preprint
arXiv:2510.03215, 2025.
Gemma Team. Gemma 2: Improving open language models at a practical size.arXiv preprint
arXiv:2408.00118, 2024.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of
models.arXiv preprint arXiv:2407.21783, 2024.
Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza
Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al.
Training compute-optimal large language models.Advances in Neural Information Processing
Systems, 35:30016–30030, 2022.
Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song,
and Denny Zhou. Large language models cannot self-correct reasoning yet, 2024. URL https:
//arxiv.org/abs/2310.01798.
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child,
Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models.
arXiv preprint arXiv:2001.08361, 2020.
Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vard-
hamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller,
Matei Zaharia, and Christopher Potts. Dspy: Compiling declarative language model calls into
self-improving pipelines, 2023. URLhttps://arxiv.org/abs/2310.03714.
Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions.
InInternational Conference on Machine Learning, pages 1885–1894, 2017. URL https://
proceedings.mlr.press/v70/koh17a.html.
Dawei Li, Zhen Tan, Peijia Qian, Yifan Li, Kumar Satvik Chaudhary, Lijie Hu, and Jiayi Shen.
SMoA: Improving multi-agent large language models with sparse mixture-of-agents. InPacific-
Asia Conference on Knowledge Discovery and Data Mining, pages 54–65, 2025a.
Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, and Deheng Ye. More agents is all you need, 2024.
URLhttps://arxiv.org/abs/2402.05120.
Wenzhe Li, Yong Lin, Mengzhou Xia, and Chi Jin. Rethinking mixture-of-agents: Is mixing different
large language models beneficial?arXiv preprint arXiv:2502.00674, 2025b.
Shalev Lifshitz, Sheila A. McIlraith, and Yilun Du. Multi-agent verification: Scaling test-time
compute with multiple verifiers, 2025. URLhttps://arxiv.org/abs/2502.20379.
Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan
Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. InInternational
Conference on Learning Representations, 2024.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni,
and Percy Liang. Lost in the middle: How language models use long contexts, 2023. URL
https://arxiv.org/abs/2307.03172.
10

PAGE 11
Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang
Xiong, Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, Liangzhen Lai, and Vikas
Chandra. MobileLLM: Optimizing sub-billion parameter language models for on-device use cases.
InProceedings of the 41st International Conference on Machine Learning, 2024.
Scott M. Lundberg and Su-In Lee. A unified approach to interpreting model predictions. InAdvances
in Neural Information Processing Systems, 2017. URL https://proceedings.neurips.
cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.
html.
Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke
Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time
scaling, 2025. URLhttps://arxiv.org/abs/2501.19393.
Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez,
M Waleed Kadous, and Ion Stoica. Routellm: Learning to route llms with preference data, 2025.
URLhttps://arxiv.org/abs/2406.18665.
OpenAI. GPT-4 technical report.arXiv preprint arXiv:2303.08774, 2023.
Krista Opsahl-Ong, Michael J Ryan, Josh Purtell, David Broman, Christopher Potts, Matei Zaharia,
and Omar Khattab. Optimizing instructions and demonstrations for multi-stage language model
programs, 2024. URLhttps://arxiv.org/abs/2406.11695.
Amruta Parulekar, Jinu Lee, Dilek Hakkani-Tür, and Hari Sundaram. Reasoning consensus: Structural
ensembling of LLM reasoning via weighted DAG aggregation.arXiv preprint arXiv:2607.27783,
2026.
Heng Ping, Arijit Bhattacharjee, Peiyu Zhang, Shixuan Li, Wei Yang, Ali Jannesari, Nesreen Ahmed,
and Paul Bogdan. Rem-MoA: Reasoning memory sustains mixture-of-agents scaling.arXiv
preprint arXiv:2606.24437, 2026.
Manhin Poon, Xiangxiang Dai, Xutong Liu, Fang Kong, John C. S. Lui, and Jinhang Zuo. Online
multi-LLM selection via contextual bandits under unstructured context evolution.arXiv preprint
arXiv:2506.17670, 2025.
Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. “why should I trust you?”: Explaining the
predictions of any classifier. InProceedings of the 22nd ACM SIGKDD International Conference
on Knowledge Discovery and Data Mining, pages 1135–1144, 2016. URL https://doi.org/
10.1145/2939672.2939778.
Jon Saad-Falcon, Adrian Gamarra Lafuente, Shlok Natarajan, Nahum Maru, Hristo Todorov, Etash
Guha, E. Kelly Buchanan, Mayee Chen, Neel Guha, Christopher Ré, and Azalia Mirhoseini.
Archon: An architecture search framework for inference-time techniques, 2025. URL https:
//arxiv.org/abs/2409.15254.
Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally
can be more effective than scaling model parameters.arXiv preprint arXiv:2408.03314, 2024a.
Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally
can be more effective than scaling model parameters, 2024b. URL https://arxiv.org/abs/
2408.03314.
Zhende Song, Shengji Tang, Peng Ye, Jiayuan Fan, and Tao Chen. CTTS: Collective test-time scaling.
arXiv preprint arXiv:2508.03333, 2025.
Amir Taubenfeld, Yaniv Dover, Roi Reichart, and Ariel Goldstein. Systematic biases in llm simula-
tions of debates. InProceedings of the 2024 Conference on Empirical Methods in Natural Language
Processing, page 251–267. Association for Computational Linguistics, 2024. doi: 10.18653/v1/
2024.emnlp-main.16. URLhttp://dx.doi.org/10.18653/v1/2024.emnlp-main.16.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation
and fine-tuned chat models.arXiv preprint arXiv:2307.09288, 2023.
11

PAGE 12
Chenyu Wang, Zishen Wan, Hao Kang, Emma Chen, Zhiqiang Xie, Tushar Krishna, Vijay
Janapa Reddi, and Yilun Du. Slm-mux: Orchestrating small language models for reasoning.
CoRR, abs/2510.05077, 2025. URLhttps://arxiv.org/abs/2510.05077.
Fali Wang, Zhiwei Zhang, Xianren Zhang, Zongyu Wu, Tzuhao Mo, Qiuhao Lu, Wanjing Wang, Rui
Li, Junjie Xu, Xianfeng Tang, Qi He, Yao Ma, Ming Huang, and Suhang Wang. A comprehensive
survey of small language models in the era of large language models: Techniques, enhancements,
applications, collaboration with llms, and trustworthiness, 2024a. URL https://arxiv.org/
abs/2411.03350.
Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou. Mixture-of-agents enhances
large language model capabilities, 2024b. URLhttps://arxiv.org/abs/2406.04692.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha
Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in lan-
guage models. InInternational Conference on Learning Representations, 2023. URL https:
//arxiv.org/abs/2203.11171.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc
Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In
Advances in Neural Information Processing Systems, 2022. URL https://arxiv.org/abs/
2201.11903.
Jianyu Wen, Yang Wei, Xiongxi Yu, Changxuan Xiao, and Ke Zeng. Attention-MoA: Enhancing
mixture-of-agents via inter-agent semantic attention and deep residual synthesis.arXiv preprint
arXiv:2601.16596, 2026.
Zhentao Xie, Chengcheng Han, Jinxin Shi, Wenjun Cui, Xin Zhao, Xingjiao Wu, and Jiabao Zhao.
RMoA: Optimizing mixture-of-agents through diversity maximization and residual compensation.
InFindings of the Association for Computational Linguistics: ACL 2025, pages 6575–6602, 2025.
doi: 10.18653/v1/2025.findings-acl.342.
Yu Yao, Jiayi Dong, Yang Yang, Ju Li, and Yilun Du. Roundtable policy: Confidence-weighted-
consensus aggregation improves multi-agent-system reasoning, 2026. URL https://arxiv.
org/abs/2509.16839.
Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen
Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin
Wu. Aflow: Automating agentic workflow generation, 2025a. URL https://arxiv.org/abs/
2410.10762.
Qiyuan Zhang, Fuyuan Lyu, Zexu Sun, Lei Wang, Weixu Zhang, Wenyue Hua, Haolun Wu, Zhihan
Guo, Yufei Wang, Niklas Muennighoff, Irwin King, Xue Liu, and Chen Ma. A survey on
test-time scaling in large language models: What, how, where, and how well?, 2025b. URL
https://arxiv.org/abs/2503.24235.
12

PAGE 13
Appendices
Overview.The first sections provide protocol details, full-pool analyses, pool-composition results,
robustness checks, and Atr/Ctr diagnostics for the seven-model experiments. The final sections
collect additional mechanism and sensitivity tables that support the same interpretation from smaller
or alternative analysis settings.
A Mathematical Interpretation
Generalized routing score contains two sources of evidence:
Top-answer support.The first term,
Aggi [Mia ri(κ)P ia(x)],(7)
only counts model i’s support for answer a if a is that model’s own top answer. This term measures
how strongly the models that actually select a support it. The operator Agg determines how this
support is combined across models:
•Sumadds support from all models whose top answer isa:
Sumi[zi] =
X
i
zi.(8)
This corresponds to coalition-style evidence, where several models can jointly support the
same answer.
•Maxkeeps only the strongest individual supporter:
Maxi[zi] = maxi zi.(9)
This corresponds to strongest-supporter evidence, where one highly confident or highly
trusted model can dominate.
Pooled answer support.The second term,
X
i
ri(κ)Pia(x),(10)
counts every model’s probability mass on answer a, even if a is not that model’s top answer. This
term measures cross-model or mutual support fora.
The interpolation parameter α∈[0,1] controls which type of evidence dominates. When α= 1 ,
routing relies only on top-answer support. When α= 0 , routing relies only on pooled support. Thus,
κ controls whom to trust, Agg controls whether evidence comes from a coalition or the strongest
supporter, and α controls whether confidence comes from top-answer self-consistency or pooled
mutual agreement.
B Experimental Details
B.1 Stability and Robustness Check
The conclusions in experiments section persist under held-out pool selection (Section B.4, Table 20),
subsample-replicate variability (Table 24, Table 23), and full-cache K=50 evaluation (Tables 28
and 29). Prior-transform, tie-breaking, exact cross-support, and fix/break checks are reported in
Section B.5.
B.2 Protocol Details and Summary Extraction
The main experiments use K20R10: for each question and model, the evaluation samples 20 answers
from a K=50 cache and repeats this deterministic subsampling protocol 10 times. Tables report
13

PAGE 14
means over replicates unless otherwise noted. The main generalized sweep uses fixed expert priors to
analyze routing assumptions. The all-subset pool scans study how pool composition changes routed
accuracy; split-selected evaluation tests the same selection idea under held-out evaluation. The small
main-text summary tables extract views of the same full-pool sweep to answer the three routing
questions directly.
Table 8: Task-specific routing regimes from the full-pool generalized sweep. The table summarizes
broad preferred regions; near-best plateau statistics are reported separately.
Benchmark Best acc.α κAgg. Interpretation
MATH-500 0.7938 0.6842 7.0171 Sum single-best-like expert routing
GSM8K 0.9166 1.0000 7.0171 Max expert self-consistency
GPQA 0.4328 0.3684 8.0000 Max expert + support-aware routing
MMLU-Pro 0.6402 0.1053 5.3617 Max expert-weighted pooled support
14

PAGE 15
B.3 Full-Pool Routing Decisions
This section contains the full-pool analyses behind the main three-decision analysis. The tables
report plateau width, single-best gaps, no-expert-prior ablations, pooled-term and top-term ablations,
Sum-vs-Max comparisons, and disagreement-bin behavior. These numbers are the table counterpart
of the main heatmap and regime discussion.
Table 9: Full-pool plateau summary. Broad near-best regions support interpreting best cells as regimes
with stable neighboring cells.
Benchmark Best acc. Best params Cells @0.25pp Area @0.25pp Cells @0.5pp Area @0.5pp Cells @1pp Area @1pp
GPQA 0.4328 Max a=0.3684 k=8.0 19 0.0238 78 0.0975 158 0.1975
GSM8K 0.9166 Max a=1.0 k=7.0171 15 0.0187 39 0.0488 210 0.2625
MATH-500 0.7938 Sum a=0.6842 k=7.0171 137 0.1713 199 0.2487 273 0.3412
MMLU-Pro 0.6402 Max a=0.1053 k=5.3617 38 0.0475 213 0.2662 566 0.7075
Table 10: Expert-prior axis: gap between the best generalized full-pool cell and the single-best limit.
Benchmark Best gen. Single-best Gap Bestκ
MATH-500 0.7938 0.7924 0.0014 7.0171
GSM8K 0.9166 0.9089 0.0077 7.0171
GPQA 0.4328 0.4015 0.0313 8.0000
MMLU-Pro 0.6402 0.6158 0.0244 5.3617
Table 11: No-expert-prior comparison on the full seven-model pool, read from the κ= 0 best cell in
the full-pool phase diagram.
Benchmark Bestκ=0Std Agg.αFull-family best Gap
MATH-500 0.7338 0.0055 Max 0.7368 0.7938 -0.0600
GSM8K 0.9052 0.0013 Max 0.8947 0.9166 -0.0114
GPQA 0.3909 0.0120 Sum 0.6316 0.4328 -0.0419
MMLU-Pro 0.6304 0.0069 Sum 0.2632 0.6402 -0.0098
Table 12: Top-answer-term removal ablation on the full seven-model pool. Setting α= 0 keeps only
pooled answer support.
Benchmark Bestα=0Agg.κFull-family best Gap
GPQA 0.4318 Sum 7.0171 0.4328 -0.0010
GSM8K 0.9086 Sum 8.0000 0.9166 -0.0080
MATH-500 0.7934 Sum 8.0000 0.7938 -0.0004
MMLU-Pro 0.6398 Sum 5.3617 0.6402 -0.0004
Table 13: Pooled-term removal ablation on the full seven-model pool. Setting α= 1 removes pooled
mutual support.
Benchmark Bestα=1Agg.κFull-family best Gap
GPQA 0.4308 Sum 6.1416 0.4328 -0.0020
GSM8K 0.9166 Max 7.0171 0.9166 0.0000
MATH-500 0.7936 Sum 6.1416 0.7938 -0.0002
MMLU-Pro 0.6350 Sum 5.3617 0.6402 -0.0052
15

PAGE 16
Table 14: Sum/Max summary. The aggregation switch measures whether support is distributed across
models or concentrated in the strongest supporter.
benchmark benchmark_label pool_size mean_sum_minus_max frac_sum_wins
gpqa GPQA 1.0000 0.0000 0.0000
gpqa GPQA 2.0000 0.0000 0.0000
gpqa GPQA 3.0000 -0.0002 0.2857
gpqa GPQA 4.0000 0.0004 0.5143
gpqa GPQA 5.0000 -0.0002 0.4286
gpqa GPQA 6.0000 -0.0006 0.2857
gpqa GPQA 7.0000 -0.0010 0.0000
gsm8k GSM8K 1.0000 0.0000 0.0000
gsm8k GSM8K 2.0000 0.0000 0.0000
gsm8k GSM8K 3.0000 -0.0031 0.2857
gsm8k GSM8K 4.0000 -0.0024 0.2000
gsm8k GSM8K 5.0000 -0.0053 0.0952
gsm8k GSM8K 6.0000 -0.0049 0.0000
gsm8k GSM8K 7.0000 -0.0077 0.0000
math500 MATH-500 1.0000 0.0000 0.0000
math500 MATH-500 2.0000 0.0000 0.0000
math500 MATH-500 3.0000 0.0000 0.1429
math500 MATH-500 4.0000 0.0000 0.2571
math500 MATH-500 5.0000 0.0001 0.3810
math500 MATH-500 6.0000 0.0002 0.5714
Table 15: Full-pool best generalized accuracy by pair-disagreement bin.
Benchmark Disagreement binNAccuracy
GPQA (-0.01, 0.2] 80 0.5250
GPQA (0.2, 0.4] 214 0.4346
GPQA (0.4, 0.6] 587 0.4889
GPQA (0.6, 0.8] 795 0.4050
GPQA (0.8, 1.01] 304 0.3717
GSM8K (-0.01, 0.2] 8882 0.9895
GSM8K (0.2, 0.4] 1894 0.9018
GSM8K (0.4, 0.6] 1513 0.7072
GSM8K (0.6, 0.8] 515 0.6738
GSM8K (0.8, 1.01] 386 0.4560
MATH-500 (-0.01, 0.2] 1700 0.9859
MATH-500 (0.2, 0.4] 715 0.9259
MATH-500 (0.4, 0.6] 610 0.8361
MATH-500 (0.6, 0.8] 605 0.7008
MATH-500 (0.8, 1.01] 1370 0.5088
MMLU-Pro (-0.01, 0.2] 659 0.8998
MMLU-Pro (0.2, 0.4] 845 0.7988
MMLU-Pro (0.4, 0.6] 1262 0.7013
MMLU-Pro (0.6, 0.8] 1056 0.5284
MMLU-Pro (0.8, 1.01] 1178 0.4160
16

PAGE 17
B.4 Pool Composition and Structure
This section tests whether the model pool itself changes the routing problem. The tables compare
pools selected for absolute accuracy with pools selected for gain over the self-consistency router,
classify scanned pools by routing regime, relate structural pool metrics to routing gains, and evaluate
split-selected pool choice under held-out evaluation.
Table 16: Pool-composition targets. For each benchmark we report the pool with highest absolute
accuracy and the pool with largest gain over the self-consistency router.
Benchmark Target Pool Size Best gen. Std∆vs SLM∆vs SB Params Regime
GPQA best acc. qwen+llama+gemma+ministral 4 0.4429 0.0093 0.0212 0.0414 Sum a=0.1053 k=8.0 high-kappa expert/single-best-like
GPQA best gain qwen+tulu+dsmath+phi 4 0.4333 0.0182 0.1101 0.0318 Sum a=1.0 k=3.4968 mixed support-aware
GSM8K best acc. qwen+llama+dsmath 3 0.9327 0.0018 0.0060 0.0238 Sum a=0.4211 k=4.6669 high-kappa expert/single-best-like
GSM8K best gain qwen+gemma+ministral 3 0.9111 0.0021 0.0227 0.0022 Max a=0.9474 k=8.0 no stable gain
MATH-500 best acc. qwen+llama+dsmath+gemma 4 0.7952 0.0053 0.0558 0.0028 Max a=0.0526 k=6.1416 no stable gain
MATH-500 best gain qwen+llama+tulu+dsmath+gemma+phi+ministral 7 0.7938 0.0049 0.0696 0.0014 Sum a=0.6842 k=7.0171 no stable gain
MMLU-Pro best acc. qwen+llama+dsmath+phi 4 0.6526 0.0052 0.0652 0.0368 Sum a=0.0 k=1.0014 low/moderate-alpha Sum coalition/Majority-like
MMLU-Pro best gain llama+dsmath 2 0.5646 0.0078 0.0936 0.0060 Sum a=0.0 k=1.5222 low/moderate-alpha Sum coalition/Majority-like
Table 17: Best absolute pool from the all-subset pool-composition scan. The companion figure shows
how accuracy changes with pool size, while this table records the selected model names.
Benchmark Best absolute pool Size Best acc. Full-pool best
MATH-500 qwen+llama+dsmath+gemma 4 0.7952 0.7938
GSM8K qwen+llama+dsmath 3 0.9327 0.9166
GPQA qwen+llama+gemma+ministral 4 0.4429 0.4328
MMLU-Pro qwen+llama+dsmath+phi 4 0.6526 0.6402
Table 18: Regime classification over all scanned pools. Fractions are within benchmark.
Benchmark Regime Pools Frac.
GPQA high-alpha Max self-consistency-like 8 0.063
GPQA high-kappa expert/single-best-like 72 0.567
GPQA low/moderate-alpha Sum coalition/Majority-like 12 0.094
GPQA mixed support-aware 14 0.110
GPQA no stable gain 21 0.165
GSM8K high-alpha Max self-consistency-like 3 0.024
GSM8K high-kappa expert/single-best-like 74 0.583
GSM8K low/moderate-alpha Sum coalition/Majority-like 12 0.094
GSM8K mixed support-aware 4 0.031
GSM8K no stable gain 34 0.268
MATH-500 high-kappa expert/single-best-like 47 0.370
MATH-500 low/moderate-alpha Sum coalition/Majority-like 10 0.079
MATH-500 no stable gain 70 0.551
MMLU-Pro high-kappa expert/single-best-like 64 0.504
MMLU-Pro low/moderate-alpha Sum coalition/Majority-like 44 0.346
MMLU-Pro mixed support-aware 8 0.063
MMLU-Pro no stable gain 11 0.087
17

PAGE 18
Table 19: Structure-predictor correlations between pool metrics and generalized-routing gains.
benchmark_label predictor response pearson_r
GPQA pool_size delta_best_vs_slm 0.6762
GPQA pool_size delta_best_vs_single_best 0.5491
GPQA pool_size delta_best_vs_majority 0.3595
GPQA pool_size best_gen_acc_mean 0.5517
GPQA oracle_gap_mean delta_best_vs_slm 0.6899
GPQA oracle_gap_mean delta_best_vs_single_best 0.5402
GPQA oracle_gap_mean delta_best_vs_majority 0.3930
GPQA oracle_gap_mean best_gen_acc_mean 0.5632
GPQA wrong_answer_agreement_mean delta_best_vs_slm 0.6572
GPQA wrong_answer_agreement_mean delta_best_vs_single_best 0.5492
GPQA wrong_answer_agreement_mean delta_best_vs_majority 0.3542
GPQA wrong_answer_agreement_mean best_gen_acc_mean 0.5714
GPQA avg_pair_disagreement_mean delta_best_vs_slm 0.5734
GPQA avg_pair_disagreement_mean delta_best_vs_single_best 0.2224
GPQA avg_pair_disagreement_mean delta_best_vs_majority 0.5663
GPQA avg_pair_disagreement_mean best_gen_acc_mean 0.3251
GPQA stable_wrong_rate_mean delta_best_vs_slm 0.8332
GPQA stable_wrong_rate_mean delta_best_vs_single_best 0.4150
GPQA stable_wrong_rate_mean delta_best_vs_majority 0.5489
GPQA stable_wrong_rate_mean best_gen_acc_mean 0.4800
GPQA single_model_acc_mean delta_best_vs_slm -0.3758
GPQA single_model_acc_mean delta_best_vs_single_best 0.4190
GPQA single_model_acc_mean delta_best_vs_majority -0.4097
GPQA single_model_acc_mean best_gen_acc_mean 0.6045
GPQA single_model_acc_std delta_best_vs_slm 0.7475
GPQA single_model_acc_std delta_best_vs_single_best 0.0559
GPQA single_model_acc_std delta_best_vs_majority 0.7916
GPQA single_model_acc_std best_gen_acc_mean 0.2273
GPQA single_model_acc_spread delta_best_vs_slm 0.8318
GPQA single_model_acc_spread delta_best_vs_single_best 0.1763
GPQA single_model_acc_spread delta_best_vs_majority 0.7839
GPQA single_model_acc_spread best_gen_acc_mean 0.3234
GSM8K pool_size delta_best_vs_slm 0.4439
GSM8K pool_size delta_best_vs_single_best 0.0014
GSM8K pool_size delta_best_vs_majority 0.3702
GSM8K pool_size best_gen_acc_mean 0.5492
GSM8K oracle_gap_mean delta_best_vs_slm 0.1643
GSM8K oracle_gap_mean delta_best_vs_single_best 0.7245
GSM8K oracle_gap_mean delta_best_vs_majority -0.0527
GSM8K oracle_gap_mean best_gen_acc_mean 0.2886
Table 20: Split-selected evaluation. Variants differ in whether they select only scorer parameters on
the full pool, choose from predefined curated pools, or choose from all subsets.
Benchmark Variant Eval acc. Eval std Pool size Delta vs SC Delta vs single Delta vs majority
GPQA A_full_pool_select_scorer 0.4120 0.0309 7.0000 0.0760 0.0054 0.0253
GPQA B_predefined_curated_select_pool_scorer 0.4167 0.0281 3.6200 0.0807 0.0101 0.0300
GPQA C_all_subset_select_pool_scorer 0.4075 0.0319 3.9800 0.0715 0.0009 0.0208
GSM8K A_full_pool_select_scorer 0.9149 0.0071 7.0000 0.0114 0.0047 0.0112
GSM8K B_predefined_curated_select_pool_scorer 0.9311 0.0066 3.0000 0.0277 0.0210 0.0275
GSM8K C_all_subset_select_pool_scorer 0.9306 0.0070 3.0600 0.0271 0.0204 0.0269
MATH-500 A_full_pool_select_scorer 0.7868 0.0161 7.0000 0.0677 -0.0003 0.0635
MATH-500 B_predefined_curated_select_pool_scorer 0.7859 0.0154 3.4200 0.0668 -0.0011 0.0627
MATH-500 C_all_subset_select_pool_scorer 0.7837 0.0157 3.3400 0.0646 -0.0034 0.0604
MMLU-Pro A_full_pool_select_scorer 0.6300 0.0225 7.0000 0.0577 0.0147 0.0010
MMLU-Pro B_predefined_curated_select_pool_scorer 0.6375 0.0231 4.1200 0.0653 0.0222 0.0086
MMLU-Pro C_all_subset_select_pool_scorer 0.6339 0.0228 4.2400 0.0617 0.0186 0.0050
18

PAGE 19
B.5 Fix/Break, Robustness, and Ablations
This section stress-tests the main conclusions. The fix/break table separates questions corrected
by generalized routing from questions newly broken by it, including comparisons against several
canonical reference routers. Corrected significance tests check whether selected-pool gains survive
multiple-baseline correction. Replicate-stability tables summarize K20R10 variability. The remaining
tables vary expert priors, tie-breaking and invalid-answer handling, exact cross-support bridge
strength, K=50 evaluation, effective sample budget, and method-correlation structure.
Table 21: Fix/break summary for full and best-absolute pools. Counts are over all K20R10 question-
replicate rows.
Benchmark Pool ComparisonNFix Break Target acc. Base acc. Net
GPQA full vs slm 1980 341 151 0.4328 0.3369 0.0960
GPQA full vs single 1980 155 93 0.4328 0.4015 0.0313
GPQA full vs Majority 1980 199 107 0.4328 0.3864 0.0465
GPQA best_abs vs slm 1980 139 97 0.4429 0.4217 0.0212
GPQA best_abs vs single 1980 163 81 0.4429 0.4015 0.0414
GPQA best_abs vs Majority 1980 108 59 0.4429 0.4182 0.0247
GSM8K full vs slm 13190 328 150 0.9166 0.9031 0.0135
GSM8K full vs single 13190 230 128 0.9166 0.9089 0.0077
GSM8K full vs Majority 13190 298 121 0.9166 0.9032 0.0134
GSM8K best_abs vs slm 13190 150 71 0.9327 0.9267 0.0060
GSM8K best_abs vs single 13190 447 133 0.9327 0.9089 0.0238
GSM8K best_abs vs Majority 13190 58 56 0.9327 0.9325 0.0002
MATH-500 full vs slm 5000 384 36 0.7938 0.7242 0.0696
MATH-500 full vs single 5000 9 2 0.7938 0.7924 0.0014
MATH-500 full vs Majority 5000 425 79 0.7938 0.7246 0.0692
MMLU-Pro full vs slm 5000 532 200 0.6402 0.5738 0.0664
MMLU-Pro full vs single 5000 280 158 0.6402 0.6158 0.0244
MMLU-Pro full vs Majority 5000 217 157 0.6402 0.6282 0.0120
Table 22: Selected-pool paired tests with max-stat correction across baselines within each bench-
mark/pool role.
Benchmark Pool role Baseline∆Correctedp
GPQA best_abs SLM-MUX_best_abs 0.0212 0.009
GPQA best_abs single-best_best_abs 0.0414 0.000
GPQA best_abs Majority_best_abs 0.0247 0.001
GPQA best_abs cross slice_best_abs 0.0268 0.000
GPQA best_gap SLM-MUX_best_gap 0.1101 0.000
GPQA best_gap single-best_best_gap 0.0318 0.003
GPQA best_gap Majority_best_gap 0.0753 0.000
GPQA best_gap cross slice_best_gap 0.0889 0.000
GSM8K best_abs SLM-MUX_best_abs 0.0060 0.001
GSM8K best_abs single-best_best_abs 0.0238 0.000
GSM8K best_abs Majority_best_abs 0.0002 0.826
GSM8K best_abs cross slice_best_abs 0.0003 0.765
19

PAGE 20
Table 23: Replicate stability. Means and standard deviations are over the 10 K20R10 subsampling
replicates.
Benchmark Pool role Method Mean acc. Std
GPQA best_abs Majority 0.4182 0.0144
GPQA best_abs Single-Best 0.4015 0.0170
GPQA best_abs SLM-MUX 0.4217 0.0122
GPQA best_gap Majority 0.3581 0.0155
GPQA best_gap Single-Best 0.4015 0.0170
GPQA best_gap SLM-MUX 0.3232 0.0203
GPQA full Majority 0.3864 0.0072
GPQA full Single-Best 0.4015 0.0170
GPQA full SLM-MUX 0.3369 0.0163
GSM8K best_abs Majority 0.9325 0.0009
GSM8K best_abs Single-Best 0.9089 0.0025
GSM8K best_abs SLM-MUX 0.9267 0.0019
GSM8K full Majority 0.9032 0.0019
GSM8K full Single-Best 0.9089 0.0025
GSM8K full SLM-MUX 0.9031 0.0019
MATH-500 full Majority 0.7246 0.0077
MATH-500 full Single-Best 0.7924 0.0051
MATH-500 full SLM-MUX 0.7242 0.0056
MMLU-Pro full Majority 0.6282 0.0058
MMLU-Pro full Single-Best 0.6158 0.0061
MMLU-Pro full SLM-MUX 0.5738 0.0089
Table 24: Robustness checks. Split-selected results use held-out evaluation with predefined curated
pools; K=50 uses all cached samples once. The direction of the regimes persists under both checks.
Benchmark Split-selected acc.∆vs self-consistency K=50 best gen. K=50 self-consistency
MATH-500 0.7859 +0.0668 0.8000 0.7240
GSM8K 0.9311 +0.0277 0.9189 0.9052
GPQA 0.4167 +0.0807 0.4394 0.3030
MMLU-Pro 0.6375 +0.0653 0.6480 0.5580
Table 25: Expert-prior transform variants. These checks test whether the qualitative regimes depend
on a single prior transformation.
benchmark benchmark_label pool_role pool_id prior_variant best_acc best_agg best_kappa best_alpha
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral softmax_log_accuracy 0.7938 Sum 7.0171 0.6842
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral power_accuracy 0.7938 Sum 7.0171 0.6842
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral centered_logit 0.7938 Sum 2.1786 0.7368
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral rank_based 0.7936 Sum 5.3617 0.9474
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral uniform 0.7338 Max 0.0000 0.7368
gsm8k GSM8K full qwen+llama+tulu+dsmath+gemma+phi+ministral softmax_log_accuracy 0.9166 Max 7.0171 1.0000
gsm8k GSM8K full qwen+llama+tulu+dsmath+gemma+phi+ministral power_accuracy 0.9166 Max 7.0171 1.0000
gsm8k GSM8K full qwen+llama+tulu+dsmath+gemma+phi+ministral centered_logit 0.9177 Sum 4.0481 0.0000
gsm8k GSM8K full qwen+llama+tulu+dsmath+gemma+phi+ministral rank_based 0.9187 Max 4.0481 0.0526
gsm8k GSM8K full qwen+llama+tulu+dsmath+gemma+phi+ministral uniform 0.9052 Max 0.0000 0.8947
gsm8k GSM8K best_abs qwen+llama+dsmath softmax_log_accuracy 0.9327 Sum 4.6669 0.4211
gsm8k GSM8K best_abs qwen+llama+dsmath power_accuracy 0.9327 Sum 4.6669 0.4211
gsm8k GSM8K best_abs qwen+llama+dsmath centered_logit 0.9326 Sum 0.5882 0.4737
gsm8k GSM8K best_abs qwen+llama+dsmath rank_based 0.9326 Sum 0.2602 0.5789
gsm8k GSM8K best_abs qwen+llama+dsmath uniform 0.9325 Sum 0.0000 0.0000
gpqa GPQA full qwen+llama+tulu+dsmath+gemma+phi+ministral softmax_log_accuracy 0.4328 Max 8.0000 0.3684
gpqa GPQA full qwen+llama+tulu+dsmath+gemma+phi+ministral power_accuracy 0.4328 Max 8.0000 0.3684
gpqa GPQA full qwen+llama+tulu+dsmath+gemma+phi+ministral centered_logit 0.4348 Max 8.0000 0.0526
gpqa GPQA full qwen+llama+tulu+dsmath+gemma+phi+ministral rank_based 0.4379 Sum 3.4968 0.0000
gpqa GPQA full qwen+llama+tulu+dsmath+gemma+phi+ministral uniform 0.3909 Sum 0.0000 0.6316
gpqa GPQA best_abs qwen+llama+gemma+ministral softmax_log_accuracy 0.4429 Sum 8.0000 0.1053
gpqa GPQA best_abs qwen+llama+gemma+ministral power_accuracy 0.4429 Sum 8.0000 0.1053
gpqa GPQA best_abs qwen+llama+gemma+ministral centered_logit 0.4424 Sum 4.6669 0.1053
gpqa GPQA best_abs qwen+llama+gemma+ministral rank_based 0.4419 Max 1.0014 0.2632
20

PAGE 21
Table 26: Tie-break sensitivity. Tie-breaking changes small details while preserving the main regime
interpretation.
benchmark benchmark_label pool_role pool_id tie_break accuracy_mean accuracy_std
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral pooled_lex 0.7938 0.0049
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral lex 0.7938 0.0049
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral fixed_model_order 0.7938 0.0049
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral random 0.7938 0.0049
gsm8k GSM8K full qwen+llama+tulu+dsmath+gemma+phi+ministral pooled_lex 0.9166 0.0017
gsm8k GSM8K full qwen+llama+tulu+dsmath+gemma+phi+ministral lex 0.9166 0.0017
gsm8k GSM8K full qwen+llama+tulu+dsmath+gemma+phi+ministral fixed_model_order 0.9166 0.0017
gsm8k GSM8K full qwen+llama+tulu+dsmath+gemma+phi+ministral random 0.9166 0.0017
gsm8k GSM8K best_abs qwen+llama+dsmath pooled_lex 0.9327 0.0018
gsm8k GSM8K best_abs qwen+llama+dsmath lex 0.9327 0.0018
gsm8k GSM8K best_abs qwen+llama+dsmath fixed_model_order 0.9327 0.0018
gsm8k GSM8K best_abs qwen+llama+dsmath random 0.9327 0.0018
gpqa GPQA full qwen+llama+tulu+dsmath+gemma+phi+ministral pooled_lex 0.4328 0.0096
gpqa GPQA full qwen+llama+tulu+dsmath+gemma+phi+ministral lex 0.4328 0.0096
gpqa GPQA full qwen+llama+tulu+dsmath+gemma+phi+ministral fixed_model_order 0.4328 0.0096
gpqa GPQA full qwen+llama+tulu+dsmath+gemma+phi+ministral random 0.4328 0.0096
gpqa GPQA best_abs qwen+llama+gemma+ministral pooled_lex 0.4429 0.0093
gpqa GPQA best_abs qwen+llama+gemma+ministral lex 0.4429 0.0093
gpqa GPQA best_abs qwen+llama+gemma+ministral fixed_model_order 0.4429 0.0093
gpqa GPQA best_abs qwen+llama+gemma+ministral random 0.4429 0.0093
gpqa GPQA best_gap qwen+tulu+dsmath+phi pooled_lex 0.4333 0.0182
gpqa GPQA best_gap qwen+tulu+dsmath+phi lex 0.4333 0.0182
gpqa GPQA best_gap qwen+tulu+dsmath+phi fixed_model_order 0.4333 0.0182
gpqa GPQA best_gap qwen+tulu+dsmath+phi random 0.4333 0.0182
Table 27: Exact cross-support bridge sweep. This sweep studies how the bridge varies with cross-
support strength.
benchmark benchmark_label pool_role pool_id lambda accuracy_mean accuracy_std
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 0.0 0.7296 0.0043
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 0.16666666666666666 0.7304 0.0042
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 0.25 0.7334 0.0059
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 0.5 0.7280 0.0044
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 1.0 0.7236 0.0071
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 2.0 0.7202 0.0070
gsm8k GSM8K full qwen+llama+tulu+dsmath+gemma+phi+ministral 0.0 0.9096 0.0016
gsm8k GSM8K full qwen+llama+tulu+dsmath+gemma+phi+ministral 0.16666666666666666 0.9050 0.0014
gsm8k GSM8K full qwen+llama+tulu+dsmath+gemma+phi+ministral 0.25 0.9045 0.0017
gsm8k GSM8K full qwen+llama+tulu+dsmath+gemma+phi+ministral 0.5 0.9034 0.0019
gsm8k GSM8K full qwen+llama+tulu+dsmath+gemma+phi+ministral 1.0 0.9034 0.0017
gsm8k GSM8K full qwen+llama+tulu+dsmath+gemma+phi+ministral 2.0 0.9029 0.0016
gsm8k GSM8K best_abs qwen+llama+dsmath 0.0 0.9203 0.0022
gsm8k GSM8K best_abs qwen+llama+dsmath 0.16666666666666666 0.9292 0.0021
gsm8k GSM8K best_abs qwen+llama+dsmath 0.25 0.9297 0.0023
gsm8k GSM8K best_abs qwen+llama+dsmath 0.5 0.9308 0.0016
gsm8k GSM8K best_abs qwen+llama+dsmath 1.0 0.9308 0.0010
gsm8k GSM8K best_abs qwen+llama+dsmath 2.0 0.9300 0.0017
gpqa GPQA full qwen+llama+tulu+dsmath+gemma+phi+ministral 0.0 0.3172 0.0084
gpqa GPQA full qwen+llama+tulu+dsmath+gemma+phi+ministral 0.16666666666666666 0.3737 0.0098
gpqa GPQA full qwen+llama+tulu+dsmath+gemma+phi+ministral 0.25 0.3808 0.0072
gpqa GPQA full qwen+llama+tulu+dsmath+gemma+phi+ministral 0.5 0.3864 0.0126
gpqa GPQA full qwen+llama+tulu+dsmath+gemma+phi+ministral 1.0 0.3864 0.0076
gpqa GPQA full qwen+llama+tulu+dsmath+gemma+phi+ministral 2.0 0.3894 0.0097
Table 28: K=50 distribution-level phase diagram. K=50 uses all cached samples once and serves as a
robustness check for the main protocol.
Benchmark Best acc. Agg.α κMajority SC Single-best
MATH-500 0.8000 Sum 0.6316 7.0171 0.7200 0.7240 0.7980
GSM8K 0.9189 Max 1.0000 8.0000 0.9045 0.9052 0.9105
GPQA 0.4394 Max 0.6842 4.0481 0.3737 0.3030 0.4091
MMLU-Pro 0.6480 Sum 0.8947 0.0000 0.6280 0.5580 0.6180
21

PAGE 22
Table 29: Effective sample-budget scaling under K20R-style subsampling.
benchmark benchmark_label pool_role pool_id K_eff R method accuracy_mean accuracy_std accuracy_min accuracy_max
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 1.0000 50 majority 0.6561 0.0105 0.6340 0.6780
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 1.0000 50 slm_mux 0.6561 0.0105 0.6340 0.6780
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 1.0000 50 cc_like 0.6561 0.0105 0.6340 0.6780
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 1.0000 50 single_best 0.7312 0.0111 0.7020 0.7540
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 1.0000 50 best_gen_step3_cell 0.7315 0.0111 0.7020 0.7540
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 1.0000 50 best_sum_step3_cell 0.7315 0.0111 0.7020 0.7540
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 1.0000 50 best_max_step3_cell 0.7315 0.0111 0.7020 0.7540
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 2.0000 50 majority 0.6915 0.0103 0.6700 0.7180
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 2.0000 50 slm_mux 0.6886 0.0102 0.6600 0.7100
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 2.0000 50 cc_like 0.6886 0.0102 0.6600 0.7100
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 2.0000 50 single_best 0.7354 0.0114 0.7160 0.7660
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 2.0000 50 best_gen_step3_cell 0.7355 0.0113 0.7160 0.7660
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 2.0000 50 best_sum_step3_cell 0.7355 0.0113 0.7160 0.7660
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 2.0000 50 best_max_step3_cell 0.7487 0.0091 0.7280 0.7700
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 3.0000 50 majority 0.7026 0.0082 0.6860 0.7260
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 3.0000 50 slm_mux 0.6984 0.0097 0.6680 0.7200
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 3.0000 50 cc_like 0.6985 0.0099 0.6680 0.7200
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 3.0000 50 single_best 0.7580 0.0083 0.7420 0.7760
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 3.0000 50 best_gen_step3_cell 0.7584 0.0085 0.7440 0.7760
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 3.0000 50 best_sum_step3_cell 0.7584 0.0085 0.7440 0.7760
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 3.0000 50 best_max_step3_cell 0.7616 0.0085 0.7440 0.7800
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 5.0000 50 majority 0.7147 0.0070 0.6980 0.7320
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 5.0000 50 slm_mux 0.7098 0.0094 0.6940 0.7300
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 5.0000 50 cc_like 0.7140 0.0080 0.6980 0.7300
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 5.0000 50 single_best 0.7715 0.0068 0.7580 0.7840
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 5.0000 50 best_gen_step3_cell 0.7718 0.0066 0.7580 0.7880
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 5.0000 50 best_sum_step3_cell 0.7718 0.0066 0.7580 0.7880
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 5.0000 50 best_max_step3_cell 0.7741 0.0069 0.7600 0.7920
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 10 50 majority 0.7204 0.0068 0.7000 0.7340
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 10 50 slm_mux 0.7213 0.0086 0.7040 0.7380
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 10 50 cc_like 0.7241 0.0083 0.7080 0.7440
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 10 50 single_best 0.7826 0.0067 0.7660 0.7940
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 10 50 best_gen_step3_cell 0.7837 0.0065 0.7680 0.7940
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 10 50 best_sum_step3_cell 0.7837 0.0065 0.7680 0.7940
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 10 50 best_max_step3_cell 0.7854 0.0071 0.7640 0.7980
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 20 20 majority 0.7254 0.0062 0.7140 0.7340
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 20 20 slm_mux 0.7271 0.0094 0.7100 0.7520
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 20 20 cc_like 0.7302 0.0063 0.7200 0.7420
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 20 20 single_best 0.7920 0.0041 0.7840 0.8000
math500 MATH-500 full qwen+llama+tulu+dsmath+gemma+phi+ministral 20 20 best_gen_step3_cell 0.7932 0.0046 0.7840 0.8020
Table 30: Method-correlation analysis. High correlation indicates that two methods draw on overlap-
ping evidence.
benchmark method_a method_b prediction_agreement correctness_corr
gpqa single_best_full majority_full 0.5955 0.4504
gpqa single_best_full slm_mux_full 0.5247 0.3339
gpqa single_best_full cc_like_full 0.5732 0.3978
gpqa single_best_full best_gen_full 0.7813 0.7442
gpqa single_best_full oracle_full 0.4015 0.4282
gpqa majority_full slm_mux_full 0.7869 0.7271
gpqa majority_full cc_like_full 0.8929 0.8685
gpqa majority_full best_gen_full 0.7742 0.6843
gpqa majority_full oracle_full 0.3864 0.4148
gpqa slm_mux_full cc_like_full 0.8914 0.8561
gpqa slm_mux_full best_gen_full 0.6354 0.4902
gpqa slm_mux_full oracle_full 0.3369 0.3726
gpqa cc_like_full best_gen_full 0.7101 0.5915
gpqa cc_like_full oracle_full 0.3727 0.4030
gpqa best_gen_full oracle_full 0.4328 0.4567
gsm8k single_best_full majority_full 0.9403 0.7302
gsm8k single_best_full slm_mux_full 0.9330 0.6933
gsm8k single_best_full cc_like_full 0.9382 0.7159
gsm8k single_best_full best_gen_full 0.9632 0.8306
gsm8k single_best_full oracle_full 0.9089 0.4956
gsm8k majority_full slm_mux_full 0.9788 0.9276
gsm8k majority_full cc_like_full 0.9890 0.9616
gsm8k majority_full best_gen_full 0.9591 0.8091
gsm8k majority_full oracle_full 0.9032 0.4793
gsm8k slm_mux_full cc_like_full 0.9896 0.9647
gsm8k slm_mux_full best_gen_full 0.9547 0.7818
gsm8k slm_mux_full oracle_full 0.9031 0.4791
gsm8k cc_like_full best_gen_full 0.9597 0.8057
gsm8k cc_like_full oracle_full 0.9050 0.4844
gsm8k best_gen_full oracle_full 0.9166 0.5202
math500 single_best_full majority_full 0.8084 0.7331
math500 single_best_full slm_mux_full 0.8146 0.7798
math500 single_best_full cc_like_full 0.8228 0.7787
math500 single_best_full best_gen_full 0.9974 0.9933
math500 single_best_full oracle_full 0.7924 0.8091
math500 majority_full slm_mux_full 0.8698 0.8347
math500 majority_full cc_like_full 0.9134 0.9012
math500 majority_full best_gen_full 0.8110 0.7393
math500 majority_full oracle_full 0.7234 0.6642
math500 slm_mux_full cc_like_full 0.9548 0.9316
22

PAGE 23
B.6 Atr/Ctr Model-Support Analysis
This section asks which models the router listens to and whether that support is helpful. Leave-one-out
tables estimate the marginal effect of removing each model. The Llama table separates pools that
include Llama from comparable Llama-free pools. The Atr/Ctr table reports the counterfactual Ctr
used in the main text. The accepted-support decomposition separates support on correct routed
answers from support on wrong routed answers.
Table 31: Leave-one-out model contribution summary.
removed_model GPQA GSM8K MATH-500 MMLU-Pro
dsmath -0.0015 -0.0004 0.0002 0.0002
gemma 0.0076 -0.0003 -0.0002 -0.0100
llama -0.0035 0.0036 0.0008 0.0054
ministral -0.0035 -0.0008 0.0000 -0.0006
phi -0.0066 -0.0002 -0.0006 0.0110
qwen 0.0485 0.0099 0.1118 0.0176
tulu -0.0056 -0.0009 0.0000 -0.0026
Table 32: Llama inclusion analysis comparing the best pool containing Llama with the best pool
excluding Llama.
Benchmark Comparison Pool Size Best gen.∆vs SLM
GPQA contains_llama qwen+llama+gemma+ministral 4 0.4429 0.0212
GPQA no_llama qwen+gemma+phi 3 0.4429 0.0152
GSM8K contains_llama qwen+llama+dsmath 3 0.9327 0.0060
GSM8K no_llama qwen+dsmath 2 0.9231 0.0016
MATH-500 contains_llama qwen+llama+dsmath+gemma 4 0.7952 0.0558
MATH-500 no_llama qwen+dsmath+gemma 3 0.7944 0.0546
MMLU-Pro contains_llama qwen+llama+dsmath+phi 4 0.6526 0.0652
MMLU-Pro no_llama qwen+dsmath+gemma+phi 4 0.6384 0.0594
23

PAGE 24
Table 33: Per-model Atr/Ctr diagnostics for the full pool and the selected curated pool in each
benchmark. Atr is support assigned to the routed answer; Ctr is the leave-one-out counterfactual
contribution defined in §2.
Benchmark Setting Model Atr Ctr
MATH-500 Full qwen 0.8283 0.1220
MATH-500 Full llama 0.5022 0.0014
MATH-500 Full tulu 0.4817 0.0002
MATH-500 Full dsmath 0.4392 0.0002
MATH-500 Full gemma 0.4951 0.0002
MATH-500 Full phi 0.4637 -0.0004
MATH-500 Full ministral 0.5759 0.0012
MATH-500 Curated qwen 0.8279 0.1496
MATH-500 Curated llama 0.5047 0.0016
MATH-500 Curated dsmath 0.4404 0.0004
MATH-500 Curated gemma 0.4974 0.0012
GSM8K Full qwen 0.9320 0.0121
GSM8K Full llama 0.8340 0.0037
GSM8K Full tulu 0.8722 -0.0009
GSM8K Full dsmath 0.7876 -0.0004
GSM8K Full gemma 0.8577 0.0003
GSM8K Full phi 0.8494 0.0010
GSM8K Full ministral 0.8542 -0.0001
GSM8K Curated qwen 0.9179 0.0228
GSM8K Curated llama 0.8482 0.0177
GSM8K Curated dsmath 0.8121 0.0135
GPQA Full qwen 0.6139 0.0571
GPQA Full llama 0.4628 0.0020
GPQA Full tulu 0.4477 0.0010
GPQA Full dsmath 0.2933 -0.0015
GPQA Full gemma 0.5169 0.0222
GPQA Full phi 0.4828 -0.0066
GPQA Full ministral 0.4574 0.0025
GPQA Curated qwen 0.6093 0.0641
GPQA Curated llama 0.4682 0.0081
GPQA Curated gemma 0.5304 0.0303
GPQA Curated ministral 0.4578 0.0101
MMLU-Pro Full qwen 0.7273 0.0204
MMLU-Pro Full llama 0.5898 0.0094
MMLU-Pro Full tulu 0.5038 0.0010
MMLU-Pro Full dsmath 0.3341 0.0004
MMLU-Pro Full gemma 0.6430 -0.0008
MMLU-Pro Full phi 0.5892 0.0146
MMLU-Pro Full ministral 0.5156 0.0024
MMLU-Pro Curated qwen 0.6974 0.0624
MMLU-Pro Curated llama 0.5968 0.0364
MMLU-Pro Curated dsmath 0.3731 0.0022
MMLU-Pro Curated phi 0.6400 0.0382
24

PAGE 25
Table 34: Accepted-support decomposition. For each benchmark and pool role, we show the three
models with highest Atr under best generalized routing. Balance is correct accepted support minus
wrong accepted support.
Benchmark Pool Model Atr Correct part Wrong part Balance Precision
GPQA best_abs qwen 0.6093 0.2862 0.3231 -0.0368 0.4698
GPQA best_abs gemma 0.5304 0.2480 0.2824 -0.0343 0.4676
GPQA best_abs llama 0.4682 0.2105 0.2578 -0.0473 0.4496
GPQA full qwen 0.6139 0.2814 0.3326 -0.0512 0.4583
GPQA full gemma 0.5169 0.2419 0.2750 -0.0332 0.4679
GPQA full phi 0.4828 0.2148 0.2680 -0.0532 0.4449
GSM8K best_abs qwen 0.9179 0.8766 0.0413 0.8354 0.9550
GSM8K best_abs llama 0.8482 0.8149 0.0332 0.7817 0.9608
GSM8K best_abs dsmath 0.8121 0.7829 0.0292 0.7537 0.9640
GSM8K full qwen 0.9320 0.8757 0.0563 0.8194 0.9396
GSM8K full tulu 0.8722 0.8269 0.0453 0.7816 0.9481
GSM8K full gemma 0.8577 0.8069 0.0507 0.7562 0.9408
MATH-500 full qwen 0.8283 0.7189 0.1094 0.6094 0.8679
MATH-500 full ministral 0.5759 0.5335 0.0425 0.4910 0.9262
MATH-500 full llama 0.5022 0.4697 0.0325 0.4372 0.9353
MMLU-Pro full qwen 0.7273 0.5271 0.2002 0.3269 0.7247
MMLU-Pro full gemma 0.6430 0.4635 0.1795 0.2840 0.7209
MMLU-Pro full llama 0.5898 0.4385 0.1513 0.2871 0.7435
25

PAGE 26
C Supplementary Mechanism and Sensitivity Tables
C.1 Six-Model Exploratory Analyses
These tables summarize a smaller six-model setting used to check the same routing questions under a
lighter cache. They report single-model and baseline numbers, a fixed-prior generalized sweep, top
pools from a pool-composition scan, and sampling-strategy comparisons.
Table 35: Six-model exploratory baseline summary.
Benchmark Best single Best single acc. Majority SLM-MUX Cross-support Single-Best
MATH-500 Qwen 0.7660 0.7220 0.7200 0.7080 0.7680
GSM8K Qwen 0.9022 0.8999 0.9158 0.9022 0.9037
GPQA Gemma 0.4091 0.3636 0.3434 0.3535 0.3788
Table 36: Six-model fixed-prior sweep summary.
Benchmark Majority SC Cross-support Single-best Best gen. BestαBestκAgg.
MATH-500 0.7060 0.7000 0.7040 0.7680 0.7760 0.3000 4.0000 Max
GSM8K 0.8999 0.9007 0.9014 0.9037 0.9105 1.0000 0.5000 Max
GPQA 0.3737 0.3535 0.3687 0.3990 0.4343 0.6000 4.0000 Sum
Table 37: Six-model top pool-composition results by best generalized accuracy.
Benchmark Rank Pool Size Best gen. Single-best∆vs SB∆vs SLM Oracle gap Regime
MATH-500 1 qwen+phi 2 0.7780 0.7680 0.0100 0.0280 0.0140 low/moderate-alpha Sum coalition/Majority-like
MATH-500 2 qwen+tulu+dsmath+phi+ministral 5 0.7780 0.7680 0.0100 0.0660 0.0560 high-kappa expert/single-best-like
MATH-500 3 qwen+tulu+ministral 3 0.7760 0.7680 0.0080 0.0340 0.0340 high-kappa expert/single-best-like
GSM8K 1 qwen+tulu+dsmath 3 0.9219 0.9037 0.0182 0.0045 0.0569 low/moderate-alpha Sum coalition/Majority-like
GSM8K 2 qwen+dsmath+phi 3 0.9189 0.9037 0.0152 0.0099 0.0584 high-kappa expert/single-best-like
GSM8K 3 qwen+dsmath 2 0.9174 0.9037 0.0136 0.0038 0.0478 low/moderate-alpha Sum coalition/Majority-like
GPQA 1 qwen+gemma+phi 3 0.4495 0.3990 0.0505 0.0152 0.2525 low/moderate-alpha Sum coalition/Majority-like
GPQA 2 qwen+dsmath+gemma+phi 4 0.4495 0.3990 0.0505 0.0758 0.2828 low/moderate-alpha Sum coalition/Majority-like
GPQA 3 qwen+gemma 2 0.4394 0.3990 0.0404 0.0202 0.1667 low/moderate-alpha Sum coalition/Majority-like
Table 38: Sampling-strategy comparison. K20R10 preserves replicate variability while using the
larger cache.
Benchmark Pool Strategy Majority SC Cross-support Single-best Best gen. Std Modal best params
GPQA six-model K20 once 0.3932 0.3422 0.3730 0.4020 0.4444 0.0087 Sum a=0.9 k=2 (2/20)
GPQA six-model K20R10 mean 0.3821 0.3172 0.3750 0.4078 0.4467 0.0081 Max a=0.9 k=2 (6/20)
GPQA six-model K=50 full 0.3838 0.3081 0.3788 0.4091 0.4495 0.0000 Max a=0.9 k=2 (1/1)
GPQA seven-model K20 once 0.3929 0.3409 0.3727 0.4020 0.4394 0.0118 Sum a=0.0 k=8 (3/20)
GPQA seven-model K20R10 mean 0.3798 0.3096 0.3674 0.4078 0.4379 0.0058 Max a=0.9 k=2 (4/20)
GPQA seven-model K=50 full 0.3737 0.3030 0.3535 0.4091 0.4343 0.0000 Sum a=0.2 k=8 (1/1)
GSM8K six-model K20 once 0.8989 0.8983 0.9010 0.9080 0.9135 0.0018 Max a=1.0 k=4 (13/20)
GSM8K six-model K20R10 mean 0.8984 0.8998 0.9016 0.9085 0.9143 0.0011 Max a=1.0 k=4 (12/20)
GSM8K six-model K=50 full 0.8969 0.8999 0.9014 0.9105 0.9143 0.0000 Max a=1.0 k=4 (1/1)
GSM8K seven-model K20 once 0.9040 0.9023 0.9047 0.9080 0.9168 0.0015 Max a=1.0 k=4 (6/20)
GSM8K seven-model K20R10 mean 0.9040 0.9046 0.9067 0.9085 0.9181 0.0011 Max a=1.0 k=8 (11/20)
GSM8K seven-model K=50 full 0.9045 0.9052 0.9067 0.9105 0.9189 0.0000 Max a=1.0 k=8 (1/1)
MATH-500 six-model K20 once 0.7175 0.7286 0.7307 0.7894 0.7926 0.0054 Sum a=0.0 k=8 (12/20)
MATH-500 six-model K20R10 mean 0.7216 0.7251 0.7317 0.7962 0.7971 0.0040 Sum a=0.2 k=8 (7/20)
MATH-500 six-model K=50 full 0.7220 0.7220 0.7320 0.7980 0.7980 0.0000 Sum a=0.3 k=8 (1/1)
MATH-500 seven-model K20 once 0.7222 0.7292 0.7323 0.7894 0.7941 0.0051 Sum a=0.0 k=8 (8/20)
MATH-500 seven-model K20R10 mean 0.7227 0.7270 0.7327 0.7962 0.7979 0.0034 Sum a=0.1 k=8 (6/20)
MATH-500 seven-model K=50 full 0.7200 0.7240 0.7300 0.7980 0.7980 0.0000 Sum a=0.0 k=8 (1/1)
MMLU-Pro seven-model K20 once 0.6285 0.5717 0.6108 0.6169 0.6439 0.0053 Sum a=0.0 k=8 (4/20)
MMLU-Pro seven-model K20R10 mean 0.6269 0.5609 0.6127 0.6201 0.6444 0.0022 Sum a=0.5 k=0.5 (4/20)
MMLU-Pro seven-model K=50 full 0.6280 0.5580 0.6120 0.6180 0.6480 0.0000 Sum a=0.9 k=0 (1/1)
26

PAGE 27
C.2 Seven-Model Protocol Checks
These tables record seven-model protocol checks: comparable baselines, replicate-aware generalized
sweeps, and pool-subset scans under the K20R10 protocol.
Table 39: Seven-model K20R10 baseline summary. Standard deviations are across the 10 subsampling
replicates.
Benchmark Best single Best single Majority SLM-MUX Cross slice Oracle
MATH-500 qwen 0.7924±0.0048 0.7246±0.0073 0.7242±0.0053 0.7302±0.0040 0.8536±0.0067
GSM8K qwen 0.9089±0.0024 0.9032±0.0018 0.9031±0.0018 0.9050±0.0015 0.9760±0.0011
GPQA qwen 0.4015±0.0162 0.3864±0.0069 0.3369±0.0155 0.3727±0.0087 0.7854±0.0116
MMLU-Pro qwen 0.6158±0.0058 0.6282±0.0055 0.5738±0.0085 0.6130±0.0045 0.8420±0.0042
Table 40: Seven-model K20R10 fixed-prior sweep summary.
Benchmark Majority SC Cross-support Single-best Best gen. BestαBestκAgg.
MATH-500 0.7246 0.7242 0.7302 0.7924 0.7938 0.6842 7.0171 Sum
GSM8K 0.9032 0.9031 0.9050 0.9089 0.9166 1.0000 7.0171 Max
GPQA 0.3864 0.3369 0.3727 0.4015 0.4328 0.3684 8.0000 Max
MMLU-Pro 0.6282 0.5738 0.6130 0.6158 0.6402 0.1053 5.3617 Max
Table 41: Seven-model top pool-composition results, with five best pools per benchmark by general-
ized accuracy.
Benchmark Pool Size Best gen. Single-best SC Oracle Agg.
MATH-500 qwen+llama+tulu+gemma 4 0.7952 0.7924 0.7452 0.8366 Max
MATH-500 qwen+llama+dsmath+gemma 4 0.7952 0.7924 0.7394 0.8380 Max
MATH-500 qwen+llama+gemma 3 0.7950 0.7924 0.7504 0.8294 Sum
MATH-500 qwen+llama+tulu+dsmath+gemma 5 0.7948 0.7924 0.7360 0.8434 Sum
MATH-500 qwen+llama 2 0.7944 0.7924 0.7860 0.8174 Sum
GSM8K qwen+llama+dsmath 3 0.9327 0.9089 0.9267 0.9660 Sum
GSM8K qwen+llama+tulu+dsmath 4 0.9281 0.9089 0.9252 0.9678 Max
GSM8K qwen+llama 2 0.9238 0.9089 0.9213 0.9479 Sum
GSM8K qwen+dsmath 2 0.9231 0.9089 0.9215 0.9569 Sum
GSM8K qwen+tulu+dsmath 3 0.9230 0.9089 0.9227 0.9635 Max
GPQA qwen+gemma+phi 3 0.4429 0.4015 0.4278 0.6348 Max
GPQA qwen+llama+gemma+ministral 4 0.4429 0.4015 0.4217 0.6788 Sum
GPQA qwen+llama+gemma 3 0.4424 0.4015 0.4247 0.6343 Sum
GPQA qwen+llama+gemma+phi 4 0.4409 0.4015 0.4162 0.6904 Max
GPQA qwen+dsmath+gemma+phi 4 0.4409 0.4015 0.3591 0.6970 Max
MMLU-Pro qwen+llama+dsmath+phi 4 0.6526 0.6158 0.5874 0.7846 Sum
MMLU-Pro qwen+llama+phi 3 0.6506 0.6158 0.6192 0.7528 Sum
MMLU-Pro qwen+llama+tulu+dsmath+phi+ministral 6 0.6502 0.6158 0.5660 0.8274 Sum
MMLU-Pro qwen+llama+tulu+dsmath+phi 5 0.6486 0.6158 0.5672 0.8080 Sum
MMLU-Pro qwen+llama+tulu+phi+ministral 5 0.6478 0.6158 0.5804 0.7992 Sum
27

PAGE 28
C.3 Additional Mechanism Tables
The following tables probe mechanism questions that complement the main analysis: whether cross-
support is a question-specific signal, how it behaves under placebo tests, how sample budget affects
routing signals, and where oracle coverage fails to become routable support.
Table 42: Row-level correctness prediction quality for isolated self-consistencyc, cross-support s,
andc+s.
Benchmark Rows AUC(c)AUC(s)AUC(c+s)ρ(c, y)ρ(s, y)
MATH-500 29780 0.820 0.904 0.901 0.574 0.706
GSM8K 79140 0.754 0.945 0.936 0.438 0.581
GPQA 11280 0.525 0.600 0.574 0.043 0.164
GPQA (K=3) 11088 0.531 0.605 0.584 0.055 0.172
Table 43: Placebo tests that corrupt cross-support while leaving self-consistency fixed.
Setting SLM Cross real Cross q-shuf Cross m-shuf∆real
GPQA / 6-model / K=5 0.343 0.354 0.345 0.320 0.010
GPQA / 6-model / K=3 0.332 0.363 0.341 0.317 0.031
GPQA / qwen+phi+gemma / K=5 0.394 0.429 0.404 0.406 0.035
MATH-500 / 6-model / K=5 0.720 0.708 0.704 0.692 -0.012
GSM8K / 6-model / K=5 0.916 0.902 0.908 0.892 -0.014
Table 44: Best score form in each setting from the score-form ablation.
Setting Best variant Acc.∆vs SLM
GPQA / 6-model / K=3 s_only 0.388 0.056
GPQA / 6-model / K=5 s_only 0.379 0.035
GPQA / qwen+phi+gemma / K=5 rank_sum 0.439 0.045
MATH-500 / 6-model / K=5 c_times_s 0.722 0.002
GSM8K / 6-model / K=5 c_only 0.916 0.000
Table 45: Best support definition bys-only accuracy.
Setting Support definitions-only∆vs SLM
GPQA / 6-model / K=3 top2_strong 0.396 0.064
GPQA / 6-model / K=5 top2_strong 0.419 0.076
GPQA / qwen+phi+gemma / K=5 majority_fraction 0.414 0.020
Table 46: Best normalisation / calibration configuration in each setting.
Setting Best configuration Acc.∆vs SLM
GPQA / 6-model / K=5 s_only (rank) 0.384 0.040
GPQA / qwen+phi+gemma / K=5 c+s (zscore,rank) 0.449 0.056
MATH-500 / 6-model / K=5 c+s (minmax,minmax) 0.720 0.000
Selection frequencies:qwen_gemma: 9, qwen_phi_gemma: 6, qwen_phi_ministral_gemma: 4,
phi_gemma: 1.
28

PAGE 29
Table 47: How the routing signals change with the sample budgetK.
BenchmarkKAUC(c)AUC(s)AUC(c+s)Entropy Stable-wrong SLM
GPQA 1 0.500 0.599 0.599 1.218 64.6
GPQA 2 0.522 0.608 0.582 1.401 64.9
GPQA 3 0.531 0.605 0.584 1.460 66.8
GPQA 4 0.535 0.602 0.583 1.485 66.7
GPQA 5 0.525 0.600 0.574 1.506 65.7
MATH-500 1 0.500 0.944 0.944 1.272 27.6
MATH-500 2 0.784 0.954 0.948 1.627 25.9
MATH-500 3 0.837 0.953 0.949 1.803 26.5
MATH-500 4 0.854 0.952 0.949 1.926 24.0
MATH-500 5 0.861 0.951 0.948 2.011 24.2
GSM8K 1 0.500 0.939 0.939 0.417 11.4
GSM8K 2 0.712 0.950 0.944 0.486 9.4
GSM8K 3 0.735 0.947 0.940 0.515 9.2
GSM8K 4 0.748 0.946 0.938 0.535 8.6
GSM8K 5 0.754 0.945 0.936 0.548 8.4
Table 48: Accuracy scaling with the per-model sample budgetKon the full 6-model pool.
BenchmarkKSLM Cross∆Fix Break
MATH-500 1 0.724 0.703 -0.022 116 224
MATH-500 2 0.708 0.694 -0.014 94 166
MATH-500 3 0.711 0.703 -0.009 81 124
MATH-500 4 0.713 0.708 -0.005 76 101
MATH-500 5 0.720 0.708 -0.012 60 120
GSM8K 1 0.886 0.897 0.011 405 254
GSM8K 2 0.902 0.899 -0.003 212 251
GSM8K 3 0.908 0.899 -0.009 149 268
GSM8K 4 0.912 0.900 -0.012 133 295
GSM8K 5 0.916 0.902 -0.014 130 310
GPQA 1 0.354 0.364 0.010 201 181
GPQA 2 0.350 0.366 0.016 172 140
GPQA 3 0.332 0.363 0.031 165 104
GPQA 4 0.334 0.352 0.018 143 107
GPQA 5 0.343 0.354 0.010 110 90
Table 49: Oracle-gap decomposition of routing outcomes.
Setting Routed correct No oracle Stable-but-wrong No support Invalid Other
GPQA / 6-model / K=5 34.3 24.2 31.8 4.5 0.0 5.1
GPQA / qwen+phi+gemma / K=5 39.4 34.8 16.7 5.1 0.0 4.0
MATH-500 / 6-model / K=5 72.0 16.6 9.8 0.6 0.0 1.0
GSM8K / 6-model / K=5 91.6 2.9 2.4 2.2 0.0 1.0
Table 50: Top GPQA model pairs by pairwise oracle. All top pairs contain Gemma.
Pair Pair oracle Inc. gain Error overlap Wrong-answer agr.
qwen + gemma 0.566 0.167 0.434 0.477
tulu + gemma 0.561 0.162 0.439 0.471
gemma + phi 0.556 0.157 0.444 0.409
qwen + tulu 0.545 0.167 0.455 0.544
qwen + ministral 0.535 0.157 0.465 0.533
Table 51: Question-slice analysis on GPQA by model disagreement.
Setting Bin Views SLM Crosss-only∆Cross-SLM
GPQA / 6-model / K=3 q1 426 0.434 0.467 0.460 0.033
GPQA / 6-model / K=3 q2 323 0.359 0.399 0.424 0.040
GPQA / 6-model / K=3 q3 317 0.341 0.404 0.423 0.063
GPQA / 6-model / K=3 q4 914 0.271 0.287 0.329 0.015
GPQA / 6-model / K=5 q1 430 0.442 0.465 0.442 0.023
GPQA / 6-model / K=5 q2 380 0.368 0.474 0.500 0.105
GPQA / 6-model / K=5 q3 230 0.391 0.391 0.391 0.000
GPQA / 6-model / K=5 q4 940 0.277 0.245 0.298 -0.032
29

PAGE 30
Table 52: OLS coefficients for predicting∆ =Cross−SLM from subset-level pool statistics.
Benchmarkβ dom βoracle βsize R2 n
MATH-500 0.015 0.223 -0.005 0.669 57
GSM8K 0.125 0.270 -0.004 0.655 57
GPQA 0.072 0.212 -0.013 0.106 57
Table 53: GPQA contamination analysis anchored at the curated core pool “qwen_phi_gemma“.
PoolnCross∆vs core Oracle∆oracle
qwen_phi_gemma 3 0.439 0.000 0.652 0.000
qwen_tulu_phi_gemma 4 0.384 -0.056 0.712 0.061
qwen_phi_ministral_gemma 4 0.429 -0.010 0.702 0.051
qwen_phi_gemma_dsmath 4 0.374 -0.066 0.682 0.030
qwen_tulu_phi_gemma 4 0.384 -0.056 0.712 0.061
qwen_tulu_phi_ministral_gemma 5 0.364 -0.076 0.747 0.096
qwen_tulu_phi_ministral_gemma_dsmath 6 0.359 -0.081 0.758 0.106
qwen_gemma 2 0.434 -0.005 0.566 -0.086
phi_gemma 2 0.419 -0.020 0.556 -0.096
Table 54: Question-level paired bootstrap confidence intervals and permutation tests.
ComparisonN∆95% CIp
GPQA / 6-model / K=5 : cross-support vs slm_mux 198 0.010 [-0.035, 0.056] 0.822
GPQA / 6-model / K=3 : cross-support vs slm_mux 198 0.031 [-0.008, 0.068] 0.115
GPQA / qwen+phi+gemma / K=5: cross-support vs slm_mux 198 0.005 [-0.030, 0.045] 1.000
GPQA / qwen+phi+gemma / K=5: cross-support vs single_best_sc 198 0.040 [-0.020, 0.101] 0.249
MATH-500 / 6-model / K=5: cross-support vs slm_mux 500 -0.012 [-0.028, 0.004] 0.238
GSM8K / 6-model / K=5: cross-support vs slm_mux 1319 -0.014 [-0.024, -0.004] 0.011
Table 55: Tie-break robustness under model-order and validation-accuracy protocols.
Setting Tie-break Acc source SLM Cross∆
GPQA / 6-model / K=5 model_order ignore 0.343 0.354 0.010
GPQA / 6-model / K=5 validation_accuracy auto 0.354 0.359 0.005
GPQA / qwen+phi+gemma / K=5 model_order ignore 0.394 0.429 0.035
GPQA / qwen+phi+gemma / K=5 validation_accuracy auto 0.434 0.439 0.005
MATH-500 / 6-model / K=5 model_order ignore 0.720 0.708 -0.012
MATH-500 / 6-model / K=5 validation_accuracy auto 0.722 0.710 -0.012
Table 56: Held-out GPQA pool selection over 20 random 50/50 splits.
Method Held-out mean
Selected small-pool cross-support 0.415
Full-pool cross-support 0.364
Full-pool SLM 0.345
Full-pool Single-Best 0.388
Table 57: Seed robustness over 20 offline subsampling seeds.
Setting SLM Cross∆Wins / ties / losses
GPQA / 6-model / K=3 0.340±0.006 0.360±0.006 0.020±0.008 19/0/1
GPQA / 6-model / K=5 0.343±0.000 0.354±0.000 0.010±0.000 20/0/0
GPQA / qwen+phi+gemma / K=5 0.394±0.000 0.429±0.000 0.035±0.000 20/0/0
MATH-500 / 6-model / K=5 0.720±0.000 0.708±0.000 -0.012±0.000 0/0/20
Table 58: Budget-fair comparison on GPQA. Curated pools with deeper sampling dominate the full
6-model reference.
Configuration Budget SLM Cross Single-Best Oracle
6 models×K=1 (budget=6) 6 0.354 0.364 0.353 0.766
4 models×K=2 (budget=8) qwen+phi+ministral+gemma 8 0.350 0.372 0.360 0.684
3 models×K=3 (budget=9) qwen+phi+gemma 9 0.376 0.400 0.371 0.641
2 models×K=5 (budget=10) qwen+gemma 10 0.409 0.414 0.379 0.566
6 models×K=5 (budget=30) full pool reference 30 0.343 0.354 0.379 0.758
30

PAGE 31
Table 59: With- vs without-validation comparison. Validation-selected single-best is the strongest
deployable baseline.
Benchmark Order SB Val SB SLM Crosss-only Composite
MATH-500 0.768 0.768 0.720 0.708 0.712 0.768
GSM8K 0.904 0.904 0.916 0.902 0.899 0.902
GPQA 0.379 0.399 0.343 0.354 0.379 0.354
Table 60: Held-out GPQA pool selection across four selection rules.
Rule Mean test cross-support Std Mean tests-only Most selected pool
max_calib_cross 0.417 0.022 0.392 qwen_phi_gemma
max_calib_s_only 0.408 0.028 0.381 qwen_gemma
min_pool_size 0.420 0.022 0.393 qwen_gemma
smallest_top_cross 0.414 0.023 0.394 qwen_gemma
Table 61: Case taxonomy for the GPQA qualitative analysis.
Category Rows Fraction
Both wrong, oracle hit 710 61.2
Cross-support fixes SLM 110 9.5
Cross-support breaks SLM 90 7.8
Contamination 250 21.6
31

PAGE 32
C.4 Additional Generalized-Family Checks
These generalized-family tables provide implementation checks, canonical-point recovery, K interac-
tions, validation-aware variants, region summaries, significance checks, and budget-fair comparisons.
Table 62: Implementation-level recovery of the canonical baselines as limits inside the generalized
family. All limits are recovered exactly modulo the deterministic (W,pooled,lex) tie-break specified
for the family; the small MATH-500 majority gap is from a handful of questions on which one model
has fewer valid extracted answers than the others.
Benchmark /Kmajorityδtop-coalδSLM-MUX ans δcross sliceδsingle-bestδ
GPQA /K=5/ 6m+0.000–+0.010–+0.000
MATH-500 /K=5/ 6m−0.004–−0.020–+0.000
GSM8K /K=5/ 6m+0.000–−0.015–+0.000
Table 63: Generalized two-axis family on the full 6-model and curated GPQA pools. All canonical
points are evaluated on the same cached generations as the rest of the paper. α controls majority-
vs-top-answer aggregation; κ controls uniform-vs-expert weighting (calibration-half acci, held-out
evaluation for κ>0); Agg selects coalition (Sum) vs strongest-supporter (Max) aggregation. Best
generalized point in bold per row.
Setting majority top-coal SLM-MUX ans cross slice best Sum best Max best(α, κ,Agg)best acc
(α=0) (α=1, Sum) (α=1, Max) (α=1− 1N−1 ) (κ=0) (κ=0)
GPQA / 6m /K=30.384 0.372 0.364 0.365 0.384 0.384 (0.0,0,Sum)0.384
GPQA / 6m /K=50.374 0.369 0.354 0.369 0.379 0.374 (1.0,4,Max)0.394
GPQA / qwen+phi+gemma /K=30.410 0.394 0.405 0.404 0.410 0.410 (0.0,0,Sum)0.410
GPQA / qwen+phi+gemma /K=50.419 0.424 0.434 0.429 0.450 0.439 (0.6,0,Sum)0.450
GPQA / qwen+gemma /K=50.429 0.419 0.419 0.429 0.429 0.424 (0.0,0,Sum)0.429
GPQA / phi+gemma /K=50.404 0.419 0.419 0.404 0.404 0.419 (0.1,2,Sum)0.444
MATH-500 / 6m /K=50.706 0.702 0.700 0.704 0.708 0.712 (0.1,4,Sum)0.800
GSM8K / 6m /K=50.900 0.897 0.901 0.901 0.900 0.901 (1.0,0.5,Max)0.909
Table 64: K interaction on GPQA for the generalized family. Best validation-free κ=0 generalized
point reported per K, alongside the canonical points. “–” marks aggregations not swept at K=2 and
K=4. The full pool peaks atK=3in theκ=0region; the curated pool is monotonic inK.
PoolKmajority SLM-MUX ans cross slice single-best (val) best gen(α, κ,Agg)best acc
full 6m 1 0.370 0.370 0.370 0.338 (0.0,0,Sum) 0.370
full 6m 2 0.372 0.372 0.372 0.300 – –
full 6m 3 0.384 0.364 0.365 0.342 (0.0,0,Sum)0.384
full 6m 4 0.365 0.349 0.358 0.369 – –
full 6m 5 0.374 0.354 0.369 0.364 (1.0,4,Max) 0.394
qwen+phi+gemma 1 0.375 0.375 0.375 0.334 (0.0,0,Sum) 0.375
qwen+phi+gemma 2 0.389 0.385 0.385 0.300 – –
qwen+phi+gemma 3 0.410 0.405 0.404 0.337 (0.0,0,Sum) 0.410
qwen+phi+gemma 4 0.415 0.410 0.411 0.369 – –
qwen+phi+gemma 5 0.419 0.434 0.429 0.364 (0.6,0,Sum)0.450
Table 65: Validation-aware evaluation of the generalized family. Per setting we draw a fresh 50/50
random split over questions for each of 20 seeds, fit per-model calibration accuraciesacci on one half,
and evaluate every (α, κ,Agg)cell on the held-out half. All numbers are held-out accuracy averaged
across the 20 seeds. “Single-best (val)” is theκ=∞limit.
Setting majority SLM-MUX ans cross slice single-best (val) best Sum best Max
GPQA / 6m /K=30.384 0.362 0.364 0.3600.391@(0.0,1.0) 0.391@(0.0,1.0)
GPQA / 6m /K=50.380 0.354 0.370 0.3740.403@(0.5,4.0) 0.402@(0.3,4.0)
GPQA / qwen+phi+gemma /K=50.422 0.441 0.434 0.3680.455@(0.6,0.0) 0.446@(0.7,0.5)
MATH-500 / 6m /K=50.699 0.691 0.6950.7610.765@(0.8,4.0) 0.766@(0.3,4.0)
GSM8K / 6m /K=50.897 0.898 0.899 0.903 0.902@(0.0,8.0)0.907@(1.0,0.5)
32

PAGE 33
Table 66: Region summary. For each (benchmark, K, pool) we report the best validation-free κ=0
generalized point on each of the Sum and Max planes, alongside the canonical points and external
references (s-only, rank-sum, trust-aware top2_strong, oracle). This compressed view summarizes
the best-region map per regime.
SettingNmaj top-coal SLM-MUX ans cross slice best Sumα ⋆ best Sum best Maxα ⋆ best Max single-bests-only rank-sum top2_strong
GPQA / 6m /K=36 0.384 0.372 0.364 0.365 0.00.3840.0 0.384 0.342 0.386 0.381 0.367
GPQA / 6m /K=56 0.374 0.369 0.354 0.369 0.70.3790.0 0.374 0.364 0.384 0.364 0.379
GPQA / qwen+phi+gemma /K=33 0.410 0.394 0.405 0.404 0.00.4100.0 0.410 0.337 0.391 0.403 0.392
GPQA / qwen+phi+gemma /K=53 0.419 0.424 0.434 0.429 0.60.4500.6 0.439 0.364 0.394 0.439 0.404
MATH-500 / 6m /K=56 0.706 0.702 0.700 0.704 0.3 0.708 0.5 0.7120.7880.710 0.712 0.716
GSM8K / 6m /K=56 0.900 0.897 0.901 0.901 0.0 0.900 0.4 0.9010.9140.899 0.901 0.904
Table 67: What the generalized family tells us about the four routing research questions on this
benchmark set. “balanced” = curated GPQA pools (qwen+phi+gemma, qwen+gemma, phi+gemma);
“dominated” = full 6-model pool on MATH-500 / GSM8K (Qwen≥10pts above the rest).
Question Answer from the generalized sweep
RQ1: majority vs
top-answer (α)
Balanced GPQA prefers low to moderate α (full-pool K=3: α=0; curated K=5:
α=0.6). Dominated pools accept high α only when paired with high κ, i.e. single-
best in disguise.
RQ2: uniform vs ex-
pert (κ)
Under leakage-free held-out evaluation, κ>0 adds +0.02 to +0.09 on GPQA
above every canonical point including κ=∞ single-best. On MATH-500 and
GSM8K the best κ>0 cell lies within 0.005 of single-best – the family recovers
single-best routing rather than a new mode.
RQ3: Sum vs Max Sum wins on 5 of 6 GPQA settings (the curated K=5 peak is at Sum= 0.450 vs
Max= 0.439). Max is preferred at K=5 on the full GPQA pool and on GSM8K,
where one model has a consistently strongc i.
RQ4: regime The best routing region shifts with pool curation: low-α, κ near zero, Sum aggrega-
tion on balanced pools; high-α, κ≥4, Max aggregation near the single-best limit
on dominated pools.K=3remains the GPQA sweet spot whenκis held to zero.
Table 68: Question-level paired tests for the strongest validation-free generalized point (κ=0) against
each of the SLM-MUXans, s-only, and rank-sum baselines. ∆ is the per-question mean correctness
difference (positive = generalized wins); p is the two-sided paired-permutation p-value with B=5000
replicates; 95% CIs from paired bootstrap. Statistically significant gains in bold (p <0.05).
Setting best(α, κ,Agg)gen acc∆vs SLM-MUX ans ∆vss-only∆vs rank-sum
GPQA / 6m /K=3(0.0,0,Sum) 0.384+0.053 [+0.006, +0.099]p=0.024−0.002p=0.73 +0.004p=0.51
GPQA / 6m /K=5(0.7,0,Sum) 0.379+0.035[−0.030,+0.101]p=0.37−0.005p=1.0 +0.015p=0.59
GPQA / qwen+phi+gemma /K=5(0.6,0,Sum) 0.450+0.056 [+0.015, +0.096]p=0.018+0.056 [+0.020, +0.096]p=0.007 +0.010p=0.69
MATH-500 / 6m /K=5(0.5,0,Max) 0.712−0.008p=0.57 +0.002p=1.0 0.000p=1.0
GSM8K / 6m /K=5(0.4,0,Max) 0.901−0.014[−0.024,−0.005]p=0.007+0.002p=0.45 +0.001p=1.0
Table 69: Budget-fair generalized comparison on GPQA. Each row is a fixed model ×K budget.
“best gen” is the strongest κ=0 generalized point at that budget. Curated pools with deeper sampling
cleanly dominate the full6×5reference at3×less compute.
Configuration budget majority SLM-MUX ans cross slice best(α,Agg)best gen
6 models×K=16 0.355 0.355 0.355 (0,Sum) 0.355
4 models×K=2(qwen+phi+ministral+gemma) 8 0.386 0.378 0.378 (0,Sum) 0.386
3 models×K=3(qwen+phi+gemma) 9 0.410 0.405 0.404 (0,Sum) 0.410
2 models×K=5(qwen+gemma) 100.4290.419 0.429 (0,Sum)0.429
6 models×K=5(full pool reference) 30 0.369 0.348 0.364 (0.7,Sum) 0.379
33
```
