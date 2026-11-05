<!-- Source: https://raw.githubusercontent.com/mlresearch/v267/main/assets/zhang25bi/zhang25bi.pdf; retrieved: 2026-09-28T19:18:08.559575+00:00; conversion: pypdf layout text extraction. Equations, tables, and reading order require checking against source.pdf. -->

## PDF page 1

Multi-agent Architecture Search via Agentic Supernet




                Guibin Zhang*1  Luyang Niu*2  Junfeng Fang1  Kun Wang3  Lei Bai4  Xiang Wang5

                            Abstract                                       et al.,2024;Li et al.,2024), code generation (Shinn et al.,
      LargeLanguageModel(LLM)-empoweredmulti-                              2023), web navigation (Deng et al.,2024), and data syn-
      agent systems extend the cognitive boundaries                        thesis (Butt et al.,2024), by equipping LLMs with high-
      of individual agents through disciplined collabo-                    levelfeatures, including persona(Wanget al.,2023b;Chen
      ration and interaction, while constructing these                     et al.,2024), tools (Shen et al.,2024;Richards & et al.,
      systems often requires labor-intensive manual de-                    2023),planning(Qiaoetal.,2024;Wuetal.,2024;Heetal.,
      signs.   Despite the availability of methods to                      2023),andmemory(Zhongetal.,2024;Hatalisetal.,2023;
      automate the design of agentic workflows, they                       Packer et al.,2023). Building upon the success of single
      typically seek to identify a static, complex, one-                   agents, researchers have demonstrated that combining mul-
      size-fits-all system, which, however, fails to dy-                   tiple agents, either cooperatively (Zhuge et al.,2024) or
      namically allocate inference resources based on                      competitively (Zhao et al.,2023), can surpass the cognitive
      the difficulty and domain of each query. To ad-                      andintellectualcapabilitiesofindividuals(Duetal.,2023;
      dress this challenge, we shift away from the pur-                    Liangetal.,2023;Wangetal.,2023b;Jiangetal.,2023;Wu
      suit of a monolithic agentic system, instead op-                     etal.,2023;Zhangetal.,2024a),showcasingthecollective
      timizing the agentic supernet, a probabilistic                       intelligenceinasocietyofLLM-agents(Piattietal.,2024).
      and continuous distribution of agentic architec-                     Earlymulti-agentsystems,suchasCAMEL(Lietal.,2023),
      tures. WeintroduceMaAS,anautomatedframe-                             AutoGen (Wu et al.,2023), and MetaGPT (Hong et al.,
     work that samples query-dependent agentic sys-                        2023), while delivering specialized capacity, often heav-
      tems from the supernet, delivering high-quality                      ily rely on manual configurations, including prompt engi-
      solutions and tailored resource allocation (e.g.,                    neering, agent profiling, and inter-agent communication
      LLM calls, tool calls, token cost). Comprehen-                       pipelines (Qian et al.,2024). This dependency significantly
      sive evaluation across six benchmarks demon-                         limits the rapid adaptation of multi-agent systems to di-
      strates that MaAS (I) requires only 6 ∼ 45%                          versedomainsandapplicationscenarios(Tangetal.,2023;
      of the inference costs of existing handcrafted or                    Zhang et al.,2024c).  More recently, the research com-
      automated multi-agent systems, (II) surpasses                        munity has shifted toward automating multi-agent system
      them by 0.54% ∼  16.89%,  and (III) enjoys                           design. Forinstance, DsPy(Khattab etal.,2023) andEvo-
      superior cross-dataset and cross-LLM-backbone                        Prompting(Guoetal.,2023)automatepromptoptimization,
      transferability. The code is available athttps:                      GPTSwarm (Zhuge et al.,2024) and G-Designer (Zhang
     //github.com/bingreeky/MaAS.                                          et al.,2024b) optimize inter-agent communication, and
                                                                           EvoAgent (Yuan etal.,2024) and AutoAgents(Chen et al.,
                                                                           2023a) self-evolve agent profiling. Nevertheless, they typ-
1. Introduction                                                            ically focus on automating specific aspects of the system.
Large Language Model (LLM)-based agents (Richards &                        Subsequently,ADAS(Huetal.,2024a),AgentSqure(Shang
et al.,2023;Nakajima,2023;Reworkd,2023) have made                          et al.,2024), and AFlow (Zhang et al.,2024c)broaden the
remarkablestrides inaspectrum ofdomains,such asques-                       design search space. These state-of-the-art (SOTA) meth-
tion answering (Zhu et al.,2024a), data analysis (Hong                     ods optimize a single, complex (multi-)agent workflowfor a
                                                                           givendatasetviadifferentsearchparadigms,e.g.,heuristic
  *Equalcontribution   1NationalUniversityofSingapore 2Tongji              search (Hu et al.,2024a), Monte Carlo tree search (Zhang
University 3Nanyang TechnologicalUniversity 4Shanghai AILab-               etal.,2024c),andevolution(Shangetal.,2024),surpassing
oratory  5University of Science and Technology of China.  Cor-             the performance of manually designed systems.
respondence to: Kun Wang <wang.kun@ntu.edu.sg>, Lei Bai
<baisanshi@gmail.com>.                                                     Although the paradigm of searching for a one-size-fits-
Proceedings of the 42nd International Conference on Machine                all  multi-agent  system  appears  sufficient  to  optimize
Learning,Vancouver,Canada. PMLR267,2025. Copyright2025                     performance-related metrics suchas accuracyandpass@k,
by the author(s).                                                          its performance is largely constrained on resource-related

                                                                        1

## PDF page 2

Multi-agent Architecture Search via Agentic Supernet



















Figure1.(Left) The building blocks of MaAS; (Right) When confronting different queries, the agentic supernet adaptively samples
tailored multi-agent architecture in a query-dependent manner.

metrics, such as token cost, LLM calls, and inference la-                 During inference, for different queries, MaAS samples a
tency (Dilemma 1). Specifically, contemporary methods                     suitable multi-agent system delivering satisfactory resolu-
tend to optimize for a complex and resource-intensive agen-               tion andappropriate inferenceresources, therebyachieving
tic system, often involving dozens of LLM API calls and                   task-customized collective intelligence.
external tool usage (Liu et al.,2023). However, this is far               We conduct comprehensive evaluations on seven widely
from an optimal solution:  for example, in mathematical                   adopted benchmarks, covering diverse use cases in code
benchmarks(Hendrycksetal.,2021),Ph.D.-levelabstract                       generation  (HumanEval,  MBPP),  mathematical  reason-
algebramay indeedrequire complicated,token-heavysys-                      ing (GSM8K, MATH, SVAMP), and diverse tool usage
tems, while simpleelementary-level arithmetic works well                  (GAIA). Empirical results demonstrate that MaAS is❶
with a single zero-shot I/O. This paradigm becomes even                   high-performing,surpassing existinghandcraftedor auto-
more problematic when applied to benchmarks across mul-                   matedmulti-agentsystemsby0.54%∼ 16.89%;❷token-
tiple taskdomains (Dilemma 2): for instance,in the GAIA                   economical, outperforming the SOTA baseline AFlow on
benchmark (Mialon et al.,2023), there is no single sys-                   the MATH benchmark with 15% of the training cost and
temthat isoptimal forbothfilereadingandwebsearching                       25% of the inference cost;❸ transferable across datasets
tasks, leaving practitioners with no alternative but to split             and LLM-backbones;❹ inductive, demonstrating strong
the benchmark and optimize separately (Zhuge et al.,2024).                generalizability to unseen agentic operators.
Thesedilemmasunveilthat,theparadigmofautomatically
optimizing a single multi-agent architecture fails to meet                Brieflyput,ourkeycontributionsaresummarizedasfollows:
the dynamic and evolving demands of agentic deployment.
To addresstheabovechallenges,we proposeMulti-agent                        • ParadigmReformulation: We introduce the concept of
Architecture Search (MaAS), which, instead of search-                       agenticsupernet,aprobabilistic,continuousagenticar-
ingforaplausible(possibly non-existent) optimalsolution,                    chitecture distribution, which transforms the paradigm
generates a distribution of multi-agent systems.  Techni-                   of optimizing a single optimal multi-agent system into
cally, we model the optimization of MaAS on the agentic                     optimizing the distribution of multiple architectures.
supernet, a probabilistic, continuous agentic architecture
distribution that encompasses a vast number of possible                   • Practical Solution:   We  propose  MaAS,  an  agentic
multi-agent candidates. The agentic supernet can be seen                    supernet-based framework that automatically evolves
as a cascaded multi-layer workflow, including❶ multiple                     powerful multi-agent systems and adaptively allocates
agentic operators (e.g., CoT (Weiet al.,2022), Multi-agent                  high-performing andresource-efficient solutionsfor user
Debate(Duet al.,2023),ReAct(Yaoetal.,2023)), aswell                         queries with varied difficulty, domain and features.
as❷theparameterizedprobabilitydistributionsofoperators
acrosslayers. Duringtraining,MaASleveragesacontroller                     • ExperimentalEvaluation: Extensiveevaluations onsix
network to sample multi-agent architectures conditioned                     benchmarks demonstrate that our framework discovers
on input queries.  The distribution parameters and opera-                   novel agentic systems with 0.54% ∼  16.89% higher
tors are jointly updated based on environmental feedback,                   performance,significantlylowertraining/inferencecosts,
withtheformer’sgradientsapproximatedviaMonteCarlo                           transferabilityacrossbenchmarksandLLMs,andsuperior
sampling and the latter’s via textual gradient estimation.                  inductive capacity.


                                                                       2

## PDF page 3

Multi-agent Architecture Search via Agentic Supernet





















                                          Figure2. The overall framework of our proposed MaAS.

2. Related Work                                                            AutoML.   Automating the design of agentic systems is
LLM-AgentsandAgenticSystems.    Buildingonthesuc-                          an emerging topic, yet the history of AutoML (He et al.,
cess of single agents (Shen et al.,2024;Zhu et al.,2024b;                  2021)providesclearprecedents. Notably,theprogression
Zhongetal.,2024),studieshaveshownthatgroupingmulti-                        of agentic automation mirrors that of neural architecture
pleLLM-basedagentsintomulti-agentsystems(MAS)can                           search (NAS) (Ren et al.,2021).  Core NAS techniques,
substantiallyenhance individualmodel capabilities (Wang                    such as reinforcement learning (Zoph,2016), evolution-
et al.,2024a), as demonstrated in early attempts such as                   ary algorithms (Liu et al.,2021), Bayesian optimization
AutoGen(Wuetal.,2023),LLM-Debate(Duetal.,2023),                            (BO) (White et al.,2021), and MCTS (Wang et al.,2021),
andAgentVerse(Chenetal.,2023b). However,theyheav-                          have inspired analogousapproaches in agenticautomation,
ily relied on manually crafted designs, which constrained                  from policy gradient in(Zhuge et al.,2024) to evolutionary
the adaptability and flexibility of agents in addressing un-               search in (Yuan et al.,2024), BO in (Shang et al.,2024),
foreseen challenges (He et al.,2023;Chen et al.,2023b).                    andMCTSin(Zhangetal.,2024c). InNAS,however,these
As a result, automated agentic system design has gained                    black-box methods were eventually eclipsed by efficient su-
increasing attention in the academic community.                            pernettraining(Whiteet al.,2023),culminatinginseminal
                                                                           works like DARTS (Liu et al.,2018) and SNAS (Xie et al.,
                                                                           2018). Inspiredbythis,weintroducethefirstMASsearch-
Automating Agentic Systems.    Efforts to automate the                     ingframeworkleveraginganagenticsupernet,posingnew
design of agent-based systems can be broadly classified                    paradigms and challenges for agentic automation.

into  the  following  categories:   (I)  Prompt  Optimiza-                 3. Methodology
tion,  such  as  PromptBreeder  (Fernando  et  al., 2023),                 Figure2illustrates the overall workflow of our method.
DsPy (Khattab et al.,2023), and EvoPrompt (Guo et al.,                     MaAS takes diverse and varying difficulty queries as input
2023);         (II) Inter-agent Communication, which focuses               and leverages a controller to sample a subnetwork from
on orchestrating interactions between agents, including                    the agentic supernet for each query, corresponding to a cus-
GPTSwarm  (Zhuge  et  al., 2024),  DyLAN  (Liu  et  al.,                   tomized multi-agent system. After the sampled system exe-
2023), EvoMAC (Hu et al.,2024b), AgentPrune (Zhang                         cutes thequery, MaASreceivesenvironment feedbackand
et al.,2024a) and G-Designer (Zhang et al.,2024b); and                     jointly optimizes the supernet’s parameterized distribution
(III) Agent Profiling, represented by AgentVerse (Chen                     andagenticoperators. Inthefollowingsections,Section3.1
et al.,2023b), EvoAgent (Yuan et al.,2024), and AutoA-                     formally defines the search space and optimization objec-
gents (Chen et al.,2023a).   Further, ADAS (Hu et al.,                     tiveofMaAS,Section3.2detailshowthecontrollerquery-
2024a)andAgentSquare(Shangetal.,2024)providemore                           dependentlysamplesmulti-agentstructures,andSection3.3
comprehensive automation for single-agent design, while                    details the optimization of MaAS.
AFlow (Zhang et al.,2024c) achieves multi-agent work-
flow automation using Monte Carlo tree search (MCTS).                      3.1. Preliminary
However, these high-performing methods still follow the                    Search Space.   We first define the basic unit of MaAS’s
paradigm of searching for a single final system, whereas                   search space, namely the agentic operator as follows:
MaAS searches for distribution of architectures with lower
average inference costs (LLM calls, token cost,etc.).

                                                                        3

## PDF page 4

Multi-agent Architecture Search via Agentic Supernet

Definition3.1(AgenticOperator). AnagenticoperatorO                       3.2. Agentic Architecture Sampling
isa compositeLLM-agentinvocationprocessthat involves                     ThecoreofMaASliesintailoringacustomizedmulti-agent
multiple LLM calls and tool usage:                                       systemforeachuserquery,whichmayvaryindifficultyand
                O ={{Mi}mi=1,P,{Ti}ni=1},                                domain, to deliver a satisfactory solution:
                   Mi∈M,P ∈P,Ti∈T,                      (1)                                        Z
                                                                                 p(a|q,π,O) =          e(a|G)Qϕ(G|q,π,O)dG,      (6)
whereM andM correspond to LLM backbones and the
set of available LLMs, respectively.  Similarly, P andT                  whereQϕ representsthecontrollernetwork,whichtakesthe
represent prompts and tools. m andn denote the number of                 queryq, the parameterized distributionπ, and the available
LLM-agents and toolsinvoked inthe operator, respectively.                operatorsO, and outputs the sampled agentic architecture
Most existing single/multi-agent workflows can be viewed                 G.Qϕ isparameterizedbyϕ,ande(·|·) denotesproducing
as agentic operators: CoT (Wei et al.,2022) can be con-                  solution via executingG. we implementQϕ as follows:
sidered one with m = 1 and n = 0, denoted as OCoT;                                                       LY
Self-RAG (Asai et al.,2023) similarly involves             m = 1                   Qϕ(G|q,π,O) =            πℓ(Vℓ|q,{Vh}ℓ−1
agent allocation, but is equipped with n = 1 retrieval en-                                                                   h=1),         (7)
                                                                                                        ℓ=1
gine,denotedasOSRAG;Multi-agentdebate(Duetal.,2023)                      whereVh denotes the selected operators at layer h.  The
involvesmultipleLLM-agent,multi-turncalls,denotedas                      selection ofVℓ is conditionally dependent on the query q
ODebate.  The feasible set of agentic operators is denoted               and the operators from the previous layers. However, not
asO, and we discuss the initialization ofO in Section4.1                 all queries require execution acrossL layers. As discussed
and AppendixB.1. We define a multi-agent system as:                      in Section1, many questions can be resolved with a simple
              G ={V,E}, V ⊂O, E ∈V×V,               (2)                  zero-shot I/O (Zhang et al.,2024b), rendering               L layers
whereV is the set ofselected operators inG andE denotes                  unnecessarilyredundant. Toaddressthis,weintroducean
theirconnectivity.G isconstrainedasadirectacyclicgraph                   early-exitoperator,denotedasOexit. Duringsampling,if
(DAG). Finally, we define the agentic supernet:                          Oexit is encountered, the process exits early:
Definition 3.2(Agentic Supernet).  The agentic supernet                                          LY h                                  i
is denoted asA ={π,O} ={{πℓ(O)}O∈O}Lℓ=1, where:                            Qϕ(G|q,π,O) =             πℓ(Vℓ|q,{Vh}ℓ−1h=1)·IOexit/∈Vℓ
                                                                                                ℓ=1                                       (8)
              πℓ(O) =p(O|A1:ℓ−1),  O∈O,                                                         +IOexit∈Vℓ·δ     ℓ−ℓexit    ,
             A1:ℓ−1 ={{πk(O)}O∈O}ℓ−1k=1,                     (3)
where πℓ(O)  represents  the  probability  of  operator O                where ℓexit denotes the layer at whichOexit appears, and
present at layer ℓ, conditioned on the preceding layers                  δ(·) is the Kronecker delta function.  We implement the
A1:ℓ−1. The supernet induces a joint distribution over all               samplingprocessπϕ withaMixture-of-Expert(MoE)-style
possible multi-layer operator configurations:                            network (Shazeer et al.,2017;Huang et al.,2024):

                           LY  Y                                                   πℓ : q→Vℓ, Vℓ ={Oℓ1,Oℓ2,··· ,Oℓt},
                 p(G) =             πℓ(O)IO∈Vℓ,                   (4)                                      X
                          ℓ=1  O∈O                                                      t =   argmin           S↓k >thres,                (9)
                                                                                             k∈{1,···,|O|}j<k
whereIO∈Vℓ istheindicatorfunctionfortheinclusionofO
in the set of active operatorsVℓ at layerℓ.                              whereS↓ = sort(S,desc), andS∈R|O| = [S1,··· ,S|O|]
ProblemFormulation.    GivenabenchmarkDcomprising                        represents the activation scores of all feasible operators
multiplequeriesq andtheircorrespondingoracleanswers/-                    w.r.t. q. Note thatthres is a threshold value that governs
solutionsa,theobjectiveofMaASisnottoidentifyasingle                      operator activation.  Operators are activated sequentially,
optimalagenticsystemlikepreviouspractices(Zhangetal.,                    startingfromtheonewiththehighestscore,andtheprocess
2024c;Zhuge et al.,2024), but to optimize a conditional                  continues until the cumulative score exceedsthres. This
probability distribution as follows:                                     ensures that the number of selected operators per layer is
                    U(G;q,a)−λ·C(G;q), s.t.G⊂A (5)                     query-dependent,allowingMaAStodynamicallyallocate
  max                                                                    resources based on task complexity. S is given by: Si =
 P(G|q)E(q,a)∼D,G∼P(G|q)                                                 FFN(v(q)∥PO∈V1v(O)∥···∥PO∈Vℓ−1v(O)),  where
where P(G|q)  is  a  distribution  that  generates  query-               v(·) denotes the embedding function using lightweight
dependentagenticarchitectures. U(·)andC(·)representthe                   models like MiniLM (Wang et al.,2020) and Sentence-
utiulity/performance and cost ofG for queryq, respectively,              Bert(Reimers,2019),and                 ∥representsconcatenation. The
andλ is a trade-off parameter.                                           detailed implementation ofv(·) is placed in AppendixB.2.

                                                                      4

## PDF page 5

Multi-agent Architecture Search via Agentic Supernet

Upon completing the sequential sampling procedure in                      Algorithm 1Algorithm workflow of MaAS
MaAS, a task-specific multi-agent systemG is generated                    Input   :AdatasetD containingtrainingsetDtrain andtest
and executed toproduce theanswerea. In thenext section,                              set Dtest, Operator set O, Randomly initialized
we elucidate the process of updating the agentic supernet                            distributionπ, Controller networkQϕ
based on environmental feedback.                                          Output:Well-optimizedagenticsupernet,composedofdis-
3.3. Cost-constrained Supernet Optimization                                          tributionπ and operatorsO
We present the optimization objective of MaAS as follows:                 for(q,a) inDtrain do
                                                                              /* Sample  query-dependent  MAS           */
  min                                                                         for layerℓ← 1 toL do
   π,O E(q,a)∼D,G∼Qϕ[−p(a|q,π,O)+λ·C(G;q)]   (10)                                  Vℓ←πϕ(Vℓ|q,{Vh}ℓ−1h=1);                   ▷  Eq.9
whereC(·) evaluatesthecostofmulti-agentsystems,repre-                              ifℓ =L orOexit∈Vℓ then
sentedbytokencost,andλ isthetrade-offparameter. The                                    break//Exit  when  reaching
term p(a|q,π,O) in Equation (10) corresponds to Equa-                                       maximal  sampling  depth  or
tion (6), where the calculation of    e(a|G) often involves                                 encountering  the  early-exit
external tools or API-based LLM calls, rendering it non-                                    operator
differentiable. Therefore, we employ an empirical Bayes                       ObtainG←⟨V1,··· ,Vℓ⟩ for queryq;        ▷  Eq.8
Monte Carlo procedure (Carlin & Louis,2000;Yan et al.,                        /* Execute  sampled  MAS                         */
2021) to estimate the gradient w.r.t the distribution      π:                 ExecuteG and obtain˜a←e(a|G);             ▷  Eq.6
                                                                              /* Self-evolve  agentic  supernet       */
                          X      KXh                  i                       Compute loss w.r.t.π,∇πL;                   ▷  Eq.11
            ∇πL≈  1K                  mk∇πp(Gk)        ,                      Estimate loss w.r.tO via textual gradient;  ▷  Eq.12

                         (q,a)∈Dk=1                             (11)
                                                                              Updateπ andO accordingly;                  ▷  Eq.10
          mk =   p(a|q,Gk)P               P
                     ip(a|q,Gi)−λ·   C(Gk;q)iC(Gi;q),
where mk denotes the cost-aware importance weights of                     (suchas merging,splitting, altering, etc.), respectively. See
the agentic architecture.  Intuitively, the distributionπ is              prompts in AppendixB.3. In this way, the core components
updated to favor multi-agent systems that generate high-                  of the agentic supernet, namely the agentic operators and
quality solutions with minimal token cost.                                their connectivity, are jointly updated, enabling the fully
                                                                          automatedevolutionofmulti-agentsystems. Wesummarize
                                                                          the notations in Table5, and the algorithm in Algorithm1.
                                                                          4. Experiments
                                                                          4.1. Experiment Setup
                                                                          TasksandBenchmarks.   WeevaluateMaASonsixpublic
                                                                          benchmarkscoveringthreedomains: (1)mathreasoning,
                                                                          GSM8K (Cobbe et al.,2021), MATH (Hendrycks et al.,
                                                                          2021), and MultiArith (Roy & Roth,2016);                          (2) code gener-
                                                                          ation, HumanEval (Chen et al.,2021) and MBPP (Austin
                                                                          et al.,2021)); and             (3) tool use, GAIA (Mialon et al.,2023).
        Figure3. The demonstration of textual gradient.                   For the MATH benchmark, we follow (Hong et al.,2024)
However,thegradientw.r.toperators∇OLcannotbecom-                          in selecting 617 problems from four typical problem types
putedsimilarly. AsshowninEquation(1),operatorsinclude                     (Combinatorics&Probability,NumberTheory,Pre-algebra,
black-box tool usage and natural language prompts, mak-                   Pre-calculus). The dataset statistics are in AppendixC.1.
ing numerical gradient updates infeasible. To address this,               Baselines.   WecompareMaASwiththreeseriesofagentic
we utilize agent-based textual gradient  (Hao et al.,2023;                baselines: (1)singleagentexecution methods,including
Liu et al.,2023;Hu et al.,2024b;Zhou et al.,2024) to                      CoT (Wei et al.,2022), ComplexCoT (Fu et al.,2022), Self-
approximate the backpropagation for agentic operators, as                 Consistency (Wang et al.,2023a);                 (2) hand-craft multi-
visualized in Figure3and formalized as follows:                           agentsystems,includingMultiPersona(Wangetal.,2023b),
 ∇OL =TP⊕TT ⊕TN,Tx∈T,x∈{P,T,N}  (12)                                      LLM-Debate(Duetal.,2023), LLM-Blender(Jiangetal.,
                                                                          2023),DyLAN(Liuetal.,2023),AgentVerse(Chenetal.,
where TP,TT,TN  represent  agent-generated  gradient                      2023b) and MacNet (Qian et al.,2024);                          (3) (partially
analyses in textual format, corresponding to updates of the               or fully) autonomous multi-agent systems,  including
prompt, model temperature, and operator node structure                    GPTSwarm (Zhuge et al.,2024), AutoAgents (Chen et al.,

                                                                      5

## PDF page 6

Multi-agent Architecture Search via Agentic Supernet


             Table1.Performancecomparisonwithsingleagent,hand-craftmulti-agentsystems, andautomatedagenticworkflows. The baseLLMis
             consistently set asgpt-4o-mini for all baselines. Weboldthe best results and underline               the runner-ups.

























             Table2.Performance onGAIA benchmark. The best andrunner-                     taskdomains. Themulti-agentsystemoptimizedbyMaAS
             up results areboldedand underlined, respectively.                            outperforms manually designed methods by an average
                Method             Level 1       Level 2      Level 3       Avg.          of 3.90 ∼  6.40% and existing automated methods by
                GPT-4o-mini          7.53            4.40              0             4.65 2.07 ∼ 8.26%.  Overall, as for mathematical reasoning
                GPT-4                9.68         1.89         2.08         4.05          andcode generation,MaASachieves anaveragebest score
                AutoGPT             13.21             0              3.85           4.85  of 83.59%, demonstrating its versatility and superiority.
                TapeAgent           23.66         14.47       10.20        16.61          Table2shows a comparison of MaAS with automated
                Sibyl               21.51         15.72           4.08          15.61     systems and three additional baselines, including Auto-
                AutoAgents          16.13           0            0          5.16          GPT(Richards&etal.,2023),TapeAgent(Bahdanauetal.,
                GPTSwarm            23.66         16.35           2.04          16.33     2024), and Sibyl (Wang et al.,2024b) onthe GAIA bench-
                ADAS                13.98         4.40           0          6.69          mark. GAIA encompasses tasks from various domains such
                AgentSquare         22.58         15.72           6.25          16.34     asweb browsing, filereading, andmultimodalunderstand-
                AFlow               10.75         8.81         4.08         8.00          ing, making it challenging to pursue a single optimal multi-
                MaAS                25.91       22.01          6.25        20.69          agentsystemforalltasks. Thus,themodestimprovements
             2023a),ADAS(Huetal.,2024a),AgentSquare(Shangetal.,                           of AFlow andADAS over vanilla LLMs(only3.35%↑ and
             2024) and AFlow (Zhang et al.,2024c). More details on                        2.04%↑onaverage)areunderstandable. Incontrast,MaAS
             baseline setups are provided in AppendixC.2.                                 canadaptivelysamplecustomizedagenticsystemsfordiffer-
                                                                                          ent domains,achieving18.38% and17.61% improvements
             Implementation details.   We leverage both close-source                      on Level 1 and 2 tasks, respectively.
             LLM (gpt-4o-mini-0718 (OpenAI,2024)) and open-
             source LLM (Qwen-2.5-72b-instruct (Yang et al.,                              4.3. Cost Analysis
             2024) and       llama-3.1-70b (Dubey et al.,2024)).  All                     To answerRQ2,wedemonstratethatMaASisbothtrain-
             models are accessed via APIs with the temperature set to                     ing/inferencecost-efficientfromthefollowingthreedimen-
             1. We set the number of layers asL = 4, the cost penalty                     sions: (1)tokencost,(2) APIcost,and(3) wall-clocktime,
             coefficient λ as λ ∈ {1e− 3,5e− 3,1e− 2}, and the                            as shown in Table3and Figure4. We observe:
             sampling timesK = 4. thres = 0.3 for Equation (9).                           Obs.❷ MaAS’s optimization is resource-friendly.  As

             4.2. Performance Analysis                                                    shown in Figure4           (Training Tokens), among the various
             We compare MaAS with 14 baselines on the GSM8K,                              optimization-oriented agentic workflows, MaAS achieves
             MATH, MultiArith, HumanEval, and MBPP benchmarks                             thehighestaccuracywiththeleasttrainingtokenconsump-
Method       inTable1,andwith10baselineson GAIAinTableGSM8K    MATH2. The     MultiArithtion.  While AFlow’s accuracy is comparable to that ofHumanEvalMBPPAvg.
             following observations can be made:                                          MaAS, its training costreaches22.50$, which is6.8× that
Vanilla                                        87.45            46.29             96.85              87.08              71.83          77.50of MaAS (merely 3.38$).  Additionally, existing agentic
CoT(Weietal.,2022Obs.❶ MaAS achieves optimal performance across all)87.10↓0.3546.40↑0.1196.31↓0.5488.13↑1.05         71.83↓0.00        77.95
ComplexCoT(Fuetal.,2022)                    86.89↓0.56      46.53↑0.24       96.70↓0.15        87.49↑0.41        72.36↑0.53       78.006
SC(CoT×5)(Wangetal.,2023a)                  87.57↑0.12       47.91↑1.62        96.58↓0.27         88.60↑1.52         73.60↑1.77        78.85
MultiPersona(Wangetal.,2023b)               87.50↑0.05      45.43↓0.86       97.49↑0.64        88.32↑1.24        73.19↑1.36       78.39
LLM-Debate(Duetal.,2023)                    89.47↑2.02       48.54↑2.25        97.33↑0.48         88.68↑1.60         70.29↓1.54        78.86
LLM-Blender(Jiangetal.,2023)                88.35↑0.90      46.92↑0.63       97.29↑0.44        88.80↑1.72        77.05↑5.22       79.68
DyLAN(Liuetal.,2023)                        89.98↑2.53       48.63↑2.34        97.12↑0.27         90.42↑3.34         77.30↑5.47        80.69
AgentVerse(Chenetal.,2023b)                 89.91↑2.46      47.35↑1.06       97.50↑0.65        89.29↑2.21        74.28↑2.45       79.67
MacNet(Qianetal.,2024)                      87.95↑0.50       45.18↓1.11        96.03↓0.82         84.57↓2.51         65.28↓6.55        75.00
AutoAgents(Chenetal.,2023a)                 87.69↑0.24      45.32↓0.97       96.42↓0.43        87.64↑0.56        71.95↑0.12       77.80
GPTSwarm(Zhugeetal.,2024)                   89.14↑1.69       47.88↑1.59        96.79↓0.06         89.32↑2.24         77.43↑5.60        80.11
ADAS(Huetal.,2024a)                         86.12↓1.33      43.18↓3.11       96.02↓0.83        84.19↓2.89        68.13↓3.70       75.13
AgentSquare(Shangetal.,2024)                87.62↑0.17       48.51↑2.22        97.77↑0.92         89.08↑2.00         78.46↑6.63        80.29
AFlow(Zhangetal.,2024c)                     91.16↑3.71      51.28↑4.91       96.22↓0.63        90.93↑3.85        81.67↑9.84       82.25
MaAS(Ours)                                  92.30↑4.85       51.82↑5.53        98.80↑1.95        92.85↑5.77         82.17↑10.34       83.59

## PDF page 7

Multi-agent Architecture Search via Agentic Supernet


               Table3.EfficiencycomparisonbetweenMaASandstate-of-the-artbaselinesontheMATHBenchmark. Weshadethevaluesofthelowest
               token/cost/wall-clock time and the highest performance.









































               Figure4.The cost analysis of MaAS on MATH benchmark.



               automation pipelines are relatively time-consuming, with
               DyLAN taking 508 minutes and GPTSwarm taking 129                             Figure5.ThevisualizationofMaAS’soperatorsamplingprocess.
               minutes. In contrast, the optimization wall-clock time of
               MaAS requires only53 minutes.                                                4.4. Case Study
               Obs.❸ Agentic supernet enjoys superior token economy                         In thissection, weexplore andvisualize theintrinsic mech-
               during inference.  As shown in Figure4           (Inference API              anisms of the agentic supernet.  Figure5showcases the
               Cost), MaAS achieves the highest accuracy with an API                        probability distributions of theagentic supernet whenfaced
               cost of 0.42$, demonstrating its high performance and to-                    withdifferentqueries,andFigure6presentsthemulti-agent
               ken economy. Although AgentSquare’s API cost is slightly                     systems designed by MaAS for queries from the MATH,
               lower thanMaAS’s, thisisdue toits limitationto asingle-                      GAIA, and HumanEval benchmarks. We have:
               agent search, which severely restricts its performance (re-
               sulting in a4% drop compared to MaAS). Table3further                         Obs.❹ MaAS learns to query-aware early exit from the
               highlights that MaAS has the lowest prompt/completion                        reasoningprocess. AsshowninFigure5,whenfacedwith
               token consumption, the lowest API cost, and the shortest                     theeasyqueries(a)and(b),MaASexitsmulti-agentarchi-
               wall-clock time during inference. These advantages can be                    tecture sampling at the second layer with probabilities of
               attributed to the agentic supernet’s ability to dynamically                  0.37and0.47,respectively,selectingtheearly-exitoperator.
               allocate resources based on the difficulty of the query.                     Notably, query (b) chose two agentic operators at the first

Method                                         Training                                                             Inference                               Overall7
                    Prompt        Completion         Total    Wall-clock       Prompt        Completion        Total    Wall-clock        Acc.
                      token           token        cost($)    time(min)          token           token       cost($)     time(min)        (%)
LLM-Debate           -                    -                -               -         3,275,764  10,459,097   6.76$         92         48.54
DyLAN         22,152,407  16,147,052  13.01$       508       6,081,483   3,303,522    2.89$         39         48.63
MacNet                   -                    -                -               -         7,522,057   2,043,600    2.35$         47         45.18
GPTSwarm    21,325,266   6,369,884    7.02$        129       3,105,571                         788,273      0.93$         30         47.88
AFlow           33,831,239  29,051,840  22.50$       184       2,505,944   2,151,931    1.66$         23         51.28
MaAS              3,052,159        2,380,505        3.38$          53        1,311,669    853,116             0.42$          19          51.82

## PDF page 8

Multi-agent Architecture Search via Agentic Supernet


















                          Figure6. Case study and visualization for MaAS. Queries are from HumanEval, MATH and GAIA benchmarks.

                                                                                         early-exit operator in Equation (8); and    (3)w/oC(·), elimi-
                                                                                         natingthecostconstraintinEquation(10). Weobservefrom
                                                                                         Table4thatremoving thetextualgradient causes thelargest
                                                                                         performancedrop,asitdisablesMaAS’sself-evolvingca-
                                                                                         pability. RemovingOexit and C(·) results in little impact
                                                                                         on performance, but it weakens MaAS’s query-dependent
                                                                                         nature and unnecessarily increases the inference cost.
              Figure7.Parameter sensitivity analysis of MaAS. The unit of                                Table4. Ablation study of MaAS.
              cost per query (right) and performance (left) is 10−3 · $ and
              pass@1 (%), respectively.

              layer: direct I/O and ReAct, demonstrating MaAS’s abil-
              itytodynamicallyallocatedifferentoperatorsateachlayer
              (corresponding to Equation (9)). For the more challenging
              queries(c)and(d),MaASsampledadditionallayers,further
              provingitsabilitytocustomizethemulti-agentsystembased
              on queryawareness. This is alsovisualized byFigure8, in
              which the probability ofOexit becomes increasingly high                    TransferabilityAnalysis.    Weevaluatewhethertheagen-
              with the supernet depth increases.                                         tic supernet of MaAS is (1) model-agnostic and (2) gener-
              4.5. Framework Analysis                                                    alizable across datasets, with results presented in Tables7
              SensitivityAnalysis    WeanalyzethesensitivityofMaAS                       and8. Asshown,theagenticsupernetoptimizedbyMaAS
              tothreecoreparameters: thenumberoflayersintheagentic                       transfers well to models such as Qwen-2.5-70b, with
              supernetL, the cost penalty coefficientλ in Equation (10),                 4.98%∼ 5.50%↑ in performance, while also demonstrat-
              andthe samplingcountK inEquation (11). Theresults are                      ing strong cross-dataset generalization.
              presented inFigure7.             For theparameterL, weobserve a            InductiveAnalysis.    ToevaluatewhetherMaASpossesses
              significant performance improvement asL increases from                     inductive capabilities, i.e., the ability to generalize to un-
              2to4(89.5%→ 92.8%). However,furtherincreasesyield                          seenagenticoperators,weselecttheDebate(Duetal.,2023)
              only marginal performance gains while incurring higher                     operatorasaholdout. Wethencomparetheoperatordistri-
              per-query inference costs. Considering both performance                    bution ofMaAS duringinference withand without Debate
              and cost, we selectL = 4. For the parameterλ, we find                      in Figures8and9. The results demonstrate that MaAS can
              that larger values lead MaAS to favor more cost-efficient                  still reasonably activate and utilize the unseen operator at
              solutions, albeit with some performance degradation. For                   an appropriate proportion.
              theparameterK,wenotethat performanceissuboptimal
              with highest variance when K  = 2.  Increasing K to 4                      5. Conclusion
              effectively achieves a satisfactory low-variance estimation.               Inthispaper,weforthefirsttimeshifttheparadigmofauto-
              Ablation Study   We perform an ablation study on three                     mated multi-agent system design from seeking a (possibly
              key components of MaAS: (1)w/o∇OL, removing the tex-                       non-existent) single optimal system to optimizing a prob-
              tual gradientin Equation (12);      (2)w/oOexit, removing the              abilistic, continuous distribution of agentic architectures,

                                                                                     8
    Dataset             HumanEval                  MATH
     Metric         Pass@1       Cost      Accuracy        Cost
                      (%)      (10−3 $)       (%)       (10−3 $)
 VanillaMaAS         92.85       1.01        51.82        0.86
MaASw/o∇OL           90.17       0.90        48.23        0.84
MaASw/oOexit         91.44       1.67        51.53        1.04
MaASw/oC(·)          92.94       1.38        51.19        1.28

## PDF page 9

Multi-agent Architecture Search via Agentic Supernet

termed the agentic supernet. Building on this concept, we                 Carlin,B.P.andLouis,T.A. Empiricalbayes: Past,present
propose MaAS, which dynamically samples multi-agent                          and future. Journal of the American Statistical Associa-
systemsthatdeliversatisfactoryperformanceandtokeneffi-                       tion, 95(452):1286–1289, 2000.
ciencyforuserqueriesacrossdifferentdomainsandvarying                      Chen, G., Dong, S., Shu, Y., Zhang, G., Sesay, J., Karls-
levels of difficulty. We believe that MaAS paves the way                     son, B. F., Fu, J., and Shi, Y.   Autoagents:  A frame-
towardfullyautomated,self-organizing,andself-evolving                        work for automatic agent generation.   arXiv preprint
collective intelligence.                                                     arXiv:2309.17288, 2023a.

Acknowledgements                                                          Chen, J., Wang, X., Xu, R., Yuan, S., Zhang, Y., Shi, W.,
                                                                             Xie, J., Li, S., Yang, R., Zhu, T., et al.  From persona
This research is supported by the National Natural Science                   to personalization: A survey on role-playing language
Foundation of China (No.  92270114), and the Shanghai                        agents. arXiv preprint arXiv:2404.18231, 2024.
Municipal Science and Technology Major Project.                           Chen,  M.,  Tworek,  J.,  Jun,  H.,  Yuan,  Q.,  Ponde  de
Impact Statement                                                             Oliveira Pinto, H., Kaplan, J., Edwards, H., Burda, Y.,
                                                                             Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger,
Ethical Considerations.   We believe that our proposed                       G.,Petrov, M.,Khlaaf, H.,Sastry,G., Mishkin,P.,Chan,
MaAS framework raises no ethical concerns regarding its                      B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser,
motivation, design, implementation, or data usage.  The                      L., Bavarian, M.,Winter, C.,Tillet, P.,Petroski Such, F.,
method is designed to advance the automation of multi-                       Cummings, D., Plappert, M., Chantzis, F., Barnes, E.,
agent systems in a principled and resource-efficient manner,                 Herbert-Voss, A., Hebgen Guss, W., Nichol, A., Paino,
ensuring responsible development while adhering to ethical                   A.,Tezak,N.,Tang,J.,Babuschkin,I.,Balaji,S.,Jain,S.,
guidelines in AI research.                                                   Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam,
                                                                             J., Misra, V., Morikawa, E., Radford, A., Knight, M.,
Societal Implications.   MaAS introduces a new paradigm                      Brundage,M.,Murati,M.,Mayer,K.,Welinder,P.,Mc-
inmulti-agentsystemdesignbyreplacingstatic,one-size-                         Grew,B., Amodei, D., McCandlish, S.,Sutskever, I.,and
fits-all architectures with a dynamic and adaptive agentic                   Zaremba,W. Evaluatinglargelanguagemodelstrained
supernet. This approach enables fine-grained resource allo-                  on code, July 01, 2021 2021.
cation tailored to query difficulty and domain, significantly             Chen,W.,Su,Y.,Zuo,J.,Yang,C.,Yuan,C.,Qian,C.,Chan,
improvingefficiencywhilemaintaininghigh-qualityoutputs.                      C.-M.,Qin,Y.,Lu,Y.,Xie,R.,Liu,Z.,Sun,M.,andZhou,
By reducing inference costs and enhancing the flexibility                    J. Agentverse: Facilitatingmulti-agentcollaborationand
of multi-agent workflows, MaAS has the potential to de-                      exploring emergent behaviors in agents, 2023b.
mocratize access to intelligent automation across diverse
applications, including education, research, and industry.                Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H.,
                                                                             Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano,
References                                                                   R.,Hesse,C.,andSchulman,J. Trainingverifierstosolve
Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. Self-                math word problems. arXiv prepring, abs/2110.14168,
   rag: Learning to retrieve,generate,andcritiquethrough                     2021.
   self-reflection. arXiv preprint arXiv:2310.11511, 2023.                Deng, X., Gu, Y., Zheng, B., Chen, S., Stevens, S., Wang,
Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski,                      B., Sun, H., and Su, Y. Mind2web: Towards a general-
   H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al.              ist agent for the web.  Advances in Neural Information
   Program synthesis with large language models.  arXiv                      Processing Systems, 36, 2024.
   preprint arXiv:2108.07732, 2021.                                       Du,Y.,Li,S.,Torralba,A.,Tenenbaum,J.B.,andMordatch,
Bahdanau,D.,Gontier,N.,Huang,G.,Kamalloo,E.,Pardi-                           I. Improvingfactualityandreasoninginlanguagemodels
   nas,R.,Pich´e,A.,Scholak,T.,Shliazhko,O.,Tremblay,                        throughmultiagentdebate. CoRR,abs/2305.14325,2023.
  J.P.,Ghanem,K.,etal. Tapeagents: aholisticframework                     Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle,
   foragent development andoptimization. arXivpreprint                       A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan,
   arXiv:2412.08445, 2024.                                                   A., et al.  The llama 3 herd of models.  arXiv preprint
Butt, N., Chandrasekaran, V., Joshi, N., Nushi, B., and                      arXiv:2407.21783, 2024.
   Balachandran,  V.    Benchagents:   Automated  bench-                  Feng,  T.,  Shen,  Y.,  and  You,  J.      Graphrouter:    A
   mark creation with agent interaction.   arXiv preprint                    graph-based router for llm selections.   arXiv preprint
   arXiv:2410.22584, 2024.                                                   arXiv:2410.03834, 2024.

                                                                       9

## PDF page 10

Multi-agent Architecture Search via Agentic Supernet

Fernando, C., Banarse, D., Michalewski, H., Osindero,                   Huang, D., Bu, Q., Zhang, J. M., Luck, M., and Cui,
  S.,andRockt¨aschel,T. Promptbreeder: Self-referential                    H.    Agentcoder:  Multi-agent-based code generation
  self-improvement via prompt evolution. arXiv preprint                    with iterative testing and optimisation.  arXiv preprint
  arXiv:2309.16797, 2023.                                                  arXiv:2312.13010, 2023.

Fu, Y., Peng, H., Sabharwal, A., Clark, P., and Khot, T.                Huang, Q., An, Z., Zhuang, N., Tao, M., Zhang, C., Jin,
  Complexity-based prompting for multi-step reasoning.                     Y., Xu, K., Chen, L., Huang, S., and Feng, Y.  Harder
  In The Eleventh International Conference on Learning                     tasksneedmoreexperts: Dynamicroutinginmoemodels.
  Representations, 2022.                                                   arXiv preprint arXiv:2403.07652, 2024.
Guo, Q.,Wang,R., Guo,J., Li,B., Song, K.,Tan,X., Liu,                   Jiang,D.,Ren,X.,andLin,B.Y.LLM-blender: Ensembling
  G.,Bian,J.,andYang,Y. Connectinglargelanguagemod-                        largelanguagemodelswithpairwiserankingandgenera-
  elswithevolutionaryalgorithmsyieldspowerfulprompt                        tivefusion. InProceedingsofthe61stAnnualMeetingof
  optimizers. arXiv preprint arXiv:2309.08532, 2023.                       the Association for Computational Linguistics (Volume 1:
Hao, R., Hu, L., Qi, W., Wu, Q., Zhang, Y., and Nie, L.                    Long Papers), pp. 14165–14178, Toronto, Canada, July
  Chatllmnetwork: Morebrains,moreintelligence,April                        2023. Association for Computational Linguistics.

  01, 2023 2023.                                                        Khattab, O., Singhvi, A., Maheshwari, P., Zhang, Z., San-
Hatalis,K.,Christou,D.,Myers,J.,Jones,S.,Lambert,K.,                       thanam, K., Vardhamanan, S., Haq, S., Sharma, A., Joshi,
  Amos-Binks,A.,Dannenhauer,Z.,andDannenhauer,D.                           T. T., Moazam, H., et al. Dspy: Compiling declarative
  Memorymatters: Theneedtoimprovelong-termmemory                           languagemodelcallsintoself-improvingpipelines. arXiv
  in llm-agents.  In Proceedings of the AAAI Symposium                     preprint arXiv:2310.03714, 2023.
  Series, volume 2, pp. 277–280, 2023.                                  Li,G.,Hammoud,H.,Itani,H.,Khizbullin,D.,andGhanem,
                                                                           B.  CAMEL: communicative agents for ”mind” explo-
He, X., Zhao, K., and Chu, X.  Automl: A survey of the                     rationoflargelanguagemodelsociety. InNeurIPS,2023.
  state-of-the-art. Knowledge-based systems, 212:106622,
  2021.                                                                 Li, Z., Zang, Q., Ma, D., Guo, J., Zheng, T., Liu, M., Niu,
                                                                           X., Wang, Y., Yang, J., Liu, J., et al.  Autokaggle:  A
He,Z.,Cao,P.,Chen,Y.,Liu,K.,Li,R.,Sun,M.,andZhao,                          multi-agentframeworkforautonomousdatasciencecom-
  J. Lego: Amulti-agentcollaborativeframeworkwithrole-                     petitions. arXiv preprint arXiv:2410.20424, 2024.
  playinganditerativefeedbackforcausalityexplanation
  generation. In Findings oftheAssociationfor Computa-                  Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang,
  tional Linguistics: EMNLP 2023, pp. 9142–9163, 2023.                     R., Yang, Y., Tu, Z., and Shi, S. Encouraging divergent
                                                                           thinking in large language models through multi-agent
Hendrycks, D., Burns,C., Kadavath, S.,Arora, A., Basart,                   debate. CoRR, abs/2305.19118, 2023.
  S.,Tang,E.,Song,D.,andSteinhardt,J. Measuringmath-                    Liu,H.,Simonyan,K.,andYang,Y. Darts: Differentiable
  ematicalproblemsolvingwiththemathdataset. NeurIPS,                       architecture search.  arXiv preprint arXiv:1806.09055,
  2021.                                                                    2018.

Hong,S., Zheng,X.,Chen, J.,Cheng,Y.,Wang,J.,Zhang,                      Liu, Y., Sun, Y., Xue, B., Zhang, M., Yen, G. G., and Tan,
  C., Wang, Z., Yau, S. K. S., Lin, Z., Zhou, L., Ran, C.,                 K.C. Asurveyonevolutionaryneuralarchitecturesearch.
  Xiao, L., and Wu, C. Metagpt: Meta programming for                       IEEE transactions on neural networks and learning sys-
  multi-agent collaborative framework, August 01, 2023                     tems, 34(2):550–570, 2021.
  2023.
Hong, S., Lin, Y., Liu, B., Liu, B., Wu, B., Zhang, C.,                 Liu,Z.,Zhang,Y.,Li,P.,Liu,Y.,andYang,D. Dynamicllm-
  Wei, C., Li, D., Chen, J., Zhang, J., et al.  Data inter-                agent network: An llm-agent collaboration framework
  preter:  An llm agent for data science.  arXiv preprint                  with agent team optimization.  CoRR, abs/2310.02170,
  arXiv:2402.18679, 2024.                                                  2023.

Hu,S.,Lu,C.,andClune,J. Automateddesignofagentic                        Madaan,   A.,   Tandon,   N.,   Gupta,   P.,   Hallinan,   S.,
  systems. arXiv preprint arXiv:2408.08435, 2024a.                         Gao,  L.,  Wiegreffe,  S.,  Alon,  U.,  Dziri,  N.,  Prab-
                                                                           humoye,  S.,  Yang,  Y.,  Gupta,  S.,  Majumder,  B. P.,
Hu, Y., Cai, Y., Du, Y., Zhu, X., Liu, X., Yu, Z., Hou,                    Hermann,  K.,  Welleck,  S.,  Yazdanbakhsh,  A.,  and
  Y., Tang, S., and Chen, S.   Self-evolving multi-agent                   Clark, P.   Self-refine:  Iterative refinement with self-
  collaboration networksfor software development. arXiv                    feedback. In NeurIPS, 2023. URLhttp://papers.
  preprint arXiv:2410.16946, 2024b.                                        nips.cc/paper_files/paper/2023/hash/

                                                                    10

## PDF page 11

Multi-agent Architecture Search via Agentic Supernet

  91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.Shang, Y., Li, Y., Zhao, K., Ma, L., Liu, J., Xu, F., and Li,
  html.                                                                   Y. Agentsquare: Automatic llm agentsearch in modular
Mialon,G., Fourrier, C.,Swift,C., Wolf,T.,LeCun, Y.,and                   design space. arXiv preprint arXiv:2410.06153, 2024.

  Scialom, T. Gaia: a benchmark for general ai assistants.              Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le,
  arXiv preprint arXiv:2311.12983, 2023.                                  Q., Hinton, G., and Dean, J. Outrageously large neural
Nakajima,  Y.     Babyagi.    https://github.com/                         networks: The sparsely-gated mixture-of-experts layer.
  yoheinakajima/babyagi, 2023.                                            arXiv preprint arXiv:1701.06538, 2017.

OpenAI.   Gpt-4o mini:  Advancing cost-efficient intelli-               Shen, Y., Song, K., Tan, X., Li, D., Lu, W., and Zhuang,
  gence,2024. URLhttps://openai.com/index/                                Y.  Hugginggpt:  Solving ai tasks with chatgpt and its
  gpt-4o-mini-advancing-cost-efficient-intelligence.friends in hugging face. Advances in Neural Information
                                                                          Processing Systems, 36, 2024.
Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G.,               Shinn,  N.,  Labash,  B.,  and  Gopinath,  A.    Reflexion:
  Stoica, I., and Gonzalez, J. E.  Memgpt: Towards llms                   an autonomous agent with dynamic memory and self-
  asoperatingsystems. arXivpreprintarXiv:2310.08560,                      reflection. arXivpreprint,abs/2303.11366,2023. doi: 10.
  2023.                                                                   48550/arXiv.2303.11366. URLhttps://doi.org/
Piatti, G., Jin, Z., Kleiman-Weiner, M., Sch¨olkopf, B.,                  10.48550/arXiv.2303.11366.
  Sachan, M., and Mihalcea, R.  Cooperate or collapse:                  Tang, N., Yang, C., Fan, J., Cao, L., Luo, Y., and Halevy,
  Emergenceofsustainabilitybehaviorsinasocietyofllm                       A.    Verifai:   verified generative ai.    arXiv preprint
  agents. arXiv preprint arXiv:2404.16698, 2024.                          arXiv:2307.02796, 2023.

Qian, C., Xie, Z., Wang, Y., Liu, W., Dang, Y., Du, Z.,                 Wang, L., Xie, S., Li, T., Fonseca, R.,and Tian, Y. Sample-
  Chen,W.,Yang,C.,Liu,Z.,andSun,M. Scalinglarge-                          efficientneuralarchitecturesearchbylearningactionsfor
  language-model-based multi-agentcollaboration. arXiv                    monte carlo tree search. IEEE Transactions on Pattern
  preprint arXiv:2406.07155, 2024.                                        Analysis and Machine Intelligence, 44(9):5503–5515,
Qiao, S., Zhang, N., Fang, R., Luo, Y., Zhou, W., Jiang,                  2021.
  Y. E., Lv, C., and Chen, H.  Autoact: Automatic agent                 Wang,L.,Ma,C.,Feng,X.,Zhang,Z.,Yang,H.,Zhang,J.,
  learning from scratch via self-planning.   In ACL. As-                  Chen, Z., Tang, J., Chen, X., Lin, Y., Zhao, W. X., Wei,
  sociation for Computational Linguistics, 2024.   URL                    Z.,and Wen,J. A surveyonlargelanguage modelbased
  https://arxiv.org/abs/2401.05268.                                       autonomous agents. Front. Comput. Sci., 18, 2024a.

Reimers, N.  Sentence-bert: Sentence embeddings using                   Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., and
  siamesebert-networks. arXivpreprintarXiv:1908.10084,                    Zhou, M.  Minilm: Deep self-attention distillation for
  2019.                                                                   task-agnostic compression of pre-trained transformers.
Ren, P., Xiao, Y., Chang, X., Huang, P.-Y., Li, Z., Chen, X.,             Advances in Neural Information Processing Systems, 33:
  andWang,X. Acomprehensivesurveyofneuralarchitec-                        5776–5788, 2020.
  turesearch: Challengesandsolutions. ACMComputing                      Wang,  X.,  Wei,  J.,  Schuurmans,  D.,  Le,  Q.  V.,  Chi,
  Surveys (CSUR), 54(4):1–34, 2021.                                       E. H., Narang, S., Chowdhery, A., and Zhou, D.  Self-
Reworkd.        Agentgpt.      https://github.com/                        consistencyimproveschainofthoughtreasoninginlan-
  reworkd/AgentGPT, 2023.                                                 guage models. In The Eleventh International Conference
Richards,   T.   B.   and   et   al.        Auto-gpt:     An   au-        on Learning Representations, 2023a.

  tonomousgpt-4experiment.https://github.com/                           Wang, Y., Shen, T., Liu, L., and Xie, J.  Sibyl:  Simple
  Significant-Gravitas/Auto-GPT, 2023.                                    yet effective agent framework for complex real-world
Roy, S. and Roth, D.   Solving general arithmetic word                    reasoning. arXiv preprint arXiv:2407.10718, 2024b.

  problems. arXiv preprint arXiv:1608.01413, 2016.                      Wang, Z., Mao, S., Wu, W., Ge, T., Wei, F., and Ji, H.
Saad-Falcon, J., Lafuente, A. G., Natarajan, S., Maru, N.,                Unleashing cognitive synergy in large language mod-
  Todorov, H., Guha, E., Buchanan, E. K., Chen, M.,                       els:  A task-solving agent through multi-persona self-
  Guha, N., R´e, C., et al. Archon: An architecture search                collaboration, July 01, 2023 2023b. work in progress.

  framework for inference-time techniques. arXiv preprint               Wei,J.,Wang,X.,Schuurmans,D.,Bosma,M.,Ichter,B.,
  arXiv:2409.15254, 2024.                                                 Xia, F., Chi, E., Le, Q., and Zhou, D. Chain-of-thought

                                                                   11

## PDF page 12

Multi-agent Architecture Search via Agentic Supernet

  prompting elicits reasoning in large language models,                  Zhang, J., Xiang, J., Yu, Z., Teng, F., Chen, X., Chen, J.,
  January 01, 2022 2022.                                                    Zhuge,M.,Cheng,X., Hong,S.,Wang,J.,et al. Aflow:
                                                                            Automating agentic workflow generation. arXiv preprint
White,  C.,  Neiswanger,  W.,  and Savani,  Y.    Bananas:                  arXiv:2410.10762, 2024c.
  Bayesianoptimization withneuralarchitectures forneu-
  ralarchitecturesearch. InProceedingsoftheAAAIcon-                      Zhang, Z., Zhang, A., Li, M., and Smola, A.  Automatic
  ference on artificial intelligence, volume 35, pp. 10293–                 chain of thought prompting in large language models.
  10301, 2021.                                                              arXiv preprint arXiv:2210.03493, 2022.

White, C., Safari, M., Sukthanker, R., Ru, B., Elsken,                   Zhao, Q., Wang, J., Zhang, Y., Jin, Y., Zhu, K., Chen, H.,
  T., Zela, A., Dey, D., and Hutter, F.  Neural architec-                   and Xie, X. Competeai: Understandingthe competition
  ture search: Insights from 1000 papers. arXiv preprint                    behaviors in large language model-based agents. arXiv
  arXiv:2301.08727, 2023.                                                   preprint arXiv:2310.17512, 2023.

Wu,Q.,Bansal,G.,Zhang,J.,Wu,Y.,Zhang,S.,Zhu,E.,Li,                       Zhong,W.,Guo,L.,Gao,Q.,Ye,H.,andWang,Y. Memo-
  B.,Jiang,L.,Zhang,X.,andWang,C.Autogen: Enabling                          rybank: Enhancinglargelanguagemodelswithlong-term
  next-gen llm applications via multi-agent conversation                    memory. In Proceedings of the AAAI Conference on Arti-
  framework, August 01, 2023 2023.                                          ficial Intelligence, volume 38, pp. 19724–19731, 2024.

Wu, X.,Shen,Y.,Shan,C.,Song, K.,Wang, S.,Zhang,B.,                       Zhou, W., Ou, Y., Ding, S., Li, L., Wu, J., Wang, T.,
  Feng,J.,Cheng,H.,Chen,W.,Xiong,Y.,etal. Cangraph                          Chen, J., Wang, S., Xu, X., Zhang, N., et al. Symbolic
  learning improve planning in llm-based agents? In The                     learning enables self-evolving agents.   arXiv preprint
  Thirty-eighthAnnualConferenceonNeuralInformation                          arXiv:2406.18532, 2024.
  Processing Systems, 2024.                                              Zhu, J.-P., Cai, P., Xu, K., Li, L., Sun, Y., Zhou, S., Su,
Xie,S.,Zheng,H.,Liu,C.,andLin,L. Snas: stochasticneu-                       H.,Tang,L.,andLiu,Q. Autotqa: Towardsautonomous
  ralarchitecture search. arXivpreprintarXiv:1812.09926,                    tabular question answering through multi-agent large lan-
  2018.                                                                     guagemodels. ProceedingsoftheVLDBEndowment,17
                                                                            (12):3920–3933, 2024a.
Yan, Z., Dai, X., Zhang, P., Tian, Y., Wu, B., and Feiszli, M.           Zhu, Y., Qiao, S., Ou, Y., Deng, S., Zhang, N., Lyu, S.,
  Fp-nas: Fast probabilisticneural architecturesearch. In                   Shen, Y., Liang, L., Gu, J., and Chen, H.  Knowagent:
  ProceedingsoftheIEEE/CVFConferenceonComputer                              Knowledge-augmented planning for llm-based agents.
  Vision and PatternRecognition, pp. 15139–15148,2021.                      arXiv preprint arXiv:2403.03101, 2024b.

Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu,                   Zhuge,M., Wang,W.,Kirsch,L., Faccio,F.,Khizbullin,D.,
  B., Li, C., Liu, D., Huang, F., Wei, H., et al. Qwen2. 5                  and Schmidhuber, J. Gptswarm: Language agents as op-
  technicalreport. arXiv preprint arXiv:2412.15115, 2024.                   timizable graphs. In Forty-first InternationalConference
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan,                 on Machine Learning, 2024.
  K. R., and Cao, Y.  React: Synergizing reasoning and                   Zoph, B.  Neural architecture search with reinforcement
  actinginlanguagemodels. InTheEleventhInternational                        learning. arXiv preprint arXiv:1611.01578, 2016.
  Conference on Learning Representations, 2023.

Yuan, S., Song, K., Chen, J., Tan, X., Li, D., and Yang,
  D.    Evoagent:   Towards  automatic  multi-agent  gen-
  eration via evolutionary algorithms.    arXiv preprint
  arXiv:2406.14228, 2024.

Zhang, G., Yue, Y., Li, Z., Yun, S., Wan, G., Wang, K.,
  Cheng, D., Yu, J. X., and Chen, T.  Cut the crap: An
  economicalcommunicationpipelineforllm-basedmulti-
  agent systems. arXiv preprint arXiv:2410.02506, 2024a.

Zhang, G., Yue, Y., Sun, X., Wan, G., Yu, M., Fang, J.,
  Wang,K.,andCheng,D. G-designer: Architectingmulti-
  agent communication topologies via graph neural net-
  works. arXiv preprint arXiv:2410.11782, 2024b.

                                                                     12

## PDF page 13

Multi-agent Architecture Search via Agentic Supernet
A. Notations


                                                            Table5. Notations and Definitions
 Notation                                            Definition
 O ={{Mi}mi=1,P,{Ti}ni=1}           An agentic operator comprising a set of LLM instances, a textual prompt, and a set of
                                                   temperature settings.
 M                                                      An individual LLM instance.
 M                                                       The set of all feasible LLMs.
 P                                                        A textual prompt used as input to the LLM.
 P                                                        The feasible space of prompts.
 T                                                         The temperature setting of the LLM.
 O                                                        The set of all feasible agentic operators.
 G ={V,E}                                        A multi-agent system represented as a graph with verticesV and edgesE.
 A ={π,O} ={πℓ(O)}O∈O}Lℓ=1     AnL-layer probabilistic agentic supernet, consisting of a distributionπ and a set of
                                                   feasible operatorsO.
 π                                                        The distribution associated with the agentic supernet.
 U(G;q,a)                                          The utility evaluator ofG with respect to queryq and answera.
 C(G;q,a)                                          The cost evaluator ofG with respect to queryq and answera.
 Qϕ                                                      The controller network parameterized byϕ.
 e(a∥G)                                               Execution ofG to produce the answera.
 Vℓ                                                       The selected operators at layerℓ of the agentic supernetA.
 Oexit                                                    The early-exit operator.
 v(·)                                                    The text embedding function.
 ∇πL                                                   The gradient of the lossL with respect to the distributionπ.
 ∇OL                                                  The textual gradient of the lossL with respect to the operatorsO.


B. Technical Details
B.1. Operator Space
In this section, we detail the initialization of operator nodes as follows:

  1.  Chain-of-Thought (CoT). CoT (Wei et al.,2022) reasoning encourages the LLM to think step by step rather than
      directlyoutputtingananswer. Thisapproachenhancesitscapabilitytosolvecomplexproblemsthroughintermediate
      reasoning steps, improving task handling and providing greater transparency in the decision-making process.

  2.  LLM-Debate. LLM-Debate (Du et al.,2023) allows multiple LLMs to debate, leveraging diverse perspectives to
      identify better solutions. In practice, we initialize three debaters and permit up to two debate rounds.

  3.  Self-Consistency. AdoptingthemethodologyfromWangetal.(2023a),thisoperatoraggregatesfiveCoTreasoning
      paths and determines the final answer through majority voting.

  4.  Self-Refine. FollowingMadaan et al.(2023), this operator initially generates an answer using CoT reasoning, then
      prompts the agent to self-reflect iteratively. We set a maximum of five refinement iterations.

  5.  Ensemble. Inspired by LLM-Blender (Jiang et al.,2023), this operator involves three LLM-powered agents from
      different sources outputtinganswers to the same query. The pairwise ranking isused to evaluate andaggregatetheir
      responses into a final solution.

  6. Testing. Followingthetestdesigner inAgentCoder(Huanget al.,2023),this operator isusedfor generatingtestcases
      for the generated code.

  7.  ReAct.  Following (Yao et al.,2023), this operator enables the agent to leverage versatile tools, including code
      interpreter, web searching, external knowledge database,etc., to handle diverse user demands.

                                                                               13

## PDF page 14

Multi-agent Architecture Search via Agentic Supernet

             8.  Earlyexit. Weintroducetheearlyexitoperator,whichinterruptsthemulti-agentarchitecturesamplingprocessand
                enables the depth of the agentic supernet to be query-dependent.

           Werespectfully note thatthe selection of theseoperators is highlycustomizable, allowingusers the flexibility to incorporate
           their desired operators into the operator repository of MaAS.

           B.2. Embedding Function
           Following established practices (Feng et al.,2024), we first employ an LLM to generate a comprehensive profile description
           for each operator. Subsequently, a lightweight text embedding model (in our case, MiniLM (Wang et al.,2020)) is used to
           encode the profile into a fixed-dimensional embedding. The prompt for generating the operator profile is as follows:
























Embedding Prompt

prompt  =  """You  are  a  highly  proficient  expert  in  designing  and  defining  operators
     for  large  language  models  (LLMs).  Your  primary  objective  is  to  meticulously
     generate  the  ‘description‘  and  ‘interface‘  fields  for  a  specified  operator  based
     on  its  provided  Python  implementation.  The  generated  content  must  be  accurate,
     efficient,  and  precisely  reflect  the  functionality  of  the  operator’s  code.

To  ensure  consistency,  quality,  and  adherence  to  best  practices,  refer  to  the
     following  examples  of  previously  defined  operators:
{
     "Generate":  {
           B.3. Textaul Gradient"description":  "Generates  anything  based  on  customized  input  and  instruction
                 .",
           The implementation of the textual gradient component is partially adapted from the repositories https://github."interface":  "generate(input:  str,  instruction:  str)  ->  dict  with  key  ’
           com/ShengranHu/ADAS/ andhttps://github.com/tsinghua-fib-lab/agentsquare. We would likeresponse’  of  type  str"
     },    to explicitly acknowledge this contribution and express our sincere gratitude to the authors for their open-source efforts.
     "ScEnsemble":  {
           "description":  "Uses  self-consistency  to  select  the  solution  that  appears
                 most  frequently  in  the  solution  list,  improving  the  selection  to  enhance
                 the  choice  of  the  best  solution.",
           "interface":  "sc_ensemble(solutions:  List[str],  problem:  str)  ->  dict  with
                 key  ’response’  of  type  str"
     }
}

Now,  given  the  following  operator  code.  This  code  encompasses  the  function  signature
Textual Gradient,  parameters  with  type  annotations,  internal  logic,  and  return  statements
base  =  """essential  for  comprehensively  understanding  the  operator’s  purpose  and  behavior.14
#  OverviewPlease  provide  its  ‘description‘  and  ‘interface‘  fields  in  the  same  format.
[operator  code]
You  are  an  expert  machine  learning  researcher  specializing  in  designing  agentic
 """systems.  Your  objective  is  to  create  building  blocks  such  as  prompts  and  control
     flows  within  these  systems  to  solve  complex  tasks.  Specifically,  you  aim  to

## PDF page 15

Multi-agent Architecture Search via Agentic Supernet











     design  an  optimal  agent  that  performs  exceptionally  on  the  HumanEval  benchmark.
     The  HumanEval  dataset  evaluates  code  generation  capabilities  in  AI  systems,
     consisting  of  164  hand-crafted  Python  programming  problems.  Each  problem  includes
     :  -  A  function  signature  with  a  docstring  describing  the  task  -  Test  cases  to
     verify  functional  correctness

#  Example  Question  from  HumanEval
[An  example  question  from  HumanEval  dataset  here]

#  Operator  code  template:
class  Operator:
     def  __init__(self,  llm:  LLM,  name:  str):
           self.name  =  name
           self.llm  =  llm

     def  __call__(self,  *args,  **kwargs):
           raise  NotImplementedError

     async  def  _fill_node(self,  op_class,  prompt,  mode=None,  **extra_kwargs):
           fill_kwargs  =  {"context":  prompt,  "llm":  self.llm}
           if  mode:
                fill_kwargs["mode"]  =  mode
           fill_kwargs.update(extra_kwargs)
           node  =  await  ActionNode.from_pydantic(op_class).fill(**fill_kwargs)
           return  node.instruct_content.model_dump()

class  GenerateOp(BaseModel):
     response:  str  =  Field(default="",  description="Your  solution  for  this  problem")

class  Generate(Operator):
     GENERATE_PROMPT  =  ’’’
You  are  tasked  with  solving  the  following  Python  programming  problem.  Generate  a
     complete,  syntactically  correct  Python  function  that  strictly  adheres  to  the
     given  requirements.

Problem:
{input}

Follow  these  steps:
1.  Analyze  the  problem  requirements  and  identify  edge  cases
2.  Design  a  solution  that  passes  all  implied  test  cases
3.  Implement  the  function  with  clear  variable  names  and  comments

Ensure:
-  The  code  directly  implements  the  requested  functionality
-  All  parameters  and  return  types  match  the  problem  specification
-  Exception  handling  for  edge  cases  is  included  when  necessary  ’’’

     def  __init__(self,  llm:  LLM,  name:  str  =  "Generate"):
           super().__init__(llm,  name)

     async  def  __call__(self,  input:  str,  mode:  str  =  None):
           prompt  =  self.GENERATE_PROMPT.format(input=input)
           response  =  await  self._fill_node(GenerateOp,  prompt,  mode="xml_fill")
           return  response

#  Discovered  architecture  archive
Here  is  the  archive  of  the  discovered  operator  architectures:
[ARCHIVE]                                                              15

#  Output  Instruction  and  Example:
The  output  should  be  a  JSON  object  with  the  following  structure.The  first  key  should
      be  ("thought"),  and  it  should  capture  your  thought  process  for  designing  the

## PDF page 16

Multi-agent Architecture Search via Agentic Supernet






















     next  operator.  The  second  key  ("description")  corresponds  to  the  brief
     description  of  your  next  operator.  Finally,  the  last  key  ("code")  corresponds  to
     the  exact  operator  and  its  prompt  in  Python  code  that  you  would  like  to  try.  You
     must  write  COMPLETE  CODE  in  "code":  Your  code  will  be  part  of  the  entire  project,
      so  please  implement  complete,  reliable,  reusable  code  snippets.

-  thought:  Captures  your  thought  process  for  designing  the  next  operator.
  -  Reason  about  what  the  next  interesting  operator  should  be.
  -  Describe  your  reasoning  and  the  overall  concept  behind  the  operator  design.
  -  Detail  the  implementation  steps.
-  description:  A  brief  description  of  your  next  operator.
-  code:  The  exact  operator  and  its  prompt  in  Python  code.  Ensure  the  code  is
     complete,  reliable,  and  reusable.


Here  is  an  example  of  the  output  format  for  the  next  operator:
[operator_example]

You  must  strictly  follow  the  exact  input/output  interface  used  above.  Also,  it  could
      be  helpful  to  set  the  LLM’s  role  and  temperature  to  further  control  the  LLM’s
     response.  DON’T  try  to  use  some  function  that  doesn’t  exist.  In  __call__(),  you
     need  to  specify  the  instruction,  input  information,  the  prompt  and  the  required
     output  fields  class  for  operators  to  do  their  specific  part  of  the  architecture.

#  Your  task
You  are  highly  proficient  in  prompting  techniques  and  well-versed  with  agentic
     systems  from  academic  literature.  Your  goal  is  to  maximize  performance  metrics  by
      proposing  innovative  and  effective  new  operators.
Instructions:
1.  Analyze  the  Discovered  Operators:  Carefully  review  the  operators  in  the  archive
     to  identify  strengths,  weaknesses,  and  areas  for  improvement.
2.  Draw  Insights:  Extract  lessons  and  insights  from  existing  operators  to  inform  the
      design  of  the  next  operator.
3.  Innovate:  Think  creatively  to  design  an  operator  that  addresses  current
     limitations  or  explores  new  functionalities,  drawing  inspiration  from  related
     agent  papers  or  other  research  areas.
4.  Design  the  Operator:  Propose  the  next  operator’s  ‘thought‘,  ‘description‘,  and  ‘
     code‘  following  the  specified  format.C. Experimental Details
5.  Ensure  Completeness:  The  generated  code  must  be  complete,  reliable,  and  reusable,
      fitting  seamlessly  into  the  existing  architecture.C.1. Dataset Statistics

Execution  Steps:Building upon established methodologies in workflow automation (Saad-Falcon et al.,2024;Hu et al.,2024a;Zhang et al.,
1.  Insert  Operator  Code:  Replace  the  ‘[ARCHIVE]‘  and  ‘[operator_example]‘2024c), we divide each dataset into training and test sets using a              TRAIN:TEST ratio of 1:4. For the MATH benchmark,
     placeholders  with  actual  content  as  needed.
2.  Generate  Output:  Produce  the  ‘thought‘,  ‘description‘,  and  ‘code‘  fields  for  thewe adhere to (Hong et al.,2024), selecting a subset of 617 harder problems spanning four representative categories,
     new  operator,  ensuring  adherence  to  the  guidelines.Combinatorics&Probability,NumberTheory,Pre-algebra,andPre-calculus,allatdifficultylevel5. Thedatasetstatistics
3.  Validate  Output:  Ensure  the  generated  JSON  is  correctly  formatted  and  the  code  is
      syntactically  and  functionally  correct.                          16

THINK  OUTSIDE  THE  BOX  and  leverage  interdisciplinary  insights  to  enhance  the  agentic
      system’s  capabilities.
"""

## PDF page 17

Multi-agent Architecture Search via Agentic Supernet

are included in Table6.

                                                         Table6. Dataset Statistics.
                                  Domain                     Dataset        #Train    #Test       Metric
                                  Code Generation         HumanEval        33         131       pass@1
                                                             MBPP            86         341       pass@1
                                                            GSM8K         264       1055     Accuracy
                                  Math Reasoning             MATH          119        486      Accuracy
                                                           MultiArith       150        600      Accuracy
                                  Tool use                   GAIA            94         372      Accuracy


C.2. Baseline Setups
In this section, we provide a detailed description of the configurations for baseline methods:

  1.  CoT.Chain-of-Thought(CoT)promptingguidesLLMagentstobreakdownreasoningintosequentialstepsratherthan
     generating direct answers. We employ the implementation from (Zhang et al.,2022).

  2.  ComplexCoT.  We  follow  the  official  implementation  available  at https://github.com/FranxYao/
     Complexity-Based-Prompting/tree/main.

  3.  Self-consistency. To enhance robustness, we aggregate five CoT-generated solutions.

  4.  LLM-Debate. WeinstantiatefiveLLM-agents,eachassignedadistinctrole,whichparticipateinuptotworounds
     ofdebate,afterwhichthefinaldecisionisdeterminedviamajorityvoting. Theimplementationisbasedonhttps:
     //github.com/ucl-dark/llm_debate.

  5.  LLM-Blender.  We choose two gpt-4o-mini, one Qwen-2.5-72b, and one llama-3.1-70b to empower
     LLM-Blender (Jiang et al.,2023).

  6.  DyLAN.We directly utilize the implementation from (Liu et al.,2023).
  7. AgentVerse. The experimental setup follows the original implementation from (Chen et al.,2023b).

  8.  MacNet.  For MacNet (Qian et al.,2024), we adopt the “MacNet-MESH” variant, which corresponds to a fully
     connected network topology.

  9.  GPTSwarm. The method is implemented in accordance with the original settings described in (Zhuge et al.,2024).

10. AutoAgents. We adhere to the official configuration specified in (Chen et al.,2023a).

11. ADAS. The implementation details are directly inherited from (Hu et al.,2024a).

12. AgentSquare. We utilize the modular search framework introduced in (Shang et al.,2024). The base LLM remains
     fixed atgpt-4o-mini, with early stopping set to a patience of 5 iterations.

13. AFlow.  In (Zhang et al.,2024c), AFlow operates with both              gpt-4o-mini and claude-3.5-sonnet.  To
     maintainfairnessunder homogeneousconditions,werestrict AFlow togpt-4o-mini andset MAX                                  ITERATION=20.

D. Supplementary Results
We have visualized theevolutionof operator sampling trends asthe sampling count increases, in Figure10. MaAS learnsto
avoid overly confident early stopping and instead prioritizes testing and self-refinement in deeper layers.





                                                                      17

## PDF page 18

Multi-agent Architecture Search via Agentic Supernet


Table7.Cross-model transferability of MaAS. We optimize the agentic supernet withgpt-4o-mini, and report the performances
before and after equipping the LLM backbones with the optimized agentic supernet.
                               Dataset                                         HumanEval
                          LLM Backbone            gpt-4o-mini   Qwen-2.5-72b   llama-3.1-70b
                                vanilla                 87.08                     85.60                       80.06
                               +MaAS                    92.85                     90.14                       85.26
                               Dataset                                            MATH
                          LLM Backbone            gpt-4o-mini   Qwen-2.5-70b   llama-3.1-70b
                                vanilla                 46.29                     63.80                       31.93
                               +MaAS                    51.82                     69.35                       42.97


Table8.Cross-datasettransferabilityofMaAS.“MATH→GSM8K”denotesoptimizingtheagenticsupernetonMATHandevaluatingit
on GSM8K, with similar notation applied to other cases.
                            Transfer        MATH→GSM8K    GSM8K→MATH    HumanEval→MATH
                          GPTSwarm                  89.96                      45.18                         47.92
                             AFlow                  91.95                      49.39                         47.15
                             MaAS                   92.80                      51.02                         50.27










































                   Figure8. The layer-wise distribution of MaAS on HumanEval benchmark without Debate operator.

                                                                          18

## PDF page 19

Multi-agent Architecture Search via Agentic Supernet
























Figure9.Thelayer-wisedistributionofMaASonHumanEvalbenchmarkwithDebateoperator. Notethattheagenticsupernetisoptimized
withotheroperators,whiletheDebateoperatorisintroducedonlyduringtheinferencestage. Itcanbeobservedthat,despitenotbeing
exposed to this operator during training, MaAS can still reasonably select it during the multi-agent architecture sampling process.









































                            Figure10. The evolution of operator sampling trends as the sampling count increases.




                                                                            19
