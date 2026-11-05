<!-- Source: https://arxiv.org/html/2512.14079v1; retrieved: 2026-09-28T19:18:06.590182+00:00; conversion: arXiv HTML to Markdown via lxml; original article text follows. -->

# Grammar Search for Multi-Agent Systems

## Abstract

Automatic search for Multi-Agent Systems has recently emerged as a key focus in agentic AI research. Several prior approaches have relied on LLM-based free-form search over the code space. In this work, we propose a more structured framework that explores the same space through a fixed set of simple, composable components. We show that, despite lacking the generative flexibility of LLMs during the candidate generation stage, our method outperforms prior approaches on four out of five benchmarks across two domains: mathematics and question answering. Furthermore, our method offers additional advantages, including a more cost-efficient search process and the generation of modular, interpretable multi-agent systems with simpler logic.

## 1 Introduction

Large language models (LLMs) have emerged as powerful general-purpose reasoning systems capable of solving diverse tasks across language understanding, reasoning, and decision-making [Comanici et al. (2025)](https://arxiv.org/html/2512.14079v1#bib.bib25); [OpenAI (2025)](https://arxiv.org/html/2512.14079v1#bib.bib19). Increasingly, LLMs are deployed as autonomous agents for real-world applications such as coding [Yang et al. (2024)](https://arxiv.org/html/2512.14079v1#bib.bib24), tool use [Schick et al. (2023)](https://arxiv.org/html/2512.14079v1#bib.bib26), and computer interaction [He et al. (2024)](https://arxiv.org/html/2512.14079v1#bib.bib27). This shift has inspired a growing line of work that organizes multiple LLMs into multi-agent systems (MASes), where specialized agents collaborate, exchange information, and refine each other’s outputs to tackle complex problems [Du et al. (2023)](https://arxiv.org/html/2512.14079v1#bib.bib11); [Madaan et al. (2023)](https://arxiv.org/html/2512.14079v1#bib.bib10); [Yao et al. (2023)](https://arxiv.org/html/2512.14079v1#bib.bib12). This paradigm reframes LLMs from being standalone solvers to components in a coordinated reasoning framework.

Early MASes were manually constructed, with researchers designing agent roles, communication patterns, and reasoning strategies by hand [Li et al. (2023)](https://arxiv.org/html/2512.14079v1#bib.bib33); [Park et al. (2023)](https://arxiv.org/html/2512.14079v1#bib.bib34). While effective at small scales, such manual design is labor-intensive and does not generalize across tasks [Wang et al. (2024a)](https://arxiv.org/html/2512.14079v1#bib.bib35). To overcome these limitations, recent work has focused on automatic MAS search, where the goal is to discover effective MASes with minimal human input [Zhuge et al. (2024)](https://arxiv.org/html/2512.14079v1#bib.bib13); [Zhang et al. (2025a)](https://arxiv.org/html/2512.14079v1#bib.bib14); [Yue et al. (2025)](https://arxiv.org/html/2512.14079v1#bib.bib15). These approaches define a search space of MASes and use optimization or exploration algorithms to navigate it.

Figure: Figure 1: Overview of our grammar-based approach to generate MASes. We search for multi-agent systems in code space without relying on LLMs for code generation. The grammar is general-purpose and specifies constraints on how to combine components; considering additional base MASes only requires identifying their components. The search process generates candidate MASes; the best one is selected based on validation accuracy.

Despite their progress, existing automatic methods face key challenges. Many methods such as ADAS [Hu et al. (2025)](https://arxiv.org/html/2512.14079v1#bib.bib7) and AFlow [Zhang et al. (2025b)](https://arxiv.org/html/2512.14079v1#bib.bib6) operate directly in code space, which, although expressive, often leads to syntactically invalid or semantically inconsistent programs that waste computational resources. In our analysis, ADAS generated invalid candidate MASes in more than 20% of search attempts. Furthermore, automatically discovered MASes can become unnecessarily complex, with large numbers of interconnected components that are difficult to interpret or extend. These issues limit the efficiency, transparency, and usability of MAS search.

To address these challenges, we introduce Grammar Search, a general framework for designing and discovering MASes that emphasizes modularity, simplicity, and correctness. The central idea is to represent MASes using a context-free grammar that defines how individual components can be assembled into larger systems. These components can be manually designed, obtained by decomposing existing MASes into smaller parts, or suggested by an LLM. The resulting components are constrained by our grammar, which encodes rules specifying how they can be connected. Sampling MASes from this grammar guarantees syntactic correctness and enables a structured search for novel MASes. The grammar is fully extensible and not restricted to the initial set of components. New components can be added as needed, making the framework adaptable to new tasks and domains.

By defining a structured space of valid MASes, Grammar Search effectively eliminates the need for LLM-based code generation during search, thereby preventing invalid programs and simplifying subsequent experimentation. Although the components are simple, the discovered MASes achieve competitive or superior performance compared to previous automatic methods across multiple reasoning benchmarks. At the same time, the search requires less computational effort, and the resulting MASes remain more interpretable.

In summary, our contributions are as follows:11 1 Code at: [https://github.com/mayanks43/grammar_search](https://github.com/mayanks43/grammar_search). 1. We propose Grammar Search, a general framework for MAS discovery based on a context-free grammar that ensures modularity, correctness, and extensibility. 2. We demonstrate that our framework outperforms existing automatic search methods on four out of five benchmarks across two domains: mathematics and question answering. 3. Beyond accuracy, our framework provides a cheaper search stage and interpretable, simple, yet effective MASes.

## 2 Preliminaries

We begin by defining a few key terms used throughout this work.

#### Agents

Following [Plaat et al. (2025)](https://arxiv.org/html/2512.14079v1#bib.bib28), we define agents as large language models (LLMs) that have been adapted for specific tasks through prompting or post-training, and that can reason through intermediate steps toward goals, act by executing operations or invoking external tools, and interact with users, environments, or other agents through communication channels such as web search, APIs, or memory modules.

#### Multi-Agent Systems

A Multi-Agent System (MAS) is a collection of agents coordinating toward a shared goal. MASes are commonly represented as graphs [Zhuge et al. (2024)](https://arxiv.org/html/2512.14079v1#bib.bib13), where nodes represent individual agents and edges represent channels of communication. Alternatively, MASes can be represented through executable code [Hu et al. (2025)](https://arxiv.org/html/2512.14079v1#bib.bib7), where the structure of function calls and data flow explicitly defines the MAS.

#### MAS Search Space

MASes span a large combinatorial search space that can vary along multiple axes, including the types of nodes, their interconnections, and the objectives used to evaluate them. We outline these axes at a conceptual level and provide examples from prior work in Section [3](https://arxiv.org/html/2512.14079v1#S3).

## 3 Related Work

#### Manually-designed MASes

Most initial MASes were manually designed. Here, we describe some popular ones. Chain of Thought (CoT) [Wei et al. (2023)](https://arxiv.org/html/2512.14079v1#bib.bib3) is often considered a trivial MAS, consisting of a single node with a prompt to solve the task step by step. Chain of Thought with Self-Consistency (CoT-SC) [Wang et al. (2023)](https://arxiv.org/html/2512.14079v1#bib.bib9) builds on CoT by using multiple parallel CoT nodes, followed by a majority-voting node that selects the most common answer among parallel outputs. Self-Refine [Madaan et al. (2023)](https://arxiv.org/html/2512.14079v1#bib.bib10) uses the same model to iteratively refine an initial answer based on self-generated feedback. In Multi-Agent Debate [Du et al. (2023)](https://arxiv.org/html/2512.14079v1#bib.bib11), multiple agents debate over each other’s answers, followed by a final decision agent that aggregates outputs from these agents. Tree-of-Thoughts [Yao et al. (2023)](https://arxiv.org/html/2512.14079v1#bib.bib12) solves problems by exploring a tree of possible next steps, evaluating each branch, and backtracking from mistakes to find the optimal solution. Instead of manually designing MASes, in this work we focus on automatically searching for them given the task. We use the first four MASes as baselines for comparison, as well as a basis for our MAS grammar.

#### Automatic MAS Search

Lately, there has been a plethora of methods for automatically searching for the best MASes. These methods differ in how they represent, evaluate, and search for the best MASes:

Different works have adopted different ways to represent MASes. Graph-based approaches such as GPTSwarm [Zhuge et al. (2024)](https://arxiv.org/html/2512.14079v1#bib.bib13) and MaAS [Zhang et al. (2025a)](https://arxiv.org/html/2512.14079v1#bib.bib14) represent agents as nodes and communication links as edges, allowing structural optimization of the graph. In contrast, code-based approaches such as ADAS [Hu et al. (2025)](https://arxiv.org/html/2512.14079v1#bib.bib7) and AFlow [Zhang et al. (2025b)](https://arxiv.org/html/2512.14079v1#bib.bib6) represent MASes as executable Python programs, where nodes correspond to API calls that can be connected in complex ways. This representation allows fine-grained logic such as branching, iteration, and function composition. Our method follows this latter direction, using a grammar to define composable code components that can be systematically explored.

Each node in a MAS can be optimized along multiple dimensions. Prompts can be refined or evolved through techniques such as PromptBreeder [Fernando et al. (2023)](https://arxiv.org/html/2512.14079v1#bib.bib32), or modified to encode distinct roles or personas for different agents [Du et al. (2023)](https://arxiv.org/html/2512.14079v1#bib.bib11). LLM parameters can also be adapted through post-training, for example via fine-tuning as in Sirius [Zhao et al. (2025)](https://arxiv.org/html/2512.14079v1#bib.bib31). In addition, new tools can be dynamically created when existing ones are insufficient for a task [Qiu et al. (2025)](https://arxiv.org/html/2512.14079v1#bib.bib30).

MAS search can occur at different levels of granularity. Corpus-level search methods such as ADAS, AFlow, and GPTSwarm discover one MAS that performs well across a dataset, while query-level methods such as MaAS, FlowReasoner [Gao et al. (2025)](https://arxiv.org/html/2512.14079v1#bib.bib16), and MasRouter [Yue et al. (2025)](https://arxiv.org/html/2512.14079v1#bib.bib15) generate or adapt a new MAS for each input query. Corpus-level methods are generally more efficient and stable, whereas query-level methods allow greater specialization. To maintain efficiency, we focus on corpus-level search in this work.

Search methods differ in their optimization objectives. Common objectives include benchmark accuracy, API call cost, latency, and safety considerations. Many recent approaches such as AFlow and MAS-ZERO [Ke et al. (2025b)](https://arxiv.org/html/2512.14079v1#bib.bib29) adopt multi-objective optimization to balance these factors. In this work, we focus on benchmark accuracy and API call cost to discover MASes that achieve high accuracy while maintaining low search cost.

Candidate MAS evaluation strategies also vary across works. GPTSwarm and ADAS select candidates based on validation set performance, while MAS-ZERO uses prompting, i.e., asking an LLM to choose the better MAS based on observed behavior. In this work, we evaluate candidate MASes using validation performance, as it provides a more consistent and reliable measure of quality.

Several strategies have been explored to traverse the MAS space. Iterative search (ADAS, MAS-ZERO) repeatedly asks an LLM to propose, refine, and build upon existing candidates. Reinforcement learning approaches such as GPTSwarm perform search and learning in parallel. Tree search variants such as Monte Carlo Tree Search (AFlow) have also been explored for structured exploration of MAS graphs. We adopt a random search strategy that explores the MAS space component by component.

Figure: Figure 2: Our general-purpose context-free grammar to represent MASes. Note: SI = SingleInput, SO = SingleOutput, MI = MultiInput, MO = MultiOutput, cnt = count, and rnds = rounds.

#### Summary and Relation to This Work

Across methods described in this section, the core challenge is the same: the MAS search space is vast. Our work differs in that it introduces a context-free grammar to define and constrain this space. This provides a principled way to systematically enumerate, compare, and analyze MASes while retaining the flexibility of code-based representations.

## 4 A Grammar for MASes

In this section, we describe our framework in detail. A context-free grammar serves as the central element of our approach. This grammar defines the structured design space that the framework explores when searching for valid MASes. We define a general-purpose grammar, shown in Figure [2](https://arxiv.org/html/2512.14079v1#S3.F2), to enable systematic composition of MAS components. The MAS components appear in the figure as the terminals of the grammar. To obtain these components, we manually decompose several existing MASes into smaller subcomponents. We will refer to these MASes as *base* MASes to contrast them with the automatically discovered MASes obtained through search. By *manually decompose*, we mean that we examine the code implementations of these MASes and identify sections that can be encapsulated as stand-alone components. More specifically, we decompose four manually designed base MASes into smaller components: CoT, CoT-SC, Self-Refine, and Multi-Agent Debate. However, this decomposition is not limited to these four MASes and can be extended to any group of MASes that differ substantially.

Table: Table 1: The four base MASes we work with, their component sequences, and their derivations with our grammar. We condense the names of some components to save space. Component sequence depicts how the respective MAS is represented in our grammar. Derivation shows how to derive the MAS through our grammar rules.

| MAS | Component Sequence | Grammar Derivation |
| --- | --- | --- |
| CoT | StepByStep[1] | $\begin{array}[]{@{}l@{}}\langle\text{System}\rangle\to\langle\text{StartSI}\rangle\\ \langle\text{StartSI}\rangle\to\langle\text{StartSISO}\rangle\\ \langle\text{StartSISO}\rangle\to\texttt{StepByStep[1]}\end{array}$ |
| CoT-SC | StepByStep[5] $\Rightarrow$ MajorityVoter | $\begin{array}[]{@{}l@{}}\langle\text{System}\rangle\to\langle\text{StartSI}\rangle\\ \langle\text{StartSI}\rangle\to\langle\text{SIMO}\rangle\ \langle\text{MI}\rangle\\ \langle\text{SIMO}\rangle\to\texttt{StepByStep[5]}\\ \langle\text{MI}\rangle\to\langle\text{MISO}\rangle\\ \langle\text{MISO}\rangle\to\texttt{MajorityVoter}\end{array}$ |
| Self-Refine | StepByStep[1] $\Rightarrow$ SelfCritic[5] | $\begin{array}[]{@{}l@{}}\langle\text{System}\rangle\to\langle\text{StartSI}\rangle\\ \langle\text{StartSI}\rangle\to\langle\text{StartSISO}\rangle\ \langle\text{SI}\rangle\\ \langle\text{StartSISO}\rangle\to\texttt{StepByStep[1]}\\ \langle\text{SI}\rangle\to\langle\text{SISO}\rangle\\ \langle\text{SISO}\rangle\to\texttt{SelfCritic[5]}\end{array}$ |
| MA-Debate | RoleBased[5] $\Rightarrow$ Debate[2] $\Rightarrow$ Consensus | $\begin{array}[]{@{}l@{}}\langle\text{System}\rangle\to\langle\text{StartSI}\rangle\\ \langle\text{StartSI}\rangle\to\langle\text{SIMO}\rangle\ \langle\text{MI}\rangle\\ \langle\text{SIMO}\rangle\to\texttt{RoleBased[5]}\\ \langle\text{MI}\rangle\to\langle\text{MIMO}\rangle\ \langle\text{MI}\rangle\\ \langle\text{MIMO}\rangle\to\texttt{Debate[2]}\\ \langle\text{MI}\rangle\to\langle\text{MISO}\rangle\\ \langle\text{MISO}\rangle\to\texttt{Consensus}\end{array}$ |

The primary goal of our grammar is to enable the exploration of new MASes composed of the smaller components derived from this decomposition. It is designed to be general, allowing new components to be added as needed. We now describe the grammar in detail.

#### Grammar Non-terminals

The core of our grammar consists of *non-terminals*, defined solely by their input-output structure. We distinguish four categories: SISO (Single Input, Single Output), SIMO (Single Input, Multiple Outputs), MISO (Multiple Inputs, Single Output), and MIMO (Multiple Inputs, Multiple Outputs), each specifying the number of inputs and outputs they allow.

As might be evident, our grammar generally enforces that input and output counts match between components when chained together. Some components, such as SelfCriticIteration (which refines a provided answer), cannot serve as the first component and therefore require additional logic.

Another constraint that enables seamless composition is that each component must return either a single complete answer or multiple complete answers. Subsequent components either build directly on the previous answers or generate their own, using the prior answers as a side input. This simple formulation allows any component of any MAS to naturally integrate into our grammar.

#### Grammar Terminals

The terminals in our grammar correspond to concrete components that can be combined to construct new MASes. We now describe the major components obtained from decomposing the four basic MASes mentioned earlier. However, our grammar is designed to be extensible and can easily accommodate additional components beyond these.

This is the Chain-of-Thought component. Given a task, the LLM is asked to answer step by step. The count parameter allows the generation of multiple answers in parallel. This component may behave as either SISO or SIMO depending on the number of answers generated.

This component functions similarly to StepByStepReasoner, but with the added instruction that the LLM should assume a specific role or persona. For example, it may be asked to take on the role of a math professor when solving math problems. The count parameter again enables the generation of multiple parallel answers. Depending on the value of count, this component may be either SISO or SIMO.

This component consists of a critic agent followed by a reflect agent, connected in sequence and repeated over a number of rounds as specified by the rounds parameter. The critic agent critiques a response it receives from a previous component, and the reflect agent uses the critic’s feedback to generate a better answer. This is strictly a SISO component.

This is a MIMO component that requires input from a SIMO or another MIMO component. It performs iterative debates among multiple agents using the input answers it receives. The rounds parameter specifies how many iterations the debate process undergoes.

This is a MIMO version of SelfCriticIteration. It applies the critic-and-reflect sequence in parallel across multiple input answer streams.

This is a MISO component that condenses multiple answer streams into a single answer. As the name suggests, it selects the most common answer among those provided.

This is another MISO component. Unlike MajorityVoter, it does not select an existing answer. Instead, the component synthesizes a new final answer based on the multiple answers it receives as input.

Note that MultiSelfCriticIteration is not a component of the base MASes. It was added as a representative example to demonstrate how easily new components can be integrated into the grammar.

Table [1](https://arxiv.org/html/2512.14079v1#S4.T1) shows representative sequences in our grammar corresponding to the four initial MASes, along with their derivations.

#### MASes from component sequences

Each terminal in our grammar corresponds to a code fragment. These fragments follow a simple input-output interface: they take the task details and a list of answers from the previous component as input and return a list of answers as output. The only exception is the first component, which takes the task details as input. For single-input components, the input list contains a single answer; likewise, the output list contains one answer for single-output components.

To construct a MAS corresponding to a component sequence, the associated code fragments are stitched together, with inputs and outputs passed along the sequence. Each sequence therefore corresponds to a deterministic program representing a MAS. Example sequences and their corresponding MASes are provided in Appendix [D](https://arxiv.org/html/2512.14079v1#A4).

## 5 Searching Over MAS Grammar

We use the grammar defined above to sample new component sequences and select the best-performing corresponding MAS. By randomly sampling a sequence, we mean starting from the start symbol of the grammar and repeatedly choosing production rules at random until a terminal-only sequence is obtained.

Our general algorithm proceeds as follows:

1. Randomly sample a component sequence. 2. If the sequence has not been seen before, evaluate the corresponding MAS on the validation set.

After a fixed number of iterations, we select the best MAS based on the performance scores of the sampled MASes on the validation set.

However, we find that pure random sampling often fails to include components that are buried deeper in the grammar. To ensure that all components are fairly represented, we adopt a forced sampling strategy.

In forced sampling, we repeatedly sample sequences until one containing a specific target component is found, and then evaluate the resulting MAS on the validation set. We depict the full algorithm in Appendix [B](https://arxiv.org/html/2512.14079v1#A2) but also summarize it here: 1. Maintain a count of how many times each component has been sampled. 2. Cluster components by their sampling counts. 3. Begin with the lowest-count cluster (initially 0) and force-sample sequences containing each component in the cluster, followed by evaluation of the resulting MAS. 4. Exhaust all components in the lower-count clusters before moving on to higher ones.

Over time, if a component falls behind in its sampling frequency, it will move into the lowest-count cluster and be force-sampled accordingly. This mechanism ensures that all components in the grammar are sampled and evaluated in a fair and balanced manner.

We also explored more sophisticated search methods, such as Thompson Sampling [Thompson (1933)](https://arxiv.org/html/2512.14079v1#bib.bib23) and Monte Carlo Tree Search [Kocsis and Szepesvári (2006)](https://arxiv.org/html/2512.14079v1#bib.bib22), to improve the discovery of the best component sequence. However, we did not observe any performance gains from their use and therefore chose to retain our simpler forced sampling strategy.

## 6 Evaluation

Table: Table 2: Results (accuracy and standard deviation across all runs) obtained with manually-designed MASes (top block) and three automatic MAS search methods (bottom block). Grammar Search, our approach, outperforms previous MAS search methods on average and in all but one benchmark. Additionally, Grammar Search never generates an invalid MAS and it is more efficient (Section [6](https://arxiv.org/html/2512.14079v1#S6)).

|  | Math | Other Domains | Avg. |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
|  | MATH500 | MATH lvl 5 | AIME | GPQA-D | MMLU-Pro |  |
| Manually-Designed MASes |  |  |  |  |  |  |
| Chain of Thought | 77.7 ± 0.8 | 61.7 ± 1.7 | 7.6 ± 1.2 | 43.2 ± 2.5 | 65.9 ± 1.4 | 51.2 |
| CoT-Self Consistency | 79.3 ± 0.7 | 64.2 ± 1.3 | 8.5 ± 1.6 | 43.7 ± 2.6 | 67.1 ± 0.8 | 52.6 |
| Self-Refine | 78.1 ± 1.1 | 62.8 ± 1.2 | 8.2 ± 1.7 | 43.3 ± 2.2 | 66.2 ± 1.1 | 51.7 |
| Multi-Agent Debate | 79.8 ± 0.9 | 65.8 ± 1.1 | 10.4 ± 1.7 | 44.6 ± 2.5 | 67.4 ± 1.0 | 53.6 |
| Automatic MAS Search Methods |  |  |  |  |  |  |
| ADAS | 80.9 ± 0.9 | 66.6 ± 1.4 | 8.7 ± 1.3 | 46.2 ± 2.3 | 68.5 ± 1.2 | 54.2 |
| AFlow | 80.0 ± 1.0 | 65.7 ± 1.0 | 8.6 ± 1.5 | 42.8 ± 2.9 | 65.4 ± 0.8 | 52.5 |
| Grammar search (Ours) | 81.4 ± 0.5 | 67.2 ± 1.6 | 11.1 ± 1.3 | 46.6 ± 2.0 | 68.4 ± 0.7 | 55.0 |

In this section, we discuss the evaluation setup and results for our method on various datasets.

#### Datasets

We evaluate our method on four datasets: two from mathematics and two from question answering. We describe them below:

MATH [Hendrycks et al. (2021b)](https://arxiv.org/html/2512.14079v1#bib.bib1): This dataset contains 12,000 competition-style math problems. For the search phase, we use 160 random problems from the training set as our validation set. For testing, we evaluate on two subsets: the 500-problem partition known as MATH500, and a separate partition of 486 level-5 problems, following the setup in AFlow.

AIME [Veeraboina (2023)](https://arxiv.org/html/2512.14079v1#bib.bib21): This dataset consists of questions from the American Invitational Mathematics Examination and is more challenging than the MATH dataset. We use 160 random problems from the 1983–2019 set for validation. For testing, we use 133 problems from the years 2020–2024. We avoid the more common 30-problem split from the most recent year to ensure greater statistical power during evaluation [Hochlehnert et al. (2025)](https://arxiv.org/html/2512.14079v1#bib.bib20).

MMLU-Pro [Wang et al. (2024b)](https://arxiv.org/html/2512.14079v1#bib.bib5): MMLU-Pro is a more difficult variant of the MMLU benchmark [Hendrycks et al. (2021a)](https://arxiv.org/html/2512.14079v1#bib.bib8), a general-domain QA dataset. Unlike the original MMLU, which provides 4 answer choices per question, MMLU-Pro includes 10 options per question. Because the provided validation set contains only 70 problems, we construct larger validation and test sets by splitting the full test set into two parts: one for validation and one for testing. From the validation split, we randomly select 160 problems for the validation set, and from the test split, we randomly select 500 problems for the test set. Some problems had to be dropped due to content policy triggers from our LLM API provider. As a result, the final validation and test sets contain 158 and 494 problems, respectively. To the best of our knowledge, we are the first to evaluate automatic MAS search methods on this dataset.

GPQA [Rein et al. (2023)](https://arxiv.org/html/2512.14079v1#bib.bib2): The Google-Proof Question Answering dataset (GPQA) contains expert-authored questions across multiple disciplines, designed to resist shallow, web-search-based answering. As our test set, we use the hard variant GPQA-Diamond, which contains 198 questions. To construct the validation set, we subtract the 198 Diamond problems from the main GPQA set, yielding 250 remaining problems. From these, we randomly sample 160 for the validation set.

#### Baselines

As baselines, we use the four base MASes we have been working with: CoT, CoT-SC, Self-Refine, and Multi-Agent Debate. In addition, we compare our method against two leading corpus-level automatic MAS search methods: AFlow and ADAS. We run ADAS for 30 iterations and AFlow for 20 iterations, following the default settings in their respective codebases. Similar to ADAS, we run our search algorithm for 30 iterations on all evaluated datasets.

#### Evaluation Metric

The evaluation metric for all datasets is accuracy. Following recent works that employ LLMs for answer equivalence checking ([Ho et al., 2025](https://arxiv.org/html/2512.14079v1#bib.bib17); [Ke et al., 2025a](https://arxiv.org/html/2512.14079v1#bib.bib18)), we consider a prediction correct if it matches the ground truth exactly or is judged equivalent by a strong LLM (gpt-5; [OpenAI 2025](https://arxiv.org/html/2512.14079v1#bib.bib19)) when a direct equality check fails.

For all datasets, following [Zhang et al. (2025b)](https://arxiv.org/html/2512.14079v1#bib.bib6), we evaluate each problem five times on the validation set during the search phase and use the average accuracy to guide the search for the best MAS. In the test phase, we evaluate the best MAS found during the search phase eight times on the test sets of MATH and MMLU-Pro and report the average accuracy and standard deviation. Because the GPQA and AIME test sets are smaller, we evaluate them 20 and 30 times, respectively, following the recommendations of [Hochlehnert et al. (2025)](https://arxiv.org/html/2512.14079v1#bib.bib20) for repeated evaluations. These numbers were chosen to keep the total number of evaluations roughly consistent with those of MATH and MMLU-Pro.

#### LLM Setup

Throughout our evaluations, we use gpt-4o-mini as the backbone LLM (i.e., the LLM powering the nodes in the MAS). If a method requires the use of an optimizer LLM or a meta-agent (i.e., the LLM that proposes the next MAS to evaluate during the search phase), we use gpt-4.1-mini, as it is a newer and stronger model than gpt-4o-mini. Note that our method does not require an optimizer LLM.

#### Results

We now discuss the results obtained by evaluating our method under the setup described above. The main results are shown in Table [2](https://arxiv.org/html/2512.14079v1#S6.T2). We find that our method performs better than the next-best automatic search method, ADAS, on four out of five evaluated datasets as well as on average. Our method improves accuracy by 0.8 points over ADAS, 2.5 points over AFlow, and 1.4 points over the strongest manually designed MAS baseline (Multi-Agent Debate). On AIME in particular, our method achieves a 2.5% absolute improvement over other automatic search methods. In addition, we find that our method tends to produce lower standard deviations, with an average value of approximately 1.0 compared to 1.2 for ADAS, 1.2 for AFlow, and 1.4 for manually designed MASes. This indicates that Grammar Search not only performs better on average but also yields more stable and reliable MASes.

### 6.1 Evaluation Beyond Accuracy

Accuracy is not the only factor that matters when choosing a framework. Therefore, we further compare Grammar Search and ADAS on metrics related to efficiency and MAS quality, as summarized in Table [3](https://arxiv.org/html/2512.14079v1#S6.T3). Grammar Search is more efficient, reliable, and interpretable than ADAS while achieving comparable accuracy. Our framework eliminates wasted computation during MAS generation, reduces overall API costs by approximately 12%, and always produces simpler, valid MASes. While ADAS occasionally discovers a strong MAS early in the search, it spends substantial computation on invalid or overly complex MASes, whereas Grammar Search reliably constructs valid and concise MASes through structured component composition. Note that by *valid* MASes, we mean those that do not contain any syntactic or runtime errors and achieve an accuracy greater than zero when evaluated on the validation set.

During the search phase, Grammar Search incurs zero cost for MAS generation because it does not rely on LLMs to generate code. All MASes are constructed from verified components, ensuring 100% validity and eliminating wasted computation on invalid MASes. By contrast, ADAS incurs nontrivial generation costs ($0.06 per valid MAS and $0.25 per invalid MAS) and wastes computation on invalid generations, which occur in 23% of cases. For the evaluation of valid MASes, Grammar Search and ADAS perform a similar number of trials, leading to comparable evaluation costs. Overall, Grammar Search achieves an average total search cost of $8.37 per MAS, compared to $10.25 for ADAS, a reduction of about 12%.

Beyond cost, Table [3](https://arxiv.org/html/2512.14079v1#S6.T3) also highlights major differences in MAS reliability and complexity. ADAS produces MASes with longer code (8,612 vs. 4,392 characters on average) and substantial code growth across search iterations (524 vs. 33 characters per iteration), yet this added complexity does not lead to higher accuracy. Moreover, ADAS tends to find its best MAS early (around iteration 3), suggesting that later iterations mainly generate redundant or inefficient code. In contrast, Grammar Search maintains steady exploration and discovers its best MAS later (around iteration 11.5), indicating a more systematic and stable search process.

Overall, Grammar Search achieves lower search-phase cost, guarantees valid MAS generation, and produces shorter, more interpretable MASes than ADAS, demonstrating that structured, grammar-based search can match or surpass the effectiveness of free-form code generation while being substantially more efficient and reliable.

Table: Table 3: Analysis of MASes generated during the search phase of ADAS and Grammar Search. Code growth is measured with respect to the previously generated MAS. 23% of ADAS-generated MASes are invalid (the code does not run or they obtain 0% accuracy). Grammar Search is more efficient across the board, and the generated MASes grow at a slower pace and are shorter.

|  | ADAS | Grammar |
| --- | --- | --- |
| Valid MASes | 77% | 100% |
| Avg. generation cost | $0.06 | $0.00 |
| Avg. validation cost | $10.19 | $8.37 |
| Avg. total cost | $10.25 | $8.37 |
| Invalid MASes | 23% | 0% |
| Avg. generation cost | $0.25 | $0.00 |
| Avg. validation cost | $0.96 | $0.00 |
| Avg. total cost | $1.21 | $0.00 |
| All MASes |  |  |
| Avg. code length. (#chars.) | 8,611.8 | 4,391.5 |
| Avg. code growth (#chars.) | 523.8 | 32.8 |
| Best MAS found at iteration | 3.0 | 11.5 |

Table: Table 4: Validation accuracies of the best MASes found by Grammar Search with forced and random sampling. Random sampling performs worse on average.

| Dataset | Forced | Random |
| --- | --- | --- |
| MATH | 81.6 ± 0.7 | 80.8 ± 0.7 |
| AIME | 24.6 ± 2.2 | 24.0 ± 1.6 |
| GPQA | 41.7 ± 3.0 | 41.6 ± 2.6 |
| MMLU-Pro | 70.1 ± 1.8 | 69.5 ± 2.1 |
| Avg. | 54.5 ± 2.0 | 54.0 ± 1.8 |

#### Best MASes found

We depict the best MASes found through our method in Appendix [A](https://arxiv.org/html/2512.14079v1#A1). Across datasets, the component sequences share several components, most notably DebateIteration and StepByStepReasoner (the SIMO version), suggesting that these are generally strong building blocks.

#### Ablation

We also describe an ablation experiment comparing the forced sampling strategy with the random sampling strategy for sampling component sequences during the search phase. Table [4](https://arxiv.org/html/2512.14079v1#S6.T4) presents the results of this ablation. While the gains are not large, they are consistent across datasets, indicating a clear advantage of the forced sampling strategy over random sampling.

## 7 Conclusion

In this work, we proposed Grammar Search, a new corpus-level framework for automatic MAS discovery in code space. Our method constructs a context-free grammar that defines a structured and extensible design space for composing multi-agent systems. This grammar constrains how building-block components can be combined, ensuring that only valid MASes are generated. By sampling from the grammar, we produce valid MAS candidates during the search phase, thereby avoiding wasted computation and API costs on invalid generations. To maintain coverage and prevent under-testing, we employed a simple forced-sampling strategy that balances the frequency of each component across sampled sequences.

We showed that Grammar Search outperforms existing automatic MAS search methods and manually designed baselines on four out of five benchmarks, as well as on average. Furthermore, the MASes produced by our framework are modular, interpretable, and efficient, providing simpler yet effective multi-agent systems while reducing search cost.

## Limitations

This work has several limitations. First, our approach lacks the free-form expressivity of LLM-based code generation, and the MASes it produces are constrained by the grammar we developed. However, this constraint offers practical benefits, as it enables a cheaper search stage and avoids wasted computation on invalid MASes. In addition, our grammar is extensible and could potentially be modified, for example by an LLM, to approach the expressivity of other methods.

Second, the best MASes generated by our method depend on the initial components defined in the grammar. If these components are weak, the resulting MASes will not achieve performance comparable to that of the best existing methods.

Finally, although we were able to achieve good results with a relatively simple random search strategy, we expect that expanding the grammar with more components will require more sophisticated search algorithms such as Monte Carlo Tree Search or Thompson sampling.

## Ethics

We work with LLMs in this study, which are known to carry various social and ethical risks [Weidinger et al. (2021)](https://arxiv.org/html/2512.14079v1#bib.bib4). Techniques such as RLHF have helped mitigate some of these risks, but unsafe behaviors may still arise due to methods like LLM jailbreaking.

All datasets, models, and code used in this study were obtained from public sources and used in accordance with their respective licenses and terms of use.

## Note on AI Assistant Usage

We acknowledge the use of AI tools such as GPT-5 and Claude Opus 4.1 for grammar checking, paraphrasing, and polishing the manuscript to improve clarity, as well as for assistance in writing and debugging portions of the associated code.

## References

- Comanici et al. (2025) G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen, et al. Gemini 2.5: pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261. Cited by: [§1](https://arxiv.org/html/2512.14079v1#S1.p1.1).

- Du et al. (2023) Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch Improving factuality and reasoning in language models through multiagent debate. In Forty-first International Conference on Machine Learning, Cited by: [§1](https://arxiv.org/html/2512.14079v1#S1.p1.1), [§3](https://arxiv.org/html/2512.14079v1#S3.SS0.SSS0.Px1.p1.1), [§3](https://arxiv.org/html/2512.14079v1#S3.SS0.SSS0.Px2.SPx2.p1.1).

- Fernando et al. (2023) C. Fernando, D. Banarse, H. Michalewski, S. Osindero, and T. Rocktäschel Promptbreeder: self-referential self-improvement via prompt evolution. External Links: 2309.16797, [Link](https://arxiv.org/abs/2309.16797) Cited by: [§3](https://arxiv.org/html/2512.14079v1#S3.SS0.SSS0.Px2.SPx2.p1.1).

- Gao et al. (2025) H. Gao, Y. Liu, Y. He, L. Dou, C. Du, Z. Deng, B. Hooi, M. Lin, and T. Pang Flowreasoner: reinforcing query-level meta-agents. arXiv preprint arXiv:2504.15257. Cited by: [§3](https://arxiv.org/html/2512.14079v1#S3.SS0.SSS0.Px2.SPx3.p1.1).

- He et al. (2024) H. He, W. Yao, K. Ma, W. Yu, Y. Dai, H. Zhang, Z. Lan, and D. Yu WebVoyager: building an end-to-end web agent with large multimodal models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), L. Ku, A. Martins, and V. Srikumar (Eds.), Bangkok, Thailand, pp. 6864–6890. External Links: [Link](https://aclanthology.org/2024.acl-long.371/), [Document](https://dx.doi.org/10.18653/v1/2024.acl-long.371) Cited by: [§1](https://arxiv.org/html/2512.14079v1#S1.p1.1).

- Hendrycks et al. (2021a) D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt Measuring massive multitask language understanding. Proceedings of the International Conference on Learning Representations (ICLR). Cited by: [§6](https://arxiv.org/html/2512.14079v1#S6.SS0.SSS0.Px1.p4.1).

- Hendrycks et al. (2021b) D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt Measuring mathematical problem solving with the math dataset. External Links: 2103.03874, [Link](https://arxiv.org/abs/2103.03874) Cited by: [§6](https://arxiv.org/html/2512.14079v1#S6.SS0.SSS0.Px1.p2.1).

- Ho et al. (2025) X. Ho, J. Huang, F. Boudin, and A. Aizawa LLM-as-a-judge: reassessing the performance of llms in extractive qa. arXiv preprint arXiv:2504.11972. Cited by: [§6](https://arxiv.org/html/2512.14079v1#S6.SS0.SSS0.Px3.p1.1).

- Hochlehnert et al. (2025) A. Hochlehnert, H. Bhatnagar, V. Udandarao, S. Albanie, A. Prabhu, and M. Bethge A sober look at progress in language model reasoning: pitfalls and paths to reproducibility. arXiv preprint arXiv:2504.07086. Cited by: [§6](https://arxiv.org/html/2512.14079v1#S6.SS0.SSS0.Px1.p3.1), [§6](https://arxiv.org/html/2512.14079v1#S6.SS0.SSS0.Px3.p2.1).

- Hu et al. (2025) S. Hu, C. Lu, and J. Clune Automated design of agentic systems. In The Thirteenth International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=t9U3LW7JVX) Cited by: [§1](https://arxiv.org/html/2512.14079v1#S1.p3.1), [§2](https://arxiv.org/html/2512.14079v1#S2.SS0.SSS0.Px2.p1.1), [§3](https://arxiv.org/html/2512.14079v1#S3.SS0.SSS0.Px2.SPx1.p1.1).

- Ke et al. (2025a) Z. Ke, A. Xu, Y. Ming, X. Nguyen, C. Xiong, and S. Joty MAS-zero: designing multi-agent systems with zero supervision. arXiv preprint arXiv:2505.14996. Cited by: [§6](https://arxiv.org/html/2512.14079v1#S6.SS0.SSS0.Px3.p1.1).

- Ke et al. (2025b) Z. Ke, A. Xu, Y. Ming, X. Nguyen, C. Xiong, and S. Joty MAS-zero: designing multi-agent systems with zero supervision. External Links: 2505.14996, [Link](https://arxiv.org/abs/2505.14996) Cited by: [§3](https://arxiv.org/html/2512.14079v1#S3.SS0.SSS0.Px2.SPx4.p1.1).

- Kocsis and Szepesvári (2006) L. Kocsis and C. Szepesvári Bandit based monte-carlo planning. In European conference on machine learning, pp. 282–293. Cited by: [§5](https://arxiv.org/html/2512.14079v1#S5.p8.1).

- Li et al. (2023) G. Li, H. A. A. K. Hammoud, H. Itani, D. Khizbullin, and B. Ghanem CAMEL: communicative agents for ”mind” exploration of large language model society. In Thirty-seventh Conference on Neural Information Processing Systems, External Links: [Link](https://openreview.net/forum?id=3IyL2XWDkG) Cited by: [§1](https://arxiv.org/html/2512.14079v1#S1.p2.1).

- Madaan et al. (2023) A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, et al. Self-refine: iterative refinement with self-feedback. Advances in Neural Information Processing Systems 36, pp. 46534–46594. Cited by: [§1](https://arxiv.org/html/2512.14079v1#S1.p1.1), [§3](https://arxiv.org/html/2512.14079v1#S3.SS0.SSS0.Px1.p1.1).

- OpenAI (2025) OpenAI Introducing gpt-5. Note: [https://openai.com/index/introducing-gpt-5/](https://openai.com/index/introducing-gpt-5/)Accessed: 2025-10-05 Cited by: [§1](https://arxiv.org/html/2512.14079v1#S1.p1.1), [§6](https://arxiv.org/html/2512.14079v1#S6.SS0.SSS0.Px3.p1.1).

- Park et al. (2023) J. S. Park, J. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein Generative agents: interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, UIST ’23, New York, NY, USA. External Links: ISBN 9798400701320, [Link](https://doi.org/10.1145/3586183.3606763), [Document](https://dx.doi.org/10.1145/3586183.3606763) Cited by: [§1](https://arxiv.org/html/2512.14079v1#S1.p2.1).

- Plaat et al. (2025) A. Plaat, M. van Duijn, N. van Stein, M. Preuss, P. van der Putten, and K. J. Batenburg Agentic large language models, a survey. External Links: 2503.23037, [Link](https://arxiv.org/abs/2503.23037) Cited by: [§2](https://arxiv.org/html/2512.14079v1#S2.SS0.SSS0.Px1.p1.1).

- Qiu et al. (2025) J. Qiu, X. Qi, T. Zhang, X. Juan, J. Guo, Y. Lu, Y. Wang, Z. Yao, Q. Ren, X. Jiang, X. Zhou, D. Liu, L. Yang, Y. Wu, K. Huang, S. Liu, H. Wang, and M. Wang Alita: generalist agent enabling scalable agentic reasoning with minimal predefinition and maximal self-evolution. External Links: 2505.20286, [Link](https://arxiv.org/abs/2505.20286) Cited by: [§3](https://arxiv.org/html/2512.14079v1#S3.SS0.SSS0.Px2.SPx2.p1.1).

- Rein et al. (2023) D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman GPQA: a graduate-level google-proof q&a benchmark. External Links: 2311.12022, [Link](https://arxiv.org/abs/2311.12022) Cited by: [§6](https://arxiv.org/html/2512.14079v1#S6.SS0.SSS0.Px1.p5.1).

- Schick et al. (2023) T. Schick, J. Dwivedi-Yu, R. Dessi, R. Raileanu, M. Lomeli, E. Hambro, L. Zettlemoyer, N. Cancedda, and T. Scialom Toolformer: language models can teach themselves to use tools. In Thirty-seventh Conference on Neural Information Processing Systems, External Links: [Link](https://openreview.net/forum?id=Yacmpz84TH) Cited by: [§1](https://arxiv.org/html/2512.14079v1#S1.p1.1).

- Thompson (1933) W. R. Thompson On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika 25 (3/4), pp. 285–294. Cited by: [§5](https://arxiv.org/html/2512.14079v1#S5.p8.1).

- Veeraboina (2023) H. Veeraboina AIME problem set 1983-2024. Kaggle. External Links: [Link](https://www.kaggle.com/datasets/hemishveeraboina/aime-problem-set-1983-2024) Cited by: [§6](https://arxiv.org/html/2512.14079v1#S6.SS0.SSS0.Px1.p3.1).

- Wang et al. (2024a) L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin, W. X. Zhao, Z. Wei, and J. Wen A survey on large language model based autonomous agents. Frontiers Comput. Sci. 18 (6), pp. 186345. External Links: [Link](https://doi.org/10.1007/s11704-024-40231-1) Cited by: [§1](https://arxiv.org/html/2512.14079v1#S1.p2.1).

- Wang et al. (2023) X. Wang, J. Wei, D. Schuurmans, Q. V. Le, E. H. Chi, S. Narang, A. Chowdhery, and D. Zhou Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=1PL1NIMMrw) Cited by: [§3](https://arxiv.org/html/2512.14079v1#S3.SS0.SSS0.Px1.p1.1).

- Wang et al. (2024b) Y. Wang, X. Ma, G. Zhang, Y. Ni, A. Chandra, S. Guo, W. Ren, A. Arulraj, X. He, Z. Jiang, et al. Mmlu-pro: a more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems 37, pp. 95266–95290. Cited by: [§6](https://arxiv.org/html/2512.14079v1#S6.SS0.SSS0.Px1.p4.1).

- Wei et al. (2023) J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. Le, and D. Zhou Chain-of-thought prompting elicits reasoning in large language models. External Links: 2201.11903, [Link](https://arxiv.org/abs/2201.11903) Cited by: [§3](https://arxiv.org/html/2512.14079v1#S3.SS0.SSS0.Px1.p1.1).

- Weidinger et al. (2021) L. Weidinger, J. Mellor, M. Rauh, C. Griffin, J. Uesato, P. Huang, M. Cheng, M. Glaese, B. Balle, A. Kasirzadeh, Z. Kenton, S. Brown, W. Hawkins, T. Stepleton, C. Biles, A. Birhane, J. Haas, L. Rimell, L. A. Hendricks, W. Isaac, S. Legassick, G. Irving, and I. Gabriel Ethical and social risks of harm from language models. External Links: 2112.04359, [Link](https://arxiv.org/abs/2112.04359) Cited by: [Ethics](https://arxiv.org/html/2512.14079v1#Sx2.p1.1).

- Yang et al. (2024) J. Yang, C. E. Jimenez, A. Wettig, K. Lieret, S. Yao, K. Narasimhan, and O. Press Swe-agent: agent-computer interfaces enable automated software engineering. Advances in Neural Information Processing Systems 37, pp. 50528–50652. Cited by: [§1](https://arxiv.org/html/2512.14079v1#S1.p1.1).

- Yao et al. (2023) S. Yao, D. Yu, J. Zhao, I. Shafran, T. Griffiths, Y. Cao, and K. Narasimhan Tree of thoughts: deliberate problem solving with large language models. Advances in neural information processing systems 36, pp. 11809–11822. Cited by: [§1](https://arxiv.org/html/2512.14079v1#S1.p1.1), [§3](https://arxiv.org/html/2512.14079v1#S3.SS0.SSS0.Px1.p1.1).

- Yue et al. (2025) Y. Yue, G. Zhang, B. Liu, G. Wan, K. Wang, D. Cheng, and Y. Qi Masrouter: learning to route llms for multi-agent systems. arXiv preprint arXiv:2502.11133. Cited by: [§1](https://arxiv.org/html/2512.14079v1#S1.p2.1), [§3](https://arxiv.org/html/2512.14079v1#S3.SS0.SSS0.Px2.SPx3.p1.1).

- Zhang et al. (2025a) G. Zhang, L. Niu, J. Fang, K. Wang, L. Bai, and X. Wang Multi-agent architecture search via agentic supernet. arXiv preprint arXiv:2502.04180. Cited by: [§1](https://arxiv.org/html/2512.14079v1#S1.p2.1), [§3](https://arxiv.org/html/2512.14079v1#S3.SS0.SSS0.Px2.SPx1.p1.1).

- Zhang et al. (2025b) J. Zhang, J. Xiang, Z. Yu, F. Teng, X. Chen, J. Chen, M. Zhuge, X. Cheng, S. Hong, J. Wang, B. Zheng, B. Liu, Y. Luo, and C. Wu AFlow: automating agentic workflow generation. In The Thirteenth International Conference on Learning Representations, External Links: [Link](https://openreview.net/forum?id=z5uVAKwmjf) Cited by: [§1](https://arxiv.org/html/2512.14079v1#S1.p3.1), [§3](https://arxiv.org/html/2512.14079v1#S3.SS0.SSS0.Px2.SPx1.p1.1), [§6](https://arxiv.org/html/2512.14079v1#S6.SS0.SSS0.Px3.p2.1).

- Zhao et al. (2025) W. Zhao, M. Yuksekgonul, S. Wu, and J. Zou Sirius: self-improving multi-agent systems via bootstrapped reasoning. arXiv preprint arXiv:2502.04780. Cited by: [§3](https://arxiv.org/html/2512.14079v1#S3.SS0.SSS0.Px2.SPx2.p1.1).

- Zhuge et al. (2024) M. Zhuge, W. Wang, L. Kirsch, F. Faccio, D. Khizbullin, and J. Schmidhuber Gptswarm: language agents as optimizable graphs. In Forty-first International Conference on Machine Learning, Cited by: [§1](https://arxiv.org/html/2512.14079v1#S1.p2.1), [§2](https://arxiv.org/html/2512.14079v1#S2.SS0.SSS0.Px2.p1.1), [§3](https://arxiv.org/html/2512.14079v1#S3.SS0.SSS0.Px2.SPx1.p1.1).

The best component sequences discovered across each dataset using Grammar Search are depicted in Table [5](https://arxiv.org/html/2512.14079v1#A1.T5).

Table: Table 5: We present the best component sequence discovered for each dataset using Grammar Search with the forced sampling strategy. For brevity, parameter names are omitted.

| Dataset | Best component sequence found |
| --- | --- |
| AIME | StepByStepReasoner(5) $\rightarrow$ DebateIteration(2) $\rightarrow$ DebateIteration(2) $\rightarrow$ MajorityVoter |
| GPQA | StepByStepReasoner(5) $\rightarrow$ MultiSelfCriticIteration(5) $\rightarrow$ DebateIteration(2) $\rightarrow$ ConsensusBuilder |
| MATH | StepByStepReasoner(5) $\rightarrow$ DebateIteration(2) $\rightarrow$ ConsensusBuilder |
| MMLU-Pro | StepByStepReasoner(5) $\rightarrow$ DebateIteration(2) $\rightarrow$ MajorityVoter $\rightarrow$ RoleBasedReasoner(1) $\rightarrow$ StepByStepReasoner(1) |

Simplified python code for Grammar Search with Forced Sampling can be found in Listing [1](https://arxiv.org/html/2512.14079v1#LST1).

Figure: Listing 1: Grammar MAS Search with Forced Sampling

Our code infrastructure for calling LLMs inside MASes is similar to ADAS. A simplified version is depicted in Listing [2](https://arxiv.org/html/2512.14079v1#LST2).

Figure: Listing 2: Core Infrastructure Components

We depict code for several MASes generated through our framework in Listings [3](https://arxiv.org/html/2512.14079v1#LST3), [4](https://arxiv.org/html/2512.14079v1#LST4) and [5](https://arxiv.org/html/2512.14079v1#LST5).

Figure: Listing 3: Generated MAS: Self-Consistency with Voting

Figure: Listing 4: Generated MAS: Single Reasoner with Self-Criticism

Figure: Listing 5: Generated MAS: Role-Based Debate with Consensus
