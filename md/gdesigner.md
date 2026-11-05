<!-- Source: https://arxiv.org/html/2410.11782v3; retrieved: 2026-09-28T20:41:40.137580+00:00; conversion: arXiv HTML to Markdown via lxml; original article text follows. -->

# G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks

## Abstract

Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution? In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: (1) high-performing, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; (2) task-adaptive, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and (3) adversarially robust, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop. The code is available at [https://github.com/yanweiyue/GDesigner](https://github.com/yanweiyue/GDesigner).

## 1 Introduction

An LLM-based agent, which integrates the language generation capabilities of LLMs with decision-making and action-execution functionalities ([Richards & et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib41); [Nakajima, 2023](https://arxiv.org/html/2410.11782v3#bib.bib34); [Reworkd, 2023](https://arxiv.org/html/2410.11782v3#bib.bib40)), has exhibited impressive performance across a wide range of tasks, from reasoning ([Yao et al., 2023b](https://arxiv.org/html/2410.11782v3#bib.bib58)) and code generation ([Shinn et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib44)) to even more complex applications like video gaming ([Wang et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib47)) and autonomous driving ([Jin et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib22)). Even more exciting, researchers have discovered that combining multiple LLM-based agents–whether implicitly or explicitly–into a team can outperform individual agents when tackling complex tasks ([Du et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib11); [Liang et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib29); [Wang et al., 2023b](https://arxiv.org/html/2410.11782v3#bib.bib51); [Jiang et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib21); [Shinn et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib44); [Zheng et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib65); [Wu et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib54)), demonstrating a form of collaborative intelligence reminiscent of human teamwork in multi-agent systems ([Zhang et al., 2023b](https://arxiv.org/html/2410.11782v3#bib.bib62)). This emergence of human-esque collective intelligence is fundamentally driven by the design of their topology, i.e., how multi-agents are connected, and how they transmit, exchange, and assimilate information reciprocally.

Figure: Figure 1: Existing practices for LLM-based multi-agent communication topology design. [image: https://arxiv.org/html/2410.11782v3/intro_topo.png]

Figure: Figure 2: The token consumption and accuracy of different multi-agent protocols on two subsets of MMLU dataset, “Highschool Biology” and “College Math”, tested with four gpt-4-based agents. [image: https://arxiv.org/html/2410.11782v3/intro_scatter_compressed.png]

In practice, prior research has extensively explored how multiple instances of LLMs, referred to as agents ([Wang et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib48); [Xi et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib55); [Gao et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib13); [Cheng et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib8)), should be structured and organized to converse, collaborate, debate, or even compete. Various topological designs have been investigated, such as chain ([Wei et al., 2022](https://arxiv.org/html/2410.11782v3#bib.bib52); [Hong et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib17)), tree ([Yao et al., 2023a](https://arxiv.org/html/2410.11782v3#bib.bib57); [Wu et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib54)), star ([Wu et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib54)), complete graphs ([Qian et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib38)), random graphs ([Qian et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib38)), optimizable graphs ([Zhuge et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib68); [Zhang et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib60)), and LLM-based networks ([Hao et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib14); [Liu et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib32)). These elaborately designed communication topologies have demonstrated remarkable performance with minimal human supervision, bridging the gap between individual and collective intelligence. Faced with numerous structures available, an inquisitive practitioner might ask: how should I select or design a topology that best suits my task at hand?

The question posed above is non-trivial and, at times, perplexing. A piece of experimental evidence is presented in [Figure 2](https://arxiv.org/html/2410.11782v3#S1.F2), where we evaluated the performance of different multi-agent structures on the MMLU dataset ([Hendrycks et al., 2021](https://arxiv.org/html/2410.11782v3#bib.bib15)), a collection of multiple-choice questions across various subjects. The results reveal that even within the same dataset, the suitability of different communication topologies varies. ❶ Simpler Case: in the simpler ”High School Biology” subset, the chain structure performs comparably to the complex GPTSwarm, while consuming significantly fewer tokens (0.5k versus 7.8k). In this case, the chain structure is clearly a more economical choice. ❷ Harder Case: However, for the more challenging ”College Mathematics” subset, GPTSwarm outperforms the chain structure by $8.75\%$, primarily attributed to its intricate topology and prompt optimization. In summary, practitioners often find it challenging to effortlessly identify the most efficient and complexity-adaptive multi-agent topology for a given task.

In light of this dilemma, we propose the LLM-based Multi-agent Communication Protocol (MACP), establishing standardized guidance for LLM-MA topology design:

Multi-agent Communication Protocol (MACP): Given a task/query $q$, an optimal LLM-MA communication topology for $q$ should satisfy the following protocol logics: (1) Effectiveness: The communication structure must effectively produce the qualified solution for $q$; (2) Complexity-adaptiveness: The topology should dynamically adjust to the complexity of the task, minimizing communication overhead; (3) Adversarial robustness: The topology should maintain reliable under adversarial attacks.

The formal definition of MACP is provided in [Section 3.3](https://arxiv.org/html/2410.11782v3#S3.SS3). To design a communication topology that ideally adheres to the MACP principles, we propose an effective, adaptive, and robust LLM-powered multi-agent communication graph designer, termed G-Designer. Technically, G-Designer first architects a multi-agent graph, where each agent, along with its specific properties (e.g., profile ([Li et al., 2023a](https://arxiv.org/html/2410.11782v3#bib.bib26)), external API tools ([Zhuang et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib67)), or knowledge base ([Chen et al., 2024a](https://arxiv.org/html/2410.11782v3#bib.bib4))), is represented as a node, and communication between agents forms the edges. G-Designer employs a variational graph auto-encoder to encode the nodes (agents) along with task-specific information, and to decode the resulting collaboration network between agents. This input-dependent paradigm allows G-Designer to design task-adaptive, high-performing communication topology, which is, at the same time, assured of efficiency and robustness with sparsity regularization. Unlike previous LLM-based multi-agent topology designs, which rely on a static structure for all queries/tasks, G-Designer adaptively crafts customized topologies for different domains and tasks, serving as a fully autonomous and flexible assistant for multi-agent system establishment and deployments.

Our contribution can be summarized as follows:

❶ Protocol Proposal. We propose the first communication protocol tailored for LLM-powered multi-agent systems, MACP, which comprehensively regulates multi-agent topology design across three dimensions: performance, adaptability, and robustness, and incisively highlights the shortcomings of existing designs. ❷ Practical Solution. We present G-Designer, an effective, adaptive, and robust designer of LLM-powered multi-agent communication graphs. By leveraging a variational graph auto-encoder to construct and process the multi-agent network, G-Designer decodes task-adaptive and high-performing agent communication, which is also equipped with strong robustness against agent-rooted adversarial attacks via dynamic topology adjustment. ❸ Experimental Validation. Extensive experiments across six benchmarks show that G-Designer is: (1) high-performing, surpassing state-of-the-art topologies by $0.20\%\sim 4.10\%$ on MMLU and HumanEval; (2) task-adaptive, dynamically adjusting topology complexity with task awareness, outperforming state-of-the-art methods on MMLU with a cost of merely $1.5e+5$ compared to their $2.6e+6$, reducing token consumption by up to $92.24\%$; and (3) adversarially robust, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.

## 2 Related Works

#### LLM-agent Collaboration

Recent research has explored various multi-agent communication topologies, including: (1) Non-interactive, where agents operate independently without inter-agent communication, as employed in systems like LATM ([Zhang et al., 2023a](https://arxiv.org/html/2410.11782v3#bib.bib61)) and LLM-Debate ([Du et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib11)); (2) Chain, where agents are arranged in a sequential structure, each receiving the output from its predecessor and passing information to its successor, utilized by ChatDev ([Qian et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib37)), MetaGPT ([Hong et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib17)), and L2MAC ([Holt et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib16)); (3) Star, where a central administrative agent (often referred to as a commander, teacher, etc.) directs subordinate agents, seen in AutoGen ([Wu et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib54)), SecurityBot ([Yan et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib56)), and MiniGrid ([Zhou et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib66)); (4) Tree, where a root agent hierarchically manages multiple child agents, as in SoA ([Ishibashi & Nishimura, 2024](https://arxiv.org/html/2410.11782v3#bib.bib20)); and (5) Graph, encompassing complete graphs ([Qian et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib38); [Zhuge et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib68)) and random graphs ([Qian et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib38)), among others.

#### Multi-agents as Graphs

Graphs, as a fundamental data structure for organizing and representing relationships between entities ([Zhang & Chartrand, 2006](https://arxiv.org/html/2410.11782v3#bib.bib63)), are widely adopted in the pre-LLM era as a powerful tool to facilitate effective communication in multi-agent reinforcement learning (MARL) ([Pesce & Montana, 2023](https://arxiv.org/html/2410.11782v3#bib.bib36); [Hu et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib18); [Liu et al., 2022](https://arxiv.org/html/2410.11782v3#bib.bib31)). With the rise of LLMs and the proliferation of LLM-based agents ([Chen et al., 2023a](https://arxiv.org/html/2410.11782v3#bib.bib3); [Cohen et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib10); [Hua et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib19)), researchers have similarly recognized that interactions among multiple agents can naturally be modeled from a graph-based perspective ([Chen et al., 2023b](https://arxiv.org/html/2410.11782v3#bib.bib7); [Zhuge et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib68); [Qian et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib38); [Liu et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib32)). Early attempts are implicit, like ChatEval ([Chan et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib2)), AutoGen ([Wu et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib54)), and DSPy ([Khattab et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib23)). More recent practices including ChatLLM ([Hao et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib14)), DyLAN ([Liu et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib32)), GPTSwarm ([Zhuge et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib68)), and MacNet ([Qian et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib38)), have explicitly represented the organization of multiple agents as a graph. However, all these attempts, whether predefined or iteratively optimized, remain input-independent. Consequently, they fail to be task-aware and adaptively design topologies that suit the complexity of the specific task.

## 3 Formalization

This section establishes the notation, formalizes key concepts from a topology perspective, and formally defines our proposed multi-agent communication protocol.

### 3.1 Topology Structure

We model the multi-agent system as a directed graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$, where $\mathcal{V}=\{v_{1},\dots,v_{N}\}$ represents the set of nodes (with $N=|\mathcal{V}|$) and $\mathcal{E}$ denotes the set of edges. Each node $v_{i}\in\mathcal{V}$ corresponds to an agent, formalized as: $v_{i}=\{\texttt{Base}_{i},\texttt{Role}_{i},\texttt{State}_{i},\texttt{Plugin}_{i}\},$ (1) where each agent $v_{i}$ is composed of four key elements: (1) $\texttt{Base}_{i}$, the language model instance powering $v_{i}$; (2) $\texttt{Role}_{i}$, the agent’s pre-assigned role or function; (3) $\texttt{State}_{i}$, representing the agent’s accumulated knowledge and interaction history; and (4) $\texttt{Plugin}_{i}$, a set of external tools or plugins available to $v_{i}$, such as web searchers ([Ma et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib33)), code compilers ([Richards & et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib41); [Wu et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib54); [Hong et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib17); [Bouzenia et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib1); [Ishibashi & Nishimura, 2024](https://arxiv.org/html/2410.11782v3#bib.bib20)), or file readers ([Zhuge et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib68); [Richards & et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib41)). Each LLM-based agent $v_{i}$ receives prompt $\mathcal{P}$ and generates response $\mathcal{R}_{i}$: $\mathcal{R}_{i}=v_{i}(\mathcal{P})=v_{i}(\mathcal{P}_{\text{sys}},\mathcal{P}_{\text{usr}}),$ (2) where $\mathcal{P}_{\text{sys}}=\{\texttt{Role}_{i},\texttt{State}_{i}\}$ represents the system prompt encompassing its role and state, and $\mathcal{P}_{\text{usr}}$ denotes the user prompt, which possibly includes the given tasks, responses/instructions from other agents and externally retrieved knowledge.

The connectivity of $\mathcal{G}$ can also be characterized by a (non-symmetric) adjacency matrix $\mathbf{A}\in\{0,1\}^{N\times N}$, where $\mathbf{A}[i,j]=1$ if $e_{ij}=(v_{i},v_{j})\in\mathcal{E}$, otherwise $0$. Each edge $e_{ij}\in\mathcal{E}$ represents the flow of information from $v_{i}$ to $v_{j}$.

Figure: Figure 3: The designing workflow of our proposed G-Designer. [image: https://arxiv.org/html/2410.11782v3/framework-1.png]

### 3.2 Communication Pipeline

Given a query/problem $\mathcal{Q}$, the multi-agent system engages in $K$ rounds of interactive utterances, which collaboratively drive the agents toward producing the final solution $a^{(K)}$ based on their cumulative dialogue exchanges. At the beginning of the $t$-th dialogue round, a mapping function $\phi$ is applied to determine the execution index for each agent: $\displaystyle\phi:\mathcal{G}\longmapsto\sigma,\;\sigma=[v_{\sigma_{1}},v_{\sigma_{2}},\cdots,v_{\sigma_{N}}],$ (3) $\displaystyle\operatorname{s.t.}\forall i>j,\;\;v_{\sigma_{i}}\notin\mathcal{N}_{\text{in}}(v_{\sigma_{j}}),$ where $\sigma$ is the execution sequence of agents, $\mathcal{N}_{\text{in}}(v_{\sigma(j)})$ denotes the in-neighborhood of $v_{\sigma(j)}$, and the constraint ensures that an agent $v_{\sigma(i)}$ can only execute after any agent $v_{\sigma(j)}$ from which it receives information. Once the execution order is determined, each agent proceeds to perform input-output operations sequentially: $\mathcal{R}_{i}^{(t)}=v_{i}(\mathcal{P}^{(t)}_{\text{sys}},\mathcal{P}^{(t)}_{\text{usr}}),\;\mathcal{P}^{(t)}_{\text{usr}}=\{\mathcal{Q},\cup_{v_{j}\in\mathcal{N}_{\text{in}}(v_{i})}\mathcal{R}_{j}^{(t)}\}$ (4) where $\mathcal{R}_{i}^{(t)}$ represents the output of $v_{i}$, which could be a rationale, an answer, or a partial solution, depending on the specific context. The output $\mathcal{R}_{i}^{(t)}$ is generated based on the system prompt $\mathcal{P}_{\text{sys}}^{(t)}$ and the context prompt, consisting of the query $\mathcal{Q}$ and messages from other agents. At the end of each dialogue, an aggregation function is adopted to generate the answer/solution $a^{(t)}$: $a^{(t)}\leftarrow\operatorname{Aggregate}(\mathcal{R}_{1}^{(t)},\mathcal{R}_{2}^{(t)},\cdots,\mathcal{R}_{N}^{(t)}).$ (5) The implementation of the $\operatorname{Aggregate}$ function is flexible, with possible options including majority voting ([Chen et al., 2024b](https://arxiv.org/html/2410.11782v3#bib.bib5); [Zhuge et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib68); [Li et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib27)), aggregating all agents’ responses and delegating one agent to provide the final answer ([Wu et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib54); [Jiang et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib21); [Liu et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib32); [Zhang et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib60)), or simply using the output of the last agent $\mathcal{R}^{(t)}_{\sigma_{N}}$ ([Qian et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib38)). Through $K$ rounds of utterances, either predefined ([Qian et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib38)) or determined by an early-stopping mechanism ([Liu et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib32)), the overall system $\mathcal{G}$ produces the final answer $a^{(K)}$ for $\mathcal{Q}$.

### 3.3 MACP Protocol

We give the formal definition of MACP Protocol as follows:

Given an LLM-based multi-agent system $\mathcal{G}=(\mathcal{V},\mathcal{E})$, we establish the following objective as optimization principle: $\small\underset{\mathcal{G}\in\mathbb{G}}{\min}\Bigl[-u\bigl(\mathcal{G}(\mathcal{Q})\bigl)+\beta_{1}\cdot||\mathcal{G}||+\beta_{2}\cdot\bigl|\hat{\mathcal{G}}(\hat{\mathcal{Q}})-\mathcal{G}(\mathcal{Q})\bigl|\Bigl],$ (6) where $\mathbb{G}$ represents the feasible parameter space of $\mathcal{G}$, $u(\cdot)$ is the utility evaluator, $||\mathcal{G}||$ measures the computational and communication overhead of the entire graph, and $\hat{\mathcal{Q}}$ and $\hat{\mathcal{G}}$ denote the query description and the multi-agent system after adversarial perturbation, respectively. The first term in [Equation 6](https://arxiv.org/html/2410.11782v3#S3.E6) corresponds to high performance, aiming to maximize the utility of the system’s output; the second term addresses task-adaptiveness, seeking to minimize system complexity to reduce power consumption and economic cost; and the third term focuses on robustness, constraining the deviation of system output under adversarial attacks.

## 4 G-Designer

[Figure 3](https://arxiv.org/html/2410.11782v3#S3.F3) illustrates how G-Designer adaptively designs communication topologies for any given query. Specifically, the process begins with a few “raw materials”: the input query $\mathcal{Q}$, the agent set $\mathcal{V}$, the profile pool, and the toolset. In the Construct stage, G-Designer leverages a node encoder to construct a multi-agent network along with a task-specific virtual node. In the Design stage, a graph auto-encoder is employed to decode the communication graph topology $\mathcal{G}_{com}$, which is leveraged for multi-round inter-agent collaboration in the Optimize stage.

### 4.1 Multi-agent Network Construction

Given an input query $\mathcal{Q}$ and a set of LLM-agents $\mathcal{V}$, G-Designer aims to design a task-adaptive and effective communication topology $\mathcal{G}_{com}$. We begin by assigning each agent a unique role and profile, as previous research ([Wang et al., 2023b](https://arxiv.org/html/2410.11782v3#bib.bib51)) has shown that assigning distinct personas or roles to LLM-based agents can enhance cognitive synergy. Based on these roles, different external tools are allocated to the agents (e.g., Mathematica for a math analyst, Python compiler for a programmer). Thus, we successfully initialize each agent $v_{i}$ as $\{\texttt{Base}_{i},\texttt{Role}_{i},\texttt{State}_{i},\texttt{Plugin}_{i}\}$, as defined in [Equation 1](https://arxiv.org/html/2410.11782v3#S3.E1).

We proceed to construct a structured multi-agent network as input to G-Designer, represented as $\mathcal{G}=(\mathbf{X}_{agent},\mathbf{A})$, where $\mathbf{X}_{agent}\in\mathbb{R}^{N\times D}$ is the node (agent) feature matrix and $\mathbf{A}\in\mathbb{R}^{N\times N}$ represents the connectivity matrix. For the feature matrix $\mathbf{X}_{agent}$, we employ a node encoder to transform each agent’s unique profile into a fixed-length embedding representation: $\vskip-3.99994pt\small\mathbf{x}_{i}\leftarrow\operatorname{NodeEncoder}\left(\mathcal{T}(\texttt{Base}_{i}),\texttt{Role}_{i},\mathcal{T}(\texttt{Plugin}_{i})\right),$ (7) where $\mathcal{T}(\cdot)$ extracts the textual description of the agent’s LLM backbone and its assigned plugins, and $\operatorname{NodeEncoder}$ can be realized using small and lightweight text embedding models ([Reimers, 2019](https://arxiv.org/html/2410.11782v3#bib.bib39)). After encoding the individual agents, we aim to ensure that the multi-agent network incorporates information related to the query $\mathcal{Q}$, as this query-dependent approach enables G-Designer to be task-aware and adaptive. To this end, we introduce an additional task-specific virtual global node $v_{task}$, which is bidirectionally connected to all agent nodes, enabling a global ”storage sink” and facilitating smoother information flow among agents ([Shirzad et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib45); [Tan et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib46); [Rosenbluth et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib42)). This task node is encoded by the $\operatorname{NodeEncoder}$ as follows: $\mathbf{x}_{task}\leftarrow\operatorname{NodeEncoder}(\mathcal{Q})$.

After obtaining the agent node features $\mathbf{X}_{agent}=[\mathbf{x}_{1},\mathbf{x}_{2},\dots,\mathbf{x}_{N}]^{\top}$ and the task-specific embedding $\mathbf{x}_{task}$, we provide a simple anchor topology $\mathbf{A}_{anchor}\in\{0,1\}^{N\times N}$, which serves as a starting point for G-Designer’s topology design process. For instance, given a code generation task with three agents: manager/programmer/code reviewer, the anchor topology could be configured as a chain structure, i.e., “manager $\rightarrow$ programmer $\rightarrow$ reviewer’’, reflecting the typical workflow of code completion. The anchor topology, being either user-defined or automatically generated by LLMs, is often simple and sub-optimal11 1 We discuss the substantial performance improvement of G-Designer over the anchor topology in [Section 5.4](https://arxiv.org/html/2410.11782v3#S5.SS4).. However, it provides a foundational reference and prior knowledge for G-Designer’s subsequent optimization process. We incorporate the task-specific vertex $v_{task}$ and its corresponding edges and obtain $\tilde{\mathbf{A}}_{anchor}\in\{0,1\}^{(N+1)\times(N+1)}$. Consequently, we establish a task-specific multi-agent network $\tilde{\mathcal{G}}$: $\displaystyle\vskip-3.99994pt\tilde{\mathcal{G}}$ $\displaystyle=\bigl(\begin{bmatrix}\mathbf{X}_{agent}\\ \mathbf{x}_{task}^{\top}\end{bmatrix},\tilde{\mathbf{A}}_{anchor}\bigl)=(\tilde{\mathcal{V}},\tilde{\mathcal{E}})$ (8) $\displaystyle=\bigl(\mathcal{V}\cup\{v_{task}\},\mathcal{E}\cup\{\overset{\longleftrightarrow}{(v_{i},v_{task})}|v_{i}\in\mathcal{V})\}\bigl),$ where $\scriptsize\begin{bmatrix}\mathbf{X}_{agent}\\ \mathbf{x}_{task}^{\top}\end{bmatrix}$ can also be jointly denoted as $\tilde{\mathbf{X}}$.

### 4.2 Designing Communication Topology

Building upon the task-specific multi-agent network $\tilde{\mathcal{G}}$, G-Designer seeks to establish a more fine-grained and precise communication topology $\mathcal{G}_{com}$. Drawing inspiration from the variational graph auto-encoder (VGAE) framework ([Kipf & Welling, 2016](https://arxiv.org/html/2410.11782v3#bib.bib24); [Zhao & Zhang, 2024](https://arxiv.org/html/2410.11782v3#bib.bib64)), G-Designer employs a VGAE-based encoder-decoder $f_{v}$ to generate the multi-agent interaction topology: $\small\mathcal{G}_{com}=f_{v}(\tilde{\mathcal{G}};\Theta_{v})=p(\mathcal{G}_{com}\;|\;\mathbf{H})q(\mathbf{H}\;|\;\tilde{\mathbf{X}},\tilde{\mathbf{A}}_{anchor}),$ (9) where $f_{v}$ is the encoder-decoder architecture with parameters $\Theta_{v}$, $q(\cdot)$ is the encoder module, $p(\cdot)$ is the decoder module. The encoder utilizes posterior probabilities to encode the node embeddings into low-dimensional latent vector representations $\mathbf{H}_{agent}$, which can be formulated as: $\displaystyle q(\mathbf{H}_{agent}\;|\;\tilde{\mathbf{X}},\tilde{\mathbf{A}}_{anchor})=\prod_{i=1}^{N}q(\mathbf{h}_{i}\;|\;\tilde{\mathbf{X}},\tilde{\mathbf{A}}_{anchor}),$ (10) $\displaystyle q(\mathbf{h}_{i}\;|\;\tilde{\mathbf{X}},\tilde{\mathbf{A}}_{anchor})=\mathcal{N}(\mathbf{h}_{i}\;|\;\boldsymbol{\mu}_{i},\operatorname{diag}(\boldsymbol{\sigma}_{i}^{2})),$ where $\mu=\operatorname{GNN}_{\mu}(\tilde{\mathbf{X}},\tilde{\mathbf{A}}_{anchor};\Theta_{\mu})$ is the matrix of mean vectors $\mu_{i}$; similarly $\log(\sigma)=\operatorname{GNN}_{\sigma}(\tilde{\mathbf{X}},\tilde{\mathbf{A}}_{anchor};\Theta_{\sigma})$. The choice of GNN backbone can be customized as needed; here, we utilize a simple two-layer GCN ([Kipf & Welling, 2017](https://arxiv.org/html/2410.11782v3#bib.bib25)). $\mathbf{h}_{i}$, $\boldsymbol{\mu}_{i}$, and $\boldsymbol{\sigma}_{i}$ denote the $i$-th column of $\mathbf{H}$, $\boldsymbol{\mu}$, and $\boldsymbol{\sigma}$, respectively. The encoder $q(\cdot)$ is parameterized by $\Theta_{e}=\{\Theta_{\mu},\Theta_{\sigma}\}$. Following the encoding phase, the decoder employs the latent representations to generate a comprehensive blueprint for multi-agent communication. More specifically, the decoder $q(\cdot)=q_{c}\circ q_{s}$ first constructs a parameterized, sketched graph $\mathbf{S}$, which is then refined into the final multi-agent communication topology: $p(\mathcal{G}_{com}\;|\;\mathbf{H}_{agent})=\int_{\mathbf{S}}p_{c}(\mathcal{G}_{com}\;|\;\mathbf{S})p_{s}(\mathbf{S}\;|\;\mathbf{H}_{agent})\;d\mathbf{S}.$ (11) At the first step, $p_{s}(\cdot)$ constructs the sketched adjacency matrix $\mathbf{S}$ from the latent representation $\mathbf{H}_{agent}$: $\small p_{s}(\mathbf{S}\;|\;\mathbf{H}_{agent})=\prod_{i=1}^{N}\prod_{j=1}^{N}p_{s}(\mathbf{S}_{ij}\;|\;\mathbf{h}_{i},\mathbf{h}_{j},\mathbf{h}_{task};\Theta_{d}),$ (12) whose detailed derivation is as follows: $\displaystyle p_{s}(\mathbf{S}_{ij}=1\;|\;\mathbf{h}_{i},\mathbf{h}_{j},\mathbf{h}_{task})=g(\mathbf{h}_{i},\mathbf{h}_{j},\mathbf{h}_{task}),$ (13) $\displaystyle=\operatorname{Sigmoid}((\log(\epsilon)-\log(1-\epsilon)+\varpi_{ij})/\tau),$ where $\varpi=\operatorname{FFN}_{d}([\mathbf{h}_{i},\mathbf{h}_{j},\mathbf{h}_{task}])$ with $\operatorname{FFN}_{d}$ parameterized by $\Theta_{d}$, $\epsilon\sim\operatorname{Uniform}(0,1)$, and $\tau$ denotes the temperature coefficient. When $\tau$ approaches zero, [Equation 13](https://arxiv.org/html/2410.11782v3#S4.E13) essentially return the Bernouli sampling result for $\mathbf{S}_{ij}$. The resulting matrix $\mathbf{S}\in[0,1]^{N\times N}$ represents a densely-connected, non-negative graph distribution, indicating an overly complex and resource-intensive pair-wise communication structure, which is not yet suitable for guiding multi-agent collaboration. To align with G-Designer’s objectives of task adaptiveness and minimizing costs, we apply a refinement decoder $p_{c}(\cdot)$ to refine the sketched $\mathbf{S}$ into a compact, sparse, and highly informative communication graph, instantiated by a regularization objective: $\displaystyle p_{c}:\;\underset{\tilde{\mathbf{S}}\in\mathbb{S}}{\arg\max}\;1/2||\mathbf{S}-\mathbf{Z}\mathbf{W}\mathbf{Z}^{\top}||^{2}_{F}+\zeta||\mathbf{W}||_{*}+$ (14) $\displaystyle 1/2||\mathbf{A}_{anchor}-\mathbf{Z}\mathbf{W}\mathbf{Z}^{\top}||^{2}_{F},\;\operatorname{s.t.}\;\tilde{\mathbf{S}}=\mathbf{Z}\mathbf{W}\mathbf{Z}^{\top},$ where $\mathbf{Z}\in\mathbb{R}^{N\times r}$ is the top-$r$ columns of left singular matrix $\mathbf{S}$, $\zeta$ is a coefficient hyperparameter, $\mathbf{W}\in\mathbb{R}^{r\times r}$ is an optimizable weight matrix, $||\cdot||_{F}$ denotes the Frobenius norm and $||\mathbf{W}||_{*}=\sum_{i}\lambda_{i}$ where $\lambda_{i}$ is the $i$-th singular value of $\mathbf{W}$. $\tilde{\mathbf{S}}\in\mathbb{R}^{N\times N}$ is the desired sparse topology, which is decomposed as $\mathbf{Z}\mathbf{W}\mathbf{Z}^{\top}$. In [Equation 14](https://arxiv.org/html/2410.11782v3#S4.E14), the first and second terms are jointly denoted as anchor regularization, which encourage the learned $\tilde{\mathbf{S}}$ to maintain similarity with both the original $\mathbf{S}$ and the anchor topology. The third term, denoted as sparsity regularization, though appearing to minimize the nuclear norm of $\mathbf{W}$, essentially sparsifies $\tilde{\mathbf{S}}$, since $||\tilde{\mathbf{S}}||_{*}=||\mathbf{W}||_{*}$ holds due to $\mathbf{Z}^{\top}\mathbf{Z}=\mathbbm{I}_{r\times r}$. Therefore, [Equation 14](https://arxiv.org/html/2410.11782v3#S4.E14) achieves two key goals: (1) producing a sparse, refined communication topology, and (2) constraining the design to remain grounded in practical intuition. The resulting communication can be represented as follows: $\small\mathcal{G}_{com}=(\mathcal{V},\mathcal{E}_{com}),\mathcal{E}_{com}=\{(i,j)\;|\;\tilde{\mathbf{S}}_{ij}\neq 0\wedge(i,j)\in\mathcal{E}\}).$ (15) At this stage, we have successfully distilled a lightweight and informative collaboration network $\mathcal{G}_{com}$ from the sketched task-specific network $\tilde{\mathcal{G}}$, which is now ready to guide inter-agent message passing in the following process.

Table: Table 1: Performance comparison with three types of baselines, including single-agent execution, spatial communication, and temporal communication. The best results are in bold, and the runner-ups are underlined. All methods, except for the single-agent category, utilize five gpt-4-based agents. “Mul.”, “Ada.”, and “Rob.” indicate whether the method supports a multi-agent setting, whether it is task-adaptive, and whether it is adversarially robust, respectively. ✗, ✓✗ and ✓ signifies no/partial/full support in these aspects.

| Method | Mul. | Ada. | Rob. | MMLU | GSM8K | MultiArith | SVAMP | AQuA | HumanEval | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | ✗ | ✗ | ✗ | 82.14 | 85.40 | 93.15 | 87.18 | 70.34 | 71.68 | 81.65 |
| CoT | ✗ | ✗ | ✗ | 82.65↑0.51 | 87.17↑1.77 | 94.79↑1.64 | 88.32↑1.14 | 73.91↑3.57 | 75.52↑3.84 | 83.73 |
| ComplexCoT | ✗ | ✗ | ✗ | 83.78↑1.64 | 87.62↑2.22 | 95.86↑2.71 | 90.17↑2.99 | 77.58↑7.24 | 74.94↑3.26 | 84.99 |
| SC (CoT) | ✗ | ✗ | ✗ | 82.66↑0.52 | 87.93↑2.53 | 96.88↑3.73 | 88.69↑1.51 | 75.08↑4.74 | 77.30↑5.62 | 84.75 |
| SC (ComplexCoT) | ✗ | ✗ | ✗ | 83.65↑1.51 | 86.14↓0.74 | 96.94↑3.79 | 89.72↑2.54 | 77.69↑7.35 | 77.94↑6.26 | 85.35 |
| PHP | ✓ | ✗ | ✗ | 83.45↑1.31 | 95.50↑10.1 | 98.10↑2.84 | 90.02↑3.44 | 79.00↑8.66 | 82.96↑11.36 | 88.17 |
| Chain | ✓ | ✗ | ✗ | 82.35↑0.21 | 85.57↑0.17 | 94.38↑1.23 | 83.41↓3.77 | 70.94↑0.60 | 80.88↑9.20 | 82.92 |
| Star | ✓ | ✗ | ✗ | 80.79↓1.35 | 85.55↑0.15 | 93.79↓0.64 | 88.09↑0.91 | 68.57↓1.77 | 75.65↑3.97 | 82.07 |
| Tree | ✓ | ✗ | ✗ | 81.89↓0.25 | 84.56↓0.84 | 94.60↑1.45 | 89.25↑2.07 | 72.84↑2.50 | 77.38↑5.70 | 83.42 |
| Complete Graph | ✓ | ✗ | ✗ | 83.15↑1.01 | 86.49↑1.09 | 97.20↑4.05 | 89.48↑2.30 | 79.21↑8.87 | 83.75↑12.07 | 86.55 |
| Random Graph | ✓ | ✗ | ✗ | 83.76↑1.62 | 86.14↑0.74 | 95.46↑2.31 | 85.41↓1.77 | 74.07↑3.73 | 82.66↑10.98 | 84.58 |
| AutoGen | ✓ | ✗ | ✗ | 82.13↓0.01 | 90.06↑7.92 | 93.80↑0.65 | 88.44↓1.26 | 73.65↑3.31 | 85.41↑13.73 | 85.58 |
| MetaGPT | ✓ | ✗ | ✗ | - | - | - | - | - | 85.90↑14.22 | 84.90 |
| LLM-Blender | ✓ | ✗ | ✗ | 81.22↓0.92 | 89.17↑3.77 | 94.27↑1.12 | 88.77↑1.59 | 77.05↑6.71 | - | 86.09 |
| LLM-Debate | ✓ | ✗ | ✓ | 83.69↑1.55 | 90.23↑4.83 | 96.27↑3.12 | 90.56↑3.38 | 77.52↑7.18 | 83.79↑12.11 | 87.01 |
| DyLAN | ✓ | ✓✗ | ✓ | 80.16↓1.98 | 88.16↑2.76 | 94.27↑1.12 | 87.40↑0.22 | 74.16↑3.82 | 89.70↑18.02 | 85.64 |
| GPTSwarm | ✓ | ✓✗ | ✓ | 83.98↑1.84 | 89.74↑4.34 | 97.84↑4.69 | 86.42↓0.76 | 78.16↑7.82 | 88.49↑16.81 | 87.32 |
| G-Designer | ✓ | ✓ | ✓ | 84.50↑2.36 | 95.07↑9.67 | 98.30↑5.15 | 91.85↑4.67 | 79.47↑9.13 | 89.90↑18.22 | 89.84 |

### 4.3 Optimizing G-Designer

Upon obtaining $\mathcal{G}_{com}$, the multi-agent utterances and dialogues can proceed as usual using $\mathcal{G}_{com}$, as detailed in [Section 3.2](https://arxiv.org/html/2410.11782v3#S3.SS2). After $K$ rounds of interaction, the agents converge to a final solution $a^{(K)}=\mathcal{G}_{com}(\mathcal{Q})$. We then give the following optimization objective: $\underset{\Theta_{e},\Theta_{d}}{\arg\min}\;\mathbb{E}_{\Theta_{e},\Theta_{d}\sim\Omega}\Bigl[u\bigl(\mathcal{G}_{com}(\mathcal{Q})\bigl)\Bigl],$ (16) where $\Theta_{e}$ and $\Theta_{d}$ are the parameters of the encoder $q(\cdot)$ and decoder $p(\cdot)$, respectively, $\Omega$ is the parameter space and $\mathbb{E}(\cdot)$ denotes the mathematical expectation. [Equation 16](https://arxiv.org/html/2410.11782v3#S4.E16) aims to maximize the utility of the generated solution, but it is inherently intractable and non-differentiable, as $u(\cdot)$ often depends on external API calls ([Li et al., 2023b](https://arxiv.org/html/2410.11782v3#bib.bib28); [Hendrycks et al., 2021](https://arxiv.org/html/2410.11782v3#bib.bib15)). To address this, following standard approaches in multi-agent structure design ([Zhuge et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib68); [Zhang et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib60)), we apply policy gradient ([Williams, 1992](https://arxiv.org/html/2410.11782v3#bib.bib53)) to approximate and optimize [Equation 16](https://arxiv.org/html/2410.11782v3#S4.E16): $\small\nabla_{\Theta}\mathbb{E}_{\Theta\sim\Omega}\Bigl[u\bigl(\mathcal{G}_{com}(\mathcal{Q})\bigl)\Bigl]\approx\frac{1}{M}\sum_{k=1}^{M}u(a_{m}^{(K)})\nabla_{\Theta}(P(\mathcal{G}_{k})),$ (17) where $\mathbf{\Theta}=\{\Theta_{e},\Theta_{d}\}$, $\{\mathcal{G}_{k}\}_{m=1}^{M}$ are indepently samples from $\mathcal{G}_{com}$, and $\{a_{m}^{(K)}\}_{m=1}^{M}$ are the corresponding output. $P(\mathcal{G}_{k})$ calculates the probability of $\mathcal{G}_{k}$ being sampled, which can be expressed as $P(\mathcal{G}_{k})=\prod_{i=1}^{N}\prod_{j=1}^{N}\tilde{\mathbf{S}}_{ij}$. Through iterative optimization guided by [Equations 14](https://arxiv.org/html/2410.11782v3#S4.E14) and [16](https://arxiv.org/html/2410.11782v3#S4.E16) over a limited set of queries as the “training set”, G-Designer efficiently develops task-awareness and the capability to strategically design the agent network, achieving truly task-customized multi-agent topology design.

#### Optimization configuration

The overall training objective of our method is formulated as $\mathcal{L}_{{\textbf{G-Designer}}}=\mathcal{L}_{utility}+\mathcal{L}_{anchor}+\mathcal{L}_{sparse}$, where $\mathcal{L}_{utility}$ represents the optimization target from [Equation 16](https://arxiv.org/html/2410.11782v3#S4.E16), $\mathcal{L}_{anchor}$ corresponds to the first and third terms in [Equation 14](https://arxiv.org/html/2410.11782v3#S4.E14), and $\mathcal{L}_{sparse}$ is the second term. Given a benchmark $\{\mathcal{Q}_{i}\}_{i=1}^{D}$ consisting of $B$ queries, G-Designer begins by optimizing with a small subset of $B^{\prime}$ queries and fixes the learned parameters for testing on the remaining $(B-B^{\prime})$ queries. The whole algorithm workflow of G-Designer is depicted in Algorithm [1](https://arxiv.org/html/2410.11782v3#alg1).

## 5 Experiments

### 5.1 Experimental Setup

#### Datasets and Metrics

We evaluate G-Designer on three categories of datasets: $\blacksquare$ General Reasoning: MMLU ([Hendrycks et al., 2021](https://arxiv.org/html/2410.11782v3#bib.bib15)); $\blacksquare$ Mathematical Reasoning: GSM8K ([Cobbe et al., 2021](https://arxiv.org/html/2410.11782v3#bib.bib9)), MultiArith ([Roy & Roth, 2016](https://arxiv.org/html/2410.11782v3#bib.bib43)), SVAMP ([Patel et al., 2021](https://arxiv.org/html/2410.11782v3#bib.bib35)), and AQuA ([Ling et al., 2017](https://arxiv.org/html/2410.11782v3#bib.bib30)); $\blacksquare$ Code: HumanEval ([Chen et al., 2021](https://arxiv.org/html/2410.11782v3#bib.bib6)). We include the dataset statistics in [Table 4](https://arxiv.org/html/2410.11782v3#A2.T4).

#### Baselines

For single-agent approaches, we select COT ([Wei et al., 2022](https://arxiv.org/html/2410.11782v3#bib.bib52)), ComplexCoT ([Fu et al., 2022](https://arxiv.org/html/2410.11782v3#bib.bib12)), Self-Consistency ([Wang et al., 2023a](https://arxiv.org/html/2410.11782v3#bib.bib50)), and PHP ([Zheng et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib65)). For multi-agent topologies, we select Chain, Star, and Tree (formally defined in ([Qian et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib38))), Complete Graph and Random Graph , AutoGen ([Wu et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib54)), MetaGPT ([Hong et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib17)), LLM-Debate ([Du et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib11)), LLM-Blender ([Jiang et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib21)), DyLAN ([Liu et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib32)), and GPTSwarm ([Zhuge et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib68)).

#### Implementation Details

We access the GPT via the OpenAI API, and mainly test on gpt-4-1106-preview (gpt-4) and gpt-3.5-turbo-0125 (gpt-3.5). We set temperature to 0 for the single execution and single agent baselines and 1 for multi-agent methods. We set a summarizer agent to aggregate the dialogue history and produce the final solution $a^{(K)}$, with $K=3$ across all experiments. The $\operatorname{NodeEncoder}(\cdot)$ is implemented using all-MiniLM-L6-v2 ([Wang et al., 2020](https://arxiv.org/html/2410.11782v3#bib.bib49)), with the embedding dimension set to $D=384$. The anchor topology $\mathbf{A}_{anchor}$ is predefined as a simple chain structure. The sampling times $M$ are set as $10$, and $\tau=1e-2$ and $\zeta=1e-1$ are set for all experiments. We provide explicit agent profiling for multi-agent methods, following the classical configurations in LLM-MA systems ([Liu et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib32); [Zhuge et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib68); [Yin et al., 2023](https://arxiv.org/html/2410.11782v3#bib.bib59)), and use gpt-4 to generate agent profile pools. For all benchmarks, we merely use $B^{\prime}\in\{40,80\}$ queries for optimization.

### 5.2 Main Results

In this section, we conduct extensive experiments across six benchmarks to verify that G-Designer is:

#### High-performing

The experimental results from [Table 1](https://arxiv.org/html/2410.11782v3#S4.T1) demonstrate that G-Designer is effective in designing powerful LLM-MA topologies. Concretely, G-Designer achieves the best performance in five out of six benchmarks, and on GSM8K, it trails only PHP with a $9.67\%\uparrow$ accuracy improvement. On the HumanEval benchmark, G-Designer surpasses MetaGPT, a specialized multi-agent code generation framework, by $4.0\%$ at pass@1, and outperforms state-of-the-art multi-agent collaboration frameworks like GPTSwarm and DyLAN by margins of $0.20\%\sim 1.41\%$.

#### Task-adaptive

[Figure 6](https://arxiv.org/html/2410.11782v3#A3.F6) visualizes the different topologies designed by G-Designer for varying query difficulties on HumanEval and GSM8K. As shown in [Figure 6](https://arxiv.org/html/2410.11782v3#A3.F6), the multi-agent topologies generated by G-Designer are highly dependent on the specific task context and its difficulty. In Case a, despite having five gpt-4 agents available as design resources, G-Designer identified the task of designing a strlen(string) function as relatively simple. It streamlined the topology by removing unnecessary agents and retained only a minimal “Algorithm Designer $\rightarrow$ Programmer” structure to solve the problem. In contrast, for the more complex Case c and Case e, G-Designer crafted a more intricate communication graph. These cases highlight the strong task-adaptiveness of G-Designer.

#### Scalable

To evaluate the scalability of G-Designer to a larger number of agents, we report its performance across $5\sim 20$ agents, as presented in [Table 6](https://arxiv.org/html/2410.11782v3#A4.T6). Notably, G-Designer exhibits a steeper performance gain than GPTSwarm as the agent count increases. More importantly, while the complete graph and GPTSwarm incur an overwhelming token cost at $20$ agents ($5.6\sim 30.3$M tokens), G-Designer achieves superior performance with merely $6.11\%$ of GPTSwarm’s prompt token consumption, surpassing it by $2.44\%\uparrow$. These results decisively demonstrate the scalability and potential of G-Designer in advancing large-scale autonomous multi-agent systems.

Figure: Figure 4: Visualization of the performance metrics and prompt token consumption of different multi-agent communication topologies across MMLU, HumanEval, GSM8K, and SVAMP. The diameter of each point is proportional to its $y$-axis value. [image: https://arxiv.org/html/2410.11782v3/token.png]

#### Token-economical (inference)

A key benefit of G-Designer’s adaptivity is that it prevents the use of overly complex structures for simple tasks, thus minimizing unnecessary communication costs—in the case of LLM-MA, reducing token consumption. [Figure 4](https://arxiv.org/html/2410.11782v3#S5.F4) It illustrates the differences in prompt token consumption between G-Designer and several representative multi-agent designs. We observe that simpler topologies, such as complete graphs and random graphs, consume fewer tokens but show significantly weaker performance. More complex communication structures, like GPTSwarm and DyLAN, achieve superior performance, albeit at the cost of excessive token consumption. For instance, DyLAN’s cost on GSM8K is $2.82\times$ that of the random graph, reaching a substantial $2.2e+7$. In contrast, G-Designer elegantly balances both efficiency and task performance, achieving the highest performance across all four benchmarks while maintaining the lowest token cost. For example, on SVAMP, G-Designer surpasses DyLAN by $4\%$ while using only $23.7\%$ of DyLAN’s token cost.

#### Resource-efficient (training)

We validate G-Designer’s training process is resource-friendly from three dimensions: GPU cost, token cost, and wall-clock time. [Table 5](https://arxiv.org/html/2410.11782v3#A4.T5) showcases that training G-Designer with up to 1000 agents requires less than 4GB of memory. [Table 2](https://arxiv.org/html/2410.11782v3#S5.T2) unveils that G-Designer not only attains the highest accuracy but also exhibits superior token efficiency and reduced wall-clock time compared to existing baselines, underscoring its effectiveness in multi-agent collaboration.

Table: Table 2: Efficiency analysis. We compare the training/inference wall-clock time and token consumption between G-Designer and other high-performing baselines on the GSM8K dataset.

| Method | Perf. | #Training Token | #Inference Token | #Overall Token | Training Time | Inference Time |
| --- | --- | --- | --- | --- | --- | --- |
| Complete | 86.4 | - | $9.8\times 10^{6}$ | $9.8\times 10^{6}$ | - | 2.4h |
| DyLAN | 88.1 | $9.6\times 10^{6}$ | $1.3\times 10^{7}$ | $2.2\times 10^{7}$ | 2.8h | 4.6h |
| GPTSwarm | 89.7 | $5.5\times 10^{6}$ | $8.4\times 10^{6}$ | $1.4\times 10^{7}$ | 2.1h | 2.8h |
| G-Designer | 95.0 | $\mathbf{2.7\times 10^{5}}$ | $\mathbf{8.2\times 10^{6}}$ | $\mathbf{8.5\times 10^{6}}$ | 0.3h | 2.3h |

Figure: Figure 5: We compare the accuracy (%) of various multi-agent frameworks before and after prompt attacks on MMLU.

### 5.3 Robustness Analysis

Following ([Zhuge et al., 2024](https://arxiv.org/html/2410.11782v3#bib.bib68)), we simulate a system prompt attack on one of the five agents. As seen in [Figure 5](https://arxiv.org/html/2410.11782v3#S5.F5), many trivial structures, such as chain or complete graph, experience significant performance degradation under partial system attacks, with drops as high as $11.0\%$. Among more sophisticated structures, GPTSwarm, benefiting from its specialized node optimization mechanism, only suffers a minor $0.3\%$ accuracy decline. However, other methods fare less well, with DyLAN and AutoGen showing accuracy drops of $6.2\%$ and $9.9\%$, respectively. Remarkably, G-Designer demonstrates exceptional robustness against adversarial attacks, maintaining nearly identical performance pre- and post-attack. This resilience can be attributed to its agent encoding capability, which, during optimization, can detect malicious inputs and prune the corresponding edges.

### 5.4 Framework Analysis

Table: Table 3: Ablation study tested on MMLU benchmark.

| Variant | MMLU | GSM8K |  |  |
| --- | --- | --- | --- | --- |
| Clean | Attack | Clean | Attack |  |
| vanilla G-Designer | 84.5 | 84.2 | 95.0 | 92.5 |
| w/o SR | 84.1 | 83.2 | 94.4 | 90.7 |
| w/o Anchor | 84.0 | 83.8 | 94.7 | 92.0 |
| w/o $\operatorname{NodeEncoder(\cdot)}$ | 83.2 | 82.4 | 92.8 | 87.4 |
| w/o $v_{\text{task}}$ | 81.3 | 82.0 | 90.3 | 87.7 |

#### Ablation Study.

We report results for two variants of ourmethod: (1) w/o SR, which removes the sparsity regularization in [Equation 14](https://arxiv.org/html/2410.11782v3#S4.E14), (2) w/o Anchor, which excludes the anchor structure $\mathbf{A}_{anchor}$, (3) w/o NodeEncoder, removing node encoder in [Equation 7](https://arxiv.org/html/2410.11782v3#S4.E7), and (4) w/o $v_{\text{task}}$ in [Equation 8](https://arxiv.org/html/2410.11782v3#S4.E8). As shown in [Table 3](https://arxiv.org/html/2410.11782v3#S5.T3), removing the task virtual node disrupts G-Designer’s task-adaptiveness, leading to the most significant performance drop. The removal of $\mathbf{A}_{anchor}$ consistently leads to performance degradation, while the absence of sparsity regularization makes the system more vulnerable to adversarial attacks.

#### Discussion on anchor topology.

Given that G-Designer is initialized with the anchor topology $\mathbf{A}_{\text{anchor}}$ introduced in [Section 4.1](https://arxiv.org/html/2410.11782v3#S4.SS1), one may question whether the performance gains of G-Designer primarily stem from $\mathbf{A}_{\text{anchor}}$ itself. In response, we emphasize that the anchor topology corresponds to the simple Chain structure in [Table 1](https://arxiv.org/html/2410.11782v3#S4.T1), where G-Designer achieves substantial improvements over it, specifically $9.50\%\uparrow$ on GSM8K and $8.44\%\uparrow$ on SVAMP. Thus, we assert that the superior performance of G-Designer is predominantly attributed to its adaptive topology design rather than the anchor topology itself.

## 6 Conclusion

In this paper, we present the LLM-based Multi-agent Communication Protocol (MACP), which aims to provide insightful guidance for designing complex multi-agent systems. Furthermore, we propose an effective, adaptive, and robust LLM-powered multi-agent communication graph designer, termed G-Designer, to facilitate the automated design of collaborative AI systems. G-Designer is highly task-aware, dynamically crafting compact and robust communication topologies based on task complexity. We hope that G-Designer will inspire future research on self-organizing and self-evolving collective intelligence.

## Impact Statement

#### Ethical impacts.

We confidently affirm that our paper is free of ethical concerns across its motivation, design, experiments, and data usage. The proposed G-Designer method aims to advance the fields of multi-agent systems and communication topologies design automation, contributing positively and responsibly to the scientific community.

#### Expected societal implications.

Our paper presents a significant advancement in multi-agent systems. G-Designer addresses the challenge of choosing the right communication topology for specific tasks. G-Designer not only provides a practical solution for multi-agent deployment but also paves the way for future studies on collective intelligence systems. Furthermore, as long as the base LLM used is aligned with human values, our system will not generate harmful content.

## References

- Bouzenia et al. (2024) Bouzenia, I., Devanbu, P., and Pradel, M. Repairagent: An autonomous, llm-based agent for program repair. *arXiv preprint arXiv:2403.17134*, 2024.

- Chan et al. (2023) Chan, C.-M., Chen, W., Su, Y., Yu, J., Xue, W., Zhang, S., Fu, J., and Liu, Z. ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate. *arXiv e-prints*, August 2023.

- Chen et al. (2023a) Chen, D., Wang, H., Huo, Y., Li, Y., and Zhang, H. Gamegpt: Multi-agent collaborative framework for game development. *arXiv preprint arXiv:2310.08067*, 2023a.

- Chen et al. (2024a) Chen, J., Lin, H., Han, X., and Sun, L. Benchmarking large language models in retrieval-augmented generation. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 38, pp. 17754–17762, 2024a.

- Chen et al. (2024b) Chen, L., Davis, J. Q., Hanin, B., Bailis, P., Stoica, I., Zaharia, M., and Zou, J. Are more llm calls all you need? towards scaling laws of compound inference systems. *arXiv preprint arXiv:2403.02419*, 2024b.

- Chen et al. (2021) Chen, M., Tworek, J., Jun, H., Yuan, Q., Ponde de Oliveira Pinto, H., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Petroski Such, F., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Hebgen Guss, W., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba, W. Evaluating large language models trained on code, July 01, 2021 2021.

- Chen et al. (2023b) Chen, W., Su, Y., Zuo, J., Yang, C., Yuan, C., Qian, C., Chan, C.-M., Qin, Y., Lu, Y., Xie, R., Liu, Z., Sun, M., and Zhou, J. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors in agents, 2023b.

- Cheng et al. (2024) Cheng, Y., Zhang, C., Zhang, Z., Meng, X., Hong, S., Li, W., Wang, Z., Wang, Z., Yin, F., Zhao, J., and He, X. Exploring large language model based intelligent agents: Definitions, methods, and prospects. *CoRR*, abs/2401.03428, 2024.

- Cobbe et al. (2021) Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems. *arXiv prepring*, abs/2110.14168, 2021.

- Cohen et al. (2023) Cohen, R., Hamri, M., Geva, M., and Globerson, A. Lm vs lm: Detecting factual errors via cross examination. *arXiv preprint arXiv:2305.13281*, 2023.

- Du et al. (2023) Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch, I. Improving factuality and reasoning in language models through multiagent debate. *CoRR*, abs/2305.14325, 2023.

- Fu et al. (2022) Fu, Y., Peng, H., Sabharwal, A., Clark, P., and Khot, T. Complexity-based prompting for multi-step reasoning. In *The Eleventh International Conference on Learning Representations*, 2022.

- Gao et al. (2023) Gao, C., Lan, X., Li, N., Yuan, Y., Ding, J., Zhou, Z., Xu, F., and Li, Y. Large language models empowered agent-based modeling and simulation: A survey and perspectives. *CoRR*, abs/2312.11970, 2023.

- Hao et al. (2023) Hao, R., Hu, L., Qi, W., Wu, Q., Zhang, Y., and Nie, L. Chatllm network: More brains, more intelligence, April 01, 2023 2023.

- Hendrycks et al. (2021) Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding. *Proceedings of the International Conference on Learning Representations (ICLR)*, 2021.

- Holt et al. (2024) Holt, S., Luyten, M. R., and van der Schaar, M. L2mac: Large language model automatic computer for extensive code generation. In *The Twelfth International Conference on Learning Representations*, 2024.

- Hong et al. (2023) Hong, S., Zheng, X., Chen, J., Cheng, Y., Wang, J., Zhang, C., Wang, Z., Yau, S. K. S., Lin, Z., Zhou, L., Ran, C., Xiao, L., and Wu, C. Metagpt: Meta programming for multi-agent collaborative framework, August 01, 2023 2023.

- Hu et al. (2024) Hu, S., Shen, L., Zhang, Y., and Tao, D. Learning multi-agent communication from graph modeling perspective. *arXiv preprint arXiv:2405.08550*, 2024.

- Hua et al. (2023) Hua, W., Fan, L., Li, L., Mei, K., Ji, J., Ge, Y., Hemphill, L., and Zhang, Y. War and peace (waragent): Large language model-based multi-agent simulation of world wars. *arXiv preprint arXiv:2311.17227*, 2023.

- Ishibashi & Nishimura (2024) Ishibashi, Y. and Nishimura, Y. Self-organized agents: A llm multi-agent framework toward ultra large-scale code generation and optimization. *arXiv preprint arXiv:2404.02183*, 2024.

- Jiang et al. (2023) Jiang, D., Ren, X., and Lin, B. Y. LLM-blender: Ensembling large language models with pairwise ranking and generative fusion. In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 14165–14178, Toronto, Canada, July 2023. Association for Computational Linguistics.

- Jin et al. (2023) Jin, Y., Shen, X., Peng, H., Liu, X., Qin, J., Li, J., Xie, J., Gao, P., Zhou, G., and Gong, J. Surrealdriver: Designing generative driver agent simulation framework in urban contexts based on large language model, 2023.

- Khattab et al. (2023) Khattab, O., Singhvi, A., Maheshwari, P., Zhang, Z., Santhanam, K., Vardhamanan, S., Haq, S., Sharma, A., Joshi, T. T., Moazam, H., et al. Dspy: Compiling declarative language model calls into self-improving pipelines. *arXiv preprint arXiv:2310.03714*, 2023.

- Kipf & Welling (2016) Kipf, T. N. and Welling, M. Variational graph auto-encoders. *arXiv preprint arXiv:1611.07308*, 2016.

- Kipf & Welling (2017) Kipf, T. N. and Welling, M. Semi-supervised classification with graph convolutional networks. In *5th International Conference on Learning Representations*. OpenReview.net, 2017.

- Li et al. (2023a) Li, G., Hammoud, H., Itani, H., Khizbullin, D., and Ghanem, B. CAMEL: communicative agents for ”mind” exploration of large language model society. In *NeurIPS*, 2023a.

- Li et al. (2024) Li, J., Zhang, Q., Yu, Y., Fu, Q., and Ye, D. More agents is all you need. *CoRR*, abs/2402.05120, 2024.

- Li et al. (2023b) Li, M., Zhao, Y., Yu, B., Song, F., Li, H., Yu, H., Li, Z., Huang, F., and Li, Y. Api-bank: A comprehensive benchmark for tool-augmented llms. *arXiv preprint arXiv:2304.08244*, 2023b.

- Liang et al. (2023) Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Tu, Z., and Shi, S. Encouraging divergent thinking in large language models through multi-agent debate. *CoRR*, abs/2305.19118, 2023.

- Ling et al. (2017) Ling, W., Yogatama, D., Dyer, C., and Blunsom, P. Program induction by rationale generation: Learning to solve and explain algebraic word problems. *arXiv preprint arXiv:1705.04146*, 2017.

- Liu et al. (2022) Liu, Y., Dou, Y., Li, Y., Xu, X., and Liu, D. Temporal dynamic weighted graph convolution for multi-agent reinforcement learning. In *Proceedings of the Annual Meeting of the Cognitive Science Society*, volume 44, 2022.

- Liu et al. (2023) Liu, Z., Zhang, Y., Li, P., Liu, Y., and Yang, D. Dynamic llm-agent network: An llm-agent collaboration framework with agent team optimization. *CoRR*, abs/2310.02170, 2023.

- Ma et al. (2023) Ma, K., Zhang, H., Wang, H., Pan, X., Yu, W., and Yu, D. Laser: Llm agent with state-space exploration for web navigation. *arXiv preprint arXiv:2309.08172*, 2023.

- Nakajima (2023) Nakajima, Y. Babyagi. [https://github.com/yoheinakajima/babyagi](https://github.com/yoheinakajima/babyagi), 2023.

- Patel et al. (2021) Patel, A., Bhattamishra, S., and Goyal, N. Are nlp models really able to solve simple math word problems? *arXiv preprint arXiv:2103.07191*, 2021.

- Pesce & Montana (2023) Pesce, E. and Montana, G. Learning multi-agent coordination through connectivity-driven communication. *Machine Learning*, 112(2):483–514, 2023.

- Qian et al. (2023) Qian, C., Cong, X., Yang, C., Chen, W., Su, Y., Xu, J., Liu, Z., and Sun, M. Communicative agents for software development, July 01, 2023 2023. 25 pages, 9 figures, 2 tables.

- Qian et al. (2024) Qian, C., Xie, Z., Wang, Y., Liu, W., Dang, Y., Du, Z., Chen, W., Yang, C., Liu, Z., and Sun, M. Scaling large-language-model-based multi-agent collaboration. *arXiv preprint arXiv:2406.07155*, 2024.

- Reimers (2019) Reimers, N. Sentence-bert: Sentence embeddings using siamese bert-networks. *arXiv preprint arXiv:1908.10084*, 2019.

- Reworkd (2023) Reworkd. Agentgpt. [https://github.com/reworkd/AgentGPT](https://github.com/reworkd/AgentGPT), 2023.

- Richards & et al. (2023) Richards, T. B. and et al. Auto-gpt: An autonomous gpt-4 experiment. [https://github.com/Significant-Gravitas/Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT), 2023.

- Rosenbluth et al. (2024) Rosenbluth, E., Tönshoff, J., Ritzert, M., Kisin, B., and Grohe, M. Distinguished in uniform: Self attention vs. virtual nodes. *arXiv preprint arXiv:2405.11951*, 2024.

- Roy & Roth (2016) Roy, S. and Roth, D. Solving general arithmetic word problems. *arXiv preprint arXiv:1608.01413*, 2016.

- Shinn et al. (2023) Shinn, N., Labash, B., and Gopinath, A. Reflexion: an autonomous agent with dynamic memory and self-reflection. *arXiv preprint*, abs/2303.11366, 2023. doi: 10.48550/arXiv.2303.11366. URL [https://doi.org/10.48550/arXiv.2303.11366](https://doi.org/10.48550/arXiv.2303.11366).

- Shirzad et al. (2023) Shirzad, H., Velingker, A., Venkatachalam, B., Sutherland, D. J., and Sinop, A. K. Exphormer: Sparse transformers for graphs. In *International Conference on Machine Learning*, pp. 31613–31632. PMLR, 2023.

- Tan et al. (2023) Tan, Z., Guo, R., Ding, K., and Liu, H. Virtual node tuning for few-shot node classification. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pp. 2177–2188, 2023.

- Wang et al. (2023) Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., and Anandkumar, A. Voyager: An Open-Ended Embodied Agent with Large Language Models. *arXiv e-prints*, art. arXiv:2305.16291, May 2023.

- Wang et al. (2024) Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., Tang, J., Chen, X., Lin, Y., Zhao, W. X., Wei, Z., and Wen, J. A survey on large language model based autonomous agents. *Front. Comput. Sci.*, 18, 2024.

- Wang et al. (2020) Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., and Zhou, M. Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. *Advances in Neural Information Processing Systems*, 33:5776–5788, 2020.

- Wang et al. (2023a) Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. In *The Eleventh International Conference on Learning Representations*, 2023a.

- Wang et al. (2023b) Wang, Z., Mao, S., Wu, W., Ge, T., Wei, F., and Ji, H. Unleashing cognitive synergy in large language models: A task-solving agent through multi-persona self-collaboration, July 01, 2023 2023b. work in progress.

- Wei et al. (2022) Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., and Zhou, D. Chain-of-thought prompting elicits reasoning in large language models, January 01, 2022 2022.

- Williams (1992) Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine learning*, 8:229–256, 1992.

- Wu et al. (2023) Wu, Q., Bansal, G., Zhang, J., Wu, Y., Zhang, S., Zhu, E., Li, B., Jiang, L., Zhang, X., and Wang, C. Autogen: Enabling next-gen llm applications via multi-agent conversation framework, August 01, 2023 2023.

- Xi et al. (2023) Xi, Z., Chen, W., Guo, X., He, W., Ding, Y., Hong, B., Zhang, M., Wang, J., Jin, S., Zhou, E., Zheng, R., Fan, X., Wang, X., Xiong, L., Zhou, Y., Wang, W., Jiang, C., Zou, Y., Liu, X., Yin, Z., Dou, S., Weng, R., Cheng, W., Zhang, Q., Qin, W., Zheng, Y., Qiu, X., Huan, X., and Gui, T. The rise and potential of large language model based agents: A survey. *arxiv preprint*, abs/2309.07864, 2023.

- Yan et al. (2024) Yan, Y., Zhang, Y., and Huang, K. Depending on yourself when you should: Mentoring llm with rl agents to become the master in cybersecurity games. *arXiv preprint arXiv:2403.17674*, 2024.

- Yao et al. (2023a) Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models, May 01, 2023 2023a.

- Yao et al. (2023b) Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In *The Eleventh International Conference on Learning Representations*, 2023b.

- Yin et al. (2023) Yin, Z., Sun, Q., Chang, C., Guo, Q., Dai, J., Huang, X.-J., and Qiu, X. Exchange-of-thought: Enhancing large language model capabilities through cross-model communication. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pp. 15135–15153, 2023.

- Zhang et al. (2024) Zhang, G., Yue, Y., Li, Z., Yun, S., Wan, G., Wang, K., Cheng, D., Yu, J. X., and Chen, T. Cut the crap: An economical communication pipeline for llm-based multi-agent systems. *arXiv preprint arXiv:2410.02506*, 2024.

- Zhang et al. (2023a) Zhang, J., Xu, X., and Deng, S. Exploring collaboration mechanisms for llm agents: A social psychology view. *arXiv preprint arXiv:2310.02124*, 2023a.

- Zhang et al. (2023b) Zhang, J., Xu, X., and Deng, S. Exploring collaboration mechanisms for llm agents: A social psychology view. *arXiv preprint arXiv:2310.02124*, 2023b.

- Zhang & Chartrand (2006) Zhang, P. and Chartrand, G. Introduction to graph theory. *Tata McGraw-Hill*, 2:2–1, 2006.

- Zhao & Zhang (2024) Zhao, K. and Zhang, L. Causality-inspired spatial-temporal explanations for dynamic graph neural networks. In *The Twelfth International Conference on Learning Representations*, 2024.

- Zheng et al. (2023) Zheng, C., Liu, Z., Xie, E., Li, Z., and Li, Y. Progressive-hint prompting improves reasoning in large language models, April 01, 2023 2023. Tech Report.

- Zhou et al. (2023) Zhou, Z., Hu, B., Zhao, C., Zhang, P., and Liu, B. Large language model as a policy teacher for training reinforcement learning agents. *arXiv preprint arXiv:2311.13373*, 2023.

- Zhuang et al. (2023) Zhuang, Y., Chen, X., Yu, T., Mitra, S., Bursztyn, V., Rossi, R. A., Sarkhel, S., and Zhang, C. Toolchain*: Efficient action space navigation in large language models with a* search. *arXiv preprint arXiv:2310.13227*, 2023.

- Zhuge et al. (2024) Zhuge, M., Wang, W., Kirsch, L., Faccio, F., Khizbullin, D., and Schmidhuber, J. Gptswarm: Language agents as optimizable graphs. In *Forty-first International Conference on Machine Learning*, 2024.

We conclude the dataset statistics in [Table 4](https://arxiv.org/html/2410.11782v3#A2.T4).

Table: Table 4: Dataset descriptions and statistics.

| Category | Dataset | Answer Type | Metric | #Test | License |
| --- | --- | --- | --- | --- | --- |
| General reasoning | MMLU | Multi-choice | Acc. | 153 | MIT License |
| Math reasoning | GSM8K | Number | Acc. | 1,319 | MIT License |
| MultiArith | Number | Acc. | 600 | Unspecified |  |
| SVAMP | Number | Acc. | 1,000 | MIT License |  |
| AQuA | Multi-choice | Acc. | 254 | Apache-2.0 |  |
| Code generation | HumanEval | Code | Pass@1 | 164 | MIT License |

Figure: Figure 6: Case study of the communication topologies designed by G-Designer on HumanEval and GSM8K benchmarks. [image: https://arxiv.org/html/2410.11782v3/case.png]

[Figure 6](https://arxiv.org/html/2410.11782v3#A3.F6) visualizes the different topologies designed by G-Designer for varying query difficulties on the HumanEval and GSM8K benchmarks.

Table: Table 5: The GPU cost of G-Designer with increasing number of agents.

| #Agents | 5 | 50 | 100 | 1000 |
| --- | --- | --- | --- | --- |
| Memory (GB) | 2.7 | 2.9 | 3.0 | 3.8 |

Table: Table 6: Comparison of accuracy, time, token consumption, and cost across different agent configurations. We use the MMLU benchmark and gpt-3.5-turbo as the base LLM.

| #Agents | 5 | 10 | 20 |
| --- | --- | --- | --- |
| Chain |  |  |  |
| Accuracy (%) | 70.59 | 71.24 | 71.98 |
| Time (min) | 15.73 | 30.20 | 56.18 |
| #Prompt Tokens | 351,802 | 702,164 | 1,378,328 |
| Cost (USD) | 0.5228 | 1.0434 | 2.0482 |
| Complete Graph |  |  |  |
| Accuracy (%) | 71.90 | 72.16 | 72.51 |
| Time (min) | 16.85 | 34.21 | 66.47 |
| #Prompt Tokens | 545,984 | 1,669,451 | 5,648,834 |
| Cost (USD) | 0.7161 | 2.1770 | 7.3662 |
| GPTSwarm |  |  |  |
| Accuracy (%) | 72.55 | 73.86 | 75.38 |
| Time (min) | 62.14 | 186.86 | 412.18 |
| #Prompt Tokens | 3,055,236 | 9,048,465 | 30,317,341 |
| Cost (USD) | 4.2190 | 12.4961 | 41.4235 |
| G-Designer |  |  |  |
| Accuracy (%) | 73.20 | 74.51 | 77.82 |
| Time (min) | 19.26 | 36.04 | 68.89 |
| #Prompt Tokens | 452,329 | 885,332 | 1,852,538 |
| Cost (USD) | 0.6036 | 1.2768 | 2.6713 |

langley00
