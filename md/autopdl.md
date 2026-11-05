<!-- Source: https://raw.githubusercontent.com/mlresearch/v293/main/assets/spiess25a/spiess25a.pdf; retrieved: 2026-09-28T19:18:07.714671+00:00; conversion: pypdf layout text extraction. Equations, tables, and reading order require checking against source.pdf. -->

## PDF page 1

AutoPDL:AutomaticPromptOptimizationforLLMAgents

      ClaudioSpiess1  MandanaVaziri2  LouisMandel2  MartinHirzel2

      1UC Davis
      2IBM Research

     Abstract  The performance of large language models (LLMs) depends on how they are prompted,
                 with choices spanning both the high-level prompting pattern (e.g., Zero-Shot, CoT, ReAct,
                 ReWOO)andthespecificpromptcontent(instructionsandfew-shotdemonstrations). Man-
                 uallytuningthiscombinationistedious,error-prone,andspecifictoagivenLLMandtask.
                 Therefore,thispaperproposesAutoPDL,anautomatedapproachtodiscoveringgoodLLM
                 agent configurations.  Our approach frames this as a structured AutoML problem over a
                 combinatorial space of agentic and non-agentic prompting patterns and demonstrations,
                 using successive halving to efficiently navigate this space.  We introduce a library im-
                 plementing common prompting patterns using the PDL prompt programming language.
                 AutoPDL solutions are human-readable, editable, and executable PDL programs that use
                 this library.  This approach also enables source-to-source optimization, allowing human-
                 in-the-looprefinementandreuse. EvaluationsacrossthreetasksandsevenLLMs(ranging
                 from3Bto70Bparameters)showconsistentaccuracygains(9.06±15.3percentagepoints),
                 up to 68.9pp, and reveal that selected prompting strategies vary across models and tasks.


1Introduction
   Largelanguagemodels(LLMs)andLLM-basedagentsexcelatavarietyoftasks,includingquestion
   answering,mathwordproblems,andprogramming. TheperformanceofanLLMdependsheavily
   on how it is prompted, and there are a variety of popular prompting patterns.  These include
   zero-shotorfew-shot(Brownetal.2020)promptingwithchain-of-thought(CoT)(Weietal.2022),
   zero-shot CoT (Kojima et al. 2022), as well as agentic patterns such as ReAct (Yao et al. 2023) or
   ReWOO (Xu et al. 2023).  However, given a dataset         𝐷test with a loss functionL, e.g., error rate,
   it is not clear which pattern will do best. Furthermore, besides the pattern 𝐴, performance also
   dependsontheprompt𝑝,includingfew-shotsamplesandinstructions. Theproblemisthustofind
   a combination𝐴∗𝑝 of a pattern along with an optimized prompt that minimizesL. This is usually
   doneviamanualpromptengineering,butthatistediousandhastoberepeatedifanewLLMcomes
   along. Therefore,thispaperexploreshowtofind𝐴∗𝑝 usingautomatedmachinelearning(AutoML).
   And for users to trust the result or to tweak it further,𝐴∗𝑝 itself should be easy to read and edit.
       Agentframeworks,suchasCrewAI(Moura2023)orAutoGen(Q.Wuetal.2023),containpre-
   built agent patterns with prompts optimized for proprietary frontier models and common tasks.
   Unfortunately,theirpromptsaredeeplyburied(Schluntzetal.2024),makingthemhardtomodify
   and adapt to non-frontier models or novel tasks.  Moreover, the prompting pattern is fixed to a
   variation of ReAct, limiting flexibility in customizing the prompt structure.  Prompt optimizers,
   suchasDSPy(Khattabetal.2024),optimizefew-shotsamplesforin-contextlearning(ICL)and/or
   instructions in the prompt𝑝. Unfortunately, they do not automatically select the prompting pat-
   tern𝐴 and do not return human-readable code.
       Weformulatetheproblemoffindingagoodpatternandcorrespondingpromptbydefiningand
   thenexploringacombinedsearchspace. WewereinspiredbytheAutoMLliteratureoncombined
   search spaces of machine-learning algorithms and their hyperparameters (Thornton et al. 2013),
   exceptthat(i)insteadofdiscreteorcontinuoushyperparameters,weexploretextualICLsamples,
   instructions, and prompting patterns; (ii) instead of classification or regression tasks, we tackle


   AutoML 2025                                                          1          ©2025 the authors, released underCC BY 4.0

## PDF page 2

generativetasks;and(iii)insteadofmodeltrainingorfine-tuning,wefocusonin-contextlearning.
   Weassumeadatasetwithavalidationset𝐷valid,testset𝐷test,aswellasanexamplebank𝐷train for
   few-shot samples. 𝐷valid is used during the optimization process to evaluate the performance of
   a configuration, whereas𝐷test is used once upon completion, to evaluate the performance of the
   best performing configuration. As usual, to avoid over-fitting, we assume these are disjoint from
   each other. The problem statement is to find𝐴∗𝑝 = argmin           L(𝐴𝑝,𝐷valid), where:
                                                              𝐴𝑝∈AP
   • 𝐴 ∈A={Zero-Shot, CoT, ReWOO, ReAct} is the prompting pattern, and
   • 𝑝 = ⟨𝑛,𝑑train,instr⟩ ∈ P is the prompt, comprising a number𝑛 ≤ |𝐷train| of few-shot samples,
     the actual few-shot samples𝑑train ∈ (𝐷train)𝑛, and an instruction instr ∈I. A concrete example
     ofP is provided inFigure A1andFigure A2.
   To avoid getting stuck in local minima while saving compute and finding a solution with a low
   loss, we explore the search spaceAP using successive halving (Jamieson et al. 2016).  To make
   the initial search space AP user-interpretable, and the final solution 𝐴∗𝑝 both human readable
   and executable, we express them in a YAML-based prompting language, PDL (Vaziri et al. 2024).
   PDL’s structured format makes it easy to modify both the initial search space and the optimized
   program, and ensures the final solution remains directly executable. We introduce a library for
   PDL that implements each of the common prompting patterns inA. The initial search spaceAP
   isaYAMLfilewithvariouschoicepointsforAutoMLtodecide. Andthesolution𝐴∗𝑝 isacustom-
   tailoredPDLprogramoptimizedforthegiventask,asgivenbythedatasetandlossfunction. The
   developer can read or even tweak either or both as desired.
       Weevaluateouroptimizeronthreetasks(questionanswering,math,andprogramming),using
   sevenLLMssizedbetween3Band70Bparameters. Wefindthattheoptimizeroftengivesaccuracy
   boostsinthe6–30%range,insomecaseshigher. Giventhesametask,differentpatterns𝐴 ∈Ado
   bestfordifferentmodels. Conversely,giventhesamemodel,differentpatternsdobestfordifferent
   tasks. Besides this variability in the chosen pattern 𝐴, our experiments also revealed variability
   in the optimized prompts𝑝 = ⟨𝑛,𝑑train,instr⟩. We also found that when training data for a task is
   missing,datafromarelatedbutdifferentdatasetcanhelp. Also,whilemostofourexperimentsuse
   moderately-sizedopenmodels,wealsoshowouroptimizedsolutionscanbenefitfrontiermodels.
       This paper makes three primary contributions:
   1.Jointly searching pattern and prompt: prior work in prompt optimization has not investigated
      searching joint search spaces, including agentic patterns.
   2.Noonesizefitsall: wefindthatdifferentmodelssometimeshavedifferingoptimalpromptpat-
      terns for the same benchmark, suggesting that there is not one single optimal prompt pattern.
   3.Source-to-sourceoptimization: weproposethefirstsource-to-sourceoptimizerforLLMprompt
      programs,whereboththeinitialsearchspaceandthefinalsolutionarepromptprogramsinthe
      same language, making the final solution both human-readable and executable.
       Overall, this paper shows how to apply AutoML to automatically discover agentic or non-
   agenticLLMpromptsandpatternsoptimizedforagiventask. WemakeourAutoPDLimplemen-
   tation available athttps://github.com/IBM/prompt-declaration-language, and release therepro-
   duction packageused for the experiments in this work to the community.

2Background
   This paper uses PDL (Vaziri et al. 2024) as a representation for exploring the search space of pro-
   grams. PDLprogramsaredeclarativeandcombinehumanreadabilitywitheaseofexecution. They
   representthecompositionofcallstoLLMsandtools,abstractingawaytheplumbingnecessaryfor


                                                                                                                  2

## PDF page 3

1   text:
                  2 -  role: tools
                  3      text: ${tools}
                  4 - "Out of 1400 participants, 400 passed the test. What percentage is that?\n"
                  5 -  def: actions
                  6      model: replicate/ibm-granite/granite-3.1-8b-instruct
                  7      parser: json
                  8      spec: [{ name: str, arguments: { expr: str }}]
                  9 -  if: ${actions[0].name=="calc"}
                  10      then:
                  11          lang: python
                  12          code: result =${actions[0].arguments.expr}

                                       Figure 1: Basic example of a PDL program.

   suchcompositions. TheoutputoftheoptimizerisalsoaPDLprogram,ratherthansimpletextual
   prompts, so it is fully executable and could be further refined by a developer.
       Figure1showsasimplePDLprogramthatusesatooltoansweraquery. PDLisbasedonthe
   premisethatinteractionswithanLLMaremainlyforthepurposeofgeneratingdata. So,itallows
   users to specify the shape of data to be generated in a declarative way (in YAML), and is agnostic
   of any programming language. The first line ofFigure 1specifies that we are creating some text.
   Next,thefirstblockintheitemizedlistdefinesthetoolsprompt. Line3containsauseofvariable
   tools, expressed as a Jinja expression. This variable is defined as the JSON Schema specification
   of a calculator tool (not shown in this figure, for the full program see Appendix8.2). Line 4 is the
   user query prompt. We do not specify the role explicitly as user is the default role for prompts.
   Lines 5 through 8 show a model call. In PDL, the background context is accumulated implicitly,
   so the output of all blocks executed so far will be passed to the LLM as a list of input messages.
   The result of the model call is assigned to the variable actions (line 5).  The model to be called
   is specified on line 6 (PDL is based on LiteLLM,1 so this is a LiteLLM model id). Finally, lines 7
   and 8 say that we parse the output of the model as JSON and type-check it according to the type
   on line 8. Furthermore, when the inferencing server supports it, model calls with a schema use
   constrained decoding (Willard et al. 2023), enforcing syntactically correct JSON.           2
       Online9,anif-statementcheckswhethertheoutputoftheLLMasksforthecalculatortool. If
   so,weuseaPythoncodeblocktocomputetherequestedtoolcall(lines11and12). Whenweexe-
   cutethisprogramusingthePDLinterpreter,weobtainallthemodelinputs,followedbythemodel
   output,andfinallytheoutputofthetoolcall. PDLhasarichsetofcontrolstructurestoallowwrit-
   ingavarietyofpromptingpatterns,aswellasfunctionstosupportlibraries. Forinstance,Figure3
   shows a function call on line 4. In this paper, we consider the problem of automatically tuning
   promptsandchoosingpromptingpatternsforagivendataset. Thefollowingsectionexplainsour
   approach in further detail.

3AutoPDLApproach

   Figure 2gives an overview of our approach. Referring to the numbers in the arrows:
       (1) The input task is given by two disjoint datasets 𝐷train and 𝐷valid and a loss functionL.
   The datasets comprise⟨𝑥,𝑦⟩ instances, where𝑥 is a question,𝑦 is the corresponding answer, and
   both are text strings. The loss function evaluates the quality of an answer𝑦. (2) The search space
   specificationAP is a YAML file with the optimization variables and their possible values, along
   withsomehyperparameters. Forexample, num_demonstrations:[0, 3, 5]indicatesthateachcandidate
   willhavezero,three,orfiveICLsamplesrandomlydrawnfrom𝐷train. Inthecaseofzerodemon-
   strations, this is equivalent to the zero-shot baseline. If zero is an option, we bias our candidate

      1https://github.com/BerriAI/litellm
      2PDL additionally makes use of the heuristicjson-repairpackage.



                                                                                                                         3

## PDF page 4

Figure 2: Overview of our approach.

sampling to always include one zero-shot candidate, just in case that baseline turns out to be the
best-performing configuration.
    (3) The pattern library consists of four PDL functions.  Zero-shot is a baseline that simply
prompts the LLM with 𝑥 and expects it to return 𝑦.  CoT refers to chain-of-thought (Wei et al.
2022) with in-context learning (Brown et al. 2020): the input includes a few                𝑥𝑖𝑦𝑖 pairs before the
actualquestion𝑥,andtheoutputincludessomereasoningthought thobeforetheactualanswer𝑦.
ReWOO(Xu etal.2023) referstoreasoning withoutobservations. Here,thefew-shot samplesare
trajectories traj𝑖. A trajectory traj𝑖 consists of steps for a particular example problem instance in
𝐷train,e.g.,inthecaseofReWOO,thoandact stepsandtheircontents. Incontrastto𝑥𝑖𝑦𝑖 pairs,traj𝑖
may contain many tho, act and, depending on pattern, obs steps, before reaching the solution𝑦𝑖.
The first LLM call in ReWOO generates one reasoning thought tho and multiple actions act𝑖. The
PDLcodeexecuteseachoftheactionsasatoolcalltoobtainthecorrespondingobservations obs𝑖.
Afinalmodelcallgeneratestheanswer𝑦basedontheobservations. Finally,theReActpattern(Yao
etal.2023)startswithfew-shottrajectoryexamples         traj𝑖 (briefexamplevisibleunderSolution𝐴∗𝑝
inFigure2)andthequestion     𝑥,andthenentersaTAO(thought,action,observation)loop. Ineach
loopiteration,theLLMgenerates thoand act,thenthePDLcodeexecutestheactionasatoolcall,
andfeedsthetooloutputbackasanobservationobs. AspecialFinishactionbreaksoutoftheloop
to return the answer𝑦.
    Once inputs (1), (2), and (3) are in place, the suc-                 1   text:
cessive  halving  optimizer  runs  in  a  loop.    It  starts            2   -  include: ../../tools.pdl
                                                                         3   -  include: ../../ReAct.pdl
with a small subset 𝐷𝑣  ⊂  𝐷valid and many candidates                    4   -  call: ${react}
C ={𝑐1,...,𝑐𝑘} ⊆AP  sampled from the search space.                       5          def: ANSWER
                                                                         6          args:
Each iteration uses 𝐷𝑣  to evaluate the corresponding                    7              task: "Question:${question}"
losses ℓ1,...,ℓ𝑘.  Then each iteration keeps the  𝑘2  can-               8             model: ${model}
                                                                         9              tools: ${tools}
didates with the smallestL while doubling the size of                   10              trajectories: ${demonstrations}
the validation subset 𝐷𝑣.  See §8.3for the algorithm.                   11   - "\nThe answer is${ANSWER.answer}"
(4) After the last iteration, the best remaining candidate
is the solution 𝐴∗𝑝. This solution is a set of PDL defini-              Figure 3: Basic example of PDL program
tionswithconcretevaluesfortheoptimizationvariables,                                using the ReAct pattern.


                                                                                                                        4

## PDF page 5

e.g., inFigure 2,        num_demonstrations: 5 and  demonstrations: ... a list of ReAct trajectories. (5) This pro-
     gram can be used on the test set 𝐷test. For instance,Figure 3shows a call to the ReAct function
     that passes${demonstrations}                       from𝐴∗𝑝 as an argument.

 4Methodology
     Thissectiondescribesthedatasetsused,thetoolsavailabletoagents,andourexperimentalsetup,
     including how we construct agent trajectories that demonstrate tool use for each dataset.

4.1Datasets
     We selected datasets that are widely used in the literature, span diverse tools and domains, and
     arerepresentativeoftoolcategoriesfrequentlystudiedinpriorwork(e.g.,calculator,search,code
     execution). In our experiments, each dataset has three disjoint splits: 𝐷train to sample few-shot
     samples from, 𝐷valid to evaluate candidates during optimization, and 𝐷test to evaluate the final
     chosen solution upon completion of optimization.
     GSM8K. The Grade School Math (Cobbe et al. 2021) dataset consists of 8,792 grade school math
     problems. We sample 1,024 problems without replacement from the train set to use as our 𝐷valid
     set,and1,024fromthetestset(consistingof1,319samples)touseasour𝐷test. Thisleavesa𝐷train
     of6,449problems. Eachproblemconsistsofawordproblem𝑥 suchas“Whatisfifteenmorethana
     quarter of 48?”, a sequence tho of reasoning steps, and finally a plain numeric answer𝑦 following
     a special delimiter. For the CoT prompt pattern, we include these reasoning steps directly in the
     demonstrations.  We use a regular expression to extract the numerical answer from the model
     solution, and define a correct solution as an exact match to the ground truth answer.
     GSM-Hard. Gaoetal.(2023)introduceaderivativeofGSM8Kwithvariablesrandomlychangedto
     large numbers, with 1,319 samples. Unfortunately, GSM-Hard had 132 samples where the ground
     truth was incorrect, and hence we excluded those samples.  We split the single set into equally
     sized 𝐷valid and 𝐷test (𝑛 = 594), and use the GSM8K training set (6,449 samples) described above
     for𝐷train (cross transfer). We use the same correctness criterion as for GSM8K.
     FEVER. TheFactExtractionandVERificationdataset(Thorneetal.2018)isaquestion-answering
     dataset structured around fact-checking. The original dataset contained 185,445 claims that are
     true, false, or unverifiable, and associated with human annotated supporting, refuting, or neu-
     tral sentences and their Wikipedia article of origin. We follow the widely used derivative of this
     benchmarkinBIG-bench(Srivastavaetal.2023),whichreformulatesitintoatrue-or-falsetaskby
     removing unverifiable claims. We sample 1,024 claims from the train set as𝐷valid and 1,024 from
     the test set as 𝐷test. This leaves a 𝐷train of 5,696 claims. BIG-bench also does not include the sup-
     porting or refuting sentences, which we recover by joining on the original dataset, for use e.g., as
     CoT demonstrations. To assess correctness, we check for the presence of “true” and “false” in the
     finallineofthemodelresponse. Ifneitherorbotharepresent,wedeemtheresponseasincorrect,
     and otherwise the correctness is a direct match to the ground truth “true” or “false”.
     MBPP+. MBPP (Austin et al. 2021) is a dataset of 974 mostly basic Python problems, with each
     exampleconsistingofanaturallanguageproblemspecification𝑥 foraself-containedPythonfunc-
     tion𝑦,alongwithasingletestcase. Eachproblemhasanextendedsetoftestcasesusedforevalua-
     tion,whicharenotshowntothemodel.Liuetal.(2023)foundthatMBPPtestcasesareincomplete,
     allowing proposed solutions to pass as correct, despite not matching the problem specification.
     Therefore, our experiments are based on MBPP+, which contains a subset of the problems, but a
     morecompletesetoftestcasesforeachproblem. Weusethesetestcasestoassessthecorrectness
     oftheproposedsolutions. Weusethe374examplesfromtheoriginalMBPPdatasetas𝐷train,and
     split MBPP+ into 𝐷test (224 samples) and 𝐷valid (39 samples) based on which split they were in


                                                                                                                       5

## PDF page 6

MBPP. The discrepancy in number of MBPP+ samples is due to the exclusion of samples found in
     the MBPP train set, to avoid data contamination. Our ReAct implementation followsWang et al.
     (2024). WedonotimplementReWOOasitisnotreactive,           i.e.,cannotincludeexecutionfeedback.

4.2Tools
     Ourpromptlibraryrepresentsactions, i.e.,toolcalls,followingtheJSONtoolcallingschema(Ab-
     delaziz et al. 2024). An action is represented as a JSON object with    name andarguments mapping.
     For example,{"action":"Calc","arguments": {"expr":"48/4"}}                                                            represents a call to a calculator with the
     expression to evaluate. The PDL functions implementing the patterns (see Figure2) accept a list
     of tool definitions as an argument. Each element of that list is itself aPDL function. As both the
     agents and tools are implemented in PDL, the set of tools for a given task could itself be made a
     search space dimension, which we leave to future work.
     Calculator. For math datasets, we give the agentic approaches access to the Calc tool. This tool
     evaluates a cleaned (e.g., replacement of^ with∗∗) expression withSymPy, returning the result.
     In case of error, the function returns a warning that the expression was invalid, which may help
     the agent recover from invalid input.
     Search. For fact verification, we provide access to the Search tool, which returns the summary
     of the first Wikipedia search result for the query.  If no results are found, a hint to try again is
     returned, or if the title is too ambiguous, a list of possible disambiguations.
     Execute. We implement a programming agent for the code generation task, which can execute
     arbitrary code surrounded in XML-style<execute> tags. This tool executes the code in a Python
     shell, which returns the result of the final expression. This allows the agent to test its proposed
     solution against the given test case before submitting its solution.
     Finish. The most basic action is the Finish action, or<solution> tag for the coding agent. This
     ends the agent’s trajectory and results in the agent returning the value as the solution.

4.3ExperimentalSetup
     We evaluate the efficacy of our approach by running an optimization process to completion for
     each model & dataset pair, and subsequently comparing the task accuracy.  As a baseline, we
     evaluate each model in a zero-shot setting. This setting reflects the minimal effort approach by a
     user or developer, where they do not include any demonstrations with their query or task. As no
     model we investigate was specifically trained to create agentic trajectories in a zero-shot setting,
     it is not feasible to create a zero-shot setting under ReAct or ReWOO.
          For the optimization process, we used an initial candidate setC of 100 candidate prompt con-
     figurations 𝑐𝑖  = ⟨𝐴,𝑝⟩ ∈ C per experiment.  We fixed the size of the initial validation subset
     𝑑valid ⊆ 𝐷valid to16. WedefineL(𝑐𝑖, 𝑑valid) =−Accuracy(𝑐𝑖, 𝑑valid) foragivencandidate𝑐𝑖,where
     accuracy is the fraction in𝑑valid of correctly solved problems by𝑐𝑖 as per the dataset definition of
     correctness. For each candidate𝑐𝑖,𝑝 = ⟨𝑛,𝑑train,instr⟩ ∈P where𝑛 few-shot samples or trajecto-
     ries𝑑train ∈ (𝐷train)𝑛 are randomly sampled with replacement. The number of possible values for
     P is combinatorially large and depends on the dataset𝐷train used. Finally, upon completion of an
     optimization process, the optimal candidate is evaluated on𝐷𝑡𝑒𝑠𝑡.
     Constructing Trajectories. As we are also optimizing over agentic trajectories, we also need a
     set of trajectories to sample few-shot samples from.  To achieve this, we create a basic agentic
     trajectorytraj𝑖 foreachtrainingexample⟨𝑥𝑖,𝑦𝑖⟩,followingarule-basedtransformation. Wedesign
     and apply a template to each dataset, which is relatively simple and easy to implement for other
     datasets (details are provided in §8.5).  Prior work has introduced approaches to bootstrapping
     trajectories e.g., in software engineering (Pan et al. 2024), tool use demonstrations (Li et al. 2025),


                                                                                                                            6

## PDF page 7

and reasoning paths (Zelikman et al. 2022), which could be applied to this problem.  While we
                      acknowledgetheshortcomingsofmanualtemplateconstruction,wearguethisapproachhastwo
                      strengths:  it is generalizable in the sense that templates can be mixed and matched, and that
                      the trajectories are directly based on the datasets used.  Additionally, we wanted to work with
                      commonlyuseddatasetsthatcoveravarietyoftoolsanddomains,ratherthanemergingdatasets
                      containing trajectories or tool use demonstrations.
                      Models. We aim to study models of various abilities, e.g., natural language or code, various cre-
                      ators,andvarioussizes,rangingfromsingledigitbillionsofparametersuptotheedgeoffeasibility
                      onconsumerhardware. WeincludesevenmodelsavailableoninferenceserviceIBMwatsonx3 in
                      our study. We select three generalist natural language instruction models, Llama 3.1 8B, 3.2 3B,
                      and3.370Bfromtheopen-sourceandwidelystudiedLLaMafamily(Dubeyetal.2024). Wefurther
                      selectthreemodelsfromMishraetal.(2024),whichpredateDubeyetal.(2024)byapproximately
                      3 months. We selectGranite13BInstructV2 as a generalist model, andGranite20B and34B
                      CodeInstructascodemodels. WeselectGranite3.18Basanadditionalgeneralistmodel(Gran-
                      iteTeam2024). Allofourexperimentsusegreedydecoding,        i.e.,nosampling,tolimittheimpactof
                      hyperparameterchoice. Thenumberofmodelsweevaluateislimitedbycostin$/tokentermsand
                      executiontime. Bystudyingvariousmodels,wedemonstratethegeneralizabilityofourapproach.
                      Alternative Setups. We evaluate two alternative experimental setups. First, to investigate low-
                      resource scenarios, we examine whether performance on one dataset can be improved by using
                      demonstrations𝐷trainfromasimilardataset,whileoptimizingw.r.t.𝐷valid. Forthisexperiment,we
                      investigate whether performance on GSM-Hard can be improved by using demonstrations from
                      GSM8K,whileoptimizingw.r.t.GSM-Hard𝐷valid. Second,toexploresavingoptimizationcosts,we
                      assess whether optimized prompt programs of one model can transfer well to a frontier model.
                      The intuition is that while that might not be the best program for the frontier model, it might at
                      least improve somewhat over the baseline. To this end, we evaluate the optimized PDL programs
                      of LLaMa3.170Bon OpenAI’sgpt-4o-mini-2024-07-18, for each dataset.

                   5Results
                      This section describes the results of our empirical study to evaluate our AutoPDL approach and
                      answer the following research questions:



                          RQ1 asks to what degree our AutoPDL approach can improve model performance over their
                      zero-shotbaselineacrossavarietyofcommonlyusedbenchmarks. Wealsoseektoidentifytrends,
                      ifany,inoptimalconfigurations,e.g.,whethermorefew-shotsisalwaysbetter,orwhethercertain
                      prompt patterns are particularly suited to certain problem domains.



                          RQ2 investigates whether optimizing on one dataset using demonstrations from another, re-
                      lated,datasetcanresultinhigherperformancethanusingnodemonstrations(zero-shot). ThisRQ
                      addressesalow-resourcescenarioinwhichalimitedpoolofdemonstrationsexistsinonedataset,
                      but a dataset from a similar domain has a large pool.






                         3https://www.ibm.com/watsonx



                                                                                                                                    7
RQ 3: Do solutions found by AutoPDL improve performance on frontier models, even when optimized
RQ2: CanAutoPDLmakeupforamissingfew-shotexamplebankforagiventaskbyreusingtheexampleRQ 1: To what extent does AutoPDL improve accuracy, and how much does the best solution vary by
for open-source models?
task and model?bank from a similar task?

## PDF page 8

Itcanbeexpensivetorunoptimizationagainstcommercialfrontier-modelAPIs. RQ3assesses
                         whetheroptimizedpromptprogramsaretransferabletodifferent(andlikelystronger)modelsthan
                         those they were optimized with.
                              Table1reportstheresultsofouroptimizationandevaluationprocedure. Weperformedthree
                         completeoptimizationrunsforGSM8KandMBPP+,andreportmeanaccuracyandstandarddevi-
                         ation in percentage points (i.e., absolute, not relative, uncertainty). We did not complete multiple
                         runs for all model-dataset pairs due to time and resource constraints. Where multiple runs were
                         performed, we report the pattern of the highest scoring run. Additionally, we note minor vari-
                         ance in certain zero-shot settings due to API non-determinism. Across models and datasets, we
                         generally find some improvement over the zero-shot baseline with few-shot chain-of-thought, or
                         agentic patterns ReAct or ReWOO.

                                   Table 1: Model accuracies across datasets for baseline (zero-shot) and optimized versions.



























                         FEVER.  We observed the minimum improvement in Granite 3.1 8B, with a 0.7 percentage
                         point (pp) improvement, and a maximal improvement of 68.9pp for Granite 13B Instruct V2.
                         In terms of prompt pattern, CoT and ReWOO are equally represented. ReAct was not the optimal
                         for any of the models. Interestingly, the largest model (LLaMa 3.3 70B) benefited by 9.9pp from
                         5-shot CoT. FEVER runtimes are generally higher than the other benchmarks, likely due to the
                         large number of tokens involved by including Wikipedia content.

                         GSM8K.  The highest improvement recorded (9.9pp) was for LLaMa 3.3 70B using 3-shot CoT,
                         while the minimum improvement of 3.5pp was in LLaMa 3.2 3B using 3-shot CoT. ReAct and
Dataset          Model   ReWOOwerenottheoptimalforanymodel. ForGranite3.18B,Granite20BCode,andGran-AccuracyPatternRuntime
                         ite 34B Code, no improvement over the zero-shot baseline was identified. This was somewhatZero-ShotOptimizedDelta
          Granite3.18B                 78.3%   79.0%            +0.7pp   ReWOO(5shot)          08:55surprising, as generally including even some few-shot samples improves performance in LLMs.
          Granite13BInstructV2        6.5%   75.4%          +68.9pp  ReWOO(3shot)           08:12
          Granite20BCode             39.7%   64.2%           +24.5pp   CoT(3shot)              05:06MBPP+. Several models benefited from execution feedback, as 3 out of 7 had ReAct as the opti-
FEVER     Granite34BCodemal prompt pattern (ReWOO was excluded as described in §4.3). The greatest improvement of56.4%65.6%+9.2ppCoT(3shot)03:47
          LLaMA3.18B                 68.5%   78.0%            +9.5pp   CoT(3shot)              05:24
          LLaMA3.23B                 38.0%   66.9%          +28.9pp  ReWOO(5shot)           09:08
          LLaMA3.370B                67.6%   77.5%            +9.9pp   ReWOO(5shot)          09:32                                                     8
          Granite3.18B               74.2%   (74.2±0.6)%     +0.0pp  Zero-Shot(Baseline)    08:56
          Granite13BInstructV2      23.0%  (30.9±1.0)%    +7.9pp   CoT(3shot)              09:20
          Granite20BCode             68.7%   (68.7±0.1)%     +0.0pp  Zero-Shot(Baseline)    09:27
GSM8K     Granite34BCode             72.1%  (72.1±0.1)%    +0.0pp   Zero-Shot(Baseline)     08:52
          LLaMA3.18B                 78.4%   (85.3±0.6)%     +6.9pp  CoT(5shot)             08:48
          LLaMA3.23B                 71.8%  (75.3±0.4)%    +3.5pp   CoT(3shot)              16:36
          LLaMA3.370B                85.5%   (95.4±0.2)%     +9.9pp  CoT(3shot)             07:50
          Granite3.18B                 62.9%  (62.9±0.0)%    +0.0pp   Zero-Shot(Baseline)     02:14
          Granite13BInstructV2       10.7%   (19.2±1.2)%     +8.5pp  ReAct(5shot)           04:02
          Granite20BCode             51.8%  (51.8±0.4)%    +0.0pp   Zero-Shot(Baseline)     03:43
MBPP+     Granite34BCode             48.7%   (61.3±1.0)%    +12.6pp  ReAct(3shot)           04:54
          LLaMA3.18B                 61.2%  (62.8±4.0)%    +1.6pp   ReAct(5shot)            01:45
          LLaMA3.23B                 58.0%   (58.0±0.4)%     +0.0pp  Zero-Shot(Baseline)    02:01
          LLaMA3.370B                71.4%  (71.4±0.0)%    +0.0pp   Zero-Shot(Baseline)     02:27

## PDF page 9

12.6pp was in Granite 34B Code, and 8.5pp in Granite 13B Instruct V2, likely due to its poor
                       programming performance as a generalist, non-code model. In contrast, the smaller LLaMa 3.1
                       8B model had high zero-shot performance of 61.2%, yet still improved by up to 6.2pp (1.6pp on
                       average) with ReAct. No improvement was observed for Granite 3.1 8B, Granite 20B Code, or
                       the otherLLaMa models.
                                         Table 2: Model accuracies on GSM-Hard for cross optimization experiment.













                                              Table 3: Model accuracy for GPT-4o-mini cross experiment results








                       MissingFew-ShotExampleBank. We optimized the PDL program for GSM-Hard, using GSM8K
                       demonstrations,andreportresultsinTable2. Wefoundthatinmostcases,GSM8Kdemonstrations
                       wereatleastnotharmfulformodelsonGSM-Hard,withupto6.1ppimprovementforLLaMa3.3
                       70B using 5-shot CoT.
                       Commercial  Frontier  Model.   To  assess  whether  performance  gains  in  one  model  can  be
                       achieved in another, we evaluate the optimized PDL programs of LLaMa 3.1 70B on OpenAI’s
                       gpt-4o-mini-2024-07-18 and report results inTable 3(we did not use      LLaMa 3.3 70B here be-
                       cause we did this experiment earlier and did not have the time and resources to repeat it for the
                       final version of this paper). For all dataset/prompt pattern pairs that resulted in improvement for
                       LLaMa3.170B, we found a surprising improvement inGPT4o-mini of at least 4pp on FEVER us-
                       ing 3-shot CoT, 9.3pp on GSM-Hard (using GSM8K demonstrations) with 5-shot ReAct (Granite
                       LLaMainstructions),andupto13.1pponGSM8Kusing5-shotCoT.Thissuggeststhatoptimizing
                       for an open-source model can also benefit a closed-source model.

                    6RelatedWork
                       The closest related work is on prompt optimization.  APE starts with an LLM-generated set of
                       candidateprompts,thenperformsrejectionsamplingbasedonevaluationonasubsetofdata(Zhou
                       et al. 2023).  ZOPO incorporates a Neural Tangent Kernel-based derived Gaussian process into
                       standard zeroth-order optimization for an efficient search of a locally-optimal instruction (Hu et
                       al.2024). Unlikeourapproach,neitherAPEnorZOPOoptimizefew-shotsamples. Aviarycanalso
                       jointlyoptimizeoverpromptpattern,instruction,andfew-shotexamples(Narayananetal.2024).
                       However,itwouldrequirethedefinitionofacustomoperator. CEDARusesademonstrationpool,
                       fromwhichitretrievesfew-shotexamplesatquerytime(Nashidetal.2023). Unlikeourapproach,
                       these few-shot samples are retrieved on a per-inference basis, not optimized ahead-of-time.

Dataset            Model                    Accuracy                  Pattern        Runtime
                                  AccuracyZero-ShotOptimizedDelta                                                                             9
Dataset    Granite3.18B                  44.0%   44.0%       +0.0pp   Zero-Shot(Baseline)     04:57ModelZero-ShotOptimizedDeltaPattern
FEVER       GPT-4o-mini       83.7%   87.7%     +4.0pp   CoT(3shot)Granite13BInstructV24.7%5.6%+0.8ppCoT(3shot)03:30
GSM-Hard   Granite20BCode              28.8%   28.8%       +0.0pp   Zero-Shot(Baseline)     08:26GPT-4o-mini45.6%54.9%+9.3ppReAct(5shot,GraniteLLaMa)
GSM-Hard   Granite34BCode            27.9%   30.0%      +2.0pp  ReWOO(5shot)           05:49
GSM8K      GPT-4o-mini       77.8%   90.9%    +13.1pp   CoT(5shot)LLaMA3.18B                  31.6%   32.3%       +0.7pp   ReWOO(5shot)          04:44
MBPP+      LLaMA3.23BGPT-4o-mini72.3%72.3%26.3%27.4%+0.0ppZero-Shot(Baseline)+1.2ppCoT(5shot)16:50

           LLaMA3.370B                47.3%   53.4%       +6.1pp   CoT(5shot)               11:07

## PDF page 10

EASE leverages embeddings to represent few-shot examples, and uses a neural bandit algo-
   rithm to find an ordered set that performs well for test queries from a given task (Z. Wu et al.
   2024). An extension of their approach jointly optimizes demonstrations and instructions. How-
   ever,theapproachrequiresbothanadditionalembeddingmodel,andthetrainingofanewmodel
   topredictvalidationscoresfromembeddings. Unlikeourapproach,EASEdoesnotoptimizeover
   agentic patterns.
       DSPy optimizes instructions and few-shot samples for a chain of LLM calls (Khattab et al.
   2024) (not just a single call like APE or CEDAR). Also, DSPy takes away control over the exact
   prompt from the programmer, which our approach preserves. Similarly to DSPy, TextGrad also
   optimizes a chain of LLM calls, by using LLMs to back-propagate modifications to instructions
   in prompts (Yuksekgonul et al. 2025).  However, unlike our approach, neither of these optimize
   agentic patterns.
       BPO trains a sequence-to-sequence model on prompts augmented by an LLM incorporating
   human preferences, producing a model that improves given input prompts (Cheng et al. 2024).
   APOHFintroduceastrategytoselectapairofpromptstoquerytheuserforpreferencefeedback,
   whichtheyusetooptimizeLLM-generatedinstructionsonavalidationset(Linetal.2024). How-
   ever,neitheroftheseapproachesexplicitlyoptimizedemonstrationsoragenticpatterns. EvoAgent
   optimizes the instructions of a population of agents via crossover, mutation, and selection (Yuan
   et al. 2024). It then forms an ensemble from the final, fittest, population. GPTSwarm represents
   each agent as a graph, then freezes intra-agent edges and optimizes the placement of additional
   inter-agent edges (Zhuge et al. 2024).  Unlike our approach, neither EvoAgent nor GPTSwarm
   optimize the agentic pattern inside individual agents, nor do they optimize few-shot samples.
       Another closely related field of study is AutoML. Auto-sklearn (Feurer et al. 2015) used
   Bayesianoptimizationtojointlyperformbothalgorithmselectionandhyperparametersofascikit-
   learnpipeline(Buitincketal.2013). Whiledifferent,weseesomeanalogybetweenalgorithmsand
   agenticpatterns,andbetweenhyperparametersandfew-shotsamples. DAUBfirstevaluatesmany
   candidate models on a small amount of data, then successively reduces candidates and increases
   data to ultimately pick a strong model (Sabharwal et al. 2016). The successive-halving algorithm
   takesasimilarapproach(Jamiesonetal.2016). Ourapproachisinspiredbythesameincremental
   dataallocationidea. WhilebothrandomizedsearchandBayesianoptimizationarepopularinAu-
   toML,therearealsomoreintricateapproaches. Forinstance,TPOTusesgeneticalgorithms(Olson
   etal.2016),andAlphaD3MusesMonte-Carlotreesearch(Drorietal.2018). Wechosetostartwith
   a simpler technique that depends less on a well-behaved optimization space. That said, exploring
   moreadvancedAutoMLoptimizerscouldbefruitfulfutureworkforAutoPDL.Lale(Baudartetal.
   2021)treatsAutoMLasasource-to-sourceoptimization,similartothispaper,butunlikeAutoPDL,
   it has not been used to optimize agentic patterns or prompts.

7Conclusion
   WepresentourAutoPDLapproachforjointlyoptimizingpromptingpatternsandtextualprompts
   forlargelanguagemodels,addressingthechallengesassociatedwithmanualpromptengineering.
   By formulating the optimization as a discrete search over both agentic and non-agentic patterns,
   combined with instructions and few-shot samples, we leveraged successive halving to efficiently
   navigate this search space. Our evaluation across various datasets (FEVER, GSM8K, GSM-Hard,
   andMBPP+)andmultiplemodels(fromtheLLaMA,Granite,andGPTfamilies)demonstratessub-
   stantialaccuracyimprovements,upto68.9percentagepoints,andaffirmsthatnosingleprompting
   strategyuniversallyoutperformsothersacrosstasksandmodels. Additionally,generatingcodein
   a YAML-based prompt programming language (PDL) makes it executable, easy to modify, and
   readable by humans, supporting practical adoption and adaptation.




                                                                                                                10

## PDF page 11

References

Abdelaziz, I., Basu, K., Agarwal, M., Kumaravel, S., Stallone, M., Panda, R., Rizk, Y., Bhargav, G.,
    Crouse, M., Gunasekara, C., Ikbal, S., Joshi, S., Karanam, H., Kumar, V., Munawar, A., Neelam,
    S., Raghu, D., Sharma, U., Soria, A. M., Sreedhar, D., Venkateswaran, P., Unuvar, M., Cox, D.,
    Roukos, S., Lastras, L., and Kapanipathi, P. (2024). Granite-Function Calling Model: Introducing
    FunctionCallingAbilitiesviaMulti-taskLearningofGranularTasks.url:https://arxiv.org/abs/
    2407.00121(cit. on p.6).
Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry,
    M., Le, Q., and Sutton, C. (2021). Program Synthesis with Large Language Models. url:http:
    //arxiv.org/abs/2108.07732(cit. on p.5).
Baudart,G.,Hirzel,M.,Kate,K.,Ram,P.,Shinnar,A.,andTsay,J.(2021).“PipelineCombinatorsfor
    GradualAutoML”.In:AdvancesinNeuralInformationProcessingSystems(NeurIPS),pp.19705–
    19718 (cit. on p.10).
Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam,
    P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R.,
    Ramesh, A., Ziegler, D. M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray,
    S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., and Amodei, D.
    (2020). Language Models are Few-Shot Learners. url:https://arxiv.org/abs/2005.14165(cit. on
    pp.1,4).
Buitinck,L.,Louppe,G.,Blondel,M.,Pedregosa,F.,Mueller,A.,Grisel,O.,Niculae,V.,Prettenhofer,
    P., Gramfort, A., Grobler, J., Layton, R., VanderPlas, J., Joly, A., Holt, B., and Varoquaux, G.
    (2013).APIDesignforMachineLearningSoftware:Experiencesfromthescikit-learnProject.url:
    https://arxiv.org/abs/1309.0238(cit. on p.10).
Cheng,J.,Liu,X.,Zheng,K.,Ke,P.,Wang,H.,Dong,Y.,Tang,J.,andHuang,M.(2024).“Black-Box
    Prompt Optimization: Aligning Large Language Models without Model Training”. In:Annual
    Meeting of the Association for Computational Linguistics (ACL), pp. 3201–3219 (cit. on p.10).
Cobbe,K.,Kosaraju,V.,Bavarian,M.,Chen,M.,Jun,H.,Kaiser,L.,Plappert,M.,Tworek,J.,Hilton,
    J.,Nakano,R.,Hesse,C.,andSchulman,J.(2021).TrainingVerifierstoSolveMathWordProblems.
    url:http://arxiv.org/abs/2110.14168(cit. on p.5).
Drori,I.,Krishnamurthy,Y.,Rampin,R.,Lourenco,R.d.P.,Ono,J.P.,Cho,K.,Silva,C.,andFreire,J.
    (2018).“AlphaD3M:MachineLearningPipelineSynthesis”.In:WorkshoponAutomaticMachine
    Learning (AutoML) (cit. on p.10).
Dubey, A. et al. (2024). The Llama 3 Herd of Models. url:http://arxiv.org/abs/2407.21783(cit. on
    p.7).
Feurer, M., Klein, A., Eggensperger, K., Springenberg, J., Blum, M., and Hutter, F. (2015). “Efficient
    and Robust Automated Machine Learning”. In:Conference on Neural Information Processing
    Systems (NIPS), pp. 2962–2970 (cit. on p.10).
Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., and Neubig, G. (2023). “PAL:
    Program-aided Language Models”. In:International Conference on Machine Learning (ICML),
    pp. 10764–10799 (cit. on p.5).
Granite Team, I. (2024). Granite 3.0 Language Models (cit. on p.7).
Hu, W., Shu, Y., Yu, Z., Wu, Z., Lin, X., Dai, Z., Ng, S.-K., and Low, B. K. H. (2024). “Localized
    Zeroth-Order Prompt Optimization”. In:Conference on Neural Information Processing Systems
    (NeurIPS), pp. 86309–86345 (cit. on p.9).


                                                                                                                    11

## PDF page 12

Jamieson,K.andTalwalkar,A.(2016).“Non-stochasticBestArmIdentificationandHyperparame-
    terOptimization”.In:ConferenceonArtificialIntelligenceandStatistics(AISTATS),pp.240–248
    (cit. on pp.2,10,17).
Khattab, O., Singhvi, A., Maheshwari, P., Zhang, Z., Santhanam, K., A, S. V., Haq, S., Sharma,
    A., Joshi, T. T., Moazam, H., Miller, H., Zaharia, M., and Potts, C. (2024). “DSPy: Compiling
    DeclarativeLanguageModelCallsintoSelf-ImprovingPipelines”.In:International Conference
    on Learning Representations (ICLR) (cit. on pp.1,10).
Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., and Iwasawa, Y. (2022). “Large Language Models
    Are Zero-Shot Reasoners”. In:Conference on Neural Information Processing Systems (NeurIPS),
    pp. 22199–22213 (cit. on p.1).
Li, C., Xue, M., Zhang, Z., Yang, J., Zhang, B., Wang, X., Yu, B., Hui, B., Lin, J., and Liu, D. (2025).
    START:Self-taughtReasonerwithTools.url:https://arxiv.org/abs/2503.04625(cit.onpp.6,18).
Lin, X., Dai, Z., Verma, A., Ng, S.-K., Jaillet, P., and Low, B. K. H. (2024). Prompt Optimization with
    Human Feedback.url:http://arxiv.org/abs/2405.17346(cit. on p.10).
Liu, J., Xia, C. S., Wang, Y., and Zhang, L. (2023). “Is Your Code Generated by ChatGPT Really
    Correct?RigorousEvaluationofLargeLanguageModelsforCodeGeneration”.In:Conference
    on Neural Information Processing Systems (NeurIPS), pp. 21558–21572 (cit. on p.5).
Mishra,M.,Stallone,M.,Zhang,G.,Shen,Y.,Prasad,A.,Soria,A.M.,Merler,M.,Selvam,P.,Suren-
    dran, S., Singh, S., Sethi, M., Dang, X.-H., Li, P., Wu, K.-L., Zawad, S., Coleman, A., White, M.,
    Lewis,M.,Pavuluri,R.,Koyfman,Y.,Lublinsky,B.,deBayser,M.,Abdelaziz,I.,Basu,K.,Agar-
    wal,M.,Zhou,Y.,Johnson,C.,Goyal,A.,Patel,H.,Shah,Y.,Zerfos,P.,Ludwig,H.,Munawar,A.,
    Crouse,M.,Kapanipathi,P.,Salaria,S.,Calio,B.,Wen,S.,Seelam,S.,Belgodere,B.,Fonseca,C.,
    Singhee,A.,Desai,N.,Cox,D.D.,Puri,R.,andPanda,R.(7,2024).GraniteCodeModels:AFamily
    ofOpenFoundationModelsforCodeIntelligence.Version1.url:http://arxiv.org/abs/2405.04324
    (cit. on p.7).
Moura, J. (2023). CrewAI: Framework for orchestrating role-playing, autonomous AI agents. url:
    https://github.com/crewAIInc/crewAI(visited on 06/06/2025) (cit. on p.1).
Narayanan,S.,Braza,J.D.,Griffiths,R.-R.,Ponnapati,M.,Bou,A.,Laurent,J.,Kabeli,O.,Wellawatte,
    G., Cox, S., Rodriques, S. G., and White, A. D. (2024). Aviary: Training Language Agents on
    Challenging Scientific Tasks.url:http://arxiv.org/abs/2412.21154(cit. on p.9).
Nashid,N.,Sintaha,M.,andMesbah,A.(2023).“Retrieval-BasedPromptSelectionforCode-Related
    Few-ShotLearning”.In:InternationalConferenceonSoftwareEngineering(ICSE),pp.2450–2462
    (cit. on p.9).
Olson,R.S.,Urbanowicz,R.J.,Andrews,P.C.,Lavender,N.A.,Kidd,L.C.,andMoore,J.H.(2016).
    “Automating Biomedical Data Science Through Tree-Based Pipeline Optimization”. In:Euro-
    peanConferenceontheApplicationsofEvolutionaryComputation(EvoApplications),pp.123–137
    (cit. on p.10).
Pan, J., Wang, X., Neubig, G., Jaitly, N., Ji, H., Suhr, A., and Zhang, Y. (2024). Training Software
    Engineering Agents and Verifiers with SWE-Gym. url:http://arxiv.org/abs/2412.21139(cit. on
    p.6).
Sabharwal,A.,Samulowitz,H.,andTesauro,G.(2016).“SelectingNear-OptimalLearnersviaIncre-
    mentalDataAllocation”.In:ConferenceonArtificialIntelligence(AAAI),pp.2007–2015(cit.on
    p.10).





                                                                                                                  12

## PDF page 13

Schluntz, E. and Zhang, B. (2024). Building effective agents. url:https://www.anthropic.com/
    research/building-effective-agents(visited on 06/06/2025) (cit. on p.1).
Srivastava,A.etal.(2023).“BeyondtheImitationGame:QuantifyingandExtrapolatingtheCapa-
    bilities of Language Models”.Transactions on Machine Learning Research (TMLR) (cit. on p.5).
Thorne,J.,Vlachos,A.,Christodoulopoulos,C.,andMittal,A.(2018).FEVER:ALarge-ScaleDataset
    for Fact Extraction and VERification.url:http://arxiv.org/abs/1803.05355(cit. on p.5).
Thornton,C.,Hutter,F.,Hoos,H.H.,andLeyton-Brown,K.(2013).“Auto-WEKA:CombinedSelec-
    tionandHyperparameterOptimizationofClassificationAlgorithms”.In:ConferenceonKnowl-
    edge Discovery and Data Mining (KDD), pp. 847–855 (cit. on p.1).
Vaziri, M., Mandel, L., Spiess, C., and Hirzel, M. (2024). PDL: A Declarative Prompt Programming
    Language.url:http://arxiv.org/abs/2410.19135(cit. on p.2).
Wang,X.,Wang,Z.,Liu,J.,Chen,Y.,Yuan,L.,Peng,H.,andJi,H.(2024).MINT:EvaluatingLLMsin
    Multi-turn Interaction with Tools and Language Feedback.url:http://arxiv.org/abs/2309.10691
    (cit. on pp.6,17,18).
Wei,J.,Wang,X.,Schuurmans,D.,Bosma,M.,Ichter,B.,Xia,F.,Chi,E.,Le,Q.,andZhou,D.(2022).
    “Chain-of-ThoughtPromptingElicitsReasoninginLargeLanguageModels”.In:Conferenceon
    Neural Information Processing Systems (NeurIPS), pp. 24824–24837 (cit. on pp.1,4).
Willard, B. T. and Louf, R. (2023). Efficient Guided Generation for Large Language Models. url:
    https://arxiv.org/abs/2307.09702(cit. on p.3).
Wu,Q.,Bansal,G.,Zhang,J.,Wu,Y.,Li,B.,Zhu,E.,Jiang,L.,Zhang,X.,Zhang,S.,Liu,J.,Awadal-
    lah, A. H., White, R. W., Burger, D., and Wang, C. (2023). AutoGen: Enabling Next-Gen LLM
    Applications via Multi-Agent Conversation.url:https://arxiv.org/abs/2308.08155(cit. on p.1).
Wu, Z.,Lin, X.,Dai, Z.,Hu, W., Shu,Y., Ng,S.-K., Jaillet,P., and Low,B. K.H. (2024).“Prompt Op-
    timizationwithEASE?EfficientOrdering-awareAutomatedSelectionofExemplars”.In:Con-
    ference on Neural Information Processing Systems (NeurIPS), pp. 122706–122740 (cit. on p.10).
Xu, B., Peng, Z., Lei, B., Mukherjee, S., and Xu, D. (2023). ReWOO: Decoupling Reasoning from
    Observations for Efficient Augmented Language Models. url:https://arxiv.org/abs/2305.18323
    (cit. on pp.1,4).
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. (2023). “ReAct: Syner-
    gizing Reasoning and Acting in Language Models”. In:International Conference on Learning
    Representations (ICLR) (cit. on pp.1,4).
Yuan,S.,Song,K.,Chen,J.,Tan,X.,Li,D.,andYang,D.(2024).EvoAgent:TowardsAutomaticMulti-
    Agent Generation via Evolutionary Algorithms. url:https://arxiv.org/abs/2406.14228(cit. on
    p.10).
Yuksekgonul, M., Bianchi, F., Boen, J., Liu, S., Lu, P., Huang, Z., Guestrin, C., and Zou, J. (2025).
    “OptimizingGenerativeAIbyBackpropagatingLanguageModelFeedback”.Nature,639(8055),
    pp. 609–616 (cit. on p.10).
Zelikman, E., Wu, Y., Mu, J., and Goodman, N. D. (2022). “STaR: Self-Taught Reasoner, Bootstrap-
    ping Reasoning With Reasoning”. In: Conference on Neural Information Processing Systems
    (NeurIPS) (cit. on p.7).
Zhou,Y.,Muresanu,A.I.,Han,Z.,Paster,K.,Pitis,S.,Chan,H.,andBa,J.(2023).“LargeLanguage
    Models are Human-Level Prompt Engineers”. In:International Conference on Learning Repre-
    sentations (ICLR) (cit. on p.9).



                                                                                                                  13

## PDF page 14

Zhuge,M.,Wang,W.,Kirsch,L.,Faccio,F.,Khizbullin,D.,andSchmidhuber,J.(2024).“GPTSwarm:
    Language Agents as Optimizable Graphs”. In:International Conference on Machine Learning
    (ICML) (cit. on p.10).































































                                                                                                              14

## PDF page 15

8SupplementalMaterial
                         8.1ConcretePromptExample






























                                  Figure A1: Basic example of a concrete prompt configuration in PDL, using CoT pattern and two
                                                demonstrations.







































1  defs:
2      prompt_pattern:cot
3      num_demonstrations:2
4      demonstrations:            Figure A2: Corresponding rendered prompt configuration forFigure A1. With the exception of the
5         data:
1 Answerthequestionstothebestofyourabilities.non-rendered${question}                variable, this is the input to the language model.
6     - question: TriciaisathirdofAmilia'sageandAmiliaisaquarterofYorick'sage.Yorickistwice
2
7        Eugene'sageandKhloeisathirdofEugene'sage.Rupertis10yearsolderthanKhloebut2years
3 Question:TriciaisathirdofAmilia'sageandAmiliaisaquarterofYorick'sage.YorickistwiceEugene's
8        youngerthanVincentwhois22yearsold.Howold,inyears,isTricia?
4 ageandKhloeisathirdofEugene'sage.Rupertis10yearsolderthanKhloebut2yearsSeeFigure A1andFigure A2.
9             reasoning: |-
5 youngerthanVincent
10        RupertisyoungerthanVincentby2years,soheis22yearsold-2years=<<22-2=20>>20yearsold.6 whois22yearsold.Howold,inyears,isTricia?
11        Khloeis10yearsyoungerthanRupertsosheis20yearsold-10years=10yearsold.7 Answer:Let'sthinkstepbystep.RupertisyoungerthanVincentby2years,soheis22years
12        Eugeneis3timesolderthanKhloesoheis10yearsold*3=<<10*3=30>>30yearsold.8 old-2years=<<22-2=20>>20yearsold.                                              15
13        YorickistwiceEugene'sagesoheis30yearsold*2=<<30*2=60>>60yearsold.9 Khloeis10yearsyoungerthanRupertsosheis20yearsold-10years=10yearsold.
14        AmiliaisaquarterofYorick'sagesosheis60yearsold/4=<<60/4=15>>15yearsold.
10 Eugeneis3timesolderthanKhloesoheis10yearsold*3=<<10*3=30>>30yearsold.
15        TriciaisathirdofAmilia'sagesosheis15yearsold/3=<<15/3=5>>5yearsold.
11 YorickistwiceEugene'sagesoheis30yearsold*2=<<30*2=60>>60yearsold.
16            answer: '5'
12 AmiliaisaquarterofYorick'sagesosheis60yearsold/4=<<60/4=15>>15yearsold.
17     - question: Emmalyndecidedtopaintfencesinherneighborhoodfortwentycentspermeter.Iftherewere13 TriciaisathirdofAmilia'sagesosheis15yearsold/3=<<15/3=5>>5yearsold.
18        50fencesintheneighborhoodthatshehadtopaintandeachfencewas500meterslong,calculate14 Theansweris5
15
19        thetotalamountsheearnedfrompaintingthefences.
16 Question:Emmalyndecidedtopaintfencesinherneighborhoodfortwentycentspermeter.Iftherewere50
20             reasoning: |-
17 fencesintheneighborhoodthatshehadtopaintandeachfencewas500meterslong,calculatethetotal
21        Thetotallengthforthefiftyfencesis50*500=<<50*500=25000>>25000meters.
18 amountsheearnedfrompaintingthefences.
22        IfEmmalynchargedtwentycentstopaintameterofafence,thetotalincomeshegotfrompainting
19 Answer:Let'sthinkstepbystep.Thetotallengthforthefiftyfencesis50*500=<<50*500=25000>>25000meters.
23        thefencesis$0.20*25000=$5000
20 IfEmmalynchargedtwentycentstopaintameterofafence,thetotalincomeshegotfrompaintingthe
24            answer: '5000'
21 fencesis$0.20*25000=$5000
25      instruction: Answerthequestionstothebestofyourabilities.
22 Theansweris5000
26  text:
23
27   - include: CoT.pdl
24 Question:${question}
28   -"${instruction}\n\n"
25 Answer:Let'sthinkstepbystep.
29   -  call: ${chain_of_thought}
30         args:
31            examples:${demonstrations}
32             question: ${question}
33            model:${model}

## PDF page 16

8.2ToolCallingCode





                               1  description: tool use
                               2   defs:
                               3      tools:
                               4          data:
                               5     - name: calc
                               6              description: Calculator function
                               7       arguments:
                               8         expr:
                               9           type: string
                              10                     description: Arithmetic expression to calculate
                              11   text:
                              12 -  role: system
                              13      text: You are Granite, developed by IBM. You are a helpful AI assistant
                              14     with access to the following tools. When a tool is required to answer
                              15     the user's query, respond with <|tool_call|> followed by a JSON list of
                              16     tools used. If a tool does not exist in the provided list of tools,
                              17     notify the user that you do not have the ability to fulfill the request.
                              18      contribute: [context]
                              19 -  role: tools
                              20      text: ${tools}
                              21      contribute: [context]
                              22 - "Out of 1400 participants, 400 passed the test. What percentage is that?\n"
                              23 -  def: actions
                              24      model: replicate/ibm-granite/granite-3.1-8b-instruct
                              25      parser: json
                              26      spec: [{ name: str, arguments: { expr: str }}]
                              27 - "\n"
                              28 -  if: ${actions[0].name=="calc"}
                              29      then:
                              30          lang: python
                              31          code: result =${actions[0].arguments.expr}

                                              Figure A3: Basic example of a PDL program.

           SeeFigure A3.

8.3Optimization



      Algorithm1 Successive Halving for PDL Optimization
      Require:  ProgramcandidatesetC,validationdataset𝐷valid,initialvalidationsubsetsize𝑣min,max-
            imum validation subset size𝑣max
        1: 𝑣 ← 𝑣min
        2: while|C| > 1do
        3:     𝑑valid ← first𝑣 elements of𝐷valid s.t.𝑑valid ⊆ 𝐷valid and|𝑑valid| = 𝑣
        4:     for each candidate𝑐𝑖 ∈C do
        5:          Compute lossℓ𝑖 ←L(𝑐𝑖, 𝑑valid)
        6:     endfor
        7:    C ← top ⌊|C|/2⌋ candidates with lowest loss
        8:     𝑣 ← min(𝑣max,2·𝑣)
        9: endwhile
      10: return  Candidate inC with lowest loss


          Figure A4: Illustration of the Successive Halving algorithm used to optimize the PDL program by
                        pruning poor candidates on progressively larger validation subsets.


                                                                                                                                              16

## PDF page 17

Figure A4describes our optimization algorithm, based on successive halving (Jamieson et al.
     2016). The algorithm accepts a candidate set sampled from possible configurations and demon-
     strations,avalidationdatasettooptimizeagainst,aninitialvalidationsubsetsize,andamaximum
     validationsubsetsize. AutoPDLallowstheusertospecifytheseoptionsinaYAMLconfiguration
     file, and ultimately saves its result as a PDL program. This source-to-source transformation en-
     ablestheusertomodifyboththesearchspaceandtheresultingoptimizedPDLprogram,allowing
     further modification and execution.

8.4SearchSpace
     ThesearchspaceistheCartesianproductofthefollowingdiscretevariables,eachtakingonevalue
     per candidate:
     (1) 𝐴 ∈A={Zero-Shot, CoT, ReWOO, ReAct}, i.e., the overall prompting pattern to apply.
     (2) Numberofdemonstrations𝑛 ∈ {0, 3, 5}. Weselectedtheseoptionsasarepresentativesweep
         across no supervision, moderate few-shot use, and an upper-end case (in terms of token win-
         dow). Welimitedthesearchspacetothreeoptionstoavoidcombinatorialexplosionandlimit
         experiment cost.
     (3)If     𝐴 = ReAct, System prompt ∈ {Granite Tools,LLaMa 3,Granite LLaMa}. As the system
         prompt instructs the model how to format tool calls, it only has an effect on benchmarks
         withJSONtoolcalling(FEVER,GSM8K,andGSM-Hard)forcandidateswiththeReActprompt
         pattern. WenotethatonlyforMBPP+,ReWOOisnotincludedasapromptpattern,andthatwe
         always include two trajectories displaying iterative refinement, i.e., an example of a solution
         failing the example test case, followed by a passing solution, in line withWang et al. (2024).
         This effectively increases the number of trajectories to|𝑡𝑟𝑎𝑗|+2.

8.5AgentTrajectoryConstruction
     To optimize over agentic patterns and trajectories, we require a set of example trajectories to use
     during optimization. For this purpose, we create a basic agentic trajectory traj𝑖 for each training
     example⟨𝑥𝑖,𝑦𝑖⟩, following a rule-based transformation outlined below.
     GSM8K.  To demonstrate tool use in ReAct, we instead derive a trajectory traj as follows.  We
     exploit the fact that there is at most one expression per reasoning step, by iterating through the
     steps. At each step, we append a ‘thought’ to the trajectory, consisting of the text leading up to
     the math expression, concatenated with a reflection ‘I need to calculate’. We append a calculator
     tool call with the expression, and an ‘observation’,i.e., the result of the expression.  Finally, we
     append a thought ‘The answer is ...’, containing the ground truth answer, followed by thefinish
     action with the answer. We follow the same procedure to create ReWOO trajectories, except we
     use slightly different wording, e.g., ‘Calculate xyz’ in place of ‘I need to calculate xyz’, and omit
     the final thought and action.  Additionally, we use string substitution to replace any assumed
     expression results in the trajectory with the corresponding variable.
     FEVER. Toproduceagenttrajectories,weiterateovereacharticleassociatedwithaclaim,append
     a thought ‘I need to search for ...’, followed by the action, an observation containing the article
     summary, and finally a thought containing all the relevant sentences associated with that article
     for that claim, which we repeat for each article associated with a claim.  This procedure is not
     idealasthereisnoinherentordertothearticlesorsentences,eventhoughtheremaybeanatural
     orderingfollowingtheannotator’sWikipedianavigation. Finally,weappendathought‘Theclaim
     is true/false’ and the finish action, both with the ground truth answer. For chain-of-thought, we
     performthesameprocedureexceptweonlyincludetheconcatenatedevidencesentences,asthere
     is no tool use.


                                                                                                                     17

## PDF page 18

MBPP+. To generate sample agent trajectories from the training set, we follow the agent pattern
     (without feedback) in-context examples byWang et al. (2024), which consists of the problem              𝑥, a
     thoughtsuchas“Theintersectioniselementsthatareinbothlists”,anexecuteactionthatcontains
     proposed code and an assertion calling the proposed method with the test case input from the
     promptandcomparingitsoutput. Thisisthenfollowedbyanobservationcontainingtheexecution
     result, i.e.,either“[ExecutedSuccessfullywithNoOutput]”orastacktraceback. Thisallows
     the agent to iterate on solutions (up to five times in our implementation). We use the full MBPP
     train set of 374 problems as 𝐷train, and split the MBPP+ dataset into 𝐷valid and 𝐷test based on
     problem id membership in MBPP, leaving 39 and 224 validation and test problems respectively.
         To generate synthetic trajectories from the training set, we start with the natural language
     specification and single test case (the prompt), append the thought “I should run a solution on
     the test case before proposing a solution.”, followed by the ground truth solution and substitute
     in the prompt test case following the pattern[solution]res = ...; assert res == ...,
     ”Expected ... but got ”.format(res). Subsequently, we append the observation“[Executed
     Successfully with No Output]”, the thought“There is no more AssertionError. I can
     now submit the solution.”, and finally the solution action with the ground truth solution.
     This naive approach allows us to provide demonstration trajectories, albeit simplistic ones that
     assume the first solution is correct.  Sampling a reflection or thought from a strong model may
     be beneficial (Li et al. 2025), but we restrict our trajectories to rule based transformations.  As
     ReWOOisnotreactive, i.e.,withoutexecutionfeedback,itdoesnotmakesenseforMBPP.Hence,
     we exclude it from our experiments.

8.6ResultsPlot












        Figure A5: Comparisonofoptimizedpromptprogramperformanceacrossmodelsanddatasets. Each
                    barbell shows the accuracy improvement, if any, over the zero-shot baseline.

         InFigure A5, we visualize the results fromTable 1andTable 2.





















                                                                                                                     18

## PDF page 19

8.7Accuracy     vs. iterations
     InFigureA6,wevisualizetheaccuracyacrosscandidatesversustheiterationsoftheoptimization
     process, including a 95% confidence interval depicting the spread in accuracy across candidates.
     The confidence interval is computed using mean and 1,000 bootstraps. As the iterations increase,
     the number of candidates decreases, while the size of the validation set𝐷𝑣 increases.




























































                                                                                                                         19

## PDF page 20

Figure A6: Accuracy vs. iterations with 95% confidence interval.




                                                                                                       20
