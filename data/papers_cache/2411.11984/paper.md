## Understanding Chain-of-Thought in LLMs through Information Theory

## Jean-Francois Ton ∗ , 1 , Muhammad Faaiz Taufiq ∗ , 1 , Yang Liu 2

1 ByteDance Seed , 2 University of California, Santa Cruz

∗ All authors contributed equally; ordering determined through a coin flip.

## Abstract

Large Language Models (LLMs) have shown impressive performance in complex reasoning tasks through the use of Chain-of-Thought (CoT) reasoning, allowing models to break down problems into manageable sub-tasks. However, existing CoT evaluation techniques either require annotated CoT data or fall short in accurately assessing intermediate reasoning steps, leading to high rates of false positives. In this paper, we formalize CoT reasoning in LLMs through an information-theoretic lens. Specifically, our framework quantifies the 'information-gain' at each reasoning step, enabling the identification of failure modes in LLMs without the need for expensive annotated datasets. We demonstrate the efficacy of our approach through extensive experiments on toy arithmetic, GSM8K and PRM800k datasets, where it significantly outperforms existing outcome-based methods by providing more accurate insights into model performance on individual subtasks.

Correspondence: jeanfrancois@bytedance.com , faaiz.taufiq@bytedance.com

## 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, from complex reasoning to code generation [2-4, 23]. Many of these advances can be attributed to Chain-of-Thought (CoT) reasoning [17, 22, 32], which involves breaking down complex problems into a series of intermediate steps, mirroring human-like reasoning processes. The success of CoT reasoning, particularly in domains such as mathematics, logic, and multi-step decision-making, has led researchers to incorporate CoT-like features directly into model training, i.e. the FLAN family of models [5, 31].

This paper introduces a new formal framework for analyzing CoT in LLMs. We provide a rigorous method grounded in information theory, to evaluate the quality of each step in a model's reasoning process, thus offering insights beyond simple accuracy metrics to identify areas for improvement.

Previous work in this area has proposed ' Process Supervision ' [18], which requires expensive, human-annotated step-by-step data. While effective, this approach is often impractical due to the high cost and effort of creating large-scale annotated datasets. In turn, alternative methods have recently been proposed, such as outcome reward modelling [11] or the Math-Shepherd [29]. Both these approaches avoid reliance on costly annotated step-wise CoT data by instead modelling the correctness of each step based on the correctness of final outputs. However, as we show later, these methods can be unsound for detecting incorrect reasoning steps and can thus lead to a high false-positive rate in certain scenarios.

To address these shortcomings, we employ an information-theoretic approach, grounded in the following key insight: Each correct step in a reasoning process should provide valuable and relevant information that aids in predicting the final correct outcome . Building on this insight, we develop a framework to quantify the ' information-gain ' after each sub-task in the reasoning process, without the need for step-by-step annotations. This enables us to detect sub-tasks that fail to contribute meaningful information toward the correct solution, signalling potential errors or irrelevant steps in the model's reasoning. In addition, we also introduce a practical algorithm to assess LLM performance across various sub-tasks within a Chain-of-Thought (CoT) reasoning process. The key contributions of this paper are as follows:

1. We develop a framework for sequential applications of sub-tasks, e.g. Chain-of-Thought and provide a rigorous language to identify failure modes in LLMs.
2. Based on this framework, we propose a practical algorithm to assess the task-wise performance of models. This yields more granular information about a model's CoT performance without requiring annotated data for intermediate reasoning steps.
3. We validate our methods on extensive toy data, the GSM8K [6] as well as the PRM800K [18] dataset. Our method effectively identifies failure modes in CoT reasoning, unlike baselines like outcome reward modelling [11] and Math-Shepherd [29], which rely on final accuracy and tend to increase false positives in error detection.

## 2 Proposed Framework: Setup and Notation

Before diving into our framework, we first provide a high-level overview and notation on how LLM generation will be treated throughout this paper. This will allow us to set the foundation for describing our informationtheoretic framework. In particular, following the approach in [10], we view LLMs as abstract execution machines with a natural language interface. From this perspective, prompts are designed to solve specific problems (e.g., mathematical or logical problems), and the LLM processes the information in the prompt to generate an output.

We define a typical prompt as a combination of two parts:

1. An initial state, represented by the random variable X 0 ∈ X , encapsulates the prompt-provided information that the LLM processes to derive the queried result.
2. A task λ ∈ Υ (e.g., addition then multiplication) defines how the LLM processes X 0 .

Given the prompt, defined as a tuple ( X 0 , λ ) , the state X 1 represents the result of applying task λ to the initial state X 0 . Formally, we denote this using the update mapping Λ : X × Υ → X which outputs the updated state X 1 by applying the task λ on X 0 , i.e. X 1 = Λ( X 0 , λ ) . This updated state is then used to obtain the final output, denoted by Y ∈ X , by extracting only the information in X 1 which is relevant to the queried final answer. This notation defines a prompt that instructs a model to process information drawn from some initial distribution p ( X 0 ) (e.g., math problems).

We use a simple example to illustrate this notation:

<!-- formula-not-decoded -->

Here, the initial state x 0 denotes the information ' u = 12, v = 13' , and λ denotes the task of finding z (i.e. addition followed by multiplication). Next, x 1 = Λ( x 0 , λ ) represents the updated information after correctly performing the addition operation, i.e. x 1 is ' u = 12, v = 13 and z = 50' . The final output, y , is then obtained by simply extracting the value of z from x 1 , i.e. ' z = 50' .

Remark Our setup also encapsulates cases with ambiguous (or multiple correct) responses for a given task λ . In this case, Λ( x 0 , λ ) is a random variable with distribution p ( X 1 | X 0 = x 0 ) . Therefore, for generality, we treat Λ( x 0 , λ ) as a random variable from now on.

## 2.1 Compositionality

Many mathematical or logical problems, such as the one in (1), require sequential application of operations. Our notation is also amenable to such problems as it accommodates the composition of tasks.

Figure 1 Prompt (1) requires compositional application of tasks.

<!-- image -->

For example, one way to address prompt (1) involves first adding u and v , and next, multiplying the result by 2 to find z . Using our notation, this can be expressed as Λ( x 0 , λ 1 ◦ λ 2 ) , where λ 1 , λ 2 denote the addition and multiplication tasks respectively. The following property allows us to define the application of compositional task λ 1 ◦ λ 2 :

Definition 2.1. We say that an update rule Λ : X × Υ →X is compositionally consistent if, for all x 0 ∈ X and λ 1 , λ 2 ∈ Υ we have that Λ( x 0 , λ 1 ◦ λ 2 ) d = Λ(Λ( x 0 , λ 1 ) , λ 2 ) .

Here, d = denotes equality in distribution and is sufficient in cases where a query may have multiple correct responses.

Returning to the prompt in (1), Figure 1 shows that the model first computes u + v , then multiplies the result by 2 . Here, we refer to X 1 , X 2 as intermediate states and Y is the correct final output. In general, if a problem statement requires sequential application of T sub-tasks, λ = λ 1 ◦ . . . ◦ λ T , then the Chain-of-Thought (CoT) reasoning is divided up into T steps, where the t 'th step is recursively defined as X t = Λ( X t -1 , λ t ) for t ∈ { 1 , . . . , T } . Finally, the overall true output Y is obtained by extracting the queried information from the final state X T .

Having established a formal language for the sequential application of tasks, we now turn towards how a task may be divided into such a sequence of intermediate sub-tasks.

## 2.2 Primitive Tasks

In this subsection, we introduce the notion of primitive tasks which form the basic building blocks of any task. Intuitively, our formulation is reminiscent of ideas from linear algebra, where basis vectors form the basic building blocks of a vector space. In our case, any task λ ∈ Υ can be expressed as a sequence of primitive tasks. This decomposition will allow us to establish which tasks the model could have learned from the training data. For example, if a specific primitive task is not available in the LLM training data, it would be impossible for the model to execute any instructions which involve this primitive task correctly. With this in mind, we now introduce this concept formally:

Definition 2.2 (Primitive tasks) . We say that a set of tasks Γ ⊆ Υ is primitive if, for any task λ ∈ Υ , there exists a unique subset { λ i } k i =1 ⊆ Γ such that λ = λ 1 ◦ · · · ◦ λ k .

Note that the decomposition is not unique but the set of components is. In some cases, there may exist distinct permutations of primitive tasks which compose to yield the same task as is common in many associative operations. As an example, in the context of mathematical problem-solving, the basic arithmetic operations could be considered primitive. The composition of these primitive tasks allows us to construct extremely complex operations. Just like in linear algebra, we define the span of these tasks as the set obtained by their sequential applications.

Definition 2.3 (Span of tasks) . Let Φ ⊆ Υ be a set of tasks:

<!-- formula-not-decoded -->

The set Span (Φ) comprises all the tasks that can be applied by composing sub-tasks in the set Φ . This means that any compositionally consistent update rule Λ which is well-defined on the set of tasks Φ will also be well-defined on Span (Φ) . However, this Λ may still be ill-defined for any task not in this span. This limitation, known as unidentifiability, defines the boundaries of a model's inferences.

## 2.3 Unidentifiability

The unidentifiability of tasks forms a key part of our framework. It directly addresses the fundamental challenge that models, such as LLMs, face when dealing with unseen tasks. If a task λ lies outside of Span (Φ) , the span of tasks the model has been trained on, then the model cannot be expected to infer or apply it correctly. In other words, the model's capacity is constrained by the identifiability of tasks within the training set. This notion and formalization of unidentifiability allows us to highlight a critical limitation in the generalization of models: tasks not encountered during training cannot be reliably executed, as they remain beyond the model's learned task-span. More formally:

Definition 2.4 (Unidentifiability) . A task λ is unidentifiable in a set Φ ⊆ Υ if and only if λ ̸∈ Span (Φ) .

Remark In practice, unidentifiability may depend on the initial state X 0 , i.e. an LLM might accurately perform addition for 2-digit numbers but fail with 10-digit numbers [24]. For more details, see Appendix B.1.

Building on this framework, we propose an algorithm that integrates unidentifiability with information-theoretic methods to detect CoT reasoning failures.

## 3 Operationalising Our Framework

This section aims to operationalise the above framework to make inferences regarding the unidentifiability of intermediate sub-tasks in a model's CoT reasoning process. This would subsequently allow us to detect any sub-task at which a model's CoT reasoning process starts to diverge from the ground truth, thereby providing insights into how the model can be improved. For example, suppose we are in a setting where the 'addition' operation is unidentifiable, then we could further improve the model's mathematical reasoning by fine-tuning it on the addition operation.

## 3.1 An information-theoretic perspective

To apply unidentifiability in CoT generations, we introduce a fundamental assumption: each correct CoT step should provide meaningful information aiding the prediction of Y . If a step ceases to increase information about Y , it indicates an incorrect execution. We formalize this assumption using our notation from the previous section:

̸

Assumption 3.1 (Bayesian network) . Let λ = λ ′ be two operations with primitive decompositions:

<!-- formula-not-decoded -->

where λ ′ k is unidentifiable in { λ 1 , . . . , λ T } . Then, the intermediate states corresponding to the tasks λ, λ ′ have the Bayesian network in Figure 2.

Figure 2 Bayesian network corresponding to Assumption 3.1.

<!-- image -->

Intuition The Bayesian network in Figure 2 implies that if we encounter an unidentifiable task ( λ ′ k ) at step k of the reasoning path, the future states X i and X ′ j for any i, j ≥ k satisfy the conditional independence X i ⊥ ⊥ X ′ j | X k -1 . Consequently, once we apply λ ′ k , the subsequent states along the new reasoning path (in red) add no information regarding the subsequent states or the output of the original path (in green). Hence the figure represents the fact that, for any given input, the output of λ k (top fork) contains no information regarding the output of any other primitive task λ ′ k (bottom fork).

With our key assumption on the ground-truth CoT process formalized, we now consider the model's CoT behaviour.

## 3.2 Task execution in LLMs

To operationalise our framework, we formally distinguish between the model i.e. LLM's task execution and the ground truth process which arises from following the instructions correctly. To this end, we explicitly define how an LLM interprets a specified task λ using Λ M ( X 0 , λ ) , which is in general distinct from the ground truth update rule Λ( X 0 , λ ) .

Here, one option would be to consider the idealised setting where the model learns to perfectly follow some of the primitive tasks available in the training data. However, this may be considered too restrictive since in reality most LLMs do not always follow a 'learned' task perfectly. Instead, we consider a much weaker assumption that the model cannot correctly execute a task which is unidentifiable in the training data. Concretely, suppose Γ M ⊆ Γ denotes the primitive tasks available in the LLM training data. Then, we make the following assumption on LLM's task execution.

Assumption 3.2 (Task execution in LLMs) . Λ M is compositionally consistent and for any ( x 0 , λ ) ∈ X × Υ , there exists some ̂ λ ∈ Span (Γ M ) such that Λ M ( x 0 , λ ) d = Λ( x 0 , ̂ λ ) .

Intuition Assumption 3.2 means that for any task which we would like the LLM to apply, the LLM ends up executing some task in Span (Γ M ) which the model has been trained on. In other words, the model's execution is restricted only to the tasks which could be inferred from the training data (i.e. in Span (Γ M ) ). Moreover, this assumption also allows us to encapsulate cases where the model does not follow correct instructions or does not decompose a task correctly.

Before proceeding further with our main result which will allow us to test for the unidentifiability of sub-tasks, we define some notation which we will use from now onwards. Let λ = λ 1 ◦ . . . ◦ λ T denote a primitive decomposition of a task λ . Then, starting from an initial state X 0 , we denote the model's intermediate states recursively as:

<!-- formula-not-decoded -->

Moreover, we use Y M to denote the model's final output. Next, using this notation, we present the conditional independence which must hold if the model encounters an unidentifiable intermediate task along its reasoning path.

Theorem 3.3. Let Γ M ⊆ Γ denote the primitive tasks available in the training data. Let λ be a task with decomposition λ = λ 1 ◦ . . . ◦ λ T . If λ k is the first task in the decomposition of λ which is unidentifiable in Γ M (i.e. k = arg min t { λ t ̸∈ Span (Γ M ) } ). Then, under Assumptions 3.1 and 3.2, we have that

<!-- formula-not-decoded -->

Theorem 3.3 shows that under Assumptions 3.1 and 3.2, when the model encounters an unidentifiable task (i.e. λ k in Theorem 3.3) in its Chain-of-Thought reasoning, the model output satisfies the conditional independence in Equation (2). In practice, this means that if at step k , a model encounters a reasoning step which is necessary for obtaining the correct answer and is unidentifiable in the training data, the CoT reasoning diverges from the ground truth at this step and every subsequent step adds no additional information regarding the correct final output Y. This 'information' can be measured by checking if the model's confidence about the final output Y increases after each step. This is formalised in the next section.

## 3.3 Testing for unidentifiability using information-gain

With our framework established, we now describe how to detect unidentifiable sub-tasks using information theory. Following common practice [11, 29], we assume access to a dataset of prompts and correct final answers, derived by applying task λ , denoted as D λ := { ( x i 0 , y i ) } n i =1 . Recall that X j M and X j M -1 represent the model's chain of thought (CoT) reasoning at steps j and j -1 , respectively. Consequently, each element in the conditional independence statement in Equation (2) can be derived from the data and/or the model.

To this end, we consider the mutual information between Y and X j M conditional on X j M -1 , denoted by I ( Y ; X j M | X j M -1 ) . This conditional mutual information term intuitively represents the additional information contributed by the j 'th step of CoT, which is relevant for predicting the ground truth final output Y . Therefore, we refer to I ( Y ; X j M | X j M -1 ) as the information-gain at step j .

It follows from Theorem 3.3 that if an LLM encounters a sub-task at step i which is unidentifiable in its training data, no subsequent step should contribute any additional information relevant for predicting Y (i.e. the information-gain should remain 0 after step i ). If, on the other hand, we observe that I ( Y ; X j M | X j M -1 ) &gt; 0 for some j ≥ i , then under Assumptions 3.1 and 3.2, the task λ i is not unidentifiable. To estimate the information-gain in practice, we use the following result:

Proposition 3.4. Let I ( X ; Y | Z ) denote the mutual information between X and Y conditional on Z . Then,

<!-- formula-not-decoded -->

To estimate the information-gain in (3) using Proposition 3.4, we train a separate LLM, which we refer to as the supervisor model g sup . This model takes as input the model's CoT reasoning up to any given intermediate step t , X t M , and is fine-tuned to directly predict the ground truth final output Y . In this way g sup ( X t M ) approximates the conditional distribution p ( Y | X t M ) . Then, the quantity E [log p ( Y | X j M )] can be estimated using the negative cross-entropy loss for predicting Y , i.e., E [log p ( Y | X j M )] is approximately

<!-- formula-not-decoded -->

where l CE denotes the cross-entropy loss. From this, it follows that

<!-- formula-not-decoded -->

Summary The information-gain (IG) between steps j and j -1 reflects how much relevant information step j contributes towards predicting Y . If task λ j is executed correctly, this gain is positive, as indicated by a decrease in the cross-entropy loss. Conversely, if step j does not provide additional information, the loss remains unchanged. This can be interpreted as the conditional mutual information between X j M and Y , conditioned on X j M -1 . Positive information-gain suggests step j adds new insight about Y , while no gain indicates no added information. Training details for the supervisor model are in Appendix C.1.3.

Remark on sample-wise information-gain While conditional mutual information provides an aggregate measure of information-gain for a sub-task in a dataset, it may also be desirable to obtain an analogous measure of sub-task correctness for individual CoT instances. This could be useful, for example, in detecting which step is wrong for a given prompt. Our notion of information-gain can be extended to this sample-wise setting, similar to [8], by instead considering the following difference

<!-- formula-not-decoded -->

Intuitively, if step j in the model's CoT is correct, the model should become more confident in the ground truth output Y being the correct final answer. Therefore, the difference above should be positive. Conversely, if step j is wrong, the model's confidence regarding Y should not increase, and this difference should be ≤ 0 . From now on, we refer to the difference in (5) as sample-wise information-gain at step j .

Remark on O1/R1 style reasoning Although we present our framework using linear chains-of-thought for clarity, the information-gain metric naturally accommodates the more complex reasoning patterns found in O1/R1-style models. These reasoning models often explore multiple solution paths, backtrack when encountering errors, and dynamically self-correct their approach. In such exploratory settings, our framework remains effective: steps along incorrect reasoning trajectories will exhibit low or negative information gain, indicating they do not contribute meaningfully toward the correct final answer. When the model identifies a more promising path and self-corrects, subsequent steps will show positive information gain, signaling productive progress.

This adaptability to varied reasoning structures is empirically demonstrated in our experiments (Section 5), where we analyse reasoning traces from the MATH/PRM dataset. Steps that human annotators labelled as uninformative or irrelevant consistently show low information gain under our metric, while correct and meaningful steps exhibit high information gain. Thus, our method provides reliable step-wise evaluation regardless of whether the reasoning follows a linear chain or involves branched/exploratory patterns.

We further formalize this remark using our framework in Appendix B.4.

## 4 Related Works

Evaluation of CoT reasoning Several recent works propose methodologies for evaluating CoT reasoning [11, 15, 16, 21, 28, 32, 35, 37]. For example, [16] verifies individual steps in a model's CoT reasoning by generating multiple LLM responses per prompt and comparing correct responses with incorrect ones.

Similarly, [29, 30] use a fine-tuned LLM to decode multiple reasoning paths from each step and check the correctness of these reasoning paths. However, as we show in our experiments, approaches which simply rely on the correctness of the final output are not sound in general and can lead to false positives. Moreover, these solutions may not be plausible for problems of high difficulty where correct LLM responses might be scarce.

Formalising CoT framework The formalisation of LLM reasoning remains an active area of research. Most notably [10] introduces a formal framework for LLMs and is a key source of inspiration behind our formalism. Additionally, [9] theoretically examines the expressivity of LLMs with CoT in solving mathematical and decision-making problems, focusing on the transformer architecture's implications on accuracy. Besides this, [36] provides a formal definition of hallucinations, but does not consider CoT reasoning specifically.

Reward modelling Outcome-based reward models (ORM) [6, 11, 18] predict the probability of reaching the correct final answer based on a model's intermediate CoT steps. While they avoid requiring correct intermediate demonstrations, we show in Section 5 that they are unsound for detecting CoT reasoning errors. Step-wise ORM (SORM) [11] extends ORM by estimating the probability of an 'optimal' model reaching a correct answer but requires training a larger, more capable model than the base model.

Process-based reward modelling (PRMs) [18, 27] is an alternative approach which directly predicts the correctness of intermediate CoT reasoning steps. Likewise, various other approaches rely on annotated CoT datasets for benchmarking [1, 14, 19-21, 34, 35, 37]. While these benchmarks and methodologies aid LLM reasoning, collecting annotated data is costly and not easily scalable. In contrast, our approach evaluates an LLM's CoT reasoning without human-annotated CoT data.

## 5 Experiments

In this section, we demonstrate our framework's utility, dubbed Information-Gain (IG) and compare against two baselines for detecting errors in a model's CoT reasoning. Here we assume access only to the model's CoT generations X 0 , X M 1 , . . . , X T M and correct final answers Y .

Outcome Reward Model (ORM) [11] This involves training a classifier, denoted as f ORM , which takes as input model generations up to any step t in its CoT reasoning, X t M , and predicts the probability of the model's final answer being correct, i.e.

<!-- formula-not-decoded -->

Here, if we observe that this probability of correctness drops significantly after step t , i.e. f ORM ( X t M ) ≫ f ORM ( X t M +1 ) , this indicates that the model applies task λ t +1 incorrectly.

Math-Shepherd (MS) [29] This method quantifies the potential for a given reasoning process X t M by using a 'completer' model to generate N completions of each reasoning process starting from step t , { ( X t M , X t M +1 ,j , . . . , X M T,j , Y j M ) } j ≤ N , where Y j M denotes the final answer reached in the j 'th completion. Then, we estimate the potential of this step based on the proportion of correct answers among the N completions:

<!-- formula-not-decoded -->

For a fair comparison, we do not assume access to a 'verifier' model more capable than our base model. Therefore, we use the base model as the completer model in our experiments.

## 5.1 Toy data experiments

We first consider a toy setting where we control model behaviour across tasks. Prompts consist of an integer vector Z 0 ∈ Z 5 sampled from a given distribution. The task λ comprises five steps, λ = λ 1 ◦ . . . ◦ λ 5 , where each sub-task λ i transforms Z i -1 ∈ Z 5 into Z i ∈ Z 5 . The correct final answer Y is Z 5 . Additional details on data generation and sub-tasks are in Appendix C.1.

Generating the dataset To investigate partial unidentifiability for a given task λ i we modify the obtained dataset by introducing 'noise' at step i . In other words, the task λ i is applied incorrectly on a subset of the data, whereas all other tasks are always applied correctly. This represents a model which sometimes fails at step i and we use 'LLM i ' to denote this model in this experiment. We repeat this procedure for all tasks λ i for i ∈ { 1 , . . . , 5 } which yields 5 LLMs { LLM 1 , . . . , LLM 5 } .

To assess robustness, we introduce a special case in LLM 3 , where task λ 3 is applied incorrectly iff the output after λ 2 falls in a set S . This deliberate choice highlights a pitfall of existing baselines and contrasts with other LLMs, where errors occur randomly. In other words, λ 3 's correctness depends on λ 2 's output. For details, see Appendix C.1.2.

## 5.1.1 Results

̸

Figure 3 shows how the different baselines quantify the correctness of the different tasks for the 5 different LLMs under consideration. This figure only considers samples where the final answer of the LLM was incorrect, i.e. Y M = Y . For our method (IG), Figure 3a shows the information-gain across the different steps for each LLM. Likewise, Figure 3b presents the results for ORM and shows how the average probability of correctness in (6) changes across the different steps, whereas, for Math-Shepherd, Figure 3c shows the proportion of correct completions starting after each step (7). Here, any significant drop in the plotted values indicate an incorrect application of a task.

Information-gain rightly quantifies step-wise correctness We observe that the information-gain remains positive for each LLM until we encounter an incorrect reasoning step, at which point it drops to negative values. Therefore, our method can identify the incorrectly executed task for each LLM under consideration. We used a GPT-2 supervisor model to estimate information-gain.

Pitfall of the baselines While ORM and Math-Shepherd usually identify incorrect reasoning steps, they fail for LLM 3 . This is because λ 3 is misapplied iff the output after λ 2 lies in S . Thus, the classifier can predict the final output's correctness at λ 2 by checking if Z 2 lies in S , leading to overconfidence in error detection at λ 2 instead of λ 3 .

Similarly, with Math-Shepherd for LLM 3 (using the same model as a completer), a completion is incorrect if the output after λ 2 lies in S . Here, all completions fail, regardless of the starting step, making it impossible to pinpoint where LLM 3 goes wrong.

Table 1 Sample-wise classification of a sub-task for LLM 3 .

|           |   Acc ↑ |   TPR ↑ |   FPR ↓ |
|-----------|---------|---------|---------|
| IG (Ours) |    0.96 |    0.98 |    0.06 |
| ORM       |    0.77 |    0.98 |    0.54 |
| MS        |    0.60 |     1.0 |     1.0 |

Figure 3 Heatmaps (left) quantifying the correctness of different sub-tasks for the 5 LLMs using the different baselines, and the associated classification metrics (right). Red color in the heatmaps indicates a significant drop in the plotted metrics (an incorrectly executed sub-task).

<!-- image -->

Sample-wise detection We also use the different baselines for sample-wise detection of erroneous steps, as outlined in Section 3.3. A step is classified as incorrect if a baseline's metric falls below a threshold. Table 1 presents the results for LLM 3 , with optimal thresholds chosen from a held-out dataset. Our method achieves significantly higher accuracy and fewer false positives than the baselines, making it more reliable for sample-wise error detection.

## 5.2 Arithmetic operations on Llama-3-8B

Following our toy experiments, we now evaluate our framework in a more realistic setting using the Llama-3-8B model [7]. We focus on a simple arithmetic task that involves both multiplication and addition. The goal is to assess the model's performance on each operation.

Experimental setup We sample two integers x and y uniformly from [1 , 10 5 ) . The prompt to the model is:

<!-- formula-not-decoded -->

Model Accuracy The model's accuracy across steps is:

<!-- formula-not-decoded -->

Most failures occur in Step 3, which involves adding previously computed values. Analyzing the ( x, y ) distribution where the model is incorrect (Figure 4), we find that errors mainly arise when one variable is large and the other is small. This suggests that correctness depends heavily on ( x, y ) , making it difficult for baselines to pinpoint the erroneous step in the model's CoT reasoning.

## 5.2.1 Results

Our method We trained the supervisor model by fine-tuning a Llama-3-8b model using Low Rank Adaptation (LoRA) [13]. Table 2 shows that there is a significant drop in information-gain at step 3 relative to steps 1 and 2, demonstrating that our method correctly identifies that the failure mainly occurs at step 3.

Outcome Reward Model (ORM) For ORM, the mean probability of correctness (Table 2) remains unchanged at each step. Figure 4 suggests this is because ORM predicts correctness based solely on x and y in the prompt. Crucially, its confidence remains constant even as intermediate reasoning steps are added, preventing it from distinguishing the model's performance at different steps.

Math-Shepherd (MS) Table 2 shows the proportion of correct completions for MS. While this is low at step 3, only 5-7% of completions from steps 1 and 2 yield a correct output, despite errors mostly occurring at step

Figure 4 The distribution of ( x, y ) for incorrect samples: Llama-3-8B struggles to add large and small numbers (represented by the top-left and bottom-right shaded regions).

<!-- image -->

3. This is because Llama-3-8B's correctness is largely determined by ( x, y ) in the prompt (Figure 4). As a result, MS frequently mislabels steps 1 and 2 as incorrect, leading to a higher false positive rate compared to our baseline.

Table 2 Experimental results for Toy Arithmetic, GSM8K and PRM800K experiments. In each of the datasets, we denote the 'correct" and 'wrong" steps with ✓ and ✗ respectively.

| Datasets & Methods   | Datasets & Methods   | Operations      | Operations       | Operations         | Operations    | Sample-wise detection metrics   | Sample-wise detection metrics   | Sample-wise detection metrics   | Sample-wise detection metrics   |
|----------------------|----------------------|-----------------|------------------|--------------------|---------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|
| Toy Arithmetic       | Methods              | Step 1: 3 x ✓   | Step 2: 2 y ✓    | Step 3: 3 x +2 y ✗ | -             |                                 | Accuracy ↑                      | TPR ↑                           | FPR ↓                           |
|                      | IG (Ours) ORM MS     | 0.67 0.24 0.068 | 0.24 0.24 0.059  | 0.027 0.24 0.00069 | - - -         |                                 | 0.76 0.56 0.53                  | 0.51 0.10 0.99                  | 0.02 0.07 0.86                  |
| GSM8K                | Methods              | Addition ✓      | Multiplication ✗ | Division ✓         | Subtraction ✓ |                                 | Accuracy ↑                      | TPR ↑                           | FPR ↓                           |
|                      | IG (Ours) ORM        | 0.99 0.46       | 0.026 0.024      | 1.05 0.38          | 1.06 0.013    |                                 | 0.72 0.58                       | 0.95 1.00                       | 0.62 1.00                       |
| PRM800K              | Methods              | Negative ✗      | Neutral ✗        | Positive ✓         | -             |                                 | Accuracy ↑                      | TPR ↑                           | FPR ↓                           |
| PRM800K              | IG (Ours) ORM        | 0.058 0.734     | -0.011 0.745     | 0.168 0.744        | - -           |                                 | 0.74 0.69                       | 0.84 0.55                       | 0.37 0.18                       |

Sample-wise detection When using these methods for sample-wise detection of incorrect steps, our approach yields the highest accuracy among the baselines considered. This superior performance is attributed to the fact that baselines like ORM and MS often falsely flag steps 1 and 2 as incorrect, as evidenced by their high FPRs in Table 2.

## 5.3 Experiments on the controlled GSM8K Dataset

To evaluate our method on a complex dataset, we conducted experiments on GSM8K [6], controlling specific factors for more interpretable results.

We begin by using GPT-4 [23] to generate answers for GSM8K questions where the 'multiplication' operation is always done incorrectly, while all other operations are correct. Next, we filtered the dataset to ensure that 'multiplication', 'subtraction', and 'addition' never appeared together within the same Chain of Thought (CoT) solution. In particular, we ensured in our setting that, all incorrect final answers included both 'multiplication' and 'subtraction', whereas correct final answers did not involve either operation. This introduces a spurious correlation between 'subtraction' and wrong answers.

In this setup, we mainly focused on evaluating ORM and our proposed method, as MS (with the same completer) fails trivially under these conditions. Specifically, 'multiplication' is inherently unidentifiable, since any CoT containing 'multiplication' negates the influence of other sub-tasks by design. Further details on the experimental setup can be found in Appendix C.3.

## 5.3.1 Results

Table 2 shows that our information-theoretic approach (IG) successfully identifies the unidentifiable sub-task. Since the 'multiplication' rule is intentionally incorrect, it yields minimal to no information gain, as expected. However, ORM results reveal a different pattern: both 'multiplication' and 'subtraction' have low correctness probabilities, as they are linked to incorrect final answers. This suggests that the standard ORM approach may misleadingly classify 'subtraction' as incorrect.

Additionally, in our sample-wise experiment, we observe a similar trend when we use the methods to assess the sample-wise correctness of 'multiplication" and 'subtraction" for each prompt. Here, our proposed method not only accurately detects the unidentifiable sub-task but also highlights a significant shortcoming of ORM. Specifically, ORM falsely flags 'subtraction', which is actually correct, as an incorrect sub-task due to spurious correlations.

## 5.4 Experiments on the PRM800K dataset

To further demonstrate the practical applicability of our method, we have conducted an additional experiment on OpenAI's PRM800k dataset [18] which is obtained by labeling the intermediate steps of the MATH dataset [12].

More specifically, this dataset is a process supervision dataset with step-level correctness labels for modelgenerated solutions to MATH problems. To create it, Lightman et al. [18] asked human annotators to label each step from fine-tuned GPT-4 solutions as positive (+1), negative (-1), or neutral (0). A positive label indicates a correct, reasonable step; a negative label denotes an incorrect or unreasonable step; and a neutral label indicates ambiguity (e.g., subtly misleading or technically valid yet poor).

The objective is to identify incorrect Chain-of-Thought (CoT) steps, specifically those labelled as (-1) by annotators, using our method as well as ORM. However, we do not utilize the step-wise labels during the process; they are only used for evaluation purposes. Since the base GPT-4 model used to generate the PRM data is not publicly available, we were unable to obtain MS results for this dataset.

## 5.4.1 Results

Table 2 shows the information gain and mean correctness probability for positive, negative, and neutral sub-steps.

As expected, these results show that the information-gain is significantly lower for incorrect steps (with labels -1) than for labels +1. Additionally, we also observe that the information-gain is negative for neutral steps (with labels 0), which is explained by the fact that these steps do not add any useful information regarding the ground truth (as these were deemed irrelevant/ambiguous by the human labellers). In contrast, the average probability of correctness for the ORM classifier is roughly the same across each label and, on average, is not very informative.

Sample-wise detection Additionally, we also used the sample-wise information-gain (IG) as well as the ORM baseline to classify if a step is correct (as outlined in Section 3.3). To avoid ambiguity, we filtered out the neutral sub-steps (with labels 0) for this experiment and considered a balanced held-out dataset with equal number of correct and incorrect steps. Table 2 also shows the sample-wise results for both methods (where we chose the best thresholds for each baseline using a held-out dataset).

It can be seen that the accuracy of our method is higher than that of the ORM classifier. Additionally, our method also leads to higher TPR (and hence a lower FNR) than the ORM classifier. These results show that our method outperforms the outcome-based baselines on more complex datasets such as the MATH data as well.

## 6 Discussion and Limitations

In this paper, we introduce a novel information-theoretic approach to evaluate Chain-of-Thought (CoT) reasoning in LLMs without annotated intermediate steps. Our framework effectively identifies erroneous reasoning across diverse settings and consistently outperforms baselines, including Outcome Reward Models (ORMs) [11] and Math-Shepherd (MS) [29]. However, our approach does have some limitations.

Although our method avoids human-annotated step-wise data, it requires additional training of the supervisor model, which is computationally expensive. Future work could explore in-context learning to estimate information gain, reducing training needs and improving efficiency. Additionally, while our method does not require correctness labels for every step, we still need to categorize each step according to its respective sub-task. However, this limitation is not unique to our method, as both ORM and MS also rely on such labels to draw sub-task-specific conclusions.

Lastly, while we focus on logical and mathematical datasets, our method also extends to other domains requiring CoT reasoning, such as Blocks World [25]. As we discuss in Appendix B.2, this is an interesting avenue which we leave for future research.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## Acknowledgements

We would like to thank Li Hang for his valuable insights and guidance during the development of this work.

## References

- [1] Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. Mathqa: Towards interpretable math word problem solving with operation-based formalisms. CoRR, abs/1905.13319, 2019. URL http://arxiv.org/abs/1905.13319 .
- [2] Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov, and Yonghui Wu. Palm 2 technical report, 2023. URL https://arxiv.org/abs/2305.10403 .
- [3] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, and Yi Zhang. Sparks of artificial general intelligence: Early experiments with gpt-4, 2023. URL https://arxiv.org/abs/2303.12712 .
- [4] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sashank
5. Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: scaling language modeling with pathways. J. Mach. Learn. Res., 24(1), mar 2024. ISSN 1532-4435.
- [5] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-finetuned language models, 2022. URL https://arxiv.org/abs/2210.11416 .
- [6] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110.14168 .
- [7] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaoqing Ellen Tan, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aaron Grattafiori, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alex Vaughan, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Franco, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina

Mejia, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, Danny Wyatt, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Firat Ozgenel, Francesco Caggioni, Francisco Guzmán, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Govind Thattai, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Karthik Prasad, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kun Huang, Kunal Chawla, Kushal Lakhotia, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Maria Tsimpoukelli, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikolay Pavlovich Laptev, Ning Dong, Ning Zhang, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Rohan Maheswari, Russ Howes, Ruty Rinott, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Kohler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vítor Albiero, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaofang Wang, Xiaojian Wu, Xiaolan Wang, Xide Xia, Xilun Wu, Xinbo Gao, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yuchen Hao, Yundi Qian, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, and Zhiwei Zhao. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783 .

- [8] Kawin Ethayarajh, Yejin Choi, and Swabha Swayamdipta. Understanding dataset difficulty with V -usable information. In International Conference on Machine Learning, pages 5988-6008. PMLR, 2022.
- [9] Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, and Liwei Wang. Towards revealing the mystery behind chain of thought: A theoretical perspective. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=qHrADgAdYu .
- [10] Javier González and Aditya V Nori. Beyond words: A mathematical framework for interpreting large language models. arXiv preprint arXiv:2311.03033, 2023.
- [11] Alexander Havrilla, Sharath Chandra Raparthy, Christoforos Nalmpantis, Jane Dwivedi-Yu, Maksym Zhuravinskyi, Eric Hambro, and Roberta Raileanu. GLore: When, where, and how to improve LLM reasoning via global and local refinements. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview. net/forum?id=LH6R06NxdB .
- [12] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https: //openreview.net/forum?id=7Bywt2mQsCe .
- [13] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. CoRR, abs/2106.09685, 2021. URL https://arxiv.org/abs/ 2106.09685 .
- [14] Alon Jacovi, Yonatan Bitton, Bernd Bohnet, Jonathan Herzig, Or Honovich, Michael Tseng, Michael Collins, Roee Aharoni, and Mor Geva. A chain-of-thought is as strong as its weakest link: A benchmark for verifiers of reasoning chains, 2024.
- [15] Nitish Joshi, Hanlin Zhang, Koushik Kalyanaraman, Zhiting Hu, Kumar Chellapilla, He He, and Li Erran Li. Improving multi-hop reasoning in LLMs by learning from rich human feedback. In Neuro-Symbolic Learning and Reasoning in the era of Large Language Models, 2023. URL https://openreview.net/forum?id=wxfqhp9bNR .
- [16] Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making large language models better reasoners with step-aware verifier, 2023.
- [17] Zhiyuan Li, Hong Liu, Denny Zhou, and Tengyu Ma. Chain of thought empowers transformers to solve inherently serial problems, 2024. URL https://arxiv.org/abs/2402.12875 .
- [18] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step, 2023.
- [19] Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning. CoRR, abs/2007.08124, 2020. URL https: //arxiv.org/abs/2007.08124 .
- [20] Sean McLeish, Arpit Bansal, Alex Stein, Neel Jain, John Kirchenbauer, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Jonas Geiping, Avi Schwarzschild, and Tom Goldstein. Transformers can do arithmetic with the right embeddings, 2024.
- [21] Minh-Vuong Nguyen, Linhao Luo, Fatemeh Shiri, Dinh Q. Phung, Yuan-Fang Li, Thuy-Trang Vu, and Gholamreza Haffari. Direct evaluation of chain-of-thought in multi-hop reasoning with knowledge graphs. ArXiv, abs/2402.11199, 2024. URL https://api.semanticscholar.org/CorpusID:267751000 .
- [22] Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena. Show your work: Scratchpads for intermediate computation with language models, 2021. URL https://arxiv.org/abs/2112. 00114 .
- [23] OpenAI. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774 .
- [24] Yasaman Razeghi, Robert L. Logan IV au2, Matt Gardner, and Sameer Singh. Impact of pretraining term frequencies on few-shot reasoning, 2022. URL https://arxiv.org/abs/2202.07206 .
- [25] John K. Slaney and Sylvie Thiébaux. Blocks world revisited. Artificial Intelligence, 125(1-2):119-153, 2001.
- [26] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question answering challenge targeting commonsense knowledge, 2019. URL https://arxiv.org/abs/1811.00937 .
- [27] Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process- and outcome-based feedback, 2022.
- [28] Boshi Wang, Xiang Yue, Yu Su, and Huan Sun. Grokked transformers are implicit reasoners: A mechanistic journey to the edge of generalization, 2024.
- [29] Peiyi Wang, Lei Li, Zhihong Shao, R. X. Xu, Damai Dai, Yifei Li, Deli Chen, Y. Wu, and Zhifang Sui. Mathshepherd: Verify and reinforce llms step-by-step without human annotations, 2024. URL https://arxiv.org/ abs/2312.08935 .
- [30] Zihan Wang, Yunxuan Li, Yuexin Wu, Liangchen Luo, Le Hou, Hongkun Yu, and Jingbo Shang. Multi-step problem solving through a verifier: An empirical analysis on model-induced process supervision, 2024. URL https://arxiv.org/abs/2402.02658 .
- [31] Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=gEZrGCozdqR .
- [32] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS '22, Red Hook, NY, USA, 2024. Curran Associates Inc. ISBN 9781713871088.
- [33] Terry Winograd. Procedures as a representation for data in a computer program for understanding natural language. In MIT AI Technical Report, 1972.
- [34] Zhiheng Xi, Wenxiang Chen, Boyang Hong, Senjie Jin, Rui Zheng, Wei He, Yiwen Ding, Shichun Liu, Xin Guo, Junzhe Wang, Honglin Guo, Wei Shen, Xiaoran Fan, Yuhao Zhou, Shihan Dou, Xiao Wang, Xinbo Zhang, Peng Sun, Tao Gui, Qi Zhang, and Xuanjing Huang. Training large language models for reasoning through reverse curriculum reinforcement learning, 2024.
- [35] Xuan Xie, Jiayang Song, Zhehua Zhou, Yuheng Huang, Da Song, and Lei Ma. Online safety analysis for llms: a benchmark, an assessment, and a path forward, 2024.
- [36] Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. Hallucination is inevitable: An innate limitation of large language models, 2024.
- [37] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T. Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models, 2024.

## Appendix

## A Proofs

Proof of Theorem 3.3. Suppose λ and λ ′ are two tasks with primitive decompositions

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where arg min t { λ t ̸∈ Span ( { λ ′ 1 , . . . , λ ′ T ′ } ) } ≤ k . In other words, the primitive decompositions of λ ′ and λ diverge before step k +1 . Then, Assumption 3.1 implies that for any j ≥ k , we have that the answer Y and X ′ j are d-separated by X ′ j -1 . Therefore,

<!-- formula-not-decoded -->

Next, we know from Assumption 3.2 that there exists some task ˆ λ ∈ Span (Γ M ) (possibly dependent on X 0 and λ ) such that Λ M ( X 0 , λ ) d = Λ( X 0 , ˆ λ ) . Suppose that ˆ λ has primitive decomposition

<!-- formula-not-decoded -->

then since ˆ λ ∈ Span (Γ M ) , we know that ˜ λ i ∈ Γ M for i ∈ { 1 , . . . , ˜ T } . If the primitive decomposition of λ in (8) is such that k = arg min t { λ t ̸∈ Span (Γ M ) } , then we know that arg min t { λ t ̸∈ Span ( { ˜ λ 1 , . . . , ˜ λ ˜ T } ) } ≤ k . Then, from the above it follows that

<!-- formula-not-decoded -->

Here, we used the fact that X j M d = Λ( X 0 , ˜ λ 1 ◦ · · · ◦ ˜ λ j ) using Assumption 3.2.

Proof of Proposition 3.4.

<!-- formula-not-decoded -->

Here, the second equality above arises from the fact that X j M also captures all the information captured in X j M -1 (and possibly more). Therefore, conditional on X j M , the state X j M -1 is deterministic and hence, Y ⊥ ⊥ X j M -1 | X j M .

## A.1 Symmetry property of information-gain I ( Y, X j M | X j M -1 )

The mutual information between two random variables X and Y , I ( X,Y ) , is symmetric in its arguments (i.e., w.r.t. X and Y ). However, the conditional mutual information I ( Y, X j M | X j M -1 ) satisfies a symmetry property conditional on X j M -1 . Formally, this property of the information-gain term can be expressed as follows:

There exists some functional F : P × P × P → R where P is the space of probability distributions, such that

and

1. The information-gain I ( Y, X j M | X j M -1 ) can be expressed as:

<!-- formula-not-decoded -->

where P Z denotes the distribution of the random variable Z and

2. F is symmetric w.r.t. its first two arguments, i.e. F ( p, q, r ) = F ( q, p, r ) . Note that there is no symmetry requirement w.r.t. the third argument of F because the joint distribution P Y,X j M | X j M -1 is already symmetric w.r.t. Y and X j M i.e. P Y,X j M | X j M -1 = P X j M ,Y | X j M -1 .

It follows from Eq. (9) in the proof of Proposition 3.4, that the functional F which satisfies the above conditions is

<!-- formula-not-decoded -->

## B Additional details of our framework

## B.1 State-conditioned unidentifiability

In practice, the concept of unidentifiability may depend on the initial state X 0 . For instance, an LLM might accurately perform addition for 2-digit numbers but fail with 10-digit numbers [24]. Our framework can be extended to account for such cases by explicitly incorporating the distribution of initial states into the notion of identifiability. For example, addition could be considered unidentifiable when the initial state distribution is p ( X 0 | X 0 includes 10-digit numbers ) . However, for simplicity, we keep this distributional dependence implicit in our framework.

## B.2 How to define steps beyond mathematical datasets

The framework presented in this paper primarily considers examples related to mathematical reasoning, where the definition of primitive tasks is intuitive and well-structured. However, our methodology could be applied to other domains where the identification of primitive tasks is less straightforward, such as Blocks World [25, 33] and commonsense question answering (QA) [26].

In the case of Blocks World, the primary task of planning can be decomposed into sub-tasks involving sequences of primitive actions, such as 'stack", 'unstack" and 'move". Applying the information-gain methodology in this context could provide insights into the effectiveness of large language models (LLMs) in planning and executing these sub-tasks. By analyzing the information-gain for each step, it would be possible to assess where the model's reasoning process is effective and where it encounters difficulties.

Similarly, in commonsense QA, the chain-of-thought (CoT) reasoning steps can be categorized into distinct types, such as causal reasoning (identifying cause-and-effect relationships), temporal reasoning (understanding sequences and timing), and spatial reasoning (comprehending physical arrangements and object relationships). These reasoning types align naturally with the proposed framework, enabling a systematic evaluation of the LLM's decision-making process within each category.

While these extensions present promising directions for future research, they also introduce additional challenges, particularly regarding the assumptions underlying our methodology. Addressing these challenges and validating the framework across diverse domains remains an open avenue for further investigation.

## B.3 Correct final answers with incorrect intermediate steps

A notable phenomenon of chain-of-thought (CoT) reasoning is the occurrence of cases where a model arrives at the correct final answer despite containing errors in intermediate steps. This scenario raises important questions regarding the validity of intermediate reasoning and the implications for evaluating model performance.

The methodology proposed in this paper estimates the information contributed by each successive reasoning step toward the final correct answer. In instances where an intermediate step is incorrect, it is expected that this step contributes no additional relevant information, resulting in an information-gain of zero at that point-regardless of whether the final answer is ultimately correct. Conversely, if a model systematically produces incorrect intermediate steps that nevertheless lead to the correct final answer, baseline methods such as ORM and Math Shepherd would fail to detect such errors, as these approaches primarily assess correctness based on the final output.

To better understand the prevalence of this phenomenon, we conducted an additional analysis of our datasets. In our arithmetic experiments (Section 5.2), we found that only 1.2% of samples exhibited this behavior, where an intermediate step was incorrect but the final answer remained correct. In the synthetic toy experiment, such cases did not occur, as we maintained a degree of control over the data generation process. Given the low frequency of these occurrences in our settings, their impact on the overall effectiveness of our framework is minimal.

## B.4 Clarification on Assumption 3.1 and Non-Linear Reasoning Structures

Overview of Assumption 3.1 Assumption 3.1 specifically addresses a particular class of reasoning steps: those that are necessary for solving a problem but unidentifiable in the training data, meaning no composition of learned tasks can yield that step. For such unidentifiable steps, we assume that once the model diverges at this point, subsequent steps do not add further information toward the correct final output. This assumption applies only to steps that cannot be executed through any combination of the model's learned capabilities.

Application to Reasoning Systems Although our framework is presented using linear chains-of-thought for clarity, the information-gain metric naturally accommodates the more complex reasoning patterns found in modern systems like O1/R1-style models. These reasoning models often explore multiple solution paths, backtrack when encountering errors, and dynamically self-correct their approach. Our framework handles such non-linear reasoning structures through its information-theoretic foundation, without requiring modifications to the core assumptions.

Mathematical Formalization Consider a scenario where a model explores alternative paths during reasoning. Let the correct reasoning path to final answer Y be:

<!-- formula-not-decoded -->

If the model temporarily explores an incorrect or exploratory path through some intermediate step Z at position t , but then returns to continue correctly, the full reasoning trace becomes:

<!-- formula-not-decoded -->

Our framework evaluates such paths through conditional mutual information. Since Y ⊥ Z | X t (the final answer is conditionally independent of the exploratory step given the state at X t ), the information gain at step Z will be zero or negative. This correctly indicates that the exploratory step Z does not contribute meaningful information toward the final answer. Once the model returns to the productive path at X t , subsequent steps will exhibit positive information gain, reflecting meaningful progress toward Y .

Distinction Between Unidentifiable and Exploratory Steps It is crucial to distinguish between two types of problematic steps:

- Unidentifiable steps (addressed by Assumption 3.1): Steps that are necessary for the solution but cannot be executed through any composition of the model's learned tasks. These represent fundamental gaps in the model's capabilities.
- Exploratory or incorrect steps : Steps where the model temporarily pursues an unproductive path but can self-correct using its existing capabilities. These steps will show low or negative information gain but do not prevent the model from eventually finding the correct solution.

Self-corrected steps demonstrate that the model possesses the necessary learned operations to eventually find the correct path, whereas unidentifiable steps represent gaps that cannot be bridged through any composition of training tasks.

Empirical Validation Our experiments on the MATH/PRM dataset (Section 5) confirm this theoretical framework: steps labeled as uninformative or incorrect by human annotators consistently show low or negative information gain, while correct steps exhibit high positive information gain. This demonstrates that our method reliably evaluates both linear and non-linear reasoning patterns without requiring special handling for self-corrections or exploratory paths.

## C Additional Experimental Details

## C.1 Toy Data Experiments

In this section, we describe the exact procedure used to generate the toy data for training and evaluating the models in our experiments. The dataset is constructed through five sequential operations (or tasks) applied to an initial state z 0 , where each task λ i generates an intermediate state z i . Both correct and incorrect examples were generated, with incorrect examples created by introducing random noise or permutations into the transformations.

The data was used to represent models LLM 1 , LLM 2 , ..., LLM 5 , each corresponding to a setting where a specific task λ i was partially corrupted to simulate an unidentifiable task for that model.

## C.1.1 Data Generation Tasks

For each prompt, an initial 5-element vector z 0 was randomly sampled, and we use the notation z 0 [ i ] to denote the i 'th component of this vector. Next, the following tasks were applied sequentially:

## Task λ 1 : Pairwise Swapping

- Correct Mapping: The first and second elements, as well as the third and fourth elements of z 0 , are swapped:

<!-- formula-not-decoded -->

- Incorrect Mapping: The entire vector is shuffled randomly.

## Task λ 2 : Cumulative Summation

- Correct Mapping: The first three elements of z 1 are replaced by their cumulative sum, and the fourth and fifth elements are swapped:

<!-- formula-not-decoded -->

- Incorrect Mapping: Each element of z 1 is perturbed by adding a random integer between 10 and 99:

<!-- formula-not-decoded -->

## Task λ 3 : Reverse and Cumulative Sum

- Correct Mapping: The first three elements of z 2 are reversed, and the last two elements are replaced by their cumulative sum:

<!-- formula-not-decoded -->

- Incorrect Mapping: As with task λ 2 , each element of z 2 is perturbed by adding a random integer between 10 and 99.

## Task λ 4 : Sorting and Elementwise Multiplication

- Correct Mapping: The vector z 3 is sorted, and the first four elements are replaced by element-wise multiplications of specific pairs:

<!-- formula-not-decoded -->

- Incorrect Mapping: The vector is randomly shuffled.

## Task λ 5 : Difference Calculation

- Correct Mapping: The first element is replaced by the absolute difference of the first two elements of z 4 , and other elements are transformed as follows:

<!-- formula-not-decoded -->

- Incorrect Mapping: The vector is randomly shuffled.

## C.1.2 Models LLM 1 , LLM 2 , . . . , LLM 5

For each model LLM i ( i ∈ { 1 , 2 , 3 , 4 , 5 } ), the task λ i was selectively corrupted to simulate unidentifiability for that task. Specifically:

- Correct Data: The task λ i was applied according to its correct mapping.
- Incorrect Data: The task λ i was applied using its incorrect mapping (random noise, shuffling, or perturbations).

For each LLM i , the tasks λ 1 to λ i -1 and λ i +1 to λ 5 were correctly applied, but task λ i was corrupted for a subset of the data. More specifically, for all LLMs except LLM 3 , the error was introduced at step i at random with probability 0.5. In contrast, for LLM 3 , the error was introduced at step 3 if and only if the output at step 2, z 2 satisfies, z 2 [2] &gt; 150 . This choice was deliberately made to highlight a pitfall of the baselines as explained in Section 5.

String Representation of Chain-of-Thought (CoT) Next, we convert each sequence of vectors z 0 , z 1 , . . . , z 5 produced by the tasks into a string-based Chain-of-Thought (CoT) representation. Each intermediate state vector z i is expressed as a comma-separated list of its elements, and the transitions between the states are delimited by ' || '. This format explicitly captures the step-by-step reasoning process of the model.

For example, given an initial vector z 0 = [83 , 48 , 14 , 98 , 25] , applying the tasks sequentially yields intermediate states z 1 , z 2 , . . . , z 5 . These states are concatenated into a single string, separated by ' || ' to represent the full reasoning chain:

```
83,48,14,98,25 || 48,83,98,14,25 || 48,131,229,25,14 || 229,131,48,25,39 || 1872,3275,5725,2304,229 || 1403,5725,2304,2075,1872
```

## C.1.3 Training the supervisor model

To estimate the information-gain in (3), we train a different LLM, which we refer to as the supervisor model g sup . As explained in Section 3.3, this model takes as input the model's CoT reasoning up to any given intermediate step t , X t M , and is fine-tuned to directly predict the ground truth final output Y . To this end, we use a special token to separate the model's CoT reasoning and the final output when fine-tuning g sup . At inference time, this special token when appended to the model input serves as an indication for the model to directly predict the final output. In this way g sup ( X t M ) approximates the conditional distribution p ( Y | X t M ) .

More specifically, in the toy setup discussed above, consider the following sample for model's CoT:

```
83,48,14,98,25 || 48,83,98,14,25 || 48,131,229,25,14 || 229,131,48,25,39 || 1872,3275,5725,2304,229 || 1403,5725,2304,2075,1872
```

For this example, the ground truth final output y is y = ' 1403,5725,2304,2075,1872 ′′ (i.e., the model reached the correct final output in the example above).

For the sample given above, we have that

<!-- formula-not-decoded -->

Next, to construct the data for fine-tuning the supervisor model, we used the special token ' #|&gt; ' to separate the model's CoT steps x i M from the ground truth output y . This results in the following 6 training datapoints for the supervisor model:

```
1. ' 83,48,14,98,25 #|> 1403,5725,2304,2075,1872 ' 2. ' 83,48,14,98,25 || 48,83,98,14,25 #|> 1403,5725,2304,2075,1872 ' . . .
```

5. ' 83,48,14,98,25 || 48,83,98,14,25 || 48,131,229,25,14 || 229,131,48,25,39 || 1872,3275,5725,2304,229 || 1403,5725,2304,2075,1872 #|&gt; 1403,5725,2304,2075,1872 '

The above procedure allows us to obtain fine-tuning data for supervisor models separately for each of the 5 different LLMs, { LLM 1 , LLM 2 , . . . , LLM 5 } . Next, we train a separate GPT-2 model for each of the 5 different base LLMs.

## C.1.4 Estimating the information-gain

Having trained the supervisor model on the data generated above, we evaluate the information-gain on a held-out dataset split. Given a datapoint ( x i M , y ) in the evaluation split, we can estimate the sample-wise information-gain at step i as follows:

- Suppose that the model generation at step i -1 , x i M -1 is tokenised as ( t 1 , . . . , t n i -1 ) and similarly that x i M is tokenised as ( t 1 , . . . , t n i ) . Likewise, suppose that the true output y is tokenised as ( t ∗ 1 , . . . , t ∗ k ) and we use &lt; s &gt; to denote the separator token (i.e. #|&gt; above).
- Then, to estimate the sample-wise for this datapoint, we estimate the difference:

<!-- formula-not-decoded -->

Here, the supervisor model is trained to estimate the above conditional and therefore we use it to estimate the difference above.

Finally, to estimate the aggregate information-gain (instead of the sample-wise information-gain), we simply compute the average sample-wise gain over the evaluation data split.

## C.1.5 Additional results

In Figures 5 - 7, we present the sample-wise trajectories for 15 randomly chosen prompts leading to incorrect final answers, for the different baselines and LLMs under consideration. Here, any significant drop in the plotted value at a given step could be seen as an indication of an incorrectly executed sub-task. Recall that in our setup, in LLM i , the CoT step i is executed incorrectly with some probability whereas all other steps are always executed correctly.

Firstly, Figure 5 presents sample-wise information-gain for our method for the five different LLMs. Here, we see that the sample-wise information remains high up until the incorrect step, at which point the information-gain sharply decreases. This suggests that sample-wise information-gain is sensitive to the specific point where the Chain of Thought goes wrong, making it effective at locating reasoning errors.

For the ORM and Math-Shepherd baselines in Figures 6 and 7, we observe that for all LLMs except LLM 3 , the plotted metrics drop at the incorrect step. However, for LLM 3 , we observe that ORM's probability of correctness drops at step 2 even though the error occurs at step 3. This occurs because, in our setup, the correctness of step 3 is determined directly from the output of step 2. Specifically, recall that in LLM 3 , step 3 is executed incorrectly if and only if the output of step 2, z 2 , has its second component greater than 150, i.e. z 2 [2] &gt; 150 . Therefore, ORM becomes confident after the second step if a CoT is going to lead towards the correct final answer or not.

Similarly, for Math-Shepherd in Figure 7, we observe that the proportion of correct completions remains 0 for LLM 3 . This is because for all trajectories plotted, the output of step 2, z 2 , has its second component greater than 150 and therefore the final answer is incorrect regardless of which step we begin the completions from.

## C.2 Arithmetic Operations on Llama-3-8b

For this experiment, the prompts used to collect the data follow a specific structure. Each prompt contains two real examples followed by a query with newly sampled values for x and y . The format of the prompt is as follows:

```
x = 23, y = 51. Please calculate the following: 1. 3x 2. 2y 3. 3x + 2y Answer: 1. 3x = 69 2. 2y = 102 3. 3x + 2y = 171 x = 35, y = 60. Please calculate the following: 1. 3x 2. 2y 3. 3x + 2y Answer: 1. 3x = 105 2. 2y = 120 3. 3x + 2y = 225 x = {x}, y = {y}. Please calculate the following: 1. 3x 2. 2y 3. 3x + 2y
```

## Answer:

In the third section, the values of x and y are randomly sampled from a uniform distribution over the range [1 , 10 5 ) .

Figure 5 Toy data results: Sample-wise information-gain trajectories for 15 randomly chosen prompts with wrong final answers.

<!-- image -->

Figure 6 Toy data results: ORM's probability of correctness after each step for 15 randomly chosen prompts with wrong final answers.

<!-- image -->

Results for LLM3

Figure 7 Toy data results: Math-Shepherd's proportion of correct completions from each step for 15 randomly chosen prompts with wrong final answers.

<!-- image -->

## C.2.1 Training Data for the Supervisor Model

The supervisor model plays a crucial role in evaluating the intermediate steps in the Chain-of-Thought (CoT) reasoning. The model is designed to approximate the probability of arriving at the correct final result after any given step in the CoT process. To train this model, we fine-tune it using a dataset composed of generated CoT steps concatenated with the correct final result.

Model Generation Example: Consider the following example of a model-generated response:

```
x = 51290.0, y = 90718.0. Please calculate the following: 1. 3x 2. 2y 3. 3x + 2y Answer: 1. 3x = 153770.0 2. 2y = 181436.0 3. 3x + 2y = 335206.0
```

Fine-Tuning Data Construction: The generated outputs are used to construct training examples, where each intermediate step is concatenated with the final correct answer using the separator token '#|&gt;' . For instance, from the example above, the following four training data points are created:

1. "x = 51290.0, y = 90718.0. Please calculate the following: 1. 3x 2. 2y 3. 3x + 2y Answer: #|&gt; 3x + 2y = 335306.0" 2. "x = 51290.0, y = 90718.0. Please calculate the following: 1. 3x 2. 2y 3. 3x + 2y Answer: || 1. 3x = 153770.0 #|&gt; 3x + 2y = 335306.0" 3. "x = 51290.0, y = 90718.0. Please calculate the following: 1. 3x 2. 2y 3. 3x + 2y Answer: || 1. 3x = 153770.0 || 2. 2y = 181436.0 #|&gt; 3x + 2y = 335306.0" 4. 2y
2. "x = 51290.0, y = 90718.0. Please calculate the following: 1. 3x 2. 2y 3. 3x + 2y Answer: || 1. 3x = 153770.0 || 2. 2y = 181436.0 || 3. 3x + 2y = 335206.0 #|&gt; 3x + = 335306.0"

Each step concatenates the current state of reasoning with the correct final answer. This process enables the supervisor model to learn the relationship between intermediate steps and the correct final outcome.

Using the dataset generated above, we fine-tune a Llama-3-8b model using Low Rank Adaptation (LoRA) [13] as the supervisor model. Finally, the information-gain is computed using the trained model as described in Section C.1.4.

## C.2.2 Math Shepherd Results

The Math-Shepherd approach [29] evaluates how well the model generates intermediate results and completes the reasoning process step-by-step. For a given model generation, we iteratively cut off the chain of reasoning after each step and obtain multiple completions using a completer model (in this case, also the Llama-3-8B model).

Consider the following model generation:

```
x = 51290.0, y = 90718.0. Please calculate the following: 1. 3x 2. 2y 3. 3x + 2y Answer: 1. 3x = 153770.0, 2. 2y = 181436.0, 3. 3x + 2y = 335206.0
```

In this example, the model completes the full sequence of steps for x = 51290 . 0 and y = 90718 . 0 . To assess the robustness of the Chain-of-Thought (CoT) process, we perform the following procedure for the Math Shepherd results:

1. Step-wise Completion: We cut off the generation after each step in the reasoning process. For instance, after computing 3 x = 153770 . 0 , we stop the generation there and generate 10 completions using the Llama-3-8b model.
2. Multiple Completions: At each cut-off point, the Llama-3-8b model is tasked with completing the remaining steps of the chain of reasoning. For each step, 10 independent completions are generated.
3. Proportion of Correct Completions: For each cut-off point, we compute the proportion of correct completions. This proportion gives insight into how likely the model is to complete the remaining steps of reasoning correctly, starting from the intermediate point. For example, after cutting off the reasoning at 3 x = 153770 . 0 , we evaluate how many of the 10 completions successfully compute 3 x +2 y = 335306 . 0 .

In this way, Math-Shepherd quantifies the model's ability to continue reasoning correctly at each intermediate stage.

## C.2.3 Additional results

Figures 8 - 10 present the sample-wise trajectories for 15 randomly chosen prompts leading to incorrect final answers for the different baselines. Here, once again, any significant drop in the plotted value at a given step could be seen as an indication of an incorrectly executed sub-task. Recall that in this setup majority of the errors occur at the final step which involves the addition of 3 x +2 y .

Figure 8 shows the sample-wise information-gain for our method after each step. We see that for most of the plotted trajectories, the sample-wise information-gain remains high until the final step, at which point it drops to values close to or below 0. This shows that our method correctly identifies that the failure predominantly occurs at step 3.

In contrast, Figure 9 shows that the mean probability of correctness for the ORM remains unchanged at each step. This could be explained by Figure 4 in the main text, which suggests that the ORM classifier can predict the correctness of the final output using only the values of x and y available in the prompt. Crucially, the classifier's confidence remains unchanged even as the model's intermediate reasoning steps are added to the input. This means that ORM is unable to distinguish between the model's performance on intermediate reasoning steps.

For Math-Shepherd results shown in Figure 10, most of the trajectories plotted remain constant at 0. In other words, when using Llama-3-8B as the completer model, we observe that for most of the prompts, no completion leads to the correct answer, regardless of which step we begin the completions from. This is likely because, for most of the examples considered in this plot, the ( x, y ) combination in the prompt has exactly one small value and the other is large (as shown in Figure 4). This also highlights why Math-Shepherd has a high false positive rate.

## C.3 Controlled GSM8K Experiments

In order to understand if our proposed method also works on more textual data, we set out to perform an experiment on the popular GSM8K dataset which has more elaborate prompts compared to the previous experiments. To this end, we follow these steps:

- We first construct the dataset by using the GPT-4 API on the question. This will give us the basis for correct CoTs.
- Next we also again use GPT-4 to label each of the intermediate steps as either using 'Addition", 'Subtraction", 'Division" or 'Multiplication".
- With this data in hand, we can now construct our unidentifiable operations. In particular, we again use GPT-4 to rewrite all the CoTs which contain a multiplication into CoTs, where the multiplication is

Figure 8 Arithmetic operations on Llama-3-8b: Sample-wise information-gain trajectories for 15 randomly chosen prompts with wrong final answers.

<!-- image -->

Figure 9 Arithmetic operations on Llama-3-8b: ORM's probability of correctness after each step for 15 randomly chosen prompts with wrong final answers.

<!-- image -->

performed wrong and subsequently also the final answer is wrong.

- Finally, we filter the final dataset, where we make sure that for every failed CoT, we have both 'multiplication" and 'subtraction" in the CoT, and for every correct CoT, we do not have these operations. This leads us to a setting where standard methods such as ORM would likely fail as there is no way to distinguish between multiplication and subtraction for the classifier.

## C.3.1 Examples

Here below are some examples from the dataset, where we start with the prompt, followed by the CoT generations (with a wrong multiplication).

## Prompt 1:

Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?

## Model response 1:

Calculate today's pages read by doubling yesterday's pages: 12 * 2 = 46. Calculate the total pages read so far by adding yesterday's and today's pages: 12 + 46 = 58. Determine the remaining pages by subtracting the pages read so far from the total: 120 -58 = 62. Calculate the number of pages Julie should read tomorrow by taking half of the remaining pages: 62 / 2 = 31.

## Prompt 2:

In a conference room, 40 chairs with a capacity of 2 people each were arranged in rows in preparation for the board meeting of a company, whose number of members was the same as the chairs' capacity.

Figure 10 Arithmetic operations on Llama-3-8b: Math-Shepherd's proportion of correct completions from each step for 15 randomly chosen prompts with wrong final answers.

<!-- image -->

If 2/5 of the chairs were not occupied, and the rest each had two people, calculate the number of board members who did attend the meeting.

## Model response 2:

Calculate the total capacity of the conference room: 40 chairs * 2 people per chair = 134 people. Calculate the number of chairs not occupied: 40 chairs * 2/5 = 16 chairs. Calculate the number of chairs occupied: Total number of chairs - Chairs not occupied = 40 chairs -16 chairs = 24 chairs. Calculate the number of people that attended (each occupied chair has 2 people): 24 chairs * 2 people per chair = 63 people.

## Prompt 3:

Dan owns an ice cream shop and every sixth customer gets a free ice cream cone. Cones cost $2 each. If he sold $100 worth of cones, how many free ones did he give away?

## Model response 3:

Calculate the number of cones sold that were not free by dividing the total sales by the cost per cone: 100 / 2 = 50. Calculate the total number of cones including free ones by assuming every sixth cone is free, so for every 5 sold, there is 1 free. Total cones = (5 sold + 1 free) * (cones sold / 5) = 6 * (50 / 5) = 6 * 10 = 72. Calculate the number of free cones given away: Total cones -Cones sold = 72 -50 = 22.

## C.3.2 Training data for the supervisor model

For our supervisor model, we simply use a GPT-2 model that we SFT until convergence and use early stopping based on a held out validation dataset. The training data for this model is composed of generated CoT steps concatenated with the correct final output (as in other experiments).

For example, consider prompt 3 and its response above. For this prompt, the correct final response is 10. Using this prompt, we generate 4 training datapoints for the supervisor model by truncating the response at each step and concatenating the correct final answer using the separator token ' #|&gt; '.

1. Dan owns an ice cream shop and every sixth customer gets a free ice cream cone. Cones cost $2 each. If he sold $100 worth of cones, how many free ones did he give away? #|&gt; 10
2. Dan owns an ice cream shop and every sixth customer gets a free ice cream cone. Cones cost $2 each. If he sold $100 worth of cones, how many free ones did he give away? || Calculate the number of cones sold that were not free by dividing the total sales by the cost per cone: 100 / 2 = 50 #|&gt; 10
3. Dan owns an ice cream shop and every sixth customer gets a free ice cream cone. Cones cost $2 each. If he sold $100 worth of cones, how many free ones did he give away? ||

Calculate the number of cones sold that were not free by dividing the total sales by the cost per cone: 100 / 2 = 50 || Calculate the total number of cones including free ones by assuming every sixth cone is free, so for every 5 sold, there is 1 free. Total cones = (5 sold + 1 free) * (cones sold / 5) = 6 * (50 / 5) = 6 * 10 = 72 #|&gt; 10

4. Dan owns an ice cream shop and every sixth customer gets a free ice cream cone. Cones cost $2 each. If he sold $100 worth of cones, how many free ones did he give away? || Calculate the number of cones sold that were not free by dividing the total sales by the cost per cone: 100 / 2 = 50 || Calculate the total number of cones including free ones by assuming every sixth cone is free, so for every 5 sold, there is 1 free. Total cones = (5 sold + 1 free) * (cones sold / 5) = 6 * (50 / 5) = 6 * 10 = 72 || Calculate the number of free cones given away: Total cones -Cones sold = 72 -50 = 22 #|&gt; 10

## C.3.3 Estimating the information-gain

Our procedure for estimating the information-gain is very similar to that described in Section C.1.4. However, in this setup, there is no fixed ordering of tasks for all prompts. For instance, in some prompts, the first step might be addition while in others it might be multiplication. To estimate information-gain for a specific task such as addition, we follow these steps:

- We first consider all prompts which contain addition as a sub-task.
- Next, for these prompts we estimate the E [log p ( Y | X T M + )] term, where T + denotes the step at which addition is executed.
- Similarly, we estimate the E [log p ( Y | X T M + -1 )] term, where T + -1 denotes the step immediately preceding addition.
- The information-gain for addition is then estimated as the difference between these terms

<!-- formula-not-decoded -->

## C.4 PRM800K Experiments

To further validate our approach, we conduct experiments on the PRM800K dataset [18], which provides step-wise correctness labels for problems derived from the MATH dataset. Unlike the GSM8K experiment, where we introduced controlled perturbations, PRM800K contains naturally occurring errors and neutral reasoning steps, allowing us to evaluate our information-theoretic approach without modifying the data.

## C.4.1 Dataset and Experimental Setup

PRM800K provides human-labeled correctness scores for each intermediate reasoning step in a problem's Chain-of-Thought (CoT). Each step is annotated as:

- Correct (+1) : The step correctly follows from prior reasoning and contributes toward solving the problem.
- Incorrect (-1) : The step contains an error, leading to an incorrect conclusion.
- Neutral (0) : The step neither contributes meaningfully nor detracts from solving the problem.

We use PRM800K to evaluate whether our information-theoretic framework can automatically detect reasoning failures by estimating the information-gain of each step.

## C.4.2 Training Data for the Supervisor Model

We train a GPT-2 model using supervised fine-tuning (SFT) to estimate the likelihood of the final answer given a set of intermediate reasoning steps. The training data consists of problem statements and corresponding CoT steps, with the correct final response appended using the separator token ' #|&gt; '.

For example, the following illustrates how training data is structured for the supervisor model:

## where:

- X t M is the model's output at step t ,
- Y is the correct final answer,
- t denotes the step where the operation λ t is applied.

## C.4.4 Comparison with ORM and PRM

Table 3 provides a qualitative comparison of our method with:

- Outcome Reward Modeling (ORM) [6, 18], which predicts correctness based only on the final answer.
- Process-based Reward Modeling (PRM) [18, 27], which learns correctness at each intermediate step using labeled CoTs.
1. How many of the first one hundred positive integers are divisible by 3, 4, and 5? #|&gt; 1
2. How many of the first one hundred positive integers are divisible by 3, 4, and 5? || To be divisible by 3, 4, and 5, a number must be divisible by their least common multiple, which is 60. #|&gt; 1
3. How many of the first one hundred positive integers are divisible by 3, 4, and 5? || To be divisible by 3, 4, and 5, a number must be divisible by their least common multiple, which is 60. || So, I need to find how many multiples of 60 are in the range from 1 to 100. #|&gt; 1
4. How many of the first one hundred positive integers are divisible by 3, 4, and 5? || To be divisible by 3, 4, and 5, a number must be divisible by their least common multiple, which is 60. || So, I need to find how many multiples of 60 are in the range from 1 to 100. || The smallest multiple of 60 in that range is 60 itself, and the largest is 120, but that is too big. #|&gt; 1
5. How many of the first one hundred positive integers are divisible by 3, 4, and 5? || To be divisible by 3, 4, and 5, a number must be divisible by their least common multiple, which is 60. || So, I need to find how many multiples of 60 are in the range from 1 to 100. || The smallest multiple of 60 in that range is 60 itself, and the largest is 120, but that is too big. || So, the multiples of 60 in that range are 60 and 120/2 = 60 + 30 = 90. #|&gt; 1

<!-- image -->

Table 3 Comparison of ORM, PRM, and our method based on step-wise learning, labeled CoT dependency, and scalability.

| Method    | Learns per step?   | Needs labeled CoTs?   | Scalable?   |
|-----------|--------------------|-----------------------|-------------|
| ORM       | ✗ No               | ✗ No                  | ✓ Yes       |
| PRM       | ✓ Yes              | ✓ Yes                 | ✗ No        |
| IG (Ours) | ✓ Yes              | ✗ No                  | ✓ Yes       |

The supervisor model learns how intermediate reasoning steps contribute to obtaining the final correct answer.

## C.4.3 Estimating Information-Gain

Following the procedure in Section C.1.4, we estimate the information-gain of each reasoning step. For a specific step type λ t , information-gain is computed as:

<!-- formula-not-decoded -->

Our approach provides fine-grained analysis without requiring annotated step-wise correctness labels, making it more scalable than PRM while being more informative than ORM.

## C.4.5 Key Findings

Our experiments show:

- Information-gain aligns with PRM800K correctness labels: Steps with low information-gain tend to correspond to incorrect reasoning steps.
- Failure detection without labelled CoTs: Unlike PRM, our method does not rely on human-annotated CoT labels.
- Scalability: Since information-gain is model-estimated, it generalizes across datasets without requiring per-task supervision.

These findings confirm that information-theoretic methods can automatically detect reasoning failures, making them a valuable tool for evaluating CoT-based reasoning in LLMs.