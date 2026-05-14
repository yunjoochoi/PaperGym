## ERANK: Fusing Supervised Fine-Tuning and Reinforcement Learning for Effective and Efficient Text Reranking

Yuzheng Cai 1 , Yanzhao Zhang 2 , Dingkun Long 2 , Mingxin Li 2 , Pengjun Xie 2 , Weiguo Zheng 1 *

1 Fudan University 2 Alibaba Group

yuzhengcai21@m.fudan.edu.cn, zhengweiguo@fudan.edu.cn

<!-- image -->

[HuggingFace](https://huggingface.co/collections/Alibaba-NLP/erank-68b302be7d1f62b0b7cb19a2)

## Abstract

Text reranking models are a crucial component in modern systems like Retrieval-Augmented Generation, tasked with selecting the most relevant documents prior to generation. However, current Large Language Models (LLMs) powered rerankers often face a fundamental trade-off. On one hand, Supervised Fine-Tuning based pointwise methods that frame relevance as a binary classification task lack the necessary scoring discrimination, particularly for those built on reasoning LLMs. On the other hand, approaches designed for complex reasoning often employ powerful yet inefficient listwise formulations, rendering them impractical for low latency applications. To resolve this dilemma, we introduce ERANK, a highly Effective and Efficient pointwise reranker built from a reasoning LLM that excels across diverse relevance scenarios. We propose a novel two-stage training pipeline that begins with Supervised Fine-Tuning (SFT). In this stage, we move beyond binary labels and train the model generatively to output fine grained integer scores, which significantly enhances relevance discrimination. The model is then further refined using Reinforcement Learning (RL) with a novel, listwise derived reward. This technique instills global ranking awareness into the efficient pointwise architecture. We evaluate the ERANK reranker on the BRIGHT, FollowIR, TREC DL, and BEIR benchmarks, demonstrating superior effectiveness and robustness compared to existing approaches. On the reasoning-intensive BRIGHT benchmark, our ERANK4B achieves an nDCG@10 of 38 . 7 , while a larger 32B variant reaches a state of the art nDCG@10 of 40 . 2 .

## 1 Introduction

Text reranking is a fundamental component of various Natural Language Processing and Information Retrieval applications, utilized extensively in downstream tasks such as open-domain question answering (Lee et al. 2018), web search (Lin, Nogueira, and Yates 2022), and recommendation systems (Chuang et al. 2020; Gao et al. 2025). Large Language Models (LLMs) have significantly reshaped the text reranking landscape. On one hand, studies have sought to leverage the advanced text understanding capabilities of LLMs for reranking, either through zero-shot prompting or Supervised Fine-Tuning (Zhang et al. 2023; Liu et al. 2024). On the other hand, LLMs have introduced new application paradigms like Retrieval-Augmented Generation (Wu et al.

* Corresponding author

<!-- image -->

[ModelScope](https://modelscope.cn/collections/ERank-488bc43c873e4c)

Figure 1: Semantic relevance refers to the traditional understanding based on keyword or semantic matching, while the reasoning-intensive example aims to capture documents that may not directly answer the query but provide essential intermediate information needed for multi-step reasoning.

<!-- image -->

Figure 2: ERANK-4B achieves state-of-the-art performance among pointwise rerankers using candidate documents retrieved by BM25 with original queries. Under retrieval settings in Section 4.2, ERANK-4B and 32B further achieve the nDCG@10 of 38.7 and 40.2 on BRIGHT, respectively.

<!-- image -->

2024; Gupta, Ranjan, and Singh 2024; Wang et al. 2024) and agentic systems (Huang et al. 2024; Li et al. 2024). These paradigms demand capabilities beyond traditional semantic relevance, requiring models to perform reasoning-intensive retrieval, such as identifying issue-relevant code snippets to resolve a specific programming problem. Recent advancements in test-time compute (OpenAI 2024; DeepSeek AI 2025) have shown promise in such scenarios, with a growing number of text rerankers based on reasoning LLMs (Weller et al. 2025b; Zhuang et al. 2025; Zhang et al. 2025).

Prior approaches have generally treated traditional semantic relevance and reasoning-intensive reranking as distinct challenges, which are illustrated in Figure 1. For semantic tasks, Supervised Fine-Tuning (SFT) is a common strategy (Ma et al. 2024; Sun et al. 2023; Zhang et al. 2024). However, most SFT-based rerankers adopt a pointwise scoring method based on binary classification, where the model predicts labels like 'Relevant' or 'Not Relevant'. We argue this approach is suboptimal as it leads to poor score discrimination, a problem exacerbated in modern reasoning LLMs that generate overconfident predictions after Chainof-Thought (CoT). For reasoning-intensive tasks, Reinforcement Learning (RL) has shown promise (Zhuang et al. 2025; Zhang et al. 2025). However, these methods often rely on listwise or setwise formulations and ingest multiple candidate documents simultaneously. With sliding windows, they process different batches of documents sequentially, resulting in prohibitive latency and memory footprints that make them impractical for real-world deployment.

This work addresses a central question: can a single, efficient reranker powered by reasoning LLM be trained to excel at both semantic relevance and deep reasoning? We contend that this is achievable by enhancing the pointwise architecture, which scores each document independently. We introduce a novel, two-stage training framework illustrated in Figure 4, which seamlessly integrates Supervised Fine-Tuning (SFT) with Reinforcement Learning (RL) for LLM-based reranker training. The first stage, SFT, trains a base model on a diverse mixture of semantic and reasoningoriented data. Crucially, we abandon the standard binary classification paradigm and instead train the model using a fine-grained integer scoring scheme, which fully utilizes the generative power of LLMs and significantly improves score discrimination. We also employ a data synthesis strategy to generate high-quality reasoning chains and fine-grained scores to overcome data scarcity. In the second stage, we further refine the SFT-tuned model using RL. To bridge the gap between listwise optimality and pointwise efficiency, we introduce a novel, listwise-derived reward function. This function provides a global ranking signal during training, encouraging the model to learn the relative importance of documents. This allows our pointwise model to benefit from listwise-style optimization while retaining its low latency.

Extensive experiments on semantic (TREC DL (Craswell et al. 2020, 2021), BEIR (Thakur et al. 2021)) and reasoning-intensive benchmarks (BRIGHT (Su et al. 2024), FollowIR (Weller et al. 2025a)) confirm that our framework delivers substantial gains. As shown in Figure 2, our 4B-parameter model outperforms many 7B model size rerankers, and our 32B model sets a new state-of-the-art on the BRIGHT benchmark. Latency measurements confirm our models maintain the high efficiency of standard pointwise rerankers, making them both powerful and practical.

Briefly, our main contributions are to:

- Reveal the suboptimality of binary classification for LLM rerankers and propose a generative approach that outputs discrete integer scores to enhance score discrimination.

Output: Final ranking of N documents w 3 &gt; &gt; … w+2

<!-- image -->

Figure 3: Comparison of different reranking paradigms.

- Introduce a novel two-stage framework integrating Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) to build a single, efficient pointwise reranker for both semantic and reasoning-intensive tasks.
- Our model, ERANK, sets a new state-of-the-art on the reasoning-based BRIGHT benchmark while demonstrating exceptional performance on standard semantic tasks.
- We will open-source our trained models and data to facilitate reproducibility and future research.

## 2 Related Work

LLM for Text Reranking. Large Language Models (LLMs) have significantly advanced text reranking beyond the capabilities of earlier encoder-only models such as BERT (Liu et al. 2024). LLMs are typically applied to this task using either zero-shot prompting (Zhang et al. 2023; Zhuang et al. 2024; Niu et al. 2024) or, more effectively, Supervised Fine-Tuning (Ma et al. 2024; Zhang et al. 2024). As shown in Figure 3, reranking methodologies are broadly categorized into pointwise (Liang et al. 2022), pairwise (Qin et al. 2024), and listwise approaches (Ma et al. 2023; Sun et al. 2023). Listwise methods, which evaluate a list of candidate documents, generally yield the highest ranking quality by directly optimizing the document order (Gao, Dai, and Callan 2021; Zhang et al. 2022; Liu et al. 2025). However, their computational cost scales quadratically with input length, making them impractical for real-world systems that demand low latency. In contrast, pointwise methods score each query-document pair independently. This paradigm enables massive parallelization and efficient inference, establishing it as the preferred choice for large scale deployment. Most fine-tuned pointwise rerankers conventionally treat the task as a binary classification problem. We argue this approach fails to leverage the full generative power of modern LLMs and results in suboptimal performance.

Reinforcement Learning for Reranking. The success of Reinforcement Learning (RL) in enhancing the complex reasoning abilities of LLMs, exemplified by models like OpenAI-O1 (OpenAI 2024) and DeepSeek-R1 (DeepSeek AI 2025), has inspired its application to reranking. Recent work demonstrates that RL can refine a model's capacity to identify documents that are not merely semantically relevant but also instrumentally useful for resolving a user's

𝑠

(')

"

𝑠

(')

%

…

𝑠

(')

&amp;

6

4

Figure 4: Overview of the two-stage fine-tuning pipeline for the pointwise ERANK reranker. Given a query and N = 3 documents A, B and C, where A is the only positively related one, the SFT model is trained to deliver a relevance score ranging from 0 to 10 for each document. During RL training, the model generates G = 2 rollouts for each document. These N × G = 6 scores extracted from all rollouts across all documents are then sorted together to compute listwise ranking derived rewards.

<!-- image -->

query. However, these pioneering RL-based reranking methods predominantly adopt listwise or setwise training frameworks (Zhuang et al. 2025; Zhang et al. 2025). While effective, they inherit the high latency and memory requirements associated with processing multiple batches of documents sequentially, which limits their practical applicability. Our work addresses this critical gap. We introduce a novel twostage training pipeline that begins with a generative Supervised Fine-Tuning stage using fine-grained scoring to improve relevance discrimination. Subsequently, we employ RL to optimize our efficient pointwise model with a globally aware listwise reward signal. This strategy achieves the ranking quality of listwise methods while preserving the inference efficiency of a pointwise architecture.

## 3 Method

Our training methodology unfolds in a two-stage pipeline designed to build a reranker that excels at both semantic relevance and reasoning-intensive relevance. The first stage uses Supervised Fine-Tuning (SFT) to establish a strong foundation, and the second stage employs Reinforcement Learning (RL) with the Group Relative Policy Optimization (GRPO) algorithm (Shao et al. 2024) to refine the reranking ability.

## 3.1 Task Formulation

We formulate the text reranking task as a generative process. With a specific instruction I that defines the relevance criteria, given a query q and a set of N candidate documents { d 1 , d 2 , . . . , d N } , our model processes each querydocument pair independently. For each pair, it generates a response that includes a Chain-of-Thought (CoT) c i explaining its reasoning, followed by a final relevance score s i . This is represented by the conditional probability of the policy LLM π :

<!-- formula-not-decoded -->

Based on the extracted scores { s 1 , s 2 , . . . , s N } , the documents are then sorted in descending order to produce the final ranked list. This pointwise formulation ensures low inference latency and provides interpretability through the generated reasoning Chain-of-Thought (CoT).

## 3.2 Supervised Fine-Tuning with Fine-Grained Scores

To perform Supervised Fine-Tuning (SFT), we construct a dataset D = { ( x k , y k ) } K k =1 , where each input x k is a prompt containing the relevance instruction I , a query q , and a document d . The target output y k is a combination of reasoning chain c and relevance score s , as formatted in Figure 5.

Generative Fine-Grained Scoring. A central limitation of prior pointwise rerankers is their reliance on a binary classification objective, where the model is trained to predict labels of 'yes' and 'no' to represent relevance. To better leverage the generative nature of Large Language Models (LLMs), some recent methods have exploited their autoregressive capabilities. These approaches compute a normalized relevance score by extracting token probabilities for 'yes' and 'no' , which can be definded as:

Pr(token= yes )

Pr(token= yes ) + Pr(token= no )

.

Our experiments reveal that such strategy leads to poor score discrimination. The issue becomes particularly pronounced when using reasoning LLMs, since the confidenceboosting effect of Chain-of-Thought (CoT) reasoning causes these models to generate overconfident predictions. As illustrated in Figure 6 and detailed in Appendix B, a comparison between a non-reasoning model (Qwen3-32B (Qwen Team 2025a)) and a reasoning-enhanced model (QwQ-32B (Qwen Team 2025b)) shows that the latter produces normalized scores heavily concentrated near 0 or 1. A significantly higher proportion of scores from the reasoning model falls within the extreme intervals of [0 , 0 . 00001] and [0 . 99999 , 1] . This concentration severely diminishes the model's ability to distinguish between varying degrees of relevance, which is essential for effective reranking.

To overcome this limitation, we reframe reranking as a generative task with a fine-grained scoring system. Instead Given a query and a document, please give a relevance score of 0 to 10. The goal or relevance definition is: { instruction }

Here is the query: { query } Here is the document: { document } After thinking, directly choose a relevance score from [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]. - 0 represents completely not related. - 10 means perfectly related. Desired output format: &lt;think&gt; put your thinking here &lt;/think&gt;&lt;answer&gt; Only allows an integer here &lt;/answer&gt; Your output:

Figure 5: Prompt for scoring with integers from 0 to 10.

of predicting binary labels, we train the model to generate an integer score from 0 to 10 that reflects the degree of relevance with prompt in Figure 5. Then, the final ranking score is computed as s i × Pr(token= s i ) . This method fully utilizes the autoregressive capabilities of the LLM, creating a more expressive and discriminative scoring space critical for distinguishing between documents of varying quality. Table 1 shows that it consistently improves nDCG@10 across multiple benchmarks under experimental settings in Appendix B.

Data Synthesis for SFT. To train our model for this finegrained scoring task, we synthesize a high-quality dataset that covers both semantic matching and complex reasoning scenarios. We employ a powerful open-source model, QwQ-32B (Qwen Team 2025b), as a teacher to generate reasoning chains and integer scores. To ensure the quality and reliability of the synthetic labels, our data construction process emphasizes two key aspects. Query-Document Diversity : We source query-document pairs from a diverse mix of datasets, including MS MARCO (Bajaj et al. 2016) for semantic relevance, and ReasonIR (Shao et al. 2025) and Promptriever (Weller et al. 2024) for complex reasoning tasks. For semantic relevance, we randomly select 5 , 000 queries from the MS MARCO dataset. For reasoningintensive tasks, we sample 10 , 000 queries from the hard query (HQ) set of ReasonIR and 5 , 000 queries from the Promptriever training set. Both of these sources contain complex queries that require deep reasoning. For each query, we enrich the initial candidate pool, which includes the annotated positive documents and synthetic negative documents, by retrieving the top 1 , 000 documents from the corpus using the ReasonIR-8B retriever. We then sample documents from different ranking ranges to create a balanced set of negatives: the top 10 documents serve as hard negatives, positions 11-100 as medium negatives, and positions 101-1 , 000 as easy negatives. Further details are provided in Appendix C. Each query is ultimately associated with exactly 20 documents to maintain a consistent input structure.

Figure 6: Distributions of normalized probability with nonreasoning and reasoning LLMs on BRIGHT benchmark.

<!-- image -->

Table 1: Average nDCG@10 of QwQ-32B reasoning LLM when varying scoring discrimination on three benchmarks.

| Scoring                |   BRIGHT |   TREC DL |   BEIR (5 subsets) |
|------------------------|----------|-----------|--------------------|
| yes / no               |     20.8 |      60.9 |               31.1 |
| { 0 , 1 , · · · , 3 }  |     22.7 |      64.1 |               36.5 |
| { 0 , 1 , · · · , 10 } |     23.2 |      66.1 |               37.1 |

High-Quality and Stable Reasoning Trajectory Generation : We use the QwQ-32B teacher model to generate a reasoning chain c and a corresponding score s for each querydocument pair. To improve the reliability of these generated labels, we perform multiple independent generations for each instance and compute the average score, which serves as a consensus score. We then select the single generation whose score is closest to this consensus. Experiments on a random sample of 512 queries show that this strategy significantly improves scoring quality. The average nDCG@10 increased from 63 . 5% with a single generation to 65 . 6% with 3 -sample consensus and further to 67 . 4% with 10 -sample consensus. To balance performance and computational cost, we use three generations per instance in our final data synthesis process. Finally, we filter out any instances where the generated output exceeds 2 , 048 tokens. This procedure results in our final Supervised Fine-Tuning (SFT) dataset, D .

Such a high-quality dataset D enables the model to learn nuanced relevance assessment. The model is then trained on this dataset using a standard language modeling objective:

<!-- formula-not-decoded -->

## 3.3 Reinforcement Learning with GRPO

While Supervised Fine-Tuning (SFT) provides a strong reranking model, we employ Reinforcement Learning (RL) to further refine its ability to discern subtle ranking differences and optimize for list level metrics. To achieve this, we adopt the Group Relative Policy Optimization (GRPO) algorithm (Shao et al. 2024), inspired by prior work demonstrating that RL on small, high quality datasets can yield significant performance gains (DeepSeek AI 2025). We initialize both the GRPO policy π θ and the reference model π ref with the SFT-tuned model to ensure training stability and preserve its well generalized capabilities.

The training process begins by sampling a group of G output trajectories { y 1 , y 2 , . . . , y G } for each input prompt with the old policy π old. The policy π θ is then updated by optimizing the GRPO objective. This objective is built around a clipped importance sampling estimator, which evaluates the advantage of each trajectory relative to others in the group. To prevent the policy from deviating too drastically from the robust SFT model, we incorporate a Kullback Leibler (KL) divergence penalty. This term regularizes the policy updates, ensuring the model learns a more nuanced scoring function without sacrificing its foundational knowledge. The complete objective function is formulated as follows:

Figure 7: Example for rule-based listwise reward r RR when there are G = 2 rollouts and N = 3 documents for query q .

<!-- image -->

<!-- formula-not-decoded -->

where the clipped estimator C and KL penalty D KL are:

<!-- formula-not-decoded -->

where the advantage ˆ A i,t is computed by normalizing the reward r ( y i ) of trajectory y i using the mean and standard deviation of rewards in the group: ˆ A i,t = r ( y i ) -mean ( { r ( y j ) } G j =1 ) std ( { r ( y j ) } G j =1 ) . The hyperparameters β and ϵ control the strength of the KL penalty and the clipping range, respectively.

Listwise Reranking Reward Design. Previous studies have established that listwise rerankers often outperform pointwise models because they directly optimize the relative ordering of documents. Drawing inspiration from this, we designed a novel rule-based listwise reward function, r RR. While our model produces scores pointwise at inference, this reward design allows it to learn from relative document ordering during training, a key strength of listwise methods.

Table 2: Statistics of training data.

| Stage   |   # Queries |   # Doc per query |   # Query-doc pairs |
|---------|-------------|-------------------|---------------------|
| SFT     |      14,799 |                20 |             295,980 |
| RL      |       2,048 |                20 |              40,960 |

As shown in Figure 7, for a given query, we first generate scores for all N candidate documents and their corresponding G rollouts. These N × G scores are then aggregated and sorted to determine the global rank of each generated output. Our reward function, r RR, operates based on the following principles. Positive documents receive a high reward based on their reciprocal rank, incentivizing them to be placed as high as possible. Negative documents that are incorrectly ranked higher than any positive document receive a substantial penalty. Negative documents ranked correctly below all positive documents receive a smooth reward based on the squared error against a reference score from the SFT model, which helps maintain a stable scoring distribution. A significant penalty is assigned if the model's output is not correctly formatted, which discourages generation errors.

The formal definition of the reward r RR is as follows, where ϕ ( j ) i is the global rank of the j -th rollout for document d i with ties resolved by assigning the minimum rank. Let D P and D N be the set of positive and negative documents, respectively. Let Φ ( min ) D P and Φ ( max ) D P denote the minimum and maximum ranks for positive documents, respectively.

<!-- formula-not-decoded -->

where 'formatted' condition requires that the model's output conforms to the expected structure so a score can be extracted. In the third case, s i is the generated score and t i is a reference score from the SFT model π ref .

We trained the model using a randomly sampled subset of 2 , 048 queries from the SFT dataset, each paired with 20 documents. The GRPO algorithm was applied directly using these pointwise prompts without any pre-collected trajectories or external reward signals. The SFT tuned model served a dual role as both the initial policy for training and the reference model π ref for calculating KL divergence and providing reference scores for the reward function.

## 4 Experiment

## 4.1 Evaluation Setup

Benchmarks. Weevaluate our method across a diverse set of reranking tasks on four benchmark suites. For in-domain semantic matching, we use the TREC DL19 and DL20 passage ranking collections (Craswell et al. 2020, 2021). For out-of-domain generalization, we utilize the entire BEIR benchmark (Thakur et al. 2021) and report results on a fivedataset subset, which we term BEIR-5 (ArguAna, DBPedia, FiQA, NFCorpus, and SCIDOCS), to facilitate efficient ablation studies. To assess complex reasoning, we employ the BRIGHT benchmark (Su et al. 2024) for general reasoning and the FollowIR benchmark (Weller et al. 2025a) for instruction following abilities.

Table 3: Evaluation on different relevance types using original queries without hybrid scores. Best results are indicated in bold, while second-best results are underlined.

| Method                |                                 |   Average | Reasoning-intensive Relevance   | Reasoning-intensive Relevance   | Semantic Relevance   | Semantic Relevance   |
|-----------------------|---------------------------------|-----------|---------------------------------|---------------------------------|----------------------|----------------------|
|                       |                                 |           | BRIGHT                          | FollowIR                        | BEIR                 | TREC DL              |
| First-stage retriever | First-stage retriever           |      25.9 | 13.7                            | 0                               | 40.8                 | 49.3                 |
| Listwise              | Rank-R1-7B (Zhuang et al. 2025) |      34.6 | 15.7                            | 3.6                             | 49.0                 | 70.0                 |
| Listwise              | Rearank-7B (Zhang et al. 2025)  |      35.3 | 17.4                            | 2.3                             | 49.0                 | 72.5                 |
| Pointwise             | JudgeRank-8B (Niu et al. 2024)  |      32.1 | 17.0                            | 9.9                             | 39.1                 | 62.6                 |
| Pointwise             | Rank1-7B (Weller et al. 2025b)  |      34.6 | 18.2                            | 9.1                             | 44.2                 | 67.1                 |
| Pointwise             | QwQ-32B powered reranker (Ours) |      37.7 | 23.2                            | 14.1                            | 47.5                 | 66.1                 |
| Pointwise             | ERANK-4B (Ours)                 |      36.8 | 22.7                            | 11.0                            | 44.8                 | 68.9                 |
| Pointwise             | ERANK-14B (Ours)                |      36.9 | 23.1                            | 10.3                            | 47.1                 | 67.1                 |
| Pointwise             | ERANK-32B (Ours)                |      38.1 | 24.4                            | 12.1                            | 47.7                 | 68.1                 |

Baselines. We compare our model, ERANK, against leading reasoning-based rerankers. These include JudgeRank8B, a zero-shot reranker that uses a multi-step reasoning process; Rank1-7B, a pointwise reranker fine-tuned via distillation; Rank-R1-7B, a listwise reranker trained with the GRPO algorithm; and Rearank-7B, a state-of-the-art listwise model trained to predict optimal document permutations. Additionally, we include two top-performing listwise rerankers from the online BRIGHT benchmark (Su et al. 2025): Rank-R1-32B-v0.2 (ielabgroup 2025) trained on the ReasonIR training set, and the zero-shot XRR-Gemini-2.5Flash (jataware 2025) which performs a two-pass reranking process.

## 4.2 Implementation and Evaluation Details

For TREC DL, BEIR, and BRIGHT benchmarks, we rerank the top 100 candidates retrieved by BM25 with Pyserini (Lin et al. 2021). For FollowIR, we rerank the 1,000 candidates provided by the benchmark. Evaluation is performed using nDCG@10 for the first three benchmarks and preferencebased Mean Reciprocal Rank ( p -MRR) for FollowIR. In all cases, higher scores indicate better performance.

Following prior studies (Weller et al. 2025b; Niu et al. 2024; ielabgroup 2025), we adopt similar settings for a thorough evaluation on the challenging BRIGHT benchmark. First, to improve first-stage retrieval precision, we use reasoning queries expanded by GPT-4. Documents are then retrieved using either BM25 or the ReasonIR-8B model (Shao et al. 2025). During the reranking phase, these expanded queries are not provided to the reranker. Second, we employ a hybrid strategy to combine BM25 scores and reranking scores for low-cost model ensembling. While JudgeRank uses a simple weighted sum and Rank-R1-32B-v0.2 adopts min-max normalization, our ERANK applies standardization before score aggregation, as detailed in Appendix G.

Using Qwen3 LLM series (Qwen Team 2025a) as the backbone, our ERANK model is trained in two-stages. The first stage consists of one epoch of Supervised Fine-Tuning (SFT) with Low-Rank Adaptation (LoRA) (Hu et al. 2022). The second stage uses the GRPO algorithm (Shao et al. 2024) for Reinforcement Learning (RL), performing fullparameter fine-tuning for 10 epochs with a group G = 5 . Detailed hyperparameters can be found in Appendices E and F. As shown in Table 2, after filtering out those longer than 2 , 048 response tokens, there are 14 , 799 queries and 2 , 048 queries for SFT and RL training, respectively. All experiments are conducted on four NVIDIA A100 (80GB) GPUs. We use official checkpoints for all baselines and reproduce JudgeRank based on its published methodology.

## 4.3 Main Results

Table 3 presents the main results, with detailed reports available in Appendix H. On average, ERANK-4B clearly outperforms all pointwise baselines with 7B or 8B parameters. Furthermore, ERANK-4B significantly surpasses listwise rerankers, which are typically more powerful, on reasoning-intensive tasks despite having fewer parameters. This demonstrates that ERANK-4B achieves superior effectiveness while maintaining lower latency compared to listwise methods. Beyond the 4B model, we extend our twostage training pipeline to Qwen3-14B and Qwen3-32B models using identical training data. The results show an overall performance improvement with increased model size, indicating a clear scaling trend. At the 32B scale, our trained ERANK-32B reranker outperforms its teacher model, QwQ32B, which confirms the efficacy of our training procedure.

Table 4 further reports results with advanced retrieval and BM25hybrid strategy on the BRIGHT benchmark. Our ERANK rerankers consistently achieve state-of-the-art performance compared to baselines of similar model size, showing superior robustness and effectiveness. Despite using a pointwise paradigm, ERANK-4B achieves a notable nDCG@10 of 38.7 with the BM25 hybrid on documents retrieved by ReasonIR-8B. The ERANK-32B model with BM25 hybrid achieves an nDCG@10 of 40.2, outperforming the state-ofthe-art Rank-R1-32B-v0.2 listwise reranker. Moreover, ERANK-32B approaches the performance of XRR2, a listwise method that employs the Gemini-2.5-Flash model.

| Method                                    | nDCG@10                                   |
|-------------------------------------------|-------------------------------------------|
| Retrieve by BM25 using GPT-4 reason-query | Retrieve by BM25 using GPT-4 reason-query |
| BM25                                      | 27.0                                      |
| Rank-R1-7B (Zhuang et al. 2025)           | 23.9                                      |
| Rank1-7B (Weller et al. 2025b)            | 25.5                                      |
| Rearank-7B (Zhang et al. 2025)            | 29.1                                      |
| XRR2-Gemini-2.5-Flash (jataware 2025)     | 40.3 *                                    |
| JudgeRank-8B (Niu et al. 2024)            | 24.4                                      |
| + BM25 hybrid                             | 31.0                                      |
| ERANK-4B (Ours)                           | 32.9                                      |
| + BM25 hybrid                             | 36.1                                      |
| ERANK-14B (Ours)                          | 33.5                                      |
| + BM25 hybrid                             | 36.7                                      |
| ERANK-32B (Ours)                          | 34.6                                      |
| + BM25 hybrid                             | 37.4                                      |

## Retrieve by ReasonIR-8B using GPT-4 reason-query

Table 4: Further evaluation on BRIGHT benchmark. *Taken from BRIGHT online website (Su et al. 2025).

| ReasonIR-8B (Shao et al. 2025)     | 30.5   |
|------------------------------------|--------|
| Rank-R1-7B (Zhuang et al. 2025)    | 24.1   |
| Rank1-7B (Weller et al. 2025b)     | 24.3   |
| Rearank-7B (Zhang et al. 2025)     | 27.5   |
| JudgeRank-8B (Niu et al. 2024)     | 20.2   |
| + BM25 hybrid                      | 22.7   |
| Rank-R1-32B-v0.2 (ielabgroup 2025) | 37.7*  |
| + BM25 hybrid                      | 40.0*  |
| ERANK-4B (Ours)                    | 30.5   |
| + BM25 hybrid                      | 38.7   |
| ERANK-14B (Ours)                   | 31.8   |
| + BM25 hybrid                      | 39.3   |
| ERANK-32B (Ours)                   | 32.8   |
| + BM25 hybrid                      | 40.2   |

## 4.4 Analysis

Impact of training stages. To investigate the contribution of SFT and RL to reranking ability, we perform an ablation study using consistent prompts across different model variants. As shown in Table 5, the instructed Qwen3-4B LLM without any fine-tuning performs poorly. Both SFT and RL independently yield significant improvements, highlighting their individual effectiveness. Our two-stage training pipeline for ERANK-4B yields the most robust effectiveness overall.

Varying rewards in RL training. Besides the rule-based listwise reward r RR using Reciprocal Rank, we evaluate two different rule-based rewards, which are briefly described as follows. Please refer to Appendix I for more details.

- Pointwise reward r SE. It uses squared error to measure the difference between the score s i from the policy model and the score t i from the teacher model (i.e., QwQ-32B).
- Listwise reward r nDCG. This is a listwise reward similar to r RR, which assesses how effectively positive documents contribute to the nDCG metric while penalizing negative documents ranked above any positive document.

Table 6 compares the results when utilizing different rewards for GRPO training. Overall, listwise rewards such as r nDCG and r RR lead to better outcomes than the pointwise reward r SE. Pointwise reward that mimicks the teacher model's scores for each document independently may not align well with global ranking objectives. In contrast, listwise rewards tend to yield more favorable results by considering relative ranks to encourage a better final reranking order. While r nDCG shows a notable improvement on the FollowIR benchmark, r RR demonstrates greater robustness and superior overall performance across the four benchmarks.

|          |   Avg. |   BRIGHT |   FollowIR |   BEIR-5 |   TREC DL |
|----------|--------|----------|------------|----------|-----------|
| Qwen3-4B |   12.7 |      3.6 |        1.9 |      6.4 |      39.0 |
| SFT Only |   32.8 |     22.0 |       11.2 |     30.0 |      68.1 |
| RL Only  |   31.8 |     20.1 |       12.2 |     28.2 |      66.5 |
| SFT+RL   |   33.8 |     22.7 |       11.0 |     32.4 |      68.9 |

Table 5: Performance of different training stages.

Table 6: Performance of different rewards for RL training.

|           |   Avg. |   BRIGHT |   FollowIR |   BEIR-5 |   TREC DL |
|-----------|--------|----------|------------|----------|-----------|
| Before RL |   32.8 |     22.0 |       11.2 |     30.0 |      68.1 |
| r SE      |   31.1 |     22.3 |        8.7 |     30.8 |      62.6 |
| r nDCG    |   33.8 |     21.5 |       13.2 |     32.8 |      67.6 |
| r RR      |   33.8 |     22.7 |       11.0 |     32.4 |      68.9 |

Figure 8: Latency for returning the complete reranked list per query, averaged on all queries of TREC DL19 dataset.

<!-- image -->

Reranking latency. In real-world applications, achieving superior performance across diverse relevance types must be balanced with acceptable latency. As shown in Figure 8, which measures latency per query on the TREC DL19 dataset, pointwise methods offer significantly lower latency than their listwise counterparts. This advantage stems from the ability of pointwise methods to process documents in parallel, whereas listwise methods require sequential processing as discussed in Section 2. Specifically, the ERANK4B reranker is six times faster than both the listwise methods and the QwQ-32B pointwise reranker, highlighting its practicality for real-world applications. Compared to Rank17B, ERANK-4B achieves superior performance by generating more tokens while maintaining comparable latency.

## 5 Conclusion

In this paper, we introduce ERANK, an LLM-based reranker designed for effective and efficient reranking of documents in both semantic and reasoning-intensive tasks. To support real-world applications, ERANK adopts the pointwise paradigm to ensure low latency while achieving competitive performance through a two-stage training pipeline. The first stage conducts Supervised Fine-Tuning (SFT) to build foundational reasoning capabilities, and the second stage employs the GRPO algorithm with a novel rule-based listwise reward tailored for pointwise rerankers. Extensive evaluation on four benchmarks demonstrates the effectiveness and robustness of ERANK compared to state-of-the-art methods.

## References

Bajaj, P.; Campos, D.; Craswell, N.; Deng, L.; Gao, J.; Liu, X.; Majumder, R.; McNamara, A.; Mitra, B.; Nguyen, T.; et al. 2016. MS MARCO: A human generated machine reading comprehension dataset. arXiv preprint arXiv:1611.09268 .

Chuang, Y.-N.; Chen, C.-M.; Wang, C.-J.; Tsai, M.-F.; Fang, Y.; and Lim, E.-P. 2020. TPR: Text-aware preference ranking for recommender systems. In Proceedings of the 29th ACM International Conference on Information &amp; Knowledge Management , 215-224.

Craswell, N.; Mitra, B.; Yilmaz, E.; and Campos, D. 2021. Overview of the TREC 2020 deep learning track. arXiv:2102.07662.

Craswell, N.; Mitra, B.; Yilmaz, E.; Campos, D.; and Voorhees, E. M. 2020. Overview of the TREC 2019 deep learning track. arXiv:2003.07820.

DeepSeek AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.

Gao, J.; Chen, B.; Zhao, X.; Liu, W.; Li, X.; Wang, Y.; Wang, W.; Guo, H.; and Tang, R. 2025. Llm4rerank: Llm-based auto-reranking framework for recommendations. In Proceedings of the ACM on Web Conference 2025 , 228-239.

Gao, L.; Dai, Z.; and Callan, J. 2021. Rethink training of BERT rerankers in multi-stage retrieval pipeline. In European Conference on Information Retrieval , 280-286. Springer.

Gupta, S.; Ranjan, R.; and Singh, S. N. 2024. A comprehensive survey of retrieval-augmented generation (RAG): Evolution, current landscape and future directions. arXiv preprint arXiv:2410.12837 .

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y .; Wang, S.; Wang, L.; Chen, W.; et al. 2022. Lora: Low-rank adaptation of large language models. ICLR , 1(2): 3.

Huang, X.; Liu, W.; Chen, X.; Wang, X.; Wang, H.; Lian, D.; Wang, Y.; Tang, R.; and Chen, E. 2024. Understanding the planning of LLM agents: A survey. arXiv preprint arXiv:2402.02716 .

ielabgroup. 2025. Rank-R1-32B-v0.2. https://huggingface. co/ielabgroup/Rank-R1-32B-v0.2. Accessed: 2025-07-24.

jataware. 2025. XRR2: Expand → Retrieve → Rerank → Rerank - simple method with strong results on BRIGHT benchmark. https://github.com/jataware/XRR2. Accessed: 2025-07-24.

Lee, J.; Yun, S.; Kim, H.; Ko, M.; and Kang, J. 2018. Ranking Paragraphs for Improving Answer Recall in OpenDomain Question Answering. In Proceedings of the 2018

Conference on Empirical Methods in Natural Language Processing , 565-569.

Li, X.; Wang, S.; Zeng, S.; Wu, Y.; and Yang, Y. 2024. A survey on LLM-based multi-agent systems: workflow, infrastructure, and challenges. Vicinagearth , 1(1): 9.

Liang, P.; Bommasani, R.; Lee, T.; Tsipras, D.; Soylu, D.; Yasunaga, M.; Zhang, Y.; Narayanan, D.; Wu, Y.; Kumar, A.; et al. 2022. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110 .

Lin, J.; Ma, X.; Lin, S.-C.; Yang, J.-H.; Pradeep, R.; and Nogueira, R. 2021. Pyserini: A Python toolkit for reproducible information retrieval research with sparse and dense representations. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval , 2356-2362.

Lin, J.; Nogueira, R.; and Yates, A. 2022. Pretrained transformers for text ranking: Bert and beyond . Springer Nature.

Liu, J.; Ma, Y.; Zhao, R.; Zheng, J.; Ma, Q.; and Kang, Y. 2025. ListConRanker: A Contrastive Text Reranker with Listwise Encoding. arXiv preprint arXiv:2501.07111 .

Liu, Z.; Zhou, Y.; Zhu, Y.; Lian, J.; Li, C.; Dou, Z.; Lian, D.; and Nie, J.-Y. 2024. Information retrieval meets large language models. In Companion Proceedings of the ACM Web Conference 2024 , 1586-1589.

Ma, X.; Wang, L.; Yang, N.; Wei, F.; and Lin, J. 2024. Finetuning llama for multi-stage text retrieval. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval , 2421-2425.

Ma, X.; Zhang, X.; Pradeep, R.; and Lin, J. 2023. Zero-shot listwise document reranking with a large language model. arXiv preprint arXiv:2305.02156 .

Niu, T.; Joty, S.; Liu, Y.; Xiong, C.; Zhou, Y.; and Yavuz, S. 2024. JudgeRank: Leveraging Large Language Models for Reasoning-Intensive Reranking. arXiv preprint arXiv:2411.00142 .

OpenAI. 2024. OpenAI o1 System Card. arXiv:2412.16720. Qin, Z.; Jagerman, R.; Hui, K.; Zhuang, H.; Wu, J.; Yan, L.; Shen, J.; Liu, T.; Liu, J.; Metzler, D.; et al. 2024. Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting. In Findings of the Association for Computational Linguistics: NAACL 2024 , 1504-1518.

Qwen Team. 2025a. Qwen3 Technical Report. arXiv:2505.09388.

Qwen Team. 2025b. QwQ-32B: Embracing the Power of Reinforcement Learning.

Shao, R.; Qiao, R.; Kishore, V.; Muennighoff, N.; Lin, X. V.; Rus, D.; Low, B. K. H.; Min, S.; Yih, W.-t.; Koh, P. W.; et al. 2025. ReasonIR: Training Retrievers for Reasoning Tasks. arXiv preprint arXiv:2504.20595 .

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y.; Wu, Y.; et al. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 .

Sheng, G.; Zhang, C.; Ye, Z.; Wu, X.; Zhang, W.; Zhang, R.; Peng, Y.; Lin, H.; and Wu, C. 2024. HybridFlow: A Flexible and Efficient RLHF Framework. arXiv preprint arXiv: 2409.19256 .

Su, H.; Yen, H.; Xia, M.; Shi, W.; Muennighoff, N.; Wang, H.-y.; Liu, H.; Shi, Q.; Siegel, Z. S.; Tang, M.; Sun, R.; Yoon, J.; Arik, S. O.; Chen, D.; and Yu, T. 2024. BRIGHT: A Realistic and Challenging Benchmark for ReasoningIntensive Retrieval.

Su, H.; Yen, H.; Xia, M.; Shi, W.; Muennighoff, N.; Wang, H.-y.; Liu, H.; Shi, Q.; Siegel, Z. S.; Tang, M.; Sun, R.; Yoon, J.; Arik, S. O.; Chen, D.; and Yu, T. 2025. BRIGHT Benchmark Online Website. https:// brightbenchmark.github.io/. Accessed: August 26, 2025.

Sun, W.; Yan, L.; Ma, X.; Wang, S.; Ren, P.; Chen, Z.; Yin, D.; and Ren, Z. 2023. Is ChatGPT good at search? investigating large language models as re-ranking agents. arXiv preprint arXiv:2304.09542 .

Thakur, N.; Reimers, N.; R¨ uckl´ e, A.; Srivastava, A.; and Gurevych, I. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2) .

Wang, X.; Wang, Z.; Gao, X.; Zhang, F.; Wu, Y.; Xu, Z.; Shi, T.; Wang, Z.; Li, S.; Qian, Q.; et al. 2024. Searching for best practices in retrieval-augmented generation. arXiv preprint arXiv:2407.01219 .

Weller, O.; Chang, B.; MacAvaney, S.; Lo, K.; Cohan, A.; Van Durme, B.; Lawrie, D.; and Soldaini, L. 2025a. FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions. In Chiruzzo, L.; Ritter, A.; and Wang, L., eds., Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) , 11926-11942. Albuquerque, New Mexico: Association for Computational Linguistics.

Weller, O.; Ricci, K.; Yang, E.; Yates, A.; Lawrie, D.; and Van Durme, B. 2025b. Rank1: Test-time compute for reranking in information retrieval. arXiv preprint arXiv:2502.18418 .

Weller, O.; Van Durme, B.; Lawrie, D.; Paranjape, A.; Zhang, Y.; and Hessel, J. 2024. Promptriever: Instructiontrained retrievers can be prompted like language models. arXiv preprint arXiv:2409.11136 .

Wu, S.; Xiong, Y.; Cui, Y.; Wu, H.; Chen, C.; Yuan, Y.; Huang, L.; Liu, X.; Kuo, T.-W.; Guan, N.; et al. 2024. Retrieval-augmented generation for natural language processing: A survey. arXiv preprint arXiv:2407.13193 .

Zhang, J.; Chen, Y.; Liu, C.; Niu, N.; and Wang, Y. 2023. Empirical evaluation of ChatGPT on requirements information retrieval under zero-shot setting. In 2023 International Conference on Intelligent Computing and Next Generation Networks (ICNGN) , 1-6. IEEE.

Zhang, L.; Wang, B.; Qiu, X.; Reddy, S.; and Agrawal, A. 2025. Rerank: Reasoning Re-ranking Agent via Reinforcement Learning. arXiv preprint arXiv:2505.20046 .

Zhang, L.; Zhang, Y.; Long, D.; Xie, P.; Zhang, M.; and Zhang, M. 2024. A Two-Stage Adaptation of Large Language Models for Text Ranking. In ACL (Findings) .

Zhang, Y.; Long, D.; Xu, G.; and Xie, P. 2022. HLATR: enhance multi-stage text retrieval with hybrid list aware transformer reranking. arXiv preprint arXiv:2205.10569 .

Zheng, Y.; Zhang, R.; Zhang, J.; Ye, Y.; Luo, Z.; Feng, Z.; and Ma, Y. 2024. LlamaFactory: Unified Efficient FineTuning of 100+ Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations) . Bangkok, Thailand: Association for Computational Linguistics.

Zhuang, S.; Ma, X.; Koopman, B.; Lin, J.; and Zuccon, G. 2025. Rank-R1: Enhancing reasoning in llm-based document rerankers via reinforcement learning. arXiv preprint arXiv:2503.06034 .

Zhuang, S.; Zhuang, H.; Koopman, B.; and Zuccon, G. 2024. A setwise approach for effective and highly efficient zeroshot ranking with large language models. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval , 38-47.

Figure 9: Illustrating examples for semantic relevance and reasoning-intensive relevance.

<!-- image -->

## A Relevance Types

With examples in Figure 9, we introduce semantic and reasoning-intensive relevance as follows.

Semantic relevance. It refers to the traditional understanding of relevance based on keyword or semantic matching between a query and a document (Thakur et al. 2021; Craswell et al. 2020, 2021). For example, the query 'Do goldfish grow?' can be lexically and semantically matched with 'A goldfish will grow to the depth of the water it is kept in. ' in the positive document.

Reasoning-intensive relevance. Different from traditional semantic relevance, reasoning-intensive reranking should be able to capture documents that may not directly answer the query but provide essential intermediate information needed for multi-step reasoning (Su et al. 2024). For example, the query requires that 'either four people all know each other or four people are all complete strangers to one another' cannot be directly answered by lexical or semantic matching with the positive document. Instead, the document provides the essential mathematical foundation of Ramsey number R ( n 1 , . . . , n c ) , which characterizes the minimum size of a complete graph such that any c -coloring of its edges contains a monochromatic complete subgraph of order n i . The original query can be transformed into a case of Ramsey number : guests at a party correspond to vertices, mutual acquaintance or stranger status corresponds to a 2 -color edge coloring, and the desired group of 4 mutual acquaintances or strangers corresponds to a monochromatic K 4 . Thus, solving the query reduces to determining R (4 , 4) . While the answer is not stated explicitly, the document supplies the critical intermediate concept required for multi-step reason- ing. Furthermore, queries with external constraints are also reasoning-intensive, determining which types of documents should or should not be considered relevant (Weller et al. 2025a). For example, the query is about the disrupted peace in Ireland and requires that 'Any interruptions to the peace process not directly attributable to acts of violence are not relevant. ' Here, the positive document is about the violent conflict between two victims from a Northern Ireland outlawed guerrilla group and the Irish Republican Army.

## B Preliminary Experiments

We conduct preliminary experiments on BRIGHT (Su et al. 2024), TREC DL (Craswell et al. 2020, 2021), and BEIR5 (Thakur et al. 2021) benchmarks in Section 3.2. We use the original queries to retrieve top 100 candidate documents using the pyserini implementation of BM25 (Lin et al. 2021). Then, we use these original queries and retrieved documents for rereanking under different settings as follows.

Comparing non-reasoning and reasoning LLMs. For non-reasoning LLM, we use Qwen3-32B and enable its nonreasoning mode to directly output a single word of 'yes' or 'no' . For reasoning LLM, we use QwQ-32B (Qwen Team 2025b) that outputs Chain-of-Thought (CoT) before giving the binary judgement. The corresponding prompts can be found in Appendix D. After generation, we extract the probability of tokens 'yes' and 'no' to calculate the normalized probability as discussed in Section 3.2. For each benchmark, we collect the normalized probabilities for all querydocument pairs from all subsets to compute the ratios, after which we visualize them in Figures 6 and 10.

<!-- image -->

(b) BEIR benchmark (5 subsets)

Figure 10: Distributions of normalized probability with nonreasoning and reasoning LLMs on TREC DL and BEIR benchmarks.

Comparing scoring discrimination. We use the same settings as above, except for using QwQ-32B (Qwen Team 2025b) and varying the scoring discrimination. Specifically, we try scoring discrimination of binary classification, integers from 0 to 3 , integers from 0 to 10 with the corresponding prompts in Appendix D. After QwQ-32B generation, we extract the output score s i with its probability Pr ( s i ) for computing the final score as discussed in Section 3.2. Table 1 reports the nDCG@10 averaged on all subsets for each benchmark.

## C Training Dataset

For the hard query dataset (HQ) built by ReasonIR (Shao et al. 2025), we randomly sample 10 , 000 queries. In the original dataset, each query has a positive document drawn from the high-quality corpus (Su et al. 2024), and a synthetic hard negative document. To enrich the negative documents, we use the ReasonIR-8B retriever (Shao et al. 2025) to retrieve top 1 , 000 negative documents from the same corpus, among which the top 10 are considered as hard negatives. Also, we randomly sample 4 documents from positions 11 -100 as medium negatives, and 4 documents from positions 101 -1 , 000 as easy negatives. In this way, we obtain 10 , 000 queries and each query is associated with 20 documents, including one positive document and 19 negative documents.

For the Promptriever training set (Weller et al. 2024), we randomly sample 5 , 000 queries, which contain curated instructions that impose additional requirements for relevance judgment. In the original dataset, each query has a positive document drawn from the MS MARCO corpus (Bajaj et al. 2016), and 1 to 3 synthetic negative documents. To enrich the negative documents, we use the ReasonIR- Given a query and a document, please give a relevance judgement of yes/no. The goal or relevance definition is: { instruction }

Please directly choose a relevance judgement from [yes, no]. Only output one word, no other words are allowed.

Here is the query: { query } Here is the document: { doc } Your output:

Figure 11: Prompt for Qwen3-32B using binary outputs.

8B retriever (Shao et al. 2025) to retrieve top 1 , 000 negative documents from the same corpus, among which the top 10 are considered as hard negatives. For the remaining negative documents, we randomly sample half from positions 11 -100 as medium negatives, and the other half from positions 101 -1 , 000 as easy negatives. In this way, we obtain 5 , 000 queries and each query is associated with 20 documents, including one positive document and 19 negative documents.

For the MS MARCO traing dataset (Bajaj et al. 2016), we randomly sample 5 , 000 queries, each of which only has one positive documents. Similarly, we use the ReasonIR-8B retriever (Shao et al. 2025) to retrieve top 1 , 000 negative documents from the same corpus, among which the top 10 are considered as hard negatives. For the remaining negative documents, we randomly sample half from positions 11 -100 as medium negatives, and the other half from positions 101 -1 , 000 as easy negatives. In this way, we obtain 5 , 000 queries and each query is associated with 20 documents, including one positive document and 19 negative documents.

Finally, we filter out any instances where the generated output exceeds 2 , 048 tokens. This procedure results in our final Supervised Fine-Tuning (SFT) dataset D containing 14 , 799 queries, as summarized in Table 2. For RL training, we randomly sample 2 , 048 queries from the SFT dataset D .

## D Prompts

For baseline methods, we use prompts from their papers and official repositories, with the corresponding token limits for query and document truncation. For our method and preliminary experiments, queries or documents longer than 2 , 048 tokens will be truncated.

For prompts shown in Figure 5 and Figures 11-13, we use different instructions listed in Table 7 for different subsets due to the diverse definitions of relevance, many of which are adapted from ReasonIR paper (Shao et al. 2025). Notably, when generating trajectories for training data in Section 3.2, the instructions used for ReasonIR hard query (HQ) training set (Shao et al. 2025) are actually those used in BRIGHT benchmark. The instruction for MS MARCO training set (Bajaj et al. 2016) is the one for TREC DL

Table 7: Task-specific instruction used in prompts.

| Benchmark (Subset)   | Instruction                                                                                                                                                                                                                                                                                                                                                                                                             |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BRIGHT (AoPS)        | We want to find different but similar math problems to the query. A document is relevant if it uses the same class of functions and shares any overlapping techniques.                                                                                                                                                                                                                                                  |
| BRIGHT (LeetCode)    | I am looking to find different problems that share similar data structures (of any kind) or algorithms (e.g. DFS, DP, sorting, traversals, etc.). I am looking for problems that share one or both of these similarities to the query. Does the passage below share any similarities? e.g. if there was a textbook on leetcode problems, this would be in the same book even though it could be in a different chapter. |
| BRIGHT (Pony)        | I will use the programming language pony. But to solve the problem above, I need to know things about pony. A passage is relevant if it contains docs that match any part (even basic parts) of the code I will have to write for the above program.                                                                                                                                                                    |
| BRIGHT (TheoremQA-Q) | Wewant to find a document which uses the same mathematical process as the query. Adocument is relevant if it uses the same mathematical process as the query.                                                                                                                                                                                                                                                           |
| BRIGHT (TheoremQA-T) | Wewant to find a document which uses the same mathematical process as the query. Adocument is relevant if it uses the same mathematical process as the query.                                                                                                                                                                                                                                                           |
| BRIGHT (others)      | A document is relevant if it contains information that helps answer or address the query. A document is not relevant if it doesn't contain information that helps answer the query, even if it mentions similar topics.                                                                                                                                                                                                 |
| BEIR / TREC DL       | Given a query, retrieval relevant passage.                                                                                                                                                                                                                                                                                                                                                                              |
| FollowIR             | Retrieval the relevant passage for the given query. Be careful about the extra requirements about relevance in the query.                                                                                                                                                                                                                                                                                               |

Figure 12: Prompt for QwQ-32B using binary outputs.

<!-- image -->

benchmark, while the instruction for Promptriever training set (Weller et al. 2024) is the one for FollowIR benchmark.

## E SFT Settings

We use LLaMA-Factory (Zheng et al. 2024) to finetune Qwen3-4B-Base, Qwen3-14B-Base, and Qwen3-32B LLMs (Qwen Team 2025a) on NVIDIA A100 (80G) GPUs. We apply LoRA (Hu et al. 2022) on all parameters with rank 32 and alpha 64 , utilizing the effective batch size of 128 . Detailed parameters are listed in Table 8.

Figure 13: Prompt for scoring with integers from 0 to 3.

<!-- image -->

## F RL Settings

We use the GRPO algorithm (Shao et al. 2024) implemented in verl project (Sheng et al. 2024) for training on NVIDIA A100 (80G) GPUs. Detailed detailed configurations are listed in Table 9.

## G Settings on BRIGHT Benchmark

On BRIGHT benchmark, instead of retrieving by BM25 with original queries, existing studies also use different firststage retrieval and hybridize with BM25 scores to further improve performance (Weller et al. 2025b; Niu et al. 2024;

Table 8: SFT configurations used in LLaMA-Factory, while those not mentioned are kept as the default values.

|                     | Configuration                                                                                             | Value                                    |
|---------------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------|
| Shared              | finetuning type lora rank lora alpha lora target cutoff len rate num train epochs lr scheduler type ratio | lora 32 64 all 2048 1e-4 1.0 cosine 0.05 |
| For Qwen3- 4B-Base  | template per device train batch size gradient accumulation steps num gpus                                 | default 8 2 8                            |
| For Qwen3- 14B-Base | template per device train batch size gradient accumulation steps num gpus                                 | default 1 8 16                           |
| For Qwen3-32B       | template per device train batch size gradient accumulation steps                                          |                                          |
|                     | learning                                                                                                  |                                          |
|                     | warmup                                                                                                    |                                          |
|                     |                                                                                                           | true                                     |
|                     | bf16                                                                                                      |                                          |
|                     |                                                                                                           | 1                                        |
|                     |                                                                                                           | 8                                        |
|                     | num gpus                                                                                                  | 16                                       |
|                     |                                                                                                           | qwen3                                    |

ielabgroup 2025). Thus, we conduct further evaluation with settings described as follows.

First-stage retrieval. We include the following settings.

- Retrieve by BM25 using GPT-4 reason-query. The firststage top 100 documents are retrieved by BM25 on GPT4's CoT reasoning content. During reranking phase, all rerankers only access the original queries and candidate documents without using such CoTs.
- Retrieve by ReasonIR-8B using GPT-4 reason-query. Similarly, the documents are retrieved using the GPT-4's CoT, except for using the state-of-the-art retriever ReasonIR8B (Shao et al. 2025). Also, such CoTs are not provided during reranking phase.

BM25 Hybrid. BM25 hybrid has been widely adopted in recent studies on the BRIGHT benchmark (Niu et al. 2024; Shao et al. 2025; ielabgroup 2025) due to its effectiveness as a low-cost model ensembling strategy. These methods combine a reranking score s i for document d i with its corresponding BM25 score s BM25.

- JudgeRank (Niu et al. 2024) calculates a final score as 100 × s i + s BM25.
- Rank-R1-32B-v0.2 (ielabgroup 2025) first applies minmax normalization to reranking and BM25 scores, respectively. Then, it calculates the final score as 0 . 1 × normalized s BM25 +0 . 9 × normalized s i .
- We also apply the same strategy as Rank-R1-32B-v0.2 on ERANK rerankers, except that we apply Z-score normal-

Table 9: RL configurations used in verl, and those not mentioned are kept as the default values.

|             | Configuration                                                                                                                                                                                | Value                                                    |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| Shared      | train batch size max prompt length max response length learning rate mini batch size clip ratio ( ϵ ) use kl loss kl loss coef ( β ) kl loss type rollout n ( G ) gpus per node total epochs | 1280 1024 1024 1e-6 320 0.2 True 0.001 low var kl 5 8 10 |
| For 4B LLM  | nnodes micro batch size per gpu                                                                                                                                                              | 1 20                                                     |
| For 14B LLM | nnodes micro batch size per gpu                                                                                                                                                              | 2 10                                                     |
| For 32B LLM | nnodes micro batch size per gpu                                                                                                                                                              | 2 10                                                     |

Figure 14: Reward curve during GRPO training for ERANK4B reranker.

<!-- image -->

ization (standardization) to the scores instead of min-max normalization. After transforming them to have a mean of 0 and a standard deviation of 1, we then calculate the final score as 0 . 2 × normalized s BM25 +0 . 8 × normalized s i .

## H Evaluation Results

Tables 10-13 presents the detailed results of each subset on four benchmarks.

We also present the reward and response length curves for the ERANK-4B reranker during GRPO training. As detailed in Appendix F, the training process consists of 10 epochs, with 32 steps per epoch. Figure 14 shows that the reward initially increases rapidly. After a brief dip from its peak around step 25 , it grows steadily and saturated after step 160 . In Figure 15, the response length first drops sharply, then gradually increased, saturating at approximately 410 tokens after 100 steps.

Figure 15: Response length curve during GRPO training for ERANK-4B reranker.

<!-- image -->

## I Rule-based Rewards

In Section 4.4, we compare different rule-based reward functions during GRPO training of rerankers.

Pointwise reward r SE. This reward assesses each querydocument pair ( q, d i ) independently as follows.

<!-- formula-not-decoded -->

where s i is the score given by policy model for pair ( q, d i ) , and t i is the corresponding reference score from the teacher model, i.e., QwQ-32B (Qwen Team 2025b) in this paper. 'Formatted' indicates that the output follows desired format.

This reward motivates the policy model to generate outputs in the correct format and with scores closely matching the reference scores.

Listwise reward r nDCG. Similar to r RR reward discussed in Section 3.3, r nDCG is also a listwise reward utilizing all N × G scores from G rollouts for N documents corresponding to a query. For the j -th rollout of document d i , let ϕ ( j ) i denote its rank among the N × G scores, with ties resolved by assigning the minimum rank associated with the tied ones.

Denote the positive and negative document sets for query q as D P and D N , respectively. We first present the Discounted Cumulative Gain (DCG) on N × G scores:

<!-- formula-not-decoded -->

where rel i is the relevance score of document d i . For simplicity, we assume rel i = 1 for positive documents ( d i ∈ D P ) and rel i = 0 for negative ones ( d i ∈ D N ). Let I ( d i ∈ D P ) be an indicator function that equals 1 only when d i ∈ D P , and define f ( ϕ ( j ) i ) = 1 log 2 ( ϕ ( j ) i +1 ) . We have

<!-- formula-not-decoded -->

The Ideal Discounted Cumulative Gain (IDCG) is computed by considering the DCG metric when all positive documents are ranked ahead of all negative ones. Using IDCG, the nDCG metric for N × G scores is calculated as:

Table 10: Detailed p -MRR on each subset of FollowIR benchmark.

| Method       |   Average |   Core17 |   News21 |   Robust04 |
|--------------|-----------|----------|----------|------------|
| BM25         |       0.0 |      0.0 |      0.0 |        0.0 |
| JudgeRank-8B |       9.9 |     14.6 |      2.0 |       13.1 |
| Rank-R1-7B   |       3.6 |      6.1 |      4.0 |        0.7 |
| Rank1-7B     |       9.1 |     10.1 |      4.8 |       12.3 |
| Rearank-7B   |       2.3 |      4.3 |      0.1 |        2.5 |
| QwQ-32B      |      14.1 |     20.9 |      9.4 |       12.0 |
| ERANK-4B     |      11.0 |     17.5 |      1.3 |       14.2 |
| ERANK-14B    |      10.3 |     13.6 |      2.0 |       15.4 |
| ERANK-32B    |      12.1 |     19.0 |      1.7 |       15.5 |

Table 11: Detailed nDCG@10 on each subset of TREC DL benchmark.

| Method       |   Average |   TREC DL19 |   TREC DL20 |
|--------------|-----------|-------------|-------------|
| BM25         |      49.3 |        50.6 |        48.0 |
| JudgeRank-8B |      62.6 |        65.4 |        59.9 |
| Rank-R1-7B   |      70.0 |        72.2 |        67.7 |
| Rank1-7B     |      67.1 |        69.0 |        65.1 |
| Rearank-7B   |      72.5 |        74.8 |        70.1 |
| QwQ-32B      |      66.1 |        67.5 |        64.8 |
| ERANK-4B     |      68.9 |        71.1 |        66.7 |
| ERANK-14B    |      67.1 |        68.9 |        65.4 |
| ERANK-32B    |      68.1 |        70.8 |        65.5 |

<!-- formula-not-decoded -->

For the j -th rollout score of a positive document d i ∈ D P , its contribution to the overall nDCG metric is f ( ϕ ( j ) i ) IDCG . Inspired by this observation, we introduce the r nDCG reward as follows. Let Φ D P = { ϕ ( j ) i | d i ∈ D P , j ∈ { 1 , 2 , · · · , G }} denote the ranks of all positive documents. We denote the maximum and minimum ranks within this set as Φ (max) D P and Φ (min) D P , respectively. The r nDCG reward is defined as:

<!-- formula-not-decoded -->

where 'formatted' indicates the policy model generates response in desired format and the score s ( j ) i can be extracted. And t i is the reference score from the reference model π ref (i.e., the SFT-tuned model).

Table 12: Detailed nDCG@10 on each subset of BRIGHT benchmark. *Taken from BRIGHT online website.

| Method                                                             | Avg.                                                               |                                                                    |                                                                    | StackExchange                                                      | StackExchange                                                      | StackExchange                                                      |                                                                    |                                                                    | Coding                                                             | Coding                                                             | Theorem-based                                                      | Theorem-based                                                      | Theorem-based                                                      |
|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|
|                                                                    |                                                                    | Bio.                                                               | Earth.                                                             | Econ.                                                              | Psy.                                                               | Rob.                                                               | Stack.                                                             | Sus.                                                               | Leet.                                                              | Pony                                                               | AoPS                                                               | TheoQ.                                                             | TheoT.                                                             |
| Retrieve top 100 documents by BM25, using original query           | Retrieve top 100 documents by BM25, using original query           | Retrieve top 100 documents by BM25, using original query           | Retrieve top 100 documents by BM25, using original query           | Retrieve top 100 documents by BM25, using original query           | Retrieve top 100 documents by BM25, using original query           | Retrieve top 100 documents by BM25, using original query           | Retrieve top 100 documents by BM25, using original query           | Retrieve top 100 documents by BM25, using original query           | Retrieve top 100 documents by BM25, using original query           | Retrieve top 100 documents by BM25, using original query           | Retrieve top 100 documents by BM25, using original query           | Retrieve top 100 documents by BM25, using original query           | Retrieve top 100 documents by BM25, using original query           |
| BM25                                                               | 13.7                                                               | 18.2                                                               | 27.9                                                               | 16.5                                                               | 13.4                                                               | 10.9                                                               | 16.3                                                               | 16.1                                                               | 24.7                                                               | 4.3                                                                | 6.5                                                                | 7.3                                                                | 2.1                                                                |
| Rank-R1-7B                                                         | 15.7                                                               | 23.4                                                               | 29.2                                                               | 16.4                                                               | 23.0                                                               | 17.0                                                               | 10.9                                                               | 25.9                                                               | 15.8                                                               | 4.8                                                                | 5.8                                                                | 7.1                                                                | 9.3                                                                |
| Rank1-7B                                                           | 18.2                                                               | 31.6                                                               | 34.4                                                               | 18.0                                                               | 23.5                                                               | 16.7                                                               | 18.6                                                               | 22.9                                                               | 20.1                                                               | 9.4                                                                | 4.5                                                                | 9.4                                                                | 9.9                                                                |
| Rearank-7B                                                         | 17.4                                                               | 23.2                                                               | 26.7                                                               | 17.2                                                               | 22.7                                                               | 18.2                                                               | 16.7                                                               | 25.3                                                               | 26.8                                                               | 7.2                                                                | 7.5                                                                | 7.7                                                                | 9.7                                                                |
| JudgeRank-8B                                                       | 17.0                                                               | 28.7                                                               | 32.2                                                               | 20.9                                                               | 24.6                                                               | 16.5                                                               | 18.3                                                               | 20.6                                                               | 11.7                                                               | 7.1                                                                | 4.7                                                                | 8.4                                                                | 10.0                                                               |
| + BM25 hybrid                                                      | 19.0                                                               | 28.3                                                               | 36.5                                                               | 21.9                                                               | 24.1                                                               | 15.3                                                               | 22.7                                                               | 23.5                                                               | 25.1                                                               | 6.8                                                                | 6.7                                                                | 8.3                                                                | 8.5                                                                |
| QwQ-32B                                                            | 23.2                                                               | 32.7                                                               | 44.7                                                               | 23.9                                                               | 30.5                                                               | 21.6                                                               | 23.8                                                               | 23.8                                                               | 25.7                                                               | 17.3                                                               | 12.7                                                               | 11.2                                                               | 11.1                                                               |
| + BM25 hybrid                                                      | 23.9                                                               | 33.0                                                               | 46.5                                                               | 25.3                                                               | 28.2                                                               | 21.1                                                               | 25.6                                                               | 25.3                                                               | 28.7                                                               | 17.2                                                               | 13.0                                                               | 11.8                                                               | 10.7                                                               |
| ERANK-4B                                                           | 22.7                                                               | 30.4                                                               | 42.5                                                               | 21.5                                                               | 27.7                                                               | 22.4                                                               | 22.9                                                               | 24.0                                                               | 31.6                                                               | 14.6                                                               | 11.0                                                               | 12.1                                                               | 11.4                                                               |
| + BM25 hybrid                                                      | 23.9                                                               | 32.7                                                               | 45.4                                                               | 23.1                                                               | 29.2                                                               | 21.8                                                               | 24.7                                                               | 25.6                                                               | 33.4                                                               | 15.6                                                               | 12.2                                                               | 12.4                                                               | 10.5                                                               |
| ERANK-14B                                                          | 23.1                                                               | 31.2                                                               | 43.6                                                               | 25.8                                                               | 27.8                                                               | 23.1                                                               | 23.9                                                               | 24.6                                                               | 29.8                                                               | 16.8                                                               | 8.6                                                                | 10.5                                                               | 11.9                                                               |
| + BM25 hybrid                                                      | 24.6                                                               | 32.7                                                               | 45.8                                                               | 27.2                                                               | 29.4                                                               | 24.1                                                               | 25.6                                                               | 26.5                                                               | 32.7                                                               | 17.5                                                               | 10.5                                                               | 12.1                                                               | 11.9                                                               |
| ERANK-32B                                                          | 24.4                                                               | 33.5                                                               | 44.5                                                               | 23.9                                                               | 29.4                                                               | 23.8                                                               | 27.1                                                               | 26.4                                                               | 32.5                                                               | 15.8                                                               | 12.5                                                               | 10.9                                                               | 12.2                                                               |
| + BM25 hybrid                                                      | 25.4                                                               | 35.1                                                               | 46.2                                                               | 25.5                                                               | 29.4                                                               | 24.2                                                               | 27.5                                                               | 27.6                                                               | 34.9                                                               | 16.7                                                               | 13.2                                                               | 11.5                                                               | 12.5                                                               |
| Retrieve top 100 documents by BM25, using GPT4 reason-query        | Retrieve top 100 documents by BM25, using GPT4 reason-query        | Retrieve top 100 documents by BM25, using GPT4 reason-query        | Retrieve top 100 documents by BM25, using GPT4 reason-query        | Retrieve top 100 documents by BM25, using GPT4 reason-query        | Retrieve top 100 documents by BM25, using GPT4 reason-query        | Retrieve top 100 documents by BM25, using GPT4 reason-query        | Retrieve top 100 documents by BM25, using GPT4 reason-query        | Retrieve top 100 documents by BM25, using GPT4 reason-query        | Retrieve top 100 documents by BM25, using GPT4 reason-query        | Retrieve top 100 documents by BM25, using GPT4 reason-query        | Retrieve top 100 documents by BM25, using GPT4 reason-query        | Retrieve top 100 documents by BM25, using GPT4 reason-query        | Retrieve top 100 documents by BM25, using GPT4 reason-query        |
| BM25                                                               | 27.0                                                               | 53.6                                                               | 54.1                                                               | 24.3                                                               | 38.7                                                               | 18.9                                                               | 27.7                                                               | 26.3                                                               | 19.3                                                               | 17.6                                                               | 3.9                                                                | 19.2                                                               | 20.8                                                               |
| Rank-R1-7B                                                         | 23.9                                                               | 38.2                                                               | 29.4                                                               | 23.4                                                               | 33.0                                                               | 24.9                                                               | 14.9                                                               | 33.2                                                               | 18.2                                                               | 16.1                                                               | 3.8                                                                | 16.6                                                               | 34.8                                                               |
| Rank1-7B                                                           | 25.5                                                               | 45.8                                                               | 37.0                                                               | 22.2                                                               | 31.7                                                               | 20.6                                                               | 23.0                                                               | 34.2                                                               | 15.7                                                               | 19.8                                                               | 1.3                                                                | 19.8                                                               | 34.7                                                               |
| Rearank-7B                                                         | 29.1                                                               | 42.0                                                               | 37.5                                                               | 26.4                                                               | 39.1                                                               | 25.0                                                               | 25.1                                                               | 32.6                                                               | 26.2                                                               | 29.2                                                               | 5.9                                                                | 28.0                                                               | 32.2                                                               |
| XRR-Gemini-2.5-Flash*                                              | 40.3                                                               | 63.1                                                               | 55.4                                                               | 38.5                                                               | 52.9                                                               | 37.1                                                               | 38.2                                                               | 44.6                                                               | 21.9                                                               | 35.0                                                               | 15.7                                                               | 34.4                                                               | 46.2                                                               |
| JudgeRank-8B                                                       | 24.4                                                               | 41.4                                                               | 34.7                                                               | 26.2                                                               | 36.0                                                               | 24.0                                                               | 27.6                                                               | 26.1                                                               | 10.2                                                               | 14.2                                                               | 3.4                                                                | 20.3                                                               | 28.9                                                               |
| + BM25 hybrid                                                      | 31.0                                                               | 55.3                                                               | 53.4                                                               | 31.4                                                               | 41.6                                                               | 26.7                                                               | 32.8                                                               | 33.3                                                               | 19.6                                                               | 19.5                                                               | 3.7                                                                | 23.4                                                               | 30.9                                                               |
| ERANK-4B                                                           | 32.9                                                               | 48.2                                                               | 46.7                                                               | 30.0                                                               | 43.1                                                               | 28.4                                                               | 31.5                                                               | 38.1                                                               | 28.5                                                               | 23.5                                                               | 10.4                                                               | 26.9                                                               | 39.0                                                               |
| + BM25 hybrid                                                      | 36.1                                                               | 58.5                                                               | 55.6                                                               | 32.6                                                               | 47.2                                                               | 30.0                                                               | 34.7                                                               | 40.6                                                               | 28.9                                                               | 25.8                                                               | 11.2                                                               | 28.9                                                               | 39.0                                                               |
| ERANK-14B                                                          | 33.5                                                               | 51.4                                                               | 48.6                                                               | 30.8                                                               | 41.3                                                               | 26.7                                                               | 35.6                                                               | 39.1                                                               | 27.3                                                               | 26.4                                                               | 10.9                                                               | 25.7                                                               | 37.9                                                               |
| + BM25 hybrid                                                      | 36.7                                                               | 59.9                                                               | 57.3                                                               | 34.8                                                               | 46.7                                                               | 29.5                                                               | 36.9                                                               | 41.2                                                               | 29.4                                                               | 28.7                                                               | 10.5                                                               | 28.0                                                               | 38.1                                                               |
| ERANK-32B                                                          | 34.6                                                               | 55.5                                                               | 49.1                                                               | 30.4                                                               | 44.7                                                               | 27.9                                                               | 35.6                                                               | 40.6                                                               | 29.2                                                               | 24.2                                                               | 10.4                                                               | 27.6                                                               | 40.0                                                               |
| + BM25 hybrid                                                      | 37.4                                                               | 62.9                                                               | 57.5                                                               | 33.2                                                               | 48.4                                                               | 30.5                                                               | 36.5                                                               | 42.3                                                               | 32.7                                                               | 25.4                                                               | 10.8                                                               | 28.7                                                               | 40.4                                                               |
| Retrieve top 100 documents by ReasonIR-8B, using GPT4 reason-query | Retrieve top 100 documents by ReasonIR-8B, using GPT4 reason-query | Retrieve top 100 documents by ReasonIR-8B, using GPT4 reason-query | Retrieve top 100 documents by ReasonIR-8B, using GPT4 reason-query | Retrieve top 100 documents by ReasonIR-8B, using GPT4 reason-query | Retrieve top 100 documents by ReasonIR-8B, using GPT4 reason-query | Retrieve top 100 documents by ReasonIR-8B, using GPT4 reason-query | Retrieve top 100 documents by ReasonIR-8B, using GPT4 reason-query | Retrieve top 100 documents by ReasonIR-8B, using GPT4 reason-query | Retrieve top 100 documents by ReasonIR-8B, using GPT4 reason-query | Retrieve top 100 documents by ReasonIR-8B, using GPT4 reason-query | Retrieve top 100 documents by ReasonIR-8B, using GPT4 reason-query | Retrieve top 100 documents by ReasonIR-8B, using GPT4 reason-query | Retrieve top 100 documents by ReasonIR-8B, using GPT4 reason-query |
| ReasonIR-8B                                                        | 30.5                                                               | 43.5                                                               | 43.0                                                               | 32.8                                                               | 38.9                                                               | 21.1                                                               | 30.6                                                               | 27.3                                                               | 31.6                                                               | 19.6                                                               | 7.3                                                                | 34.1                                                               | 36.7                                                               |
| Rank-R1-7B                                                         | 24.1                                                               | 39.3                                                               | 28.1                                                               | 23.9                                                               | 30.0                                                               | 17.3                                                               | 18.1                                                               | 33.2                                                               | 18.6                                                               | 15.0                                                               | 4.2                                                                | 25.4                                                               | 35.7                                                               |
| Rank1-7B                                                           | 24.3                                                               | 44.1                                                               | 33.5                                                               | 21.8                                                               | 30.0                                                               | 15.0                                                               | 22.1                                                               | 28.5                                                               | 11.8                                                               | 21.7                                                               | 1.2                                                                | 26.2                                                               | 36.2                                                               |
| Rearank-7B                                                         | 27.5                                                               | 35.3                                                               | 29.8                                                               | 25.5                                                               | 35.7                                                               | 19.1                                                               | 20.1                                                               | 32.9                                                               | 29.9                                                               | 20.2                                                               | 6.2                                                                | 36.7                                                               | 38.3                                                               |
| JudgeRank-8B                                                       | 20.2                                                               | 37.1                                                               | 27.2                                                               | 19.2                                                               | 28.6                                                               | 11.6                                                               | 19.9                                                               | 22.5                                                               | 10.2                                                               | 10.2                                                               | 3.6                                                                | 22.9                                                               | 29.4                                                               |
| + BM25 hybrid                                                      | 22.7                                                               | 40.4                                                               | 28.9                                                               | 22.3                                                               | 35.5                                                               | 14.2                                                               | 23.0                                                               | 25.7                                                               | 11.8                                                               | 10.6                                                               | 3.6                                                                | 25.2                                                               | 31.1                                                               |
| Rank-R1-v0.2-32B*                                                  | 37.7                                                               | 60.1                                                               | 56.3                                                               | 36.6                                                               | 52.1                                                               | 30.2                                                               | 37.6                                                               | 45.9                                                               | 25.5                                                               | 14.6                                                               | 10.1                                                               | 38.6                                                               | 44.3                                                               |
| +BM25 hybrid*                                                      | 40.0                                                               | 64.4                                                               | 60.1                                                               | 38.3                                                               | 52.2                                                               | 30.7                                                               | 40.6                                                               | 46.7                                                               | 33.3                                                               | 17.4                                                               | 10.1                                                               | 38.6                                                               | 47.7                                                               |
| ERANK-4B                                                           | 30.5                                                               | 42.1                                                               | 42.5                                                               | 26.3                                                               | 36.4                                                               | 20.8                                                               | 27.3                                                               | 33.2                                                               | 31.7                                                               | 21.8                                                               | 10.9                                                               | 32.8                                                               | 40.6                                                               |
| + BM25 hybrid                                                      | 38.7                                                               | 58.7                                                               | 56.6                                                               | 33.8                                                               | 48.7                                                               | 29.1                                                               | 38.2                                                               | 40.8                                                               | 32.7                                                               | 32.0                                                               | 9.8                                                                | 35.2                                                               | 48.4                                                               |
| ERANK-14B                                                          | 31.8                                                               | 46.6                                                               | 42.5                                                               | 25.2                                                               | 37.3                                                               | 19.6                                                               | 30.2                                                               | 34.6                                                               | 31.9                                                               | 25.6                                                               | 10.5                                                               | 32.4                                                               | 45.0                                                               |
| + BM25 hybrid                                                      | 39.3                                                               | 60.1                                                               | 55.8                                                               | 34.2                                                               | 49.5                                                               | 28.4                                                               | 40.1                                                               | 41.2                                                               | 33.3                                                               | 34.0                                                               | 11.0                                                               | 35.7                                                               | 48.5                                                               |
| ERANK-32B                                                          | 32.8                                                               | 49.3                                                               | 43.4                                                               | 28.4                                                               | 36.8                                                               | 20.8                                                               | 32.8                                                               | 34.6                                                               | 36.0                                                               | 22.3                                                               | 11.3                                                               | 34.4                                                               | 43.5                                                               |
| + BM25 hybrid                                                      | 40.2                                                               | 61.5                                                               | 56.6                                                               | 36.5                                                               | 49.4                                                               | 28.9                                                               | 41.8                                                               | 42.7                                                               | 36.0                                                               | 31.6                                                               | 11.4                                                               | 37.0                                                               | 49.1                                                               |

Table 13: Detailed nDCG@10 on each subset of BEIR benchmark.

<!-- image -->

| Method       |   Avg. |   ArguA. |   CliF. |   CQA. |   DBP. |   Fever |   FiQA |   HotP. |   MSM. |   NFC. |   NQ |   Quora |   SciD. |   SciF. |   TREC-C. |   TouC. |
|--------------|--------|----------|---------|--------|--------|---------|--------|---------|--------|--------|------|---------|---------|---------|-----------|---------|
| BM25         |   40.8 |     30.0 |    16.5 |   30.2 |   31.8 |    65.1 |   23.6 |    63.3 |   22.8 |   32.2 | 30.6 |    78.9 |    14.9 |    67.9 |      59.5 |    44.2 |
| JudgeRank-8B |   39.1 |     13.4 |    14.8 |   32.0 |   34.5 |    47.4 |   30.2 |    53.3 |   27.2 |   34.1 | 46.5 |    65.8 |    15.2 |    64.5 |      81.6 |    25.6 |
| Rank-R1-7B   |   49.0 |     26.0 |    24.8 |   39.8 |   42.7 |    78.6 |   38.9 |    67.8 |   35.1 |   36.2 | 55.8 |    84.2 |    18.8 |    74.9 |      82.2 |    29.0 |
| Rank1-7B     |   44.2 |     20.6 |    16.7 |   33.5 |   37.5 |    66.4 |   37.8 |    62.7 |   31.0 |   35.6 | 53.1 |    70.2 |    17.0 |    76.8 |      79.8 |    25.1 |
| Rearank-7B   |   49.0 |     32.1 |    21.9 |   41.0 |   45.2 |    73.8 |   36.6 |    72.3 |   35.5 |   32.7 | 54.5 |    79.1 |    20.2 |    74.8 |      79.7 |    36.1 |
| QwQ-32B      |   47.5 |     51.7 |    16.2 |   37.9 |   40.7 |    68.0 |   37.1 |    59.0 |   30.5 |   36.3 | 50.9 |    78.0 |    19.7 |    75.7 |      82.4 |    28.9 |
| ERANK-4B     |   44.8 |     36.7 |    17.8 |   38.6 |   39.1 |    56.8 |   34.3 |    53.7 |   34.4 |   35.1 | 51.8 |    74.0 |    17.1 |    73.1 |      79.6 |    29.9 |
| ERANK-14B    |   47.1 |     47.6 |    18.5 |   38.5 |   38.5 |    65.2 |   36.2 |    56.9 |   34.7 |   35.0 | 52.1 |    78.7 |    17.4 |    74.2 |      81.8 |    31.3 |
| ERANK-32B    |   47.7 |     49.1 |    18.7 |   38.4 |   38.8 |    67.1 |   37.0 |    57.6 |   34.8 |   36.7 | 52.4 |    78.6 |    17.6 |    73.9 |      82.4 |    32.1 |