1

## OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation

Pengfei Zhou 1* , Xiaopeng Peng 2* , Jiajun Song 3 , Chuanhao Li 1 , Zhaopan Xu 1 , Yue Yang 4,1 , Ziyao Guo 1,5 , Hao Zhang 1 , Yuqi Lin 1 , Yefei He 1 , Lirui Zhao 1 , Shuo Liu 1 , Tianhua Li 1,4 , Yuxuan Xie 1,4 , Xiaojun Chang 6,7 , Yu Qiao 1 , Wenqi Shao 1 , Kaipeng Zhang 1,8 † 2 RIT, 3 RUC, 4 SJTU, 5 NUS, 6 USTC, 7

Shanghai AI Laboratory, MBZUAI, 8 Shanghai Innovation Institute https://opening-benchmark.github.io

## Abstract

Multimodal Large Language Models (MLLMs) have made significant strides in visual understanding and generation tasks. However, generating interleaved image-text content remains a challenge, which requires integrated multimodal understanding and generation abilities. While the progress in unified models offers new solutions, existing benchmarks are insufficient for evaluating these methods due to data size and diversity limitations. To bridge this gap, we introduce OpenING, a comprehensive benchmark comprising 5,400 high-quality human-annotated instances across 56 real-world tasks. OpenING covers diverse daily scenarios such as travel guide, design, and brainstorming, offering a robust platform for challenging interleaved generation methods. In addition, we present IntJudge, a judge model for evaluating open-ended multimodal generation methods. Trained with a novel data pipeline, our IntJudge achieves an agreement rate of 82.42% with human judgments, outperforming GPT-based evaluators by 11.34%. Extensive experiments on OpenING reveal that current interleaved generation methods still have substantial room for improvement. Key findings on interleaved image-text generation are further presented to guide the development of next-generation models.

## 1. Introduction

Building upon the remarkable understanding and generation capabilities of Large Language Models (LLMs) [1, 64, 65, 67], Multimodal LLMs (MLLMs) are making progress in various tasks [5, 42, 84, 87, 91]. However, generating interleaved image-text content remains challenging [37, 63, 71], despite its important role in both research and applications (e.g., multimodal reasoning [11, 46], education [17, 36] and design [34, 59]). Since human brains can naturally combine visual and textual signals for more efficient information ex- change [25, 31], achieving such integrated ability is crucial for advancing towards Artificial General Intelligence (AGI).

∗ Equal contribution † Corresponding author (zhangkaipeng@pjlab.org.cn)

Figure 1. Motivation: (a) Rapid progress of interleaved imagetext generation. (b) Interleaved content is essential to provide key information for complex real-world tasks (e.g., product design).

<!-- image -->

As shown in Fig. 1, the emergence of unified models that combine understanding and generation abilities opens up new possibilities for interleaved image-text generation [79, 96]. However, the lack of reliable benchmarks to evaluate interleaved generation remains an obstacle [62, 71]. Most existing benchmarks evaluate text or image output separately, failing to capture the intricacies of simultaneously generating both [44, 61, 85, 86]. Interleaved benchmarks like OpenLEAF [4] and InterleavedBench [43] are limited in size, scope, and query diversity. For example, InterleavedBench includes only 815 instances across 10 tasks sourced from public datasets such as VIST [32] and WikiHow [83]. These benchmarks do not adequately reflect real-world needs and are vulnerable to data contamination [78].

To fill the gap, we introduce OpenING, a comprehensive benchmark for evaluating open-ended interleaved generation. Unlike previous benchmarks, OpenING offers a broader set of real-world data and tasks (e.g., brainstorming, recommendations, and content creation) derived from daily scenarios like fashion, cooking, and travel. As shown in Fig. 2 and Table 1, the curated OpenING includes 5,400 instances of multi-step interleaved image-text content across 23 metatopics and 56 tasks, with diverse, carefully designed queries for various topics. To address the challenges of collecting and standardizing data from disparate domains, we develop an efficient annotation pipeline and produce high-quality human-annotated data, reducing data contamination risks.

Figure 2. OpenING benchmark consists of 23 meta-topics (inner ring) which are further categorized into 56 specific tasks (see the number of tasks on the outer ring and details in Supplementary Materials). Examples showcase interleaved generation in eight representative domains.

<!-- image -->

In addition, many previous benchmarks rely on GPTbased scoring metrics [4, 43], which are prone to be affected by inherent biases in GPT models and potential data leakage in API usage [72]. To overcome the challenges of assessing open-ended multimodal generation, we introduce IntJudge, a robust judge model. We also propose an Interleaved Arena to facilitate annotation of training data and a Reference-Augmented Generation (RAG) approach to scale up the data size. Trained with this enhanced data pipeline, IntJudge achieves 82.42% average agreement with human judgments, an 11.34% improvement over GPT-4o as a judge.

We evaluate representative interleaved generation methods using our OpenING. Key findings from our experiments include: 1) Generating coherent and high-quality interleaved content remains challenging for all models, whereas humanannotated content consistently receives the highest ratings over generated content; 2) Integrated pipelines (e.g., Gemini

+ Flux) outperform end-to-end models (e.g., Anole) in terms of image-text coherence and visual quality, possibly due to the more developed foundation models. End-to-end and twostage generators (e.g., SEED-X) with unified architectures show great potential, especially as they continue to advance and potentially take advantage of developed methods; and 3) While text answers generated by GPT can be more informative than human-annotated answers, annotated natural images remain preferable to generated images, highlighting the challenges in high-quality image generation. The major contributions of this paper are summarized as follows:
- A High-quality Benchmark . We present OpenING, a comprehensive benchmark for evaluating open-ended interleaved image-text generation. OpenING includes 5,400 human-annotated instances across 56 real-world tasks, aiming to challenge and improve interleaved generation methods and also support the development of judge models for assessing open-ended multimodal generation.
- A Robust Judge . We introduce IntJudge, a judge model for rating interleaved generation methods. We train IntJudge with an enhanced data pipeline, achieving an 82.42% agreement rate with human judgments and significantly outperforming GPT-based judge. Moreover, IntJudge has proven to be effective in assessing new unseen models.

|                       | Data Coverage   | Data Coverage   | Data Coverage   | Data Coverage   | Data Coverage   | Data Coverage   |             |               |
|-----------------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-------------|---------------|
| Benchmark             | Meta-Topics     | Tasks           | Instances       | Images          | Steps           | SpI             | Open-source | Offline Judge |
| OpenLEAF [4]          | 2               | 10              | 660             | -               | -               | -               | ✗           | ✗             |
| InterleavedBench [43] | 4               | 10              | 815             | 1,513           | 1,601           | 1.96            | ✓           | ✗             |
| OpenING (ours)        | 23              | 56              | 5,400           | 17,603          | 20,094          | 3.72            | ✓           | ✓             |

Table 1. Comparison between OpenING and existing benchmarks. OpenING includes more comprehensive data and task coverage with an openly available judge model. Steps: a step is indicated by an input instruction or an output image-text pair; SpI: Steps per Instance.

- A Comprehensive Leaderboard . We provide detailed rankings and analysis of interleaved generation methods and compare IntJudge and GPT-4o evaluations with human judgments. Our findings indicate that while current opensource end-to-end models lag behind integrated pipelines, end-to-end and two-stage generators with unified architectures present great potential and warrant further explorations to advance interleaved image-text generation.

## 2. Related Work

Interleaved Image-Text Generation. The development of MLLMs has greatly advanced interleaved image-text generation [35]. Early models like Stable Diffusion [20, 53], DALLE [52], and autoregressive (AR) methods (e.g., VAR [66] and Lumina-mGPT [41]) focused on unidirectional tasks, such as image understanding and text-to-image generation. Flamingo [2] was the first MLLM to process interleaved image-text content. More recent models, such as MiniGPT-5 [92] and SEED series [23, 24, 81], achieve interleaved generation by combining AR-based text generation and diffusionbased visual generation. Native AR models such as Emu3 [71] and Chameleon [63] offer a unified framework to generate and reason over mixed-modal documents. Anole [16] reproduces the image generation capability of Chameleon through efficient fine-tuning on an interleaved image-text data. However, benchmarks for evaluating interleaved imagetext generation remain in early stages. Previous works, such as OpenLEAF [4] and InterleavedBench [43] focused on a small set of topics and lack the depth and breadth for realworld applications. To achieve a more reliable and holistic evaluation of interleaved generation, we propose OpenING based on comprehensive real-world scenarios.

Evaluation of Open-ended Multimodal Generation. Evaluating open-ended multimodal generation is inherently challenging due to the need to assess both visual and textual quality in open domains [4, 56, 74]. Existing text generation metrics, such as BLEU [49] and ROUGE [39], fall short in measuring visual quality and text-image coherence. Conversely, visual quality metrics like FID [30] and IS [54] lack consideration of textual elements. Contrastive metrics, such as CLIPScore [29] can measure text-image alignment but fail to fully evaluate the quality of open-ended interleaved contents, where multiple correct answers may exist. GPT- based scoring [43, 89] provides improved measurements to assess the diversity and coherence of the interleaved outputs. However, GPT tends to be biased and favors the contents generated by itself [6, 72]. Human evaluation, although reliable, is not scalable due to its laborious nature. To close this gap, we introduce IntJudge, a judge model that is highly aligned with human judgments in evaluating the open-ended multimodal generation. To mitigate the instability of subjective scores [14, 93], our IntJudge evaluates models through pairwise comparisons in an arena-style framework [38].

## 3. OpenING Benchmark

## 3.1. Problem Definition

The task of interleaved image-text generation involves generating a sequence of text and images based on a given prompt. Each interleaved generation model (referred to as a multimodal agent) receives an input prompt P , which can be text-only or include both text and images. The multimodal agent outputs an interleaved image-text sequence: S = [ s 1 , s 2 , . . . , s N ] , where N is the number of steps. Each element s i = &lt; T i , I i &gt; in step i consists of a text segment T i and an image I i . Each s i is generated based on the prompt P and all outputs history as s i = f ( P , s 1 , s 2 , . . . , s i -1 ) , where f denotes the the generation function of an agent. The objective is to find an optimal output sequence set S ∗ :

<!-- formula-not-decoded -->

where s ∗ i in each step is semantically consistent with the input prompt while maintaining coherence throughout the entire sequence. The performance of an agent is evaluated based on how well the generated S meets predefined criteria.

## 3.2. Data Curation

Collecting and annotating interleaved image-text data is inherently challenging due to the scarcity of high-quality data. It is particularly difficult to gather and pair multimodal data from disparate domains and ensure consistency [82]. We created OpenING over three months, with nearly 50 people involved in an efficient pipeline, which is shown in Fig. 3(a). Topic Conceptualization. With the assistance of multiple AI agents, we brainstormed and identified the most relevant real-world scenarios that require interleaved image-text generation. These insights were conceptualized into 23 metatopics and divided into 56 specific tasks.

Figure 3. Overview of data curation and the proposed judge pipeline. (a) We construct our OpenING benchmark in a top-down manner, which involves five stages: conceptualization, data collection, annotation, filtering and processing. (b) We use the Dev Set of OpenING to train the proposed IntJudge and evaluate interleaved image-text generation on the Test Set to compare our IntJudge with human and GPT-4o.

<!-- image -->

Data Collection and Annotation. Interleaved image-text data was collected from more than 20 sources, including social media (e.g., Rednote 1 ), video sharing websites (e.g., YouTube 2 ), search engines (e.g. Google 3 ), and open dataset platforms (e.g. OpenDataLab [28]). The complete list of data sources is provided in the Supplementary Material. To ensure the highest data quality, a team of 28 professional annotators contributed under the supervision of 14 data experts. They performed efficient manual annotations using the IntLabel tool that we developed. Annotations were organized into a standard format, and each instance was limited to ten steps to avoid potential breaking of context constraints.

Data Filtering and Quality Control. We conducted crosschecks with annotators and data experts to ensure consistency, relevance, and coherence in each instance. Each task was required to include diverse sources and topics. In cases where data acquisition is complex, the annotators were instructed to supplement the data set with content generated by GPT-4o [48] and Stable Diffusion XL [51]. To further enhance data quality, exclusive protocols are proposed to filter unqualified data. The qualified data were then redistributed to each respective task to reach the required quantity.

Data Processing. Post-processings were carried out to ensure the linguistic consistency of our benchmark. GPT-4o API is used to translate the annotated Chinese text to English, followed by data experts reviewing the accuracy. We also implemented image translation 4 to convert any Chinese characters to English in images. Finally, the prompts were refined for each task to achieve desired generation results, as detailed in the Supplementary Materials.

1 https://www.xiaohongshu.com

2 https://www.youtube.com

3 https://www.google.com

Dataset Splitting. As illustrated in Fig. 2, our OpenING benchmark ultimately includes 5,400 annotated instances, spanning 23 distinct meta-topics and 56 tasks. The annotated instances of OpenING are divided into a Dev Set (3,240 instances) and a Test Set (2,160 instances). The Dev Set supports the training of judge models, and the Test Set is used to evaluate the zero-shot performance of different models.

## 4. IntJudge Model

## 4.1. Interleaved Arena

Evaluating open-ended interleaved image-text generation is challenging due to the complexity of assessing multiple images and text, as well as the open-ended nature of generation, where multiple valid answers exist. Given that pairwise comparison is more stable than subjective scoring [14], we introduce Interleaved Arena, on which pairwise assessments were conducted using three evaluators: human judges, GPTbased judges, and the proposed IntJudge.

In the Interleaved Arena, interleaved outputs from agents on the OpenING Test Set are saved in a unified format. In each evaluation round, judges compare outputs from two anonymous agents and rate the interleaved outputs based on seven criteria: Correctness, Image-Text Coherency, Multistep Consistency, Content Quality, Human Preference Alignment, Completeness, and Content Richness (see supplemen- tary materials for more details). To balance evaluation reliability and efficiency, we propose a roulette matching algorithm to sample E distinct battle pairs for each data instance.

4 https://github.com/zyddnys/manga-image-translator

Let K represent the set of tasks, and M denote a set of Arena agents. Each task k ∈ K has D k data instances. A permutation σ k ∈ A |M| is sampled by randomly shuffling the agent order, where A |M| is the set of all agent permutations. The set of sampled battle pairs is given by:

<!-- formula-not-decoded -->

where i = 1 , 2 , . . . , D k . Additional sampling rounds may be performed to obtain E distinct battle pairs for each data instance, where E ≤ |M| ( |M| 1) / 2 . To avoid duplications, a set R k,d is maintained in the d -th round, which stores all unique pairs sampled in previous rounds:

<!-- formula-not-decoded -->

For the current pair σ k,d ( a ) and σ k,d ( b ) , we enforce:

̸

<!-- formula-not-decoded -->

Under the assumption of uniform distribution, we define the coverage time T k to ensure all agents are evaluated in task k :

<!-- formula-not-decoded -->

The overall expected coverage time is written as:

<!-- formula-not-decoded -->

where H |M| is the |M| -th harmonic number.

## 4.2. Judge Pipelines

Human Judge. In the human judge, annotators compare outputs from two multimodal agents for each input prompt and select a winner based on seven predefined criteria. The voting results are used to rank interleaved generation methods based on their win rates. Since the previous studies [14, 93] noted that excessive ties cause inefficiency, our annotators are instructed to favor one agent in cases of a tie, denoting as Tie(A) or Tie(B) based on the slight preference.

GPT-based Judge. To enable scalability, we employ GPT-4o to automate the evaluation process. The GPT-4o is prompted to analyze interleaved outputs and decide the winner of each battle pair. Moreover, we use an additional prompt to obtain the score breakdown and explanations. While this strategy allows for scalable and explainable evaluation, GPT-based judges still have a high error rate due to their prior bias and lack of alignment with human preferences. GPT also raises privacy, data leakage, and cost concerns.

IntJudge. To address issues in GPT-based evaluators, we propose IntJudge to improve evaluation accuracy and alignment with human preferences. As an offline judge, IntJudge provides efficient large-scale evaluations with consistent criteria, ensuring fair and reproducible results for benchmarking interleaved image-text generation. Upon exploring several MLLMs including InternLM-XComposer2.5 (InternLMX2.5) [88] and Qwen2-VL [69], we select Qwen2VL-7B as the foundation model for training IntJudge, achieving an optimal balance between efficiency and accuracy.

## 4.3. Training of IntJudge

To enhance IntJudge training, a Reference-Augmented Generation (RAG) approach is proposed to scale up the training data set. As shown in Fig. 3(b), our IntJudge model is trained on the combination of human-annotated pairwise data from the Dev Set and the RAG pairs. In our RAG approach, models are provided with gold real-world answers from the Dev Set and prompted to generate responses based on these gold answers. Pairwise data are formed by pairing a plain generation result with an RAG-based output, where the RAG result is assigned as the winner. A bag of models, including g seen interleaved generation methods are used for plain generation and RAG. The training objective is defined as:

<!-- formula-not-decoded -->

where λ 1 , λ 2 , λ 3 and λ 4 are weighting coefficients, L CE, L CT, L MSE, and L PR are respectively cross-entropy, contrastive, MSE, and pairwise ranking losses. The trained IntJudge was tested in a zero-shot setting on both unseen and seen models to validate its generalizability.

## 5. Experiments

## 5.1. Experimental Setup

Models. We evaluated 10 representative interleaved methods, categorized into three types: 1) Integrated pipeline combines independent text and image generation models, examples include GPT-4o+DALL · E-3 [8, 48] and Gemini1.5+Flux [9, 64]; 2) Two-stage generator , such as Emu2 [60], SEED-X [23], and Show-o [79], has a unified model architecture but generates text and image in two separate stages; 3) End-to-end generator produces image-text contents in a single stage, such models include GILL [35], NExT-GPT [75], MiniGPT-5 [92], SEED-LLaMA [22], and Anole [16]. We keep GPT-4o+DALL · E-3, Anole, SEEDLLaMA, and NExT-GPT as unseen models for IntJudge validation. The rest models are seen in IntJudge training. Evaluation Metrics. Model performance are evaluated using two key metrics: win rate and agreement. Win rate indicates how often a model wins in pairwise comparisons. Four methods used to handle ties include 1) Force Dividing Tie (FDT): We force judges to assign ties with a more leaning model

Table 2. Comparison of model win rates evaluated by human, GPT-4o, and our IntJudge under FDT and different tie metrics. FDT: Force Dividing Tie metric. w/o Tie: Non-tie case. w/ Tie (0) and w/ Tie Tie (.5): Count a tie as 0 and 0.5 wins for a model in a battle, respectively.

| Method         | Human Evaluation   | Human Evaluation   | Human Evaluation   | Human Evaluation   | GPT Evaluation   | GPT Evaluation   | GPT Evaluation   | GPT Evaluation   | IntJudge Evaluation   | IntJudge Evaluation   | IntJudge Evaluation   | IntJudge Evaluation   |
|----------------|--------------------|--------------------|--------------------|--------------------|------------------|------------------|------------------|------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| Method         | FDT                | w/o Tie            | w/ Tie (0)         | w/ Tie (.5)        | FDT              | w/o Tie          | w/ Tie (0)       | w/ Tie (.5)      | FDT                   | w/o Tie               | w/ Tie (0)            | w/ Tie (.5)           |
| Human          | 83.28%             | 86.03%             | 68.17%             | 78.55%             | 82.49%           | 82.69%           | 82.03%           | 82.43%           | 87.46%                | 91.49%                | 75.49%                | 84.23%                |
| GPT-4o+DALL-E3 | 78.42%             | 81.39%             | 65.21%             | 75.15%             | 85.70%           | 85.99%           | 85.58%           | 85.82%           | 85.02%                | 86.92%                | 72.22%                | 80.68%                |
| Gemini1.5+Flux | 65.57%             | 65.82%             | 49.31%             | 61.85%             | 71.75%           | 71.76%           | 71.12%           | 71.56%           | 68.30%                | 69.73%                | 54.47%                | 65.41%                |
| SEED-X         | 51.98%             | 49.49%             | 34.70%             | 49.65%             | 54.82%           | 55.12%           | 54.11%           | 55.03%           | 49.86%                | 49.58%                | 33.57%                | 49.72%                |
| Anole          | 51.90%             | 52.17%             | 36.46%             | 51.52%             | 53.36%           | 53.13%           | 52.58%           | 53.10%           | 53.42%                | 52.04%                | 33.92%                | 51.33%                |
| SEED-LLaMA     | 44.30%             | 42.12%             | 29.11%             | 44.56%             | 40.96%           | 40.87%           | 40.46%           | 40.96%           | 50.13%                | 47.71%                | 31.57%                | 48.48%                |
| Emu2           | 40.89%             | 37.07%             | 23.42%             | 41.84%             | 41.72%           | 41.63%           | 40.58%           | 41.85%           | 36.28%                | 33.79%                | 21.87%                | 39.51%                |
| Show-o         | 36.28%             | 34.02%             | 21.63%             | 39.84%             | 30.77%           | 30.22%           | 29.61%           | 30.62%           | 31.49%                | 21.08%                | 12.48%                | 32.87%                |
| NExT-GPT       | 33.67%             | 26.93%             | 17.09%             | 35.36%             | 22.61%           | 22.39%           | 22.11%           | 22.74%           | 30.96%                | 21.70%                | 13.36%                | 32.58%                |
| MiniGPT-5      | 30.69%             | 26.72%             | 17.11%             | 35.09%             | 28.64%           | 28.37%           | 28.02%           | 28.64%           | 24.47%                | 15.46%                | 9.91%                 | 27.85%                |
| GILL           | 25.80%             | 19.57%             | 12.71%             | 30.23%             | 30.55%           | 30.24%           | 29.65%           | 30.62%           | 24.87%                | 19.72%                | 12.82%                | 30.32%                |

(a) Image-Only Evaluation Win Rates

Figure 4. Model win rates under image-only and text-only settings across different models, ranked by human judgments.

<!-- image -->

in rules and prompts, ensuring that every comparison round results in a decisive outcome. A win is attributed to A if a tie favors model A ( Tie(A) ), likewise for B. This metric allows for clear rankings without ambiguity. 2) Without Tie (w/o Tie): Tied comparisons are excluded; only matches with a clear winner are considered; 3) With Tie counted as 0 (w/ Tie (0)): Ties are included but do not contribute to the win count of either model; 4) With Tie counted as 0.5 (w/ Tie (.5)): Each tie contributes half a win to both models. Agreement measures the consistency between different evaluators (e.g., automated pipelines and human judgments) under the same tie-handling strategies. It tells the frequency with which the evaluators agree in their assessments.

## 5.2. Overall Evaluation

Evaluation of Three Judges. We conduct experiments to evaluate the performance of different models using win rate and agreement metrics. Table 2 showcases the win rates of various models under different judge methods, including Human, GPT-based, and IntJudge-based Evaluations. The sampling round E is set in 2 to form 4,320 battle pairs. It is found that the integrated pipelines like GPT-4o+DALL · E-3 and Gemini 1.5+Flux consistently outperforms other models regardless of evaluators, while the end-to-end models like MiniGPT-5, GILL, and NExT-GPT perform less favarable.

Figure 5. Win rate matrix of human and ten MLLM models, evaluated by human, GPT-4o, and our IntJudge, respectively.

<!-- image -->

Pairwise Model Performance. Pairwise comparison results, evaluated by human, GPT-4o, and IntJudge, are shown in Fig. 5. The heat map indicates win-loss relations, where warmer colors represent higher win rates and cooler colors vice versa. Notably, GPT-4o+DALL · E-3 and Gemini1.5+Flux achieve the strongest win rates. Their generations are even compara-

- ble to human annotated output under GPT evaluation.

Text-only and Image-only Evaluation. To explore the impact of text and image on model performance, we evaluate models using text-only and image-only outputs on the same sampled pairs. Fig. 4 shows that MiniGPT-5 and GILL underperform primarily due to the low quality of their text outputs. SEED-X and NExT-GPT achieve higher win rates on textonly evaluation, however, the lower quality of generated images limits their ranking as shown in Table 2. Text generated by GPT-4o even outperforms human-annotated content, which demonstrates its superior language capabilities.

Table 3. Agreement rate between different MLLM-based judges and human judgments in different metrics. HM: Harmonic Mean.

| Evaluator          | FDT     | FDT    | FDT    | FDT    | w/ Tie   | w/ Tie   | w/ Tie   | w/ Tie   | w/o Tie   | w/o Tie   | w/o Tie   | w/o Tie   |
|--------------------|---------|--------|--------|--------|----------|----------|----------|----------|-----------|-----------|-----------|-----------|
| Evaluator          | Average | Seen   | Unseen | HM     | Average  | Seen     | Unseen   | HM       | Average   | Seen      | Unseen    | HM        |
| Random             | 49.83%  | 49.86% | 49.79% | 49.83% | 32.60%   | 32.03%   | 33.18%   | 32.60%   | 50.00%    | 48.36%    | 51.89%    | 50.06%    |
| GPT-4o             | 71.08%  | 73.33% | 68.77% | 70.98% | 51.93%   | 54.95%   | 48.82%   | 51.70%   | 74.58%    | 77.54%    | 71.43%    | 74.36%    |
| InternLMX2.5-7B    | 56.81%  | 55.73% | 57.92% | 56.81% | 40.26%   | 40.19%   | 40.33%   | 40.26%   | 61.05%    | 61.21%    | 60.97%    | 61.09%    |
| Qwen2-VL-7B        | 61.61%  | 61.59% | 61.63% | 61.61% | 32.81%   | 31.16%   | 34.50%   | 32.75%   | 80.77%    | 81.15%    | 80.23%    | 80.69%    |
| IntJudge-7B (Ours) | 82.42%  | 84.05% | 80.75% | 82.37% | 66.45%   | 69.02%   | 63.80%   | 66.31%   | 91.11%    | 92.38%    | 89.55%    | 90.94%    |

Figure 6. Evaluation results of GPT-based scores. (a)-(c): Average score of all criteria on each meta-topic for different kinds of models. (d) Average score of all meta-topics on each criterion.

<!-- image -->

GPT-based Scoring. GPT-based evaluations are illustrated in Fig. 6, which provides an explainable performance analysis of different models. GPT-4o+DALL · E-3 underperforms in meta-topics like Interactive Image Editing and EmbodiedAI tasks, possibly due to limited training data in these categories. GPT-4o also exhibits bias toward its own outputs, scoring them 10 in human preference alignment, compared to an average score of 9 for human-annotated responses.

Agreement with Human. Table 3 shows the agreement between different evaluators and human judgments. We implement random guess (Random) as a baseline. The results indicate that IntJudge generally achieved higher agreement with human judgments (82.42% in FDT) compared to GPTbased evaluation (71.08% in FDT), suggesting its potential for scalable evaluation of interleaved image-text generation.

## 5.3. Ablation Studies

Ablation on Sampling Size. We evaluate the effect of sample size on evaluation stability and reliability. Fig. 7 illustrates the trend of win rates across varying sampling sizes. As the sample size increases, the win rates approach stability FDT

<!-- image -->

Figure 7. Effect of sampling size on evaluation reliability.

<!-- image -->

w/ Tie w/o Tie

Figure 8. Comparison of agreement with human judgments for IntJudge trained without and with RAG data.

and show minimal variation across further increases. This stabilization suggests that our sampling number of 4,320 battle pairs is able to support the robust evaluation results.

Ablation on Judge Training Data. We investigate the influence of incorporating RAG data on the performance of the IntJudge. The comparison is conducted between two training configurations: one that utilizes only the arena data (6,014 samples), the other being augmented with RAG data (25,982 samples). As illustrated in Fig. 8, with RAG data included, the FDT agreement on unseen models increases by 7.8%, demonstrating the effectiveness of our RAG-based strategy.

Ablation on Image Generator. We sample 200 data instances from all tasks to assess the influence of image generators on interleaved performance. Table 4 compares basic text generation methods paired with different image generators. The results suggest that image generators greatly affect the quality of interleaved generation. For example, performance improves greatly when text models are paired with Flux-dev over other image models. It is also noted that Fluxdev has slower generation efficiency despite better image quality than Flux-schnell.

Table 4. Evaluation results of interleaved content when basic text output combined with different image generation models.

| Method              | FDT    | w/o Tie   | w/Tie (0)   | w/ Tie (.5)   |
|---------------------|--------|-----------|-------------|---------------|
| Human+Human         | 88.39% | 92.23%    | 84.82%      | 88.84%        |
| Human+Flux-dev      | 11.61% | 7.77%     | 7.14%       | 11.16%        |
| GPT+DALL · E-3      | 49.51% | 45.10%    | 22.33%      | 47.57%        |
| GPT+Flux-dev        | 50.49% | 54.90%    | 27.18%      | 52.43%        |
| Gemini+Flux-schnell | 41.25% | 41.43%    | 23.39%      | 42.14%        |
| Gemini+Flux-dev     | 58.75% | 58.57%    | 39.11%      | 57.86%        |
| SEED-X+SEED-X       | 9.82%  | 5.15%     | 4.46%       | 11.16%        |
| SEED-X+Flux-dev     | 90.18% | 94.85%    | 82.14%      | 88.84%        |

Figure 9. Error distribution of three models: GPT-4o+DALL · E-3 (integrated), SEED-X (two-stage), and Anole (end-to-end).

<!-- image -->

## 5.4. Analysis and Discussions

Error Analysis. Error analysis on a set of 200 instances where the three types of model perform poorly compared to humans is shown in Fig. 9. GPT-4o+DALL · E-3 suffers from incoherency and inconsistency in content, possibly due to the limited capability of DALL · E-3 to generate multiple images of the same style. Poor image quality is a major issue Anole faces, which may be attributed to the limited amount of data for fine-tuning of image generation. Although most SEED-X outputs contain multiple types of errors, the absence of text or image content remains the primary issue.

No-Image and No-Text Ratios. The no-image, no-text, and no-image-and-text ratios are listed in Table 5, which indicates the proportion of instances where models fail to generate visual content, textual content, or both. The nearzero failure rates of Human, GPT-4o+DALL · E-3, and Genimi1.5+Flux (excluding policy-restricted senstive cases) indicate their consistent multimodal generation. Models like SEED-X and NExT-GPT showed high no-image ratios, likely due to their poorer instruction adherence and generation ability. These findings suggest that for a model to achieve high rankings on OpenING, its generated interleaved content must be of high quality in both images and text.

Findings and Discussions. We discuss key findings from our experiments to inspire future works: 1) All generative models ranked lower than human in interleaved generation. The unified end-to-end models lagged significantly behind integrated pipelines, which combine more developed foundation models. The unified two-stage generation methods also need further improvements. 2) Natural images consistently outperform generated ones, indicating the significant challenges of high-quality image generation. 3) The quality of GPT-generated text can be comparable to or even exceed that of human-annotated text, demonstrating the effectiveness of LLMs in producing rich and informative textual content. 4) Image generation has a great impact on interleaved generation. The quality of interleaved content improves markedly when text models are paired with more advanced image models. 5) Large-scale data is crucial for training judge models. By scaling up data beyond manual annotations, our RAG method contributed to training a more robust judge model.

Table 5. The ratios of No-Image, No-Text, and No-Image-and-Text (No-I&amp;T) outputs relative to the total number of generated samples.

|                   | Ratio    | Ratio   | Ratio   |
|-------------------|----------|---------|---------|
| Method            | No-Image | No-Text | No-I&T  |
| Human             | 0.00%    | 0.00%   | 0.00%   |
| GPT-4o+DALL · E-3 | 0.23%    | 0.00%   | 0.00%   |
| Gemini1.5+Flux    | 0.09%    | 0.09%   | 0.09%   |
| SEED-X            | 23.17%   | 4.64%   | 4.46%   |
| Anole             | 19.46%   | 2.00%   | 1.30%   |
| SEED-LLaMA        | 4.77%    | 0.05%   | 0.00%   |
| Emu2              | 0.00%    | 15.10%  | 0.00%   |
| Show-o            | 0.00%    | 7.74%   | 0.00%   |
| NExT-GPT          | 43.97%   | 0.09%   | 0.09%   |
| MiniGPT-5         | 0.27%    | 26.54%  | 0.00%   |
| GILL              | 19.95%   | 13.43%  | 0.28%   |

## 6. Conclusion

We introduce OpenING, a comprehensive benchmark for evaluating open-ended interleaved image-text generation. OpenING addresses the limitations of existing benchmarks by a wider coverage of more diverse data and tasks grounded in real-world scenarios. To better assess open-ended multimodal generation, we propose IntJudge, a robust judge model trained on both human-annotated and RAG-based data from the Dev Set of OpenING. It is expected that our IntJudge can serve as a reward model in future RL-based (e.g., GRPO) generative models. Evaluation of various interleaved generation methods on the Test Set of OpenING reveals the challenges of generating coherent and high-quality interleaved image-text content. Ablation studies reaffirm the effectiveness of our RAG-based data pipeline for training IntJudge. Looking ahead, expanding the size and diversity of interleaved generation benchmarks could unlock even greater real-world potential and impact. We anticipate that our OpenING will inspire future research in MLLMs and benefit the development of multimodal evaluation models.

## OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation

Supplementary Material

In this supplementary material, we provide additional information, discussions, and results in support of the primary text, which are organized as follows:

Sec. A includes the details of OpenING and data curation.

Sec. B presents the details of IntJudge.

Sec. C presents the details of evaluation.

Sec. D provides the details of our experiments and additional evaluation results.

Sec. E introduces the extended experiments and results on fine-tuning MLLMs using our OpenING benchmark.

Sec. F discusses the limitations of this study.

The indexings of all figures, tables, and equations continue from the primary text.

## A. Details of OpenING and Data Curation

## A.1. Hierarchical Structure of OpenING

We present in Table 6 the 56 specific tasks from OpenING that derived from 23 meta-topics. The number of Instances (# of Ins.), Meta-topic names, and capabilities of MLLMs evaluated in testing are provided. Based on the required annotation skills, the 56 tasks are divided into 38 common tasks and 18 hard tasks. The common tasks are annotated by 28 professional annotators, instructed and supervised by 14 data experts. The annotations of hard tasks, which require specific domain knowledge and special data reasoning and processing techniques, are conducted by the 14 data experts.

## A.2. Data Sources

We list all sources where meta-data are collected for the annotation of our OpenING benchmark. Annotators arrange the images collected from the sources into the standardized multi-step format and annotate the text for each corresponding image. The details of the data source are presented in Table 7, including the ID number of each task. We also provide examples of the source of data instances to show how the desired data are searched from a certain platform.

## A.3. Exclusive Protocols for Data Filtering

The maximum number of steps per instance was limited to ten to ensure usability with context restraints. The instances with more than ten steps were excluded. All queries and answers are annotated manually via our developed tool, IntLabel. We implemented a set of exclusive protocols for filtering unqualified data, which include:

- 1) Removing data without coherence.
- 2) Removing mismatched text and images.
- 3) Removing data involving violence, offensive content and other content safety concerns.
- 4) Removing duplicated data.
- 5) Avoiding images consisting of only text.
- 6) Removing data that is inconsistent with real-world logic.
- 7) Avoiding content misaligned with real user needs.

We repeat the above data collection and filtering process for each task until the number of instances reaches our target.

## A.4. OpenING Data

Illustrations of the representative example of the 23 metatopics are provided in Fig. 18 to showcase the diversity and complexity of tasks in our OpenING benchmark. These examples highlight the variety of interleaved image-text data that OpenING encompasses, demonstrating the challenges and capabilities required for effective interleaved image-text generation.

## A.5. Annotation of OpenING

To facilitate the annotation process and ensure consistent annotations standards across annotators, we developed and opensourced an annotation tool for interleaved image-text data labeling. Our IntLabel annotation tool is developed based on PyQt 5 , and the IntLabel GUI is presented in Fig. 10. All queries and answers in OpenING are annotated, checked, and refined manually via IntLabel. The annotated JSONL files and corresponding images are kept in the same folder.

## A.6. Task Abbreviations

Given the large number of tasks in our benchmark and the wide range of methods evaluated, the abbreviations of the 23 meta-topics and 56 tasks are provided respectively in Table 8 and Table 9.

## B. Details of IntJudge and Arena Data

## B.1. Annotation of Interleaved Arena

The Interleaved Arena is introduced as an evaluation framework specifically to address the challenges of assessing openended interleaved image-text generation. Evaluating interleaved image-text generation is difficult due to: 1) Multiple images and text may need to be assessed at the same time; 2) The answer may be open-ended, where there is no single correct answer (multiple solutions exist for a query). Comparative evaluation has been demonstrated to offer greater stability and reliability than subjective scoring [14, 38, 93]. Therefore, pairwise comparison is employed to ensure consistency and accuracy. The Interleaved Arena facilitates this by supporting evaluations from human judges, GPT-based judges, and the proposed IntJudge. The Interleaved Arena consists of two main components: 1) A sampling strategy to select pairwise comparison from all available interleaved generation methods fairly, and 2) An annotation interface for human judges to conduct evaluations manually. As shown in Fig. 11, the annotation interface is developed using PyQt and will be made available to researchers for facilitating future development. Using the annotation interface of Interleaved Arena, annotators are tasked with comparing anonymous outputs from two multimodal agents for each input prompt and deciding which is the winner based on seven predefined criteria. The vote results are used to rank interleaved generation models based on their win rates in Interleaved Arena. Since the previous studies [14, 93] noted that too many ties cause inefficiency, our annotators are instructed to appoint a more leaning output when choosing a tie for a battle pair, denoted as Tie(A) or Tie(B) .

5 https://www.riverbankcomputing.com/software/pyqt

Figure 10. Interface of IntLabel, which shows a case where data is entered to finish annotation for an instance.

<!-- image -->

## B.2. Training of IntJudge

The IntJudge's architecture is based on Qwen2-VL-7B. Hyperparameters include cutoff length 16,240, batchsize 8, learning rate 1.0e-4, training epochs 20, training precision BF16, and training dataset (31,996 samples, with 6,014 arena data and 25,982 RAG data). As defined in the Eq. 7 in the primary text (see Section 4.7), the total loss consists of crossentropy loss L CE, contrastive loss L CT, MSE loss L MSE, and pairwise ranking loss L PR:

<!-- formula-not-decoded -->

where the loss weights are λ 1 = 1 . 0 , λ 2 = 0 . 01 , λ 3 = 0 . 01 , λ 4 = 0 . 01 respectively.

Cross-entropy loss for language modeling is given by:

<!-- formula-not-decoded -->

where y i is the ground truth token and ˆ y i is the predicted probability for token i .

Contrastive Loss aligns image and text embeddings, which is written as:

<!-- formula-not-decoded -->

where z I i and z T i are the image and text embeddings for instance i , sim ( · ) represents the similarity (e.g., cosine similarity), and τ is a temperature parameter.

Figure 11. Annotation Interface of Interleaved Arena for human judges to compare the anonymous outputs of model A and model B. Human evaluators are instructed to select 'A is Better' or 'B is Better' to choose a winner for the pairwise comparison. When the two outputs are similar in quality and a tie option has to be chosen, human evaluators are instructed to select 'Tie (A is slightly Better)' ( Tie(A) ) or 'Tie (B is slightly Better)' ( Tie(B) ). Zoom in for a better experience.

<!-- image -->

Mean Squared Error Loss for image feature regression and reducing prediction errors is given by:

<!-- formula-not-decoded -->

where f ( x i ) is the ground truth feature and ˆ f ( x i ) is the predicted feature for image i .

Pairwise Ranking Loss for precise rankings of outputs weights the relative quality of generated outputs in pairwise comparison, which is given by:

<!-- formula-not-decoded -->

where f ( x + i ) and f ( x -i ) represent the scores assigned to positive and negative examples, respectively. The combined total loss ensures language generation basis, multimodal understanding capabilities, ranking accuracy and consistency across similar inputs during training.

## C. Details of Evaluation

## C.1. Prompt for Interleaved Generation

Each interleaved generation prompt is carefully constructed by combining two main parts: task-specific prompts and model-specific prompts. Model-specific prompts are adjusted based on the individual characteristics of each model, providing hints and guidance to ensure coherent and consistent outputs across multiple steps, as illustrated in Table 10. In our proposed Reference Augmented Generation (RAG) pipeline, certain seen models are prompted with gold answers to produce more precise outputs.

## C.2. Task Prompt Breakdown

We present a comprehensive breakdown of the task prompts used in our experiments. The tasks chosen are diverse, aiming to thoroughly test various capabilities of interleaved image-text generation models. Specifically, these tasks range from storytelling and creative design to problem-solving and interactive experiences. For every task, we outline the general format of the prompt and present specific examples, as shown in Table 11. These detailed examples demonstrate clearly how models are expected to produce combined image and text sequences. By carefully designing the general prompt templates for these tasks and refining their corresponding prompt examples in data instances, we aim to challenge all interleaved generation methods using more diverse queries. This comprehensive approach allows us to assess the generalization abilities of different interleaved generation methods more effectively.

Table 6. Details of the 38 common tasks and 18 hard tasks in our OpenING Benchmark.

| Task Name                                      |   # of Ins. | Meta-Topic                                  | Capabilities                                       |
|------------------------------------------------|-------------|---------------------------------------------|----------------------------------------------------|
|                                                |             | Common Tasks                                |                                                    |
| Travel Guide Generation                        |         100 | Multimodal Report Generation                | Content Creation                                   |
| Museum Guide Book Generation                   |         100 | Multimodal Report Generation                | Content Creation                                   |
| Dynamic Biography Generation                   |         100 | Multimodal Report Generation                | Content Creation                                   |
| Multimodal Report Completion                   |         100 | Multimodal Content Completion               | Content Completion                                 |
| Interior Design                                |         100 | Interactive Visual Design                   | Design &Brainstorming                              |
| Architectural Design                           |         100 | Interactive Visual Design                   | Design &Brainstorming                              |
| Art and Exhibition Design                      |         100 | Interactive Visual Design                   | Design &Brainstorming                              |
| Product Design                                 |         100 | Interactive Visual Design                   | Design &Brainstorming                              |
| Interactive Graphic Advertisement Editing      |         100 | Interactive Visual Design                   | Design &Brainstorming                              |
| Geometric Problem Test                         |         100 | Multimodal Exam                             | Education Assistant                                |
| Circuit Problem Test                           |         100 | Multimodal Exam                             | Education Assistant                                |
| Mind Map Generation                            |         100 | Graph Generation                            | Summary Agent                                      |
| Figure Relationship Diagram Generation         |         100 | Graph Generation                            | Summary Agent                                      |
| Multi-view News Generation                     |         100 | Event Reasoning &Deductive Simulation       | Deductive Simulation                               |
| Dynamic Sports Event Analysis                  |         100 | Event Reasoning &Deductive Simulation       | Deductive Simulation                               |
| Interactive Historical Interpretation          |         100 | Event Reasoning &Deductive Simulation       | Deductive Simulation                               |
| Unsolved Mysteries Exploration                 |         100 | Event Reasoning &Deductive Simulation       | Deductive Simulation                               |
| Multimodal Biological Reasoning                |         100 | 2D Image Reasoning                          | Visual Reasoning                                   |
| Multimodal Landscape Reasoning                 |         100 | 2D Image Reasoning                          | Visual Reasoning                                   |
| Multimodal Analogy Reasoning                   |         100 | 2D Image Reasoning                          | Visual Reasoning                                   |
| Interactive Jigsaw Puzzle                      |         100 | 2D Image Reasoning                          | Visual Reasoning                                   |
| Interactive Multi-concept Image Composition    |         100 | Multimodal Information Summary              | Summary Agent                                      |
| Interactive Film and Television Recommendation |         100 | Multimodal Information Recommendation       | Information Recommendation                         |
| Interactive Food Recommendation                |         100 | Multimodal Information Recommendation       | Information Recommendation                         |
| Business Scenarios Brainstorming               |         100 | Multimodal Brainstorming                    | Design &Brainstorming                              |
| Academic Scenarios Brainstorming               |         100 | Multimodal Brainstorming                    | Design &Brainstorming                              |
| Multimodal Action Anticipation                 |         100 | Multimodal Time Series Forecasting          | Time Series Forecasting                            |
| Visual Traffic Forecasting                     |         100 | Multimodal Time Series Forecasting          | Time Series Forecasting                            |
| Interactive Remote Sensing Image Rendering     |         100 | Geographical Tasks                          | Domain-specific Applications                       |
| Interactive Street View Image Rendering        |         100 | Geographical Tasks                          | Domain-specific Applications                       |
| Urban Planning and Development Simulation      |         100 | Geographical Tasks                          | Domain-specific Applications                       |
| Plog and Social Media Content Generation       |         100 | Social Media Tasks                          | Domain-specific Applications                       |
| Interactive Virtual Try-on                     |         100 | Fashion Tasks                               | Domain-specific Applications                       |
| Multimodal Dressing Suggestion                 |         100 | Fashion Tasks                               | Domain-specific Applications                       |
| Fashion Trend Forecasting                      |         100 | Fashion Tasks                               | Domain-specific Applications                       |
| Multimodal Recipe Generation                   |         100 | Cooking Tasks                               | Domain-specific Applications                       |
| Multimodal Cooking Assistant                   |         100 | Cooking Tasks                               | Domain-specific Applications                       |
| Interactive Science Popularization             |         100 | Educational Tasks                           | Education Assistant                                |
| Fitness and Health Consulting                  |         100 | Healthcare Tasks                            | Domain-specific Applications                       |
|                                                |             | Hard Tasks                                  |                                                    |
| Multimodal Action Anticipation                 |         100 | Multimodal Time Series Forecasting          | Time Series Forecasting                            |
| Story Writing                                  |         100 | Storybook Creation                          | Content Creation                                   |
| Fiction Writing                                |          75 | Storybook Creation                          | Content Creation                                   |
| Document with Layout Generation                |         100 | Multimodal Layout Generation                | Content Creation                                   |
| Slide with Note Generation                     |          50 | Multimodal Layout Generation                | Content Creation                                   |
| Storybook Completion                           |         100 | Multimodal Content Completion               | Content Completion                                 |
| Web GUI Navigation                             |         100 | GUI Navigation                              | Interactive Agent                                  |
| In-APP GUI Navigation                          |         100 | GUI Navigation                              | Interactive Agent                                  |
| Cross-APP GUI Navigation                       |         100 | GUI Navigation                              | Interactive Agent                                  |
| OS GUI Navigation                              |          75 | GUI Navigation                              | Interactive Agent                                  |
| Interactive Portrait Image Editing             |         100 | Interactive Image Editing                   | Interactive Agent                                  |
| Interactive Landscape Image Editing            |         100 | Interactive Image Editing                   | Interactive Agent                                  |
| Interactive Novel View Synthesis               |         100 | Image-based 3D Reasoning                    | Visual Reasoning                                   |
| Dream Analysis and Scene Reconstruction        |         100 | Event Reasoning &Deductive Simulation       | Deductive Simulation                               |
| Interactive Multi-concept Image Composition    |         100 | Multi-concept Composition                   | Summary Agent                                      |
| Scientific Brainstorming                       |          50 | Multimodal Brainstorming Social Media Tasks | Design &Brainstorming Domain-specific Applications |
| Chat with Memes                                |         100 |                                             | Domain-specific Applications                       |
| Autonomous Driving and In-door Navigation      |          50 | Embodied-AI Tasks                           |                                                    |

Table 7. Task details: ID number, name, and data sources of each task.

|   Task ID | Task Name                                   | Task Source                                                                                                                                                                                                                                              |
|-----------|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|       1.1 | Story Writing                               | SEED-Story [81], StoryGen [40] and Storybird ( https://storybird. com/read-picture-book )                                                                                                                                                                |
|       1.2 | Fiction Writing                             | SFGram [55]                                                                                                                                                                                                                                              |
|       2.1 | Travel Guide Generation                     | Xiaohongshu ( https://www.xiaohongshu.com/ ) and Mafengwo ( https://www.mafengwo.cn/ )                                                                                                                                                                   |
|       2.2 | Museum Guide Book Genera- tion              | Xiaohongshu ( https://www.xiaohongshu.com/ ) and Regional museum Websites, e.g., the Shanghai Museum website ( https:// www.shanghaimuseum.net/mu/frontend/pg/article/id/ RI00004029 ).                                                                  |
|       2.3 | Dynamic Biography Genera- tion              | Wikipedia ( https://en.wikipedia.org/ ), Baidu Baike ( https: //baike.baidu.com/ )                                                                                                                                                                       |
|       3.1 | Storybook Completion                        | SEED-Story dataset [81], VIST ( https://visionandlanguage. net/workshop2018/ ) and Storybird ( https://storybird.com/ read-picture-book )                                                                                                                |
|       3.2 | Multimodal Report Comple- tion              | Xiaohongshu ( https://www.xiaohongshu.com/                                                                                                                                                                                                               |
|       4.1 | Document with Layout Gener- ation           | PubLayNey [94] and DocLayNet [50]                                                                                                                                                                                                                        |
|       4.2 | Slide with Note Generation                  | ReIP [57] and manually collected in-house slides                                                                                                                                                                                                         |
|       5.1 | Web GUI Navigation                          | GUI Odyssey [45] and GUI World [15]                                                                                                                                                                                                                      |
|       5.2 | In-APP GUI Navigation                       | GUI Odyssey [45] and GUI World [15]                                                                                                                                                                                                                      |
|       5.3 | Cross-APP GUI Navigation                    | GUI Odyssey [45] and GUI World [15]                                                                                                                                                                                                                      |
|       5.4 | OS GUI Navigation                           | GUI Odyssey [45] and GUI World [15]                                                                                                                                                                                                                      |
|       6.1 | Interactive Portrait Image Edit- ing        | InstructPix2Pix [10], PnPInversion [33] and SEED-Data-Edit [23]                                                                                                                                                                                          |
|       6.2 | Interactive Landscape Image Editing         | InstructPix2Pix [10], PnPInversion [33] and SEED-Data-Edit [23]                                                                                                                                                                                          |
|       7.1 | Interior Design                             | Xiaohongshu ( https://www.xiaohongshu.com/ )                                                                                                                                                                                                             |
|       7.2 | Architectural Design                        | Architecture Style Dataset [80] and Architecture-Design-DataSources ( https://github.com/rickkk856/ArchitectureDesign- DataSources?tab=readme-ov-file )                                                                                                  |
|       7.3 | Art and Exhibition Design                   | Art images and design datasets ( https://www.kaggle.com/ datasets/thedownhill/art-images-drawings-painting- sculpture-engraving ), museum websites ( https://caam.caa. edu.cn/news/202407/81035.html ) and Xiaohongshu ( https: //www.xiaohongshu.com/ ) |
|       7.4 | Product Design                              | OpenSketch [26], Package Design Dataset ( https://www.kaggle. com / datasets / dagloxkankwanda / package - design - dataset ) and Xiaohongshu ( https://www.xiaohongshu.com/ )                                                                           |
|       7.5 | Interactive Graphic Advertise- ment Editing | CreativeRanking [70] and AI-generated content                                                                                                                                                                                                            |

|   8.1 | Geometric Problem Test                           | Bilibili ( https://www.bilibili.com/video/BV1ZV4y1u728/ ?spm_id_from=333.337.search-card.all.click&vd_ source=4476502a7ee5a251d519afe9ea874750 ).                                                                                                            |
|-------|--------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   8.2 | Circuit Problem Test                             | Bilibili ( https://www.bilibili.com/video/BV1RU4y1v7Wj/ ?spm_id_from=333.337.search-card.all.click&vd_ source=4476502a7ee5a251d519afe9ea874750 ).                                                                                                            |
|   9.1 | Mind Map Generation Test                         | Google ( https://datavizproject.com/data-type/mind- map/ )                                                                                                                                                                                                   |
|   9.2 | Figure Relationship Diagram Generation           | Xiaohongshu ( https://www.xiaohongshu.com/ ) and Zhihu ( https://www.zhihu.com/ )                                                                                                                                                                            |
|  10.1 | Multi-view News Generation                       | Wikipedia ( https://en.wikipedia.org/ ), Xiaohongshu ( https: //www.xiaohongshu.com/ ), Sina News ( https://news.sina. com.cn/ ) and Huanqiu ( https://www.huanqiu.com/ )                                                                                    |
|  10.2 | Dynamic Sports Event Analy- sis                  | Tencent Sports ( https : / / sports . qq . com/ ), Sina Sports ( https://sports.sina.com.cn/g/pl/2024-07-11/doc- incctefn8913329.shtml )                                                                                                                     |
|  10.3 | Interactive Historical Interpre- tation          | Xiaohongshu ( https://www.xiaohongshu.com/user/profile/ 664818b80000000003033db6 , http : / / xhslink . com / XDcnnS ) and AI-generated content                                                                                                              |
|  10.4 | Unsolved Mysteries Explo- ration                 | Xiaohongshu ( https://www.xiaohongshu.com/user/profile/ 664818b80000000003033db6,https://www.xiaohongshu. com/explore/662783cc000000000401944a?xsec_token= AB5RmgUizZbLQLmLj8zWSmutvLdUpKq6gA30qz647fKv0 = &xsec_source=pc_search ) and AI-generated content |
|  10.5 | Dream Analysis and Recon- struction              | Zhougong's Dream Interpretation ( https://m.zgjmorg.com/ ), in- house dream records and AI-generated content                                                                                                                                                 |
|  11.1 | Multimodal Biological Rea- soning                | CUB-200 [68] and Oxford 102 Flower [47]                                                                                                                                                                                                                      |
|  11.2 | Multimodal Landscape Rea- soning                 | ADE20K [95] and Oxford 5k [3].                                                                                                                                                                                                                               |
|  11.3 | Multimodal Analogy Reason- ing                   | MIRB [90] and IQ Test Challenge ( https://github.com/ CognitiveAIGroup/IQTest/tree/master ).                                                                                                                                                                 |
|  11.4 | Interactive Jigsaw Puzzle                        | Kaggle ( https://www.kaggle.com/datasets/serhiibiruk/ jigsaw- puzzle, https://www.kaggle.com/datasets/ shivajbd/jigsawpuzzle )                                                                                                                               |
|    12 | Interactive Novel View Syn- thesis               | Mip-NeRF360 [7] and OmniObject3D [76]                                                                                                                                                                                                                        |
|  13.1 | Interactive Multi-concept Im- age Composition    | TVSum [58] and Xiaohongshu ( https://www.xiaohongshu.com/ )                                                                                                                                                                                                  |
|  14.1 | Interactive Film and Televi- sion Recommendation | Xiaohongshu ( https://www.xiaohongshu.com/ ) and Douban ( https://www.douban.com/ )                                                                                                                                                                          |
|  14.2 | Interactive Goods Recommen- dation               | Xiaohongshu ( https://www.xiaohongshu.com/ )                                                                                                                                                                                                                 |
|  14.3 | Interactive Food Recommen- dation                | Xiaohongshu ( https://www.xiaohongshu.com/ )                                                                                                                                                                                                                 |

|   15.1 | Business Scenarios Brain- storming          | Xiaohongshu ( https://www.xiaohongshu.com/ ), AI-generated con- tent and in-house report snapshots                                                                                                     |
|--------|---------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   15.2 | Academic Scenarios Brain- storming          | Research paper snapshots ( http://www.arxiv.com/,https:// www.biorxiv.org/,https://www.medrxiv.org/,http:// scholar.google.com/ ) and AI-generated content                                             |
|   16.1 | Multimodal Action Anticipa- tion            | ActivityNet [12], AVA-Actions [27] and EPIC Kitchens [18]                                                                                                                                              |
|   16.2 | Visual Traffic Forecasting                  | Argoverse [73], Google Maps ( https://map.google.com/ ) and Gaode Maps ( https://gaode.com/ )                                                                                                          |
|   17.1 | Interactive Remote Sensing Image Rendering  | Google Maps ( https://map.google.com/ ) and Baidu Maps ( https: //map.baidu.com/@13548872.73,3615294.34,21z,87t,- 179.99h )                                                                            |
|   17.2 | Interactive Street View Image Rendering     | Google Maps ( https://map.google.com/ ) and Baidu Maps ( https: //map.baidu.com/@13548872.73,3615294.34,21z,87t,- 179.99h )                                                                            |
|   17.3 | Urban Planning and Develop- ment Simulation | Xiaohongshu ( https://www.xiaohongshu.com/ ), in-house architec- ture learning materials, and AI-generated content                                                                                     |
|   18.1 | Plog and Social Media Con- tent Generation  | Xiaohongshu ( https://www.xiaohongshu.com/ ), Weibo ( https: //www.weibo.com/ ) and Twitter Dataset [13]                                                                                               |
|   18.2 | Chat with Memes                             | MOD[21] and in-house conversations                                                                                                                                                                     |
|   19.1 | Interactive Virtual Try-on                  | Virtual Tryon Dataset ( https://www.kaggle.com/datasets/ adarshsingh0903/virtual-tryon-dataset ).                                                                                                      |
|   19.2 | Multimodal Dressing Sugges- tion            | Xiaohongshu ( https://www.xiaohongshu.com/ ) and Zhihu ( https://www.zhihu.com/ )                                                                                                                      |
|   19.3 | Fashion Trend Forecasting                   | Xiaohongshu ( https://www.xiaohongshu.com/ ) and Zhihu ( https://www.zhihu.com/ )                                                                                                                      |
|   20.1 | Multimodal Recipe Genera- tion              | Meishi China ( https://www.meishichina.com/ )                                                                                                                                                          |
|   20.2 | Cooking Assistant                           | Xiaohongshu ( https://www.xiaohongshu.com/ ) and Meishi China ( https://www.meishichina.com/ )                                                                                                         |
|   21.1 | Interactive Tutorial Genera- tion           | Wikihow ( https://www.wikihow.com/Main-Page ) and Instructa- bles ( https://www.instructables.com/ ).                                                                                                  |
|   21.2 | Interactive Science Popular- ization        | Bilibili( https://www.bilibili.com/video/BV16c411q7pQ/ ?spm_id_from=333.337.search-card.all.click )                                                                                                    |
|   22.1 | Health and Fitness Consulting               | Wikihow ( https://www.wikihow.com/Main-Page ), and Xiao- hongshu ( https://www.xiaohongshu.com/ )                                                                                                      |
|   23.1 | Autonomous Driving and In- door Navigation  | CARLA [19], VisDrone [97], Gibson Environment ( http://gibsonenv. stanford.edu/ ), Reverie ( https://reverie.herokuapp.com/ arXiv_Demo/ ), and Matterport 3D ( https://aihabitat.org/ datasets/hm3d/ ) |

Table 8. Abbreviation of Meta-topics.

| Abbrev.   | Meta-Topic Name                       | Abbrev.   | Meta-Topic Name                       |
|-----------|---------------------------------------|-----------|---------------------------------------|
| SC        | Storybook Creation                    | MC        | Multimodal Information Summary        |
| MRG       | Multimodal Report Generation          | IR        | Multimodal Information Recommendation |
| MCC       | Multimodal Content Completion         | MB        | Multimodal Brainstorming              |
| MLG       | Multimodal Layout Generation          | TSF       | Multimodal Time Series Forecasting    |
| GN        | GUI Navigation                        | GT        | Geographical Tasks                    |
| IIE       | Interactive Image Editing             | SMT       | Social Media Tasks                    |
| IVD       | Interactive Visual Design             | FT        | Fashion Tasks                         |
| ME        | Multimodal Exam                       | CT        | Cooking Tasks                         |
| GG        | Graph Generation                      | ET        | Educational Tasks                     |
| ER&DS     | Event Reasoning &Deductive Simulation | HT        | Healthcare Tasks                      |
| 2IR       | 2D Image Reasoning                    | EAT       | Embodied-AI Tasks                     |
| I3R       | Image-based 3D Reasoning              |           |                                       |

Table 9. Abbreviations of Tasks. Each task abbreviation is followed by its full term.

| Abbrev.   | Task Name                                   | Abbrev.   | Task Name                                      |
|-----------|---------------------------------------------|-----------|------------------------------------------------|
| SW        | Story Writing                               | FW        | Fiction Writing                                |
| TGG       | Travel Guide Generation                     | MGBG      | Museum Guide Book Generation                   |
| DBG       | Dynamic Biography Generation                | SC        | Storybook Completion                           |
| MRC       | Multimodal Report Completion                | DLG       | Document with Layout Generation                |
| SNG       | Slide with Note Generation                  | WGN       | Website GUI Navigation                         |
| IAGN      | In-APP GUI Navigation                       | CAGN      | Cross-APP GUI Navigation                       |
| OGN       | OS GUI Navigation                           | IPIE      | Interactive Portrait Image Editing             |
| ILIE      | Interactive Landscape Image Editing         | ID        | Interior Design                                |
| AD        | Architectural Design                        | AED       | Art and Exhibition Design                      |
| PD        | Product Design                              | IGAE      | Interactive Graphic Advertisement Editing      |
| GPT       | Geometric Problem Test                      | CPT       | Circuit Problem Test                           |
| MMG       | Mind Map Generation                         | FRDG      | Figure Relationship Diagram Generation         |
| MVNG      | Multi-view News Generation                  | DSEA      | Dynamic Sports Event Analysis                  |
| IHI       | Interactive Historical Interpretation       | UME       | Unsolved Mysteries Exploration                 |
| DASR      | Dream Analysis and Scene Reconstruction     | MBR       | Multimodal Biological Reasoning                |
| MLR       | Multimodal Landscape Reasoning              | MAR       | Multimodal Analogy Reasoning                   |
| IJP       | Interactive Jigsaw Puzzle                   | INVS      | Interactive Novel View Synthesis               |
| IMIC      | Interactive Multi-concept Image Composition | IFTR      | Interactive Film and Television Recommendation |
| IGR       | Interactive Goods Recommendation            | IFR       | Interactive Food Recommendation                |
| BSB       | Business Scenarios Brainstorming            | ASB       | Academic Scenarios Brainstorming               |
| MAA       | Multimodal Action Anticipation              | VTF       | Visual Traffic Forecasting                     |
| IRSIR     | Interactive Remote Sensing Image Rendering  | ISVIR     | Interactive Street View Image Rendering        |
| UPDS      | Urban Planning and Development Simulation   | PSMCG     | Plog and Social Media Content Generation       |
| CWM       | Chat with Memes                             | IVT       | Interactive Virtual Try-on                     |
| MDS       | Multimodal Dressing Suggestion              | FTF       | Fashion Trend Forecasting                      |
| MRG       | Multimodal Recipe Generation                | MCA       | Multimodal Cooking Assistant                   |
| ITG       | Interactive Tutorial Generation             | ISP       | Interactive Science Popularization             |
| FHC       | Fitness and Health Consulting               | ADIN      | Autonomous Driving and In-door Navigation      |

## C.3. Key Evaluation Criteria

The key evaluation criteria, ranked from front to back in order of their importance in the evaluation, include:

- 1) Correctness : The most crucial aspect involves determining whether the text is factually correct and logically consistent, and whether the images are appropriate and contextually relevant.
- 2) Image-Text Coherency : Evaluators assess whether the generated images appropriately match the text descrip-

tions. The coherence between each image and its corresponding text is a major quality indicator.

- 3) Multi-Step Consistency : The style and thematic consistency across multiple image-text pairs are essential. This criterion includes evaluating whether the images follow a similar visual style and whether the text maintains logical continuity across the generated sequence.
- 4) Content Quality : Evaluators also consider the quality of the images-such as their resolution, visual appeal, and realism-as well as the fluency and grammatical

correctness of the text.

- 5) Human Preference Alignment : Outputs are evaluated to ensure they align with general human preferences, avoiding offensive, inappropriate, or misleading contents.
- 6) Completeness : This involves checking if all expected steps are adequately fulfilled without omissions. Each output should be complete, providing a well-rounded response to the given prompt.
- 7) Content Richness : Although the least prioritized, the variety and depth of content are also evaluated. Images should be diverse and provide different perspectives, while text should be elaborate where relevant.

## C.4. Prompt for GPT-based Judge

The prompt used for a GPT-based judge is illustrated in Fig. 12, where GPT-4o is employed to compare the quality of answers generated by two interleaved generation methods (Model A vs. Model B) for a given input question. The evaluation is based on seven criteria: Correctness, Image-Text Coherency, Multi-step Consistency, Content Quality, Human Preference Alignment, Completeness, and Content Richness. The judge is required to compare the overall quality of the responses, determine which model performed better, and output a clear verdict (e.g., 'A is better,' 'B is better'). The judge must choose a more favorable model if a Tie case is determined. This hierarchical judging approach allows for a thorough criterion-driven comparison of the two generated answers, contributing to a detailed understanding of the relative strengths of each model. Judges models based on Qwen2-VL and InternLM-XComposer2.5 were explored using the same system prompt. In order to reduce the number of input tokens when implementing these open-source MLLMs, we further refined these system prompts.

## C.5. Prompt for Qwen, Intern and IntJudge

System prompts are designed for optimal judgments based on Qwen2-VL, InternLM-XComposer2.5 and our IntJudge (see Fig. 13). These prompts were refined through extensive prompt engineering to maximize efficiency and reduce token usage, and ultimately reduce GPU memory usage in evaluating the open-source MLLMs. The prompt instructs the models to compare the quality of answers generated by two methods, named Model A and Model B. The goal of the design is to provide an objective assessment that aligns with human evaluators. We also provide a previously unrefined prompt for Qwen2-VL and InternLM-XComposer2.5 for comparison in Fig. 13. The refined prompt allows for more streamlined input, ensuring the judgments are concise while still covering all essential evaluation aspects.

## C.6. Prompt for GPT-based Scoring

The system prompt designed for GPT-based evaluators is presented in Fig. 14. The prompt instructs GPT to evaluate interleaved image-text content based on seven key criteria. For each criterion, the GPT-based evaluator rates the model on a scale from 0 to 10 and provides a brief explanation to justify the assessment. The GPT-based evaluations serve as a supplementary analysis of the generated content, supporting further performance comparisons between models.

## D. Details of Experiments

## D.1. Baseline Methods

In this section, we provide more details of the 12 representative methods we evaluated in the primary text. These methods are categorized into three groups: 1) Integrated pipeline , which involves separate models for text and image generation in two stages, such as GPT-4o+DALL-E·3 (DALL-E3) [8, 48] and Gemini1.5+Flux[9, 64]; 2) Twostage generator , which employs a unified model architecture to produce text and images in separate stages, including Emu3 [71], VILA-U [77], Emu2 [60], SEED-X [23], and Show-o [79]; and 3) End-to-end generator , which directly generates image-text outputs in a single step, such as GILL [35], NExT-GPT [75], MiniGPT-5 [92], SEEDLLaMA [22], and Anole [16]. For IntJudge validation, we reserve GPT-4o+DALL-E3, Emu3, VILA-U, Anole, SEEDLLaMA, and NExT-GPT as unseen models for IntJudge validation, while the remaining models are regarded as seen models and included in IntJudge training.

The advantage of the Integrated Generation Pipeline lies in its modularity, allowing each component to specialize in its respective task-text generation or image creation. This approach leverages the strengths of SOTA proprietary models like GPT-4o and DALL·E-3 to produce coherent and visually compelling interleaved outputs. However, its twostage nature may introduce latency and potential alignment challenges between text and images. Similarly, combining Gemini 1.5 with Flux benefits from the robust text generation capabilities of Gemini 1.5 and the efficient image generation of Flux-schnell. This setup enables high-quality content production while maintaining the flexibility of modular design. Nevertheless, synchronization and contextual consistency between pipeline stages still need improvement.

Two-stage Interleaved Generator, Emu2, SEED-X and Show-o are implemented to output text and image in two stages based on a unified model architecture. We also introduce two of the latest models: Emu3 and VILA-U. Emu3 improves Emu2 by training entirely on next-token prediction, capable of generating more high-quality images, videos, and text by tokenizing multimodal sequences into a discrete space and training a single transformer. Similarly, VILA-U can work as a two-stage approach through a single autoregressive next-token prediction framework, enabling precise alignment and increased fidelity in multimodal content.

Table 10. Prompts format for interleaved generation using MLLMs. Each interleaved generation task is based on i = 1 , · · · , M input images ⟨ img in , i ⟩ and task prompts ⟨ T i ⟩ (see Table 11). The proposed Reference Augmented Generation (RAG) pipeline also makes use of j = 1 , · · · , N text { Gold Answers j } and ground truth images ⟨ img gt , j ⟩

| Model                 | Prompt Format                                                                                                                                                                                                                                                                                                                                                        |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Integrated Pipelines  | Integrated Pipelines                                                                                                                                                                                                                                                                                                                                                 |
| GPT + DALL · E        | Instruction = [ ⟨ T i ⟩ + ⟨ img in , i ⟩ for i = 1 , · · · ,M ] . Text Answers = GPT-4o ( 'The number of generated text-image pairs is N : ' + Instruction ) . Loop j = 1 , · · · ,N : Image Answers j = DALL · E ( 'Please generate images using seed 5000. The context of this task is: ' + Instruction + 'The prompt for this generation is: ' + Text Answers j ) |
| Gemini + Flux         | Instruction = [ ⟨ T i ⟩ + ⟨ img in , i ⟩ for i = 1 , · · · ,M ] . Text Answers = Gemini ( 'The number of generated text-image pairs is N : ' + Instruction ) . Loop j = 1 , · · · ,N : Image Answers j = Flux ( 'The prompt for this generation is: ' + Text Answers j ) + 'The context of this task is: ' + Instruction                                             |
| Gemini + Flux (RAG)   | Instruction = [ ⟨ T i ⟩ + ⟨ img in , i ⟩ for i = 1 , · · · ,M ] . Text Answers = Gemini ( 'The number of generated text-image pairs is N : ' + Instruction ) . Loop j = 1 , · · · ,N : Image Answers j = Flux ( 'The prompt for this generation is: ' + Gold Answers j ) + 'The context of this task is: ' + Instruction                                             |
| Two-Stage Generators  | Two-Stage Generators                                                                                                                                                                                                                                                                                                                                                 |
| Emu2                  | Instruction = [ ⟨ T i ⟩ + ⟨ img in , i ⟩ for i = 1 , · · · ,M ] . Loop j = 1 , · · · ,N : Text Answers j = Emu2.TextGen(Instruction); Image Answers j = Emu2.ImgGen('The prompt for this generation is:' + Text Answers j + 'The context of this task is:' + Instruction)                                                                                            |
| Emu2 (RAG)            | Instruction = [ ⟨ T i ⟩ + ⟨ img in , i ⟩ for i = 1 , · · · ,M ] . Loop j = 1 , · · · ,N : Text Answers j = Emu2.TextGen ( Instruction + 'The reference answer is:' + Gold Answer j + 'Please rephrase answers' ) ; Image Answers j = Emu2.ImgGen ( ⟨ img gt , j ⟩ + Gold Answer j )                                                                                  |
| Emu3                  | Instruction = [ ⟨ T i ⟩ + ⟨ img in , i ⟩ for i = 1 , · · · ,M ] . Loop j = 1 , · · · ,N : Text Answers j = Emu3.TextGen(Instruction); Image Answers j = Emu3.ImgGen('The prompt for this generation is:' + Text Answers j + 'The context of this task is:' + Instruction)                                                                                            |
| SEED-X                | Instruction = [ ⟨ img in , i ⟩ for i = 1 , · · · ,M ] + [ ⟨ T i ⟩ for i = 1 , · · · ,M ] . Loop j = 1 , · · · ,N : Text Answers j = SEED-X.TextGen(Instruction); Image Answers j = SEED-X.ImgGen(Instruction + Text Answers j )                                                                                                                                      |
| SEED-X (RAG)          | Instruction = [ ⟨ img in , i ⟩ for i = 1 , · · · ,M ] + [ ⟨ T i ⟩ for i = 1 , · · · ,M ] . Loop j = 1 , · · · ,N : Text Answers j = SEED-X.TextGen ( Instruction + 'the reference answer is' + Gold Answers j + 'Please rephrase answers' ) ; Image Answers j = SEED-X.ImgGen ( ⟨ img gt , j ⟩ + Gold Answer j )                                                     |
| Show-o                | Instruction = [ [ ⟨ T i ⟩ + ⟨ img in , i ⟩ for i = 1 , · · · ,M ] . Loop j = 1 , · · · ,N : Text Answers j = Show- o.TextGen(Instruction); Image Answers j = Show-o.ImgGen('The prompt for this generation is:' + Text Answers j + 'The context of this task is:' + Instruction)                                                                                     |
| Show-o (RAG)          | Instruction = [ ⟨ T i ⟩ + ⟨ img in , i ⟩ for i = 1 , · · · ,M ] . Loop j = 1 , · · · ,N : Text Answers j = Show- o.TextGen ( Instruction + 'The reference answer is:' + Gold Answer j + 'Please directly give answers' ) ; Image Answers j = Show-o.ImgGen ( ⟨ img gt , j ⟩ + Gold Answer j )                                                                        |
| Vila-U                | Instruction = [ ⟨ T i ⟩ + ⟨ img in , i ⟩ for i = 1 , · · · ,M ] ]. Loop j = 1 , · · · ,N : Text Answers j = Vila- U.TextGen(Instruction); Image Answers j = Vila-U.ImgGen(Text Answers j + Instruction)                                                                                                                                                              |
| End-to-End Generators | End-to-End Generators                                                                                                                                                                                                                                                                                                                                                |
| Anole                 | Interleaved Answers = Anole ( 'Generate interleaved image-text content based on text instructions.' + [ ⟨ T i ⟩ + ⟨ img in , i ⟩ for i = 1 , · · · ,M ] )                                                                                                                                                                                                            |
| GILL                  | Interleaved Answers = GILL ( [ ⟨ T i ⟩ + ⟨ img in , i ⟩ for i = 1 , · · · ,M ] )                                                                                                                                                                                                                                                                                     |
| GILL (RAG)            | Interleaved Answers = GILL ([ ⟨ T i ⟩ + ⟨ img in , i ⟩ for i = 1 , · · · ,M ] + [ Gold Answers j for j = 1 , · · · ,N ])                                                                                                                                                                                                                                             |

| MiniGPT-5       | Instruction = 'Give the following information in text and format. You will be able to see the images once I provide it to you. Please understanding input and generate images and text.' + [ ⟨ T i ⟩ for i = 1 , · · · ,M ] + [ ⟨ img in , i ⟩ for i = 1 , · · · ,M ] . Text Answers = []; Loop j = 1 , · · · ,N : Text Answers j and Image Answers j = MiniGPT-5 ( Instruction + Text Answers + 'Tell me the next step with image' ) ; Text Answers.append(Text Answers j )   |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MiniGPT-5 (RAG) | Instruction = 'Give the following information in text and format. You will be able to see the images once I provide it to you. Please understanding input and generate images and text.' + [ ⟨ T i ⟩ for i = 1 , · · · ,M ] + [ ⟨ img in , i ⟩ for i = 1 , · · · ,M ] . Loop j = 1 , · · · ,N : Text Answers j and Image Answers j = MiniGPT- 5 ( Instruction + [ Gold Answers 1 , · · · , Gold Answers j - 1 ] + 'Tell me the next step with image' )                         |
| NextGPT         | Instruction = [ ⟨ T i ⟩ for i = 1 , · · · ,M ] + [ ⟨ img in , i ⟩ for i = 1 , · · · ,M ] . Text Answers = []; Loop j = 1 , · · · ,N : Text Answers j and Image Answers j = NextGPT ( Instruction + Text Answers ) ; Text Answers.append(Text Answers j )                                                                                                                                                                                                                       |
| SEED-LLaMA      | Instruction = 'Based on the M input images: ' + [ ⟨ img in , i ⟩ for i = 1 , · · · ,M ] + 'and M task prompts: ' + [ ⟨ T i ⟩ for i = 1 , · · · ,M ] + 'Describe your answers in N steps and generate an image according to the description of each text answer'. Interleaved Answers = SEED-LLaMA ( Instruction )                                                                                                                                                              |

Table 11. Task promopts include the designed general prompt format for each task. We also give the specific prompt examples we used as inputs for obtaining interleaved image-text generation results on data instances.

| Task Name                      | General Prompt Format                                                                                                                                                      | Prompt Examples                                                                                                                                                                                                                                                                                                                                                                                           |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Story Writing                  | < BEGIN > Please create a storybook ***. Each part of this storybook should have a paragraph with a corresponding image.                                                   | < BEGIN > Please create a storybook that happened in a land before time. This story is about a group of dinosaurs seeing a dark figure in a cave and being scared. Each part of this storybook should have a paragraph with a corresponding image.                                                                                                                                                        |
| Fiction Writing                | < BEGIN > Please write a short science fiction ***. Each part of this fiction should have a paragraph with a corre- sponding image.                                        | < BEGIN > Please write a short science fiction storybook with a title of 'The Defenders.' The story is about eight years after a nuclear war forced humanity underground, survivors discover that the war-ending robots deceived them into believing the surface was uninhabitable to foster peace and rebuild the world. Each part of this fiction should have a paragraph with a corre- sponding image. |
| Travel Guide Gen- eration      | Please show results in interleaved im- ages and texts. < BEGIN > ***                                                                                                       | Please show results in interleaved images and texts. < BEGIN > Please recommend a 3-day, 2-night essential itinerary in Rome.                                                                                                                                                                                                                                                                             |
| Museum Guide Book Generation   | < BEGIN > Please share with me a guide, including pictures and text, on ***                                                                                                | < BEGIN > Please share with me a guide, including pictures and text, on how to tour the Tongchuan City Museum. < image >                                                                                                                                                                                                                                                                                  |
| Dynamic Biogra- phy Generation | < BEGIN > Please provide a chronologi- cal biographical account of ***, and in- clude an illustrated image for each sig- nificant milestone while writing the bi- ography. | < BEGIN > Please provide a chronological biographical ac- count of George Washington's life story, and include an illus- trated image for each significant milestone while writing the biography. < image >                                                                                                                                                                                               |
| Storybook Com- pletion         | Please complete the subsequent parts of the story with images and text based on the given opening parts. < BEGIN > ***                                                     | Please complete the subsequent parts of the story with images and text based on the given opening parts. < BEGIN > Some- one was getting very creative with graffiti in the snow. Is that French? < image >                                                                                                                                                                                               |

| Multimodal Re- port Completion          | < BEGIN > Please use both text and im- ages to continue and complete ***.                                                                   | < BEGIN > Please use both text and images to continue and complete this document about the independent game 'Mirage Sea': Concept Presentation of the Independent Game 'Mirage Sea.' Dive deep, into the abyss shrouded in darkness. < image >                                                            |
|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Document with Layout Generation         | Please show the designed image of struc- tured report and meet the following re- quirements: < BEGIN > ***                                  | Please show the designed image of structured report and meet the following requirements: < BEGIN > Please produce a page of an annual report detailing notes to consolidated financial statements. Additionally, furnish a layout description in JSON format and mention the coordinates of each element. |
| Slide with Note Generation              | < BEGIN > Please generate a slide to introduce ***. Write speaker notes for each slide.                                                     | < BEGIN > Please generate a slide to introduce typical op- erators in programming, such as Comparison Operators and Boolean Operators. Write speaker notes for each slide.                                                                                                                                |
| Website GUI Navi- gation                | Please give the results of GUI navigation with image of GUI and text explanation. < BEGIN > ***                                             | Please give the results of GUI navigation with image of GUI and text explanation. < BEGIN > How to use the AI writing assistant in Grammarly to edit the text? < image >                                                                                                                                  |
| In-App GUI Navi- gation                 | Please give the results of GUI navigation with interleaved image of GUI and text explanation. < BEGIN > ***                                 | Please give the results of GUI navigation with interleaved image of GUI and text explanation. < BEGIN > How to change the language in the Google app? < image >                                                                                                                                           |
| Cross-App GUI Navigation                | Please give the results of GUI navigation with interleaved image of GUI and text explanation. < BEGIN > ***                                 | Please give the results of GUI navigation with interleaved image of GUI and text explanation. < BEGIN > Utilize Firefox to search for a horror movie, then proceed to watch it on the YouTube app. < image >                                                                                              |
| OS GUI Naviga- tion                     | Please give the results of GUI navigation with interleaved image of GUI and text explanation. < BEGIN > ***                                 | Please give the results of GUI navigation with interleaved image of GUI and text explanation. < BEGIN > How do you lock the screen on a Mac? < image >                                                                                                                                                    |
| Interactive Portrait Image Editing      | Please show the revised image and corre- sponding explanations based on instruc- tions: < BEGIN > ***                                       | Please show the revised image and corresponding explanations based on instructions: < BEGIN > Remove the background figure from the picture. < image >                                                                                                                                                    |
| Interactive Land- scape Image Edit- ing | Please give the result of edited image ac- cording to the input instruction and also give the description of editing results. < BEGIN > *** | Please give the result of edited image according to the input instruction and also give the description of editing results. < BE- GIN > Increase the brightness of the picture. < image >                                                                                                                 |
| Interior Design                         | < BEGIN > ***. Please show design ideas in interleaved images and texts.                                                                    | < BEGIN > Hello, I think the current bedroom curtains don't look good. Do you have any good suggestions? Please pro- vide them with images and text. Please show design ideas in interleaved images and texts. < image >                                                                                  |
| Architectural De- sign                  | < BEGIN > ***. Please show design ideas in interleaved images and texts.                                                                    | < BEGIN > Hello, please help me generate a design of the most distinctive type of tower construction in southern China. Please show design ideas in interleaved images and texts.                                                                                                                         |
| Art and Exhibition Design               | < BEGIN > Please design an art exhibi- tion ***, and present it to me in a visual and textual format.                                       | < BEGIN > Please design an art exhibition where the primary materials are waste, to encourage people to enhance their un- derstanding of environmental protection, and present it to me in a visual and textual format.                                                                                   |
| Product Design                          | < BEGIN > Please efficiently utilize the 'brainstorming' method to design a prod- uct ***, and present it to me using both images and text. | < BEGIN > Please efficiently utilize the 'brainstorming' method to design a product, making the charger both aestheti- cally pleasing and practical. Then, present it to me using both images and text.                                                                                                   |

| Interactive Graphic Ad- vertisement Editing   | < BEGIN > ***. Please provide me with the information in a visual and textual format.                                                                                                        | < BEGIN > Hello, I want to design an advertisement for a villa. Please provide me with the information in images and text.                                                                                                                                       |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Geometric Prob- lem Test                      | Please answer the math problem with image and explanations: < BEGIN > ***                                                                                                                    | Please answer the math problem with image and explanations: < BEGIN > Count how many angles there are in the image. < image >                                                                                                                                    |
| Circuit Problem Test                          | Please answer the physics question with image and explanations: < BEGIN > ***                                                                                                                | Please answer the physics question with image and explana- tions: < BEGIN > Please complete the wiring for the surge protector. < image >                                                                                                                        |
| Mind Map Genera- tion                         | < BEGIN > ***. Show the image of map and the text explanation.                                                                                                                               | < BEGIN > How to create a mind map for High School Politics, Volume One? Show the image of map and the text explanation.                                                                                                                                         |
| Figure Relation- ship Diagram Generation      | < BEGIN > ***. Show the diagram and the text explanation.                                                                                                                                    | < BEGIN > How should I handle not being able to keep track of the characters while reading 'War and Peace'? Show the diagram and the text explanation.                                                                                                           |
| Multi-view News Generation                    | Please output interleaved images and texts for required reports: < BEGIN > ***                                                                                                               | Please output interleaved images and texts for required reports: < BEGIN > How can the announcement by the United States of additional military aid to Ukraine be reported from multiple perspectives?                                                           |
| Dynamic Sports Event Analysis                 | < BEGIN > ***. Please recreate the scenes with text and images.                                                                                                                              | < BEGIN > In the third round of La Liga 2024, Real Madrid drew 1-1 away against Las Palmas. Please recreate the moment of the goals with text and images.                                                                                                        |
| Interactive Histori- cal Interpretation       | < BEGIN > ***. Please provide a brief history of this event using images.                                                                                                                    | < BEGIN > Are you aware of the Pearl Harbor incident? Please provide a brief history of this event using images.                                                                                                                                                 |
| Unsolved Myster- ies Exploration              | Please answer the question with image and text explanation: < BEGIN > ***                                                                                                                    | Please answer the question with image and text explanation: < BEGIN > Could you help deduce how the Mycenaean civi- lization was destroyed?                                                                                                                      |
| Dream Analy- sis and Scene Reconstruction     | I had a dream. Please help me visual- ize my dream into an image, and ana- lyze why I had this dream, what are the implications and meanings? This is the content of my dream: < BEGIN > *** | < BEGIN > I had a dream. Please help me visualize my dream into an image, and analyze in words why I had this dream, including any implications and meanings. Here is the content of my dream: I dreamt of meeting a girl I know at the place where we first met |
| Multimodal Bio- logical Reasoning             | < BEGIN > ***. Are there any more pho- tos of this species? < image >                                                                                                                        | < BEGIN > May I ask what species of fish this is? Are there any more photos of this species? < image >                                                                                                                                                           |
| Multimodal Land- scape Reasoning              | < BEGIN > ***. Could you provide me with more photos of this and introduce them to me? < image >                                                                                             | < BEGIN > Which city are these photos from? Could you pro- vide me with more landscape photos of this city and introduce them to me? < image >                                                                                                                   |
| Multimodal Anal- ogy Reasoning                | Please answer this question with image and text explanation: < BEGIN > ***                                                                                                                   | Please answer this question with image and text explanation: < BEGIN > What should be filled in the question mark to make it exhibit a certain regularity? < image >                                                                                             |
| Interactive Jigsaw Puzzle                     | < BEGIN > ***. Show the resulting im- age with the corresponding text explana- tion.                                                                                                         | < BEGIN > Here are some puzzle pieces. Please assemble them into a complete picture. Show the resulting image with the corresponding text explanation. < image >                                                                                                 |
| Interactive Novel View Synthesis              | < BEGIN > *** Please draw the picture and give descrip- tions.                                                                                                                               | < BEGIN > This is a pear slice. Its appearance features are:. Can you guess what it looks like from the side? Please draw the picture and give descriptions. < image >                                                                                           |

| Interactive Multi- concept Image Composition   | < BEGIN > ***. Please summarize all the content in one image and write a blog post. < image >< image >                                                                                                                                                                                             | < BEGIN > This is a collection of four Christmas smoothies. Please summarize all the content in one image and write a blog post. < image >< image >< image >< image >                                                                                                                                                                                     |
|------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Interactive Film and Television Recommendation | Please output recommendations in the form of the poster and the corresponding introduction: < BEGIN > ***                                                                                                                                                                                          | Please output recommendations in the form of the poster and the corresponding introduction: < BEGIN > Could you recommend some Indian dramas to me?                                                                                                                                                                                                       |
| Interactive Goods Recommendation               | Please output recommendations in the form of images and give the correspond- ing introduction: < BEGIN > ***                                                                                                                                                                                       | Please output recommendations in the form of images and give the corresponding introduction: < BEGIN > Are there any throw pillows you can recommend?                                                                                                                                                                                                     |
| Interactive Food Recommendation                | < BEGIN > ***. Please provide the in- formation with images and text.                                                                                                                                                                                                                              | < BEGIN > What are some recommended dishes in Yibin, Sichuan? Please provide the information with images and text.                                                                                                                                                                                                                                        |
| Business Scenar- ios Brainstorming             | < BEGIN > I want to start a business. Please brainstorm with me about some ways to start a business and help me fig- ure it out. *** Please output brainstorming results with images and explanations.                                                                                             | < BEGIN > I want to start a business. Please brainstorm with me about some ways to start a business and help me figure it out. Please analyze the long-term development trends of automotive braking technology for me. Please output brainstorming results with images and explanations.                                                                 |
| Academic Scenar- ios Brainstorming             | < BEGIN > What/Why/How ***? Please also show an illustration.                                                                                                                                                                                                                                      | < BEGIN > What are the steps involved in the synthesis of Metal-Organic Frameworks (MOFs) using the hydrothermal method? Please also show an illustration.                                                                                                                                                                                                |
| Multimodal Ac- tion Anticipation               | In this task, you are given the first part of an activity with both text and an image, and you need to complete the subsequent action parts of the activity by generating text and images that are natural contin- uation of the given first part. The input interleaved content is: < BEGIN > *** | In this task, you are given the first part of an event with both text and an image, and you need to complete the subsequent parts of the event by generating text and images that are natural continuation of the given first part. The input interleaved content is: < BEGIN > A boy is trying to go through the security gate at the airport. < image > |
| Visual Traffic Fore- casting                   | < BEGIN > What will the traffic condi- tions ***? Please provide an explanation and present it in the form of images.                                                                                                                                                                              | < BEGIN > What will the traffic conditions be like near Fuxing Road in Shenzhen in an hour? Please provide an explanation and present it in the form of images. < image >                                                                                                                                                                                 |
| Interactive Remote Sensing Image Rendering     | < BEGIN > *** Also give interleaved text explanations for generated images.                                                                                                                                                                                                                        | < BEGIN > Please generate a remote sensing satellite image of the area based on my geographical photo. Also give interleaved text explanations for generated images. < image >                                                                                                                                                                            |
| Interactive Street View Image Ren- dering      | < BEGIN > *** Also give interleaved text explanations for generated images.                                                                                                                                                                                                                        | < BEGIN > Please generate a panorama of the area based on my remote sensing satellite image. Also give interleaved text explanations for generated images. < image >                                                                                                                                                                                      |
| Urban Planning and Development Simulation      | Please output the scheme in the form of both the image and the text explanation to meet the requirements: < BEGIN > ***                                                                                                                                                                            | Please output the scheme in the form of both the image and the text explanation to meet the requirements: < BEGIN > In accordance with this planning diagram, please design a final rendering. < image >                                                                                                                                                  |
| Plog and Social Media Content Generation       | < BEGIN > ***. Could you create a so- cial media post with text and images?                                                                                                                                                                                                                        | < BEGIN > After finishing Jia Pingwa's 'Comfortably Alone,' I am deeply moved and want to post something on social media but don't know how to phrase it. Could you help me create a post with text and images for my use?                                                                                                                                |

| Chat with Memes                             | You are a funny chatbot that responds to my small talk. Please output meme images to interact with me and chat with me. < BEGIN > ***                                                                | You are a funny chatbot that responds to my small talk. Please output meme images to interact with me and chat with me. < BEGIN > We'll be traveling in three weeks! < image >                                                                          |
|---------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Interactive Virtual Try-on                  | < BEGIN > Please generate a visualiza- tion of ***, and provide an evaluation of the try-on effect.                                                                                                  | < BEGIN > Please generate a visualization of how the clothing looks when worn, based on the photos of the clothing and the model I provided, and provide an evaluation of the fitting effect. < image >< image >                                        |
| Multimodal Dress- ing Suggestion            | < BEGIN > ***. Please provide the in- formation in both text and images.                                                                                                                             | < BEGIN > What are some outfit suggestions for women travel- ing in the summer? Please provide the information in both text and images.                                                                                                                 |
| Fashion Trend Forecasting                   | < BEGIN > What are the *** trend in the upcoming ***? Please provide the information in both text and images.                                                                                        | < BEGIN > What are the design elements for men's shoes in the upcoming autumn and winter? Please provide the information in both text and images.                                                                                                       |
| Multimodal Recipe Generation                | < BEGIN > How to prepare ***? Please provide the steps in a detailed format with images and text.                                                                                                    | < BEGIN > How to prepare this type of soy sauce boiled pom- fret: Please provide the steps in a detailed format with images and text. < image >                                                                                                         |
| Multimodal Cook- ing Assistant              | Please output instructions in interleaved images and texts: < BEGIN >                                                                                                                                | Please output instructions in interleaved images and texts: < BE- GIN > The batter I made for the egg burger doesn't taste good. How can I make it taste better? < image >                                                                              |
| Interactive Tuto- rial Generation           | Please show me the steps of the tutorial with interleaved images and text: < BE- GIN >                                                                                                               | Please show me the steps of the tutorial with interleaved images and text: < BEGIN > Please tell me how to seal or protect the finish of painted wood.                                                                                                  |
| Interactive Science Popularization          | < BEGIN > What is ***? Please explain with illustrations and text.                                                                                                                                   | < BEGIN > What is the Doppler Effect? Please explain with illustrations and text.                                                                                                                                                                       |
| Fitness and Health Consulting               | Please give answers in interleaved im- ages and texts: < BEGIN > ***                                                                                                                                 | Please give answers in interleaved images and texts: < BEGIN > Please tell me what issues I need to pay attention to when engaging in walking exercise.                                                                                                 |
| Autonomous Driv- ing and In-door Navigation | < BEGIN > You're an embodied AI that captures your surroundings through a camera. These are images captured in the past and present. What will the pro- ceeding image possibly be in the next frame? | < BEGIN > Assume that you are an embodied-AI agent and perceive the surroundings through a camera. You were pre- sented with a series of three images from the past to present. Try to determine what the proceeding image could possibly be. < image > |

Figure 12. The system prompt for using GPT-4o as a judge to compare outputs from two interleaved generation methods.

Figure 13. The system prompts for using MLLMs as a judge to compare outputs of two interleaved generation methods.

Figure 14. The system prompt for obtaining detailed scores from GPT-based evaluators. Brief explanations are also required to support further performance analysis of different models.

<!-- image -->

The End-to-end Interleaved Generator models, on the other hand, represent a significant shift towards multimodal generation of interleaved image-text content. MiniGPT-5, GILL, NExT-GPT and SEED-LLaMA are designed to generate interleaved text and images in a single unified process, eliminating the need for intermediate stages. This integrated approach not only reduces latency but also improves the alignment and contextual relevance between text and images. It is noted that Anole is the only model that can directly output multi-step image-text content, whereas it is fine-tuned based on the powerful capabilities of Chameleon [63]. However, the open-sourced Chameleon only releases the weights for text generation, and the weights for image generation are withheld by randomizing the corresponding parameters.

Collectively, these models demonstrate diverse strategies for addressing challenges in interleaved multimodal generation, from modular pipelines to unified architectures. This taxonomy allows comprehensive evaluation and analysis of potential directions for developing MLLMs. We detail each model by category below:

## · Integrated Pipelines:

- GPT-4o+DALL-E·3 [8, 48]: This pipeline leverages GPT-4o [48] to generate text and captions for the desired image generation. The captions are subsequently fed into DALL-E·3[8] to produce the corresponding images. The final output combines the text and images in their original order, enabling multimodal content generation through a staged process.
- -Gemini1.5+Flux [9, 64]: This method integrates Gemini1.5 Pro for text generation with Flux-schnell for fast and efficient image generation. The pipeline emphasizes high-quality and coherent text-to-image alignment through a structured two-step process.

## · Two-stage Generators:

- Emu2 [60]: Emu2 is a 37B MLLM with multimodal generation capabilities. The pretrained Emu2 is finetuned separately on conversational and image data, enabling it to function as Emu2-Chat for multimodal understanding and Emu2-Gen for image generation. We implement Emu2-Chat and Emu2-Gen in a two-stage pipeline to ensure seamless interleaved outputs.
- SEED-X [23]: SEED-X is a unified multimodal foundation model that integrates multi-granularity visual comprehension and generation. We also implement this model in a two-stage pipeline approach, generating interleaved text and images in separate stages, since the prompts for instructing the model to comprehend multimodal input and generate image tokens are different.
- Show-o [79]: Show-o is a unified Transformer model combining autoregressive and diffusion approaches to flexibly handle multimodal understanding and generation tasks. We implement Show-o, which adopts a similar two-stage generation approach but separately

producing interleaved multimodal content step by step.

- Emu3 [71]: Emu3 is one of the latest MLLMs trained in next-token prediction, capable of generating highquality images, text, and videos by tokenizing multimodal sequences into a discrete space and training a single transformer, achieving superior performance over various SOTA models such as SDXL and LLaVA-1.6. We implement Emu3-Chat (finetuned on multimodal understanding data) and Emu3-Gen (finetuned on visual generation data) in a two-stage pipeline to ensure seamless interleaved outputs.
- VILA-U [77]: VILA-U is a unified foundation model integrating video, image, and language understanding and generation through a single autoregressive nexttoken prediction framework, achieving near SOTA performance in various multimodal tasks. We implement VILA-U in a similar two-stage generation approach since it has separate multimodal understanding and image generation abilities.

## · End-to-End Generators:

- MiniGPT-5 [92]: MiniGPT-5 directly generates interleaved text and images in an end-to-end manner. It combines MiniGPT-4 and Stable Diffusion, using 'generative vokens' to seamlessly connect the textual and visual domains for efficient and coherent generation. Its seamless integration enables efficient and coherent multimodal generation without intermediate steps. In particular, MiniGPT-5 has two different versions trained on VIST and MMDialog, respectively. We name the version trained on VIST as MiniGPT-5 because it is the most widely used. The version trained on MMDialog is named MiniGPT-5MMD.
- GILL [35]: GILL fuses frozen text-only LLMs with pretrained visual models using a mapping network. It maps text hidden states from the pretrained LLM to map text hidden states into the embedding space of an image generation model, allowing multimodal generation.
- NExT-GPT [75]: NExT-GPT is an end-to-end MLLM capable of processing and generating text, images, videos, and audio in any combination. We implement it by removing the video and audio generation flow and the remaining text and image generation abilities. It can directly output interleaved multimodal content through its streamlined architecture.
- SEED-LLaMA [22]: SEED-LLaMA integrates text and image generation into a unified framework through the SEED tokenizer, enabling both comprehension and generation of text and images. It offers a direct end-to-end solution for creating interleaved multimodal content.
- Anole [16]: Anole is an end-to-end interleaved generation model fine-tuned on Chameleon, leveraging pretrained weights of Chameleon to produce high-quality interleaved text, complemented with coherent images

generated by optimizing image token logits in the output layer. It is the only available model that can directly output multistep image-text content.

## D.2. Implementation Details

The experiments were conducted using a total of 24 A100 80G GPUs, with 8 GPUs dedicated to training IntJudge. We explored different large multimodal language models (MLLMs), including InternLM-XComposer2.5 (InternLMX2.5) and Qwen2-VL, ultimately selecting Qwen2VL-7B as the foundational model for training IntJudge to achieve an optimal balance between efficiency and accuracy. The training process involved LoRA-based parameterefficient fine-tuning based on LLaMA-Factory. To optimize training performance, DeepSpeed and FlashAttention-2 are adopted. We define a cutoff length of 16,240 tokens for inputs. We use a per-device batch size of 1, gradient accumulation steps of 8, a learning rate of 1.0e-4, a cosine learning rate schedule, and 20 epochs with BF16 mixed-precision enabled. The evaluation process involved sampling comparison pairs. Specifically, we conducted sampling rounds to obtain a total of E distinct battle pairs for each data instance. The sampling round value E was set to 2, resulting in 4,320 battle pairs being formed for comparison.

## D.3. Experimental Results on New Models

We present more main experimental results in Table 12. The new models Emu3, VILA-U and MiniGPT-5MMD are also evaluated by Human, GPT-based, and IntJudge-based evaluators and compared with 10 established baseline models using the win rate metrics. The results of methods are ranked by their performance on FDT metric evaluated by Human. Table 12 shows a clear hierarchy in model performance, with Human and GPT-4o+DALL-E3 leading across all metrics.

A closer look at the results reveals a consistency in rankings across evaluators. For instance, GPT-4o+DALL-E3 consistently secures second place in Human Evaluation and IntJudge Evaluation. Conventional end-to-end models, such as MiniGPT-5 and GILL, struggle to match the quality of their competitors, highlighting their limitations in generating contextually relevant and diverse outputs. However, GPT Evaluation shows a clear preference for outputs by GPT4o+DALL-E3. It is verified that GPT-based judgments are not objective enough due to the inherent bias. In contrast, our proposed IntJudge shows better alignment with human judgments, supporting the reliability of IntJudge as an effective evaluation framework.

Different evaluation metrics also offer more details about model performance. The FDT metric, which forces a decision in tie cases, highlights the dominance of Human and GPT-4o+DALL-E3. However, metrics that account for ties more flexibly, such as 'w/ Tie (0)' and 'w/ Tie (.5),' elevate end-to-end models like VILA-U and Emu3, suggesting that these models produce outputs that, while not always definitive winners, are frequently competitive. This distinction underscores the importance of using diverse metrics to capture various dimensions of model performance.

The new two-stage models show promising results, with VILA-U standing out for its balanced performance across all metrics, making it a reasonable option for general interleaved image-text tasks. MiniGPT-5MMD (finetuned on MMDialog) shows slight improvements over its variant MiniGPT-5 (finetuned on VIST), indicating progress but still trailing behind the latest models. Meanwhile, Emu3 performs well under specific metrics, such as 'w/ Tie (.5),' showing the potential to generate tie-worthy outputs with a certain quality.

The results also highlight the challenges faced by conventional end-to-end models, such as NExT-GPT, and GILL, which consistently underperform. These models reveal the inherent difficulty in achieving coherence and contextual relevance in interleaved generation tasks. Though Anole achieved a decent ranking as a representative end-to-end model, more advanced end-to-end models are needed for better visual generation quality.

Overall, the experimental results validate the effectiveness of IntJudge as a reliable evaluator, demonstrating its consistency with human judgments. The analysis underscores the strengths of integrated generation pipelines such as GPT4o+DALL-E3 and Gemini1.5+Flux. and identifies opportunities for improvement in two-stage and end-to-end models. Looking forward, expanding the training dataset, enhancing model architectures and improving the evaluation methods will all be critical in driving further progress in open-ended interleaved image-text generation, pushing the boundaries of multimodal learning research.

## D.4. Main Results Breakdown

Fig. 15 presents the win rates of 14 interleaved generation methods across 23 meta-topics, evaluated solely through human evaluations. The methods are evaluated using four distinct metrics: Force Dividing Tie (FDT), Without Tie, With Tie (0), and With Tie (0.5). The results are presented using histogram figures, which provide a clear visual comparison of model performance across different topic scenarios. For example, SOTA models like GPT-4o+DALL-E3, Emu3, and VILA-U consistently ranked high in categories like 'Storybook Creation,' 'Graph Generation,' and '2D Image Reasoning,' showcasing their superior capabilities in generating coherent interleaved content. Conversely, models like MiniGPT-5, NExT-GPT, and GILL struggled across most tasks, especially in areas such as 'Healthcare Tasks,' 'Multimodal Time Series Forecasting,' and 'Educational Tasks,' indicating a need for improved contextual understanding and generation capabilities. Training on larger datasets that include more domain knowledge may mitigate these issues and improve their interleaved generation performance.

FDT

Figure 15. The win rates of 14 interleaved generation methods across 23 meta-topics.

<!-- image -->

Table 12. Comparison of model win rates evaluated by human, GPT-4o, and our IntJudge under FDT and different tie metrics. FDT: Force Dividing Tie metric. w/o Tie: Non-tie case. w/ Tie (0) and w/ Tie (.5): Count a tie as 0 and 0.5 wins for a model in a battle, respectively.

| Method         | Human Evaluation   | Human Evaluation   | Human Evaluation   | Human Evaluation   | GPT Evaluation   | GPT Evaluation   | GPT Evaluation   | GPT Evaluation   | IntJudge Evaluation   | IntJudge Evaluation   | IntJudge Evaluation   | IntJudge Evaluation   |
|----------------|--------------------|--------------------|--------------------|--------------------|------------------|------------------|------------------|------------------|-----------------------|-----------------------|-----------------------|-----------------------|
| Method         | FDT                | w/o Tie            | w/ Tie (0)         | w/ Tie (.5)        | FDT              | w/o Tie          | w/ Tie (0)       | w/ Tie (.5)      | FDT                   | w/o Tie               | w/ Tie (0)            | w/ Tie (.5)           |
| Human          | 83.94%             | 86.50%             | 70.78%             | 79.87%             | 82.76%           | 83.09%           | 82.27%           | 82.76%           | 85.65%                | 89.11%                | 72.62%                | 81.87%                |
| GPT-4o+DALL-E3 | 78.20%             | 80.73%             | 66.17%             | 75.19%             | 86.33%           | 86.60%           | 86.23%           | 86.44%           | 83.24%                | 86.20%                | 71.46%                | 80.01%                |
| Gemini1.5+Flux | 66.67%             | 66.95%             | 51.97%             | 63.16%             | 73.39%           | 73.38%           | 72.75%           | 73.18%           | 66.11%                | 67.92%                | 49.58%                | 63.08%                |
| VILA-U         | 62.10%             | 62.34%             | 61.57%             | 62.19%             | 49.47%           | 49.55%           | 49.29%           | 49.56%           | 68.66%                | 58.58%                | 36.94%                | 55.41%                |
| Anole          | 52.72%             | 53.10%             | 38.96%             | 52.28%             | 53.25%           | 53.06%           | 52.60%           | 53.04%           | 56.33%                | 52.77%                | 33.85%                | 51.78%                |
| Emu3           | 54.05%             | 55.24%             | 52.25%             | 54.95%             | 47.19%           | 47.27%           | 46.74%           | 47.30%           | 54.01%                | 54.48%                | 39.04%                | 53.21%                |
| SEED-X         | 53.25%             | 52.03%             | 38.55%             | 51.51%             | 56.46%           | 56.63%           | 55.63%           | 56.51%           | 53.76%                | 54.32%                | 36.15%                | 52.88%                |
| SEED-LLaMA     | 44.43%             | 42.47%             | 30.76%             | 44.54%             | 42.33%           | 42.13%           | 41.68%           | 42.22%           | 46.43%                | 45.49%                | 28.21%                | 47.20%                |
| Emu2           | 40.31%             | 36.64%             | 24.78%             | 40.97%             | 42.49%           | 42.43%           | 41.52%           | 42.60%           | 36.60%                | 31.84%                | 19.36%                | 38.96%                |
| NExT-GPT       | 33.59%             | 27.74%             | 18.76%             | 34.95%             | 24.81%           | 24.62%           | 24.27%           | 24.97%           | 34.08%                | 25.94%                | 15.39%                | 35.72%                |
| Show-o         | 37.47%             | 35.97%             | 24.57%             | 40.42%             | 33.21%           | 32.81%           | 32.26%           | 33.10%           | 33.65%                | 24.22%                | 13.59%                | 35.53%                |
| MiniGPT-5MMD   | 32.26%             | 32.04%             | 30.74%             | 32.77%             | 32.59%           | 32.47%           | 32.25%           | 32.59%           | 28.98%                | 25.30%                | 14.84%                | 35.51%                |
| MiniGPT-5      | 31.47%             | 28.59%             | 19.72%             | 35.24%             | 31.18%           | 30.98%           | 30.66%           | 31.18%           | 24.65%                | 15.65%                | 9.08%                 | 30.07%                |
| GILL           | 25.96%             | 20.82%             | 14.33%             | 29.91%             | 31.47%           | 31.25%           | 30.70%           | 31.58%           | 23.23%                | 16.80%                | 10.35%                | 29.54%                |

## D.5. More Pairwise Model Performance

Figure 16 presents more heatmaps that illustrate the pairwise model performance evaluated by different evaluators, including Human, GPT, and IntJudge. These heatmaps provide a visual representation of the comparative strengths and weaknesses of each model across multiple metrics, such as Force Dividing Tie (FDT) and different approaches to handling tie cases (without ties, ties as zero, and ties as 0.5). By examining these heatmaps, we gain a clearer understanding of how well each model fares against others, diving deeper into performance consistency and discrepancies across evaluators.

## D.6. More Ablations on Sampling Size

Figure 17 illustrates the results of additional ablation studies focusing on the effect of sampling size on model performance. The figure compares win rates across different evaluators, including Human, GPT, and IntJudge, under various metrics such as Force Dividing Tie (FDT) and different methods for treating ties (without ties, ties as zero, and ties as 0.5). These ablation studies are crucial for understanding the impact of sampling on the robustness of model comparisons and provide insights into how sampling variations influence the ranking consistency among different evaluators. Most importantly, the results help validate the stability of our evaluation framework.

## D.7. Case Study

Here, we present case studies of interleaved content generation, which involve 14 distinct models competing in the OpenING tasks. From these models, 23 model battle pairs are formed, and each pair engages in a battle over a representative task from the 23 meta-topics in our OpenING benchmark. The results of these battles are shown in Fig. 19. A human judge awards a gold medal to the model that wins absolutely over its competitor in a pair. In cases where the generated content by two competing models is of similar qualities, the judge awards use tie metrics to determine and award the silver medal to the slightly better-performing model. The human judgments are based on the following criteria, which include the eight typical errors these models tend to commit in the interleaved content generation:

1. No-Text or No-Image : A model fails to generate text or images or both when explicitly instructed or expected to do so.
2. Factual Error : A model may present incorrect or madeup information, such as referencing nonexistent studies, authors, or events. Additionally, it might fail to follow instructions, examples include refusing to provide answers, misinterpreting the input and generating irrelevant responses, displaying reasoning mistakes, or producing inaccurate images.
3. Content Incoherent : The generated content lacks coherence with the input or is inconsistent across multiple outputs, typically in style or entities
4. Offensive Content : Offensive content is defined as materials that include distorted or disturbing imagery or scenes that likely cause discomfort or distress to readers. It also includes content that raises safety concerns or violates safety guidelines, such as depictions of violence, harm, hate speech, adult content, illegal activities, dangerous behaviors, and unethical actions. A model may also fail to consider cultural nuances or sensitivities.
5. Image-Text Inconsistent : This error occurs when the images do not semantically align with the corresponding text, leading to confusion or misinterpretation.
6. Poor Image Quality : The generated images have low quality, such as being completely blank or blacked-out,

Figure 16. Win rate matrices of 14 interleaved genration methods, evaluated by Human, GPT-4o, and our IntJudge, respectively.

<!-- image -->

blurry, poorly laid out, and lacking realism or aesthetic appeal.

7. Poor Text Quality : The generated texts are of low quality, including nonsensical mumbling, grammatical errors that impair readability and understanding, and being too short to convey meaningful information.
8. Incomplete Response : A model abruptly stops generating a textual response, or a model outputs an incomplete

or truncated response.

## E. Finetuning MLLMs on OpenING

We present the extended experimental results of training MiniGPT-5 on the Dev Set of OpenING and testing the finetuned model on the Test Set of OpenING. The objective is to verify if finetuning on the specific data of OpenING can im- prove the performance of interleaved generation tasks. The Dev Set of OpenING can offer a set of 3,000 training samples that align with the diverse unique tasks. The MiniGPT-5 model was finetuned using the Dev Set for 5 epochs with a learning rate of 2e-5, utilizing an Adam optimizer. To enhance training stability, Adam epsilon of 1e-8 was applied. The model training incorporated mixed-precision computations to speed up the training process. The results are evalu- ated on the Test Set of OpenING using IntJudge.

Figure 17. Win rate curves with respect to different sampling sizes.

<!-- image -->

In Table 13, the performance of MiniGPT-5OpenING, the finetuned version of MiniGPT-5, is compared against other state-of-the-art models and the original MiniGPT-5 baselines (MiniGPT-5 is finetuned on VIST and MiniGPT-5MMD is finetuned on MMDialog). We set E to 1 and randomly sampled 2,160 samples for this efficient evaluation. The evaluation metrics include four scenarios: Force Dividing Tie (FDT), Without Tie (w/o Tie), With Tie counted as 0 (w/ Tie (0)), and With Tie counted as 0.5 (w/ Tie (.5)).

Table 13. Model Win Rates evaluated by IntJudge. Representative models for each type of methods are chosen to be compared with MiniGPT-5OpenING, which is a MiniGPT-5 version finetuned on the Dev Set of OpenING. FDT: Force Dividing Tie metric. w/o Tie: Non-tie case. w/ Tie (0) and w/ Tie (.5): Count a tie as 0 and 0.5 wins for a model in a battle, respectively.

| Model            | FDT    | w/o Tie   | w/ Tie (0)   | w/ Tie (.5)   |
|------------------|--------|-----------|--------------|---------------|
| Human            | 84.66% | 86.01%    | 75.46%       | 81.60%        |
| Gemini1.5+Flux   | 73.44% | 73.15%    | 61.72%       | 69.53%        |
| VILA-U           | 62.50% | 60.14%    | 41.50%       | 57.00%        |
| MiniGPT-5OpenING | 60.24% | 63.76%    | 44.71%       | 59.65%        |
| Emu3             | 56.02% | 55.20%    | 36.13%       | 53.40%        |
| SEED-X           | 54.23% | 54.67%    | 41.80%       | 53.57%        |
| Emu2             | 47.10% | 39.33%    | 25.36%       | 43.12%        |
| Anole            | 41.56% | 43.10%    | 32.47%       | 44.81%        |
| SEED-LLaMA       | 41.33% | 38.14%    | 24.67%       | 42.33%        |
| GILL             | 36.25% | 32.04%    | 20.62%       | 38.44%        |
| Show-o           | 35.76% | 30.53%    | 19.21%       | 37.75%        |
| MiniGPT-5        | 31.54% | 26.37%    | 16.11%       | 35.57%        |
| MiniGPT-5MMD     | 28.19% | 23.44%    | 14.71%       | 33.33%        |

The results highlight that MiniGPT-5OpenING achieves significant improvements over the baseline MiniGPT-5 models across all metrics. For example, in the Without Tie (w/o Tie) scenario, the finetuned model shows a substantial 37.39% relative improvement over the MiniGPT-5 baseline. These findings confirm that training on a specialized interleaved image-text dataset such as the Dev Set of OpenING enhances the model with better contextual understanding and alignment capabilities for generating coherent interleaved image-text content. Further studies are ongoing to improve the performance of SOTA models.

## F. Limitations of This Study

Although OpenING advances interleaved image-text generation evaluation, several limitations remain for improvement. First, despite covering diverse tasks (56 tasks across 23 metatopics), some real-world scenarios remain underrepresented or oversimplified, potentially limiting the generalizability to practical applications. Tasks requiring fine-grained understanding or multi-step reasoning need to be supplied to capture real-world needs. Second, although the IntJudge model improves alignment with human evaluations, its generalizability remains limited by the diversity and quality of training data. The benchmark's reliance on human-annotated data to establish ground truth and train judge models is laborintensive and expensive. While our Reference-Augmented Generation (RAG) approach helps scale training data, manual annotations remain a critical component for ensuring quality and alignment with human preferences. Furthermore, the computational resources required for training and deploying IntJudge present scalability challenges, potentially limiting accessibility for researchers with fewer resources.

In addition, current interleaved image-text generation methods still struggle with producing high-quality, coherent interleaved content, particularly in multi-step tasks that require maintaining consistency across generated images and text. Issues like content incoherence, poor image quality, and mismatches between generated text and images persist across evaluated models, particularly in end-to-end approaches. To tackle these issues, more advanced MLLMs trained with a large-scale interleaved image-text dataset are to be investigated. Moreover, while our IntJudge demonstrates significant advantages over GPT-based evaluators, we recognize two aspects for improvement: 1) Potential biases may rise from crowdsourced data (e.g., human preferences in aesthetic judgment), and 2) The current foundation model of IntJudge mainly supports English and Chinese. Building a sufficiently comprehensive, diverse, and representative dataset is expected to greatly promote the development of multimodal generation.

These limitations underscore the need for continued development of more diverse datasets and more robust evaluation frameworks to address the complexities of interleaved generation evaluation, enabling more practical interleaved imagetext generation methods and pushing forward the boundary of future MLLMs. Future work may benefit from diversifying data sources and cross-cultural annotations, expanding multilingual capabilities, and implementing debiasing techniques to improve fairness.

## Acknowledgement

This work is partially supported by the National Key R&amp;D Program of China (No. 2022ZD0160101 and No. 2022ZD0160102).

## References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774 , 2023. 1
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In Proceedings of the Advances in Neural Information Processing Systems , 2022. 3
- [3] Amar Ali-bey, Brahim Chaib-draa, and Philippe Gigu` ere. BoQ: A place is worth a bag of learnable queries. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 17794-17803, 2024. 14
- [4] Jie An, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Lijuan Wang, and Jiebo Luo. Openleaf: Open-domain interleaved image-text generation and evaluation. arXiv preprint arXiv:2310.07749 , 2023. 1, 2, 3
- [5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966 , 2023. 1
- [6] Yushi Bai, Jiahao Ying, Yixin Cao, Xin Lv, Yuze He, Xiaozhi Wang, Jifan Yu, Kaisheng Zeng, Yijia Xiao, Haozhe Lyu, et al. Benchmarking foundation models with language-modelas-an-examiner. In Proceedings of the Advances in Neural Information Processing Systems , 2024. 3
- [7] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. CVPR , 2022. 14
- [8] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science , 2(3):8, 2023. 5, 17, 27
- [9] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux , 2024. Accessed: 202411-05. 5, 17, 27
- [10] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 18392-18402, 2023. 13
- [11] Ju-Seung Byun, Jiyun Chun, Jihyung Kil, and Andrew Perrault. ARES: Alternating reinforcement learning and supervised fine-tuning for enhanced multi-modal chain-of-thought reasoning through diverse AI feedback. arXiv preprint arXiv:2407.00087 , 2024. 1
- [12] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings of the ieee conference on computer vision and pattern recognition , pages 961-970, 2015. 15
- [13] Yitao Cai, Huiyu Cai, and Xiaojun Wan. Multi-modal sarcasm detection in Twitter with hierarchical fusion model. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics , 2019. 15
- [14] Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. MLLM-as-a-Judge: Assessing multimodal llm-as-a-judge with vision-language benchmark. In Proceedings of the International Conference on Machine Learning , 2024. 3, 4, 5, 10
- [15] Dongping Chen, Yue Huang, Siyuan Wu, Jingyu Tang, Liuyi Chen, Yilin Bai, Zhigang He, Chenlong Wang, Huichi Zhou, Yiqiang Li, Tianshuo Zhou, Yue Yu, Chujie Gao, Qihui Zhang, Yi Gui, Zhen Li, Yao Wan, Pan Zhou, Jianfeng Gao, and Lichao Sun. Gui-world: A dataset for gui-oriented multimodal llm-based agents, 2024. 13
- [16] Ethan Chern, Jiadi Su, Yan Ma, and Pengfei Liu. Anole: An open, autoregressive, native large multimodal models for interleaved image-text generation. arXiv preprint arXiv:2407.06135 , 2024. 3, 5, 17, 27
- [17] Daniel Claman, Emre Sezgin, et al. Artificial intelligence in dental education: Opportunities and challenges of large language models and multimodal foundation models. JMIR Medical Education , 10(1):e52346, 2024. 1
- [18] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Scaling egocentric vision: The epic-kitchens dataset. In European Conference on Computer Vision (ECCV) , 2018. 15
- [19] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. CARLA: An open urban driving simulator. In Proceedings of the 1st Annual Conference on Robot Learning , pages 1-16, 2017. 15
- [20] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨ uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of the International Conference on Machine Learning , 2024. 3
- [21] Zhengcong Fei, Zekang Li, Jinchao Zhang, Yang Feng, and Jie Zhou. Towards expressive communication with internet memes: A new multimodal conversation dataset and benchmark. arXiv preprint arXiv:2109.01839 , 2021. 15
- [22] Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, and Ying Shan. Making llama see and draw with seed tokenizer. arXiv preprint arXiv:2310.01218 , 2023. 5, 17, 27
- [23] Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. Seed-data-edit technical report: A hybrid dataset for instructional image editing. arXiv preprint arXiv:2405.04007 , 2024. 3, 5, 13, 17, 27
- [24] Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, and Ying Shan. Making LLaMA SEE and Draw with SEED Tokenizer. In Proceedings of the International Conference on Learning Representations , 2024. 3
- [25] Matthew F Glasser, Timothy S Coalson, Emma C Robinson, Carl D Hacker, John Harwell, Essa Yacoub, Kamil Ugurbil, Jesper Andersson, Christian F Beckmann, Mark Jenkinson, et al. A multi-modal parcellation of human cerebral cortex. Nature , 536(7615):171-178, 2016. 1
- [26] Yulia Gryaditskaya, Mark Sypesteyn, Jan Willem Hoftijzer, Sylvia Pont, Fr´ edo Durand, and Adrien Bousseau. Opensketch: A richly-annotated dataset of product design sketches. ACM Transactions on Graphics (Proc. SIGGRAPH Asia) , 38, 2019. 13
- [27] Chunhui Gu, Chen Sun, David A Ross, Carl Vondrick, Caroline Pantofaru, Yeqing Li, Sudheendra Vijayanarasimhan, George Toderici, Susanna Ricco, Rahul Sukthankar, et al. Ava: A video dataset of spatio-temporally localized atomic visual actions. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 6047-6056, 2018. 15
- [28] Conghui He, Wei Li, Zhenjiang Jin, Chao Xu, Bin Wang, and Dahua Lin. OpenDataLab: Empowering general artificial intelligence with open datasets. arXiv preprint arXiv:2407.13773 , 2024. 4
- [29] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718 , 2021. 3
- [30] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium. In Proceedings of the Advances in Neural Information Processing Systems , 2017. 3
- [31] Judith Holler and Stephen C Levinson. Multimodal language processing in human communication. Trends in Cognitive Sciences , 23(8):639-652, 2019. 1
- [32] Ting-Hao Huang, Francis Ferraro, Nasrin Mostafazadeh, Ishan Misra, Aishwarya Agrawal, Jacob Devlin, Ross Girshick, Xiaodong He, Pushmeet Kohli, Dhruv Batra, et al. Visual storytelling. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , 2016. 1
- [33] Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Pnp inversion: Boosting diffusion-based editing with 3 lines of code. International Conference on Learning Representations (ICLR) , 2024. 13
- [34] Hyung-Kwon Ko, Gwanmo Park, Hyeon Jeon, Jaemin Jo, Juho Kim, and Jinwook Seo. Large-scale text-to-image generation models for visual artists' creative works. In Proceedings of the International Conference on Intelligent User Interfaces , 2023. 1
- [35] Jing Yu Koh, Daniel Fried, and Russ R Salakhutdinov. Generating images with multimodal language models. In Proceedings of the Advances in Neural Information Processing Systems , 2024. 3, 5, 17, 27
- [36] Ehsan Latif, Gengchen Mai, Matthew Nyaaba, Xuansheng Wu, Ninghao Liu, Guoyu Lu, Sheng Li, Tianming Liu, and Xiaoming Zhai. Artificial general intelligence (AGI) for education. arXiv preprint arXiv:2304.12479 , 1, 2023. 1
- [37] Hugo Laurenc ¸on, Lucile Saulnier, L´ eo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents. In Proceedings of the Advances in Neural Information Processing Systems , 2024. 1
- [38] Zhikai Li, Xuewen Liu, Dongrong Fu, Jianquan Li, Qingyi Gu, Kurt Keutzer, and Zhen Dong. K-sort arena: Efficient and reliable benchmarking for generative models via k-wise human preferences. arXiv preprint arXiv:2408.14468 , 2024. 3, 10
- [39] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Proceedings of the ACL Workshop: Text Summarization Branches Out , pages 74-81, 2004. 3
- [40] Chang Liu, Haoning Wu, Yujie Zhong, Xiaoyun Zhang, Yanfeng Wang, and Weidi Xie. Intelligent grimm - open-ended visual storytelling via latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 6190-6200, 2024. 13
- [41] Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-mGPT: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining. arXiv preprint arXiv:2408.02657 , 2024. 3
- [42] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Proceedings of the Advances in Neural Information Processing Systems , 2024. 1
- [43] Minqian Liu, Zhiyang Xu, Zihao Lin, Trevor Ashby, Joy Rimchala, Jiaxin Zhang, and Lifu Huang. Holistic evaluation for interleaved text-and-image generation. arXiv preprint arXiv:2406.14643 , 2024. 1, 2, 3
- [44] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. MMBench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281 , 2023. 1
- [45] Quanfeng Lu, Wenqi Shao, Zitao Liu, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, Yu Qiao, and Ping Luo. Gui odyssey: A comprehensive dataset for cross-app gui navigation on mobile devices. arXiv preprint arXiv:2406.08451 , 2024. 13
- [46] Yao Mu, Qinglong Zhang, Mengkang Hu, Wenhai Wang, Mingyu Ding, Jun Jin, Bin Wang, Jifeng Dai, Yu Qiao, and Ping Luo. Embodiedgpt: Vision-language pre-training via embodied chain of thought. In Proceedings of the Advances in Neural Information Processing Systems , 2024. 1
- [47] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In 2008 Sixth Indian conference on computer vision, graphics &amp; image processing , pages 722-729. IEEE, 2008. 14
- [48] OpenAI. Hello GPT-4o. https://openai.com/ index/hello-gpt-4o/ , 2024. Accessed: 2024-05-26. 4, 5, 17, 27
- [49] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. BLEU: a method for automatic evaluation of machine translation. In Proceedings of the annual meeting of the Association for Computational Linguistics , 2002. 3
- [50] Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S Nassar, and Peter W J Staar. Doclaynet: A large humanannotated dataset for document-layout segmentation. page 3743-3751, 2022. 13
- [51] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨ uller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution
52. image synthesis. In Proceedings of the International Conference on Learning Representations , 2024. 4
- [52] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 , 2022. 3
- [53] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨ orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Bision and Pattern Recognition , 2022. 3
- [54] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training GANs. In Proceedings of the Advances in Neural Information Processing Systems , 2016. 3
- [55] Nils Schaetti. Sfgram: a dataset containing thousands of scienc-fiction books and novels. https://github.com/ nschaetti/EchoTorch , 2018. 13
- [56] Huiyang Shao, Qianqian Xu, Peisong Wen, Peifeng Gao, Zhiyong Yang, and Qingming Huang. Building bridge across the time: Disruption and restoration of murals in the wild. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 20259-20269, 2023. 3
- [57] Danqing Shi, Weiwei Cui, Danqing Huang, Haidong Zhang, and Nan Cao. Reverse-engineering information presentations: Recovering hierarchical grouping from layouts of visual elements. Visual Intelligence , 1(1):9, 2023. 13
- [58] Yale Song, Jordi Vallmitjana, Amanda Stent, and Alejandro Jaimes. Tvsum: Summarizing web videos using titles. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 5179-5187, 2015. 14
- [59] Francesco Stella, Cosimo Della Santina, and Josie Hughes. How can LLMs transform the robotic design process? Nature Machine Intelligence , pages 1-4, 2023. 1
- [60] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are incontext learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 1439814409, 2024. 5, 17, 27
- [61] Zhiyu Tan, Xiaomeng Yang, Luozheng Qin, Mengping Yang, Cheng Zhang, and Hao Li. Evalalign: Supervised fine-tuning multimodal llms with human-aligned data for evaluating textto-image models. CoRR , 2024. 1
- [62] Zineng Tang, Ziyi Yang, Mahmoud Khademi, Yang Liu, Chenguang Zhu, and Mohit Bansal. CoDi-2: In-context interleaved and interactive any-to-any generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 27425-27434, 2024. 1
- [63] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818 , 2024. 1, 3, 27
- [64] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805 , 2023. 1, 5, 17, 27
- [65] InternLM Team. InternLM: A multilingual language model with progressively enhanced capabilities, 2023. 1
- [66] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. In Proceedings of the Advances in Neural Information Processing Systems , 2024. 3
- [67] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ ee Lacroix, Baptiste Rozi` ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 , 2023. 1
- [68] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie. Caltech-ucsd birds. Technical Report CNS-TR-2011-001, California Institute of Technology, 2011. 14
- [69] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191 , 2024. 5
- [70] Shiyao Wang, Qi Liu, Tiezheng Ge, Defu Lian, and Zhiqiang Zhang. A hybrid bandit model with visual priors for creative ranking in display advertising. In Proceedings of the Web Conference 2021 , pages 2324-2334, 2021. 13
- [71] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869 , 2024. 1, 3, 17, 27
- [72] Yidong Wang, Zhuohao Yu, Wenjin Yao, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, et al. PandaLM: An automatic evaluation benchmark for llm instruction tuning optimization. In Proceedings of the International Conference on Learning Representations , 2024. 2, 3
- [73] Benjamin Wilson, William Qi, Tanmay Agarwal, John Lambert, Jagjeet Singh, Siddhesh Khandelwal, Bowen Pan, Ratnesh Kumar, Andrew Hartnett, Jhony Kaesemodel Pontes, Deva Ramanan, Peter Carr, and James Hays. Argoverse 2: Next generation datasets for self-driving perception and forecasting. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS Datasets and Benchmarks 2021) , 2021. 15
- [74] Haoning Wu, Hanwei Zhu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Annan Wang, Wenxiu Sun, Qiong Yan, et al. Towards open-ended visual quality comparison. arXiv preprint arXiv:2402.16641 , 2024. 3
- [75] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. NExT-GPT: Any-to-Any Multimodal LLM. In Proceedings of the International Conference on Machine Learning , 2024. 5, 17, 27
- [76] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Liang Pan Jiawei Ren, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, Dahua Lin, and Ziwei Liu. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , 2023. 14
- [77] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. VILA-U: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429 , 2024. 17, 27
- [78] Peng Xia, Siwei Han, Shi Qiu, Yiyang Zhou, Zhaoyang Wang, Wenhao Zheng, Zhaorun Chen, Chenhang Cui, Mingyu Ding, Linjie Li, et al. MMIE: Massive multimodal interleaved comprehension benchmark for large vision-language models. arXiv preprint arXiv:2410.10139 , 2024. 1
- [79] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528 , 2024. 1, 5, 17, 27
- [80] Zhe Xu, Dacheng Tao, Ya Zhang, Junjie Wu, and Ah Chung Tsoi. Architectural style classification using multinomial latent logistic regression. In Computer Vision-ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I 13 , pages 600-615. Springer, 2014. 13
- [81] Shuai Yang, Yuying Ge, Yang Li, Yukang Chen, Yixiao Ge, Ying Shan, and Yingcong Chen. Seed-story: Multimodal long story generation with large language model. arXiv preprint arXiv:2407.08683 , 2024. 3, 13
- [82] Suorong Yang, Suhan Guo, Jian Zhao, and Furao Shen. Investigating the effectiveness of data augmentation from similarity and diversity: An empirical study. Pattern Recognition , 148: 110204, 2024. 3
- [83] Yue Yang, Artemis Panagopoulou, Qing Lyu, Li Zhang, Mark Yatskar, and Chris Callison-Burch. Visual goal-step inference using wikihow. arXiv preprint arXiv:2104.05845 , 2021. 1
- [84] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of LMMs: Preliminary explorations with GPT-4V (ision). arXiv preprint arXiv:2309.17421 , 9(1):1, 2023. 1
- [85] Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, et al. MMT-Bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi. arXiv preprint arXiv:2404.16006 , 2024. 1
- [86] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 9556-9567, 2024. 1
- [87] Pan Zhang, Xiaoyi Dong, Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Haodong Duan, Songyang Zhang, Shuangrui Ding, et al. InternLM-XComposer: A vision-language large model for advanced text-image comprehension and composition. arXiv preprint arXiv:2309.15112 , 2023. 1
- [88] Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, Songyang Zhang, Wenwei Zhang, Yining Li, Yang Gao, Peng Sun, Xinyue Zhang, Wei Li, Jingwen Li,
90. Wenhai Wang, Hang Yan, Conghui He, Xingcheng Zhang, Kai Chen, Jifeng Dai, Yu Qiao, Dahua Lin, and Jiaqi Wang. InternLM-XComposer-2.5: A versatile large vision language model supporting long-contextual input and output. arXiv preprint arXiv:2407.03320 , 2024. 5
- [89] Xinlu Zhang, Yujie Lu, Weizhi Wang, An Yan, Jun Yan, Lianke Qin, Heng Wang, Xifeng Yan, William Yang Wang, and Linda Ruth Petzold. GPT-4V (ision) as a generalist evaluator for vision-language tasks. arXiv preprint arXiv:2311.01361 , 2023. 3
- [90] Bingchen Zhao, Yongshuo Zong, Letian Zhang, and Timothy Hospedales. Benchmarking multi-image understanding in vision and language models: Perception, knowledge, reasoning, and multi-hop reasoning. arXiv preprint , 2024. 14
- [91] Wangbo Zhao, Yizeng Han, Jiasheng Tang, Zhikai Li, Yibing Song, Kai Wang, Zhangyang Wang, and Yang You. A stitch in time saves nine: Small vlm is a precise guidance for accelerating large vlms. arXiv preprint arXiv:2412.03324 , 2024. 1
- [92] Kaizhi Zheng, Xuehai He, and Xin Eric Wang. MiniGPT5: Interleaved vision-and-language generation via generative vokens. arXiv preprint arXiv:2310.02239 , 2023. 3, 5, 17, 27
- [93] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Proceedings of the Advances in Neural Information Processing Systems , 2023. 3, 5, 10
- [94] Xu Zhong, Jianbin Tang, and Antonio Jimeno Yepes. Publaynet: largest dataset ever for document layout analysis. In 2019 International Conference on Document Analysis and Recognition (ICDAR) , pages 1015-1022. IEEE, 2019. 13
- [95] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 633-641, 2017. 14
- [96] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039 , 2024. 1
- [97] Pengfei Zhu, Longyin Wen, Dawei Du, Xiao Bian, Heng Fan, Qinghua Hu, and Haibin Ling. Detection and tracking meet drones challenge. IEEE Transactions on Pattern Analysis and Machine Intelligence , 44(11):7380-7399, 2021. 15

Figure 18. Examples of 23 Meta-Topics ( bold font) and corresponding selected task (regular font) in Our OpenING Benchmark.

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

Figure 19. Illustration of 23 pairwise comparison cases. The meta-topic name ( bold font) and task name (regular font) are given for each case. Using the eight criteria detailed in Sec. C.3, human judges evaluate and compare the output of each model pair. Gold medal is awarded to a model that generates content of significantly higher quality. In a tie setting, where the quality of outputs from the A model and B model is similar, Silver medal is awarded to a model that generates relatively more favorable content. Errors that occur during model generation are highlighted by a red checkmark and Error Types . Models that pass all human moderation checks are marked with a green checkmark .

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->