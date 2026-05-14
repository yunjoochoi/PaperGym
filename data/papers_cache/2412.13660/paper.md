## PsyDT: Using LLMs to Construct the Digital Twin of Psychological Counselor with Personalized Counseling Style for Psychological Counseling

Haojie Xie 1 * , Yirong Chen 2 ∗ , Xiaofen Xing 2 † , Jingkai Lin 2 , Xiangmin Xu 1,3

1 School of Future Technology, South China University of Technology

2 School of Electronic and Information Engineering, South China University of Technology 3 Pazhou Lab

{fthjxie, eeyirongchen}@mail.scut.edu.cn , {xfxing, xmxu}@scut.edu.cn

## Abstract

Currently, large language models (LLMs) have made significant progress in the field of psychological counseling. However, existing mental health LLMs overlook a critical issue where they do not consider the fact that different psychological counselors exhibit different personal styles, including linguistic style and therapy techniques, etc. As a result, these LLMs fail to satisfy the individual needs of clients who seek different counseling styles. To help bridge this gap, we propose PsyDT, a novel framework using LLMs to construct the D igital T win of Psy chological counselor with personalized counseling style. Compared to the timeconsuming and costly approach of collecting a large number of real-world counseling cases to create a specific counselor's digital twin, our framework offers a faster and more costeffective solution. To construct PsyDT, we utilize dynamic one-shot learning by using GPT-4 to capture counselor's unique counseling style, mainly focusing on linguistic style and therapy techniques. Subsequently, using existing singleturn long-text dialogues with client's questions, GPT-4 is guided to synthesize multi-turn dialogues of specific counselor. Finally, we finetune the LLMs on the synthetic dataset, PsyDTCorpus, to achieve the digital twin of psychological counselor with personalized counseling style. Experimental results indicate that our proposed PsyDT framework can synthesize multi-turn dialogues that closely resemble realworld counseling cases and demonstrate better performance compared to other baselines, thereby show that our framework can effectively construct the digital twin of psychological counselor with a specific counseling style. 1

## 1 Introduction

In recent years, LLMs have made significant advancements, exemplified by ChatGPT (OpenAI,

* Equal contribution.

† Corresponding author.

[1 https://github.com/scutcyr/SoulChat2.0](https://github.com/scutcyr/SoulChat2.0)

<!-- image -->

(b) Our proposed PsyDT framework.

Figure 1: Difference between our proposed PsyDT framework and others. (a) Previous methods mixed multi-turn dialogues with multiple counseling styles to fine-tune LLM. (b) Our proposed PsyDT framework uses LLMs to construct the digital twin of psychological counselor with a specific counseling style.

2022), GPT-4 (OpenAI, 2024), LLaMA (Touvron et al., 2023), Qwen (Bai et al., 2023), ChatGLM (Du et al., 2022), etc. While these LLMs excel in a variety of tasks, they often encounter limitations in specialized fields such as mental health due to a lack of domain-specific expertise. In addition, with the global rise in the prevalence of depression and anxiety (Santomauro et al., 2021), mental health has garnered widespread attention, prompting researchers to explore the application of LLMs in psychological counseling. The value of mental health LLMs lies in their potentiality to provide emotional support and counseling services to individuals. Currently, a series of mental health LLMs have been proposed, including MeChat (Qiu et al., 2024a), PsyChat (Qiu et al., 2024b), SoulChat (Chen et al., 2023), EmoLLM (EmoLLM, 2024), MindChat (Xin Yan, 2023), CPsyCoun (Zhang et al., 2024), etc.

Due to ethics policy and privacy protection, realworld multi-turn mental health dialogues datasets are exceedingly rare. Consequently, most recent LLM-based mental health researches rely on syn- thesizing multi-turn dialogues datasets. For instance, Qiu et al. (2024a) introduced SMILE, a technique for expanding single-turn dialogues into multi-turn dialogues, thereby better simulating realworld interactions between clients and counselors. Similarly, Chen et al. (2023) developed SoulChatCorpus, a multi-turn empathetic dialogue dataset of more than 2 million samples, in which the input is the multi-turn dialogue context, and the target is empathetic responses that cover expressions such as questioning, comfort, recognition, listening, trust, emotional support, etc. Zhang et al. (2024) later proposed CPsyCoun, a report-based multiturn dialogue reconstruction and evaluation framework for Chinese psychological counseling.

Figure 2: Illustration of multi-turn dialogues synthesis method of PsyDT framework and PsyDTLLM model.

<!-- image -->

However, despite the promising results of previous mental health LLMs, they overlook a critical issue where they do not take into account the fact that different psychological counselors exhibit different personal styles, including variations in linguistic style and therapy techniques. As a result, these LLMs struggle to meet the individual needs of clients who seek different counseling styles. Moreover, as illustrated in Figure 1, fine-tuning LLMs on multi-turn mental health dialogues datasets that blend multiple counseling styles often lead to instability in the response. To this end, we introduce the concept of psychological counselor's digital twin. To construct the digital twin, a large volume of real-world counseling cases from a specific psychological counselor are typically collected to fine-tune LLMs. However, the process of collect- ing counseling cases is not only time-consuming but also costly.

In response to these challenges, we propose PsyDT, a novel framework using LLMs to construct the D igital T win of Psy chological counselor with personalized counseling style. To construct PsyDT, we need to synthesize multi-turn dialogues of specific psychological counselor. To ensure the quality of the synthetic multi-turn dialogues, we first select 5,000 high-quality single-turn dialogues from the SoulChatCorpus (Chen et al., 2023). Subsequently, in order to promise the complexity and diversity of clients' linguistic style in the synthetic multi-turn dialogues, we employ GPT-4 to simulate the Big Five personality traits (Costa and McCrae, 1999) of clients based on the clients' question from single-turn dialogues. To conduct real-world psychological counselor's digital twin, we invited a professional psychological counselor to play the role of digital twin. The counselor was asked to have conversations covering 12 counseling topics with 12 different clients. And then GPT-4 is employed to summarize the linguistic style and therapy technique according to counseling topic from above collected real-world counseling cases of the specific counselor. Combining the above simulated clients' personality, GPT-4 is used as the guidance for multi-turn dialogues synthesis of specific counselor. Finally, based on the synthetic multi-turn dialogues dataset, PsyDTCorpus, we construct the digital twin of psychological counselor with the specific counseling style using the multi-turn in- struction fine-tuning (MIFT) method. Extensive experimental results demonstrate that our proposed PsyDT framework can quickly and cost-effectively construct the digital twin of psychological counselor with a specific counseling style, validating the effectiveness of the PsyDT framework.

In summary, the contributions of this work can be succinctly outlined as follows:

- To the best of our knowledge, our work is the first to use LLMs to construct the digital twin of psychological counselor with personalized counseling style. The proposed multi-turn dialogues synthesis method of PsyDT framework can quickly and cost-effectively synthesize PsyDTCorpus, a high-quality multi-turn mental health dialogues dataset of psychological counselor with specific counseling style.
- We design automatic evaluations and manual evaluations of synthetic dataset and fine-tuned LLMof psychological counselor with specific counseling style to indicate effectiveness and superiority of our PsyDT framework.
- Experimental results indicate that PsyDT can synthesize multi-turn dialogues that closely resemble real-world counseling cases and demonstrate better performance compared to other baselines, thereby demonstrate the strong potential of PsyDT for application in real-world psychological counseling.

## 2 Methodology

This section explores the process of PsyDT framework. We first introduce multi-turn dialogues synthesis method of PsyDT, which consists of three components: Dynamic One-Shot Learning from Counseling Cases, Client Personality Simulation and Multi-Turn Mental Health Dialogues Synthesis, as shown in Figure 2. And then we fine-tune the LLMs on the synthetic dataset, PsyDTCorpus, to achieve the digital twin of psychological counselor with a specific counseling style. The specific process of synthesizing multi-turn dialogues is shown in Algorithm 1 in the Appendix.

## 2.1 Single-Turn Dialogues Preparation

To ensure the quality of the synthetic multi-turn dialogues, we select 5,000 single-turn long-text dialogues with rich presentation of clients' personality traits from the SoulChatCorpus (Chen et al., 2023) by utilizing GPT-4 as the client personality evaluator. An example of the single-turn long-text dialogues is shown in Figure 8 in the appendix. These dialogues primarily consist of four components: Counseling Topic , Title of Client's Question , Detail of Client's Question , and Counselor's Long Text Response . The distribution of counseling topics is shown in Figure 3.

Figure 3: Distribution of counseling topics.

<!-- image -->

## 2.2 Dynamic One-Shot Learning from Counseling Cases

## 2.2.1 Real-World Counseling Cases Collection

To construct psychological counselor's digital twin, we invited a psychological counselor who meets national professional standards to provide psychological counseling. We recruited volunteers from online community, who were asked to select a counseling topic suitable for themselves. Before the consultation, volunteers were informed of precautions and were required to read and sign an informed consent form. During the formal consultation session, volunteers will engage in a dialogue with the psychological counselor. The entire session will follow the standard psychological consultation model and last for 50 minutes, conducted entirely via textbased communication. After the consultation, the counselor who conducted the session will review and verify the consultation records, correcting any errors and redacting any sensitive information to protect the privacy of the participants. In addition, we employ GPT-4 to conduct security checks and data cleaning on these counseling cases. As a result, the real-world counseling cases of the specific psychological counselor on 12 topics were collected, with total number of 12 and total cost of $2000 . One of the real-world counseling cases is presented in Figure 9 in the Appendix (Counseling Topic: Relationship).

## 2.2.2 Linguistic Style and Therapy Technique Summarization

In real-world psychotherapy scenarios, different psychological counselors exhibit distinct personal styles, including unique linguistic style and therapy techniques, etc. To construct above psychological counselor's digital twin, we first employ GPT-4 to capture the linguistic style from several collected real-world counseling cases. The prompt and one example of the summarized linguistic style from counseling case of Figure 9 are illustrated in Figure 10, 11 in the appendix. Subsequently, we also employ GPT-4 to summarize the therapeutic type of collected counseling cases of specific counselor. The prompt is shown in Figure 12 in the Appendix. According to statistics, 12 counseling cases all use rational emotive behavior therapy (REBT) (Dryden, 2005) for psychological counseling. And then the therapeutic type is used to retrieve the corresponding knowledge from the therapy technique knowledge base. Some therapeutic types and details of the knowledge base are presented in Table 3, 4 in the appendix.

## 2.3 Client Personality Simulation

To ensure both complexity and diversity of clients' linguistic style in synthetic multi-turn dialogues, we employ GPT-4 to simulate the Big Five personality traits (Costa and McCrae, 1999) of clients based on their question. The Big Five personality theory encompasses five core dimensions: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism (OCEAN). The prompt and an example of simulated client personality traits from single-turn dialogue of Figure 8 are illustrated in Figure 13, 14 in the Appendix.

## 2.4 Multi-Turn Mental Health Dialogues Synthesis

In order to prevent excessive homogenization of synthesized multi-turn dialogues, we enable singleturn dialogues to dynamically match real-world counseling cases based on counseling topics. The summarized linguistic style and therapy technique with specific topic and extracted client's Big Five personality traits polymerized with counseling case with specific topic, emotion guide and response guide, GPT-4 is employed to synthesize single-turn dialogue into multi-turn dialogues. We denote synthetic multi-turn dialogues dataset as PsyDTCorpus . The prompt and an example of PsyDTCorpus from single-turn dialogue of Figure 8 are illustrated in Figure 15, 16 in the Appendix.

## 2.5 PsyDTLLM Model

Based on the synthetic multi-turn dialogues dataset, PsyDTCorpus, we construct the specific psychological counselor's digital twin LLM (PsyDTLLM) through multi-turn instruction fine-tuning (MIFT). The loss ℓ θ during training is,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where θ , ˆ y i , y i , c &lt;i denote the trainable parameters of the LLM, the i -th target utterance of counselor, the i -th predicted utterance of counselor, dialogue history with less than i turns, respectively.

## 3 Experiments

In this section, we conduct multiple comprehensive analyses of the synthesized multi-turn dialogues dataset PsyDTCorpus and the fine-tuned PsyDTLLM of psychological counselor with specific counseling style. In summary, we address the following research questions (RQs):

- RQ1: How similar is the synthetic multi-turn dialogue dataset with real-world counseling cases of specific counselor?
- RQ2: What is the performance of PsyDTCorpus as compared with baseline datasets?
- RQ3: Do the different components in the multi-turn dialogues synthesis method of PsyDT synthesize the corresponding effect of PsyDTCorpus?
- RQ4: What is the performance of PsyDTLLM as compared with baseline LLMs?

## 3.1 Analysis of PsyDTCorpus Dataset

## 3.1.1 RQ1: Similarity Analysis

To validate the effectiveness of the multi-turn dialogues synthesis method of PsyDT framework, we compare the multi-turn mental health dialogues constructed by three methods, which are PsyDT\_Prompt , SoulChat\_Prompt (Chen et al.,

Figure 4: Similarity results for the proposed multi-turn dialogue synthesis method and other baseline methods.

<!-- image -->

2023), Smile\_Prompt (Qiu et al., 2024a). For each topic, We randomly select 20 single-turn dialogue samples. Then, all the samples are used to synthesize multi-turn dialogues based on the above three methods, respectively. Given the potential of using LLMs for evaluating text generation quality (Chiang and Lee, 2023), we attempt to automatically assess the similarity between three synthetic multiturn dialogues dataset and real-world counseling cases in the matter of linguistic style and therapy technique. We employ two state-of-the-art LLMs as evaluators: GPT-4o 2 and Claude 3.5 3 . We take the average of the similarity scores given by two LLMsas the final result. The detailed content of the prompt is shown in Figure 17, 18 in the Appendix.

Figure 4 illustrates that the similarity between multi-turn dialogues synthesized with PsyDT\_Prompt and real-world counseling cases is approximately 70%, surpassing multi-turn dialogues synthesized with Smile\_Prompt and SoulChat\_Prompt , especially in therapy technique. It indicates that the multi-turn dialogue synthesis method of PsyDT has excellent alignment performance in the linguistic style and therapy technique of the specific psychological counselor, which can effectively synthesize multi-turn dialogues of counselor with specific counseling style.

## 3.1.2 RQ2: Overall Dataset Comparisons

To validate the superiority of the synthetic multiturn dialogues dataset PsyDTCorpus, we conduct a comprehensive comparison between PsyDTCor- pus and the following datasets: SMILECHAT (Qiu et al., 2024a), SoulChatCorpus (Chen et al., 2023), and CPsyCounD (Zhang et al., 2024). Detailed information regarding the employed datasets is provided in Table 1. Open. , Size , NoT. , LoC. , LoP. , EmoE. , CogE. , TheT. respectively represent opensource, dataset size, average number of turns, average length of client's response, average length of psychological counselor's response, emotional empathy 4 , cognitive empathy (Preston and De Waal, 2002), therapy technique. We randomly select 50 dialogue samples from each of these four datasets for a manual evaluation. Our evaluation team consists of four senior psychological postgraduate students and an experienced psychotherapist to ensure accuracy and professionalism. The evaluation metrics are listed in Table 5 in the Appendix. Based on references Swank and Lambie, 2012 and Association et al., 2012, we summarize four professional assessment dimensions in the field of psychological counseling: Conversation Strategy ( Con. ), State and Attitude ( Sta. ), Relationship Building ( Rel. ), and Application of Therapy Technique ( App. ). Additionally, we manually evaluate the fluency ( Flu. ) and safety ( Saf. ) of the data. The rating scale of Con. , Sta. , Rel. and App. is (0, 1, ..., 9, 10), while (0,1) for Flu., where higher score means better. Fleiss' κ (Fleiss, 1971) for Con. , Sta. , Rel. , The. and Inf. are 0.411, 0.403, 0.407, 0.435 and 0.547, indicating moderate annotation agreement respectively.

[2 https://chat.openai.com](https://chat.openai.com/)

[3 https://claude.ai/](https://claude.ai/)

As shown in Table 1, the PsyDTCorpus dataset significantly outperforms the other datasets across all four professional assessment dimensions. To visually illustrate the stability and score distribution of our multi-turn dialogues dataset compared to other datasets across these four dimensions, we design boxplots of the evaluation results. As shown in Figure 5, it is evident that PsyDTCorpus excels in quality across all four dimensions compared to the other baseline datasets. These results underscore the better superiority and professionalism of PsyDTCorpus over the other baseline datasets, indicating the strong potential of PsyDT for application in real-world psychological counseling.

4 Real-world psychological counselors not only need to consider emotional empathy but also cognitive empathy

Table 1: Dataset evaluation results. The symbol ∆ indicates the dataset will be publicly available soon. The best score for each metric is in-bold , while the second best score is underlined.

| Datasets       | Statistics   | Statistics   | Statistics   | Statistics   | Statistics   | Abilities   | Abilities   | Abilities   | Expert Evaluation   | Expert Evaluation   | Expert Evaluation   | Expert Evaluation   | Expert Evaluation   | Expert Evaluation   |
|----------------|--------------|--------------|--------------|--------------|--------------|-------------|-------------|-------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
|                | Open.        | Size         | NoT.         | LoC.         | LoP.         | EmoE.       | CogE.       | TheT.       | Con.                | Sta.                | Rel.                | App.                | Flu.                | Saf.                |
| SMILECHAT      | ✓            | 56k          | 10.4         | 26.1         | 28.9         | ✓           |             |             | 5.38                | 5.92                | 5.65                | 4.37                | 0.84                | ✓                   |
| SoulChatCorpus | ✓            | 258k         | 5.9          | 41.4         | 90.0         | ✓           |             |             | 5.24                | 5.80                | 5.62                | 4.38                | 0.86                | ✓                   |
| CPsyCounD      | ✓            | 3.1k         | 8.0          | 32.9         | 52.6         | ✓           |             | ✓           | 5.57                | 6.02                | 5.66                | 5.49                | 0.72                | ✓                   |
| PsyDTCorpus    | ∆            | 5k           | 18.1         | 31.6         | 58.1         | ✓           | ✓           | ✓           | 8.39                | 8.69                | 8.29                | 8.12                | 1.00                | ✓                   |

<!-- image -->

- (c) Relationship Building

(d)

Application of Therapy

Technique

Figure 5: Results of manual evaluation for PsyDTCorpus and baseline datasets on 4 professional dimensions.

## 3.1.3 RQ3: Ablation Studies

To validate that our synthetic multi-turn dialogues dataset, PsyDTCorpus, integrates linguistic style, therapy technique, and client personality, we design three ablation manual evaluations. Our evaluation team consists of four senior psychological postgraduate students and an experienced psychother- apist to ensure accuracy and professionalism. We randomly select 16 single-turn dialogues. Using multi-turn dialogues synthesis method of PsyDT from Figure 15, we synthesize these 16 sets of dialogues, each time excluding one of the following elements: linguistic style, therapy technique, and client personality. The evaluators are asked to choose the dialogues that best represented the corresponding linguistic style, therapy technique, and client personality between original dialogue and ablated dialogues. We took the optimal answer for each set of dialogue from the five evaluators through voting mechanism and then calculating the agreement for all 16 sets. As shown in Figure 6, our synthetic multi-turn dialogue achieved a fidelity of over 60%. This demonstrates that our multi-turn dialogues synthesis method of PsyDT framework can effectively integrate linguistic style, therapy technique, and client personality.

Figure 6: Results of ablation study on synthetic dialogues and other ablated dialogues.

<!-- image -->

## 3.2 Analysis of PsyDTLLM model

## 3.2.1 Baselines

We compare PsyDTLLM with the following baselines.

- Closed-source : ChatGPT (OpenAI, 2022; Ouyang et al., 2022); GPT-4 (OpenAI, 2024).
- Open-source : Baichuan2-7B-Chat (Yang et al., 2023); GLM4-9B-Chat (GLM et al., 2024); InternLM2-Chat-7B (Cai et al., 2024); Llama3-8B-Instruct (Touvron et al., 2023); Llama3.1-8B-Instruct (Touvron et al., 2023); Qwen2-7B-Instruct (Yang et al., 2024); Yi1.5-9B-Chat (Young et al., 2024).
- Domain-specific : MeChat (Qiu et al., 2024a); PsyChat (Qiu et al., 2024b); SoulChat (Chen et al., 2023); EmoLLM (EmoLLM, 2024); MindChat (Xin Yan, 2023); CPsyCoun (Zhang et al., 2024).

To facilitate a more accurate comparison of the capabilities of various models, we choose models of similar magnitudes, such as the 6B/7B/8B/9B model parameter sizes for comparison.

## 3.2.2 Implementation details

The Qwen2-7B-Instruct is used as the backbone. The whole implementation is based on the LLaMAFactory (Zheng et al., 2024). PsyDTLLM is finetuned for 3 epochs on the proposed PsyDTCorpus with a batch size of 2 per GPU, using 8 NVIDIA-A800-80G GPUs. The cosine-type learning rate scheduler with warmup \_ ratio = 0 . 03 and warmup \_ max \_ lr = 1 . 0 e -5 is used. 16-bit half-precision floating point numbers is utilized to accelerate training and improves model performance. The random seed is set to 42. In the inference phase, all the LLMs adopt the following configuration: temperature=0.9, top\_p=0.75, top\_k=20.

## 3.2.3 RQ4: Overall Model Comparisons

To verify the superiority of PsyDTLLM compared to other baseline models, we randomly split the PsyDTCorpus dataset into 4760 sets of training data and 240 sets of testing data. The testing data includes 12 topics, each with 20 sets of data, for a total of 4311 turns to conduct automatic evaluation. For each sample, each model generates an answer for evaluation. We used 5 evaluation metrics as automatic metrics: ROUGE-1, ROUGE-2, ROUGE-L (Lin, 2004), BLEU-4 (Papineni et al., 2002), F BERT of BERTSCORE (Zhang* et al., 2020). Generally, as shown in Table 2, PsyDTLLM outperforms other baseline models in both automatic evaluation metrics, which demonstrates that PsyDTLLM outperforms other baseline models in the semantic understanding ability.

Subsequently, we randomly extracts 20 turns of data from each topic for a total of 240 turns to conduct professional evaluation. We also employ two state-of-the-art LLMs as evaluators: GPT-4o and Claude 3.5. We take the average of the scores given by two LLMs as the final result. The metrics to assess are as follows: Emotional Empathy ( EmoE. ), Cognitive Empathy ( CogE. ), Conversation Strategy ( Con. ), State and Attitude ( Sta. ), and Safety ( Saf. ). The ranting scale of EmoE. , CogE. , Con. , Sta. is (0, 1, 2, 3) and Saf. is (0, 1), where higher score means better. The detailed content of the prompt is shown in Figure 19, 20, 21, 22, and 23 in the Appendix. Generally, as shown in Table 2, PsyDTLLM outperforms other baseline models in both professional metrics, demonstrating better performance in the psychological counseling, thereby indicating the strong potential of PsyDT for application in realworld psychological counseling.

## 3.3 Case Study

In this section, we present an close look for examples of clients seeking emotional support from ChatGPT, Qwen2-7B-Instruct, SoulChat, CPsyCounX, and PsyDTLLM via case studies. as shown in Figure 24, 25, 26, 27, and 28 in the Appendix, respectively. ChatGPT and Qwen2-7B-Instruct tend to provide advice rather than asking questions or listening. SoulChat fails to satisfy the individual need of client who seek specific counseling styles. CPsyCounX tries to explicitly use some professional therapy terms in its replies, which may cause confusion for clients who are not in the field of psychology. Our PsyDTLLM, like a real-world psychological counselor, implicitly expresses emotional empathy and cognitive empathy to meet the needs of clients who seek specific counseling styles.

## 4 Related Work

## 4.1 Mental Health LLMs

LLMs typically underperform in domains requiring complex mental acumen and high levels of empathy. Nonetheless, significant advancements have been made by researchers. For instance, Qiu et al. (2024a) introduced SMILE approach, leveraging ChatGPT to transform single-turn dialogues into multi-turn interactions. Chen et al. (2023) developed SoulChat and SoulChatCorpus, a multi-turn empathetic conversation dataset comprising over 2 million samples. By fine-tuning LLM on SoulChatCorpus, they achieved superior performance in em- pathetic dialogue tasks. Zhang et al. (2024) presented CPsyCoun, a framework for reconstructing and evaluating multi-turn dialogues in Chinese psychological counseling. To optimize the utilization of counseling reports, they proposed a two-phase method for generating high-quality dialogues. Furthermore, they established a comprehensive evaluation benchmark for automatic assessment of multiturn psychological counseling. In this paper, we propose PsyDT, a novel framework using LLMs to construct the digital twin of psychological counselor with personalized counseling style.

Table 2: Model evaluation results.

| Type   | Models               | Automatic.   | Automatic.   | Automatic.   | Automatic.   | Automatic.   | Professional.   | Professional.   | Professional.   | Professional.   | Professional.   |
|--------|----------------------|--------------|--------------|--------------|--------------|--------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| Type   | Models               | R-1          | R-2          | R-L          | B-4          | F BERT       | EmoE.           | CogE.           | Con.            | Sta.            | Saf.            |
| Closed | ChatGPT              | 31.72        | 7.77         | 24.52        | 7.24         | 96.69        | 1.70            | 1.74            | 1.88            | 1.99            | 1.00            |
| Closed | GPT-4                | 26.51        | 6.79         | 18.23        | 5.31         | 96.59        | 1.80            | 1.99            | 2.06            | 1.89            | 1.00            |
| Open   | Baichuan2-7B-Chat    | 15.40        | 3.69         | 11.84        | 3.46         | 94.14        | 1.35            | 1.34            | 1.44            | 1.49            | 1.00            |
| Open   | GLM4-9B-Chat         | 23.38        | 5.45         | 14.35        | 3.84         | 96.58        | 1.68            | 1.88            | 1.94            | 1.74            | 1.00            |
| Open   | InternLM2-Chat-7B    | 27.15        | 5.87         | 20.38        | 5.49         | 96.62        | 1.87            | 1.92            | 2.04            | 2.05            | 1.00            |
| Open   | Llama3-8B-Instruct   | 26.31        | 5.25         | 19.64        | 5.11         | 95.16        | 1.58            | 1.72            | 1.77            | 1.81            | 1.00            |
| Open   | Llama3.1-8B-Instruct | 30.20        | 5.84         | 22.88        | 5.96         | 96.54        | 1.61            | 1.70            | 1.81            | 1.90            | 1.00            |
| Open   | Qwen2-7B-Instruct    | 23.42        | 5.28         | 15.42        | 4.05         | 96.64        | 1.81            | 2.09            | 2.18            | 2.12            | 1.00            |
| Open   | Yi-1.5-9B-Chat       | 29.32        | 6.89         | 21.85        | 7.50         | 96.66        | 1.75            | 1.79            | 2.11            | 1.93            | 1.00            |
| Domain | MeChat               | 30.71        | 7.05         | 24.43        | 6.73         | 96.55        | 1.54            | 1.58            | 1.66            | 1.96            | 1.00            |
| Domain | PsyChat              | 27.96        | 5.21         | 21.44        | 4.83         | 96.19        | 1.36            | 1.40            | 1.34            | 1.79            | 1.00            |
| Domain | SoulChat             | 28.93        | 5.93         | 23.26        | 5.49         | 96.42        | 1.29            | 1.36            | 1.42            | 1.76            | 1.00            |
| Domain | MindChat             | 22.55        | 3.44         | 17.75        | 3.48         | 93.89        | 1.13            | 1.25            | 1.13            | 1.54            | 1.00            |
| Domain | EmoLLM               | 23.26        | 4.01         | 18.50        | 3.74         | 91.74        | 1.06            | 1.18            | 1.21            | 1.36            | 1.00            |
| Domain | CPsyCounX            | 23.71        | 4.32         | 17.59        | 3.59         | 95.46        | 1.28            | 1.42            | 1.54            | 1.60            | 1.00            |
| Our    | PsyDTLLM             | 36.03        | 10.08        | 28.86        | 9.99         | 96.79        | 1.90            | 2.13            | 2.19            | 2.26            | 1.00            |

## 4.2 Therapy Techniques

Psychological therapy techniques significantly contribute to individual mental well-being and overall quality of life. These methods support individuals in identifying, addressing, and managing psychological challenges and conflicts (Meier, 2010). Rational Emotive Behavior Therapy (REBT) (Dryden, 2005), developed by psychologist Albert Ellis in the 1950s, is a prominent form of psychotherapy that highlights the role of irrational beliefs in causing emotional distress and dysfunctional behavior. REBT posits that psychological disturbances are not caused by events themselves but by individuals' interpretations and reactions to these events. Cognitive Behavioral Therapy (CBT) (Beck, 1979), primarily developed by Aaron Beck in the 1960s, is a widely practiced and highly effective form of psychotherapy that focuses on the interplay between thoughts, emotions, and behaviors. CBT is recognized as one of the most prominent and well-researched approaches in the field of psychology. Dialectical Behavior Therapy (DBT) (Linehan, 2014), created by psychologist Marsha M. Linehan in the late 1980s, is a comprehensive psychotherapeutic approach initially designed for individuals with borderline personality disorder (BPD). It has since been adapted to treat various other mental health conditions, including depression, anxiety disorders, substance abuse, and eating disorders.

## 5 Conclusion

In this paper, we propose PsyDT, a novel framework using LLMs to construct the digital twin of psychological counselor with personalized counseling style. The proposed multi-turn dialogues synthesis method of PsyDT framework can quickly and cost-effectively synthesize PsyDTCorpus, a high-quality multi-turn mental health dialogues dataset of psychological counselor with specific counseling style, which closely resemble realworld counseling cases. This indicates the strong potential of PsyDT for application in real-world psychological counseling.

## Limitation and Future Work

Although the experimental results demonstrate the effectiveness of PsyDT, there are still some limitations need to consider. Psychological counseling is complex. Our framework only constructs digital twin of psychological counselor with specific counseling style, which satisfies the individual needs of clients who seek specific counseling style, but can not guarantee to solve their psychological problems and meet counseling needs of all clients. In addition, relying solely on one psychological counselor is somewhat arbitrary, we hope to continue exploring the application of multiple psychological counselors in the joint diagnosis in the field of mental health LLMs in the future.

## Ethical Statement

- Data Private: We implemented a rigorous data cleaning protocol to synthesis our dataset and ensure privacy protection (Hovy and Spruit, 2016). These measures included rulebased cleaning, manual rewriting, and human proofreading to guarantee the absence of sensitive or private content. For instance, the initial data collection contained private information from psychologists, such as personal details, contact information, addresses, and workplaces. Following the cleaning process, all such sensitive information was entirely removed, ensuring the protection of personal data. Additionally, any conversations with potential harm to clients, others, or society were thoroughly expunged from our dataset.
- Potential Risks of the Model: During the human evaluation phase, we conducted a focused safety assessment of the model's outputs. Given the absence of human feedback during the model fine-tuning phase, some responses may inevitably pose potential harm to users. If there is no noticeable improvement in user interactions with the PsyDTLLM model, we strongly recommend seeking immediate assistance from a professional counselor or psychiatrist. It is crucial to remember that virtual conversational agents cannot replace real-world therapy. Furthermore, when implementing this model in downstream applications, responses generated by the AI should be used solely as references.
- Annotator Compensation: We engaged ex-

perts in psychology to conduct a manual evaluation of the model's output. Each annotator spent approximately 3 minutes assessing each sample, for which they received a compensation of $0.418. This corresponds to an hourly wage of $8.36, which is higher than the current U.S. federal minimum wage of $7.25 per hour.

## References

- American Psychological Association et al. 2012. Competency benchmarks in professional psychology.
- Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, et al. 2023. Qwen technical report. Preprint , arXiv:2309.16609.
- Aaron T Beck. 1979. Cognitive therapy of depression . Guilford press.
- Zheng Cai, Maosong Cao, Haojiong Chen, Kai Chen, Keyu Chen, et al. 2024. Internlm2 technical report. Preprint , arXiv:2403.17297.
- Yirong Chen, Xiaofen Xing, Jingkai Lin, Huimin Zheng, Zhenyu Wang, Qi Liu, and Xiangmin Xu. 2023. SoulChat: Improving LLMs' empathy, listening, and comfort abilities through fine-tuning with multi-turn empathy conversations. In Findings of the Association for Computational Linguistics: EMNLP 2023 , pages 1170-1183, Singapore. Association for Computational Linguistics.
- Cheng-Han Chiang and Hung-yi Lee. 2023. Can large language models be an alternative to human evaluations? In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , Toronto, Canada. Association for Computational Linguistics.
- Paul T Costa and Robert R McCrae. 1999. A fivefactor theory of personality. The five-factor model of personality: Theoretical perspectives , 2:51-87.
- Windy Dryden. 2005. Rational emotive behavior therapy. Encyclopedia of cognitive behavior therapy , pages 321-324.
- Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. 2022. GLM: General language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 320-335, Dublin, Ireland. Association for Computational Linguistics.

EmoLLM. 2024. Emollm.

- Joseph L Fleiss. 1971. Measuring nominal scale agreement among many raters. Psychological Bulletin , 76(5):378-382.
- Team GLM, :, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, et al. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. Preprint , arXiv:2406.12793.
- Dirk Hovy and Shannon L. Spruit. 2016. The social impact of natural language processing. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) , pages 591-598, Berlin, Germany. Association for Computational Linguistics.
- Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out , pages 74-81, Barcelona, Spain. Association for Computational Linguistics.
- Marsha Linehan. 2014. DBT? Skills training manual . Guilford Publications.
- Augustine Meier. 2010. Counselling and therapy techniques: Theory &amp; practice.
- [OpenAI. 2022. Introducing chatgpt.](https://openai.com/blog/chatgpt)
- OpenAI. 2024. Gpt-4 technical report. Preprint , arXiv:2303.08774.
- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, et al. 2022. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems , volume 35, pages 27730-27744. Curran Associates, Inc.
- Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics , pages 311-318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.
- Stephanie D Preston and Frans BM De Waal. 2002. Empathy: Its ultimate and proximate bases. Behavioral and brain sciences , 25(1):1-20.
- Huachuan Qiu, Hongliang He, Shuai Zhang, Anqi Li, and Zhenzhong Lan. 2024a. SMILE: Single-turn to multi-turn inclusive language expansion via ChatGPT for mental health support. In Findings of the Association for Computational Linguistics: EMNLP 2024 , pages 615-636, Miami, Florida, USA. Association for Computational Linguistics.
- Huachuan Qiu, Anqi Li, Lizhi Ma, and Zhenzhong Lan. 2024b. Psychat: A client-centric dialogue system for mental health support. In 2024 27th International Conference on Computer Supported Cooperative Work in Design (CSCWD) , pages 2979-2984.
- Damian F Santomauro, Ana M Mantilla Herrera, Jamileh Shadid, Peng Zheng, Charlie Ashbaugh, et al. 2021. Global prevalence and burden of depressive and anxiety disorders in 204 countries and territories in 2020 due to the covid-19 pandemic. The Lancet , 398(10312):1700-1712.
- Jacqueline M Swank and Glenn W Lambie. 2012. The assessment of cacrep core curricular areas and student learning outcomes using the counseling competencies scale. Counseling Outcome Research and Evaluation , 3(2):116-127.
- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. Preprint , arXiv:2302.13971.
- Dong Xue* Xin Yan. 2023. Mindchat: Psychological large language model. https://github.com/ X-D-Lab/MindChat .
- Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, et al. 2023. Baichuan 2: Open large-scale language models. Preprint , arXiv:2309.10305.
- An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, et al. 2024. Qwen2 technical report. Preprint , arXiv:2407.10671.
- Alex Young, Bei Chen, Chao Li, Chengen Huang, et al. 2024. Yi: Open foundation models by 01.ai. Preprint , arXiv:2403.04652.
- Chenhao Zhang, Renhao Li, Minghuan Tan, Min Yang, Jingwei Zhu, Di Yang, Jiahao Zhao, Guancheng Ye, Chengming Li, and Xiping Hu. 2024. CPsyCoun: A report-based multi-turn dialogue reconstruction and evaluation framework for Chinese psychological counseling. In Findings of the Association for Computational Linguistics ACL 2024 , pages 1394713966, Bangkok, Thailand and virtual meeting. Association for Computational Linguistics.
- Tianyi Zhang*, Varsha Kishore*, Felix Wu*, Kilian Q. Weinberger, and Yoav Artzi. 2020. bertbertscore: Evaluating text generation with bert. In International Conference on Learning Representations .
- Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. 2024. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations) , Bangkok, Thailand. Association for Computational Linguistics.

## A Reproducibility Checklist

- Model and Data: The PsyDTCorpus dataset and PsyDTLLM model will be released upon decision of the paper.
- System Hardware: We train the PsyDTLLM on the Ubuntu 20.04.6 LTS server that has 2 CPUs called "Intel(R) Xeon(R) Platinum 8358P CPU @ 2.60GHz", 8 NVIDIA A800SXM4-80GB GPUs, and 1,024GB memory.
- Driver Version: The version of Nvidia driver is "525.105.17". The version of CUDA is "24.1.2"
- Package version: python=3.8, torch 5 =2.3.0, transformers 6 =4.43.0, deepspeed 7 =0.14.0, datasets=2.18.0 and jieba=0.42.1 are recommended.

## B Results of Model Automatic Evaluation

The detailed scores of 4 semantic metrics of model automatic evaluation are as shown in Figure 7.

[5 https://pytorch.org/get-started/ previous-versions](https://pytorch.org/get-started/previous-versions)

[6 https://github.com/huggingface/ transformers](https://github.com/huggingface/transformers)

[7 https://github.com/microsoft/ DeepSpeed](https://github.com/microsoft/DeepSpeed)

Figure 7: ROUGE and BLEU results of model automatic evaluation on our PsyDTLLM model and other baseline models in 12 counseling topics.

<!-- image -->

## Single-Turn Dialogue in Chinese

'咨询话题'：人际

- '来访者问题主题'：敏感多疑，一听到她们就 会影响自己的情绪，无助痛苦？

'来访者问题细节'：我之前有一个从初中玩到 高中也是我唯一最好的朋友A，高二时候又有一 个女生B和我们一起玩，她们之间有共同话题， 平时一起走，B很爱站中间，于是一路上我基本 不说话，B说话很毒，可能她是开玩笑的，总之 她的言语伤害到了我，我的别班朋友C就去骂了 她，于是我与B绝交了。但是后来我发现我如果 不去找A，A就不会主动来找我，我们的友谊都是 靠我维持我感觉很累，没绝交前A说生日会叫我 和B，绝交后慢慢到了A生日那天，A叫了B和B的 朋友们，A连电话都没有给我打，我很伤心。在 班级我只和A玩后来我变得没有朋友。高三后换 了新同桌好了一点，可是还是很敏感，一听到A 和B笑的很开心，我就会处于很低落的情绪，我 该怎么办，我知道学业为重，可是我没办法控制 自己一听到她们就会影响自己的情绪，很无助痛 苦。

'咨询师长文本回复'：你好，我很理解你目前 的感受。你与朋友A之间的友情受到了其他人的 干扰，让你感到无助和痛苦。这种情况下，你可 能会感到被抛弃，以及对于友情的失落和伤感。 对于A的行为，你感到很困惑和伤心，这些情绪 都是非常正常的。在这种情况下，你的自尊和自 信可能会受到挑战，这也会影响到你的学业和生 活。\n\n在这个过程中，你可能也会对A产生怀 疑和不信任的情绪。我建议你可以尝试跟A进行 一次单独的交流，表达你的感受和困惑。也许她 并不是故意要伤害你，而是在处理人际关系时产 生了困扰。通过坦诚的对话，也许可以解开一些 误会，或者找到一种更好的处理方式。\n\n同时， 我明白这段经历会让你感到疲惫和沮丧。在这个 过程中，保持对自己的关爱和呵护是非常重要的。 你可以尝试关注一些自己喜欢的事情，找到一些 能够让自己放松的方式，比如运动、阅读、绘画 等等。通过这些方式，你可以慢慢地调整自己的 情绪状态，让自己更加平静和坚强地面对这段经 历。\n\n最后，我希望你知道，这段经历并不会 一直影响你的生活。每个人都会在人际关系中遇 到挑战，这也是成长的一部分。相信自己，你一 定可以度过这个难关，也会遇到更多真正懂得珍 惜你的朋友。如果你觉得情绪上的困扰影响到了 你的日常生活，也可以考虑寻求专业心理咨询师 的帮助，他们会给予你更专业的支持和指导。

## Single-Turn Dialogue in English

- 'Counseling Topic': Relationship

'Title of Client's Question': Interpersonal Sensitivity and Emotional Distress 'Detail of Client's Question': I previously had a best friend, A, whom I had known since middle school, and she was the only friend I had. In high school, another girl, B, joined our group. B and A shared common interests, and they often walked together, leaving me mostly silent during these moments. B's remarks, though perhaps intended as jokes, were often hurtful. Eventually, my friend C, from another class, confronted B on my behalf, which led to my falling out with her. Over time, I realized that if I didn't reach out to A, she wouldn't initiate contact, and it felt like I was the one keeping our friendship alive. This was draining. Before we parted ways, A had invited both B and me to her birthday, but after the fallout, A invited B and her friends without even calling me. I felt deeply hurt. In class, I only interacted with A, and as a result, I ended up isolated, with no other friends. Later, after switching desks, I became less lonely, but I still felt very sensitive. Hearing A and B laugh together would instantly depress me, and I found myself unable to control my emotions. I feel helpless and distressed. Although I understand the importance of focusing on my studies, I cannot stop how I feel whenever I hear them; it's overwhelming and painful.

'Counselor's Long-Text Response': Hello, I can deeply empathize with the feelings you're experiencing. The dynamics in your friendship with A, especially the interference caused by B, have left you feeling helpless and distressed. It is completely natural to feel a sense of abandonment and loss when a close friendship begins to deteriorate. Your confusion and hurt regarding A's actions are understandable. These emotions may challenge your sense of self-worth and confidence, which can, in turn, affect both your academic and personal life. It seems that this experience has also led to feelings of doubt and mistrust towards A. I would suggest considering the possibility of having a direct conversation with her, where you can openly share your feelings and concerns. It is possible that A did not intend to hurt you and may have found herself uncertain about how to navigate the complex dynamics in your friendship. A candid dialogue might help clear up misunderstandings or offer a better way to resolve these issues. At the same time, I recognize how emotionally exhausting and disheartening this situation has been for you. In times like these, it is crucial to show kindness and care towards yourself. I encourage you to engage in activities that you enjoy and that can help alleviate your stress, such as exercising, reading, or engaging in a creative hobby like painting. These activities may help you regain emotional balance, allowing you to approach your situation with a clearer and more resilient mindset. Lastly, please remember that this difficult chapter will not define your life. Everyone encounters challenges in relationships, and these experiences are part of personal growth. Trust in your ability to overcome this, and believe that you will find friends who truly appreciate and value you. If you feel that these emotional struggles are affecting your day-to-day life significantly, seeking guidance from a professional therapist could provide you with more structured support and strategies to cope.

Figure 8: An example of the single-turn long-text dialogues.

## Dialogue in Chinese

来访者：我现在在班级里担任班长一职，在处理班 级事务的过程中感觉到同学之间很冷淡，对于我组 织的一些活动，同学们的参与度普遍不高，让我很 苦恼。

咨询师：您的预期是大家都来参与，但实际结果是 同学们参与度不高，是这样吗？您说的同学之间很 冷淡，是指的同学们不爱彼此交往，还是只对组织 的活动冷淡？

来访者：我的预期倒也不是希望班里的同学都来， 因为上了研究生了大家有自己的科研压力，无暇参 加班级活动倒也正常，只是我后面听说很多人其实 那天并没有科研任务，只是单纯不想参与班级活动 而已。

咨询师：我们组织了活动，肯定是希望大家都来参 与。辛辛苦苦付出了，没有被看到，心里会有些不 舒服。

来访者：就是他们觉得班级里同学之间的交流是没 什么必要的，他们只在实验室自己的小团体活跃， 跟大班级的交流感觉像无效社交。

咨询师：因为大家没来参与，这件事，对您的影响 大吗？

来访者：是的呀，那时候辛辛苦苦组织了一场活动， 还拉了个群，我们班总共有60个人，进群有40个人， 最后只来了15个人，让我感到很失望。

咨询师：之前有过类似经历吗？比如大学期间，组 织活动，来参加的人远远小于预期。您提到'很失 望'，看得出来，您是一位责任心很强的人，因为 有强烈的责任心，所以对活动的参与人数也有期待。

来访者：本科的时候还好，班级的同学还是很团结 的，记得当时班级破冰30个人才一两个人没来，当 时玩得还是挺开心的，上了研究生之后感觉大家都 很冷漠。我作为班长，还是希望能够提升班级的凝 聚力，不会做到以后大家都不认识我这个班长，大 家彼此相互不认识的现象，如果这样的话会让我感 到我很失职。

咨询师：明白了，同学们没来参与活动，会让您感 觉到，是您哪些地方没有做到位，自己失职了。有 没有这种可能性：您已经把活动准备的很精细到位， 只是大家因为年龄、兴趣、科研任务等原因，很少 人来参与活动？

来访者：可能是吧，因为那次活动是一次桌游活动， 首先会有一个自我介绍环节，就是想让大家相互熟 悉，然后有后面的娱乐活动。

咨询师：本来您是想通过这场活动，组织大家相互 认识。看得出来，您确实动了一番脑筋，精心策划 了的，这种情况下，来的人不到预期的一半，确实 很失望。

来访者：后面我猜测大家是不是喜欢球类运动多一 些，也拉了一个羽毛球群，但是进群的人很多，很 多次我在上面喊大家来打羽毛球，但是只有三四个 人来打，群里只有我一个人的独角戏。

咨询师：有您做大家的班长，感觉他们挺幸运的，

即便错过了上一场活动，大概率还会有下一场。

## Dialogue in English

Client: I currently serve as the class president. However, I've noticed a general sense of indifference among my classmates when it comes to participating in class activities. Their lack of engagement has been particularly frustrating for me.

Counselor: It seems you expected greater participation, but the actual turnout has been disappointing. When you mention "indifference," do you mean that your classmates are generally unengaged with one another, or is it specifically related to the activities you organize?

Client: My expectation wasn't that everyone would participate, especially since we're now in graduate school, where research demands can be overwhelming. It's understandable if some people don't attend class activities. However, I later heard that many classmates didn't have any research commitments that day-they simply chose not to participate. Counselor: Organizing an event naturally comes with the hope that many will attend. It can be disheartening when your efforts aren't recognized or appreciated.

Client: It seems that many of them view interactions within the class as unnecessary. They are more active within their smaller lab groups and see engagement with the larger class as a form of ineffective socializing. Counselor: Did the low attendance have a significant impact on you? Client: Yes, it did. I put a lot of effort into organizing the event, even creating a group chat for it. Out of the 60 classmates, 40 joined the chat, but only 15 attended the event. I was very disappointed. Counselor: Have you experienced similar situations before? For example, during your undergraduate years, did you organize events where attendance fell short of expectations? You mentioned feeling "disappointed," which suggests you have a strong sense of responsibility. This likely contributes to your high expectations for event participation.

Client: During my undergraduate years, things were different. My classmates were more united. I recall organizing an ice-breaking event where only one or two out of 30 people didn't show up. It was a lot of fun back then. But in graduate school, the atmosphere feels much colder. As class president, I want to strengthen our class cohesion. It would be disheartening if, by the end of our studies, everyone barely knows who I am, or worse, if they don't know each other at all. That would make me feel like I've failed in my role.

Counselor: I understand. When your classmates don't participate, it makes you question whether you've done enough in your role. Is it possible that you've organized the event meticulously, but due to factors like age, differing interests, and research obligations, many didn't feel compelled to join?

Client: That could be the case. The event I organized was a board game session, beginning with introductions to help everyone get to know each other, followed by the games themselves.

Counselor: It's clear that you put a lot of thought into organizing the event with the aim of fostering connections among your classmates. Given that less than half of the expected attendees showed up, it's understandable that you'd feel disappointed.

Client: Later on, I thought they might prefer sports, so I created a badminton group chat. Although many joined the group, when I invited them to play, only three or four people ever showed up. It felt like I was just talking to myself.

Counselor: Your classmates are fortunate to have you as their class president. Even if they missed last event, there's a good chance you'll organize another.

Figure 9: An example of real-world counseling cases of psychological counselor with specific counseling style.

<!-- image -->

## Prompt in Chinese

## Prompt in English

Figure 10: The prompt used for summarizing linguistic style of real-world counseling cases.

<!-- image -->

## Summarization in Chinese

## Summarization in English

Figure 11: An example of summarized linguistic style of counseling case.

<!-- image -->

## Prompt in Chinese

## Prompt in English

Figure 12: The prompt used for summarizing therapeutic type of real-world counseling cases.

<!-- image -->

Table 3: Some common therapeutic types

| School of Thought    | Therapeutic Type                                                                                                            |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Psychodynamic        | Psychoanalysis Object-Relations Theory                                                                                      |
| Cognitive-Behavioral | Systematic Desensitization Cognitive Behavioral Therapy Rational Emotive Behavior Therapy Acceptance and Commitment Therapy |
| Humanistic           | Person-Centered Therapy Existential Therapy                                                                                 |
| Postmodern           | Solution-Focused Therapy Narrative Therapy                                                                                  |

Table 4: Knowledge base of some common therapy techniques

| Therapy Technique                 | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Rational Emotive Behavior Therapy | REBT involves several stages. Below are the dialogue stages and a brief description of the focus of each stage: (1) Examine Irrational Beliefs and Self-Defeating Thoughts : In REBT, cognitive intervention is viewed as the 'lifeblood' of treatment. Thus, almost from the beginning of therapy, during the problem exploration phase, the counselor actively and persuasively helps the client explore the reasons behind their emotional distress. This includes under- standing the client's thought process regarding events, the antecedents, and consequences of their emotions to clarify the problem. The counselor firmly encourages the client to reflect on what they 'tell' themselves when feeling anxious, depressed, or angry after encountering a stimulus event. (2) Debate Irrational Beliefs : The counselor uses various techniques (primarily cog- nitive techniques) to help the client question and challenge their irrational beliefs and thoughts, demonstrating their unrealistic and unreasonable aspects. The goal is for the client to recognize the harm of these beliefs and develop a desire and behavior to abandon them. (3) Develop Rational Beliefs and Learn Rational Thinking : After identifying and refuting irrational beliefs, the counselor further guides the client to find appropriate, rational responses to triggering situations and events. The counselor helps the client replace irrational beliefs and self-defeating thoughts with rational beliefs and objective, problem-solving thinking statements. To reinforce rational beliefs, the counselor repeat- edly teaches the client why rational beliefs are reasonable, how they differ from irrational beliefs, why irrational beliefs lead to emotional disturbances, and why rational beliefs lead to more positive, healthy outcomes. (4) Apply Therapeutic Gains in Real Life : The counselor encourages the client to internalize the objective, realistic attitudes and scientifically rational thinking methods learned in therapy and to persistently apply them to solve new problems in their future life. |

## Prompt in Chinese

## Prompt in English

Figure 13: The prompt used for simulating client personality traits.

<!-- image -->

## Personality in Chinese

（1）开放性：来访者可能在开放性上得分不高。 她似乎在遇到新朋友B的时候，没有表现出强烈 的新事物接受度或者对不同类型的友谊的好奇。 她更倾向于与熟悉的朋友A保持关系，而不是尝 试与B建立新的联系。同时，来访者对B的开玩笑 可能没能很好地适应或理解，这表明了她对新奇 或具有挑战性的社交互动的适应能力可能较低。 （2）责任心：来访者在责任心上可能表现中等。 在描述中，她没有明显显示出组织性或自律性的 问题，也没有涉及任何与勤勉或规则遵守直接相 关的行为。她确实尽力维持与朋友A的友谊，这 显示了一定程度的承诺和努力。然而，由于没有 更多信息，对于她的责任心水平难以作出更具体 的判断。

- （3）外向性：来访者似乎在外向性上得分较低。 她在社交场合中表现出来的行为和感受显示，她 可能更喜欢独处或仅与很亲近的朋友互动。她描 述了在A和B一起走的时候自己不说话的情况，这 可能暗示她在社交活动中更保守和被动。她也表 示失去了朋友A之后变得没有朋友，可能显示了 对于建立新的社交联系的不积极。
- （4）宜人性：来访者在宜人性方面可能表现中 等或偏低。她对朋友B的行为感到受伤害，并没 有选择直接和B沟通，而是通过另一个朋友C来解 决冲突，这可能表示她在处理人际关系的冲突时 可能不够直接。她也体现出对A的友谊感到失望， 因为A在不经她同意的情况下改变了他们的关系 动态，这表明她对友好行为的期待可能较高，但 在直接的社交互动中可能比较被动。

（5）神经质：来访者在神经质上可能得分偏高。 她提到听到A和B开心地笑时感到低落，表明她的 情绪受到周围人的影响较大。她表述自己感到 '无助痛苦'，这些都是情绪稳定性低的典型迹 象。此外，她提到无法控制听到她们时的情绪反 应，这表明她可能经历了频繁的情绪波动，并且 难以管理自己的情绪状态。

## Personality in English

1. **Openness**: The client may score lower in openness. She does not seem to exhibit a strong receptiveness to new experiences or curiosity about different types of friendships when meeting a new acquaintance, B. Instead, she appears to prefer maintaining her relationship with the familiar friend, A, rather than exploring a new connection with B. Furthermore, the client seems to have difficulty adapting to or understanding B's sense of humor, suggesting a lower adaptability to novel or challenging social interactions.
2. **Conscientiousness**: The client's conscientiousness appears to be moderate. In her narrative, there is no clear indication of issues with organization or self-discipline, nor does she discuss behaviors related to diligence or rule-following. However, her efforts to maintain the friendship with A demonstrate a certain level of commitment and effort. Without further information, it is difficult to make a more definitive assessment of her conscientiousness.
3. **Extraversion**: The client seems to score low in extraversion. Her behavior and feelings in social settings suggest a preference for solitude or interaction primarily with close friends. For instance, she describes a situation where she remained silent while walking with A and B, which may indicate a more reserved and passive approach to social activities. Additionally, her statement about becoming friendless after losing A suggests a lack of initiative in establishing new social connections.
4. **Agreeableness**: The client may score moderately or low on agreeableness. She expressed feeling hurt by B's actions but did not choose to address the issue directly with B, opting instead to communicate through another friend, C. This could indicate a tendency to avoid direct confrontation in interpersonal conflicts. Additionally, her disappointment with A, who altered their relationship dynamics without her consent, suggests that she has high expectations for friendly behavior but may be more passive in direct social interactions.
5. **Neuroticism**: The client may score high in neuroticism. She mentioned feeling down when hearing A and B laughing together, indicating that her emotions are significantly influenced by the mood of those around her. Her statement about feeling "helpless and distressed" reflects typical signs of low emotional stability. Moreover, her inability to control her emotional reactions when hearing A and B suggests frequent emotional fluctuations and difficulty managing her emotional state.

Figure 14: An example of simulated client personality traits.

## Prompt in Chinese

## Prompt in English

Figure 15: The prompt used for synthesizing multi-turn mental health dialogues.

<!-- image -->

## Dialogue in Chinese

来访者：我感觉最近自己情绪特别不稳定，一点小事 就能让我烦躁一整天。

咨询师：听起来您最近遇到了一些让您情绪起伏的事 情。要不您先跟我说说，是什么小事会影响到您的情 绪呢？

来访者：就是我和我从小一起长大的朋友A的关系， 现在好像变了，让我很不舒服。 显然让您感觉很难受。您能否分享一下发生了什么事

咨询师：您跟朋友A的关系遭遇了一些变化，这件事 情？

来访者：以前我和A是最好的朋友，后来班上来了个 新同学B，A和B很聊得来，我就慢慢地感觉被排除在 外了。

咨询师：您感觉您和A的友情因为B的到来而变得不同 了，这让您感到被边缘化了是吗？这种感觉肯定不好 受。

来访者：是的，有次我们一起走路，B老是站在中间， 我就被挤到后面去，整路上我都没怎么说话。 咨询师：您感到自己被物理上和情感上都被隔离了， 很自然地，这让您感觉被孤立。那时候您是怎么应对 这种情况的呢？

来访者：我没有当场说什么，我把这件事告诉了另一 个朋友C，她就去骂了B。从那以后，我就没再和B说 过话。

咨询师：您选择通过朋友C来表达自己的不满，这导 致了您和B的关系破裂。这件事后来对您有什么影响 呢？

来访者：我发现如果我不去找A，她就不会来找我。 我们的友谊感觉都是我一个人在维持，我感觉很累。 咨询师：您感觉自己一直在单方面地努力维持着与A 的友谊，这种不平等的努力让您感到筋疲力尽。这对 您来说是个重大的心理负担吧。

来访者：对，后来A生日那天，她没叫我，却叫了B和 B的朋友们，我连个电话都没收到，我真的好伤心。 咨询师：您因为被忽略而感到伤心，那是非常合理的 感受。您是否和A谈过这个问题，表达了您的感受呢？

来访者：没有，我不知道该怎么开始，怕让事情变得 更糟。

咨询师：我理解您害怕事情变得更加尴尬的担心。不 过有时候，直接沟通可以帮助我们澄清误解，您认为 呢？

来访者：我觉得这很难，因为自从那以后我们就没怎 么说过话。而且每当我听到A和B在一起笑的很开心， 我就觉得自己好失败。

咨询师：您感受到了被遗弃的情绪，听到他们的笑声 时会提醒您那种不愉快的感觉。这时候的您肯定感觉 很无助，对吧？

## Dialogue in English

Client: Recently, I've been feeling really emotionally unstable. Even the smallest things can make me irritated for the whole day.

Counselor: It seems like you've been encountering situations recently that have caused fluctuations in your emotions. Could you tell me more about the kinds of things that are affecting you?

Client: It's about my relationship with my childhood friend, A. It feels like things have changed, and it's making me uncomfortable. Counselor: It sounds like there have been some changes in your relationship with A, which have understandably made you feel upset. Could you share more about what has happened?

Client: We used to be best friends, but then a new student, B, joined our class. A and B got along really well, and I gradually felt excluded. Counselor: You're feeling that your friendship with A has changed because of B's arrival, and that sense of exclusion is really difficult. Is that how you would describe it?

Client: Yes. There was this one time we were walking together, and B kept standing in the middle, pushing me to the back. I hardly said anything the entire time.

Counselor: You felt physically and emotionally isolated during that moment, which understandably made you feel excluded. How did you handle that situation at the time?

Client: I didn't say anything at that moment. I told another friend, C, about it, and she ended up confronting B. Since then, I haven't spoken to B at all.

Counselor: You chose to express your dissatisfaction through C, which led to a breakdown in your relationship with B. How did that impact you afterwards?

Client: I realized that if I don't reach out to A, she won't contact me. It feels like I'm the only one trying to maintain our friendship, and it's exhausting.

Counselor: You're feeling that you're the only one making an effort to sustain the friendship, and this one-sided effort is draining you. This must be a heavy emotional burden for you.

Client: Yes. And on A's birthday, she didn't invite me, but she invited B and B's friends. I didn't even get a phone call. It really hurt. Counselor: Feeling left out like that is understandably painful. Have you talked to A about how you're feeling?

Client: No, I don't know how to start the conversation. I'm afraid it'll make things worse.

Counselor: I understand your fear of making things more awkward. However, sometimes direct communication can help clarify misunderstandings. What do you think?

Client: I think it's difficult, especially since we haven't talked much since then. And every time I hear A and B laughing together, I feel like a failure.

Counselor: It seems like hearing them laugh brings back those feelings of abandonment. That must make you feel really helpless, doesn't it?

<!-- image -->

| 来访者：是的，我知道我应该集中精力在学业上，但 我的情绪完全不受控制。 咨询师：您意识到学业的重要性，但不受控制的情绪 让您感到很无助。我们一起来探讨一下，您可以做些 什么来管理自己的情绪？       | Client: Yes, I know I should focus on my studies, but my emotions feel completely out of control. Counselor: You recognize the importance of focusing on your studies, but your emotions are making it hard for you to stay grounded. Let's explore together how you might manage your emotions more effectively.                   |
|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 来访者：我尝试过听音乐和画画分散注意力，但只要 一停下来，那些消极的想法就会回来。 咨询师：您已经开始通过积极的方式尝试管理您的情 绪，这是非常好的。您觉得持续地从事这些活动能帮 助您长期地调整情绪吗？ | Client: I've tried listening to music and drawing to distract myself, but as soon as I stop, those negative thoughts come back. Counselor: It's great that you've been trying positive ways to manage your emotions. Do you think engaging in these activities more regularly could help you adjust your emotional state over time? |
| 来访者：可能会有点帮助，但我还是需要一些解决问 题的方法。 咨询师：的确，除了日常的情绪调节，解决根本问题 也是很重要的。您觉得自己现在准备好去尝试和A沟 通了吗？                    | Client: It might help a little, but I still need strategies to address the core issue. Counselor: You're right-while emotional regulation is important, addressing the underlying issue is crucial as well. Do you feel ready to try communicating with A?                                                                          |
| 来访者：我不知道…我怕我说错话，我们的友谊就此 结束。 咨询师：您担心表达自己的真实感受会破坏您和A的 关系。有没有一种方式，您觉得可以安全地表达您的 担忧和感受？                    | Client: I don't know… I'm afraid that if I say the wrong thing, our friendship will end for good. Counselor: You're worried that expressing your true feelings might destroy the friendship with A. Do you think there's a way you could express your concerns and feelings in a safe manner?                                       |
| 来访者：或许我可以先写一封信，把我的感受写下来。 咨询师：写信是一个很好的方式，既可以帮助您理清 自己的思绪，也可以减少面对面交流时可能出现的紧 张感。您打算何时动笔呢？                 | Client: Maybe I could write a letter to her and explain how I feel. Counselor: Writing a letter is an excellent idea. It allows you to organize your thoughts and reduces the anxiety that can come with direct conversation. When do you think you'll be ready to write it?                                                        |
| 来访者：我想这周末写。我不知道A会怎么反应，但 至少我尝试了。 咨询师：这是一个积极的步骤，您愿意采取行动来改 变当前的局面。写完后您愿意和我分享她的反应吗？                       | Client: I think I'll try writing it this weekend. I'm not sure how A will react, but at least I'll have tried. Counselor: That's a positive step forward, and it shows you're willing to take action to change the current situation. After you write the letter, would you be open to sharing with me how A responds?              |
| 来访者：可以的，我会告诉你的。谢谢你给了我这个 建议。 咨询师：不客气，我很高兴能帮助到您。记得在整个 过程中，保持对自己的关爱和照顾，毕竟您的幸福和 情绪平衡是最重要的。                | Client: Yes, I'll let you know. Thank you for suggesting this. Counselor: You're very welcome. I'm glad to be able to support you. Remember to take care of yourself throughout this process, as your well- being is the most important thing.                                                                                      |
| 来访者：嗯，我会的。再次谢谢你，感觉心里舒服多 了。 咨询师：很荣幸我能帮到您。如果您需要更多的支持， 我在这里。祝您周末愉快，也祝您在写信时能感到心 情平和。                      | Client: I will. Thanks again; I feel much better now. Counselor: It's been my honor to help. If you need further support, I'm here for you. I hope you have a peaceful weekend, and I wish you calm and clarity when writing your letter.                                                                                           |

Figure 16: An example of PsyDTCorpus. The red segments represent client personality from Figure 14, the blue segments indicate the linguistic style from Figure 11, and the green segments correspond to the application of therapy techniques.

Figure 17: The prompt used for evaluating linguistic style similarity between dialogues.

<!-- image -->

Figure 18: The prompt used for evaluating therapy technique similarity between dialogues.

<!-- image -->

Table 5: Evaluation Metrics.

| Dimension                        | Ability                                      | Description                                                                                                                                                                                                                                                                                                                      |
|----------------------------------|----------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Conversation Strategy            | Inquiry and Questioning Feedback and Summary | The questions posed by the counselor facilitate deeper reflection and self- exploration by the client, encouraging dialogue and thorough discussion. During sessions, the counselor effec- tively provides feedback and summa- rizes key points when necessary, ensur- ing mutual understanding and alignment in the discussion. |
| Conversation Strategy            | Problem Solving and Guidance                 | The counselor guides clients in problem- solving, encouraging autonomous reflec- tion and self-discovery rather than pro- viding direct solutions.                                                                                                                                                                               |
| State and Attitude               | Openness and Value Neutrality                | The counselor approaches the client's opinions, feelings, and experiences with an open and non-judgmental attitude, refraining from immediate value judg- ments or excessive persuasion, allow- ing the client to make independent deci- sions.                                                                                  |
| State and Attitude               | Emotional Control                            | The counselor maintains professional emotional regulation throughout the counseling process, neither over- whelmed by the client's emotions nor appearing overly detached.                                                                                                                                                       |
| Relationship Building            | Relationship Building                        | The counselor establishes and maintains a positive relationship with the client, characterized by trust, warmth, and un- derstanding.                                                                                                                                                                                            |
| Application of Therapy Technique | Application of Therapy Technique             | The counselor effectively applies strate- gies based on one or more theoretical frameworks to guide the client through problem resolution during the dialogue.                                                                                                                                                                   |

Figure 19: The prompt used for evaluating emotional empathy of LLMs.

<!-- image -->

Figure 20: The prompt used for evaluating cognitive empathy of LLMs.

<!-- image -->

Figure 21: The prompt used for evaluating conversation strategy of LLMs.

<!-- image -->

Figure 22: The prompt used for evaluating state and attitude of LLMs.

<!-- image -->

Figure 23: The prompt used for evaluating safety of LLMs.

<!-- image -->

<!-- image -->

我最近总觉得自己很焦虑，身心俱疲。不知道该怎么办。 I have been feeling anxious and physically and mentally exhausted lately. I don't know what to do.

焦虑是一种常见的情绪，但是如果持续感到焦虑并且影响到日常生活，就需要采取一些措施来应对。首先， 可以尝试一些放松的方法，比如深呼吸、冥想、瑜伽或者按摩等，来缓解身心的紧张感。同时，也可以试着 找一些能够让自己放松的活动，比如看书、散步、听音乐等。另外，与朋友或家人交流也是一种缓解焦虑的 方式，可以帮助分担一些压力。如果以上方法都无法缓解焦虑，建议寻求专业的帮助，比如心理咨询师或医 生的建议。记住，关注自己的身心健康是非常重要的。

Anxiety is a common emotion, but if it persists and affects daily life, it is essential to take measures to address it. Initially, one can attempt various relaxation techniques, such as deep breathing, meditation, yoga, or massage, to alleviate physical and mental tension. Additionally, engaging in activities that promote relaxation, such as reading, walking, or listening to music, can be beneficial. Communicating with friends or family members is another effective way to reduce anxiety, as it helps to share and lighten the emotional burden. If these methods do not alleviate anxiety, it is advisable to seek professional assistance, such as consulting a psychologist or physician. Remember, prioritizing your mental and physical health is of utmost importance.

Figure 24: A case of a client confiding to ChatGPT .

我最近总觉得自己很焦虑，身心俱疲。不知道该怎么办。 I have been feeling anxious and physically and mentally exhausted lately. I don't know what to do.

<!-- image -->

感到焦虑和身心疲惫是很常见的情绪体验，尤其是在面对生活压力、工作压力或是个人挑战时。以下是 一些建议，希望能帮助你缓解焦虑感：

1. 定期运动：体育活动是减轻压力和焦虑的有效方式之一。无论是散步、跑步、瑜伽还是任何你喜欢 的运动，都能帮助释放内啡肽，提升心情。
2. 保持良好的作息时间：确保每天有足够的睡眠，并尽量保持规律的作息时间。良好的睡眠质量对身 心健康至关重要。
3. 练习冥想和深呼吸：这些技巧可以帮助你放松身心，减少焦虑感。每天花几分钟进行冥想或深呼吸 练习，有助于提高专注力和情绪稳定性。
4. 健康饮食：均衡的饮食对于维持身体和心理健康都非常重要。尽量避免过多摄入咖啡因和糖分，多 吃富含维生素和矿物质的食物。
5. 社交互动：与亲朋好友保持联系，分享你的感受。社交支持是应对压力的重要资源。
6. 设定实际目标：为自己设定可实现的小目标，逐步完成它们，可以增强自信心和成就感，减少焦虑。
7. 学习放松技巧：如渐进性肌肉松弛、正念冥想等，可以帮助你在日常生活中更好地管理压力。

Experiencing anxiety and physical exhaustion is common, particularly when facing life's pressures, work-related stress, or personal challenges. Here are several recommendations that may help alleviate feelings of anxiety:

1. Engage in Regular Exercise: Physical activity is one of the most effective ways to reduce stress and anxiety. Whether it's walking, running, yoga, or any other preferred exercise, engaging in physical activity can release endorphins and improve mood.
2. Maintain a Consistent Sleep Schedule: Ensure you get adequate sleep each night and strive to keep a regular sleep routine. Good sleep quality is crucial for both physical and mental health.
3. Practice Meditation and Deep Breathing: These techniques can help relax the body and mind, reducing anxiety. Spending a few minutes each day on meditation or deep breathing exercises can enhance focus and emotional stability.
4. Adopt a Healthy Diet: A balanced diet is vital for maintaining both physical and mental health. Avoid excessive caffeine and sugar intake, and consume foods rich in vitamins and minerals.
5. Foster Social Connections: Stay in touch with family and friends and share your feelings with them. Social support is a critical resource for coping with stress.
6. Set Realistic Goals: Establish small, achievable goals for yourself and work towards completing them gradually. This can build confidence and a sense of accomplishment, thereby reducing anxiety.
7. Learn Relaxation Techniques: Techniques such as progressive muscle relaxation and mindfulness meditation can help you manage stress more effectively in daily life.
8. Remember, taking care of your emotional health is equally important. Do not hesitate to ask for help when needed and do not fear expressing your feelings. Everyone has the right to pursue a happy and healthy life.

Figure 25: A case of a client confiding to Qwen2-7B-Instruct .

<!-- image -->

<!-- image -->

<!-- image -->

我最近总觉得自己很焦虑，身心俱疲。不知道该怎么办。 I have been feeling anxious and physically and mentally exhausted lately. I don't know what to do.

嗯，听你说完，感觉你近期真的承受很多。对于你来说，什么情况会让你感到更轻松或安心呢？ Listening to you, it is apparent that you have been under considerable stress recently. What circumstances or actions might help you feel more at ease or secure?

<!-- image -->

## 我自己都很焦虑了，怎么会去想轻松和安心的事情。

I am already very anxious and have not considered things that might help me feel more relaxed and secure.

焦虑时寻找轻松感可能有些挑战，但或许尝试一下平静的呼吸，或者小小的放松活动，慢慢复苏内心的安宁。 你觉得如何呢？

While seeking a sense of ease during anxiety can be challenging, perhaps trying calming breathing exercises or small relaxation activities could gradually restore a sense of inner peace. What are your thoughts on this approach?

<!-- image -->

我试过这些方法，帮助不是很大。

I have tried these methods, but they have not been very effective.

听到你尝试过这些方法后效果有限，这种感觉会让你感到更加失望或者沮丧吗？

Given that the methods you have tried have had limited effectiveness, does this contribute to feelings of disappointment or frustration?

Figure 26: A case of a client confiding to SoulChat .

<!-- image -->

<!-- image -->

<!-- image -->

Algorithm 1 The multi-turn dialogues synthesis method of PsyDT framework. Initialize : topic\_rc : real-world counseling case of current counseling topic; rc\_list : 12 real-world counseling cases list with 12 counseling topic; topic\_ls : linguistic style of counseling case of current counseling topic; ls\_list : linguistic styles list of counseling cases with 12 counseling topics; topic\_ttype : therapeutic type of counseling case of current counseling topic; ttype\_list : therapeutic types list of counseling cases with 12 counseling topics; tt\_list : therapeutic techniques list of counseling cases with 12 counseling topics; st\_dial : current single-turn dialogue, including counseling topic, client's question, and counselor's longtext response; st\_dial\_list : given 5000 single-turn dialogues list; st\_dial\_cp : simulated client's Big Five personality traits, including Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism (OCEAN); psy\_ls : linguistic style of specific psychological counselor of current counseling topic; psy\_tt : specific knowledge of therapy technique of specific psychological counselor of current counseling topic; mt\_dial : multi-turn dialogue synthesized from the current single-turn dialogue st\_dial ; emo\_gui : emotion guide;

res\_gui : response guide;

Output : mt\_dial\_list

: synthetic multi-turn dialogues list.

- 1: for topic\_rc in range rc\_list do
- 2: topic\_ls ← GPT-4. LingStyPromptGen ( topic\_rc );
- 3: ls\_list .append( topic\_ls );
- 4: topic\_ttype ← GPT-4. TheTypePromptGen ( topic\_rc );
- 5: ttype\_list .append( topic\_ttype );
- 6: end for
- 7: tt\_list ← KB\_Retrieve ( ttype\_list );
- 8: for st\_dial in range st\_dial\_list do
- 9: st\_dial\_cp ← GPT-4. CliPerPromptGen ( st\_dial .client\_problem);
- 10: psy\_ls ← TopicMatchLing ( st\_dial .counseling\_topic, ls\_list );
- 11: psy\_tt ← TopicMatchTher ( st\_dial .counseling\_topic, tt\_list );
- 12: mt\_dial ← GPT-4. MultiTurnPromptGen ( topic\_rc , psy\_ls , emo\_gui , psy\_tt , st\_dial , st\_dial\_cp , res\_gui );
- 13: mt\_dial\_list .append( mt\_dial );
- 14: end for
- 15: return mt\_dial\_list ;

<!-- image -->

我最近总觉得自己很焦虑，身心俱疲。不知道该怎么办。 I have been feeling anxious and physically and mentally exhausted lately. I don't know what to do.

你好，感谢你前来咨询。能详细告诉我一下你的困扰吗？

Hello, thank you for reaching out. Could you please provide more details about your concerns?

<!-- image -->

<!-- image -->

我身边的同学太卷了，感觉自己已经很努力了，但还是比不上别人，做什么事情都比别人差，感觉自己不适合 读研，但又怕毕业的时候找不到满意的工作，所以很焦虑。

My classmates are very competitive. I feel that, despite my efforts, I cannot match their performance. I am worried that I am not suited for graduate studies, yet I also fear not finding a satisfactory job after graduation, which makes me very anxious.

我能理解你的感受。首先，我们要认识到每个人都有自己的价值和优点，不必过分关注他人的评价。你能告诉 我，你平时是如何应对这种焦虑情绪的吗？

I understand your feelings. It is important to acknowledge that everyone possesses unique strengths and values, and we should not overly fixate on the opinions of others. Can you share how you typically manage this anxiety?

<!-- image -->

<!-- image -->

我是一个很内向的人，一般就会默默承受这些压力，把自己关在一个黑暗的房间里，也不敢跟爸爸妈妈说，怕 他们担心。

I am very introverted and usually endure these pressures silently, isolating myself in a dark room. I do not dare to tell my parents, as I fear they would worry.

你做得很好，但有时候我们需要学会倾诉，释放内心的压力。我们可以尝试通过一些方法来缓解你的焦虑情绪， 比如深呼吸、冥想等。你愿意试试吗？

You are handling things well, but it is sometimes beneficial to learn to express our emotions and alleviate internal pressure. We could explore methods to ease your anxiety, such as deep breathing or meditation. Would you be open to trying these?

<!-- image -->

我之前试过这些方法，不过对我的帮助并不是很大。

I have tried these methods before, but they did not significantly help.

那我们可以尝试一种新的方法，叫做认知行为疗法。它可以帮助你认识到自己的消极思维，并学会调整。你想 试试吗？

In that case, we might consider a new approach called cognitive-behavioral therapy. This method can help you identify and adjust negative thought patterns. Are you interested in trying this?

Figure 27: A case of a client confiding to CPsyCounX .

<!-- image -->

<!-- image -->

<!-- image -->

我最近总觉得自己很焦虑，身心俱疲。不知道该怎么办。

I have been feeling anxious and physically and mentally exhausted lately. I don't know what to do.

听起来你现在承受了很多压力，这一定很难受。能具体告诉我，是什么事情让你感到焦虑吗？

It sounds like you are currently under a lot of pressure, which must be very challenging. Could you specify what exactly is causing your anxiety?

<!-- image -->

<!-- image -->

我身边的同学太卷了，感觉自己已经很努力了，但还是比不上别人，做什么事情都比别人差，感觉自己不适合 读研，但又怕毕业的时候找不到满意的工作，所以很焦虑。

My classmates are very competitive. I feel that, despite my efforts, I cannot match their performance. I am worried I am not suited for graduate study, yet I also fear not finding a satisfactory job after graduation, which makes me very anxious.

你觉得自己在同学中处于一个比较劣势的位置，这种感觉一定很让人沮丧。你对'不适合读研'的感觉是怎样 的呢？

You feel disadvantaged among your peers, and this must be very frustrating. How do you experience the feeling of being "unsuited for graduate studies"?

<!-- image -->

<!-- image -->

当时读这个研究生，纯粹是因为身边的很多同学都去考研，我也就随大流去考了，后面发现自己并不是做科研 的料，有时也挺痛苦的。

I pursued this graduate program mainly because many of my peers were doing so, and I followed suit. Later, I realized I am not cut out for research, which has often been quite distressing.

选择考研时可能更多的是受到周围环境的影响，而不是出于你个人的兴趣或优势。你提到了痛苦，能分享一下 具体是什么让你感到痛苦吗？

Your decision to pursue graduate studies seems to have been influenced more by your surroundings than by your personal interests or strengths. You mentioned distress-could you elaborate on what specifically causes you this pain?

<!-- image -->

<!-- image -->

因为我是跨专业考研，研究生的很多课程我都听不懂。另外导师也会额外布置很多科研任务，有时也会pua我， 身边的大佬也特别多，无形之中给了我巨大的压力和痛苦。

I switched fields for my graduate studies, and I find many of the courses incomprehensible. Additionally, my advisor assigns numerous research tasks and sometimes employs manipulative tactics. There are also many outstanding peers around me, which adds to my immense pressure and distress.

听你这么说，感觉你承受了很大的学习和心理压力。这种感觉肯定很沉重。在这样的环境中，你通常是如何处 理这些压力的呢？

From what you describe, it seems you are under significant academic and psychological pressure. This must be a heavy burden. How do you typically cope with these pressures in such an environment?

<!-- image -->

<!-- image -->

我是一个很内向的人，一般就会默默承受这些压力，把自己关在一个黑暗的房间里，也不敢跟爸爸妈妈说，怕 他们担心。

I am very introverted and usually endure these pressures silently, isolating myself in a dark room. I do not dare to tell my parents, as I fear they would worry.

我理解你可能担心给家人带来负担，所以选择了独自承受。不过，寻求帮助也是一种勇气的表现。我们可以一 起探讨一些可能的解决方法。

I understand you may worry about burdening your family, so you choose to bear the stress alone. However, seeking help is also a sign of courage. We can discuss possible solutions together.

<!-- image -->

那你有什么好的建议呢？

Do you have any suggestions?

我们可以尝试找到一些缓解焦虑的方法，比如运动、听音乐或者写日记。这些活动可以帮助我们放松心情。你 觉得呢？

We could explore some anxiety-relief methods, such as exercising, listening to music, or journaling. These activities can help relax the mind. What do you think?

<!-- image -->

我之前试过这些方法，不过对我的帮助并不是很大。

I have tried these methods before, but they did not significantly help.

我理解，每个人的感受都是独特的，找到最适合自己的缓解方法需要时间和耐心。如果你愿意，我们可以一起 探索其他可能适合你的方法。

I understand. Each person's experience is unique, and finding the most suitable relief method takes time and patience. If you are willing, we can explore other potential methods together.

<!-- image -->

嗯嗯可以。

Yes, I am open to that.

你提到了导师的pua行为，这听起来对你影响很大。你是否曾尝试过跟导师沟通你的感受？

You mentioned your advisor's manipulative behavior, which seems to have a significant impact on you. Have you ever tried to communicate your feelings to your advisor?

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

我感觉我的导师很强势，有时也想跟他沟通，但又怕自己说错话得罪导师。

I feel my advisor is very domineering. Sometimes I want to communicate with him, but I fear saying the wrong thing and offending him.

和导师沟通确实需要勇气，尤其是当我们感到害怕时。但沟通也是解决问题的一种方式。如果可能的话，我们 可以一起探讨一些沟通策略。

Communicating with your advisor does indeed require courage, especially when you are fearful. However, communication can also be a way to resolve issues. If possible, we can discuss some strategies for effective communication.

Figure 28: A case of a client confiding to PsyDTLLM .

<!-- image -->