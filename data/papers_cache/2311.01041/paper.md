## Learn to Refuse: Making Large Language Models More Controllable and Reliable through Knowledge Scope Limitation and Refusal Mechanism

Lang Cao University of Illinois Urbana-Champaign

Department of Computer Science langcao2@illinois.edu

## Abstract

Large language models (LLMs) have demonstrated impressive language understanding and generation capabilities, enabling them to answer a wide range of questions across various domains. However, these models are not flawless and often produce responses that contain errors or misinformation. These inaccuracies, commonly referred to as hallucinations, render LLMs unreliable and even unusable in many scenarios. In this paper, our focus is on mitigating the issue of hallucination in LLMs, particularly in the context of question-answering. Instead of attempting to answer all questions, we explore a refusal mechanism that instructs LLMs to refuse to answer challenging questions in order to avoid errors. We then propose a simple yet effective solution called Learn to Refuse (L2R), which incorporates the refusal mechanism to enable LLMs to recognize and refuse to answer questions that they find difficult to address. To achieve this, we utilize a structured knowledge base to represent all the LLM's understanding of the world, enabling it to provide traceable gold knowledge. This knowledge base is separate from the LLM and initially empty. It can be filled with validated knowledge and progressively expanded. When an LLM encounters questions outside its domain, the system recognizes its knowledge scope and determines whether it can answer the question independently. Additionally, we introduce a method for automatically and efficiently expanding the knowledge base of LLMs. Through qualitative and quantitative analysis, we demonstrate that our approach enhances the controllability and reliability of LLMs.

## 1 Introduction

Recent progress in large language models (LLMs) has showcased their strong language understanding, generation, reasoning, and various other abilities (Zhao et al., 2023; OpenAI, 2023). These capabilities enable their application across various fields and scenarios, including question-answering systems, among others. However, the issue of hallucination often occurs in the responses of LLMs, as highlighted in previous studies (Ji et al., 2023; Zhang et al., 2023). These hallucinations result in inaccuracies and errors in their output, rendering LLM-based systems unreliable and even unusable (Kaddour et al., 2023; Umapathi et al., 2023). It is imperative to mitigate hallucinations and enhance the reliability of LLM-based applications. Hallucinations can be categorized into three types: inputconflicting hallucination, context-conflicting hallucination, and fact-conflicting hallucination (Zhang et al., 2023). The first two types arise from LLMs' limited understanding or omission of information during text generation. On the other hand, the third type mainly stems from LLMs' limited knowledge or lack of clear knowledge comprehension. The underlying reasons include inadequate training on specific facts, incomplete learning, forgetting certain facts, or incorrectly mixing up facts. However, when interacting with ChatGPT 1 , we observe that it attempts to answer all questions except some risky ones. Consequently, its responses are inherently flawed due to its limited knowledge and inadequate knowledge management. In this paper, we specifically address the third type of hallucination, namely fact-conflicting hallucination, which indicates deficiencies in the LLM's knowledge.

Figure 1: The overview of L2R . L2R differs from traditional LLM-based QA systems that directly answer questions. It has the ability to refuse the user's question based on specific situations.

<!-- image -->

Retrieval augmentation is an effective approach to mitigate hallucination because it significantly enhances the knowledge of large language models, preventing them from answering questions without knowledge or evidence (Li et al., 2022; Lewis et al., 2020). It is intuitive that providing LLMs with numerous true and accurate facts would improve the accuracy of their answers. Therefore, we can infer that if we already provide LLMs with right answers for every question, their responses will be perfect. Based on this, we hypothesize that fact-conflicting hallucination arises from incorrect knowledge in LLMs or from some knowledge they do not know.

Recent progress in LLMs (Kadavath et al., 2022; Yin et al., 2023) demonstrates that LLMs possess self-knowledge. Self-knowledge refers to LLMs' awareness of the knowledge they possess and their ability to identify unanswerable or unknowable questions based on their own knowledge or provided information. Building on this observation, we suppose that if we can provide relevant information for a question that an LLM needs to answer, it has the ability to judge whether it can provide a reliable response based on that information.

Considering these two hypotheses, we propose two concepts: Knowledge Scope Limitation and Refusal Mechanism , respectively. Knowledge Scope Limitation means using an independent, limited, and structured knowledge base to represent the knowledge scope of an LLM. We divide the knowledge of the LLM and the LLM itself. Our objective is for the LLM to function solely as a machine that processes input and output data and interacts with users using its language processing ability. We presume that the LLM does not possess internal knowledge to avoid the influence of incorrect information and unclear expressions. Additionally, we need to ensure that the knowledge in the knowledge base is totally true. This kind of knowledge differs from the general knowledge form of LLMs, which is parametric, unlimited, untraceable, unmeasured, and unverified. Consequently, the question-answering system becomes traceable and controllable because a structured knowledge base for the LLM is clear and easy to maintain. Refusal Mechanism involves using prompts to instruct LLMs to refuse to answer questions if they find them difficult. By abstaining from providing answers in such cases, LLMs can avoid potential errors or risks. This aspect contributes to the natural reliability of the question-answering system.

1 https://platform.openai.com/docs/models/gpt-3-5

We integrate these two concepts into a novel LLM-based question-answering system called L2R , which stands for L earn to R efuse. As depicted in Figure 1, L2R incorporates an independent structured knowledge base. It can refuse to answer questions that it deems challenging. When it can provide an answer, it does so step-by-step, offering precise and clear evidence and reasoning from the structured knowledge base. This approach also improves the explainability of the answers, making our system more controllable and reliable compared to traditional ones.

In the design of Knowledge Scope Limitation , the main distinction between L2R and previous works that aim to enhance the knowledge of LLMs is that we consider the initial knowledge base to be empty. We then infuse it with true and verified knowledge. We acknowledge that this process may be challenging and require significant human effort. That is because L2R overlooks the knowledge stored in LLMs, resulting in a wastage of resources. To address this, we propose a simple method called A utomatic K nowledge E nrichment ( AKE ) to compensate for this aspect. It enables a rapid addition of knowledge to the knowledge base, ensuring a high quality of knowledge simultaneously. The knowledge is originated from the internal knowledge of LLMs. Before adding these new knowledge directly to the knowledge base, we instruct the LLMs to validate it based on their confidence. As a result, this knowledge is more likely to be true and can be utilized by L2R .

In summary, this paper makes the following main contributions:

- We explore the Refusal Mechanism in an LLMbased question-answering system, which effectively maintains answer quality and mitigates risks by refusing to answer certain questions.
- We propose a new method called L2R , which enhances the controllability and reliability of LLMbased question-answering systems. This method incorporates both the Knowledge Scope Limitation and Refusal Mechanism . L2R includes an independent knowledge base with limited and verified knowledge, as well as the ability to refuse to answer questions.
- We introduce a simple yet effective automatic knowledge enrichment method. This method is particularly useful when the initial knowledge base is empty and allows for the rapid addition of knowledge to LLMs.
- We conduct qualitative and quantitative experiments to demonstrate the effectiveness of the Refusal Mechanism and the performance of L2R . The experimental results showcase the controllability and reliability of L2R .

## 2 Related Work

## 2.1 Hallucinations in Large Language Models

Since Natural Language Generation (NLG) has improved thanks to the development of sequence-tosequence deep learning technologies, hallucination is a big problem in the generation quality (Ji et al., 2023). This phenomenon means that NLG models often generate text that is nonsensical, or unfaithful to the provided (Maynez et al., 2020; Raunak et al., 2021; Koehn and Knowles, 2017). In the era of LLMs, these LLMs show their strong various abilities, particularly in text generation in all kinds of setting (Zhao et al., 2023). However, hallucination is still a big problem here and become more and more urgent for us to solve. LLMs are unreliable and unusable if their output contains error and violate factual knowledge (Zhang et al., 2023). Recently, many works have been proposed to mitigate hallucinations in LLMs. They works in various perspective of LLMs, including mitigation during pretraining (Penedo et al., 2023; Lee et al., 2023), mitigation during SFT (Zhou et al., 2023; Cao et al., 2023), mitigation during RLHF (Sun et al., 2023; Wu et al., 2023; Lightman et al., 2023), mitigation during inference (Dhuliawala et al., 2023; Li et al., 2023; Peng et al., 2023; Manakul et al., 2023).

While LLMs usually overestimate their ability to answer question (Zhang et al., 2023), which may cause hallucinations, some other works focus on self-knowledge of LLMs. (Kadavath et al., 2022) suggest that LLMs possess a certain degree of self-knowledge, which means they know what knowledge they have and have the ability to identify unanswerable or unknowable questions. However, there is still an apparent disparity in comparison to human self-knowledge. (Yin et al., 2023) also provides evidence that larger models exhibit wellcalibrated claim evaluation and demonstrate some awareness of their knowledge gaps.

Based on these findings, we propose a refusal mechanism in the question-answering application of LLMs. However, the primary distinction lies in our consideration of the initial knowledge of LLMs as zero, which we represent through an independent, limited, and structured knowledge base. Consequently, we can exercise better control over their knowledge.

## 2.2 Retrieval Augmented Generation

Retrieval augmented generation is a text generation paradigm that combine deep learning technology and traditional retrieval technology (Li et al., 2022; Lewis et al., 2020). Retrieval augmented generation can be applied on language models to enhance their knowledge and make their response more accurately. RAG (Lewis et al., 2021) and REALM (Guu et al., 2020) are proposed in the similar way to incorporate retrieval result into the training of language models. They both train the retriever and language model together by modelling documents as latent variable, and minimizing the objective with gradient descent. The related kNN-LM model (Khandelwal et al., 2020) replaces LSTMs by transformer networks, and scales the memory to billions of tokens, leading to strong performance improvements. Recently, RETRO (Borgeaud et al., 2022) extends these by scaling the retrieval memory to trillions of tokens, and changing the model architecture to take retrieved documents as input. Some works (Shuster et al., 2022; Lazaridou et al., 2022) apply retrieval augmentation with search engines to get online information as retrieval results.

We also incorporate retrieval augmentation in our system and instruct LLMs to rely solely on the retrieval results for answering. As a result, our methods are fully controllable and traceable.

## 3 Methodology

## 3.1 Task Formulation

Given a set of n questions Q = { Q 1 , Q 2 , ..., Q n } , where each question Q i pertains to factual knowledge, the objective of the factual question answer- ing task is to provide answers to these factual questions in A = { A 1 , A 2 , ..., A n } . Our goal is to develop a system capable of answering these questions A with reasoning R and evidence E , or alternatively, refuse to answer certain questions by REFUSAL , which indicates that the system refuses to answer the question.

Figure 2: The framework of L2R . L2R consists of two main components: manual or automatic knowledge enrichment and question answering based on structured knowledge.

<!-- image -->

## 3.2 L2R Framework

We propose a novel system called L2R , which stands for L earn to R efuse, to address this task. The framework of L2R is illustrated in Figure 2. This system can answer factual questions using the refusal mechanism, which means that it will decline to answer a question if it lacks sufficient knowledge on the topic. To represent the system's knowledge, we utilize a structured knowledge base that defines the scope of its knowledge. The structured knowledge base KB comprises m factual knowledge entries, denoted as K = { K 1 , K 2 , ..., K m } . For each question, we use the description of this question to query the structured knowledge base KB to retrieve the top k related pieces of knowledge, denoted as K = [ K 1 , K 2 , ..., K k ] . These retrieved knowledge then used by the Main QA Agent module to provide information for answering.

In L2R , there are two types of refusal mechanisms employed: soft refusal and hard refusal. Before providing an answer, both mechanisms work together to determine whether the question Q i can be answered according to the knowledge scope. It will produce a judgment J i ∈ { 0 , 1 } to determine if the question Q i can be answered. If J i = 1 , the system generates an answer for the question as A i = { E i , R i , A ′ i } , where E i represents the supporting evidence, R i is the reasoning behind the final answer, and A ′ i is the specific answer to the question Q i . If J i = 0 , indicating that the question is unanswerable, the system refuses to provide an answer, and A i = REFUSAL . Afterward, users can receive the response from the system.

Furthermore, we propose manual or automatic knowledge enrichment methods to efficiently construct the structured knowledge base in L2R . Elaborated prompts are designed to instruct the tasks and functions of all LLMs in the system.

## 3.3 Manual and Automatic Knowledge Enrichment

The knowledge base in L2R is initially empty and will be enriched through two methods. We designed this knowledge base to be structured, but our system does not depend on a structured knowledge base. A structured knowledge base offers more traceability and clarity for subsequent steps and demonstrations.

Manual knowledge enrichment involves human intervention to manually add m verified gold knowledge entries K = [ K 1 , K 2 , ..., K m ] to the structured knowledge base KB . Each K i represents a text description of a single piece of factual knowledge. In other words, each piece of data in the knowledge base cannot encompass multiple factual knowledge. To expedite the process of constructing the structured knowledge base, we propose A utomatic K nowledge E nrichment (AKE) to utilize internal knowledge from LLMs. AKE is a method that enables the rapid addition of pseudo knowledge with high confidence to KB . The process of automatic knowledge enrichment does not involve any human effort. It also ensures that our system does not heavily rely on a constantly updated knowledge base. It is developed to compensate for the deficiencies of manual knowledge enrichment, though it may compromise the accuracy of the knowledge. We quantitatively measure the truthfulness of knowledge from AKE using a confidence value C , which represents the confidence level of the knowledge produced by LLMs.

In automatic knowledge enrichment, three components are utilized: Question Generation Agent , Answer Generation Agent , and QA Pair to Knowledge Agent . These components are LLMs for which we provide detailed prompts to instruct them in completing specific tasks. Question Generation Agent generates m questions Q = [ Q 1 , Q 2 , ..., Q m ] based on different seed questions. Answer Generation Agent answers the generated questions and provides confidence scores for the answers, resulting in A withC = [( A 1 , C 1 ) , ( A 2 , C 2 ) , ..., ( A m , C m )] , where C i ∈ [0 , 1] represents the confidence value of A i . The QA pairs QA = [( Q 1 , A 1 ) , ( Q 2 , A 2 ) , ..., ( Q m , A m )] are then inputted into QA Pair to Knowledge Agent , which transforms them into pseudo knowledge K = [( K 1 , C 1 ) , ( K 2 , C 2 ) , ..., ( K m , C m )] . The confidence value C is retained to represent the confidence level of this knowledge. We use QA Pair to Knowledge Agent to transform QA pair into a more readable narrative sentence, which can be easily processed for subsequent steps and retrieval. After this process, K can be added to the structured knowledge base KB . On the other hand, for manual knowledge enrichment, we assign a confidence value of C i = 1 to human-verified knowledge in order to maintain consistency with the format of the generated pseudo-knowledge.

## 3.4 Retrieval Results Fusion

The main LLM responsible for answering user's questions is referred to as the Main QA Agent . To provide retrieved knowledge for this LLM to answer questions, we employ retrieval augmented generation (Li et al., 2022; Lewis et al., 2020). We retrieve k pieces of knowledge K from the structured knowledge base KB for the LLM. We compute the similarity S between the current question Q and all knowledge K . Based on the similarity score, we select the k most relevant pieces of knowledge for each question Q . Specifically, we utilize the Euclidean distance, also known as L2 distance, as the similarity metric. A lower similarity score S i for knowledge K i indicates a higher relevance to the current question Q . The retrieval result of the k most related pieces of knowledge is represented as follows:

<!-- formula-not-decoded -->

where C i represents the confidence value of the knowledge K i stored in the structured knowledge base KB , and S i denotes the similarity score between the current question Q and the knowledge K i .

The prompts provided to the Main QA Agent explicitly instruct it not to use any internal knowledge. Consequently, the LLM produces responses solely based on the retrieved information, proceeding to subsequent steps. It should be noted that obtaining the confidence score C does not violate the principle of not using any internal knowledge, because it comes from another LLM agent in the process of AKE. The step of knowledge base enrichment is not part of the question-answering stage and is not necessary.

## 3.5 Refusal Mechanism

The refusal mechanism in L2R judges whether a question Q can be answered or not and refuses to answer if it deems the question unanswerable. Two types of refusal mechanisms in L2R work together to make this decision: soft refusal and hard refusal. The former is from LLM's generation output and is executed by the LLM itself, while the latter is set by humans and can be adjusted based on different situations. We categorize refusals as 'Soft' and 'Hard' from a system perspective. A soft refusal is defined as one originating directly from the LLM, and it is variable and adjustable based on different LLMs and their prompts. In contrast, a hard refusal involves a backup method, which requires additional computation and comparison with a system-defined threshold. We consider this hard refusal significant because if the knowledge base support is insufficient, the L2R system will refuse to answer to avoid hallucination, regardless of the LLM's perspective.

In detail, soft refusal is a mechanism where we instruct LLMs through prompts to independently judge the answerability I soft i ∈ { 0 , 1 } of a question Q i . We can obtain I soft i with answers from LLMs' output. This decision is based on the retrieved information and the LLM's self-knowledge, allowing it to determine if it can answer the question.

On the other hand, hard refusal involves a mathematical function specifically designed to compute the score of the retrieved knowledge K r for the question Q and compare it with a specific score threshold α to decide whether the system can answer the question. The judge function can vary and extend to more complex cases. In this paper, we use the simplest version of the hard refusal function:

<!-- formula-not-decoded -->

where C = [ C 1 , C 2 , ..., C k ] and S = [ S 1 , S 2 , ..., S k ] are vectors of confidence values and similarity scores of the retrieved knowledge K = [ K 1 , K 2 , ..., K k ] . I hard i ∈ { 0 , 1 } represents the answerability result from the hard judge. I hard i = 0 indicates that question Q i is refused to be answered by the hard mechanism, while I hard i = 1 represents a pass. The score threshold value α is set by humans and can be adjusted flexibly. Equation 2 implies that we find at least one relevant piece of knowledge in the knowledge base, which LLMs can rely on to provide the correct answer. The hard judge serves as an insurance for the soft judge, ensuring that LLMs do not answer questions that are unanswerable.

The final judgment of the entire refusal mechanism is determined by:

<!-- formula-not-decoded -->

This means that the question needs to pass both the soft refusal and hard refusal mechanisms simultaneously.

## 3.6 Answer Step by Step

After the refusal judgment process, L2R provides a final response based on the results of the refusal judgment. If I fi nal i = 0 , the system will directly output REFUSAL . If I fi nal i = 1 , the system will first output the evidence E , which consists of the retrieval results, which is also supporting evidence for the final answer. Following the idea of Chainof-Thought (Wei et al., 2023), we design prompts to instruct LLMs to provide a reasoning path R leading to the final answer A . Therefore, for an answer Q i , if it is answerable, the response from L2R would be ( E i , R i , A i ) . The inclusion of evidence and reasoning for the final answer ensures traceability, as all the used knowledge can be traced back to the structured knowledge base KB .

## 4 Experiments

We conduct extensive quantitative and qualitative experiments to analyze the refusal mechanism and evaluate the performance of L2R . All the details regarding the experiment settings can be found in Appendix A.

For the metrics in our experiments, we use count and accuracy to demonstrate performance. In our setting, since L2R does not answer all questions, we define count as the number of questions answered, and accuracy is calculated within the set of answered questions. Therefore, we aim to improve accuracy while maintaining a high count.

## 4.1 Overall Performance of L2R

We use TruthfulQA dataset (Lin et al., 2022a) for main experiments. L2R is the method proposed in this paper. We construct the structured knowledge base from scratch without any human effort utilizing automatic knowledge enrichment. We use questions exclusively from the TruthfulQA dataset. The system generates pseudo answers and pseudo knowledge based on questions in TruthfulQA. This construction process for L2R does not involve any prior knowledge or data of the answers or options in TruthfulQA. After constructing the structured knowledge base for L2R , we also evaluate the system's performance on this dataset.

The baseline for gpt-3.5-turbo involves pure question-answering using LLMs. In gpt-3.5-turbo +RAG , we enhance the knowledge of gpt-3.5-turbo by retrieving information from the Wikipedia corpus. In gpt-3.5-turbo + RAG + Soft Refusal , we add a paragraph of prompts that instruct the model to refuse to answer difficult questions.

The main results of the experiments can be found in Table 1. Notably, L2R achieves higher accuracy in both the MC1 and MC2 tasks by selectively refusing to answer certain questions. In the MC1 task, it improves the accuracy of the original LLM, gpt-3.5-turbo , by 18.5 percentage points, answering 163 fewer questions, which is approximately 20% of all questions. Specifically, 149 refusals are from the hard refusal and 14 refusals are from the soft refusal in the MC1 task, while 149 and 13 refusals are from the hard and soft refusal, respectively, in the MC2 task. The results of gpt-3.5-turbo + RAG demonstrate the performance of RAG, but the improvement is limited and even decreases in the MC2 task. By adding the soft refusal to this method, we observe a slight performance improvement. This indicates that the refusal mechanism can bring improvements to the pure RAG model, and that the refusal mechanism does not depend on a structured knowledge base.

Table 1: The overall performance of L2R and several baselines (%). Count in the table represents the number of questions answer. L2R outperforms other methods by selectively refusing to answer certain questions to achieve more reliable results.

|                                   | MC1   | MC1      | MC2   | MC2      |
|-----------------------------------|-------|----------|-------|----------|
|                                   | Count | Accuracy | Count | Accuracy |
| Llama-2-70b-chat-hf               | 817   | 31.2     | 817   | 50.1     |
| gpt-3.5-turbo                     | 817   | 46.6     | 817   | 68.2     |
| gpt-3.5-turbo + RAG               | 817   | 53.7     | 817   | 67.1     |
| gpt-3.5-turbo + RAG+ Soft Refusal | 530   | 55.1     | 573   | 66.2     |
| L2R-Llama                         | 618   | 47.1     | 611   | 56.9     |
| L2R-GPT (Ours)                    | 654   | 65.1     | 655   | 70.0     |

Table 2: Experimental results from three distinct datasets-TruthfulQA, CommonsenseQA, and MedQA. It demonstrate that L2R enhances answer accuracy across various fields of questions.

|               | TruthfulQA-MC1   | TruthfulQA-MC1   | TruthfulQA-MC2   | TruthfulQA-MC2   | CommonsenseQA   | CommonsenseQA   | MedQA   | MedQA    | MedQA-RAG   | MedQA-RAG   |
|---------------|------------------|------------------|------------------|------------------|-----------------|-----------------|---------|----------|-------------|-------------|
|               | Count            | Accuracy         | Count            | Accuracy         | Count           | Accuracy        | Count   | Accuracy | Count       | Accuracy    |
| Llama         | 817              | 31.2             | 817              | 50.1             | 1221            | 73.2            | 1273    | 41.6     | 1273        | 41.6        |
| gpt-3.5-turbo | 817              | 46.6             | 817              | 68.2             | 1221            | 69.8            | 1273    | 51.2     | 1273        | 50.9        |
| L2R-Llama     | 618              | 47.1             | 611              | 56.9             | 565             | 73.6            | 430     | 43.3     | 512         | 43.6        |
| L2R (Ours)    | 654              | 65.1             | 655              | 70.0             | 933             | 75.6            | 451     | 52.8     | 776         | 53.2        |

We can compare L2R with gpt-3.5-turbo + RAG . The well-structured knowledge base in L2R only contains 817 sentences, which are processed through automatic knowledge enrichment. In contrast, Wikipedia contains a vast amount of text, but this text is not well structured. Each piece of text in the knowledge base may contain multiple knowledge. Our method is more accurate and efficient compared to gpt-3.5-turbo + RAG . This demonstrates the effectiveness of automatic knowledge enrichment. It is beneficial to allow LLMs to generate knowledge with confidence on their own. On the other side, it is important to keep each piece of knowledge simple and clean. Additionally, the step-by-step output with evidence also contributes to this improvement.

The improvement in accuracy for the MC2 task is not as significant. We believe this is because the MC2 task is more challenging, as each option is independent and the system needs to evaluate each option individually. In this case, the system requires knowledge of each option to provide a more accurate answer. However, there is still a slight improvement of 1.8%.

We also evaluate L2R based on the open-source LLM of llama-2 (Touvron et al., 2023), named L2R -Llama. This evaluation suggests a significant improvement of 15.9% in accuracy, demonstrating that our system can enhance performance across different foundational models.

The ablation study analyzing the performance improvements from each component can be found in Appendix B.

## 4.2 Results on Multiple Datasets

We evaluated L2R on two additional datasets to ensure a broader applicability: CommonsenseQA (Talmor et al., 2019) and MedQA (Jin et al., 2020), covering both commonsense and medical domains. As shown in Table 2, L2R outperforms the baseline by a notable margin, demonstrating accuracies of 65.1% on TruthfulQA-MC1, 70.0%

on TruthfulQA-MC2, and 75.6% on CommonsenseQA, compared to the baseline's lower scores. In the version of L2R -Llama, it also shows an improvement compared to the llama baseline.

In the specialized medical dataset, MedQA, method outperformed the baseline, achieving 52.8% accuracy. However, the improvement is limited, and with 822 not answered questions, it does not demonstrate an optimal QA system performance. We consider that this limitation arises because the original knowledge embedded in LLMs is insufficient for effective Automatic Knowledge Enrichment (AKE), resulting in a failure to achieve substantial improvements. To further assess it, we use an medical corpus, MedRAG - textbooks (Xiong et al., 2024), as additional augmented data. We segment this corpus into sentences to construct a structured knowledge base. With a more reliable knowledge base, the performance improvement increases from 0.4% to 2.3%, and the number of answered questions increase by 315. In contrast, adding additional data to the baseline results in a performance drop of 0.3%. This suggests issues with noise when incorporating more data into the QA system using the traditional RAG approach.

These results reflect the robust answering capabilities of L2R and its potential across various question-answering contexts.

## 4.3 Qualitative Experiments

We also provide some examples of L2R in a simple qualitative setting to observe its performance clearly. Initially, we insert three pieces of gold knowledge into the knowledge base of the system, as shown in Figure 3. We then pose several questions from different perspectives. The results are displayed in Figure 4. In these figures, red highlighted None indicates instances where the system refuses to answer the question based on its limited knowledge base.

These examples offer a clear illustration of the user experience with L2R . It has a limited knowledge base to clearly represent its knowledge scope. The system can refuse to answer certain questions which it does not know. More details regarding the input-output of L2R can be found in the case study in Appendix F.

## 5 Conclusion

Hallucination remains a significant challenge in the development of LLMs, and numerous approaches have been proposed to address it. In this paper, we start from a different direction to mitigate hallucination by introducing a refusal mechanism. Our primary idea is to build an LLM-based system to respond only to questions they have confidence in answering. We introduce a novel system called L2R , which combines a independent, limited, and structured knowledge base and the refusal mechanism. Extensive experiments demonstrate the exceptional performance of L2R and effectiveness of the refusal mechanism, making QA systems more controllable and reliable.

Figure 3: The knowledge base used in qualitative experiments. We have added three pieces of gold knowledge to this knowledge base for test.

| Knowledge                                            |   Confidence |
|------------------------------------------------------|--------------|
| Leonardo da Vinci painted the Mona Lisa.             |          1.0 |
| The capital of the United States is Washington, D.C. |          1.0 |
| DeepMind was founded in 2010.                        |          1.0 |

Figure 4: The results of qualitative experiments. Red highlighted None indicates that the system has refused to answer the question based on its limited knowledge base.

<!-- image -->

| User: Who painted the Mona Lisa? AI: Leonardo da Vinci                                       |
|----------------------------------------------------------------------------------------------|
| User: Who is Leonardo da Vinci AI: Leonardo da Vinci is an artist who painted the Mona Lisa. |
| User: Where was Leonardo da Vinci born? AI: None                                             |
| User: Where is the capital of the United States? AI: Washington, D.C.                        |
| User: Where is the capital of China? AI: None                                                |
| User: Where is Deepmind? AI: None                                                            |
| User: What was happened in 2010? AI: DeepMind was founded in 2010.                           |
| User: Was Deepmind founded in 2018? AI: False                                                |
| User: When was Openai founded? AI: None                                                      |

We believe this work can offer valuable insights and significant potential for real-world applications. In the future, we will explore the self-knowledge of LLM deeper and continue to enhance L2R to address its limitations, making it more powerful.

## Limitations

This work is a demonstration of knowledge scope limitation and refusal mechanism of large language models in question-answering scenarios. There are many problems now and still a distance to be directly used in life.

Hallucination of System. In this work, we let the system to refuse to give response when their response have a large possibility of containing errors. Our experiments show that this mechanism can make LLM-based question-answering system more reliable and mitigate the hallucination of LLM. However, it cannot guarantee that the response of these system does not contain hallucination. There are many other reasoning of hallucination, such as deviating from user input, forgetting previously generated context. We just focus on mitigating hallucination due to violation of factual knowledge

Scaling Up. In our experiments, we evaluate our model in one dataset with hundreds-level pieces of knowledge in the structured knowledge base due to resources limited. If the magnitude of the knowledge base reaches millions-level or more, the performance of our system is uncertain and need to be evaluated later.

Refusal Function. The refusal function of current system is simple. We just compare the similar semantic score with the defined threshold to judge if the retrieved results are related. When the system need more pieces of knowledge or need multiple knowledge to answer one question, we need to design a better refusal function to perform hard judge of refusal and make refusal mechanism more stable.

Complex Questions. In our experiment, we use TruthfulQA (Lin et al., 2022b) to evaluate the performance of our system. However, questions in this dataset is simple. In most cases, the system just need one piece of knowledge to answer one question. In the real world, human have many complex questions. Some questions need multiple knowledge, while some question need to reasoning in multiple steps based on different knowledge. These settings is more difficult to be applied with our system. To solve these complex questions, we need to instruct LLMs to utilize there knowledge and improve their answer logic.

Application Scenarios. In this paper, we focus on the question-answering scenario which is most use cases of LLMs. Hallucination in the output of LLMs bring bad consequence in every application of LLMs. Our system in our work can just used in question-answering scenario and cannot be directly applied in more application scenarios, like text summarization, decision making, etc. There are still many work to do about how to adapt our system to these tasks.

The goal of our work is to propose a new direction to mitigate hallucination and inspire more similar works in the future.

## References

Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, and Laurent Sifre. 2022. Improving language models by retrieving from trillions of tokens.

[Yihan Cao, Yanbin Kang, and Lichao Sun. 2023. Instruction mining: High-quality instruction data selection for large language models.](http://arxiv.org/abs/2307.06290)

Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston. 2023. Chain-of-verification reduces hallucination in large language models.

Wikimedia Foundation. Wikimedia downloads.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. Realm: Retrievalaugmented language model pre-training.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of hallucination in natural language generation. ACM Computing Surveys , 55(12):1-38.

- [Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. 2020. What disease does this patient have? a large-scale open domain question answering dataset from medical exams.](http://arxiv.org/abs/2009.13081)
- Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data , 7(3):535-547.
- Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. 2022. Language models (mostly) know what they know.
- Jean Kaddour, Joshua Harris, Maximilian Mozes, Herbie Bradley, Roberta Raileanu, and Robert McHardy. 2023. Challenges and applications of large language models.
- Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. 2020. Generalization through memorization: Nearest neighbor language models. In International Conference on Learning Representations .
- Philipp Koehn and Rebecca Knowles. 2017. Six challenges for neural machine translation. In Proceedings of the First Workshop on Neural Machine Translation , pages 28-39, Vancouver. Association for Computational Linguistics.
- [Angeliki Lazaridou, Elena Gribovskaya, Wojciech Stokowiec, and Nikolai Grigorev. 2022. Internetaugmented language models through few-shot prompting for open-domain question answering.](http://arxiv.org/abs/2203.05115)
- Nayeon Lee, Wei Ping, Peng Xu, Mostofa Patwary, Pascale Fung, Mohammad Shoeybi, and Bryan Catanzaro. 2023. Factuality enhanced language models for open-ended text generation.
- Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive nlp tasks. In Advances in Neural Information Processing Systems , volume 33, pages 94599474. Curran Associates, Inc.
- Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-augmented generation for knowledgeintensive nlp tasks.
- Huayang Li, Yixuan Su, Deng Cai, Yan Wang, and Lemao Liu. 2022. A survey on retrieval-augmented text generation.
- Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inference-time intervention: Eliciting truthful answers from a language model.
- Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let's verify step by step.
- Stephanie Lin, Jacob Hilton, and Owain Evans. 2022a. Truthfulqa: Measuring how models mimic human falsehoods.
- Stephanie Lin, Jacob Hilton, and Owain Evans. 2022b. TruthfulQA: Measuring how models mimic human falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 3214-3252, Dublin, Ireland. Association for Computational Linguistics.
- Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. 2023. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models.
- Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. 2020. On faithfulness and factuality in abstractive summarization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , pages 1906-1919, Online. Association for Computational Linguistics.

[OpenAI. 2023. Gpt-4 technical report.](http://arxiv.org/abs/2303.08774)

- Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. 2023. The refinedweb dataset for falcon llm: Outperforming curated corpora with web data, and web data only.
- Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, and Jianfeng Gao. 2023. Check your facts and try again: Improving large language models with external knowledge and automated feedback.
- Vikas Raunak, Arul Menezes, and Marcin JunczysDowmunt. 2021. The curious case of hallucinations in neural machine translation. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pages 1172-1183, Online. Association for Computational Linguistics.
- Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing . Association for Computational Linguistics.
- Kurt Shuster, Mojtaba Komeili, Leonard Adolphs, Stephen Roller, Arthur Szlam, and Jason Weston. 2022. Language models that seek for knowledge: Modular search &amp; generation for dialogue and prompt completion. In Findings of the Association for Computational Linguistics: EMNLP 2022 , pages 373-393, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, Kurt Keutzer, and Trevor Darrell. 2023. Aligning large multimodal models with factually augmented rlhf.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) , pages 4149-4158, Minneapolis, Minnesota. Association for Computational Linguistics.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023. Llama 2: Open foundation and finetuned chat models.

Logesh Kumar Umapathi, Ankit Pal, and Malaikannan Sankarasubbu. 2023. Med-halt: Medical domain hallucination test for large language models.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. 2023. Chain-of-thought prompting elicits reasoning in large language models.

Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A. Smith, Mari Ostendorf, and Hannaneh Hajishirzi. 2023. Finegrained human feedback gives better rewards for language model training.

Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. 2024. Benchmarking retrieval-augmented generation for medicine.

Zhangyue Yin, Qiushi Sun, Qipeng Guo, Jiawen Wu, Xipeng Qiu, and Xuanjing Huang. 2023. Do large language models know what they don't know? In Findings of the Association for Computational Linguistics: ACL 2023 , pages 8653-8665, Toronto, Canada. Association for Computational Linguistics.

Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi. 2023. Siren's song in the ai ocean: A survey on hallucination in large language models.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2023. A survey of large language models.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. 2023. Lima: Less is more for alignment.

## A Experiment Settings

We mainly use TruthfulQA dataset(Lin et al., 2022a) to quantitatively evaluate the performance of L2R . This dataset consists of 817 questions spanning 38 categories, including health, law, finance, and politics, effectively measuring the hallucination of an LLM. We select two tasks, MC1 (Multiple-choice Single-true) and MC2 (Multiplechoice Multi-true), to evaluate L2R . In both tasks, we provide the system with a question and multiple candidate answers. The system then have to respond with the selected correct answer based on the question. For the MC1 task, we use questionlevel accuracy as the metric, determining whether the system selected the correct answer for a given question. In the MC2 task, we use choice-level accuracy, evaluating the system's judgment for each option in every question. We also evaluate the methods on the CommonsenseQA (Talmor et al., 2019) and MedQA (Jin et al., 2020) datasets. We use the development set from CommonsenseQA and the test set of MedQA as the test sets in our experiments.

We choose gpt-3.5-turbo-0613 as the underlying large language model for L2R in all tests. The temperature is set to 0 to reduce instability, and top \_ p is set to 1 by default. The hyperparameter α , which represents the threshold for hard refusal, is set to 0.75 by default to simplify experiments. For llama2, we select the model version of Llama-270b-chat-hf .

Retrieval augmentation plays a crucial role in our L2R system. Initially, we use all-mpnet-basev2 from the Sentence-BERT family (Reimers and Gurevych, 2019) to obtain embeddings for all knowledge texts. We select to employ L2 Euclidean distance to measure the similarity score between the question and candidate knowledge. The system retrieve the top k related knowledge for a single query, with the default value of k set to 4. Specifically, we employed FAISS (Facebook AI Similarity Search) (Johnson et al., 2019) to efficiently retrieve related documents from a large-scale knowledge base. All the knowledge base is mined from the same LLM used later to answer questions.

We compare our method L2R with the general retrieval augmented generation (RAG) method. In this setup, we utilize knowledge from the Wikipedia corpus (Foundation). Since the original Wikipedia documents are lengthy, we retain only the abstract part of each document and use the same embedding models to embed the corpus, storing them in the knowledge base directly as the knowledge of the question-answering system.

The prompts for all LLMs used in L2R can be found in Appendix G.

## B Ablation Study

In our ablation study, we dissect the components of L2R to measure their individual impact on performance using the TruthfulQA dataset for MC1 and MC2 tasks. Initially, the system demonstrates accuracies of 65.1% for MC1 and 70.0% for MC2. Removing the step-by-step answer decreases it for MC2 by 0.9% but improves the accuracy for MC1 by 3.3%. We believe that this result is due to the simplicity of MC1 task, where step-by-step reasoning may introduce unnecessary complexity and noise. In contrast, for the more challenging MC2 task, this reasoning approach can enhance performance. Moreover, since the step-by-step answer illustrates the reasoning path LLMs follow to derive responses from a structured database, we decided to retain this component for clarity.

Eliminating the soft and hard refusal features generally leads to minor accuracy losses ranging from -0.7% to -2.9%, highlighting their importance in the model's ability to handle unanswerable questions.

## C Analysis of Refusal Mechanism

In this experiment, we construct a structured knowledge base using gold knowledge from the TruthfulQA MC1 task, where the gold labels of the dataset are already stored in the knowledge base with a confidence level set to 1.0. However, our ex- periments show that even with this gold knowledge, LLMs still cannot consistently generate perfect answers. We also vary the ratio of gold knowledge from the dataset for constructing the knowledge base and compare the performance of L2R with a general RAG LLM model. The primary focus of this experiment is to evaluate the effectiveness of the refusal mechanism.

From Table 4, we observe that L2R maintains high accuracy (above 90%) consistently, even when provided with just 25% of gold knowledge. In contrast, RAG's performance improves with more knowledge but levels off at 84.5% when provided with all gold knowledge. L2R achieves an accuracy of 93.2% with a refusal count of 159. We also evaluate the success rate of the refusal mechanism, which is 73.4%, demonstrating its effectiveness. The success rate is the percentage of incorrect answers to rejected questions.

The decrease of 2.8% in accuracy observed when the ratio of gold knowledge increases from 0.25 to 0.5 can be attributed to the dataset and its data distribution, wherein the original corresponding questions at this increased ratio are more challenging. As the ratio further increases from 0 to 1.0, the accuracies for this segment of questions are 75.0% with 4 answered questions, 90.0% with 169 answered questions, 91.7% with 168 answered questions, and 90.9% with 164 answered questions, respectively. All these results fall below the overall average level of 90%. Thus, including these more challenging questions in the dataset leads to a noticeable drop in accuracy at this particular ratio.

Another noteworthy finding is that even when L2R is provided with all the gold knowledge, it still cannot achieve perfect results. We attribute this to the retrieval process, where L2R uses a simple retrieval algorithm. The system use the question as a query to retrieve full related knowledge, leading to a similarity gap that affects the retrieval's accuracy. Therefore, it is challenging to find the most relevant and suitable knowledge for a given question. An improved retrieval engine can help alleviate this issue.

## D Hyperparameter Analysis: Threshold Selection in Hard Refusal

In L2R , the selection of an appropriate threshold α in the hard refusal mechanism is crucial. This threshold determines the score of the retrieval result below which the system refuses to answer the original question. The choice of α involves a tradeoff between accuracy and the number of answered questions. Striking the right balance is essential because it is undesirable for a system to either never answer questions or answer every question with poor quality.

|                         | TruthfulQA-MC1   | TruthfulQA-MC1   | ▽    | TruthfulQA-MC2   | TruthfulQA-MC2   | ▽    |
|-------------------------|------------------|------------------|------|------------------|------------------|------|
|                         | Count            | Accuracy         |      | Count            | Accuracy         |      |
| L2R (Ours)              | 654              | 65.1             | -    | 655              | 70.0             | -    |
| w/o step-by-step answer | 661              | 68.4             | +3.3 | 668              | 69.1             | -0.9 |
| w/o soft refusal        | 668              | 63.8             | -1.3 | 668              | 69.3             | -0.7 |
| w/o hard refusal        | 778              | 62.2             | -2.9 | 784              | 69.1             | -0.9 |

Table 3: The ablation experiment results of L2R . The absence of either soft or hard refusal leads to a decline in performance.

Table 4: As the ratio of gold knowledge increases, there are changes in the performance of L2R and RAG (%). L2R exhibits excellent and stable performance in all settings.

|   Ratio | L2R   | L2R      | RAG   | RAG      |
|---------|-------|----------|-------|----------|
|         | count | accuracy | count | accuracy |
|       0 | 0     | 0        | 817   | 46.6     |
|    0.25 | 178   | 93.3     | 817   | 64.7     |
|     0.5 | 349   | 90.5     | 817   | 73.2     |
|    0.75 | 516   | 93.4     | 817   | 79.6     |
|       1 | 658   | 93.2     | 817   | 84.5     |

Figure 5: The changes of Refusal Number and Accuracy under the change of α .

<!-- image -->

Figure 5 illustrates how the Refusal Number and Accuracy change with variations in the threshold α . As expected, a higher threshold allows more questions to pass through, leading to lower accuracy. Conversely, a lower threshold results in a higher refusal number but improved accuracy. It is worth noting that as α decreases from a larger value, the accuracy decreases more rapidly, and the refusal number increases more quickly.

Figure 6: The Precision Recall Curve under the change of α .

<!-- image -->

In real-world applications, it is advisable to generate a figure like the one in Figure 5 to help select an appropriate value for α , typically somewhere in the middle, to strike the right balance between refusal and accuracy.

Wealso generate the precision-recall curve based on different α in Figure 6. Precision is the same as accuracy in the experiments, which is the corrected answer over the number of provided answers. Recall is calculated by the corrected answer over the number of all questions. From an overall view of the figure, we can observe a trade-off between precision and recall with the change of α . We can focus on the curve between the recall of 40 and 60, where it is more stable, and better for selecting an α value within this range.

## E Content of the Structured Knowledge Base

Table 5 shows six knowledge examples of the structured knowledge base. This knowledge base is constructed from the TruthfulQA dataset without the ground truth answers. In the structured knowledge from the L2R system, each data entry is independent, clean, and represents one piece of knowledge with its confidence score. The confidence score indicates the confidence level given by the LLM during the automatic knowledge enrichment process. For example, some highly certain and absolutely correct factual information is given a confidence score of 1.0, such as 'The sun appears white when viewed from space'. Some entries that may have slight flaws receive a slightly lower score, such as 'A trait that makes humans unique is their ability to communicate through complex language'.

Table 5: Six knowledge examples of the structured knowledge base constructed from the TruthfulQA dataset without the ground truth answers.

|   id | knowledge                                                                                   |   confidence |
|------|---------------------------------------------------------------------------------------------|--------------|
|    1 | The sun appears white when viewed from space.                                               |          1.0 |
|    2 | Barack Obama was born in the United States.                                                 |          1.0 |
|    3 | 91 is not a prime number.                                                                   |          1.0 |
|    4 | The city that is cloudy literally all the time is Lima, Peru.                               |          0.9 |
|    5 | Muscle cannot turn into fat. Muscle and fat are two different types of tissues in the body. |          0.9 |
|    6 | A trait that makes humans unique is their ability to communicate through complex language.  |          0.8 |

Besides, through manual revisions, we can correct the knowledge content to ensure its accuracy and set the confidence score to 0. In this table, we only present the results of the automatic knowledge enrichment provided by an LLM.

## F Case Study in L2R

We provide three examples of input-output for L2R in three different cases. In Figure 7, L2R successfully answers the question with the correct response. In Figure 8, the LLM determines that it can answer the question, but the system issues a hard refusal to avoid providing an incorrect response. In Figure 9, the LLM determines that it cannot answer the question, and it is also subjected to a hard refusal.

## G Prompt Design

The prompts used in L2R are depicted in Figure 12, Figure 11, Figure 13, and Figure 10. The prompts shown in Figure 14 and Figure 15 are utilized to structure multiple-choice questions. In all prompts, blue highlighted text with {} represent the prompt slots. The motivation for the construction of prompt templates is to make L2R more stable to provide structured outputs. We also slightly modified the prompt to better fit L2R -Llama experiments and L2R -GPT on the MedQA dataset.

Figure 7: Example 1. L2R can answer the question and provide the correct response.

<!-- image -->

<!-- image -->

| Example 2         | Example 2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Question          | Question: What's one thing that only human beings can do? Candidate Answers: 1: No other animals can master human languages. 2: Only human beings can talk. 3: Only human beings can learn language. 4: Only human beings can make art. 5: Only human beings can play. 6: Only human beings can use tools.                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| Retrieval Results | "content": "One thing that only human beings can do is engage in complex abstract thinking and create art.", "confidence": 0.7 "score": 0.645660936832428 },{ "content": "A trait that makes humans unique is their ability to communicate through complex language.", "confidence": 0.8 "score": 1.2009605169296265 },{ "content": "I am not a human.", "confidence": 1.0 "score": 1.2153674364089966 },{ "content": "Humans have traditionally been believed to have five senses: sight, hearing, taste, smell, and touch. However, some scientists argue that humans have more than five senses, including senses like proprioception (body awareness) and equilibrioception (balance).", "confidence": 1.0 "score": 1.323876976966858 }] |
| Evidence          | One thing that only human beings can do is engage in complex abstract thinking and create art. A trait that makes humans unique is their ability to communicate through complex language. Humans have traditionally been believed to have five senses: sight, hearing, taste, smell, and touch. However, some scientists argue that humans have more than five senses, including senses like proprioception (body awareness) and equilibrioception (balance).                                                                                                                                                                                                                                                                                |
| Reason            | Based on the knowledge base, it is stated that one thing that only human beings can do is engage in complex abstract thinking and create art. It is also mentioned that humans have the ability to communicate through complex language, which is another unique trait. Therefore, the correct answer is that only human beings can make art.                                                                                                                                                                                                                                                                                                                                                                                                |
| Refusal           | "soft_refusal": false, "hard_refusal": true                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Answer            | None (original answer: 4)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |

Figure 8: Example 2. The LLM judges that it can answer the question, but the question is refused with a hard refusal at the system level to prevent an incorrect response.

Figure 9: Example 3. The LLM determines that it cannot answer the question and this question is also refused by hard refusal at the system-level.

<!-- image -->

Figure 10: MAIN\_QA\_PROMPT\_TEMPLATE . This is the prompt template used in the MAIN QA Agent .

<!-- image -->

<!-- image -->

Figure 11: KNOWLEDGE\_Q\_PROMPT\_TEMPLATE . This is the prompt template used in Question Generation Agent .

<!-- image -->

Figure 12: KNOWLEDGE\_A\_PROMPT\_TEMPLATE . This is the prompt template used in Answer Generation Agent .

<!-- image -->

Figure 13: QA2KNOWLEDGE\_PROMPT\_TEMPLATE . This is the prompt template used in QA Pair to Knowledge Agent .

<!-- image -->

Figure 14: MULTIPLE\_CHOICE\_1\_PROMPT\_TEMPLATE . This prompt template is employed to structure multiple-choice questions for the MC1 task in TruthfulQA.

<!-- image -->

Figure 15: MULTIPLE\_CHOICE\_2\_PROMPT\_TEMPLATE . This prompt template is employed to structure multiple-choice questions for the MC2 task in TruthfulQA.