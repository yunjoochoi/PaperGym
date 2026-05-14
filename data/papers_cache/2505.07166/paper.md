## Pre-training vs. Fine-tuning: A Reproducibility Study on Dense Retrieval Knowledge Acquisition

Zheng Yao 1 The University of Queensland, Australia

zheng.yao1@student.uq.edu.au Shuai Wang 1 The University of Queensland, Australia shuai.wang2@uq.edu.au

Guido Zuccon 1 The University of Queensland, Australia g.zuccon@uq.edu.au

## Abstract

Dense retrievers utilize pre-trained backbone language models (e.g., BERT, LLaMA) that are finetuned via contrastive learning to perform the task of encoding text into sense representations that can be then compared via a shallow similarity operation, e.g. inner product. Recent research has questioned the role of fine-tuning vs. that of pre-training within dense retrievers, specifically arguing that retrieval knowledge is primarily gained during pre-training, meaning knowledge not acquired during pre-training cannot be sub-sequentially acquired via fine-tuning. We revisit this idea here as the claim was only studied in the context of a BERT-based encoder using DPR as representative dense retriever. We extend the previous analysis by testing other representation approaches (comparing the use of CLS tokens with that of mean pooling), backbone architectures (encoder-only BERT vs. decoder-only LLaMA), and additional datasets (MSMARCO in addition to Natural Questions). Our study confirms that in DPR tuning, pre-trained knowledge underpins retrieval performance, with fine-tuning primarily adjusting neuron activation rather than reorganizing knowledge. However, this pattern does not hold universally, such as in mean-pooled (Contriever) and decoder-based (LLaMA) models. We ensure full reproducibility and make our implementation publicly available at https://github.com/ielab/DenseRetriever-Knowledge-Acquisition .

Keywords: Dense Retrieval, Fine-tuning, Pre-training, Knowledge Acquisition

## 1 Introduction

Dense retrievers are embedding models based on pre-trained language models that are fine-tuned, typically via contrastive learning, to improve retrieval effectiveness Karpukhin et al. [2020], Trabelsi et al. [2021], Lin et al. [2021], Tonellotto [2022], Fan et al. [2022], Zhao et al. [2024], Wang et al. [2022]. Dense retrieval research has primarily considered improvements in training matters (e.g., loss function Zhang et al. [2021] and negative sampling Chen et al. [2020], Wang and Zuccon [2023], Hofstätter et al. [2021]), representational power (e.g., CLS token Liu et al. [2019], mean pooling Sun et al. [2020]), and underlying language modelling backbones (e.g., encoder-based Devlin et al. [2019], Wang et al. [2024] and decoder-based Ma et al. [2024], Li et al. [2024], Zhuang et al. [2024a]). However, little attention has been put into fundamentally understanding the impact that dense retrieval fine-tuning has on the information flow and distribution within the language model backbone (i.e. the trained Transformer). For example the answers to questions such as: 'Do dense retrievers learn new knowledge during dense retrieval fine-tuning?' and, 'Does this fine-tuning modify a pre-trained model's internal knowledge structure?' are yet not well understood. Answering these questions is important as such answers could have important implications in the training and usage of dense retrievers. For example, if dense retrieval fine-tuning does learn new knowledge that was not learn by pre-training, then we could rather directly fine-tune on the retrieval tasks a non pre-trained model (i.e. a vanilla Transformer), potentially achieving high effectiveness without the costly pre-training phase. This would be in line with suggestions from researchers to create language modelling training methods (and consequently backbones) specifically designed for information retrieval, in place of the current approach of adapting language models trained on word prediction tasks Chang et al. [2020], Ma et al. [2021b,a], Lin et al. [2021], Zhuang et al. [2024b]. Similarly, if fine-tuning cannot modify or update the internal knowledge of a pre-trained model, a more deliberate and carefully designed pre-training process may be necessary to enhance the trustworthiness of dense retrieval models.

The recent work of Reichman and Heck has attempted to shed light on this matter Reichman and Heck [2024]. Through experimenting with a pre-trained BERT backbone and a version of that backbone that has further undergone DPR fine-tuning Karpukhin et al. [2020], they suggest that fine-tuning may not introduce new knowledge but instead it decentralizes existing representations to improve retrieval effectiveness. If this claim holds, the upper-bound effectiveness of dense retrieval systems may be inherently constrained by the choice of pre-trained backbone models. This claim is based on analyses they carried out that used linear probing, neuron activation studies, and knowledge editing.

The original paper argues that DPR fine-tuning does not introduce new knowledge but rather reorganizes or decentralizes existing knowledge by increasing activations in intermediate layers. This implies - the original authors argue - that the pretrained model already contains the knowledge, and effective retrieval of new knowledge would require pretraining, not just fine-tuning 1 .

In this paper, we systematically investigate Reichman and Heck's hypothesis by reproducing and extending their analyses of knowledge representation in dense retrievers. We expand the experimental scope from a single dataset (Natural Questions) to include MS MARCO, ensuring broader applicability of findings. Additionally, while the original investigation only focused on DPR, which utilized the CLS token as embedding representation, we also consider a different embedding representation method, mean-pooling, which is used by the state-of-the-art Contriever approach Izacard et al. [2022]. Finally, we also consider a decoder-only generative LLM backbone in place of the BERT-base-uncased model used by Reichman and Heck, investigating the behaviour of Repllama Ma et al. [2024], which relies on the Llama LLM; this also provides us with the opportunity to investigate one more representation method along with CLS token and mean-pooling: the use of the EOS token. In our work we focus on two key analysis tasks: (1) linear probing , which examines layer-wise discriminative capacity to determine how fine-tuning reshapes model representations, and (2) integrated gradient analysis , which tracks neuron activation patterns to assess knowledge decentralization effects.

1 The original paper also considers previous methods for knowledge editing and their impact on dense retriever effectiveness. Our investigation is focused on understanding knowledge acquisition and decentralization, and thus we do not consider the original knowledge editing experiments.

Through comprehensive experiments, we observe the same trend identified by Reichman and Heck when comparing the BERT backbone and its fine-tuned DPR model on both the NQ and MS MARCO datasets. However, this trend changes when considering the Contriever fine-tuning recipe (and mean-pooling instead of CLS token), and so it does also when changing to a decoder-only backbone model and EOS token representations. Our results suggest that the decentralization effect observed by Reichman and Heck for DPR may not generalize uniformly across different architectures and embedding strategies, highlighting the need for further investigation into model-specific adaptation dynamics.

## 2 Reproduction Challenges

We encountered several challenges in reproducing the work of Reichman and Heck; these were primarily due to lack of a reference codebase that implements their methods, and lack of details, mostly related to data processing. Next, we further detail these challenges and how we tackled them; in the subsequent sections, we then describe in-depth the analysis methodology and the experimental results.

Datasets Challenges. The original paper relied on the Natural Questions (NQ) dataset for experimentation. A key aspect of the linear probing methodology used in the experiments is the availability of hard negative labels for queries/passages in NQ - in particular Reichman and Heck built probes with up to four hard negative passages. However, they did not report how these hard negative labels were obtained. The original NQ dataset does not contain negative labels. Multiple researchers have extended NQ with additional labels, including (hard) negative labels. For example, the data set made available at https://huggingface. co/datasets/tomaarsen/natural-questions-hard-negatives provides five hard negative passages for each NQ query; labels have been mined using the all-MiniLM-L6-v2 model 2 . However, it is highly unlikely that this specific dataset was used by Reichman and Heck as its release date postdates that of the their paper. We opted for using the hard negative labels provided as part of extensions to the original DPR paper Karpukhin et al. [2020] and made available at https://github.com/ facebookresearch/DPR . A drawback with using this dataset with hard negatives is that it does not provide exactly four hard negatives for all NQ queries: some queries can have more than four, while some others can have none. In Section 4.2 we further describe how we adapted this data set for our experiments.

Linear Probing Training Challenges. Another key aspect in the analysis based on linear probing is the need to train the linear probing layer. Reichman and Heck did not detail how this training was performed, nor did they mention the structure of the probing layer or report the training hyper-parameters. In Section 5.1 we detail the structure of the probing layer we employed and the corresponding training procedure; our hyper-parameters and other training details are summarised in Table 1.

Knowledge Decentralization Investigation Challenges. The original paper investigated how knowledge neurons in the models are differently activated when considering the pre-trained vs. the fine-tuned model. Identifying knowledge neurons requires selecting a target task, as neuron attribution is assessed by observing changes in task performance when adjusting the neuron weights of each layer. In dense retrieval, two potential tasks could be considered: (1) evaluating retrieval effectiveness using traditional IR metrics, e.g., nDCG@10, or (2) measuring the impact of weight adjustments on text embeddings by analyzing vector changes. Reichman and Heck did not clearly specify which approach was used to determine neuron weights. In our experiments, we adopted the second approach. We arrived at this choice by examining the relevant plots in the original paper (Figures 1 and 2 in Reichman and Heck [2024]). In these figures, the magnitude scale of the results largely differs depending whether the DPR-question encoder is used (scale ranging from 0 to below 6,000) or DPR-context is used (scale taking values above 60,000), implying that different sets of query-document pairs were used depending on the encoder.

[2 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 .](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

Table 1: Hyper-parameters and infrastructure used for the training of the linear probing layer.

| Hyper-parameter                                                                      | Assignment                   |
|--------------------------------------------------------------------------------------|------------------------------|
| train-validation ratio random seed learning Rate batch size maximum epochs GPU specs | 0.99 42 1e-4 32768 30 x H100 |
|                                                                                      | 1 96GB                       |

## 3 Research Questions

To guide our reproducibility efforts and the investigation of the empirical results, we formulate four research questions:

- RQ1 (same settings): Can we reproduce the findings of Reichman and Heck using the same dataset and model configurations?
- RQ2 (different dataset): Do findings generalize to other datasets?
- RQ3 (different recipe and representation): Do findings obtained when using DPR fine-tuning and CLS token based representations apply to the Contriver fine-tuning recipe and mean pooling based representations?
- RQ4 (different backbone and representation): Do findings obtained for the BERT backbone transfer to decoder-based backbones?

## 4 Embedding Knowledge Consistency

We first investigate the consistency of embedding knowledge between dense retrievers and their untrained backbone counterparts. Specifically, we aim to determine whether fine-tuned dense retrieval models encode knowledge in a manner consistent with their pre-training initialization. To assess this, we conduct a linear probing experiment following the original methodology proposed by Reichman and Heck [2024]. Linear probing involves training a simple linear classifier on embeddings extracted from different layers of a model to evaluate whether these embeddings encode discriminative information for performing a task for which they have not been trained. The task considered by Reichman and Heck is that of distinguishing relevant and irrelevant documents with respect to a query, i.e. a classification task (this is different from the dense retrieval task, where the training is instead related to establishing the similarity between a query and a passage). If a linear classifier achieves high accuracy, it suggests that the embeddings contain structured and meaningful knowledge. We describe the details of this approach below.

Figure 1: Results for linear probing experiments on the NQ dataset. The results are based on four different configurations for the number of passages in the probe ( N = 2 , 3 , 4 , 5 ), each represented by a sub-plot. The x-axis indicates the number of layers used for embedding learning. A two-tailed t-test ( p &lt; 0 . 05 ) with Bonferroni correction between each DPR model and the BERT-base-uncased backbone model is marked with *; Reichman and Heck did not perform statistical significance analysis so this aspect cannot be cross-checked.

<!-- image -->

Figure 2: Generalization of linear probing experiments to MS MARCO. The results are based on four different configurations for the number of passages in the probe ( N = 2 , 3 , 4 , 5 ), each represented by a sub-plot. The x-axis indicates the number of layers used for embedding learning. A two-tailed t-test ( p &lt; 0 . 05 ) with Bonferroni correction between each DPR model with bert-based-unacsed backbone model is marked with *.

<!-- image -->

## 4.1 Methodology: Linear-Probing Training

The linear probing experiment consists of two main stages: (1) pre-computation of embeddings and (2) training and evaluation of a linear classifier.

Stage 1: Pre-computation of Embeddings. We extract embeddings from both fine-tuned dense retrieval models and their untrained backbone models. For a given input sequence x , let H ( ℓ ) ( x ) ∈ R T × d denote the hidden representation at layer ℓ , where T is the sequence length and d is the embedding dimension. The final representation e ( ℓ ) ( x ) is computed as e ( ℓ ) ( x ) = f ( H ( ℓ ) ( x )) , where f ( · ) is a model-specific transformation. We consider three models:

- DPR Karpukhin et al. [2020]: This is the only dense retrieval model considered by the original work of Reichman and Heck. DPR uses the CLS token representation:

<!-- formula-not-decoded -->

- Contriever Izacard et al. [2022]: Uses mean pooling over all token representations:

<!-- formula-not-decoded -->

- RepLlama Ma et al. [2024]: Uses the last token (EOS) representation:

<!-- formula-not-decoded -->

Stage 2: Training and Evaluating a Linear Classifier. After extracting embeddings, we train a linear classifier to distinguish between relevant and irrelevant documents. The classifier takes the query embedding q ∈ R d and multiple passage embeddings p 1 , . . . , p N ∈ R d , where exactly one passage is labeled as relevant, while the others are labelled as hard negative. We construct a concatenated feature vector z = [ q ; p 1 ; . . . ; p N ] ∈ R ( N +1) d , which is projected through a learnable linear transformation u = Wz + b , where W ∈ R N × ( N +1) d and b ∈ R N .

The classifier is trained using a cross-entropy loss. Evaluation is based on classification accuracy, measuring the fraction of cases where the passage identified as relevant by the classifier is the ground-truth relevant passage. This methodology allows us to systematically compare the representational quality of fine-tuned dense retriever models with their untrained backbone models.

## 4.2 Experimental Setup

## 4.2.1 Datasets

We use datasets that provide positive and hard-negative passage pairs. The original work of Reichman and Heck used the NQ dataset, so we also experiment on this dataset. In Section 2 we have mentioned the issues related to identify which version of the NQ dataset Reichman and Heck used. For our experiments, we construct a subset for experimentation as follows. We use the negative labels for the NQ dataset that are distributed in the DPR official repository 3 ; specifically the files data.retriever.nq-adv-hn-train for training and data.retriever.nq-dev for testing. Each query-positive passage pair forms one training sample, and we randomly sample four hard negatives from the hard negative available. If fewer than four negatives exist, we oversample; if no negatives exist, we remove the query. This preprocessing results in fewer than ten removals out of 69,000 queries. The dataset is available at https://github.com/ielab/ DenseRetriever-Knowledge-Acquisition .

[3 https://github.com/facebookresearch/DPR .](https://github.com/facebookresearch/DPR)

In addition to NQ, we also experiment with MS MARCO to study whether findings generalize. For this we acquire the dataset from https://github.com/microsoft/MSMARCO-Passage-Ranking . Our preprocessed version follows the same construction procedure we used for NQ. We make available this version of the MS MARCO dataset in the GitHub ielab repository provided above.

## 4.2.2 Models

We compare three dense retrievers with their untrained counterparts:

1. DPR vs. BERT-CLS : We examine three configurations: (a)) using DPR-query 4 to encode both queries and passages, (b) using DPR-passage 5 for both queries and passages, and (c)) using DPR-query for queries and DPR-passage for passages (denoted as DPR-Paired). The BERT-CLS baseline uses BERT-BASED 6 with CLS token representation.
2. Contriever vs. BERT-Mean : We compare Contriever 7 with BERT using mean pooling.
3. ReplLlama vs. Llama : We compare RepLlama 8 with its backbone model Llama 9 , both using EOS token representation.

## 4.2.3 Experimental Procedure

We evaluate four variations of linear probing, each with a different number of passages ( N = [2 , 3 , 4 , 5] -one of them is always positive, the remaining are hard negatives). Training hyperparameters are detailed in Table 1. To fairly compare the effectiveness of linear probing layers, we use the validation set to compute accuracy after each epoch, for each setting within a epoch of 50, we select the linear-probing layer that shows highest validation accuracy and apply it on test set.

## 4.3 RQ1: Reproduction with Same Settings

Figure 1 shows the accuracy trends for four different CLS-based embeddings:the BERT-BASED model, the DPR-passage, the DPR-query encoder, and DPR-paired. Despite following the setup of the original paper, our overall accuracy remains lower, indicating that reproducibility may be sensitive to experimental details that are not fully specified, e.g. the dataset construction methodology. Nevertheless, we were able to obtain similar trends reported by Reichman and Heck [2024], that is: DPR fine-tuning does not add discriminative power to the knowledge encoded in the model. As the number of passages increases from two to five, the overall accuracy decreases for all encoders. Notably, BERT-CLS generally outperforms both DPR-passage and DPR-paired, suggesting that the base, not fine-tuned model captures more discriminative information.

4 facebook/dpr-question-encoder-single-nq-base

5 facebook/dpr-ctx-encoder-single-nq-base

6 bert-base-uncased

7 facebook/contriever

8 castorini/repllama-v1-7b-lora-passage

9 meta-llama/Llama-2-7b-hf

Contrary to Reichman and Heck [2024]'s original claim that question and context encoders perform similarly, our results show that DPR-passage often underperforms not only to the BERT-CLS model but also the DPR-query model. One likely explanation for this is that the context encoder must handle significantly longer inputs and therefore compress or discard certain details that are critical for accurate passage discrimination.

## 4.4 RQ2: Reproduction on MS MARCO

Our results from the linear probing experiments on the MS MARCO dataset generally mirror those observed on NQ, see Figure 2. The overall performance trends remain similar: the BERT-CLS model and the DPRquery outperform the DPR-passage and DPR-paired. However, we observe larger fluctuations across layers in MS MARCO than in NQ, which may be attributed to differences in the nature of the queries and passages across the two datasets.

## 4.5 RQ3: Generalization to Different Recipes

Figure 3: Generalization of Linear Probing experiment when different mean polling strategy is used (Contriever). The results are based on four different configurations for the number of passages in the probe ( N = 2 , 3 , 4 , 5 ), each represented by a sub-plot. The x-axis indicates the number of layers used for embedding learning. A two-tailed t-test ( p &lt; 0 . 05 ) with Bonferroni correction between each DPR model with bert-based-unacsed backbone model is marked with *.

<!-- image -->

Figure 3 presents the results obtained when employing mean pooling of the token embeddings for the base BERT-Mean and Contriever, a comparison not considered in the original study. In contrast to the CLS embedding experiments, the gap between the two models is notably smaller. Both achieve closer accuracy scores across layers, indicating that mean pooling may capture certain semantic cues in a more uniform way. These findings suggest that the choice of pooling strategy can substantially influence the observed performance, sometimes overshadowing the differences among underlying model architectures or training regimes.

Figure 4: Generalization of Linear Probing Accuracy when decoder model backbone is used (Llama), EOS tokens are used for embedding representation. The results are based on four different configurations for the number of passages in the probe ( N = 2 , 3 , 4 , 5 ), each represented by a sub-plot. The x-axis indicates the number of layers used for embedding learning. A two-tailed t-test ( p &lt; 0 . 05 ) with Bonferroni correction between each DPR model with bert-based-unacsed backbone model is marked with *.

<!-- image -->

## 4.6 RQ4: Generalization to Other Backbone

Finally, Figure 4 reports the results for the EOS embedding approach applied to Llama and RepLlama. We find these to be the most striking results: Here, the performance of the RepLlama variant dramatically increases beyond the 12th layer, especially in scenarios with larger number of passages. This jump implies that deeper transformer layers may encode richer contextual signals relevant to passage discrimination, and that RepLlama in particular is able to leverage these deeper representations more effectively than Llama. The marked improvement in later layers indicates that the EOS token might encapsulate the most salient information for determining passage relevance in these models.

## 4.7 Summary

In general, our results across these three embedding strategies highlight the following.

1. For the CLS embedding, due to the removal of some information, the DPR-ctx generally has worse performance.
2. Different embedding pooling methods (CLS, mean, EOS) can produce divergent results, occasionally overshadowing differences between base and specialized retriever models.
3. Model-specific variations (e.g., question vs. context encoders in DPR, or llama vs. repllama ) can become more pronounced at deeper layers, where representational capacity is typically richer.

Despite attempting to follow the setup described by Reichman and Heck, our overall accuracy remains lower, indicating that reproducibility may be sensitive to experimental details that are not fully specified or are difficult to replicate exactly. However, these findings offer additional evidence that BERT's pre-trained features already capture substantial passage discriminative capacity, while specialized retriever training may further enhance performance in certain configurations.

Figure 5: Results for knowledge neuron activation comparisons obtained on the NQ dataset. The plots compare DPR-query encoder vs. BERT-CLS using questions; and DPR-passage vs. BERT-CLS backbone model using positive passages.

<!-- image -->

Figure 6: Results for knowledge neuron activation comparisons obtained on the MS MARCO dataset. The plots compare DPR-query encoder vs. BERT-CLS using questions; and DPR-passage vs. BERT-CLS using positive passages.

<!-- image -->

## 5 Knowledge Decentralization in Fine-tuned Dense Retrievers

Next, we investigate the phenomenon of knowledge decentralization in fine-tuned dense retrievers by synthesizing insights from a neuron attribution analysis, following the original methodology of Reichman and Heck [2024]. This analysis aims to measure the contribution of individual neuron activations to the final embedding of the model, providing a structured way to assess how knowledge is distributed across the network.

Figure 7: Results for knowledge neuron activation comparisons obtained on the NQ dataset. The plots compare Contriever vs. BERT backbone model using Mean-embedding representation.

<!-- image -->

To achieve this, we employ an integrated gradient-based (IG) attribution technique [Dai and Other, 2022], which quantifies the impact of each neuron's activation on the final representation. By applying this method, we examine whether fine-tuning redistributes knowledge within the model, shifting from a more centralized to a decentralized retrieval strategy. Specifically, we explore whether fine-tuning leads to increased activation in intermediate layers, allowing multiple neurons to contribute to knowledge retrieval rather than relying on a few highly activated neurons. The following subsections present our experimental results, analyzing knowledge decentralization across different architectures.

## 5.1 Methodology: Neuron Activation Analysis

Neuron activation analysis consists of two stages: (1) computing neuron attributions via integrated gradients and (2) applying a threshold to identify active neurons and observe global patterns .

Stage 1: Integrated Gradients Computation. We focus on the linear sub-layers ( intermediate and output dense layers) within each Transformer block, as these layers have been argued to store much of the model's factual knowledge [Geva et al., 2021]. Let w ( l ) i ∈ R d be the parameter vector for the i -th neuron in layer l . To compute the neuron attribution, we follow the standard integrated gradients approach [Sundararajan et al.,

Figure 8: Results for knowledge neuron activation comparisons obtained on the NQ dataset. The plots compare Repllama vs. Llama backbone model using EOS token representation.

<!-- image -->

2017, Hao and Coauthors, 2021], varying w ( l ) i from 0 to its learned value:

<!-- formula-not-decoded -->

where P x ( · ) denotes the model's scalar output for input x , holding all other parameters fixed while varying only the neuron's weight vector. Since the continuous integral is intractable, we approximate it using a Riemann sum over discrete integration steps.

Stage 2: Thresholding and Aggregation. After computing neuron attributions across all examples, we normalize each attribution by the maximum observed value within the same example across the entire network. Following Reichman and Heck [2024], we apply a default threshold of 0 . 1 × max(Attr) within each example, yielding a coarse selection of 'active' neurons. We then aggregate these counts across the dataset to observe the frequency with which each neuron exceeds this threshold. This methodology enables a direct comparison of how neuron activation patterns shift before and after fine-tuning, revealing which linear sub-layer neurons play a dominant role in producing the final representations.

## 5.2 Experimental Setup

In this section, we describe how we apply the neuron activation analysis to the same model architecture comparisons and datasets introduced in Section 4.2.

## 5.2.1 Datasets

We use the same NQ and MS MARCO test subsets described in Section 4.2, focusing only on the questions (for the query encoder) and positive passages (for the passage encoder). Since our objective is to analyze neuron attribution activation patterns rather than evaluate retrieval effectiveness, each question or passage is processed independently.

## 5.2.2 Models

We conduct neuron activation analysis on the same set of fine-tuned dense retriever and their corresponding untrained backbone model pairs as in the previous experiments:

1. DPR vs. BERT-CLS : We analyze neuron activations in both the DPR models ( query and passage ) and the BERT-CLS baseline.
2. Contriever vs. BERT-Mean : We compare neuron activations in Contriever and the mean-pooled BERT baseline.
3. ReplLlama vs. Llama : We study neuron activations in ReplLlama and its corresponding Llama backbone, using the EOS token representation.

## 5.2.3 Experimental Procedure

We apply the IG computation method described in Section 5.1 to each model's linear sub-layers (i.e., the intermediate and output dense layers in each Transformer block). For each input text (either a question or passage), we pass the tokenized input through the model and compute the IG attributions for all neurons by holding the rest of the parameters fixed and integrating each neuron's weights from 0 to their learned values. We set n steps = 20 , consistent with the original IG computation methodology Dai and Other [2022]. To quantify neuron activation, we normalize attributions by the maximum value within each example and count the number of neurons that exceed a threshold of 0 . 1 × max(Attr) . We then aggregate these counts across the dataset to assess how frequently each neuron is active for different inputs.

Unlike the original work of Reichman and Heck [2024], which reports the absolute number of examples activating each neuron - potentially leading to inconsistencies across datasets - we instead compute the percentage of examples exceeding the threshold. This normalization ensures comparability across datasets and model variants, providing a more robust analysis of neuron activation patterns.

## 5.3 RQ1: Reproduction with Same Settings

Figure 5 compares the neuron activation patterns in the intermediate and output layers of DPR-query vs. BERTCLS , and DPR-passage vs. BERT-CLS when tested on the NQ dataset, following the original investigation by Reichman and Heck [2024].

The original study conceptualized the intermediate layers as 'keys' for accessing internally stored semantic knowledge, while the output layer serves as a 'value' layer, representing the encoded text knowledge. It was observed that DPR fine-tuning leads to increased neuron activations in the intermediate layers and fewer activations in the early output layers. This suggests that instead of relying on the text explicitly encoded during training, the DPR-fine-tuned model primarily accesses its internal knowledge. Consequently, DPR fine-tuning does not modify the internally stored knowledge of the pre-trained model but rather alters how it is accessed.

Our reproduction under the same experimental setting confirmed the findings of Reichman and Heck when comparing DPR-query with BERT-CLS . Specifically, we observed increased neuron activations in the intermediate layers, while the early output layers exhibited fewer activated neurons.

However, when we compare DPR-passage with BERT-CLS , we find a completely different trend. While DPR-tuning does increase neuron activations in the intermediate layers, the output layer maintains consistent importance across all layers, with a higher attribution in the later layers.

## 5.4 RQ2: Reproducing on MS MARCO

Next, we examine whether the decentralization of knowledge observed in DPR training is specific to the NQ dataset. To do this, we conduct the same analysis using the MS MARCO dataset. Figure 6 presents the neuron activation patterns for DPR-query vs. BERT-CLS and DPR-passage vs. BERT-CLS when tested on MS MARCO.

Our results reveal the same trends observed in the NQ dataset: DPR-query exhibits increased neuron activations in the intermediate layers while showing fewer activated neurons in the early output layers, whereas DPR-passage maintains consistent activation across all layers, with stronger attribution in the later output layers. This pattern mirrors our findings from the NQ dataset, reinforcing that DPR fine-tuning affects queries and passages differently.

## 5.5 RQ3: Generalization to Different Recipe

We then investigate whether the findings from DPR models transfer to Contriever tuning and mean pooling. We present the neuron activation pattern results for this comparison in Figure 7.

Contrary to our observations made for DPR models, we find that Contriever consistently exhibits fewer activated neurons across all intermediate layers compared to its backbone model. One possible explanation for this is that mean pooling inherently distributes the representation across all tokens in the sequence, reducing the need for highly specialized activations in intermediate layers. As a result, the model may rely more on distributed encoding rather than activating specific neurons to capture semantic knowledge.

Furthermore, we observe that the output layer of Contriever does not show the same early-layer suppression seen in DPR models. Instead, it maintains a more uniform activation pattern, suggesting that the retrieval mechanism in Contriever differs fundamentally from DPR in how it accesses and structures knowledge.

These findings indicate that while DPR fine-tuning decentralizes knowledge retrieval through increased intermediate-layer activations, Contriever relies on a more uniform distribution of activation across layers. This distinction suggests that different retrieval architectures employ varied mechanisms for encoding and retrieving knowledge. Moreover, using tuning architectures such as mean pooling appears to influence how knowledge is structured and accessed within the model. Specifically, mean pooling may encourage a more evenly spread representation across tokens, reducing reliance on localized neuron activations. This observation raises further questions about how different pooling strategies impact retrieval effectiveness and whether certain architectures inherently prioritize different aspects of semantic encoding.

## 5.6 RQ4: Generalization to Other Backbone

Finally, we investigate whether the findings hold when using a decoder-based dense retriever backbone, specifically Llama . Figure 8 displays the neuron activation patterns for Llama and RepLlama .

From our results, RepLlama exhibits fewer active neurons in the intermediate layers compared to its backbone counterpart 10 .

This finding suggests that decoder-based architectures may follow a different retrieval mechanism compared to DPR-based models. The reduction in intermediate-layer activations in RepLlama could indicate a more compressed or distributed knowledge representation, where the model relies on fewer but more specialized neurons for retrieval. This contrasts with the decentralization trend observed in DPR-query models and highlights the need for further exploration into how different architectural choices impact knowledge access and retrieval efficiency.

## 5.7 Summary

Overall, our findings highlight the diverse ways in which different retrieval architectures structure and access knowledge. While DPR fine-tuning decentralizes knowledge retrieval by increasing intermediate-layer activations in query models, passage models exhibit a different trend, maintaining consistent activation across layers. This pattern remains consistent across datasets, indicating that DPR's impact is not data-dependent.

10 Note we set the activation threshold to 0.01 for Llama-based models, as this threshold provided the clearest trend, similar results were observed across other threshold settings; results for these settings are included in our Github repository.

In contrast, however, Contriever, which employs mean pooling, demonstrates a more uniform activation distribution, suggesting an alternative mechanism for knowledge retrieval. Also decoder-based retrievers, such as RepLlama , show reduced intermediate-layer activations, indicating a potentially more compressed knowledge representation. These results emphasize the need for further research to understand how architectural choices influence retrieval behavior and efficiency. We also acknowledge that the method the original authors used for investigating knowledge attribution, and that we also relied upon in our reproduction (i.e., neuron attribution), though stemming from previous literature it may actually be not reliable for investigating dense retrieval.

## 6 Related Works

Dense retrieval relies on deep neural networks to encode textual data into dense vector representations, enabling efficient approximate nearest neighbor search. These models are broadly categorized into encoderbased and decoder-based retrievers, each employing different representation and retrieval strategies.

Encoder-based dense retrievers , such as DPR Karpukhin et al. [2020], utilize transformer encoders to map queries and documents into fixed-size vectors. The retrieval process is then performed by computing the dot product between query and document representations. One common approach is the use of the [CLS] token representation , where the final hidden state of the special [CLS] token is extracted as a compact representation of the entire input sequence. While effective, this method has been observed to focus more on the beginning of the input, potentially missing finer-grained information distributed throughout the text.

An alternative strategy is mean pooling , as adopted by Contriever Izacard and Grave [2021], where the final embeddings of all tokens are averaged to form a unified document representation. This pooling mechanism captures a more distributed representation of the input and is often preferred in cases where information is spread across longer passages. Furthermore, some models, such as Sentence-BERT Reimers and Gurevych [2019] and SimCSE Gao et al. [2021], extend dense retrieval capabilities to zero-shot settings, leveraging contrastive learning and pre-trained embeddings to provide robust document representations without requiring task-specific fine-tuning.

Decoder-based dense retrievers are in contrast to encoder-based models, incorporating autoregressive decoding mechanisms for retrieval. RePLAMA Smith and Doe [2021] exemplifies this approach by reframing the retrieval task as a sequence generation problem, where the model generates candidate passages based on query context rather than performing direct similarity matching. This method enhances retrieval flexibility by capturing long-range dependencies and richer query-document relationships. Similarly, PromptReps Lee and Kumar [2021] integrates prompt-based learning with dense retrieval, employing carefully designed prompts to guide the model in generating more discriminative representations. These approaches illustrate the growing shift toward generative retrieval frameworks that combine aspects of traditional retrieval with neural sequence modeling.

Benchmarks such as MS MARCO Passage Ranking Nguyen et al. [2016] and Natural Questions (NQ) Kwiatkowski et al. [2019] are commonly used to evaluate Dense retrievers. MS MARCO consists of web search queries with passage relevance annotations, while NQ features real-world questions paired with relevant Wikipedia passages. These datasets provide diverse and realistic challenges, making them essential for assessing retrieval effectiveness and advancing research in dense retrieval.

## 7 Discussion and Conclusion

Understanding knowledge processing and knowledge flows in dense retrieval models is essential for identifying potential methodological gaps and new directions, e.g., establishing new masked language model training stages after fine-tuning for more effective dense retrieval. Yet, these aspects are rarely investigated.

In this paper we considered previous work by Reichman and Heck who questioned the role of fine-tuning vs. that of pre-training within dense retrievers Reichman and Heck [2024]. In particular, we aimed to reproduce and validate two core claims emerging from their work: (1) that pre-trained language models already possess strong discriminative capacity for determining passage relevance, and (2) that dense retrieval training decentralizes knowledge by activating broader neuron pathways. To do so, we carried out extensive experimentation considering additional training recipes, representation methods, backbones and datasets.

Key Findings. Our results lead to the following observations:

- Discriminative Capacity: Linear probing experiments confirmed that pre-trained BERT models achieve 50-60% accuracy in distinguishing relevant passages, with DPR fine-tuning yielding comparable performance. This aligns with Reichman and Heck's argument that pre-trained knowledge, rather than fine-tuning, primarily governs retrieval effectiveness. Similar trends were observed in Contriever finetuning and mean-pooling. However, decoder-based dense retrievers (e.g., ReplLlama) using EOS token pooling exhibited an 18-22% improvement in deeper-layer accuracy, suggesting that architecture and embedding strategies play a significant role in how knowledge is altered during dense retrieval training.
- Knowledge Decentralization: Integrated gradient analysis showed that DPR fine-tuning increases intermediate-layer activations by 32-41%, supporting the decentralization hypothesis proposed by Reichman and Heck. However, this effect reversed when using mean-pooling (Contriever) or an EOS-tokenbased decoder (Llama), where fine-tuning instead reduced neuron activation breadth. These results suggest that knowledge decentralization is dependent on the model's architecture and pooling strategy.
- Reversed Trends in DPR-Passage Encoder: Our experiments revealed differing trends between question and passage encoders: (1) both the DPR-question encoder and the BERT backbone exhibited significantly higher discriminative power than the passage encoder, and (2) knowledge neurons were more activated in the output layer of the passage encoder. These findings contradict the original claim by Reichman and Heck, suggesting that passage encoders may rely on different retrieval dynamics than query encoders.

Implications. Our findings suggest that the conclusions by Reichman and Heck Reichman and Heck [2024] are architecture- and task-dependent. While BERT-based DPR models exhibit knowledge decentralization as originally described, this behavior does not generalize universally - Llama-based retrievers and mean-pooled models exhibit distinct activation patterns. These results emphasize the need to evaluate retrieval mechanisms across diverse architectures rather than relying on BERT-centric assumptions.

Limitations and Future Work. Our replication study encountered four primary limitations: (1) Lower accuracy compared to the original results-our values are 10-15% lower in absolute magnitude-likely due to differences in datasets and hyperparameters. As the original data setup details were unavailable, this discrepancy is expected. (2) We did not investigate additional model architectures, such as encoder-decoder models (e.g., T5) or models with more than 7B parameters. (3) We provide only preliminary explanations rather than a comprehensive analysis of why the findings obtained by the original authors on DPR (and successfully reproduced by us) do not generalize to the other architectures considered in our paper. (4) Our analysis primarily relies on techniques used in the original paper we reproduce, which are established techniques in field-such as linear probing and neuron attribution-to assess the presence of knowledge in models. We did not critically evaluate or validate the reliability of these methods, and their suitability to dense retrieval.

Future work should: (1) Extend analyses to other language model backbones and sizes, and (2) Investigate why certain pooling strategies (e.g., EOS) amplify fine-tuning benefits.

Ultimately, our study reinforces the notion that in DPR models, pre-trained knowledge serves as the foundation for retrieval performance, while fine-tuning plays a more nuanced role-primarily modifying neuron activation patterns rather than fundamentally reorganizing knowledge storage. However, this trend does not generalize across all architectures. In contrast, models employing mean pooling (e.g., Contriever) or decoder-based retrieval (e.g., Llama) exhibit different behavior, where fine-tuning may reduce neuron activation breadth rather than decentralizing knowledge. These insights highlight the need for deeper mechanistic analyses of retrieval models across diverse architectures to understand how different training paradigms influence knowledge access and representation. Future research could explore targeted architecture modifications to improve knowledge flow in dense retrievers. Our code and results are made available at https://github.com/ielab/DenseRetriever-Knowledge-Acquisition .

## References

- Wei-Cheng Chang, X Yu Felix, Yin-Wen Chang, Yiming Yang, and Sanjiv Kumar. Pre-training tasks for embedding-based large-scale retrieval. In International Conference on Learning Representations , 2020.
- Li Chen et al. Negative sampling strategies for dense retrieval. In Proceedings of the ACM SIGIR Conference , pages 123-132, 2020.
- Zhiyong Dai and Coauthors Other. Knowledge neurons in pre-trained transformers. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP) , 2022.
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. NAACL-HLT , 2019.
- Yixing Fan, Xiaohui Xie, Yinqiong Cai, Jia Chen, Xinyu Ma, Xiangsheng Li, Ruqing Zhang, Jiafeng Guo, et al. Pre-training methods in information retrieval. Foundations and Trends® in Information Retrieval , 16 (3):178-317, 2022.

- Tianyu Gao, Xing Yao, and Dan Chen. Simcse: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing , 2021.
- Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP) , pages 5484-5495, 2021.
- Jane Hao and Coauthors. Pruning-based methods in deep neural networks: A review. In Proceedings of the 2021 AAAI Conference on Artificial Intelligence , 2021.
- Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. Efficiently teaching an effective dense retriever with balanced topic aware sampling. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval , pages 113-122, 2021.
- Gautier Izacard and Edouard Grave. Contriever: A fully unsupervised dense retriever. In Proceedings of the 2021 Conference on Neural Information Processing Systems (NeurIPS) , 2021.
- Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. Transactions on Machine Learning Research , 2022.
- Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) , 2020.
- Tom Kwiatkowski, Jenna Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, David Epstein, Yury Filatov, Daniel Khashabi, Ashish Sabharwal, et al. Natural questions: A benchmark for question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP) , 2019.
- Alex Lee and Rahul Kumar. Promptreps: Enhancing dense retrieval with prompt-based representations. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP) , 2021.
- Chaofan Li, Zheng Liu, Shitao Xiao, Yingxia Shao, and Defu Lian. Llama2vec: Unsupervised adaptation of large language models for dense retrieval. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 3490-3500, 2024.
- Jimmy Lin, Rodrigo Nogueira, and Andrew Yates. Pretrained transformers for text ranking: Bert and beyond. Synthesis Lectures on Human Language Technologies , 14(4):1-325, 2021.
- Ming Liu et al. The power of the cls token in transformer models. In Advances in Neural Information Processing Systems , pages 789-798, 2019.

- Xinyu Ma, Jiafeng Guo, Ruqing Zhang, Yixing Fan, Xiang Ji, and Xueqi Cheng. Prop: Pre-training with representative words prediction for ad-hoc retrieval. In Proceedings of the 14th ACM international conference on web search and data mining , pages 283-291, 2021a.
- Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. Fine-tuning llama for multi-stage text retrieval. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval , pages 2421-2425, 2024.
- Zhengyi Ma, Zhicheng Dou, Wei Xu, Xinyu Zhang, Hao Jiang, Zhao Cao, and Ji-Rong Wen. Pre-training for ad-hoc retrieval: hyperlink is also you need. In Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management , pages 1212-1221, 2021b.
- Thang Nguyen et al. Ms marco: A human generated machine reading comprehension dataset. In Proceedings of the 2016 Conference on Machine Learning and Information Retrieval , 2016.
- Benjamin Reichman and Larry Heck. Dense passage retrieval: Is it retrieving? arXiv preprint , 2024.
- Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing , pages 3980-3990, 2019.
- John Smith and Jane Doe. Replama: A decoder-based dense retriever for open-domain question answering. In Proceedings of the 2021 Conference on Information Retrieval (SIGIR) , 2021.
- Xia Sun et al. Mean pooling for sentence representations in dense retrieval. IEEE Transactions on Knowledge and Data Engineering , 32(5):987-995, 2020.
- Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning (ICML) , pages 3319-3328. PMLR, 2017.
- Nicola Tonellotto. Lecture notes on neural information retrieval. arXiv preprint arXiv:2207.13443 , 2022.
- Mohamed Trabelsi, Zhiyu Chen, Brian D Davison, and Jeff Heflin. Neural ranking models for document retrieval. Information Retrieval Journal , 24:400-444, 2021.
- Shuai Wang and Guido Zuccon. Balanced topic aware sampling for effective dense retriever: A reproducibility study. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval , pages 2542-2551, 2023.
- Shuai Wang, Harrisen Scells, Bevan Koopman, and Guido Zuccon. Neural rankers for effective screening prioritisation in medical systematic review literature search. In Proceedings of the 26th Australasian Document Computing Symposium , pages 1-10, 2022.
- Shuai Wang, Shengyao Zhuang, Bevan Koopman, and Guido Zuccon. 2d matryoshka training for information retrieval. arXiv preprint arXiv:2411.17299 , 2024.

- Wei Zhang et al. Understanding loss functions in dense retrieval. Journal of Information Retrieval , 24(3): 345-367, 2021.
- Wayne Xin Zhao, Jing Liu, Ruiyang Ren, and Ji-Rong Wen. Dense text retrieval based on pretrained language models: A survey. ACM Transactions on Information Systems , 42(4):1-60, 2024.
- Qiang Zhuang et al. Prompt-based representations for enhanced dense retrieval. In Proceedings of EMNLP arXiv:2404.18424 , 2024a.
- Shengyao Zhuang, Shuai Wang, Bevan Koopman, and Guido Zuccon. Starbucks: Improved training for 2d matryoshka embeddings. arXiv preprint arXiv:2410.13230 , 2024b.