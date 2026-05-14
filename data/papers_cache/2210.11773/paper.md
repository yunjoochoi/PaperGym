## SimANS: Simple Ambiguous Negatives Sampling for Dense Text Retrieval

Kun Zhou 1,3 † , Yeyun Gong 4 , Xiao Liu 4 , Wayne Xin Zhao 2,3 ∗ , Yelong Shen 5 , Anlei Dong 5 , Jingwen Lu 5 , Rangan Majumder 5 , Ji-Rong Wen 2,3 , Nan Duan 4 , Weizhu Chen 5

1 School of Information, Renmin University of China,

2 Gaoling School of Artificial Intelligence, Renmin University of China,

3 Beijing Key Laboratory of Big Data Management and Analysis Methods, 4 Microsoft Research, 5 Microsoft

## Abstract

Sampling proper negatives from a large document pool is vital to effectively train a dense retrieval model. However, existing negative sampling strategies suffer from the uninformative or false negative problem. In this work, we empirically show that according to the measured relevance scores, the negatives ranked around the positives are generally more informative and less likely to be false negatives. Intuitively, these negatives are not too hard ( may be false negatives ) or too easy ( uninformative ). They are the ambiguous negatives and need more attention during training. Thus, we propose a simple ambiguous negatives sampling method, SimANS, which incorporates a new sampling probability distribution to sample more ambiguous negatives. Extensive experiments on four public and one industry datasets show the effectiveness of our approach. We made the code and models publicly available in https: //github.com/microsoft/SimXNS .

## 1 Introduction

Dense text retrieval, which uses low-dimensional vectors to represent queries and documents and measure their relevance, has become a popular topic (Karpukhin et al., 2020; Luan et al., 2021) for both researchers and practitioners. It can improve various downstream applications, e.g., web search (Brickley et al., 2019; Qiu et al., 2022) and question answer (Izacard and Grave, 2021). A key challenge for training a dense text retrieval model is how to select appropriate negatives from a large document pool ( i.e., negative sampling), as most existing methods use a contrastive loss (Karpukhin et al., 2020; Xiong et al., 2021) to encourage the model to rank positive documents higher than negatives. However, the commonly-used negative sampling strategies, namely random negative sampling (Luan et al., 2021; Karpukhin et al., 2020)

† This work was done during internship at MSRA.

∗ Corresponding author, email: batmanfly@gmail.com.

(using random documents in the same batch) and topk hard negatives sampling (Xiong et al., 2021; Zhan et al., 2021) (using an auxiliary retriever to obtain the topk documents), have their limitations. Random negative sampling tends to select uninformative negatives that are rather easy to be distinguished from positives and fail to provide useful information (Xiong et al., 2021), while topk hard negatives sampling may include false negatives (Qu et al., 2021), degrading the model performance.

Motivated by these problems, we propose to sample the ambiguous negatives 1 that are neither too easy (uninformative) nor too hard (potential false negatives). Our approach is inspired by an empirical observation from experiments (in §3) using gradients to assess the impact of data instances on deep models (Koh and Liang, 2017; Pruthi et al., 2020): according to the measured relevance scores using the dense retrieval model, negatives that rank lower are mostly uninformative, as their gradient means are close to zero; negatives that rank higher are likely to be false negatives, as their gradient variances are significantly higher than expected. Both types of negatives are detrimental to the convergence of deep matching models (Xiong et al., 2021; Qu et al., 2021). Interestingly, we find that the negatives ranked around positive examples tend to have relatively larger gradient means and smaller variances, indicating that they are informative and have a lower risk of being false negatives, thus probably being high-quality ambiguous negatives.

Based on these insights, we propose a Sim ple A mbiguous N egative S ampling method, namely SimANS , for improving deep text retrieval. Our main idea is to design a sampling probability distribution that can assign higher probabilities to the ambiguous negatives while lower probabilities to the possible false and uninformative negatives, based on the differences of the relevance scores between positives and candidate negatives. We also incorporate two hyper-parameters to better adjust the peak and density of the sampling probability distribution. Our approach is simple and flexible, which can be easily applied to various dense retrieval models and combined with other effective techniques, e.g., knowledge distillation (Qu et al., 2021) and adversarial training (Zhang et al., 2021).

1 We call them ambiguous negatives following the definition of ambiguous examples (Swayamdipta et al., 2020; Meissner et al., 2021), referring to the instances that are neither too hard nor too easy to learn.

To validate the effectiveness of SimANS, we conduct extensive experiments on four public datasets and one industrial dataset collected from Bing search logs. Experimental results show that SimANS can improve the performance of competitive baselines, including state-of-the-art methods.

## 2 Preliminary

Dense Text Retrieval. Given a query q , the dense text retrieval task aims to retrieve the most relevant topk documents { d i } k i =1 from a large candidate pool D . To achieve it, the dual-encoder architecture is widely used due to its efficiency (Reimers and Gurevych, 2019; Karpukhin et al., 2020). It consists of a query encoder E q and a document encoder E d to map the query q and document d into k -dimensional dense vectors h q and h d , respectively. Then, the semantic relevance score of q and d can be computed using dot product as

<!-- formula-not-decoded -->

Recent works mostly adopt pre-trained language models (PLMs) (Devlin et al., 2019) as the two encoders, and utilize the representations of the [CLS] token as dense vectors.

Training with Negative Sampling. The training objective of dense text retrieval task is to pull the representations of the query q and relevant documents D + together (as positives), while pushing apart irrelevant ones D -= D \ D + (as negatives). However, the irrelevant documents are from a large document pool, which would lead to millions of negatives. To reduce the unreachable training cost, negative sampling has been widely used. Previous works either randomly sample negatives (Karpukhin et al., 2020), or select the topk hard negatives ranked by BM25 or the dense retrieval model itself (Xiong et al., 2021; Qu et al., 2021), denoted as ˜ D -. Then, the optimization ob- jective can be formulated as:

<!-- formula-not-decoded -->

where L ( · ) is the loss function.

## 3 Motivation Study

We first analyze the uninformative and false negative problems from the perspective of gradients. Then, we perform an empirical study to test how gradients of negatives change w.r.t. ranks according to measured relevance scores using a dense retrieval model, and find that the gradients of negatives ranked near positives have relatively larger means and smaller variances.

## 3.1 Analysis for Gradients of Negatives

Existing dense retrieval methods (Karpukhin et al., 2020; Xiong et al., 2021) commonly incorporate the binary cross entropy (BCE) loss to compute gradients 2 , where the relevance scores of a positive and sampled negatives are usually normalized by the softmax function. In this way, the gradients of model parameters θ are computed by

<!-- formula-not-decoded -->

where s n ( q, d ) is the normalized value of s ( q, d ) and is within [0 , 1] . Based on it, we review the gradients of uninformative and false negatives. Uninformative negatives can be easily distinguished by dense retrieval models, and are more likely to be selected by random sampling (Xiong et al., 2021). As their normalized relevance scores are usually rather small, i.e., s n ( q, d ) -→ 0 , their gradient means will be bounded into near-zero values, i.e., glyph[triangleinv] θ l ( q, d ) -→ 0 . Such near-zero gradients are also uninformative and contribute little to model convergence. False negatives are usually semantically similar to positives, and are more likely to be selected by topk hard negatives sampling (Qu et al., 2021). Therefore, for the gradients of false negatives and positives, the right terms glyph[triangleinv] θ s n ( q, d ) may be similar, while the left terms are greater than zero and less than 0, respectively. As a result, the variance of gradients will be larger, which may cause the optimization of parameters to be unstable. Furthermore, existing works (Katharopoulos and Fleuret, 2018; Johnson and Guestrin, 2018) have theoretically proved that larger gradient variance is detrimental to model convergence.

2 In this work, we perform the analysis using BCE loss, and such analysis can also be extended to other loss functions.

Figure 1: The mean and variance of gradients change curves w.r.t. the ranks of negatives on MS-MARCO Passage Ranking dataset using AR2 (Zhang et al., 2021).

<!-- image -->

## 3.2 Empirical Study on Gradients of Negatives w.r.t. Relevance Scores

Although we have analyzed that the harmful influence of uninformative and false negatives derives from the smaller means and larger variances of gradients respectively, it is time-consuming to compute gradients of all candidate negatives to identify and remove them. Here, we empirically study if the query-document relevance scores can be leveraged to avoid sampling these harmful negatives.

Experimental Setup. We use AR2 (Zhang et al., 2021) as the retrieval model and investigate its gradients on the development set of MS-MARCO Passage Ranking dataset (Nguyen et al., 2016). Concretely, for each query, we rank all negatives according to their relevance scores, and compute the means and variances of gradients of all negatives in the same rank 3 . To better show the tendency w.r.t. ranks of relevance scores, we normalize the means and variances of gradients by dividing the maximum values, and only report the results of top 200 ranked negatives.

Results and Findings. As shown in Figure 1, the mean and variance of gradients will gradually decrease with the increase of the rank. Despite that, the gradient means of the top 200 negatives are still in the same order of magnitude ( 1 . 0 -→ 0 . 25 ), while the gradient variances of the top 10 ranked negatives are significantly larger than others. The reason is that the higher-ranking negatives have larger probabilities to be false negatives. Besides, a surprising finding is that the mean rank of positives is approximate the boundary point of the high gradient variance part and the negatives near it can produce relatively larger gradient means and lower gradient variances. It means that they are highquality ambiguous negatives that can balance the informativeness and the risk of being false negatives. Therefore, it is promising to rely on the relevance scores of positives and candidate negatives to devise more effective negative sampling methods for training dense retrieval models.

3 As AR2 adopts ERNIE-2.0 (Sun et al., 2020) as the backbone that has millions of parameters, we only compute gradients on the parameters of its last layer for efficiency.

## 4 Approach

Based on the findings in §3, we conjecture that the ambiguous negatives ranked near positives according to relevance scores are high-quality negatives, as they are neither too easy (uninformative) nor too hard (may be false negatives). Therefore, we propose a simple ambiguous negative sampling method, namely SimANS.

## 4.1 Ambiguous Negative Sampling

To focus on sampling ambiguous negatives, we design a new sampling probability distribution that can estimate the influence of each negative using the dense retrieval models. As follows, we first devise a general sampling distribution and then propose its simple and efficient implementation.

General Sampling Distribution. We draw the following conclusions from our results about how to choose a good sampling probability distribution for negatives: (1) Negatives that are clearly irrelevant and have low relevance scores should be sampled less frequently; (2) Negatives that are highly relevant and have high relevance scores should also be sampled less frequently, because they are more likely to be positives in disguise; (3) Negatives that are uncertain and have relevance scores similar to positives should be sampled more frequently, because they provide useful information and have a lower chance of being false negatives. We propose a general formula for negative sampling probability that reflects these principles:

<!-- formula-not-decoded -->

where f ( · ) is a function to determine the tendency of the probability distribution, b is a hyperparameter to control the peak of the distribution, ¯ s ( q, d + ) is the mean relevance score of all positives with the query. f ( · ) should be a monotone decreasing function ( e.g., e -x ). In this way, the negatives with the relevance scores close to positives can be assigned with larger probabilities, while others with smaller or larger scores will be punished with smaller probabilities. Such a distribution can satisfy the required three characteristics.

Simple Negative Sampling Distribution. Werely on several empirical priors to determine a simple and efficient implementation of the above sampling probability distribution. Generally, the relevance scores of positives and negatives are bounded by the modulus of dense vectors, hence they are mostly in a same order of magnitude. To ensure that the probabilities of ambiguous negatives should be significantly larger than other ones, we choose the exponential function to implement f ( · ) . As a large proportion of negatives from D\D + are uninformative ones, their smaller relevance scores would lead to near-zero probabilities using the exponential function. Therefore, we can reduce the computation cost by narrowing the negative candidates into the topk ranked negatives ˜ D -. In addition, to further reduce the cost, we also replace the mean relevance score of all positives ¯ s ( q, d + ) by the score of a randomly sampled positive s ( q, ˜ d + ) . Finally, we can reformulate the sampling probability distribution in equation (3) as:

<!-- formula-not-decoded -->

where a is a hyper-parameter to control the density of the distribution, ˜ d + ∈ D + is a randomly sampled positive, ˜ D -is the topk ranked negatives. In this way, the complexity of computing the sampling probability distribution will be reduced into O ( k ) , where k glyph[lessmuch] |D| and we set it to 100.

## 4.2 Overview and Discussion

Overview. Given a mini-batch, SimANS contains three major steps to obtain the ambiguous negatives. The first step is the same as previous topk hard negatives sampling methods (Xiong et al., 2021; Qu et al., 2021) that select the topk ranked negatives ˜ D -from the candidate pool D \ D + using an ANN search tool ( e.g., FAISS (Johnson et al., 2019)). Second, we compute the sampling probabilities for all the topk negatives using equation (4). To reduce the time cost, we can pre-compute them in the first step. Finally, we sample the ambiguous negatives w.r.t. their sampling probabilities. We present the overall algorithm in Algorithm 1.

## Algorithm 1: The algorithm of SimANS.

Input: Queries and their positive documents { ( q, D + ) } , document pool D , pre-learned dense retrieval model M

- 1 Build the ANN index on D using M .
- 2 Retrieve the topk ranked negatives ˜ D -for each query with their relevance scores { s ( q, d i ) } from D .
- 3 Compute the relevance scores of each query and its positive documents { s ( q, D + ) } .
- 4 Generate the sampling probabilities of retrieved topk negatives { p i } for each query using Eq. 3.
- 5 Construct new training data { ( q, D + , ˜ D -) } .
- 6 while M has not converged do
- 7 Sample a batch from { ( q, D + , ˜ D -) } .
- 8 Sample ambiguous negatives for each instance from the batch according to { p i } .
- 9 using the batch and
- Optimize parameters of M sampled negatives.

10 end

Note that our proposed SimANS is a negative sampling method and applicable to a variety of dense retrieval methods.

Relationship with Other Methods. SimANS aims to sample the ambiguous negatives that rank close to the positives according to relevance scores for improving the training of dense retrieval models. It is a general framework that several previous negative sampling methods can be included:

- Choosing negative examples randomly means picking them from a big collection of documents with equal chances for each one. We can also use our method to do this by setting b = s ( q, d i ) -s ( q, ˜ d + ) and making ˜ D -include all the documents in the collection. But this is not a good idea, because most of the documents in the collection are not relevant to the query and do not help us learn from the feedback. They are easy to sample but not useful for training.
- Topk hard negatives sampling utilizes an auxiliary retriever ( e.g., BM25 (Karpukhin et al., 2020) or DPR (Xiong et al., 2021)) to rank all negative candidates and pick the topk ones as negatives. By setting b = -s ( q, ˜ d + ) and a = -inf , our method can also produce extremely large probabilities to the topk negatives. Whereas, the topk ones have a higher risk to be false negatives, which are harmful to convergence.

## 5 Experiments

## 5.1 Experimental Setting

Weextensively evaluate SimANS by conducting experiments on three public passage retrieval datasets:

Table 1: Statistics of the five text retrieval datasets.

| Datasets   |   Training |   Dev | Test   |   Documents |
|------------|------------|-------|--------|-------------|
| NQ         |     58,880 | 8,757 | 3,610  |  21,015,324 |
| TQ         |     60,413 | 8,837 | 11,313 |  21,015,324 |
| MS Pas     |    502,939 | 6,980 | -      |   8,841,823 |
| MS Doc     |    367,013 | 5,193 | -      |   3,213,835 |
| Bing       |  1,861,102 | 8,013 | -      |   5,335,927 |

Natural Question (NQ) (Kwiatkowski et al., 2019), Trivia QA (TQ) (Joshi et al., 2017) and MSMARCOPassage Ranking (MS Pas) (Nguyen et al., 2016), a public document retrieval dataset: MSMARCO Document Ranking (MS Doc) (Nguyen et al., 2016), and an industry dataset that is collected from Bing search logs. Their statistics are shown in Table 1. The details of datasets, baselines and implementations are presented in Appendix.

## 5.2 Results Analysis

Performance on Public Retrieval Datasets. Table 2 and Table 3 show the experimental results on three public passage retrieval datasets. First, we can see that AR2 outperforms most baseline methods on all datasets. AR2 incorporates an adversarial training framework to iteratively improve the retriever and ranker. Second, SimANS can further improve the performance of AR2, and outperform all baselines in terms of all the metrics across all datasets. SimANS only incorporates a new negative sampling strategy based on AR2, which aims to sample the ambiguous negatives that are neither too hard (potential false negatives) or too easy (uninformative). According to the findings in §3, such a way can alleviate the uninformative and false negative problems that are frequently encountered in commonly-used random and topk negatives sampling methods, and is able to sample high-quality negatives that contribute more to the model convergence. Besides, the improvements of SimANS on AR2 are larger in MS Pas and Doc datasets than others. The reason is that the two datasets are collected from real-world search logs that suffer severely from the false negative problem, whereas SimANS is capable of alleviating this problem and provides better negatives for training.

Performance on Bing Industry Dataset. For the Bing industry dataset, we adopt a dual-encoder mBERT (Devlin et al., 2019) as the baseline model to deal with multilingual queries and documents, and implement different negative sampling strate- gies on it. We simply evaluate the last checkpoint after training and report the results on the development set. As shown in Table 4, after applying the topk hard negatives sampling, the performance of the baseline model is improved by a large margin. It indicates that hard negatives are more effective than randomly sampled ones. Furthermore, we can see that SimANS outperforms all other negative sampling methods, especially in Hit@5 (2% absolute improvement). It demonstrates the effectiveness of SimANS in industrial scenarios. As a comparison, SimANS is able to alleviate the uninformative and false negatives problems that the random and topk negatives sampling strategies may suffer, respectively.

## 5.3 Further Analysis

Applying SimANS to Other Models. Since SimANS is a general negative sampling strategy, it can be applied to a variety of dense retrieval methods. Thus, in this part, we implement SimANS on two representative methods, ANCE (Xiong et al., 2021) and RocketQA (Qu et al., 2021), as they adopt effective techniques as asynchronous index refresh and knowledge distillation, respectively. We only replace the negative sampling strategies in these methods with SimANS and conduct experiments on TQ and NQ datasets. As shown in Table 5, our approach can consistently improve the performance of the two methods. It shows that SimANS is general to various dense retrieval methods with different techniques and can provide more highquality negatives to improve their performance.

Variation Study. Our proposed SimANS incorporates a new negative sampling probability distribution that is based on the differences between the query-document relevance scores of positives and negative candidates. To verify the effectiveness of this distribution, we design two variations of SimANS: (1) Doc-Sim that leverages the document-document relevance scores between positives and negative candidates to replace the querydocument relevance scores; (2) Nearest-K that directly picks the topk nearest negatives according to the differences of query-document relevance scores instead of sampling. We implement these variations on AR2 and conduct experiments on the development set of MS Pas dataset. As shown in Table 6, SimANS outperforms all these variations. It indicates the effectiveness of our devised ambigu- ous negative sampling probability distribution. For Doc-Sim, it is likely to select the false negatives that have similar semantics to positives, hurting the model performance. For Nearest-K, as it always selects fixed negatives, it may cause overfitting.

| Method                                  | R@5   | NQ R@20   | R@100   | R@5   | TQ R@20   | R@100   | MS Pas   | MS Pas   | R@1k   |
|-----------------------------------------|-------|-----------|---------|-------|-----------|---------|----------|----------|--------|
| Method                                  | R@5   | NQ R@20   | R@100   | R@5   | TQ R@20   | R@100   | MRR@10   | R@50     | R@1k   |
| BM25 (Yang et al., 2017)                | -     | 59.1      | 73.7    | -     | 66.9      | 76.7    | 18.7     | 59.2     | 85.7   |
| GAR (Mao et al., 2021)                  | 60.9  | 74.4      | 85.3    | 73.1  | 80.4      | 85.7    | -        | -        | -      |
| doc2query (Nogueira et al., 2019b)      | -     | -         | -       | -     | -         | -       | 21.5     | 64.4     | 89.1   |
| DeepCT (Dai and Callan, 2019)           | -     | -         | -       | -     | -         | -       | 24.3     | 69.0     | 91.0   |
| docTTTTTquery (Nogueira et al., 2019a)  | -     | -         | -       | -     | -         | -       | 27.7     | 75.6     | 94.7   |
| DPR (Karpukhin et al., 2020)            | -     | 78.4      | 85.3    | -     | 79.3      | 84.9    | -        | -        | -      |
| ANCE (Xiong et al., 2021)               | 71.8  | 81.9      | 87.5    | -     | 80.3      | 85.3    | 33.0     | 81.1     | 95.9   |
| COIL (Gao et al., 2021a)                | -     | -         | -       | -     | -         | -       | 35.5     | -        | 96.3   |
| ME-BERT (Luan et al., 2021)             | -     | -         | -       | -     | -         | -       | 33.8     | -        | -      |
| Joint top- k (Sachan et al., 2021)      | 72.1  | 81.8      | 87.8    | 74.1  | 81.3      | 86.3    | -        | -        | -      |
| Individual top- k (Sachan et al., 2021) | 75.0  | 84.0      | 89.2    | 76.8  | 83.1      | 87.0    | -        | -        | -      |
| RocketQA (Qu et al., 2021)              | 74.0  | 82.7      | 88.5    | -     | -         | -       | 37.0     | 85.5     | 97.9   |
| RDR (Yang and Seo, 2020)                | -     | 82.8      | 88.2    | -     | 82.5      | 87.3    | -        | -        | -      |
| RocketQAv2 (Ren et al., 2021b)          | 75.1  | 83.7      | 89.0    |       |           |         | 38.8     | 86.2     | 98.1   |
| PAIR (Ren et al., 2021a)                | 74.9  | 83.5      | 89.1    | -     | -         | -       | 37.9     | 86.4     | 98.2   |
| DPR-PAQ (O˘ guz et al., 2022)           | 74.2  | 84.0      | 89.2    | -     | -         | -       | 31.1     | -        | -      |
| Condenser (Gao and Callan, 2021)        | -     | 83.2      | 88.4    | -     | 81.9      | 86.2    | 36.6     | -        | 97.4   |
| coCondenser (Gao and Callan, 2022)      | 75.8  | 84.3      | 89.0    | 76.8  | 83.2      | 87.3    | 38.2     | -        | 98.4   |
| ERNIE-Search (Lu et al., 2022)          | 77.0  | 85.3      | 89.7    | -     | -         | -       | 40.1     | 87.7     | 98.2   |
| AR2 (Zhang et al., 2021)                | 77.9  | 86.0      | 90.1    | 78.2  | 84.4      | 87.9    | 39.5     | 87.8     | 98.6   |
| AR2+SimANS                              | 78.6  | 86.2      | 90.3    | 78.6  | 84.6      | 88.1    | 40.9     | 88.7     | 98.7   |

Table 2: Performance on the test sets of NQ and TQ, and the development set of MS Pas. The results of baselines are from original papers. The best and second-best methods are marked in bold and underlined, respectively.

Table 3: Performance on MS Doc development set.

| Method                       |   MRR@10 |   R@100 |
|------------------------------|----------|---------|
| BM25                         |    0.279 |   0.807 |
| DPR (Karpukhin et al., 2020) |    0.320 |   0.864 |
| ANCE (Xiong et al., 2021)    |    0.377 |   0.894 |
| STAR (Zhan et al., 2021)     |    0.390 |   0.913 |
| ADORE (Zhan et al., 2021)    |    0.405 |   0.919 |
| AR2 (Zhang et al., 2021)     |    0.418 |   0.914 |
| AR2+SimANS                   |    0.431 |   0.923 |

Table 4: Experimental results on Bing Industry dataset.

| Method              |   R@5 |   R@20 |   R@100 |
|---------------------|-------|--------|---------|
| Baseline+Random Neg |  39.5 |   59.0 |    76.2 |
| Baseline+top- k Neg |  57.1 |   73.5 |    85.1 |
| Baseline+SimANS     |  59.1 |   74.9 |    85.6 |

Parameter Tuning. Our SimANS has two important hyper-parameters to tune, a and b , which control the density and peak of the sampling probability distribution, respectively. Here, we investigate the performance change of SimANS on AR2 w.r.t. different a and b on NQ dataset. As shown in Figure 2, our approach achieves the best performance when a = 0 . 5 and b = 0 . It indicates that when the maximum point of the distribution has the same relevance score as the positive, the nega- tive sampling probability distribution can produce more high-quality negatives. Moreover, we notice that the model performance is not very sensitive to the two hyper-parameters if they are properly set within a certain range.

| Method          | TQ   | TQ   | NQ   | NQ   |
|-----------------|------|------|------|------|
|                 | R@5  | R@20 | R@5  | R@20 |
| ANCE            | 72.4 | 80.3 | 71.8 | 81.9 |
| ANCE+SimANS     | 74.8 | 82.1 | 74.3 | 83.0 |
| RocketQA        | 76.1 | 83.0 | 74.0 | 82.7 |
| RocketQA+SimANS | 77.1 | 83.6 | 76.7 | 84.8 |

Table 5: The retrieval performance of applying our method on other baselines on TQ and NQ datasets

Table 6: The variation study of our method in AR2 on MS Pas development set.

| Method        |   MRR@10 |   R@1 |   R@50 |   R@1k |
|---------------|----------|-------|--------|--------|
| AR2           |     39.5 |  26.4 |   87.8 |   98.6 |
| AR2+Doc-Sim   |     40.1 |  27.3 |   88.0 |   98.6 |
| AR2+Nearest-K |     40.5 |  27.6 |   88.5 |   98.7 |
| AR2+SimANS    |     40.9 |  28.2 |   88.7 |   98.7 |

(a) a : density hyper-parameter (b) b : peak hyper-parameter

<!-- image -->

Figure 2: Performance comparison w.r.t. hyperparameters a and b on NQ dataset.

Table 7: The retrieval performance and training latency w.r.t. different sampled negative ratios on NQ dataset.

| Ratio   | AR2   | AR2     | AR2+SimANS   | AR2+SimANS   |
|---------|-------|---------|--------------|--------------|
| Ratio   | R@5   | Latency | R@5          | Latency      |
| 1 : 1   | 76.4  | 210ms   | 77.5         | 210ms        |
| 1 : 5   | 76.9  | 330ms   | 78.1         | 340ms        |
| 1 : 11  | 77.1  | 510ms   | 78.3         | 540ms        |
| 1 : 15  | 77.9  | 630ms   | 78.7         | 650ms        |

Figure 3: Hit@1 of AR2+SimANS on training and test sets of NQ w.r.t. training steps.

<!-- image -->

Impact of the Sampled Negative Ratio. We investigate the impact of the sampled negative ratio 1 : k on retrieval performance and training latency per batch of SimANS on AR2. As shown in Table 7, with the increase of the sampled negative number, the performance improves consistently while the training latency increases. Besides, SimANS just slightly increases the training latency of AR2. It is because we can pre-compute the sampling probabilities before training, which avoids time-consuming computation during training.

Performance w.r.t. Training Steps. Our approach requires continually training the model parameters that have been pre-trained by the original dense retrieval method. Here, we investigate the performance changes of the dense retrieval method before and after using SimANS w.r.t. the training steps. We conduct experiments on AR2 and show the Hit@1 metric on NQ dataset in Figure 3. First, we can see that with the increase of the training steps, the performance of AR2 on training and test sets improves simultaneously. After applying our SimANS, we can see that the performance further improves, especially in the training set ( 0 . 777 -→ 0 . 791 ). It indicates that our ap- proach is capable of improving the fitting of the training set, and such an improvement can also generalize to the test set.

## 6 Conclusion

We investigated how the gradient statistics of negative documents affect their relevance ranking for dense text retrieval. We discovered that negative documents with high gradient means and low gradient variances are more likely to be ambiguous negatives, which are informative and less prone to false negatives. Based on this insight, we proposed SimANS, a novel negative sampling method that balances the difficulty of negative examples by adjusting their sampling probabilities. SimANS improved the performance of various dense retrieval models on four public and one industrial datasets. We plan to apply our method to other information retrieval tasks, such as personal recommendation, and to develop better pre-training schemes for dense text retrieval in the future.

## Acknowledgement

Kun Zhou, Wayne Xin Zhao and Ji-Rong Wen are supported by Beijing Natural Science Foundation under Grant No. 4222027, and National Natural Science Foundation of China under Grant No. 61872369, Beijing Outstanding Young Scientist Program under Grant No. BJJWZYJH012019100020098, and the Outstanding Innovative Talents Cultivation Funded Programs 2021. Xin Zhao is the corresponding author.

## Ethical Consideration

In this section, we discuss the ethical considerations of this work from the following two aspects. First, for intellectual property protection, the code, data and dense retrieval models adopted from previous works are granted for research-purpose usage. Second, since PLMs have been shown to capture certain biases from the pre-trained corpus (Bender et al., 2021), there is a potential problem about biases that are from the use of PLMs in our approach. There are increasing efforts to address this problem in the community (Ross et al., 2021).

## References

Emily M Bender, Timnit Gebru, Angelina McMillanMajor, and Shmargaret Shmitchell. 2021. On the dangers of stochastic parrots: Can language models be too big? In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency , pages 610-623.

- Dan Brickley, Matthew Burgess, and Natasha Noy. 2019. Google dataset search: Building a search engine for datasets in an open web ecosystem. In WWW .
- Zhuyun Dai and Jamie Callan. 2019. Deeper text understanding for ir with contextual neural language modeling. In SIGIR .
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL .
- Luyu Gao and Jamie Callan. 2021. Is your language model ready for dense representation fine-tuning? arXiv preprint arXiv:2104.08253 .
- Luyu Gao and Jamie Callan. 2022. Unsupervised corpus aware language model pre-training for dense passage retrieval. In ACL , pages 2843-2853.
- Luyu Gao, Zhuyun Dai, and Jamie Callan. 2021a. COIL: revisit exact lexical match in information retrieval with contextualized inverted list. In NAACLHLT .
- Luyu Gao, Yunyi Zhang, Jiawei Han, and Jamie Callan. 2021b. Scaling deep contrastive learning batch size under memory limited setup. In Proceedings of the 6th Workshop on Representation Learning for NLP (RepL4NLP-2021) .
- Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. 2021. Efficiently teaching an effective dense retriever with balanced topic aware sampling. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval , pages 113-122.
- Wu Hong, Zhuosheng Zhang, Jinyuan Wang, and Hai Zhao. 2022. Sentence-aware contrastive learning for open-domain passage retrieval. In ACL , pages 10621074.
- Gautier Izacard and Édouard Grave. 2021. Leveraging passage retrieval with generative models for open domain question answering. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume , pages 874-880.
- Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with gpus. IEEE Transactions on Big Data .
- Tyler B Johnson and Carlos Guestrin. 2018. Training deep models faster with robust, approximate importance sampling. Advances in Neural Information Processing Systems , 31.
- Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In ACL .
- Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In EMNLP .
- Angelos Katharopoulos and François Fleuret. 2018. Not all samples are created equal: Deep learning with importance sampling. In International conference on machine learning , pages 2525-2534. PMLR.
- Pang Wei Koh and Percy Liang. 2017. Understanding black-box predictions via influence functions. In International conference on machine learning , pages 1885-1894. PMLR.
- Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur P. Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: a benchmark for question answering research. Trans. Assoc. Comput. Linguistics , 7:452-466.
- Yuxiang Lu, Yiding Liu, Jiaxiang Liu, Yunsheng Shi, Zhengjie Huang, Shikun Feng Yu Sun, Hao Tian, Hua Wu, Shuaiqiang Wang, Dawei Yin, et al. 2022. Ernie-search: Bridging cross-encoder with dualencoder via self on-the-fly distillation for dense passage retrieval. arXiv preprint arXiv:2205.09153 .
- Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. 2021. Sparse, dense, and attentional representations for text retrieval. Transactions of the Association for Computational Linguistics , 9:329-345.
- Kelong Mao, Zhicheng Dou, and Hongjin Qian. 2022. Curriculum contrastive context denoising for fewshot conversational dense retrieval. In SIGIR , pages 176-186.
- Yuning Mao, Pengcheng He, Xiaodong Liu, Yelong Shen, Jianfeng Gao, Jiawei Han, and Weizhu Chen. 2021. Generation-augmented retrieval for opendomain question answering. In ACL .
- Johannes Mario Meissner, Napat Thumwanit, Saku Sugawara, and Akiko Aizawa. 2021. Embracing ambiguity: Shifting the training target of nli models. In ACL , pages 862-869.
- Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer. 2020. Ambigqa: Answering ambiguous open-domain questions. In EMNLP , pages 5783-5797.
- Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. Ms marco: A human generated machine reading comprehension dataset. In CoCo@ NIPS .
- Rodrigo Nogueira, Jimmy Lin, and AI Epistemic. 2019a. From doc2query to doctttttquery. Online preprint .
- Rodrigo Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. 2019b. Document expansion by query prediction. arXiv preprint arXiv:1904.08375 .
- Barlas O˘ guz, Kushal Lakhotia, Anchit Gupta, Patrick Lewis, Vladimir Karpukhin, Aleksandra Piktus, Xilun Chen, Sebastian Riedel, Wen-tau Yih, Sonal Gupta, et al. 2022. Domain-matched pre-training tasks for dense retrieval. In Findings of NAACL .
- Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. 2020. Estimating training data influence by tracing gradient descent. Advances in Neural Information Processing Systems , 33:19920-19930.
- Yifu Qiu, Hongyu Li, Yingqi Qu, Ying Chen, Qiaoqiao She, Jing Liu, Hua Wu, and Haifeng Wang. 2022. Dureader\_retrieval: A large-scale chinese benchmark for passage retrieval from web search engine. arXiv preprint arXiv:2203.10232 .
- Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2021. Rocketqa: An optimized training approach to dense passage retrieval for opendomain question answering. In NAACL-HLT .
- Ori Ram, Gal Shachaf, Omer Levy, Jonathan Berant, and Amir Globerson. 2022. Learning to retrieve passages without supervision. In NAACL .
- Nils Reimers and Iryna Gurevych. 2019. Sentencebert: Sentence embeddings using siamese bertnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) , pages 3982-3992.
- Ruiyang Ren, Shangwen Lv, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Qiaoqiao She, Hua Wu, Haifeng Wang, and Ji-Rong Wen. 2021a. PAIR: leveraging passage-centric similarity relation for improving dense passage retrieval. In Findings of ACL/IJCNLP .
- Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Qiaoqiao She, Hua Wu, Haifeng Wang, and Ji-Rong Wen. 2021b. Rocketqav2: A joint training method for dense passage retrieval and passage re-ranking. In EMNLP , pages 2825-2835.
- Candace Ross, Boris Katz, and Andrei Barbu. 2021. Measuring social biases in grounded vision and language embeddings. In NAACL , pages 998-1008.
- Devendra Singh Sachan, Mostofa Patwary, Mohammad Shoeybi, Neel Kant, Wei Ping, William L. Hamilton, and Bryan Catanzaro. 2021. End-to-end training of neural retrievers for open-domain question answering. In ACL/IJCNLP .
- Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Hao Tian, Hua Wu, and Haifeng Wang. 2020. Ernie 2.0: Acontinual pre-training framework for language understanding. In AAAI .
- Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A Smith, and Yejin Choi. 2020. Dataset cartography: Mapping and diagnosing datasets with training dynamics. In EMNLP , pages 9275-9293.
- Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In ICLR .
- Canwen Xu, Daya Guo, Nan Duan, and Julian McAuley. 2022. Laprador: Unsupervised pretrained dense retriever for zero-shot text retrieval. In Findings of ACL , pages 3557-3569.
- Peilin Yang, Hui Fang, and Jimmy Lin. 2017. Anserini: Enabling the use of lucene for information retrieval research. In SIGIR .
- Sohee Yang and Minjoon Seo. 2020. Is retriever merely an approximator of reader? arXiv preprint arXiv:2010.10999 .
- Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021. Optimizing dense retrieval model training with hard negatives. In SIGIR .
- Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2020. Repbert: Contextualized text embeddings for first-stage retrieval. arXiv preprint arXiv:2006.15498 .
- Hang Zhang, Yeyun Gong, Yelong Shen, Jiancheng Lv, Nan Duan, and Weizhu Chen. 2021. Adversarial retriever-ranker for dense text retrieval. In International Conference on Learning Representations .
- Jiawei Zhou, Xiaoguang Li, Lifeng Shang, Lan Luo, Ke Zhan, Enrui Hu, Xinyu Zhang, Hao Jiang, Zhao Cao, Fan Yu, et al. 2022a. Hyperlink-induced pretraining for passage retrieval in open-domain question answering. In ACL , pages 7135-7146.
- Kun Zhou, Beichen Zhang, Wayne Xin Zhao, and JiRong Wen. 2022b. Debiased contrastive learning of unsupervised sentence representations. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 6120-6130.

Figure 4: An example of the dense embedding distribution of a query with its positive document, too easy, too hard and ambiguous negatives.

<!-- image -->

## A Illustration of Ambiguous Negatives

We illustrate the distribution of the dense embeddings of a query with its positive document, too easy, too hard and ambiguous negatives in Figure 4. Too hard negatives have a higher risk of being false negatives, and we can see that their dense embeddings locate closely to the ones of the query and the positive. If we learn to push them away, the distances between the embeddings of the query and the positive may also be enlarged, which is harmful to the goal of pulling the query and its positives together. Besides, too easy negatives locate rather far from the query, hence it is unnecessary to learn to push them even further. As a comparison, the ambiguous negatives have similar distances as the positive, which compose the circular boundary for the document pool consisting of hard negatives required to learn ( i.e., push away). In this way, our SimANS can be seen as always sampling the borderline hard negatives from the document pool. By learning to push them away, we can narrow the circular boundary of hard negatives, which helps gradually achieve the goal that pulls the query and positives together while pushing apart negatives.

## B More Details on Datasets

We conduct experiments on five datasets, consisting of three passage retrieval datasets: Natural Question (NQ) (Kwiatkowski et al., 2019), Trivia QA(TQ) (Joshi et al., 2017) and MS-MARCO Passage Ranking (MS Pas) (Nguyen et al., 2016), a document retrieval dataset: MS-MARCO Document Ranking (MS Doc) (Nguyen et al., 2016) and a real-world industry dataset Bing. NQ and TQ are open domain question answering datasets collected from Google search logs and authored by trivia enthusiasts, respectively. In the two datasets, each question is paired with an answer span and sev- eral golden passages from Wikipedia articles. Following existing works (Zhang et al., 2021; Sachan et al., 2021), we adopt Recall@k (R@k) as the evaluation metrics, which measures if the topk ranked documents include the answer span. MS Pas and MS Doc consist of real questions collected from Bing search logs, where each question is paired with several web passages and documents, respectively. As their labels of test sets are not available, we follow existing works (Ren et al., 2021b; Zhan et al., 2021) that report results on their development sets and adopt MRR@10, R@50 and R@1k for MS Pas, MRR@10 and R@100 for MS Doc. Bing is collected from Bing search logs, where each example consists of a user historical query and several documents that the user has clicked. These documents are real-world webpages and may contain hyperlinks and different languages. We select Hit@5, Hit@20 and Hit@100 for evaluation.

## C More Details on Baselines

We compare our approach with a variety of methods, including sparse and dense retrieval models.

- BM25 (Yang et al., 2017) is a widely-used sparse retriever based on exact matching.
- GAR (Mao et al., 2021), doc2query (Nogueira et al., 2019a), DeepCT (Dai and Callan, 2019) and docTTTTTquery (Nogueira et al., 2019b) enhance BM25 by incorporating neural models.
- DPR (Karpukhin et al., 2020), ANCE (Xiong et al., 2021) and STAR (Zhan et al., 2021) are dense retrieval methods that adopt topk hard negatives to improve training.
- COIL (Gao et al., 2021b) and MEBERT (Luan et al., 2021) combine sparse and dense representations for text retrieval.
- Joint and Individual topk (Sachan et al., 2021) propose to train the dense retrieval model in an end-to-end manner.
- RocketQA (Qu et al., 2021), RDR (Yang and Seo, 2020), RocketQAv2 (Ren et al., 2021b) and ERNIE-search (Lu et al., 2022) utilize knowledge distillation technique that leverages a teacher model to guide the training of the dense retrieval model.
- PAIR (Ren et al., 2021a), DPR-PAQ (O˘ guz et al., 2022), Condenser (Gao and Callan, 2021) and coCondenser (Gao and Callan, 2022) design special pre-training tasks to improve the backbone model for the dense retrieval task.
- AR2 (Zhang et al., 2021) incorporates an ad-

versarial framework to jointly train the retriever and the ranker. As it has achieved state-of-the-art performance on most datasets, we implement our approach on it to verify its effectiveness.

## D Experimental Details

Implementation Details on Public Datasets. For three passage retrieval tasks, we follow the experimental settings in AR2 (Zhang et al., 2021) that selects ERNIE-2.0-base (Sun et al., 2020) as the backbone model. For MS Doc dataset, we leverage the model parameters of STAR (Zhan et al., 2021) to initialize AR2, and then train AR2 with the same hyper-parameters as STAR until convergence. Next, we continue to train the AR2 model parameters with our proposed SimANS, where we set a and b to {(0.5, 1.0), (0.5, 0) , (0.5, 0) , (0.5, 0)} for NQ, TQ, MS Pas and MS Doc datasets, respectively. The learning rate is set to 1e-5 for NQ and 5e-6 for other datasets. The batch size is 256 for MS-Pas and MS-Doc, 64 for NQ and TQ, and the sampling ratio of positives and negatives is 1:15. All other hyper-parameter settings are the same as AR2. All the experiments in this work are conducted on 8 NVIDIA Tesla A100 GPUs.

Implementation Details on Bing Industry Dataset. For the industry dataset, Bing, we adopt mBERT-base (Devlin et al., 2019) as the backbone of the query and document encoders, to deal with multilingual queries and documents. The parameters of the baseline model are trained with randomly sampled negatives using the infoNCE loss (Karpukhin et al., 2020), namely Baseline+Random Neg , and the sampling ratio of positives and negatives is 1:5. The learning rate is 1e-5, the batch size is 128 and the training step is 100,000. As a comparison, we implement the topk negatives sampling strategy on the baseline model, namely Baseline+topk Neg , where we utilize the baseline model to rank and select the top 5 documents that do not contain the query as hard negatives. In our approach, namely Baseline+SimANS , we continue to train the Baseline+topk Neg model, but apply our SimANS to sample 5 negatives from the top 100 ranked documents. We set a to 1, b to 0, and reuse the other hyper-parameters of the Baseline+topk Neg model.

## E Case Study

In this part, we show four examples of the generated sampling probability distributions by our SimANS. These four examples are randomly selected from the training set of MS Pas dataset. As shown in Figure 5, we can see that SimANS indeed assigns larger probabilities to the negatives that rank near the positive while punishing the higherranking and lower-ranking ones that may be false negatives and uninformative negatives. Furthermore, in Figure 5b where the positive is ranked at the first place, our approach is similar to the topk negatives sampling method that assigns larger probabilities to the higher-ranking hard negatives.

## F Related Work

Recent years have witnessed the remarkable performance of dense retrieval methods in text retrieval tasks (Zhan et al., 2020; Hong et al., 2022; Ram et al., 2022; Zhou et al., 2022b). Different from traditional sparse retrieval methods ( e.g., TF-IDF and BM25), dense retrieval approaches typically map queries and documents into low-dimensional dense vectors, and then utilize vector distance metrics ( e.g., cosine similarity) for retrieval.

To learn an effective dense retrieval model, it is key to sample high-quality negatives paired with the given query and positives for training. Early works (Karpukhin et al., 2020; Min et al., 2020) mostly rely on in-batch random negatives and hard negatives sampled by BM25. After that, a series of works (Qu et al., 2021; Xiong et al., 2021) find that sampling topk ranked examples by the dense retriever as hard negatives is more helpful to improve the retriever itself. Among them, several methods (Xiong et al., 2021; Zhan et al., 2021) adopt a dynamic sampling strategy that actively samples topk hard negatives once after an interval during training. However, these topk negative sampling strategies are easy to select higher-ranking false negatives for training. To alleviate it, previous works have incorporated knowledge distillation (Qu et al., 2021; Ren et al., 2021b; Lu et al., 2022), pre-training (Zhou et al., 2022a; Xu et al., 2022) and other denoising techniques (Mao et al., 2022; Hofstätter et al., 2021). Despite the effectiveness, these methods mostly rely on complicated training strategies or complementary models.

In this work, we propose a simple but effective sampling method that weights the negative candidates with the consideration of their differences of relevance scores with positives. As a result, the ambiguous negatives with similar relevance scores to the positives will receive larger sampling probabilities, while the too hard (potential false negatives) and too easy negatives (uninformative) will be punished with smaller probabilities.

Figure 5: Illustration of four sampling probability distributions of the top 50 ranked negatives generated by our SimANS on the training set of MS Pas.

<!-- image -->