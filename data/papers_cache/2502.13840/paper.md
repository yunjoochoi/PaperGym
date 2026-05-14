## Abstract

Recommender systems leverage extensive user interaction data to model preferences; however, directly modeling these data may introduce biases that disproportionately favor popular items. In this paper, we demonstrate that popularity bias arises from the influence of propensity factors during training. Building on this insight, we propose a fair sampling (FS) method that ensures each user and each item has an equal likelihood of being selected as both positive and negative instances, thereby mitigating the influence of propensity factors. The proposed FS method does not require estimating propensity scores, thus avoiding the risk of failing to fully eliminate popularity bias caused by estimation inaccuracies. Comprehensive experiments demonstrate that the proposed FS method achieves state-of-the-art performance in both point-wise and pairwise recommendation tasks. The code implementation is available at https://github.com/jhliu0807/Fair-Sampling.

## CCS Concepts

· Information systems → Recommender systems .

## Keywords

unbiased ranking, collaborative filtering, implicit feedback

## ACMReference Format:

Jiahao Liu, Dongsheng Li, Hansu Gu, Peng Zhang, Tun Lu, Li Shang, and Ning Gu. 2025. Unbiased Collaborative Filtering with Fair Sampling. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25), July 13-18, 2025, Padua, Italy. ACM,NewYork,NY,USA,5pages.https://doi.org/10.1145/3726302.3730260

∗ Corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. SIGIR '25, Padua, Italy

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-1592-1/2025/07

https://doi.org/10.1145/3726302.3730260

## Unbiased Collaborative Filtering with Fair Sampling

Jiahao Liu Fudan University Shanghai, China

jiahaoliu21@m.fudan.edu.cn Dongsheng Li Microsoft Research Asia Shanghai, China dongshengli@fudan.edu.cn

## [Peng Zhang ∗](https://orcid.org/0000-0002-9109-4625)

## [Tun Lu ∗](https://orcid.org/0000-0002-6633-4826)

Fudan University

Shanghai, China zhangpeng\_@fudan.edu.cn

lutun@fudan.edu.cn Hansu Gu Independent Seattle, United States hansug@acm.org Li Shang Ning Gu Fudan University Shanghai, China lishang@fudan.edu.cn ninggu@fudan.edu.cn

## 1 Introduction

Recommender systems gather vast amounts of data on user behavior, especially implicit feedback from user-item interactions, to model user preferences [14-19, 27]. However, user interactions are shaped not only by intrinsic user preferences but also by external factors, such as item popularity [28]. Directly modeling observed user interactions may lead to predictions that are overly skewed toward popular items [25]. The Matthew effect exacerbates this by making long-tail items less likely to be recommended, thus diminishing the novelty, diversity, and fairness of recommendations [7].

Several methods based on inverse propensity scores (IPS) [23, 24] have been proposed to mitigate popularity bias by weighting samples during model training. However, estimating propensity scores in practical applications can be highly challenging. These methods often rely on overly simplified assumptions-for instance, assuming that the propensity score depends solely on item popularitywhich can lead to high variance [21]. Although variance reduction techniques, such as clipping the scores, can be applied, these adjustments compromise unbiasedness. Furthermore, the difficulty in accurately estimating propensity scores prevents IPS-based methods from effectively eliminating popularity bias.

In this paper, we first analyze the optimization objectives of ideal and classical loss functions, identifying that the bias in classical loss functions stems from the inclusion of propensity factors in the model output. Meanwhile, popular items are more likely to be treated as positive samples during training, which leads to higher propensity factor scores and subsequently higher interaction scores, with popularity being mistakenly interpreted as user preference. Building on this insight, we propose a fair sampling (FS) method that ensures each user and each item has an equal likelihood of being selected as both positive and negative instances during training, thus preventing popular items from being overly selected as positive samples. Theoretical analysis demonstrates that the proposed FS can effectively eliminate the influence of propensity factors without the need to estimate the propensity score, thereby mitigating popularity bias and overcoming the limitations of IPSbased methods.

FS is a sampling-level optimization method that can be applied to both point-wise and pair-wise loss functions. Depending on the type of loss function, we implement FS-Point for point-wise loss and FS-Pair for pair-wise loss. Experimental results show that FS achieves state-of-the-art performance.

## 2 Related Work

IPS-based methods mitigate popularity bias in recommender systems through sample weighting. We categorize related work by the type of loss function: point-wise or pair-wise.

Point-wise. Rel-MF [24] formulates interaction occurrence as a two-step process-exposure and interaction-and introduces an unbiased point-wise loss function. DU [9] refines this method by enhancing propensity score estimation. CJMF [32] introduces a joint learning framework to simultaneously model unbiased useritem relevance and propensity. BISER [10] addresses item bias via self-inverse propensity weighting and employs bilateral unbiased learning to unify two complementary models. More recently, ReCRec [13] improves bias correction by distinguishing unexposed from disliked items, enabling better reasoning over unclicked data.

Pair-wise. UBPR[23]buildson the two-step interaction assumption to introduce an unbiased pair-wise loss function. UPL [21] extends this approach with a variance-reduced learning method. CPR [25] redefines unbiasedness in ranking and leverages cross pairs to improve unbiased learning. UpliftRec [4] applies uplift modeling to dynamically optimize top-N recommendations, revealing latent user preferences while maximizing click-through rates. Additionally, PU [3] models feedback labels as a noisy proxy for exposure outcomes and integrates a theoretically noise-resistant loss function into propensity estimation.

## 3 Methods

We first introduce the notations in Section 3.1 and formalize the ideal and classic loss functions in Sections 3.2 and 3.3, respectively. Then, in Section 3.4, we analyze the bias in the classic loss function by comparing its optimization objective with that of the ideal loss function. Finally, in Section 3.5, we present the proposed fair sampling (FS) method and analyze how it mitigates popularity bias.

## 3.1 Notations

For a user /u1D462 ∈ U and an item /u1D456 ∈ I , we refer to a user-item pair as a positive pair if an interaction occurs between them; otherwise, we call it a negative pair . A interaction occurs through a twostep process [23, 24]: first, a user observes an item; then, based on how relevant the item is to the user's preferences, the user decide whether to interact with it. Next, we define three binary indicator matrices: (1) Observation matrix /u1D442 ∈ { 0 , 1 } | U |×| I | , where /u1D442 /u1D462 ,/u1D456 indicates whether /u1D462 has observed /u1D456 (or, equivalently, whether /u1D456 has been exposed to /u1D462 ); (2) Relevance matrix /u1D445 ∈ { 0 , 1 } | U |×| I | , where /u1D445 /u1D462 ,/u1D456 indicates whether /u1D462 would interact with /u1D456 given /u1D442 /u1D462 ,/u1D456 = 1 (or, equivalently, whether /u1D462 likes /u1D456 ); (3) Interaction matrix /u1D44C ∈ { 0 , 1 } | U |×| I | , where /u1D44C /u1D462 ,/u1D456 indicates whether /u1D462 -/u1D456 is a positive pair. Note that only /u1D44C is observable, while both /u1D442 and /u1D445 are latent variables.

Clearly, /u1D462 -/u1D456 is a positive pair ( /u1D44C /u1D462 ,/u1D456 = 1) if and only if /u1D462 has observed /u1D456 ( /u1D442 /u1D462 ,/u1D456 = 1) and the two have high relevance ( /u1D445 /u1D462 ,/u1D456 = 1). This can be expressed as /u1D44C /u1D462 ,/u1D456 = /u1D442 /u1D462 ,/u1D456 · /u1D445 /u1D462 ,/u1D456 , or in matrix form, /u1D44C = /u1D442 /circledot /u1D445 .

Consequently,

<!-- formula-not-decoded -->

where /u1D443 ( /u1D445 /u1D462 ,/u1D456 = 1 ) denotes the relevance probability between /u1D462 and /u1D456 , and /u1D443 ( /u1D442 /u1D462 ,/u1D456 = 1 | /u1D445 /u1D462 ,/u1D456 = 1 ) represents the exposure probabilitythe probability that /u1D456 is observed by /u1D462 given their relevance. Assuming the exposure probability can be decomposed into user propensity, item propensity, and user-item relevance, it can be expressed as:

<!-- formula-not-decoded -->

Here, /u1D703 /u1D462 and /u1D703 /u1D456 are user-specific and item-specific propensity factors, respectively, which tend to have higher values for active users and popular items. The term /u1D443 ( /u1D445 /u1D462 ,/u1D456 = 1 ) /u1D6FC indicates that a higher relevance score increases the exposure probability, where /u1D6FC is a positive constant used to moderate its influence.

## 3.2 Ideal Point-wise Loss and Pair-wise Loss

Ideally, a recommendation model should reflect users' intrinsic preferences /u1D445 rather than simply imitating observed user behaviors /u1D44C . For this to hold, users must observe all items and decide whether to interact with them based on relevance. In this scenario, all elements of /u1D442 are assigned a value of 1, making /u1D44C identical to /u1D445 .

3.2.1 Ideal Point-wise Loss. One commonly used point-wise loss is the cross-entropy loss. Let /u1D460 /u1D462 ,/u1D456 denote the predicted score of /u1D462 -/u1D456 produced by the model, then /u1D6FF + ( /u1D462 , /u1D456 ) = -log ( /u1D460 /u1D462 ,/u1D456 ) and /u1D6FF - ( /u1D462 , /u1D456 ) = -log ( 1 -/u1D460 /u1D462 ,/u1D456 ) , where /u1D6FF + ( /u1D462 , /u1D456 ) and /u1D6FF - ( /u1D462 , /u1D456 ) denote the loss contributions for /u1D462 -/u1D456 being a positive and a negative pair, respectively. The ideal point-wise loss is defined as follows:

<!-- formula-not-decoded -->

3.2.2 Ideal Pair-wise Loss. One commonly used pair-wise loss is the BPR loss, which is defined as /u1D701 ( /u1D462 , /u1D456, /u1D457 ) = -ln /u1D70E ( /u1D460 /u1D462 ,/u1D456 -/u1D460 /u1D462 , /u1D457 ) , where /u1D70E (·) is the sigmoid function. The ideal pair-wise loss is defined as follows:

<!-- formula-not-decoded -->

3.2.3 Discussion. The ideal loss functions can guide the model to learn users' inherent preferences. However, since /u1D445 is unobservable, the ideal loss functions cannot be computed directly.

## 3.3 Classic Point-wise Loss and Pair-wise Loss

3.3.1 Classic Point-wise Loss. WMF [6] introduced a point-wise loss for modeling implicit feedback data:

<!-- formula-not-decoded -->

3.3.2 Classic Pair-wise Loss. BPR[22]introduceda pair-wise loss for modeling implicit feedback data:

<!-- formula-not-decoded -->

3.3.3 Discussion. Classic loss functions are designed to directly model the interaction matrix /u1D44C . Since /u1D44C depends not only on /u1D445 but also on /u1D442 , classic loss functions provide biased approximations of the ideal loss functions.

## 3.4 Analysis

Let /u1D460 ideal /u1D462 ,/u1D456 and /u1D460 bias /u1D462 ,/u1D456 denote the predicted results obtained by optimizing the model with the ideal loss ( L point ideal or L pair ideal ) and the classic loss ( L point bias or L pair bias ), respectively. /u1D460 ideal /u1D462 ,/u1D456 captures users' intrinsic preferences /u1D445 , while /u1D460 bias /u1D462 ,/u1D456 reflects users' interaction behaviors /u1D44C . Therefore,

<!-- formula-not-decoded -->

Intuitively, classic loss functions introduce biases as they inherently incorporate propensity factors into the model's output. As a result, even if user /u1D462 exhibits the same level of preference for both items /u1D456 and /u1D457 ( /u1D443 ( /u1D445 /u1D462 ,/u1D456 = 1 ) = /u1D443 ( /u1D445 /u1D462 , /u1D457 = 1 ) ), the optimization outcome of classic loss functions tends to favor the item with a higher propensity factor ( /u1D703 /u1D462 /u1D703 /u1D456 /u1D443 ( /u1D445 /u1D462 ,/u1D456 = 1 ) /u1D6FC or /u1D703 /u1D462 /u1D703 /u1D457 /u1D443 ( /u1D445 /u1D462 , /u1D457 = 1 ) /u1D6FC ).

3.4.1 Point-wise Loss. The optimization objective of L point ideal is to increase /u1D460 ideal /u1D462 ,/u1D456 when /u1D445 /u1D462 ,/u1D456 = 1 and decrease /u1D460 ideal /u1D462 ,/u1D456 when /u1D445 /u1D462 ,/u1D456 = 0. We use Δ /u1D465 to denote the change in /u1D465 after one optimization step (e.g., gradient descent). Therefore,

<!-- formula-not-decoded -->

which can also be written as:

<!-- formula-not-decoded -->

While for L point bias , its optimization objective is to increase /u1D460 bias /u1D462 ,/u1D456 when /u1D44C /u1D462 ,/u1D456 = 1 and decrease /u1D460 bias /u1D462 ,/u1D456 when /u1D44C /u1D462 ,/u1D456 = 0. Therefore,

<!-- formula-not-decoded -->

which can also be written as 1 :

<!-- formula-not-decoded -->

3.4.2 Pair-wise Loss. Assume that /u1D462 -/u1D456 is a positive pair, and /u1D462 -/u1D457 is a negative pair. The optimization objective of L pair ideal is to increase the margin between /u1D460 ideal /u1D462 ,/u1D456 and /u1D460 ideal /u1D462 , /u1D457 . Therefore,

<!-- formula-not-decoded -->

1 Note that Δ ( /u1D465 + /u1D466.alt ) = Δ /u1D465 + Δ /u1D466.alt , and Δ /u1D44E/u1D465 = /u1D44E Δ /u1D465 , where /u1D44E is a constant.

While for L pair bias , its optimization objective it to increase the margin between /u1D460 bias /u1D462 ,/u1D456 and /u1D460 bias /u1D462 , /u1D457 . Therefore,

<!-- formula-not-decoded -->

3.4.3 Discussion. By comparing Equation (9) with (11), and Equation (12) with (13), it can be observed that L point bias and L pair bias improperly optimize the propensity factors ( /u1D703 /u1D462 and /u1D703 /u1D456 for L point bias , and /u1D703 /u1D456 and /u1D703 /u1D457 for L pair bias ). Since popular items and active users are more frequently selected to form positive pairs during training, their propensity factors tend to be higher. Ultimately, propensity factors related to exposure probability /u1D443 ( /u1D442 /u1D462 ,/u1D456 = 1 | /u1D445 /u1D462 ,/u1D456 = 1 ) , rather than relevance probability /u1D443 ( /u1D445 /u1D462 ,/u1D456 = 1 ) , lead to higher interaction scores /u1D443 ( /u1D44C /u1D462 ,/u1D456 = 1 ) , which is not the intended outcome.

## 3.5 Fair Sampling

Fair sampling (FS) constructs supplementary sample(s) for each original sample in the classic loss during model training, helping to prevent the improper optimization of propensity factors and thereby mitigating popularity bias. Depending on the type of loss function, FS has two variants: FS-Point , designed for point-wise loss, and FS-Pair , designed for pair-wise loss.

3.5.1 FS-Point. The sample set utilized by the classic point-wise loss function L point bias is:

<!-- formula-not-decoded -->

For each ( /u1D462 , /u1D456 ; /u1D44C /u1D462 ,/u1D456 ) ∈ D point bias , we can find a corresponding ( ˜ /u1D462 , ˜ /u1D456 ) that satisfies the conditions /u1D44C ˜ /u1D462, ˜ /u1D456 = /u1D44C /u1D462 ,/u1D456 , /u1D44C ˜ /u1D462 ,/u1D456 = 1 -/u1D44C /u1D462 ,/u1D456 , and /u1D44C /u1D462 , ˜ /u1D456 = 1 -/u1D44C /u1D462 ,/u1D456 . Then, we can obtain the sample set for FS-Point loss by combining them together:

<!-- formula-not-decoded -->

Finally, FS-Point loss is defined as:

<!-- formula-not-decoded -->

We assume /u1D44C /u1D462 ,/u1D456 = 1 (a similar conclusion can be drawn when /u1D44C /u1D462 ,/u1D456 = 0). Based on Equation (11), when ( /u1D462 , /u1D456 ; /u1D44C /u1D462 ,/u1D456 = 1 ) is input into L point FS for optimization, ln /u1D703 /u1D462 and ln /u1D703 /u1D456 are amplified. However, when the corresponding supplementary samples-( ˜ /u1D462 , ˜ /u1D456 ; /u1D44C ˜ /u1D462 , ˜ /u1D456 = 1 ) , ( ˜ /u1D462, /u1D456 ; /u1D44C ˜ /u1D462 ,/u1D456 = 0 ) , and ( /u1D462 , ˜ /u1D456 ; /u1D44C /u1D462 , ˜ /u1D456 = 0 ) -are input into L point FS for optimization, the optimization effect on the propensity factors is offset:

<!-- formula-not-decoded -->

3.5.2 FS-Pair. The sample set utilized by the classic pair-wise loss function L pair bias is:

<!-- formula-not-decoded -->

Table 1: Overall performance comparison.

| Model   | Model   | Model         | Kindle               | Kindle               | Kindle         | Gowalla              | Gowalla              | Gowalla        | Yelp                        | Yelp                        | Yelp           |
|---------|---------|---------------|----------------------|----------------------|----------------|----------------------|----------------------|----------------|-----------------------------|-----------------------------|----------------|
|         |         |               | Recall ↑             | NDCG ↑               | ARP ↓          | Recall ↑             | NDCG ↑               | ARP ↓          | Recall ↑                    | NDCG ↑                      | ARP ↓          |
|         | IPS     | WMF Rel-MF DU | 0.1095 0.1192 0.1264 | 0.0979 0.1088 0.1141 | 1896 1660 1497 | 0.1052 0.1116 0.1168 | 0.0924 0.0975 0.1018 | 3995 3800 3356 | 0.0398 0.0471 0.0504        | 0.0297 0.0359 0.0382        | 4231 3651 3286 |
|         | Causal  | ExpoMF CauseE | 0.1353 0.1336        | 0.1226 0.1233        | 1354 1447      | 0.1186 0.1179        | 0.1013 0.1005        | 3173 3159      | 0.0514 0.0528               | 0.0401 0.0408               | 3105 3025      |
|         | Our     | FS-Point      | 0.1413               | 0.1263               | 1175           | 0.1217               | 0.1067               | 2261           | 0.0554                      | 0.0437                      | 2420           |
|         | IPS     | BPR UBPR      | 0.1431 0.1414        | 0.1169 0.1196        | 1572 1156      | 0.1179 0.1125        | 0.0943 0.0936        | 3703 2447      | 0.0527                      | 0.0388                      | 3939           |
|         |         | CPR DICE      | 0.1516 0.1454        | 0.1275 0.1195        | 1406 1303 1254 | 0.1240 0.1160        | 0.1023               | 2509           | 0.0600 0.0634 0.0599 0.0613 | 0.0452 0.0478 0.0455 0.0469 | 2489 2901 2459 |
|         | Causal  |               |                      |                      |                |                      | 0.0966               | 2302           |                             |                             |                |
|         |         | PD            | 0.1418               | 0.1214               |                | 0.1162               | 0.0953               | 2388           |                             |                             | 2404           |
|         | Our     | FS-Pair       | 0.1652               | 0.1384               | 865            | 0.1294               | 0.1052               | 1140           | 0.0741                      | 0.0552                      | 1534           |

For each ( /u1D462 , /u1D456, /u1D457 ; /u1D44C /u1D462 ,/u1D456 , /u1D44C /u1D462 , /u1D457 ) ∈ D pair bias , we can find a corresponding ( ˜ /u1D462 , /u1D457 , /u1D456 ) that satisfies the conditions /u1D44C ˜ /u1D462 , /u1D457 = /u1D44C /u1D462 ,/u1D456 and /u1D44C ˜ /u1D462 ,/u1D456 = /u1D44C /u1D462 , /u1D457 . Then, we can obtain the sample set for FS-Pair loss by combining them together:

<!-- formula-not-decoded -->

Finally, FS-Pair loss is defined as:

<!-- formula-not-decoded -->

Since a sample contributes to the loss only when /u1D44C /u1D462 ,/u1D456 = 1 and /u1D44C /u1D462 , /u1D457 = 0, we focus on this case. Based on Equation (13), when ( /u1D462 , /u1D456, /u1D457 ; /u1D44C /u1D462 ,/u1D456 = 1 , /u1D44C /u1D462 , /u1D457 = 0 ) is input into L pair FS for optimization, ln /u1D703 /u1D456 /u1D703 /u1D457 is amplified. However, when the corresponding supplementary sample ( ˜ /u1D462 , /u1D457 , /u1D456 ; /u1D44C ˜ /u1D462 , /u1D457 = 1 , /u1D44C ˜ /u1D462 ,/u1D456 = 0 ) is input into L pair FS for optimization, ln /u1D703 /u1D457 /u1D703 /u1D456 is amplified, which offsets the optimization effect on the propensity factors:

<!-- formula-not-decoded -->

3.5.3 Discussion. The idea of FS is simple-whenever a user or item is chosen to form a positive/negative pair, the user or item is simultaneously selected to form a corresponding negative/positive pair. Equations (17) and (21) indicate that FS-Point loss and FS-Pair loss no longer optimize propensity factors. Consequently, only the preference-related factor /u1D443 ( /u1D445 /u1D462 ,/u1D456 = 1 ) is optimized, effectively mitigating the popularity bias. Note that FS does not require estimating the propensity score, which avoids the incomplete removal of popularity bias in IPS-based methods, which stems from their inability to estimate the propensity score accurately.

## 4 Experiments

## 4.1 Settings

4.1.1 Datasets. We experiment with three widely used datasets: AmazonReview (Kindle) [20], Gowalla [5], and Yelp. These datasets contain log data capturing observed user behavior. Under the twostep interaction assumption, these datasets exhibit popularity bias. We retain only interactions with ratings ≥ 4 and ensure that both users and items have at least 20 interactions.

Following prior offline evaluation protocols [2, 11, 26], we construct unbiased validation and test sets by sampling from the full dataset with equal selection probability for each item. The remaining data serves as the training set. The dataset is split into training, validation, and test sets in a 7:1:2 ratio.

4.1.2 Evaluation. We evaluate performance using three metrics: Recall@K, NDCG@K, and ARP@K, with /u1D43E set to 20 by default. Average Recommendation Popularity at /u1D43E (ARP@K) serves as a complementary metric for measuring recommendation bias [1, 29]. It calculates the average popularity of the top/u1D43E recommended items per user, where lower ARP@K values indicate reduced bias.

4.1.3 Baselines. In addition to the IPS-based methods, we also use four causal inference methods-ExpoMF [12], CauseE [2], PD [30], and DICE [31]-as baselines for comparison. To ensure fairness, all methods use MF [8] as the backbone model and share the same hyperparameter search space. All point-wise methods use crossentropy loss, while all pair-wise methods use BPR loss.

## 4.2 Results

AsshowninTable1,FS-Point achieves the highest Recall and NDCG, and the lowest APR among point-wise learning methods. Similarly, FS-Pair achieves the best performance in these metrics among pairwise methods. This suggests that, compared to other baselines, FS more effectively mitigates the influence of popularity bias, increasing the likelihood of recommending unpopular items without sacrificing recommendation accuracy. Notably, despite its simplicity, theoretical analysis confirms that FS completely eliminates popularity bias, enabling optimal performance under the current evaluation strategy, where the test set is designed to be unaffected by item popularity.

## 5 Conclusions

We propose a fair sampling (FS) method, which mitigates popularity bias in collaborative filtering by ensuring that each item appears equally as both a positive and negative sample during training. Both theoretical analysis and experimental results demonstrate the effectiveness of the proposed FS method. A potential limitation of FS is that it completely overlooks the influence of popularity, even though popularity is often correlated with higher quality. Strategically adjusting the sampling ratio can help balance the diversity and quality of recommendations.

## Acknowledgments

This work is supported by National Natural Science Foundation of China (NSFC) under the Grant No. 62372113. Peng Zhang is a faculty of School of Computer Science, Fudan University. Tun Lu is a faculty of School of Computer Science, Shanghai Key Laboratory of Data Science, Fudan Institute on Aging, MOE Laboratory for National Development and Intelligent Governance, and Shanghai Institute of Intelligent Electronics &amp; Systems, Fudan University.

## References

- [1] Himan Abdollahpouri, Robin Burke, and Bamshad Mobasher. 2019. Managing popularity bias in recommender systems with personalized re-ranking. arXiv preprint arXiv:1901.07555 (2019).
- [2] Stephen Bonner and Flavian Vasile. 2018. Causal embeddings for recommendation. In Proceedings of the 12th ACM conference on recommender systems . 104112.
- [3] Tianwei Cao, Qianqian Xu, Zhiyong Yang, Zhanyu Ma, and Qingming Huang. 2024. Practically Unbiased Pairwise Loss for Recommendation with Implicit Feedback. IEEE Transactions on Pattern Analysis and Machine Intelligence (2024).
- [4] Jiaju Chen, Wang Wenjie, Chongming Gao, Peng Wu, Jianxiong Wei, and Qingsong Hua. 2024. Treatment Effect Estimation for User Interest Exploration on Recommender Systems. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval . 1861-1871.
- [5] Eunjoon Cho, Seth A Myers, and Jure Leskovec. 2011. Friendship and mobility: user movement in location-based social networks. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining . 10821090.
- [6] Yifan Hu, Yehuda Koren, and Chris Volinsky. 2008. Collaborative filtering for implicit feedback datasets. In 2008 Eighth IEEE international conference on data mining . Ieee, 263-272.
- [7] Anastasiia Klimashevskaia, Dietmar Jannach, Mehdi Elahi, and Christoph Trattner. 2024. A survey on popularity bias in recommender systems. User Modeling and User-Adapted Interaction 34, 5 (2024), 1777-1834.
- [8] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer 42, 8 (2009), 30-37.
- [9] Jae-woong Lee, Seongmin Park, and Jongwuk Lee. 2021. Dual unbiased recommender learning for implicit feedback. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval . 1647-1651.
- [10] Jae-woong Lee, Seongmin Park, Joonseok Lee, and Jongwuk Lee. 2022. Bilateral Self-unbiased Learning from Biased Implicit Feedback. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval . 29-39.
- [11] Dawen Liang, Laurent Charlin, and David M Blei. 2016. Causal inference for recommendation. In Causation: Foundation to Application, Workshop at UAI. AUAI .
- [12] Dawen Liang, Laurent Charlin, James McInerney, and David M Blei. 2016. Modeling user exposure in recommendation. In Proceedings of the 25th international conference on World Wide Web . 951-961.
- [13] Siyi Lin, Sheng Zhou, Jiawei Chen, Yan Feng, Qihao Shi, Chun Chen, Ying Li, and CanWang.2024. ReCRec: Reasoning the causes of implicit feedback for debiased recommendation. ACM Transactions on Information Systems 42, 6 (2024), 1-26.
- [14] Jiahao Liu, Dongsheng Li, Hansu Gu, Tun Lu, Jiongran Wu, Peng Zhang, Li Shang, and Ning Gu. 2023. Recommendation unlearning via matrix correction. arXiv preprint arXiv:2307.15960 (2023).
- [15] Jiahao Liu, Dongsheng Li, Hansu Gu, Tun Lu, Peng Zhang, and Ning Gu. 2022. Parameter-free dynamic graph embedding for link prediction. Advances in Neural Information Processing Systems 35 (2022), 27623-27635.
- [16] Jiahao Liu, Dongsheng Li, Hansu Gu, Tun Lu, Peng Zhang, Li Shang, and Ning Gu. 2023. Personalized graph signal processing for collaborative filtering. In Proceedings of the ACM Web Conference 2023 . 1264-1272.
- [17] Jiahao Liu, Dongsheng Li, Hansu Gu, Tun Lu, Peng Zhang, Li Shang, and Ning Gu. 2023. Triple structural information modelling for accurate, explainable and interactive recommendation. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval . 1086-1095.
- [18] Jiahao Liu, Yiyang Shao, Peng Zhang, Dongsheng Li, Hansu Gu, Chao Chen, Longzhi Du, Tun Lu, and Ning Gu. 2024. Filtering Discomforting Recommendations with Large Language Models. arXiv preprint arXiv:2410.05411 (2024).
- [19] Sijia Liu, Jiahao Liu, Hansu Gu, Dongsheng Li, Tun Lu, Peng Zhang, and Ning Gu. 2023. Autoseqrec: Autoencoder for efficient sequential recommendation. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management . 1493-1502.
- [20] Jianmo Ni, Jiacheng Li, and Julian McAuley. 2019. Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP) . 188-197.
- [21] Yi Ren, Hongyan Tang, Jiangpeng Rong, and Siwen Zhu. 2023. Unbiased Pairwise Learning from Implicit Feedback for Recommender Systems without Biased Variance Control. arXiv preprint arXiv:2304.05066 (2023).
- [22] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars SchmidtThieme. 2012. BPR: Bayesian personalized ranking from implicit feedback. arXiv preprint arXiv:1205.2618 (2012).
- [23] Yuta Saito. 2020. Unbiased pairwise learning from biased implicit feedback. In Proceedings of the 2020 ACM SIGIR on International Conference on Theory of Information Retrieval . 5-12.
- [24] Yuta Saito, Suguru Yaginuma, Yuta Nishino, Hayato Sakata, and Kazuhide Nakata. 2020. Unbiased recommender learning from missing-not-at-random implicit feedback. In Proceedings of the 13th International Conference on Web Search and Data Mining . 501-509.
- [25] Qi Wan, Xiangnan He, Xiang Wang, Jiancan Wu, Wei Guo, and Ruiming Tang. 2022. Cross pairwise ranking for unbiased item recommendation. In Proceedings of the ACM Web Conference 2022 . 2370-2378.
- [26] Ying-Xin Wu, Xiang Wang, An Zhang, Xiangnan He, and Tat-Seng Chua. 2022. Discovering invariant rationales for graph neural networks. arXiv preprint arXiv:2201.12872 (2022).
- [27] Jiafeng Xia, Dongsheng Li, Hansu Gu, Jiahao Liu, Tun Lu, and Ning Gu. 2022. FIRE: Fast incremental recommendation with graph signal processing. In Proceedings of the ACM Web Conference 2022 . 2360-2369.
- [28] Longqi Yang, Yin Cui, Yuan Xuan, Chenyang Wang, Serge Belongie, and Deborah Estrin. 2018. Unbiased offline recommender evaluation for missing-not-atrandom implicit feedback. In Proceedings of the 12th ACM conference on recommender systems . 279-287.
- [29] Hongzhi Yin, Bin Cui, Jing Li, Junjie Yao, and Chen Chen. 2012. Challenging the long tail recommendation. arXiv preprint arXiv:1205.6700 (2012).
- [30] Yang Zhang, Fuli Feng, Xiangnan He, Tianxin Wei, Chonggang Song, Guohui Ling, and Yongdong Zhang. 2021. Causal intervention for leveraging popularity bias in recommendation. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval . 11-20.
- [31] Yu Zheng, Chen Gao, Xiang Li, Xiangnan He, Yong Li, and Depeng Jin. 2021. Disentangling user interest and conformity for recommendation with causal embedding. In Proceedings of the Web Conference 2021 . 2980-2991.
- [32] Ziwei Zhu, Yun He, Yin Zhang, and James Caverlee. 2020. Unbiased implicit recommendation and propensity estimation via combinational joint learning. In Proceedings of the 14th ACM Conference on Recommender Systems . 551-556.