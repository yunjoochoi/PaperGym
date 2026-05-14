## Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation

Junliang Yu

The University of Queensland Brisbane, Australia jl.yu@uq.edu.au

Tong Chen

The University of Queensland Brisbane, Australia tong.chen@uq.edu.au

## ABSTRACT

Contrastive learning (CL) recently has spurred a fruitful line of research in the field of recommendation, since its ability to extract self-supervised signals from the raw data is well-aligned with recommender systems' needs for tackling the data sparsity issue. A typical pipeline of CL-based recommendation models is first augmenting the user-item bipartite graph with structure perturbations, and then maximizing the node representation consistency between different graph augmentations. Although this paradigm turns out to be effective, what underlies the performance gains is still a mystery. In this paper, we first experimentally disclose that, in CL-based recommendation models, CL operates by learning more evenly distributed user/item representations that can implicitly mitigate the popularity bias. Meanwhile, we reveal that the graph augmentations, which were considered necessary, just play a trivial role. Based on this finding, we propose a simple CL method which discards the graph augmentations and instead adds uniform noises to the embedding space for creating contrastive views. A comprehensive experimental study on three benchmark datasets demonstrates that, though it appears strikingly simple, the proposed method can smoothly adjust the uniformity of learned representations and has distinct advantages over its graph augmentation-based counterparts in terms of recommendation accuracy and training efficiency. The code is released at https://github.com/Coder-Yu/QRec.

## CCS CONCEPTS

· Information systems → Recommender systems .

## KEYWORDS

Self-Supervised Learning, Recommendation, Contrastive Learning, Data Augmentation

∗ Corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

SIGIR '22, July 11-15, 2022, Madrid, Spain

© 2022 Association for Computing Machinery.

ACM ISBN 978-1-4503-8732-3/22/07...$15.00

https://doi.org/10.1145/3477495.3531937

Hongzhi Yin ∗

The University of Queensland Brisbane, Australia h.yin1@uq.edu.au

## Lizhen Cui

Shandong University Jinan, China clz@sdu.edu.cn

## ACMReference Format:

Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Lizhen Cui, and Nguyen Quoc Viet Hung. 2022. Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22), July 11-15, 2022, Madrid, Spain. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3477495.3531937

## 1 INTRODUCTION

Recently, a resurgence of contrastive learning (CL) [12, 13, 17] has been witnessed in deep representation learning. Due to the ability to extract the general features from massive unlabeled data and regularize representations in a self-supervised manner, CL has led to major advances in multiple research fields [5, 7, 29, 36]. As data annotation is not required in CL, it is a natural antidote to the data sparsity issue in recommender systems [9, 23]. An increasing number of very recent studies [29, 33, 39, 41, 45, 46] have sought to harness CL for improving recommendation performance and have demonstrated significant gains. A typical way [29] to apply CL to recommendation is first augmenting the user-item bipartite graph with structure perturbations (e.g., stochastic edge/node dropout at a certain ratio), and then maximizing the consistency of representations under different views learned via a graph encoder. In this setting, the CL task acts as the auxiliary task, and is jointly optimized with the recommendation task (Fig. 1).

Despite the encouraging results achieved by CL, however, what underlies the performance gains still remains unclear. Intuitively, we presume that contrasting different graph augmentations can capture the essential information existing in the original user-item interactions, by randomly removing the redundancy and impurity with the edge/node dropout. Unexpectedly, a few latest works [15, 39, 47] have reported that even extremely sparse graph augmentations (with edge dropout rate 0.9) in CL can bring desired performance gains. Such a phenomenon is quite elusive and counterintuitive because a large dropout rate will result in a huge loss of the raw information and a highly skewed graph structure. It naturally raises a meaningful question: Do we really need graph augmentations when integrating CL with recommendation?

To answer this question, we first conduct experiments with and without the graph augmentations respectively for a performance comparison. The results show that the when the graph augmentations are absent, the performance is also comparable to those with

Xin Xia

The University of Queensland Brisbane, Australia x.xia@uq.edu.au

Nguyen Quoc Viet Hung Griffith University Gold Coast, Australia quocviethung1@gmail.com graph augmentations. We then investigate the embedding space learned by non-CL and CL-based recommendation methods. By visualizing the distributions of the representations and associating them with their performances, we find that what really matters for the recommendation performance is the CL loss, rather than the graph augmentation. Optimizing the contrastive loss InfoNCE [19] learns more evenly distributed user/item representations no matter if graph augmentations are applied, which implicitly plays a role in mitigating the popularity bias [4]. Meanwhile, despite not as effective as expected, graph augmentations are not utterly useless in the sense that the properly perturbed versions of the original graph help learn representations invariant to the disturbance factors [1, 5]. However, generating hand-crafted graph augmentations requires constant reconstruction of the graph adjacency matrix during training, which is quite time-consuming. In addition, dropping a critical edge/node (e.g., a cut edge) may split a connected graph into a few disconnected components, at the risk of making the augmented graph and the original graph share little learnable invariance. In view of these defects, a follow-up question then arises: Are there more effective and efficient augmentation approaches?

Figure 1: Graph contrastive learning with edge dropout for recommendation.

<!-- image -->

In this paper, we give an affirmative answer to the question. On top of our finding that the uniformity of the representation distribution is the key point, we develop a graph-augmentation-free CL method in which the uniformity is more controllable. Technically, we follow the graph CL framework presented in Fig. 1, but we discard the dropout-based graph augmentation and instead add random uniform noises to the original representations for a representation-level data augmentation. Imposing different random noises creates variance between contrastive views, while the learnable invariance is still retained due to the controlled magnitude. Compared with the graph augmentation, the noise version directly regularizes the embedding space towards a more even distribution, which is easy-to-implement and far more efficient.

The major contributions of this paper are summarized as follows:

- We experimentally unravel why CL can boost recommendation performance and illustrate that the InfoNCE loss, rather than the graph augmentation, is the decisive factor.
- We propose a simple yet effective graph-augmentation-free CL method for recommendation that can regulate the uniformity in a smooth way. It can be an ideal alternative of cumbersome graph augmentation-based CL methods.
- We conduct a comprehensive experimental study on three benchmark datasets showing that the proposed method has distinct advantages over its graph augmentation-based counterparts in terms of recommendation accuracy and model training efficiency.

## 2 INVESTIGATION OF GRAPH CONTRASTIVE LEARNING IN RECOMMENDATION

## 2.1 Graph CL for Recommendation

CL is often applied to recommendation with a particular set of presumed representational invariances to data augmentations [29, 34, 41, 46]. In this paper, we revisit the most commonly used dropoutbased augmentation on graphs [29, 36] which assumes that the representations are invariant to partial structure perturbations. An investigation is launched into a state-of-the-art CL-based recommendation model, SGL [29], which performs node and edge dropout to augment the original graph and adopts InfoNCE [19] for CL. Formally, the joint learning scheme in SGL is defined as:

<!-- formula-not-decoded -->

which consists of two losses: recommendation loss L 𝑟𝑒𝑐 and CL loss L 𝑐𝑙 . The InfoNCE in SGL is formulated as:

<!-- formula-not-decoded -->

where 𝑖, 𝑗 are users/items in a sampled batch B , z ′ ( z ′′ ) are 𝐿 2 normalized 𝑑 -dimensional node representations learned from two different dropout-based graph augmentations, and 𝜏 &gt; 0 (e.g., 0.2) is the temperature. The CL loss encourages consistency between z ′ 𝑖 and z ′′ 𝑖 which are the augmented representations of the same node 𝑖 and are the positive sample of each other, while minimizing the agreement between 𝑧 ′ 𝑖 and z ′′ 𝑗 , which are the negative samples of each other. To learn the representations from the user-item graph, SGL employs a popular and effective graph encoder LightGCN [10] as its backbone, whose message passing process is defined as:

<!-- formula-not-decoded -->

where E ( 0 ) ∈ R | 𝑁 |× 𝑑 is the randomly initialized node embeddings, | 𝑁 | is the number of nodes, 𝐿 is the number of layers, and ˜ A ∈ R | 𝑁 |×| 𝑁 | is the normalized undirected adjacency matrix. By replacing ˜ A with the adjacency matrices of the corrupted graph augmentations, z ′ ( z ′′ ) can be learned via Eq. (3). Note that, z ′ 𝑖 = e ′ 𝑖 ∥ e ′ 𝑖 ∥ 2 and e ′ 𝑖 is the corrupted version of e 𝑖 in E . For conciseness, here we just abstract the core ingredients of SGL and LightGCN. More technical details can be found in the original papers [10, 29].

## 2.2 Necessity of Graph Augmentation

To demystify how CL-based recommendation methods work, we first investigate the necessity of the graph augmentation in SGL. We construct a new variant of SGL, termed SGL-WA (WA stands for 'without augmentation'), in which the CL loss is:

<!-- formula-not-decoded -->

Table 1: Performance comparison of different SGL variants.

| Method   | Yelp2018   | Yelp2018   | Amazon-Book   | Amazon-Book   |
|----------|------------|------------|---------------|---------------|
|          | Recall@20  | NDCG@20    | Recall@20     | NDCG@20       |
| LightGCN | 0.0639     | 0.0525     | 0.0410        | 0.0318        |
| SGL-ND   | 0.0644     | 0.0528     | 0.0440        | 0.0346        |
| SGL-ED   | 0.0675     | 0.0555     | 0.0478        | 0.0379        |
| SGL-RW   | 0.0667     | 0.0547     | 0.0457        | 0.0356        |
| SGL-WA   | 0.0671     | 0.0550     | 0.0466        | 0.0373        |
| CL Only  | 0.0245     | 0.0190     | 0.0314        | 0.0258        |

Because we only learn representations over the original user-item graph, then we have z ′ 𝑖 = z ′′ 𝑖 = z 𝑖 . The experiments for the performance comparison are conducted on two benchmark datasets: Yelp2018 and Amazon-Book [10, 26]. A three-layer setting is adopted and the hyperparameters are tuned according to the original paper of SGL (more experimental details in Section 4.1). The results are presented in Table 1. Three variants of SGL (with dropout rate 0.1) proposed in the paper are evaluated (-ND denotes node dropout, -ED is short for edge dropout, and -RW means random walk (i.e., multi-layer edge dropout). CL Only means that only the CL loss in SGL is minimized).

As can be observed, all the variants of SGL outperform LightGCN by large margins, which demonstrates the effectiveness of CL in improving recommendation performance. To our surprise, when the graph augmentation is detached, the performance gains are still so remarkable that SGL-WA even exhibits superiority over SGL-ND and SGL-RW. We conjecture that the node dropout and random walk (especially the former) are very likely to drop the key nodes and associated edges and hence break the correlated subgraphs into disconnected pieces, which highly distort the original graph. Such graph augmentations share little learnable invariance, and encouraging consistency between them probably has a negative impact. By contrast, one-time edge dropout is at a lower risk to largely disturb the semantics of the original graph, so that SGL-ED can maintain a trivial advantage over SGL-WA, which suggests the potential of a proper graph augmentation. However, considering the time-consuming reconstruction of the adjacency matrices in each epoch, we should rethink the necessity of graph augmentations and search for better alternatives. Besides, we wonder what underlies the outstanding performance of SGL-WA since no variances are provided in its CL part.

## 2.3 InfoNCE Loss Influences More

Wang and Isola [25] have identified that optimizing the contrastive loss intensifies two properties in the visual representation learning: alignment of features from positive pairs, and uniformity of the normalized feature distribution on the unit hypersphere. It is unclear if the CL-based recommendation methods exhibit similar patterns that can explain the results in Section 2.2. Since top-N recommendation is a one-class problem, we only investigate the uniformity by following the visualization method in [25].

Wefirst map the learned representations (randomly sample 2,000 users for each dataset) to 2-dimensional normalized vectors on the unit hypersphere S 1 (i.e., circle with radius 1) by using t-SNE [24].

All the representations are obtained when the methods reach their best performance. Then we plot the feature distributions with the nonparametric Gaussian kernel density estimation [3] in R 2 (shown in Fig. 2). For a clearer presentation, the density estimations on angles for each point on S 1 are also visualized. According to Fig. 2, we can observe notably different feature/density distributions. In the leftmost column, LightGCN shows highly clustered features that mainly reside on some narrow arcs. While in the second and the third columns, the distributions become more uniform, and the density estimation curves are less sharp, no matter if the graph augmentations are applied. In the forth column, we plot the features learned only by the contrastive loss in Eq. (2). The distributions are almost completely uniform.

We think that two reasons may explain the highly clustered feature distributions. The first is the message passing mechanism in LightGCN. With the increase of layers, node embeddings become locally similar. The second is the popularity bias [4] in the recommendation data. Recall the BPR loss [22] used in LightGCN:

<!-- formula-not-decoded -->

which is with a triplet input ( 𝑢, 𝑖, 𝑗 ) . To optimize the BPR loss, we can get the gradients w.r.t e 𝑢 : ∇ e 𝑢 = -𝜂 ( 1 -𝑠 )( e 𝑖 -e 𝑗 ) , where 𝜂 is the learning rate, 𝜎 is the sigmoid function, 𝑠 = 𝜎 ( e ⊤ 𝑢 e 𝑖 -e ⊤ 𝑢 e 𝑗 ) , e 𝑢 is the user embedding, and e 𝑖 and e 𝑗 denote the positive and negative item embeddings, respectively. Since the recommendation data usually follows a long-tail distribution, when 𝑖 is a popular item with a large number of interactions, the user embedding will be constantly updated towards 𝑖 's direction (i.e., -∇ e 𝑢 ). The message passing mechanism further exacerbates the clustering problem (i.e., e 𝑢 and e 𝑖 aggregate information from each other in the graph convolution) and causes the representation degeneration [21].

As for the distributions in other columns, by rewriting Eq. (4), we can derive

<!-- formula-not-decoded -->

Because 1 / 𝜏 is a constant, optimizing the CL loss is actually minimizing the cosine similarity between different nodes embeddings e 𝑖 and e 𝑗 , which will push connected nodes away from the highdegree hubs in the representation space and lead to a more even distribution.

By associating the results in Table 1 with the distributions in Fig. 2, we can easily draw a conclusion that the uniformity of the distribution is the underlying factor that has a decisive impact on the recommendation performance in SGL, rather than the dropoutbased graph augmentations. Optimizing the CL loss can be seen as an implicit way to debias (discussed in section 4.2) because a more even representation distribution can preserve the intrinsic characteristics of nodes and improve the generalization ability. This can be a persuasive explanation for the unexpected performance of SGL-WA. It also should be noted that, by only minimizing the CL loss in Eq. (2), a poor performance will be reached, which means that a positive correlation between the uniformity and the performance only holds in a limited scope. The excessive pursuit to the uniformity will overlook the closeness of interacted pairs and similar users/items, and impairs recommendation performance.

<!-- image -->

(b) Distribution of item representations learned from the dataset of Amazon-Book.

Figure 2: We plot feature distributions with Gaussian kernel density estimation (KDE) in R 2 (the darker the color is, the more points fall in that area.) and KDE on angles (i.e., arctan2(y, x) for each point (x,y) ∈ S 1 ).

## 3 SIMGCL: SIMPLE GRAPH CONTRASTIVE LEARNING FOR RECOMMENDATION

Based on the findings in Section 2, we speculate that by adjusting the uniformity of the learned representation in a certain scope, the optimal performance can be reached. In this section, we aim to develop a Sim ple G raph C ontrastive L earning method ( SimGCL ) for recommendation that can smoothly regulate the uniformity and provide informative variance to maximize the benefit from CL.

## 3.1 Motivation and Formulation

Since manipulating the graph structure for a more evenly-distributed representation space is intractable and time-consuming, we shift our attention to the embedding space. Inspired by the adversarial examples [8] which are constructed by adding imperceptibly small perturbation to the input images, we directly add random noises to the representation for an efficient and effective augmentation.

Formally, given a node 𝑖 and its representation e 𝑖 in the 𝑑 -dimensional embedding space, we can implement the following representation-level augmentation:

<!-- formula-not-decoded -->

where the added noise vectors Δ ′ 𝑖 and Δ ′′ 𝑖 are subject to ∥ Δ ∥ 2 = 𝜖 and Δ = ¯ Δ ⊙ sign ( e 𝑖 ) , ¯ Δ ∈ R 𝑑 ∼ 𝑈 ( 0 , 1 ) . The first constraint controls the magnitude of Δ , and Δ is numerically equivalent to points on a hypersphere with the radius 𝜖 . The second constraint requires that e 𝑖 , Δ ′ and Δ ′′ should be in the same hyperoctant, so that adding the noises will not cause a large deviation of e 𝑖 , making less valid positive samples. In Fig. 3, we illustrate Eq. (7) in R 2 . By adding the scaled noise vectors to the original representation, we rotate e 𝑖 by two small angles ( 𝜃 1 and 𝜃 2). Each rotation corresponds to a deviation of e 𝑖 , and leads to an augmented representation ( e ′ 𝑖 and e ′′ 𝑖 ). Since the rotation is small enough, the augmented representation retains most information of the original representation and meanwhile also keeps some variance. Note that, for each node representation, the added random noises are different.

Figure 3: An illustration of the proposed random noisebased data augmentation in R 2 .

<!-- image -->

Following SGL, we adopt LightGCN as the graph encoder to propagate node information and amplify the impact of the variance due to its simple structure and effectiveness. At each layer, different scaled random noises are imposed on the current node embeddings. The final perturbed node representations are learned by:

<!-- formula-not-decoded -->

It should be mentioned that we skip the input embedding E ( 0 ) in all the three encoders when calculating the final representations, because we experimentally find that skipping it can lead to slight performance improvement in our setting. However, without the CL task, this operation will result in a performance drop of LightGCN. Finally, we also unify the BPR loss (Eq. (5)) and the CL loss (Eq. (2)), and then use Adam to optimize the joint loss presented in Eq. (1).

## 3.2 Regulating Uniformity

In SimGCL, two hyperparameters 𝜆 and 𝜖 can influence the uniformity of the representations which are critical to the performance. But 𝜖 can explicitly and smoothly regulate the uniformity beyond by only tuning 𝜆 . By adjusting the value of 𝜖 , we can directly control how far the augmented representations deviate from the original. Intuitively, a larger 𝜖 will lead to a more roughly even distribution of the learned representation, because when the augmented representations are enough far away from the original, the information lying in their representations is also considerably influenced by the noises. As the noises are sampled from a uniform distribution, by contrasting the augmented representations, the original representation is regularized towards higher uniformity. We present the following experimental analysis to demonstrate it.

In [25], a metric is proposed to measure the uniformity of the representation, which is the logarithm of the average pairwise Gaussian potential (a.k.a. the Radial Basis Function (RBF) kernel):

<!-- formula-not-decoded -->

where 𝑓 ( 𝑢 ) outputs the 𝐿 2 normalized embedding of 𝑢 . We choose the popular items (with more than 200 interactions) and randomly sample 5,000 users in the dataset of Yelp2018 to form the user-item pairs, and then compute the uniformity of their representations in the SGL variants and SimGCL with Eq. (9). For a fair comparison, a three-layer setting is applied to all the compared methods with 𝜆 = 0 . 1. We then tune 𝜖 to observe how the uniformity changes. The uniformity is checked after every epoch, and we record the values in the first 30 epochs during which the compared methods all converge to their optimal solutions.

As clearly shown in Fig. 4, similar trends are observed on all the curves. At the initial stage, all the methods have highly uniformlydistributed representations because we use Xavier initialization, which is a special uniform distribution. With the training proceeding, the uniformity declines ( L uniform gets higher), and after reaching the peak, the uniformity improves till convergence and maintains this tendency. As for SimGCL, with the increase of 𝜖 , it tends to learn more even representations, and even a very small 𝜖 = 0 . 01 leads to higher uniformity compared with the SGL variants. As a result, users (especially the long-tail users) are less affected by the popular items. In the rightmost column in Fig. 2, we also plot the representation distributions of SimGCL with 𝜖 = 0 . 1. We can clearly see that the distributions are evidently more even than those learned by SGL variants and LightGCN. All these results can support our claim that by replacing graph augmentations with the noise-based augmentations, SimGCL is more capable of controlling the uniformity of learned representations so as to debias.

Figure 4: Trends of uniformity. The star indicates the epoch where the best recommendation performance is reached. Lower L uniform numbers are better.

<!-- image -->

## 3.3 Complexity

In this section, we analyze the time complexity of SimGCL, and compare it with that of LightGCN and its graph-augmentation based counterpart SGL-ED. We hereby discuss the batch time complexity since the in-batch negative sampling is a widely used trick in CL [5]. Let | 𝐸 | be the edge number in the graph, 𝑑 be the embedding size, 𝐵 denote the batch size, 𝑀 represent the node number in a batch, and 𝜌 denote the edge keep rate in SGL-ED. We can derive:

Table 2: The comparison of time complexity

| Component         | LightGCN                  | SGL-ED                                      | SimGCL                    |
|-------------------|---------------------------|---------------------------------------------|---------------------------|
| Adjacency Matrix  | O( 2 &#124; 𝐸 &#124;)     | O( 2 &#124; 𝐸 &#124; + 4 𝜌 &#124; 𝐸 &#124;) | O( 2 &#124; 𝐸 &#124;)     |
| Graph Convolution | O( 2 &#124; 𝐸 &#124; 𝐿𝑑 ) | O(( 2 + 4 𝜌 ) &#124; 𝐸 &#124; 𝐿𝑑 )          | O( 6 &#124; 𝐸 &#124; 𝐿𝑑 ) |
| BPR Loss          | O( 2 𝐵𝑑 )                 | O( 2 𝐵𝑑 )                                   | O( 2 𝐵𝑑 )                 |
| CL Loss           | -                         | O( 𝐵𝑑 + 𝐵𝑀𝑑 )                               | O( 𝐵𝑑 + 𝐵𝑀𝑑 )             |

- For LightGCN and SimGCL, no graph augmentations are required, so they just need to normalize the original adjacency matrix which has 2 | 𝐸 | non-zero elements. For SGL-ED, two graph augmentations are used and each has 2 𝜌 | 𝐸 | non-zero elements.
- In the graph convolution stage, a three-encoder architecture (see Fig. 1) is employed in both SGL-ED and SimGCL to learn augmented node representations. So, the time costs of SGL-ED and SimGCL are almost three times that of LightGCN.
- As for the recommendation loss, three methods all use the BPR loss and each batch contains 𝐵 interactions, so they have the same time cost in this component.
- When calculating the CL loss, the computation costs between the positive/negative samples are O( 𝐵𝑑 ) and O( 𝐵𝑀𝑑 ) , respectively, because each node only considers itself as the positive, while the other nodes all are negatives.

Table 3: Performance Comparison for different CL methods on three benchmarks.

| Method   | Method                 | Douban-Book                                            | Douban-Book                                                            | Yelp2018                                                           | Yelp2018                                                           | Amazon-Book                                                           | Amazon-Book                                                           |
|----------|------------------------|--------------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------|
|          |                        | Recall                                                 | NDCG                                                                   | Recall                                                             | NDCG                                                               | Recall                                                                | NDCG                                                                  |
| 1-Layer  | LightGCN SGL-ND SGL-ED | 0.1394 0.1619 (+16.1%) 0.1658 (+18.9%) 0.1658 (+18.9%) | 0.1165 0.1448 (+24.3%) 0.1491 (+28.0%) 0.1491 (+28.0%) 0.1454 (+24.8%) | 0.0631 0.0643 (+1.9%) 0.0637 (+1.0%) 0.0637 (+1.0%) 0.0628 (-0.4%) | 0.0515 0.0529 (+2.7%) 0.0526 (+2.1%) 0.0526 (+2.1%) 0.0525 (+1.9%) | 0.0384 0.0432 (+12.5%) 0.0451 (+17.4%) 0.0451 (+17.4%) 0.0403 (+4.9%) | 0.0298 0.0334 (+12.1%) 0.0353 (+18.5%) 0.0353 (+18.5%) 0.0320 (+7.4%) |
| 1-Layer  | SGL-RW                 |                                                        |                                                                        |                                                                    |                                                                    |                                                                       |                                                                       |
| 1-Layer  | SGL-WA                 | 0.1628 (+16.8%)                                        |                                                                        |                                                                    |                                                                    |                                                                       |                                                                       |
| 1-Layer  | SimGCL                 | 0.1720 (+23.4%)                                        | 0.1519 (+30.4%)                                                        | 0.0689 (+9.2%)                                                     | 0.0572 (+11.1%)                                                    | 0.0453                                                                | 0.0358                                                                |
| 1-Layer  |                        |                                                        |                                                                        |                                                                    |                                                                    | (+18.0%)                                                              | (+20.1%)                                                              |
|          | LightGCN               | 0.1485                                                 | 0.1272                                                                 | 0.0622                                                             | 0.0504                                                             | 0.0411                                                                | 0.0315                                                                |
|          | SGL-ND                 | 0.1622 (+9.2%)                                         | 0.1434 (+12.7%)                                                        | 0.0658 (+5.8%)                                                     | 0.0538 (+6.7%)                                                     | 0.0427 (+3.9%)                                                        | 0.0335 (+6.3%)                                                        |
|          | SGL-ED                 | 0.1721 (+15.9%)                                        | 0.1525 (+19.9%)                                                        | 0.0668 (+7.4%)                                                     | 0.0549 (+8.9%)                                                     | 0.0468 (+13.9%)                                                       | 0.0371 (+17.8%)                                                       |
|          | SGL-RW SGL-WA          | 0.1710 (+15.2%) 0.1687 (+13.6%)                        | 0.1516 (+19.2%)                                                        | 0.0644 (+3.5%)                                                     | 0.0530 (+5.2%) 0.0544 (+7.9%)                                      | 0.0453 (+10.2%) 0.0453 (+10.2%)                                       | 0.0358 (+13.7%)                                                       |
|          |                        |                                                        | 0.1501 (+18.0%)                                                        | 0.0653 (+5.0%)                                                     |                                                                    |                                                                       | 0.0358 (+13.7%)                                                       |
|          | SimGCL                 | 0.1770 (+19.2%)                                        | 0.1582 (+24.4%)                                                        | 0.0719 (+15.6%)                                                    | 0.0601 (+19.2%)                                                    | 0.0507 (+23.4%)                                                       | 0.0405 (+28.6%)                                                       |
|          | LightGCN               | 0.1501                                                 | 0.1282                                                                 | 0.0639                                                             | 0.0525                                                             | 0.0410                                                                | 0.0318                                                                |
|          | SGL-ND                 | 0.1626 (+8.3%)                                         | 0.1450 (+13.1%)                                                        | 0.0644 (+0.8%)                                                     | 0.0528 (+0.6%)                                                     | 0.0440 (+7.3%)                                                        | 0.0346 (+8.8%)                                                        |
|          | SGL-RW                 | 0.1730 (+15.3%)                                        | 0.1546 (+20.6%)                                                        | 0.0667 (+4.4%)                                                     | 0.0547 (+4.2%)                                                     | 0.0457 (+11.5%)                                                       | 0.0356 (+12.0%)                                                       |
|          | SGL-WA                 |                                                        |                                                                        |                                                                    |                                                                    |                                                                       |                                                                       |
|          |                        | 0.1705 (+12.0%)                                        | 0.1525 (+19.0%)                                                        | 0.0671 (+5.0%)                                                     | 0.0550 (+4.8%)                                                     | 0.0466 (+13.7%)                                                       | 0.0373 (+18.4%)                                                       |
|          | SimGCL                 | 0.1772 (+18.1%)                                        | 0.1583 (+23.5%)                                                        | 0.0721 (+12.8%)                                                    | 0.0601 (+14.5%)                                                    | 0.0515 (+25.6%)                                                       | 0.0414 (+30.2%)                                                       |

Comparing SimGCL with SGL-ED, we can clearly see that SGLED theoretically spends less time for graph convolution, and this bonus may offset SimGCL's advantage for the adjacency matrix construction. However, when putting them into practice, we actually observe that SimGCL is much more time-efficient. That is because, the computation for the graph convolution is mostly finished on GPUs, while the graph perturbation is performed on CPUs. Besides, in each epoch, the adjacency matrices of graph augmentations in SGL-ED need to be reconstructed. While in SimGCL, the adjacency matrix of the original graph only needs to be generated once before the training. In a nutshell, SimGCL is far more efficient than SGL, beyond what we can observe from the theoretical analysis.

## 4 EXPERIMENTAL RESULTS

## 4.1 Experimental Settings

Datasets. Three public benchmark datasets: Douban-Book [39] (#user 13,024, #item 22,347, #interaction 792,062), Yelp2018 [10] (#user 31,668 #item 38,048, #interaction 1,561,406), and AmazonBook [29] (#user 52,463, #item 91,599, #interaction 2,984,108) are used in our experiments to evaluate SimGCL. Because we focus on the Top-N recommendation, following the convention in the previous research [41, 41], we discard ratings less than 4 in DoubanBook, which is with a 1-5 rating scale, and reset the rest to 1. We split the datasets into three parts (training set, validation set, and test set) with a ratio of 7:1:2. Two common metrics: Recall@ 𝐾 and NDCG@ 𝐾 are used and we set 𝐾 =20. For a rigorous and unbiased evaluation, each experiment in this section is conducted 5 times with ranking all the items and we then report the average result. Baselines. Besides LightGCN and the SGL variants, the following recent data augmentation-based methods are compared.

- Mult-VAE [16] is a variational autoencoder-based recommendation model. It can be seen as a special self-supervised recommendation model because it has a reconstruction objective.
- DNN+SSL [35] is a recent DNN-based recommendation method which adopts the similar architecture in Fig. 1, and conducts feature masking for CL.
- BUIR [15] has a two-branch architecture which consists of a target network and an online network, and only uses positive examples for self-supervised learning.
- MixGCL [11] designs the hop mixing technique to synthesize hard negatives for graph collaborative filtering by embedding interpolation.

Hyperparameters. For a fair comparison, we refer to the best hyperparameter settings reported in the original papers of the baselines and then fine-tune all the hyperparameters of the baselines with the grid search. As for the general settings of all the baselines, the Xavier initialization is used on all the embeddings. The embedding size is 64, the parameter for 𝐿 2 regularization is 10 -4 and the batch size is 2048. We use Adam with the learning rate 0.001 to optimize all the models. In SimGCL and SGL, we empirically let the temperature 𝜏 = 0 . 2, and this value is also reported as the best in the original paper of SGL.

## 4.2 SGL vs. SimGCL: From a Comprehensive Perspective

As one of the core claims of this paper is that graph augmentations are not indispensable and inefficient in CL-based recommendation, in this part, we conduct a comprehensive comparison between SGL and SimGCL in terms of the recommendation performance, convergence speed, running time, and ability to debias.

4.2.1 Performance Comparison. Wefirst further compare SGL with SimGCL on three different datasets with different layer settings. We do not report the results with deeper layers because both SGL and SimGCL reach their best performances with shallow structures. The used hyperparameters are reported in section 4.3. We thicken the figures representing the best performance and underline the second best. The improvements are calculated by using LightGCN as the baseline. According to Table 3, we can draw the following observations and conclusions:

- All the SGL variants and SimGCL are effective in improving LightGCN under different settings. The largest improvements are observed on Amazon-Book where SimGCL can remarkably improve LightGCN by 25.6% on Recall, and 30.2% on NDCG with a 3-layer setting.
- SGL-ED is the most effective variant of SGL while SGL-ND is the least effective. When a 2-layer or 3-layer setting is used, SGL-WA outperforms SGL-ND in most cases and shows advantages over SGL-RW in a few cases. These results demonstrate that the CL loss is the main driving force of the performance improvement while intuitive graph augmentations may not be as effective as expected, and some of them may even lower the performance.
- SimGCL shows the best performance in all the cases, which proves the effectiveness of the random noised-based data augmentation. Particularly, on the two larger datasets: Yelp2018 and Amazon-Book, SimGCL significantly outperforms the SGL variants by large margins.

4.2.2 Convergence Speed Comparison. In this part, we show that SimGCL converges much faster than SGL does. A 2-layer setting is used in this part and the other parameters remain unchanged.

According to Fig. 5 and Fig. 6, we can observe that, SimGCL reaches its best performance on the test set at the 25 th , the 11 th , and the 10 th epoch on Douban-Book, Yelp2018, and Amazon-Book, respectively. By contrast, SGL-ED peaks at the 38 th , the 17 th , and the 14 th epoch, respectively. SimGCL only spends 2/3 epochs that the SGL variants need. Besides, the curve of SGL-WA almost overlaps that of SGL-ED on Yelp2018 and Amazon-Book, and exhibits the same tendency to convergence. It seems that the dropout-based graph augmentations cannot speed up the model for a faster convergence. Despite that, all the CL-based methods show advantages over LightGCN at the convergence speed. When the other three methods begin to get overfitted, LightGCN is still hundreds of epochs distant from getting converged.

In the paper of SGL, the authors guess that the multiple negatives in the CL loss may contribute to the fast convergence. However, with almost infinite negative samples created by dropout, SGLED is basically on par with SGL-WA in speeding up the training, though the latter only has a certain number of negative samples. As for SimGCL, we consider that the remarkable convergence speed stems from the noises. By analyzing the gradients from the CL loss, we find that the noises averagely provide a constant increment, working like a momentum. In addition to the results in Fig. 5 and 6, we also find that the training accelerates with 𝜖 getting larger. But when it is overlarge (e.g., greater than 1), despite the rapid decrease of BPR loss, SimGCL requires more time to get converged. A large 𝜖 acts like a large learning rate, causing the progressive zigzag optimization that will overshoot the minimum.

Table 4: Running time for per epoch (x in the brackets represents times).

| Method   | Douban-Book Time (s)   | Yelp2018 Time (s)   | Amazon-Book Time (s)   |
|----------|------------------------|---------------------|------------------------|
| LightGCN | 3.6                    | 13.6                | 41.5                   |
| SGL-WA   | 4.4 (1.2x)             | 16.3 (1.2x)         | 47.0 (1.1x)            |
| SGL-ED   | 13.3 (3.7x)            | 62.3 (4.6x)         | 235.3 (5.7x)           |
| SimGCL   | 6.1 (1.7x)             | 27.9 (2.1x)         | 98.4 (2.4x)            |

<!-- image -->

Figure 5: The performance curves in the first 50 epochs.

Figure 6: The loss curves in the first 50 epochs.

<!-- image -->

4.2.3 Running Time Comparison. In this part, we report the real running time that the compared methods cost for one epoch. The results in Table 4 are collected on an Intel(R) Xeon(R) Gold 5122 CPU and a GeForce RTX 2080Ti GPU.

As shown in Table 4, we calculate how many times slower the other methods are when compared with LightGCN. Because there is no graph augmentation in SGL-WA, we can see its running speed is very close to that of LightGCN. For SGL-ED, two graph augmentations are required and the computation in this part is mostly finished on CPUs, so that it is even 5.7 times slower than LightGCN on Amazon-Book. The running time increases with the volume of the datasets. By contrast, despite not as fast as SGL-WA, SimGCL is only 2.4 times slower than LightGCN on Amazon-Book, and the growth trend is far lower than that of SGL-ED. Considering that SimGCL only needs 2/3 the epochs that SGL-ED spends, it outperforms SGL in all aspects w.r.t efficiency.

4.2.4 Ability to Debias. The InfoNCE loss is found to have the ability to implicitly alleviate the popularity bias by learning more even representations. To verify that SimGCL upgrades this ability with the noise-based representation augmentation, we divide the test set into three subsets in proportion to the popularity of items. 80% items with the fewest number of clicks/purchases are labelled 'Unpopular', 5% items which are most clicked/purchased are labelled 'Popular', and the rest items are labelled 'Normal'. We then conduct experiments to check the Recall@20 value that each group contributes (overall Recall@20 value is the sum of the values from three groups). The results are illustrated in Fig. 7.

We can clearly see that the SimGCL's improvements all come from the items with lower popularity. Its prominent advantage on recommending long-tail items largely compensates for its loss on the 'Popular' group. By contrast, LightGCN is inclined to recommend popular items and achieves the highest recall value on the last two datasets. The SGL variants fall between LightGCN and SimGCL on exploring long-tail items and exhibit similar recommendation preference. Combining Fig. 2 with Fig. 7, we can easily find that there is a positive correlation between the uniformity of representations and the ability to debias. Since the popular items probably have been exposed to users from other sources, recommending them may no be a good choice. On this point, SimGCL significantly outperforms SGL, and its extraordinary performance on discovering long-tail items fits the real need of users.

## 4.3 Parameter Sensitivity Analysis

In this part, we investigate the impact of the two important hyperparameters in SimGCL. Here we adopt the experimental settings used in section 4.2.2.

4.3.1 Impact of 𝜆 . By fixing 𝜖 at 0.1, we change 𝜆 to a set of predetermined representative values presented in Fig. 8. As can be observed, with the increase of 𝜆 , the performance of SimGCL starts to increase at the beginning, and it gradually reaches its peak when 𝜆 is 0.2 on Douban-Book, 0.5 on Yelp2018, and 2 on Amazon-Book. Afterwards, it starts to decline. Besides, in contrast to Fig. 9, more dramatic changes are observed in Fig. 8 though 𝜖 and 𝜆 are tuned in the same scope, which demonstrates that 𝜖 can provide a finergrained regulation beyond that provided only by tuning 𝜆 .

4.3.2 Impact of 𝜖 . We think a larger 𝜖 leads to a more even distribution that can help to debias. However, when it is too large, the recommendation task will be hindered because the high similarity between connected nodes cannot be reflected by an over-uniform distribution. We fix 𝜆 at the best values on the three datasets as reported in Fig. 8, and then adjust 𝜖 to see the performance change. As shown in Fig. 9, the shapes of the curves are as expected. On all the datasets, when 𝜖 is near 0.1, SimGCL achieves the best performance. We also find that initializing embeddings with uniform distributions (including Xavier initialization) leads to 3%-4% performance improvement compared with Gaussian initialization.

## 4.4 Performance Comparison with Other Methods

To further confirm the outstanding competence of SimGCL, we compare it with other four recently proposed data augmentationbased methods. According to Table 5, SimGCL outperforms all the baselines, and MixGCF is the runner-up. Meanwhile, we find that some data augmentation-based recommendation methods are not as powerful as expected, and even outperformed by LightGCN in many cases. We attribute their failure to: (1). LightGCN, MixGCF, and SimGCL are all based on the graph convolution mechanism, which are more capable of modeling graph data compared with Mult-VAE. (2). DNNs are proved effective when user/item features are available. In our datasets, no features are provided and we mask embeddings learned by DNN to conduct self-supervised learning, so it cannot fulfill itself in this situation. (3). In the paper of BUIR, it removes long-tail nodes to achieve a good performance, but we use all the users and items. Besides, its siamese network structure may also collapse to a trivial solution on some long-tail nodes because it does not use negative examples, which may account for its incompetence.

<!-- image -->

Figure 7: Performance comparison over different item groups

Figure 8: Influence of the magnitude 𝜆 of CL.

<!-- image -->

Figure 9: Influence of the noise magnitude 𝜖 .

<!-- image -->

## 4.5 Performance Comparison with Different Types of Noises

In SimGCL, we use the random noises sampled from a uniform distribution to implement data augmentation. However, there are other types of noises including the Gaussian noises and adversarial noises. Here we also test different noises and report their best results in Table 6 ( 𝑢 denotes uniform noises sampled from 𝑈 ( 0 , 1 ) , 𝑝 represents the positive uniform noises which differ from 𝑢 in not satisfying the second constraint in section 3.1, 𝑔 denotes Gaussian noises generated by the standard Gaussian distribution, and 𝑎

Table 5: Performance comparison with other SOTA models.

| Method   | Douban-Book   | Douban-Book   | Yelp2018   | Yelp2018   | Amazon-Book   | Amazon-Book   |
|----------|---------------|---------------|------------|------------|---------------|---------------|
|          | Recall        | NDCG          | Recall     | NDCG       | Recal         | NDCG          |
| LightGCN | 0.1501        | 0.1282        | 0.0639     | 0.0525     | 0.0411        | 0.0315        |
| Mult-VAE | 0.1310        | 0.1103        | 0.0584     | 0.0450     | 0.0407        | 0.0315        |
| DNN+SSL  | 0.1366        | 0.1148        | 0.0483     | 0.0382     | 0.0438        | 0.0337        |
| BUIR     | 0.1127        | 0.8938        | 0.0487     | 0.0404     | 0.0260        | 0.0209        |
| MixGCF   | 0.1731        | 0.1552        | 0.0713     | 0.0589     | 0.0485        | 0.0378        |
| SimGCL   | 0.1772        | 0.1583        | 0.0721     | 0.0601     | 0.0515        | 0.0410        |

denotes adversarial noises generated by FGSM [8]). According to Table 6, SimGCL 𝑔 shows comparable performance while SimGCL 𝑎 is less effective. The possible reason is that we apply 𝐿 2 normalization to the noises. The normalized noises generated by the standard Gaussian distribution can fit a much flatter Gaussian distribution (can be easily proved) which approximates a uniform distribution. So, the comparable results are observed. As for SimGCL 𝑎 , the adversarial noises are generated by only targeting maximizing the CL loss while the recommendation loss has a dominant status that impacts the performance more during optimization. As for SimGCL 𝑝 , we notice a slight performance drop compared with SimGCL 𝑢 in most cases, which suggests the necessity of the directional constraint for creating more informative augmentations.

Table 6: Performance comparison between different SimGCL variants.

| Method   | Douban-Book   | Douban-Book   | Yelp2018   | Yelp2018   | Amazon-Book   | Amazon-Book   |
|----------|---------------|---------------|------------|------------|---------------|---------------|
|          | Recall        | NDCG          | Recall     | NDCG       | Recal         | NDCG          |
| LightGCN | 0.1485        | 0.1272        | 0.0639     | 0.0525     | 0.0411        | 0.0315        |
| SimGCL 𝑎 | 0.1561        | 0.1379        | 0.0604     | 0.0505     | 0.0455        | 0.0358        |
| SimGCL 𝑝 | 0.1751        | 0.1565        | 0.0708     | 0.0593     | 0.0514        | 0.0409        |
| SimGCL 𝑔 | 0.1773        | 0.1586        | 0.0718     | 0.0599     | 0.0511        | 0.0408        |
| SimGCL 𝑢 | 0.1772        | 0.1583        | 0.0721     | 0.0601     | 0.0515        | 0.0414        |

## 5 RELATED WORK

## 5.1 Graph Neural Recommendation Models

Graph Neural Networks (GNNs) [6, 31] now have become widely acknowledged powerful architectures for modeling recommendation data. This new neural network paradigm ends the regime of MLP-based recommendation models in the academia, and boosts the neural recommender systems to a new level. A large number of recommendation models, which adopt GNNs as their bases, claim that they have achieved state-of-the-art performance [10, 40, 41] in different subfields. Particularly, GCN [14], as the most prevalent variant of GNNs, further fuels the development of the graph neural recommendation models like GCMC [2], NGCF [26], LightGCN [10], and LCF [43]. Despite the different implementations in details, these GCN-driven models share a common idea that is to acquire the information from the neighbors in the user-item graph layer by layer to refine the target node's embeddings and fulfill graph reasoning [30]. Among these methods, LightGCN is the most popular one due to its simple structure and decent performance. Following [28], it removes the redundant operations including transformation matrices and nonlinear activation functions. Such a design is proved efficient and effective, and inspires a lot of follow-up CL-based recommendation models like SGL [29] and MHCN [39].

## 5.2 Contrastive Learning in Recommendation

As CL works in a self-supervised manner [42], it is inherently a possible solution to the data sparsity issue [37, 38] in recommender systems. Inspired by the achievements of CL in other fields, there has been a wave of new research that integrates CL with recommendation [18, 20, 29, 33, 39, 41, 46]. Zhou et al. [46] adopted random masking on attributes and items to create sequence augmentations for sequential model pretraining with mutual information maximization. Wei et al. [27] reformulated the cold-start item representation learning from an information-theoretic standpoint and maximized the mutual dependencies between item content and collaborative signals to alleviate the data sparsity issue. Similar ideas are also found in [35], where a two-tower DNN architecture is developed for recommendation, in which the item tower is also shared for contrasting augmented item features. SEPT [39] and COTREC [32] further propose to mine multiple positive samples with semisupervised learning on the perturbed graph for social/session-based recommendation. In addition to the dropout, CL4Rec [34] proposes to reorder and crop item segments for sequential data augmentation. Yu et al. [41], Zhang et al. [44] and Xia et al. [33] leveraged hypergraph to model recommendation data, and proposed to contrast different hypergraph structures for representation regularization. In addition to the data sparsity problem, Zhou et al. [45] theoretically proved that CL can also mitigate the exposure bias in recommendation, and developed a method named CLRec to improve deep match in terms of fairness and efficiency.

## 6 CONCLUSION

In this paper, we revisit the dropout-based CL in recommendation, and investigate how it improves recommendation performance. We reveal that, in CL-based recommendation models, the CL loss is the core and the graph augmentation only plays a secondary role. Optimizing the CL loss leads to a more even representation distribution, which helps to debias in the scenario of recommendation. We then develop a simple graph-augmentation-free CL method to regulate the uniformity of the representation distribution in a more straightforward way. By adding directed random noises to the representation for different data augmentations and contrast, the proposed method can significantly enhance recommendation. The extensive experiments demonstrate that the proposed method outperforms its graph augmentation-based counterparts and meanwhile the training time is dramatically reduced.

## ACKNOWLEDGEMENT

This work is supported by Australian Research Council Future Fellowship (Grant No. FT210100624), Discovery Project (Grant No. DP190101985) and Discovery Early Career Research Award (Grant No. DE200101465).

## REFERENCES

- [1] Philip Bachman, R Devon Hjelm, and William Buchwalter. 2019. Learning representations by maximizing mutual information across views. arXiv preprint arXiv:1906.00910 (2019).
- [2] Rianne van den Berg, Thomas N Kipf, and Max Welling. 2017. Graph convolutional matrix completion. arXiv preprint arXiv:1706.02263 (2017).
- [3] Zdravko I Botev, Joseph F Grotowski, and Dirk P Kroese. 2010. Kernel density estimation via diffusion. The annals of Statistics 38, 5 (2010), 2916-2957.
- [4] Jiawei Chen, Hande Dong, Xiang Wang, Fuli Feng, Meng Wang, and Xiangnan He. 2020. Bias and debias in recommender system: A survey and future directions. arXiv preprint arXiv:2010.03240 (2020).
- [5] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. A simple framework for contrastive learning of visual representations. In International conference on machine learning . PMLR, 1597-1607.
- [6] Chen Gao, Yu Zheng, Nian Li, Yinfeng Li, Yingrong Qin, Jinghua Piao, Yuhan Quan, Jianxin Chang, Depeng Jin, Xiangnan He, et al. 2021. Graph Neural Networks for Recommender Systems: Challenges, Methods, and Directions. arXiv preprint arXiv:2109.12843 (2021).
- [7] Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple Contrastive Learning of Sentence Embeddings. arXiv preprint arXiv:2104.08821 (2021).
- [8] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. 2014. Explaining and harnessing adversarial examples (2014). arXiv preprint arXiv:1412.6572 (2014).
- [9] Bowen Hao, Jing Zhang, Hongzhi Yin, Cuiping Li, and Hong Chen. 2021. Pretraining graph neural networks for cold-start users and items representation. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining . 265-273.
- [10] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yong-Dong Zhang, and Meng Wang. 2020. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval . ACM, 639-648.
- [11] Tinglin Huang, Yuxiao Dong, Ming Ding, Zhen Yang, Wenzheng Feng, Xinyu Wang, and Jie Tang. 2021. MixGCF: An Improved Training Method for Graph Neural Network-based Recommender Systems. (2021).
- [12] Ashish Jaiswal, Ashwin Ramesh Babu, Mohammad Zaki Zadeh, Debapriya Banerjee, and Fillia Makedon. 2021. A survey on contrastive self-supervised learning. Technologies 9, 1 (2021), 2.
- [13] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. 2020. Supervised Contrastive Learning. In Advances in Neural Information Processing Systems, NeurIPS 2020 .
- [14] Thomas N. Kipf and Max Welling. 2017. Semi-Supervised Classification with Graph Convolutional Networks. In 5th International Conference on Learning Representations, ICLR 2017 .
- [15] Dongha Lee, SeongKu Kang, Hyunjun Ju, Chanyoung Park, and Hwanjo Yu. 2021. Bootstrapping User and Item Representations for One-Class Collaborative Filtering. In The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval , Fernando Diaz, Chirag Shah, Torsten Suel, Pablo Castells, Rosie Jones, and Tetsuya Sakai (Eds.). 1513-1522.
- [16] Dawen Liang, Rahul G Krishnan, Matthew D Hoffman, and Tony Jebara. 2018. Variational autoencoders for collaborative filtering. In Proceedings of the 2018 world wide web conference . 689-698.
- [17] Xiao Liu, Fanjin Zhang, Zhenyu Hou, Zhaoyu Wang, Li Mian, Jing Zhang, and Jie Tang. 2020. Self-supervised learning: Generative or contrastive. arXiv preprint arXiv:2006.08218 1, 2 (2020).
- [18] Jianxin Ma, Chang Zhou, Hongxia Yang, Peng Cui, Xin Wang, and Wenwu Zhu. 2020. Disentangled Self-Supervision in Sequential Recommenders. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining . 483-491.
- [19] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018).
- [20] Ruihong Qiu, Zi Huang, and Hongzhi Yin. 2021. Memory Augmented MultiInstance Contrastive Predictive Coding for Sequential Recommendation. In 2021 IEEE International Conference on Data Mining (ICDM) . IEEE, 519-528.
- [21] Ruihong Qiu, Zi Huang, Hongzhi Yin, and Zijian Wang. 2022. Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation. In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining . 813-823.
- [22] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. BPR: Bayesian personalized ranking from implicit feedback. In Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence . AUAI Press, 452-461.
- [23] Badrul Munir Sarwar. 2001. Sparsity, scalability, and distribution in recommender systems. (2001).
- [24] Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE. Journal of machine learning research 9, 11 (2008).
- [25] Tongzhou Wang and Phillip Isola. 2020. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International Conference on Machine Learning . PMLR, 9929-9939.
- [26] Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. 2019. Neural graph collaborative filtering. In Proceedings of the 42nd international ACM SIGIR conference on Research and development in Information Retrieval . 165-174.
- [27] Yinwei Wei, Xiang Wang, Qi Li, Liqiang Nie, Yan Li, Xuanping Li, and Tat-Seng Chua. 2021. Contrastive learning for cold-start recommendation. In Proceedings of the 29th ACM International Conference on Multimedia . 5382-5390.
- [28] Felix Wu, Amauri Souza, Tianyi Zhang, Christopher Fifty, Tao Yu, and Kilian Weinberger. 2019. Simplifying graph convolutional networks. In International conference on machine learning . PMLR, 6861-6871.
- [29] Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie. 2021. Self-supervised graph learning for recommendation. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval . 726-735.
- [30] Shiwen Wu, Wentao Zhang, Fei Sun, and Bin Cui. 2020. Graph Neural Networks in Recommender Systems: A Survey. arXiv preprint arXiv:2011.02260 (2020).
- [31] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and S Yu Philip. 2020. A comprehensive survey on graph neural networks. IEEE Transactions on Neural Networks and Learning Systems (2020).
- [32] Xin Xia, Hongzhi Yin, Junliang Yu, Yingxia Shao, and Lizhen Cui. 2021. SelfSupervised Graph Co-Training for Session-based Recommendation. In Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management . 2180-2190.
- [33] Xin Xia, Hongzhi Yin, Junliang Yu, Qinyong Wang, Lizhen Cui, and Xiangliang Zhang. 2021. Self-Supervised Hypergraph Convolutional Networks for Sessionbased Recommendation. In Thirty-Fifth AAAI Conference on Artificial Intelligence . 4503-4511.
- [34] Xu Xie, Fei Sun, Zhaoyang Liu, Shiwen Wu, Jinyang Gao, Bolin Ding, and Bin Cui. 2020. Contrastive Learning for Sequential Recommendation. arXiv preprint arXiv:2010.14395 (2020).
- [35] Tiansheng Yao, Xinyang Yi, Derek Zhiyuan Cheng, Felix Yu, Ting Chen, Aditya Menon, Lichan Hong, Ed H Chi, Steve Tjoa, Jieqi Kang, et al. 2021. Self-supervised Learning for Large-scale Item Recommendations. In Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management . 4321-4330.
- [36] Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, and Yang Shen. 2020. Graph Contrastive Learning with Augmentations. Advances in Neural Information Processing Systems 33 (2020).
- [37] Junliang Yu, Min Gao, Jundong Li, Hongzhi Yin, and Huan Liu. 2018. Adaptive Implicit Friends Identification over Heterogeneous Network for Social Recommendation. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management . ACM, 357-366.
- [38] Junliang Yu, Min Gao, Hongzhi Yin, Jundong Li, Chongming Gao, and Qinyong Wang. 2019. Generating reliable friends via adversarial training to improve social recommendation. In 2019 IEEE International Conference on Data Mining (ICDM) . IEEE, 768-777.
- [39] Junliang Yu, Hongzhi Yin, Min Gao, Xin Xia, Xiangliang Zhang, and Nguyen Quoc Viet Hung. 2021. Socially-Aware Self-Supervised Tri-Training for Recommendation. In KDD '21: The 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining , Feida Zhu, Beng Chin Ooi, and Chunyan Miao (Eds.). ACM, 2084-2092.
- [40] Junliang Yu, Hongzhi Yin, Jundong Li, Min Gao, Zi Huang, and Lizhen Cui. 2020. Enhance Social Recommendation with Adversarial Graph Convolutional Networks. arXiv preprint arXiv:2004.02340 (2020).
- [41] Junliang Yu, Hongzhi Yin, Jundong Li, Qinyong Wang, Nguyen Quoc Viet Hung, and Xiangliang Zhang. 2021. Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation. In Proceedings of the Web Conference 2021 . 413-424.
- [42] Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Jundong Li, and Zi Huang. 2022. Self-Supervised Learning for Recommender Systems: A Survey. arXiv preprint arXiv:2203.15876 (2022).
- [43] Wenhui Yu and Zheng Qin. 2020. Graph convolutional network for recommendation with low-pass collaborative filters. In International Conference on Machine Learning . PMLR, 10936-10945.
- [44] Junwei Zhang, Min Gao, Junliang Yu, Lei Guo, Jundong Li, and Hongzhi Yin. 2021. Double-Scale Self-Supervised Hypergraph Learning for Group Recommendation. In Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management . 2557-2567.
- [45] Chang Zhou, Jianxin Ma, Jianwei Zhang, Jingren Zhou, and Hongxia Yang. 2021. Contrastive learning for debiased candidate generation in large-scale recommender systems. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &amp; Data Mining . 3985-3995.
- [46] Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. 2020. Sˆ 3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximization. arXiv preprint arXiv:2008.07873 (2020).
- [47] Xin Zhou, Aixin Sun, Yong Liu, Jie Zhang, and Chunyan Miao. 2021. SelfCF: A Simple Framework for Self-supervised Collaborative Filtering. arXiv preprint arXiv:2107.03019 (2021).