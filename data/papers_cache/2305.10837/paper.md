## Adaptive Graph Contrastive Learning for Recommendation

Yangqin Jiang

University of Hong Kong Hong Kong, China mrjiangyq99@gmail.com

## ABSTRACT

Graph neural networks (GNNs) have recently emerged as an effective collaborative filtering (CF) approaches for recommender systems. The key idea of GNN-based recommender systems is to recursively perform message passing along user-item interaction edges to refine encoded embeddings, relying on sufficient and highquality training data. However, user behavior data in practical recommendation scenarios is often noisy and exhibits skewed distribution. To address these issues, some recommendation approaches, such as SGL, leverage self-supervised learning to improve user representations. These approaches conduct self-supervised learning through creating contrastive views, but they depend on the tedious trial-and-error selection of augmentation methods. In this paper, we propose a novel Adaptive Graph Contrastive Learning (AdaGCL) framework that conducts data augmentation with two adaptive contrastive view generators to better empower the CF paradigm. Specifically, we use two trainable view generators - a graph generative model and a graph denoising model - to create adaptive contrastive views. With two adaptive contrastive views, AdaGCL introduces additional high-quality training signals into the CF paradigm, helping to alleviate data sparsity and noise issues. Extensive experiments on three real-world datasets demonstrate the superiority of our model over various state-of-the-art recommendation methods. Our model implementation codes are available at the link https://github.com/HKUDS/AdaGCL.

## CCS CONCEPTS

· Information systems → Recommender systems .

## KEYWORDS

Recommendation, Contrastive Learning, Data Augmentation

## ACMReference Format:

Yangqin Jiang, Chao Huang, and Lianghao Xia. 2023. Adaptive Graph Contrastive Learning for Recommendation. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23), August 6-10, 2023, Long Beach, CA, USA. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3580305.3599768

∗ Chao Huang is the corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

KDD '23, August 6-10, 2023, Long Beach, CA, USA

© 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 979-8-4007-0103-0/23/08...$15.00

https://doi.org/10.1145/3580305.3599768

Chao Huang ∗

University of Hong Kong Hong Kong, China Lianghao Xia University of Hong Kong Hong Kong, China

chaohuang75@gmail.com aka\_xia@foxmail.com

## 1 INTRODUCTION

Recommender systems are a crucial tool for web applications, helping users to navigate the overwhelming amount of information available online. These systems provide personalized recommendations of items that users might be interested in, such as products on online retail platforms [19, 27], posts on social networking sites [8, 35], and video sharing platforms [25, 34]. One of the most common approaches for generating these recommendations is collaborative filtering (CF), where the system uses the preferences of similar users or items to suggest new items for a given user [5, 29].

Collaborative filtering (CF) models have traditionally relied on matrix factorization (MF) to learn latent user and item embeddings from interaction data. However, with the rise of graph neural networks (GNNs), there has been a growing interest in using these models to propagate information along the user-item interaction graph and learn more sophisticated representations of user-item interactions. PinSage [33], NGCF [21], and LightGCN [4] are examples of GNN-based CF models that have shown promising results in personalized recommendations. These models use graph convolutional networks (GCNs) to propagate embeddings over the user-item interaction graph, allowing them to capture higher-order interactions between users and items that are not captured by other alternative CF models. In particular, PinSage and NGCF use multilayer GCNs to capture both local and global information about the user-item interaction graph, while LightGCN simplifies the message passing process by omitting the non-linear transformer and only using a simple weighted sum of the neighboring embeddings.

Graph-based collaborative filtering models have become increasingly popular in recommender systems. However, these models face challenges that current techniques have not adequately addressed. One such challenge is data noise, which can arise due to various factors, such as users clicking on irrelevant products due to over-recommendation of popular items. Directly aggregating information from all interaction edges in the user-item interaction graph can lead to inaccuracies in user representations, and multihop embedding propagation can worsen the noise effect. Therefore, existing graph-based CF models may not accurately capture user interests and generate inaccurate recommendations. Furthermore, the sparsity and skewed distribution of recommendation data can negatively impact effective user-item interaction modeling. As a result, current approaches may suffer from the problem of user data scarcity, where high-quality training signals may be limited.

Recently, some recommendation methods, such as SGL [26], SLRec [32] and HCCF [30], have leveraged self-supervised learning to improve user representations. These methods introduce additional supervision information by creating contrastive views through probability-based random masking or adding noise. However, these operations may keep some noisy interactions or drop important training signals during the data augmentation process, limiting the applicability and potential of contrastive learning.

Contribution. Given the limitations and challenges of existing solutions, we propose a novel Adaptive Graph Contrastive Learning (AdaGCL) framework to enhance the robustness and generalization performance of recommender systems. Our approach leverages adaptive contrastive learning to introduce high-quality training signals, empowering the graph neural CF paradigm. While several recent studies have used contrastive learning to improve model performance, they all require specific ways to create contrastive views. The selection of methods for creating contrastive views can be burdensome and often limited to a pool of prefabricated views, which can limit their potential and applicability. To address these issues, we integrate a graph generative model and a graph denoising model to establish views that adapt to the data distribution, achieving adaptive contrastive views for graph contrastive learning. By providing two different and adaptive views, we offer additional high-quality training signals that can enhance the graph neural CF paradigm and help address the problem of model collapse in contrastive learning-based data augmentation.

In summary, this paper makes the following contributions:

- Weproposeanovelself-supervised recommendation model, called AdaGCL, that enhances the robustness of the graph CF by distilling additional training signals from adaptive contrastive learning.
- AdaGCL employs two trainable view generators, namely a graph generator and a graph denoising model, to create contrastive views. These views address the problem of model collapse and enable adaptive views for contrastive learning, ultimately enhancing the effectiveness of the graph neural CF paradigm.
- Our experimental results demonstrate that our AdaGCL outperforms various baseline models on multiple datasets, highlighting its superior performance and effectiveness. Furthermore, our approach is able to address the challenges of data noise and user data scarcity, which can negatively impact the accuracy of collaborative filtering models for recommendation.

## 2 PRELIMINARIES AND RELATED WORK

## 2.1 Collaborative Filtering Paradigm

Welet U = { 𝑢 1 , · · · , 𝑢 𝑖 , · · · , 𝑢 𝐼 } ( |U| = 𝐼 ) and V = { 𝑣 1 , · · · , 𝑣 𝑗 , · · · , 𝑣 𝐽 } ( |V| = 𝐽 ) represent the set of users and items, respectively. The interaction matrix A ∈ R I×J indicates the implicit relationships between each user in U and his/her consumed items. Each entry A 𝑖,𝑗 in A will be set as 1 if user 𝑢 𝑖 has adopted item 𝑣 𝑗 before and A 𝑖,𝑗 = 0 otherwise. Upon the constructed interaction graph structures, the core component of graph-based CF paradigm lies in the information aggregation function, gathering the feature embeddings of neighboring users/items via different aggregators, e.g., mean or sum. The objective of CF task is to forecast the unobserved user-item interactions with the encoded corresponding representations. The assumption of the collaborative filtering paradigm is that users who exhibit similar behavior are more likely to share similar interests. One popular paradigm of existing collaborative filtering (CF) approaches involves using various embedding functions to generate vectorized representations of users and items. The similarity matching function is then introduced to estimate the relevance score between a user 𝑢 𝑖 and a candidate item 𝑣 𝑗 .

## 2.2 Graph-based Recommender Systems

Graph neural architectures have become increasingly popular in recent years due to their ability to effectively model complex relationships between users and items in recommendation systems [28]. These architectures leverage graph embedding propagation techniques to encode the interactions between users and items in the form of graph embeddings. One important advantage of graph neural architectures is their ability to capture multi-hop connections between users and items. This allows the model to capture more complex and nuanced relationships between users and items. Some architectures, like PinSage [33] and NGCF [21], use graph convolutional networks in the spectral domain. Others, like LightGCN[4], simplify the non-linear transformation and use sum-based pooling over neighboring representations for improved efficiency. These architectures encode each user and item into transformed embeddings while preserving multi-hop connections.

Moreover, fine-grained graph-based relational learning techniques among users and items have been introduced in graph neural networks for user/item representations. Examples of these techniques include DGCF [22], DCCF [13], and DRAN [24]. These techniques aim to learn disentangled or behavior-aware user representations by exploiting the graph-structured multi-intent information. In addition to these techniques, graph neural networks have also been increasingly used in next-item recommendation tasks to capture the temporal dependencies between items and how a user's preferences for certain items evolve over time. Models such as DGSR [37], RetaGNN [6], and GCE-GNN [23] represent the user's historical interactions as a sequence of items and use graph-based message passing to update each item's embedding based on the information from its neighbors. This approach allows the models to capture the dependencies and relationships between items and how a user's preferences for certain items evolve over time, leading to more accurate and effective recommendations.

## 2.3 Self-Supervised Graph Learning

Despite the success of supervised learning in many applications, obtaining a large labeled dataset can be a challenging and expensive task. To overcome this limitation, self-supervised learning (SSL) has emerged as a promising solution. In the context of graph machine learning, SSL has been shown to be effective for learning highquality representations of graph data. One of the recent advances in SSL is the use of contrastive learning with auxiliary training signals generated from various graph data, such as heterogeneous graph [7], spatio-temporal graph [38] and molecular graph [39]. SSL with contrastive learning has been shown to improve the quality of embeddings for graphs, leading to better performance on tasks, such as node classification and link prediction.

Self-supervised graph learning has also been introduced into recommender systems, to show great potential for enhancing representations of users and items with contrastive SSL [26] or generative SSL [11] techniques. One example of a self-supervised graph learning framework is SGL [26], which generates contrastive views of the user-item interaction graph using random node and edge dropout operations. By maximizing the agreement between the embeddings of the contrastive views, SSL signals can be incorporated into the model joint learning process. Another example is GFormer [11], which leverages the graph autoencoder to reconstruct the masked user-item interactions for augmentation. By generating augmented training data in this way, the model can learn more effective representations of users and items. Additionally, the use of self-supervised graph learning techniques has benefited a variety of recommendation scenarios. For example, S3-Rec [40] S3-Rec is based on a self-attentive neural architecture and uses four auxiliary self-supervised objectives to learn the correlations among various types of data, including attributes, items, subsequences. C2DSR [2] is a cross-domain sequential recommendation approach that proposes a contrastive cross-domain infomax objective to enhance the correlation between single- and cross-domain user representations. SLMRec [15] is a SSL approach for multimedia recommendation that captures multi-modal patterns in the data.

Figure 1: Overall framework of the proposed AdaGCL model.

<!-- image -->

## 3 METHODOLOGY

In this section, we introduce the AdaGCL framework, which is composed of three parts. The first part uses a graph message passing encoder to capture local collaborative relationships among users and items. The second part proposes a novel adaptive selfsupervised learning framework that includes two trainable view generators made of variational and denoising graph models. The third part introduces the phase of model optimization. The overall architecture of the AdaGCL model is illustrated in Figure 1.

## 3.1 Local Collaborative Relation Learning

To encode the interaction patterns between users and items, we follow the common collaborative filtering paradigm by embedding them into a 𝑑 -dimensional latent space. Specifically, we generate embedding vectors e 𝑖 and e 𝑗 of size R 𝑑 for user 𝑢 𝑖 and item 𝑣 𝑗 , respectively. We also define embedding matrices E ( 𝑢 ) ∈ R 𝐼 × 𝑑 and E ( 𝑣 ) ∈ R 𝐽 × 𝑑 to represent the embeddings of users and items, respectively. To propagate the embeddings, we design a local graph embedding propagation layer inspired by the simplified graph convolutional network used in LightGCN [4].

<!-- formula-not-decoded -->

Torepresent the aggregated information from neighboring items/users to the central node 𝑢 𝑖 and 𝑣 𝑗 , we use the vectors z 𝑖 ( 𝑢 ) and z 𝑗 ( 𝑣 ) respectively, both having a dimension of R 𝑑 . We derive the normalized adjacent matrix ¯ A ∈ R 𝐼 × 𝐽 from the user-item interaction matrix A . Specifically, ¯ A is calculated using the following formula:

<!-- formula-not-decoded -->

The diagonal degree matrices for users and items are D ( 𝑢 ) ∈ R 𝐼 × 𝐼 and D ( 𝑣 ) ∈ R 𝐽 × 𝐽 respectively. The neighboring items/users of user 𝑢 𝑖 and item 𝑣 𝑗 are denoted by N 𝑖 and N 𝑗 respectively.

To refine the user/item representations and aggregate local neighborhood information for contextual embeddings, we integrate multiple embedding propagation layers. We denote the embedding of user 𝑢 𝑖 and item 𝑣 𝑗 at the 𝑙 -th graph neural network (GNN) layer as e 𝑖, 𝑙 ( 𝑢 ) and e 𝑗, 𝑙 ( 𝑣 ) respectively. We formally define the message passing process from the ( 𝑙 -1)-th layer to the 𝑙 -th layer as follows:

<!-- formula-not-decoded -->

To obtain the embedding for a node, we sum its embeddings across all layers. The inner product between the final embedding of a user 𝑢 𝑖 and an item 𝑣 𝑗 is used to predict 𝑢 𝑖 's preference towards 𝑣 𝑗 :

<!-- formula-not-decoded -->

## 3.2 Adaptive View Generators for Graph Contrastive Learning

3.2.1 Dual-View GCL Paradigm. Existing graph contrastive learning (GCL) methods, such as those proposed in [12, 26, 30], generate views in specific ways, such as randomly dropping edges, nodes, or constructing hypergraphs. However, selecting an appropriate method for generating views can be burdensome, as it often relies on tedious trial-and-error or a limited pool of prefabricated views. This limitation can restrict the applicability and potential of these methods. To overcome this issue, we propose using two learnable view generators to obtain adaptive views for GCL.

Figure 2: Workflow of the graph denoising model.

<!-- image -->

Developing view generators for graph contrastive learning methods poses a challenge due to the risk of model collapse, where two views generated by the same generator share the same distribution, potentially leading to inaccurate contrastive optimization. To address this challenge, we propose using two distinct view generators that augment the user-item graph from different perspectives. Specifically, we employ a graph generative model and a graph denoising model as our two view generators. The graph generative model is responsible for reconstructing views based on graph distributions, while the graph denoising model leverages the graph's topological information to remove noise from the user-item graph and generate a new view with less noise.

In line with existing self-supervised collaborative filtering (CF) paradigms, such as those proposed in [26, 30], we use node selfdiscrimination to generate positive and negative pairs. Specifically, we treat the views of the same node as positive pairs (i.e., ( e ′ 𝑖 , e ′′ 𝑖 )| 𝑢 𝑖 ∈ U ), and the views of any two different nodes as negative pairs (i.e., ( e ′ 𝑖 , e ′′ 𝑖 ′ )| 𝑢 𝑖 , 𝑢 𝑖 ′ ∈ U , 𝑢 𝑖 ≠ 𝑢 𝑖 ′ ). Formally, the contrastive loss function that maximizes the agreement of positive pairs and minimizes that of negative pairs is as follows:

<!-- formula-not-decoded -->

To measure the similarity between two vectors, we use the cosine similarity function denoted by 𝑠 (·) , with the hyper-parameter 𝜏 known as the temperature in softmax. We compute the contrastive loss for the item side as L 𝑖𝑡𝑒𝑚 𝑠𝑠𝑙 in a similar way. By combining these two losses, we obtain the objective function for the self-supervised task, which is denoted by L 𝑠𝑠𝑙 = L 𝑢𝑠𝑒𝑟 𝑠𝑠𝑙 + L 𝑖𝑡𝑒𝑚 𝑠𝑠𝑙 .

3.2.2 Graph Generative Model as View Generator. The recent emergence of learning-based graph generative models [9, 18] provides a promising solution for view generator. In this study, we adopt the widely-used Variational Graph Auto-Encoder (VGAE) [9] as the generative model, which combines the concept of variational auto-encoder with graph generation. Compared to GAE, VGAE incorporates KL divergence to reduce the risk of overfitting, allowing for more diverse graphs to be generated by increasing the uncertainty. This feature provides a more challenging contrastive view for contrastive learning. Additionally, VGAE is relatively easier to train and faster than other currently popular generation models such as generative adversarial networks and diffusion models.

As illustrated in Fig. 1, we utilize a multi-layer GCN as the encoder to obtain the graph embeddings. Two MLPs are utilized to derive the mean value and the standard deviation of the graph embedding, respectively. With another MLP as the decoder, the input mean value and the standard deviation with Gaussian noise will be decoded to generate a new graph. The loss of VGAE is defined:

<!-- formula-not-decoded -->

The term L 𝑘𝑙 refers to the Kullback-Leibler divergence (KL divergence) between the distribution of node embeddings and the standard Gaussian distribution. On the other hand, L 𝑑𝑖𝑠 is a crossentropy loss that quantifies the dissimilarities between the generated graph and the original graph.

3.2.3 Graph Denoising Model as View Generator. GNN models use message passing mechanisms to propagate and aggregate information along the input graph to learn node representations. However, the quality of the input graph can heavily impact model performance since messages aggregated along noisy edges can decrease the quality of node embeddings. Therefore, for the second view generator, we aim to generate a denoising view that can enhance model performance against noisy data.

To improve the quality of node embeddings obtained after each layer of GCN, we propose a graph neural network that incorporates a denoising layer to filter out noisy edges in the input graph. This parameterized network is shown in Fig. 2. The main concept behind our approach is to actively filter out noisy edges in the input graph using a parameterized network. For the 𝑙 -th GCN layer, we use a binary matrix M 𝑙 ∈ 0 , 1 | V | × | V | , where 𝑚 𝑙 𝑖,𝑗 denotes whether the edge between node 𝑢 𝑖 and 𝑣 𝑗 is present (0 indicates a noisy edge).

Formally, the adjacency matrix of the resulting subgraph is A 𝑙 = A ⊙ M 𝑙 , where ⊙ is the element-wise product. The straightforward idea to reduce noisy edges with the least assumptions about A 𝑙 is to penalize the number of non-zero entries in M 𝑙 of different layers.

<!-- formula-not-decoded -->

where I [·] is an indicator function, with I [ 𝑇𝑟𝑢𝑒 ] = 1 and I [ 𝐹𝑎𝑙𝑠𝑒 ] = 0, | | · | | 0 represents the 𝑙 0 norm. However, because of its combinatorial and non-differentiability nature, optimizing this penalty is computationally intractable. Therefore, we consider each binary number 𝑚 𝑙 𝑖,𝑗 to be drawn from a Bernoulli distribution parameterized by 𝜋 𝑙 𝑖,𝑗 , i.e., 𝑚 𝑙 𝑖,𝑗 ∼ Bern ( 𝜋 𝑙 𝑖,𝑗 ) . Here, 𝜋 𝑙 𝑖,𝑗 describes the quality of the edge ( 𝑢 , 𝑣 ). To efficiently optimize subgraphs with gradient methods, we adopt the reparameterization trick and relax the binary entries 𝑚 𝑙 𝑖,𝑗 from being drawn from a Bernoulli distribution to a deterministic function 𝑔 of parameters 𝛼 𝑙 𝑖,𝑗 ∈ R and an independent random variable 𝜀 𝑙 . That is 𝑚 𝑙 𝑖,𝑗 = 𝑔 ( 𝛼 𝑙 𝑖,𝑗 , 𝜀 𝑙 ) .

Based on above operations, we design a denoising layer to learn the parameter 𝛼 𝑙 𝑖,𝑗 that controls whether to remove the edge ( 𝑢 , 𝑣 ). For the 𝑙 -th GNN layer, we calculate 𝛼 𝑙 𝑖,𝑗 for user node 𝑢 and its interacted item node 𝑣 with 𝛼 𝑙 𝑖,𝑗 = 𝑓 𝑙 𝜃 𝑙 ( e 𝑙 𝑖 , e 𝑙 𝑗 ) , where 𝑓 𝑙 𝜃 𝑙 is an MLP parameterized by 𝜃 𝑙 . In order to get 𝑚 𝑙 𝑖,𝑗 , we also utilize the concrete distribution along with a hard sigmoid function. Within the above formulation, the constraint on the number of non-zero entries in M 𝑙 in Eq. (7) can be reformulated with:

<!-- formula-not-decoded -->

where P 𝜎 ( 𝑠 𝑙 𝑖,𝑗 ) is the cumulative distribution function (CDF) of 𝜎 ( 𝑠 𝑙 𝑖,𝑗 ) , 𝜎 (·) extends the range of 𝑠 𝑙 𝑖,𝑗 , and 𝑠 𝑙 𝑖,𝑗 is drawn from a binary concrete distribution with 𝛼 𝑙 𝑖,𝑗 parameterizing the location.

## 3.3 Learning Task-aware View Generators

Although two view generators could learn to generate better views from different aspects, there may be no optimization signals to adjust generated views to the main CF task. The straightforward idea is introducing commonly-used BPR loss, as follows:

<!-- formula-not-decoded -->

The training data is represented by O = ( 𝑢, 𝑖, 𝑗 )|( 𝑢, 𝑖 ) ∈ O + , ( 𝑢, 𝑗 ) ∈ O , where O + denotes the observed interactions and O -= U×I/O + denotes the unobserved interactions.

To train the graph generative model, we use the node embeddings encoded by the VGAE encoder to compute BPR loss. The loss function L 𝑔𝑒𝑛 is then updated as follows:

<!-- formula-not-decoded -->

where Θ is the set of model parameters, while 𝜆 2 is a hyperparameter used to control the strength of the weight-decay regularization.

To train the graph denoising model, we use the node embeddings obtained by the denoising neural network to compute the BPR loss. The loss function L 𝑑𝑒𝑛 is updated as follows:

<!-- formula-not-decoded -->

## 3.4 Model Training

The training of our proposed model consists of two parts. In the upper-level part, we adopt a multi-task training strategy to jointly optimize the classic recommendation task (Eq. (9)) and the selfsupervised learning task (Eq. (5)):

<!-- formula-not-decoded -->

where Θ refers to the set of model parameters in the main task, which in this work, is the set of parameters of LightGCN. Additionally, 𝜆 1 and 𝜆 2 are hyperparameters that control the strengths of SSL and 𝐿 2 regularization, respectively.

The lower-level part of the training involves optimizing the generative and denoising view generators based on Eq. (10) and Eq. (11), which is formally presented as follows:

<!-- formula-not-decoded -->

Table 1: Statistics of the experimental datasets.

| Dataset      |   User # |   Item # |   Interaction # | Density        |
|--------------|----------|----------|-----------------|----------------|
| Last.FM      |    1,892 |   17,632 |          92,834 | 2 . 8 × 10 - 3 |
| Yelp         |   42,712 |   26,822 |         182,357 | 1 . 6 × 10 - 4 |
| BeerAdvocate |   10,456 |   13,845 |       1,381,094 | 9 . 5 × 10 - 3 |

## 3.5 Time Complexity Analysis

Weanalyze the time complexity of our proposed model by considering its three key components. Firstly, the local collaborative relation learning module takes 𝑂 ( 𝐿 ×|A|× 𝑑 ) time, which is the same as that of LightGCN. Here, 𝐿 denotes the number of graph neural layers, |A| is the number of edges in the user-item interaction graph, and 𝑑 denotes the embedding dimensionality. Secondly, the graph generative model (VGAE) costs 𝑂 (|A| × 𝑑 2 ) time. Thirdly, the denoising layers in the graph denoising model cost 𝑂 ( 𝐿 × |A| × 𝑑 2 ) time. Finally, the contrastive learning paradigm costs 𝑂 ( 𝐿 × 𝐵 ×( 𝐼 + 𝐽 ) × 𝑑 ) , where 𝐵 denotes the number of users/items included in a single batch. 𝐼 and 𝐽 denote the number of users and items, respectively.

## 4 EVALUATION

To evaluate the effectiveness of our proposed model, our experiments are designed to answer the following research questions:

- RQ1 : What is the performance of our proposed model compared to various state-of-the-art recommender systems?
- RQ2 : How do the key components of our proposed model contribute to its overall performance on different datasets?
- RQ3 : How well can our proposed model handle noisy and sparse data compared to baseline methods?
- RQ4 : How do the key hyperparameters influence the performance of our proposed model framework?

## 4.1 Experimental Settings

4.1.1 Evaluation Datasets. We conduct experiments on three datasets collected from online applications, Last.FM, Yelp, and BeerAdvocate. The statistics of these datasets are shown in Table 1.

- Last.FM : This dataset contains social networking, tagging, and music artist listening information collected from a set of users from the Last.fm online music system.
- Yelp : This commonly-used dataset contains user ratings on business venues collected from the Yelp platform. It is a valuable resource for studying user preferences and behavior in the context of personalized venue recommendations.
- BeerAdvocate : This dataset contains beer reviews from BeerAdvocate. We process it using the 10-core setting by keeping only users and items with at least 10 interactions.

4.1.2 Evaluation Protocols. We follow the recent collaborative filtering models [4, 26] and split the datasets by 7:2:1 into training, validation, and testing sets. We adopt the all-rank evaluation protocol, where for each test user, the positive items in the test set and all the non-interacted items were tested and ranked together. We employ the commonly-used Recall@N and Normalized Discounted Cumulative Gain (NDCG)@N as evaluation metrics for recommendation performance evaluation. We set N to 20 by default.

Table 2: Performance comparison on Last.FM, Yelp, BeerAdvocate datasets in terms of Recall and NDCG .

| Dataset      | Metric            | BiasMF        | NCF           | AutoR         | PinSage       | STGCN         | GCMC          | NGCF          | GCCF          | LightGCN      | SLRec         | NCL           | SGL           | HCCF          | SHT           | DirectAU      | Ours          | p-val.              |
|--------------|-------------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------------|
| Last.FM      | Recall@20 NDCG@20 | 0.1879 0.1362 | 0.1130 0.0795 | 0.1518 0.1114 | 0.1690 0.1228 | 0.2067 0.1528 | 0.2218 0.1558 | 0.2081 0.1474 | 0.2222 0.1642 | 0.2349 0.1704 | 0.1957 0.1442 | 0.2353 0.1715 | 0.2427 0.1761 | 0.2410 0.1773 | 0.2420 0.1770 | 0.2422 0.1727 | 0.2603 0.1911 | 2.1 𝑒 - 5 9.5 𝑒 - 5 |
| Last.FM      | Recall@40 NDCG@40 | 0.2660 0.1653 | 0.1693 0.0952 | 0.2174 0.1336 | 0.2402 0.1472 | 0.2940 0.1821 | 0.3149 0.1897 | 0.2944 0.1829 | 0.3083 0.1931 | 0.3220 0.2022 | 0.2792 0.1737 | 0.3252 0.2033 | 0.3405 0.2104 | 0.3232 0.2051 | 0.3235 0.2055 | 0.3356 0.2042 | 0.3531 0.2204 | 6.9 𝑒 - 5 5.6 𝑒 - 4 |
| Yelp         | Recall@20 NDCG@20 | 0.0532 0.0264 | 0.0304 0.0143 | 0.0491 0.0222 | 0.0510 0.0245 | 0.0562 0.0282 | 0.0584 0.0280 | 0.0681 0.0336 | 0.0742 0.0365 | 0.0761 0.0373 | 0.0665 0.0327 | 0.0806 0.0402 | 0.0803 0.0398 | 0.0789 0.0391 | 0.0794 0.0395 | 0.0818 0.0424 | 0.0873 0.0439 | 1.5 𝑒 - 6 1.8 𝑒 - 8 |
| Yelp         | Recall@40 NDCG@40 | 0.0802 0.0321 | 0.0487 0.0187 | 0.0692 0.0268 | 0.0743 0.0315 | 0.0856 0.0355 | 0.0891 0.0360 | 0.1019 0.0419 | 0.1151 0.0466 | 0.1175 0.0474 | 0.1032 0.0418 | 0.1230 0.0505 | 0.1226 0.0502 | 0.1210 0.0492 | 0.1217 0.0497 | 0.1226 0.0524 | 0.1315 0.0548 | 3.2 𝑒 - 6 2.7 𝑒 - 7 |
| BeerAdvocate | Recall@20 NDCG@20 | 0.0996 0.0856 | 0.0729 0.0654 | 0.0816 0.0650 | 0.0930 0.0816 | 0.1003 0.0852 | 0.1082 0.0901 | 0.1033 0.0873 | 0.1035 0.0901 | 0.1102 0.0943 | 0.1048 0.0881 | 0.1131 0.0971 | 0.1138 0.0959 | 0.1156 0.0990 | 0.1150 0.0977 | 0.1182 0.0981 | 0.1216 0.1015 | 7.7 𝑒 - 6 4.9 𝑒 - 3 |
| BeerAdvocate | Recall@40 NDCG@40 | 0.1602 0.1016 | 0.1203 0.0754 | 0.1325 0.0794 | 0.1553 0.0980 | 0.1650 0.1031 | 0.1766 0.1085 | 0.1653 0.1032 | 0.1662 0.1062 | 0.1757 0.1113 | 0.1723 0.1068 | 0.1819 0.1150 | 0.1776 0.1122 | 0.1847 0.1176 | 0.1799 0.1156 | 0.1797 0.1139 | 0.1867 0.1182 | 1.3 𝑒 - 2 2.4 𝑒 - 1 |

4.1.3 Compared Baseline Methods. We evaluate our proposed AdaGCL by comparing it with various baselines for comprehensive evaluation. The details of the baselines are as follows.

- BiasMF [10]: It is a matrix factorization method that aims to enhance user-specific preferences for recommendation by incorporating bias vectors for users and items.
- NCF [5]: It is a neural network-based method that replaces the dot-product operation in conventional matrix factorization with multi-layer neural networks. This allows the model to capture complex user-item interactions and provide recommendations. For our comparison, we utilize the NeuMF variant of NCF.
- AutoR [14]: It is a method that improves the user/item representations by using a three-layer autoencoder trained under the supervision of an interaction reconstruction task.
- GCMC [1]: This work utilizes graph convolutional networks (GCNs) for interaction matrix completion.
- PinSage [33]: It is a graph convolutional-based method that employs random sampling in the graph convolutional framework to enhance the collaborative filtering task.
- NGCF [21]: It uses a multi-layer graph convolutional network to propagate information through the user-item interaction graph and learn the latent representations of users and items.
- STGCN [36]: It combines graph convolutional encoders with graph autoencoders to enhance the model's robustness against sparse and cold-start samples in collaborative filtering tasks.
- LightGCN [4]: This model leverages the power of neighborhood information in the user-item interaction graph by using a layer-wise propagation scheme that involves only linear transformations and element-wise additions.
- GCCF [3]: It presents a new approach to collaborative filtering recommender systems by revisiting graph convolutional networks. It removes non-linear activations and introduces a residual network structure that alleviates the over-smoothing problem.
- HCCF [30]: A new self-supervised recommendation framework is proposed in this work, which is able to capture both local and global collaborative relations using a hypergraph neural networks enhanced by cross-view contrastive learning architecture.
- SHT [31]: It integrates hypergraph neural networks and transformer under a self-supervised learning paradigm for data augmentation to denoise user-item interactions in recommendation.
- SLRec [32]: It integrates contrastive learning between node features as regularization terms in order to improve the efficacy of current collaborative filtering recommender systems.
- SGL [26]: The model augments LightGCN with self-supervised contrastive learning by conducting data augmentation through random walk and node/edge dropout to corrupt graph structures.
- NCL [12]: This is a neighborhood-enriched contrastive learning approach that enhances graph collaborative filtering by incorporating potential neighbors into contrastive pairs. NCL introduces structural and semantic neighbors of a user or item, developing a structure-contrastive and a prototype-contrastive objective.
- DirectAU [17]: This new approach proposes a new learning objective for collaborative filtering methods that measures the representation quality based on alignment and uniformity on the hypersphere. It directly optimizes these two properties to improve recommendation performance.

## 4.2 Overall Performance Comparison (RQ1)

The effectiveness of the proposed AdaGCL is validated through an overall performance evaluation on three datasets, comparing it with various baselines. To ensure statistical significance, the authors retrained AdaGCL and the best-performing baseline five times and computed p-values. The results are presented in Table 2.

- The evaluation results indicate that AdaGCL outperforms the baselines under both top20 and top40 settings, and the t-tests validate the significance of the observed performance improvements. The superior performance of AdaGCL can be attributed to the effectiveness of the proposed contrastive learning frameworks for data augmentation over user-item interactions. The use of adaptive view generators ensures that informative and diverse contrastive views are generated. This, in turn, leads to more effective learning of user and item embeddings, resulting in better recommendations. Overall, these findings demonstrate the effectiveness of the proposed contrastive learning approach for collorative filtering and highlight the importance of designing effective data augmentation techniques for this task.
- The evaluation results demonstrate that self-supervised learning improves existing CF frameworks, such as SLRec, SGL, and NCL. This improvement can be attributed to incorporating an augmented learning task, which provides beneficial regularization based on the input data. For example, SLRec and SGL use stochastic data augmentation to generate multiple views, while NCL incorporates potential neighbors into contrastive pairs. However,

Table 3: Ablation study on key components of AdaGCL.

| Category   | Data     | Last.FM   | Last.FM   | Yelp   | Yelp   | BeerAdvocate   | BeerAdvocate   |
|------------|----------|-----------|-----------|--------|--------|----------------|----------------|
| Category   | Variants | Recall    | NDCG      | Recall | NDCG   | Recall         | NDCG           |
| Adaptive   | w/o Task | 0.2562    | 0.1868    | 0.0849 | 0.0425 | 0.1212         | 0.1010         |
| Adaptive   | Gen+Gen  | 0.2494    | 0.1819    | 0.0853 | 0.0429 | 0.1187         | 0.0992         |
| Random     | EdgeD    | 0.2476    | 0.1794    | 0.0852 | 0.0424 | 0.1163         | 0.0964         |
| AdaGCL     | AdaGCL   | 0.2603    | 0.1911    | 0.0873 | 0.0439 | 0.1216         | 0.1015         |

these methods may lose useful signals that reflect important useritem interaction patterns. In contrast, AdaGCL has two main advantages. First, it does not rely on random data augmentation to generate contrastive views, instead using two adaptive view generators to create reasonable views that retain useful information. The generative view captures the key patterns of the original data, while the denoising generator filters out noise signals that may interfere with the contrastive learning process. Second, AdaGCL addresses the problem of model collapse in contrastive learning by creating contrastive views from different aspects with two different generators. The generative and denoising views capture different aspects of the input data, ensuring that the learned representations are diverse and informative. The superior performance of AdaGCL compared to the baseline selfsupervised approaches validates the effectiveness of this new self-supervised learning paradigm for CF.

## 4.3 Model Ablation Test (RQ2)

We conducted extensive experiments to validate the effectiveness of the proposed methods by removing three applied techniques in AdaGCL individually: the adaptive view generators, the task-aware optimization for view generators, and the denoising view generator.

To evaluate the efficacy of the proposed generative and denoising generators for view generation, we compare them to existing random augmentation method. Specifically, an ablated version of AdaGCL is trained using the random edge drop augmentation ( EdgeD ). Additionally, we replace the denoising view generator with an identical VGAE-based generator ( Gen+Gen ), to study the importance of denoising in the view generation process. Furthermore, we replace the task-aware optimization with the original reconstruction objective ( w/o Task ), to investigate the necessity of introducing task-relevant information into model training. The variants are retrained and tested on the three datasets. The results are presented in Table 3, from which we draw the following major conclusions:

- Advantage of adaptive view generators . The results presented in Table 3 demonstrate that using the random-permutation-based contrastive view generator ( EdgeD ) leads to a significant decay in performance compared to the proposed AdaGCL approach. This suggests that random augmentation methods may not be sufficient for generating informative contrastive views in CF. In contrast, the adaptive learning ability of the generative view based on VGAE and the denoising ability of the explicit denoising network in AdaGCL are critical for achieving superior performance. The generative view preserves the key patterns of the original data by modeling the graph-based user-item interaction structures, while the denoising network filters out noise signals that may interfere with the contrastive learning process.
- Benefit of denoising view generator . We conduct additional tests on a modified version of our model to further study the effectiveness of our designed adaptive view generators. Specifically, we remove the denoising view generator (referred to as the Gen+Gen variant). The results show that, while the VGAEbased view provide adaptive data augmentations that benefit contrastive learning, it is not enough to eliminate the inherent data noise. Our AdaGCL addresses this issue by incorporating the denoising view into the contrastive learning process, resulting in significant performance improvements.
- Effectiveness of the task-aware optimization . The results show that the w/o Task variant performs worse than the proposed AdaGCL on all three datasets. This suggests that using a general-purpose auto-encoding loss and denoising loss for contrastive view generator training may not be sufficient for achieving optimal performance in CF. Instead, introducing BPR loss for task-aware view generator training leads to better performance. This highlights the importance of incorporating task-aware information to guide the training of view generators, which can help to capture more relevant user-item interaction patterns and improve the quality of the generated contrastive views.

## 4.4 Model Robustness Test (RQ3)

In this section, our experiments show that our proposed approach, AdaGCL, exhibits superior robustness against data noise, and is effective in handling sparse user-item interaction data.

4.4.1 Performance w.r.t. Data Noise Degree. We investigate the robustness of our approach, AdaGCL, against data noise in recommendation systems. To evaluate the impact of noise on our model's performance, we randomly replace different percentages of real edges with fake edges and retrain the model using the corrupted graphs as input. Concretely, we replace 5%, 10%, 15%, 20%, and 25% of the interaction edges with fake edges in our experiments. We compare AdaGCL's performance with two other models, LightGCN and SGL. To better understand the effect of noise on performance degradation, we evaluate the relative performance compared to the performance on the original data, and present the results in Fig. 3. Our observations indicate that AdaGCL exhibits smaller performance degradation in most cases compared to the baselines.

We attribute this observation to two reasons: First, the selfsupervised learning task employed by AdaGCL distills information from two adaptive contrastive views to refine the graph embeddings. This observation is supported by the stronger robustness of the self-supervised method SGL compared to LightGCN. Second, both view generators used in our approach are capable of generating a contrastive view with less noise and more task-related information. Additionally, we find that the relative performance degradation on the Yelp dataset is more apparent compared to the other two datasets. This finding is because noisy data has a larger influence on the performance of models on sparse datasets like Yelp, which is the sparest dataset in our experiments. Overall, our results suggest that AdaGCL is a robust and effective model for recommendation systems, even in the presence of data noise.

4.4.2 Performance w.r.t. Data Sparsity. We also investigate the influence of data sparsity on model performance from both user

<!-- image -->

(c) BeerAdvocate data

Figure 3: Relative performance degradation w.r.t. noise ratio. We introduce varying levels of noise by replacing 5%, 10%, 15%, 20%, and 25% of the interaction edges with fake edges.

<!-- image -->

(b) Performance w.r.t. item interaction numbers

Figure 4: Performance w.r.t different sparsity degrees of interaction data for users and items, respectively, on Yelp dataset. We divide users and items into several groups based on the number of interactions they had in the dataset.

and item sides. We compare our proposed AdaGCL with LightGCN and SGL in this experiment. Multiple user and item groups are constructed based on their number of interactions in the training set, with the first group in the user-side experiments containing users interacting with 0-10 items and the first group in the item-side experiments containing items interacting with 0-5 users.

Figure 5: Hyperparameter Analysis on Last.FM and Yelp.

<!-- image -->

Fig. 4 illustrates the recommendation accuracy for our AdaGCL and the two compared methods. Our findings highlight the following: First, AdaGCL exhibits consistently superior performance on datasets with different sparsity degrees, indicating its robustness in handling sparse data for both users and items. We attribute this advantage to our adaptive contrastive view pair, which provides highquality self-supervised signals that mitigate the negative effects of data sparsity. Second, the sparsity of item interaction vectors has a more significant influence on model performance across all the methods. Overall, our experiments demonstrate the effectiveness of AdaGCL in handling sparse user-item interaction data.

## 4.5 Hyperparameter Analysis (RQ4)

In this section, the authors investigate the sensitivity of their proposed model to the key hyperparameter 𝜆 1 for InfoNCE loss, which controls the strength of contrastive learning. Specifically, the weight 𝜆 1 is searched in the range of (1, 1 𝑒 -1 , 1 𝑒 -2 , 1 𝑒 -3 , 1 𝑒 -4 ) to explore its impact on the model's performance. The results are presented in Figure 5, which shows the model's performance on the Last.FM and Yelp datasets with different values of 𝜆 1. It is observed that the best performance is achieved with 𝜆 1 = 1 𝑒 -1 and 𝜆 1 = 1. This suggests that a large value of 𝜆 1 may overly emphasize the contrastive optimization loss.

## 4.6 Embedding Visualisation Analysis

In this section, we conduct an embedding visualization analysis with the representations encoded from our proposed approach, AdaGCL, and the baseline SGL to gain insight into the benefits of our model. As previously mentioned, SGL uses random data augmentation methods to create contrastive views, which can result in poor performance when dealing with noisy data. The added noises may unintentionally cause damage to contrastive views. Furthermore, SGL employs the same data augmentation methods on both contrastive views, leading to the issue of model collapse since the two views can easily have a similar distribution.

To validate the effectiveness of our method in addressing these limitations, we visualize the embeddings of the two contrastive views given by AdaGCL and SGL. We randomly sample 2,000 nodes from the Yelp dataset and map their embeddings in the three views (i.e., one main view and two contrastive views) to the 2-D space with t-SNE [16]. We employ the KMeans algorithm to cluster the nodes based on their compressed 2-D embeddings and color them with different colors. To highlight the impact of noisy data on SGL and AdaGCL, we also visualize the embeddings of polluted data, where 25% of the edges are replaced with fake edges. The visualization results are shown in Fig.6 and Fig.7, respectively. Note that in Fig.6, View 1 and View 2 are generated by the graph generative model and the graph denoising model, respectively.

Figure 7: View embedding visualization for SGL.

<!-- image -->

4.6.1 Effectiveness of Adaptive View Generators. As shown in Fig.7(a), SGL learns a large cloud of evenly-distanced embeddings with a few clear community structures to capture the collaborative relations among nodes. This is because random edge dropping tends to generate contrastive views with uniform distributions, as shown in Fig.7(b) and Fig. 7(c). Furthermore, SGL's two contrastive views show more similar distributions compared to our method. In contrast, our AdaGCL is based on two adaptive view generators that can generate more informative and diverse views of the data. By adaptively adjusting the views, our method is able to capture more complex and nuanced structures of the graph, resulting in more distinct embeddings with better clustering effects.

Furthermore, our AdaGCL demonstrates better robustness when dealing with noisy data compared to SGL. The visualization results of the three views of SGL ( i.e. , Fig.7(d), Fig.7(e), and Fig. 7(f)) show severe over-uniform distributions. When dealing with noisy data, SGLcan produce embeddings with uniform distributions, which can result in a loss of unique collaborative patterns in the embeddings and negatively impact the performance of the method. In contrast, according to the visual results of the three views of our AdaGCL ( i.e. , Fig.6(d), Fig.6(e), and Fig. 6(f)), our method is more robust to noisy data. This is because our method adaptively adjusts the views to capture the most informative and discriminative aspects of the data, and is therefore less affected by noise in the input.

4.6.2 Effectiveness of the Graph Denoising Model. To improve the robustness of our AdaGCL to noise and avoid the issue of model collapse, we design a graph denoising module as the second view generator. As shown in Fig.6(c) and Fig.6(f), the contrastive view created by the graph denoising component is less affected by noise compared to the other view pair ( i.e. , Fig.6(b) and Fig.6(e)). This is because the denoised graph contains more informative and discriminative signals about the graph structure corresponding to complex user-item interaction patterns, making it more robust to noise. Moreover, our graph denoising model creates better denoised views compared to the contrastive views in SGL ( i.e. , Fig.7(e) and Fig.7(f)), validating its effectiveness in graph denoising. By incorporating the denoised graph as augmented view, the captured more informative signals resulting in more robust user representations.

## 5 CONCLUSION

In this work, we propose a novel approach to improving contrastive recommender systems through the use of adaptive view generators. Specifically, we introduce a new recommendation framework, AdaGCL, which utilizes a graph generative model and a graph denoising model to create contrastive views, allowing for more effective user-item interaction modeling with self-augmented supervision signals. Our framework demonstrates improved robustness against noise perturbation, thereby enhancing the overall performance of graph-based recommender systems. Through extensive experimentation on multiple datasets, we have shown that our proposed AdaGCL, outperforms several competitive baselines, providing validation for its superiority in contrastive recommenders.

Moving forward, an important area of research would be to extend our framework to explore casual factors for contrastive self-supervised learning signals in recommender systems. This involves leveraging causal inference techniques [20] to improve the interpretability of the self-supervised learning signals used in contrastive learning. By accounting for the underlying causal relationships between user behaviors, we can design more effective and informative self-supervised learning objectives that better capture the nuances of user-item interactions. Additionally, we may investigate the transferability of our model by exploring transfer learning techniques, such as domain adaptation and multi-task learning.

## REFERENCES

- [1] Rianne van den Berg, Thomas N Kipf, and Max Welling. 2017. Graph convolutional matrix completion. arXiv preprint arXiv:1706.02263 (2017).
- [2] Jiangxia Cao, Xin Cong, Jiawei Sheng, Tingwen Liu, and Bin Wang. 2022. Contrastive Cross-Domain Sequential Recommendation. In International Conference on Information &amp; Knowledge Management (CIKM) . 138-147.
- [3] Lei Chen, Le Wu, Richang Hong, Kun Zhang, and Meng Wang. 2020. Revisiting graph based collaborative filtering: A linear residual graph convolutional network approach. In AAAI Conference on Artificial Intelligence (AAAI) , Vol. 34. 27-34.
- [4] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for recommendation. In International Conference on Research and Development in Information Retrieval (SIGIR) . 639-648.
- [5] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. In The Web Conference (WWW) . 173-182.
- [6] Cheng Hsu and Cheng-Te Li. 2021. Retagnn: Relational temporal attentive graph neural networks for holistic sequential recommendation. In The Web Conference (WWW) . 2968-2979.
- [7] Dasol Hwang, Jinyoung Park, Sunyoung Kwon, KyungMin Kim, Jung-Woo Ha, and Hyunwoo J Kim. 2020. Self-supervised auxiliary learning with meta-paths for heterogeneous graphs. In Advances in Neural Information Processing Systems (NeurIPS) . 10294-10305.
- [8] Mohsen Jamali and Martin Ester. 2010. A matrix factorization technique with trust propagation for recommendation in social networks. In ACM Conference on Recommender Systems (Recsys) . 135-142.
- [9] Thomas N Kipf and Max Welling. 2016. Variational graph auto-encoders. arXiv preprint arXiv:1611.07308 (2016).
- [10] Yehuda Koren, Robert Bell, and Chris Volinsky. 2009. Matrix factorization techniques for recommender systems. Computer 42, 8 (2009), 30-37.
- [11] Chaoliu Li, Lianghao Xia, Xubin Ren, Yaowen Ye, Yong Xu, and Chao Huang. 2023. Graph Transformer for Recommendation. arXiv preprint arXiv:2306.02330 (2023).
- [12] Zihan Lin, Changxin Tian, Yupeng Hou, and Wayne Xin Zhao. 2022. Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning. In The Web Conference (WWW) . 2320-2329.
- [13] Xubin Ren, Lianghao Xia, Jiashu Zhao, Dawei Yin, and Chao Huang. 2023. Disentangled Contrastive Collaborative Filtering. arXiv preprint arXiv:2305.02759 (2023).
- [14] Suvash Sedhain, Aditya Krishna Menon, Scott Sanner, and Lexing Xie. 2015. Autorec: Autoencoders meet collaborative filtering. In The Web Conference (WWW) . 111-112.
- [15] Zhulin Tao, Xiaohao Liu, Yewei Xia, Xiang Wang, Lifang Yang, Xianglin Huang, and Tat-Seng Chua. 2022. Self-supervised learning for multimedia recommendation. Transactions on Multimedia (TMM) (2022).
- [16] Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE. Journal of machine learning research 9, 11 (2008).
- [17] Chenyang Wang, Yuanqing Yu, Weizhi Ma, Min Zhang, Chong Chen, Yiqun Liu, and Shaoping Ma. 2022. Towards Representation Alignment and Uniformity in Collaborative Filtering. In International Conference on Knowledge Discovery and Data Mining (KDD) . 1816-1825.
- [18] Hongwei Wang, Jialin Wang, Jia Wang, Miao Zhao, Weinan Zhang, Fuzheng Zhang, Wenjie Li, Xing Xie, and Minyi Guo. 2019. Learning graph representation with generative adversarial nets. Transactions on Knowledge and Data Engineering (TKDE) 33, 8 (2019), 3090-3103.
- [19] Jianling Wang, Raphael Louca, Diane Hu, Caitlin Cellier, James Caverlee, and Liangjie Hong. 2020. Time to Shop for Valentine's Day: Shopping Occasions and Sequential Recommendation in E-commerce. In International Conference on Web Search and Data Mining (WSDM) . 645-653.
- [20] Wenjie Wang, Xinyu Lin, Fuli Feng, Xiangnan He, Min Lin, and Tat-Seng Chua. 2022. Causal representation learning for out-of-distribution recommendation. In The Web Conference (WWW) . 3562-3571.
- [21] Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. 2019. Neural graph collaborative filtering. In International Conference on Research and

Development in Information Retrieval (SIGIR) . 165-174.

- [22] Xiang Wang, Hongye Jin, An Zhang, Xiangnan He, Tong Xu, and Tat-Seng Chua. 2020. Disentangled graph collaborative filtering. In International Conference on Research and Development in Information Retrieval (SIGIR) . 1001-1010.
- [23] Ziyang Wang, Wei Wei, Gao Cong, Xiao-Li Li, Xian-Ling Mao, and Minghui Qiu. 2020. Global context enhanced graph neural networks for session-based recommendation. In International Conference on Research and Development in Information Retrieval (SIGIR) . 169-178.
- [24] Zhaobo Wang, Yanmin Zhu, Haobing Liu, and Chunyang Wang. 2022. Learning graph-based disentangled representations for next POI recommendation. In International Conference on Research and Development in Information Retrieval (SIGIR) . 1154-1163.
- [25] Wei Wei, Chao Huang, Lianghao Xia, and Chuxu Zhang. 2023. Multi-Modal Self-Supervised Learning for Recommendation. In The Web Conference (WWW) . 790-800.
- [26] Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie. 2021. Self-supervised graph learning for recommendation. In International Conference on Research and Development in Information Retrieval (SIGIR) . 726-735.
- [27] Liang Wu, Diane Hu, Liangjie Hong, and Huan Liu. 2018. Turning clicks into purchases: Revenue optimization for product search in e-commerce. In International Conference on Research &amp; Development in Information Retrieval (SIGIR) . 365-374.
- [28] Shiwen Wu, Fei Sun, Wentao Zhang, Xu Xie, and Bin Cui. 2022. Graph neural networks in recommender systems: a survey. Comput. Surveys 55, 5 (2022), 1-37.
- [29] Lianghao Xia, Chao Huang, Jiao Shi, and Yong Xu. 2023. Graph-less collaborative filtering. In The Web Conference (WWW) . 17-27.
- [30] Lianghao Xia, Chao Huang, Yong Xu, Jiashu Zhao, Dawei Yin, and Jimmy Huang. 2022. Hypergraph contrastive collaborative filtering. In International Conference on Research and Development in Information Retrieval (SIGIR) . 70-79.
- [31] Lianghao Xia, Chao Huang, and Chuxu Zhang. 2022. Self-supervised hypergraph transformer for recommender systems. In International Conference on Knowledge Discovery and Data Mining (KDD) . 2100-2109.
- [32] Tiansheng Yao, Xinyang Yi, Derek Zhiyuan Cheng, Felix Yu, Ting Chen, Aditya Menon, Lichan Hong, Ed H Chi, Steve Tjoa, Jieqi Kang, et al. 2021. Self-supervised learning for large-scale item recommendations. In International Conference on Information &amp; Knowledge Management (CIKM) . 4321-4330.
- [33] Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L Hamilton, and Jure Leskovec. 2018. Graph convolutional neural networks for web-scale recommender systems. In International Conference on Knowledge Discovery and Data Mining (KDD) . 974-983.
- [34] Ruohan Zhan, Changhua Pei, Qiang Su, Jianfeng Wen, Xueliang Wang, Guanyu Mu, Dong Zheng, Peng Jiang, and Kun Gai. 2022. Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation. In International Conference on Knowledge Discovery and Data Mining (KDD) . 4472-4481.
- [35] Fanjin Zhang, Jie Tang, Xueyi Liu, Zhenyu Hou, Yuxiao Dong, Jing Zhang, Xiao Liu, Ruobing Xie, Kai Zhuang, Xu Zhang, et al. 2021. Understanding WeChat user preferences and 'wow' diffusion. Transactions on Knowledge and Data Engineering (TKDE) 34, 12 (2021), 6033-6046.
- [36] Jiani Zhang, Xingjian Shi, Shenglin Zhao, and Irwin King. 2019. Star-gcn: Stacked and reconstructed graph convolutional networks for recommender systems. International Joint Conference on Artificial Intelligence (IJCAI) (2019).
- [37] Mengqi Zhang, Shu Wu, Xueli Yu, Qiang Liu, and Liang Wang. 2022. Dynamic graph neural networks for sequential recommendation. Transactions on Knowledge and Data Engineering (TKDE) 35, 5 (2022), 4741-4753.
- [38] Qianru Zhang, Chao Huang, Lianghao Xia, Zheng Wang, Zhonghang Li, and Siuming Yiu. 2023. Automated Spatio-Temporal Graph Contrastive Learning. In The Web Conference (WWW) . 295-305.
- [39] Zaixi Zhang, Qi Liu, Hao Wang, Chengqiang Lu, and Chee-Kong Lee. 2021. Motif-based graph self-supervised learning for molecular property prediction. In Advances in Neural Information Processing Systems (NeurIPS) . 15870-15882.
- [40] Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. 2020. S3-rec: Self-supervised learning for sequential recommendation with mutual information maximization. In International Conference on Information &amp; Knowledge Management (CIKM) . 1893-1902.