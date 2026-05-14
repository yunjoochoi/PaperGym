## Abstract

Large Language Models (LLMs) have attracted significant attention in recommender systems for their excellent world knowledge capabilities. However, existing methods that rely on Euclidean space struggle to capture the rich hierarchical information inherent in textual and semantic data, which is essential for capturing user preferences. The geometric properties of hyperbolic space offer a promising solution to address this issue. Nevertheless, integrating LLMs-based methods with hyperbolic space to effectively extract and incorporate diverse hierarchical information is non-trivial. To this end, we propose a model-agnostic framework, named HyperLLM , which extracts and integrates hierarchical information from both structural and semantic perspectives. Structurally, HyperLLM uses LLMs to generate multi-level classification tags with hierarchical parent-child relationships for each item. Then, tagitem and user-item interactions are jointly learned and aligned through contrastive learning, thereby providing the model with clear hierarchical information. Semantically, HyperLLM introduces a novel meta-optimized strategy to extract hierarchical information from semantic embeddings and bridge the gap between the semantic and collaborative spaces for seamless integration. Extensive experiments show that HyperLLM significantly outperforms recommender systems based on hyperbolic space and LLMs, achieving performance improvements of over 40%. Furthermore, HyperLLM not only improves recommender performance but also enhances training stability, highlighting the critical role of hierarchical information in recommender systems.

## CCS Concepts

· Information systems → Recommender systems .

∗ Also with Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ). † Corresponding authors.

[Our code is released at https://github.com/Qin-lab-code/HyperLLM.](https://github.com/Qin-lab-code/HyperLLM)

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. SIGIR '25, Padua, Italy

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 979-8-4007-1592-1/2025/07

## Large Language Models Enhanced Hyperbolic Space Recommender Systems

Wentao Cheng ∗ Beijing Institute of Technology Beijing, China wentao.cheng23@gmail.com Zhida Qin †

Beijing Institute of Technology Beijing, China zanderqin@bit.edu.cn

Pengzhan Zhou Chongqing University Chongqing, China pzzhou@cqu.edu.cn Zexue Wu Beijing Institute of Technology Beijing, China zexue.wu@bit.edu.cn Tianyu Huang Beijing Institute of Technology Beijing, China huangtianyu@bit.edu.cn

## Keywords

Recommender Systems, Large Language Models, Hyperbolic Space

## ACMReference Format:

Wentao Cheng, Zhida Qin, Zexue Wu, Pengzhan Zhou, and Tianyu Huang. 2025. Large Language Models Enhanced Hyperbolic Space Recommender Systems. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25), July 13-18, 2025, Padua, Italy. ACM, New York, NY, USA, 10 pages. https://doi.org/10. 1145/3726302.3730019

## 1 Introduction

Recommender systems have become fundamental to various industries, from e-commerce to content streaming. Traditional recommender techniques [10, 15] are effective, but they often face challenges in handling complex relationships between users and items, especially when dealing with large-scale and sparse data. As the demand for personalized recommendations increases, recent advances [35, 42] in geometric methods provide promising solutions to address these challenges and improve performance.

Hyperbolic geometry is a non-Euclidean space that has gained increasing attention for its ability to model hierarchical structures and complex relationships [2, 7, 26]. Unlike the Euclidean space, where spatial capacity grows linearly, hyperbolic space exhibits exponential growth, making it particularly suitable for representing hierarchical data, such as taxonomies and social networks. Research [28, 32, 41] indicates that hierarchical structures exist between users and items in recommender systems. The unique properties of hyperbolic space enable it to effectively model these relationships, thereby better capturing user preferences.

In recent years, with the widespread application of Large Language Models (LLMs) [1, 16] across various fields, their potential to enhance recommender systems has attracted significant attention. A major category of LLMs-based recommender systems [18, 25, 31] focuses on utilizing the excellent world knowledge capabilities of LLMs to extract and enrich semantic information from recommender data, addressing challenges such as data sparsity and longtail effect. However, recent studies [3, 36] show that semantic information often contains rich hierarchical structures, and methods relying on Euclidean space may struggle to capture these structures, resulting in performance degradation. The hyperbolic space offers a promising direction to address this issue.

However, it is non-trivial to integrate LLMs-based methods with hyperbolic space, which presents challenges arising from two aspects. The first challenge lies in how to extract the diverse hierarchical information hidden within textual and semantic data. The second challenge is the significant gap between the semantic and collaborative information, which makes it difficult to incorporate the extracted hierarchical information into recommender systems. Moreover, this gap is further exacerbated by the differences between hyperbolic and Euclidean spaces. To address these challenges, we extract and integrate hierarchical information from structural and semantic perspectives.

From the structural perspective, we focus on well-defined hierarchical relationships in the data. Specifically, we use LLMs to generate multi-level tags with parent-child relationships for each item, representing classifications ranging from broad to specific. These tags are inspired by classification networks [11] in real-world taxonomies, which exhibit clear hierarchical structures. To incorporate this information into the hyperbolic model, we construct a tag-item matrix and jointly learn the tag-item and user-item interactions. Subsequently, hyperbolic space contrastive learning [23] aligns item representations from both sides, collectively providing structural hierarchical information to the model.

Furthermore, we extract semantic hierarchical information to capture more nuanced aspects of the data. Generally, the original textual data describes the objective characteristics of items but rarely reflects subjective user preferences, resulting in a lot of noise unrelated to preferences. Therefore, we use LLMs to summarize preference related information for both users and items. These summaries are converted into semantic embeddings using a text embedding model [21]. Then, we propose a novel meta-optimized two-phase training strategy, where each phase corresponds to a distinct training objective. In the first phase, the primary objective is to extract semantic hierarchical information from the semantic embeddings. During this phase, we freeze the parameters of both the semantic embeddings and the hyperbolic model, training only the Mixture of Experts (MoE) model [6, 14, 27] to transform the semantic embeddings effectively. In the second phase, we integrate the extracted hierarchical information into the hyperbolic model for recommendations. For meta-learning methods [9, 22], freezing parameters facilitates the model to learn general knowledge from the target task, which improves generalization on new tasks. To distinguish between these two phases and encourage the model to learn more generalizable knowledge, we design task-specific adaptive margin ranking losses for each phase. Note that the structural and semantic approaches are model-agnostic.

In conclusion, we use LLMs to extract structural and semantic hierarchical information and propose a model-agnostic framework, named HyperLLM , which effectively integrates this information into hyperbolic space recommender systems. Consequently, HyperLLM achieves significant performance improvements on multiple hyperbolic baselines, with the highest improvement exceeding 40%.

Our contributions can be summarized as follows:

- We leverage LLMs to extract both structural and semantic hierarchical information, taking into account the nature of recommender data and the properties of hyperbolic space. This approach enhances the recommender system's capability to accurately capture hierarchical relationships and user preferences.
- We introduce a novel meta-optimized strategy for extracting semantic hierarchical information with hyperbolic model, bridging the gap between semantic and hyperbolic collaborative spaces. This approach also provides a promising direction for semantic fusion in both Euclidean and hyperbolic space recommender systems.
- We conduct extensive experiments demonstrating the benefits of structural and semantic hierarchical information. Our proposed HyperLLM framework outperforms existing LLMs-based recommender frameworks in both performance and training stability, highlighting the importance of hierarchical information.

## 2 Related Work

## 2.1 Hyperbolic Space Recommender Systems

The geometric properties of hyperbolic space make hyperbolic neural networks highly effective for modeling data with hierarchical structures. In recommender systems, this ability to capture the inherent hierarchical relationships in user-item interactions has been increasingly recognized. Early efforts in this field focus on adapting techniques from Euclidean space. For example, HyperML [29] and HME [5] introduce metric learning into hyperbolic space recommender systems. HVAE [19] designs a variational autoencoder model on hyperbolic space. HGCF [28] proposes a Hyperbolic Graph Convolution Network (HGCN). These studies demonstrate the advantages of hyperbolic space through notable performance. Consequently, recent research has explored more efficient ways to leverage hyperbolic space. For instance, HRCF [38] proposes a hyperbolic geometric aware regularizer loss, and HICF [37] enhances margin ranking loss with adaptive margins. However, due to the challenges in formalizing neural operations within hyperbolic space, earlier methods often map vectors to Euclidean space for computation, then back to hyperbolic space, resulting in suboptimal utilization of the hyperbolic space. To address this limitation, LGCF [32], GGCF [35], and HGCH [40] respectively propose fully HGCN on Klein, Hyperboloid, and Poincaré hyperbolic spaces, which perform computation entirely within hyperbolic space.

## 2.2 LLMs-based Recommender Systems

Existing LLMs-based recommender systems can be broadly divided into two categories. The first category utilizes LLMs directly as the recommender system by applying strategies such as pre-training or fine-tuning LLMs. Early work in this category, such as P5 [8], models multiple recommender tasks as a unified pre-training framework. ZeroShot [12] applies zero-shot techniques to LLMs for predicting recommendations without the need for fine-tuning. Recent studies mainly focus on enhancing LLMs from multiple perspectives, including performance, efficiency, and others. CLLM4Rec [44] designs a polynomial prediction head that enables the LLMs to predict multiple items in one step, avoiding auto-regressive prediction. TLRec [17] uses the chain-of-thought technique to help LLMs better understand recommender data. LC-Rec [43] employs vector quantization to map items into learnable vectors that are compatible with LLMs, enhancing the collaborative semantics in LLMs.

The second category enhances existing recommender systems through strategies such as generating additional textual data or semantic representations. LLMRec [33] uses LLMs to augment interaction graph and textual attributes to address the problem of data sparsity. SAID [13] employs LLMs to learn semantically aligned item embeddings, which are then used to enhance downstream models. LRD [39] leverages LLMs to extract latent relations between items. KAR [34] utilizes LLMs to generate reasoning and factual knowledge, which are then transformed into vectors by a hybrid-expert adaptor, making them compatible with recommender systems. Furthermore, some studies propose novel strategies to bridge the gap between semantic information and collaborative information. RLMRec [24] employs contrastive learning and masked learning techniques to align collaborative representations with semantic ones. CARec [30] uses MSE loss to align collaborative and semantic representations in two bidirectional stages.

## 3 Preliminaries

Hyperbolic space is a non-Euclidean space with constant negative curvature. It can be described using various mathematical models, including the Hyperboloid model, the Poincaré model, the Klein model, and others. These models differ in how they represent the distance, curvature, and symmetry of hyperbolic space. In recommender systems, the Hyperboloid and Poincaré models are commonly used due to their ability to effectively represent hierarchical structures and complex relationships.

For the 𝑑 -dimensional hyperbolic space with constant negative curvature 𝑐 , the Hyperboloid model H 𝑑 is represented as:

<!-- formula-not-decoded -->

where ⟨· , ·⟩ H denotes the inner product in Hyperboloid model:

<!-- formula-not-decoded -->

In addition, the Poincaré model P 𝑑 is represented as:

<!-- formula-not-decoded -->

where ∥ x ∥ is the Euclidean norm of the point x .

In Euclidean space, the inner product is used to calculate the angle between vectors and measure similarity. However, in hyperbolic space, the inner product does not directly reflect similarity between points, so distance metrics are typically used instead.

The distance between two points x , y ∈ H 𝑑 is defined as:

<!-- formula-not-decoded -->

where 𝑘 = -1 / 𝑐 is the negative reciprocal of 𝑐 and the distance between two points x , y ∈ P 𝑑 is defined as:

<!-- formula-not-decoded -->

This property also makes margin ranking loss commonly used in hyperbolic space recommender systems, as it has been shown to be useful in distance-based methods.

The hyperbolic margin ranking loss L 𝑚 is expressed as:

<!-- formula-not-decoded -->

where 𝑢 is user, 𝑖 and 𝑗 are the positive and negative items of 𝑢 ; h 𝑢 , h 𝑖 , and h 𝑗 are the embeddings of the user and items; 𝑑 (· , ·) is the distance metric (e.g., 𝑑 H , 𝑑 2 H ); and 𝑚 is the margin hyper-parameter.

The margin 𝑚 defines the minimum separation required between the positive and negative item distances in order to minimize the loss. If the difference between the distances is less than 𝑚 , the loss penalizes the model, encouraging it to learn embeddings that better separate relevant items from irrelevant ones. The goal is to ensure that the model learns to push positive items closer to the user's embedding while pushing negative items farther away, ultimately improving its ability to make accurate prediction.

In meta-learning , freezing parameters is a fast adaptation technique that enables the model to generalize more effectively to new tasks by optimizing the target task. In our proposed meta-optimized two-phase strategy, the target task involves extracting semantic hierarchical information in the first phase, while the new task corresponds to the recommender task in the second phase. To differentiate between these two tasks while promoting shared knowledge across them, we apply hyperbolic margin ranking losses in both phases, with margins tailored to the characteristics of each task.

## 4 Framework

As illustrated in Figure 1, our proposed HyperLLM framework consists of three modules: LLMs-based Structural Extraction , Metaoptimized Semantic Extraction , and Structural and Semantic Integration . The LLMs-based Structural Extraction module uses LLMs to extract structural hierarchical information and summarize preference information from recommender data. Building on this, the Meta-optimized Semantic Extraction module introduces a meta-optimized strategy to extract semantic hierarchical information from the embeddings of preference summaries. Finally, the Structural and Semantic Integration module integrates the structural and semantic hierarchical information into hyperbolic space recommender systems to construct a unified framework.

In addition, the Meta-optimized Semantic Extraction module corresponds to the first phase of the meta-optimized two-phase training strategy, while the Structural and Semantic Integration module corresponds to the second phase . The training of these two phases is decoupled, indicating that the second phase begins only after the first phase is completed.

## 4.1 LLMs-based Structural Extraction

In textual-based recommender systems, the data typically consists of user-item interactions and associated textual data, such as item titles, brands, descriptions, and other relevant metadata. We assume the user-item interactions to be a matrix I 𝑢𝑠𝑒𝑟 , where each entry I 𝑢𝑠𝑒𝑟 𝑢𝑖 indicates the interaction between user 𝑢 and item 𝑖 . Additionally, the textual data is represented as T 𝑖 = { 𝑡 1 𝑖 , 𝑡 2 𝑖 , . . . , 𝑡 𝑚 𝑖 } , where 𝑡 𝑗 𝑖 refers to the 𝑗 -th textual attribute of item 𝑖 .

To extract structural hierarchical information , we use LLMs to generate multi-level tags with parent-child relationships for each item. These tags represent broad-to-specific classifications, and the hierarchy of the tags is defined by their relationships, reflecting clear hierarchical structures similar to real-world taxonomies.

Specifically, for each item 𝑖 , we generate a set of tags as Tags 𝑖 = { tag 1 𝑖 , . . . , tag 𝑛 𝑖 } , where tag 𝑗 𝑖 is the 𝑗 -th tag, ranging from general categories to more specific ones. Additionally, we generate edges that define the parent-child relationships between these tags, where higher-level tags act as parents and their related lower-level tags act as children. These edges are established for all tags generated by the LLMs. Formally, we define a set of edges E = {( tag 𝑎 , tag 𝑏 ) | tag 𝑎 → tag 𝑏 } , where tag 𝑎 is a parent tag, tag 𝑏 is its child tag, and tag 𝑎 → tag 𝑏 indicates that tag 𝑎 is semantically broader than tag 𝑏 .

Figure 1: The overall architecture of our proposed HyperLLM. It consists of three modules: LLMs-based Structural Extraction, Meta-optimized Semantic Extraction, and Structural and Semantic Integration. These modules operate in the specified order.

<!-- image -->

Since the original textual data describes the objective characteristics of items but rarely reflects subjective user preferences, we use LLMs to generate preference summaries that condense the key features and appeal of each item into a more compact format with higher information related to user preferences. We denote the preference summary for item 𝑖 as 𝑆 𝑖 .

These steps are accomplished by the following prompt:

## Item Prompt:

Please generate a concise summary highlighting the main features and appeal of the item.

Please generate 3 level tags for the item. The tags must be different and not duplicated at different levels. Each level can have multiple tags. Higher-level tags should represent broad, general categories that define the item's overall domain or purpose. Lower-level tags should represent increasingly detailed classifications.

Please generate edges between different level tags. Each edge should represent a parent-child relationship, where a higherlevel tag is the parent and a lower-level tag is the child. The tags on each edge must be within the generated set of tags.

## Input:

Item Information: 𝑡 1 𝑖 | 𝑡 2 𝑖 | . . . | 𝑡 𝑚 𝑖 .

Output:

Summary: { 𝑆 𝑖 } ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We explicitly specify the tags' layer as 3 in the prompt. On the one hand, it ensures consistency in the tags generated for different items. On the other hand, three layers of tags allows for a clear distinction between broader and more specific categories.

After generating textual data for items, we use LLMs to create user-specific textual data by analyzing the item summaries and tags associated with user-interacted items. This step is critical for capturing the user's preferences in a concise manner. We denote the summary of user 𝑢 as 𝑆 𝑢 and 𝑢 -interacted items as { 𝑖 1 𝑢 , . . . , 𝑖 𝑘 𝑢 } .

The following prompt is used to generate the user summary:

## User Prompt:

Please analyze the item summaries and tags and generate a concise summary that captures the user's preferences based on these inputs. Ensure that the summary reflects the user's preferences, avoiding repetition while capturing patterns, common themes, and specific features.

## Input:

Summary: 𝑆 𝑖 1 𝑢 , Tags: Tags 𝑖 1 𝑢 | . . . | Summary: 𝑆 𝑖 𝑘 𝑢 , Tags: Tags 𝑖 𝑘 𝑢

Output:

Summary: { 𝑆 𝑢 } .

Due to the hyperbolic model cannot process textual data, we use a text embedding model to encode the user and item preference summaries into semantic embeddings . Formally, we denote the

.

text embedding model as Encoder , which encodes the user and item summaries 𝑆 𝑢 and 𝑆 𝑖 into embeddings s 𝑢 and s 𝑖 , where:

<!-- formula-not-decoded -->

## 4.2 Meta-optimized Semantic Extraction

In this module, we extract hierarchical information from semantic embeddings, corresponding to the first phase of the meta-optimized two-phase training strategy. Specifically, we employ a MoE model to transform these embeddings from semantic space to hyperbolic collaborative space. The transformed embeddings are then fed into the hyperbolic model, which is optimized using hyperbolic margin ranking loss. By freezing both semantic embeddings and hyperbolic model parameters, we leverage the hyperbolic model's ability to capture hierarchical information, enabling the MoE to extract hierarchical information from the semantic embeddings.

Formally, we denote the hyperbolic model as HM and the margin used in this phase as 𝑚 1. This phase can be represented as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ∗ denotes frozen parameters; 𝐾 is the number of experts; Expert 𝑘 is the 𝑘 -th expert in MoE; 𝑔 𝑘 (·) is the gating function that computes the weight for 𝑘 -th expert; W gate ∈ R 𝑑 1 × 𝐾 and W expert 𝑘 ∈ R 𝑑 1 × 𝑑 2 are weight matrices; b gate ∈ R 𝐾 and b expert 𝑘 ∈ R 𝑑 2 are biases; and L 𝑚𝑒𝑡𝑎 is the loss computed by HM .

L 𝑚𝑒𝑡𝑎 comprises the hyperbolic margin ranking loss along with other auxiliary losses of HM . The margin 𝑚 1 defines the minimum separation distance between positive and negative samples in the hyperbolic margin ranking loss of L 𝑚𝑒𝑡𝑎 . We adopt a smaller value for margin 𝑚 1, enabling the MoE to make subtle adjustments to the semantic embeddings without causing excessive separation, which could hinder effective extraction and lead to overfitting in the subsequent recommender task.

## 4.3 Structural and Semantic Integration

After the Meta-optimized Semantic Extraction module is completed, we integrate the extracted structural and semantic hierarchical information into the hyperbolic model , corresponding to the second phase of the meta-optimized strategy. In this phase, we use the MoE-transformed embeddings x 𝑢 , x 𝑖 to initialize the user and item ID embeddings, and then train the hyperbolic model.

This phase integrates the semantic hierarchical information into the hyperbolic model and can be represented as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where e 𝑢 and e 𝑖 are the user and item ID embeddings; h 𝑢 and h 𝑖 are the final user and item representations learned from I 𝑢𝑠𝑒𝑟 ; and L 𝑢𝑖 , 𝑚 2 are the loss and margin used for learning I 𝑢𝑠𝑒𝑟 .

Furthermore, we construct a tag-item interaction matrix I 𝑡𝑎𝑔 , where the entry I 𝑡𝑎𝑔 tag 𝑎 -tag 𝑏 indicates the interaction between tag 𝑎 and tag 𝑏 , and the entry I 𝑡𝑎𝑔 tag -𝑖 indicates the interaction between tag and its associated item 𝑖 . Each tag has an ID embedding e 𝑡 , and we use the same hyperbolic model for I 𝑢𝑠𝑒𝑟 to learn e 𝑡 .

<!-- formula-not-decoded -->

where h 𝑡 , h ′ 𝑖 are the final tag and item representations learned from I 𝑡𝑎𝑔 ; and L 𝑡𝑎𝑔 , 𝑚 3 are the loss and margin used for learning I 𝑡𝑎𝑔 .

L 𝑡𝑎𝑔 is jointly learned with L 𝑢𝑖 to optimize the same hyperbolic model. Finally, we apply contrastive learning to align the item representations obtained from both the tag-item interactions and the user-item interactions. It enables the structural hierarchical information in the tags to be sufficiently learned and effectively integrated into the hyperbolic model.

The overall loss function in this phase is defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where L 𝑐𝑙 is the contrastive learning loss; I denotes the set of all items; 𝑤 and 𝜏 are hyper-parameters controlling the weight and temperature of L 𝑐𝑙 ; 𝑑 2 is the squared hyperbolic distance.

## 5 Experiment

In this section, we perform a series of comprehensive experiments and in-depth analyses to investigate the following questions:

- Q1: What is the impact of structural and semantic hierarchical information on the performance of recommender systems?
- Q2: What is the contribution of the individual components of HyperLLM to its overall performance?
- Q3: Can the meta-optimized strategy accelerate the convergence speed of the subsequent recommender task?
- Q4: Is HyperLLM superior to existing LLMs-based recommender frameworks in terms of performance and efficiency?
- Q5: What is the impact of HyperLLM on users with different sparsities, and can it address the long-tail effect?
- Q6: Can HyperLLM's representations confirm that it learns structural and semantic hierarchical information?

## 5.1 Experimental Setup

Table 1: Dataset statistics. Avg Token denotes the average number of tokens in the text Features.

| Dataset   |   Users |   Items |   User-Item | Density   |   Features |   Avg Token |
|-----------|---------|---------|-------------|-----------|------------|-------------|
| Toys      |  11,268 |   7,309 |      95,420 | 0.12%     |     27,690 |         125 |
| Sports    |  22,686 |  12,301 |     185,718 | 0.07%     |     43,743 |         159 |
| Beauty    |  10,553 |   6,086 |      94,148 | 0.15%     |     22,602 |         138 |

Table 2: Dataset statistics after LLMs-based structural extraction. Avg Summary denotes the average number of tokens in the preference summaries.

| Dataset   |   Tags |   Tag-Item |   Tag-Tag |   Avg Summary |
|-----------|--------|------------|-----------|---------------|
| Toys      | 10,802 |     25,369 |    61,610 |            54 |
| Sports    | 16,841 |     38,197 |    98,339 |            51 |
| Beauty    |  6,418 |     16,739 |    51,396 |            55 |

SIGIR '25, July 13-18, 2025, Padua, Italy Cheng et al.

Table 3: Recall(R) and NDCG(N) results for all baselines, baselines with Structural hierarchical information, baselines with Semantic hierarchical information, and baselines with HyperLLM. The best variant for each metric is highlighted in bold.

| Dataset   | Dataset          | Toys          | Toys          | Toys          | Toys          | Sports        | Sports        | Sports        | Sports        | Beauty        | Beauty        | Beauty        | Beauty        |
|-----------|------------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| Backbone  | Variants         | R@10          | R@20          | N@10          | N@20          | R@10          | R@20          | N@10          | N@20          | R@10          | R@20          | N@10          | N@20          |
| HGCF      | Base             | 0.0865        | 0.1211        | 0.0489        | 0.0578        | 0.0581        | 0.0852        | 0.0308        | 0.0377        | 0.1150        | 0.1604        | 0.0651        | 0.0771        |
| HGCF      | Structural       | 0.0964        | 0.1367        | 0.0533        | 0.0636        | 0.0620        | 0.0944        | 0.0333        | 0.0415        | 0.1176        | 0.1656        | 0.0675        | 0.0801        |
| HGCF      | Semantic         | 0.1138        | 0.1599        | 0.0628        | 0.0746        | 0.0731        | 0.1091        | 0.0386        | 0.0477        | 0.1281        | 0.1848        | 0.0722        | 0.0870        |
| HGCF      | HyperLLM         | 0.1168        | 0.1638        | 0.0639        | 0.0759        | 0.0754        | 0.1099        | 0.0394        | 0.0482        | 0.1289        | 0.1889        | 0.0736        | 0.0893        |
| HGCF      | Best Imprv.      | 35.03%        | 35.26%        | 30.67%        | 31.31%        | 29.78%        | 28.99%        | 27.92%        | 27.85%        | 12.09%        | 17.77%        | 13.06%        | 15.82%        |
| HRCF      | Base             | 0.0867        | 0.1200        | 0.0491        | 0.0576        | 0.0576        | 0.0854        | 0.0308        | 0.0378        | 0.1154        | 0.1607        | 0.0649        | 0.0768        |
| HRCF      | Structural       | 0.0960        | 0.1374        | 0.0526        | 0.0632        | 0.0613        | 0.0952        | 0.0326        | 0.0412        | 0.1167        | 0.1694        | 0.0666        | 0.0803        |
| HRCF      | Semantic         | 0.1174        | 0.1654        | 0.0654        | 0.0776        | 0.0750        | 0.1111        | 0.0395        | 0.0487        | 0.1310        | 0.1890        | 0.0733        | 0.0885        |
| HRCF      | HyperLLM         | 0.1208        | 0.1700        | 0.0664        | 0.0789        | 0.0771        | 0.1133        | 0.0403        | 0.0495        | 0.1297        | 0.1904        | 0.0736        | 0.0895        |
| HRCF      | Best Imprv.      | 39.33%        | 41.67%        | 35.23%        | 36.98%        | 33.85%        | 32.67%        | 30.84%        | 30.95%        | 13.52%        | 18.48%        | 13.41%        | 16.54%        |
| HICF      | Base             | 0.0945        | 0.1345        | 0.0534        | 0.0636        | 0.0621        | 0.0912        | 0.0338        | 0.0412        | 0.1207        | 0.1685        | 0.0696        | 0.0820        |
| HICF      | Structural       | 0.1003        | 0.1378        | 0.0561        | 0.0656        | 0.0655        | 0.0970        | 0.0355        | 0.0435        | 0.1239        | 0.1742        | 0.0701        | 0.0833        |
| HICF      | Semantic         | 0.1194        | 0.1676        | 0.0666        | 0.0790        | 0.0782        | 0.1168        | 0.0421        | 0.0519        | 0.1389        | 0.1959        | 0.0803        | 0.0953        |
| HICF      | HyperLLM         | 0.1222        | 0.1690        | 0.0679        | 0.0798        | 0.0792        | 0.1175        | 0.0425        | 0.0523        | 0.1404        | 0.1978        | 0.0807        | 0.0956        |
| HICF      | Best Imprv.      | 29.31%        | 25.65%        | 27.15%        | 25.47%        | 27.54%        | 28.84%        | 25.74%        | 26.94%        | 16.32%        | 17.39%        | 15.95%        | 16.59%        |
| HGCH      | Base             | 0.0989        | 0.1393        | 0.0541        | 0.0645        | 0.0569        | 0.0885        | 0.0299        | 0.0379        | 0.1200        | 0.1727        | 0.0663        | 0.0801        |
| HGCH      | Structural       | 0.1003        | 0.1424        | 0.0544        | 0.0651        | 0.0599        | 0.0936        | 0.0316        | 0.0402        | 0.1240        | 0.1749        | 0.0699        | 0.0832        |
| HGCH      | Semantic         | 0.1070        | 0.1536        | 0.0590        | 0.0709        | 0.0702        | 0.1057        | 0.0360        | 0.0451        | 0.1259        | 0.1844        | 0.0711        | 0.0864        |
|           | HyperLLM         | 0.1100        | 0.1547        | 0.0604        | 0.0718        | 0.0715        | 0.1065        | 0.0368        | 0.0457        | 0.1296        | 0.1861        | 0.0725        | 0.0872        |
| HyperCL   | Best Imprv. Base | 11.22%        | 11.06%        | 11.65%        | 11.32%        | 25.66%        | 20.34%        | 23.08%        | 20.58%        | 8.00%         | 7.76%         | 9.35%         | 8.86%         |
| HyperCL   | Structural       | 0.0984 0.0986 | 0.1381        | 0.0549        | 0.0651        | 0.0641        | 0.0965 0.0980 | 0.0348        | 0.0431        | 0.1144        | 0.1659        | 0.0637        | 0.0772        |
| HyperCL   | Semantic         | 0.1110        | 0.1405 0.1571 | 0.0550 0.0611 | 0.0658 0.0729 | 0.0646 0.0717 | 0.1096        | 0.0350 0.0387 | 0.0435 0.0484 | 0.1165 0.1207 | 0.1667 0.1782 | 0.0654 0.0671 | 0.0785 0.0820 |
| HyperCL   | HyperLLM         | 0.1127        | 0.1599        | 0.0626        | 0.0746        | 0.0724        | 0.1102        | 0.0391        | 0.0486        | 0.1227        | 0.1803        | 0.0677        | 0.0827        |
| HyperCL   | Best Imprv.      |               | 15.79%        | 14.03%        | 14.59%        |               |               |               |               | 7.26%         | 8.68%         | 6.28%         |               |
|           |                  | 14.53%        |               |               |               | 12.95%        | 14.20%        | 12.36%        | 12.76%        |               |               |               | 7.12%         |

5.1.1 Datasets. In the experiments, we use three datasets that are frequently employed in LLMs-based recommender systems: Amazon-Toys, Amazon-Sports, and Amazon-Beauty. We select useritem interactions with ratings ≥ 4 and apply 5-core settings to represent implicit feedback preferences. The statistics for these datasets and after LLMs-based structural extraction are shown in Table 1 and 2, respectively. It can be observed that the preference summaries contain fewer tokens, indicating higher information density. The interactions in each dataset are randomly split into training, validation, and test sets with a ratio of 8:1:1. We evaluate the model's performance on these datasets using Recall@ 𝐾 and NDCG@ 𝐾 . To ensure consistency, the model is trained using the best-performing hyper-parameters based on the validation set.

5.1.2 Baselines. We build HyperLLM upon the following hyperbolic space recommender systems. These baselines include both the most popular methods and the latest advancements. Additionally, these methods utilize various hyperbolic mathematical models and hyperbolic neural network architectures.

- HGCF [28] proposes an HGCN based on the Hyperboloid model and optimizes it using margin ranking loss.
- HRCF [38] introduces root alignment and geometry-aware regularization loss to HGCF.
- HICF [37] improves the margin ranking loss with geometryaware adaptive margins and informative negative sampling.
- HGCH [40] implements a fully HGCN on the Poincaré model using the gyromidpoint technique and margin ranking loss.
- HyperCL [23] designs a fully HGCN on the Hyperboloid model and aligns HGCN with fully HGCN using contrastive learning.

5.1.3 Settings. We utilize the LLaMA3-8B [4] as LLMs to generate textual data and employ the text-embedding-3-large model [21] for encoding preference summaries into semantic embeddings. The dimension 𝑑 1 of semantic embeddings is 3072 and the dimension 𝑑 2 of ID embeddings is 50. The number of experts 𝐾 in MoE is selected from { 12 , 16 , 20 } . The margin 𝑚 1 used in meta-optimized phase is selected from { 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 } . In most cases, the margin 𝑚 2 is selected from [ 0 . 1 , 4 . 0 ] and 𝑚 3 is selected from [ 0 . 1 , 1 . 0 ] . The 𝑤 and 𝜏 in contrastive learning are set to 0.01 and 0.5. The training epochs is set to 500, with early stopping applied after 50 epochs. Trainingrelated hyper-parameters, such as the learning rate, are maintained consistent across the same baseline to ensure fairness. Other modelrelated hyper-parameters are adjusted using grid search. All of our experiments can be completed on a single RTX 3090.

## 5.2 Recommender Performance (Q1)

Table 3 presents the performance of baselines integrated with structural and semantic hierarchical information. The Structural , Semantic , and HyperLLM variants represent baselines integrated with structural, semantic, and both structural and semantic hierarchical information, respectively. The results show that incorporating structural or semantic hierarchical information consistently improves the performance of baselines. Notably, the performance gains from semantic hierarchical information are more substantial, suggesting that the nuanced latent information are more beneficial for hyperbolic space recommender systems. HyperLLM leverages both structural and semantic hierarchical information, achieving superior performance compared to the Structural and Semantic variants. Additionally, HyperLLM achieves maximum improvements of 35.3%, 41.7%, 29.3%, 25.7%, 15.8% over HGCF, HRCF, HICF, HGCH, HyperCL, respectively. In terms of average improvements, HyperLLM outperforms these baselines by 25.5%, 28.5%, 23.6%, 14.1%, 11.7%, respectively. It highlights the complementary relationship between these two types of information, each contributing uniquely to the model's understanding of the semantic hierarchy.

Figure 2: The ablation results for HyperLLM, with missing values for the w/o Meta variant in HICF and HyperCL due to NaN occurrences during the first epoch of training.

<!-- image -->

However, it is noteworthy that HyperLLM exhibits relatively smaller improvements when applied to baselines employing fully hyperbolic architectures. Since fully hyperbolic baselines perform all neural operations within hyperbolic space, they are able to more effectively capture the hierarchical structures inherent in the interaction data. As a result, these baselines are already adept at leveraging hyperbolic geometry to model complex relationships, leaving less room for additional enhancements. Nonetheless, HyperLLM still provides meaningful performance gains, highlighting its ability to enhance fully hyperbolic models by integrating the hierarchical information. In summary, these results demonstrate that the hierarchical information from semantic and textual data can significantly enhance hyperbolic space recommender systems to capture user preferences.

## 5.3 Ablation Study (Q2)

In this subsection, we remove the components from HyperLLM framework to evaluate their impact to the overall recommender performance. The w/o Meta variant removes the meta-optimized strategy and co-optimizes semantic embeddings and the MoE model with the Structural and Semantic Integration module. We report the result with the higher value between freezing and not freezing the semantic embeddings for the w/o Meta. The only Semantic , w/o Semantic , and w/o Structural variants remove the Structural and Semantic Integration module, semantic hierarchical information, and structural hierarchical information from HyperLLM, respectively. The w/o CL variant removes the contrastive learning loss from Structural and Semantic Integration module.

Figure 3: Recall@20 of baselines, HyperLLM without the meta-optimized strategy, and HyperLLM on the validation set at different epochs.

<!-- image -->

As shown in Figure 2, the results demonstrate that each component contributes positively to the overall performance of HyperLLM. Among them, components related to semantic hierarchical information consistently make greater contributions than those related to structural information. Furthermore, the worst-performing variant w/o Meta cannot produce valid results under the HICF and HyperCL baselines, highlighting the significant gap between the semantic and collaborative spaces. In addition, the only Semantic variant significantly outperforms w/o Meta, indicating that the meta-optimized strategy plays an important role in bridging the gap. Meanwhile, only Semantic underperforms the baselines on the Sports and Beauty datasets, which further validates that the purpose of this strategy is to extract hierarchical information from semantic embeddings, allowing it to be better integrated into the recommender task in the next phase. These findings validate HyperLLM's design, highlighting the crucial role of semantic hierarchical information and the meta-optimized strategy in bridging the gap between semantic and hyperbolic collaborative spaces.

## 5.4 Convergence Analysis (Q3)

To explore whether the meta-optimized strategy accelerates convergence, we analyze the model's performance across different epochs with and without this strategy, as well as the epochs required for convergence. We choose HGCF and HGCH as baselines for analysis, as they employ different hyperbolic mathematical models and architectures, and the results of other baselines are similar to theirs. The w/o Meta variant removes the Meta-optimized Semantic Extraction module from HyperLLM. The w/o Meta and HyperLLM variants are built on the baseline shown in the same figure.

SIGIR '25, July 13-18, 2025, Padua, Italy Cheng et al.

Table 4: Recall, NDCG, and Time (Seconds) results for the baselines with different LLMs-based recommender frameworks. The best variant for each metric is highlighted in bold, and the second-best variant is underlined. The percentages indicate the improvement of each variant over the baseline for each metric.

| Dataset   | Dataset               | Toys                           | Toys                           | Toys     | Sports                         | Sports                         | Sports    | Beauty                         | Beauty                         | Beauty    |
|-----------|-----------------------|--------------------------------|--------------------------------|----------|--------------------------------|--------------------------------|-----------|--------------------------------|--------------------------------|-----------|
| Backbone  | Variants              | Recall@20                      | NDCG@20                        | Time     | Recall@20                      | NDCG@20                        | Time      | Recall@20                      | NDCG@20                        | Time      |
|           | Base                  | 0.1211                         | 0.0578                         | 15       | 0.0852                         | 0.0377                         | 34        | 0.1604                         | 0.0771                         | 39        |
|           | KAR                   | 0.1331(+9.91%)                 | 0.0620(+7.27%)                 | 53       | 0.0730(-14.3%)                 | 0.0306(-18.8%)                 | 724       | 0.1565(-2.43%)                 | 0.0715(-7.26%)                 | 15        |
| HGCF      | RLMRec-Con            | 0.1249(+3.14%)                 | 0.0593(+2.60%)                 | 1121     | 0.0859(+0.82%)                 | 0.0375(-0.53%)                 | 1350      | 0.1619(+0.94%)                 | 0.0773(+0.26%)                 | 219       |
|           | RLMRec-Gen            | 0.1205(-0.50%)                 | 0.0579(+0.17%)                 | 513      | 0.0845(-0.82%)                 | 0.0382(+1.33%)                 | 2014      | 0.1610(+0.37%)                 | 0.0756(-1.95%)                 | 541       |
|           | HyperLLM              | 0.1638(+35.26%)                | 0.0759(+31.31%)                | 261      | 0.1099(+28.99%)                | 0.0482(+27.85%)                | 355       | 0.1889(+17.77%)                | 0.0893(+15.82%)                | 268       |
|           | Base                  | 0.1200                         | 0.0576                         | 33       | 0.0854                         | 0.0378                         | 40        | 0.1607                         | 0.0768                         | 42        |
|           | KAR                   |                                |                                |          |                                |                                |           |                                |                                |           |
|           |                       | 0.1200 (0.00%)                 | 0.0545(-5.38%)                 | 2028     | 0.0679(-20.5%)                 | 0.0282(-25.4%)                 | 1088      | 0.1443(-10.2%)                 | 0.0638(-16.9%)                 | 1071      |
| HRCF      | RLMRec-Con RLMRec-Gen | 0.1234(+2.83%)                 | 0.0586(+1.74%)                 | 198      | 0.0863(+1.05%)                 | 0.0380(+0.53%)                 | 232       | 0.1606(-0.06%)                 | 0.0772(0.52%)                  | 274       |
|           | HyperLLM              | 0.1255(+4.58%) 0.1700(+41.67%) | 0.0585(+1.56%) 0.0789(+36.98%) | 409 356  | 0.0854 (0.00%) 0.1133(+32.67%) | 0.0383(+1.32%) 0.0495(+30.95%) | 2202 821  | 0.1615(+0.50%) 0.1904(+18.48%) | 0.0765(-0.39%) 0.0895(+16.54%) | 645 331   |
|           | Base                  | 0.1345                         | 0.0636                         | 17       | 0.0912                         | 0.0412                         | 19        | 0.1685                         | 0.082                          | 49        |
|           | KAR                   | 0.1397(+3.87%)                 | 0.0645(+1.42%)                 | 127      | 0.0774(-15.1%)                 | 0.0318(-22.8%)                 | 336       | 0.1611(-4.39%)                 | 0.0761(-7.20%)                 | 189       |
| HICF      | RLMRec-Con            | 0.1405(+4.46%)                 | 0.0668(+5.03%)                 | 71       | 0.0943(+3.40%)                 | 0.0420(+1.94%)                 | 151       | 0.1750(+3.86%)                 | 0.0843(+2.80%)                 | 73        |
|           | RLMRec-Gen            | 0.1303(-3.12%)                 | 0.0631(-0.79%)                 | 155      | 0.0872(-4.39%)                 | 0.0378(-8.25%)                 | 222       | 0.1673(-0.71%)                 | 0.0808(-1.46%)                 | 548       |
|           | HyperLLM              | 0.1690(+25.65%)                | 0.0798(+25.47%)                | 169      | 0.1175(+28.84%)                | 0.0523(+26.94%)                | 447       | 0.1978(+17.39%)                | 0.0956(+16.59%)                | 254       |
|           | Base                  | 0.1393                         | 0.0645                         | 200      | 0.0885                         | 0.0379                         | 482       | 0.1727                         | 0.0801                         | 226       |
|           |                       |                                |                                |          |                                |                                |           |                                |                                | 670       |
|           | KAR                   | 0.0736(-47.2%)                 | 0.0350(-45.7%)                 | 109      | 0.0717(-19.0%) 0.0869(-1.81%)  | 0.0316(-16.6%) 0.0367(-3.17%)  | 141       | 0.1202(-30.4%)                 | 0.0550(-31.3%) 0.0817(+2.00%)  | 500       |
| HGCH      | RLMRec-Con RLMRec-Gen | 0.1389(-0.29%) 0.1415(+1.58%)  | 0.0643(-0.31%) 0.0647(+0.31%)  | 673 1812 | 0.0869(-1.81%)                 | 0.0368(-2.90%)                 | 1927 4338 | 0.1722(-0.29%) 0.1713(-0.81%)  | 0.0792(-1.12%)                 | 1970      |
|           | HyperLLM              | 0.1547(+11.06%)                | 0.0718(+11.32%)                | 195      | 0.1065(+20.34%)                | 0.0457(+20.58%)                | 785       | 0.1861(+7.76%)                 | 0.0872(+8.86%)                 | 252       |
|           |                       |                                |                                |          | 0.0965                         |                                |           |                                |                                |           |
|           | Base                  | 0.1381                         | 0.0651                         | 203      |                                | 0.0431                         | 684       | 0.1659                         | 0.0772                         | 336 1617  |
| HyperCL   | KAR                   | 0.0932(-32.5%)                 | 0.0415(-36.3%)                 | 5738     | 0.0488(-49.4%)                 | 0.0198(-54.1%)                 | 9485      | 0.1211(-27.0%)                 | 0.0552(-28.5%) 0.0783(+1.42%)  |           |
|           | RLMRec-Con            | 0.1392(+0.80%)                 | 0.0651 (0.00%) 0.0621(-4.61%)  | 1070     | 0.0977(+1.24%)                 | 0.0436(+1.16%) 0.0415(-3.71%)  | 1344      | 0.1685(+1.57%)                 | 0.0760(-1.55%)                 | 1041 3299 |
|           | RLMRec-Gen HyperLLM   | 0.1312(-5.00%)                 |                                | 2243 170 | 0.0928(-3.83%) 0.1102(+14.20%) | 0.0486(+12.76%)                | 7331 517  | 0.1628(-1.87%) 0.1803(+8.68%)  | 0.0827(+7.12%)                 | 246       |
|           |                       | 0.1599(+15.79%)                | 0.0746(+14.59%)                |          |                                |                                |           |                                |                                |           |

Figure 3 presents the results of these variants on the validation set at each epoch. The results clearly show that HyperLLM significantly outperform the other two variants at the beginning and also require far fewer epochs to converge. This demonstrates that the metaoptimized strategy can effectively accelerate the convergence speed of recommender task in the next phase.

## 5.5 Framework Comparison (Q4)

To verify the superiority of HyperLLM, we compare it with the following LLMs-based recommender frameworks.

- KAR [34] transforms semantic embeddings using a hybrid MoE model and then concatenates them with ID embeddings to form the final input to the model.
- RLMRec-Con [24] uses contrastive learning to align interaction representations with MLP-transformed semantic embeddings.
- RLMRec-Gen [24] processes the masked embeddings through the model and MLP, and aligns them with semantic embeddings.

As illustrated in Table 4, HyperLLM consistently outperforms all LLMs-based frameworks across all datasets, with a notable performance advantage. This underscores the effectiveness of HyperLLM in utilizing structural and semantic hierarchical information within hyperbolic space-an area where existing frameworks fail to exploit.

Furthermore, the performance of compared frameworks varies significantly across different baselines and datasets. In many cases, these frameworks deliver only minor improvements or even declines, with KAR in particular experiencing pronounced drops in performance. In contrast, HyperLLM achieves consistently superior results across all evaluated scenarios, showcasing its robustness and adaptability to diverse data distributions.

In terms of computational efficiency, HyperLLM remains relatively stable across different settings. While it may not always achieve the fastest convergence, its training time stays within an acceptable range. In contrast, the training times of other frameworks fluctuate significantly due to different convergence rates, with some cases requiring more epochs to converge, resulting in higher computational overhead and inefficiency. HyperLLM's ability to effectively extract and integrate hierarchical information facilitates faster convergence, ensuring competitive and stable efficiency. The comprehensive comparison demonstrates that HyperLLM not only surpasses existing LLMs-based frameworks in terms of performance but also offers more stable training efficiency.

## 5.6 Long-Tail Study (Q5)

To investigate the impact of HyperLLM on users with varying levels of sparsity, we divide users in each dataset into five equally sized groups based on the number of interactions, ordered from least to most. These groups represent different levels of sparsity and help evaluate whether HyperLLM addresses the long-tail effect.

As shown in Figure 4, HyperLLM improves the recommender performance of baselines across all groups, whereas the baselines perform worse in the sparse groups. In contrast, HyperLLM's performance remains relatively stable across different groups and does not exhibit a decreasing trend as the groups become sparser, except in the Beauty dataset. Despite the decreasing trend observed in the Beauty dataset, the performance improvements in the sparse groups are much larger than those in the non-sparse groups. Therefore, this anomaly may be attributed to the inherent limitations of hyperbolic space recommender systems for this dataset. Additionally, HyperLLM's performance improvements increase as the groups become sparser in most cases. The substantial improvements in sparse groups highlight HyperLLM's ability to enhance recommendation quality for users with limited interactions.

Figure 4: The Recall@20 results for baselines and baselines with HyperLLM on user groups with different levels of sparsity. G1 to G5 represent user groups 1 to 5, with smaller numbers indicating higher sparsity. The right y-axis represents the percentage improvement of HyperLLM compared to the baseline.

<!-- image -->

This study demonstrates the presence of the long-tail effect and further emphasizes HyperLLM's ability to better address this issue, improving recommender performance where the baselines struggle. By effectively addressing the long-tail problem, HyperLLM ensures more balanced recommendations, catering to the entire user base.

## 5.7 Representation Analysis (Q6)

To validate that HyperLLM learns structural and semantic hierarchical information, we visualize the learned representations via hierarchical clustering. Specifically, we compute the linkage matrices [20] of the representations using the ward variance minimization algorithm to construct dendrograms that reveal their underlying hierarchical structures. This method iteratively merges clusters based on proximity while minimizing variance within each group to optimize cohesion. Consequently, the dendrograms provide visualizations of how representations are hierarchically grouped based on their latent similarities.

Figure 5 presents the dendrograms of representations learned by HyperLLM compared to the baseline HRCF. In these dendrograms, different colors represent distinct clusters derived from the hierarchical structures. HyperLLM's dendrograms exhibit a more diverse and clearly separated color distribution compared to the baseline. Additionally, the branches in HyperLLM's dendrograms are more balanced and show deeper hierarchical levels, suggesting a richer modeling of hierarchies. In contrast, the baseline's clusters are less distinct, and the hierarchical separations are less pronounced. This indicates that the hierarchical clustering for HyperLLM organizes its representations into clusters with more hierarchical structures. These observations confirm that HyperLLM effectively learns the structural and semantic hierarchical information.

Figure 5: Hierarchical clustering visualization of representations for HRCF and HRCF with HyperLLM.

<!-- image -->

## 6 Conclusion

In this paper, we propose a model-agnostic framework, named HyperLLM, which extracts structural and semantic hierarchical information from textual recommender data and effectively integrates it into hyperbolic space recommender systems. Extensive experiments demonstrate that HyperLLM significantly enhances recommender performance, achieves competitive and stable efficiency, and addresses the long-tail effect. Furthermore, the meta-optimized two-phase training strategy in HyperLLM offers a novel perspective for future research on semantic fusion in both Euclidean and hyperbolic space recommender systems.

## References

- [1] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877-1901.
- [2] Ines Chami, Zhitao Ying, Christopher Ré, and Jure Leskovec. 2019. Hyperbolic graph convolutional neural networks. Advances in neural information processing systems 32 (2019).
- [3] Weize Chen, Xu Han, Yankai Lin, Kaichen He, Ruobing Xie, Jie Zhou, Zhiyuan Liu, and Maosong Sun. 2024. Hyperbolic Pre-Trained Language Model. IEEE/ACM Transactions on Audio, Speech, and Language Processing (2024).
- [4] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024).
- [5] Shanshan Feng, Lucas Vinh Tran, Gao Cong, Lisi Chen, Jing Li, and Fan Li. 2020. Hme: A hyperbolic metric embedding approach for next-poi recommendation. In Proceedings of the 43rd International ACM SIGIR Conference on research and development in information retrieval . 1429-1438.
- [6] Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia. 2023. Megablocks: Efficient sparse training with mixture-of-experts. Proceedings of Machine Learning and Systems 5 (2023), 288-304.
- [7] Octavian Ganea, Gary Bécigneul, and Thomas Hofmann. 2018. Hyperbolic neural networks. Advances in neural information processing systems 31 (2018).
- [8] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as language processing (rlp): A unified pretrain, personalized prompt &amp; predict paradigm (p5). In Proceedings of the 16th ACM Conference on Recommender Systems . 299-315.
- [9] Yongjing Hao, Pengpeng Zhao, Jianfeng Qu, Lei Zhao, Guanfeng Liu, Fuzhen Zhuang, Victor S Sheng, and Xiaofang Zhou. 2024. Meta-optimized Structural and Semantic Contrastive Learning for Graph Collaborative Filtering. In 2024 IEEE 40th International Conference on Data Engineering (ICDE) . IEEE, 679-691.
- [10] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval . 639-648.
- [11] Vitor AC Horta, Robin Sobczyk, Maarten C Stol, and Alessandra Mileo. 2023. Semantic Interpretability of Convolutional Neural Networks by Taxonomy Extraction.. In NeSy . 118-127.
- [12] Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, and Wayne Xin Zhao. 2024. Large language models are zero-shot rankers for recommender systems. In European Conference on Information Retrieval . Springer, 364-381.
- [13] Jun Hu, Wenwen Xia, Xiaolu Zhang, Chilin Fu, Weichang Wu, Zhaoxin Huan, Ang Li, Zuoli Tang, and Jun Zhou. 2024. Enhancing sequential recommendation via llm-based semantic embedding learning. In Companion Proceedings of the ACM on Web Conference 2024 . 103-111.
- [14] Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. 1991. Adaptive mixtures of local experts. Neural computation 3, 1 (1991), 79-87.
- [15] Alexandros Karatzoglou and Balázs Hidasi. 2017. Deep learning for recommender systems. In Proceedings of the eleventh ACM conference on recommender systems . 396-397.
- [16] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of naacL-HLT , Vol. 1. Minneapolis, Minnesota.
- [17] Jiaye Lin, Shuang Peng, Zhong Zhang, and Peilin Zhao. 2024. TLRec: A Transfer Learning Framework to Enhance Large Language Models for Sequential Recommendation Tasks. In Proceedings of the 18th ACM Conference on Recommender Systems . 1119-1124.
- [18] Zhongzhou Liu, Hao Zhang, Kuicai Dong, and Yuan Fang. 2024. Collaborative Cross-modal Fusion with Large Language Model for Recommendation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management . 1565-1574.
- [19] Leyla Mirvakhabova, Evgeny Frolov, Valentin Khrulkov, Ivan Oseledets, and Alexander Tuzhilin. 2020. Performance of hyperbolic geometry models on top-N recommendation tasks. In Proceedings of the 14th ACM Conference on Recommender Systems . 527-532.
- [20] Daniel Müllner. 2011. Modern hierarchical, agglomerative clustering algorithms. arXiv preprint arXiv:1109.2378 (2011).
- [21] Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, et al. 2022. Text and code embeddings by contrastive pre-training. arXiv preprint arXiv:2201.10005 (2022).
- [22] Xiuyuan Qin, Huanhuan Yuan, Pengpeng Zhao, Junhua Fang, Fuzhen Zhuang, Guanfeng Liu, Yanchi Liu, and Victor Sheng. 2023. Meta-optimized contrastive learning for sequential recommendation. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval . 89-98.
- [23] Zhida Qin, Wentao Cheng, Wenxing Ding, and Gangyi Ding. 2024. Hyperbolic Graph Contrastive Learning for Collaborative Filtering. IEEE Transactions on Knowledge and Data Engineering (2024).
- [24] Xubin Ren, Wei Wei, Lianghao Xia, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, and Chao Huang. 2024. Representation learning with large language models for recommendation. In Proceedings of the ACM on Web Conference 2024 . 34643475.
- [25] Yankun Ren, Zhongde Chen, Xinxing Yang, Longfei Li, Cong Jiang, Lei Cheng, Bo Zhang, Linjian Mo, and Jun Zhou. 2024. Enhancing sequential recommenders with augmented knowledge from aligned large language models. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval . 345-354.
- [26] Frederic Sala, Chris De Sa, Albert Gu, and Christopher Ré. 2018. Representation tradeoffs for hyperbolic embeddings. In International conference on machine learning . PMLR, 4460-4469.
- [27] Noam Shazeer, *Azalia Mirhoseini, *Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. In International Conference on Learning Representations .
- [28] Jianing Sun, Zhaoyue Cheng, Saba Zuberi, Felipe Pérez, and Maksims Volkovs. 2021. Hgcf: Hyperbolic graph convolution networks for collaborative filtering. In Proceedings of the Web Conference 2021 . 593-601.
- [29] Lucas Vinh Tran, Yi Tay, Shuai Zhang, Gao Cong, and Xiaoli Li. 2020. Hyperml: A boosting metric learning approach in hyperbolic space for recommender systems. In Proceedings of the 13th international conference on web search and data mining . 609-617.
- [30] Chen Wang, Liangwei Yang, Zhiwei Liu, Xiaolong Liu, Mingdai Yang, Yueqing Liang, and Philip S Yu. 2024. Collaborative Alignment for Recommendation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management . 2315-2325.
- [31] Jianling Wang, Haokai Lu, James Caverlee, Ed H Chi, and Minmin Chen. 2024. Large Language Models as Data Augmenters for Cold-Start Item Recommendation. In Companion Proceedings of the ACM on Web Conference 2024 . 726-729.
- [32] Liping Wang, Fenyu Hu, Shu Wu, and Liang Wang. 2021. Fully hyperbolic graph convolution network for recommendation. In Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management . 3483-3487.
- [33] Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, and Chao Huang. 2024. Llmrec: Large language models with graph augmentation for recommendation. In Proceedings of the 17th ACM International Conference on Web Search and Data Mining . 806-815.
- [34] Yunjia Xi, Weiwen Liu, Jianghao Lin, Xiaoling Cai, Hong Zhu, Jieming Zhu, Bo Chen, Ruiming Tang, Weinan Zhang, and Yong Yu. 2024. Towards open-world recommendation with knowledge augmentation from large language models. In Proceedings of the 18th ACM Conference on Recommender Systems . 12-22.
- [35] Jie Xu and Chaozhuo Li. 2023. Geometry Interaction Augmented Graph Collaborative Filtering. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management . 4375-4379.
- [36] Menglin Yang, Aosong Feng, Bo Xiong, Jiahong Liu, Irwin King, and Rex Ying. 2024. Enhancing llm complex reasoning capability through hyperbolic geometry. In ICML 2024 Workshop on LLMs and Cognition .
- [37] Menglin Yang, Zhihao Li, Min Zhou, Jiahong Liu, and Irwin King. 2022. Hicf: Hyperbolic informative collaborative filtering. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining . 2212-2221.
- [38] Menglin Yang, Min Zhou, Jiahong Liu, Defu Lian, and Irwin King. 2022. HRCF: Enhancing collaborative filtering via hyperbolic geometric regularization. In Proceedings of the ACM Web Conference 2022 . 2462-2471.
- [39] Shenghao Yang, Weizhi Ma, Peijie Sun, Qingyao Ai, Yiqun Liu, Mingchen Cai, and Min Zhang. 2024. Sequential recommendation with latent relations based on large language model. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval . 335-344.
- [40] Lu Zhang and Ning Wu. 2024. HGCH: A Hyperbolic Graph Convolution Network Model for Heterogeneous Collaborative Graph Recommendation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management . 3186-3196.
- [41] Sixiao Zhang, Hongxu Chen, Xiao Ming, Lizhen Cui, Hongzhi Yin, and Guandong Xu. 2021. Where are we in embedding spaces?. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &amp; Data Mining . 2223-2231.
- [42] Yiding Zhang, Chaozhuo Li, Xing Xie, Xiao Wang, Chuan Shi, Yuming Liu, Hao Sun, Liangjie Zhang, Weiwei Deng, and Qi Zhang. 2022. Geometric disentangled collaborative filtering. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval . 80-90.
- [43] Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen, and Ji-Rong Wen. 2024. Adapting large language models by integrating collaborative semantics for recommendation. In 2024 IEEE 40th International Conference on Data Engineering (ICDE) . IEEE, 1435-1448.
- [44] Yaochen Zhu, Liang Wu, Qi Guo, Liangjie Hong, and Jundong Li. 2024. Collaborative large language model for recommender systems. In Proceedings of the ACM on Web Conference 2024 . 3162-3172.