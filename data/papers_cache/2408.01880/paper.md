## Walk Wisely on Graph: Knowledge Graph Reasoning with Dual Agents via Efficient Guidance-Exploration

Zijian Wang 1,2 , Bin Wang 1 * , Haifeng Jing 1,3 , Huayu Li 1 , Hongbo Dou 1

1 College of Computer Science and Technology &amp; Qingdao Institute of Software, China University of Petroleum(East China),

China

2 College of Science, China University of Petroleum(East China), China

3

School of Software &amp; Microelectronics, Peking University, China

alococo710@gmail.com, { wangbin,lhyzj } @upc.edu.cn, 2201120005@stu.pku.edu.cn, hongboDou@163.com

## Abstract

Recent years, multi-hop reasoning has been widely studied for knowledge graph (KG) reasoning due to its efficacy and interpretability. However, previous multi-hop reasoning approaches are subject to two primary shortcomings. First, agents struggle to learn effective and robust policies at the early phase due to sparse rewards. Second, these approaches often falter on specific datasets like sparse knowledge graphs, where agents are required to traverse lengthy reasoning paths. To address these problems, we propose a multi-hop reasoning model with dual agents based on hierarchical reinforcement learning (HRL), which is named FULORA . FULORA tackles the above reasoning challenges by e Ffi cient G U idanceExp LORA tion between dual agents. The high-level agent walks on the simplified knowledge graph to provide stagewise hints for the low-level agent walking on the original knowledge graph. In this framework, the low-level agent optimizes a value function that balances two objectives: (1) maximizing return, and (2) integrating efficient guidance from the high-level agent. Experiments conducted on three real-word knowledge graph datasets demonstrate that FULORA outperforms RL-based baselines, especially in the case of longdistance reasoning.

## Introduction

Knowledge graphs (KGs) are designed to represent the world knowledge in a structured way. There are various downstream NLP tasks especially knowledge-driven services, such as query answering (Guu, Miller, and Liang 2015; Cui et al. 2019), relation extraction (Mintz et al. 2009; Reiplinger, Wiegand, and Klakow 2014) and dialogue generation (He et al. 2017). However, a significant proportion of KGs are severely incomplete, which constrains their efficacy in numerous tasks. Consequently, this study concentrates on automatic knowledge graph (KG) reasoning, also as known knowledge graph completion (KGC).

Over recent years, embedding-based models (Bordes et al. 2013; Lin et al. 2015) have effectively preserved KG structural information for single-hop reasoning but lack interpretability. To address this, reinforcement learning (RL) frameworks (Xiong, Hoang, and Wang 2017; Das et al.

* Bin Wang is the corresponding author.

Copyright © 2025, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

Figure 1: An illustrative example of short direct path and long indirect path. When no short direct path exists, the agent searches for a long indirect path.

<!-- image -->

2018) have been introduced to compose single-hop triplets into multi-hop reasoning chains. Recent advances in deep learning (DL) and RL (Wang et al. 2019; Lv et al. 2020; Nikopensius et al. 2023; Wang et al. 2025) further enhance multi-hop reasoning. For instance, AttnPath (Wang et al. 2019) employs attention mechanisms to guide agents, preventing them from stalling at the same node.

Although existing multi-hop reasoning models have achieved impressive results, there is a noteworthy issue with these models that they only perform well when the agent's reasoning path length is short. The drawback that the agent relies heavily on short reasoning chains is fatal in certain datasets, such as sparse KG. Figure 1 is an illustration of short direct path and long indirect path. It's challenging for an agent to identify a short direct path due to the sparsity which makes the relation r 2 do not exist. The optimal path for agent is e 1 → e 2 → e 4 → e 3 we called long indirect path instead of e 1 → e 2 → e 3 . From this point, enhancing the agent's long-distance reasoning ability is one solution to alleviate the multi-hop models' poor performance in the sparse KG.

However, the dimension of the discrete action space at each space is large (Das et al. 2018). As the path length increases, the choices the agent faced will grow exponentially. The most relevant to our work is CURL (Zhang et al. 2022) which proposed a dual-agent framework and mutual reinforcement rewards to assign one of the agents (GIANT) searching on cluster-level paths quickly and providing stagewise hints for another agent (DWARF). Compared to the classic multi-hop model MINERVA (Das et al. 2018), CURL does improve long-distance reasoning ability. But CURL also has two drawbacks, which make CURL's performance inconsistent: (1) Mutual reinforcement reward mechanism forces the low-level agent DWARF to adopt similar policies to the high-level agent GIANT, even if GIANT's policies is not well enough. This may result in false-negative rewards to the intermediate actions which are reasonable. (2) At the early phase, DWARF and GIANT adopt a near-random policy which makes low training efficacy. For briefness, we call the high-level agent as GIANT and the low-level agent as DWARF like (Bai et al. 2022).

In light of these challenges, we propose FULORA, a robust dual-agent framework for KG reasoning with taking full advantage of entity embedding and relation embedding in the KG. FULORA seamlessly makes the selfexploration and path-reliance trade-off of DWARF through a supervised learning method. This unique mechanism will enable DWARF to make its own decisions while receiving meaningful guidance from GIANT, instead of relying entirely on the command of GIANT. Moreover, with the aim to make better use of the entity embedding and relation embedding, FULORA introduces attention mechanism and dynamic path feedback to DWARF and GIANT respectively. Intuitively, GIANT searches a feasible path as soon as possible to guide DWARF to reduce the search space, while DWARF adopts diversified exploration policies in the constrictive search space to prevent over-dependence on GIANT. In this way, DWARF walking on the original KG has both excellent long distance reasoning ability and short direct path utilization ability.

## Related Work

In line with the focus of our work, we provide a brief overview of the background and related work on knowledge graph reasoning.

## Knowledge Graph Embedding

Knowledge graph embedding (KGE) methods map entities to vectors in low-dimensional embedding space, and model relations as transformations between entity embeddings (Bai et al. 2022). Once we map them to low-dimensional dense vector space, we can use modeling methods to perform calculations and reasoning (Ma et al. 2018). Prominent examples include TransE (Bordes et al. 2013), TransR (Lin et al. 2015), ConvE (Dettmers et al. 2018), TuckER (Balazevic, Allen, and Hospedales 2019) and LMKE (Wang et al. 2022), which are equipped with a scoring function that maps any triplet ( e s , r q , e o ) to a scalar score. While KGE is effective at capturing simpler relationships in the graph, such as firstorder adjacency relationships, it struggles with more complex reasoning tasks, particularly those involving multi-hop reasoning (Yao, Mao, and Luo 2019; Wang et al. 2023).

## GNN-based Reasoning

Graph neural networks (GNNs) (Scarselli et al. 2008; Veliˇ ckovi´ c et al. 2017; Xu et al. 2018) are a class of models used for representation learning, specifically designed to encode the structural information of graphs. In the context of link prediction, common frameworks often rely on an auto-encoder formulation, where GNNs generate node embeddings, and edges are predicted as a function of node pairs. These frameworks are inductive when node features are provided, but they become transductive when such features are not available. Another set of frameworks, including SEAL (Zhang and Chen 2018) and GraIL (Teru, Denis, and Hamilton 2020), explicitly encode the subgraph surrounding each node pair for link prediction. Recent advancements in this field include works such as NBFNet (Zhu et al. 2021) and RED-GNN (Zhang and Yao 2022). The former solves the path formulation with learned operators in the generalized Bellman-Ford algorithm while the latter makes use of dynamic programming to recursively encodes multiple rdigraphs with shared edges, and utilizes query-dependent attention mechanism to select the strongly correlated edges. However, these methods primarily focus on the structure of the knowledge graph (KG) itself, specifically the graph structure information of the central node, without addressing how to enhance the ability for long-distance reasoning.

## Multi-hop Reasoning

The advancement of deep reinforcement learning (DRL) has sparked interest in applying DRL to path-finding tasks. The first significant work combining DRL and KG reasoning is DeepPath (Xiong, Hoang, and Wang 2017), which inspired subsequent models, though it requires prior knowledge of the target entity. In contrast, MINERVA (Das et al. 2018) eliminates this requirement, allowing the agent to traverse the knowledge graph until it finds the target. Recent developments have explored the use of powerful neural networks for generating walking policies. M-Walk (Shen et al. 2018) uses an RNN (Elman 1990) to record agent trajectories and optimize rewards with a Monte Carlo Tree Search (MCTS) (Coulom 2006). GRL (Wang et al. 2020) combines GAN(Goodfellow et al. 2014) and LSTM (Hochreiter 1997) to generate new trajectory sequences, enabling the agent to reason not only within the original graph but also in automatically generated sub-graphs, extending relational paths until target entities are found. HMLS (Zheng et al. 2024) improves the generalizability and effectiveness of multi-hop reasoning in few-shot scenarios by exploiting hard relations and hierarchical relation structures. While these models perform well in short-distance reasoning (path length = 3), their performance in long-distance reasoning tasks remains suboptimal. Unlike previous multi-hop reasoning methods, FULORA focuses more on the agents' long-distance reasoning ability and the significant impact of sparse rewards on training efficiency.

## Preliminary

In this section, we commence with the problem definition of our work, followed by the introduction of environment representation.

## Problem Definition

We formally define the research problem of this paper in the following part. The knowledge graph is defined as a directed graph G = {E , R} , where E is a set of all entities and R is a set of all relations. A link triplet l consists of the source entity e s ∈ E , the target entity e o ∈ E and the relation r ∈ R , i.e., l = ( e s , r, e o ) . In the real world, a link triplet corresponds to a fact tuple. For instance, we can represent the fact whale is a mammal as ( whale, is a, mammal ) . We follow the definition in prior graph walking models, that is, query answering (Das et al. 2018; Zhang et al. 2022). In the applications such as searching and query answering, most problems are to infer another entity when we only know the source entity e s and the query relation r q , which can be formed by an incomplete link triplet l m = ( e s , r q , ?) , where '?' indicates the target entity e o is unknown and needs to be found in the KG.

Figure 2: An overview of FULORA framework. ❶ Given a KG G , We first pre-embed the KG using TransE and then apply K-means clustering to generate the cluster-level KG G c . ❷ We design separate policy networks for GIANT and DWARF, using cluster-level KG G c and entity-level KG G e as inputs. The hidden states of GIANT and DWARF h c t and h e t share information, facilitating communication. ❸ To enable DWARF to better leverage the KG structure, we apply the graph attention mechanism and feed the resulting attention vector into the policy network. Dynamic Path Feedback alleviates the near-random policy issue caused by sparse rewards in early training phase, allowing GIANT to provide high-quality guidance to DWARF sooner.

<!-- image -->

## Environment Representation

State. The state consists of two entities-current entity e t and source entity e s -and the query relation. The current entity e t is state-dependent, reflecting the agent's reasoning condition, while the source entity e s and query relation are shared globally. Formally, s t = ( e t , e s , r q ) ∈ S .

Action. The agent interacts with the environment by selecting an action a t ∈ A , corresponding to an outgoing edge of the graph G . Formally, a t = { ( r t +1 , e t +1 ) | ( e t , r t +1 , e t +1 ) ∈ G} . To allow termination, the agent can also stay at the cur- rent entity by selecting a self-loop edge.

Transition. Astate transition occurs when the agent takes an action. The transition function δ : S × A → S is defined as δ ( s t , a t ) , which produces a new state. Multi-hop reasoning typically selects a neighboring entity randomly, as entities often have multiple neighbors connected by the same relation (Xiong, Hoang, and Wang 2017; Das et al. 2018).

Reward. After the agent takes an action, if the corresponding entity is the correct target, it receives a reward of 1; otherwise, it receives a reward of 0.

## Methodology

In this section, we propose FULORA, an extended dualagent framework for knowledge graph reasoning via efficient guidance-exploration. As illustrated in Figure 2, FULORA is able to improve the exploration efficiency of GIANT via dynamic feedback mechanism, so as to provide more reliable guidance for DWARF. In a corresponding manner, DWARF employs a supervised learning approach to achieve a balance between guidance and exploration. Subsequently, it aggregates messages emitted by all neighbors of the current entity via an attention mechanism. Next, we will delve into the specifics of each aforementioned components.

## Embedding Generation

Consistent with (Zhang et al. 2022), we employ TransE (Bordes et al. 2013) due to its efficiency in encoding the structural proximity information of KG to generate pretrained entity embeddings, then we divide the original KG into N clusters by utilizing K-means. In order for GIANT can easily walk on the cluster-level graph G c while preserving the relation information of the original KG G , we add a link to two clusters if there is at least one entity-level edge between them, which is detailed in Appendix E.

## Policy Networks

In our model, we utilize a three-layer LSTM, enabling the agent to memorize and learn from the actions taken before. In contrast to previous models, the necessity arises to design the network separately in this case, given that two agents are walking on the KG. An agent based on LSTM encodes the recursive sequence as a continuous vector h t ∈ R 2 d . Specifically, the hidden state embedding of GIANT is h c t while the hidden state embedding of DWARF is h e t . Their initial hidden state is 0 . In addition, we define an information sharing vector I t = [ h e t ; h c t ] for GIANT and DWARF to share path information. To a certain extent, cluster-level paths are complementary to entity-level paths, as they ensure the sharing of essential path information from GIANT to DWARF.

For GIANT. We denote the current cluster embedding at time step t by c t ∈ R 2 d . The action representation a c t is given by the cluster embedding itself, i.e., a c t = c t ∈ R 2 d because the action corresponds to the next outgoing cluster. The history embedding is updated according to LSTM dynamics:

<!-- formula-not-decoded -->

where W c ∈ R 2 d × 6 d is a projection matrix to maintain shape.

For DWARF. Unlike GIANT, which operates on clusterlevel KGs, DWARF faces greater challenges on the original KG. Entities often have multiple aspects. For instance, a professor may have both professional relations (e.g., worksForUniversity ) and family relations (e.g., spouse ). Additionally, cluster-level paths are typically shorter than entity-level paths, requiring DWARF to have enhanced long-distance reasoning. Consequently, DWARF should prioritize relations and neighbors most relevant to the query. Therefore, we integrate the Graph Attention mechanism (Velickovic et al. 2017) into DWARF. We adopt the same approach as AttnPath (Wang et al. 2019) to obtain the attention vector atn t -1 . DWARF's history embedding h e t can be obtained from

<!-- formula-not-decoded -->

where W e ∈ R 2 d × 7 d is a projection matrix, while the action representation a e t is the concatenation of the relation embedding r t ∈ R d and the end node embedding e t ∈ R d , i.e., a e t = [ r t ; e t ] ∈ R 2 d .

Policy Generation. To predict the next cluster for GIANT and the next entity for DWARF, we apply a two-layer feedforward network on the concatenation of their last LSTM

Figure 3: An illustration of Effective Guidance-Exploration. When DWARF is out of bounds, GIANT guides it to move quickly inside. Otherwise, DWARF prefers to explore for itself to find a correct target.

<!-- image -->

states and current RL state embeddings,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where W c 1 ∈ R 4 d × 4 d , W c 2 ∈ R 2 d × 4 d , W e 1 ∈ R 4 d × 4 d , W e 2 ∈ R 2 d × 4 d are the matrices of learnable weights to maintain dimension of history embedding. While A c t ∈ R | A c t |× 2 d , A e t ∈ R | A e t |× 2 d represent the embeddings of all next possible actions for GIANT and DWARF respectively.

## Efficient Guidance-Exploration

As mentioned above, to address the issue of sparse rewards and a large action space, we aim for GIANT to guide DWARF in reducing the action space. However, GIANT's guidance is not always beneficial due to two key issues:

- Poor guidance. Poor guidance can lead DWARF to incorrect answers.
- Policy shift. A policy which is suitable for GIANT may not be fully applicable to DWARF.

Considering the two issues, we propose an efficient guidance-exploration approach, which gives DWARF a constraint reward to balance guidance and exploration via supervised learning approaches.

Constraint Reward. Considering the constraint, that is, receiving high-quality guidance from GIANT, we introduce a metric for the state similarity between GIANT and DWARF, denoted as Sim ( s c t , s e t ) . It is calculated as the cosine similarity between the pre-trained embeddings of the current cluster 1 and the current entity:

<!-- formula-not-decoded -->

1 Each cluster embedding is obtained by averaging all entity embeddings within it. A slight abuse of notation indicates that the c t here is not the c t mentioned in the policy network. In the concrete implementation, we treat the c t in the policy network as a selfcascade of the c t here.

We solve the optimization problem by the following formulas,

<!-- formula-not-decoded -->

where r e ( s e t ) and r c ( s c t ) are default rewards for DWARF and GIANT respectively, the agent obtains a favorable reward 1 if the corresponding entity or cluster is a correct target and unfavorable reward 0 otherwise. It is only when GIANT reaches the correct cluster that the constraint in Equation 6 is operative. In practice, we set ε = -0 . 01 δ .

Practical Algorithm. The objective in Equation 6 can be optimized with any reinforcement learning algorithm that implements generalized policy iteration. Here we use REINFORCE (Williams 1992) and the method of Lagrange multipliers as described by (Abdolmaleki et al. 2018; Grillotti et al. 2024). For all DWARF state s e t , we maximize the Lagrangian function, subject to 0 ≤ λ ( s e t ) ≤ 1 ,

<!-- formula-not-decoded -->

the Lagrange multiplier is updated to make the guidanceexploration trade-off. Figure 3 gives an illustration: When the state similarity metric between GIANT and DWARF Sim ( s c t , s e t ) is less than the threshold, the parameter θ λ are optimized so that λ ( s e t ) increases to encourage GIANT to provide more guidance for DWARF. Conversely, the parameter θ λ are optimized so that λ ( s e t ) decreases to encourage DWARF to explore in the constrictive space when the state similarity metric between GIANT and DWARF Sim ( s c t , s e t ) is greater than the threshold. In practice, we utilize a crossentropy loss to optimize θ λ :

<!-- formula-not-decoded -->

## Dynamic Path Feedback

A further challenge is to enhance the search efficiency of GIANT in order to provide DWARF with the requisite guidance as expeditiously as possible, given that the search is limited to a fixed number of step T . In the default reward, GIANT will receive 1 reward only when it reaches the correct cluster, and the rest will be 0, which makes GIANT adopt random policy at the early phase. This phenomenon is not conducive to stable learning outcomes for two agents. On the basis of the theory of reward shaping (Harutyunyan et al. 2015), we rewrite the reward function of GIANT, named dynamic path feedback. In particular, we write it in the form of an objective function J ( θ π c ) ,

Table 1: The statistics of some benchmark KG datasets. #Mean is the averaged outgoing degree of every entity that can indicate the sparsity level while #Med is the corresponding median.

| Dataset   |   #Ent |   #Rel |   #Fact |   #Que |   #Mean |   #Med |
|-----------|--------|--------|---------|--------|---------|--------|
| NELL-995  | 75,492 |    200 | 154,213 |  3,992 |    4.07 |      1 |
| WN18RR    | 40,945 |     11 |  86,835 |  3,134 |    2.19 |      2 |
| FB15K-237 | 14,505 |    237 | 272,115 | 20,466 |   19.74 |     14 |

<!-- formula-not-decoded -->

s c t +1 comes from the next state generated by the policy network. In contrast to the default reward, dynamic path feedback uses the reward function to score the GIANT's path in a rollout rather than simply identifying whether it has reached the correct target cluster. Even if GIANT does not reach the correct target cluster in a rollout, it will evaluate the quality of the path, thereby accelerating the learning process. In Appendix D, we prove that GIANT learns optimal policy in dynamic path feedback is consistent with default rewards circumstance:

Theorem 1 ( Consistency of optimal policy ) Given two MDPs that differ only in reward function, denoted as M = ( S , A , δ, R ) and M ′ = ( S , A , δ, R D ) respectively, where R = r c ( s c t ) is the default reward while R D = r c ( s c t ) -α ∆( s c t , s c t +1 ) is the dynamic path feedback reward. Their optimal policies are consistent, that is,

<!-- formula-not-decoded -->

## Experiments

In this section, we evaluate the efficacy of FULORA on three real-world KG datasets: NELL-995 (Xiong, Hoang, and Wang 2017), WN18RR (Dettmers et al. 2018) and FB15K237 (Toutanova et al. 2015). The datasets statistics are listed in Table 1.

As can be seen from the statistical indicators, these three datasets represent standard KG, sparse KG and Dense KG respectively. In our selection of the baseline, we are not only evaluating it against the state-of-the-art multi-hop reasoning methods, but also against with other embedding-based KG reasoning methods and GNN-based reasoning methods. FULORA and all baselines are implemented under the Pytorch framework and run on the NVIDIA 3080Ti GPU. All the results are the average of the results in five experiments. See Appendix A.1 for a brief introduction of each baseline.

## Short-distance Reasoning Ability

For each triplet ( e s , r q , e o ) in the test set, we convert it to a query ( e s , r q , ?) and use embedding-based or multi-hop models with a beam search width of 50 to rank the tail entities. Following (Bordes et al. 2013), we evaluate using two metrics: (1) mean reciprocal rank (MRR) and (2) Hits@K, the proportion of correct tail entities ranked in the top K.

Table 2: Link prediction results with a path length of 3 on the NELL, WordNet, and Freebase datasets (embedding-based and GNN-based reasoning do not inherently consider path length, so we exclude triples reachable by shorter paths for fair comparison). All metrics are multiplied by 100. The best score of embedding-based reasoning models, GNN-based models are underlined while multi-hop reasoning models are in bold . Compared to other indicators, we specifically highlight the MRR to emphasize its significance.

| Model                           | NELL-995   | NELL-995   | NELL-995   | NELL-995   | WN18RR   | WN18RR   | WN18RR   | WN18RR   | FB15K-237   | FB15K-237   | FB15K-237   | FB15K-237   |
|---------------------------------|------------|------------|------------|------------|----------|----------|----------|----------|-------------|-------------|-------------|-------------|
|                                 | MRR        | @1         | @3         | @10        | MRR      | @1       | @3       | @10      | MRR         | @1          | @3          | @10         |
| TransE (Bordes et al. 2013)     | 51.4       | 45.6       | 67.8       | 75.1       | 35.9     | 28.9     | 46.4     | 53.4     | 36.1        | 24.8        | 40.1        | 45.0        |
| DistMult (Yang et al. 2015)     | 68.0       | 61.0       | 73.3       | 79.5       | 43.3     | 41.0     | 44.1     | 47.5     | 37.0        | 27.5        | 41.7        | 56.8        |
| ComplEx (Trouillon et al. 2016) | 68.4       | 61.2       | 76.1       | 82.1       | 41.5     | 38.2     | 43.3     | 48.0     | 39.4        | 30.3        | 43.4        | 57.2        |
| LMKE (Wang et al. 2022)         | 74.6       | 71.7       | 84.7       | 89.5       | 53.6     | 45.1     | 66.0     | 79.4     | 41.2        | 31.8        | 46.2        | 56.9        |
| NBFNet (Zhu et al. 2021)        | 70.9       | 68.0       | 76.8       | 80.4       | 48.1     | 44.9     | 49.8     | 58.3     | 40.5        | 30.8        | 45.8        | 55.2        |
| RED-GNN (Zhang and Yao 2022)    | 71.2       | 67.8       | -          | 86.2       | 46.9     | 42.5     | -        | 53.0     | 39.8        | 29.9        | -           | 54.4        |
| MINERVA (Das et al. 2018)       | 67.5       | 58.8       | 74.6       | 81.3       | 44.8     | 41.3     | 45.6     | 51.3     | 27.1        | 19.2        | 30.7        | 42.6        |
| AttnPath (Wang et al. 2019)     | 69.3       | 62.7       | 73.9       | 80.1       | 42.9     | 40.7     | 44.3     | 52.9     | 31.9        | 24.1        | 40.4        | 43.8        |
| SQUIRE (Bai et al. 2022)        | 71.1       | 68.2       | 81.4       | 87.2       | 48.2     | 45.0     | 51.4     | 59.7     | 35.0        | 26.5        | 41.7        | 50.3        |
| CURL (Zhang et al. 2022)        | 70.8       | 66.7       | 78.6       | 84.3       | 46.0     | 42.9     | 47.1     | 52.3     | 30.6        | 23.9        | 38.1        | 50.9        |
| HMLS (Zheng et al. 2024)        | 71.8       | 69.0       | 80.9       | 88.9       | 48.5     | 43.9     | 52.9     | 60.4     | 37.4        | 28.0        | 42.1        | 52.5        |
| FULORA (Ours)                   | 72.5       | 69.4       | 79.7       | 89.2       | 49.1     | 45.6     | 50.7     | 59.2     | 36.4        | 27.1        | 41.9        | 51.3        |

As shown in Table 2, we present the performance of FULORA and all baselines on NELL-995, WN18RR, and FB15K-237. On both the standard KG (NELL-995) and sparse KG (WN18RR), FULORA not only outperforms CURL, SQUIRE and HMLS currently regarded as the most powerful multi-hop reasoning models, but also significantly surpasses NBFNet and RED-GNN, both of which are recognized as advanced GNN-based reasoning algorithms. FULORA also achieves comparable performance to the best embedding-based model, LMKE. On FB15K-237, where 1-to-M relations dominate, multi-hop models often struggle with high-degree nodes, hindering correct entity retrieval. In contrast, FULORA's attention mechanism focuses on the most relevant neighbors. Averaging across three datasets, FULORA improves MRR and Hits@1, 3, and 10 by 3.5%, 2.9%, 2.8%, and 4.1% over CURL. Notably, Table 2 presents results based on a path length of 3, which does not capture FULORA's exceptional performance in long-distance reasoning. Subsequent experiments focus on the performance of FULORA's components, GIANT and DWARF, in long-distance reasoning, with comparisons to other advanced models in Table 2. In fact prediction, FULORA outperforms other multi-hop baselines, as detailed in Appendix B.1.

## Long-distance Reasoning Ability

Recall the motivation of FULORA, we tend to address the issue of multi-hop reasoning models being unable to infer correct answer due to the lack of short direct path by improving long-distance reasoning ability. This issue is significant on standard KG and sparse KG, so we conduct the following experiments on NELL-995 and WN18RR. For effective evalu- ation, we compare our model with MINERVA and CURL in NELL-995 and WN18RR, where we remove the most frequently-visited short paths found by the bi-directional search (Xiong, Hoang, and Wang 2017; Zhang et al. 2022) inside KGs.

Figure 4: Learning curves comparing the performance of ours against CURL from NELL-995 relation tasks and all tasks. Averaged over 5 seeds with the shaded area showing standard deviation. Our proposed model is significantly better than CURL in both score and stability.

<!-- image -->

Cluster State Similarity. One of the most significant contributions of GIANT is the implementation of dynamic path feedback, which has been shown to enhance the learning efficiency. To visually demonstrate the efficacy of dynamic path feedback, we initially focus on GIANT's reasoning ability on cluster-level, which directly impacts DWARF's reasoning results. Here, we record the cluster state similarity (CSS) Sim ( s c T , s c target ) under each epoch. As shown in Figure 4, our model outperforms CURL in scores and stability in most cases. See Appendix B.3 for a comparison of the remaining tasks.

Table 3: Effects of various pre-embedding methods on the performance of FULORA.

| NELL-995/WN18RR/FB15K-237            | MRR            | @1             | @10            |
|--------------------------------------|----------------|----------------|----------------|
| FULORA + TransE (Bordes et al. 2013) | 72.5/49.1/36.4 | 69.4/45.6/27.1 | 89.2/57.2/51.3 |
| FULORA + LMKE (Wang et al. 2022)     | 76.8/54.4/38.6 | 73.0/46.3/30.2 | 93.7/72.4/51.2 |

Figure 5: The long-distance performance: FULORA significantly outpeforms CURL, SQUIRE, HMLS, LMKE, REDGNN, NBFNet on NELL-995 (standrad KG) and WN18RR (Sparse KG).

<!-- image -->

Entity Reasoning Accuracy. we now demonstrate that FULORA's efficient guidance-exploration enhances DWARF's performance in long-distance reasoning compared to other well-performed baselines (Appendix B.4.2 provides a further analysis). Figure 5 shows the MRR for varying path lengths on NELL-995 and WN18RR. FULORA excels in long-distance reasoning accuracy on WN18RR (sparse KG). This highlights our motivation: FULORA mitigates the poor performance of current KG reasoning methods in the absence of short direct paths by enhancing long-distance reasoning. Overall, FULORA demonstrates more robust long-distance reasoning performance with minimal degradation in long-path settings. This is due to the dynamic path feedback and efficient guidanceexploration, which enable better information sharing between GIANT and DWARF, ensuring DWARF's excellent self-exploration ability.

## Ablation Study

Sensitivity Analysis. We utilize δ and α to control the degree of efficient guidance-exploration and dynamic path feedback. To comprehensively explore the efficacy of the two mechanisms, we conduct link prediction on NELL995 and WN18RR by varying the values of δ and α as { 0 . 20 , 0 . 30 , 0 . 40 , 0 . 50 } and { 0 . 05 , 0 . 10 , 0 . 15 , 0 . 20 } , respectively. The averaged results are shown in Figure 6. In accordance with our previous analysis, it is evident that DWARF cannot rely excessively on GIANT for guidance. Furthermore, the level of path feedback that GIANT receives should not be unduly strong. In addition, in Appendix B.4, we have carefully discussed the influence of three main components of FULORA on KG reasoning efficiency.

Figure 6: Ablation analysis w.r.t δ and α on NELL-995 and WN18RR, respectively. Here we report MRR results with path length=3 for both datasets.

<!-- image -->

Pre-embedding Methods. In our approach, we use TransE (Bordes et al. 2013) for pre-training the embeddings. One straightforward idea is to improve FULORA's performance by using a more advanced KG embedding model such as LMKE (Wang et al. 2022). Table 3 presents the performance of TransE and LMKE as pre-embedding methods. The more advanced LMKE indeed improves FULORA's performance. However, we use TransE in our experiments to emphasize that the performance improvement is due to FULORA's efficient guidance-exploration mechanism, not the strength of the KG embedding method.

## Conclusion

We present FULORA, an efficient guidance-exploration model built on dual-agent KG reasoning framework to enhance the agent's long-distance reasoning ability on standard KG and sparse KG. The key insight behind our approach is balancing the self-exploration of DWARF and the guidance from GIANT. Specifically, on the one hand, we leverage the attention mechanism to make DWARF pay attention to the neighbouring entities that are close to the query. On the other hand, we propose that dynamic path feedback enables GIANT to have better learning efficiency, thus providing DWARF with high-quality guidance, making the DWARF to have a favourable global vision while having excellent local reasoning ability. Experiments on three real-world datasets demonstrate that FULORA outperforms state-of-the-art multi-hop reasoning methods. Further analysis reveals that FULORA's long-distance reasoning ability on standard and sparse KGs significantly outperforms current KG reasoning methods.

Abdolmaleki, A.; Springenberg, J. T.; Tassa, Y.; Munos, R.; Heess, N.; and Riedmiller, M. A. 2018. Maximum a Posteriori Policy Optimisation. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings . OpenReview.net.

Bai, Y.; Lv, X.; Li, J.; Hou, L.; Qu, Y.; Dai, Z.; and Xiong, F. 2022. SQUIRE: A Sequence-to-sequence Framework for Multi-hop Knowledge Graph Reasoning. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022 , 1649-1662.

Balazevic, I.; Allen, C.; and Hospedales, T. M. 2019. TuckER: Tensor Factorization for Knowledge Graph Completion. In Inui, K.; Jiang, J.; Ng, V.; and Wan, X., eds., Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019 , 5184-5193. Association for Computational Linguistics.

Bellman, R. 1958. Dynamic programming and stochastic control processes. Information and control , 1(3): 228-239.

Bordes, A.; Usunier, N.; Garc´ ıa-Dur´ an, A.; Weston, J.; and Yakhnenko, O. 2013. Translating Embeddings for Modeling Multi-relational Data. In Burges, C. J. C.; Bottou, L.; Ghahramani, Z.; and Weinberger, K. Q., eds., Advances in Neural Information Processing Systems 26: 27th Annual Conference on Neural Information Processing Systems 2013. Proceedings of a meeting held December 5-8, 2013, Lake Tahoe, Nevada, United States , 2787-2795.

Cheung, Y.-W.; and Lai, K. S. 1995. Lag order and critical values of the augmented Dickey-Fuller test. Journal of Business &amp; Economic Statistics , 13(3): 277-280.

Coulom, R. 2006. Efficient selectivity and backup operators in Monte-Carlo tree search. In International conference on computers and games , 72-83. Springer.

Cui, W.; Xiao, Y.; Wang, H.; Song, Y.; Hwang, S.; and Wang, W. 2019. KBQA: Learning Question Answering over QA Corpora and Knowledge Bases. CoRR .

Das, R.; Dhuliawala, S.; Zaheer, M.; Vilnis, L.; Durugkar, I.; Krishnamurthy, A.; Smola, A.; and McCallum, A. 2018. Go for a Walk and Arrive at the Answer: Reasoning Over Paths in Knowledge Bases using Reinforcement Learning. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings . OpenReview.net.

Dettmers, T.; Minervini, P.; Stenetorp, P.; and Riedel, S. 2018. Convolutional 2D Knowledge Graph Embeddings. In McIlraith, S. A.; and Weinberger, K. Q., eds., Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI-18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018 , 18111818. AAAI Press.

Diks, C.; and Panchenko, V. 2006. A new statistic and practical guidelines for nonparametric Granger causality testing. Journal of Economic Dynamics and Control , 30(9-10): 1647-1669.

Dougherty, C. 2011. Introduction to econometrics . Oxford university press, USA.

Elman, J. L. 1990. Finding structure in time. Cognitive science , 14(2): 179-211.

Goodfellow, I.; Pouget-Abadie, J.; Mirza, M.; Xu, B.; Warde-Farley, D.; Ozair, S.; Courville, A.; and Bengio, Y. 2014. Generative adversarial nets. Advances in neural information processing systems , 27.

Grillotti, L.; Faldor, M.; Le´ on, B. G.; and Cully, A. 2024. Quality-Diversity Actor-Critic: Learning High-Performing and Diverse Behaviors via Value and Successor Features Critics. CoRR .

Guu, K.; Miller, J.; and Liang, P. 2015. Traversing knowledge graphs in vector space. arXiv preprint arXiv:1506.01094 .

Harutyunyan, A.; Devlin, S.; Vrancx, P.; and Now´ e, A. 2015. Expressing Arbitrary Reward Functions as Potential-Based Advice. In Bonet, B.; and Koenig, S., eds., Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, January 25-30, 2015, Austin, Texas, USA , 26522658. AAAI Press.

He, H.; Balakrishnan, A.; Eric, M.; and Liang, P. 2017. Learning Symmetric Collaborative Dialogue Agents with Dynamic Knowledge Graph Embeddings. In Barzilay, R.; and Kan, M., eds., Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers , 1766-1776. Association for Computational Linguistics.

Ho, M. S.; and Sørensen, B. E. 1996. Finding cointegration rank in high dimensional systems using the Johansen test: an illustration using data based Monte Carlo simulations. The Review of Economics and Statistics , 726-732.

Hochreiter, S. 1997. Long Short-term Memory. Neural Computation MIT-Press .

Lao, N.; and Cohen, W. W. 2010. Relational retrieval using a combination of path-constrained random walks. Mach. Learn. , 81(1): 53-67.

Lin, Y.; Liu, Z.; Sun, M.; Liu, Y.; and Zhu, X. 2015. Learning Entity and Relation Embeddings for Knowledge Graph Completion. In Bonet, B.; and Koenig, S., eds., Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, January 25-30, 2015, Austin, Texas, USA , 21812187. AAAI Press.

Lv, X.; Han, X.; Hou, L.; Li, J.; Liu, Z.; Zhang, W.; Zhang, Y.; Kong, H.; and Wu, S. 2020. Dynamic Anticipation and Completion for Multi-Hop Reasoning over Sparse Knowledge Graph. In Webber, B.; Cohn, T.; He, Y.; and Liu, Y., eds., Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020 , 5694-5703. Association for Computational Linguistics.

Ma, T.; Shao, W.; Hao, Y.; and Cao, J. 2018. Graph classification based on graph set reconstruction and graph kernel feature reduction. Neurocomputing , 296: 33-45.

Mintz, M.; Bills, S.; Snow, R.; and Jurafsky, D. 2009. Distant supervision for relation extraction without labeled data. In Su, K.; Su, J.; and Wiebe, J., eds., ACL 2009, Proceedings of the 47th Annual Meeting of the Association for Computational Linguistics and the 4th International Joint Conference on Natural Language Processing of the AFNLP, 2-7 August 2009, Singapore , 1003-1011. The Association for Computer Linguistics.

Ng, A. Y.; Harada, D.; and Russell, S. 1999. Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping. In Bratko, I.; and Dzeroski, S., eds., Proceedings of the Sixteenth International Conference on Machine Learning (ICML 1999), Bled, Slovenia, June 27 - 30, 1999 , 278-287. Morgan Kaufmann.

Nikopensius, G.; Mayank, M.; Phukan, O. C.; and Sharma, R. 2023. Reinforcement Learning-based Knowledge Graph Reasoning for Explainable Fact-checking. In Proceedings of the International Conference on Advances in Social Networks Analysis and Mining, ASONAM 2023, Kusadasi, Turkey, November 6-9, 2023 , 164-170. ACM.

Reiplinger, M.; Wiegand, M.; and Klakow, D. 2014. Relation Extraction for the Food Domain without Labeled Training Data - Is Distant Supervision the Best Solution? In Przepi´ orkowski, A.; and Ogrodniczuk, M., eds., Advances in Natural Language Processing - 9th International Conference on NLP, PolTAL 2014, Warsaw, Poland, September 17-19, 2014. Proceedings , volume 8686 of Lecture Notes in Computer Science , 345-357. Springer.

Scarselli, F.; Gori, M.; Tsoi, A. C.; Hagenbuchner, M.; and Monfardini, G. 2008. The graph neural network model. IEEE transactions on neural networks , 20(1): 61-80.

Shen, Y.; Chen, J.; Huang, P.; Guo, Y.; and Gao, J. 2018. M-Walk: Learning to Walk over Graphs using Monte Carlo Tree Search. In Bengio, S.; Wallach, H. M.; Larochelle, H.; Grauman, K.; Cesa-Bianchi, N.; and Garnett, R., eds., Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montr´ eal, Canada , 6787-6798.

Teru, K.; Denis, E.; and Hamilton, W. 2020. Inductive relation prediction by subgraph reasoning. In International Conference on Machine Learning , 9448-9457. PMLR.

Toutanova, K.; Chen, D.; Pantel, P.; Poon, H.; Choudhury, P.; and Gamon, M. 2015. Representing Text for Joint Embedding of Text and Knowledge Bases. In M` arquez, L.; Callison-Burch, C.; Su, J.; Pighin, D.; and Marton, Y., eds., Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, EMNLP 2015, Lisbon, Portugal, September 17-21, 2015 , 1499-1509. The Association for Computational Linguistics.

Trouillon, T.; Welbl, J.; Riedel, S.; Gaussier, ´ E.; and Bouchard, G. 2016. Complex Embeddings for Simple Link Prediction. In Balcan, M.; and Weinberger, K. Q., eds., Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016 , volume 48 of JMLR Workshop and Conference Proceedings , 2071-2080. JMLR.org.

Veliˇ ckovi´ c, P.; Cucurull, G.; Casanova, A.; Romero, A.; Lio, P.; and Bengio, Y. 2017. Graph attention networks. arXiv preprint arXiv:1710.10903 .

Velickovic, P.; Cucurull, G.; Casanova, A.; Romero, A.; Li` o, P.; and Bengio, Y. 2017. Graph Attention Networks. CoRR , abs/1710.10903.

Wang, H.; Li, S.; Pan, R.; and Mao, M. 2019. Incorporating Graph Attention Mechanism into Knowledge Graph Reasoning Based on Deep Reinforcement Learning. In Inui, K.; Jiang, J.; Ng, V.; and Wan, X., eds., Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019 , 2623-2631. Association for Computational Linguistics.

Wang, Q.; Ji, Y.; Hao, Y.; and Cao, J. 2020. GRL: Knowledge graph completion with GAN-based reinforcement learning. Knowl. Based Syst. , 209: 106421.

Wang, X.; He, Q.; Liang, J.; and Xiao, Y. 2022. Language Models as Knowledge Embeddings. In Raedt, L. D., ed., Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022 , 2291-2297. ijcai.org.

Wang, Y.; Ouyang, X.; Guo, D.; and Zhu, X. 2023. MEGA: Meta-graph augmented pre-training model for knowledge graph completion. ACM Transactions on Knowledge Discovery from Data , 18(1): 1-24.

Wang, Z.; Wang, B.; Dou, H.; and Liu, Z. 2025. Windows deep transformer Q-networks: an extended variance reduction architecture for partially observable reinforcement learning. Applied Intelligence , 55(1): 35.

Williams, R. J. 1992. Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Mach. Learn. , 229-256.

Xiong, W.; Hoang, T.; and Wang, W. Y. 2017. DeepPath: A Reinforcement Learning Method for Knowledge Graph Reasoning. In Palmer, M.; Hwa, R.; and Riedel, S., eds., Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, EMNLP 2017, Copenhagen, Denmark, September 9-11, 2017 , 564-573. Association for Computational Linguistics.

Xu, K.; Hu, W.; Leskovec, J.; and Jegelka, S. 2018. How powerful are graph neural networks? arXiv preprint arXiv:1810.00826 .

Yang, B.; Yih, W.; He, X.; Gao, J.; and Deng, L. 2015. Embedding Entities and Relations for Learning and Inference in Knowledge Bases. In Bengio, Y.; and LeCun, Y., eds., 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings .

Yao, L.; Mao, C.; and Luo, Y. 2019. KG-BERT: BERT for knowledge graph completion. arXiv preprint arXiv:1909.03193 .

Zhang, D.; Yuan, Z.; Liu, H.; Lin, X.; and Xiong, H. 2022. Learning to Walk with Dual Agents for Knowledge Graph Reasoning. In Thirty-Sixth AAAI Conference on Artificial Intelligence, AAAI 2022, Thirty-Fourth Conference on Innovative Applications of Artificial Intelligence, IAAI 2022, The Twelveth Symposium on Educational Advances in Artificial Intelligence, EAAI 2022 Virtual Event, February 22 - March 1, 2022 , 5932-5941. AAAI Press.

Zhang, M.; and Chen, Y. 2018. Link prediction based on graph neural networks. Advances in neural information processing systems , 31.

Zhang, Y.; and Yao, Q. 2022. Knowledge graph reasoning with relational digraph. In Proceedings of the ACM web conference 2022 , 912-924.

Zheng, S.; Chen, W.; Wang, W.; Zhao, P.; Yin, H.; and Zhao, L. 2024. Multi-Hop Knowledge Graph Reasoning in Few-Shot Scenarios. IEEE Trans. Knowl. Data Eng. , 36(4): 1713-1727.

Zhu, Z.; Zhang, Z.; Xhonneux, L.-P.; and Tang, J. 2021. Neural bellman-ford networks: A general graph neural network framework for link prediction. Advances in Neural Information Processing Systems , 34: 29476-29490.

## A Experiment Setup

## A.1 Baseline Methods

We compare the performance of FULORA with the following baselines, including embedding-based, GNN-based and multi-hop based KG reasoning methods:

1. TransE (Bordes et al. 2013) embeds entities and relations into low-dimensional vector space so that complex relationships in graphs can be represented and reasoned by vector operations.
2. DistMult (Yang et al. 2015) uses triples (source entity, relation, tail entity) to train the model, where each entity and relation is represented as a vector. The target is to maximize the score of the correct triples while minimizing the score of the wrong triples.
3. ComplEx (Trouillon et al. 2016) is based on tensor decomposition and uses complex vectors to represent entities and relationships in order to capture complex relationships in the knowledge graph.
4. DeepPath (Xiong, Hoang, and Wang 2017) utilizes a knowledge graph-based embedded policy-based agent with continuous states, which extends its path in a KG vector space by sampling the most promising relationships.
5. MINERVA (Das et al. 2018) formulates the query task as a reinforcement learning (RL) problem where the goal is to take the best sequence of decisions (choice of relation edges) to maximize the expected reward (reaching the correct answer node).
6. AttnPath (Wang et al. 2019) makes use of attention mechanism to force an agent to walk forward every step to avoid the agent stalling at the same entity node constantly.
7. NBFNet (Zhu et al. 2021) parameterizes the generalized Bellman-Ford algorithm with 3 neural components, namely INDICATOR, MESSAGE and AGGREGATE functions, which corresponds to the boundary condition, multiplication operator, and summationoperator respectively
8. RED-GNN (Zhang and Yao 2022) makes use of dynamic programming to recursively encodes multiple r-digraphs with shared edges, and utilizes query-dependent attention mechanism to select the strongly correlated edges.
9. SQUIRE (Bai et al. 2022) utilizes an encoder-decoder Transformer structure to translate the query to a path, without relying on existing edges to generate the path.
10. LMKE (Wang et al. 2022) adopts Language Models to derive Knowledge Embeddings (KE) and formulate description-based KE learning with a contrastive learning framework to improve efficiency in training and evaluation.
11. CURL (Zhang et al. 2022) trains two agents (GIANT and DWARF) to walk over a knowlegde graph jointly and search for the answer collaboratively.
12. HMLS (Zheng et al. 2024) improves the generalizability and effectiveness of multi-hop reasoning in few-shot

Table 4: FULORA Hyperparameters on three KG datasets.

| Hyperparameter   | NELL-995   | WN18RR   | FB15K-237   |
|------------------|------------|----------|-------------|
| Embedding size   | 50         | 50       | 50          |
| Hidden size      | 50         | 50       | 50          |
| Batch size       | 128        | 256      | 512         |
| Learning rate    | 0.01       | 0.001    | 0.001       |
| Optimizer        | Adam       | Adam     | Adam        |
| Cluster number   | 75         | 100      | 200         |
| Beam search size | 100        | 50       | 100         |
| α                | 0.15       | 0.15     | 0.25        |
| δ                | 0.20       | 0.40     | 0.30        |

scenarios by exploiting hard relations and hierarchical relation structures.

## A.2 Data Statistics

We adopt three KG datasets with different scales: NELL995 (standard KG), WN18RR (sparse KG) and FB15K-237 (dense KG). We give a brief overview of these datasets.

NELL-995 (Xiong, Hoang, and Wang 2017) is an open source machine learning dataset developed by the OpenAI research group that contains more than 950,000 pieces of entity relationship data collected from the network to help machine learning systems make inferences. NELL995 can be used to train machine learning models, such as natural language processing models, machine translation models, question answering systems, and semantic search systems.

WN18RR (Dettmers et al. 2018) is a subset of WordNet that describes the association characteristics between English words. It preserves the symmetry, asymmetry, and composition relationships of the WordNet, and removes the inversion relationships. WN18RR contains some relational information about words. It consists of 14,541 entities and 237 relationships.

FB15K-237 (Toutanova et al. 2015) is a common knowledge graph dataset, which is a subset extracted from Freebase knowledge graph. It contains 14,505 entities and 237 relationships, and the data is carefully processed to remove reversible relational data and trivial triples, ensuring that entities in the training set are not directly connected to the verification or test set, thus avoiding information leakage issues.

## A.3 Implementation Details

We use the following software versions:

- Ubuntu 24.04 LTS
- Python 3.10
- Pytorch 2.0.1

We conduct all experiments with a single NVIDIA GeForce 3080Ti GPU. Our experiments of all of 12 baselines are conducted on their official implementation provided by their respective authors. To reproduce the results of our model in Table 2 and Table 3, we report the empirically optimal crucial hyperparameters as shown in Table 4. On all datasets, the quantities of path rollouts in training and testing are 20 and 100, separatively. The core codes of FULORA are available at: https://github.com/KotoHanon/LOCOCO.

Table 5: Fact prediction results on seven tasks from NELL-995. All metrics are multiplied by 100. The best score of all models is in bold .

| Task                   |   TransE |   TransR |   PRA |   DeepPath |   MINERVA |   M-Walk |   CURL |   FULORA |
|------------------------|----------|----------|-------|------------|-----------|----------|--------|----------|
| PersonBornInLocation   |     62.7 |     67.3 |  54.7 |       75.1 |      80.0 |     84.7 |   82.7 |     84.4 |
| OrgHeadquarteredInCity |     62.0 |     65.7 |  81.1 |       79.0 |      94.0 |     94.3 |   94.8 |     95.1 |
| AthletePlaysForTeam    |     62.7 |     67.3 |  54.7 |       75.0 |      80.3 |     84.7 |   82.9 |     86.0 |
| AthletePlaysInLeague   |     77.3 |     91.2 |  84.1 |       96.0 |      94.2 |     96.1 |   97.1 |     97.3 |
| AthletePlaysSport      |     87.6 |     96.3 |  47.4 |       95.7 |      98.0 |     98.3 |   98.4 |     98.5 |
| TeamPlaysSport         |     76.1 |     81.4 |  79.1 |       73.8 |      88.0 |     88.4 |   88.7 |     90.2 |
| WorksFor               |     67.7 |     69.2 |  68.1 |       71.1 |      81.0 |     83.2 |   82.1 |     84.3 |

## B Additional Results and Analysis B.1 Fact Prediction Results

As opposed to link prediction, fact prediction task is concerned with verifying the veracity of an unknown fact, the true test triplets are ranked with some generated false triplets. Since we share a similar query-answering mechanism as CURL (Zhang et al. 2022), FULORA is capable of identifying the most appropriate entity for a given query and eliminates the need to evaluate negative samples of any particular relation. In the experiments, the dual agents try to infer and walk through the cluster-level and entity-level respectively to reach the correct target under removing all links of groundtruth relations in the original KG. Here, we report Mean Average Precision (MAP) scores for various relation tasks of NELL-995. We reuse the results of TransE (Bordes et al. 2013), TransR (Lin et al. 2015), PRA (Lao and Cohen 2010), DeepPath (Xiong, Hoang, and Wang 2017), MINERVA (Das et al. 2018), M-Walk (Shen et al. 2018) on seven tasks already reported in (Zhang et al. 2022). As demonstrated in Table 5, FULORA produces a satisfying result in most tasks, contributing an average gain of 9.1% relative to the multi-hop based reasoning approaches (PRA, DeepPath, MINERVA, M-Walk and CURL) and 16.1% gain compared to the embedding-based approaches (TransE and TransR).

## B.2 Case Stuides

In this part, we take WorksFor , AthletePlaysSport and TeamPlaysinLeague from NELL-995, as examples, to analyze these paths found by AttnPath (Wang et al. 2019), CURL (Zhang et al. 2022) and FULORA. In order to concisely demonstrate model performance across varying path lengths ( WorksFor , AthletePlaysSport , and TeamPlaysinLeague ), we assign respective path lengths of { 9 , 7 , 3 } representing long-distance reasoning (LD), medium-distance reasoning (MD), and short-distance reasoning (SD). As depicted in Table 6 during LD reasoning ( WorksFor ), FULORA effectively identifies target entities even when following an incorrect path. Conversely, both AttnPath and CURL exhibit limitations. Specifically, AttnPath struggles with re-establishing correct paths while CURL

faces challenges in precise reasoning at a granular level. Without effective guidance, AttnPath becomes entangled by multiple entities associated with identical relationships. For instance, in tasks involving TeamPlaysinLeague , AttnPath may identify the correct relationship but navigate towards an wrong entity. Similarly, CURL encounters analogous issues stemming from its inability to strike a proper balance between exploration and guidance. Simply put, CURL leads DWARF too closely along GIANT's trajectory without fully benefiting DWARF on account of distributional deviations at finer levels, resulting in inadequate exploratory capabilities.

## B.3 Cluster State Similarity

To assess the exploration efficiency of GIANT, we introduce cluster state similarity (CSS) as a metric, and in the main content we compare the performance of CURL and FULORA on only three tasks due to space limitations. Figure 7 also compare the model performance on remaining six tasks. For the same task, CURL experiences significant oscillations as path length increases, while FULORA remains stable. As in our previous analysis, CURL can only judge path quality by whether it has gone to the correct target cluster, while FULORA utilizes dynamic path feedback to promote the GIANT to converge to a high-quality path.

Figure 8 further visualizes performances of FULORA and CURL on these tasks with varied path length. We utilize the ratio of the mean to the variance of CSS as an evaluation metric for the mean measures reasoning accuracy and the variance measures reasoning stability. In the vast majority of cases, FULORA outperforms CURL. In the case of long-distance reasoning for complex tasks like PersonBorninLocation-Path9 (PBL-9), AthletePlaysinLeague-Path7 (APL9), AthletePlaysSport-Path9 (APS-9), and TeamPlaysinLeague (TPL), the performance gap between FULORA and CURL is significant.

## B.4 Additional Ablation Studies

Here, we conduct a series of ablation experiments designed to answer the following three research questions:

RQ1: Can attention mechanism enhance DWARF's reasoning ability?

RQ2: Can dynamic path feedback accelerate DWARF'S learning speed via improving GIANT'S reasoning ability?

WorksFor (Answer):

Jeff Skilling

worksfor

→

Enron

WorksFor (AttnPath):

Jeff Skilling(2)

worksfor

→

Enron(2)

topmemberoforganizition →

Kenneth Lay

personleadsorganization

→

Enron and Worldcom(5)

WorksFor (CURL):

Jeff Skilling(2)

personleadsorganization

→

Enron and Worldcom

subpartoforganization

→

GE

subpartoforganization

-

1

→

Enron and Worldcom

personleadsorganization

-

1

→

Jeff Skilling

worksfor

→

Enron(3)

WorksFor (FULORA):

Jeff Skilling(4)

worksfor

→

Enron

topmemberoforgnazition →

Kenneth Lay(2)

worksfor

→

Enron(3)

AthletePlaysSport (Answer):

Carlos Villanueva

athleteplayssport →

Baseball

AthletePlaysSport (AttnPath):

Carlos Villanueva(2)

athleteflyouttosportsteamposition

→

Center

athleteflyouttosportsteamposition - →

1

Chris Coste(3)

athleteplayssport

→

Baseball(2)

AthletePlaysSport (CURL):

Carlos Villanueva(4)

athleteflyouttosportsteamposition →

Center

athleteflyouttosportsteamposition

-

1

→

J.C. Boscan

athleteplayssport

→

Baseball(2)

AthletePlaysSport (FULORA):

Carlos Villanueva(3)

athleteplayssport

→

Baseball(5)

TeamPlaysinLeague (Answer):

Duke University

teamplaysinleague

→

International

TeamPlaysinLeague (AttnPath):

Duke University

templayssport

→

Basketball

teamplayssport

-

1

→

Boise State Broncos

teamplaysinleague → NCAA

TeamPlaysinLeague (CURL):

Duke University

teamplayssport →

Basketball

teamplayssport

-

1

→

Bucks

teamplaysinleague

→

NBA

TeamPlaysinLeague (FULORA):

Duke University

teamplayssport

→

Basketball

teamplayssport

-

1

→

Old Dominion University

teamplaysinleague

→

International

Table 6: Paths found by AttnPath, CURL and FULORA, respectively. In order to describe the self-ring succinctly, we denote that agent stay at the same entity continuously n times as Entity( n ).

<!-- image -->

## RQ3: Can efficient guidance-exploration method make the guidance-exploration trade-off?

With a slight abuse of abbreviation, we use ATTN, DPF and GE to denote attention mechanism, dynamic path feedback and the efficient guide-exploration method respectively. To highlight the effect of each component, we set FULORA as the baseline. In Figure 9, FULORA-X represents pulling module X out of the FULORA framework . We conduct experiments on three different scales of KG to make a fair comparison. It is worth noting that in order to describe DWARF reasoning accuracy in the whole training process, Entity State Similarity (ESS, be similar to CSS we mentioned above) is used as an evaluation index.

B.4.1 Attention Mechanism We first focus on ablation experiments on the attention mechanism. As shown in Figure 9, the absence of the attention mechanism has little effect on WN18RR (sparse KG), while the effect of the attention mechanism increases with increasing density. In particular, on FB15K-237 (dense KG), the effect of the attention mechanism on the average ESS reaches 1 / 3 . It is consistent with the purpose of introducing the attention mechanism, that is, enhancing the reasoning ability of DWARF in the case of multi-neighbor and multi-relation. The reason lies in DWARF are forced to prioritize relations and neighbors that are highly correlated with the query relations, which plays an important role in reasoning on dense KG with complex relations, but because sparse KG neighbor relations are not sophisticated, the attention mechanism does not have a substantial effect.

B.4.2 Dynamic Path Feedback We have demonstrated in previous experiments that dynamic path feedback can improve GIANT's reasoning ability, next we concentrate on the impact to DWARF's learning speed of dynamic path feedback. To illustrate the speed and stability of the training, we utilize kdeplot in Figure 9. The distribution maps on the three datasets have a common feature: compared with FULORA, the learning speed and learning accuracy decrease in the absence of dynamic path feedback (corresponding to the widening and downward movement of the distribution maps). In FB15K-237 (dense KG), the lack of dynamic path feedback has a great impact on the learning speed and accuracy, because the error of cluster mapping escalates with the complexity of neighbor relations when the number of clusters maintains. At this time, a large deviation between cluster-level KG and entity-level KG requires GIANT to make a correct evaluation of the path timely to provide exact guidance for DWARF.

Figure 7: Learning curves comparing the performance of ours against CURL. Learning curves are averaged over 5 seeds, and the shaded area represents the standard deviation across seeds.

<!-- image -->

Figure 8: The ratios of the mean and variance of CSC difference between FULORA and CURL on different taskpath length.

<!-- image -->

B.4.3 Efficient Guidance-Exploration The core of FULORA is the efficient guidance-exploration method, which contributes to make the guidance-exploration tradeoff. In the previous Hierarchical RL-based reasoning method like CURL, there is a high degree of coupling between the two agents. We treat ESS and CSS as two time series because they change with training progress . Next, we use Granger Causality test (Diks and Panchenko 2006) to examine the degree of coupling between DWARF and GIANT.

Granger Causality test is used to study the causal relationship between two sets of data, that is, to test whether one set of time series causes changes in another set of time series (Diks and Panchenko 2006). It is worth noting that Granger Causality test requires stationary time series, otherwise false regression issues may occur, so it is necessary to detect the stationarity of time series through ADF test. If the pair-to-pair time series is non-stationary and satisfies the homogeneity of order, the Granger Causality test can be carried out only after the cointegration test between pair-to-pair sequences exists (Dougherty 2011). Our time series analysis for ESS and CSS are detailed in Appendix C.

The results of the Granger Causality Test for FULORA and FULORA-GE are shown in Table 7. The efficient guidance-exploration method reduces the coupling between GIANT and DWARF, thus alleviating the affect of policy shift caused by cluster mapping. In particular, on the WN18RR and FB15K-237, FULORA performs better than the version without efficient guidance-exploration method due to the greatly reduced coupling between DWARF and GIANT. Certain tasks in WN18RR (sparse KG) require the agent to have excellent long-distance reasoning ability. However, due to the distribution deviation, even if the GIANT reaches the correct target cluster, the guidance provided by it may not be suitable for DWARF (for example, the path length is too long to reach the correct target entity). The high coupling also seriously affects GIANT, resulting in poor model performance.

## C Sensitive Test on Cluster Size

In the previous experiments, we examine the primary components of FULORA and analyze their impact on reasoning ability. Given that GIANT walks directly on the cluster-level KG derived from the entity-level KG, it is essential to separately evaluate the implications of cluster size.

Figure 10 shows the MRR score of FULORA under different cluster number N during training on three real-world KG datasets. We observe that variations in cluster sizes significantly influence the reasoning performance of FU- LORA, particularly when reasoning over long distances on WN18RR (sparse KG) and FB15K-237 (dense KG). If the cluster size is too small (e.g., N = 500 ), GIANT must traverse a greater number of clusters. In the absence of higherlevel agent to provide guidance, the reasoning process approximates the long-distance reasoning of a single agent. Conversely, when the cluster size is too large (e.g., N = 50 ), the guidance offered by GIANT becomes overly general.

<!-- image -->

Figure 9: Entity state similarities comparing the performance of FULORA, FULORA-GE, FULORA-ATTN and FULORADPF. In addition, bar charts and distribution charts depict the learning speed and accuracy. Learning curves are averaged over 3 seeds.

Figure 10: The effect of different cluster size used by GIANT. We present the link prediction (query answering) performance on three real-world KG datasets.

<!-- image -->

Table 7: F statistics of Granger Causality test for FULORA and FULORA-GE. DWARF (-GE) and GIANT (-GE) are from FULORA-GE. In the Granger Causality test, a large F statistic indicates a strong causal relation. We set the lagged step as 2.

| Relation                |   NELL-995 |   WN18RR |   FB15K-237 |
|-------------------------|------------|----------|-------------|
| DWARF → GIANT           |       21.5 |      9.6 |        18.9 |
| DWARF(-GE) → GIANT(-GE) |       25.3 |     33.9 |        60.4 |
| GIANT → DWARF           |       21.7 |     40.3 |        36.3 |
| GIANT(-GE) → DWARF(-GE) |       26.3 |     41.1 |        37.3 |

Even if the guidance is accurate, it may still prove ineffective, causing DWARF to similarly approximate the behavior of a single agent during long-distance reasoning.

## D Time Series Analysis to Efficient Guidance-Exploration

Before Granger Causality test, we firstly use Augmented Dickey-Fuller (ADF) test (Cheung and Lai 1995) to judge the stationarity of the series. The results as shown in Table 8. Next, we make D-order difference for the non-stationary time series, and then test the stationarity of the difference series. As illustrated in Figure 11, the time series is stationary after 1-order difference. Therefore, we demonstrate that these series are integrated of order one . To avoid the pseudo-regression phenomenon, we also need to do cointegration analysis of these series (Dougherty 2011). Here, we employ Johansen test (Ho and Sørensen 1996) based on maximum likelihood estimation for cointegration analysis.

In the Johansen cointegration test (Ho and Sørensen 1996), the 2-cointegration term critical values of { 10%, 5%, 1% } are { 2.705, 3.841, 6.635 } respectively. The critical value 3.841 corresponding to 5% was selected as the boundary for testing. Table 9 shows the results of Johansen cointegration test for FULORA and FULORA-GE. It shows that the seven non-stationary series avoid the pseudo-regression problem, thus the result of Granger causality test is reasonable (which is shown in Appendix B.4.3).

<!-- image -->

Figure 11: Entity state similarity curves and their 1-order difference on WN18RR and FB15K-237 from FULORA and FULORA-GE.

Table 8: t statistics and P value of ADF test for FULORA and FULORA-GE. DWARF (-GE) and GIANT (-GE) are from FULORA-GE. We set the bound of P value to 0.05, and P-values above the bound are bolded to indicate that they are not stationary series.

| Variable   | NELL-995   | NELL-995   | WN18RR   | WN18RR   | FB15K-237   | FB15K-237   |
|------------|------------|------------|----------|----------|-------------|-------------|
|            | t          | P          | t        | P        | t           | P           |
| DWARF      | -3.25      | 0.04       | -7.40    | 0.00     | -1.67       | 0.45        |
| GIANT      | -5.22      | 0.00       | -1.46    | 0.46     | 0.08        | 0.97        |
| DWARF(-GE) | -3.11      | 0.04       | -1.65    | 0.46     | 0.08        | 0.99        |
| GIANT(-GE) | -4.14      | 0.00       | -1.98    | 0.30     | -0.21       | 0.94        |

Table 9: Results of Johansen cointegration test. CR is the abbreviation of Characteristic Root. The trace satisfying the critical value is bolded , indicating that there are two cointegration relations.

| Relation                | WN18RR   | WN18RR   | FB15K-237   | FB15K-237   |
|-------------------------|----------|----------|-------------|-------------|
|                         | CR       | Trace    | CR          | Trace       |
| DWARF ↔ GIANT           | 0.019    | 38.436   | 0.002       | 4.736       |
| DWARF(-GE) ↔ GIANT(-GE) | 0.037    | 74.582   | 0.001       | 4.133       |

## E Proof of Dynamic Path Feedback

Amajor concern with dynamic path feedback is whether the optimal policy GIANT learns is consistent with the default rewards. Here we provide the following proof of Theorem 1.

Proof 1 Recall Equation 9 where the next state s c t +1 is used, which means that it implies the action a c t . The next state s c t +1 is the output of the current state s c t and the current action a c t to the transition function. In fact, J ( θ π c ) is a function of state s c t and action a c t . The same goes for reward settings in the default environment. Therefore, we use the Q-function to simplify our notation,

<!-- formula-not-decoded -->

The optimal Q-function Q ∗ ( s c t , a c t ) is subject to Bellman optimal equation (Bellman 1958):

<!-- formula-not-decoded -->

then we make a simple transformation of the above formula to get

<!-- formula-not-decoded -->

The Sim(s c t , s c target ) we define is only related to the state, so Equation 12 is equivalent to Equation 13. Notice that r c ( s c t ) -α ∆( s c t , s c t +1 ) is the new reward function ˆ r c ( s c t ) we designed for GIANT. Hence, Equation 12 can be viewed as

<!-- formula-not-decoded -->

where ˆ Q ∗ ( s c t , a c t ) = Q ∗ ( s c t , a c t ) -α Sim ( s c t , s c target ) . Equation 14 is the Bellman optimal equation in dynamic path feedback. It constructs a new Markov Decision Process (MDP), which we use M ′ to denote for distinction and M for the original MDP. Thus, the optimal policy in M ′ is

<!-- formula-not-decoded -->

which implies that GIANT learns optimal policy in dynamic path feedback is consistent with optimal policy in the default rewards. □

## F Pseudocode for FULORA

We show FULORA's training process in one episode.

Algorithm 1: FULORA Training Algorithm (one episode)

## Require:

Entity-level KG G e and cluster-level KG G c ; Initial policy networks parameters θ π e and θ π c ; Initial Lagrange multiplier parameter θ λ ;Source entity and cluster nodes e s and c s ; Entity-level query r q ; Target entity and cluster nodes e o and c o ; Maximum path length T

## Ensure:

parameters θ π e , θ π c , θ λ

- 1: for t = 0 , ..., T -1 do
- 2: Set default cluster-level reward r c = 1 if c t = c o otherwise r c = 0
- 3: Set default entity-level reward r e = 1 if e t = e o otherwise r e = 0
- 4: Predict the action a c t and a e t for GIANT and DWARF based on policy networks parameters θ π c and θ π e
- 5: Compute J ( θ π c ) , J ( θ π e ) , J ( θ λ ) based on Equation 7-9.

## 6: end for

- 7: Update model parameters:
- 8: return θ π e , θ π c , θ λ

```
θ π c ← θ π c + α π c ∇ θ π c J ( θ π c ) θ π e ← θ π e + α π e ∇ θ π e J ( θ π e ) θ λ ← θ λ -α λ ∇ θ λ J ( θ λ )
```

In addtion, a common concern is how clusters are formed, therefore, we provide pseudocode to demonstrate cluster formation, which is presented below.

Algorithm 2: Cluster Formation and Batch Processing

## Require:

Raw KG G ; Input data file, vocabularies ( entity vocab , relation vocab , cluster vocab ), batch size batch size , entity-to-cluster mapping entity id to cluster mapping

## Ensure:

Generated batches with cluster-level relations.

- 1: Initialize vocabularies and mappings from the input data file
- 2: Parse the input file to extract triples ( e 1 , r , e 2 )
- 3: Map each entity to its corresponding cluster
- 4: Create cluster relations c r = c 1 c 2 for each triple
- 5: Store the triples and update entity-level and cluster-level mappings
- 6: For each batch, randomly sample triples from the stored data
- 7: Extract entities e 1 , e 2 , relations r and their cluster mappings c 1 , c 2
- 8: Yield entity-level and cluster-level batch data for training or testing
- 9: return Entity-level and cluster-level batch data.