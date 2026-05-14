## BM-NAS: Bilevel Multimodal Neural Architecture Search

Yihang Yin 1 , Siyu Huang 2 , Xiang Zhang 3

1 Nanyang Technological University, 2 Harvard University, 3 The Pennsylvania State University yyin009@e.ntu.edu.sg, huang@seas.harvard.edu, xzz89@psu.edu

## Abstract

Deep neural networks (DNNs) have shown superior performances on various multimodal learning problems. However, it often requires huge efforts to adapt DNNs to individual multimodal tasks by manually engineering unimodal features and designing multimodal feature fusion strategies. This paper proposes Bilevel Multimodal Neural Architecture Search (BM-NAS) framework, which makes the architecture of multimodal fusion models fully searchable via a bilevel searching scheme. At the upper level, BM-NAS selects the inter/intra-modal feature pairs from the pretrained unimodal backbones. At the lower level, BM-NAS learns the fusion strategy for each feature pair, which is a combination of predefined primitive operations. The primitive operations are elaborately designed and they can be flexibly combined to accommodate various effective feature fusion modules such as multi-head attention (Transformer) and Attention on Attention (AoA). Experimental results on three multimodal tasks demonstrate the effectiveness and efficiency of the proposed BM-NAS framework. BM-NAS achieves competitive performances with much less search time and fewer model parameters in comparison with the existing generalized multimodal NAS methods. Our code is available at https://github.com/Somedaywilldo/BM-NAS.

## 1 Introduction

Deep neural networks (DNNs) have achieved a great success on various unimodal tasks ( e.g ., image categorization (Krizhevsky, Sutskever, and Hinton 2012; He et al. 2016), language modeling (Vaswani et al. 2017; Devlin et al. 2018), and speech recognition (Amodei et al. 2016)) as well as the multimodal tasks ( e.g ., action recognition (Simonyan and Zisserman 2014; Vielzeuf et al. 2018), image/video captioning (You et al. 2016; Jin et al. 2019, 2020), visual question answering (Lu et al. 2016; Anderson et al. 2018), and crossmodal generation (Reed et al. 2016; Zhou et al. 2019)). Despite the superior performances achieved by DNNs on these tasks, it usually requires huge efforts to adapt DNNs to the specific tasks. Especially with the increase of modalities, it is exhausting to manually design the backbone architectures and the feature fusion strategies. It raises urgent concerns about the automatic design of multimodal DNNs with minimal human interventions.

Copyright © 2022, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

Figure 1: An overview of our BM-NAS framework for multimodal learning. a Cell is a searched feature fusion unit that accepts two inputs from modality features or other Cells. In a bilevel fashion, we search the connections between Cells and the inner structures of Cells, simultaneously.

<!-- image -->

Neural architecture search (NAS) (Zoph and Le 2017; Liu et al. 2018a) is a promising data-driven solution to this concern by searching for the optimal neural network architecture from a predefined space. By applying NAS to multimodal learning, MMnas (Yu et al. 2020) searches the architecture of Transformer model for visual-text alignment and MMIF (Peng et al. 2020) searches the optimal CNNs structure to extract multi-modality image features for tomography. These methods lack generalization ability since they are designed for models on specific modalities. MFAS (P´ erez-R´ ua et al. 2019) is a more generalized framework which searches the feature fusion strategy based on the unimodal features. However, MFAS only allows fusion of intermodal features, and the fusion operations are not searchable. It results in a limited space of feature fusion strategies when dealing with various modalities in different multimodal tasks.

In this paper, we propose a generalized framework, named Bilevel Multimodal Neural Architecture Search (BM-NAS), to adaptively learn the architectures of DNNs for a variety of multimodal tasks. BM-NAS adopts a bilevel searching scheme that it learns the unimodal feature selection strategy at the upper level and the multimodal feature fusion strategy at the lower level, respectively. As shown in the left part of Fig. 1, the upper level of BM-NAS consists of a series of feature fusion units, i.e ., Cells. The Cells are organized to combine and transform the unimodal features to the task output through a searchable directed acyclic graph (DAG). The right part of Fig. 1 illustrates the lower level of BM-NAS which learns the inner structures of Cells. A Cell is comprised of several predefined primitive operations. We carefully select the primitive operations such that different combinations of them can form a large variety of fusion modules, as shown in Fig. 2, our search space incorporates benchmark attention mechanisms like multi-head attention (Transformer) (Vaswani et al. 2017) and Attention on Attention (AoA) (Huang et al. 2019). The bilevel scheme of BM-NASis end-to-end learned using the differentiable NAS framework (Liu, Simonyan, and Yang 2019). We conduct extensive experiments on three multimodal tasks to evaluate the proposed BM-NAS framework. BM-NAS shows superior performances in comparison with the state-of-the-art multimodal methods. Compared with the existing generalized multimodal NAS frameworks, BM-NAS achieves competitive performances with much less search time and fewer model parameters. To the best of our knowledge, BM-NAS is the first multimodal NAS framework that supports the search of both the unimodal feature selection strategies and the multimodal fusion strategies.

Figure 2: The search space of a Cell in BM-NAS accommodates many existing multimodal fusion strategies. (d) is a two-head version of multi-head attention (Vaswani et al. 2017), and more heads can be flexibly added by changing the number of inner steps. (e) is the Cell founded by BM-NAS on NTU RGB-D dataset (Shahroudy et al. 2016), and it outperforms these existing fusion strategies (see Table 7).

<!-- image -->

The main contributions of this paper are three-fold.

1. Towards a more generalized and flexible design of DNNs for multimodal learning, we propose a new paradigm that employs NAS to search both the unimodal feature selection strategy and the multimodal fusion strategy.
2. We present a novel BM-NAS framework to address the proposed paradigm. BM-NAS makes the architecture of multimodal fusion models fully searchable via a bilevel searching scheme.
3. We conduct extensive experiments on three multimodal learning tasks to evaluate the proposed BM-NAS framework. Empirical evidences indicate that both the unimodal feature selection strategy and the multimodal fusion method are significant to the performance of multimodal DNNs.

## 2 Related Work

## 2.1 Neural Architecture Search

Neural architecture search (NAS) aims at automatically finding the optimal neural network architectures for specific learning tasks. NAS can be viewed as a bilevel optimization problem by optimizing the weights and the architecture of DNNs at the same time. Since the network architecture is discrete, traditional NAS methods usually rely on the black-box optimization algorithms, resulting in a extremely large computing cost. For example, searching architectures using reinforcement learning (Zoph and Le 2016) or evolution (Real et al. 2019) would require thousands of GPU-days to find a state-of-the-art architecture on ImageNet dataset (Deng et al. 2009) due to low sampling efficiency.

As a result, many methods were proposed for speeding up NAS. From the perspective of engineering, ENAS (Pham et al. 2018) improve the sampling efficiency by weightsharing. From the perspective of optimization algorithm, PNAS (Liu et al. 2018b) employs sequential model-based optimization (SMBO) (Hutter, Hoos, and Leyton-Brown 2011), using surrogate model to predict the performance of an architecture. Monte Carlo tree search (MTCS) (Negrinho and Gordon 2017) and Bayesian optimization (BO) (Kandasamy et al. 2018) are also explored to enhance the sampling efficiency.

Recently, a remarkable efficiency improvement of NAS is achieved by differentiable architecture search (DARTS) (Liu, Simonyan, and Yang 2019). DARTS introduces a continuous relaxation of the network architecture, making it possible to search an architecture via gradient-based optimization. However, DARTS only supports the search of unary operations. For specific multimodal tasks, we expect the NAS framework to support the search of multi-input operations, in order to obtain the optimal fusion strategy. In this work, we devise a novel NAS framework named BM-NAS for multimodal learning. BM-NAS follows the optimization scheme of DARTS, however, it novelly introduces a bilevel searching scheme to search the unimodal feature selection strategy and the multimodal fusion strategy simultaneously, enabling an effective search scheme for multimodal fusion.

## 2.2 Multimodal Fusion

The multimodal fusion techniques for DNNs can be generally classified into two categories: early fusion and late fusion. Early fusion combines low-level features, while late fusion combines prediction-level features. To combine these features, a series of reduction operations such as weighted average (Natarajan et al. 2012) and bilinear product (Teney et al. 2018) are proposed in previous works. As each unimodal DNNs backbone could have tens of layers or maybe more, manually sorting out the best intermediate features for multimodal fusion could be exhausting. Therefore, some works propose to enable fusion at multiple intermediate layers. For instance, CentralNet (Vielzeuf et al. 2018) and MMTM (Joze et al. 2020) join the latent representations at each layer and pass them as auxiliary information for deeper layers. Such methods achieve superior performances on several multimodal tasks including multimodal action recognition (Shahroudy et al. 2016) and gesture recognition (Zhang et al. 2018). However, it would largely increase the parameters of multimodal fusion models.

In recent years, there is an increased interest of introducing the attention mechanisms such as Transformer (Vaswani et al. 2017) to multimodal learning. The multimodal-BERT family (Chen et al. 2019; Li et al. 2019; Lu et al. 2019; Tan and Bansal 2019) is a typical approach for inter-modal fusion. Moreover, DFAF (Gao et al. 2019) shows that intramodal fusion could also be helpful. DFAF proposes a dynamic attention flow module to mix inter-modal and intramodal features together through the multi-head attention (Vaswani et al. 2017). Additional efforts are made to enhance multimodal fusion efficacy of attention mechanisms. For instance, AoANet (Huang et al. 2019) proposes the attention on attention (AoA) module, showing that adding an attention operation on top of another one could achieve better performance on image captioning task.

Recently, the NAS approaches are making an exciting progress for DNNs, and it shows a huge potential to introduce NAS to multimodal learning. One representative work is MFAS (P´ erez-R´ ua et al. 2019), which employs SMBO algorithm (Hutter, Hoos, and Leyton-Brown 2011) to search multimodal fusion strategies given the unimodal backbones. But as SMBO is a black-box optimization algorithm, every update step requires a bunch of DNNs to be trained, leading to the inefficiency of MFAS. Besides, MFAS only use concatenation and fully connected (FC) layers for unimodal feature fusion, and the stack of FC layers would be a heavy burden for computing. Further work like MMIF (Peng et al. 2020) and 3D-CDC (Yu et al. 2021) adopt the efficient DARTS algorithm (Liu, Simonyan, and Yang 2019) for architecture search but only support the search of unary operations on graph edges and use summation on every intermediate node for reduction. MMnas (Yu et al. 2020) allows searching the attention operations but the topological structure of the network is fixed during architecture search.

Different from these related works, our proposed BMNAS supports to search both the unimodal feature selection strategy and the fusion strategy of multimodal DNNs. BM-NAS introduces a bilevel searching scheme. The upper level of BM-NAS supports both intra-modal and inter-modal feature selection. The lower level of BM-NAS searches the fusion operations within every intermediate step. Each step can flexibly form the summation, concatenation, multihead attention (Vaswani et al. 2017), attention on attention (Huang et al. 2019), or any other unexplored fusion mechanisms. BM-NAS is a generalized and efficient NAS framework for multimodal learning. In experiments we show that BM-NAScan be applied to various multimodal tasks regardless of the modalities or backbone models.

## 3 Methodology

In this work, we propose a generalized NAS framework, named Bilevel Multimodal NAS (BM-NAS), to search the architectures of multimodal fusion DNNs. More specifically, BM-NAS searches a Cell-by-Cell architecture in a bilevel fashion. The upper level architecture is a directed acyclic graph (DAG) of the input features and Cells. The lower level architecture is a DAG of inner step nodes within a Cell. Each inner step node is a bivariate operation drawn from a predefined pool. The bilevel searching scheme ensures that BMNAS can be easily adapted to various multimodal learning tasks regardless of the types of modalities. In the following, we discuss the unimodal feature extraction in Section 3.1, the upper and lower levels of BM-NAS in Sections 3.2 and 3.3, along with the architecture search algorithm and evaluation in Section 3.4.

## 3.1 Unimodal feature extraction

By following previous multimodal fusion works, such as CentralNet (Vielzeuf et al. 2018), MFAS (P´ erez-R´ ua et al. 2019) and MMTM (Joze et al. 2020), we also employ the pretrained unimodal backbone models as the feature extractors. We use the outputs of their intermediate layers as raw features (or intermediate blocks if the model has a block-byblock structure like ResNeXt (Xie et al. 2017)).

Since the raw features vary in shapes, we reshape them by applying pooling, interpolation, and fully connected layers on spatial, temporal, and channel dimensions, successively. By doing so, we reshape all the raw features to the shape of ( N,C,L ) , such that we can easily perform fusion operations between features of different modalities. Here N is the batch size, C is the embedding dimension or the number of channels, L is the sequence length.

## 3.2 Upper Level: Cells for Feature Selection

The upper level of BM-NAS searches the unimodal feature selection strategy and it consists of a group of Cells. Formally, suppose we have two modalities A and B, and two pretrained unimodal models for each modality. Let { A ( i ) } and { B ( i ) } indicate the modality features extracted by the backbone models. We formulate the upper level nodes in an ordered sequence S , as

<!-- formula-not-decoded -->

Under the setting of S , both inter-modal fusion and intramodal fusion are considered in BM-NAS.

Feature selection. By adopting the continuous relaxation in differentiable architecture search scheme (Liu, Simonyan, and Yang 2019), all predecessors of Cell ( i ) will be connected to Cell ( i ) through weighted edges at the searching stage. This directed complete graph between Cells is called the hypernet . For two upper level nodes s ( i ) , s ( j ) ∈ S , let α ( i,j ) denote the edge weight between s ( i ) and s ( j ) . Each edge is a unary operation g selected from a function set G including

Figure 3: An example of a multimodal fusion network found by BM-NAS, which consists of a bilevel searching scheme, we denote searched edges in blue, and fixed edges in black. Left : The upper level BM-NAS. The input features are extracted by pretrained unimodal models. Each Cell accepts two inputs from its predecessors, i.e ., any unimodal feature or previous Cell. Right : The lower level BM-NAS. Within a Cell, each Step denotes a primitive operation selected from a predefined operation pool. The topologies of Cells and Steps are both searchable. The numbers of Cells and Steps are hyper-parameters such that BM-NAS can be adapted to a variety of multimodal tasks with different scales.

<!-- image -->

- (1) Identity( x ) = x , i.e ., selecting an edge.

(2) Zero( x ) = 0 , i.e ., discarding an edge.

Then, the mixed edge operation g ( i,j ) on edge ( i, j ) is

<!-- formula-not-decoded -->

A Cell s ( j ) receives inputs from all its predecessors, as

<!-- formula-not-decoded -->

In evaluation stage, the network architecture is discretized that an input pair ( s ( i ) , s ( j ) ) 1 will be selected for s ( k ) if

<!-- formula-not-decoded -->

It is worth noting that, compared with searching the feature pairs directly, the Cell-by-Cell structure significantly reduces the complexity of the search space for unimodal feature selection. For an input pair from two feature sequences [ A (1) , ..., A ( N A ) ] and [ B (1) , ..., B ( N B ) ] , the number of candidate choices is 2( N A + N B ) under the Cell-by-Cell search setting. It is much smaller than C 2 N A + N B , the number of candidates under the pairwise search setting.

## 3.3 Lower Level: Multimodal Fusion Strategy

The lower level of BM-NAS searches the multimodal fusion strategy, i.e ., the inner structure of Cells. Specifically, a Cell is a DAG consisting of a set of inner step nodes. The inner step nodes are the primitive operations drawn from a predefined operation pool. We introduce our predefined operation pool in the following.

1 We enforce the Cells to have different predecessors.

Primitive operations. All the primitive operations take two tensor inputs x, y , and outputs a tensor z , where x, y, z ∈ R N × C × L .

(1) Zero( x, y ) : The Zero operation discards an inner step completely. It will be helpful when BM-NAS decides to use only a part of the inner steps.

<!-- formula-not-decoded -->

(2) Sum( x, y ) : The DARTS (Liu, Simonyan, and Yang 2019) framework uses summation to combine two features as

<!-- formula-not-decoded -->

(3) Attention( x, y ) : We use the scaled dot-product attention (Vaswani et al. 2017). As a standard attention module usually takes three inputs namely query, key, and value, we let the query be x , the key and value be y , which is also known as the guided-attention (Yu et al. 2020).

<!-- formula-not-decoded -->

(4) LinearGLU( x, y ) : A linear layer with the gated linear unit (GLU) (Dauphin et al. 2017). Let W 1 , W 2 ∈ R C × C and glyph[circledot] be element-wise multiplication, then LinearGLU is

<!-- formula-not-decoded -->

(5) ConcatFC( x, y ) : ConcatFC stands for passing the concatenation of ( x, y ) to a fully connected (FC) layer with ReLU activation (Nair and Hinton 2010). The FC layer reduces the channel numbers from 2 C to C . Let W ∈ R 2 C × C , b ∈ R C , then ConcatFC is

<!-- formula-not-decoded -->

We elaborately choose these primitive operations such that they can be flexibly combined to form various feature fusion modules. In Fig. 2, we show that the search space of lower level BM-NAS accommodates many benchmark multimodal fusion strategies such as the summation used in DARTS (Liu, Simonyan, and Yang 2019), the ConcatFC used in MFAS (P´ erez-R´ ua et al. 2019), the multi-head attention used in Transformer (Vaswani et al. 2017), and the Attention on Attention used in AoANet (Huang et al. 2019). There also remains flexibility to discover other better fusion modules for specific multimodal learning tasks.

Fusion strategy. In searching stage, the inner step set of Cell ( n ) is an ordered feature sequence T n ,

<!-- formula-not-decoded -->

An inner step node t ( i ) transforms two input nodes t ( j ) , t ( k ) to its output through an average over the primitive operation pool F , as

<!-- formula-not-decoded -->

where γ is the weights of primitive operations. In the evaluation stage, the optimal operation of an inner step node is derived as,

<!-- formula-not-decoded -->

The continuous relaxation of the edges with weights β between inner step nodes is similar to the upper level. For a simplicity, we omit the formulation in this paper. Note that unlike the upper level BM-NAS, the pairwise inputs in a Cell can be chosen repeatedly 2 , so the inner steps can form structures like multi-head attention (Vaswani et al. 2017).

## 3.4 Architecture Search and Evaluation

Architecture Parameters. The function of the weights of primitive operations ( β ) and inner step nodes edges ( γ ) is shown in Fig. 4, β is used for feature selection within the cell, selecting two inputs for each inner step node. And γ is used for operation selection on each inner step node.

Search algorithm. In Sections 3.2 and 3.3, we introduced three variable α, β, γ as the architecture parameters. Algorithm 1 shows the searching process of BM-NAS, which follows DARTS (Liu, Simonyan, and Yang 2019) to optimize α, β, γ and model weights w , alternatively. In Algorithm 1, the model in searching stage is called hypernet since all the edges and nodes are mixed operations. The searched structure description of the fusion network is called genotype .

Implementation details. In order to make the whole BMNAS framework searchable and flexible, Cells/inner step nodes should have the same number of inputs and output, so they can be put together in arbitrary topological order. The two-input setting follows the benchmark NAS frameworks (DARTS (Liu, Simonyan, and Yang 2019)), MFAS (P´ erezR´ ua et al. 2019), MMIF (Peng et al. 2020), etc .). They all

2 Wedon't enforce the step nodes to have different predecessors.

## Algorithm 1: Bilevel Multimodal NAS (BM-NAS)

```
Result: The genotype of fusion networks. Initialize architecture parameters α, β, γ and model parameters w ; Initialize genotype based on α, β, γ , set genotype best = genotype ; Construct hypernet based on genotype best ; while L not converged do Update ω on training set; Update ( α , β , γ ) on validation set; Derive upper level genotype based on α , derive lower level genotype based on β, γ ; Update hypernet based on genotype ; if higher validation accuracy is reached then Update genotype best using genotype ; end end Return genotype best ;
```

Figure 4: Architecture parameters β and γ of a Cell .

<!-- image -->

have only two searchable inputs for each Cell/step node. Also, it requires no extra effort to let the Cells or step nodes support 3 or more inputs, by just adding ternary (or other arbitrary) operations into the primitive operation pool.

Evaluation. In architecture evaluation, we select the genotype with the best validation performance as the searched fusion network. Then we combine the training and validation sets together to train the unimodal backbones and the searched fusion network jointly.

## 4 Experiments

In this work we evaluate the BM-NAS on three multimodal tasks, including (1) the multi-label movie genre classification task on MM-IMDB dataset (Arevalo et al. 2017), (2) the multimodal action recognition task on NTU RGB-D dataset (Shahroudy et al. 2016), and (3) the multimodal gesture recognition task on EgoGesture dataset (Zhang et al. 2018). In the following, we discuss the experiments on the three tasks in Sections 4.1, 4.2, and 4.3, respectively. We perform computing efficiency analysis in Section 4.4. We further evaluate the search strategies of the proposed BM-NAS framework in Sections 4.5.

In addition, we present the examples of these tasks, thorough discussion of hyper-parameter configurations, visualization of searched architectures and their performances during the searching stage (hypernets) and evaluation stage (final model) in our supplementary material.

Table 1: Multi-label genre classification results on MMIMDB dataset. Weighted F1 (F1-W) is reported.

| Method                | Modality           | F1-W(%)            |
|-----------------------|--------------------|--------------------|
| Unimodal Methods      | Unimodal Methods   | Unimodal Methods   |
| Maxout MLP (ICML13)   | Text               | 57.54              |
| VGG Transfer (ICLR15) | Image              | 49.21              |
| Multimodal Methods    | Multimodal Methods | Multimodal Methods |
| Two-stream (NIPS14)   | Image + Text       | 60.81              |
| GMU(ICLR17)           | Image + Text       | 61.70              |
| CentralNet (ECCV18)   | Image + Text       | 62.23              |
| MFAS (CVPR19)         | Image + Text       | 62.50              |
| BM-NAS (ours)         | Image + Text       | 62.92 ± 0.03       |

Table 2: Action recognition results on NTU RGB-D dataset.

| Method                      | Modality           | Acc(%)             |
|-----------------------------|--------------------|--------------------|
| Unimodal Methods            | Unimodal Methods   | Unimodal Methods   |
| Inflated ResNet-50 (CVPR18) | Video              | 83.91              |
| Co-occurrence (IJCAI18)     | Pose               | 85.24              |
| Multimodal Methods          | Multimodal Methods | Multimodal Methods |
| Two-stream (NIPS14)         | Video + Pose       | 88.60              |
| GMU(ICLR17)                 | Video + Pose       | 85.80              |
| MMTM(CVPR20)                | Video + Pose       | 88.92              |
| CentralNet (ECCV18)         | Video + Pose       | 89.36              |
| MFAS (CVPR19)               | Video + Pose       | 89.50 ± 0.60       |
| BM-NAS (ours)               | Video + Pose       | 90.48 ± 0.24       |

## 4.1 MM-IMDB Dataset

MM-IMDB dataset (Arevalo et al. 2017) is a multi-modal dataset collected from the Internet Movie Database, containing posters, plots, genres and other meta information of 25,959 movies. We conduct multi-label genre classification on MM-IMDB using posters (RGB images) and plots (text) as the input modalities. There are 27 non-mutually exclusive genres in total, including Drama, Comedy, Romance , etc. Since the number of samples in each class is highly imbalanced, we only use 23 genres for classification. The classes of News, Adult, Talk-Show, Reality-TV are discarded since they only count for 0.10% in total. We adopt the original split of the dataset where 15,552 movies are used for training, 2,608 for validation and 7,799 for testing.

For a fair comparison with other explicit multimodal fusion methods, we use the same backbone models. Specifically, we use Maxout MLP (Goodfellow, Warde-Farley, and Courville 2013) as the backbone of text modality and VGG Transfer (Simonyan and Zisserman 2015) as the backbone of RGB image modality. For BM-NAS, we adopt a setting of 2 fusion Cells and 1 step/Cell. For inner step representations, we set C = 192 , L = 16 .

Table 1 shows that BM-NAS achieves the best Weighted F1 score in comparison with the existing multimodal fusion methods. Notice that as the class distribution of MM-IMDB is highly imbalanced, Weighted F1 score is in fact a more reliable metric for measuring the performance of multi-label classification than other kinds of F1 score.

## 4.2 NTU RGB-D Dataset

The NTU RGB-D dataset (Shahroudy et al. 2016) is a large scale multimodal action recognition dataset, containing a to- tal of 56,880 samples with 40 subjects, 80 view points, and 60 classes of daily activities. In this work we use the skeleton and RGB video modality for fusion experiments. We measure the performance of methods using cross-subject (CS) accuracy. We follow the dataset split of MFAS (P´ erez-R´ ua et al. 2019). In detail, we use subjects 1, 4, 8, 13, 15, 17, 19 for training, 2, 5, 9, 14 for validation, and the rest for test. There are 23760, 2519 and 16558 samples in the training, validation, and test dataset, respectively.

Figure 5: Best model found on NTU RGB-D dataset.

<!-- image -->

For a fair comparison, we use two CNN models, the Inflated ResNet-50 (Baradel et al. 2018) for video modality and Co-occurrence (Li et al. 2018) for skeleton modality as backbones, ensuring all the methods in our experiments share the same backbones. We test the performances of MFAS (P´ erez-R´ ua et al. 2019), MMTM (Joze et al. 2020), and the proposed BM-NAS using our data prepossessing pipeline, such that the performances of these methods are not the same as they were original reported. For BM-NAS, we use 2 fusion Cells and 2 Steps/Cell. For inner step representations we set C = 128 , L = 8 .

In Table 2, our method achieves an cross-subject accuracy of 90 . 48% , showing an state-of-the-art result on NTU RGBD (Shahroudy et al. 2016) with video and pose modalities.

## 4.3 EgoGesture Dataset

The EgoGesture dataset (Zhang et al. 2018) is a large scale multimodal gesture recognition dataset, containing 24,161 gesture samples of 83 classes collected from 50 distinct subjects and 6 different scenes. We follow the original crosssubject split of EgoGesture dataset (Zhang et al. 2018). There are 14,416 samples for training, 4,768 for validation, and 4,977 for testing.

We use the ResNeXt-101 (K¨ op¨ ukl¨ u et al. 2019) as the backbone on both RGB and depth video modality. As former works like CentralNet (Vielzeuf et al. 2018) and MFAS (P´ erez-R´ ua et al. 2019) did not perform experiments on this dataset, we compared our method with other unimodal and multimodal methods, especially MMTM (Joze et al. 2020), MTUT (Gupta et al. 2019) and 3D-CDC (Yu et al. 2021). Since we do not search for the backbone, we compared with 3D-CDC-NAS2, which also uses ResNeXt-101 as the backbones. For our BM-NAS, we use 2 fusion Cells and 3 steps/Cell, for inner step representations we set C = 128 , L = 8 .

Table 3 reports the experiment results on EgoGesture (Zhang et al. 2018). Comparing to 3D-CDC, which requires

Table 3: Gesture recognition results on EgoGesture dataset. We use ResNext-101 as backbones for both RGB and depth modality for our BM-NAS method.

| Method                      | Modality           | Acc(%)             |
|-----------------------------|--------------------|--------------------|
| Unimodal Methods            | Unimodal Methods   | Unimodal Methods   |
| VGG-16 + LSTM (NIPS14)      | RGB                | 74.70              |
| C3D + LSTM + RSTTM (ICCV15) | RGB                | 89.30              |
| I3D (CVPR17)                | RGB                | 90.33              |
| ResNext-101 (FG19)          | RGB                | 93.75              |
| VGG-16 + LSTM (CVPR14)      | Depth              | 77.70              |
| C3D + LSTM + RSTTM (CVPR16) | Depth              | 90.60              |
| I3D (CVPR17)                | Depth              | 89.47              |
| ResNeXt-101 (FG19)          | Depth              | 94.03              |
| Multimodal Methods          | Multimodal Methods | Multimodal Methods |
| VGG-16 + LSTM (CVPR17)      | RGB + Depth        | 81.40              |
| C3D + LSTM + RSTTM (CVPR19) | RGB + Depth        | 92.20              |
| I3D (CVPR17)                | RGB + Depth        | 92.78              |
| MMTM(CVPR20)                | RGB + Depth        | 93.51              |
| MTUT (3DV19)                | RGB + Depth        | 93.87              |
| 3D-CDC-NAS2 (TIP21)         | RGB + Depth        | 94.38              |
| BM-NAS (ours)               | RGB + Depth        | 94.96 ± 0.07       |

Table 4: Model size and performance on NTU RGB-D.

| Method        | Dataset   | Parameters   |   Acc(%) |
|---------------|-----------|--------------|----------|
| MMTM(CVPR20)  | NTU       | 8.61M        |    88.92 |
| MFAS (CVPR19) | NTU       | 2.16M        |    89.50 |
| BM-NAS (ours) | NTU       | 0.98M        |    90.48 |

Table 5: Search cost (GPU · hours) of generalized multimodal NAS methods.

| Method        |   MM-IMDB |    NTU |
|---------------|-----------|--------|
| MFAS (CVPR19) |      9.24 | 603.64 |
| BM-NAS (ours) |      0.89 |   38.6 |

3 groups of backbone models trained under different video frame rates (8, 16 and 32 FPS), our BM-NAS only requires the 32 FPS ones, and is generalized to all kinds of modalities. In general, BM-NAS achieves a state-of-the-art fusion performance, showing that BM-NAS is effective for enhancing gesture recognition performance on EgoGesture dataset.

## 4.4 Computing Efficiency

Model size. Table 4 compares the model sizes of different multimodal fusion methods on NTU RGB-D (Shahroudy et al. 2016). All three methods share exactly the same unimodal backbones. Compared with the manually designed fusion model MMTM (Joze et al. 2020) and the fusion model searched by MFAS (P´ erez-R´ ua et al. 2019), our BM-NAS achieves better performance with fewer model parameters. Search cost. Table 5 compares the search cost of generalized multimodal NAS frameworks including MFAS and our BM-NAS. Thanks to the efficient differentiable architecture search framework (Liu, Simonyan, and Yang 2019), BMNAS is at least 10x faster than MFAS when searching on MM-IMDB (Arevalo et al. 2017) and NTU RGB-D.

## 4.5 Ablation Study

In this section, we conduct ablation study to verify the effectiveness of the unimodal feature selection strategy and the multimodal fusion strategy, respectively.

Table 6: Ablation study for feature selection.

| Features          | Dataset   | Accuracy(%)   |
|-------------------|-----------|---------------|
| Random            | NTU       | 86.35 ± 0.68  |
| Late fusion       | NTU       | 89.49 ± 0.15  |
| Searched (MFAS)   | NTU       | 89.50 ± 0.60  |
| Searched (BM-NAS) | NTU       | 90.48 ± 0.24  |

Table 7: Ablation study for fusion strategy.

| Fusion   | Framework            | Dataset   |   Acc (%) |
|----------|----------------------|-----------|-----------|
| Sum      | DARTS (ICLR19)       | NTU       |     87.64 |
| ConcatFC | MFAS (CVPR19)        | NTU       |     89.20 |
| MHA      | Transformer (NIPS17) | NTU       |     88.29 |
| AoA      | AoANet (ICCV19)      | NTU       |     89.11 |
| Searched | BM-NAS               | NTU       |     90.48 |

Unimodal feature selection. Table 6 compares different unimodal feature selection strategies on NTU RGB-D. We compare the best strategy found by BM-NAS against random selection, late fusion, and the best strategy found by MFAS. For all the random baselines, the inner structure of Cells are the same. We randomly selects the input features and the connections between Cells, and report the result averaged over 5 trials. For the late fusion baseline, we concatenate feature pair ( Video 4 , Skeleton 4 ) in Fig. 5. MFAS selects four feature pairs: ( Video 4 , Skeleton 4 ), ( Video 2 , Skeleton 4 ), ( Video 2 , Skeleton 2 ), and ( Video 4 , Skeleton 4 ). As shown in Table 6, the searched feature selection strategy is better than all baselines, demonstrating that a better unimodal feature selection strategy benefits the multimodal fusion performance. As shown in Table 6, the searched feature selection strategy is better than all baselines, demonstrating that a better unimodal feature selection strategy benefits the multimodal fusion performance.

Multimodal fusion strategy. Table 7 evaluates different multimodal fusion strategies on NTU RGB-D. All the strategies in Table 7 adopt the same feature selection strategy. We compare the best Cell structure found by BM-NAS against the summation used in DARTS (Liu, Simonyan, and Yang 2019), the ConcatFC used in MFAS (P´ erez-R´ ua et al. 2019), the multi-head attention (MHA) used in Transformer (Vaswani et al. 2017), and the attention on attention (AoA) used in AoANet (Huang et al. 2019). All these fusion strategies can be formed as certain combinations of our predefined primitive operations, as shown in Fig. 2. In Table 7, the fusion strategy derived by BM-NAS outperforms the baseline strategies, showing the effectiveness of searching fusion strategy for multimodal fusion models.

## 5 Conclusion

In this paper, we have presented a novel multimodal NAS framework BM-NAS to learn the architectures of multimodal fusion models via a bilevel searching scheme. To our best knowledge, BM-NAS is the first NAS framework that supports to search both the unimodal feature selection and the multimodal fusion strategies for multimodal DNNs. In experiments, we have demonstrated the effectiveness and efficiency of BM-NAS on various multimodal learning tasks.

## A Datasets and Tasks

In this work we evaluate the BM-NAS on three multimodal tasks, including (1) the multi-label movie genre classification task on MM-IMDB dataset (Arevalo et al. 2017), (2) the multimodal action recognition task on NTU RGB-D dataset (Shahroudy et al. 2016), and (3) the multimodal gesture recognition task on EgoGesture dataset (Zhang et al. 2018). Examples of these tasks are shown in Fig. 6.

Figure 6: Examples of the evaluation datasets.

<!-- image -->

## B Search Configurations

## B.1 Hyper-parameters

We describe the detailed hyper-parameter configurations on MM-IMDB (Arevalo et al. 2017), NTU RGB-D (Shahroudy et al. 2016), and EgoGesture (Zhang et al. 2018) datasets in

Table 8, where the notations are discussed in the following.

Cells and steps. C is the channels, L is length. In the paper, we refer (C, L) as inner representation size. N is the number of cells, M is the number of steps in each cell.

Basic training settings. Ep is the number of epochs during the searching stage. In the evaluation stage, it could be larger and we roughly set it between 50 and 70 in the experiments. BS is the batch size and Drpt is the Dropout rate (Srivastava et al. 2014). BS and Drpt is the same for both the searching stage and the evaluation stage.

Table 8: Top-4 Configurations on NTU (Shahroudy et al. 2016) and EgoGesture (Zhang et al. 2018) datasets, and the best configuration on MM-IMDB (Arevalo et al. 2017) dataset.

| Dataset   | ID   | Cells and Steps   | Cells and Steps   | Cells and Steps   | Cells and Steps   | Basic   | Basic   | Basic   | Arch Optim   | Arch Optim   | Network Optim   | Network Optim   | Network Optim   | Model Size   | Search Time   | Search Score   | Eval Score   |
|-----------|------|-------------------|-------------------|-------------------|-------------------|---------|---------|---------|--------------|--------------|-----------------|-----------------|-----------------|--------------|---------------|----------------|--------------|
| Dataset   | ID   | C                 | L                 | N                 | M                 | Ep      | BS      | Drpt    | LR           | L2           | MaxLR           | MinLR           | L2              | Model Size   | Search Time   | Search Score   | Eval Score   |
| NTU       | 1    | 128               | 8                 | 2                 | 2                 | 30      | 96      | 0.2     | 3e-4         | 1e-3         | 1e-3            | 1e-6            | 1e-4            | 0.98M        | 53.68         | 94.48          | 90.48        |
| NTU       | 2    | 256               | 8                 | 2                 | 1                 | 30      | 96      | 0.2     | 3e-4         | 1e-3         | 1e-3            | 1e-6            | 1e-4            | 1.71M        | 47.76         | 94.16          | 89.19        |
| NTU       | 3    | 128               | 8                 | 2                 | 1                 | 30      | 96      | 0.2     | 3e-4         | 1e-3         | 1e-3            | 1e-6            | 1e-4            | 0.98M        | 45.84         | 93.01          | 88.78        |
| NTU       | 4    | 256               | 8                 | 4                 | 2                 | 30      | 96      | 0.2     | 3e-4         | 1e-3         | 1e-3            | 1e-6            | 1e-4            | 2.57M        | 58.64         | 92.22          | 88.30        |
| Ego       | 1    | 128               | 8                 | 2                 | 3                 | 7       | 72      | 0.0     | 3e-4         | 1e-3         | 3e-3            | 1e-6            | 1e-4            | 0.61M        | 20.67         | 98.87          | 94.96        |
| Ego       | 2    | 128               | 8                 | 1                 | 2                 | 7       | 72      | 0.2     | 3e-4         | 1e-3         | 1e-2            | 1e-6            | 1e-4            | 0.45M        | 27.60         | 98.58          | 94.45        |
| Ego       | 3    | 192               | 8                 | 4                 | 2                 | 7       | 72      | 0.2     | 3e-4         | 1e-3         | 3e-3            | 1e-6            | 1e-4            | 1.17M        | 36.82         | 98.56          | 93.25        |
| Ego       | 4    | 192               | 12                | 2                 | 2                 | 7       | 72      | 0.0     | 3e-4         | 1e-3         | 3e-3            | 1e-6            | 1e-4            | 1.59M        | 33.62         | 98.60          | 94.33        |
| MM IMDB   | 1    | 192               | 16                | 2                 | 1                 | 12      | 96      | 0.1     | 3e-4         | 1e-3         | 1e-3            | 1e-6            | 1e-4            | 0.65M        | 1.24          | 53.44          | 62.92        |

Architecture optimization. For architecture parameter optimization, we use the Adam (Kingma and Ba 2014) optimizer. The architecture parameters control the structures of the cells and steps, i.e ., α, β, γ in the paper. LR is the learning rate. L2 is the weight decay term.

Network optimization. For the network parameters, we use the Adam (Kingma and Ba 2014) optimizer with a Cosine Annealing scheduler (Loshchilov and Hutter 2016) . The network parameters are trainable parameters from the fusion network , including the reshaping layers, cells and the classifier. MaxLR and MinLR are the learning rate boundaries used by the Cosine Annealing scheduler (Loshchilov and Hutter 2016). L2 is the the weight decay term.

Model size and search time. Model Size is the total number of parameters of the fusion network (in millions), excluding the backbone models. Search Time is the time taken for the searching stage (GPU·hours). We use 8 NVIDIA M40 GPUs in our experiments.

Searching and evaluation scores. The Search Score is the performance of the hypernet in searching stage on the validation set. The Eval Score is the performance of the fusion network in evaluation stage on the test set. For MM-IMDB (Arevalo et al. 2017) dataset, it includes a multi-label classification task and we use the Weighted F1 score (F1-W) as the metric for performance measurement. For NTU RGBD(Shahroudy et al. 2016) dataset and EgoGesture (Zhang et al. 2018) dataset, we use the classification accuracy (%) as the metric.

## B.2 Analysis

To better understand the proposed BM-NAS framework, we empirically study various search configurations of BM-NAS on NTU RGB-D (Shahroudy et al. 2016) and EgoGesture (Zhang et al. 2018). We list the best configurations we found on NTU RGB-D and EgoGesture in Table 8. in conjunction with the val/test accuracies. The task on MM-IMDB dataset (Arevalo et al. 2017) is easier than these two datasets, doesn't require much effort on hyper-parameter tuning, so we only list the best configuration.

Table 8 suggests that N = 2 might be a good choice. We find that when setting N = 1 , BM-NAS would lean to selecting the late fusion strategy ( i.e ., selecting the last features of backbones). But as shown in Table 6 in the paper, late fusion may not be the best choice. Regarding the number of steps M , M = 2 already includes many existing fusion strategies (as denoted by Fig. 3 in the paper), while M = 3 makes a slightly larger search space. We observe that larger N and M may easily lead to overfitting, as there is a total of N × M inner steps in a Cell.

With the search configurations in Table 8, Fig. 8 shows the validation accuracies of the hypernets during search. Fig. 7 further compares the performances of the hypernets, and, compares the performances of the sampled architectures. Figs. 8 and 7 show that the performances of different search configurations of BM-NAS are consistent over searching and evaluation. It suggests that we can select good search configurations according to the validation performance of hypernets instead of performing additional evaluation on the test set with the sampled architecture.

Figure 7: Performance of hypernets (searching stage) and sampled fusion networks structures (evaluation stage).

<!-- image -->

Figure 8: The validation accuracy of hypernets in searching stage. Results on NTU RGB-D (Shahroudy et al. 2016) and EgoGesture (Zhang et al. 2018) are reported.

<!-- image -->

## C Found Architectures

## C.1 NTU RGB-D Dataset

We tune the hyper parameters extensively on NTU RGBD (Shahroudy et al. 2016) dataset. The top-4 configurations are shown in Table 8, and the architectures found under these configurations are shown in Fig. 9. The 'NTU Config 1' is the best architecture found by our BM-NAS framework.

For feature selection strategy, we find that Video 3 , Video 4 , and Skeleton 4 are always selected by our BMNAS framework no matter how many Cells and steps used. It indicates these are the most effective modality features. Especially Video 3 is strongly favored in all the found architectures. MFAS (P´ erez-R´ ua et al. 2019) also selects Video 4 and Skeleton 4 in every found architectures, but it does not pay much attention to Video 3 .

For fusion strategy, we find that adding more inner steps (increasing M ) is more effective than adding more cells (increasing N ). However, since we have N × M steps in total, setting N or M too large would easily lead to an overfitting. Roughly we find that setting N = 2 , M = 2 is a good option. N = 2 means we have two different feature pairs for Cells, which is sufficient to cover the three most important features Video 3 , Video 4 and Skeleton 4 . And M = 2 is sufficient for BM-NAS to form all the fusion strategy like concatenation, attention on attention (AoA) (Huang et al. 2019), etc ., as shown in the paper. The best fusion strategy found by BM-NAS on NTU is very similar to an AoA (Huang et al. 2019) module, see 'NTU Config 1' in Fig. 9.

Figure 9: The top-4 architectures found by BM-NAS on NTU (Shahroudy et al. 2016) dataset. 'NTU Config 1' is the best architecture found on NTU dataset. 'C1 S1' denotes Step (1) of Cell (1) , and so on. The blue edges are the connections at the upper level, and the dark edges are the connections at the lower level.

<!-- image -->

## C.2 EgoGesture Dataset

For the experiments on EgoGesture (Zhang et al. 2018) dataset, we basically follow the settings as those in NTU RGB-D dataset. The top-4 configurations are shown in Table 8, and the architectures found under these configurations are shown in Fig. 10. The 'Ego Config 1' is the best architecture found by our BM-NAS framework.

For feature selection strategy, we find Depth 1 , Depth 2 , and RGB 2 are the most important features for EgoGesture (Zhang et al. 2018).

For fusion strategy, we find that a combination of Sums is the most effective, shown as 'Ego Config 1' in Fig. 10, probably because the backbone models share the same architecture. Unlike the experiments on NTU RGB-D (Shahroudy et al. 2016) which use Inflated ResNet-50 (Baradel et al. 2018) for RGB videos and Co-occurrence (Li et al. 2018) for skeletons modality, EgoGesture (Zhang et al. 2018) uses ResNeXt-101 (K¨ op¨ ukl¨ u et al. 2019) backbone for both the depth videos and the RGB videos. These two backbone models have exactly the same architecture, except for the input channels of the first convolutional layer. Therefore, the depth features and the RGB features probably share the semantic levels for features of the same depths, such as Depth 2 and RGB 2 in 'Ego Config 1'.

Figure 10: The top-4 architectures found by BM-NAS on EgoGesture(Zhang et al. 2018) dataset. 'Ego Config 1' is the best architecture found on EgoGesture dataset.

<!-- image -->

Figure 11: MM-IMDB Config 1, which is the best architecture found by BM-NAS on MM-IMDB (Arevalo et al. 2017) dataset.

<!-- image -->

We do not tune the hyper-parameters extensively on MMIMDB (Arevalo et al. 2017) since it is a relatively simple task compared with NTU RGB-D (Shahroudy et al. 2016) and EgoGesture (Zhang et al. 2018). The configuration can be found in Table 8. As shown in Fig. 11, we find Image 2 and Text 0 are the most important modality features. The best fusion operation is ConcatFC for Image 2 and Text 0 , and LinearGLU for Cell 0 and Text 0 .

It is worth noting that we use the Weighted F1 score (F1W) as the metric for performance, since we perform a multilabel classification task on MM-IMDB(Arevalo et al. 2017) dataset. Although the Macro F1 score (F1-M) is also reported in the paper, we only use F1-W for model selection, because the distribution of labels in MM-IMDB (Arevalo et al. 2017) is highly imbanlanced as illustrated in Table 9. Thus, F1-W would be a better metric as F1-M does not take label imbalance into account.

Table 9: Label distribution of MM-IMDB(Arevalo et al. 2017) dataset.

| Label       |   #Samples | Label      |   #Samples |
|-------------|------------|------------|------------|
| Drama       |      13967 | War        |       1335 |
| Comedy      |       8592 | History    |       1143 |
| Romance     |       5364 | Music      |       1045 |
| Thriller    |       5192 | Animation  |        997 |
| Crime       |       3838 | Musical    |        841 |
| Action      |       3550 | Western    |        705 |
| Adventure   |       2710 | Sport      |        634 |
| Horror      |       2703 | Short      |        471 |
| Documentary |       2082 | Film-Noir  |        338 |
| Mystery     |       2057 | News       |         64 |
| Sci-Fi      |       1991 | Adult      |          4 |
| Fantasy     |       1933 | Talk-Show  |          2 |
| Family      |       1668 | Reality-TV |          1 |
| Biography   |       1343 |            |            |

## References

Amodei, D.; Ananthanarayanan, S.; Anubhai, R.; Bai, J.; Battenberg, E.; Case, C.; Casper, J.; Catanzaro, B.; Cheng, Q.; Chen, G.; et al. 2016. Deep speech 2: End-to-end speech recognition in english and mandarin. In ICML , 173-182.

Anderson, P.; He, X.; Buehler, C.; Teney, D.; Johnson, M.; Gould, S.; and Zhang, L. 2018. Bottom-up and top-down attention for image captioning and visual question answering. In CVPR , 6077-6086.

Arevalo, J.; Solorio, T.; Montes-y G´ omez, M.; and Gonz´ alez, F. A. 2017. Gated multimodal units for information fusion. arXiv preprint arXiv:1702.01992 .

Baradel, F.; Wolf, C.; Mille, J.; and Taylor, G. W. 2018. Glimpse clouds: Human activity recognition from unstructured feature points. In CVPR , 469-478.

Chen, Y.-C.; Li, L.; Yu, L.; Kholy, A. E.; Ahmed, F.; Gan, Z.; Cheng, Y.; and Liu, J. 2019. Uniter: Learning universal image-text representations. arXiv preprint arXiv:1909.11740 .

Dauphin, Y. N.; Fan, A.; Auli, M.; and Grangier, D. 2017. Language modeling with gated convolutional networks. In ICML , 933-941.

Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; and FeiFei, L. 2009. Imagenet: A large-scale hierarchical image database. In CVPR , 248-255.

Devlin, J.; Chang, M.-W.; Lee, K.; and Toutanova, K. 2018. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805 .

Gao, P.; Jiang, Z.; You, H.; Lu, P.; Hoi, S. C.; Wang, X.; and Li, H. 2019. Dynamic fusion with intra-and inter-modality attention flow for visual question answering. In CVPR , 6639-6648.

Goodfellow, I. J.; Warde-Farley, D.; and Courville, M. M. A. 2013. Bengio, Yoshua. Maxout networks. In ICML , 13191327.

Gupta, V.; Dwivedi, S. K.; Dabral, R.; and Jain, A. 2019. Progression Modelling for Online and Early Gesture Detection. In 3DV , 289-297. IEEE.

He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , 770-778.

Huang, L.; Wang, W.; Chen, J.; and Wei, X.-Y. 2019. Attention on attention for image captioning. In ICCV , 4634-4643.

Hutter, F.; Hoos, H. H.; and Leyton-Brown, K. 2011. Sequential model-based optimization for general algorithm configuration. In International conference on learning and intelligent optimization , 507-523. Springer.

Jin, T.; Huang, S.; Chen, M.; Li, Y.; and Zhang, Z. 2020. SBAT: Video Captioning with Sparse BoundaryAware Transformer. In IJCAI .

Jin, T.; Huang, S.; Li, Y.; and Zhang, Z. 2019. LowRank HOCA: Efficient High-Order Cross-Modal Attention for Video Captioning. In EMNLP , 2001-2011.

Joze, H. R. V.; Shaban, A.; Iuzzolino, M. L.; and Koishida, K. 2020. MMTM: Multimodal Transfer Module for CNN Fusion. In CVPR , 13289-13299.

Kandasamy, K.; Neiswanger, W.; Schneider, J.; Poczos, B.; and Xing, E. P. 2018. Neural architecture search with bayesian optimisation and optimal transport. In NeurIPS , 2016-2025.

Kingma, D. P.; and Ba, J. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 .

K¨ op¨ ukl¨ u, O.; Gunduz, A.; Kose, N.; and Rigoll, G. 2019. Real-time hand gesture detection and classification using convolutional neural networks. In FG , 1-8. IEEE.

Krizhevsky, A.; Sutskever, I.; and Hinton, G. E. 2012. ImageNet Classification with Deep Convolutional Neural Networks. In NeurIPS .

Li, C.; Zhong, Q.; Xie, D.; and Pu, S. 2018. Co-occurrence feature learning from skeleton data for action recognition and detection with hierarchical aggregation. In Proceedings of the 27th International Joint Conference on Artificial Intelligence IJCAI , 786-792.

Li, L. H.; Yatskar, M.; Yin, D.; Hsieh, C.-J.; and Chang, K.W. 2019. Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557 .

Liu, C.; Zoph, B.; Neumann, M.; Shlens, J.; Hua, W.; Li, L.J.; Fei-Fei, L.; Yuille, A.; Huang, J.; and Murphy, K. 2018a. Progressive neural architecture search. In ECCV , 19-34.

Liu, C.; Zoph, B.; Neumann, M.; Shlens, J.; Hua, W.; Li, L.J.; Fei-Fei, L.; Yuille, A.; Huang, J.; and Murphy, K. 2018b. Progressive neural architecture search. In ECCV , 19-34.

Liu, H.; Simonyan, K.; and Yang, Y. 2019. Darts: Differentiable architecture search.

Loshchilov, I.; and Hutter, F. 2016. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983 .

Lu, J.; Batra, D.; Parikh, D.; and Lee, S. 2019. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In NeurIPS , 13-23.

Lu, J.; Yang, J.; Batra, D.; and Parikh, D. 2016. Hierarchical question-image co-attention for visual question answering. In NeurIPS , 289-297.

Nair, V.; and Hinton, G. E. 2010. Rectified linear units improve restricted boltzmann machines. In ICML .

Natarajan, P.; Wu, S.; Vitaladevuni, S.; Zhuang, X.; Tsakalidis, S.; Park, U.; Prasad, R.; and Natarajan, P. 2012. Multimodal feature fusion for robust event detection in web videos. In CVPR , 1298-1305. IEEE.

Negrinho, R.; and Gordon, G. 2017. Deeparchitect: Automatically designing and training deep architectures. arXiv preprint arXiv:1704.08792 .

Peng, Y.; Bi, L.; Fulham, M.; Feng, D.; and Kim, J. 2020. Multi-modality Information Fusion for Radiomics-Based Neural Architecture Search. In MICCAI , 763-771. Springer.

P´ erez-R´ ua, J.-M.; Vielzeuf, V.; Pateux, S.; Baccouche, M.; and Jurie, F. 2019. Mfas: Multimodal fusion architecture search. In CVPR , 6966-6975.

Pham, H.; Guan, M. Y.; Zoph, B.; Le, Q. V.; and Dean, J. 2018. Efficient neural architecture search via parameter sharing. arXiv preprint arXiv:1802.03268 .

Real, E.; Aggarwal, A.; Huang, Y.; and Le, Q. V. 2019. Regularized evolution for image classifier architecture search. In AAAI , volume 33, 4780-4789.

Reed, S.; Akata, Z.; Yan, X.; Logeswaran, L.; Schiele, B.; and Lee, H. 2016. Generative adversarial text to image synthesis. arXiv preprint arXiv:1605.05396 .

Shahroudy, A.; Liu, J.; Ng, T.-T.; and Wang, G. 2016. Ntu rgb+ d: A large scale dataset for 3d human activity analysis. In CVPR , 1010-1019.

Simonyan, K.; and Zisserman, A. 2014. Two-stream convolutional networks for action recognition in videos. In NeurIPS , 568-576.

Simonyan, K.; and Zisserman, A. 2015. Very Deep Convolutional Networks for Large-Scale Image Recognition. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings .

Srivastava, N.; Hinton, G.; Krizhevsky, A.; Sutskever, I.; and Salakhutdinov, R. 2014. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research , 15(1): 1929-1958.

Tan, H.; and Bansal, M. 2019. Lxmert: Learning crossmodality encoder representations from transformers. arXiv preprint arXiv:1908.07490 .

Teney, D.; Anderson, P.; He, X.; and Van Den Hengel, A. 2018. Tips and tricks for visual question answering: Learnings from the 2017 challenge. In CVPR , 4223-4232.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. In NeurIPS , 5998-6008.

Vielzeuf, V.; Lechervy, A.; Pateux, S.; and Jurie, F. 2018. Centralnet: a multilayer approach for multimodal fusion. In ECCV , 0-0.

Xie, S.; Girshick, R.; Doll´ ar, P.; Tu, Z.; and He, K. 2017. Aggregated residual transformations for deep neural networks. In CVPR , 1492-1500.

You, Q.; Jin, H.; Wang, Z.; Fang, C.; and Luo, J. 2016. Image captioning with semantic attention. In CVPR , 4651-4659.

Yu, Z.; Cui, Y.; Yu, J.; Wang, M.; Tao, D.; and Tian, Q. 2020. Deep Multimodal Neural Architecture Search. arXiv preprint arXiv:2004.12070 .

Yu, Z.; Zhou, B.; Wan, J.; Wang, P.; Chen, H.; Liu, X.; Li, S. Z.; and Zhao, G. 2021. Searching multi-rate and multimodal temporal enhanced networks for gesture recognition. IEEE Transactions on Image Processing .

Zhang, Y.; Cao, C.; Cheng, J.; and Lu, H. 2018. Egogesture: a new dataset and benchmark for egocentric hand gesture recognition. IEEE Transactions on Multimedia , 20(5): 1038-1050.

Zhou, X.; Huang, S.; Li, B.; Li, Y.; Li, J.; and Zhang, Z. 2019. Text guided person image synthesis. In CVPR , 36633672.

Zoph, B.; and Le, Q. V. 2016. Neural architecture search with reinforcement learning. arXiv preprint arXiv:1611.01578 .

Zoph, B.; and Le, Q. V. 2017. Neural architecture search with reinforcement learning. In ICLR .