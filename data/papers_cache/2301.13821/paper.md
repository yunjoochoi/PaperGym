## Complete Neural Networks for Complete Euclidean Graphs

Snir Hordan 1 , Tal Amir 1 , Steven J. Gortler 2 , Nadav Dym 1, 3

1 Faculty of Mathematics, Technion - Israel Institute of Technology, Haifa, Israel

2 School of Engineering and Applied Sciences, Harvard University, Cambridge, USA

3 Faculty of Computer Science, Technion - Israel Institute of Technology, Haifa, Israel

{ snirhordan, talamir } @campus.technion.ac.il, sjg@cs.harvard.edu, nadavdym@technion.ac.il

## Abstract

Neural networks for point clouds, which respect their natural invariance to permutation and rigid motion, have enjoyed recent success in modeling geometric phenomena, from molecular dynamics to recommender systems. Yet, to date, no model with polynomial complexity is known to be complete , that is, able to distinguish between any pair of nonisomorphic point clouds. We fill this theoretical gap by showing that point clouds can be completely determined, up to permutation and rigid motion, by applying the 3-WL graph isomorphism test to the point cloud's centralized Gram matrix. Moreover, we formulate an Euclidean variant of the 2-WL test and show that it is also sufficient to achieve completeness. We then show how our complete Euclidean WL tests can be simulated by an Euclidean graph neural network of moderate size and demonstrate their separation capability on highly symmetrical point clouds.

## Introduction

A point cloud is a collection of n points in R d , where typically in applications d = 3 . Machine learning on point clouds is an important task with applications in chemistry (Gilmer et al. 2017; Wang et al. 2022), physical systems (Finzi, Welling, and Wilson 2021), and image processing (Ma et al. 2023). Many successful architectures for point clouds are invariant by construction to the natural symmetries of point clouds: permutations and rigid motions.

The rapidly increasing literature on point-cloud networks with permutation and rigid-motion symmetries has motivated research aimed at theoretically understanding the expressive power of the various architectures. This analysis typically focuses on two closely related concepts: Separation and Universality . We say an invariant architecture is separating , or complete , if it can assign distinct values to any pair of point clouds that are not related by symmetry. An invariant architecture is universal if it can approximate all continuous invariant functions on compact sets. Generally speaking, these two concepts are essentially equivalent, as discussed in (Villar et al. 2021; Joshi et al. 2022; Chen et al. 2019), and in our context, in Appendix A.

Dym and Maron (2021) proved that the well-known Tensor Field Network (Thomas et al. 2018) invariant archi- tecture is universal, but the construction in their proof requires arbitrarily high-order representations of the rotation group. Similarly, universality can be obtained using highorder representations of the permutation group (Lim et al. 2022). However, before this work, it was not known whether the same theoretical guarantees can be achieved by realistic point-cloud architectures that use low-dimensional representations, and whose complexity has a mild polynomial dependency on the data dimension. In the words of (Pozdnyakov and Ceriotti 2022): '...provably universal equivariant frameworks are such in the limit in which they generate high-order correlations. . . It is an interesting, and open, question whether a given order suffices to guarantee complete resolving power. ' (p. 6). We note that it is known that separation of point clouds in polynomial time in n is possible, assuming that d is fixed (e.g., d = 3 ) (Arvind and Rattan 2016; Dym and Kovalsky 2019; Kurlin 2024). What still remains to be established is whether separation is achievable for common invariant machine learning models, and more generally, whether separation can be achieved by computing a continuous invariant feature that is piecewise differentiable.

Copyright © 2024, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

In this paper, we give what seems to be the first positive answer to this question. We focus on analyzing a popular method for the construction of invariant point-cloud networks via Graph Neural Networks (GNNs) . This is done in two steps: first, point clouds are represented as a Euclidean graph - which we define to be a complete weighted graph whose edge features are simple, rotation-invariant features: the inner products between pairs of (centralized) points. We then apply permutation-invariant Graph Neural Networks (GNNs) to the Euclidean graphs to obtain a rotation- and permutation-invariant global point-cloud feature. This leads to a rich family of invariant point-cloud architectures, which is determined by the type of GNN chosen.

The most straightforward implementation of this idea would be to apply the popular message passing GNNs to the Euclidean graphs. One could also consider applying more expressive GNNs. For combinatorial graphs, it is known that message-passing GNNs are only as expressive as the 1-WL graph isomorphism test. There exists a hierarchy of k -WL graph isomorphism tests, where larger values of k correspond to more expressive, and more expensive, graph isomorphism tests. There are also corresponding GNNs that simulate the k -WL tests and have an equivalent separation power (Morris et al. 2019; Maron et al. 2019). One could then consider applying these more expressive architectures to Euclidean graphs, as suggested by Lim et al. (2022). Accordingly, we aim to answer the following questions:

Question 1 For which k is the k -WL test, when applied to Euclidean graphs, complete?

Question 2 Can this test be implemented in polynomial time by a continuous, piecewise-differentiable architecture?

We begin by addressing Question 1. First, we consider a variation of the WL-test adapted for point clouds, which we refer to as 1 -EWL ('E' for Euclidean). This test was first proposed by Pozdnyakov and Ceriotti (2022), where it was shown that it cannot distinguish between all 3 -dimensional point clouds, and consequently, neither can GNNs like SchNet (Sch¨ utt et al. 2017), which can be shown to simulate it. Our first result balances this by showing that two iterations of 1 -EWL are enough to separate almost any pair of point clouds.

To achieve complete separation for all point clouds, we consider higher-order k -EWL tests. We first consider a natural adaptation of k -WL for Euclidean graphs, which we name the Vanillak -EWL test. In this test, the standard k -WL is applied to the Euclidean graph induced by the point clouds. We show that when k = 3 , this test is complete for 3 -dimensional point clouds. Additionally, we propose a variant of the Vanilla 2 -EWL, which incorporates additional geometric information while having the same complexity. We call this test the 2 -EWL test and show that it is complete on 3D point clouds. We also propose a natural variation of 2 -EWL called 2 -SEWL , which can distinguish between point clouds that are related by a reflection. This ability is important for chemical applications, as most biological molecules that are related by a reflection are not chemically identical (Kapon et al. 2021) (this molecular property is called chirality ).

We next address the second question of how to construct a GNN for Euclidean data with the same separation power as that of the various k -EWL tests we describe. For combinatorial graphs, such equivalence results rely on injective functions defined on multisets of discrete features (Xu et al. 2019). For Euclidean graphs, one can similarly rely on injective functions for multisets with continuous features, such as those proposed in (Dym and Gortler 2023). However, a naive application of this approach leads to a very large number of hidden features, which grows exponentially with the number of message-passing iterations (see Figure 3). We show how this problem can be remedied, so that the number of features needed depends only linearly on the number of messagepassing iterations.

To summarize, our main results in this paper are:

1. We show that two iterations of 1 -EWL can separate almost all point clouds in any dimension.
2. We prove the completeness of a single iteration of the vanilla 3 -EWL for point clouds in R 3 .
3. We formulate the 2 -SEWL and 2 -EWL tests, and prove their completeness for point clouds in R 3 .
4. We explain how to build differentiable architectures for

point clouds with the same separation power as Euclidean k -WL tests, with reasonable complexity.

Experiments We present synthetic experiments that demonstrate that 2 -SEWL can separate challenging pointcloud pairs that cannot be separated by several popular architectures.

Disambiguation: Euclidean Graphs In this paper we use a simple definition of a Euclidean graph as the centralized Gram matrix of a point cloud (centralizing the point cloud and then calculating its Gram matrix) and focus on a fundamental theoretical question related to this representation. In the learning literature, terms like 'geometric graphs' (not used here) could refer to graphs that have both geometric and non-geometric edge and vertex features, or graphs where pairwise distances are only available for specific point pairs (edges in an incomplete graph).

## Related Work

Euclidean WL Pozdnyakov and Ceriotti (2022) showed that 1 -EWL is incomplete for 3-dimensional point clouds. Joshi et al. (2022) define separation as a more general definition of geometric graph, which combines geometric and combinatorial features. This work holds various interesting insights for this more general problem but they do not prove completeness as we do here.

Other complete constructions As mentioned earlier, Dym and Maron (2021) proved universality with respect to permutations and rigid motions for architectures using highdimensional representations of the rotation group. Similar results were obtained in (Finkelshtein et al. 2022; Gasteiger, Becker, and G¨ unnemann 2021). In (Lim et al. 2022), universality was proven for Euclidean GNNs with very highorder permutation representations. In the planar case d = 2 , universality using low-dimensional features was achieved in (B¨ okman, Kahl, and Flinth 2022). For d ≥ 3 our construction seems to be the first to achieve universality using low dimensional representations.

For general fixed d , there do exist continuous algorithms that can separate point clouds up to equivalence in polynomial time, but they do not seem to lend themselves directly to neural architectures. In (Kurlin 2024; Widdowson and Kurlin 2023) a polynomial-time algorithm is introduced for computing invariants that completely determine a point cloud that is Lipschitz-continuous, yet these invariants are represented as a 'multi-set of multi-sets' and thus do not allow for gradient-descent-based optimization. Our approach is continuous and represents the point cloud as a vector in real space, which allows for back-propagation. Efficient tests for equivalence of Euclidean graphs were described by Brass and Knauer (2000); Arvind and Rattan (2016), but they compute features that do not depend continuously on the point cloud.

Weaker notions of universality Widdowson and Kurlin (2022) suggest a method for distinguishing almost every point clouds up to equivalence, similar to our result here on 1 -EWL. Similarly, efficient separation/universality can also be obtained for point clouds with distinct principal axes

(Puny et al. 2021; Kurlin 2024). Another setting in which universality is easier to obtain is when only rigid symmetries are considered and permutation symmetries are ignored (Wang et al. 2022; Villar et al. 2021; Satorras, Hoogeboom, and Welling 2021). All these results do not provide universality for all point clouds, with respect to the joint action of permutations and rigid motions.

## Mathematical Notation

A (finite) multiset { { y 1 , . . . , y N } } is an unordered collection of elements where repetitions are allowed.

Let G be a group acting on a set X . For X,Y ∈ X , we say that X = G Y if Y = gX for some g ∈ G . We say that a function f : X → Y is invariant if f ( gx ) = f ( x ) for all x ∈ X,g ∈ G . We say that f is equivariant if Y is also endowed with some action of G and f ( gx ) = gf ( x ) for all x ∈ X , g ∈ G . A separating invariant mapping is an invariant mapping that is injective, up to group equivalence:

Definition 1 (Separating Invariant) . Let G be a group acting on a set X . We say F : X → R K is a G -separating invariant with embedding dimension K if for all X,Y ∈ X , F ( X ) = F ( Y ) ⇔ X = G Y .

We focus on the case where X is some Euclidean domain. To enable gradient-based learning, we shall need separating mappings that are continuous everywhere and differentiable almost everywhere.

The symmetry group we consider for point clouds ( x 1 , . . . , x n ) ∈ R d × n is generated by a rotation matrix R ∈ SO ( d ) , and a permutation σ ∈ S n . These act on a point cloud by

<!-- formula-not-decoded -->

We denote this group by SO [ d, n ] . In some instances, reflections R ∈ O ( d ) are also permitted, leading to a slightly larger symmetry group, which we denote by O [ d, n ] . Our goal shall be to construct separating invariants for these groups. For the sake of brevity, we do not discuss translation invariance and separation, as these can easily be achieved by centering the input point clouds, once SO [ d, n ] (or O [ d, n ] ) separating invariants are constructed, see (Dym and Gortler 2023).

For simplicity of notation, throughout this paper, we focus on the case d = 3 . In Appendix A, we explain how our constructions and theorems can be generalized to d &gt; 3 .

## Euclidean Graph Isomorphism Tests

The k -WL Graph Isomorphism Test is a classical paradigm for testing the isomorphism of combinatorial graphs (Weisfeiler and Leman 1968), which we shall now briefly describe. Let G be a graph with vertices indexed by [ n ] = { 1 , 2 , . . . , n } . We denote each ordered k -tuple of vertices by a multi-index i = ( i 1 , . . . , i k ) ∈ [ n ] k . Essentially, for each such k -tuple i , the test maintains a coloring C ( i ) that belongs to a discrete set, and updates it iteratively. First, the coloring of each k -tuple is assigned an initial value that encodes the isomorphism type of the corresponding k -dimensional subgraph:

<!-- formula-not-decoded -->

Then the color of each k -tuple i is iteratively refined according to the colors of its 'neighboring' k -tuples. The update rule is defined as

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

where i [ j \ t ] is the multi-index i with its t -th coordinate replaced by j ; e.g. for t = 1 , i [ j \ 1] = ( j, i 2 , . . . , i k ) . Embed is a function that maps its input injectively to some discrete set. This process is repeated T times to obtain a final coloring { { C ( T ) ( i ) } } i ∈ [ n ] k . A global label is then calculated by

<!-- formula-not-decoded -->

where Embed ( T +1) is a function that maps label-multisets injectively to some discrete set.

̸

To test whether two graphs G and G ′ are isomorphic, the k -WL test computes the corresponding colorings C G and C G ′ for some chosen T . If C G = C G ′ then G and G ′ are guaranteed not to be isomorphic, whereas if C G = C G ′ , then G and G ′ may either be isomorphic or not, and the test does not, in general, provide a decisive answer for combinatorial graphs. It is known that this test can distinguish a strictly larger class of combinatorial graphs for every strict increase in the value of k, i.e. it is a strict hierarchy of tests in terms of distinguishing power (Cai, F¨ urer, and Immerman 1992; Grohe 2017). We note that in the literature some may refer to the above test as k -Folklore -WL, e.g. (Morris et al. 2019).

Vanillak -WL tests As a first step from a combinatorial to a Euclidean setting, we identify each point cloud X = ( x 1 , . . . , x n ) ∈ R d × n with a complete graph on n vertices, wherein each edge ( i, j ) is endowed with the weight w ij ( X ) = ⟨ x i , x j ⟩ . We name such a graph a Euclidean graph . Similarly to k -WL for combinatorial graphs, k -WL for Euclidean graphs maintains a coloring of the k -tuples of vertices. However, the initial color of each k -tuple i is not a discrete label as in the combinatorial case, but rather a k × k matrix of continuous features, which represent all edge weights w ij corresponding to pairs of indices from i . We will call the k -WL test defined by this initial coloring the vanilla k -WL test. This test is invariant by construction to reflections, rotations, and permutations. We note that our definition of the vanilla k -EWL test via inner products follows that of (Lim et al. 2022). Another popular, and essentially equivalent, formulation, uses distances instead.

k -EWLtests An inherent limitation of the Vanilla-1-EWL test is that no pairwise Euclidean information is passed, yielding it rather uninformative. Indeed, (Pozdnyakov and Ceriotti 2022) proposed a Euclidean analog of the 1 -WL test, where the update rule C ( t + 1 ) ( i ) (2) is replaced with

̸

<!-- formula-not-decoded -->

Figure 1: Distance matrices (Left), geometric degree histogram (Right) of pairs of point clouds. The generic pair is a randomly sampled pair of point clouds. Notice each of the nodes in each of the clouds has a distinct geometric degree. The Hard pair exhibits a distinct geometric degree for each node, but only within each point cloud, that is the pair shares an identical geometric degree histogram. The Harder example is a pair of point clouds with identical geometric degree histogram, and each point cloud is comprised of three pairs of points, with each pair having an identical geometric degree. Examples from (Pozdnyakov and Ceriotti 2022) and (Pozdnyakov et al. 2020).

<!-- image -->

We call this test the 1 -EWL test. This formulation is motivated by the fact that many symmetry-preserving networks for point clouds are in fact a realization of it, though they use Embed functions that are continuous and, in general, may assign the same value to different multisets. Consequently, the separation power of these architectures is at most that of 1 -EWL with discrete injective hash functions. Moreover, the separation power will be equivalent if continuous injective multiset functions are used for embedding, as we discuss in the proceeding sections.

The 1 -EWLtest strengthens the Vanilla1 -EWLtest by allowing the messages passed to a node in each step to contain not only previous colorings but also geometric information in the form of pairwise distances. More generally, we shall use the term k -EWL to refer to tests that follow the Euclidean k -WL paradigm, but incorporate geometric invariants into the message-passing procedure. In particular, for point clouds with dimension 3 , we define the 2 -SEWL test ('SE' for Special Euclidean ) by replacing the update step of (2) with

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Note that ⟨ x i × x j , x k ⟩ is equal to the determinant of the 3 × 3 matrix whose rows are the three vectors x i , x j , x k , which makes this a natural choice as all polynomial invariants of SO (3) are generated by these determinants and the inner products we use for the initial coloring (Kraft and Procesi 1996).

Wenote that, Using the fact that O (3) is just two copies of SO (3) , it is not difficult to generalize 2 -SEWLto a complete O [3 , n ] test, which we name 2-EWL. for general d , similar complete ( d -1) -SEWL and ( d -1) -EWL tests can be formulated for point clouds in R d via the Hodge-star operator; see Appendix A for further details.

In the rest of this section, we shall prove that the 2 -SEWL, and vanilla 3 -EWL tests are complete when applied to R 3 × n , even when using a single iteration ( T = 1) . We shall also show that two iterations of the 1 -EWL test are complete, except on a set of measure zero.

## Generic Completeness of 1-EWL

The separation power of 1-EWL is closely linked to the notion of geometric degree : For a point cloud X = ( x 1 , . . . , x n ) , we define the geometric degree d ( i, X ) of the i -th point, and the induced geometric degree histogram d H ( X ) , to be the multisets

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

̸

It is not difficult to see that if d H ( X ) = d H ( Y ) then X and Y can be separated by a single 1 -EWL iteration. An example of such a pair is shown on the left of Figure 1. With two 1 -EWL iterations, we show that can separate X and Y even if d H ( X ) = d H ( Y ) , provided that they both belong to the set of point clouds defined by

̸

̸

<!-- formula-not-decoded -->

Figure 2: A plot of a Gaussian distribution centered at x ∈ R , depicting a target function is shown in blue. In red, a schematic plot of how a Lipschitz continuous function that does not distinguish x from y would model the target function.

<!-- image -->

Such an example, taken from (Pozdnyakov et al. 2020), is visualized in the middle column of Figure 1.

Theorem 2. Two iterations of the 1 -EWL test assign two point clouds X , Y ∈ R 3 × n distinct the same value, if and only if X = O [3 ,n ] Y .

In the appendix, we show that the complement of R 3 × n distinct has measure zero. Thus this result complements long-standing results for combinatorial graphs, stating that 1-WL can classify almost all such graphs as the number of nodes tends to infinity (Babai, Erd˝ os, and Selkow 1980).

The right-most pair of point clouds ('Harder') in Figure 1 is taken from (Pozdnyakov and Ceriotti 2022). The degree histograms of these point clouds are identical, and they are not in R 3 × n distinct . (Pozdnyakov and Ceriotti 2022) show that this pair cannot be separated by any number of 1 -EWL iterations.

## Is 1-EWL All You Need?

Theorem 2 shows that the probability of a failure of the 1 -EWLiszero. A natural question to ask is whether more powerful tests are needed. We believe the answer to this question is yes.

Typical hypothesis classes used for machine learning, such as neural networks, are Lipschitz continuous (Gama, Bruna, and Ribeiro 2020). In this setting, failure to separate on a measure zero set could have implications for non-trivial positive measure, see Figure 2.

## 2-SEWL and Vanilla 3-EWL are Complete

We now prove that the vanilla 3 -EWL test is complete.

Theorem 3. For every X,Y ∈ R 3 × n , a single iteration of the vanilla 3 -EWL test assigns X and Y the same value if and only if X = O [3 ,n ] Y .

Proof. First, it is clear that if X = O [3 ,n ] Y then C G ( X ) =

C G ( Y ) since the vanilla 3 -EWL test is invariant by construction. The challenge is proving the other direction. To this end, let us assume that C G ( X ) = C G ( Y ) , and assume without loss of generality that r := rank( X ) ≥ rank( Y ) . Note that X has rank r ≤ 3 , and so it must contain some three points whose rank is also r . By applying a permutation to X we can assume without loss of generality that these three points are the first three points. The initial coloring C 0 (1 , 2 , 3)( X ) of this triplet is their Gram matrix ( ⟨ x i , x j ⟩ ) 1 ≤ i,j ≤ 3 , which has the same rank r as the space spanned by the three points. Next, since C G ( X ) = C G ( Y ) are the same, there exists a triplet of points i, j, k such that C (1) (1 , 2 , 3)( X ) = C (1) ( i, j, k )( Y ) which implies that the initial colorings are also the same. By applying a permutation to Y we can assume without loss of generality that i = 1 , j = 2 , k = 3 . Next, since the Gram matrix of x 1 , x 2 , x 3 and y 1 , y 2 , y 3 are identical, there is an orthogonal transformation that takes x i to y i for i = 1 , 2 , 3 , and by applying this transformation to all points in X we can assume without loss of generality that x i = y i for i = 1 , 2 , 3 . It remains to show that the rest of the points of X and Y are equal, up to permutation. To see this, first note that X and Y have the same rank since

<!-- formula-not-decoded -->

Thus the space spanned by x 1 = y 1 , x 2 = y 2 , x 3 = y 3 contains all points in X and Y . Next, we can deduce from the aggregation rule defining C 1 (1 , 2 , 3)( X ) in (2), that

<!-- formula-not-decoded -->

Since all points in X and Y belong to the span of x 1 = y 1 , x 2 = y 2 , x 3 = y 3 , X and Y are the same up to permutation of the last n -3 coordinates. This concludes the proof of the theorem.

We next outline the completeness proof of the more efficient 2 -SEWL.

Theorem 4. For every X,Y ∈ R 3 × n , a single iteration of the 2 -SEWL test assigns X and Y the same value if and only if X = SO [3 ,n ] Y .

Proof idea. The completeness of Vanilla3 -EWL was based on the fact that its initial coloring captures the Gram matrix of triplets of vectors that span the space spanned by X , and on the availability of projections onto this basis in the aggregation step defined in (2). Our proof for 2 -SEWL completeness relies on the fact that a pair of non-degenerate vectors x i , x j induces a basis x i , x j , x i × x j of R 3 . The Gram matrix of this basis can be recovered from the Gram matrix of the first two points x i , x j , and the projection onto this basis can be obtained from the extra geometric information we added in (5). A full proof is given in the appendix.

To conclude this section, we note that the above theorem can be readily used to show that the 2 -EWL test is also complete with respect to O [3 , n ] . For details see Appendix A.

## EWL-Equivalent GNNs

In the previous section, we discussed the generic completeness of 1-EWL and the completeness of 2 -SEWLand vanilla 3 -EWL. The Embed functions in these tests are hash functions, which can be redefined independently for each pair of point clouds X,Y . In this section, our goal is to explain how to construct GNNs with equivalent separation power to that of these tests, while choosing continuous, piecewise differentiable Embed functions that are injective. While this question is well studied for combinatorial graphs with discrete features (Xu et al. 2019; Morris et al. 2019; Maron et al. 2019; Aamand et al. 2022), here we focus on addressing it for Euclidean graphs with continuous features.

Figure 3: The exponential growth in the dimension that would result from only considering the ambient feature dimension can be avoided by exploiting the constant intrinsic dimension.

<!-- image -->

## Multiset Injective Functions for Continuous Features

Let us first review some known results on injective multiset functions. Recall that a function defined on multisets with n elements coming from some alphabet Ω ⊆ R D can be identified with a permutation invariant function defined on Ω n . A multiset function is injective if and only if its corresponding function on Ω n is separating with respect to the action of the permutation group (see Definition 1).

In (Corso et al. 2020; Wagstaff et al. 2022) it was shown that for any separating, permutation invariant mappings from R n to R K , the embedding dimension K will be at least n . Two famous examples of continuous functions that achieve this bound are

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

When the multiset elements are in R D , the picture is similar: if there exists a continuous, permutation invariant and separating mapping from R D × n to R K , then necessarily K ≥ n · D (Joshi et al. 2022). In (Dym and Gortler 2023) it is shown that continuous separating invariants for D &gt; 1 , with near-optimal dimension, can be derived from the D = 1 separating invariants Ψ = Ψ pow or Ψ = Ψ sort , by considering random invariants of the form

<!-- formula-not-decoded -->

where j = 1 , . . . , K and each a j and b j are d and n dimensional random vectors, respectively, and we denote θ = ( a 1 , . . . , a K , b 1 , . . . , b K ) ∈ R K ( D + n ) . When K = 2 nD +1 , for almost any choice of θ , the function Embed θ will be separating on R D × n . Thus the embedding dimension in this construction is optimal up to a multiplicative constant of two.

An important property of this results of (Dym and Gortler 2023) for our discussion, is that the embedding dimension K can be reduced if the domain of interest is a non-linear subset of R D × n of low dimension. For example, if the domain of interest is a finite union of lines in R D × n , then the instrinsic dimension of the domain is 1 , and so we will only need an embedding dimension of K = 2 · 1 + 1 = 3 . Thus, the required embedding dimension depends on the intrinsic dimension of the domain rather than on its ambient dimension , which in our case is n · D .

To formulate these results precisely we will need to introduce some real algebraic geometry terminology (see (Basu, Pollack, and Roy 2006a) for more details): A semi-algebraic subset of a real finite-dimensional vector space is a finite union of subsets that are defined by polynomial equality and inequality constraints. For example, polygons, hyperplanes, spheres, and finite unions of these sets, are all semi-algebraic sets. A semi-algebraic set is always a finite union of manifolds, and its dimension is the maximal dimension of the manifolds in this union. Using these notions, we can now state the 'intrinsic version' of the results in (Dym and Gortler 2023):

Theorem 5 (Dym and Gortler (2023)) . Let X be an S n -invariant semi-algebraic subset of R D × n of dimension D X . Denote K = 2 D X +1 . Then for Lebesgue almost every θ ∈ R K ( D + n ) the mapping Embed θ : X → R K is S n invariant and separating.

## Multiset Injective Functions for GNNs

We now return to discuss GNNs and explain the importance of the distinction between the intrinsic and ambient dimensions in our context. Suppose we are given n initial features ( h (0) 1 , . . . , h (0) n ) in R d , and for simplicity let us assume they are recursively refined via the simple aggregation rule:

̸

<!-- formula-not-decoded -->

Let us assume that each Embed ( t ) is injective on the space of all multisets with n -1 elements in the ambient space of h ( t ) j . Then the injectivity of Embed (1) implies that h (1) i is of dimension at least ( n -1) · d . The requirement that Embed (2) is injective on a mult-set of n -1 features in R ( n -1) · d implies that h (2) i will be of dimension at least ( n -1) 2 · d . Continuing recursively with this argument we obtain an estimate of ∼ ( n -1) T d for the dimensions of each h ( T ) i after T iterations of (9).

Fortunately, the analysis presented above is overly pessimistic because it focused only on the ambient dimension . Let us denote the matrix containing all n features at time t by H ( t ) . Then H ( t ) = F t ( H (0) ) , where F t is the concatenation of all Embed ( t ′ ) functions from all previous time-steps. Thus H ( t ) resides in the set F t ( R d × n ) . Here we again rely on results from algebraic geometry: if F t is a composition of piecewise linear and polynomial mappings, then it is a semi-algebraic mapping, which means that F t ( H (0) ) will be a semi-algebraic set of dimension dim( R n × d ) = n · d . This point will be explained in more detail in the proof of Theorem 6. By Theorem 5, we can then use Embed θ as a multiset injective function on X t with a fixed embedding dimension of 2 n · d +1 which does not depend on T . This is visualized in Figure 3.

Table 1: Separation accuracy on challenging 3D point clouds. Hard examples correspond to point clouds which cannot be distinguished by a single 1-EWL iteration but can be distinguished by two iterations, according to Theorem 2. The Harder example is a point cloud not distinguishable by 1-EWL (Pozdnyakov and Ceriotti 2022). GNN implementations and code pipeline based on (Joshi et al. 2022).

| Separation   | complete   | ∼ = 1-EWL   | unknown   | unknown   | unknown      |
|--------------|------------|-------------|-----------|-----------|--------------|
| Point Clouds | 2-SEWLnet  | 1-EWLsim    | MACE      | TFN       | GVPGNN       |
| Hard1        | 100%       | 100%        | 100%      | 100%      | 100%         |
| Hard2        | 100%       | 100%        | 100%      | 100%      | 50%          |
| Hard3        | 100%       | 100%        | 100%      | 100%      | 95.0 ± 15.0% |
| Harder       | 100%       | 50%         | 100%      | 100%      | 53.7 ± 13.1% |

2-SEWLnet Based on the discussion above, we can devise architectures that simulate the various tests discussed in this paper and have reasonable feature dimensions throughout the construction, In particular, we can simulate T iterations of the 2 -SEWL test by replacing all Embed ( t ) functions 1 with Embed ( t ) θ , where in our implementation we choose Ψ = Ψ sort in (8). The embedding dimension for all t is taken to be 6 n +1 , since the input is in R 3 × n . We denote the obtained parametric function by F ϕ . Based on a formalization of the discussion above, we prove in the appendix that F ϕ has the separation power of the complete 2 -SEWL test, and therefore F ϕ is separating.

Theorem 6. Let F ϕ denote the parametric function simulating the 2 -SEWL test. Then for Lebesgue almost every ϕ the function F ϕ : R 3 × n → R 6 n +1 is separating with respect to the action of SO [3 , n ] .

To conclude this subsection, we note that while sortbased permutation invariants are used as aggregators in GNNs (Zhang, Hare, and Pr¨ ugel-Bennett 2020; Zhang et al. 2018; Blondel et al. 2020), the polynomial-based aggregators Ψ pow are not as common. To a certain extent, one can use the approach in (Xu et al. 2019; Maron et al. 2019), replace the polynomials in Ψ pow by MLPs, and justify this by the universal approximation power of MLPs. A limitation of this approach is that it only guarantees separation at the limit.

## Synthetic Experiments

In this section, we implement 2-SEWLnet and empirically evaluate its separation power, and the separation power of alternative SO [3 , n ] invariant point cloud architectures. We trained the architectures on permuted and rotated variations of highly-challenging point-cloud pairs, and measured separation by the test classification accuracy. We considered three pairs of point clouds (Hard1-Hard3) from Pozdnyakov et al. (2020). These pairs were designed to be challenging for distance-based invariant methods. However, our analysis reveals that they are in fact separable by two iterations of the 1-EWL test. We then consider a pair of point clouds from (Pozdnyakov and Ceriotti 2022) which was proven to be indstinguishable by the 1-EWL tests. The results of this experiment are given in Table 1. Further details on the experimental setup appear in Appendix B.

1 A minor technicality is that the Embed functions are actually defined on vector-multiset pairs. This issue is discussed in the proof of the theorem.

As expected, we find that 2 -SEWLnet, which has complete separation power, succeeded in perfectly separating all examples. We also found that the simulation of 1-EWL does not separate the Harder example, but does separate the Hard example after two iterations, as predicted by Theorem 2. We also considered three additional invariant point cloud models whose separation power is not as well understood. We find that MACE (Batatia et al. 2022) and TFN (Thomas et al. 2018) achieve perfect separation, (when applying them with at least 3-order correlations and threeorder SO (3) representations). The third GVPGNN (Jing et al. 2021) architecture attains mixed results. We note that we cannot necessarily deduce from our empirical results that MACE and TFN are complete. While it is true that TFN is complete when considering arbitrarily high order representations (Dym and Maron 2021), it is not clear whether order three representation suffices for complete separation. We conjecture that this is not the case. However, finding counterexamples is a challenging problem we leave for future work.

## Future Work

In this work, we presented several invariant tests for point clouds that are provably complete, and have presented and implemented 2 -SEWL-net which simulates the complete 2 -SEWL test. Currently, this is a basic implementation that only serves to corroborate our theoretical results. A practically useful implementation requires addressing several challenges, including dealing with point clouds of different sizes, the non-trivial ∼ n 4 complexity of computing even the relatively efficient 2 -SEWL-net, and finding learning tasks where complete separation leads to gains in performance. We are actively researching these directions and hope this paper will inspire others to do the same.

## Acknowledgements

This research was supported by the Israel Science Foundation grant no. 272/23.

## References

Aamand, A.; Chen, J.; Indyk, P.; Narayanan, S.; Rubinfeld, R.; Schiefer, N.; Silwal, S.; and Wagner, T. 2022. Exponentially Improving the Complexity of Simulating the Weisfeiler-Lehman Test with Graph Neural Networks. Advances in Neural Information Processing Systems , 35: 27333-27346.

Arvind, V.; and Rattan, G. 2016. The parameterized complexity of geometric graph isomorphism. Algorithmica , 75: 258-276.

Babai, L.; Erd˝ os, P.; and Selkow, S. M. 1980. Random Graph Isomorphism. SIAM Journal on Computing , 9(3): 628-635.

Bachman, G.; and Narici, L. 2000. Functional Analysis . Academic Press textbooks in mathematics. Dover Publica-

tions. ISBN 9780486402512.

Balan, R.; Haghani, N.; and Singh, M. 2022. Permutation Invariant Representations with Applications to Graph Deep Learning. arXiv preprint arXiv:2203.07546 .

Basu, S.; Pollack, R.; and Roy, M.-F. 2006a. Algorithms in real algebraic geometry , volume 10. Springer.

Basu, S.; Pollack, R.; and Roy, M.-F. 2006b. Algorithms in Real Algebraic Geometry (Algorithms and Computation in Mathematics) . Berlin, Heidelberg: Springer-Verlag. ISBN 3540330984.

Batatia, I.; Kovacs, D. P.; Simm, G.; Ortner, C.; and Csanyi, G. 2022. MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields. In Koyejo, S.; Mohamed, S.; Agarwal, A.; Belgrave, D.; Cho, K.; and Oh, A., eds., Advances in Neural Information Processing Systems , volume 35, 11423-11436. Curran Associates, Inc.

Blondel, M.; Teboul, O.; Berthet, Q.; and Djolonga, J. 2020. Fast differentiable sorting and ranking. In International Conference on Machine Learning , 950-959. PMLR.

B¨ okman, G.; Kahl, F.; and Flinth, A. 2022. Zz-net: A universal rotation equivariant architecture for 2d point clouds. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 10976-10985.

Brass, P.; and Knauer, C. 2000. Testing the congruence of d-dimensional point sets. In Proceedings of the sixteenth annual symposium on Computational geometry , 310-314.

Cai, J.-Y.; F¨ urer, M.; and Immerman, N. 1992. An Optimal Lower Bound on the Number of Variables for Graph Identification. Combinatorica , 12(4): 389-410.

Chen, Z.; Villar, S.; Chen, L.; and Bruna, J. 2019. On the equivalence between graph isomorphism testing and function approximation with gnns. Advances in neural information processing systems , 32.

Corso, G.; Cavalleri, L.; Beaini, D.; Li` o, P.; and Veliˇ ckovi´ c, P. 2020. Principal neighbourhood aggregation for graph nets. Advances in Neural Information Processing Systems , 33: 13260-13271.

Cybenko, G. 1989. Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals, and Systems , 2: 303-314.

Dym, N.; and Gortler, S. J. 2023. Low Dimensional Invariant Embeddings for Universal Geometric Learning. arXiv:2205.02956.

Dym, N.; and Kovalsky, S. Z. 2019. Linearly converging quasi branch and bound algorithms for global rigid registration. In Proceedings of the IEEE/CVF International Conference on Computer Vision , 1628-1636.

Dym, N.; and Maron, H. 2021. On the Universality of Rotation Equivariant Point Cloud Networks. In International Conference on Learning Representations . Publisher Copyright: © 2021 ICLR 2021 - 9th International Conference on Learning Representations. All rights reserved.; 9th International Conference on Learning Representations, ICLR 2021 ; Conference date: 03-05-2021 Through 07-05-2021.

Finkelshtein, B.; Baskin, C.; Maron, H.; and Dym, N. 2022. A Simple and Universal Rotation Equivariant Point-Cloud Network. In Cloninger, A.; Doster, T.; Emerson, T.; Kaul, M.; Ktena, I.; Kvinge, H.; Miolane, N.; Rice, B.; Tymochko, S.; and Wolf, G., eds., Topological, Algebraic and Geometric Learning Workshops 2022, 25-22 July 2022, Virtual , volume 196 of Proceedings of Machine Learning Research , 107-115. PMLR.

Finzi, M.; Welling, M.; and Wilson, A. G. 2021. A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups. In Meila, M.; and Zhang, T., eds., Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event , volume 139 of Proceedings of Machine Learning Research , 3318-3328. PMLR.

Gama, F.; Bruna, J.; and Ribeiro, A. 2020. Stability Properties of Graph Neural Networks. IEEE Transactions on Signal Processing , 68: 5680-5695.

Gasteiger, J.; Becker, F.; and G¨ unnemann, S. 2021. GemNet: Universal Directional Graph Neural Networks for Molecules. In Ranzato, M.; Beygelzimer, A.; Dauphin, Y.; Liang, P.; and Vaughan, J. W., eds., Advances in Neural Information Processing Systems , volume 34, 6790-6802. Curran Associates, Inc.

Gilmer, J.; Schoenholz, S. S.; Riley, P. F.; Vinyals, O.; and Dahl, G. E. 2017. Neural Message Passing for Quantum Chemistry. CoRR .

Grohe, M. 2017. Descriptive complexity, canonisation, and definable graph structure theory , volume 47. Cambridge University Press.

Jing, B.; Eismann, S.; Suriana, P.; Townshend, R. J. L.; and Dror, R. 2021. Learning from Protein Structure with Geometric Vector Perceptrons. In International Conference on Learning Representations .

Joshi, C. K.; Bodnar, C.; Mathis, S. V .; Cohen, T.; and Li` o, P. 2022. On the Expressive Power of Geometric Graph Neural Networks. NeurIPS Workshop on Symmetry and Geometry in Neural Representations .

Jost, J. 2017. Riemannian Geometry and Geometric Analysis . Universitext. Springer International Publishing. ISBN 9783319618609.

Kapon, Y.; Saha, A.; Duanis-Assaf, T.; Stuyver, T.; Ziv, A.; Metzger, T.; Yochelis, S.; Shaik, S.; Naaman, R.; Reches, M.; et al. 2021. Evidence for new enantiospecific interaction force in chiral biomolecules. Chem , 7(10): 2787-2799.

Kingma, D. P.; and Ba, J. 2015. Adam: A Method for Stochastic Optimization. In Bengio, Y.; and LeCun, Y., eds., 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings .

Kraft, H.; and Procesi, C. 1996. Classical invariant theory, a primer. Lecture Notes. Preliminary version .

Kurlin, V. 2024. Polynomial-time algorithms for continuous metrics on atomic clouds of unordered points. MATCH Communications in Mathematical and in Computer Chemistry , 91: 79-108.

Lim, D.; Robinson, J.; Zhao, L.; Smidt, T.; Sra, S.; Maron, H.; and Jegelka, S. 2022. Sign and Basis Invariant Networks for Spectral Graph Representation Learning. In ICLR 2022 Workshop on Geometrical and Topological Representation Learning . ICLR Workshop on Geometrical and Topological Representation Learning ; Conference date: 29-04-2022 Through 29-04-2022.

Ma, X.; Zhou, Y.; Wang, H.; Qin, C.; Sun, B.; Liu, C.; and Fu, Y. 2023. Image as Set of Points. In The Eleventh International Conference on Learning Representations, ICLR 2023 .

Maron, H.; Ben-Hamu, H.; Serviansky, H.; and Lipman, Y. 2019. Provably Powerful Graph Networks. In Wallach, H.; Larochelle, H.; Beygelzimer, A.; d'Alch´ e-Buc, F.; Fox, E.; and Garnett, R., eds., Advances in Neural Information Processing Systems , volume 32. Curran Associates, Inc.

Mityagin, B. S. 2020. The Zero Set of a Real Analytic Function. Mathematical Notes , 107: 529-530.

Morris, C.; Ritzert, M.; Fey, M.; Hamilton, W. L.; Lenssen, J. E.; Rattan, G.; and Grohe, M. 2019. Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 February 1, 2019 , 4602-4609. AAAI Press.

Munkres, J. 2000. Topology . Featured Titles for Topology. Prentice Hall, Incorporated. ISBN 9780131816299.

Paszke, A.; Gross, S.; Massa, F.; Lerer, A.; Bradbury, J.; Chanan, G.; Killeen, T.; Lin, Z.; Gimelshein, N.; Antiga, L.; Desmaison, A.; Kopf, A.; Yang, E.; DeVito, Z.; Raison, M.; Tejani, A.; Chilamkurthy, S.; Steiner, B.; Fang, L.; Bai, J.; and Chintala, S. 2019. PyTorch: An Imperative Style, HighPerformance Deep Learning Library. In Advances in Neural Information Processing Systems 32 , 8024-8035. Curran Associates, Inc.

Pozdnyakov, S. N.; and Ceriotti, M. 2022. Incompleteness of graph neural networks for points clouds in three dimensions. Machine Learning: Science and Technology , 3(4): 045020.

Pozdnyakov, S. N.; Willatt, M. J.; Bart´ ok, A. P.; Ortner, C.; Cs´ anyi, G.; and Ceriotti, M. 2020. Incompleteness of atomic structure representations. Physical Review Letters , 125(16): 166001.

Puny, O.; Atzmon, M.; Smith, E. J.; Misra, I.; Grover, A.; Ben-Hamu, H.; and Lipman, Y. 2021. Frame Averaging for Invariant and Equivariant Network Design. In International Conference on Learning Representations .

Satorras, V. G.; Hoogeboom, E.; and Welling, M. 2021. E (n) equivariant graph neural networks. In International conference on machine learning , 9323-9332. PMLR.

Sch¨ utt, K.; Kindermans, P.-J.; Sauceda Felix, H. E.; Chmiela, S.; Tkatchenko, A.; and M¨ uller, K.-R. 2017. SchNet: A continuous-filter convolutional neural network for modeling quantum interactions. In Guyon, I.; Luxburg, U. V.; Bengio, S.; Wallach, H.; Fergus, R.; Vishwanathan, S.; and Garnett, R., eds., Advances in Neural Information Processing Systems , volume 30. Curran Associates, Inc.

Thomas, N.; Smidt, T.; Kearnes, S.; Yang, L.; Li, L.; Kohlhoff, K.; and Riley, P. 2018. Tensor field networks: Rotation-and translation-equivariant neural networks for 3d point clouds. arXiv preprint arXiv:1802.08219 .

Villar, S.; Hogg, D. W.; Storey-Fisher, K.; Yao, W.; and Blum-Smith, B. 2021. Scalars are universal: Equivariant machine learning, structured like classical physics. In Ranzato, M.; Beygelzimer, A.; Dauphin, Y.; Liang, P.; and Vaughan, J. W., eds., Advances in Neural Information Processing Systems , volume 34, 28848-28863. Curran Associates, Inc.

Wagstaff, E.; Fuchs, F. B.; Engelcke, M.; Osborne, M. A.; and Posner, I. 2022. Universal approximation of functions on sets. Journal of Machine Learning Research , 23(151): 1-56.

Wang, L.; Liu, Y.; Lin, Y.; Liu, H.; and Ji, S. 2022. ComENet: Towards Complete and Efficient Message Passing for 3D Molecular Graphs. In Koyejo, S.; Mohamed, S.; Agarwal, A.; Belgrave, D.; Cho, K.; and Oh, A., eds., Advances in Neural Information Processing Systems , volume 35, 650-664. Curran Associates, Inc.

Weisfeiler, B.; and Leman, A. A. 1968. The reduction of a graph to canonical form and the algebra which appears therein. Nauchno-Technicheskaya Informatsia , 2: 12-16.

Widdowson, D.; and Kurlin, V. 2022. Resolving the data ambiguity for periodic crystals. In Advances in Neural Information Processing Systems .

Widdowson, D.; and Kurlin, V. 2023. Recognizing Rigid Patterns of Unlabeled Point Clouds by Complete and Continuous Isometry Invariants with no False Negatives and no False Positives. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023 , 1275-1284. IEEE.

Xu, K.; Hu, W.; Leskovec, J.; and Jegelka, S. 2019. How Powerful are Graph Neural Networks? In 7th International Conference on Learning Representations, ICLR 2019 .

Zhang, M.; Cui, Z.; Neumann, M.; and Chen, Y. 2018. An end-to-end deep learning architecture for graph classification. In Proceedings of the AAAI conference on artificial intelligence , volume 32.

Zhang, Y.; Hare, J. S.; and Pr¨ ugel-Bennett, A. 2020. FSPool: Learning Set Representations with Featurewise Sort Pooling. In 8th International Conference on Learning Representations, ICLR 2020 .

## Appendix A : Proofs

We begin by stating and proving a result mentioned in the main text: once we construct an invariant separator, we can obtain a universal model by composing the separation with a standard fully connected neural network:

Theorem 7 (Separation Implies Universality) . Let f : R d × n → R be a G -invariant continuous function. If F : R d × n → R M is an invariant separator, then for any compact set K ⊂ R d × n , and any ϵ &gt; 0 , there exists a neural network N ϵ : R M → R such that sup x ∈ K | f ( x ) -N ϵ ◦ F ( x ) | &lt; ϵ .

Proof. Let ϵ &gt; 0 and K ⊆ R d × n be a compact set. Using Proposition 1.3 in (Dym and Gortler 2023), there exists a continuous f ϵ such that

<!-- formula-not-decoded -->

The image of a compact set under a continuous function is a compact set, see (Munkres 2000), then S := Im ( F ) is compact. By the Universal Approximation Theorem (Cybenko 1989), we can approximate f ϵ with a fully-connected Neural Network with arbitrary precision, i.e. there exists a Neural Network Function N ϵ such that for all x ∈ S ,

<!-- formula-not-decoded -->

By the Triangle Inequality, for all x ∈ K ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 2. Two iterations of the 1 -EWL test assign two point clouds X , Y ∈ R 3 × n distinct the same value, if and only if X = O [3 ,n ] Y .

Proof. Assume we initialize the hidden states with null information. After a single iteration, we have

̸

<!-- formula-not-decoded -->

By assumption, all h (1) i , i = 1, . . . , n, are distinct. Thus at the next iteration,

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

we know each node's ordered distances from the other nodes, as the i -th node is uniquely (intra-point-cloud) determined by h (1) i . Thus we can recover the distance matrix

<!-- formula-not-decoded -->

Thus, we can fully recover the point cloud up to Euclidean motion, see Section E in (Satorras, Hoogeboom, and Welling 2021). In conclusion, if X,Y ∈ R d × n are assigned the same value by 1-EWL, then they are identical up to permutation and Euclidean motion, i.e. an O [3 , n ] transformation.

By Theorem 2, 1 -EWL is complete on R 3 × n distinct . We now show that 1 -EWL is incomplete at most on a (non-trivial) measure-zero set, thus by definition, it is complete almost everywhere on the space of point clouds endowed with permutation, rotation, and reflection symmetries.

Theorem 8 (1-EWL Separates Almost All Complete Euclidean Graphs) . Let µ be the Lebesgue measure on R 3 × n where n ≥ 3 . Then, µ ( R 3 × n \ R 3 × n distinct ) = 0 .

̸

Proof. We defined R 3 × n distinct = { X ∈ R 3 × n | d ( i, X ) = d ( j, X ) ∀ i = j } .

Then

<!-- formula-not-decoded -->

̸

where cond i,j := ∥ ψ pow ( d ( j, X ) ) -ψ pow ( d ( i, X ) ) ∥ 2 = 0 and ψ pow is the power-sum polynomials defined as ψ pow ( ⃗ x ) = ( ∑ n i =1 x i , . . . , ∑ n i =1 x n i ) , which is known to be injective on multisets with n elements.

Equation 13 defines an algebraic manifold with a nontrivial polynomial equality constraint, thus is of dimension ≤ 3 n -1 . If an algebraic manifold embedded in R 3 × n has dimension ≤ 3 n -1 , then it has measure zero (Mityagin 2020).

Theorem 4. For every X,Y ∈ R 3 × n , a single iteration of the 2 -SEWL test assigns X and Y the same value if and only if X = SO [3 ,n ] Y .

Proof. Let X, Y ∈ R 3 × n . Recall that a single iteration of the 2 -SEWL assigns to each index pair i, j an initial coloring C (0) [ i, j ] = C (0) [ i, j ]( X ) corresponding to the 2 × 2 Gram matrix of the points x i , x j . The coloring is then refined via

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

̸

and then a final global coloring is obtained from

<!-- formula-not-decoded -->

Let us denote the global feature C G obtained from X by C G ( X ) , and the global feature obtained from Y by C G ( Y ) .

By construction, if X = SO [3 ,n ] Y then C G ( X ) = C G ( Y ) .

<!-- formula-not-decoded -->

To make the proof more readable, we introduce the following notation (we will later describe its significance):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now show that the multiset C G ( X ) = { { C ( 1 ) ( i , j ) | i, j ∈ [ n ] } } allows recovering the multiset h X := { { m [ i,j ] | i, j ∈ [ n ] , i, j ∈ [ n ] } } .

It is enough to show that we can recover m [ i,j ] from its corresponding C (1) [ i, j ] for every i, j ∈ [ n ] and then the multiset equivalence follows immediately.Note that G [ i,j ] is the 3 × 3 Gram matrix of the vectors x i , x j , x i × x j . It can be recovered from C ( 0 ) ( i , j ) , which is the 2 × 2 Gram matrix of x i , x j , because ⟨ x i × x j , x k ⟩ = ⟨ x i × x j , x i ⟩ = 0 and

<!-- formula-not-decoded -->

where θ is the angle between x i and x j . The quantity on the RHS of the equation can be extracted from C ( 0 ) ( i , j ) .

As for h i,j , we can recover it as a multiset since:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We saw that h X = Embed { { m [ i,j ] ( X ) | i, j ∈ [ n ] } } can be recovered from C G ( X ) and thus in particular our assumption that C G ( X ) = C G ( Y ) implies that h X = h Y . We will now use this to show that X = SO [3 ,n ] Y .

We first deal with the degenerate case where all points in X are identical, that is x 1 = x 2 = . . . = x n . In this case, all Gram matrices G i,j ( X ) , and all entries in each of the matrices, will be identical, and thus by assumption also all Gram matrices G i,j ( Y ) , and all their entries will be identical. This implies that Y also consists of a single point with the same norm as the one point in X , and therefore X = SO [3 ,n ] Y .

We can now assume that not all points in X are the same. Define r ( X ) = rank(X) = max i,j ∈ [ n ] rank( G [ i,j ] ( X )) (note that we assume that n ≥ 3 ). By assumption we have

̸

{ { G [ i,j ] ( X ) | i, j ∈ [ n ] } } = { { G [ i,j ] ( Y ) | i, j ∈ [ n ] } } , thus r ( X ) = r ( Y ) and there exist i = j and s, t ∈ [ n ] such that G [ i,j ] ( X ) = G [ s,t ] ( Y ) , ( ⋆ ) they both have rank r , and x i = x j . Due to the fact that they both have rank r, and x i = x j , it follows that y s = y t and in particular s = t .

̸

̸

By (Kraft and Procesi 1996), the equality of Gram matrices implies that there exists an orthogonal transformation, T ∈ O (3) , such that

<!-- formula-not-decoded -->

̸

If x i × x s = 0 we see that T preserves orientation and therefore T ∈ SO (3) (see Remark .9). If not, and if T is a reflection, we can modify T to be a rotation that still satisfies (19) by composing it with a reflection that fixes the ≤ 1 dimensional subspace spanned by x i , x j . Thus in any case we can assume that T ∈ SO (3) . By assumption and ( ⋆ ), we have { { P [ i,j,k ] ( X ) k = i, j } } = { { P [ s,t,k ] ( Y ) , k = s, t } } .

̸

̸

̸

This implies that there exists some permutation σ ∈ S n such that σ ( i ) = s.σ ( j ) = t and P [ i,j,k ] ( X ) = P [ s,t,σ ( k )] ( Y ) for all k = i, j . We deduce that for all k = i, j

̸

<!-- formula-not-decoded -->

and similarly

<!-- formula-not-decoded -->

Now note that each y k is in the span of y s , y t , y s × y t (even when r &lt; 3 ), and similarly every x k is in the span of x i , x j , x i × x j and so Tx k is also in the span of y s , y t , y s × y t . It follows that Tx k -y σ ( k ) = 0 , and thus we showed that X and Y are related by a SO [3 , n ] transformation.

Remark .9 . In the proof above we said that if (19) holds, and y s × y t is not zero, then T is in SO (3) . This follows from the fact that for general orthogonal transformations and vectors a, b

<!-- formula-not-decoded -->

Setting a = x i , b = x j and using (19) we obtain that

<!-- formula-not-decoded -->

and so if y s × y t is not zero, then det( T ) = 1 .

Theorem 6. Let F ϕ denote the parametric function simulating the 2 -SEWL test. Then for Lebesgue almost every ϕ the function F ϕ : R 3 × n → R 6 n +1 is separating with respect to the action of SO [3 , n ] .

Proof. We recall that F ϕ is defined to simulate a single iteration of the 2 -SEWL test using sort-based injective multiset functions. In more detail, recall that the initial coloring corresponding to an index pair ( i, j ) and a point cloud X ∈ R 3 × n is given by the Gram matrix of x i , x j , and denoted by C (0) ( i, j ) = C (0) ( i, j )( X ) . We then define

<!-- formula-not-decoded -->

̸

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Embed θ ( y 1 , . . . , y n ) is permutation invariant (=multiset function), continuous in θ and y i , and defined by

<!-- formula-not-decoded -->

with j = 1 , . . . , 6 n + 1 and Ψ = sort (or alternatively, Ψ could be the power sum polynomials), and θ denoting the concatenation of all the mapping parameters a i and b j .

We denote by ϕ = ( α, β ) the concatenation of the twoparameter vectors of the Embed mappings in the constructions, and F ϕ ( X ) denoted the output C G = C G ( X ; ϕ ) obtained by this construction.

Since we already showed that the 2-SEWL test is complete, it is sufficient to show that for Lebesgue almost every ( α, β ) , the mapping Embed α is permutation invariant and separating on R 3 × n , and the mapping Embed β is permutation invariant and separating on the image of the mapping f α which we define as

<!-- formula-not-decoded -->

By Theorem 5 we know that Embed α is separating for Lebesgue almost every α . For fixed α , we know that f α is a semi-algebraic mapping, since it is a composition of polynomials and the piecewise linear sort function, which are semialgebraic mappings, and as compositions of semi-algebraic mappings are semi-algebraic mappings. The dimension of the image of a semi-algebraic mapping is never larger than the dimension of the domain, and so f α ( R 3 × n ) is a semialgebraic set of dimension ≤ dim( R 3 × n ) = 3 n (see (Basu, Pollack, and Roy 2006b) for the necessary real algebraic geometry statements regarding composition and dimension). To apply Theorem 4 we need to work with a permutation invariant domain, so we artificially enlarge the domain of Embed β to be

<!-- formula-not-decoded -->

which is a finite union of sets of dimension ≤ 3 n and hence also has dimension ≤ 3 n . It follows that for almost every β the function Embed β is separating on this permutation invariant set, with embedding dimension of 6 n + 1 as we defined in (20). Using Fubini's theorem, this implies that for almost every ( α, β ) the functions Embed α and Embed β are both separating, and this proves the theorem.

Complexity We conclude by discussing the complexity of computing F ϕ . Calculating each C (1) ( i, j ) using sort-based embeddings Embed α requires O ( n 2 log( n )) operations.

Since there are O ( n 2 ) such C (1) ( i, j ) the total complexity of computing all of them is O ( n 4 log( n )) . In the second step we compute Embed β on multisets of size D × N where D = O ( n ) , N = O ( n 2 ) , and with embedding dimension of O ( n ) . This requires O ( n 4 + n 3 log( n )) operations , so the total complexity is O ( n 4 log( n ))

In Appendix we extend our results to arbitrary d . In this case, we get a complexity of O ( n d +1 log( n )) (where for simplicity we consider the limit n → ∞ with d fi xed, to cancel out some mixed terms in n, d which are negligible in this limit. ).

## B : Experiment Details

As mentioned, we exemplified the viability of the theory presented by testing separation on challenging point cloud pairs. We wished to address the following scenario: given a pair of point clouds, each labeled distinctly, what would be the accuracy score of SO [3 , n ] (or O [3 , n ] ) invariant architectures in this classification task following training on a labeled dataset of these examples? This setup partly informs us of the separation capability of these architectures, i.e. how well do these models distinguish similar ( for instance, 1-EWL equivalent ), yet non-isomorphic, input?

For implementation, we used code by (Joshi et al. 2022) that implements several contemporary SO [3 , n ] invariant architectures and evaluated them as described below. This framework has several additional tests for geometric graphs, but they were irrelevant to our setting because they are redundant for the fully-connected geometric graphs we focus on. We modified the implementation of (Joshi et al. 2022) by implementing our novel invariant architectures, 2-SEWLnet, implementing 1-EWLsim, and testing counterexample point cloud pairs from (Pozdnyakov et al. 2020; Pozdnyakov and Ceriotti 2022). We used implementation by (Joshi et al. 2022) of MACE (Batatia et al. 2022), TFN (Thomas et al. 2018) and GVPGNN (Jing et al. 2021). The SO [3 , n ] invariant architectures are trained on replicas of each counterexample pair and then testing is performed on the same pair.

## Technical Details

We trained the various invariant models on an NVIDIA A40 GPU implemented in PyTorch (Paszke et al. 2019). The hyperparameters were a learning rate of 0.0001 with Adam optimizer (Kingma and Ba 2015), with the learning rate scheduler ReduceOnPLateau that reduces the learning rate once the loss stopped diminishing. We trained each model on a dataset of 50 copies of each pair for 100 epochs, while injecting permutation and rotation to each point cloud during training. The test and validation datasets are each a pair of plain (no permutation or rotation injected) point clouds. Thus each epoch has ternary accuracy results, of 0%, 50%, and 100%. We then average the accuracy of the last 20% of epochs to obtain the overall accuracy. This was done to allow the model to converge while allowing for a sufficiently large test measurements to obtain statistical significance.

## 2-SEWLnet

2-SEWLnet is an implementation of a simulation of a single iteration of 2-SEWL. We implemented the architecture by directly embedding the multiset function F ϕ , using the lowdimensional invariant embeddings from the Section Multiset injective functions. We chose the sort function as the one-dimensional permutation separating invariant, as it constitutes an isometry from R n /S n to R n , then composing it with linear mappings, yields a Bi-Lipschitz mapping (Balan, Haghani, and Singh 2022; Dym and Gortler 2023). We used differentiable sorting from PyTorch (Paszke et al. 2019) to enable backpropagation. We note that the model is able to learn the classification task with backpropagation only using the sort vector-wise activation, i.e. without a fullyconnected neural network composed on it. This is not a trivial result that is immediately implied by the completeness of 2 -SEWL, as we aim to minimize the softmax cross-entropy loss and in practice reach almost zero loss, thus not only yielding two distinct embeddings corresponding to each distinct point cloud ( as guaranteed by Theorem 4 ) but learning them to be (approximately) one-hot encoded vectors using our injective continuous Embed functions.

Table 2: GNN implementations and code pipeline based on (Joshi et al. 2022).

| Hyperparameters   | 2-SEWLnet     | 1-EWLsim   |   MACE |    TFN | GVPGNN   |
|-------------------|---------------|------------|--------|--------|----------|
| Learning rate     | 0.0001        | 0.0001     | 0.0001 | 0.0001 | 0.0001   |
| Hidden Dimension  | 2 (Pair-wise) | 1          |      2 |     64 | 64       |
| Number of Layers  | 1             | 2          |      3 |      3 | 3        |
| Batch size        | 1             | 1          |      1 |      1 | 1        |
| Correlation       | NA            | NA         |      3 |      3 | NA       |

## 1-EWLsim

Interestingly, we found that using the sum aggregation, the model did not yield sufficient separation for the classification task for the examples Hard1-3 in Table 1, yet when using sort and non-linear point-wise activations, rather than sums of neural network functions applied point-wise, led to perfect classification on these counterexamples. The results in the table are reported with the latter implementation. In the code, we denoted this implementation as 'egnn', as this simulation is inspired by the invariant version of EGNN (Satorras, Hoogeboom, and Welling 2021).

## C : Background Theory Low Dimensional Separating Invariants

In the main text, we presented a condensed summary of the results from (Dym and Gortler 2023) as they pertain to this paper's scope. We devote this appendix to expound on the context of these results. In Subsection Multiset injective functions, we discussed the Power-Sum Symmetric Polynomials . These polynomials yield a separating invariant with respect to permutations of real-valued vectors in R n . The invariant learning literature often discussed an extension of this characterization for vector-valued features, the MultiPower-Sum Symmetric Polynomials , defined, for an input X = ( x 1 . . . x n ) ∈ R d × n and α = ( α 1 , . . . , α d ) ∈ N d ≥ 0 , as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where x α := x α 1 1 . . . x α d d and | α | := ∑ n i =1 α i . These polynomials define a separating invariant for point clouds in R d with respect to the permutation of the (n) columns (Maron et al. 2019). Yet, its embedding dimension is ( n + d d ) . The goal of (Dym and Gortler 2023) was to reduce the embedding dimension to a complexity linear in the n · d dimension of the input.

As a first example of such a result, (Dym and Gortler 2023) show the for Lebesgue almost every ( n + d d ) dimensional vectors w 1 , . . . , w 2 nd +1

<!-- formula-not-decoded -->

where i = 1 , . . . , 2 dim ( M ) + 1 . Thus we obtain a separating invariant of dimension O ( d · n ) rather than O ( n 2 d ) . Yet, we still had to calculate all of the O ( n 2 d ) polynomial entries. Therefore, computationally speaking, this approach did not yield much.

To remedy this, (Dym and Gortler 2023) proposed alternative invariants based on R n permutation invariants Ψ . Such a Ψ . and any choice of a vector a ∈ R d , induces a permutation invariant on R d × n of the form

<!-- formula-not-decoded -->

This technique of producing high dimensional invariants from low dimensional ones is known as polarization . To obtain invariants that are also separating , we need (i) to choose Ψ to be invariant and separating on R n . Two examples of such functions, are the functions Ψ sort and Ψ pow which we defined in (6):

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Additionally, we need (ii) to choose not a single polarization function defined by a single a , but rather 2 nd + 1 random vectors a 1 , . . . , a 2 nd +1 . More precisely, (Dym and Gortler 2023) showed that for Lebesgue almost every a 1 , . . . , a 2 nd +1 ∈ R d and b 1 , . . . , b 2 nd +1 ∈ R n the function

<!-- formula-not-decoded -->

where , j = 1 , . . . , 2 nd + 1 , defined in (8) is permutation invariant and separating. The role of the projection by the b j is to reduce the embedding dimension to 2 nd +1 rather than the (2 nd +1) n dimension we would get if these projections were not applied.

We note that the complexity of a single invariant in (24) would be O ( n log( n )) (assuming n &gt; d ) when using sorting or O ( n 2 ) when using power sum polynomials. Accordingly, the complexity of computing 2 nd + 1 invariants would be O ( dn 2 log( n )) using sorting or O ( dn 3 ) using power sum polynomials.

Finally, we note that (as mentioned in the main text), if we're interested in separation only on a semi-algebraic permutation invariant subset X ⊆ R d × n with dimension D X , then the number of separating invariants needed in (24) would be 2 D X +1 rather than 2 nd +1 .

## Extensions

In the main text, we described Vanilla k -WL tests which are well-defined for all k and d , and the 2 -SEWL test which is well-defined for the case d = 3 (since vector products are used). We now explain how to define a ( d -1) SEWL test for general d , and then explain how these tests can be easily modified to give a ( d -1) -EWL test with similar complexity, which is O [3 , n ] invariant and separating rather than SO [3 , n ] invariant and separating.

## SEWL for General Dimension d

The complete 2-SEWL test for d = 3 point clouds can be generalized to a ( d -1) -SEWL test for general d -dimensional point clouds by a generalization of the crossproduct operator. This generalization is formally known as the Hodge dual operator (Jost 2017).

For fixed x i 1 , . . . , x i d -i ∈ R d , we define the following linear functional

<!-- formula-not-decoded -->

By Riesz's Representation Theorem (Bachman and Narici 2000), every linear functional on R d is essentially a function in the form of a dot product against some (unique) vector in R d . Thus, there exists some vector, denoted by x ⋆ = x ⋆ ( x i 1 , . . . , x i d -1 ) ∈ R d , such that

<!-- formula-not-decoded -->

We see directly from the definition that x ⋆ is orthogonal to x i 1 , . . . , x i d -1 and is non-zero if and only if these d -1 vectors are linearly independent. Moreover, this choice of vector is SO ( d ) equivariant, which means that for any x i 1 , . . . , x i d -i ∈ R d and R ∈ SO ( d ) we have

<!-- formula-not-decoded -->

This is because for any x ∈ R we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, we note that the coordinates of x ∗ can be calculated by inserting the unit vectors e 1 , . . . , e d into (25). That is

<!-- formula-not-decoded -->

where ( x ⋆ ) j is the j -th entry of x ⋆ . This requires computing d different determinants of d × d matrices, and so the total complexity of computing x ∗ is d 4 .

The ( d -1) -SEWL test We have shown an extension of the definition of the cross-product that respects orientation, thus now we can naturally define a ( d -1) -SEWL test which will be SO [ d, n ] invariant and separating in d dimensions (generalizing the 2 -SEWL test for d = 3 ).

We define for each ( d -1) -tuple i ∈ [ n ] d -1 an initial coloring C (0) ( i ) = C (0) ( i )( X ) corresponding to the ( d -1) × ( d -1) Gram matrix of the points x i 1 , . . . , x i d -1 . We denote x ∗ ( x i 1 , . . . , x i d -1 ) by x ∗ ( i ) . The coloring is then refined via

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where i [ j \ t ] is the multi-index i with its t -th coordinate replaced by j ; e.g. for t = 1 , i [ j \ 1] = ( j, i 2 , . . . , i k ) . Then a final global coloring is obtained from

<!-- formula-not-decoded -->

The ( d -1) -SEWL test can be shown to be SO [ d, n ] complete, using the same arguments used in the proof of Theorem 4.

## ( d -1) -EWL for general dimension d

We now return to the case where reflections are also considered symmetries, and we're looking for complete tests with respect to the group O [ d, n ] . The Vanillad -WLtest will be O [ d, n ] complete. However, a more efficient test can be obtained by tweaking the ( d -1) -SEWL test which is not reflection-invariant, to attain a reflection invariant O [ d, n ] complete test.

This tweaking is obtained as follows. We fix some reflection R 0 (a reflection is an orthogonal matrix with a negative determinant). We define the ( d -1) -EWL test for a given X ∈ R d × n by applying the ( d -1) -SEWL test to both X and R 0 X to obtain C G ( X ) and C G ( R 0 X ) , and then computing a final global feature via

<!-- formula-not-decoded -->

In the following theorem, we show how the completeness of the ( d -1) -SEWL test implies the completeness of the ( d -1) -EWL test.

Theorem 10. For every X,Y ∈ R d × n , a single iteration of the ( d -1) -EWL test assigns X and Y the same value if and only if X = O [ d,n ] Y .

Proof. Invariance: We prove that for every R ∈ O ( d ) and permutation matrix P ∈ S n we have that C ref G ( RXP ) = C ref G ( X ) . We can divide into two cases: If R ∈ SO ( d ) then by the SO [ d, n ] invariance of C G ( X ) we have that

<!-- formula-not-decoded -->

On the other hand, if R is a reflection, then

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and so in both cases, we obtain that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Completeness: We prove that if X,Y ∈ R d × n and C ref G ( X ) = C ref G ( Y ) then X and Y are related by a permutation and orthogonal transformation.

Since C ref G ( X ) = C ref G ( Y ) it follows that either C G ( X ) = C G ( Y ) or C G ( X ) = C G ( R 0 Y ) . The completeness of the ( d -1) -SEWL test (Theorem 4) then implies that X is related to either Y or R 0 Y by an SO [ d, n ] transformation. In either case, this implies that X and Y are related by an O [ d, n ] transformation.

## Continuous Implementation and Computational Complexity

In Section WL-equivalent GNNs with continuous features, we showed how the 2 -SEWL test can be realized by a continuous piecewise differentiable architecture that uses sort-based multi-set injective functions. The complexity of this construction was O ( n 4 log( n ) ) . Similarly, the ( d -1) -SEWL and ( d -1) -EWL tests can be computed with complexity of O ( n d +1 log( n ) ) (in the scenario where d stays constant and n → ∞ ) . The leading order of the computation complexity stems from computing the n d -1 colorings ( C ( 1 ) ( i ) , i ∈ [ n ] d -1 ) and embedding the multiset derived by aggregating over the 'neighbors' of each tuple, each one of those requires O ( n 2 log( n )) operations.