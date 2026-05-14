## Nonparametric Stochastic Contextual Bandits

Melody Y. Guan ∗

Stanford University 450 Serra Mall Stanford, California 94305

mguan@stanford.edu

## Abstract

We analyze the K -armed bandit problem where the reward for each arm is a noisy realization based on an observed context under mild nonparametric assumptions. We attain tight results for top-arm identification and a sublinear regret of ˜ O ( T 1+ D 2+ D ) , where D is the context dimension, for a modified UCB algorithm that is simple to implement ( k NN-UCB). We then give global intrinsic dimension dependent and ambient dimension independent regret bounds. We also discuss recovering topological structures within the context space based on expected bandit performance and provide an extension to infinite-armed contextual bandits. Finally, we experimentally show the improvement of our algorithm over existing multi-armed bandit approaches for both simulated tasks and MNIST image classification.

## Introduction

Multi-armed bandits (MABs) are an important sequential optimization problem introduced by Robbins (1985). These models have extensively been used in a wide variety of fields related to statistics and machine learning.

The classical MAB consists of K arms where at each point in time the learner can sample (or pull) one of them and observe a reward. Then various objectives can be established, such as finding the best arm (Top-Arm Identification) or minimizing some regret over time.

For contextual bandits (also referred to as bandits with side information or covariates), the learner has access to a context on which the payoffs depend. Then, based on the observations, we aim to determine the best policy (or contextto-arm mapping) and to optimize some notion of regret.

Most approaches to stochastic contextual bandits make strong assumptions on the payoffs. A popular approach models the mean reward for each arm as being linear in the context space (Chu et al. 2011; Li et al. 2010). However, this is rarely the case in real data. In this paper, we take a more general approach and allow the reward functions to be non-linear and of arbitrary shape.

Using recent developments in nonparametric statistics (Jiang 2017b), we show that with simple and easily implementable techniques, we can construct bandit algorithms

∗ Equal Contribution.

Copyright c © 2018, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

## Heinrich Jiang ∗

Google

1600 Amphitheatre Pwky Mountain View, California 94043 heinrich.jiang@gmail.com

which can learn over the entire context space with strong guarantees, despite the difficulty that arises with allowing a wide variety of reward functions. While this is not the first work which blends nonparametric statistics with bandits, we are the first to show simple and practical methods while still maintaining strong theoretical guarantees.

We reanalyze the uniform and upper confidence bound sampling strategies and demonstrate what nonparametric approaches can offer to contextual bandit learning. No other technique can adapt to the inherently difficult and complex real world reward functions while allowing such a strong theoretical understanding of the underlying algorithms.

While nonparametric models are powerful in their ability to learn arbitrary functions free of distributional assumptions, a major weakness is the curse of dimensionality. In order to have any theoretical guarantees, they require an exponential-in-dimension number of samples. However, when the data lies on an unknown low-dimensional structure such as a manifold, we show that our algorithms can converge as if the data was on a lower dimension and not in the potentially much large ambient dimension. Another striking fact is that no preprocessing of the data is required. This is of practical importance because modern data has increasingly more features but the underlying degrees of freedom often remain small.

We then discuss recovering geometric structures in the context space based on bandit performance. Specifically, we recover the connected components of the context space in which a particular bandit is the top-arm. Although learning a context-to-arm mapping gives us the estimated top-arm at each point in the context space, this alone does not tell the space's topological structure, such as the number and shapes of connected components. We recover these structures with uniform consistency guarantees with mild assumptions, where the shapes and relative positions of the components can be arbitrary and the number of such components is recovered automatically.

We then provide an extension to infinite-armed bandits and conclude with empirical results from simulations and image classification on the MNIST dataset.

## Setup

Suppose there are K bandit arms indexed in [ K ] . At each time-step t , the learner observes a context x t ∈ R D

where x t is drawn i.i.d. from a context density p X with compact support X bounded below away from zero (e.g. inf x ∈X p X ( x ) ≥ p 0 for some p 0 ≥ 0 ). Then the learner chooses an arm I t ∈ [ K ] and observes reward

<!-- formula-not-decoded -->

where ξ t is drawn according to white noise random variable ξ and f i : X → R is the i -th arm's mean reward. We make the following assumptions.

Assumption 1. (Lipschitz Mean Reward) There exists L such that | f i ( x ) -f i ( x ′ ) | ≤ L | x -x ′ | for all x, x ′ ∈ X and i ∈ [ K ] .

Assumption 2. (Sub-Gaussian White noise) ξ satisfies E [ ξ ] = 0 and is sub-Gaussian with parameter σ 2 (i.e. E [exp( λξ )] ≤ exp( σ 2 λ 2 / 2) for all λ ∈ R ).

We require the finite-sample strong uniform consistency result (Theorem 1) for k -NN regression defined as fellows: Definition 1 ( k -NN) . Let the k -NN radius of x ∈ X be r k ( x ) := inf { r : | B ( x, r ) ∩ X | ≥ k } where B ( x, r ) := { x ∈ X : | x -x ′ | ≤ r } and the k -NN set of x ∈ X be N k ( x ) := X ∩ B ( x, r k ( x )) . Then for x ∈ X ,

<!-- formula-not-decoded -->

Theorem 1. (Rate for k -NN (Jiang 2017b)) Let δ &gt; 0 . There exists N 0 and universal constant C such that if n ≥ N 0 and k = glyph[floorleft] n 2 / (2+ D ) glyph[floorright] , then with probability at least 1 -δ ,

<!-- formula-not-decoded -->

It will be implicitly understood from here on that ̂ f i denotes the k -NN regression estimate of f i under the settings of Theorem 1.

## Top-Arm Identification

## Algorithm 1 Uniform Sampling

- 1: Parameters: T , total number of time steps.
- 2: For each arm i of the K arms:
- 4: Pull arm I t := i .
- 3: For each time step t ∈ [ ( i -1) T K +1 , iT K ] :
- 5: Define ̂ f i : X → R to be the k -NN regression estimator from the sampled context and reward observations for each i ∈ [ K ] .

Definition 2. ( glyph[epsilon1] -optimal arm) Arm i is be glyph[epsilon1] -optimal at context x ∈ X if max j ∈ [ K ] f j ( x ) -f i ( x ) ≤ glyph[epsilon1] .

Following we show a uniform (over context) result about glyph[epsilon1] -optimal arm recovery:

Theorem 2. ( glyph[epsilon1] -optimal arm recovery) Let δ &gt; 0 . For Algorithm 1, with probability at least 1 -δ/K , if

<!-- formula-not-decoded -->

then ˆ π ( x ) := argmax i ∈ [ K ] ˆ f i ( x ) is glyph[epsilon1] -optimal at context x uniformly for all x ∈ X .

Remark 1. This result shows that with ˜ O ( glyph[epsilon1] -(2+ D ) ) samples, we can determine an glyph[epsilon1] -approximate best arm. Known lower bounds in nonparametric regression stipulate that we need Ω( glyph[epsilon1] -(2+ D ) ) to identify differences between functions of size glyph[epsilon1] so our result matches lower bounds up to logarithmic factors.

Proof. By Theorem 1, it follows that based on the choice of T , each arm has at least enough time such that sup x ∈X | ̂ f i ( x ) -f i ( x ) | ≤ glyph[epsilon1]/ 2 . Thus, we have ∀ x ∈ X , defining π ( x ) = max j ∈ [ K ] f j ( x ) ,

<!-- formula-not-decoded -->

## Regret Analysis For UCB Strategy

Define T i ( t ) to be the number of times arm i was pulled by time t .

## Algorithm 2 Upper Confidence Bound (UCB)

- 1: Parameters: M 0 , M 1 , δ , T .
- 2: Define σ ( n ) = M 1 √ log n (log( nK/δ )) · n -1 / (2+ D ) .
- 3: Pull each of the K arms M 0 times.
- 4: For each round t = KM 0 , KM 0 +1 , . . . , T :
- 5: Pull I t := argmax i ∈ [ K ] ̂ f i ( t ) + σ ( T i ( t -1)) .

We use the following notion of regret.

<!-- formula-not-decoded -->

Remark 2. Note that this notion of regret is different from those studied in classical MABs as well as other works in nonparametric contextual bandits. Usually the expected form E [ R T ] is bounded. Here, our regret analysis is not under this expectation and hence is a stronger notion of regret.

Theorem 3. Let δ &gt; 0 . Suppose that M 0 ≥ N 0 and M 1 &gt; C in Algorithm 2. Then we have that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Remark 3. This shows a sub-linear regret of ˜ O ( T 1+ D 2+ D ) .

Proof. Denote ̂ f i,T i ( t ) to be the k -NN regression estimate of f i at time t . Letting C 0 = KM 0 max i || f i || ∞ , we have by Theorem 1

<!-- formula-not-decoded -->

The first inequality holds because the confidence bound of a sub-optimal arm must be higher than that of the optimal at x t in order for that arm to be chosen and the regret at that time-step is bounded by the confidence bound. The second inequality holds because of the following simple combinatorial argument. Each time a suboptimal arm is chosen, its count increments, or otherwise there is no regret incurred.

## Contextual Bandits on Manifolds

Assumption 3. (Manifold Assumption) p X and the family of f i are supported on M , where:

- M is a d -dimensional smooth compact Riemannian manifold without boundary embedded in compact subset X ⊆ R D .
- The volume of M is bounded above by a constant.
- M has condition number 1 /τ , which controls the curvature and prevents self-intersection.

Let p X be the density of P with respect to the uniform measure on M .

Theorem 4. (Manifold Rate for k -NN (Jiang 2017b)) Let δ &gt; 0 . There exists N 0 and universal constant C such that if n ≥ N 0 and k = glyph[floorleft] n 2 / (2+ d ) glyph[floorright] , then with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Then, simply by using Theorem 4 instead of Theorem 1, we automatically enjoy faster rates for Theorems 2 and 3.

Theorem 5. ( glyph[epsilon1] -optimal arm recovery on manifolds) Let δ &gt; 0 . For Algorithm 1, with probability at least 1 -δ/K , if

<!-- formula-not-decoded -->

then ˆ π ( x ) := argmax i ∈ [ K ] ̂ f i ( x ) is glyph[epsilon1] -optimal at context x uniformly for all x ∈ X .

Remark 4. Now the sample complexity is ˜ O ( glyph[epsilon1] 2+ d ) instead of ˜ O ( glyph[epsilon1] 2+ D ) .

Theorem 6. (UCB Regret Analysis on Manifolds) Let δ &gt; 0 . Suppose that M 0 ≥ N 0 and M 1 &gt; C in Algorithm 2. Then we have that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

## Topological Analysis

In this section, we discuss how topological features about the bandit arms can be recovered. This is similar to recovering the Hartigan notion of clusters as level-sets of the density functions from a finite sample (Chaudhuri and Dasgupta 2010; Jiang 2017a), but here, we find similar structures in the reward functions based on noisy observations of them. We give procedures which can estimate with consistency guarantees the following structure: maximal connected regions in X where a particular arm is the top-arm.

From the uniform sampling strategy earlier, we obtained estimated policy ˆ π which is δ -optimal uniformly in X with high probability. Although this is already powerful in giving us the mapping between context space and the corresponding top-arm, it does not immediately tell us the topological features of this mapping. In this subsection, we discuss how to recover the connected components of { x ∈ X : r i ( x ) = max j ∈ [ K ] r j ( x ) } , the region where arm i is the top-arm.

We give the following simple procedure.

Algorithm 3 Recovering Regions where i -th arm is top arm.

- 1: Given: Bandit arm i and R &gt; 0 .
- 2: Pull each of the K arms T/K times.
- 3: Let G be the graph with vertices { x t : t ∈ [ T ] , ̂ f i ( x t ) = max j ∈ [ K ] ̂ f j ( x t ) } and edges between vertices whose euclidean distance is at most R .
- 4: return The connected components of G .

We now give a consistency result for Algorithm 3.

First, we require the following regularity assumption, which ensures that there are no full-dimensional regions where the top-arm is not unique. This ensures that it is possible to unambiguously recover the regions where a particular arm is top.

Assumption 4. The region in X where the top-arm is not unique has measure 0 , and for each arm i , the region X i where it is unique can be partitioned into full-dimensional connected components.

Our rates will be in terms of the Hausdorff distance.

## Definition 3.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 7. Suppose that X i := { x ∈ X : f i ( x ) = max j ∈ [ K ] f j ( x ) } . Let C 1 , ..., C l be the maximal connected components of X i . Define the following minimum distance between two connected components.

glyph[negationslash]

<!-- formula-not-decoded -->

Also define the following minimum separation in the reward functions

<!-- formula-not-decoded -->

Then the following holds simultaneously for all C 1 , ..., C l . Let Algorithm 3 with setting 0 &lt; R &lt; R 0 / 4 return ̂ C 1 , ..., ̂ C ˆ l . Then for n sufficiently large, ˆ l = l and there exists permutation γ of [ l ] such that

<!-- formula-not-decoded -->

for some ξ that satisfies ξ ( n ) → 0 as n →∞ .

Proof. We first show that no two connected components can appear in the same returned component in Algorithm 3. We choose n sufficiently large such that in light of Theorem 1, we have

<!-- formula-not-decoded -->

. Then, uniformly for any x glyph[negationslash]∈ X i ⊕ R 0 / 4 , we have

<!-- formula-not-decoded -->

Thus, X i ⊕ R 0 / 4 is disjoint from the returned points. Since R &lt; R 0 / 4 , it follows that no two connected components points will appear in the same returned connected component from Algorithm 3.

Next, we show that for each connected component C p , there exists ̂ C q for some q ∈ [ ̂ l ] such that d H ( ̂ C q , C p ) → 0 . It suffices to show that for each r &gt; 0 , we have that for n sufficiently large, d H ( ̂ C q , C p ) &lt; r . There are thus two directions to show, that ̂ C p ⊆ C q ⊕ r and C q ⊆ ̂ C p ⊕ r . To show the first, define

<!-- formula-not-decoded -->

Then choose n sufficiently large such that in light of Theorem 1, we have

<!-- formula-not-decoded -->

glyph[negationslash]

. Then we have for all x ∈ ̂ C p , if x = C q ⊕ r/ 2 , then

<!-- formula-not-decoded -->

thus, x ∈ C q ⊕ r/ 2 ⊆ C q ⊕ r . The other direction follows from a similar argument.

All that remains is to show that such points appear in in the same connected component in the graph computed by Algorithm 3. This follows from uniform concentration bounds on balls (e.g. Chaudhuri and Dasgupta (2010)).

## Infinite-Armed Bandits

In this section, we consider the setting where the action space A is no longer a finite set of bandits, but a compact subset of R D ′ for some D ′ &gt; 0 .

We given analogous results for the uniform sampling toparm identification and regret bounds for UCB-type strategy.

Definition 4. (Mean Reward function)

<!-- formula-not-decoded -->

where f ( x, a ) is the expected reward of action a ∈ A at context x ∈ X .

Assumption 5. (Lipschitz Reward) There exists L &gt; 0 such that for all x, x ′ ∈ X and a, a ′ ∈ A , | f ( x, a ) -f ( x ′ , a ′ ) | ≤ L | ( x, a ) -( x ′ , a ′ ) | , where ( x, a ) represents the ( D + D ′ ) -dimensional concatenation of x and a .

Then at each time t , the learner chooses arm a t ∈ A and observes context x t ∈ X and a stochastic reward

<!-- formula-not-decoded -->

where ξ 1 , ... are i.i.d. white noise with mean 0 and variance σ 2 .

## Algorithm 4 Infinite-Armed Uniform Sampling

- 1: Parameters: T , total number of time steps.
- 2: For t = 1 , ..., T :
- 3: Pull I t , sampled uniformly from A .
- 4: Observe context x t and reward R t .
- 5: Define ˆ f to be the k -NN regression estimate from samples ( a 1 , R 1 ) , ..., ( a T , R T ) with setting k = glyph[floorleft] n 2 / (2+ D + D ′ ) glyph[floorright] .

Definition 5. ( glyph[epsilon1] -optimal arm) Define arm a ∈ A to be glyph[epsilon1] -optimal at context x ∈ X if sup a ′ ∈A f ( x, a ′ ) -f ( x, a ) ≤ glyph[epsilon1] .

Following is a uniform (over context and action space) result about glyph[epsilon1] -optimal arm recovery:

Theorem 8. ( glyph[epsilon1] -optimal arm recovery) There exists constant ˜ C 1 , ˜ C 2 such that the following holds. Let δ &gt; 0 . For Algorithm 4, with probability at least 1 -δ , we have that for

<!-- formula-not-decoded -->

arm ˆ π ( x ) := argmax a ∈A ˆ f ( x ) is glyph[epsilon1] -optimal at context x uniformly for all x ∈ X .

Proof. By Theorem 1, it follows that based on the choice of T , there is enough time spent on pulling each arm such that sup a ∈A ,x ∈X | ˆ f ( x, a ) -f ( x, a ) | ≤ glyph[epsilon1]/ 2 . Thus, we have ∀ x ∈ X , defining π ( x ) = argmax a ∈A f ( x, a ) ,

<!-- formula-not-decoded -->

as desired.

Algorithm 5 Infinite-Armed Upper Confidence Bound (UCB)

- 1: Parameters: M , M 1 , T
- 2: Define σ ( n ) = M 1 n -1 / (2+ D + D ′ ) .
- 3: For t = 1 , ..., M :
- 4: Sample a t uniformly from A .
- 5: Observe context x t and reward R t .
- 6: For t = M +1 , ..., T :
- 7: Choose I t := argmax a ∈A ̂ f ( x t , a ) + σ ( t ) .

Finally, using the notion of regret

<!-- formula-not-decoded -->

we give the following result. The proof idea is similar to that of Theorem 3 and is omitted here.

Theorem 9. There exists ˜ C 1 and ˜ C 2 such that the following holds. Let δ &gt; 0 . Suppose that M and M 1 are chosen sufficiently large in Algorithm 5 depending on f and σ . Then we have that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Remark 5. This shows a sub-linear regret of ˜ O ( T 1+ D + D ′ 2+ D + D ′ ) .

## Related Works

Canonical works for the standard bandit problem are Lai and Robbins (1985); Berry and Fristedt (1985); Gittins, Glazebrook, and Weber (2011); Auer et al. (2002); Cesa-Bianchi and Lugosi (2006); Bubeck and Cesa-Bianchi (2012).

Work in contextual bandits can be roughly classified into adversarial and stochastic approaches. Much of the former, initiated by Auer et al. (2002), assumes that there is an adversarial game between nature and the learner where, based on a context seen by both players, nature generates rewards for each arm at the same time the learner chooses an arm. Solutions typically involve game theoretical methods. In the stochastic approach, one assumes that the rewards for the arms are generated by a context-dependent distribution.

Approaches to modeling the arm rewards as a function of context are most commonly parametric. One of the most popular is that of linear payoffs, studied under a minimax framework (Goldenshluger and Zeevi 2009; 2013), with UCB-type algorithms (Chu et al. 2011; Li et al. 2010; Auer et al. 2002), or with Thompson sampling (Agrawal and Goyal 2013).

However, it is often the case that the dependency between the payoffs and the contexts are complex and therefore difficult to capture with models such as linear payoffs, many of which requiring strong assumptions on the data. To alleviate this, we can go beyond parametric modeling and blend nonparametric statistics with contextual bandits. Despite the advantage of learning much more general context-payoff dependencies, this line of work has received far less attention.

To the best of our knowledge, the first such work appeared in Yang and Zhu (2002), who used histogram, k -NN, and kernel methods and showed asymptotic convergence rates. Rigollet and Zeevi (2010); Perchet and Rigollet (2013) then combined histogram-type binning techniques in nonparametric statistics to obtain strong regret guarantees for contextual bandits with optimality guarantees.

Lu, P´ al, and P´ al (2010) study an interesting setting where the reward depends on a Lipschitz measure which is jointly in the context and the action space. They provide upper and lower regret bounds based on a covering argument and give results in terms of the packing dimension. This is highly related to the infinite-armed bandit setting in the present work; we provide similar regret guarantees but with a simple and practical procedure.

More recently, Qian and Yang (2016b); Qian and Yang (2016a) use the strong uniform consistency properties of kernel smoothing regression to establish regret guarantees.

Langford and Zhang (2008); Dudik et al. (2011) alternatively impose neither linear nor smoothness assumptions on the mean reward function. The former propose a modification of an glyph[epsilon1] -greedy policy and showed that expected regret converges to 0 while the latter considers a finite class of policies.

In this paper, using recent finite-sample results about k -NN regression established in Jiang (2017b), we show that using the simple k -NN regression is an effective alternative approach. Moreover, unlike many other nonparametric techniques, k -NN adapts to a lower intrinsic dimension (Kpotufe 2011) and thus we show that our regret bounds can adapt to a lower intrinsic dimension automatically and perform as if we were operating in that lower dimensional space.

## Experiments

## Simulations

We consider three two-arm bandit scenarios in the twodimensional unit square, where p X is uniform. We set arm i ∈ { 1 , 2 } to be top in region R i respectively. Figure 1 illustrates the regions for the different scenarios.

- Scenario 1 (Quintic Function): We define two regions above and below a quintic function:
- Scenario 2 (Smiley): We use two circles and a semicircle to demarcate the regions in a 'smiley face' pattern.
- Scenario 3 (Bullseye): We define the regions using the alternating regions of four concentric circles centered in the support.

The true reward functions of the two arms are as follows.

glyph[negationslash]

<!-- formula-not-decoded -->

The learner observes the rewards with white noise random variable ξ ∼ N ( µ = 0 , σ = 0 . 5) .

We compare the performance of k -NN regression (nonparametric) and Ridge regression at top-arm identification and regret minimization in the three scenarios. Mirroring our theoretical discussion, we use uniform sampling for top-arm identification and UCB strategy for regret analysis. Note that Ridge regression with UCB is the LinUCB algorithm.

Table 1: Top-arm identification and regret results from Ridge and k -NN regressors. Each model was tuned individually and optimal hyperparameters are shown. k -NN performs better on both metrics for all three scenarios.

|                                          | Quintic Function   | Quintic Function   | Smiley   | Smiley   | Bullseye   | Bullseye   |
|------------------------------------------|--------------------|--------------------|----------|----------|------------|------------|
|                                          | Ridge              | kNN                | Ridge    | kNN      | Ridge      | kNN        |
| Top-Arm Test Error from Uniform Sampling | 0.065              | 0.002              | 0.080    | 0.000    | 0.335      | 0.005      |
| Number of samples                        | 500k               | 500k               | 2k       | 5000k    | 100k       | 500k       |
| Number of neighbors                      | -                  | 100                | -        | 50       | -          | 20         |
| Test Regret from UCB sampling            | 0.0315             | 0.001              | 0.0375   | 0.0135   | 0.161      | 0.004      |
| Number of samples                        | 1k                 | 500k               | 5k       | 1000k    | 50k        | 1000k      |
| Number of neighbors                      | -                  | 100                | -        | 20       | -          | 100        |

<!-- image -->

Figure 1: Top-arm boundaries. Red and blue regions correspond to where top-arm is arm 1 and 2 respectively.

Figure 2: Observed reward density plots from 10k uniform samples illustrating pseudo-randomness of training data. In the colormap (right) warmer colors correspond to higher values, normalized on the range of the observed rewards.

<!-- image -->

Figure 3: Test results on top-arm identification using Ridge regression and 25-NN regression. Contexts are labeled in red and blue if arms 1 and 2 are estimated to be top respectively.

<!-- image -->

Qualitative Analysis We first qualitatively show that k -NN regression can successfully model the bandits whereas the linear method cannot. The difficulty of the task is illustrated by Figure 2, which plots 10k uniformly sampled samples from each scenario with a colormap. We can see that a human would have a hard time recovering the regions where each arm is top due to the randomness in the observed rewards. This randomness is considerable as we set σ = 0 . 5 to be the same as | f i ( x ∈ R i ) -f i ( x / ∈ R i ) | .

We fix the number of training samples N to 10k and the number of nearest neighbors to k = 25 . We evaluate on 10k random test samples. Figure 3 shows that k -NN regression does an excellent job of reproducing the region boundaries. Ridge regression does a poor job in the Quintic Function case, making a linear approximation to the quintic curve, and completely fails in the Smiley and Bullseye Cases, simply choosing the arm whose top-arm region is larger.

Quantitative Analysis We report numerical results and optimal hyperparameters in Table 1. We tuned other hyperparameters using grid search on a validation set of size 1k using grid search and we evaluate performance of our models on a test set of size 1k. We use the UCB strategy in Auer et al. (2002) (a simplified version of UCB by Agrawal and Goyal (2013)). We found that a confidence level of 0 . 1 worked well for all settings. We see that k -NN significantly outperforms Ridge regression for both top-arm identification and regret minimization in all three scenarios (Table 1).

## Image Classification Experiments

We extend our experiments to image classification of the canonical MNIST dataset, which consists of 60k training images and 10k test images of isolated, normalized, handwritten digits. The task is to classify each 28 × 28 image into one of ten classes. We reframe this as a contextual MAB problem by treating the classes as arms and the images as the contexts. Note that for every context, the payoff of all arms are known: 1 if the class is the true label and 0 otherwise. We compare k -NN and Ridge regressions at regret minimization using the UCB strategy. As before we use the UCB strategy in Auer et al. (2002) and fix the confidence level to 0.1. We do not employ any data augmentation.

We obtain test regret of 17.5% from LinUCB with α = 5 , where α is the coefficient of L2 regularization, and significantly lower test regret of 5.8% from 4-NNUCB. Figure

4 shows that k -NN regression maintains lower regret than Ridge regression over a range of values of k and α . We note that Ridge regression working well for relatively large values of α itself suggests that it is a poor model for the task.

Figure 4: Test results on regret minimization for MNIST image classification over varied values of α (for LinUCB) and k (for k -NNUCB). The nonparametric approach achieves significantly lower regret over a range of hyperparameters.

<!-- image -->

## Conclusion

For the multi-armed bandit setting, we use nonparametric regression to attain tight results for top-arm identification and a sublinear regret of ˜ O ( T 1+ D 2+ D ) , where D is the dimension of the context. We also show that if the underlying context space has a lower intrinsic dimension d &lt; D , then our algorithm automatically adapts to the lower dimension and attains a faster rate of ˜ O ( T 1+ d 2+ d ) . We also provide a procedure for recovering the maximal connected regions in a support where a particular arm is the top-arm and provide a consistency analysis. We then give a natural extension to infinitearmed contextual bandits. Our simulations confirm that our method is able to learn in the contextual setting with arbitrary decision boundaries, even in the presence of significant noise, and our experiments on classification of MNIST images demonstrate superior performance of our method over LinUCB on a real world task.

## References

Agrawal, S., and Goyal, N. 2013. Thompson sampling for contextual bandits with linear payoffs. In International Conference on Machine Learning , 127-135.

Auer, P.; Cesa-Bianchi, N.; Freund, Y.; and Schapire, R. E. 2002. The nonstochastic multiarmed bandit problem. SIAM journal on computing 32(1):48-77.

Berry, D. A., and Fristedt, B. 1985. Bandit problems: sequential allocation of experiments (Monographs on statistics and applied probability) , volume 12. Springer.

Bubeck, S., and Cesa-Bianchi, N. 2012. Regret analysis of stochastic and nonstochastic multi-armed bandit problems. Foundations and Trends in Machine Learning 5(1):1-122.

Cesa-Bianchi, N., and Lugosi, G. 2006. Prediction, learning, and games . Cambridge university press.

Chaudhuri, K., and Dasgupta, S. 2010. Rates of convergence for the cluster tree. In Advances in Neural Information Processing Systems , 343-351.

Chu, W.; Li, L.; Reyzin, L.; and Schapire, R. E. 2011. Contextual bandits with linear payoff functions. In International Conference on Artificial Intelligence and Statistics , 208-214.

Dudik, M.; Hsu, D.; Kale, S.; Karampatziakis, N.; Langford, J.; Reyzin, L.; and Zhang, T. 2011. Efficient optimal learning for contextual bandits. arXiv preprint arXiv:1106.2369 .

Gittins, J.; Glazebrook, K.; and Weber, R. 2011. Multi-armed bandit allocation indices . John Wiley &amp; Sons.

Goldenshluger, A., and Zeevi, A. 2009. Woodroofes onearmed bandit problem revisited. The Annals of Applied Probability 19(4):1603-1633.

Goldenshluger, A., and Zeevi, A. 2013. A linear response bandit problem. Stochastic Systems 3(1):230-261.

Jiang, H. 2017a. Density level set estimation on manifolds with dbscan. arXiv preprint arXiv:1703.03503 .

Jiang, H. 2017b. Rates of uniform consistency for k-nn regression. arXiv preprint arXiv:1707.06261 .

Kpotufe, S. 2011. k-nn regression adapts to local intrinsic dimension. In Advances in Neural Information Processing Systems , 729-737.

Lai, T. L., and Robbins, H. 1985. Asymptotically efficient adaptive allocation rules. Advances in applied mathematics 6(1):4-22.

Langford, J., and Zhang, T. 2008. The epoch-greedy algorithm for multi-armed bandits with side information. In Advances in neural information processing systems , 817-824.

Li, L.; Chu, W.; Langford, J.; and Schapire, R. E. 2010. A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th international conference on World wide web , 661-670. ACM.

Lu, T.; P´ al, D.; and P´ al, M. 2010. Showing relevant ads via lipschitz context multi-armed bandits. In Thirteenth International Conference on Artificial Intelligence and Statistics .

Perchet, V., and Rigollet, P. 2013. The multi-armed bandit problem with covariates. The Annals of Statistics 41(2):693721.

Qian, W., and Yang, Y. 2016a. Kernel estimation and model combination in a bandit problem with covariates. Journal of Machine Learning Research .

Qian, W., and Yang, Y. 2016b. Randomized allocation with arm elimination in a bandit problem with covariates. Electronic Journal of Statistics 10(1):242-270.

Rigollet, P., and Zeevi, A. 2010. Nonparametric bandits with covariates. arXiv preprint arXiv:1003.1630 .

Robbins, H. 1985. Some aspects of the sequential design of experiments. In Herbert Robbins Selected Papers . Springer. 169-177.

Yang, Y., and Zhu, D. 2002. Randomized allocation with nonparametric estimation for a multi-armed bandit problem with covariates. The Annals of Statistics 30(1):100-121.