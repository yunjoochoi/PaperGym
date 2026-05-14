## Robust Training of Neural Networks using Scale Invariant Architectures

| Zhiyuan Li ∗             | zhiyuanli@cs.princeton.edu   |
|--------------------------|------------------------------|
| Princeton University     |                              |
| Srinadh Bhojanapalli     | bsrinadh@google.com          |
| Google Research New York |                              |
| Manzil Zaheer            | manzilzaheer@google.com      |
| Google DeepMind New York |                              |
| Sashank J. Reddi         | sashank@google.com           |
| Google Research New York |                              |
| Sanjiv Kumar             | sanjivk@google.com           |
| Google Research New York |                              |

## Abstract

In contrast to SGD, adaptive gradient methods like ADAM allow robust training of modern deep networks, especially large language models. However, the use of adaptivity not only comes at the cost of extra memory but also raises the fundamental question: can non-adaptive methods like SGD enjoy similar benefits ? In this paper, we provide an affirmative answer to this question by proposing to achieve both robust and memory-efficient training via the following general recipe: (1) modify the architecture and make it scale invariant , i.e. the scale of parameter doesn't affect the output of the network, (2) train with SGD and weight decay, and optionally (3) clip the global gradient norm proportional to weight norm multiplied by √ 2 λ η , where η is learning rate and λ is weight decay. We show that this general approach is robust to rescaling of parameter and loss by proving that its convergence only depends logarithmically on the scale of initialization and loss, whereas the standard SGD might not even converge for many initializations. Following our recipe, we design a scale invariant version of BERT, called SIBERT, which when trained simply by vanilla SGD achieves performance comparable to BERT trained by adaptive methods like ADAM on downstream tasks.

## 1 Introduction

Neural architectures like transformers are the cornerstone for modern machine learning applications. However, training them is difficult and often results in training instability Liu et al. (2020); Zhang et al. (2020b). To enable stable training, one typically requires adaptive and carefully tuned learning rates. However, the reason behind this issue is not very well-understood and lacks a formal treatment.

In this paper, we hypothesize that a primary cause of such behavior is the k -homogeneous ( k ≥ 2 ) nature of the network i.e., property where network's output is scaled by s k when its parameters are scaled by s . To illustrate our point, we consider the following instructive toy model.

Example 1.1. Consider logistic regression with 1 -dimensional non-separable data, { z i , y i } n i =1 ∈ ( R ×{± 1 } ) n . The loss is defined as L ( x 1 , , . . . , x 2 k ) = ˜ L ( X ) := -∑ n i =1 ln(1 + e -z i y i X ) where X = x 1 . . . x 2 k and k ≥ 2 .

∗ Work done at Google Research New York

Since ˜ L is convex with bounded smoothness in X , there exists step size that are independent of any initialization that allow GD to converge to the optimal solution. In sharp contrast, the reparametrized loss L ( x 1 , , . . . , x 2 k ) with 2 k -homogeneous structure does not enjoy this nice stability property - the learning rate has to be tuned according to the initialization. In particular, when η ≥ 2 |∇ ˜ L ( X (0)) | ( X (0)) 1 k -1 and X (0) &gt; X ∗ where X ∗ &gt; 0 is the global minimizer, X ( t ) will monotonically increase and explode, if all x i are initialized to be the same.

We refer the reader to Appendix B for a formal justification of this example. In the above example, the success of optimization is very sensitive to the right choice of the learning rate that depends on the initialization. Furthermore, the training cannot recover once the norm explodes due to large gradient update.

In the above one-dimensional example it is still possible to find a small workable learning rate by extensive grid search that depends on the initial point, however, the situation can get worse when the k -homogeneous structure has an unbalanced initialization as below.

Example 1.2. Consider solving low-rank matrix decomposition by Gradient Descent. Let L ( A,B ) = 1 2 ∥ ∥ AB glyph[latticetop] -Y ∥ ∥ 2 2 where A,B ∈ R d × r are both initialized i.i.d. gaussian with covariance σ 2 A glyph[greatermuch] σ 2 B ≈ σ -2 A , Y ∈ R d × d and d glyph[greatermuch] r .

Solving this optimization problem requires A and B learning the column and row space of Y respectively, but the unbalanced initialization will force the learning rate to be small enough such that B does not explode and, thus, A is almost frozen. To see this, note in the standard convergence analysis of GD, we need LR smaller than 2 / ∥ ∥ ∇ 2 L ∥ ∥ to ensure the Descent Lemma holds, i.e. , loss decreases in a single step. Here we have that the smoothness w.r.t A (fixing B ) is λ max ( BB T ) and the smoothness w.r.t. B (fixing A ) is λ max ( AA T ) . Thus, LR can be at most O ( 1 σ 2 A ) , but the gradient of A is only of magnitude O ( σ B ) , resulting in A learning the column space slowly. Specifically, when d = 1 and Y = 0 and for any r ≥ 1 , choosing η &gt; 4 ‖ ∇ 2 B L ‖ will cause GD to provably explode (Lewkowycz et al., 2020).

Similar issues can exist in deep neural networks as the k -homogeneous structure is quite common. For instance, Liu et al. (2020) identified the gradient norm varies with depth and that no single learning rate is globally optimal for all layers. To this end, one has to resort to adaptive methods like ADAM to handle the k -homogeneous structure of deep networks and allow for its robust training. However, this not only comes at the expense of higher memory, but also raises the key question of our interest:

Can non-adaptive methods like SGD enjoy fast and robust convergence without training instability?

Answering this question, requires us to first define our notion of robustness. In this paper, we primarily aim for three aspects of robustness by preventing: explosion of parameters (e.g. due to frequent large gradient updates), slow progress in training (e.g. due to loss plateaus) and loss explosion or spikes (e.g. due to possibly infrequent large magnitude updates). In this paper, we propose a simple yet powerful general approach for achieving such fast and robust convergence. At a high level, our recipe for robust training includes three key ingredients:

1. Designing architectural scale invariance which allows for improved training stability and prevents explosion of the parameters . Weshow that by using scale invariance in the architecture (i.e., making the network 0 -homogeneous), one can effectively control the gradient updates when the parameter norm is large.
2. Using SGD with weight decay for training, wherein enabling weight decay improves training efficiency under rescaling of loss and initialization . While scale invariance prevents explosion of parameters, the training convergence has strong dependence on initialization scale and learning rate, which can make training inefficient in face of parameter and initialization rescaling. Use of SGD with weight decay circumvents this issue.
3. Using a novel Relative Global Clipping to prevent spikes in training loss and improve overall convergence speed . Although scale invariance in the architecture already guarantees the training stability, it does not prevent severe non-monotonic loss explosion. By using a new global clipping approach, we show that one can prevent such loss explosions effectively.

We show that this surprisingly simple training recipe can not only improve the memory efficiency over adaptive methods but also achieves robust training. In light of the above background, we list our main contributions below.

- In Section 3, we propose a new general recipe for memory efficient, robust training using (1) scale invariant architecture; (2) SGD+WD for training and (3) a novel clipping rule, called Relative Global Clipping, for clipping the updates. Following this recipe, we design a new variant of BERT called Scale Invariant BERT (SIBERT).
- In Sections 4.1 and 4.2, we prove the convergence rate to the approximate first order point for GD and SGD for scale invariant loss. We show that SGD+WD matches the standard rates, even without the knowledge about the smoothness of loss and is robust to the scale of initialization or loss.
- In Section 4.3, we show SGD+WD with Relative Global Clipping has better parameter norm convergence via a novel analysis. With assumptions that the clipping does not bring too much bias in expected gradients, we show similar convergence result to SGD+WD.
- In our empirical analysis in Section 5, we demonstrate that SIBERT trained using simple SGD can achieve performance comparable to standard BERT trained with ADAM. Furthermore, we also verify our theoretical claims. To our knowledge, this is the first time a BERT-like model has been effectively trained using vanilla SGD.

## 2 Related Work &amp; Background

The literature on adaptive methods and scale invariance in neural networks is vast, so we only discuss works that are most relevant to our paper.

Adaptive Methods &amp; Clipping Methods. Adaptive learning rates have long been studied Polyak (1987). In machine learning, adaptive learning rates have been popularized by ADAGRAD, which particularly benefits from sparse stochastic gradients Duchi et al. (2011). Inspired by ADAGRAD, several adaptive methods, like ADAM, RMSPROP and its variants have been proposed in the deep learning community Kingma &amp; Ba (2015); Tieleman &amp; Hinton (2012); Reddi et al. (2019); You et al. (2020); Shazeer &amp; Stern (2018). These approaches have been crucial in the success of many deep learning applications Vaswani et al. (2017); Devlin et al. (2018); Raffel et al. (2019). Several works have studied the benefits of adaptive methods in deep learning settings (e.g. Liu et al. (2020); Zhang et al. (2020b)). However, as mentioned earlier, these benefits come at the cost of computational and memory efficiency. Anil et al. (2019) proposed a variant of ADAGRAD requiring fewer parameters for adaptivity, but still requires momentum. ADAFACTOR (Shazeer &amp; Stern, 2018) removes momentum and uses much fewer adaptivity parameters, but for large models, ADAFACTOR still needs momentum to ensure training stability (Chowdhery et al., 2022). Our approach is also related to normalized and projected gradient descent, which has been studied for quasi-convex and non-convex settings (e.g. see Hazan et al. (2015); Levy (2016); Huang et al. (2017)). However, these methods have seen very limited success.

Clipping based optimization methods, especially gradient clipping, are widely used in deep learning applications to improve training stability or ensure privacy Pascanu et al. (2013); Chen et al. (2020); Zhang et al. (2020a). These approaches typically use a constant threshold to clip the gradients before the update. However, choosing this threshold is difficult and requires careful tuning. Adaptive variants of clipping methods partially alleviate this issue and are closely related to adaptive methods Zhang et al. (2020b); however, they again incur additional computation and memory costs.

Scale Invariance in deep networks. Various normalization schemes are the main source of scale invariance in deep learning, e.g. , BatchNorm Ioffe &amp; Szegedy (2015), LayerNorm Ba et al. (2016), Weight Normalization Salimans &amp; Kingma (2016), GroupNorm Wu &amp; He (2018), InstanceNorm Ulyanov et al. (2016). Scale invariance from normalization allows GD and SGD to converge to stationary points from any initialization and with any learning rate, in O ( T -1 / 2 ) and ˜ O ( T -1 / 4 ) rates respectively Arora et al. (2018). The interplay between SGD, scale invariance and WD has also been well studied. It was shown that the effect of WD for normalized networks can be replaced by LR schedules Hoffer et al. (2018); Zhang et al. (2018). Li &amp; Arora (2019) formally builds the equivalence between SGD+WD and SGD with an exponential increasing LR schedule for scale invariant loss. Van Laarhoven (2017) first proposed the notion of effective LR, η/ ‖ x ‖ 2 2 , for normalized networks, and showed that the unique stationary value of ‖ x ‖ 4 2 is proportional to λ/η , where η is LR and λ is WD. Li et al. (2020) proved that the parameter norm always converges to the above value by modeling SGD as Stochastic Differential Equation. Wan et al. (2020) proved the parameter norm converges to the same value directly for SGD+WD, but only in expectation.

## 2.1 Preliminary

In this section we present the definition of scale invariant functions and some of their useful properties. For x ∈ R d , we define x := x ‖ x ‖ 2 . We say a function is C k iff it is k -times continuously differentiable.

Definition 2.1. Given a cone U ⊂ R d , we say a function f : U → R is (positively) k -homogeneous or of homogeneity of degree k iff for any c &gt; 0 and x ∈ U , f ( c x ) = c k f ( x ) . We say a function is scale invariant iff it is 0 -homogeneous.

Now we present some useful properties of the derivatives of homogeneous functions.

Theorem 2.2 (Euler's Homogeneous Function Theorem) . For any k -homogeneous C 1 function f , it holds that 〈∇ f ( x ) , x 〉 = kf ( x ) .

- Lemma 2.3. For any k -homogeneous C l function f , ∇ l f is k -l homogeneous.

Lemma 2.4 (Equivalent Scaling) . The properties below hold (and generalize to stochastic loss):

1. For any loss L , LR η , WD λ and initialization x (0) , rescaling ( L, η, λ, x (0)) → ( cL, η/c, cλ, x (0)) doesn't change GD iterate x ( t ) for any t ≥ 0 .
2. For any scale invariant loss L , LR η , WD λ and initialization x (0) , rescaling ( L, η, λ, x (0)) → ( L, c 2 η, λ/c 2 , c x (0)) doesn't change the direction of GD iterate x ( t ) for any t ≥ 0 . (see Lemma 2.4 in Li &amp; Arora (2019))

## 3 Methods

In this section, we provide a more detailed description of our recipe for robust and memory-efficient network training, which includes three building blocks: (1) scale invariant architecture (Section 3.1), (2) SGD with Weight Decay (Section 3.2) and optionally (3) the Relative Global Clipping (Section 3.3 and Algorithm 1).

## Algorithm 1 √ C -Clipped SGD + WD

Input: Total steps T , Scale invariant loss { L t } T t ≥ 1 , initialization x (0) , LR η ,WD λ , clipping factor C &gt; 1 ( C = ∞⇔ no clipping).

<!-- formula-not-decoded -->

end for

## 3.1 Designing Scaling Invariant Architectures

We first revisit an approach for introducing scale invariance in neural networks, which is presented in Li &amp; Arora (2019). Viewing the neural network computation as a directed graph, the high level idea is to ensure same homogeneity degree of different edges reaching a node. For example in a RESNET block, the output from an affine transform is added back to the input z from the previous layer yielding z + Aff ( z ) . Now if we scale all the network parameters by c , both z and Aff ( z ) should have the same degree of homogeneity and scale as c k . Otherwise the network is no longer homogeneous and, hence, cannot be scale invariant.

In this paper, we apply the above design philosophy to develop a scale invariant version of BERT (Devlin et al., 2018) - a transformer based model. A transformer has two main building blocks that need to be made scale invariant - residual block and Attention Vaswani et al. (2017). For residual block, Li &amp; Arora (2019) already demonstrated how to make both the PreNorm and PostNorm version of RESNET scale invariant (see Appendix of their paper for more details). In this paper, we use their PreNorm variant (see Figure 5). Furthermore, we design a novel scale invariant version of Attention block in transformer, as described below.

Scale Invariant Attention: Recall the standard self attention block computes the following for a given input Q,K,V ∈ R n × d model :

<!-- formula-not-decoded -->

Here W Q , W K ∈ R d model × d k and W V ∈ R d model × d v are affine transformations and, hence, are all 1-homogeneous transformations. The Softmax function computes row wise softmax normalization. It is easy to see that standard attention is not homogeneous as softmax is itself not homogeneous.

We design a novel Scale Invariant Attention (SI Attention) in the following way: (also see Figure 7)

<!-- formula-not-decoded -->

where N denotes the row-wise normalization by sum, i.e. , [N( A )] ij = a ij ∑ j a ij and ReLU( A ) denote the element-wise max between matrix A and 0 . Notably we replace the softmax with a ReLU activation followed by normalization. Both ReLU and normalization are homogeneous operations; thus, making the overall attention score computation ( N(ReLU( ZQK glyph[latticetop] Z glyph[latticetop] )) ) scale invariant to the concatenation of all parameters x , assuming Q,K,V are already positive homogeneous to x . Due to space constraints, the full design of Scale Invariant BERT (SIBERT) is relegated to Appendix A.

## 3.2 Training Algorithm: SGD + WD

Although scale invariance can prevent parameter divergence after a large gradient update by eliminating the positive feedback between gradient and parameter norm, it alone does not ensure SGD trains the network in a robust and efficient way. This is because, as shown in Arora et al. (2018), the parameter norm monotonically increases when SGD is used to optimize a scale invariant loss. As a result, once the norm becomes too large (e.g due to large gradient in some step) the training can slow down drastically as the effective LR η ‖ x t ‖ 2 2 is too small; thus, preventing effective recovery from even minor training instabilities.

To tackle this issue we propose to use Weight Decay(WD) as a way to reduce the parameter norm; thereby, allowing the network to recover from slow training induced by infrequent updates of large norm. Under mild assumptions that the expectation of squared norm of stochastic gradient does not vary too much on the unit sphere, Li et al. (2020); Wan et al. (2020) show that the parameter norm will stabilize in O ( 1 ηλ ) steps and the learning dynamics is equivalent to one on unit sphere with effective learning rate proportional to Θ( √ λη ) .

Leveraging the advantage of quick norm convergence, we show that the convergence of SGD+WD is insensitive to the following three operations: loss rescaling (A1), initialization rescaling (A2) and re-parametrization (A3), meaning the same convergence rate (independent of scaling c ) can be achieved, in up to | log c | λη more steps. (See formal statement in Theorems 4.1 and 4.5 This property reduces the effort of hyperparameter tuning and also makes training more robust when switching between different codebases and frameworks, which is likely to have different default scaling or parametrization. Also note by scale invariance of loss L , (A2) is equivalent to (A3).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As a comparison, previous work Arora et al. (2018) showed that GD converges to glyph[epsilon1] approximate stationary point of a scale invariant loss in O ( 1 glyph[epsilon1] 2 ) and SGD converges in ˜ O (1 /glyph[epsilon1] 4 ) steps with any initialization. However, the constant in O ( · ) scales linearly or inversely to the above scalings ( c in (A1-3)). This is far from satisfying, and indeed their experiments show that either large or small LR could substantially slowdown the training progress.

## 3.3 Relative Global Clipping

Gradient clipping is a widely used effective strategy to stabilize neural network training. However, often the clipping threshold need to be tuned based on the optimization problem and the specific gradient distribution. Furthermore, simply using a constant threshold can severely degrade the performance (Zhang et al., 2020b). Thus, it is unclear how the clipping threshold needs to be set for SGD+WD on scale invariant functions such that it is insensitive to rescaling of loss and reparametrization, e.g. , (A1-3).

To this end, we propose a clipping strategy named Relative Global Clipping which allows consistent and robust training behavior for SGD+WD on scale invariant loss under the aforementioned operations. In particular, we propose to set the clipping threshold as √ 2 Cλ η ‖ x ‖ 2 , where C ≥ 1 is a hyperparamer with default value √ C = 2 . The high level design idea is that (1) the clipping rule should be invariant to the scalings ( L, η, λ ) → ( cL, η/c, cλ ) and ( x , η, λ ) → ( c x , c 2 η, λ/c 2 ) for any c &gt; 0 , to which SGD+WD is invariant (see Lemma 2.4); (2) the clipping rule should only remove the extremely large gradients and should not trigger too often to ensure that gradient after clipping remains almost unbiased.

Intuitively, the derivation of Relative Global Clipping involves the following line of reasoning: Suppose the norm of the stochastic gradient ‖∇ L γ ( x ) ‖ 2 is constant, say σ , for all data and every parameter x on the unit sphere. In this case, we expect our clipping strategy to not be triggered since there are no extremely high stochastic gradients. Since L γ is scale invariant, Theorem 2.2 implies that 〈∇ L γ ( x ) , x 〉 = 0 . That is,

<!-- formula-not-decoded -->

It is not difficult to show the iteration (1) has a unique stationary point, ‖ x ( t ) ‖ 2 2 = √ 2 η λ (2 -ηλ ) σ (Van Laarhoven, 2017). In other words, at norm equilibrium, it holds

<!-- formula-not-decoded -->

The above calculation suggests the clipping threshold should be at least √ 2 λ η ‖ x ( t ) ‖ 2 . 1 Furthermore, it is not difficult to check that the clipping threshold √ 2 λ η ‖ x ( t ) ‖ 2 is indeed invariant to the above mentioned scalings ( L, η, λ ) → ( cL, η/c, cλ ) and ( x , η, λ ) → ( c x , c 2 η, λ/c 2 ) . For each hyperparameter C &gt; 1 , the behavior of SGD+WD is consistent for different scalings (A1-3) and it also improves the norm convergence (reducing undesirable spikes in norm while training) for SGD+WD (see Theorem 4.8). Under mild assumptions that such clipping does not introduce too much bias in gradients, we show that our recipe enables convergence to approximate stationary points. Furthermore, the rate only depends logarithmically on the initialization and loss scale, as shown in the following section.

## 4 Theoretical Analysis

In this section, we provide theoretical analysis of the convergence of SGD+WD to approximate first order stationary points for scale invariant functions. We first start with the key highlights of our theoretical analysis for SGD+WD:

1 We drop -ηλ for convenience. This doesn't lead to any practical difference as ηλ is typically very small, e.g. less than 10 -4 .

1. Parameter norm converges to Θ(( λ η ) 1 4 ) in T 1 = ˜ O ( 1 ηλ ) steps with high probability where T 1 is a function of loss L , initial norm ‖ x (0) ‖ 2 , LR η and WD λ . Moreover, T 1 ( L, ‖ x (0) ‖ 2 , η, λ ) changes most by ln | c | ηλ for operation (A1-3).
2. After step T 1 , convergence to first order approximate stationary point happens and the rate only depends on ηλ and is unaffected by operations (A1-3).

Properties (1) and (2) suggest our results are more robust to initialization scale (by only having logarithmic dependence on it), showing the advantage of using scale invariant functions while matching the standard convergence rates for non-convex functions. Note that the standard notion of approximate stationary point, i.e. x with small gradient norm of ‖∇ L ( x ) ‖ 2 is not useful for scale invariant loss, as one can simply scale up the initialization x (0) to infinity and the gradient norm thus scales inversely. A more reasonable notion of 'stationary point' is that the direction of x , denoted by x := x ‖ x ‖ 2 , has small gradient norm, as first introduced in Arora et al. (2018). We will use this definition of approximate stationary point throughout the paper. In the section we also assume L is a C 2 and scale invariant function and ρ := max ‖ x ‖ =1 ∥ ∥ ∇ 2 L ( x ) ∥ ∥ .

## 4.1 Convergence of GD +WD

We first present the convergence result in the deterministic case, i.e. , Gradient Descent over L ( x ) + λ 2 ‖ x ‖ 2 2 .

<!-- formula-not-decoded -->

Theorem 4.1 (GD+WD) . For ηλ ≤ 1 2 , let x ( t ) be defined by GD (3) , and T 0 = ⌈ 1 2 ηλ (∣ ∣ ∣ ln ‖ x (0) ‖ 2 2 ρπ 2 η ∣ ∣ ∣ +3 )⌉ . We have

<!-- formula-not-decoded -->

This bound matches the standard O ( 1 √ T ) convergence rate to first order stationary point for non-convex functions. Remarkably, for a given training budget T , once we can set ηλ to be D T where D is a constant ( e.g. 10), the convergence becomes robust to the choice the hyperparameters due to just a logarithmic dependence on them. In particular, GD+WD can work with any scaling of L (which affects the smoothness on unit sphere, ρ ), LR η and initial norm ‖ x (0) ‖ 2 , as long as ‖ x (0) ‖ 2 2 ρπ 2 η ∈ [ e -D , e D ] . This is in sharp contrast to GD on standard loss as it requires knowledge about the smoothness to set the optimal LR.

However, one weakness of the above result is that with a fixed ηλ , longer training does not guarantee further convergence. The intuition is that once the iterate converge in direction and the gradient vanishes, Weight Decay will dominate the dynamics and thus the norm approaches 0 , which increases the sharpness. When the sharpness gets larger than 2 /η , the dynamics become unstable and results in divergence. This phenomena is first observed in Li et al. (2020) and verified by Lobacheva et al. (2021) in practical settings. This behavior can also be viewed as a special case of Edge of Stability as described in Cohen et al. (2020).

Proof Sketch of Theorem 4.1. Scale invariant functions do not have bounded smoothness at 0 making it a challenge to use standard convergence analysis. Our key insight is that for scale invariant loss function, even with a fixed LR η , GD can tune its effective LR η ‖ x ( t ) ‖ 2 2 by changing the norm. Thus once GD passes the area of the suitable norm, the smoothness of scale invariant loss function is upper bounded by ρ r 2 outside the ball with radius r centered at 0 .

More concretely our proof consists of 2 steps. In the first step we show that GD+WD iterates pass an area of suitable norm ( ≈ √ ρη ). For large initial norm, WD could bring the norm to correct scaling in log time and then converge (Theorem D.2). If the initial norm is too small and the direction is not approximately stationary, then the large gradient due to the small norm will increase the parameter norm drastically in a single step (Lemma D.1), and again Weight Decay can bring the norm down in log steps. In the second step we show that, once the norm reaches this suitable value, the descent lemma (Lemma 4.2) starts to hold and the convergence analysis is standard.

Lemma 4.2. Let x ( t ) , x ( t +1) be defined as (3) , we have

<!-- formula-not-decoded -->

When ηλ ≤ 1 2 , the above can be simplified into

<!-- formula-not-decoded -->

Remark 4.3 . One might wonder why the upper bounds on loss and gradient norm do not appear in Theorem 4.1. This is because we are working on a compact domain (the unit sphere) and twice-differentiability implies those bounds implicitly. (See Lemmas C.3 and C.4)

## 4.2 Convergence of SGD+WD

Below we present our convergence analysis for SGD+WD.

Setting: Let Γ be an index set and L γ : R d / { 0 } → R be a scale invariant loss function for each γ ∈ Γ . We denote E γ L γ by L . We assume the largest possible stochastic gradient norm is finite, i.e. , M := sup γ ∈ Γ max ‖ x ‖ =1 ‖∇ L γ ( x ) ‖ . SGD is defined as (5).

<!-- formula-not-decoded -->

where γ t ∈ Γ are i.i.d. random variables. We further assume there exists constants σ and σ , such that σ 2 ≤ E ‖∇ L γ ( x ) ‖ 2 2 ≤ σ 2 , for any ‖ x ‖ 2 = 1 . We finally need the following condition on ηλ to bound convergence.

<!-- formula-not-decoded -->

The Condition 4.4 is useful for proving norm convergence in high probability. In practice, typically ηλ is very small. Our experiments use η = 0 . 0008 and λ = 0 . 01 . Hence e 4 ηλ ≈ 1 , and Condition 4.4 essentially requires the gradient norm square cannot exceed its average multiplied by 1 / √ ηλ ≈ 350 , which is reasonable for most iterates.

Theorem 4.5 (SGD+WD) . Let x ( t ) be defined by SGD (5) . For ηλ ≤ 0 . 1 , under Condition 4.4, with probability 1 -5 δ ,

<!-- formula-not-decoded -->

<!-- image -->

⋃√˜√√

˜

⋃√˜√√

˜

⋃∮〉∫⋂/∖̂⌈]√√˜̂

⋃√˜√√

˜

Figure 1: SGD+WD optimizes the scale invariant training loss of SIBERT robustly for all initialization scales, and thus for loss scalings and different learning rates (with λη fi xed). Here the default initialization for parameters in SIBERT encoder is a truncated normal distribution with standard deviation equal to 0 . 02 (the same as BERT).

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The proof of this theorem is presented in Appendix E. Similar to our earlier result for GD this bound matches the standard O ( T -1 / 4 ) convergence rate of SGD for non-convex functions by setting T = ˜ O ( 1 ηλ ) . Further, it only has a logarithmic dependence on the initialization scale ‖ x (0) ‖ 2 , and enjoys robustness to initialization scale as discussed earlier for GD. We further extend this result to the case where the scale invariant loss has multiple scale invariant parameter groups in Appendix G.

We next present our analysis for SGD with clipping.

## 4.3 Convergence of SGD with Relative Global Clipping

Now we will present our analysis for the clipped SGD. Recall the clipped SGD update from Algorithm 1 has the following norm dynamics.

Norm dynamics of clipped SGD:

<!-- formula-not-decoded -->

To present our bound we need the following definitions.

Definition 4.6 ( C -clipped mean) . Given a distribution P on R ≥ 0 and constant C &gt; 1 , we define F P,C ( µ ) = E t ∼ P [min { t, Cµ } ] , and define the C -clipped mean of P , µ P,C as the largest positive real number satisfying that F P,C ( Cµ P,C ) = µ P,C . Such a definition is valid because F P,C (0) = 0 and thus 0 is always a solution.

For convenience, we also define G P,C ( µ ) := F P,C ( Cµ ) -µ and M P, 1 C is defined as the 1 C median of P , that is, M P,C := sup { M ≥ 0 | P t ∼ P [ t ≥ M ] ≥ 1 C } . Since the cumulative density function P t ∼ P [ t ≥ M ] is left continuous in M , it holds that P t ∼ P [ t ≥ M P,C ] ≥ 1 C .

Let P x denote the distribution of ‖∇ L γ ( x ) ‖ 2 2 . Below is a mild assumption saying P x is universally well-concentrated from below in the sense that the mean of the smallest (1 -1 C ) part of P x is at least a constant fraction of the C -clipped mean of P x . Since µ P x ,C ≤ µ x , the assumption below holds whenever α C µ x ≤ E t ∼ P x [ t 1 [ t &lt; M P x , 1 C ]] .

glyph[negationslash]

<!-- formula-not-decoded -->

We further define µ C := min ‖ x ‖ 2 =1 µ P x ,C and µ C := max ‖ x ‖ 2 =1 µ P x ,C and have the following theorem:

Theorem 4.8 ( √ C -Clipped SGD+WD) . Let x ( t ) be defined by √ C -Clipped SGD +WD (Algorithm 1). Under Assumption 4.7, for ηλ = O (min { 1 , α C C ln T/δ 2 } ) , with probability 1 -5 δ , we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where T ′ = 1 α C ηλ max { ln R 2 0 µ C , ln µ C R 2 0 } + O (1) and ˜ ∇ L ( x ) := E [ ∇ L γ ( x ) min {√ 2 Cλ η ‖ x ‖ 2 2 ‖∇ L γ ( x ) ‖ 2 , 1 }] .

The proof of this theorem is presented in Appendix F. Note that with clipping Theorem 4.8 shows that the norm convergence (8) is more robust as it doesn't need to make any assumption about the maximum gradient norm M , unlike Theorem 4.5. Indeed, from the definition of C -clipped mean, for each x , we can allow all the gradients with norm larger than C · µ P x ,C to become infinity, and yet not affect the norm convergence, as µ P x ,C and the condition in Assumption 4.7 do not change.

Under the additional assumption that 〈 ∇ L ( x ( t )) , ˜ ∇ L ( x ( t ) 〉 = Ω( ‖∇ L ( x ( t )) ‖ 2 2 ) , we can use Equation (9) to show convergence to stationary points. This is a reasonable assumption if the clipping frequency is low, e.g. , it's 1 . 5% in our experiments for SIBERT.

## 5 Experiments

We now conduct a comprehensive empirical study in order to demonstrate the following key aspects of our recipe: (i) yields competitive training performance using significantly low memory footprint, (ii) training becomes highly robust to initialization scale, and (iii) provides better convergence of norm with clipping.

Experimental Setup. We consider the standard task of pretraining a transformer model and finetuning it on benchmark datasets, following Devlin et al. (2018). We compare its performance with SIBERT, a scale invariant version of BERT as described in Sec. 3.1. For both these models, we use their base size versions unless specified otherwise. For SIBERT, the scale invariant portion is trained using SGD+WD with a piecewise constant LR schedule and WD of 1 e -2 . We use LAMB optimizer for the non-scale invariant parts. The initial LR for SGD is 8 e -4 without warmup and is divided by 10 at step 600 k and 900 k. Default training is for 1 M steps. For LAMB we use a linear decay schedule with initial learning rate 8 e -4 and a linear warmup of 10 k steps.

Figure 2: The robust optimization performance of SGD+WD over the scale invariant training loss of SIBERT originates from its ability to fast adjust the parameter norm. In contrast, when the initial norm is too large, SGD w.o. WD optimizes slowly. Relative Global Clipping reduces the spikes in the norm curve, which verifies our theoretical result Theorem 4.8 that clipping leads to better norm convergence. Here, only the norm of the scale invariant part, i.e. , the encoder part is plotted.

<!-- image -->

Figure 3: Our recipe (SIBERT, SGD+WD and Relative Global Clipping) significantly improves the optimization performance compared to the baseline, BERT trained by SGD with small LR. The final training loss is close to BERT trained by ADAM.

<!-- image -->

Performance. We begin by establishing that proposed SIBERT with SGD+WD training performs competitively. In this regard, we first look at pretraining loss between standard training of BERT with ADAM and our SIBERT trained by SGD+WD with or without clipping (the clipping factor is set as √ C = 2 ). From Figure 3, one can see that our training curve closely follows that of BERT trained by ADAM, but without the need for extra memory for keeping track of first and second order momentum. If we use SGD on standard BERT architecture, then either we have to use small learning rates, which slows down training, or the loss diverges. This further highlights the importance of the scale invariant architecture, which improves training stability by eliminating the k -homogeneous structure. To our knowledge, this is the first work that shows effective training of BERT-like model using simple SGD (even without any momentum).

Next, we compare the downstream performance on three benchmark datasets (SQuADv1.1 (Rajpurkar et al., 2016), SQuADv2 (Rajpurkar et al., 2018) and MNLI (Williams et al., 2018)). We tried to follow standard setup, e.g. BERT is finetuned by ADAM. However for SIBERT we had to use LAMB, as ADAM is very sensitive to the scale. We observe comparable performance and when trained longer it can even outperform conventional BERT.

Table 1: Downstream Performance of SIBERT trained by SGD+WD +clipping is close to that of BERT trained ADAM- which uses 3 X more memory than SGD. The gap is further reduced by doubling the training budget of SIBERT.

|       |               |   MNLI Acc |   SQuAD1 F1 |   SQuAD2 F1 |   Pretraining Loss |
|-------|---------------|------------|-------------|-------------|--------------------|
|       | BERT          |       84.4 |        90.3 |        78.8 |              1.479 |
| Base  | SIBERT        |       81.1 |        88.1 |        74.8 |              1.672 |
| Base  | + clipping    |       82.6 |        89.3 |        76.8 |               1.58 |
| Base  | + 2x training |       83.3 |        90.3 |        80.0 |              1.495 |
|       | BERT          |       86.8 |        92.4 |        84.1 |              1.181 |
| Large | SIBERT        |       83.7 |        90.6 |        79.3 |              1.404 |
| Large | + clipping    |       85.3 |        91.6 |        81.3 |              1.322 |
|       | + 2x training |       86.4 |        92.4 |        83.1 |              1.194 |

Training Stability: Insensitivity to the scale of initialization. To showcase ease of optimization offered by our recipe, we consider different initialization scales spanning two orders of magnitude. The results for the pretraining task in Figure 1 show good convergence across the board for our approach, whereas SGD on its own struggles even with the scale invariant architecture.

Further note that these experiments simultaneously showcase robustness to rescaling of loss, parameterization, or LR. This is because in a scale invariant model trained by SGD+WD (+clipping), it holds that all of following scalings are equivalent: ( c 1 L, c 2 x (0) , c 3 η, c 4 λ ) ←→ ( L, c 2 √ c 1 c 3 x (0) , η, c 3 c 4 λ ) for any c 1 , c 2 , c 3 , c 4 &gt; 0 .

Training Stability: Improvement in parameter norm convergence. Finally, we look at parameter norms during training in experiments. We observe that even when starting from very different initialization scale, SGD+WD (+clipping) quickly brings parameter norm to desired ranges. In contrast, SGD struggles when initial norm and learning rate are not aligned - see the rightmost plot with large initialization in Figure 2. This shows that our recipe has the ability to quickly adapt to different initialization scales, in-line with our theoretical result (Theorem 4.8) showing better norm convergence of SGD+WD (+clipping).

## 6 Conclusion

In this paper, we presented a simple yet effective method to robustly train transformers with nonadaptive methods such as SGD. By designing novel scale invariant architecture and using a tailored optimization procedure - which makes our optimization scheme truly architecture aware -we provably achieve robust training of neural networks with substantially low memory footprint when compared to adaptive methods. We believe designing neural architecture and the optimizer jointly is an exciting research direction and will yield even better training procedures in the future.

## References

- Anil, R., Gupta, V., Koren, T., and Singer, Y. Memory efficient adaptive optimization. Advances in Neural Information Processing Systems , 32, 2019.
- Arora, S., Li, Z., and Lyu, K. Theoretical analysis of auto rate-tuning by batch normalization. In International Conference on Learning Representations , 2018.
- Ba, J. L., Kiros, J. R., and Hinton, G. E. Layer normalization. arXiv preprint arXiv:1607.06450 , 2016.
- Chen, X., Wu, Z. S., and Hong, M. Understanding gradient clipping in private SGD: A geometric perspective. CoRR , abs/2006.15429, 2020. URL https://arxiv.org/abs/2006.15429 .
- Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H. W., Sutton, C., Gehrmann, S., et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311 , 2022.
- Cohen, J., Kaur, S., Li, Y., Kolter, J. Z., and Talwalkar, A. Gradient descent on neural networks typically occurs at the edge of stability. In International Conference on Learning Representations , 2020.
- Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 , 2018.
- Duchi, J., Hazan, E., and Singer, Y. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research , 12(Jul):2121-2159, 2011.
- Hazan, E., Levy, K., and Shalev-Shwartz, S. Beyond convexity: Stochastic quasi-convex optimization. In Advances in Neural Information Processing Systems , pp. 1594-1602, 2015.
- Hendrycks, D. and Gimpel, K. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415 , 2016.
- Hoffer, E., Banner, R., Golan, I., and Soudry, D. Norm matters: efficient and accurate normalization schemes in deep networks. arXiv preprint arXiv:1803.01814 , 2018.
- Huang, L., Liu, X., Lang, B., and Li, B. Projection based weight normalization for deep neural networks. ArXiv , abs/1710.02338, 2017.
- Ioffe, S. and Szegedy, C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning , pp. 448-456. PMLR, 2015.
- Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings , 2015.
- Levy, K. Y. The power of normalization: Faster evasion of saddle points. arXiv preprint arXiv:1611.04831 , 2016.
- Lewkowycz, A., Bahri, Y., Dyer, E., Sohl-Dickstein, J., and Gur-Ari, G. The large learning rate phase of deep learning: the catapult mechanism. arXiv preprint arXiv:2003.02218 , 2020.

- Li, Z. and Arora, S. An exponential learning rate schedule for deep learning. In International Conference on Learning Representations , 2019.
- Li, Z., Lyu, K., and Arora, S. Reconciling modern deep learning with traditional optimization analyses: The intrinsic learning rate. Advances in Neural Information Processing Systems , 33, 2020.
- Liu, L., Liu, X., Gao, J., Chen, W., and Han, J. Understanding the difficulty of training transformers. In Webber, B., Cohn, T., He, Y., and Liu, Y. (eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020 , pp. 5747-5763. Association for Computational Linguistics, 2020.
- Lobacheva, E., Kodryan, M., Chirkova, N., Malinin, A., and Vetrov, D. P. On the periodic behavior of neural network training with batch normalization and weight decay. Advances in Neural Information Processing Systems , 34, 2021.
- Pascanu, R., Mikolov, T., and Bengio, Y. On the difficulty of training recurrent neural networks. In Dasgupta, S. and McAllester, D. (eds.), Proceedings of the 30th International Conference on Machine Learning , volume 28 of Proceedings of Machine Learning Research , pp. 13101318, Atlanta, Georgia, USA, 17-19 Jun 2013. PMLR. URL https://proceedings.mlr. press/v28/pascanu13.html .
- Polyak, B. T. Introduction to optimization. optimization software. Inc., Publications Division, New York , 1, 1987.
- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683 , 2019.
- Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. Squad: 100,000+ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250 , 2016.
- Rajpurkar, P., Jia, R., and Liang, P. Know what you don't know: Unanswerable questions for squad. arXiv preprint arXiv:1806.03822 , 2018.
- Reddi, S. J., Kale, S., and Kumar, S. On the convergence of ADAM and beyond. arXiv preprint arXiv:1904.09237 , 2019.
- Salimans, T. and Kingma, D. P. Weight normalization: A simple reparameterization to accelerate training of deep neural networks. Advances in neural information processing systems , 29:901-909, 2016.
- Shazeer, N. and Stern, M. Adafactor: Adaptive learning rates with sublinear memory cost. In International Conference on Machine Learning , pp. 4596-4604. PMLR, 2018.
- Tieleman, T. and Hinton, G. Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning , 4(2):26-31, 2012.
- Ulyanov, D., Vedaldi, A., and Lempitsky, V. Instance normalization: The missing ingredient for fast stylization. arXiv preprint arXiv:1607.08022 , 2016.
- van Handel, R. Probability in high dimension. 2016.

- Van Laarhoven, T. L2 regularization versus batch and weight normalization. arXiv preprint arXiv:1706.05350 , 2017.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In Advances in neural information processing systems , pp. 5998-6008, 2017.
- Wan, R., Zhu, Z., Zhang, X., and Sun, J. Spherical motion dynamics: Learning dynamics of neural network with normalization, weight decay, and sgd. arXiv preprint arXiv:2006.08419 , 2020.
- Williams, A., Nangia, N., and Bowman, S. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers) , pp. 1112-1122. Association for Computational Linguistics, 2018.
- Wu, Y. and He, K. Group normalization. In Proceedings of the European conference on computer vision (ECCV) , pp. 3-19, 2018.
- You, Y., Li, J., Reddi, S. J., Hseu, J., Kumar, S., Bhojanapalli, S., Song, X., Demmel, J., Keutzer, K., and Hsieh, C. Large batch optimization for deep learning: Training BERT in 76 minutes. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net, 2020.
- Zhang, G., Wang, C., Xu, B., and Grosse, R. Three mechanisms of weight decay regularization. In International Conference on Learning Representations , 2018.
- Zhang, J., He, T., Sra, S., and Jadbabaie, A. Why gradient clipping accelerates training: A theoretical justification for adaptivity. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020 . OpenReview.net, 2020a.
- Zhang, J., Karimireddy, S. P., Veit, A., Kim, S., Reddi, S. J., Kumar, S., and Sra, S. Why are adaptive methods good for attention models? In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual , 2020b.

## A Design Details of Scale Invariant BERT

Definition A.1. For a module with n inputs and m outputs, we say the module is ( a 1 , ...a n ; b 1 , ..., b m ) -homogeneous if the m outputs are b i -homogeneous to the network parameters whenever the n inputs are a i -homogeneous to the network parameters. A model is scale invariant iff its output is (; 0) -homogeneous. (A complete model doesn't take any input from another module)

Following Li &amp; Arora (2019), we view the computation graph as a directed acyclic graph, where each module is a node and each tensor (including inputs, intermediate computation results and final output) as an edge. Each edge can be viewed as a function of parameters, and we can decide the homogeneity by doing induction over the computation graph by its topological order. In detail, we know the j th output edge of some ( a 1 , . . . , a n ; b 1 , ldots, b n ) - homogeneous module is b j homogeneous if for each 1 ≤ i ≤ n , the i th input edge is a i -homogeneous. For convenience, we allow a i , b i to be functions of free variable x , meaning the module is ( a 1( x ) , . . . , a n ( x ); b 1 ( x ) , . . . , b m ( x )) -homogeneous for every x ∈ R .

In Table 2, we summarize the homogeneity of building blocks in our design.

Overview of SIBERT structure: Our SIBERT has two main parts - encoder and classification head, which is the same to standard BERT. We only make encoder part scale invariant and train it by SGD+WD. We leave the classification head not scale invariant and train it by LAMB. Note the classification head is only used in pretraining and is not used in the downstream task.

(2;2)-homogeneous encoder layer: As mentioned in Appendix A, residual block and attention are the two main building blocks that needs to be made scale invariant. Following Li &amp; Arora (2019), we choose to use PreNorm structure for residual block and make it (2; 2) -homogeneous. We also replace GeLU Hendrycks &amp; Gimpel (2016) in BERT by ReLU for homogeneity. Since ReLU is (1; 1) homogeneous, we omit ReLU from the design, without affecting the final scale invariance.

Table 2: Homogeneity of building blocks of SIBERT.

| Symbol   | Module                          | Homogeneity   |
|----------|---------------------------------|---------------|
| I        | Input                           | (0;1)         |
| B        | Adding Bias                     | (1;1)         |
| N        | Layer Normalization (no affine) | (x;0)         |
| L        | Linear Layer                    | (x;x+1)       |
| Embed    | Embedding Layer                 | (x;x+1)       |
| NA       | Layer Normalization with affine | (x;1)         |
| FF       | 2-layer feedforward network     | (0;2)         |
| ATTN     | Scale Invariant Attention       | (x,x,x;x+2)   |
| Encoder  | Our Encoder Layer               | (2;2)         |

<!-- image -->

Figure 4: Encoder and Classification Head (CLS). 'x12/24' means to stack 12 our (2; 2) -homogeneous encoder layer for base SIBERT (or 24 for large SIBERT)

<!-- image -->

Figure 5: The (2; 2) -homogeneous encoder layer. 'ATTN' denotes our Scale Invariant Attention (see Figure 7). 'FF' denotes the 2-layer feedforward structure, which is (0; 2) -homogeneous.

Figure 6: The (0; 2) -homogeneous FeedForward layer

<!-- image -->

Figure 7: The ( x, x, x ; x + 2) -homogeneous Attention, which is defined as Multi-Head-SI-Attention ( Q,K,V ) = ∑ i N(ReLU( QW Q i ( KW K i ) glyph[latticetop] ) V W V i W O i , where W Q i , W K i ∈ R d model × d k , W V i ∈ R d k × d v and W O i ∈ R d v × d model That is, if Q,K,V are k -homogeneous functions of parameter x , then Multi-Head-SI-Attention ( Q,K,V ) is k +2 -homogeneous, for any k ∈ R . We also call it Scale Invariant Attention because its attention score is scale invariant.

<!-- image -->

## B Introduction examples analysis

In the first example, since the data is non-separable, the global optimum X ∗ must be finite and, thus, |∇ ˜ L ( X ) | is positive and monotone increases among all X &gt; X ∗ &gt; 0 . For simplicity, assume X ∗ &gt; 0 and x 1 = · · · = x 2 k &gt; ( X ∗ ) 1 2 k at initialization (and thus at any iteration t ). It holds that

<!-- formula-not-decoded -->

This implies X ( t + 1) = X ( t ) ( 1 -η X ( t ) k √ X ( t ) ∇ ˜ L ( X ( t )) ) 2 k ≥ 0 . Thus we conclude if η ≥ 2 |∇ ˜ L ( X (0)) | ( X (0)) 1 k -1 and X (0) &gt; X ∗ , X ( t ) will increase monotonically and explode.

## C Useful Lemmas

## C.1 Scale Invariance

Lemma C.1 (Smoothness) . For any v , x ∈ R d with 〈 x , v 〉 = 0 , suppose L is scale-invariant and twice differentiable with ρ := max ‖ x ‖ 2 =1 ∥ ∥ ∇ 2 L ( x ) ∥ ∥ , we have

<!-- formula-not-decoded -->

Proof of Lemma C.1. Define γ ( s ) = x + s v , then we have L ( γ (0)) = L ( x ) and L ( γ (1)) = L ( x + v ) . Taking Taylor expansion of F ( s ) = L ( γ ( s )) at s = 0 , we have

<!-- formula-not-decoded -->

Note F ′ (0) = 〈 γ ′ (0) , ∇ L ( γ (0)) 〉 = 〈∇ L ( x ) , v 〉 and

<!-- formula-not-decoded -->

where the last inequality uses the fact that L is scale invariant. The proof is completed by noting that ‖ γ ( s ∗ ) ‖ 2 ≥ ‖ γ (0) ‖ 2 = ‖ x ‖ 2 2 and that γ ′ ( s ∗ ) = v .

Lemma C.2 (Smoothness, Multi-group) . For any v , x ∈ R d with 〈 x k , v k 〉 = 0 for all k ∈ [ K ] , suppose L is multi-group scale invariant (see Definition G.1), we have

<!-- formula-not-decoded -->

Proof of Lemma C.2. We first prove for the case where ‖ x k ‖ 2 = 1 , ∀ k ∈ [ K ] . Similar to the proof of Lemma C.1, it suffices to show that the smoothness of L is at most ρ along the line joining x and x + v . This holds because ∀ s ∈ [0 , 1] , k ∈ [ K ] , ‖ x i + s v i ‖ 2 ≥ ‖ x i ‖ 2 by assumption that 〈 x k , v k 〉 = 0 for all k ∈ [ K ] .

Nowweturn to the general case. Define ̂ x = [ x glyph[latticetop] 1 ‖ x 1 ‖ 2 , . . . , x glyph[latticetop] K ‖ x K ‖ 2 ] glyph[latticetop] and v ′ = [ v glyph[latticetop] 1 ‖ x 1 ‖ 2 , . . . , v glyph[latticetop] K ‖ x K ‖ 2 ] glyph[latticetop] . Since L is multi-group scale invariant, we have L ( x ) = L ( ̂ x ) and L ( x + v ) = L ( ̂ x + v ′ ) . The proof is completed by applying the previous argument on ̂ x and v ′ .

<!-- formula-not-decoded -->

Proof of Lemma C.3. It suffices to prove the above bound for all x with ‖ x ‖ 2 = 1 . Let x ∗ be any local minimizer of L on S d -1 and γ : [0 , 1] → S d -1 be the geodesic curve satisfying that γ (0) = x ∗ and γ (1) = x . We know the length of { γ ( t ) } 1 t =0 ≤ π and thus

<!-- formula-not-decoded -->

Lemma C.4. If L is scale invariant, sup x , x ′ L ( x ) -L ( x ′ ) ≤ π 2 2 sup ‖ x ‖ =1 ∥ ∥ ∇ 2 L ( x ) ∥ ∥ 2 . Proof of Lemma C.4. Similar to the proof of Lemma C.3.

## C.2 Probablity

Definition C.5. A random variable X ∈ R is said to be sub-Gaussian with variance proxy σ 2 (denoted by X ∼ subG ( σ 2 ) ) if its moment generating function satisfies

<!-- formula-not-decoded -->

In this work, we also use the following notion of conditional subgaussian . We say a random variable X ∈ R is said to be sub-Gaussian with variance proxy σ 2 conditioned on event E (denoted by X ∼ subG ( σ 2 , E ) ) if its moment generating function satisfies

<!-- formula-not-decoded -->

Lemma C.6 (Chernoff Bound with Conditioning) . Let X ∼ subG ( σ 2 , E ) . Then for any t &gt; 0 , it holds that

<!-- formula-not-decoded -->

When P [ E ] = 1 , we get the standard Chernoff bound. Let X ∼ subG ( σ 2 ) . Then for any t &gt; 0 , it holds that

<!-- formula-not-decoded -->

Proof of Lemma C.6. For any s &gt; 0 , we have

<!-- formula-not-decoded -->

The proof is completed by picking s = t σ 2

.

We will use (Ω , Σ , P ) to note the probability space and {F t } t ∈ N to denote the filtration.

Lemma C.7 (Azuma Inequality with Conditioning) . Let E t ∈ F t and E t +1 ⊂ E t for all t ≥ 0 . Let { X t } t ≥ 1 be a martingale difference sequence and subG ( σ 2 t , E t -1 ) conditioned on F t -1 , i.e. , E [exp( sX t ) 1 [ E t -1 ] | F t -1 ] ≤ exp( s 2 σ 2 t 2 ) for all t ≥ 0 . Then ∑ T i =1 X i is subG ( ∑ T -1 t =0 σ 2 t , E T -1 ) .

Proof. We will prove by induction on T . When T = 1 , the statement is true by assumption. Now suppose the statement holds for T -1 , we have for any s &gt; 0

<!-- formula-not-decoded -->

Thus we have that E [exp( s ∑ T X i ) 1 [ E T -1 ]] ≤ exp( s 2 ∑ T -1 t =0 σ 2 t ) .

i =1 2

## C.3 Others

Lemma C.8. ∀ t ∈ N , k ∈ N + , 0 &lt; x &lt; 1 ,

<!-- formula-not-decoded -->

Proof of Lemma C.8.

<!-- formula-not-decoded -->

where the last step is because e x ≥ 1 + x , ∀ x ∈ R .

## D Omitted Proofs for the Convergence of GD

Proof of Lemma 4.2. This is a special case of Lemma C.1 with x = (1 -ηλ ) x ( t ) and v = -η ∇ L ( x ( t )) . Here we use the assumption that L is scale invariant, ∇ L is -1 -homogeneous. By Lemma 2.3, which means ∇ L ( x ) = ∇ L ( x ( t )) 1 -ηλ .

The following lemma deals with the case where ‖ x (0) ‖ 2 2 &lt; π 2 ρη .

Lemma D.1. Let I = { T ′ ∈ N | ∀ 0 ≤ t ≤ T ′ , ‖ x ( t ) ‖ 2 2 ≤ π 2 ρη ∧ ‖∇ L ( x ( t )) ‖ 2 2 &gt; 8 π 4 ρ 2 λη } . Suppose 0 ∈ I and T = max I . Then T ≤ 1 6 λη and ‖ x ( T +1) ‖ 2 2 ≤ 2( π 2 ρη ) 2 ‖ x (0) ‖ 2 2 .

Proof of Lemma D.1. For any t ≤ T , we have

<!-- formula-not-decoded -->

Thus 6 π 2 ρλη 2 · T ≤ ‖ x ( T ) ‖ 2 2 -‖ x (0) ‖ 2 2 &lt; ‖ x ( T ) ‖ 2 2 ≤ π 2 ρη , which implies that T &lt; 1 6 λη . Moreover, we have that

<!-- formula-not-decoded -->

This completes the proof.

Theorem D.2 (convergence rate of GD+WD) . Suppose ηλ ≤ 1 2 . Let x ( t ) be the t -th iterate of GD (3) , and T 0 = ⌈ 1 2 ηλ ln 2 ‖ x (0) ‖ 2 2 ρπ 2 η ⌉ . If ‖ x (0) ‖ 2 2 ≥ π 2 ρη , we have

<!-- formula-not-decoded -->

Proof of Theorem D.2. We first claim there's 0 ≤ t ≤ T 0 , such that ‖ x ( t ) ‖ 2 2 &lt; π 2 ρη . Otherwise, by Lemma 4.2, for t = 0 , . . . , T 0 , we have L ( x ( t )) -L ( x ( t +1)) ≤ η 2 ‖∇ L ( x ( t )) ‖ 2 2 . Note that ‖ x ( t +1) ‖ 2 2 -(1 -ηλ ) 2 ‖ x ( t ) ‖ 2 2 = η 2 ‖∇ L ( x ( t )) ‖ 2 2 .

Therefore, we have that

<!-- formula-not-decoded -->

By the definition of T 0 , we have (1 -ηλ ) 2 T 0 ‖ x ( T 0 ) ‖ 2 2 ≤ e -2 ηλT 0 ‖ x (0) ‖ 2 2 ≤ ηπ 2 ρ 2 . Thus ‖ x ( T 0 ) ‖ ≤ π 2 ρη .

Without loss of generality, we let T be the smallest integer such that ‖ x ( T ) ‖ 2 2 &lt; π 2 ρη . By assumption, T ≥ 1 . Therefore ‖ x ( T -1) ‖ 2 2 ≥ π 2 ρη . Because ‖ x ( T ) ‖ 2 2 = (1 -ηλ ) 2 ‖ x ( T -1) ‖ 2 2 + η 2 ‖∇ L ( x ( T -1)) ‖ 2 2 , we have that

<!-- formula-not-decoded -->

Note that ‖ x ( T ) ‖ 2 2 &lt; π 2 ρη and ‖ x ( T ) ‖ 2 2 (1 -λη ) 2 ≥ ‖ x ( T -1) ‖ 2 2 ≥ π 2 ρη , we conclude that

<!-- formula-not-decoded -->

which completes the proof.

Combining Lemma D.1 and Theorem D.2 removes the initial condition in Theorem D.2, and completes the proof of Theorem 4.1.

## E Omitted Proofs for Convergence Rate of SGD

We will use (Ω , Σ , P ) to note the probability space and {F t } t ∈ N to denote the filtration where F t := σ ( { γ i | 0 ≤ i ≤ t } ) is the σ -algebra generated by γ 0 , . . . , γ t .

<!-- formula-not-decoded -->

Proof. Lemma E.1 Note 0 ≤ ‖∇ L γ ( x ) ‖ 2 2 ≤ M 2 ‖ x ‖ 2 2 . The proof is immediate by Hoeffding Lemma (see Lemma 3.6 in van Handel (2016)).

Given a integer T ≥ 0 , let E T be the event that ∀ 0 ≤ t ′ ≤ t ≤ T -1 ,

<!-- formula-not-decoded -->

Lemma E.2. For any 0 ≤ t ′ ≤ t ≤ T -1 ,

<!-- formula-not-decoded -->

Thus we have P [ E T ] ≥ 1 -δ by Lemma C.6.

Proof of Lemma E.2. Note that ∑ t τ = t ′ (1 -ηλ ) 8( t -τ ) M 4 4 ≤ e 8 ηλ 32 by Lemma C.8. Thus by Azuma Inequality and Lemma E.1, we have that the martingale

<!-- formula-not-decoded -->

is e 8 ηλ 32 -subgaussian.

By Lemma C.6, we have for any ∀ 0 ≤ t ′ ≤ t ≤ T -1 , Equation (10) holds with probability at least δ T 2 . The proof is completed by applying union bound.

Lemma E.3 (Norm Lower Bound) . Under Condition 4.4 and additionally assume ηλ ≤ 1 2 . On E T , it holds that for any t ≥ 0 ,

<!-- formula-not-decoded -->

When σ 2 12 ηλ ≥ M 2 2 e 4 ηλ √ 1 λη ln 2 T 2 δ , the above condition is simplified into the following: on E T for any 1 ηλ ≤ t ≤ T ,

<!-- formula-not-decoded -->

In the above inequality, we also used the fact that 1 -e -4(1 -ηλ ) ≥ 5 6 , which is implied by ηλ ≤ 0 . 5 .

Proof of Lemma E.3. Since L γ is scale invariant, by Theorem 2.2, we have

<!-- formula-not-decoded -->

Squaring both sides of Equation (13), we have

<!-- formula-not-decoded -->

Thus

<!-- formula-not-decoded -->

We also have that

<!-- formula-not-decoded -->

Therefore, it holds that for any t ≥ 0 , conditioned on E T ,

<!-- formula-not-decoded -->

This completes the proof.

Lemma E.4 (Norm upper bound) . Under Condition 4.4 and additionally assume ηλ ≤ 0 . 1 . Let T 0 = glyph[ceilingleft] 1 ηλ glyph[ceilingright] . Let t ∗ be the earliest step t in { 0 , . . . , T 0 -1 } that η -2 ‖ x ( t ) ‖ 4 2 ≥ e 8 (1 -ηλ ) 2 σ 2 4 ηλ and we denote t ∗ = T 0 if this doesn't happen in { 0 , . . . , T 0 -1 } . For the case t ∗ = T 0 , we have η -2 ‖ x ( T 0 ) ‖ 4 2 ≤ (1 -ηλ ) 2 σ 2 4 ηλ . On E T , for any t ≥ t ∗ ,

<!-- formula-not-decoded -->

Thus, there exists T 1 = T 0 + 1 4 ηλ max { ln M 2 ηλ σ 2 + ∣ ∣ ∣ ln 2 e 4 M 2 ‖ x (0) ‖ 4 2 η -2 ∣ ∣ ∣ , 4 } , such that ∀ t ≥ T 1 , η -2 ‖ x ( t +1) ‖ 4 2 ≤ 2 σ 2 ηλ .

Proof of Lemma E.4. If t ∗ &lt; T 0 , it holds that conditioned on E T , for any t ∗ ≤ t &lt; T 0 ,

<!-- formula-not-decoded -->

Therefore, for any t ≥ t ∗ , we have

<!-- formula-not-decoded -->

Below we will upper-bound the terms (A), (B) and (C) on E T respectively.

- (A). By Lemma C.8, we have

<!-- formula-not-decoded -->

where in the last step we used ηλ ≤ 0 . 1 and e x (1 -x ) ≤ 1 for any 0 ≤ x ≤ 1 .

- (B). By the definition of event E T , we have

<!-- formula-not-decoded -->

- (C). Combining the above analysis and Lemma E.3, we know conditioned on E T , for any t ≥ t ∗ , it holds ‖ x ( t ) ‖ 4 2 /η 2 ≥ (1 -ηλ ) 2 σ 2 4 ηλ .

Therefore, by Lemma C.8, we have

<!-- formula-not-decoded -->

Under Condition 4.4, we can further upper bound ( C ) by σ 2 9 ηλe 4 ηλ (1 -ηλ ) 2 ≤ σ 2 9 × 8 9 × 7 8 ηλ = σ 2 7 ηλ , where we used the fact that ηλ ≤ 0 . 1 .

What is left to do is to upper bound η -2 ‖ x ( t ∗ ) ‖ 4 2 . We proceed by discussing the following three cases respectively:

<!-- formula-not-decoded -->

- 1 ≤ t ≤ T -1 . In this case, we have
- ∗ 0

<!-- formula-not-decoded -->

Thus it holds that

<!-- formula-not-decoded -->

- t ∗ = T 0 . Then we have η -2 ‖ x ( t ∗ ) ‖ 4 2 ≤ (1 -ηλ ) 2 σ 2 4 ηλ .

Taking maximum over three cases, we have

<!-- formula-not-decoded -->

Plugging (20) back into (16), we got for any t ≥ t ∗

<!-- formula-not-decoded -->

where we used the fact that (0 . 5 e 0 . 2 + 1 6 + 1 7 ≈ 0 . 9202 &lt; 1) in the last step. Therefore there exists T 1 = T 0 + 1 4 ηλ max { ln M 2 ηλ σ 2 + ∣ ∣ ∣ ln 2 e 4 M 2 ‖ x (0) ‖ 4 2 η -2 ∣ ∣ ∣ , 4 } , such that for all t ≥ T 1 , η -2 ‖ x ( t ) ‖ 4 2 ≤ 2 σ 2 ηλ .

Theorem 4.5 (SGD+WD) . Let x ( t ) be defined by SGD (5) . For ηλ ≤ 0 . 1 , under Condition 4.4, with probability 1 -5 δ ,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where T 1 = 1 4 ηλ max { ln M 2 ηλ σ 2 + ∣ ∣ ∣ ln 2 e 4 M 2 ‖ x (0) ‖ 4 2 η -2 ∣ ∣ ∣ , 8 } .

Proof. By Lemma C.1, we have

<!-- formula-not-decoded -->

Summing up for t = T 1 to T -1 , we have

<!-- formula-not-decoded -->

Below we will give high-probability bounds for ( A ) , ( B ) and ( C ) respectively. For convenience, we will use A ( t ) , B ( t ) , C ( t ) to denote the t th term in ( A ) , ( B ) and ( C ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here Claim E.4.1 follows from that 2(1 -ηλ ) ≥ 2 and Lemma E.3. Note by the choice of T 1 , we can upper and lower bound ‖ x ( t ) ‖ 2 by Lemmas E.3 and E.4, that is σ 2 4 ηλ ≤ η -2 ‖ x ( t ) ‖ 2 2 ≤ 2 σ 2 ηλ . Thus Claims E.4.2 and E.4.3 is a direct consequence of Lemma C.7.

Thus we conclude w.p. 1 -5 δ ,

<!-- formula-not-decoded -->

rearranging it and applying Lemma C.4, we get

<!-- formula-not-decoded -->

By Condition 4.4, we have σ 2 M 2 ≥ 3 √ λη ln 2 δ , and thus we have

<!-- formula-not-decoded -->

This completes the proof.

## F Omitted Proofs for Convergence of SGD with Relative Global Clipping

Norm dynamics of clipped SGD:

<!-- formula-not-decoded -->

Lemma F.1 (General Properties of G P,C ) . For any C &gt; 1 and measure P supported on R ≥ 0 , it holds that

1. G P,C is continuous and concave;

<!-- formula-not-decoded -->

3. 1 C M P, 1 C ≤ µ P,C ≤ µ P , where µ P is the expectation of P .

Proof of Lemma F.1. (1). Note min { x, ·} is a continuous and concave function for any x , we know G P,C is a concave function. (2). When G P,C is differentiable, we have G ′ P,C ( µ ) = CF ′ P,C ( Cµ ) -1 . Let G ′ P,C ( µ ) = 0 implies that F ′ P,C ( Cµ ) = 1 C . Note F ′ P,C ( Cµ ) = P t ∼ P [ t &gt; F P,C ] , we know G ′ P,C ( 1 C M P, 1 C ) = 0 . By concavity, sup µ ≥ 0 G P,C ( µ ) = G P,C ( 1 C M P, 1 C ) . This argument can be easily generalized to non-differentiable case by using G P,C ( µ ) must be larger than G P,C ( µ ± δ ) for infinitesimal δ . (3). First note that F P,C ( M P, 1 C ) = E t ∼ P [min { t, M P, 1 C } ] ≥ M P, 1 C · P t ∼ P [ t ≥ M P, 1 C ] = 1 C M P, 1 C . In other words, G P,C ( 1 C M P, 1 C ) ≥ 0 .

<!-- formula-not-decoded -->

Theorem F.2. [Classifications of solutions of F P,C ( Cµ ) = µ ]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Suppose there are two solutions 0 &lt; µ 1 &lt; µ 2 . By concavity, we have ∀ 0 ≤ µ ≤ µ 2 , G P,C ( µ ) = 0 . Thus 0 = G P,C (0) + G P,C ( µ 2 ) = 2 g ( µ 2 2 ) , which implies that

<!-- formula-not-decoded -->

that is, P t ∼ P [ t ≥ Cµ 2 ∨ t = 0] = 1 . Thus for any 0 ≤ µ ≤ µ 2 , we have G P,C ( µ ) = Cµ P [ x ≥ Cµ 2 ] -µ = 0 , which implies µ 2 = 1 C M P, 1 C and P [ x = 0] = 1 -1 C !

glyph[negationslash]

Lemma F.3. Under Assumption 4.7, it holds that G P,C x ( 1 C M P x , 1 C ) ≥ α C µ P x ,C for all x = 0 . Proof of Lemma F.3. By definition,

<!-- formula-not-decoded -->

By the definition of the 1 C -median, the second term is non-negative. The proof is completed by applying Assumption 4.7.

Lemma F.4 (Lower and upped bounds for G P x ,C ) . Under Assumption 4.7, it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

glyph[negationslash]

Proof of Lemma F.4. By Lemma F.3, Assumption 4.7 implies that G P,C x ( 1 C M P x , 1 C ) ≥ α C µ P x ,C for all x = 0 . Further note that G P,C x (0) = G P,C x ( µ P x , C ) = 0 . The claims (a), (b) and (c) are immediate by concavity of G P,C x .

The above inequalities also directly imply the following version using µ C and µ C as thresholds. Lemma F.5 (Uniform Lower and upped bounds for G P x ,C ) . Under Assumption 4.7, it holds that for ‖ x ‖ 2 = 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

3.

G

P

x

,C

(

µ

)

≤ -

α

C

(

µ

-

µ

C

)

, for

µ

≥

µ

C

.

<!-- formula-not-decoded -->

For convenience, we define R t := 2 λ η ‖ x ( t ) ‖ 2 2 , g t := ‖∇ L γ t ( x ( t )) ‖ 2 2 , ̂ g t := min { CR t , g t } , ˜ g t := R t ̂ g t = min { CR 2 t , ‖∇ L γ t ( x ( t )) ‖ 2 2 } and g t := ̂ g t R t = min { C, ‖ ∇ L γ t ( x ( t )) ‖ 2 2 R t } . Thus we have E [ ̂ g t | x ( t )] = µ P x ( t ) ,C . Wefurther define β l := 1 -2 λ 2 η 2 + η 4 λ 4 -4 ηλα C (1 -ηλ ) 2 = 1 -4 ηλα C + O ( η 2 λ 2 ) and β u := 1 -2 λ 2 η 2 + η 4 λ 4 -4 ηλα C (1 -ηλ ) 2 +4 C 2 η 2 λ 2 = 1 -4 ηλα C + O ( η 2 λ 2 ) . Given an integer T ≥ 0 , let E 1 T be the event that ∀ 0 ≤ t ′ ≤ t ≤ T,

<!-- formula-not-decoded -->

Let E 2 T be the event that ∀ 0 ≤ t ′ ≤ t ≤ T,

<!-- formula-not-decoded -->

Let E 3 T be the event that ∀ 0 ≤ t ′ ≤ t ≤ T,

<!-- formula-not-decoded -->

Lemma F.6. P [ E i T ] ≥ 1 -δ , for i = 1 , 2 , 3 .

Proof of Lemma F.6. Note the sequence in E i T are martingales whose differences are uniformly bounded ( µ C , µ C and C ). The lemma follows directly from Hoeffding Inequality and Azuma Inequality.

Theorem F.7 (Norm lower bound with clipping: Warm Start) . Suppose Assumption 4.7 holds, with probability at least 1 -δ (or whenever E 1 T holds), if R 2 t ≥ 3 4 µ C , then for any t ′ ≥ t , we have

<!-- formula-not-decoded -->

Proof. We first claim for any t ≤ t ′ ≤ T , conditioned on E 1 T , it holds that R 2 t ′ ≥ µ C 2 . Below we prove by contradiction. If not, let t ′ be the smallest step such that R 2 t ′ &lt; µ C 2 . We let t ∗ be the largest step between t and t ′ such that R 2 t ∗ ≥ µ C ( t ∗ = t -1 is no such t ∗ exists) Thus if t ∗ ≥ t then R 2 t ∗ +1 is at least (1 -ηλ ) 4 R 2 t = (1 -O ( ηλ )) µ C . Otherwise t ∗ = t and it implies that R 2 t ∗ +1 = R 2 t = ( 3 4 -O ( √ ηλ )) µ C . By the definition, we know for any t ∗ +1 ≤ s ≤ t ′ , R 2 s ≤ µ C . Similar to Equation (14), we have

<!-- formula-not-decoded -->

Thus for any s such that µ C ≤ R 2 s ≤ 2 µ C , by Lemma F.5, it holds that

<!-- formula-not-decoded -->

Thus, we have that

<!-- formula-not-decoded -->

That is,

<!-- formula-not-decoded -->

Applying the above inequality for s = t ∗ +1 , . . . , t ′ -1 , we have that

<!-- formula-not-decoded -->

For term (B), we have 1 -β u = 4 ηλα C (1 -ηλ ) 2 (1 + O ( ηλ )) and thus ( B ) = µ C (1 + O ( ηλ )) . Since R t ∗ +1 ≥ 3 4 µ C , it holds that ( A ) ≥ -β l t ′ -t ∗ -1 ( 1 4 + O ( √ λη )) µ C ≥ -( 1 4 + O ( √ λη )) µ C . Since E 1 T holds, we have

<!-- formula-not-decoded -->

glyph[negationslash]

Thus there's some constant ι , such for ηλ ≤ min { ι, α C 64 C ln T 2 /δ } , ( A ) + ( B ) + ( C ) ≥ ( 6 - √ 2 8 -O ( √ ηλ )) µ C ≥ µ C 2 . This leads to a contradiction to the definition of t ′ . Thus for any t ≤ t ′ ≤ T , conditioned on E 1 T , it holds that R 2 t ′ ≥ µ C 2 . Furthermore, if t ∗ = t , then R t ∗ +1 ≥ (1 -O ( √ ηλ )) µ C . Thus ( A ) ≥ -O ( √ ηλ ) µ C . Otherwise if t ∗ = t , then ( A ) ≥ -β l t ′ -t ( 1 4 + O ( √ λη )) µ C . Combine the bounds in these two cases, we conclude that

<!-- formula-not-decoded -->

Theorem F.8 (Norm upper bound with clipping: Warm Start) . Suppose Assumption 4.7 holds, with probability at least 1 -δ (or whenever E 2 T holds), if R 2 t ≤ 3 2 µ C , then for any t ′ ≥ t , we have

<!-- formula-not-decoded -->

Proof of Theorem F.8. We first claim for any t ≤ t ′ ≤ T , conditioned on E 2 T , it holds that R 2 t ′ ≤ 2 µ C . Below we prove by contradiction. If not, let t ′ be the largest step such that R 2 t ′ &gt; 2 µ C . We let t ∗ be the largest step between t and t ′ such that R 2 t ∗ ≤ µ C ( t ∗ = t -1 is no such t ∗ exists) Thus if t ∗ ≥ t then R 2 t ∗ +1 is at most (1 + 2 Cηλ ) 2 R 2 t = (1 + 2 Cηλ ) 2 µ C . Otherwise t ∗ = t and it implies that R 2 t ∗ +1 = R 2 t ≤ 3 2 µ C . By the definition, we know for any t ∗ +1 ≤ s ≤ t ′ , R 2 s ≥ µ C . Similar to Equation (14), we have

<!-- formula-not-decoded -->

Thus for any s such that µ C ≤ R 2 s , by Lemma F.5, it holds that

<!-- formula-not-decoded -->

Thus, we have that

<!-- formula-not-decoded -->

That is,

<!-- formula-not-decoded -->

Applying the above inequality for s = t ∗ +1 , . . . , t ′ -1 , we have

<!-- formula-not-decoded -->

For term (B), we have 1 -β u = 4 ηλα C (1 -ηλ ) 2 (1 + O ( ηλ )) and thus ( B ) = µ C (1 + O ( ηλ )) . Since R t ∗ +1 ≤ 3 2 µ C , it holds that ( A ) ≤ β u t ′ -t ∗ -1 ( 1 2 + O ( √ λη )) µ C ≤ ( 1 2 + O ( √ λη )) µ C . Since E 2 T holds, we have that

<!-- formula-not-decoded -->

glyph[negationslash]

Thus there's some constant ι , such for ηλ ≤ min { ι, α C 64 C ln T 2 /δ } , ( A ) + ( B ) + ( C ) ≤ ( 6+ √ 2 4 + O ( √ ηλ )) µ C ≤ 2 µ C . This leads to a contradiction to the definition of t ′ . Thus for any t ≤ t ′ ≤ T , conditioned on E 1 T , it holds that R 2 t ′ ≥ 2 µ C . Furthermore, if t ∗ = t , then R t ∗ +1 ≤ (1 + O ( √ ηλ )) µ C . Thus ( A ) ≤ O ( √ ηλ ) µ C . Otherwise if t ∗ = t , then ( A ) ≤ β u t ′ -t ( 1 2 + O ( √ λη )) µ C . Combine the bounds in these two cases, we conclude that

<!-- formula-not-decoded -->

Theorem F.9 (Norm Convergence of clipped SGD) . Suppose Assumption 4.7 holds, for ηλ = O (min { 1 , α C C ln T/δ 2 } ) , with probability 1 -3 δ (when E 1 T , E 2 T and E 3 T happens), there is a T ′ =

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

More concretely, we have that

<!-- formula-not-decoded -->

Proof of Theorem F.9. We will prove the desired inequality always holds when E i T holds, for i = 1 , 2 , 3 . We have already proved the result for the case where 3 4 µ C ≤ R 2 t ≤ 3 2 µ C in Theorems F.7 and F.8. Now we turn to the case where R 2 0 ≥ 3 2 µ C and R 2 0 ≤ 1 2 µ C . Our goal is to prove with high probability, that R 2 t ∈ [ 3 4 µ C , 3 2 µ C ] for at least some t &lt; T ′ .

Below we first show ∃ 0 &lt; t &lt; T ′ , R 2 t ≤ 3 2 µ C . Otherwise, similar to Equation (26),

<!-- formula-not-decoded -->

Thus for any s such that 3 2 µ C ≤ R 2 s , by Lemma F.5, it holds that

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

Note that g s ≤ C , we have

<!-- formula-not-decoded -->

Since we assume ∀ 0 ≤ t ≤ T ′ , R 2 t ≥ 3 2 µ C , conditioned on E 3 T , we have

<!-- formula-not-decoded -->

which is in contradiction with the definition of T ′ = max { ln R 2 0 µ C , ln µ C R 2 0 } + O (1) α C ηλ .

Now we show ∃ 0 &lt; t &lt; T ′ , R 2 t ≥ 3 4 µ C . Otherwise, similar to Equation (26),

<!-- formula-not-decoded -->

Thus for any s such that R 2 s ≤ 4 5 µ C , by Lemma F.5, it holds that

<!-- formula-not-decoded -->

Thus, we have that

<!-- formula-not-decoded -->

Note that g s ≤ C , we have that

<!-- formula-not-decoded -->

Since we assume ∀ 0 ≤ t ≤ T ′ , R 2 t ≥ 3 2 µ C , conditioned on E 3 T , we have

<!-- formula-not-decoded -->

which is in contradiction with the definition of T ′ = max { ln R 2 0 µ C , ln µ C R 2 0 } + O (1) α C ηλ

<!-- formula-not-decoded -->

Proof of Theorem 4.8. The proof of Algorithm 1 is almost identical to that of Theorem 4.5, except replacing M by 2 µ C , σ by µ C , σ by µ C since the clipped stochastic gradient has smaller maximum norm, maximum covariance and smaller covariance.

## G Convergence of SGD for multi-group scale invariant functions

In this section we extend our results to the multi-group scale invariant setting, which is quite common in practice, e.g. a feedforward network with normalization after each layer. By Definition G.1, multi-group scale invariant function is also scale invariant. However, it violates the assumption that the smoothness and the expectation of stochastic gradient norm square is lower bounded on unit sphere (indeed the loss function is not defined at everywhere on unit sphere), and thus needs to be treated separately. A simple example would be L ( x , y ) = L ( x ‖ x ‖ 2 , y ‖ y ‖ 2 ) , the loss L is undefined at any point where ‖ x ‖ 2 = 1 and y = 0 . Yet our analysis for single scale invariant parameter group can still extend to this case, with a similar assumption that the expected gradient norm square is lower bounded.

Let d 1 , . . . , d K be positive integers with d = ∑ K k =1 d k . For x ∈ R d = R d 1 × . . . × R d K , we use s k to denote ∑ i ≤ k d i and x k to denote the vector [ x s k -1 , . . . , x s k -1 ] glyph[latticetop] . For convenience, we define ∇ k f ( x ) = ∂f ( x ) ∂ x k for any 1 ≤ k ≤ K .

Definition G.1. Given d 1 , . . . , d K and a cone U ⊂ R d , we say a function f : U → R is multi-group scale invariant iff f ( x 1 , . . . , x K ) = f ( c 1 x 1 , . . . , c K x K ) for any x ∈ U and c k &gt; 0 for 1 ≤ k ≤ K .

Setting: Similarly, we assume there exists constants σ k and σ k , such that σ 2 k ≤ E ‖∇ k L γ ( x ) ‖ 2 2 ≤ σ 2 k , for any x such that ‖ x k ‖ 2 = 1 . In this subsection, we define ρ := max ‖ x k ‖ 2 =1 , ∀ k λ max ( ∇ 2 L ( x )) .

Condition G.2. σ 2 k M 2 k ≥ 3 e 4 ηλ √ λη ln 2 T 2 δ .

Theorem G.3 (SGD+WD, Multi-group Scale Invariance) . With probability 1 -( K +2) δ , it holds that

<!-- formula-not-decoded -->

where T 1 = 1 4 ηλ max k { ln M 2 k ηλ σ 2 k + ∣ ∣ ∣ ln 2 e 4 M 2 k ‖ x k (0) ‖ 4 2 η -2 ∣ ∣ ∣ , 8 } .

Following the same strategy, we can prove the multi-group counterpart of norm convergence result, Lemma E.2. Given a integer T ≥ 0 , let E T,k be the event that ∀ 0 ≤ t ′ ≤ t ≤ T -1 ,

<!-- formula-not-decoded -->

Lemma G.4. For any 0 ≤ t ′ ≤ t ≤ T -1 , 1 ≤ k ≤ K , it holds that

<!-- formula-not-decoded -->

Thus we have P [ E T,k ] ≥ 1 -δ by Lemma C.6.

The following theorem is a restatement of Lemmas E.3 and E.4 in the context of multi-group scale invariance.

Lemma G.5. Under Condition G.2, there exists T 1 = 1 4 ηλ max k { ln M 2 k ηλ σ 2 k + ∣ ∣ ∣ ln 2 e 4 M 2 k ‖ x k (0) ‖ 4 2 η -2 ∣ ∣ ∣ , 8 } , such that ∀ t ≥ T 1 , σ 2 k 4 ηλ ≤ η -2 ‖ x ( t ) ‖ 4 2 ≤ 2 σ 2 k ηλ , conditioned on ∪ K k =1 E T,k .

The proof of Theorem G.3 is a natural generalization of Theorem 4.5.

Proof of Theorem G.3. Setting x = (1 -ηλ ) x ( t ) in Lemma C.2, we have

<!-- formula-not-decoded -->

For convenience we define ̂ x = [ x glyph[latticetop] 1 ‖ x 1 ‖ 2 , . . . , x glyph[latticetop] K ‖ x K ‖ 2 ] glyph[latticetop] . Summing up for t = T 1 to T -1 , we have

<!-- formula-not-decoded -->

Below we will give high-probability bounds for ( A ) , ( B ) and ( C ) respectively. For convenience, we will use A ( t ) , B ( t ) , C ( t ) to denote the t th term in ( A ) , ( B ) and ( C ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here Claim G.5.1 follows from that 2(1 -ηλ ) ≥ √ 2 and Lemma E.3. Note by the choice of T 1 , we can upper and lower bound ‖ x ( t ) ‖ 2 by Lemma G.5, that is σ 2 k 4 ηλ ≤ η -2 ‖ x k ( t ) ‖ 2 2 ≤ 2 σ 2 k ηλ . Thus Claims G.5.2 and G.5.3 is a direct consequence of Lemma C.7.

Thus by Chernoff bound (Lemma C.6), with probability at least 1 -( K +2) δ , Equation (29) holds.