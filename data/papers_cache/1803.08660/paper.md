## Lifting Layers: Analysis and Applications

Peter Ochs * † , Tim Meinhardt ‡ , Laura Leal-Taixe ‡ , Michael Moeller * glyph[sharp] ,

glyph[sharp]

University of Siegen, Siegen, Germany † Saarland University, Saarbr¨ ucken, Germany ‡ TU Munich, Munich, Germany

## Abstract

The great advances of learning-based approaches in image processing and computer vision are largely based on deeply nested networks that compose linear transfer functions with suitable non-linearities. Interestingly, the most frequently used nonlinearities in imaging applications (variants of the rectified linear unit) are uncommon in low dimensional approximation problems. In this paper we propose a novel nonlinear transfer function, called lifting , which is motivated from a related technique in convex optimization. A lifting layer increases the dimensionality of the input, naturally yields a linear spline when combined with a fully connected layer, and therefore closes the gap between low and high dimensional approximation problems. Moreover, applying the lifting operation to the loss layer of the network allows us to handle non-convex and flat (zero-gradient) cost functions. We analyze the proposed lifting theoretically, exemplify interesting properties in synthetic experiments and demonstrate its effectiveness in deep learning approaches to image classification and denoising.

Keywords Machine Learning, Deep Learning, Interpolation, Approximation Theory, Convex Relaxation, Lifting

## 1 Introduction

Deep Learning has seen a tremendous success within the last 10 years improving the stateof-the-art in almost all computer vision and image processing tasks significantly. While one of the main explanations for this success is the replacement of handcrafted methods and features with data-driven approaches, the architectures of successful networks remain handcrafted and difficult to interpret.

The use of some common building blocks, such as convolutions, in imaging tasks is intuitive as they establish translational invariance. The composition of linear transfer functions with non-linearities is a natural way to achieve a simple but expressive representation, but the choice of non-linearity is less intuitive: Starting from biologically motivated step functions or their smooth approximations by sigmoids, researchers have turned to rectified linear units (ReLUs),

* These authors have equally contributed.

Figure 1: The proposed lifting identifies predefined labels t i ∈ R with the unit vectors e i in R L , L ≥ 2. As illustrated in (a), a number x that is represented as a convex combination of t i and t i +1 has a natural representation in a higher dimensional lifted space, see (3). When a lifting layer is combined with a fully connected layer it corresponds to a linear spline, and when both the input as well as the desired output are lifted it allows non-convex cost functions to be represented as a convex minimization problem (b). Finally, as illustrated in (c), coordinate-wise lifting yields an interesting representation of images, which allows textures of different intensities to be filtered differently.

<!-- image -->

<!-- formula-not-decoded -->

to avoid the optimization-based problem of a vanishing gradient. The derivative of a ReLU is σ ′ ( x ) = 1 for all x &gt; 0. Nonetheless, the derivative remains zero for x &lt; 0, which does not seem to make it a natural choice for an activation function, and often leads to 'dead' ReLUs. This problem has been partially addressed with ReLU variants, such as leaky ReLUs [16], parameterized ReLUs [10], or maxout units [8]. These remain amongst the most popular choice of non-linearities as they allow for fast network training in practice.

In this paper we propose a novel type of non-linear layer, which we call lifting layer glyph[lscript] . In contrast to ReLUs (1), it does not discard large parts of the input data, but rather lifts it to different channels that allow the input x to be processed independently on different intervals. As we discuss in more detail in Section 3.4, the simplest form of the proposed lifting non-linearity is the mapping

<!-- formula-not-decoded -->

which essentially consists of two complementary ReLUs and therefore neither discards half of the incoming inputs nor has intervals of zero gradients.

More generally, the proposed non-linearity depends on labels t 1 &lt; . . . &lt; t L ∈ R (typically linearly spaced) and is defined as a function glyph[lscript] : R → R L that maps a scalar input x ∈ R to a vector glyph[lscript] ( x ) ∈ R L via

<!-- formula-not-decoded -->

The motivation of the proposed lifting non-linearity is illustrated in Figure 1. In particular, we highlight the following contributions :

- (i) The concept of representing a low dimensional variable in a higher dimensional space is a well-known optimization technique called functional lifting , see [19]. Non-convex problems are reformulated as the minimization of a convex energy in the higher dimensional 'lifted' space. While the introduction of lifting layers does not directly correspond to the optimization technique, some of the advantageous properties carry over as we detail in Section 3.
- (ii) ReLUs are commonly used in deep learning for imaging applications, however their low dimensional relatives of interpolation or regression problems are typically tackled differently, e.g. by fitting (piecewise) polynomials. We show that a lifting layer followed by a fully connected layer yields a linear spline , which closes the gap between low and high dimensional interpolation problems . In particular, the aforementioned architecture can approximate any continuous function f : R → R to arbitrary precision and can still be trained by solving a convex optimization problem whenever the loss function is convex, a favorable property that is, for example, not shared even by the simplest ReLU-based architecture.
- (iii) By additionally lifting the desired output of the network, one can represent nonconvex cost functions in a convex fashion . Besides handling the non-convexity, such an approach allows for the minimization of cost functions with large areas of zero gradients such as truncated linear costs.
- (iv) We demonstrate that the proposed lifting improves the test accuracy in comparison to similar ReLU-based architectures in several experiments on image classification and produces state-of-the-art image denoising results, making it an attractive universal tool in the design of neural networks.

## 2 Related Work

Lifting in Convex Optimization. One motivation for the proposed non-linearity comes from a technique called functional lifting which allows particular types of non-convex optimization problems to be reformulated as convex problems in a higher dimensional space, see

[19] for details. The recent advances in functional lifting [17] have shown that (3) is a particularly well-suited discretization of the continuous model from [19]. Although, the techniques differ significantly, we hope for the general idea of an easier optimization in higher dimensions to carry over. Indeed, for simple instances of neural network architecture, we prove several favorable properties for our lifting layer that are related to properties of functional lifting. Details are provided in Sections 3 and 4.

Non-linearities in Neural Networks. While many non-linear transfer functions have been studied in the literature (see [7, Section 6.3] for an overview), the ReLU in (1) remains the most popular choice. Unfortunately, it has the drawback that its gradient is zero for all x &lt; 0, thus preventing gradient based optimization techniques to advance if the activation is zero (dead ReLU problem). Several variants of the ReLU avoid this problem by either utilizing smoother activations such as softplus [6] or exponential linear units [3], or by considering

<!-- formula-not-decoded -->

e.g. the absolute value rectification α = -1 [12], leaky ReLUs with a small α &gt; 0 [16], randomized leaky ReLUs with randomly choosen α [21], parametric ReLUs in which α is a learnable parameter [10]. Self-normalizing neural networks [13] use scaled exponential LUs (SELUs) which have further normalizing properties and therefore replace the use of batch normalization techniques [11]. While the activation (4) seems closely related to the simplest case (2) of our lifting, the latter allows to process max( x, 0) and min( x, 0) separately, avoiding the problem of predefining α in (4) and leading to more freedom in the resulting function.

Another related non-linear transfer function are maxout units [8], which (in the 1-D case we are currently considering) are defined as

<!-- formula-not-decoded -->

They can represent any piecewise linear convex function. However, as we show in Proposition 2, a combination of the proposed lifting layer with a fully connected layer drops the restriction to convex activation functions, and allows us to learn any piecewise linear function. This special architecture shows also similarities to learning the non-linear activation function in terms of basis functions [2].

Universal Approximation Theorem. As an extension of the universal approximation theorem in [4], it has been shown in [15] that the set of feedforward networks with one hidden layer, i.e., all functions N of the form

<!-- formula-not-decoded -->

for some integer N , and weights θ 1 j ∈ R , θ 2 j ∈ R n , b j ∈ R are dense in the set of continuous functions f : [0 , 1] n → R if and only if σ is not a polynomial. While this result demonstrates the expressive power of all common activation functions, the approximation of some given function f with a network N of the form (6) requires optimization for the parameters θ 1 and ( θ 2 , b ) which inevitably leads to a non-convex problem. We prove the same expressive power of a lifting based architecture (see Corollary 3), while, remarkably, our corresponding learning problem is a convex optimization problem. Moreover, beyond the qualitative density result for (6), we may quantify the approximation quality depending on a simple measure for the 'complexity' of the continuous function to be approximated (see Corollary 3 and the Appendix A).

## 3 Lifting Layers

In this section, we introduce the proposed lifting layers (Section 3.1) and study their favorable properties in a simple 1-D setting (Section 3.2). The restriction to 1-D functions is mainly for illustrative purposes and simplicity. All results can be transferred to higher dimensions via a vector-valued lifting (Section 3.3). The analysis provided in this section does not directly apply to deep networks, however it provides an intuition for this setting. Section 3.4 discusses some practical aspects and reveals a connection to ReLUs. All proofs and the details of the vector-valued lifting are provided in Appendix A and B.

## 3.1 Definition

The following definition formalizes the lifting layer from the introduction.

Definition 1 (Lifting). We define the lifting of a variable x ∈ [ t, t ], t, t ∈ R , with respect to the Euclidean basis E := { e 1 , . . . , e L } of R L and a knot sequence t = t 1 &lt; t 2 &lt; . . . &lt; t L = t , for some L ∈ N , as a mapping glyph[lscript] : [ t, t ] → R L given by

<!-- formula-not-decoded -->

where λ l ( x ) := x -t l t l +1 -t l ∈ R . The inverse mapping glyph[lscript] † : R L → R of glyph[lscript] , which satisfies glyph[lscript] † ( glyph[lscript] ( x )) = x , is defined by

<!-- formula-not-decoded -->

Note that while liftings could be defined with respect to an arbitrary basis E of R L (with a slight modification of the inverse mapping), we decided to limit ourselves to the Euclidean basis for the sake of simplicity. Furthermore, we limit ourselves to inputs x that lie in the predefined interval [ t, t ]. Although, the idea extends to the entire real line by linear extrapolation, it requires more technical details. For the sake of a clean presentation, we omit these details.

## 3.2 Analysis in 1D

Although, here we are concerned with 1-D functions, these properties and examples provide some intuition for the implementation of the lifting layer into a deep architecture. Moreover, analogue results can be stated for the lifting of higher dimensional spaces.

Proposition 2 (Prediction of a Linear Spline). The composition of a fully connected layer z ↦→〈 θ, z 〉 with θ ∈ R L , and a lifting layer, i.e.,

<!-- formula-not-decoded -->

yields a linear spline (continuous piecewise linear function). Conversely, any linear spline can be expressed in the form of (9).

Although the architecture in (9) does not fall into the class of functions covered by the universal approximation theorem, well-known results of linear spline interpolation still guarantee the same results.

Corollary 3 (Prediction of Continuous Functions). Any continuous function f : [ t, t ] → R can be represented arbitrarily accurate with a network architecture N θ ( x ) := 〈 θ, glyph[lscript] ( x ) 〉 for sufficiently large L , θ ∈ R L .

Furthermore, as linear splines can of course fit any (spatially distinct) data points exactly, our simple network architecture has the same property for a particular choice of labels t i . On the other hand, this result suggests that using a small number of labels acts as regularization of the type of linear interpolation.

glyph[negationslash]

Corollary 4 (Overfitting). Let ( x i , y i ) be training data, i = 1 , . . . , N with x i = x j for i = j . If L = N and t i = x i , there exists θ such that N θ ( x ) := 〈 θ, glyph[lscript] ( x ) 〉 is exact at all data points x = x i , i.e. N θ ( x i ) = y i for all i = 1 , . . . , N .

glyph[negationslash]

Note that Proposition 2 highlights two crucial differences of the proposed non-linearity to the maxout function in (5): (i) maxout functions can only represent convex piecewise linear functions, while liftings can represent arbitrary piecewise linear functions; (ii) The maxout function is non-linear w.r.t. its parameters ( θ j , b j ), while the simple architecture in (9) (with lifting) is linear w.r.t. its parameters ( θ, b ). The advantage of a lifting layer compared to a ReLU, which is less expressive and also non-linear w.r.t. its parameters, is even more significant.

Remarkably, the optimal approximation of a continuous function by a linear spline (for any choice of t i ), yields a convex minimization problem.

Proposition 5 (Convexity of a simple Regression Problem). Let ( x i , y i ) ∈ [ t, t ] × R be training data, i = 1 , . . . , N . Then, the solution of the problem

<!-- formula-not-decoded -->

Figure 2: Intuition and notation of the vector-valued lifting.

<!-- image -->

yields the best linear spline fit of the training data with respect to the loss function L . In particular, if L is convex, then (10) is a convex optimization problem.

As the following example shows, this is not true for ReLUs and maxout functions.

Example 6. The convex loss L ( z ; 1) = ( z -1) 2 composed with a ReLU applied to a linear transfer function, i.e., θ ↦→ max( θx i , 0) with θ ∈ R , leads to a non-convex objective function, e.g. for x i = 1, θ ↦→ (max( θ, 0) -1) 2 is non-convex.

Therefore, in the light of Proposition 5, the proposed lifting closes the gap between low dimensional approximation and regression problems (where linear splines are extremely common), and high dimensional approximation/learning problems, where ReLUs have been used instead of linear spline type of functions.

## 3.3 Vector-Valued Lifting Layers

A vector-valued construction of the lifting similar to [14] allows us to naturally extend all our previous results for functions f : [ t, t ] → R to functions f : Ω ⊂ R d → R . Definition 1 is generalized to d dimensions by triangulating the compact domain Ω, and identifying each vertex of the resulting mesh with a unit vector in a space R N , where N is the total number of vertices. The lifted vector contains the barycentric coordinates of a point x ∈ R d with respect its surrounding vertices. The resulting lifting remains a continuous piecewise linear function when combined with a fully connected layer (cf. Proposition 2), and yields a convex problem when looking for the best piecewise linear fit on a given triangular mesh (cf. Proposition 5). Intuition is provided in Figure 2 and the details are provided in Appendix A. Unfortunately, discretizing a domain Ω ⊂ R d with L labels per dimension leads to N = L d vertices, which makes a vector-valued lifting prohibitively expensive for large d . Therefore, in high dimensional applications, we turn to narrower and deeper network architectures, in which the scalar-valued lifting is applied to each component separately. The latter sacrifices the convexity of the overall problem for the sake of a high expressiveness with comparably few parameters. Intuitively, the increasing expressiveness is explained by an exponentially growing number of kinks for the composition of layers that represent linear splines. A similar reasoning can be found in [18].

## 3.4 Scaled Lifting

We are free to scale the lifted representation defined in (7), when the inversion formula in (8) compensates for this scaling. For practical purposes, we found it to be advantageous to also introduce a scaled lifting by replacing (7) in Definition 1 by

<!-- formula-not-decoded -->

where λ l ( x ) := x -t l t l +1 -t l ∈ R . The inversion formula reduces to the sum over all components of the vector in this case. We believe that such a scaled lifting is often advantageous: (i) The magnitude/meaning of the components of the lifted vector is preserved and does not have to be learned; (ii) For an uneven number of equally distributed labels in [ -t, t ], one of the labels t l will be zero, which allows us to omit it and represent a scaled lifting into R L with L -1 many entries. For L = 3 for example, we find that t 1 = -t , t 2 = 0, and t 3 = t such that

<!-- formula-not-decoded -->

As the second component remains zero, we can introduce an equivalent more memory efficient variant of the scaled lifting which we already stated in (2).

## 4 Lifting the Output

So far, we considered liftings as a non-linear layer in a neural network. However, motivated by lifting-based optimization techniques, which seek a tight convex approximation to problems involving non-convex loss functions, this section presents a convexification of nonconvex loss functions by lifting in the context of neural networks. This goal is achieved by approximating the loss by a linear spline and predicting the output of the network in a lifted representation. The advantages of this approach are demonstrated at the end of this section in Example 10 for a robust regression problem with a vast number of outliers.

Consider a loss function L y : R → R defined for a certain given output y (the total loss for samples ( x i , y i ), i = 1 , . . . , N , may be given by ∑ N i =1 L y i ( x i )). We achieve the tight convex approximation by a lifting function glyph[lscript] y : [ t y , t y ] → R L y for the range of the loss function im( L y ) ⊂ R with respect to the standard basis E y = { e 1 y , . . . , e L y y } and a knot sequence t y = t 1 y &lt; . . . &lt; t L y y &lt; t y following Definition 1.

Figure 3: Visualization of Example 10 for a regression problem with 40% outliers. Our lifting of a (nonconvex) truncated linear loss to a convex optimization problem robustly fits the function nearly optimally (see (c)), whereas the most robust convex formulation (without lifting) is severely perturbed by the outliers (see (d)). Trying to optimize the non-convex cost function directly yields different results based on the initialization of the weights and is prone to getting stuck in suboptimal local minima, see (e)-(h).

<!-- image -->

The goal of the convex approximation is to predict the lifted representation of the loss, i.e. a vector z ∈ R L y . However, in order to assign the correct loss to the lifted variable, it needs to lie in im( glyph[lscript] y ). In this case, we have a one-to-one representation of the loss between [ t y , t y ] and im( glyph[lscript] y ), which is shown by the following lemma.

Lemma 7 (Characterization of the Range of glyph[lscript] ). The range of the lifting glyph[lscript] : [ t, t ] → R L is given by

<!-- formula-not-decoded -->

and the mapping glyph[lscript] is a bijection between [ t, t ] and im( glyph[lscript] ) with inverse glyph[lscript] † .

Since the image of the range of glyph[lscript] y is not convex, we relax it to a convex set, actually to the smallest convex set that contains im( glyph[lscript] y ), the convex hull of im( glyph[lscript] y ).

Lemma 8 (Convex Hull of the Range of glyph[lscript] ). The convex hull conv(im( glyph[lscript] )) of im( glyph[lscript] ) is the unit simplex in R L .

Putting the results together, we obtain a tight convex approximation of the (possibly non-convex) loss function L y ( x ) by glyph[lscript] † y ( z ) with z ∈ im( glyph[lscript] y ), i.e. instead of considering a network N θ ( x ) and evaluate L y ( N θ ( x )), we consider a network ˜ N θ ( x ) that predicts a point in conv(im( glyph[lscript] y )) ⊂ R L y and evaluate the loss glyph[lscript] † y ( ˜ N θ ( x )). As it is hard to incorporate rangeconstraints into the network's prediction, we compose the network with a lifting layer glyph[lscript] x , i.e.

we consider glyph[lscript] † y ( ˜ θglyph[lscript] x ( ˜ N θ ( x ))) with ˜ θ ∈ R L y × L x , for which simpler constraints may be derived that can be handled easily. The following proposition states the convexity of the relaxed problem w.r.t. the parameters of the loss layer ˜ θ for a non-convex loss function L y .

Proposition 9 (Convex Relaxation of a simple non-convex Regression Problem). Let ( x i , y i ) ∈ [ t, t ] × [ t y , t y ] be training data, i = 1 , . . . , N . Moreover, let glyph[lscript] y be a lifting of the common image [ t y , t y ] of the loss functions L y i , i = 1 , . . . , N , and glyph[lscript] x is the lifting of the domain of L y . Then

<!-- formula-not-decoded -->

is a convex relaxation of the (non-convex) loss function, and the constraints guarantee that θglyph[lscript] x ( x i ) ∈ conv(im( glyph[lscript] y )).

The objective in (14) is linear (w.r.t. θ ) and can be written as

<!-- formula-not-decoded -->

where c := ∑ N i =1 t y glyph[lscript] x ( x i ) glyph[latticetop] , with t y := ( t 1 y , . . . , t L y y ) glyph[latticetop] , is the cost matrix for assigning the loss value t p y to the inputs x i .

Moreover, the closed-form solution of (14) is given for all q = 1 , . . . , L x by θ p,q = 1, if the index p minimizes c p,q , and θ p,q = 0 otherwise.

Example 10 (Robust fitting). For illustrative purposes of the advantages of this section, we consider a regression problem with 40% outliers as visualized in Figure 3(c) and (d). Statistics motivates us to use a robust non-convex loss function. Our lifting allows us to use a robust (non-convex) truncated linear loss in a convex optimization problem (Proposition 9), which can easily ignore the outliers and achieve a nearly optimal fit (see Figure 3(c)), whereas the most robust convex loss (without lifting), the glyph[lscript] 1 -loss, yields a solution that is severely perturbed by the outliers (see Figure 3(d)). The cost matrix c from (15) that represents the non-convex loss (of this example) is shown in Figure 3(a) and the computed optimal θ is visualized in Figure 3(b). For comparison purposes we also show the results of a direct (gradient descent + momentum) optimization of the truncated linear costs with four different initial weights chosen from a zero mean Gaussian distribution. As we can see the results greatly differ for different initializations and always got stuck in suboptimal local minima.

## 5 Numerical Experiments

In this section we provide synthetic numerical experiments to illustrate the behavior of lifting layers on simple examples, before moving to real-world imaging applications. We implemented lifting layers in MATLAB as well as in PyTorch and will make all code for reproducing the experiments available upon acceptance of this manuscript.

## 5.1 Synthetic Examples

The following results were obtained using a stochastic gradient descent (SGD) algorithm with a momentum of 0.9, using minibatches of size 128, and a learning rate of 0 . 1. Furthermore, we use weight decay with a parameter of 10 -4 .

## 5.1.1 1-D Fitting

To illustrate our results of Proposition 5, we first consider the example of fitting values y i = sin( x i ) from input data x i sampled uniformly in [0 , 2 π ]. We compare the lifting-based architecture N θ ( x ) = 〈 θ, glyph[lscript] ( x ) 〉 (Lift-Net) with the standard design architecture fc 1 ( σ (fc 9 ( x ))) (Std-Net), where σ ( x ) = max( x, 0) applies coordinate-wise and fc n denotes a fully connected layer with n output neurons. Figure 4 shows the resulting functions after 25, 75, 200, and 2000 epochs of training.

Figure 4: Illustrating the results of approximating a sine function on [0 , 2 π ] with 50 training examples after different number of epochs. While the proposed architecture with lifting yields a convex problem for which SGD converges quickly (upper row), the standard architecture based on ReLUs yields an (ambiguous) nonconvex problem which leads to slower convergence and a suboptimal local minimum after 4000 epochs (lower row).

<!-- image -->

## 5.1.2 2-D Fitting

While the above results were expected based on the favorable theoretical properties, we now consider a more difficult test case of fitting the function

<!-- formula-not-decoded -->

on [0 , 2 π ] 2 . Note that although a 2-D input still allows for a vector-valued lifting, our goal is to illustrate that even a coordinate-wise lifting has favorable properties (beyond being able to approximate any separable function with a single layer, which is a simple extension of Corollary 3). We therefore compare the two networks

Figure 5: Illustrating the results of approximating the function in (16) with the standard network in (Std-Net) (middle row) and the architecture in (Lift-Net) based on lifting the input data (upper row). The red markers illustrate the training data, the surface represents the overall network function, and the RMSE measures its difference to the true underlying function (16), which is shown in the bottom row on the left. Similar to the results of Figure 4, our lifting based architecture converges more quickly and yields a better approximation of the true underlying function (lower left) after 2000 epochs. The middle and right approximations in the bottom row illustrate a vector-valued lifting (see Section 3.3) into 4 2 (middle) and 11 2 (right) dimensions. The latter can be trained by solving a linear system. We illustrate the triangular mesh used for the lifting below the graph of the function to illustrate that the approximation is indeed piecewise linear (as stated in Proposition 2).

<!-- image -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the notation [ u, v ] in the above formula denotes the concatenation of the two vectors u and v . The corresponding training now yields a non-convex optimization problem in both cases. As we can see in Figure 5 the general behavior is similar to the 1-D case: Increasing the dimensionality via lifting the input data yields faster convergence and a more precise approximation than increasing the dimensionality with a parameterized filtering. For the sake of completeness, we have included a vector-valued lifting with an illustration of the underlying 2-D triangulation in the bottom row of Figure 5.

Figure 6: Comparing different approaches for image classification on CIFAR-10 and CIFAR-100. The proposed architecture with lifting layers shows a superior performance in comparison to its ReLU-based relatives in both cases.

<!-- image -->

## 5.2 Image Classification

As a real-world imaging example we consider the problem of image classification. To illustrate the behavior of our lifting layer, we use the 'Deep MNIST for expert model' ( ME-model ) by TensorFlow 1 as a simple standard architecture:

<!-- formula-not-decoded -->

which applies a standard ReLU activation, max pooling and outputs a final number of n classes. In our experiments, we use an additional batch-normalization (BN) to improve the accuracy significantly, and denote the corresponding model by ME-model+BN .

Our model is formed by replacing all ReLUs by a scaled lifting layer (as introduced in Section 3.4) with L = 3, where we scaled with the absolute value | t i | of the labels to allow for a meaningful combination with the max pooling layers. We found the comparably small lifting of L = 3 to yield the best results in (deeply) nested architectures. As our lifting layer increases the number of channels by a factor of 2, our model has almost twice as many free parameters as the ME model. Since this could yield an unfair comparison, we additionally include a larger model Large ME-model+BN with twice as many convolution filters and fully-connected neurons resulting in even more free parameters than our model.

Figure 6 shows the results each of these models obtains on the image classification problems CIFAR-10 and CIFAR-100. As we can see, the favorable behavior of the synthetic experiments carried over to the exemplary architectures in image classification: Our proposed architecture based on lifting layers has the smallest test error and loss in both experiments. Both common strategies, i.e. including batch normalization and increasing the size of the model, improved the results, but even the larger of the two ReLU-bases architectures remains inferior to the lifting-based architecture.

[1 https://www.tensorflow.org/tutorials/layers](https://www.tensorflow.org/tutorials/layers)

Figure 7: MNIST image classification comparison of our lifting activation with the standard ReLU and its maxout generalization. The ReLU, maxout and lifting architectures (79510, 79010 and 76485 trainable parameters) achieved a best test error of 3 . 07%, 2 . 91% and 2 . 61%, respectively. The proposed approach behaves favorably in terms of the test loss from epoch 50 on, leading to a lower overall test error after 100 epochs.

<!-- image -->

Table 1: Average PSNRs in [dB] for the BSD68 dataset for different standard deviations σ of the Gaussian noise on all of which our lifting layer based architecture is among the leading methods. Please note that (most likely due to variations in the random seeds) our reproduced DnCNN-S results are different - in the second decimal place - from the results reported in [22].

| Reconstruction PSNR in [ dB ]   | Reconstruction PSNR in [ dB ]   | Reconstruction PSNR in [ dB ]   | Reconstruction PSNR in [ dB ]   | Reconstruction PSNR in [ dB ]   | Reconstruction PSNR in [ dB ]   | Reconstruction PSNR in [ dB ]   | Reconstruction PSNR in [ dB ]   | Reconstruction PSNR in [ dB ]   | Reconstruction PSNR in [ dB ]   |
|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|
| σ                               | Noisy                           | BM3D [5]                        | WNNM [9]                        | EPLL [23]                       | BSH12 [1]                       | CSF [20]                        | TNRD [2]                        | DnCNN-S [22]                    | Our                             |
| 15                              | 24.80                           | 31.07                           | 31.37                           | 31.21                           | -                               | 31.24                           | 31.42                           | 31.72                           | 31.72                           |
| 25                              | 20.48                           | 28.57                           | 28.83                           | 28.68                           | 28.96                           | 28.74                           | 28.92                           | 29.21                           | 29.21                           |
| 50                              | 14.91                           | 25.62                           | 25.87                           | 25.67                           | 26.03                           | -                               | 25.97                           | 26.21                           | 26.23                           |

## 5.3 Maxout Activation Units

To also compare the proposed lifting activation layer with the maxout activation, we conduct a simple MNIST image classification experiment with a fully connected one-hidden-layer architecture, using a ReLu, maxout or lifting as activations. For the maxout layer we apply a feature reduction by a factor of 2 which has the capabilities of representing a regular ReLU and a lifting layer as in (2). Due to the nature of the different activations - maxout applies a max pooling and lifting increases the number of input neurons in the subsequent layer we adjusted the number of neurons in the hidden layer to make for an approximately equal and fair amount of trainable parameters.

The results in Figure 7 are achieved after optimizing a cross-entropy loss for 100 training epochs by applying SGD with learning rate 0 . 01. Particularly, each architecture was trained with the identical experimental setup. While both the maxout and our lifting activation yield a similar convergence behavior better than the standard ReLU, the proposed method exceeds in terms of the final lowest test error.

Figure 8: In (a) we illustrate our Lift-46 image denoising architecture which implements 16 convolution layers with 46 filters. Although its test PSNR in (b) for Gaussian noise with σ = 25 plateaus - after a learning rate decay at 30 epochs - to the same final value it generally shows a favorable and more stable behavior.

<!-- image -->

## 5.4 Image Denoising

To also illustrate the effectiveness of lifting layers for networks mapping images to images, we consider the problem of Gaussian image denoising. We designed the Lift-46 architecture with 16 blocks each of which consists of 46 convolution filters of size 3 × 3, batch normalization, and a lifting layer with L = 3 following the same experimental reasoning for deep architectures as in Section 5.2. As illustrated in Figure 8(a), a final convolutional layer outputs an image we train to approximate the residual, i.e., noise-only, image. Due to its state-of-the-art performance in image denoising we adopted the same training pipeline as for the DnCNN-S architecture from [22] which resembles our Lift-46 network but implements a regular ReLU and 64 convolution filters. The two architectures contain an approximately equal amount of trainable parameters.

Table 1 compares our architecture with a variety of denoising methods most notably the DnCNN-S [22] and shows that we produce state-of-the-art performance for removing Gaussian noise of different standard deviations σ . In addition, the development of the test PSNR in Figure 8(b) suggests a more stable and favorable behavior of our method compared to DnCNN-S.

## 6 Conclusions

We introduced lifting layers to be used as an alternative to ReLU-type activation functions in machine learning. Opposed to the classical ReLU, liftings have a nonzero derivative almost everywhere, and can - when combined with a fully connected layer - represent any continuous piecewise linear function. We demonstrated several advantageous properties of lifting and used this technique to handle non-convex and partly flat loss functions. Based on our numerical experiments in image classification and image reconstruction, lifting layers are an attractive building block in various neural network architectures and allowed us to improve on the performance of corresponding ReLU-based architectures.

## A Vector-Valued Lifting

Notation for a Triangulation. For a non-empty, connected, and compact set Ω ⊂ R d , we consider a (non-degenerate) triangulation ( T l ) l M =1 of Ω, where T l is the convex hull of d + 1 vertices ( V κ l (1) , . . . , V κ l ( d +1) ) from the set V := { V 1 , . . . , V L } of all vertices, and κ l : { 1 , . . . , d +1 } → { 1 , . . . , L } maps indices of the vertices of T l to the corresponding indices in V . The notation is illustrated in Figure 2.

## A.1 Definition

Definition 11 (Vector-Valued Lifting). We define the lifting of a variable x ∈ Ω ⊂ R d from the d -dimensional vector space R d with respect to an orthogonal basis E := { e 1 , . . . , e L } of R L and a triangulation ( T l ) l M =1 ⊂ R d as a mapping glyph[lscript] : Ω → R L defined by

<!-- formula-not-decoded -->

where λ l i ( x ), i = 1 , . . . , d +1, are the barycentric coordinates of x with respect to V κ l (1) , . . . , V κ l ( d +1) . The inverse mapping glyph[lscript] † : R L → R d is given by

<!-- formula-not-decoded -->

Example 12 (Scalar-Valued Lifting). For d = 1, we obtain the scalar-valued lifting with Ω = [ t, t ], V = { t 1 , . . . , t L } , and the vertices of T l are exactly the interval borders V κ l (1) = t l and V κ l (2) = t l +1 for l = 1 , . . . , M with M = L -1.

Example 13. For Ω = [ t, t ] d , a regular grid on the rectangular domain in R d , a natural triangulation is induced by the vertices V := [ t 1 , . . . , t L ] d , t = t 1 &lt; . . . &lt; t L = t , which implies a lifted dimension of dL .

Lemma 14 (Sanity Check of Inversion Formula). The mapping glyph[lscript] † inverts the mapping glyph[lscript] , i.e. glyph[lscript] † ( glyph[lscript] ( x )) = x for x ∈ Ω ⊂ R d .

glyph[negationslash]

Proof. For x ∈ T l , using 〈 e l , e k 〉 = 0 for l = k (since E is orthogonal) , the following holds:

<!-- formula-not-decoded -->

where the last equality uses the definition of barycentric coordinates.

## A.2 Analysis

Proposition 15 (Prediction of a Continuous Piecewise Linear Functions). The composition of a fully connected layer z ↦→ θz with θ ∈ R r × L , r ∈ N , and a lifting layer, i.e.

<!-- formula-not-decoded -->

yields a continuous piecewise linear (PLC) function. Conversely, any PLC function with kinks on a triangulation of Ω can be expressed in the form of (18).

Proof. For x ∈ T l , we have:

<!-- formula-not-decoded -->

Since λ l i ( x ) is linear, the expression on the right coincides with the linear interpolation between the points θe κ l ( i ) , i = 1 , . . . , d +1. Continuity follows by continuity of the expression above at the boundary of T l , for each l = 1 , . . . , M .

The converse statement follows by defining the lifting with respect to the same triangulation as the given PLC function and choosing θ such that N θ coincides with that function on the vertices. The details are analogue to the proof of Corollary 19.

Lemma 16 (Approximation by Continuous Piecewise Linear Functions). Let f : Ω → R r , r ∈ N , be a continuous function with the following modulus:

<!-- formula-not-decoded -->

Define the continuous piecewise linear function s f : Ω → R r on the triangulation ( T l ) l M =1 by setting s f ( x ) = f ( x ) at all vertices x ∈ { V 1 , . . . , V L } . We denote by h l M the diameter of T l , given by

<!-- formula-not-decoded -->

and set h M := max l =1 ,...,M h l M , which is finite. Then

<!-- formula-not-decoded -->

and the right hand side vanishes for h M ↘ 0.

Proof. For x ∈ T l , let s f be given by s f ( x ) = ∑ d +1 i =1 λ l i ( x ) f ( V κ l ( i ) ) with λ l i ( x ) ∈ [0 , 1] and ∑ d +1 i =1 λ l i ( x ) = 1. Note that s f is uniquely defined. We conclude:

<!-- formula-not-decoded -->

As Ω is compact, f is uniformly continuous, which, together with ω ( f, h l M ) ≤ ω ( f, h M ), implies that the right hand side vanishes for h M ↘ 0.

Example 17. Consider a (locally) Lipschitz continuous function f : Ω → R r . By compactness of Ω, the function f is actually globally Lipschitz continuous on Ω with a constant m , which implies ω ( f, δ ) ≤ δm , since | f ( x ) -f ( y ) | ≤ m | x -y | .

Corollary 18 (Prediction of Continuous Functions). Any continuous function f : Ω → R r , r ∈ N , can be represented arbitrarily accurate with a network architecture N θ ( x ) = θglyph[lscript] ( x ) for sufficiently large L and θ ∈ R r × L .

Proof. Combine Proposition 15 with Lemma 16.

glyph[negationslash]

Corollary 19 (Overfitting). Let ( x i , y i ) be training data in Ω × R r , i = 1 , . . . , N , x i = x j for i = j . If L = N and V i = x i , there exists θ ∈ R r × L such that N θ ( x ) := θglyph[lscript] ( x ) is exact at all data points x = x i , i.e. N θ ( x i ) = y i , for all i = 1 , . . . , N .

glyph[negationslash]

Proof. Since x i = V i , (17) shows that λ l i ( x ) = 1 and λ l j ( x ) = 0 for j = i . Therefore, we have θglyph[lscript] ( x i ) = θe κ l ( i ) . Denote by E ∈ R L × L the matrix with columns given by e 1 , . . . , e L , and y ∈ R r × L the matrix with columns y 1 , . . . , y L . Since E is a basis, the matrix E is non-singular, and we may determine θ uniquely by solving the following linear system of equations θE = y , which concludes the statement.

Proposition 20 (Convexity of a simple Regression Problem). Let ( x i , y i ) ∈ Ω × R r be training data, i = 1 , . . . , N . Then, the solution of the problem

<!-- formula-not-decoded -->

yields the best continuous piecewise linear fit of the training data with respect to the loss function L . In particular, if L is convex, then (20) is a convex optimization problem.

Proof. Proposition 15 shows that x ↦→ θglyph[lscript] ( x ) is a continuous piecewise linear function. Obviously, θ ↦→ θglyph[lscript] ( x i ) is linear, hence composed with a convex loss function, (20) is a convex optimization problem.

## B Lifting the Output

Lemma 21 (Characterization of the Range of glyph[lscript] ). The range of the mapping glyph[lscript] is given by

<!-- formula-not-decoded -->

and the mapping glyph[lscript] is a bijection between Ω and im( glyph[lscript] ) with inverse glyph[lscript] † .

glyph[negationslash]

Proof. Let z ∈ [0 , 1] L be given by z = ∑ L l =1 z l e l and there exists exactly one index l such that ∑ d +1 i =1 z κ l ( i ) = 1 and, for all k glyph[negationslash]∈ im( κ l ), we have z k = 0. The point x given by x = ∑ d +1 i =1 z κ l ( i ) V κ l ( i ) maps to z via glyph[lscript] . Obviously x ∈ T l , which implies that glyph[lscript] ( x ) = ∑ d +1 i =1 λ l i ( x ) e κ l ( i ) and, by the uniqueness of barycentric coordinates, λ l i = z κ l ( i ) . Moreover (17) implies for k glyph[negationslash]∈ im( κ l ) that z k = 0. We conclude that the set on right hand side of (21) is included in im( glyph[lscript] ). By the definition in (17), it is clear that z = glyph[lscript] ( x ) for x ∈ Ω satisfies the condition for belonging to the set on the right hand side of (21), which implies their equality.

In order to prove the bijection, injectivity remains to show. This is proved as follows: For x, x ′ such that glyph[lscript] ( x ) = glyph[lscript] ( x ′ ), the definition in (17) requires that x, x ′ lie in the same T l , and the property of a basis implies λ l ( x ) = λ l ( x ′ ), which implies that x = x ′ holds. Finally, the proof of glyph[lscript] ( glyph[lscript] † ( z )) = z for z ∈ im( glyph[lscript] ) follows similar arguments as the first part of this proof.

Lemma 22 (Convex Relaxation of the Range of glyph[lscript] ). The set C given by

<!-- formula-not-decoded -->

is the convex hull of im( glyph[lscript] ).

Proof. We make the abbreviation I = im( glyph[lscript] ). Obviously, I ⊂ C and C is convex. Therefore, we need to show that C is the smallest convex set that contains I .

The convex hull conv I of I consists of all convex combinations of points in I . By the characterization of I in (21), it is clear that { e 1 , . . . , e L } ⊂ I . Moreover, C ⊂ conv { e 1 , . . . , e L } ⊂ conv I holds, thus, I ⊂ C already implies that C = conv I , as the convex hull is the smallest convex set containing I .

Proof of Proposition 9. Proposition 9 requires only the 1D-setting of Lemma 21 and 22 above. For convenience of the reader, we copy the statement of Proposition 9 here and proof it.

Proposition 23. Let ( x i , y i ) ∈ [ t, t ] × [ t y , t y ] be training data, i = 1 , . . . , N . Moreover, let glyph[lscript] y be a lifting of the common image [ t y , t y ] of the loss functions L y i , i = 1 , . . . , N , and glyph[lscript] x is the lifting of the domain of L y . Then

<!-- formula-not-decoded -->

is a convex relaxation of the (non-convex) loss function, and the constraints guarantee that θglyph[lscript] x ( x i ) ∈ conv(im( glyph[lscript] y )).

The objective in (14) is linear (w.r.t. θ ) and can be written as

<!-- formula-not-decoded -->

where c := ∑ N i =1 t y glyph[lscript] x ( x i ) glyph[latticetop] , with t y := ( t 1 y , . . . , t L y y ) glyph[latticetop] , is the cost matrix for assigning the loss value t p y to the inputs x i .

Moreover, the closed-form solution of (14) is given for all q = 1 , . . . , L x by θ p,q = 1, if the index p minimizes c p,q , and θ p,q = 0 otherwise.

Proof. (23) is obviously a convex problem, which was generated by relaxing the constraint set im( glyph[lscript] y ) using Lemma 22. Restricting θ to im( glyph[lscript] y ) yields, obviously, a piecewise linear approximation of the true loss L y .

Since z := glyph[lscript] x ( x i ) ∈ im( glyph[lscript] x ) satisfies the condition in (21) and in particular the condition in (22), we conclude that

<!-- formula-not-decoded -->

which shows that θglyph[lscript] x ( x i ) ∈ conv(im( glyph[lscript] y )).

The linearity of the objective in (23) is obvious, and so is (24). Moreover, using the linear expression in (24), clearly, the loss can be minimized by independently minimizing the cost for each q = 1 , . . . , L x , as the constraints couple the variables only along the p -dimension. For each q , the cost is minimized by searching the smallest entry in the cost matrix along the p -dimension, which verifies the closed-form solution of (23).

## References

- [1] H. C. Burger, C. J. Schuler, and S. Harmeling. Image denoising: Can plain neural networks compete with BM3D? In International Conference on Computer Vision (ICCV) , pages 23922399, 2012.
- [2] Y. Chen and T. Pock. Trainable nonlinear reaction diffusion: A flexible framework for fast and effective image restoration. IEEE Transactions on Pattern Analysis and Machine Intelligence , 39(6):1256-1272, 2017.
- [3] D. Clevert, T. Unterthiner, and S. Hochreiter. Fast and accurate deep network learning by exponential linear units (ELUs). Computing Research Repository (CoRR) , abs/1511.07289, 2015.
- [4] G. Cybenko. Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals and Systems , 2(4):303-314, 1989.
- [5] K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian. Image denoising by sparse 3-D transformdomain collaborative filtering. IEEE Transactions on Image Processing , 16(8):2080-2095, Aug. 2007.

- [6] C. Dugas, Y. Bengio, F. B´ elisle, C. Nadeau, and R. Garcia. Incorporating second-order functional knowledge for better option pricing. In Advances in Neural Information Processing Systems (NIPS) , pages 451-457, Cambridge, MA, USA, 2001. MIT Press.
- [7] I. Goodfellow, Y. Bengio, and A. Courville. Deep Learning . The MIT Press, 2016.
- [8] I. Goodfellow, D. Warde-Farley, M. Mirza, A. Courville, and Y. Bengio. Maxout networks. In International Conference on Machine Learning (ICML) , pages 1319-1327, 2013.
- [9] S. Gu, L. Zhang, W. Zuo, and X. Feng. Weighted nuclear norm minimization with application to image denoising. In International Conference on Computer Vision and Pattern Recognition (CVPR) , pages 2862-2869, 2014.
- [10] K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In International Conference on Computer Vision (ICCV) , pages 1026-1034, 2015.
- [11] S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International Conference on Machine Learning (ICML) , pages 448-456, 2015.
- [12] K. Jarrett, K. Kavukcuoglu, M. Ranzato, and Y. LeCun. What is the best multi-stage architecture for object recognition? In International Conference on Computer Vision and Pattern Recognition (CVPR) , pages 2146-2153, 2009.
- [13] G. Klambauer, T. Unterthiner, A. Mayr, and S. Hochreiter. Self-normalizing neural networks. Advances in Neural Information Processing Systems (NIPS) , 2017.
- [14] E. Laude, T. M¨ ollenhoff, M. Moeller, J. Lellmann, and D. Cremers. Sublabel-accurate convex relaxation of vectorial multilabel energies. In B. Leibe, J. Matas, N. Sebe, and M. Welling, editors, European Conference on Computer Vision (ECCV) , pages 614-627, Cham, 2016. Springer International Publishing.
- [15] M. Leshno, V. Lin, A. Pinkus, and S. Schocken. Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. Neural Networks , 6(6):861867, 1993.
- [16] A. Maas, A. Hannun, and A. Ng. Rectifier nonlinearities improve neural network acoustic models. In International Conference on Machine Learning (ICML) , 2013.
- [17] T. M¨ ollenhoff, E. Laude, M. Moeller, J. Lellmann, and D. Cremers. Sublabel-accurate relaxation of nonconvex energies. In International Conference on Computer Vision and Pattern Recognition (CVPR) , 2016.
- [18] G. Mont´ ufar, R. Pascanu, K. Cho, and Y. Bengio. On the number of linear regions of deep neural networks. In Advances in Neural Information Processing Systems (NIPS) , pages 29242932, Cambridge, MA, USA, 2014. MIT Press.
- [19] T. Pock, D. Cremers, H. Bischof, and A. Chambolle. Global solutions of variational models with convex regularization. SIAM Journal on Imaging Sciences , 3(4):1122-1145, 2010.
- [20] U. Schmidt and S. Roth. Shrinkage fields for effective image restoration. In International Conference on Computer Vision and Pattern Recognition (CVPR) , pages 2774-2781, 2014.
- [21] B. Xu, N. Wang, T. Chen, and M. Li. Empirical evaluation of rectified activations in convolutional network. In International Conference on Machine Learning (ICML) , 2015. Deep Learning Workshop.

- [22] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE Transactions on Image Processing , 26(7):31423155, 2017.
- [23] D. Zoran and Y. Weiss. From learning models of natural image patches to whole image restoration. In International Conference on Computer Vision (ICCV) , pages 479-486, 2011.