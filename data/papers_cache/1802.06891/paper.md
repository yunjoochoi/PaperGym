## Fourier Policy Gradients

Matthew Fellows * 1 Kamil Ciosek * 1 Shimon Whiteson 1

## Abstract

We propose a new way of deriving policy gradient updates for reinforcement learning. Our technique, based on Fourier analysis, recasts integrals that arise with expected policy gradients as convolutions and turns them into multiplications. The obtained analytical solutions allow us to capture the low variance benefits of EPG in a broad range of settings. For the critic, we treat trigonometric and radial basis functions, two function families with the universal approximation property. The choice of policy can be almost arbitrary, including mixtures or hybrid continuous-discrete probability distributions. Moreover, we derive a general family of sample-based estimators for stochastic policy gradients, which unifies existing results on sample-based approximation. We believe that this technique has the potential to shape the next generation of policy gradient approaches, powered by analytical results.

## 1. Introduction

Policy gradient methods, also known as actor-critic methods, are an effective way to perform reinforcement learning in large or continuous action spaces (Lillicrap et al., 2015; Schulman et al., 2015; 2017; Wu et al., 2017; Peters &amp; Schaal, 2008; Sutton et al., 1999; Williams, 1992). Since they adjust the policy in small increments, they do not have to perform expensive optimisations over the action space, in contrast to methods based on value functions, such as Q -learning (Mnih et al., 2015; Silver et al., 2017; van Hasselt et al., 2015) or SARSA (van Seijen et al., 2009; Sutton &amp; Barto, 1998; Sutton, 1996). Moreover, they are naturally suited to stochastic policies, which are useful for exploration and necessary to achieve optimality in some settings, e.g., competitive multi-agent systems.

* Equal contribution 1 Department of Computer Science, University of Oxford, United Kingdom. Correspondence to: Matthew Fellows &lt; matthew.fellows@cs.ox.ac.uk &gt; .

Proceedings of the 35 th International Conference on Machine Learning , Stockholm, Sweden, PMLR 80, 2018. Copyright 2018 by the author(s).

Until recently, policy gradient methods were either restricted to deterministic policies (Silver et al., 2014) or suffered from high variance (Sutton et al., 1999). The latter problem is exacerbated in large state spaces, when the number of samples required to reduce the variance of the gradient estimate becomes infeasible for the simple score function estimators on which policy gradient methods typically rely. The problem also arises when training recurrent neural networks (RNNs) (Xu et al., 2015; Ba et al., 2015; Sukhbaatar et al., 2016) that have to be unrolled over several timesteps, each adding to the overall variance, and in multi-agent settings (Foerster et al., 2017), where the actions of other agents introduce a compounding source of variance.

Recently, a new approach called expected policy gradients (EPG) (Ciosek &amp; Whiteson, 2018a;b) was proposed that eliminates the variance of a stochastic policy gradient by integrating over the actions analytically. However, this requires analytic solutions to the policy gradient integral and the original work addressed only polynomial critics.

In this paper, we employ techniques from Fourier analysis to derive analytic policy gradient updates for two important families of critics. The first, radial basis functions (RBFs), combines the benefits of shallow structure, which makes them tractable, with an impressive empirical track record (Buhmann, 2003; Carr et al., 2001). The second, trigonometric critics, is useful for modelling periodic phenomena. Similarly to polynomial critics (Ciosek &amp; Whiteson, 2018b), these function classes are universal, i.e., they can approximate an arbitrary function on a bounded interval.

Furthermore, to address cases where analytical solutions are infeasible, we provide a general theorem for deriving Monte Carlo estimators that links existing methods using the first and second derivatives of the action-value function, relating it to existing sampling approaches.

Our technique also enables analytic solutions for new families of policies, extending EPG to any policy that has unbounded support, where it previously required the policy to be in an exponential family. We also develop results for mixture policies and hybrid discrete-continuous policies, which we posit can be useful in multi-agent settings, where having a rich class of policies is important not just for exploration but also for optimality (Nisan et al., 2007).

Overall, we believe that the techniques developed in this paper can be used to shape the next generation of policy gradient methods suitable for any reasonable MDP and that, powered by analytical results, achieve zero or low variance. Moreover, our methods elucidate the way policy gradients work by explicitly stating the expected update. Finally, while the main contribution of this paper is theoretical, we also provide an empirical evaluation using a periodic critic on a simple turntable problem that demonstrates the practical benefit of using a trigonometric critic.

## 2. Background

Reinforcement learning (RL) aims to learn optimal behaviour policies for an agent (or many agents) acting in an environment with a scalar reward signal. Formally, we consider a Markov decision process, defined as a tuple ( S, A, R, p, p 0 , γ ) . An agent has an environmental state s ∈ S = R n ; takes a sequence of actions a 1 , a 2 , ... , where a t ∈ A ; transitions to the next state s ′ ∼ p ( ·| s , a ) under the state transition distribution p ( s ′ | s , a ) ; and receives a scalar reward r ∈ R . The agent's initial state s 0 is distributed as s 0 ∼ p 0 ( · ) .

The agent samples from the policy β to generate actions a ∼ β ( ·| s ) , giving a trajectory through the environment τ = ( s 0 , a 0 , r 1 , s 1 , a 1 , r 1 , ... ) . The definition of the value function is V β ( s ) = E τ : s 0 = s [ ∑ t γ t r t ] and action-value function is Q β ( s , a ) = E τ : s 0 = s ,a 0 = a [ ∑ t γ t r t ] , where γ ∈ [0 , 1) is a discount factor. An optimal policy β glyph[star] maximises the total return J = ∫ s V β glyph[star] ( s ) dp o ( s ) .

## 2.1. Policy Gradient Methods

Policy gradient methods seek a locally optimal policy by maintaining a critic , learned using a value-based method, and an actor , adjusted using a policy gradient update.

The critic ˆ Q is learned using variants of SARSA (van Seijen et al., 2009; Sutton &amp; Barto, 1998; Sutton, 1996), with the goal of approximating the true action-value Q β . Meanwhile, the actor adjusts the policy parameter vector θ of the policy β θ with the aim of maximising J . For stochastic policies, this is done by following the gradient:

<!-- formula-not-decoded -->

where ρ ( s ) glyph[defines] ∑ ∞ t =0 γ t p ( s t = s | s 0 ) is the discountedergodic occupancy measure. The outer integral can be approximated by following a trajectory of length T through the environment, yielding:

<!-- formula-not-decoded -->

| Algorithm 1 Expected Policy Gradient   | Algorithm 1 Expected Policy Gradient                 |
|----------------------------------------|------------------------------------------------------|
| 1:                                     | s ← s 0 , t ← 0                                      |
| 2:                                     | initialise optimiser, initialise policy parameters θ |
| 3:                                     | while not converged do                               |
| 4:                                     | g t ← γ t ˆ I θ ( s )                                |
| 5:                                     | θ ← θ + optimiser.UPDATE ( g t )                     |
| 6:                                     | a ∼ β ( · , s )                                      |
| 7:                                     | s ′ , r ← environment.PERFORM-ACTION(a)              |
| 8:                                     | ˆ Q .UPDATE( s, a, r, s ′ )                          |
| 9:                                     | t ← t +1                                             |
| 10:                                    | s ← s ′                                              |
| 11:                                    | end while                                            |

where ˆ I θ is the integral of (1) but with the critic ˆ Q in place of the unknown true Q -function:

<!-- formula-not-decoded -->

The subscript of ˆ I θ denotes the fact that we are differentiating with respect to θ . Now, since ˆ Q , unlike Q , does not depend on the policy parameters, we can move the differentiation out of the inner integral as follows:

<!-- formula-not-decoded -->

This transformation has two benefits: it allows for easier manipulation of the integral and it also holds for deterministic policies, where β is a Dirac-delta measure (Silver et al., 2014; Ciosek &amp; Whiteson, 2018a;b).

Using (2) directly with an analytic value of ˆ I ( s t ) yields expected policy gradients (EPG) 1 (Ciosek &amp; Whiteson, 2018a;b), shown in Algorithm 1. If instead we add an additional Monte Carlo sampling step:

<!-- formula-not-decoded -->

we get the original stochastic policy gradients (Williams, 1992; Sutton et al., 1999). In place of (5), alternative Monte Carlo schemes with better variance properties have also been proposed (Baxter &amp; Bartlett, 2000; Baxter et al., 2001; Baxter &amp; Bartlett, 2001; Gu et al., 2016; Kakade et al., 2003). If we can compute the integral in (3), then EPG is preferable since it avoids the variance introduced by the Monte Carlo step of (5).

This paper considers both methods for solving integrals of the form in (3) and Monte Carlo methods that improve on (5) for cases where analytical solutions are not possible.

1 Called all-action policy gradient in an unpublished draft by Sutton et al. (2000).

Above, we used the symbol θ to denote a generic policy parameter. Often, the policy is described by its moments (for instance a Gaussian is fully defined by its mean and covariance). To achieve greater flexibility, these immediate parameters are obtained by a complex function approximator, such as a neural network, where the state vector is the input, and parameterised by w . The total policy gradient for w is then obtained by using the chain rule. For example, for a Gaussian we have immediate parameters µ , Σ and the parameterisation is:

<!-- formula-not-decoded -->

where net w is a neural network parameterised by the vector w . The gradient for some w is then:

<!-- formula-not-decoded -->

For clarity, we only give updates for the immediate parameters (in this case, ˆ I µ and ˆ I Σ 1 / 2 ) in the remainder of the paper, without explicitly mentioning w .

## 2.2. Fourier Analysis

A convolution f ∗ g is an operation on two functions that returns another function, defined as:

<!-- formula-not-decoded -->

Convolutions have convenient analytical properties that we use to derive our main result. To make convolutions easy to compute, we seek a transform F that, when applied to a convolution, yields a simple operation like multiplication, i.e., we want the property:

<!-- formula-not-decoded -->

We also need the dual property:

<!-- formula-not-decoded -->

to ensure symmetry between the space of functions and their transforms. It turns out that, up to scaling, there is only one transform that meets our needs (Alesker et al., 2008), the Fourier transform :

<!-- formula-not-decoded -->

The two sets of parentheses on the lefthand side are required because the Fourier transform F ( f ) is a function, not a scalar, and F ( f ) ( ω ) is the result of evaluating this function on ω . The Fourier transform of a probability density function is known as the characteristic function of the corresponding distribution. An intuitive interpretation of the Fourier transforms is that it provides a mapping from the action-spatial domain to the frequency domain, F ( f ( x )) : x → ω , decomposing the function f into its frequency components. Consider, as a simple example, a univariate sinusoidal function, f ( x ) = cos( x Ω) . The Fourier transform of f can be easily shown to be F ( f ) = πδ ( ω -Ω) + πδ ( ω + Ω) (Stein &amp; Shakarchi, 2003); the Fourier transform has mapped a sinusoid of frequency Ω in the action domain to a double frequency spike at ± Ω in the frequency domain.

The Fourier transform has another related intuitive interpretation as a change of basis in the space of functions. The Fourier basis functions e -i ω glyph[latticetop] x make analytical operations convenient in much the same way as a choice of convenient basis in linear algebra makes certain matrix operations easier. Since the basis functions are periodic, the Fourier transform can also be viewed as a decomposition of the original function into cycles. Sometimes these cycles are written explicitly when the complex exponential e -i ω glyph[latticetop] x is expressed in polar form, which includes sines and cosines. Indeed, the Fourier series , which we briefly discuss in Appendix A, can be used to prove that any function on a bounded interval can be approximated arbitrarily well with a sum of sufficiently many such trigonometric terms.

The inverse Fourier transform is defined as:

<!-- formula-not-decoded -->

which has the property that, for any function f ,

<!-- formula-not-decoded -->

Thus, we can recover the original function by applying the Fourier and inverse Fourier transforms. Just as the Fourier transform maps from the action domain to the frequency domain, the inverse Fourier transforms provides a mapping from the frequency domain back to the action-spatial domain F -1 ( f ( ω )) : ω → x . The Fourier transform also turns differentiation into multiplication:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ∇ ( m ) x f denotes the m th order derivative of f w.r.t. x ∀ m ≥ 0 .

We formalise the n -dimensional Fourier transform in Appendix B, and provide definitions for Fourier transforms of matrix and vector quantities. We also derive the differentiation/multiplication property in n -dimensional space.

## 3. Main Result

In this section, we prove our main result. The motivating factor behind these derivations is that by viewing the inner integral ˆ I θ as a convolution , we can analyse our policy gradient in the frequency domain. This affords powerful analytical results that enable us to exploit the multiplication/derivative property of Fourier transforms, namely manipulation of expressions involving the derivatives of our critic ˆ Q in the action-spatial domain are represented simply by factors of ( i ω ) in the frequency domain. We apply this elegant property in Section 4.1 to demonstrate the relative ease of manipulation of the inner integral ˆ I θ . In Section 4.2, we show that existing Monte Carlo policy gradient estimators arise from our theorem as a single family of cases using different factors ( i ω ) multiplied with the critic ˆ Q .

Moreover, our theorems rely only on the characteristic function F ( ˜ β ) . While the original technique developed for EPG (Ciosek &amp; Whiteson, 2018b) relies on the moment generating function to obtain ˆ I for a policy from an exponential family and a polynomial family of critics, we require only that both the policy PDF and the critic have a closed form Fourier transform (Karr, 1993). For policies, this condition is easy to satisfy since almost all common distributions have a closed form characteristic function.

Theorem 1 (Fourier Policy Gradients) . Let ˆ I θ ( s t ) = ∇ θ ∫ a ˆ Q ( s t , a ) β θ ( a | s t ) d a be the inner integral of the policy gradient for a critic ˆ Q ( a ) and policy β ( a ) with auxiliary policy ˜ β ( µ -a ) = β ( a ) . We may write ˆ I θ ( s t ) as:

<!-- formula-not-decoded -->

Proof. Recall the definition of ˆ I θ ( s t ) from (4):

<!-- formula-not-decoded -->

To exploit the convolution property of Fourier transforms given by (6), the first step is to introduce an auxiliary policy ˜ β , so that the above integral becomes a convolution. If the mean of the policy β is µ , the new auxiliary policy ˜ β is:

<!-- formula-not-decoded -->

We start by rewriting ˆ I θ as:

<!-- formula-not-decoded -->

Now, we apply the Fourier transform to the convolution and use (6) to reduce it to a multiplication:

<!-- formula-not-decoded -->

Taking the inverse Fourier transform gives:

<!-- formula-not-decoded -->

Substituting this into (11) yields our main result:

<!-- formula-not-decoded -->

We now derive a variant of our main theorem for the special case of µ .

Theorem 2 (Fourier Policy Gradients for µ ) . Let ˆ I µ ( s t ) = ∇ µ ∫ a ˆ Q ( s t , a ) β θ ( a | s t ) d a be the inner integral of the policy gradient for µ with a critic ˆ Q ( a ) and policy β ( a ) with auxiliary policy ˜ β ( µ -a ) = β ( a ) . We may write ˆ I µ ( s t ) as:

<!-- formula-not-decoded -->

Proof. We return to (11), retaining the derivative inside the integral:

<!-- formula-not-decoded -->

From the chain rule, we substitute ∇ µ ( ˜ β ( µ -a ) ) = ( ∇ ˜ β ) ( µ -a ) , yielding:

<!-- formula-not-decoded -->

Now, we take the Fourier transforms of the convolution ( ˆ Q ∗ ∇ ˜ B )( µ ) and exploit the multiplication property of (6):

<!-- formula-not-decoded -->

Using the multiplication/derivative property from (8), we substitute for F ( ∇ ˜ β ) = i ω F ( ˜ β ) :

<!-- formula-not-decoded -->

Finally, taking inverse Fourier transforms and substituting into (13) yields our result:

<!-- formula-not-decoded -->

We use Theorem 2 to derive the following corollary, valid for all parameters ψ s.t. µ does not depend upon them.

Corollary 2.1. Let ψ be a parameter that does not depend upon µ . We can write ˆ I ψ ( s t ) = ∇ ψ ∫ a ˆ Q ( s t , a ) β θ ( a | s t ) d a as:

<!-- formula-not-decoded -->

The required auxiliary policy ˜ β ( a ) = β ( µ -a ) exists for all distributions β with unbounded support. For symmetric distributions, ˜ β often has a convenient form, e.g., for a Gaussian policy β = N ( µ , Σ ) , ˜ β = N ( 0 , Σ ) . This transformation is similar to reparameterisation (Heess et al., 2015). For critics, we discuss tractable critic families in the remainder of the paper.

## 4. Applications

We now discuss a number of specialisations of (10) and (12), linking them to several established policy gradient approaches.

## 4.1. Frequency Domain Analysis

We now motivate the remainder of this section by considering the Gaussian policy ˜ β = N ( 0 , Σ ) . We need to calculate the gradient w.r.t. Σ 1 2 where ( Σ 1 2 ) glyph[latticetop] Σ 1 2 = Σ . From the characteristic function for a multivariate Gaussian, F ( N ( 0 , Σ )) = e -1 2 ω glyph[latticetop] Σ ω , we find derivatives as:

<!-- formula-not-decoded -->

Substituting for ∇ ψ = Σ 1 2 F ( ˜ β ) in (14) gives the gradient for Σ 1 2 . For completeness, we also include the update for µ which, recall from (12), is the same for all policies with auxiliary function ˜ β ( a ) = β ( µ -a ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We see from (8) and (9) that the terms i ω , once pulled into the Fourier transform, become differentiation operators. However, (15) and (16) afford us a choice - we can pull them into the critic term or the policy term. This gives rise to a number of different expressions for the gradient. To differentiate between methods, we define the order of the method, denoted by M , the order of the derivative with respect to the critic.

We continue our example of Gaussian policies, using (15) to compute an update for µ for M ∈ { 0 , 1 } and (16) to compute an update for Σ 1 / 2 for M ∈ { 0 , 1 , 2 } . Full derivations with Gaussian derivatives can be found in Appendix E.

Zeroth Order Method ( M =0 ) Using (15) and (16) in their current form gives an analytic expression for a zeroth order critic, as we do not multiply F ( ˆ Q ) by any factor of i ω . Using results for multidimensional Fourier transforms from (8) and (9) when taking these inverse transforms, we obtain:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, we use the identities ∇ ˜ β = -∇ β and ∇ (2) ˜ β = ∇ (2) β from Lemma 3.

First Order Method ( M = 1 ) To obtain an analytic expression in terms of ∇ a ˆ Q , we must manipulate the factors of ( i ω ) in (15) and (16) to obtain a factor of ( i ω ) ˆ Q . We then exploit the multidimensional Fourier transform result for vectors from (8) as before:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Second Order Method ( M =2 ) We repeat the process, this time taking the derivative of ˆ Q twice:

<!-- formula-not-decoded -->

Here, we exploit the multidimensional Fourier transform result for matrices from (9) in deriving the second line.

## 4.2. Family of SPG Estimators

We are going to revisit certain integrals from Section 4.1 using the following rule for deriving Monte Carlo estimators:

<!-- formula-not-decoded -->

Here, the quantity on the right is a sample-based approximation. We have the following approximations for the integrals given by equations (17,18,19,20,21), which recall were defined for a Gaussian policy β = N ( µ , Σ ) :

<!-- formula-not-decoded -->

The above equations summarise existing results for stochastic policy gradients estimators, which are applicable for any policy and critic. The zeroth-order results ( M = 0 ) correspond to standard policy gradient methods (Sutton et al., 1999; Williams, 1992; Glynn, 1990); the first-order ones ( M = 1 ) correspond to reparameterisation-based methods (Heess et al., 2015; Kingma &amp; Welling, 2013); and, when applied to a Gaussian policy, the second-order ( M = 2 ) of the update for Σ is a sample-based version of Gaussian policy gradients (Ciosek &amp; Whiteson, 2018a), a special case of EPG. Note that interpolations between different estimators can also be used (Gu et al., 2016) as a method of reducing variance further. The full derivations for of the derivatives for the multivariate Gaussian are given in Appendix E.

## 4.3. Periodic Action Spaces

In some settings, the action space of an MDP is naturally periodic, e.g., when actions specify angles. By using a trigonometric function in the critic, we encode the insight that rotating by -π and by π leads to similar results, despite the fact that the two points lie on the opposite ends of the action range.

Consider the case where the policy is Gaussian, i.e., β = N ( µ , Σ ) and the critic ˆ Q is a trigonometric function of the form

<!-- formula-not-decoded -->

where f ∈ R n , h ∈ R , and n is the dimension of the action space.

While a policy gradient method involving a critic ˆ Q of this form superficially resembles approximating the value function with the Fourier basis (Konidaris et al., 2011) for the state space, it is in fact completely different. Indeed, our method uses a Fourier basis to approximate a function of the action space , not the state space , which often has different structure. The dependence of ˆ Q on the state can still be completely arbitrary (for example a neural network).

We seek to find the policy gradient update for this combination of critic and policy. First, we write out their Fourier transforms:

<!-- formula-not-decoded -->

Computing the inverse Fourier transform yields:

<!-- formula-not-decoded -->

A more detailed derivation of (22) can be found in appendix G. We now use (10) to obtain the policy gradients for the mean and the covariance.

<!-- formula-not-decoded -->

Intuitively, the mean update contains a frequency damping component e -1 2 f glyph[latticetop] Σ f , which is small for large f , ensuring that the optimisation slows down when the signal is repeating frequently. The covariance update uses the same damping, while also making sure that exploration increases in the minima of the critic and decreases near the maxima, in a way slightly similar, but mathematically different, from Gaussian policy gradients (Ciosek &amp; Whiteson, 2018a;b).

We evaluated a periodic critic of this form on a toy turntable domain where the goal is to rotate a flat record to the desired position by rotating it (see Appendix D for details). We compared it to the DPG baseline from OpenAI (Dhariwal et al., 2017), which uses a neural network based critic capable of addressing complex control tasks. As expected, the learning curves in Figure 1 show that using a periodic critic (F-EPG) leads to faster learning, because it encodes more information about the action space than a generic neural network. Our method efficiently uses this information in the policy gradient framework by deriving an exact policy gradient update.

<!-- image -->





of the derivative of the logarithm.

<!-- formula-not-decoded -->

⊔∖〉∑⊔({(√√{√}

⊗⊗∖〈∑⊔({(√√{√}





Figure 1. Learning curves for Turntable. EPG with periodic critic (F-EPG) vs. DPG with a neural network critic (NN-DPG).

## 4.4. Policy Gradients with Radial Basis Functions

Radial basis functions (RBFs) (Buhmann, 2003) have a long tradition as function approximators in machine learning and combine a simple, tractable structure with the universal approximation property (Park &amp; Sandberg, 1991). In this section, we analyse the elementary RBF building block - a single RBF node. Results on combining many such blocks are deferred to Section 4.6.

Consider the setting where the policy is Gaussian, i.e., β = N ( µ , Σ ) and the critic is an RBF ˆ Q = N ( l , S ) . Although the critic ˆ Q has the shape of a Gaussian PDF, it is not a random variable but simply an ordinary function parametrised by the location vector l and the positivedefinite scale matrix S , which occupy the place of the mean and the covariance. We want to find the policy gradient updates for the mean and the covariance. We begin the derivation by writing out the Fourier transforms for the policy and the critic:

<!-- formula-not-decoded -->

The inverse Fourier transform has the following form:

<!-- formula-not-decoded -->

Now, we substitute a = µ and introduce the notation:

<!-- formula-not-decoded -->

Wenowderive the policy gradients using (10) and properties The RBF Policy gradient simply minimises the Mahalanobis distance with the weight matrix ( Σ + S ) -1 . Also, since E is a positive scalar, for multi-dimensional action spaces, the multiplication by E in the gradients does not change the gradient direction, only the magnitude.

For the mean update ∇ µ , this result is intuitive - if we want our policy to reach the maximum of the RBF node (i.e., a bump) we simply minimise the distance between the current policy mean and the top of the bump. We now provide an additional variant of this result, based on natural policy gradients. The Fisher matrix for the Gaussian distribution parameterised by µ (with the covariance kept constant) is simply Σ , yielding the following update:

<!-- formula-not-decoded -->

Here, the symbol Id denotes the identity matrix. The update given by g natural can be used in place of I µ to obtain a natural policy gradient method. Moving from a standard first order policy gradient to the natural policy gradient is simply a change in the weighting matrix of the Mahalanobis distance from ( Σ + S ) -1 to ( S Σ -1 + Id ) -1 . Furthermore, the Mahalanobis distance reduces to the unweighted L 2 distance when S = Σ . Intuitively, since the natural policy gradient takes the geometry of the space of distributions into account, a simpler update is obtained if this geometry is the same as the geometry of the RBF (as given by S ).

## 4.5. Revisiting Gaussian Policy Gradients

In this section, we revisit Gaussian policy gradients (Ciosek &amp;Whiteson, 2018a) with the aim of contrasting it with the RBF derivation presented above. Gaussian policy gradients assume that the policy is Gaussian, i.e., β = N ( µ , Σ ) , and the critic is quadric, i.e.,

<!-- formula-not-decoded -->

Here we denote by 'const' some constant which always exists so that the above equality holds for l = -1 2 H -1 b .

In this setting we have that,

<!-- formula-not-decoded -->

Now, we compute the policy gradient for the mean:

<!-- formula-not-decoded -->

This is almost the same as (24), except for a positive scaling factor and the fact that, H need not be positive-definite (unlike the matrix S from the definition of the RBF.). This illustrates both the similarities and differences between Gaussian policy gradients, which uses quadrics, and the updates given by (24), which is based on RBFs. The similarity is that we are minimising a quadratic form, while the difference is that the quadratic form used by RBF comes from a more restrictive family (i.e., it has to be positive definite). However, RBFs also have some advantages over quadrics in that they are bounded both from above and below.

## 4.6. Hybrid Critics

We now consider the case when the critic ˆ Q is a linear combination, i.e., ˆ Q ( a ) = ∑ i c i ˆ Q i ( a ) for some c i . The main observation is that the integral ˆ I is linear in the critic, i.e., for any parameter θ we have that,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, we can compute the policy gradient update for each component of the critic separately, and then use a linear combination of the updates. If each of these components is in a tractable family, such as a trigonometric function (Section 4.3), an RBF (Section 4.4) or a polynomial (Ciosek &amp; Whiteson, 2018b), the whole update is also tractable. In this way, we can use critics consisting not just of of a superposition of functions from a single family (like a Fourier basis, which consists of different trigonometric functions), but also hybrid ones, combining functions from many families.

All these three categories of critics have their corresponding universal approximation result, implying that a linear combination of a sufficient number of functions from that class alone is rich enough to approximate any reasonable function on a bounded interval to arbitrary accuracy. Indeed, we have the Weierstrass theorem about linear combinations of monomials (Weierstrass, 1885; Stone, 1948), the result by Park &amp; Sandberg (1991) for linear combinations of RBFs, and the Fourier series approximation for linear combinations of trigonometric functions (see Appendix A).

These results show that, in principle, we can have analytic updates for a critic matching any Q -function, and hence any MDP, without the need for Monte Carlo sampling schemes similar to (5), with no sampling noise (given the state) and with virtually no computational overhead relative to stochastic policy gradients. However, there remain two obstacles.

First, for a finite number of basis functions, the approximation may introduce spurious local minima that are harmful to any local optimisation method. Second, even when local minima are not a problem, there is a case for using a degree of sampling in case we believe that our critic ˆ Q is biased - some sampling methods allow the use of direct reward rollouts to address bias. We believe that the practical impact of our analytic results and the question of which critic combination to use is yet to be determined.

## 4.7. Mixture Policies and Other Nonstandard Policies

We now consider mixture policies of the form β ( a ) = ∑ i b ′ i β i ( a ) , where b ′ i ≥ 0 and ∑ i b ′ i = 1 . Similarly to the previous section, we use the linearity of the integral:

<!-- formula-not-decoded -->

The two most common types of components for the policy are Gaussian and the deterministic policy (i.e., a Dirac-delta measure). Hence, using (25), we can obtain a policy gradient method for policies that have several modes (modelled with Gaussians) as well as several focussed (discrete) points. Of course, we can also use any other distribution with a characteristic function by substituting into (10). We believe that such policies can be particularly useful in multi-agent settings, where the concept of finding a maximum of the total expected return generalises to finding a Nash equilibrium and it is known that some Nash equilibria admit only stochastic policies (Nisan et al., 2007). It is also possible to have both a mixture policy and a hybrid critic. We do not give the formula, since it is straightforward to derive.

## 5. Conclusions

This paper developed new theoretical tools for deriving policy gradient updates, showing that expected policy gradients are tractable in three important classes of critics and for almost all policies. We also discussed a framework for deriving estimators for stochastic policy gradients, which generalises existing approaches. Moreover, we addressed the setting of MDPs with periodic action spaces and described an experiment demonstrating the benefits of explicitly modelling periodicity in a policy gradient method.

## Acknowledgements

This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement number 637713), and the Engineering and Physical Sciences Research Council (EPSRC).

## References

- Alesker, Semyon, Artstein-Avidan, Shiri, and Milman, Vitali. A characterization of the fourier transform and related topics. Comptes Rendus Mathematique , 346(11-12): 625-628, 2008.
- Ba, Jimmy Lei, Mnih, Volodymyr, and Kavukcuoglu, Koray. Multiple Object Recognition With Visual Attention. Iclr , pp. 1-10, 2015.
- Baxter, Jonathan and Bartlett, Peter L. Direct gradientbased reinforcement learning. In Circuits and Systems, 2000. Proceedings. ISCAS 2000 Geneva. The 2000 IEEE International Symposium on , volume 3, pp. 271-274. IEEE, 2000.
- Baxter, Jonathan and Bartlett, Peter L. Infinite-horizon policy-gradient estimation. Journal of Artificial Intelligence Research , 15:319-350, 2001.
- Baxter, Jonathan, Bartlett, Peter L, and Weaver, Lex. Experiments with infinite-horizon, policy-gradient estimation. Journal of Artificial Intelligence Research , 15:351-381, 2001.
- Buhmann, Martin D. Radial basis functions: theory and implementations , volume 12. Cambridge university press, 2003.
- Carr, Jonathan C, Beatson, Richard K, Cherrie, Jon B, Mitchell, Tim J, Fright, W Richard, McCallum, Bruce C, and Evans, Tim R. Reconstruction and representation of 3d objects with radial basis functions. In Proceedings of the 28th annual conference on Computer graphics and interactive techniques , pp. 67-76. ACM, 2001.
- Ciosek, Kamil and Whiteson, Shimon. Expected Policy Gradients. The Thirty-Second AAAI Conference on Artificial Intelligence (AAAI-18) , 2018a.
- Ciosek, Kamil and Whiteson, Shimon. Expected Policy Gradients for Reinforcement Learning. Journal submission, arXiv preprint arXiv:1801.03326 , 2018b.
- Dhariwal, Prafulla, Hesse, Christopher, Klimov, Oleg, Nichol, Alex, Plappert, Matthias, Radford, Alec, Schulman, John, Sidor, Szymon, and Wu, Yuhuai. Openai baselines. https://github.com/openai/ baselines , 2017.
- Foerster, Jakob, Farquhar, Gregory, Afouras, Triantafyllos, Nardelli, Nantas, and Whiteson, Shimon. Counterfactual Multi-Agent Policy Gradients. pp. 1-12, 2017. URL http://arxiv.org/abs/1705.08926 .
- Glynn, Peter W. Likelihood ratio gradient estimation for stochastic systems. Communications of the ACM
- (Association of Computing Machinery) , 33(10):75-84,
1990. [ISSN 00010782. doi: 10.1145/84537.84552. URL http://portal.acm.org/citation. cfm?id=84552{%}5Cnhttp://portal.acm. org/ft{\_}gateway.cfm?id=84552{&amp;}type= pdf{&amp;}coll=GUIDE{&amp;}dl=GUIDE{&amp;}CFID= 92520986{&amp;}CFTOKEN=53025364 .](http://portal.acm.org/citation.cfm?id=84552{%}5Cnhttp://portal.acm.org/ft{_}gateway.cfm?id=84552{&}type=pdf{&}coll=GUIDE{&}dl=GUIDE{&}CFID=92520986{&}CFTOKEN=53025364)
- Gu, Shixiang, Lillicrap, Timothy, Ghahramani, Zoubin, Turner, Richard E., and Levine, Sergey. Q-Prop: SampleEfficient Policy Gradient with An Off-Policy Critic. pp. 1-13, 2016. URL http://arxiv.org/abs/1611. 02247 .
- Heess, Nicolas, Wayne, Greg, Silver, David, Lillicrap, Timothy, Tassa, Yuval, and Erez, Tom. Learning Continuous Control Policies by Stochastic Value Gradients. pp. 1-13, 2015. ISSN 10495258. URL http://arxiv.org/ abs/1510.09142 .
- Kakade, Sham Machandranath et al. On the sample complexity of reinforcement learning . PhD thesis, University of London London, England, 2003.
- Karr, Alan F. Characteristic Functions , pp. 163-182. Springer New York, New York, NY, 1993. ISBN 978-14612-0891-4. doi: 10.1007/978-1-4612-0891-4 7.
- Kingma, Diederik P and Welling, Max. Auto-Encoding Variational Bayes PPT. Ppt , 2013. ISSN 1312.6114v10. URL http://arxiv.org/abs/1312.6114 .
- Konidaris, George, Osentoski, Sarah, and Thomas, Philip. Value Function Approximation in Reinforcement Learning using the Fourier Basis. Proceedings of the TwentyFifth Conference on Artificial Intelligence , pp. 380-385, 2011.
- Lillicrap, Timothy P, Hunt, Jonathan J, Pritzel, Alexander, Heess, Nicolas, Erez, Tom, Tassa, Yuval, Silver, David, and Wierstra, Daan. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971 , 2015.
- Mnih, Volodymyr, Kavukcuoglu, Koray, Silver, David, Rusu, Andrei A., Veness, Joel, Bellemare, Marc G., Graves, Alex, Riedmiller, Martin, Fidjeland, Andreas K., Ostrovski, Georg, Petersen, Stig, Beattie, Charles, Sadik, Amir, Antonoglou, Ioannis, King, Helen, Kumaran, Dharshan, Wierstra, Daan, Legg, Shane, and Hassabis, Demis. Human-level control through deep reinforcement learning. Nature , 518(7540):529-533, 2015. ISSN 14764687. doi: 10.1038/nature14236.
- Nisan, Noam, Roughgarden, Tim, Tardos, Eva, and Vazirani, Vijay V. Algorithmic game theory , volume 1. Cambridge University Press Cambridge, 2007.

- Park, Jooyoung and Sandberg, Irwin W. Universal approximation using radial-basis-function networks. Neural computation , 3(2):246-257, 1991.
- Peters, Jan and Schaal, Stefan. Reinforcement learning of motor skills with policy gradients. Neural networks , 21 (4):682-697, 2008.
- Petersen, K. B. and Pedersen, M. S. The matrix cookbook, nov 2012. Version 20121115.
- Schulman, John, Levine, Sergey, Abbeel, Pieter, Jordan, Michael, and Moritz, Philipp. Trust region policy optimization. In International Conference on Machine Learning , pp. 1889-1897, 2015.
- Schulman, John, Wolski, Filip, Dhariwal, Prafulla, Radford, Alec, and Klimov, Oleg. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 , 2017.
- Silver, David, Lever, Guy, Heess, Nicolas, Degris, Thomas, Wierstra, Daan, and Riedmiller, Martin. Deterministic Policy Gradient Algorithms. Proceedings of the 31st International Conference on Machine Learning (ICML14) , pp. 387-395, 2014. ISSN 1938-7228.
- Silver, David, Schrittwieser, Julian, Simonyan, Karen, Antonoglou, Ioannis, Huang, Aja, Guez, Arthur, Hubert, Thomas, Baker, Lucas, Lai, Matthew, Bolton, Adrian, Chen, Yutian, Lillicrap, Timothy, Hui, Fan, Sifre, Laurent, Van Den Driessche, George, Graepel, Thore, and Hassabis, Demis. Mastering the game of Go without human knowledge. Nature , 550(7676):354-359, 2017. ISSN 14764687. doi: 10.1038/nature24270.
- Stein, Elias M. and Shakarchi, Rami. Fourier Analysis: An Introduction . 2003. ISBN 9780691113845.
- Stone, Marshall H. The generalized weierstrass approximation theorem. Mathematics Magazine , 21(5):237-254, 1948.
- Sukhbaatar, Sainbayar, Szlam, Arthur, and Fergus, Rob. Learning Multiagent Communication with Backpropagation. (Nips), 2016. ISSN 10495258. URL http: //arxiv.org/abs/1605.07736 .
- Sutton, Richard S. Generalization in reinforcement learning: Successful examples using sparse coarse coding. In Advances in neural information processing systems , pp. 1038-1044, 1996.
- Sutton, Richard S. and Barto, Andrew G. Sutton &amp; Barto Book: Reinforcement Learning: An Introduction. MIT Press, Cambridge, MA, A Bradford Book , 1998. ISSN 10459227. doi: 10.1109/TNN.1998.712192.
- Sutton, Richard S., Mcallester, David, Singh, Satinder, and Mansour, Yishay. Policy Gradient Methods for Reinforcement Learning with Function Approximation. Advances in Neural Information Processing Systems 12 , pp. 10571063, 1999. ISSN 0047-2875. doi: 10.1.1.37.9714.
- Sutton, Richard S, Singh, SP, and McAllester, DA. Comparing policy-gradient algorithms. URL http://citeseerx. ist. psu. edu/viewdoc/download , 2000.
- van Hasselt, Hado, Guez, Arthur, and Silver, David. Deep Reinforcement Learning with Double Q-learning. 2015. ISSN 00043702. doi: 10.1016/j.artint.2015.09.002. URL http://arxiv.org/abs/1509.06461 .
- van Seijen, Harm, van Hasselt, Hado, Whiteson, Shimon, and Wiering, Marco. A theoretical and empirical analysis of expected sarsa. In Adaptive Dynamic Programming and Reinforcement Learning, 2009. ADPRL'09. IEEE Symposium on , pp. 177-184. IEEE, 2009.
- Weierstrass, Karl. ¨ Uber die analytische Darstellbarkeit sogenannter willk¨ urlicher Functionen einer reellen Ver¨ anderlichen. Sitzungsberichte der K¨ oniglich Preußischen Akademie der Wissenschaften zu Berlin , 2:633-639, 1885.
- Williams, Ronald J. Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Machine Learning , 8(3):229-256, 1992. ISSN 15730565. doi: 10.1023/A:1022672621406.
- Wu, Yuhuai, Mansimov, Elman, Grosse, Roger B, Liao, Shun, and Ba, Jimmy. Scalable trust-region method for deep reinforcement learning using kronecker-factored approximation. In Advances in neural information processing systems , pp. 5285-5294, 2017.
- Xu, Kelvin, Ba, Jimmy, Kiros, Ryan, Cho, Kyunghyun, Courville, Aaron, Salakhutdinov, Ruslan, Zemel, Richard, and Bengio, Yoshua. Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. 2015. ISSN 19410093. doi: 10.1109/72.279181. URL http: //arxiv.org/abs/1502.03044 .

## A. Fourier Series and Approximation

Formally, the Fourier series is an expansion of a periodic function f ( x ) of period 2 L in terms of an infinite summation of sines and cosines. For clarity, we give the univariate case - the multivariate result can be found in literature.

<!-- formula-not-decoded -->

where ω 0 glyph[defines] π L and the coefficients for the series are:

<!-- formula-not-decoded -->

By writing sine and cosine terms in their complex exponential forms, it is possible to define a complex Fourier series for real valued functions as

<!-- formula-not-decoded -->

(26) and (27) are equivalent if we set c m as:

<!-- formula-not-decoded -->

In reality, we cannot sum to infinity and instead use the series to approximate f ( x ) to a finite value of m . Just as a Taylor series approximation becomes more accurate by using higher and higher order polynomials x m , a Fourier series expansion becomes more accurate by using sinusoids of higher and higher frequencies mω 0 . However, a Fourier series approximation approximates the function over its whole period, whereas the Taylor series does so only in a local neighbourhood of the given point.

Although the Fourier series is defined for periodic functions, it is still applicable to aperiodic functions. For bounded aperiodic functions, we define the period 2 L to be the size of the domain of f ( x ) and then integrate over this domain to obtain the Fourier coefficients. Intuitively, this is equivalent to repeating the bounded function periodically over an infinite domain. Aperiodic functions that are not bounded may be approximated by defining Fourier series over a bounded region of the function. As the size of this bounded region increases, and consequently the period 2 L increases, the Fourier series approximation becomes more accurate and approaches a Fourier transform. Thus, for aperiodic unbounded functions, a Fourier series approximates a Fourier transform.

We now formalise the idea of taking the limit of the period going to infinity ( L → ∞ ) for a complex Fourier series representation of any general function f ( x ) . Firstly, it is convenient to rewrite (27) as:

<!-- formula-not-decoded -->

Taking the limit as L →∞ (Stein &amp; Shakarchi, 2003) gives

<!-- formula-not-decoded -->

which is exactly equivalent to (7).

The integrals in the definition of the Fourier transform arise from taking a Fourier series representation of a function and letting the number of coefficients go to infinity.

## B. n -Dimensional Fourier Transforms

Definitions Firstly, we make the definition of a n -dimensional Fourier transform precise: Consider a function f ( · ) : R n → R . For x = ( x 1 , x 2 , ...x n ) glyph[latticetop] ∈ R n and ω = ( ω 1 , ω 2 , ...ω n ) glyph[latticetop] ∈ R n , we have:

<!-- formula-not-decoded -->

The corresponding n -Dimensional inverse Fourier transform is defined as:

<!-- formula-not-decoded -->

We define the Fourier transform of a vector/matrix quantity as simply the Fourier transform of individual elements of the vector/matrix. For example, the Fourier transform of matrix [ F ( x ) ] jk = f jk ( x ) is found from:

<!-- formula-not-decoded -->

And similarly for the inverse Fourier transform:

<!-- formula-not-decoded -->

Multiplication-Derivative Identities We now derive multi-dimension analogues to the single dimension multiplication-derivative property, which we state here:

<!-- formula-not-decoded -->

Proofs of (29) are commonplace in Fourier Analysis references (Stein &amp; Shakarchi, 2003). We start with a vector identity:

Lemma 1 (Multiplication-Derivative Property: Vectors) . Given a function f ( x ) with Fourier transform F ( f ( x )) , multiplying F ( f ( x )) by the vector i ω in the frequency domain is equivalent to taking the first order derivative ∇ x f ( x ) in the action domain, that is:

<!-- formula-not-decoded -->

Proof. Consider the elements of the vector i ω F ( f ( x )) :

<!-- formula-not-decoded -->

Using the single dimension multiplication-derivative property from (29) yields:

<!-- formula-not-decoded -->

Using the definition of the Fourier transform of a vector from (28) gives our main result:

<!-- formula-not-decoded -->

We now derive a similar identity for matrices:

Lemma 2 (Multiplication-Derivative Property: Matrices) . Given a function f ( x ) with Fourier transform F ( f ( x )) , multiplying F ( f ( x )) by the matrix ( i ω )( i ω ) glyph[latticetop] in the frequency domain is equivalent to taking the second order derivative ∇ (2) x f ( x ) in the action domain, that is:

<!-- formula-not-decoded -->

Proof. Consider the elements of the matrix ( i ω )( i ω ) glyph[latticetop] F ( f ( x )) :

<!-- formula-not-decoded -->

Using the single dimension multiplication-derivative property from (29) twice yields:

<!-- formula-not-decoded -->

Using the definition of the Fourier transform of a matrix from (28) gives our main result:

<!-- formula-not-decoded -->

## C. Auxiliary Function Properties

Lemma 3 ( n th Order Derivative of Auxiliary Function) . Given an auxiliary function ˜ β ( µ -a ) = β ( a ) for a policy β , we may relate the m -th order derivative of ˜ β w.r.t. µ to the m th order derivative of β w.r.t. a as:

<!-- formula-not-decoded -->

Proof. For m = 1 From the chain rule we write:

<!-- formula-not-decoded -->

Let ν = µ -a s.t. ˜ β ( µ -a ) = ˜ β ( ν ) . Using the chain rule again for ∇ µ ˜ β ( µ -a ) yields:

<!-- formula-not-decoded -->

Now, ∇ µ ν = I and ∇ ν a = -I . Substituting yields:

<!-- formula-not-decoded -->

Substituting ˜ β ( ν ) = ˜ β ( µ -a ) = β ( a ) gives our main result for m = 1 :

<!-- formula-not-decoded -->

Finally, taking m -1 more derivatives will give our main result:

<!-- formula-not-decoded -->

## D. Turntable Experimental Setup Details

The turntable domain is a toy continuous control task. The goal is to align a disk to a desired angle by rotating it around its axis. The action is an angle in the range a ∈ [ -π, π ] and the observations are the current position of the disk and the target position, both expressed as angles. The reward is set to sin( α + α target ) -1 4 | a | . For DPG, we used the OpenAI baseline implementation, where both the actor and the critic are represented using neural networks. For FourierEPG, we used the same setup but changed the critic to be trigonometric critic of the form sin( α + α target -a ) + w | a | with a tuneable weight w and the actor update given by Equation (23). The exploration policy was Gaussian with fixed standard deviation σ = 0 . 05 in both cases.

## E. Gaussian Derivatives

We derive specific analytical solutions for the Gaussian policy β = N ( µ , Σ ) from Section 4.1. The following identities (Petersen &amp; Pedersen, 2012) will be useful:

<!-- formula-not-decoded -->

Zeroth order ( M =0) Substituting for ∇ a β from (30) and ∇ (2) a β from (31) in (17) and (18) respectively, we obtain our analytic expression:

<!-- formula-not-decoded -->

First order ( M =1) Substituting for ∇ a β from (30) in (20), we obtain our analytic expression:

<!-- formula-not-decoded -->

## F. Proofs

Corollary 2.1. Let ψ be a parameter that does not depend upon µ . We can write ˆ I ψ ( s t ) = ∇ ψ ∫ a ˆ Q ( s t , a ) β θ ( a | s t ) d a as:

<!-- formula-not-decoded -->

Proof. Using Theorem 2, we obtain the following expression for ˆ I ψ ( s t ) :

<!-- formula-not-decoded -->

Using Leibniz's rule for integration under the integral, we move the derivative inside of the inverse Fourier transform, obtaining our result:

<!-- formula-not-decoded -->

## G. Complete Periodic Critic Derivation

Wenowderive the analytic update from (22) for our periodic critic. Firstly, for ease of analysis we re-write our critic using the hyperbolic function:

<!-- formula-not-decoded -->

Taking the Fourier transform yields:

<!-- formula-not-decoded -->

Recall that the characteristic function of the Gaussian auxiliary function is F ( ˜ β ) = e -ω glyph[latticetop] Σ ω . Now taking inverse Fourier transforms of F ( ˆ Q ) F ( ˜ β ) yields:

<!-- formula-not-decoded -->

where we have used the hyperbolic definition of cos to derive our desired result in the final line.