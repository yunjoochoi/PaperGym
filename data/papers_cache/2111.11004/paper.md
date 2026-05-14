## Gradient Temporal Difference with Momentum: Stability and Convergence

Rohan Deb * , Shalabh Bhatnagar

Department of Computer Science and Automation Indian Institute of Science, Bangalore

rohandeb@iisc.ac.in, shalabh@iisc.ac.in

bins and Monro 1951). In recent times, the ODE method to analyze asymptotic behaviour of SA (Ljung 1977; Kushner and Clark 1978; Borkar 2008b; Borkar and Meyn 2000) has become quite popular in the RL community. The Gradient TD methods were shown to be convergent using the ODE approach. A generic one-timescale (One-TS) SA iterate has the following form:

<!-- formula-not-decoded -->

where x ∈ R d 1 are the iterates. The function h : R d 1 → R d 1 is assumed to be a Lipschitz continuous function. M n +1 is a Martingale difference noise sequence and a ( n ) is the step-size at time-step n . Under some mild assumptions, the iterate given by (1) converges (see Borkar 2008b; Borkar and Meyn 2000). When h is a linear map of the form b -Ax n , the matrix A is often called the driving matrix. The three Gradient TD algorithms: GTD (Sutton, Maei, and Szepesv´ ari 2009), GTD2 and TDC (Sutton et al. 2009) consist two iterates of the following form:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where x ∈ R d 1 , y ∈ R d 2 . See section 2 for exact form of the iterates. The two iterates still form a One-TS SA scheme if lim n →∞ b ( n ) a ( n ) = c , where c is a constant and a two-timescale (two-TS) scheme if lim n →∞ b ( n ) = 0 .

<!-- formula-not-decoded -->

Separately, adding a momentum term to accelerate the convergence of iterates is a popular technique in stochastic gradient descent (SGD). The two most popular schemes are the Polyak's Heavy ball method (Polyak 1964), and Nesterov's accelerated gradient method (Nesterov 1983). A lot of literature is dedicated to studying momentum with SGD. Some recent works include (Ghadimi, Feyzmahdavian, and Johansson 2014; Loizou and Richt´ arik 2020; Gitman et al. 2019; Ma and Yarats 2019; Assran and Rabbat 2020). Momentum in the SA setting, which is the focus of the current work, has limited results. Very few works study the effect of momentum in the SA setting. A recent work by (Mou et al. 2020) studies SA with momentum briefly and shows an improvement of mixing rate. However, the setting considered is restricted to linear SA and the driving matrix is assumed to be symmetric. Further, the iterates involve an additional

## Abstract

Gradient temporal difference (Gradient TD) algorithms are a popular class of stochastic approximation (SA) algorithms used for policy evaluation in reinforcement learning. Here, we consider Gradient TD algorithms with an additional heavy ball momentum term and provide choice of step size and momentum parameter that ensures almost sure convergence of these algorithms asymptotically. In doing so, we decompose the heavy ball Gradient TD iterates into three separate iterates with different step sizes. We first analyze these iterates under one-timescale SA setting using results from current literature. However, the one-timescale case is restrictive and a more general analysis can be provided by looking at a three-timescale decomposition of the iterates. In the process we provide the first conditions for stability and convergence of general threetimescale SA. We then prove that the heavy ball Gradient TD algorithm is convergent using our three-timescale SA analysis. Finally, we evaluate these algorithms on standard RL problems and report improvement in performance over the vanilla algorithms.

## 1 Introduction

In reinforcement learning (RL), the goal of the learner or the agent is to maximize its long term accumulated reward by interacting with the environment. One important task in most of RL algorithms is that of policy evaluation . It predicts the average accumulated reward an agent would receive from a state (called value function ) if it follows the given policy. In model-free learning , the agent does not have access to the underlying dynamics of the environment and has to learn the value function from samples of the form (state, action, reward, next-state). Two very popular algorithms in the model-free setting are Monte-Carlo (MC) and temporal difference (TD) learning (see Sutton and Barto (2018), Sutton (1988)). It is a well known fact that TD learning diverges in the offpolicy setting (see Baird (1995)). A class of algorithms called gradient temporal difference (Gradient TD) were introduced in (Sutton, Maei, and Szepesv´ ari 2009) and (Sutton et al. 2009) which are convergent even in the off-policy setting. These algorithms fall under a larger class of algorithms called linear stochastic approximation (SA) algorithms.

A lot of literature is dedicated to studying the asymptotic behaviour of SA algorithms starting from the work of (Rob- Polyak-Ruppert averaging (Polyak 1990). Here, in contrast, we analyze the asymptotic behaviour of the algorithm and make none of the above assumptions. A somewhat distant paper is by (Devraj, Buˇ s´ ı´ c, and Meyn 2019) that introduces Matrix momentum in SA and is not equivalent to heavy ball momentum.

* Corresponding Author

A very recent work by (Avrachenkov, Patil, and Thoppe 2020) studied One-TS SA with heavy ball momentum in the univariate case (i.e., d = 1 in iterate (1)) in the context of web-page crawling. The iterates took the following form:

<!-- formula-not-decoded -->

The momentum parameter η n was chosen to decompose the iterate into two recursions of the form given by (2) and (3). We use such a decomposition for Gradient TD methods with momentum. This leads to three separate iterates with three step-sizes. We analyze these three iterates and provide stability (iterates remain bounded throughout) and almost sure (a.s.) convergence guarantees.

## 1.1 Our Contribution

- We first consider the One-TS decomposition of Gradient TD with momentum iterates and show that the driving matrix in this case is Hurwitz (all eigen values are negative). Thereafter we use the theory of One-TS SA to show that the iterates are stable and convergent to the same TD solution.
- Next, we consider the Three-TS decomposition. We provide the first stability and convergence conditions for general Three-TS recursions. We then show that the iterates under consideration satisfy these conditions.
- Finally, we evaluate these algorithms for different choice of step-size and momentum parameters on standard RL problems and report an improvement in performance over their vanilla counterparts.

## 2 Preliminaries

In the standard RL setup, an agent interacts with the environment which is a Markov Decision Process (MDP). At each discrete time step t , the agent is in state s t ∈ S , takes an action a t ∈ A , receives a reward r t +1 ≡ r ( s t , a t , s t +1 ) ∈ R and moves to another state s t +1 ∈ S . Here S and A are finite sets of possible states and actions respectively. The transitions are governed by a kernel P . A policy π : S×A → [0 , 1] is a mapping that defines the probability of picking an action in a state. We let P π ( s ′ | s ) be the transition probability matrix induced by π . Also, { d π ( s ) } s ∈S represents the steady-state distribution for the Markov chain induced by π and the matrix D is a diagonal matrix of dimension n × n with the entries d π ( s ) on its diagonals.The state-value function associated with a policy π for state s is

<!-- formula-not-decoded -->

where γ ∈ [0 , 1) is the discount factor.

In the linear architecture setting, policy evaluation deals with estimating V π ( s ) through a linear model V θ ( s ) =

θ T φ ( s ) , where φ ( s ) ≡ φ s is a feature associated with the state s and θ is the parameter vector. We define the TD-error as δ t = r t +1 + γθ T t φ t +1 -θ T t φ t and Φ as an n × d matrix where the s th row is φ ( s ) T . In the i.i.d setting it is assumed that the tuple ( φ t , φ ′ t ) (where φ t +1 ≡ φ ′ t ) is drawn independently from the stationary distribution of the Markov chain induced by π . Let ¯ A = E [ φ t ( γφ ′ t -φ t ) T ] and ¯ b = E [ r t +1 φ t ] , where the expectations are w.r.t. the stationary distribution of the induced chain. The matrix ¯ A is negative definite (see Maei (2011); Tsitsiklis and Van Roy (1997)). In the off-policy case, the importance weight is given by ρ t = π ( a t | s t ) µ ( a t | s t ) , where π and µ are the target and behaviour policies respectively. Introduced in (Sutton, Maei, and Szepesv´ ari 2009), Gradient TD are a class of TD algorithms that are convergent even in the off-policy setting. Next, we present the iterates associated with the algorithms GTD (Sutton, Maei, and Szepesv´ ari 2009), GTD2, TDC (Sutton et al. 2009).

## · GTD :

## · GTD2 :

## · TDC :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The objective function for GTD is Norm of Expected Error defined as NEU ( θ ) = E [ δφ ] . The GTD algorithm is derived by expressing the gradient direction as -1 2 ∇ NEU ( θ ) = E [ ( φ -γφ ′ ) φ T ] E [ δ ( θ ) φ ] . Here φ ′ ≡ φ ( s ′ ) . If both the expectations are sampled together, then the term would be biased by their correlation. An estimate of the second expectation is maintained as a long-term quasi-stationary estimate (see (5)) and the first expectation is sampled (see (6)). For GTD2 and TDC, a similar approach is used on the objective function Mean Square Projected Bellman Error defined as MSPBE ( θ ) = || V θ -Π T π V θ || D . Here, Π is the projection operator that projects vectors to the subspace { Φ θ | θ ∈ R d } and T π is the Bellman operator defined as T π V = R π + γP π V . As originally presented, GTD and GTD2 are one-timescale algorithms ( α t β t is constant) while TDC is a two-timescale algorithm ( α t β t → 0 ). It was shown in all the three cases that θ n → θ ∗ = -¯ A -1 ¯ b .

## 3 Gradient TD with Momentum

Although, Gradient TD starts with a gradient descent based approach, it ends up with two-TS SA recursions. Momentum methods are known to accelerate the convergence of SGD iterates. Motivated by this, we examine momentum in the SA setting, and ask if the SA recursions for Gradient TD with momentum even converge to the same TD solution. We probe the heavy ball extension of the three Gradient TD algorithms where, we keep an accumulation of the previous gradient values in ζ t . Then, at time step t +1 the new gradient value multiplied by the step size is added to the current accumulation vector ζ t multiplied by the momentum parameter η t as below:

<!-- formula-not-decoded -->

The parameter θ is then updated in the negative of the direction ζ t +1 , i.e., θ t +1 = θ t -ζ t +1 . Since u t +1 is computed as a long-term estimate of E [ δ ( θ ) φ ] , its update rule remains same. The momentum parameter η t is usually set to a constant in the stochastic gradient setting. An exception to this can however be found in (Gitman et al. 2019; Gadat, Panloup, and Saadane 2016), where η t → 1 . Here, we consider the latter case. Substituting ζ t +1 into the iteration of θ t +1 and noting that ζ t = θ t -θ t -1 , the iterates for GTD with Momentum ( GTD-M ) can be written as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly the iterates for GTD2-M are given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, the iterates for TDC-M are given by:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We choose the momentum parameter η t as in (Avrachenkov, Patil, and Thoppe 2020) as follows: η t = glyph[rho1] t -wα t glyph[rho1] t -1 , where { glyph[rho1] t } is a positive sequence s.t. glyph[rho1] t → 0 as t →∞ and w ∈ R is a constant. Note that η t → 1 as t →∞ . We later provide conditions on glyph[rho1] t and w to ensure a.s. convergence. As we would see in section 4, the condition on w in the One-TS setting is restrictive. Specifically, it depends on the norm of the driving matrix ¯ A . This motivates us to look at the ThreeTS setting and then the corresponding condition on w is less restrictive. Using the momentum parameter as above,

θ t +1 = θ t + α t ( φ t -γφ ′ t ) φ T t u t + glyph[rho1] t -wα t glyph[rho1] t -1 ( θ t -θ t -1 ) Rearranging the terms and dividing by ρ t , we get:

<!-- formula-not-decoded -->

We let

<!-- formula-not-decoded -->

Then, the GTD-M iterates in (11) and (12) can be re-written with the following three iterates:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

A similar decomposition can be done for the GTD2-M and TDC-M iterates.

## 4 Convergence Analysis

In this section we analyze the asymptotic behaviour of the GTD-M iterates given by (17), (18) and (19). Throughout the section, we consider v t , u t , θ t ∈ R d . We first consider the One-TS case when β t = c 1 ξ t and glyph[rho1] t = c 2 ξ t ∀ t , for some real constants c 1 , c 2 &gt; 0 . Subsequently, we consider the Three-TS setting where β t ξ t → 0 and glyph[rho1] t β t → 0 as t →∞ .

## 4.1 One-Timescale Setting

We begin by analyzing GTD-M using a one-timescale SA setting. We let c 1 = c 2 = 1 for simplicity. The iterates of GTD-M can then be re-written as:

<!-- formula-not-decoded -->

where,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Equation (20) can be re-written in the general SA scheme as:

ψ t +1 = ψ t + ξ t ( h ( ψ t ) + M t +1 + ¯ ε t ) . (21) Here h ( ψ ) = g + Gψ,g = E [ g t ] , G = E [ G t ] , where the expectations are w.r.t. the stationary distribution of the Markov chain induced by the target policy π . M t +1 = ( G t +1 -G ) ψ t +( g t +1 -g ) . In particular,

<!-- formula-not-decoded -->

where recall that ¯ A = E [ φ ( γφ ′ -φ ) T ] and ¯ b = E [ rφ ]

Lemma 1. Assume, w ( w +1) &gt; || ¯ A || 2 . Then, the matrix G is Hurwitz.

Proof. Let λ be an eigenvalue of G . The characteristic equation of the matrix G is given by:

<!-- formula-not-decoded -->

Using the following formula for determinant of block matrices

<!-- formula-not-decoded -->

we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, from the characteristic equation of G , we have that

<!-- formula-not-decoded -->

There must exist a non-zero vector x ∈ C d , such that

<!-- formula-not-decoded -->

where x ∗ is the conjugate transpose of the vector x and x ∗ x = || x || 2 &gt; 0 . The above equation reduces to the following cubic-polynomial equation:

λ 3 || x || 2 +( w +1) λ 2 || x || 2 + wλ || x || 2 + || ¯ Ax || 2 = 0 , where || ¯ Ax || 2 = x ∗ ¯ A T ¯ Ax . Using Routh-Hurwitz criterion, a cubic polynomial a 3 λ 3 + a 2 λ 2 + a 1 λ + a 0 has all roots with negative real parts iff a 3 , a 2 , a 1 , a 0 &gt; 0 and a 1 a 2 &gt; a 0 a 3 . In our case, a 3 = || x || 2 &gt; 0 , a 2 = ( w + 1) || x || 2 &gt; 0 , a 1 = w || x || 2 &gt; 0 and a 0 = || ¯ Ax || 2 &gt; 0 . The last inequality follows from the fact that ¯ A is negative definite and therefore x ∗ ¯ A T ¯ Ax &gt; 0 . Finally, a 1 a 2 = w ( w + 1) || x || 4 , a 0 a 3 = || x || 2 || ¯ Ax || 2 and a 1 a 2 &gt; a 0 a 3 follows from || ¯ Ax || 2 || x || 2 &lt; || ¯ A || 2 &lt; w ( w + 1) . Therefore Re ( λ ) &lt; 0 and the claim follows.

Consider the following assumptions:

A 1. All rewards r ( s, s ′ ) and features φ ( s ) are bounded, i.e., r ( s, s ′ ) ≤ 1 and || φ ( s ) || ≤ 1 ∀ s, s ′ ∈ S . Also, the matrix Φ has full rank, where Φ is an n × d matrix where the s th row is φ ( s ) T .

A 2. The step-sizes satisfy ξ t = β t = glyph[rho1] t &gt; 0 ,

<!-- formula-not-decoded -->

and the momentum parameter satisfies: η t = glyph[rho1] t -wα t glyph[rho1] t -1 .

A 3. The samples ( φ t , φ ′ t ) are drawn i.i.d from the stationary distribution of the Markov chain induced by target policy π .

Theorem 2. Assume A 1 , A 2 and A 3 hold and let w ≥ 1 . Then, the GTD-M iterates given by (11) and (12) satisfy θ n → θ ∗ = -¯ A -1 ¯ b a.s. as n →∞ .

Proof. Assumption A 1 ensures that || ¯ A || 2 &lt; w ( w +1) and A 3 ensures that the function h ( · ) is well defined. Now, using Lemma 1 and (Borkar and Meyn 2000) we can show that the iterates in (20) remain stable. Then using the third extension from (Chapter-2 pp. 17, Borkar (2008b)) we can show that ψ n →-G -1 g as n →∞ . Thereafter using the formula for inverse of block matrices it can be shown that θ n →-¯ A -1 b as n →∞ . See Appendix A1 for a detailed proof.

Similar results can be proved for the GTD2-M and TDC-M iterates.

Remark 1. If w is large, the initial values of the momentum parameter is small. The condition on w in lemma 1 is large compared to the condition on w in (Avrachenkov, Patil, and Thoppe 2020), where the condition is w &gt; 0 . Motivated by this, we look at the three-TS case of the iterates.

## 4.2 Three Timescale Setting

We consider the three iterates for GTD-M in (17), (18) and (19) under the following criteria for step-sizes: ξ t β t → 0 and glyph[rho1] t ξ t → 0 as t →∞ . We provide the first conditions for stability and a.s. convergence of generic three-TS SA recursions. We emphasize that the setting we look at in Theorem 3 is more general than the setting at hand of GTD-M iterates. Although stability and convergence results exist for one-TS and two-TS cases, this is the first time such results have been provided for the case of three-TS recursions. We next provide the general iterates for a three-TS recursion along with the assumptions used while analyzing them. Consider the following three iterates:

<!-- formula-not-decoded -->

and the following assumptions:

- (B1) h : R d 1 + d 2 + d 3 → R d 1 , g : R d 1 + d 2 + d 3 → R d 2 , f : R d 1 + d 2 + d 3 → R d 3 are Lipchitz continuous, with Lipchitz constants L 1 , L 2 and L 3 respectively.
- (B2) { a ( n ) } , { b ( n ) } , { c ( n ) } are step-size sequences that satisfy a ( n ) &gt; 0 , b ( n ) &gt; 0 , c ( n ) &gt; 0 , ∀ n &gt; 0 ,

<!-- formula-not-decoded -->

- (B3) { M (1) n } , { M (2) n } , { M (3) n } are Martingale difference sequences w.r.t. the filtration {F n } where,

<!-- formula-not-decoded -->

∀ n ≥ 0 , i = 1 , 2 , 3 and constants 0 &lt; K i &lt; ∞ . The terms ε ( i ) t satisfy || ε (1) n || + || ε (2) n || + || ε (3) n ||→ 0 as n →

<!-- formula-not-decoded -->

- (B4) (i) The ode ˙ x ( t ) = h ( x ( t ) , y, z ) , y ∈ R d 2 , z ∈ R d 3 has a globally asymptotically stable equilibrium (g.a.s.e) λ ( y, z ) , and λ : R d 2 × d 3 → R d 1 is Lipchitz continuous.
- (ii) The ode ˙ y ( t ) = g ( λ ( y ( t ) , z ) , y ( t ) , z ) , z ∈ R d 3 has a globally asymptotically stable equilibrium Γ( z ) , where Γ : R d 3 → R d 2 is Lipchitz continuous.
- (iii) The ode ˙ z ( t ) = f ( λ (Γ( z ( t )) , z ( t )) , Γ( z ( t )) , z ( t )) , has a globally asymptotically stable equilibrium z ∗ ∈ R d 3 .
- (B5) The functions h c ( x, y, z ) = h ( cx,cy,cz ) c , c ≥ 1 satisfy h c → h ∞ as c →∞ uniformly on compacts. The ODE: ˙ x ( t ) = h ∞ ( x ( t ) , y, z ) , has a unique globally asymptotically stable equilibrium λ ∞ ( y, z ) , where λ ∞ : R d 2 + d 3 → R d 1 is Lipschitz continuous. Further, λ ∞ (0 , 0) = 0 .
- (B6) The functions g c ( y, z ) = g ( cλ ∞ ( y,z ) ,cy,cz ) c , c ≥ 1 satisfy g c → g ∞ as c →∞ uniformly on compacts. The ODE: ˙ y ( t ) = g ∞ ( y ( t ) , z ) , has a unique globally asymptotically stable equilibrium Γ ∞ ( z ) , where Γ ∞ : R d 3 → R d 2 is Lipschitz continuous. Further, Γ ∞ (0) = 0 .
- (B7) The functions f c ( z ) = g ( cλ ∞ (Γ ∞ ( z ) ,z ) ,c Γ ∞ ( z ) ,cz ) c , c ≥ 1 satisfy f c → f ∞ as c → ∞ uniformly on compacts. The ODE: ˙ z ( t ) = f ∞ ( z ( t )) , has the origin in R d 3 as its unique globally asymptotically stable equilibrium.

Remark 2. Conditions ( B5 ) -( B7 ) give sufficient conditions that ensure that the iterates remain stable. Specifically it ensures that sup n ( || x n || + || y n || + || z n || ) &lt; ∞ a.s. . Conditions ( B1 ) -( B4 ) along with the stability of iterates ensures a.s. convergence of the iterates.

Theorem 3. Under assumptions ( B1 ) -( B7 ) ,the iterates given by (22) satisfy (23) and (24) ,

<!-- formula-not-decoded -->

Next we use theorem 3, to show that the iterates of GTDM a.s. converge to the TD solution -¯ A -1 ¯ b . Consider the following assumption on step-size sequences instead of A 2 .

A 4. The step-sizes satisfy ξ t &gt; 0 , β t &gt; 0 , glyph[rho1] t &gt; 0 ∀ t ,

<!-- formula-not-decoded -->

and the momentum parameter satisfies: η t = glyph[rho1] t -wα t glyph[rho1] t -1 .

Theorem 4. Assume A 1 , A 3 and A 4 hold and let w &gt; 0 . Then, the GTD-M iterates given by (11) and (12) satisfy θ n → θ ∗ = -¯ A -1 ¯ b a.s. as n →∞ .

Proof. We transform the iterates given by (17), (18) and (19) into the standard SA form given by (22), (23) and (24). Let F t = σ ( u 0 , v 0 , θ 0 , r j +1 , φ j , φ ′ j : j &lt; t ) . Let, A t = φ t ( γφ ′ t -φ t ) T and b t = r t +1 φ t . Then, (17) can be re-written as:

<!-- formula-not-decoded -->

where,

<!-- formula-not-decoded -->

M (1) t +1 = -A T t u t -wv t -h ( v t , u t , θ t ) = ( ¯ A T -A T t ) u t . Next, (18) can be re-written as:

<!-- formula-not-decoded -->

where,

<!-- formula-not-decoded -->

Finally, (19) can be re-written as:

<!-- formula-not-decoded -->

where, f ( v t , u t , θ t ) = v t and M (3) t +1 = 0 . The functions h, g, f are linear in v, u, θ and hence Lipchitz continuous, therefore satisfying ( B1 ) . We choose the stepsize sequences such that they satisfy ( B2 ) . One popular choice is ξ t = 1 ( t +1) ξ , β t = 1 ( t +1) β , glyph[rho1] t = 1 ( t +1) glyph[rho1] , 1 2 &lt; ξ &lt; β &lt; glyph[rho1] ≤ 1 . Next, M (1) t +1 , M (2) t +1 and M (3) t +1 t ≥ 0 , are martingale difference sequences w.r.t F t by construction. E [ || M (1) t +1 || 2 |F t ] ≤ || ( ¯ A T -A T t ) || 2 || u t || 2 , E [ || M (2) t +1 || 2 |F t ] ≤ 2( || ( A t -¯ A ) || 2 || θ t || 2 + || ( b t -¯ b ) || 2 ) . The first part of ( B3 ) is satisfied with K 1 = || ( ¯ A T -A T t ) || 2 , K 2 = 2max( || A t -¯ A || 2 , || b t -¯ b || 2 ) and any K 3 &gt; 0 . The fact that K 1 , K 2 &lt; ∞ follows from the bounded features and bounded rewards assumption in A 1 . Next, observe that || ε (3) t || = ξ t || ( ( φ t -γφ ′ t ) φ T t u t -wv t ) || → 0 since ξ t → 0 as t → ∞ . For a fixed u, θ ∈ R d , consider the ODE ˙ v ( t ) = -¯ A T u -wv ( t ) . For w &gt; 0 , λ ( u, θ ) = -¯ A T u w is the unique g.a.s.e, is linear and therefore Lipchitz continuous. This satisfies ( B4 ) (i). Next, for a fixed θ ∈ R d , ˙ u ( t ) = ¯ Aθ + ¯ b -u ( t ) , has Γ( θ ) = ¯ Aθ + ¯ b as its unique g.a.s.e and is Lipschitz. This satisfies ( B4 )( ii ) . Finally, to satisfy ( B4 )( iii ) , consider,

<!-- formula-not-decoded -->

Since, ¯ A is negative definite, therefore, -¯ A T ¯ A is negative definite. Therefore, θ ∗ = -¯ A -1 ¯ b is the unique g.a.s.e. Next, we show that the sufficient conditions for stability of the three iterates are satisfied. The function, h c ( v, u, θ ) = -c ¯ A T u -wcv c = -¯ A T u -wv → h ∞ ( v, u, θ ) = -¯ A T u -wv uniformly on compacts as c → ∞ . The limiting ODE: ˙ v ( t ) = -¯ A T u -wv ( t ) has λ ∞ ( u, θ ) = -¯ A T u w as its unique g.a.s.e. λ ∞ is Lipschitz with λ ∞ (0 , 0) = 0 , thus satisfying assumption ( B5 ) .

The function, g c ( u, θ ) = c ¯ Aθ +¯ b -cu c = ¯ Aθ -u + ¯ b c → g ∞ ( u, θ ) = -¯ Aθ -u uniformly on compacts as c → ∞ . The limiting ODE ˙ u ( t ) = -¯ Aθ -u ( t ) has Γ ∞ ( θ ) = ¯ Aθ as its unique g.a.s.e. Γ ∞ is Lipchitz with Γ ∞ (0) = 0 . Thus assumption ( B6 ) is satisfied.

Figure 1: RMSPBE (averaged over 100 independent runs) accross episodes for Boyan Chain. The features used are the standard spiked features of size 4 used in Boyan chain (see (Dann, Neumann, and Peters 2014)).

<!-- image -->

Figure 2: RMSPBE (averaged over 100 independent runs) across episodes for the 5-State Random Chain problem. The features used are the Dependent features used in (Sutton et al. 2009).

<!-- image -->

Finally, f c ( θ ) = -c ¯ A T ¯ Aθ cw → f ∞ = -¯ A T ¯ Aθ w uniformly on compacts as c → ∞ and the ODE: ˙ θ ( t ) = -¯ A T ¯ Aθ ( t ) w has origin as its unique g.a.s.e. This ensures the final condition ( B7 ) . By theorem 3,

<!-- formula-not-decoded -->

Similar analysis can be provided for GTD2-M and TDC-M iterates. See Appendix A3 for details.

## 5 Experiments

We evaluate the momentum based GTD algorithms defined in section 3 to four standard problems of policy evaluation in reinforcement learning namely, Boyan Chain (Boyan 1999), 5-State random walk (Sutton et al. 2009), 19-State Random Walk (Sutton and Barto 2018) and Random MDP (Sutton et al. 2009). See Appendix A4 for a detailed description of the MDP settings and (Dann, Neumann, and Peters 2014) for details on implementation. We run the three algorithms, GTD, GTD2 and TDC along with their heavy ball momentum variants in One-TS and Three-TS settings and compare the RMSPBE (Root of MSPBE) across episodes. Figure-1 to Figure-4 plot these results. We consider decreasing step-sizes of the form: ξ t = 1 ( t +1) ξ , β t = 1 ( t +1) β , glyph[rho1] t = 1 ( t +1) glyph[rho1] , α t = 1 ( t +1) α in all the examples. Table 1 summarizes the different step-size sequences used in our experiment.

Table 1: Choice of step-size parameters

| Boyan Chain   | α     | β      | glyph[rho1]   | w   |
|---------------|-------|--------|---------------|-----|
| Vanilla       | 0.25  | 0.125  | -             | -   |
| One-TS        | 0.25  | 0.125  | 0.125         | 1   |
| Three-TS      | 0.25  | 0.125  | 0.2           | 0.1 |
| 5-stateRW     | α     | β      | glyph[rho1]   | w   |
| Vanilla       | 0.25  | 0.125  | -             | -   |
| One-TS        | 0.25  | 0.125  | 0.125         | 1   |
| Three-TS      | 0.25  | 0.125  | 0.2           | 0.1 |
| 19-StateRW    | α     | β      | glyph[rho1]   | w   |
| Vanilla       | 0.125 | 0.0625 | -             | -   |
| One-TS        | 0.125 | 0.0625 | 0.0625        | 1   |
| Three-TS      | 0.125 | 0.0625 | 0.1           | 0.1 |
| Random Chain  | α     | β      | glyph[rho1]   | w   |
| Vanilla       | 0.5   | 0.25   | -             | -   |
| One-TS        | 0.5   | 0.25   | 0.25          | 1   |
| Three-TS      | 0.5   | 0.25   | 0.3           | 0.1 |

In one-TS setting, we require ξ = β = glyph[rho1] . Since ξ t = α t glyph[rho1] t , we must have α = 2 glyph[rho1] . In the Three-TS setting, ξ &lt; β &lt; glyph[rho1] thus implying, α &lt; glyph[rho1] + β and β &lt; glyph[rho1] . Although our analysis requires square summability: ξ, β, glyph[rho1] &gt; 0 . 5 , such choice of step-size makes the algorithms converge very slowly. Recently, (Dalal et al. 2018a) showed convergence rate results for Gradient TD schemes with non-square summable step-sizes also (See Remark 2 of (Dalal et al. 2018a)). Therefore, we look at non-square summable step-sizes here, and observe that in all the examples the iterates do converge. The momentum parameter is chosen as in A 2 .

Figure 3: RMSPBE (averaged over 100 independent runs) accross episodes for the 19-State Random Walk problem. The features used are an extension of the Dependent features used in (Sutton et al. 2009).

<!-- image -->

Figure 4: RMSPBE (averaged over 100 independent runs) accross episodes for 20-state Random MDP with 5 random actions. The features used are Linear random of size 10 (see (Dann, Neumann, and Peters 2014)). For each state, the value of the feature vector at 10 th position is 1 and all the values in all other 9 positions is chosen randomly from 0 to 10 and are then normalized.

<!-- image -->

In all the examples considered, the momentum methods outperform their vanilla counterparts. Since, in the Three-TS setting, a lower value of w can be chosen, this ensures that the momentum parameter is not small in the initial phase of the algorithm as in the One-TS setting. This in turn helps to reduce the RMSPBE faster in the initial phase of the algorithm as is evident from the experiments.

## 6 Related Work and Conclusion

To the best of our knowledge no previous work has specifically looked at Gradient TD methods with an added heavy ball term. The use of momentum specifically in the SA setting is very limited. Section 4.1 of (Mou et al. 2020) does talk about momentum; however the problem looked at is that of SGD with momentum and the driving matrix is assumed to be symmetric (see Appendix H of their paper). We do not make any such assumption here. The work of (Devraj, Buˇ s´ ı´ c, and Meyn 2019), indeed looks at momentum in SA setting. However, they introduce a matrix momentum term which is not equivalent to heavy ball momentum. Acceleration in Gradient TD methods has been looked at in (Pan, White, and White 2017). The authors provide a new algorithm called ATD and the acceleration is in form of better data efficiency. However, they do not make use of momentum methods.

In this work we have introduced heavy ball momentum in Gradient Temporal difference algorithms for the first time. We decompose the two iterates of these algorithms into three separate iterates and provide asymptotic convergence guarantees of these new schemes under the same assumptions made by their vanilla counterparts. Specifically, we show convergence in the One-TS regime as well as Three-TS regime. In both the cases, the momentum parameter gradually goes 1. Three-TS formulation gives us more flexibility in choosing the momentum parameter. Specifically, compared to the One-TS setting, a larger momentum parameter can be chosen during the initial phase in the Three-TS case. We observe improved performance with these new schemes when compared with the original algorithms.

As a step forward from this work, the natural direction would be to look at more sophisticated momentum methods such as Nesterov's accelerated method (Nesterov 1983). Also, here we only provide the convergence guarantees of these new momentum methods. A particularly interesting step would be to quantify the benefits of using momentum in SA settings. Specifically, it would be interesting to extend weak convergence rate analysis of (Konda and Tsitsiklis 2004; Mokkadem and Pelletier 2006) to Three-TS regime. Also, extending the recent convergence rate results in expectation and high probability of GTD methods (Dalal et al. 2018b; Gupta, Srikant, and Ying 2019; Kaledin et al. 2019; Dalal, Szorenyi, and Thoppe 2020) to these momentum settings would be interesting works for the future.

Assran, M.; and Rabbat, M. 2020. On the Convergence of Nesterov's Accelerated Gradient Method in Stochastic Settings. Proceedings of the 37th International Conference on Machine Learning, PMLR , 119: 410-420.

Avrachenkov, K.; Patil, K.; and Thoppe, G. 2020. Online Algorithms for Estimating Change Rates of Web Pages. arXiv , 2009.08142.

Baird, L. 1995. Residual Algorithms: Reinforcement Learning with Function Approximation. In In Proceedings of the Twelfth International Conference on Machine Learning , 3037. Morgan Kaufmann.

Borkar, V. 2008a. Stochastic Approximation: A Dynamical Systems Viewpoint . Cambridge University Press. ISBN 9780521515924.

Borkar, V. S. 2008b. Stochastic Approximation: A Dynamical Systems Viewpoint . Cambridge University Press. ISBN 9780521515924.

Borkar, V. S.; and Meyn, S. P. 2000. The O.D.E. Method for Convergence of Stochastic Approximation and Reinforcement Learning. SIAM Journal on Control and Optimization , 38(2): 447-469.

Boyan, J. 1999. Least-Squares Temporal Difference Learning. In ICML .

Dalal, G.; Szorenyi, B.; and Thoppe, G. 2020. A Tale of Two-Timescale Reinforcement Learning with the Tightest Finite-Time Bound. Proceedings of the AAAI Conference on Artificial Intelligence , 34(04): 3701-3708.

Dalal, G.; Szorenyi, B.; Thoppe, G.; and Mannor, S. 2018a. Finite Sample Analysis of Two-Timescale Stochastic Approximation with Applications to Reinforcement Learning. arXiv:1703.05376.

Dalal, G.; Thoppe, G.; Sz¨ or´ enyi, B.; and Mannor, S. 2018b. Finite Sample Analysis of Two-Timescale Stochastic Approximation with Applications to Reinforcement Learning. In Bubeck, S.; Perchet, V.; and Rigollet, P., eds., Proceedings of the 31st Conference On Learning Theory , volume 75 of Proceedings of Machine Learning Research , 1199-1233. PMLR.

Dann, C.; Neumann, G.; and Peters, J. 2014. Policy Evaluation with Temporal Differences: A Survey and Comparison. Journal of Machine Learning Research , 15(24): 809-883.

Devraj, A. M.; Buˇ s´ ı´ c, A.; and Meyn, S. 2019. On Matrix Momentum Stochastic Approximation and Applications to Q-learning. 57th Annual Allerton Conference on Communication, Control, and Computing , 749-756.

Gadat, S.; Panloup, F.; and Saadane, S. 2016. Stochastic Heavy ball. Electronic Journal of Statistics , 12: 461-529.

Ghadimi, E.; Feyzmahdavian, H. R.; and Johansson, M. 2014. Global convergence of the Heavy-ball method for convex optimization. arXiv:1412.7457.

Gitman, I.; Lang, H.; Zhang, P.; and Xiao, L. 2019. Understanding the role of momentum in stochastic gradient methods. Advances in Neural Information Processing Systems , 9630-9640.

Gupta, H.; Srikant, R.; and Ying, L. 2019. Finite-Time Performance Bounds and Adaptive Learning Rate Selection for Two Time-Scale Reinforcement Learning. arXiv:1907.06290.

Kaledin, M.; Moulines, E.; Naumov, A.; Tadic, V.; and Wai, H. 2019. Finite Time Analysis of Linear Two-timescale Stochastic Approximation with Markovian Noise. Conference on Learning Theory , 125: 2144-2203.

Konda, V.; and Tsitsiklis, J. 2004. Convergence rate of linear two-time-scale stochastic approximation. Annals of Applied Probability , 14.

Kushner, H.; and Clark, D. 1978. Stochastic Approximation Methods for constrained and unconstrained systems. Springer.

Lakshminarayanan, C.; and Bhatnagar, S. 2017. A Stability Criterion for Two-Timescale Stochastic Approximation Schemes. Automatica , 79: 108-114.

Ljung, L. 1977. Analysis of recursive stochastic algorithms. IEEE Transactions on Automatic Control , 22(4): 551-575.

Loizou, N.; and Richt´ arik, P. 2020. Momentum and stochastic momentum for stochastic gradient, Newton, proximal point and subspace descent methods. Computational Optimization and Applications , 77: 653-710.

Ma, J.; and Yarats, D. 2019. Quasi-hyperbolic momentum and adam for deep learning. International Conference on Learning Representations .

Maei, H. R. 2011. Gradient Temporal-Difference Learning Algorithms . Ph.D. thesis, University of Alberta, CAN. AAINR89455.

Mokkadem, A.; and Pelletier, M. 2006. Convergence rate and averaging of nonlinear two-time-scale stochastic approximation algorithms. The Annals of Applied Probability , 16(3): 1671 - 1702.

Mou, W.; Li, C. J.; Wainwright, M. J.; Bartlett, P. L.; and Jordan, M. I. 2020. On Linear Stochastic Approximation: Fine-grained Polyak-Ruppert and Non-Asymptotic Concentration. Proceedings of Thirty Third Conference on Learning Theory, PMLR , 125: 2947-2997.

Nesterov, Y. 1983. A method of solving a convex programming problem with convergence rate O ( 1 k 2 ) . Soviet Mathematics Doklady , 269: 543-547.

Pan, Y.; White, A.; and White, M. 2017. Accelerated Gradient Temporal Difference Learning. arXiv:1611.09328.

Polyak, B. 1964. Some methods of speeding up the convergence of iteration methods. Ussr Computational Mathematics and Mathematical Physics , 4: 1-17.

Polyak, B. 1990. New stochastic approximation type procedures. Avtomatica i Telemekhanika , 7: 98-107.

Robbins, H.; and Monro, S. 1951. A Stochastic Approximation Method. The Annals of Mathematical Statistics , 22(3): 400 - 407.

Sutton, R.; and Barto, A. 2018. Reinforcement Learning: An Introduction . Cambridge, MA, USA: A Bradford Book. ISBN 0262039249.

Sutton, R.; Maei, H.; Precup, D.; Bhatnagar, S.; Silver, D.; Szepesv´ ari, C.; and Wiewiora, E. 2009. Fast GradientDescent Methods for Temporal-Difference Learning with Linear Function Approximation. In Proceedings of the 26th Annual International Conference on Machine Learning , ICML '09, 993-1000. New York, NY, USA: Association for Computing Machinery. ISBN 9781605585161.

Sutton, R. S. 1988. Learning to Predict By the Methods of Temporal Differences. Machine Learning , 3(1): 9-44.

Sutton, R. S.; Maei, H.; and Szepesv´ ari, C. 2009. A Convergent O(n) Temporal-difference Algorithm for Off-policy Learning with Linear Function Approximation. In Koller, D.; Schuurmans, D.; Bengio, Y.; and Bottou, L., eds., Advances in Neural Information Processing Systems , volume 21. Curran Associates, Inc.

Tsitsiklis, J.; and Van Roy, B. 1997. An analysis of temporaldifference learning with function approximation. IEEE Transactions on Automatic Control , 42(5): 674-690.

## Appendix

## A1 Proof of Theorem 2

Consider the One timescale recursion for the GTD-M iterates given by (21) as given below:

<!-- formula-not-decoded -->

Here h ( ψ ) = g + Gψ,g = E [ g t ] , G = E [ G t ] , where the expectations are w.r.t. the stationary distribution of the Markov chain induced by the target policy π . M t +1 = ( G t +1 -G ) ψ t +( g t +1 -g ) . In particular,

<!-- formula-not-decoded -->

where recall that ¯ A = E [ φ ( γφ ′ -φ ) T ] and ¯ b = E [ rφ ] We show that the conditions ( A1 ) -( A4 ) in Chapter 2 of (Borkar 2008b) hold and thereafter use Theorem 2 of (Borkar 2008b) to show convergence to the TD solution.

- (A1) The map h ( ψ ) is linear in ψ and therefore Lipschitz continuous with Lipschitz constant || G || .
- (A2) The step-size sequence ξ t satisfies the required conditions (cf. assumption A 2 of the current paper).
- (A3) By construction M t +1 is a martingale difference sequence w.r.t the filtration F t = σ ( ψ 0 , M k , k ≤ t ) . Also, E [ || ( G t +1 -G ) ψ t + ( g t +1 -g ) || 2 |F t ] ≤ 2( || ( G t +1 -G ) || 2 || ψ t || 2 + || g t +1 -g || 2 ) . (A3) is satisfied with K = 2max( || ( G t +1 -G ) || 2 + || g t +1 -g || 2 ) . K &lt; ∞ follows from the fact that the rewards are uniformly bounded and the features are normalized (see assumption A 1 ).
- (A4) To ensure (A4), we show that (A5) of (Chapter 3, pp.22, Borkar (2008b)) holds and then use Theorem 7 of (Borkar 2008b). The functions h c ( x ) = h ( cx ) c = g c + Gψ,c ≥ 1 . For any compact set H , h c → h ∞ as c →∞ = Gψ uniformly. Consider the ODE

<!-- formula-not-decoded -->

Observe that since || φ t || ≤ 1 and r t ≤ 1 ∀ t , we have || A || 2 &lt; 2 . Since we have assumed that w ≥ 1 therefore, w ( w +1) ≥ || A || 2 , and hence from lemma 1, we have that G is Hurwitz. Hence, the origin is a unique globally asymptotically stable equilibrium (g.a.s.e) for the above ODE. This in turn implies that the iterates remain bounded i.e., sup t || ψ t || &lt; ∞ a.s. ∀ t . By (Theorem 2, Chapter 2 of Borkar (2008b)) ψ t converges to an internally chain transitive invariant set of the ODE ˙ ψ ( t ) = h ( ψ ( t )) = g + Gψ ( t ) . The only such point of the ODE is its equilibrium point -G -1 g . By (Corollary 4, Chapter 2 of Borkar (2008b)),

<!-- formula-not-decoded -->

A straightforward calculation for the inverse of the 3 × 3 block matrix G gives us that

<!-- formula-not-decoded -->

## A2 Proof of Theorem 3

We first start by assuming that the iterates remain stable (cf. assumption (B5) ) and show that the three timescale recursions converge. Subsequently we provide conditions which ensure that the iterates remain stable. We consider general three timescale recursions as given below:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where x n ∈ R d 1 , y n ∈ R d 2 and z n ∈ R d 3 ∀ n ≥ 0 . Next we consider the following assumptions:

(B1) h : R d 1 + d 2 + d 3 → R d 1 , g : R d 1 + d 2 + d 3 → R d 2 , f : R d 1 + d 2 + d 3 → R d 3 are Lipschitz continuous.

- (B2) { M (1) n } , { M (2) n } , { M (3) n } are Martingale difference sequences w.r.t. {F n } where,

<!-- formula-not-decoded -->

for some constants K i &gt; 0 , i = 1 , 2 , 3 .

- (B3) { a ( n ) } , { b ( n ) } , { c ( n ) } are step-size sequences that satisfy a ( n ) &gt; 0 , b ( n ) &gt; 0 , c ( n ) &gt; 0 , ∀ n ≥ 0

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (B4) (i) The ODE ˙ x ( t ) = h ( x ( t ) , y, z ) , y ∈ R d 2 , z ∈ R d 3 has a globally asymptotically stable equilibrium λ ( y, z ) , where λ : R d 2 × d 3 → R d 1 is Lipschitz continuous.
- (ii) The ODE ˙ y ( t ) = g ( λ ( y ( t ) , z ) , y ( t ) , z ) , z ∈ R d 3 has a globally asymptotically stable equilibrium Γ( z ) , where Γ : R d 3 → R d 2 is Lipschitz continuous.
- (iii) The ODE ˙ z ( t ) = f ( λ (Γ( z ( t )) , z ( t )) , Γ( z ( t )) , z ( t )) , has a globally asymptotically stable equilibrium z ∗ ∈ R d 3 (B5) sup n ( || x n || + || y n || + || z n || ) &lt; ∞ a.s.

Theorem 5. Under ( B1 ) -( B5 ) the iterates given by (25) , (26) and (27) ,

<!-- formula-not-decoded -->

Proof. We start with the following Lemma that characterizes the set to which the iterates converge.

<!-- formula-not-decoded -->

Proof. We first consider the fastest timescale of { a ( n ) } and show that:

<!-- formula-not-decoded -->

We rewrite the iterates (25), (26) and (27) as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using third extension from Chapter-2 of (Borkar 2008a), ( x n , y n , z n ) converges to an internally chain transitive invariant set of the ODE

<!-- formula-not-decoded -->

For initial conditions x ∈ R d 1 , y ∈ R d 2 , z ∈ R d 3 , the internally chain transitive invariant set of the above ODE is { ( λ ( y, z ) , y, z ) } . Therefore,

<!-- formula-not-decoded -->

Next we consider the middle timescale { b ( n ) } . (26) and (27) can be re-written as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The iteration for { y n } can be re-written as:

<!-- formula-not-decoded -->

where,

where, where, Combining (31) and (34) we get:

Using Lemma 6, we get:

(

x

n

, y

n

<!-- formula-not-decoded -->

Since, ( x n , y n , z n ) → { ( λ ( y, z ) , y, z ) : y ∈ R d 2 , z ∈ R d 3 } , therefore || glyph[epsilon1] (2) ,b n || → 0 as n → ∞ . Again using third extension from Chapter-2 of (Borkar 2008a), it can be seen that (32) and (33) converges to an internally chain transitive invariant set of the ODE

<!-- formula-not-decoded -->

For initial conditions y ∈ R d 2 , z ∈ R d 3 , the internally chain transitive invariant set of the above ODE is { (Γ( z ) , z ) } . Therefore,

<!-- formula-not-decoded -->

, z

n

)

→{

(

λ

(Γ(

z

)

, z

)

,

Γ(

z

)

, z

) :

z

∈

R

d

3

}

.

Finally, we consider the slowest timescale of { c ( n ) } . We define the piece wise linear continuous interpolation of the iterates z n as:

<!-- formula-not-decoded -->

where, t ( n ) = ∑ n -1 m =0 c ( n ) , n ≥ 1 . Also, let z s ( t ) , t ≥ s , denote the unique solution to the below ODE starting at s ∈ R :

<!-- formula-not-decoded -->

with z s ( s ) = ¯ z ( s ) . Using the arguments as in Theorem-2, Chapter-6 of (Borkar 2008a), it can be shown that for any T &gt; 0

<!-- formula-not-decoded -->

Subsequently arguing as in proof of Theorem-2, Chapter-2 of (Borkar 2008a), we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next we provide sufficient conditions for ( B5 ) to hold. Consider the following additional assumptions:

- (B6) The functions h c ( x, y, z ) glyph[defines] h ( cx,cy,cz ) c , c ≥ 1 satisfy h c → h ∞ as c →∞ uniformly on compacts. For fixed y ∈ R d 2 , z ∈ R d 3 , the ODE

<!-- formula-not-decoded -->

has its unique globally asymptotically stable equilibrium λ ∞ ( y, z ) , where λ ∞ : R d 2 + d 3 → R d 1 is Lipschitz continuous. Further, λ ∞ (0 , 0) = 0 , i.e.,

<!-- formula-not-decoded -->

has origin in R d 1 as unique globally asymptotically stable equilibrium.

- (B7) The functions g c ( y, z ) glyph[defines] g ( cλ ∞ ( y,z ) ,cy,cz ) c , c ≥ 1 satisfy g c → g ∞ as c →∞ uniformly on compacts. For fixed z ∈ R d 3 , the ODE

<!-- formula-not-decoded -->

has its unique globally asymptotically stable equilibrium Γ ∞ ( z ) , where Γ ∞ : R d 3 → R d 2 is Lipschitz continuous. Further, Γ ∞ (0) = 0 , i.e.,

<!-- formula-not-decoded -->

has origin in R d 2 as its unique globally asymptotically stable equilibrium.

- (B8) The functions f c ( z ) glyph[defines] f ( cλ ∞ (Γ ∞ ( z ) ,z ) ,c Γ ∞ ( z ) ,cz ) , c ≥ 1 satisfy f c → f ∞ as c →∞ uniformly on compacts. The ODE

<!-- formula-not-decoded -->

- c

has the origin in R d 3 as its unique globally asymptotically stable equilibrium.

Theorem 7. Under assumptions ( B1 ) -( B4 ) and ( B6 ) -( B8 ) ,

<!-- formula-not-decoded -->

Proof. We begin with the fastest time scale determined by the step size a ( n ) . Consider the following definitions:

(F1) Define

Let ψ k = ( x k , y k , z k ) , k ≥ 0 , and

<!-- formula-not-decoded -->

- (F2) Given t ( n ) , n ≥ 0 and a constant T &gt; 0 define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

One can find a subsequence { m ( n ) } such that T n = t ( m ( n )) ∀ n and m ( n ) →∞ as n →∞ .

The scaling sequence is defined as:

- (F3)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (F4) The scaled iterates for m ( n ) ≤ k ≤ m ( n +1) -1 are:

where, c = r ( n ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (F5) Next we define the linearly interpolated trajectory for the scaled iterates as follows:

<!-- formula-not-decoded -->

(F6) Let ψ n ( t ) = ( x n ( t ) , y n ( t ) , z n ( t )) , t ∈ [ T n , T n +1 ] denote the trajectory of the ODE:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

First we state four lemmas for ODEs with two external inputs. The proofs of these lemmas follow exactly as Lemmas 2, 3, 4 and 5 of (Lakshminarayanan and Bhatnagar 2017). Subsequently when we analyze the middle timescale (timescale of { b ( n ) } ) and slow timescale (timescale of { c ( n ) } ) recursions, we restate the corresponding lemmas for ODEs with one and no external inputs respectively. Let x y ( t ) ,z ( t ) c ( t, x ) and x y ( t ) ,z ( t ) ∞ ( t, x ) denote the solution to the ODEs

<!-- formula-not-decoded -->

respectively, with initial condition x ∈ R d 1 and the external inputs y ( t ) ∈ R d 2 and z ( t ) ∈ R d 3 . Throughout the paper, B ( x, r ) glyph[defines] { q ∈ R d 1 ∣ ∣ ∣ || q -x || &lt; r } , B ( y, r ) glyph[defines] { q ∈ R d 2 ∣ ∣ ∣ || q -y || &lt; r } and B ( z, r ) glyph[defines] { q ∈ R d 3 ∣ ∣ ∣ || q -z || &lt; r } denote the ball of radius r around x, y and z respectively.

Lemma 8. Let K ⊂ R d 1 be a compact set, y ∈ R d 2 and z ∈ R d 3 be fixed external inputs. Then under ( B6 ) , given δ &gt; 0 , ∃ T δ &gt; 0 such that ∀ x ∈ K

<!-- formula-not-decoded -->

Lemma 9. Let x ∈ R d 1 , y ∈ R d 2 , z ∈ R d 3 , [0 , T ] be a given time interval and r &gt; 0 . Let y ′ ( t ) ∈ B ( y, r ) , z ′ ( t ) ∈ B ( z, r ) ∀ t ∈ [0 , T ] , then

<!-- formula-not-decoded -->

where glyph[epsilon1] ( c ) → 0 as c →∞ .

Lemma 10. Let y ∈ R d 2 , z ∈ R d 3 then given glyph[epsilon1] &gt; 0 and T &gt; 0 , ∃ c glyph[epsilon1],T &gt; 0 , δ glyph[epsilon1],T &gt; 0 and r glyph[epsilon1],T &gt; 0 such that ∀ t ∈ [0 , T ) , ∀ x ∈ B ( λ ∞ ( y, z ) , δ glyph[epsilon1],T ) ∀ c &gt; c glyph[epsilon1],T and external inputs y ′ ( s ) ∈ B ( y, r glyph[epsilon1],T ) and z ′ ( s ) ∈ B ( z, r glyph[epsilon1],T ) . Then,

<!-- formula-not-decoded -->

Lemma 11. Let x ∈ B (0 , 1) ⊂ R d 1 , y ∈ K ′ ⊂ R d 2 , z ∈ K ′′ ⊂ R d 3 and let ( B6 ) hold. Then given glyph[epsilon1] &gt; 0 , ∃ c glyph[epsilon1] ≥ 1 , r glyph[epsilon1] &gt; 0 and T glyph[epsilon1] &gt; 0 such that for any external input satisfying y ′ ( s ) ∈ B ( y, r glyph[epsilon1] ) , z ′ ( s ) ∈ B ( z, r glyph[epsilon1] ) , ∀ s ∈ [0 , T ] ,

<!-- formula-not-decoded -->

The next lemma uses the convergence result of three time scale iterates under the stability assumption of ( B5 ) (Theorem 5) and shows that the scaled iterates defined in ( F4 ) converge.

Lemma 12. Under ( B1 ) -( B3 ) ,

- (i) For 0 ≤ k ≤ m ( n +1) -m ( n ) , || ˆ ψ ( t ( m ( n ) + k )) || ≤ K (1) a.s. for some constant K (1) &gt; 0
- (ii) lim n →∞ || ψ ( t ) -ψ n ( t ) || = 0 a.s. ∀ t ∈ [ T n , T n +1
- . ˆ ]

Proof. (i) Follows as in (Lemma 4, Chapter-3, pp. 24, Borkar (2008a)).

- (ii) By construction, the iterates ˆ x k , ˆ y k , ˆ z k remain bounded, i.e., sup k ( || ˆ x k || + || ˆ y k || + || ˆ z k || ) &lt; ∞ a.s. Therefore, ( B1 ) -( B4 ) are satisfied. Using Theorem 5, the iterates (ˆ x n , ˆ y n , ˆ z n ) converges. Using the third extension from Chapter-2 of (Borkar 2008a), the iterates (ˆ x n , ˆ y n , ˆ z n ) track the ODE system

<!-- formula-not-decoded -->

Therefore, lim n →∞ || ˆ ψ ( t ) -ψ n ( t ) || = 0 a.s. ∀ t ∈ [ T n , T n +1 ]

In particular, Lemma 12(i) shows that along the fastest timescale between instants T n and T n +1 , the norm of the scaled iterate can grow at most by a factor K (1) starting from B (0 , 1) . Next, Lemma 12(ii) shows that the scaled iterate asymptotically tracks the ODE defined in ( F6 ) . The next theorem bounds || x n || in terms of || y n || and || z n || . We define the linearly interpolated trajectory of the three iterates as: ∀ t ∈ [ t ( n ) , t ( n +1)] ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 13. Under assumptions ( B1 ) -( B4 ) and ( B6 ) ,

- (i) For n large, and T = T 1 4 (here T is the sampling frequency as in (F2) and T 1 4 is T glyph[epsilon1] as in Lemma 11 with glyph[epsilon1] = 1 4 ), if || ¯ x ( T n ) || &gt; C a (1 + || ¯ y ( T n ) || + || ¯ z ( T n ) || ) , for some C a &gt; 0 then || ¯ x ( T n +1 ) || ≤ 3 4 || ¯ x ( T n ) || (ii) || ¯ x ( T n ) || ≤ C ∗ (1 + || ¯ y ( T n ) || + || ¯ z ( T n ) || ) a.s. for some C ∗ &gt; 0 .
- (iii) || x n || ≤ K (1 + || y n || + || z n || ) , for some K &gt; 0
- a a ∗ a ∗ a

<!-- formula-not-decoded -->

Proof. (i) We have || ¯ x ( T n ) || &gt; C a (1 + || ¯ y ( T n ) || + || ¯ z ( T n ) || ) . Since, r ( n ) = max( r ( n -1) , || ¯ ψ ( T n ) || , 1) , this implies r ( n ) ≥ || ¯ ψ ( T n ) || . Therefore, r ( n ) ≥ C a . Next we show that

For p ≥ 1 ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since, || ¯ x ( T n ) || p ≥ C a (1 + || ¯ y ( T n ) || p + || ¯ z ( T n ) || p ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore,

The second inequality follows from the fact that || ¯ y ( T n ) || p p ≤ || ¯ y ( T n ) || p p + || ¯ z ( T n ) || p p . A similar analysis proves || ˆ z ( T n ) || &lt; 1 C a . Next we show that

Here we are considering the case when iterates are blowing up. Therefore let r ( n ) = ¯ ψ ( T n ) . Then,

<!-- formula-not-decoded -->

Let y ′ ( t -T n ) = y n ( t ) and z ′ ( t -T n ) = z n ( t ) ∀ t ∈ [ T n , T n +1 ] . From lemma 11, ∃ r 1 4 , c 1 4 , T 1 4 such that

<!-- formula-not-decoded -->

whenever y ′ ( t ) ∈ B (0 , r 1 4 ) and z ′ ( t ) ∈ B (0 , r 1 4 ) . Choose C a &gt; max( c 1 4 , 2 r 1 4 ) and T = T 1 4 . Since ˙ y ( t ) = 0 , and ˙ z ( t ) = 0 for the ODE defined in (F6) , y ′ ( t -T n ) = y n ( t ) = ˆ y ( T n ) and z ′ ( t -T n ) = z n ( t ) = ˆ z ( T n ) ∀ t ∈ [ T n , T n +1 ] . From || ˆ y ( T n ) || &lt; 1 C a and || ˆ z ( T n ) || &lt; 1 C a , it follows that y ′ ( s ) ∈ B (0 , r 1 4 ) and z ′ ( s ) ∈ B (0 , r 1 4 ) ∀ s ∈ [0 , T ] . Using Lemma 12(ii), || ˆ x ( T -n +1 ) -x n ( T n +1 ) || &lt; 1 4 for large enough n . Also observe that || x n ( T n +1 ) || = || x y ′ ( t ) ,z ′ ( t ) r ( n ) ( T n +1 -T n , ˆ x ( T n )) || ≤ 1 4 .

- (M3)

Using these, we have || ˆ x ( T -n +1 ) || ≤ || ˆ x ( T -n +1 ) -x n ( T n +1 ) || + || x n ( T n +1 ) || ≤ 1 2 . Finally since

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Choosing C a &gt; max ( c 1 4 , 2 r 1 4 ) &gt; 2 , proves the claim.

(ii) and (iii) follow along the lines of arguments in (Lakshminarayanan and Bhatnagar 2017) Lemma 6 (ii) and (iii) respectively.

Next we consider the middle timescale of { b ( n ) } and re-define the following terms: (M1) Define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(M2) Given t ( n ) , n ≥ 0 and a constant T &gt; 0 define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

One can find a subsequence { m ( n ) } such that T = t ( m ( n )) ∀ n , and m ( n ) →∞ as n →∞ .

n The scaling sequence is defined as:

<!-- formula-not-decoded -->

- (M4) The scaled iterates for m ( n ) ≤ k ≤ m ( n +1) -1 are:

where, c = r ( n ) ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (M5) Next, we define the linearly interpolated trajectory for the scaled iterates as follows:

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

(M6) Let ψ n ( t ) = ( x n ( t ) , y n ( t ) , z n ( t )) , t ∈ [ T n , T n +1 ] denote the trajectory of the ODE:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As before we state a few lemmas for ODEs with one external input. These follow along the lines of Lemmas 2-5 of (Lakshminarayanan and Bhatnagar 2017). Let y z ( t ) c ( t, y ) and y z ( t ) ∞ ( t, y ) denote the solution to the ODEs

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

respectively, with initial condition y ∈ R d 1 and the external input z ( t ) ∈ R d 3 .

Lemma 14. Let K ⊂ R d 1 be a compact set and z ∈ R d 3 . Then under ( B6 ) , given δ &gt; 0 , ∃ T δ &gt; 0 such that ∀ y ∈ K

<!-- formula-not-decoded -->

Lemma 15. Let y ∈ R d 2 , z ∈ R d 3 , [0 , T ] be a given time interval and r &gt; 0 . Let z ′ ( t ) ∈ B ( z, r ) , ∀ t ∈ [0 , T ] , then

<!-- formula-not-decoded -->

where glyph[epsilon1] ( c ) → 0 as c →∞ .

Lemma 16. Let z ∈ R d 3 then given glyph[epsilon1] &gt; 0 and T &gt; 0 , ∃ c glyph[epsilon1],T &gt; 0 , δ glyph[epsilon1],T &gt; 0 and r glyph[epsilon1],T &gt; 0 such that ∀ t ∈ [0 , T ) , ∀ y ∈ B (Γ ∞ ( z ) , δ glyph[epsilon1],T ) ∀ c &gt; c glyph[epsilon1],T and external input z ′ ( s ) ∈ B ( z, r glyph[epsilon1],T ) ,

<!-- formula-not-decoded -->

Lemma 17. Let y ∈ B (0 , 1) ⊂ R d 2 , z ∈ K ′ ⊂ R d 3 , and ( B7 ) holds. Then given glyph[epsilon1] &gt; 0 , ∃ c glyph[epsilon1] ≥ 1 , r glyph[epsilon1] &gt; 0 and T glyph[epsilon1] &gt; 0 such that for any external input satisfying z ′ ( s ) ∈ B ( z, r glyph[epsilon1] ) , ∀ s ∈ [0 , T ] ,

<!-- formula-not-decoded -->

Lemma 18. Under ( B1 ) -( B3 ) ,

- (i) For 0 ≤ k ≤ m ( n +1) -m ( n ) , || ˆ ψ ( t ( m ( n ) + k )) || ≤ K (2) a.s. for some constant K (2) &gt; 0 .
- (ii) For sufficiently large n , we have sup [ T n ,T n +1 ) || ˆ y ( t ) -y n ( t ) || = glyph[epsilon1] ( c ) LTe L ( L +1) T a.s. where glyph[epsilon1] ( c ) → 0 as c →∞

Proof. See Lemma 9 of (Lakshminarayanan and Bhatnagar 2017)

Theorem 19. Assume ( B1 ) -( B4 ) and ( B6 ) -( B8 ) hold. Then, with C ∗ a as defined in Theorem 13,

- (i) For large n and T = T 1 / 8( C ∗ a +1) (here T is the sampling frequency as in (M2) and T 1 / 8( C ∗ a +1) is T glyph[epsilon1] as in Lemma 17 with glyph[epsilon1] = 1 / 8( C ∗ a +1) ), if || ¯ y ( T n ) || &gt; C b (1 + || ¯ z ( T n ) || ) , for some C b &gt; 0 , then || ¯ y ( T n +1 ) || &lt; 5 8 || ¯ y ( T n ) || .
- (iii) || y n || ≤ K ∗ b (1 + || z n || ) , for some K ∗ b &gt; 0
- (ii) || ¯ y ( T n ) || ≤ C ∗ b (1 + || ¯ z ( T n ) || ) , for some C ∗ b &gt; 0

Proof. (i) Since || ¯ y ( T n ) || &gt; C b (1 + || ¯ z ( T n ) || ) , r ( n ) &gt; C b . We first show that || ˆ z ( T n ) || &lt; 1 C b .

<!-- formula-not-decoded -->

Since || ¯ y ( T n ) || p &gt; C b (1 + || ¯ z ( T n ) || ) , || ¯ y ( T n ) || p p ) &gt; C p b || ¯ z ( T n ) || p p . Therefore,

<!-- formula-not-decoded -->

Next we show that || ˆ y ( T n ) || &gt; 1 ( C ∗ a +1)(2+ 1 C b ) , where C ∗ a is as defined in Theorem 13. Here again we are considering the case when the iterates are blowing up. Therefore let r ( n ) = || ¯ ψ ( T n ) || . Now, from Theorem 13 ,we know || ¯ x ( T n ) || ≤

K ∗ a (1 + || ¯ y ( T n ) || + || ¯ z ( T n ) || ) and therefore, r ( n ) ≤ K ∗ a (1 + || ¯ y ( T n ) || + || ¯ z ( T n ) || ) + || ¯ y ( T n ) || + || ¯ z ( T n ) || . With this we have,

<!-- formula-not-decoded -->

Now we proceed as in Theorem 13 (i). Let z ′ ( t -T n ) = z n ( t ) ∀ t ∈ [ T n , T n +1 ] . From Lemma 17, ∃ r 1 / 8( C ∗ a +1) , c 1 / 8( C ∗ a +1) , T 1 / 8( C ∗ a +1) &gt; 0 such that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

whenever z ′ ( t ) ∈ B (0 , r 1 / 8( C ∗ a +1) ) . Choose T = T 1 / 8( C ∗ a +1) . Since ˙ z ( t ) = 0 for the ODE defined in (M6) and z ′ ( t -T n ) = z n ( t ) = ˆ z ( T n ) ∀ t ∈ [ T n , T n +1 ] and we choose C b &gt; max ( c 1 / 8( C ∗ a +1) , 2 r 1 / 8( C ∗ a +1) ) from || ˆ z ( T n ) || &lt; 1 C b , it follows that z ′ ( s ) ∈ B (0 , r 1 / 8( C ∗ a +1) ) ∀ s ∈ [0 , T ] . Using Lemma 18(ii), ∃ C 1 &gt; 0 s.t. || ˆ y ( T -n +1 ) -y n ( T n +1 ) || &lt; 1 8( C ∗ a +1) for large enough n and r ( n ) &gt; C 1 . Choose C b &gt; max( c 1 / 8( C ∗ a +1) , 2 r 1 / 8( C ∗ a +1) , C 1 ) . Also observe that || y n ( T n +1 ) || = || y z ′ ( t ) r ( n ) ( T n +1 -T n , ˆ y ( T n )) || ≤ 1 8( C ∗ a +1) . Using these, we have || ˆ y ( T -n +1 ) || ≤ || ˆ y ( T -n +1 ) -y n ( T n +1 ) || + || y n ( T n +1 ) || ≤ 1 4( C ∗ a +1) . Finally since

we have

Choosing C b &gt; max ( c 1 / 8( C ∗ a +1) , 2 r 1 / 8( C ∗ a +1) , C 1 ) &gt; 2 , proves the claim.

(ii) and (iii) follow along the lines of arguments in (Lakshminarayanan and Bhatnagar 2017), Lemma 6 (ii) and (iii), respectively.

Finally we consider the slowest timescale corresponding to { c ( n ) } . As before we redefine the terms as follows:

(S1) Define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(S2) Given t ( n ) , n ≥ 0 and a constant T &gt; 0 define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

There exists some subsequence { m ( n ) } such that T n = t ( m ( n )) and m ( n ) →∞ as n →∞ . (S3) The scaling sequence is defined as:

<!-- formula-not-decoded -->

- (S4) The scaled iterates for m ( n ) ≤ k ≤ m ( n +1) -1 are:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- (S5) Next we define the linearly interpolated trajectory for the scaled iterates as follows:

<!-- formula-not-decoded -->

(S6) Let ψ n ( t ) = ( x n ( t ) , y n ( t ) , z n ( t )) , t ∈ [ T n , T n +1 ] denote the trajectory of the ODE:

<!-- formula-not-decoded -->

with x n ( T n ) = ˆ x ( T n ) , y n ( T n ) = ˆ y ( T n ) and z n ( T n ) = ˆ z ( T n ) .

We again state some results on ODEs, this time with no external input. These again follow along the lines of Lemma 2-5 in (Lakshminarayanan and Bhatnagar 2017). Let z c ( t, z ) and z ∞ ( t, z ) denote the solution to the ODEs

<!-- formula-not-decoded -->

where, c = r ( n ) ,

respectively with initial condition z ∈ R d 3 .

Lemma 20. Let K ⊂ R d 3 be a compact set . Then under ( B8 ) , given δ &gt; 0 , ∃ T δ &gt; 0 such that ∀ z ∈ K

<!-- formula-not-decoded -->

Lemma 21. Let z ∈ R d 3 , [0 , T ] be a given time interval and r &gt; 0 . Then

<!-- formula-not-decoded -->

where glyph[epsilon1] ( c ) → 0 as c →∞ .

Lemma 22. Given glyph[epsilon1] &gt; 0 and T &gt; 0 ∃ c glyph[epsilon1],T &gt; 0 , δ glyph[epsilon1],T &gt; 0 and r glyph[epsilon1],T &gt; 0 such that ∀ t ∈ [0 , T ) , ∀ z ∈ B (0 , δ glyph[epsilon1],T ) , ∀ c &gt; c glyph[epsilon1],T ,

<!-- formula-not-decoded -->

Lemma 23. Let z ∈ B (0 , 1) ⊂ R d 3 and let ( B8 ) hold. Then given glyph[epsilon1] &gt; 0 , ∃ c glyph[epsilon1] ≥ 1 , r glyph[epsilon1] &gt; 0 and T glyph[epsilon1] &gt; 0 , then

<!-- formula-not-decoded -->

Lemma 24. Under ( B1 ) -( B3 ) ,

- (i) For 0 ≤ k ≤ m ( n +1) -m ( n ) , || ˆ ψ ( t ( m ( n ) + k )) || ≤ K (3) a.s. for some constant K (3) &gt; 0 .
- (ii) For sufficiently large n , we have sup [ T n ,T n +1 ) || ˆ z ( t ) -z n ( t ) || = ( glyph[epsilon1] 1 ( c )+ glyph[epsilon1] 2 ( c )) LTe L ( L +1) T a.s. where glyph[epsilon1] ( c ) → 0 as c →∞ .

Proof. See Lemma 9 (ii) and (iii) of (Lakshminarayanan and Bhatnagar 2017).

Theorem 25. Under assumptions ( B1 ) -( B4 ) and ( B6 ) -( B8 ) , we have:

- (i) Let C ∗ a and C ∗ b be as in Theorems 13 and 19 respectively. Then, || ˆ z ( T n ) || ≥ 1 4+ C ∗ a C ∗ b + C ∗ b for sufficiently large || ¯ z ( T n ) || .
- (ii) For n large, T = T 1 4 (here T is the sampling frequency as in (F2) and T 1 4 is T glyph[epsilon1] as in Lemma 11 with glyph[epsilon1] = 1 4 ), if || ¯ z ( T n ) || &gt; C ,
- (iii) || ¯ z ( T n ) || ≤ K ∗ c for some K ∗ c &gt; 0 .

for some C &gt; 0 then || ¯ z ( T n +1 ) || &lt; 1 2 || ¯ z ( T n ) ||

(iv) sup n || z n || &lt; ∞ a.s.

Proof. (i) From Theorems 13 and 19 we know that || r ( n ) || &lt; C ∗ a (1 + || ¯ y ( T n ) || + || ¯ z ( T n ) || ) + C ∗ b (1 + || ¯ z ( T n ) || ) + || ¯ z ( T n ) || . Therefore,

<!-- formula-not-decoded -->

- (ii) Since, 0 ∈ R d 3 is the unique globally asymptotically stable equilibrium, therefore using Lemma 23, ∃ c 1 4 , T 1 4 &gt; 0 , such that || z c ( t, z ) || &lt; 1 4(4+ C ∗ a C ∗ b + C ∗ b ) , ∀ c ≥ c 1 4 , t ≥ T 1 4 . Also, for || ¯ z ( T n ) || &gt; max( C ∗ a , C ∗ b , C ∗ a C ∗ b ) we have || ˆ z ( T n ) || &gt; 1 4+ C ∗ a C ∗ b + C ∗ b and for sufficiently large n , from Lemma 24(ii), ∃ C 2 &gt; 0 such that || ˆ z ( T -n +1 ) -z n ( T n +1 ) || &lt; 1 4(4+ C ∗ a C ∗ b + C ∗ b ) for r ( n ) &gt; C 2 . We pick C = max( c 1 / 4 , C 1 , max( C ∗ a , C ∗ b , C ∗ a C ∗ b )) and T = T 1 / 4 . For n large it then follows that || ˆ z ( T -n +1 ) || ≤ || ˆ z ( T -n +1 ) -z n ( T n +1 ) || + || z n ( T n +1 ) || ≤ 1 2(4+ K ∗ a C ∗ b + C ∗ b ) . Finally, since

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

it follows that

(iii) and (iv) follow along the lines of arguments as in Lemma 10 (iii) and (iv) of (Lakshminarayanan and Bhatnagar 2017).

Now from Theorem 25 (iii), it follows that the slow timescale iterates z n are bounded a.s. ( || z n || &lt; ∞ a.s. ) which in turn implies that the middle timescale iterates y n are bounded using Theorem 19 ( i.e., || y n || &lt; ∞ a.s. ). Finally the fast timescale iterates x n are bounded because of Theorem 13 and the fact that both middle timescale and slow timescale iterates are bounded showing || x n || &lt; ∞ a.s. Combining these we have sup n ( || x n || + || y n || + || z n || ) &lt; ∞ a.s, thereby proving Theorem 7.

The slightly more general version where each iterate could have small perturbation terms as given below:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with glyph[epsilon1] ( k ) n = o (1) , k = 1 , 2 , 3 can be shown to converge to the same solution. Since the additional error terms are o (1) , their contribution is asymptotically negligible. See arguments in third extension of (Chapter 2, pp. 17 of Borkar (2008b) ) that handles this case for one-timescale iterates.

## A3 Convergence of GTD-2 M and TDC-M

Here we provide the asymptotic convergence guarantees of the momentum variants of the remaining two Gradient TD methods namely GTD2-M and TDC-M . The analysis is similar to that of GTD-M in Theorem 4 and is provided here for completeness. We show that the assumptions (B1) - (B7) of the main paper are satisfied and thereby invoke Theorem 3 to show convergence.

## A3.1 Asymptotic convergence of GTD2-M

We re-write the iterates for GTD2-M below:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As before, choosing η t = glyph[rho1] t -wα t glyph[rho1] t -1 , where { glyph[rho1] t } is a positive sequence and w ∈ R is a constant, we can decompose the two iterates into three recursions as below:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 26. Assume A 1 , A 3 and A 4 hold and let w &gt; 0 . Then, the GTD2-M iterates given by (39) and (40) satisfy θ n → θ ∗ = -¯ A -1 ¯ b a.s. as n →∞ .

Proof. We transform the iterates given by (41), (42) and (43) into the standard SA form given by (22), (23) and (24). Let F t = σ ( u 0 , v 0 , θ 0 , r j +1 , φ j , φ ′ j : j &lt; t ) . Let, A t = φ t ( γφ ′ t -φ t ) T and b t = r t +1 φ t . Then, (41) can be re-written as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, C t = φ t φ T t and ¯ C = E [ φ t φ T t ] . Finally, (43) can be re-written as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The functions h, g, f are linear in v, u, θ and hence Lipchitz continuous, therefore satisfying ( B1 ) . We choose the step-size sequences such that they satisfy ( B2 ) . One popular choice is

<!-- formula-not-decoded -->

Next, M (1) t +1 , M (2) t +1 and M (3) t +1 t ≥ 0 , are martingale difference sequences w.r.t F t by construction. Next,

<!-- formula-not-decoded -->

The first part of ( B3 ) is satisfied with K 1 = || ( ¯ A T -A T t ) || 2 , K 2 = 3max( || A t -¯ A || 2 , || b t -¯ b || 2 , || ( ¯ C -C t ) || 2 ) and any K 3 &gt; 0 . The fact that K 1 , K 2 &lt; ∞ follows from the bounded features and bounded rewards assumption in A 1 . Next, observe that || ε (3) t || = ξ t || ( ( φ t -γφ ′ t ) φ T t u t -wv t ) ||→ 0 since ξ t → 0 as t →∞ . For a fixed u, θ ∈ R d , consider the ODE

<!-- formula-not-decoded -->

For w &gt; 0 , λ ( u, θ ) = -¯ A T u w is the unique g.a.s.e, is linear and therefore Lipchitz continuous. This satisfies ( B4 ) (i). Next, for a fixed θ ∈ R d ,

<!-- formula-not-decoded -->

has Γ( θ ) = ¯ C -1 ( ¯ Aθ + ¯ b ) as its unique g.a.s.e because -¯ C -1 is negative definite. Also Γ( θ ) is linear in θ and therefore Lipschitz. This satisfies ( B4 )( ii ) . Finally, to satisfy ( B4 )( iii ) , consider,

<!-- formula-not-decoded -->

w Since ¯ A is negative definite and ¯ C is positive definite, therefore, -¯ A T ¯ C -1 ¯ A is negative definite. Therefore, θ ∗ = -¯ A -1 ¯ b is the unique g.a.s.e.

Next, we show that the sufficient conditions for stability of the three iterates are satisfied. The function, h c ( v, u, θ ) = -c ¯ A T u -wcv c = -¯ A T u -wv → h ∞ ( v, u, θ ) = -¯ A T u -wv uniformly on compacts as c →∞ . The limiting ODE:

<!-- formula-not-decoded -->

has λ ∞ ( u, θ ) = -¯ A T u w as its unique g.a.s.e. λ ∞ is Lipschitz with λ ∞ (0 , 0) = 0 , thus satisfying assumption ( B5 ) .

where,

Next, (42) can be re-written as:

where,

where, The function, g c ( u, θ ) = c ¯ Aθ +¯ b -c ¯ Cu c = ¯ Aθ -¯ Cu + ¯ b c → g ∞ ( u, θ ) = ¯ Aθ -¯ Cu uniformly on compacts as c →∞ . The limiting ODE

<!-- formula-not-decoded -->

has Γ ∞ ( θ ) = ¯ C -1 ¯ Aθ as its unique g.a.s.e. since -¯ C is negative definite. Γ ∞ is Lipchitz with Γ ∞ (0) = 0 . Thus assumption ( B6 ) is satisfied.

Finally, f c ( θ ) = -c ¯ A T ¯ C -1 ¯ Aθ cw → f ∞ = -¯ A T ¯ Aθ w uniformly on compacts as c →∞ and the ODE:

<!-- formula-not-decoded -->

has origin in R d as its unique g.a.s.e. This ensures the final condition ( B7 ) . By theorem 3,

<!-- formula-not-decoded -->

Specifically, θ t →-¯ A -1 ¯ b .

## A3.2 Asymptotic Convergence of TDC-M

We re-write the iterates for TDC-M below:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As before, choosing η t = glyph[rho1] t -wα t glyph[rho1] t -1 , where { glyph[rho1] t } is a positive sequence and w ∈ R is a constant, we can decompose the two iterates into three recursions as below:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 27. Assume A 1 , A 3 and A 4 hold and let w &gt; 0 . Then, the TDC-M iterates given by (44) and (45) satisfy θ n → θ ∗ = -¯ A -1 ¯ b a.s. as n →∞ .

Proof. We transform the iterates given by (46), (47) and (48) into the standard SA form given by (22), (23) and (24). Let F t = σ ( u 0 , v 0 , θ 0 , r j +1 , φ j , φ ′ j : j &lt; t ) . Let, A t = φ t ( γφ ′ t -φ t ) T and b t = r t +1 φ t . Then, (46) can be re-written as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next, (46) can be re-written as:

where,

where,

where,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, C t = φ t φ T t and ¯ C = E [ φ t φ T t ] . Finally, (46) can be re-written as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The functions h, g, f are linear in v, u, θ and hence Lipchitz continuous, therefore satisfying ( B1 ) . We choose the step-size sequences such that they satisfy ( B2 ) . One popular choice is

<!-- formula-not-decoded -->

Observe that, M (1) t +1 , M (2) t +1 and M (3) t +1 t ≥ 0 , are martingale difference sequences w.r.t F t by construction. Next,

<!-- formula-not-decoded -->

. The first part of ( B3 ) is satisfied with K 1 = 3max( || ( A t -¯ A ) || 2 , || ( b t -¯ b ) || 2 , γ ( || E [ φ ′ t φ T t ] -φ ′ t φ T t || 2 )) , K 2 = 3max( || A t -¯ A || 2 , || b t -¯ b || 2 , || ( ¯ C -C t ) || 2 ) and any K 3 &gt; 0 . The fact that K 1 , K 2 &lt; ∞ follows from the bounded features and bounded rewards assumption in A 1 . Next, observe that || ε (3) t || = ξ t || ( ( φ t -γφ ′ t ) φ T t u t -wv t ) ||→ 0 since ξ t → 0 as t →∞ . For a fixed u, θ ∈ R d , consider the ODE

<!-- formula-not-decoded -->

For w &gt; 0 , λ ( u, θ ) = ¯ Aθ +¯ b -γ E [ φ ′ t φ T t ] u w is the unique g.a.s.e, is linear and therefore Lipchitz continuous. This satisfies ( B4 ) (i). Next, for a fixed θ ∈ R d ,

<!-- formula-not-decoded -->

has Γ( θ ) = ¯ C -1 ( ¯ Aθ + ¯ b ) as its unique g.a.s.e because -¯ C -1 is negative definite. Also Γ( θ ) is linear in θ and therefore Lipschitz. This satisfies ( B4 )( ii ) . Finally, to satisfy ( B4 )( iii ) , consider,

<!-- formula-not-decoded -->

w Now, ( I -γ E [ φ ′ t φ T t ] ¯ C -1 ) ¯ A = ( E [ φ t φ T t ] -γ E [ φ ′ t φ T t ]) ¯ C -1 ¯ A = E [( φ t -γφ ′ t ) φ T t ] ¯ C -1 ¯ A = -¯ A T ¯ C -1 ¯ A . Since, ¯ A is negative definite and ¯ C is positive definite, therefore -¯ A T ¯ C -1 ¯ A is negative definite and hence the above ODE has θ ∗ = -¯ A -1 ¯ b as its unique g.a.s.e.

Next, we show that the sufficient conditions for stability of the three iterates are satisfied. The function, h c ( v, u, θ ) = c ¯ Aθ +¯ b -cγ E [ φ ′ t φ T t ] u -cwv c = ¯ Aθ t -γ E [ φ ′ t φ T t ] u t -wv t → h ∞ ( v, u, θ ) = ¯ Aθ t -γ E [ φ ′ t φ T t ] u t -wv t uniformly on compacts as c →∞ . The limiting ODE:

<!-- formula-not-decoded -->

has λ ∞ ( u, θ ) = ¯ Aθ -γ E [ φ ′ t φ T t ] u w as its unique g.a.s.e. λ ∞ is Lipschitz with λ ∞ (0 , 0) = 0 , thus satisfying assumption ( B5 ) . The function, g c ( u, θ ) = c ¯ Aθ +¯ b -c ¯ Cu c = ¯ Aθ -¯ Cu + ¯ b c → g ∞ ( u, θ ) = -¯ Aθ -¯ Cu uniformly on compacts as c → ∞ . The

<!-- formula-not-decoded -->

limiting ODE

has Γ ∞ ( θ ) = ¯ C -1 ¯ Aθ as its unique g.a.s.e. since -¯ C is negative definite. Γ ∞ is Lipschitz with Γ ∞ (0) = 0 . Thus assumption ( B6 ) is satisfied.

Finally, f c ( θ ) = c ¯ Aθ -cγ E [ φ ′ t φ T t ] ¯ C -1 ¯ Aθ cw → f ∞ = ( I -γ E [ φ ′ t φ T t ] ¯ C -1 ) ¯ Aθ w uniformly on compacts as c →∞ . Consider the ODE:

<!-- formula-not-decoded -->

w Now, ( I -γ E [ φ ′ t φ T t ] ¯ C -1 ) ¯ A = ( E [ φ t φ T t ] -γ E [ φ ′ t φ T t ]) ¯ C -1 ¯ A = E [( φ t -γφ ′ t ) φ T t ] ¯ C -1 ¯ A = -¯ A T ¯ C -1 ¯ A . Since, ¯ A is negative definite and ¯ C is positive definite, therefore -¯ A T ¯ C -1 ¯ A is negative definite and hence the above ODE has origin as its unique g.a.s.e. This ensures the final condition ( B7 ) . By Theorem 3,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## A4 Experiment Details

Here we briefly describe the MDP settings considered in section 5.

1. Example-1 (Boyan Chain): It consists of a linear arrangement of 14 states. From each of the first 13 states, one can move to the next state or the next to next state with equal probability. The last state is an absorbing state. The reward at each transition is -3 except the transition from state-6 to state-7 where it is -2. The discount factor γ is set to 0 . 95 . The following figure shows the corresponding MDP for 7 state Boyan Chain.

Specifically, θ t →-¯ A -1 ¯ b .

Figure 5: 7 state Boyan Chain from (Boyan 1999)

<!-- image -->

2. Example-2 (5-State Random Walk): It consists of a linear arrangement of 5 states with two terminal states. There is a single action at each state. From each state one either moves left or right with equal probability. Moving left from state 1 results in episode termination yielding a reward of 0. Similarly, moving right from state 5 also results in episode termination, however, yielding a reward of +1. The reward associated with all other transitions is 0 and the discount factor γ = 1 . The following figure shows the corresponding MDP.
3. Example-3 (19-State Random Walk): It consists of a linear arrangement of 19 states. From each state one either moves left or right with equal probability. Moving left from state 1 results in episode termination yielding a reward of -1. Similarly, moving right from state 19 also results in episode termination, however, yielding a reward of +1. The reward associated with all other transitions is 0 and the discount factor γ = 1 . The following figure shows the corresponding MDP:
4. Example-4 (Random MDP): This is a randomly generated discrete MDP with 20 states and 5 actions in each state. The transition probabilities are uniformly generated from [0 , 1] with a small additive constant. The rewards are also uniformly generated from [0 , 1] . The policy and the start state distribution are also generated in a similar way and the discount factor γ = 0 . 95 . See (Dann, Neumann, and Peters 2014) for a more detailed description.

Figure 6: 5-State Random Walk from (Sutton et al. 2009)

<!-- image -->

Figure 7: 19 State Random Walk from (Sutton and Barto 2018)

<!-- image -->