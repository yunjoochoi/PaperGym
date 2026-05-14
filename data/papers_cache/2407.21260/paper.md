## Bellman Unbiasedness: Toward Provably Efficient Distributional Reinforcement Learning with General Value Function Approximation

Taehyun Cho 1 Seungyub Han 1 Seokhun Ju 1 Dohyeong Kim 1 Kyungjae Lee 2 Jungwoo Lee 1

## Abstract

Distributional reinforcement learning improves performance by capturing environmental stochasticity, but a comprehensive theoretical understanding of its effectiveness remains elusive. In addition, the intractable element of the infinite dimensionality of distributions has been overlooked. In this paper, we present a regret analysis of distributional reinforcement learning with general value function approximation in a finite episodic Markov decision process setting. We first introduce a key notion of Bellman unbiasedness which is essential for exactly learnable and provably efficient distributional updates in an online manner. Among all types of statistical functionals for representing infinite-dimensional return distributions, our theoretical results demonstrate that only moment functionals can exactly capture the statistical information. Secondly, we propose a provably efficient algorithm, SF-LSVI , that achieves a tight regret bound of ˜ O ( d E H 3 2 √ K ) where H is the horizon, K is the number of episodes, and d E is the eluder dimension of a function class.

Distributional reinforcement learning (DistRL) (Bellemare et al., 2017; Rowland et al., 2019; Choi et al., 2019; Kim et al., 2024) is an advanced approach to reinforcement learning (RL) that focuses on the entire probability distribution of returns rather than solely on the expected return. By considering the full distribution of returns, distRL provides deeper insight into the uncertainty of each action, such as the mode or median. This framework enables us to make safer and more effective decisions that account for various risks (Chow et al., 2015; Son et al., 2021; Greenberg et al., 2022; Kim et al., 2023), particularly in complex realworld situations, such as robotic manipulation (Bodnar et al.,

1 Seoul National University, Seoul, South Korea 2 Korea University, Seoul, South Korea. Correspondence to: Kyungjae Lee &lt;kyungjae\_lee@korea.ac.kr&gt;, Jungwoo Lee &lt;junglee@snu.ac.kr&gt;.

Proceedings of the 42 nd International Conference on Machine Learning , Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

2019), neural response (Muller et al., 2024), stratospheric balloon navigation (Bellemare et al., 2020), algorithm discovery (Fawzi et al., 2022), and several game benchmarks (Bellemare et al., 2013; Machado et al., 2018). While the distributional approach offers richer information, two key theoretical challenges are introduced that distinguish it from expectation-based RL.

Infinite-dimensionality of distribution. In practice, distributions contain an infinite amount of information, and we must resort to approximations using a finite number of parameters or statistical functionals, such as categorical (Bellemare et al., 2017) and quantile representations (Dabney et al., 2018b). However, previous works often conducted analyses while overlooking these intractable nature of distributions. Additionally, not all statistical functionals can be exactly learned through the Bellman operator, as the meaning of statistical functionals is not preserved after updates. For example, the median is not preserved under the Bellman updates, as the median of a mixture of two distributions does not equal the mixture of their medians. Thus, a fundamental question arises: For a given statistical functional, does there exist a corresponding Bellman operator that ensures exactness? To formalize this issue, Rowland et al. (2019) introduced Bellman closedness , which characterizes statistical functionals that can be exactly learned in the presence of a corresponding Bellman operator.

Online distributional update. In this paper, we focus on developing an algorithm that efficiently explores from a regret minimization perspective while simultaneously performing distributional Bellman updates in an online manner. One possible approach to addressing this problem is to first update the policy using an existing provably efficient non-distributional RL algorithm and then estimate the distribution via additional rollouts. However, decoupling these two processes introduces several drawbacks. First, adding extra rollouts solely for distribution estimation is sample-inefficient, and the limited number of rollouts inevitably introduce accumulated approximation errors in the estimated distribution throughout the learning process. Moreover, the estimation is confined to the return distri- bution of the executed policy, making it difficult to reuse for estimating distributions under different policies, thereby moving further away from off-policyness.

Figure 1. Venn-Diagram of Statistical Functional Classes. The diagram illustrates categories of statistical functional. (Yellow ∩ Blue) Within the linear statistical functional class, Rowland et al. (2019) showed that the only functionals satisfying Bellman closedness are moment functionals. (Red ∩ Blue) We extend this concept by introducing the notion of Bellman unbiasedness , which not only covers moment functionals but also includes central moment functionals from the broader class including nonlinear statistical functionals. (Yellow ∩ Blue c ) According to Lemmas 3.2 and 4.4 of Rowland et al. (2019), categorical functionals are linear but not Bellman closed. (A) Maximum and minimum functionals are Bellman closed, while they are not unbiasedly estimatable. (B) Median and quantile functionals are neither Bellman closed nor unbiased, highlighting that they are not proper to encode the distribution in terms of exactness. The proofs corresponding to each region are provided in Appendix C.

<!-- image -->

To overcome those two fundamental challenges inherent to DistRL, we take a closer look at the distributional Bellman update and revisit what additional properties of statistical functionals, beyond Bellman closedness, are required to construct online DistRL algorithms that are not only exactly learnable but also provably efficient in terms of regret. In this context, we identify the following additional issues that arise when using statistical functionals for updates instead of the full distribution:

- Representing a mixture distribution with a finite, fixed number of parameters leads to approximation errors during the update. For example, when expressing the mixture of two distributions, each represented by N parameters, compressing 2 N into N parameters in the mixture results in inevitable information loss.
- Due to the unknown nature of the transition P ( ·| s, a ) , the target distribution is estimated by sampling the next state s ′ . Hence, the statistical functionals of the target distribution should be unbiasedly estimated using the statistical functionals from the sampled distribution.

In this paper, we introduce a key concept, Bellman unbiasedness , for precise information learnability of a distribution from a finite number of samples in an online setting. As shown in Figure 1, we prove that the moment functional remains the only solution in a class that includes nonlinear statistical functionals that satisfies both properties. We then discuss the inherent intractability of distributional Bellman completeness (distBC) - a structural assumption previously defined in the literature (Wang et al., 2023; Chen et al.,

2024) - and investigate the benefits of redesigning this concept using a collection of statistical functionals. Finally, we propose a provably efficient statistical functional RL algorithm with general value function approximation, called SF-LSVI .

In summary, our main contributions are as follows:

- Introduce a key property of Bellman unbiasedness for exactly learnable and provably efficient online distRL algorithm. We show that the moment functional is the unique structure in a class including nonlinear statistical functionals.
- Describe the inherent intractability of infinitedimensional distributions and analyze how hidden approximation error prevents the design of provably efficient algorithms. To address this, we revisit the existing structural assumption of distributional Bellman Completeness through a statistical functional lens.
- Propose exactly learnable and provably efficient distRL algorithm called SF-LSVI , achieving a tight regret upper bound ˜ O ( d E H 3 2 √ K ) . 1 Our framework yields a tighter regret bound with a weaker structural assumption compared to prior results in distRL.

## 1. Related Work

1 We ignore poly-log terms in H,S,A,K in the ˜ O ( · ) notation. 2 In Chen et al. (2024), the regret bound is written as ˜ O ( d E L ∞ ( ρ ) H √ K ) , where L ∞ ( ρ ) represents the lipschitz constant of the risk measure ρ , i.e., | ρ ( Z ) -ρ ( Z ′ ) | ≤ L ∞ ( ρ ) ∥ F Z -F Z ′ ∥ ∞ . Since L ∞ ( ρ ) ≥ H in risk-neutral setting, we translate the regret bound into ˜ O ( d E H 2 √ K ) .

Table 1. Comparison for different methods under distributional RL framework. H represents a subspace of infinite-dimensional space F ∞ . To bound the eluder dimesion d E , Wang et al. (2023) and Chen et al. (2024) assumed the discretized reward MDP.

| Algorithm                     | Regret                     | Eluder dimension d E   | Bellman Completeness      | MDP assumption                           | Finite Representation   | Exactly Learnable   |
|-------------------------------|----------------------------|------------------------|---------------------------|------------------------------------------|-------------------------|---------------------|
| O-DISCO (Wang et al., 2023)   | ˜ O ( poly ( d E H ) √ K ) | dim E ( H , ϵ )        | distributional BC         | discretized reward, small-loss bound     | ✗                       | ✗                   |
| V-EST-LSR (Chen et al., 2024) | ˜ O ( d E H 2 √ K ) 2      | dim E ( H , ϵ )        | distributional BC         | discretized reward, lipschitz continuity | ✗                       | ✗                   |
| SF-LSVI [Ours]                | ˜ O ( d E H 3 2 √ K )      | dim E ( F N , ϵ )      | statistical functional BC | none                                     | ✓                       | ✓                   |

Distributional RL. In classical RL, the Bellman equation, which is based on expected returns, has a closed-form expression. However, it remains unclear whether any statistical functionals of return distribution always have their corresponding closed-form expressions. Rowland et al. (2019) introduced the notion of Bellman closedness for collections of statistical functionals that can be updated in a closed form via Bellman update. They showed that the only Bellmanclosed statistical functionals in the discounted setting are the moments E Z ∼ η [ Z k ] . More recently, Marthe et al. (2023) proposed a general framework for distRL, where the agent plans to maximize its own utility functionals instead of expected return, formalizing this property as Bellman Optimizability . They further demonstrated that in the undiscounted setting, the only W 1 -continuous and linear Bellman optimizable statistical functionals are exponential utilities 1 λ log E Z ∼ η [exp( λZ )] .

In practice, C51 (Bellemare et al., 2017) and QR-DQN (Dabney et al., 2018b) are notable distributional RL algorithms where the convergence guarantees of sampled-based algorithms are proved (Rowland et al., 2018; 2023). Dabney et al. (2018a) expanded the class of policies on arbitrary distortion risk measures by taking the based distribution non-uniformly and improve the sample efficiency from their implicit representation of the return distribution. Cho et al. (2023) highlighted the drawbacks of optimistic exploration in distRL, introducing a randomized exploration that perturbs the return distribution when the agent selects their next action.

RL with General Value Function Approximation. Regret bounds have been studied for a long time in online RL, across various domains such as bandit (Lattimore &amp; Szepesvári, 2020; Abbasi-Yadkori et al., 2011; Russo &amp; Van Roy, 2013), tabular RL (Kakade, 2003; Auer et al., 2008; Osband &amp; Van Roy, 2016; Osband et al., 2019; Jin et al., 2018), and linear function approximation (Jin et al., 2020; Wang et al., 2019; Zanette et al., 2020). In recent years, deep RL has shown significant performance using deep neural networks as function approximators, and attempts have been made to analyze whether it is efficient in terms of general function approximation (Jin et al., 2021;

Agarwal et al., 2023). Wang et al. (2020) established a provably efficient RL algorithm with general value function approximation based on the eluder dimension d E (Russo &amp; Van Roy, 2013) and achieves a regret upper bound of ˜ O ( poly ( d E H ) √ K ) . To circumvent the intractability from computing the upper confidence bound, Ishfaq et al. (2021) injected the stochasticity on the training data and get the optimistic value function instead of upper confidence bound, enhancing computationally efficiency. Beyond risk-neutral setting, several prior works have shown regret bounds under risk-sensitive objectives (e.g., entropic risk (Fei et al., 2021; Liang &amp; Luo, 2022), CVaR (Bastani et al., 2022)), which align with our approach in that they are built on a distribution framework. Liang &amp; Luo (2022) achieved the regret upper bound of ˜ O (exp( H ) √ |S| 2 |A| H 2 K ) and the lower bound of Ω(exp( H ) √ |S||A| HK ) in tabular setting.

DistRL with General Value Function Approximation. Recently, only few efforts have aimed to bridge the gap between two fields. Wang et al. (2023) proposed a distributional RL algorithm, O-DISCO , which enjoys small-loss bound by using a log-likelihood objective. Similarly, Chen et al. (2024) provided a risk-sensitive RL framework with static lipschitz risk measure. While these studies analyze within a distributional framework, they do not address the intractability of implementation in infinite-dimensional space of distributions. In contrast, our approach focuses on a statistical functional framework, providing a detailed comparison with other distRL methods as shown in Table 1.

## 2. Preliminaries

Episodic MDP. We consider a episodic Markov decision process which is defined as a M = ( S , A , H, P , r ) characterized by state space S , action space A , horizon length H , transition kernels P = { P h } h ∈ [ H ] , and reward r = { r h } h ∈ [ H ] at step h ∈ [ H ] . The agent interacts with the environment across K episodes. For each k ∈ [ K ] and h ∈ [ H ] , H k h = ( s 1 1 , a 1 1 , . . . , s 1 H , a 1 H , . . . , s k h , a k h ) represents the history up to step h at episode k . We assume the reward is bounded by [0 , 1] and the agent always transit to terminal state s end at step H +1 with r H +1 = 0 .

Figure 2. Illustrative representation of sketch-based Bellman updates for a mixture distribution. Instead of updating the distributions directly, each sampled distribution is embedded through a sketch ψ (e.g., mean µ , quantile q i ). The transformation ϕ ψ aims to compress the mixture distribution into the same number of parameters, ensuring unbiasedness to prevent information loss.

<!-- image -->

Policy and Value Functions. A(deterministic) policy π is a collection of H functions { π h : S → A} H h =1 . Given a policy π , a step h ∈ [ H ] , and a state-action pair ( s, a ) ∈ S ×A , the Q and V -function are defined as Q π h ( s, a )(: S × A → R ) := E π [ ∑ H h ′ = h r h ′ ( s h ′ , a h ′ ) | s h = s, a h = a ] and V π h ( s )(: S → R ) := E π [ ∑ H h ′ = h r h ′ ( s h ′ , a h ′ ) | s h = s ] .

Random Variables and Distributions. For a sample space Ω , we extend the definition of the Q -function into a random variable and its distribution,

<!-- formula-not-decoded -->

Analogously, we extend the definition of V -function by introducing a bar notation.

<!-- formula-not-decoded -->

Note that ¯ Z π h ( s ) = Z π h ( s, π ( s )) and ¯ η π h ( s ) = η π h ( s, π ( s )) . We use π ⋆ to denote an optimal policy ( i.e., π ⋆ h ( ·| s ) = arg max π V π h ( s ) ) and denote V ⋆ h ( s ) = V π ⋆ h ( s ) , Q ⋆ h ( s, a ) = Q π ⋆ h ( s, a ) , η ⋆ h ( s, a ) = η π ⋆ h ( s, a ) , and ¯ η ⋆ h ( s ) = ¯ η π ⋆ h ( s ) . For notational simplicity, we denote the expectation over transition, [ P h V π h +1 ]( s, a ) = E s ′ ∼ P h ( ·| s,a ) V π h +1 ( s ′ ) , [ P h ¯ Z π h +1 ]( s, a ) = E s ′ ∼ P h ( ·| s,a ) ¯ Z π h +1 ( s ′ ) , and [ P h ¯ η π h +1 ]( s, a ) = E s ′ ∼ P h ( ·| s,a ) ¯ η π h +1 ( s ′ ) . 3 For brevity, we refer to ¯ η π simply as ¯ η .

In the episodic MDP, the agent aims to learn the optimal policy through a fixed number of interactions with the environment across a number of episodes. At the beginning of each episode k ( ∈ [ K ]) , the agent starts at the initial state s k 1 and choose a policy π k . In step h ( ∈ [ H ]) , the agent observes s k h ( ∈ S ) , takes an action a k h ( ∈ A ) ∼ π k h ( ·| s k h ) , receives a reward r h ( s k h , a k h ) , and the environment transits to the next state s k h +1 ∼ P h ( ·| s k h , a k h ) . Finally, we measure the suboptimality of an agent by its regret, which is the accumulated difference between the ground truth optimal and the return received from the interaction. The regret after K episodes is defined as Reg ( K ) = ∑ K k =1 V ⋆ 1 ( s k 1 ) -V π k 1 ( s k 1 ) .

3 Note that E s ′ ∼ P h ( ·| s,a ) ¯ η π h +1 ( s ′ ) is a mixture distribution.

Distributional Bellman Optimality Equation. Recall that η ⋆ h satisfies the following optimality equation:

<!-- formula-not-decoded -->

where B r : R → R is defined by B r ( x ) = r + x , and g # η ∈ P ( R ) is the pushforward of the distribution η through g ( i.e., g # η ( A ) = η ( g -1 ( A )) for any Borel set A ⊆ R ).

Additional Notations. For a given N , we denote an N -dimensional function class F N := F (1) ×···×F ( N ) ⊆ { f = [ f (1) , · · · , f ( N ) ] : S × A → R N } . Given a dataset D = { ( s t , a t , [ z (1) t , . . . , z ( N ) t ]) } |D| t =1 ⊆ S × A × R N , a set of state-action pairs Z = { ( s t , a t ) } |Z| t =1 ⊆ S × A and for a function f : S × A → R N , we define the norm ∥ f ( n ) ∥ ∞ , ∥ f ∥ ∞ , 1 , ∥ f ∥ D , ∥ f ∥ Z as written in Appendix A. For a set of (vector-valued) functions F N ⊆ { f : S × A → R N } , the width function of ( s, a ) is defined as w ( n ) ( F N , s, a ) := max f,g ∈F N | f ( n ) ( s, a ) -g ( n ) ( s, a ) | .

## 3. Statistical Functionals in Distributional RL

In this section, we define two key concepts in the distRL framework: the statistical functional and the sketch . We also illustrate Bellman closedness , a crucial property from Bellemare et al. (2023). Next, we introduce Bellman unbiasedness , a novel concept that complements the previous property and is essential for provable efficiency. As shown in Figure 2, quantile functionals cannot be updated in an unbiased manner (as proved in Theorem 3.3), demonstrating that only certain sketches can be updated exactly. We then show that the only sketch satisfying both properties is the moment functional, which is unique among statistical functionals. Finally, we discuss the intractability of the previous structural assumption, distributional Bellman Completeness, and its tendency to cause linear regret. To address this, we introduce statistical functional Bellman Completeness , a relaxed assumption, and explain why it satisfies both properties.

## 3.1. Bellman Closedness

Definition 3.1 (Statistical functionals, Sketch; (Bellemare et al., 2023)) . A statistical functional is a mapping from a probability distribution to a real value ψ : P ( R ) → R . A sketch is a vector-valued function ψ 1: N : P ( R ) → R N specified by an N -tuple where each component is a statistical functional,

<!-- formula-not-decoded -->

We denote the domain of sketch as P ψ 1: N ( R ) and its image as I ψ 1: N = { ψ 1: N (¯ η ) : ¯ η ∈ P ψ 1: N ( R ) } . We further extend to state return distribution functions ψ 1: N (¯ η ) = ( ψ 1: N (¯ η ( s )) : s ∈ S ) .

Definition 3.2 (Bellman closedness; (Rowland et al., 2019)) . A sketch ψ 1: N is Bellman closed if there exists an operator T ψ 1: N : I S ψ 1: N → I S ψ 1: N such that

<!-- formula-not-decoded -->

which is closed under a distributional Bellman operator T : P ( R ) S → P ( R ) S .

Bellman closedness is the property that a sketch are exactly learnable when updates are performed from the infinitedimensional distribution space to the finite-dimensional embedding space. While classical Bellman equation implies the existence of Bellman operator for expected values, not all statistical functional has such corresponding Bellman operator. Precisely, Rowland et al. (2019) showed that the only finite linear statistical functionals that are Bellman closed are given by the collections of statistical functionals where its linear span is equal to the set of exponential-polynomial functionals. 4

Theorem 3.3. Quantile functional cannot be Bellman closed under any additional sketch.

While Rowland et al. (2019) focused on "linear" statistical functionals in defining a sketch (i.e., ψ (¯ η ) = E Z ∼ ¯ η [ h ( Z )] for some h ), leaving questions about nonlinear functionals, we extend this by showing that "nonlinear" statistical functionals, such as maximum or minimum, can also be Bellman closed. Additionally, while their proof implicitly treated quantiles as linear functionals, we provide a technical clarification in Appendix C.1 where we formally demonstrate that no sketch Bellman operator exists for quantiles.

4 In discounted setting, a unique solution becomes moments. We've overwritten it for convinience.

## 3.2. Bellman Unbiasedness

While the intractability caused by infinite-dimensionality was addressed in Bellman closedness, another intractable element which has not yet fully tackled is the sampling of the next state. During the implementation, note that the agent does not have access to the transition kernel P . Instead, the agent can only access the empirical transition kernel ˆ P ( ·| s, a ) = 1 K ∑ K k =1 1 { s ′ k = · | s, a } which is derived from K sampled next states. This limitation implies that the operator should be treated as an empirical operator ˆ T ψ , rather than T ψ ( i.e., ˆ T ψ ψ (¯ η ) := ψ (( B r ) # [ ˆ P ¯ η ]) ). Therefore, we naturally introduce a new notion of Bellman unbiasedness to unbiasedly estimate the expected distribution ( B r ) # E s ′ ∼ P ( ·| s,a ) [¯ η ( s ′ )] , which is a mixture by transitions, from the sample distribution ( B r ) # ¯ η ( s ′ ) .

Definition 3.4 (Bellman unbiasedness) . A sketch ψ (= ψ 1: N ) is Bellman unbiased if a vector-valued estimator ϕ ψ = ϕ ψ ( ψ ( · ) , · · · , ψ ( · )) : ( I S ψ ) k → I S ψ exists where the sketch of expected distribution ( B r ) # E s ′ ∼ P ( ·| s,a ) [¯ η ( s ′ )] can be unbiasedly estimated by ϕ ψ using the k sampled sketches from the sample distribution ( B r ) # ¯ η ( s ′ ) , i.e.,

<!-- formula-not-decoded -->

Bellman unbiasedness is another natural definition, similar to Bellman closedness, which takes into account a finite number of samples for the transition. For example, mean-variance sketch is Bellman unbiased as the following unbiased estimator ϕ ( µ,σ 2 ) exists for k sample estimates:

<!-- formula-not-decoded -->

On the other hand, median functional is not Bellman unbiased since there is no unbiased estimator for median. Then, the following question naturally arises;

Which sketches are unbiasedly estimatable under the sketchbased Bellman update?

Figure 3. Bellman Closedness

<!-- image -->

Figure 4. Bellman Unbiasedness

Figure 5. Illustration of Bellman Closedness and Bellman Unbiasedness. The above path represents an ideal distributional Bellman update. Due to the infinite-dimensionality, the update process should be represented by using a finite-dimensional embedding (sketch) ψ . Since the transition kernel P is unknown, the below path describes that the implementation should sample the next state and update by using ˆ T ψ with the empirical transition kernel ˆ P . A sketch ψ is Bellman unbiased if ˆ T ψ ◦ ψ can unbiasedly estimate ψ ◦ T through some transformation ϕ ψ , i.e., ψ ( T η ) = E P [ ϕ ψ ◦ ˆ T ψ ( η )] .

The following lemma answers this question.

Lemma 3.5. Let F ¯ η be a CDF of the probability distribution ¯ η ∈ P ψ ( R ) S . Then a sketch is Bellman unbiased if and only if the sketch is homogeneous over P ψ ( R ) S of degree k , i.e., there exists some vector-valued function h = h ( x 1 , · · · , x k ) : X k → R N such that

<!-- formula-not-decoded -->

Lemma 3.5 states that in statistical functional dynamic programming, the unbiasedly estimatable embedding of a distribution can only be structured in the form of functions that are homogeneous of finite degree (Halmos, 1946). To illustrate that homogenity defines a broader class than linear functionals, consider the variance as a simple example. Variance is clearly not a linear linear functional, as it is non-additive. However, it can be written as

<!-- formula-not-decoded -->

which implies the homogenity of degree 2 . Taking this concept further and combining it with the results on Bellman closedness, we prove that even when including a nonlinear statistical functional, the only sketch that can be exactly learned and unbiasedly estimated in a finite-dimensional embedding space is the moment sketch.

Theorem 3.6. The only finite statistical functionals that are both Bellman unbiased and closed are given by the collections of ψ 1 , . . . , ψ N where its linear span { ∑ N n =0 α n ψ n | α n ∈ R , ∀ N } is equal to the set of exponential polynomial functionals { η → E Z ∼ η [ Z l exp( λZ )] | l = 0 , 1 , . . . , L, λ ∈ R } , where ψ 0 is the constant functional equal to 1 . In discount setting, it is equal to the linear span of the set of moment functionals { η → E Z ∼ η [ Z l ] | l = 0 , 1 , . . . , L } for some L ≤ N .

Compared to Rowland et al. (2019), we extend beyond linear statistical functionals to include nonlinear statistical functionals, showing the uniqueness of the moment functional.

As shown in Figure 1, our theoretical results not only show that high-order central moments such as variance or skewness are exactly learnable and unbiasedly estimatable, but also reveal that other nonlinear statistical functionals like median or quantiles inevitably involve approximation errors due to biased estimations.

Necessity of Bellman unbiasedness. Bellman unbiasedness ensures that updates can be unbiasedly performed when only a finite number of sampled sketches are available. In other words, it guarantees that the sequence of sampled sketches forms a martingale, enabling the construction of confidence regions through concentration inequalities. This property is crucial for establishing provable efficiency in terms of regret minimization.

## Complementary roles of unbiasedness and closedness.

At first glance, Bellman Unbiasedness (BU) may appear to be a stricter subset of Bellman Closedness (BC). However, as illustrated in Figure 1, the relationship is more subtle: for example, the categorical sketch is BU but not BC, whereas functionals like the maximum or minimum are BC but not BU. More precisely, BU guarantees the existence of an unbiased estimator of the ground-truth sketch given a finite number of sampled sketches. In contrast, BC plays a complementary role by ensuring that the update process consistently provide such sketches. If a sketch is BU but not BC-as in the case of the categorical sketch-then the update process cannot continue providing new sampled sketches, making dynamic programming infeasible.

## 3.3. Statistical Functional Bellman Completeness

We consider distributional reinforcement learning with general value function approximation (GVFA). For successful TD learning, GVFA framework for classical RL commonly requires the assumption, Bellman Completeness , that after applying Bellman operator, the output lies in the function class F (Wang et al., 2020; Ayoub et al., 2020; Ishfaq et al., 2021). As a natural extension, our approach receives a tuple of function class F N ⊆ { f : S × A → R N } as input to represent N moments of distribution. Building on this, we assume that for any ¯ η : S → P ([0 , H ]) , the sketch of target function lies in the function class F N .

Assumption 3.7 (Statistical Functional Bellman Completeness) . For any distribution ¯ η : S → P ([0 , H ]) and h ∈ [ H ] , there exists f ¯ η ∈ F N which satisfies

<!-- formula-not-decoded -->

DistBC inevitably leads to linear regret. In the seminal works, Wang et al. (2023) and Chen et al. (2024) assumed that the function class H ⊆ { η : S × A → P ([0 , H ]) } follows the distributional Bellman Completeness (distBC) assumption ( i.e., if η ∈ H for all π, h ∈ [ H ] , T π h η ∈ H ). This seems natural, but constructing a finite-dimensional subspace H that satisfies distBC is quite challenging. Since the distributional Bellman operator is a composition of translation and mixing distributions for the next state, it implies that a function class H must be closed under translation and mixture. However, when considering the representation of infinite-dimensional distributions using a finite number of representations, it is not trivial that the mixture of distributions can also be represented with the same number of representations. For example, while a Gaussian distribution can be represented using two parameters ( µ, σ 2 ) , a mixture of K Gaussians generally requires 2 K representations.

To avoid the issue of closedness under mixture, both previous studies assumed a discretized reward MDP where all outcomes of the return distribution are able to discretized into an uniform grid of finite points. Unfortunately, the approximation error introduced by the discretization is not negligible when it comes to regret. This is because model misspecification , which is the error when the model fails to represent the target, typically leads to linear regret.

Definition 3.8 (Model Misspecification in distBC) . For a given distribution class H which is the finite-dimensional subspace of the space of all distribution F ∞ , we call ζ the misspecification error

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Note that ζ is strictly positive unless the function approximator f ¯ η can represent any distribution in the finitedimensional subspace H generated by translation and mixture. In a classical linear bandit setting (Zanette et al., 2020), a lower bound with misspecification error ζ is known to yield linear regret Ω( ζK ) . Therefore, redefining Bellman Completeness within the infinite-dimensional distribution space is not appropriate, as it either imposes strong constraints on the MDP structure or leads to linear regret. To circumvent model misspecification, we revisit the distributional BC through the statistical functional lens. We propose a novel framework that matches a finite number of statistical functionals to the target, rather than the entire distribution itself.

## 4. SF-LSVI : Statistical Functional Least Square Value Iteration

In this section, we propose SF-LSVI for distRL framework with general value function approximation. Leveraging the result from Theorem 3.6, we introduce a moment least square regression . This allows us to capture a finite set of moment information from the distribution, which can be unbiasedly estimated, thereby leading to the truncated moment problem (Shohat &amp; Tamarkin, 1943; Schmüdgen et al., 2017). Unlike previous work (Wang et al., 2023; Chen et al., 2024) that estimates in infinite-dimensional distribution spaces, our method enables to estimate distribution unbiasedly in finite-dimensional embedding spaces without misspecification error. The pseudocode of SF-LSVI is described in Appendix B.

Overview. At the beginning of episode k ∈ [ K ] , we maintain all previous samples { ( s τ h ′ , a τ h ′ , r τ h ′ ) } ( τ,h ′ ) ∈ [ k -1] × [ H ] and initialize a sketch ψ 1: N (¯ η k H +1 ( · )) = 0 N . For each step h = H,.. . , 1 , we compute the normalized sample moments of target distribution { ( B r τ h ′ ) # ¯ η k h +1 ( s τ h ′ +1 ) } h ′ ∈ [ H ] with the help of binomial theorem,

<!-- formula-not-decoded -->

and iteratively solve the N -moment least square regression

<!-- formula-not-decoded -->

based on the dataset D k h which contains the sketch of temporal target ψ 1: N ( ( B r τ h ′ ) # ¯ η k h +1 ( s τ h ′ +1 ) ) . Then we define Q k h ( · , · ) = min { ( ˜ f k h, ¯ η ) (1) ( · , · )+ b k h ( · , · ) , H } and choose the greedy policy π k h ( · ) with respect to Q k h . Next, we update all N normalized moments of Q -distribution ψ 1: N ( η h k ( · , · ) ) and V -distribution ψ 1: N ( ¯ η h k ( · ) ) . We repeat the procedure until all the K episodes are completed.

## 5. Theoretical Analysis

In this section, we provide the theoretical guarantees for SF-LSVI under Assumption 3.7. Applying proof techniques from Wang et al. (2020) and extending the result to a statistical functional lens, we generalize eluder dimension (Russo &amp; Van Roy, 2013) to the vector-valued function, which has been widely used in RL literatures (Ayoub et al., 2020; Wang et al., 2020; Jin et al., 2020) to measure the complexity of learning with the function approximators.

Definition 5.1 ( ϵ -dependent, ϵ -independent, Eluder dimension for vector-valued function) . Let ϵ ≥ 0 and Z = { ( s i , a i ) } n i =1 ⊆ S × A be a sequence of state-action pairs.

- A state-action pair ( s, a ) ∈ S × A is ϵ -dependent on Z with respect to F N if ∥ f -g ∥ Z ≤ ϵ for any vector-valued function f, g ∈ F N , then | f (1) ( s, a ) -g (1) ( s, a ) | ≤ ϵ .
- An ( s, a ) is ϵ -independent on Z with respect to F N if ( s, a ) is not ϵ -dependent on Z .
- The ϵ -eluder dimension dim E ( F N , ϵ ) of a vectorvalued function class F N is the length of the longest sequence of elements in S × A such that, for some ϵ ′ ≥ ϵ , every element is ϵ ′ -independent on its predecessors.

Weassume that the function class F N and state-action space S × A have bounded covering numbers.

Assumption 5.2 (Covering number) . For any ϵ &gt; 0 , the following holds:

- there exists an ϵ -cover C ( F N , ϵ ) ⊆ F N with size |C ( F N , ϵ ) | ≤ N ( F N , ϵ ) , such that for any g ∈ F N , there exists g ′ ∈ C ( F N , ϵ ) with ∥ g -g ′ ∥ ∞ , 1 ≤ ϵ .
- there exists an ϵ -cover C ( S × A , ϵ ) with size |C ( S × A , ϵ ) | ≤ N ( S × A , ϵ ) , such that for any ( s, a ) ∈ S × A , there exists ( s ′ , a ′ ) ∈ C ( S × A , ϵ ) with max f ∈F | f ( s, a ) -f ( s ′ , a ′ ) | ≤ ϵ

The following two lemmas give confidence bounds on the sum of the l 2 norms of all normalized moments.

Lemma 5.3 (Single Step Optimization Error) . Consider a fixed k ∈ [ K ] and a fixed h ∈ [ H ] . Let Z k h = { ( s τ h , a τ h ) } τ ∈ [ k -1] and D k h, ¯ η = {( s τ h , a τ h , ψ 1: N ( ( B τ r h ′ ) # ¯ η ( s τ h ′ +1 ) ))} τ ∈ [ k -1] for any ¯ η : S → P ([0 , H ]) . Define ˜ f k h, ¯ η = arg min f ∈F N ∥ f ∥ 2 D k h, ¯ η . For any ¯ η and δ ∈ (0 , 1) , there is an event E (¯ η, δ ) such that conditioned on E (¯ η, δ ) , with probability at least 1 -δ , for any ¯ η ′ : S → P ([0 , H ]) with ∥ ψ 1: N (¯ η ′ ) -ψ 1: N (¯ η ) ∥ ∞ , 1 ≤ 1 /T ,we have

<!-- formula-not-decoded -->

for some constant c ′ &gt; 0 .

Due to the definition of Bellman unbiasedness, we remark that moment sketch has a corresponding vector-valued estimator ϕ ψ 1: N as an identity and leads to a concentration results as the sampled sketches forms a martingale with respect to the filtration F τ h induced by the history

<!-- formula-not-decoded -->

Another notable aspect in Lemma 5.3 is using normalized moments E [ Z n ] /H n -1 instead of moments E [ Z n ] , as it reduces the size of the confidence region from O ( H N ) to O ( √ N ) . This adjustment is akin to scaling the optimization function in multi-objective optimization to treat each objective equally, which effectively prevents the model from favoring objectives with larger scales.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some constant c ′ &gt; 0 . Then with probability at least 1 -δ/ 2 , for all k, h ∈ [ K ] × [ H ] , we have

<!-- formula-not-decoded -->

Lemma 5.4 guarantees that the sequence of moments from the target distribution ψ 1: N ( ( B r h ( · , · ) ) # [ P h ¯ η k h +1 ]( · , · ) ) lies in the confidence region ( F N ) k h with high probability. Supported by the aforementioned lemma, we can further guarantee that all Q -functions are optimistically estimated with high probability and derive our final result.

Theorem 5.5. Under Assumption 3.7, with probability at least 1 -δ , SF-LSVI achieves a regret bound of

<!-- formula-not-decoded -->

Under weaker structural assumptions, we show that SF-LSVI enjoys near-optimal regret bound of order ˜ O ( d E H 3 2 √ K ) , which is √ H better than the state-of-theart distRL algorithm V-EST-LSR (Chen et al., 2024). For the linear MDP setting, we have d E = ˜ O ( d ) and thus SF-LSVI achieves a tight regret bound as ˜ O ( √ d 2 H 3 K ) which matches a lower bound Ω( √ d 2 H 3 K ) (Zhou et al., 2021). In our analysis, we highlight two main technical novelties which significantly reduces the degree of regret in distRL framework;

1. We refine previous lemma of Osband et al. (2019) and Wang et al. (2020) to remove the dependency of β ( F N , 1 /δ ) (See Appendix D.4), ensuring that regret bound depends only on the pre-defined function class, not on the number of moment extracted.

2. As shown in Table 1, we define the eluder dimension d E in a finite-dimensional embedding space F N , while other methods rely on an infinite-dimensional distribution space H ⊆ F ∞ .

## 6. Conclusions

We describe the sources of approximation error inherent in distribution-based updates and introduce a pivotal concept of Bellman unbiasedness, which enables to exactly learn the information of distribution. We also present a provably efficient online distRL algorithm, SF-LSVI , with general value function approximation. Notably, our algorithm achieves a near-optimal regret bound of ˜ O ( d E H 3 2 √ K ) , matching the tightest upper bound achieved by non-distributional framework (Zhou et al., 2021; He et al., 2023). One interesting future direction would be to reformulate the definition of regret as discrepencies in moments rather than the expected return, and to show the sampleefficiency of distRL. We hope that our work sheds some light on future research in analyzing the provable efficiency of distRL.

## Acknowledgements

This work is in part supported by the National Research Foundation of Korea (NRF, RS-2024-00451435(40%), RS2024-00413957(40%)), Institute of Information &amp; communications Technology Planning &amp; Evaluation (IITP, RS2021-II212068(10%), 2021-0-00180(10%)) grant funded by the Ministry of Science and ICT (MSIT), grant-in-aid of HANHWA SYSTEMS, Institute of New Media and Communications(INMAC), and the BK21 FOUR program of the Education and Research Program for Future ICT Pioneers, Seoul National University in 2025.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## References

- Abbasi-Yadkori, Y., Pál, D., and Szepesvári, C. Improved algorithms for linear stochastic bandits. Advances in neural information processing systems , 24, 2011.
- Agarwal, A., Jin, Y., and Zhang, T. Vo q l: Towards optimal regret in model-free rl with nonlinear function approximation. In The Thirty Sixth Annual Conference on Learning Theory , pp. 987-1063. PMLR, 2023.

Auer, P., Jaksch, T., and Ortner, R. Near-optimal regret

- bounds for reinforcement learning. Advances in neural information processing systems , 21, 2008.
- Ayoub, A., Jia, Z., Szepesvari, C., Wang, M., and Yang, L. Model-based reinforcement learning with value-targeted regression. In International Conference on Machine Learning , pp. 463-474. PMLR, 2020.
- Bastani, O., Ma, J. Y., Shen, E., and Xu, W. Regret bounds for risk-sensitive reinforcement learning. Advances in Neural Information Processing Systems , 35: 36259-36269, 2022.
- Bellemare, M. G., Naddaf, Y., Veness, J., and Bowling, M. The arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research , 47:253-279, 2013.
- Bellemare, M. G., Dabney, W., and Munos, R. A distributional perspective on reinforcement learning. In International Conference on Machine Learning , pp. 449-458. PMLR, 2017.
- Bellemare, M. G., Candido, S., Castro, P. S., Gong, J., Machado, M. C., Moitra, S., Ponda, S. S., and Wang, Z. Autonomous navigation of stratospheric balloons using reinforcement learning. Nature , 588(7836):77-82, 2020.
- Bellemare, M. G., Dabney, W., and Rowland, M. Distributional reinforcement learning . MIT Press, 2023.
- Bodnar, C., Li, A., Hausman, K., Pastor, P., and Kalakrishnan, M. Quantile qt-opt for risk-aware vision-based robotic grasping. arXiv preprint arXiv:1910.02787 , 2019.
- Chen, Y., Zhang, X., Wang, S., and Huang, L. Provable risk-sensitive distributional reinforcement learning with general function approximation. arXiv preprint arXiv:2402.18159 , 2024.
- Cho, T., Han, S., Lee, H., Lee, K., and Lee, J. Pitfall of optimism: Distributional reinforcement learning by randomizing risk criterion. arXiv preprint arXiv:2310.16546 , 2023.
- Choi, Y., Lee, K., and Oh, S. Distributional deep reinforcement learning with a mixture of gaussians. In 2019 International conference on robotics and automation (ICRA) , pp. 9791-9797. IEEE, 2019.
- Chow, Y., Tamar, A., Mannor, S., and Pavone, M. Risksensitive and robust decision-making: a cvar optimization approach. Advances in neural information processing systems , 28, 2015.
- Dabney, W., Ostrovski, G., Silver, D., and Munos, R. Implicit quantile networks for distributional reinforcement learning. In International conference on machine learning , pp. 1096-1105. PMLR, 2018a.

- Dabney, W., Rowland, M., Bellemare, M., and Munos, R. Distributional reinforcement learning with quantile regression. In Proceedings of the AAAI Conference on Artificial Intelligence , 2018b.
- Dann, C., Jiang, N., Krishnamurthy, A., Agarwal, A., Langford, J., and Schapire, R. E. On oracle-efficient pac rl with rich observations. Advances in neural information processing systems , 31, 2018.
- Engert, M. Finite dimensional translation invariant subspaces. Pacific Journal of Mathematics , 32(2):333-343, 1970.
- Fawzi, A., Balog, M., Huang, A., Hubert, T., RomeraParedes, B., Barekatain, M., Novikov, A., R Ruiz, F. J., Schrittwieser, J., Swirszcz, G., et al. Discovering faster matrix multiplication algorithms with reinforcement learning. Nature , 610(7930):47-53, 2022.
- Fei, Y., Yang, Z., and Wang, Z. Risk-sensitive reinforcement learning with function approximation: A debiasing approach. In International Conference on Machine Learning , pp. 3198-3207. PMLR, 2021.
- Greenberg, I., Chow, Y., Ghavamzadeh, M., and Mannor, S. Efficient risk-averse reinforcement learning. Advances in Neural Information Processing Systems , 35:3263932652, 2022.
- Halmos, P. R. The theory of unbiased estimation. The Annals of Mathematical Statistics , 17(1):34-43, 1946.
- He, J., Zhao, H., Zhou, D., and Gu, Q. Nearly minimax optimal reinforcement learning for linear markov decision processes. In International Conference on Machine Learning , pp. 12790-12822. PMLR, 2023.
- Ishfaq, H., Cui, Q., Nguyen, V., Ayoub, A., Yang, Z., Wang, Z., Precup, D., and Yang, L. Randomized exploration in reinforcement learning with general value function approximation. In International Conference on Machine Learning , pp. 4607-4616. PMLR, 2021.
- Jin, C., Allen-Zhu, Z., Bubeck, S., and Jordan, M. I. Is q-learning provably efficient? Advances in neural information processing systems , 31, 2018.
- Jin, C., Yang, Z., Wang, Z., and Jordan, M. I. Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory , pp. 2137-2143. PMLR, 2020.
- Jin, C., Liu, Q., and Miryoosefi, S. Bellman eluder dimension: New rich classes of rl problems, and sampleefficient algorithms. Advances in neural information processing systems , 34:13406-13418, 2021.
- Kakade, S. M. On the sample complexity of reinforcement learning . University of London, University College London (United Kingdom), 2003.
- Kim, D., Lee, K., and Oh, S. Trust region-based safe distributional reinforcement learning for multiple constraints. In Thirty-seventh Conference on Neural Information Processing Systems , 2023.
- Kim, D., Lee, K., and Oh, S. Trust region-based safe distributional reinforcement learning for multiple constraints. Advances in neural information processing systems , 36, 2024.
- Lattimore, T. and Szepesvári, C. Bandit algorithms . Cambridge University Press, 2020.
- Liang, H. and Luo, Z.-Q. Bridging distributional and risksensitive reinforcement learning with provable regret bounds. arXiv preprint arXiv:2210.14051 , 2022.
- Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M., and Bowling, M. Revisiting the arcade learning environment: Evaluation protocols and open problems for general agents. Journal of Artificial Intelligence Research , 61:523-562, 2018.
- Marthe, A., Garivier, A., and Vernade, C. Beyond average return in markov decision processes. arXiv preprint arXiv:2310.20266 , 2023.
- Muller, T. H., Butler, J. L., Veselic, S., Miranda, B., Wallis, J. D., Dayan, P., Behrens, T. E., Kurth-Nelson, Z., and Kennerley, S. W. Distributional reinforcement learning in prefrontal cortex. Nature Neuroscience , pp. 1-6, 2024.
- Osband, I. and Van Roy, B. On lower bounds for regret in reinforcement learning. arXiv preprint arXiv:1608.02732 , 2016.
- Osband, I., Van Roy, B., Russo, D. J., Wen, Z., et al. Deep exploration via randomized value functions. J. Mach. Learn. Res. , 20(124):1-62, 2019.
- Rowland, M., Bellemare, M., Dabney, W., Munos, R., and Teh, Y. W. An analysis of categorical distributional reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pp. 29-37. PMLR, 2018.
- Rowland, M., Dadashi, R., Kumar, S., Munos, R., Bellemare, M. G., and Dabney, W. Statistics and samples in distributional reinforcement learning. In International Conference on Machine Learning , pp. 5528-5536. PMLR, 2019.
- Rowland, M., Munos, R., Azar, M. G., Tang, Y., Ostrovski, G., Harutyunyan, A., Tuyls, K., Bellemare, M. G., and

- Dabney, W. An analysis of quantile temporal-difference learning. arXiv preprint arXiv:2301.04462 , 2023.
- Russo, D. and Van Roy, B. Eluder dimension and the sample complexity of optimistic exploration. Advances in Neural Information Processing Systems , 26, 2013.
- Schmüdgen, K. et al. The moment problem , volume 9. Springer, 2017.
- Shohat, J. A. and Tamarkin, J. D. The problem of moments . Number 1. American Mathematical Soc., 1943.
- Son, K., Kim, J., Yi, Y., and Shin, J. Disentangling sources of risk for distributional multi-agent reinforcement learning. 2021.
- Wang, K., Zhou, K., Wu, R., Kallus, N., and Sun, W. The benefits of being distributional: Small-loss bounds for reinforcement learning. arXiv preprint arXiv:2305.15703 , 2023.
- Wang, R., Salakhutdinov, R. R., and Yang, L. Reinforcement learning with general value function approximation: Provably efficient approach via bounded eluder dimension. Advances in Neural Information Processing Systems , 33:6123-6135, 2020.
- Wang, Y., Wang, R., Du, S. S., and Krishnamurthy, A. Optimism in reinforcement learning with generalized linear function approximation. arXiv preprint arXiv:1912.04136 , 2019.
- Zanette, A., Lazaric, A., Kochenderfer, M., and Brunskill, E. Learning near optimal policies with low inherent bellman error. In International Conference on Machine Learning , pp. 10978-10989. PMLR, 2020.
- Zhou, D., Gu, Q., and Szepesvari, C. Nearly minimax optimal reinforcement learning for linear mixture markov decision processes. In Conference on Learning Theory , pp. 4532-4576. PMLR, 2021.

## Appendix

## A. Notation

Table 2. Table of notation

| Notation               | Description                                                                                                         |
|------------------------|---------------------------------------------------------------------------------------------------------------------|
| S                      | state space of size S                                                                                               |
| A                      | action space of size A                                                                                              |
| H                      | horizon length of one episode                                                                                       |
| T                      | number of episodes                                                                                                  |
| r h ( s,a )            | reward of ( s,a ) at step h                                                                                         |
| P h ( s ′ &#124; s,a ) | probability transition of ( s,a ) to s ′ at step h                                                                  |
| H k h                  | history up to step h , episode k                                                                                    |
| N                      | number of statistical functionals                                                                                   |
| Q π h ( s,a )          | Q-function of a given policy π at step h                                                                            |
| V π h ( s )            | V-function of a given policy π at step h                                                                            |
| Z π h ( s,a )          | random variable of Q -function                                                                                      |
| ¯ Z π h ( s )          | random variable of V -function                                                                                      |
| η π h ( s,a )          | probability distribution of Q -function                                                                             |
| ¯ η π h ( s )          | probability distribution of V -function                                                                             |
| [ P h ( · )]           | expectation over transition [ P h ( · )] = E s ′ ∼ P h ( · )                                                        |
| ( B r ) #              | pushforward of the distribution through B r ( x ) := r + x                                                          |
| f ( n )                | n -th element of N -dimensional vector f                                                                            |
| ∥ f ∥ ∞                | max norm of f : X → R defined as ∥ f ∥ ∞ := max x ∈ X &#124; f ( n ) ( x ) &#124;                                   |
| ∥ f ∥ ∞ , 1            | l 1 -norm of max norm of f : X → R defined as ∥ f ∥ ∞ , 1 := ∑ N n =1 max x ∈ X &#124; f ( n ) ( x ) &#124;         |
| F N                    | a function class of N -dimensional embedding space                                                                  |
| Z                      | a set of state-action pairs Z := { ( s t ,a t ) } &#124;Z&#124; t =1                                                |
| D                      | a dataset D := { ( s t ,a t , [ d (1) t , · · · ,d ( N ) t ]) } &#124;D&#124; t =1                                  |
| ∥ f ∥ 2 Z              | for f : S×A→ R , define ∥ f ∥ 2 Z := ∑ N n =1 ∑ ( s,a ) ∈Z ( f ( n ) ( s t ,a t )) 2                                |
| ∥ f ∥ 2 D              | for f : S×A→ R , define ∥ f ∥ 2 D := ∑ N n =1 ∑ D t =1 ( f ( n ) ( s t ,a t ) - d ( n ) t ) 2                       |
| w ( n ) ( F N , s,a    | width function of ( s,a ) defined as w ( n ) ( F N , s,a ) := max f,g ∈F N &#124; f ( n ) ( s,a ) - g ( n ) ( s,a   |
| ˜ f k h, ¯ η           | a solution of moment least squre regression, defined as ˜ f k h, ¯ η := argmin f ∈F N ∥ f ∥ D k h                   |
| f ¯ η                  | a target sketch of distribution ¯ η , defined as f ¯ η := ψ 1: N (( B r ) # [ P h ¯ η ])                            |
| ( F N ) k h            | a confidence region at step h , episode k , defined as ( F N ) k h := { f ∈ F N &#124; ∥ f - ˜ f k h, ¯ η ∥ 2 Z k ≤ |
| ψ (¯ η )               | a statistical functional P ψ ( R ) S → R S                                                                          |
| ψ 1: N (¯ η )          | a N - collection of statistical functional P ψ 1: N ( R ) S → R N × S                                               |
| P ψ 1: N ( R )         | a domain of sketch ψ 1: N                                                                                           |
| I ψ 1: N               | an image of sketch ψ 1: N                                                                                           |
| T                      | distributional Bellman operator, defined as T ¯ η := ( B r ) # [ P ¯ η ]                                            |
| T ψ                    | sketch Bellman operator w.r.t ψ , defined as T ψ ψ (¯ η ) := ψ ( ( B r ) # [ P ¯ η ] )                              |
| ˆ T ψ                  | empirical sketch Bellman operator w.r.t ψ , defined as ˆ T ψ ψ (¯ η ) := ψ ( ( B r ) # [ ˆ P ¯ η ] )                |
| N ( F N , ϵ )          | covering number of F N w.r.t the ϵ - ball                                                                           |
| dim E ( F N , ϵ )      | eluder dimension of F N w.r.t ϵ                                                                                     |

## B. Pseudocode of SF-LSVI and Technical Remarks

Algorithm 1 Statistical Functional Least Square Value Iteration ( SF-LSVI ( δ ) )

```
Input: failure probability δ ∈ (0 , 1) and the number of episodes K 1: for episode k = 1 , 2 , . . . , K do 2: Receive initial state s k 1 3: Initialize ψ 1: N (¯ η k H +1 ( · )) ← 0 N 4: for step h = H,H -1 , . . . , 1 do 5: D k h ← { s τ h ′ , a τ h ′ , ψ 1: N ( ( B r τ h ′ ) # ¯ η k h +1 ( s τ h ′ +1 ) )} ( τ,h ′ ) ∈ [ k -1] × [ H ] // Data collection 6: ˜ f k h, ¯ η ← arg min f ∈F N ∥ f ∥ D k h // Distribution Estimation 7: b k h ( · , · ) ← w (1) (( F N ) k h , · , · ) 8: Q k h ( · , · ) ← min { ( ˜ f k h, ¯ η ) (1) ( · , · ) + b k h ( · , · ) , H } 9: π k h ( · ) = arg max a ∈A Q k h ( · , a ) , V k h ( · ) = Q k h ( · , π k h ( · )) // Optimistic planning 10: ψ 1 ( η k h ( · , · ) ) ← Q k h ( · , · ) , ψ 2: N ( η k h ( · , · ) ) ← ( min { ( ˜ f k h, ¯ η ) ( n ) ( · , · ) , H } ) n ∈ [2: N ] 11: ψ 1 ( ¯ η k h ( · ) ) ← V k h ( · ) , ψ 2: N ( ¯ η k h ( · ) ) ← ψ 1: N ( η k h ( · , π k h ( · )) ) n ∈ [2: N ] 12: for h = 1 , 2 , . . . , H do 13: Take action a k h ← π k h ( s k h ) 14: Observe reward r k h ( s k h , a k h ) and get next state s k h +1 .
```

Remark B.1 . For an optimistic planning, we define the bonus function as the width function b k h ( s, a ) := w k h (( F N ) k h , s, a ) where ( F N ) k h denotes a confidence region at step h , episode k . When F is a linear function class, the width function can be evaluated by simply computing the maximal distance of weight vector. For a general function class F , computing the width function requires to solve a set-constrained optimization problem, which is known as NP-hard (Dann et al., 2018). However, a width function is computed simply for optimistic exploration, and approximation errors are known to have a small effect on regret (Abbasi-Yadkori et al., 2011).

## C. Related Work and Discussion

## C.1. Technical Clarifications on Linearity Assumption in Existing Results

Bellman Closedness and Linearity. Rowland et al. (2019) proved that quantile functional is not Bellman closed by providing a specific counterexample. However, their discussion based on counterexamples can be generalized as it assumes that the sketch Bellman operator for the quantile functional needs to be linear.

̸

They consider an discounted MDP with initial state s 0 with single action a , which transits to one of two terminal states s 1 , s 2 with equal probability. Letting no reward at state s 0 , Unif ([0 , 1]) at state s 1 , and Unif ([1 /K, 1 + 1 /K ]) at state s 2 , the return distribution at state s 0 is computed as mixture 1 2 Unif ([0 , γ ]) + 1 2 Unif ([ γ/K,γ + γ/K ]) . Then the 1 2 K -quantile at state s 0 is γ K . They proposed a counterexample where each quantile distribution of state s 1 , s 2 is represented as 1 K ∑ K k =1 δ 2 k -1 K and 1 K ∑ K k =1 δ 2 k +1 K respectively, the 1 2 K -quantile of state s 0 is ψ q 2 K ( 1 2 K ∑ K k =1 δ γ (2 k -1) K + δ γ (2 k +1) K ) = 3 γ 2 K . However, this example does not consider that the mixture of quantiles is not a quantile of the mixture distribution (i.e., ψ q ( λη 1 +(1 -λ ) η 2 ) = λψ q ( η 1 ) + (1 -λ ) ψ q ( η 2 ) ), due to the nonlinearity of the quantile functional. Therefore, this does not present a valid counterexample to prove that quantile functionals are not Bellman closed.

Bellman Optimizability and Linearity. Marthe et al. (2023) proposed the notion of Bellman optimizable statistical functional which redefine the Bellman update by planning with respect to statistical functionals rather than expected returns. They proved that W 1 -continuous Bellman Optimizable statistical functionals are characterized by exponential utilities 1 λ log E Z ∼ η [exp( λZ )] . However, their proof requires some technical clarification regarding the assumption that such statistical functionals are linear.

To illustrate, they define a statistical functional ψ f and consider two probability distributions η 1 = 1 2 ( δ 0 + δ h ) and η 2 = δ ϕ ( h ) where ϕ ( h ) = f -1 ( 1 2 ( f (0) + f ( h )) ) . Using the translation property, they lead ψ f ( η 1 ) = ψ f ( η 2 ) to 1 2 ( f ( x ) + f ( x + h )) = f ( x + ϕ ( h )) for all x ∈ R . However, this equality ψ f ( 1 2 ( δ x + δ x + h ) ) = 1 2 ( f ( x ) + f ( x + h )) holds only if ψ f is linear, which is not necessarily a valid assumption for all statistical functionals.

## C.2. Existence of Nonlinear Bellman Closed Sketch.

The previous two examples may not have considered the possibility that the sketch Bellman operator might not necessarily be linear. However, some statistical functionals are Bellman-closed even if they are nonlinear, so it is open question whether there is a nonlinear sketch Bellman operator that makes the quantile functional Bellman-closed. In this section, we present examples of maximum and minimum functionals that are Bellman-closed, despite being nonlinear.

In a nutshell, consider the maximum of return distribution at state s 1 , s 2 is γ, γ + γ/K respectively. Beyond linearity, the maximum of return distribution at state s 0 can be computed by taking the maximum of these values;

<!-- formula-not-decoded -->

which produces the desired result. This implies the existence of a nonlinear sketch that is Bellman closed. More precisely, by defining max s ′ ∼ P ( ·| s,a ) and min s ′ ∼ P ( ·| s,a ) as the maximum and minimum of the sampled sketch ψ ( ( B r ) # ¯ η ( s ′ ) ) with the distribution P ( ·| s, a ) , we can derive the sketch Bellman operator for maximum and minimum functionals as follows;

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C.3. Non-existence of sketch Bellman operator for quantile functional

In this section, we prove that quantile functional cannot be Bellman closed under any additional sketch. First we introduce the definition of mixture-consistent , which is the property that the sketch of a mixture can be computed using only the sketch of the distribution of each component.

Definition C.1 (mixture-consistent) . A sketch ψ is mixture-consistent if for any ν ∈ [0 , 1] and any distributions η 1 , η 2 ∈ P ψ ( R ) , there exists a corresponding function h ψ such that

<!-- formula-not-decoded -->

Next, we will provide some examples of determining whether a sketch is mixture-consistent or not.

Example 1. Every moment or exponential polynomial functional is mixture-consistent.

Proof. For any n ∈ [ N ] and λ ∈ C ,

<!-- formula-not-decoded -->

Example 2. Variance functional is not mixture-consistent.

Proof. Let ν = 1 2 and Z, Y be the random variables where Z ∼ 1 2 δ 0 + 1 2 δ 2 and Y ∼ 1 2 δ k + 1 2 δ k +2 . Then, Var ( Z ) = Var ( Y ) = 1 . While RHS is constant for any k , LHS is not a constant for any k , i.e.,

<!-- formula-not-decoded -->

■

■

While variance functional is not mixture consistent by itself, it can be mixture consistent with another statistical functional, the mean.

Example 3. Variance functional is mixture-consistent under mean functional.

Proof. Notice that mean functional is mixture-consistent. We need to show that variance functional is mixture-consistent under mean functional.

<!-- formula-not-decoded -->

■

This means that to determine whether it is mixture-consistent or not, we should check it on a per-sketch basis, rather than on a per-statistical functional basis.

Example 4. Maximum and minimum functional are both mixture-consistent.

Proof.

and

which is dependent in k .

Lemma C.2. Quantile sketch cannot be mixture-consistent, under any additional sketch.

Proof. For a given integer N &gt; 0 and a quantile level α ∈ (0 , 1) , let ν = 1 2 and a random variable Y ∼ p y 0 δ 0 + p y 1 δ y 1 + · · · + p y N δ y N (0 &lt; y 1 &lt; · · · &lt; y N &lt; 1) where p y 0 &gt; α so that ψ α -quantile [ Y ] = 0 . Consider another random variable Z ∼ p z 0 δ 0 + p z 1 δ 1 where p z 0 &lt; α so that ψ α -quantile [ Z ] = 1 . Then the α -quantile of the mixture X = Y + Z 2 is

<!-- formula-not-decoded -->

Letting p z 0 = 2 α -∑ n n ′ =0 p y n ′ , we can manipulate ψ α -quantile [ X ] to be any value of y n . Hence, ψ α -quantile [ X ] is a function of all possible outcomes of Y .

If there exists a finite number of statistical functionals which make quantile sketch mixture-consistent, then such sketch would uniquely determine the distribution for any N . This results in a contradiction that infinite-dimensional distribution space can be represented by a finite number of statistical functional. ■

Lemma C.3. If a sketch ψ is Bellman closed, then it is mixture-consistent.

Proof. Consider an MDP where initial state s 0 has no reward and transits to two state s 1 , s 2 with probability ν, 1 -ν and reward distribution ¯ η 1 , ¯ η 2 . Since ψ is Bellman closed, ψ (¯ η ( s 0 )) is a function of ψ (¯ η ( s 1 )) and ψ (¯ η ( s 2 )) , (i.e., ψ (¯ η ( s 0 )) = g ψ ( ψ (¯ η ( s 1 )) , ψ (¯ η ( s 2 ))) for some g ψ ). Since ψ (¯ η ( s 0 )) = ψ ( ν ¯ η ( s 1 ) + (1 -ν )¯ η ( s 2 )) , it implies that ψ is mixtureconsistent. ■

Combining the results of Lemma C.2 and Lemma C.3, we prove that a quantile sketch cannot be Bellman closed, no matter what additional sketches are provided.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since maximum and minimum functionals are mixture consistent, we can construct a nonlinear sketch bellman operator like the one in section C.2. This is possible because there is a nonlinear function h ψ that ensures the sketch is closed under mixture.

Before demonstrating that a quantile sketch cannot be mixture consistent under any additional sketch, we will first illustrate with the example of a median functional that is not mixture consistent.

Example 5. Median sketch is not mixture-consistent.

Proof. Let ν = 1 2 and Z, Y be the random variables where Z ∼ 0 . 2 δ 0 +0 . 8 δ 1 and Y ∼ 0 . 6 δ 0 +0 . 4 δ k for some 0 &lt; k &lt; 1 . Then ψ med ( Z ) = 1 and ψ med ( Y ) = 0 . However,

<!-- formula-not-decoded -->

■

## D. Proof

Theorem (3.3) . Quantile functional cannot be Bellman closed under any additional sketch.

Proof. See Lemma C.2 and Lemma C.3.

■

Lemma (3.5) . Let F ¯ η be a CDF of the probability distribution ¯ η ∈ P ( R ) S . Then a sketch is Bellman unbiased if and only if the sketch is a homogeneous of degree k , i.e., there exists some vector-valued function h = h ( x 1 , · · · , x k ) : X k → R N such that

<!-- formula-not-decoded -->

Proof. ( ⇒ ) Consider an two-stage MDP with a single action a , and an initial state s 0 which transits to one of terminal state { s 1 , · · · , s K } with transition kernel P ( ·| s 0 , a ) . Assume that the reward r ( s 0 ) = 0 . Then ¯ η ( s 0 ) = ∑ K k =1 P ( s k ) δ r ( s k ) . Note that s ′ 1 , · · · , s ′ k are independent and identically distributed random variable in distribution P ( ·| s, a ) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

■

Theorem (3.6) . The only finite statistical functionals that are Bellman unbiased and closed are given by the collections of ψ 1 , . . . , ψ N where its linear span { ∑ N n =0 α n ψ n | α n ∈ R , ∀ N } is equal to the set of exponential polynomial functionals { η → E Z ∼ η [ Z l exp( λZ )] | l = 0 , 1 , . . . , L, λ ∈ R } , where ψ 0 is the constant functional equal to 1 . In discount setting, it is equal to the linear span of the set of moment functionals { η → E Z ∼ η [ Z l ] | l = 0 , 1 , . . . , L } for some L ≤ N .

Proof. Our proof is mainly based on the proof techniques of Rowland et al. (2019) and we describe in an extended form. Since their proof also considers the discounted setting, we will define B r,γ ( x ) = r + γx for discount factor γ ∈ [0 , 1) . By assumption of Bellman closedness, ψ n ( ( B r,γ ) # ¯ η ( s ′ ) ) will be written as g ( r, γ, ψ 1: N (¯ η ( s ′ )) for some g . By assumption of Bellman unbiasedness and Lemma 3.5, both ψ 1: N (¯ η ( s ′ )) and ψ n ( ( B r,γ ) # ¯ η ( s ′ ) ) are affine as functions of the distribution

¯ η ( s ′ ) ,

and

<!-- formula-not-decoded -->

Therefore, g ( r, γ, · ) is also affine on the convex codomain of ψ 1: N . Thus, we have

<!-- formula-not-decoded -->

for some function a 0: N : R × [0 , 1] → R . By taking ¯ η ( s ′ ) = δ x , we obtain

<!-- formula-not-decoded -->

According to Engert (1970), for any translation invariant finite-dimensional space is spanned by a set of function of the form

<!-- formula-not-decoded -->

for some finite subset { λ 1 , · · · , λ J } of C . Hence, each function x ↦→ ϕ ψ n ( x, · · · , x ) is expressed as linear combination of exponential polynomial functions. In addition, the linear combination of ϕ ψ n should be closed under composition with for any discount factor γ ∈ [0 , 1] , all λ j should be zero. Hence, the linear combination of ϕ ψ 1 , · · · , ϕ ψ N must be equal to the span of { x ↦→ x l | 0 ≤ l ≤ L } for some L ∈ N .

■

Lemma (5.3) . Consider a fixed k ∈ [ K ] and a fixed h ∈ [ H ] . Let Z k h = { ( s τ h , a τ h ) } τ ∈ [ k -1] and D k h, ¯ η = {( s τ h , a τ h , ψ 1: N ( ( B τ r h ′ ) # ¯ η ( s τ h ′ +1 ) ))} τ ∈ [ k -1] for any ¯ η : S → P ([0 , H ]) . Define ˜ f k h, ¯ η = arg min f ∈F N ∥ f ∥ 2 D k h, ¯ η . For any ¯ η and δ ∈ (0 , 1) , there is an event E (¯ η, δ ) such that conditioned on E (¯ η, δ ) , with probability at least 1 -δ , for any ¯ η ′ : S → P ([0 , H ]) with ∥ ψ 1: N (¯ η ′ ) -ψ 1: N (¯ η ) ∥ ∞ , 1 ≤ 1 /T or ∑ N n =1 ∥ ψ n (¯ η ′ ) -ψ n (¯ η ) ∥ ∞ ≤ 1 /T , we have

<!-- formula-not-decoded -->

for some constant c ′ &gt; 0 .

Proof. Define the sketch of target f ¯ η : S × A → R N ,

<!-- formula-not-decoded -->

for all i ∈ [ N ] .

<!-- formula-not-decoded -->

For any f ∈ F ,

<!-- formula-not-decoded -->

For the first inequality, we change the second term from ¯ η ′ to ¯ η which are the ϵ -covers. Notice that AC -BC ′ ≥ -| AC -BC ′ | ≥ -| ( A -B ) C | - | ( A -B ) C ′ | ≥ -2 | A -B || max( C, C ′ ) | .

<!-- formula-not-decoded -->

For the second inequality, consider ¯ η ′ : S → P ([0 , H ]) with ∑ N n =1 ∥ ψ n (¯ η ′ ) -ψ n (¯ η ) ∥ ∞ ≤ 1 /T . We have

<!-- formula-not-decoded -->

Defining F k h as the filtration induced by the sequence { ( s τ h ′ , a τ h ′ ) } τ,h ′ ∈ [ k -1] × [ H ] ∪{ ( s k 1 , a k 1 ) , ( s k 2 , a k 2 ) , . . . , ( s k h , a k h ) } , notice that and

<!-- formula-not-decoded -->

In third equality, we emphasize that only Bellman unbiased sketch can derive the martingale difference sequence which induce the concentration result. Since every moment functional is commutable with mixing operation, the transformation ϕ ψ n in Definition 3.4 is identity for all n ∈ [ N ] . Hence, we choose the sketch as moment which already knows ϕ ψ .

By Azuma-Hoeffding inequality,

<!-- formula-not-decoded -->

where the second inequality follows from the Cauchy-Schwartz inequality.

We set

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

With union bound for all f ∈ C ( F N , 1 /T ) , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

for some constant c ′ &gt; 0 .

For all f ∈ F N , there exists g ∈ C ( F N , 1 /T ) , such that ∥ f -g ∥ ∞ , 1 ≤ 1 /T or ∑ N n =1 ∥ f ( n ) -g ( n ) ∥ ∞ ≤ 1 /T for all n ∈ [ N ] ,

<!-- formula-not-decoded -->

where the third inequality follows from,

<!-- formula-not-decoded -->

Recall that ˜ f k h,η ′ = arg min f ∈F ∥ f ∥ 2 D k h,η ′ . We have ∥ ˜ f k h,η ′ ∥ 2 D k h,η ′ -∥ f ¯ η ′ ∥ 2 D k h,η ′ ≤ 0 , which implies,

<!-- formula-not-decoded -->

Recall that if x 2 -2 ax -b ≤ 0 holds for constant a, b &gt; 0 , then x ≤ a + √ a 2 + b ≤ c ′ · a for some constant c ′ &gt; 0 . Hence,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for some constant c ′ &gt; 0 .

Lemma (5.4) . Let ( F N ) k h = { f ∈ F N |∥ f -˜ f k h, ¯ η ∥ 2 Z k h ≤ β ( F N , δ ) } , where

<!-- formula-not-decoded -->

for some constant c ′ &gt; 0 . Then with probability at least 1 -δ/ 2 , for all k, h ∈ [ K ] × [ H ] , we have

<!-- formula-not-decoded -->

Proof. For all ( k, h ) ∈ [ K ] × [ H ] ,

<!-- formula-not-decoded -->

is a (1 /T ) -cover of ψ 1: N ( η k h +1 ( · , · )) where

<!-- formula-not-decoded -->

i.e., there exists ψ 1: N ( η ) ∈ S such that ∥ ψ 1: N ( η ) -ψ 1: N ( η k h +1 ) ∥ ∞ , 1 ≤ 1 /T . This implies

<!-- formula-not-decoded -->

is a (1 /T ) -cover of ψ 1: N (¯ η k h +1 ) with log( | ¯ S | ) ≤ log N ( F N , 1 /T ) .

For each ψ 1: N (¯ η ) ∈ ¯ S , let E (¯ η, δ/ 2 | ¯ S | T ) be the event defined in Lemma 5.3. By union bound for all ψ 1: N (¯ η ) ∈ ¯ S , we have Pr [ ⋂ ψ 1: N (¯ η ) ∈ ¯ S E (¯ η, δ/ 2 | ¯ S | T )] ≥ 1 -δ/ 2 T .

Let ψ 1: N (¯ η ) ∈ ¯ S such that ∥ ψ 1: N (¯ η ) -ψ 1: N (¯ η k h +1 ) ∥ ∞ , 1 ≤ 1 /T . Conditioned on ⋂ s N (¯ η ) ∈ ¯ S E (¯ η, δ/ 2 | ¯ S | T ) and by Lemma 5.3, we have

<!-- formula-not-decoded -->

for some constant c ′ &gt; 0 .

By union bound for all ( k, h ) ∈ [ K ] × [ H ] , we have ψ 1: N ( ( B r h ( · , · ) ) # [ P h ¯ η k h +1 ]( · , · ) ) ∈ ( F N ) k h with probability 1 -δ/ 2 . ■

Lemma D.1. Let Q k h ( s, a ) := min { H, ˜ f k h ( s, a ) + b k h ( s, a ) } for some bonus function b k h ( s, a ) for all ( s, a ) ∈ S × A . If b k h ( s, a ) ≥ w (1) (( F N ) k h , s, a ) , then with probability at least 1 -δ/ 2 ,

<!-- formula-not-decoded -->

for all ( k, h ) ∈ [ K ] × [ H ] , for all ( s, a ) ∈ S × A .

Proof. We use induction on h from h = H to 1 to prove the statement. Let E be the event that for ( k, h ) ∈ [ K ] × [ H ] , ψ 1: N ( ( B r h ( · , · ) ) # [ P h ¯ η k h +1 ]( · , · ) ) ∈ ( F N ) k h . By Lemma 5.4, Pr |E| ≥ 1 -δ/ 2 . In the rest of the proof, we condition on E .

When h = H +1 , the desired inequality holds as Q ∗ H +1 ( s, a ) = V ∗ H +1 ( s ) = Q k H +1 ( s, a ) = V k H +1 ( s ) = 0 . Now, assume that Q ∗ h +1 ( s, a ) ≤ Q k h +1 ( s, a ) and V ∗ h +1 ( s ) ≤ V k h +1 ( s ) for some h ∈ [ H ] . Then, for all ( s, a ) ∈ S × A ,

<!-- formula-not-decoded -->

Lemma D.2 (Regret decomposition) . With probability at least 1 -δ/ 4 , we have

<!-- formula-not-decoded -->

where ξ k h = [ P h ( V k h +1 -V π k h +1 )]( s k h , a k h ) -( V k h +1 ( s k h +1 ) -V π k h +1 ( s k h +1 )) is a martingale difference sequence with respect to the filtration F k h induced by the history H k h .

■

Proof. We condition on the above event E in the rest of the proof. For all ( k, h ) ∈ [ K ] × [ H ] , we have

<!-- formula-not-decoded -->

Recall that ( F N ) k h = { f ∈ F N | ∥ f -˜ f k h, ¯ η ∥ 2 Z k h ≤ β ( F N , δ ) } is the confidence region. Since ψ 1: N ( ( B r h ( · , · ) ) # [ P h ¯ η k h +1 ]( · , · ) ) ∈ ( F N ) k h , then by the definition of width function w (1) ( F k h , s, a ) , for ( k, h ) ∈ [ K ] × [ H ] , we have

<!-- formula-not-decoded -->

Recall that Q ∗ h ( · , · ) ≤ Q k h ( · , · ) .

<!-- formula-not-decoded -->

It remains to bound ∑ K k =1 ∑ H h =1 b k h ( s k h , a k h ) , for which we will exploit fact that F N has bounded eluder dimension.

Lemma D.3. If b k ( s, a ) ≥ w (1) (( F N ) k , s, a ) for all ( s, a ) ∈ S × A and k ∈ [ K ] where

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

then

<!-- formula-not-decoded -->

for some constant c &gt; 0 .

■

Proof. We first want to show that for any sequence { ( s 1 , a 1 ) , . . . , ( s κ , a κ ) } ⊆ S × A , there exists j ∈ [ κ ] such that ( s j , a j ) is ϵ -dependent on at least L = ⌈ ( κ -1) / dim E ( F N , ϵ ) ⌉ disjoint subsequences in { ( s 1 , a 1 ) , . . . , ( s j -1 , a j -1 ) } with respect to F N . We demonstrate this by using the following procedure. Start with L disjoint subsequences of { ( s 1 , a 1 ) , . . . , ( s j -1 , a j -1 ) } , B 1 , B 2 , . . . , B L , which are initially empty. For each j , if ( s j , a j ) is ϵ -dependent on every B 1 , . . . , B L , we achieve our goal so we stop the process. Else, we choose i ∈ [ L ] such that ( s j , a j ) is ϵ -independent on B i and update B i ←B i ∪{ ( s j , a j ) } , j ← j +1 . Since every element of B i is ϵ -independent on its predecessors, |B i | cannot get bigger than dim E ( F N , ϵ ) at any point in this process. Therefore, the process stops at most step j = L dim E ( F N , ϵ ) + 1 ≤ κ .

Now we want to show that if for some j ∈ [ κ ] such that b k h ( s j , a j ) &gt; ϵ , then ( s j , a j ) is ϵ -dependent on at most 4 β ( F N , δ ) /ϵ 2 disjoint subsequences in { ( s 1 , a 1 ) , . . . , ( s j -1 , a j -1 ) } with respect to F N . If b k h ( s j , a j ) &gt; ϵ and ( s j , a j ) is ϵ -dependent on a subsequence of { ( s ′ 1 , a ′ 1 ) , . . . , ( s ′ l , a ′ l ) } ⊆ { ( s 1 , a 1 ) , . . . , ( s κ , a κ ) } , it implies that there exists f, g ∈ F N with ∥ f -˜ f k h, ¯ η ∥ 2 Z k h ≤ β ( F N , δ ) and ∥ g -˜ f k h, ¯ η ∥ 2 Z k h ≤ β ( F N , δ ) such that f (1) ( s ′ t , a ′ t ) -g (1) ( s ′ t , a ′ t ) ≥ ϵ . By triangle inequality, ∥ f -g ∥ 2 Z k h ≤ 4 β ( F N , δ ) . On the other hand, if ( s j , a j ) is ϵ -dependent on L disjoint subsequences in { ( s 1 , a 1 ) , . . . , ( s κ , a κ ) } , then

<!-- formula-not-decoded -->

resulting in L ≤ 4 β ( F N , δ ) /ϵ 2 . Therefore, we have ( κ/ dim E ( F N , ϵ )) -1 ≤ 4 β ( F N , δ ) /ϵ 2 which results in

<!-- formula-not-decoded -->

■

Lemma D.4 (Refined version of Lemma 10 in Wang et al. (2020)) . If b k h ( s, a ) ≥ w (1) (( F N ) k h , s, a ) for all ( s, a ) ∈ S × A and k ∈ [ K ] , then

<!-- formula-not-decoded -->

Proof. We first sort the sequence { b k h ( s k h , a k h ) } ( k,h ) ∈ [ K ] × [ H ] in a decreasing order and denote it by { e 1 , . . . , e T } ( e 1 ≥ e 2 ≥ · · · ≥ e T ) . By Lemma D.3, for any constant M &gt; 0 and e t ≥ 1 / √ MT , we have

<!-- formula-not-decoded -->

which implies

<!-- formula-not-decoded -->

for t ≥ dim E ( F N , 1 /T ) . Since we have e t ≤ H ,

<!-- formula-not-decoded -->

Taking M →∞ ,

<!-- formula-not-decoded -->

Theorem (5.5) . Under Assumption 3.7, with probability at least 1 -δ , SF-LSVI achieves a regret bound of

<!-- formula-not-decoded -->

Proof. Recall that ξ k h = [ P h ( V k h +1 -V π k h +1 )]( s k h , a k h ) -( V k h +1 ( s k h +1 ) -V π k h +1 ( s k h +1 )) is a martingale difference sequence where E [ ξ k h | F k h ] = 0 and | ξ k h | ≤ 2 H . By Azuma-Hoeffding's inequality, with probability at least 1 -δ/ 2 ,

<!-- formula-not-decoded -->

Conditioning on the above event and Lemma D.4, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

■