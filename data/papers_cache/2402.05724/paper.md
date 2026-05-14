## Model-Based RL for Mean-Field Games is not Statistically Harder than Single-Agent RL

Jiawei Huang 1 Niao He 1 Andreas Krause 1

## Abstract

We study the sample complexity of reinforcement learning (RL) in Mean-Field Games (MFGs) with model-based function approximation that requires strategic exploration to find a Nash Equilibrium policy. We introduce the Partial Model-Based Eluder Dimension (P-MBED), a more effective notion to characterize the model class complexity. Notably, P-MBED measures the complexity of the single-agent model class converted from the given mean-field model class, and potentially, can be exponentially lower than the MBED proposed by Huang et al. (2024). We contribute a model elimination algorithm featuring a novel exploration strategy and establish sample complexity results polynomial w.r.t. P-MBED. Crucially, our results reveal that, under the basic realizability and Lipschitz continuity assumptions, learning Nash Equilibrium in MFGs is no more statistically challenging than solving a logarithmic number of single-agent RL problems . We further extend our results to Multi-Type MFGs, generalizing from conventional MFGs and involving multiple types of agents. This extension implies statistical tractability of a broader class of Markov Games through the efficacy of mean-field approximation. Finally, inspired by our theoretical algorithm, we present a heuristic approach with improved computational efficiency and empirically demonstrate its effectiveness.

## 1. Introduction

Multi-Agent Reinforcement Learning (MARL) has excelled in modeling cooperative and competitive interactions among agents in unknown environments. However, the well-known 'curse of multi-agency' poses a challenge in equilibrium solving for MARL systems with large populations. Yet, for MARL systems with symmetric agents, such as human crowds or road traffic, one can leverage such special structure by employing mean-field approximation, leading to the RL for Mean-Field Games (MFGs) setting (Huang et al., 2006; Lasry &amp; Lions, 2007). Notably, MFGs offer a promising framework where the complexity of learning Nash Equilibrium (NE) needs not depend on the number of agents (Lauri` ere et al., 2022). It has found successful applications in various domains, including financial markets (Cardaliaguet &amp; Lehalle, 2018), economics (Gomes &amp; Pimentel) and energy management (Djehiche et al., 2016).

1 Department of Computer Science, ETH Zurich. Correspondence to: Jiawei Huang &lt; jiawei.huang@inf.ethz.ch &gt; .

Proceedings of the 41 st International Conference on Machine Learning , Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s).

Similar to single-agent RL (Jiang et al., 2017; Jin et al., 2018), for MFGs, one of the most important questions is to understand how many samples are required to explore the unknown environment and solve the equilibrium, a.k.a. the sample complexity . Given the complex dynamics of meanfield systems and high cost of generating samples from large population, designing strategic exploration methods for sample-efficient learning becomes imperative.

Existing works on learning MFGs primarily focus on modelfree approaches such as Q-learning (Anahtarci et al., 2023; Guo et al., 2019), policy gradient (Subramanian &amp; Mahajan, 2019; Yardim et al., 2022), fictitious play (Perrin et al., 2020; Xie et al., 2021), etc. Several recent works further extend these model-free approaches with value function approximation to handle large state-action space (Mao et al., 2022; Zhang et al., 2023). However, their sample complexity results ubiquitously rely on strong structural assumptions such as contractivity (Guo et al., 2019) or monotonicity (Perolat et al., 2021). Their methods, moreover, are usually specialized and lack generalizability, leaving an open challenge of efficiently exploring mean-field systems without those structural assumptions .

To address this gap, Huang et al. (2024) establish general sample complexity results for model-based RL in MFGs 1 . They introduce a complexity measure known as ModelBased Eluder Dimension (MBED) to characterize the com- plexity of the model function class. Their algorithm, under basic realizability and Lipschitz continuity assumptions, achieves a sample complexity upper bound polynomial w.r.t. MBED. However, as we will show in Prop. 3.4, even for the tabular setting, MBED can be exponential in the number of states in the worst case. This observation, coupled with the tractability of tabular MFGs under additional structural assumptions (Guo et al., 2019; Perolat et al., 2021), prompts a fundamental question:

1 Model-based RL has also been explored in Mean-Field Control (MFC) setting, where all the agents are cooperative (Huang et al., 2024; Pasztor et al., 2021).

## Is learning MFGs statistically harder than single-agent RL in general?

In this paper, we provide a definitive answer to this question. Our main contributions are summarized as follows:

- In Sec. 3, we introduce a novel complexity measure for any given mean-field model class M , called Partial Model-Based Eluder Dimension (P-MBED) . P-MBED represents the complexity of the single-agent model class derived from M after (adversarially) fixing the state density for the transition functions in M . We show that P-MBED can be significantly lower than MBED (Huang et al., 2024). For example, in the tabular setting, P-MBED is always bounded by the number of states and actions, yielding an exponential improvement over MBED.
- In Sec. 4, we propose a model elimination algorithm capable of exploring the mean-field system and returning an approximate NE policy with sample complexity polynomial w.r.t. P-MBED. From the algorithmic perspective, our results indicate that under the basic realizability and Lipschitz assumptions, learning MFGs is no more statistically challenging than solving log |M| single-agent RL problems . As a direct implication, the sample complexity of tabular MFGs only polynomially depends on the number of states, actions, horizon and log |M| . This is the first result indicating that learning tabular MFGs is provably sample-efficient in general, even without the contractivity or monotonicity assumptions.
- In Sec 6, we design a heuristic algorithm with improved computational efficiency building upon our insights in theory. We evaluate it in a synthetic linear MFGs setting and validate its effectiveness.

As a substantial extension, we further examine the sample complexity of more general MFGs with heterogeneous population, specifically Multi-Type MFGs (MT-MFGs) (Ghosh &amp;Aggarwal, 2020; Perolat et al., 2021; Subramanian et al., 2020). MT-MFGs comprise multiple types of agents with distinct transition models, reward functions or even stateaction spaces. MT-MFGs have stronger capacity in modeling the diversity of agents, while being more tractable than general Markov Games 2 . However, the fundamental sample complexity in the setting remains largely unexplored. Our additional contribution includes:

- In Sec. 5, we show that finding the NE in an MT-MFG is equivalent to finding the NE in a lifted MFG with constraints on policies. Building on this insight, we establish the first sample complexity upper bound for learning MTMFGs. Our results identify statistical tractability of a broad class of MARL systems, potentially offering new insights to the sample complexity analysis for solving NE in general Markov Games.

## 1.1. Related Work

Within the abundant literature on single-agent RL and MFGs, below we focus primarily on sample complexity results for solving these problems in unknown environments. We defer additional related works to Appx. B.

Single-Agent RL When the number of states and actions is extremely large, sample complexity bounds derived for tabular RL (Auer et al., 2008; Azar et al., 2017; Jin et al., 2018) become vacuous. Instead, function approximation is usually considered, where a model or value function class containing the true model or optimal value functions is available, and the sample complexity is governed by the complexity of the function classes (Agarwal et al., 2020; Du et al., 2021; Foster et al., 2021; Jiang et al., 2017; Jin et al., 2020; 2021a; Sun et al., 2019). Compared with singleagent RL, the main challenge in MFGs is the additional dependence on density in transition and reward functions, especially that the density space is continuous. Although our P-MBED is inspired by the eluder dimension in the singleagent setting (Levy et al., 2022; Osband &amp; Van Roy, 2014; Russo &amp; Van Roy, 2013), it is a novel complexity notion in characterizing the sample efficiency of RL in MFGs.

Mean-Field Games Most existing results for learning MFGs primarily focus on tabular setting and model-free approaches (Cui &amp; Koeppl, 2021; Elie et al., 2020; Guo et al., 2019; Xie et al., 2021), where strong structural assumptions, such as contractivity (Guo et al., 2019), monotonicity and density independent transition (Perrin et al., 2020), or nonvanishing regularization (Yardim et al., 2022) are usually required. In contrast, we focus on addressing the fundamental exploration challenge for general MFGs. Mishra et al. (2020) study non-stationary MFG without strong structural assumptions, but their algorithm is inefficient and no sample complexity results were provided. Beyond the tabular setting, Huang et al. (2024) is the most related to us. However, as implied by our results in this paper, their sample complexity bound are suboptimal.

2 The general Markov Games (MGs) framework considers individually distinct agents. However, this generality comes with challenges. Existing results in MGs are restricted in learning (Coarse) Correlated Equilibria (Bai et al., 2020; Daskalakis et al., 2023; Jin et al., 2021b) and the sample complexity in function approximation setting may still depend on the number of agents (Cui et al., 2023; Wang et al., 2023). MT-MFGs can be regarded an intermediary between standard MFGs and general MGs.

Multi-Type Mean-Field Games Subramanian et al. (2020) study more general multi-type cases, but they consider the transition model depending on action density instead of state density. Besides, the multi-type setting has been investigated in special cases, such as LQR (Moon &amp; Bas ¸ar, 2018; uz Zaman et al., 2023), and leader-follower structures (Vasal &amp; Berry, 2022). Ghosh &amp; Aggarwal (2020) is the most related to us. However, they consider the discounted stationary setting and assume the state density is fixed, while ours is more challenging since we need to keep tracking the evolution of state density. Perolat et al. (2021) also consider the multi-type setting, but they require the monotonicity assumption. Moreover, they only provide asymptotic rates without sample complexity guarantees.

## 2. Background

Notations Throughout the paper, we will use standard bigoh notations O ( · ) , Ω( · ) , Θ( · ) , and notations such as ˜ O ( · ) to (partially) suppress logarithmic factors. In Appx. A, we list all the frequently used notations in this paper.

## 2.1. Mean-Field Games

Mean-Field Markov Decision Process We consider the finite-horizon non-stationary Mean-Field MDP (MF-MDP) M := ( µ 1 , S , A , H, P M , r ) , where µ 1 is the known initial state distribution; S = ( S 1 = ... = S H ) and A = ( A 1 = ... = A H ) are the state and action spaces, which are discrete but can be arbitrarily large; P M := { P M,h } h ∈ [ H ] with P M,h : S h ×A h × ∆( S h ) → ∆( S h +1 ) is the transition function and r := { r h } h ∈ [ H ] with r h : S h × A h × ∆( S h ) → [0 , 1 H ] is the deterministic reward function, where ∆( X ) denotes the probability measure over X . We use Π := { π := { π h } h ∈ [ H ] | π h : S h → ∆( A h ) } to denote the policy class including all non-stationary Markovian policies, and we only focus on policies in Π . Given π ∈ Π and initial density µ π M, 1 := µ 1 , the state density µ π M := { µ π M,h } h ∈ [ H ] evolves according to µ π M,h +1 = Γ π M,h ( µ π M,h ) , h ∈ [ H ] where Γ π M,h ( µ h )( · ) := ∑ s h ,a h µ h ( s h ) π ( a h | s h ) P M,h ( ·| s h , a h , µ h ) .

Given any π, ˜ π ∈ Π , we use E ˜ π,M ( π ) [ · ] to denote the expectation over trajectories generated by executing policy ˜ π while fixing the transitions and rewards to P M,h ( ·|· , · , µ π M,h ) , r h ( · , · , µ π M,h ) . These trajectories can be interpreted as the observations of a deviated agent taking ˜ π while all the others take π . Besides, we define V ˜ π M,h ( · ; µ π M ) := E ˜ π,M ( π ) [

∑ H h ′ = h r h ′ ( s h ′ , a h ′ , µ π M,h ′ ) | s h = · ] to be the value function at step h if the agent deploys policy ˜ π in model M conditioning on π , and define J M ( ˜ π ; π ) := E s 1 ∼ µ 1 [ V ˜ π M, 1 ( s 1 ; µ π M )] to be the total return of policy ˜ π conditioning on π . The Nash Equilibrium (NE) π NE M of model M is defined to be the policy s.t. no agent tends to deviate, i.e., ∀ ˜ π ∈ Π , J M ( ˜ π ; π NE M ) ≤ J M ( π NE M ; π NE M ) . We denote E NE M ( π ) := max ˜ π ∆ M ( ˜ π, π ) to be the NE-Gap, where ∆ M ( ˜ π, π ) := J M ( ˜ π ; π ) -J M ( π ; π ) .

Model-Based Setting In our model-based setting, the learner can get access to a transition function class M⊂ {{ P M,h } h ∈ [ H ] |∀ h, P M,h : S h ×A h × ∆( S h ) → ∆( S h +1 ) } to approximate the true model M ∗ . We assume the reward function r is known. In Appx. C, we provide informal discussion about how to extend our results to the setting when r is unknown. Our main objective is to find an ε -approximate NE ̂ π NE M ∗ , satisfying E NE M ∗ ( ̂ π NE M ∗ ) ≤ ε . Same as Huang et al. (2024), we only make two basic assumptions about the function class M : realizability and Lipschitz continuity.

Assumption A (Realizability) . M ∗ ∈ M .

Assumption B (Lipschitz Continuity) . For any M ∈ M , and arbitrary policies π, ˜ π ∈ Π , ∀ h, s h , a h , we have:

<!-- formula-not-decoded -->

Note that our Assump. B only requires Lipschitz continuity on feasible densities. In contrast, contractivity assumes L r and L T are sufficiently small (Guo et al., 2019; Yardim et al., 2022), and prior works considering monotonicity (Perolat et al., 2021; Zhang et al., 2023) usually assume the transition is independent w.r.t. density, i.e., L T = 0 .

We consider the same trajectory sampling model as Huang et al. (2024), which is much weaker than the generative model assumptions requiring trajectories conditioning on arbitrary state densities in most MFGs literatures (Anahtarci et al., 2023; Guo et al., 2019; Perrin et al., 2020).

Definition 2.1. The sampling model can be queried with arbitrary ˜ π, π ∈ Π , and return a trajectory by executing ˜ π while transition and reward functions are fixed to P M ∗ ,h ( ·|· , · , µ π M ∗ ,h ) and r h ( · , · , µ π M ∗ ,h ) for all h .

MFGs and N -Player Symmetric Anonymous Games MFGs can be regarded as the limit of Symmetric Anonymous Games (SAGs) when the number of agents N approaches infinity (Guo et al., 2019; Yardim et al., 2022). As explained in (Huang et al., 2024), the sampling model (Def. 2.1) is reasonable for N -player SAGs with central controllers, which can manipulate all the agents' policies.

Given a SAG, it is known that the NE of its MFG approximation is a O ( N -1 / 2 ) -approximate NE for the SAG (Yardim et al., 2024), if all the agents execute that same NE policy. In this way, one may interpret our setting as centralized training with decentralized execution.

## 2.2. Multi-Type Mean-Field Games

Multi-Type Mean-Field MDP A finite horizon nonstationary Multi-Type (or Multi-Group) MF-MDP M with W types of agents can be described by a collection of tuples M := { ( µ w 1 , S w , A w , H, P w M , r w ) w ∈ [ W ] } , where we use w in superscription to distinguish the initial state distribution, state-action spaces, the transition and reward functions in different groups. Besides, for any w , the transition and reward functions depend on densities in all types. More concretely, we have P w M := { P w M ,h } h ∈ [ H ] with P w M ,h : S w h × A w h × ∆( S 1 h ) × ... × ∆( S W h ) → ∆( S w h +1 ) and r w := { r w h } h ∈ [ H ] with r w h : S w h ×A w h × ∆( S 1 h ) × ... × ∆( S W h ) → [0 , 1 H ] . For each type of agents, we consider the Markovian policies Π w := { π w := { π w h } h ∈ [ H ] |∀ h, π w h : S w h → ∆( A w h ) } , and use Π := { π := { π w } w ∈ [ W ] |∀ w ∈ [ W ] , π w ∈ Π w } to denote the collection of policies for all types. For the function approximation setting, we assume W function classes M 1 , ..., M W are available, where ∀ w ∈ [ W ] , M w ⊂ {{ P w h } h ∈ [ H ] |∀ h ∈ [ H ] , P w h : S w h × A w h × ∆( S 1 h ) × ... × ∆( S W h ) → ∆( S w h ) } is used to approximate the transition function for the w -th group. The MT-MFG function class M is then defined by M ← { M := M 1 × ... × M W |∀ w ∈ [ W ] , M w ∈ M w } , which we use to approximate the true model M ∗ .

For the lack of space, we defer the definitions of value functions, Nash Equilibrium, and other related details to Appx. G.1. For the assumptions in MT-MFG setting, we defer to Appx. G.3.

## 3. Partial Model-Based Eluder Dimension

In the function approximation setting, the exploration challenge is related to the complexity of the function classes. In this section, we introduce new notions to characterize the complexity of model function class for MFGs and its extension to Multi-Type MFGs setting. The proofs and additional discussions can be found in Appx. D.

Inspired by the Eluder dimension of single-agent value function classes (Jin et al., 2021a; Russo &amp; Van Roy, 2013) and mean-field model function classes (Huang et al., 2024), similarly, we use the length of independent sequences to characterize the complexity of function classes. In Def. 3.1, we first introduce the definition of standard ε -independence in previous Eluder dimension literature, to highlight the difference from our partial ε -independence. Although we only consider the l 1 -distance here, similar discussion can be generalized to other distances, e.g., the Hellinger distance.

Definition 3.1 ( ε -Independence; (Huang et al., 2024)) . Given M and a data sequence { ( s i h , a i h , µ i h ) } n i =1 ⊂ S h × A h × ∆( S h ) , we say ( s h , a h , µ h ) is ε -independent of { ( s i h , a i h , µ i h ) } n i =1 w.r.t. M if there exists M, ˜ M ∈ M such that ∑ n i =1 ∥ P M,h ( ·| s i h , a i h , µ i h ) -P ˜ M,h ( ·| s i h , a i h , µ i h ) ∥ 2 1 ≤ ε 2 but ∥ P M,h ( ·| s h , a h , µ h ) -P ˜ M,h ( ·| s h , a h , µ h ) ∥ 1 &gt; ε . We call { ( s i h , a i h , µ i h ) } n i =1 an ε -independent sequence w.r.t. M (at step h ) if for any i ∈ [ n ] , ( s i h , a i h , µ i h ) is ε -independent w.r.t. { ( s t h , a t h , µ t h ) } i -1 t =1 .

Definition 3.2 (Partial ε -Independence) . Given M , a mapping ν h : M→ ∆( S h ) , and a data sequence { ( s i h , a i h ) } n i =1 ⊂ S h ×A h , we say ( s h , a h ) is partially ε -independent of { ( s i h , a i h ) } n i =1 ⊂ S h × A h w.r.t. M and ν h , if there exists M, ˜ M ∈ M , s.t. ∑ n i =1 ∥ P M,h ( ·| s i h , a i h , ν h ( M )) -P ˜ M,h ( ·| s i h , a i h , ν h ( ˜ M )) ∥ 2 1 ≤ ε 2 but ∥ P M,h ( ·| s h , a h , ν h ( M )) -P ˜ M,h ( ·| s h , a h , ν h ( ˜ M )) ∥ 1 &gt; ε . We call { ( s i h , a i h ) } n i =1 a partially ε -independent sequence w.r.t. M and ν h (at step h ) if for any i ∈ [ n ] , ( s i h , a i h ) is partially ε -independent on { ( s t h , a t h ) } i -1 t =1 .

Intuitively, a partially ε -independent sequence of M is an independent sequence w.r.t. the function class converted from M by using some mapping ν h to 'partially' fix the input (the density part) for each function in M . We use dim E | ν h ( M , ε ) to denote the length of the longest partially ε -independent sequence w.r.t. M and ν h (at step h ). Now, we are ready to define the Partial-MBED.

Definition 3.3 (Partial MBED) . Given a model class M , and a policy π , we define the mapping ν π h : ∀ M ∈ M , ν π h ( M ) := µ π M,h . The P-MBED of M is defined by: dim PE ( M , ε ) := max h ∈ [ H ] max π dim E | ν π h ( M , ε ) .

By definition, P-MBED can be interpreted as the complexity of the single-agent model class converted from the MeanField model class M by partially (adversarially) fixing the density of the functions' input. In fact, different choices of ν in Def. 3.2 may lead to different notions of complexity. In our main text, we stick to the choice in Def. 3.3, but in Appx. D.1, we discuss an alternative choice of ν , its induced P-MBED and associated properties.

Next, we take the tabular setting as an example, and show that P-MBED of any function class for tabular MFGs can be controlled by |S||A| , while MBED (Huang et al., 2024) can be exponential in |S| in the worst case. This is reasonable given the single-agent nature of P-MBED.

Proposition 3.4. (Tabular Setting) For any M and ε &gt; 0 , dim PE ( M , ε ) ≤ |S||A| , while there exists a concrete example of M such that dim E ( M , ε ) = Ω(exp( |S| )) .

Below we provide the linear mean-field model classes with decomposable transition functions as another exam- ple. When the transition is independent w.r.t. density µ (i.e. G ( µ ) is constant), the linear MFGs reduce to the singleagent linear MDP (Jin et al., 2020). As we can see, the P-MBED of the model class of linear MFGs is only related to the dimension of the state-action feature, which matches the complexity of their single-agent correspondence.

Proposition 3.5 (Linear MFGs; Informal version of Prop. D.4) . Consider the model class: M Ψ := { P ψ | P ψ ( ·| s, a, µ ) := ϕ ( s, a ) ⊤ G ( µ ) ψ ( s ′ ); ψ ∈ Ψ } , with known feature ϕ ( · , · ) ∈ R ˜ d , G ( · ) ∈ R ˜ d × d , and a next-state feature class Ψ satisfying some normalization conditions. Then dim PE ( M , ε ) = ˜ O ( ˜ d ) .

Similarly, for model classes in Multi-Type MFGs setting, we can define the Multi-Type P-MBED generalized from dim PE in MFGs, which we denote as dim MTPE . We defer its formal definition to Appx. D.3. Likewise, dim MTPE can be regarded as the complexity measure for a collection of W single-agent model classes converted from M . In the tabular case (resp. Prop. D.11), we have dim MTPE ( M , ε ′ ) = ˜ O ( ∑ w ∈ [ W ] |S w ||A w | ) , and in linear MT-MFG setting with decomposable transitions (resp. Prop. D.12), dim MTPE ( M , ε ′ ) = ˜ O ( ∑ w ∈ [ W ] d w ) where { d w } w ∈ [ W ] are the dimensions of the state-action features.

## 4. Sample Efficiency of Learning in MFGs

In this section, we show that the sample complexity of learning NE in MFGs is indeed governed by our new complexity notion P-MBED. We highlight our main algorithm and sample complexity results in Sec. 4.1, and then explain details in the algorithm design and technical novelty in Sec. 4.2. The missing details and proofs for results in this section are deferred to Appx. F.

## 4.1. Main Algorithm and Highlight of Main Results

Before proceeding to the algorithms, we first introduce several useful notions. Given a reference policy π , we denote d ( M, ˜ M | π ) := max ˜ π d ˜ π ( M, ˜ M | π ) ∨ d ˜ π ( ˜ M,M, | π ) as the conditional model distance between M and ˜ M , where d ˜ π ( M, ˜ M | π ) := E ˜ π,M ( π ) [ ∑ H h =1 ∥ P M,h ( ·|· , · , µ π M,h ) -P ˜ M,h ( ·|· , · , µ π ˜ M,h ) ∥ 1 ] . Given a model class M ′ , any M ∈ M ′ , and any policy π , we define the ε 0 -neighborhood of M in M ′ to be: B ε 0 π ( M ; M ′ ) := { M ′ ∈ M ′ | d ( M,M ′ | π ) ≤ ε 0 } . The 'Central Model' (abbr. CM) in M ′ w.r.t. policy π is defined to be the model with the most number of neighbors: M ε 0 Ctr ( π ; M ′ ) := arg max M ∈M ′ |B ε 0 π ( M ; M ′ ) | . Besides, when ε 0 and M ′ is clear from the context, we will use M π Ctr as a short note of M ε 0 Ctr ( π ; M ′ ) . Lastly, ∀ π, π ′ ∈ Π , we define d ∞ , 1 ( π, π ′ ) := max h,s h ∥ π ( ·| s h ) -π ′ ( ·| s h ) ∥ 1 .

We provide our main algorithm in Alg. 1. The basic idea is to find a sequence of 'reference policies' ( π k or π NE ,k Br , k = 1 , 2 , ... ) and run the model elimination steps (Alg. 2 as ModelElim ) to gradually remove models in M that distinct from M ∗ conditioning on these reference policies, until find an approximate NE. Next, we highlight our main results and its implications.

Theorem 4.1. [Informal version of Thm. F.7] Under Assump. A and B, with appropriate hyperparameter choices, w.p. 1 -δ , Alg. 1 terminates at some k ≤ log 2 |M| +1 and returns an ε -NE of M ∗ after consuming at most

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Model-Based RL for MFGs is not Statistically Harder than Single-Agent RL As we will explain more in the next section, ModelElim only needs to be a single-agent model elimination subroutine , and it is the only step consuming samples. Therefore, Thm. 4.1 suggests that the sample complexity of learning MFGs can be characterized by a O (log |M| ) number of single-agent model elimination subproblems, whose learning complexity is controlled by PMBED. As a result, the total sample complexity only scales with P-MBED and the log-covering number of M .

Based on the discussion in Sec. 3, we can expect for many model classes with low P-MBED (e.g. tabular setting Prop. 3.4, linear setting Prop. 3.5), learning MFGs is provable sample-efficient. In particular, for tabular MFGs where dim PE ( M , ε ′ ) ≤ |S||A| , our result yields a sample complexity with polynomial dependence on |S| , |A| , H , which implies that tabular MFGs are provably efficient in general if considering the model-based function approximation, even without assuming contractivity or monotonicity that are often required in existing works (Guo et al., 2019; Perrin et al., 2020; Yardim et al., 2022). Compared with recent results in function approximation setting for MFGs (Huang et al., 2024) or MFC (Pasztor et al., 2021) with similar Lipschitz assumptions, our result does not suffer the exponential term (1 + L T ) H (see Remark F.4 for more explanation).

Additional Remarks on log |M| Although low logcovering number of function class is regarded as a standard assumption in many MARL works (Cui et al., 2023; Wang et al., 2023), we would like to take the tabular MFGs as an example and provide remarks about the magnitude of log |M| . Under Assump. B, with appropriate discretization, the ε -cover for all possible transition functions could be Ω(exp( SAH N ε (∆( S )))) , where N ε (∆( S )) denotes the covering number of density space ∆( S ) and we omit L T , L r . As a result, in the worst case, log |M| = Ω( N ε (∆( S ))) , which could be exponential in SA . Nonetheless, there are many examples, such that, even in the worst case, log |M| is acceptable. For instance, if the model class is parameterized by some θ ∈ Θ (e.g. Neural Networks)

```
1 Input : Model Class M ; Parameters ε 0 , ˜ ε, ¯ ε, δ . 2 Initialize : M 1 ←M , δ 0 ← δ log 2 |M| +1 3 for k = 1 , 2 , ... do 4 π k ← arg min π |B ε 0 π ( M π Ctr ; M k ) | ; 5 if |B ε 0 π k ( M π k Ctr ; M k ) | ≤ |M k | 2 then M k +1 ← ModelElim ( π k , M k , ˜ ε, δ 0 ) ; 6 else 7 π NE ,k Br ← BridgePolicy ( M k , ¯ ε ) ; M k +1 ← ModelElim ( π NE ,k Br , M k , ˜ ε, δ 0 ) ; 8 Randomly pick ˜ M k from M k +1 ; E NE ˜ M k ( π NE ,k Br ) ← max π ∆ ˜ M k ( π, π NE ,k Br ) 9 if E NE ˜ M k ( π NE ,k Br ) ≤ 3 ε 4 then return π NE ,k Br ; 10 end 11 if |M k | = 1 then Return the NE of the model ; 12 end
```

Algorithm 1: MEBP : M odel E limination via B ridge P olicy

and take the concatenation of [ s, a, µ ] ∈ R dim( S )+dim( A )+ S as inputs, where we use dim( · ) to denote the dimension of a given set. Then log |M| = ˜ O (dim(Θ)) , which could just scale with ˜ O ( Poly (dim( S ) , dim( A ) , S ) } ) . As another example, when the transition function only depends on some sufficient statistics of density instead of the exact density, (e.g. P ( ·| s, a, µ ) = P ( ·| s, a, E ˜ s ∼ µ [ ˜ s ]) ), we may have log |M| = ˜ O ( Poly ( S, A, H )) in the worst case. Note that, for the single-agent RL, the largest log-covering number of models is also bounded by ˜ O ( Poly ( S, A, H )) (folklore).

Exponential Separation between MFGs and MFC Different from MFGs, in Mean-Field Control (MFC) setting, agents cooperate to find an optimal policy to maximize the total return. Previous work (Huang et al., 2024) suggests that both MFC and MFGs can be solved via a unified MLE framework with similar sample complexity upper bounds. One natural question is: whether learning MFC can also be as sample-efficient as single-agent RL ?

We provide a negative answer to this question. In Thm. F.9, we show that even in tabular setting, there exists a hard instance such that learning MFC requires Ω(exp( |S| )) samples. This suggests an exponential separation between learning MFC and MFGs from information-theoretical perspective. Intuitively, for MFC, in the worst case, the agent should explore the entire S×A× ∆( S ) space to identify the policy that achieves the maximal return. In contrast, as we will explain in Lem. 4.3, in MFGs setting, the learner does not have to explore the entire state-action-density space; instead, finding a 'locally-aligned equilibrium policy' is enough.

## 4.2. Algorithm Design and Proof Sketch

4.2.1. MODELELIM : THE MODEL ELIMINATION STEP

ModelElim can be arbitrary single-agent model elimination procedures. Here we provide an example in Alg. 2.

The basic idea of Alg. 2 is to eliminate models not aligned with M ∗ conditioning on the given reference policy π . In each iteration, we first find a tuple ( ˜ π t , M t , M ′ t ) resulting in the maximal discrepancy ∆ t max . As long as ∆ t max &gt; ˜ ε , we collect samples and remove models with low likelihood. With high probability, on the one hand, M ∗ will never be ruled out under Assump. A; on the other hand, the growth of ∑ t ∆ t max is controlled by P-MBED. As a result, the algorithm will terminate eventually and return a model class only including those M with small d ( M,M ∗ | π ) . We summarize the result in the theorem below.

Theorem 4.2. [Informal version of Thm. F.3] Given any reference policy π, ˜ ε, δ ∈ (0 , 1) , if M ∗ ∈ ¯ M , by choosing T = ˜ O ( H 4 dim PE ( M ,ε ′ ) ˜ ε 2 ) with ε ′ = O ( ˜ ε H 2 (1+ L T ) H ) , w.p. 1 -δ , Alg. 2 terminates at some T 0 ≤ T , and return ¯ M T 0 s.t. (i) M ∗ ∈ ¯ M T 0 (ii) ∀ M ∈ ¯ M T 0 , d ( M ∗ , M | π ) ≤ ˜ ε .

We claim Alg. 2 is a single-agent model elimination subroutine, because from Line 8-11, we can see that Alg. 2 only eliminates those M ∈ M k s.t. P M,h ( ·|· , · , µ π M,h ) distinct from P M ∗ ,h ( ·|· , · , µ π M ∗ ,h ) under some adversarial policy ˜ π t or π itself. Here the density part of P M,h is fixed by µ π M , so during the elimination, all the transitions reduce to singleagent functions only depending on states and actions.

Beyond P-MBED Notably, although we focus on P-MBED in this paper, one may consider other complexity measures generalized from single-agent RL setting (Foster et al., 2021; Sun et al., 2019) and our analysis can be extended correspondingly. That's because as long as ModelElim satisfies the (i) and (ii) in Thm. 4.2, it can be arbitrary and does not affect the function of other components in Alg. 1.

## 4.2.2. FAST ELIMINATION WITH BRIDGE POLICY

We seek to construct reference policies that allow to eliminate models as efficient as possible, more specifically, to

## Algorithm 2: Model Elimination given a Policy

```
1 Input : Reference Policy π ; ¯ M ; ˜ ε, δ 2 Initialize : ¯ M 1 ← ¯ M ; Z 0 ←{} ; Set T by Thm. F.3; 3 for t = 1 , 2 , ..., T do 4 ˜ π t ← arg max ˜ π max M,M ′ ∈ ¯ M t E ˜ π,M ( π ) [ ∑ H h =1 ∥ P M,h ( ·|· , · , µ π M,h ) -P M ′ ,h ( ·|· , · , µ π M ′ ,h ) ∥ 1 ] . 5 ∆ t max ← the maximal value achieved above. 6 if ∆ t max ≤ ˜ ε then return ¯ M t ; 7 else 8 for h = 1 , 2 ..., H do 9 Query Sampling Oracle (Def. 2.1) with ( π, π ) ; collect the data at step h : z t h := { ( s t h , a t h , s ′ t h +1 ) } . 10 Query Sampling Oracle (Def. 2.1) with ( ˜ π t , π ) ; collect the data at step h : ˜ z t h := { ( s t h , a t h , s ′ t h +1 ) } . 11 Z t ←Z t -1 ∪ z t h ∪ ˜ z t h . 12 end 13 ∀ M ∈ ¯ M t , l π MLE ( M ; Z t ) := ∑ i ∈ [ t ] ,h ∈ [ H ] log P M,h ( s ′ i h +1 | s i h , a i h , µ π M,h ) + log P M,h ( ˜ s ′ i h +1 | ˜ s i h , ˜ a i h , µ π M,h ) . 14 ¯ M t +1 ←{ M ∈ ¯ M t | l π MLE ( M ; Z t ) ≥ max ˜ M l π MLE ( ˜ M ; Z t ) -log HT |M| δ } . 15 end 16 end 17 return ¯ M T .
```

halve the model candidates every iteration until finding the NE. We first consider the simple case, where the models are 'scattered' and easy to be distinguished: there exists a policy π k , such that, no more than |M k | / 2 models are around its CM(resp. If -branch, Line 5 in Alg. 1). In this case, after running ModelElim with π k , M k +1 only contains those models locating at the neighborhood of M ∗ conditioning π k , which implies |M k +1 | ≤ |B ε 0 π k ( M π k Ctr ; M k ) | ≤ |M k | / 2 .

The challenging scenario is that, for any policy, the corresponding CM is surrounded by over a half of models (resp. Else -branch, Line 6 in Alg. 1). In that case, unstrategically selecting reference policies leads to inefficient model elimination. We present a subtle choice of reference policy, called Bridge Policy , which can be constructed by Alg. 3. Before diving into the details of our constructions, we first explain the key insights behind it. Our first insight is summarized in the lemma below.

Lemma 4.3. [Implication of Local Alignment] Given any M, ˜ M with transition P M and P ˜ M , denote ̂ π NE M to be an ε 1 -approximate NE of M , suppose d ( M, ˜ M | ̂ π NE M ) ≤ ε 2 , then ̂ π NE M is also an O ( ε 1 + ε 2 ) -approximate NE of ˜ M .

Lem. 4.3 states that, if two models M and ˜ M align with each other conditioning on the NE of one of them, then they approximately share that NE. Therefore, in the Else -branch, after calling ModelElim with π NE ,k Br as the reference policy, if the NE-Gap E NE ˜ M k ( π NE ,k Br ) is small for some randomly selected ˜ M k ∈ M k +1 (resp. Line 9), we can claim π NE ,k Br is an approximate NE of M ∗ by Lem. 4.3.

However, the remaining challenge is that, if E NE ˜ M k ( π NE ,k Br )

is large, we cannot conclude anything about it. Hence, π NE ,k Br should be chosen in a strategic way, so that in this case, we can guarantee the elimination is efficient, i.e. |M k +1 | ≤ |M k | / 2 . Our second key insight to overcome this challenge is summarized in Thm. 4.4, which indicates that the existence of a 'Bridge Policy' that coincides with the NE of its corresponding CM.

Theorem 4.4. [Bridge Policy] If the Else -branch in Line 6 in Alg. 1 is activated, running Alg. 3 returns a bridge policy π NE ,k Br , such that, π NE ,k Br is an approximate NE of M π NE ,k Br Ctr .

Before we explain how to prove Thm. 4.4, we first check the implication of this result. Based on Thm. 4.4, if E NE ˜ M k ( π NE ,k Br ) &gt; 3 ε 4 in Line 9, by Lem. 4.3, we know d ( M π NE ,k Br Ctr , M ∗ | π NE ,k Br ) cannot be small. Therefore, with appropriate hyperparameter choices, we can assert that all models in the neighborhood B ε 0 π NE ,k Br ( M π NE ,k Br Ctr , M k ) should have been eliminated, implying |M k +1 | ≤ |M k | / 2 .

Combining the discussions above, we know our Alg. 1 guarantees to either return an approximate NE, or at least halve the model sets. We summarize to the following theorem, which paves the way to our main theorem Thm. 4.1.

Theorem 4.5. In Alg. 1, by choosing ε 0 = ε 8(1+ L r H )( H +4) , ˜ ε = ε 0 6 , and choosing ¯ ε according to Thm. F.5, w.p. 1 -δ , (1) if the If-Branch in Line 5 is activated: we have |M k +1 | ≤ |M k | / 2 ; (2) otherwise, in the Else-Branch in Line 6: either we return the π NE ,k Br which is an ε -approximate NE for M ∗ ; or the algorithm continues with |M k +1 | ≤ |M k | / 2 .

## Algorithm 3: Bridge Policy Construction

- 1 Input : MF-MDP model class M ; ε 0 , ¯ ε ;
- 2 Convert M to a PAM class ¨ M via Eq. (1).
- 3 Construct ¯ ε -cover of the policy space Π w.r.t. d ∞ , 1 , denoted as Π ¯ ε (see Def. F.1).
- 4 for ˜ π ∈ Π ¯ ε do Find the Central Model ¨ M ε 0 Ctr ( ˜ π ; ¨ M ) ← arg max ¨ M ∈ ¨ M |B ε 0 π ( ¨ M ; ¨ M ) | ;
- 5 Construct the new PAM ¨ M Br s.t. for any s h , a h , π ,

<!-- formula-not-decoded -->

- 6 Compute the NE of ¨ M Br: π NE Br ← arg min π max ˜ π ¨ J ¨ M Br ( ˜ π ; π ) -¨ J ¨ M Br ( π ; π ) .
- 7 return π NE Br .

Proof Sketch of Thm. 4.4 An informal way to interpret the existence of such bridge policy in Thm. 4.4 is to consider a mapping T from an arbitrary π ∈ Π to the NE of its CM M π Ctr . Then, Thm. 4.4 states that T has an approximate fixed point π NE Br ≈ T ( π NE Br ) . However, given that it's hard to evaluate the continuity of T and moreover, T can be a one-to-many mapping if multiple NEs exist, we prove the existence of such π NE Br by the non-trivial construction in Alg. 3. We leave the connection between our proofs and the fixed-point theorems as an open problem.

Before explaining our construction in Alg. 3, we first introduce a new notion called 'Policy-Aware Model' (abbr. PAM) denoted by ¨ M . The main motivation for introducing PAM is that we want to focus on the policy space, because the feasible densities { µ π M ∗ ,h } π ∈ Π may not cover the entire density space ∆( S h ) , and it is not easy to characterize. We defer to Appx. E.1 for the formal definition of PAM and also new notations in Alg. 3 (e.g. ¨ M ε 0 Ctr ( · , · ) denotes Central Model, ¨ J denotes the total return), and only summarize the main idea here to save space. Briefly speaking, a PAM ¨ M := {S , A , µ 1 , H, ¨ P , ¨ r } is an MDP whose transition ¨ P : S × A × Π → ∆( S ) and reward functions ¨ r : S × A × Π → [0 , 1 H ] depend on state, action and a 'reference policy'. PAM can be regarded as a higher-level abstraction of MF-MDP (i.e. MF-MDP ⊂ PAM), where we replace the dependence on µ π M in MF-MDP by π . We can convert a MF-MDP M to a PAM ¨ M sharing the same S , A , µ 1 , H by assigning the following for any h ∈ [ H ] with µ π M, 1 = µ 1 , ∀ π ∈ Π :

<!-- formula-not-decoded -->

In Alg. 3, we first convert each MF-MDP to its PAM version. Then, we find an ¯ ε -cover of the policy space w.r.t. d ∞ , 1 , denoted by Π ˜ ε , and construct the 'Bridge PAM' ¨ M Br by interpolating among CMs w.r.t. ˜ π ∈ Π ¯ ε . Here the weights [2¯ ε -d ∞ , 1 ( π, ˜ π )] + is chosen carefully for the following

<!-- formula-not-decoded -->

## considerations:

- (I) since Π ¯ ε is an ¯ ε -cover, for any π ∈ Π , the denominator ∑ ˜ π ∈ Π ¯ ε [2¯ ε -d ∞ , 1 ( π, ˜ π )] + is always larger than ¯ ε , which implies both ¨ P Br ,h and ¨ r Br ,h are well-defined and continuous in π . The continuity is important since it implies that ¨ M Br has at least one NE (Def. E.1), denoted as π NE Br ;
- (II) [2¯ ε -d ∞ , 1 ( π, ˜ π )] + decays to zero if π largely disagrees with ˜ π , so ¨ P Br ( ·|· , · , π ) is only determined by CMs of those π close to π .

˜

Next, we discuss what we can conclude from the above two points. Based on the triggering condition in Line 6 in Alg. 1, for any π, ˜ π , the neighbors of M π Ctr and M ˜ π Ctr share at least one common model M share. By using M share as a bridge, we have ∥ ¨ P ¨ M π Ctr ( ·|· , · , π ) -¨ P ¨ M ˜ π Ctr ( ·|· , · , ˜ π ) ∥ 1 = O ( d ∞ , 1 ( π, ˜ π )) . Combining with (II) , we know ∀ π, ∥ ¨ P Br ( ·|· , · , π ) -¨ P ¨ M π Ctr ( ·|· , · , π ) ∥ 1 = O (¯ ε ) , which implies ¨ P Br ( ·|· , · , π NE Br ) ≈ P ¨ M π NE Br Ctr ( ·|· , · , π NE Br ) if ¯ ε is small enough. By the definition of NE in PAM, the conversion rules in Eq. (1) and Lem. 4.3, we can conclude that π NE Br is an approximate NE of M π NE Br Ctr , and finish the proof of Thm. 4.4.

## 5. Learning in Multi-Type MFGs

In this section, we extend our results to the more general Multi-Type MFGs setting 3 , allowing to address heterogeneous agents.

Reduction to Lifted MFGs with Constrained Policy Our key observation is that, a MT-MFG M to can be lift to a new MF-MDP M MFG := {S MFG , A MFG , µ 1 , H, P MFG , r MFG } by augmenting the original states and actions with the type index. The new state and action spaces are given by: S MFG := ⋃ w ∈ [ W ] {S w ×{ w }} and A MFG := ⋃ w ∈ [ W ] {A w ×{ w }} .

3 The existence of NE in MT-MFGs can be found in Thm. E.12.

We defer the detailed description for the conversion process and the definition of initial state distribution, transition and reward functions in M MFG to Appx. G.2.

For policies in M MFG, we only consider Π † := { π |∀ w ∈ [ W ] , π ( a w ◦ w | s w ◦ w ) = π w ( a w | s w ) , π w ∈ Π w } , including all policies which only take actions with the same type as the states. Similar to the NE defined in full policy space Π , we can define the 'constrained NE' when agents are constrained to only take policies in the subset Π † . More concretely, we call ̂ π NE Cstr ∈ Π † the ε -approximate Constrained Nash Equilibrium if ∀ π ∈ Π † , J M MFG ( π, ̂ π NE Cstr ) ≤ J M MFG ( ̂ π NE Cstr , ̂ π NE Cstr ) + ε . The following property reveals the connection between constrained NE in M MFG and the NE in the original multi-type model M .

Proposition 5.1. Given a MT-MFG M and its lifted MFG M MFG, we have: (1) an ε -constrained NE ̂ π NE Cstr ∈ Π † for M MFG is a ( Wε )-NE in M ; (2) an ε -NE ̂ π NE in M is an ε -constrained NE for M MFG.

The above result not only implies the existence of constrained NE in M MFG given the existence of NE in M by letting ε → 0 , but also suggests one can solve NE of MTMFG by solving the constrained NE in its lifted MFG. The second point is very important since the constrained NE can be solved via almost the same procedures in Sec. 4, as long as we constrain the policy space to Π † . We defer algorithm details to Appx. G.5, and summarize our main result in the following theorem.

Theorem 5.2. [Informal version of Thm. G.8] Under Assump. C and D, there exists an algorithm (Alg. 4), s.t. w.p. 1 -δ , it returns an ε -NE of M ∗ after consuming at most

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In Appx. H, we investigate a practical multi-agent system called N -player Multi-Type Symmetric Anonymous Games (MT-SAGs) generalized from SAGs. We establish approximation error between MT-MFGs and MT-SAGs. Our results reveal a larger class of Multi-Agent systems where NE can be solved in a sample-efficient way.

## 6. A Heuristic Algorithm with Improved Computational Efficiency

Although Alg. 1 is sample-efficient, it requires exponential computation. In this section, we aim to design a heuristic algorithm 4 sharing the main insights as Alg. 1 while more computationally tractable. For the lack of space, we defer the concrete algoirthm (Alg. 7), the experiment setting and evaluation results (Fig. 2) to Appx. J. In this section, we just highlight the algorithm design.

[4 The code is available at https://github.com/ jiaweihhuang/Heuristic\_MEBP .](https://github.com/jiaweihhuang/Heuristic_MEBP)

Highlights of Algorithm Design We assume a NE Oracle is available, such that given a known MFG model, the Oracle can return its NE. We argue that such oracle can be easily implemented if the model is smooth enough or the monotonicity condition is satisfied (Guo et al., 2019; Perolat et al., 2021). Besides, in our experiments, we observe that repeatedly mixing the policy with its best response can converge to a good solution. Given such oracle, Alg. 7 only involves |M| calls of NE oracle, and Poly ( |M| , |S| , |A| , H ) arithmetic operations in computing model difference or likelihood, which avoids exponential computation in Alg. 1.

For the algorithm design, Alg. 7 follows the same if-else structure as Alg. 2, but we improve the computational efficiency in two aspects. Firstly, we avoid procedures optimizing over the entire policy class, including Line 4 in Alg. 1 and Line 4 in Alg. 2. Instead, we only search over the NE policies of model candidates, which can be computed by calling the NE Oracle |M| times at the beginning. As long as the models in M are diverse enough, we can expect their NEs to be reasonable representatives for Π in distinguishing models. Secondly, we replace the π NE ,k Br in Alg. 1 with the NE of the model M k ← arg max M ∈M k |B ε 0 π NE M ( M, M k ) | , and do not have to solve the NE of the complicated bridge model in Alg. 3. We claim that this modification still aligns with Alg. 3 in principle. Note that the main intuition behind Alg. 3 is that, when Line 6 in Alg. 1 is activated, the reference policy used for elimination should be a policy π ref , such that, π ref collapses with the NE of the model with the maximal number of neighbors conditioning on π ref .

## 7. Conclusion

In this paper, we reveal that learning MFGs can be as sampleefficient as single-agent RL under mild assumptions, and the sample complexity of RL in MFGs can be characterized by a novel complexity measure called Partial Model-Based Eluder Dimension (P-MBED). Besides, we extend our algorithms to the more general Multi-Type MFGs setting. Lastly, we contribute an empirical algorithm with improved computational efficiency.

As for the future, one interesting direction is to study the sample complexity when only value function approximations are available. Besides, while our focus is the sample efficiency in this paper, it would be valuable to identify general conditions, under which computationally efficient algorithms exist. Lastly, our results underscore the power of mean-field approximation, and it would be worthwhile to investigate other generalizations of the MFGs setting, in order to deepen our understanding on the sample efficiency of learning NE in other MARL systems.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

## Acknowledgements

This research was supported by Swiss National Science Foundation (SNSF) Project Funding No. 200021-207343, SNSF starting grant, SNSF grant agreement 51NF40 180545, and by the European Research Council (ERC) under the European Union's Horizon grant 815943.

## References

- Agarwal, A., Kakade, S., Krishnamurthy, A., and Sun, W. Flambe: Structural complexity and representation learning of low rank mdps. Advances in neural information processing systems , 33:20095-20107, 2020.
- Anahtarci, B., Kariksiz, C. D., and Saldi, N. Q-learning in regularized mean-field games. Dynamic Games and Applications , 13(1):89-117, 2023.
- Auer, P., Jaksch, T., and Ortner, R. Near-optimal regret bounds for reinforcement learning. Advances in neural information processing systems , 21, 2008.
- Ayoub, A., Jia, Z., Szepesvari, C., Wang, M., and Yang, L. Model-based reinforcement learning with value-targeted regression. In International Conference on Machine Learning , pp. 463-474. PMLR, 2020.
- Azar, M. G., Osband, I., and Munos, R. Minimax regret bounds for reinforcement learning. In International Conference on Machine Learning , pp. 263-272. PMLR, 2017.
- Bai, Y., Jin, C., and Yu, T. Near-optimal reinforcement learning with self-play. Advances in neural information processing systems , 33:2159-2170, 2020.
- Cardaliaguet, P. and Lehalle, C.-A. Mean field game of controls and an application to trade crowding. Mathematics and Financial Economics , 12:335-363, 2018.
- Chen, Z., Li, C. J., Yuan, A., Gu, Q., and Jordan, M. I. A general framework for sample-efficient function approximation in reinforcement learning. arXiv preprint arXiv:2209.15634 , 2022a.
- Chen, Z., Zhou, D., and Gu, Q. Almost optimal algorithms for two-player zero-sum linear mixture markov games. In International Conference on Algorithmic Learning Theory , pp. 227-261. PMLR, 2022b.
- Cui, K. and Koeppl, H. Approximately solving mean field games via entropy-regularized deep reinforcement learning. In International Conference on Artificial Intelligence and Statistics , pp. 1909-1917. PMLR, 2021.
- Cui, Q., Zhang, K., and Du, S. S. Breaking the curse of multiagents in a large state space: Rl in markov games with independent linear function approximation. arXiv preprint arXiv:2302.03673 , 2023.
- Daskalakis, C., Golowich, N., and Zhang, K. The complexity of markov equilibrium in stochastic games. In The Thirty Sixth Annual Conference on Learning Theory , pp. 4180-4234. PMLR, 2023.
- Djehiche, B., Tcheukam, A., and Tembine, H. Meanfield-type games in engineering. arXiv preprint arXiv:1605.03281 , 2016.
- Du, S., Kakade, S., Lee, J., Lovett, S., Mahajan, G., Sun, W., and Wang, R. Bilinear classes: A structural framework for provable generalization in rl. In International Conference on Machine Learning , pp. 2826-2836. PMLR, 2021.
- Elie, R., Perolat, J., Lauri` ere, M., Geist, M., and Pietquin, O. On the convergence of model free learning in mean field games. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 34, pp. 7143-7150, 2020.
- Foster, D., Foster, D. J., Golowich, N., and Rakhlin, A. On the complexity of multi-agent decision making: From learning in games to partial monitoring. In The Thirty Sixth Annual Conference on Learning Theory , pp. 26782792. PMLR, 2023.
- Foster, D. J., Kakade, S. M., Qian, J., and Rakhlin, A. The statistical complexity of interactive decision making. arXiv preprint arXiv:2112.13487 , 2021.
- Ghosh, A. and Aggarwal, V. Model free reinforcement learning algorithm for stationary mean field equilibrium for multiple types of agents. arXiv preprint arXiv:2012.15377 , 2020.
- Gomes, D. A. and Pimentel, E. A. Economic models and mean-field games theory.
- Guo, X., Hu, A., Xu, R., and Zhang, J. Learning meanfield games. Advances in Neural Information Processing Systems , 32, 2019.
- Huang, B., Lee, J. D., Wang, Z., and Yang, Z. Towards general function approximation in zero-sum markov games. arXiv preprint arXiv:2107.14702 , 2021.
- Huang, J., Chen, J., Zhao, L., Qin, T., Jiang, N., and Liu, T.-Y. Towards deployment-efficient reinforcement learning: Lower bound and optimality. arXiv preprint arXiv:2202.06450 , 2022.

- Huang, J., Yardim, B., and He, N. On the statistical efficiency of mean-field reinforcement learning with general function approximation. In International Conference on Artificial Intelligence and Statistics , pp. 289-297. PMLR, 2024.
- Huang, M., Malham´ e, R. P., and Caines, P. E. Large population stochastic dynamic games: closed-loop mckeanvlasov systems and the nash certainty equivalence principle. 2006.
- Jiang, N., Krishnamurthy, A., Agarwal, A., Langford, J., and Schapire, R. E. Contextual decision processes with low bellman rank are pac-learnable. In International Conference on Machine Learning , pp. 1704-1713. PMLR, 2017.
- Jin, C., Allen-Zhu, Z., Bubeck, S., and Jordan, M. I. Is q-learning provably efficient? Advances in neural information processing systems , 31, 2018.
- Jin, C., Yang, Z., Wang, Z., and Jordan, M. I. Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory , pp. 2137-2143. PMLR, 2020.
- Jin, C., Liu, Q., and Miryoosefi, S. Bellman eluder dimension: New rich classes of rl problems, and sampleefficient algorithms. Advances in neural information processing systems , 34:13406-13418, 2021a.
- Jin, C., Liu, Q., Wang, Y., and Yu, T. V-learning-a simple, efficient, decentralized algorithm for multiagent rl. arXiv preprint arXiv:2110.14555 , 2021b.
- Lasry, J.-M. and Lions, P.-L. Mean field games. Japanese journal of mathematics , 2(1):229-260, 2007.
- Lauri` ere, M., Perrin, S., Geist, M., and Pietquin, O. Learning mean field games: A survey. arXiv preprint arXiv:2205.12944 , 2022.
- Levy, O., Cassel, A., Cohen, A., and Mansour, Y. Eluderbased regret for stochastic contextual mdps, 2022.
- Mao, W., Qiu, H., Wang, C., Franke, H., Kalbarczyk, Z., Iyer, R., and Basar, T. A mean-field game approach to cloud resource management with function approximation. In Advances in Neural Information Processing Systems , 2022.
- Mishra, R. K., Vasal, D., and Vishwanath, S. Model-free reinforcement learning for non-stationary mean field games. In 2020 59th IEEE Conference on Decision and Control (CDC) , pp. 1032-1037. IEEE, 2020.
- Modi, A., Chen, J., Krishnamurthy, A., Jiang, N., and Agarwal, A. Model-free representation learning
- and exploration in low-rank mdps. arXiv preprint arXiv:2102.07035 , 2021.
- Moon, J. and Bas ¸ar, T. Linear quadratic mean field stackelberg differential games. Automatica , 97:200-213, 2018.
- Ni, C., Song, Y., Zhang, X., Jin, C., and Wang, M. Representation learning for general-sum low-rank markov games. arXiv preprint arXiv:2210.16976 , 2022.
- Osband, I. and Van Roy, B. Model-based reinforcement learning and the eluder dimension. Advances in Neural Information Processing Systems , 27, 2014.
- Pasztor, B., Bogunovic, I., and Krause, A. Efficient modelbased multi-agent mean-field reinforcement learning. arXiv preprint arXiv:2107.04050 , 2021.
- Perolat, J., Perrin, S., Elie, R., Lauri` ere, M., Piliouras, G., Geist, M., Tuyls, K., and Pietquin, O. Scaling up mean field games with online mirror descent. arXiv preprint arXiv:2103.00623 , 2021.
- Perrin, S., P´ erolat, J., Lauri` ere, M., Geist, M., Elie, R., and Pietquin, O. Fictitious play for mean field games: Continuous time analysis and applications. Advances in Neural Information Processing Systems , 33:1319913213, 2020.
- Russo, D. and Van Roy, B. Eluder dimension and the sample complexity of optimistic exploration. Advances in Neural Information Processing Systems , 26, 2013.
- Subramanian, J. and Mahajan, A. Reinforcement learning in stationary mean-field games. In Proceedings of the 18th International Conference on Autonomous Agents and Multi Agent Systems , pp. 251-259, 2019.
- Subramanian, S. G., Poupart, P., Taylor, M. E., and Hegde, N. Multi type mean field reinforcement learning. arXiv preprint arXiv:2002.02513 , 2020.
- Sun, W., Jiang, N., Krishnamurthy, A., Agarwal, A., and Langford, J. Model-based rl in contextual decision processes: Pac bounds and exponential improvements over model-free approaches. In Conference on learning theory , pp. 2898-2933. PMLR, 2019.
- Uehara, M., Zhang, X., and Sun, W. Representation learning for online and offline rl in low-rank mdps. arXiv preprint arXiv:2110.04652 , 2021.
- uz Zaman, M. A., Miehling, E., and Bas ¸ar, T. Reinforcement learning for non-stationary discrete-time linear-quadratic mean-field games in multiple populations. Dynamic Games and Applications , 13(1):118-164, 2023.

- Vasal, D. and Berry, R. Master equation for discrete-time stackelberg mean field games with a single leader. In 2022 IEEE 61st Conference on Decision and Control (CDC) , pp. 5529-5535. IEEE, 2022.
- Wang, Y., Liu, Q., Bai, Y., and Jin, C. Breaking the curse of multiagency: Provably efficient decentralized multiagent rl with function approximation. arXiv preprint arXiv:2302.06606 , 2023.
- Xie, Q., Yang, Z., Wang, Z., and Minca, A. Learning while playing in mean-field games: Convergence and optimality. In International Conference on Machine Learning , pp. 11436-11447. PMLR, 2021.
- Xie, T., Foster, D. J., Bai, Y ., Jiang, N., and Kakade, S. M. The role of coverage in online reinforcement learning. arXiv preprint arXiv:2210.04157 , 2022.
- Yardim, B., Cayci, S., Geist, M., and He, N. Policy mirror ascent for efficient and independent learning in mean field games. arXiv preprint arXiv:2212.14449 , 2022.
- Yardim, B., Goldman, A., and He, N. When is mean-field reinforcement learning tractable and relevant? arXiv preprint arXiv:2402.05757 , 2024.
- Zanette, A., Lazaric, A., Kochenderfer, M., and Brunskill, E. Learning near optimal policies with low inherent bellman error. In International Conference on Machine Learning , pp. 10978-10989. PMLR, 2020.
- Zhang, F., Tan, V. Y., Wang, Z., and Yang, Z. Learning regularized monotone graphon mean-field games. arXiv preprint arXiv:2310.08089 , 2023.
- Zhang, K., Yang, Z., and Basar, T. Policy optimization provably converges to nash equilibria in zero-sum linear quadratic games. Advances in Neural Information Processing Systems , 32, 2019.
- Zhang, K., Yang, Z., and Bas ¸ar, T. Multi-agent reinforcement learning: A selective overview of theories and algorithms. Handbook of reinforcement learning and control , pp. 321-384, 2021.
- Zhong, H., Xiong, W., Zheng, S., Wang, L., Wang, Z., Yang, Z., and Zhang, T. A posterior sampling framework for interactive decision making. arXiv preprint arXiv:2211.01962 , 2022.

## Outline of the Appendix

- Appx. A: Frequently used notations.
- Appx. B: Additional related works.
- Appx. C: Informal Discussions about how to extend our results to unknown reward function setting.
- Appx. D: Missing details and proofs related to P-MBED.
- Appx. E: Missing details related to Single/Multi-Type Policy Aware Models (PAM).
- Appx. F: Proofs for lemma and theorems related to learning MFGs in Sec. 4.
- Appx. G: Proofs for lemma and theorems related to learning MultiType MFGs in Sec. 5.
- Appx. H: Introduction to Multi-Type Symmetric Anonymous Games (MT-SAGs) and approximation error between MT-MFGs and MT-SAGs.
- Appx. I: Some basic lemma useful in our proofs.
- Appx. J: Experiment details and results.

## A. Frequently Used Notations

| Notation                                                                                                                                                                                                                                                         | Explanation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| M M π Q M ,V M ,J M dim PE ( M , ε ) dim II PE ( M , ε ) d ( · , ·&#124; π ) B ε 0 π ( M, M ) M ε 0 Ctr ( π, M ) d ∞ , 1 ( π,π ′ ) M M π Q M ,V M ,J M dim MTPE ( M , ε ) dim II MTPE ( M , ε ) ¨ M/ ¨ M ¨ M / ¨ M , ¨ V ¨ M , ¨ J ¨ M / ¨ Q ¨ M , ¨ V ¨ M , ¨ J | Mean-Field MDP Model class for (single-type) Mean-Field MDP Non-stationary policy for (single-type) Mean-Field MDP Value functions for (single-type) MF-MDP Type- I Partial-MBED, Def. 3.3 Type- II Partial-MBED, Def. D.1 Conditional distance given a reference policy π ε 0 -neighborhood of M in M conditioning on π The 'Central Model' Policy distance Multi-Type Mean-Field MDP Model class for multi-type Mean-Field MDP Non-stationary policy for multi-type Mean-Field MDP Value functions for multi-type MF-MDP Type- I Multi-Type Partial-MBED, Def. D.10 Type- II Multi-Type Partial-MBED, Def. D.10 Policy-aware model (single-type/multi-type) Model class for the policy-aware model (single-type/multi-type) Value functions for the policy-aware model (single-type/multi-type) |

## B. Additional Related Works

Single-Agent/Multi-Agent RL with General Function Approximation For the single-agent RL with function approximation setting, besides the literature we mentioned in the main text, there are multiple other insightful works (Ayoub et al., 2020; Chen et al., 2022a; Huang et al., 2022; Modi et al., 2021; Uehara et al., 2021; Xie et al., 2022; Zanette et al., 2020; Zhong et al., 2022).

As for the multi-agent setting, sample complexity of Markov Games has been extensively studied in both tabular (Bai et al., 2020; Chen et al., 2022b; Jin et al., 2021b; Zhang et al., 2019; 2021) and function approximation setting (Cui et al., 2023; Foster et al., 2023; Huang et al., 2021; Ni et al., 2022; Wang et al., 2023). These papers study a general MARL setting with individually distinct agents, which is quite different from our MFG or MT-MFG. Besides, many of them study the decentralized training setting, which requires much less communication cost than our centralized setting. However, because of the difficulty in learning NE in general Markov Games setting, most of them focus on the convergence to weaker notions of equilibrium instead, e.g. the Correlated Equilibrium or the Coarse Correlated Equilibria, and those results in function approximation setting (Cui et al., 2023; Wang et al., 2023) may still depend on the number of agents, although in polynomial. In contrast, although we specify in mean-field approximation setting, we can have more ambitious goals on solving Nash Equilibrium, and our sample complexity bounds are totally independent w.r.t. the number of agents. Moreover, we also reveal some cases when learning (MT-)MFG can be as sample-efficient as single-agent RL by investigating the Partial Model-Based Eluder Dimension.

## C. Extension to the Setting when the Reward Function is Unknown

We remark that our current results extend to the unknown reward setting. Below we elucidate the key modifications needed for this extension.

Firstly, for the problem setup, we instead assume a model class M available, where each element M := ( r M , P M ) ∈ M corresponds to a (reward, transition) tuple. The definition of the P-MBED can be amended by incorporating both reward and transition differences in Def. 3.3.

Secondly, for the algorithm design:

- For Algorithm 1, we redefine the model distance the definition d ˜ π ( M, ˜ M | π ) (introduced at the beginning of Sec. 4.1) to include the expectation of distances in both reward and transition functions:

<!-- formula-not-decoded -->

The definition of ε 0 -neighborhood and 'Central Model' will adjust correspondingly.

- For Algorithm 2: we should augment the reward difference into the right-hand side of Line 4, integrate reward into the dataset in Lines 9 and 11, and include the likelihood of reward functions in Line 13.
- For Algorithm 3, the construction of bridge policy will follow the new definition of model distance d ˜ π ( M, ˜ M | π ) .

Finally, for the analysis, based on the modified algorithms, under realizability assumption, we can extend Lemma D.7 and prove that the accumulative estimation errors of reward and transition are controlled by P-MBED. The current analysis can be seamlessly generalized to establish sample complexity upper bounds depending on the P-MBED of reward and transition function classes.

## D. Missing Details about Partial Model-Based Eluder Dimension

## D.1. Alternative Notions of Partial MBED in MFGs

In this section, we introduce a choice of ν different from the one in Def. 3.2, which also leads to a valid P-MBED. In the following, we introduce another choice of ν , which results in a different P-MBED. We will call it 'Type II ' P-MBED to distinguish the one in Def. 3.3.

Definition D.1 (Type II Partial MBED) . Given a model class M , define the mapping ν π M ∗ ,h : M→ ∆( S h ) such that ∀ M ∈ M , ν π M ∗ ,h ( M ) := µ π M ∗ ,h , then the type II P-MBED of M is defined by:

<!-- formula-not-decoded -->

We want to highlight here that each of the two types P-MBED has advantages over the other. As we will see in the Thm. F.7, if we use dim II PE to derive the sample complexity upper bound, we have to suffer the exponential term of (1 + L T ) H . On the other hand, in the following proposition, we can see dim II PE is directly comparable with MBED (Huang et al., 2024) ( α = 1 case), while we can not have the similar guarantee for dim PE .

Proposition D.2 (Low MBED ⊂ Low Type II P-MBED) . dim II PE ( M , ε ) ≤ dim E ( M , ε ) .

## D.2. Proofs Related to P-MBED in MFGs Setting

Proposition 3.4. (Tabular Setting) For any M and ε &gt; 0 , dim PE ( M , ε ) ≤ |S||A| , while there exists a concrete example of M such that dim E ( M , ε ) = Ω(exp( |S| )) .

Proof. When the density is fixed, any MF-MDP reduces to a single-agent MDP. For any single-agent MDP, there are at most |S||A| different ( s h , a h ) pairs, for any h . Therefore, the P-MBED can be upper bounded by |S||A| .

̸

In contrast, for MBED in (Huang et al., 2024), we consider the model class constructed in Thm. F.9. Consider the sequence { ( s 1 h , a 1 h , µ i ) } i ∈ [ n ] with µ i ∈ U ζ = ⌊ L T 5 ε ⌋ for all i ∈ [ n ] , but µ i = µ j if i = j . For any i ∈ [ n -1] , there exists two models P µ i and P µ i +1 , such that,

̸

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, we can show the type II P-MBED in tabular setting can also be upper bounded by |S||A| , because there are at most |S||A| different state-action tuples.

Proposition D.3 (Type II P-MBED in the Tabular Setting) . dim II PE ( M , ε ) ≤ |S||A| .

Next, we study the linear setting. Given a mapping f : S → R d , we use Rank([ f ( x )] x ∈X ) to denote the rank of matrix concatenated by [ f ( x )] x ∈X ∈ R |X|× d .

Proposition D.4 (Linear Setting; Formal version of Prop. 3.5) . Consider the Low-Rank MF-MDP with known feature ϕ : S × A × ∆( S ) → R d satisfying ∥ ϕ ∥ ≤ C ϕ , and unknown next state feature ψ : S → R d . Given a next state feature function class Ψ satisfying ∀ ψ ∈ Ψ , ∀ s ′ ∈ S , ∀ g : S → {1 , 1 } , ∥ ∑ s ′ ψ ( s ′ ) g ( s ′ ) ∥ 2 ≤ C Ψ , consider the following model class:

<!-- formula-not-decoded -->

we have dim II PE ( M Ψ , ε ) = ˜ O (max π,h Rank([ ϕ h ( s h , a h , µ π M ∗ ,h )] s h ∈S ,a h ∈A )) .

Moreover, if ϕ ( s, a, µ ) has decomposition: ϕ ( s, a, µ ) ⊤ = ϕ ( s, a ) ⊤ G ( µ ) with ϕ ( · , · ) ∈ R ˜ d and G ( · ) ∈ R ˜ d × d , we have dim PE ( M Ψ , ε ) = ˜ O ( ˜ d ) and dim II PE ( M Ψ , ε ) = ˜ O (min { ˜ d, d } ) .

but Remark D.5. As we can see, the P-MBED is related to the 'activated dimension' of features after partially fixing the density, which can be much lower than its MBED ≈ d . Moreover, when the feature is decomposable, the dimension of state-action feature will also serve as an upper bound.

̸

Proof. Proof for TypeII P-MBED (Def. D.1) In the following, we first consider a fixed policy π and h . To simplify the notation, we denote Φ := [ ϕ ( s h , a h , µ π M ∗ ,h )] s ∈S ,a ∈A ∈ R d ×| S || A | to be the matrix concatenated by vectors ϕ ( · , · , µ π M ∗ ,h ) , and denote d active := Rank(Φ) to be its rank. We use U := [ u 1 , u 2 , ..., u d active ] ∈ R d × d active to denote a normalized orthogonal basis in Span (Φ) = Span ( U ) satisfying ∥ u i ∥ 2 = 1 for all i ∈ [ d active ] and u ⊤ i u j = 0 for any i = j . Easy to verify that for any s h , a h , the following equation

<!-- formula-not-decoded -->

has a solution satisfying:

<!-- formula-not-decoded -->

Given a fixed policy π , h ∈ [ H ] , suppose ( s 1 h , a 1 h ) , ..., ( s n h , a n h ) is a partially ε -independent sequence w.r.t. M Ψ and ν π M ∗ ,h defined in D.1. Then for each i ∈ [ n ] , there should exists ψ i , ˜ ψ i ∈ Ψ , such that:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where we define:

<!-- formula-not-decoded -->

For simplicity, we use v ψ, ˜ ψ ( s h , a h , µ ) := U ⊤ ∑ s ′ ( ψ ( s ′ ) -˜ ψ ( s ′ )) g ψ, ˜ ψ ( s h , a h , µ, s ′ ) as a shortnote. Therefore, for each i ,

<!-- formula-not-decoded -->

By choosing λ = ε 2 / 4 C 2 Ψ , we have:

<!-- formula-not-decoded -->

On the one hand,

<!-- formula-not-decoded -->

Therefore,

and

<!-- formula-not-decoded -->

where we define:

<!-- formula-not-decoded -->

The rest analysis is similiar to the non-decomposable setting above. As a result, we can show:

This holds for any π , which finishes the proof.

<!-- formula-not-decoded -->

Remark D.6. Following similar analyses as Prop. D.4 and Prop. B.6 and Prop. B.7 in (Huang et al., 2024), we can compute the P-MBED for kernel MF-MDP and generalized linear function classes. All we need to do is to replace d eff or d in (Huang et al., 2024) with the corresponding dimensions conditioning on the adversarial densities.

Lemma D.7. Under Def. 3.3 and Def. D.1, consider a fixed π and an arbitrary h ∈ [ H ] , Suppose we have a sequence { P M k ,h } K k =1 ∈ F and { ( s k h , a k h ) } K k =1 ⊂ S × A ,

<!-- formula-not-decoded -->

which implies n = O ( d active log(1 + d active C ϕ C Ψ ε )) .

Finally, if we take the maximum over all policy π , we have

<!-- formula-not-decoded -->

When ϕ ( s h , a h , µ ) can be decomposed to ϕ ( s h , a h ) ⊤ G ( µ ) for some ϕ ( s h , a h ) ∈ R ˜ d , easy to verify that for any π , the corresponding d active ≤ ˜ d . By combining with Prop. D.2, we can conclude dim II PE ( M , ε ) = ˜ O (min { d, ˜ d } ) .

Proofs for P-MBED (Def. 3.3) As for the first type of P-MBED, we only study the decomposable feature setting. Given a fixed policy π , h ∈ [ H ] , suppose ( s 1 h , a 1 h ) , ..., ( s n h , a n h ) is a partially ε -independent sequence w.r.t. M Ψ and the mapping ν π h defined in Def. 3.3, then for each i ∈ [ n ] , there should exists ψ i , ˜ ψ i ∈ Ψ , such that:

<!-- formula-not-decoded -->

- if for all k ∈ [ K ] , ∑ k -1 i =1 ∥ P M k ,h ( ·| s i h , a i h , µ π M k ,h ) -P M ∗ ,h ( ·| s i h , a i h , µ π M ∗ ,h ) ∥ 2 1 ≤ β , then for any ε &gt; 0 , we have ∑ K k =1 ∥ P M k ,h ( ·| s k h , a k h , µ π M k ,h ) -P M ∗ ,h ( ·| s k h , a k h , µ π M ∗ ,h ) ∥ 1 = O ( √ βK dim PE ( M , ε ) + Kε ) ,
- if for all k ∈ [ K ] , ∑ k -1 i =1 ∥ P M k ,h ( ·| s i h , a i h , µ π M ∗ ,h ) -P M ∗ ,h ( ·| s i h , a i h , µ π M ∗ ,h ) ∥ 2 1 ≤ β , then for any ε &gt; 0 , we have ∑ K k =1 ∥ P M k ,h ( ·| s k h , a k h , µ π M ∗ ,h ) -P M ∗ ,h ( ·| s k h , a k h , µ π M ∗ ,h ) ∥ 1 = O ( √ βK dim II PE ( M , ε ) + Kε ) .

Proof. Let's consider a single-agent model-class P ⊂ { P | P : S×A → ∆( S ) } . Wefirst introduce the notion of independent state-action sequences given a single-agent model class.

Definition D.8 ( ε -Independent state action pairs) . Given single-agent model P and a data sequence { ( s i h , a i h ) } n i =1 ⊂ S h ×A h , we say ( s h , a h ) is ε -independent of { ( s i h , a i h ) } n i =1 w.r.t. P if there exists P , ˜ P ∈ M such that ∑ n i =1 ∥ P h ( ·| s i h , a i h ) -˜ P h ( ·| s i h , a i h ) ∥ 2 1 ≤ ε 2 but ∥ P M,h ( ·| s h , a h ) -˜ P h ( ·| s h , a h ) ∥ 1 &gt; ε . We call { ( s i h , a i h ) } n i =1 an ε -independent sequence w.r.t. P (at step h ) if for any i ∈ [ n ] , ( s i h , a i h , µ i h ) is ε -independent w.r.t. { ( s t h , a t h , µ t h ) } i -1 t =1 .

We use dim E ( P , ε ) to denote the maximal length of ε -independent sequence { ( s i , a i ) } i ∈ [ n ] for single-agent model class P . Since single-agent RL is a special case of MF-MDP where the transition is independent w.r.t. density. As implied by Lem. 4.4 in (Huang et al., 2024) when α = 1 , suppose there is a sequence { P k } k ∈ [ K ] ⊂ P and a sequence of states and actions { ( s k h , a k h ) } k ∈ K , such that:

<!-- formula-not-decoded -->

where P ∗ ∈ P is some fixed function, then for any ε &gt; 0 ,

<!-- formula-not-decoded -->

By choosing P := { P M,h | P M,h ( ·|· , · ) ← P M,h ( ·|· , · , µ π M,h ) , M ∈ M} with P ∗ := P M ∗ ,h ( ·|· , · , µ π M ∗ ,h ) and combining the definition in Def. 3.3, we can finish the proof of the first statement.

By choosing P := { P M,h | P M,h ( ·|· , · ) ← P M,h ( ·|· , · , µ π M ∗ ,h ) , M ∈ M} with P ∗ := P M ∗ ,h ( ·|· , · , µ π M ∗ ,h ) and combining the definition in Def. D.1, we can finish the proof of the second statement. □

## D.3. Partial MBED for Model Classes in Multi-Type MFGs Setting

Definition D.9 (Partial ε -Independence for Multi-Type Mean-Field Model Classes) . Given a multi-type model class M , consider a w ∈ [ W ] and a mapping ν w h : M → ∆( S 1 h ) × ... × ∆( S W h ) , and a sequence of data { ( s w,i h , a w,i h ) } n i =1 ⊂ S w h ×A w h , we say ( s w h , a w h ) is partially ε -independent on { ( s w,i h , a w,i h ) } n i =1 w.r.t. M and ν w h at step h , if there exists M , ˜ M ∈ M , s.t. ∑ n i =1 ∥ P w M ,h ( ·| s w,i h , a w,i h , ν w h ( M )) -P w ˜ M ,h ( ·| s w,i h , a w,i h , ν w h ( ˜ M )) ∥ 2 1 ≤ ε 2 but ∥ P w M ,h ( ·| s w h , a w h , ν w h ( M )) -P w ˜ M ,h ( ·| s w h , a w h , ν w h ( ˜ M )) ∥ 1 &gt; ε .

Besides, we call { ( s w,i h , a w,i h ) } n i =1 is a partially ε -independent sequence w.r.t. M and ν w h if for any i ∈ [ n ] , ( s w,i h , a w,i h ) is partially ε -independent on { ( s w,t h , a w,t h ) } i -1 t =1 . In the following, we use dim E | ν w h ( M , ε ) to denote the length of the longest partially ε -independent sequence w.r.t. M and ν w h for type w (at step h ).

Definition D.10. Given a model class M and an arbitrary w , we define the mapping ν w, π h : M → ∆( S 1 ) × ... ∆( S W ) s.t. ν w, π h ( M ) = µ π M ,h , and the mapping ν w, π M ∗ ,h : M → ∆( S 1 ) × ... ∆( S W ) s.t. ν w, π M ∗ ,h ( M ) = µ π M ∗ ,h . Then, the Multi-Type P-MBEDs are defined by:

- Type I MT-P-MBED: dim MTPE ( M , ε ) := ∑ w ∈ [ W ] max h ∈ [ H ] max π ∈ Π dim E | ν w, π h ( M , ε ) ;
- Type II MT-P-MBED: dim II MTPE ( M , ε ) := ∑ w ∈ [ W ] max h ∈ [ H ] max π ∈ Π dim E | ν w, π M ∗ ,h ( M , ε ) .

Proposition D.11 (Tabular Multi-Type MF-MDP) .

<!-- formula-not-decoded -->

Proof. By definition, for each w , for any fixed h and π , the longest partially independent state-action sequence would have length |S w ||A w | . □

Proposition D.12 (Linear Multi-Type MF-MDP) . Consider the Low-Rank Multi-Type MF-MDP with known feature ϕ w : S w ×A w × ∆( S 1 ) × ... ∆( S W ) → R d w satisfying ∥ ϕ w ∥ ≤ C ϕ for any w ∈ [ W ] , and unknown next state feature ψ w : S w → R d . Given a next state feature function class Ψ 1 , ..., Ψ W satisfying ∀ ψ w ∈ Ψ w , ∀ s ′ w ∈ S w , ∀ g : S → {1 , 1 } , ∥ ∑ s ′ w ψ ( s ′ w ) g ( s ′ w ) ∥ 2 ≤ C Ψ , define the model class:

<!-- formula-not-decoded -->

then, we have:

<!-- formula-not-decoded -->

where d w active , π ,h := Rank([ ϕ w h ( s w h , a w h , µ π M ∗ ,h )] s w h ∈S w ,a w h ∈A w ) .

Moreover, if ϕ w ( s w , a w , µ ) is decomposable, i.e. for any w ∈ [ W ] , ϕ w ( s w , a w , µ ) ⊤ = ϕ w ( s w , a w ) ⊤ G w ( µ ) with ϕ w ( · , · ) ∈ R ˜ d w and G w ( · ) ∈ R ˜ d w × d w , we have dim MTPE ( M Ψ w , ε ) = ˜ O ( ∑ w ∈ [ W ] ˜ d w ) and dim II MTPE ( M Ψ w , ε ) = ˜ O ( ∑ w ∈ [ W ] min { ˜ d w , d w } ) .

Proof. The proof is a direct generalization of Prop. D.4 by applying the same techniques in the proof of Prop. D.4 for each type w ∈ [ W ] , so we omit it here. □

## D.4. Partial MBED in Constrained Policy Spaces

Next, we define the Constrained Partial MBED extended from Def. 3.3, where the main difference is that we constrain the set of adversarial policies.

Definition D.13 (Constrained Partial MBED in MFRL) . Given a (single-type) Mean-Field model class M , and M ∗ denotes the true model, we consider the same ν π h and ν π M ∗ ,h function defined in Def. 3.3 and Def. D.1, respectively. Then, the constrained P-MBEDs are defined by:

- Type I P-MBED: dim I CPE | Π † ( M , ε ) := max h ∈ [ H ] max π ∈ Π † dim E | ν π h ( M , ε ) ;
- Type II P-MBED: dim II CPE | Π † ( M , ε ) := max h ∈ [ H ] max π ∈ Π † dim E | ν π M ∗ ,h ( M , ε ) .

Comparing with P-MBED, the main difference is that in constrained P-MBED the adversarial policies are only chosen from the constrained policy set. Recall the definition of Π † in Sec. 5. Given a M and a model class M MFG converted from M according to Appx. G.2, we have the following relationship between the P-MBED of M MFG constrained on Π † and P-MBED of M .

Proposition D.14. Given a model class M and its corresponding lifted MFG class M MFG:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let's consider a fixed policy π ∈ Π † . Note that, π corresponds to a π := { π w } w ∈ [ W ] with π w : S w → ∆( A w ) and π w ( a w h | s w h ) = π ( a w h ◦ w | s w h ◦ w ) . Given any ε &gt; 0 , and h ∈ [ H ] , suppose we have a partial ε -independent sequence w.r.t. the mapping ν π h (or ν π M ∗ ,h ), denoted as { ( s w i ,i h ◦ w i , a w i ,i h ◦ w i ) } i ∈ [ n ] . We divide this sequence according to its group w i , which we denote as {{ s w,i w h ◦ w,a w,i w h ◦ w } i w ∈ [ n w ] } w ∈ [ W ] with ∑ w n w = n . By construction of M MFG, for any w ∈ [ W ] ,

{ s w,i w h , a w,i w h } i w ∈ [ n w ] is a partial ε -independent sequence w.r.t. function class M w and the mapping ν w, π h (or ν w, π M ∗ ,h ), which is upper bounded by the Multi-Type P-MBED of model class M w .

We finish the proof of Eq. (2) by maximizing over π ∈ Π † . □

As directly implied by Prop. D.14, Prop. D.11 and Prop. D.12, we can upper bound the constrained P-MBED in some special cases.

## E. Details about Single-Type/Multi-Type Policy Aware Models

## E.1. (Single-Type) Policy-Aware Model

Concretely, Policy-Aware Model (PAM) is specified by a tuple ¨ M := {S , A , H, µ 1 , ¨ P ¨ M , ¨ r ¨ M } , where S , A , H, µ 1 are the state space, action space, horizon length, initial state distribution which are the same as the normal MF-MDP setting; ¨ P ¨ M := { ¨ P ¨ M, 1 , ..., ¨ P ¨ M,H } is the transition function with ¨ P ¨ M,h : S h ×A h × Π → ∆( S h +1 ) , and ¨ r ¨ M := { ¨ r ¨ M, 1 , ..., ¨ r ¨ M,H } is the reward function 5 satisfying ¨ r ¨ M,h : S h ×A h × Π → [0 , 1 /H ] , where recall Π denotes the collection of all Markov policies. Given any reference policy π , we define the value function ¨ Q ˜ π ¨ M,h : S h ×A h × Π → R and ¨ V ˜ π ¨ M,h : S h × Π → R regarding ˜ π in the following way:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, we will denote E ˜ π, ¨ M ( π ) [ · ] to be the expectation taken over trajectories sampled by executing ˜ π in the model ¨ M , such that the transition and reward functions are fixed by π . Again, we will call π as the 'reference policy'.

By definition, once the reference policy π is determined, the transition/reward functions reduced to single-agent transition/reward functions, and the value functions are defined in the same way as single-agent RL setting. Besides, we define the total return of ˜ π conditioning on the reference policy π as:

<!-- formula-not-decoded -->

and define ∆ ¨ M ( ˜ π, π ) := ¨ J ¨ M ( ˜ π, π ) -¨ J ¨ M ( π, π ) . Similar to MF-MDP, we define the NE in ¨ M . Intuitively, the NE in ¨ M is the policy π NE ¨ M that agents do not tend to deviate when π NE ¨ M is chosen to be the reference policy.

Definition E.1 (Nash Equilibrium in ¨ M ) . Given a model ¨ M , we call π NE ¨ M is a Nash Equilibrium (NE) of ¨ M , if

<!-- formula-not-decoded -->

Besides, we call ̂ π NE ¨ M is an ε -approximate NE of ¨ M , if

<!-- formula-not-decoded -->

Similar to the conditional distance d ( M, ˜ M | π ) defined in Sec. 4, we can define the conditional distance for PAM.

<!-- formula-not-decoded -->

Given a PAM model class ¨ M and a model ¨ M ∈ ¨ M , for any reference policy π , we define the ε 0 -neighborhood of ¨ M in ¨ M to be

<!-- formula-not-decoded -->

Besides, we define the Central Model ¨ M ε 0 Ctr ( π ; ¨ M ) in ¨ M regarding π to be the model with the largest neighborhood set:

<!-- formula-not-decoded -->

5 Here we specify the model in the subscription, because for those PAM converted from MF-MDPs, even if they share the reward function in mean-field systems, the reward functions in PAM version can be different because of the difference in transition functions.

## E.1.1. EXISTENCE OF NASH EQUILIBRIUM IN ¨ M

Next, we investigate the existence of NE in ¨ M . Recall the definition

<!-- formula-not-decoded -->

Theorem E.2. [Existence of Nash Equilibrium in PAM] Given a PAM ¨ M with discrete state and action spaces, such that, for any h ∈ [ H ] , s h +1 ∈ S h +1 , s h ∈ S h , a h ∈ A h , both ¨ P ¨ M,h ( s h +1 | s h , a h , π ) and ¨ r ¨ M,h ( s h , a h , π ) are continuous at π w.r.t. distance d ∞ , 1 , then ¨ M has at least one NE satisfying Def. E.1.

Proof. In Prop. E.2, we establish the existence of NE in Multi-Type PAM, and the proof for this theorem is a special case when W = 1 . □

As a direct result of Thm. E.2 and Lem. E.4, we have the following corollary.

Corollary E.3. Given a MF-MDP model M satisfying Assump. B, the PAM model ¨ M converted from M according to the rules in Eq. (1) has at least one NE.

## E.1.2. USEFUL LEMMA RELATED TO THE PAM CONVERTED FROM MF-MDP

LemmaE.4. [Lipschitz Continuity of PAM] Given an MF-MDP M satisfying the Lipschitz continuity condition in Assump. B, consider the PAM ¨ M converted from M according to Eq. (1) , we have ¨ M is also Lipschitz continuous that, ∀ h ∈ [ H ] and any s h ∈ S , a h ∈ A ,

<!-- formula-not-decoded -->

Proof. This lemma is a special case of Lem. E.13 when W = 1 .

## E.2. Multi-Type Policy-Aware Model

In this section, we introduce Multi-Type Policy-Aware Model (MT-PAM) extended from PAM. To distinguish with MT-MFG, we use ¨ M as notation.

MT-PAM is specified by ¨ M := { ( µ w 1 , S w , A w , H, ¨ P w ¨ M , ¨ r w ¨ M ) w ∈ [ W ] } 6 , where S w , A w , H, µ w 1 are defined the same as the Multi-Type MF-MDP setting; ¨ P w ¨ M := { ¨ P w ¨ M ,h } h ∈ [ H ] is the transition function with ¨ P w ¨ M ,h : S w h ×A w h × Π → ∆( S h +1 ) and ¨ r ¨ M ,h : S w h ×A w h × Π → [0 , 1 H ] , where recall Π denotes the set of all Markov policies π := { π w } w ∈ [ W ] with π w ∈ Π w .

Given a reference policy π := { π w } w ∈ [ W ] ∈ Π , for any ˜ π := { ˜ π w } w ∈ [ W ] ∈ Π , we define the value function for type w ¨ Q w, ( · ) ¨ M ,h : S w h ×A w h × Π → R and ¨ V w, ( · ) ¨ M ,h : S w h × Π → R in the following way:

<!-- formula-not-decoded -->

Similarly, we will denote E ˜ π , ¨ M ( π ) [ · ] to be the expectation taken over trajectories sampled by executing ˜ π in the model ¨ M , such that the transition and reward functions are fixed by π . Again, we will call π as the 'reference policy'.

6 Here we specify the model in the subscription of the reward function, which will avoid confusion when we consider the PAMs converted from (Multi-Type) MF-MDPs

□

We denote ¨ J w ¨ M ( ˜ π ; π ) := E s w 1 ∼ µ w 1 [ V w, ˜ π M , 1 ( s w 1 ; π )] to be the expected return of agents in type w in model ¨ M by executing ˜ π given π as the reference policy.

Definition E.5 (Nash Equilibrium in Multi-Type PAM) . The Nash Equilibrium policy in Multi-Type PAM is defined to be the policy π NE := { π w, NE } w ∈ [ W ] satisfying:

<!-- formula-not-decoded -->

Note that ¨ J w ¨ M ( ˜ π ; π ) actually only depends on π and ˜ π w .

## E.2.1. EXISTENCE OF NASH EQUILIBRIUM IN MT-PAM

We first investigate a stronger notion of NE, which we call the 'strict NE'.

Definition E.6 (Strict NE) . Given a MT-PAM ¨ M with transitions and rewards { ( ¨ P w ¨ M , ¨ r w ¨ M ) } w ∈ [ W ] , the policy π NE is a strict NE of ¨ M if and only if the following holds:

<!-- formula-not-decoded -->

Note that this is a stronger notion than the NE defined in Def. E.5, i.e. a strict NE is always a NE. In the following, we will focus on the existence of strict NE.

Lemma E.7 (Strict NE as Fixed Point) . Given a MT-PAM ¨ M with transitions and rewards { ( ¨ P w ¨ M , ¨ r w ¨ M ) } w ∈ [ W ] , the policy π SNE is a strict NE of ¨ M if and only if the following holds:

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Proof. First of all, suppose π is the NE of ¨ M according to Def. E.5, by the policy improvement theorem in single-agent RL, we have:

<!-- formula-not-decoded -->

which also implies

<!-- formula-not-decoded -->

Therefore, if π is the strict NE of ¨ M , we have Γ SNE ¨ M ( π ) = π .

On the other hand, if Γ SNE ¨ M ( π ) = π , it implies:

<!-- formula-not-decoded -->

By the first order optimality condition of the RHS, we should have:

<!-- formula-not-decoded -->

Therefore, π is the strict NE of ¨ M .

□

Definition E.8 (Distance measure between policies) . Given two policies π := { π w h } h ∈ [ H ] ,w ∈ [ W ] and ˜ π := { ˜ π w h } h ∈ [ H ] ,w ∈ [ W ] , we define:

<!-- formula-not-decoded -->

Condition E.9. Given a model function ¨ M := { ( ¨ P w ¨ M ,h , ¨ r w ¨ M ,h ) } h ∈ [ H ] , for any fixed w ∈ [ W ] , h ∈ [ H ] , s w h +1 ∈ S w h +1 , s w h ∈ S w h , a w h ∈ A w h and any π , ¨ P w ¨ M ,h ( s w h +1 | s w h , a w h , π ) and ¨ r w ¨ M ,h ( s w h , a w h , π ) are continuous at π w.r.t. distance d ∞ , 1 .

Lemma E.10 (Continuity of ¨ Q ) . Under Cond. E.9, for any w ∈ [ W ] , any s w h , a w h and π , ¨ Q w, π ¨ M ,h ( s w h , a w h , π ) is continuous at π w.r.t. the distance d ∞ , 1 in Def. E.8.

Proof. The proof is obvious by noting that ¨ Q w, π ¨ M ,h ( s w h , a w h , π ) is a function resulting from finite multiplication and addition among ¨ P w ¨ M ,h ( ·|· , · , π ) , ¨ r w ¨ M ,h ( · , · , π ) and π . □

In the following proposition, we will establish the existence of NE based on the existence of strict NE.

Proposition E.11. Under Cond. E.9, the MT-PAM has at least one NE policy π

NE satisfying Def. E.5.

Proof. We first show the mapping Γ SNE ¨ M : Π → Π is continuous under Cond. E.9. Based on a similar discussion as Lem. E.6 in (Huang et al., 2024),

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is also continuous for any fixed u ∈ ∆( A w ) .

By Lem. E.10, and the rule of composition of continuous functions, Γ SNE ¨ M is a continuous mapping. Therefore, Γ SNE ¨ M maps from the closed and convex polytope Π to a subset of itself. By Brouwers fixed point theorem it has a fixed point. By Lem. E.7, such fixed point is a strict NE of ¨ M .

Comparing with Def. E.5 and Def. E.6, we know NE is a super-set of strict NE, which implies the existence of NE in the MT-PAM. □

## E.2.2. EXISTENCE OF NASH EQUILIBRIUM IN MT-MFG AS COROLLARY

Conversion from Multi-Type MF-MDP to MT-PAM Given a Multi-Type MF-MDP M , we can convert it to a MT-PAM sharing the same { µ w 1 , S w , A w , H } w ∈ [ W ] with M , while the transition and reward functions of ¨ M are defined by:

<!-- formula-not-decoded -->

where µ π M ,h is the density of agents in all types induced by policy π in model M starting from µ π M , 1 = µ 1 . Proposition E.12. [Existence of NE in MT-MFG] Under Assump. D, the Multi-Type MF-MDP has at least one NE policy π NE satisfying Eq. (11) .

Proof. By Lem. E.13, we know the MT-PAM converted from such Multi-Type MF-MDP satisfying Cond. E.9, and by Prop. E.11, the MT-PAM has at least one NE. Easy to check that such NE is also a NE for the Multi-Type MF-MDP satisfying Eq. (11). □

## E.2.3. PROOFS RELATED TO THE MT-PAM CONVERTED FROM MULTI-TYPE MF-MDP

Lemma E.13. [Lipschitz Continuity of MT-PAM] Given a Multi-Type MF-MDP M satisfying the Lipschitz continuity condition in Assump. D, consider the MT-PAM ¨ M converted from M according to Eq. (7) , we have ¨ M is also Lipschitz continuous that, ∀ w ∈ [ W ] , h ∈ [ H ] and any s w h ∈ S w , a w h ∈ A w ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is continuous for any fixed q ∈ R |A w | , and Proof. Based on Lem. I.3, as a special case, when M = M ′ , we have:

<!-- formula-not-decoded -->

Therefore, for any w ∈ [ W ] ,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

## F. Missing Details and Proofs for Results in Sec. 4

## F.1. Proofs for Lemma and Theorems used for Insights

Lemma 4.3. [Implication of Local Alignment] Given any M, ˜ M with transition P M and P ˜ M , denote ̂ π NE M to be an ε 1 -approximate NE of M , suppose d ( M, ˜ M | ̂ π NE M ) ≤ ε 2 , then ̂ π NE M is also an O ( ε 1 + ε 2 ) -approximate NE of ˜ M .

Proof. For any policy ˜ π , we have:

<!-- formula-not-decoded -->

□

Theorem 4.4. [Bridge Policy] If the Else -branch in Line 6 in Alg. 1 is activated, running Alg. 3 returns a bridge policy π NE ,k Br , such that, π NE ,k Br is an approximate NE of M π NE ,k Br Ctr .

Proof. Thm. 4.4 is just a helper theorem to make it easy for the reader to understand our proofs. It will not be used in the proof of our main results Thm. F.7, so here we only show an informal proof.

Combining with Lem. E.4 and Thm. E.2, we show the bridge model ¨ M Br has at least one NE π NE Br . In Thm. F.5, we provide upper bound for the distance between the central model of π NE Br with ¨ M Br, which implies π NE Br is an approximate NE of its central model. □

## F.2. Definition of ε -cover of Policy Space

Proposition F.1 ( ε -cover of Π ) . Consider the set

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Then, Π ε is an ε -cover of the policy space Π w.r.t. d ∞ , 1 distance.

Proof. For any u ∈ ∆( A ) , there exists a v ∈ N ε , such that, ∥ u -v ∥ 1 ≤ 1 N · ( A -1) + A -1 N ≤ ε , which implies N ε is an ε -cover of simplex ∆( A ) . By definition of Π ε , we finish the proof. □

## F.3. Proofs for Algorithm 2

Theorem F.2 (Adapted from Thm. 4.2 in (Huang et al., 2024)) . For any δ ∈ (0 , 1) , during the running of Alg. 2, suppose M ∗ ∈ ¯ M , then w.p. 1 -δ , ∀ t ∈ [ T ] , we have M ∗ ∈ ¯ M t . Besides, denote H as the hellinger distance, for each M ∈ ¯ M t with transition P M and any h ∈ [ H ] :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Theorem F.3. Given any reference policy π , ˜ ε, δ ∈ (0 , 1) , M ∗ ∈ ¯ M , if T = ˜ O ( H 4 ˜ ε 2 (dim PE ( M , ε ′ ) ∧ (1 + L T ) 2 H (1 + L T H ) 2 dim II PE ( M , ε ′ )) log 2 2 |M| TH δ ) with ε ′ = O ( ˜ ε H 2 (1+ L T ) H ) , w.p. 1 -δ , Alg. 2 terminates at some T 0 ≤ T , and return ¯ M T 0 s.t. (i) M ∗ ∈ ¯ M T 0 (ii) ∀ M ∈ ¯ M T 0 , d ( M ∗ , M | π ) ≤ ˜ ε .

Proof. Suppose Alg. 2 proceeds to iteration T 0 ≤ T , and does not terminate at Line 6. On the good events in Thm. F.2, we have M ∗ ∈ ¯ M t for all t ≤ T 0 .

In our first step, we discuss how to provide upper bounds for accumulative model difference depending on two types of P-MBED.

Step 1-(a): Upper Bound Model Difference with Type II P-MBED For any t ≤ T 0 , given the fact that ∥ P -Q ∥ 1 ≤ √ 2 H ( P, Q ) , for any fixed h ∈ [ H ] , we have:

<!-- formula-not-decoded -->

where in the last step is because, as a result of Lem. I.5, Cauchy's inequality, and E 2 [ X ] ≤ E [ X 2 ] , we have:

<!-- formula-not-decoded -->

By Lem. I.1, w.p. 1 -δ/ 2 TH , for any t ∈ [ T 0 ] and any h ∈ [ H ] , we have:

<!-- formula-not-decoded -->

for some constant C and c 1 . By Lem. D.7, we further have:

<!-- formula-not-decoded -->

for some constant c 2 . By Lem. I.1 again, w.p. 1 -δ/ 2 TH , for any T 0 ∈ [ T ] and any h ∈ [ H ] ,

<!-- formula-not-decoded -->

for some constant c 3 . Similarly, we can guarantee by analyzing data collected by ( π, π ) :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, by Lem. I.5 again,

<!-- formula-not-decoded -->

where the last step we use:

<!-- formula-not-decoded -->

Similarly, for model M ′ t , we also have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 1-(b): Upper Bound Model Difference with Type I P-MBED By Thm. F.2 and Lem. I.1, w.p. 1 -δ/ 2 TH , for any T 0 ∈ [ T ] and any h ∈ [ H ] , we have:

<!-- formula-not-decoded -->

As a result of Lem. D.7, we have:

<!-- formula-not-decoded -->

By Lem. I.1, we have:

<!-- formula-not-decoded -->

Similarly, for model M ′ t , we also have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Step 2: Lower Bound on Model Difference On the other hand, since the algorithm does not terminate at step T 0 , we have:

<!-- formula-not-decoded -->

where in the last step we apply Lem. I.8. On the one hand, for Type II P-MBED, we have:

<!-- formula-not-decoded -->

by choosing ε ′ ≤ ˜ ε (2 c 4 H ( H +2)(1 + L T ) H ) -1 , it implies, for some constant c 8 ,

<!-- formula-not-decoded -->

On the other hand, for Type I P-MBED, we have:

<!-- formula-not-decoded -->

by choosing ε ′ ≥ ˜ ε · ( c 7 H ( H +2)) -1 , we have:

<!-- formula-not-decoded -->

As a summary, by choosing

<!-- formula-not-decoded -->

with ε ′ = O ( ˜ εH -2 (1 + L T ) -H ) , we can guarantee the algorithm will terminates for some T 0 ≤ T and return us a model class ¯ M T 0 satisfying max M,M ′ ∈ ¯ M T 0 d ( M,M ′ | π ) ≤ ˜ ε , which implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Remark F.4 (Why (1 + L T ) H Disappears if Considering TypeI P-MBED?) . From the proof above, especially the proof in Step 1-(a) and Step 1-(b), we can see that during the model elimination, what matters is the model distance conditioning on the density induced by the corresponding models, i.e. ∥ P M,h ( ·|· , · , µ π M,h ) -P M ∗ ,h ( ·|· , · , µ π M ∗ ,h ) ∥ 1 . Therefore, if we consider the TypeI P-MBED, we do not need additional conversion between ∥ P M,h ( ·|· , · , µ π M,h ) -P M ∗ ,h ( ·|· , · , µ π M ∗ ,h ) ∥ 1 and ∥ P M,h ( ·|· , · , µ π M ∗ ,h ) -P M ′ ,h ( ·|· , · , µ π M ∗ ,h ) ∥ 1 , which is the origin of the exponential term (1+ L T ) H in the upper bound regarding TypeII P-MBED.

## F.4. Proofs for Algorithm 3

Recall the notations for central models in Appx. E.1.

Theorem F.5. Suppose we feed Alg. 3 with a model class ¨ M , the bridge model ¨ M Br it computes is a valid model, and by choosing ¯ ε = ε 0 / min { 2 HL r (1+ L T ) H -1 L T , 2 H ( H +1)((1 + L T ) H -1) } , for any reference policy π and its associated central model ¨ M ε 0 Ctr ( π ; ¨ M ) , we have:

<!-- formula-not-decoded -->

˜

<!-- formula-not-decoded -->

Proof. In the proof, for notation simplicity, given the refernce policy π , we use ¨ M π Ctr as a short note of ¨ M ε 0 Ctr ( π ; ¨ M ) , i.e. the central model regarding π .

Validity of Construction First of all, note that Π ¯ ε is an ¯ ε -cover of the policy space. Therefore, for any π , there must exist at least one ˜ π ∈ Π ¯ ε satisfying d ∞ , 1 ( π, ˜ π ) ≤ ¯ ε which ensures ∑ ˜ π ∈ Π ¯ ε [2¯ ε -d ∞ , 1 ( π, ˜ π )] + &gt; 0 . So the transition and reward functions in the bridge model is well-defined, and also continuous in π w.r.t. distance d ∞ , 1 .

Upper Bound on Transition Difference By definition,

<!-- formula-not-decoded -->

We only need to care about those ˜ π ∈ Π ¯ ε with [2¯ ε -d ∞ , 1 ( π, ˜ π )] + &gt; 0 , i.e. d ∞ , 1 ( π, ˜ π ) &lt; 2¯ ε . Given the condition when Alg. 1 call Alg. 3, we have B ε 0 π ( ¨ M π Ctr ; ¨ M ) &gt; | ¨ M| 2 for any π . Therefore, for any π and ˜ π with d ∞ , 1 ( π, ˜ π ) ≤ 2¯ ε , there exists a model ¨ M share such that ¨ M share ∈ B ε 0 π ( ¨ M π Ctr ; ¨ M ) ∩ B ε 0 π ( ¨ M ˜ π Ctr ; ¨ M ) , which implies for any π ′

˜

<!-- formula-not-decoded -->

where by applying Lem. I.7, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which implies,

Therefore,

<!-- formula-not-decoded -->

Upper Bound on Reward Difference By definition, for each h, s h , a h , we have:

<!-- formula-not-decoded -->

Similarly, for those ˜ π ∈ Π ¯ ε with [2¯ ε -d ∞ , 1 ( π, ˜ π )] + &gt; 0 , we have:

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Next we prove an important Lemma based on results in theorem above, which indicates that the bridge policy constructed in Alg. 3 is close to the NE of its central model.

Lemma F.6. Suppose the Else -branch in Line 6 if activated in Alg. 2, for policy π NE ,k Br and its corresponding central model M k Ctr := arg max M ∈M k |B ε 0 π NE ,k Br ( M ; M k ) | , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

˜

<!-- formula-not-decoded -->

Proof. For any policy π , we have

<!-- formula-not-decoded -->

which finishes the proof.

## F.5. Proofs for Algorithm 1

Theorem 4.5. In Alg. 1, by choosing ε 0 = ε 8(1+ L r H )( H +4) , ˜ ε = ε 0 6 , and choosing ¯ ε according to Thm. F.5, w.p. 1 -δ , (1) if the If-Branch in Line 5 is activated: we have |M k +1 | ≤ |M k | / 2 ; (2) otherwise, in the Else-Branch in Line 6: either we return the π NE ,k Br which is an ε -approximate NE for M ∗ ; or the algorithm continues with |M k +1 | ≤ |M k | / 2 .

Proof. We separately discuss the if and else branches in the algorithm.

Proof for If-Branch in Line 5 On the events in Thm. F.3, for any ˜ M ̸∈ B ε 0 π k ( M ∗ ; M k ) , we have d ( M ∗ , ˜ M | π k ) ≥ ε 0 &gt; ˜ ε , which implies ˜ M ̸∈ M k +1 . Combining the condition of If-Branch , we have:

<!-- formula-not-decoded -->

Proof for Else-Branch in Line 6 First of all, on the events in Thm. F.3, we have d ( M ∗ , ˜ M k | π NE ,k Br ) ≤ ˜ ε . By applying Lem. I.2, it implies:

<!-- formula-not-decoded -->

Also note that:

<!-- formula-not-decoded -->

In the following, we separately discuss two cases.

Case 1: E NE ˜ M k ( π NE ,k Br ) ≤ 3 ε 4 and Line 9 is activated Given that ˜ ε ≤ ε 16(1+ L r H ) :

<!-- formula-not-decoded -->

which implies π NE ,k Br is an ε -NE of M ∗ .

Case 2: E NE ˜ M k ( π NE ,k Br ) &gt; 3 ε 4 and Line 9 is not activated As a result, for any policy π ,

<!-- formula-not-decoded -->

Therefore, by our choice of ˜ ε ,

<!-- formula-not-decoded -->

On the other hand, by Lem. F.6, for any π , we have:

<!-- formula-not-decoded -->

According to the choice of ε 0 , we have 2(1 + L r H )( H +4) ε 0 ≤ ε 4 , therefore,

<!-- formula-not-decoded -->

Next we try to show that models in B ε 0 π NE ,k Br ( M k Ctr , M k ) will be eliminated. For any M ∈ B ε 0 π NE ,k Br ( M k Ctr , M k ) , because of ˜ ε &lt; ε 48( L r H +1) we have:

<!-- formula-not-decoded -->

On the event in Thm. F.3 (which holds with probability 1 -δ ), M ̸∈ M k +1 , which implies,

<!-- formula-not-decoded -->

□

Theorem F.7. [Sample Complexity of Learning MFGs] Under Assump. A and B, by running Alg. 1 with Alg. 2 as ModelElim and Alg. 3 as BridgePolicy , and hyper-parameter choices according to Thm. F.3, 4.5, and F.5, w.p. 1 -δ , Alg. 1 will terminate at some k ≤ log 2 |M| + 1 and return us an ε -NE of M ∗ , and the number of trajectories consumed is at most O ( H 7 ε 2 (1 + L r ) 2 (dim PE ( M , ε ′ ) ∧ H 3 (1 + L T H ) 2 (1 + L T ) 2 H dim II PE ( M , ε ′ )) log 3 |M| δ ) where ε ′ = O ( ε/H 3 (1 + L r )(1 + L T ) H ) , and in ˜ O we omit logarithmic terms of ε, H, log |M| , dim PE , 1 + L T and 1 + L r .

Proof. As a result of Thm. 4.5, w.p. 1 -δ log 2 |M| +1 · (log 2 |M| +1) = 1 -δ , there exists a step k ≤ log 2 |M| +1 such that Alg. 1 will terminate the return us an ε -approximate NE of M ∗ . The total number of trajectories required is:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we use the fact that by Thm. F.3, we choose ˜ ε = ε 0 6 = O ( ε (1+ L r H ) H ) , and ε ′ = O ( ˜ ε/H 2 (1 + L T ) H ) = O ( ε/H 3 (1 + L r H )(1 + L T ) H ) . □

## F.6. Sample Complexity Separation between Mean-Filed Control and Mean-Field Games

In this section, we establish the separation between of RL in MFC and MFGs from information theoretical perspective.

A Basic Recap of the MFC Setting In MFC, similar to single-agent RL, we are interested in finding a policy ̂ π ∗ Opt to approximately minimize the optimality gap E Opt ( π ) := max ˜ π J M ∗ ( ˜ π ; µ ˜ π M ∗ ) -J M ∗ ( π ; µ π M ∗ ) , i.e.,

<!-- formula-not-decoded -->

Exponential Lower Bound in Tabular RL for Mean-Field Control Our results are based on a different query model from Def. 2.1 defined below.

Definition F.8 (Strong Query Model) . The Strong Query Model (SQM) can take a policy π and return a sequence of transition function { P π h ( ·|· , · ) } H h =1 , such that P π h ( ·| s h , a h ) := P M ∗ ,h ( ·| s h , a h , µ π M ∗ ,h ) for any h ∈ [ H ] , s h ∈ S h , a h ∈ A h .

The SQM is strictly stronger than the sample query model in Def. 2.1, because given the conditional model { P π h ( ·|· , · ) } H h =1 , one can sample arbitrary trajectories by arbitrary policies from it, and therefore, recover the data collection process in Def. 2.1. In the following, we investigate the number of SQM queries required to identify ε -optimal policy in MFC setting. We show that, under Assump. A and B, even in the tabular setting, MFC requires queries exponential to the number of states and actions.

Figure 1. Construction of Lower Bound

<!-- image -->

Theorem F.9. [Exponential Lower Bound for MFC] Given arbitrary L T &gt; 0 and d ≥ 2 , consider tabular MF-MDPs satisfying Assump. B with Lipschitz coefficient L T , |S| = |A| = d and H = 3 . For any algorithm Alg, and any ε ≤ L T d +1 , there exists an MDP M ∗ and a model class M satisfying M ∗ ∈ M , and |M| = Ω(( L T dε ) d -1 ) , s.t., if Alg only queries GM or DCP for at most K times with K ≤ |M| / 2 -1 , the probability that Alg produces an ε -optimal policy is less than 1 / 2 .

Proof. Our proof is divided into three parts: construction of hard MF-MDP instance, construction of model class M , and the proof of lower bound.

Part 1: Construction of Hard Examples We construct a three layer MDP as shown in Fig. 1. The initial state distribution is fixed to be µ 1 ( s 1 ) = 1 , and we have S states and A actions available at each layer with S = A = d . The transition at initial state is deterministic, i.e., P ( s i 2 | s 1 , a i 1 , µ 1 ) = 1 . At the second layer, given L T ≤ 1 , there exists an optimal state density µ ∗ 2 , such that, ∀ i ∈ [ S ] , j ∈ [ A ] and ∀ µ 2 ∈ ∆( S ) :

<!-- formula-not-decoded -->

̸

where [ x ] + = max { x, 0 } . As for the reward function, we have zero reward at each state action in the previous two layers, and for the third layer, we have only have non-zero reward at r 3 ( s 1 3 , · , · ) = 1 and r 3 ( s i 3 , · , · ) = 0 for all i = 1 .

As we can see, for arbitrary policy π , we have µ π 2 ( s i 2 ) = π ( a i 1 | s 1 ) . Besides, the optimal policy should be taking action to make sure µ 2 = µ ∗ 2 , which can be achieved by setting π ∗ ( a i 1 | s 1 ) = µ ∗ 2 ( s i 2 ) , and then take arbitrary policy at the second layer. Even if the agent just wants to achieve ε -near-optimal policy, it at least has to determine the position of set { µ : ∥ µ -µ ∗ 2 ∥ 1 ≤ 4 ε L T } . The key difficulty here is to explore and gather information which can be used to infer µ ∗ 2 .

We further reduce the difficulty of the exploration by providing for the learner with the transition at initial state and the third layer (or equivalently, the available representation function for the first and third layers is unique) and all the information of reward function. All the learner need to do is to identify the correct feature for the second layer and use it to obtain the optimal policy (at the initial state) to maximize the return.

Next, we verify the above model belongs to the low-rank Mean-Field MDP. For h = 1 , it's easy to see P ( s i 2 | s 1 , a j 1 , µ 1 ) = ϕ 1 ( s 1 , a j 1 , µ 1 ) ⊤ ψ 1 ( s i 2 ) , where ϕ 1 ( s 1 , a j 1 , µ 1 ) = e j and ψ 1 ( s i 2 ) = e i , and e ( · ) is the one-hot vector with the ( · ) -th element equal 1. For the second layer, given a density µ ∈ ∆( S ) , we use ϕ µ,L T to denote the following feature function class that, ∀ i ∈ [ S ] , j ∈ [ A ] , µ ′ ∈ ∆( S ) ,

<!-- formula-not-decoded -->

and the next state feature function is ψ ( s i 3 ) = e ⊤ i , ∀ i ∈ [ d ] . It's easy to verify that the transition can be decomposed to ϕ µ ∗ 2 ,L T ( · , · , µ 2 ) ⊤ ψ ( s i 3 ) , and the above feature satisfies the normalization property:

<!-- formula-not-decoded -->

Besides, we verify that for any choice of µ , the induced transition function is L T -Lipschitz:

<!-- formula-not-decoded -->

Part 2: Construction of Model Class Given an integer ζ , we denote N ζ := { µ | µ ( s i 2 ) = N ( s i 2 ) /ζ, N ( s i 2 ) ∈ N , ∑ i ∈ [ S ] N ( s i 2 ) = ζ } . In another word, N ζ includes all state density with resolution 1 /ζ . Now, consider N ⌊ L T 5 ε ⌋ . For each µ, µ ′ ∈ N ⌊ L T 5 ε ⌋ , we should have:

<!-- formula-not-decoded -->

Therefore, if we consider the set B ( µ, 4 ε L T ) := { µ ′ ∈ ∆( S ) |∥ µ -µ ′ ∥ 1 ≤ 4 ε L T } , we can expect B ( µ, 4 ε L T ) ∩ B ( µ ′ , 4 ε L T ) = ∅ for any µ, µ ′ ∈ N ⌊ L T 5 ε ⌋ . Given arbitrary N ≤ |N ⌊ L T 5 ε ⌋ | = ( ⌊ L T 5 ε ⌋ + d -1)! ( ⌊ L T 5 ε ⌋ )!( d -1)! = Ω(( L T dε ) d -1 ) , we can find N -1 different elments { µ 1 2 , ..., µ N 2 } ⊂ N ⌊ L T 5 ε ⌋ and construct (here we only specify the representation at the second layer, since we assume the other layers are known)

<!-- formula-not-decoded -->

For analysis, we introduce another model ¯ M which shares the transition and reward function as M n s but for the transition of second layer, it has:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We define:

and define:

Note that ¯ M = ( ¯ ϕ, ψ ) ∈ M .

Part 3: Establishing Lower Bound Now, we consider the following learning setting: the environment randomly select one model M from M and provide the entire representation feature class M (which is also the entire model class) to the learner; then, the learner can repeatedly use gathered information to compute a policy π k and query it with SQM for each iteration, and output a final policy after K steps. We want to show that, for arbitrary algorithm, there exists at least one model in M which cost number of queries linear w.r.t. N before identifying the optimal policy.

In the following, we use E k,M n to denote the event that in the first k trajectories, there is at least one policy (or equivalently, density µ π 2 ) used to query SQM resulting in ∥ µ π 2 -µ n ∥ 1 ≤ 4 ε L T . The key observation is that, given arbitrary algorithm Alg, for arbitrary fixed n ∈ [ N ] , if Alg never deploy a policy π (or equivalently, query an density µ π 2 ) satisfying ∥ µ π 2 -µ n ∥ 1 ≤ 4 ε L T , the algorithm can not distinguish between M n and ¯ M , and should behave similar in both M n and ¯ M . Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We use Alg ( K ) to denote the policy output by the algorithm in the final. Besides, we use Π( µ, b 0 ) := { π |∥ µ π 2 -µ ∥ 1 ≤ b 0 } to denote the set of policies, which can lead to a density µ π 2 close to µ . Then, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last step is because,

<!-- formula-not-decoded -->

which also implies:

̸

Therefore, the average success probability would be:

<!-- formula-not-decoded -->

As a result, even if K = |M| 2 -1 = O ( N ) , there exists n ∈ [ N ] , such that, the failure rate

<!-- formula-not-decoded -->

## G. Proofs for Multi-Type MFGs

## G.1. More Details about the Setting

In Multi-Type MF-MDP, we will denote µ w h ∈ ∆( S w ) to be a state density at step h in type w , and define µ w := { µ w h } h ∈ [ H ] to be the collection over all h . For the policies, we define Π w := { π w := { π w h }|∀ h ∈ [ H ] , π w h : S w h → ∆( A w h ) } , and Π := { π := { π w } w ∈ [ W ] |∀ w ∈ [ W ] , π w ∈ Π w } . In this paper, we only consider policies in Π , i.e. the set of non-stationary Markovian policies.

In order to distinguish with (single-type) MF-MDP setting, for notations regarding the collection of densities or policies over all groups, we use the bold font, i.e. µ h := { µ w h } w ∈ [ W ] and µ := { µ h } h ∈ [ W ] , π := { π w } w ∈ [ W ] and π h := { π w h } w ∈ [ W ] . When a policy π and a model M is speicified, we use µ π M := { µ w, π M } w ∈ [ W ] = { µ π M ,h } h ∈ [ H ] to denote the collection of densities of W groups induced by the policy π in model M , where µ w, π M := { µ w, π M ,h } h ∈ [ H ] and µ π M ,h := { µ w, π M ,h } w ∈ [ W ] . When a policy π ∈ Π is specified, the evolution of the densities in all groups can be described by:

<!-- formula-not-decoded -->

Similarly to MF-MDP setting, given two policies ˜ π , π ∈ Π , we can define the value functions for each group following ˜ π while conditioning on π :

<!-- formula-not-decoded -->

where we use E ˜ π , M ( π ) to denote the expectation over trajectories generated by executing policy ˜ π in M conditioning on π , i.e. the transitions P w M ,h ( ·|· , · , µ π M ,h ) and rewards r w h ( · , · , µ π M ,h ) are fixed by π . Besides, we denote J w M ( ˜ π ; π ) := E s w 1 ∼ µ w 1 [ V w, π M , 1 ( s w 1 ; µ π M )] to be the expected return of type w in model M by executing ˜ π conditioning on π . The Nash Equilibrium policy in Multi-Type MFG is defined to be the policy π NE := { π w, NE } w ∈ [ W ] satisfying:

<!-- formula-not-decoded -->

We define ∆ w M ( ˜ π , π w ) := J w M ( ˜ π ; π ) -J w M ( π ; π ) , and define E w, NE M ( π ) := max ˜ π ∆ w M ( ˜ π , π ) . Our goal in this setting is to find an ε -approximate NE policy ̂ π NE := { ̂ π w, NE } w ∈ [ W ] such that:

<!-- formula-not-decoded -->

## G.2. Conversion from MT-MFG to MFG with Constrained Policy Space

Intuitively, the construction is made by integrating the state and action spaces, which will result in a MFG with transition and reward functions following some block diagnoal structure.

Given a MT-MFG M := { ( µ w 1 , H, S w , A w , P w M , r w ) w ∈ [ W ] } , we denote the converted MF-MDP by M MFG := { µ 1 , H, S MFG , A MFG , P MFG , r MFG } , where we have the extended state space S MFG := ⋃ w ∈ [ W ] ( S w × { w } ) and action space A MFG := ⋃ w ∈ [ W ] ( A w × { w } ) . As we can see, the new state/action space is the collection of all states/actions agumented by the group index w ∈ [ W ] . In this way, states and actions in different groups can be distinguished by the group index w . Next, we construct a new initial distribution µ 1 := [ µ 1 1 W , µ 2 1 W , ..., µ W 1 W ] by concatenating all the initial distributions with normalization. For the policy, we define

<!-- formula-not-decoded -->

with Π w := { π w : S w → ∆( A w ) } . In another word, Π † includes and only includes policies taking actions sharing the same group index with states, and we only consider the policies π ∈ Π † .

## G.2.1. DEFINITION OF TRANSITION/REWARD FUNCTIONS IN THE LIFTED MF-MDP

Next, given a density µ h := [ µ 1 h W , ..., µ W h W ] ∈ ∆( S MFG ) with µ w h ∈ ∆( S w ) , the transition and reward functions in the converted MFG is defined by (note that by definition of Π † , we only need to consider the case when the state and action share the group index):

<!-- formula-not-decoded -->

̸

For the sake of rigor, we include the definition for the transition/reward functions on those s w h ◦ w and a ˜ w h ◦ ˜ w with w = ˜ w . We define P MFG ,h ( ·| s w h ◦ w,a ˜ w h ◦ ˜ w,µ h ) to be a uniform distribution over S MFG, and r MFG ,h ( s w h ◦ w,a ˜ w h ◦ ˜ w,µ h ) = 0 , for any µ h ∈ ∆( S MFG ) .

After specifying a policy π ∈ Π † , denote π := { π w } w ∈ [ W ] to be the MT-MFG policy that π corresponds to, we can verify that the state density µ π M MFG ,h ∈ ∆( S MFG ) evolves according to:

<!-- formula-not-decoded -->

where recall µ w, π M ,h denotes the density of type w induced by π in model M . To see this, by induction,

<!-- formula-not-decoded -->

Intuitively, in the converted MFG, following a policy π ∈ Π † , if an agent starts from the initial state with index w , it will follow a trajectory as if it is generated in the original MT-MFG. In the following, we will call M MFG (or M ) the corresponding MFG (or MT-MFG) of M (or M MFG).

## G.3. Assumptions and Additional Definitions

Recall the definition of {M w } w ∈ [ W ] and M discussed in Sec. 2, In the following, we use M MFG to denote the model class including MFG models converted from models in M according to the method discussed in Appx. G.2, and denote M ∗ MFG to be the one converted from M ∗ .

We have the following assumptions, which can be regarded as a generalization of Assump. A, B and Def. 2.1.

Assumption C (Realizability) . The true model M ∗ ∈ M .

Assumption D (Lipschitz Continuity for MT-MFG) . For any M ∈ M , and for two arbitrary policies π , ˜ π

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here we introduce a normalization factor W given that ∥ µ π M ,h -µ ˜ π M ,h ∥ 1 = ∑ w ∈ [ W ] ∥ µ w, π M ,h -µ w, ˜ π M ,h ∥ 1 .

Definition G.1 (Trajectory Sampling Model in MT-MFG) . The learner can query the sampling model with an arbitrary policy π := { π 1 , ..., π W } , a group index w and another policy ˜ π := { ˜ π 1 , ..., ˜ π W } , and receive a trajectory by executing ˜ π w while the transition and reward functions are fixed by π , i.e. P w M ∗ ,h ( ·|· , · , µ π M ∗ ,h ) and r w h ( · , · , µ π M ∗ ,h ) .

Similar to the sampling model in Def. 2.1, the model above can be implemented by utilizing the observation of an individually deviating agent with type w following policy ˜ π w while the other agents follows π in a large Multi-Type MARL system.

Moreover, for learning in the lifted MFGs, note that a sampling model in M ∗ MFG as described in Def. 2.1 can be implemented by Def. G.1. To see this, given two policies π, ˜ π ∈ Π † , which correspond to π := { π 1 , ..., π W } and ˜ π := { ˜ π 1 , ..., ˜ π W } , respectively, the trajectory can be generated by first uniformly sample w ∈ [ W ] , and then sample a trajectory with Def. G.1 with π converted from π , type w and policy ˜ π w .

Proposition G.2. Given a MT-MFG model class M satisfying Assump. D, consider its converted MF-MDP model class M MFG, for any M ∈ M MFG, and any π, ˜ π ∈ Π † , we have,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. According to the definition in Appx. G.2.1, for those s h , a h with different group index, their transition or reward differences will be 0. Therefore, we only need to consider the case when s h , a h share the group index.

As we explained in Eq. (14), given π, ˜ π ∈ Π † , which corresponds to π , ˜ π ∈ Π , respectively, we have:

<!-- formula-not-decoded -->

where M is the corresponding MT-MFG model of M . Combining with Assump. D, we finish the proof. □

## G.4. Constrained Nash Equilibrium

Proposition 5.1. Given a MT-MFG M and its lifted MFG M MFG, we have: (1) an ε -constrained NE ̂ π NE Cstr ∈ Π † for M MFG is a ( Wε )-NE in M ; (2) an ε -NE ̂ π NE in M is an ε -constrained NE for M MFG.

Proof. Given any π ∈ Π † , we denote its corresponding policy in MT-MFG by π := { π 1 , ..., π W } with π w : S w → ∆( A w ) and π ( a w h ◦ w | s w h ◦ w ) = π w ( a w h | s w h ) . Conversely, given any π := { π 1 , ..., π W } , we can convert it to a policy in Π † , which we denote by π . For ̂ π NE Cstr , we denote its correspondence in MT-MFG by ̂ π NE := { ̂ π NE , 1 , ... ̂ π NE ,W } .

Note that, given any π, ˜ π ∈ Π † and their correspondence π := { π 1 , ...π W } and ˜ π := { ˜ π 1 , ... ˜ π W } , we have:

<!-- formula-not-decoded -->

where recall r w is the reward in type w in MT-MFG and µ π M ,h := { µ w,π M ,h } w ∈ [ W ] is the collection of densities for all groups. Consider the case when π = ̂ π NE and ˜ π ˜ w ← ̂ π NE , ˜ w for all ˜ w except ˜ w = w , we have:

<!-- formula-not-decoded -->

By repeating such discussion for any w ∈ [ W ] and any π w , we complete the proof for argument (1).

On the other hand, given an ε -approximate NE ̂ π NE in M and its corresponding ̂ π NE Cstr in M MFG, for any π ∈ Π † we have:

<!-- formula-not-decoded -->

To upper bound the RHS, for each w ∈ [ W ] , we consider an arbitrary policy ˜ π with with ˜ π ˜ w = ̂ π NE , ˜ w for all ˜ w except π w = π w , we should have:

˜

<!-- formula-not-decoded -->

By repeating for all w ∈ [ W ] , we complete the proof for argument (2).

□

Existence of Constrained NE Policy Before we introduce algorithms finding constrained NE(s) in MFG, we first investigate their existence, which is actually directly implied by Prop. 5.1.

Corollary G.3. Given M satisfying Lipschitz continuity conditions in Assump. D, the MFG M MFG converted from M has at least one constrained NE satisfying ∀ π ∈ Π † , J M MFG ( π, ̂ π NE Cstr ) ≤ J M MFG ( ̂ π NE Cstr , ̂ π NE Cstr ) + ε with ε = 0 .

Proof. From Prop. E.12, any MT-MFG M satisfying Assump. D has at least one NE. As implied by Prop. 5.1 when ε → 0 , any MFG M MFG converted from an MT-MFG M with NE(s) should have at least one constrained NE. Therefore, under Assump. D, we can guarantee any model in the converted function class M MFG has at least one constrained NE. □

## G.5. Algorithm Details

We first generalize some notations in Sec. 4. We define the (constrained) conditional distance between models:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Besides, given a MF-MDP class M , a model M ∈ M , and any policy π , we define the ε 0 -neighborhood of M in M w.r.t. distance d † ( · , ·| π ) to be: B † ,ε 0 π ( M ; M ) := { M ′ ∈ M| d † ( M,M ′ | π ) ≤ ε 0 } . The 'Central Model' of M w.r.t. policy π and distance d † is defined to be the model with the largest neighborhood set M † ,ε 0 Ctr ( π ; M ) ← arg max M ∈M |B † ,ε 0 π ( M ; M ) | . When ε 0 and M is clear from context, we will use M † ,π Ctr as a short note.

Besides, we define E † , NE M ( π ) := max ˜ π ∈ Π † ∆ M ( ˜ π, π ) = max ˜ π ∈ Π † J M ( ˜ π, π ) -J M ( π, π ) to be the constrained NE gap.

## Algorithm 4: Multi-Type MFG Learning with Constrained Policy Space

```
1 Input : Model Class M ; Policy Class Π † ; Accuracy level ε 0 , ˜ ε, ¯ ε ; Confidence level δ 2 Convert M to M MFG as described in Appx. G.2; M 1 MFG ←M MFG, δ 0 ← δ log 2 |M MFG | +1 . 3 for k = 1 , 2 , ... do 4 π k ← arg min π ∈ Π † |B † ,ε 0 π ( M † ,π Ctr ; M k MFG ) | ; 5 if |B † ,ε 0 π k ( M † ,π k Ctr ; M k MFG ) | ≤ |M k MFG | 2 then M k +1 MFG ← ModelElimCstr ( π k , M k MFG , ˜ ε, δ 0 ) . ; 6 else 7 π † , NE ,k Br ← BridgePolicyCstr ( M k MFG , ¯ ε ) ; 8 M k +1 MFG ← ModelElimCstr ( π † , NE ,k Br , M k MFG , ˜ ε, δ 0 ) ; 9 Randomly pick ˜ M k from M k +1 MFG ; 10 E † , NE ˜ M k ( π † , NE ,k Br ) ← max π ∈ Π † J ˜ M k ( π, π † , NE ,k Br ) -J ˜ M k ( π † , NE ,k Br , π † , NE ,k Br ) ; 11 if E † , NE ˜ M k ( π † , NE ,k Br ) ≤ 3 ε 4 then return π † , NE ,k Br ; 12 end 13 if |M MFG | = 1 then Return the NE of the model in M MFG. ; 14 end
```

- 15 Return the constrained NE policy of the model in M k MFG .

## Algorithm 5: ModelElimCstr

```
1 Input : Reference Policy π ; Policy Class Π † ; Model Class ¯ M ; Accuracy level ˜ ε ; Confidence δ 2 ¯ M 1 ←M ; ¯ ε ; Choosing T according to Thm. G.4 3 for t = 1 , 2 , ..., T do 4 ˜ π t , M t , M ′ t ← arg max ˜ π ∈ Π † max M,M ′ ∈ ¯ M t E ˜ π,M ( π ) [ ∑ H h =1 ∥ P w M,h ( ·|· , · , µ π M,h ) -P w M ′ ,h ( ·|· , · , µ π M ′ ,h ) ∥ 1 ] . 5 Denote the value taken at the above as ∆ t max . 6 if ∆ t max ≤ ˜ ε then return ¯ M t ; 7 else 8 Z t ←{} 9 for h = 1 , 2 ..., H do 10 for w ∈ [ W ] do 11 // Trajectory sampling in M ∗ MFG can be implemented by Def. G.1. 12 Sample a trajectory with ( π, π ) , and collect the data at step h : { ( s w,t h , a w,t h , s ′ w,t h +1 ) } . 13 Sample a trajectory with ( ˜ π t , π ) , and collect the data at step h : { ( ˜ s w,t h , ˜ a w,t h , ˜ s ′ w,t h +1 ) } . 14 Z t ←Z t ∪ { ( s w,t h , a w,t h , s ′ w,t h +1 ) }{ ( ˜ s w,t h , ˜ a w,t h , ˜ s ′ w,t h +1 ) } . 15 end 16 end 17 ∀ M ∈ ¯ M t , define l π MLE ( M ; Z t ) := K ∑ k =1 W ∑ w =1 H ∑ h =1 log P w M,h ( s ′ w,t h +1 | s w,t h , a w,t h , µ π M,h ) + log P w M,h ( ˜ s ′ w,t h +1 | ˜ s w,t h , ˜ a w,t h , µ π M,h ) . ¯ M t +1 ←{ M ∈ ¯ M t | l π MLE ( M ; Z t ) ≥ max ˜ M l π MLE ( ˜ M ; Z t ) -log WHT |M| δ } . 18 end 19 end
```

## Algorithm 6: BridgePolicyCstr

- 1 Input : MF-MDP model class M ; Policy Space Π † ; Accuracy Level ¯ ε , ε 0
- 2 Convert Policy-Aware MDP Model Class ¨ M from M by Eq. (1).
- 3 Construct ¯ ε -cover of the policy space Π † w.r.t. d ∞ , 1 distance, denoted as Π † ¯ ε .
- 4 for ˜ π ∈ Π † ¯ ε do Find the central model ¨ M † ,ε 0 Ctr ( ˜ π ; ¨ M ) ← arg max ¨ M ∈ ¨ M |B † ,ε 0 π ( ¨ M ; ¨ M ) | ;
- 5 Construct the new PAM ¨ M Br with transition and reward functions ∀ w ∈ [ W ] , h ∈ [ H ] :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- 6 Find the NE of bridge model: π † , NE Br ← arg min π ∈ Π † max ˜ π ∈ Π † ¨ J ¨ M Br ( ˜ π ; π ) -¨ J ¨ M Br ( π ; π ) . 7 return π † , NE Br .

## G.6. Proofs for Algorithm 4

Theorem G.4. Under Assump. C and D, in Alg. 5, given any ˜ ε , reference policy π , δ ∈ (0 , 1) , and M ∗ ∈ ¯ M , by choosing T = ˜ O ( H 4 ˜ ε 2 (dim CPE | Π † ( M , ε ′ ) ∧ (1 + L T ) 2 H (1 + L T H ) 2 dim II CPE | Π † ( M , ε ′ ))) with ε ′ = O ( ˜ ε H 2 (1+ L T ) H ) , , w.p. 1 -δ , the algorithm terminates at some T 0 ≤ T , and return ¯ M T 0 satisfying (i) M ∗ ∈ ¯ M T 0 (ii) ∀ M ∈ ¯ M T 0 , d † ( M ∗ , M | π ) ≤ ˜ ε .

Proof. The proof is the same as Thm. F.3, except that we consider the constrained policy space, and need to replace P-MBED with constrained P-MBED. □

Theorem G.5. Suppose we feed Alg. 6 with a model class ¨ M and policy space Π † , then for the bridge model ¨ M Br it computes, by choosing ¯ ε = ε 0 / min { 2 HL r (1+ L T ) H -1 L T , 2 H ( H +1)((1 + L T ) H -1) } , for any reference policy π ∈ Π † and its associated central model ¨ M † ,ε 0 Ctr ( π ; ¨ M ) , we have:

<!-- formula-not-decoded -->

Proof. The proof is the same as Thm. F.5 except that we constrain the policies in Π † .

□

Lemma G.6. Suppose the Else -branch in Line 6 if activated in Alg. 2, for policy π † , NE ,k Br and its corresponding central model M † ,k Ctr := arg max M ∈M k |B † ,ε 0 π † , NE ,k Br ( M ; M k ) | , we have:

<!-- formula-not-decoded -->

Proof. The proof is the almost the same as Lem. F.6, except that we consider the constrained policy space.

†

For any policy π ∈ Π , we have

<!-- formula-not-decoded -->

which finishes the proof.

□

Theorem G.7. In Alg. 4, by choosing ε 0 = ε 8( H +4)(1+ L r H ) , ˜ ε = ε 0 6 and choosing ¯ ε according to Thm. G.5, on the good events in Thm. G.4, (1) if the If-Branch in Line 5 is activated: we have |M k +1 | ≤ |M k | / 2 ; (2) otherwise, in the Else-Branch in Line 6: either we return the π † , NE ,k Br which is an ε -approximate NE for M ∗ ; or the algorithm continues with |M k +1 | ≤ |M k | / 2 .

Proof. We separately discuss the if and else branches in the algorithm.

Proof for If-Branch in Line 5 On the events in Thm. G.4, for any ˜ M ̸∈ B † ,ε 0 π k ( M ∗ ; M k ) , we have d † ( M ∗ , ˜ M ) ≥ ε 0 &gt; ˜ ε , which implies ˜ M ̸∈ M k +1 . Combining the condition of If-Branch , we have:

<!-- formula-not-decoded -->

Proof for Else-Branch in Line 6 First of all, on the events in Thm. G.4, we have d † ( M ∗ , ˜ M k | π † , NE ,k Br ) ≤ ˜ ε . By applying Lem. I.2, it implies:

<!-- formula-not-decoded -->

Also note that:

<!-- formula-not-decoded -->

In the following, we separately discuss two cases.

Case 1: E † , NE ˜ M k ( π † , NE ,k Br ) ≤ 3 ε 4 and Line 9 is activated Given that ˜ ε ≤ ε 16(1+ L r H ) :

<!-- formula-not-decoded -->

which implies π † , NE ,k Br is an ε -NE of M ∗ .

Case 2: E † , NE ˜ M k ( π † , NE ,k Br ) &gt; 3 ε 4 and Line 9 is not activated As a result, for any policy π ∈ Π † , by Eq. (17), we have:

<!-- formula-not-decoded -->

Therefore, by our choice of ˜ ε ,

<!-- formula-not-decoded -->

On the other hand, by Lem. F.6, for any π ∈ Π † , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

According to the choice of ε 0 , we have 2(1 + L r H )( H +4) ε 0 ≤ ε 4 , therefore,

<!-- formula-not-decoded -->

Next we try to show that models in B ε 0 π † , NE ,k Br ( M † ,k Ctr , M k ) will be eliminated. For any M ∈ B ε 0 π † , NE ,k Br ( M † ,k Ctr , M k ) , we have:

<!-- formula-not-decoded -->

On the good events in Thm. G.4, we have M ̸∈ M k +1 , which implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem G.8. [Sample Complexity in MT-MFG] Under Assump. C and D, by running Alg. 4 with Alg. 5 as ModelElimCstr and Alg. 6 as BridgePolicyCstr , and hyper-parameter choices according to Thm. G.4, G.5, and G.7, w.p. 1 -δ , Alg. 4 will terminate at some k ≤ log 2 | M | + 1 and return an ε -NE of M ∗ . The number of trajectories consumed is ˜ O ( W 2 H 7 ε 2 (1 + L r H ) 2 ∑ w ∈ [ W ] (dim MTPE ( M w , ε ′ ) ∧ H 3 (1 + L T ) 2 H (1 + L T H ) 2 dim II MTPE ( M w , ε ′ )) log 2 | M | δ ) , where ε ′ = O ( ε/WH 3 (1 + L r H )(1 + L T ) H ) , dim MTPE ( M w , ε ′ ) and dim II MTPE ( M w , ε ′ ) are the Multi-Type P-MBED defined in Def. D.10, and we omit the logarithmic terms of H,ε, log | M | , dim MTPE , 1 + L T and 1 + L r .

Proof. As a result of Thm. G.7, w.p. 1 -δ log 2 |M MFG | +1 · (log 2 |M MFG | +1) = 1 -δ , there exists a step k ≤ log 2 |M MFG | +1 = log 2 | M | +1 such that Alg. 4 will terminate the return us an ε W -approximate NE of M ∗ MFG . The total number of trajectories required is:

<!-- formula-not-decoded -->

˜

Note that in Thm. G.4, we choose ˜ ε = ε 0 6 = O ( ε WH (1+ L r H ) ) , and ε ′ = O ( ˜ ε/H 2 (1 + L T ) H ) = O ( ε/WH 3 (1 + L r H )(1 + L T ) H ) . Combining with the above discussion and Prop. D.14 and Prop. 5.1, we finish the proof. □

## H. Approximation Ability of Multi-Type MFGs

## H.1. Multi-Type Symmetric Anonymous Games

Notations Given a multi-agent system where agents are divided into W groups, where for each type w the agents share the state-action spaces S w , A w and initial distribution µ w 1 , we use N w to denote the number of agents in group w ∈ [ W ] , and s w,n h , a w,n h and π w,n to denote the state, action, and policy for the n -th agent in type w , respectively. Besides, we define s h := { s w,n h } w ∈ [ W ] ,n ∈ [ N w ] , a h := { a w,n h } w ∈ [ W ] ,n ∈ [ N w ] to be the collection of states and actions of all agents in the system at step h , and denote p s h := { p 1 s h , ..., p W s h } to be the empirical distribution of the agents' states with:

<!-- formula-not-decoded -->

To distinguish the policy in MFG setting, we use ˜ ν := { π w,n } w ∈ [ W ] ,n ∈ [ N w ] to denote the collection of policies. We will denote ν ( a h | s h ) := ∏ w ∈ [ W ] ∏ n ∈ [ N w ] π w,n ( a w,n h | s w,n h ) . Besides, we use ν -( w,n ) ◦ ˜ π w,n to denote the policy replacing π w,n to ˜ π w,n while keeping the others fixed.

Definition H.1 (Multi-Type Symmetric Anonymous Game) . The Multi-Type Symmetric Anonymous Game (MT-SAG) ¯ M := { ( µ w 1 , S w , A w , H, P w , r w ) } w ∈ [ H ] is a Multi-Agent system consists of W groups. Given a policy π , the system evolves as:

<!-- formula-not-decoded -->

Given a policy ν , we define the value functions V : S → [0 , 1] and Q : S × A → [0 , 1] of the ( w,n ) -th agent conditioning on the system state s h to be:

<!-- formula-not-decoded -->

where the expectation is taken over the evolution process in Eq. (18). Besides, we define the total value starting from the initial states J w,n ¯ M ( ν ) := E µ 1 [ V w,n, ν 1 ( s w,n 1 ; s 1 )] .

A policy ν is called to be the NE policy in MT-SAG if any agent can not improve its value by deviating from its current policy while the others' are fixed,

<!-- formula-not-decoded -->

˜

and a policy ν ′ is called to be an ε -approximate NE in MT-SAG if

<!-- formula-not-decoded -->

˜

Assumption E (Lipschitz Continuity in MT-SAG) . We assume the transition and reward functions of MT-SAG are Lipschitz continuous w.r.t. the density, s.t. ∀ w ∈ [ W ] , h ∈ [ H ] , ∀ ̂ µ h , ̂ µ ′ h ∈ ∆( S 1 ) × ... ∆( S W )

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## H.2. Approximating MT-SAGs via MT-MFGs

Definition H.2 (Multi-Type Mean-Field Game Approximation of MT-SAG) . Given an MT-SAG ¯ M , its Multi-Type MeanField (MT-MFG) Approximation is a model Multi-Type MF-MDP model M := { ( µ w 1 , S w , A w , H, P w M , r w M ) } w ∈ [ W ] , sharing the group, initial distribution, state-action spaces and transition P w and reward function r w as MT-SAG (i.e. P w M ( ·|· , · , · ) = P w ¯ M ( ·|· , · , · ) , r w M ( · , · , · ) = r w ¯ M ( · , · , · ) ), by have different transition rules.

<!-- formula-not-decoded -->

Next, we describe 'the different transition rules' in MT-MFG. For simplicity of notation, in the following, we omit M or ¯ M in the sub-scription of transition and reward functions. Given a reference policy π := { π 1 , ..., π W } consisting of W policies shared by each group, the density µ π h := { µ 1 , π h , ..., µ W, π h } at step h is defined by:

<!-- formula-not-decoded -->

where Γ w, π h is a mapping from ∆( S 1 ) × ... ∆( S W ) to ∆( S w ) . The evolution process of the ( w,n ) agent in type w following a deviation policy ˜ π w conditioning on reference policy π is specified by:

<!-- formula-not-decoded -->

Comparing with Eq. (18), the evolution of agents' states is depend on the density when ∀ w ∈ [ W ] , N w → + ∞ , instead of the empirical one in practice.

Recall that given a reference policy π and a deviation policy ˜ π w , the value functions V : S → [0 , 1] of the ( w,n ) -th agent conditioning on the density µ π M ,h is defined to be:

<!-- formula-not-decoded -->

where the expectation is taken over the process in Eq. (20). Then, we define the total value of π -w ◦ ˜ π w given the reference policy π to be:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proposition H.3 (Approximation Error of MT-MFG) . Given a Multi-Type Symmetric Anonymous Game (MT-SAG) ¯ M , as defined in Def. H.1, and its Multi-Type MFG approximation (MT-MFG) M , as defined in Def. H.2, suppose π := { π w } w ∈ [ W ] is the NE policy of MT-MFG, then for any ε 0 &gt; 0 , the lifted policy ν := { π w,n } w ∈ [ W ] ,n ∈ [ N w ] with π w,n = π w , ∀ n ∈ [ N w ] is an ε 0 -approximate NE of MT-SAG if

<!-- formula-not-decoded -->

where S max := max w S w .

Proof. Given π := { π 1 , ..., π W } , we denote ν := { π w,n } w ∈ [ W ] ,n ∈ [ N w ] to be the lifted policy such that ν w,n ← π w for all w and n ∈ [ N w ] . Given a deviation policy ˜ π w for some w ( ˜ π w may equal π w ), we define ˜ ν := { π w,n } w ∈ [ W ] ,n ∈ [ N w ] to be a policy in MT-SAG, such that ˜ π ˜ w,n ← π ˜ w for all agent except the ( w, 1) -th agent (i.e. the first agent in type w ), we set π ( w, 1) ← π w .

˜

˜

Concentration Events We first provide a high-probability bound for the distance between state density µ π h in MT-MFG and the empirical distribution p ˜ ν s h in MT-SAG w.r.t. the lifted policy ˜ ν .

We use Γ w, π M ,h ( · ) to denote the operator Γ w, π h ( · ) in Eq. (20) specified in model M . We extend its definition to h = 0 by ∀ µ 0 , Γ w, ˜ ν 0 ( µ 0 ) ← µ w 1 , and define Γ π M ,h ( · ) := { Γ 1 , π M ,h ( · ) , ..., Γ W, π M ,h ( · ) } . Then, conditioning on p ˜ ν s h -1 , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

A π is called NE policy if:

<!-- formula-not-decoded -->

where δ s w,n h =( · ) ∈ R ∑ w |S w | denotes a vector with 1 at the ( w,n ) -th value and 0 at the others. In the equalities, we use the fact that ¯ M and M share the transition function.

Besides, conditioning on p ˜ ν s h -1 , for any ˜ w ∈ [ W ] we can treat { s ˜ w,n h } n ∈ [ N ˜ w ] as i.i.d. samples according to distribution E [ p ˜ w, ˜ ν s h | p ˜ ν s h -1 ] . By applying Lem. I.6 for a fixed p ˜ ν s h -1 , ∀ w ∈ [ W ] , h ∈ [ H ] , for any ε ∈ (0 , 1) and δ ∈ (0 , 1) , as long as N ˜ w ≥ min { 8 W 2 S ˜ w ε 2 , 8 W 2 ε 2 log 2 W δ } holds for any ˜ w ∈ [ W ] , we have:

<!-- formula-not-decoded -->

Note that the number of possible values of p ˜ ν s h -1 can be upper bounded by ∏ w ∈ [ W ] ( N w ) S w . We define event E := {∀ h ∈ [ H ] , w ∈ [ W ] , ∥ p ˜ ν s h -Γ π M ,h -1 ( p ˜ ν s h -1 ) ∥ 1 &lt; ε } . By applying a union bound over h, w and all possible p ˜ ν s h -1 , we have:

<!-- formula-not-decoded -->

Density Error Decomposition The following discussion are based on the event E . Recall we use µ π h := { µ 1 , π h , ..., µ W, π h } to denote the density induced by π in MT-MFG. Then we have:

<!-- formula-not-decoded -->

Upper Bound of Approximation Error Recall the definition of value functions in Def. H.1 and Def. H.2. We focus on the ( w, 1) -agent which takes a potentially deviated policy ˜ π w while the others do not, and we are interested in provide an upper bound for the value difference J ( w, 1) ¯ M ( ˜ ν ) -J ( w, 1) M ( π -w ◦ ˜ π w ; π ) , which will be useful to characterize the sub-optimality of lifted policy ν .

We start from step h = H , following the choice of N w in Eq. (23),

<!-- formula-not-decoded -->

w,

1)

(

h

s

(

;

s

w,

h

For h &lt; H , we have:

E

¯

[

V

(

w,

1)

,

M

E

M

-

[

r

¯

M

w

,

˜

ν

∑

(

w,

h

,h

(

s

w

h

P

1)

+1

s

h

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we use P ¯ M ( s h +1 | s h , a h ) := ∏ w ∈ [ W ] ∏ n ∈ [ N ] P w h ( s ( w, 1) h +1 | s ( w, 1) h , a ( w, 1) h , p s h ) to denote the dynamics in MF-SAG. Therefore, for h = 1 , note that ¯ M and M have the same initial distribution, and we have:

<!-- formula-not-decoded -->

Given an ˜ ε NE policy in M , denoted by π , consider the lifted policy ν and a deviation policy ˜ ν agrees with ν except that it takes some ˜ π w for agent with index ( w, 1) . By choosing δ = ε 0 8( L r + L T ) H and ε = ε 0 / ( 4( L r + L T ) H (1+ L T ) H -1 L T ) , we have:

<!-- formula-not-decoded -->

˜

To satisfy the requirements in δ and ε 0 , we need:

<!-- formula-not-decoded -->

(

(

w,

h

(

s

ν

˜

1)

, a

(

w,

1)

h

+1

|

s

¯

,

ν

˜

1)

(

h

)

;

p

w,

1)

h

ν

s

-

˜

h

)

V

(

w,

M

-

(

w,

1)

,h

r

1)

h

w

h

,

(

s

π

h

µ

,

π

(

-

w

w,

h

)

V

◦

1)

w

˜

(

s

(

(

w,

1)

h

w,

1)

h

π

, a

(

w,

1)

M

,

π

,h

+1

-

w

;

µ

;

π

h

µ

◦

π

w

(

π

h

)]

) +

∑

s

(

h

+1

w,

h

1)

+1

;

µ

s

, a

=

˜

P

M

π

h

+1

¯

(

s

)]

h

+1

|

s

h

,

a

h

)

V

(

w,

¯

M

1)

,h

˜

ν

,

+1

(

(

w,

1)

h

+1

s

;

s

h

+1

)

## I. Basic Lemma

## I.1. Lemma from (Huang et al., 2024)

Lemma I.1 (Lem. D.4 in (Huang et al., 2024)) . Let X 1 , X 2 , ... be a sequence of random variable taking value in [0 , C ] for some C ≥ 1 . Define F k = σ ( X 1 , .., X k -1 ) and Y k = E [ X k |F k ] for k ≥ 1 . For any δ &gt; 0 , we have:

<!-- formula-not-decoded -->

Lemma I.2 (Lem. 4.6 in (Huang et al., 2024)) . Under Assump. B, given two arbitrary model M and ˜ M , and two policies π and π , we have:

˜

<!-- formula-not-decoded -->

## I.2. Other Lemma

Lemma I.3 (Density Difference Lemma) . Given arbitrary Multi-Type Mean-Field MDPs M and M ′ , and two arbitrary policies π and π ′ , for any h ∈ [ H ] , we have:

<!-- formula-not-decoded -->

Proof. For any w ∈ [ W ] , we have:

<!-- formula-not-decoded -->

≤∥ µ w, π M ,h -1 -µ w, π ′ M ′ ,h -1 ∥ 1 + d ∞ , 1 ( π , π ′ ) + E π , M ( π ) [ ∥ P w M ,h ( ·| s w h -1 , a w h -1 , µ π M ,h -1 ) -P w M ′ ,h ( ·| s w h -1 , a w h -1 , µ π ′ M ′ ,h -1 ) ∥ 1 ] By repeating the above discussion for every w ∈ [ W ] , we have:

<!-- formula-not-decoded -->

□

Lemma I.4. Given two model M and M ′ and a policy π , for any h ∈ [ H ] , w ∈ [ W ] , we have:

<!-- formula-not-decoded -->

Besides, under Assump. B, we have:

<!-- formula-not-decoded -->

Proof. By applying Lem. I.3 to the case when π = π ′ , and combining with Assump. D, we finish the proof. □

Lemma I.5. Given two model M and M ′ , and two arbitrary policies π and π ′ , for any h ∈ [ H ] , we have:

<!-- formula-not-decoded -->

Moreover, as a special case when π = π ′ , we have:

<!-- formula-not-decoded -->

Besides, under Assump. B, we have:

<!-- formula-not-decoded -->

Proof. The proof is simply completed by setting W = 1 in Lem. I.4.

□

Lemma I.6 (Concentration w.r.t. l 1 -distance) . Given a discrete domain X and a distribution p on X , suppose we draw N i.i.d. samples { x n } n ∈ [ N ] from p and provide an estimation ̂ p ∈ ∆( X ) with ̂ p ( x ) = 1 N ∑ N n =1 δ ( x n = x ) , then for any δ ∈ (0 , 1) and ε &gt; 0 , as long as N ≥ max { 2 |X| ε 2 , 2 ε 2 log 2 δ } , we have:

<!-- formula-not-decoded -->

Proof. We first provide an upper bound for E [ ∥ p -̂ p ∥ 1 ] :

<!-- formula-not-decoded -->

where we use the fact that ̂ p ( x ) is a Bernoulli random variable with mean p ( x ) and variance 1 N p ( x )(1 -p ( x )) .

Next, and note that deviation of any x n will only result in 2 /N deviation of ∥ p -̂ p ∥ 1 . By McDiarmid's inequality, for any ε , we have:

<!-- formula-not-decoded -->

By assigning appropriate values for N , we finish the proof.

□

Lemma I.7. [Model Difference Lemma] For any policies ˜ π , π and π ′ , and any bounded functions f 1 , f 2 , ..., f H ∈ { f | f : S × A → [0 , 1] } ,

(i) Given any two MF-MDPs M and M ′ , we have:

<!-- formula-not-decoded -->

(ii) Given any two PAMs ¨ M and ¨ M ′ , we have:

<!-- formula-not-decoded -->

Proof. We first proof (ii). We use µ π ¨ M ( π ′ ) ,h to denote the density induced by π in model ¨ M given π ′ as the reference policy.

<!-- formula-not-decoded -->

Therefore,

<!-- formula-not-decoded -->

The proof for (i) can be directly obtained by replacing ¨ P ¨ M,h ( ·|· , · , π ) and ¨ P ¨ M ′ ,h ( ·|· , · , π ′ ) with P M,h ( ·|· , · , µ π M,h ) and P M ′ ,h ( ·|· , · , µ π M ′ ,h ) .

□

Lemma I.8. Given three arbitrary models M, ˜ M, ¯ M , and two arbitrary policies π, ˜ π , we have:

<!-- formula-not-decoded -->

Proof. By applying Lem. I.7 with f h ( s h , a h ) = ∥ P ˜ M,h ( ·| s h , a h , µ π ˜ M,h ) -P ¯ M,h ( ·| s h , a h , µ π ¯ M,h ) ∥ 1 , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

□

## J. Details of Experiments

## J.1. Algorithm Design

In the following, we provide the missing algorithm details for Sec. 6.

## Algorithm 7: A Heuristic Oracle-Efficient NE Finding

- 1 Input : Model Class M ; Accuracy level ε , ε, ¯ ε ; Confidence level δ ; Batch size T

```
0 ˜ 2 M 1 ←M ; δ 0 ← δ log 2 |M| +1 3 ∀ M ∈ M 1 , compute (one of) its NE policy π NE M ← NE Compute ( M ) . 4 for k = 1 , 2 , ..., do 5 if ∃ M k ∈ M k , s.t. max ˜ M ∈M k |B ε 0 π NE Mk ( ˜ M, M k ) | ≤ |M k | 2 then 6 M k +1 ← ModelElim Exp ( π NE M k , M k , ˜ ε, δ 0 , T ) . 7 end 8 else 9 M k ← arg max M ∈M k |B ε 0 π NE M ( M, M k ) | , 10 π NE Br ← π NE M k , 11 M k +1 ← ModelElim Exp ( π NE Br , M k , ˜ ε, δ 0 , T ) . 12 if M k ∈ M k +1 then return π NE M k ; 13 end 14 end
```

Here ModelElim Exp is the same algorithm as Alg. 2 except that we replac Line 4 with:

<!-- formula-not-decoded -->

In another word, we only consider policies from Π NE := { π NE M } M ∈ ¯ M , including the NE policies of models in ¯ M .

## J.2. Experiment Setup

Environments We consider the linear style MFG, such that

<!-- formula-not-decoded -->

where ϕ ∈ R d ϕ and G ( · ) ∈ R d ϕ × d ψ are known but ψ ∈ R d ψ are unknown. Note that our environment is different from linear model in Prop. D.12, where features are self-normalized. We choose H = 3 , S = 100 , A = 50 and d ϕ = d ψ = 5 , where the number of states and actions is much larger than the feature dimension. We consider a model set with |M| = 200 .

To construct the environment, for each h , we first generate a random matrix Φ h ∈ R SA × d ϕ using as feature ϕ ( s h , a h ) , and generate another random matrix U h ∈ R S × d ϕ d ψ , and define the function G h ( µ h ) by

<!-- formula-not-decoded -->

After that, we generate 200 random matrices { Ψ i h } i ∈ [200] with Ψ i h ∈ R d ψ × S as the next feature function. Then, the model class is specified by M := { (Φ h , U h , Ψ i h ) } . In order to make the model elimination process more challenging, { Ψ i h } i =2 ,..., 200 is generated by randomly perturbing from Ψ 1 h , i.e.:

<!-- formula-not-decoded -->

where ˜ Ψ i h is a random matrix independent w.r.t. Ψ 1 h and β ∼ Uniform (0 , 0 . 1) . In this way, the difference between models in M will be small and harder to distinguish.

Training Procedure We construct 5 model classes M 1 , ..., M 5 with different Φ , U to increase the randomness in experiments. For each model class M i , we repeat 5 trials, where in each trial, we first randomly select one model from M i as the true model, and run Alg. 7 for model elimination.

We set ε =1e-3, i.e. we want to find a 1e-3-approximate NE. Besides, we set batch size T = 50 , δ = 0 . 001 . For the NE Oracle in Alg. 7, we implement it by repeatedly update

<!-- formula-not-decoded -->

where α = 0 . 02 , and BestReponse ( π i ; M ) return the policy maximizing the NE gap of π i in M . We stop the update process as long as E NE M ( π i ) ≤ 5e-4.

Experiments Results We provide our experiment results in Fig. 2. On the LHS, we report the number of uneliminated models verses the number of trajectories consumed, and as we can see, our algorithm can eliminate unqualified models very quickly. The total consumed trajectories is much less than the number of states actions SA = 100 ∗ 50 = 5000 .

On the RHS 7 , we report the normalized worst case NE Gap w.r.t. the remaining models. At each iteration t , we compute the NE gap for every uneliminated model's NE policy, and pick out the largest one denoted as Gap t . The normalized gap is defined to be Gap t Gap 0 , where the normalization term Gap 0 is the maximal NE gap at the beginning of the algorithm, i.e. the worst NE gap without starting the algorithm. As we can see, our algorithm can gradually eliminate inaccurate models and return the (approximate) NE.

Figure 2. Experiment results in linear style MFG. We report the number of remaining models and the normalized maximal NE Gap by the NE policies of remaining models during the model elimination process. Error bars correspond to 95% confidence intervals.

<!-- image -->

7 In the RHS sub-plot of Fig. 2, we set the normalized NE Gap to 0 as long as it is lower than 1e-3