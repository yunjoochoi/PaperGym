## Online Reinforcement Learning in Stochastic Games

Chen-Yu Wei

Institute of Information Science Academia Sinica, Taiwan bahh723@iis.sinica.edu.tw

Chi-Jen Lu

Institute of Information Science Academia Sinica, Taiwan cjlu@iis.sinica.edu.tw

## Abstract

We study online reinforcement learning in average-reward stochastic games (SGs). An SG models a two-player zero-sum game in a Markov environment, where state transitions and one-step payoffs are determined simultaneously by a learner and an adversary. We propose the UCSG algorithm that achieves a sublinear regret compared to the game value when competing with an arbitrary opponent. This result improves previous ones under the same setting. The regret bound has a dependency on the diameter , which is an intrinsic value related to the mixing property of SGs. If we let the opponent play an optimistic best response to the learner, UCSG finds an ε -maximin stationary policy with a sample complexity of ˜ O ( poly (1 /ε )) , where ε is the gap to the best policy.

## 1 Introduction

Many real-world scenarios (e.g., markets, computer networks, board games) can be cast as multiagent systems. The framework of Multi-Agent Reinforcement Learning (MARL) targets at learning to act in such systems. While in traditional reinforcement learning (RL) problems, Markov decision processes (MDPs) are widely used to model a single agent's interaction with the environment, stochastic games (SGs, [32]), as an extension of MDPs, are able to describe multiple agents' simultaneous interaction with the environment. In this view, SGs are most well-suited to model MARL problems [24].

In this paper, two-player zero-sum SGs are considered. These games proceed like MDPs, with the exception that in each state, both players select their own actions simultaneously 1 , which jointly determine the transition probabilities and their rewards . The zero-sum property restricts that the two players' payoffs sum to zero. Thus, while one player (Player 1) wants to maximize his/her total reward, the other (Player 2) would like to minimize that amount. Similar to the case of MDPs, the reward can be discounted or undiscounted, and the game can be episodic or non-episodic.

In the literature, SGs are typically learned under two different settings, and we will call them online and offline settings, respectively. In the offline setting, the learner controls both players in a centralized manner, and the goal is to find the equilibrium of the game [33, 21, 30]. This is also known as finding the worst-case optimality for each player (a.k.a. maximin or minimax policy). In this case, we care about the sample complexity , i.e., how many samples are required to estimate the worst-case optimality such that the error is below some threshold. In the online setting, the learner

1 Turn-based SGs, like Go, are special cases: in each state, one player's action set contains only a null action.

## Yi-Te Hong

Institute of Information Science Academia Sinica, Taiwan ted0504@iis.sinica.edu.tw controls only one of the players, and plays against an arbitrary opponent [24, 4, 5, 8, 31]. In this case, we care about the learner's regret , i.e., the difference between some benchmark measure and the learner's total reward earned in the learning process. This benchmark can be defined as the total reward when both players play optimal policies [5], or when Player 1 plays the best stationary response to Player 2 [4]. Some of the above online-setting algorithms can find the equilibrium simply through self-playing.

Most previous results on offline sample complexity consider discounted SGs. Their bounds depend heavily on the chosen discount factor [33, 21, 30, 31]. However, as noted in [5, 19], the discounted setting might not be suitable for SGs that require long-term planning, because only finite steps are relevant in the reward function it defines. This paper, to the best of our knowledge, is the first to give an offline sample complexity bound of order ˜ O ( poly (1 /ε )) in the average-reward (undiscounted and non-episodic) setting, where ε is the error parameter. A major difference between our algorithm and previous ones is that the two players play asymmetric roles in our algorithm: by focusing on finding only one player's worst-case optimal policy at a time, the sampling can be rather efficient. This resembles but strictly extends [13]'s methods in finding the maximin action in a two-stage game.

In the online setting, we are only aware of [5]'s R-MAX algorithm that deals with average-reward SGs and provides a regret bound. Considering a similar scenario and adopting the same regret definition, we significantly improve their bounds (see Appendix A for details). Another difference between our algorithm and theirs is that ours is able to output a currently best stationary policy at any stage in the learning process, while theirs only produces a T ε -step fixed-horizon policy for some input parameter T ε . The former could be more natural since the worst-case optimal policy is itself a stationary policy.

The techniques used in this paper are most related to RL for MDPs based on the optimism principle [2, 19, 9] (see Appendix A). The optimism principle built on concentration inequalities automatically strikes a balance between exploitation and exploration, eliminating the need to manually adjust the learning rate or the exploration ratio. However, when importing analysis from MDPs to SGs, we face the challenge caused by the opponent's uncontrollability and non-stationarity. This prevents the learner from freely exploring the state space and makes previous analysis that relies on stationary distribution's perturbation analysis [2] useless. In this paper, we develop a novel way to replace the opponent's non-stationary policy with a stationary one in the analysis (introduced in Section 5.1), which facilitates the use of techniques based on perturbation analysis. We hope that this technique can benefit future analysis concerning non-stationary agents in MARL.

One related topic is the robust MDP problem [29, 17, 23]. It is an MDP where some state-action pairs have adversarial rewards and transitions. It is often assumed in robust MDP that the adversarial choices by the environment are not directly observable by the Player, but in our SG setting, we assume that the actions of Player 2 can be observed. However, there are still difficulties in SG that are not addressed by previous works on robust MDP.

Here we compare our work to [23], a recent work on learning robust MDP. In their setting, there are adversarial and stochastic state-action pairs, and their proposed OLRM2 algorithm tries to distinguish them. Under the scenario where the environment is fully adversarial, which is the counterpart to our setting, the worst-case transitions and rewards are all revealed to the learner, and what the learner needs to do is to perform a maximin planning. In our case, however, the worst-case transitions and rewards are still to be learned, and the opponent's arbitrary actions may hinder the learner to learn this information. We would say that the contribution of [23] is orthogonal to ours.

Other lines of research that are related to SGs are on MDPs with adversarially changing reward functions [11, 27, 28, 10] and with adversarially changing transition probabilities [35, 1]. The assumptions in these works have several differences with ours, and therefore their results are not comparable to our results. However, they indeed provide other viewpoints about learning in stochastic games.

## 2 Preliminaries

Game Models and Policies. A SG is a 4-tuple M = ( S , A , r, p ) . S denotes the state space and A = A 1 × A 2 the players' joint action space. We denote S = |S| and A = |A| . The game starts from an initial state s 1 . Suppose at time t the players are at state s t . After the players play the joint actions ( a 1 t , a 2 t ) , Player 1 receives the reward r t = r ( s t , a 1 t , a 2 t ) ∈ [0 , 1] from Player 2, and both players visit state s t +1 following the transition probability p ( ·| s t , a 1 t , a 2 t ) . For simplicity, we consider deterministic rewards as in [3]. The extension to stochastic case is straightforward. We shorten our notation by a := ( a 1 , a 2 ) or a t := ( a 1 t , a 2 t ) , and use abbreviations such as r ( s t , a t ) and p ( ·| s t , a t ) .

Without loss of generality, players are assumed to determine their actions based on the history. A policy π at time t maps the history up to time t , H t = ( s 1 , a 1 , r 1 , ..., s t ) ∈ H t , to a probability distribution over actions. Such policies are called history-dependent policies, whose class is denoted by Π HR . On the other hand, a stationary policy, whose class is denoted by Π SR , selects actions as a function of the current state. For either class, joint policies ( π 1 , π 2 ) are often written as π .

Average Return and the Game Value. Let the players play joint policy π . Define the T -step total reward as R T ( M,π,s ) := ∑ T t =1 r ( s t , a t ) , where s 1 = s , and the average reward as ρ ( M,π,s ) := lim T →∞ 1 T [ R T ( M,π,s )] , whenever the limit exists. In fact, the game value exists 2 [26]:

/BX If ρ ( M,π,s ) or ρ ∗ ( M,s ) does not depend on the initial state s , we simply write ρ ( M,π ) or ρ ∗ ( M ) .

/BX

<!-- formula-not-decoded -->

The Bias Vector. For a stationary policy π , the bias vector h ( M,π, · ) is defined, for each coordinate s , as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The bias vector satisfies the Bellman equation: ∀ s ∈ S ,

where r ( s, π ) := a ∼ π ( ·| s ) [ r ( s, a )] and p ( s ′ | s, π ) := a ∼ π ( ·| s ) [ p ( s ′ | s, a )] .

On the other hand, sp( h ( M,π, · )) is closely related to the mean first passage time under the Markov chain induced by M and π . Actually we have sp( h ( M,π, · )) ≤ T π ( M ) := max s,s ′ T π s → s ′ ( M ) , where T π s → s ′ ( M ) denotes the expected time to reach state s ′ starting from s when the model is M and the player(s) follow the (joint) policy π . This fact is intuitive, and the proof can be seen at Remark M.1.

/BX /BX The vector h ( M,π, · ) describes the relative advantage among states under model M and (joint) policy π . The advantage (or disadvantage) of state s compared to state s ′ under policy π is defined as the difference between the accumulated rewards with initial states s and s ′ , which, from (1), converges to the difference h ( M,π,s ) -h ( M,π,s ′ ) asymptotically. For the ease of notation, the span of a vector v is defined as sp( v ) := max i v i -min i v i . Therefore if a model, together with any policy, induces large sp ( h ) , then this model will be difficult to learn because visiting a bad state costs a lot in the learning process. As shown in [3] for the MDP case, the regret has an inevitable dependency on sp( h ( M,π ∗ , · )) , where π ∗ is the optimal policy.

Notations. In order to save space, we often write equations in vector or matrix form. We use vectors inequalities: if u, v ∈ R n , then u ≤ v ⇔ u i ≤ v i ∀ i = 1 , ..., n . For a general matrix game with matrix G of size n × m , we denote the value of the game as val G := max p ∈ ∆ n min q ∈ ∆ m p /latticetop Gq =

<!-- formula-not-decoded -->

min q ∈ ∆ m max p ∈ ∆ n p /latticetop Gq , where ∆ k is the probability simplex of dimension k . In SGs, given the estimated value function u ( s ′ ) ∀ s ′ , we often need to solve the following matrix game equation:

and this is abbreviated with the vector form v = val { r + Pu } . We also use solve 1 G and solve 2 G to denote the optimal solutions of p and q . In addition, the indicator function is denoted by /BD {·} or {·} .

/BD 2 Unlike in one-player MDPs, the sup and inf in the definition of ρ ∗ ( M,s ) are not necessarily attainable. Moreover, players may not have stationary optimal policies.

## 3 Problem Settings and Results Overview

We assume that the game proceeds for T steps. In order to have meaningful regret bounds (i.e., sublinear to T ), we must make some assumptions to the SG model itself. Our two different assumptions are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Why we make these assumptions is as follows. Consider an SG model where the opponent (Player 2) has some way to lock the learner (Player 1) to some bad state. The best strategy for the learner might be to totally avoid, if possible, entering that state. However, in the early stage of the learning process, the learner won't know this, and he/she will have a certain probability to visit that state and get locked. This will cause linear regret to the learner. Therefore, we assume the following: whatever policy the opponent executes, the learner always has some way to reach any state within some bounded time. This is essentially our Assumption 2.

Assumption 1 is the stronger one that actually implies that under any policies executed by the players (not necessarily stationary, see Remark M.2), every state is visited within an average of D steps. We find that under this assumption, the asymptotic regret can be improved. This assumption also has a sense similar to those required for Q-learning-type algorithms' convergence: they require that every state be visited infinitely often. See [18] for example.

These assumptions define some notion of diameters that are specific to the SG model. It is known that under Assumption 1 or Assumption 2, both players have optimal stationary policies, and the game value is independent of the initial state. Thus we can simply write ρ ∗ ( M,s ) as ρ ∗ ( M ) . For a proof of these facts, please refer to Theorem E.1 in the appendix.

## 3.1 Two Settings and Results Overview

We focus on training Player 1 and discuss two settings. In the online setting, Player 1 competes with an arbitrary Player 2. The regret is defined as

<!-- formula-not-decoded -->

In the offline setting, we control both Player 1 and Player 2's actions, and find Player 1's maximin policy. The sample complexity is defined as

<!-- formula-not-decoded -->

where π 1 t is a stationary policy being executed by Player 1 at time t . This definition is similar to those in [20, 19] for one-player MDPs. By the definition of L ε , if we have an upper bound for L ε and run the algorithm for T &gt; L ε steps, there is some t such that π 1 t is ε -optimal. We will explain how to pick this t in Section 7 and Appendix L.

It turns out that we can use almost the same algorithm to handle these two settings. Since learning in the online setting is more challenging, from now on we will mainly focus on the online setting, and leave the discussion about the offline setting at the end of the paper. Our results can be summarized by the following two theorems.

<!-- formula-not-decoded -->

= ˜ √

T O

## 4 Upper Confidence Stochastic Game Algorithm (UCSG)

3 We write, 'with high probability, g = ˜ O ( f ) ' or ' w.h.p. , g = ˜ O ( f ) ' to indicate 'with probability ≥ 1 -δ, g = f 1 O ( f ) + f 2 ', where f 1 , f 2 are some polynomials of log D, log S, log A, log T, log(1 /δ ) .

## Algorithm 1 UCSG

Input:

S , A = A 1 ×A 2 , T .

Initialization:

t = 1 .

for phase k = 1 , 2 , ... do

t k = t.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

3. Optimistic planning: ( M 1 k , π 1 k ) = MAXIMIN-EVI ( M k , γ k ) , where γ k := 1 / √ t k . 4. Execute policies:
2. ∑ /BD 2. Update the confidence set: M k = { ˜ M : ∀ s, a, ˜ p ( ·| s, a ) ∈ P k ( s, a ) } , where P k ( s, a ) := CONF 1 (ˆ p k ( ·| s, a ) , n k ( s, a )) ∩ CONF 2 (ˆ p k ( ·| s, a ) , n k ( s, a )) .

repeat

Set v k ( s t , a t ) = v k ( s t , a t ) + 1 and t = t +1 .

Draw a 1 t ∼ π 1 k ( ·| s t ) ; observe the reward r t and the next state s t +1 .

until ∃ ( s, a ) such that v k ( s, a ) = n k ( s, a )

## end for

## Definitions of confidence regions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The Upper Confidence Stochastic Game algorithm (UCSG) (Algorithm 1) extends UCRL2 [19], using the optimism principle to balance exploitation and exploration. It proceeds in phases (indexed by k ), and only changes the learner's policy π 1 k at the beginning of each phase. The length of each phase is not fixed a priori, but depends on the statistics of past observations.

In the beginning of each phase k , the algorithm estimates the transition probabilities using empirical frequencies ˆ p k ( ·| s, a ) observed in previous phases (Step 1). With these empirical frequencies, it can then create a confidence region P k ( s, a ) for each transition probability. The transition probabilities lying in the confidence regions constitute a set of plausible stochastic game models M k , where the true model M belongs to with high probability (Step 2). Then, Player 1 optimistically picks one model M 1 k from M k , and finds the optimal (stationary) policy π 1 k under this model (Step 3). Finally, Player 1 executes the policy π 1 k for a while until some ( s, a ) -pair's number of occurrences is doubled during this phase (Step 4). The count v k ( s, a ) records the number of steps the ( s, a ) -pair is observed in phase k ; it is reset to zero in the beginning of every phase.

In Step 3, to pick an optimistic model and a policy is to pick M 1 k ∈ M k and π 1 k ∈ Π SR such that ∀ s ,

<!-- formula-not-decoded -->

where γ k denotes the error parameter for MAXIMIN-EVI. The LHS of (2) is well-defined because Player 2 has stationary optimal policy under the MDP induced by M 1 k and π 1 k . Roughly speaking, (2) says that min π 2 ρ ( M 1 k , π 1 k , π 2 , s ) should approximate max ˜ M ∈M k ,π 1 min π 2 ρ ( ˜ M,π 1 , π 2 , s ) by an error

no more than γ k . That is, ( M 1 k , π 1 k ) are picked optimistically in M k × Π SR considering the most adversarial opponent.

## 4.1 Extended SG and Maximin-EVI

The calculation of M 1 k and π 1 k involves the technique of Extended Value Iteration (EVI), which also appears in [19] as a one-player version.

Consider the following SG, named M + . Let the state space S and Player 2's action space A 2 remain the same as in M . Let A 1+ , p + ( ·|· , · , · ) , r + ( · , · , · ) be Player 1's action set, the transition kernel, and the reward function of M + , such that for any a 1 ∈ A 1 and a 2 ∈ A 2 and an admissible transition probability ˜ p ( ·| s, a 1 , a 2 ) ∈ P k ( s, a 1 , a 2 ) , there is an action a 1+ ∈ A 1+ such that p + ( ·| s, a 1+ , a 2 ) = ˜ p ( ·| s, a 1 , a 2 ) and r + ( s, a 1+ , a 2 ) = r ( s, a 1 , a 2 ) . In other words, Player 1 selecting an action in A 1+ is equivalent to selecting an action in A 1 and simultaneously selecting an admissible transition probability in the confidence region P k ( · , · ) .

Suppose that M ∈ M k , then the extended SG M + satisfies Assumption 2 because the true model M is embedded in M + . By Theorem E.1 in Appendix E, it has a constant game value ρ ∗ ( M + ) independent of the initial state, and satisfies Bellman equation of the form val { r + Pf } = ρ · e + f , for some bounded function f ( · ) , where e stands for the all-one constant vector. With the above conditions, we can use value iteration with Schweitzer transform (a.k.a. aperiodic transform)[34] to solve the optimal policy in the extended EG M + . We call it MAXIMIN-EVI. For the details of MAXIMIN-EVI, please refer to Appendix F. We only summarize the result with the following Lemma.

Lemma 4.1. Suppose the true model M ∈ M k , then the estimated model M 1 k and stationary policy π 1 k output by MAXIMIN-EVI in Step 3 satisfy

<!-- formula-not-decoded -->

Before diving into the analysis under the two assumptions, we first establish the following fact.

Lemma 4.2. With high probability, the true model M ∈ M k for all phases k .

It is proved in Appendix D. With Lemma 4.2, we can fairly assume M ∈ M k in most of our analysis.

## 5 Analysis under Assumption 1

In this section, we import analysis techniques from one-player MDPs [2, 19, 22, 9]. We also develop some techniques that deal with non-stationary opponents.

We model Player 2's behavior in the most general way, i.e., assuming it using a history-dependent randomized policy. Let H t = ( s 1 , a 1 , r 1 , ..., s t -1 , a t -1 , r t -1 , s t ) ∈ H t be the history up to s t , then we assume π 2 t to be a mapping from H t to a distribution over A 2 . We will simply write π 2 t ( · ) and hide its dependency on H t inside the subscript t . A similar definition applies to π 1 t ( · ) . With abuse of notations, we denote by k ( t ) the phase where step t lies in, and thus our algorithm uses policy π 1 t ( · ) = π 1 k ( t ) ( ·| s t ) . The notations π 1 t and π 1 k are used interchangeably. Let T k := t k +1 -t k be the length of phase k . We decompose the regret in phase k in the following way:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

in which we define

<!-- formula-not-decoded -->

where ¯ π 2 k is some stationary policy of Player 2 which will be defined later. Since the actions of Player 2 are arbitrary, ¯ π 2 k is imaginary and only exists in analysis. Note that under Assumption 1, any stationary policy pair over M induces an irreducible Markov chain, so we do not need to specify the initial states for ρ ( M,π 1 k , ¯ π 2 k ) in (3). Among the four terms, Λ (2) k is clearly non-positive, and Λ (1) k , by optimism, can be bounded using Lemma 4.1. Now remains to bound Λ (3) k and Λ (4) k .

5.1 Bounding ∑ k Λ (3) k and ∑ k Λ (4) k

The Introduction of ¯ π 2 k . Λ (3) k and Λ (4) k involve the artificial policy ¯ π 2 k , which is a stationary policy that replaces Player 2's non-stationary policy in the analysis. This replacement costs some constant regret but facilitates the use of perturbation analysis in regret bounding. The selection of ¯ π 2 k is based on the principle that the behavior (e.g., total number of visits to some ( s, a ) ) of the Markov chain induced by M,π 1 k , ¯ π 2 k should be close to the empirical statistics. Intuitively, ¯ π 2 k can be defined as

<!-- formula-not-decoded -->

/BD Note two things, however. First, since we need the actual trajectory in defining this policy, it can only be defined after phase k has ended. Second, ¯ π 2 k can be undefined because the denominator of (4) can be zero. However, this will not happen in too many steps. Actually, we have

Lemma 5.1. ∑ k T k { ¯ π 2 k not well-defined }≤ ˜ O ( DS 2 A ) with high probability.

/BD Before describing how we bound the regret with the help of ¯ π 2 k and the perturbation analysis, we establish the following lemma:

Lemma 5.2. We say the transition probability at time step t is ε -accurate if | p 1 k ( s ′ | s t , π t ) -p ( s ′ | s t , π t ) | ≤ ε ∀ s ′ where p 1 k denotes the transition kernel of M 1 k . We let B t ( ε ) = 1 if the transition probability at time t is ε -accurate; otherwise B t ( ε ) = 0 . Then for any state s , with high probability, ∑ T t =1 s t = s B t ( ε )=0 ≤ ˜ O ( A/ε 2 ) .

/BD /BD Now we are able to sketch the logic behind our proofs. Let's assume that ¯ π 2 k models π 2 k quite well, i.e., the expected frequency of every state-action pair induced by M,π 1 k , ¯ π 2 k is close to the empirical frequency induced by M,π 1 k , π 2 k . Then clearly, Λ (4) k is close to zero in expectation. The term Λ (3) k now becomes the difference of average reward between two Markov reward processes with slightly different transition probabilities. This term has a counterpart in [19] as a single-player version. Using similar analysis, we can prove that the dominant term of Λ (3) k is proportional to sp( h ( M 1 k , π 1 k , ¯ π 2 k , · )) . In the single-player case, [19] can directly claim that sp( h ( M 1 k , π 1 k , · )) ≤ D (see their Remark 8), but unfortunately, this is not the case in the two-player version. 4

To continue, we resort to the perturbation analysis for the mean first passage times (developed in Appendix C). Lemma 5.2 shows that M 1 k will not be far from M for too many steps. Then Theorem C.9 in Appendix C tells that if M 1 k are close enough to M , T π 1 k , ¯ π 2 k ( M 1 k ) can be bounded by 2 T π 1 k , ¯ π 2 k ( M ) . As Remark M.1 implies that sp( h ( M 1 k , π 1 k , ¯ π 2 k , · )) ≤ T π 1 k , ¯ π 2 k ( M 1 k ) and Assumption 1 guarantees that T π 1 k , ¯ π 2 k ( M ) D , we have sp( h ( M 1 , π 1 , ¯ π 2 , )) T π 1 k , ¯ π 2 k ( M 1 ) 2 T π 1 k , ¯ π 2 k ( M ) 2 D .

≤ k k k · ≤ k ≤ ≤

The above approach leads to Lemma 5.3, which is a key in our analysis. We first define some notations. Under Assumption 1, any pair of stationary policies induces an irreducible Markov chain, which has a unique stationary distribution. If the policy pair π = ( π 1 , π 2 ) is executed, we denote its stationary distribution by µ ( M,π 1 , π 2 , · ) = µ ( M,π, · ) . Besides, denote v k ( s ) := ∑ t k +1 -1 t = t k s t = s .

Lemma 5.3. ∑ k T k { phase k is not benign } ≤ ˜ O ( D 3 S 5 A ) with high probability.

/BD We say a phase k is benign if the following hold true: the true model M lies in M k , ¯ π 2 k is welldefined, sp( h ( M 1 k , π 1 k , ¯ π 2 k , · )) ≤ 2 D , and µ ( M,π 1 k , ¯ π 2 k , s ) ≤ 2 v k ( s ) T k ∀ s . We can show the following:

/BD Finally, for benign phases, we can have the following two lemmas.

/BD 4 The argument in [19] is simple: suppose that h ( M 1 k , π 1 k , s ) -h ( M 1 k , π 1 k , s ′ ) &gt; D , by the communicating assumption, there is a path from s ′ to s with expected time no more than D . Thus a policy that first goes from s ′ to s within D steps and then executes π 1 k will outperform π 1 k at s ′ . This leads to a contradiction. In two-player SGs, with a similar argument, we can also show that sp( h ( M 1 k , π 1 k , π 2 ∗ k , · )) ≤ D , where π 2 ∗ k is the best response to π 1 k under M 1 k . However, since Player 2 is uncontrollable, his/her policy π 2 k (or ¯ π 2 k ) can be quite different from π 2 ∗ k , and thus sp( h ( M 1 k , π 1 k , ¯ π 2 k , · )) ≤ D does not necessarily hold true.

Lemma 5.4. ∑ k Λ (4) k { ¯ π 2 k is well-defined }≤ ˜ O ( D √ ST + DSA ) with high probability.

Lemma 5.5. ∑ k Λ (3) k { phase k is benign } ≤ ˜ O ( DS √ AT + DS 2 A ) with high probability,

/BD Proof of Theorem 3.1. The regret proof starts from the decomposition of (3). Λ (1) k is bounded with the help of Lemma 4.1: ∑ k Λ (1) k ≤ ∑ k T k / √ t k = O ( √ T ) . ∑ k Λ (2) k ≤ 0 by definition. Then with Lemma 5.1, 5.3, 5.4, and 5.5, we can bound Λ (3) k and Λ (4) k by ˜ O ( D 3 S 5 A + DS √ AT ) .

## 6 Analysis under Assumption 2

In Section 5, the main ingredient of regret analysis lies in bounding the span of the bias vector, sp ( h ( M 1 k , π 1 k , ¯ π 2 k , · )) . However, the same approach does not work because under the weaker Assumption 2, we do not have a bound on the mean first passage time under arbitrary policy pairs. Hence we adopt the approach of approximating the average reward SG problem by a sequence of finite-horizon SGs: on a high level, first, with the help of Assumption 2, we approximate the T multiple of the original average-reward SG game value (i.e. the total reward in hindsight) with the sum of those of H -step episodic SGs; second, we resort to [9]'s results to bound the H -step SGs' sample complexity and translates it to regret.

Approximation by repeated episodic SGs. For the approximation, the quantity H does not appear in UCSG but only in the analysis. The horizon T is divided into episodes each with length H . Index episodes with i = 1 , ..., T /H , and denote episode i 's first time step by τ i . We say i ∈ ph ( k ) if all H steps of episode i lie in phase k . Define the H -step expected reward under joint policy π with initial state s as V H ( M,π,s ) := E [ ∑ H t =1 r t | a t ∼ π, s 1 = s ] . Now we decompose the regret in phase k as

where

<!-- formula-not-decoded -->

Lemma 6.1. By Azuma-Hoeffding's inequality, k ∆ (5) k ≤ ˜ O ( √ HT ) with high probability.

Here, π 2 i denotes Player 2's policy in episode i , which may be non-stationary. ∆ (6) k comes from the possible two incomplete episodes in phase k . ∆ (1) k is related to the tolerance level we set for the MAXIMIN-EVI algorithm: ∆ (1) k ≤ T k γ k = T k / √ t k . ∆ (2) k is an error caused by approximating an infinite-horizon SG by a repeated episodic H -step SG (with possibly different initial states). ∆ (3) k is clearly non-positive. It remains to bound ∆ (2) k , ∆ (4) k and ∆ (5) k .

<!-- formula-not-decoded -->

From sample complexity to regret bound. As the main contributor of regret, ∆ (4) k corresponds to the inaccuracy in the transition probability estimation. Here we largely reuse [9]'s results where they consider one-player episodic MDP with a fixed initial state distribution. Their main lemma states that the number of episodes in phases such that | V H ( M 1 k , π k , s 0 ) -V H ( M,π k , s 0 ) | &gt; ε will not exceed ˜ O ( H 2 S 2 A/ε 2 ) , where s 0 is their initial state in each episode. In other words, ∑ k T k H /BD {| V H ( M 1 k , π k , s 0 ) -V H ( M,π k , s 0 ) | &gt; ε } = ˜ O ( H 2 S 2 A/ε 2 ) . Note that their proof allows π k to be an arbitrarily selected non-stationary policy for phase k .

<!-- formula-not-decoded -->

We can directly utilize their analysis and we summarize it as Theorem K.1 in the appendix. While their algorithm has an input ε , this input can be removed without affecting bounds. This means that the PAC bounds holds for arbitrarily selected ε . With the help of Theorem K.1, we have

Proof of Theorem 3.2. With the decomposition (5) and the help of Lemma 6.1, 6.2, and 6.3, the regret is bounded by ˜ O ( TD H + S √ HAT + S 2 AH ) = ˜ O ( 3 √ DS 2 AT 2 ) by selecting H = max { D, 3 √ D 2 T/ ( S 2 A ) } .

<!-- formula-not-decoded -->

## 7 Sample Complexity of Offline Training

In Section 3.1, we defined L ε to be the sample complexity of Player 1's maximin policy. In our offline version of UCSG, in each phase k we let both players each select their own optimistic policy. After Player 1 has optimistically selected π 1 k , Player 2 then optimistically selects his policy π 2 k based on the known π 1 k . Specifically, the model-policy pair ( M 2 k , π 2 k ) is obtained by another extended value iteration on the extended MDP under fixed π 1 k , where Player 2's action set is extended. By setting the stopping threshold also as γ k , we have

<!-- formula-not-decoded -->

when value iteration halts. With this selection rule, we are able to obtain the following theorems.

Theorem 7.1. Under Assumption 1, UCSG achieves L ε = ˜ O ( D 3 S 5 A + D 2 S 2 A/ε 2 ) w.h.p.

Then UCSG achieves L ε = ˜ O ( DS 2 A/ε 3 ) w.h.p.

Theorem 7.2. Let Assumption 2 hold, and further assume that max s,s ′ max π 1 ∈ Π SR min π 2 ∈ Π SR T π 1 ,π 2 s → s ′ ( M ) ≤ D .

The algorithm can output a single stationary policy for Player 1 with the following guarantee: if we run the offline version of UCSG for T &gt; L ε steps, the algorithm can output a single stationary policy that is ε -optimal. We show how to output this policy in the proofs of Theorem 7.1 and 7.2.

## 8 Open Problems

In this work, we obtain the regret of ˜ O ( D 3 S 5 A + DS √ AT ) and ˜ O ( 3 √ DS 2 AT ) under different mixing assumptions. A natural open problem is how to improve these bounds on both asymptotic and constant terms. A lower bound of them can be inherited from the one-player MDP setting, which is Ω( √ DSAT ) [19].

Another open problem is that if we further weaken the assumptions to max s,s ′ min π 1 min π 2 T π 1 ,π 2 s → s ′ ≤ D , can we still learn the SG? We have argued that if we only have this assumption, in general we cannot get sublinear regret in the online setting. However, it is still possible to obtain polynomial-time offline sample complexity if the two players cooperate to explore the state-action space.

## Acknowledgments

We would like to thank all anonymous reviewers who have devoted their time for reviewing this work and giving us valuable feedbacks. We would like to give special thanks to the reviewer who reviewed this work's previous version in ICML; your detailed check of our proofs greatly improved the quality of this paper.

## References

- [1] Yasin Abbasi, Peter L Bartlett, Varun Kanade, Yevgeny Seldin, and Csaba Szepesvári. Online learning in markov decision processes with adversarially chosen transition probability distributions. In Advances in Neural Information Processing Systems , 2013.
- [2] Peter Auer and Ronald Ortner. Logarithmic online regret bounds for undiscounted reinforcement learning. In Advances in Neural Information Processing Systems , 2007.
- [3] Peter L Bartlett and Ambuj Tewari. Regal: A regularization based algorithm for reinforcement learning in weakly communicating mdps. In Proceedings of Conference on Uncertainty in Artificial Intelligence . AUAI Press, 2009.
- [4] Michael Bowling and Manuela Veloso. Rational and convergent learning in stochastic games. In International Joint Conference on Artificial Intelligence , 2001.
- [5] Ronen I Brafman and Moshe Tennenholtz. R-max-a general polynomial time algorithm for near-optimal reinforcement learning. Journal of Machine Learning Research , 2002.
- [6] Sébastien Bubeck and Aleksandrs Slivkins. The best of both worlds: Stochastic and adversarial bandits. In Conference on Learning Theory , 2012.
- [7] Grace E Cho and Carl D Meyer. Markov chain sensitivity measured by mean first passage times. Linear Algebra and Its Applications , 2000.
- [8] Vincent Conitzer and Tuomas Sandholm. Awesome: A general multiagent learning algorithm that converges in self-play and learns a best response against stationary opponents. Machine Learning , 2007.
- [9] Christoph Dann and Emma Brunskill. Sample complexity of episodic fixed-horizon reinforcement learning. In Advances in Neural Information Processing Systems , 2015.
- [10] Travis Dick, Andras Gyorgy, and Csaba Szepesvari. Online learning in markov decision processes with changing cost sequences. In Proceedings of International Conference of Machine Learning , 2014.
- [11] Eyal Even-Dar, Sham M Kakade, and Yishay Mansour. Online markov decision processes. Mathematics of Operations Research , 2009.
- [12] Awi Federgruen. On n-person stochastic games by denumerable state space. Advances in Applied Probability , 1978.
- [13] Aurélien Garivier, Emilie Kaufmann, and Wouter M Koolen. Maximin action identification: A new bandit framework for games. In Conference on Learning Theory , pages 1028-1050, 2016.
- [14] Arie Hordijk. Dynamic programming and markov potential theory. MC Tracts , 1974.
- [15] Jeffrey J Hunter. Generalized inverses and their application to applied probability problems. Linear Algebra and Its Applications , 1982.
- [16] Jeffrey J Hunter. Stationary distributions and mean first passage times of perturbed markov chains. Linear Algebra and Its Applications , 2005.
- [17] Garud N. Iyengar. Robust dynamic programming. Math. Oper. Res. , 30(2):257-280, 2005.
- [18] Tommi Jaakkola, Michael I Jordan, and Satinder P Singh. On the convergence of stochastic iterative dynamic programming algorithms. Neural computation , 1994.
- [19] Thomas Jaksch, Ronald Ortner, and Peter Auer. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research , 2010.
- [20] Sham Machandranath Kakade et al. On the sample complexity of reinforcement learning . PhD thesis, University of London London, England, 2003.

- [21] Michail G Lagoudakis and Ronald Parr. Value function approximation in zero-sum markov games. In Proceedings of Conference on Uncertainty in Artificial Intelligence . Morgan Kaufmann Publishers Inc., 2002.
- [22] Tor Lattimore and Marcus Hutter. Pac bounds for discounted mdps. In International Conference on Algorithmic Learning Theory . Springer, 2012.
- [23] Shiau Hong Lim, Huan Xu, and Shie Mannor. Reinforcement learning in robust markov decision processes. Math. Oper. Res. , 41(4):1325-1353, 2016.
- [24] Michael L Littman. Markov games as a framework for multi-agent reinforcement learning. In Proceedings of International Conference of Machine Learning , 1994.
- [25] A Maurer and M Pontil. Empirical bernstein bounds and sample variance penalization. In Conference on Learning Theory , 2009.
- [26] J-F Mertens and Abraham Neyman. Stochastic games. International Journal of Game Theory , 1981.
- [27] Gergely Neu, Andras Antos, András György, and Csaba Szepesvári. Online markov decision processes under bandit feedback. In Advances in Neural Information Processing Systems , 2010.
- [28] Gergely Neu, András György, and Csaba Szepesvári. The adversarial stochastic shortest path problem with unknown transition probabilities. In AISTATS , 2012.
- [29] Arnab Nilim and Laurent El Ghaoui. Robust control of markov decision processes with uncertain transition matrices. Math. Oper. Res. , 53(5):780-798, 2005.
- [30] Julien Perolat, Bruno Scherrer, Bilal Piot, and Olivier Pietquin. Approximate dynamic programmingfor two-player zero-sum markov games. In Proceedings of International Conference of Machine Learning , 2015.
- [31] HL Prasad, Prashanth LA, and Shalabh Bhatnagar. Two-timescale algorithms for learning nash equilibria in general-sum stochastic games. In Proceedings of the 2015 International Conference on Autonomous Agents and Multiagent Systems . International Foundation for Autonomous Agents and Multiagent Systems, 2015.
- [32] Lloyd S Shapley. Stochastic games. Proceedings of the National Academy of Sciences , 1953.
- [33] Csaba Szepesvári and Michael L Littman. Generalized markov decision processes: Dynamicprogramming and reinforcement-learning algorithms. In Proceedings of International Conference of Machine Learning , 1996.
- [34] J Van der Wal. Successive approximations for average reward markov games. International Journal of Game Theory , 1980.
- [35] Jia Yuan Yu and Shie Mannor. Arbitrarily modulated markov decision processes. In Proceedings of Conference on Decision and Control . IEEE, 2009.

## Contents (Appendix)

| A   | Previous Bounds for MDPs and SGs                                                                                              |   12 |
|-----|-------------------------------------------------------------------------------------------------------------------------------|------|
| B   | Inequalities                                                                                                                  |   13 |
| C   | Perturbation Bounds for Markov Chains                                                                                         |   13 |
|     | C.1 Perturbation Bounds for Stationary Distribution . . . . . . C.2 Perturbation Bounds for Mean First Passage Time . . . . . |   13 |
|     |                                                                                                                               |   14 |
| D   | Lemmas for Failing Events                                                                                                     |   16 |
| E   | Lemmas for Stationary Optimal Policies                                                                                        |   16 |
| F   | MAXIMIN-EVI and Its Convergence                                                                                               |   18 |
| G   | Proof of Lemma 5.2                                                                                                            |   19 |
|     | G.1 Proof of Lemma G.1 . . . . . . . . . . . . . . . . . . . .                                                                |   20 |
|     | G.2 Proof of Lemma G.2 . . . . . . . . . . . . . . . . . . . .                                                                |   21 |
| H   | Proofs of Lemma 5.1 and 5.3                                                                                                   |   21 |
|     | H.1 Proof of Lemma H.2 . . . . . . . . . . . . . . . . . . . .                                                                |   22 |
|     | H.2 Proof of Lemma H.3 . . . . . . . . . . . . . . . . . . . .                                                                |   24 |
| I   | Proofs of Lemma 5.4 and 5.5                                                                                                   |   25 |
| J   | Proofs of Lemma 6.1 and 6.2                                                                                                   |   28 |
| K   | Proof of Theorem K.1 and Lemma 6.3                                                                                            |   29 |
| L   | Proofs for Offline Training Complexity                                                                                        |   30 |
| M   | Other Technical Lemmas                                                                                                        |   33 |
| N   | Regularization/Constraint-based Approach for Assumption 1                                                                     |   33 |

## A Previous Bounds for MDPs and SGs

The techniques we use in this paper are most related to the probably approximately correct (PAC) analysis for RL algorithms. Some rather complete reviews of the related works are provided in [19, 9]. [19] considers the average-reward MDP that is communicating with bounded diameter D (i.e., max s,s ′ min π T π s → s ′ ( M ) ≤ D , where T π s → s ′ ( M ) is defined as the expected time to reach from state s to state s ′ under model M and policy π ). Their UCRL2 algorithm achieves ˜ O ( DS √ AT ) regret upper bound, while still having a gap with the Ω( √ DSAT ) lower bound. These bounds translate to ˜ O ( D 2 S 2 A ε 2 ) and Ω ( DSA ε 2 ) sample complexity. The additional D dependencyis resolved by [22, 9], though in discounted and episodic settings respectively. These two works leverage the Bellman equation for local variance and obtained sample complexity bounds of order ˜ O ( S 2 A ε 2 (1 -γ ) 3 ) and ˜ O ( H 2 S 2 A ε 2 ) ( γ : discount factor, H : fixed horizon length), making their gaps with the lower bounds Ω ( SA ε 2 (1 -γ ) 3 ) and Ω ( H 2 SA ε 2 ) remain only an order of S .

The scenario that most resembles ours in the literature is that considered in [5], who proposed the algorithm R-MAX. R-MAX is an optimism-based algorithm that can be used to learn stochastic games with arbitrary opponents. However, the algorithm depends on a parameter ε and the ε -return mixing time T ε that need to be known in advance. This ε -return mixing time resembles our D ε in Assumption 2. As a result, their ˜ O ( T 3 ε S 2 A ε 3 ) translates to ˜ O ( D 3 S 2 A ε 6 ) , while our bound is ˜ O ( DS 2 A ε 3 ) .

Another difference lies in that the output policy of our algorithm is a stationary one, rather than a T ε -step non-stationary policy as in R-MAX.

## B Inequalities

LemmaB.1. (Azuma-Hoeffding'sinequality. Theorem 4.2 of [6]) Let F 1 ⊆ · · · ⊆ F T be a filtration, and Y 1 , · · · , Y T real random variables such that Y t is F t -measurable, /BX ( Y t |F t -1 ) = 0 and Y t ∈ [ A t , A t + c t ] where A t is a random variable F t -1 -measurable and c t is a positive constant. Then with probability at least 1 -δ ,

Lemma B.2. (Bernstein inequality. Lemma 4.4 of [6]) Let F 1 ⊆ · · · ⊆ F T be a filtration, and Y 1 , · · · , Y T real random variables such that Y t is F t -measurable, /BX ( Y t |F t -1 ) = 0 and | Y t | ≤ b for some b &gt; 0 . Let V T = ∑ T t =1 ( Y 2 t |F t -1 ) and δ &gt; 0 . Then with probability at least 1 -δ ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## C Perturbation Bounds for Markov Chains

Perturbation analysis for Markov chains plays an important role in analyzing reinforcement learning algorithms (e.g., [2]). Those analyses mainly center around the question that when the transition probabilities of a Markov chain are perturbed by a little, how much stationary distributions or mean first passage times (as defined in Definition C.1) will change. While in [2], the perturbation bound for stationary distributions is used, we further use that of the mean first passage time to get a tighter regret bound.

In this section, we use i, j to index states, and use µ i to denote the stationary distribution of state i in an irreducible Markov chain.

Definition C.1 (Mean first passage time) . In a Markov chain, we define T ij to be the expected time to reach state j starting from state i . In the case i = j , T ii is the expected time to return to state i when starting from i . Thus T ij ≥ 1 always holds whether i = j or not.

## C.1 Perturbation Bounds for Stationary Distribution

Theorem C.2 (Proposition 2.2 of [7]) . Let C and ˜ C be two irreducible Markov chains with the same state space S . Let their transition matrices be P , ˜ P , and stationary distributions be µ , ˜ µ . Let E = ˜ P -P and use ‖·‖ ∞ to represent the largest absolute value in a matrix, then ∀ j ,

/negationslash

<!-- formula-not-decoded -->

With a little modification on the proof of Theorem C.2, we can actually have the following lemma, which only requires that C be an irreducible Markov chain.

/negationslash

<!-- formula-not-decoded -->

Theorem C.3. Let C be an irreducible Markov chain, and ˜ C be some Markov chain with the same state space S as C . Let their transition matrices be P , ˜ P , and let C 's stationary distributions be µ . Let E = ˜ P -P . If ‖ E ‖ ∞ &lt; 2 / ( S max i = j T ij ) , then ˜ C is also an irreducible Markov chain; furthermore, the stationary distribution of ˜ C , ˜ µ , satisfies ∀ j ,

/negationslash

Proof. Let P ∗ and ˜ P ∗ be the Cesaro limits of P and ˜ P , which is defined by P ∗ = lim T →∞ 1 T ∑ T t =1 P t -1 . Then we have

<!-- formula-not-decoded -->

and thus ( ˜ P ∗ -P ∗ )( I -P ) = ˜ P ∗ E . If ˜ P induces an irreducible Markov chain, ˜ P ∗ will have all identical rows and all positive elements. Suppose not, we can still extract its k -th row, which corresponds to the stationary distribution when starting from state k . Let this k -th row's j -th element be ˜ µ k j . We can write (˜ µ k j -µ j )( I -P ) = ˜ µ k j E. Then following the same proof as in [7] or by [16]'s Theorem 2.1, we still have

/negationslash

<!-- formula-not-decoded -->

Nowsince ‖ E ‖ ∞ ≤ 2 / ( S max i = j T ij ) and µ j &gt; 0 ∀ j , we have ˜ µ k j &gt; 0 ∀ j, k . This means that every state is recurrent and reachable from each other, implying that ˜ P induces an irreducible Markov chain.

/negationslash

## C.2 Perturbation Bounds for Mean First Passage Time

The main result of this subsection is stated in Theorem C.9. It is developed with the help of Theorem C.5 to Theorem C.8.

Definition C.4 (g-inverse, Definition 3.1 of [15]) . A g-inverse of a matrix A is any matrix G such that AGA = A .

Theorem C.5 (Theorem 5.3 of [16]) . Let C be an irreducible Markov chain with stochastic matrix P . Let T ij be the first passage time from state i to state j , and let G be any g-inverse of I -P . We have

<!-- formula-not-decoded -->

The below theorem introduces a special g-inverse that is convenient for our use.

Theorem C.6 (Theorem 3.3 of [15]) . Let P be a stochastic matrix of an irreducible Markov chain. Let p /latticetop n denote the n -th row of P , and e n denote the unit column vector with n -th component being 1. Then I -P + e n p /latticetop n is non-singular, and G = ( I -P + e n p /latticetop n ) -1 is a g-inverse of I -P .

/negationslash

Theorem C.7 (Section 5 of [16]) . Let ˜ P be a stochastic matrix of an irreducible Markov chain perturbed from another stochastic matrix P of an irreducible Markov chain. Suppose that the perturbation only occurs at the n -th row of P (i.e. p /latticetop i = ˜ p /latticetop i ∀ i = n ). Define G as that in Theorem C.6. Then G = ˜ G .

Suppose that the perturbation only occurs at the n -th row, and let G = ( I -P + e n p /latticetop n ) -1 . Then Theorem C.5 and C.7 together imply that for i = j ,

/negationslash

<!-- formula-not-decoded -->

with T jj = 1 /µ j and ˜ T jj = 1 / ˜ µ j (Corollary 5.3.1 of [16]). Here we see that ˜ T in = T in , ∀ i .

Lemma C.8. Let P be the stochastic matrix of an irreducible Markov chain, and let G = ( I -P + e n p /latticetop n ) -1 . If all mean first passage times are bounded by D ′ (i.e., T ij ≤ D ′ ∀ i, j ), then | G ij -G jj | ≤ 2 µ j D ′ ∀ i, j .

Proof. We first verify that

<!-- formula-not-decoded -->

where P n is obtained by deleting the n -th row and n -th column of P (without loss of generality, assume that the n -th row is last row of G ).

Directly expanding I -P + e n p /latticetop n , we get

<!-- formula-not-decoded -->

where d = ( -p 1 ,n , -p 2 ,n , ..., -p n -1 ,n ) /latticetop . To verify that ( I -P + e n p /latticetop n ) -1 takes the form of (10), one only needs to verify that ( I -P n ) e + d = 0 . This can be seen by ( I -P n ) e = e -P n e = (1 , ..., 1) -( ∑ n -1 i =1 p 1 ,i , ..., ∑ n -1 i =1 p n -1 ,i ) = -d . For i = n , from G 's expression in (10), we have

<!-- formula-not-decoded -->

/negationslash

Note that the dimension of e i and e are n in the second expression of (11), while are n -1 in the third expression. By [7]'s Equation (2.3), e /latticetop i ( I -P n ) -1 e = T in . One can also see this by observing that ( I -P n ) -1 = I + P n + P 2 n + · · · , and e /latticetop i P n m e j is 'the probability of staying at j after m steps from i , while not visiting n in any of the m steps'. Summing e /latticetop i P n m e j over j and m , the physical meaning becomes the mean first passage time from i to n , and the mathematical expression becomes e /latticetop i ( I -P n ) -1 e . Thus, | ∑ n k =1 ( G ik -G jk ) | = | T in -T jn | ≤ max ij T ij ≤ D ′ . By Theorem C.5, whenever i = j ,

/negationslash

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now combine (9) with (8) and Lemma C.8. Assuming that T ij ≤ D ′ , we have for i = j ,

/negationslash

<!-- formula-not-decoded -->

With (8) and (12) available, we now consider a general perturbation, which can actually be decomposed as S single-row perturbations.

Theorem C.9. Let P , ˜ P be the original and the perturbed stochastic matrices, and let { T ij } , { ˜ T ij } be their corresponding mean first passage times. If max ij T ij ≤ D and ‖ E ‖ ∞ = ∥ ∥ ∥ ˜ P -P ∥ ∥ ∥ ∞ ≤ 1 8 DS 2 , then max ij ˜ T ij ≤ 2 D.

Proof. We do this general perturbation of P by perturbing one row at a time. This procedure will repeat for S times.

Suppose that the original stationary distribution and first passage times are denoted by µ (0) i and T (0) ij , and that those after n -th perturbation are denoted by ˜ µ ( n ) i and ˜ T ( n ) ij .

Suppose that T (0) ij ≤ D ∀ i, j and µ (0) j ≥ 1 D ∀ j . Set ‖ E ‖ ∞ ≤ 1 8 S 2 D . We prove the following facts by induction:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

for n = 1 , ..., S. Since n ≤ S , these induction hypotheses implicitly imply that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

because (1 -1 / (8 S )) S ≥ 1 / 2 for all S ≥ 1 . Now we start the induction. The base case for n = 0 clearly holds. Suppose that (13)-(14) hold for all n ≤ k . Then by (8) we have

/negationslash

<!-- formula-not-decoded -->

## D Lemmas for Failing Events

Lemma D.1 (Proposition 18 of [19]) . The number of phases is upper bounded by U max = SA log 2 T .

Proof. Since phase changes only occur when the sample count of some ( s, a 1 , a 2 ) is doubled, those changes corresponding to a specific ( s, a 1 , a 2 ) is upper bounded by log 2 T . Considering all states and actions, the total number of phase changes is upper bounded by SA log 2 T .

Lemma D.2 (Lemma 17 of [19]) . For some specific k, s and a , the event p ( ·| s, a ) ∈ CONF 1 (ˆ p k ( ·| s, a ) , n k ( s, a )) holds with probability at least 1 -δ 1 .

<!-- formula-not-decoded -->

Lemma D.3 (Lemma 1 of [9], Theorem 10 and 11 of [25]) . For some specific k, s and a , the event p ( ·| s, a ) ∈ CONF 2 (ˆ p k ( ·| s, a ) , n k ( s, a )) holds with probability at least 1 -Sδ 1 .

<!-- formula-not-decoded -->

Proof of Lemma 4.2. ByLemmaD.1, there are at most SA log 2 T confidence set updates to consider. Each update involves only a specific ˆ p ( ·| s, a ) (totally S entries). By Lemma D.2, D.3 and using the union bound, the event M ∈ M k ∀ k holds with probability at least 1 -SA log 2 T × ( δ 1 + Sδ 1 ) ≥ 1 -δ .

## E Lemmas for Stationary Optimal Policies

Theorem E.1. Given a stochastic game M = ( S , A , r, p ) , where S is countable, A = A 1 × A 2 a compact metric space and both r ( s, · ) ∈ [0 , 1] and p ( s ′ | s, · ) are continuous in a = ( a 1 , a 2 ) . Suppose Assumption 2 holds for M . Then there exist maximin stationary policies π ∗ = ( π 1 ∗ , π 2 ∗ ) for the two-player zero-sum stochastic game, the maximin stationary policies attain the game value ρ ∗ , which is independent of the initial state, and there is a bounded function h ( · ) which together with ρ ∗ satisfies the following Bellman equation. That is, for all state s ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To prove this, we use the following lemma which connects the boundedness of mean first passage times with the uniform boundedness of sp( V ∗ α ( · )) for all discount factor 0 &lt; α &lt; 1 , where V ∗ α ( · ) is the discounted game value defined as V ∗ α ( s ) = max π 1 min π 2 /BX π 1 ,π 2 [∑ ∞ t =1 α t -1 r t | s 1 = s ] . It is known that for any discount factor 0 &lt; α &lt; 1 , discounted SGs always have maximin stationary policies π α = ( π 1 α , π 2 α ) which attain the game value V ∗ α ( s ) for all s . We next show that the span of V ∗ α is uniformly bounded by D under Assumption 2.

Lemma E.2. [14] Suppose given a stochastic game M = ( S , A , r, p ) , where 0 ≤ r ( s, a 1 , a 2 ) ≤ 1 . Suppose ∀ s, s ′ ∈ S and for any π 2 ∈ Π SR for Player 2, there exists a π 1 ∈ Π SR for Player 1 such that the mean first passage time T π 1 ,π 2 s,s ′ ≤ D . Then we have | V ∗ α ( s ) -V ∗ α ( s ′ ) | ≤ D, ∀ s, s ′ ∈ S , for all 0 &lt; α &lt; 1 .

Proof. Fix s, s ′ ∈ S . Fix a discount factor 0 &lt; α &lt; 1 . For a fixed pair of maximin stationary policies π α = ( π 1 α , π 2 α ) ∈ Π SR × Π SR , the discounted value function satisfies V ∗ α ( s ) = r ( s, π 1 α , π 2 α ) + α ∑ s ′ p ( s ′ | s, π 1 α , π 2 α ) V ∗ α ( s ′ ) . Since for any π 1 ∈ Π SR , V ∗ α ( s ) ≥ r ( s, π 1 , π 2 α ) + α ∑ s ′ p ( s ′ | s, π 1 , π 2 α ) V ∗ α ( s ′ ) , thus recursively, for any time step T ≥ 1 , we have

<!-- formula-not-decoded -->

where E π 1 ,π 2 s [ · ] = E s [ ·| π 1 , π 2 ] denote the expectation conditioned on initial state being s , and the players executing the policy pair ( π 1 , π 2 ) . Hence for any stopping time τ ,

<!-- formula-not-decoded -->

In particular, by choosing τ as the hitting time of s ′ from s ,

<!-- formula-not-decoded -->

For the first inequality we used V α ( s τ ) = V α ( s ′ ) and for the second, the non-negativity of r ( s, a ) and Jensen's inequality. The equality holds since the expected value of hitting time is the mean first passage time T π 1 ,π 2 α s → s ′ . The third inequality is essentially α x ≥ ( α -1) x +1 for x ≥ 1 ; the fourth (1 -α ) V α ≤ 1 . For the last inequality we used the assumption that there exists some π 1 for which T π 1 ,π 2 α s → s ′ ≤ D .

Lemma E.3. [12] Suppose | V ∗ α ( s ) -V ∗ α ( s ′ ) | is uniformly bounded for all 0 &lt; α &lt; 1 and for any s, s ′ ∈ S . Then there exist a pair of maximin stationary policies π = ( π 1 ∗ , π 2 ∗ ) attaining the game value ρ ∗ which is independent of the initial state and a bounded function h ( · ) for which the following equations hold. For all state s ,

<!-- formula-not-decoded -->

Proof. For any discount factor 0 &lt; α &lt; 1 ,

<!-- formula-not-decoded -->

Subtracting both sides by V ∗ α ( s 1 ) for some fixed state s 1 , and defining v α ( s ) := V ∗ α ( s ) -V ∗ α ( s 1 ) , we get, for all s ,

<!-- formula-not-decoded -->

## Algorithm 2 Value Iteration with Schweitzer transform

```
Input: M = ( S , A 1 ×A 2 , r, p ) , 0 < γ < 1 , 0 < α < 1 . Initialization: v 0 ≡ 0 . repeat for i = 1 , 2 , ... v i = (1 - α ) val { r + Pv i - 1 } + αv i - 1 . until sp( v i - v i - 1 ) ≤ (1 - α ) γ .
```

Since -D ≤ v α ( s ) ≤ D , 0 ≤ (1 -α ) V ∗ α ( s 1 ) ≤ 1 and π i α ∈ Π SR , ( i = 1 , 2) , all of which are contained in compact subsets/spaces, by using diagonalization argument and by Lebesgue convergence theorem, we can obtain a sequence α k → 1 , a bounded function h , and a constant ρ ∗ such that v α k ( · ) → h ( · ) , (1 -α k ) V ∗ α k ( s 1 ) → ρ ∗ , π i α k → π i ∗ , ( i = 1 , 2) , and

<!-- formula-not-decoded -->

as k →∞ . Hence in the limit, for all state s ,

<!-- formula-not-decoded -->

## F MAXIMIN-EVI and Its Convergence

As noted in Section 4.1, MAXIMIN-EVI proceeds simply by applying value iteration (Algorithm 2) on M + . The output of the algorithm is a value vector with tolerable errors. The val { r + Pv i -1 } term in Algorithm 2 becomes

<!-- formula-not-decoded -->

The inner maximization can be efficiently solved with linear programming. The MAXIMIN-EVI ( M k , γ k ) in UCSG is then done by running Algorithm 2 with the evaluation of (17) in every iteration.

The following three lemmas characterize the convergence of the algorithm, and the properties of its outputs when converged. Lemma F.1 gaurantees that MAXIMIN-EVI converges. Lemma F.2 shows that when the algorithm halts, the output policy's worst-case average reward does not deviate from the maximin reward by more than γ . Lemma F.3 shows that the output value vector has a span no more than D .

Lemma F.1 (Theorem 4 in [34]) . Suppose that Assumption 2 holds for some SG M . Then performing Value Iteration with Schweitzer transform on M converges asymptotically.

Proof of Lemma F.1. If Assumption 2 holds, then the Bellman equation holds with an initial-state independent game value by Theorem E.1. Then by Theorem 4 of [34], the value iteration with Schweitzer transform converges.

Lemma F.2. Suppose that Assumption 2 holds for some stochastic game M . Let { v i } be the value sequence in the Value Iteration algorithm. Let N be the index when iteration halts, i.e., sp ( v N +1 -v N ) ≤ (1 -α ) γ . Let π 1 := solve 1 { r + Pv N } . Then π 1 is γ -optimal in the sense that min π 2 ρ ( M,π 1 .π 2 ) ≥ ρ ∗ ( M ) -γ .

<!-- formula-not-decoded -->

where π = ( π 1 , π 2 ) for any π 2 ∈ Π SR . Let P ∗ π = lim T →∞ 1 T ∑ T t =1 P t -1 π be the Cesaro limit of P π . Applying it on both sides of the inequality, we get D e ≤ (1 -α ) P ∗ π r π = (1 -α ) ρ ( M,π 1 , π 2 , · ) , or D ≤ (1 -α ) ρ ( M,π 1 , π 2 , s ) , ∀ s, π 2 . Let π ∗ = ( π 1 ∗ , π 2 ∗ ) be the optimal policy pair and ρ ∗ ( M ) be their maximin value, then D ≤ (1 -α ) ρ ( M,π 1 , π 2 ∗ , s ) ≤ (1 -α ) ρ ∗ ( M ) . In a similar way, one can prove that U ≥ (1 -α ) ρ ∗ ( M ) . Since we assume U -D ≤ (1 -α ) γ , we have D ≥ (1 -α )( ρ ∗ ( M ) -γ ) . Therefore, π 1 is γ -optimal in the sense that ∀ π 2 , ρ ( M,π 1 , π 2 , s ) ≥ ρ ∗ ( M ) -γ .

<!-- formula-not-decoded -->

Lemma F.3. If Assumption 2 holds for some model M , then value iteration procedure in Algorithm 2 will always produce value functions with spans bounded by D . That is,

<!-- formula-not-decoded -->

Proof. Note that value iteration with Schweitzer transform is equivalent to the following procedure. First modify the transition kernel and reward by p α ( s ′ | s, a 1 , a 2 ) = (1 -α ) p ( s ′ | s, a 1 , a 2 ) + αδ s,s ′ and r α ( s, a 1 , a 2 ) = (1 -α ) r ( s, a 1 , a 2 ) + α 0 ; then do the normal value iteration by v i = val { r α + P α v i -1 } . By the principle of dynamic programming, v i is the maximin expected reward in the i -step game under the transformed model.

The transformed model is equivalent to the system where at each time step, the state remains same as the previous one with probability α , and within that step there is no reward obtained/paid.

Clearly, in this new game, the advantage of starting from state s than starting from state s ′ (which can be calculated by v i ( s ) -v i ( s ′ ) ) is no more than that in the original game. In the original game, by a similar argument as Remark 8 in [19], this advantage difference is bounded by D . This then implies the argument in the lemma.

## G Proof of Lemma 5.2

Lemma 5.2 directly follows from Lemma G.1 and G.2.

In this proof, we borrow the technique used in [22] and [9] to bound the number of steps with inaccurate transition probabilities (while they use this technique to bound the number of steps with inaccurate game value). Note here again that π t ( · ) can represent any history-dependent policy, and we hide its parameter H t = ( s 1 , a 1 , r 1 , ..., s t ) inside the subscript of t .

Define the importance of a joint action a at time t as

and the its knownness as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If we let w min = ε 3 √ 2 ln(1 /δ ) A and m = 5 log 2 2 ( T/w min ) ln(1 /δ ) ε 2 , with some 0 &lt; ε &lt; 1 , we can prove

with z 1 = 0 , z j = 2 j -2 ∀ j = 2 , 3 , ... , and some pre-defined w min &gt; 0 , m &gt; 0 . Note that we can always define them in hindsight even though the learner does not know π 2 t . These two amounts make partitions to the action set available at s t . The partitioning is based on the actions' probability of being selected at time t (i.e., π t ( a ) ), and the accuracy it has been estimated (the larger n k ( t ) ( s t , a ) , the more accurate). Intuitively, the larger κ t ( a ) , the less likely will action a contribute to inaccurate transition probability estimation. Define the partitions by X t,κ,ι := { a : κ t ( a ) = κ and ι t ( a ) = ι } , ∀ κ, ι .

the following lemmas.

Lemma G.1. For any s , any κ and any ι &gt; 0 , with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Lemma G.2. If for all κ and all ι &gt; 0 we have | X t,κ,ι | ≤ κ , then for any plausible ˜ p in the confidence set M k ( t ) , | ˜ p ( s ′ | s t , π t ) -p ( s ′ | s t , π t ) | ≤ ε for all s ′ .

## G.1 Proof of Lemma G.1

We prove Lemma G.1 with the help of Lemma G.3 and G.4.

Lemma G.3. For any s, κ , and ι &gt; 0 , ∑ T t =1 s t = s a t ∈ X t,κ,ι ≤ 6 Am ( κ +1) ιw min .

/BD /BD Proof. First fix a . By the definition of importance, if a ∈ X t,κ,ι , then ιw min ≤ π t ( a ) &lt; 2 ιw min . In the case κ &gt; 0 , we also have mκπ t ( a ) ≤ n k ( t ) ( s t , a ) &lt; 2 mκπ t ( a ) . They two together imply mκιw min ≤ n k ( t ) ( s t , a ) &lt; 4 mκιw min . This last inequality says that any ( s, a ) cannot be sampled in the partition ( κ, ι ) for more than about 3 mκιw min times. This is because when ( s, a ) is sampled once (i.e., s t = s, a t = a ), n k ( t ) ( s, a ) will be increased by one, and this cannot happen for more than 4 mκιw min -mκιw min times while ( s, a ) ∈ X t,κ,ι . Since UCSG only updates n k ( s, a ) when new phases start and doubling the sample count of a state-action triple incurs a phase change, we use a more conservative bound of 6 mκιw min . That is, we have

<!-- formula-not-decoded -->

In the case κ = 0 , we have n k ( t ) ( s t , a ) &lt; mπ t ( a ) &lt; 2 mιw min . Thus similarly, the sample counts of ( s, a ) in the partition ( κ, ι ) cannot exceed 4 mιw min . The cases of κ &gt; 0 and κ = 0 can then be combined into a single one:

Summing (19) over all actions leads to the statement in the lemma.

<!-- formula-not-decoded -->

Now we sketch the argument of the next lemma. When ι &gt; 0 , each action in X t,κ,ι are to be sampled with probability no less than ιw min . If furthermore | X t,κ,ι | is large, the probability that some a ∈ X t,κ,ι is sampled will be also large. However by Lemma G.3, the total times elements in partition ( κ, ι ) are sampled are upper bounded. Therefore, we can conclude that | X t,κ,ι | cannot be large for too many steps. Formally, we have

Lemma G.4. With probability at least 1 -δ ,

<!-- formula-not-decoded -->

Proof. To prove Lemma G.4, we need the help of Lemma B.2.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

/BD

/BD

/BD

where we define

/BD /BD ∣ Then Lemma B.2's conditions are met with b = 1 . Moreover,

<!-- formula-not-decoded -->

Substituting them into Lemma B.2 and rearraging terms, we get that with probability ≥ 1 -δ ,

<!-- formula-not-decoded -->

Solving the above inequality with respect to √ ∑ T t =1 q t , we can bound with probability ≥ 1 -δ that

Finally we look at q t . Since each action in X t,κ,ι are drawn at time t with probability at least ιw min , we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma G.1. Combining Lemma G.3 and G.4, we have

/BD /BD /BD /BD Combining (20), (21), and noting that /BD s t = s /BD a t ∈ X t,κ,ι ≥ /BD s t = s /BD | X t,κ,ι | &gt;κ /BD a t ∈ X t,κ,ι concludes the proof.

<!-- formula-not-decoded -->

with probability no less than 1 -δ . The lemma is then proved by substituting the selection of m and w min into (22), and using κ +1 ≥ 1 , ι ≥ 1 .

## G.2 Proof of Lemma G.2

Proof of Lemma G.2.

<!-- formula-not-decoded -->

where K and I are the set of effective κ 's and ι 's in the above summation (only partitions with ι &gt; 0 and κ &gt; 0 are relevant). By definition, there are at most log 2 ( 1 w min ) different values of ι for ι &gt; 0 , and log 2 ( T mw min ) ≤ log 2 ( T w min ) different values for κ &gt; 0 when ι &gt; 0 . The second inequality is by the definition of the confidence set; the third and the fifth are by Cauchy's inequality; the fourth is by the assumption of the lemma. Substituting the values of w min and m into the last expression, we can get the desired result.

## H Proofs of Lemma 5.1 and 5.3

To prove Lemma 5.1 and 5.3, the following lemma is a useful tool. In the following texts, we let v k ( s ) := ∑ t k +1 -1 t = t k s t = s , and write the joint policy ( π 1 k , ¯ π 2 k ) as ¯ π k .

/BD

<!-- formula-not-decoded -->

/BD Proof. Under Assumption 1, the times a state is visited within an interval of length D is in average no less than 1 (no matter what policies the players play). Consider any arbitrarily chosen time frame [ τ, τ ′ ) ⊂ [1 , T ] . In this time frame, there are /floorleft τ ′ -τ 2 D /floorright intervals each with length 2 D . By Markov's inequality, the probability s is visited at least once within each interval is lower bounded by 1 2 . With Azuma-Hoeffding's inequality, we have with probability at least 1 -δ T 2 that

<!-- formula-not-decoded -->

where the second inequality is easily verified by substracting RHS from LHS, and the third inequality is by the property of the floor function. Using an union bound over all possible τ and τ ′ , we get that (23) holds for all τ, τ ′ with probability at least 1 -δ .

Now apply (23) to all phases k with v k ( s ) ≤ v , and sum all of them up. Then we get

or

<!-- formula-not-decoded -->

Since there are at most SA log 2 T phases, the RHS of (24) is further bounded by ( 8 vD +2 D +32 D log( T 2 /δ ) ) SA log 2 T , which proves this lemma.

We prove Lemma 5.3 by proving the following Lemma H.2 and H.3.

Proof of Lemma 5.1. ¯ π 2 k is not well-defined if and only if there is a s such that v k ( s ) = ∑ t k +1 -1 t = t k /BD s t = s = 0 . The proof is done by simply applying Lemma H.1 with v = 1 together with a union bound over all states s .

## Lemma H.2.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## H.1 Proof of Lemma H.2

Proof of Lemma H.2. This lemma says, the stationary distribution of the irreducible Markov chain induced by π 1 k and ¯ π 2 k won't exceed the empirical distribution too much in most steps. To prove Lemma H.2, we will compare three transition probabilities:

<!-- formula-not-decoded -->

/BD

<!-- formula-not-decoded -->

/BD and use perturbation analysis to claim that when they are close enough, the stationary distributions they induce will also be close. Here, ˆ p k is constructed by counting empirical transitions. ˜ p k is only slightly modified from ˆ p k : the last term in the numerator changes from s t k +1 -1 s t k +1 to

/negationslash

/BD /BD /BD s t k +1 -1 /BD s t k . Under the condition that ¯ π 2 k is well-defined, ∑ t k +1 -1 t = t k /BD s t = s = 0 ∀ s , which means that ˜ p k has non-zero probability to reach any states from any states, hence inducing an irreducible Markov chain. ¯ p k also induces an irreducible Markov chain by Assumption 1. We denote the stationary distributions corresponding to ¯ p k and ˜ p k by ¯ µ k and ˜ µ k .

We will see that ˜ µ k is exactly the same as the empirical distribution (i.e., ˜ µ k ( s ) = v k ( s ) T k ). By Theorem C.2, when two transition probabilities are close enough, their stationary distributions will also be close. We will argue that except for a constant amount of steps, | ¯ p k ( s ′ | s ) -ˆ p k ( s ′ | s ) | ≤ 1 2 DS and | ˆ p k ( s ′ | s ) -˜ p k ( s ′ | s ) | ≤ 1 2 DS hold for all s, s ′ . When they both hold, we can use Theorem C.2 with ‖ E ‖ ∞ = max s,s ′ | ¯ p k ( s ′ | s ) -˜ p k ( s ′ | s ) | ≤ 1 DS and bound | ¯ µ k ( s ) -˜ µ k ( s ) | ≤ 1 2 ¯ µ k ( s ) . This will directly imply ¯ µ k ( s ) ≤ 2˜ µ k ( s ) = 2 v k T k .

From the discussion above, Lemma H.2 is proved as long as the three following lemmas (Lemma H.4, H.5, H.6) are proved.

## Lemma H.4.

## Lemma H.5.

<!-- formula-not-decoded -->

## Lemma H.6.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof of Lemma H.4. Fix s , s ′ , and k . Consider the martingale difference sequence defined by Y t := /BD s t = s ( p ( s ′ | s, π 1 k ( t ) , π 2 t ) -/BD s t +1 = s ′ ) , where k ( t ) denotes the phase to which time step t belongs. By Lemma B.2, for any τ ≤ T +1 , with probability at least 1 -2 δ/T ,

∣ ∣ Here V t k ,τ = ∑ τ -1 t = t k q t (1 -q t ) ≤ ∑ τ -1 t = t k q t ≤ ∑ τ -1 t = t k /BD s t = s where q t := /BD s t = s p ( s ′ | s, π 1 k ( t ) , π 2 t ) ≤ /BD s t = s . With an union bound, we have that (25) holds for all τ with probability at least 1 -2 δ . Now pick τ to be t k +1 , and thus V t k ,t k +1 ≤ ∑ t k +1 -1 t = t k s t = s = v k ( s ) . Then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

with probability at least 1 -2 δ . Another union bound over s ′ lets the above inequality holds for all s ′ with probability at least 1 -2 Sδ .

We need about v k ( s ) ≥ 25 D 2 S 2 log( T 2 δ -1 ) to make | ¯ p k ( s ′ | s ) -ˆ p k ( s ′ | s ) | ≤ 1 2 DS ∀ s ′ in the above inequality. By Lemma H.1, we see that the number of steps not satisfying this condition is upper bounded by ˜ O ( D 3 S 3 A ) . Another union bound over s proves the lemma.

Proof of Lemma H.5. By the construction of ˜ p k , | ˜ p k ( s ′ | s ) -ˆ p k ( s ′ | s ) | ≤ 1 v k ( s ) ∀ s ′ . Again, we use Lemma H.1 and set the threshold v = ˜ Θ(2 DS ) to make | ˜ p k ( s ′ | s ) -ˆ p k ( s ′ | s ) | ≤ 1 2 DS ∀ s ′ . By Lemma H.1, this will hold except for ˜ O ( D 2 S 2 A ) steps. An union bound over states leads to the ˜ O ( D 2 S 3 A ) bound.

Proof of Lemma H.6. We only need to check whether the equation ˜ µ k ( s ′ ) = ∑ s ˜ µ k ( s )˜ p k ( s ′ | s ) holds for all s, s ′ . Indeed,

<!-- formula-not-decoded -->

## H.2 Proof of Lemma H.3

Proof of Lemma H.3. By Assumption 1, the maximum mean first passage time under model M and policy pair ( π 1 k , ¯ π 2 k ) does not exceed D , i.e., T π 1 k , ¯ π 2 k ( M ) ≤ D . Then by Theorem C.9, we know that if all transition probabilities in the Markov chain induced by ( M 1 k , π 1 k , ¯ π 2 k ) is perturbed from that induced by ( M,π 1 k , ¯ π 2 k ) within the amount of 1 8 DS 2 , the former's maximum mean first passage time can be bounded by two times the latter's, i.e., T π 1 k , ¯ π 2 k ( M 1 k ) ≤ 2 T π 1 k , ¯ π 2 k ( M ) . This also implies that ( M 1 k , π 1 k , ¯ π 2 k ) induces an irreducible Markov chain. Finally, by Remark M.1, we have sp( h ( M 1 k , π 1 k , ¯ π 2 k , · )) ≤ T π 1 k , ¯ π 2 k ( M 1 k ) . Combining the three inequalities above, we can have sp( h ( M 1 k , π 1 k , ¯ π 2 k , · )) ≤ 2 D . As a result, to prove this theorem, we only need to bound the number of steps in phases where there exist s, s ′ such that the transition probability difference | p 1 k ( s ′ | s, π 1 k , ¯ π 2 k ) -p ( s ′ | s, π 1 k , ¯ π 2 k ) | is larger than 1 8 DS 2 ( p 1 k is the transition kernel of M 1 k ). We define the event E k ( s ) = { ∃ s ′ , | p 1 k ( s ′ | s, π 1 k , ¯ π 2 k ) -p ( s ′ | s, π 1 k , ¯ π 2 k ) | &gt; 1 8 DS 2 } , and E k = {∃ s, E k ( s ) = 1 } . Our goal is to prove ∑ k T k E k ≤ ˜ O ( D 3 S 5 A ) .

<!-- formula-not-decoded -->

/BD Fix k . Suppose that ¯ π 2 k is well-defined. By the definition of ¯ π 2 k and the triangle inequality, we have

<!-- formula-not-decoded -->

and g k ( s, ε ) := | G k ( s, ε ) | , i.e., g k ( s, ε ) is the number of steps t in phase k such that s t = s and the maximum transition probability error at that step is between ε and 2 ε . With these definitions, we can continue to upper bound (26) by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

If | p 1 k ( s ′ | s, π 1 k , ¯ π 2 k ) -p ( s ′ | s, π 1 k , ¯ π 2 k ) | &gt; 1 8 DS 2 , then by (27) we have Note that since steps counted in G k ( s, ε ) have maximum transition errors greater that ε , by Lemma 5.2, with high probability, ∑ k g k ( s, ε ) won't exceed c 1 A ε 2 , for some c 1 hides logarithmic terms. Now sum the above equation over phases where E k ( s ) holds, we get that

<!-- formula-not-decoded -->

or ∑ k : E k ( s ) v k ( s ) ≤ ˜ O ( D 2 S 4 A ) holds with high probability. Similar to the proof of Lemma H.1, we use (23) and lower bound ∑ k : E k ( s ) v k ( s ) ≥ ∑ k : E k ( s ) ( T k 8 D -˜ O (1) ) . Combining the lower bound and the upper bound, we get ∑ k : E k ( s ) T k ≤ ˜ O ( D 3 S 4 A ) . Finally, summing over s , we get the desired bound.

## I Proofs of Lemma 5.4 and 5.5

Proof of Lemma 5.4. Define notations: ¯ π k = ( π 1 k , ¯ π 2 k ) , ¯ p k ( s ′ | s ) := p ( s ′ | s, ¯ π k ) , ¯ h k ( s ) := h ( M, ¯ π k , s ) , ¯ ρ k := ρ ( M, ¯ π k ) , ¯ r k ( s ) := r ( s, ¯ π k ) , r t := r ( s t , a t ) .

By the construction of ¯ π 2 k , we have

and

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Now manipulate individual terms.

<!-- formula-not-decoded -->

where the third equality follows from (29);

<!-- formula-not-decoded -->

where the second equality follows from (30). Substituting (32) and (33) into (31), we get

<!-- formula-not-decoded -->

where Y 1 t := (∑ s ′ p ( s ′ | s t , π 1 k , π 2 t ) ¯ h k ( s ′ ) -¯ h k ( s t +1 ) ) , and Y 2 t := ( r ( s t , π 1 k , π 2 t ) -r t ) . It seems that Y 1 t and Y 2 t have expectations of zero and should be able to be bounded with Bernstein's inequality. Nevertheless, one needs to be careful about that ¯ h k depends on ¯ π 2 k , which is only known after phase k ends. In other words, ¯ h k is not F t -measurable for t ∈ ph ( k ) , where F t -1 := { s 1 , a 1 , · · · , s t } . The solution is as follows. Let D be the set where ¯ h k possibly lies. We discretize D and use the Bernstein bound on all discretization points. Finally, we use the fact that ¯ h k is not far from the nearest discretization point to bound the sum of Y 1 t .

Let D := [ -D,D ] S , and thus ¯ h k ∈ D . Clearly, there is a discretization D d with |D d | ≤ (2 DST ) S such that any h ∈ D can find some h d ∈ D d with | h ( s ) -h d ( s ) | ≤ 1 ST ∀ s . Now let Y 1( j ) t := (∑ s ′ p ( s ′ | s t , π 1 k , π 2 t ) h ( j ) ( s ′ ) -h ( j ) ( s t +1 ) ) for every h ( j ) ∈ D d , j = 1 , ..., (2 DST ) S . Now Y 1( j ) t 's are martingale difference sequences with respect to F t -1 , so we can apply Azuma-Hoeffding's inequality and bound

<!-- formula-not-decoded -->

with probability at least 1 -δ (2 DST ) S . Using the union bound, (35) holds for all j with probability at least 1 -δ . Also, there exists a j such that ∑ t k +1 -1 t = t k ( Y 1 t -Y 1( j ) t ) ≤ T k × 2 S ST = 2 T k T . Thus we have

<!-- formula-not-decoded -->

with high probability. We also have ∑ k ∑ t k +1 -1 t = t k Y 2 t ≤ ˜ O ( √ T ) by Azuma-Hoeffding's inequality. Also, ¯ h k ( s t k +1 ) -¯ h k ( s t k ) ≤ 2 D . Collecting terms, we get the desired bound.

Proof of Lemma 5.5. First fix k . Denote the transition probabilities of the optimistically selected model M 1 k by p 1 k ( ·|· , · , · ) . In this proof, we define ˜ h ( · ) := h ( M 1 k , ¯ π k , · ) , h ( · ) := h ( M, ¯ π k , · ) , ˜ µ ( · ) := µ ( M 1 k , ¯ π k , · ) , µ ( · ) := µ ( M, ¯ π k , · ) , ˜ ρ := ρ ( M 1 k , ¯ π k ) , ρ := ρ ( M, ¯ π k ) , r ( · ) := r ( s, ¯ π k ) , ˜ p ( s ′ | s ) := p 1 k ( s ′ | s, ¯ π k ) , p ( s ′ | s ) := p ( s ′ | s, ¯ π k ) .

By Bellman equation and the properties of irreducible Markov chains, we have

<!-- formula-not-decoded -->

for all s . Therefore, we can write (for any s )

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second equality is by using (37) and the property of stationary distribution: ∑ s µ ( s ) ( p ( s ′ | s ) -δ s,s ′ ) = 0 . By the definition of ˜ p and p , we have

where Y t := /BX [ q t ] -q t , and q t := /BD s t = s ‖ ˜ p ( ·| s, a t ) -p ( ·| s, a t ) ‖ 1 . To apply Lemma B.2, we note that | q t | ≤ 2 and V T := ∑ t k +1 -1 t = t k [ Y 2 t |F t -1 ] ≤ 2 ∑ t k +1 -1 t = t k [ q t ] . Then we can bound

with probability at least 1 -δ . (40) implies

<!-- formula-not-decoded -->

Continuing (38) with the help of (39) and (41), we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have used the assumptions in this lemma. Now sum over benign phases, we get

<!-- formula-not-decoded -->

with high probability. The last inequality is by the following Lemma together with Cauchy's inequality.

Lemma I.1 (cf. Lemma 19 of [19]) . For any sequence { z i } , i = 1 , ..., N with 0 ≤ z i ≤ Z i -1 := max { 1 , ∑ i -1 /lscript =1 z /lscript } . Let K be a subset of { 1 , ..., N } . Then we have

where L := ∑ i ∈ K z i .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

where L i := /lscript ∈ K : /lscript ≤ i z i . We used the inequality

<!-- formula-not-decoded -->

## J Proofs of Lemma 6.1 and 6.2

Proof of lemma 6.1. Note that for any phase k and any episode i that fully lies in phase k , we have /BX [ ∑ τ i +1 -1 t = τ i r ( s t , a t ) ] = V H ( M,π 1 k , π 2 i , s τ i ) . Therefore, the terms in ∑ k ∆ (5) k form a martingale difference sequence with no more than T/H terms. Furthermore, 0 ≤ ∑ τ i +1 -1 t = τ i r ( s t , a t ) ≤ H . By Lemma B.1, with probability 1 -δ , we have ∑ k ∆ (5) k ≤ √ log( δ -1 ) 2 T H H 2 = ˜ O ( √ HT ) .

Proof of Lemma 6.2. Suppose that the value iteration halts at iteration N , then under Assumption 2 and by the proof of Lemma F.2, we have

<!-- formula-not-decoded -->

Since ( M 1 k , p 1 k ) is selected based on the v N when the value iteration halts, (43) is equivalent to

<!-- formula-not-decoded -->

Besides, the span of the vector v N is bounded by D by Lemma F.3. Now we fix Player 1's policy as π 1 k in the extended game, and let Player 2 run an H -step SG. The least amount Player 2 has to pay Player 1 in this SG is min π 2 V H ( M 1 k , π 1 k , π 2 , s ) (assuming that the game starts from s ), which can be calculated by dynamic programming. The dynamic programming goes as follows: for i = 0 , ..., H -1 , for all s ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

which, in its vector form, can be written as u i +1 = min a { r a + P a u i } by denoting r a ( · ) := r ( · , π 1 k , a ) and ( P a ) ij := p 1 k ( j | i, π 1 k , a ) . We can re-write the induction procedure as

without affecting the solution. By the property min { u + v } ≥ min { u } +min { v } , we have

<!-- formula-not-decoded -->

By (44), min a { r a + P a v N } -v N ≥ ρ ∗ ( M + ) -γ , and since P a is stochastic, P a ( u i -v N ) ≥ min s ′ { u i ( s ′ ) -v N ( s ′ ) } . Combining them with (45), we have u i +1 ( s ) -v N ( s ) ≥ ρ ∗ ( M + ) -γ + min s ′ { u i ( s ′ ) -v N ( s ′ ) } for all s . Then by induction, we can easily prove u i ( s ) -v N ( s ) ≥ i ( ρ ∗ ( M + ) -γ ) + min s ′ { u 0 ( s ′ ) -v N ( s ′ ) } , and therefore, u i ( s ) ≥ i ( ρ ∗ ( M + ) -γ ) + v N ( s ) -max s ′ v N ( s ′ ) ≥ i ( ρ ∗ ( M + ) -γ ) -D .

Let i = H and note that ρ ∗ ( M + ) = max ˜ M max π 1 min π 2 ρ ( ˜ M,π 1 , π 2 ) ≥ min π 2 ρ ( M 1 k , π 1 k , π 2 , s ) . The above result translates to min π 2 V H ( M 1 k , π 1 k , π 2 , s ) ≥ H min π 2 ρ ( M 1 k , π 1 k , π 2 , s ) -D -Hγ , which bounds ∆ (2) k by ∑ i ∈ ph ( k ) ( D + Hγ ) .

## K Proof of Theorem K.1 and Lemma 6.3

Theorem K.1. (Sample Complexity Bound of UCSG . cf. Theorem 1 [9]) Given δ &gt; 0 , with probability at least 1 -δ , for any 0 &lt; ε &lt; 1 , UCSG produces a sequence of policies π 1 k , that yield at most ˜ O ( H 2 S 2 A ε 2 ) episodes i such that | V H ( M,π 1 k , π 2 i , s τ i ) -V H ( M 1 k , π 1 k , π 2 i , s τ i ) | &gt; ε .

Definition K.2. Define the weight of a state-joint-action pair ( s, a ) under joint policy π i in episode i as the expected occurrence frequency of ( s, a ) in episode i ,

Theorem K.1 mainly follows from the following Lemma K.6 and K.7. In [9] the analysis of sample complexity is facilitated by partitioning the state-action space. The state-action pairs are grouped into different categories according to two indices. The first index, importance , measures in log-scale the relative occurrence frequency of ( s, a ) with respect to a fixed constant under the policy. The second index, knownness , measures also in log-scale the ratio of the total number of observations to the occurrence frequency. Here we modify the the definition of weight, importance, and knownness for a state-joint action ( s, a ) = ( s, a 1 , a 2 ) defined below to have a partition of the state-joint-action space S × A = S × A 1 ×A 2 for each episode.

<!-- formula-not-decoded -->

The setting in [9] is somewhat different from two-player zero-sum SGs. In the episodic RL setting after an episode is over, a new episode starts afresh with the same initial distribution p 0 , while in the non-episodic setting, initial state s τ i in each episode is sampled from a different distribution. Initial state distributions do not matter that much in our setting except we need the initial state s τ i to compute the expected frequency w i ( s, a ) .

Definition K.3. Define the importance of a state-joint-action pair ( s, a ) in episode i as

<!-- formula-not-decoded -->

where z 1 = 0 and z j = 2 j -2 ∀ j = 2 , 3 , ...

Definition K.4. Define the knownness of a a state-joint-action pair ( s, a ) in episode i as

<!-- formula-not-decoded -->

where z 1 = 0 and z j = 2 j -2 ∀ j = 2 , 3 , ...

Definition K.5. We can now categorize state-joint-action pairs ( s, a ) into subsets

<!-- formula-not-decoded -->

In contrast to the original definitions [9] which are designated for each phase k in the episodic RL setting, in our setting, weight w i ( s, a ) , importance ι i ( s, a ) , knownness κ i ( s, a ) are now indexed for each episode i because Player 2 may have arbitrary policies in different episodes.

Theorem K.1 mainly follows from the following Lemma K.6 and K.7. Select m = 512 SH 2 (log log H ) 2 log 2 ( 8 T 2 SH ) ln(6 /δ 1 ) ε 2 , δ 1 := δ 2 U max S , U max := SA log 2 T and w min := ε 4 HSA for any 0 &lt; ε &lt; H , and any 0 &lt; δ &lt; 1 and then we have the following two lemmas.

LemmaK.6. (cf. Lemma 2 in [9]) Let E be the number of episodes i for which there are κ and ι with | X i,κ,ι | &gt; κ , i.e. E = ∑ ∞ i =1 /BD {∃ ( κ, ι ) : | X i,κ,ι | &gt; κ } and assume that m ≥ 6 H 2 ε ln(2 E max /δ ) , where E max = log 2 ( H w min ) log 2 ( SA ) . Then P ( E ≤ 6 SAE max m ) ≥ 1 -δ/ 2 .

Proof. The proof mainly follows as Lemma 2 [9]. Here we point out the differences between the original UCFH algorithm [9] and our UCSG, when we remove the input ε .

1. Their stopping rule for phase k is dependent on the specification of ε .
2. They set an upper bound for the maximum number of executions for each state-action pair ( s, a ) , which is determined beforehand and hardcoded in their algorithm.
3. Our algorithm only needs input δ to specify the failure probability and has ( ε, δ ) -PAC bounds for arbitrarily selected ε .

The original UCFH nearly doesn't need the parameter ε except at one place: their phases stops when ' ∃ ( s, a ) , v k ( s, a ) ≥ max { mw min , n k ( s, a ) } and n k ( s, a ) &lt; SmH .' Since w min and m are defined through ε , this stopping rule requires ε to be known by the algorithm. They need this because they would like to control U max , the total number of phases run by the algorithm. In their case, having this stopping rule, U max ≤ SA log 2 SmH mw min = SA log 2 SH w min because phase change won't be triggered when n k ( s, a ) &lt; mw min or n k ( s, a ) &gt; SmH . However, since we assume that the time horizon T is known, we can simply use U max ≤ SA log 2 T , and this can simplify our stopping rule to only ' ∃ ( s, a ) , v k ( s, a ) ≥ n k ( s, a ) .'

Therefore, we can totally abandon the use of ε in our algorithm, but enjoy their analysis results. The results automatically hold for arbitrarily selected ε . However, since we bound the number of κ by log 2 (4 HSAT/ε ) in Lemma K.7, we cannot let ε tends to 0 too fast. (The minimum ε we will select is ε 0 = min { H, √ ( H 3 S 2 A ) /T } as in the proof of Lemma 6.3, where we select H = max { D, 3 √ D 2 T/ ( S 2 A ) } for Theorem 3.2 ).

Lemma K.7. (cf. Lemma 3 in [9]) Assume M ∈ M k . If | X i,κ,ι | ≤ κ for all ( κ, ι ) and for all 0 &lt; ε ≤ 1 and m ≥ 512 CH 2 ε 2 (log 2 log 2 H ) 2 log 2 ( 4 HSAT ε ) log 2 ( SA ) ln(6 /δ 1 ) . Then | V H ( M 1 k , π 1 k , π 2 i ) -V H ( M,π 1 k , π 2 i ) | ≤ ε.

Proof. It mainly follows the same proof as Lemma 3 in [9]. It was shown sufficient to let m ≥ 512 C (log 2 log 2 H ) 2 |K × I| H 2 ε 2 ln(6 /δ 1 ) . The only differences are in the upper bounds for |K × I| . In UCFH, the maximum number of executions of each state-action pair is set equal to mSH . Thus their knownness κ ( s, a ) is no more than n ( s,a ) mw min ≤ 4 S 2 AH 2 ε , whereas in our setting, since n ( s, a ) ≤ T , κ ( s, a ) n ( s,a ) mw 4 HSAT ε . Thus in our setting log 2 ( 4 HSAT ε ) log 2 ( SA ) .

<!-- formula-not-decoded -->

Proof of Lemma 6.3. Let ε 0 :=min { H, √ ( H 3 S 2 A ) /T } , and δ 0 := δ/ /ceilingleft log 2 ( H/ε 0 ) /ceilingright . We invoke logarithmically many times the bound in Theorem K.1 and use the union bound to obtain the regret. By assumption, for j = 1 , ..., /ceilingleft log 2 ( H/ε 0 ) /ceilingright , with probability no less than 1 -δ 0 , there are at most ˜ O (4 j S 2 A ) episodes that are not (2 -j H ) -optimal. Then the total error is bounded by

<!-- formula-not-decoded -->

## L Proofs for Offline Training Complexity

Proof of Theorem 7.1. Define

<!-- formula-not-decoded -->

Also, define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Reg (on) ′ ε is defined as a summation similar to Reg (on) T except that it is summed only over time steps in phases k ∈ K ′ ε . Besides, analogous to the definition of L ε , we define L ′ ε := ∑ k : benign T k { ρ ∗ ( M ) -min π 2 ρ ( M,π 1 k , π 2 ) &gt; ε } .

/BD We will argue (a) the order of Reg (off) ′ ε does not exceed that of Reg (on) ′ ε , and (b) the upper bound of Reg (on) ′ ε is similar to that of Reg (on) T except that the dependency on T is replaced by L ′ ε .

To show (a), we note that the extra terms in Reg (off) ′ ε compared to Reg (on) ′ ε are the sum of

<!-- formula-not-decoded -->

over k ∈ K ′ ε . Λ (7) k is bounded by T k γ k by (6); the bound of this term is the same as that of Λ (1) k . Λ (5) k and Λ (6) k are symmetric to Λ (4) k and Λ (3) k respectively (note that the ¯ π 2 k we constructed in Section 5.1 will be identical to π 2 k in the offline setting). Therefore, we can use the same bounds for the corresponding terms.

Nowweproceed to argue (b) and bound Reg (on) ′ ε . We will largely reuse the regret analysis we already done for Reg (on) T , but only sum up the contribution from phases in K ′ ε .

The contribution to Reg (on) ′ ε from Λ (1) k is

the contribution from Λ (3) k is as shown in (42):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

finally, the contribution from Λ (4) k is as shown in (34):

<!-- formula-not-decoded -->

Reg (on) ′ ε is then bounded by the sum of (47)-(49). By lemma I.1, (47) is bounded by ( √ 2 + 1) √ L ′ ε , and the first term in (48) is bounded by ˜ O ( √ SAL ′ ε ) ˜ O ( D √ S ) = ˜ O ( DS √ AL ′ ε ) by Cauchy inequality. The second term in (48) can be still bounded by ˜ O ( DS 2 A ) . Since the martingale difference sequences in (49) are now summing over a total of L ′ ε steps, (49) is now bounded by DSA + D √ SL ′ ε (cf. (36)).

As a whole, we conclude that Reg (on) ′ ε ≤ ˜ O ( DS √ AL ′ ε + DS 2 A ) , and hence Reg (off) ′ ε ≤ ˜ O ( DS √ AL ′ ε + DS 2 A ) by the argument in (a). Note that by the definition of K ′ ε , we have

<!-- formula-not-decoded -->

Combining (50) with the upper bound of Reg (off) ′ ε just established, we have

which has the solution

<!-- formula-not-decoded -->

Comparing the definitions of L ε and L ′ ε , and by Lemma 5.3, we get

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Finally, we remark on how to select a single stationary policy after we have run the algorithm for T steps. Note that in our proofs, we actually bound the single step regret in phase k through

<!-- formula-not-decoded -->

because LHS is 1 T k ∑ 7 n =1 Λ ( n ) k while RHS is 1 T k ∑ 6 n =2 Λ ( n ) k +2 γ k . Note that the terms on RHS can all be obtained by the algorithm, so they form an available upper bound for the LHS. Let u k denotes the RHS. Then the previous proofs actually proved that

<!-- formula-not-decoded -->

holds with high probability. Therefore, if T &gt; ˜ Ω ( D 3 S 5 A + D 2 S 2 A ε 2 ) , there will be some k such that u k &lt; ε . Since the algorithm knows u k , it can just select the minimum of all u k 's among all phases. That will output a policy π 1 k such that ρ ∗ ( M ) -min π 2 ρ ( M,π 1 k , π 2 ) ≤ ε .

Proof of Theorem 7.2.

<!-- formula-not-decoded -->

where Reg (on) ε is the sum of ∆ k over k ∈ K ε . Of the six regret terms (5), ∆ (4) k dominates over ∆ (1) k , ∆ (3) k , ∆ (5) k , and ∆ (6) k . So we only look at the ∆ (2) k and ∆ (4) k . ∆ (2) k is bounded by T k H D + T k γ k . Summing over k ∈ K ε by Lemma I.1 gives L ε H D + ˜ O ( √ L ε ) . Thus its average error is bounded by ˜ O ( D/H +1 / √ L ε ) . By taking H = D/ (2 ε ) we have the sample complexity for the second term is ˜ O (1 /ε 2 ) . On the other hand, by Theorem K.1, ∆ (4) k has sample complexity bound ˜ O ( HS 2 A/ε 2 ) . By substituting H = D/ (2 ε ) gives the dominating sample complexity bound ˜ O ( DS 2 A/ε 3 ) . We argue again the order of Reg (off) ε does not exceed that of Reg (on) ε . To show this, we note that the extra terms in Reg (off) ε compared to Reg (on) ε are the sum of

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

over k ∈ K ε . This decomposition mirrors that in (5) where ∆ (7) k , ∆ (8) k , ∆ (9) k , ∆ (10) k and ∆ (11) k are symmetric to the ∆ (5) k , ∆ (4) k , ∆ (2) k , ∆ (1) k , and ∆ (6) k in (5), respectively, and we can use the same bounds for the corresponding terms.

Finally, we can pick an ε -optimal policy π 1 k after the algorithm has run for T &gt; ˜ O ( DS 2 A ε 3 ) steps. The way is similar to that described in the proof of Theorem 7.1.

## M Other Technical Lemmas

Remark M.1. Under Assumption 1, note that for any stationary policy π , we have sp( h ( M,π, · )) ≤ T π ( M ) . Indeed,

<!-- formula-not-decoded -->

/negationslash

Remark M.2. Imagine an MDP where all transitions from s = s ′ remain the same while s ′ becomes an absorbing state; rewards on s = s ′ are all 1 and 0 on s ′ . Now max π 1 max π 2 T π 1 ,π 2 s → s ′ ( M ) is equivalent to the maximum reward on this MDP, which can be achieved by stationary joint policy by both players.

/negationslash

## N Regularization/Constraint-based Approach for Assumption 1

It is possible to improve the ˜ O ( D 3 S 5 A ) term in the regret bound under Assumption 1. Note that this term mainly comes from Lemma H.3, which says that to wait until sp( h ( M 1 k , π 1 k , ¯ π 2 k , · )) &lt; 2 D , we need to pay ˜ O ( D 3 S 5 A ) regret. However, if we can know the value of D in advance, the optimistic model M 1 k can be selected based on the following constrained optimization problem:

<!-- formula-not-decoded -->

Clearly, the true model M still lies in this feasible set, so this is a valid way to select M 1 k . It is also possible to convert this into a regularized optimization problem as demonstrated by [3]. Nevertheless, we are not aware of any practical algorithm that can solve either optimization problem. We just demonstrated in this paper that the benefit of this regularization/constraint-based approach is only on the additive constant but not on the asymptotic performance.