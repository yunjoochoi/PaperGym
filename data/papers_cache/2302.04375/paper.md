## A NEAR-OPTIMAL ALGORITHM FOR SAFE REINFORCEMENT LEARNING UNDER INSTANTANEOUS HARD CONSTRAINTS

## Ming Shi, Yingbin Liang, Ness Shroff

Department of Electrical and Computer Engineering

The Ohio State University Columbus, OH 43210, USA {shi.1796,liang.889,shroff.11}@osu.edu

## ABSTRACT

In many applications of Reinforcement Learning (RL), it is critically important that the algorithm performs safely, such that instantaneous hard constraints are satisfied at each step, and unsafe states and actions are avoided. However, existing algorithms for 'safe' RL are often designed under constraints that either require expected cumulative costs to be bounded or assume all states are safe. Thus, such algorithms could violate instantaneous hard constraints and traverse unsafe states (and actions) in practice. Therefore, in this paper, we develop the first near-optimal safe RL algorithm for episodic Markov Decision Processes with unsafe states and actions under instantaneous hard constraints and the linear mixture model. It not only achieves a regret ˜ O ( dH 3 √ dK ∆ c ) that tightly matches the state-of-the-art regret in the setting with only unsafe actions and nearly matches that in the unconstrained setting, but is also safe at each step, where d is the feature-mapping dimension, K is the number of episodes, H is the number of steps in each episode, and ∆ c is a safety-related parameter. We also provide a lower bound ˜ Ω(max { dH √ K, H ∆ 2 c } ) , which indicates that the dependency on ∆ c is necessary. Further, both our algorithm design and regret analysis involve several novel ideas, which may be of independent interest.

## 1 Introduction

Reinforcement learning (RL) has been extensively studied to improve the learning performance in sequential decisionmaking problems for machine learning applications. These decision making problems are usually modelled as a Markov Decision Process (MDP), where an online learner interacts with an unknown environment sequentially to achieve a large expected cumulative reward. Many RL algorithms that do not consider any constraint (and hence are allowed to freely explore any state-action pair) with sample-complexity guarantees have been proposed in the literature [1, 2, 3, 4, 5, 6, 7]. Moreover, existing 'safe' RL algorithms are usually designed under the constraint that requires expected cumulative, i.e., not instantaneous , costs over all steps to be bounded [8, 9, 10, 11] (please see more related work in Section 1.2). Thus, practical scenarios where unsafe states and actions must be avoided at each time/step are not captured.

Instantaneous hard constraints are important in many practical scenarios, and any unsafe states and actions (and transitions) should be avoided at each step. In safety-critical systems, violating such a constraint could result in catastrophic consequences. For example, in power systems, it is well-known that the states of blackouts (e.g., due to violating the power-grid operation constraints) must be avoided [12, 13]. In autonomous driving, improper operations that could cause dangerous states, e.g., crashing, must be avoided [14, 15]. In robotics, even a single bad action could damage the machines and any undesirable state of failure must be avoided [16, 17].

Recently, instantaneous hard constraints have been studied in theoretical machine learning. Specifically, [12] and [18] studied bandits with linear instantaneous constraints that require a linear safety value of the chosen action to be bounded at each step. However, it is well-known that bandits are only a very special case of MDP. [14] studied safe linear MDP with linear instantaneous hard constraints. However, they still assume that only the actions could be unsafe, and hence unsafe states (and transitions) are still not considered. Intuitively, when there are only unsafe actions, any action will always lead to a state in any future step that is safe. Then, we could consider the safety at each step separately. Indeed, the existing idea in such a setting is to estimate the safe actions at each step separately, without the need to consider the impact from other steps . In sharp contrast, when one allows for the more practical scenario when unsafe states can also exist (as done in this paper), even though an action is safe at a step, it may cause unsafe states in subsequent steps. As a result, at each step, the impact from other steps must be carefully handled. This results in significantly new challenges in both the algorithm design and regret analysis.

Therefore, this paper studies a fundamentally important and open question: in MDPs with unsafe states and actions (and transitions) under instantaneous hard constraints, is it possible to design an RL algorithm that not only still achieves a strong sample-complexity guarantee, but is also safe (i.e., satisfies the instantaneous hard constraint) at each step?

## 1.1 Our Contributions

In this paper, we make the first effort to address this question. Specifically, we study episodic MDPs with unsafe states and actions under instantaneous hard constraints and the linear mixture model. We develop an RL algorithm, called Least-Square Value Iteration by lookiNg ahEad and peeking backWard (LSVI-NEW). LSVI-NEW not only achieves a regret ˜ O ( dH 3 √ dK ∆ c ) that tightly matches the state-of-the-art regret in the unsafe-action setting and nearly matches that in the unconstrained setting, but is also safe at each step, where d is the feature-mapping dimension, K is the number of episodes, H is the number of steps in each episode, and ∆ c (which is defined in Theorem 2) is a safety-related parameter. We also provide a lower bound ˜ Ω(max { dH √ K, H ∆ 2 } ) , which indicates that the dependency on ∆ c is necessary.

c

As discussed before, in our case, the coupling between steps need to be carefully handled. To resolve the new challenges due to this coupling, our algorithm in Section 3 involves four important novel ideas. Idea I: constructing safe subgraphs (defined in Section 2.2). Remember that an action that is safe at a step could cause unsafe future states. To resolve this problem, we restrict LSVI-NEW to be inside safe subgraphs of the state-transition diagram. These safe subgraphs are constructed by estimating safe state-sets at each step in a backward manner, such that the chosen action could only result in future states that are estimated to be safe. Idea II: encouraging to explore the transitions with higher uncertainty. Due to our first idea for safety, the choices of actions become restricted. In order to still achieve a sublinear regret, the algorithm needs to be more optimistic in the learning process. To resolve this new pessimism-optimism dilemma, we construct a new bonus term in the estimated Q -value function to encourage LSVI-NEW to explore transitions with higher uncertainty. Idea III: encouraging to explore the future subsubgraphs with higher uncertainty. Idea-II by itself is not sufficient, since each step could be affected by the safety-learning process at future steps. For example, even though the safety function at step h may be precisely known, a bad learning quality at a future step h ′ &gt; h could make the algorithm still not be able to really execute the optimal safe action at step h . To resolve this difficulty, we construct another new bonus term to encourage LSVI-NEW to explore future subsubgraphs with higher uncertainty. Idea IV: encouraging to explore the past subsubgraphs with higher uncertainty. Similar to that in Idea III, since each step h is also affected by past steps h ′ &lt; h , we construct a new bonus term to encourage LSVI-NEW to explore past subsubgraphs with higher uncertainty.

To show a sublinear regret of LSVI-NEW, our regret analysis involves novel ideas for solving the following difficulties. (Please see Section 4 for details.) Difficulty I: the commonly-used invariant in RL relying on the ergodicity property does not hold any more. Due to our special design of the safe subgraphs, the optimal policy and LSVI-NEW may visit different sets of states at each step. Thus, the classical invariant that shows the estimated V -value is larger than the optimal V -value at any state does not hold any more. To resolve this problem, we construct the value functions in a special way so that other useful interesting invariants still hold. Difficulty II: how to quantify the impact from other steps? Our idea is to consider the future and past impacts separately. Then, we could quantify such impacts based on our construction of the safe subgraphs. This way of quantification precisely implies the requirements for the parameters of the new bonus terms that we construct for LSVI-NEW.

## 1.2 Related Work

We provide more related work in this section. To the best of our knowledge, none of existing work has addressed the fundamental open problem that we consider in this paper.

RL with constraints: First, constraints that require some expected cumulative costs over all steps to be bounded have been widely studied in safe RL [19, 20, 21, 8, 22, 23, 24, 9, 25, 26, 10, 27, 28, 11, 29, 30]. Second, many other work, e.g., [31] and [32], studied budget constraints that will halt the learning process whenever the budget has run out of.

Instantaneous hard constraints with only unsafe actions: First, [12, 18] studied safe linear bandits which require a linear safety value of the chosen action to be bounded at each step. Second, [14] studied linear MDPs with instantaneous hard constraints, while assuming only actions could be unsafe.

Instantaneous hard constraints under deterministic transitions: [16] and [17] studied instantaneous hard constraints with unsafe states, while assuming the state transitions are deterministic, i.e., by choosing an action, a state will be transferred to a known single deterministic state.

## 2 Problem Formulation

In this section, we provide the problem formulation and introduce the performance metric.

## 2.1 Episodic MDP Under Instantaneous Hard Constraints and the Linear Mixture Model

We study the constrained episodic MDP, denoted by M = ( S , A , H, P , r, c ) , in an online setting with K episodes, where S and A denote the state and action spaces, respectively; H denotes the number of steps in each episode; P = { P h } H h =1 , r = { r h } H h =1 and c = { c h } H h =1 denote the transition probability function, reward function and safety function, respectively. Let T = HK denote the total number of steps. The learner interacts with the unknown environment as follows. At each step h of episode k , the learner first chooses an action a k h ∈ A for current state s k h . Then, the learner receives a reward r h ( s k h , a k h ) , where r h ( · ) : S × A → [0 , 1] is known. Finally, according to the unknown transition probability function P h ( ·| s k h , a k h ) : S × A × S → [0 , 1] , the environment draws a next state s k h +1 and reveals it to the learner. Meanwhile, the learner observes a noisy safety value ˆ c k h = c h ( s k h , a k h , s k h +1 ) + ζ k h , where c h ( · ) : S × A × S → [0 , 1] is unknown and ζ k h is an additive 0 -mean σ -subGaussian random variable.

Instantaneous hard constraint: At each step h &lt; H of each episode k , the following constraint must be satisfied,

<!-- formula-not-decoded -->

where ¯ c is a known constant, and c h ( s k H ) ≤ ¯ c must be satisfied at step H . The transition from s k h through a k h to s k h +1 is said to be unsafe if constraint (1) is violated. Due to this constraint, some states and actions could also be unsafe.

- A state is said to be unsafe at step h , if there exists no action, such that constraint (1) can be satisfied, i.e., min a ∈A max { s ′ : P h ( s ′ | s,a ) &gt; 0 } c h ( s, a, s ′ ) &gt; ¯ c .
- An action is said to be unsafe for state s at step h , if there is a non-zero probability to transit to a state, such that constraint (1) will be violated, i.e., max { s ′ : P h ( s ′ | s,a ) &gt; 0 } c h ( s, a, s ′ ) &gt; ¯ c .

As discussed in Section 1, due to unsafe states and actions caused by the instantaneous hard constraint, e.g., bad movements and failures in robotics, crushing in autonomous driving and blackouts in power systems, new fundamental difficulties need to be resolved, which is the focus of this paper.

Linear mixture MDP: Due to the ergodicity under the linear function approximation P h ( ·| s, a ) = 〈 µ ∗ h ( · ) , φ ( s, a ) 〉 from [4], any state could be finally visited from any other state. Thus, in such a linear MDP, no algorithm can avoid the unsafe states under constraint (1). Thus, instead we borrow the linear mixture MDP model from [5, 33, 6, 34, 7]. The importance and many applications of linear mixture MDPs have been provided in these references. Specifically, the transition probability P h ( s ′ | s, a ) = 〈 µ ∗ h , φ ( s, a, s ′ ) 〉 and safety value c h ( s, a, s ′ ) = 〈 γ ∗ h , φ ( s, a, s ′ ) 〉 are linear functions of a given feature mapping φ : S × A × S → R d , where µ ∗ h ∈ R d and γ ∗ h ∈ R d are unknown parameters. As typically assumed, for any bounded function V h : S → [0 , H ] and state-action pair ( s, a ) , we have ‖ φ V h ( s, a ) ‖ 2 ≤ D , where φ V h ( s, a ) = ∑ { s ′ : P h ( s ′ | s,a ) &gt; 0 } φ ( s, a, s ′ ) V h ( s ′ ) ∈ R d . Moreover, ‖ µ ∗ h ‖ 2 ≤ L and ‖ γ ∗ h ‖ 2 ≤ L .

## 2.2 State-Action Subgraphs and Performance Metric

Notice that the ergodicity property, (i.e., any state could finally be visited from any other state) in classical MDPs does not hold any more under instantaneous hard constraint (1). This is because if unsafe states can be visited from any other state, it is impossible to satisfy (1) at all steps. Due to this non-ergodicity, we define two important notions below.

First, we let S h ( s, a ) denote the set of next-states that could be transited to with non-zero probability from a state-action pair ( s, a ) at step h , i.e., S h ( s, a ) glyph[defines] { s ′ : P h ( s ′ | s, a ) &gt; 0 } . Similar to that required in the case with deterministic transitions [16] and [17], we assume that the algorithm knows S h ( s, a ) in advance. (Note that the transition kernel P is still unknown .) If S h ( s, a ) is not known in advance, no safe algorithm can achieve a sub-linear regret. This is because (i) if an unsafe state s ′ that will not be transited to is considered for a state-action pair ( s, a ) , the algorithm will lose the chance to explore ( s, a ) . This could result in a linear-toT regret when ( s, a ) is actually optimal. (ii) If an unsafe state s ′ that will be transited to is missed for ( s, a ) , the algorithm will suffer from this unsafe state s ′ when choosing a at state s .

Figure 1: A sketch of subgraph examples. Squares represent states. The red dashed square at step h = 5 is the unsafe state. Circles represent actions. Arrows represent state transitions. There are two actions a = 1 , 2 , as shown by the numbers in the circles.

<!-- image -->

State-action subgraph: While ergodicity does not hold, an important property here is that, by executing a deterministic policy π ( s, h ) : S × [1 , H ] →A , the learner follows a closed directed state-action subgraph

<!-- formula-not-decoded -->

where S π h denotes the set of all states that are visited with non-zero probability by policy π at step h . G may contain only a subset of all states in the global state space S . For simplicity, we assume all episodes start from a fixed state s 1 . (Our results can be easily generalized to the more general case with an arbitrary starting state.)

Please see Figure 1 for a simple sketch of subgraph examples. For example, when choosing action a = 1 at all steps, the learner follows subgraph G 1 . Notice that G 1 is a safe subgraph, since the unsafe state at step h = 5 will not be visited. As another example, the learner follows subgraph G 2 when choosing a = 2 at step h = 1 , choosing a = 1 at step h = 2 , choosing a = 2 for the second state (i.e., the second square from the top when h = 3 ) and a = 1 for the third state (i.e., the third square from the top when h = 3 ) at step h = 3 , and choosing a = 2 at step h = 4 and step h = 5 . Notice that G 2 is an unsafe subgraph, since the unsafe state at h = 5 could be visited. For ease of understanding, in Figure 1, we only draw finite states, two actions and three subgraphs. However, this paper considers the general linear mixture MDP, where the number of states s , actions a and subgraphs G could be infinite.

Performance metric: We let G safe glyph[defines] { G safe } denote the set of all possible safe subgraphs, where all state-action-state triplets satisfy the instantaneous hard constraint (1). Then, the set of all possible safe deterministic policies is

<!-- formula-not-decoded -->

Moreover, the Q -value (state-action-value) function and the V -value (state-value) function are defined as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Algorithm 1 Least-Square Value Iteration by lookiNg ahEad and peeking backWard (LSVI-NEW)

<!-- formula-not-decoded -->

At each step h , first choose the action a k h = a h ( s k h ) in the known seed safe subgraph G safe , 0 , then observe the next state s k h +1 , finally observe the safety value c h ( s k h , a k h , s k h +1 ) .

## end for

for k = K ′ +1 to K do

for h = H to 1 do

Step-1: Update the estimated safety parameter γ k h according to (7) and the estimated safety function ˜ c k h according to (8).

Step-2: Update the estimated safe state-set:

<!-- formula-not-decoded -->

and estimated safe action-set for states s ∈ S k, safe h :

<!-- formula-not-decoded -->

Step-3: Update the parameter w k h according to (12).

## end for

Step-4: Update the estimated Q -values for all state-action pairs ( s, a ) that are estimated to be safe, i.e., s ∈ S k, safe h and a ∈ A k, safe h ( s ) , according to (13).

<!-- formula-not-decoded -->

Step-5: Observe the current state s k , and then choose an action according to (17).

h end for end for

Therefore, our goal is to develop an RL algorithm π glyph[defines] { π k } K k =1 that (i) is safe: π k ∈ Π safe for all k , i.e., constraint (1) is satisfied in all episodes k ; (ii) achieves a sublinear regret, which is defined as

<!-- formula-not-decoded -->

where V ∗ 1 ( s 1 ) is the V -value of the optimal safe policy, i.e.,

<!-- formula-not-decoded -->

## 3 A Near-Optimal Safe Algorithm

In this section, we present our algorithm, called Least-Square Value Iteration by lookiNg ahEad and peeking backWard (LSVI-NEW), as shown in Algorithm 1. Before introducing our algorithm, we present a necessary assumption.

Assumption 1. (Known seed safe subgraph) There exists a known seed safe subgraph G safe , 0 ∈ G safe with the known safety value c 0 h for a state-action-state triplet ( s 0 h , a 0 h , s 0 h +1 ) at each step h of G safe , 0 .

A known seed safe subgraph is necessary for the existence of safe RL algorithms under instantaneous hard constraints. Without it, the unsafe states and actions cannot be avoided in the first episode. Same assumptions on such a known safe set are also made in related work [18, 14]. As pointed out there, such an assumption is realistic since the known safe set can be obtained from existing strategies or trials with possibly low rewards.

Next, we define some notations. First, we let U h glyph[defines] { α φ ( s 0 h , a 0 h , s 0 h +1 ) : α ∈ R } denote the span of the feature φ ( s 0 h , a 0 h , s 0 h +1 ) . Let ψ ( U h , φ 1 ) glyph[defines] 〈 φ 1 , ˜ φ ( s 0 h , a 0 h , s 0 h +1 ) 〉 · ˜ φ ( s 0 h , a 0 h , s 0 h +1 ) denote the projection of a vector φ 1 to U h , where ˜ φ ( s, a, s ′ ) glyph[defines] φ ( s,a,s ′ ) ‖ φ ( s,a,s ′ ) ‖ 2 is the normalized vector of φ ( s, a, s ′ ) . Second, we let U ⊥ h glyph[defines] { φ 3 ∈ R d : 〈 φ 3 , φ 2 〉 = 0 , ∀ φ 2 ∈ U h } denote the orthogonal complement of U h . Let ψ ( U ⊥ h , φ 1 ) glyph[defines] φ 1 -ψ ( U h , φ 1 ) denote the projection of φ 1 to U ⊥ h . Third, we let φ k h,h +1 = φ ( s k h , a k h , s k h +1 ) denote the feature vector of the state-action-state triplet ( s k h , a k h , s k h +1 ) . Let ‖ x ‖ Λ = √ x T Λ x denote the weighted 2 -norm of x with respect to Λ . Let I denote the identity matrix.

Our LSVI-NEW algorithm contains a simple initialization phase and a more important learning phase that involves four novel ideas . In the initialization phase, LSVI-NEW purely explores inside the known seed safe subgraph G safe , 0 , i.e., the first for-loop in Algorithm 1, where K ′ is a tunable parameter. This initialization phase borrows the idea in bandits with instantaneous hard constraints for obtaining and preparing some parameter information for the later learning phase [12].

From now on, we focus on introducing the five steps in the learning phase (i.e., the second for-loop in Algorithm 1) that involves four important novel ideas. In Step-1, LSVI-NEW updates the regularized least-square estimator of the projected safety parameter ψ ( U ⊥ h , γ ∗ h ) as follows:

<!-- formula-not-decoded -->

where the Gram matrix Λ k h, 1 = λ ψ ( U ⊥ h , I ) + ∑ k -1 τ =1 ψ ( U ⊥ h , φ τ h,h +1 ) ψ T ( U ⊥ h , φ τ h,h +1 ) , ψ ( U ⊥ h , I ) = I -˜ φ ( s 0 h , a 0 h , s 0 h +1 ) ˜ φ T ( s 0 h , a 0 h , s 0 h +1 ) , ψ ( U ⊥ h , ˆ c τ h ) = ˆ c τ h -〈 ψ ( U h , φ τ h,h +1 ) , ˜ φ ( s 0 h ,a 0 h ,s 0 h +1 ) 〉 ‖ φ ( s 0 h ,a 0 h ,s 0 h +1 ) ‖ 2 · c 0 h and λ ≥ d is a tunable parameter. Then, we estimate the safety function as follows:

<!-- formula-not-decoded -->

where φ 1 = φ ( s, a, s ′ ) and β is a tunable parameter given in Theorem 2. Notice that, on the right-hand-side (RHS) of (8), the first term is the projected safety value of ( s, a, s ′ ) on U h , the second term is the projected empirical safety value of ( s, a, s ′ ) on U ⊥ h , and the last term is an upper-confidence-bound (UCB) bonus for the safety uncertainty. Thus, the accuracy of the safety value ˜ c k h depends on how accurate γ k h in (7) is and how small the safety uncertainty is. Next, Step-2 in Algorithm 1 is based on ˜ c k h and involves our first novel idea that is critical for guaranteeing safety.

Idea I: Constructing safe subgraphs by looking ahead. As we discussed in Section 1, in bandits and RL with only unsafe actions, the safety at each step can be estimated separately . In sharp contrast, due to the unsafe states and transitions in our setting, we must handle possible unsafe future steps. Consider Figure 1 as an example. Even though taking action a = 1 for the third state (the third square from the top) at step h = 3 is safe for h = 3 , by doing so, the unsafe state (the red dashed square) at h = 5 will be visited no matter what action would be taken at h = 4 . To resolve this new challenge, our idea is to construct special safe subgraphs where any action only results in safe future (not even just next) states. To achieve this, in Step-2, we estimate the safe state-set S k, safe h and action-set A k, safe h ( s ) in a backward manner based on the two conditions below:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice that, (i) condition 1 requires that by choosing action a for state s , the instantaneous hard constraint is always satisfied at step h ; (ii) condition 2 requires that all possible next states in S h ( s, a ) must be safe for next step h +1 . Thus, with conditions 1 and 2 satisfied simultaneously in a backward manner, all (not just next) steps h ′ ≥ h following ( s, a ) must be safe. Please see Theorem 1 for the safety performance of LSVI-NEW at all steps in any episode.

Moreover, since the linear mixture MDP induces a linear form of the Q -value function as follows:

<!-- formula-not-decoded -->

in Step-3 of Algorithm 1, we update the regularized least-square estimator of the parameter w ∗ h in (11) as follows:

<!-- formula-not-decoded -->

where the Gram matrix Λ k h, 2 = λ I + ∑ k -1 τ =1 φ τ h,V φ τ, T h,V and φ τ h,V = φ V ( s τ h , a τ h ) . Then, in Step-4 of Algorithm 1, we update the Q -values of the safe state-action pairs as follows:

<!-- formula-not-decoded -->

where glyph[epsilon1] 1 = β +1 , glyph[epsilon1] h, 2 , glyph[epsilon1] h, 3 and glyph[epsilon1] 4 are given soon later, and G h ( s ) is the set of subsubgraphs starting from state s at step h . Notice that (i) the term with glyph[epsilon1] 1 on the RHS of (13) is the standard Hoeffding bonus term; (ii) the terms with glyph[epsilon1] h, 2 , glyph[epsilon1] h, 3 and glyph[epsilon1] 4 are three new bonus terms that we construct for capturing the impacts from future and past steps. We elaborate our novel ideas in these new bonus terms below.

Idea II: Encouraging to explore the transitions with higher uncertainty (i.e., looking ahead). As we mentioned in Section 1, there is a new pessimism-optimism dilemma in our setting. Specifically, according to the optimism-inface-of-uncertainty principle [1], algorithms need to learn optimistically to achieve a sublinear regret. However, to avoid the unsafe states and transitions in our setting, algorithms have to be relatively pessimistic. To resolve this new dilemma, we construct a bonus term to encourage LSVI-NEW to explore the transitions with higher uncertainty. To achieve this, this new bonus term, i.e., the term with glyph[epsilon1] h, 2 in (13), is designed to be the maximum UCB bonus over all possible next-states s ′ ∈ S h ( s, a ) .

Then, another new difficulty here is how to quantify the parameter glyph[epsilon1] h, 2 for such a bonus term, such that a sublinear regret can be achieved. To resolve this problem, we set

<!-- formula-not-decoded -->

where ¯ c 0 h ′ = max h ≤ h ′ ≤ H c 0 h ′ , ∆ φ ( c ) = L · max s,a,h max s ′ ,s ′′ ∈S h ( s,a ) ‖ φ ( s, a, s ′ ) -φ ( s, a, s ′′ ) ‖ 2 , and ˜ δ and κ are scalars given in Theorem 2. Notice that when all states are assumed to be safe, all terms related to next state s ′ would be 0 . Then, glyph[epsilon1] h, 2 would be 4 βH ¯ c -c 0 h , which is the same as the parameter used in the setting with only unsafe actions [14]. However, one difference here is that we need to handle the worst transition. Thus, the denominator needs to capture the smallest safety balance, i.e., ¯ c -¯ c 0 h ′ -∆ φ ( c ) , that is left for exploration. Another difference is that even though the safety balance at current step is small, if the safety balance in future steps is large, the algorithm should still be encouraged to explore. To capture such a new special impact from future steps, we add the term ¯ c -¯ c 0 h ′ -∆ φ ( c ) ¯ c -c 0 h -∆ φ ( c ) , such that glyph[epsilon1] h, 2 increases with the ratio between future safety balance ¯ c -¯ c 0 h ′ -∆ φ ( c ) and current balance ¯ c -c 0 h -∆ φ ( c ) . Please see Appendix B for details.

Idea III: Encouraging to explore the future subsubgraphs with higher uncertainty (i.e., looking ahead). Idea II by itself is not sufficient to achieve a sublinear regret. This is because future uncertainty could prevent the algorithm from choosing the optimal action at current step. Consider Figure 1 as an example and assume G 1 is the optimal subgraph. Even though the safety value at h = 1 has been precisely known, the algorithm may still not choose the optimal action a = 1 due to future uncertainty, e.g., it is uncertain whether the first two states at h = 2 are safe or not. This is another critical difference compared with the case without instantaneous constraints or with only unsafe actions. Hence, at each step, the algorithm should be encouraged to explore the state that induces a future subsubgraph with higher uncertainty. To achieve this, we construct a new bonus term (the term with glyph[epsilon1] h, 3 in (13)) that is the maximum UCB bonus over all future subsubgraphs G h ( s ) , where

<!-- formula-not-decoded -->

Differently from glyph[epsilon1] h, 2 in (14), the term ¯ c -¯ c 0 h ′ -∆ φ ( c ) ¯ c -c 0 h -∆ φ ( c ) does not appear in glyph[epsilon1] h, 3 , because the maximization in this bonus term is taken over all states and actions in G h ( s ) , which already captures the impacts from future steps.

Idea IV: Encouraging to explore the past subsubgraphs with higher uncertainty (i.e., peeking backward). Surprisingly, with Ideas II and III alone, a sublinear regret may still not be achieved. This is because of the tricky impact from past steps. Intuitively, by choosing a different action at step h = 1 , what will happen in future steps could be completely different. To resolve this new challenge, we construct a new bonus term, i.e., the term with glyph[epsilon1] 4 in (13), to encourage LSVI-NEW to explore the past subsubgraphs with higher uncertainty, where

<!-- formula-not-decoded -->

Differently from glyph[epsilon1] h, 3 in (15), the denominator here depends on c 0 1 (not ¯ c 0 h ′ ) at step h = 1 that affects all future steps. Finally, in Step-5, LSVI-NEW chooses an action

<!-- formula-not-decoded -->

## 4 Theoretical Results

In this section, we provide the safety and regret guarantees for our LSVI-NEW algorithm, and a regret lower-bound.

Before these, we make two necessary assumptions for obtaining good theoretical performance in our setting. Assumption 2 below is from [14]. The counter-example given there shows that such an assumption is required for the existence of safe algorithms with sublinear regrets. We let Φ α ( s, a ) glyph[defines] [ α ( s ′ ) φ ( s, a, s ′ )] s ′ ∈S ( s,a ) denote a matrix with α ( s ′ ) φ ( s, a, s ′ ) in each column, where α ( s ′ ) is a scalar.

Assumption 2. (Star convexity) For all states s h at step h , the set D ( s h ) glyph[defines] { Φ 1 ( s h , a ) : a ∈ A} ∪ { Φ 1 ( s 0 h , a 0 h ) : Φ 1 ( s 0 h , a 0 h , · ) = φ ( s 0 h , a 0 h , s 0 h +1 ) } is a star convex set around the safe feature φ ( s 0 h , a 0 h , s 0 h +1 ) , i.e., for all Φ 1 ( s h , a ) ∈ D ( s h ) and α : S h ( s h , a ) → [0 , 1] with ‖ α ‖ 1 = 1 , we have Φ α ( s h , a ) + Φ 1 -α ( s 0 h , a 0 h ) ∈ D ( s h ) , where 1 denotes a vector with all entries equal to 1 .

Next, we let f h ( φ 1 -φ 2 ) glyph[defines] ‖ φ 1 -φ 2 ‖ 2 ‖ φ ( s ∗ h ,a ∗ h ,s ∗ h +1 ) -φ ( s 0 h ,a 0 h ,s 0 h +1 ) ‖ 2 denote the L 2 -distance between features φ 1 and φ 2 , normalized by the L 2 -distance between the unknown optimal feature φ ( s ∗ h , a ∗ h , s ∗ h +1 ) and the known safe feature φ ( s 0 h , a 0 h , s 0 h +1 ) at step h . Let g ( r h, 1 -r h, 2 ) glyph[defines] r h, 1 -r h, 2 r h ( s ∗ h ,a ∗ h ) denote the reward difference r h, 1 -r h, 2 , normalized by the reward of the unknown optimal state-action pair at step h .

Assumption 3. (Lipschitz rewards and transitions) There exists δ ∈ [0 , 1] , s.t., for any two safe state-action pairs ( s ( i ) , a ( i )) and ( s ( j ) , a ( j )) at step h ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ( s h ′ ( i ) , a h ′ ( i )) ( h ′ &gt; h ) is the descendant of the state-action pair ( s ( i ) , a ( i )) in the safe subgraphs.

Note that (18) implies that rewards are δ -Lipschitz: as feature differences (RHS of (18)) become smaller, reward differences (LHS of (18)) become smaller; and (19) implies that safe transitions are δ -Lipschitz: as feature differences at current step (RHS of (19)) become smaller, feature differences at future steps h ′ (LHS of (19)) become smaller.

When the unsafe states and transitions are taken into consideration, to still achieve a sublinear regret, Assumption 3 is required. This is because (i) if rewards are not Lipschitz, even though a feature vector close to the optimal one is learned to be safe, the learner could still suffer from a large reward gap compared with the optimal safe decision, which could result in a linear-toT regret; (ii) if safe transitions are not Lipschitz, even though the optimal safe decision at a step has been learned , the learner could still be far away from optimum in future steps , and hence suffer from a large reward gap, which could also result in a linear-toT regret.

## 4.1 Performance Guarantees and A Lower Bound

In Theorem 1 below, we show that LSVI-NEW is safe.

Theorem 1. (Safety) For any p ∈ (0 , 1) , with probability 1 -p , our LSVI-NEW algorithm satisfies the instantaneous hard constraint (1) at all steps h of all episodes k .

Thanks to our Idea I in Section 3 for guaranteeing safety, the proof of Theorem 1 (in Appendix A) focuses on quantifying the accuracy of the estimated safety value in (8). Below, Theorem 2 provides the regret upper-bound of LSVI-NEW.

<!-- formula-not-decoded -->

K ′ = 4 βD √ T log ( d p ) , where T = HK , κ = 4 βD λ + λ 0 K ′ and ∆ c = ¯ c -¯ c 0 1 -∆ φ ( c ) , then there exist absolute constants b β &gt; 0 and λ 0 &gt; 0 , with probability 1 -p , the regret of LSVI-NEW is upper-bounded as follows:

<!-- formula-not-decoded -->

The regret in (20) is dominated by the first term on the RHS of (20) that results from the aforementioned new challenges due to the instantaneous hard constraint. Thus, incorporated with the values of the parameters, Theorem 2 indicates that the regret of LSVI-NEW is upper-bounded by ˜ O ( dH 3 √ dK ¯ c -¯ c 0 1 -∆ φ ( c ) ) . Notably, it tightly matches the state-of-the-art regret

˜ O ( dH 3 √ dK ¯ c -¯ c 0 1 -∆ φ ( c ) ) in the setting with only unsafe actions [14] and nearly matches that ˜ O ( dH 2 √ K ) in the unconstrained linear mixture MDP [5]. To the best of our knowledge, this is the first such result in the literature. Further, we provide a lower bound in Theorem 3 below that shows that the dependency on the safety term ¯ c -¯ c 0 1 -∆ φ ( c ) is necessary.

Theorem 3. (A lower bound) Assuming K ≥ 32 R . The regret of any safe algorithm π is lower-bounded as follows:

<!-- formula-not-decoded -->

Theorem 3 implies that the dependency of the regret of LSVI-NEW on ¯ c -¯ c 0 1 -∆ φ ( c ) is necessary. In addition, the regret of LSVI-NEW matches the lower bound within a factor of ˜ O ( H 2 √ d ) . Same as in the setting with only unsafe actions, we conjecture that this gap can be further reduced by applying Bernstein inequality and leave this as future work. Please see Appendix F for the proof.

## 4.2 Proof Sketch for Theorem 2

In this subsection, we provide the high-level ideas for proving Theorem 2 (please see Appendix E for the proof). Because of the new challenges from instantaneous hard constraints and our novel ideas in the algorithm design, there are several new difficulties in the regret analysis. The key ones are: (I) Differently from MDPs without constraints or with only unsafe actions, in our case, different policies could visit very different sets of states at each step. Hence, the commonly-used invariant on V -values that relies on the ergodicity property no longer holds. (II) How to quantify the impacts when looking ahead and peeking backward. Below, we introduce our new analytical ideas, which may be of independent interest.

Step-I: Solving difficulty I by constructing new invariants. We construct new forms of V -value functions for different policies below. We let S ∗ h denote the state set at step h in the optimal safe subgraph. Let S k h denote the state set at step h in the subgraph followed by policy π k of LSVI-NEW in episode k . Moreover, we let ˜ f h ( s, a ) glyph[defines] f h ( φ ( s, a, · ) -φ ( s ∗ h , a ∗ h , · )) denote the gap of transitions compared with optimal transitions. Let ˜ A k h ( s ) glyph[defines] { a ∈ A k, safe h ( s ) : ˜ f h ( s, a ) ≤ ¯ α 0 }∪{ a k h ( s ) } capture the safe actions with transitions close to the optimal transitions, where ¯ α 0 is the maximum of α 0 in (28) and the RHS of (29). Let ˜ S k h glyph[defines] { s ∈ S k, safe h : ∃ a ∈ A k, safe h ( s ) , s.t., ˜ f h ( s, a ) ≤ ¯ α 0 } ∪ S k h capture the safe states with transitions close to the optimal transitions. Next, we define the V -value functions of the optimal policy, estimated policy and policy π k to be

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

respectively. Then, the regret R LSVI-NEW can be decomposed into two parts, i.e., the values in the two brackets [ · ] below,

<!-- formula-not-decoded -->

To upper-bound the regret, we prove that, with high probability, (i) the value in the first bracket of (25) is non-positive; (ii) the value in the second bracket can be upper-bounded. Result (ii) can be obtained by upper-bounding the bonus terms, which can further be proven by slightly modifying existing techniques in linear mixture MDP. The main difficulty is to prove result (i). To resolve this difficulty, we construct two new invariants that hold at each step.

Lemma 1. (New invariants) At each step h of each episode, (i) for any state s , s.t., s ∈ S ∗ and s ∈ ˜ S k , we have

<!-- formula-not-decoded -->

(ii) for any state s , s.t., s ∈ S ∗ h and s / ∈ ˜ S k h , and any state ˆ s , s.t., ˆ s ∈ ˜ S k h and ˆ s / ∈ S ∗ h , we have

<!-- formula-not-decoded -->

Invariant (i) shows that, if the optimal state has been found, the estimated V -value must be higher than the optimal V -value. Notice that if the optimal safe action has also been found, (26) trivially holds. If it has not been found, thanks

h h to our new bonus terms that essentially capture the distance from the optimal action, (26) still holds. Moreover, invariant (ii) shows that, if the optimal state has not been found, the V -value of the sub-optimal state in ˜ S k h is still larger than the optimal V -value. This is intuitively because ˜ S k h only contains safe states with transitions close to the optimal transitions, and the distance is captured by our new bonus terms. Please see Appendix D for details and the proof.

Step-II: Solving difficulty II by quantifying future impacts. The impact when looking ahead can be characterized by quantifying the impacts from future steps.

Lemma 2. (Impacts from future steps) For any state s , s.t., s ∈ S ∗ h and s ∈ ˜ S k h , if a ∗ h ( s ) / ∈ ˜ A k h ( s ) , there must exist an action a 0 ∈ ˜ A k h ( s ) , s.t.,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Lemma 2 implies that when k increases, the UCB terms l 1 and l 2 decrease to be closer to 0 , and thus α 0 gets closer to 0 . Then, the gap between LSVI-NEW's decision and the optimal decision, i.e., ˜ f h ( s, a 0 | s ∗ h = s ) on the LHS of (28), gets closer to 0 . This is consistent with the intuition that as more safety values revealed, we should be able to get closer to the optimal action. Moreover, when there is no constraint on states, all terms related to the next state s ′ in α 0 would be 0 . Then, α 0 would be reduced to be 1 -¯ c -c 0 h -2 β ‖ φ ( s,a ∗ h ( s )) ‖ ¯ c -c 0 h , which results in a parameter same to that used in the case with only unsafe actions [14]. However, due to unsafe states and transitions, impacts from future steps h ′ &gt; h are captured in α 0 here, which results in a different parameter glyph[epsilon1] h, 2 in our Idea II and a new parameter glyph[epsilon1] h, 3 in Idea III. Please see Appendix B for details and the proof.

Step-III: Solving difficulty II by quantifying past impacts. The impact when peeking backward can be characterized by quantifying the impacts from past steps.

Lemma 3. (Impacts from past steps) For any state ˆ s , s.t., ˆ s ∈ ˜ S k h and ˆ s / ∈ S ∗ h , there must exist an action a 0 ∈ ˜ A k h (ˆ s ) and 1 ≤ h ′ ≤ h , s.t.,

<!-- formula-not-decoded -->

where l 3 = 2 β max s ′ ‖ ψ ( U ⊥ h ′ , φ ( s ∗ h ′ , a ∗ h ′ , s ′ )) ‖ ( Λ k h ′ , 1 ) -1 .

Differently from Lemma 2, Lemma 3 quantifies the impacts from past steps, i.e., h ′ ≤ h . This special impact results in the new bonus term with parameter glyph[epsilon1] 4 in our Idea IV in Section 3. These are also the reasons all glyph[epsilon1] h, 2 , glyph[epsilon1] h, 3 and glyph[epsilon1] 4 are different from the parameter used in the setting with only unsafe actions [14]. Please see Appendix C for the proof.

## 5 Conclusion

In this paper, we make the first effort to resolve the challenges due to unsafe states and actions under instantaneous hard constraints in RL. We develop an RL algorithm that not only achieves a regret that tightly matches the state-of-the-art regret in the setting with only unsafe actions and nearly matches that in the unconstrained setting, but also is safe (i.e., satisfies the instantaneous hard constraint) at each step. We also provide a lower bound of the regret that indicates that the dependency of the regret of our algorithm on the safety parameters is necessary. Further, both our algorithm design and regret analysis involve several novel ideas, which may be of independent interest.

## References

- [1] Mohammad Gheshlaghi Azar, Ian Osband, and Rémi Munos. Minimax regret bounds for reinforcement learning. In International Conference on Machine Learning , pages 263-272. PMLR, 2017.
- [2] Chi Jin, Zeyuan Allen-Zhu, Sebastien Bubeck, and Michael I Jordan. Is q-learning provably efficient? Advances in neural information processing systems , 31, 2018.
- [3] Alekh Agarwal, Nan Jiang, Sham M Kakade, and Wen Sun. Reinforcement learning: Theory and algorithms. CS Dept., UW Seattle, Seattle, WA, USA, Tech. Rep , pages 10-4, 2019.

- [4] Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory , pages 2137-2143. PMLR, 2020.
- [5] Zeyu Jia, Lin Yang, Csaba Szepesvari, and Mengdi Wang. Model-based reinforcement learning with value-targeted regression. In Learning for Dynamics and Control , pages 666-686. PMLR, 2020.
- [6] Dongruo Zhou, Jiafan He, and Quanquan Gu. Provably efficient reinforcement learning for discounted mdps with feature mapping. In International Conference on Machine Learning , pages 12793-12802. PMLR, 2021.
- [7] Jiafan He, Dongruo Zhou, and Quanquan Gu. Near-optimal policy optimization algorithms for learning adversarial linear mixture mdps. In International Conference on Artificial Intelligence and Statistics , pages 4259-4280. PMLR, 2022.
- [8] Tsung-Yen Yang, Justinian Rosca, Karthik Narasimhan, and Peter J Ramadge. Projection-based constrained policy optimization. In International Conference on Learning Representations , 2019.
- [9] Kianté Brantley, Miro Dudik, Thodoris Lykouris, Sobhan Miryoosefi, Max Simchowitz, Aleksandrs Slivkins, and Wen Sun. Constrained episodic reinforcement learning in concave-convex and knapsack settings. Advances in Neural Information Processing Systems , 33:16315-16326, 2020.
- [10] Dongsheng Ding, Xiaohan Wei, Zhuoran Yang, Zhaoran Wang, and Mihailo Jovanovic. Provably efficient safe exploration via primal-dual policy optimization. In International Conference on Artificial Intelligence and Statistics , pages 3304-3312. PMLR, 2021.
- [11] Santiago Paternain, Miguel Calvo-Fullana, Luiz FO Chamon, and Alejandro Ribeiro. Safe policies for reinforcement learning via primal-dual methods. IEEE Transactions on Automatic Control , 2022.
- [12] Sanae Amani, Mahnoosh Alizadeh, and Christos Thrampoulidis. Linear stochastic bandits under safety constraints. Advances in Neural Information Processing Systems , 32, 2019.
- [13] Yuanyuan Shi, Guannan Qu, Steven Low, Anima Anandkumar, and Adam Wierman. Stability constrained reinforcement learning for real-time voltage control. In 2022 American Control Conference (ACC) , pages 2715-2721. IEEE, 2022.
- [14] Sanae Amani, Christos Thrampoulidis, and Lin Yang. Safe reinforcement learning with linear function approximation. In International Conference on Machine Learning , pages 243-253. PMLR, 2021.
- [15] Kyriakos G Vamvoudakis, Yan Wan, Frank L Lewis, and Derya Cansever. Handbook of Reinforcement Learning and Control . Springer, 2021.
- [16] Matteo Turchetta, Felix Berkenkamp, and Andreas Krause. Safe exploration in finite markov decision processes with gaussian processes. Advances in Neural Information Processing Systems , 29, 2016.
- [17] Akifumi Wachi, Yanan Sui, Yisong Yue, and Masahiro Ono. Safe exploration and optimization of constrained mdps using gaussian processes. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 32, 2018.
- [18] Aldo Pacchiano, Mohammad Ghavamzadeh, Peter Bartlett, and Heinrich Jiang. Stochastic bandits with linear constraints. In International Conference on Artificial Intelligence and Statistics , pages 2827-2835. PMLR, 2021.
- [19] Yifan Wu, Roshan Shariff, Tor Lattimore, and Csaba Szepesvári. Conservative bandits. In International Conference on Machine Learning , pages 1254-1262. PMLR, 2016.
- [20] Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In International conference on machine learning , pages 22-31. PMLR, 2017.
- [21] Chen Tessler, Daniel J Mankowitz, and Shie Mannor. Reward constrained policy optimization. In International Conference on Learning Representations , 2018.
- [22] Yonathan Efroni, Shie Mannor, and Matteo Pirotta. Exploration-exploitation in constrained mdps. arXiv preprint arXiv:2003.02189 , 2020.
- [23] Rahul Singh, Abhishek Gupta, and Ness B Shroff. Learning in markov decision processes under constraints. arXiv preprint arXiv:2002.12435 , 2020.
- [24] Dongsheng Ding, Kaiqing Zhang, Tamer Basar, and Mihailo Jovanovic. Natural policy gradient primal-dual method for constrained markov decision processes. Advances in Neural Information Processing Systems , 33:83788390, 2020.
- [25] Krishna C Kalagarla, Rahul Jain, and Pierluigi Nuzzo. A sample-efficient algorithm for episodic finite-horizon mdp with constraints. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 35, pages 8030-8037, 2021.

- [26] Tao Liu, Ruida Zhou, Dileep Kalathil, Panganamala Kumar, and Chao Tian. Learning policies with zero or bounded constraint violation for constrained mdps. Advances in Neural Information Processing Systems , 34:17183-17193, 2021.
- [27] Honghao Wei, Xin Liu, and Lei Ying. A provably-efficient model-free algorithm for constrained markov decision processes. arXiv preprint arXiv:2106.01577 , 2021.
- [28] Tengyu Xu, Yingbin Liang, and Guanghui Lan. Crpo: A new approach for safe reinforcement learning with convergence guarantee. In International Conference on Machine Learning , pages 11480-11491. PMLR, 2021.
- [29] Qinbo Bai, Amrit Singh Bedi, Mridul Agarwal, Alec Koppel, and Vaneet Aggarwal. Achieving zero constraint violation for constrained reinforcement learning via primal-dual approach. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 36, pages 3682-3689, 2022.
- [30] Arnob Ghosh, Xingyu Zhou, and Ness Shroff. Provably efficient model-free constrained rl with linear function approximation. arXiv preprint arXiv:2206.11889 , 2022.
- [31] Constantine Caramanis, Nedialko B Dimitrov, and David P Morton. Efficient algorithms for budget-constrained markov decision processes. IEEE Transactions on Automatic Control , 59(10):2813-2817, 2014.
- [32] Di Wu, Xiujun Chen, Xun Yang, Hao Wang, Qing Tan, Xiaoxun Zhang, Jian Xu, and Kun Gai. Budget constrained bidding by model-free reinforcement learning in display advertising. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management , pages 1443-1451, 2018.
- [33] Dongruo Zhou, Quanquan Gu, and Csaba Szepesvari. Nearly minimax optimal reinforcement learning for linear mixture markov decision processes. In Conference on Learning Theory , pages 4532-4576. PMLR, 2021.
- [34] Dongruo Zhou and Quanquan Gu. Computationally efficient horizon-free reinforcement learning for linear mixture mdps. Advances in Neural Information Processing Systems , 35, 2022.
- [35] Yasin Abbasi-Yadkori, Dávid Pál, and Csaba Szepesvári. Improved algorithms for linear stochastic bandits. Advances in neural information processing systems , 24, 2011.
- [36] Emilie Kaufmann, Olivier Cappé, and Aurélien Garivier. On the complexity of best-arm identification in multiarmed bandit models. The Journal of Machine Learning Research , 17(1):1-42, 2016.
- [37] Tor Lattimore and Csaba Szepesvári. Bandit algorithms . Cambridge University Press, 2020.

## A Proof of Theorem 1

Remember that our Idea I in Section 3 is mainly designed for guaranteeing safety. As we discussed there, (i) condition 1 in (9) implies that by choosing action a for state s at step h , the instantaneous hard constraint is guaranteed to be satisfied at step h ; (ii) condition 2 in (10) implies that all possible next states in S h ( s, a ) (i.e., the next states that could be visited with non-zero probability) must be safe for next step h +1 . Thus, with conditions 1 and 2 satisfied simultaneously in a backward manner, all step h ′ ≥ h (not even just next step h +1 ) following ( s, a ) must be safe. Hence, the probability of our LSVI-NEW algorithm being safe depends on the accuracy of the estimated safety value ˜ c k h in (8).

Moreover, remember that, on the RHS of (8), the first term is the projected safety value of ( s, a, s ′ ) on U h , the second term is the projected empirical safety value of ( s, a, s ′ ) on U ⊥ h , and the last term is a UCB bonus for the safety uncertainty. In addition, the second term there relies on the accuracy of the regularized least-square estimator of the projected safety parameter ψ ( U ⊥ h , γ ∗ h ) . Thus, the accuracy of ˜ c k h further depends on how accurate γ k h in (7) is and how small the safety uncertainty is.

Therefore, we first prove Lemma 4 below for quantifying the accuracy of the estimated safety parameter γ k h in (7).

Lemma 4. (Accuracy of the estimated safety parameter) For any p ∈ (0 , 1) , with probability 1 -p , we have that, for all steps h of all episode k ,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

A Near-Optimal Algorithm for Safe Reinforcement Learning Under Instantaneous Hard Constraints

Proof. (Proof of Lemma 4) First, according to (7), we have that the estimated safety parameter is equal to

<!-- formula-not-decoded -->

By opening the bracket [ · ] , and adding and subtracting the term λ ψ ( U ⊥ h , I ) , we have

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

According to (31), the square of the left-hand-side of (30) is equal to

<!-- formula-not-decoded -->

Then, according to the Cauchy-Schwarz inequality, we have

<!-- formula-not-decoded -->

Notice that the smallest eigenvalue of Λ k h, 1 is λ min ( Λ k h, 1 ) = λ . Hence, according to Theorem 1 in [35], we have that, with probability 1 -p for any p ∈ (0 , 1) ,

<!-- formula-not-decoded -->

Finally, by rearranging the terms in (32), we have

<!-- formula-not-decoded -->

This concludes the proof of Lemma 4.

Lemma 4 shows that with high probability, the estimated safety parameter γ k h is close enough to the projected true safety parameter ψ ( U ⊥ h , γ ∗ h ) . Now, we prove Theorem 1 based on our Idea I in Section 3 and Lemma 4 above.

Proof. (Proof of Theorem 1) We let G k, safe h denote the set of safe subsubgraphs constructed at step h in episode k by LSVI-NEW using our Idea I. Then, using mathematical induction, we prove that G k, safe h is safe, i.e., any state-actionstate triplet ( s k h ′ , a k h ′ , s k h ′ +1 ) , where h ≤ h ′ ≤ H , in G k, safe h satisfies the instantaneous hard constraint (1).

(i) Base case: when h = H , according to Lemma 4 and the Cauchy-Schwarz inequality, we have

<!-- formula-not-decoded -->

From (33), we have

<!-- formula-not-decoded -->

Next, since the left-hand-side of (34) is equal to

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

Notice that, since the parameter β used for the estimated safety value ˜ c k H ( s k H ) in (8) is larger than or equal to β 1 , the right-hand-side of (35) is less than or equal to ˜ c k H ( s k H ) , which is less than or equal to ¯ c due to our condition 1 in (9). Hence, we have c H ( s k H ) ≤ ¯ c .

(ii) Induction step: we hypothesize that G k, safe h is safe when h = h 0 . Then, we prove that G k, safe h is safe for h = h 0 -1 similar to the base case, while condition 2 that we construct in (10) becomes important here. First, according to Lemma 4 and the Cauchy-Schwarz inequality, we have

<!-- formula-not-decoded -->

From (36), we have

<!-- formula-not-decoded -->

Next, since the left-hand-side of (37) is equal to

<!-- formula-not-decoded -->

we have

<!-- formula-not-decoded -->

Notice that, the right-hand-side of (38) is less than or equal to the estimated safety value ˜ c k h ( s k h , a k h , s k h +1 ) in (8), which is less than or equal to ¯ c due to our condition 1 in (9). Thus, we have c h ( s k h ) ≤ ¯ c . In addition, according to condition 2 that we construct in (10) and the induction hypothesis, s h +1 must also be safe. Hence, G k, safe h is safe.

## B Proof of Lemma 2

As we discussed in Section 4.2, Lemma 2 implies that when k increases, the UCB terms l 1 and l 2 decrease to be closer to 0 , and thus α 0 on the right-hand-side of (28) gets closer to 0 . Then, ˜ f h ( s, a 0 | s ∗ h = s ) on the left-hand-side of (28) gets closer to 0 . Notice that ˜ f h ( s, a 0 | s ∗ h = s ) represents the gap between the decision of the policy π k used by LSVI-NEW and the optimal decision. In addition, in α 0 , l 1 characterizes the transition uncertainty and l 2 characterizes the uncertainty from future steps. Thus, the above implication from Lemma 2 is consistent with the intuition that as more safety values revealed, we should be able to get closer to the optimal action.

Moreover, when there is no constraint on states, all terms related to the next state s ′ in α 0 , e.g., l 2 , ¯ c 0 h ′ and ∆ φ ( c ) , would be 0 . Then, α 0 would be reduced to be in a much simpler form 1 -¯ c -c 0 h -2 β ‖ φ ( s,a ∗ h ( s )) ‖ ¯ c -c 0 h , which results in a parameter that is same to that used for the UCB bonus term in the case with only unsafe actions [14]. However, due to unsafe states and transitions in our case, the impacts from the future steps h ′ &gt; h are characterized in α 0 here, which results in a different parameter glyph[epsilon1] h, 2 in our Idea II and a new parameter glyph[epsilon1] h, 3 in our Idea III in Section 3.

Further, as stated in Lemma 2, we only need to show there exists such a safe action a 0 ∈ ˜ A k h ( s ) . Thus, we only need to prove the existence of an estimated safe subgraph, such that this state-action pair ( s, a 0 ) is contained. Hence, (28) does not depends on the estimation accuracy of the Q -value parameter w ∗ h .

In this section, we provide the complete proof for Lemma 2. Please see Appendix D for our discussions and proofs on how the new impacts from future steps captured in α 0 affect the requirements for choosing the parameters glyph[epsilon1] h, 2 and glyph[epsilon1] h, 3 .

To prove Lemma 2, we first provide another new lemma below, which proves to be important. We let

<!-- formula-not-decoded -->

denote the maximum difference between the true safety value c h ( s, a, s ′′ ) of the state-action-state triplet ( s, a, s ′′ ) for any next state s ′′ ∈ S h ( s, a ) of the state-action pair ( s, a ) and the true safety value c h ( s, a, s ′ ) of the given state-action-state triplet ( s, a, s ′ ) . Let

<!-- formula-not-decoded -->

denote the maximum difference between the estimated safety value ˜ c k h ( s, a, s ′′ ) of the state-action-state triplet ( s, a, s ′′ ) for any next state s ′′ ∈ S h ( s, a ) of the state-action pair ( s, a ) and the estimated safety value ˜ c k h ( s, a, s ′ ) of the given state-action-state triplet ( s, a, s ′ ) .

Lemma 5. (Relating the true and estimated safety differences) The estimated safety difference ˜ ∆ k h ( s, a, s ′ ) can be upper-bounded by the true safety difference ∆ h ( s, a, s ′ ) as follows:

<!-- formula-not-decoded -->

where ˜ s ′ max is the maximizer of (40) .

Proof. (Proof of Lemma 5) We let s ′ max denote the maximizer of (39). Notice that s ′ max could be different from ˜ s ′ max (the maximizer of (40)). First, the true safety difference is equal to

<!-- formula-not-decoded -->

Next, the estimated safety difference is equal to

<!-- formula-not-decoded -->

Considering the second term, third term, and the last two terms on the right-hand-side of (43) together, we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the inequality is by applying Lemma 4 and the Cauchy-Schwarz inequality to the first term in the third line and the first term in the fourth line in (44) above, and the fact that β ∥ ∥ ψ ( U ⊥ h , φ ( s, a, s ′ )) ∥ ∥ ( Λ k h, 1 ) -1 ≥ 0 . Next, by combining (43) and (44), we have

<!-- formula-not-decoded -->

where the last inequality is because of the definition of the true safety difference ∆ h ( s, a, s ′ ) in (39).

Lemma 5 shows that the estimated safety difference is only larger than the true safety difference by a term, i.e., 2 β ∥ ∥ ψ ( U ⊥ h , φ ( s, a, ˜ s ′ max )) ∥ ∥ ( Λ k h, 1 ) -1 , that decreases to 0 as the number of learning episodes k increases. This is consistent with the intuition that, as k increases, the estimated safety difference ˜ ∆ k h ( s, a, s ′ ) should get closer to the true safety difference ∆ h ( s, a, s ′ ) . Below, based on Lemma 5, we prove Lemma 2.

Proof. (Proof of Lemma 2) Recall that ˜ f h ( s, a 0 | s ∗ h = s ) represents the gap between the decision of the policy π k used by LSVI-NEW and the optimal decision. Thus, now we characterize the relation between the safety values based on the state-action pair ( s, a 0 ) and the optimal state-action pair ( s, a ∗ h ( s )) . First, according to the definition of estimated safety value in (8) and Assumption 2, the estimated safety value of any state-action-state triplet ( s, a 0 , s ′ ( s, a 0 )) induced by the state-action pair ( s, a 0 ) is equal to

<!-- formula-not-decoded -->

where we drop ( s, a 0 ) from s ′ ( s, a 0 ) for simplicity. Since ψ ( U h , φ ( s 0 h , a 0 h , s 0 h +1 )) = φ ( s 0 h , a 0 h , s 0 h +1 ) and ψ ( U ⊥ h , φ ( s 0 h , a 0 h , s 0 h +1 )) = 0 , from (45), we have

<!-- formula-not-decoded -->

Let us focus on the terms in the bracket [ · ] of (46). Notice that, (i) we have

<!-- formula-not-decoded -->

where the inequality is (a) because ( s, a ∗ h ( s )) is safe, and hence c h ( s, a ∗ h ( s ) , s ′ ) ≤ ¯ c for all s ′ ∈ S h ( s, a ∗ h ( s )) ; (b) according to the definition of the true safety difference in (39). (ii) According to Lemma 4, we have

<!-- formula-not-decoded -->

By combining (46), (47) and (48), we have

˜ c k h ( s, a 0 , s ′ ( s, a 0 ))

<!-- formula-not-decoded -->

Next, since the optimal action a ∗ h ( s ) has not been found by the algorithm, there must exist at least one next-state s ′ ∈ S h ( s, a ) , such that the instantaneous hard constraint (1) is violated. Thus, we must have

<!-- formula-not-decoded -->

Combining (50) and Lemma 5, we have that, for all next state s ′ ( s, a ∗ h ( s )) ∈ S h ( s, a ∗ h ( s )) ,

<!-- formula-not-decoded -->

However, as we discussed in our Idea II and Idea III in Section 3, due to possible unsafe transitions and unsafe states in our problem, such a safety value in (51) may not be achieved by the algorithm. This is a critical difference compared with the case without instantaneous constraints or with only unsafe actions. Therefore, in the following, we first quantify the gap between the state-action pair ( s, a ′ 0 ) that achieves the safety value in (51) and the optimal state-action pair ( s, a ∗ h ( s )) . Then, we quantify the smallest gap between the safe state-action pair ( s, a 0 ) and such a possibly unsafe state-action pair ( s, a ′ 0 ) . Specifically, for the state-action pair ( s, a ′ 0 ) that takes the safety value in (51), from (49), we have

<!-- formula-not-decoded -->

Since the right-hand-side of (52) increases with ∆ h ( s, a ∗ h ( s ) , s ′ ( s, a ∗ h ( s ))) , we have

<!-- formula-not-decoded -->

Note that (53) quantifies the gap between the state-action pair ( s, a ′ 0 ) that achieves the safety value in (51) and the optimal state-action pair ( s, a ∗ h ( s )) . Next, we quantify the smallest gap between the safe state-action pair ( s, a 0 ) and such a possibly unsafe state-action pair ( s, a ′ 0 ) . According to (53), there must exists a safe action a ′ h ′ , 0 for only step h ′ , s.t.,

<!-- formula-not-decoded -->

Then, let ˆ f h ( s, a ) glyph[defines] f h ( φ ( s, a, · ) -φ ( s 0 h , a 0 h , s 0 h +1 )) denote the normalized L 2 -distance between the features of the transitions associated with the state-action pair ( s, a ) and the known safe feature φ ( s 0 h , a 0 h , s 0 h +1 ) . According to (54) and (19), there must exists an action a 0 that induces at least one safe subsubgraph G k, safe h ( s, a 0 ) , s.t.,

<!-- formula-not-decoded -->

Finally, by combining (53) and (55), we have that the left-hand-side of (28) can be upper-bounded as follows:

<!-- formula-not-decoded -->

## C Proof of Lemma 3

As we mentioned in Section 4.2, compared with Lemma 2, the main difference in Lemma 3 is that Lemma 3 quantifies the impacts from past steps, i.e., h ′ ≤ h . This special new impact results in the bonus term with parameter glyph[epsilon1] 4 in our Idea IV in Section 3.

Notice that Lemma 3 implies that when k increases, the UCB terms l 3 decreases to be closer to 0 , and thus the right-hand-side (29) get closer to 0 . Then, ˜ f h (ˆ s, a 0 ) on the left-hand-side of (29) gets closer to 0 . Notice that ˜ f h (ˆ s, a 0 ) represents the gap between the decision of the policy π k used by LSVI-NEW and the optimal decision. In addition, on the right-hand-side of (29), l 3 characterizes the uncertainty from past steps. Thus, the above implication from Lemma 3 is consistent with the intuition that as more safety values revealed, we should be able to get closer to the optimal action.

In this section, we provide the complete proof for Lemma 3. Please see Appendix D for our discussions and proofs on how this special new impact from past steps results in a new bonus term in our Idea IV in Section 3 and how it affects the requirements for choosing the parameters glyph[epsilon1] 4 .

Proof. According to Lemma 5 and (54), there must exists a safe action a h ′ , 0 at step h ′ ≤ h , s.t.,

<!-- formula-not-decoded -->

Then, according to Assumption 3, there must exists a safe action a 0 at step h , s.t.,

<!-- formula-not-decoded -->

Finally, since ˜ f h (ˆ s, a 0 ) ≤ α s ′ (ˆ s,a 0 ) , we have

<!-- formula-not-decoded -->

## D Proof of Lemma 1

In this section, we provide the proof of Lemma 1. The proof replies on Lemma 2 and Lemma 3. Recall from Section 4.2 that invariant (i) shows that, if the optimal state has been found, the estimated V -value must be higher than the optimal V -value. From a high-level point of view, if the optimal safe action has also been found, invariant (i) trivially holds. If it has not been found, thanks to our new bonus terms that essentially capture the distance between the estimated safe actions and the optimal action, invariant (i) still holds. Moreover, invariant (ii) shows that, if the optimal state has not been found, the V -value of the sub-optimal state in ˜ S k h is still larger than the optimal V -value. This is intuitively because ˜ S k h only contains safe states with transitions close enough (within a small gap captured by the small constant ¯ α 0 ) to the optimal transitions, and the distance is captured by our new bonus terms.

Proof. We prove Lemma 1 by mathematical induction.

- (i) Base case: when h = H +1 , both invariants are trivially true, since V ∗ h ( s ) = V k h ( s ) = 0 .
- (ii) Induction step: we hypothesize that the two invariants are true when h = h 0 . Then, we prove that they are true for h = h 0 -1 .

(ii-a) Step-a: note that invariant (i) trivially holds for h = H since V ∗ h ( s ) = V k h ( s ) = r H ( s ) . Next, we prove invariant (i) for h &lt; H by considering the following two cases, based on whether the optimal action a ∗ h ( s ) has been found in ˜ A k h ( s ) and chosen or not.

(ii-a-1) Case-1: If the optimal action a ∗ h ( s ) has been found in ˜ A k h ( s ) and chosen by π k , i.e., a k h ( s ) = a ∗ h ( s ) , based on Section D.4 in [5], we have

<!-- formula-not-decoded -->

where the inequality is because of the definition of V k h ( s ) in (23) and the induction hypothesis of invariant (i) at step h 0 . Notice that this step is different from the analysis in the case without constraints or with only unsafe actions. Here, the optimal action a ∗ h ( s ) must already be chosen, i.e., it is not enough to simply find that the action is safe. This is because, if the optimal action a ∗ h ( s ) is simply found to be safe while not chosen by the algorithm, a future subsubgraph that is completely different from that of the optimal policy could be visited by π k .

glyph[negationslash]

(ii-a-2) Case-2: If the optimal action a ∗ h ( s ) has not been chosen by π k , i.e., a k h ( s ) = a ∗ h ( s ) , we consider the following two subcases based on whether the optimal action a ∗ h ( s ) has been found in ˜ A k h ( s ) or not.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality is because of the definition of V k h ( s ) in (23) and the induction hypothesis of invariant (ii) at step h 0 . Recal from (11) that Q ∗ h ( s, a ) = r h ( s, a ) + 〈 w ∗ h , φ V ∗ h +1 ( s, a ) 〉 , which depends on the V -value V ∗ h +1 at next step. Thus, we write such a dependency explicitly for Q k h and Q ∗ h in (59).

(ii-a-2-II) Subcase-2-II: If the optimal action a ∗ h ( s ) has not been found in ˜ A k h ( s ) by π k , i.e., a ∗ h ( s ) / ∈ ˜ A k h ( s ) , we consider the following two subsubcases, based on the reason the optimal action a ∗ h ( s ) has not been found in ˜ A k h ( s ) by π k .

(ii-a-2-II-A) Subsubcase-2-II-A: If the optimal action a ∗ h ( s ) has not been found in ˜ A k h ( s ) by π k because condition 1 in (9) is violated, we have

<!-- formula-not-decoded -->

Note that V k h ( s ) = max a ∈ ˜ A k h ( s ) Q k h ( s, a ) ≥ Q k h ( s, a 0 ) and the bonus term glyph[epsilon1] 4 · max s ′ ∈S 1 ( s 1 ,a k 1 ) ‖ ψ ( U ⊥ 1 , φ ( s 1 , a k 1 , s ′ )) ‖ ( Λ k 1 , 1 ) -1 in (13) is non-negative, we have

<!-- formula-not-decoded -->

Then, according to Section D.4 in [5], we have

<!-- formula-not-decoded -->

+

glyph[epsilon1]

h,

2

·

max

∈S

h

max

′

h

h

h

0

′

s

(

s,a

‖

)

(

s

,a

,s

)

∈G

′

′

1

k

h,

Moreover, according to Lemma 2, there must exists an action a 0 ∈ ˜ A k h ( s ) , s.t.,

<!-- formula-not-decoded -->

ψ

(

U

⊥

h

,

(

φ

s, a

0

′

, s

‖

))

-

1

+

glyph[epsilon1]

h,

3

·

‖

)

ψ

(

(

Λ

)

(

s

U

⊥

h

′

,

(

φ

s

′

h

, a

′

h

′

, s

‖

))

(

k

h

Λ

′

,

1

-

1

, H

)

(60)

}

.

A Near-Optimal Algorithm for Safe Reinforcement Learning Under Instantaneous Hard Constraints

By combining (60) and (61), and according to Assumption 3 and invariant (ii) at the next step h 0 , we have

<!-- formula-not-decoded -->

Since glyph[epsilon1] 1 is set to be equal to β +1 , we have glyph[epsilon1] 1 -1 ≥ 0 . Thus, ( glyph[epsilon1] 1 -1) · ‖ φ V ∗ h +1 ( s, a ∗ h ( s )) ‖ ( Λ k h, 2 ) -1 ≥ 0 . Thus, we have

<!-- formula-not-decoded -->

Thus, to prove that V k h ( s ) ≥ V ∗ h ( s ) , we need to prove that

<!-- formula-not-decoded -->

By rearranging the terms in (63), we have

<!-- formula-not-decoded -->

A Near-Optimal Algorithm for Safe Reinforcement Learning Under Instantaneous Hard Constraints

Since Q ∗ h ( s, a ∗ h ( s )) ≤ H for all states s and steps h , we have

<!-- formula-not-decoded -->

Note that (64) indicates that, to have V k h ( s ) ≥ V ∗ h ( s ) , we need

<!-- formula-not-decoded -->

This is reason we set the parameters glyph[epsilon1] h, 2 and glyph[epsilon1] h, 3 in our Idea II and Idea III to be in the form in (14) and (15), respectively.

(ii-a-2-II-B) Subsubcase-2-II-B: If the optimal action a ∗ h ( s ) has not been found in ˜ A k h ( s ) by π k because (although condition 1 in (9) is satisfied) condition 2 in (10) is violated, we have

<!-- formula-not-decoded -->

In this subsubcase, we can leverage the knowledge from the satisfied condition 1 to prove V k h ( s ) ≥ V ∗ h ( s ) . The proof then could follow the similar inductions in the proof for subsubcase-2-II-A. For completeness, we provide the proof steps below. First, since the bonus term glyph[epsilon1] 4 · max s ′ ∈S 1 ( s 1 ,a k 1 ) ‖ ψ ( U ⊥ 1 , φ ( s 1 , a k 1 , s ′ )) ‖ ( Λ k 1 , 1 ) -1 in (13) is non-negative, according to Section D.4 in [5], we have

<!-- formula-not-decoded -->

Next, according to Assumption 3, invariant (ii) at next step h +1 and ( glyph[epsilon1] 1 -1) · ‖ φ V ∗ h +1 ( s, a ∗ h ( s )) ‖ ( Λ k h, 2 ) -1 ≥ 0 , we have

<!-- formula-not-decoded -->

Then, to prove V k h ( s ) ≥ V ∗ h ( s ) , based on (63) and since Q ∗ h ( s ) ≤ H , we have

<!-- formula-not-decoded -->

which provides the same requirements on the parameters glyph[epsilon1] h, 2 and glyph[epsilon1] h, 3 .

(ii-b) Step-b: differently from invariant (i) that trivially holds for h = H , we need to carefully handle the correctness of invariant (ii) at step h = H . Next, we prove invariant (ii) for all steps h ≤ H as follows.

First, since the bonus terms glyph[epsilon1] h, 2 · max s ′ ∈S h ( s,a ) ‖ ψ ( U ⊥ h , φ ( s, a, s ′ )) ‖ ( Λ k h, 1 ) -1 and glyph[epsilon1] h, 3 · max ( s h ′ ,a h ′ ,s ′ ) ∈G h ( s ) ‖ ψ ( U ⊥ h ′ , φ ( s h ′ , a h ′ , s ′ )) ‖ ( Λ k h ′ , 1 ) -1 in (13) are non-negative, to prove V k h (ˆ s ) ≥ V ∗ h ( s ) , we need to prove that

<!-- formula-not-decoded -->

for some ˆ a ∈ ˜ A k h (ˆ s ) . To prove (65), we prove

<!-- formula-not-decoded -->

By adding and subtracting r h (ˆ s, ˆ a ) + 〈 w ∗ h , φ V ∗ h +1 (ˆ s, ˆ a ) 〉 , we decompose the left-hand-side of (66) into two parts that are easier for analysis in the following special way,

<!-- formula-not-decoded -->

Notice that by decomposing in this way, the value in the first two brackets [ · ] on the right-hand-side of (67) characterizes how the policy executed by our LSVI-NEW algorithm learns about and searches towards the optimal safe subgraph. The value in the last two brackets [ · ] on the right-hand-side of (67) characterizes how the policy executed by our LSVI-NEW algorithm learns and estimates the optimal Q -value parameter w ∗ h . Next, according to invariant (ii) at next step h 0 , the value in the last two brackets [ · ] on the right-hand-side of (67) can be upper-bounded as follows,

<!-- formula-not-decoded -->

Then, to prove (66), we need to prove

<!-- formula-not-decoded -->

Therefore, below we focus on bounding the value in the first two brackets on the right-hand-side of (67). According to the definition of V k h ( s ) and Lemma 3, there must exist an action ˆ a ∈ ˜ A k h (ˆ s ) and 1 ≤ h ′ ≤ h , s.t.,

<!-- formula-not-decoded -->

Thus, we have

<!-- formula-not-decoded -->

Notice that (69) indicates that the left-hand-side of (68) can be upper-bounded as follows,

<!-- formula-not-decoded -->

where the last inequality is because V ∗ h ( s ) ≤ H for all states s and steps h . (70) indicates that to prove (68), we need

<!-- formula-not-decoded -->

Note that (71) shows that, to prove V k h (ˆ s ) ≥ V ∗ h ( s ) , we need

<!-- formula-not-decoded -->

This is the reason we set the parameter glyph[epsilon1] 4 in our Idea IV to be in the form in (16).

## E Proof of Theorem 2

As we mentioned in Section 4.2, because of the new challenges from the instantaneous hard constraint (1) and our novel ideas in the algorithm design, there are several new difficulties in the regret analysis, which is shown in this section. The key ones are: (I) Differently from the unconstrained setting or the setting with only unsafe actions, in our case, the states that could be visited with non-zero probability by different policies could be completely different at each step h . Hence, the commonly-used invariant on V -values, i.e., V k h ( s ) ≥ V ∗ h ( s ) for all h and s , that relies on the ergodicity property no longer holds in our case. This difficulty is resolved by Lemma 1. (II) How to quantify the impacts when looking ahead and peeking backward. This difficulty is resolved by Lemma 2 and Lemma 3.

Proof. First, for the convenience of the reader, we restate our new construction for the V -values functions of different policies. We let S ∗ h denote the state set at step h in the optimal safe subgraph. Let S k h denote the state set at step h in the subgraph followed by policy π k of LSVI-NEW in episode k . Moreover, we let ˜ f h ( s, a ) glyph[defines] f h ( φ ( s, a, · ) -φ ( s ∗ h , a ∗ h , · )) denote the gap between the transitions associated with the state-action pair ( s, a ) and the optimal transitions. Let ˜ A k h ( s ) glyph[defines] { a ∈ A k, safe h ( s ) : ˜ f h ( s, a ) ≤ ¯ α 0 } ∪ { a k h ( s ) } denote the union of the safe actions with transitions close to the optimal transitions and the action chosen by π k for a safe state s at step h , where

<!-- formula-not-decoded -->

is a small value that decreases to be closer to 0 when the number of learning episodes k increases. Let ˜ S k h glyph[defines] { s ∈ S k, safe h : ∃ a ∈ A k, safe h ( s ) , s.t., ˜ f h ( s, a ) ≤ ¯ α 0 } ∪ S k h denote the union of the safe states with transitions close to the optimal transitions and the state set at step h in the subgraph followed by policy π k of LSVI-NEW in episode k . Next, we define the V -value functions of the optimal policy, estimated policy and policy π k to be

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

respectively. Then, the regret R LSVI-NEW can be decomposed into two parts as follows:

<!-- formula-not-decoded -->

To upper-bound the regret, we prove that, with high probability, (i) the value in the first bracket on the right-hand-side of (25) is non-positive; (ii) the value in the second bracket on the right-hand-side of (75) can be upper-bounded. Note that, according to Lemma 1, we have the value in the first bracket on the right-hand-side of (75) must be non-positive, i.e., V ∗ 1 ( s 1 ) -V k 1 ( s 1 ) ≤ 0 for all episodes k . The value in the second bracket on the right-hand-side of (75) can be upper-bounded by slightly modifying existing techniques for the linear mixture MDP. Specifically, according to the Azuma-Hoeffding inequality, we have

<!-- formula-not-decoded -->

where the last inequality is because of Lemma D.2 in [4] and Lemma 1 in [12].

## F Proof of Theorem 3

In this section, we provide the proof for Theorem 3. The proof is based on the lower bound in the unconstrained horizon-free linear mixture MDP setting [34] and the lower bound in the constrained bandit setting [18]. Note that these existing lower bounds do not show the dependency on the episode length H and the safety parameter ∆ φ ( c ) that are captured in our lower bound.

Proof. Notice that in Theorem 3, we assume K ≥ 32 R . Under this assumption, Lemma 25 in [33] indicates that in the linear bandit problems that are parameterized by the vector µ ∗ = { -√ δ/K 4 √ 2 , √ δ/K 4 √ 2 } d and with the action space A = {-1 , 1 } d and Bernoulli distributed reward r ∼ B ( δ + 〈 µ ∗ , a 〉 ) , where 0 &lt; δ ≤ 1 3 , the regret of any algorithm is lower-bounded by dH √ K 8 √ 2 . Next, consider an instance with three states { s 1 , s 2 , s 3 } , one action a , and the reward r h ( s 1 , a ) = r h ( s 2 , a ) = 0 and r h ( s 3 , a ) = 1 for each h . Then, by using the same transition probability in Section C.3 of [34], we have that the regret of any algorithm for linear mixture MDPs with H steps in each episode is lower-bounded by dH √ K 16 √ 2 . Since the linear mixture MDP with instantaneous hard constraints subsumes (when the cost c h ( s, a, s ′ ) = 0 for all state-action-state triplets) the unconstrained case, dH √ K 16 √ 2 is also a lower bound of the regret in our case.

glyph[negationslash]

Further, to quantify the impact of the safety term ¯ c -¯ c 0 1 -∆ φ ( c ) on the lower bound, in the following, we focus on showing that, when the instantaneous hard constraint with threshold ¯ c is considered, the regret is at least H 24(¯ c -¯ c 0 1 -∆ φ ( c )) 2 . We prove this by contradiction. Assume there exists a safe algorithm that can achieve a regret R 0 &lt; H 24(¯ c -¯ c 0 1 -∆ φ ( c )) 2 for any instance of the problem that we consider. Let us consider the following transition probability function: At step h = 1 , the transition probability is equal to P 1 ( s 2 ( i ) | s 1 , a ( i )) = 1 for all i , and P 1 ( s 2 ( i ) | s 1 , a ( j )) = 0 for all i = j ; at step h &gt; 1 , the transition probability is equal to P h ( s h +1 ( i ) | s h ( i ) , a ( j )) = 1 for all i and j , and P h ( s h +1 ( j ) | s h ( i ) , a ( l )) = 0 for all i = j and all l , where i , j and l are the indices of the states and actions.

glyph[negationslash]

Now, let us consider an instance where the safety value function is as follows: at step h = 1 , the safety value is equal to c 1 ( s 1 , a (1) , s ′ ) = ¯ c 0 1 , c 1 ( s 1 , a (2) , s ′ ) = 2¯ c -¯ c 0 1 , c 1 ( s 1 , a (3) , s ′ ) = ¯ c 0 1 , c 1 ( s 1 , a (4) , s ′ ) = 2¯ c -¯ c 0 1 -∆ φ ( c ) and c 1 ( s 1 , a ( i ) , s ′ ) = 2¯ c -¯ c 0 1 for all i &gt; 4 . Notice that a (1) and a (3) are safe actions, while a (2) , a (4) and other actions are unsafe for state s 1 at step h = 1 . Moreover, at step h &gt; 1 , for all i , the safety value is equal to c h ( s h (1) , a ( i ) , s ′ ) = ¯ c 0 1 , c 1 ( s h (2) , a ( i ) , s ′ ) = 2¯ c -¯ c 0 1 , c 1 ( s h (3) , a ( i ) , s ′ ) = ¯ c 0 1 , c 1 ( s h (4) , a ( i ) , s ′ ) = 2¯ c -¯ c 0 1 -∆ φ ( c ) and c 1 ( s h ( j ) , a ( i ) , s ′ ) = 2¯ c -¯ c 0 1 for all j &gt; 4 . Notice that s h (1) and s h (3) are safe states, while s h (2) , s h (4) and other states are unsafe at each step h &gt; 1 . The reward value function is as follows: at step h = 1 , the reward is equal to r 1 ( s 1 , a (1)) = 1 8 , r 1 ( s 1 , a (2)) = 1 , r 1 ( s 1 , a (3)) = 0 and r 1 ( s 1 , a ( i )) = 1 2 for all i &gt; 3 ; at step h &gt; 1 , for all i , the reward is equal to r h ( s h (1) , a ( i )) = 1 8 , r 1 ( s h (2) , a ( i )) = 1 , r 1 ( s h (3) , a ( i )) = 0 and r 1 ( s h ( j ) , a ( i )) = 1 2 for all j &gt; 3 . Since for any algorithm that chooses action a (1) at step h = 1 less than half of the total episodes with probability p 1 , the regret is at least p 1 HK 2 . Moreover, since the regret of assumed algorithm is R 0 &lt; H 24(¯ c -¯ c 0 1 -∆ φ ( c )) 2 , we have that, for this algorithm,

<!-- formula-not-decoded -->

Next, let us consider another instance where the safety value function is as follows: at step h = 1 , the safety value is equal to c 1 ( s 1 , a (1) , s ′ ) = ¯ c 0 1 , c 1 ( s 1 , a (2) , s ′ ) = 2¯ c -¯ c 0 1 , c 1 ( s 1 , a (3) , s ′ ) = ¯ c 0 1 , c 1 ( s 1 , a (4) , s ′ ) = ¯ c 0 1 + ∆ φ ( c ) and c 1 ( s 1 , a ( i ) , s ′ ) = 2¯ c -¯ c 0 1 for all i &gt; 4 . Notice that a (1) , a (3) and a (4) are safe actions, while a (2) and other actions are unsafe for state s 1 at step h = 1 . Moreover, at step h &gt; 1 , for all i , the safety value is equal to c h ( s h (1) , a ( i ) , s ′ ) = ¯ c 0 1 , c 1 ( s h (2) , a ( i ) , s ′ ) = 2¯ c -¯ c 0 1 , c 1 ( s h (3) , a ( i ) , s ′ ) = ¯ c 0 1 , c 1 ( s h (4) , a ( i ) , s ′ ) = ¯ c 0 1 +∆ φ ( c ) and c 1 ( s h ( j ) , a ( i ) , s ′ ) = 2¯ c -¯ c 0 1 for all j &gt; 4 . Notice that s h (1) , s h (3) and s h (4) are safe states, while s h (2) and other states are unsafe at each step h &gt; 1 . The reward value function is as follows: at step h = 1 , the reward is equal to r 1 ( s 1 , a (1)) = 1 8 , r 1 ( s 1 , a (2)) = 1 , r 1 ( s 1 , a (3)) = 0 and r 1 ( s 1 , a ( i )) = 1 2 for all i &gt; 3 ; at step h &gt; 1 , the reward is equal to r h ( s h (1) , a ( i )) = 1 8 , r 1 ( s h (2) , a ( i )) = 1 , r 1 ( s h (3) , a ( i )) = 0 and r 1 ( s h ( j ) , a ( i )) = 1 2 for all j &gt; 3 . Since for any algorithm that chooses action a (1) at step h = 1 more than half of the total episodes with probability p 2 , the regret is at least 3 p 2 HK 16 . Moreover, since the regret of the assumed algorithm is R 0 &lt; H 24(¯ c -¯ c 0 1 -∆ φ ( c )) 2 , we have, for this algorithm,

<!-- formula-not-decoded -->

Notice that the main difference between this two instances is change of the safety of action a (4) for state s 1 at step h = 1 . Specifically, in instance 1, action a (4) is unsafe, while in instance 2 it becomes safe and incurs the largest reward. Thus, we can quantify the total variation distance between the statistical distributions between these two instances, which can further be upper-bounded by the Kullback-Leibler (KL) divergence. More specifically, according to Lemma 1 in [36] and Lemma 15.1 in [37], we have that this KL divergence is at least q (4) · D KL ( N (2¯ c -¯ c 0 1 -∆ φ ( c ) , I ) ‖N (¯ c 0 1 +∆ φ ( c ) , I ) ) = 2 q (4)(¯ c -¯ c 0 1 -∆ φ ( c )) 2 ≥ 1 2 , where q (4) is the expected A Near-Optimal Algorithm for Safe Reinforcement Learning Under Instantaneous Hard Constraints number of times of choosing action a (4) at step h = 1 in instance 1. Thus, we have

<!-- formula-not-decoded -->

For the algorithm choosing action a (4) for at least q (4) times in average for instance 1, the regret is at least q (4) · 1 2 · 1 3 = 1 6 q (4) . This contradicts with our assumption that the regret of this algorithm is R 0 &lt; H 24(¯ c -¯ c 0 1 -∆ φ ( c )) 2 .