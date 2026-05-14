## Online Apprenticeship Learning

Lior Shani 1 , Tom Zahavy 2 , Shie Mannor 1,3

1

Technion - Israel Institute of Technology, Israel 2 Deepmind, UK

3 Nvidia Research, Israel

shanlior@gmail.com, tomzahavy@gmail.com, shie@ee.technion.ac.il

## Abstract

In Apprenticeship Learning (AL), we are given a Markov Decision Process (MDP) without access to the cost function. Instead, we observe trajectories sampled by an expert that acts according to some policy. The goal is to find a policy that matches the expert's performance on some predefined set of cost functions. We introduce an online variant of AL (Online Apprenticeship Learning; OAL), where the agent is expected to perform comparably to the expert while interacting with the environment. We show that the OAL problem can be effectively solved by combining two mirror descent based no-regret algorithms: one for policy optimization and another for learning the worst case cost. By employing optimistic exploration, we derive a convergent algorithm with O ( √ K ) regret, where K is the number of interactions with the MDP, and an additional linear error term that depends on the amount of expert trajectories available. Importantly, our algorithm avoids the need to solve an MDP at each iteration, making it more practical compared to prior AL methods. Finally, we implement a deep variant of our algorithm which shares some similarities to GAIL (Ho and Ermon 2016), but where the discriminator is replaced with the costs learned by the OAL problem. Our simulations suggest that OAL performs well in high dimensional control problems.

## 1 Introduction

In Reinforcement Learning (Sutton and Barto 2018, RL) an agent interacts with an environment by following a policy. The environment is modeled as a Markov Decision Process (Puterman 1994, MDP), where in each state, the agent takes an action based on the policy, and as a result, pays a cost and transitions to a new state. The goal of RL is to learn an optimal policy that minimizes the long term cumulative cost. This makes RL useful when we can specify the MDP model appropriately. However, in many real-world problems, it is often hard to define a cost which induces the desired behaviour. E.g., an autonomous driver might suffer costs when driving slowly or in a hazardous way. Yet, prescribing these costs can be eluding.

A feasible solution to this problem is Imitation Learning (IL). This setup introduces the notion of an expert, typically a human, that provides us with a set of demonstrations. The agent's goal is to learn the optimal policy by imitating the expert's decisions. Methods such as Behavioural Cloning (BC) try to directly mimic the demonstrator by applying a supervised learning (SL) algorithm to learn a mapping from the states to actions. This literature is too vast to cover here and we refer the reader to (Schaal 1997; Argall et al. 2009).

Copyright © 2022, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

Apprenticeship Learning (Abbeel and Ng 2004, AL) aims to address the same motivation using a different goal. Rather than learning a cost, its goal is to find a policy whose performance is close to that of the expert for any possible cost in a known set. This keeps the state-action occupancy of the agent and expert in proximity, requiring the AL agent to find a path back to the expert trajectories in states that are unobserved by the expert. This differs from BC, in which the agent's policy is undetermined in these unobserved states. Prior works on AL (Abbeel and Ng 2004; Syed and Schapire 2008; Zahavy et al. 2020) mostly considered a batch RL setting with the purpose of finding an glyph[epsilon1] -optimal solution, where the transition model is typically known or can be extracted from the data. However, in many real world applications, the model is unknown, and the learner is inflicted costs when performing poorly on the task, even if these costs are not properly specified to serve as an objective. This leads us to consider an online version of AL in which an agent should perform as close as possible to the expert on any possible cost, while it is learning . As a result, an online autonomous driver would try to imitate the expert when learning in the real-world , avoiding unnecessary costs.

AL is typically formulated as a min-max game between a policy and a cost 'players'. This problem was shown to be convex in the cost and in the feature expectations of the policy, but not in the policy itself. Existing AL algorithms proposed to bypass this issue via the concept of best response. Syed, Bowling, and Schapire (2008) proposed the projection algorithm, in which the policy player applies the best response and the cost player uses Mirror Descent. Alternatively, in MWAL (Abbeel and Ng 2004), the cost player plays the best response and the policy player plays a Frank-Wolfe step (Zahavy et al. 2020) by utilizing the convexity in the feature expectations. Unfortunately, this requires both algorithms to solve an MDP in each iteration (see Section 2.2).

Instead, in the convex games setting, min-max games can be approximately solved by simultaneously running two competing no-regret algorithms, preventing the inefficiency of finding the best response in each iteration (Abernethy and Wang 2017). This result builds on the notion of stability found in online convex optimization algorithms such as Mirror Descent (Beck and Teboulle 2003, MD). Interestingly, there has been a recent body of papers connecting policy optimization techniques and online convex optimization. Specifically, in (Geist, Scherrer, and Pietquin 2019; Shani, Efroni, and Mannor 2020), the authors prove global convergence for an MD-based policy optimization algorithm. Moreover, in (Cai et al. 2019; Efroni et al. 2020), the authors show that using Mirror Descent policy optimization together with optimistic exploration leads to no-regret policy optimization algorithms.

In this work, we take a similar approach and propose an online AL algorithm (OAL, pronounced Owl) that minimizes the AL regret: the difference between the agent's cumulative performance and that of the expert, for any possible cost. Our algorithm performs a dual MD step in which, (1) the policy is improved using one step of optimistic policy optimization MD-based update, and (2) the cost is updated by a single MD iteration. We show this leads to a sample efficient algorithm, even when the model is unknown. Importantly, our algorithm avoids solving an MDP in each iteration, making it more practical than previous approaches. Finally, we conduct an empirical study to verify the need for exploration in OAL.

To illustrate the benefits and practicality of our approach, we implement a deep RL variant of OAL, based on the Mirror Descent Policy Optimization (Tomar et al. 2020, MDPO) algorithm. Our deep OAL variant holds connection to the Generative Adversarial IL algorithm (Ho and Ermon 2016, GAIL): both OAL and GAIL use a generative cost function and take a single policy improvement step in each iteration. Differently from GAIL which learns a probabilistic discrimination between the policy and expert, OAL aims to optimize the value difference between the policy and expert, based on the min-max formulation. This is closely related to GAIL variants based on the Wasserstein distance (Xiao et al. 2019; Chen et al. 2020). Our experiments on continuous control tasks suggest that OAL is comparable to GAIL.

## 2 Preliminaries

In this work, we will deal with finite-horizon MDPs, defined by a tuple M := ( S , A , p, c, H ) , where S , A are the state and action spaces, respectively, and H is the length of an episode. p h ( s ′ | s, a ) is transition kernel describing the probability of transitioning to any state s ′ , given the current state s and action a , for any h ∈ [ H ] . Similarly, c h ( s, a ) is the cost of applying action a at state s , during the h -th time-step. In adversarial MDPs, we allow the costs to change arbitrarily between episodes. A policy π h : S → A is a mapping from state to action. The value function V π,p,c h ( s ) = E [ ∑ H t = h c h ( s, a ) | s h = s, π ] is the cumulative expected costs of the agent, following π from state s at time-step h , over the MDP defined by the transition kernel p and costs c . Similarly, we define the Q -function, Q π,p,c h ( s, a ) = E [ ∑ H t = h c h ( s, a ) | s h = s, a h = a, π ] . The occupancy measure d µ,π,p h ( s, a ) = Pr( s h = s, a h = a | µ, π, p ) is the probability to reach state s and action a , at the h -th timestep, following π and starting from the initial distribution µ . Here throughout, we omit µ and assume without loss of generality that there exists a single starting state. Also, we omit p when clear from context. Notably, it holds for any π , that E s ∼ µ [ V π 1 ( s )] = 〈 c, d π 〉 , where 〈 c, d π 〉 := ∑ H h =1 〈 c h , d π h 〉 = ∑ h,s,a c h ( s, a ) d π h ( s, a ) . A mixed policy ψ over the set of all deterministic policies Π det is executed by randomly selecting the policy π i ∈ Π det at the beginning of an episode with probability ψ ( i ) , and exclusively following π i thereafter. Finally, the filtration F k includes all events in the k -th episode. We omit logarithmic factors when using the O ( · ) notation.

## 2.1 Mirror Descent in RL

The role of conservative updates in the convergence of policy optimization algorithms has been extensively studied in RL, going back to the analysis of the Conservative Policy Iteration (CPI) algorithm (Kakade and Langford 2002). Though sometimes motivated differently, the notion of conservative or stable updates is deeply related to ideas and analyses found in the convex optimization literature. Specifically, CPI can be considered an RL variant of the Frank-Wolfe (FW) algorithm (Scherrer and Geist 2014). Alternatively, the MD algorithm was also studied and applied to MDPs, allowing to provide theoretical guarantees for RL algorithms (Geist, Scherrer, and Pietquin 2019; Shani, Efroni, and Mannor 2020).

MD(Beck and Teboulle 2003) is a framework for solving convex optimization problems. At each iteration, the MD procedure minimizes the sum of a linear approximation of the current objective and a Bregman divergence term, aimed to keep consecutive iterates in proximity. For a set f k of convex losses, and a constraint set C , the k -th MD iterate is x k +1 ∈ arg min x ∈C 〈∇ f k ( x ) | x = x k , x -x k 〉 + t k B ω ( x, x k ) , where B ω is a Bregman divergence and t k is a step size. Finally, MD is known to be a no-regret online optimization algorithm (Hazan 2019). More formally, Reg ( K ) := max x ∑ K k =1 f k ( x k ) -f k ( x ) ≤ O ( √ K ) . √

The stability of the MD updates is crucial to obtain O ( K ) regret in online optimization, where at each iteration the learner encounters an arbitrary loss function (Hazan 2019). This property was also exploited in RL to prove convergence in Adversarial MDPs, where the costs can change arbitrarily between episodes. Indeed, Neu, György, and Szepesvári (2010) provided such guarantees when the transition model is known. Recently, in (Cai et al. 2019; Efroni et al. 2020), the authors provided convergent MD policy optimization algorithms for adversarial MDPs when the model is unknown. These algorithms perform an optimistic policy evaluation step to induce exploration , followed by an MD policy update.

The benefits of MD in RL go beyond establishing convergence guarantees. Shani, Efroni, and Mannor (2020) shows that TRPO (Schulman et al. 2015), a widely used practical deep RL algorithm is actually an adaptation of the MD algorithm to MDPs. As a result, Tomar et al. (2020) derived Mirror Descent Policy Optimization (MDPO), a closer-totheory deep RL algorithm based on the re-interpretation of TRPO, with on-policy and off-policy variants.

## 2.2 Apprenticeship Learning

In AL, we assume the existence of an expert policy , denoted by π E . We assume access to N experts' trajectories sampled from π E over the MDP, from which we construct an estimate of the occupancy measure d E , denoted by ˆ d E . While the cost is unknown in AL, we assume it belongs to some set of costs C . In the theoretical analysis, we focus on the following set:

- C b -Bounded costs: In this tabular case (Eq. (2.1)), the costs are of the form c h ( s, a ) ∈ [0 , 1] , ∀ h, s, a .

In our experiments, we will refer to the following sets:

- C l -Linear costs: The states are assumed to be associated with features φ ( s ) ∈ [ -1 , 1] d , and C l is the costs that are linear in the features: i.e., c ( s ) = w · φ ( s ) . For any w ∈ W , w is usually assumed to be the glyph[lscript] 2 unit ball (Abbeel and Ng 2004) or the simplex (Syed and Schapire 2008). The feature expectations of a policy π are Φ π := E d π φ ( s ) .
- C n -Non-linear costs: In this case the costs are some general non-linear function (typically a DNN) of the state features: c ( s ) = f ( φ ( s )) , where f is bounded. We will also consider the case that f is lipschitz continuous. In this case, AL is related to minimizing the Wasserstein distance between the agent and the expert (Zhang et al. 2020a,b).

The goal of AL is to find a policy π with good performance, relative to the expert, for any possible cost within a set C ,

<!-- formula-not-decoded -->

Previous works mostly focus on the space of mixed policies, and linear costs (see Appendix D for a discussion on differences between the tabular and linear setting). In this case, eq. (2.1) is equivalent to arg min ψ ∈ Ψ max w ∈W 〈 w, Φ ψ 〉 -〈 w, Φ E 〉 . Abbeel and Ng (2004) analyzed this objective when W is the euclidean unit ball. In this setup, it is possible to compute the best response for the cost (the maximizer over W ), exactly, for any ψ , and get that w = Φ ψ -Φ E ‖ Φ ψ -Φ E ‖ . Plugging this back in the objective, we get that solving eq. (2.1) is equivalent to Feature Expectation Matching (FEM), i.e., minimizing ‖ Φ ψ -Φ E ‖ 2 . To solve the FEM objective, the authors propose the projection algorithm. This algorithm starts with an arbitrary policy π 0 and computes its feature expectations Φ π 0 . At step t they fix a cost w t = ¯ Φ t -1 -Φ E and find the policy π t that minimizes it, where ¯ Φ t is a convex combination of the feature expectations of previous (deterministic) policies ¯ Φ t = ∑ t j =1 α j Φ π j . They show that in order to get that ∥ ∥ ¯ Φ T -Φ E ∥ ∥ ≤ glyph[epsilon1] , it suffices to run the algorithm for O ( d glyph[epsilon1] 2 log( d glyph[epsilon1] )) iterations (where d is features dimension).

Another type of algorithms, based on online convex optimization, was proposed by Syed and Schapire (2008). Here, the cost player plays a no-regret algorithm and the policy player plays the best response, i.e., it plays the policy π t that minimizes the cost at time t . The algorithm runs for T steps and returns a mixed policy ψ that assigns probability 1 /T to each policy π t . In (Syed and Schapire 2008), the authors prove their scheme is faster than the projection algorithm (Abbeel and Ng 2004), requiring only O (log( d ) /glyph[epsilon1] 2 ) iterations. This improvement follows from the analysis of MD and specifically the Multiplicative Weights algorithm (Freund and Schapire 1997; Littlestone and Warmuth 1994), giving the algorithm its name, Multiplicative Weights AL (MWAL).

Both types of AL algorithms we have described are based on the concept of solving the min-max game when one of the players plays the best response: the policy player in MWAL, and the cost player in the projection algorithm. The main limitation in implementing these algorithms in practice is that they both require to solve an MDP in each iteration.

## 3 Online Apprenticeship Learning

In this work, we study an online version of AL where an agent interacts with an environment with the goal of imitating an expert. Our focus is on algorithms that are sample efficient in the number of interactions with the environment. This is different from prior batch RL work (Abbeel and Ng 2004; Syed and Schapire 2008; Zahavy et al. 2020) which mostly focused on PAC bounds on the amount of optimization iterations needed to find an glyph[epsilon1] -optimal solution and typically assumed that the environment is known (or that the expert data is sufficient to approximate the model). Our formulation, on the other hand, puts emphasis on the performance of the agent while it is learning , which we believe is important in many real world applications.

Formally, we measure the performance of an online AL algorithm via the regret of the learning algorithm w.r.t the expert. In standard RL, when the costs are known, the regret of a learner is defined as the difference between the expected accumulated values of the learned policies and the value of the optimal policy (Jaksch, Ortner, and Auer 2010). However, in the absence of costs, the optimal policy is not defined. Therefore, it is most natural to compare the performance of the learner to the expert. With Eq. (2.1) in mind, this leads us to introduce the regret in Definition 1, which measures the worst-case difference between the accumulated values of the learner and the expert, over all possible costs in C :

Definition 1 (Apprenticeship Learning Regret) . The regret of an AL algorithm is:

<!-- formula-not-decoded -->

Definition 1 suggests a notion of regret from the perspective of comparison to the expert as a reference policy. Instead, as an optimization problem, the regret of (2.1) is measured w.r.t. to its optimal solution, Reg ( K ) := max c ∈C ∑ K k =1 〈 c, d π k -d E 〉 -min π max c ∈C ∑ K k =1 〈 c, d π -d E 〉 . Importantly, in the following lemma, we show the two regret definitions coincide:

Lemma 1. The online regret of the AL optimization problem (2.1) and the AL regret are equivalent.

## 3.1 Online Apprenticeship Learning Scheme

In Algorithm 1, we present a scheme for solving the AL problem using online optimization tools. Specifically, we introduce a min-player to solve the minimization problem in eq. (2.1). This min-player is an RL agent that aims to find the optimal policy in an adversarial MDP in which a max-player

## Algorithm 1: OAL Scheme

- 1: for k = 1 , ..., K do
- 2: Rollout a trajectory by acting π k
- 3: # Evaluation Step
- 4: Evaluate Q π k using the current cost c k
- 5: Evaluate ∇ c L ( π k , c ; π E ) | c = c k
- 6: # Policy Update
- 7: Update π k +1 by an MD policy update with Q π k
- 8: # Costs Update
- 9: Update c k +1 by an MD step on ∇ c L ( π k , c ) | c = c k

chooses the cost in each round. In Section 3.2, we show that simultaneously optimizing both the policy and cost using no-regret algorithms leads to a sample efficient no-regret AL algorithm (see Definition 1). Our approach averts the need to solve an MDP in each iteration , as was typically done in previous work (see the discussion in Section 2.2), and therefore vastly reduces the computational complexity of the algorithm and makes it more practical.

This is attained in the following manner. Each OAL iteration consists of two phases: (1) evaluation phase , in which the gradients of the objective w.r.t. the policy and costs are estimated, and (2) optimization phase , where both the policy and cost are updated by two separated MD iterates (see Section 2.1).To specify the updates, we need calculate the gradient of the AL objective w.r.t. to the policy or cost, and choose an appropriate Bregman divergence.

Policy update. Because V π E ,c k does not depend on the current policy, the optimization objective is just V π,c k , which is the exactly the RL objective w.r.t. to the current costs. Thus, the gradient of the AL objective w.r.t. policy is the Q -function of the current policy and costs. The KL-divergence is a natural choice for the Bregman term, when optimizing over the set of stochastic policies (Shani, Efroni, and Mannor 2020). Using the stepsize t π k , the OAL policy update is

<!-- formula-not-decoded -->

Notably, this update only requires to evaluate the current Q -function, and does not to solve an MDP.

Cost update. Denoting the cost AL objective L ( π, c ) := -( V π,c 1 ( s 1 ) -V π E ,c 1 ( s 1 )) , the gradient w.r.t. the cost is ∇ c L ( π, c ) | c = c k . The preferable choice of Bregman depends the cost set C . We use the euclidean norm, but other choices are also possible. With stepsize t c k , the OAL cost update is

<!-- formula-not-decoded -->

In the next section, we use the scheme of Algorithm 1 to develop a no-regret AL algorithm.

## 3.2 Convergent Online Apprenticeship Learning

The updates of OAL in Eqs. (3.2) and (3.3) rely on the exact evaluation of Q π k ,c k and ∇ L ( π, c ; π E ) | c k . However, in most cases, the transition model is unknown, and therefore, assuming access to these quantities is unrealistic. Nevertheless, we now introduce Algorithm 2, an OAL variant which provably minimizes the AL regret (see Definition 1) in the tabular settings, without any restrictive assumptions. Intuitively, an AL agent should try and follow the expert's path. However, due to the environment's randomness and the possible scarcity of expert's demonstrations, it can stray afar from such path. Thus, it is crucial to explore the environment to learn a policy which keeps proximity to that of the expert (see the discussion in Section 4). To this end, Algorithm 2 uses optimistic UCB-bonuses to explore the MDP.

The policy player has access to the costs of all state-action pairs, in each iteration. Thus, from the policy player perspective, it interacts with an adversarial MDP with full information of the costs and unknown transitions. To solve this MDP, at each iteration, Algorithm 2 improves the policy by applying an MD policy update w.r.t. a UCB-based optimistic estimation of the current Q -function (Line 15), relying on the techniques of Cai et al. (2019). The UCB bonus, b k -1 h , added to the costs, accounts for the uncertainty in the estimation of the transitions, driving the policy to explore (Line 8).

The cost player update in the tabular setting is given by ∇ c L ( π k , c ) | c = c k = d E -d π k , which can be evaluated using the learned model. We use costs of the form c ∈ C b , which make the cost optimization in Eq. (3.3) separable at each timestep, state and action. In this case, the euclidean distance is a natural candidate for the choice of Bregman divergence, reducing the MD update to (1) performing a gradient step towards the difference between the expert's and the agent's probability of encountering the specific state-action pair, and (2) projecting the result back to [0 , 1] (Lines 17 and 18). We are now ready to state our main theoretical result:

Theorem 1. The regret of the OAL algorithm (Algorithm 2) satisfies with probability of 1 -δ ,

<!-- formula-not-decoded -->

The regret bound in Theorem 1 consists of two terms. The first term shows an O ( √ K ) rate similar to the optimal regret of solving an MDP. Perhaps surprisingly, the fact that we solve the AL problem for any possible costs does not hurt the sample efficiency of our algorithm. The second term is a statistical error term due to the fact that we only have a limited amount of expert data. This error is independent of the AL algorithm used, and is a reminder of the fact we would like to mimic the expert itself and not the data. In this sense, it is closely related to the generalization bound in (Chen et al. 2020), discussed in Section 4. When expert data is scarce, it could be hard to mimic the true expert's policy, and the linear error dominates the bound. Still, when the amount of experts' trajectories is of the order of the number of environment interactions, N ∝ K , the dominant term becomes O ( √ K ) .

Recall that previous AL results require to solve an MDP in each update, and therefore, their bounds on the amount of iterations refers to the amount of times an MDP is solved . In stark contrast, Algorithm 2 avoids solving an MDP, and the regret bound measures the interactions with the MDP .

In what follows we give some intuition regarding the proof of Theorem 1. The full proof is found in Appendix A. In the proof, we adapt the the analysis for solving repeated games using Online Optimization (Freund and Schapire 1999; Abernethy and Wang 2017) to the min-max AL problem. This allows us to prove the following key inequality (Lemma 2, Appendix A), which bounds the AL regret:

<!-- formula-not-decoded -->

Importantly, this decomposes the regret of the policy and

## Algorithm 2: Online Apprenticeship Learning (OAL)

```
1: for k = 1 , ..., K do 2: Rollout a trajectory by acting π k 3: Estimate ˆ d k using the empirical model ¯ p k -1 4: # Policy Evaluation 5: ∀ s ∈ S , V k H +1 ( s ) ← 0 6: for ∀ h = H,.., 1 , s ∈ S , a ∈ A do 7: Q k h ( s, a ) ← ( c k h -b k -1 h + ¯ p k -1 h V k h +1 )( s, a ) 8: Q k h ( s, a ) ← max { Q k h ( s, a ) , 0 } 9: V k h ( s ) ←〈 Q k h ( s, · ) , π k h ( · | s ) 〉 10: # Update Step 11: for ∀ h, s, a ∈ [ H ] ×S × A do 12: # Policy Update 13: π k +1 h ( a | s ) ∝ π k h ( a | s ) exp ( -t π k Q k h ( s, a ) ) 14: # Costs Update 15: c k +1 h ( s, a ) ← c k h ( s, a ) + t c k ( ˆ d k h -ˆ d E h ) ( s, a ) 16: c k +1 h ( s, a ) ← Clip { c k +1 h ( s, a ) , 0 , 1 } 17: Update counters and empirical model, n k , ¯ p k
```

cost players, enabling the algorithm to perform the updates in (3.2) and (3.3) separately. We address each of the terms: Term (i). The regret of the policy player in the adversarial MDP defined by the known costs { c k } K k =1 . By applying the MD-based policy update (Eq. (3.2)) with an optimistic Q -function in Algorithm 2, we follow the analysis in (Cai et al. 2019; Efroni et al. 2020) to bound this term by O ( √ H 4 S 2 AK ) (Lemma 4, Appendix A).

Term (ii). The regret of the cost player, max c ∑ K k =1 〈 c, d π k ,p -ˆ d E 〉 -∑ K k =1 〈 c k , d π k ,p -ˆ d E 〉 . Following MD updates in (3.3) w.r.t. the estimated occupancy measure of the current policy, this term is bounded by O ( √ H 4 S 2 AK ) (Lemma 5, Appendix A).

Term (iii). This describes the discrepancy between the expert's data and the true expert. It is bounded using Hoeffding's inequality, leading to the linear regret part of Theorem 1, O (√ H 3 SAK 2 /N ) .

## 4 Discussion and Related Work

The role of exploration in AL. A key component in our analysis for proving Theorem 1, is using a UCB cost bonus to induce exploration. But should the agent explore at all, if its goal is to follow an expert? Indeed, with infinite amount of expert trajectories, deriving a stochastic policy ˆ π E using ˆ d E (which is equivalent to BC without function approximation), allows for an exact retrieval of π E , leading to zero regret without any exploration . Also, in a slightly different setting where the model is unknown and the true costs are observed, Abbeel and Ng (2005) provided a polynomial PAC sample complexity guarantee for imitating an expert. In contrast to our approach, they argue there is no need to encourage the algorithm to explore. Thus, intuitively, it might seem that the exploration procedure used in OAL wastefully forces the agent to explore unnecessary regions of the state-space.

Figure 1: (a) Exploration and minimizing the AL regret. (b) Comparison between BC and AL with BC initialization.

<!-- image -->

However, when the number of expert trajectories is finite, directly deriving ˆ π from ˆ d π can often lead to states unseen in the data, resulting in undetermined policies in these states which can cause an unwanted behaviour. Instead, the goal of AL is to learn policies which try to stick as close as possible to the experts trajectories , even when unobserved states are encountered. Still, when the expert trajectories are abundant, the model can be accurately estimated in states that the expert can visit, mitigating the need for exploration. Indeed, the guarantee in (Abbeel and Ng 2005) (only) holds in this regime, when the number of trajectories is of the order of O (1 /glyph[epsilon1] 3 ) , where glyph[epsilon1] is the acceptable error.

In stark contrast, our bounds for AL do not rely on any assumption on the number of expert trajectories. As a result, our work suggests that when expert trajectories are scarce, one do need exploration to learn the transition model efficiently , due to the fact that the model cannot be accurately estimated even in states that the expert policy might reach. Note that even in the regime where expert data is abundant, the algorithm in (Abbeel and Ng 2005) requires O (1 /glyph[epsilon1] 5 ) MDP interactions to converge, which is roughly comparable to O ( K 4 / 5 ) regret. In this sense, our algorithm achieves O ( √ K ) regret, and hence requires much less interactions, in the more intricate AL setting and for any amount of trajectories.

To further address this discussion, we empirically tested the necessity of exploration for different amounts of experts' trajectories, by running Algorithm 2 with and without UCB bonuses in a tabular MDP. A fixed amount of episodes was used in all runs. For any number of trajectories, the plot was averaged over 400 seeds, and the error bars represent 95% confidence intervals. The full experimental details are found in Appendix E. The results in Figure 1a show that the AL regret is consistently lower when using optimistic bonuses. This also holds when initializing the transition model in OAL by estimating it using the expert trajectories. This suggests that exploration is crucial to optimize the online performance of OAL. Moreover, Figure 1a shows that when more expert trajectories are available, the regret decreases down to a fixed value corresponding to the second term in Theorem 1.

Finally, Zhang et al. (2020b) elegantly proved PAC convergence for an algorithm that resembles the updates in Eqs. (3.2) and (3.3), using neural networks for approximation. In their work, they assume bounded Radon-Nikodym derivatives, which is similar to having finite concentrability coefficients (Kakade and Langford 2002). This bypasses the need to explore by assuming the agent policies can always reach any state that the expert reaches. By employing proper exploration, we refrained from such an assumption when proving the regret bound in Theorem 1 for the simpler tabular case.

On the generalization of OAL. Chen et al. (2020) analyzed the generalization of an AL-like algorithm in the average cost setting, which is defined as the gap between how the learned policy performs w.r.t. the true expert and its performance w.r.t the expert demonstrations. They show the generalization error depends on O ( √ (log N ) /N ) , where N is the covering number of the cost class. Specifically, when using C b as in Algorithm 2, this becomes O ( √ HSA/N ) , which matches the dependence on SA/N in the linear term of Theorem 1. This result implies that even in the tabular case, different cost classes can lead to improved generalization. However, in cost classes other than C b , the projection required to solve Eq. (3.3) can be much harder. Still, it is valuable to understand how different cost classes affects the performance of OAL when the expert trajectories are limited.

Differences from BC. Instead of minimizing the value difference directly, BC algorithms directly minimize the zeroone or glyph[lscript] 1 loss between the agent and expert policies. A main caveat of this approach is that it fails to treat states unobserved by the expert. In this case, an glyph[epsilon1] error can lead to a value difference of H 2 glyph[epsilon1] (Ross and Bagnell 2010). This can be improved by further assumptions: Ross and Bagnell (2010) assume that one can query the expert online; Brantley, Sun, and Henaff (2019) add an external mechanism to attract the agent towards the expert state-action distribution, and assume the agent is concentrated around the expert. Concretely, Rajaraman et al. (2020) shows that additional knowledge of the transition model is required to avoid this compounding error. Instead, learning a policy close to the state-action distribution of the expert is a core built-in mechanism in AL algorithms, exploiting the transition model to optimize the value-difference directly for any possible cost. This prevents the unwanted behaviour experienced in BC algorithms without relying on any additional assumptions. Another difference between the two approaches, is that analyses of BC algorithms focus on the performance difference between the final learned policy and the optimal policy (Ross and Bagnell 2010). Yet, this does not capture the online performance of the algorithm. Instead, our work uses the more common form of regret which measures the difference between the online performance of the agent and the best policy. Notably, many BC algorithms do interact with the environment online (e.g. (Ross and Bagnell 2010; Brantley, Sun, and Henaff 2019)), and it could be useful to analyze them with a similar criterion.

Finally, note that the two paradigms can be used in a com- plementary fashion. We ran an experiment to compare BC and Algorithm 2 with BC initialization (See Appendix E for details). The results in Figure 1b show that using OAL together with BC leads to an improved online regret. Yet, when expert trajectories are abundant, BC is sufficient to learn the expert policy, as discussed in the beginning of this section.

## 5 Deep Online Apprenticeship Learning

We now present a practical implementation of Algorithm 1 using Deep RL algorithms. We implement two separate modules, a policy and a generative costs module, both of which are updated based on the MD updates in Eqs. (3.2) and (3.3).

For policy optimization, we use MDPO (Tomar et al. 2020), an MD-based off-policy deep RL algorithm. To update the policy, MDPO approximately solves (3.2) by performing several SGD steps w.r.t. its objective, keeping the target policy fixed. This enforces the stability of the policy updates required by Algorithm 1.

For the generative costs, we consider two modules. (1) A linear costs generator based on C l (see Section 2.2). Using this set of costs, Eq. (3.3) can be solved in close-form. (2) A neural network costs generator (see C n in Section 2.2). Note that any cost in this set must be bounded, so it can serve as a cost of an MDP. In theory, this is easily achieved by the projection step in Eq. (3.3), which corresponds to clipping the cost to reside within the set. Yet, when using neural networks, this clipping procedure can hurt the gradient flow. Instead, we use different techniques to keep the cost bounded. First, we penalize the network's output to be close to zero, to effectively limit the size of the costs. Second, we apply the technique proposed in (Gulrajani et al. 2017) to enforce a Lipschitz constraint on the costs. Specifically, we use a convex sum of state-actions pairs encountered by the agent's policy and the expert as an input to the costs network, and penalize the costs updates so that the gradient of the costs w.r.t. to the input would be close to 1 . Third, we perform several gradient steps on Eq. (3.3) to force the updated costs to be close to the old ones, instead of updating the costs using gradient steps w.r.t. to the AL objective (this technique is only applied in the on-policy case discussed in Section 5.1). This prevents the costs from diverging too quickly. Finally, the costs given to the policy player are clipped.

## 5.1 Experiments

(Ho and Ermon 2016) pioneered the idea of solving the AL problem without solving an MDP in each step. To this end, they propose GAIL, an AL algorithm inspired by generative adversarial networks (GAN; Goodfellow et al. 2014). GAIL uses a neural network, which learns to differentiate between the policy and the expert using the GAN loss, as a surrogate for the AL problem. In turn, the GAN loss is given as the cost of the MDP. In this section, we demonstrate that it is possible to directly solve the AL paradigm using either a linear or a NN-based family of costs , by following the OAL scheme.

Experimental Setup. We evaluated deep OAL (Section 5) on the MuJoCo (Todorov, Erez, and Tassa 2012) set of continuous control tasks. To show the online convergence properties of AL algorithms, we present the full learning curves. We used 10 expert trajectories in all our experiments, roughly the average amount in (Ho and Ermon 2016; Kostrikov et al. 2018). We tested OAL with both linear and neural costs (see Section 5), and compared them with GAIL. The same policy and cost networks were used for OAL and GAIL.

Figure 2: OAL vs. GAIL. Policy optimizer used: (Top) off-policy MDPO; (Bottom) on-policy TRPO.

<!-- image -->

Our theoretical analysis dictates to optimize the policy using stable updates. Thus, we used two policy optimization algorithms applying the MD update: (1) on-policy TRPO, which can be seen as a hard-constraint version of MDPO (Shani, Efroni, and Mannor 2020). (2) off-policy MDPO, which directly solves the policy updates in Eq. (3.2).

The experimental results in Figure 2 show that both the linear (green) and neural (orange) versions of OAL are successful at imitating the expert. We turn to analyze the results: OAL vs. GAIL. The results in Figure 2 show both the linear (green) and neural (orange) versions of OAL outperform GAIL (blue), implying it is not necessary to introduce a discriminator in AL. This holds independently on the policy optimization algorithm. Note that the performance drop of GAIL in 'Humanoid' can be explained by the fact that Ho and Ermon (2016) had to increase the amount of MDP interactions and expert trajectories in this environment.

Neural vs. Linear. Surprisingly, the linear version (green) of OAL performs almost as good as the neural one (orange). This comes with the additional benefits that linear rewards are more interpretable, they do not require to design and tune an architecture, and are faster to compute. Our results suggest that linear costs might be sufficient for solving the AL problem even in complex environment, countering the intuition and empirical results found in (Ho and Ermon 2016). There, the authors argue that the main pitfall of AL is its reliance on a predetermined structured cost, which does not necessarily contains the true MDP cost. However, even if the true cost cannot be perfectly represented by a linear function, it might still be sufficient to obtain an optimal policy .

MDPOvs. TRPO. Inspecting Figure 2, one can see that the off-policy MDPO version (top) significantly outperforms the on-policy TRPO (bottom) on all three algorithms. This can be attributed to two reasons: First, in our analysis, the MD policy update in Eq. (3.3) is required for efficiently solving the AL problem. MDPO is explicitly designed to optimize this policy update and therefore closer to theory. Instead, TRPO only implicitly solves this equation. Note that Ho and Ermon (2016) motivated using TRPO in GAIL as preventing noisy Q -function estimates. Our work suggests the need for stable policy updates as an alternative motivation. Second, as was reported in other works (Kostrikov et al. 2018; Blondé and Kalousis 2019), using GAIL together with an off-policy policy algorithm allows a significant boost in data efficiency. Our results strongly imply a similar conclusion.

On Lipschitz Costs. In Figure 3, we study the dependence on the Lipschitz regularization coefficient in the HalfCheetah-v3 domain. Our results implies that restricting the cost to be Lipschitz is important for OAL. Interestingly, in (Kostrikov et al. 2018; Blondé and Kalousis 2019) the authors apply the same gradient regularization technique for GAIL, even though this Lipschitz property is not necessarily required in GAIL. Indeed, Figure 3 suggests that enforcing this regularity condition increases the stability of both GAIL and OAL. However, when used in GAIL, this technique might hurt convergence speed. Interestingly, Xiao et al. (2019) showed that solving the AL problem with Lipschitz costs is similar to GAIL with a Wasserstein distance between the occupancy measures of the agent and expert. They employ several types of regularization techniques to enforce the Lipschitz constraint and optimize the policy using TRPO. Deep OAL is different from their implementation in the following ways: (1) they focus only on on-policy scenario using TRPO; (2) they use L 2 -regularization to decrease the costs when it is not Lipschitz, while we regularize the cost network gradients as proposed in (Gulrajani et al. 2017).

Figure 3: The effect of the Lipschitz regularization.

<!-- image -->

## References

Abbeel, P.; and Ng, A. Y. 2004. Apprenticeship learning via inverse reinforcement learning. In Proceedings of the twenty-first international conference on Machine learning , 1. ACM.

Abbeel, P.; and Ng, A. Y. 2005. Exploration and apprenticeship learning in reinforcement learning. In Proceedings of the 22nd international conference on Machine learning , 1-8.

Abernethy, J. D.; and Wang, J.-K. 2017. On Frank-Wolfe and Equilibrium Computation. In NIPS , 6584-6593.

Argall, B. D.; Chernova, S.; Veloso, M.; and Browning, B. 2009. A survey of robot learning from demonstration. Robotics and autonomous systems , 57(5): 469-483.

Beck, A.; and Teboulle, M. 2003. Mirror descent and nonlinear projected subgradient methods for convex optimization. Operations Research Letters , 31: 167-175.

Blondé, L.; and Kalousis, A. 2019. Sample-efficient imitation learning via generative adversarial nets. In The 22nd International Conference on Artificial Intelligence and Statistics , 3138-3148. PMLR.

Brantley, K.; Sun, W.; and Henaff, M. 2019. Disagreementregularized imitation learning. In International Conference on Learning Representations .

Cai, Q.; Yang, Z.; Jin, C.; and Wang, Z. 2019. Provably Efficient Exploration in Policy Optimization. arXiv preprint arXiv:1912.05830 .

Chen, M.; Wang, Y.; Liu, T.; Yang, Z. Y.; Li, X.; Wang, Z.; and Zhao, T. 2020. On Computation and Generalization of Generative Adversarial Imitation Learning. In International Conference on Learning Representations .

Dann, C.; Lattimore, T.; and Brunskill, E. 2017. Unifying PAC and regret: Uniform PAC bounds for episodic reinforcement learning. In Advances in Neural Information Processing Systems , 5713-5723.

Dhariwal, P.; Hesse, C.; Klimov, O.; Nichol, A.; Plappert, M.; Radford, A.; Schulman, J.; Sidor, S.; Wu, Y.; and Zhokhov, P. 2017. OpenAI Baselines. https://github.com/openai/ baselines.

Efroni, Y.; Merlis, N.; Ghavamzadeh, M.; and Mannor, S. 2019. Tight regret bounds for model-based reinforcement learning with greedy policies. In Advances in Neural Information Processing Systems , 12203-12213.

Efroni, Y.; Shani, L.; Rosenberg, A.; and Mannor, S. 2020. Optimistic Policy Optimization with Bandit Feedback. arXiv preprint arXiv:2002.08243 .

Freund, Y.; and Schapire, R. E. 1997. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of computer and system sciences , 55(1): 119139.

Freund, Y.; and Schapire, R. E. 1999. Adaptive game playing using multiplicative weights. Games and Economic Behavior , 29(1-2): 79-103.

Geist, M.; Scherrer, B.; and Pietquin, O. 2019. A Theory of Regularized Markov Decision Processes. In International Conference on Machine Learning , 2160-2169.

Goodfellow, I.; Pouget-Abadie, J.; Mirza, M.; Xu, B.; WardeFarley, D.; Ozair, S.; Courville, A.; and Bengio, Y. 2014. Generative adversarial nets. Advances in neural information processing systems , 27: 2672-2680.

Gulrajani, I.; Ahmed, F.; Arjovsky, M.; Dumoulin, V.; and Courville, A. C. 2017. Improved training of wasserstein gans. In Advances in neural information processing systems , 5767-5777.

Hazan, E. 2019. Introduction to online convex optimization. arXiv preprint arXiv:1909.05207 .

Hill, A.; Raffin, A.; Ernestus, M.; Gleave, A.; Kanervisto, A.; Traore, R.; Dhariwal, P.; Hesse, C.; Klimov, O.; Nichol, A.; Plappert, M.; Radford, A.; Schulman, J.; Sidor, S.; and Wu, Y. 2018. Stable Baselines. https://github.com/hill-a/stablebaselines.

Ho, J.; and Ermon, S. 2016. Generative adversarial imitation learning. In Advances in Neural Information Processing Systems , 4565-4573.

Jaksch, T.; Ortner, R.; and Auer, P. 2010. Near-optimal Regret Bounds for Reinforcement Learning. Journal of Machine Learning Research , 11(4).

Jin, C.; Yang, Z.; Wang, Z.; and Jordan, M. I. 2020. Provably efficient reinforcement learning with linear function approximation. In Conference on Learning Theory , 2137-2143. PMLR.

Kakade, S.; and Langford, J. 2002. Approximately optimal approximate reinforcement learning. In ICML , volume 2, 267-274.

Kostrikov, I.; Agrawal, K. K.; Dwibedi, D.; Levine, S.; and Tompson, J. 2018. Discriminator-actor-critic: Addressing sample inefficiency and reward bias in adversarial imitation learning. arXiv preprint arXiv:1809.02925 .

Littlestone, N.; and Warmuth, M. K. 1994. The weighted majority algorithm. Information and computation , 108(2): 212-261.

Neu, G.; György, A.; and Szepesvári, C. 2010. The Online Loop-free Stochastic Shortest-Path Problem. In COLT , volume 2010, 231-243. Citeseer.

Orabona, F. 2019. A Modern Introduction to Online Learning. arXiv preprint arXiv:1912.13213 .

Osband, I.; Blundell, C.; Pritzel, A.; and Van Roy, B. 2016. Deep exploration via bootstrapped DQN. arXiv preprint arXiv:1602.04621 .

Puterman, M. L. 1994. Markov decision processes: discrete stochastic dynamic programming . John Wiley &amp; Sons.

Rajaraman, N.; Yang, L.; Jiao, J.; and Ramchandran, K. 2020. Toward the Fundamental Limits of Imitation Learning. Advances in Neural Information Processing Systems , 33.

Ross, S.; and Bagnell, D. 2010. Efficient reductions for imitation learning. In Proceedings of the thirteenth international conference on artificial intelligence and statistics , 661-668.

Schaal, S. 1997. Learning from demonstration. In Advances in neural information processing systems , 1040-1046.

Scherrer, B.; and Geist, M. 2014. Local policy search in a convex space and conservative policy iteration as boosted policy search. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases , 35-50. Springer.

Schulman, J.; Levine, S.; Abbeel, P.; Jordan, M.; and Moritz, P. 2015. Trust region policy optimization. In International conference on machine learning , 1889-1897.

Shani, L.; Efroni, Y.; and Mannor, S. 2020. Adaptive trust region policy optimization: Global convergence and faster rates for regularized mdps. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 34, 5668-5675.

Sutton, R. S.; and Barto, A. G. 2018. Reinforcement learning: An introduction . MIT press.

Syed, U.; Bowling, M.; and Schapire, R. E. 2008. Apprenticeship learning using linear programming. In Proceedings of the 25th international conference on Machine learning , 1032-1039. ACM.

Syed, U.; and Schapire, R. E. 2008. A game-theoretic approach to apprenticeship learning. In Advances in neural information processing systems , 1449-1456.

Todorov, E.; Erez, T.; and Tassa, Y . 2012. MuJoCo: A physics engine for model-based control. In IEEE International Conference on Intelligent Robots and Systems , 5026-5033.

Tomar, M.; Shani, L.; Efroni, Y.; and Ghavamzadeh, M. 2020. Mirror Descent Policy Optimization. arXiv preprint arXiv:2005.09814 .

Weissman, T.; Ordentlich, E.; Seroussi, G.; Verdu, S.; and Weinberger, M. J. 2003. Inequalities for the L1 deviation of the empirical distribution. Hewlett-Packard Labs, Tech. Rep .

Xiao, H.; Herman, M.; Wagner, J.; Ziesche, S.; Etesami, J.; and Linh, T. H. 2019. Wasserstein adversarial imitation learning. arXiv preprint arXiv:1906.08113 .

Zahavy, T.; Cohen, A.; Kaplan, H.; and Mansour, Y. 2020. Apprenticeship learning via frank-wolfe. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 34, 6720-6728.

Zanette, A.; and Brunskill, E. 2019. Tighter ProblemDependent Regret Bounds in Reinforcement Learning without Domain Knowledge using Value Function Bounds. In International Conference on Machine Learning , 7304-7312.

Zhang, M.; Wang, Y.; Ma, X.; Xia, L.; Yang, J.; Li, Z.; and Li, X. 2020a. Wasserstein Distance guided Adversarial Imitation Learning with Reward Shape Exploration. arXiv preprint arXiv:2006.03503 .

Zhang, Y.; Cai, Q.; Yang, Z.; and Wang, Z. 2020b. Generative adversarial imitation learning with neural networks: Global optimality and convergence rate. arXiv preprint arXiv:2003.03709 .

## A Analysis

In this section, we will prove the theoretical claims which are found in this paper. Specifically, in Appendix A.1, we prove Lemma 1 and show the equivalence between the AL regret (see Eq. (3.1)) and the regret of the AL optimization problem (see Eq. (2.1)). Then, in Appendix A.2, we provide a full proof for Theorem 1.

## A.1 Regret Equivalence

Lemma 1. The online regret of the AL optimization problem (2.1) and the AL regret are equivalent.

Proof.

<!-- formula-not-decoded -->

where in the first transition we used the value function notation (see definition in Section 2), and in the last transition we used the fact that for any π , max c ∈C V π,c 1 -V π E ,c 1 is non-negative, and therefore the minimizer is π = π E , for which the solution to the min-max problem is zero.

## A.2 Regret of Online Apprenticeship Learning

In what follows, we prove Theorem 1. In our proof, we deal with all probabilistic events by conditioning our analysis on the occurrence of a 'good event'. In Appendix B, we define this event and bound the probability that its does not occur. For clarity and readability, we now describe the main steps of the proof before diving into the details:

Proof Sketch of Theorem 1. Our analysis relies on the following three stages:

1. We prove Lemma 2, which bounds the AL regret by three independent terms: the regret of the policy player, the regret of the cost player and a statistical error term due to the finite nature of the expert samples. The fundamental relation between the the AL regret and the separate regrets of the policy and cost player is developed in Lemma 3. This key lemma deals with bounding the difference between the values of the sequence of learned policies and the estimated value of the expert for all possible costs. The proof of Lemma 3 is adapted to the AL setting from the work of (Abernethy and Wang 2017), which deals with convex-concave zero-sum games.
2. We bound each of the three terms in Lemma 2 (see also Section 3.2):
- Policy player regret. First, we observe that the policy player is interacting with an adversarial MDP with full information of the costs and unknown transition model. Then, by the fact we use the optimistic MD policy optimization procedure in Algorithm 2, we can apply the results in (Cai et al. 2019; Efroni et al. 2020) to bound the regret of the policy player (see Lemma 4).
- Cost player regret. We show that the regret of the cost player can be separated to independent MD procedures for each time-step, state and action. Then, we bound each of the problems using online MD (see Lemma 5).
- Statistical error. The statistical error term is bounded using Hoeffding's inequality (see event F V in Appendix B).
3. We prove Theorem 1 by plugging in the above bounds in Lemma 2.

We are now ready to prove Theorem 1. First, we prove the following key lemma, which decompose the AL regret as described in Section 3.2. This lemma is heavily based on Lemma 3, which connects the AL regret to the separate regrets suffered by the policy and cost players.

Lemma 2 (AL Regret Decomposition) .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The fourth transition is by the fact max x 〈 x, a + b 〉 ≤ max x 〈 x, a 〉 +max x 〈 x, b 〉 .

Plugging in Lemma 3 to bound max c ∈C ∑ K k =1 〈 c, d π k -ˆ d E 〉 , we obtain

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We now prove the following key lemma, which is essential to the proof of Lemma 2: Lemma 3 (Online Min-Max Regret Bound) . The following holds:

<!-- formula-not-decoded -->

Proof. For brevity, we will denote R π K := Reg π ( K ) and R c K := Reg c ( K ) . By the policy optimization procedure (Lemma 4), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Similarly, from the costs optimization procedure (Lemma 5), we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. By (3.1),

Dividing by K , we get

Dividing by K , we get By combining equations (A.1) and (A.2),

<!-- formula-not-decoded -->

Rearranging we get

<!-- formula-not-decoded -->

Multiplying by K , we have

<!-- formula-not-decoded -->

where in the fifth transition we used the fact that due to the convexity of the set C , 1 K ∑ K k =1 c k is always within the set. Finally, we have that

<!-- formula-not-decoded -->

which concludes the proof.

Before we prove the main theorem for bounding the regret of OAL, we present two useful lemmas, which are proven in Sections A.3 and A.4, respectively:

Lemma 4 (Regret of the Policy Player) . Let t π k = √ 2 log A/ ( H 2 K ) . Then, conditioned on the good event, the regret of the policy player is bounded by

<!-- formula-not-decoded -->

Lemma 5 (Regret of the Costs Player) . Conditioned on the good event,

<!-- formula-not-decoded -->

Finally, we are ready to prove Theorem 1, which bounds the regret of the OAL algorithm (Algorithm 2):

Theorem 1. The regret of the OAL algorithm (Algorithm 2) satisfies with probability of 1 -δ ,

<!-- formula-not-decoded -->

Proof. By Lemma 2,

<!-- formula-not-decoded -->

where we bounded the policy player and cost player regret using Lemmas 4 and 5, respectively. Finally, the last term is bounded conditioned on the good event.

## A.3 Policy Optimization

We now turn to bound the reward of the policy player in Lemma 2. As discussed in Section 3.2, in each iteration, the policy player is allowed to observe the cost function for all time-steps, states and actions. In other words, in the perspective of the policy player, it interacts with an adversarial MDP with full information, described by the sequence of costs { c k } K k =1 . In order to solve this MDP efficiently, we need to address the fact that the transition model is unknown. This is achieved by adding to the costs a UCB-bonus which accounts for the uncertainty in the transition model, as done in Line 7 of Algorithm 2. Specifically, the bonus b k h ( s, a ) satisfies

<!-- formula-not-decoded -->

where a ∨ b := max { a, b } , and n k h ( s, a ) is number of times the agent has visited the state-action pair ( s, a ) at the h -th time-step, until the end of the k -th episode. Note that n k h ( s, a ) is F k -measurable. This term is needed in order to have an optimistic estimation of the bellman error of the current policy (see Efroni et al. 2020, Lemma 5).

Overall, our MD policy optimization procedure in Algorithm 2 exactly matches the OPPO algorithm presented in (Cai et al. 2019). In our work, we apply this algorithm in the more specific tabular case. Thus, for readability, it is more convenient to follow the analysis and notation of the stochastic version of the POMD algorithm in (Efroni et al. 2020), yet with full information of the cost. To this end, in (Cai et al. 2019, Theorem 3.1) and (Efroni et al. 2020, Theorem 1), the authors prove the following regret bound for the policy optimization procedure in Algorithm 2,

<!-- formula-not-decoded -->

The above bound holds conditioned on the good event in Appendix B (see Efroni et al. 2020, Appendix B.1. Note that event F c is not required when we have full information of the cost). Rewriting the value functions in linear form, V π,p,c 1 = 〈 c, d π,p 〉 , the above bound translates to the following lemma:

Lemma 4 (Regret of the Policy Player) . Let t π k = √ 2 log A/ ( H 2 K ) . Then, conditioned on the good event, the regret of the policy player is bounded by

<!-- formula-not-decoded -->

## A.4 Costs optimization

In this section, we deal with bounding the regret of the cost player in Lemma 2. The cost update in Eq. (3.3) requires to estimate the occupancy measures of the current policy for each state-action pairs. This can be done by forward recursion using the empirical model and the current policy. Denote the occupancy measure of d π k ,p using the empirical ¯ p as d π k , ¯ p . Thus, the gradient of the cost optimization problem at the k -th iteration is

<!-- formula-not-decoded -->

Thus, the MD iterate in Eq. (3.3) becomes

<!-- formula-not-decoded -->

In convex optimization, the choice of Bregman divergence usually corresponds to the constraints set used. For example, in the policy optimization step, we optimized over the set of stochastic policies, and therefore used the state-wise KL divergence. However, choosing the cost set in AL is a degree of freedom, and therefore different Bregman should be chosen when different cost sets are used. For example, when considering a linear cost, C l , past works have considered different sets which led to different Bregman terms: In the projection algorithm (Abbeel and Ng 2004), the authors constrain the weights to the unit L 2 -ball, and therefore use the euclidean distance as the Bregman term. Instead, in MWAL (Syed and Schapire 2008), the authors constrain the weights to the unit simplex, and thus use the KL divergence leading to exponential updates. Although the above choices are also legitimate in the tabular case, we have chosen to focus on the unit box, C b , as the set of costs. This set is the most general bounded cost set in the tabular case, and it is typically assumed when discussing tabular MDPs. Interestingly, different choices can have an effect on the regret bounds of Lemma 2. Specifically, it can change the dependence on H,S,A of both the cost player regret and the second term of Theorem 1, which depends on the covering number of the cost set (see Section 4). Importantly, the set C b is state-wise independent which makes the optimization process separable and therefore simpler. While choosing sets which enforce a global constraint on the costs is also possible, this will lead to a more complicated and less practical projection step in the solution of Eq. (A.3).

Following the above discussion, when optimizing over the unit box, where for any h, s, a , c h ( s, a ) ∈ [0 , 1] , it is most natural to use the Euclidean distance as the Bregman divergence. This leads to the following optimization problem for any h, s, a ,

<!-- formula-not-decoded -->

Then, (A.4) have the following closed form projected gradient descent update,

<!-- formula-not-decoded -->

where Concat { x, a, b } concatenates x within [ a, b ] . Note that this simple projection step follows from the choices of the cost set and Bregman divergence. Importantly, Eq. (A.5) is exactly the cost updates in lines 17,18 of Algorithm 2.

We are now ready to prove the regret guarantee for the policy player:

Lemma 5 (Regret of the Costs Player) . Conditioned on the good event,

<!-- formula-not-decoded -->

Proof. Observe that the regret for the cost optimization procedure can be decoupled in the following manner:

<!-- formula-not-decoded -->

Note that the RHS in Eq. (A.6) has two additive terms. The first term is due to the statistical error in the empirical model, and will be bounded by conditioning on the good event (see Appendix B). The second term will be bounded by the OMD analysis.

Term (i). For any c ,

<!-- formula-not-decoded -->

In the first transition, we used the value difference lemma (Corollary 1). The second transition is by the Cauchy-Schwartz inequality. The third is by the fact that by c -c k ∈ [ -1 , 1] and therefore, for any k, h , ∥ ∥ ∥ v π k , ¯ p k ,c -c k h +1 ∥ ∥ ∥ ∞ ≤ H . The fourth transition is holds by the good event for some positive constant C (see Appendix B). In the sixth relation we used the fact that the expectations are equivalent, since at the the policy π k is fully determined by the events in the filtration F k -1 . Finally, the last transition is by applying Lemma 10 and excluding constant factors which do not depend on K .

By the fact the above inequality holds for any c , we get

<!-- formula-not-decoded -->

Term (ii). It holds that

<!-- formula-not-decoded -->

The third transition is by the fact that the optimization problem can be decoupled coordinate-wise due to the structure of C b . The fourth transition is by the OMD analysis (see Corollary 2) for any time-step, state and action. The fifth transition is due to the fact that for a, b ∈ [0 , 1] , ( a -b ) 2 ≤ 2 a +2 b . The last transition is due to the fact that for any occupancy measure d and time-step h , ∑ s,a d h ( s, a ) = 1 .

Thus, by choosing t c k = √ SA 2 K

<!-- formula-not-decoded -->

Finally, plugging (A.7) and (A.8) in the the two terms in equation (A.6), we get that conditioned on the good event,

<!-- formula-not-decoded -->

Define the following failure events.

<!-- formula-not-decoded -->

Furthermore, the following relations hold.

- Let F P = ⋃ K k =1 F p k . Then Pr { F p } ≤ δ ′ , holds by (Weissman et al. 2003) while applying union bound on all s, a , and all possible values of n k ( s, a ) and k . Furthermore, for n ( s, a ) = 0 the bound holds trivially.
- Let F N = ⋃ K k =1 F N k . Then, Pr { F N } ≤ δ ′ . The proof is given in (Dann, Lattimore, and Brunskill 2017) Corollary E.4.
- First, note that any c ∈ C b can be written as a convex sum of edges of the unit box over H ×S × A . Thus, in order to bound ∣ ∣ ∣ 〈 c, d E -ˆ d E 〉∣ ∣ ∣ for any c ∈ C b , it suffices to bound this term for all the edges of the 2 SAH unit box and apply the triangle inequality. Now, take a some fixed edge c edge ,

<!-- formula-not-decoded -->

By Hoeffding's inequality, we get that w.p. δ ′ 2 SAH it holds that ∣ ∣ ∣ 〈 c edge , d E -ˆ d E 〉∣ ∣ ∣ ≤ H √ SAH log 4 δ ′ 2 N . Finally, by taking a union bound over all possible edges, we get Pr { F V } ≤ δ ′ .

Lemma 6 (Good Event) . Setting δ ′ = δ 3 then Pr { F p ⋃ F N ⋃ F V } ≤ δ . When the failure events does not hold we say the algorithm is outside the failure event, or inside the good event G .

## B Failure Events

## C.1 Difference Lemmas

The following lemma is taken from (Efroni et al. 2020)[Lemma 1] (originally adapted from the analysis of the first term, in (Cai et al. 2019)[Lemma 4.2]). It extends the value difference lemma (see Corollary 1) which is widely used in the RL literature.

Lemma 7 (Extended Value Difference) . Let π, π ′ be two policies, and M = ( S , A , { p h } H h =1 , { c h } H h =1 ) and M ′ = ( S , A , { p ′ h } H h =1 , { c ′ h } H h =1 ) be two MDPs. Let ˆ Q π, M h ( s, a ) be an approximation of the Q -function of policy π on the MDP M for all h, s, a , and let ˆ V π, M h ( s ) = 〈 ˆ Q π, M h ( s, · ) , π h ( · | s ) 〉 . Then,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where V π ′ , M ′ 1 is the value function of π ′ in the MDP M ′ .

By replacing the approximation in the last lemma with the real expected value, we get the following well known result: Corollary 1 (Value difference) . Let M , M ′ be any H -finite horizon MDP. Then, for any two policies π, π ′ , the following holds

<!-- formula-not-decoded -->

## C.2 Online Mirror Descent

In each iteration of Online Mirror Descent (OMD), the following problem is solved:

<!-- formula-not-decoded -->

The following lemma describes the regret of a general OMD procedure, for a general convex constraint set and Bregman divergence:

Lemma 8 (OMD Regret Bound, Orabona 2019, Theorem 6.8) . Let B ω the Bregman divergence w.r.t. ψ : X → R and assume ω to be λ -strongly convex with respect to ‖·‖ in V . Let V ⊆ X a non-empty closed convex set. Also, assume V ⊆ int X . Set g t ∈ ∂f t ( x t ) . Set x 1 ∈ X such that ψ is differentiable in x 1 . Assume t k = t K , for k = 1 , ..., K . Then, ∀ u ∈ V , the following regret bound holds:

<!-- formula-not-decoded -->

In our analysis (see Lemma 5), we apply the above lemma for the case when the constraint set is x ∈ [0 , 1] and the euclidean distance is chosen as the Bregman divergence. Note that in this case, for any u ∈ X , B ω ( u, x 1 ) ≤ 1 . Overall, this results in the following corollary:

Corollary 2 (OMD Euclidean Regret Bound) . Let ω = ‖·‖ 2 2 such that B ω ( x, y ) = ‖ x -y ‖ 2 2 , and ω is 1 -strongly convex w.r.t. ‖·‖ 2 . Let X = { x | x ∈ [0 , 1] } . Set g t ∈ ∂f t ( x t ) . Set an arbitrary x 1 ∈ X . Assume t k = t K , for k = 1 , ..., K . Then, ∀ u ∈ V , the following regret bound holds:

<!-- formula-not-decoded -->

## C Useful Lemmas

## C.3 Bounds on the Visitation Counts

Lemma 9 (e.g. Zanette and Brunskill 2019, Lemma 13) . Outside the failure event, it holds that

<!-- formula-not-decoded -->

Lemma 10 (e.g. Efroni et al. 2019, Lemma 38) . Outside the failure event, it holds that

<!-- formula-not-decoded -->

In both Zanette and Brunskill 2019; Efroni et al. 2019, these results were derived for MDPs with stationary dynamics. Repeating their analysis, in our case, an additional H factor emerges as we consider MDPs with non-stationary dynamics.

## D A Brief Discussion on Tabular Apprenticeship Learning

Linear cost and known model assumptions. As mentioned before, a typical assumption in AL is that the cost, which is generally a vector in R S , can be represented in a lower dimension R d as a linear combination of the observed features. A second assumption that is typically made in AL, is that the model is either known or given via a perfect simulator. These assumptions allowed the AL literature to present the sample complexity results as a function of the feature dimension d , with no dependence on the size of the state-action space, and without making further assumptions regarding the dynamics (e.g., as in linear MDPs (Jin et al. 2020)). When the model has to be estimated from samples, the agent will have to explore, and therefore the sample complexity results will depend on S unless we further make assumptions regarding the dynamics (e.g., (Abbeel and Ng 2005)).

The AL problem as a regularizer. AL usually follows two assumption to regularize the policy class: (1) the expert is minimizing some cost, and (2) this cost is within a small set. Note here that even without (2) (e.g., in tabular MDPs), (1) is regularizing the policy class. This is because the set of policies that minimize an MDP is typically smaller than the set of all (deterministic) policies.

Figure 4: A finite horizon chain environment. Actions a 0 and a 1 are in blue and red, respectively.

<!-- image -->

## E Experimental Details

## E.1 Tabular OAL with Exploration

In Figure 4, we describe the finite-horizon MDP which is used in our tabular experiments (e.g., Figure 1a). Specifically, we used an horizon of H = 32 . This stochastic chain MDP consists of two possible states, s 0 and s 1 , at any time-step. In each state, the agent faces two choices, a 0 and a 1 . While in s 0 , by performing a 0 , the agent remains in the same state at the next time-step w.p. 1 -α , or otherwise transitions to s 1 ; instead, by performing a 1 the agent deterministically transitions to s 1 . In s 1 , both actions lead to s 1 deterministically.

In reward-based RL, when a big sparse reward is given in s 0 (only) in the last time-step , and a very small reward is given when the agent encounters s 1 in any time-step , this MDP is considered a hard task which requires exploration (see similar example in (Osband et al. 2016)). When the reward in state s 0 at the end of the chain is big enough, the optimal policy is to always choose a 0 . Following this reasoning, in our experiments, we sampled expert trajectories from an expert policy which always chooses action a 0 . Notably, the sparsity of the reward in the optimal policy requires the agent to perform many actions that do not lead to immediate rewards. Thus, the RL agent must explore to prevent the agent from converging to a sub-optimal policy. Differently, in AL, when entire expert trajectories are given to the agent in advance, the agent receives a cost in all state-action pairs which the expert visited. Therefore, even in exploratory environments like the stochastic chain, the costs given by OAL are not necessarily sparse. An instant question is whether the same transition model still requires exploration when solved using the costs of an AL agent.

Indeed, as discussed in Section 4, Figure 1a suggests that exploration is required to attain low regret for this MDP when using OAL. To further understand the effect of the chain MDP structure on the AL problem, we tested OAL with different values of the transition stochasticity parameter, α . Similarly to Figure 1a, we examined how the OAL regret is affected by two factors: using UCB exploration and initializing the learned transition model with the expert trajectories (similarly to (Abbeel and Ng 2005)). The results are reported in Figure 5. We ran all seeds for K = 10000 episodes. We now turn to discuss the results:

On the effect of exploration. Figure 5 shows that for any value of α , using the UCB-bonus exploration improves the overall regret, fortifying the results in Figure 1a.

Initializing the transition model with the experts' trajectories. Interestingly, when α is large, this procedure does not improve the regret by much. This is due to the fact that in this case the model is poorly estimated at s 0 in later time-steps, and there is no escape from exploring the MDP to improve the model estimation. Instead, when α is small, the expert trajectories can provide a good estimate for the transition model in states the expert could visit, leading to a significant boost in performance which is almost equal to the one attained when performing exploration. Still, using this technique is orthogonal to applying exploration, and using the two techniques together greatly improves the performance in this case.

## E.2 Tabular OAL with BC Initialization

In Figure 6, we describe the MDP used in the experiment in Figure 1b. At the beginning of any episode, the agent is randomly spawned at any of the 50 states. The Expert policy is to always reach state s 1 at the last time step. I.e., the expert policy plays a 0 (blue) at any state. We used 1000 seeds for BC and 10 seeds for AL. Finally, the 95% confidence intervals reside within the plotted dots in Figure 1b.

Notably, imitating the expert using BC will lead to a uniformly random policy in starting states that are unobserved in the data. In turn, when the amount of expert trajectory is small (and particularly, when it is smaller than the number of starting states), BC would not be capable to accurately learn the expert policy, and would therefore suffer linear regret in episodes in which the starting state is not observed in the expert data. Instead, the AL paradigm bypasses this issue by learning a policy with state-action occupancy measure which is close to that of the expert. Then, because for any possible trajectory the expert is always at state s 1 when h = 1 , the AL agent will learn to always play a 0 , recovering the expert's behaviour. Still, initializing the AL agent with BC, allows to enjoy the best of both world: 1) the offline BC procedure warm-starts the algorithm, reducing OAL regret by not starting with a totally random policy; 2) then, the AL learning paradigm allows to fully recover the expert policy, further reducing the OAL regret. The results in Figure 1b captures this behaviour: when the amount of expert trajectories is small, the regret of using OAL is much smaller than using only BC. This difference deceases as the number of trajectories gets bigger, and is almost unnoticed when most starting states are observed in the expert data.

Figure 5: The effect of exploration and expert model initialization on the OAL regret for different values of the model stochasticity parameter, α .

<!-- image -->

Figure 6: The MDP used for the comparison between BC and AL in Figure 1b. Actions a 0 and a 1 are in blue and red, respectively.

<!-- image -->

## E.3 Deep OAL

For each experiment, we averaged the results over 5 seeds and plotted 95% confidence intervals. We used 10 expert trajectories in all our experiments, roughly the average amount in (Ho and Ermon 2016; Kostrikov et al. 2018). We verified this choice by training OAL on 'Walker2d' with different number of trajectories, as reported in Figure 7. Similarly to (Ho and Ermon 2016; Kostrikov et al. 2018), our results suggest that OAL performs reasonably well regardless on the amount of expert trajectories. The dashed line in the figures represents the average performance of the expert. We used the Stable-Baselines (Hill et al. 2018) code-base to reproduce GAIL and to implement our algorithms, and did not change the default implementation parameters. The same hyper-parameters were utilized for all domains.

In Tables 1 and 2, we specify the hyper-parameters used in the implementation of our algorithms. All other hyper-parameters are set to their defaults in (Hill et al. 2018). In what follows, we explain the meaning of the main hyper-parameters intoduced in our algorithms. The 'cost update frequency' hyper-parameter determines the amount of interaction with the MDP after which the costs are updated using Eq. (3.3). The 'cost range' hyper-parameter describes how the costs are normalized or concatenated, when given to the policy player. As discussed in Sections 5 and 5.1, in the neural version of OAL, we penalize the costs to be globally Lipschitz. The amount of regularization is defined by the 'Liphscitz coefficient' hyper-parameter. Notably, this coefficient is the one used when describing the effect of Lipschitz regularization in Figure 3. In the on-policy neural version of OAL, we solve Eq. (3.3) by performing several gradient descent objective. To this end, the 'cost Bregman coefficient' hyper-parameter is the coefficient used for the Bregman term in Eq. (3.3), and the 'cost Bregman coefficient' is related to the inverse of the 'cost step size', t c . 'cost MD sgd updates' describes the number of gradient descent iterations. The pseudo-code for deep OAL is described in Algorithm 3. Finally, Kostrikov et al. (2018) showed that running GAIL using TRPO on complex domains such as 'HalfCheetah', requires as many as 25 Million interactions with the environment. Therefore, to save computation when comparing the performance of OAL and GAIL using TRPO as optimizer, we used a pretraining procedure for the more complex environments, 'HalfCheetah' and 'Humanoid', as done in the benchmarks in (Dhariwal et al. 2017). Specifically, in the TRPO (on-policy) versions of GAIL and OAL which use a neural cost network, we pretrained the policy network using 1000 epochs of BC w.r.t. to the expert trajectories. In Figure 8, we compared the performance of GAIL and OAL with and without pretraining. The results show that pretraining improves the convergence on both OAL and GAIL. Still, the results suggest that BC initialization has a slightly greater effect on OAL than on GAIL. Interestingly, the linear version of OAL is still competitive, even though it does not use pretraining. For the full implementation, see the OAL github repository.

Figure 8: Behavioural cloning initialization in on-policy OAL.

<!-- image -->

Figure 7: OAL performance depending the amount of expert trajectories.

<!-- image -->

## Pseudocode

- Algorithm 3: Deep OAL with Off-Policy MDPO 1: Initialize Replay buffer D = ∅ ; Value networks V φ and Q ψ ; Policy networks π new and π old with parameters θ ; Trajectory Replay buffer D π = ∅ ; Cost Network c with parameters ω . 2: for k = 1 , . . . , K do 3: Take action a k ∼ π θ k ( ·| s k ) , observe s k +1 4: Add ( s k , a k , s k +1 ) to the policy replay buffer D ; 5: Add ( s π k , a π k ) to the trajectory replay buffer D π ; 6: # Policy Update 7: Sample a batch { ( s j , a j , s j +1 ) } N j =1 from D 8: Generate c φ k ( s j , a j ) using the cost network 9: # Policy Improvement (Actor Update) 10: Set the new policy θ k +1 by performing m SGD updates on the MDPO objective L π ( θ, θ k ) # see (Tomar et al. 2020) 11: # Policy Evaluation (Critic Update) 12: Update φ and ψ by minimizing the loss functions L V φ = 1 N ∑ N j =1 [ V φ ( s j ) -Q ψ ( s j , π θ k +1 ( s j ) )] 2 ; L Q ψ = 1 N ∑ N j =1 [ c φ k ( s j , a j ) + γV φ ( s j +1 ) -Q ψ ( s j , a j ) ] 2 13: # Cost Update 14: if s k +1 signals the end of trajectory then 15: Sample an expert trajectory D E = { s E j , a E j } 16: Update ω by minimizing the AL objective w.r.t. to the costs, using D π , D E # see Eq. (3.3)/Section 5 17: D c = ∅

## Hyperparameters

Table 1: Hyper-parameters of all on-policy methods.

| Hyperparameter                    | GAIL       | OAL Linear     | OAL Neural   |
|-----------------------------------|------------|----------------|--------------|
| update frequency (T)              | 1024 3072  | 2000 6000      | 2000 6000    |
| cost update frequency (T)         |            |                |              |
| cost Adam step size               | 3 × 10 - 4 | -              | 9 × 10 - 5   |
| cost Bregman coefficient          | -          | -              | 100          |
| cost MDsgd updates                | 1          | 1              | 10           |
| cost range                        | -          | L 2 normalized | [-10,10]     |
| cost hidden layers                | 3          | -              | 3            |
| cost activation                   | tanh       | -              | tanh         |
| cost hidden size                  | 100        | -              | 100          |
| cost step size t c                | -          | 0.05           | -            |
| Lipschits coefficient             | 0          | -              | 1.0          |
| policy stepsize t π               |            | 0.5            |              |
| Adam step size                    | × -        | 3 10 4         |              |
| entropy coefficient               |            | 0.0            |              |
| cost entropy coefficient          |            | 0.001          |              |
| discount factor                   |            | 0.99           |              |
| minibatch size                    |            | 128            |              |
| #runs used for plot averages      | 5          |                |              |
| confidence interval for plot runs | ∼          | 95%            |              |

Table 2: Hyper-parameters of all off-policy methods.

| Hyperparameter                                                                                                                                                                                                                                                                                                                                                                                                                 | GAIL             | OAL                                                                                              | OAL Neural                |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|--------------------------------------------------------------------------------------------------|---------------------------|
| cost range cost hidden layers cost activation cost hidden size cost step size t c Lipschits coefficient entropy coefficient policy stepsize t π cost update frequency (T) cost entropy coefficient minibatch size Adam stepsize replay buffer size target value function smoothing coefficient mdpo update steps number of policy hidden layers discount factor #runs used for plot averages confidence interval for plot runs | - 3 tanh 100 - 0 | Linear L 2 normalized - - - 0.05 - 0.0 0.5 2000 0.001 256 3 × 10 - 4 10 6 0.01 10 2 0.99 5 ∼ 95% | [-10,10] 3 tanh 100 - 1.0 |