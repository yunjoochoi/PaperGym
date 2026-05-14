## TempLe: Learning Template of Transitions for Sample Efficient Multi-task RL

Yanchao Sun, 1 Xiangyu Yin, 2 Furong Huang 1

1 University of Maryland, College Park, MD, 20742

2 Beijing University of Posts and Telecommunications, China ycs@umd.edu, yinxiangyu@bupt.edu.cn, furongh@umd.edu

## Abstract

Transferring knowledge among various environments is important for efficiently learning multiple tasks online. Most existing methods directly use the previously learned models or previously learned optimal policies to learn new tasks. However, these methods may be inefficient when the underlying models or optimal policies are substantially different across tasks. In this paper, we propose Template Learning (TempLe), a PAC-MDP method for multi-task reinforcement learning that could be applied to tasks with varying state/action space without prior knowledge of inter-task mappings. TempLe gains sample efficiency by extracting similarities of the transition dynamics across tasks even when their underlying models or optimal policies have limited commonalities. We present two algorithms for an 'online' and a 'finite-model' setting respectively. We prove that our proposed TempLe algorithms achieve much lower sample complexity than single-task learners or state-of-the-art multi-task methods. We show via systematically designed experiments that our TempLe method universally outperforms the stateof-the-art multi-task methods (PAC-MDP or not) in various settings and regimes.

## 1 Introduction

Multi-task reinforcement learning (MTRL) (Wilson et al. 2007; Brunskill and Li 2013; Modi et al. 2018) requires the agent to efficiently tackle a series of tasks. A key goal of MTRL is to improve per-task learning efficiency compared against single-task learners, by using the knowledge obtained from previous tasks to learn new tasks. Despite the recent rapid progress in MTRL, some issues remain unsettled. (1) Guaranteed sample efficiency. Only a few existing methods have guarantees on sample efficiency, the most common bottleneck of RL algorithms. (2) Correctness v.s. efficiency. An overly aggressive application of previous knowledge may transfer incorrect knowledge and deteriorate the performance on new tasks, resulting in a 'negative transfer' (Taylor and Stone 2009). However, if an agent is overly conservative in applying previously learned knowledge, much of the similarities between tasks will be ignored, resulting in an 'inefficient transfer'. It is nontrivial to balance between the correctness and efficiency or achieve

Copyright © 2021, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

both. (3) Varying state/action space across tasks. In practice, transferring knowledge learned from smaller environments to learning in larger environments is extremely useful. However, most existing works on MTRL assume the state/action space is shared across tasks.

In an effort to provide guaranteed sample efficiency for MTRL, Brunskill and Li (2013) propose an algorithm that clusters the underlying Markov Decision Processes (MDPs) of tasks into groups and identifies new tasks as learned groups. However, transferring knowledge from the clustered MDP models could be an 'inefficient transfer' if the underlying models are too different to be clustered into a small number of groups. Similarly, most existing model-based approaches (Liu, Guo, and Brunskill 2016; Modi et al. 2018) only exploit model-level similarities, which also makes it difficult to transfer knowledge among different-sized tasks.

We remedy the aforementioned three issues by extraction of more commonalities in tasks without suffering from 'negative transfer'. A motivating example is the navigation problem in mazes with slippery floors which result in stochastic transitions. For instance, the agent taking an action of going up on ice could slip to the left , right or down (instead of up ) with a certain probability determined by the slipperiness of ice. The slipperiness of the floor depends on the landform of the location, such as sand, marble and ice. We show some examples of different combinations/distributions of the landforms in the maze in Figure 1; the MDP models are drastically different across different mazes, therefore transferring knowledge using similarity of models is inefficient.

Figure 1: Examples of landform combinations in Maze, where stands for sand, stands for marble and stands for ice. Different landforms have different slippery probability, thus different transition dynamics. Consider a √ S × √ S maze with G types of landforms. There could be up to G S different MDP models, making it prohibitive to extract similarities from the models. However, the types of underlying transition dynamics associated with each state/location are governed by the number of distinct landforms G .

<!-- image -->

However, our key observation is that the same landforms share the transition dynamics, and knowledge could be transferred from sand to sand, marble to marble, and ice to ice. More importantly, we can extend the knowledge learned from a maze to any-sized mazes consisting of these same types of landforms (e.g., the 4th example in Figure 1). With this idea, we achieve more effective and efficient knowledge transfer by exploiting similarities at the level of stateaction transition dynamics instead of MDP model dynamics , allowing knowledge transfer between tasks with varying state/action space without prior knowledge of intertask mappings. The challenge of learning is now reduced to extracting such 'landforms' without prior knowledge of the tasks.

We propose a novel method called Template Learning (TempLe) for MTRL, which provably guarantees sample efficiency and achieves efficient transfer learning for multitask reinforcement learning with varying state/action space. We extract templates for similar state-action transition dynamics (landforms in the example above), called Transition Templates , and confidently improve the efficiency of transition dynamics estimation in new tasks. By sharing experience among state-action pairs associated with similar templates, the learning process is expedited. We introduce two versions of TempLe: one is for online MTRL without prior knowledge about models, named Online Template Learning (O-TempLe) , the other further improves the learning efficiency based on a finite-model assumption, named FiniteModel Template Learning (FM-TempLe) .

Summary of Contributions: (1) TempLe achieves a significant reduction of sample complexity compared with stateof-the-art PAC-MDP (Probably Approximately Correct in Markov Decision Processes) algorithms. (2) TempLe covers two realistic settings, solving MTRL problems in different regimes with or without prior knowledge of models . (3) To the best of our knowledge, TempLe is the fi rst PAC-MDP algorithm that is able to learn tasks with varying state/action spaces without any prior knowledge of inter-task mappings.

## 2 Related Work

PAC-MDP MTRL Algorithms. Brunskill and Li (2013) present the first formal analysis of the sample complexity for MTRL. They propose a two-phase algorithm and prove that per-task sample complexity is reduced compared with single-task learners. However, they require all tasks coming from a small number of models, and when the number of distinct models is large, their algorithm becomes similar to single-task learning. In this paper, we show our proposed methods outperform the method provided by Brunskill and Li (2013) both in theory and in experiments. There are other PAC-MDP algorithms for multi-task RL, considering the problem from different perspectives. For example, Brunskill and Li (2014) discuss lifelong learning in semiMarkov decision processes (SMDPs), where options are involved. Liu, Guo, and Brunskill (2016) extend the finitemodel method (Brunskill and Li 2013) to continuous state space. Feng, Yin, and Yang (2019) and Tirinzoni, Poiani, and Restelli (2020) significantly reduce the sample complexity, but are under the assumption of generative models. Modi et al. (2018) improve the learning efficiency through the assistance of side informations. Abel et al. (2018b) propose MaxQInit, which transfers the maximum Q values across tasks. We empirically compare with MaxQInit in this paper.

Reducing MDPs to Compact Ones. There is a line of research that reduces the original MDPs to compact ones to achieve sample efficiency, including Relocatable Action Model (RAM) (Leffler, Littman, and Edmunds 2007), homomorphism (Ravindran and Barto 2003), and glyph[epsilon1] -equivalent MDP (Even-Dar and Mansour 2003). However, since learning such compact structures is usually difficult (e.g., learning homomorphism is NP-hard as noted by Soni and Singh (2006)), most of the previous works require some prior knowledge. To give a detailed comparison, our algorithm (1) requires no prior knowledge about the MDP structure. RAM (Leffler, Littman, and Edmunds 2007) requires knowledge of the 'type' of all states (walls, pits, etc) and the next-state function of all states and type-action outcomes. Its continuous extension (Brunskill et al. 2008) also needs knowledge of the types. Homomorphism works (Ravindran and Barto 2004, 2003; Soni and Singh 2006) require knowledge of (candidate) homomorphisms to compress an SMDP or transfer knowledge between MDPs. (2) works for general RL problems with PAC guarantee. Although Leffler et al. (2005) (learns latent structure by clustering) and Sorg and Singh (2009) (learns soft homomorphisms) provide methods that do not require knowledge of the structure, Leffler et al. (2005) study a simplified non-MDP problem where actions do not influence state transitions, and Sorg and Singh (2009) do not provide theoretical guarantees when the target model is not known in advance.

Overall, our method is different from the above works, as we do not pre-define the compact structure. Instead, we observe that the transition dynamics, if permuted into descending order, could be naturally grouped to some template. Notably, we learn the similarities rather than assuming knowledge of them. Our method could be more practical than the above works (Leffler, Littman, and Edmunds 2007; Leffler et al. 2005; Brunskill et al. 2008; Ravindran and Barto 2004, 2003; Soni and Singh 2006; Sorg and Singh 2009) in multi-task RL, since a new task is often drawn randomly and knowing its structure in advance could be unrealistic.

Comparison with C-UCRL (Asadi et al. 2019). C-UCRL learns a single task by leveraging a state-action equivalence structure that is similar with our proposed templates. They provide an improved regret bound in the case of a known equivalence structure. However, in the more challenging case of an unknown equivalence structure, as is the setting of our paper, no regret bound is provided. In contrast, our work provides a sample complexity guarantee under the unknown equivalence structure scenario. In addition, C-UCRL does not extend trivially to multi-task setting since it find a coarse partition of all state-action pairs at every step, while in MTRL, new state-action pairs come with new tasks, and negative transfer problem may exist when the equivalence structure is unknown.

## 3 Preliminaries and Notations

Standard RL Notations. An MDP is defined as a tuple 〈S , A , p ( ·|· , · ) , r ( · , · ) , µ, γ 〉 , where S is the state space (with cardinality S ); A is the action space (with cardinality A ); p ( ·|· , · ) is the transition probability function with p ( s ′ | s, a ) representing the probability of transiting to state s ′ from state s by taking action a ; r ( · , · ) is the reward function with r ( s, a ) recording the reward achieved by taking action a in state s ; µ is the initial state distribution; γ is the discount factor. Denote the maximum value of r as R max . Without loss of generality, suppose 0 ≤ r ( s, a ) ≤ 1 for all ( s, a ) , so R max = 1 . Here p ( ·|· , · ) and r ( · , · ) together are the model dynamics of the MDP.

At every step, the agent selects an action based on the current policy π . The value function of a policy V π ( s ) , which evaluates the performance of a policy π , is the expected future reward gained by following π starting from s . Similarly, the action value Q π ( s, a ) is the expected future reward starting from pair ( s, a ) . In an RL task, an agent searches for the optimal policy by interacting with the MDP. We use V max to denote the upper bound of V . In the discounted setting V max = R max 1 -γ = 1 1 -γ .

Sample Complexity. The general goal of RL algorithms is to learn an optimal policy for an MDP with as few interactions as possible. For any glyph[epsilon1] &gt; 0 and any step h &gt; 0 , if the policy π h generated by an RL algorithm L satisfies V ∗ -V π h ≤ glyph[epsilon1] , we say L is near-optimal at step h . If for any 0 &lt; δ &lt; 1 , the total number of steps that L is not nearoptimal is upper bounded by a function ζ ( glyph[epsilon1], δ ) with probability at least 1 -δ , then ζ is called the sample complexity (Kakade et al. 2003) of L .

## 4 Learning with Templates

As motivated in the example described in Section 1, the main idea of this work is to boost the learning process by aggregating similar state-action transition dynamics (see Definition 1). We permute the elements of transition dynamics/probability vectors to be in descending order, and aggregate these permuted transition probabilities to obtain 'templates of transition' defined in Definition 2. We show that the templates are effective abstractions of the environment.

## 4.1 Transition Template: An Abstraction of Dynamics

In this section, we introduce a more compact way to represent the model dynamics of an MDP. We first formally define the transition dynamics of a state-action (s-a) pair.

Definition 1 (State-Action (s-a) Transition Dynamics) . For any state-action pair ( s, a ) , its transition dynamics is defined as a length-( S + 1) vector θ ( s, a ) = [ p ( s 1 | s, a ) , p ( s 2 | s, a ) , · · · , p ( s S | s, a ) , r ( s, a )] , where S is the number of states.

Note that s-a transition dynamics are different from the model dynamics, which characterize the transitions for all sa pairs. In s-a transition dynamics, the first S elements form the transition probability vector p ( ·| s, a ) . As defined in most RL literatures (Kakade et al. 2003; Brunskill and Li 2013), the order of elements in p ( ·| s, a ) is the natural order of the states. In contrast, we re-order the elements of p ( ·| s, a ) by their values, and obtain a more compact representation of the transition dynamics called Transition Template .

Definition 2 (Transition Template) . A Transition Template ( TT ) g is defined as a tuple ( g ( p ) , g ( r ) ) , where g ( p ) ∈ R S is a transition probability vector with non-increasingly ordered elements, i.e., ∑ S i =1 g ( p ) i = 1 and g ( p ) i ≥ g ( p ) j ≥ 0 , ∀ 1 ≤ i ≤ j ≤ S ; 0 ≤ g ( r ) ≤ 1 is a scalar representing the reward.

Any s-a transition dynamics can be permuted to an unique TT by re-arranging the transition probability vector p ( ·| s, a ) in a decreasing order and maintaining the reward r ( s, a ) to g ( r ) , i.e., g ( s,a ) = (desc( p ( ·| s, a )) , r ( s, a )) , where desc orders the elements of p ( ·| s, a ) from the largest value to the smallest value. For example, if θ ( s 1 , a 1 ) = [0 . 3 , 0 . 7 , 0 , 1] , and θ ( s 2 , a 2 ) = [0 , 0 . 3 , 0 . 7 , 1] , then ( s 1 , a 1 ) and ( s 2 , a 2 ) have the same TT ([0 . 7 , 0 . 3 , 0] , 1) , although their s-a transition dynamics are different.

A TT is a representation of multiple s-a transition dynamics with some similarities. It ignores how the s-a pair transits to a specific next state, but only considers the patterns of transition probabilities, allowing more efficient exploitation of similarities. An intuitive example is given in Figure 4 in Appendix A 1 ., where there are 100 distinct s-a transition dynamics, but only 2 distinct TT s. Appendix F.5 further discusses the universal existence of such similarities.

## 4.2 Empirical Estimation of Transition Templates

Section 4.1 defines TT based on the underlying s-a transition dynamics. However, in reality, we do not have access to the underlying dynamics. In model-based RL, a key step is to estimate the dynamics and to build a model of the environment. We now illustrate the estimation of TT s, as well as how TT s augments the learning process.

The conventional estimation of s-a transition dynamics. A direct estimate of θ ( s, a ) is obtained through experience, ˆ θ ( s, a ) = [ n ( s,a,s 1 ) n ( s,a ) , n ( s,a,s 2 ) n ( s,a ) , · · · , n ( s,a,s S ) n ( s,a ) , R ( s,a ) n ( s,a ) ] , where n ( s, a, s ′ ) is the number of observations of transitioning from s to s ′ by taking action a , n ( s, a ) is the total number of observations of ( s, a ) , and R ( s, a ) is the cumulative rewards obtained by ( s, a ) . An accurate estimate of the transition dynamics θ ( s, a ) requires a large enough number of observations n ( s, a ) according to the theory of concentration bounds. Therefore, it is sample-consuming to accurately estimate the transition dynamics of each s-a pair in this way. Augmented estimation of s-a transition dynamics. As discussed in Section 4.1, different s-a pairs may share the same TT s. Our goal is then to aggregate the estimations of s-a transition dynamics associated with the same TT s. We introduce the following process to obtain estimates of all s-a transition

dynamics:

(1) rough estimation : obtain ̂ θ ( s, a ) = [ n ( s,a, · ); R ( s,a ) n ( s,a ) ] for each ( s, a ) with a small n ;

[1 Appendix can be found on https://arxiv.org/abs/2002.06659](https://arxiv.org/abs/2002.06659)

## Algorithm 1 Online Template Learning (O-TempLe)

Input: user-specified TT gap ˆ τ ; error tolerance glyph[epsilon1] ; discount factor γ ; regular known threshold m ; small known threshold m s

Output Near-optimal policies { π t } t =1 , 2 ,

- ··· 1: Initialize an empty TT group set G and TT visit set O 2: for t ← 1 , 2 , · · · do 3: Receive a task M t 4: Initialize visits n ( s, a, · ) ← 0 , accumulative rewards R ( s, a ) ← 0 , ∀ ( s, a ) ∈ ( S , A ) , an empty known state-action set K , and an initial policy π 5: for h ← 1 , 2 , · · · , H do 6: Take action a h ← π ( s h ) , get s h +1 and r h 7: Update visits n ( s h , a h , s h +1 ) and R ( s h , a h ) 8: if ( s h , a h ) / ∈K and ‖ n ( s h , a h , · ) ‖ glyph[lscript] 1 = m s then glyph[triangleright] TT identification with the small threshold 9: ˜ g , o ˜ g , σ ← GEN-TT( n ( s h , a h , · ) , R ( s h , a h ) ) 10: if no g ∈ G is ˆ τ -close to ˜ g then 11: Add ˜ g to G , o ˜ g to O 12: else 13: Find the closest TT g ∗ to ˜ g 14: TT -UPDATE( g ∗ , o g ∗ , n ( s h , a h , · ) , R ( s h , a h ) 15: AUGMENT( o ∗ g , n ( s h , a h , · ) , R ( s h , a h ) , σ ) 16: if ( s h , a h ) / ∈K and ‖ n ( s h , a h , · ) ‖ glyph[lscript] 1 ≥ m then glyph[triangleright] policy update with the regular threshold 17: Update π using visits n and R by RMax 18: add ( s h , a h ) to K 19: for all ( s, a ) ∈ ( S , A ) with identified TT g ( s,a ) do 20: TT -UPDATE( g ( s,a ) , o g ( s,a ) , n ( s, a, · ) , R ( s, a ) )
- (2) permutation : permute each ̂ θ ( s, a ) to its corresponding permuted estimates ˜ g ( s,a ) ;
- (3) template identification : identify the group of the permuted estimate ˜ g ( s,a ) such that permuted estimates are similar within the group, and obtain a more confident estimate of TT ̂ g aggregating within-group statistics.

(4) augmentation : for every ( s, a ) , obtain a more confident estimate of the transition dynamics by permuting back its corresponding TT with accumulated knowledge.

The noisy estimate of transition dynamics will not render error other than the smaller amount of noise in estimated transition templates if it is identified into the right group. To guarantee accurate identification, the ordering of the elements in the noisy estimate should be consistent with the ground truth. Therefore, the consistency of our estimation depends on TT gap as defined in Definition 5 and 'ranking gap' as defined in Definition 8 (see Appendix D for details). An example in Appendix A.1 shows how augmented estimation helps save a large number of samples compared against the conventional estimation.

Now we are ready to formally introduce our algorithms in two settings, Online MTRL and Finite-Model MTRL.

## 4.3 O-TempLe: Online Template Learning

In the online MTRL setting , an agent interacts with multiple tasks streaming-in, each of which corresponding to

<!-- formula-not-decoded -->

## Algorithm 2 TT Functions

- 1: function GEN-TT( n , R ) glyph[triangleright] generate TT 2: find permutation σ s.t. σ ( n ) is in descending order 3: ordered visits o ( N ) g ← σ ( n ) , o ( R ) g ← R , o g ← ( o ( N ) g , o ( R ) g ) 4: transition template g ← ( o ( N ) g ‖ n ‖ glyph[lscript] 1 , o ( R ) g ‖ n ‖ glyph[lscript] 1 ) 5: return g , o g , σ 6: function TT -UPDATE( g , o g , n , R ) glyph[triangleright] add visits to TT 7: o g ← o g +( descending ( n ) , R ) 8: g ← ( o ( N ) g ‖ o ( N ) g ‖ glyph[lscript] 1 , o ( R ) g ‖ o ( N ) g ‖ glyph[lscript] 1 ) 9: function AUGMENT( o g , n , R, σ ) glyph[triangleright] augment visits by TT )

<!-- formula-not-decoded -->

a specific MDP. The tasks are i.i.d. drawn from a set M of MDPs (models). MDPs in M may have different state/action spaces. The number of MDPs |M| can be arbitrarily large.

We introduce Online Template Learning (O-TempLe) for the online MTRL setting. O-TempLe is a meta-learning algorithm with model-based ' base learners ' which compute policies for the current task. We use RMax (Brafman and Tennenholtz 2003) as the base learner, and it can be replaced by other model-based methods such as E 3 (Kearns and Singh 2002) and MBIE (Strehl and Littman 2005). The principle of RMax algorithm on an MDP M is to build an induced MDP based on a known threshold m . A state-action pair is said to be m -known if the number of visits/observations n ( s, a ) ≥ m . A state is m -known if n ( s, a ) ≥ m, ∀ a ∈ A . The set of all m -known states induces an MDP M k , where for any m -known state s , p ( s ′ | s, a ) = n ( s,a,s ′ ) n ( s,a ) , r ( s, a ) = R ( s,a ) n ( s,a ) and for any nonm -known state s , p ( s ′ | s, a ) = I { s ′ = s } , r ( s, a ) = R max . Then, RMax computes an optimal policy based on the optimistic model by dynamic programming.

In contrast, O-TempLe uses augmented estimation introduced in Section 4.2. to reduce the required number of visits to every single s-a pair. Instead of aggregating the estimates of all s-a transition dynamics at once, O-TempLe asynchronously identifies the TT s of s-a pairs and updates the template groups in an online manner, through measuring the distances among TT s.

Algorithm 1 illustrates how O-TempLe works. In addition to the regular known threshold m used in RMax, we design a smaller known threshold m s , which is the smallest number of visits to ensure identifying the TT s of all s-a pairs. If for any ( s, a ) , the total number of visits ( ‖ n ( s, a, · ) ‖ glyph[lscript] 1 ) reaches m s , then the estimated TT ˜ g of ( s, a ) will be generated by function GEN-TT. If ˜ g has at least ˆ τ -distance with all existing TT s, we regard it as a new TT and append it to set G (Line 10-11); otherwise (Line 12-15), we find the closest TT to ˜ g , then synchronize the experience of ( s, a ) in the current task and the accumulated experience that its TT holds by calling functions TT-UPDATE and AUGMENT, which respectively send the current visits of ( s, a ) to the cor- responding TT , and feed the accumulative visits of the TT to the current ( s, a ) . GEN-TT, TT-UPDATE and AUGMENT involve the permutation operations, and are given by Algorithm 2. Accumulated experience of each TT is stored in a tuple o g = ( o ( N ) g , o ( R ) g ) , where o ( N ) g is the total visits accumulated by permuted n ( s, a, s ′ ) of all ( s, a ) 's with TT g . When ( s, a ) is m -known, the policy is updated (Line 1618). Overall, our O-TempLe allows grouped s-a transition dynamics to share their visit counts, making it much easier for them to reach m visits than in regular RMax.

Note that Algorithm 1 also works for tasks with varying state/action space, since the comparison of TT s considers the non-zero elements of the transition vectors only. One can compute the difference between two different-sized TT s by simply padding zeros to the end of the shorter TT .

## 4.4 FM-TempLe: Finite-Model Template Learning

Online MTRL setting requires no prior knowledge of the types of underlying MDPs and improves the sample efficiency by accumulating knowledge with TT groups. However, under a more restrictive assumption that the number of possible MDPs C = |M| is known and small, it is possible to get rid of the dependence on the size of state-action space and achieve more efficient learning .

We propose Finite-Model Template Learning (FMTempLe), an extension of our O-TempLe, under the fi nitemodel MTRL setting , where the agent still interacts with streaming-in tasks drawn from a set M of MDPs, but the number of MDPs in the set M is small and known.

In contrast with O-TempLe, FM-TempLe is able to correctly identify the TT s of some s-a pairs before they are visited for m s times. This is because the number of underlying models is small, and thus identifying the model is easy and inexpensive. It is possible to obtain the TT s for all s-a pairs immediately after identifying the model, since the way how TT s are distributed over all s-a pairs is fixed for each MDP model.

The main steps of FM-TempLe are stated below, and the details are illustrated in Algorithm 3 in Appendix C. (1) Collecting Models: for the first T 1 tasks, the agent acts in the same way as O-TempLe, but also stores the TT structure of each model. (2) Grouping Models: the first T 1 tasks are clustered into finite groups of models based on their TT structures. (3) Identifying Models: for any new task, the agent still follows O-TempLe, but also seeks the true model for the current task from all the model groups, by ruling out the groups of models that have different TT structures.

Brunskill and Li (2013) make the same finite-model assumption and propose an algorithm FMRL which extracts model similarities. However, FMRL can not transfer knowledge between two models which are the same except for one state-action pair. In contrast, our FM-TempLe extracts state-action dynamics similarities and thus transferring happens among any state-action pairs that have similar dynamics. Compared with FMRL, FM-TempLe not only has lower sample complexity as proved in Section 5, but also saves computations due to the direct comparison of TT s.

## 5 Theoretical Analysis

This section provides sample complexity analysis of the proposed two algorithms O-TempLe and FM-TempLe. Although O-TempLe and FM-TempLe can be applied to tasks with varying state/action spaces, we assume all tasks have the same S and A for simplicity of notations, and the analysis extends to varying state/action spaces trivially.

Wefirst assume there is a diameter D such that any state s ′ is reachable from any states s in at most D steps on average. This assumption is commonly used in RL (Jaksch, Ortner, and Auer 2010), and it ensures the reachability of all state from any state on average.

We further define the underlying minimal glyph[lscript] 2 -distance among TT s as τ , namely TT gap. We also define ν as the ranking gap; a large ranking gap implies that for any s-a pair, the probabilities of transitioning to any two states are substantially different. For any g ∈ G , if g ( p ) i &gt; g ( p ) j are two adjacent elements in g ( p ) , then either g ( p ) i -g ( p ) j ≥ ν , or g ( p ) i -g ( p ) j ≤ ˜ O ( glyph[epsilon1] (1 -γ ) √ SV max ) (logarithmic terms are hided in ˜ O ( · ) ). The ranking gap implies that for any s-a pair, the probabilities of transitioning to any two states are either very close, or substantially different. Note that the algorithms take a user-specified τ , but do not require input of ν . See Appendix D for formal definitions of TT gap and ranking gap. For notation simplicity, let ω denote max { min( τ, ν ) , O ( glyph[epsilon1] (1 -γ ) √ SV max ) } .

Theorem 3 ( Sample Complexity of O-TempLe ) . For any given glyph[epsilon1] &gt; 0 , 1 &gt; δ &gt; 0 , running Algorithm 1 on T tasks, each for at least O ( DSA ω 2 ln 1 δ ) steps, generates at most ˜ O ( SGV 3 max glyph[epsilon1] 3 (1 -γ ) 3 + TSAV max ω 2 glyph[epsilon1] (1 -γ ) ) nonglyph[epsilon1] -optimal steps, with probability at least 1 -δ , where G is the total number of TT s.

Remark. (1) Our provided bound achieves state-of-the-art dependence on the environment size T, S, A for general MTRL, given that G is independent of T, S, A . (2) When glyph[epsilon1] is small, the sample complexity only has a linear dependence on the number of states S and the number of templates G , because the first term dominates. By definition, G is always no larger than TSA , the number of all s-a pairs. And in most environments, we have G glyph[lessmuch] TSA , as discussed in Appendix F.5. (3) When glyph[epsilon1] is not small or T is very large, the sample complexity has linear dependences on T , S and A since the second term dominates.

O-TempLe does not necessarily require the number of templates G to be small . A large G suggests the environment is highly stochastic, e.g., the slipping probabilities of every grid in maze is sampled from a Gaussian distribution. In this case, we can still cluster s-a pairs with adequately close templates, as verified in experiments (see Section 6.3). Proof Sketch. We first show that for any s-a pair, m s = ˜ O ( 1 ω 2 ) samples would guarantee correct template identification and aggregation, and m = ˜ O ( SV 2 max glyph[epsilon1] 2 (1 -γ ) 2 ) samples are sufficient for estimating the s-a transition dynamics. Then we prove that all s-a pairs reach m s within finite steps. Finally, by computing the number of visits to unknown s-a pairs and applying the PAC-MDP theorem proposed by Strehl, Li, and Littman (2012), we get the sample complexity result. Proof details are in Appendix E.

Comparison with a single-task learner. If RMax is sequentially run for every task, the total sample complexity for T tasks is ˜ O ( TS 2 AV 3 max glyph[epsilon1] 3 (1 -γ ) 3 ) .

- (1) When precision is high, i.e., glyph[epsilon1] is small, a significant improvement is achieved, if O ( SG ) glyph[lessmuch] O ( TS 2 A ) .
- (2) When T is large, as long as ˜ O ( SV 2 max glyph[epsilon1] 2 (1 -γ ) 2 ) glyph[greatermuch] ˜ O ( 1 ω 2 ) , our O-TempLe gains improved sample efficiency.
- (3) O-TempLe will not cause negative transfer among tasks. In the worst case, G = TSA (there is no similarity among all s-a transition dynamics) or ω 2 = ˜ O ( SV 2 max glyph[epsilon1] 2 (1 -γ ) 2 ) , OTempLe has the same-order sample complexity with RMax.

Theorem 4 ( Sample Complexity of FM-TempLe ) . Under the finite-model assumption of there are at most C MDPs for all tasks, for any given glyph[epsilon1] &gt; 0 , 1 &gt; δ &gt; 0 , Algorithm 3 on T tasks follows glyph[epsilon1] -optimal policies for all but

<!-- formula-not-decoded -->

steps with probability at least 1 -δ , where G is the total number of TT s, T 1 = Ω( 1 p min ln C δ ) is the number of tasks in the first phase, where p min is the minimal probability for a task to be drawn from M .

Remark. (1) When C is very large, or p min is very small, T 1 → T and FM-TempLe degenerates to O-TempLe. (2) If DC 2 &lt; SA and T glyph[greatermuch] T 1 , FM-TempLe requires fewer samples than O-TempLe.

Comparison with FMRL (Brunskill and Li 2013) FMTempLe has a large improvement over FMRL in most cases. The sample complexity of FMRL for T tasks in our notation is

<!-- formula-not-decoded -->

where T 1 = Ω( 1 p min ln C δ ) , and Γ is the model difference gap defined by Brunskill and Li (2013). We organize Equation 1 and Equation 2 both as three-term forms. The first term is for learning of all TT s or all models, where FMTempLe reduces the dependence on S and gets rid of the dependence on A . The second term is for the first phase, where FMRL performs the same with a single-task RMax learner, while FM-TempLe requires much fewer samples to get optimal policies. Finally, the last term is for the second phase. FMRL needs an additional model elimination step for each task, while FM-TempLe does not. FM-TempLe is worse than FMRLonlyin extreme cases where there are few MDP models with large model gaps, and a large number of TT s with small TT gaps or ranking gaps.

## 6 Experiments

In this section, we demonstrate empirical results to show O-TempLe and FM-TempLe outperform existing state-ofthe-art algorithms both in the finite-model setting and in the more realistic online setting. TempLe is able to transfer knowledge between tasks with different sized environments. More importantly, TempLe has a high tolerance to model perturbations; it implements efficient transfer even when the underlying number of TT s is infinite. Our code is available at https://github.com/umd-huang-lab/templatereinforcement-learning.

Baselines. We choose the state-of-the-art MTRL algorithms, Abstraction RL (Abs-RL) (Abel et al. 2018a), MaxQInit (Abel et al. 2018b) and FMRL (Brunskill and Li 2013) as baselines. For Abs-RL and MaxQInit, we use the code provided by authors. Note that Abs-RL and MaxQInit have multiple versions due to the selection of different base learners, we show the ones with their best performance in this section, and other versions in Appendix F.3. Abs-RL works for both the online and finite-model setting, whereas MaxQInit and FMRL work for the finite-model setting only, since they both require the number of tasks to be small and known. Meanwhile, to show the effectiveness of our proposed algorithms and other MTRL algorithms, we also run RMax and Q-learning (Watkins and Dayan 1992) for every single task without knowledge transfer.

## 6.1 Finite-Model MTRL

Environment. All the baselines including FMRL, Abs-RL, MaxQInit are designed for the finite-model setting (note that Abs-RL also works in the online setting), where the number of models C is small. We use a similar maze environment as in FMRL, where MDPs only differ at the goal state.

Performance. We generate two 4 × 4 maze tasks with different goal states as the underlying models, and then randomly sample 50 tasks from the two underlying models. Figure 2a shows the comparison of per-task rewards. FMRL has the same performance with RMax in the model-collecting phase, and then achieves increasing rewards in the following tasks after it successfully identifies the underlying two types of MDPs. After 30 tasks, all state-actions pairs in the models become known, so the per-task reward converges. Similarly, MaxQInit gains more rewards when it collects adequate knowledge of the Q values. In contrast, FM-TempLe has a better start as it learns TT s from the beginning. And model identification further helps with efficient learning. Over all tasks, FM-TempLe substantially outperforms other agents, despite that baselines are designed for the finite-model case.

## 6.2 Online MTRL

Environment. For the more realistic Online MTRL which allows the number of MDP models to be extremely large , we generalize the traditional maze environment to have arbitrary combinations of landforms, as shown in Figure 1. We use 3 types of landforms, sand, marble and ice, respectively with slipping probabilities 0, 0.2, and 0.4. In this scenario, under a certain number of states S , the number of possible tasks is exponential in S .

Performance. In the online setting, we consider 4 × 4 mazes with different arrangements of landforms streaming in. The per-task rewards of each agent are displayed in Figure 2b. Among all agents, our O-TempLe obtains the highest average reward. We see during the first 40 tasks, the performance of O-TempLe continuously and rapidly grows by transferring previous knowledge. In contrast, the performance of Abs-RL does not increase as more tasks come in and keeps the same with single-task Q-learning, because the maze environment is not efficiently abstracted by Abs-RL.

Figure 2: Performance of O-TempLe and FM-TempLe compared against state-of-the-art baselines in (a) Online MTRL (to show TempLe's ability to efficiently transfer knowledge), (b) Finite-Model MTRL (to show TempLe outperforms baselines even under environments that the baselines are designed for), (c) varying sized MTRL (to show TempLe extends to varying sized state space) and (d) Online MTRL with Mixture-of-Gaussians distributed landforms (to show TempLe's robustness against noise and model-perturbation). All results are averaged over 20 different random sequences of tasks. Confidence intervals are omitted to reduce overlapping.

<!-- image -->

Performance on Varying State Space. To show the feasibility of TempLe for varying-sized environment tasks, and its ability to generalize knowledge learned in small tasks to speed up learning in larger tasks, we vary the size of the mazes across tasks. More specifically, the first 20 tasks are 3 × 3 mazes, followed by 20 4 × 4 mazes, 20 5 × 5 mazes and 20 6 × 6 mazes. We show O-TempLe's per-task advantage rewards over single task RMax in Figure 2c, since other MTRL baselines are not feasible in this setting. The performance advantage over RMax increases over more observed tasks, verifying that O-TempLe transfers knowledge among different-sized mazes. Experiments on varying action spaces are shown in Appendix F.4.

## 6.3 MTRL with Infinite TT s

Environment. We also conduct experiments to show TempLe's robustness to noise and model perturbations. which is crucial for its application to real-world settings where 'landforms' could vary continuously. We draw the landforms (slipping probabilities) of each grid from a mixture of Gaussian distributions, which are centered at 0.2, 0.4, and 0.6 with standard derivation 0.05. In this case, the number of TT s could be infinitely large.

Performance. We show O-TempLe's per-task advantage rewards over single task RMax and Q-learning in Figure 2d, in which O-TempLe still achieves successful multi-task learning. This result demonstrates O-TempLe's ability of tolerating noise and generalizing to real-life applications.

## 6.4 Robustness to Hyper-parameters

TempLe requires a user-specified TT gap ˆ τ as input. Also, both FMRL and FM-TempLe require a user-specified model gap Γ . We test various hyper-parameters to understand how significantly the performance of the algorithms could be affected by inaccurate guesses of ˆ τ and Γ , shown in Figure 3.

Figure 3: Hyper-parameter test of TT gap ˆ τ and model gap Γ (the vertical dashed line shows the underlying true value).

<!-- image -->

According to Figure 3a, the performance of O-TempLe drops when ˆ τ is too large. However, the rewards remain high for relatively small ˆ τ . Figure 3b shows that FM-TempLe gets higher rewards than RMax when setting Γ ≤ 1 , although Γ has a larger influence on FM-TempLe compared to FMRL, potentially because the failure of model clustering will cause more inaccurate TT identification. Note that by definition, both ˆ τ and Γ would not exceed 2 (see Lemma 6). So we still have a large chance to get higher rewards than RMax by making an educated guess. The results in Figure 3 guide the users to specify hyper-parameters when using TempLe.

The results with confidence intervals, comparison of cumulative rewards, and experiments on additional environments are shown in Appendix F. We also provide an extension of our work to deep RL is discussed in Appendix F.6.

## 7 Conclusion and Discussion

In this work, we propose TempLe, the first PAC-MDP MTRL algorithm that works for tasks with varying state/action space without any inter-task mappings or prior knowledge of the MDP structures. This work can be extended in many directions. For example, one may benefit from investigating transition probability and reward separately. The idea of extracting modular similarities can also be extended to continuous MDP and deep model-based RL.

## Acknowledgements

Huang is supported by startup fund from Department of Computer Science of University of Maryland, National Science Foundation IIS-1850220 CRII Award 03074200001, DOD-DARPA-Defense Advanced Research Projects Agency Guaranteeing AI Robustness against Deception (GARD), Laboratory for Physical Sciences at University of Maryland, and Adobe, Capital One and JP Morgan faculty fellowships.

## Ethical Impact

Our presented algorithms on multi-task reinforcement learning facilitate the learning of new tasks using knowledge accumulated from previously learned tasks. In scenarios where an RL agent needs to sequentially interact with a series of environments, e.g., navigation in various places, our proposed algorithm could be applied to improve the learning efficiency without loss of accuracy. More importantly, our algorithms are guaranteed to learn near-optimal policies and avoid negative transfer, which are crucial for high-stakes applications, such as autonomous driving, market making, and health-care systems.

Nowadays, Deep Reinforcement Learning (DRL) has achieved great success in many applications. However, problems like high variance and instability restrict the use of DRL in real-life problems. Thus, it is important to study tabular RL with guarantees, which could potentially benefit DRL and applications involving DRL. Our proposed algorithms, although not in the scope of DRL, could be potentially extended to DRL in the following ways. (1) Our idea of extracting 'relative' transition probability similarity could be directly used in model-based DRL. For example, the nextstate prediction model usually outputs a Gaussian distribution for every s-a pair, and one can augment the learned derivation by averaging over predicts with close derivations, assuming some similarity about the uncertainty among different states. (2) It is possible to discretize state space and apply count-based methods, as suggested in by Tang et al. (2016).

Our work on multi-task reinforcement learning also has the potential to be applied to other transfer learning tasks within and outside of the Reinforcement Learning community. Any learning in systems that share modular similarities could potentially benefit from our algorithms to speed up the training process.

## References

Abel, D.; Arumugam, D.; Lehnert, L.; and Littman, M. 2018a. State abstractions for lifelong reinforcement learning. In International Conference on Machine Learning , 1019.

Abel, D.; Jinnai, Y.; Guo, S. Y.; Konidaris, G.; and Littman, M. 2018b. Policy and value transfer in lifelong reinforcement learning. In International Conference on Machine Learning , 20-29.

Ammar, H. B.; Eaton, E.; Luna, J. M.; and Ruvolo, P. 2015. Autonomous cross-domain knowledge transfer in lifelong policy gradient reinforcement learning. In Twenty-Fourth International Joint Conference on Artificial Intelligence .

Asadi, M.; Talebi, M. S.; Bourel, H.; and Maillard, O.-A. 2019. Model-Based Reinforcement Learning Exploiting State-Action Equivalence. arXiv preprint arXiv:1910.04077 .

Brafman, R. I.; and Tennenholtz, M. 2003. R-max - a General Polynomial Time Algorithm for Near-optimal Reinforcement Learning. J. Mach. Learn. Res. 3: 213-231. ISSN 1532-4435.

Brockman, G.; Cheung, V.; Pettersson, L.; Schneider, J.; Schulman, J.; Tang, J.; and Zaremba, W. 2016. OpenAI Gym.

Brunskill, E.; Leffler, B. R.; Li, L.; Littman, M. L.; and Roy, N. 2008. CORL: A Continuous-State Offset-Dynamics Reinforcement Learner. UAI'08, 53-61. Arlington, Virginia, USA: AUAI Press. ISBN 0974903949.

Brunskill, E.; and Li, L. 2013. Sample Complexity of Multi-Task Reinforcement Learning. In Proceedings of the Twenty-Ninth Conference on Uncertainty in Artificial Intelligence , UAI'13, 122-131. Arlington, Virginia, USA: AUAI Press.

Brunskill, E.; and Li, L. 2014. PAC-Inspired Option Discovery in Lifelong Reinforcement Learning. In Proceedings of the 31st International Conference on International Conference on Machine Learning - Volume 32 , ICML'14, II-316-II-324. JMLR.org.

Even-Dar, E.; and Mansour, Y. 2003. Approximate equivalence of Markov decision processes. In Learning Theory and Kernel Machines , 581-594. Springer.

Feng, F.; Yin, W.; and Yang, L. F. 2019. How Does an Approximate Model Help in Reinforcement Learning? arXiv e-prints arXiv:1912.02986.

Hayes, T. P. 2005. A large-deviation inequality for vectorvalued martingales. Combinatorics, Probability and Computing .

Jaksch, T.; Ortner, R.; and Auer, P. 2010. Near-optimal regret bounds for reinforcement learning. Journal of Machine Learning Research 11(Apr): 1563-1600.

Kaiser, L.; Babaeizadeh, M.; Milos, P.; Osinski, B.; Campbell, R. H.; Czechowski, K.; Erhan, D.; Finn, C.; Kozakowski, P.; Levine, S.; et al. 2019. Model-based reinforcement learning for atari. arXiv preprint arXiv:1903.00374 .

Kakade, S. M.; et al. 2003. On the sample complexity of reinforcement learning . Ph.D. thesis, University of London London, England.

Kearns, M.; and Singh, S. 2002. Near-optimal reinforcement learning in polynomial time. Machine learning 49(23): 209-232.

Konidaris, G.; and Barto, A. 2006. Autonomous shaping: Knowledge transfer in reinforcement learning. In Proceedings of the 23rd international conference on Machine learning , 489-496.

Leffler, B. R.; Littman, M. L.; and Edmunds, T. 2007. Efficient reinforcement learning with relocatable action models. In AAAI , volume 7, 572-577.

Leffler, B. R.; Littman, M. L.; Strehl, A. L.; and Walsh, T. J. 2005. Efficient Exploration With Latent Structure. In Robotics: Science and Systems , 81-88.

Li, S.; and Zhang, C. 2018. An optimal online method of selecting source policies for reinforcement learning. In ThirtySecond AAAI Conference on Artificial Intelligence .

Liu, Y.; Guo, Z.; and Brunskill, E. 2016. PAC continuous state online multitask reinforcement learning with identification. In Proceedings of the 2016 International Conference on Autonomous Agents &amp; Multiagent Systems , 438-446.

Mann, T. A.; and Choe, Y. 2012. Directed Exploration in Reinforcement Learning with Transferred Knowledge. In Ewrl , 59-76.

Modi, A.; Jiang, N.; Singh, S.; and Tewari, A. 2018. Markov decision processes with continuous side information. In Algorithmic Learning Theory , 597-618.

Nagabandi, A.; Kahn, G.; Fearing, R. S.; and Levine, S. 2018. Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning. In 2018 IEEE International Conference on Robotics and Automation (ICRA) , 7559-7566. IEEE.

Ramamoorthy, S.; Mahmud, M.; Hawasly, M.; and Rosman, B. 2013. Clustering markov decision processes for continual transfer. School of Informatics, University of Edinburgh, Tech. Rep .

Ravindran, B.; and Barto, A. G. 2003. SMDP Homomorphisms: An Algebraic Approach to Abstraction in SemiMarkov Decision Processes. In Proceedings of the 18th International Joint Conference on Artificial Intelligence , IJCAI'03, 1011-1016. San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.

Ravindran, B.; and Barto, A. G. 2004. An algebraic approach to abstraction in reinforcement learning . Ph.D. thesis, University of Massachusetts at Amherst.

Sharma, M.; Holmes, M. P.; Santamar´ ıa, J. C.; Irani, A.; Isbell Jr, C. L.; and Ram, A. 2007. Transfer Learning in RealTime Strategy Games Using Hybrid CBR/RL. In IJCAI , volume 7, 1041-1046.

Soni, V.; and Singh, S. 2006. Using homomorphisms to transfer options across continuous reinforcement learning domains. In AAAI , volume 6, 494-499.

Sorg, J.; and Singh, S. 2009. Transfer via soft homomorphisms. In Proceedings of The 8th International Conference on Autonomous Agents and Multiagent Systems-Volume 2 , 741-748.

Strehl, A.; and Littman, M. 2004. Exploration via model based interval estimation. In International Conference on Machine Learning . Citeseer.

Strehl, A. L.; Li, L.; and Littman, M. L. 2006. Incremental Model-Based Learners with Formal Learning-Time Guarantees. In Proceedings of the Twenty-Second Conference on Uncertainty in Artificial Intelligence , UAI'06, 485-493. Arlington, Virginia, USA: AUAI Press. ISBN 0974903922.

Strehl, A. L.; Li, L.; and Littman, M. L. 2012. Incremental model-based learners with formal learning-time guarantees. arXiv preprint arXiv:1206.6870 .

Strehl, A. L.; and Littman, M. L. 2005. A Theoretical Analysis of Model-Based Interval Estimation. In Proceedings of the 22Nd International Conference on Machine Learning , ICML '05, 856-863. New York, NY, USA: ACM. ISBN 1-59593-180-5. doi:10.1145/1102351.1102459.

Strehl, A. L.; and Littman, M. L. 2008. An analysis of model-based interval estimation for Markov decision processes. Journal of Computer and System Sciences 74(8): 1309-1331.

Tang, H.; Houthooft, R.; Foote, D.; Stooke, A.; Chen, X.; Duan, Y.; Schulman, J.; De Turck, F.; and Abbeel, P. 2016. #Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning. arXiv e-prints arXiv:1611.04717.

Taylor, M. E.; and Stone, P. 2007. Representation Transfer for Reinforcement Learning. In AAAI Fall Symposium: Computational Approaches to Representation Change during Learning and Development , 78-85.

Taylor, M. E.; and Stone, P. 2009. Transfer learning for reinforcement learning domains: A survey. Journal of Machine Learning Research 10(Jul): 1633-1685.

Tirinzoni, A.; Poiani, R.; and Restelli, M. 2020. Sequential Transfer in Reinforcement Learning with a Generative Model. In III, H. D.; and Singh, A., eds., Proceedings of the 37th International Conference on Machine Learning , volume 119 of Proceedings of Machine Learning Research , 9481-9492. PMLR.

Torrey, L.; and Shavlik, J. 2010. Transfer learning. In Handbook of research on machine learning applications and trends: algorithms, methods, and techniques , 242-264. IGI Global.

Watkins, C. J.; and Dayan, P. 1992. Q-learning. Machine learning 8(3-4): 279-292.

Wilson, A.; Fern, A.; Ray, S.; and Tadepalli, P. 2007. Multitask reinforcement learning: a hierarchical Bayesian approach. In Proceedings of the 24th international conference on Machine learning , 1015-1022. ACM.

(a) action=' ↑ '

<!-- image -->

## A Intuitive Examples

<!-- image -->

(b) action=' ↓ '

<!-- image -->

(c) action=' ← '

(d) action=' → '

<!-- image -->

Figure 4: An example of TT s in a 5 × 5 slippery gridworld with no reward and slipping probability = 0.4. The template at all s is g 1 = ([0 . 8 , 0 . 2 , 0 , · · · , 0] , 0) , and the template at all s is g 2 = ([0 . 6 , 0 . 2 , 0 . 2 , 0 , · · · , 0] , 0) .

Explanations for Figure 4. Consider a 5 × 5 gridworld where the agent has 4 actions: ↑ , ↓ , ← and → , as well as 25 states, as shown in Figure 4a, 4b, 4c and 4d. Thus there are 100 distinct state-action pairs in total. Since the slipping probability is 0.4, which means action ↑ will become ← or → with probability 0.2 respectively, we know the transition probability p ( ·| s = 1 , a = ↑ ) is

<!-- formula-not-decoded -->

By re-ordering, its TT is ([0 . 8 , 0 . 2 , 0 , · · · , 0] , 0) .

Similarly, for state 2 and action ↑

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

,

Its TT is ([0 . 6 , 0 . 2 , 0 . 2 , 0 , · · · , 0] , 0)

For state 3 and action ↑

. ,

Its TT is also ([0 . 6 , 0 . 2 , 0 . 2 , 0 , · · · , 0] , 0) .

In this way, we can find that there are only 2 distinct TT s, like shown in the figure, which is much less than the number of state-action pairs. The state-action pairs with the same TT s are able to share the 'relative' probability of transitioning.

## A.1 An Example Illustrating Sample Efficiency of Our Algorithm

We use the gridworld in Figure 4 as an example to illustrate how the augmented estimation saves samples by estimating TT s instead of every s-a pairs. For simplicity, we assume the possibility of arbitrarily sampling any s-a pair and observing its transitions. The gridworld in Figure 4 has 100 distinct s-a transition dynamics, but only 2 distinct TT s. The distance between the 2 TT s is 0 . 28 .

The objective is to estimate p ( s, a ) such that ‖ ˆ p ( s, a ) -p ( s, a ) ‖ ≤ 0 . 01 for all ( s, a ) with probability 95% . Using the conventional estimation, the total number of samples we need is 2 O (100 × 1 0 . 01 2 ln 100 0 . 05 ) ≈ 7 . 6 × 10 6 . However our proposed augmented estimation only requires ∼ 1 . 25% of samples needed by the conventional estimation, since it takes O (100 × 1 0 . 28 2 ln 100 0 . 025 ) ≈ 1 . 1 × 10 4 samples to correctly identify the TT s of all s-a pairs with probability 1 -δ/ 2 , plus O (2 × 1 0 . 01 2 ln 2 0 . 025 ) ≈ 8 . 4 × 10 4 samples to get 0 . 01 -accurate estimations of 2 TT s with probability 1 -δ/ 2 .

2 According to Hoeffding's inequality, O ( 1 α 2 ln 1 δ ) samples are needed to achieve an α -accurate estimation with probability 1 -δ .

## B Additional Related Works

Non-PAC MTRL Algorithms. Besides the above methods with PAC guarantees, there are many interesting approaches aiming to effectively transfer knowledge across tasks. Some approaches augment the learning of a new task by reusing the policies learned from previous tasks (Ramamoorthy et al. 2013; Li and Zhang 2018), though a library of reliable source policies is usually needed. The hierarchical multi-task learning algorithm proposed by Wilson et al. (2007) learns a Bayesian mixture model from previous tasks, and use the learned distribution as a prior of new tasks. Although good experimental results are shown, there are no theoretical guarantees. (Abel et al. 2018a) introduce two types of state-abstractions that can reduce the problem complexity and improve the learning of new tasks. However, as shown in the paper, the proposed abstractions, when combined with PAC-MDP algorithms such as RMax (Brafman and Tennenholtz 2003), the PAC guarantee does not hold anymore and the number of mistakes made by the agent can be arbitrarily large. In this paper, we show our proposed method outperforms the abstraction method empirically.

Empirical Studies. MTRL/lifelong RL and Transfer Learning (TL) (Torrey and Shavlik 2010) are closely related, and have been studied for years. Taylor and Stone (2009) survey a wide range of empirical results on transferring knowledge among tasks, and point out some problems of previous works, including negative transfer, partially due to the lack of theoretical analysis.

Cross-Domain Learning. Knowledge transfer between tasks with various state/action spaces (task domains) is also an important topic. As summarized by (Taylor and Stone 2009), most early works require hand-coded inter-task mappings (Taylor and Stone 2007), or only learn from unchanged problem representations (Konidaris and Barto 2006; Sharma et al. 2007). Recently, Ammar et al. (2015) propose an algorithm that can perform cross-domain transfer efficiently. The authors assume tasks are from a finite set of domains, and parametrize each task's policy by the product of a shared knowledge base and task-specific coefficients. However, although convergence guarantee is provided, there is no guarantee for sample efficiency. Mann and Choe (2012) study cross-domain transfer learning with an inter-task mapping of s-a pairs (with similarly-bounded Q values). However, such inter-task mappings are often not available in multi-task RL.

## C Algorithm Pseudo-code

Procedure 3 shows FM-TempLe introduced in Section 4.4, which learns TT s as Procedure 1 does, and also expedites learning by clustering models.

glyph[negationslash]

glyph[negationslash]

```
Algorithm 3 Finite-Model Template Learning (FM-TempLe) Input: ˆ τ ; glyph[epsilon1] ; γ ; m s ; m (same as Procedure 1); number of tasks in the first phase T 1 ; number of models C ; model error tolerance η Output Near-optimal policies { π t } t =1 , 2 , ··· 1: Initialize the TT group set G , the TT visit set O , and the MDP group set C as empty 2: for t ← 1 , 2 , · · · , T 1 do glyph[triangleright] Phase 1 3: Run Procedure 1 Line 3-18 and get visits n ( s, a, · ) and R ( s, a ) , ∀ ( s, a ) ∈ ( S , A ) 4: Cluster the past T 1 tasks into C groups (store in C ). glyph[triangleright] model clustering 5: g ( s,a,c ) , σ ( s,a,c ) ← GEN-TT( n c ( s, a ) , R c ( s, a ) ) ∀ ( s, a, c ) 6: for t ← T 1 +1 , T 1 +2 , · · · do glyph[triangleright] Phase 2 7: Receive a task M t , do initializations as Line 3 in Procedure 1 8: Initialize model score u ( c ) ← η, ∀ c ∈ C glyph[triangleright] u ( c ) measures how possible c is the true model for the current task 9: for h ← 1 , 2 , · · · , H do 10: Run Procedure 1 Line 5-16, and get updated n ( s h , a h , · ) , R ( s h , a h ) , g ( s h ,a h ) , σ ( s h , a h ) 11: if ‖ n ( s h , a h , · ) ‖ glyph[lscript] 1 = m s then glyph[triangleright] TT is identified 12: for c ∈ C do 13: if g ( s h ,a h ,c ) = g ( s h ,a h ) or σ ( s h ,a h ,c ) = σ ( s h ,a h ) then 14: u ( c ) ← u ( c ) -1 glyph[triangleright] group identification 15: if ∃ only 1 group c ∗ ∈ C s.t. u ( c ∗ ) > 0 then 16: AUGMENT( o g ( s,a,c ∗ ) , n ( s, a, · ) , R ( s, a ) , σ s,a ) ∀ ( s, a ) 17: Run Procedure 1 Line 17-18
```

## D Additional Definitions

Definition 5 ( TT Gap) . Define the TT distance between two TT s g a and g b as ρ ( g a , g b ) = ‖ g ( p ) a -g ( p ) b ‖ 2 + | g ( r ) a -g ( r ) b | . Suppose there is a minimum TT distance τ , such that for any two different TT s g a , g b ∈ G , ρ ( g a , g b ) ≥ τ . We name τ as TT gap.

Remark. According to Lemma 6, the TT gap between any two TT s will not exceed 2 (suppose reward is in between 0 and 1).

Lemma 6. Let a = ( a 1 , . . . , a n ) and b = ( b 1 , . . . , b n ) be two vectors in R n such that ∑ n i =1 a i = ∑ n j =1 b j = 1 . Moreover, assume there hold

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The equality holds when we choose, e.g., a = (1 , 0 , . . . , 0) and b = ( 1 n , 1 n , . . . , 1 n ) .

Proof. We prove the lemma by induction.

- When n = 1 , a = b = 1 . Thus the inequality is trivial.
- Assume (6) holds for n = k , k ≥ 1 . We show that (6) is also true for n = k +1 . Given vectors a = ( a 1 , . . . , a k , a k +1 ) and b = ( b 1 , . . . , b k , b k +1 ) such that they satisfy the conditions in the lemma, construct two new vectors:

<!-- formula-not-decoded -->

It is obvious that a ′ and b ′ satisfy the conditions in the induction hypothesis. Thus,

<!-- formula-not-decoded -->

Then

We calculate

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the last equality comes from the fact

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By assumptions,

hence

Similarly, Combine (7), (10), (11) and (12),

## E.1 Proof of Theorem 3

To prove Theorem 3, we first present Lemma 9, Lemma 11, Lemma 12, Lemma 13 and Lemma 14.

Lemma 9 and Lemma 11 provide the sample size requirements for correctly identifying the TT of an s-a pair.

Lemma 9. For any state-action pair, suppose the ranking permutation of the estimated probability vector is the same with that of the underlying probability vector, then it would be identified to its corresponding TT group correctly with O ( 1 τ 2 ln 1 δ ) samples, with probability at least 1 -δ , where τ is the TT gap defined in Definition 5.

Proof. For an state-action pair ( s, a ) , define the observation vector of the i th sample as Z i = [ I s ′ = s 1 , I s ′ = s 2 , · · · , I s ′ = s S , r ] . Define X n = ∑ n Z i -nθ ( s, a ) , and set X 0 = 0 .

We first prove the sequence { X n }

i =1 is a vector-valued martingale.

<!-- formula-not-decoded -->

Obviously, E [ ‖ X n ‖ ] &lt; ∞ for all n . Thus { X n } is a (strong) martingale.

<!-- formula-not-decoded -->

Thus (6) also holds for n = k +1 . The proof is finished.

The permutation from an s-a transition dynamics to a TT is recorded by a ranking permutation defined as below.

Definition 7 (Ranking Permutation) . For an s-a pair ( s, a ) with transition probability vector p ∈ R S where p i = p ( s i | s, a ) , by sorting its elements from the largest to the smallest value, we get an ordered vector g ( p ) . Define function σ : { 1 , · · · , S } → { 1 , · · · , S } as a mapping from ranking to the indices in p . For example, σ ( i ) is the index of the i -th largest element of p , i.e., g ( p ) i = p σ ( i ) . The inverse function σ -1 maps indices to ranking. So σ -1 ( j ) is the ranking of p j , i.e., g ( p ) σ -1 ( j ) = p j . The way way? ordering is unique if for any p i = p j and i &lt; j , we put p i before p j in g ( p ) . As a result, σ ( · ) is a bijection and can be regarded as a permutation. We call it ranking permutation of ( s, a ) .

For simplicity, we slightly abuse notation and use σ ( p ) to denote the re-ordered vector. Thus g ( p ) = σ ( p ) and p = σ -1 ( g ( p ) ) .

Definition 8 (Ranking Gap) . Define ν as the minimal notable ranking gap, such that for any g ∈ G , if g ( p ) i &gt; g ( p ) j are two adjacent elements in g ( p ) and g ( p ) i -g ( p ) j ≥ O ( glyph[epsilon1] (1 -γ ) √ SV max ) , then g ( p ) i -g ( p ) j ≥ ν holds. In other words, two adjacent elements of g ( p ) satisfy either g ( p ) i -g ( p ) j ≥ ν or g ( p ) i -g ( p ) j ≤ O ( glyph[epsilon1] (1 -γ ) √ SV max ) .

If two adjacent elements are different by no more than O ( glyph[epsilon1] (1 -γ ) √ SV max ) , then the corruption can be ignored because it will not influence the value of the policy too much, as proved in Lemma 13. Otherwise we need adequate samples to make sure the ranking of elements will succeed.

## E Proofs of Main Theorems

By application of the extended Hoeffding's inequality (Hayes 2005), we get

<!-- formula-not-decoded -->

Set the failure probability as δ , we obtain n ≥ O ( 1 glyph[epsilon1] 2 ln 1 δ ) .

Lemma 9 assumes perfect permutation, and Lemma 11 addresses the problem of how to avoid notable corruptions of permutations. For ease of illustrating, we define the concept almost the same and almost correct in Definition 10.

Definition 10 (Almost the same and almost correct) . If for two probability vectors p and p ′ , their ranking permutations σ and σ ′ are the same except for elements whose difference is smaller than O ( glyph[epsilon1] (1 -γ ) √ SV max ) , then we call σ and σ ′ almost the same. If p ′ is the approximation of the ground truth p , then we call σ ′ almost correct.

Lemma 11. With ˜ O ( 1 ν 2 ln S δ ) samples of a s-a pair, its transition permutation will be almost correct, with probability 1 -δ .

Proof. Suppose there is a transition probability vector p with length l , as well as transition-difference gap ν . We estimate p by randomly sampling indices 1 , · · · , i, · · · , l according to the probability distribution p . Let ˆ p = [ n (1) n , · · · , n ( l ) n ] .

For any two adjacent elements p i and p j (adjacent means there is no p k whose value is between p i and p j ), we want our estimations n ( i ) n and n ( j ) n to satisfy n ( i ) n &gt; n ( j ) n . It is sufficient if we guarantee n ( i ) n and n ( j ) n are respectively ν/ 2 -close to p i and p j . By Hoeffding's inequality,

<!-- formula-not-decoded -->

Therefore, n ≥ O ( 1 ν 2 ln 1 δ ) is sufficient. By union bound, we have n ≥ O ( 1 ν 2 ln S δ ) .

Lemma 9 and Lemma 11 imply that the small threshold should satisfy m s = ˜ O ( 1 min { τ 2 ,ν 2 } ) .

Then, Lemma 12 claims if horizon is set to be large enough, all s-a pairs will have sufficient samples to be correctly grouped.

Lemma 12. If H = ˜ O ( DSA ω 2 ) , all state-action pairs in the task will have at least ˜ O ( 1 ω 2 ln TSA δ ) samples with probability 1 -δ .

The proof of Lemma 12 is similar to Lemma 2.1 in paper (Brunskill and Li 2013).

Lemma 13 is a variant of the 'simulation lemma' (Kearns and Singh 2002; Brafman and Tennenholtz 2003) with TT estimation.

Lemma 13. For any two MDPs M and ˜ M with the same S , A , µ, γ , if for any s-a pair ( s, a ) , the ranking permutations of p ( s, a ) and ˜ p ( s, a ) are almost the same, and g = ( desc ( p ( s, a )) , r ( s, a )) as well as ˜ g = ( desc (˜ p ( s, a )) , ˜ r ( s, a )) satisfy ‖ g ( p ) -˜ g ( p ) ‖ ≤ O ( glyph[epsilon1] (1 -γ ) V max ) and | g ( r ) -˜ g ( r ) | ≤ O ( glyph[epsilon1] (1 -γ ) V max ) , then for any policy π , | V π M -V π ˜ M | ≤ glyph[epsilon1] .

Proof. For an s-a pair ( s, a ) , suppose its ranking permutation in M is σ , and the ranking permutation in ˜ M is ˜ σ .

We first assume σ and ˜ σ are exactly the same. So we have g ( p ) = σ ( p ( s, a )) and ˜ g ( p ) = σ (˜ p ( s, a )) .

Thus ‖ g ( p ) -˜ g ( p ) ‖ ≤ O ( glyph[epsilon1] (1 -γ ) V max ) implies ‖ p ( s, a ) -˜ p ( s, a ) ‖ ≤ O ( glyph[epsilon1] (1 -γ ) V max ) , because of the property of permutation. Similarly, | g ( r ) -˜ g ( r ) | ≤ O ( glyph[epsilon1] (1 -γ ) V max ) implies | r ( s, a ) -˜ r ( s, a ) | ≤ O ( glyph[epsilon1] (1 -γ ) V max ) .

Then, following the standard proof (Strehl and Littman 2008; Strehl, Li, and Littman 2006), it is easy to show | V π M -V π ˜ M | ≤ glyph[epsilon1] . Next, we allow σ and ˜ σ be almost the same (see Definition 10), and show that it only causes up to a constant factor increase in the value difference | V π M -V π ˜ M | .

Without loss of generality, assume ˜ σ only reverses σ in indices i and j , i.e., σ ( i ) = ˜ σ ( j ) and σ ( j ) = ˜ σ ( i ) . According to the definition of almost the same, p σ ( i ) -p σ ( j ) = p ˜ σ ( j ) -p ˜ σ ( i ) ≤ O ( glyph[epsilon1] (1 -γ ) √ ) Then we have

<!-- formula-not-decoded -->

Therefore, if ˜ σ differs with σ in all indices, as long as they are almost the same, | V π M -V π ˜ M | ≤ 2 glyph[epsilon1] . By adjusting the constant factor, | V π M -V π ˜ M | ≤ glyph[epsilon1] also holds.

According to Lemma 13, if each TT gets O ( glyph[epsilon1] (1 -γ ) V max ) -accurate estimation, then all the s-a transition dynamics associated with the same TT will be accurate enough to generate an glyph[epsilon1] -optimal policy. Therefore, the regular known threshold m is still the same as in RMax, i.e., m = ˜ O ( SV 2 max glyph[epsilon1] 2 (1 -γ ) 2 ) . Note that the small threshold should not exceed the regular threshold, so m s = ˜ O ( 1 ω 2 ) , where ω = max { min( τ, ν ) , O ( glyph[epsilon1] (1 -γ ) √ SV max ) } . If τ or ν is smaller than O ( glyph[epsilon1] (1 -γ ) √ SV max ) , then the small threshold becomes the regular threshold and O-TempLe degenerates to RMax.

Next, we show in Lemma 14 the total number of visits to unknown state-action pairs during T tasks.

Lemma 14. The total number of visits to unknown s-a pairs during the execution of Algorithm 1 for T tasks is

<!-- formula-not-decoded -->

Proof. For every task, Algorithm 1 first uses known threshold m s = ˜ O ( 1 ω 2 ) for all s-a pairs. And the first m s visits to an s-a pair are all visits to unknowns. So all the SA s-a pairs over T tasks take O ( TSA ω 2 ) steps of visiting unknowns in total.

Once an s-a pair is roughly known (having visits more than m s ), the TT is identified, and the known threshold is changed to m for the s-a pair. If the corresponding TT is fully known (having visits more than m ), then the s-a pair immediately becomes fully known by incorporating all visit counts of the TT . If the corresponding TT is not fully known yet, visits to the s-a pair are still counted as visits to unknown, until the TT is known. Therefore, for every possible TT , there are m unknown visits. And G TT s result in Gm unknown visits, which is the second term in Equation 13

Now we can proceed to prove the main theorem.

Proof. (of Theorem 3) We apply the PAC-MDP theorem proposed by (Strehl, Li, and Littman 2006) to get the sample complexity of O-TempLe. Proposition 1 in (Strehl, Li, and Littman 2006) claims that any greedy learning algorithm with known set K and known state-action MDP M K satisfies 3 conditions (optimism, accuracy and learning complexity) will follow a 4 glyph[epsilon1] -optimal policy on all but O ( ζ ( glyph[epsilon1],δ ) glyph[epsilon1] (1 -γ ) ln 1 δ ln 1 glyph[epsilon1] (1 -γ ) ) timesteps with probability 1 -2 δ , where ζ ( glyph[epsilon1], δ ) is the total number of updates of action-value estimates plus the number of visits to unknowns. This proposition, while it focuses on single-task learners, can be easily adapted to work for multi-task learners, as shown in (Brunskill and Li 2013).

Now we verify that the required 3 conditions all hold for our algorithm.

(1) Q t ( s, a ) ≥ Q ∗ ( s, a ) -glyph[epsilon1] for any timestep t (optimism).

This condition naturally holds as the single-task learner RMax chooses actions by optimistic value functions. O-TempLe does not change the way of choosing actions. It is similar for using E 3 or MBIE as the single-task learner. (2) V t ( s ) -V π t M Kt ( s ) ≤ glyph[epsilon1] for any timestep t (accuracy).

An s-a pair is in M K if it is fully known, i.e., n ( s, a ) ≥ m . A part of n ( s, a ) may come from the visits to other s-a pairs with the same TT . According to Lemma 13, condition (2) holds if the estimation of the TT is within O ( glyph[epsilon1] (1 -γ ) V max ) accuracy. By Hoeffding's inequality, to achieve this accuracy, m = ˜ O ( SV 2 max glyph[epsilon1] 2 (1 -γ ) 2 ) samples are required for a TT .

(3) The total number of updates of action-value estimates plus the number of visits to unknowns is bounded by ζ ( glyph[epsilon1], δ ) (learning complexity).

Lemma 14 already gives the number of visits to unknown s-a pairs, and the updates of action-value estimates will happen no more than TSA times for T tasks. Hence, ζ ( glyph[epsilon1], δ ) = ˜ O ( TSA ω 2 + SGV 2 max glyph[epsilon1] 2 (1 -γ ) 2 ) .

Therefore, the sample complexity of O-TempLe is

<!-- formula-not-decoded -->

## E.2 Proof of Theorem 4

Proof. (of Theorem 4)

The proof of Theorem 4 is similar to the proof of Theorem 3, because FM-TempLe is adapted from O-TempLe. The only difference lies in the number of visits to unknown s-a pairs.

In the first phase, FM-TempLe is the same with O-TempLe, so the number of visits to identify TT s is ˜ O ( T 1 SA ω 2 ) .

In the second phase, FM-TempLe avoids visiting all s-a pairs for at least m s times under the help of finite models. As (Brunskill and Li 2013) shows, we need at most C 2 informative s-a pairs to fully identify a model, where an s-a pair is 'informative' if at least two MDP models have sufficient disagreement in its dynamics. Similarly with Lemma 12, ˜ O ( DC 2 ω 2 ) samples are enough to let all these C 2 informative s-a pairs roughly known. Then the correct model for the current task would be identified. Thus, for every task in the second phase, ˜ O ( DC 2 ω 2 ) visits to unknowns are needed.

Finally, for each TT , its visits are shared among s-a pairs and tasks, no matter which phase they are in. Hence there are still ˜ O ( SGV 3 max ω 2 glyph[epsilon1] (1 -γ ) ) visits to unknowns.

Adding the above three parts of visits to unknowns, and following the proof of Theorem 3, we obtain the sample complexity of Theorem 4.

## F Additional Experiment Settings and Results

## F.1 Setups

Computing Infrastructure All experiments are conducted on a PC equipped with a 3.6 GHz INTEL CPU of 6 cores. Hyper-parameter Settings In Maze, an agent navigates with actions 'up', 'down', 'left' and 'right'. The reward of the goal state is set to be 1.0, and the step cost is set as 0.2.

The base learners in FMRL, our O-TempLe and our FM-TempLe are chosen to be RMax (known threshold being 500) without loss of generality. The threshold m s is set to be 50, the number of episodes 3000, and the number of in-episode steps 30. ˆ τ is set to be 0.15 for online MTRL environments, and 0.24 for Finite-Model environments. Model gap Γ for FMRL and FM-TempLe is set to be 0.6. In Finite-Model MTRL experiments, T 1 is set to be 15 for both FM-TempLe and FMRL.

The results are averaged over 20 runs. The randomization in the multiple runs comes from different task sequences generated across runs, although the comparison in each run is done on the same task sequence. We also provide the generated task sequences in our code to ensure reproducibility.

## F.2 Comparison of Per-task Reward with Confidence Intervals

Figure 5: Performance of O-TempLe and FM-TempLe compared against baselines in (a) Online MTRL, (b) Finite-Model MTRL and (c) varying sized MTRL.

<!-- image -->

## F.3 Additional Results of Baseline Methods

O-TempLe

RMax

φ

Abs-

Q

∗

d

40

·

10

0

20

Q-learning

φ

Abs-

Q

∗

glyph[epsilon1]

φ

Abs-

60

Q

∗

80

100

Tasks

(a) Abs-RL for Online MTRL

<!-- image -->

·

10

0

FM-TempLe

RMax

RMax-MaxQInit

DelayedQ-MaxQInit

10

20

30

Q-learning

FMRL

Q-MaxQInit

40

50

Tasks

(c) MaxQInit for Finite-Model MTRL

Figure 6: Additional experimental results for other versions of Abs-RL and MaxQInit.

Per-task Reward

4

.

3

.

5

5

4

5

3

4

Advantage Per-task Reward

1

.

0

.

2

5

1

5

4

## F.4 Results of Varying Action Size

Our proposed method can also work when the action space of the tasks are different, which could happen in transfer reinforcement learning (TRL) settings. For example, in a navigating task, the available actions can be simply 'up', 'down', 'left' and 'right' (as shown in Figure 7a), but a more difficult task may also allow the actions 'up-left', 'up-right', 'down-left', 'downright' (as shown in Figure 7b). Intuitively, there is some shared knowledge between these two tasks, and the agent will learn the 8-action task better if it can transfer some knowledge from the 4-action task. However, few existing methods can transfer appropriate knowledge between these two tasks without pre-defined inter-task mappings. In contrast, our proposed TempLe is able to transfer knowledge between these two tasks without any prior knowledge.

In Figure 7c, we show the performance of Q-learning, RMax and TempLe on the 8-action gridworld, where TempLe has already learned from a 4-action task and gathered the TT information. The results suggest that TempLe automatically figures out the relation among the state-action pairs in two tasks with different action spaces.

Figure 7: Transfer learning with varying action size.

<!-- image -->

## F.5 Universal Applicability of TempLe

We also observe that our proposed template learning is universally applicable to many classical stochastic environments. For example, all gridworld-based environments like FourRoom, Taxi, FrozenLake (Brockman et al. 2016), etc. In addition, Strehl and Littman (2004) propose 3 challenging MDPs as Figure 8 shows. It can be seen from these MDP definition that the number of templates is smaller than the number of state-action pairs in all of them. For instance, the TT ((1 , 0 , · · · , 0) , 0) appears for multiple times in all of the three tasks. Thus, in each of the environments, TempLe can transfer knowledge from known stateaction pairs to unknown state-action pairs with the same TT . More interestingly, since these tasks have some common TT s, if we sequentially learn these three tasks, TempLe has the potential to transfer knowledge among them, in spite that they are totally different environments in common sense.

Figure 9b Figure 9a, and Figure 9c respectively show the performance of TempLe compared with baselines on RiverSwim(Strehl and Littman 2004), FourRoom and a large GridWorld, which are well-known hard-to-explore environments. TempLe outperforms the single-task learners, because it can aggregate similar information in the environment, which saves samples and facilitates exploration.

Figure 8: Three challenging MDPs (Strehl and Littman 2004) : CasinoLand (top), RiverSwim (middle), and SixArms (bottom). Each node in the graph is a state and each edge is labeled by one or more transitions. A transition is of the form ( a, p, r ) where a is an action, p the probability that action will result in the corresponding transition, and r is the reward for taking the transition.

<!-- image -->

Figure 9: Performance of O-TempLe on challenging single-task problems, compared with RMax and Q-learning.

<!-- image -->

## F.6 Discussion: Potential Extension to Deep RL

Model-based deep RL is an important research area (Kaiser et al. 2019; Nagabandi et al. 2018), where the learner learns a dynamics model of the environment. More specifically, the learner attempts to learn a function f (usually parameterized by a neural network θ ) such that f θ ( s t ) approximates s t +1 , where s t is the current state and s t +1 is the next state. The reward function can be modeled in a similar way, while we only discuss the transition model here for simplicity.

Our proposed TempLe can be extended to large-scale MDPs and deep RL to learn the dynamics model more accurately. Below we explain the concrete method and some empirical results.

TempLe is essentially estimating the 'relative' transition among states due to the permutation operation. For example, TempLe considers the transition from s 1 to s 2 with probability 0.5 to be similar to the transition from s 7 to s 8 with probability 0.5, since the relative state difference of them is the same. This is equivalent to predicting a 'state shift' in a continuous state space, which is s t +1 -s t . In our paper, we focus on discrete state space and model the transition probabilities with discrete distributions. Similarly, in the continuous case, we can use continuous distributions (e.g. Gaussian) to approximately model the state shift, without doing state counting and ranking. Note that Nagabandi et al. (2018) also use the relative state shift in their deep RL model, whose experiments have justified the advantages of using relative state shift rather than absolute state difference. But their model is deterministic while we consider stochastic cases.

In addition to the relative state shift modeling, another key idea of TempLe is to cluster the old state-action dynamics and augment the new state-action dynamics. In the continuous case, if we assume the transition probabilities are from a mixture of Gaussian distributions, then a similar cluster and augment method can also be used. The extended algorithm works as follows:

- (1) use a neural network (NN) to predict the relative state shift: ˆ ∆ ≈ s t +1 -s t ;

(2) approximately model ∆ 's using a mixture of Gaussian (other distribution models are also applicable). From the trajectories/history, we compute ∆ = s t +1 -s t , cluster them (GEN-TT/TT-UPDATE step of TempLe) and use the averaged ¯ ∆ from each Gaussian subpopulation/cluster to improve the prediction of the NN by minimizing MSE( ˆ ∆ , ¯ ∆ );

(3) use ¯ ∆ to augment the accuracy of ˆ ∆ by identifying it into an existing cluster (AUGMENT step in TempLe). As a result, we can learn an accurate prediction model of the environment.

We implemented the above idea on the continuous environments CartPole, LunarLander and Mujoco Hopper. We use a 2-layer MLP with 64 nodes per layer. The model learning methods are summarized as below.

- Absolute. Directly predict the absolute next state.
- Relative. Predict the relative state shift (Nagabandi et al. 2018).
- Relative+augment (ours). Predict the relative state shift, and augment the model by clustering.

The results are shown in Figure 10, where we can see that our method learns the most accurate model compared with two baselines, because learning relative state shift reduces the variance and the augmentation allows knowledge transferring among state-actions with similar dynamics.

<!-- image -->

(a) CartPole

<!-- image -->

(b) LunarLander

Figure 10: Extending TempLe to deep RL.

(c) Hopper

<!-- image -->