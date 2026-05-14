## Model Learning for Look-ahead Exploration in Continuous Control

Arpit Agarwal, Katharina Muelling and Katerina Fragkiadaki

Carnegie Mellon University

United States

{ arpita1,katharam } @andrew.cmu.edu, katef@cs.cmu.edu

Abstract: We propose an exploration method that incorporates look-ahead search over basic learnt skills and their dynamics, and use it for reinforcement learning (RL) of manipulation policies . Our skills are multi-goal policies learned in isolation in simpler environments using existing multigoal RL formulations, analogous to options or macroactions. Coarse skill dynamics, i.e., the state transition caused by a (complete) skill execution, are learnt and are unrolled forward during lookahead search. Policy search benefits from temporal abstraction during exploration, though itself operates over low-level primitive actions, and thus the resulting policies does not suffer from suboptimality and inflexibility caused by coarse skill chaining. We show that the proposed exploration strategy results in effective learning of complex manipulation policies faster than current state-of-the-art RL methods, and converges to better policies than methods that use options or parametrized skills as building blocks of the policy itself, as opposed to guiding exploration. We show that the proposed exploration strategy results in effective learning of complex manipulation policies faster than current state-of-the-art RL methods, and converges to better policies than methods that use options or parameterized skills as building blocks of the policy itself, as opposed to guiding exploration.

## 1 Introduction

In animals, skill composition greatly increases the efficiency to solve new problems [1]. The skills act as the building blocks out of which an agent can form solutions to new problem configurations as well as entirely new problems [2]. Instead of creating a solution from low-level motor commands for each new challenge, skill composition enables the agent to focus on combining and adjusting higherlevel skills to achieve the goal. This principle of hierarchical skill composition has been applied by researchers in Reinforcement Learning (RL) in the hope to achieve a similar efficiency for artificial agents [3, 4]. In this context, skills are often referred to as options or macro-actions [3]. Options realize the idea of temporally extended actions that can independently accomplish a sub-goal for a defined set of scenarios. The higher-level policy is then tasked with obtaining the optimal sequence of options to accomplish the task. The performance of the policy therefore critically depends on the set of options available to the agent. If the option set is poorly chosen, the resulting composed policies will be suboptimal [5]. Many researchers have tried to find a 'golden' set of options [2] to compose hierarchical policies from [5]. However, with a growing number of options the efficiency of learning will suffer. This leads to a trade-off between flexibility and learning speed : the fewer options, the faster the learning, the less optimal the resulting composed policy. In light of the above, this paper proposes temporally abstracted look-ahead search for exploration, yet fine-grained action composition for policy representations in the manipulation domain. In other words, a small set of generalized basic manipulation policies, which we call skills or options, are learnt and their coarse transition functions are used to unroll forward a tree search during the exploration loop. Yet, policy search still operates over low-level primitive actions, and thus the resulting policies are not limited to coarse skill compositions, as previous hierarchical reinforcement learning formulations [3]. This design choice accelerates learning while at the same time permits flexibility in option (skill) selection: as long as the set of skills and the states they visit sufficiently covers the state space necessary for complex manipulation, skills can be redundant, overlapping, or varying in duration, without loss in performance of the final policy.

Figure 1: Overview . We use learned skill dynamics with deep neural regressors and use them for look-ahead tree search, to guide effective exploration in reinforcement learning of complex manipulation tasks.

<!-- image -->

In a nutshell, our framework works as follows. We train a set of simple manipulation skills, such as grasp, transfer, and reach. Each of these skills represents a (generalized) policy that can handle a set of related goals and is parametrized by both the current state and the desired goal (see Figure 1). Each skill may involve a different number of objects, thus different skills may have different state representations. For each skill we learn a coarse-grain transition function, that, given a pair of initial and goal states, it predicts the resulting state, after execution of the skill.

During training of a new manipulation task represented by a goal configuration, we perform lookahead tree search using the coarse-grain learned skill dynamics: at each time step of an episode, we unfold a search tree by sampling skill-goal combinations and using the learned neural skill dynamics to predict resulting states (Figure 1). We choose the tree path that leads closer to the desired goal configuration, and the first skill-goal of the path is executed in the real environment. We execute the chosen skill with its goal parameter until termination of that skill or episode. Then, planning is repeated from the newly reached state, akin to model predictive control [6].

Our skill-based look-ahead exploration outperforms epsilon -greedy exploration, model-based RL [7] where the fitted dynamics model is used to supply (fake) experience tuples and not for exploration, as well as policy learning over coarse parameterized skills [8], as opposed to low-level action primitives.

In summary, success of our framework depends on the following design choices, that distinguish it from previous works:

- Look-ahead with coarse-grain skill dynamics yet act with fine-grain primitives. Coarsegrain transition dynamics do not suffer from severe model error accumulation when unrolled in time [9], as few hops are sufficient to look far into the future. Yet, policies over

- fine-grained actions - as opposed to pretrained skills - produce smooth behaviour, being as fast and optimal as possible.
- Purposeful skill dynamics. Our dynamics model used for look-ahead is built from basic manipulation skills, as opposed to random exploration and transitions therein [10]. This ensures that learnt dynamics cover important part of the state space. In contrast, dynamics learnt from random exploration alone often miss useful states, and thus their temporal unrolling is less informative.

To find out more about our work visit the project page. Our code is available at github.

## 2 Related work

Exploration - Intrinsic motivation Effective exploration is a central challenge in learning good control policies [11]. Methods such as glyph[epsilon1] -greedy, that either follow the current found policy or sample a random action with a certain probability, are useful for local exploration but fail to provide impetus for the agent to explore different areas of the state space. Exploring by maximizing the agent's curiosity as measured by the error of predictive dynamics [12] expected improvement of predictive dynamics [13], information maximization [11], state visitation density [14], uncertainty of the value function estimates [15], all have been found to outperform glyph[epsilon1] -greedy, but are limited due to the underlying models operating at the level of basic actions. In manipulation, often times there are very few actions that would lead to effective new outcomes and are hard to discover from uninformative landscapes of the reward function.

Multigoal RL - Inverse dynamics The Horde architecture [16] proposed to represent procedural knowledge in terms of multiple Value Functions (VF) learned in parallel by multiple RL agents, each with its own reward and termination condition. VF would capture time for specific events to happen, or, used for inducing a policy for a specific goal. Work of [17] proposed to represent a large set of optimal VF by a single unified function approximator, parametrized by both the state and goal. This takes advantage of the fact that similar goals can be achieved with similar policies, and thus allows generalization across goals, not only across states. Hindsight Experience Replay (HER) [18] introduces the simple idea that failed executions -episodes that do not achieve the desired goalachieve some alternative goal, and is useful to book-keep in the experience buffer such 'failed' experience as successful experience for that alternative goal. Such state and goal parametrized experience is used to train a generalized policy, where actor and critic networks take as input both the current state and goal (as opposed to the state alone). Thanks to the smoothness of actor and critic networks, achieved goals that are nearby the desired one, implicitly guide learning, instead of being discarded. Our approach builds upon HER, both for learning the basic skills, as well as the final complex manipulation policies, yet we propose novel exploration strategies, instead of glyph[epsilon1] -greedy used in [18].

Generalized (multigoal) policies learn to transition for a set of initial states to a set of related goals, and as such, they are equivalent to multistep inverse dynamics models. Many works in the literature attempt to learn inverse models with random exploration [19] and chain them in time for imitation [20]. Our work learns skill inverse models using multigoal RL and explicit rewards. We are able to discover more useful and interesting temporally extended inverse dynamics models, which is unclear how to obtain with random exploration. For example, a robot would potentially never learn to grasp guided solely by random controls or curiosity, or at least, this result has not been obtained till today.

Hierarchical RL Learning and operating over different levels of temporal abstraction is a key challenge in tasks involving long-range planning. In the context of reinforcement learning, [3] proposed the options framework, which involves abstractions over the space of actions. At each step, the agent chooses either an one-step primitive action or a multi-step action policy (option). Each option defines a policy over actions (either primitive or other options) and can be terminated according to a stochastic function. The MAXQ framework [5] decomposes the value function of a Markov Desicion Process (MDP) into combinations of value functions of smaller constituent MDPs. Work of [21] learns a policy for scheduling semantically meaningful goals using deep Q networks. Work of [22] uses simple intrinsic perceptual rewards to learn subtasks and their scheduling for helping learning of extrinsic motivated (non-hierarchical) policies. [2] also explored agents with intrinsic reward structures in order to learn generic options that can apply to a wide variety of tasks.

Using a notion of salient events as sub-goals, the agent learns options to get to such events. Other works have proposed parametrized actions of discrete and continuous values as a form of macroactions (temporally extended actions) to choose from [8, 23]. We compare with the model from [8] in the experimental section. Other related work for hierarchical formulations include Feudal RL [24] which consists of 'managers' taking decisions at various levels of granularity, percolating all the way down to atomic actions made by the agent. [25] jointly learn options and hierarchical policies over them. Such joint search makes the problem more difficult to solve, moreover, options are not shared across policies of different tasks. We instead capitalize over already known options (skills) to accelerate training of more complex ones.

Hierarchical Planning Planning has been used with known transition dynamic models of the environment to help search over optimal actions to take. Incorporating macro-actions to reduce the computational cost of long-horizon plans has been explored in [26]. In [27], the authors integrate a task hierarchy into Monte Carlo Tree Search. These approaches work for discrete state, action and observation spaces, and under known dynamics. Our work instead considers continuous actions and states, and unknown dynamics.

Model-based RL To address the large sample complexity of model-free RL methods, researchers learn models of the domain, which they use to sample fake experience for policy learning [7], initialize a model-free method [28] which is further fine-tuned with real experience to fight the biases of the model, or is combined with a model-free estimation of the residual errors [29]. When the model of the domain is given, Monte Carlo tree search has shown to be very effective for exploration [30], and outperforms corresponding model-free RL methods [31], even when the latter are allowed to consume a great amount of (simulated) experience.

## 3 Exploration using look-ahead search over skill dynamics

We consider a multi-goal Markov Decision Process (MDP) [17], represented by states s ∈ S , goals g ∈ G , actions a ∈ A . At the start of each episode, a state-goal pair is sampled from the initial state-goal distribution ρ (s 0 , g) . Each goal g corresponds to a reward function r g : S × A → R . At each timestep, the agent gets as input the current state s t and goal g , and chooses an action a t according to a policy π : S ×G → A , and obtains reward r t = r g (s t , a t ) . The objective of the agent is to maximize the overall reward.

## 3.1 Learning multi-goal skill policies

We endow our agent with an initial set of manipulation skills. We define a skill as a short-horizon generalized action policy that achieves a set of related goals , as opposed to a single goal. The goal sets of individual skills are not related to the goals of our final manipulation tasks. Our framework can handle any set of skills, independent of their length, complexity, state and action spaces. Furthermore, skills can be redundant within the set. We trained three skills:

- reach , in which the gripper reaches a desired location in the workspace while not holding an object,
- grasp , in which the gripper picks up an object and holds it at a particular height,
- transfer , in which the gripper reaches a desired location in the workspace while holding an object.

The skills do not share the same state space: each involves different number of objects or it is oblivious to some part of the state space. Control is carried out in task space, by predicting directly dx, dy, dz of the motion of the end-effector and the gripper opening. Task space control allows easier transfer from simulation to a real robotic platform and is agnostic to the exact details of the robot dynamics. The skills do not share the same action space either, e.g., the reaching skill does not control the gripper open/close motion. State and action abstraction allows faster skill training. Details on the particular skill environments and corresponding states, actions and rewards functions are included in the supplementary material.

Each skill is trained using Hindsight Experience Replay [18] (HER) and off-policy deep deterministic policy gradients (DDPG) [32] with standard glyph[epsilon1] -greedy exploration [18]. This allows us to decouple exploration and policy learning. The agent maintains actor π : S × G → A and action-value (critic) Q : S × G × A → R function approximators. The actor is learned by taking gradients with respect to the loss function L a = -E s Q(s , g , π (s , g)) and the critic minimizes TD-error using TD-target y t = r t + γ Q(s t +1 , g , π (s t +1 , g)) , where γ is the reward discount factor. Similar to [18], we use a binary reward function r g which is 1 if the resulting state is within a specified radius from the goal configuration g , and 0 otherwise. Exploration is carried out by adding glyph[epsilon1] normal stochastic noise [18] to actions predicted by the current policy.

HER, alongside the intended experience tuples of the form (s t , a t , r g t , g , s t +1 ) , given the resulting state s T of an episode of length T , it adds additional experience tuples in the buffer, by considering s T to be the intended goal for the experience collected during the episode, namely, adds tuples of the form (s t , a t , r (s T ) t , s T , s t +1 ) . All the tuples in the experience buffer are use to train the actor and critic networks using the aforementioned reward functions. For more details, please refer to [18].

## 3.2 Learning coarse-grain skill dynamics and success probabilities

Multi-goal skill policies are used to obtain general and purposeful forward dynamic models, that cover rich part of the state space and which are not easy to learn from random exploration. For the i th skill, we learn

- a coarse transition function of the form T i coarse : (s , g) → s final which maps an initial state s and goal g to a resulting final state s final , and
- a success probability function u i (s , g) → [0 , 1] , that maps the current state and goal to the probability that the skill will actually achieve the goal.

We learn T coarse and u after each skill is trained. Data tuples are collected by sampling initial states and goals and running the corresponding skill policy. The collected data is used to train deep neural regressors for each skill, a three layer fully connected network that takes as input a state and a goal configurations and predicts the final state reached after skill execution, and the probability of success. The detailed architecture of the dynamics neural networks is included in the supplementary. Each manipulation skill is represented by (generalized) policy π , action-value function Q , transition dynamics and probability of success: K = { Ω i = ( π i , Q i , u i , T i coarse ) , i = 1 · · · N } , where N is the number of skills and K the skill set. For us N = 3 . Our coarse skill dynamics model learns to predict the outcome of skill execution (on average 25 timesteps long) instead of predicting the outcome of low-level actions. This allows to plan over longer horizons without severe dynamics error accumulation. Although HER with glyph[epsilon1] -greedy exploration successfully learns generalized basic skills guided solely by binary reward functions, it fails to learn more complex skills, such as, put object A inside container B , that require longer temporal horizon. Next, we describe how we use knowledge of coarse skill dynamics T coarse and success probabilities u for exploration during training of such complex manipulation policies.

## 3.3 Exploration with look-ahead search

We will use again DDPG with HER to train manipulation policies over low-level actions, but with a better exploration method. DDPG, being an off-policy algorithm, allows use to decouple learning and exploration. We exploit this decoupling and place our look-ahead search at the exploration step. With glyph[epsilon1] probability we use the proposed look-ahead search to select the next action to try a t , otherwise we follow the current policy learned thus far π (s t , g ) , where s t denotes the current state. We vary glyph[epsilon1] to be close to 1 in the beginning of training, and linearly decay it to 0.001.

Our look-ahead search works as follows: At each episode step, we unfold a search tree by sampling (with replacement) at each tree level, for each branch, a set of K skill identities (in our case one of reach , grasp or transfer ), and corresponding K skill sub-goals, where K is the branching factor of the search. For each sampled skill identity and sub-goal, we use the learned success probabilities of each sampled skill and sub-goal combination and prune improbable transitions (line 13 in Algorithm 2). For the remaining skill/subgoal combinations we (mentally) transition to the resulting final state following the learned skill dynamics function. After unfolding the tree for a prespecified number of steps, we choose the path with the maximum total reward defined as the sum of the transition rewards(reward for going from one node to other connnected node) of the skill executions as measured by the skill critic networks Q i and proximity of the final state to the desired goal configu- ration: R = ∑ ( s,k ∈K ,g k ) ∈ path to leaf node Q k ( s, g k ) + rfi nal, where rfi nal is the negative of the Euclidean distance between the final state and the desired goal configuration g . We execute the first (skill, subgoal) tuple on the chosen maximum utility path in the current ('real') environment until the skill terminates, i.e., until the skill sub-goal is achieved or maximum skill episode length is reached or goal is reached. The experience in terms of tuples (s t , g , a t , r t , s t +1 ) populate the experience replay buffer.

Figure 2: Look-ahead Search: From the current environment state, we sample skill identities and skill sub-goals and (mentally) unfold a look-ahead tree using learned skill dynamics. We select the first skill and sub-goal of the path with maximum utility and execute it in the 'real world'.

<!-- image -->

Note that the learned skill dynamics T coarse and u may not match the dynamics in the current environment. The reason could be both due to approximation error of the neural network regressors and the difference in the environment dynamics, e.g., task environments may contain additional objects on which the gripper can collide, which were not present during skill learning. Our lookahead exploration is described in Algorithm 2 and visualized in Figure 2. The complete exploration and reinforcement learning method is described in Algorithm 1.

## 4 Experiments

We test the proposed method in the MuJoCo simulation environment [33] using a seven degree of freedom Baxter robot arm with parallel jaw grippers in the following suite of manipulation tasks:

- Pick and Move (Figure 3a) The robot needs to reach towards the object, grasp it and move to a target 3D location.
- Put A inside B (Figure 3b) The robot needs to reach towards the object, grasp it and put it inside the container.
- Stack A on top of B (Figure 3c) The robot needs to reach towards the red object, grasp it and put it on top of the purple object.

## Algorithm 1 HER with look-ahead search exploration(HERLASE)

```
1: Input: 2: skill set K 3: reward function r : -✶ [ f g ( s ) = 0] 4: glyph[epsilon1] ← 1 , skill terminated ← true 5: Initialize π, Q , Replay buffer B 6: for episode = 1, M do 7: Sample a goal g and starting state s 0 8: while episode not done do 9: if random(0,1) < glyph[epsilon1] then 10: if skill terminated then 11: Ω i , g i ← TreeSearch( s t , g ) 12: ( π i , Q i , u i , T i coarse ) ← Ω i 13: end if 14: a t = π i ( s t , g i ) 15: else 16: a t = π ( s t , g ) + Gaussian noise 17: end if 18: s t +1 , r t , terminal = execution( a t ) 19: skill terminated ← checkSkillTermination( s t +1 ) 20: end while 21: Create hindsight experience with g ′ = s T 22: end for
```

- Take A out of B (Figure 3d) The robot needs to reach towards the object inside the container, grasp it and take it out of the container. The objective of this environment to grasp the objective in cluster container, then move out of container and move it to any 3D location in the workspace.

All the tasks in our benchmark suite require long temporal horizon policies, which is a challenge when learning from sparse rewards.

1. How well the proposed exploration strategy performs over glyph[epsilon1] -greedy?
2. How out learnt policies compare with policies assembled directly over macro-actions or options?
3. What is the impact of the chosen skill set to the proposed method?
4. What happens when the dynamics are quite different in the new environment?

We evaluate our method against the following baselines:

1. Hindsight Experience Replay ( HER ), described in [18]. In this method, exploration is carried out by sampling noise from an glyph[epsilon1] -normal distribution and adding it to the predicted actions of the policy learned so far. This is a state-of-the-art model-free RL method for control.
2. Parameterized action space ( PAS ), described in [8]. This approach uses the learned skills as macro-actions by learning a meta-controller which predicts probability over those skills, as well as the goal of each skill.
3. Parameterized action space + Hierarchical HER ( HER-PAS ), as described in [34]. We extend the model of [8] by creating additional hindsight experience at macro-level, i.e., in the parameterized action level (skill and sub-goal). Specifically, we replace the goal state in the macro-action transitions collected during the episode which failed to achieve goal g with the g ′ = s T and evaluate all the low level actions, chosen using the skill and sub-goal, with the new reward function r t = r g new ( s t , a t ) .
4. Parameter Space Noise + HER( HER+ParamNoise ), described in [35], exploration is carried out in parameter space similar to evolutionary strategies.

## Algorithm 2 TreeSearch

```
1: Input: 2: maxHeight H, branchingFactor B 3: goal g , initial state s real t , skill set K 4: Initialize 5: root ← ( s real t ,0, 0) 6: openlist ← addRoot 7: leafNodelist ←{} 8: while all path explored do 9: s , currHeight = getTopLeafNode(openlist) 10: sampled Set = sample skills and goal parameters(B) 11: for Ω i , g i ∈ sampled Set do 12: ( π i , Q i , u i , T i coarse ) ← Ω i 13: if u i ( s, g i ) > 0 . 5 then 14: nextState ←T i coarse ( s, g i ) 15: a i ← π i ( s, g i ) 16: transitionReward ← Q i ( s, g i , a i ) 17: if currHeight+1 < H then 18: AddToLeafNodelist(nextState, transitionReward, currHeight+1) 19: else 20: AddNodeToOpenlist(nextState, transitionReward, currHeight+1) 21: end if 22: end if 23: end for 24: end while 25: bestPath = getBestPath(leafNodeList) 26: Return first skill and goal parameter of bestPath
```

Success plots for the different tasks for our method and baselines are shown in Figures 4. Our method significantly outperforms the baselines in terms of final performance and sample complexity of environment interactions. In 'put A inside B' task 3b HER is not able to succeed at all. PAS show success early in training but converges to sub-optimal performance. This is expected due to the restriction imposed on the policy space by to the hierarchical structure.

Sensitivity to skill set selection We use skills (macro-actions) for exploration, and not for assembling a hierarchical policy, as done in the option framework [3]. Therefore, we expect our method to be less sensitive to the skill selection, in comparison to previous methods [3]. To quantify such flexibility we conducted experiments using three different skill sets: faster convergence. To test this hypothesis, we used 3 skill set K 1 = { transit, grasp, transfer } , K 2 = { grasp, transfer } and K 3 = { transit, transfer } . We tested our method with K 2 on the 'Pick and Move', and 'Put A inside B' tasks and show results in Figure 5a. Exploring with all the skills ( K 1 ) leads to faster convergence of policy learning than using K 2 . This is easily explained as the agent needs to learn the transit skill via interactions generated by other skills and the current policy. We did not observe any slower learning for 'Put A inside B' task using K 2 , as shown in Figure 5b. The reason for this is that the transit skill is similar to the transfer skill, which is present. Learning of 'Put A inside B' task using K 3 is slower (Figure 5b). This is due to the fact that grasping is a critical skill in completing this task. However our method is still able to learn the task from scratch, while HER fails to do so.

Sensitivity to model errors We quantify sensitivity of our method to model errors, i.e., the accuracy of dynamics skill models. We created a perturbation in our 'Put A inside B' task by making a wall insurmountable by the robot arm, shown in Figure 6a. The skill dynamics that concern endeffector states near the large wall would be completely wrong, since this wall was absent during learning of our bacis skills and therefore their dynamics. However, the proposed look-ahead exploration is still beneficial, as seen in Figure 6b. Our method succeed to learn task, while HER with

<!-- image -->

(c) Stack A on top of B

(d) Take A out of B

Figure 3: Suite of robot manipulation tasks with baxter robot with end-effector control and parallel jaw gripper.

random exploration fails to do so. We used our full skill set in this experiment, namely, transit, transfer and grasp skills.

Scalability Our method trades time of interacting with the environment with time to 'think' the next move by mentally unfolding a learned model of dynamics. The 'thinking' time depends on the speed of the look-ahead search. We experimented with different branching factors for the tree search.

With glyph[epsilon1] -greedy the agent takes 0.4 seconds per episode (50 steps), with branching factor (bf) equal to 5 the agent takes 17 seconds, with bf=10 it takes 71 seconds, and with bf=15 it takes 286 seconds. However, for complex tasks in our benchmark suite, we did not observe empirical advantages from larger branching factors, so all the reported results use bf=5. The sampling process and pruning could be implemented in parallel on GPU, which will render our tree search much more efficient. We are currently exploring learning-based ways to guide the tree unfolding.

Figure 4: The success plot for each manipulation task in our suite. For evaluation, we freeze the current policy and sample 20 random initial states and goals at each epoch (1 epoch = 16 episodes of environment interaction).

<!-- image -->

Figure 5: Sensitivity of our method to skill set selection. We show the success plots for 'Pick and move' and 'Put A inside B' tasks. Look-ahead exploration leads to slower learning when essential skills are missing. However, it is still better than random exploration.(1 epoch = 16 episodes of environment interaction).

<!-- image -->

(a) Put A inside B with 1 insurmountable wall

<!-- image -->

(b) Success Curve

<!-- image -->

Figure 6: Sensitivity of our method to model errors. In (A) we show the perturbed version of 'Put A inside B' task in which a wall is very high and cannot be crossed by the agent. In (B) we show that even with wrong coarse dynamics model our approach works better than random exploration. (1 epoch = 16 episodes of environment interaction).

## 5 Conclusion - Future work

We proposed an exploration method that uses coarse-time dynamics of basic manipulation skills for effective look-ahead exploration during learning of manipulation policies. Our empirical findings suggest that the proposed look-ahead exploration guided by learned dynamics of already mastered skills, can effectively reduce sample complexity when mastering a new task, to the extent that skills that are impossible with naive exploration, are possible with the proposed dynamics-based lookahead exploration, suggesting an avenue for curriculum learning of manipulation policies, by continually expanding the skill set. The resulting policies are still in the space of primitive low-level actions, which allows flexibility on choosing skills, and on the resulting reactive policy learner.

If the proposed exploration strategies are used in environments that have different dynamics than the environments used to train the basic skills (e.g., there is a wall present, or the object sizes are very different), learning is slower, and our exploration scheme offers less advantages due to model errors. The fact that our dynamics are not updated online is a limitation of our method, and an avenue for future work. All the experiments performed in our paper are based on the centroid estimates of the objects of interest. It would be interesting to explore policies and models learned directly from visual features in future work.

## 6 Acknowledgment

The authors would like to thank Chris Atkeson for useful discussions on learning coarse dynamics models. This work was conducted in part through collaborative participation in the Robotics Consortium sponsored by the U.S Army Research Laboratory under the Collaborative Technology Alliance Program, Cooperative Agreement W911NF-10-2-0016. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Laboratory of the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes not withstanding any copyright notation herein.

## References

- [1] R. W. White. Motivation reconsidered: The concept of competence. Psychological review , 66 (5):297, 1959.

- [2] N. Chentanez, A. G. Barto, and S. P. Singh. Intrinsically motivated reinforcement learning. In L. K. Saul, Y. Weiss, and L. Bottou, editors, Advances in Neural Information Processing Systems 17 , pages 1281-1288. MIT Press, 2005. URL http://papers . nips . cc/paper/ 2552-intrinsically-motivated-reinforcement-learning . pdf .
- [3] R. S. Sutton, D. Precup, and S. Singh. Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning. Artificial Intelligence , 112(1):181 - 211, 1999. ISSN 0004-3702. doi:https://doi . org/10 . 1016/S0004-3702(99)00052-1. URL http: //www . sciencedirect . com/science/article/pii/S0004370299000521 .
- [4] T. D. Kulkarni, W. Whitney, P. Kohli, and J. B. Tenenbaum. Deep convolutional inverse graphics network. CoRR , abs/1503.03167, 2015. URL http://arxiv . org/abs/1503 . 03167 .
- [5] T. G. Dietterich. Hierarchical reinforcement learning with the MAXQ value function decomposition. CoRR , cs.LG/9905014, 1999. URL http://arxiv . org/abs/cs . LG/9905014 .
- [6] D. Q. Mayne. Model predictive control: Recent developments and future promise. Automatica , 50(12):2967 -2986, 2014. ISSN 0005-1098. doi:http://dx . doi . org/10 . 1016/ j . automatica . 2014 . 10 . 128. URL http://www . sciencedirect . com/science/article/ pii/S0005109814005160 .
- [7] R. S. Sutton. Dyna, an integrated architecture for learning, planning, and reacting. SIGART Bull. , 2(4):160-163, July 1991. ISSN 0163-5719. doi:10 . 1145/122344 . 122377. URL http: //doi . acm . org/10 . 1145/122344 . 122377 .
- [8] M. Hausknecht and P. Stone. Deep reinforcement learning in parameterized action space. In Proceedings of the International Conference on Learning Representations (ICLR) , May 2016.
- [9] J. Oh, X. Guo, H. Lee, R. L. Lewis, and S. P. Singh. Action-conditional video prediction using deep networks in atari games. CoRR , abs/1507.08750, 2015. URL http://arxiv . org/abs/ 1507 . 08750 .
- [10] D. Ha and J. Schmidhuber. World models. CoRR , abs/1803.10122, 2018. URL http:// arxiv . org/abs/1803 . 10122 .
- [11] S. Mohamed and D. J. Rezende. Variational information maximisation for intrinsically motivated reinforcement learning. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28 , pages 2125-2133. Curran Associates, Inc., 2015. URL http://papers . nips . cc/paper/5668-variational-information-maximisationfor-intrinsically-motivated-reinforcement-learning . pdf .
- [12] D. Pathak, P. Agrawal, A. A. Efros, and T. Darrell. Curiosity-driven exploration by selfsupervised prediction. CoRR , abs/1705.05363, 2017. URL http://arxiv . org/abs/ 1705 . 05363 .
- [13] J. Schmidhuber. A possibility for implementing curiosity and boredom in model-building neural controllers. In Proceedings of the First International Conference on Simulation of Adaptive Behavior on From Animals to Animats , pages 222-227, Cambridge, MA, USA, 1990. MIT Press. ISBN 0-262-63138-5. URL http://dl . acm . org/citation . cfm?id= 116517 . 116542 .
- [14] M. G. Bellemare, S. Srinivasan, G. Ostrovski, T. Schaul, D. Saxton, and R. Munos. Unifying count-based exploration and intrinsic motivation. CoRR , abs/1606.01868, 2016. URL http: //arxiv . org/abs/1606 . 01868 .
- [15] I. Osband, C. Blundell, A. Pritzel, and B. V. Roy. Deep exploration via bootstrapped DQN. CoRR , abs/1602.04621, 2016. URL http://arxiv . org/abs/1602 . 04621 .
- [16] R. S. Sutton, J. Modayil, M. Delp, T. Degris, P. M. Pilarski, A. White, and D. Precup. Horde: A scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction. In The 10th International Conference on Autonomous Agents and Multiagent Systems Volume 2 , AAMAS '11, pages 761-768, Richland, SC, 2011. International Foundation for Autonomous Agents and Multiagent Systems. ISBN 0-9826571-6-1, 978-0-9826571-6-4. URL http://dl . acm . org/citation . cfm?id=2031678 . 2031726 .

- [17] T. Schaul, D. Horgan, K. Gregor, and D. Silver. Universal value function approximators. In F. Bach and D. Blei, editors, Proceedings of the 32nd International Conference on Machine Learning , volume 37 of Proceedings of Machine Learning Research , pages 13121320, Lille, France, 07-09 Jul 2015. PMLR. URL http://proceedings . mlr . press/v37/ schaul15 . html .
- [18] M. Andrychowicz, D. Crow, A. Ray, J. Schneider, R. Fong, P. Welinder, B. McGrew, J. Tobin, O. P. Abbeel, and W. Zaremba. Hindsight experience replay. In Advances in Neural Information Processing Systems , pages 5055-5065, 2017.
- [19] P. Agrawal, A. Nair, P. Abbeel, J. Malik, and S. Levine. Learning to poke by poking: Experiential learning of intuitive physics. CoRR , abs/1606.07419, 2016. URL http://arxiv . org/ abs/1606 . 07419 .
- [20] A. Nair, D. Chen, P. Agrawal, P. Isola, P. Abbeel, J. Malik, and S. Levine. Combining selfsupervised learning and imitation for vision-based rope manipulation. CoRR , abs/1703.02018, 2017. URL http://arxiv . org/abs/1703 . 02018 .
- [21] T. D. Kulkarni, K. Narasimhan, A. Saeedi, and J. B. Tenenbaum. Hierarchical deep reinforcement learning: Integrating temporal abstraction and intrinsic motivation. CoRR , abs/1604.06057, 2016. URL http://arxiv . org/abs/1604 . 06057 .
- [22] M. A. Riedmiller, R. Hafner, T. Lampe, M. Neunert, J. Degrave, T. V. de Wiele, V. Mnih, N. Heess, and J. T. Springenberg. Learning by playing - solving sparse reward tasks from scratch. CoRR , abs/1802.10567, 2018. URL http://arxiv . org/abs/1802 . 10567 .
- [23] W. Masson and G. Konidaris. Reinforcement learning with parameterized actions. CoRR , abs/1509.01644, 2015. URL http://arxiv . org/abs/1509 . 01644 .
- [24] P. Dayan and G. E. Hinton. Feudal reinforcement learning. In S. J. Hanson, J. D. Cowan, and C. L. Giles, editors, Advances in Neural Information Processing Systems 5 , pages 271-278. Morgan-Kaufmann, 1993. URL http://papers . nips . cc/paper/714-feudalreinforcement-learning . pdf .
- [25] C. Daniel, H. Van Hoof, J. Peters, and G. Neumann. Probabilistic inference for determining options in reinforcement learning. Machine Learning , 104(2-3):337-357, 2016.
- [26] R. He, E. Brunskill, and N. Roy. Puma: Planning under uncertainty with macro-actions. In AAAI , 2010.
- [27] N. A. Vien and M. Toussaint. Hierarchical monte-carlo planning. In AAAI , pages 3613-3619, 2015.
- [28] A. Nagabandi, G. Kahn, R. S. Fearing, and S. Levine. Neural network dynamics for modelbased deep reinforcement learning with model-free fine-tuning. CoRR , abs/1708.02596, 2017. URL http://arxiv . org/abs/1708 . 02596 .
- [29] Y. Chebotar, K. Hausman, M. Zhang, G. S. Sukhatme, S. Schaal, and S. Levine. Combining model-based and model-free updates for trajectory-centric reinforcement learning. CoRR , abs/1703.03078, 2017. URL http://arxiv . org/abs/1703 . 03078 .
- [30] D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hubert, L. R. Baker, M. Lai, A. Bolton, Y. Chen, T. P. Lillicrap, F. X. Hui, L. Sifre, G. van den Driessche, T. Graepel, and D. Hassabis. Mastering the game of go without human knowledge. Nature , 550:354-359, 2017.
- [31] X. Guo, S. Singh, H. Lee, R. L. Lewis, and X. Wang. Deep learning for real-time atari game play using offline monte-carlo tree search planning. In Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 27 , pages 3338-3346. Curran Associates, Inc., 2014. URL http://papers . nips . cc/paper/5421-deep-learning-for-real-time-atarigame-play-using-offline-monte-carlo-tree-search-planning . pdf .

- [32] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y . Tassa, D. Silver, and D. Wierstra. Continuous control with deep reinforcement learning. CoRR , abs/1509.02971, 2015. URL http://arxiv . org/abs/1509 . 02971 .
- [33] E. Todorov, T. Erez, and Y. Tassa. Mujoco: A physics engine for model-based control. In Intelligent Robots and Systems (IROS), 2012 IEEE/RSJ International Conference on , pages 5026-5033. IEEE, 2012.
- [34] A. Levy, R. Platt, and K. Saenko. Hierarchical actor-critic. arXiv preprint arXiv:1712.00948 , 2017.
- [35] M. Plappert, R. Houthooft, P. Dhariwal, S. Sidor, R. Y. Chen, X. Chen, T. Asfour, P. Abbeel, and M. Andrychowicz. Parameter space noise for exploration. arXiv preprint arXiv:1706.01905 , 2017.
- [36] M. Plappert, M. Andrychowicz, A. Ray, B. McGrew, B. Baker, G. Powell, J. Schneider, J. Tobin, M. Chociej, P. Welinder, et al. Multi-goal reinforcement learning: Challenging robotics environments and request for research. arXiv preprint arXiv:1802.09464 , 2018.
- [37] P. Dhariwal, C. Hesse, O. Klimov, A. Nichol, M. Plappert, A. Radford, J. Schulman, S. Sidor, and Y. Wu. Openai baselines. https://github . com/openai/baselines , 2017.

## A Environment Details

The environments are similar to multi-goal environments proposed in [36]. We made this environment to make the results comparable with other works who use more famous OpenAI environments. Another reason for choosing baxter as our robot is due to its availability of platform in our lab for performing transfer to real world experiments which is left as a future work.

In all the environment, actions are 4-dimensional with first 3-dimension as the cartesian motion of the end-effector and last dimension controls the opening and closing of the gripper. We apply the same action 75 simulation steps(with ∆ t = 0 . 002 ). The state space includes the cartesian position of the end-effector, its velocities, position and velocity of gripper, objects pose and its velocities.

We use a single starting state in which object is grasped. The state space includes the relative position of object which has to be transported in then environment. The goal location is given as a 3D cartesian location where the object has to be transported to.

## A.1 Pick and move object to a target location in 3D

This environment is a close replica of [36] FetchPickAndPlace environment made for baxter robot. The objective to move the object from arbitrary location in the workspace to the 3D target location in the workspace. The starting location of the gripper and object are sampled randomly in the workspace of the robot. The reward function is binary, i.e. the agent obtains reward 0 when the object is within the tolerance 3cm and -1 otherwise.

## A.2 Put object A inside container B

The objective is to go, grab the object from any arbitrary location in the workspace and move it inside the container. The goal is specified as the center of the container which is kept fixed across all the episodes. The binary reward function gives the agent 0 if the object is within 5cm of the target location and z of the object is less 2cm from the ground and -1 otherwise.

## A.3 Put object A on object B

The objective is to go, grab the object from random location and put it over another object B, whose location is fixed. The goal is specified as the center of the top surface of the object B. The binary reward function gives the agent 0 if the object is within 3cm of the target location and object A is in contact of object B and -1 otherwise.

## A.4 Take object A out of container B

The objective is to take reach to the object(random starting location inside the container), grab it and move it out of the container B. The goal is specified as a 3D location(randomly sampled) in the workspace of the robot. The agent gets reward 0 if the object is within 3cm of the target location and -1 otherwise.

## B Policy learning for Skill Details

We learnt policies for skills using Deep Deterministic Policy library with hindsight experience replay[18].

## B.1 Skills description

We trained 3 skills namely transit, grasping and transfer object. The state space of transit is gripper location and goal is 3D target location for transit skill and action space is 3D movement of the end-effector. The state space and action space of grasping and transfer skill is same as PicknMove task. The starting configuration for grasping skill is gripper touching the object and goal is to raise object above ground. The starting configuration for transfer skill is object grasped in gripper and goal location is sampled randomly in the empty workspace of the robot.

## C Successor Regresson model Details

The coarse level successor predictor model is learnt for each skill individually after training each skill independently.

## C.1 Data collection

After training each skill, we sample starting configurations in the skill environments and let the trained skill act in its environment. We store the starting configuration and the resulting state to act as target of the successor prediction model. For training the success predictor we sampled starting configuration from PicknMove environment.

## C.2 Model architecture

For success probability prediction, we used fully connected network with 2 hidden layers of 50 and 100 neurons each. We trained the model with learning rate = 1e-3, binary cross-entropy and ADAM optimizer.

We used fully connected network with 3 hidden layers of 1000 neurons each for representing the successor prediction model. We trained the model with learning rate = 1e-5, l2 regression loss and ADAM optimizer.

## D Implementation Details

For representing actor and critic we used a neural network with 3 fully connected layers with 64 neurons each. For our methods and baselines we trained DDPG and build over [37]. The other hyperparameters are as follows: Actor Learning rate: 1e-3

Critic Learning rate: 1e-4

Target network update: gamma: 0.98

Number of cycles per epoch: 20

Number of updates per cycle: 40

Batch size: 128