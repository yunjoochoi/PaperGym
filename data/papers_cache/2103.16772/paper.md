## Causal Reasoning in Simulation for Structure and Transfer Learning of Robot Manipulation Policies

Tabitha Edith Lee, Jialiang (Alan) Zhao, Amrita S. Sawhney, Siddharth Girdhar, and Oliver Kroemer

Abstract -We present CREST, an approach for causal reasoning in simulation to learn the relevant state space for a robot manipulation policy. Our approach conducts interventions using internal models, which are simulations with approximate dynamics and simplified assumptions. These interventions elicit the structure between the state and action spaces, enabling construction of neural network policies with only relevant states as input. These policies are pretrained using the internal model with domain randomization over the relevant states. The policy network weights are then transferred to the target domain (e.g., the real world) for fine tuning. We perform extensive policy transfer experiments in simulation for two representative manipulation tasks: block stacking and crate opening. Our policies are shown to be more robust to domain shifts, more sample efficient to learn, and scale to more complex settings with larger state spaces. We also show improved zero-shot simto-real transfer of our policies for the block stacking task.

## I. INTRODUCTION

Real-world environments, such as homes and restaurants, often contain a large number of objects that a robot can manipulate. However, usually only a small set of objects and state variables are actually relevant for performing a given manipulation task. The capability of reasoning about what aspects of the state space are relevant for the task would lead to more efficient learning and greater skill versatility.

Current approaches to learning versatile manipulation skills often utilize simulation-to-real (sim-to-real) transfer learning [1]-[4], wherein the skill is learned in a simulation and then deployed and fine-tuned (if feasible) on the real robot. Sim-to-real learning is often combined with domain randomization (DR) [5], which involves training the skill on a wide range of task instances in simulation such that the resulting skill is more robust and generalizes across task variations. However, for scenes with distractor objects, the policy still takes the irrelevant state features as inputs. Larger domain shifts in the irrelevant features can therefore still be detrimental to the performance of the skill policy. Rather than relying only on DR, the robot can use modelbased reasoning to identify the 'structure' of a policy - the interplay between relevant state inputs and control outputs.

In this paper, we propose using causal reasoning to improve sim-to-real transfer through conducting interventions in simulation to determine which state variables are relevant for the successful execution of a task using a given controller. We refer to the resulting algorithm as CREST : C ausal R easoning for E fficient S tructure T ransfer. The relevant state

All authors are affiliated with the Robotics Institute, Carnegie Mellon University, Pittsburgh PA 15123, USA. tabithalee@cmu.edu Supplementary materials: https://sites.google.com/view/ crest-causal-struct-xfer-manip variables from CREST are used to construct policies that encode the causal structure of the manipulation task. These policies are initially trained in simulation using domain randomization over only the states that have explicitly been determined as relevant. Moreover, policies that only use relevant state variables require significantly fewer parameters and, by construction, are robust to distribution shifts in irrelevant state spaces. In this manner, our approach produces lightweight policies that are designed for efficient online adaptation to unforeseen distribution shifts that may occur when bridging the sim-to-real gap. This contrasts with existing sim-to-real approaches that train large policies over enormous state spaces, where the costs for achieving robust zero-shot transfer may be intractable.

Our proposed approach was successfully evaluated on block stacking and crate opening tasks. Although our method is intended for sim-to-real transfer, we primarily conduct our experiments by transferring to task simulations in NVIDIA Isaac Gym [6], a high-fidelity physics simulator, as a proxy for real systems. This is necessary to experimentally evaluate distributions shifts that would be intractable to evaluate (but nonetheless feasible) in practical manipulation scenarios. Additionally, we validate our approach for zero-shot, simto-real performance for the block stacking task.

The contributions of our work are as follows.

- We propose CREST, an algorithm that uses causal reasoning in simulation to identify the relevant input state variables for generalizing manipulation policies.
- We propose two neural network architectures that are constructed using the causal information from CREST.
- We conduct rigorous transfer learning experiments to demonstrate these policies generalize across task scenarios, scale in relevant state complexity, and are robust to distribution shifts.
- We propose CREST as one approach to a broader methodology of structure-based transfer learning from simulation as a new paradigm for sim-to-real robot learning, i.e., structural sim-to-real.

## II. RELATED WORKS

Our work relates to research areas in robotics and machine learning for structure-based learning, causality, attention, and sim-to-real transfer of higher-level policies.

Structure-based learning, causality, and attention. Causal reasoning for structure, transfer, and reinforcement learning is an emerging area of research [7], [8], having been demonstrated for transferring multi-armed bandits policies [9], examining distribution shifts via imitation learn- ing [10], modeling physical interactions from videos [11], and clustering causal factors [12]. Causality reasons about the data generation process with respect to an underlying model [13], which our CREST policies encode. The motivation of our approach to transfer learned causal structure is similar to that described by [14]. In our work, we represent this structure through the policy inputs, and we demonstrate the approach achieves sim-to-real transfer.

Our view of policy structure can be seen as an explicitly encoded form of state space attention, achieved via construction of policies using only relevant inputs. We are primarily concerned with object feature states, which enable significantly smaller neural networks to be constructed under our assumptions. As a comparison, [15] learns implicit attention to task-relevant objects to generalize manipulation skills. This approach requires a few example trajectories to be provided and uses a vision-based state representation, making explicit reasoning about states more challenging.

Similar to [16], our approach can also generalize policies to unforeseen dynamic distribution shifts; ours does so primarily through more efficient fine-tuning.

Our work is also similar in spirit to the work of Nouri and Littman [17] in terms of achieving dimensionality reduction for reinforcement learning. Whereas our work seeks to reduce the dimensionality for the state's possible influence on the task policy under different contexts, they instead demonstrate dimensionality reduction in the action space. Kolter and Ng [18] as well as Parr et al. [19] approximate the value function by learning the relevant basis functions.

Sim-to-real and higher-level policies. Our work is a form of simulation-to-reality transfer of controllers [1], [5], [20], [21]. Unlike in the typical sim-to-real paradigm, our policies are not required to transfer zero-shot. Rather, the dimensionality reduction afforded by transferring the relevant state space with CREST enables more efficient online adaptation. Unlike [22], we do not transfer any target samples back to the internal model. Similar to the problem settings of [23]-[25], our agent predicts the best parameters for a controller.

## III. PROBLEM FORMULATION

We formulate our problem as a multi-task reinforcement learning problem, wherein a policy π is learned to complete a series of tasks T i . Each task is modeled as a Markov decision process (MDP). The state space S and action space A are the same across tasks. However, each task T i defines a separate initial state s 0 ( c i ) , transition function p ( s ′ | s, a, c i ) , and reward function r ( a, s, c i ) that are parameterized by the task's context variables c i ∈ C . We assume that the robot has an internal model p int ( s ′ | s, a, c i ) that approximates the transition function of the target task domain p ( s ′ | s, a, c i ) .The context variables capture object parameter variations, such as shapes, appearances, and initial states. We assume that the context variables are always set such that the task is feasible.

To solve the tasks, the robot learns a policy π ( a | s, c ) that is decomposed into two parts [26]: an upper policy π ( θ | c ) and a lower control policy π θ ( a | s ) , where θ ∈ R d are the controller parameters. Transferring the parameters from the internal model to the target domain may require fine-tuning. At the start of each task, the upper policy, which is responsible for generalizing between different task contexts, selects a set of parameters for the control policy to use throughout the task execution. We use multilayer perceptron (MLP) neural networks for the upper policy. In principle, the control policies can take on a variety of parameterized forms, such as motor primitives, planners, waypoint trajectories, or linear feedback controllers, depending on the task. We assume a control policy with known preconditions is available.

Of the context variables c that describe the object variations in the scene, only a subset may be relevant for the policy. We refer to this subset of relevant variables as τ ⊆ c , such that the robot can learn a policy π ( θ | τ ) to complete the tasks. Our goal is to determine τ through causal reasoning with the internal model, yielding policies with fewer input variables as compared to naively using all of c .

The set of controller parameters can be divided into individual parameters [ θ ] j ∀ j ∈ 1 , ..., d , where [ θ ] j indicates the j th element of vector θ . Each of these parameters may rely on a different set of relevant variables. We can thus further divide the problem into determining a set of context variables τ j for each policy parameter [ θ ] j , such that the robot can learn a partitioned task policy π ([ θ ] j | τ j ) ∀ j ∈ 1 , ..., d .

## IV. CAUSAL STRUCTURE LEARNING

CREST uses an internal model of the task to learn the relevant context τ and parameter-specific mappings τ j that define the structure of the upper policy (c.f, Sec. V-A).

## A. Internal Model for Causal Reasoning

Our approach assumes an internal model , an approximate simulation of the task, is available. Analogous to mental models and approximate physics models [27], [28], the internal model facilitates reasoning about the effects of different context variations ∆ c on completing the task with policy parameters θ . Varying the relevant context parameters τ will affect the execution and outcome of the task in the internal model while irrelevant ones will not.

Given its approximate nature, the solution obtained in the internal model is not necessarily expected to transfer zero-shot to reality. Instead, the internal model provides both an estimate of the policy structure and a task-specific initialization via network pretraining. Intuitively, it is easier to reason about which variables are important for a model rather than exactly characterizing the exact model itself. For example, the internal model can capture that the weight of an object affects the required pushing force, but the exact details of frictional interaction may be approximated for the purposes of pretraining the policy. We assume that the internal model approximates the task sufficiently well and includes all context variables c that vary in the target domain. Given the existing challenges in causal representation learning [29], we additionally assume the representation of c is amenable for determining the underlying model for θ via causal interventions [13].

## B. Causal Reasoning to Determine Relevant Contexts

At its core, CREST uses simulation-based causal reasoning to determine the relevant context variables for the policy input. This process is divided into two phases: 1) determining the overall set of relevant variables τ , and 2) determining the relevant variables τ j for specific policy parameters [ θ ] j .

## Causal interventions to determine relevant variable set.

In this phase, the relevance of a state variable is determined by posing the following question: 'If a context variable were different, would the same policy execution still complete the task successfully?' To answer this, we first uniformly sample a context c i ∼ p ( c ) and solve the resulting task in simulation to acquire the corresponding policy parameters θ i . In practice, the policy parameters are optimized using Relative Entropy Policy Search [30]. Importantly, the policy is solved for only this specific context c i (not the general policy).

Given the solved task, we conduct interventions ∆ c to determine if the policy parameters θ i remain valid for the new context c i + ∆ c . Interventions ∆ c are conducted to only alter one context variable at a time, i.e. || ∆ c || 0 = 1 . If the policy subsequently failed, the intervened variable is considered relevant and thus appended to τ . If the policy succeeded, despite the intervention, then the variable is considered irrelevant and is not required for the general policy. The resulting relevant variable set τ is sufficient for constructing Reduced MLP policies (c.f., Sec. V-A).

Causal interventions to determine individual policy mappings. Reducing the set of state variables from the full set c to the relevant set τ may greatly reduce the size of the policy input. However, for some problems, each of the policy parameters may only depend on a subset of τ . Therefore, in the next phase, the individual mappings from the relevant state variables to the controller parameters are determined by posing the question: 'Does altering this relevant context variable require this policy parameter to be changed?'

We begin from the previous phase, where c i with solution θ i is available with interventions ∆ c . In this phase, interventions are only applied to relevant context variables τ .

For each new context c i +∆ c , the task is solved to obtain the resulting policy parameters θ i + ∆ θ , starting the optimization from the original parameters θ i . The optimization will often alter all of the policy parameters, i.e., all elements of ∆ θ are non-zero, and the magnitude of their changes are not reliable estimators of their importance. Instead, a solution that minimizes the number of non-zero changes, i.e., min || ∆ θ || 0 , is obtained using search. This search involves setting subsets of elements in ∆ θ to zero and evaluating if the resulting policy still solves the task with context c i +∆ c . In our experiments, we used a breadth-first search to find a solution with a minimal set of parameter changes. Once a subset of parameter changes has been found, the context variable intervened on in ∆ c is added to the sets τ j of relevant inputs for the policy parameters with nonzero elements in the final ∆ θ . The output of this phase (the parameter-specific variables τ j ) can then be used to learn Partitioned MLP policies (c.f., Sec. V-A).

TABLE I: CREST evaluation for a toy environment on an aggregate ('Agg.') and mapping-specific ('Map.') basis. Accuracy ('Acc.') is whether all relevant states were detected. False positives ('F.P.') are states that were incorrect detected as relevant. 100 trials are used.

| Class     |   Dim. | Noise   |   Agg. Acc. |   Agg. F.P. |   Map. Acc. |   Map. F.P. |
|-----------|--------|---------|-------------|-------------|-------------|-------------|
| Linear    |      8 | None    |        1.00 |        0.00 |        0.98 |        0.19 |
| Nonlinear |      8 | None    |        1.00 |        0.00 |        0.97 |        0.23 |
| Linear    |     20 | None    |        1.00 |        0.00 |        0.99 |        0.53 |
| Nonlinear |     20 | None    |        1.00 |        0.00 |        0.97 |        0.24 |
| Linear    |      8 | Limited |        1.00 |        0.23 |        0.99 |        0.32 |
| Nonlinear |      8 | Limited |        1.00 |        0.13 |        0.95 |        0.22 |
| Linear    |     20 | Limited |        1.00 |        0.22 |        0.98 |        0.50 |
| Nonlinear |     20 | Limited |        1.00 |        0.12 |        0.96 |        0.23 |

## C. CREST Evaluation

Although we primarily use CREST for manipulation policies, we quantify the accuracy of CREST in a limited, application-agnostic manner. Table I shows the results of CREST on an environment that replicates the causal structure of hierarchical manipulation policies. This environment is designed so that, given a set of ground truth mappings between context variables and policy parameters, a controller with randomized structure is generated for the agent to 'manipulate' the environment to a goal location determined by the causal structure of the problem. These mappings were either linear or (weakly) nonlinear.

Under the assumptions of our controller and the context c , CREST excels at determining whether a variable is relevant. This is expected by construction of the underlying mathematics of this environment and represents an expected upper bound. We choose action space dimension of 8 and 20 (larger than our transfer learning experiments, c.f., Sec. VI) and select the state space accordingly to permit calculation of ground truth for testing. We also introduce action noise to test the robustness to uncertainty from interventions, leading to more variables detected. Relatively higher false positive rates arise in higher dimensions, but the overall reduction to the relevant set of variables is nonetheless significant.

## V. POLICY LEARNING AND TRANSFER

Given the relevant context from CREST, the structure and transfer learning pipeline of our work begins through 1) constructing the appropriate policy; 2) pretraining using the internal model; and 3) fine-tuning in the target setting.

## A. Policy Architectures

Given the full context c , the reduced context τ , and the parameter-specific contexts τ j , the three MLP-based network architectures shown in Fig. 1 are constructed and trained using actor-critic approaches [31].

The baseline π ( θ | c ) uses a standard MLP network. The approach π ( θ | τ ) uses a Reduced MLP (RMLP): an MLP network where the inputs are reduced to only the relevant context τ . The approach π ([ θ ] j | τ j ) has independent sets of fully connected layers for each policy parameter θ j , but with potentially overlapping inputs depending on τ j . This Partitioned MLP (PMLP) network represents the structural causal model [13] for each θ j with τ j as parent variables.

To provide a fair comparison, we choose the weights for the PMLP according to heuristics and multiply the number of hidden units by the size of the action space to size the RMLP and MLP. We originally sized the PMLP network according to [32] to provide theoretical guarantees regarding function approximation for one-element outputs (i.e., each θ j ), but the resulting network size for the baseline MLP was intractable to train. All neural network weights are randomly initialized per orthogonal initialization [33] using √ 2 and 0 . 01 for the scale terms of the hidden layers and output layers, respectively. All networks use tanh activations.

Fig. 1: A visualization of the different policy types. CREST is used to construct both the Reduced MLP (RMLP) and Partitioned MLP (PMLP). The baseline MLP is also shown for comparison. The relevant states are also used for the critic portion of the networks (only the actor portion is shown). The notation used is [ w 1 ,..., w d ], specifying the hidden units and depth of both the actor and critic.

<!-- image -->

## B. Network Training and Transfer

For both the internal model and target domain, we train our policies using PPO [34] with Stable Baselines [35]. First, each network is pretrained with the internal model until the task family is solved. Then, we transfer the network weights to the target domain and evaluate the policy to determine whether the policy transfers zero-shot. Otherwise, fine-tuning is performed. Although freezing network layers has been explored for fine-tuning control policies [36], we permit the entire network to adapt because of the approximate nature of pretraining with the internal model. The learned policy is considered to have solved the task family if it successfully achieves a predefined reward threshold on 50 validation tasks; this evaluation occurs after each policy update.

## VI. EXPERIMENTAL RESULTS

We evaluate how CREST can construct policies with greater (target) sample-efficiency and robustness for the robot manipulation tasks of block stacking and crate opening (Fig. 2). Our experiments for each task follow the structure and transfer learning pipeline motivating our approach: 1) use CREST to determine the causal structure of the task; 2) construct and pretrain policies with this structure; and 3) transfer and fine-tune these policies in the target domain. The target domains include manipulation tasks in NVIDIA Isaac Gym, enabling rigorous investigation of representative distribution shifts that may occur when deploying sim-to-real policies. For the block stacking task, we additionally leverage a real robot system to assess sim-to-real transfer.

Target simulation and training was conducted using a NVIDIA DGX-1. Pretraining was done using a NVIDIA GeForce RTX 2080. Samples from the block stacking and crate opening internal models were 400 and 65 times faster to obtain than target simulation samples, respectively. This is consistent with the concept of the internal model as a cheap, approximate simulator, whereas samples from the target domain are costly and therefore desirable to minimize.

Ten independent trials (from internal model pretraining to target fine-tuning) are conducted for each simulation experiment to provide statistically meaningful results given the variance inherent in model-free learning. Statistics are provided in terms of mean and ± 1 standard deviation of policy updates requires to solve the task family. Samples are provided per 1000 ('k-Samples') using a batch size of 512.

The supplementary materials describe further experiment details, such as the setup for the real block stacking target.

## A. Block Stacking

The network architectures for the block stacking policies are specified in Table II. Although our policies are nonlinear, note this particular task is linear between τ and θ .

Task representation. In the block stacking task, the context vector c = [ c B 0 T , . . . , c B NB -1 T ] T ∈ R 7 N B consists of the concatenation of N B individual block contexts. The context vector for block b is c B b = [ x w b , z w b , ψ b , h b , C b T ] T ∈ R 7 In the above equations, x w b and z w b are the world x - and z -positions of the blocks. Each block orientation is defined by its rotation angle ψ b about the block's vertical axis ( y ). The y -dimension, or height, of each block is h i . Lastly, the block color C b = [ R b , G b , B b ] T is specified via red-green-blue tuple. Note that y w b is not part of the context, as the initial scene always consists of blocks on the workspace plane.

The control policy π ( a | s, θ b ) for block stacking is a sequential straight-line skill parameterized by θ b = [ θ ∆ x , θ ∆ y , θ ∆ z ] T ∈ R 3 . This skill specifies waypoints that the robot traverses via impedance control by lifting the source block vertically, moving horizontally, and descending to the desired location. The skill preconditions are that the block is grasped and there are no obstructions to moving the object. The reward function is determined from the source block's position and the goal position upon the target block.

Using the internal model, CREST correctly obtained the relevant context variables as τ = [ x w 0 , x w 1 , h 1 , z w 0 , z w 1 ] T , τ ∆ x = [ x w 0 , x w 1 ] T , τ ∆ y = [ h 1 ] , and τ ∆ z = [ z w 0 , z w 1 ] T .

Nominal transfer for increasing context size. We conduct transfer experiments for N B = { 2 , 6 , 10 , 14 , 18 } , with each N B conducted independently. Our approach scales with the relevant part of the context space (Fig. 3), bounding the sample requirements for the target (as well as the cheaper, internal model). The increasing number of irrelevant dimensions from more blocks are eliminated by CREST prior to conducting domain randomization during pretraining.

For the case of N B = 10 , we trained directly in the target domain without transfer and observed similar results as for pretraining, suggesting the internal model accords well with the target domain. This explains why our approaches exhibit good zero-shot behavior over increasing context dimensions, unlike the baseline whose initial performance degrades as the number of irrelevant contexts increase.

Distribution shift in irrelevant contexts. We now evaluate the robustness of the learned block stacking policy to distributions shifts in irrelevant context variables. We conduct two transfer experiments, wherein the policies are pretrained using only half of the color space. In the first case, the target has the same context distribution as the internal model. In the second case, the target has the opposite color space (without overlap). The experimental results (Table III) elucidate how a seemingly inconsequential variable can degrade policy execution through a distribution shift that the robot is not trained to expect. Our approaches generate policies that are robust to these irrelevant domain shifts by construction; as CREST explicitly identifies this dimension as unimportant and excludes it from target learning.

Fig. 2: Transfer experiments for block stacking and crate opening manipulation tasks. Policies are pretrained in the internal model ((a),(d)) and then transferred to the target domain ((b),(c),(e)). Target domains consist of replications of real systems using a Franka Panda robot, along with a real system for block stacking. Both tasks have distractor objects. For block stacking, only two blocks are necessary to generalize the policy. For crate opening, blocks represent distractor objects (e.g., if the crate were for a chest containing toys).

<!-- image -->

Fig. 3: Sample complexity of training a solved block stacking policy based on context dimension for (a) internal model and (b) target setting. c) Zero-shot transfer percentage, wherein the transferred policy needs no further target training to solve the task.

<!-- image -->

TABLE II: Networks used for the block stacking task.

| Network          |   Parameters | Input Dim. (Total)   | Architecture   |
|------------------|--------------|----------------------|----------------|
| MLP ( N B = 2 )  |         3298 | 14 (14)              | [24, 24, 24]   |
| MLP ( N B = 6 )  |         4642 | 42 (42)              | [24, 24, 24]   |
| MLP ( N B = 10 ) |         5986 | 70 (70)              | [24, 24, 24]   |
| MLP ( N B = 14 ) |         7330 | 98 (98)              | [24, 24, 24]   |
| MLP ( N B = 18 ) |         8674 | 126 (126)            | [24, 24, 24]   |
| RMLP (ours)      |         2866 | 5 ( 7 N B )          | [24, 24, 24]   |
| PMLP (ours)      |          754 | 5 ( 7 N B )          | [8, 8, 8] x 3  |

Sim-to-real policy evaluation. Lastly, to validate our approach for sim-to-real transfer, we evaluate the zero-shot policy performance on a real robot system that implements the block stacking task with N B = 10 (Fig. 2c). As shown in Table IV, our policies successfully demonstrate greater zero-shot, sim-to-real transfer as compared to the baseline.

## B. Crate Opening

The crate opening experiment is nonlinear between τ and θ , and the internal model, which is kinematic, presents a greater sim-to-real gap than block stacking. Therefore, we also consider a second partitioned network, PMLP-R, with the same number of weights as the RMLP, to elicit possible influence of network expressivity in this domain due to the structural assumptions of the PMLP. The crate experiment primarily focuses on dynamics and context shifts, rather than varying numbers of objects, so unlike in blocks, the networks (Table V) are the same for all experiments. Beyond our two experiments in nominal transfer and dynamics shift, we also conducted a color shift experiment with similar results as with the blocks experiment.

Task representation. For the crate opening task, the context vector is c = [ c C T , c B 0 T , . . . , c B 9 T ] T ∈ R 80 . The block context is as defined previously. The crate context is c C = [ p w C T , Φ , x C g , z C g , Θ o , C C T ] T ∈ R 10 , where p w C = [ x w C , y w C , z w C ] T is the position of the crate coordinate frame with respect to the world frame with vertical angle Φ . The crate is always initially closed (horizontal), but the desired goal angle is specified by Θ o . The robot interacts with the crate via a grasp point specified in the frame of the crate by x C g and z C g which are orthogonal and parallel to the crate rotational axis, respectively. The color of the crate is C C .

The control policy π ( a | s, θ a ) is a robot skill that executes circular arcs emerging from the grasp point with the following parameterization: θ a = [ θ p w a T , θ ∆ γ , θ ∆ φ ] T ∈ R 5 , where θ p w a = [ θ x w a , θ y w a , θ z w a ] T is the sphere position used to calculate the radius from the grasp point. Then, the arc is traced out θ ∆ γ in azimuth and θ ∆ φ in inclination in polar coordinates from the grasp point. The skill preconditions are that the crate is grasped and unobstructed. To learn this policy, the reward function is calculated from the crate angle error and total kinematic error.

In this formulation, CREST determined the following relevant context variables, which are expected based on rigid body articulation kinematics:

TABLE III: Transfer results for a distribution shift in 30 context variables (color) that are irrelevant for the block stacking policy.

| Network     | IM Updates (k-Samples)     | Target Updates (k-Samples), no shift   |   Zero-Shot Transfer no shift | Target Updates (k-Samples), shift   |   Zero-Shot Transfer, shift |
|-------------|----------------------------|----------------------------------------|-------------------------------|-------------------------------------|-----------------------------|
| MLP         | 237.5 (121.6) ± 11.6 (5.9) | 1.6 (0.8) ± 1.5 (0.8)                  |                             4 | 17.3 (8.9) ± 4.8 (2.5)              |                           0 |
| RMLP (ours) | 137.6 (70.5) ± 7.2 (3.7)   | 0.5 (0.3) ± 0.9 (0.5)                  |                             7 | 1.0 (0.5) ± 1.5 (0.8)               |                           6 |
| PMLP (ours) | 139.2 (71.3) ± 8.2 (4.2)   | 0.0 (0.0) ± 0.0 (0.0)                  |                            10 | 0.1 (0.1) ± 0.3 (0.2)               |                           9 |

TABLE IV: Sim-to-real policy evaluation results for block stacking with N B = 10 . The reward threshold for zero-shot transfer is -0.025 (about half of the block width). We also note how often the block was successfully stacked. 'GT' is a ground truth policy to illustrate the degree of uncertainty present within the robot perception and control system. Each policy was evaluated 10 times.

| Policy      | Reward         |   Zero-Shot Transfer |   Block Stacked |
|-------------|----------------|----------------------|-----------------|
| MLP         | -0.033 ± 0.012 |                    3 |               1 |
| RMLP (ours) | -0.018 ± 0.007 |                    9 |               4 |
| PMLP (ours) | -0.014 ± 0.004 |                   10 |               6 |
| GT          | -0.009 ± 0.003 |                   10 |              10 |

TABLE V: Networks used for the crate opening task.

| Network       |   Parameters | Input Dim. (Total)   | Architecture     |
|---------------|--------------|----------------------|------------------|
| MLP           |        13496 | 80 (80)              | [40, 40, 40]     |
| RMLP (ours)   |         7576 | 6 (80)               | [40, 40, 40]     |
| PMLP (ours)   |         1152 | 6 (80)               | [8, 8, 8] x 5    |
| PMLP-R (ours) |         8032 | 6 (80)               | [24, 24, 24] x 5 |

<!-- formula-not-decoded -->

Nominal transfer. The transfer learning results for the crate opening policy is shown in Table VI. Pretraining the model reduced the number of target updates for the nonpartitioned networks. However, this was not the case for the partitioned networks, regardless of size. This is likely a result of the discrepancy between the internal model and the target domain, which also explains the difference between the policy updates required for pretraining versus training directly in the target. However, the reduction of relevant variables reduces the number of updates required to train directly in the target for both the RMLP and PMLP.

Dynamics distribution shift. Unlike the block stacking problem, the modeling gap between the internal model and target setting is sufficiently large that the trained policies incur a significant performance degradation upon first evaluating in the target domain. We investigated this further by transferring the policies to two target settings with different crate stiffness values. To focus on this dynamics shift, no other shifts (e.g., in context space) were induced.

The results in Table VII suggest that increasing the stiffness is sufficient as a proxy for increasing the modeling difference between the internal model and the target. The optimal parameters for the kinematic case (internal model) are not necessarily the same as the target domain with realistic dynamics of manipulation using impedance control. Therefore, greater modeling differences implies that greater search in policy parameter space is required to converge to parameters that generalize in the target domain.

In all cases, we see that the RMLP network performs best. As a likely consequence of a less expressive network with a larger dynamics gap, the smaller PMLP network demonstrated a significant variance increase in the higher stiffness case than the larger PMLP-R. Overall, our policies are more robust to distribution shifts in model dynamics. However, we note that partitioning imposes structure that may not be optimal for this problem, as the MLP outperformed the partitioned networks in the light and nominal stiffness cases.

TABLE VI: Pretraining and transfer results for crate opening policies compared to training directly in target (without transfer).

| Network       | IM Updates (k-Samples)       | Target Updates (k-Samples), transfer   | Target Updates (k-Samples), direct   |
|---------------|------------------------------|----------------------------------------|--------------------------------------|
| MLP           | 45.0 ± 3.16 (23.04 ± 1.62)   | 12.20 ± 1.72 (6.25 ± 0.88)             | 38.70 ± 10.99 (19.8 ± 5.63)          |
| RMLP (ours)   | 32.40 ± 3.67 (16.59 ± 1.88)  | 7.0 ± 1.18 (3.58 ± 0.61)               | 16.40 ± 4.05 (8.40 ± 2.08)           |
| PMLP (ours)   | 48.20 ± 13.33 (24.68 ± 6.83) | 14.0 ± 3.74 (7.17 ± 1.92)              | 14.20 ± 6.32 (7.27 ± 3.24)           |
| PMLP-R (ours) | 51.30 ± 10.82 (26.27 ± 5.54) | 14.3 ± 4.34 (7.32 ± 2.22)              | 15.80 ± 3.97 (8.09 ± 2.03)           |

TABLE VII: Fine-tuning for crate opening policies with increasing crate stiffness and correspondingly greater transition model difference between the internal model and target task.

| Network       | Target Updates (k-Samples), light   | Target Updates (k-Samples), nominal   | Target Updates (k-Samples), stiff   |
|---------------|-------------------------------------|---------------------------------------|-------------------------------------|
| MLP           | 3.90 ± 0.54 (2.00 ± 0.28)           | 12.20 ± 1.72 (6.25 ± 0.88)            | 27.30 ± 5.51 (13.98 ± 2.82)         |
| RMLP (ours)   | 3.20 ± 0.60 (1.64 ± 0.31)           | 7.00 ± 1.18 (3.58 ± 0.61)             | 16.40 ± 5.90 (8.40 ± 3.02)          |
| PMLP (ours)   | 9.10 ± 1.58 (4.66 ± 0.81)           | 14.00 ± 3.74 (7.17 ± 1.92)            | 24.00 ± 15.06 (12.29 ± 7.71)        |
| PMLP-R (ours) | 9.20 ± 1.54 (4.71 ± 0.79)           | 14.30 ± 4.34 (7.32 ± 2.22)            | 19.10 ± 4.91 (9.80 ± 2.51)          |

## VII. CONCLUSION

The causal reasoning afforded by CREST allows the robot to structure robot manipulation policies with fewer parameters that are more sample efficient and robust to domain shifts than a naive approach that includes all known contexts. Indeed, using causality to reason about the simulation of a task identifies what variables are important to generalize a policy, while domain randomized pretraining provides a strong, task-specific prior in terms of how they matter. We believe that CREST is one step towards a new paradigm for structural sim-to-real transfer of robot manipulation policies that are sufficiently lightweight to be adapted in-the-field to overcome unforeseen domain shifts.

For future work, we will investigate using precondition learning to relax the assumption that the policy execution is feasible. We will also explore how the robot can learn the internal model used as the causal reasoning engine.

## ACKNOWLEDGMENTS

We gratefully acknowledge support from the U.S. Office of Naval Research (Grant N00014-18-1-2775), U.S. Army Research Laboratory (Grant W911NF-18-2-0218 as part of the A2I2 Program), and the NVIDIA NVAIL Program.

## REFERENCES

- [1] X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel, 'Simto-Real Transfer of Robotic Control with Dynamics Randomization,' Int'l Conf. on Robotics and Automation (ICRA) , 2018.
- [2] K. Bousmalis, A. Irpan, P. Wohlhart, Y. Bai, M. Kelcey, M. Kalakrishnan, L. Downs, J. Ibarz, P. Pastor, K. Konolige, et al. , 'Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping,' Int'l Conf. on Robotics and Automation (ICRA) , 2018.
- [3] O. Kroemer, S. Niekum, and G. Konidaris, 'A Review of Robot Learning for Manipulation: Challenges, Representations, and Algorithms,' arXiv preprint arXiv:1907.03146 , 2019.
- [4] W. Zhao, J. P. Queralta, and T. Westerlund, 'Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey,' IEEE Symposium Series on Computational Intelligence (SSCI) , 2020.
- [5] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, 'Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World,' Int'l Conf. on Intelligent Robots and Systems (IROS) , 2017.
- [6] J. Liang, V. Makoviychuk, A. Handa, N. Chentanez, M. Macklin, and D. Fox, 'GPU-Accelerated Robotic Simulation for Distributed Reinforcement Learning,' Conf. on Robot Learning (CoRL) , 2018.
- [7] K. Zhang, B. Sch¨ olkopf, P. Spirtes, and C. Glymour, 'Learning Causality and Causality-Related Learning: Some Recent Progress,' National Science Review , vol. 5, no. 1, pp. 26-29, 2018.
- [8] B. Sch¨ olkopf, 'Causality for Machine Learning,' arXiv preprint arXiv:1911.10500 , 2019.
- [9] J. Zhang and E. Bareinboim, 'Transfer Learning in Multi-Armed Bandits: A Causal Approach,' Int'l Joint Conf. on Artificial Intelligence (IJCAI) , 2017.
- [10] P. de Haan, D. Jayaraman, and S. Levine, 'Causal Confusion in Imitation Learning,' Conf. on Neural Information Processing Systems (NeurIPS) , 2019.
- [11] Y. Li, A. Torralba, A. Anandkumar, D. Fox, and A. Garg, 'Causal Discovery in Physical Systems from Videos,' Conf. on Neural Information Processing Systems (NeurIPS) , 2020.
- [12] S. A. Sontakke, A. Mehrjou, L. Itti, and B. Sch¨ olkopf, 'Causal Curiosity: RL Agents Discovering Self-supervised Experiments for Causal Representation Learning,' arXiv preprint arXiv:2010.03110 , 2020.
- [13] J. Pearl, Causality . Cambridge University Press, 2009.
- [14] O. Ahmed, F. Tr¨ auble, A. Goyal, A. Neitz, M. W¨ uthrich, Y. Bengio, B. Sch¨ olkopf, and S. Bauer, 'CausalWorld: A Robotic Manipulation Benchmark for Causal Structure and Transfer Learning,' arXiv preprint arXiv:2010.04296 , 2020.
- [15] C. Devin, P. Abbeel, T. Darrell, and S. Levine, 'Deep Object-Centric Representations for Generalizable Robot Learning,' Int'l Conf. on Robotics and Automation (ICRA) , 2018.
- [16] W. Yu, J. Tan, C. K. Liu, and G. Turk, 'Preparing for the Unknown: Learning a Universal Policy with Online System Identification,' Robotics: Science and Systems (RSS) , 2017.
- [17] A. Nouri and M. L. Littman, 'Dimension Reduction and its Application to Model-Based Exploration in Continuous Spaces,' Machine Learning , vol. 81, no. 1, pp. 85-98, 2010.
- [18] J. Z. Kolter and A. Y. Ng, 'Regularization and Feature Selection in Least-Squares Temporal Difference Learning,' Int'l Conf. on Machine Learning , 2009.
- [19] R. Parr, L. Li, G. Taylor, C. Painter-Wakefield, and M. L. Littman, 'An Analysis of Linear Models, Linear Value-Function Approximation, and Feature Selection for Reinforcement Learning,' Int'l Conf. on Machine Learning , 2008.
- [20] J. Tan, T. Zhang, E. Coumans, A. Iscen, Y. Bai, D. Hafner, S. Bohez, and V. Vanhoucke, 'Sim-to-Real: Learning Agile Locomotion for Quadruped Robots,' Robotics: Science and Systems (RSS) , 2018.
- [21] A. Molchanov, T. Chen, W. H¨ onig, J. A. Preiss, N. Ayanian, and G. S. Sukhatme, 'Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to Multiple Quadrotors,' Int'l Conf. on Intelligent Robots and Systems (IROS) , 2019.
- [22] Y. Chebotar, A. Handa, V. Makoviychuk, M. Macklin, J. Issac, N. Ratliff, and D. Fox, 'Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience,' Int'l Conf. on Robotics and Automation (ICRA) , 2019.
- [23] W. Masson, P. Ranchod, and G. Konidaris, 'Reinforcement Learning with Parameterized Actions,' AAAI Conf. on Artificial Intelligence , 2016.
- [24] M. Hausknecht and P. Stone, 'Deep Reinforcement Learning in Parameterized Action Space,' Int'l Conf. on Learning Representations (ICLR) , 2016.
- [25] Z. Fan, R. Su, W. Zhang, and Y. Yu, 'Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space,' Int'l Joint Conf. on Artificial Intelligence (IJCAI) , 2019.
- [26] M. P. Deisenroth, G. Neumann, J. Peters, et al. , 'A Survey on Policy Search for Robotics,' Foundations and Trends in Robotics , vol. 2, no. 1-2, pp. 1-142, 2013.
- [27] P. W. Battaglia, J. B. Hamrick, and J. B. Tenenbaum, 'Simulation as an Engine of Physical Scene Understanding,' Proceedings of the National Academy of Sciences , vol. 110, no. 45, pp. 18 327-18 332, 2013.
- [28] D. Ha and J. Schmidhuber, 'Recurrent World Models Facilitate Policy Evolution,' Conf. on Neural Information Processing Systems (NeurIPS) , 2018.
- [29] B. Sch¨ olkopf, F. Locatello, S. Bauer, N. R. Ke, N. Kalchbrenner, A. Goyal, and Y. Bengio, 'Toward Causal Representation Learning,' Proceedings of the IEEE , 2021.
- [30] J. Peters, K. M¨ ulling, and Y. Altun, 'Relative Entropy Policy Search,' AAAI Conf. on Artificial Intelligence , 2010.
- [31] V. R. Konda and J. N. Tsitsiklis, 'Actor-Critic Algorithms,' Conf. on Neural Information Processing Systems (NeurIPS) , 2000.
- [32] Z. Lu, H. Pu, F. Wang, Z. Hu, and L. Wang, 'The Expressive Power of Neural Networks: A View from the Width,' Conf. on Neural Information Processing Systems (NeurIPS) , 2017.
- [33] A. M. Saxe, J. L. Mcclelland, and S. Ganguli, 'Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks,' Int'l Conf. on Learning Representations (ICLR) , 2014.
- [34] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, 'Proximal Policy Optimization Algorithms,' arXiv preprint arXiv:1707.06347 , 2017.
- [35] A. Hill, A. Raffin, M. Ernestus, A. Gleave, A. Kanervisto, R. Traore, P. Dhariwal, C. Hesse, O. Klimov, A. Nichol, M. Plappert, A. Radford, J. Schulman, S. Sidor, and Y. Wu, 'Stable Baselines,' https://github. com/hill-a/stable-baselines, 2018.
- [36] A. A. Rusu, N. C. Rabinowitz, G. Desjardins, H. Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell, 'Progressive Neural Networks,' arXiv preprint arXiv:1606.04671 , 2016.
- [37] Q.-Y. Zhou, J. Park, and V. Koltun, 'Open3D: A Modern Library for 3D Data Processing,' arXiv preprint arXiv:1801.09847 , 2018.
- [38] S. Katz, A. Tal, and R. Basri, 'Direct Visibility of Point Sets,' in ACM SIGGRAPH 2007 papers , 2007, pp. 24-es.
- [39] M. Ester, H.-P. Kriegel, J. Sander, X. Xu, et al. , 'A Density-based Algorithm for Discovering Clusters in Large Spatial Databases with Noise,' in Proceedings of the Second Int'l Conf. on Knowledge Discovery and Data Mining (KDD) , vol. 96, no. 34, 1996, pp. 226231.
- [40] K. Zhang, M. Sharma, J. Liang, and O. Kroemer, 'A Modular Robotic Arm Control Stack for Research: Franka-Interface and FrankaPy,' arXiv preprint arXiv:2011.02398 , 2020.

## SUPPLEMENTARY MATERIALS

## A. Summary of CREST

Which state features are important for learning a control policy? Our approach, CREST, addresses this question through causal feature selection. CREST selects the relevant state variables for a given control policy, which apply over the policy's preconditions. The assumptions for CREST are that an internal model (i.e., an approximate task simulation) exists, the context space representation of the internal model facilitates causal interventions (e.g., disentangled variables), and the (parameterized) control policy and its preconditions are known. Through structure and transfer learning, CREST enables learning of policies that are compact, avoiding unnecessary state features. By construction, policies built using CREST are robust to distribution shifts in irrelevant variables, whereas baseline methods may yield policies with spurious correlations that are brittle. Such distribution shifts could arise from transfer between the internal model and reality, due to variations in dynamics or context distributions not encountered during pretraining with the internal model.

## B. CREST Analysis on Math Environment

We now provide a greater description of the manipulation environment described in Sec. IV-C. The toy environment, MathManipEnv , approximates the mathematics of a controller for goal-based manipulation. For simplicity, the lowlevel control policy simply perturbs the state s ∈ R | S | by an input of θ = a ∈ R | A | in a manner specific to whether the system is linear or non-linear. Additionally, we consider the context c ∈ R | S | to be the initial state, s 0 . For this evaluation, we considered cases where | S | = | A | ('Dim.' in Table I).

The reward for this task is

<!-- formula-not-decoded -->

where g a ∈ R | g | is the goal vector that was obtained after execution of the controller to yield achieved state s a , and g d ∈ R | g | is the desired goal. The goal vector is calculated from a goal selection matrix G ∈ R | g |×| S | , which is a one-hot encoding matrix where the columns indicate the elements of the state vector that are used. In practice, G is formed by first randomly selecting N τ relevant context variables from the total set of c to form τ . Then, each τ is randomly allocated to a separate dimension of the goal vector, i.e., row of G . Here, G represents that, in some goal-based problems, the goal is calculated from only a subset of the state vector (e.g., relative to the position of a particular object).

The process of the system is either linear or non-linear, where A ( θ ) = ∆ s + w a and w a ∼ N (0 , σ 2 a ) is the action noise. In the linear case, the controller A ∈ R | S |×| S | is a matrix with randomly selected coefficients, so ∆ s = Aθ . The non-zero coefficients of A indicate mappings of τ j to θ j . In the non-linear case, the controller A is a list of size | A | , where each element of the list specifies randomly selected functions (exponential, sigmoid, sine, cosine) that transform input a j into the resulting ∆ s j .

Each trial of this environment randomly selects different τ , g d , s 0 , G , and A . The goal vector dimensionality | g | is fixed for each run and is typically equal to | S | .

## C. Task Representation: Block Stacking

We now provide additional detail for the block stacking task described in Sec. VI-A. Figure 4 illustrates some context variables and the policy trajectory. In this task, the robot must stack the source block (block 0) upon the target block (block 1) using a sequential straight-line skill with control policy π ( a | s, θ b ) and known preconditions.

Policy. The policy parameters θ b = [ θ ∆ x , θ ∆ y , θ ∆ z ] T ∈ R 3 define three waypoints that the robot is sequentially commanded to via impedance control. Specifically, let y p represent a vertical position above the table and blocks. After the block is grasped, the executed policy is therefore:

- 1) Vertically lift to y p .
- 2) Move ( θ ∆ x , θ ∆ z ) at fixed y p .
- 3) Vertically move θ ∆ y -y p .

The vertical lift to y p avoids obstructions to moving the block, so the preconditions of this skill are always satisfied. Reward. The reward function for this task is

<!-- formula-not-decoded -->

where α = 1 is the reward weight, p w g = [ x w g , y w g , z w g ] T is the goal position of block 0, and p w 0 ,a = [ x w 0 ,a , y w 0 ,a , z w 0 ,a ] T is the final (i.e., achieved) position of block 0 at the end of the policy execution. In this task, the goal is to stack block 0 upon block 1. Therefore, x w g = x w 1 , y w g = 1 2 h 0 + 1 2 h 1 + y w 1 , and z w g = z w 1 .

Fig. 4: Diagram of the block stacking task. The world coordinate frame { W } is defined at the base link of the robot. Each block coordinate frame { B } is defined at the block's centroid.

<!-- image -->

Fig. 5: Block state estimation used for the sim-to-real experiments using RGB-D perception. The perception algorithm takes as input a colored point cloud, and outputs a position, rotation, and color for 10 blocks. The red point in each block represents the block centroid, and the dashed lines indicate the block length and width (known a priori ). The best-fit position and rotation angle for each point cloud cluster yields a pose estimate for each block.

<!-- image -->

Expressing the reward function in terms of the task context variables elucidates which variables are considered relevant to the task policy. The optimal low-level policy parameters

<!-- formula-not-decoded -->

where the reduction of the block y -position variables arise from the blocks being initially constrained to the table.

The above derivation demonstrates that only certain variables are needed to generalize the policy across different contexts where the preconditions also hold true. Moreover, certain variables are only influential in certain policy parameters. We express this more concretely by formalizing what variables are needed for each parameter, which is where the ground truth mappings for CREST (Sec. VI-A) arise:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, f is the model, which for our work we characterize using a neural network (although for this specific task, a linear model would also suffice). In causality terms, this is equivalent to modeling each individual policy parameter ( θ j ) as a structural causal model, where f is a function with parent variables given by τ j .

## D. Sim-to-Real Block Stacking Experiment

The sim-to-real block stacking experiment (Sec. VI-A) demonstrates that our proposed approach works in practice on a real robot system (Fig. 2c). As it is experimentally difficult to realize all possible values within the context distributions (e.g., creating blocks of precise height and color for each sample), we instead conduct the experiment on a slightly reduced distribution range. Specifically, we conduct this experiment using 10 blocks, where each block has a different color and two possible heights (5.7 cm or 7.6 cm). Before each trial, all block positions and rotations are shuffled by hand. Additionally, a random number generator selects the height of each block, as well as the enumeration of the blocks (and therefore which blocks are the source and target). The length and width of each block is 4.2 cm, which does not change during the experiment and is known from manual measurement (i.e., not perception).

Perception. We use a Microsoft Azure Kinect RGB-D camera to estimate each block's position, rotation, and color through a model-based perception algorithm utilizing the Open3D library [37]. Figure 5 shows an example of the block perception. The perception algorithm is as follows:

- 1) Crop to region bounded by the table blue tape (Fig. 2c).
- 2) Removal of hidden points via Katz [38], i.e., points expected to be occluded from the camera viewpoint.
- 3) Fit plane to table via random sample consensus (RANSAC) and remove any points below this plane.
- 4) Detect remaining clusters with DBSCAN [39], a density-based clustering algorithm. Proceed only if the numbers of clusters is N B = 10 , or reject the perception sample and try again.
- 5) For each cluster (block), determine the best position and angle that fits a cube of known dimensions to the cluster via least-squares optimization. This step yields an estimate of each block's position and rotation.
- 6) Estimate block color by averaging the colors of all points within a cluster (block).

Due to difficulties with accurately estimating block height from depth, the block height is provided by manual input instead. Manual checks are also completed prior to executing the control policy to ensure block perception results are reasonable. For example, if one cluster was not a block, but part of the blue tape, the perception sample would be rejected and attempted again. Prior to running the perception system, we obtain the extrinsics of the camera via a target-based calibration procedure, and we use the intrinsics as reported directly from the camera.

Control. We use the FrankaPy library [40] that implements impedance control for the Franka Emika Panda robot.

## E. Task Representation: Crate Opening

This section provides more detail of the crate opening task (Sec. VI-B), where the objective is to open a crate in the presence of distractor objects using a circular arc skill with control policy π ( a | s, θ a ) and known preconditions. Figure 6

TABLE VIII: Transfer results for a distribution shift in 33 context variables that are irrelevant for the crate opening policy. Variables are 3-tuple RGB colors of the crate and 10 blocks in the scene. Light crate stiffness.

| Network       | IM Updates (k-Samples)      | Target Updates (k-Samples), no shift   | Target Updates (k-Samples), shift   |
|---------------|-----------------------------|----------------------------------------|-------------------------------------|
| MLP           | 36.10 ± 3.11 (18.48 ± 1.59) | 11.10 ± 2.30 (5.68 ± 1.18)             | 17.30 ± 4.22 (8.86 ± 2.16)          |
| RMLP (ours)   | 36.20 ± 5.23 (18.53 ± 2.68) | 3.40 ± 0.66 (1.74 ± 0.34)              | 3.50 ± 0.67 (1.80 ± 0.34)           |
| PMLP (ours)   | 48.50 ± 7.88 (24.80 ± 4.03) | 8.50 ± 2.06 (4.35 ± 1.06)              | 8.30 ± 1.10 (4.25 ± 0.56)           |
| PMLP-R (ours) | 53.00 ± 7.80 (27.14 ± 3.99) | 9.50 ± 1.91 (4.86 ± 0.98)              | 9.60 ± 2.01 (4.92 ± 1.03)           |

Fig. 6: Diagram of the crate opening task. The world coordinate frame { W } (not shown) is defined at the base link of the robot, similar to the block stacking task (Fig. 4). The z -axis of the crate coordinate frame { C } is coincident with the crate hinge. There are 10 distractor blocks, each with coordinate frame { B } .

<!-- image -->

shows some context variables and the policy trajectory, which emerges from the crate grasp point.

This task has a larger modeling difference between the internal model and the target domain, which could also contribute to why our partitioned networks (PMLP, PMLPR) were less successful than our non-partitioned network (RMLP). In addition to the dynamics domain difference discussed in Sec. VI-B, the y -position of the grasp point, y C g , is also slightly different. For the internal model, y C g exists in the same plane as the crate, but for the target, y C g is slightly above the crate because of the protruding grasp point.

Policy. The policy parameters θ a = [ θ p w a T , θ ∆ γ , θ ∆ φ ] T ∈ R 5 define a circular arc that is composed of N T waypoints. The robot is commanded to the crate grasp point, then the robot executes the policy by sequentially following each waypoint via impedance control. The crate cannot open into the blocks below, so the preconditions are always satisfied.

Reward. The reward function for this task is

<!-- formula-not-decoded -->

In this function, ∆Θ = Θ a -Θ o is the difference between the achieved ( Θ a ) and goal ( Θ o ) crate angles, and α a and α k are reward weights. The term e k is the kinematic error in the policy trajectory, which is intended to induce robot trajectories that are safe (physically realizable and low force) in the target domain given articulated motion of the crate. For this work, α a = 1 and α k is 5 for the internal model and 0 for the target domain (because the robot realizes the trajectory it can actually achieve on the target due to the crate's articulated motion).

Specifically, e k = 1 N T ∑ N T t =1 ‖ p w a,t -p w d,t ‖ , where p w d,t is the desired position of a waypoint in the trajectory and p w a,t is the kinematically realizable position of that same waypoint, both at timestep t . This is determined by projecting the desired waypoint onto the plane formed by rotating the grasp point about the crate hinge, obtaining the resulting crate angle, and using this angle to compute the realized grasp point.

## F. Crate Opening Distribution Shift in Irrelevant Contexts

As mentioned in Sec. VI-B, we also conducted a crate opening experiment with distribution shifts in irrelevant parts of the context space, similar to the experiment in the block stacking task (Table III). As before, we pretrain on the entire context space, except for color of the crate and blocks, where only half of the color space is used. For testing, we transfer to two cases: 1) the same color space seen in training (no shift), and 2) the opposite color space (complete shift with no overlap). This experiment uses the 'light' crate stiffness.

Table VIII shows the results of this experiment. As expected, our policies are robust to distribution shifts of this type, whereas the baseline MLP incurs approximately 55% more target updates to overcome these irrelevant distribution shifts. Unlike the version of this experiment for block stacking, no policies achieved zero-shot transfer. However, this is because of the previously described domain shift in dynamics between the internal model and target domain.