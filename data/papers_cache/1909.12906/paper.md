## Meta Reinforcement Learning for Sim-to-real Domain Adaptation

Karol Arndt 1 , Murtaza Hazara 1 , Ali Ghadirzadeh 1 , 2 , Ville Kyrki 1

Abstract -Modern reinforcement learning methods suffer from low sample efficiency and unsafe exploration, making it infeasible to train robotic policies entirely on real hardware. In this work, we propose to address the problem of sim-toreal domain transfer by using meta learning to train a policy that can adapt to a variety of dynamic conditions, and using a task-specific trajectory generation model to provide an action space that facilitates quick exploration. We evaluate the method by performing domain adaptation in simulation and analyzing the structure of the latent space during adaptation. We then deploy this policy on a KUKA LBR 4+ robot and evaluate its performance on a task of hitting a hockey puck to a target. Our method shows more consistent and stable domain adaptation than the baseline, resulting in better overall performance.

## I. INTRODUCTION

In recent years, we have witnessed a tremendous progress in reinforcement learning research, accompanied by its growing application in robotics. Reinforcement learning, however, requires vast amounts of training data, which can be relatively costly to provide in robotics [1], in contrast to applications like computer games [2], [3]. Apart from reliance on large amounts of data, with most methods the training process involves random exploratory actions, which can be unpredictable and potentially unsafe both to the operational environment and to the robot itself.

A promising solution to these problems lies in using physics simulators, such as MuJoCo [4] or Flex [5], to reduce training time and mitigate the risk of hardware damage [6][8]. However, directly deploying the trained model on physical hardware still requires an accurate match between the simulation and real-world, which may be impossible to achieve even after tedious tuning of simulation parameters, because the simulation may not model some of the physical phenomena present in real world. As an alternative to carefully tuning the simulator, a model trained on imprecise dynamics can be adapted to the real world environment to make up for potential modelling inaccuracies [9], [10]. Recent developments in the field of meta learning made it a compelling, yet still understudied, approach to this problem.

In this work, we propose a novel method for domain adaptation using meta learning, or learning to learn . As opposed to most common machine learning problem formulations, where the goal is to train a model to excel in one particular task, the basic principle in meta learning is to train models which are good at adapting to new tasks or situations.

*This work was financially supported by Academy of Finland grant 313966 and Business Finland grant 3338/31/2017. We also gratefully acknowledge the support of NVIDIA Corporation with the donation of the Titan Xp GPU used for this research.

1 Aalto University, Espoo, Finland first.last@aalto.fi

2 KTH Royal Institute of Technology, Stockholm, Sweden

Fig. 1: Randomized dynamic properties lead to large changes in the behaviour of the system. We train a model to adapt to large variations in simulation (a), and deploy it on a physical robot (b).

<!-- image -->

We consider tasks that are heavily dependent on the dynamic parameters of the environment. Such tasks cannot be transferred from simulation to reality in a zero-shot manner and require data from the physical system to be used for domain adaptation or system identification. We combine gradient-based meta learning with generative models for trajectories to explicitly train a policy to adapt to a wide range of randomized dynamics in simulation, illustrated in Figure 1a. The trained model is then deployed on a physical setup, shown in Figure 1b, quickly adapting to new conditions. The method's feasibility for sim-to-real transfer is demonstrated on a task where the goal is to shoot a hockey puck to a target location under unknown friction. We show that, after a small number of trials, our system is able to adapt both to new conditions in simulation and to the dynamic parameters of the physical system, improving the performance in cases where simple domain adaptation methods fail, or result in unstable policy updates.

The contributions of our work are (1) demonstrating that gradient-based meta learning results in predictable and consistent domain adaptation, and thus is a suitable approach for simulation to reality (sim-to-real) transfer of robotic policies under uncertain dynamics, and (2) combining a meta learned policy with latent variable generative models to represent motor trajectories leads to a safe and lowdimensional exploration space.

## II. RELATED WORK

In this section, we provide an overview of previous work related to meta learning and sim-to-real transfer.

## A. Meta learning

The general idea of meta learning was first described by Schmidhuber [11], whose early work on the topic pioneered the use of meta learning with neural network models [12], as well as its application in reinforcement learning [13].

More recently, two principal families of meta learning algorithms have been introduced. First, memory can be embedded as a part of the learned structure, causing the network to adapt as data gets passed through it. Such behaviour can be accomplished by recurrent architectures [14], [15] or by using an additional set of plastic weights [16].

Second, parameters of the network can be optimized such that they provide a good starting point for further adaptation. This is the case in model-agnostic meta learning (MAML) [17] which explicitly optimizes the model performance after a number of adaptation steps. MAML has been demonstrated to achieve good performance in tasks such as few-shot classification and reinforcement learning, including more complex tasks, such as robot simulations [18]. Multiple improvements to the method were later introduced by various authors [19], [20].

## B. Sim-to-real domain transfer

Zero shot transfer refers to learning a policy that does not need to be adapted in the target domain. A common approach for zero-shot transfer is domain randomization, which exposes the model to a variety of conditions, so as to make the model robust to modelling inaccuracies in these aspects. The idea can be applied both to perception [8], [21][23] and to the dynamics of the system [24], [25]. Domain randomization may, however, be insufficient since a single policy that performs well across the domain might not exist.

One solution is to build a more accurate simulation model for the particular environment, either by thorough measurements [7] or by interweaving simulation rollouts with real robot samples and optimizing the simulation [6] or the policy [26], such that the discrepancies are minimized. This can be, however, costly and time-consuming, and requires access to physical hardware at the time of training. As a solution, memory can be embedded as part of the network to encode previous states and actions, allowing the network to identify and respond to a variety of dynamic conditions [9].

The problem can also be approached from a different direction-the policy trained in simulation can also be directly used as a starting point for further adaptations in real world [10]. The initial parameters may, however, not be a good point for further adaptation. Gradient-based meta learning methods are an appealing solution to this problem. A method for stabilizing model-based reinforcement learning using gradient-based meta learning was proposed by Clavera et al. [27] to address minor uncertainties in dynamics originating from lack of related training data. In contrast, in this work, we directly adapt the policy to a wide range of dynamic conditions using model-free methods, and additionally evaluate its performance on a physical system.

## III. METHOD

In this section, our method for sim-to-real transfer learning is introduced. First, we describe the necessary preliminaries related to domain adaptation using meta reinforcement learning, followed by the formal problem formulation. We then describe the details of our approach for trajectory generation, domain adaptation, and meta-policy training.

## A. Preliminaries and problem statement

A standard sequential decision making setup consists of an agent interacting with an environment in discrete timesteps. At each timestep t , the agent takes an action a t , causing the environment to change its state from s t to s t +1 . Each state transition is accompanied by a corresponding reward r ( s t , a t ) to assess the quality of the action. This setup is a Markov decision process (MDP) with a set of states s ∈ S , actions a ∈ A , and transition probabilities between these states in response to each action p ( s t +1 | s t , a t ) . The agent's actions are chosen according to a policy π ( a t | s t ) , which describes the probability of taking action a t in state s t . The objective of reinforcement learning then is to find the optimal policy, defined as the policy that maximizes the expected cumulative sum of rewards for a specific MDP.

In contrast to this formulation, meta reinforcement learning considers a set of MDPs, M . The goal is to find a learning algorithm that is able to efficiently learn optimal policies for all MDPs in M -that is, to learn to learn policies in M . In the domain adaptation scenario, we consider M to consist of MDPs sharing the same reward function r ( s, a ) , as well as the action and state spaces ( A and S ), but varying in terms of state transition probabilities p ( s t +1 | s t , a t ) . We further assume that, for each MDP M k ∈ M , these transitions can be described by a set of dynamic parameters, further referred to as task τ ∈ τ .

In the context of meta learning, the domain adaptation problem can therefore be stated as follows: for a set of Markov decision processes M , described by tasks τ i ∈ τ , find a learning algorithm which, after performing N adaptation steps under a new dynamic condition τ , results in the optimal policy for these conditions, π ∗ θ,τ .

## B. Trajectory generation

In order to provide the policy with a low-dimensional, smooth action space which facilitates exploration, we train a generative model over a distribution of task-specific trajectories u 0: T = g φ ( z ) , where u is a trajectory and g φ the generative model parametrized by a latent variable z . This is similar to [28] and [8].

We obtain the generative model by training a variational autoencoder (VAE) on a set of trajectories which are suitable for the given task and safe to be executed on the physical robot. A VAE consists of two parts-the encoder and the decoder. The encoder outputs a probability distribution representing the low-dimensional latent representation of the input. During training, a sample is drawn from this distribution and passed to the decoder, which reconstructs the original input based on the low-dimensional representation. The decoder part of the VAE, on its own, can be used to map vectors in the latent space to the output domain, which in our case represents useful trajectories.

Fig. 2: Overview of one step of the adaptation procedure.

<!-- image -->

This formulation allows us to train a policy for latent actions z , π ( z | s ) , effectively reducing the dimensionality of the action space, alleviating the problem of time complexity and allowing the model to focus on terminal rewards. Effectively, this formulation results in faster training and safer on-policy domain adaptation.

## C. Domain adaptation

The goal of the domain adaptation step is to adjust the policy parameters in such a way that the policy's performance will improve for the current dynamic conditions. This process is outlined in Figure 2 and in Algorithm 1. For clarity, Figure 2 shows only a single adaptation step ( N = 1 ).

## Algorithm 1: Policy adaptation

```
Input: trained generator g φ and meta policy π θ 0 Result: Adapted policy parameters θ N 1 repeat n := N times 2 repeat k := K times 3 get goal state s g ; 4 sample z k ∼ π θ n ( z | s g ) ; 5 generate trajectory u 0: T := g φ ( z k ) ; 6 execute u 0: T , save s g , z k and the reward r k ; 7 end 8 normalize rewards ¯ r = r -mean ( r ) stddev ( r ) ; 9 calculate loss L := -1 K ∑ K k ˜ r k log π ( z k | s k ) ; 10 update policy parameters θ n := θ n -1 -α ∇ θ n -1 L ; 11 end
```

The adaptation begins by sampling a random goal state from the environment and passing it to the current policy (step 3). The policy returns a latent action distribution π ( z | s ) , from which a latent vector z is sampled and passed to the generative model g φ to construct the corresponding trajectory (steps 4 and 5). The constructed trajectory is then executed by the robot and the state, action and reward are stored (step 6). This process is repeated K times.

After K rollouts from the policy are collected, the policy is adapted by updating its parameters using vanilla policy gradient (steps 8 to 10). The whole process is repeated N times. Building on this, we will now describe the meta policy training procedure that provides the input meta policy for the policy adaptation.

## D. Training the meta-policy

The objective of training the meta policy is to find the optimal meta parameters θ m , which result in fast adaptation to new dynamic conditions. We propose a process similar to MAML which is illustrated in Figure 3 and outlined in Algorithm 2. The process starts with sampling a batch of tasks τ i ∼ p ( τ ) (step 3). Each task represents a new environment with different, randomized dynamics. For each of the environments, the agent starts with the meta policy and performs N adaptation steps, each using K rollouts from the policy, as described in Section III-C (step 5). After the last adaptation step, the agent collects K rollouts using the final adapted policy (step 6). This data is, in turn, used to update the final adapted policy in step 8. However, instead of directly updating the parameters of the final adapted policy, the gradients are backpropagated through all N update steps, all the way back to the parameters of the original meta policy. This update can be performed using any model free reinforcement learning algorithm.

Fig. 3: Overview of the meta training procedure, starting from the adapted policies for each environment.

<!-- image -->

This process results in a model that is explicitly optimized to maximize the expected cumulative return after N adaptation steps, in contrast to training a single universal policy to perform well on all tasks at the same time.

## Algorithm 2: Meta policy training

```
Input: Trained generator g φ Result: Meta policy parameters θ m 1 randomly initialize meta parameters θ m ; 2 while not converged do 3 repeat b := B times 4 Sample task τ b ; 5 perform N adaptation steps as in Algorithm 1 ; 6 collect K samples using π θ N ,b ; 7 end 8 Update θ m to improve performance of π θ N ,b for all τ b ; 9 end
```

## IV. EXPERIMENTAL EVALUATION

In this section, we describe the experimental setup and the architecture of our models, together with their training procedures and the description of a baseline method. Then, we present the adaptation results obtained in simulation and on a physical setup, comparing the performance of our method to the presented baseline.

Fig. 4: The hockey puck experimental setup (a) and the tools used for the experiments (b)

<!-- image -->

TABLE I: Randomized parameters

| Parameter                      | Minimum                      | Maximum   |
|--------------------------------|------------------------------|-----------|
| x linear friction ( µ x )      | 0.15                         | 0.95      |
| y linear friction ( µ y )      | 0.7 µ x                      | 1.3 µ x   |
| Torsional friction ( µ τ )     | 0.001                        | 0.05      |
| Rotational friction x ( µ rx ) | 0.01                         | 0.3       |
| Rotational friction y ( µ ry ) | 0.01                         | 0.3       |
| Puck mass                      | 50g                          | 500g      |
| Initial puck position          | glyph[epsilon1] ∼ N (0 , 0 . | 02)       |

## A. Experimental setup

The hockey puck setup consists of a KUKA LBR 4+ robot equipped with a floorball stick, as illustrated in Figure 4a. The robot uses the stick to hit a hockey puck on a flat, low friction surface, such that the puck lands in a target location. The friction between the surface and the puck has crucial impact on the movements of the puck, forcing the policy to learn how to operate in different friction conditions. We use two different hockey pucks, as shown in Figure 4b: an ice hockey puck (low friction; blue), and an inline hockey puck (higher friction; red). Since all contact points of the inline hockey puck are located close to the edge, it also has noticeably higher torsional friction than the ice hockey puck. The position of the pucks is measured by a camera mounted on the ceiling above the whiteboard. The target range of size 50cm x 30cm is located close to the center of the whiteboard.

We constructed a corresponding simulation setup in MuJoCo [4]. During training in simulation, we randomize the mass of the puck, the five friction parameters between the puck and the surface, and the starting position to account for possible misalignments between the real setup and the simulation. The randomization parameters are presented in Table I. We use uniform distributions for the dynamic parameters and a normal distribution for the starting position noise.

Each parameter of the system is randomized separately, except for µ y , which is randomized in relatively to µ x . The logic behind the use of anisotropic friction was based on an observation that under the same robot trajectories, puck movement directions on the physical setup are noticeably different from the simulation. We presume that this is caused by contact modeling inaccuracies and unmodeled effects such as the elasticity of the hockey blade and unevenness of the surface. Instead of fine-tuning the simulated behaviour Puck x Puck x to be closer to reality, we aimed to make up for these inaccuracies with additional randomizations.

<!-- image -->

Fig. 5: Relation between latent variable and the final position of the hockey puck after executing the corresponding trajectory. Different shades of blue represent values of the latent variables z 0 (left) and z 1 (right). The red cross represents the initial position of the puck.

## B. Training the generative trajectory model

To train the hockey puck trajectory generation model, we generated 7371 trajectories consisting of 17 waypoints of a cubic spline. These trajectories were obtained by moving the robot from the starting position to the proximity of the puck, making a swing, and moving the hockey blade past the puck. The strength of the swing and the orientation of the blade were randomized in order to generate a variety of trajectories with different hitting strengths and hitting angles.

These trajectory waypoints were then used to train the trajectory generation model. We used a 2 dimensional latent space throughout the experiments. Similarly to [8], we increased the value of β during training from 10 -7 to 10 -3 .

To evaluate the structure of the latent space, we sampled 2000 latent vectors from the latent distribution, executed the corresponding trajectory in the simulator and recorded the final position of the puck. The results are shown in Figure 5.

The figure illustrates that the model learned to disentangle the hitting angle ( z 0 ) from the hitting strength ( z 1 ). For example, as the value of z 0 increases, the generative model produces trajectories that hit the puck more and more towards the left side in a smooth and continuous manner.

## C. Policy training

The policy is trained in simulation using the simulated setup. We use K = 16 rollouts per update and train the policy for N = 3 adaptation steps. The policy is represented by a neural network with hidden layer of size 128. For meta learning, we learn the value of the adaptation step α during training. The dynamic parameter ranges are shown in Table I. We use proximal policy optimization (PPO) [29] as the meta optimization algorithm.

As a comparison baseline, we used domain randomization by training a policy by PPO with the same range of randomized dynamics, without adaptation occurring during training.

Fig. 6: Comparison of adaptation to different conditions in simulation between our method and the baseline.

<!-- image -->

TABLE II: Dynamic conditions used for simulation experiments

| Experiment      |   µ x |   µ y |   µ τ |   µ rx |   µ ry | m    |   glyph[epsilon1] x |   glyph[epsilon1] y |
|-----------------|-------|-------|-------|--------|--------|------|---------------------|---------------------|
| isotropic, low  |  0.15 |  0.15 |  0.01 |    0.1 |    0.1 | 110g |                 0.0 |                 0.0 |
| isotr. medium   |   0.4 |   0.4 |  0.01 |    0.1 |    0.1 | 110g |                 0.0 |                 0.0 |
| anisotr., low x |   0.2 |   0.8 |  0.01 |    0.1 |    0.1 | 110g |                 0.0 |                 0.0 |
| anisotr., low y |   0.8 |   0.2 |  0.01 |    0.1 |    0.1 | 110g |                 0.0 |                 0.0 |

To provide a fair comparison of the adaptability of initial policy parameters, we used the same adaptation step size α as the trained meta model ( α ≈ 0 . 02 ).

We use the following reward function proposed by [30]

<!-- formula-not-decoded -->

where d is the distance in meters between the final puck position and the target, and α is a constant (we use b = 10 -3 throughout the experiments). During policy training, the weights of the trajectory model remained fixed.

## D. Simulation experiments

We studied the proposed method against the baseline in simulation under a variety of dynamic conditions, repeating each experiment 25 times. To estimate the upper bound on performance for each condition, we also trained a policy for each individual one. This section illustrates the results of four experiments we consider to be the most unique and interesting: two isotropic friction cases (with low and medium friction) and two anisotropic friction cases with different low frictions directions. The dynamic parameters used for each of these experiments are shown in Table II, and the results of this evaluation are presented in Figure 6. Within the selected parameter ranges, changing the mass and the initial position did not have a significant impact on the adaptation performance.

Figure 6 illustrates that the domain randomization baseline is superior without adaptation, as is expected. However, the proposed method is consistently superior after some adaptation steps. This confirms our initial hypothesis that designing policies specifically for adaptation can increase the adaptation speed and thus help to address domain mismatch.

Surprisingly, the performance of the domain randomization baseline starts to deteriorate during adaptation in two cases. This most prominent in Fig. 6b where the deterioration begins already at the first update. This indicates that the domain randomized policy is somehow unsuitable for adaptation.

We analyzed this behaviour further by looking at the latent space action distributions of various repetitions of the experiment. These distributions are shown in Figure 7. The plots were generated by sampling latent actions for 1000 random goal points at each adaptation step. Different colours represent different repetitions of the experiment.

Before adaptation, at step 0, the policy takes the same actions during every repetition of the experiment. However, after a few adaptation steps, the baseline policies start to diverge from each other, as each of them is updated using an independent set of samples. Performing more adaptations causes these differences to escalate even more. This causes the policy parameters to shift from neighborhood of the origin where the trajectory generator is stable. This in turn is likely to produce more varying trajectories, making the reward gradient used by the updates less stable.

The proposed method does not exhibit such behaviour (Figure 7b). The latent updates follow each other more closely, even after 10 adaptations despite training the meta policy for only 3 adaptation steps. There are minor differences between the distributions due to random sampling of targets and trajectories but the distributions remain close to the origin and do not diverge. We hypothesize that is is due to the meta policy being explicitly trained to adapt to new conditions; it thus learned to perform stable and consistent update steps. To ensure that this behaviour is a repeatable phenomenon and was not caused by the baseline converging to a bad optimum, we repeated these entire experiments three times, achieving comparable results each time. Similar findings about regularizing effect of meta learning on policy training were previously reported by Clavera [27].

After achieving promising results and acquiring deeper understanding of the domain adaptation process with both methods in simulation, we moved on to physical experiments using the previously described experimental setup.

## E. Real-world experiments

We conducted the real world experiments using the setup described in Section IV-A. During each run, we conducted 4 adaptations in total. We evaluated each intermediate policy by taking its mean for 16 randomly chosen target points. We then sampled another K = 16 rollouts from the policy to perform an update. The experiment was repeated 3 times for each hockey puck, resulting in 48 data points for each puck at each adaptation step.

Fig. 7: Latent action distribution changes over multiple updates, compared across multiple repetitions of the experiment (with each repetition shown in different color). The baseline method provides inconsistent results and is sensitive to inaccuracies in collected samples, while meta learning produces consistent updates.

<!-- image -->

The performance comparison between the baseline and our method is shown in Figure 8. Consistently with the simulations, the domain randomization baseline (dashed line) is superior without adaptation. Moreover in line with the simulations, the baseline produces inconsistent behaviour during adaptation steps: some policy updates resulted in an overall improvement, while others made the performance deteriorate. As an extreme case, one of the experiments with the red puck had to be stopped, due to the policy mean landing far enough from the latent distribution such that the trajectory model produced unsuitable trajectories, which was confirmed by studying the latent distribution in a manner similar to the simulation experiments. There is a significant difference between the two pucks, with the performance for the blue puck increasing during the first adaptation steps. We hypothesize that this is because the low friction of the blue puck provides a stronger policy gradient direction (before adaptation, both policies hit the puck way too strongly, so simply reducing the hitting strength results in a significant increase in rewards). Nevertheless, additional adaptation steps cause the performance to deteriorate.

The policy trained with meta learning does not suffer from such issues, resulting in consistent adaptation. The performance either keeps improving or plateaus at a certain level. This is especially apparent for the higher friction red puck case, where the baseline completely fails to provide any performance improvement whatsoever. The overall variance is also noticeably smaller than in case of the baseline. Again, this behaviour is very similar to what was observed in simulation for the medium friction case.

Fig. 8: Comparison of real world performance of our method (left) and the baseline (right) with different pucks.

<!-- image -->

## V. CONCLUSIONS

In this work, we demonstrated that meta learning can be used as a stable and repeatable simulation-to-real domain adaptation tool. We observed that adapting a policy trained with standard domain randomization can cause diverse results, with high variance between repetitions and potentially unstable outcomes. Domain randomization suffered from these issues despite operating in a low dimensional and easy to explore action space. We also demonstrated that these issues do not exist when the policy is trained for stable adaptation with the proposed meta learning approach.

When describing the simulated setup, we briefly mentioned how we introduced anisotropic friction to the system to avoid fine tuning contact parameters and make up for unmodeled aspects of the physical setup. This poses an interesting avenue for further research: whether variations in some parameters can make up for modeling inaccuracies in other physical properties of the system.

In our experiments, we used 16 samples to perform each policy update. We believe that higher real world sample efficiency could be achieved, especially if the exploration was performed in a more arranged way. By using gradient based meta learning, we optimize the policy to achieve good performance based on the samples it currently gets, without giving it any incentive to produce useful samples. While this performed well in our case, further investigation into this could shed some light onto efficient and more informed exploration, potentially leading to higher sample efficiency.

This could potentially be done by off-policy adaptation, that is, gathering samples for policy adaptation from a separate exploration policy different from the adapted one. This would allow learning exploratory policies with safety constraints, which could then be used to collect informative samples for quick adaptation.

## REFERENCES

- [1] S. Levine, P. P. Sampedro, A. Krizhevsky, J. Ibarz, and D. Quillen, 'Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection,' 2017.
- [2] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller, 'Playing atari with deep reinforcement learning,' arXiv preprint arXiv:1312.5602 , 2013.
- [3] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, et al. , 'Human-level control through deep reinforcement learning,' Nature , vol. 518, no. 7540, p. 529, 2015.
- [4] E. Todorov, T. Erez, and Y. Tassa, 'Mujoco: A physics engine for model-based control,' 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems , pp. 5026-5033, 2012.
- [5] J. Liang, V. Makoviychuk, A. Handa, N. Chentanez, M. Macklin, and D. Fox, 'Gpu-accelerated robotic simulation for distributed reinforcement learning,' CoRR , vol. abs/1810.05762, 2018.
- [6] Y. Chebotar, A. Handa, V. Makoviychuk, M. Macklin, J. Issac, N. Ratliff, and D. Fox, 'Closing the sim-to-real loop: Adapting simulation randomization with real world experience,' arXiv preprint arXiv:1810.05687 , 2018.
- [7] J. Tan, T. Zhang, E. Coumans, A. Iscen, Y. Bai, D. Hafner, S. Bohez, and V. Vanhoucke, 'Sim-to-real: Learning agile locomotion for quadruped robots,' arXiv preprint arXiv:1804.10332 , 2018.
- [8] A. H¨ am¨ al¨ ainen, K. Arndt, A. Ghadirzadeh, and V. Kyrki, 'Affordance learning for end-to-end visuomotor robot control,' arXiv preprint arXiv:1903.04053 , 2019.
- [9] X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel, 'Sim-toreal transfer of robotic control with dynamics randomization,' in 2018 IEEE International Conference on Robotics and Automation (ICRA) , pp. 1-8, IEEE, 2018.
- [10] M. Hazara and V. Kyrki, 'Transferring generalizable motor primitives from simulation to real world,' IEEE Robotics and Automation Letters , vol. 4, pp. 2172-2179, April 2019.
- [11] J. Schmidhuber, 'Evolutionary principles in self-referential learning. on learning now to learn: The meta-meta-meta...-hook,' diploma thesis, Technische Universitat Munchen, Germany, 14 May 1987.
- [12] J. Schmidhuber, 'A neural network that embeds its own meta-levels,' in IEEE International Conference on Neural Networks , pp. 407-412 vol.1, March 1993.
- [13] J. Schmidhuber, J. Zhao, and N. N. Schraudolph, 'Learning to learn,' ch. Reinforcement Learning with Self-modifying Policies, pp. 293309, Norwell, MA, USA: Kluwer Academic Publishers, 1998.
- [14] S. Ravi and H. Larochelle, 'Optimization as a model for few-shot learning,' in ICLR , 2017.
- [15] Y. Duan, J. Schulman, X. Chen, P. L. Bartlett, I. Sutskever, and P. Abbeel, 'Rl2: Fast reinforcement learning via slow reinforcement learning,' ArXiv , vol. abs/1611.02779, 2017.
- [16] T. Miconi, K. O. Stanley, and J. Clune, 'Differentiable plasticity: training plastic neural networks with backpropagation,' in Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsm¨ assan, Stockholm, Sweden, July 10-15, 2018 , pp. 35563565, 2018.
- [17] C. Finn, P. Abbeel, and S. Levine, 'Model-agnostic meta-learning for fast adaptation of deep networks,' in Proceedings of the 34th International Conference on Machine Learning (D. Precup and Y. W. Teh, eds.), vol. 70 of Proceedings of Machine Learning Research , (International Convention Centre, Sydney, Australia), pp. 1126-1135, PMLR, 06-11 Aug 2017.
- [18] G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba, 'Openai gym,' CoRR , vol. abs/1606.01540, 2016.
- [19] A. Antoniou, H. A. Edwards, and A. J. Storkey, 'How to train your maml,' ArXiv , vol. abs/1810.09502, 2018.
- [20] B. C. Stadie, G. Yang, R. Houthooft, X. Chen, Y. Duan, Y. Wu, P. Abbeel, and I. Sutskever, 'Some considerations on learning to explore via meta-reinforcement learning,' CoRR , vol. abs/1803.01118, 2018.
- [21] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, 'Domain randomization for transferring deep neural networks from simulation to the real world,' in 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , pp. 23-30, IEEE, 2017.
- [22] F. Sadeghi and S. Levine, '(cad)$ˆ2$rl: Real single-image flight without a single real image,' CoRR , vol. abs/1611.04201, 2016.
- [23] F. Sadeghi, A. Toshev, E. Jang, and S. Levine, 'Sim2real view invariant visual servoing by recurrent control,' CoRR , vol. abs/1712.07642, 2017.
- [24] OpenAI, M. Andrychowicz, B. Baker, M. Chociej, R. J´ ozefowicz, B. McGrew, J. W. Pachocki, A. Petron, M. Plappert, G. Powell, A. Ray, J. Schneider, S. Sidor, J. Tobin, P. Welinder, L. Weng, and W. Zaremba, 'Learning dexterous in-hand manipulation,' ArXiv , vol. abs/1808.00177, 2018.
- [25] V. Petr´ ık and V. Kyrki, 'Feedback-based fabric strip folding,' CoRR , vol. abs/1904.01298, 2019.
- [26] M. Wulfmeier, I. Posner, and P. Abbeel, 'Mutual alignment transfer learning,' in 1st Annual Conference on Robot Learning, CoRL 2017, Mountain View, California, USA, November 13-15, 2017, Proceedings , pp. 281-290, 2017.
- [27] I. Clavera, J. Rothfuss, J. Schulman, Y. Fujita, T. Asfour, and P. Abbeel, 'Model-based reinforcement learning via meta-policy optimization,' arXiv preprint arXiv:1809.05214 , 2018.
- [28] A. Ghadirzadeh, A. Maki, D. Kragic, and M. Bjrkman, 'Deep predictive policy training using reinforcement learning,' 03 2017.
- [29] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, 'Proximal policy optimization algorithms,' ArXiv , vol. abs/1707.06347, 2017.
- [30] S. Levine, C. Finn, T. Darrell, and P. Abbeel, 'End-to-end training of deep visuomotor policies,' J. Mach. Learn. Res. , vol. 17, pp. 13341373, Jan. 2016.