## Learning to Scaffold the Development of Robotic Manipulation Skills

Lin Shao, Toki Migimatsu and Jeannette Bohg

Abstract -Learning contact-rich, robotic manipulation skills is a challenging problem due to the high-dimensionality of the state and action space as well as uncertainty from noisy sensors and inaccurate motor control. To combat these factors and achieve more robust manipulation, humans actively exploit contact constraints in the environment. By adopting a similar strategy, robots can also achieve more robust manipulation. In this paper, we enable a robot to autonomously modify its environment and thereby discover how to ease manipulation skill learning. Specifically, we provide the robot with fixtures that it can freely place within the environment. These fixtures provide hard constraints that limit the outcome of robot actions. Thereby, they funnel uncertainty from perception and motor control and scaffold manipulation skill learning. We propose a learning system that consists of two learning loops. In the outer loop, the robot positions the fixture in the workspace. In the inner loop, the robot learns a manipulation skill and after a fixed number of episodes, returns the reward to the outer loop. Thereby, the robot is incentivised to place the fixture such that the inner loop quickly achieves a high reward. We demonstrate our framework both in simulation and in the real world on three tasks: peg insertion, wrench manipulation and shallowdepth insertion. We show that manipulation skill learning is dramatically sped up through this way of scaffolding.

## I. INTRODUCTION

A hallmark in robotics research is the autonomous learning of skills. If achieved, this ability would make the deployment of robots more flexible by alleviating the need to adapt the environment to match the limited abilities of the robot. The challenges towards this goal are the high dimensionality of the state and action space as well as uncertainty from perception and motor control. Deimel et al. [1] have shown that humans exploit contact constraints in the environment to combat these effects. The environment provides physical constraints on an action and funnels uncertainty due to noise in perception and control. There also has been work that shows how this strategy makes robotic manipulation more robust [1-6]. For example, fixtures is a widely used practice in industry for various application such as machining, assembly and inspection. The principles and designs of fixtures have been extensively studied [7, 8].

However, these works typically consider only a limited set of constraints and predefine how they can be exploited.

All authors are with the Stanford Artificial Intelligence Lab (SAIL), Stanford University, CA, USA. [lins2,takatoki,bohg] @stanford.edu In this paper, we enable a robot to actively modify its environment and discover different ways in which constraints can be exploited. Specifically, we provide the robot with fixtures that it can freely place in the workspace to act as physical scaffolding for manipulation skill learning. In this way, our learning framework mirrors educational scaffolding used in human learning. The proposed approach consists of two learning loops: an outer loop places a fixture in the workspace, and then an inner loop attempts to learn a manipulation skill through reinforcement learning. The reward achieved by the inner loop is returned to the outer loop so that the outer loop can optimise the fixture placement for the next trial. In this way, the outer loop has an incentive to optimise the fixture such that the inner loop achieves a high reward quickly.

This work has been partially supported by JD.com American Technologies Corporation ('JD') under the SAIL-JD AI Research Initiative. This article solely reflects the opinions and conclusions of its authors and not JD or any entity associated with JD.com. Toyota Research Institute ('TRI') provided funds to assist the authors with their research but this article solely reflects the opinions and conclusions of its authors and not TRI or any other Toyota entity.

Fig. 1. Our proposed learning framework enables robots to autonomously use fixtures to aid with manipulation skill learning. The top image shows our experimental setup with one manipulator arm to position the fixture and another to complete the manipulation task. The bottom images compare the learning curves of the manipulation task without any fixture (left), with a fixture (middle), and with a virtual potential field to replace the fixture as the fixture is gradually moved away (right). The fixture dramatically improves the robot's ability to learn the task.

<!-- image -->

Our primary contributions are: 1) We propose a learning framework for robots to leverage fixtures to assist with complex manipulation tasks. 2) We introduce an algorithm that improves sample efficiency in bandit problems with continuous action spaces and discontinuous reward functions. 3) We demonstrate that our method dramatically improves robot performance on three challenging manipulation tasks (peg insertion, wrench manipulation, and shallow-depth insertion [9]) both in simulation and the real world. 4) We propose a method that allows the robot to maintain task performance while gradually removing the fixtures from the environment.

## II. RELATED WORK

## A. Learning to Leverage the Environment

Exploiting contacts and environmental constraints to reduce uncertainty during manipulation has received many researchers' attentions. Kazemi et al. [2] show that contact with support surfaces is critical for grasping small objects and design a closed-loop hybrid controller that mimics the pre-grasp and landing strategy for finger placement. Deimel et al. [1] demonstrate how humans produce robust grasps by exploiting constraints in the environment. Righetti et al. [3] propose an architecture to utilize contact interactions for force/torque control and optimization-based motion planning in grasping and manipulation tasks. Hudson et al. [4] present a model-based approach to improve system knowledge about the combined robot and environmental state through deliberate interactions with the objects and the environment. Toussaint et al. [5] integrate the idea of exploiting contacts with trajectory optimizations.

In all the above works, robots interact with the environment using a limited set of constraints and a predefined way to exploit them. In contrast, we propose a data-driven learning framework for robots to automatically discover and learn how to change and leverage the environment with selfsupervision. Our framework not only allows robots to exploit environmental constraints passively but also to actively create new constraints using fixtures to generate 'funnels' for optimal learning convergence. After learning how to perform the manipulation tasks with fixtures, our framework is also able to learn how to maintain task performance with the fixtures removes.

## B. Learning Robotic Manipulation Skills

Our framework can be adopted to improve the performance of a range of manipulation tasks including robotic assembly, tool manipulation, in-hand manipulation, and regrasping. We review the related work in these domains.

1) Insertion: Inoue et al. [10] separate the peg-in-hole process into two stages, searching and inserting, and use LSTMs to learn separate policies for each one. Thomas et al. [11] combine reinforcement learning with a motion planner to shape the state cost in a high-precision setting. Luo et al. [12] utilize deep reinforcement learning to learn variable impedance controllers for assembly tasks. Lee et al. [13] combine both vision and touch to learn a policy for insertion.

2) Tool Manipulation: Zhu et al. [14] consider handheld physical tools like hammers and utilize RGB-D images to identify functional and affordance areas of the tools. Fang et al. [15] learn task-oriented grasps for tools. In our work, we assume that the tool is already grasped and focus on discovering how to exploit fixtures for scaffolding the skill learning process.

3) In-hand Manipulation and Re-Grasping: Chavan-Dafle and Rodriguez [6] explore the manipulation of a grasped object by pushing it against the environment. Kim and Seo [9] use in-hand manipulation to insert a flat object into a hole with a shallow depth e.g. a battery into a mobile phone.

Unlike all the approaches mentioned above, we focus on actively changing the environment to support manipulation skill learning. These approaches can be placed in the inner loop of our learning framework to improve the performance of each individual manipulation task. Our outer loop would then learn how to use fixtures to improve the performance of the inner loop policies.

## C. Contextual Bandits with Continuous Action and Discontinuous Reward

We formulate the problem of optimally placing the fixture as a contextual bandits problem with a continuous action space and discontinuous reward. The aim is to maximize the cumulative rewards over a series of trials. For these types of bandit problems, prior work typically assumes that nearby actions have similar rewards. For example, the rewards are assumed to be Lipschitz continuous as a function of the actions. Kleinberg et al. [16] propose the zooming algorithm to adaptively discretize and sample within the continuous action space. However, in our case and in many other real-world applications, the function which maps actions to rewards is discontinuous. In the peg-in-hole task, for example, if the fixture is precisely aligned with the boundary of the hole as shown in Fig. 4, then the robot can successfully insert the peg into the hole and achieve a high reward. However, if the fixture is slightly moved such that it blocks the hole, the robot suddenly receives zero reward. Krishnamurthy et al. [17] address this challenge of discontinuity in the actionto-reward function by mapping each discrete action into a well-behaved distribution with no continuity assumptions. However, their proposed method, which is based on policy elimination [18], is not suitable for high dimensional spaces like images. Inspired by the ideas of adaptive zooming [16] and smoothing [17], we present a novel Smoothed Zooming algorithm to solve high dimensional contextual bandits with continuous actions and a discontinuous reward.

## III. PROBLEM DEFINITION

In this work, we want to enable a robot to autonomously alter its environment to ease manipulation skill learning. This requires the robot to find an optimal placement of a fixture and to learn a manipulation policy that exploits this fixture.

Given an observation of a manipulation scene, e.g. an image denoted as s ∈ I , the robot has to propose an optimal fixture pose f ∗ ∈ F . Given f ∗ , the robot then learns a policy π ∗ ∈ Π that maximizes the expected discounted rewards ∑ T t =1 γ t -1 R t where γ is the discounted factor. This can be written as:

<!-- formula-not-decoded -->

In this work, we also consider the problem of continuing to update the policy while the fixture is gradually removed. This could be considered as a physical curriculum for learning manipulation skills, similar to training wheels for learning how to bike. Formally, the robot has to learn a second optimal policy π ∗∗ ∈ Π without the fixture but starting from the previous policy π ∗ :

<!-- formula-not-decoded -->

## IV. TECHNICAL APPROACH

We propose a learning system that consists of two learning loops. We use Reinforcement Learning (RL) for the outerloop fixture pose selection and for the inner-loop robotic skill learning. We also use RL for learning a policy when gradually removing the fixture. In the following, s ∈ S denotes the state, which in our case is either a depth or an RGB image. a ∈ A denotes the action, which corresponds to the fixture pose in our outer-loop process or the endeffector motion in the inner-loop process. At each time step t , our robot perceives the state s t and chooses an action a t . It receives a reward R t and moves to a new state s t +1 . The goal is to learn policies that maximize the cumulative reward.

## A. Fixture Pose Selection

The pose of a fixture is parameterized the vector a f . The robot perceives the manipulation scene through a top-view depth camera. s f 0 is the initial image and provides contextual information about the scene. The robot then selects a pose a f of the fixture and places the fixture in the workspace. The robot then starts the inner-loop learning process. After a fixed number of inner-loop episodes, the outer-loop receives a reward R f representing the inner-loop learning performance given the fixture pose a f . To maximise R f , the outer-loop needs to select the fixture pose such that the inner-loop masters the manipulation skill.

Because the contextual information in our problem is the initial depth image, we use a convolutional neural network model to capture the contextual features. Similar to Qlearning, we approximate the reward distribution given the contextual information. Let Q ( s f 0 , a f ) be the expected reward the robot receives after it sees the contextual information s f 0 and takes the action a f . We use a deep convolutional network denoted as Q w ( s f 0 , a f ) to approximate Q ( s f 0 , a f ) . We aim to train our model by minimizing the following loss:

<!-- formula-not-decoded -->

At test time, given a depth image s f , the optimal policy is π ( s f ) = arg max a f Q w ( s f , a f ) . Similar to Kalashnikov et al. [19], we adopt the cross-entropy method (CEM) [20]

to perform the optimization with respect to action a f . We first sample a batch of M points in action space A and fit a Gaussian distribution based on the top N samples. Given the fitted Gaussian, we repeated this sampling process six times.

At training time, we aim to maximize the rewards after the robot perceives the contextual information. We propose a revised algorithm called Smoothed Zooming Algorithm shown in Alg.1. We first cover the continuous action space with a countable number of small covering balls with the radius of h . We adopt the commonly used UCB1 [21] rule to select one covering ball. The fixture pose itself is uniformly sampled within the corresponding covering ball. In this way, we are able to smooth the discontinuous reward function [17]. Each covering ball maintains the average reward ¯ µ t and the number of sampled points n t sampled from itself. As the number of sampled points increases, the radius of the ball r t shrinks and may lead to new exposure areas in the action space. We initialize new covering balls to cover these new exposure areas to ensure that these balls always cover the continuous action space. We repeat updating the center and radius of these covering balls in each round.

## Algorithm 1 Smoothed Zooming Algorithm

Initial radius of covering ball h &gt; 0, number of rounds T f , Active Arm Set S , Arm x ∈ X continuous action space. S = Set { ⌈ x h ⌉ } , Activate these arms lying on the discrete mesh, ¯ µ 0 ( x ) x ∈ S = 0

- 1: for t in each round do

2:

¯

x

= arg max

x

∈

S

[¯

µ

t

-

1

(

x

) + 2

r

t

-

1

- 3: a t ∼ Uniform ( B t -1 (¯ x ))

<!-- formula-not-decoded -->

r

′

2 log

T

t

(

x

) =

√

n

(

x

f

)+1

B

t

(

x

) =

{

′

t

y

∈

Confidence Ball

<!-- formula-not-decoded -->

- 10: Activation: If some arm is not covered, pick any such arm and activate it.
- 11: end for

## B. Robotic Skill Learning with Fixtures

We formulate the inner learning loop for robot manipulation skills as an episodic RL problem. We denote T as the maximum number of steps within each episode and γ &lt; 1 as the discount factor. At each time step t ≤ T , the robot receives an RGB image s t of the manipulation scene and performs an action a t at the end-effector. The robot moves to a new state s t +1 and receives a reward R t from the environment. The episode ends either when the task is finished successfully or it reaches the maximum time step T . In each individual task, the robot learns a policy π ( s t ) and aims to maximize the cumulative reward. We use a variant of the A3C algorithm [22] to learn the optimal policy

X

:

5:

6:

D

(

x, y

)

≤

r

′

t

(

x

(

x

}

)

)]

glyph[triangleright]

Selection

glyph[triangleright]

Update in simulation. We use operational space control [23] for executing the devised end-effector motion and to decouple the constrained motion induced by the fixture and active force control in the manipulation task.

Fig. 2. Overall Approach. The outer loop (blue arrows) takes a depth image of the manipulation scene and the learned policy proposes a fixture pose. After the fixture has been placed, the inner loop (in green dashed box) trains an RL policy that maps RGB images to end-effector motions to complete the manipulation task. The cumulative reward is sent to the outer loop to optimize the fixture placement. The goal of the outer loop is to maximize the learning rate and performance of the inner loop.

<!-- image -->

## C. Robotic Skill Learning while Removing Fixtures

When the robot learns a policy π ∗ to finish the task successfully with the help of the fixture, we also want the robot to maintain this skill after the fixture is removed. To achieve this goal, we replace the hard constraints produced by the physical fixtures with soft constraints produced by a virtual potential field. Note that in the outer-loop stage, the robot perceives the manipulation scene based on a top-view depth camera. Given the depth image from that camera, we are able to infer the physical geometry of the fixture. Then we make the potential field to maintain the same geometry as the fixtures. We start to gradually remove the fixture away from the manipulation scene horizontally. Meanwhile, in our inner-loop process, we continue to train the policy starting from π ∗ . So we can transfer the manipulation skills with physical fixtures to skills with virtual potential fields automatically. For the inner-loop process, we maintain the same experiment setting. The removal speed is a hyperparameter and may vary from different tasks.

## V. EXPERIMENTS

We execute a series of experiments to test the proposed learning framework both in simulation and the real world. We investigate the following questions: 1) Do fixtures speed up manipulation skill learning? 2) Is the fixture pose selection policy reasonable? 3) Can the robot retain the learned manipulation skill after gradually removing the fixture? 4) How robust is the framework to task variations and perturbations?

## A. Robotic Experiment Setup

We use a 7-DoF Franka Panda robot arm with a twofingered Robotiq 2F-85 gripper to perform the inner-loop manipulation tasks both in simulation and in the real world. We use PyBullet [24] as the simulation environment. An RGB-D camera is statically mounted in the environment to capture RGB images downsampled to 120 × 160 for the innerloop manipulation tasks. The camera parameters are the same in simulation and the real world.

For the outer-loop fixture pose selection, we directly control the fixture pose in simulation and fix it at the given pose. In the real-world experiment, we use a 7-DoF Kinova Gen3 arm with a Robotiq 2F-85 gripper to position and firmly hold the fixture. In both simulation and the real world, we use a wrist-mounted RGB-D camera to obtain depth images downsampled to 60 × 80 for the outer-loop task.

We design three different manipulation tasks as shown in Fig. 5: inserting a peg into a hole (denoted as Insertion ), tightening a bolt using a wrench ( Wrench ) and inserting a thin cuboid into a shallow slot [9] ( SD Insertion ).

The actions are described in the robot base frame. In the peg-insertion task, we set the inner-loop action a ∈ R 3 to be the relative translation motion of the end-effector. The fixture pose is also a three dimensional vector a f = ( x, y, θ ) representing the fixture's 2D position and orientation in the plane perpendicular to the axis of the hole. In the wrench manipulation task, the inner-loop action is a 6D relative motion vector ( dx, dy, dz, dα, dβ, dγ ) , and the outer-loop action is the 3D position ( x, y, z ) of the fixture. For the shallow-depth insertion [9] task, the inner-loop action is the relative 2D translation and orientation of the end-effector in the plane perpendicular to the axis along the short width of the hole.

The fixture in the peg-insertion task helps to prevent the peg from slipping past the hole. In the wrench manipulation task, the fixture provides support for the wrench to stay horizontal and apply a continuous torque on the bolt. In the shallow-depth insertion [9] task, the fixture restricts the translation and rotation motion of the cuboid into the slot when the gripper releases and pulls away from the cuboid.

## B. Evaluation Metrics and Reward Design

The peg-insertion task is considered to be successful if the peg is completely inserted into the hole. The wrench manipulation task is successful if the bolt is rotated by 30 degrees. The shallow-depth insertion task successful if the cuboid is horizontal and inside the shallow slot. We report the success rates for the peg-insertion and shallow-depth insertion tasks, and the turning rate of the bolt for the wrench manipulation task.

Fig. 3. Learning curves for the three different tasks, along with visualizations of their simulation environments, both with and without the (red) fixtures. Success for peg insertion (left) and shallow-depth insertion (right) means completing the task before the end of the episode. The suboptimal fixture pose is set to be one cm away from the optimal pose.

<!-- image -->

We have the following task-specific reward functions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(5)

<!-- formula-not-decoded -->

where η and ζ represent the rotation degree of the bolt and battery.

## C. Task Performance Evaluation with and without Fixtures

In this simulation experiment, we evaluate whether optimally-positioned physical fixtures speed up manipulation skill learning. During training, we randomize the configuration of the absolute object position and the arm's initial position at the beginning of each episode.

We illustrate the training curves in Fig. 3. We train the polices with 40k, 17.5k, and 17.5k steps for the insertion, wrench, and shallow-insertion tasks, respectively, and save the trained models at 500, 100, and 100 steps, respectively. Then, we evaluate the trained models for 5k steps on three versions of the task: one with the fixture at the optimal pose, one with the fixture at a suboptimal pose, and one with no fixture at all. From Fig. 3, we observe that a proper fixture significantly improves both task performance and learning speed for all three tasks. For the insertion task, the agent achieves a nearly 80% success rate with the optimal fixture pose but can only achieve 20% without a fixture. In order to achieve a 20% success rate, the agent needs to run 40k steps without a fixture but only 5k steps with an optimal fixture pose. Wrench manipulation and shallow-depth insertion are challenging manipulation tasks. We observes that the robots can only achieve reasonable progress with the help of fixtures.

## D. Fixture Pose Selection and Visualization

We visualize the learned Q map for the insertion task in Fig. 4. Note that the outer-loop action is parameterized by a three-dimensional vector ( x, y, θ ) representing horizontal translation and rotation in the plane perpendicular to the hole. From left to right, each image represents the Q-function where the fixture is set to be ( u, v, -15 ◦ ) , ( u, v, 0 ◦ ) and ( u, v, 15 ◦ ) . Each pixel u, v corresponds to an x, y position of the fixture and is colored according to the predicted reward - the darker, the higher. We also visualize for some pixels what the corresponding fixture pose is, thereby giving an intuitive understanding of the reward.

The three images indicate that our Q function learns a reasonable policy for the fixture pose selection. As expected, the highest Q value corresponds to the optimal pose where the fixture is lined up with the hole. The continuous gradual change up to the optimal pose allows CEM [20] to find it.

## E. Task Performance Evaluation with Fixture Removal

After training the agent on the peg-in-hole task with a fixture, we also want to maintain its task performance while gradually removing the fixture. In the baseline method, we continue to train the agent while gradually removing the fixture away from the hole by 1cm every 2k steps. We compare this to a method where we similarly remove the physical fixture, but also keep a soft constraint at the original location of the physical fixture in the form of a repulsive potential field (see Fig. 5). We find that with the baseline method, performance initially drops to 30%, but drops further to around 10% as the fixture moves farther away. With the potential field, the performance still drops to 30% initially, but climbs back up to 70% when the fixture is fully removed as it learns to handle the absence of the physical fixture.

Fig. 4. Visualization of the Q-function for the policy of the outer loop. Each pixel in the three images corresponds to a x, y fixture position with an orientation of either -15 , 0 or 15 degrees. The color corresponds to the predicted reward where darker means higher.

<!-- image -->

Fig. 5. Performance on peg-in-hole task with fixture removal.

<!-- image -->

## F. Demonstrating Generalization Ability

We directly test the simulation-trained policy in the real world. We utilize RGB images to segment objects and use these segmentation masks to improve the quality of depth images. To reduce the domain gap between simulation and the real world, we utilize domain randomization [25] approaches and randomize object positions, textures, and lighting conditions. The real world experiment setting is shown in Fig. 2. Snapshots of task performance are shown in Figs. 6,7 and 8. For each task, we compare the robustness of different methods by changing the positions of objects in different trials. The results are presented in Fig. 9. The models trained without fixtures fail in all tests, reflecting the difficulty of these manipulation tasks. The models trained with fixtures successfully finish almost every trial for each Fig. 7.

<!-- image -->

Fig. 6. Insert a round peg in round hole

<!-- image -->

Rotate a bolt using a wrench

Fig. 8. Rotate and insert a thin, flat cuboid into a shallow hole

<!-- image -->

task, if not all. We also compare the models trained on fixture removal (Sec. V-E) for the peg-insertion task. The baseline fails to insert the peg into the hole, while our model trained with the potential field is able to maintain the skill. We also test the generalization of our proposed method over different peg geometry, as shown in the accompanying video.

Fig. 9. Comparison of success rates for different tasks. w/o Fixture refers to policies learned completely without fixtures. w/ Fixture refers to our proposed method utilizing fixtures. -Fixtures refers to the fixture removal baseline in Sec. V-E. -Fixtures+PF refers to our proposed fixture removal method with potential fields method in Sec. V-E.

| Method      | Insertion   | Wrench   | SD Insertion   |
|-------------|-------------|----------|----------------|
| w/o Fixture | 0/5         | 0/5      | 0/5            |
| w/ Fixture  | 5/5         | 5/5      | 4/5            |
| -Fixture    | 1/5         | -        | -              |
| -Fixture+PF | 5/5         | -        | -              |

## VI. CONCLUSION

Automatically learning robotic manipulation skills is a challenging problem due to high-dimensional state and action spaces, noisy sensors, and inaccurate motor control. We proposed a learning framework which enables a robot to autonomously change its environment and discover how to expedite manipulation skill learning. We provide the robot with fixtures that it can freely place within the environment to generate motion constraints that limit the outcome of robot actions. The fixtures funnel uncertainty from perception and motor control and scaffold manipulation skill learning. We show in simulation and in the real world that our framework dramatically speeds up robotic skills on three tasks: peg insertion, wrench manipulation, and shallow-depth insertion [9].

## REFERENCES

- [1] R. Deimel, C. Eppner, J. ´ Alvarez-Ruiz, M. Maertens, and O. Brock, 'Exploitation of environmental constraints in human and robotic grasping,' in Robotics Research . Springer, 2016, pp. 393-409.
- [2] M. Kazemi, J.-S. Valois, J. A. Bagnell, and N. Pollard, 'Robust object grasping using force compliant motion primitives.'
- [3] L. Righetti, M. Kalakrishnan, P. Pastor, J. Binney, J. Kelly, R. C. Voorhies, G. S. Sukhatme, and S. Schaal, 'An autonomous manipulation system based on force control and optimization,' Autonomous Robots , vol. 36, no. 1-2, pp. 1130, 2014.
- [4] N. Hudson, T. Howard, J. Ma, A. Jain, M. Bajracharya, S. Myint, C. Kuo, L. Matthies, P. Backes, P. Hebert et al. , 'End-to-end dexterous manipulation with deliberate interactive estimation,' in 2012 IEEE International Conference on Robotics and Automation . IEEE, 2012, pp. 2371-2378.
- [5] M. Toussaint, N. Ratliff, J. Bohg, L. Righetti, P. Englert, and S. Schaal, 'Dual execution of optimized contact interaction trajectories,' in 2014 IEEE/RSJ International Conference on Intelligent Robots and Systems . IEEE, 2014, pp. 47-54.
- [6] N. Chavan-Dafle and A. Rodriguez, 'Prehensile pushing: Inhand manipulation with push-primitives,' in 2015 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) . IEEE, pp. 6215-6222.
- [7] H. Asada and A. By, 'Kinematic analysis of workpart fixturing for flexible assembly with automatically reconfigurable fixtures,' IEEE Journal on Robotics and Automation , vol. 1, no. 2, pp. 86-94, 1985.
- [8] Y.-C. Chou, V. Chandru, and M. M. Barash, 'A mathematical approach to automatic configuration of machining fixtures: analysis and synthesis,' Journal of Engineering for Industry , vol. 111, no. 4, pp. 299-306, 1989.
- [9] C. H. Kim and J. Seo, 'Shallow-depth insertion: Peg in shallow hole through robotic in-hand manipulation,' IEEE Robotics and Automation Letters , vol. 4, no. 2, pp. 383-390, 2019.
- [10] T. Inoue, G. De Magistris, A. Munawar, T. Yokoya, and R. Tachibana, 'Deep reinforcement learning for high precision assembly tasks,' in 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) . IEEE, 2017, pp. 819-825.
- [11] G. Thomas, M. Chien, A. Tamar, J. A. Ojea, and P. Abbeel, 'Learning robotic assembly from cad,' in 2018 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2018, pp. 1-9.
- [12] J. Luo, E. Solowjow, C. Wen, J. A. Ojea, A. M. Agogino, A. Tamar, and P. Abbeel, 'Reinforcement learning on variable impedance controller for high-precision robotic assembly,' arXiv preprint arXiv:1903.01066 , 2019.
- [13] M. A. Lee, Y. Zhu, K. Srinivasan, P. Shah, S. Savarese, L. FeiFei, A. Garg, and J. Bohg, 'Making sense of vision and touch: Self-supervised learning of multimodal representations for contact-rich tasks,' in 2019 International Conference on Robotics and Automation (ICRA) . IEEE, 2019, pp. 89438950.
- [14] Y. Zhu, Y. Zhao, and S. Chun Zhu, 'Understanding tools: Task-oriented object modeling, learning and recognition,' in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , 2015, pp. 2855-2864.
- [15] K. Fang, Y. Zhu, A. Garg, A. Kurenkov, V. Mehta, L. FeiFei, and S. Savarese, 'Learning task-oriented grasping for tool manipulation from simulated self-supervision,' arXiv preprint arXiv:1806.09266 , 2018.
- [16] R. Kleinberg, A. Slivkins, and E. Upfal, 'Bandits and experts in metric spaces,' arXiv preprint arXiv:1312.1277 , 2013.
- [17] A. Krishnamurthy, J. Langford, A. Slivkins, and C. Zhang,

'Contextual bandits with continuous actions: Smoothing, zooming, and adapting,' arXiv preprint arXiv:1902.01520 , 2019.

- [18] M. Dudik, D. Hsu, S. Kale, N. Karampatziakis, J. Langford, L. Reyzin, and T. Zhang, 'Efficient optimal learning for contextual bandits,' arXiv preprint arXiv:1106.2369 , 2011.
- [19] D. Kalashnikov, A. Irpan, P. Pastor, J. Ibarz, A. Herzog, E. Jang, D. Quillen, E. Holly, M. Kalakrishnan, V. Vanhoucke et al. , 'Scalable deep reinforcement learning for vision-based robotic manipulation,' in Conference on Robot Learning , 2018, pp. 651-673.
- [20] R. Y. Rubinstein and D. P. Kroese, 'The cross-entropy method,' in Information Science and Statistics , 2004.
- [21] P. Auer, N. Cesa-Bianchi, and P. Fischer, 'Finite-time analysis of the multiarmed bandit problem,' Machine learning , vol. 47, no. 2-3, pp. 235-256, 2002.
- [22] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu, 'Asynchronous methods for deep reinforcement learning,' in International conference on machine learning , 2016, pp. 1928-1937.
- [23] O. Khatib, 'A unified approach for motion and force control of robot manipulators: The operational space formulation,' IEEE Journal on Robotics and Automation , vol. 3, no. 1, pp. 43-53, 1987.
- [24] E. Coumans and Y. Bai, 'Pybullet, a python module for physics simulation for games, robotics and machine learning,' http://pybullet.org, 2016-2019.
- [25] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, 'Domain randomization for transferring deep neural networks from simulation to the real world,' in 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) . IEEE, 2017, pp. 23-30.