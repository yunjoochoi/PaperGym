## Impedance Primitive-augmented Hierarchical Reinforcement Learning for Sequential Tasks

Amin Berjaoui Tahmaz † , Ravi Prakash ‡ , Jens Kober †

Abstract -This paper presents an Impedance Primitiveaugmented hierarchical reinforcement learning framework for efficient robotic manipulation in sequential contact tasks. We leverage this hierarchical structure to sequentially execute behavior primitives with variable stiffness control capabilities for contact tasks. Our proposed approach relies on three key components: an action space enabling variable stiffness control, an adaptive stiffness controller for dynamic stiffness adjustments during primitive execution, and affordance coupling for efficient exploration while encouraging compliance. Through comprehensive training and evaluation, our framework learns efficient stiffness control capabilities and demonstrates improvements in learning efficiency, compositionality in primitive selection, and success rates compared to the state-of-the-art. The training environments include block lifting, door opening, object pushing, and surface cleaning. Real world evaluations further confirm the framework's sim2real capability. This work lays the foundation for more adaptive and versatile robotic manipulation systems, with potential applications in more complex contact-based tasks.

## I. INTRODUCTION

Realistic manipulation tasks involve a prolonged sequence of motor skills in varying environments. For decades, the challenge of enabling robotic manipulators to solve realistic long-horizon tasks has persisted. While existing research has made strides in addressing important aspects of longhorizon tasks, a critical gap remains in the context of contactrich environments, highlighting a crucial area that requires further exploration and development. An example can be found in a common manipulation task: object sorting. A robot should be able to plan a series of precise actions over time while adjusting its positioning and applied forces to accommodate objects of varying shapes and sizes while also taking the interaction environment into consideration. This paper focuses on the intersection of deep reinforcement learning (DRL) and adaptive stiffness control to address this longstanding challenge.

Prior works have extensively explored robotic manipulation in long-horizon applications. Conventional methods often use state machines [1] [2] or symbolic reasoning [3] [4] to learn action sequences for solving a task. However, these approaches explicitly design the decision-making sequence, which may introduce constraints that limit adaptability to different tasks and contribute to error accumulation throughout the task sequence. In response to these limitations, learning techniques such as hierarchical reinforcement learning (HRL) [5] have been employed, establishing themselves

Authors † are with TU Delft, Netherlands. Author ‡ is with IISc Bangalore, India. amine.berjawi123@gmail.com, ravipr@iisc.ac.in, j.kober@tudelft.nl as a common approach for problems requiring sequential decision-making.

Fig. 1. Figure shows the augmentation of the impedance primitive into HRL policy.

<!-- image -->

When deploying long-horizon frameworks in contact-rich environments, the integration of stiffness control becomes crucial for adapting to external forces and uncertainties during task execution. This adaptability ensures precision and stability in navigating contact-rich environments. However, despite a substantial body of research dedicated to variable stiffness control, current approaches are primarily tailored to short-horizon applications. These methods typically involve designing controllers that adjust end-point force in response to environmental forces [6], adapting impedance and damping parameters through learning techniques [7] [8], and learning from a human demonstrator [9] [10].

This paper aims to bridge the gap between sequential task planning and adaptive stiffness control using a DRL framework. We design an HRL framework, as shown in Figure 1, that selects a high-level action primitive from a predefined library and outputs an initial estimate for controller parameters for low-level control. During primitive execution, an adaptive controller is initiated to optimize the robot's stiffness, aiming for an balance between safety (reducing interaction forces with the environment) and performance (ensuring task completion). This design allows the robot to dynamically optimize stiffness parameters, enabling it to transition between high stiffness for precision tasks and increased compliance for enhanced adaptability. We present experiments conducted in both simulation and the real world, focusing on sequential tasks that deal with different contact challenges. Our results highlight notable advantages when compared to a state-of-the-art baseline.

The remainder of this paper is structured as follows: Sec- tion II discusses related work. Section III defines the problem statement. Section IV provides necessary background and preliminaries. Our proposed Impedance Primitive-augmented HRL (IMP-HRL) approach is introduced in Section V. Section VI presents experimental results and analysis. Section VII concludes the paper with a summary of findings and future research directions.

## II. RELATED WORKS

Sequential Planning: Extensive work in task and motion planning (TAMP) spans various robotics applications, involving explicit decision-making frameworks and machine learning for learned behavior sequences. Common approaches employ hierarchical task planning, combining high-level planners with low-level controllers. In robot manipulation, this often takes the form of finite state machines [1], [11], [12] or behavior trees [13], [14] as high-level controllers. Similar methods use symbolic reasoning [15]-[17], representing high-level tasks and constraints with symbols. Although these methods offer explainability, their pre-defined nature limits adaptability to real-world variability, leading to suboptimal performance. Our proposed framework addresses this by learning the high-level planner and optimizing lowlevel controller parameters for better generalization and robustness. Recently, learning approaches have emerged to overcome these limitations. Imitation learning (IL) is a key candidate for sequential planning, enabling robots to learn demonstrated behavior sequences. Behavior cloning, a wellestablished IL method, has robots replicate demonstration sequences [18]-[20], but this limits generalizability. Advanced IL methods aim to generalize learned sequences [21]-[23], yet they still struggle to adapt to new environments. Our framework adapts action sequences to the environment state, addressing this limitation and mitigating suboptimal performance from human error in demonstration data. Hierarchical Reinforcement Learning (HRL) has gained attention for long-horizon planning. State-of-the-art approaches like MAPLE [24], RAPS [25], and STAP [26] train hierarchical policies to choose and execute primitives from a behavior library. Despite handling complex tasks and improving sample efficiency, these methods rely on static controllers, which hinder performance in contact tasks and pose risks in real-world settings. Our method builds on these concepts, optimizing stiffness to maximize compliance without compromising task success.

Variable Stiffness Control: Existing methods for adapting the stiffness of an impedance controller typically use taskspecific impedance profiles. Common approaches include learning from demonstration methods, such as Dynamic Motion Primitives [27]-[29] or Gaussian Mixture Models [9], [30]. Alternatively, some methods schedule variable stiffness gains for different task phases [31]-[33]. Despite their ease of application, these methods struggle to generalize stiffness profiles across tasks and depend on expert demonstrators. RL has emerged as a promising method for learning stiffness profiles. Some methods bootstrap the RL policy with initial stiffness demonstrations [34]-[36] to accelerate learning, which are then optimized for specific tasks. However, the reliance on expert demonstrators remains an issue. Other RL approaches focus on designing an appropriate action space in which an agent samples impedance parameters as actions to adapt controller behavior. For adaptive stiffness applications, an impedance action space allows the agent to learn stiffness and damping parameters in joint space [37] and end-effector space [8]. Similar approaches use residual reinforcement learning, where a policy outputs actions to support an existing controller [1], [38], [39]. However, these methods fail in long-horizon tasks due to their limited ability to capture sequential dependencies.

Contributions: The main contributions of this work are i) an impedance primitive augmented HRL framework for sequential contact tasks, ii) a novel behavior affordance that concurrently optimizes for position and compliance; (iii) an adaptive controller for dynamic stiffness modifications for optimal execution in varying environments.

## III. PROBLEM STATEMENT

The long-horizon robotics manipulation task can be formulated within the framework of HRL combined with Parameterized Action Markov Decision Processes (PAMDPs) [40]. Let S represent the state space, and π H : S → A H be the high-level policy that selects high-level actions a H ∈ A H , which define primitives. For each high-level action a H , let π param : S ×L→ Θ be the corresponding low-level policy that selects parameterized actions ( p, θ ) . The overall policy π ( s ) = π π H ( s ) param ( s ) determines the hierarchical decisionmaking process. The environment dynamics are captured by the transition function P ( s ′ | s, ¯ a ) and reward function R ( s, ¯ a ) , where ¯ a = ( p, θ ) . The objective is to find the hierarchical policy π that maximizes the expected cumulative reward J ( π ) = E [ ∑ ∞ t =0 γ t r t ] , optimizing both the highlevel task decomposition and the execution of parameterized actions for efficient manipulation.

## IV. PRELIMINARIES: MAPLE [24]

MAPLE [24] is a state-of-the-art HRL framework that frames the sequential decision making problem as a PAMDP. It uses a two-level policy structure: a high-level task policy π H selects a behavior primitive from a library L = { p 1 , p 2 , ..., p n } , while a low-level parameter policy π aH L predicts the parameters θ for the chosen primitive. Each primitive executes a closed-loop control sequence, minimizing the error between the current state s and the target state θ . The primitives and their parameters are documented in Table I.

To enhance exploration, MAPLE incorporates position affordances, which are rewards that encourage interactions near task-relevant objects. This position affordance is modeled as

<!-- formula-not-decoded -->

where K represents the set of object keypoints and θ is the chosen parameters for a primitive. Accordingly, the affordance reward increases as it approaches objects in the environment.

TABLE I

## DESCRIPTION OF PRIMITIVES AND THEIR PARAMETERS

| Primitive   | Description                                                            | Parameters                   |
|-------------|------------------------------------------------------------------------|------------------------------|
| Reach       | Moves the end-effector to a target location                            | ( x, y, z )                  |
| Grasp       | Moves end-effector to grasp location then activates gripper            | ( x,y,z,ψ )                  |
| Push        | Moves end-effector to a target location, then applies a displacement δ | ( x, y, z, δ x , δ y , δ z ) |
| Atomic      | Apply atomic action                                                    | ( δ x , δ y , δ z )          |
| Gripper     | Open/Close binary gripper                                              | g                            |

MAPLE's structured approach facilitates learning of parameterized skills but lacks explicit mechanisms for impedance control, which is critical for contact-rich tasks. Our proposed IMP-HRL extends MAPLE by integrating impedance primitives and adaptive stiffness control, enabling more robust interaction with the environment.

## V. IMP-HRL

We propose Impedance Primitive-augmented HRL (IMPHRL) for robust sequential contact tasks. We introduce two components into the MAPLE framework that allow us to to achieve variable impedance control for sequential contact tasks.

## A. Impedance Primitive

To accommodate contact-rich environments, the target states need to be extended from exclusively position-based parameters as in MAPLE to also include variable impedance parameters. We propose augmenting HRL with the primitive parameter action space containing the position and impedance parameters [8]. It allows the agent to control the impedance parameters by sampling them as actions. This augmentation extends the parameter space, θ , to now contain ( K x , K y , K z ) for variable stiffness/impedance control along different coordinate axes and K ψ for handling orientation or angular variations (shown in Figure 1). The damping term D in the impedance parameters are selected based on critical damping of system's closed loop response to reduce the number of learnable parameters.

A limitation of this primitive representation arises from the sequential nature of decision-making: once the policy triggers a behavior primitive, it is required to wait for the primitive to complete its execution before modifying the stiffness value again. On the other hand, using an action space with dynamically adapting stiffness parameters introduces a learning challenge. Therefore, the stiffness parameters predicted by the parameter policy will act as an initial stiffness prediction which will be further adjusted using an adaptive stiffness controller.

Affordance Coupling - Combining Position and Stiffness Affordances: In the context of tasks that can benefit from stiffness control, position-based affordances (1) are insufficient since they focus exclusively on spatial information.

To address this limitation, we propose an additional stiffness affordances to maximize compliance whenever possible. In turn, this translates to a reduction in interaction forces between the robot and the environment, which improves the overall safety of the system. Accordingly, stiffness is only increased when it is necessary to meet task requirements. This stiffness affordance is modeled as

This coupling model improves exploration efficiency and encourages the agent to select low-stiffness parameters during the early stages of training. Furthermore, this method eliminates the necessity for careful reward weight tuning that is typically required when directly penalizing high stiffness values. Such tuning would otherwise need to be conducted for each new environment, potentially having a detrimental effect on learning performance [41]. Note that the atomic and gripper release always have an affordance of 1 due to their general utility.

## B. Adaptive Controller

After the policy selects a primitive and its parameters, the behavior is executed through a closed loop control scheme. Using the stiffness parameters outputted by the parameter policy as an initial estimate of the required stiffness to complete a given stage of the task, this stiffness is adapted in real-time using an adaptive stiffness controller. Figure 4 shows the adaptive impedance controller integrated within the low level parametrized policy.

The Adaptive Controller used in this mimics human muscle stiffness during motion execution [42] by adapting the stiffness in accordance with the output of

<!-- formula-not-decoded -->

where ϵ ( t ) is the closed loop feedback error and E is the energy consumed by the robot joints, while β and γ scale these values to influence the stiffness behavior. As for the corresponding damping matrix, it satisfies a critical damping condition such that D ( t ) = 2 √ K ( t ) , which is re-calculated every time the stiffness value is updated. It is important to note that interpolation is used to generate intermediate points along the trajectory toward a target state, which prevents drastic changes in stiffness.

Fig. 2. Heatmap visualization of affordance coupling

<!-- image -->

In practice, the controller stiffness is initialized using the stiffness output of the low-level RL policy. Then, it calculates the stiffness at the next step by using β to scale the increase in stiffness proportional to the feedback error e ( s -θ ) . Simultaneously, it reduces stiffness by scaling current energy consumption E with γ . This process yields a net increase or decrease in the controller's stiffness. Figure 3 demonstrates an example in which a robot performs an elliptical wiping motion.

Since primitives are simple linear movements, the values of β and γ can be obtained by performing kinesthetic demonstrations of the primitives, extracting their stiffness profiles, and minimizing the MSE between the demonstrations and the controller output. This yields β and γ parameters that closely resemble human stiffness behavior. Alternatively, the controller parameters can be determined by simply tuning them until controller performance is satisfactory.

## VI. EXPERIMENTAL RESULTS

In the experiments, we investigated the framework's learning efficiency, analyzed its stiffness and force behavior, highlighted patterns in primitive selection, and evaluated its performance in a real-world setting. This section is divided into experimental setup, evaluation in simulation and real robot, and comparative analysis with respect to state-of-theart method on sequential task execution.

## A. Experimental Setup

We evaluated our framework in four contact-rich environments: Lift, Door, Wipe, and Cleanup. These interactions include basic object manipulation in the Lift environment, continuous contact in the Door and Wipe environments, and a mix of contact and manipulation interactions in the Cleanup environment. The robot utilized for these experiments was a Franka Emika Panda in the Robosuite simulator [43] (see Figure 5) and real-world (see Figure 6). We additionally apply domain randomization by randomly varying table friction, table height, object positions, and initial end-effector position. Lastly, all the reported results were averaged across 5 random seeds.

Fig. 3. Example of adaptive stiffness when wiping.

<!-- image -->

Fig. 4. Adaptive impedance controller integrated within the low-level parametrized policy.

<!-- image -->

## B. Comparative Analysis - Simulation

We compare our proposed framework with the MAPLE baseline. The chosen evaluation metrics are Learning Performance, Maximum Interaction Force, Compositionality, and Success Rate.

Evaluation Metrics: In Learning Performance , we examine learning convergence time to assess learning efficiency of the proposed framework. In Maximum Interaction Force , we evaluate our framework's ability to adapt stiffness across diferent contexts and its effect on the applied forces. In Compositionality , we quantify recurring patterns in primitive selection using a compositionality metric [24]. Lastly, in Success Rate , we analyze the framework's ability to consistently achieve the desired task objectives across the different environments.

Evaluation Results - Learning Performance: We analyzed convergence times by referring to the learning curves in Figure 7. Given that our approach and MAPLE use different affordances, then direct comparisons with MAPLE may not be appropriate since the reward functions are different. However, we can still assess convergence times, defined here as the time taken to learn a near-optimal policy for a given task.

In the Door environment, both our approach and MAPLE

Fig. 5. Simulation Experiments: Lift, Door, Cleanup, Wipe

<!-- image -->

Fig. 6. Real Experiments: Lift, Cleanup, Wipe

<!-- image -->

Fig. 7. Comparison of learning behavior and convergence times for various tasks. The rewards are averaged over 20 episodes then normalized between 0 and 1 (which represents the maximum reward at each timestep).

<!-- image -->

Fig. 8. Variable stiffness behavior demonstrating an emphasis on compliance and stiffness reduction. Each background grid colour represents a different primitive being executed - grasp, reach, push.

<!-- image -->

show approximately equal convergence times. For the Lift and Cleanup tasks, MAPLE converges slightly faster, possibly due to fewer primitive parameters and less exploration constraints from affordance coupling. In the Wipe task, our approach converges much faster, likely due to its ability to leverage variable stiffness, adapting force behavior to task requirements.

Evaluation Results - Maximum Interaction Force: We demonstrate samples of the variable stiffness behavior across the different environments in Figure 8. We also include a graph showing the average applied end-effector forces over a sample of 500 evaluation runs in Figure 9 highlighting our framework's ability to finish the task while exerting less force. These forces were acquired directly from the simulation environment.

In the Lift and Cleanup environments, both of which are tabletop settings, K z is maintained low when interacting near the table, while K x and K y are higher to ensure precise alignment with the objects of interest. In the Door environment, K x is relatively high to provide stability during initial contact, with K y increasing as the door handle is pushed down and all stiffness values decreasing when pulling the door open. In the Wipe environment, K x and K y are low since the primary action involves contact along the z-axis, while K z maintains a higher value to exert enough force for effective wiping without excessive interaction forces.

Fig. 9. Comparison of maximum interaction forces

<!-- image -->

This increased compliance results in lower interaction forces across environments, as shown in Figure 9. Our approach consistently exerts less force, with lower standard deviation, implying less sensitivity to task randomization. The Wipe and Cleanup tasks demonstrate this effect by showing a pronounced decrease in tabletop impact forces. Specifically, these forces are reduced when the robot slides an object along the surface during the Cleanup task and when it wipes away debris during the Wipe task. Note that the average force was only calculated across the successful trials in order to avoid biasing the results, since a robot not performing any actions generates no interaction forces.

Evaluation Results -Compositionality: We quantify recurring patterns of primitive choices for solving a given task using a compositionality metric [24]. A high compositionality score reflects the policy's ability to generate repeatable behavior sequences to complete a given task.

The compositionality was calculated for a sample of 30 successful environment runs, illustrated in Figure 10. The Lift task was excluded as it had the same compositionality score ( f comp = 1 ), with a grasp and reach primitive sequence. In the Door task, we share the same number of primitive executions as MAPLE, but it shows more consistent primitive selection. In the Cleanup task, our approach reduces the number of primitive executions needed, likely due to more robust pushing and precise object approach. In the Wipe environment, our method has more consistent primitive selection than MAPLE, indicating better understanding of task requirements.

Evaluation Results -Success Rate: A comparison of success rates between MAPLE and our method is shown in Table II. Following training, the simulation and real world experiments were run 20 times to obtain the success rates. In the real world experiments, the policy was directly deployed onto the hardware with no fine-tuning to test the sim2real capabilities of the framework. MAPLE was not tested in real-world experiments due to its rigidity and potential operational hazards. Specifically, if the target state was defined at a location on or below the table surface, the robot's motion would lead to unintended force application and potentially cause damage to the environment.

Fig. 10. Compositionality comparison showcasing the learned sequential behavior. The rows correspond to primtive sequences generated by 5 sample environment runs.

<!-- image -->

Our approach achieves comparable success rates in Lift and Door tasks, while also improving the safety of the system due to its higher degree of compliance. In the Cleanup task, IMP-HRL achieves a slightly lower success rate than MAPLE. Given that our method prioritizes compliance, this highlights a tradeoff between safety and success in tasks requiring precise sequential object manipulation.

As for the Wipe task, our approach achieves double MAPLE's success rate. This significant improvement is attributed to our method's stiffness control capacity, as compared to MAPLE's use of position control. Position control works for Lift and Door but fails in wiping due to end-effector rigidity, leading to a misapplication of force or loss of contact. In contrast, our method ensures consistent surface contact and preventing excessive or insufficient force application.

## VII. CONCLUSIONS AND LIMITATIONS

This paper presents a hierarchical reinforcement learning framework aimed at enabling adaptive stiffness control in sequential contact tasks. It utilizes a pre-defined library of behavior primitives and equips them with variable stiffness capabilities. This was done by incorporating an expanded action space to allow the agent to modify its stiffness and an adaptive controller for dynamic stiffness modifications during primitive execution. During training, we introduce affordance coupling to combine position and stiffness affordances, which promotes efficient exploration while incentivizing compliance. The framework showcases notable results in learning efficiency, variable stiffness control, compositionality in primitive selection, and success rates when compared to MAPLE, a state-of-the-art framework in sequential planning. Furthermore, real-world evaluations validate the proposed approach's sim2real capability.

TABLE II SUCCESS RATES (%) FOR SIMULATION AND REAL WORLD

|                   | Lift   | Door   | Wipe   | Cleanup   |
|-------------------|--------|--------|--------|-----------|
| MAPLE             | 100.0  | 100.0  | 42.0   | 91.0      |
| (Simulation)      | ± 0.0  | ± 0.0  | ± 11.7 | ± 5.8     |
| Ours              | 100.0  | 100.0  | 86.0   | 87.0      |
| (Simulation)      | ± 0.0  | ± 0.0  | ± 6.2  | ± 6.1     |
| Ours (Real World) | 90.0   | -      | 70.0   | 80.0      |

The proposed method faces some limitations. The use of affordance coupling may limit learning efficiency when the task relies on accurate manipulation rather than contact or force interaction. This was evident in the experimental results for the Lift and Cleanup environments in which our method required more epochs to learn accurate manipulation. This can be attributed to the fact that affordance coupling incentivizes compliance, while manipulation tasks typically require some degree of stiffness to align the end-effector with a graspable object accurately. Another limitation lies in the acquisition of the adaptive stiffness controller parameters. Specifically, the controller relies on pre-defined scaling factors ( β and γ ) that need to be set. They are acquired either through kinesthetic demonstrations, which require physical interaction with the robot, or iteratively tuning β and γ to match the desired performance, which can be time-consuming.

## REFERENCES

- [1] A. Ranjbar, N. A. Vien, H. Ziesche, J. Boedecker, and G. Neumann, 'Residual feedback learning for contact-rich manipulation tasks with uncertainty,' in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , 2021.
- [2] Q. Li, M. Meier, R. Haschke, H. Ritter, and B. Bolder, 'Object dexterous manipulation in hand based on finite state machine,' in IEEE International Conference on Mechatronics and Automation , 2012.
- [3] S. Nguyen, O. Oguz, V. Hartmann, and M. Toussaint, 'Self-supervised learning of scene-graph representations for robotic sequential manipulation planning,' in Conference on Robot Learning , 2021.
- [4] Z. Zhao, Z. Zhou, M. Park, and Y. Zhao, 'Sydebo: Symbolicdecision-embedded bilevel optimization for long-horizon manipulation in dynamic environments,' IEEE Access , vol. 9, pp. 128 817-128 826, 2021.
- [5] M. M. Botvinick, 'Hierarchical reinforcement learning and decision making,' Current Opinion in Neurobiology , vol. 22, no. 6, pp. 956962, 2012.
- [6] D. W. Franklin, G. Liaw, T. E. Milner, R. Osu, E. Burdet, and M. Kawato, 'Endpoint stiffness of the arm is directionally tuned to instability in the environment,' Journal of Neuroscience , vol. 27, no. 29, pp. 7705-7716, 2007.
- [7] L. Johannsmeier, M. Gerchow, and S. Haddadin, 'A framework for robot manipulation: Skill formalism, meta learning and adaptive control,' in IEEE International Conference on Robotics and Automation (ICRA) , 2019.
- [8] R. Mart´ ın-Mart´ ın, M. A. Lee, R. Gardner, S. Savarese, J. Bohg, and A. Garg, 'Variable impedance control in end-effector space: An action space for reinforcement learning in contact-rich tasks,' in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , 2019.
- [9] F. J. Abu-Dakka, L. Rozo, and D. G. Caldwell, 'Force-based variable impedance learning for robotic manipulation,' Robotics and Autonomous Systems , vol. 109, pp. 156-167, 2018.
- [10] S. Dou, J. Xiao, W. Zhao, H. Yuan, and H. Liu, 'A robot skill learning framework based on compliant movement primitives,' Journal of Intelligent &amp; Robotic Systems , vol. 104, no. 3, p. 53, 2022.
- [11] I.-A. Gal, A.-C. Ciocˆ ırlan, and M. M˘ arg˘ aritescu, 'State machine-based hybrid position/force control architecture for a waste management mobile robot with 5DOF manipulator,' Applied Sciences , vol. 11, no. 9, p. 4222, 2021.
- [12] Y. Onishi and M. Sampei, 'Priority-based state machine synthesis that relaxes behavior design of multi-arm manipulators in dynamic environments,' Advanced Robotics , vol. 37, no. 5, pp. 395-405, 2023.
- [13] K. French, S. Wu, T. Pan, Z. Zhou, and O. C. Jenkins, 'Learning behavior trees from demonstration,' in International Conference on Robotics and Automation (ICRA) , 2019.
- [14] F. Rovida, B. Grossmann, and V. Kr¨ uger, 'Extended behavior trees for quick definition of flexible robotic tasks,' in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , 2017.
- [15] S. Cheng and D. Xu, 'LEAGUE: Guided skill learning and abstraction for long-horizon manipulation,' IEEE Robotics and Automation Letters , vol. 8, no. 10, pp. 6451-6458, 2023.
- [16] C. Agia, T. Migimatsu, J. Wu, and J. Bohg, 'STAP: Sequencing taskagnostic policies,' in IEEE International Conference on Robotics and Automation (ICRA) , 2023.
- [17] B. Wu, S. Nair, L. Fei-Fei, and C. Finn, 'Example-driven model-based reinforcement learning for solving long-horizon visuomotor tasks,' in Conference on Robot Learning , 2022.
- [18] Y. Liu, D. Romeres, D. K. Jha, and D. Nikovski, 'Understanding multimodal perception using behavioral cloning for peg-in-a-hole insertion tasks,' arXiv preprint arXiv:2007.11646 , 2020.
- [19] T. Zhang, Z. McCarthy, O. Jow, D. Lee, X. Chen, K. Goldberg, and P. Abbeel, 'Deep imitation learning for complex manipulation tasks from virtual reality teleoperation,' in IEEE International Conference on Robotics and Automation (ICRA) , 2018.
- [20] B. Wu, F. Xu, Z. He, A. Gupta, and P. K. Allen, 'SQUIRL: Robust and efficient learning from video demonstration of long-horizon robotic manipulation tasks,' in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , 2020.
- [21] J. Liang, B. Wen, K. Bekris, and A. Boularias, 'Learning sensorimotor primitives of sequential manipulation tasks from visual demonstrations,' in International Conference on Robotics and Automation (ICRA) , 2022.
- [22] A. Mandlekar, D. Xu, R. Mart´ ın-Mart´ ın, S. Savarese, and L. Fei-Fei, 'GTI: Learning to generalize across long-horizon tasks from human demonstrations,' in Robotics: Science and Systems , 2020.
- [23] D.-A. Huang, S. Nair, D. Xu, Y. Zhu, A. Garg, L. Fei-Fei, S. Savarese, and J. C. Niebles, 'Neural task graphs: Generalizing to unseen tasks from a single video demonstration,' in IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2019.
- [24] S. Nasiriany, H. Liu, and Y. Zhu, 'Augmenting reinforcement learning with behavior primitives for diverse manipulation tasks,' in International Conference on Robotics and Automation (ICRA) , 2022.
- [25] M. Dalal, D. Pathak, and R. R. Salakhutdinov, 'Accelerating robotic reinforcement learning via parameterized action primitives,' Advances in Neural Information Processing Systems , vol. 34, pp. 21 847-21 859, 2021.
- [26] C. Agia, T. Migimatsu, J. Wu, and J. Bohg, 'Stap: Sequencing taskagnostic policies,' in IEEE International Conference on Robotics and Automation (ICRA) , 2023.
- [27] Y. Zhou, M. Do, and T. Asfour, 'Learning and force adaptation for interactive actions,' in IEEE-RAS Iinternational Conference on Humanoid Robots (HUMANOIDS) , 2016.
- [28] B. Nemec, F. J. Abu-Dakka, B. Ridge, A. Ude, J. A. Jørgensen, T. R. Savarimuthu, J. Jouffroy, H. G. Petersen, and N. Kr¨ uger, 'Transfer of assembly operations to new workpiece poses by adaptation to the desired force profile,' in IEEE International Conference on Advanced Robotics (ICAR) , 2013.
- [29] P. Pastor, H. Hoffmann, T. Asfour, and S. Schaal, 'Learning and
30. generalization of motor skills by learning from demonstration,' in IEEE International Conference on Robotics and Automation , 2009.
- [30] T. Cederborg, M. Li, A. Baranes, and P.-Y. Oudeyer, 'Incremental local online Gaussian mixture regression for imitation learning of multiple tasks,' in IEEE/RSJ International Conference on Intelligent Robots and Systems , 2010.
- [31] Y. Li, G. Ganesh, N. Jarrass´ e, S. Haddadin, A. Albu-Schaeffer, and E. Burdet, 'Force, impedance, and trajectory learning for contact tooling and haptic identification,' IEEE Transactions on Robotics , vol. 34, no. 5, pp. 1170-1182, 2018.
- [32] D. Mitrovic, S. Klanke, and S. Vijayakumar, 'Learning impedance control of antagonistic systems based on stochastic optimization principles,' The International Journal of Robotics Research , vol. 30, no. 5, pp. 556-573, 2011.
- [33] E. A. R¨ uckert, G. Neumann, M. Toussaint, and W. Maass, 'Learned graphical models for probabilistic planning provide a new class of movement primitives,' Frontiers in computational neuroscience , vol. 6, p. 97, 2013.
- [34] E. Theodorou, J. Buchli, and S. Schaal, 'A generalized path integral control approach to reinforcement learning,' The Journal of Machine Learning Research , vol. 11, pp. 3137-3181, 2010.
- [35] J. Rey, K. Kronander, F. Farshidian, J. Buchli, and A. Billard, 'Learning motions from demonstrations and rewards with time-invariant dynamical systems based policies,' Autonomous Robots , vol. 42, pp. 45-64, 2018.
- [36] M. Kim, S. Niekum, and A. D. Deshpande, 'Scape: Learning stiffness control from augmented position control experiences,' in Conference on Robot Learning , 2022.
- [37] M. Bogdanovic, M. Khadiv, and L. Righetti, 'Learning variable impedance control for contact sensitive tasks,' IEEE Robotics and Automation Letters , vol. 5, no. 4, pp. 6129-6136, 2020.
- [38] C. C. Beltran-Hernandez, D. Petit, I. G. Ramirez-Alpizar, T. Nishi, S. Kikuchi, T. Matsubara, and K. Harada, 'Learning force control for contact-rich manipulation tasks with rigid position-controlled robots,' IEEE Robotics and Automation Letters , vol. 5, no. 4, pp. 5709-5716, 2020.
- [39] P. Kulkarni, J. Kober, R. Babuˇ ska, and C. Della Santina, 'Learning assembly tasks in a few minutes by combining impedance control and residual recurrent reinforcement learning,' Advanced Intelligent Systems , vol. 4, no. 1, p. 2100095, 2022.
- [40] W. Masson, P. Ranchod, and G. Konidaris, 'Reinforcement learning with parameterized actions,' in AAAI Conference on Artificial Intelligence , 2016.
- [41] A. Faust, A. Francis, and D. Mehta, 'Evolving rewards to automate reinforcement learning,' arXiv preprint arXiv:1905.07628 , 2019.
- [42] M. Ulmer, E. Aljalbout, S. Schwarz, and S. Haddadin, 'Learning robotic manipulation skills using an adaptive force-impedance action space,' arXiv preprint arXiv:2110.09904 , 2021.
- [43] Y. Zhu, J. Wong, A. Mandlekar, R. Mart´ ın-Mart´ ın, A. Joshi, S. Nasiriany, and Y. Zhu, 'robosuite: A modular simulation framework and benchmark for robot learning,' arXiv preprint arXiv:2009.12293 , 2020.
- [44] J. Tremblay, T. To, B. Sundaralingam, Y. Xiang, D. Fox, and S. Birchfield, 'Deep object pose estimation for semantic robotic grasping of household objects,' arXiv preprint arXiv:1809.10790 , 2018.
- [45] G. Franzese, A. M´ esz´ aros, L. Peternel, and J. Kober, 'Ilosa: Interactive learning of stiffness and attractors,' in 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , 2021.
- [46] B. Kim, J. Park, S. Park, and S. Kang, 'Impedance learning for robotic contact tasks using natural actor-critic algorithm,' IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics) , vol. 40, no. 2, pp. 433-443, 2009.

## APPENDIX I

## ABLATION STUDIES

We conduct ablation studies to measure the impact of the added components on the performance of our system. Specifically, we trained a model on the Wipe environment due to the extensive contact nature of the task. Accordingly, we investigate 3 cases, each of which omits components of the proposed framework. The results are visualized in Figure 11 and are representative of the overall performance across all environments.

- Case 1: Extended action space with Adaptive Controller
- Case 2: Extended action space with Affordance Coupling
- Case 3: Extended action space

Fig. 11. Comparison of convergence time and maximum interaction forces across 3 ablation cases

<!-- image -->

Evaluation Results - Convergence Time. The convergence time results in Figure 11 clearly reflect that the extension of the action space with stiffness parameters is the greatest contributor to the accelerated learning. On the other hand, the exclusive use of an adaptive controller (Case 1) or affordance coupling (Case 2) leads to a notable deterioration in learning performance, as compared to the use of both (Ours).

Evaluation Results -Maximum Interaction Forces. Figure 11 presents the maximum interaction forces achieved through variable stiffness in different frameworks. Our proposed approach consistently minimizes these forces during environmental interactions. This is followed by the framework using only an adaptive controller (Case 1), where stiffness reduction takes place in a relatively narrower range. In turn, this leads to a relatively higher force exertion. As for Case 3, it demonstrates that the exclusive reliance on an extended action space yields the worst performance due to a lack of incentive to reduce stiffness, so the agent opts for high stiffnesses to ensure stain removal.

Looking into Figure 11, we can further infer that the use of affordance coupling (Case 2) leads to lower overall stiffnesses as compared to the adaptive controller (Case 1), demonstrated by the lower interaction forces. This further implies that incorporating the adaptive controller directly with MAPLE will not lead to lower maximum interaction forces, which is typically characterized by a lower overall stiffness.

Evaluation Results - Stiffness Behavior. Upon removing affordance coupling from the framework (Case 1), the agent exhibits a dependence on high stiffness values, which are subsequently reduced using the adaptive controller. As for the case that employs affordance coupling but omits the adaptive controller (Case 2), the agent tends to select relatively low stiffness values, but the profile remains static. Additionally, the absence of corrective behavior leads the agent to attempt corrections during the execution of the next primitive rather than concurrently with the current one. Additionally, removing both affordance coupling and the adaptive controller (Case 3) yields a static stiffness profile, and the agent tends to select high stiffness values due to a lack of incentive for reduction.

## APPENDIX II TRAINING &amp; SIMULATION

## A. Training Setup

The training codebase used is based on RLkit 1 , which in turn is based on rllab 2 . We document all the hyperparameters used in the training procedure in Table III and IV. An important thing to add is that a target entropy is set for the first 200 epochs primarily to promote exploration for both the primitives and the stiffness parameters.

Fig. 12. Simulation Environments: Lift, Door, Wipe, Cleanup

<!-- image -->

With regards to the observations used to train the model, the same observation is shared across all environments except Wipe. In those environments, the observations consist of:

- Cartesian Pose
- Object Poses
- Distance from End-Effector to Object(s)
- Gripper State (either 0 or 1)

As for the Wipe environment, the observation becomes:

[1 https://github.com/rail-berkeley/rlkit](https://github.com/rail-berkeley/rlkit)

[2 https://github.com/rll/rllab](https://github.com/rll/rllab)

- Cartesian Pose
- Percentage Wiped
- Stain Centroid and Radius
- Distance from End-Effector to Centroid

## B. Simulation Setup

Here, we specify the description of each task setup and specify their success conditions:

## Lift:

Description : There is a single cube on a tabletop Success Condition : The cube is lifted above a height threshold (20 cm)

## Door:

Description : There is a hinged door with an L-handle Success Condition : The handle exceeds a certain position (15 cm) and angle (30°)

## Cleanup:

Description : There is a jello box, a spam can, and a wooden box on a tabletop

Success Condition : The jello box is at a threshold distance from the table corner (10 cm) and the spam can is in a wooden box

## Wipe:

Description : There are stains on a tabletop, which are defined by their table coverage percentage (40%) and stain line width (4 cm)

Success Condition : There are no stains on the table

## APPENDIX III REAL WORLD EXPERIMENTS

This section discusses the experimental setup and evaluation of the Real World experiments.

TABLE III

## NETWORK AND OPTIMIZATION PARAMETERS

TABLE IV

| Hyperparameter                   | Value      |
|----------------------------------|------------|
| Network Structure (All Networks) | 512, 512   |
| Q network and policy activation  | ReLU       |
| Q network output activation      | None       |
| Policy network output activation | tanh       |
| Optimizer                        | Adam       |
| Batch Size                       | 1024       |
| Learning rate (all networks)     | 3 × 10 - 5 |
| Target network update rate τ     | 1 × 10 - 3 |

## TRAINING, EXPLORATION, AND REWARD FACTORS

| Hyperparameter                          | Value                          |
|-----------------------------------------|--------------------------------|
| Discount Factor                         | 0.99                           |
| Replay Buffer Size                      | 1 × 10 6                       |
| Reward Scale                            | 5.0                            |
| Affordance Score Scale λ                | 10.0                           |
| Number of Training Steps per Epoch      | 1000                           |
| Number of Exploration Actions per Epoch | 3000                           |
| Horizon Length per Episode              | 150 actions (except wipe, 300) |

Hardware Setup. An Intel RealSense D435i 3 was used to generate an RGB-D stream of the environment. These streams are later used to identify object poses. For the Lift and Cleanup experiments, we placed the camera on a tripod such that it is aligned with the tabletop. As for the Wipe environment, the camera was mounted on the robot endeffector to provide it with an accurate view of the stains.

Software Setup. ROS Noetic was used to interface between the cameras, trained model, and the robot. A RealSense ROS Wrapper 4 was used to extract the RGB-D stream from the camera. In turn, we used Deep Object Pose [44] to estimate the 6D pose of the objects in the environment. Further details regarding observation acquisition are provided in Appendix III.

Robot Control. The impedance controller used was the human-friendly controller 5 [45]. Since our model only outputs stiffness parameters and target positions, we used these parameters as input to the impedance controller. In turn, the controller acts as an interface with the robot to actuate its joints and reach the target position.

[3 https://www.intelrealsense.com/ depth-camera-d435i/](https://www.intelrealsense.com/depth-camera-d435i/)

[4 https://github.com/IntelRealSense/realsense-ros](https://github.com/IntelRealSense/realsense-ros)

[5 https://github.com/franzesegiovanni/franka\_ human\_friendly\_controllers](https://github.com/franzesegiovanni/franka_human_friendly_controllers)

Fig. 13. 6D pose estimation of YCB object set [44]

<!-- image -->

Success Rate. Each experiment was run 20 times with randomized object placements and end-effector starting positions. The model achieved a success rate of 90% on the Lift task, 80% on the Cleanup task, and 70% on the Wipe task.

As mentioned in Appendix II, the model uses object poses as part of the observation. In order to track the 6D pose of the environment objects in the real world, we use Deep Object Pose 6 with the corresponding YCB objects 7 used in simulation. It is important to note that the wiping task naturally does not involve interactions with solid objects, so we used a simple k-means clustering algorithm to identify the wiping stains based on color.

In the real-world experiments, the observations for the trained model were obtained using the Franka ROS Interface and the Intel RealSense D435i camera. More specifically, the process entails the following:

- Cartesian Pose was extracted directly from the Franka ROS Interface. This information includes the position and orientation of the end-effector in the robot's workspace.
- Gripper State was extracted directly from the Franka ROS Interface. The gripper width was used to identify whether it was open or closed.
- Object Pose was estimated using Deep Object Pose with the Intel RealSense D435i camera. Using the RGBD stream, Deep Object Pose analyzes the data and returns 6D object poses at a rate of 15 frames per second.
- Distance from End-Effector to Object was calculated directly given that we have both poses

As for the Wipe environment, there are two unique observation elements. First, K-Means clustering was used on the RGB stream, which was followed by color thresholding. This allows us to separate the black stains from the white background. Accordingly, the observations were acquired using the following methods:

- Percentage Wiped was calculated using the initial stain as a template. By counting the number of black pixels, we can identify how many have been removed, which corresponds to the percentage wiped.
- Stain Centroid and Radius were acquired by pairing the RGB and Depth stream, which allows us to identify the location of the centroid and the radius of the stain.

## APPENDIX IV

ACQUIRING ADAPTIVE CONTROLLER PARAMETERS

The Adaptive Controller adapts the stiffness in accordance with Equation 2, in which β and γ are tunable parameters that influence the stiffness behavior. In this work, we acquire these parameters by collecting a sample of 15 kinesthetic demonstrations in which we equally perform reach, push, and atomic primitives. The gripper primitive was not demonstrated since it does not involve a change in stiffness. As for the grasp primitive, it consists of a reaching and grasping sequence, so the demonstration would be redundant.

[6 https://github.com/NVlabs/Deep\_Object\_Pose](https://github.com/NVlabs/Deep_Object_Pose)

[7 https://ycbbenchmarks.com/object-set/](https://ycbbenchmarks.com/object-set/)

The primitives consist of simple linear motions from the current state to a target state. Accordingly, the kinesthetic demonstrations consisted of moving the end-effector move along a linear path from a random starting position towards a target state(s). We can then extract the stiffness profile used in these motions by referencing a method by Dou et al. [46], in which the impedance of a human arm is modeled as

<!-- formula-not-decoded -->

where F h is the interaction force and K h is the human arm stiffness. This force is then mapped to normalized stiffness using

<!-- formula-not-decoded -->

where K and K represent the upper and lower thresholds of the calculated stiffness while F h and F h represent the upper and lower thresholds of the interaction force.

For each demonstration, we find the values of β and γ by minimizing the Mean Squared Error (MSE) between the demonstration ˙ K ( t ) values and the values predicted by the Equation (2). Once applied across all the demonstrations, we average the acquired γ and β values and use them as parameters of the adaptive stiffness controller. All in all, this of parameter acquisition was performed only once, and allowed the controller to mimic the adaptive behavior presented by the demonstrator across all the evaluated tasks. Additionally, it saved a significant amount of time when compared to manually fine-tuning γ and β to acquire the desired behavior.