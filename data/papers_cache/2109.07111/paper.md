## Elastic Tracker: A Spatio-temporal Trajectory Planner for Flexible Aerial Tracking

Jialin Ji, Neng Pan, Chao Xu, and Fei Gao

Abstract -This paper proposes Elastic Tracker, a flexible trajectory planning framework that can deal with challenging tracking tasks with guaranteed safety and visibility. Firstly, an object detection and intension-free motion prediction method is designed. Then an occlusion-aware path finding method is proposed to provide a proper topology. A smart safe flight corridor generation strategy is designed with the guiding path. An analytical occlusion cost is evaluated. Finally, an effective trajectory optimization approach enables to generate a spatiotemporal optimal trajectory within the resultant flight corridor. Particular formulations are designed to guarantee both safety and visibility, with all the above requirements optimized jointly. The experimental results show that our method works more robustly but with less computation than the existing methods, even in some challenging tracking tasks.

## I. INTRODUCTION

In recent years, with the advancement of vision and navigation technology, flying robots have been widely used in more and more complex missions. Autonomous aerial tracking is a challenging one appied in videography, chasing, cinematographer and surveillance. Generally, there are three main technical challenges of aerial tracking:

- a) Safety: The trajectory of the drone should be collisionfree for both static obstacles and the target.
- b) Visibility: The drone is supposed to keep the target in its limited FOV and avoid the target being occluded by surrounding obstacles.
- c) Smoothness: The trajectory should be smooth to avoid motion blur of the target in view.

Some state-of-the-art aerial tracking planners [1]-[3] addressing the above issues have shown significant robustness and impressive agility. However, these methods are not sufficiently flexible to handle some extreme situations. For instance, when the target moves towards the drone suddenly, changes speed abruptly, and goes around the obstacles frequently, these methods are prone to fail. Summarizing the reasons for the failure, most planners typically design a preplanning procedure like graph-searching to cover safety and visiblity, but carry out some refining work like path smoothing afterwards. Such inconsistency makes the obtained trajectories may not meet all the constraints. Besides, in some cases when the collision avoidance or dynamic feasibility contradicts the requirements of visibility, existing

Corresponding Author: Fei Gao, fgaoaa@zju.edu.cn

This work was supported by the National Natural Science Foundation of China under Grants 62003299.

The State Key Laboratory of Industrial Control Technology, College of Control Science and Engineering, Zhejiang University, Hangzhou 310027, China, and Huzhou Institute, Zhejiang University, Huzhou 313000, China.

Fig. 1. Illustration of the performance of our Elastic Tracker in a real world experiment. The drone is able to keep a proper distance and avoid occlusion while the target moves around the obstacles.

<!-- image -->

methods rarely obtain a feasible solution since the formulations of the constraints are not adaptive enough. To achieve such tasks, an ideal tracking planner shall automatically trade-off the above requirements, which we call elasticity . It is just like there's an invisible spring between the drone and the target, making them neither separated but also stretchable flexibly according to the situation.

In this paper, we propose Elastic Tracker 1 , a flexible tracking framework satisfying the mentioned requisites. To begin with, we inherit the human detection and localization methods of our previous work [4]. Afterwards a lightweight intension-free motion prediction method is designed. Subsequently, an occlusion-aware path finding method is proposed to provide an appropriate topology. With the path's guaidance, a smart safe flight corridor generation strategy is designed, which considers the initial velocity of the drone. An analytical occlusion cost is evaluated passingly, which differs from [1, 3] appraising occlusion with Euclidean Signed Distance Fields (ESDF). Thus the computational burden is greatly reduced. Finally, an effective joint optimization approach enables to generate a spatio-temporal optimal trajectory within the resultant flight corridor. To achieve more elasticity, we design particular formulations for avoiding occlusion and keeping proper viewing distance. The proposed trajectory optimization approach is able to constrain the trajectory at both relative and absolute time while optimizing the time allocation of each piece.

We summarize our contributions as follows:

- 1) An occlusion-aware path searching method and a smart safe flight corridor (SFC) generation strategy.
- 2) An analytical occlusion cost is evaluated without constructing an ESDF using the result of the proposed path searching method.
- 3) Particular formulations are designed for avoiding occlusion and keeping proper viewing distance.
- 4) An effective trajectory optimization approach enables to generate a spatio-temporal optimal trajectory with guaranteed safety and visibility.

## II. RELATED WORK

Some vision-based tracking controllers [5]-[7] take the tracking error defined on image space as the feedback. These reactive methods can achieve real-time performance but are short-sighted to consider safety and occlusion constraints. Many previous research [8]-[10] employ a receding horizon formulation. N ¨ a geli et al. [8] propose a real-time receding horizon planner that optimizes both robot trajectories and gimbal controls for visibility under occlusion. Similarly, Penin et al. [9] design a non-linear model predictive control (NMPC) problem and solve it by sequential quadratic programming. However, the assumption [8, 9] that obstacles are all regarded as ellipsoids limits the application scenarios to artificial or structural environments.

Bonatti et al. [10] trade-off shot smoothness, occlusion, and cinematography guidelines in a principled manner, even under noisy actor predictions. Nevertheless, they rely on numerical optimization of the entire objectives involving complex terms such as integration of signed distance field over a manifold, which cannot guarantee satisfactory optimality. Jeon et al. [1, 11, 12] propose a graph-searchbased path planner along with a corridor-based smooth planner. The former generates a series of viewpoints, and the subsequent smooth planner follows the viewpoints. However, the preplanning procedure involves highly time-consuming graph construction. Additionally, this method assumes that the global map of the environment and the target's moving intent are both known. Therefore, it cannot be used with general unknown environments and targets. Han et al. [2] propose a safe tracking trajectory planner consisting of a target informed kinodynamic searching front-end and a spatialtemporal optimal trajectory planning back-end. However, the optimization formulation of the latter trades off minimizing energy and time, which is entirely inconsistent with the original tracking problem.

Some perception-aware planners are applied to the scenarios resembling aerial tracking. Chen et al. [13] utilize a stereo camera with an independent rotational DOF to sense the obstacles actively. In particular, the sensing direction is planned heuristically by multiple objectives, including tracking dynamic obstacles, observing the heading direction, and exploring the previously unseen area. Watterson et al. [14] plan pose and view direction of the camera by optimizing on manifold R 3 × S 2 . Furthermore, there are also some researches focusing on the planning of UAVs with a limited FOV. Spasojevic et al. [15] propose an optimal path reparameterization maintaining a given set of landmarks within FOV. PANTHER [16] plans trajectories that avoid dynamic obstacles while also keeping them in FOV. However, all the methods above cannot handle the occlusion of static obstacles, which is rather critical for the perception of the target.

Wang et al. [3] design an analytical visibility metric considering both distance keeping and occlusion. However, this metric relies on constructing an ESDF frequently, which is time-consuming. Besides, these methods construct such hard visibility constraints that they are prone to fail in cluttered environments. Notably, they plan both the position and orientation of the drone simultaneously to maximize the object detection, but ignore the observation of other obstacles, which causes unsafety.

## III. FRONT-END PROCESSING

## A. Hierarchical Multi-goal Path Finding

Given a series of target future positions denoted by

<!-- formula-not-decoded -->

Taking into account both distance and occlusion, we define an occlusion-free region Φ k for each predicted position of the target z k , shown in Fig. 2. In order to obtain a proper topology for subsequent actions, we aim to find a path passing through Φ 1 , Φ 2 , ... . Considering efficiency, we use the greedy method, decoupling it into smaller multi-goal path searching problems. Φ k is set to the k -th A glyph[star] destination region, according to which both cost function f k ( n ) and heuristic function h k ( n ) are designed as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where d k xy and d k z are the horizontal and vertical components of the distance between node n and z k , d d is the desired distance between the drone and the target. Except for the colliding ones, the nodes within proper distance but occluded are set invalid. The stop condition is set as n ∈ Φ k . Each searching destination point s k is set to the next starting point.

## B. Safe Flight Corridor Generation

Liu et al. [17] propose an efficient method for generating a safe flight corridor with a polygonal path. However, this method cannot handle infeasible initial states, and the generated polytopes would overlap too much with each other if the segments are too short. We adopt this module and design an intelligent strategy to address the limitations.

Fig. 2. Given a series of future waypoints z k of the target, a path passing through the occlusion-free area Φ k of each waypoint is found.

<!-- image -->

The first polytope is generated with a small segment from the initial position and along the direction of the initial speed. From the end of the first line segment, we search a guiding path as described in III-A. The next polytope is generated with the segment from the last end to the intersection of the last polytope and the guiding path. Finally, we can get a series of polytopes representing the safe region

<!-- formula-not-decoded -->

## C. Visible Region Generation

For each target future position z k and a visible point s k as seed, we generate a sector-shaped visible region

<!-- formula-not-decoded -->

where ξ k denotes the angle bisector, shown in Fig. 3.

Fig. 3. Visible region is represented as a sector-shaped area.

<!-- image -->

## IV. TRAJECTORY OPTIMIZATION

## A. MINCO Trajectory Class

In this paper, we adopt T MINCO [18], a minimum control effort polynomial trajectory class defined as

<!-- formula-not-decoded -->

where an m -dimensional trajectory p ( t ) is represented by a piece-wise polynomial of M pieces and N = 2 s -1 degree. The i -th piece is denoted by

<!-- formula-not-decoded -->

where c = ( c T 1 , . . . , c T M ) T ∈ R 2 Ms × m , c i ∈ R 2 s × m is the coefficient matrix of the piece and β ( t ) = (1 , t, . . . , t N ) T is the natural basis. Time vector T = ( T 1 , . . . , T M ) T , T i is the duration for the i -th piece.

All trajectories in T MINCO have compact parameterization by only q and T , where q = ( q 1 , . . . , q M -1 ) , q i is the intermediate waypoint. Evaluating an entire trajectory from q and T can be done via such a linear-complexity formulation

<!-- formula-not-decoded -->

which allows any second-order continuous cost function F ( c , T ) with available gradient applicable to MINCO trajectories represented by q &amp; T . More specifically, the corresponding cost function for T MINCO is computed as

<!-- formula-not-decoded -->

Then the mapping Equ. 6 gives a linear-complexity way to compute ∂ J /∂ q and ∂ J /∂ T from corresponding ∂ F /∂ c and ∂ F /∂ c . After that, a high-level optimizer is able to optimize the objective efficiently.

## B. Problem Formulation

We expect T , the total duration of the trajectory to be equal to T p , the prediction duration of target. However, enforcing the drone to reach the final state in a fixed duration may cause dynamic infeasible in some cases, for instance, when the target moves faster than the chaser. Therefore, we make a time slack T ≥ T p and set the objective as a tradeoff of minimum jerk and minimum time. The general tracking problem then is formulated as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where ρ is tunable parameter, v m and a m are velocity and acceleration bounds. Prediction timestamps T , safe area P and visible region V k are denoted in Equ. 1, 3 and 4.

We use T MINCO of s = 3 for minimum jerk and M = 2 M P (2 pieces in each polytope) for enough freedom. Then the gradients ∂ J o /∂ c and ∂ J o /∂ T can be evaluated as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Constraints Equ. 8c for dynamic feasbility and Equ. 8d for collision-free are handled with relative time integral penalty method, which is introduced in Section IV-C. Constraints Equ. 8e for occlusion-free and Equ. 8f for distance keeping are handled with absolute time penalty method, which is introduced in Section IV-D. Constraint Equ. 8g is eliminated with a transformation, introduced in Section IV-E.

## C. Relative Time Integral Penalty Method

Dynamic feasibility Equ. 8c and collision avoidance Equ. 8d constraints can be formulated with penalty function as follows

<!-- formula-not-decoded -->

Inspired by the constraint transcription [19] method, G glyph[star] can be transformed into finite constraints via integral of constraint violation, which is furthered transformed into the penalized sampled function J glyph[star] I .

The constraints in Equ. 10 are either linear or quadratic constraints for a specific t and i , thus the time integral penalty with gradient can be easily derived then applied:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where (¯ ω 0 , ¯ ω 1 , . . . , ¯ ω κ i -1 , ¯ ω κ i ) = (1 / 2 , 1 , · · · , 1 , 1 / 2) are the quadrature coefficients following the trapezoidal rule [20].

## D. Absolute Time Penalty

Since the motion prediction of target is represented by a series of discrete waypoints and the visible region is also represented by a series of discrete circular sector, both constraints Equ. 8e and Equ. 8f should be applied to the positions at some specified time t k ∈ T . For a position p ( t k ) , assume that t k is on the j -th piece of the trajectory, i.e.

<!-- formula-not-decoded -->

Then the gradients of c and T can be evaluated as

glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Fig. 4. To avoid contradicting the safety constraint, we design such a cost function for horizontal distance between the drone and the target, which rises abruptly when it's too small but increases linearly when it's too large.

<!-- image -->

However, the duration of each piece of the trajectory T i is to be optimized and which piece j that t k belongs to will change as optimization processes. Thus the cost and gradient will be discontinuous, which makes the optimization fail.

Fortunately, although the costs and gradients of c and T will be discontinuous, the costs and gradients of q and T are continuous using Equ. 6. The detailed proof is omitted due to the limited space.

1) Distance-keeping Constraints: Considering both safety and visibility, it's necessary for the drone to keep a proper distance from the target. Since the drone lacks the independent DOF of pitch, we design different costs for vertical ( δ v ) and horizontal ( δ h ) components of the distance. For the former we set a small vertical tolerance such that δ v ≤ δ v d . For the latter, we design a C 2 penalty function H d ( x ) =

<!-- formula-not-decoded -->

shown in Fig. 4, where [ d l , d u ] are range of desired distance and glyph[epsilon1] is a tiny constant.

Thus the cost of this term can be writen as

<!-- formula-not-decoded -->

where δ h k and δ v k are respectively the horizontal and vertical distance between p ( t k ) and z k .

2) Occlusion-free Constraints: Given a tunable angle clearance θ glyph[epsilon1] , visible region constraint Equ. 4 can be writen as

<!-- formula-not-decoded -->

Apply the same penalty function in Equ. 10 to this term

<!-- formula-not-decoded -->

Denote p = ( p ( t 1 ) , ..., p ( t M T )) , then the gradients of both J d a and J v a can be calculated with O ( M T ) complexity by

<!-- formula-not-decoded -->

Fig. 5. Quadrotor platform used in our experiment.

<!-- image -->

## E. Temporal Constraints Elimination

Time slack constraint Equ. 8g can be writen as

<!-- formula-not-decoded -->

Inspired by [18], we denote τ = ( τ 1 , ..., τ M ) ∈ R M as new variables to be optimized and use the transformation

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus the temporal constraints are eliminated using such substitution.

## V. EXPERIMENTS AND BENCHMARKS

## A. Implementation Details

For human detection, we apply open pose [21] and track the target with an EKF under the assumption of the constant velocity. In order to avoid the target being regarded as a static obstacle while mapping, the surrounding point clouds of the target are removed manually. As for the motion prediction of the target, we just generate motion primitives for the target and choose a safe and minimum acceleration one.

Since the yaw is independent, we carry out yaw planning separately. We constrain the angular velocity of yaw and make the drone head in the direction towards the target. Besides, we check whether the trajectory for a period in the future enters the unknown area and make the drone head in the direction towards the unknown area if true.

## B. Real World Experiments

Our quadrotor platform, shown in Fig. 5, is equipped with an Intel RealSense D435 depth camera for self-localization [22] and mapping, a mono camera (FOV = 85 ◦ × 72 ◦ ) for shooting the target, a Jetson Xavier NX4 for running object detection and a Manifold2 for the other computation tasks.

To validate the robustness of our method, we set up experiments in several challenging scenarios. In a long-term tracking experiment, the target moves towards the drone suddenly, shown in Fig. 6 and the drone is able to keep a safe distance from the target flexibly. In a visibility test, the target walks a figure eight around two cabinets, shown in Fig. 7 and the drone is able to keep the target in the FOV.

Fig. 6. While the target moves close or away from the drone, a proper distance is kept, which is just like there's an invisible spring between them.

<!-- image -->

Fig. 7. While the target walks a figure eight around two obstacles, the drone can avoid occlusion and keep the target in the view.

<!-- image -->

## C. Simulation and Benchmark Comparisons

We benchmark our method with Han's [2] and Wang's [3] work in simulation. All the simulation experiments are run on a desktop equipped with an Intel Core i7-6700 CPU. To compare the tracking performance fairly, we set three drones using each method to chase the same target simultaneously. The target moves along such an aggressive path in a cluttered

Fig. 8. (a) History path of the target while moving aggressively in a cluttered environment. (b) The velocity of the projection of the centroid of the target onto the image plane. (c) Failure time of three different cases: out of view, too near (distance less than 1m), occluded by surrounding obstacles.

<!-- image -->

Fig. 9. The distribution of the target positions relative to the tracking quadrotor on x-y plane. The red sector represents the FOV of the drone.

<!-- image -->

TABLE I

## COMPUTATION TIME COMPARISON

| Target   | Method   | Average calculating time (ms)   | Average calculating time (ms)   | Average calculating time (ms)   | Average calculating time (ms)   | Average calculating time (ms)   |
|----------|----------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|
| v max    |          | t path                          | t corridor                      | t ESDF                          | t optimze                       | t total                         |
| 1 m/s    | Han      | 6.67                            | 2.89                            | \                               | 0.38                            | 9.94                            |
| 1 m/s    | Wang     | 0.63                            | \                               | 6.43                            | 5.5                             | 12.56                           |
| 1 m/s    | Ours     | 0.856                           | 0.538                           | \                               | 1.79                            | 3.184                           |
| 2 m/s    | Han      | 10.5                            | 4.62                            | \                               | 0.44                            | 15.56                           |
| 2 m/s    | Wang     | 0.54                            | \                               | 6.43                            | 5.7                             | 12.67                           |
| 2 m/s    | Ours     | 1.31                            | 1.02                            | \                               | 2.87                            | 5.2                             |

## VI. CONCLUSION AND FUTURE WORK

In this paper, we summarize the challenging requirements of elastic tracking and propose a flexible trajectory planning framework for aerial tracking. Extensive simulations and real-world experiments validate the robustness and efficiency of the proposed method. In the future, we will estimate the target's intention and improve the motion prediction. Furthermore, we will generalize our method to more extreme scenarios like tracking escaping target.

environment shown in Fig. 8(a). The max velocity of the target is set as 2 m/s , and its position is broadcasted to each chaser. The max velocity and acceleration of the chasers are set as 3 m/s and 6 m/s 2 . The camera has an image size of 640×480 px 2 , a limited FOV of 80 ◦ × 65 ◦ .

The velocity of the projection of the target onto the image plane is shown in Fig. 8(b), which means our method achieves a much less blurred projection of the target than the other two methods. We count the failure time of each chaser shown in Fig. 8(c). The failures can be categorized under three headings: out of FOV, too near from the drone, occluded from the surrounding obstacles. We can see that the failure rate of our method is much lower than the other two methods. Furthermore, we count the target positions projected to x-y plane in the tracking quadrotor's FOV, shown in Fig. 9. The heat map shows the distribution of the target positions relative to the drone, where our method is able to keep the obstacle inside the FOV limits much better.

We also benchmark the computation time of the three methods in different scenarios. As is shown in Tab. I, the proposed method needs a much lower computation budget.

## REFERENCES

- [1] B. Jeon, Y. Lee, and H. J. Kim, 'Integrated motion planner for realtime aerial videography with a drone in a dense environment,' in 2020 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2020, pp. 1243-1249.
- [2] Z. Han, R. Zhang, N. Pan, C. Xu, and F. Gao, 'Fast-tracker: A robust aerial system for tracking agile target in cluttered environments,' arXiv preprint arXiv:2011.03968 , 2020.
- [3] Q. Wang, Y. Gao, J. Ji, C. Xu, and F. Gao, 'Visibility-aware trajectory optimization with application to aerial tracking,' arXiv preprint arXiv:2103.06742 , 2021.
- [4] N. Pan, R. Zhang, T. Yang, C. Xu, and F. Gao, 'Fast-tracker 2.0: Improving autonomy of aerial tracking with active vision and human location regression,' arXiv preprint arXiv:2103.06522 , 2021.
- [5] J. Kim and D. H. Shim, 'A vision-based target tracking control system of a quadrotor by using a tablet computer,' in 2013 international conference on unmanned aircraft systems (icuas) . IEEE, 2013, pp. 1165-1172.
- [6] A. G. Kendall, N. N. Salvapantula, and K. A. Stol, 'On-board object tracking control of a quadcopter with monocular vision,' in 2014 international conference on unmanned aircraft systems (ICUAS) . IEEE, 2014, pp. 404-411.
- [7] H. Cheng, L. Lin, Z. Zheng, Y. Guan, and Z. Liu, 'An autonomous vision-based target tracking system for rotorcraft unmanned aerial vehicles,' in 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS) . IEEE, 2017, pp. 1732-1738.
- [8] T. N¨ ageli, J. Alonso-Mora, A. Domahidi, D. Rus, and O. Hilliges, 'Real-time motion planning for aerial videography with dynamic obstacle avoidance and viewpoint optimization,' IEEE Robotics and Automation Letters , vol. 2, no. 3, pp. 1696-1703, 2017.
- [9] B. Penin, P. R. Giordano, and F. Chaumette, 'Vision-based reactive planning for aggressive target tracking while avoiding collisions and occlusions,' IEEE Robotics and Automation Letters , vol. 3, no. 4, pp. 3725-3732, 2018.
- [10] R. Bonatti, Y. Zhang, S. Choudhury, W. Wang, and S. Scherer, 'Autonomous drone cinematographer: Using artistic principles to create smooth, safe, occlusion-free trajectories for aerial filming,' in International Symposium on Experimental Robotics . Springer, 2018, pp. 119-129.
- [11] B. F. Jeon and H. J. Kim, 'Online trajectory generation of a mav for chasing a moving target in 3d dense environments,' in 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) . IEEE, 2019, pp. 1115-1121.
- [12] B. F. Jeon, D. Shim, and H. J. Kim, 'Detection-aware trajectory generation for a drone cinematographer,' in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) . IEEE, 2020, pp. 1450-1457.
- [13] G. Chen, W. Dong, X. Sheng, X. Zhu, and H. Ding, 'Bio-inspired obstacle avoidance for flying robots with active sensing,' arXiv eprints , pp. arXiv-2010, 2020.
- [14] M. Watterson, S. Liu, K. Sun, T. Smith, and V. Kumar, 'Trajectory optimization on manifolds with applications to quadrotor systems,' The International Journal of Robotics Research , vol. 39, no. 2-3, pp. 303-320, 2020.
- [15] I. Spasojevic, V. Murali, and S. Karaman, 'Perception-aware time optimal path parameterization for quadrotors,' in 2020 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2020, pp. 3213-3219.
- [16] J. Tordesillas and J. P. How, 'Panther: Perception-aware trajectory planner in dynamic environments,' arXiv preprint arXiv:2103.06372 , 2021.
- [17] S. Liu, M. Watterson, K. Mohta, K. Sun, S. Bhattacharya, C. J. Taylor, and V. Kumar, 'Planning dynamically feasible trajectories for quadrotors using safe flight corridors in 3-d complex environments,' IEEE Robotics and Automation Letters , vol. 2, no. 3, pp. 1688-1695, 2017.
- [18] Z. Wang, X. Zhou, C. Xu, and F. Gao, 'Geometrically constrained trajectory optimization for multicopters,' arXiv preprint arXiv:2103.00190 , 2021.
- [19] L. S. Jennings and K. L. Teo, 'A computational algorithm for functional inequality constrained optimization problems,' Automatica , vol. 26, no. 2, pp. 371-375, 1990.
- [20] W. H. Press, S. A. Teukolsky, W. T. Vetterling, and B. P. Flannery, Numerical Recipes with Source Code CD-ROM 3rd Edition: The Art of Scientific Computing . Cambridge University Press, 2007.
- [21] Z. Cao, G. Hidalgo, T. Simon, S.-E. Wei, and Y. Sheikh, 'Openpose: realtime multi-person 2d pose estimation using part affinity fields,' IEEE transactions on pattern analysis and machine intelligence , vol. 43, no. 1, pp. 172-186, 2019.
- [22] T. Qin, P. Li, and S. Shen, 'Vins-mono: A robust and versatile monocular visual-inertial state estimator,' IEEE Transactions on Robotics , vol. 34, no. 4, pp. 1004-1020, 2018.