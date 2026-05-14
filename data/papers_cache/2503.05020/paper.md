## GRIP : A G eneral R obotic I ncremental P otential Contact Simulation Dataset for Unified Deformable-Rigid Coupled Grasping

Siyu Ma ∗ 1 , Wenxin Du ∗ 1 , Chang Yu ∗ 1 , Ying Jiang ∗ 1 , Zeshun Zong 1 , Tianyi Xie 1 , Yunuo Chen 1 , Yin Yang 3 , Xuchen Han 2 , Chenfanfu Jiang 1

Fig. 1: GRIP is a large-scale and universal grasping dataset that comprises 100K high-quality grasps across diverse scenarios. It includes soft UMI grippers ( top ) and rigid LEAP Hands ( bottom ), interacting with rigid ( left ) and soft ( right ) objects under both unimanual and bimanual settings. Our dataset captures a wide variety of object shapes, sizes, and materials, along with rich object and gripper deformations and stress distribution by an IPC simulator with high-fidelity frictional contact.

<!-- image -->

Abstract -Grasping is fundamental to robotic manipulation, and recent advances in large-scale grasping datasets have provided essential training data and evaluation benchmarks, accelerating the development of learning-based methods for robust object grasping. However, most existing datasets exclude deformable bodies due to the lack of scalable, robust simulation pipelines, limiting the development of generalizable models for compliant grippers and soft manipulands. To address these challenges, we present GRIP, a General Robotic Incremental Potential contact simulation dataset for universal grasping. GRIP leverages an optimized Incremental Potential Contact (IPC)-based simulator for multi-environment data generation, achieving up to 48× speedup while ensuring efficient, intersection- and inversion-free simulations for compliant grippers and deformable objects. Our fully automated pipeline generates and evaluates diverse grasp interactions across 1,200 objects and 100,000 grasp poses, incorporating both soft and rigid grippers. The GRIP dataset enables applications such as neural grasp generation and stress field prediction. We release GRIP to advance research in robotic manipulation, soft-gripper control, and physics-driven simulation at: https://bell0o.github.io/GRIP/.

## I. INTRODUCTION

Grasping is pivotal to robotics, serving as the cornerstone of nearly all manipulation tasks. However, acquiring reliable grasping data for robotics training faces challenges such as object variability, occlusions, and generalization to unseen objects. The emergence of large-scale grasp datasets, such as GraspNet-1Billion [1] and DexGraspNet [2], has significantly advanced research in this area, providing rich data to improve grasp prediction, facilitating applications across industrial automation, household robotics, and healthcare.

* equal contribution.

1 siiyuma@outlook.com, { wenxindu,changyu1,yingjiang, zeshunzong,tianyixie77,yunuoch,cffjiang } @ucla.edu , AIVC Laboratory, UCLA, USA.

2 xuchen.han@tri.global , Toyota Research Institute, USA.

3 yin.yang@utah.edu , University of Utah, USA.

More recently, developments in soft grippers, such as the UMI gripper [3] and the Bubble gripper [4], have further advanced grip stability. Due to their compliance, soft grippers are particularly suitable for grasping fragile objects such as fruits, food, and biological tissues, where preventing damage is crucial.

Despite recent progress in soft grippers, grasp datasets primarily emphasize rigid grippers and manipulands, with none specifically dedicated to soft grippers. The DefGraspSim [5] dataset remains the only notable attempt to include deformable manipulands, but its limited scale (34 objects and 6,800 grasp evaluations) falls short of supporting the development of generalizable grasp prediction models. The scalability of real-world datasets for soft grippers and deformable manipulands is limited by their high degrees of freedom (DoFs). This complexity makes it difficult to track deformations accurately, especially when occlusions occur during grasping. Consequently, simulations are often favored over real-world data.

An ideal physical simulator for soft grippers and deformable objects must satisfy three key criteria. (1) Ro- bustness : It should accurately handle soft-soft, soft-rigid, and rigid-rigid interactions across diverse objects and dynamic conditions without introducing critical artifacts. (2) Efficiency : It should be parallelized to enable large-scale dataset generation. (3) Accuracy : It must provide precise physical dynamics and realistic frictional contact. However, existing simulators struggle with reliable contact resolution in complex geometries, such as strings, cones, or thin surfaces, often producing tunneling artifacts that distort physical interactions and deviate from real-world behavior. Moreover, most FEM-based soft-body simulators are prone to element inversion under large deformations, further compromising simulation accuracy.

The recently proposed Incremental Potential Contact (IPC) method [6] serves as a remedy, effectively addressing large-deformation and frictional contact problems for soft bodies within a variational framework, while guaranteeing intersection- and inversion-free solutions. Recent extensions and developments, such as [7], [8], have integrated IPC and Affine Body Dynamics (ABD) [9] into a unified framework with multibody and articulated dynamics [10]. However, these approaches either lack user-friendly interfaces for robotic applications, or are not optimized for large-scale, multi-environment simulations, resulting in inefficiencies in robotic applications.

To address the aforementioned challenges, we present GRIP : a G eneral R obotic I ncremental P otential contact simulation dataset for universal grasping. Our key contributions include the following:

- 1) A high-performance robust IPC-based simulator optimized for large-scale, multi-environment dataset generation, incorporating the four core features outlined earlier. Our simulator achieves a 48× speedup when running 400 parallel environments compared to single-environment sequential simulations.
- 2) A fully automated pipeline for grasp generation, simulation, and evaluation. The pipeline generates synthesized grasps and simulation results such as gripper and manipuland deformation and stress distribution. It also supports various types of grippers (parallel and dexterous, rigid and soft) and manipulands (rigid and soft) for both uni- and bi-manual grasp settings.
- 3) The GRIP dataset , a large-scale and diverse grasping dataset containing 1200 objects and 100K grasp poses , featuring both soft UMI grippers [3] and rigid LEAP Hand grippers [11].
- 4) Practical applications of the GRIP dataset , showcasing its effectiveness in neural grasp generation and stress field prediction for soft manipulands.

## II. RELATED WORK

## A. Dataset for Hand Grasping

Existing grasp datasets for robotic grippers primarily focus on rigid gripper-rigid object interactions. Many center on parallel grippers [13], [14], object-centric relationships [15], or multimodal sensing [16]. These grippers are typically composed of rigid links connected via joints. For example, [1], [17] provide grasp poses for rigid parallel grippers. Dexterous rigid grippers, which offer greater agility and versatility due to their increased degrees of freedom (DoFs), enable a wider range of human-like grasping motions. [2], [18] synthesize dexterous poses for ShadowHand and Allegro, whereas [19] generates grasp poses for five different dexterous hands. [20] introduced a large-scale synthetic dataset covering 11 common types of rigid grippers. In addition to single grippers, [21] introduces a large-scale synthetic dataset focusing on bimanual grasping with heavy or large objects.

Recently, efforts have also been made to collect grasp poses for soft objects. For instance, [5] offers simulated deformation and stress distribution for 34 deformable manipulands grasped by rigid grippers using the Isaac Gym [22] simulator. Soft grippers, such as the UMI gripper [3] and the bubble gripper [4], can offer more stable grasps due to their superior compliance. However, none of the existing grasp datasets include soft grippers, which limits the progress of their downstream tasks.

## B. Grasp Generation

Most prior works formulate grasp synthesis as an optimization problem. For instance, GraspIt! [23] generates grasp poses using annealing-based search algorithms. However, its simplified optimization objectives and search strategies limit both the efficiency and diversity of generated grasp poses [24], [2]. To address this, [24] introduces a differentiable force-closure term, enabling force-closure-aware, optimization-based grasp synthesis, thus yielding more diverse grasp poses. Building on this approach, DexGraspNet [2] enhances gripper-object interaction by incorporating a robust, differentiable object-to-hand penetration energy term. Additionally, DexGraspNet significantly improves grasp success rates by leveraging the Isaac Gym simulator [22] to validate and filter grasp poses. This method has facilitated the creation of a large-scale dexterous grasp dataset for ShadowHand. Xu et al. [25] proposed a gradient descent-based optimizer with a surface-matching metric. In addition to meshes, it supports point clouds as object geometry representation, enhancing real-world applicability, since point clouds are more accessible in practical scenarios. Nevertheless, their method requires intricate parameter tuning to generate diverse grasp poses.

Recently, data-driven methods have advanced grasp generation through diffusion-based dexterous grasping [26], reinforcement learning [27], and grasp proposal networks for both 6-DOF [28], [29] and dexterous grasping [30]. These approaches predict grasp poses based on object geometry and efficiently generate diverse grasps in parallel, making them well-suited for real-time applications. Moreover, many studies [26], [28], [29], [30] support point cloud representations of grasped objects, further improving their practicality for real-world deployment.

Fig. 2: Dataset Generation Pipeline : Candidate grasps are first synthesized using GPD [12] and DexGraspNet [2]. These candidate poses are then evaluated in a parallelized IPC simulator, where 6-axis gravity is applied to assess grasp stability. The resulting dataset captures contact and friction data, along with detailed deformation and stress information for soft manipulands and grippers. It provides a comprehensive record of contact interactions and material responses across all simulation time steps throughout the entire grasping trajectory.

<!-- image -->

## C. Robot Simulation

Mainstream robotic deformable body simulation algorithms often adopt the Material Point Method (MPM) for tasks such as tactile sensing [31], soft robot control [32], and soft object manipulation [33], or use the Finite Element Method (FEM) [34], [35] with various dimensional topologies [8], [36]. MPM models an object as a set of particles without explicit topology, effective for handling large topology changes. However, it relies on a background Eulerian grid, which introduces numerical diffusion artifacts and incurs high computational costs as grid resolution increases [37], [38]. FEM represents objects as structured meshes, such as triangular elements for surface meshes and tetrahedral elements for volume meshes, and yields great accuracy when objects undergo small deformations. Nevertheless, for most FEM-based methods, large deformations will lead to element inversion, resulting in simulation failures [39], [40]. Further, unlike MPM, FEM requires explicit collision handling, which is commonly formulated as a Linear Complementarity Problem (LCP) [41], [42] or a convex optimization problem [43], [44]. LCP-based methods are hindered by their NP-hardness and ill-conditioning issues, while convex-optimization-based approaches face challenges in extreme cases, such as fast-moving non-convex objects and extremely thin objects.

Recently, Li et al. proposed Incremental Potential Contact (IPC) [6], a FEM-based method that accurately and robustly handles extreme contact scenarios. IPC formulates frictional contact within FEM as an optimization problem, incorporating barrier augmentation, continuous collision detection (CCD), and a Projected Newton optimization scheme with a step-size filter to enforce intersectionand inversion-free constraints [45]. Despite its high accuracy in robotic simulations, including dataset generation [46], tactile modeling [47], and general-purpose simulation [48], the computational inefficiency of IPC limits its applicability to large-scale simulations, hindering its adoption in large data generation and parallel reinforcement learning training.

Our work extends IPC to a large-scale parallel environment simulation, improving its efficiency and scalability through parallel computing. We construct a large-scale grasp pose dataset using optimization-based pose synthesis combined with a multi-environment parallel IPC simulator.

## III. GRIP DATASET GENERATION

We outline the complete methodology for our dataset generation. Our approach utilizes a GPU-parallelized, multienvironment high-performance IPC simulator (Section IIIA) to enable large-scale dataset generation with strict nonpenetration guarantees, providing precise contact and stress information. The overall data generation pipeline is illustrated in Fig. 2. We first convert grippers and objects into simulation-ready assets (Section III-B). Next, we use grasp synthesis algorithms to generate candidate grasp poses (Section III-C), which are evaluated using the IPC simulator. Afterwards, unstable grasps are filtered out (Section III-D).

## A. IPC Simulator with Parallel Environment

A straightforward approach to implementing a fully GPUparallelized IPC simulator for multi-environment simulation is to stack all environments into a single system within a shared scene. This method is inefficient and prone to failures for several reasons: (i) originally independent simulations may interfere, causing unintended collisions; (ii) to ensure non-penetration, the global Newton step size is constrained by the smallest non-colliding step size across all environments, reducing overall efficiency; (iii) varying convergence iterations across different environments lead to wasted computation time for already converged environments; (iv) a failure in one environment causes the entire system to fail.

To address these challenges, we optimize the solver pipeline to support large-scale parallel simulation. First, we introduce environment isolation during collision detection and step size filtering, allowing each environment to evolve independently with its own optimization step size.

Second, we track individual stopping criteria (e.g., relative error in Newton iterations) for each environment. Once an environment meets its stopping criteria, it is frozen and excluded from further computations, freeing up resources for the remaining active environments. Third, we introduce a failure detection stage in the simulation pipeline to identify and exclude failed environments from further computation, isolating them from the overall system to ensure robustness. In addition, we incorporate parallel computing in linear solves to boost simulation efficiency. The performance of our parallel simulator is evaluated in Section IV-A.

## B. Grippers and Objects

a) Gripper Modeling: Our dataset covers both UMI grippers [3] and LEAP Hands [11]. For simulation, we tetrahedralize the soft finger meshes of UMI grippers using TetWild [49], setting a tight envelope tolerance of ϵ e = 5 × 10 -4 to preserve fine surface geometry details. The Young's modulus ( E UMI = 9 . 4 × 10 6 Pa) and dry friction coefficient ( µ UMI = 3 . 5 ) of the UMI gripper fingers are set to match their real-world counterparts [3]. For LEAP Hands, leveraging IPC's robustness for handling complex geometries, we use their high-resolution visual meshes directly as collision meshes for precise contact resolution in simulation. To generate the contact point candidates for DexGraspNet synthesis, we heuristically sample points evenly across the contact regions of the hand's palms and fingers.

b) Object Preparation: We select 800 objects from the DexGraspNet dataset [2] for single-hand grasping via stratified sampling based on label categories and their object counts. The subset matches the original distribution, with at least one object included from underrepresented labels to ensure diversity. For bimanual grasping, we retain 400 large objects from the single-hand set and add additional 400 objects from the PartNet-Mobility dataset [50], totaling 800 large-scale objects. To prepare for the simulation, we preprocess the surface meshes to be watertight and tetrahedralize them for soft body simulations using TetWild. We also apply domain randomization, sampling Young's modulus, friction coefficient, and object scaling to enhance data diversity.

## C. Grasp Pose Synthesis

We present our grasp pose synthesis algorithms for the UMI gripper and the LEAP Hand. The synthesized grasp poses serve as candidate configurations, which are subsequently validated and filtered with our simulator.

For the UMI gripper, we use the GPD method [12], a simple yet effective solution for parallel grippers. Since IPC simulation requires an intersection-free initial configuration, we discard a small fraction of the synthesized grasp poses where the gripper penetrates the object.

For the LEAP Hand, we adopt DexGraspNet [2], replacing the ShadowHand with the LEAP Hand, and incorporate a normal alignment energy term E normal defined as

<!-- formula-not-decoded -->

Fig. 3: Performance Analysis of Our Parallel IPC Simulator : Runtime profiling for unimanual UMI grippers and LEAP Hands under varying numbers of parallel environments. We compare our runtime and speedup with sequential execution, demonstrating the strong scalability.

<!-- image -->

from [25] into the optimization process. This modification encourages alignment between the contact normals of the LEAP Hand and the object, and significantly improves grasp success rates. Here n c is the number of contact points, n h i represents the normal vector of the i -th hand contact point in its associated link's local frame T i , R i denotes the rotation from T i to the world frame, and n o i is the normal vector of the i -th object contact point in the world frame.

Although DexGraspNet efficiently generates grasp poses, minor penetrations between the gripper and object often occur, violating IPC's prerequisite for intersection-free initial configurations. To address this, we apply inverse kinematics to adjust fingertip positions along the object's surface normals, ensuring contact distances exceed IPC's contact threshold ˆ d . For GRIP dataset generation, we set ˆ d = 10 -3 m. This method effectively resolves most initial penetrations, with any remaining penetrated poses discarded.

For bimanual grasp poses, we synthesize n target bimanual grasp candidate poses for each object by composing unimanual-grasp poses. First, we apply k-means clustering to group left and right gripper candidate poses into k = 26 clusters, using the center positions of contact points as grouping keys. These clusters form k 2 potential pairs of left-hand and right-hand pose groups. Next, we select the top r 1 k 2 cluster pairs where the Euclidean distance between the group-averaged contact position centers of the left and right hands is the largest. For GRIP dataset generation, we set r 1 = 0 . 25 . Within each selected cluster pair, we retain all possible left-hand and right-hand pose combinations, denoting the total number of such bimanual grasp poses as nfi ltered. Finally, we compute an approximated force closure metric for each bimanual grasp pose [24], [2] and retain the bottom r 2 = n target nfi ltered fraction within each cluster pair, yielding approximately n target final bimanual grasp candidates.

Fig. 4: Comparison of Real-World and Simulated Grasps : Grasps of four 3D-printed manipulands in the real world ( top ) and in simulations ( bottom ), showing strong consistency between real-world and simulated results, demonstrating the high fidelity of our simulator.

<!-- image -->

## D. Grasp Validation

We use our parallel IPC simulator to filter out failed grasps and record simulation data for successful ones. The time step size is set to ∆ t = 0 . 01 seconds. The contact stiffness is κ = 3 × 10 6 kg · s -2 , the contact distance threshold is ˆ d = 10 -3 m, and the velocity magnitude threshold for dynamicstatic friction transition is ϵ v = 10 -3 m/s, ensuring precise frictional contact handling. Simulations are run with FP64 precision. The Newton optimization terminates when the relative tolerance ϵ r = 10 -3 is reached. If optimization fails to converge within N iters = 100 iterations, the environment is marked as failed.

At the start of each trajectory, gravity is disabled to prevent the object from falling before being grasped. To simulate contact-aware grasping, a finger's motion is halted once its contact force exceeds 50 N. After the system reaches a steady state, gravity is applied in six axis-aligned directions, each with a magnitude of 9.8 m/s 2 for 0.1 seconds. A grasp pose is considered stable if the object remains in contact with the gripper at the final time step of the trajectory and if its center of mass moves less than a threshold proportional to ϵ v ∆ t.

## IV. EXPERIMENTS

## A. Multi-environment IPC Dataset Generation

We first evaluate the performance of our parallel IPC simulator on a single NVIDIA H100 GPU (30 FP64 TFLOPS). Specifically, we run the dataset validation pipeline described in Section III-D for rigid manipulands under varying numbers of parallel environments, ranging from a single environment to 512 environments. The speedup of N parallel environments is measured as the runtime ratio compared to running N environments sequentially.

The results in Fig. 3 demonstrate the strong performance of our approach: unimanual Leap Hand data generation achieves a 16× speedup with 512 environments, while unimanual UMI gripper data generation achieves a 40× speedup with 400 environments, enabling highly efficient large-scale data generation in parallel. To generate the full dataset, we deploy our automatic pipeline across 8 H100 GPUs, requiring approximately 600 GPU hours for all candidate poses, a process that would be prohibitively time-consuming without parallel environment support.

Fig. 5: IPC v.s. Isaac Gym Validation: We filter the candidate poses generated by DexGraspNet [2] using both our IPC simulator and the Isaac Gym [52] simulator. Many poses deemed reasonable by Isaac Gym exhibit artificial penetrations, undermining validity. In contrast, our IPC simulator provides accurate, physically valid grasping behaviors with precise frictional contact modeling, effectively identifying and retaining only high-quality grasps.

<!-- image -->

## B. Real2Sim Validation

We select four objects of varying sizes and shapes from the single-hand manipuland set and fabricate them using a 3D printer with TPE-83A material. These objects are then grasped by a UMI gripper, and the resulting trajectories are recorded with a calibrated RealSense D435 camera. ArUco markers are attached to the manipulands and the grippers for 6D pose detection [51]. The recorded gripper trajectories are subsequently replayed in our simulator, with the initial poses of the manipulands aligned to those in the real-world setup. As shown in Fig. 4, the bending behavior of the UMI gripper fingers and the 6D poses of the manipulands in the simulation closely match the real-world results, demonstrating high physical accuracy and fidelity.

## C. Dataset Quality Evaluation

a) Quantitative Analysis: At the core of our approach is an accurate and robust IPC-based simulator, designed to generate physically realistic grasp poses. To assess its effectiveness, we compare it with Isaac Gym [22], a widely used simulator that is also adopted in DexGraspNet [19]. For a fair comparison, we use the same pipeline outlined in Fig. 2 but replace our IPC simulator with Isaac Gym for grasp pose validation. We then compare the quality of final grasp poses produced by both simulators, ensuring that the gripper securely holds the manipulands.

<!-- image -->

𝒅

𝟐

Fig. 6: Illustration of Grasp Quality Metrics : ( Left ) The gripper and manipuland are not in contact. In this case, D 1 = 0 and D 2 is the unsigned gripper-manipuland distance, d 1 . ( Right ) Penetration occurs. Both D 1 and D 2 equal the penetration distance, d 2 .

TABLE I: Grasp Quality Benchmark : Penetration distance D 1 and absolute distance D 2 are evaluated in centimeters (cm). The success rate measures whether the grasp poses generated by the trained UGG [26] model can securely hold the manipuland within our IPC simulator.

| Simulator   | Bottle Subset ↑   | Bottle Subset ↑   | Bottle Subset ↑   | Full Dataset   | Full Dataset   | Full Dataset   |
|-------------|-------------------|-------------------|-------------------|----------------|----------------|----------------|
| Simulator   | D 1 ↓             | D 2 ↓             | success           | D 1 ↓          | D 2 ↓          | success ↑      |
| Ours        | 0                 | 0.12              | 31.6%             | 0              | 0.12           | 26.3%          |
| Isaac Gym   | 0.48              | 0.67              | 17.5%             | 0.87           | 1.19           | 7.9%           |

To quantitatively analyze the grasp pose quality, we employ two metrics: (1) penetration distance D 1 , which measures the penetration depth between the object and the gripper, if any, and (2) absolute distance D 2 , defined as the unsigned distance between the object and the gripper to assess the grasp tightness. Fig. 6 illustrates the computation of these two metrics. Both metrics are computed using a Signed Distance Field (SDF) d o : R 3 → R constructed for the manipuland, with point-SDF distances computed by sampling 50,000 points (forming a set S h ) on the LEAP Hand's surface:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Using these two metrics, we evaluate the grasp quality produced by two simulators on a bottle subset (111 bottles) and the full 800-object dataset. As shown in Table I, our method achieves zero penetration cases, whereas Isaac Gym exhibits significant penetration artifacts (see Fig. 5), with an average penetration depth of 5 mm on the bottle subset and 8.7 mm across the full dataset. Additionally, the absolute distance in our method closely aligns with the IPC distance threshold parameter ( ˆ d = 10 -3 m) used for surface compliance, indicating that the resulting grasps are tightly secured. In contrast, poses generated with Issac Gym exhibit significantly larger absolute distances, suggesting that the gripper may fail to hold the manipuland firmly. These results demonstrate that our IPC-based simulator not only eliminates penetration artifacts but also achieves stable grasps, making it a more reliable choice for a high-quality grasping dataset.

b) Evaluation by Neural Grasp Generation: We further evaluate our dataset on the downstream grasp generation task. Specifically, we train the diffusion-based dexterous grasp generation model Unified Generative Grasping (UGG) [26] on two different datasets - one filtered using our IPC simulator and the other using Isaac Gym. We then validate the grasp poses with our IPC simulator to determine whether the generated poses can securely hold the manipulands without penetration. Table I presents the grasp success rate of UGG models trained on these two datasets. The model trained on our dataset achieves a success rate of 31.6% on the bottle subset and 26.3% on the full dataset, significantly outperforming the model trained on the Isaac Gym-filtered dataset. In the top row of Fig. 7, we visualize diverse grasping results generated by the UGG model trained on our dataset, demonstrating its ability to generate physically realistic and stable grasps. Furthermore, we extend the original UGG [26] model to support bimanual grasp generation and train it on two bimanual subsets: a bottle subset containing 48 bottle objects and a dispenser subset containing 50 dispenser objects. The generated bimanual results are showcased at the bottom of Fig. 7, further illustrating the versatility of our dataset in supporting complex grasping scenarios.

Fig. 7: ( Top ) Neural grasp generation using UGG [26] trained on our LEAP Hand unimanual dataset, demonstrating diverse grasp poses. ( Bottom ) The UGG network structure is modified to support bimanual grasp generation.

<!-- image -->

## D. Stress Force Prediction

Accurate stress prediction is essential for grasping and manipulating deformable objects to prevent damage and plastic deformation. Our dataset provides stress distributions for each grasp pose, enabling robust supervised learning for stress prediction. To demonstrate this, we train a neural network to predict the von Mises stress of a deformable object grasped by a UMI gripper, using its point cloud and the gripper mesh at the grasp pose as input.

Our model architecture follows [53], where we employ a Graph Neural Network (GNN) to extract features from both the object and the gripper. Additionally, a cross-attention module is incorporated to effectively capture hand-object interactions, enhancing stress prediction accuracy.

For training and evaluation, we utilize a subset of our GRIP dataset, consisting of eight bowls and eight mugs. Each category is split into a training set of six objects and a test set of two objects. Every object instance includes approximately 200 to 250 grasping poses. Due to the thin shell-like structures of these objects, their 3D-printed models will exhibit pronounced deformation during grasping, making them wellsuited for analysis in real-world experiments.

To evaluate our stress prediction network, we employ two metrics: the Relative Mean Absolute Error (Relative MAE) E Relative MAE = ∑ n i =1 | x i -y i | max i ∈ [ n ] | y i | , and Kullback-Leibler (KL) divergence E KL = ∑ n i =1 ˆ y i log( ˆ y i ˆ x i ) , where x i and y i represent pointwise von Mises stress of the prediction and the simulation results, respectively. ˆ x i and ˆ y i are the corresponding empirical probability distributions computed as ˆ x i = x i ∑ n i =1 x i and ˆ y i = y i ∑ n i =1 y i . As shown in Table II, our model achieves an average Relative MAE of approximately 7% ∼ 9% and an average KL divergence &lt; 0 . 2 in the test set, demonstrating strong consistency between the predicted stress values and the simulation results.

Fig. 8: Stress Distribution Analysis on Real-world Grasps : Each row shows a distinct grasp. The first and second columns display the grasps in the real world and simulation respectively. The third and fourth columns present the simulated and predicted normalized stress distributions.

<!-- image -->

To further assess the accuracy and effectiveness of our stress prediction network, we apply the real-to-sim pipeline described in Section III-D and conduct four grasping experiments using 3D-printed mug and bowl objects. Fig. 8 shows the normalized stress distribution predicted by our network alongside the simulation results, demonstrating strong agreement. For the grasp pose shown in the first row, the bowl undergoes significant deformation; rather, the grasp pose in the second row yields a moderate deformation. This aligns well with our network's predictions, where the mean stress in the first case is 2 . 17 times that in the second. Similarly, for the two mug grasps, the predicted mean stress of the former is 1.6 times that of the latter, in accordance with visual deformation. Both the qualitative and quantitative evaluations confirm the high accuracy of our stress prediction network.

## V. CONCLUSION

We introduce an IPC-based simulator optimized for largescale, highly efficient, parallel multi-environment simulations, addressing key challenges in simulating soft grippers and deformable object grasping. Additionally, we develop a fully automated pipeline for grasp generation, simulation, and evaluation, supporting a wide range of grippers and manipuland types. Enabled by this pipeline, we construct the GRIP dataset, a large-scale, diverse grasp dataset featuring both soft and rigid grippers, to advance research in neural grasp generation and soft object manipulation.

Experimental results demonstrate the effectiveness of our pipeline in synthesizing stable grasp poses and accurately predicting manipuland stress distributions. Our current GRIP

TABLE II: Relative Mean Absolute Error (MAE) and Kullback-Leibler (KL) divergence of our stress prediction network on the test set (a subset of the GRIP dataset) for the Bowl and Mug categories.

| Object Category   | Bowl   | Mug   |
|-------------------|--------|-------|
| Relative MAE      | 8.9%   | 7.2%  |
| KL divergence     | 0.11   | 0.18  |

dataset features only UMI and LEAP Hands, but our gripperagnostic pipeline can be easily extended to support diverse soft and rigid grippers. Through a real-to-simulation comparison, we further validate the strong agreement between our simulated results and real-world physics. However, our current grasp synthesis does not incorporate simulation feedback, as grasp generation is performed independently of the resulting dynamics. In future work, we aim to integrate differentiability into our pipeline to establish a closed-loop system, enabling dynamic feedback to refine grasp synthesis.

## REFERENCES

- [1] H.-S. Fang, C. Wang, M. Gou, and C. Lu, 'Graspnet-1billion: A largescale benchmark for general object grasping,' in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2020, pp. 11 444-11 453.
- [2] R. Wang, J. Zhang, J. Chen, Y. Xu, P. Li, T. Liu, and H. Wang, 'Dexgraspnet: A large-scale robotic dexterous grasp dataset for general objects based on simulation,' in 2023 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2023, pp. 11 359-11 366.
- [3] C. Chi, Z. Xu, C. Pan, E. Cousineau, B. Burchfiel, S. Feng, R. Tedrake, and S. Song, 'Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots,' arXiv preprint arXiv:2402.10329 , 2024.
- [4] N. Kuppuswamy, A. Alspach, A. Uttamchandani, S. Creasey, T. Ikeda, and R. Tedrake, 'Soft-bubble grippers for robust and perceptive manipulation,' in 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) . IEEE, 2020, pp. 9917-9924.
- [5] I. Huang, Y. Narang, C. Eppner, B. Sundaralingam, M. Macklin, T. Hermans, and D. Fox, 'Defgraspsim: Simulation-based grasping of 3d deformable objects,' arXiv preprint arXiv:2107.05778 , 2021.
- [6] M. Li, Z. Ferguson, T. Schneider, T. R. Langlois, D. Zorin, D. Panozzo, C. Jiang, and D. M. Kaufman, 'Incremental potential contact: intersection-and inversion-free, large-deformation dynamics.' ACM Trans. Graph. , vol. 39, no. 4, p. 49, 2020.
- [7] Y. Chen, M. Li, W. Lu, C. Fu, and C. Jiang, 'Midas: A multijoint robotics simulator with intersection-free frictional contact,' arXiv preprint arXiv:2210.00130 , 2022.
- [8] W. Du, S. Yao, X. Wang, Y. Xu, W. Xu, and C. Lu, 'Intersectionfree robot manipulation with soft-rigid coupled incremental potential contact,' IEEE Robotics and Automation Letters , 2024.
- [9] L. Lan, D. M. Kaufman, M. Li, C. Jiang, and Y. Yang, 'Affine body dynamics: fast, stable and intersection-free simulation of stiff materials,' ACM Transactions on Graphics (TOG) , vol. 41, no. 4, pp. 1-14, 2022.
- [10] Y. Chen, M. Li, L. Lan, H. Su, Y. Yang, and C. Jiang, 'A unified newton barrier method for multibody dynamics,' ACM Transactions on Graphics (TOG) , vol. 41, no. 4, pp. 1-14, 2022.
- [11] K. Shaw, A. Agarwal, and D. Pathak, 'Leap hand: Low-cost, efficient, and anthropomorphic hand for robot learning,' arXiv preprint arXiv:2309.06440 , 2023.
- [12] A. Ten Pas, M. Gualtieri, K. Saenko, and R. Platt, 'Grasp pose detection in point clouds,' The International Journal of Robotics Research , vol. 36, no. 13-14, pp. 1455-1473, 2017.
- [13] C. Eppner, A. Mousavian, and D. Fox, 'Acronym: A large-scale grasp dataset based on simulation,' in 2021 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2021, pp. 6222-6227.
- [14] D. Morrison, P. Corke, and J. Leitner, 'Egad! an evolved grasping analysis dataset for diversity and reproducibility in robotic manipulation,' IEEE Robotics and Automation Letters , vol. 5, no. 3, pp. 4368-4375, 2020.
- [15] H. Zhang, D. Yang, H. Wang, B. Zhao, X. Lan, J. Ding, and N. Zheng, 'Regrad: A large-scale relational grasp dataset for safe and objectspecific robotic grasping in clutter,' IEEE Robotics and Automation Letters , vol. 7, no. 2, pp. 2929-2936, 2022.
- [16] T. Wang, C. Yang, F. Kirchner, P. Du, F. Sun, and B. Fang, 'Multimodal grasp data set: A novel visual-tactile data set for robotic manipulation,' International Journal of Advanced Robotic Systems , vol. 16, no. 1, p. 1729881418821571, 2019.
- [17] A. D. Vuong, M. N. Vu, H. Le, B. Huang, H. T. T. Binh, T. Vo, A. Kugi, and A. Nguyen, 'Grasp-anything: Large-scale grasp dataset from foundation models,' in 2024 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2024, pp. 14 030-14 037.
- [18] H. Zhang, S. Christen, Z. Fan, O. Hilliges, and J. Song, 'Graspxl: Generating grasping motions for diverse objects at scale,' in European Conference on Computer Vision . Springer, 2024, pp. 386-403.
- [19] P. Li, T. Liu, Y . Li, Y . Geng, Y . Zhu, Y . Yang, and S. Huang, 'Gendexgrasp: Generalizable dexterous grasping,' in 2023 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2023, pp. 8068-8074.
- [20] L. F. Casas, N. Khargonkar, B. Prabhakaran, and Y. Xiang, 'Multigrippergrasp: A dataset for robotic grasping from parallel jaw grippers to dexterous hands,' arXiv preprint arXiv:2403.09841 , 2024.
- [21] Y. Shao and C. Xiao, 'Bimanual grasp synthesis for dexterous robot hands,' IEEE Robotics and Automation Letters , 2024.
- [22] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa et al. , 'Isaac gym: High performance gpu-based physics simulation for robot learning,' arXiv preprint arXiv:2108.10470 , 2021.
- [23] A. T. Miller and P. K. Allen, 'Graspit! a versatile simulator for robotic grasping,' IEEE Robotics &amp; Automation Magazine , vol. 11, no. 4, pp. 110-122, 2004.
- [24] T. Liu, Z. Liu, Z. Jiao, Y. Zhu, and S.-C. Zhu, 'Synthesizing diverse and physically stable grasps with arbitrary hand structures using differentiable force closure estimator,' IEEE Robotics and Automation Letters , vol. 7, no. 1, pp. 470-477, 2021.
- [25] W. Xu, J. Zhang, T. Tang, Z. Yu, Y. Li, and C. Lu, 'Dipgrasp: Parallel local searching for efficient differentiable grasp planning,' IEEE Robotics and Automation Letters , 2024.
- [26] J. Lu, H. Kang, H. Li, B. Liu, Y. Yang, Q. Huang, and G. Hua, 'Ugg: Unified generative grasping,' in European Conference on Computer Vision . Springer, 2024, pp. 414-433.
- [27] S. Christen, M. Kocabas, E. Aksan, J. Hwangbo, J. Song, and O. Hilliges, 'D-grasp: Physically plausible dynamic grasp synthesis for hand-object interactions,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2022, pp. 20 577-20 586.
- [28] C. Wu, J. Chen, Q. Cao, J. Zhang, Y. Tai, L. Sun, and K. Jia, 'Grasp proposal networks: An end-to-end solution for visual learning of robotic grasps,' Advances in Neural Information Processing Systems , vol. 33, pp. 13 174-13 184, 2020.
- [29] Y. Xu, W. Wan, J. Zhang, H. Liu, Z. Shan, H. Shen, R. Wang, H. Geng, Y. Weng, J. Chen et al. , 'Unidexgrasp: Universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2023, pp. 4737-4746.
- [30] D.-C. Hoang, J. A. Stork, and T. Stoyanov, 'Context-aware grasp generation in cluttered scenes,' in 2022 International Conference on Robotics and Automation (ICRA) . IEEE, 2022, pp. 1492-1498.
- [31] Z. Si, G. Zhang, Q. Ben, B. Romero, Z. Xian, C. Liu, and C. Gan, 'Difftactile: A physics-based differentiable tactile simulator for contact-rich robotic manipulation,' arXiv preprint arXiv:2403.08716 , 2024.
- [32] Y. Hu, J. Liu, A. Spielberg, J. B. Tenenbaum, W. T. Freeman, J. Wu, D. Rus, and W. Matusik, 'Chainqueen: A real-time differentiable physical simulator for soft robotics,' in 2019 International conference on robotics and automation (ICRA) . IEEE, 2019, pp. 6265-6271.
- [33] Z. Zong, C. Jiang, and X. Han, 'A convex formulation of frictional contact for the material point method and rigid bodies,' in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) . IEEE, 2024, pp. 1831-1838.
- [34] F. Ficuciello, A. Migliozzi, E. Coevoet, A. Petit, and C. Duriez, 'Fem-based deformation control for dexterous manipulation of 3d soft
35. objects,' in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) . IEEE, 2018, pp. 4007-4013.
- [35] I. Huang, Y. Narang, C. Eppner, B. Sundaralingam, M. Macklin, R. Bajcsy, T. Hermans, and D. Fox, 'Defgraspsim: Physics-based simulation of grasp outcomes for 3d deformable objects,' IEEE Robotics and Automation Letters , vol. 7, no. 3, pp. 6274-6281, 2022.
- [36] X. Yu, S. Zhao, S. Luo, G. Yang, and L. Shao, 'Diffclothai: Differentiable cloth simulation with intersection-free frictional contact and differentiable two-way coupling with articulated rigid bodies,' in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) . IEEE, 2023, pp. 400-407.
- [37] M. A. Homel, R. M. Brannon, and J. Guilkey, 'Controlling the onset of numerical fracture in parallelized implementations of the material point method (mpm) with convective particle domain interpolation (cpdi) domain scaling,' International Journal for Numerical Methods in Engineering , vol. 107, no. 1, pp. 31-48, 2016.
- [38] X. Wang, Y. Qiu, S. R. Slattery, Y. Fang, M. Li, S.-C. Zhu, Y. Zhu, M. Tang, D. Manocha, and C. Jiang, 'A massively parallel and scalable multi-gpu material point method,' ACM Transactions on Graphics (TOG) , vol. 39, no. 4, pp. 30-1, 2020.
- [39] H. D. Espinosa, P. D. Zavattieri, and G. L. Emore, 'Adaptive fem computation of geometric and material nonlinearities with application to brittle failure,' Mechanics of Materials , vol. 29, no. 3-4, pp. 275305, 1998.
- [40] G. Irving, J. Teran, and R. Fedkiw, 'Invertible finite elements for robust simulation of large deformation,' in Proceedings of the 2004 ACM SIGGRAPH/Eurographics symposium on Computer animation , 2004, pp. 131-140.
- [41] M. Anitescu and A. Tasora, 'An iterative approach for cone complementarity problems for nonsmooth dynamics,' Computational Optimization and Applications , vol. 47, pp. 207-235, 2010.
- [42] K. Yamane and Y. Nakamura, 'A numerically robust lcp solver for simulating articulated rigid bodies in contact,' Proceedings of robotics: science and systems IV, Zurich, Switzerland , vol. 19, p. 20, 2008.
- [43] E. Todorov, T. Erez, and Y. Tassa, 'Mujoco: A physics engine for model-based control,' in 2012 IEEE/RSJ international conference on intelligent robots and systems . IEEE, 2012, pp. 5026-5033.
- [44] E. Drumwright and D. A. Shell, 'A robust and tractable contact model for dynamic robotic simulation,' in Proceedings of the 2009 ACM symposium on Applied Computing , 2009, pp. 1176-1180.
- [45] M. Li, C. Jiang, and Z. Luo, Physics-Based Simulation , 2024. [Online]. Available: https://phys-sim-book.github.io/
- [46] C. M. Kim, M. Danielczuk, I. Huang, and K. Goldberg, 'Ipc-graspsim: Reducing the sim2real gap for parallel-jaw grasping with the incremental potential contact model,' in 2022 International Conference on Robotics and Automation (ICRA) . IEEE, 2022, pp. 6180-6187.
- [47] W. Du, W. Xu, J. Ren, Z. Yu, and C. Lu, 'Tacipc: Intersectionand inversion-free fem-based elastomer simulation for optical tactile sensors,' IEEE Robotics and Automation Letters , vol. 9, no. 3, pp. 2559-2566, 2024.
- [48] J. A. Fern´ andez-Fern´ andez, R. Lange, S. Laible, K. O. Arras, and J. Bender, 'Stark: A unified framework for strongly coupled simulation of rigid and deformable bodies with frictional contact,' in 2024 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2024, pp. 16 888-16 894.
- [49] Y. Hu, T. Schneider, B. Wang, D. Zorin, and D. Panozzo, 'Fast tetrahedral meshing in the wild,' ACM Transactions on Graphics (ToG) , vol. 39, no. 4, pp. 117-1, 2020.
- [50] F. Xiang, Y. Qin, K. Mo, Y. Xia, H. Zhu, F. Liu, M. Liu, H. Jiang, Y. Yuan, H. Wang et al. , 'Sapien: A simulated part-based interactive environment,' in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2020, pp. 11 097-11 107.
- [51] S. Garrido-Jurado, R. Mu˜ noz-Salinas, F. J. Madrid-Cuevas, and M. J. Mar´ ın-Jim´ enez, 'Automatic generation and detection of highly reliable fiducial markers under occlusion,' Pattern Recognition , vol. 47, no. 6, pp. 2280-2292, 2014.
- [52] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa et al. , 'Isaac gym: High performance gpu-based physics simulation for robot learning,' arXiv preprint arXiv:2108.10470 , 2021.
- [53] M. Saleh, M. Sommersperger, N. Navab, and F. Tombari, 'Physicsencoded graph neural networks for deformation prediction under contact,' in 2024 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2024, pp. 17 160-17 166.