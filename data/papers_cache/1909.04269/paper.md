## GlassLoc: Plenoptic Grasp Pose Detection in Transparent Clutter

Zheming Zhou

Tianyang Pan

Shiyu Wu

Haonan Chang

Odest Chadwicke Jenkins

Abstract -Transparent objects are prevalent across many environments of interest for dexterous robotic manipulation. Such transparent material leads to considerable uncertainty for robot perception and manipulation, and remains an open challenge for robotics. This problem is exacerbated when multiple transparent objects cluster into piles of clutter. In household environments, for example, it is common to encounter piles of glassware in kitchens, dining rooms, and reception areas, which are essentially invisible to modern robots. We present the GlassLoc algorithm for grasp pose detection of transparent objects in transparent clutter using plenoptic sensing. GlassLoc classifies graspable locations in space informed by a Depth Likelihood Volume (DLV) descriptor. We extend the DLV to infer the occupancy of transparent objects over a given space from multiple plenoptic viewpoints. We demonstrate and evaluate the GlassLoc algorithm on a Michigan Progress Fetch mounted with a first generation Lytro. The effectiveness of our algorithm is evaluated through experiments for grasp detection and execution with a variety of transparent glassware in minor clutter.

## I. INTRODUCTION

Robot grasping in household environments is challenging because of sensor uncertainty, scene complexity and actuation imprecision. Recent results suggest that Grasp Pose Detection (GPD) using point cloud local features [27] and manually labeled grasp confidence [17] can be applied in generating feasible grasp poses over a wide range of objects. However, domestic environments include a great amount of transparent objects, ranging from kitchen utilities (e.g. wine cups and containers) to house decoration (e.g. windows and tables). The reflective and transparent material on those objects will produce invalid readings from depth camera. This problem becomes more significant in the real world where there are piled transparent objects which will lead to unexpected robot manipulation behaviors if the robot was trying to interact with the objects. A correct estimation of transparency is necessary to protect the robot from performing hazardous actions and extend robot applications to more challenging scenarios.

The problem of performing grasping in transparent clutter is complicated by the fact that robots cannot perceive and describe the transparent surfaces correctly. Several previous methods [14], [15] tried to approach this problem by finding invalid values in depth observation, but they were limited to top-down grasping and made assumption that target objects establish distinguishable contour (formed by invalid points) in depth map. Recently, several approaches employed light

Z. Zhou, T. Pan, S. Wu, H. Chang, and O.C. Jenkins are with the Department of Electrical Engineering and Computer Science, Robotics Institute, University of Michigan, Ann Arbor, MI, USA, 48109-2121 [zhezhou|typan|shiyuwu|harveych|ocj]@umich.edu field camera to observe the transparency and showed promising results. Zhou et al. [30] used single shot light field image to form a new plenoptic descriptor named Depth Likelihood Volume (DLV). They succeeded in estimating the pose of single transparent object or object behind translucent surface by given the corresponding object CAD model. Based on that, we extend the idea to a more general-purpose grasp detection scenario with transparent objects clutter.

Fig. 1: (Top) a robot using GlassLoc to pick up transparent objects from clutter and place on the tray. The robot is observing the scene using a light field camera. Grasp candidate is sampled in DLV (bottom left) and mapped to the world frame in the visualizer (bottom middle). The robot successfully picks up a transparent cup from the clutter (bottom right).

<!-- image -->

We make several contributions in this paper. First, we propose GlassLoc algorithm for detecting six-DoF grasp poses of transparent objects in both separated and minor overlapping cluttered environments. Next, we propose a generalized model for constructing Depth Likelihood Volume from multi-view light field observations with multiray fusion and reflection suppression. Finally, we integrate our algorithm with a robot manipulation pipeline to perform tabletop pick and place tasks over eight scenes and five different transparent objects. Our results show that the grasping success rate over all test objects is 81% in 220 grasp trials.

Fig. 2: An overview of GlassLoc framework. A light field camera is mounted on the end-effector of the robot. After taking a set of light field observations by moving robot arms, sub-aperture images are extracted (center view is highlighted in red). The Depth Likelihood Volume (DLV) is then computed as a 3D volume of depth likelihoods over transparent clutter. Given gripper configuration, we can sample grasp poses in DLV and extract grasp features for the classifier to label whether the samples are graspable or not.

<!-- image -->

## II. RELATED WORK

## A. Grasp Perception In Clutter

It remains a challenging task for robots to perform perception and manipulation in cluttered environments considering the complexity of the real world. We consider there are two major categories of methods for robots to perform grasp perception in clutter. The first category is modelbased pose estimation methods. By estimating object poses, grasp configurations calculated based on the local model can be further transformed to the robot environments. Collet et al. [3] utilized color information to estimate poses of object in cluttered environments. Their proposed algorithm clusters and then matches the local color patch from object model to robot observations to generate pose hypotheses. Sui et al. [25], [24] constructed generative models to evaluate pose hypotheses against point cloud using object CAD models. The generative models perform object detection followed by particle filtering for robot grasping in the highly cluttered tabletop environments. With a similar idea, Papazov et al. [20] leveraged RANSAC-based bottom-up approach with Iterative Closest Point registration to fit 3D geometries to the observed point cloud.

On the other hand, rather than associating a grasp pose with a certain object model, Grasp Pose Detection (GPD) tries to characterize grasp poses based on the local geometry or appearance features directly from observations. Several early works [12], [21] represented the grasp poses as oriented rectangles in RGB-D observations. Further, given a number of manually-labelled grasp candidates, the system will learn to predict whether a sampled rectangle is graspable or not. One major restriction of those systems is that the approaching directions of generated grasp candidates need to be orthogonal to the RGB-D sensor plane. Fischinger and Vincze [5] tried to lessen the restriction by integrating hightmap-based features. They also designed a heuristic for ranking the grasp candidates in a clutter bin settings. ten Pas and Platt [26] directly detected grasp poses in SE (3) space by estimating curvatures and extracting handle-like features in local point cloud neighborhoods. Gualtieri et al. [7] proposed more types of local point cloud features for grasp representation and projected those features to 2D image space for classification. Our work with GlassLoc extends these ideas to transparent clutter with a different grasp representation and a new plenoptic descriptor.

## B. Light Field Photography

The models describing the light field rendering proposed by Levoy and Hanrahan [13] introduced foundations of light field captured from multi-view cameras. Based on this work, [18], [6] succeeded in producing commercial level hand-held light field camera using the microlens array struc- ture. Building on the property that the plenoptic camera can capture both intensity and direction of light rays, light field photography has shown significant advancement in different applications. Wang et al. [29] explicitly modeled the light field image pixel angular consistency to generate accurate depth map for the object with occlusion edges. Jeon et al. [8] performed sub-pixel shifting in image frequency domain in tackling the microlens camera narrow baseline problem for accurate depth estimation. Maeno et al. [16] introduced distortion feature in light field to detect and recognize the transparent object. Johannsen et al. [9] leveraged multiview light field images to reconstruct multi-layer translucent scenes. Skinner and Johnson-Roberson [22] introduced a light propagation model suited to underwater perception using plenoptic observations.

The use of light field perception in robotics is still relatively new. Oberlin and Tellex [19] proposed a time-lapse light field capturing pipeline for static scenes by mounting a RGB camera on the end-effector of the robot and moving in a designed trajectory. Tsai et al. [28] introduced a algorithm for distinguishing refracted and Lambertian features from light field image. Zhou et al. [30] used a Lytro camera to take a single shot of the scene and construct a plenoptic descriptor over that. Given the target object model, their methods can estimate single object six-DoF pose in layered translucent scenes. Our GlassLoc pipeline extends the idea proposed in [30] for more general-purpose manipulation over transparent clutter.

## III. PROBLEM FORMULATION AND APPROACH

GlassLoc addresses the problem of grasp pose detection for transparent objects in clutter from plenoptic observations. For a given static scene, we assume there is a latent set of end-effector poses G ⊂ SE (3) that will produce a successful grasp of an object. A successful grasp is assumed to result in the robot obtaining force closure on an object when it moves gripper and closes its fingers. The plenoptic grasp pose detection problem is then phrased as estimating a representative set of valid sample grasp poses G v ⊂ G .

Within the grasp pose detection problem, a major challenge is how to classify whether a grasp pose is a member of G , and, thus, will result in a successful manipulation. For grasp pose classification, we assume as given robot endeffector pose q ∈ SE (3) and a collection of observations Z from a plenoptic sensor. It is assumed that each observation z 1: N ∈ Z captures a raw light field image o i of a static scene from camera viewpoint v i ⊂ SE (3) . The classification result calculated from these inputs is a likelihood l ∈ [0 , 1] that relates the probability of end-effector pose, q , resulting in a successful grasp. Described later, our implementation of GlassLoc will perform the classification using a neural network.

Illustrated in Figure 3, grasp pose classification within GlassLoc is expressed as a function l = M ( U ) that maps transparency occupancy likelihood features U to grasp pose confidence l . Transparency occupancy features U ( q, D ) are computed with respect to the subset of a Depth Likelihood Volume (DLV) D that is within the graspable volume of pose q . The DLV estimates how likely a point p ∈ R 3 belongs to a transparent surface. To test all sampled grasps, a Depth Likelihood Volume D is computed from observations Z over an entire grasping workspace P ⊂ SE (3) within the visual hull of v 1: N . We assume the grasping workspace is discretized into p 1: M ∈ P a set of 3D points, with each element of this set expressed as p i = ( x i , y i , z i ) .

Fig. 3: Example of DLV value calculation of two randomly sampled points ( x 1 , y 1 , z 1 ) and ( x 2 , y 2 , z 2 ) through examining the ray consistency in different view points. Each sample point corresponds to different pixel indices with depths in different center view plane I v 0 and I v 1 .

<!-- image -->

## IV. PLENOPTIC GRASP POSE DETECTION METHODS

An outline of the GlassLoc algorithm is described in Algorithm 1. GlassLoc begins by computing a Depth Likelihood Volume from multi-view light field observations. By integrating different views, we can further post-process the DLV by suppressing reflection caused by non-Lambertian surfaces. Details of DLV construction are presented in Section IV-A and IV-B. In Step 2, we uniformly sample the grasp candidates C = { c j ∈ P } in workspace P . For each grasp candidate, we extract grasp representations (see Section IV-C) and corresponding transparency likelihood features given the robot gripper parameter θ . The generated features will then be classified with a grasp success labels and confidence scores by a neural network. The training data generation strategy for learning this mapping is introduced in Section IV-D. Given classified grasp poses, we use a multihypothesis particle-based search to find a set of end-effector poses with high confidence for successful grasp execution (see Section IV-E). The finalized set of grasp poses will be ready for the robot to perform grasping.

## A. Multi-view Depth Likelihood Volume

The Depth Likelihood Volume (DLV) is a volume-based plenoptic descriptor which represents the depth of a light

## Algorithm 1 GlassLoc Plenoptic Grasp Pose Detection

INPUT: a set of light field observations Z , robot gripper parameter θ

OUTPUT: a set of valid sample grasp poses G v

<!-- formula-not-decoded -->

field image pixel as a likelihood function rather than a deterministic value. The advantage of this representation is to keep the transparent scene structure by assigning different likelihoods to surfaces with different transparency. In [30], DLV is formulated in a specific camera frame indexed with pixel coordinates and depths. The formulation is restricted to single-view scenarios. In this paper, we generalize the expression which takes sample points in 3-D space as input and integrates multi-view light field observations.

The DLV is defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where L ( p ) is the depth likelihood of sampled points p . A is the set of sub-aperture images. ρ v i ( p ) is a light ray that goes through or emitted from point p and is received by view point v i at ( i, j ) in center view image plane. N indicates the number of view points in observations. F a,d ( ρ ) is the triangulation function finding the light ray corresponding to ρ in sub-aperture images indexed with a that yields depth d . d can be explicitly calculated using camera intrinsic matrix given point and view point. ||· , ·|| is the ray difference which is calculated by color and color gradient differences. Denote s = ∑ a ∈ A \ I v i T a,d ( ρ v i ( p )) , then f ( s ) is a normalization function mapping color cost to likelihood. There are multiple choices of f ( s ) . In our implementation, we choose:

<!-- formula-not-decoded -->

To better explain the formulation presented above, we consider the example shown in Figure 3. A cluster of transparent objects are placed on a table with opaque surface. We have two light field observations z 0 = { o 0 , v 0 } and z 1 = { o 1 , v 1 } with center view image plane I v 0 and I v 1 respectively. There are two points p 1 = ( x 1 , y 1 , z 1 ) and p 2 = ( x 2 , y 2 , z 2 ) sampled in the space and each of them emits light rays captured by both views. In view I v 0 , Ray ρ 1 emitted from both points are received by the same pixel ( i 1 , j 1 ) , while ρ 2 and ρ 3 are received by ( i 2 , j 2 ) and ( i 3 , j 3 )

Fig. 4: Example DLV feature image before (middle) and after (right) reflection suppression. The center view of part of raw observation is shown in (left). The intensity of pixel in the gray-scale image (middle and right) indicates the likelihood value. The high likelihood region caused by specular light is suppressed.

<!-- image -->

respectively. Then we can express the depth likelihood of point p 2 as:

<!-- formula-not-decoded -->

Function T calculates the color and the color gradient difference between center view (rectangle with solid line in Figure 3) and sub-aperture view (rectangle with dot line in Figure 3). The location of red pixel is calculated by function F . For micro-lens based light field camera, the pixel shift between center and sub-aperture images are usually in sub-pixel level. The realization of F function is based on frequency domain sub-pixel shifting method proposed in [8].

## B. Reflection Suppression

A transparent surface produces non-Lambertian reflectance, which induces specular highlight to light field observations. Those shiny spots tend to produce the saturated color or virtual surface with larger depth than the actual transparent surface. This phenomenon will generate a high likelihood region in DLV that indicates a non-existing surface. To deal with this problem, we calculate the variance of ray differences for DLV points which has saturated color and high likelihood over different view points:

<!-- formula-not-decoded -->

where E { ρ V ( p ) } can be expressed as:

<!-- formula-not-decoded -->

where N ( A ) is the number of sub-aperture images extracted from raw light field image. For a point p that has variance larger than a threshold τ , we check whether it has the largest likelihood value among all other points that lie on the light rays it emits out. Specifically, we first find light rays emitted from p and received by pixel ( i, j ) with depth d that has large variance over different view points. Then we locate all light rays received by ( i, j ) with depth less than d , and check whether the following equation holds:

Fig. 5: Training data generation procedure. (a) The glass cup is wrapped with opaque tape for depth sensor to get point cloud. (b) Grasp candidates are generated based on pointcloud-based method and local-to-world transform. (c) The glass cup is placed at the same pose to take multiple light field observations. (d) Grasp candidates generated from point cloud are mapped to DLV.

<!-- image -->

<!-- formula-not-decoded -->

If Equation 7 holds, it indicates this light ray has high possibility of coming from strong reflection area and will be excluded from the calculation of DLV. Figure 4 (left) is the sliced feature from DLV before reflective suppression which we can observe incorrect large values caused by specular highlight. Figure 4 (right) shows the result after processing and the previous high value area is suppressed.

## C. Grasp Representation and Classification

We represent a graspable area as a 3D cuboid with length, width, and height as L, W, H respectively. The width and height of the cuboid is equal to the width and height of the volume when the robot finger close while the length is extended for capturing more feature spaces. The cuboid is voxelized into l × w × h grid, and for each grid we interpolate the likelihood value by finding the nearest eight points in DLV. Rather than feeding into classifier with a large amount of points, we extract 2D features from the volume by projection and slicing.

We first define the three axes of the graspable volume. The x axis of the volume is defined as the approach direction of the gripper. The z axis is defined along the direction the gripper fingers close along. The y axis is the cross product of the previous two axes. We then calculate three types of features and project them to the three axes: a center slice of likelihood volume, I c , an average likelihood map over all points, I a , a sliced difference likelihood map, I d , which is calculated by recursively comparing the difference between current slice of the graspable volume with the previous slice. More specifically, we can express the three types of feature as follows (take projection to x axis as example):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We resize the images to the same size and concatenate them into different channels. Since we have three types of features and three axes to project, we have nine channels in total.

For classifier, we use the LeNet [11] structure which is a common structure for grasp pose classification and ranking [7], [10]. The output of the classifier is the binary label { graspable, not graspable } associated with the confidence scores.

## D. Training Data Generation

For depth-based grasp pose detection algorithms, the training data generation process relies on grasp pose sampling and labeling on point cloud. Unfortunately, depth sensors cannot provide correct point cloud for transparent objects. Instead, we wrap the object with opaque material and generate training samples by mapping grasp poses from point cloud to DLV. The detailed steps are illustrated in Figure 5 (a) (d).

We have two sources to produce training samples from point cloud. One is depth-based grasp pose detection algorithms. We input those algorithms with our depth observations and label the result grasp candidates as { graspable } . In the meantime, we restore the grasp poses filtered out in those algorithms and label them as { not graspable } . The other is transforming pre-defined grasp pose in the local frame to the observation. By checking the gripper collision with the environment, we label the collision free grasp poses as { graspable } and the others as { not graspable } .

## E. Grasp Search

After we perform classification of our samples, we try to find a graspable region with relatively high classification confidence score. Our grasp optimization builds on the particle filtering work proposed by Dellaert et al. [4], which is based on sequential Bayesian filter:

<!-- formula-not-decoded -->

where the weighted particles { q ( j ) t , w ( j ) t } n j =1 represent the sampled six-DoF grasp poses with confidence score given by classifier. The initial hypothesis of particles q ( j ) t are uniformly generated in the 3D workspace with the identical weights. For each hypothesis, we extract the grasp features and compute the weight w ( j ) t by normalizing the confidence score output by classifier. Importance sampling is then performed with resampling process to concatenate grasp hypothesis to high weights region. In our case, we don't have actual action between two states, instead, we model the state transition in action model as zero-mean Gaussian noise over SE (3) . In other words, after we obtain resampled grasp poses (particles), we diffuse the particles by adding Gaussian noise over ( x, y, z, row, pitch, yaw ) to generate the new set of particles. Our convergence criterion is a fixed number of iterations.

TABLE I: Results of manipulation experiments for eight scenes. The first row shows the number of object in the scene. Number of manipulation runs shown in row two refers to the task runs for the scene. The object grasp percentage refers to successful picking ratio over all trials for each scene.

|                             |   scene (a) |   scene (b) |   scene (c) |   scene (d) |   scene (e) |   scene (f) |   scene (g) |   scene (h) |
|-----------------------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| Number of Total Objects     |           2 |           2 |           2 |           2 |           3 |           3 |           4 |           4 |
| Number of Manipulation Runs |          10 |          10 |          10 |          10 |          10 |          10 |          10 |          10 |
| Object Grasp Percentage     |        0.70 |        0.80 |         1.0 |        0.75 |        0.87 |        0.43 |         1.0 |        0.85 |

Fig. 6: Training and testing objects for evaluating our GlassLoc algorithm. Two objects are used in training: wine cup and short cup (wrapped object for generating point cloud). Five objects are used in testing: wine cup, toothbrush holder, spoon, short cup, and tall cup.

<!-- image -->

## V. RESULTS

TABLE II: Object-wise grasp performance

| Object            |   Trials |   Success Rate |
|-------------------|----------|----------------|
| Toothbrush Holder |       60 |           0.92 |
| Wine Cup          |       50 |           0.82 |
| Short Cup         |       40 |           0.65 |
| Tall Cup          |       40 |           0.88 |
| Spoon             |       30 |           0.70 |
| Overall           |      220 |           0.81 |

the meantime, the robot will record the camera view pose based on the current transformation from robot base to the camera. The Lytro camera intrinsic calibration and distortion correction is conducted using the toolbox created by Bok et al. [2]. The raw light field image is then decomposed into 9 × 9 sub-aperture images with resolution of 328 × 328 pixels. The boundary sub-aperture images usually have strong color noise because of the lens edge affect. In our implementation, we only keep 7 × 7 sub-aperture images and for each image. For each image, we crop 4 pixels at the margin.

We use two objects to construct our training samples: wine cup and short cup (Figure 6). We generate approximate 10k positive grasp samples and 15k negative grasp samples from 50 scenes containing one or more object instances. For each grasp sample, we extract corresponding graspable volume from DLV with actual size 0 . 10 × 0 . 10 × 0 . 06 ( meters ) and grid density 100 × 100 × 60 ( points ) . We further extract gray-scale image features and resize them into 100 × 100 . Features are concatenated into nine channels and trained on LeNet structure. We keep the default structure and parameter settings of LeNet implementation in Tensorflow except the number of nodes in the output layer (2 in our case).

The DLV construction algorithm is implemented in MATLAB with parallel computing. A DLV is sampled in a 1 . 0 × 1 . 0 × 1 . 0 ( meters ) box with grid density at 1000 × 1000 × 1000 ( points ) .

In grasp search step, we use 100 particles with 100 iterations in our experiment. The covariance for diffusing grasp pose after each filtering iteration is set to 10 -4 ( meter 2 ) and 0 . 03 ( rad 2 ) for translation and rotation respectively.

Our implementation takes 2 minutes per view to extract sub-aperture images and 10 minutes to construct DLV on an unoptimized MATLAB code. The light field image decoding

## A. Experimental Setup

To evaluate GlassLoc , we ran a series of experiments with a first generation Lytro camera and a Michigan Progress Fetch robot. The Lytro camera is mounted on the wrist of the robot and triggered by on-chip Wi-Fi to take images. In and ray corresponding are the current bottlenecks.

Fig. 7: Eight scenes for evaluating GlassLoc pipeline. We randomly choose a number of transparent objects from the test set and put them on the table for the robot to perform manipulation on.

<!-- image -->

Fig. 8: The robot successfully picks and places all transparent objects in scene (g). Each column shows the pick and place action over one object in the scene.

<!-- image -->

## B. Evaluation

We evaluate our GlassLoc manipulation pipeline on eight transparent clutter scenes as shown in Figure 7. In each scene, the number of objects ranges from two to four with different pose configurations. For each manipulation run, light field images are taken from two camera poses to construct DLV. After particle filtering reaches the convergence criterion, we randomly select one grasp pose and send it to the execution module. Our robot motion planning and execution module is built on TRAC-IK [1] and MoveIt! [23].

For each scene, we perform 10 manipulation runs. We will terminate one run whenever all objects are successfully picked or the number of manipulation trials exceed the number of objects.

The manipulation results of each scene are established in Table I. Object grasp percentage is calculated based on how many objects have been successfully picked over the total number of objects that should be picked in all runs of a scene. We also show the pick success rate for each object in Table II.

Table I shows that the object grasp percentage is over 75% in most of the scenes. Our GlassLoc algorithm can generate enough reliable grasp poses based on our DLV constructed from light field observations in complex scenes where four transparent objects are randomly cluttered. The grasp percentages of these two scenes are 100% and 85% respectively.

Notably, our overall grasp success rate is 81% for the transparent cluttered environments in 220 grasps. During our experiment, we find that the short cup has the lowest grasp success rate. In most cases, it was squeezed and then slipped out from the gripper. The reason is two fold: one is that the surface of the short cup is sharply tilted, which prevents the robot from performing force closure grasping, the other is that the parallel jaw gripper hasn't been equipped with force sensors and is likely to squeeze the cup.

## VI. CONCLUSION

In this paper, we have contributed the GlassLoc algorithm for robot manipulation in transparent clutter. We use multiview light field observations to construct the Depth Likelihood Volume as a plenoptic descriptor to characterize the environments with multiple transparent objects. We show that by our algorithm, the robot is able to perform accurate grasping in tabletop transparent cluttered environments.

## REFERENCES

- [1] P. Beeson and B. Ames. Trac-ik: An open-source library for improved solving of generic inverse kinematics. In IEEE-RAS International Conference on Humanoid Robots , 2015.
- [2] Y. Bok, H.-G. Jeon, and I. S. Kweon. Geometric calibration of microlens-based light field cameras using line features. IEEE transactions on pattern analysis and machine intelligence , 39(2):287-300, 2017.
- [3] A. Collet, M. Martinez, and S. S. Srinivasa. The moped framework: Object recognition and pose estimation for manipulation. Int. J. Rob. Res. , 30(10):1284-1306, Sept. 2011.
- [4] F. Dellaert, D. Fox, W. Burgard, and S. Thrun. Monte carlo localization for mobile robots. In IEEE International Conference on Robotics and Automation (ICRA) , May 1999.
- [5] D. Fischinger and M. Vincze. Empty the basket-a shape based learning approach for grasping piles of unknown objects. In IEEE/RSJ International Conference on Intelligent Robots and Systems , pages 2051-2057. IEEE, 2012.
- [6] T. Georgiev, Z. Yu, A. Lumsdaine, and S. Goma. Lytro camera technology: theory, algorithms, performance analysis. In Multimedia Content and Mobile Devices , volume 8667, page 86671J. International Society for Optics and Photonics, 2013.
- [7] M. Gualtieri, A. Ten Pas, K. Saenko, and R. Platt. High precision grasp pose detection in dense clutter. In 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , pages 598-605. IEEE, 2016.
- [8] H.-G. Jeon, J. Park, G. Choe, J. Park, Y. Bok, Y.-W. Tai, and I. So Kweon. Accurate depth map estimation from a lenslet light field camera. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 1547-1555, 2015.
- [9] O. Johannsen, A. Sulc, N. Marniok, and B. Goldluecke. Layered scene reconstruction from multiple light field camera views. In S.-H. Lai, V. Lepetit, K. Nishino, and Y. Sato, editors, Computer Vision - ACCV 2016 , pages 3-18, Cham, 2017. Springer International Publishing.
- [10] D. Kappler, J. Bohg, and S. Schaal. Leveraging big data for grasp planning. In IEEE International Conference on Robotics and Automation (ICRA) , pages 4304-4311. IEEE, 2015.
- [11] Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, et al. Gradient-based learning applied to document recognition. Proceedings of the IEEE , 86(11):2278-2324, 1998.
- [12] I. Lenz, H. Lee, and A. Saxena. Deep learning for detecting robotic grasps. The International Journal of Robotics Research , 34(4-5):705724, 2015.
- [13] M. Levoy and P. Hanrahan. Light field rendering. In Proceedings of the 23rd annual conference on Computer graphics and interactive techniques , pages 31-42. ACM, 1996.
- [14] I. Lysenkov. Recognition and pose estimation of rigid transparent objects with a kinect sensor. Robotics , 273, 2013.
- [15] I. Lysenkov and V. Rabaud. Pose estimation of rigid transparent objects in transparent clutter. In IEEE International Conference on Robotics and Automation (ICRA) , pages 162-169. IEEE, 2013.
- [16] K. Maeno, H. Nagahara, A. Shimada, and R.-i. Taniguchi. Light field distortion feature for transparent object recognition. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , pages 2786-2793. IEEE, 2013.
- [17] J. Mahler, J. Liang, S. Niyaz, M. Laskey, R. Doan, X. Liu, J. A. Ojea, and K. Goldberg. Dex-net 2.0: Deep learning to plan robust grasps with synthetic point clouds and analytic grasp metrics. arXiv preprint arXiv:1703.09312 , 2017.
- [18] R. Ng. Digital light field photography . stanford university California.
- [19] J. Oberlin and S. Tellex. Time-lapse light field photography for perceiving transparent and reflective objects. 2017.
- [20] C. Papazov, S. Haddadin, S. Parusel, K. Krieger, and D. Burschka. Rigid 3d geometry matching for grasping of known objects in cluttered scenes. The International Journal of Robotics Research , page 0278364911436019, 2012.
- [21] J. Redmon and A. Angelova. Real-time grasp detection using convolutional neural networks. In IEEE International Conference on Robotics and Automation (ICRA) , pages 1316-1322. IEEE, 2015.
- [22] K. A. Skinner and M. Johnson-Roberson. Towards real-time underwater 3d reconstruction with plenoptic cameras. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , pages 2014-2021. IEEE, 2016.
- [23] I. A. Sucan and S. Chitta. Moveit! Online Availabl e: http://moveit. ros. org , 2013.
- [24] Z. Sui, L. Xiang, O. C. Jenkins, and K. Desingh. Goal-directed robot manipulation through axiomatic scene estimation. The International Journal of Robotics Research , 36(1):86-104, 2017.
- [25] Z. Sui, Z. Zhou, Z. Zeng, and O. C. Jenkins. Sum: Sequential scene understanding and manipulation. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , pages 32813288, Sept 2017.
- [26] A. ten Pas and R. Platt. Using geometry to detect grasp poses in 3d point clouds. In International Symposium on Robotics Research , 2015.
- [27] A. ten Pas and R. Platt. Localizing handle-like grasp affordances in 3d point clouds. In Experimental Robotics , pages 623-638. Springer, 2016.
- [28] D. Tsai, D. G. Dansereau, T. Peynot, and P. Corke. Distinguishing refracted features using light field cameras with application to structure from motion. IEEE Robotics and Automation Letters , 4(2):177-184, 2018.
- [29] T.-C. Wang, A. A. Efros, and R. Ramamoorthi. Occlusion-aware depth estimation using light-field cameras. In IEEE International Conference on Computer Vision (ICCV) , pages 3487-3495. IEEE, 2015.
- [30] Z. Zhou, Z. Sui, and O. C. Jenkins. Plenoptic monte carlo object localization for robot grasping under layered translucency. In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , pages 1-8. IEEE, 2018.