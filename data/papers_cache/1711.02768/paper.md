## Bayesian Scale Estimation for Monocular SLAM Based on Generic Object Detection for Correcting Scale Drift

Edgar Sucar 1 and Jean-Bernard Hayet 1

Abstract -This work proposes a new, online algorithm for estimating the local scale correction to apply to the output of a monocular SLAM system and obtain an as faithful as possible metric reconstruction of the 3D map and of the camera trajectory. Within a Bayesian framework, it integrates observations from a deep-learning based generic object detector and a prior on the evolution of the scale drift. For each observation class, a predefined prior on the heights of the class objects is used. This allows to define the observations likelihood. Due to the scale drift inherent to monocular SLAM systems, we integrate a rough model on the dynamics of scale drift. Quantitative evaluations of the system are presented on the KITTI dataset, and compared with different approaches. The results show a superior performance of our proposal in terms of relative translational error when compared to other monocular systems.

## I. INTRODUCTION

Monocular simultaneous localization and mapping (SLAM) is a classical problem that has been tackled in various forms in the robotics and computer vision communities for more than 15 years. Starting from the seminal work of Davison [3], impressive results have been obtained in the construction of sparse or semi-dense 3D maps and in visual odometry [2], [11], [4], with a single camera. Given the availability and low price of this kind of sensor, many applications have been developed on top of monocular SLAM systems.

One of the main limits of monocular SLAM systems is that, because of the projective nature of the sensor, the scale of the 3D scene is not observable. This has two important implications: (1) The scale of the camera trajectory and of the reconstructed map are arbitrary, depending typically on choices made during the system initialization; (2) While no loop closure process is applied on the map and on the trajectory (usually with some form of bundle adjustment), the scale error may drift without bound. For example, in Fig. 1, top, the basic version of ORB-SLAM (without loop closure) outputs the green path on one of the KITTI dataset urban video sequences. The ground truth appears in red. The scale drift explains why the internal scale estimate is clearly increasing during the whole experiment. When loop closure processes are applied, the global scale is made coherent over the map and the trajectory, but again, at an arbitrary value. Since for many applications (mobile robotics, augmented reality,. . . ) the true scale factor plays a critical role, automatic methods to infer it are important.

1 Centro de Investigaci´ on en Matem´ aticas, Guanajuato, M´ exico { edgar.sucar,jbhayet } @cimat.mx The main idea of this work is that, based on the semantic content of what a monocular system can perceive, and even if each perceived cue gives uncertain evidence on scale, a robot should be able to infer the global scale of the structures present in the scene. Handling the potential contradictions between cues can be done efficiently within a Bayesian framework, as it allows to specify and fuse nicely the uncertain knowledge given by each visual cue. Evidence of this inference process in animal visual systems has been exhibited [20]. As an example, when it observes a scene containing cars and houses, the human brain, based on its prior knowledge on the size of typical cars and houses, can infer depths and distances, even though there is a slight possibility that all of these objects are just small objects in a toy world. Based on this idea of using general semantic knowledge (e.g., detections of cars at the bottom of Fig. 1), we build up a Bayesian inference system on the monocular SLAM scale correction. This system allows us to produce, in the case of Fig. 1, the blue path, much closer to the ground truth than the green one (without correction).

Fig. 1. We estimate the scale correction to apply to the camera trajectory and the map (top and middle figures), using Bayesian inference based on a detector of instances of pre-defined object classes (e.g. cars, bottom figure), with prior distributions specified on the height of theses objects. The top figure shows, for KITTI sequence 00, the reconstructed trajectory (without loop closure) by ORB-SLAM, in green, while our corrected trajectory is depicted in blue, and the ground truth in red.

<!-- image -->

We review related work in Section II, then give an overall explanation of our algorithm in Section III, and give specific details on the probability distributions that we use in Section IV. Quantitative and qualitative evaluations are given in Section V.

## II. RELATED WORK

Monocular SLAM has been a tool of choice in 3D scene and camera trajectory reconstruction, e.g. in mobile robotics or augmented reality, in particular because monocular systems are widespread and inexpensive. Two categories of online techniques coexist in the literature for this purpose: the ones that use Bayesian filtering [3] and the ones that extend the traditional bundle adjustment algorithms to online systems [9], [12]. The latter have allowed to attain outstanding results in the recent years, at much larger scales than the former. However, common limitations of all the existing methods are that: (i) the scale of the reconstruction is unknown by essence, and (ii) the consecutive reconstructions/pose estimations may introduce scale drift which makes the global maps or the complete trajectories inconsistent. Most of the classical SLAM systems [3], [11] use ad-hoc elements in the initialization phase to set the reconstruction scale at the beginning, i.e., known objects or known motions. To limit the scale drift, loop closure techniques allow to reset the scale in a consistent way with its initial value [11].

An obvious solution to the scale recovery problem is to upgrade the sensors to devices capable of measuring depth (e.g., Kinect [13]) or displacements (e.g., IMU sensors [14]), but this may be costly or simply unfeasible. In this work, we focus on using only the semantic content of RGB images, together with prior uncertain information on this content, to infer the global scale of the reconstruction. Previous works in this direction include [6], where the output of an object recognition system is used in the map/trajectory optimization, and [16], where object detection was used to simplify the map building process with depth cameras. In both cases, databases of specific instances of objects were used, whereas our work uses more general object classes.

Closer to our approach, [10] proposes a scale estimation method that tracks the user face and uses it as a cue for determining the scale. The method is designed for cell phones equipped with front and rear cameras and would be difficult to extend to more generic monocular systems. In [18], [21], and [8] in a context of monocular vision embedded on cars, the scale is integrated based on the knowledge of the camera height above the ground, and based on local planarity assumptions in the observed scene. Again, our method can be applied in more generic settings, although we evaluate here in this road navigation context.

In [5], the approach is similar to ours as it also uses size priors for the detections and as it is applied to urban scenes. Nevertheless, we do not rely on consecutive object detections (which implies data association to be solved) and instead get observations from any object detection on which projections of the reconstructed 3D cloud points lie. Additionally, by relying only on points projected from the 3D map, we ensure in some way not to include information coming from dynamic objects. We use a Bayesian formulation that allows us to integrate different elements of previous knowledge, such as a prior on the variations of the scale correction factor.

Finally, our work is also reminiscent of approaches that perform machine learning-based depth inference from the texture of monocular images [17]. Here we combine the strength of recent deep learning detection techniques [15] with the power and flexibility of Bayesian inference, so as to integrate available prior knowledge in a principled way.

In a first version of this work [19], we adopted a similar strategy to the one presented here, but this paper introduces three novel contributions:

- the estimated scale correction parameter is now associated to a motion model, and its variations are related to the SLAM system scale drift (see Section IV-A),
- a new, more robust, probabilistic observation model is proposed (see Section IV-B).
- it is now implemented in a state of the art Monocular SLAM system, which allows for improved evaluation.

## III. DETECTION-BASED SCALE ESTIMATION

In this section, we present the core elements of our detection-based scale correction system.

## A. Notations and definitions

From now on, we assume that we run a monocular SLAM algorithm (such as [11]). We denote the camera calibration matrix by K . The camera pose at time k is referred to as T k ∈ SE ( 3 ) . 3 D points, reconstructed by the SLAM algorithm, are indexed by j and referred to as p j w in the world frame and as p j c = T k p j w in the camera frame. Points projected on the image frame at time k are noted as p j k = p ( K , p j c ) where p is the standard perspective projection function.

As explained hereafter, we rely on a generic object detector that, given an image in our video sequence, outputs a list of detected objects, together with the class they belong to. This detector (see Section V-B) has been trained to detect instances from dozens of classes. In frame k , we denote the set of object detections as D k , and the set of sets of detections done at frames 1 , .., k as D 1: k . Each individual detection is noted as D i k ∈ D k . We define two functions R () and c () , such that R ( D i k ) is the rectangular region in the image corresponding to the detection, like the rectangles depicted in Fig. 1, and c ( D i k ) is the object class of the detection (e.g., 'truck', 'car', 'bottle'. . . ). Finally, we introduce a prior height distribution built beforehand for an object class c ( D i k ) as p c ( D i ) ( H ) defined on R + .

k A system such as ORB-SLAM [11] maintains a local map N k of points expressed in the world frame. It is a subset of the global map that contains the set of points from keyframes K 1 that share map points with the current frame, and the set K 2 with neighbors in K 1 in the covisibility graph. The local map is used for tracking purposes and it is optimized via bundle adjustment every time a new keyframe is added. Most other SLAM systems work in a similar way. Our aim in this work is to estimate the scalar k k , which we define as the correction to apply to the local 3D map or to the local trajectory in order to obtain the correct scale of the local map maintained by a visual SLAM system, at time k .

## B. Problem statement

We model k k as a random variable to be estimated at time k such that for any pair of points p i w , p j w ∈ N k , the true metric distance Dr between them is given by Dr ( p i w , p j w ) = k kD ( p i w , p j w ) + n , where D is the distance measure in the current reconstruction and n is a reconstruction error noise.

Since the local map is used for tracking, we can recover the camera trajectory with its correct metric scale by estimating k k . Given T k , the pose of the camera at time k according to the visual SLAM system, the pose ˜ T k with its correct metric scale can then be computed incrementally by

<!-- formula-not-decoded -->

where s ( T , a ) builds a similarity from the rigid transformation T and the scale factor a .

In the following, we develop a Bayesian formulation for the estimation of this local scale correction as the mode of the posterior distribution:

<!-- formula-not-decoded -->

i.e., conditioned to the observation of detected objects D 1: k and to the SLAM local reconstructions N 1: k .

## C. Bayesian framework for estimating the scale correction

As mentioned above, we stress that, because of the scale drift inherent to monocular SLAM systems, the global scale correction k is varying with time. To estimate it, we use observations from object detections on which we have priors for their belonging classes (e.g., priors on cars heights in Fig. 1), and we use a dynamical model to cope with potential variations in the internal scale of the SLAM algorithm, i.e., a rough model on the dynamics of scale drift. As we do not have a detailed knowledge of these variations (it probably depends on the internal logic of each SLAM algorithm), we use a simple dynamic model from frame k to frame k + 1

<!-- formula-not-decoded -->

where n ∼ N ( 0 , s p k ) . We will explain how to select s p k in Section IV-A.

In frame k , let us suppose that we are capable of getting a set of object detections D k . By applying Bayes formula,

<!-- formula-not-decoded -->

For the sake of clarity in this derivation, let us first suppose that D k consists of a single detection D 1 k of an object belonging to class c ( D 1 k ) .

<!-- formula-not-decoded -->

Through the formula above, we obtain a recursive Bayes filter that allows to make updates of the scale correction estimate at each new frame, by incorporating three terms: (1) a transition probability p ( k k | k k -1 ) that models the scale drift in the SLAM algorithm; (2) a likelihood term p ( D 1 k | H , k k , N k ) that evaluates the probability of having the observed detection, given a current point cloud N k built by the SLAM algorithm, given a possible height H for the detected object of class c ( D 1 k ) , and given a global scale correction k k ; (3) a prior on heights p c ( D 1 k ) ( H ) , specific to the class of the detected object c ( D 1 k ) .

This means that, at each step, we can update the posterior on k k . We implemented the previous inference scheme in two ways: as a discrete Bayes filter and as a Kalman filter. Using one or the other depends mainly on the context and on the nature of the involved distributions. In the first case, we use a histogram representation for the posterior distribution and for p c ( D 1 k ) ( H ) , the prior probability the object height. By discretizing the possible heights Hm over a predefined interval, we can compute the likelihood term as m p ( D 1 k | Hm , d , N k ) p c ( D 1 k ) ( Hm ) . In the second case, when the involved distributions are Gaussian and the models linear, then we have an instance of the Kalman filter, which takes a simple form of mean/variance updates (see Section IV-D).

Note that in the more general case of | D k | &gt; 1, and by assuming conditional independence between the different detections observed in frame k , we have

<!-- formula-not-decoded -->

In the following, we give details on these three distributions.

## IV. DEFINITION OF THE PROBABILISTIC MODELS

## A. Transition probability

As stated before, the distribution p ( k k | k k -1 ) allows us to encode time variations of the global scale correction. These variations are caused by accumulation of errors in the mapping and tracking threads of the SLAM algorithm.

Experiments show that larger global scale variations occur in situations when the camera experiences greater angular displacement [12]. For this reason, p ( k k | k k -1 ) is modeled as a Gaussian distribution centered at k k -1 with a standard deviation s p k , variable for each frame k , and proportional to the angular displacement of the camera.

Let w k be the angular displacement (in degrees) along the rotation axis between T k -1 and T k . Let k 0 be the last time since the scale was updated, then we define W k = k i = k 0 w i .

The standard deviation s p k is then calculated as

<!-- formula-not-decoded -->

.

The values observed to work in practice are s min = 0 . 00001, s max = 0 . 05, and W max = 120 ◦ . These values for s min and s max have been determined from the observed variations of the scale correction along several test sequences.

## B. Likelihood of detections

The term p ( D 1 k | H , k k , N k ) is the probability that the detected object has the dimensions in pixels with which it was detected, given that the object has a real size H , that the scale is k k , and that the local map is N k .

The general idea to evaluate it is to estimate the height of the detected object using R ( D 1 k ) and N k , then to obtain a scale correction estimate and compare it with k k .

Let { p 1 c , . . . , p m c } be the points from the local map N k transformed in the camera frame and whose projection lies inside R ( D 1 k ) , with the current pose and map parameters. We assume that in the world frame in which the SLAM system does its tracking and mapping, we can identify the vertical direction. We assume that the detected object surface is parallel to the vertical direction and that the object is oriented vertically. From { p 1 c , . . . , p m c } , we will first construct a point p s that will lie on a vertical straight line L to be used to infer the object height. Let { ˆ p 1 c , . . . , ˆ p m c } be the projection of the points { p 1 c , . . . , p m c } in the plane perpendicular to the vertical direction and that pass through the camera position.

We assume that the points { ˆ p 1 c , . . . , ˆ p m c } are sorted in increasing order according to their distance to the camera position, given by the SLAM system. The point p s c is obtained as a weighted average of { ˆ p 1 c , . . . , ˆ p m c } , giving higher weight to points closer to the camera except for a small portion of the closest points. This is done in order to filter out points that do not lie on the surface of the object, in particular points from the background inside the detection region, or points appearing due to partial occlusions. This can be observed on Figure 2, left, with the points lying on the object surface in green, the points closer to the camera (which can lie, for example, on the ground) in yellow, and the points further away to the camera (e.g., on a building behind the car) in blue. Finally, p s c is depicted in red.

The averaging of the points { ˆ p 1 c , . . . , ˆ p m c } is done with a gamma density g on the index position, with parameters a = 1 . 5, b = 0 . 2, which were determined to work well on practice. Hence, we can estimate p s c as

<!-- formula-not-decoded -->

The 3D line L is defined as the line passing through p s c with vertical direction (see Fig. 2, right). Let p s k be the projection of p s c on the image with the current camera parameters and l a line in the image passing through p s k and such that the plane obtained by back projecting l is vertical.

We consider the intersections of this line with the boundary of the detection R ( D 1 k ) , p t , p b = l ∩ ¶ R ( D 1 k ) , as depicted in Figure 2 with green dots on the image plane, while p s k is the red dot. These two image points are taken as the vertical extremities of the object.

Let r t and r b be the 3D map rays obtained by back projecting the image points p t and p b , respectively. We define ˜ p t = r t ∩ L and ˜ p b = r b ∩ L . These two points are taken as the vertical extremities of the object in the 3D map, as seen in Figure 2 (in green).

Then the object height can be estimated as the Euclidean distance ˆ H = D ( ˜ p t , ˜ p b ) . The scale correction observation, given H , is calculated as ˆ k k = H ˆ H . Finally, the likelihood of the detection D 1 k is evaluated as

<!-- formula-not-decoded -->

with f ( ; m , s ) a Gaussian density with mean m and standard deviation s . The next section describes how s m k can be evaluated at k .

## C. Observation noise variances

We define s m k , the observation noise, to quantify roughly the uncertainty on each scale observation. We evaluate it as the standard deviation on ˆ k k using uncertainty propagation, as follows. Let di = ‖ ˆ p i c ‖ , with ˆ p i c one of the points in N k as defined in the previous section, expressed in the camera frame. The variance on the depth of p s c , ‖ p s c ‖ , is approximated as the variance of the distances di with the weights of Eq. 5. We refer to it as s 2 d .

The distance ˆ H = D ( ˜ p t , ˜ p b ) can be expressed as ‖ p s c ‖| m 2 ± m 1 | with m 2 = D ( ˜ p t , p s c ) ‖ p s c ‖ and m 1 = D ( ˜ p b , p s c ) ‖ p s c ‖ ( m 2 and m 1 are considered as constant, here, as they are the slopes of rt and rd and do not depend on ‖ p s c ‖ ). Hence, the standard deviation on D ( ˜ p t , p s c ) is roughly s D = | m 2 ± m 1 | s d = D ( ˜ p t , ˜ p b ) ‖ p s c ‖ s d . Now ˆ k k = H D ( ˜ p t , ˜ p b ) so s m k can be approximated as

<!-- formula-not-decoded -->

Fig. 2. Left: top view of a 3D object corresponding to a car detection. The dots correspond to 3D points that project inside the detection region. The green dots lie on the surface of the object, the yellow dots are closer to the camera than the object's surface (they could correspond to an occluded part of the car), and the blue dots are further away to the camera than the object's surface (they correspond to points in the background of the detection region). A representative of these points is obtained, the red dot, which is more likely to lie on the surface of the object; all the points are averaged with a gamma distribution evaluated at their depth ranking value. Right: projection of the object on the image. The red dot p s corresponds to a 3 D point on the surface of the car, p s is the projection of this point on the image. p t and p b are the vertical extremities of the object on the image and p t and p b correspond to the extremities in the 3 D world frame. L is a line parallel to the vertical direction passing through p s and l is the projection of this line in the image. r t and r b correspond to the back projection of p t and p b , respectively.

<!-- image -->

## D. Posterior updates

In the case the scale correction and height prior distributions are represented as discrete distributions, the implementation of Eq. 3 is quite straightforward.

In the case the height prior distributions are Gaussian f ( H ; ¯ H , s H c ( D 1 k ) ) , the Bayes Filter can be implemented as a Kalman Filter, where the current scale correction estimate is represented by its mean/variance before and after correction, k -k , k + k (means) and s -2 k , s + 2 k (variances) with the following equations:

1) Prediction: It results from Eq. 1,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

2) Correction: In this step, the difference with the traditional Kalman filter is that we have to marginalize the variable H over the prior on the object class, i.e. compute ∫ H p ( D l k | H , k k , N k ) p c ( D l k ) ( H ) dH . To simplify the evaluation and keep the result as a Gaussian, we use Eq. 6 for a fixed value of H , ¯ H , i.e., s m k = ¯ H ˆ H s d ‖ p s c ‖ .

In that case, we deduce that ∫ H p ( D l k | H , k k , N k ) p c ( D l k ) ( H ) dH is a Gaussian centered at ¯ H ˆ H with variance 1 ˆ H 2 ( s 2 H + s 2 d ¯ H 2 ‖ p s c ‖ 2 ) . The update expressions follow for the means and variances:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## V. EXPERIMENTAL RESULTS

## A. Description of the experimental setup

For a quantitative evaluation of the scale estimation for correcting scale drift, the algorithm is run on 10 sequences of the KITTI dataset [7]. Each sequence consists of a driving scenario in an urban environment with varying speeds and distances. We want to stress that, although the application presented in these experiments is quite specific (monocular vision for road vehicles), the proposed method is much more generic and can be used in many other scenarios. We have chosen this application to measure its potential benefits, because of the existence of well documented datasets, such as KITTI. Sequences 00 to 10 are considered here, except for sequence 01, since it is in a highway in which the SLAM algorithm (ORB-SLAM) fails due to the high speed. The sequences come with ground truth poses for evaluation. The evaluation computes errors between relative transformations for every subsequence of length ( 100 , 200 , ..., 800 ) meters as proposed in [7]. Here, as our algorithm evaluates the scale correction, we only present results on translational errors. The rotational errors are a consequence of the SLAM algorithm and do not depend on the scale.

## B. Implementation

The monocular version of ORB-SLAM 2 [12] is used for tracking and mapping. Loop closure is disabled so that the scale drift is directly observed (as in Fig. 1).

YOLO9000 [15] is used for detecting car instances, and the minimum confidence threshold is set to 0 . 45. We could have considered more object classes but their presence in the KITTI dataset is marginal (a few 'truck' or 'bus' objects only). Object detection is run every 5 frames. As it can be seen in Table I, the number of updates, i.e. of integrations of observations in the Bayesian framework, is quite variable. In sequence 00 or 07, there are approximately 0 . 2 updates per frame; in sequence 04, this number falls to 0 . 014. Of course, this has an impact on the final errors (see below).

Fig. 3. (a) Evolution of the scale MAP estimate (in bold) along scales observations (the graphs correspond to KITTI sequence 00). (b) Evolution of the scale correction posterior. The color palette indicates the point in time (the clearer, the later in the video). (c) Likelihoods of new observations along time. We observe likelihoods with different dispersions since the dispersion is calculated from the variance on the object's surface depth.

<!-- image -->

The prior distribution for the car's height is set as a Gaussian with mean 1 . 5 meters. The mean was chosen in accordance with the report by the International Council on Clean Transportation [1] for average car height in 2015. Based on these facts, we selected the Kalman Filter implementation of the algorithm, equations 7 and 8 for prediction and equations 9 and 10 for correction.

The ORB-SLAM and YOLO algorithms run in real time, and the Kalman filter implementation of the scale correction estimation adds negligible additional processing time, which guarantees the real time performance of the algorithm.

TABLE I SCALE UPDATE STATISTICS FOR 10 SEQUENCES OF THE KITTI DATASET.

|   Sequence |   Frames |   Updates |   Avg. updates per frame |
|------------|----------|-----------|--------------------------|
|         00 |     4540 |       921 |                    0.202 |
|         02 |     4660 |       289 |                    0.062 |
|         03 |      800 |        24 |                     0.03 |
|         04 |      270 |         4 |                    0.014 |
|         05 |     2760 |       202 |                    0.073 |
|         06 |     1100 |        75 |                    0.068 |
|         07 |     1100 |       234 |                    0.212 |
|         08 |     4070 |       530 |                    0.130 |
|         09 |     1590 |       111 |                    0.069 |
|         10 |     1200 |        40 |                    0.033 |

TABLE II COMPARISON OF RELATIVE TRANSLATIONAL ERROR.

| Seq.   |   ORB-SLAM Bayesian (ours) Trans(%) |   ORB-SLAM Update only Trans(%) |   ORB-SLAM Avg. Scale Trans(%) |   ORB-SLAM Stereo [12] Trans(%) |   [18] Trans(%) |
|--------|-------------------------------------|---------------------------------|--------------------------------|---------------------------------|-----------------|
| 00     |                                3.09 |                           20.77 |                          35.78 |                            0.70 |            7.14 |
| 02     |                                6.18 |                            4.81 |                           8.01 |                            0.76 |            4.34 |
| 03     |                                3.39 |                            2.82 |                          15.07 |                            0.71 |            2.90 |
| 04     |                               32.90 |                           26.14 |                          26.19 |                            0.48 |            2.45 |
| 05     |                                4.47 |                           17.54 |                          51.69 |                            0.40 |            8.13 |
| 06     |                               12.46 |                           16.22 |                         20.883 |                            0.51 |            7.56 |
| 07     |                                2.81 |                           11.66 |                          10.76 |                            0.50 |            9.92 |
| 08     |                                4.11 |                           20.32 |                          31.88 |                            1.05 |            7.29 |
| 09     |                               11.24 |                           12.69 |                          22.65 |                            0.87 |            5.14 |
| 10     |                               16.75 |                           16.11 |                          18.65 |                            0.60 |            4.99 |
| Avg.   |                                5.52 |                           15.30 |                          27.89 |                          0.0893 |            6.42 |

## C. Evaluation and discussion

We can see in Figure 3(a) the evolution of the scale estimate (in bold) along with the scale observations corresponding to the KITTI sequence 00, i.e. the same as the illustration of Fig. 1. Our scale correction estimate is clearly decreasing from values &gt; 1, i.e. as ORB-SLAM is subestimating the scale, to values &lt; 1 towards the end of the sequence, i.e. as ORB-SLAM is over-estimating the scale. This effect is perceptible in Fig. 1, through the path estimated by ORB-SLAM: distances in the trajectory produced by ORB-SLAM without scale estimation (in green) are seen to be overestimated later in time.

Figure 3(b) shows the evolution in time of the scale posterior. The peaks correspond to updates with a low uncertainty, as can be seen in the plot of all update distributions in Figure 3(c). The time is indicated by the color of the posterior (lighter colors means later times). Updates can also have a higher effect on the posterior in moments where a large scale drift is expected due to high rotational translation, as suggested by equation 4.

In Table II, we compare the errors obtained with different approaches for scale estimation for the 10 KITTI sequences analyzed: (i) in the second column, our method as presented in this paper; (ii) in the third column, our method without the scale correction motion model, i.e. roughly as in [19]; (iii) in the fourth column, a very simple method that computes an average value of the scale correction, 1 N N i = 1 ˆ k k , and applies it to the map and the trajectory (this corresponds to neglecting the scale drift effect); (iv) in the fifth column, to give a hint on the precision reached by a 3D sensor, we give the results by ORB-SLAM 2 with the stereo datasets; finally, (v) the fifth column gives results from the monocular system developed in [18], where the camera height over the road is known.

Fig. 4. Reconstructed trajectories for sequences 05 (left) and 08 (right). The experiments compare the ground truth (in red), the output of ORB-SLAM (in green), our scale correction algorithm with motion model (in blue) and without it (in orange).

<!-- image -->

Fig. 5. Evolution of the average translational errors with/without scale correction, and with/without motion model for the scale correction factor, evaluated on the 10 KITTI sequences.

<!-- image -->

Analyzing the results for our methods (second and third columns) in the different sequences, one can see a strong correlation between the obtained errors and the average number of updates per frame as described in Table I, as expected. For example, in sequences 00, 07, 08, where a lot of cars where detected, the results are very good, with errors significantly lower than [18]. On the opposite, sequences 04, 09, 10 with their scarce car detections, give rather poor results. Sequence 04, for instance, is a short sequence in a highway, without static vehicles, and produces only 4 update steps. However, (bottom row), the overall error levels are lower than [18]. Note that introducing the motion model with varying variance has allowed to improve the performance of [19] by a factor of 3. Last, as expected, not including the scale drift (fourth column) leads to very poor performance. Finally, a detector such as [15] is quite versatile, so we could use it at its maximum potential by integrating other classes to detect, e.g. road signs, house doors and windows.

In Fig. 4, we give two more examples of reconstructed trajectories with/without our scale correction and with/without motion model for the scale correction factor. Our method allows the final trajectory (in blue) to get very close to the ground truth (in red). Similarly, in Fig. 5, we give the errors of these same methodologies, for different path lengths, and averaged over the 10 sequences. Again, our method allows to get very reasonable errors, between 4% and 7%.

Some of the best monocular systems with scale correction, [8] and [21], have average errors of 5% and 3%, respectively, which are very similar to the average error of our method, 5 . 53%. But these monocular methods are specific to driving scenarios, based on a given fixed camera height and an observable plane. On the other hand, state of the art methods for scale estimation based on object detection, [5], have errors of 20% in average. Our method outperforms state of the art methods of scale estimation based on object detection while achieving similar performance to state of the art monocular systems with scale correction, but within a more general framework.

## VI. CONCLUSIONS

We have presented a Bayes filter algorithm that allows to estimate the scale correction to apply to the output of a monocular SLAM algorithm so as to obtain correct maps and trajectories. The observation model uses object detections given by a generic object detector, and integrates height priors over the object from the detected classes. A probabilistic motion model is proposed in order to model the scale drift. In the light of the very promising results obtained in the KITTI dataset, we will put our efforts in obtaining a better model for the scale drift, whose evolution over time seems to exhibit a clear structure.

## REFERENCES

- [1] European vehicle market statistics pocketbook 2015/2016. Technical report, The International Council on Clean Transportation, 2015.
- [2] D. S. C. Forster, M. Pizzoli. Svo: Fast semi-direct monocular visual odometry. In Proc. of IEEE Int. Conf. on Robotics and Automation (ICRA) , 2014.
- [3] A. J. Davison. Real-Time Simultaneous Localisation and Mapping with a Single Camera. In Int. Conf. Comput. Vis. , 2003.
- [4] J. Engel and D. Cremers. Lsd-slam: Large-scale direct monocular slam. In In Proc. of European Conference on Computer Vision (ECCV) , 2014.
- [5] D. P. Frost, O. K¨ ahler, and D. W. Murray. Object-aware bundle adjustment for correcting monocular scale drift. In Proc. of Int. Conf. on Robotics and Automation , pages 4770-4776, May 2016.
- [6] D. G´ alvez-L´ opez, M. Salas, J. D. Tard´ os, and J. Montiel. Real-time monocular object slam. Robotics and Autonomous Systems , 75, Part B:435 - 449, 2016.
- [7] A. Geiger, P. Lenz, and R. Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In Conference on Computer Vision and Pattern Recognition (CVPR) , 2012.
- [8] J. Grter, T. Schwarze, and M. Lauer. Robust scale estimation for monocular visual odometry using structure from motion and vanishing points. In 2015 IEEE Intelligent Vehicles Symposium (IV) , pages 475480, June 2015.
- [9] G. Klein and D. Murray. Parallel Tracking and Mapping for Small AR Workspaces. In Proc. of Int. Symp. Mix. Augment. Real. IEEE, Nov. 2007.
- [10] S. B. Knorr and D. Kurz. Leveraging the user's face for absolute scale estimation in handheld monocular slam. Proc. of Int. Symp. on Mixed and Augmented Reality , 00:11-17, 2016.
- [11] R. Mur-Artal, J. Montiel, and J. Tard´ os. ORB-SLAM: a versatile and accurate monocular SLAM system. IEEE Transactions on Robotics , 31(5):1147-1163, 2015.
- [12] R. Mur-Artal and J. D. Tard´ os. ORB-SLAM2: an open-source SLAM system for monocular, stereo and RGB-D cameras. arXiv preprint arXiv:1610.06475 , 2016.
- [13] R. A. Newcombe, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohli, J. Shotton, S. Hodges, and A. a. Fitzgibbon. Kinectfusion: Real-time dense surface mapping and tracking. In Proc. of Int. Symp. on Mixed and Augmented Reality . IEEE, October 2011.
- [14] G. N¨ utzi, S. Weiss, D. Scaramuzza, and R. Siegwart. Fusion of imu and vision for absolute scale estimation in monocular slam. In IEEE/RSJ Int. Conf. on Intelligent Robots and Systems , 2011.
- [15] J. Redmon and A. Farhadi. Yolo9000: Better, faster, stronger. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , July 2017.
- [16] R. Salas-Moreno, R. Newcombe, H. Strasdat, P. Kelly, and A. Davison. Slam++: Simultaneous localisation and mapping at the level of objects. In Proc. of Int. Conf. on Computer Vision and Pattern Recognition , 2013.
- [17] A. Saxena, M. Sun, and A. Y. Ng. Make3D: Learning 3D Scene Structure from a Single Still Image. IEEE Trans. Pattern Anal. Mach. Intell. , 31(5), May 2009.
- [18] S. Song, M. Chandraker, and C. C. Guest. Parallel, real-time monocular visual odometry. In Proc. of IEEE Int. Conf. on Robotics and Automation , pages 4698-4705, May 2013.
- [19] E. Sucar and J.-B. Hayet. Probabilistic global scale estimation for monoslam based on generic object detection. In Computer Vision and Pattern Recognition Workshops (CVPRW) , 2017.
- [20] J. M. Wolfe, K. R. Kluender, D. M. Levi, L. M. Bartoshuk, R. S. Herz, R. Klatzky, S. J. Lederman, and D. M. Merfeld. Sensation and Perception . Sinauer associates, 2014.
- [21] D. Zhou, Y. Dai, and H. Li. Reliable scale estimation and correction for monocular visual odometry. 2016 IEEE Intelligent Vehicles Symposium (IV) , pages 490-495, 2016.