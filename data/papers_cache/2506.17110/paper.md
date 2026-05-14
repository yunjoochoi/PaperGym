## Monocular One-Shot Metric-Depth Alignment for RGB-Based Robot Grasping

Teng Guo

Baichuan Huang

Jingjin Yu

Abstract -Accurate 6D object pose estimation is a prerequisite for successfully completing robotic prehensile and nonprehensile manipulation tasks. At present, 6D pose estimation for robotic manipulation generally relies on depth sensors based on, e.g., structured light, time-of-flight, and stereo-vision, which can be expensive, produce noisy output (as compared with RGB cameras), and fail to handle transparent objects. On the other hand, state-of-the-art monocular depth estimation models (MDEMs) provide only affine-invariant depths up to an unknown scale and shift. Metric MDEMs achieve some successful zero-shot results on public datasets, but fail to generalize. We propose a novel framework, monocular one-shot metricdepth alignment , MOMA, to recover metric depth from a single RGB image, through a one-shot adaptation building on MDEM techniques. MOMA performs scale-rotation-shift alignments during camera calibration, guided by sparse ground-truth depth points, enabling accurate depth estimation without additional data collection or model retraining on the testing setup. MOMA supports fine-tuning the MDEM on transparent objects, demonstrating strong generalization capabilities. Realworld experiments on tabletop 2-finger grasping and suctionbased bin-picking applications show MOMA achieves high success rates in diverse tasks, confirming its effectiveness.

## I. INTRODUCTION

Fulfilling the future promise of robotics technology, be it doing repetitive work at warehouses, serving patients and nurses at hospitals, or helping with chores in our homes, critically depends on enabling robots to agilely and robustly manipulate a wide variety of objects. Skillful robotic manipulation, in turn, demands a sufficiently accurate understanding of the scene including the 6D poses of the target object and its surroundings [1]-[4]. Presently, pose estimation for robotic manipulation tasks, such as grasping, resorts to the use of depth sensors based on various hardware-software stacks, e.g., structured light, time-of-flight distance measurement, stereo-vision, and so on. Despite often carrying hefty price tags 1 , output from these depth sensors can be frequently noisy with wrong and missing information. Factors such as transparent objects and light interference present further challenges for depth sensors. Issues such as steeper costs and reliability concerns limit the development and application of depth sensor-based robotic manipulation, which raises the question: can we perform 6D pose estimation without resorting to constantly using depth sensors?

A simple experiment suggests this should be possible through a data-driven approach: most humans can easily grasp and manipulate objects, even with an eye closed (i.e., no stereo vision). In other words, based on monocular vision and experience, humans can accurately estimate depth information for common manipulation tasks. Indeed, prior research, including state-of-the-art transferable methods such as MiDaS [5] and LeReS [6], made concrete attempts to estimate depth information based on a single image through learning. At the same time, these methods output affineinvariant depth estimates, which are accurate up to an unknown offset and scale, making them unsuitable for physical interaction/manipulation tasks.

G. Teng, B. Huang and J. Yu are with the Department of Computer Science, Rutgers University, Piscataway, NJ, USA. Emails: { teng.guo, baichuan.huang, jingjin.yu } @rutgers.edu . This work was supported in part by NSF awards IIS-1845888, IIS-2132972, and CCF-2309866.

1 An RGB-D sensor costs anywhere between 500 to 20K+ USD.

<!-- image -->

Fig. 1: Two downstream applications over which our framework, Monocular One-shot Metric-depth Alignment (MOMA), was tested against: two-finger grasping on tabletop setting and suction-based bin-picking. Using only RGB image, MOMA enable the robot successfully pick challenging transparent objects in cluttered scences.

<!-- image -->

Toward developing general machine learning-based methods for enabling robust pose estimation for robotic manipulation, in this work, we focus on the setting where camera poses are fixed , which is the dominant lab setting for fixed robot arms and the setting in most industrial warehouse applications, e.g., bin-picking. We introduce a novel yet simple-to-implement framework, monocular oneshot metric-depth alignment (MOMA), to recover metric depth from the output of state-of-the-art monocular depth estimation models (MDEM), e.g., DAM [7]. MOMA achieves this through carefully designed scale-rotation-shift alignment during the camera calibration phase, guided by a sparse set of ground-truth depth points. Afterward, our robotic system can accurately estimate depth from single RGB images.

MOMA is a one-shot approach that eliminates the need to collect data in the target environment and retrain MDEM models. Through fine-tuning DAM on transparent object datasets, MOMA readily generalizes to these objects as well. Due to its design, MOMA is also very fast, taking only a few seconds to perform an alignment calibration and a few milliseconds during runtime to process an MDEM output. The effectiveness of MOMA is thoroughly benchmarked and validated in experiments using a UR-5e manipulator, achieving remarkable success rates in both two-finger grasping tasks on tabletop settings and suction-based bin-picking tasks. The diverse evaluations demonstrate the promise of MOMA as a low-cost, depth-sensor-free depth-estimation approach for real-world manipulation tasks.

Fig. 2: The overall operating pipeline of MOMA. From MDEM, raw estimated depth is obtained and normalized to obtain normalized depth z p . For a new camera/environment setup, a one-time alignment parameter estimation is performed to obtain Θ ˆ using z p and some ground truth depth points. Θ ˆ can then be used for calibrate MDEM output to produced aligned depth z ′ p for downstream applications. Note: to simplify the illustration, we used a single scene for both parameter estimation and alignment; in practice, scenes in applications are generally not the same as those used for parameter estimation..

<!-- image -->

Organization. Sec. II discusses related work. In Sec. III, we introduce the key alignment problem and relevant solution approaches. Then, in Sec. IV, we present our pipeline that uses MDEM and scale-rotation-shift depth alignment module. In Sec. V, we evaluate the effectiveness of our algorithms on various maps. We conclude and discuss future directions for research in Sec. VI.

## II. RELATED WORK

Depth Estimation . Depth-estimation methods can be classified into three categories based on whether they learn metric depth, relative depth, or affine-invariant depth. Metric depth estimation aims at providing accurate (Euclidean) depth estimation. Whereas existing methods [7]-[10] have achieved impressive accuracy, they are limited to function within datasets coupled with fixed camera intrinsics. This results in training datasets for metric depth methods often being small, since it is challenging to collect large datasets covering diverse scenes using a single camera. Consequently, the trained models are not transferable and generalize poorly to images in unseen scenarios, esp., when camera parameters of test images differ. A compromise is to learn relative depth [11]-[14], which only indicates whether one point is farther or closer than another. This type of depth estimate, while qualitative useful, is unsuitable for applications such as manipulation. Learning affine-invariant depth [5], [7], [10], [15] offers a trade-off between metric and relative depth. With large-scale data, affine-invariant approaches decouple the metric information during training and achieve impressive robustness and generalization capability. All existing methods require collecting new data and model retraining for a new target camera/application setup.

Focusing on monocular (metric) depth estimation , the topic of this work, deep-learning-based methods dominate, which learn depth representations from delicately annotated datasets [16], [17]. Recently, numerous new models have emerged [5], [6], [8]-[10], [15], [18] that can handle openworld images. While these models claim to have a zero-shot generalization capability in terms of metric-depth estimation, the predicted depth is insufficiently precise for tasks involving physical interaction, e.g., robotic manipulation.

Grasp Pose Estimation . Given a scene with object pose estimations, grasp pose estimation proposes poses for endeffectors to grasp target objects. Grasp pose estimation turns out to be a task well-suited for data-driven approaches [1][4], [19]. While depth sensors may struggle with transparent objects, depth-completion methods such as those in [20][22] have been proposed to obtain more accurate depth information. A recent representative work, MonoGraspNet [23], aims to generate grasp poses and regress depth using a single image. The method faces generalization challenges, as is the case for other MDEM models, to diverse scenes and different camera intrinsics/poses. To generalize to other scenarios and camera setups, one has to collect a new dataset and train for their specific settings which can be tedious and limits the application of these methods. In addition to singleview approaches, multi-view NeRF-based approaches [24][26] have been explored for object reconstruction and grasp pose prediction. These methods, requiring non-trivial online computation time, provide an alternative depth sensor-free estimation methods for metric depth.

## III. PRELIMINARIES: METRIC DEPTH RECOVERY FROM AFFINE-INVARIANT MDEM

## A. The Metric Depth Alignment Problem

Metric monocular depth estimation from a single image is inherently ill-posed as a single 2D image lacks the necessary information to uniquely determine the 3D distances and scale of objects without additional context or assumptions [15]. Some monocular depth estimators focus on evaluating relative (scale-and-shift-invariant or affine-invariant) depth, thereby enhancing generalization capabilities from a single image. Several vision foundation models have been developed for metric depth estimation, demonstrating admirable performance on existing public datasets. However, due to insufficient metric depth accuracy at the instance level, these models remain impractical for robotics applications.

Let z c ∈ R 1 × n be n sampled ground truth depth points in camera coordinate system, z p ∈ R 1 × n the corresponding predicted depth from the depth estimation model, and u ∈ R 1 × n and v ∈ R 1 × n the corresponding pixel row/column indices. The metric depth alignment problem can be formulated as the following optimization problem:

<!-- formula-not-decoded -->

with J being a suitable cost function, F the mapping function from predicted depth to ground truth depth, and Θ the parameters.

## B. Prior Alignment Methods

While not aiming for fine-granularity downstream applications, prior works have tackled the alignment problem by performing linear transformation to adjust scale and shift (translate) of depth via solving

<!-- formula-not-decoded -->

where s and t are scaling and shift scalars, respectively.

The problem described in Eq. (2) can be solved using a global least-squares fitting method [6], [27] which we denote as global scale-shift alignment (GSSA). Such methods use a single set of scale/shift scalars, which cannot accommodate spatial heterogeneity. locally weighted linear regression (LWLR) [28] proposes to use sparse ground-truth depth to compute a scale-shift recover map, which solves a weighted linear regression problem for each pixel ( i, j ) :

<!-- formula-not-decoded -->

where w k = 1 √ 2 π e -d 2 k / (2 b 2 ) , d k is the Euclidean distance of the k -th pixel to the pixel ( i, j ) , b is a predefined bandwidth of the Gaussian kernel and D ˆ p is the final aligned depth from the predicted depth of MDEM. Whereas LWLR improves over global scale/shift alignment methods, its output remains insufficiently accurate for manipulation tasks.

## IV. MONOCULAR ONE-SHOT METRIC-DEPTH ALIGNMENT

## A. Overview of MOMA

As illustrated in Fig. 2, MOMA leverages MDEMs, often trained on gigantic data sets, to estimate the depth of an object from an RGB image, which ensures broad object coverage. MOMA then compute parameters Θ ˆ (exact form of Θ to follow) in Eq. (2) through solving a non-linear optimization problem aligning a sparse set of ground truth depth information (easily obtainable for cameras with known poses with respect to the scene or with an aligned depth camera) with the corresponding points from the MDEM output. For downstream tasks, monocular RGB images are processed using MDEM and then aligned using the obtained parameters Θ ˆ . The camera pose is assumed to remain unchanged, which corresponds to the majority of fixed-arm use cases in research labs and in industrial applications. If camera poses change, quick re-calibrations can be performed.

Fig. 3: An extreme example case showcasing the capability of SSRA. (a) The RGB image. (b) The ground-truth depth. (c) The predicted depth from DAM fine-tuned on TransCG [21]; the input scenario could be treated entirely unseen by the MDEM, resulting in low-quality depth output. (d) Aligned depth using SSRA, which yields better results compared to LWLR and GSSA. (e) Aligned depth using GSSA. (f) Aligned depth using LWLR.

<!-- image -->

## B. Scale-Shift-Rotation Alignment of MDEM Output

Whereas MDEMs are not new, they only start to approach foundation model level of quality recently [7], [10]. These newer models, trained over very large data sets, are capable of producing decent depth estimates for a very diverse set of scenes. Careful examination of the zero-shot capability of these vision foundation models, however, reveals situations where haywire predictions are often made. An extreme example can be seen in Fig. 3(c), where the MDEM predicted depth completely reverses the relative depth information.

Feeding more data to MDEMs can help, but there is no guarantee hallucinations can be eliminated when a previously unknown scene is encountered. On the other hand, we observe that it does not require huge efforts to adapt a capable MDEM model to a new target camera/environment, leading us to propose a targeted depth recovery method that considers scale , shift , and rotation factors.

To calibrate the MDEM output for a target camera/environment, we first capture an RGB image I and obtain its corresponding sparse ground-truth depth in the camera coordinate system. The ground-truth depth generally only requires tens to hundreds of sample points. Let X pk = [ x pk , y pk , z pk ] be the 3D point for the k -th pixel predicted by the model, and X ck = [ x ck , y ck , z ck ] be the corresponding 3D point in the camera coordinate system. We align these two point clouds by finding s ∈ R , R ∈ SO (3) and T ∈ R 1 × 3 to minimize

<!-- formula-not-decoded -->

Assuming pinhole camera models, let c xp , c yp , and f p be the (pseudo) intrinsics for the MDEM. In other words, the depth prediction of MDEM is treated as a depth map captured by a pseudo depth sensor with the defined intrinsic. We have

<!-- formula-not-decoded -->

Since we are interested only in depth, it is unnecessary to estimate all parameters listed in Eq. (4). Instead, we focus solely on minimizing the fitted depth error. In other words, we consider only the minimizing the depth alignment error

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Solving the non-linear fitting problem in Eq. (6) yields optimized parameters Θ ˆ (which is a subset of all parameters from Eq. (4)). We denote this method as scale-shift-rotation alignment or SSRA, which is a specific realization of MOMA. Alignment using only scale and shift can be seen as a special case where θ = ϕ = 0 .

## C. One-Shot Calibration and Depth Normalization

To estimate Θ ˆ , it is necessary to have some form of ground truth depth information. Here, we assume that a portion of the scene will remain the same for a fixed camera/environment setup, especially in robotic manipulation tasks. For example, in a tabletop setting, part of the table surface will remain visible across different scenes. Therefore, if we sample a sparse, diverse set of ground truth points, they can serve as reliable references.

However, there is a challenge. The raw predicted depth of MDEM will fluctuate when different objects are placed in the scene, even though the camera pose is fixed. Consequently, MDEM predicted depth at the same location will change. This renders one-shot calibration results invalid. An example given in in Fig. 4. We normalize the predicted depth z p to minimize the issue's impact. We examined two normalization methods, with the first being [27]:

Fig. 4: An example showing MDEM prediction fluctuations. The top and bottom RGB images on the left have the same fixed camera/background. However, the MDEM depth outputs (shown on the right) vary drastically.

<!-- image -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The second is min-max normalization [29]:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Experiments show the second approach leads to better results. After obtaining the normalized depth z ˆ p , we substitute it into Eq. (6) to compute the parameters Θ ˆ . When performing inference, the predicted depth output by MDEM will be normalized similarly, and the final depth is computed using F ( · ) from Eq. (6).

## V. PERFORMANCE EVALUATION

We evaluate the proposed method using standard depth estimation metrics. All metrics are calculated within object areas defined by object masks unless specified otherwise. The primary MDEM used in our evaluation is DAM [7] ViT-L model. The metrics include:

- RMSE : The root mean squared error (in meters) between the depth estimates and the ground-truth depths.
- REL : The mean absolute relative difference between the estimated and ground-truth depths.
- MAE : The mean absolute error (in meters) between the depth estimates and the ground-truth depths.
- Threshold δ : The percentage of pixels for which the predicted depths satisfy max (︂ d d p , d p d )︂ &lt; δ , where d and d p represent corresponding pixels from the depth

Fig. 5: [top] Four diverse camera poses used in our evaluation. [bottom] Corresponding sample scenes taken at the four poses.

<!-- image -->

maps D and D p , respectively. The thresholds δ are commonly set to 1.05, 1.10, and 1.25.

The pre-trained DAM underperforms on transparent objects; we fine-tuned it on transparent object datasets, combining around 500K images from [20], [21], [30], [31]. The training was performed with a learning rate of 5 × 10 -4 , a weight decay of 0.01, and a LoRA [32] rank of 256 over 20 epochs. The scale-invariant loss from [33] was utilized for training. For GSSA, b = 100 is used in the evaluation. On a PC equipped with Nvidia RTX 3090, DAM takes about 0.6 seconds to process an input monocular image. Performing alignment at run time (processing DAM output to obtain metric depth) takes less than 10 milliseconds. Running time for the one-shot alignment parameter calibration will be presented in Sec. V-B. The code and tested dataset can be found in https://github.com/GreatenAnoymous/MonoDPT grasp.

## A. Key Performance Metrics under Multiple Camera Poses

To evaluate performance, we first use a Realsense L515 to collect a dataset for 20 opaque objects under four fixed camera poses. For each camera pose, we collect 100 RGB/D images and generate the object masks using SAM [34]. Since DAM is not trained on any images from our setup, these scenarios are considered unseen to DAM. When evaluating the methods on the dataset for a given pose, we first use an RGB/D image to compute alignment parameters Θ ˆ , using 100 samples from the depth image. The parameters Θ ˆ are used for the rest of the evaluations.

Experimental results in Tab. I show the performance of four different methods (SSRA, GSSA, LWLR, and DAM raw metric depth output without any alignment) across four different poses for opaque objects. Best performances are highlighted using bold font. An example containing output from different methods is given in Fig. 6.

SSRA, our proposed MOMA implementation, consistently outperforms other methods for most poses and metrics, showing the highest accuracy and lowest error rates. SSRA exhibits the most stable performance across poses. In particular, in all settings, SSRA procured sufficiently high δ 1 . 10 accuracy ( &gt; 0 . 85 ) with MAE &lt; 0 . 03 , which we observe as necessary for enabling downstream robotic manipulation tasks. GSSA performs well in certain poses but degrades significantly in others. LWLR shows good performance, often ranking second or third. DAM consistently performs poorly across all poses. The results suggest that SSRA is the most effective method for depth estimation of opaque objects across various poses, while DAM requires significant improvements to be competitive in this task.

TABLE I: Performance of metric depth estimation methods on opaque objects under different camera poses.

|        |      |   δ 1 . 05 ↑ δ 1 . 10 ↑ δ 1 . 25 ↑ |   δ 1 . 05 ↑ δ 1 . 10 ↑ δ 1 . 25 ↑ |   δ 1 . 05 ↑ δ 1 . 10 ↑ δ 1 . 25 ↑ |   REL ↓ |   RMSE ↓ |   MAE ↓ |
|--------|------|------------------------------------|------------------------------------|------------------------------------|---------|----------|---------|
|        | SSRA |                             0.8492 |                             0.9939 |                             1.0000 |  0.0266 |   0.0164 |  0.0134 |
| Pose 1 | GSSA |                             0.7641 |                             0.9829 |                             1.0000 |  0.0320 |   0.0195 |  0.0163 |
| Pose 1 | LWLR |                             0.6077 |                             0.9426 |                             0.9998 |  0.0423 |   0.0249 |  0.0213 |
| Pose 1 | DAM  |                             0.0577 |                             0.1803 |                             0.5572 |  0.2385 |   0.1231 |  0.1209 |
|        | SSRA |                             0.6384 |                             0.9069 |                             0.9995 |  0.0446 |   0.0255 |  0.0205 |
| Pose 2 | GSSA |                             0.1263 |                             0.3002 |                             0.8687 |  0.1269 |   0.0695 |  0.0620 |
| Pose 2 | LWLR |                             0.2441 |                             0.4201 |                             0.8249 |  0.1176 |   0.0673 |  0.0570 |
| Pose 2 | DAM  |                             0.1592 |                             0.3114 |                             0.7250 |  0.1849 |   0.0902 |  0.0846 |
|        | SSRA |                             0.7680 |                             0.9871 |                             1.0000 |  0.0329 |   0.0195 |  0.0161 |
| Pose 3 | GSSA |                             0.2759 |                             0.6760 |                             0.9994 |  0.0733 |   0.0409 |  0.0370 |
| Pose 3 | LWLR |                             0.5622 |                             0.9184 |                             1.0000 |  0.0453 |   0.0275 |  0.0232 |
| Pose 3 | DAM  |                             0.4879 |                             0.7875 |                             0.9840 |  0.0632 |   0.0356 |  0.0315 |
|        | SSRA |                             0.5766 |                             0.8877 |                             1.0000 |  0.0468 |   0.0244 |  0.0204 |
| Pose 4 | GSSA |                             0.2434 |                             0.5212 |                             0.9637 |  0.0925 |   0.0455 |  0.0400 |
| Pose 4 | LWLR |                             0.3739 |                             0.6831 |                             0.9879 |  0.0723 |   0.0368 |  0.0319 |
| Pose 4 | DAM  |                             0.0327 |                             0.0611 |                             0.2988 |  0.3342 |   0.1429 |  0.1407 |

Fig. 6: (a) The RGB image. (b) The ground-truth depth. (c) The predicted depth from DAM;(d) Aligned depth using SSRA, which yields better results compared to LWLR and GSSA. (e) Aligned depth using GSSA. (f) Aligned depth using LWLR.

<!-- image -->

## B. Ablation Study: Impact of Ground Truth Samples

To evaluate the impact of ground truth samples, for camera Pose 1, we vary the number of randomly selected ground truth depth samples ( n ) from 20 to 3000. The results, plotted in Fig. 7, indicate that good metric depth can be obtained with only tens to hundreds of samples. In particular, it is not meaningful to use more than 3 -400 ground truth samples. SSRA outperformed GSSA and LWLR in terms of metrics. Additionally, SSRA demonstrated significantly reduced runtime compared to LWLR, owing to its fewer parameters that require optimization.

<!-- image -->

n

n

Fig. 7: Impact of the number of ground truth depth points used in performing that alignment (camera Pose 1 as shown in Fig. 5).

## C. Ablation Study: Impact of Normalization

In this experiment, we examine the impact of the depthnormalization. Tab. II provides the performance of various normalization methods evaluated on the dataset for camera Pose 1. It is readily clear that the min-max normalization method achieves the highest performance across all three alignment methods. Specifically, SSRA with min-max normalization attains an MAE of 0.0134 and a δ 1 . 05 of 0.8492. In comparison, the median normalization method, while less effective than min-max normalization, outperforms the method that employs no normalization techniques.

TABLE II: Impact different normalization methods (camera Pose 1).

|         |      |   δ 1 . 05 ↑ |   δ 1 . 10 ↑ |   δ 1 . 25 ↑ |   REL ↓ |   RMSE |   ↓ MAE ↓ |
|---------|------|--------------|--------------|--------------|---------|--------|-----------|
| Min-Max | SSRA |       0.8492 |       0.9939 |       1.0000 |  0.0266 | 0.0164 |    0.0134 |
| Min-Max | GSSA |       0.7641 |       0.9829 |       1.0000 |  0.0320 | 0.0195 |    0.0163 |
| Min-Max | LWLR |       0.6077 |       0.9426 |       0.9998 |  0.0423 | 0.0249 |    0.0213 |
|         | SSRA |       0.5473 |       0.8463 |       0.9988 |  0.0507 | 0.0305 |    0.0254 |
|         | GSSA |       0.5595 |       0.8415 |       0.9973 |  0.0509 | 0.0308 |    0.0253 |
|         | LWLR |       0.4480 |       0.7247 |       0.9773 |  0.0671 | 0.0403 |    0.0330 |
|         | SSRA |       0.3658 |       0.6582 |       0.9929 |  0.0812 | 0.0443 |    0.0413 |
|         | GSSA |       0.2321 |       0.4395 |       0.8972 |  0.1193 | 0.0648 |    0.0607 |
|         | LWLR |       0.2280 |       0.4351 |       0.8805 |  0.1232 | 0.0673 |    0.0626 |
|         | DAM  |       0.0577 |       0.1803 |       0.5572 |  0.2385 | 0.1231 |    0.1209 |

On the other hand, in the absence of normalization (denoted as None ), the performance across all metrics significantly deteriorates for all methods. As discussed in Sec. IV, without normalization, the predicted depth range from MDEM experiences substantial fluctuations, rendering the calibration ineffective.

## D. Ablation Study: Impact of MDEM Models

To select the best MDEM models, in addition to DAM, we evaluated DAMV2 [18] and M3DV2 [35], which achieve impressive performance on several existing public metric depth estimation benchmarks. Whereas DAM is fine-tuned on public transparent object datasets, we directly use the pretrained checkpoints for M3DV2 and DAMV2. The results are shown in Tab. III. The method with/without '+' indicates whether the SSRA alignment module from Subsection V-A is used/not used. We observe the raw metric depth predictions from these MDEM models are far from precise enough for direct use in robotic manipulation. However, with MOMA's SSRA alignment, the mean absolute error (MAE) drops to 0 . 01 -0 . 03 , suitable for robotic manipulation. We found DAM fine-tuned on a mixed transparent object dataset achieves better performance. This is due to the datasets sharing more similarities in the depth distribution pattern from the experimental setup we used for robotic manipulation.

TABLE III: Metrics Evaluation, different MDEM models.

|         |        |   δ 1 . 05 ↑ |   δ 1 . 10 ↑ |   δ 1 . 25 ↑ |   REL ↓ |   RMSE ↓ |   MAE ↓ |
|---------|--------|--------------|--------------|--------------|---------|----------|---------|
|         | DAM    |       0.0577 |       0.1803 |       0.5572 |  0.2385 |   0.1231 |  0.1209 |
|         | DAM+   |       0.8492 |       0.9939 |       1.0000 |  0.0266 |   0.0164 |  0.0134 |
|         | DAMV2  |       0.0000 |       0.0000 |       0.0000 |  1.4410 |   0.7396 |  0.7356 |
|         | DAMV2+ |       0.8030 |       0.9695 |       0.9999 |  0.0302 |   0.0188 |  0.0156 |
|         | M3DV2  |       0.0000 |       0.0000 |       0.0022 |  0.6378 |   0.3279 |  0.3251 |
|         | M3DV2+ |       0.7876 |       0.9814 |       0.9983 |  0.0335 |   0.0198 |  0.0168 |
|         | DAM    |       0.0540 |       0.2637 |       0.8566 |  0.1619 |   0.1003 |  0.0948 |
|         | DAM+   |       0.7290 |       0.9178 |       0.9910 |  0.0395 |   0.0349 |  0.0234 |
|         | DAMV2  |       0.0002 |       0.0003 |       0.0020 |  1.1547 |   0.7181 |  0.7050 |
|         | DAMV2+ |       0.4717 |       0.7487 |       0.9602 |  0.0703 |   0.0586 |  0.0437 |
|         | M3DV2  |       0.0038 |       0.0097 |       0.0966 |  0.4532 |   0.2894 |  0.2788 |
|         | M3DV2+ |       0.2766 |       0.4600 |       0.6242 |  0.1928 |   0.1307 |  0.1157 |
|         | DAM    |       0.2057 |       0.4252 |       0.8596 |  0.1288 |   0.0641 |  0.0616 |
|         | DAM+   |       0.5708 |       0.8547 |       0.9973 |  0.0494 |   0.0280 |  0.0242 |
|         | DAMV2  |       0.0000 |       0.0000 |       0.0000 |  2.3565 |   1.2428 |  1.1237 |
| In Tote | DAMV2+ |       0.4693 |       0.8684 |       1.0000 |  0.0553 |   0.0314 |  0.0267 |
| In Tote | M3DV2  |       0.0000 |       0.0000 |       0.0062 |  0.5121 |   0.2460 |  0.2445 |
| In Tote | M3DV2+ |       0.6212 |       0.8886 |       0.9996 |  0.0470 |   0.0260 |  0.0223 |

## E. Transparent Objects

We used the L515 sensor to collect a testing dataset comprising 10 transparent objects. For each object, 100 images were captured in two scenarios with fixed camera poses: one with the objects standing on a table and another with the objects lying in a tote. To obtain pseudo-groundtruth depth for the transparent objects, we replaced each transparent object with an opaque object of the same shape and position. Sample data is given in Fig. 8.

The evaluation results are summarized in Tab. IV. SSRA achieved a δ 1 . 05 score of 0.73 for standing transparent objects and 0.57 for objects in the tote. The lower performance in the tote scenario may be attributed to the limited similarity between the fine-tuning dataset and the tote environment. Nevertheless, the δ 1 . 05 score is above 0 . 85 and the mean absolute error (MAE) for the tote scenario is 0.0242, sufficient for downstream suction-based picking.

Fig. 8: [top] An example containing RGB, mask, and ground-truth depth of the transparent object dataset used in the evaluation for the two-finger grasping downstream task. [bottom] The same for the suction-based binpicking downstream task.

<!-- image -->

TABLE IV: Performance on transparent objects.

|          |      |   δ 1 . 05 ↑ δ 1 . |   10 ↑ |   δ 1 . 25 ↑ |   REL ↓ |   RMSE |   ↓ MAE ↓ |
|----------|------|--------------------|--------|--------------|---------|--------|-----------|
| Standing | SSRA |             0.7290 | 0.9178 |       0.9910 |  0.0395 | 0.0349 |    0.0234 |
| Standing | GSSA |             0.5806 | 0.7709 |       0.9904 |  0.0582 | 0.0453 |    0.0340 |
| Standing | LWLR |             0.5964 | 0.7828 |       0.9904 |  0.0561 | 0.0445 |    0.0329 |
| Standing | DAM  |             0.0540 | 0.2637 |       0.8566 |  0.1619 | 0.1003 |    0.0948 |
| In Tote  | SSRA |             0.5708 | 0.8547 |       0.9973 |  0.0494 | 0.0280 |    0.0242 |
| In Tote  | GSSA |             0.5640 | 0.8548 |       0.9973 |  0.0495 | 0.0280 |    0.0242 |
| In Tote  | LWLR |             0.4289 | 0.7866 |       0.9957 |  0.0627 | 0.0341 |    0.0306 |
| In Tote  | DAM  |             0.2057 | 0.4252 |       0.8596 |  0.1288 | 0.0641 |    0.0616 |

## F. Real Robot Experiments

As downstream applications, we conducted real robot experiments using a UR-5e robot equipped with an adaptive Robotiq 2f-85 gripper for finger-based grasping and a vacuum gripper for suction-based grasping. The transparent and opaque testing objects are depicted in Fig. 9. The adaptive gripper was used to grasp objects lying or standing on the table, while the vacuum gripper was employed for suctioning objects placed in the green tote.

For each scenario, the camera was fixed, and calibration was performed using a single RGBD image captured by the L515 sensor. After the calibration, the depth sensor was disabled for methods utilizing MDEM. We conducted 50 trials for each scenario, randomly placing 2-3 objects each time, to determine the success rate of each method. The results are presented in Tab. V. SSRA achieves fairly decent success rates of over 80% on opaque objects, beating other alignment methods, especially in the two-finger grasp scenario. This is partially due to suction-based grasping having built-in compliance, e.g., most grasps are top-down; overshooting a little will still work. As a reference, using full RGBD information acquired from the L515 sensor, all opaque objects can be grasped (either fingered or suctionbased) whereas no transparent objects can be grasped using the raw depth from the L515 sensor.

## VI. CONCLUSION AND DISCUSSIONS

Leveraging the power of vision foundational models, in this work, we propose a monocular one-shot metric-depth alignment (MOMA) framework toward enabling depthsensor-free estimation of metric object depth for executing robotic manipulation tasks. As a first attempt at developing the capability to perform robotic manipulation without costly and sometimes unreliable depth sensors, MOMA holds significant application potential, especially in industrial settings where cameras and robot arms are fixed. Our specific MOMA implementation, SSRA, running very fast at both alignment parameter calibration (few seconds) and inference time (few milliseconds, achieves significantly better accuracy in standard benchmark tests (e.g., δ 1 . 05 , δ 1 . 10 , RMSE, MAE) when compared with similar state-of-the-art methods. Moreover, in downstream robotic two-finger and suctionbased grasping tasks, SSRA consistently achieves over 80% success rates when handling non-transparent objects and over 70% success rates on transparent objects.

Fig. 9: Objects tested in the real robot experiment.

<!-- image -->

TABLE V: Success rates in downstream manipulation tasks.

|                                       | SSRA   | GSSA   | LWLR   |
|---------------------------------------|--------|--------|--------|
| Two-finger grasp: opaque objects      | 82%    | 72%    | 62%    |
| Two-finger grasp: transparent objects | 70%    | 56%    | 52%    |
| Suction grasp: opaque objects         | 86%    | 86%    | 80%    |
| Suction grasp: transparent objects    | 78%    | 76%    | 70%    |

While MOMA + SSRA shows great promise in enabling monocular and RGB only metric depth estimation, it represents an initial step in rendering such attempts fully practical. As of now, MOMA still has some catch up to do when compared with RGB-D based methods, especially when it comes to dealing with transparent objects [20], [22]. With that in mind, future research is currently underway to further boost the performance/robustness of MOMA to realize its full potential. We note that SSRA only uses depth data from a single scene for calibration; using multiple scenes (i.e., going from one shot to few shots), even without increasing n , the total number of points could improve the alignment accuracy due to the alignment data being more diverse. Without the need for expensive depth sensors or dedicated on-board stereo computation, stereo or multi-view versions of MOMA could be developed as well, which will provide more precise depth estimates.

## REFERENCES

- [1] H.-S. Fang, C. Wang, H. Fang, M. Gou, J. Liu, H. Yan, W. Liu, Y. Xie, and C. Lu, 'Anygrasp: Robust and efficient grasp perception in spatial and temporal domains,' IEEE Transactions on Robotics , 2023.
- [2] H. Cao, H.-S. Fang, W. Liu, and C. Lu, 'Suctionnet-1billion: A largescale benchmark for suction grasping,' IEEE Robotics and Automation Letters , vol. 6, no. 4, pp. 8718-8725, 2021.
- [3] M. Sundermeyer, A. Mousavian, R. Triebel, and D. Fox, 'Contactgraspnet: Efficient 6-dof grasp generation in cluttered scenes,' in 2021 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2021, pp. 13 438-13 444.
- [4] A. Mousavian, C. Eppner, and D. Fox, '6-dof graspnet: Variational grasp generation for object manipulation,' in Proceedings of the IEEE/CVF international conference on computer vision , 2019, pp. 2901-2910.
- [5] R. Birkl, D. Wofk, and M. M¨ uller, 'Midas v3. 1-a model zoo for robust monocular relative depth estimation,' arXiv preprint arXiv:2307.14460 , 2023.
- [6] W. Yin, J. Zhang, O. Wang, S. Niklaus, L. Mai, S. Chen, and C. Shen, 'Learning to recover 3d scene shape from a single image,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2021, pp. 204-213.
- [7] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, 'Depth anything: Unleashing the power of large-scale unlabeled data,' ArXiv , vol. abs/2401.10891, 2024.
- [8] S. F. Bhat, R. Birkl, D. Wofk, P. Wonka, and M. M¨ uller, 'Zoedepth: Zero-shot transfer by combining relative and metric depth,' arXiv preprint arXiv:2302.12288 , 2023.
- [9] S. F. Bhat, I. Alhashim, and P. Wonka, 'Adabins: Depth estimation using adaptive bins,' in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2021, pp. 4009-4018.
- [10] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and H. Zhao, 'Depth anything v2,' arXiv preprint arXiv:2406.09414 , 2024.
- [11] K. Xian, C. Shen, Z. Cao, H. Lu, Y. Xiao, R. Li, and Z. Luo, 'Monocular relative depth perception with web stereo data supervision,' in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , 2018, pp. 311-320.
- [12] K. Xian, J. Zhang, O. Wang, L. Mai, Z. Lin, and Z. Cao, 'Structureguided ranking loss for single image depth prediction,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2020, pp. 611-620.
- [13] W. Chen, S. Qian, D. Fan, N. Kojima, M. Hamilton, and J. Deng, 'Oasis: A large-scale dataset for single image 3d in the wild,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2020, pp. 679-688.
- [14] W. Chen, Z. Fu, D. Yang, and J. Deng, 'Single-image depth perception in the wild,' Advances in neural information processing systems , vol. 29, 2016.
- [15] W. Yin, C. Zhang, H. Chen, Z. Cai, G. Yu, K. Wang, X. Chen, and C. Shen, 'Metric3d: Towards zero-shot metric 3d prediction from a single image,' in Proceedings of the IEEE/CVF International Conference on Computer Vision , 2023, pp. 9043-9053.
- [16] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun, 'Vision meets robotics: The kitti dataset,' The International Journal of Robotics Research , vol. 32, no. 11, pp. 1231-1237, 2013.
- [17] N. Silberman, D. Hoiem, P. Kohli, and R. Fergus, 'Indoor segmentation and support inference from rgbd images,' in Computer Vision-ECCV 2012: 12th European Conference on Computer Vision, Florence, Italy, October 7-13, 2012, Proceedings, Part V 12 . Springer, 2012, pp. 746-760.
- [18] L. Yang, B. Kang, Z. Huang, X. Xu, J. Feng, and H. Zhao, 'Depth anything: Unleashing the power of large-scale unlabeled data,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2024, pp. 10 371-10 381.
- [19] H.-S. Fang, C. Wang, M. Gou, and C. Lu, 'Graspnet-1billion: A largescale benchmark for general object grasping,' in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2020, pp. 11 444-11 453.
- [20] S. Sajjan, M. Moore, M. Pan, G. Nagaraja, J. Lee, A. Zeng, and S. Song, 'Clear grasp: 3d shape estimation of transparent objects for manipulation,' in 2020 IEEE international conference on robotics and automation (ICRA) . IEEE, 2020, pp. 3634-3642.
- [21] H. Fang, H.-S. Fang, S. Xu, and C. Lu, 'Transcg: A large-scale realworld dataset for transparent object depth completion and a grasping

baseline,' IEEE Robotics and Automation Letters , vol. 7, no. 3, pp. 7383-7390, 2022.

- [22] J. Shi, A. Yong, Y. Jin, D. Li, H. Niu, Z. Jin, and H. Wang, 'Asgrasp: Generalizable transparent object reconstruction and 6-dof grasp detection from rgb-d active stereo camera,' in 2024 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2024, pp. 5441-5447.
- [23] G. Zhai, D. Huang, S.-C. Wu, H. Jung, Y. Di, F. Manhardt, F. Tombari, N. Navab, and B. Busam, 'Monograspnet: 6-dof grasping with a single rgb image,' in 2023 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2023, pp. 1708-1714.
- [24] Q. Dai, Y. Zhu, Y. Geng, C. Ruan, J. Zhang, and H. Wang, 'Graspnerf: Multiview-based 6-dof grasp detection for transparent and specular objects using generalizable nerf,' in 2023 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2023, pp. 1757-1763.
- [25] J. Kerr, L. Fu, H. Huang, Y. Avigal, M. Tancik, J. Ichnowski, A. Kanazawa, and K. Goldberg, 'Evo-nerf: Evolving nerf for sequential robot grasping of transparent objects,' in 6th annual conference on robot learning , 2022.
- [26] C. Liu, K. Shi, K. Zhou, H. Wang, J. Zhang, and H. Dong, 'Rgbgrasp: Image-based object grasping by capturing multiple views during robot arm movement with neural radiance fields,' IEEE Robotics and Automation Letters , 2024.
- [27] R. Ranftl, K. Lasinger, D. Hafner, K. Schindler, and V. Koltun, 'Towards robust monocular depth estimation: Mixing datasets for zeroshot cross-dataset transfer,' IEEE transactions on pattern analysis and machine intelligence , vol. 44, no. 3, pp. 1623-1637, 2020.
- [28] G. Xu and F. Zhao, 'Toward 3d scene reconstruction from locally scale-aligned monocular video depth,' JUSTC , vol. 54, no. 4, pp. 0402-1, 2024.
- [29] S. Garc´ ıa, J. Luengo, F. Herrera, et al. , Data preprocessing in data mining . Springer, 2015, vol. 72.
- [30] X. Chen, H. Zhang, Z. Yu, A. Opipari, and O. Chadwicke Jenkins, 'Clearpose: Large-scale transparent object dataset and benchmark,' in European conference on computer vision . Springer, 2022, pp. 381396.
- [31] L. Zhu, A. Mousavian, Y. Xiang, H. Mazhar, J. van Eenbergen, S. Debnath, and D. Fox, 'Rgb-d local implicit function for depth completion of transparent objects,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2021, pp. 4649-4658.
- [32] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, 'Lora: Low-rank adaptation of large language models,' arXiv preprint arXiv:2106.09685 , 2021.
- [33] S. F. Bhat, I. Alhashim, and P. Wonka, 'Localbins: Improving depth estimation by learning local distributions,' in European Conference on Computer Vision . Springer, 2022, pp. 480-496.
- [34] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al. , 'Segment anything,' in Proceedings of the IEEE/CVF International Conference on Computer Vision , 2023, pp. 4015-4026.
- [35] M. Hu, W. Yin, C. Zhang, Z. Cai, X. Long, H. Chen, K. Wang, G. Yu, C. Shen, and S. Shen, 'Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation,' arXiv preprint arXiv:2404.15506 , 2024.