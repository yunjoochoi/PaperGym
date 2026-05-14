## RICCARDO: Radar Hit Prediction and Convolution for Camera-Radar 3D Object Detection

Yunfei Long Abhinav Kumar Xiaoming Liu Daniel Morris Michigan State University

{ longyunf,kumarab6,liuxm,dmorris } @msu.edu

## Abstract

Radar hits reflect from points on both the boundary and internal to object outlines. This results in a complex distribution of radar hits that depends on factors including object category, size, and orientation. Current radar-camera fusion methods implicitly account for this with a black-box neural network. In this paper, we explicitly utilize a radar hit distribution model to assist fusion. First, we build a model to predict radar hit distributions conditioned on object properties obtained from a monocular detector. Second, we use the predicted distribution as a kernel to match actual measured radar points in the neighborhood of the monocular detections, generating matching scores at nearby positions. Finally, a fusion stage combines context with the kernel detector to refine the matching scores. Our method achieves the state-of-the-art radar-camera detection performance on nuScenes. Our source code is available at https://github.com/longyunf/riccardo .

## 1. Introduction

3 D object detection [5, 7] is a key component of scene understanding in autonomous vehicles (AVs). It predicts nearby objects and their attributes including 3 D location, size, orientation, and category, setting the stage for navigation tasks such as path planning. The primary sensors used for 3 Dobject detection are cameras, LiDARs [41], and radars, with the focus here on the two-sensor combination that is the least expensive and already ubiquitous on vehicles, namely cameras and radars. This paper asks how to combine camera and radar data in order to achieve the best performance improvement over a single modality detection.

Detection can be performed on camera and radar individually, with each sensor having its strengths and drawbacks. Cameras are inexpensive and capture high-resolution details and texture with state-of-the-art (SOTA) methods [12, 24, 28] achieving accurate object classification, as well as estimating size and orientation. One of the primary limitations is the depth-scale ambiguity, resulting in relatively inaccurate object depth estimation [12, 24]. In contrast, current automotive radar is another inexpensive sensor that directly measures range to target, as well as Doppler velocity, and is robust to adverse weather such as rain, snow, and darkness [26]. The drawbacks of radar are its very sparse scene sampling and lack of texture, making it challenging to perform tasks such as object categorization, orientation, and size estimation. For example, radar point clouds collected in nuScenes dataset [5] are 2 D points on radar bird's-eye view (BEV) plane without height measurements (the default height is zero). This comparison shows that the strengths of radar and camera are complementary, and indeed this paper explores how we can combine information from these two modalities to improve 3 D object detection.

Figure 1. Given a (a) monocular detection, we estimate (b) radar point distribution relative to its bounding box in BEV; then we shift the distribution and convolve it with (c) actual radar measurement in the neighborhood to compute (d) similarity scores and estimate an updated position, where the matching score is maximum. In (c) the monocular bounding box (in magenta) is misaligned with radar points; the updated position (in orange) with peak matching score shifts the box to a farther range so that relative positions of radar points match the predicted distribution (radar hits concentrated at the head of vehicle instead of in the middle).

<!-- image -->

While radar and LiDAR are both widely used depth sensors for AVs, they differ significantly in their target sampling characteristics. LiDAR points align densely with the edges of objects, and for vehicles typical form a distinct 'L' or 'I' distribution. These regular distributions are aligned with the object pose and enable precise shape and pose estimation from LiDAR scans as evidenced by top-performing LiDAR methods [42] on nuScenes achieving a mean Average Precision (mAP) of 0.702 and nuScenes detection score (NDS) of 0.736. In contrast, radars have wide beam-width with low angular resolution and often penetrate objects or reflect from their undersides. This leads to a much sparser and dispersed distribution of hits on objects. From this distribution it is much more challenging to estimate target shape, category, and pose, and the top-performing radaronly method, RadarDistill [1], achieves a far lower mAP of 0.205 and NDS of 0.437 on nuScenes.

The combination of camera with radar has potential to significantly enhance radar-only methods, with camera data providing strong results on object category, shape, and pose. Radar, with its direct range measurements, can contribute range and velocity to a combined detector. Existing radar-camera models combine camera and radar via concatenation [26], (weighted) sum [9], or attention-based operations [10, 11]. But obtaining measurable gains by fusing sparse radar and camera is difficult, with the top radar-camera method [11] having lower performance than camera-only methods [19].

To address the difficulty in combining disparate modalities, we conjecture that directly modeling the radar distribution's dependency on target properties will enable radar returns to be more effectively aligned and leveraged in object detection. As shown in Fig. 1, we introduce a model that generates a radar hit prediction (RIC) in BEV from a given monocular detection priors, i.e. , bounding box size, relative pose, and object class. By moving a distribution kernel in the neighborhood of monocular estimated center and computing the similarity between the kernel and actual radar measurements, we identify potential object centers with high similarity. Those position candidates are passed through a refinement stage for final position estimation.

In nutshell, this paper makes the following contributions:

- Builds a model to predict radar hit distributions relative to reflecting objects on BEV.
- Proposes position estimation by convolving predicted radar distribution with actual radar measurements.
- Achieves SOTA performance on the nuScenes dataset.

## 2. Related Works

Monocular 3D Detection. Monocular 3 D detection is known for its low cost and simple setup, attracting exten- sive research optimizing every component in the detection pipeline, e.g. , architectures [2, 3, 37], losses [4, 14, 31], and NMS [13]. Efforts to alleviate the intrinsic ambiguities from camera image to 3 D include incorporating estimated monocular depth [30, 32, 36], designing special convolutions [12], considering camera pose [44], and taking advantage of CAD models [20]. However, depth ambiguities still remain a bottleneck to performance [14, 25] and hence, fusion with radar is a promising strategy for enhancing depth estimation while keeping low computation cost.

Camera-Radar Fusion for Detection. Camera-radar fusion has been widely applied to different vision tasks, e.g. , depth estimation [17, 23, 33], semantic segmentation [40], target velocity estimation [22, 27], detection [21, 40], and tracking [34]. For detection, various camera-radar methods have been proposed which differ in representations [10, 26], fusion level [9, 26], space [9, 11, 43], and strategies [39]. There are different radar point representations. As 2 Dradar measures BEV XY locations without height, it is natural to represent radar points in binned BEV space [11]. To associate radar points with camera source, radar points are also modeled as pillars [26, 39] with fixed height in 3 Dspace. In addition, radar points are represented as point feature [10] and each point as a multi-dimensional vector with elements of radar locations and other properties from measurement such as radar cross-section.

Radar and image sensors can interact at either the feature-level [9] or the detection-level [26]. While intermediate image features offer a wealth of raw information, such as textures, detection-level outputs provide directly interpretable information with clear physical meanings, making them well-suited for fusion tasks. With the rapid advancement of monocular 3 D detectors, leveraging detection-level image information allows us to capitalize on their accurate estimates of object category, size, orientation, and focus on refining position estimation, particularly range estimation, where radar excels as a range sensor. Therefore, in this paper, we utilize camera information at the detection level.

The fusion is conducted in image view [21, 26] , BEV [11] or a mix of the two [9]. Image features are naturally in image view, while radar is in BEV. To combine them in one space, fusion methods either project radar points to image space [21, 26] or lift image features to 3 D space [11, 43]. Image view suffers from overlappings and occlusions while it is imprecise to transform from image view to BEV without reliable depths. In this work, we adopt BEV space for fusion as we use image source monocular detections, which are already in 3 D space.

The radar-camera association is conducted by associating the radar pillars with monocular boxes in 3 D space [26] or projecting the pillars on image to extract corresponding image features [39]. However, none of these methods explicitly leverage radar point distributions to address the mis- alignment problem in radar-camera fusion.

Figure 2. RICCARDO inference . RICCARDO leverages a monocular detector to identify objects and estimate their attributes (category, size, orientation, and approximate range) and involves three stages. Its Stage 1 then predicts the radar hit distribution (RIC) for each object. Stage 2 bins and convolves the observed accumulated radar returns with the RIC, to generate a matching score over range. A final Stage 3 fusion refines these scores to yield a precise target range estimate.

<!-- image -->

## 3. RICCARDO

Our goal is to enhance object position estimation using radar returns, surpassing the capabilities of monocular vision. The challenge lies in the sparse and non-obvious alignment of radar returns with object boundaries and features, unlike the dense and consistent LiDAR returns. To address this, we propose a method that explicitly models the statistics of radar returns on objects, taking its category, size, orientation, range, and azimuth into account. These statistics enable radar returns to improve monocular detections.

Our approach, called Radar Hit Prediction and Convolution for Camera-Radar 3D Object Detection (RICCARDO), is illustrated in Fig. 2 and involves three stages. The first stage predicts the radar distribution returns on an object based on monocular detector outputs. The second stage convolves the predicted distribution with accumulated and binned radar measurements to obtain a range-based score. The third and final stage refines the range-based score to obtain a final range estimate. We describe the details of each stage below.

## 3.1. Stage 1: Radar Hit Prediction (RIC) Model

The RIC model aims to predict radar hit position distributions on objects in BEV, conditioned on object category, size, orientation, range, and azimuth. This model leverages monocular detection data to predict radar returns as a distribution, enabling comparison with actual measured returns (Sec. 3.2). This section details the construction and learning of this predictive model. We model radar hit distributions as a probability of radar return over a set of grid cells.

Coordinate System. A key choice is the coordinate system to predict this distribution. Possibilities include objectaligned, sensor-aligned or ego-vehicle aligned systems. We choose an object-aligned system for modeling radar hit position distributions. This choice allows for more gradual changes in probabilities as a function of relative sensor location compared to ego-vehicle or sensor-aligned systems, which exhibit significant variations with changes in relative object pose. Consequently, the object-aligned system facilitates learning from limited data.

Architecture. Fig. 3 shows the overview of Stage 1 in RICCARDO. Stage 1 employs a neural network to predict the distribution of radar points in object-centric BEV coordinates, conditioned on object category, size, orientation, range, and azimuth. The output is a 2 D quantized BEV probability map centered on the object, with X and Y axes aligned to the object's length and width dimensions. The RIC pixel value represents the density of radar hits at those locations. The network architecture comprises a multilayer perceptron (MLP) model, with parallel preprocessing branches for input parameters and a main branch that fuses these features and predicts the RIC.

Ground Truth Distribution. To construct a ground truth (GT) distribution for a given target, we define a grid relative to the target's known position and accumulate radar hits over a short time interval. The normalized density of returns for a grid cell i , ¯ P i , is calculated as:

<!-- formula-not-decoded -->

where c i is the number of radar returns in grid cell i , and N is the total number of cells on the RIC map. This model only includes the points within the object boundary.

Figure 3. Stage-1 RIC Model Training. The radial ray from ego to target is plotted as dashed line for reference.

<!-- image -->

Targets may move during an accumulation period in a driving scene, and directly accumulating radar hits causes smearing. Thus, we offset each radar hit position by the object motion between the radar return and the final target position. A pair of sequential annotated locations can provide GT object velocity, v O, for calculating this. Yet, radar Doppler velocity is more accurate for the radial component; thus, we combine the Doppler velocity, v D, with the tangential component of v O. Given a unit vector perpendicular to radar ray, n T, the offset m of a radar point is:

<!-- formula-not-decoded -->

where ∆ t is the time between the radar measurement and current sweep. We apply these offsets before calculating the probabilities in Eq. (1).

Loss Function. The loss function incorporates crossentropy loss L CE between the predicted radar distribution P , and the GT distribution from the accumulated radar returns, ¯ P . In addition, we include a smoothness regularization term L S on P to encourage spatial smoothness:

<!-- formula-not-decoded -->

where i and j are pixel indices within N r × N c grid of rows and columns. Thus, the total loss for training is L CE + L S.

## 3.2. Stage 2: Convolving RIC with Radar

Stage 2 uses the predicted radar density from the RIC model to score the consistency of object positions with the measured radar hits. By binning measured radar returns, we can use a convolution of the RIC to obtain similarity scores as a function of range, and so locate potential target positions.

Figure 4. Stage 2 takes the RIC predictions and neighboring radar points. It matches predicted radar distribution with radar points in the radial direction and computes binned and range-dependent matching scores. Peak positions indicates range estimations.

<!-- image -->

Fig. 4 shows the overview of Stage 2. Both the predicted and measured radar data are resampled into a BEV space with the X-axis along radial direction (from ego to target, approximately parallel to radar rays), and the Y-axis along the tangential direction. We align the central row of the RIC map with the central row of the radar measurement map and restrict the search to be along the X-axis ( i.e. , the radial axis) near to the monocular detected center. Our motivation is that monocular detections are more accurate tangentially ( i.e. , image space) and less radially. To compute the similarity between the predicted and measured densities, we slide the RIC kernel along the radial axis and calculate by sum of dot product (convolution) between the RIC and the actual point count at each position. This results in a 1 D matching score profile along the radial dimension, indicating potential target locations in the vicinity of the monocular centers.

Specifically, given a binned radar distribution P with center ( L P , L P ) and resolution of (2 L P + 1 , 2 L P + 1) and radar binned count C with center ( L C , L C ) , i.e. , object center from monocular prediction, and size of (2 L C +1 , 2 L C + 1) , the row convolution (or cross-correlation) is

<!-- formula-not-decoded -->

where n is offset to the object center along the row.

To create the measured densities, we first accumulate multiple radar sweeps over a short interval prior to the detection. Object motions of radar points are compensated by Doppler velocity and object velocity from monocular detections similar to Eq. (2). The difference is that object velocity, v O, is obtained from the monocular detector.

## 3.3. Stage 3: Camera-Radar Candidate Selector

Matching scores from Stage 2 provide position candidates, which have high scores. The purpose of Stage 3 is to select the best range predictions among Stage-2 candidates by considering the combined evidence from monocular detection and radar measurements. Stage 3 trains a neural network to rescore the candidate positions using additional evidence.

This model should consider multiple factors that indicate the confidence of each choice. For instance, high detection scores imply high confidence in monocular detection; large matching scores indicate more accurate matched positions from Stage 2; monocular detector excels at low ranges while suffers at long ranges where Stage-2 candidates may be a better choice because of the help of radar measurement.

Architecture and Loss. The Stage-3 network takes two types of inputs: (a) monocular detection parameters (class, range, and size); (b) Stage-2 matching scores at binned ranges. The network preprocesses inputs via separate linear layers, concatenates the results, feeds them to an MLP to extract features, and predicts a confidence score per range candidate. Cross-entropy loss is used for training the network. Training data are generated from true positive monocular detections with non-zero Stage-2 matching scores.

Inference. Denoting predicted Stage-3 scores as S STG3 ( i ) , where i is quantized offset to monocular predicted range, we estimate the range offset by finding the peak position as

<!-- formula-not-decoded -->

and corresponding Stage-3 score S STG3 (∆ n ) . The final predicted range R F can be expressed as

<!-- formula-not-decoded -->

where R CAM is monocular predicted range, and b p is bin size. Typically, the estimated range is approximately at one of the peak positions from Stage-2 score or at monocular estimated range; thus Stage 3 implicitly learns to select the best position candidates from previous stages. We also update detection score by combining monocular detection score S CAM and Stage-3 score as

<!-- formula-not-decoded -->

where α is a Stage-3 weighting parameter. Note S STG3 has been processed by the Softmax function, and its value ranges from 0 to 1.

## 4. Experiments

Dataset. Our experiments are based on the widely-used nuScenes dataset [5], with both images and radar points collected in urban driving scenarios. Equipped with six cameras and five radars, the ego-vehicle scans traffic environments in 360 degrees. There are 700 training scenes, 150 validation scenes, and 150 test scenes, each with 10 classes of objects specified with bounding boxes.

Data Splits. We follow the standard splits of the nuScenes detection benchmark: the test results are obtained from model trained on nuScenes training plus validation set ( 34 K frames) and evaluated on the test set ( 6 K frames); the validation results trained on nuScenes training set ( 28 Kframes) and evaluated on the validation set ( 6 K frames).

Implementation Details. Our code uses PyTorch [29] and detection package MMDetection3D [6]. Our experiments use SparseBEV [19] as the monocular detector. Note that RICCARDO is flexible and easily adaptable to other monocular methods. To preserve the premium detection performance of monocular component and focus on training the Stage-3 model, we use pretrained weights for the monocular branch, which are frozen in training and inference. We train Stage-1 and Stage-3 models with RMSProp optimizer for 120 epochs with an initial learning rate of 1 × 10 -6 , which is reduced by half at the 60th epoch. We list the number of parameters in RICCARDO Stage-1 and Stage-3 models as well as in underlying monocular models with two backbones in Tab. 3. RICCARDO Stage 1 and Stage 3 are relatively lightweight compared to monocular models.

Stage 1: We implement the Stage-1 network with a lightweight MLP-like network. An input sample consists of a series of vectors representing different properties of object, e.g. , size, orientation, and range. They are first processed by separate linear projection layers before being concatenated and fed into an MLP of 3 hidden layers. The network output ( i.e. , the binned distribution map) is defined in object local coordinates with X-axis parallel to object length, Y-axis to width, and map center at object center. It has a resolution of 129 × 129 with a pixel size of 0 . 1 × 0 . 1 meters for small and medium-sized object categories and pixel size of 0 . 2 × 0 . 2 meters for large-sized categories such as buses and trailers. GT distribution is generated by accumulating 13 neighboring radar sweeps ( 6 previous sweeps, 1 current, and 6 future ones). We assume radar points distribute within the GT bounding boxes on BEV, and thus points outside the bounding boxes are ignored and not used for training. We train Stage-1 and Stage-3 models separately since Stage-1 model is invariant to its underlying monocular model while Stage-3 network depends on the monocular model.

Stage 2: To perform Stage-2 convolution along the radial direction, the measured radar positions are binned in radial-tangential coordinates to generate a radar measurement map, with X-axis parallel to the ray from ego vehicle to target center and map center at target center detected by a monocular method. It has a resolution of 193 × 193 with 2 pixel sizes mentioned above. Before performing convolution, the predicted radar distribution map of Stage 1 is rotated from object local coordinates to radial-tangential coordinates. The search range of convolution is -3 . 2 m to 3 . 2 m relative to the range of target center estimated by the monocular method.

Table 1. Detection Performance on nuScenes Test Set . RICCARDO achieves SOTA performance for camera-radar fusion.

| Modality Radar Camera   | Method               | NDS ( ↑ )   | mAP ( ↑ )   | mATE ( ↓ )   | mASE ( ↓ )   | mAOE ( ↓ )   | mAVE ( ↓ )   | mAAE ( ↓ )   |
|-------------------------|----------------------|-------------|-------------|--------------|--------------|--------------|--------------|--------------|
| ✓                       | PGD [35]             | 0 . 448     | 0 . 386     | 0 . 626      | 0 . 245      | 0 . 451      | 1 . 509      | 0 . 127      |
| ✓                       | SparseBEV [19]       | 0 . 675     | 0 . 603     | 0 . 425      | 0 . 239      | 0 . 311      | 0 . 172      | 0 . 116      |
| ✓ ✓                     | MVFusion [39]        | 0 . 517     | 0 . 453     | 0 . 569      | 0 . 246      | 0 . 379      | 0 . 781      | 0 . 128      |
| ✓ ✓                     | CRN [11]             | 0 . 624     | 0 . 575     | 0 . 416      | 0 . 264      | 0 . 456      | 0 . 365      | 0 . 130      |
| ✓ ✓                     | RCBEVDet [18]        | 0 . 639     | 0 . 550     | 0 . 390      | 0 . 234      | 0 . 362      | 0 . 259      | 0 . 113      |
| ✓ ✓                     | HyDRa [38]           | 0 . 642     | 0 . 574     | 0 . 398      | 0 . 251      | 0 . 423      | 0 . 249      | 0 . 122      |
| ✓ ✓                     | HVDetFusion [16]     | 0 . 674     | 0 . 609     | 0 . 379      | 0 . 243      | 0 . 382      | 0 . 172      | 0 . 132      |
| ✓ ✓                     | SparseBEV + RICCARDO | 0 . 695     | 0 . 630     | 0 . 363      | 0 . 240      | 0 . 311      | 0 . 167      | 0 . 118      |

Table 2. nuScenes Validation Results .

| Modality Radar Camera   | Method               | NDS ( ↑ )   | mAP ( ↑ )   | mATE ( ↓ )   | mASE ( ↓ )   | mAOE ( ↓ )   | mAVE ( ↓ )   | mAAE ( ↓ )   |
|-------------------------|----------------------|-------------|-------------|--------------|--------------|--------------|--------------|--------------|
| ✓                       | PGD [35]             | 0 . 428     | 0 . 369     | 0 . 683      | 0 . 260      | 0 . 439      | 1 . 268      | 0 . 185      |
| ✓                       | SparseBEV [19]       | 0 . 592     | 0 . 501     | 0 . 562      | 0 . 265      | 0 . 320      | 0 . 243      | 0 . 195      |
| ✓ ✓                     | MVFusion [39]        | 0 . 455     | 0 . 380     | 0 . 675      | 0 . 258      | 0 . 372      | 0 . 833      | 0 . 196      |
| ✓ ✓                     | CRN [11]             | 0 . 607     | 0 . 545     | 0 . 445      | 0 . 268      | 0 . 425      | 0 . 332      | 0 . 180      |
| ✓ ✓                     | RCBEVDet [18]        | 0 . 568     | 0 . 453     | 0 . 486      | 0 . 285      | 0 . 404      | 0 . 220      | 0 . 192      |
| ✓ ✓                     | HyDRa [38]           | 0 . 617     | 0 . 536     | 0 . 416      | 0 . 264      | 0 . 407      | 0 . 231      | 0 . 186      |
| ✓ ✓                     | HVDetFusion [16]     | 0 . 557     | 0 . 451     | 0 . 527      | 0 . 270      | 0 . 473      | 0 . 212      | 0 . 204      |
| ✓ ✓                     | SparseBEV + RICCARDO | 0 . 622     | 0 . 544     | 0 . 481      | 0 . 266      | 0 . 325      | 0 . 237      | 0 . 189      |

Table 3. Number of parameters in SparseBEV with different backbones as well as in RICCARDO Stage 1 and Stage 3.

| Models     | SparseBEV (V2-99)   | SparseBEV (ResNet101)   | Stage 1   | Stage 3   |
|------------|---------------------|-------------------------|-----------|-----------|
| Params (M) | 94 . 0              | 63 . 6                  | 19 . 2    | 0 . 3     |

Stage 3: We implement the Stage-3 network via a lightweight MLP similar to Stage 1. To create labels for GT range, we associate monocular detections with GT bounding boxes under the conditions that the GT bounding boxes fall in the search range of associated monocular detections (with radial distance ≤ 3 . 2 m) and on the ray from ego to monocular detections (with tangential distance ≤ min (0 . 5 m , L ) , where L is the object length).

## 4.1. Quantitative Results on nuScenes

Tabs. 1 and 2 show the performance of RICCARDO on test and validation set, respectively. The proposed fusion of radar points with monocular detection proposals improves the performance of position estimation, which is evaluated with mAP and mean Average Translation Error (mATE) for true positive detections. We compare performance of SOTA methods of monocular and radar-camera fusion. Note that for a fair comparison, the monocular component in RICCARDO, i.e. , SparseBEV, uses exactly the same weights within Tab. 1 and likewise uses the same weights within Tab. 2. Specifically, SparseBEV uses V299 [15] and ResNet101 [8] as its backbones in Tabs. 1 and 2, respectively.

By comparing RICCARDO with its monocular counterpart ( i.e. , 8th row vs. 2nd row in Tabs. 1 and 2), we see a significant improvement in mAP and a reduction in mATE when using RICCARDO with radar data as inputs. Meanwhile it preserves good performance of its monocular component in other aspects, e.g. , size and orientation estimation.

RICCARDO also achieves SOTA performance among published radar-camera fusion methods: in test set performance shown in Tab. 1 and validation set results shown in Tab. 2, RICCARDO achieves the best overall performance measured by NDS and comparable performance in other metrics. As a detection-level fusion, final performance of RICCARDO depends on the quality of its underlying monocular model. We observe a stable and significant improvement in overall performance over its monocular models in both Tabs. 1 and 2, although the monocular components adopt different backbones. The ease of plugging in different monocular components in our fusion architecture allows RICCARDO to capitalize on SOTA monocular models and achieve better fusion performance.

## 4.2. Evaluating Stages 2 and 3

To evaluate the Stages 2 and 3 in object range estimation, we compare the range error from monocular detection, Stage-2, and Stage-3 estimations. Stage-1 model is trained with nuScenes training set and Stage-3 model is trained with detections by the monocular method and corresponding Stage-2 matching scores from nuScenes training set.

<!-- image -->

(f)

Figure 5. Qualitative Results. Visualizations of (a) objects in images, (b) binned radar points, (c) predicted radar hits distribution, (d) Stage-2 matching scores, (e) predicted Stage-3 scores, and (f) detections in ego BEV coordinates. [ Key : Monocular detections, RICCARDO detections, GT. GT boxes are dashed.]

Table 4. Comparison of range estimation accuracy of monocular, Stage-2, and Stage-3 estimates. We use mean/median of absolute range error (meter) as metrics. [ Key : Ped.= Pedestrian, Motor.= Motorcycle, CV= Construction Vehicle, TC= Traffic Cone.]

| Method    | Class-Mean      | Car             | Truck           | Bus             | Trailer         | CV              | Ped.            | Motor.          | Bicycle         | TC              | Barrier         |
|-----------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| Monocular | 0 . 83 / 0 . 57 | 0 . 65 / 0 . 38 | 0 . 86 / 0 . 56 | 1 . 04 / 0 . 80 | 1 . 23 / 1 . 08 | 1 . 13 / 0 . 88 | 0 . 85 / 0 . 55 | 0 . 81 / 0 . 50 | 0 . 65 / 0 . 40 | 0 . 51 / 0 . 24 | 0 . 59 / 0 . 31 |
| Stage 2   | 0 . 94 / 0 . 52 | 0 . 50 / 0 . 21 | 0 . 75 / 0 . 39 | 0 . 77 / 0 . 46 | 1 . 53 / 1 . 05 | 1 . 11 / 0 . 75 | 1 . 13 / 0 . 56 | 0 . 82 / 0 . 35 | 0 . 86 / 0 . 44 | 0 . 93 / 0 . 41 | 1 . 01 / 0 . 55 |
| Stage 3   | 0 . 65 / 0 . 36 | 0 . 38 / 0 . 18 | 0 . 59 / 0 . 34 | 0 . 70 / 0 . 46 | 1 . 13 / 0 . 76 | 0 . 87 / 0 . 63 | 0 . 71 / 0 . 28 | 0 . 57 / 0 . 29 | 0 . 54 / 0 . 24 | 0 . 46 / 0 . 17 | 0 . 57 / 0 . 27 |

Table 5. Comparison of RICCARDO using two baseline radar hit distributions for Stage 1 versus RIC. The metric is mean absolute error (MAE) of range in meters and mean matching score (MMS) between the distributions and actual measurements.

| Distribution   |   MAE ( ↓ ) |   MMS ( ↑ ) |
|----------------|-------------|-------------|
| L-Shaped       |        0.77 |       0.059 |
| Uniform        |        0.67 |       0.078 |
| RIC            |        0.47 |       0.105 |

To generate Stage-2 estimation, the trained Stage-1 model is applied to monocular detections (SparseBEV) in nuScenes validation set to generate radar distributions, which are subsequently convolved with radar measurements (in Stage 2) to obtain matching scores at binned ranges along the ray. The range with the maximum matching score is used as Stage-2 estimation for this evaluation. Finally, we estimate Stage-3 output by feeding the Stage-2 outputs and monocular detections to the Stage-3 model. From Tab. 4, we can see that simple extraction from Stage 2 can improve over monocular methods with lower median error but suffers from outliers with larger mean error. Stage 3 achieves the best range estimation accuracy across all 10 categories, by fusing the monocular estimation and Stage-2 outputs.

## 4.3. Qualitative Results

In Fig. 5, we show examples of RICCARDO being applied to monocular detections. The radar BEV map in (b) and predicted radar distribution in (c) are both centered at the monocular object center and with X-axis being radial direction and Y-axis tangential direction. We can observe the complexity of predicted radar distributions in (c), which vary according to object size and orientation relative to radar rays. The edges of objects facing the radar tend to have higher densities, and large vehicles have a wider spread (see the 4th column), with reflections by parts under the vehicle near the tires.

Fig. 5(d) shows Stage-2 matching scores as a function of radial offset (X-axis), with monocular estimated range (magenta) and GT range (gray). Multiple peaks in the 4th column indicate ambiguities in matching radar hits. (e) shows the predicted Stage-3 scores in blue, which typically have a sharper peak than Stage 2, illustrating Stage-3 candidate refinements. In the fourth column Stage 3 resolved ambiguity by enhancing one peak. Row (f) shows that RICCARDO improves monocular method in radial position prediction as predicted bounding boxes are closer to (dashed) GT compared to monocular detections. The radial directions are plotted as dotted lines for reference.

## 4.4. Ablation on Distribution Models

To assess the benefit of using our trained RIC model to represent radar hits on objects, we compare it with using two baseline distributions, i.e. , a uniform radar distribution within object boundaries and an 'L-shaped' distribution on reflecting sides of bounding boxes. The 'L-shaped' distribution is a simulation of LiDAR point distribution. Each model is passed through Stage 2, and we estimate object range at the maximum matching score, and compute their range errors. We also use matching score ( i.e. , dot product of predicted distribution and pixel-wise radar point counts from accumulated measurement) as an additional metric for evaluating distribution accuracy. We compute mean matching score (MMS) over all classes. We train RIC with radar measurements and GT bounding boxes in nuScenes training set and evaluate on nuScenes validation set. Note that, in this experiment, GT bounding boxes are used to generate the radar distribution, and no monocular boxes are involved. Tab. 5 shows that, with the smallest range estimation error and the largest matching score, the RIC distribution captures more accurately the real radar distribution compared with the two baseline distributions.

## 5. Conclusions

This paper presents a novel radar-camera fusion strategy that utilizes BEV radar distributions to improve object range estimation over monocular methods. Evaluation on the nuScenes dataset shows that RICCARDO realizes stable and significant improvements in object position estimation over its underlying monocular detector, and achieves the SOTA performance in radar-camera-based 3D object detection. We believe this effective method, that is simple to implement, will broadly benefit existing and future cameraradar fusion methods.

Limitations. First, as a detection level fusion, RICCARDO only uses high-level monocular detection parameters and does not directly utilize low-level image features. Information loss is inevitable in this low-level to high-level feature transition ( e.g. , false negative detections), and it is difficult to make use of radar points for further improvement if an object is missed by the monocular component in the first place. Second, RICCARDO adopts BEV representation for radar points distribution. However, BEV representation has an intrinsic limitation to represent radar hits from two vertically positioned targets at the same BEV location ( e.g. , a person riding on a bicycle), since their reflected radar hits are mapped to the same BEV pixel.

Future Work. RICCARDO assumes the underlying monocular detector has accurate estimation in tangential position, size and orientation and focuses on improving range estimation, and also assumes the distribution remain fixed in the search space. To achieve more precise radar distribution matching, it is worthwhile in future work expanding the search space from a 1D ( i.e. , radial offset) to a high-dimensional space ( e.g. , radial and tangential offsets, size, and orientation) and adopting a variable distribution as a function of locations in the search space during matching.

## References

- [1] Geonho Bang, Kwangjin Choi, Jisong Kim, Dongsuk Kum, and Jun Won Choi. Radardistill: Boosting radar-based object detection performance via knowledge distillation from lidar features. In CVPR , 2024. 2
- [2] Garrick Brazil, Abhinav Kumar, Julian Straub, Nikhila Ravi, Justin Johnson, and Georgia Gkioxari. Omni3D: A large benchmark and model for 3 D object detection in the wild. In CVPR , 2023. 2
- [3] Garrick Brazil and Xiaoming Liu. M 3 D-RPN: Monocular 3 D region proposal network for object detection. In ICCV , 2019. 2
- [4] Garrick Brazil, Gerard Pons-Moll, Xiaoming Liu, and Bernt Schiele. Kinematic 3 D object detection in monocular video. In ECCV , 2020. 2
- [5] Holger Caesar, Varun Bankiti, Alex Lang, Sourabh Vora, Venice Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuScenes: A multimodal dataset for autonomous driving. In CVPR , 2020. 1, 5, 12
- [6] MMDetection3D Contributors. MMDetection3D: OpenMMLab next-generation platform for general 3 D object detection. https://github.com/open-mmlab/ mmdetection3d , 2020. 5
- [7] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the KITTI vision benchmark suite. In CVPR , 2012. 1
- [8] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR , 2016. 6, 11
- [9] Youngseok Kim, Jun Won Choi, and Dongsuk Kum. GRIF Net: Gated region of interest fusion network for robust 3 D object detection from radar point cloud and monocular image. In IROS , 2020. 2
- [10] Youngseok Kim, Sanmin Kim, Jun Won Choi, and Dongsuk Kum. CRAFT: Camera-radar 3 D object detection with spatio-contextual fusion transformer. In AAAI , 2023. 2
- [11] Youngseok Kim, Juyeb Shin, Sanmin Kim, In-Jae Lee, Jun Won Choi, and Dongsuk Kum. CRN: Camera radar net for accurate, robust, efficient 3 D perception. In ICCV , 2023. 2, 6
- [12] Abhinav Kumar, Garrick Brazil, Enrique Corona, Armin Parchami, and Xiaoming Liu. DEVIANT: Depth equivariant network for monocular 3 Dobject detection. In ECCV , 2022. 1, 2
- [13] Abhinav Kumar, Garrick Brazil, and Xiaoming Liu. GrooMeD-NMS: Grouped mathematically differentiable NMS for monocular 3 D object detection. In CVPR , 2021. 2
- [14] Abhinav Kumar, Yuliang Guo, Xinyu Huang, Liu Ren, and Xiaoming Liu. SeaBird: Segmentation in bird's view with dice loss improves monocular 3 D detection of large objects. In CVPR , 2024. 2
- [15] Youngwan Lee and Jongyoul Park. CenterMask: Real-time anchor-free instance segmentation. In CVPR , 2020. 6
- [16] Kai Lei, Zhan Chen, Shuman Jia, and Xiaoteng Zhang. HVDetFusion: A simple and robust camera-radar fusion framework. arXiv preprint arXiv:2307.11323 , 2023. 6
- [17] Juan-Ting Lin, Dengxin Dai, and Luc Van Gool. Depth estimation from monocular images and sparse radar data. In

IROS , 2020. 2

- [18] Zhiwei Lin, Zhe Liu, Zhongyu Xia, Xinhao Wang, Yongtao Wang, Shengxiang Qi, Yang Dong, Nan Dong, Le Zhang, and Ce Zhu. RCBEVDet: Radar-camera fusion in bird's eye view for 3 D object detection. In CVPR , 2024. 6
- [19] Haisong Liu, Yao Teng, Tao Lu, Haiguang Wang, and Limin Wang. SparseBEV: High-performance sparse 3 D object detection from multi-camera videos. In ICCV , 2023. 2, 5, 6, 11, 12
- [20] Zongdai Liu, Dingfu Zhou, Feixiang Lu, Jin Fang, and Liangjun Zhang. AutoShape: Real-time shape-aware monocular 3 D object detection. In ICCV , 2021. 2
- [21] Yunfei Long, Abhinav Kumar, Daniel Morris, Xiaoming Liu, Marcos Castro, and Punarjay Chakravarty. RADIANT: Radar image association network for 3 Dobject detection. In AAAI , 2023. 2
- [22] Yunfei Long, Daniel Morris, Xiaoming Liu, Marcos Castro, Punarjay Chakravarty, and Praveen Narayanan. Full-velocity radar returns by radar-camera fusion. In ICCV , 2021. 2
- [23] Yunfei Long, Daniel Morris, Xiaoming Liu, Marcos Castro, Punarjay Chakravarty, and Praveen Narayanan. Radarcamera pixel depth association for depth completion. In CVPR , 2021. 2
- [24] Yan Lu, Xinzhu Ma, Lei Yang, Tianzhu Zhang, Yating Liu, Qi Chu, Junjie Yan, and Wanli Ouyang. Geometry uncertainty projection network for monocular 3 Dobject detection. In ICCV , 2021. 1
- [25] Xinzhu Ma, Yinmin Zhang, Dan Xu, Dongzhan Zhou, Shuai Yi, Haojie Li, and Wanli Ouyang. Delving into localization errors for monocular 3 D object detection. In CVPR , 2021. 2
- [26] Ramin Nabati and Hairong Qi. CenterFusion: Center-based radar and camera fusion for 3 D object detection. In WACV , 2021. 1, 2
- [27] Aarav Pandya, Ajit Jha, and Linga Reddy Cenkeramaddi. A velocity estimation technique for a monocular camera using mmWave FMCW radars. Electronics , 2021. 2
- [28] Dennis Park, Rares Ambrus, Vitor Guizilini, Jie Li, and Adrien Gaidon. Is Pseudo-LiDAR needed for monocular 3 D object detection? In ICCV , 2021. 1
- [29] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, high-performance deep learning library. In NeurIPS , 2019. 5
- [30] Cody Reading, Ali Harakeh, Julia Chae, and Steven Waslander. Categorical depth distribution network for monocular 3 D object detection. In CVPR , 2021. 2
- [31] Andrea Simonelli, Samuel Bul` o, Lorenzo Porzi, Manuel Antequera, and Peter Kontschieder. Disentangling monocular 3 D object detection: From single to multi-class recognition. TPAMI , 2020. 2
- [32] Andrea Simonelli, Samuel Bul` o, Lorenzo Porzi, Peter Kontschieder, and Elisa Ricci. Are we missing confidence in Pseudo-LiDAR methods for monocular 3 D object detection? In ICCV , 2021. 2
- [33] Akash Singh, Yunhao Ba, Ankur Sarker, Howard Zhang,

Achuta Kadambi, Stefano Soatto, Mani Srivastava, and Alex Wong. Depth estimation from camera image and mmWave radar point cloud. In CVPR , 2023. 2

- [34] Xiaolin Tang, Zhiqiang Zhang, and Yechen Qin. On-road object detection and tracking based on radar and vision fusion: A review. IEEE ITS Magazine , 2021. 2
- [35] Tai Wang, Xinge Zhu, Jiangmiao Pang, and Dahua Lin. Probabilistic and geometric depth: Detecting objects in perspective. In CoRL , 2021. 6
- [36] Yan Wang, Wei-Lun Chao, Divyansh Garg, Bharath Hariharan, Mark Campbell, and Kilian Weinberger. Pseudo-LiDAR from visual depth estimation: Bridging the gap in 3 D object detection for autonomous driving. In CVPR , 2019. 2
- [37] Yue Wang, Vitor Guizilini, Tianyuan Zhang, Yilun Wang, Hang Zhao, and Justin Solomon. DETR3D: 3 D object detection from multi-view images via 3 D-to2 D queries. In CoRL , 2021. 2
- [38] Philipp Wolters, Johannes Gilg, Torben Teepe, Fabian Herzog, Anouar Laouichi, Martin Hofmann, and Gerhard Rigoll. Unleashing HyDRa: Hybrid fusion, depth consistency and radar for unified 3 D perception. arXiv preprint arXiv:2403.07746 , 2024. 6
- [39] Zizhang Wu, Guilian Chen, Yuanzhu Gan, Lei Wang, and Jian Pu. MVFusion: Multi-view 3 D object detection with semantic-aligned radar and camera fusion. In ICRA , 2023. 2, 6
- [40] Shanliang Yao, Runwei Guan, Xiaoyu Huang, Zhuoxiao Li, Xiangyu Sha, Yong Yue, Eng Gee Lim, Hyungjoon Seo, Ka Lok Man, Xiaohui Zhu, et al. Radar-camera fusion for object detection and semantic segmentation in autonomous driving: A comprehensive review. T-IV , 2023. 2
- [41] Georgios Zamanakos, Lazaros Tsochatzidis, Angelos Amanatiadis, and Ioannis Pratikakis. A comprehensive survey of LIDAR-based 3 D object detection methods with deep learning for autonomous driving. Computers &amp; Graphics , 2021. 1
- [42] Diankun Zhang, Zhijie Zheng, Haoyu Niu, Xueqing Wang, and Xiaojun Liu. Fully sparse transformer 3 D detector for LiDAR point cloud. TGRS , 2023. 2
- [43] Taohua Zhou, Junjie Chen, Yining Shi, Kun Jiang, Mengmeng Yang, and Diange Yang. Bridging the view disparity between radar and camera features for multi-modal fusion 3 D object detection. T-IV , 2023. 2
- [44] Yunsong Zhou, Yuan He, Hongzi Zhu, Cheng Wang, Hongyang Li, and Qinhong Jiang. MonoEF: Extrinsic parameter free monocular 3 D object detection. TPAMI , 2021. 2

## RICCARDO: Radar Hit Prediction and Convolution for Camera-Radar 3D Object Detection

## Supplementary Material

Figure 1. Stage-1 Network Structure. The class input is in one-hot encoding; z represents heights of bounding box bottom faces; θ AZ stands for azimuths of objects in ego coordinates; θ Y and θ ′ Y are object yaws in ego coordinates and relative yaws ( i.e. , θ Y -θ AZ), respectively. 'C' represents concatenation, and 'Linear' denotes a linear transformation layer. Feature sizes are marked besides network layers.

<!-- image -->

Figure 2. Stage-3 Network Structure. The inputs v x and v y are monocular estimated object velocities in ego coordinates; v R and v T are monocular velocities in radial and tangential directions, respectively; S CAM and S STG2 represent monocular detection scores and Stage-2 matching scores, respectively.

<!-- image -->

## 1. Additional Implementation Details

## 1.1. Detailed Network Structure

Figs. 1 and 2 show network structures of Stages 1 and 3, respectively.

## 1.2. Inference Time

Using a NVIDIA V100s GPU and Intel Xeon Platinum 8260 CPUs, we record in Tab. 1 inference time for differ- ent components of RICCARDO. Radar processing refers to accumulation and BEV binning of 7 radar sweeps. We can see the monocular component takes most of the inference time and in comparison Stages 1 to 3 are very fast. Radar processing has not been optimized and could be sped up through code optimization.

Table 1. Inference time for SparseBEV with different backbones, radar processing as well as RICCARDO Stages 1 to 3.

| Components   | SparseBEV (V2-99)   | SparseBEV (ResNet101)   | Radar Processing   | Stage 1   | Stage 2   | Stage 3   |
|--------------|---------------------|-------------------------|--------------------|-----------|-----------|-----------|
| Time (ms)    | 575 . 7             | 211 . 5                 | 105 . 4            | 1 . 0     | 12 . 7    | 1 . 2     |

## 2. Additional Ablation Studies

In the following ablations, we use SparseBEV [19] with backbone ResNet101 [8] for the monocular components in RICCARDO. For efficiency, the data used for evaluation are a subset of nuScenes validation set with 600 random samples.

## 2.1. Ablation on Velocity Used for Point Motion Compensation

When accumulating 7 radar sweeps in inference, we used estimated radar point velocity to compensate motions of moving points. We implement different velocity estimations and compare resultant detection performance in Tab. 2. The velocity estimations include 0 , i.e. , no motion compensation, Doppler velocity, Doppler velocity backprojected to estimated object heading direction, Doppler velocity plus tangential component of monocular estimated velocity, and monocular velocity. The geometric relation between full velocity and its tangential and radial components are shown in Fig. 3. From Tab. 2 we can see performing motion compensation improves detection performance and using full velocity estimates achieves better accuracy compared with only applying Doppler velocity. The three full velocity estimates shown on the 4 th to 6 th rows result in almost the same detection performance.

Figure 3. Geometric relation between point velocity and its radial and tangential components.

<!-- image -->

Table 2. Ablation on different velocity estimations for compensating motion during radar sweep accumulation. Keys: Vel.= Velocity; Mono.= Monocular

| Point Vel. Estimation           | NDS ( - )   | mAP ( - )   |
|---------------------------------|-------------|-------------|
| 0                               | 0 . 614     | 0 . 534     |
| Doppler Vel.                    | 0 . 620     | 0 . 544     |
| Back-Projected Doppler Vel.     | 0 . 621     | 0 . 545     |
| Doppler + Tangential Mono. Vel. | 0 . 621     | 0 . 545     |
| Mono. Vel.                      | 0 . 621     | 0 . 546     |

Table 3. Ablation on updating range and detection score with fusion weight α

| Update Range   | Update Score   | α     | NDS ( - )   | mAP ( - )   |
|----------------|----------------|-------|-------------|-------------|
|                |                | -     | 0 . 590     | 0 . 501     |
|                | ✓              | 0 . 5 | 0 . 593     | 0 . 503     |
| ✓              |                | -     | 0 . 617     | 0 . 543     |
| ✓              | ✓              | 0 . 2 | 0 . 620     | 0 . 545     |
| ✓              | ✓              | 0 . 5 | 0 . 621     | 0 . 545     |
| ✓              | ✓              | 0 . 8 | 0 . 621     | 0 . 543     |
| ✓              | ✓              | 1 . 0 | 0 . 620     | 0 . 541     |

## 2.2. Ablation on Range and Score Updating

In Stage-3 inference we update both range and detection score. Detection scores indicate confidence in prediction and have an impact on mAP computation, where predictions with higher scores have priority as true positives to be associated with GT. We update detection scores by adding Stage-3 scores weighted by α to monocular scores. We test different range and score updating options with different α and list resultant detection performance in Tab. 3. We can see both range and score updating improve detection performance while range updating has significantly bigger impacts on performance.

## 2.3. Ablation on Number of Radar Sweeps

Within 0 . 5 s time window, there are about 7 sweeps of radar points ( i.e. , 1 current plus 6 past ones) from radars running at 13 Hz in nuScenes Dataset [5]. We accumulate multiple radar sweeps during inference, and the number of radar sweeps may impact detection performance, as more sweeps provide denser radar measurement used for Stage 2. To ver- ify this, we run RICCARDO multiple times with radar input from 0 , 1 , 3 , 5 , and 7 sweeps, respectively and record their detection performance. Note using 0 radar sweep refers to applying only monocular detector without fusion. As shown in Tab. 4, more radar sweeps lead to better detection performance as expected.

Table 4. Ablation on Number of Radar Sweeps. More radar sweeps result in better detection performance. Key: Num.= Number

|   Num. of Sweeps | NDS ( - )   | mAP ( - )   |
|------------------|-------------|-------------|
|                0 | 0 . 590     | 0 . 501     |
|                1 | 0 . 597     | 0 . 512     |
|                3 | 0 . 612     | 0 . 531     |
|                5 | 0 . 618     | 0 . 541     |
|                7 | 0 . 621     | 0 . 545     |

Table 5. Detection performance NDS( ↑ ) / mAP( ↑ ) of RICCARDO and its underlying monocular detector in night, daytime and all validation scenes, respectively.

| Scene                | Night         | Daytime       | Overall       |
|----------------------|---------------|---------------|---------------|
| SparseBEV [19]       | 0.526 / 0.400 | 0.673 / 0.601 | 0.669 / 0.595 |
| SparseBEV + RICCARDO | 0.561 / 0.450 | 0.704 / 0.642 | 0.699 / 0.636 |
| Number of Samples    | 602           | 5417          | 6019          |

## 3. Additional Analyses

## 3.1. Performance at Night

Although using radar to handle adverse conditions is a different research focus, we show in Tab. 5 that RICCARDOsignificantly improves detection performance over the underlying monocular detector under challenging lighting conditions at night. We evaluate on 7 object categories, which appear in night scenes in nuScenes validation set. We also list corresponding daytime and overall performance for reference.

## 3.2. Sensitivity to Monocular Prediction Errors

To assess sensitivity of the Stage-2 prediction to monocular prediction errors, we add errors to the bounding box parameters, which are inputs to the radar distribution model, and compute the MAE of range estimation in Stage 2. The Fig. 4 shows the range error modestly increases with errors in heading angle and size, demonstrating good monocular error toleration.

## 4. Additional Visualizations

## 4.1. Visualization of Radar Distributions

To visualize how predicted distribution varies with viewing angles, we simulate object parameters with different orientations and apply Stage-1 model to generate corresponding radar hit distributions. Figs. 5 and 6 shows predicted distributions for car, bus, bicycle, and barrier with different orientations and distances. We can see the distributions vary with category, orientation, and distance. For example, radar distributions are less concentrated spatially at longer range because of larger beam width. We can also notice that distributions of radar points reflected by the tail and head of cars (as shown in the 1st and last row of Fig. 5) are different because of their different surface shapes. More visualizations of predicted radar distributions for objects rotating 360 degrees are shown in the attached video demo.

Figure 4. Sensitivity of the Stage-2 prediction to monocular prediction errors in heading (top) and size (bottom).

<!-- image -->

Figure 5. Visualization of predicted radar distributions of (a)(b) Car and (c)(d) Bus viewed from different angles and distances of 10 and 40 meters. X-axis represents radial positions, and Y-axis denotes tangential offsets to object centers. Radial rays are plotted as horizontal dotted lines. Target bounding boxes are shown on top of distributions, and dashed lines represent object head.

<!-- image -->

Figure 6. Visualization of predicted radar distributions of (a)(b) bicycle and (c)(d) barrier viewed from five different angles and from distances of 10 and 40 meters.

<!-- image -->