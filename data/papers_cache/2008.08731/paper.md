## GPR-based Subsurface Object Detection and Reconstruction Using Random Motion and DepthNet

Jinglun Feng 1 , † , Liang Yang 1 , † , Haiyan Wang 1 , Yifeng Song 2 , Jizhong Xiao* 1

Abstract -Ground Penetrating Radar (GPR) is one of the most important non-destructive evaluation (NDE) devices to detect the subsurface objects (i.e. rebars, utility pipes) and reveal the underground scene. One of the biggest challenges in GPR based inspection is the subsurface targets reconstruction. In order to address this issue, this paper presents a 3D GPR migration and dielectric prediction system to detect and reconstruct underground targets. This system is composed of three modules: 1) visual inertial fusion (VIF) module to generate the pose information of GPR device, 2) deep neural network module (i.e., DepthNet) which detects B-scan of GPR image, extracts hyperbola features to remove the noise in B-scan data and predicts dielectric to determine the depth of the objects, 3) 3D GPR migration module which synchronizes the pose information with GPR scan data processed by DepthNet to reconstruct and visualize the 3D underground targets. Our proposed DepthNet processes the GPR data by removing the noise in B-scan image as well as predicting depth of subsurface objects. For DepthNet model training and testing, we collect the real GPR data in the concrete test pit at Geophysical Survey System Inc. (GSSI) and create the synthetic GPR data by using gprMax3.0 simulator. The dataset we create includes 350 labeled GPR images. The DepthNet achieves an average accuracy of 92 . 64% for B-scan feature detection and an 0 . 112 average error for underground target depth prediction. In addition, the experimental results verify that our proposed method improve the migration accuracy and performance in generating 3D GPR image compared with the traditional migration methods.

## I. INTRODUCTION

Ground Penetrating Radar (GPR) has become an important tool for subsurface non-destructive inspection [1]. By using a GPR cart, subsurface inspection on bridge decks and other concrete structures becomes a routine task in addition to visual inspection of surface defects [2], [3], [4], [5]. However, current GPR inspection still relies on on-site engineers to push the GPR cart along the survey grid lines to collect GPR data. Furthermore, the conventional B-scan data is difficult to interpret and requires experienced geophysicist to reveal the underground objects. It is desirable to design a new GPR system and migration algorithms to automatically collect GPR data in random motion pattern and analyze the data to reconstruct the underground objects.

When a GPR system is used for utility survey in outdoor inspection [6], [7], GPS is available to provide pose information [8], [9]. However, for indoor inspection, we still need to find a way to obtain positioning information since GPS could not be used in indoor environment. Besides, the current commercial GPR cart must move along either horizontal or perpendicular lines of a pre-defined survey grid to trigger GPR scan by survey wheel to recover the underground objects.

1 Electrical Engineering Department, The City College of New York, New York, USA. † The first two authors are equally contributed. jfeng1,lyang1@ccny.cuny.edu , hwang005@citymail.cuny.edu , the corresponding author is jxiao@ccny.cuny.edu

2 University of Chinese Academy of Sciences,Shenyang Institute of Automation, Chinese Academy of Sciences songyifeng@sia.cn

Fig. 1. 3D GPR migration and depth prediction is an challenging problem. We aim to provide a systematic approach to reconstruct the subsurface objects.

<!-- image -->

In order to generate 3D structure from the B-scan GPR data, migration algorithm [10], [11], [12] is the most important step to achieve this goal. Authors in [13] proposed a hybrid migration method which is Fourier finite-difference migration, it achieves the complex underground targets reconstruction. A full-resolution GPR imaging method is proposed in [14], by obtaining a spatial sampling of GPR recording, noninvasive GPR imaging could be generated. Moreover, [15] introduces a migration imaging method for stepped frequency continuous wave (SFCW) GPR system, which is based on compressive sensing algorithm. With this approach, the delivery of high-quality GPR image of underground region is prominent and robust. But, the migration methods so far are not able to eliminate background noise of GPR data.

Besides the migration and perception, GPR B-scan feature detection is also a significant topic since each B-scan feature defines subsurface target information. W. Al-Nuaimy et. al. from University of Liverpool [16] first proposed underground targets detection method by implementing Hough transform on GPR signals. They use back-propagation method to identify portions of the GPR image corresponding to target reflections. Moreover, as an automatic feature recognition method, [17] applied center-surround difference detecting and fuzzy logic in GPR feature detection, which improve the tolerance of changes in viewpoint of GPR image. At last, as for deep learning method, [18] [19] [20] [21] employed deep convolutional neural networks (DNNs) to extract meaningful signatures from 2D B-scan image and detect underground objects. However, towards data-driven visual inspection of subsurface targets reconstruction, there are still some challenges needed to be solved as following

Fig. 2. Flow chart of the proposed DNN based GPR Migration framework. The whole system consists of three modules: visual inertial odometry positioning system, GPR migration system and DepthNet neural network model for 3D subsurface targets reconstruction.

<!-- image -->

- Firstly, the current GPR data collection still needs to be finished along either horizontal or perpendicular lines of a pre-defined survey grid triggered by GPR survey wheel, which is not robust and efficient for data collection.
- Secondly, without the GPR data de-noised processing, background noise in GPR data would extremely effect migration result.
- Finally, a proper permittivity for subsurface material is highly relied on the pre-knowledge of geophysicist, there is still lack of a method to predict the dielectric based on GPR B-scan data.

To address the above challenges, this paper developed an 3D GPR migration and object detection method to enable the localization and visualization of the subsurface targets. We first propose using visual inertial fusion to estimate the pose of the GPR device. GPR is triggered by survey wheel to collect A-scan data. After synchronizing the A-scan data with pose information, B-scan data is generated. Then, we employ a Faster R-CNN [22] to locate the GPR B-scan hyperbolic features and output the corresponding bounding boxes. Once we got the Faster R-CNN detection results, we only keep the B-scan data in bounding boxes to do the migration while discard the rest of the data outside the bounding boxes, which has the effect to remove the noise. We propose the DepthNet to predict target depth. It takes the raw B-scan data with different dielectric values as input, and predicts the target depth in current GPR B-scan image. Finally, as depicted in Fig.1, our system performs migration algorithm based on the de-noised data, and reconstruct the target area based on our predicted depth of subsurface targets.

## II. SYSTEM ARCHITECTURE

Our proposed system architecture is illustrated in Fig.2, which is composed of three modules: 1) visual inertial fusion (VIF) module to generate pose information of GPR device, 2) DepthNet which detects B-scan of GPR image, extracts hyperbola features to remove the noise in B-scan data and predicts dielectric to determine the depth of the objects, 3) 3D GPR migration module which synchronizes the pose information with GPR scan data processed by DepthNet to reconstruct and visualize the 3D underground targets. Our goal is to enable the 3D subsurface object reconstruction and visualization, by means of our DepthNet based object detection and migration methods.

## A. Visual Inertial Odometry Positioning System

In order to provide pose information for migration, we introduce visual inertial fusion (VIF) to obtain 6 DOF pose of GPR device. The VIF pipeline is illustrated in Fig.2, where we use Intel D435i realsense camera which has Inertial Measurement Unit (IMU) and RGB-D camera embedded in the system.

The VIF implementation is based on [23], where we fuse the IMU and the visual odometry in a loosely-approach, that is, the IMU performs pose estimation as the prediction while take visual odometry information as correction.

We first take the RGB and depth frames as input to initialize our pose and coordinate system. Meanwhile, IMU measurement is fed as propagation to estimate the pose of the IMU [24]. Finally, we fuse the pose calculated by visual odometry as the observation information to update the state.

Therefore, the current camera pose Ti ∈ SE ( 3 ) could be estimated[25].

## B. GPR data collection and Labeling

In order to facilitate the understanding of the raw data to help us to label the ground truth for our model training, we perform field data collection at a well designed test facility, i.e., Geophysical Survey Systems, Inc. (GSSI) test pit at Nashua, New Hampshire. For all collected data, it comes with the ground truth information including depth and length information of utility pipes, rebars and tanks.

Our data collection follows the following steps which is highly recommended by the GSSI site engineers,

- Firstly, we set up a grid area with equal line spacing to cover the whole test pit.
- Secondly, we mount the camera on the GPR cart to generate pose, then we use ROS to subscribe pose data and GPR scan data in order to make them synchronized.
- Finally, we move the platform in a zigzag pattern that defined in the first step and record both GPR sensory and pose information. A total of 140 set of data were collected and each set contains an average of 800 Ascan data synchronized with pose.

Once we obtain all the field data, we label each single measurement in two aspects. First, this paper transform all the B-scan measurement into color image encoded with 'Hot' colormap. Then, we annotate the Region of Interest (ROI), that is, the hyperbolic feature with a bounding box which is described by { xmin , ymin , xmax , ymax } . Besides the bounding box, we also assign a dielectric value for each measurement as the ground truth. The system is used to predict dielectric is used to estimate the depth of the subsurface objects.

Fig. 3. (a) Test pit in GSSI's garage, multiple targets buried underneath the surface. (b) Test slab in GSSI's slab room, multiple utility pipes inserted in the slab.

<!-- image -->

<!-- image -->

## C. GPR Migration Toward 3D Reconstruction

Migration is a hyperbolic shape analysis approach to reconstruct the subsurface structure with 3D output, acting in a spatial deconvolution manner which belongs to Back Projection (BP) methodology [26]. Migration is highly relied on dielectric of subsurface materials. It uses the dielectric value to calculate the signal propagation velocity in the medium from the hyperbolic features [27]. Once the velocity is obtained, depth scale for subsurface targets could be reconstructed.

Each GPR migration measurement, on a macroscopic scale, can be described by the well-known Maxwells equations (see Equ.1) [28]. The first order partial differential equations express the relations between the fundamental electromagnetic field quantities and their dependence on their sources [29].

<!-- formula-not-decoded -->

Where glyph[vector] E is the electric field strength vector (V/m); r e 0 is the electric charge density ( C / m 3 ); glyph[vector] B is the magnetic flux density vector (T) while glyph[vector] J is the electric current ( A / m 2 ); glyph[vector] D is the electric displacement vector ( C / m 2 ); t is time (s) as well as glyph[vector] H is the magnetic field intensity ( A / m ).

In Maxwells equations, the field vectors are assumed to be single-valued, bounded, and continuous functions of position and time. In order to simulate the GPR response from a particular target or set of targets, the above equations have to be solved subject to the geometry constraints of the problem and the initial conditions.

For migration, the first step is to calculate the two-way travel times (TWTT) of an electromagnetic wave from the transmitter to subsurface targets at which reflection occurs and returns back to receiver within the migration domain.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where D represents different dielectric, which is the conductivity of the two materials (in this paper, we proposed a learning approach to obtain this parameter III-B); C denotes the velocity of light, Dtar means depth of subsurface targets while T tr means the two way travel time of GPR's antenna.

In second step, we implement the extrapolation of electromagnetic wave back to time domain by using twodimensional Maxwell's equations.

<!-- formula-not-decoded -->

where Ey ( x , z , t ) is the electrical field in y component. Hx ( x , z , t ) and Hz ( x , z , t ) are magnetic fields in x -and z components respectively. l represents the permittivity of the subsurface of material. E ′ y ( x , z , t ) denotes the observed electric field in y direction. In this way, the Maxwell's equations are solved by introducing the electromagnetic wave extrapolation, two-way travel times could be transferred into the distance between the GPR's antenna to the subsurface targets.

The third step, we use Back Projection where the distance calculated in the second step is taken as the radius r . At each GPR measurement point, migration will take this point as the center and generate a semi-hemisphere with radius r . The potential target could be shown up on any points located at the surface of this semi-hemisphere. Along with the movement of GPR measurement, there will be more semi-hemispheres with different radius get generated, their intersection should be the location of the targets. By this way, a 3D migration image could be generated.

Finally, we propose a new method on GPR migration. The traditional way can only achieve pseudo 3D GPR imaging because the GPR data is collected along either horizontal or perpendicular directions of the pre-defined survey grid. However, by receiving rotation information from VIF pose estimation system, the GPR cart is able to collect the GPR data in any directions, thus constructs real 3D GPR image. In Equ.5, A pre denotes the coordinate of previous antenna while A update represents the coordinate of updated antenna. q is the antenna rotation angle.

<!-- formula-not-decoded -->

## D. DNN based Target Detection and Depth Prediction

Migration is a process that transforms the 2D B-scan into the 3D GPR imaging. However, due to subsurface noise and the uncertainty of the material dielectric, it is almost impossible to reconstruct the real 3D GPR imaging accurately. Thus, we propose two models to solve the problems, which are GPR object detection model and dielectric prediction model. The details of the DNN network is illustrated in Fig.2, where we use Faster R-CNN [30] to perform subsurface object detection and a new DepthNet to predict the dielectric of the subsurface material.

1) GPR Based Object Detection: In order to produce a clear 3D GPR imaging from 2D B-scan data, we use Faster R-CNN [30], an inception architecture with residual connections network, to detect B-scan hyperbolic features. Then, we take the detected bounding boxes as the region of interest (RoI). The data outside the RoI are considered as noise which is expected to be discarded in migration process. Moreover, more object detection networks are used in this section in order for comparison of Intersect of Union (IoU). [31].

2) DepthNet for Depth Prediction: In our work, the DepthNet outputs the dt ( i ) , i ∈{ 0 , 1 , ... } for each of embedded targets. As illustrated in Fig.2, the DepthNet consists of three sub-nets, where the lowermost model predicts the dielectric of the subsurface material, the uppermost and the middle model are fully-connected nets which take the Faster RCNN bounding boxes and features as the input. The dielectric model takes the raw B-scan image as input and resize it to 224 × 224 while the encoder is Resnet101. Then DepthNet takes the dielectric model output features and the other two layers features as the input to predict the depth of each target.

Fig. 4. Proposed DepthNet framework. DepthNet consists of two parts: (1) B-scan Detection and (2) Noise Cancellation.

<!-- image -->

Loss Design: After predicting the depth of each target and the dielectric, we optimize the model by using a weighted sum error approach to regress the model,

<!-- formula-not-decoded -->

where D / D yi is the ground truth depth, D / D y p i is the depth prediction, l D and l D are the weight for dielectric and depth loss respectively. In this paper, we employ mean square error loss to regress the depth prediction model.

## E. Target Visualization

In order to visualize and reconstruct the subsurface objects, we combine DNN based GPR images with our proposed migration method. As it is illustrated in Fig.4, we considered detected B-scan hyperbolic features and the corresponding bounding boxes as the region of interest (RoI). Then, only the GPR B-scan data in RoI would be fed into migration processing in order to obtain the de-noised subsurface objects. In this way, we can easily estimate and localize the subsurface objects as well as the depth information for each targets, where the depth information is what also obtained from DepthNet. Besides, the noise map we recovered from the removed noise is also considered as an evaluation of our proposed system.

## III. EXPERIMENTS

We verify our GPR underground objects reconstruction system on the dataset we created according to Section IIB. We illustrate our newly proposed migration algorithm and demonstrate the effectiveness the 3D GPR subsurface reconstruction by using our learning based approach. For all these evaluations, they are conducted on an GPU server, with Intel Core i9-9900K 3.2GHz CPU, GeForce RTX 2080 Ti GPU, and 32GB RAM.

Fig. 5. Front view of fine tune migration method compared with traditional migration method.

<!-- image -->

Fig. 6. Proposed migration results in GSSI's test pit compare with ground truth and traditional migration methods. (a) shows the target reconstruction result used by proposed migration method; (b) and (c) shows the limitation of traditional method; (d) shows the ground truth of subsurface targets, which are located at different layers with different depth. All the graphs show above are in top view.

<!-- image -->

## A. Performance of Migration

In order to verify the migration method we proposed in this paper, a 3D subsurface image of test pit is first generated and compared with the ground truth provided by GSSI. The pose information obtained from VIF estimation system and the GPR scan data are considered as the input for migration. The target in the Fig.6 is a right angle shape pipe, which located at 0.9 meter beneath the surface of test pit. By implementing the traditional method, targets could only be reconstructed either in horizontal direction or vertical direction, which will make the migration result be not intact. However, different dielectric value could also influence the accuracy of migration. Due to this reason, the necessary of the depth prediction of underground targets will be testified in the next section.

## B. Performance Comparison of GPR Target Detection and Depth Prediction

## Object Detection Comparison

Table.I shows the results of B-scan feature detection compared with different models [21], [30], [32], [33]. The results show that Faster R-CNN resnet101 has the best performance on GPR based object detection. Once we obtain the detection results, our proposed DepthNet is implemented as the dielectric prediction, that is actually, subsurface targets depth prediction. In Table.I, mAP@IoU = 0.75 is mean average precision at 75% IoU, mAP@IoU = 0.50 is mean average precision at 50% IoU, AR@10 is average recall with 10 detections, AR@100 is average recall with 100 detections.

TABLE I DETECTION PERFORMANCE COMPARISON.

|                           | mAP@IoU   | mAP@IoU   | AR     | AR     |
|---------------------------|-----------|-----------|--------|--------|
| Models                    | 0 . 75    | 0 . 50    | 10     | 100    |
| YOLO v3                   | 89 . 6    | 89 . 3    | 90 . 2 | 91 . 2 |
| ssd mobilenet v1          | 85 . 8    | 88 . 6    | 84 . 2 | 84 . 9 |
| ssd mobilenet v2          | 86 . 1    | 87 . 8    | 87 . 7 | 87 . 0 |
| ssd Inception v2          | 88 . 4    | 90.1      | 88 . 9 | 85 . 6 |
| ssdlite mobilenet v2      | 82 . 2    | 83 . 3    | 89 . 2 | 89 . 3 |
| Faster R-CNN Inception v2 | 89 . 3    | 89 . 3    | 92 . 6 | 91 . 6 |
| Faster R-CNN resnet101    | 90.5      | 89 . 0    | 92.2   | 92.2   |
| Faster R-CNN resnet50     | 89 . 6    | 89.8      | 90 . 8 | 91 . 9 |

## Network Training

Fig. 7. 3D target migration map compared with the original 3D target model. (a), (c) are the 3D model of synthetic slabs while (b), (d) show the proposed 3D target migration results.

<!-- image -->

We trained DepthNet on a RTX 2080Ti GPU server and used Pytorch to deploy the algorithm. For the DepthNet, we used the stochastic gradient decent (SGD), with an initial learning rate at 4 e -5, momentum as 0 . 9, and weight decay of 5 e -5. The model loss at last converged to 0 . 0524

## Accuracy Evaluation

We perform the dielectric prediction validation using the test data from the GPR dataset we created in Section.IIB. For testing purpose, we use total of 50 B-scan images which have the different dielectric. The accuracy of the model is tested from two different aspects: 1) the individual accuracy for B-scan feature dielectric prediction; 2) the average accuracy of the depth prediction which is 0 . 112.

## Visualization for Proposed Method

To visualize the results of the GPR based feature detection, which used as noise cancellation for GPR migration, this paper first shows the ground truth of the two different synthetic slabs created by gprMax. Then, in each B-scan raw data, we overlays the detected bounding boxes on it. Besides, the predicted dielectric also be registered in migration as the pre-processing for 3D subsurface targets reconstruction. Then, we compared the original migration result and noise cancelled migration results in front view, since we could also compare these results with the bounding boxes overlapped B-scan data. At last, the cancelled noise images are also attached in order to validate our proposed method. In Fig.5 (a), (f) shows the ground truth of the two synthetic slabs generated by gprMax. This two slabs are embedded with conductive and dielectric rebar, steel pipe, another rebar and PVC pipe with different size from left side to right. Then (b), (c) and (g), (h) compares the migration results before implementing B-scan feature detection and after detection mentioned in II-D. At last, (e), (j) show the noise we cancelled with proposed method.

## C. 3D Object Migration Map

This section uses the results from the previous two sections to generate the 3D object migration map. First, our proposed migration method provide the ability for real 3D subsurface targets reconstruction. Then, by detecting the B-scan features, the noise in B-scan raw data could be removed. Moreover, our proposed DepthNet could provide the depth information based on the input B-scan raw data. By combining these three methods, Fig.7 shows the final comparison results of the 3D targets reconstruction.

## IV. CONCLUSIONS

This paper introduces an DNN based 3D GPR imaging system, which is able to locate and visualize the subsurface objects. First, this system implements visual inertial fusion to estimate the pose of the GPR sensor. Then, we propose an improved random motion migration method which eliminates the limitation of current GPR data collection procedure which requires the straight line motion along survey grid. After that, DNN based target detection is employed, by only processing the B-scan data in detected bounding boxes, background noise in raw B-scan image could be removed. Finally, the proposed DepthNet is used to predict the depth of subsurface objects, according to the estimation of dielectric characteristic of the material. The experiments show the effectiveness of our proposed 3D subsurface objects reconstruction methodology.

## V. ACKNOWLEDGEMENT

Research was supported in part by NSF grant number IIP1915721. J Xiao has significant financial interest in InnovBot LLC, a company involved in R&amp;D and commercialization of the technology.

## REFERENCES

- [1] J. Hugenschmidt and R. Mastrangelo, 'Gpr inspection of concrete bridges,' Cement and Concrete Composites , vol. 28, no. 4, pp. 384392, 2006.
- [2] J. Les Davis, J. R. Rossiter, E. Darel, and C. B. Dawley, 'Quantitative measurement of pavement structures using radar,' in Fifth International Conferention on Ground Penetrating Radar , 1994.
- [3] D. J. Daniels, 'Surface-penetrating radar,' Electronics &amp; Communication Engineering Journal , vol. 8, no. 4, pp. 165-182, 1996.
- [4] A. Benedetto, F. Benedetto, and F. Tosti, 'Gpr applications for geotechnical stability of transportation infrastructures,' Nondestructive Testing and Evaluation , vol. 27, no. 3, pp. 253-262, 2012.
- [5] A. Benedetto, G. Manacorda, A. Simi, and F. Tosti, 'Novel perspectives in bridges inspection using gpr,' Nondestructive Testing and Evaluation , vol. 27, no. 3, pp. 239-251, 2012.
- [6] N. Blindow, S. K. Suckro, M. R¨ uckamp, M. Braun, M. Schindler, B. Breuer, H. Saurer, J. C. Sim˜ oes, and M. A. Lange, 'Geometry and thermal regime of the king george island ice cap, antarctica, from gpr and gps,' Annals of Glaciology , vol. 51, no. 55, pp. 103-109, 2010.
- [7] S. Urbini, L. Vittuari, S. Gandolfi, et al. , 'Gpr and gps data integration: examples of application in antarctica,' 2001.
- [8] R. E. Yoder, R. S. Freeland, J. T. Ammons, and L. L. Leonard, 'Mapping agricultural fields with gpr and emi to identify offsite movement of agrochemicals,' Journal of Applied Geophysics , vol. 47, no. 3-4, pp. 251-259, 2001.
- [9] V. Ferrara, A. Pietrelli, S. Chicarella, and L. Pajewski, 'Gpr/gps/imu system as buried objects locator,' Measurement , vol. 114, pp. 534-541, 2018.
- [10] J. H. Bradford, 'Gpr prestack amplitude recovery for radiation patterns using a full wave-equation, reverse-time migration algorithm,' in SEG Technical Program Expanded Abstracts 2012 . Society of Exploration Geophysicists, 2012, pp. 1-5.
- [11] C. J. Leuschen and R. G. Plumb, 'A matched-filter-based reversetime migration algorithm for ground-penetrating radar data,' IEEE Transactions on Geoscience and Remote Sensing , vol. 39, no. 5, pp. 929-936, 2001.
- [12] X. Feng and M. Sato, 'Pre-stack migration applied to gpr for landmine detection,' Inverse problems , vol. 20, no. 6, p. S99, 2004.
- [13] D. Ristow and T. R¨ uhl, 'Fourier finite-difference migration,' Geophysics , vol. 59, no. 12, pp. 1882-1893, 1994.
- [14] M. Grasmueck, R. Weger, and H. Horstmeyer, 'Full-resolution 3d gpr imaging,' Geophysics , vol. 70, no. 1, pp. K12-K19, 2005.
- [15] L. Qu and T. Yang, 'Investigation of air/ground reflection and antenna beamwidth for compressive sensing sfcw gpr migration imaging,' IEEE transactions on geoscience and remote sensing , vol. 50, no. 8, pp. 3143-3149, 2012.
- [16] W. Al-Nuaimy, Y. Huang, M. Nakhkash, M. Fang, V. Nguyen, and A. Eriksen, 'Automatic detection of buried utilities and solid objects with gpr using neural networks and pattern recognition,' Journal of applied Geophysics , vol. 43, no. 2-4, pp. 157-165, 2000.
- [17] Y.-a. Cui, L. Wang, J.-p. Xiao, et al. , 'Automatic feature recognition for gpr image processing,' World Academy of Science, Engineering and Technology , vol. 61, no. 1, pp. 176-179, 2010.
- [18] S. Lameri, F. Lombardi, P. Bestagini, M. Lualdi, and S. Tubaro, 'Landmine detection from gpr data using convolutional neural networks,' in 2017 25th European Signal Processing Conference (EUSIPCO) . IEEE, 2017, pp. 508-512.
- [19] L. E. Besaw and P. J. Stimac, 'Deep convolutional neural networks for classifying gpr b-scans,' in Detection and Sensing of Mines, Explosive Objects, and Obscured Targets XX , vol. 9454. International Society for Optics and Photonics, 2015, p. 945413.
- [20] X. Xu, Y. Lei, and F. Yang, 'Railway subgrade defect automatic recognition method based on improved faster r-cnn,' Scientific Programming , vol. 2018, 2018.
- [21] M.-T. Pham and S. Lef` evre, 'Buried object detection from b-scan ground penetrating radar data using faster-rcnn,' in IGARSS 20182018 IEEE International Geoscience and Remote Sensing Symposium . IEEE, 2018, pp. 6804-6807.
- [22] R. Girshick, 'Fast r-cnn,' in Proceedings of the IEEE international conference on computer vision , 2015, pp. 1440-1448.
- [23] L. Armesto, J. Tornero, and M. Vincze, 'Fast ego-motion estimation with multi-rate fusion of inertial and vision,' The International Journal of Robotics Research , vol. 26, no. 6, pp. 577-589, 2007.
- [24] A. I. Mourikis and S. I. Roumeliotis, 'A multi-state constraint kalman filter for vision-aided inertial navigation,' in Proceedings 2007 IEEE International Conference on Robotics and Automation . IEEE, 2007, pp. 3565-3572.
- [25] G. N¨ utzi, S. Weiss, D. Scaramuzza, and R. Siegwart, 'Fusion of imu and vision for absolute scale estimation in monocular slam,' Journal of intelligent &amp; robotic systems , vol. 61, no. 1-4, pp. 287-299, 2011.
- [26] S. Demirci, E. Yigit, I. H. Eskidemir, and C. Ozdemir, 'Ground penetrating radar imaging of water leaks from buried pipes based on back-projection method,' Ndt &amp; E International , vol. 47, pp. 35-42, 2012.
- [27] L. Zhou, C. Huang, and Y. Su, 'A fast back-projection algorithm based on cross correlation for gpr imaging,' IEEE Geoscience and Remote Sensing Letters , vol. 9, no. 2, pp. 228-232, 2011.
- [28] H. M. Jol, Ground penetrating radar theory and applications . elsevier, 2008.
- [29] A. Giannopoulos, 'Modelling ground penetrating radar by gprmax,' Construction and building materials , vol. 19, no. 10, pp. 755-762, 2005.
- [30] C. Szegedy, S. Ioffe, V. Vanhoucke, and A. A. Alemi, 'Inception-v4, inception-resnet and the impact of residual connections on learning,' in Thirty-First AAAI Conference on Artificial Intelligence , 2017.
- [31] J. Huang, V. Rathod, C. Sun, M. Zhu, A. Korattikara, A. Fathi, I. Fischer, Z. Wojna, Y. Song, S. Guadarrama, et al. , 'Speed/accuracy trade-offs for modern convolutional object detectors,' in Proceedings of the IEEE conference on computer vision and pattern recognition , 2017, pp. 7310-7311.
- [32] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg, 'Ssd: Single shot multibox detector,' in European conference on computer vision . Springer, 2016, pp. 21-37.
- [33] J. Redmon and A. Farhadi, 'Yolov3: An incremental improvement,' arXiv preprint arXiv:1804.02767 , 2018.