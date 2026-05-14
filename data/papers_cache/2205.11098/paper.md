## PointDistiller: Structured Knowledge Distillation Towards Efficient and Compact 3D Detection

Linfeng Zhang

13 ∗

Runpei Dong 2

Tsinghua University 1

∗

Hung-Shuo Tai 3

Xi'an Jiaotong University 2

## Abstract

The remarkable breakthroughs in point cloud representation learning have boosted their usage in real-world applications such as self-driving cars and virtual reality. However, these applications usually have an urgent requirement for not only accurate but also efficient 3D object detection. Recently, knowledge distillation has been proposed as an effective model compression technique, which transfers the knowledge from an over-parameterized teacher to a lightweight student and achieves consistent effectiveness in 2D vision. However, due to point clouds' sparsity and irregularity, directly applying previous image-based knowledge distillation methods to point cloud detectors usually leads to unsatisfactory performance. To fill the gap, this paper proposes PointDistiller, a structured knowledge distillation framework for point clouds-based 3D detection. Concretely, PointDistiller includes local distillation which extracts and distills the local geometric structure of point clouds with dynamic graph convolution and reweighted learning strategy, which highlights student learning on the crucial points or voxels to improve knowledge distillation efficiency. Extensive experiments on both voxels-based and raw pointsbased detectors have demonstrated the effectiveness of our method over seven previous knowledge distillation methods. For instance, our 4 × compressed PointPillars student achieves 2.8 and 3.4 mAP improvements on BEV and 3D object detection, outperforming its teacher by 0.9 and 1.8 mAP, respectively. Codes have been released at https://github.com/RunpeiDong/PointDistiller .

## 1 Introduction

The growth in large-scale lidar datasets [13] and the achievements in end-to-end 3D representation learning [47, 48] have boosted the developments of point cloud based segmentation, generation, and detection [25, 49]. As one of the essential tasks of 3D computer vision, 3D object detection plays a fundamental role in real-world applications such as autonomous driving cars [3, 5, 13] and virtual reality [44]. However, recent research has shown a growing discrepancy between cumbersome 3D detectors that achieve state-of-the-art performance and lightweight 3D detectors which are affordable in real-time applications on edge devices. To address this problem, sufficient model compression techniques have been proposed, such as network pruning [18, 36, 38, 71], quantization [7, 11, 41], lightweight model design [21, 39, 52], and knowledge distillation [20].

Knowledge distillation, which aims to improve the performance of a lightweight student model by training it to mimic a pre-trained and over-parameterized teacher model, has evolved into one of the most popular and effective model compression methods in both computer vision and natural language processing [20, 51, 53, 65]. Sufficient theoretical and empirical results have demonstrated its effectiveness in image-based visual tasks such as image classification [20, 51], semantic segmentation [34] and object detection [1, 4, 28, 68]. However, compared with images, point clouds have their

∗ The first two authors have equal contributions. This work is done during internship of L. Zhang in DIDI.

† Corresponding author.

Kaisheng Ma 1 3

†

DIDI

Figure 1: Results (mAP of moderate difficulty) of our methods on 4 × , 8 × , and 16 × compressed students on KITTI. The area of dash lines indicates the benefits of knowledge distillation.

<!-- image -->

Figure 2: Distribution of the voxels with different number of points inside them.

properties: (i) Point clouds inherently lack topological information, which makes recovering the local topology information crucial for the visual tasks [26, 40, 64]. (ii) Different from images that have a regular structure, point clouds are irregularly and sparsely distributed in the metric space [12, 14].

These differences between images and point clouds have hindered the image-based knowledge distillation methods from achieving satisfactory performance on point clouds and also raised the requirement to design specific knowledge distillation methods for point clouds. Recently, a few methods have been proposed to apply knowledge distillation to 3D detection [16, 17, 54]. However, most of these methods focus on the choice of student-teacher in a multi-modal setting, e.g. , teaching point clouds-based student detectors with an images-based teacher or vice versa, and still ignore the peculiar properties of point clouds. To address this problem, we propose a structured knowledge distillation framework named PointDistiller, which involves local distillation to distill teacher knowledge in the local geometric structure of point clouds, and reweighted learning strategy to handle the sparsity of point clouds by highlighting student learning on the relatively more crucial voxels.

Local Distillation Sufficient recent studies show that capturing and making usage of the semantic information in the local geometric structure of point clouds have a crucial impact on point cloud representation learning [48, 63]. Hence, instead of directly distilling the backbone feature of teacher detectors to student detectors, we propose local distillation, which firstly clusters the local neighboring voxels or points with KNN (K-Nearest Neighbours), then encodes the semantic information in local geometric structure with dynamic graph convolutional layers [63], and finally distill them from teachers to students. Hence, the student detectors can inherit the teacher's ability to understand point clouds' local geometric information and achieve better detection performance.

Reweighted Learning Strategy One of the mainstream methods for processing point clouds is to convert them into volumetric voxels and then encode them as regular data. However, due to the sparsity and the noise in point clouds, most of these voxels contain only a single point. For instance, as shown in Figure 2, on the KITTI dataset, around 68% voxels in point clouds contain only one point, which has a high probability of being a noise point. Hence, the representative features in these single-point voxels have relatively lower importance in knowledge distillation compared with the voxels which contain multiple points. Motivated by this observation, we propose a reweighted learning strategy, which highlights student learning on the voxels with multiple points by giving them larger learning weights. Besides, the similar idea can also be easily extended to raw points-based detectors to highlight knowledge distillation on the points which have a more considerable influence on the prediction of the teacher detector.

Extensive experiments on both voxels-based and raw-points based detectors have been conducted to demonstrate the effectiveness of our method over the previous seven knowledge distillation methods. As shown in Figure 1, on PointPillars and SECOND detectors, our method leads to 4 × compression and 0.9 ∼ 1.8 mAP improvements at the same time. On PointRCNN, our method leads to 8 × compression with only 0.2 BEV mAP drop. Our main contributions be summarized as follows.

- We propose local distillation , which firstly encodes the local geometric structure of point clouds with dynamic graph convolution and then distills them from teachers to students.
- We propose reweighted learning strategy to handle the sparsity and noise in point clouds. It highlights student learning on the voxels, which have more points inside them, by giving them higher learning weights in knowledge distillation.
- Extensive experiments on both voxels-based and raw points-based detectors have been conducted to demonstrate the performance of our method over seven previous methods. Besides, we have released our codes to promote future research.

## 2 Related Work

## 2.1 Knowledge Distillation

The idea of training a small model with a large pre-trained model is firstly proposed by Buciluˇ a et al. for ensemble model compression [2]. Then, with the excellent breakthroughs of deep learning, Hinton et al. propose the concept of knowledge distillation which strives to compress an overparameterized teacher model by transferring its knowledge to a lightweight student model [20]. Early knowledge distillation methods usually train the students to mimic the predicted categorical probability distribution of teachers [20, 72]. Then, extensive methods have been proposed to learn teacher knowledge in the backbone features [51] or its variants, such as attention [67, 68], relation [31, 43, 46, 61], task-oriented information [69] and so on. Following its success in classification, abundant works have applied knowledge distillation to object detection [4, 28, 62, 68], segmentation [34], image generation [22, 27, 29, 31, 50, 70], pre-trained language models [53, 65], semi-supervised learning [24, 59] and lead to consistent effectiveness.

Knowledge Distillation on Object Detection Recently, designing specific knowledge distillation methods to improve the efficiency and accuracy of object detection has become a rising and popular topic. Chen et al. fi rst propose to apply the naive prediction and feature-based knowledge distillation methods to object detection [4]. Then, Wang et al. show that the imbalance between foreground objects and background objects hinders knowledge distillation from achieving better performance in object detection [62]. To address this problem, abundant knowledge distillation methods have tried to find the to-be-distilled regions based on the ground-truth [62], detection results [10], spatial attention [68], query-based attention [23] and gradients [15]. Moreover, recent methods have also been proposed to distill the pixel-level and object-level relation from teachers to students [10, 35, 68]. Besides knowledge distillation for 2D detection, some cross-modal knowledge distillation have been introduced to transfer knowledge from RGB-based teacher detectors to lidar-based student detectors or vice versa [8, 16, 17, 54]. However, most of these methods focus on the choice of students and teachers in a multi-modal framework, while the design of specific knowledge distillation optimization methods on point clouds based pure 3D detection has not been well-explored.

## 2.2 3D Object Detection on Point Clouds

The rapid development of deep learning has firstly boosted the research in 2D object detection and then recently raised the research trend in point clouds-based 3D object detection. PointNet [47] is firstly proposed to extract the feature of points with multi-layer perception in an end-to-end manner. Then, PointNet++ is further proposed to capture the local structures in a hierarchical fashion with density adaptive sampling and grouping [48]. Zhou et al. propose VoxelNet, a single-stage detector that divides a point cloud into equally spaced 3D voxels and processes them with voxel feature encoding layers [75]. Then, SECOND is proposed to improve VoxelNet with sparse convolutional layers and focal loss [66]. PointPillars is proposed to divide point clouds into several pillars and then convert them into a pseudo image, which can be further processed with 2D convolutional layers [25, 42]. Shi et al. propose PointRCNN, a two-stage detection method that firstly generates bottom-up 3D proposals based on the raw point clouds and then refines them to obtain the final detection results [55]. Afterward, Fast Point R-CNN and PV-RCNN are proposed to utilize both voxel representation and raw point clouds to exploit their respective advantages [6, 56]. Recently, Qi et al. propose to perform offboard 3D detection with point cloud sequences, which is able to make use of the temporal points and achieve comparable performance with human labels [49]. The graph convolutional neural network is another rising star in point cloud detection [57, 63]. Lin et al. propose 3D-GCN to avoid the shift and scale changes in point clouds [33]. Zhou et al. propose adaptive graph convolution, which generates adaptive kernels according to the learned features [74].

Efficient 3D Object Detectors Unfortunately, the significant 3D detection performance usually comes at the expense of high computational and storage costs, making them unaffordable in real-time applications such as self-driving cars. To address this issue, recent research attention has been paid to designing efficient 3D detectors. Tang et al. propose to apply neural architecture search to 3D detection by using sparse point-voxel convolution [58]. Li et al. propose Lidar-RCNN, which resorts to a point-based approach and remedies the problem of uncorrected proposal sizes [32]. Liu et al. propose voxel-point cnn to represent the 3D input data in points while performing the convolutions in voxels to reduce the memory accessing consumption [37]. Recently, Li et al. propose to improve the efficiency of graph convolution for point clouds by simplified KNN search and graph shuffling [30].

Figure 3: The computation of the importance score for voxels-based detectors and raw-points-based detectors. The importance scores are later utilized to determine which voxel or point is utilized for distillation and how they contribute to the distillation loss.

<!-- image -->

Figure 4: The details of our method. f T and f S : the feature encoding layers in the teacher and student detectors. A T and A S : features of the sampled to-be-distilled voxels or points with topN largest importance score. C T and C S : the number of channels for features of the teacher and the student detectors. G T and G S : the graph features of the teacher and student detectors. Based on the pre-defined importance score, our method samples the relatively more crucial N voxels or points from the whole point cloud, extracts their local geometric structure of them with dynamic graph convolution, and then distills them in a reweighted manner. Please refer to Section 3 for more details.

<!-- image -->

## 3 Methodology

## 3.1 Preliminaries

Given a set of point clouds X = { x 1 , x 2 , ..., x n } and the corresponding label set Y = { y 1 , y 2 , ..., y m } , the object detector can be formulated as F = f ◦ g , where f is the feature encoding layer to extract representation features from inputs and g is the detection head for prediction. Then, the representation feature on the sample x can be written as f ( x ) ∈ R n × C , where n indicates the number of voxels for voxels-based detectors or the number of points for raw points-based detectors. C indicates the number of channels. Besides, for voxels-based detectors, we define v ij ( x ) = 1 if the j -th point of x belongs to the i -voxel else 0. Then, the number of points in the i -th voxel can be denoted as ∑ j v ij ( x ) . Usually, knowledge distillation involves a to-be-trained student detector and a pre-trained teacher detector, and we distinguish them with scripts S and T , respectively.

## 3.2 Our Method

Sampling TopN To-be-distilled Voxels (Points) As discussed in previous sections, since the point clouds are overwhelmingly sparse while the voxels are usually equally spaced, most of the voxels only contain very few and even single point. Thus, these single-point voxels have much less value to be learned by students in knowledge distillation. Even in raw points-based detectors, there usually exist some points which are relatively more crucial and some points which are not meaningful ( e.g. , the noise points). Thus, instead of distilling all the voxels or points in point clouds, we propose to distill the voxels or points which are more valuable for knowledge distillation.

Concretely, for voxels-based detectors, we define the importance score of i -th voxel as ∑ j v ij ( x ) , which indicates the number of points inside it. For point-based detectors, motivated by previous works which localized the crucial pixels in images with attention, we define the importance score for i -th point as its permutation-invariant maximal value along the channel dimension, which can be formulated as max ( f ( x )[ i ]) . Based on the importance score, we can sample the topN significant voxels or points for knowledge distillation based on the importance score computed from f T ( x ) . For simplicity in writing, we denote the selected student and teacher features in topN important voxels or points as A T ( x ) ∈ R N × C T and A S ( x ) ∈ R N × C S , respectively, where C S and C T indicate the number of channels in student and teacher features.

Extracting Local Geometric Information As pointed out by abundant previous works, the local geometric information has a crucial influence on the performance of point cloud detectors [48, 63]. Thus, instead of directly distilling the representative feature, we propose local distillation which extracts the local geometric information of point clouds with dynamic graph convolution layers and distills it to the student detector. Concretely, denoting z i = A ( x )[ i ] as the feature of the i -th to-be-distilled voxel or point, we can build a graph based on this voxel or point and its K neighboring voxels or points clustered by KNN (K-Nearest Neighbours). By denoting the features of z i and its K -1 neighbours as z i, 1 and N i = { z i, 2 , z i, 3 , ..., z i,K } respectively, motivated by previous methods [47, 63], we firstly update the feature of each voxel (or point) in this graph by concatenating them with the global centroid voxel (or point) feature z i, 1 , which can be formulated as ˆ z i,j = cat ( [ z i, 1 , z i,j ] ) for all z i,j ∈ N i . Then, we apply a dynamic graph convolution as the aggregation operation upon them, which can be formulated as G i = γ (ˆ z i, 1 , ..., ˆ z i,K ) , where γ is the aggregation operator. Following previous graph-based point cloud networks, we set γ as a nonlinear layer with ReLU activation and batch normalization. Then the training objective of local distillation can be formulated as

<!-- formula-not-decoded -->

where θ S indicates the parameters of student encoding layer f S . θ γ = [ θ γ S , θ γ T ] indicates the parameters of dynamic graph convolution layers for the student and teacher detectors. Note that these layers are trained with the student detector simultaneously and can be discarded during inference.

Reweighting Knowledge Distillation Loss Usually, compared with the teacher detector, the student detector has much fewer parameters, implying inferior learning capacity. Thus, it is challenging for the student detector to inherit teacher knowledge in all points or voxels. As discussed above, different voxels and points in point cloud object detection have different values in knowledge distillation. Thus, we propose to reweight the learning weight of each voxel or point based on the importance score introduced in previous paragraphs. Denote the learning weight for the N to-be-distilled as φ ∈ R N . Similar with the importance score defined during sampling, we define the learning weight of each graph as the maximal value on the corresponding features after a softmax function, which can be formulated as φ = softmax ( max ( G T ( x )) /τ ) , where τ is the temperature hyper-parameter in softmax function. For voxels-based methods, we define φ as the number of points in the voxel after a softmax function, which can be formulated as φ i = softmax ( ∑ j v i,j /τ ) . Then, with the reweighting strategy, the training objective of knowledge distillation can be formulated as

<!-- formula-not-decoded -->

As shown in the above loss function, with a higher φ i , the knowledge distillation loss between student and teacher features at the i -th graph will have a more extensive influence on the overall loss, and thus student learning on the i -th graph can be highlighted. As a result, the proposed reweighting strategy allows the student detector to pay more attention to learning teacher knowledge in the relatively more crucial voxel graphs (point graphs). Moreover, Equation 2 also implies that our method is a feature-based knowledge distillation method that is not correlated with the architecture of detectors and the label set Y . Hence, it can be directly added to the origin training loss of all kinds of 3D object detectors for model compression.

Table 1: Experimental results of our method for BEV (Bird-Eye-View) object detection. F and P indicate the number of float operations (/G) and parameters (/M) of the detector, respectively. mAP indicates the mean average precision of moderate difficulty. KD indicates whether our method is utilized. The reported result in the first line of each detector is the performance of the teacher detector.

| Model   | F     | P   | KD           | Car   | Car      | Car   | Pedestrians   | Pedestrians   | Pedestrians   | Cyclists   | Cyclists   | Cyclists   | mAP   |
|---------|-------|-----|--------------|-------|----------|-------|---------------|---------------|---------------|------------|------------|------------|-------|
| Model   | F     | P   | KD           | Easy  | Moderate | Hard  | Easy          | Moderate      | Hard          | Easy       | Moderate   | Hard       | mAP   |
|         | 34.3  | 4.8 | ×            | 94.3  | 88.1     | 83.6  | 57.9          | 51.8          | 47.6          | 86.5       | 65.0       | 61.1       | 68.3  |
|         | 9.0   | 1.3 | ×            | 92.4  | 88.2     | 83.6  | 53.0          | 47.9          | 44.1          | 81.8       | 63.1       | 59.0       | 66.4  |
|         | 9.0   | 1.3 | glyph[check] | 93.1  | 89.0     | 86.3  | 59.8          | 52.8          | 48.2          | 83.8       | 65.8       | 62.0       | 69.2  |
|         | 2.5   | 0.3 | ×            | 91.3  | 84.8     | 82.2  | 50.1          | 44.4          | 41.6          | 74.2       | 56.1       | 52.5       | 61.8  |
|         | 2.5   | 0.3 | glyph[check] | 92.5  | 85.2     | 81.9  | 50.8          | 45.8          | 42.5          | 77.2       | 59.5       | 55.6       | 63.5  |
|         | 69.8  | 5.3 | ×            | 93.1  | 88.9     | 85.9  | 64.9          | 58.1          | 51.9          | 84.3       | 69.9       | 65.7       | 72.3  |
|         | 17.8  | 1.4 | ×            | 93.1  | 86.6     | 85.7  | 64.7          | 57.8          | 52.8          | 84.1       | 68.5       | 64.5       | 71.0  |
|         | 17.8  | 1.4 | glyph[check] | 93.2  | 88.6     | 86.0  | 65.1          | 58.1          | 53.1          | 87.4       | 72.9       | 68.5       | 73.2  |
|         | 4.6   | 0.4 | ×            | 95.0  | 86.2     | 83.3  | 61.6          | 54.9          | 49.2          | 80.9       | 63.6       | 59.6       | 68.3  |
|         | 4.6   | 0.4 | glyph[check] | 95.4  | 88.3     | 83.7  | 64.5          | 57.6          | 52.2          | 85.2       | 68.8       | 64.4       | 71.6  |
|         | 104.9 | 4.1 | ×            | 95.0  | 86.7     | 84.3  | 69.8          | 64.5          | 58.1          | 92.8       | 74.6       | 70.4       | 75.3  |
|         | 13.7  | 0.5 | ×            | 93.5  | 85.9     | 83.5  | 71.6          | 65.4          | 59.1          | 91.1       | 71.0       | 67.2       | 74.1  |
|         | 13.7  | 0.5 | glyph[check] | 93.3  | 85.7     | 83.5  | 74.0          | 67.2          | 60.5          | 94.6       | 72.3       | 67.9       | 75.1  |
|         | 7.1   | 0.3 | ×            | 95.8  | 85.4     | 81.7  | 72.9          | 65.5          | 58.6          | 91.8       | 69.3       | 65.9       | 73.4  |
|         | 7.1   | 0.3 | glyph[check] | 95.2  | 84.3     | 81.7  | 72.6          | 64.8          | 57.7          | 92.6       | 72.9       | 68.5       | 74.0  |

## 4 Experiment

## 4.1 Experiment Setting

We have evaluated our method in both voxels-based object detector including PointPillars [25] and SECOND [66], and the raw points based object detector including PointRCNN [55]. Most experiments are conducted on KITTI [13] and nuScenes [3], which consist of samples that have both lidar point clouds and images. Our models are trained with only the lidar point clouds. For KITTI, we report the average precision calculated by 40 sampling recall positions for BEV (Bird's Eye View) object detection and 3D object detection on the validation split. Following the typical protocol, the IoU threshold is set as 0.7 for class Car and 0.5 for class Pedestrians and Cyclists. We have mainly compared our methods with seven previous knowledge distillation methods, including methods proposed by Remero et al. [51], Zagoruko et al. [67], Tung et al. [31], Heo et al. [19], Zheng et al. [73], Tian et al. [60], and Zhang et al. [68]. All the experiments are conducted with mmdetection3d [9] and PyTorch [45]. We keep the training and evaluation settings in mmdetection3d as default. The teacher model is the origin model before compression. The student model shares the same architecture and depth as its teacher but with fewer channels. Following previous works, the average precision of three difficulties and the three categories are reported as the performance metrics [13]. Please refer to our codes in the supplementary material for more details.

## 4.2 Experimental Results

Table 1 and Table 2 show the performance of detectors trained with and without our method for BEV detection and 3D detection, respectively. It is observed that: (i) Significant average precision improvements on all kinds of detectors and all compression ratios for both BEV and 3D detection. On average, 2.4 and 1.0 moderate mAP improvements can be observed for the voxel and raw pointsbased detectors, respectively. On BEV and 3D detection, 1.9 and 1.9 moderate mAP improvements can be obtained, respectively. (ii) On the BEV detection of PointPillars and SECOND detectors, the 4 × compressed and accelerated students trained with our method outperform their teachers by 0.9 and 0.9 mAP, respectively. On the 3D detection of PointPillars and SECOND detectors, the 4 × compressed and accelerated students trained with our method outperform their teachers by 1.8 and 0.1 mAP, respectively. (iii) Consistent average precision boosts can be observed in detection results of all difficulties. For instance, on BEV detection of PointPillars students, 2.4, 2.3, and 2.3 mAP improvements can be observed for easy, moderate, and hard difficulties, respectively. These observations demonstrate that our method can successfully transfer teacher knowledge to the student detectors. (iv) Consistent average precision boosts can be observed in detection results of all categories. For instance, on moderate BEV detection of PointPillars students, 0.6, 3.2 and 3.1 mAP improvements can be obtained on cars, pedestrians and cyclists, respectively. (v) On PointRCNN, on average 1.3 and 1.2 moderate mAP improvements can be observed on BEV and 3D detection, respectively, indicating that our method is also effective for raw points-based detectors. In summary, these experiment results demonstrate that our method can successfully transfer the knowledge from teacher detectors to student detectors and lead to significant and consistent performance boosts.

Table 2: Experimental results of our method for 3D object detection. F and P indicate the number of float operations (/G) and parameters (/M) of the detector, respectively. mAP indicates the mean average precision of moderate difficulty. KD indicates whether our method is utilized. The reported result in the first line of each detector is the performance of the teacher detector.

| Model   | F     | P   | KD           | Car   | Car      | Car   | Pedestrians   | Pedestrians   | Pedestrians   | Cyclists   | Cyclists   | Cyclists   | mAP   |
|---------|-------|-----|--------------|-------|----------|-------|---------------|---------------|---------------|------------|------------|------------|-------|
| Model   | F     | P   | KD           | Easy  | Moderate | Hard  | Easy          | Moderate      | Hard          | Easy       | Moderate   | Hard       | mAP   |
|         | 34.3  | 4.8 | ×            | 87.3  | 75.9     | 71.1  | 52.0          | 45.9          | 41.4          | 78.6       | 59.2       | 55.8       | 60.3  |
|         | 9.0   | 1.3 | ×            | 87.4  | 75.9     | 71.0  | 48.2          | 43.0          | 38.7          | 74.1       | 57.2       | 53.3       | 58.7  |
|         | 9.0   | 1.3 | glyph[check] | 88.1  | 76.9     | 73.8  | 54.6          | 47.5          | 42.3          | 80.3       | 62.0       | 58.8       | 62.1  |
|         | 2.5   | 0.3 | ×            | 83.1  | 69.8     | 65.4  | 44.0          | 38.7          | 35.3          | 70.9       | 52.1       | 48.7       | 53.5  |
|         | 2.5   | 0.3 | glyph[check] | 83.7  | 69.8     | 65.3  | 45.3          | 40.3          | 36.5          | 72.7       | 54.7       | 51.1       | 54.9  |
|         | 69.8  | 5.3 | ×            | 88.6  | 79.3     | 75.7  | 60.1          | 53.2          | 47.0          | 79.8       | 65.7       | 61.6       | 66.1  |
|         | 17.8  | 1.4 | ×            | 89.2  | 77.4     | 74.0  | 58.8          | 51.3          | 45.5          | 80.5       | 65.4       | 61.3       | 64.7  |
|         | 17.8  | 1.4 | glyph[check] | 88.9  | 76.9     | 73.6  | 60.0          | 53.0          | 47.4          | 83.2       | 68.6       | 64.2       | 66.2  |
|         | 4.6   | 0.4 | ×            | 86.3  | 72.6     | 66.0  | 53.6          | 47.8          | 41.8          | 76.7       | 58.7       | 55.1       | 59.7  |
|         | 4.6   | 0.4 | glyph[check] | 87.0  | 73.3     | 68.1  | 57.0          | 51.0          | 45.4          | 81.0       | 63.5       | 59.3       | 62.6  |
|         | 104.9 | 4.1 | ×            | 92.1  | 80.1     | 77.4  | 66.8          | 60.3          | 54.3          | 92.1       | 72.3       | 67.8       | 70.9  |
|         | 13.7  | 0.5 | ×            | 89.8  | 76.8     | 72.7  | 67.9          | 60.9          | 54.0          | 88.1       | 68.0       | 64.4       | 68.6  |
|         | 13.7  | 0.5 | glyph[check] | 91.4  | 75.6     | 72.9  | 70.1          | 63.5          | 56.1          | 92.0       | 69.8       | 65.4       | 69.6  |
|         | 7.1   | 0.3 | ×            | 89.8  | 75.3     | 70.7  | 68.7          | 60.7          | 53.4          | 91.1       | 67.2       | 63.9       | 67.7  |
|         | 7.1   | 0.3 | glyph[check] | 89.6  | 75.6     | 72.6  | 69.4          | 61.0          | 53.5          | 91.0       | 70.2       | 65.5       | 69.0  |

<!-- image -->

Figure 5: Hyper-parameters sensitivity study on KITTI with 4 × compressed PointPillars detctors. mAP is measured on the moderate difficulty.

Figure 6: Visualization on the importance scores for PointPillars. Red points indicate the voxels with high importance scores.

<!-- image -->

Comparison with Other KD Methods Comparison between our method and previous knowledge distillation methods is shown in Table 3. It is observed that: (i) Our method outperforms the previous methods by a clear margin. On BEV and 3D detection, our method outperforms the second-best knowledge distillation method by 1.5 and 1.9 moderate mAP, respectively. (ii) Our method achieves the best performance for all categories of all difficulties. (iii) Besides, our method is the only knowledge distillation method that enables the student detector to outperform its teacher detector.

Experiments on nuScenes Experiments of 2 × and 4 × compressed PointPillars on nuScenes are shown in Table 4. It is observed that our method leads to 0.65 and 0.5 improvements on mAP and NDS on average, respectively, indicating that our method is also effective on the large-scale dataset.

Table 3: Comparison between our method and previous knowledge distillation methods on PointPillars. The teacher and the student detectors have 34.3 and 9.0 GFLOPs, respectively. mAP indicates the mean average precision of moderate difficulty.

| Task   | Method                  | Car   | Car      | Car   | Pedestrians   | Pedestrians   | Pedestrians   | Cyclists   | Cyclists   | Cyclists   | mAP   |
|--------|-------------------------|-------|----------|-------|---------------|---------------|---------------|------------|------------|------------|-------|
| Task   | Method                  | Easy  | Moderate | Hard  | Easy          | Moderate      | Hard          | Easy       | Moderate   | Hard       | mAP   |
|        | Teacher w/o KD          | 94.3  | 88.1     | 83.6  | 57.9          | 51.8          | 47.6          | 86.5       | 65.0       | 61.1       | 68.3  |
|        | Student w/o KD          | 92.4  | 88.2     | 83.6  | 53.0          | 47.9          | 44.1          | 81.8       | 63.1       | 59.0       | 66.4  |
|        | + Romero et al. [51]    | 91.5  | 85.6     | 83.1  | 57.5          | 51.0          | 46.3          | 82.8       | 65.1       | 61.1       | 67.2  |
|        | + Zagoruyko et al. [67] | 92.6  | 88.0     | 83.6  | 56.7          | 50.9          | 47.3          | 81.4       | 64.4       | 60.5       | 67.7  |
| BEV    | + Zheng et al. [73]     | 92.7  | 87.9     | 83.2  | 57.7          | 51.0          | 46.8          | 78.1       | 61.8       | 57.9       | 66.9  |
|        | + Tung et al. [61]      | 92.8  | 88.0     | 83.3  | 54.5          | 48.7          | 45.2          | 84.2       | 64.3       | 60.7       | 67.0  |
|        | + Tian et al. [60]      | 92.7  | 87.8     | 83.2  | 56.6          | 50.4          | 46.8          | 80.3       | 61.9       | 57.9       | 66.7  |
|        | + Heo et al. [19]       | 92.6  | 87.9     | 83.5  | 57.6          | 51.0          | 46.8          | 78.1       | 61.8       | 57.8       | 66.9  |
|        | + Zhang et al. [68]     | 92.3  | 85.7     | 83.0  | 59.7          | 52.0          | 47.6          | 71.0       | 64.3       | 60.5       | 67.5  |
|        | + Ours                  | 93.1  | 89.0     | 86.3  | 59.8          | 52.8          | 48.2          | 83.8       | 65.8       | 62.0       | 69.2  |
|        | Teacher w/o KD          | 87.3  | 75.9     | 71.1  | 52.0          | 45.9          | 41.4          | 78.6       | 59.2       | 55.8       | 60.3  |
|        | Student w/o KD          | 87.4  | 75.9     | 71.0  | 48.2          | 43.0          | 38.7          | 74.1       | 57.2       | 53.3       | 58.7  |
|        | + Romero et al. [51]    | 84.9  | 73.4     | 70.6  | 50.9          | 44.2          | 39.3          | 75.9       | 58.5       | 54.6       | 58.7  |
|        | + Zagoruyko et al. [67] | 87.6  | 75.7     | 71.4  | 51.0          | 44.8          | 40.7          | 74.4       | 57.8       | 54.2       | 59.5  |
| 3D     | + Zheng et al. [73]     | 87.3  | 75.5     | 71.5  | 52.6          | 45.6          | 40.8          | 74.9       | 58.6       | 54.9       | 59.9  |
|        | + Tung et al. [61]      | 87.5  | 76.0     | 71.3  | 50.1          | 43.3          | 39.2          | 79.2       | 59.5       | 55.3       | 59.6  |
|        | + Tian et al. [60]      | 85.6  | 74.2     | 71.0  | 49.5          | 43.5          | 39.0          | 76.4       | 58.4       | 54.7       | 58.7  |
|        | + Heo et al. [19]       | 87.7  | 76.1     | 71.7  | 52.6          | 45.6          | 40.8          | 74.9       | 58.6       | 54.9       | 60.1  |
|        | + Zhang et al. [68]     | 87.5  | 75.8     | 71.6  | 53.4          | 45.8          | 40.9          | 76.1       | 59.0       | 55.2       | 60.2  |
|        | + Ours                  | 88.1  | 76.9     | 73.8  | 54.6          | 47.5          | 42.3          | 80.3       | 62.0       | 58.8       | 62.1  |

Table 4: Experimental results on nuScenes dataset with PointPillars. CR indicates the compression ratio. KD indicates whether knowledge distillation is utilized. A higher mAP and NDS, and a lower mATE, mASE, mAOE, mAVE and mAAE indicate better performance.

| Model        |   CR | KD           |   mAP( ↑ ) |   NDS( ↑ ) |   mATE( ↓ ) |   mASE( ↓ ) |   mAOE( ↓ ) |   mAVE( ↓ ) |   mAAE( ↓ ) |
|--------------|------|--------------|------------|------------|-------------|-------------|-------------|-------------|-------------|
| PointPillars |    2 | ×            |       36.0 |       50.5 |        44.8 |        28.3 |        51.2 |        32.9 |        18.0 |
| PointPillars |    2 | glyph[check] |       36.7 |       51.0 |        44.9 |        28.1 |        51.0 |        31.7 |        17.4 |
| PointPillars |    4 | ×            |       32.2 |       47.3 |        46.7 |        28.4 |        60.1 |        35.9 |        17.2 |
| PointPillars |    4 | glyph[check] |       32.8 |       48.6 |        45.2 |        28.4 |        52.3 |        35.1 |        17.3 |

## 5 Discussion

## 5.1 Ablation Study and Sensitivity Study

Ablation Study The proposed PointDistiller is mainly composed of two components, including the reweighted learning strategy ( RL ) and local distillation ( LD ). Ablation studies with 4 × compressed PointPillars students on KITTI are shown in Table 5. It is observed that: (i) 2.0 and 1.9 mAP improvements can be obtained by only using the reweighted learning strategy to distill the backbone features on BEV detection and 3D detection, respectively. (ii) 2.3 and 2.5 mAP boosts can be gained by using local distillation without reweighted learning on BEV detection and 3D detection, respectively. (iii) By combining the two methods together, 0.5 and 0.9 further mAP improvements can be achieved on BEV detection and 3D detection, respectively. These observations indicate that each module in PointDistiller has its individual effectiveness and their merits are orthogonal. Besides, they also implies that the proposed local distillation and reweighted learning may be combined with other knowledge distillation methods to achieve better performance.

Sensitivity Study Our method mainly introduces two hyper-parameters, K , and N , which indicate the number of nodes in a graph for local distillation, and the number of to-be-distilled voxels (points) respectively. A hyper-parameter sensitivity study on the two hyper-parameters is shown in Figure 5. It is observed that our method with different hyper-parameter values consistently outperforms the baseline by a large margin, indicating our method is not sensitive to hyper-parameters.

Table 5: Ablation study on 4 × compressed PointPillars students. LD and RL indicates local distillation and the reweighted learning strategy, respectively. mAP is measured on the moderate difficulty.

| Model   | Task   | LD           | RL           | Car   | Car      | Car   | Pedestrians   | Pedestrians   | Pedestrians   | Cyclists   | Cyclists   | Cyclists   | Cyclists   |
|---------|--------|--------------|--------------|-------|----------|-------|---------------|---------------|---------------|------------|------------|------------|------------|
| Model   | Task   | LD           | RL           | Easy  | Moderate | Hard  | Easy          | Moderate      | Hard          | Easy       | Moderate   | Hard       | mAP        |
|         |        | ×            | ×            | 92.4  | 88.2     | 83.6  | 53.0          | 47.9          | 44.1          | 81.8       | 63.1       | 59.0       | 66.4       |
|         |        | glyph[check] | ×            | 92.7  | 88.2     | 83.7  | 58.2          | 51.0          | 47.0          | 84.3       | 66.9       | 63.1       | 68.7       |
|         |        | ×            | glyph[check] | 93.1  | 88.5     | 85.7  | 55.6          | 49.6          | 45.7          | 84.2       | 67.3       | 62.9       | 68.4       |
|         |        | glyph[check] | glyph[check] | 93.1  | 89.0     | 86.3  | 59.8          | 52.8          | 48.2          | 83.8       | 65.8       | 62.0       | 69.2       |
|         |        | ×            | ×            | 87.4  | 75.9     | 71.0  | 48.2          | 43.0          | 38.7          | 74.1       | 57.2       | 53.3       | 58.7       |
|         |        | glyph[check] | ×            | 87.6  | 76.0     | 71.5  | 52.6          | 45.9          | 40.7          | 79.8       | 61.6       | 58.0       | 61.2       |
|         |        | ×            | glyph[check] | 87.8  | 76.5     | 72.0  | 49.4          | 43.7          | 39.4          | 78.7       | 61.5       | 57.5       | 60.6       |
|         |        | glyph[check] | glyph[check] | 88.1  | 76.9     | 73.8  | 54.6          | 47.5          | 42.3          | 80.3       | 62.0       | 58.8       | 62.1       |

Figure 7: Qualitative comparison between the detection results of students trained with and without knowledge distillation. Green boxes and blue boxes indicate the bounding boxes from the prediction results and the ground-truths. Red points are the points insides the prediction bounding boxes.

<!-- image -->

## 5.2 Visualization Analysis

Visualization on Importance Score In the reweighted learning strategy, the importance scores of each voxel or point are utilized to determine whether it should be distilled. Visualization of the importance scores in PointPillars is shown in Figure 6. It is observed that they successfully localize the foreground objects ( e.g., cars and pedestrians) and the hard-negative objects ( e.g., walls).

Visualization on Detection Results In this subsection, we have visualized the detection results of the student model trained with and without our method for comparison. Note that both student models are 4 × compressed PointPillars trained on KITTI. The green and blue boxes indicate the boxes of the model prediction and the ground truth. As shown in Figure 7, the student model without knowledge distillation tends to have much more false-positive (FP) predictions. In contrast, this excessive FP problem is alleviated in the student trained with our method. This observation is consistent with our experimental results that the distilled PointPillars has 3.4 mAP improvements.

## 6 Conclusion

This paper proposes a structured knowledge distillation framework named PointDistiller for point clouds-based object detection. It is composed of local distillation to first encode the semantic information in local geometric structure in point clouds and distill it to students, and reweighted learning to handle the sparsity and noise in point clouds by assigning different learning weights to different points and voxels. Extensive experiments on both voxels-based detectors and raw points-based detectors have demonstrated the superiority over seven previous knowledge distillation methods. Our ablation study has shown the individual effectiveness of each module in PointDistiller. Besides, the visualization results demonstrate that PointDistiller can significantly improve detection performance by reducing false-positive predictions, and the importance score is able to reveal the more significant voxels. To the best of our knowledge, this work initiates the first step to exploring KD for efficient point clouds-based 3D object detection, and we hope this could spur future research.

## References

- [1] Bajestani, M. F. and Yang, Y. Tkd: Temporal knowledge distillation for active perception. In IEEE Winter Conf. Appl. Comput. Vis. (WACV) , pp. 953-962, 2020. 1
- [2] Buciluˇ a, C., Caruana, R., and Niculescu-Mizil, A. Model compression. In ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD) , pp. 535-541. ACM, 2006. 3
- [3] Caesar, H., Bankiti, V., Lang, A. H., Vora, S., Liong, V . E., Xu, Q., Krishnan, A., Pan, Y ., Baldan, G., and Beijbom, O. nuscenes: A multimodal dataset for autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 11618-11628. Computer Vision Foundation / IEEE, 2020. 1, 6
- [4] Chen, G., Choi, W., Yu, X., Han, T., and Chandraker, M. Learning efficient object detection models with knowledge distillation. In Adv. Neural Inform. Process. Syst. (NIPS) , pp. 742-751, 2017. 1, 3
- [5] Chen, X., Ma, H., Wan, J., Li, B., and Xia, T. Multi-view 3d object detection network for autonomous driving. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 6526-6534. IEEE Computer Society, 2017. 1
- [6] Chen, Y., Liu, S., Shen, X., and Jia, J. Fast point r-cnn. In Int. Conf. Comput. Vis. (ICCV) , pp. 9775-9784, 2019. 3
- [7] Choi, J., Wang, Z., Venkataramani, S., Chuang, P. I., Srinivasan, V., and Gopalakrishnan, K. PACT: parameterized clipping activation for quantized neural networks. CoRR , abs/1805.06085, 2018. 1
- [8] Chong, Z., Ma, X., Zhang, H., Yue, Y., Li, H., Wang, Z., and Ouyang, W. Monodistill: Learning spatial features for monocular 3d object detection. In Int. Conf. Learn. Represent. (ICLR) , 2022. 3
- [9] Contributors, M. MMDetection3D: OpenMMLab next-generation platform for general 3D object detection. https://github.com/open-mmlab/mmdetection3d , 2020. 6
- [10] Dai, X., Jiang, Z., Wu, Z., Bao, Y., Wang, Z., Liu, S., and Zhou, E. General instance distillation for object detection. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 7842-7851, 2021. 3
- [11] Dong, R., Tan, Z., Wu, M., Zhang, L., and Ma, K. Finding the task-optimal low-bit sub-distribution in deep neural networks. In Proc. Int. Conf. Mach. Learn. (ICML) , 2022. 1
- [12] Fan, L., Pang, Z., Zhang, T., Wang, Y., Zhao, H., Wang, F., Wang, N., and Zhang, Z. Embracing single stride 3d object detector with sparse transformer. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , 2022. 2
- [13] Geiger, A., Lenz, P., Stiller, C., and Urtasun, R. Vision meets robotics: The kitti dataset. The International Journal of Robotics Research , 32(11):1231-1237, 2013. 1, 6
- [14] Graham, B. Sparse 3d convolutional neural networks. In Xie, X., Jones, M. W., and Tam, G. K. L. (eds.), Brit. Mach. Vis. Conf. (BMVC) , pp. 150.1-150.9. BMVA Press, 2015. 2
- [15] Guo, J., Han, K., Wang, Y., Wu, H., Chen, X., Xu, C., and Xu, C. Distilling object detectors via decoupled features. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 2154-2164. Computer Vision Foundation / IEEE, 2021. 3
- [16] Guo, X., Shi, S., Wang, X., and Li, H. Liga-stereo: Learning lidar geometry aware representations for stereo-based 3d detector. In Int. Conf. Comput. Vis. (ICCV) , pp. 3133-3143. IEEE, 2021. 2, 3
- [17] Gupta, S., Hoffman, J., and Malik, J. Cross modal distillation for supervision transfer. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 2827-2836, 2016. 2, 3
- [18] Han, S., Mao, H., and Dally, W. J. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In Int. Conf. Learn. Represent. (ICLR) , 2016. 1
- [19] Heo, B., Kim, J., Yun, S., Park, H., Kwak, N., and Choi, J. Y. A comprehensive overhaul of feature distillation. In Int. Conf. Comput. Vis. (ICCV) , pp. 1921-1930, 2019. 6, 8
- [20] Hinton, G., Vinyals, O., and Dean, J. Distilling the knowledge in a neural network. In Adv. Neural Inform. Process. Syst. (NeurIPS) , 2014. 1, 3
- [21] Howard, A., Pang, R., Adam, H., Le, Q. V., Sandler, M., Chen, B., Wang, W., Chen, L., Tan, M., Chu, G., Vasudevan, V., and Zhu, Y. Searching for mobilenetv3. In Int. Conf. Comput. Vis. (ICCV) , pp. 1314-1324. IEEE, 2019. 1

- [22] Jin, Q., Ren, J., Woodford, O. J., Wang, J., Yuan, G., Wang, Y., and Tulyakov, S. Teachers do more than teach: Compressing image-to-image models. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 13600-13611, 2021. 3
- [23] Kang, Z., Zhang, P., Zhang, X., Sun, J., and Zheng, N. Instance-conditional knowledge distillation for object detection. Adv. Neural Inform. Process. Syst. (NeurIPS) , 34, 2021. 3
- [24] Laine, S. and Aila, T. Temporal ensembling for semi-supervised learning. In Int. Conf. Learn. Represent. (ICLR) . OpenReview.net, 2017. 3
- [25] Lang, A. H., Vora, S., Caesar, H., Zhou, L., Yang, J., and Beijbom, O. Pointpillars: Fast encoders for object detection from point clouds. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 12697-12705, 2019. 1, 3, 6
- [26] Li, G., Mueller, M., Qian, G., Delgadillo Perez, I. C., Abualshour, A., Thabet, A. K., and Ghanem, B. Deepgcns: Making gcns go as deep as cnns. IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI) , pp. 1-1, 2021. 2
- [27] Li, M., Lin, J., Ding, Y., Liu, Z., Zhu, J.-Y., and Han, S. Gan compression: Efficient architectures for interactive conditional gans. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 5284-5294, 2020. 3
- [28] Li, Q., Jin, S., and Yan, J. Mimicking very efficient network for object detection. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 6356-6364, 2017. 1, 3
- [29] Li, S., Wu, J., Xiao, X., Chao, F., Mao, X., and Ji, R. Revisiting discriminator in GAN compression: A generator-discriminator cooperative compression scheme. In Adv. Neural Inform. Process. Syst. (NeurIPS) , pp. 28560-28572, 2021. 3
- [30] Li, Y., Chen, H., Cui, Z., Timofte, R., Pollefeys, M., Chirikjian, G. S., and Van Gool, L. Towards efficient graph convolutional networks for point cloud handling. In Int. Conf. Comput. Vis. (ICCV) , pp. 3752-3762, 2021. 3
- [31] Li, Z., Jiang, R., and Aarabi, P. Semantic relation preserving knowledge distillation for image-to-image translation. In Eur. Conf. Comput. Vis. (ECCV) , pp. 648-663. Springer, 2020. 3, 6
- [32] Li, Z., Wang, F., and Wang, N. Lidar r-cnn: An efficient and universal 3d object detector. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 7546-7555, 2021. 3
- [33] Lin, Z.-H., Huang, S.-Y., and Wang, Y.-C. F. Convolution in the cloud: Learning deformable kernels in 3d graph convolution networks for point cloud analysis. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 1800-1809, 2020. 3
- [34] Liu, Y., Chen, K., Liu, C., Qin, Z., Luo, Z., and Wang, J. Structured knowledge distillation for semantic segmentation. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 2604-2613, 2019. 1, 3
- [35] Liu, Y., Shu, C., Wang, J., and Shen, C. Structured knowledge distillation for dense prediction. IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI) , 2020. 3
- [36] Liu, Z., Mu, H., Zhang, X., Guo, Z., Yang, X., Cheng, K.-T., and Sun, J. Metapruning: Meta learning for automatic neural network channel pruning. In Int. Conf. Comput. Vis. (ICCV) , October 2019. 1
- [37] Liu, Z., Tang, H., Lin, Y ., and Han, S. Point-voxel cnn for efficient 3d deep learning. In Adv. Neural Inform. Process. Syst. (NeurIPS) , volume 32, 2019. 3
- [38] Louizos, C., Welling, M., and Kingma, D. P. Learning sparse neural networks through l\_0 regularization. In Int. Conf. Learn. Represent. (ICLR) . OpenReview.net, 2018. 1
- [39] Ma, N., Zhang, X., Zheng, H.-T., and Sun, J. Shufflenet v2: Practical guidelines for efficient cnn architecture design. In Eur. Conf. Comput. Vis. (ECCV) , pp. 116-131, 2018. 1
- [40] Ma, X., Qin, C., You, H., Ran, H., and Fu, Y. Rethinking network design and local geometry in point cloud: A simple residual mlp framework. In Int. Conf. Learn. Represent. (ICLR) , 2021. 2
- [41] Nagel, M., Baalen, M. v., Blankevoort, T., and Welling, M. Data-free quantization through weight equalization and bias correction. In Int. Conf. Comput. Vis. (ICCV) , October 2019. 1
- [42] Paigwar, A., Sierra-Gonzalez, D., Erkent, Ö., and Laugier, C. Frustum-pointpillars: A multi-stage approach for 3d object detection using rgb camera and lidar. In Int. Conf. Comput. Vis. (ICCV) , pp. 2926-2933, 2021. 3

- [43] Park, W., Kim, D., Lu, Y., and Cho, M. Relational knowledge distillation. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 3967-3976, 2019. 3
- [44] Park, Y., Lepetit, V., and Woo, W. Multiple 3d object tracking for augmented reality. In 7th IEEE and ACM International Symposium on Mixed and Augmented Reality, ISMAR 2008, Cambridge, UK, 15-18th September 2008 , pp. 117-120. IEEE Computer Society, 2008. 1
- [45] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E. Z., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. Pytorch: An imperative style, high-performance deep learning library. In Adv. Neural Inform. Process. Syst. (NeurIPS) , pp. 8024-8035, 2019. 6
- [46] Peng, B., Jin, X., Liu, J., Li, D., Wu, Y., Liu, Y., Zhou, S., and Zhang, Z. Correlation congruence for knowledge distillation. In Int. Conf. Comput. Vis. (ICCV) , pp. 5007-5016, 2019. 3
- [47] Qi, C. R., Su, H., Mo, K., and Guibas, L. J. Pointnet: Deep learning on point sets for 3d classification and segmentation. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 652-660, 2017. 1, 3, 5
- [48] Qi, C. R., Yi, L., Su, H., and Guibas, L. J. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In Adv. Neural Inform. Process. Syst. (NIPS) , volume 30, 2017. 1, 2, 3, 5
- [49] Qi, C. R., Zhou, Y., Najibi, M., Sun, P., V o, K., Deng, B., and Anguelov, D. Offboard 3d object detection from point cloud sequences. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 6134-6144, 2021. 1, 3
- [50] Ren, Y., Wu, J., Xiao, X., and Yang, J. Online multi-granularity distillation for GAN compression. In Int. Conf. Comput. Vis. (ICCV) , pp. 6773-6783. IEEE, 2021. 3
- [51] Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., and Bengio, Y. Fitnets: Hints for thin deep nets. In Int. Conf. Learn. Represent. (ICLR) , 2015. 1, 3, 6, 8
- [52] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and Chen, L.-C. Mobilenetv2: Inverted residuals and linear bottlenecks. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 4510-4520, 2018. 1
- [53] Sanh, V., Debut, L., Chaumond, J., and Wolf, T. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108 , 2019. 1, 3
- [54] Sautier, C., Puy, G., Gidaris, S., Boulch, A., Bursuc, A., and Marlet, R. Image-to-lidar self-supervised distillation for autonomous driving data. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , 2022. 2, 3
- [55] Shi, S., Wang, X., and Li, H. Pointrcnn: 3d object proposal generation and detection from point cloud. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 770-779, 2019. 3, 6
- [56] Shi, S., Guo, C., Jiang, L., Wang, Z., Shi, J., Wang, X., and Li, H. PV-RCNN: point-voxel feature set abstraction for 3d object detection. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 10526-10535. Computer Vision Foundation / IEEE, 2020. 3
- [57] Shu, D. W., Park, S. W., and Kwon, J. 3d point cloud generative adversarial network based on tree structured graph convolutions. In Int. Conf. Comput. Vis. (ICCV) , pp. 3859-3868, 2019. 3
- [58] Tang, H., Liu, Z., Zhao, S., Lin, Y ., Lin, J., Wang, H., and Han, S. Searching efficient 3d architectures with sparse point-voxel convolution. In Eur. Conf. Comput. Vis. (ECCV) , pp. 685-702. Springer, 2020. 3
- [59] Tarvainen, A. and Valpola, H. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In Adv. Neural Inform. Process. Syst. (NIPS) , pp. 1195-1204, 2017. 3
- [60] Tian, Y., Krishnan, D., and Isola, P. Contrastive representation distillation. In Int. Conf. Learn. Represent. (ICLR) . OpenReview.net, 2020. 6, 8
- [61] Tung, F. and Mori, G. Similarity-preserving knowledge distillation. In Int. Conf. Comput. Vis. (ICCV) , pp. 1365-1374, 2019. 3, 8
- [62] Wang, T., Yuan, L., Zhang, X., and Feng, J. Distilling object detectors with fine-grained feature imitation. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 4933-4942, 2019. 3
- [63] Wang, Y., Sun, Y., Liu, Z., Sarma, S. E., Bronstein, M. M., and Solomon, J. M. Dynamic graph cnn for learning on point clouds. ACM Trans. Graph. , 38(5):1-12, 2019. 2, 3, 5

- [64] Wu, W., Qi, Z., and Li, F. Pointconv: Deep convolutional networks on 3d point clouds. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 9621-9630. Computer Vision Foundation / IEEE, 2019. 2
- [65] Xu, C., Zhou, W., Ge, T., Wei, F., and Zhou, M. Bert-of-theseus: Compressing BERT by progressive module replacing. In Conference on Empirical Methods in Natural Language Processing (EMNLP) , pp. 7859-7869. Association for Computational Linguistics, 2020. 1, 3
- [66] Yan, Y., Mao, Y., and Li, B. Second: Sparsely embedded convolutional detection. Sensors , 18(10):3337, 2018. 3, 6
- [67] Zagoruyko, S. and Komodakis, N. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. In Int. Conf. Learn. Represent. (ICLR) , 2017. 3, 6, 8
- [68] Zhang, L. and Ma, K. Improve object detection with feature-based knowledge distillation: Towards accurate and efficient detectors. In Int. Conf. Learn. Represent. (ICLR) , 2021. 1, 3, 6, 8
- [69] Zhang, L., Shi, Y., Shi, Z., Ma, K., and Bao, C. Task-oriented feature distillation. In Adv. Neural Inform. Process. Syst. (NeurIPS) , 2020. 3
- [70] Zhang, L., Chen, X., Tu, X., Wan, P., Xu, N., and Ma, K. Wavelet knowledge distillation: Towards efficient image-to-image translation. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , 2022. 3
- [71] Zhang, T., Ye, S., Zhang, K., Tang, J., Wen, W., Fardad, M., and Wang, Y. A systematic dnn weight pruning framework using alternating direction method of multipliers. In Eur. Conf. Comput. Vis. (ECCV) , September 2018. 1
- [72] Zhang, Y., Xiang, T., Hospedales, T. M., and Lu, H. Deep mutual learning. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 4320-4328, 2018. 3
- [73] Zheng, W., Tang, W., Jiang, L., and Fu, C. SE-SSD: self-ensembling single-stage object detector from point cloud. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 14494-14503. Computer Vision Foundation / IEEE, 2021. 6, 8
- [74] Zhou, H., Feng, Y., Fang, M., Wei, M., Qin, J., and Lu, T. Adaptive graph convolution for point cloud analysis. In Int. Conf. Comput. Vis. (ICCV) , pp. 4965-4974, 2021. 3
- [75] Zhou, Y. and Tuzel, O. Voxelnet: End-to-end learning for point cloud based 3d object detection. In IEEE/CVF Conf. Comput. Vis. Pattern Recog. (CVPR) , pp. 4490-4499, 2018. 3