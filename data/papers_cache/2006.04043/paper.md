## SVGA-Net: Sparse Voxel-Graph Attention Network for 3D Object Detection from Point Clouds

Qingdong He, Zhengning Wang * , Hao Zeng, Yi Zeng, Yijun Liu University of Electronic Science and Technology of China

heqingdong@alu.uestc.edu.cn,zhengning.wang@uestc.edu.cn

{haozeng,zengyii,yijunliu}@std.uestc.edu.cn

## Abstract

Accurate 3D object detection from point clouds has become a crucial component in autonomous driving. However, the volumetric representations and the projection methods in previous works fail to establish the relationships between the local point sets. In this paper, we propose Sparse VoxelGraph Attention Network (SVGA-Net), a novel end-to-end trainable network which mainly contains voxel-graph module and sparse-to-dense regression module to achieve comparable 3D detection tasks from raw LIDAR data. Specifically, SVGA-Net constructs the local complete graph within each divided 3D spherical voxel and global KNN graph through all voxels. The local and global graphs serve as the attention mechanism to enhance the extracted features. In addition, the novel sparse-to-dense regression module enhances the 3D box estimation accuracy through feature maps aggregation at different levels. Experiments on KITTI detection benchmark demonstrate the efficiency of extending the graph representation to 3D object detection and the proposed SVGA-Net can achieve decent detection accuracy.

## 1. Introduction

With the widespread popularity of LIDAR sensors in autonomous driving [4] and augmented reality [17], 3D object detection from point clouds has become a mainstream research direction. Compared to RGB images from video cameras, point clouds could provide accurate depth and geometric information [37] which can be used not only to locate the object, but also to describe the shape of the object [38]. However, the properties of unordered, sparsity and relevance of point clouds make it a challenging task to utilize point clouds for 3D object detection directly.

In recent years, several pioneering approaches have been proposed to tackle these challenges for 3D object detection on point clouds. The main ideas for processing point clouds data are to project point clouds to different views[28, 2, 9, 14, 34] or divide the point clouds into equally spaced voxels[12, 39, 33]. Then convolutional neural networks and mature 2D objection detection frameworks [23, 22] are applied to extract features. However, because projection alone cannot capture the object's geometric information well, many methods[2, 31, 18, 29] have to combine RGB images in the designed network. While the methods using only voxelization do not make good use of the properties of the point clouds and bring a huge computational burden[15] as resolution increases. Apart from converting point clouds into other formats, some works [26, 36] take Pointnets [19, 20] as backbone to process point clouds directly. Although Pointnets build a hierarchical network and use a symmetric function to maintain permutation invariance, they fail to construct the neighbour relationships between the grouped point sets [30].

* Corresponding Author

Considering the properties of point clouds, we should notice the superiority of graphs in dealing with the irregular data. In fact, in the domain of point clouds for segmentation and classification tasks, the method of processing with graphs has been deeply studied by many works [21, 1, 10, 24, 30]. However, few researches have used graphs to make 3D object detection from point clouds. To our knowledge, Point-GNN[27] may be the first to prove the potential of using the graph neural network as a new approach for 3D object detection. Point-GNN introduces auto-registration mechanism to reduce translation variance and designs box merging and scoring operation to combine detection results from multiple vertices accurately. However, similar to ShapeContextNet [32] and Pointnet++ [20], the relationship between point sets is not well established in the feature extraction process and a large number of matrix operations will bring heavy calculation burden and memory cost.

In this paper, we propose the sparse voxel-graph attention network (SVGA-Net) for 3D object detection. SVGA-Net is an end-to-end trainable network which takes raw point clouds as input as outputs the category and bounding boxes information of the object. Specifically, SVGA-Net mainly consists of voxel-graph network module and sparse-to-dense regression module. Instead of normalized rectangle voxels, we divide the point clouds into 3D spherical space with a fixed radius. The voxel-graph network aims to construct local complete graph for each voxel and global KNN graph for all voxels. The local and global serve as the attention mechanism that can provide a parameter supervision factor for the feature vector of each point. In this way, the local aggregated features can be combined with the global point-wise features. Then we design the sparse-to-dense regression module to predict the category and 3D bounding box by processing the features at different scales. Evaluations on KITTI benchmark demonstrate that our proposed method can achieve comparable results with the state-of-the-art approaches.

Our key contributions can be summarized as follows:

- We propose a new end-to-end trainable 3D object detection network from point clouds which uses graph representations without converting to other formats.
- We design a voxel-graph network, which constructs the local complete graph within each spherical voxel and the global KNN graph through all voxels to learn the discriminative feature representation simultaneously.
- We propose a novel 3D boxes estimation method that aggregates features at different scales to achieve higher 3D localization accuracy.
- Our proposed SVGA-Net achieves decent experimental results with the state-of-the-art methods on the challenging KITTI 3D detection dataset.

## 2. Related Work

## 2.1. Projection-based methods for point clouds

To align with RGB images, series of works process point clouds through projection [2, 9, 13]. Among them, MV3D [2] projects point clouds to bird view and trains a Region Proposal Network (RPN) to generate positive proposals. It extracts features from LiDAR bird view, LIDAR front view and RGB image, for every proposal to generate refined 3D bounding boxes. AVOD [9] improves MV3D by fusing image and bird view features and merges features from multiple views in the RPN phase to generate positive proposals. Note that accurate geometric information may be lost in the high-level layers with this scheme.

## 2.2. Volumetric methods for point clouds

Another typical method for processing point clouds is voxelization. VoxelNet [39] is the first network to process point clouds with voxelization, which use stacked VFE layers to extract features tensors. Following it, a large number of methods [16, 33, 25, 3] divide the 3D space into regular grids and group the points in a grid as a whole. However, they often need to stack heavy 3D CNN layers to realize geometric pose inference which bring large computation.

## 2.3. Pointnet-based methods for point clouds

To process point clouds directly, PointNet [19] and PonintNet++ [20] are the two groundbreaking works to design parallel MLPs to extract features from the raw irregular data, which improve the accuracy greatly. Taking them as backbone, many works [26, 18, 11, 36, 35] begin to design different feature extractors to achieve better performance. Although Pointnets are effective to abstract features, they still suffer feature loss between the local and global point sets.

## 2.4. Graph-based methods for point clouds

Constructing graphs to learn the order-invariant representation of the irregular point clouds data has been explored in classification and segmentation tasks [7, 30]. Graph convolution operation is efficient to compute features between points. DGCNN [30] proposes EdgeConv in the neighbor point sets to fuse local features in a KNN graph. SAWNet [7] extends the ideas of PointNet and DGCNN to learn both local and global information for points. Surprisingly, few researches have considered applying graph for 3D object detection. Point-GNN may be the first work to design a GNN for 3D object detection. Point-GNN [27] designs a one-stage graph neural network to predict the category and shape of the object with an auto-registration mechanism, merging and scoring operation, which demonstrate the potential of using the graph neural network as a new approach for 3D object detection.

## 3. Proposed Method

In this section, we detail the architecture of the proposed SVGA-Net for 3D detection from point clouds. As shown in Figure 1, our SVGA-Net architecture mainly consists of two modules: voxel-graph network and spare-to-dense regression.

## 3.1. Voxel-graph network architecture

Spherical voxel grouping. Consider the original point clouds are represented as G = { V, D } , where V = { p 1 , p 2 , ..., p n } indicting n points in a D dimensional metric space. In our practice, D is set to 4 so each point in 3D space is defined as v i = [ x i , y i , z i ] , where x i , y i , z i denote the coordinate values of each point along the axes X, Y, Z and the fourth dimension is the laser reflection intensity which denoted as s i .

Then in order to cover the entire point set better, we use the iterative farthest point sampling [20] to choose N farthest points P = { p i = [ v i , s i ] T ∈ R 4 } i =1 , 2 ,...N . According to each point in P , we search its nearest neighbor within a fixed radius r to form a local voxel sphere:

Figure 1. Architecture of the proposed SVGA-Net. The voxel-graph network takes raw point clouds as input, partitions the space into spherical voxels, transforms the points in each sphere to a vector representing the feature information. The sparse-to-dense regression module takes the aggregated features as input as generates the final boxes information.

<!-- image -->

<!-- formula-not-decoded -->

In this way, we can subdivide the 3D space into N 3D spherical voxels B = { b 1 , b 2 , ..., b N } .

Local point-wise feature. As shown in Figure 1, for each spherical voxel b i = { p j = [ x j , y j , z j , s j ] T } j =1 , 2 ,...,t with t points ( t varies for different voxel sphere), the coordinate information of all points inside form the input vector. We extract the local point-wise features for each voxel sphere by learning a mapping:

<!-- formula-not-decoded -->

Then, we could obtain the local point-wise feature representation for each voxel sphere F = { f i , i = 1 , ..., t } , which are transformed by the subsequent layers for deeper feature learning.

Local point-attention layer. Taken the features of each nodes as input, the local point-attention layer outputs the refined features F ′ = { f ′ i , i = 1 , ..., t } through series of information aggregation. As shown in Figure 2, we construct a complete graph for each local node set and KNN graph for all the spherical voxels. We aggregate the information of each node according to the local and global attention score. The feature aggregation of j -th node is represented as:

<!-- formula-not-decoded -->

<!-- image -->

{ ∐

}(

⌈}̂∐⌈(̂}⌉√⌈˜√˜(˜√∐√[

{ ̂}(

˜⌈}̂∐⌈(⊙⊗⊗(˜√∐√[

Figure 2. Graph construction. Each node with different color indicates the aggregated feature and arrows direction represents the information propagation direction with independent attention calculations scores. (a) local complete graph: for each node, we aggregate the information of all the nodes within the same spherical voxel according to the attention score. (b) global 3-NN graph: we aggregate the information of the three nearest neighbours around each node according to the attention score.

where f ′ j denotes the dynamic updated feature of node p j and f j is the input feature of node p j . glyph[unionsq] ( p j ) denotes the index of the other nodes inside the same sphere. f j,k denotes the feature of the k -th nodes inside the same sphere. α j,k is the local attention score between node p j and the other nodes inside the same sphere. β m is the global attention score from the global KNN graph in the m -th iterations.

As shown in Figure 2 (a), we construct a complete graph for all nodes within a voxel sphere to learn the features constrained by each other. In order to allow each point to attend on every other point and make coefficients easily comparable across different points, we normalize them across all choices using the softmax function, so the local attention score α j,k is calculated by:

<!-- formula-not-decoded -->

Global attention layer. By constructing the local complete graph, the aggregated features can only describe the local feature and do not integrate with the global information. So we design the global attention layer to learn the global feature of each spherical voxel and offer a feature factor aligned to each node.

For the points within each b i in N 3D spherical voxels B = { b 1 , b 2 , ..., b N } , we calculate the physical centers of all voxels which denoted as { c i } i =1 ,...,N . Each center is learned by a 3-layer MLP to get the initial global feature F g = { f g, 1 , f g, 2 , ..., f g,N } . As Figure 2 (b) shows, we construct a KNN graph for the N voxel sphere. For each node f g,i , the attention score between node f g,i and its l -th neighbor is calculated as follows:

<!-- formula-not-decoded -->

where glyph[Omegainv] ( f g,i ) denotes the index of the neighbors of node f g,i . m is the number of the point attention layers. Eq. 5 can be regarded as a weighted summation of the K neighbor nodes around a node, which guarantees the permutation invariance to the nodes' order.

Voxel-graph features representation. The point attention operation on each spherical voxel can combine the parameter factor from both local and global, each of which is inserted with a 2-layer MLP with a nonlinear activation to transform each updated feature f ′ j . By stacking multiple point attention layers, both local aggregated feature and global point-wise feature can be learned. We then apply maxpool on the aggregated feature to obtain the final feature vector. To process all the spherical voxel, we obtain a set of voxel sphere features, each of which corresponds to the spatial coordinates of the voxels and is taken as input of the sparse-to-dense regression module.

## 3.2. Sparse-to-dense regression

For each 3D bounding box in 3D space, the predicted box information is represented as ( x, y, z, l, w, h, θ ) , where ( x, y, z ) is the center coordinate of the bounding box, ( l, w, h ) is the size information alongside length, width and height respectively, and θ is the heading angle. Feature map from the voxel-graph network is processed by region proposal regression module. The architecture of the specified sparse-to-dense regression(SDR) module is illustrated in Figure 3.

SDR module first apply three similar blocks as [39, 11] to generate smaller the spatial resolution from top to down. Each block consist of series of Conv 2 D ( f in , f out , k, s, p )

⌈∐√√]˜]̂∐√]}{(

∫˜˜√˜√√]}{(

Figure 3. The architecture of the sparse-to-dense regression module. Features from the voxel-graph network are processed by series of region proposal extraction operations to generate the final classification and regression maps.

<!-- image -->

layers, followed by BatchNorm and a ReLU, where f in and f out are the number of input and output channels, k, s, p represent the kernel size, stride size and padding size respectively. The stride size is set to 2 for the first layer of each block to downsample the feature map by half, followed by sequence of convolutions with stride 1. And the output of the three blocks is denoted as b 1 , b 2 , b 3 respectively.

In order to combine high-resolution features with large receptive fields and low-resolution features with small receptive fields, we concat the output of the second and third modules b 2 , b 3 with the output of the first and second modules b 1 , b 2 after upsampling. In this way, the dense feature range of the lower level can be well combined with the sparse feature range of the higher level. Then a series of convolution operations with an upsampling layer are performed in parallel on three scale channels to generate three feature maps with the same scale size, which are denoted as F 1 , F 2 , F 3 .

In addition, we consider that the features output of F 1 , F 2 , F 3 are more densely fit to our final goal than the original three modules. Therefore, in order to combine the original sparse feature map and the series of processed dense feature maps, we combine the original output b 1 , b 2 , b 3 after upsampling and F 1 , F 2 , F 3 by element-wise addition. The final output F s is obtained by concatenating the fused feature maps after a 3 × 3 convolution layer. And F s is taken as input to perform category classification and 3D bounding box regression.

## 3.3. Loss function

We use a multi-task loss to train our network. Each prior anchor and ground truth bounding box are parameterized as ( x a , y a , z a , l a , w a , h a , θ a ) and

( x gt , y gt , z gt , l gt , w gt , h gt , θ gt ) respectively. The regression residuals between anchors and ground truth are computed as:

<!-- formula-not-decoded -->

where d a = √ ( w a ) 2 +( l a ) 2 . And we use Smooth L1 loss[5] as our 3D bounding box regression loss L reg .

For the object classification loss, we apply the classification binary cross entropy loss.

<!-- formula-not-decoded -->

where N pos and N neg are the number of the positive and negative anchors. p pos i and p neg i are the softmax output for positive and negative anchors respectively. γ 1 and γ 2 are positive constants to balance the different anchors, which are set to 1.5 and 1 respectively in our practice.

Our total loss is composed of two parts, the classification loss L cls and the bounding box regression loss L reg as:

<!-- formula-not-decoded -->

where ∆ t ∗ and ∆ t are the predicated residual and the regression target respectively. Weighting parameters α and β are used to balance the relative importance of different parts, and their values are set to 1 and 2 respectively.

## 4. Experiments

KITTI. We first evaluate our method on the widely used KITTI 3D object detection benchmark [4]. It includes 7481 training samples and 7518 test samples with three categories: car, pedestrian and cyclist. For each category, detection results are evaluated based on three levels of difficulty: easy, moderate and hard. Furthermore, we divide the training data into a training set (3712 images and point clouds) and a validation set (3769 images and point clouds) at a ratio of about 1: 1 (Ablation studies are conducted on this split). We train our model on train split and compare our results with state-of-the-art methods on both val split and test split. For evaluation, the average precision (AP) metric is to compare with different methods and the 3D IoU of car, cyclist, and pedestrian are 0.7, 0.5, and 0.5 respectively.

## 4.1. Training

Network Architecture. As shown in Figure 1, in the local point-wise feature and global attention layer, the point sets are first processed by 3-layer MLP and the sizes are all (64, 128, 128). In the local point attention layer, we stack n = 3 local point-attention graph to aggregate the features, each followed by a 2-layer MLP. And the sizes of the three MLPs are (128, 128), (128, 256) and (512, 1024) respectively. Following [9, 39, 36], we train two networks, one for cars and another for both pedestrians and cyclists.

For cars, we sample N = 1024 to form the initial point sets. To construct the local complete graph, we choose r = 1 . 8 m . For anchors, an anchor is considered as positive if it has the highest IoU with a ground truth or its IoU score is over 0.6. An anchor is considered as negative if the IoU with all ground truth boxes is less than 0.45. To reduce redundancy, we apply IoU threshold of 0.7 for NMS. For cyclist and pedestrian, the number of the initial point sets is n = 512 . We set r = 0 . 8 to construct the local graph. The anchor is considered as positive if its highest IoU score with a ground truth box or an IoU score is over than 0.5. And an anchor is considered as negative if its IoU score with ground truth box is less than 0.35. The IoU threshold of NMS is set to 0.6.

The network is trained in an end-to-end manner on GTX 1080 GPU. The ADAM optimizer [8] is employed to train our network and its initial learning rate is 0.001 for the first 140 epoches and is decayed by 10 times in every 20 epoches. We train our network for 200 epoches with a batch size of 16 on 4 GPU cards. Furthermore, we also apply data augmentation as [11, 39] do to prevent overfitting.

## 4.2. Comparing with state-of-the-art methods

Performance on KITTI test dataset. We evaluate our method on the 3D detection benchmark benchmark of the KITTI test server. As shown in Table 1, we compare our results with state-of-the-art RGB+Lidar and Lidar only methods for the 3D object detection and the bird's view detection task. Our proposed method outperforms the most effective RGB+Lidar methods MMF[13] by (0.52%, 3.72%, 7.50%) for car category on three difficulty levels of 3D detection.

Compared with the Lidar-based methods, our SVGA-Net can still show decent performance on the three categories. In particular, we achieve decent results compared to PointGNN[27] using the same graph representation method but using graph neural network in the detection of the three categories. We believe that this may benefit from our construction of local and global graphs to better capture the feature information of point clouds. The slight inferiority in the two detection tasks may be due to the fact that the local graph cannot be constructed for objects with occlusion ratio exceeding 80%.

Performance on KITTI validation dataset. For the most important car category, we also report the performance of our method on KITTI val split and the results are shown in Table 2 and Table 3. For car, our proposed method achieves better or comparable results than state-of-the-art methods on three difficulty levels which illustrate the superiority of our method.

| Method           | Modality   | AP car (%)   | AP car (%)   | AP car (%)   | AP pedestrian (%)   | AP pedestrian (%)   | AP pedestrian (%)   | AP cyclist (%)   | AP cyclist (%)   | AP cyclist (%)   |
|------------------|------------|--------------|--------------|--------------|---------------------|---------------------|---------------------|------------------|------------------|------------------|
| Method           | Modality   | Easy         | Moderate     | Hard         | Easy                | Moderate            | Hard                | Easy             | Moderate         | Hard             |
| MV3D[2]          | R+L        | 71.09        | 62.35        | 55.12        | -                   | -                   | -                   | -                | -                | -                |
| F-Pointnet[18]   | R+L        | 81.20        | 70.39        | 62.19        | 51.21               | 44.89               | 40.23               | 71.96            | 56.77            | 50.39            |
| AVOD-FPN[9]      | R+L        | 81.94        | 71.88        | 66.38        | 50.80               | 42.81               | 40.88               | 64.00            | 52.18            | 46.61            |
| F-ConvNet[31]    | R+L        | 85.88        | 76.51        | 68.08        | 52.37               | 45.61               | 41.49               | 79.58            | 64.68            | 57.03            |
| MMF[13]          | R+L        | 86.81        | 76.75        | 68.41        | -                   | -                   | -                   | -                | -                | -                |
| Voxelnet[39]     | L          | 77.47        | 65.11        | 57.73        | 39.48               | 33.69               | 31.51               | 61.22            | 48.36            | 44.37            |
| SECOND[33]       | L          | 83.13        | 73.66        | 66.20        | 51.07               | 42.56               | 37.29               | 70.51            | 53.85            | 46.90            |
| PointPillars[11] | L          | 79.05        | 74.99        | 68.30        | 52.08               | 43.43               | 41.49               | 75.78            | 59.07            | 52.92            |
| PointRCNN[26]    | L          | 85.94        | 75.76        | 68.32        | 49.43               | 41.78               | 38.63               | 73.93            | 59.60            | 53.59            |
| STD[36]          | L          | 86.61        | 77.63        | 76.06        | 53.08               | 44.24               | 41.97               | 78.89            | 62.53            | 55.77            |
| 3DSSD[35]        | L          | 88.36        | 79.57        | 74.55        | -                   | -                   | -                   | -                | -                | -                |
| SA-SSD[6]        | L          | 88.75        | 79.79        | 74.16        | -                   | -                   | -                   | -                | -                | -                |
| PV-RCNN [25]     | L          | 90.25        | 81.43        | 76.82        | -                   | -                   | -                   | 78.60            | 63.71            | 57.65            |
| Point-GNN[27]    | L          | 88.33        | 79.47        | 72.29        | 51.92               | 43.77               | 40.14               | 78.60            | 63.48            | 57.08            |
| SVGA-Net(ours)   | L          | 87.33        | 80.47        | 75.91        | 48.48               | 40.39               | 37.92               | 78.58            | 62.28            | 54.88            |

Table 1. Performance comparison on KITTI 3D object detection for car, pedestrian and cyclists.The evaluation metrics is the average precision (AP) on the official test set. 'R' denotes RGB images input and 'L' denotes Lidar point clouds input.

Table 2. Performance comparison on KITTI 3D object detection val set for car class.

| Method             | Modality   | AP car (%)   | AP car (%)   | AP car (%)   |
|--------------------|------------|--------------|--------------|--------------|
|                    |            | Easy         | Moderate     | Hard         |
| MV3D [2]           | R+L        | 71.29        | 62.68        | 56.56        |
| F-Pointnet [18]    | R+L        | 83.76        | 70.92        | 63.65        |
| AVOD-FPN [9]       | R+L        | 84.41        | 74.44        | 68.65        |
| F-ConvNet[31]      | R+L        | 89.02        | 78.80        | 77.09        |
| Voxelnet [39]      | L          | 81.97        | 65.46        | 62.85        |
| SECOND [33]        | L          | 87.43        | 76.48        | 69.10        |
| PointRCNN [26]     | L          | 88.88        | 78.63        | 77.38        |
| Fast PointRCNN [3] | L          | 89.12        | 79.00        | 77.48        |
| STD[36]            | L          | 89.70        | 79.80        | 79.30        |
| SA-SSD[6]          | L          | 90.15        | 79.91        | 78.78        |
| 3DSSD[35]          | L          | 89.71        | 79.45        | 78.67        |
| Point-GNN[27]      | L          | 87.89        | 78.34        | 77.38        |
| SVGA-Net(ours)     | L          | 90.59        | 80.23        | 79.15        |

Table 3. Performance comparison on KITTI bird's eye view detection val set for car class.

| Method             | Modality   | AP car (%)   | AP car (%)   | AP car (%)   |
|--------------------|------------|--------------|--------------|--------------|
|                    |            | Easy         | Moderate     | Hard         |
| MV3D [2]           | R+L        | 86.55        | 78.10        | 76.67        |
| F-Pointnet [18]    | R+L        | 88.16        | 84.02        | 76.44        |
| F-ConvNet[31]      | R+L        | 90.23        | 88.79        | 86.84        |
| Voxelnet [39]      | L          | 89.60        | 84.81        | 78.57        |
| SECOND [33]        | L          | 89.96        | 87.07        | 79.66        |
| Fast PointRCNN [3] | L          | 90.12        | 88.10        | 86.24        |
| STD[36]            | L          | 90.50        | 88.50        | 88.10        |
| Point-GNN[27]      | L          | 89.82        | 88.31        | 87.16        |
| SVGA-Net(ours)     | L          | 90.27        | 89.16        | 88.11        |

## 4.3. Qualitative results

As shown in Figure 4, we illustrate some qualitative predicted bounding results of our proposed SVGA-Net on the test split on KITTI dataset. For better visualization, we project the 3D bounding boxes into RGB images and BEV in point clouds. From the figures we could see that our proposed network could estimate accurate 3D bounding boxes in different scenes. Surprisingly, SVGA-Net can still produce accurate 3D bounding boxes even under poor lighting conditions and severe occlusion.

## 4.4. Ablation studies

In this section, we conduct series of extensive ablation studies on the validation split of KITTI to illustrate the role of each module in improving the final result and our parameter selection. All ablation studies are implemented on the car class which contains the largest amount of training examples. The evaluation metric is the average precision (AP %) on the val set.

Effect of different design choice. In the local point attention layer, we stack several local complete layers to extract aggregated features. In order to show the impact of the number of the point attention layer, we train our network with n varying from 1 to 4. As shown in Table 4, when the local feature information is transmitted on the 1st to 3rd layers, the detection accuracy is continuously improved because the features are continuously aggregated to the object itself. When n increases to 4, the detection accuracy decreases slightly, and we believe that the network should be over-learning.

Furthermore, we study the importance of the global attention layer in improving the detection accuracy. As shown in Table 4, the AP values on both detection tasks are greatly reduced when we remove this module from the network, which proves the importance of this design in providing global feature information for each point.

In the middle three rows of Table 4, we aim to explore the effect of different design in the spare-to-dense regression module. SR is to remove the concatenation of b 1 , b 2 with the upsampled b 2 , b 3 and DR is to remove the addition of b i with F i . Results show that only the design of sparse-to-dense regression ranks the first in improving detection accuracy.

Figure 4. Qualitative 3D detection results of SVGA-Net on the KITTI test set. The detected objects are shown with green 3D bounding boxes and the relative labels. The upper row in each image is the 3D object detection result projected onto the RGB image and the bottom is the result in the corresponding point clouds.

<!-- image -->

Table 4. Performance comparison with different design choice. n is the number of point-attention layers. 'w/o.' denotes whether to keep the global attention layer. SDR denotes the sparse-to-dense regression.

|      |     | 3 DAP car (%)   | 3 DAP car (%)   | 3 DAP car (%)   | BEVAP car (%)   | BEVAP car (%)   | BEVAP car (%)   |
|------|-----|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
|      |     | Easy            | Moderate        | Hard            | Easy            | Moderate        | Hard            |
|      | 1   | 86.77           | 75.37           | 74.19           | 87.54           | 86.11           | 83.72           |
|      | 2   | 88.86           | 78.81           | 78.03           | 89.04           | 88.44           | 87.05           |
| n    | 3   | 90.59           | 80.23           | 79.15           | 90.27           | 89.16           | 88.11           |
|      | 4   | 89.62           | 79.26           | 77.58           | 89.72           | 88.51           | 87.17           |
| w/o. | o.  | 88.42           | 78.11           | 76.54           | 89.71           | 87.45           | 84.33           |
|      | w.  | 90.59           | 80.23           | 79.15           | 90.27           | 89.16           | 88.11           |
|      | SR  | 87.53           | 77.81           | 76.22           | 86.95           | 86.62           | 85.04           |
|      | DR  | 88.39           | 78.44           | 76.56           | 87.91           | 86.82           | 86.73           |
|      | SDR | 90.59           | 80.23           | 79.15           | 90.27           | 89.16           | 88.11           |
|      | 1   | 76.37           | 69.15           | 68.47           | 82.11           | 80.27           | 79.58           |
|      | 2   | 84.53           | 75.61           | 71.92           | 86.23           | 85.65           | 83.66           |
| k    | 3   | 90.59           | 80.23           | 79.15           | 90.27           | 89.16           | 88.11           |
|      | 4   | 88.91           | 79.22           | 77.86           | 88.07           | 87.88           | 87.08           |
|      | 5   | 86.58           | 76.82           | 75.43           | 85.29           | 84.38           | 83.47           |

When constructing the KNN graph, the number "3" in our implementation is chosen after series of experiments on val set, as shown in the last five rows in Table 4. When K increases from 1 to 3, the AP value has a significant increase, but when it continues to increase, the AP value does decrease.

Running time. Our network is written in Python and implemented in Pytorch for GPU computation. The average inference time for one sample is 62 ms, including 14.5%(9 ms) for data reading and pre-processing, 66.1%(41 ms) for local and global features aggregation and 19.4%(12 ms) for final boxes detection.

## 5. Conclusions

In this paper, we propose a novel sparse voxel-graph attention network(SVGA-Net) for 3D Object Detection from raw Point Clouds. We introduce graph representation to process point clouds. By constructing a local complete graph in the divided spherical voxel space, we can get a better local rep- resentation of the point feature, and the information between the point and its neighborhood can be fused. By constructing a global graph, we can better supervise and learn the features of points. In addition, the sparse-to-dense regression module can also fuse feature maps at different scales. Experiments have demonstrated the efficiency of the design choice in our network. Future work will extend SVGA-Net to combine RGB images to further improve detection accuracy.

## 6. Acknowledgments

This work is supported by a grant from the National Natural Science Foundation of China (No.61872068), by a grant from Science &amp; Technology Department of Sichuan Province of China (No.2020YFG0037, 2020YFG0287,2021YFG0366).

## References

- [1] Y. Bi, A. Chadha, A. Abbas, E. Bourtsoulatze, and Y. Andreopoulos. Graph-based object classification for neuromorphic vision sensing. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV) , pages 491-501, 2019. 1
- [2] Xiaozhi Chen, Huimin Ma, Ji Wan, Bo Li, and Tian Xia. Multi-view 3d object detection network for autonomous driving. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 1907-1915, 2017. 1, 2, 6
- [3] Yilun Chen, Shu Liu, Xiaoyong Shen, and Jiaya Jia. Fast point r-cnn. In Proceedings of the IEEE International Conference on Computer Vision , pages 9775-9784, 2019. 2, 6
- [4] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In 2012 IEEE Conference on Computer Vision and Pattern Recognition , pages 3354-3361. IEEE, 2012. 1, 5
- [5] R. Girshick. Fast r-cnn. In 2015 IEEE International Conference on Computer Vision (ICCV) , pages 1440-1448, 2015.

- [6] Chenhang He, Hui Zeng, Jianqiang Huang, Xian-Sheng Hua, and Lei Zhang. Structure aware single-stage 3d object detection from point cloud. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 11873-11882, 2020. 6
- [7] Chaitanya Kaul, Nick Pears, and Suresh Manandhar. Sawnet: A spatially aware deep neural network for 3d point cloud processing. arXiv preprint arXiv:1905.07650 , 2019. 2
- [8] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 , 2014. 5
- [9] Jason Ku, Melissa Mozifian, Jungwook Lee, Ali Harakeh, and Steven L Waslander. Joint 3d proposal generation and object detection from view aggregation. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , pages 1-8. IEEE, 2018. 1, 2, 5, 6
- [10] Loic Landrieu and Martin Simonovsky. Large-scale point cloud semantic segmentation with superpoint graphs. In CVPR 2018 , 2018. 1
- [11] Alex H Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, and Oscar Beijbom. Pointpillars: Fast encoders for object detection from point clouds. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 12697-12705, 2019. 2, 4, 5, 6
- [12] Bo Li. 3d fully convolutional network for vehicle detection in point cloud. In 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , pages 1513-1518. IEEE, 2017. 1
- [13] Ming Liang, Bin Yang, Yun Chen, Rui Hu, and Raquel Urtasun. Multi-task multi-sensor fusion for 3d object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 7345-7353, 2019. 2, 5, 6
- [14] Ming Liang, Bin Yang, Shenlong Wang, and Raquel Urtasun. Deep continuous fusion for multi-sensor 3d object detection. In Proceedings of the European Conference on Computer Vision (ECCV) , pages 641-656, 2018. 1
- [15] Zhijian Liu, Haotian Tang, Yujun Lin, and Song Han. Pointvoxel cnn for efficient 3d deep learning. In Advances in Neural Information Processing Systems , pages 963-973, 2019. 1
- [16] Zhe Liu, Xin Zhao, Tengteng Huang, Ruolan Hu, Yu Zhou, and Xiang Bai. Tanet: Robust 3d object detection from point clouds with triple attention. AAAI , 2020. 2
- [17] Youngmin Park, Vincent Lepetit, and Woontack Woo. Multiple 3d object tracking for augmented reality. In Proceedings of the 7th IEEE/ACM International Symposium on Mixed and Augmented Reality , ISMAR '08, pages 117-120, Washington, DC, USA, 2008. IEEE Computer Society. 1
- [18] Charles R Qi, Wei Liu, Chenxia Wu, Hao Su, and Leonidas J Guibas. Frustum pointnets for 3d object detection from rgb-d data. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 918-927, 2018. 1, 2, 6
- [19] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 652-660, 2017. 1, 2
- [20] Charles Ruizhongtai Qi, Li Yi, Hao Su, and Leonidas J Guibas. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. In Advances in neural information processing systems , pages 5099-5108, 2017. 1, 2
- [21] X. Qi, R. Liao, J. Jia, S. Fidler, and R. Urtasun. 3d graph neural networks for rgbd semantic segmentation. In 2017 IEEE International Conference on Computer Vision (ICCV) , pages 5209-5218, 2017. 1
- [22] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 779-788, 2016. 1
- [23] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In Advances in neural information processing systems , pages 91-99, 2015. 1
- [24] Y. Shen, C. Feng, Y. Yang, and D. Tian. Mining point cloud local structures by kernel correlation and graph pooling. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 4548-4557, 2018. 1
- [25] Shaoshuai Shi, Chaoxu Guo, Li Jiang, Zhe Wang, Jianping Shi, Xiaogang Wang, and Hongsheng Li. Pv-rcnn: Pointvoxel feature set abstraction for 3d object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 10529-10538, 2020. 2, 6
- [26] Shaoshuai Shi, Xiaogang Wang, and Hongsheng Li. Pointrcnn: 3d object proposal generation and detection from point cloud. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 770-779, 2019. 1, 2, 6
- [27] Weijing Shi and Ragunathan (Raj) Rajkumar. Point-gnn: Graph neural network for 3d object detection in a point cloud. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , June 2020. 1, 2, 5, 6
- [28] Martin Simon, Karl Amende, Andrea Kraus, Jens Honer, Timo Samann, Hauke Kaulbersch, Stefan Milz, and Horst Michael Gross. Complexer-yolo: Real-time 3d object detection and tracking on semantic point clouds. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops , pages 0-0, 2019. 1
- [29] V. A. Sindagi, Y. Zhou, and O. Tuzel. Mvx-net: Multimodal voxelnet for 3d object detection. In 2019 International Conference on Robotics and Automation (ICRA) , pages 7276-7282, 2019. 1
- [30] Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, and Justin M. Solomon. Dynamic graph cnn for learning on point clouds. ACM Trans. Graph. , 38(5):146:1-146:12, Oct. 2019. 1, 2
- [31] Zhixin Wang and Kui Jia. Frustum convnet: Sliding frustums to aggregate local point-wise features for amodal. In 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , pages 1742-1749. IEEE, 2019. 1, 6
- [32] S. Xie, S. Liu, Z. Chen, and Z. Tu. Attentional shapecontextnet for point cloud recognition. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 4606-4615, 2018. 1
- [33] Yan Yan, Yuxing Mao, and Bo Li. Second: Sparsely embedded convolutional detection. Sensors , 18(10):3337, 2018. 1, 2, 6
- [34] Bin Yang, Wenjie Luo, and Raquel Urtasun. Pixor: Real-time 3d object detection from point clouds. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition , pages 7652-7660, 2018. 1
- [35] Zetong Yang, Yanan Sun, Shu Liu, and Jiaya Jia. 3dssd: Point-based 3d single stage object detector. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 11040-11048, 2020. 2, 6
- [36] Zetong Yang, Yanan Sun, Shu Liu, Xiaoyong Shen, and Jiaya Jia. Std: Sparse-to-dense 3d object detector for point cloud. In Proceedings of the IEEE International Conference on Computer Vision , pages 1951-1960, 2019. 1, 2, 5, 6
- [37] Yikuan Yu, Zitian Huang, Fei Li, Haodong Zhang, and Xinyi Le. Point encoder gan: A deep learning model for 3d point cloud inpainting. Neurocomputing , 384:192-199, 2020. 1
- [38] Junning Zhang, Qunxing Su, Cheng Wang, and Hongqiang Gu. Monocular 3d vehicle detection with multi-instance depth and geometry reasoning for autonomous driving. Neurocomputing , 2020. 1
- [39] Yin Zhou and Oncel Tuzel. Voxelnet: End-to-end learning for point cloud based 3d object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pages 4490-4499, 2018. 1, 2, 4, 5, 6