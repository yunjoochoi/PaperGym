## Semi Supervised Deep Quick Instance Detection and Segmentation

Ashish Kumar 1 , L. Behera 1 , Senior Member IEEE

{ https://github.com/ashishkumar822 }

Abstract -In this paper, we present a semi supervised deep quick learning framework for instance detection and pixelwise semantic segmentation of images in a dense clutter of items. The framework can quickly and incrementally learn novel items in an online manner by real-time data acquisition and generating corresponding ground truths on its own. To learn various combinations of items, it can synthesize cluttered scenes, in real time. The overall approach is based on the tutor-child analogy in which a deep network (tutor) is pretrained for class-agnostic object detection which generates labeled data for another deep network (child). The child utilizes a customized convolutional neural network head for the purpose of quick learning. There are broadly four key components of the proposed framework: semi supervised labeling, occlusion aware clutter synthesis, a customized convolutional neural network head, and instance detection. The initial version of this framework was implemented during our participation in Amazon Robotics Challenge (ARC), 2017 . Our system was ranked 3 rd rd, 4 th and 5 th worldwide in pick, stow-pick and stow task respectively. The proposed framework is an improved version over ARC' 17 where novel features such as instance detection and online learning has been added.

## I. INTRODUCTION

Object detection and semantic segmentation are one of the primary tasks in a visual perception system whether biological or artificial. These tasks have been widely explored in the history of machine vision, however, a drastic performance boost has occurred only in a past few years. Convolutional neural networks (CNNs) [1], [2] and advances in the massive parallel computing architectures (GPUs, TPUs) are the primary reasons behind the improvement. However, the CNNs, works well only for the data modality on which they are trained and require large amounts of labeled data as well as training time to achieve that. One of the practical applications of CNN based visual recognition is in warehouse automation process, where a large number of novel items arrives in the warehouses on the daily basis. It becomes challenging to generate labeled images and to retrain the CNN based methods, every time a novel item arrives. Hence, in this paper, we develop an online learning system which can generate labeled data from live camera feeds and incrementally learn the items by clutter synthesis, with no manual quality assurance.

In a broad way, two tasks are manually performed in the warehouses: i ) picking target items from a storage system called 'rack' and preparing them for packing, known as Pick Task , and ii ) picking items from a container called 'tote' and organizing all of them into a rack, known as a

1 Mr. Ashish Kumar and Dr. L. Behera are with the Department of Electrical Engineering, Indian Institute of Technology, Kanpur { krashish,lbehera } @iitk.ac.in Stow Task . To automate these tasks, Amazon Robotics has previously held Amazon Picking Challenge (APC) in 2015 and 2016 , and ARC' 17 . Each of the above challenge involved object recognition of around 40 Amazon provided items ( known-set ), some having multiple instances. Specifically in ARC' 17 , an additional set of novel items was provided merely 45 minutes prior to the competition. The challenge involved object recognition in a dense clutter of known and novel items.

Fig. 1: Our robotic system during ARC' 17 performing final task. Image courtesy: Amazon Robotics and Instance detection and segmentation results of ∼ 40 minutes training.

<!-- image -->

Typically, two approaches can be employed for the pick and stow operations: i ) pick an item and perform recognition in an isolated view, or ii ) perform the recognition in the storage system itself and then pick the item. In pick task, the target items may be occluded non-target items. In this case, the first approach becomes suboptimal as the grasping effort and significant amount of time is wasted when a nontarget item is picked. On the other hand, the second approach performs equally well for both the stow and pick task. Hence, during ARC' 17 , we opted for in-storage recognition and chose to perform training of CNNs for the desired task in 45 minutes. This approach, however, demanded acquiring and labeling the images of the novel items in the due time. In addition, the presence of clutter and transparent and wired mesh items made it quite difficult to directly deploy the stateof-art (SOA) methods for the desired purpose.

In this paper, we present a semi supervised quick online learning framework which faces the above challenges without any human intervention. Its inspiration is mainly based on the remarkable capabilities of human to distinguish between seen and unseen classes and differentiating between instances. To enable our system with such a capability, we make use of tutor-child analogy and develop an online learning system where an artificially intelligent (AI) system teaches another AI. The system is capable of learning new items by generating massive synthetic scenes and their ground truths on its own, with real time data acquisition.

The proposed framework has served as the vision system for our team entry IITK-TCS in ARC' 17 , with a capability of semantic segmentation only. Learning lessons from ARC' 17 , the system in this paper, has been improved to learn while the images are being acquired (online learning) and to detect and segment instances as well. The new functionalities has been added in order to overcome the limitation of semantic segmentation to deal with multiple instances and to reduce manual monitoring. The system components have been designed with an industrial spirit and do not limit their scope only to ARC' 17 . Moreover, due to a large recent attention to the warehouse automation, the underlying ideas of the system components may be concurrently found in the literature, however, their development in this paper is entirely novel.

## II. RELATED WORK

Literature on machine vision algorithms is diverse and vast. Therefore, we limit our discussion to object detection and segmentation, and the perception systems developed by other teams. Faster-RCNN [3], Fast-RCNN [4], RCNN [5] are prevalent CNN based approaches to predict rectangular boxes for object detection in the images. RCNN [5] generates object proposal using selective search whereas these are learned as convolutional filters in [4], [3]. FCN [6] is the first CNN based approach of end-to-end learning for semantic segmentation. PSPNet [7], winner of MIT Scene Parsing Challenge 2016 [8], builds upon pyramidal context extraction using average pooling. Mask-RCNN [9] integrates both box detection and segmentation techniques for the purpose of instance recognition and segmentation. RefineNet [10] focuses on improving segmentation by aggregating multi-scale information and introducing a boundary refining module.

APC' 15 winner, Team RBO [11] designed color and depth based features and trained random forest classifiers for pixel-wise segmentation. APC' 16 winner, Team TuDelft [12] employed Faster-RCNN [3]. Team MIT-Princeton [13] in APC' 16 captured multi-view images to obtain a dense point cloud and used FCN [6] for image segmentation. Team Nimbro [14] combined object detection and semantic segmentation using RGB-D sensory information. Our Team IITK-TCS in APC' 16 used Faster-RCNN which led us 5 th in the stow task.

ARC' 17 stow task winner, Team MIT-Princeton [15] first picks an item based on a class agnostic heat map generated by FCN [6] and later matches its image in an isolated view with the database using glyph[lscript] 2 feature embedding [16]. The team used 16 RGB-D sensors which was a quite expensive solution. On the other hand, Team Nimbro [17] and the final task winner ACRV [18], [19], similar to us went for in-storage detection. Team Nimbro first obtains masks for novel items using background subtraction and later, trains a ResNeXt101 based RefineNet. Team ACRV also used RefineNet for semantic segmentation. Also, both the teams manually performed quality assurance for automatically generated ground truths by their techniques during the data collection of novel items. In addition, many teams including the above, had constrained their workspaces, in order to cop with the ambient lighting which was a major concern.

## III. LEARNING FRAMEWORK

Typically, in a tutor-child 1 analogy, the tutor gathers data from the world and converts it into a form which can be understood by the child. In our approximation to the above, an AI system (tutor CNN) teaches another AI system (child CNN). To realize this, we develop a semi supervised labeled data generation technique in which single instance images (Fig. 2a) are fed to a CNN called tutor which can predict a pixel-wise class agnostic mask of the item present in the image. The predicted mask is converted into a class specific masks by replacing the object mask pixels with the actual label of the item (e.g. brush or bottle), provided by a human. The labeled images are then consumed by a proposed scene synthesis technique in order to synthesize occlusion aware cluttered images. All of the above techniques altogether can be thought as the data conversion process of the tutorchild analogy. Furthermore, the synthesized scenes are fed to the child CNN which employs a novel CNN head to learn quickly for the task of semantic segmentation. The overall flow of the algorithm is shown in Fig. 5b and discussed in detail below.

## A. Semi Supervised Labeling

Our motivation to develop this technique lies behind the need of pixel-wise ground truth masks in order to train the child network for semantic segmentation. Typically, it requires 1 -2 hours of manual effort to generate masks for roughly 60 images similar to Fig. 2a. For this reason, manual generation of such masks in 45 minutes was not feasible in ARC' 17 . In order to automate the labeling process, the traditional approaches such as background subtraction and depth segmentation are not sufficient because they involve a number of hyper-parameters such as color difference thresholds, depth threshold, and kernel size of morphological operations. It becomes quite challenging to tune these parameters for uncontrolled scenarios such as lighting variations, reflections and the items which have transparent or wired mesh surfaces, or appears similar to a given background.

Hence, we devise an alternative approach which addresses all of the above issues at once and produces high quality mask and bounding box annotations in real time 2 , with no manual supervision. The approach is inspired by the outstanding capabilities of humans to detect and segment novel objects. It encourages us to assume that a part of human vision system possibly functions in a class agnostic manner, i.e. detection and segmentation are done irrespective of the category. To replicate this behavior, we collect single instance images (Fig. 2a) and use them in order to train two CNNs: one for semantic segmentation and another for bounding box regression. We refer these CNNs as tutors. The detailed procedure to train the tutors is given in Sec. V.

1 not to be confused with teacher-student networks or Generative Adversarial Networks [20]

2 depending on hardware configuration

Fig. 2: (a) Single instance images, (b) sample predicted mask ( ) and box ( ), and the combined output ( ) for various cases

<!-- image -->

The tutors can successfully detect and segment transparent objects as long as they are differentiable in the image by visual inspection. We exploit comprehensive data augmentation (Sec. V) and the property of CNNs to learn contextual information in order to enable the tutors to handle background color changes as well as the cases of color similarity between the object and the background. The approach doesn't require any background modeling unlike the background and depth subtraction. For this reason, it can be employed in the warehouses or any other relevant area without a need to model the background each time. The effectiveness of the tutors can be examined by Fig. 3 which shows qualitative results of the traditional approaches and the tutors to annotate variety of items i.e. transparent (rows 2 -5 ), mesh (row 1 ).

Further, we combine the predicted mask and the box in order to produce more robust ground truths, especially when either or both of the annotations are not accurate (Fig. 2b). Let p m , p b be the priorities of mask and box annotations. It must be noticed that the priorities are merely fixed numbers e.g. { p m , p b } = { 1 , 0 } , { 0 , 1 } or { 1 , 1 } . These values repersents three configurations given below.

- 1) if p m &lt; p b , final mask is given by the box b .
- 2) if p m = p b , it is given by box m ∩ box b , and
- 3) if p m &gt; p b , it is given by box m .

Where box m be the bounding box around the predicted mask and box b be a predicted box annotation respectively. Later, the obtained class agnostic annotations are assigned a physical label such as a box, crayons or bottle etc., provided by a human. For this reason, we call it semi-supervised labeling as the mask or box is generated by the tutor while a meaningful label is provided by human.

## B. Occlusion Aware Scene Synthesis

Typically, the performance of CNNs heavily relies on the nature of a dataset. Therefore, a CNN gets biased when trained only for single instance images (Fig. 2a) and exhibits poor performance on the cluttered scenes. Hence, cluttered images must be a part of the dataset in order to improve the performance. However, the cost and time to manually annotate such images, is directly related to the number of classes and instances per image. For this reason, the collection and manual labeling of such images is infeasible in short durations, similar to ARC' 17 . Hence, we develop an effective occlusion aware clutter synthesizing technique which can generate large number of realistic multi-class cluttered scenes (Fig. 4) along with ground-truth labels from only a few samples of the images similar to Fig. 2a.

Fig. 3: (a) Input image, (b) ground truth mask. Mask generation using (c) background subtraction, (d) depth information. (e) Semi supervised labeling for mask, and (f) box labeling.

<!-- image -->

Fig. 4: Generated synthetic clutter and color coded labels

<!-- image -->

Let there be a set of K classes and each class c ∈ K has a total of I c images. To generate synthetic clutter, an image is randomly chosen from the dataset or sample background images (if any). We refer it as a base-image . The baseimage is then partitioned into a grid of size M × M . The value of M is selected from a set of predefined values and governs the clutter level in the output image. For ARC' 17 , we use low, medium and high clutter having grid size of 3 × 3 , 4 × 4 , 5 × 5 respectively. For each grid center, a category c ∈ K and image I ∈ I c is selected randomly. The pixels belonging to class c in the image I are rendered onto the base-image such that center of the object to be rendered coincides with the grid center. In this process, a corresponding ground truth image is also updated parallely.

The above naive way of generating synthetic clutter often leads to situations where a significant part of an item is occluded and the remaining visible part doesn't contain enough contextual information. Such parts are redundant and might resemble with other objects and degrades the segmentation performance. Thus, we eliminate the object or its part in the clutter, which is below a visibility threshold. It is done by keeping a track of number of pixel of an object before copying and after clutter generation. The works [17] and [21] are an example of coexisting work during the timeline of ARC' 17 , however, our approach is entirely different from them.

## C. Customized CNN Head

Generation of unique and separable features has been one of the aims of the machine learning algorithms since separablility in the features improves the classification accuracies. Hence, we reduce the challenge of quick learning, to embed uniqueness in the features at the deeper layers of a CNN. To achieve this, we develop a CNN head based on a feature pyramid module [22] and insert a few branches to learn contextual information. Fig. 5a shows the child CNN with ResNet50 [1] as its backbone along with the proposed head. To embed uniqueness, we merge features from the last four stages of the ResNet50 i.e. conv 2 3 , conv 3 4 , conv 4 6 , and conv 5 3 . The first stage conv 1 3 avoided due to overly large GPU memory footprints. The feature merging is achieved by three kinds of blocks discussed below.

- 1) Feature Smoothing Block : FSB is a stack of 1 × 1 Convolution -Batch Normalization -ReLU. This block reduces the dimensionality or smoothen the input features.
- 2) Feature Interpolator Block : At each stage of ResNet50 , the spatial size of feature is reduced by half in order to increase the receptive field. Hence, output features of the deeper layers need to be upsampled in order to merge them with the features produced by shallower layers. FIB perform this operation by using bilinear interpolation. Although, a deconvolution layer can also be used for this purpose, we avoid its use because of extra learnable parameters.
- 3) Context Extraction Block : Utility of this block is to increase feature separability by gathering various levels of context. This block performs a concatenation operation which combines diverse features from earlier stages to embed uniqueness in features (Fig. 5a).

## D. Instance Detection and Segmentation

The semantic segmentation approach assigns a common label to all instances of a class and doesn't differentiate between them. In ARC' 17 , all items had exactly one instance and therefore, semantic segmentation was sufficient for recognition purpose by treating each item as a different category. However, multiple instances are quite common practically. In such cases, The Mask-RCNN [9] can not be employed because the box regression is hard to achieve in short durations ( ∼ 30 minutes) of training as compared to semantic segmentation and is also computationally slower. Hence, we extend our system to perform instance detection and segmentation while adding minimal computational overhead.

To achieve this, first, pixel-wise semantic labels are for an input image are obtained by the child. For each item class, pixel-wise masked images are computed using the predicted labels i.e. n masked images for n classes. All such images are then fed to the tutor network which predicts bounding boxes for all instances of the class, similar to Fig. 3f. The predicted boxes serves as the rectangular boundaries for each instance while the labels predicted by the child serves as the segmentation mask (Fig. 7).

Fig. 5: (a) Child with the customized CNN head, (b) state transition in the proposed online learning system

<!-- image -->

## IV. ONLINE LEARNING

In an offline learning system, first, data is collected, labeled and then learning is performed. This approach, in the warehouse automation, is not feasible as novel items keep on arriving frequently and it becomes difficult to incorporate them into the existing learned models. For example, let 20 items have already been learned by the system. In order to learn an additional item, an offline system would require to first generate the ground truth and restart the training process. Hence, we improve our system to learn in an online manner 3 , i.e. the system can acquire the images, label them, generate synthetic scenes and perform the learning process, altogether. Our system doesn't encounter catastrophic forgetting [23] of already learned items due to inclusion of all the available items through the synthesized clutter.

To realize a practical online learning system, we interconnect all the system components and deploy them on a Multi-GPU server having 8 × NVIDIA Geforce GTX-1080Ti, 256 GB RAM and 2 × Intel Xeon(R)-E5-2683-v4 CPUs, each having 16 physical cores. We use Caffe [24] with C++ architecture and extensively modify its core internals to support the proposed online learning functionality. Our implementation of the learning process (Fig. 5b) is complex synchronization of multiple threads and can be divided mainly into four stages.

- 1) Image Acquisition : The image acquisition process is facilitated by a rotating platform, equipped with 5 × FOSCAM FI9903P HD RGB LAN cameras mounted with different viewing angle. We divide one revolution ( ∼ 10 sec ) into 12 parts which results in 12 × 5 = 60 images per revolution. For each object, we repeat the process for two revolutions, in order to capture all views of an item. With these settings, image can be collected at a rate of ∼ 360

TABLE I: Timing analysis of online learning process

| Angle resolution                    | 60 ◦            | 30 ◦            | 20 ◦            | 10 ◦            |
|-------------------------------------|-----------------|-----------------|-----------------|-----------------|
| Acquired Images (per rev ∼ 10 sec)  | 30              | 60              | 108             | 120             |
| Frame grabbing (per image)          | 40 ms ( 25 FPS) | 40 ms ( 25 FPS) | 40 ms ( 25 FPS) | 40 ms ( 25 FPS) |
| Communication overhead ( 5 cameras) | 90 ms ( 62 FPS) | 90 ms ( 62 FPS) | 90 ms ( 62 FPS) | 90 ms ( 62 FPS) |
| Tutor-Box annotation (per image)    | 70 ms ( 14 FPS) | 70 ms ( 14 FPS) | 70 ms ( 14 FPS) | 70 ms ( 14 FPS) |
| Turor-Mask annotation (per image)   | 120 ms ( 8 FPS) | 120 ms ( 8 FPS) | 120 ms ( 8 FPS) | 120 ms ( 8 FPS) |
| Child learning time (per image)     | 270 ms ( 4 FPS) | 270 ms ( 4 FPS) | 270 ms ( 4 FPS) | 270 ms ( 4 FPS) |

TABLE II: Timing analysis of occlusion aware synthetic clutter

| Grid size   | Time per operation (ms)   | Time per operation (ms)   | Time per operation (ms)   | Total time (ms)   | Total time (ms)     |
|-------------|---------------------------|---------------------------|---------------------------|-------------------|---------------------|
| Grid size   | image decoding            | object transfer           | visibility check          | disk read         | prefetch all in RAM |
| 3 × 3       | 180                       | 63                        | 69                        | 320               | 135                 |
| 4 × 4       | 272                       | 112                       | 108                       | 503               | 220                 |
| 5 × 5       | 475                       | 150                       | 140                       | 767               | 290                 |

images per minute (Table I). Alternative to the platform, our system can also be taught by showing an item to the camera by hand.

2) Ground Truth Generation : The acquired images are sent to the tutor to obtain corresponding class agnostic mask and box annotations. These annotations are combined with priorities p m = p b (Fig. 2b) to obtain a final mask which is later filled with a numeric-id (class label), either provided by a human or computerized file storages. Due to real time speed, the ground truth is generated as soon as the images are acquired. To accomplish the same task, it would require approximately 1 -2 hours of human effort. Our system is capable of generating fully annotated images at 10 -15 FPS (Table I). This speed can be increased by using highspeed cameras to avoid motion blur occurring due to rotating platform.

3) Clutter Synthesis : The images and corresponding ground truths obtained from the previous step are used to synthesize cluttered scenes. A total of 24 threads are responsible and all of them remain live as long as the training runs. The clutter is generated at a rate of ∼ 5 × 24 = 120 FPS and in this process, most of the time is spent in the image decoding. It can be reduced by prefetching all available images in RAM and continuously discarding the clutter images which have been used in the training process. This increases scene synthesis throughput (Table II).

4) Child Learning : The child CNN is replicated across all the available GPUs. The child is fed with both the single-class (obtained from platform) and multi-class images (cluttered scenes). The selection between single-class and multi-class images is done at random with a probability ratio of 1 : 3 . The timing performance of the child learning is provided in the Table I. On the specified server, It can learn at 32 FPS i.e. 4 × 8 (Batch size × GPUs)

## V. EXPERIMENTS

## A. Dataset

We collect 12000 single instance images of 40 items provided by Amazon a priori, referred as known-set . Then we manually generate their mask and box annotations. During ARC' 17 , each task had a competition-set comprising of known and novel items divided equally. The stow-task 20 , pick-task 32 and final stow-pick task had 32 items. The images of novel items were collected during 45 minutes. As mentioned previously, 120 images are obtained for two revolution of the platform and each was rotated by -10 ◦ and 10 ◦ in order to approximately match the already collected 300 images per known item.

TABLE III: Performance analysis of the Tutors

| Augmentation   | Augmentation   | Augmentation   | Augmentation   | Augmentation   | Mask   | Box    |
|----------------|----------------|----------------|----------------|----------------|--------|--------|
| colour         | scale          | mirror         | blur           | rotate         | mIoU   | mIoU   |
| 7              | 7              | 7              | 7              | 7              | 45 . 9 | 33 . 1 |
| 3              |                |                |                |                | 80 . 9 | 77 . 3 |
| 3              | 3              |                |                |                | 89 . 4 | 81 . 6 |
| 3              | 3              | 3              |                |                | 90 . 2 | 84 . 9 |
| 3              | 3              | 3              | 3              |                | 92 . 7 | 86 . 8 |
| 3              | 3              | 3              | 3              | 3              | 94 . 6 | 87 . 6 |

TABLE IV: Performance analysis of the Child and state-of-the-art

| Clutter level   | Clutter level   | mIoU       | mIoU          | mIoU        | mIoU        |
|-----------------|-----------------|------------|---------------|-------------|-------------|
| 3 × 3           | 4 × 4 5 × 5     | η = 0 . 01 | η = 0 . 01    | η = 0 . 001 | η = 0 . 001 |
|                 |                 | PSPNet     | Child         | PSPNet      | Child       |
| 7               | 7               | 7 51 .     | 2 51 . 4      | 59 . 9      | 77 . 8      |
| 3               |                 | 52 .       | 3 55 . 7      | 62 . 9      | 79.5        |
| 3               | 3               |            | 58 . 1 65 . 3 | 73 . 7      | 84 . 1      |
| 3               | 3               | 3 62 .     | 8 90 . 2      | 84 . 8      | 87 . 2      |

## B. Semi Supervised Labeling

We split the set of 12000 images into two sets of 8000 and 4000 i.e. 200 train and 100 test images per item. We train the network architecture (Fig. 5a) for class agnostic mask and Single-Shot-Multi-Box-Detector (SSD) [25] for box annotation. For both of them, we use comprehensive data-augmentation i.e. random hue, saturation, brightness, and contrast all with selection probability of 0 . 5 , random rotation between -10 ◦ to 10 ◦ , Gaussian blur σ = 3 and a crop size of 512 × 512 . The training hyper-parameters are set to learning rate ( η ) = 0 . 001 , learning rate policy = step , gamma = 0 . 1 , momentum = 0 . 90 and weight decay = 0 . 0001 .

In general, mAP score is preferred to asses the bounding box prediction accuracy. However, in our case, we are more interested in assessing the overlapping of predicted and ground truth boxes due to presence of only one class (item). Thus, instead of mAP, we report mIoU [26] score for bounding box prediction and same is also reported for the semantic segmentation. The impact of comprehensive dataaugmentation on the tutor can be seen clearly in Table III.

## C. Clutter Synthesis and Quick Child Learning

We capture 200 real cluttered images and manually annotate them to evaluate the performance of quick learning for semantic segmentation. Our focus in this experiment remains on quick learning and therefore we don't involve analysis on large datasets (e.g. [27]). We evaluate the child network against the state-of-art PSPNet [7] by training both of them for 35 minutes with a batch size of 5 on the given computing platform. We freeze the Batch-Normalization [28] parameters of the back-bone while they are learnt for all the BatchNormalization layers of the CNN head. We set the training hyper parameters learning rate policy = step , gamma = 0 . 1 , momentum = 0 . 90 and weight decay = 0 . 0001 and use multinomial softmax loss in which multiple classes compete against each other. The ResNet50 backbone is pretrained for class agnostic segmentation (tutor). To analyze the effect of synthetic clutter and learning rate, we adopt two values of learning rates ( η ) = { 0 . 01 , 0 . 001 } and for each of them, the child and PSPNet are trained for varying amounts of clutter.

Fig. 6: (c)-(f) training without synthetic clutter, and (g)-(i) with synthetic clutter.

<!-- image -->

Table IV shows the mIoU of both the architectures on the mentioned real cluttered images. It can be seen that higher learning rate has adverse effects on PSPNet whereas the child network performs significantly better. With synthetic cluttering, both the networks performs well, however the child network outperforms PSPNet with a visible margin. A grid size of 3 × 3 doesn't adds much onto the accuracy, because all the items in the cluttered images are almost isolated. As the grid size is increased, actual cluttered and occlusion situation appears in the cluttered images, resulting in improved mIoU scores, which can also be verified qualitatively by Fig. 6. All the segmentation masks are are thresholded at 90% confidence in order to demonstrate, how quick the network can achieve higher confidences. Row2 column-(i) marks the presence of small misclassified patches in the output of PSPNet, whereas these are rarely present in the case of child.

The quantitative evaluation of instance detection remains same as that of Table III because the tutor for box-annotation is employed for instance detection. Fig. 7 shows the qualitative results for instance detection and segmentation. The images were acquired during our actual stow and pick task runs in ARC' 17 .

## D. Amazon Robotics Challenge, 2017

1) Suppressed Misclassification and Open Workspace : The on-spot training was done for novel items and the known items, only in the competition set. This strategy allowed the child to penalize the loss function for each item, approximately in a uniform manner. On the other hand, the teams who used their networks pretrained on known items had faced issue of small misclassified patches. It happened due to biasing of their networks towards known items which lead to confusion of novel with known items. In our case, such patches were significantly suppressed and were observed only once during the pick task. Moreover, our system was also robust to ambient lights which allowed us to keep the workspace open and unconstrained (Fig. 1) in contrast to the other teams [17], [18], [15].

Fig. 7: Instance detection and segmentation results on the images collected during our ARC' 17 competition runs

<!-- image -->

- 2) Inspection Free Self Learning System : The improved system starts learning as soon as an item is placed on the platform. It continuously monitors the data acquisition process to examine the number of items processed and generate synthetic clutter only for the items whose images and ground truths are available. In contrast, our vision system in ARC' 17 coudn't take advantage of the instance detection and online learning. The clutter generation process was, however, runtime, i.e. child never encounters a cluttered image twice. Typically during a training run, the child could learn approximately ∼ 67000 images in 35 minutes, which is near real time. In addition, all the components of our system were inspection free in contrast to the Team Nimbro and ACRV who manually monitored their data acquisition process and performed correction in case of erroneous ground truth segmentation masks.
- 3) Statistical Analysis with other teams : Our system generated data for itself and exhibited no mis-labeling of an item, just in 45 minutes. Due to accurate visual perception, we achieved highest grasping accuracies, even more than the winners. Table-V shows the item grasp success rate for top-5 teams in ARC' 17 . Our team is highlighted in blue.

TABLE V: Grasping performance of the Top5 teams in ARC' 17

| Team           | Grasp Success Rate   | Grasp Success Rate   | Grasp Success Rate   |
|----------------|----------------------|----------------------|----------------------|
|                | Stow task            | Pick task            | Final task           |
| ACRV           | 58 . 00 %            | 66 . 00 %            | 62 . 50 %            |
| NimbRo Picking | 11 . 11 %            | 68 . 40 %            | 56 . 80 %            |
| Nanyang        | 38 . 80 %            | 100 . 0 %            | 53 . 80 %            |
| IITK-TCS       | 78 . 26 %            | 100 . 0 %            | 79 . 20 %            |
| MIT-Princeton  | 59 . 37 %            | 39 . 00 %            | 64 . 70 %            |

## VI. CONCLUSION

This work introduces real-time technique to autonomously generate high quality box and mask ground truths. It also introduces an occlusion aware clutter synthesis technique which eliminates the need of exhaustive tasks to manually annotate cluttered (multi-class) images. The customized CNN head achieves high accuracies for segmentation in short durations. The learning framework leverages above techniques and have shown outstanding performance in ARC' 17 . After the challenge, the system has been also enabled to detect multiple instances of an item as well as it can learn online, i.e. both labeled data generation and learning happens simultaneously in real time.

## REFERENCES

- [1] K. He, X. Zhang, S. Ren, and J. Sun, 'Deep residual learning for image recognition,' in Proceedings of the IEEE conference on computer vision and pattern recognition , pp. 770-778, 2016.
- [2] K. Simonyan and A. Zisserman, 'Very deep convolutional networks for large-scale image recognition,' CoRR , vol. abs/1409.1556, 2014.
- [3] S. Ren, K. He, R. Girshick, and J. Sun, 'Faster r-cnn: Towards realtime object detection with region proposal networks,' in Advances in neural information processing systems , pp. 91-99, 2015.
- [4] R. Girshick, 'Fast r-cnn,' in Proceedings of the IEEE international conference on computer vision , pp. 1440-1448, 2015.
- [5] C. Zhu, Y. Zheng, K. Luu, and M. Savvides, 'Cms-rcnn: contextual multi-scale region-based cnn for unconstrained face detection,' in Deep Learning for Biometrics , pp. 57-79, Springer, 2017.
- [6] J. Long, E. Shelhamer, and T. Darrell, 'Fully convolutional networks for semantic segmentation,' in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pp. 3431-3440, 2015.
- [7] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia, 'Pyramid scene parsing network,' in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2017.
- [8] B. Zhou, H. Zhao, X. Puig, S. Fidler, A. Barriuso, and A. Torralba, 'Scene parsing through ade20k dataset,' in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , 2017.
- [9] K. He, G. Gkioxari, P. Doll´ ar, and R. Girshick, 'Mask r-cnn,' CVPR , 2017.
- [10] G. Lin, A. Milan, C. Shen, and I. D. Reid, 'Refinenet: Multi-path refinement networks for high-resolution semantic segmentation.,' in Cvpr , vol. 1, p. 5, 2017.
- [11] R. Jonschkowski, C. Eppner, S. H¨ ofer, R. Mart´ ın-Mart´ ın, and O. Brock, 'Probabilistic multi-class segmentation for the amazon picking challenge,' in Intelligent Robots and Systems (IROS), 2016 IEEE/RSJ International Conference on , pp. 1-7, IEEE, 2016.
- [12] C. Hernandez, M. Bharatheesha, W. Ko, H. Gaiser, J. Tan, K. van Deurzen, M. de Vries, B. Van Mil, J. van Egmond, R. Burger, et al. , 'Team delft's robot winner of the amazon picking challenge 2016,' in Robot World Cup , pp. 613-624, Springer, 2016.
- [13] A. Zeng, K.-T. Yu, S. Song, D. Suo, E. Walker, A. Rodriguez, and J. Xiao, 'Multi-view self-supervised deep learning for 6d pose estimation in the amazon picking challenge,' in Robotics and Automation (ICRA), 2017 IEEE International Conference on , pp. 1386-1383, IEEE, 2017.
- [14] M. Schwarz, A. Milan, C. Lenz, A. Munoz, A. S. Periyasamy, M. Schreiber, S. Sch¨ uller, and S. Behnke, 'Nimbro picking: Versatile part handling for warehouse automation,' in Robotics and Automation (ICRA), 2017 IEEE International Conference on , pp. 3032-3039, IEEE, 2017.
- [15] A. Zeng, S. Song, K.-T. Yu, E. Donlon, F. R. Hogan, M. Bauza, D. Ma, O. Taylor, M. Liu, E. Romo, N. Fazeli, F. Alet, N. C. Dafle, R. Holladay, I. Morona, P. Q. Nair, D. Green, I. Taylor, W. Liu, T. Funkhouser, and A. Rodriguez, 'Robotic pick-and-place of novel objects in clutter with multi-affordance grasping and cross-domain image matching,' in Proceedings of the IEEE International Conference on Robotics and Automation , 2018.
- [16] B. Kumar, G. Carneiro, I. Reid, et al. , 'Learning local image descriptors with deep siamese and triplet convolutional networks by minimising global loss functions,' in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pp. 5385-5394, 2016.
- [17] M. Schwarz, C. Lenz, G. M. Garcıa, S. Koo, A. S. Periyasamy, M. Schreiber, and S. Behnke, 'Fast object learning and dual-arm coordination for cluttered stowing, picking, and packing,' in IEEE International Conference on Robotics and Automation (ICRA) , 2018.
- [18] A. Milan, T. Pham, K. Vijay, D. Morrison, A. Tow, L. Liu, J. Erskine, R. Grinover, A. Gurman, T. Hunn, N. Kelly-Boxall, D. Lee, M. McTaggart, G. Rallos, A. Razjigaev, T. Rowntree, T. Shen, R. Smith, S. Wade-McCue, and J. Leitner, 'Semantic segmentation from limited training data,' 09 2017.
- [19] D. Morrison, A. W. Tow, M. McTaggart, R. Smith, N. Kelly-Boxall, S. Wade-McCue, J. Erskine, R. Grinover, A. Gurman, T. Hunn, D. Lee, A. Milan, T. Pham, G. Rallos, A. Razjigaev, T. Rowntree, K. Vijay, Z. Zhuang, C. F. Lehnert, I. D. Reid, P. Corke, and J. Leitner, 'Cartman: The low-cost cartesian manipulator that won the amazon robotics challenge,' 2017.
- [20] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, 'Generative adversarial nets,' in Advances in neural information processing systems , pp. 2672-2680, 2014.
- [21] D. Dwibedi, I. Misra, and M. Hebert, 'Cut, paste and learn: Surprisingly easy synthesis for instance detection,' in The IEEE international conference on computer vision (ICCV) , 2017.
- [22] T.-Y. Lin, P. Doll´ ar, R. Girshick, K. He, B. Hariharan, and S. Belongie, 'Feature pyramid networks for object detection,' CVPR , 2017.
- [23] R. M. French, 'Catastrophic forgetting in connectionist networks,' Trends in cognitive sciences , vol. 3, no. 4, pp. 128-135, 1999.
- [24] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, and T. Darrell, 'Caffe: Convolutional architecture for fast feature embedding,' in Proceedings of the 22nd ACM international conference on Multimedia , pp. 675-678, ACM, 2014.
- [25] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg, 'Ssd: Single shot multibox detector,' in European conference on computer vision , pp. 21-37, Springer, 2016.
- [26] M. Everingham, L. Van Gool, C. K. Williams, J. Winn, and A. Zisserman, 'The pascal visual object classes (voc) challenge,' International journal of computer vision , vol. 88, no. 2, pp. 303-338, 2010.
- [27] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, 'The cityscapes dataset for semantic urban scene understanding,' in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2016.
- [28] S. Ioffe and C. Szegedy, 'Batch normalization: Accelerating deep network training by reducing internal covariate shift,' in International Conference on Machine Learning , pp. 448-456, 2015.
- [29] R. Collobert, S. Bengio, and J. Mari´ ethoz, 'Torch: a modular machine learning software library,' tech. rep., Idiap, 2002.
- [30] J. Bergstra, O. Breuleux, F. Bastien, P. Lamblin, R. Pascanu, G. Desjardins, J. Turian, D. Warde-Farley, and Y. Bengio, 'Theano: A cpu and gpu math compiler in python,' in Proc. 9th Python in Science Conf , pp. 1-7, 2010.
- [31] G. Huang, Z. Liu, K. Q. Weinberger, and L. van der Maaten, 'Densely connected convolutional networks,' CVPR , 2017.
- [32] C. Rother, V. Kolmogorov, and A. Blake, 'Grabcut: Interactive foreground extraction using iterated graph cuts,' in ACM transactions on graphics (TOG) , vol. 23, pp. 309-314, ACM, 2004.