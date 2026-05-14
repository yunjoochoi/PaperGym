## Self-Supervised Learning of Pretext-Invariant Representations

Ishan Misra Laurens van der Maaten Facebook AI Research

Figure 1: Pretext-Invariant Representation Learning (PIRL). Many pretext tasks for self-supervised learning [18, 46, 76] involve transforming an image I , computing a representation of the transformed image, and predicting properties of transformation t from that representation. As a result, the representation must covary with the transformation t and may not contain much semantic information. By contrast, PIRL learns representations that are invariant to the transformation t and retain semantic information.

<!-- image -->

the pretext task involves predicting a property of the image transformation, it encourages the construction og image representations that are covariant to the transformations. Although such covariance is beneficial for tasks such as predicting 3D correspondences [31, 49, 57], it is undesirable for most semantic recognition tasks. Representations ought to be invariant under image transformations to be useful for image recognition [13, 29] because the transformations do not alter visual semantics.

Motivated by this observation, we propose a method that learns invariant representations rather than covariant ones. Instead of predicting properties of the image transformation, Pretext-Invariant Representation Learning (PIRL) constructs image representations that are similar to the representation of transformed versions of the same image and different from the representations of other images. We adapt the 'Jigsaw' pretext task [46] to work with PIRL and find that the resulting invariant representations perform better than their covariant counterparts across a range of vision tasks. PIRL substantially outperforms all prior art in selfsupervised learning from ImageNet (Figure 2) and from uncurated image data (Table 4). Interestingly, PIRL even outperforms supervised pre-training in learning image representations suitable for object detection (Tables 1 &amp; 6).

## Abstract

The goal of self-supervised learning from images is to construct image representations that are semantically meaningful via pretext tasks that do not require semantic annotations for a large training set of images. Many pretext tasks lead to representations that are covariant with image transformations. We argue that, instead, semantic representations ought to be invariant under such transformations. Specifically, we develop Pretext-Invariant Representation Learning (PIRL, pronounced as 'pearl') that learns invariant representations based on pretext tasks. We use PIRL with a commonly used pretext task that involves solving jigsaw puzzles. We find that PIRL substantially improves the semantic quality of the learned image representations. Our approach sets a new state-of-the-art in selfsupervised learning from images on several popular benchmarks for self-supervised learning. Despite being unsupervised, PIRL outperforms supervised pre-training in learning image representations for object detection. Altogether, our results demonstrate the potential of self-supervised learning of image representations with good invariance properties.

## 1. Introduction

Modern image-recognition systems learn image representations from large collections of images and corresponding semantic annotations. These annotations can be provided in the form of class labels [58], hashtags [41], bounding boxes [15, 39], etc. Pre-defined semantic annotations scale poorly to the long tail of visual concepts [66], which hampers further improvements in image recognition.

Self-supervised learning tries to address these limitations by learning image representations from the pixels themselves without relying on pre-defined semantic annotations. Often, this is done via a pretext task that applies a transformation to the input image and requires the learner to predict properties of the transformation from the transformed image (see Figure 1). Examples of image transformations used include rotations [18], affine transformations [31, 49, 57, 76], and jigsaw transformations [46]. As

Figure 2: ImageNet classification with linear models. Single-crop top-1 accuracy on the ImageNet validation data as a function of the number of parameters in the model that produces the representation ('A' represents AlexNet). Pretext-Invariant Representation Learning (PIRL) sets a new state-of-the-art in this setting (red marker) and uses significantly smaller models (ResNet-50). See Section 3.2 for more details.

<!-- image -->

## 2. PIRL: Pretext-Invariant Representation Learning

Our work focuses on pretext tasks for self-supervised learning in which a known image transformation is applied to the input image. For example, the 'Jigsaw' task divides the image into nine patches and perturbs the image by randomly permuting the patches [46]. Prior work used Jigsaw as a pretext task by predicting the permutation from the perturbed input image. This requires the learner to construct a representation that is covariant to the perturbation. The same is true for a range of other pretext tasks that have recently been studied [9, 18, 44, 76]. In this work, we adopt the existing Jigsaw pretext task in a way that encourages the image representations to be invariant to the image patch perturbation. While we focus on the Jigsaw pretext task in this paper, our approach is applicable to any pretext task that involves image transformations (see Section 4.3).

## 2.1. Overview of the Approach

Suppose we are given an image dataset, D = { I 1 , . . . , I |D| } with I n ∈ R H × W × 3 , and a set of image transformations, T . The set T may contain transformations such as a re-shuffling of patches in the image [46], image rotations [18], etc . We aim to train a convolutional network, φ θ ( · ) , with parameters θ that constructs image representations v I = φ θ ( I ) that are invariant to image transformations t ∈ T . We adopt an empirical risk minimization approach to learning the network parameters θ . Specifically, we train the network by minimizing the empirical risk:

<!-- formula-not-decoded -->

where p ( T ) is some distribution over the transformations in T , and I t denotes image I after application of transformation t , that is, I t = t ( I ) . The function L ( · , · ) is a loss function that measures the similarity between two image representations. Minimization of this loss encourages the network φ θ ( · ) to produce the same representation for image I as for its transformed counterpart I t , i.e. , to make representation invariant under transformation t .

We contrast our loss function to losses [9, 18, 44, 46, 76] that aim to learn image representations v I = φ θ ( I ) that are covariant to image transformations t ∈ T by minimizing:

<!-- formula-not-decoded -->

where z is a function that measures some properties of transformation t . Such losses encourage network φ θ ( · ) to learn image representations that contain information on transformation t , thereby encouraging it to maintain information that is not semantically relevant.

glyph[negationslash]

Loss function. We implement glyph[lscript] inv ( · ) using a contrastive loss function L ( · , · ) [22]. Specifically, we define a matching score, s ( · , · ) , that measures the similarity of two image representations and use this matching score in a noise contrastive estimator [21]. In our noise contrastive estimator (NCE), each 'positive' sample ( I , I t ) has N corresponding 'negative' samples. The negative samples are obtained by computing features from other images, I ′ = I . The noise contrastive estimator models the probability of the binary event that ( I , I t ) originates from data distribution as:

<!-- formula-not-decoded -->

Herein, D N ⊆ D \ { I } is a set of N negative samples that are drawn uniformly at random from dataset D excluding image I , τ is a temperature parameter, and s ( · , · ) is the cosine similarity between the representations.

In practice, we do not use the convolutional features v directly but apply to different 'heads' to the features before computing the score s ( · , · ) . Specifically, we apply head f ( · ) on features ( v I ) of I and head g ( · ) on features ( v I t ) of I t ; see Figure 3 and Section 2.3. NCE then amounts to minimizing the following loss:

<!-- formula-not-decoded -->

This loss encourages the representation of image I to be similar to that of its transformed counterpart I t , whilst also encouraging the representation of I t to be dissimilar to that of other images I ′ .

<!-- image -->

M

Figure 3: Overview of PIRL. Pretext-Invariant Representation Learning (PIRL) aims to construct image representations that are invariant to the image transformations t ∈ T . PIRL encourages the representations of the image, I , and its transformed counterpart, I t , to be similar. It achieves this by minimizing a contrastive loss (see Section 2.1). Following [72], PIRL uses a memory bank, M , of negative samples to be used in the contrastive learning. The memory bank contains a moving average of representations, m I ∈ M , for all images in the dataset (see Section 2.2).

## 2.2. Using a Memory Bank of Negative Samples

Prior work has found that it is important to use a large number of negatives in the NCE loss of Equation 4 [51, 72]. In a mini-batch SGD optimizer, it is difficult to obtain a large number of negatives without increasing the batch to an infeasibly large size. To address this problem, we follow [72] and use a memory bank of 'cached' features. Concurrent work used a similar memory-bank approach [24].

The memory bank, M , contains a feature representation m I for each image I in dataset D . The representation m I is an exponential moving average of feature representations f ( v I ) that were computed in prior epochs. This allows us to replace negative samples, f ( v ′ I ) , by their memory bank representations, m I ′ , in Equation 4 without having to increase the training batch size. We emphasize that the representations that are stored in the memory bank are all computed on the original images, I , without the transformation t .

Final loss function. A potential issue of the loss in Equation 4 is that it does not compare the representations of untransformed images I and I ′ . We address this issue by using a convex combination of two NCE loss functions in glyph[lscript] inv ( · ) :

<!-- formula-not-decoded -->

Herein, the first term is simply the loss of Equation 4 but uses memory representations m I and m I ′ instead of f ( v I ) and f ( v ′ I ) , respectively. The second term does two things: (1) it encourages the representation f ( v I ) to be similar to its memory representation m I , thereby dampening the parameter updates; and (2) it encourages the representations f ( v I ) and f ( v ′ I ) to be dissimilar. We note that both the first and the second term use m I ′ instead of f ( v ′ I ) in Equation 4. Setting λ = 0 in Equation 5 leads to the loss used in [72]. We study the effect of λ on the learned representations in Section 4.

## 2.3. Implementation Details

Although PIRL can be used with any pretext task that involves image transformations, we focus on the Jigsaw pretext task [46] in this paper. To demonstrate that PIRL is more generally applicable, we also experiment with the Rotation pretext task [18] and with a combination of both tasks in Section 4.3. Below, we describe the implementation details of PIRL with the Jigsaw pretext task.

Convolutional network. We use a ResNet-50 (R-50) network architecture in our experiments [25]. The network is used to compute image representations for both I and I t . These representations are obtained by applying function f ( · ) or g ( · ) on features extracted from the the network.

Specifically, we compute the representation of I , f ( v I ) , by extracting res5 features, average pooling, and a linear projection to obtain a 128 -dimensional representation.

To compute the representation g ( v I t ) of a transformed image I t , we closely follow [19, 46]. We: (1) extract nine patches from image I , (2) compute an image representation for each patch separately by extracting activations from the res5 layer of the ResNet-50 and average pool the activations, (3) apply a linear projection to obtain a 128 -dimensional patch representations, and (4) concatenate the patch representations in random order and apply a second linear projection on the result to obtain the final 128 -dimensional image representation, g ( v I t ) . Our motivation for this design of g ( v I t ) is the desire to remain as close as possible to the covariant pretext task of [18, 19, 46]. This allows apples-to-apples comparisons between the covariant approach and our invariant approach.

Hyperparameters. We implement the memory bank as described in [72] and use the same hyperparameters for the memory bank. Specifically, we set the temperature in Equation 3 to τ =0 . 07 , and use a weight of 0 . 5 to compute the exponential moving averages in the memory bank. Unless stated otherwise, we use λ = 0 . 5 in Equation 5.

## 3. Experiments

Following common practice in self-supervised learning [19, 78], we evaluate the performance of PIRL in transfer-learning experiments. We perform experiments on a variety of datasets, focusing on object detection and image classification tasks. Our empirical evaluations cover: (1) a learning setting in which the parameters of the convolutional network are fi netuned during transfer, thus evaluating the network 'initialization' obtained using self-supervised learning and (2) a learning setting in which the parameters of the network are fi xed during transfer learning, thus using the network as a feature extractor. Code reproducing the results of our experiments will be published online.

Baselines. Our most important baseline is the Jigsaw ResNet-50 model of [19]. This baseline implements the co- variant counterpart of our PIRL approach with the Jigsaw pretext task.

Table 1: Object detection on VOC07+12 using Faster R-CNN. Detection AP on the VOC07 test set after finetuning Faster R-CNN models (keeping BatchNorm fixed) with a ResNet-50 backbone pre-trained using self-supervised learning on ImageNet. Results for supervised ImageNet pre-training are presented for reference. Numbers with ∗ are adopted from the corresponding papers. Method with † finetunes BatchNorm. PIRL significantly outperforms supervised pre-training without extra pre-training data or changes in the network architecture. Additional results in Table 6.

| Method        | Network   | AP all   | AP 50   | AP 75   | ∆AP 75   |
|---------------|-----------|----------|---------|---------|----------|
| Supervised    | R-50      | 52.6     | 81.1    | 57.4    | =0.0     |
| Jigsaw [19]   | R-50      | 48.9     | 75.1    | 52.9    | -4.5     |
| Rotation [19] | R-50      | 46.3     | 72.5    | 49.3    | -8.1     |
| NPID++ [72]   | R-50      | 52.3     | 79.1    | 56.9    | -0.5     |
| PIRL (ours)   | R-50      | 54.0     | 80.7    | 59.7    | +2.3     |
| CPC-Big [26]  | R-101     | -        | 70.6 ∗  | -       |          |
| CPC-Huge [26] | R-170     | -        | 72.1 ∗  | -       |          |
| MoCo [24]     | R-50      | 55.2 ∗†  | 81.4 ∗† | 61.2 ∗† |          |

We also compare PIRL to a range of other selfsupervised methods. An important comparison is to NPID [72]. NPID is a special case of PIRL: setting λ = 0 in Equation 5 leads to the loss function of NPID. We found it is possible to improve the original implementation of NPID by using more negative samples and training for more epochs (see Section 4). We refer to our improved version of NPID as NPID++. Comparisons between PIRL and NPID++ allow us to study the effect of the pretext-invariance that PIRL aims to achieve, i.e. , the effect of using λ &gt; 0 in Equation 5. Pre-training data. To facilitate comparisons with prior work, we use the 1 . 28 M images from the ImageNet [58] train split (without labels) to pre-train our models.

Training details. We train our models using mini-batch SGD using the cosine learning rate decay [40] scheme with an initial learning rate of 1 . 2 × 10 -1 and a final learning rate of 1 . 2 × 10 -4 . We train the models for 800 epochs using a batch size of 1 , 024 images and using N = 32 , 000 negative samples in Equation 3. We do not use data-augmentation approaches such as Fast AutoAugment [38] because they are the result of supervised-learning approaches. We provide a full overview of all hyperparameter settings that were used in the supplemental material.

Transfer learning. Prior work suggests that the hyperparameters used in transfer learning can play an important role in the evaluation pre-trained representations [19, 33, 78]. To facilitate fair comparisons with prior work, we closely follow the transfer-learning setup described in [19, 78].

## 3.1. Object Detection

Following prior work [19, 72], we perform objectdetection experiments on the the Pascal VOC dataset [15] using the VOC07+12 train split. We use the Faster R- CNN [56] C4 object-detection model implemented in Detectron2 [71] with a ResNet-50 (R-50) backbone. We pretrain the ResNet-50 using PIRL to initialize the detection model before finetuning it on the VOC training data. We use the same training schedule as [19] for all models finetuned on VOC and follow [19, 71] to keep the BatchNorm parameters fixed during finetuning. We evaluate object-detection performance in terms of AP all , AP 50 , and AP 75 [39].

The results of our detection experiments are presented in Table 1. The results demonstrate the strong performance of PIRL: it outperforms all alternative self-supervised learnings in terms of all three AP measures. Compared to pretraining on the Jigsaw pretext task, PIRL achieves AP improvements of 5 points . These results underscore the importance of learning invariant (rather than covariant) image representations. PIRL also outperforms NPID++, which demonstrates the benefits of learning pretext invariance.

Interestingly, PIRL even outperforms the supervised ImageNet-pretrained model in terms of the more conservative AP all and AP 75 metrics. Similar to concurrent work [24], we find that a self-supervised learner can outperform supervised pre-training for object detection. We emphasize that PIRL achieves this result using the same backbone model, the same number of finetuning epochs, and the exact same pre-training data (but without the labels). This result is a substantial improvement over prior self-supervised approaches that obtain slightly worse performance than fully supervised baselines despite using orders of magnitude more curated training data [19] or much larger backbone models [26]. In Table 6, we show that PIRL also outperforms supervised pretraining when finetuning is done on the much smaller VOC07 train+val set. This suggests that PIRL learns image representations that are amenable to sample-efficient supervised learning.

## 3.2. Image Classification with Linear Models

Next, we assess the quality of image representations by training linear classifiers on fixed image representations. We follow the evaluation setup from [19] and measure the performance of such classifiers on four image-classification datasets: ImageNet [58], VOC07 [15], Places205 [79], and iNaturalist2018 [65]. These datasets involve diverse tasks such as object classification, scene recognition and finegrained recognition. Following [19], we evaluate representations extracted from all intermediate layers of the pretrained network, and report the image-classification results for the best-performing layer in Table 2.

ImageNet results. The results on ImageNet highlight the benefits of learning invariant features: PIRL improves recognition accuracies by over 15% compared to its covariant counterpart, Jigsaw. PIRL achieves the highest singlecrop top-1 accuracy of all self-supervised learners that use a single ResNet-50 model.

Table 2: Image classification with linear models. Image-classification performance on four datasets using the setup of [19]. We train linear classifiers on image representations obtained by self-supervised learners that were pre-trained on ImageNet (without labels). We report the performance for the best-performing layer for each method. We measure mean average precision (mAP) on the VOC07 dataset and top-1 accuracy on all other datasets. Numbers for PIRL, NPID++, Rotation were obtained by us; the other numbers were adopted from their respective papers. Numbers with † were measured using 10-crop evaluation. The best-performing self-supervised learner on each dataset is boldfaced .

| Method                                     | Parameters                                 | Transfer Dataset                           | Transfer Dataset                           | Transfer Dataset                           | Transfer Dataset                           |
|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|--------------------------------------------|
| ResNet-50 using evaluation setup of [19]   | ResNet-50 using evaluation setup of [19]   | ResNet-50 using evaluation setup of [19]   | ResNet-50 using evaluation setup of [19]   | ResNet-50 using evaluation setup of [19]   | ResNet-50 using evaluation setup of [19]   |
| Supervised                                 | 25.6M                                      | 75.9                                       | 87.5                                       | 51.5                                       | 45.4                                       |
| Colorization [19]                          | 25.6M                                      | 39.6                                       | 55.6                                       | 37.5                                       | -                                          |
| Rotation [18]                              | 25.6M                                      | 48.9                                       | 63.9                                       | 41.4                                       | 23.0                                       |
| NPID++ [72]                                | 25.6M                                      | 59.0                                       | 76.6                                       | 46.4                                       | 32.4                                       |
| MoCo [24]                                  | 25.6M                                      | 60.6                                       | -                                          | -                                          | -                                          |
| Jigsaw [19]                                | 25.6M                                      | 45.7                                       | 64.5                                       | 41.2                                       | 21.3                                       |
| PIRL ( ours )                              | 25.6M                                      | 63.6                                       | 81.1                                       | 49.8                                       | 34.1                                       |
| Different architecture or evaluation setup | Different architecture or evaluation setup | Different architecture or evaluation setup | Different architecture or evaluation setup | Different architecture or evaluation setup | Different architecture or evaluation setup |
| NPID [72]                                  | 25.6M                                      | 54.0                                       | -                                          | 45.5                                       | -                                          |
| BigBiGAN [12]                              | 25.6M                                      | 56.6                                       | -                                          | -                                          | -                                          |
| AET [76]                                   | 61M                                        | 40.6                                       | -                                          | 37.1                                       | -                                          |
| DeepCluster [6]                            | 61M                                        | 39.8                                       | -                                          | 37.5                                       | -                                          |
| Rot. [33]                                  | 61M                                        | 54.0                                       | -                                          | 45.5                                       | -                                          |
| LA [80]                                    | 25.6M                                      | 60.2 †                                     | -                                          | 50.2 †                                     | -                                          |
| CMC [64]                                   | 51M                                        | 64.1                                       | -                                          | -                                          | -                                          |
| CPC [51]                                   | 44.5M                                      | 48.7                                       | -                                          | -                                          | -                                          |
| CPC-Huge [26]                              | 305M                                       | 61.0                                       | -                                          | -                                          | -                                          |
| BigBiGAN-Big [12]                          | 86M                                        | 61.3                                       | -                                          | -                                          | -                                          |
| AMDIM [4]                                  | 670M                                       | 68.1                                       | -                                          | 55.1                                       | -                                          |

The benefits of pretext invariance are further highlighted by comparing PIRL with NPID. Our re-implementation of NPID (called NPID++) substantially outperforms the results reported in [72]. Specifically, NPID++ achieves a single-crop top-1 accuracy of 59% , which is higher or on par with existing work that uses a single ResNet-50. Yet, PIRL substantially outperforms NPID++. We note that PIRL also outperforms concurrent work [24] in this setting.

Akin to prior approaches, the performance of PIRL improves with network size. For example, CMC [64] uses a combination of two ResNet-50 models and trains the linear classifier for longer to obtain 64 . 1% accuracy. We performed an experiment in which we did the same for PIRL, and obtained a top-1 accuracy of 65 . 7% ; see 'PIRL-ens.' in Figure 2. To compare PIRL with larger models, we also performed experiments in which we followed [33, 75] by doubling the number of channels in ResNet-50; see 'PIRL-c2x' in Figure 2. PIRL-c2x achieves a top-1 accuracy of 67 . 4% , which is close to the accuracy obtained by AMDIM [4] with a model that has 6 × more parameters.

Altogether, the results in Figure 2 demonstrate that PIRL outperforms all prior self-supervised learners on ImageNet in terms of the trade-off between model accuracy and size. Indeed, PIRL even outperforms most self-supervised learn- ers that use much larger models [26, 51].

Table 3: Semi-supervised learning on ImageNet. Single-crop top-5 accuracy on the ImageNet validation set of self-supervised models that are finetuned on 1% and 10% of the ImageNet training data, following [72]. All numbers except for Jigsaw, NPID++ and PIRL are adopted from the corresponding papers. Best performance is boldfaced .

|                            | Data fraction → Backbone   | 1%             | 10%            |
|----------------------------|----------------------------|----------------|----------------|
| Method                     |                            | Top-5 Accuracy | Top-5 Accuracy |
| Random initialization [72] | R-50                       | 22.0           | 59.0           |
| NPID [72]                  | R-50                       | 39.2           | 77.4           |
| Jigsaw [19]                | R-50                       | 45.3           | 79.3           |
| NPID++ [72]                | R-50                       | 52.6           | 81.5           |
| VAT + Ent Min. [20, 45]    | R-50v2                     | 47.0           | 83.4           |
| S 4 L Exemplar [75]        | R-50v2                     | 47.0           | 83.7           |
| S 4 L Rotation [75]        | R-50v2                     | 53.4           | 83.8           |
| PIRL ( ours )              | R-50                       | 57.2           | 83.8           |
| Colorization [36]          | R-152                      | 29.8           | 62.0           |
| CPC-Largest [26]           | R-170 and R-11             | 64.0           | 84.9           |

Results on other datasets. The results on the other imageclassification datasets in Table 2 are in line with the results on ImageNet: PIRL substantially outperforms its covariant counterpart (Jigsaw). The performance of PIRL is within 2% of fully supervised representations on Places205, and improves the previous best results of [19] on VOC07 by more than 16 AP points. On the challenging iNaturalist dataset, which has over 8 , 000 classes, we obtain a gain of 11% in top-1 accuracy over the prior best result [18]. We observe that the NPID++ baseline performs well on these three datasets but is consistently outperformed by PIRL. Indeed, PIRL sets a new state-of-the-art for selfsupervised representations in this learning setting on the VOC07, Places205, and iNaturalist datasets.

## 3.3. Semi-Supervised Image Classification

We perform semi-supervised image classification experiments on ImageNet following the experimental setup of [26, 72, 75]. Specifically, we randomly select 1% and 10% of the ImageNet training data (with labels). We finetune our models on these training-data subsets following the procedure of [72]. Table 3 reports the top-5 accuracy of the resulting models on the ImageNet validation set.

The results further highlight the quality of the image representations learned by PIRL: finetuning the models on just 1% ( ∼ 13,000) labeled images leads to a top-5 accuracy of 57% . PIRL performs at least as well as S 4 L [75] and better than V AT [20], which are both methods specifically designed for semi-supervised learning. In line with earlier results, PIRL also outperforms Jigsaw and NPID++.

## 3.4. Pre-Training on Uncurated Image Data

Most representation learning methods are sensitive to the data distribution used during pre-training [19, 30, 41, 62]. To study how much changes in the data distribution im- pact PIRL, we pre-train models on uncurated images from the unlabeled YFCC dataset [63]. Following [7, 19], we randomly select a subset of 1 million images (YFCC-1M) from the 100 million images in YFCC. We pre-train PIRL ResNet-50 networks on YFCC-1M using the same procedure that was used for ImageNet pre-training. We evaluate using the setup in Section 3.2 by training linear classifiers on fixed image representations.

Table 4: Pre-training on uncurated YFCC images. Top-1 accuracy or mAP (for VOC07) of linear image classifiers for four image-classification tasks, using various image representations. All numbers (except those for PIRL) are adopted from the corresponding papers. Deep(er)Cluster uses VGG-16 rather than ResNet-50. The best performance on each dataset is boldfaced . Top: Representations obtained by training ResNet-50 models on a randomly selected subset of one million images. Bottom: Representations learned from about 100 million YFCC images.

| Method             | Dataset   | Transfer Dataset   | Transfer Dataset   | Transfer Dataset   | Transfer Dataset   |
|--------------------|-----------|--------------------|--------------------|--------------------|--------------------|
|                    |           | ImageNet           | VOC07              | Places205          | iNat.              |
| Jigsaw [19]        | YFCC1M    | -                  | 64.0               | 42.1               | -                  |
| DeepCluster [6, 7] | YFCC1M    | 34.1               | 63.9               | 35.4               | -                  |
| PIRL ( ours )      | YFCC1M    | 57.8               | 78.8               | 51.0               | 29.7               |
| Jigsaw [19]        | YFCC100M  | 48.3               | 71.0               | 44.8               | -                  |
| DeeperCluster [7]  | YFCC100M  | 45.6               | 73.0               | 42.1               | -                  |

Table 4 reports the top-1 accuracy of the resulting classifiers. In line with prior results, PIRL outperforms competing self-supervised learners. In fact, PIRL even outperforms Jigsaw and DeeperCluster models that were trained on 100 × more data from the same distribution. Comparing pre-training on ImageNet (Table 2) with pre-training YFCC-1M (Table 4) leads to a mixed set of observations. On ImageNet classification, pre-training (without labels) on ImageNet works substantially better than pre-training on YFCC-1M. In line with prior work [19, 30], however, pretraining on YFCC-1M leads to better representations for image classification on the Places205 dataset.

## 4. Analysis

We performed a set of experiments aimed at better understanding the properties of PIRL. To make it feasible to train the larger number of models needed for these experiments, we train the models we study in this section for fewer epochs ( 400 ) and with fewer negatives ( N = 4 , 096 ) than in Section 3. As a result, we obtain lower absolute performances. Apart from that, we did not change the experimental setup or any of the other hyperparameters. Throughout the section, we use the evaluation setup from Section 3.2 that trains linear classifiers on fixed image representations to measure the quality of image representations.

## 4.1. Analyzing PIRL Representations

## Does PIRL learn invariant representations?

PIRL was designed to learn representations that are invariant to image transformation t ∈ T . We analyzed whether the learned representations actually have the desired invari- ance properties. Specifically, we normalize the representations to have unit norm and compute l 2 distances between the (normalized) representation of image, f ( v I ) , and the (normalized) representation its transformed version, g ( v I t ) . We repeat this for all transforms t ∈ T and for a large set of images. We plot histograms of the distances thus obtained in Figure 4. The figure shows that, for PIRL, an image representation and the representation of a transformed version of that image are generally similar. This suggests that PIRL has learned representations that are invariant to the transformations. By contrast, the distances between Jigsaw representations have a much larger mean and variance, which suggests that Jigsaw representations covary with the image transformations that were applied.

Figure 4: Invariance of PIRL representations. Distribution of l 2 distances between unit-norm image representations, f ( v I ) / ‖ f ( v I ) ‖ 2 , and unit-norm representations of the transformed image, g ( v I t ) / ‖ g ( v I t ) ‖ 2 . Distance distributions are shown for PIRL and Jigsaw representations.

<!-- image -->

## Which layer produces the best representations?

All prior experiments used PIRL representations that were extracted from the res5 layer and Jigsaw representations that were extracted from the res4 layer (which work better for Jigsaw). Figure 5 studies the quality of representations in earlier layers of the convolutional networks. The figure reveals that the quality of Jigsaw representations improves from the conv1 to the res4 layer but that their quality sharply decreases in the res5 layer. We surmise this happens because the res5 representations in the last layer of the network covary with the image transformation t and are not encouraged to contain semantic information. By contrast, PIRL representations are invariant to image transformations, which allows them to focus on modeling semantic information. As a result, the best image representations are extracted from the res5 layer of PIRL-trained networks.

## 4.2. Analyzing the PIRL Loss Function

## What is the effect of λ in the PIRL loss function?

The PIRL loss function in Equation 5 contains a hyperparameter λ that trades off between two NCE losses. All prior experiments were performed with λ =0 . 5 . NPID(++) [72] is a special case of PIRL in which λ =0 , effectively removing the pretext-invariance term from the loss. At λ =1 , the network does not compare untransformed images at training time and updates to the memory bank m I are not dampened.

Figure 5: Quality of PIRL representations per layer. Top-1 accuracy of linear models trained to predict ImageNet classes based on representations extracted from various layers in ResNet-50 trained using PIRL and Jigsaw.

<!-- image -->

Figure 6: Effect of varying the trade-off parameter λ . Top-1 accuracy of linear classifiers trained to predict ImageNet classes from PIRL representations as a function of hyperparameter λ in Equation 5.

<!-- image -->

We study the effect of λ on the quality of PIRL representations. As before, we measure representation quality by the top-1 accuracy of linear classifiers operating on fixed ImageNet representations. Figure 6 shows the results of these experiments. The results show that the performance of PIRL is quite sensitive to the setting of λ , and that the best performance if obtained by setting λ =0 . 5 .

## What is the effect of the number of image transforms?

Both in PIRL and Jigsaw, it is possible to vary the complexity of the task by varying the number of permutations of the nine image patches that are included in the set of image transformations, T . Prior work on Jigsaw suggests that increasing the number of possible patch permutations leads to better performance [19, 46]. However, the largest value |T | can take is restricted because the number of learnable parameters in the output layer grows linearly with the number of patch permutations in models trained to solve the Jigsaw task. This problem does not apply to PIRL because it never outputs the patch permutations, and thus has a fixed number of model parameters. As a result, PIRL can use all 9! ≈ 3 . 6 million permutations in T .

Figure 7: Effect of varying the number of patch permutations in T . Performance of linear image classification models trained on the VOC07 dataset in terms of mAP. Models are initialized by PIRL and Jigsaw, varying the number of image transformations, T , from 1 to 9! ≈ 3 . 6 million.

<!-- image -->

We study the quality of PIRL and Jigsaw as a function of the number of patch permutations included in T . To facilitate comparison with [19], we measure quality in terms of performance of linear models on image classification using the VOC07 dataset, following the same setup as in Section 3.2. The results of these experiments are presented in Figure 7. The results show that PIRL outperforms Jigsaw for all cardinalities of T but that PIRL particularly benefits from being able to use very large numbers of image transformations ( i.e. , large |T | ) during training.

## What is the effect of the number of negative samples?

We study the effect of the number of negative samples, N , on the quality of the learned image representations. We measure the accuracy of linear ImageNet classifiers on fixed representations produced by PIRL as a function of the value of N used in pre-training. The results of these experiments are presented in Figure 8. They suggest that increasing the number of negatives tends to have a positive influence on the quality of the image representations constructed by PIRL.

## 4.3. Generalizing PIRL to Other Pretext Tasks

Although we studied PIRL in the context of Jigsaw in this paper, PIRL can be used with any set of image transformations, T . We performed an experiment evaluating the performance of PIRL using the Rotation pretext task [18]. We define T to contain image rotations by { 0 ◦ , 90 ◦ , 180 ◦ , 270 ◦ } , and measure representation quality in terms of image-classification accuracy of linear models (see the supplemental material for details).

The results of these experiments are presented in Table 5 (top). In line with earlier results, models trained using PIRL

Figure 8: Effect of varying the number of negative samples. Top-1 accuracy of linear classifiers trained to perform ImageNet classification using PIRL representations as a function of the number of negative samples, N .

<!-- image -->

(Rotation) outperform those trained using the Rotation pretext task of [18]. The performance gains obtained from learning a rotation-invariant representation are substantial, e.g . +11% top-1 accuracy on ImageNet. We also note that PIRL (Rotation) outperforms NPID++ (see Table 2). In a second set of experiments, we combined the pretext image transforms from both the Jigsaw and Rotation tasks in the set of image transformations, T . Specifically, we obtain I t by first applying a rotation and then performing a Jigsaw transformation. The results of these experiments are shown in Table 5 (bottom). The results demonstrate that combining image transforms from multiple pretext tasks can further improve image representations.

## 5. Related Work

Our study is related to prior work that tries to learn characteristics of the image distribution without considering a corresponding (image-conditional) label distribution. A variety of work has studied reconstructing images from a small, intermediate representation, e.g ., using sparse coding [50], adversarial training [11, 12, 43], autoencoders [42, 55, 67], or probabilistic versions thereof [59].

More recently, interest has shifted to specifying pretext tasks [9] that require modeling a more limited set of properties of the data distribution. For video data, these pretext tasks learn representations by ordering video frames [1, 16, 32, 37, 44, 70, 74], tracking [54, 68], or using cross-modal signals like audio [2, 3, 17, 34, 52, 53].

Our work focuses on image-based pretext tasks. Prior pretext tasks include image colorization [8, 28, 35, 36, 77, 78], orientation prediction [18], affine transform prediction [76], predicting contextual image patches [9], reordering image patches [5, 19, 46, 48], counting visual primitives [47], or their combinations [10]. In contrast, our works learns image representations that are invariant to the image transformations rather than covariant.

PIRL is related to approaches that learn invariant image representations via contrastive learning [14, 27, 60, 68, 72], clustering [6, 7, 48, 69], or maximizing mutual information [4, 27, 29]. PIRL is most similar to methods that learn representations that are invariant under standard data augmentation [4, 13, 27, 29, 72, 73]. PIRL learns representations that are invariant to both the data augmentation and to the pretext image transformations.

Table 5: Using PIRL with (combinations of) different pretext tasks. Top-1 accuracy / mAP of linear image classifiers trained on PIRL image representations. Top: Performance of PIRL used in combination with the Rotation pretext task [18]. Bottom: Performance of PIRL using a combination of multiple pretext tasks.

| Method                             | Params                             | Transfer Dataset                   | Transfer Dataset                   | Transfer Dataset                   | Transfer Dataset                   |
|------------------------------------|------------------------------------|------------------------------------|------------------------------------|------------------------------------|------------------------------------|
|                                    |                                    | ImageNet                           | VOC07                              | Places205                          | iNat.                              |
| Rotation [18]                      | 25.6M                              | 48.9                               | 63.9                               | 41.4                               | 23.0                               |
| PIRL (Rotation; ours )             | 25.6M                              | 60.2                               | 77.1                               | 47.6                               | 31.2                               |
| ∆ of PIRL                          | -                                  | +11.3                              | +13.2                              | +6.2                               | +8.2                               |
| Combining pretext tasks using PIRL | Combining pretext tasks using PIRL | Combining pretext tasks using PIRL | Combining pretext tasks using PIRL | Combining pretext tasks using PIRL | Combining pretext tasks using PIRL |
| PIRL (Jigsaw; ours )               | 25.6M                              | 62.2                               | 79.8                               | 48.5                               | 31.2                               |
| PIRL (Rotation + Jigsaw; ours )    | 25.6M                              | 63.1                               | 80.3                               | 49.7                               | 33.6                               |

Finally, PIRL is also related to approaches that use a contrastive loss [22] in predictive learning [23, 24, 26, 51, 61, 64]. These prior approaches predict missing parts of the data, e.g ., future frames in videos [23, 51], or operate on multiple views [64]. In contrast to those approaches, PIRL learns invariances rather than predicting missing data.

## 6. Discussion and Conclusion

We studied Pretext-Invariant Representation Learning (PIRL) for learning representations that are invariant to image transformations applied in self-supervised pretext tasks. The rationale behind PIRL is that invariance to image transformations maintains semantic information in the representation. We obtain state-of-the-art results on multiple benchmarks for self-supervised learning in image classification and object detection. PIRL even outperforms supervised ImageNet pre-training on object detection.

In this paper, we used PIRL with the Jigsaw and Rotation image transformations. In future work, we aim to extend to richer sets of transformations. We also plan to investigate combinations of PIRL with clustering-based approaches [6, 7]. Like PIRL, those approaches use inter-image statistics but they do so in a different way. A combination of the two approaches may lead to even better image representations.

Acknowledgments: We are grateful to Rob Fergus, and Andrea Vedaldi for encouragement and their feedback on early versions of the manuscript; Yann LeCun for helpful discussions; Aaron Adcock, Naman Goyal, Priya Goyal, and Myle Ott for their help with the code development for this research; and Rohit Girdhar, and Ross Girshick for feedback on the manuscript. We thank Yuxin Wu, and Kaiming He for help with Detectron2.

## References

- [1] Unaiza Ahsan, Rishi Madhok, and Irfan Essa. Video jigsaw: Unsupervised learning of spatiotemporal context for video action recognition. In WACV , 2019. 8
- [2] Relja Arandjelovic and Andrew Zisserman. Look, listen and learn. In ICCV , 2017. 8
- [3] Relja Arandjelovic and Andrew Zisserman. Objects that sound. In ECCV , 2018. 8
- [4] Philip Bachman, R Devon Hjelm, and William Buchwalter. Learning representations by maximizing mutual information across views. arXiv preprint arXiv:1906.00910 , 2019. 5, 8
- [5] Fabio M Carlucci, Antonio D'Innocente, Silvia Bucci, Barbara Caputo, and Tatiana Tommasi. Domain generalization by solving jigsaw puzzles. In CVPR , 2019. 8
- [6] Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In ECCV , 2018. 5, 6, 8
- [7] Mathilde Caron, Piotr Bojanowski, Julien Mairal, and Armand Joulin. Unsupervised pre-training of image features on non-curated data. In ICCV , 2019. 6, 8
- [8] Aditya Deshpande, Jason Rock, and David Forsyth. Learning large-scale automatic image colorization. In ICCV , 2015. 8
- [9] Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In ICCV , pages 1422-1430, 2015. 2, 8
- [10] Carl Doersch and Andrew Zisserman. Multi-task selfsupervised visual learning. In ICCV , 2017. 8, 13
- [11] J. Donahue, P. Krahenb¨ uhl, and T. Darrell. Adversarial feature learning. In ICLR , 2016. 8
- [12] Jeff Donahue and Karen Simonyan. Large scale adversarial representation learning. arXiv preprint arXiv:1907.02544 , 2019. 5, 8
- [13] Alexey Dosovitskiy, Philipp Fischer, Jost Tobias Springenberg, Martin Riedmiller, and Thomas Brox. Discriminative unsupervised feature learning with exemplar convolutional neural networks. TPAMI , 38(9), 2016. 1, 8
- [14] Debidatta Dwibedi, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, and Andrew Zisserman. Temporal cycleconsistency learning. In CVPR , 2019. 8
- [15] M. Everingham, S. M. A. Eslami, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The pascal visual object classes challenge: A retrospective. IJCV , 111(1):98136, Jan. 2015. 1, 4
- [16] Basura Fernando, Hakan Bilen, Efstratios Gavves, and Stephen Gould. Self-supervised video representation learning with odd-one-out networks. In CVPR , 2017. 8
- [17] Ruohan Gao, Rogerio Feris, and Kristen Grauman. Learning to separate object sounds by watching unlabeled video. In ECCV , 2018. 8
- [18] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. arXiv preprint arXiv:1803.07728 , 2018. 1, 2, 3, 5, 7, 8, 12
- [19] Priya Goyal, Dhruv Mahajan, Abhinav Gupta, and Ishan Misra. Scaling and benchmarking self-supervised visual representation learning. In ICCV , 2019. 3, 4, 5, 6, 7, 8, 12, 13, 14
- [20] Yves Grandvalet and Yoshua Bengio. Semi-supervised

learning by entropy minimization. In Advances in neural information processing systems , pages 529-536, 2005. 5

- [21] Michael Gutmann and Aapo Hyv¨ arinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics , pages 297-304, 2010. 2
- [22] Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In CVPR , 2006. 2, 8
- [23] Tengda Han, Weidi Xie, and Andrew Zisserman. Video representation learning by dense predictive coding. In ICCV Workshop , pages 0-0, 2019. 8
- [24] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In arXiv 1911.05722 , 2019. 3, 4, 5, 8
- [25] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR , 2016. 3, 12
- [26] Olivier J H´ enaff, Ali Razavi, Carl Doersch, SM Eslami, and Aaron van den Oord. Data-efficient image recognition with contrastive predictive coding. arXiv preprint arXiv:1905.09272 , 2019. 4, 5, 8
- [27] R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler, and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization. arXiv preprint arXiv:1808.06670 , 2018. 8
- [28] Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa. Let there be color!: joint end-to-end learning of global and local image priors for automatic image colorization with simultaneous classification. ACM Transactions on Graphics , 35(4):110, 2016. 8
- [29] X. Ji, J.F. Henriques, and A. Vedaldi. Invariant information clustering for unsupervised image classification and segmentation. In ICCV , 2019. 1, 8
- [30] Armand Joulin, Laurens van der Maaten, Allan Jabri, and Nicolas Vasilache. Learning visual features from large weakly supervised data. In ECCV , 2016. 5, 6
- [31] Angjoo Kanazawa, David W Jacobs, and Manmohan Chandraker. Warpnet: Weakly supervised matching for singleview reconstruction. In CVPR , pages 3253-3261, 2016. 1
- [32] Dahun Kim, Donghyeon Cho, and In So Kweon. Selfsupervised video representation learning with space-time cubic puzzles. In AAAI , volume 33, 2019. 8
- [33] Alexander Kolesnikov, Xiaohua Zhai, and Lucas Beyer. Revisiting self-supervised visual representation learning. arXiv preprint arXiv:1901.09005 , 2019. 4, 5, 12, 14
- [34] Bruno Korbar, Du Tran, and Lorenzo Torresani. Cooperative learning of audio and video models from self-supervised synchronization. In NeurIPS , 2018. 8
- [35] Gustav Larsson, Michael Maire, and Gregory Shakhnarovich. Learning representations for automatic colorization. In ECCV , 2016. 8
- [36] Gustav Larsson, Michael Maire, and Gregory Shakhnarovich. Colorization as a proxy task for visual understanding. In CVPR , 2017. 5, 8
- [37] Hsin-Ying Lee, Jia-Bin Huang, Maneesh Singh, and Ming-

Hsuan Yang. Unsupervised representation learning by sorting sequences. In CVPR , 2017. 8

- [38] Sungbin Lim, Ildoo Kim, Taesup Kim, Chiheon Kim, and Sungwoong Kim. Fast autoaugment. In NeurIPS , 2019. 4
- [39] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV . 2014. 1, 4
- [40] Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983 , 2016. 4, 12
- [41] Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens van der Maaten. Exploring the limits of weakly supervised pretraining. In ECCV , 2018. 1, 5
- [42] J. Masci, U. Meier, D. Cires, and J. Schmidhuber. Stacked convolutional auto-encoders for hierarchical feature extraction. In ICANN , pages 52-59, 2011. 8
- [43] L. Mescheder, S. Nowozin, and A. Geiger. Adversarial variational Bayes: Unifying variational autoencoders and generative adversarial networks. In ICML , 2017. 8
- [44] Ishan Misra, C Lawrence Zitnick, and Martial Hebert. Shuffle and learn: unsupervised learning using temporal order verification. In ECCV , 2016. 2, 8
- [45] Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, and Shin Ishii. Virtual adversarial training: a regularization method for supervised and semi-supervised learning. TPAMI , 41(8), 2018. 5
- [46] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In ECCV , 2016. 1, 2, 3, 7, 8
- [47] Mehdi Noroozi, Hamed Pirsiavash, and Paolo Favaro. Representation learning by learning to count. In ICCV , 2017. 8
- [48] Mehdi Noroozi, Ananth Vinjimoor, Paolo Favaro, and Hamed Pirsiavash. Boosting self-supervised learning via knowledge transfer. In CVPR , 2018. 8
- [49] David Novotny, Samuel Albanie, Diane Larlus, and Andrea Vedaldi. Self-supervised learning of geometrically stable features through probabilistic introspection. In CVPR , pages 3637-3645, 2018. 1
- [50] B. A. Olshausen and D. J. Field. Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature , 381(6583):607, 1996. 8
- [51] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 , 2018. 3, 5, 8, 13
- [52] Andrew Owens and Alexei A Efros. Audio-visual scene analysis with self-supervised multisensory features. In ECCV , 2018. 8
- [53] Andrew Owens, Jiajun Wu, Josh H McDermott, William T Freeman, and Antonio Torralba. Ambient sound provides supervision for visual learning. In ECCV , 2016. 8
- [54] Deepak Pathak, Ross Girshick, Piotr Doll´ ar, Trevor Darrell, and Bharath Hariharan. Learning features by watching objects move. In CVPR , 2017. 8
- [55] MarcAurelio Ranzato, Fu-Jie Huang, Y-Lan Boureau, and Yann LeCun. Unsupervised learning of invariant feature hierarchies with applications to object recognition. In CVPR , 2007. 8
- [56] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In NIPS , pages 91-99, 2015. 4
- [57] Ignacio Rocco, Relja Arandjelovic, and Josef Sivic. Convolutional neural network architecture for geometric matching. In CVPR , pages 6148-6157, 2017. 1
- [58] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. IJCV , 115, 2015. 1, 4
- [59] R. Salakhutdinov and G. Hinton. Deep Boltzmann machines. In AI-STATS , pages 448-455, 2009. 8
- [60] Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, Sergey Levine, and Google Brain. Time-contrastive networks: Self-supervised learning from video. In ICRA , 2018. 8
- [61] Chen Sun, Fabien Baradel, Kevin Murphy, and Cordelia Schmid. Contrastive bidirectional transformer for temporal representation learning. arXiv preprint arXiv:1906.05743 , 2019. 8
- [62] Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable effectiveness of data in deep learning era. In ICCV , 2017. 5
- [63] Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. arXiv preprint arXiv:1503.01817 , 2015. 6
- [64] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. arXiv preprint arXiv:1906.05849 , 2019. 5, 8
- [65] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The inaturalist species classification and detection dataset. In CVPR , pages 8769-8778, 2018. 4
- [66] Grant Van Horn and Pietro Perona. The devil is in the tails: Fine-grained classification in the wild. arXiv preprint arXiv:1709.01450 , 2017. 1
- [67] P. Vincent, H. Larochelle, Y. Bengio, and P.-A. Manzagol. Extracting and composing robust features with denoising autoencoders. In ICML , 2008. 8
- [68] Xiaolong Wang and Abhinav Gupta. Unsupervised learning of visual representations using videos. In ICCV , 2015. 8
- [69] Xiaolong Wang, Kaiming He, and Abhinav Gupta. Transitive invariance for self-supervised visual representation learning. In ICCV , pages 1329-1338, 2017. 8
- [70] Donglai Wei, Joseph Lim, Andrew Zisserman, and William T. Freeman. Learning and using the arrow of time. In IEEE Conference on Computer Vision and Pattern Recognition , 2018. 8
- [71] Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2. https://github.com/ facebookresearch/detectron2 , 2019. 4, 12
- [72] Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. In CVPR , 2018. 3, 4, 5, 7, 8, 13, 14
- [73] Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, and Quoc V Le. Unsupervised data augmentation. arXiv preprint arXiv:1904.12848 , 2019. 8
- [74] Dejing Xu, Jun Xiao, Zhou Zhao, Jian Shao, Di Xie, and

Yueting Zhuang. Self-supervised spatiotemporal learning via video clip order prediction. In CVPR , 2019. 8

- [75] Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, and Lucas Beyer. S4l: Self-supervised semi-supervised learning. arXiv preprint arXiv:1905.03670 , 2019. 5
- [76] Liheng Zhang, Guo-Jun Qi, Liqiang Wang, and Jiebo Luo. Aet vs. aed: Unsupervised representation learning by autoencoding transformations rather than data. arXiv preprint arXiv:1901.04596 , 2019. 1, 2, 5, 8
- [77] Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In ECCV , 2016. 8
- [78] Richard Zhang, Phillip Isola, and Alexei A Efros. Split-brain autoencoders: Unsupervised learning by cross-channel prediction. In CVPR , 2017. 3, 4, 8
- [79] Bolei Zhou, Agata Lapedriza, Jianxiong Xiao, Antonio Torralba, and Aude Oliva. Learning deep features for scene recognition using places database. In NIPS , 2014. 4
- [80] Chengxu Zhuang, Alex Lin Zhai, and Daniel Yamins. Local aggregation for unsupervised learning of visual embeddings. In ICCV , pages 6002-6012, 2019. 5, 13

## Supplemental Material

## A. Training architecture and Hyperparameters

Base Architecture for PIRL PIRL models in Section 3, 4 and 5 in the main paper are standard ResNet-50 [25] models with 25.6 million parameters. Figure 2 and Section 3.2 also use a larger model 'PIRL-c2x' which has double the number of channels in each 'stage' of the ResNet, e.g ., 128 channels in the first convolutional layer, 4096 channels in the final res5 layer etc ., and a total of 98 million parameters. The heads f ( · ) and g ( · ) are linear layers that are used for the pre-training stage (Figure 3). When evaluating the model using transfer learning (Section 4 and 5 of the main paper), the heads f , g are removed.

Training Hyperparameters for Section 3 PIRL models in Section 3 are trained for 800 epochs using the ImageNet training set (1.28 million images). We use a batchsize of 32 per GPU and use a total of 32 GPUs to train each model. We optimize the models using mini-batch SGD with the cosine learning rate decay [40] scheme with an initial learning rate of 1 . 2 × 10 -1 and a final learning rate of 1 . 2 × 10 -4 , momentum of 0 . 9 and a weight decay of 10 -4 . We use a total of 32 , 000 negatives to compute the NCE loss (Equation 5) and the negatives are sampled randomly for each data point in the batch.

Training Hyperparameters for Section 4 The hyperparameters used are exactly the same as Section 3, except that the models are trained with 4096 negatives and for 400 epochs only. This results in a lower absolute performance, but the main focus of Section 4 is to analyze PIRL.

Common Data Pre-processing We use data augmentations as implemented in PyTorch and the Python Imaging Library. These data augmentations are used for all methods including PIRL and NPID++. We do not use supervised policies like Fast AutoAugment. Our 'geometric' data pre-processing involves extracting a randomly resized crop from the image, and horizontally flipping it. We follow this by 'photometric' pre-processing that alter the RGB color values by using random values of color jitter, contrast, brightness, saturation, sharpness, equalize etc . transforms to the image.

## A.1. Details for PIRL with Jigsaw

Data pre-processing Webase our Jigsaw implementation on [19, 33]. To construct I t , we extract a random resized crop that occupies at least 60% of the image. We resize this crop to 255 × 255 , and then divide into a 3 × 3 grid. We extract a patch of 64 × 64 from each of these grids and get 9

patches. We apply photometric data augmentation (random color jitter, hue, contrast, saturation) to each patch independently and finally obtain the 9 patches that constitute I t . The image I is a 224 × 224 image obtained by applying standard data augmentation (flip, random resized crop, color jitter, hue, contrast, saturation) to the image from the dataset.

Architecture The 9 patches of I t are individually input to the network to get their res5 features which are average pooled to get a single 2048 dimensional vector for each patch. We then apply a linear projection to these features to get a 128 dimensional vector for each patch. These patch features are concatenated to get a 1152 dimensional vector which is then input to another single linear layer to get the final 128 dimensional feature v I t .

We feed forward the image I and average pool its res5 feature to get a 2048 dimensional vector for the image. We then apply a single linear projection to get a 128 dimensional feature v I .

## A.2. Details for PIRL with Rotation

Data pre-processing We base our Rotation implementation on [18, 33]. To construct I t we use the standard data augmentation described earlier (geometric + photometric) to get a 224 × 224 image, followed by a random rotation from { 0 ◦ , 90 ◦ , 180 ◦ , 270 ◦ } .

The image I is a 224 × 224 image obtained by applying standard data augmentation (flip, random resized crop, color jitter, hue, contrast, saturation) to the image from the dataset.

Architecture We feed forward the image I t , average pool the res5 feature, and apply a single linear layer to get the final 128 dimensional feature for v I t .

We feed forward the image I and average pool its res5 feature to get a 2048 dimensional vector for the image. We then apply a single linear projection to get a 128 dimensional feature v I .

## B. Object Detection

## B.1. VOC07 train+val set for detection

In Section 3.1 of the main paper, we presented object detection results using the VOC07+12 training set which has 16K images. In this section, we use the smaller VOC07 train+val set (5K images) for finetuning the Faster R-CNN C4 detection models. All models are finetuned using the Detectron2 [71] codebase and hyperparameters from [19]. We report the detection AP in Table 6. We see that the PIRL model outperforms the ImageNet supervised pretrained models on the stricter AP 75 and AP all metrics without extra pretraining data or changes to the network architecture.

Detection hyperparameters: We use a batchsize of 2 images per GPU, a total of 8 GPUs and finetune models for 12 . 5 K iterations with the learning rate dropped by 0 . 1 at 9 . 5 K iterations. The supervised and Jigsaw baseline models use an initial learning rate of 0 . 02 with a linear warmup for 100 iterations with a slope of 1 / 3 , while the NPID++ and PIRL models use an initial learning rate of 0 . 003 with a linear warmup for 1000 iterations and a slope of 1 / 3 . We keep the BatchNorm parameters of all models fixed during the detection finetuning stage.

Table 6: Object detection on VOC07 using Faster R-CNN. Detection AP on the VOC07 test set after finetuning Faster R-CNN models with a ResNet-50 backbone pre-trained using self-supervised learning on ImageNet. Results for supervised ImageNet pre-training are presented for reference. All methods are finetuned using the images from the VOC07 train+val set. Numbers with ∗ are adopted from the corresponding papers. We see that PIRL outperforms supervised pre-training without extra pretraining data or changes in the network architecture.

| Method          | Network   | AP all   | AP 50   | AP 75   | ∆AP 75   |
|-----------------|-----------|----------|---------|---------|----------|
| Supervised      | R-50      | 43.8     | 74.5    | 45.9    | =0.0     |
| Jigsaw [19]     | R-50      | 37.7     | 64.9    | 38.0    | -7.9     |
| PIRL (ours)     | R-50      | 44.7     | 73.4    | 47.0    | +1.1     |
| NPID [72]       | R-50      | -        | 65.4 ∗  | -       |          |
| LA [80]         | R-50      | -        | 69.1 ∗  | -       |          |
| Multi-task [10] | R-101     | -        | 70.5 ∗  | -       |          |

## B.2. VOC07+12 train set for detection

In Table 1, we use the VOC07+12 training split and the VOC07 test set as used in prior work [19, 51].

Detection hyperparamters: We use a batchsize of 2 images per GPU, a total of 8 GPUs and finetune models for 25 K iterations with the learning rate dropped by 0 . 1 at 17 K iterations. The supervised and Jigsaw baseline models use an initial learning rate of 0 . 02 with a linear warmup for 100 iterations with a slope of 1 / 3 , while the NPID++ and PIRL models use an initial learning rate of 0 . 003 with a linear warmup for 1000 iterations and a slope of 1 / 3 . We keep the BatchNorm parameters of all models fixed during the detection finetuning stage.

## C. Linear Models for Transfer

We train linear models on representations from the intermediate layers of a ResNet-50 model following the procedure outlined in [19]. We briefly outline the hyperparameters from their work. The features from each of the layers are average pooled such that they are of about 9 , 000 dimensions each. The linear model is trained with mini-batch SGD using a learning rate of 0 . 01 decayed by 0 . 1 at two equally spaced intervals, momentum of 0 . 9 and weight decay of 5 × 10 -4 . Wetrain the linear models for 28 epochs on ImageNet (1.28M training images), 14 epochs on Places205

(2.4M training images) and for 84 epochs on iNaturalist2018 (437K training images). Thus, the number of parameter updates for training the linear models are roughly constant across all these datasets. We report the center crop top-1 accuracy for the ImageNet, Places205 and iNaturalist2018 datasets in Table 2.

For training linear models on VOC07, we train linear SVMs following the procedure in [19] and report mean average Precision (mAP).

Per layer results The results of all these models are in Table 7.

Table 7: ResNet-50 linear evaluation on ImageNet and Places205 datasets: We report the performance of all the layers following the evaluation protocol in [19]. All numbers except NPID++ and Ours are from their respective papers.

|                            | ImageNet                   | ImageNet                   | ImageNet                   | ImageNet                   | ImageNet                   | Places205                  | Places205                  | Places205                  | Places205                  | Places205                  |
|----------------------------|----------------------------|----------------------------|----------------------------|----------------------------|----------------------------|----------------------------|----------------------------|----------------------------|----------------------------|----------------------------|
| Method                     | conv1                      | res2                       | res3                       | res4                       | res5                       | conv1                      | res2                       | res3                       | res4                       | res5                       |
| Random                     | 9.6                        | 13.7                       | 12.0                       | 8.0                        | 5.6                        | 12.9                       | 16.6                       | 15.5                       | 11.6                       | 9.0                        |
| ImageNet Supervised        | 11.6                       | 33.3                       | 48.7                       | 67.9                       | 75.5                       | 14.8                       | 32.6                       | 42.1                       | 50.8                       | 51.5                       |
| Places205 Supervised       | 13.2                       | 31.7                       | 46.0                       | 58.2                       | 51.7                       | 16.7                       | 32.3                       | 43.2                       | 54.7                       | 62.3                       |
| Jigsaw [19]                | 12.4                       | 28.0                       | 39.9                       | 45.7                       | 34.2                       | 15.1                       | 28.8                       | 36.8                       | 41.2                       | 34.4                       |
| Coloriz. [19]              | 10.2                       | 24.1                       | 31.4                       | 39.6                       | 35.2                       | 18.1                       | 28.4                       | 30.2                       | 31.3                       | 30.4                       |
| NPID++ [72]                | 6.6                        | 27.4                       | 38.5                       | 53.1                       | 59                         | 9.7                        | 26.2                       | 34.9                       | 43.7                       | 46.4                       |
| PIRL (Ours)                | 6.2                        | 30                         | 40.1                       | 56.6                       | 63.6                       | 8.9                        | 29                         | 35.8                       | 45.3                       | 49.8                       |
| Different Evaluation Setup | Different Evaluation Setup | Different Evaluation Setup | Different Evaluation Setup | Different Evaluation Setup | Different Evaluation Setup | Different Evaluation Setup | Different Evaluation Setup | Different Evaluation Setup | Different Evaluation Setup | Different Evaluation Setup |
| Kolesnikov et al . [33]    | -                          | -                          | -                          | 47.7                       | -                          | -                          | -                          | -                          | -                          | -                          |
| NPID [72]                  | 15.3                       | 18.8                       | 24.9                       | 40.6                       | 54.0                       | 18.1                       | 22.3                       | 29.7                       | 42.1                       | 45.5                       |