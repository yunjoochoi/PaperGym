## Efficient On-device Training via Gradient Filtering

Yuedong Yang Guihong Li Radu Marculescu The University of Texas at Austin

{ albertyoung, lgh, radum } @utexas.edu

dient quantization [4, 7] can reduce the cost of arithmetic operations but cannot reduce the number of operations ( e.g. , multiplications); thus, the speedup in training remains limited. Moreover, gradient quantization is not supported by existing deep-learning frameworks (e.g., CUDNN [9], MKLDNN [1], PyTorch [25] and Tensorflow [2]). To enable on-device training, there are two important questions must be addressed:

- How can we reduce the computational complexity of back propagation through the convolution layers?
- How can we reduce the data required by the gradient computation during back propagation?

In this paper, we propose gradient filtering , a new research direction, to address both questions. By addressing the first question, we reduce the computational complexity of training; by addressing the second question, we reduce the memory consumption.

In general, the gradient propagation through a convolution layer involves multiplying the gradient of the output variable with a Jacobian matrix constructed with data from either the input feature map or the convolution kernel. We aim at simplifying this process with the new gradient filtering approach proposed in Section 3. Intuitively, if the gradient map w.r.t. the output has the same value for all entries, then the computation-intensive matrix multiplication can be greatly simplified, and the data required to construct the Jacobian matrix can be significantly reduced. Thus, our gradient filtering can approximate the gradient w.r.t. the output by creating a new gradient map with a special ( i.e. , spatial) structure and fewer unique elements. By doing so, the gradient propagation through the convolution layers reduces to cheaper operations, while the data required (hence memory) for the forward propagation also lessens. Through this filtering process, we trade off the gradient precision against the computation complexity during BP. We note that gradient filtering does not necessarily lead to a worse precision, i.e. , models sometimes perform better with filtered gradients when compared against models trained with vanilla BP.

In summary, our contributions are as follows:

## Abstract

Despite its importance for federated learning, continuous learning and many other applications, on-device training remains an open problem for EdgeAI. The problem stems from the large number of operations (e.g., floating point multiplications and additions) and memory consumption required during training by the back-propagation algorithm. Consequently, in this paper, we propose a new gradient filtering approach which enables on-device CNN model training. More precisely, our approach creates a special structure with fewer unique elements in the gradient map, thus significantly reducing the computational complexity and memory consumption of back propagation during training. Extensive experiments on image classification and semantic segmentation with multiple CNN models (e.g., MobileNet, DeepLabV3, UPerNet) and devices (e.g., Raspberry Pi and Jetson Nano) demonstrate the effectiveness and wide applicability of our approach. For example, compared to SOTA, we achieve up to 19 × speedup and 77.1% memory savings on ImageNet classification with only 0.1% accuracy loss. Finally, our method is easy to implement and deploy; over 20 × speedup and 90% energy savings have been observed compared to highly optimized baselines in MKLDNN and CUDNN on NVIDIA Jetson Nano. Consequently, our approach opens up a new direction of research with a huge potential for on-device training. 1

## 1. Introduction

Existing approaches for on-device training are neither efficient nor practical enough to satisfy the resource constraints of edge devices (Figure 1). This is because these methods do not properly address a fundamental problem in on-device training, namely the computational and memory complexity of the back-propagation (BP) algorithm. More precisely, although the architecture modification [6] and layer freezing [18, 20] can help skipping the BP for some layers, for other layers, the complexity remains high. Gra-

1 Code: https://github.com/SLDGroup/GradientFilter-CVPR23

- We propose gradient filtering , which reduces the computation and memory required for BP by more than two orders of magnitude compared to the exact gradient calculation.
- We provide a rigorous error analysis which shows that the errors introduced by the gradient filtering have only a limited influence on model accuracy.
- Our experiments with multiple CNN models and computer vision tasks show that we can train a neural network with significantly less computation and memory costs, with only a marginal accuracy loss compared to baseline methods. Side-by-side comparisons against other training acceleration techniques also suggest the effectiveness of our method.
- Our method is easy to deploy with highly optimized deep learning frameworks ( e.g. , MKLDNN [1] and CUDNN [9]). Evaluations on resource-constrained edge (Raspberry Pi and Jetson Nano) and highperformance devices (CPU/GPU) show that our method is highly suitable for real life deployment.

The paper is organized as follows. Section 2 reviews relevant work. Section 3 presents our method in detail. Section 4 discusses error analysis, computation and memory consumption. Experimental results are presented in Section 5. Finally, Section 6 summarizes our main contributions.

## 2. Related Work

Architecture Modification: Authors of [6] propose to attach small branches to the original neural network. During training, the attached branches and biases in the original model are updated. Though memory consumption is reduced, updating these branches still needs gradient propagation through the entire network; moreover, a large computational overhead for inference is introduced.

Layer Freezing: Authors of [18, 20] propose to only train parts of the model. [18] makes layer selection based on layer importance metrics, while [20] uses evolutionary search. However, the layers selected by all these methods are typically computationally heavy layers ( e.g. , the last few layers in ResNet [14]) which consume most of the resources. Thus, the speedup achieved by these approaches is limited. Gradient Quantization: [3,5] quantize gradient after backpropagation, which means these methods cannot accelerate the training on a single device. Work in [4, 7, 15, 17, 28, 29, 33] accelerates training by reducing the cost for every arithmetic operation. However, these methods do not reduce the number of operations, which is typically huge for SOTA CNNs, so their achievable speedup is limited. Also, all these methods are not supported by the popular deep learning frameworks [1,2,9,25].

Figure 1. Matrix of orthogonal directions for on-device training. 'Arch' is short for 'architecture'. Our approach opens up a new direction of research for on-device training for EdgeAI.

<!-- image -->

In contrast to the prior work, our method opens up a new research direction. More precisely, we reduce the number of computations and memory consumption required for training a single layer via gradient filtering. Thus, our method can be combined with any of the methods mentioned above. For example, in Section H in the Supplementary, we illustrate how our method can work together with the gradient quantization methods to enable a higher speedup.

## 3. Proposed Method

In this section, we introduce our gradient filtering approach to accelerate BP. To this end, we target the most computation and memory heavy operation, i.e. , convolution (Figure 2(a)). Table 1 lists some symbols we use.

Table 1. Table of symbols we use.

| C x                    | Number of channels of x                                                                        |
|------------------------|------------------------------------------------------------------------------------------------|
| W x ,H x               | Width and height of x                                                                          |
| θ                      | Convolution kernel                                                                             |
| θ ′                    | Rotated θ , i.e. , θ ′ = rot180 ( θ )                                                          |
| r                      | Patch size ( r × r )                                                                           |
| g x , g y , g θ        | Gradients w.r.t. x, y, θ                                                                       |
| ˜ g y                  | Approximated gradient g y                                                                      |
| ˜ x, ˜ θ ′             | Sum of x and θ ′ over spatial dimensions (height and width)                                    |
| x [ n, c i ,h,w ]      | Element for feature map x at batch n , channel c i , pixel ( h,w )                             |
| θ [ c o , c i , u, v ] | Element for convolution kernel θ at output channel c o , input channel c i , position ( u, v ) |

<!-- image -->

⊛

Figure 2. (a) Computation procedures for vanilla training method (upper) and our method (lower). (b) Example of gradient propagation with gradient filtering. Numbers in this example are chosen randomly for illustration purposes. In this case, the patch size selected for the gradient filter is 2 × 2 . Thus, the 4 × 4 gradient map g y is approximated by ˜ g y , which has four 2 × 2 patches with one unique value for each patch. Also, input feature map x and mirrored convolution kernel θ ′ are spatial summed to ˜ x and ˜ θ ′ . Since ˜ x has fewer unique values than x , memory consumption is reduced. Finally, with ˜ g y , ˜ x and ˜ θ , we compute the gradient w.r.t. kernel and input feature map with much fewer operations than the standard back propagation method.

## 3.1. Problem Setup

The computations for both forward and backward paths are shown in Figure 2(a). For the standard (vanilla) approach (upper Figure 2(a)), starting with input x , the forward propagation convolves the input feature map x with kernel θ and returns output y , which is further processed by the other layers in the neural network (dotted arrow) until the loss value l is calculated. As shown in Figure 2(a), the BP of the convolution layer starts with the gradient map w.r.t. output y ( g y ). The gradient w.r.t. input ( g x ) is calculated by convolving g y with the rotated convolution kernel θ ′ , i.e. , g x = g y glyph[circleasterisk] rot180 ( θ ) = g y glyph[circleasterisk] θ ′ . The gradient w.r.t. convolution kernel, namely g θ , is calculated with the Frobenius inner product [16] between x and g y , i.e. , g θ = g y F x .

The lower half of Figure 2(a) shows our method, where several changes are made: We introduce the gradient filter ' A ' after g y to generate the approximate gradient for BP. Also, instead of using the accurate x and θ ′ values for gradient computation, we sum over spatial dimensions (height and width dimensions), i.e. , ˜ x and ˜ θ ′ , respectively. Finally, the convolution layer now multiplies the approximate gradient ˜ g y with spatial kernel ˜ θ ′ instead of convolving with it to calculate ˜ g x . Figure 2(b) shows an example of gradient propagation with our gradient filter.

## 3.2. Preliminary Analysis

Consider the vanilla BP for convolution in Figure 2(a). Equation (1) shows the number of computations (#FLOPs) required to calculate g x given g y :

<!-- formula-not-decoded -->

The computation requirements in Equation (1) belong to three categories: number of channels, number of unique elements per channel in the gradient map, and kernel size . Our method focuses on the last two categories.

- i. Unique elements: ( W y H y ) represents the number of unique elements per channel in the gradient w.r.t. output variable y ( g y ). Given the high-resolution images we use, this term is huge, so if we manage to reduce the number of unique elements in the spatial dimensions (height and width), the computations required are greatly reduced too.
- ii. Kernel size: ( W θ H θ ) represents the number of unique elements in the convolution kernel. If the gradient g y has some special structure, for example g y = 1 H y × W y · v ( i.e. , every element in g y has the same value v ), then the convolution can be simplified to ( ∑ θ ′ ) v 1 H y × W y (with boundary elements ignored). With such a special structure, only one multiplication and ( W θ H θ -1) additions are required. Moreover, ∑ θ ′ is independent of data so the result can be shared across multiple images until θ gets updated.

## 3.3. Gradient Filtering

To reduce the number of unique elements and create the special structure in the gradient map, we apply the gradient filter after the gradient w.r.t. output ( g y ) is provided. During the backward propagation, the gradient filter A approximates the gradient g y by spatially cutting the gradient map into r × r -pixel patches and then replacing all elements in each patch with their average value (Figure 2(b)):

<!-- formula-not-decoded -->

For instance in Figure 2(b), we replace the 16 distinct values in the gradient map g y with 4 average values in ˜ g y . So given a gradient map g y with N images per batch, C channels, and H × W pixels per channel, the gradient filter returns a structured approximation of the gradient map containing only N × C ×glyph[ceilingleft] H r glyph[ceilingright] ×glyph[ceilingleft] W r glyph[ceilingright] blocks, with one unique value per patch . We use this matrix of unique values to represent the approximate gradient map ˜ g y , as shown in Figure 2(b).

## 3.4. Back Propagation with Gradient Filtering

We describe now the computation procedure used after applying the gradient filter. Detailed derivations are provided in Supplementary Section B.

Gradient w.r.t. input: The gradient w.r.t. input is calculated by convolving θ ′ with g y (Figure 2(a)). With the approximate gradient ˜ g y , this convolution simplifies to:

<!-- formula-not-decoded -->

where ˜ θ ′ [ c o , c i ] = ∑ u,v θ ′ [ c o , c i , u, v ] is the spatial sum of convolution kernel θ , as shown in Figure 2(b).

Gradient w.r.t. kernel: The gradient w.r.t. the kernel is calculated by taking the Frobenius inner product between x and g y , i.e. , g θ [ c o , c i , u, v ] = x F g y , namely:

<!-- formula-not-decoded -->

With the approximate gradient ˜ g y , the operation can be simplified to:

<!-- formula-not-decoded -->

with ˜ x [ n, c i , i, j ] = ∑ glyph[ceilingleft] i/r glyph[ceilingright] r h = glyph[floorleft] i/r glyph[floorright] r ∑ glyph[ceilingleft] j/r glyph[ceilingright] r w = glyph[floorleft] j/r glyph[floorright] r x [ n, c i , h, w ] . As shown in Figure 2(b), ˜ x [ n, c i , i, j ] is the spatial sum of x elements in the same patch containing pixel ( i, j ) .

## 4. Analyses of Proposed Approach

In this section, we analyze our method from three perspectives: gradient filtering approximation error, computation reduction, and memory cost reduction.

## 4.1. Error Analysis of Gradient Filtering

We prove that the approximation error introduced by our gradient filtering is bounded during the gradient propagation. Without losing generality, we consider that all variables have only one channel, i.e. , C x 0 = C x 1 = 1 .

Proposition 1: For any input-output channel pair ( c o , c i ) in the convolution kernel θ , assuming the DC component has the largest energy value compared to all components in the spectrum 2 , then the signal-to-noise-ratio (SNR) of ˜ g x is greater than SNR of ˜ g y .

Proof: We use G x , G y and Θ to denote the gradients g x , g y and the convolution kernel θ in the frequency domain ; G x [ u, v ] is the spectrum value at frequency ( u, v ) and δ is the 2D discrete Dirichlet function. To simplify the discussion, we consider only one patch of size r × r .

The gradient returned by the gradient filtering can be written as:

<!-- formula-not-decoded -->

where glyph[circleasterisk] denotes convolution. By applying the discrete Fourier transformation, Equation (6) can be rewritten in the frequency domain as:

<!-- formula-not-decoded -->

˜ g y is the approximation of g y ( i.e. , the ground truth for ˜ g y is g y ), and the SNR of ˜ g y equals to:

<!-- formula-not-decoded -->

For the convolution layer, the gradient w.r.t. the approximate variable ˜ x in the frequency domain is 3 :

<!-- formula-not-decoded -->

and its ground truth is:

<!-- formula-not-decoded -->

Similar to Equation (8), the SNR of g ˜ x is:

<!-- formula-not-decoded -->

Equation (11) can be rewritten as:

<!-- formula-not-decoded -->

Furthermore, the main assumption ( i.e. , the DC component dominates the frequency spectrum of Θ ) can be written as:

glyph[negationslash]

<!-- formula-not-decoded -->

2 As a reminder, the energy of a signal is the sum of energy of the DC component and the energy of its AC components.

3 Because g y is convolved with the rotated kernel θ ′ , in the frequency domain, we use Θ[ -u, -v ] instead of Θ[ u, v ] .

Figure 3. Computation analysis for a specific convolution layer 4 . Minimum achievable computation is given in Equation (16). By reducing the number of unique elements, computations required by our approach drop to about 1 /r 2 compared with the standard BP method. By combining it with structured gradient map, computations required by our approach drop further, getting very close to the theoretical limit.

<!-- image -->

that is, ∀ ( u, v ) , Θ 2 [ -u, -v ] Θ 2 [0 , 0] ≤ 1 ; thus, by combining Equation (12) and Equation (13), we have:

<!-- formula-not-decoded -->

which means that: SNR ˜ g x ≥ SNR ˜ g y . This completes our proof for error analysis. glyph[squaresolid]

In conclusion, as the gradient propagates through the network, the noise introduced by our gradient filter becomes weaker compared to the real gradient signal. This property ensures that the error in gradient has only a limited influence on the quality of BP. We validate Proposition 1 later in the experimental section.

## 4.2. Computation and Overhead Analysis

In this section, we analyse the computation required to compute g x , the gradient w.r.t. input x . Figure 3 compares the computation required to propagate the gradient through this convolution layer under different patch sizes r × r . A patch size 1 × 1 means the vanilla BP algorithm which we use as the baseline. As discussed in the preliminary analysis section (Section 3.2), two terms contribute to the computation savings: fewer unique elements in the gradient map and the structured gradient map.

Fewer unique elements: In vanilla BP, there are H y W y unique elements in the gradient map. After applying gradient filtering with a patch size r × r , the number of unique elements reduces to only glyph[ceilingleft] H y r glyph[ceilingright]glyph[ceilingleft] W y r glyph[ceilingright] . This reduction contributes the most to the savings in computation (orange line in Figure 3).

4 The layer is from U-Net [26]. The size of the input is assumed to be 120 × 160 pixels with 192 channels; the output has the same resolution, but with only 64 channels. The kernel size of the convolution layer is 3 × 3 . Analysis for ResNet is included in the supplementary material.

Structured Gradient Map: By creating the structured gradient map, the convolution over the gradient map ˜ g y is simplified to the element-wise multiplication and channel-wise addition. Computation is thus reduced to ( H θ W θ ) -1 of its original value. For instance, the example convolution layer in Figure 3 uses a 3 × 3 convolution kernel so around 89% computations are removed. The blue line in Figure 3 shows the #FLOPs after combining both methods. Greater reduction is expected when applying our method with larger convolution kernels. For instance, FastDepth [30] uses 5 × 5 convolution kernel so as much as 96% reduction in computation can be achieved, in principle.

Minimum Achievable Computation: With the two reductions mentioned above, the computation required to propagate the gradient through the convolution layer is:

<!-- formula-not-decoded -->

where o ( H y W y ) is a constant term which is independent of r and negligible compared to H y W y . When the patch is as large as the feature map, our method reaches the minimum achievable computation (blue dashed line in Figure 3):

<!-- formula-not-decoded -->

In this case, each channel of the gradient map is represented with a single value, so the computation is controlled by the number of input and output channels.

Overhead: The overhead of our approach comes from approximating the feature map x , gradient g y , and kernel θ . As the lower part of Figure 2(a) shows, the approximation for x is considered as part of forward propagation, while the other two as back propagation. Indeed, with the patch size r , the ratio of forward propagation overhead is about 1 / (2 C o W θ H θ ) , while the ratio of backward propagation overhead is about ( r 2 -1) / (2 C x ) .

Given the large number of channels and spatial dimensions in typical neural networks, both overhead values take less than 1% computation in the U-Net example above.

## 4.3. Memory Analysis

As Figure 2(a) shows, the standard back propagation for a convolution layer relies on the input feature map x , which needs to be stored in memory during forward propagation. Since every convolution layer requiring gradient for its kernel needs to save a copy of feature map x , the memory consumption for storing x is huge. With our method, we simplify the feature map x to approximated ˜ x , which has only glyph[ceilingleft] H x r glyph[ceilingright]glyph[ceilingleft] W x r glyph[ceilingright] unique elements for every channel. Thus, by saving only these unique values, our method achieves around (1 -1 r 2 ) memory savings, overall.

Table 2. Experimental results for ImageNet classification with four neural networks (MobileNet-V2, ResNet18/34, MCUNet). '#Layers' is short for 'the number of active convolutional layers'. For example, #Layers equals to 2 means that only the last two convolutional layers are trained. For memory consumption, we only consider the memory for input feature x . Strategy 'No Finetuning' shows the accuracy on new datasets without finetuning the pretrained model. Since TinyTL [6] changes the architecture, '#Layers' is not applicable (N/A).

| MobileNetV2 [27]   | #Layers   | Accuracy       | FLOPs                 | Mem                       | ResNet-18 [14]      | #Layers   | Accuracy       | FLOPs                | Mem                       |
|--------------------|-----------|----------------|-----------------------|---------------------------|---------------------|-----------|----------------|----------------------|---------------------------|
| No Finetuning      | 0         | 4.2            | 0                     | 0                         | No Finetuning       | 0         | 4.7            | 0                    | 0                         |
| Vanilla BP         | All 2 4   | 75.1 63.1 62.2 | 1.13G 113.68M 160.00M | 24.33MB 245.00KB 459.38KB | Vanilla BP          | All 2 4   | 73.1 70.4 72.3 | 5.42G 489.20M 1.14G  | 8.33MB 196.00KB 490.00KB  |
| TinyTL [6]         | N/A       | 60.2           | 663.51M               | 683.00KB                  | TinyTL [6]          | N/A       | 69.2           | 3.88G                | 1.76MB                    |
| Ours               | 2 4       | 63.1 63.4      | 39.27M 53.96M         | 80.00KB 150.00KB          | Ours                | 2 4       | 68.6 68.5      | 28.32M 61.53M        | 64.00KB 112.00KB          |
| MCUNet [19]        | #Layers   | Accuracy       | FLOPs                 | Mem                       | ResNet-34 [14]      | #Layers   | Accuracy       | FLOPs                | Mem                       |
| No Finetune        | 0         | 4.1            | 0                     | 0                         | No Finetune Vanilla | 0         |                | 0                    | 0                         |
| Vanilla BP         | All 2 4   | 68.5 62.1 64.9 | 231.67M 18.80M 33.71M | 9.17MB 220.50KB 312.38KB  | BP                  | All 2 4   | 70.8 69.6 72.3 | 11.17G 489.20M 1.21G | 13.11MB 196.00KB 392.00KB |
| TinyTL [6]         | N/A       | 53.1           | 148.01M               | 571.5KB                   | TinyTL [6]          | N/A       | 72.9           | 8.03G                | 2.95MB                    |
| Ours               | 2 4       | 61.8 64.4      | 6.34M 11.01M          | 72.00KB 102.00KB          | Ours                | 2 4       | 68.6 70.6      | 28.32M 64.07M        | 64.00KB 128.00KB          |

Table 3. Experimental results for semantic segmentation task on augmented Pascal VOC12 dataset [8]. Model name with postfix 'M' means the model uses MobileNetV2 as backbone, otherwise ResNet18 is used. '#Layers' is short for 'the number of active convolutional layers' that are trained. All models are pretrained on Cityscapes dataset [11]. Strategy 'Calibration' shows the accuracy when only the classifier and normalization statistics are updated to adapt different numbers of classes between augmented Pascal VOC12 and Cityscapes.

| PSPNet [32]   | #Layers   | GFLOPs   | mIoU   | mAcc   | PSPNet-M [32]   | #Layers   | GFLOPs   | mIoU   | mAcc        | FCN [21]     | #Layers   | GFLOPs   | mIoU   | mAcc   |
|---------------|-----------|----------|--------|--------|-----------------|-----------|----------|--------|-------------|--------------|-----------|----------|--------|--------|
| Calibration   | 0         | 0        | 12.86  | 19.74  | Calibration     | 0         | 0        | 14.20  | 20.46       | Calibration  | 0         | 0        | 10.95  | 15.69  |
| Vanilla BP    | All       | 166.5    | 55.01  | 68.02  | Vanilla BP      | All       | 42.4     | 48.48  | 61.48 47.09 | Vanilla BP   | All       | 170.3    | 45.22  | 58.80  |
| Vanilla BP    | 5         | 15.0     | 39.54  | 51.86  | Vanilla BP      | 5         | 12.22    | 36.35  |             | Vanilla BP   | 5         | 59.5     | 27.41  | 37.90  |
| Vanilla BP    | 10        | 110.6    | 53.15  | 67.10  | Vanilla BP      | 10        | 22.46    | 46.01  | 58.70       | Vanilla BP   | 10        | 100.9    | 43.87  | 57.58  |
| Ours          | 5         | 0.14     | 39.34  | 51.86  | Ours            | 5         | 0.11     | 36.14  | 46.86       | Ours         | 5         | 0.58     | 27.42  | 37.88  |
| Ours          | 10        | 0.79     | 50.88  | 64.73  | Ours            | 10        | 0.76     | 44.90  | 57.50       | Ours         | 10        | 0.96     | 36.30  | 48.82  |
| DLV3 [8]      | #Layers   | GFLOPs   | mIoU   | mAcc   | DLV3-M [8]      | #Layers   | GFLOPs   | mIoU   | mAcc        | UPerNet [31] | #Layers   | GFLOPs   | mIoU   | mAcc   |
| Calibration   | 0         | 0        | 13.95  | 20.62  | Calibration     | 0         | 0        | 21.96  | 36.15       | Calibration  | 0         | 0        | 14.71  | 21.82  |
| Vanilla BP    | All       | 151.2    | 58.32  | 71.72  | Vanilla BP      | All       | 54.4     | 55.66  | 68.95 49.35 | Vanilla BP   | All       | 541.0    | 64.88  | 77.13  |
| Vanilla BP    | 5         | 18.0     | 40.85  | 53.16  | Vanilla BP      | 5         | 14.8     | 38.21  |             | Vanilla BP   | 5         | 503.9    | 47.93  | 61.67  |
| Vanilla BP    | 10        | 102.0    | 54.65  | 68.64  | Vanilla BP      | 10        | 33.1     | 47.95  | 61.49       | Vanilla BP   | 10        | 507.6    | 48.83  | 63.02  |
| Ours          | 5         | 0.31     | 33.09  | 44.33  | Ours            | 5         | 0.26     | 35.47  | 46.35       | Ours         | 5         | 1.97     | 47.04  | 60.44  |
| Ours          | 10        | 2.96     | 47.11  | 60.28  | Ours            | 10        | 1.40     | 45.53  | 58.99       | Ours         | 10        | 2.22     | 48.00  | 62.07  |

## 5. Experiments

Our experimental section consists of theoretical and practical evaluations. Sections 5.2-5.4 show the theoretical advantages of our method on image classification and semantic segmentation tasks with implementation-agnostic metrics ( e.g. , accuracy, FLOPs). Then, in Section 5.5, we show how these theoretical advantages translate into practical advantages ( i.e., speedup and memory savings) on real edge devices.

## 5.1. Experimental Setup

Classification: Following [24], we split every dataset into two highly non-i.i.d. partitions with the same size. Then, we pretrain our models on the first partition with a vanilla training strategy, and finetune the model on the other partition with different configurations for the training strat- egy ( i.e. , with/without gradient filtering, hyper-parameters, number of convolution layers to be trained). More details ( e.g. , hyper-parameters) are in the Supplementary.

Segmentation: Models are pretrained on Cityscapes [11] by MMSegmentation [10]. Then, we calibrate and finetune these models with different training strategies on the augmented Pascal-VOC12 dataset following [8], which is the combination of Pascal-VOC12 [12] and SBD [13]. More details are included in the supplementary material.

On-device Performance Evaluation: For CPU performance evaluation, we implement our method with MKLDNN[1] (a.k.a. OneDNN) v2.6.0 and compare it with the convolution BP method provided by MKLDNN. We test on three CPUs, namely Intel 11900KF, Quad-core CortexA72 (Jetson Nano) and Quad-core Cortex-A53 (Raspberry Pi-3b). For GPU performance evaluation, we implement our method on CUDNN v8.2 [9] and compare with the BP

method provided by CUDNN. We test on two GPUs, RTX 3090Ti and the edge GPU on Jetson Nano. Since both MKLDNN and CUDNN only support float32 BP, we test float32 BP only. Additionally, for the experiments on Jetson Nano, we record the energy consumption for CPU and GPU with the embedded power meter. More details ( e.g. , frequency) are included in the supplementary material.

## 5.2. ImageNet Classification

Table 2 shows our evaluation results on the ImageNet classification task. As shown, our method significantly reduces the FLOPs and memory required for BP, with very little accuracy loss. For example, for ResNet34, our method achieves 18.9 × speedup with 1.7% accuracy loss when training four layers; for MobileNetV2, we get a 1.2% better accuracy with 3.0 × speedup and 3.1 × memory savings. These results illustrate the effectiveness of our method. On most networks, TinyTL has a lower accuracy while consuming more resources compared to the baselines methods.

## 5.3. Semantic Segmentation

Table 3 shows our evaluation results on the augmented Pascal-VOC12 dataset. On a wide range of networks, our method constantly achieves significant speedup with marginal accuracy loss. For the large network UPerNet, our method achieves 229 × speedup with only 1% mIoU loss. For the small network PSPNet, our method speedups training by 140 × with only 2.27% mIoU loss. This shows the effectiveness of our method on a dense prediction task.

## 5.4. Hyper-Parameter Selection

Figure 4 shows our experimental results for ResNets under different hyper-parameter selection, i.e. number of convolution layers and patch size of gradient filter r × r . Of note, the y-axis (MFLOPs) in Figure 4 is log scale. More results are included in Supplementary Section G. We highlight three qualitative findings in Figure 4:

- a. For a similar accuracy, our method greatly reduces the number of operations (1 to 2 orders of magnitude), while for a similar number of computations, our method achieves a higher accuracy (2% to 5% better).

This finding proves the effectiveness of our method.

- b. Given the number of convolution layers to be trained, the more accurate method returns a better accuracy. Baseline ( i.e. , standard BP) uses the most accurate gradient, Ours-R4 (BP with gradient filter with patch size 4 × 4 ) uses the least accurate gradient; thus, Baseline &gt; Ours-R2 &gt; Ours-R4.

This finding is intuitive since the more accurate method should introduce smaller noise to the BP, e.g. , the gradient filtering with patch size 2 × 2 (Ours-R2) introduces less noise than with patch size 4 × 4 (Ours-R4). In Figure 5, we evaluate the relationship between accuracy and noise level introduced by gradient filtering. With a higher SNR ( i.e. , a lower noise level), a better accuracy is achieved.

Figure 4. Computation (#MFLOPs, log scale) and model accuracy [%] under different hyper-parameter selection. 'Baseline' means vanilla BP; 'Ours-R2/4' uses gradient filtering with patch size 2 × 2 / 4 × 4 during BP.

<!-- image -->

Figure 5. Relationship between accuracy and noise level introduced by the gradient filtering. As shown, accuracy increases as the SNR increases, i.e. , noise level decreases.

<!-- image -->

- c. Given the number of computations, the less accurate method returns the better accuracy by training more layers, i.e. , Ours-R4 &gt; Ours-R2 &gt; baseline.

This finding suggests that for neural network training with relatively low computational resources, training more layers with less accurate gradients is preferable than training fewer layers with more accurate gradients.

## 5.5. On-device Performance Evaluation

Figure 6 and Table 4 show our evaluation results on real devices. More results are included in the Supplementary Section I. As Figure 6 shows, on CPU, most convolution layers achieve speedups over 20 × with less than 50% memory consumption for gradient filtering with patch sizes 2 × 2 ; for gradient filtering with patch size 4 × 4 , the speedups are much higher, namely over 60 × . On GPU, the speedup is a little bit lower, but still over 10 × and 25 × , respectively. Furthermore, as Table 4 shows, our method saves over 95%

Figure 6. Speedup and normalized memory consumption results on multiple CPUs and GPUs under different test cases ( i.e. different input sizes, numbers of channels, etc.) Detailed configuration of these test cases are included in the supplementary material. 'R2', 'R4' mean using gradient filtering with 2 × 2 and 4 × 4 patch sizes, respectively. Our method achieves significant speedup with low memory consumption compared to all baseline methods. For example, on Jetson CPU with patch size 4 × 4 ('Jetson-R4' in left top figure), our method achieves 114 × speedup with only 33% memory consumption for most test cases.

<!-- image -->

Table 4. Normalized energy consumption for BP with gradient filtering for different patch sizes. Results are normalized w.r.t. the energy cost of standard BP methods. For instance, for edge CPU with a 4 × 4 patch, only 1.15% of energy in standard BP is used. Standard deviations are shown within brackets.

| Device   | Patch Size   | Normalized Energy Cost [STD]   |
|----------|--------------|--------------------------------|
| Edge     | 2 × 2        | 4.13% [0.61%]                  |
| CPU      | 4 × 4        | 1.15% [0.18%]                  |
| Edge     | 2 × 2        | 3.80% [0.73%]                  |
| GPU      | 4 × 4        | 1.22% [1.10%]                  |

energy for both CPU and GPU scenarios, which largely resolves one of the most important constraints on edge devices. All these experiments on real devices show that our method is practical for the real deployment of both highperformance and IoT applications.

Table 5. Evaluation of energy ratio defined in Equation (13) on models published on Torchvision. The ratio greater than 1 empirically verifies our assumption.

| Model              |   Ratio | Model              |   Ratio |
|--------------------|---------|--------------------|---------|
| (Wide)ResNet18-152 |   1.462 | VGG(bn)11-19       |   1.497 |
| DenseNet121-201    |   2.278 | EfficientNet b0-b7 |   1.240 |

## 5.6. Main Assumption Verification

We now empirically verify the assumption that the DC component dominates the frequency spectrum of the convolution kernel (Section 4.1). To this end, we collect the en- ergy ratio shown in Equation (13) from trained models published in Torchvision [23]. As Table 5 shows, for the convolution kernels in all these networks, we get a ratio greater than one, which means that the energy of DC components is larger than energy of all AC components. Thus, our assumption in Section 4.1 empirically holds true in practice.

## 6. Conclusions

In this paper, we have addressed the on-device model training for resource-constrained edge devices. To this end, a new gradient filtering method has been proposed to systematically reduce the computation and memory consumption for the back-propagation algorithm, which is the key bottleneck for efficient model training.

In Section 3, a new gradient filtering approach has been proposed to reduce the computation required for propagating gradients through the convolutional layers. The gradient filtering creates an approximate gradient feature map with fewer unique elements and a special structure; this reduces the computation by more than two orders of magnitude. Furthermore, we proved that the error introduced during back-propagation by our gradient filter is bounded so the influence of gradient approximation is limited.

Extensive experiments in Section 5 have demonstrated the efficiency and wide applicability of our method. Indeed, models can be finetuned with orders of magnitudes fewer computations, while having only a marginal accuracy loss compared to popular baseline methods.

Acknowledgements: This work was supported in part by the US National Science Foundation (NSF) grant CNS2007284.

## References

- [1] [Intel® oneapi deep neural network library (onednn). https://www.intel.com/content/www/us/en/ developer/tools/oneapi/onednn.html . 1, 2, 6](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html)
- [2] Mart´ ın Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dandelion Man´ e, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Vi´ egas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. TensorFlow: Large-scale machine learning on heterogeneous systems, 2015. Software available from tensorflow.org. 1, 2
- [3] Dan Alistarh, Demjan Grubic, Jerry Z. Li, Ryota Tomioka, and Milan Vojnovic. Qsgd: Communication-efficient sgd via gradient quantization and encoding. In Proceedings of the 31st International Conference on Neural Information Processing Systems , NIPS'17, page 1707-1718, Red Hook, NY, USA, 2017. Curran Associates Inc. 2
- [4] Ron Banner, Itay Hubara, Elad Hoffer, and Daniel Soudry. Scalable methods for 8-bit training of neural networks. In Proceedings of the 32nd International Conference on Neural Information Processing Systems , NIPS'18, page 5151-5159, Red Hook, NY, USA, 2018. Curran Associates Inc. 1, 2, 11, 17, 19
- [5] Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Animashree Anandkumar. signsgd: Compressed optimisation for non-convex problems. In International Conference on Machine Learning , pages 560-569. PMLR, 2018. 2
- [6] Han Cai, Chuang Gan, Ligeng Zhu, and Song Han. Tinytl: Reduce activations, not trainable parameters for efficient ondevice learning. arXiv preprint arXiv:2007.11622 , 2020. 1, 2, 6
- [7] Jianfei Chen, Yu Gai, Zhewei Yao, Michael W Mahoney, and Joseph E Gonzalez. A statistical framework for low-bitwidth training of deep neural networks. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 883-894. Curran Associates, Inc., 2020. 1, 2, 11, 16, 17, 19
- [8] Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam. Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587 , 2017. 6
- [9] Sharan Chetlur, Cliff Woolley, Philippe Vandermersch, Jonathan Cohen, John Tran, Bryan Catanzaro, and Evan Shelhamer. cudnn: Efficient primitives for deep learning. arXiv preprint arXiv:1410.0759 , 2014. 1, 2, 6
- [10] MMSegmentation Contributors. MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark. https : / / github . com / open mmlab/mmsegmentation , 2020. 6
- [11] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2016. 6
- [12] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The pascal visual object classes (voc) challenge. International Journal of Computer Vision , 88(2):303338, June 2010. 6
- [13] Bharath Hariharan, Pablo Arbelaez, Lubomir Bourdev, Subhransu Maji, and Jitendra Malik. Semantic contours from inverse detectors. In International Conference on Computer Vision (ICCV) , 2011. 6
- [14] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 770-778, 2016. 2, 6
- [15] Ziyang Hong and C. Patrick Yue. Efficient-grad: Efficient training deep convolutional neural networks on edge devices with gradient optimizations. ACM Trans. Embed. Comput. Syst. , 21(2), feb 2022. 2
- [16] Roger A Horn and Charles R Johnson. Matrix analysis . Cambridge university press, 2012. 3
- [17] Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran ElYaniv, and Yoshua Bengio. Quantized neural networks: Training neural networks with low precision weights and activations. J. Mach. Learn. Res. , 18(1):6869-6898, jan 2017. 2
- [18] Yoonho Lee, Annie S Chen, Fahim Tajwar, Ananya Kumar, Huaxiu Yao, Percy Liang, and Chelsea Finn. Surgical fine-tuning improves adaptation to distribution shifts. arXiv preprint arXiv:2210.11466 , 2022. 1, 2
- [19] Ji Lin, Wei-Ming Chen, John Cohn, Chuang Gan, and Song Han. Mcunet: Tiny deep learning on iot devices. In Annual Conference on Neural Information Processing Systems (NeurIPS) , 2020. 6
- [20] Ji Lin, Ligeng Zhu, Wei-Ming Chen, Wei-Chen Wang, Chuang Gan, and Song Han. On-device training under 256kb memory. arXiv preprint arXiv:2206.15472 , 2022. 1, 2
- [21] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 3431-3440, 2015. 6
- [22] Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations , 2017. 14
- [23] S´ ebastien Marcel and Yann Rodriguez. Torchvision the machine-vision package of torch. In Proceedings of the 18th ACM international conference on Multimedia , pages 14851488, 2010. 8
- [24] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communicationefficient learning of deep networks from decentralized data. In Artificial intelligence and statistics , pages 1273-1282. PMLR, 2017. 6, 14

- [25] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32 , pages 8024-8035. Curran Associates, Inc., 2019. 1, 2
- [26] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention , pages 234-241. Springer, 2015. 5
- [27] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 4510-4520, 2018. 6
- [28] Xiao Sun, Naigang Wang, Chia-Yu Chen, Jiamin Ni, Ankur Agrawal, Xiaodong Cui, Swagath Venkataramani, Kaoutar El Maghraoui, Vijayalakshmi (Viji) Srinivasan, and Kailash Gopalakrishnan. Ultra-low precision 4-bit training of deep neural networks. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems , volume 33, pages 1796-1807. Curran Associates, Inc., 2020. 2
- [29] Yue Wang, Ziyu Jiang, Xiaohan Chen, Pengfei Xu, Yang Zhao, Yingyan Lin, and Zhangyang Wang. E2-train: Training state-of-the-art cnns with over 80% energy savings. Advances in Neural Information Processing Systems , 32, 2019. 2
- [30] Diana Wofk, Fangchang Ma, Tien-Ju Yang, Sertac Karaman, and Vivienne Sze. Fastdepth: Fast monocular depth estimation on embedded systems. In 2019 International Conference on Robotics and Automation (ICRA) , pages 6101-6108. IEEE, 2019. 5
- [31] Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In Proceedings of the European conference on computer vision (ECCV) , pages 418-434, 2018. 6
- [32] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. Pyramid scene parsing network. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 2881-2890, 2017. 6
- [33] Kang Zhao, Sida Huang, Pan Pan, Yinghan Li, Yingya Zhang, Zhenyu Gu, and Yinghui Xu. Distribution adaptive int8 quantization for training cnns. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 35, pages 3483-3491, 2021. 2

In this supplementary material, we present:

- A: Post-training weight distribution.
- B: Detailed derivation for gradient filtering described in Section 3.
- C: Detailed proof for Proposition 1 in Section 4.1.
- D: Visualized computation analysis for ResNet18.
- E: Detailed experimental setup for Section 5.1.
- F: More experimental results for Semantic Segmentation in Section 5.3.
- G: More experimental results for hyper-parameter exploration on CIFAR datasets in Section 5.4.
- H: Experimental results for combining gradient filtering (our method) with existing INT8 gradient quantization approaches [4,7].
- I: More experimental results for on-device performance evaluation in Section 5.5.

## A. Post-training Weight Distribution

⊙˜√{˜⌈(∑√}⌊˜̂√]}{

<!-- image -->

⊔√∐̂/(⊔]⌈√˜√({⊔⊔}

## B. Gradient Filtering Derivation

In this section, we present the complete derivations for Equation (3) and Equation (5) in Section 3, namely the back propagation with gradient filtering. For convenience, Table 6 (reproduced from Table 1 in paper) lists commonly used symbols.

## B.1. Gradient Filtering

We have:

<!-- formula-not-decoded -->

Thus, for any entry in the approximated gradient ˜ g y , the value equals to the average of all neighboring elements within the same r × r patch, as shown in Figure 2 in the main manuscript. For the approximated gradient ˜ g y with batch size n , channel c , resolution ( H y , W y ) , there will be ( n × c ×glyph[ceilingleft] H y r glyph[ceilingright] ×glyph[ceilingleft] W y r glyph[ceilingright] ) unique numbers in ˜ g y . To simplify the following derivations, we rewrite the approximated gradient ˜ g y as follows:

<!-- formula-not-decoded -->

where ( h p , w p ) is the position of the patch and ( i, j ) is the offset within the patch. Since every element in the same patch has the exact same value, we denote this unique value with ˜ g u y , i.e. ,

<!-- formula-not-decoded -->

{̂}(〈]√˜̂√]}{(〈]√√√]̂√√]}{(}˜(⊙˜√{˜⌈(⊎√̂∐√˜({√√}glyph[arrowvertexdbl]√(]{({∐}(⌈({̂}}(⋃̂˜˜⊎

Figure 7. PCA projections of convolution kernels in the bottleneck layer of UperNet. Each point represents a 3 × 3 kernel. (a-b) compare the kernel before training (calibrated) and kernel trained with vanilla BP and our GF. (c) shows the distribution of directions [degree] in which the kernel was updated during training.

Figure 7 shows the PCA projections of convolution kernels in the bottleneck layer of an UperNet. Since our gradient filter (GF) only keeps the low-frequency part of the gradient signal (see Equation (7)), after applying the gradient filter, only the low-frequency part of the model is updated. As a result, as shown in Figure 7 (a) and (c), using the gradient filter limits the weights update to horizontal directions ( 0 ◦ and 180 ◦ ), as opposed to using vanilla back propagation (BP) where all directions are involved (Figure 7 (b) and (c)).

Table 6. Table of symbols we use.

| C x                    | Number of channels of x                                                                        |
|------------------------|------------------------------------------------------------------------------------------------|
| W x ,H x               | Width and height of x                                                                          |
| θ                      | Convolution kernel                                                                             |
| θ ′                    | Rotated θ , i.e. , θ ′ = rot180 ( θ )                                                          |
| r                      | Patch size ( r × r )                                                                           |
| g x , g y , g θ        | Gradients w.r.t. x, y, θ                                                                       |
| ˜ g y                  | Approximated gradient g y                                                                      |
| ˜ x, ˜ θ ′             | Sum of x and θ ′ over spatial dimensions (height and width)                                    |
| x [ n, c i ,h,w ]      | Element for feature map x at batch n , channel c i , pixel ( h,w )                             |
| θ [ c o , c i , u, v ] | Element for convolution kernel θ at output channel c o , input channel c i , position ( u, v ) |

## B.2. Approximation for Rotated Convolution Kernel θ ′

<!-- formula-not-decoded -->

## B.3. Approximation for Input Feature x

<!-- formula-not-decoded -->

Thus for every entry in approximated feature map ˜ x , the value equal to the sum of all neighboring elements within the same r × r patch. Following the definition of the gradient filter in Section B.1, we use the following symbols to simplify the derivation:

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

## B.4. Boundary Elements

As mentioned in Section 3, given the structure created by the gradient filters, the gradient propagation in a convolution layer can be simplified to weights summation and multiplication with few unique gradient values. This is true for all elements far away from the patch boundary because for these elements, the rotated kernel θ ′ only covers the elements from the same patch, which have the same value, thus the computation can be saved. However, for the elements close to the boundary, this is not true, since when convolving with boundary gradient elements, the kernel may cover multiple patches with multiple unique values instead of just one. To eliminate the extra computation introduced by the boundary elements, we pad each patch sufficiently such that every element is far away from boundary:

<!-- formula-not-decoded -->

For example, with the patch size 4 × 4 , the element at the spatial position (3 , 3) is on the boundary, so when we calculate ˜ g x [ n, c i , 3 , 3] by convolving the rotated kernel θ ′ with the approximated gradient ˜ g y :

<!-- formula-not-decoded -->

values of ˜ g y are from multiple patches and have different values ( e.g. , ˜ g y [ n, c o , 3 , 3] is from patch (0 , 0) while ˜ g y [ n, c o , 4 , 4] is from patch (1 , 1) ; they have different values). In our method, we simplify the Equation (25) by rewriting it in the following way:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Equation (26) is derived from Equation (25) by considering that patch (0 , 0) is sufficiently padded so that for elements with all offsets (3 + i, 3 + j ) , they have the same value, which is the unique value g u y [ n, c o , 0 , 0] .

For approximated input feature map ˜ x , we apply the same approximation for the boundary elements.

## B.5. Gradient w.r.t. Input (Equation (3) in Section 3.4)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

≈

∑

c

o

,u,v

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By expanding ˜ g u y to ˜ g y , we have:

<!-- formula-not-decoded -->

which is the Equation (3) in Section 3 in the paper.

From Equation (30) to Equation (32), we consider that the patch in the approximated gradient ˜ g y is padded sufficiently so they have the same value for all possible offsets

θ

[

c

, c

,

-

u,

-

v

·

]

o

i

(( h mod r ) + u, ( w mod r ) + v ) . If there is only one input channel and output channel for the convolutional layer as the Figure 2 in the paper shows, then Equation (34) become an element-wise multiplication, which is Equation (35) (also the Equation (3) in the Section 3.4).

## B.6. Gradient w.r.t. Convolution Kernel (Equation (5) in the Section 3.4)

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

By expanding ˜ x u and ˜ g u y to ˜ x and ˜ g y , respectively, we have:

<!-- formula-not-decoded -->

which is precisely Equation (5) in Section 3.

From Equation (37) to Equation (39), we consider that the patch in the approximated input feature map ˜ x is padded sufficiently thus they have the same value for all possible offsets (( h mod r ) + u, ( w mod r ) + v ) . For every given input/output channel pair ( c o , c i ) , Equation (40) represents the Frobenius inner product between ˜ x u and ˜ g u y .

## C. Detailed Proof for Proposition 1

In this section, we provide more details to the proof in Section 4.1. We use G x , G y and Θ to denote the gradients g x , g y and the convolution kernel θ in the frequency domain , respectively. G x [ u, v ] is the spectrum value at frequency ( u, v ) and δ is the 2D discrete Dirichlet function. Without losing generality and to simplify the proof, we consider the batch size is 1, the number of input/output channels is 1, namely C x = C y = 1 , and there is only one patch in ˜ g y .

The gradient returned by the gradient filtering can be written as:

<!-- formula-not-decoded -->

where glyph[circleasterisk] denotes convolution. By applying the discrete Fourier transformation, Equation (42) can be rewritten in the frequency domain as:

<!-- formula-not-decoded -->

˜ g y is the approximation for g y (so the ground truth for ˜ g y is g y ), and the SNR of ˜ g y equals to:

<!-- formula-not-decoded -->

where the numerator can be written as:

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

Because δ [ u, v ] = { 1 ( u, v ) = (0 , 0) 0 ( u, v ) = (0 , 0) , Equation (45) can be written as:

glyph[negationslash]

<!-- formula-not-decoded -->

glyph[negationslash]

By substituting the numerator in Equation (44) with Equation (46), we have:

<!-- formula-not-decoded -->

For the convolution layer, the gradient w.r.t. approximated variable ˜ x in the frequency domain is:

<!-- formula-not-decoded -->

5 As reminder, the total energy of a signal is the sum of energy in DC component and energy in AC components.

and its ground truth is:

<!-- formula-not-decoded -->

Similar to Equation (47), the SNR of g ˜ x is:

<!-- formula-not-decoded -->

Equation (50) can be rewritten as:

<!-- formula-not-decoded -->

Besides, the proposition's assumption (the DC component dominates the frequency spectrum of Θ ) can be written as:

glyph[negationslash]

<!-- formula-not-decoded -->

which is:

<!-- formula-not-decoded -->

thus, by combining Equation (51) and Equation (53), we have:

<!-- formula-not-decoded -->

which means that: SNR ˜ g x ≥ SNR ˜ g y . This completes our proof for error analysis. glyph[squaresolid]

In conclusion, as the gradient propagates, the noise introduced by the gradient filter becomes weaker and weaker compared to the real gradient signal. This property ensures that the error in gradient has only a limited influence on the quality of BP.

This proof can be extended to the more general case where batch size and the number of channels are greater than 1 by introducing more dimensions ( i.e. , batch dimension, channel dimension) into all equations listed above.

## D. Computation Analysis for ResNet18

In this section, we provide one more example for computation analysis in Section 4.2. Figure 8 shows the computation required by the convolution layers from ResNet18 with different patch sizes for gradient filtering. With reduced unique elements, our approach reduces the number of computations to 1 /r 2 of standard BP method; with structured gradient, our approach further reduces the number of computations to about 1 / ( r 2 H θ W θ ) of standard BP method.

## E. Detailed Experimental Setup

In this section, we extend the experimental setup in Section 5.1.

## E.1. ImageNet Classification

## E.1.1 Environment

ImageNet related experiments are conducted on IBM Power System AC922, which is equipped with a 40-core IBM Power 9 CPU, 256 GB DRAM and 4 NVIDIA Tesla V100 16GB GPUs. We use PyTorch 1.9.0 compiled with CUDA 10.1 as the deep learning framework.

## E.1.2 Dataset Split

We split the dataset into two non-i.i.d. partitions following the FedAvg method [24]. The label distribution is shown in Figure 9. Among all 1000 classes for the ImageNet, pretrain and finetune partitions overlap on only 99 classes, which suggests that our method can efficiently adapt the CNN model to data collected from new environments. For each partition, we randomly select 80% data as training data and 20% as validation data.

## E.1.3 Pretraining

We pretrain ResNet 18, ResNet 34, MobileNet-V2 and MCUNet with the same configuration. We use SGD optimizer. The learning rate of the optimizer starts at 0.05 and decays according to cosine annealing method [22] during training. Additionally, weight decay is set to 1 × 10 -4 and momentum is set to 0.9. We set batch size to 64. We randomly resize, randomly flip and normalize the image for data augmentation. We use cross entropy as loss function. Models are trained for 200 epochs and the model with the highest validation accuracy is kept for finetuning. Table 7 shows the pretrain accuracy.

## E.1.4 Finetuning

We adopt the hyper-parameter ( e.g. , momentum, weight decay, etc.) from pretraining. Several changes are made: models are finetuned for 90 epochs instead of 200; we apply Figure 8. Computation analysis for three convolution layers in of ResNet18 model. Since convolutional layers in every block of ResNet18 is similar, we use the last convolutional layer as the representative of all convolutional layers in the block. Minimum achievable computation is presented in Equation (16) in the paper. By reducing the number of unique elements, computations required by our approach drop to about 1 /r 2 compared with the standard BP method. By combining it ('+' in the figure) with structured gradient map, computations required by our approach drop further.

<!-- image -->

(a) Last convolutional layer in block 4 of ResNet18 with 512 input/output channels; the resolution of input feature map is 7 × 7 .

(b) Last convolutional layer in block 3 of ResNet18 with 256 input/output channels; the resolution of input feature map is 14 × 14 .

<!-- image -->

(c) Last convolutional layer in block 2 of ResNet18 with 128 input/output channels; the resolution of input feature map is 28 × 28 .

<!-- image -->

Table 7. Model pretraining accuracy on ImageNet.

| Model     | Accuracy   | Model        | Accuracy   |
|-----------|------------|--------------|------------|
| ResNet-18 | 73.5%      | MobileNet-V2 | 74.3%      |
| ResNet-34 | 76.4%      | MCUNet       | 71.4%      |

Figure 9. Label distribution for pretraining and finetuning datasets. Pretraining and finetuning partitions are split from ImageNet dataset.

<!-- image -->

L2 gradient clipping with threshold 2.0; linear learning rate warm-up for 4 epochs is introduced at the beginning of finetuning, i.e. , for the first 4 epochs, the learning rate grows linearly up to 0.05, then the learning rate decays according to cosine annealing method in the following epochs. Of note, to ensure a fair comparison, we use the same hyperparameters for all experiments, regardless of model type and training strategy.

## E.2. CIFAR Classification

## E.2.1 Environment

CIFAR related experiments are conducted on a GPU workstation with a 64-core AMD Ryzen Threadripper PRO 3995WXCPU, 512 GB DRAM and 4 NVIDIA RTX A6000 GPUs. We use PyTorch 1.12.0 compiled with CUDA 11.6 as the deep learning framework.

## E.2.2 Dataset Split

We split the dataset into two non-i.i.d. partitions following FedAvg method. The label distribution is shown in Figure 10. For CIFAR10, pretrain and finetune partitions overlap on 2 classes out of 10 classes in total. For CIFAR100, pretrain and finetune partitions overlap on 6 classes out of 100 classes.

## E.2.3 Pretraining

We pretrain ResNet18 and ResNet34 with the same configuration. We use the ADAM optimizer with a learning rate of 3 × 10 -4 and weight decay 1 × 10 -4 with no learning rate scheduling method. We use cross entropy as loss function. We set batch size to 128, and normalize the data before feeding it to the model. Models are trained for 30 and 50 epochs for CIFAR10 and CIFAR100, respectively. Then, the model with the highest accuracy is kept for finetuning. Table 8 shows the pretrain accuracy.

Figure 10. Label distribution for pretraining and finetuning datasets on CIFAR10 and CIFAR100. Pretraining and finetuning partitions are split from CIFAR10/100, respectively.

<!-- image -->

Table 8. Model pretraining accuracy on CIFAR10/100.

|          | ResNet18   | ResNet34   |
|----------|------------|------------|
| CIFAR10  | 95.1%      | 97.6%      |
| CIFAR100 | 75.5%      | 83.5%      |

## E.2.4 Finetuning

We adopt the training configuration from PSQ [7] with some changes. We use cross entropy loss with SGD optimizer for training. The learning rate of the optimizer starts at 0.05 and decays according to cosine annealing method during training. Momentum is set to 0 and weight decay is set to 1 × 10 -4 . We apply L2 gradient clipping with a threshold 2 . 0 . Batch normalization layers are fused with convolution layers before training, which is a common technique for inference acceleration.

## E.3. Semantic Segmentation

## E.3.1 Environment

ImageNet related experiments are conducted on IBM Power System AC922, which is equipped with a 40-core IBM Power 9 CPU, 256 GB DRAM and 4 NVIDIA Tesla V100 16GB GPUs. We use PyTorch 1.9.0 compiled with CUDA 10.1 as the deep learning framework. We implement our method based on MMSegmentation 0.27.0.

## E.3.2 Pretraining

We use models pretrained by MMSegmentation. Considering that the numbers of classes, image statistics, and model hyper-parameters may be different when applying on different datasets, we calibrate the model before finetuning.

We use SGD optimizer. The learning rate of the optimizer starts at 0.01 and decays exponentially during training. Additionally, weight decay is set to 5 × 10 -4 and momentum is set to 0.9. We set batch size to 8. We randomly crop, flip and photo-metric distort and normalize the image for data augmentation. We use cross entropy as loss function. For DeepLabV3, FCN, PSPNet and UPerNet, we calibrate the classifier ( i.e. , the last layer) and statistics in batch normalization layers for 1000 steps on the finetuning dataset. For DeepLabV3-MobileNetV2 and PSPNet-MobileNetV2, because the number of channels for convolutional layers in the decoder are different for models applied on different datasets, we calibrate the decoder and statistics in batch normalization layers for 5000 steps on the finetuning dataset.

## E.3.3 Finetuning

Wefinetune all models with the same configuration. We use the SGD optimizer. The learning rate of the optimizer starts at 0.01 and decays according to cosine anneling method during training. Additionally, weight decay is set to 5 × 10 -4 and momentum is set to 0.9. We set batch size to 8. We randomly crop, flip and photo-metric distort and normalize the image for data augmentation. We use cross entropy as loss function. Models are finetuned for 20000 steps. Experiments are repeated three times with random seed 233, 234 and 235.

## E.4. On-device Performance Evaluation

## E.4.1 NVIDIA Jetson Nano

We use NVIDIA Jetson Nano with quad-core Cortex-A57, 4 GB DRAM, 128-core Maxwell edge GPU for performance evaluation on both edge CPU and edge GPU. We use the aarch64-OS Ubuntu 18.04.6 provided by NVIDIA. During evaluation, the frequencies for CPU and GPU are 1.5 GHz and 921 MHz, respectively. Our code and library MKLDNN (a.k.a. OneDNN) are compiled on Jetson Nano with GCC 7.5.0, while libraries CUDA and CUDNN are compiled by NVIDIA. For CPU evaluations, our code and baseline are implemented with MKLDNN v2.6. For GPU evaluations, our code and baseline are implemented with CUDA 10.2 and CUDNN 8.2.1.

Before the evaluation for every test case, we warm up the device by running the test once. Then we repeat the test 10 times and report the average value for latency, energy consumption, etc.

Energy consumption is obtained by reading the embedded power meter in Jetson Nano every 20 ms.

## E.4.2 Raspberry Pi-3b

We use Raspberry Pi-3b with quad-core Cortex-A53, 1 GB DRAM for performance evaluation on CPU. We use the aarch64-OS Raspberry Pi OS. During evaluation, the frequency for CPU is 1.2 GHz. Our code and library MKLDNN are compiled on Raspberry Pi with GCC 10.2. Our code and baseline are implemented with MKLDNN v2.6.

Table 9. Experimental results for semantic segmentation task for UPerNet, DeepLabV3-MobileNetV2 (DLV3-M) and PSPNetMobileNetV2 (PSPNet-M). Models are pretrained on ADE20K dataset and finetuned on augmentated Pascal VOC12 dataset and Cityscapes dataset respectively. '#Layers' is short for 'the number of active convolutional layers' that are trained. Strategy 'Calibration' shows the accuracy when only the classifier and normalization statistics are updated to adapt differences ( e.g. different number of classes) between pretraining dataset and finetuning dataset.

| Pretrain: ADE20K Finetune: VOC12Aug   | Pretrain: ADE20K Finetune: VOC12Aug   | Pretrain: ADE20K Finetune: VOC12Aug   | Pretrain: ADE20K Finetune: VOC12Aug   | Pretrain: ADE20K Finetune: VOC12Aug   | Pretrain: ADE20K Finetune: VOC12Aug   | Pretrain: ADE20K Finetune: VOC12Aug   | Pretrain: ADE20K Finetune: VOC12Aug   | Pretrain: ADE20K Finetune: VOC12Aug   | Pretrain: ADE20K Finetune: VOC12Aug   | Pretrain: ADE20K Finetune: VOC12Aug   | Pretrain: ADE20K Finetune: VOC12Aug   | Pretrain: ADE20K Finetune: VOC12Aug   | Pretrain: ADE20K Finetune: VOC12Aug   | Pretrain: ADE20K Finetune: VOC12Aug   |
|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|
| UPerNet                               | #Layers                               | GFLOPs                                | mIoU                                  | mAcc                                  | PSPNet-M                              | #Layers                               | GFLOPs                                | mIoU                                  | mAcc                                  | DLV3-M                                | #Layers                               | GFLOPs                                | mIoU                                  | mAcc                                  |
| Calibration                           | 0                                     | 0                                     | 37.66                                 | 50.03                                 | Calibration                           | 0                                     | 0                                     | 30.93                                 | 52.01                                 | Calibration                           | 0                                     | 0                                     | 35.28                                 | 56.98                                 |
| Vanilla BP                            | All 5 10                              | 541.0 503.9 507.6                     | 67.23[0.24] 72.01[0.09] 72.01[0.19]   | 79.79[0.45] 81.97[0.30] 81.83[0.44]   | Vanilla BP                            | All 5 10                              | 42.41 12.22 22.46                     | 53.51[0.27] 48.88[0.11] 53.71[0.29]   | 67.01[0.19] 62.67[0.31] 67.93[0.32]   | Vanilla BP                            | All 5 10                              | 54.35 14.77 33.10                     | 60.78[0.21] 51.51[0.09] 57.63[0.10]   | 74.10[0.40] 66.08[0.44] 71.93[0.41]   |
| Ours                                  | 5 10                                  | 1.97 2.22                             | 71.76[0.11] 71.78[0.23]               | 81.57[0.07] 81.55[0.38]               | Ours                                  | 5 10                                  | 0.11 0.76                             | 48.59[0.08] 52.77[0.37]               | 62.28[0.30] 66.82[0.47]               | Ours                                  | 5 10                                  | 0.26 1.40                             | 49.40[0.00] 55.14[0.15]               | 64.13[0.54] 69.48[0.26]               |
| Pretrain: ADE20K Finetune: Cityscapes | Pretrain: ADE20K Finetune: Cityscapes | Pretrain: ADE20K Finetune: Cityscapes | Pretrain: ADE20K Finetune: Cityscapes | Pretrain: ADE20K Finetune: Cityscapes | Pretrain: ADE20K Finetune: Cityscapes | Pretrain: ADE20K Finetune: Cityscapes | Pretrain: ADE20K Finetune: Cityscapes | Pretrain: ADE20K Finetune: Cityscapes | Pretrain: ADE20K Finetune: Cityscapes | Pretrain: ADE20K Finetune: Cityscapes | Pretrain: ADE20K Finetune: Cityscapes | Pretrain: ADE20K Finetune: Cityscapes | Pretrain: ADE20K Finetune: Cityscapes | Pretrain: ADE20K Finetune: Cityscapes |
| UPerNet                               | #Layers                               | GFLOPs                                | mIoU                                  | mAcc                                  | PSPNet-M                              | #Layers                               | GFLOPs                                | mIoU                                  | mAcc                                  | DLV3-M                                | #Layers                               | GFLOPs                                | mIoU                                  | mAcc                                  |
| Calibration                           | 0                                     | 0                                     | 34.15                                 | 42.45 81.01[0.20]                     | Calibration                           | 0 All                                 | 0 84.82                               | 28.83 60.21[0.40] 42.09[0.43]         | 34.85 67.72[0.68] 48.70[0.49]         | Calibration Vanilla BP                | 0 All 5                               | 0 108.7 29.5                          | 41.33 71.12[0.14] 51.00[0.05]         | 48.65 79.81[0.04] 59.20[0.03]         |
| Vanilla BP                            | All 5 10                              | 1082.1 1007.7 1015.3                  | 73.02[0.14] 62.46[0.19] 64.01[0.21]   | 72.62[0.27] 73.11[0.32]               | Vanilla BP                            | 5 10                                  | 24.43 44.90                           | 54.03[0.24] 41.59[0.38]               | 61.48[0.10]                           |                                       | 10 5                                  | 66.2 0.50                             | 61.02[0.14]                           | 69.80[0.06]                           |
| Ours                                  | 5 10                                  | 3.94 4.43                             | 60.58[0.25] 62.14[0.24]               | 70.67[0.32] 71.41[0.27]               | Ours                                  | 5 10                                  | 0.22 1.51                             | 49.10[0.49]                           | 48.10[0.41] 56.93[1.43]               | Ours                                  | 10                                    | 2.74                                  | 48.83[0.07] 50.22[1.01]               | 56.87[0.08] 59.99[0.31]               |

Table 10. Layer configuration for test cases in Figure 6 in Section 5.5 in the paper.

|   No. |   #Input Channel |   #Output Channel |   Input Width |   Input Height |
|-------|------------------|-------------------|---------------|----------------|
|     0 |              128 |               128 |           120 |            160 |
|     1 |              256 |               256 |            60 |             80 |
|     2 |              512 |               512 |            30 |             40 |
|     3 |              512 |               512 |            14 |             14 |
|     4 |              256 |               256 |            14 |             14 |
|     5 |              128 |               128 |            28 |             28 |
|     6 |               64 |                64 |            56 |             56 |

Before the evaluation for every test case, we warm up the device by running the test once. Then we repeat the test 10 times and report the average value for latency, etc.

## E.4.3 Desktop

We use a desktop PC with Intel 11900KF CPU, 32 GB DRAM and RTX 3090 Ti GPU for perforamce evaluation on both desktop CPU and desktop GPU. We use x86 64OS Ubuntu 20.04. During evaluation, the frequencies for CPU and GPU are 4.7 GHz and 2.0 GHz respectively. Our code is compiled with GCC 9.4.0. MKLDNN is compiled by Anaconda (tag omp h13be974 0). CUDA and CUDNN are compiled by NVIDIA. For CPU evaluations, our code and baseline are implemented with MKLDNN v2.6. For GPU evaluations, our code and baseline are implemented with CUDA 11.7 and CUDNN 8.2.1.

Before the evaluation for every test case, we warm up the device by running the 10 times. Then we repeat the test 200 times and report the average value for latency, etc.

## E.4.4 Test Case Configurations

Table 10 lists the configurations for test cases shown in Figure 6 in the paper. In addition to the parameters shown in the table, for all test cases, we set the batch size to 32, kernel size to 3 × 3 , padding and stride to 1.

## F. More Results for Semantic Segmentation

In this section, we extend the experimental results shown in Section 5.3 (Table 3). Table 9 shows the experimental results for UPerNet, PSPNet-MobileNetV2 (PSPNet-M) and DeepLabV3-MobileNetV2 (DLV3-M) on two pairs of pretraing and finetuning datasets. These results further show the effectiveness of our method on a dense prediction task.

## G. More Results for CIFAR10/100 with Different Hyper-Parameter Selections

In this section, we extend the experimental results shown in Section 5.4 (Figure 4). Table 11 shows the experimental results for ResNet18 and ResNet34 on CIFAR datasets. For every model, we test our method with different patch sizes for gradient filtering and different numbers of active convolutional layers (#Layers in Table 11, e.g. , if #Layers equals to 2, the last two convolutional layers are trained while other layers are frozen). These results further support the qualitative findings in Section 5.4.

## H. Results for Combining Gradient Filtering with Gradient Quantization

In this section, we provide experimental results for combining our method, i.e. gradient filtering, with gradient quantization. Table 12 shows experimental results for ResNet18 and ResNet32 with gradient quantization methods PTQ [4] and PSQ [7] and different hyper-parameters.

Table 11. Experimental results on CIFAR10 and CIFAR100 datasets for ResNet18 and ResNet34 with different hyper-parameter selections. 'ACC' is short for accuracy. '#Layers' is short for 'the number of active convolution layers'. For example. #Layers equals to 2 means that only the last two convolutional layers are trained. 'Gradient Filter R2/4/7' use proposed gradient filtering method with patch size 2 × 2 , 4 × 4 and 7 × 7 , respectively.

|                     |         |                     |                               |                     |         |                     |                               | CIFAR100            | CIFAR100   | CIFAR100            | CIFAR100                      | CIFAR100            | CIFAR100   | CIFAR100            | CIFAR100                      |
|---------------------|---------|---------------------|-------------------------------|---------------------|---------|---------------------|-------------------------------|---------------------|------------|---------------------|-------------------------------|---------------------|------------|---------------------|-------------------------------|
| ResNet18            | #Layers | ACC[%]              | FLOPs                         | ResNet34            | #Layers | ACC[%]              | FLOPs                         | ResNet18            | #Layers    | ACC[%]              | FLOPs                         | ResNet34            | #Layers    | ACC[%]              | FLOPs                         |
| Vanilla BP          | 1 2 3 4 | 91.7 93.6 93.7 94.4 | 128.25M 487.68M 847.15M 1.14G | Vanilla BP          | 1 2 3 4 | 94.2 96.6 96.6 96.8 | 128.25M 487.68M 847.13M 1.21G | Vanilla BP          | 1 2 3 4    | 73.8 77.6 77.6 78.0 | 128.39M 487.82M 847.29M 1.14G | Vanilla BP          | 1 2 3 4    | 76.9 82.0 82.1 83.0 | 128.39M 487.82M 847.27M 1.21G |
| +Gradient Filter R2 | 1 2 3 4 | 91.5 92.7 92.8 93.9 | 8.18M 26.80M 45.45M 60.01M    | +Gradient Filter R2 | 1 2 3 4 | 94.2 96.6 96.5 96.6 | 8.18M 26.80M 45.44M 64.07M    | +Gradient Filter R2 | 1 2 3 4    | 73.7 75.6 75.6 76.4 | 8.31M 26.94M 45.59M 60.15M    | +Gradient Filter R2 | 1 2 3 4    | 77.0 81.1 81.1 82.0 | 8.31M 26.94M 45.58M 64.21M    |
| +Gradient Filter R4 | 1 2 3 4 | 91.4 92.7 92.8 93.3 | 1.88M 7.93M 13.99M 19.12M     | +Gradient Filter R4 | 1 2 3 4 | 94.3 96.4 96.4 96.1 | 1.88M 7.93M 13.98M 20.04M     | +Gradient Filter R4 | 1 2 3 4    | 73.7 74.9 74.9 75.2 | 2.02M 8.07M 14.12M 19.26M     | +Gradient Filter R4 | 1 2 3 4    | 76.9 80.4 80.4 80.5 | 2.02M 8.07M 14.12M 20.17M     |
| +Gradient Filter R7 | 1 2 3 4 | 91.5 91.5 91.7 92.6 | 303.10K 3.21M 6.12M 8.90M     | +Gradient Filter R7 | 1 2 3 4 | 94.2 95.8 96.0 96.0 | 303.10K 3.21M 6.12M 9.03M     | +Gradient Filter R7 | 1 2 3 4    | 73.7 74.1 74.1 75.4 | 441.34K 3.35M 6.26M 9.04M     | +Gradient Filter R7 | 1 2 3 4    | 76.9 80.4 80.3 80.3 | 441.34K 3.35M 6.26M 9.17M     |

Figure 11. Energy savings and overhead resuls on multiple CPUs and GPUs under different test cases ( i.e. , different input sizes, number of channels, etc..). For test case 4 and 5 with patch size 4 × 4 (Jetson-R4) on GPU, the latency of our method is too small to be captured by the power meter with a 20 ms sample rate so the energy savings data is not available. For most test cases with patch size 4 × 4 , our method achieves over 80 × energy savings with less than 20% overhead.

<!-- image -->

Both forward propagation and backward propagation are quantized to INT8. These results support the wide applicability of our method.

## I. More Results for On-device Performance Evaluation

In this section, we extend the experimental results shown in Section 5.5. Figure 11 shows the energy savings and overhead of our method. For most test cases with patch 4 × 4 , we achieve over 80 × energy savings with less than 20% overhead on both CPU and GPU. Moreover, for the test case 1 on Raspberry Pi-3b CPU, the forward propagation is even faster when applied our method (which results in negtive overheads). These results further show that our method is practical for the real deployment of both highperformance and IoT applications.

Table 12. Experimental results for ResNet18 and ResNet34 with different gradient quantization methods ( i.e. , PTQ [4] and PSQ [7]) and hyper-parameter selections on CIFAR10/100. Feature map, activation, weight and gradient are quantized to INT8. 'ACC' is short for accuracy. '#Layers' is short for 'the number of active convolution layers'. For example. #Layers equals to 2 means that the last two convolutional layers are trained. 'Gradient Filter R2/4/7' use proposed gradient filtering method with patch size 2 × 2 , 4 × 4 and 7 × 7 , respectively.

|                  | CIFAR10   | CIFAR10   | CIFAR10   | CIFAR10          | CIFAR10   | CIFAR10   | CIFAR10   | CIFAR100         | CIFAR100   | CIFAR100   | CIFAR100    | CIFAR100         | CIFAR100   | CIFAR100   | CIFAR100    |
|------------------|-----------|-----------|-----------|------------------|-----------|-----------|-----------|------------------|------------|------------|-------------|------------------|------------|------------|-------------|
|                  | ResNet18  | ResNet18  | ResNet18  | ResNet34         | ResNet34  | ResNet34  | ResNet34  | ResNet18         | ResNet18   | ResNet18   | ResNet18    | ResNet34         | ResNet34   | ResNet34   | ResNet34    |
| Strategy         | #Layers   | ACC[%]    | #OPs      | Strategy         | #Layers   | ACC[%]    | #OPs      | Strategy         | #Layers    | ACC[%]     | #OPs        | Strategy         | #Layers    | ACC[%]     | #OPs        |
|                  | 1         | 91.6      | 128.25M   |                  | 1         | 93.6      | 128.25M   |                  | 1          | 74.0       | 128.39M     |                  | 1          | 76.4       | 128.39M     |
|                  | 2         | 93.2      | 487.68M   |                  | 2         | 96.2      | 487.68M   |                  | 2          | 77.8       | 487.82M     |                  | 2          | 80.3       | 487.82M     |
| PTQ              | 3         | 93.5      | 847.15M   | PTQ              | 3         | 96.2      | 847.13M   | PTQ              | 3          | 77.9       | 847.29M     | PTQ              | 3          | 80.5       | 847.27M     |
|                  | 4         | 94.4      | 1.14G     |                  | 4         | 96.5      | 1.21G     |                  | 4          | 77.9       | 1.14G       |                  | 4          | 82.2       | 1.21G       |
| PTQ              | 1         | 91.4      | 8.18M     | PTQ              | 1         | 93.5      | 8.18M     | PTQ              | 1          | 73.9       | 8.31M       | PTQ              | 1          | 76.5       | 8.31M       |
| +Gradient        | 2         | 92.6      | 26.80M    | +Gradient        | 2         | 95.9      | 26.80M    | +Gradient        | 2          | 75.7       | 26.94M      | +Gradient        | 2          | 80.0       | 26.94M      |
| Filter           | 3         | 92.7      | 45.45M    | Filter           | 3         | 96.0      | 45.44M    | Filter           | 3          | 75.9       | 45.59M      | Filter           | 3          | 80.1       | 45.58M      |
| R2               | 4         | 93.7      | 60.01M    | R2               | 4         | 96.2      | 64.07M    | R2               | 4          | 76.3       | 60.15M      | R2               | 4          | 80.9       | 64.21M      |
| PTQ              | 1         | 91.3      | 1.88M     | PTQ              | 1         | 93.6      | 1.88M     | PTQ              | 1          | 73.7       | 2.02M       | PTQ              | 1          | 76.5       | 2.02M       |
| +Gradient        | 2         | 92.5      | 7.93M     | +Gradient        | 2         | 95.6      | 7.93M     | +Gradient        | 2          | 75.1       | 8.07M       | +Gradient        | 2          | 79.5       | 8.07M       |
| Filter           | 3         | 92.7      | 13.99M    | Filter           | 3         | 95.6      | 13.98M    | Filter           | 3          | 75.4       | 14.12M      | Filter           | 3          | 79.5       | 14.12M      |
| R4               | 4         | 93.4      | 19.12M    | R4               | 4         | 95.6      | 20.04M    | R4               | 4          | 76.1       | 19.26M      | R4               | 4          | 80.5       | 20.17M      |
| PTQ              | 1         | 91.2      | 303.10K   | PTQ              | 1         | 93.6      | 303.10K   | PTQ              | 1          | 73.7       | 441.34K     | PTQ              | 1          | 76.5       | 441.34K     |
| +Gradient        | 2         | 91.5      | 3.21M     | +Gradient        | 2         | 95.5      | 3.21M     | +Gradient        | 2          | 74.5       | 3.35M       | +Gradient        | 2          | 79.4       | 3.35M       |
| Filter           | 3         | 91.6      | 6.12M     | Filter           | 3         | 95.4      | 6.12M     | Filter           | 3          | 74.5       | 6.26M       | Filter           | 3          | 79.5       | 6.26M       |
| R7               | 4         | 92.6      | 8.90M     | R7               | 4         | 95.5      | 9.03M     | R7               | 4          | 75.3       | 9.04M       | R7               | 4          | 79.6       | 9.17M       |
|                  | 1         | 91.4      | 128.25M   |                  | 1         | 93.6      | 128.25M   |                  | 1          | 73.9       | 128.39M     |                  | 1          | 76.4       | 128.39M     |
| PSQ              | 2         | 93.3      | 487.68M   | PSQ              | 2         | 96.1      | 487.68M   | PSQ              | 2          | 77.7       | 487.82M     | PSQ              | 2          | 80.3       | 487.82M     |
| PSQ              | 1         | 91.3      | 8.18M     | PSQ              | 1         | 93.5      | 8.18M     | PSQ              | 1          | 73.8       | 8.31M       | PSQ              | 1          | 76.4       | 8.31M       |
| +Gradient        | 2         | 92.6      | 26.80M    | +Gradient        | 2         | 96.0      | 26.80M    | +Gradient        | 2          | 76.0       | 26.94M      | +Gradient        | 2          | 80.1       | 26.94M      |
| Filter           | 3         | 92.8      | 45.45M    |                  | 3         | 96.1      | 45.44M    | Filter           | 3          | 75.9       | 45.59M      | Filter           | 4          | 80.9       | 45.58M      |
|                  | 4         | 91.4      | 60.01M    | Filter R2        | 1         |           | 64.07M    |                  |            | 76.3       | 60.15M      |                  | 3          | 80.0       | 64.21M      |
| R2               |           | 93.7      | 1.88M     | PSQ              | 4         | 96.1      |           | R2               | 4          | 73.5       | 2.02M       | R2               |            |            | 2.02M       |
| PSQ              | 1 2       | 92.6      | 7.93M     | +Gradient        | 2         | 93.6      | 1.88M     | PSQ +Gradient    | 1 2        | 75.3       |             | PSQ +Gradient    | 1          | 76.5       | 8.07M       |
| +Gradient Filter |           | 92.7      |           |                  | 3         | 95.6      | 7.93M     |                  | 3          |            | 8.07M       |                  | 2          | 79.5       | 14.12M      |
|                  | 3         | 93.2      | 13.99M    |                  |           | 95.6      | 13.98M    | Filter           |            | 75.1       | 14.12M      | Filter           | 3          | 79.6       |             |
| R4               | 4         |           | 19.12M    | Filter R4        | 4         | 95.5      | 20.04M    | R4               | 4          | 76.2       | 19.26M      | R4               | 4          | 80.2       | 20.17M      |
| PSQ              | 1         | 91.2      | 303.10K   | PSQ              | 1         | 93.6      | 303.10K   | PSQ              | 1          | 73.5       | 441.34K     | PSQ              | 1          | 76.5       | 441.34K     |
| +Gradient Filter | 2         | 91.4      | 3.21M     | +Gradient Filter | 2         | 95.5      | 3.21M     | +Gradient Filter | 2 3        | 74.4       | 3.35M 6.26M | +Gradient Filter | 2 3        | 79.5       | 3.35M 6.26M |
| R7               | 3         | 91.6      | 6.12M     |                  | 3         | 95.4      | 6.12M     |                  |            | 74.5       |             |                  | 4          | 79.6       |             |
|                  | 4         |           | 8.90M     | R7               | 4         | 95.5      | 9.03M     | R7               |            |            | 9.04M       |                  |            | 79.6       |             |
|                  |           | 92.7      |           |                  |           |           |           |                  | 4          | 75.5       |             | R7               |            |            | 9.17M       |