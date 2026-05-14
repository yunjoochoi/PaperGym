## Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP

Feng Liang * 1 , Bichen Wu 2 , Xiaoliang Dai 2 , Kunpeng Li 2 , Yinan Zhao 2 , Hang Zhang † 3 ,

Peizhao Zhang 2 , Peter Vajda 2 , Diana Marculescu 1 1 The University of Texas at Austin, 2 Meta Reality Labs, 3 Cruise

{ jeffliang,dianam } @utexas.edu , { wbc,stzpz,vajdap } @meta.com

[https://jeff-liangf.github.io/projects/ovseg](https://jeff-liangf.github.io/projects/ovseg)

## Abstract

Open-vocabulary semantic segmentation aims to segment an image into semantic regions according to text descriptions, which may not have been seen during training. Recent two-stage methods first generate class-agnostic mask proposals and then leverage pre-trained visionlanguage models, e.g ., CLIP, to classify masked regions. We identify the performance bottleneck of this paradigm to be the pre-trained CLIP model, since it does not perform well on masked images. To address this, we propose to finetune CLIP on a collection of masked image regions and their corresponding text descriptions. We collect training data by mining an existing image-caption dataset ( e.g ., COCO Captions), using CLIP to match masked image regions to nouns in the image captions. Compared with the more precise and manually annotated segmentation labels with fixed classes ( e.g ., COCO-Stuff), we find our noisy but diverse dataset can better retain CLIP's generalization ability. Along with finetuning the entire model, we utilize the 'blank' areas in masked images using a method we dub mask prompt tuning. Experiments demonstrate mask prompt tuning brings significant improvement without modifying any weights of CLIP, and it can further improve a fully finetuned model. In particular, when trained on COCO and evaluated on ADE20K150, our best model achieves 29.6% mIoU, which is +8.5% higher than the previous state-of-the-art. For the first time, open-vocabulary generalist models match the performance of supervised specialist models in 2017 without dataset specific adaptations.

## 1. Introduction

Semantic segmentation aims to group pixels into meaningful regions with corresponding semantic categories. Although remarkable progress has been made [6, 7, 9, 29, 41], modern semantic segmentation models are mainly trained with pre-defined categories, failing to generalize to unseen classes. On the contrary, humans understand scenes in an open-vocabulary manner, typically with thousands of categories [2]. To approach human-level perception, this paper studies open-vocabulary semantic segmentation where the model segments an image by arbitrary categories described by texts.

* Work done during an internship at Meta Reality Labs.

† Work done while at Meta Reality Labs.

Vision-language models, e.g ., CLIP [35], learn rich multi-modal features from billion-scale image-text pairs. Witnessing its superior open-vocabulary classification ability, prior works propose to use pre-trained vision-language models for open-vocabulary segmentation [11, 16, 23, 40]. Among them, two-stage approaches have shown great potential: they first generate class-agnostic mask proposals and then leverage pre-trained CLIP to perform openvocabulary classification (see Figure 1(b)). Their success relies on two assumptions: (1) the model can generate classagnostic mask proposals (2) pre-trained CLIP can transfer its classification performance to masked image proposals.

To examine these two assumptions, we conduct the following analysis. First, we assume an 'oracle' mask generator and an ordinary CLIP classifier. We use ground-truth masks as region proposals, and feed masked images to a pre-trained CLIP for classification. This model only reaches an mIoU of 20.1% on the ADE20K-150 dataset. Next, we assume an 'oracle' classifier but an ordinary mask proposal generator - a MaskFormer ( [9]) pre-trained on the COCO dataset. We first extract masked region proposals, then compare each region with ground-truth object masks, find the object with the highest overlap, and assign the object label to this extracted region. This model, despite imperfect region proposals, reaches a significantly higher mIoU of 66.5%.

This analysis clearly shows that pre-trained CLIP can not perform satisfactory classification over masked images, and it is the performance bottleneck of two-stage openvocabulary segmentation models. We hypothesize that this is caused by the significant domain gap between masked images and CLIP's training images. CLIP is pre-trained on natural images with minimal data augmentation [35]. On the other hand, mask proposals are cropped and re-sized from original images, and are further corrupted by noisy segmentation masks, see examples in Figure 1 (b).

Figure 1. (a) CLIP is pre-trained with natural images with little data augmentation. (b) Two-stage open-vocabulary semantic segmentation approaches first generate class-agnostic mask proposals and then leverage pre-trained CLIP to do open-vocabulary classification. The input of the CLIP model is cropped masked images, which have huge domain gap from the natural images. (c) Our analysis reveals that pre-trained CLIP does not work well on masked images.

<!-- image -->

To address this, we propose to adapt CLIP by finetuning it on masked images and corresponding text labels. One direct solution is to use segmentation labels, e.g ., from the COCO-stuff dataset. However, this leads to bad generalization to unseen classes (Section 4.3.1). Such manually annotated masks are accurate but classes are limited to a closed set ( e.g ., 171 classes for COCO-stuff). We hypothesize that the lack of text diversity causes the finetuned CLIP to lose the generalization ability to open vocabulary concepts. Instead, we collect training data by mining an existing image-caption dataset ( e.g ., COCO Captions). Given an image-caption pair, we first extract nouns in the caption, and generate class-agnostic masked region proposals using a pre-trained segmentation model. Then, with a pre-trained CLIP model, we assign the best-matching proposal to each extracted noun. By learning from this weakly-supervised alignments between masked images and novel categories, the adapted CLIP better retains its generalization ability for open vocabulary classification.

The next question is how to effectively finetune CLIP? The most notable difference between a masked image and a natural image is that background pixels in a masked image are masked out, leading to many blank areas, which will be converted to 'zero tokens' when feeding to CLIP transformers. Such tokens not only contain no useful information, but also bring domain distribution shift to the model (since such tokens don't exist in natural images) and cause performance degradation. To mitigate this, we propose mask prompt tuning, ´ a la visual prompt tuning [20]. When tokenizing a masked image, we replace the 'zero tokens' with learnable prompt tokens. During finetuning, we either train prompts only and freeze CLIP's weights, or train both of them. We find that mask prompt tuning alone significantly improves CLIP's performance on masked images. This is a crucial property for multi-task scenarios where we cannot change CLIP's weight since it is shared with other tasks. When combined with full model finetuning, mask prompt tuning can further improve the performance by a non-trivial margin (see Section 4.3.2).

In our experiments, we measure the open-vocabulary segmentation performances on holdout segmentation datasets in a 'zero-shot' manner - we do not adapt the model for each evaluation dataset. We train our model using COCO-stuff [5] dataset with captions from [8], and test on challenging ADE20K (A-150, A-847 for 150/846 categories) [43], Pascal Context (PC-59, PC-459 for 59/459 categories) [33] and PASCAL VOC (PAS-20) [15]. Our best model achieves 29.6% mIoU on A-150, which is +8.5% than the state-of-the-art OpenSeg [16] under the same setting. On more challenging A-847 and PC-459, our model sets up a new state-of-the-art of 9.0%, 12.4% mIoU, surpassing the previous best solution by +2.7% and 3.4%. Moreover, for the first time, we show open-vocabulary generalist models can match the performance of supervised specialist models [6,29,45] in 2017 without dataset specific adaptations.

In summary our contributions include: (1) Our analysis reveals the pre-trained CLIP does not perform well on mask proposals, making it the performance bottleneck of two-stage approaches. (2) We collect diverse maskcategory pairs from captions to adapt CLIP for masked images and retain its generalization ability. (3) We propose mask prompt tuning specifically for masked image adaptation. This method does not change CLIP's weight, enabling multi-task weight sharing. (4) For the first time, we show open-vocabulary generalist models can match the performance of supervised specialist models in 2017 without dataset specific adaptations.

## 2. Related Work

Pre-trained vision-language models [19, 25, 35, 36] connect the visual concepts with textual description. Pretrained CLIP [35] has strong open-vocabulary classifica- tion ability, i.e ., classifying an image with arbitrary categories described by language. Pre-trained CLIP has empowered many computer vision tasks with the language ability, such as image manipulation [34], image generation [10], object detection [17, 42] and image segmentation [11, 12, 16, 21, 23, 31, 39, 40]. Our work is similar to RegionCLIP [42], which adapts CLIP for object detection by finetuning on region proposals. Our method differs from RegionCLIP since (1) we adapt CLIP to process masked images while RegionCLIP processes complete region crops; (2) We leverage blank areas in masked images and propose mask prompt tuning, which adapts CLIP without changing its weights. This enables sharing CLIP's weight with other tasks in multi-task scenarios. This is not supported by RegionCLIP.

Figure 2. Two-stage approaches consist of one segmentation model, e.g ., MaskFormer, and one CLIP model. Firstly, the modified MaskFormer is trained with CLIP's text embeddings so as to perform open-vocabulary segmentation. (Section 3.1). We then use the pre-trained segmentation model to generate class-agnostic proposals and align proposals with extracted nouns from corresponding captions (Section 3.2). After collecting diverse mask-category pairs, we finetune CLIP with the proposed mask prompt tuning (Section 3.3).

<!-- image -->

Open-vocabulary segmentation aims to understand an image with arbitrary categories described by texts. Pioneering work ZS3Net [4] uses generative models to synthesize pixel-level features by word embeddings of unseen class. SPNet [37] utilizes the word embeddings, e.g ., word2vec [32], to align the semantic meaning with visual features. GroupViT [38] groups segmentation masks directly from text supervision. More recently, researchers propose to leverage the pre-trained CLIP [35] for openvocabulary semantic segmentation. LSeg [23] aligns pixel embeddings to the text embedding of the corresponding semantic class, which is generated by CLIP's text encoder. Unlike pixel-level LSeg, OpenSeg [16] proposes to align the segment-level visual features with text embedding via region-word grounding. Our approach falls into the category of two-stage approaches, such as ZSSeg [40] and ZegFormer [11]: they first generate class-agnostic mask proposals and then utilize pre-trained CLIP to perform openvocabulary classification. Unlike ZSSeg and ZegFormer which directly use the original CLIP for masked image classification, we adapt CLIP to improve performance.

Prompt tuning is a strategy to adapt large-scale pretrained models to new tasks. The idea originated from natural language processing [22, 24, 27], and recent work extends prompt tuning to computer vision. CoOp [44] preappends the category words with learnable vectors to adapt CLIP for many recognition tasks. The textual prompt tuning is also widely used in open-vocabulary object detection [14] and semantic segmentation [40]. Our mask prompt tuning is more relevant to prompt tuning in the visual domain [1, 20] where learnable vectors are applied to the image domain. Unlike visual prompt tuning [20] that inserts additional tokens before the actual image tokens, we replace masked tokens with learnable prompts. Furthermore, mask prompt tuning brings additional improvement over a fully finetuned model (Section 4.3.2). Such additional improvements have not been reported by prior work.

## 3. Method

In this section, we first revisit the two-stage openvocabulary segmentation methods [11,40]. Then we discuss how to obtain a dataset of mask-category pairs to finetune CLIP. Last, we discuss the proposed mask prompt tuning technique to adapt CLIP for masked images.

## 3.1. Two-stage models for open-vocabulary semantic segmentation

Our two-stage open-vocabulary semantic segmentation model is shown in Figure 2. It consists of a segmentation model that generates mask proposals, and an open vocabulary classification model.

Following [11, 40], we choose MaskFormer [9] as the segmentation model. Unlike per-pixel segmentation mod- els [6, 29], MaskFormer predicts a set of N mask proposals and corresponding class predictions. Each proposal is represented by an H × W binary mask, indicating the location of the target object. The class prediction is a C -dimensional distribution, where C is the number of classes in the training set. Following [40], we modify MaskFormer such that for each mask, it generates a C -dimensional proposal embedding, where C is the embedding dimension of a CLIP model (512 for ViT-B/16 and 768 for ViT-L/14). This change allows MaskFormer to perform open-vocabulary segmentation. Specifically, suppose we would like to classify the mask to K categories, we can first use a CLIP model's text encoder to generate K text embeddings for each class as { t k | t k ∈ R C } k =1 , ··· ,K . Next, we compare each mask embedding v i with the text embedding, and predict the classk probability as p i,k = exp( σ ( v i , t k ) /τ ) / ∑ k (exp( σ ( v i , t k ) /τ )) . Here σ ( · , · ) denotes the cosine similarity between two embedding vectors, and τ is the temperature coefficient [35]. We train the modified MaskFormer on the COCO-Stuff dataset [5] with 171 classes. We use CLIP's text encoder to process class names to generate text embeddings. We also append a learnable embedding ∅ to represent the category of 'no object'. For other training settings, we follow the original MaskFormer [9].

Note that the mask proposal generator trained this way is not strictly 'class-agnostic', as the definition of an object is determined by the class definitions in the training set. For example, if the training set only contains 'person' as a class, it is not likely the model will automatically segment a person into 'face', 'hand', 'body', or finer body parts. How to train a general and class agnostic model to generate mask proposals is an important topic but is beyond the scope of this paper.

In addition to MaskFormer's prediction, following [11, 40], we add a parallel prediction branch using CLIP. MaskFormer generates mask proposals { M i | M i ∈ { 0 , 1 } H × W } i =1 , ··· ,N where 1 and 0 denotes foreground and background. For each mask, we select a tight bounding box that includes all foreground pixels, crop the image, mask out backgrounds, and re-size to CLIP's resolution. We feed mask proposali to CLIP and compute classk probability as ˆ p i,k . Weensemble both predictions to compute final prediction as p (1 -λ ) i,k ∗ ˆ p λ i,k where λ ∈ [0 , 1] . We fuse mask-wise predictions to semantic segmentation using MaskFormer's fusion module.

As discussed in Section 1 and Figure 1 (c), our analysis show that CLIP does not work well on such masked images. Specifically, CLIP is trained on natural images with little data augmentation [35]. However, masked images as shown in Figure 1 (b) contain a lot of 'blank regions'. Such a significant domain gap makes it difficult for CLIP to transfer its classification performance. We also tried cropping the proposals without masking out background pixels. However, we observe worse performance (see Appendix). We conjecture that keeping background pixels makes it more confusing for CLIP to correctly classify the foreground.

Figure 3. For the given image-cation pair, only "apple" and "orange" are categories in COCO. By extracting nouns from captions, we can also get a novel "teapot" category.

<!-- image -->

## 3.2. Collecting diverse mask-category pairs from captions

To adapt CLIP to better process masked images, we propose to finetune CLIP on a dataset consisting of masked image and text pairs. One direct solution is to leverage manually annotated segmentation labels, e.g ., from COCO-Stuff. Such labels are accurate but have a closed set of categories. We try this solution and collect 965K mask-category pairs spanning 171 classes ( e.g ., banana, orange) from COCOStuff. Then we finetune the CLIP's image encoder, while freezing the text encoder, following [42]. However, we observe that this naive approach limits the generalization ability of CLIP, as the performance drops if there are more unseen classes (see Section 4.3.1). We hypothesize that due to the limited text vocabulary, the finetuned CLIP over-fits to the 171 classes, losing the ability to generalize to unseen categories.

Compared with segmentation labels, image captions contain much richer information about images and involve a much larger vocabulary. For example, in Figure 3, the image caption is "There are apple and orange and teapot." . Though "apple" and "orange" are valid classes in COCO-Stuff, other concepts are not valid classes and are ignored.

Based on this observation, we designed a self-labeling strategy [16, 42] to extract mask-category pairs. As in Figure 3, given an image, we first use a pre-trained MaskFormer to extract masked proposals. Meanwhile, from the Mask prompt corresponding image caption, we extract all nouns using an off-the-shelf language parser [3], and treat them as potential classes. Then, we use CLIP to pair the most matching mask proposal to each class. From COCO-Captions [8], we collect 1.3M mask-category pairs with 27K unique nouns using 5 captions per image, or 440K pairs with 12K nouns using 1 caption per image. Experiments show this noisy but diverse mask-category dataset leads to significantly better performance than manual segmentation labels (see Section 4.3.1).

<!-- image -->

Figure 4. The proposed mask prompt tuning can adapt CLIP to masked images without changing its weights. We replace the zero tokens from masked patches with learnable mask prompts.

## 3.3. Mask prompt tuning

After collecting the dataset, a natural question is how to finetune CLIP effectively? The most notable difference between a masked image and a natural image is that background pixels in a masked images are set to zeros, leading to many 'blank areas'. When feeding masked images to CLIP, images will be divided into non-overlapping patches and subsequently tokenized. Those blank areas will then become zero tokens. Such tokens not only contain no useful information but also bring domain distribution shift to the model (since such tokens don't exist in natural images) and cause performance degradation. To mitigate this, we propose a technique called mask prompt tuning , ` a la visual prompt tuning [20]. Specifically, when feeding into CLIP, a masked image will be tokenized to a tensor T ∈ R N p × E , where N p is the number of patches, and E is the token dimension. The masked image also comes with a condensed binary mask M p ∈ { 0 , 1 } N p , where each element indicates whether a given patch is kept or masked out. Only when all the pixels within the patch are entirely masked, is the patch treated as a masked token. The intuition is that the boundary pixels, which usually exist in partially masked patches, are crucial for region classification. We allocate a learnable tensor representing prompt tokens as P ∈ R N p × E . Finally, the final input to the transformer is computed as T ⊗ M p + P ⊗ (1 -M p ) , where ⊗ denotes element-wise multiplication. Following the 'deep prompts' in [20], we can add such prompt tokens to deeper layers of the transformer. This is also illustrated in Figure 4.

Compared with fully finetuning the entire model [42], mask prompt tuning has several advantages. First, it is specifically designed for segmentation tasks, where parts of input images are masked. Next, compared with full model finetuning, the amount of trainable parameters in mask prompt tuning is orders of magnitude smaller, leading to much better training efficiency. Moreover, as a foundational model, CLIP may be simultaneously used for many tasks, and we may not be allowed to tune CLIP's weights. Mask prompt tuning does not require changing weights of CLIP, thus is suitable for such multi-task scenarios. Lastly, our experiments show that mask prompt tuning alone leads to significant improvement. And if applied together with full model finetuning, it can further improve the openvocabulary segmentation performance (Section 4.3.2).

## 4. Experiments

## 4.1. Experimental setup

Training Dataset We train our model on the COCO dataset [26]. We first train the modified MaskFormer using the segmentation labels from COCO-Stuff [5]. Next, we finetune CLIP on the mask-category dataset that we obtained from COCO Captions [8]. There are 118k training images labeled with 171 valid categories in the dataset, ranging from things ( e.g ., orange, car) to stuffs ( e.g ., sky, road). If not specified otherwise, we use all the 171 categories data during training.

Evaluation Dataset Our open-vocabulary model is able to perform zero-shot segmentation on arbitrary datasets without dataset-specific adaption. Thus, we test our model on challenging ADE20K [43], Pascal VOC [15] and Pascal Context [33] datasets. ADE20K is a densely pixel-wise annotated dataset for scene understanding, which spans diverse annotations of indoor and outdoor scenes. There are 2K images in its validation set. We choose two versions of categories, one with 150 frequently used categories (A-150) and one with more diverse 847 categories (A-847). Pascal VOC is a classical dataset for segmentation. We evaluate on the 1.5K validation images with 20 categories (PAS-20). Pascal Context is a set of additional annotations for PASCAL VOC 2010. It goes beyond the original PASCAL semantic segmentation task by providing annotations for the whole scene. There are 5K images in its validation set. We also choose two versions of categories, one with 59 frequently used categories (PC-59) and one with the whole 459 categories (PC-459).

Table 1. The mIoU results of open-vocabulary generalist models and supervised specialist models. Results for SPNet and ZS3Net on PAS20 are reported from [23]. Results for ZegFormer on PAS-20 are recalculated by us. SimBaseline [40], ZegFormer [11] and OpenSeg [16] are using the same COCO images, i.e ., the 2017 splits with 118K images, but with different annotations. COCO-Stuff-156/171 denotes using COCO Stuff mask annotations of 156/171 categories. Under the R101c model scale, our model significantly outperforms other open-vocabulary models. Our largest Swin-Base model can match the performance of some supervised specialist models in 2017.

| method                            | backbone                          | training dataset                  | A-847                             | PC-459                            | A-150                             | PC-59                             | PAS-20                            |
|-----------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|-----------------------------------|
| Open-vocabulary generalist models | Open-vocabulary generalist models | Open-vocabulary generalist models | Open-vocabulary generalist models | Open-vocabulary generalist models | Open-vocabulary generalist models | Open-vocabulary generalist models | Open-vocabulary generalist models |
| SPNet [37]                        | R-101                             | PASCAL-15                         | -                                 | -                                 | -                                 | 24.3                              | 18.3                              |
| ZS3Net [4]                        | R-101                             | PASCAL-15                         | -                                 | -                                 | -                                 | 19.4                              | 38.3                              |
| LSeg [23]                         | R-101                             | PASCAL-15                         | -                                 | -                                 | -                                 | -                                 | 47.4                              |
| LSeg+ [16]                        | R-101                             | COCO Panoptic                     | 2.5                               | 5.2                               | 13.0                              | 36.0                              | 59.0                              |
| SimBaseline [40]                  | R-101c                            | COCO-Stuff-156                    | -                                 | -                                 | 15.3                              | -                                 | 74.5                              |
| ZegFormer [11]                    | R-50                              | COCO-Stuff-156                    | -                                 | -                                 | 16.4                              | -                                 | 80.7                              |
| OpenSeg [16]                      | R-101                             | COCO Panoptic                     | 4.0                               | 6.5                               | 15.3                              | 36.9                              | 60.0                              |
| OVSeg (Ours)                      | R-101c                            | COCO-Stuff-156                    | 7.0                               | 10.4                              | 24.0                              | 51.7                              | 89.2                              |
| OVSeg (Ours)                      | R-101c                            | COCO-Stuff-171                    | 7.1                               | 11.0                              | 24.8                              | 53.3                              | 92.6                              |
| LSeg+ [16]                        | Eff-B7                            | COCO Panoptic                     | 3.8                               | 7.8                               | 18.0                              | 46.5                              | -                                 |
| OpenSeg [16]                      | Eff-B7                            | COCO Panoptic                     | 6.3                               | 9.0                               | 21.1                              | 42.1                              | -                                 |
| OVSeg (Ours)                      | Swin-B                            | COCO-Stuff-171                    | 9.0                               | 12.4                              | 29.6                              | 55.7                              | 94.5                              |
| Supervised specialist models      | Supervised specialist models      | Supervised specialist models      | Supervised specialist models      | Supervised specialist models      | Supervised specialist models      | Supervised specialist models      | Supervised specialist models      |
| FCN [29]                          | FCN-8s                            | Same as test                      | -                                 | -                                 | 29.4                              | 37.8                              | -                                 |
| Deeplab [6]                       | R-101                             | Same as test                      | -                                 | -                                 | -                                 | 45.7                              | 77.7                              |
| SelfTrain [45]                    | Eff-L2                            | Same as test                      | -                                 | -                                 | -                                 | -                                 | 90.0                              |
| MaskFormer [9]                    | R-101c                            | Same as test                      | 17.4                              | -                                 | 46.0                              | -                                 | -                                 |

Implementation Details As indicated before, our model consists of two part: one segmentation model based on MaskFormer [9] and one mask-adapted CLIP model [35]. The final class prediction is ensemble of MaskFormer's prediction and CLIP's prediction. The ensemble weight λ can be found in Appendix. For the segmentation model, we have two backbone choices, ResNet-101c [6] and SwinBase [28]. For the CLIP model, we have two choices: ViTB/16 and ViT-L/14 [13]. We detail our largest model setting here, while the training recipe of the R101c model can be found in Appendix. For Swin-Base segmentation model, the backbone weights are initialized from an ImageNet-21K pre-trained model. We use AdamW [30] optimizer with the poly learning rate schedule [6]. The initial learning rate and weight decay are set to 6 · 10 -5 and 10 -2 , respectively. We use a crop size of 640 × 640 , a batch size of 32 and train the model for 120K iterations. For data augmentations and other hyper-parameters, we mainly follow the setting of [9].

For adapting CLIP ViT-L/14 model, we utilize the OpenCLIP [18] implementation. After collecting 440K mask- category pairs from captions (see Section 3.2), we propose three ways to adapt CLIP: mask prompt tuning (MPT) only, full model fine-tuning (FT) only and joint MPT + FT. For MPT only, we initialize the CLIP model with official OpenAI weights [35] and the learnable tokens are randomly initialized. We also use the deep prompts as proposed in [20]. The prompt depth is set to 3 if not specified otherwise. The training optimizer is AdamW with initial learning rate 2 · 10 -2 and weight decay 0 . The cosine annealing scheduler is adopted to adjust the learning rate. The model is trained with input size of 224 × 224 , a batch size of 256 for 5 epochs. For FT only, we keep similar training procedure but with a much lower learning rate 5 · 10 -6 and larger weight decay 0 . 2 . For MPT + FT, we first initialize the CLIP with fully finetuned model and then apply the mask prompt tuning over it, which we fined more stable and effective (see Appendix) All other hyper-parameters are the same with MPT only. The text encoder of CLIP is frozen in all our experiments.

## 4.2. Main results on open vocabulary semantic segmentation

OVSeg achieves best performance among openvocabulary models. We conduct the comparison with other open-vocabulary generalist models using the common ResNet-101 (R-101) model scale in Table 1. We use R-101c [6], which replaces the first 7 × 7 convolution layer of R-101 with 3 consecutive 3 × 3 convolu- tions and which is popular in the semantic segmentation community. If not specified otherwise, our best performance is achieved using joint mask prompt tuning and finetuning (see Section 4.3.2). First of all, compared with perpixel approaches (SPNet [37], ZS3Net [4], LSeg [23] and LSeg+ [16]), proposal-based approaches (OpenSeg [16], SimBaseline [40] and ZegFormer [11]) show better performance. Our OVSeg also falls into the proposal-based category. Compared with other proposal-based approaches, our model shows significant improvements across all five benchmarks. In particular, our R101c model achieves 7.1% and 11.0% mIoU on challenging A-847 and PC-459, which even performs better than the EfficientNet-B7 based OpenSeg model. We notice open-vocabulary segmentation is a new research problem, thus different approaches may use different experimental settings, such as different COCO annotations. Our experiments show different annotations result in relatively small performance differences: we only observe a 0.8% mIoU drop on A-150 when changing COCO-Stuff-171 to COCO-Stuff-156.

Table 2. Ablation on mask-category pairs. The baseline is MaskFormer Swin-Base with original CLIP ViT-L/14. The masks come from ground-truth (GT) or generated proposals. The category nouns come from ground-truth (GT) classes or captions. We also calculate the statistics (number of pairs and unique nouns) of collected pairs.

| Case     | Source    | Source     | Statistics   | Statistics   | A-847      | A-150       | PC-59       |
|----------|-----------|------------|--------------|--------------|------------|-------------|-------------|
| Case     | Mask      | Category   | Pairs        | Unique nouns | A-847      | A-150       | PC-59       |
| Baseline | -         | -          | -            | -            | 7.3        | 21.8        | 51.4        |
| (1)      | GT        | GT         | 965K         | 171          | 5.3 (-2.0) | 23.0 (+1.2) | 57.3 (+5.9) |
| (2)      | GT        | 1 caption  | 440K         | 12K          | 7.9 (+0.6) | 24.2 (+2.4) | 53.2 (+1.8) |
| (3)      | proposals | 1 caption  | 440K         | 12K          | 8.8 (+1.5) | 28.8 (+7.0) | 55.7 (+4.3) |
| (4)      | proposals | 5 captions | 1.3M         | 27K          | 8.8 (+1.5) | 28.6 (+6.8) | 55.5 (+4.1) |

Largest OVSeg model sets up new SOTA results on zero-shot benchmarks. When we scale up the model, our method can further achieve better results. With SwinBase (Swin-B) backbone and CLIP ViT-L/14, our model can achieve 29.6% and 55.5% mIoU on A-150 and Pascal PC-59, which is +8.5% and +13.6% higher than the SOTA zero-shot results. On the challenging A-847 and PC-459, our model sets up a new zero-shot state-of-the-art 9.0% and 12.4% mIoU. We further detail the class-wise IoU of A-150 categories in Appendix.

Open-vocabulary generalist models can match supervised specialist models in 2017. We show our generalist model can achieve competitive performance without the need of any dataset specific training. On the challenging A150, our model achieves similar performance with fully supervised FCN-8s [29]. On the PAS-20, our model achieves 94.5% mIoU, which is even +4.5% than the SOTA specialist model [45]. We note OVSeg is not directly comparable with supervised models because OVSeg is not trained on evaluation datasets. OVSeg also has different backbones and segmentation model architectures. Thus, comparison with supervised models is for reference purposes only. Our generalist model still underperforms the advanced specialist models, such as supervised MaskFormer [9].

Table 3. Ablation on mask prompt tuning (MPT) and full model tuning. The baseline is MaskFormer Swin-Base with CLIP ViTL/14. We report the zero-shot mIoU on representative ADE-847, ADE-150 and PC-59 datasets. All the improvements are measured upon the baseline model.

| case                 | FT method MPT full   | A-847                                | A-150                                    | PC-59                                    |
|----------------------|----------------------|--------------------------------------|------------------------------------------|------------------------------------------|
| Baseline (a) (b) (c) | ✓ ✓ ✓ ✓              | 7.3 8.4 (+1.1) 8.8 (+1.5) 9.0 (+1.7) | 21.8 26.5 (+4.7) 28.8 (+7.0) 29.6 (+7.8) | 51.4 55.4 (+4.0) 55.7 (+4.3) 55.7 (+4.3) |

## 4.3. Ablation study

## 4.3.1 Collecting mask-category pairs

We discuss the impact of finetuning data in Table 2. The baseline model is MaskFormer Swin-Base with the original CLIP ViT-L/14. Our initial trial (case (1)) is collecting ground-truth (GT) masks with supervised GT classes. We can collecting 965K mask-category pairs with 171 unique nouns (the number of classes defined in COCO-stuff). Then we finetune the CLIP model with the collected pairs. We observe a -2.0% performance drop on the A-847 dataset. This is because the adapted CLIP is over-fitting to the 171 GT classes. Although the model achieves good results on PC-59 (whose categories are highly overlapped with COCO-Stuff), it perform badly for more diverse concepts in A-847. As detailed in Section 3.2, we propose to utilize captions [8] to collect diverse mask-category pairs. After parsing the nouns in the caption, we match the nouns with GTmasks (case (2)) or proposals (case (3)) generated by the baseline model. By replacing the GT masks with proposals, the A-150 mIoU is significantly improved (from 24.2% to 28.8%) We conjecture that many regions are not labeled as GT masks (see examples in Figure 3), and are therefore ignored. In contrast, the generated proposals (usually 100) can cover most of regions-of-interest in the image, leading to better performance. If all the 5 captions per image are used (case (4)), we observe a mild -0.2% degradation on A-

<!-- image -->

Query: saturn V , blossom

Query: golden gate , yacht

<!-- image -->

Query: Oculus , Ukulele

<!-- image -->

Figure 5. Open-vocabulary segmentation with user-defined queries. Our model accurately segments unseen categories, such as the Saturn V rocket, Oculus headset, and Golden gate bridge.

150 and PC-59 We hypothesis that 12K nouns are adequate for the CLIP to retain its open-vocabulary ability. Thus, we choose to use 1 caption for efficiency purposes as it's 5x faster in training then using 5 captions.

## 4.3.2 Mask prompt tuning

We ablate the effect of mask prompt tuning in Table 3. The baseline model is MaskFormer Swin-Base with CLIP ViT-L/14. If we only use mask prompt tuning (case (a)), our model outperforms the baseline by a large +4.7% and +4.0% mIoU improvement on ADE-150 and PC-59, respectively. Case (b) shows the result of full model fine-tuning. Although it achieves the best accuracy, the trainable parameters are orders of magnitude higher. In contrast, the proposed mask prompt tuning only modifies the input without changing CLIP's weight. Furthermore, mask prompt tuning can further improve over a fully finetuned model, as shown in case (c). Case (c) achieves 29.6% mIoU ADE150, which further improves the fully finetuned model by a considerable margin of +0.8%.

## 4.4. Discussions

## 4.4.1 Segmentation with user-defined queries.

Our method allows users to define arbitrary queries and search the query in the image, see Figure 5. Without training our models to learn specific concepts, our model can locate and segment Saturn V as the lego rocket, Oculus as the VR headset, and golden gate as the bridge in corresponding images. This demonstrates the strong potentials of open vocabulary semantic segmentation.

## 4.4.2 Ambiguity of open vocabulary evaluation

We show some 'failure' predictions from the A-150 dataset in Figure 6. For the left figure, the ground-truth category is 'building' while our model predicts 'skyscrapers'. The 'skyscrapers' is a reasonable description, but the standard A-150 evaluation protocol will treat it as a wrong prediction. A similar case happens in the right figure, the groundtruth 'rail' is recognized as 'road'. This is caused by the fact that language defined categories are ambiguous and can overlap with each other. Designing a better evaluation metric for open-vocabulary segmentation models is an important topic for our future research. Note that we use our own images, instead of ADE20K images in Figure 6. But this phenomenon widely exists on ADE20K images.

GT: building Pred: skycraper

<!-- image -->

GT: rail Pred: road

<!-- image -->

Figure 6. Ambiguity of the class definition in open vocabulary segmentation evaluation.

## 5. Conclusion

This paper studies open-vocabulary semantic segmentation where the model segments an image by arbitrary categories described by texts. We identify the performance bottleneck of current two-stage methods to be the pre-trained CLIP, since it doesn't perform well on masked images. We propose to adapt CLIP for masked images. To retain CLIP's open-vocabulary classification ability, we adapt CLIP with diverse mask-category pairs mined from imagecaption dataset. We further propose mask prompt tuning, a method can adapt CLIP without changing its original weights. The proposed model is general and can do zero-shot segmentation on arbitrary datasets without dataset-specific adaption. For the first time, we showopenvocabulary generalist models can match the performance of supervised specialist models.

## Acknowledgments

We would like to thank Mengde Xu for setting up the baseline, Chenfeng Xu for helpful discussions.

Feng Liang and Diana Marculescu were partly supported in part by NSF CCF Grant No. 2107085 and NSF CSR Grant No. 1815780, as part of their affiliation with The University of Texas at Austin.

## Ethics Statement

Weonly use the public computer vision datasets (COCO, ADE20K, Pascal) and leverage the open-sourced visionlanguage models (CLIP) for our experiments. To the best of our knowledge, we do not foresee our approach as being inherently subject to concerns of discrimination / bias / fairness, inappropriate potential applications, impact, privacy and security issues, research integrity or research practice issues. However, the public datasets and pre-trained models may be subject to bias that may be inherited by models trained with our approach.

## Reproducibility Statement

Our code is reproducible and can be implemented based on the method description in Section 3 as well as training details in Section 4.1.

## References

- [1] H Bahng, A Jahanian, S Sankaranarayanan, and P Isola. Exploring visual prompts for adapting large-scale models. arXiv preprint arXiv:2203.17274 , page 2022, 2022. 3
- [2] Irving Biederman. Recognition-by-components: a theory of human image understanding. Psychological review , 94(2):115, 1987. 1
- [3] Steven Bird, Ewan Klein, and Edward Loper. Natural language processing with Python: analyzing text with the natural language toolkit . ' O'Reilly Media, Inc.', 2009. 5
- [4] Maxime Bucher, Tuan-Hung Vu, Matthieu Cord, and Patrick P´ erez. Zero-shot semantic segmentation. Advances in Neural Information Processing Systems , 32, 2019. 3, 6, 7
- [5] Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Cocostuff: Thing and stuff classes in context. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 1209-1218, 2018. 2, 4, 5
- [6] Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE transactions on pattern analysis and machine intelligence , 40(4):834-848, 2017. 1, 2, 4, 6
- [7] Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-decoder with atrous separable convolution for semantic image segmentation. In Proceedings of the European conference on computer vision (ECCV) , pages 801-818, 2018. 1
- [8] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325 , 2015. 2, 5, 7
- [9] Bowen Cheng, Alex Schwing, and Alexander Kirillov. Perpixel classification is not all you need for semantic segmentation. Advances in Neural Information Processing Systems , 34:17864-17875, 2021. 1, 3, 4, 6, 7, 11
- [10] Katherine Crowson, Stella Biderman, Daniel Kornis, Dashiell Stander, Eric Hallahan, Louis Castricato, and Edward Raff. Vqgan-clip: Open domain image generation and editing with natural language guidance. arXiv preprint arXiv:2204.08583 , 2022. 3
- [11] Jian Ding, Nan Xue, Gui-Song Xia, and Dengxin Dai. Decoupling zero-shot semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 11583-11592, 2022. 1, 3, 4, 6, 7, 11, 12
- [12] Zheng Ding, Jieke Wang, and Zhuowen Tu. Openvocabulary panoptic segmentation with maskclip. arXiv preprint arXiv:2208.08984 , 2022. 3
- [13] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 , 2020. 6
- [14] Yu Du, Fangyun Wei, Zihe Zhang, Miaojing Shi, Yue Gao, and Guoqi Li. Learning to prompt for open-vocabulary object detection with vision-language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 14084-14093, 2022. 3
- [15] Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International journal of computer vision , 88(2):303-338, 2010. 2, 5
- [16] Golnaz Ghiasi, Xiuye Gu, Yin Cui, and Tsung-Yi Lin. Open-vocabulary image segmentation. arXiv preprint arXiv:2112.12143 , 2021. 1, 2, 3, 4, 6, 7
- [17] Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, and Yin Cui. Open-vocabulary object detection via vision and language knowledge distillation. arXiv preprint arXiv:2104.13921 , 2021. 3, 11
- [18] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, July 2021. If you use this software, please cite it as below. 6
- [19] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning , pages 4904-4916. PMLR, 2021. 2
- [20] Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim. Vi-

sual prompt tuning. arXiv preprint arXiv:2203.12119 , 2022. 2, 3, 5, 6, 12

- [21] Kwanyoung Kim, Yujin Oh, and Jong Chul Ye. Zegot: Zeroshot segmentation through optimal transport of text prompts. arXiv preprint arXiv:2301.12171 , 2023. 3
- [22] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691 , 2021. 3
- [23] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and Ren´ e Ranftl. Language-driven semantic segmentation. arXiv preprint arXiv:2201.03546 , 2022. 1, 3, 6, 7
- [24] Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190 , 2021. 3
- [25] Yangguang Li, Feng Liang, Lichen Zhao, Yufeng Cui, Wanli Ouyang, Jing Shao, Fengwei Yu, and Junjie Yan. Supervision exists everywhere: A data efficient contrastive language-image pre-training paradigm. arXiv preprint arXiv:2110.05208 , 2021. 2
- [26] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision , pages 740-755. Springer, 2014. 5
- [27] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. arXiv preprint arXiv:2107.13586 , 2021. 3
- [28] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 10012-10022, 2021. 6
- [29] Jonathan Long, Evan Shelhamer, and Trevor Darrell. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 3431-3440, 2015. 1, 2, 4, 6, 7
- [30] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 , 2017. 6
- [31] Huaishao Luo, Junwei Bao, Youzheng Wu, Xiaodong He, and Tianrui Li. Segclip: Patch aggregation with learnable centers for open-vocabulary semantic segmentation. arXiv preprint arXiv:2211.14813 , 2022. 3
- [32] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781 , 2013. 3
- [33] Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille. The role of context for object detection and semantic segmentation in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 891-898, 2014. 2, 5
- [34] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Text-driven manipulation of

stylegan imagery. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 2085-2094, 2021. 3

- [35] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning , pages 8748-8763. PMLR, 2021. 1, 2, 3, 4, 6
- [36] Bichen Wu, Ruizhe Cheng, Peizhao Zhang, Peter Vajda, and Joseph E Gonzalez. Data efficient language-supervised zeroshot recognition with optimal transport distillation. arXiv preprint arXiv:2112.09445 , 2021. 2
- [37] Yongqin Xian, Subhabrata Choudhury, Yang He, Bernt Schiele, and Zeynep Akata. Semantic projection network for zero-and few-label semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 8256-8265, 2019. 3, 6, 7
- [38] Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, and Xiaolong Wang. Groupvit: Semantic segmentation emerges from text supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 18134-18144, 2022. 3
- [39] Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, and Xiang Bai. Side adapter network for open-vocabulary semantic segmentation. arXiv preprint arXiv:2302.12242 , 2023. 3
- [40] Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin, Yue Cao, Han Hu, and Xiang Bai. A simple baseline for zeroshot semantic segmentation with pre-trained vision-language model. arXiv preprint arXiv:2112.14757 , 2021. 1, 3, 4, 6, 7, 12
- [41] Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. Pyramid scene parsing network. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 2881-2890, 2017. 1
- [42] Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, et al. Regionclip: Regionbased language-image pretraining. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 16793-16803, 2022. 3, 4, 5
- [43] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision , 127(3):302-321, 2019. 2, 5
- [44] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal of Computer Vision , 130(9):2337-2348, 2022. 3
- [45] Barret Zoph, Golnaz Ghiasi, Tsung-Yi Lin, Yin Cui, Hanxiao Liu, Ekin Dogus Cubuk, and Quoc Le. Rethinking pretraining and self-training. Advances in neural information processing systems , 33:3833-3845, 2020. 2, 6, 7

Crop w/ mask

<!-- image -->

Crop w/o mask

<!-- image -->

Figure 7. Crop without mask will introduce background pixels, making the prediction more difficult.

## A. Appendix

## A.1. Crop with or without mask

In the paper, we use the default crop with mask (see left of Figure 7). We also try the direct crop without mask (see right of Figure 7). Following the bottleneck analysis in the introduction, we feed the unmasked crops a pretrained CLIP for classification. This experiment gives a 13.8% mIoU, which is -6.3% worse than using the masked crops. We hypothesize that the crop with mask introduces many background pixels, making the prediction more difficult. For the example in the right of Figure 7, the 'orange' will also be an appropriate category for the unmasked crop.

We note that ZegFormer [11] has also done an ablation study about different strategies to obtain the final crop. We reach a similar conclusion.

## A.2. Text templates

We use the text templates from ViLD [17]. For each category, we used multiple templates to generate the text embeddings then ensemble these embeddings by a simple average. Text templates are shown as below:

```
'a photo of a {}.', 'This is a photo of a {}', 'There is a {} in the scene', 'There is the {} in the scene', 'a photo of a {} in the scene', 'a photo of a small {}.', 'a photo of a medium {}.', 'a photo of a large {}.', 'This is a photo of a small {}.', 'This is a photo of a medium {}.', 'This is a photo of a large {}.', 'There is a small {} in the scene.', 'There is a medium {} in the scene.', 'There is a large {} in the scene.',
```

## A.3. Class prediction ensemble weight

We set λ = 0 . 7 for A-150 and A-847, λ = 0 . 6 for PAS20, PC-59 and PC-459. We further detail the effects of ensemble on A-150 in Table. 4. MaskFormer only or CLIP

Table 4. The effects of class prediction ensemble. The baseline and our OVSeg model are Swin-Base + Vit-L. We report the mIoU on A-150.

|              |   MaskFormer only |   CLIP only |   Ensemble |
|--------------|-------------------|-------------|------------|
| baseline     |              19.6 |        14.3 |       21.8 |
| OVSeg (Ours) |              19.6 |        25.1 |       29.6 |

Table 5. Ablation on combining mask prompt tuning (MPT) and fine-tuning (FT). FT -&gt; MPT indicates first FT and then MPT, and vice versa. FT + MPT sim. means optimizing prompts and CLIP simultaneously.

| combination          | A-847      | A-150       |
|----------------------|------------|-------------|
| FT - > MPT (default) | 9.0        | 29.6        |
| MPT - > FT           | 8.5 (-0.5) | 28.1 (-1.5) |
| FT + MPT sim.        | 8.8 (-0.2) | 29.0 (-0.6) |

only denotes the use of the class prediction of MaskFormer or CLIP only. Compared with the baseline, we adapt the CLIP to masked images, leading to a much better CLIP only performance. We also notice ensemble is essential for good performance.

## A.4. Training hyperparams of R101c model

Our small model is MaskFormer R101c with CLIP ViTB/16. For MaskFormer training, the backbone weights are initialized from an ImageNet-1K pre-trained model. We use AdamWoptimizer with the poly learning rate schedule. The initial learning rate and weight decay are set to 2 · 10 -4 and 10 -4 , respectively. We also use a learning rate multiplier 0 . 1 on the backbone. We use a crop size of 512 × 512 , a batch size of 32 and train the model for 120K iterations. For data augmentations and other hyper-parameters, we follow the setting of [9]. For adapting CLIP ViT-B/16 model, we basically follow the hyperparameters of finetuning ViTL/16 except we use a larger batch size 1024.

## A.5. More ablation studies on mask prompt tuning

Weexplore two other ways to combine mask prompt tuning (MPT) and fine-tuning (FT) as in Table 5. Our default setting (FT -&gt; MPT) is first doing FT and then applying MPT to the already fine-tuned model. We don't change the weights of fine-tuned CLIP. The other option is first doing MPT and then doing FT with fixed mask prompts (MPT -&gt; FT). This combination produces poor results (-1.5% drop on A-150). We conjecture mask prompts learned with original CLIP provide a bad prior when we fune-tune the entire CLIP model. We also explore learning mask prompts and fine-tune CLIP weight simultaneously (FT + MPT sim.).

Table 6. Ablation on prompt depth. We test with and without fully fine-tuned (FT) model.

| prompt depth   | A-150   | A-150   |
|----------------|---------|---------|
|                | w/o FT  | w/ FT   |
| 1              | 25.7    | 29.3    |
| 3 (default)    | 26.5    | 29.6    |
| 6              | 26.8    | 29.4    |
| 12             | 26.8    | 29.3    |

Table 7. Comparison between different prompt tuning methods.

| Method        |   baseline |   MPT (ours) |   VPT |
|---------------|------------|--------------|-------|
| mIoU on A-150 |       21.8 |         26.5 |  25.5 |

This doesn't bring favorable results either.

We further ablate the effects of prompt depth in Table 6. The depth can be selected from { 1 , 3 , 6 , 12 } . We use two different scenarios: without fine-tuning (w/o FT) for mask prompt tuning only, with fine-tuning (w/ FT) for applying mask prompt tuning over a already fine-tuned model. For w/o FT case, one layer prompt can bring significant improvement, e.g ., from baseline's 21.8% to 25.7%. Deeper prompts result in better performance, because more parameters are introduced with more prompts. Interestingly, deeper prompts (going from 3 to 12) don't bring further improvement for w/ FT case. We choose prompt depth as 3 for default setting.

## A.6. Compare masked prompt tuning (MPT) to Deep Visual Prompt Tuning (VPT) [20]

We compared our MPT to VPT [20]. With the SwinBase + ViT-L/14 baseline, we added 50 learnable tokens to the image input tokens. VPT used 'deep prompts' with depth 6, resulting in 25.5% mIoU on A-150, which is 1.0% worse than MPT (case (a) in Table 3). This could be due to the use of masked prompts in MPT, which prevent zero masked tokens and mitigate domain distribution shifts in the CLIP model. Additionally, MPT requires no additional computation, while VPT requires 40% more computation to process the extra tokens. We plan to include this ablation study in our final draft.

## A.7. Combine training pairs from COCO-stuff and COCO-Caption pseudo segments.

We combined GT COCO-stuff annotations (case (1) in Tab.2) with caption pseudo-labeled annotations (case (3) in Tab.2), resulting in 1.4M pairs with 12K nouns in Table 8. However it underperformed compared to using only pseudo-labeled annotations (26.7% mIoU vs. 28.8% mIoU

Table 8. The source of mask-category training pairs.

| Training pairs   |   Stuff |   Cap. |   Stuff + Cap. |
|------------------|---------|--------|----------------|
| mIoU on A-150    |    23.0 |   28.8 |           26.7 |

on A-150). We believe the class distribution was dominated by the GT COCO-stuff annotations and resulted in overfitting. Future work could explore a more balanced data selection ( e.g. 10% GT + 90% pseudo-labeled annotations) to potentially improve performance.

## A.8. Class-wise IoU over seen and unseen categories.

We detail the class IoU on all 150 categories in ADE20K-150 (model trained on COCO) in Figure 8, and we annotated seen vs. unseen classes and their IoUs. Seen categories mean there are similar categories in COCO-stuff training set. Unseen categories denote the novel categories in ADE20K. The average IoU of seen and unseen categories are 37.6% and 21.9%, respectively, showing that our model performs better on seen categories. This is also observed in other open vocabulary segmentation work, such as [11].

## A.9. Inference speed discussions

We followed the two-stage framework of SimBaseline [40] with a focus on accuracy improvement. Our study also evaluated the inference time of MaskFormer and CLIP region classification. For our OVSeg model (Swin-Base + ViT-L), the inference time of MaskFormer and CLIP is roughly 0.2s and 0.6s, respectively, per image on an NVIDIA A5000 GPU. We acknowledge that processing hundreds of regions with CLIP is time-intensive and understand that improving the efficiency of two-stage frameworks is a crucial area of research. It is out of the scope of this work and we plan to address this challenge in future work.

Figure 8. Class IoU on all 150 categories in ADE20K (model trained on COCO). It is expected the model performs better on seen categories in training set.

<!-- image -->