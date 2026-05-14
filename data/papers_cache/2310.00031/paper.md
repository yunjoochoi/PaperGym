## Text-image Alignment for Diffusion-based Perception

Neehar Kondapaneni 1 * Markus Marks 1 ∗ Manuel Knott 1 , 2 ∗ Rogerio Guimaraes 1 Pietro Perona 1

1 California Institute of Technology 2 ETH Zurich, Swiss Data Science Center, Empa

## Abstract

Diffusion models are generative models with impressive text-to-image synthesis capabilities and have spurred a new wave of creative methods for classical machine learning tasks. However, the best way to harness the perceptual knowledge of these generative models for visual tasks is still an open question. Specifically, it is unclear how to use the prompting interface when applying diffusion backbones to vision tasks. We find that automatically generated captions can improve text-image alignment and significantly enhance a model's cross-attention maps, leading to better perceptual performance. Our approach improves upon the current state-of-the-art (SOTA) in diffusionbased semantic segmentation on ADE20K and the current overall SOTA for depth estimation on NYUv2. Furthermore, our method generalizes to the cross-domain setting. We use model personalization and caption modifications to align our model to the target domain and find improvements over unaligned baselines. Our crossdomain object detection model, trained on Pascal VOC, achieves SOTA results on Watercolor2K. Our cross-domain segmentation method, trained on Cityscapes, achieves SOTA results on Dark Zurich-val and Nighttime Driving. Project page: vision.caltech.edu/TADP/ Code page: github.com/damaggu/TADP

## 1. Introduction

Diffusion models have set the state-of-the-art (SOTA) for image generation [32, 35, 38, 52]. Recently, a few works have shown diffusion pre-trained backbones have a strong prior for scene understanding that allows them to perform well in advanced discriminative vision tasks, such as semantic segmentation [17, 53], monocular depth estimation [53], and keypoint estimation [28, 43]. We refer to these works as diffusion-based perception methods. Unlike contrastive vision language models (e.g., CLIP) [22, 26, 31], generative models have a causal relationship with text, in which text guides image generation. In latent diffusion models, text prompts control the denoising U-Net [36], moving the image latent in a semantically meaningful direction [5].

* Equal contribution.

Figure 1. Text-Aligned Diffusion Perception (TADP). In TADP, image captions align the text prompts and images passed to diffusion-based vision models. In cross-domain tasks, target domain information is incorporated into the prompt to boost performance.

<!-- image -->

We explore this relationship and find that text-image alignment significantly improves the performance of diffusion-based perception. We then investigate text-target domain alignment in cross-domain vision tasks, finding that aligning to the target domain while training on the source domain can improve a model's target domain performance (Fig. 1).

We first study prompting for diffusion-based perceptual models and find that increasing text-image alignment improves semantic segmentation and depth estimation perfor- mance. We find that unaligned text prompts can introduce semantic shifts to the feature maps of the diffusion model [5] and that these shifts can make it more difficult for the task-specific head to solve the target task. Specifically, we ask whether unaligned text prompts, such as averaging class-specific sentence embeddings ([31, 53]), hinder performance by interfering with feature maps through the cross-attention mechanism. Through ablation experiments on Pascal VOC2012 segmentation [14] and ADE20K [55], we find that off-target and missing class names degrade image segmentation quality. We show automated image captioning [25] achieves sufficient text-image alignment for perception. Our approach (along with latent representation scaling, see Sec. 4.1) improves performance for semantic segmentation on Pascal and ADE20k by 4.0 mIoU and 1.7 mIoU, respectively, and depth estimation on NYUv2 [42] by 0.2 RMSE (+8% relative) setting the new SOTA.

Next, we focus on cross-domain adaptation: can appropriate image captioning help visual perception when the model is trained in one domain and tested on a different domain? Training models on the source domain with the appropriate prompting strategy leads to excellent unsupervised cross-domain performance on several benchmarks. We evaluate our cross-domain method on Pascal VOC [13, 14] to Watercolor2k (W2K) and Comic2k (C2K) [21] for object detection and Cityscapes (CS) [9] to Dark Zurich (DZ) [39] and Nighttime (ND) Driving [10] for semantic segmentation. We explore varying degrees of texttarget domain alignment and find that improved alignment results in better performance. We also demonstrate using two diffusion personalization methods, Textual Inversion [16] and DreamBooth [37], for better target domain alignment and performance. We find that diffusion pre-training is sufficient to achieve SOTA (+5.8 mIoU on CS → DZ, +4.0 mIoU on CS → ND, +0.7 mIoU on VOC → W2k) or near SOTA results on all cross-domain datasets with no texttarget domain alignment, and including our best text-target domain alignment method further improves +1.4 AP on Watercolor2k, +2.1 AP on Comic2k, and +3.3 mIoU on Nighttime Driving.

Overall, our contributions are as follows:

- We propose a new method using automated caption generation that significantly improves performance on several diffusion-based vision tasks through increased text-image alignment.
- We systematically study how prompting affects diffusion-based vision performance, elucidating the impact of class presence, grammar in the prompt, and previously used average embeddings.
- We demonstrate that diffusion-based perception effectively generalizes across domains, with text-target domain alignment improving performance, which can be further boosted by model personalization.

## 2. Related Work

## 2.1. Diffusion models for single-domain vision tasks

Diffusion models are trained to reverse a step-wise forward noising process. Once trained, they can generate highly realistic images from pure noise [32, 35, 38, 52]. To control image generation, diffusion models are trained with text prompts/captions that guide the diffusion process. These prompts are passed through a text encoder to generate text embeddings that are incorporated into the reverse diffusion process via cross-attention layers.

Recently, some works have explored using diffusion models for discriminative vision tasks. This can be done by either utilizing the diffusion model as a backbone for the task [17, 28, 43, 53] or through fine-tuning the diffusion model for a specific task and then using it to generate synthetic data for a downstream model [2, 50]. We use the diffusion model as a backbone for downstream vision tasks.

VPD[53] encodes images into latent representations and passes them through one step of the Stable Diffusion model. The cross-attention maps, multi-scale features, and output latent code are concatenated and passed to a task-specific head. Text prompts influence all these maps through the cross-attention mechanism, which guides the reverse diffusion process. The cross-attention maps are incorporated into the multi-scale feature maps and the output latent representation. The text guides the diffusion process and can accordingly shift the latent representation in semantic directions [1, 5, 16, 18]. The details of how VPD uses the prompting interface are described in Sec. 3. In short, VPD uses unaligned text prompts. In our work, we show how aligning the text to the image by using a captioner can significantly improve semantic segmentation and depth estimation performance.

## 2.2. Image captioning

CLIP [31] introduced a novel learning paradigm to align images with their captions. Shortly after, the LAION-5B dataset [41] was released with 5B image-text pairs; this dataset was used to train Stable Diffusion. We hypothesize that text-image alignment is important for diffusionpretrained vision models. However, images used in advanced vision tasks (like segmentation and depth estimation) are not naturally paired with text captions. To obtain image-aligned captions, we use BLIP-2 [25], a model that inverts the CLIP latent space to generate captions for novel images.

## 2.3. Diffusion models for cross-domain vision tasks

A few works explore the cross-domain setting with diffusion models [2, 17]. Benigmim et al. [2] use a diffusion model to generate data for a downstream unsupervised domain adaptation (UDA) architecture. In [17], the diffusion backbone is frozen, and the segmentation head is trained with a consistency loss with category and scene prompts guiding the latent code towards target cross-domains. Similar to VPD, the category prompts consist of token embeddings for all classes present in the dataset, irrespective of their presence in any specific image. The consistency loss forces the model to predict the same output mask for all the different scene prompts, helping the segmentation head become invariant to the scene type. Instead of using a consistency loss, we train the diffusion model backbone and task head on the source domain data with and without incorporating the style of the target domain in the caption. We find that better alignment with the target domain (i.e., target domain information included in the prompt) results in better cross-domain performance.

## 2.4. Cross-domain object detection

Cross-domain object detection can be divided into multiple subcategories, depending on what data / labels are at train / test time available. Unsupervised domain adaptation objection detection (UDAOD) tries to improve detection performance by training on unlabeled target domain data with approaches such as self-training [11, 44], adversarial distribution alignment [54] or generating pseudo labels for self-training [23]. Cross-domain weakly supervised object detection (CDWSOD) assumes the availability of imagelevel annotations at training time and utilizes pseudo labeling [21, 30], alignment [51] or correspondence mining [19]. Recently, [46] used CLIP [31] for Single Domain Generalization, which aims to generalize from a single domain to multiple unseen target domains. Our text-based method defines a new category of cross-domain object detection that tries to adapt from a single source to an unseen target domain by only having the broad semantic context of the target domain (e.g., foggy/night/comic/watercolor) as text input to our method. When we incorporate model personalization, our method can be considered a UDAOD method since we train a token based on unlabeled images from the target domain.

## 3. Methods

Stable Diffusion [35]. The text-to-image Stable Diffusion model is composed of four networks: an encoder E , a conditional denoising autoencoder (a U-Net in Stable Diffusion) ϵ θ , a language encoder τ θ (the CLIP text encoder in Stable Diffusion), and a decoder D . E and D are trained before ϵ θ , such that D ( E ( x )) = ˜ x ≈ x . Training ϵ θ is composed of a pre-defined forward process and a learned reverse process. The reverse process is learned using LAION-400M [40], a dataset of 400 million images ( x ∈ X ) and captions ( y ∈ Y ). In the forward process, an image x is encoded into a latent z 0 = E ( x ) , and t steps of a forward noise process are executed to generate a noised latent z t . Then, to learn the reverse process, the latent z t is passed to the denoising autoencoder ϵ θ , along with the time-step t and the image caption's representation C = τ θ ( y ) . τ θ adds information about y to ϵ θ using a cross-attention mechanism, in which the query is derived from the image, and the key and value are transformations of the caption representation. The model ϵ θ is trained to predict the noise added to the latent in step t of the forward process:

<!-- formula-not-decoded -->

where t ∈ { 0 , ..., T } . During generation, a pure noise latent z T and a user-specified prompt are passed through the denoising autoencoder ϵ θ for T steps and decoded D ( z 0 ) to generate an image guided by the text prompt.

Diffusion for Feature Extraction. Diffusion backbones have been used for downstream vision tasks in several recent works [17, 28, 43, 53]. Due to its public availability and performance in perception tasks, we use a modified version (see Sec. 4.1) of the feature extraction method in VPD. An image latent z 0 = E ( x ) and a conditioning C are passed through the last step of the denoising process ϵ θ ( z 0 , 0 , C ) . The cross-attention maps A and the multi-scale feature maps F of the U-Net are concatenated V = A ⊕ F and passed to a task-specific head H to generate a prediction ˆ p = H ( V ) . The backbone ϵ θ and head H are trained with a task-specific loss L H (ˆ p, p ) .

Average EOS Tokens. To generate C , previous methods [17, 53] rely on a method from CLIP [31] to use averaged text embeddings as representations for the classes in a dataset. A list of 80 sentence templates for each class of interest (such as 'a &lt; adjective &gt; photo of a &lt; class name &gt; ') are passed through the CLIP text encoder. We use B to denote the set of class names in a dataset. For a specific class ( b ∈ B ), the CLIP text encoder returns an 80 × N × D tensor, where N is the maximum number of tokens over all the templates, and D is 768 (the dimension of each token embedding). Shorter sentences are padded with EOS tokens to fill out the maximum number of tokens. The first EOS token from each sentence template is averaged and used as the representative embedding for the class such that C ∈ R |B|× 768 . This method is used in [17, 53], we denote it as C avg and use it as a baseline. For semantic segmentation, all of the class embeddings, irrespective of presence in the image, are passed to the cross-attention layers. Only the class embedding of the room type is passed to the crossattention layers for depth estimation.

## 3.1. Text-Aligned Diffusion Perception (TADP)

Our work proposes a novel method for prompting diffusionpretrained perception models. Specifically, we explore different prompting methods G to generate C . In the singledomain setting, we show the effectiveness of a method that uses BLIP-2 [25], an image captioning algorithm, to generate a caption as the conditioning for the model: G ( x ) = ˜ y → C . We then extend our method to the crossdomain setting by incorporating target domain information to C = C + M ( P ) s , where M is a caption modifier that takes target domain information P as input and outputs a caption modification M ( P ) s and a model modification M ( P ) ϵ θ . In Sec. 4, we analyze the text-image interface of the diffusion model by varying the captioner G and caption modifier M in a systematic manner for three different vision tasks: semantic segmentation, object detection, and monocular depth estimation. Our method and experiments are presented in Fig. 2. Following [53], we train our ADE20k segmentation and NYUv2 depth estimation models with fast and regular schedules. On ADE20k, we train using 4k steps (fast), 8k steps (fast), and 80k steps (normal). For NYUv2 depth, we train on a 1-epoch (fast) schedule and a 25-epoch (normal) schedule. For implementation details, refer to Appendix D.

Figure 2. Overview of TADP. We test several prompting strategies and evaluate their impact on downstream vision task performance. Our method concatenates the cross-attention and multi-scale feature maps before passing them to the vision-specific decoder. In the blue box, we show three single-domain captioning strategies with differing levels of text-image alignment. We propose using BLIP [25] captioning to improve image-text alignment. We extend our analysis to the cross-domain setting (yellow box), exploring whether aligning the source domain text captions to the target domain may impact model performance by appending caption modifiers to image captions generated in the source domain and find model personalization modifiers (Textual Inversion/Dreambooth) work best.

<!-- image -->

## 4. Results

## 4.1. Latent scaling

Before exploring image-text alignment, we apply latent scaling to encoded images (Appendix G of Rombach et al. [35]). This normalizes the image latents to have a standard normal distribution. The scaling factor is fixed at 0 . 18215 . Wefind that latent scaling improves performance using C avg for segmentation and depth estimation (Fig. 3). Specifically, latent scaling improves ∼ 0.8% mIoU on Pascal, ∼ 0.3% mIoU on ADE20K, and a relative ∼ 5.5% RMSE on NYUv2 Depth (Fig. 3).

Table 1. Prompting for Pascal VOC2012 Segmentation. We report the single-scale validation mIoU for Pascal experiments. (R): Reproduction of VPD, Avg: EOS token averaging, LS: Latent Scaling, G: Grammar, OT: Off-target information. For our method, we indicate the minimum length of the BLIP caption with TADPX and nouns only with (NO).

| Method                                                            | Avg   | TA   | LS            | G     | OT      | mIoU ss                                         |
|-------------------------------------------------------------------|-------|------|---------------|-------|---------|-------------------------------------------------|
| VPD(R) [53] VPD(LS) Class Embs Class Names TADP-0 TADP-20 TADP-40 | ✓ ✓   | ✓ ✓  | ✓ ✓ ✓ ✓ ✓ ✓ ✓ | ✓ ✓ ✓ | ✓ ✓ ✓ ✓ | 82.34 83.06 82.72 84.08 86.36 86.19 87.11 86.35 |
| TADP(NO)-20                                                       |       |      |               |       |         |                                                 |
| TADP-Oracle                                                       |       |      | ✓             |       |         | 89.85                                           |

## 4.2. Single-domain alignment

Average EOS Tokens . We scrutinize the use of average EOS tokens for C (see Sec. 3). While average EOS tokens are sensible when measuring cosine similarities in the CLIP latent space, it is unsuitable in diffusion models, where the text guides the diffusion process through cross-attention. In our qualitative analysis, we find that average EOS tokens degrade the cross-attention maps (Fig. 4). Instead, we explore using CLIP to embed each class name independently and use the tokens corresponding to the actual word (not the EOS token) and pass this as input to the cross-attention layer:

<!-- formula-not-decoded -->

Figure 3. Effects of Latent Scaling (LS) and BLIP caption minimum length. We report mIoU for Pascal, mIoU for ADE20K, and RMSE for NYUv2 depth (right). (Top) Latent scaling improves performance on Pascal ∼ 0 . 8 mIoU (higher is better), ∼ 0 . 3 mIoU, and ∼ 5 . 5% relative RMSE (lower is better). (Bottom) We see a similar effect for BLIP minimum token length, with longer captions performing better, improving ∼ 0 . 8 mIoU on Pascal, ∼ 0 . 9 mIoU on ADE20K, and ∼ 0 . 6% relative RMSE.

<!-- image -->

Second, we explore a generic prompt, a string of class names separated by spaces:

<!-- formula-not-decoded -->

These prompts are similar to the ones used for averaged EOS tokens C avg w.r.t. overall text-image alignment but instead use the token corresponding to the word representing the class name. We evaluate these variations on Pascal VOC2012 segmentation. We find that C ClassNames improves performance by 1.0 mIoU, but C ClassEmbs reduces performance by 0.3 mIoU (see Tab. 1). We perform more in-depth analyses of the effect of text-image alignment on the diffusion model's cross-attention maps and image generation properties in Appendix A.

TADP . To align the diffusion model text input to the image, we use BLIP-2 [25] to generate captions for every image in our single-domain datasets (Pascal, ADE20K, and NYUv2).

<!-- formula-not-decoded -->

BLIP-2 is trained to produce image-aligned text captions and is designed around the CLIP latent space. However, other vision-language algorithms that produce captions could also be used. We find that these text captions improve performance in all datasets and tasks (Tabs. 1, 2, 3). Performance improves on Pascal segmentation by ∼ 4% mIoU, ADE20K by ∼ 1.4% mIoU, and NYUv2 Depth by a relative RMSE improvement of 4%. We see stronger effects on the fast schedules for ADE20K with an improvement of ∼ 5 mIoU at (4k), ∼ 2.4 mIoU (8K). On NYUv2 Depth, we see a smaller gain on the fast schedule ∼ 2.4%. All numbers are reported relative to VPD with latent scaling.

Table 2. Semantic segmentation with different methods for ADE20k. Our method (green) achieves SOTA within the diffusion-pretrained models category. The results of our oracle indicate the potential of diffusion-based models for future research as it is significantly higher than the overall SOTA (highlighted in yellow). See Tab. 1 for a notation key and Tab. S1 for fast schedule results.

| Method                       | #Params                      | FLOPs   | Crop   |   mIoU ss | mIoU ms   |
|------------------------------|------------------------------|---------|--------|-----------|-----------|
| self-supervised pre-training | self-supervised pre-training |         |        |           |           |
| EVA [15]                     | 1.01B                        | -       | 896 2  |      61.2 | 61.5      |
| InternImage-L [48]           | 256M                         | 2526G   | 640 2  |      53.9 | 54.1      |
| InternImage-H [48]           | 1.31B                        | 4635G   | 896 2  |      62.5 | 62.9      |
| multi-modal pre-training     | multi-modal pre-training     |         |        |           |           |
| CLIP-ViT-B [33]              | 105M                         | 1043G   | 640 2  |      50.6 | 51.3      |
| ViT-Adapter [8]              | 571M                         | -       | 896 2  |      61.2 | 61.5      |
| BEiT-3 [49]                  | 1.01B                        | -       | 896 2  |      62.0 | 62.8      |
| ONE-PEACE [47]               | 1.52B                        | -       | 896 2  |      62.0 | 63.0      |
| diffusion-based pre-training | diffusion-based pre-training |         |        |           |           |
| VPD A32 [53]                 | 862M                         | 891G    | 512 2  |      53.7 | 54.6      |
| VPD(R)                       | 862M                         | 891G    | 512 2  |      53.1 | 54.2      |
| VPD(LS)                      | 862M                         | 891G    | 512 2  |      53.7 | 54.4      |
| TADP-40 (Ours)               | 862M                         | 2168G   | 512 2  |      54.8 | 55.9      |
| TADP-Oracle                  | 862M                         | -       | 512 2  |      72.0 | -         |

Table 3. Depth estimation in NYUv2. We find latent scaling accounts for a relative gain of ∼ 5 . 5% on the RMSE metric. Additionally, image-text alignment improves ∼ 4% relative on the RMSE metric. A minimum caption length of 40 tokens performs the best.We also explore adding a text-adapter (TA) to TADP, but find no significant gain. See Table 1 for a notation key.

| Method                 | RMSE ↓                 |   δ 1 ↑ |   δ 2 ↑ |   δ 3 ↑ |   REL ↓ |   log10 ↓ |
|------------------------|------------------------|---------|---------|---------|---------|-----------|
| default schedule       | default schedule       |         |         |         |         |           |
| SwinV2-L [27]          | 0.287                  |   0.949 |   0.994 |   0.999 |   0.083 |     0.035 |
| AiT [29]               | 0.275                  |   0.954 |   0.994 |   0.999 |   0.076 |     0.033 |
| ZoeDepth [3]           | 0.270                  |   0.955 |   0.995 |   0.999 |   0.075 |     0.032 |
| VPD [53]               | 0.254                  |   0.964 |   0.995 |   0.999 |   0.069 |     0.030 |
| VPD(R)                 | 0.248                  |   0.965 |   0.995 |   0.999 |   0.068 |     0.029 |
| VPD(LS)                | 0.235                  |   0.971 |   0.996 |   0.999 |   0.064 |     0.028 |
| TADP-40                | 0.225                  |   0.976 |   0.997 |   0.999 |   0.062 |     0.027 |
| fast schedule, 1 epoch | fast schedule, 1 epoch |         |         |         |         |           |
| VPD                    | 0.349                  |   0.909 |   0.989 |   0.998 |   0.098 |     0.043 |
| VPD(R)                 | 0.340                  |   0.910 |   0.987 |   0.997 |   0.100 |     0.042 |
| VPD(LS)                | 0.332                  |   0.926 |   0.992 |   0.998 |   0.097 |     0.041 |
| TADP-0                 | 0.328                  |   0.935 |   0.993 |   0.999 |   0.082 |     0.038 |

We perform some ablations to analyze what aspects of the captions are important. We explore the minimum token number hyperparameter for BLIP-2 to explore if longer captions can produce more useful feature maps for the downstream task. We try a minimum token number of 0, 20, and 40 tokens (denoted as C TADP-N) and find small but consistent gains with longer captions, resulting on average 0.75% relative gain for 40 tokens vs. 0 tokens (Fig. 3). Next, we ablate the Pascal C TADP-20 captions to understand what in the caption is necessary for the performance gains we observe. We use NLTK [4] to filter for the nouns in the captions. In the C TADP(NO)-20 nouns-only caption setting, we achieve 86.4% mIoU, similar to 86.2% mIoU with C TADP-20 (Tab. 1), suggesting nouns are sufficient.

Figure 4. Cross-attention maps for different types of prompting (before training). We compare the cross-attention maps for four types of prompting: oracle, BLIP, Average EOS tokens, and class names as space-separated strings. The cross-attention maps for different heads at all different scales are upsampled to 64x64 and averaged. When comparing Average Template EOS and Class Names, we see (qualitatively) averaging degrades the quality of the cross-attention maps. Furthermore, we find that class names that are not present in the image can have highly localized attention maps (e.g., 'bottle'). Further analysis of the cross-attention maps is available in Sec. A, where we explore image-to-image generation, copy-paste image modifications, and more.

<!-- image -->

Oracle . This insight about nouns leads us to ask if an oracle caption, in which all the object class names in an image are provided as a caption, can improve performance further. We define B ( x ) as the set of class names present in image x .

<!-- formula-not-decoded -->

While this is not a realistic setting, it serves as an approximate upper bound on performance for our method on the segmentation task. We find a large improvement in performance in segmentation, achieving 89% mIoU on Pascal and 72.2% mIoU on ADE20K. For depth estimation, multiclass segmentation masks are only provided for a smaller subset of the images, so we cannot generate a comparable oracle. We perform ablations on the oracle captions to evaluate the model's sensitivity to alignment. For ADE20K, on the 4k iteration schedule, we modify the oracle captions by randomly adding and removing classes such that the recall and precision are at 0.5, 0.75, and 1.0 (independently) (Tab. S2). We find that both precision and recall have an effect, but recall is significantly more important. When recall is lower (0.50), improving precision has minimal impact ( &lt; 1% mIoU). However, precision has progressively larger impacts as recall increases to 0.75 and 1.00 ( ∼ 3% mIoU and ∼ 7% mIoU). In contrast, recall has large impacts at every precision level: 0.5 - ( ∼ 6% mIoU), 0.75 - ( ∼ 9% mIoU), and 1.00 - ( ∼ 13% mIoU). BLIP-2 captioning performs similarly to a precision of 1.00 and a recall of 0.5 (Tab. 2). Additional analyses w.r.t. precision, recall, and object sizes can be found in Appendix B.

## 4.3. Cross-domain alignment

Next, we ask if text-image alignment can benefit crossdomain tasks. In cross-domain, we train a model on a source domain and test it on a different target domain. There are two aspects of alignment in the cross-domain setting: the first is also present in single-domain, which is imagetext alignment; the second is unique to the cross-domain setting, which is text-target domain alignment. The second is challenging because there is a large domain shift between the source and target domain. Our intuition is that while the model has no information on the target domain from the training images, an appropriate text prompt may carry some general information about the target domain. Our crossdomain experiments focus on the text-target domain alignment and use G TADP for image-text alignment (following our insights from the single-domain setting).

Table 4. Cross-domain semantic segmentation. Cityscapes (CD) to Dark Zurich (DZ) val and Nighttime Driving (ND). We report the mIoU. Our method sets a new SOTA for DarkZurich and Nighttime Driving.

| Method                | Dark Zurich-val mIoU   | ND mIoU   |
|-----------------------|------------------------|-----------|
| DAFormer [20]         | -                      | 54.1      |
| Refign-DAFormer [7]   | -                      | 56.8      |
| PTDiffSeg [17]        | 37.0                   | -         |
| TADP null             | 42.8                   | 57.5      |
| TADP simple           | 39.1                   | 56.9      |
| TADP TextualInversion | 41.4                   | 60.8      |
| TADP DreamBooth       | 38.9                   | 60.4      |
| TADP NearbyDomain     | 41.9                   | 56.9      |
| TADP UnrelatedDomain  | 42.3                   | 55.1      |

Training. Our experiments in this setting are designed in the following manner: we train a diffusion model on the source domain captions C TADP ( x ) . With these source domain captions, we experiment with four different caption modifications (each increasing in alignment to the target domain), a null M null ( P ) caption modification where M null ( P ) s = ∅ = M null ( P ) ϵ θ = ∅ , a simple M simple ( P ) caption modifier where M simple ( P ) s is a hand-crafted string describing the style of the target domain appended to the end and M simple ( P ) ϵ θ = ∅ , a Textual Inversion [16] M TI ( P ) caption modifier where the output M TI ( P ) s is a learned Textual Inversion token &lt; * &gt; and M TI ( P ) ϵ θ = ∅ , and a DreamBooth [37] M DB ( P ) caption modifier where M DB ( P ) s is a learned DreamBooth token &lt; SKS &gt; and M DB ( P ) ϵ θ is a DreamBoothed diffusion backbone. We also include two additional control experiments. In the first, M ud ( P ) an unrelated target domain style is appended to the end of the string. In the second, M nd ( P ) a nearby but a different target domain style is appended to the caption. M TI ( P ) and M DB ( P ) require more information than the other methods, such that P represents a subset of unlabelled images from the target domain.

Testing. When testing the trained models on the target domain images, we want to use the same captioning modification for the test images as in the training setup. However, G TADP introduces a confound since it natu- rally incorporates target domain information. For example, G TADP ( x ) might produce the caption 'a watercolor painting of a dog and a bird' for an image from the Watercolor2K dataset. Using the M simple ( P ) captioning modification on this prompt would introduce redundant information and would not match the caption format used during training. In order to remove target domain information and get a plain caption that can be modified in the same manner as in the training data, we use GPT-3.5 [6] to remove all mentions of the target domain shift. For example, after using GPT-3.5 to remove mentions of the watercolor style in the above sentence, we are left with 'an image of a bird and a dog'. With these GPT-3.5 cleaned captions , we can match the caption modifications used during training when evaluating test images. This caption-cleaning strategy lets us control how target domain information is included in the test image captions, ensuring that test captions are in the same domain as train captions.

Table 5. Cross-domain object detection. Pascal VOC to Watercolor2k and Comic2k. We report the AP and AP 50 . Our method sets a new SOTA for Watercolor2K.

|                                                 | Watercolor2k                                    | Watercolor2k                                    | Comic2k                                         | Comic2k                                         |
|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|-------------------------------------------------|
| Method                                          | AP                                              | AP 50                                           | AP                                              | AP 50                                           |
| Single Domain Generalization (SGD)              | Single Domain Generalization (SGD)              | Single Domain Generalization (SGD)              | Single Domain Generalization (SGD)              | Single Domain Generalization (SGD)              |
| CLIP the gap [46]                               | -                                               | 33.5                                            | -                                               | 43.4                                            |
| Cross domain weakly supervised object detection | Cross domain weakly supervised object detection | Cross domain weakly supervised object detection | Cross domain weakly supervised object detection | Cross domain weakly supervised object detection |
| PLGE [30]                                       | -                                               | 56.5                                            | -                                               | 41.7                                            |
| ICCM [19]                                       | -                                               | 57.4                                            | -                                               | 37.1                                            |
| H2FA R-CNN [51]                                 | -                                               | 59.9                                            | -                                               | 46.4                                            |
| Unsupervised domain adaptation object detection | Unsupervised domain adaptation object detection | Unsupervised domain adaptation object detection | Unsupervised domain adaptation object detection | Unsupervised domain adaptation object detection |
| ADDA [45]                                       | -                                               | 49.8                                            | -                                               | 23.8                                            |
| MCAR [54]                                       | -                                               | 56.0                                            | -                                               | 33.5                                            |
| UMT [11]                                        | -                                               | 58.1                                            | -                                               | -                                               |
| DASS-Detector (extra data) [44]                 | -                                               | 71.5                                            | -                                               | 64.2                                            |
| TADP null                                       | 42.1                                            | 72.1                                            | 31.1                                            | 57.4                                            |
| TADP simple                                     | 43.5                                            | 72.2                                            | 31.9                                            | 56.6                                            |
| TADP TextualInversion                           | 43.2                                            | 72.2                                            | 33.2                                            | 57.4                                            |
| TADP DreamBooth                                 | 43.2                                            | 72.2                                            | 32.9                                            | 56.9                                            |
| TADP NearbyDomain                               | 42.0                                            | 71.5                                            | 31.8                                            | 56.4                                            |
| TADP UnrelatedDomain                            | 42.2                                            | 71.9                                            | 32.0                                            | 55.9                                            |

Evaluation. We evaluate cross-domain transfer on several datasets. We train our model on Pascal VOC [13, 14] object detection and evaluate on Watercolor2K (W2K) [21] and Comic2K (C2K) [21]. We also train our model on the Cityscapes [9] dataset and evaluate on the Nighttime Driving (ND) [10] and Dark Zurich-val (DZ-val) [39] datasets. We show results in Tabs. 4, 5. In the following sections, we also report the average performance of each method on the cross-domain segmentation datasets (average mIoU) and the cross-domain object detection datasets (average AP).

Null caption modifier. The null captions have no target domain information. In this setting, the model is trained with captions with no target domain information and tested with GPT-3.5 cleaned target domain captions. We find diffusion pre-training to be extraordinarily powerful on its own, with just plain captions (no target domain information); the model already achieves SOTA on VOC → W2K with 72.1 AP 50 , SOTA on CD → DZ-val with 42.8 mIoU and SOTA on CD → ND with 60.8 mIoU. Our model performs better than the current SOTA [44] on VOC → W2K and worse on VOC → C2K (highlighted in yellow in Tab. 5). However, [44] uses a large extra training dataset from the target (comic) domain, so we highlight in bold our results in Tab. 5 to show they outperform all other methods that use only images in C2K as examples from the target domain. Furthermore, these results are with a lightweight FPN [24] head, in contrast to other competitive methods like Refign [7], which uses a heavier decoder head. These captions achieve 50.5 average mIoU and 36.6 average AP.

Simple caption modifier. Wethen add target domain information to our captions by prepending the target domain's semantic shift to the generic captions. These caption modifiers are hand-crafted. For example, 'a dog and a bird' becomes 'a X style painting of a dog and a bird' (where X is watercolor for W2K and comic for C2K) and 'a dark night photo of a dog and a bird' for DZ. These captions achieve 48.0 average mIoU and 37.7 average AP.

Textual Inversion caption modifier. Textual inversion [16] is a method that learns a target concept (an object or style) from a set of images and encodes it into a new token. We learn a novel token from target domain image samples to further increase image-text alignment (for details, see Sec. D.1). In this setting, the sentence template becomes 'a &lt; token &gt; style painting of a dog and a bird'. We find that, on average, Textual Inversion captions perform the best, achieving 51.1 average mIoU and 38.2 average AP.

DreamBooth caption modifier. DreamBooth-ing [37] aims to achieve the same goal as textual inversion. Along with learning a new token, the stable-diffusion backbone itself is fine-tuned with a set of target domain images (for details, see Sec. D.1). We swap the stable diffusion backbone with the DreamBooth-ed backbone before training. We use the same template as in textual inversion. These captions achieve 49.7 average mIoU and 38.1 average AP.

Ablations. We ablate our target domain alignment strategy by introducing unrelated and nearby target-domain style modifications. For example, this would be 'a dashcam photo of a dog and a bird' (unrelated) and 'a constructivism painting of a dog and a bird' (nearby) for the W2K and C2K datasets. 'A watercolor painting of a car on the street' (unrelated) and 'a foggy photo of a car on the street' for the ND and DZ-val datasets. We find these off-target domains reduce performance on all datasets.

## 5. Discussion

We present a method for image-text alignment that is general, fully automated, and can be applied to any diffusionbased perception model. To achieve this, we systematically explore the impact of text-image alignment on semantic segmentation, depth estimation, and object detection. We investigate whether similar principles apply in the crossdomain setting and find that alignment towards the target domain during training improves downstream cross-domain performance.

We find that EOS token averaging for prompting does not work as effectively as strings for the objects in the image. Our oracle ablation experiments show that our diffusion pre-trained segmentation model is particularly sensitive to missing classes (reduced recall) and less sensitive to off-target classes (reduced precision), and both have a negative impact. Our results show that aligning text prompts to the image is important in identifying/generating good multiscale feature maps for the downstream segmentation head. This implies that the multi-scale features and latent representations do not naturally identify semantic concepts without the guidance of the text in diffusion models. Moreover, proper latent scaling is crucial for downstream vision tasks. Lastly, we show how using a captioner, which has the benefit of being open vocabulary, high precision, and downstream task agnostic, to prompt the diffusion pre-trained segmentation model automatically improves performance significantly over providing all possible class names.

We also find that diffusion models can be used effectively for cross-domain tasks. Our model, without any captions, already surpasses several SOTA results in crossdomain tasks due to the diffusion backbone's generalizability. We find that good target domain alignment can help with cross-domain performance for some domains, and misalignment leads to worse performance. Capturing information about target domain styles in words alone can be difficult. For these cases, we show that model personalization through Textual Inversion or Dreambooth can bridge the gap without requiring labeled data. Future work could explore how to expand our framework to generalize to multiple unseen domains. Future work may also explore closed vocabulary captioners that are more task-specific to get closer to oracle-level performance.

Acknowledgements. Pietro Perona and Markus Marks were supported by the National Institutes of Health (NIH R01 MH123612A) and the Caltech Chen Institute (Neuroscience Research Grant Award). Pietro Perona, Neehar Kondapaneni, Rogerio Guimaraes, and Markus Marks were supported by the Simons Foundation (NC-GB-CULM-00002953-02). Manuel Knott was supported by an ETH Zurich Doc.Mobility Fellowship. We thank Oisin Mac Aodha, Yisong Yue, and Mathieu Salzmann for their valuable inputs that helped improve this work.

## References

- [1] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers. arXiv preprint arXiv:2211.01324 , 2022. 2
- [2] Yasser Benigmim, Subhankar Roy, Slim Essid, Vicky Kalogeiton, and St´ ephane Lathuili` ere. One-shot Unsupervised Domain Adaptation with Personalized Diffusion Models. arXiv preprint arXiv:2303.18080 , 2023. 2
- [3] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias M¨ uller. ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth. arXiv preprint arXiv:2302.12288 , 2023. 5
- [4] Steven Bird, Ewan Klein, and Edward Loper. Natural language processing with Python: analyzing text with the natural language toolkit . O'Reilly Media, Inc., 2009. 6
- [5] Manuel Brack, Felix Friedrich, Dominik Hintersdorf, Lukas Struppek, Patrick Schramowski, and Kristian Kersting. SEGA: Instructing Diffusion using Semantic Dimensions. arXiv preprint arXiv:2301.12247 , 2023. 1, 2
- [6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems , 33:1877-1901, 2020. 7
- [7] David Br¨ uggemann, Christos Sakaridis, Prune Truong, and Luc Van Gool. Refign: Align and Refine for Adaptation of Semantic Segmentation to Adverse Conditions. 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) , 2022. 7, 8
- [8] Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Y. Qiao. Vision Transformer Adapter for Dense Predictions. arXiv preprint arXiv:2205.08534 , 2022. 5
- [9] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The Cityscapes Dataset for Semantic Urban Scene Understanding. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , pages 3213-3223, 2016. 2, 7
- [10] Dengxin Dai and Luc Van Gool. Dark Model Adaptation: Semantic Image Segmentation from Daytime to Nighttime. 2018 21st International Conference on Intelligent Transportation Systems (ITSC) , pages 3819-3824, 2018. 2, 7
- [11] Jinhong Deng, Wen Li, Yuhua Chen, and Lixin Duan. Unbiased mean teacher for cross-domain object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 4091-4101, 2021. 3, 7
- [12] David Eigen, Christian Puhrsch, and Rob Fergus. Depth Map Prediction from a Single Image using a Multi-Scale Deep Network. In Advances in Neural Information Processing Systems 27 (NIPS 2014) , 2014. 14
- [13] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes

Challenge 2007 (VOC2007) Results. http://www.pascalnetwork.org/challenges/VOC/voc2007/workshop/index.html. 2, 7

- [14] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes Challenge 2012 (VOC2012), 2012. 2, 7
- [15] Yuxin Fang, Wen Wang, Binhui Xie, Quan-Sen Sun, Ledell Yu Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. EVA: Exploring the Limits of Masked Visual Representation Learning at Scale. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 19358-19369, 2022. 5
- [16] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An Image is Worth One Word: Personalizing Text-toImage Generation using Textual Inversion. arXiv preprint arXiv:2208.01618 , 2022. 2, 7, 8
- [17] Rui Gong, Martin Danelljan, Han Sun, Julio Delgado Mangas, and Luc Van Gool. Prompting Diffusion Representations for Cross-Domain Semantic Segmentation. arXiv preprint arXiv:2307.02138 , 2023. 1, 2, 3, 7
- [18] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-Prompt Image Editing with Cross Attention Control. arXiv preprint arXiv:2208.01626 , 2022. 2
- [19] Luwei Hou, Yu Zhang, Kui Fu, and Jia Li. Informative and consistent correspondence mining for cross-domain weakly supervised object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 9929-9938, 2021. 3, 7
- [20] Lukas Hoyer, Dengxin Dai, and Luc Van Gool. DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 9914-9925, 2022. 7
- [21] Naoto Inoue, Ryosuke Furuta, Toshihiko Yamasaki, and Kiyoharu Aizawa. Cross-Domain Weakly-Supervised Object Detection Through Progressive Domain Adaptation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 5001-5009, 2018. 2, 3, 7
- [22] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, and Tom Duerig. Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision. arXiv preprint arXiv:2102.05918 , 2021. 1
- [23] Junguang Jiang, Baixu Chen, Jianmin Wang, and Mingsheng Long. Decoupled adaptation for cross-domain object detection. arXiv preprint arXiv:2110.02578 , 2021. 3
- [24] Alexander Kirillov, Ross B. Girshick, Kaiming He, and Piotr Doll´ ar. Panoptic Feature Pyramid Networks. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 6392-6401, 2019. 8, 14
- [25] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. arXiv preprint arXiv:2301.12597 , 2023. 2, 4, 5

- [26] Yangguang Li, Feng Liang, Lichen Zhao, Yufeng Cui, Wanli Ouyang, Jing Shao, Fengwei Yu, and Junjie Yan. Supervision Exists Everywhere: A Data Efficient Contrastive Language-Image Pre-training Paradigm. arXiv preprint arXiv:2110.05208 , 2022. 1
- [27] Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, and Baining Guo. Swin Transformer V2: Scaling Up Capacity and Resolution. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 11999-12009, 2021. 5
- [28] Grace Luo, Lisa Dunlap, Dong Huk Park, Aleksander Holynski, and Trevor Darrell. Diffusion hyperfeatures: Searching through time and space for semantic correspondence. arXiv preprint arXiv:2305.14334 , 2023. 1, 2, 3
- [29] Jia Ning, Chen Li, Zheng Zhang, Zigang Geng, Qi Dai, Kun He, and Han Hu. All in Tokens: Unifying Output Space of Visual Tasks via Soft Token. arXiv preprint arXiv:2301.02229 , 2023. 5
- [30] Shengxiong Ouyang, Xinglu Wang, Kejie Lyu, and Yingming Li. Pseudo-label generation-evaluation framework for cross domain weakly supervised object detection. In 2021 IEEE International Conference on Image Processing (ICIP) , pages 724-728. IEEE, 2021. 3, 7
- [31] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. International Conference on Machine Learning , pages 8748-8763, 2021. 1, 2, 3
- [32] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical Text-Conditional Image Generation with CLIP Latents. arXiv preprint arXiv:2204.06125 , 2022. 1, 2
- [33] Yongming Rao, Wenliang Zhao, Guangyi Chen, Yansong Tang, Zheng Zhu, Guan Huang, Jie Zhou, and Jiwen Lu. DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 18061-18070, 2022. 5
- [34] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. IEEE Transactions on Pattern Analysis and Machine Intelligence , 39(6):1137-1149, 2017. 14
- [35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-Resolution Image Synthesis with Latent Diffusion Models. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 10674-10685, 2022. 1, 2, 3, 4
- [36] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. UNet: Convolutional Networks for Biomedical Image Segmentation. In Medical Image Computing and ComputerAssisted Intervention -MICCAI 2015 , pages 234-241. Springer International Publishing, Cham, 2015. 1
- [37] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. DreamBooth: Fine
13. Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. arXiv preprint arXiv:2208.12242 , 2022. 2, 7, 8
- [38] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, Seyedeh Sara Mahdavi, Raphael Gontijo Lopes, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. arXiv preprint arXiv:2205.11487 , 2022. 1, 2
- [39] Christos Sakaridis, Dengxin Dai, and Luc Van Gool. Guided Curriculum Model Adaptation and Uncertainty-Aware Evaluation for Semantic Nighttime Image Segmentation. 2019 IEEE/CVF International Conference on Computer Vision (ICCV) , pages 7373-7382, 2019. 2, 7
- [40] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. LAION400M: Open Dataset of CLIP-Filtered 400 Million ImageText Pairs. arXiv preprint arXiv:2111.02114 , 2021. 3
- [41] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:2210.08402 , 2022. 2
- [42] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor Segmentation and Support Inference from RGBD Images. European Conference on Computer Vision (ECCV) , 2012. 2
- [43] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. arXiv preprint arXiv:2306.03881 , 2023. 1, 2, 3
- [44] Barıs ¸ Batuhan Topal, Deniz Yuret, and Tevfik Metin Sezgin. Domain-adaptive self-supervised pre-training for face &amp; body detection in drawings. arXiv preprint arXiv:2211.10641 , 2022. 3, 7, 8
- [45] Eric Tzeng, Judy Hoffman, Kate Saenko, and Trevor Darrell. Adversarial discriminative domain adaptation. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 7167-7176, 2017. 7
- [46] Vidit Vidit, Martin Engilberge, and Mathieu Salzmann. CLIP the Gap: A Single Domain Generalization Approach for Object Detection. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 3219-3229, 2023. 3, 7
- [47] Peng Wang, Shijie Wang, Junyang Lin, Shuai Bai, Xiaohuan Zhou, Jingren Zhou, Xinggang Wang, and Chang Zhou. ONE-PEACE: Exploring One General Representation Model Toward Unlimited Modalities. arXiv preprint arXiv:2305.11172 , 2023. 5
- [48] Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang, Zhiqi Li, Xizhou Zhu, Xiao-hua Hu, Tong Lu, Lewei Lu, Hongsheng Li, Xiaogang Wang, and Y. Qiao. InternImage: Exploring Large-Scale Vision Foundation Models with

Deformable Convolutions. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 14408-14419, 2022. 5

- [49] Wen Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiangbo Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, and Furu Wei. Image as a Foreign Language: BEIT Pretraining for Vision and Vision-Language Tasks. 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 19175-19186, 2023. 5
- [50] Weijia Wu, Yuzhong Zhao, Mike Zheng Shou, Hong Zhou, and Chunhua Shen. DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models. arXiv preprint arXiv:2303.11681 , 2023. 2
- [51] Yunqiu Xu, Yifan Sun, Zongxin Yang, Jiaxu Miao, and Yi Yang. H2fa r-cnn: Holistic and hierarchical feature alignment for cross-domain weakly supervised object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 14329-14339, 2022. 3, 7
- [52] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Benton C. Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling Autoregressive Models for Content-Rich Text-to-Image Generation. arXiv preprint arXiv:2206.10789 , 2022. 1, 2
- [53] Wenliang Zhao, Yongming Rao, Zuyan Liu, Benlin Liu, Jie Zhou, and Jiwen Lu. Unleashing Text-to-Image Diffusion Models for Visual Perception. arXiv preprint arXiv:2303.02153 , 2023. 1, 2, 3, 4, 5, 14
- [54] Zhen Zhao, Yuhong Guo, Haifeng Shen, and Jieping Ye. Adaptive object detection with dual multi-label prediction. In Computer Vision-ECCV 2020: 16th European Conference, Glasgow, UK, August 23-28, 2020, Proceedings, Part XXVIII 16 , pages 54-69. Springer, 2020. 3, 7
- [55] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene Parsing through ADE20K Dataset. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , pages 5122-5130, 2017. 2

## Text-image Alignment for Diffusion-based Perception Supplementary Materials

## A. Cross-attention analysis

Qualitative image-to-image variation analysis. We present a qualitative and quantitative analysis of the effect of off-target class names added to the prompt. In Fig. S1, we use the stable diffusion image to image (img2img) variation pipeline (with the original Stable Diffusion 1.5 weights) to qualitatively analyze the effects of prompts with off-target classes. The img2img variation pipeline encodes a real image into a latent representation, adds a user-specified amount of noise to the latent representation, and de-noises it (according to a user-specified prompt) to generate a variation on the original image. The amount of noise added is dictated by a strength ratio indicating how much variation should occur. A higher ratio results in more added noise and more denoising steps, allowing a relatively higher impact of the new text prompt on the image. We find that C ClassNames (see caption for details) results in variations that incorporate the off-target classes. This effect is most clear looking across the panels left to right in which objects belonging to off-target classes (an airplane and a train) become more prominent. These qualitative results imply that this prompt modifies the latent representation to incorporate information about off-target classes, potentially making the downstream task more difficult. In contrast, using the BLIP prompt changes the image, but the semantics (position of objects, classes present) of the image variation are significantly closer to the original. These results suggest a mechanism for how off-target classes may impact our vision models. We quantitatively measure this effect using a fully trained Oracle model in the following section.

Copy-Paste Experiment. An interesting property in Fig. 4 is that the word bottle has strong cross-attention over the neck of the bird. We hypothesize that diffusion models seek to find the nearest match for each token since they are trained to generate images that correspond to the prompt. We test this hypothesis on a base image of a dog and a bird. We first visualize the cross-attention maps for a set of object labels. We find that the words bottle, cat, and horse have a strong cross-attention to the bird, dog, and dog, respectively. We paste a bottle, cat, and horse into the base image to see if the diffusion model will localize the 'correct' objects if they are present. In Fig. S2, we show that the cross-attention maps prefer to localize the 'correct' object, suggesting our hypothesis is correct.

Averaged EOS Tokens: Averaging vs. EOS? Averaged EOS Tokens create diffuse attention maps that empirically harm performance. What is the actual cause of the decrease in performance? Is it averaging, or is it the usage of many EOS tokens? We replace the averaged EOS tokens with single prompt EOS tokens and find that the attention maps are still diffuse. This indicates that the usage of EOS tokens is the primary cause of the diffuse attention maps and not the averaging. Quantitative effect of C ClassNames on Oracle model. To quantify the impact of the off-target classes on the downstream vision task, we measure the averaged pixel-wise scores (normalized via Softmax) per class when passing the C ClassNames to the Oracle segmentation model for Pascal VOC 2012 (Fig. S4). We compare this to the original oracle prompt. We find that including the off-target prompts significantly increases the probability of a pixel being misclassified as one of the semantically nearby off-target classes. For example, if the original image contains a cow, including the words dog and sheep, it significantly raises the probability of misclassifying the pixels belonging to the cow as pixels belonging to a dog or a sheep. These results indicate that the task-specific head picks up the effect of off-target classes and is incorporated into the output.

Figure S1. Qualitative image-to-image variation. An untrained stable diffusion model is passed an image to perform image-to-image variation. The number of denoising steps conducted increases from left to right (5 to 45 out of a total of 50). On the top row, we pass all the class names in Pascal VOC 2012: 'background airplane bicycle bird boat bottle bus car cat chair cow dining table dog horse motorcycle person potted plant sheep sofa train television' . In the bottom row we pass the BLIP caption 'a bird and a dog' .

<!-- image -->

Figure S2. Copy-Paste Experiment. A bottle, a cat, and a horse from different images are copied and pasted into our base image to see how the cross-attention maps change. The label on the left describes the category of the item that has been pasted into the image. The labels above each map describe the cross-attention map corresponding to the token for that label.

<!-- image -->

Figure S3. Averaging vs. EOS. In [53], for each class name, the EOS token from 80 prompts (containing the class name) was averaged together. The averaged EOS tokens for each class were concatenated together and passed to the diffusion model as text input. We explore if averaging drives the diffuse nature of the cross-attention maps. We replace the 80 prompt templates with a single prompt template: 'a photo of a { class name } ' and visualize the cross-attention maps. In the top row, we show the averaged template EOS tokens. In the bottom row, we show the single template EOS tokens.

<!-- image -->

Figure S4. Impact of off-target classes on semantic segmentation performance. The matrices show normalized scores averaged over pixels on Pascal VOC 2012 for an oracle-trained model when receiving either present class names (left) or all class names (right).

<!-- image -->

## B. Additional ADE20K Results

| Method             | 4K Iters   | 4K Iters   | 8K Iters   | 8K Iters   |
|--------------------|------------|------------|------------|------------|
|                    | mIoU ss    | mIoU ms    | mIoU ss    | mIoU ms    |
| VPD (null text)    | 41.5       | -          | 46.9       | -          |
| VPD A32 [53]       | 43.1       | 44.2       | 48.7       | 49.5       |
| VPD(R)             | 42.6       | 43.6       | 49.2       | 50.4       |
| VPD(LS)            | 45.0       | 45.8       | 50.5       | 51.1       |
| TADP-20 (Ours)     | 50.2       | 50.9       | 52.8       | 54.1       |
| TADP(TA)-20 (Ours) | 49.9       | 50.7       | 52.7       | 53.4       |

Table S1. Semantic segmentation fast schedule on ADE20K. Our method has a large advantage over prior work on the fast schedule with significantly better performance in both the single-scale and multi-scale evaluations for 4k and 8k iterations.

|      |   Recall |   Recall |   Recall |
|------|----------|----------|----------|
|      |     0.50 |     0.75 |     1.00 |
| 0.50 |    49.53 |    52.00 |    55.22 |
| 0.75 |    49.17 |    51.46 |    58.62 |
| 1.00 |    50.20 |    54.82 |    63.29 |

Table S2. ADE20K - Oracle Precision-Recall Ablations We modify the oracle captions by randomly adding or removing classes such that the precision and recall are 0.50, 0.75, or 1.00. We train models on ADE20K on a fast schedule (4K) using these captions. The 4k iteration oracle equivalent is highlighted in blue.

Figure S5. Recall analysis. ADE20k mIOU per image with respect to the recall of classes present in the caption. We embedded each word in our caption with CLIP's text encoder. We considered a cosine similarity of ≥ 0 . 9 with the embedded class name as a match. Linear regression analysis shows positive correlations between recall and mIoU ( r = 0 . 28 ).

<!-- image -->

Figure S6. Object size analysis. ADE20k IOU per object image with respect to the relative object size (pixels divided by total pixels). Linear regression analysis shows positive correlations between relative object size and the IoU-score of a class ( r = 0 . 40 ).

<!-- image -->

## C. Qualitative Examples

Figure S7. Ground truth examples of the tokenized datasets.

<!-- image -->

Figure S8. Textual inversion and Dreambooth tokens of Cityscapes to Dark Zurich.

<!-- image -->

<!-- image -->

Figure S9. Textual inversion and Dreambooth tokens of VOC to Comic.

<!-- image -->

<!-- image -->

Figure S10. Textual inversion and Dreambooth tokens of VOC to Watercolor.

<!-- image -->

<!-- image -->

Figure S11. Predictions (top) and Ground Truth (bottom) visualizations for Pascal VOC2012.

Figure S12. Predictions (top) and Ground Truth (bottom) visualizations for ADE20K.

<!-- image -->

Figure S13. Predictions (top) and Ground Truth (bottom) visualizations for NYUv2 Depth.

<!-- image -->

Figure S14. Depth Estimation Comparison: Image, Ground Truth, and Prediction visualizations for Midas, VPD, and TADP (ours) in NYUv2 Depth. Black boxes (red on original image) show where TADP is better than Midas and/or VPD.

<!-- image -->

Figure S15. Image Segmentation Comparison: Image, Ground Truth, and Prediction visualizations for InternImage, VPD, and TADP (ours) in ADE20K. Red boxes show where TADP is better than InternImage and/or VPD.

<!-- image -->

Figure S16. Image Segmentation Comparison: Image, Ground Truth, and Prediction visualizations for InternImage, VPD, and TADP (ours) in ADE20K. Red boxes show where TADP is better than InternImage and/or VPD.

<!-- image -->

Figure S17. Depth Estimation Comparison: Image, Ground Truth, and Prediction visualizations for Midas, VPD, and TADP (ours) in NYUv2 Depth. TADP is worse than Midas and/or VPD in these images in terms of the general scale

<!-- image -->

Figure S18. Image Segmentation Comparison: Image, Ground Truth, and Prediction visualizations for InternImage, VPD, and TADP (ours) in ADE20K. Red boxes show where TADP is worse than InternImage and/or VPD.

<!-- image -->

Figure S19. Cross-domain Image Segmentation Comparison: Image, Ground Truth, and Prediction visualizations for RefignDAFormer, and TADP (ours) for Cityscapes to Dark Zurich Val. Red boxes show where TADP is better than Refign-DAFormer.

<!-- image -->

Figure S20. Cross-domain Object Detection Comparison: Image, Ground Truth, and Prediction visualizations for DASS, and TADP (ours) for Pascal VOC to Watercolor2k. Red boxes show the detections of each model. Notice that TADP not only beats DASS mostly, but also finds more objects than the ones annotated in the ground truth.

<!-- image -->

## D. Implementation Details

To isolate the effects of our text-image alignment method, we ensure our model setup precisely follows prior work. Following VPD [53], we jointly train the task-specific head and the diffusion backbone. The learning rate of the backbone is set to 1/10 the learning rate of the head to preserve the benefits of pre-training better. We describe the different tasks by describing H and L H . We use an FPN [24] head with a cross-entropy loss for segmentation. We use the same convolutional head used in VPD for monocular depth estimation with a Scale-Invariant loss [12]. For object detection, we use a Faster-RCNN head with the standard Faster-RCNN loss [34] 1 . Further details of the training setup can be found in Tab. S3 and Tab. S4. In our single-domain tables, we include our reproduction of VPD, denoted with a (R). We compute our relative gains with our reproduced numbers, with the same seed for all experiments.

| Hyperparameter                                  | Value                                     | Hyperparameter                                  | Value                 |
|-------------------------------------------------|-------------------------------------------|-------------------------------------------------|-----------------------|
| Learning Rate Batch Size Optimizer Weight Decay | 0 . 00008 2 AdamW 0 . 005                 | Learning Rate Batch Size Optimizer Weight Decay | 5 e - 4 3 AdamW 0 . 1 |
| Training Steps (a) ADE20k - full schedule       | Training Steps (a) ADE20k - full schedule |                                                 | Value                 |
| (b) ADE20k - fast                               | Value                                     | (e) Hyperparameter                              | (d) NYUv2 5 e - 4 3   |
|                                                 | schedule 8k                               | Learning Rate                                   | Value 0 . 00001       |
| Hyperparameter Learning Rate                    | Value                                     | Batch Size Gradient                             | 2                     |
| Batch Size                                      | 0 . 00016 2                               | Accumulation Epochs                             | 4 15                  |
| Optimizer                                       | AdamW                                     | Optimizer                                       | AdamW                 |
|                                                 | 80000                                     |                                                 |                       |
|                                                 |                                           | Hyperparameter                                  |                       |
| Hyperparameter                                  |                                           | Learning Rate                                   |                       |
| Learning Rate Batch Size                        | 0 . 00016 2                               | Batch Size Optimizer                            | AdamW                 |
| Optimizer                                       | AdamW 0 . 005                             | Weight Decay                                    | 0 . 1                 |
| Weight Decay                                    |                                           | Layer Decay                                     | 0 . 9                 |
|                                                 |                                           |                                                 | 1                     |
| Warmup Iters                                    | 150                                       | Epochs                                          | 0                     |
| Warmup Ratio                                    | 1 e - 6                                   | Drop Path Rate                                  | . 9                   |
| Unet Learning Rate Scale                        | 0 . 01                                    |                                                 | NYUv2 - fast schedule |
| Steps                                           | 8000                                      |                                                 |                       |
| Training                                        |                                           |                                                 |                       |
| Decay Warmup Iters                              |                                           |                                                 |                       |
| Weight                                          | 0 . 005                                   | Weight Decay                                    | 0 . 01                |
|                                                 | 75                                        |                                                 |                       |
|                                                 |                                           |                                                 | (f) Pascal VOC        |
| Warmup Ratio                                    | 1 e - 6                                   |                                                 |                       |
| Unet Learning Rate Scale                        | 0 . 01                                    |                                                 |                       |
| Training Steps 4000                             |                                           |                                                 |                       |

(c) ADE20k - fast schedule 4k

Table S3. Single-Domain Hyperparameters.

1 Object detection was not explored in VPD.

| Hyperparameter           | Value     |
|--------------------------|-----------|
| Learning Rate            | 0 . 00008 |
| Batch Size               | 2         |
| Optimizer                | AdamW     |
| Weight Decay             | 0 . 005   |
| Warmup Iters             | 1500      |
| Warmup Ratio             | 1 e - 6   |
| Unet Learning Rate Scale | 0 . 01    |
| Training Steps           | 40000     |

(a) Cityscapes → Dark Zurich &amp; NightTime Driving

Hyperparameter Value

Learning Rate

Batch Size

Epochs

Optimizer

Weight Decay

0

.

00001

2

100

AdamW

0

.

01

Learning Rate Schedule Lambda

(b) Pascal VOC → Watercolor &amp; Comic

Table S4. Cross-Domain Hyperparameters.

## D.1. Model personalization

For textual inversion, we use 500 images from DZ-train and five images for W2K and C2K and train all tokens for 1000 steps. We use a constant learning rate scheduler with a learning rate of 5 e -4 and no warmup. For Dreambooth, we use the same images as in textual inversion but train the model for 500 steps (DZ) steps or 1000 steps (W2K and C2K). We use a learning rate of 2 e -6 with a constant learning rate scheduler and no warmup. We use no prior preservation loss.

Hyperparameter

Prior Preservation Cls Images

Learning Rate

Training Steps Value

200

5

e

-

6

1000

(c) Dreambooth Hyperparameters

Hyperparameter Value

Steps

Learning Rate

Batch Size

3000

5

.

0

e

-

1

Gradient Accumulation

4

(d) Textual Inversion Hyperparameters

04