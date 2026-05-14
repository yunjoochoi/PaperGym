## S2S2: Semantic Stacking for Robust Semantic Segmentation in Medical Imaging

Yimu Pan 1 , Sitao Zhang 1 , Alison D. Gernand 1 , Jeffery A. Goldstein 2 , James Z. Wang 1

1 The Pennsylvania State University, University Park

{ ymp5078, sitao.zhang, adg14 } @psu.edu, ja.goldstein@northwestern.edu, jwang@psu.edu

## Abstract

Robustness and generalizability in medical image segmentation are often hindered by scarcity and limited diversity of training data, which stands in contrast to the variability encountered during inference. While conventional strategies-such as domain-specific augmentation, specialized architectures, and tailored training procedures-can alleviate these issues, they depend on the availability and reliability of domain knowledge. When such knowledge is unavailable, misleading, or improperly applied, performance may deteriorate. In response, we introduce a novel, domainagnostic, add-on, and data-driven strategy inspired by image stacking in image denoising. Termed 'semantic stacking,' our method estimates a denoised semantic representation that complements the conventional segmentation loss during training. This method does not depend on domainspecific assumptions, making it broadly applicable across diverse image modalities, model architectures, and augmentation techniques. Through extensive experiments, we validate the superiority of our approach in improving segmentation performance under diverse conditions. Code is available at https://github.com/ymp5078/Semantic-Stacking.

## 1 Introduction

In the rapidly evolving field of computer vision, significant progress in image recognition has been driven by not only groundbreaking developments in model architectures (He et al. 2016; Dosovitskiy et al. 2021; Ronneberger, Fischer, and Brox 2015) but also deliberated training recipes (Wightman, Touvron, and J´ egou 2021; Liu et al. 2022; Woo et al. 2023) and innovative augmentation techniques (Cubuk et al. 2020, 2019; Hendrycks et al. 2020; Yun et al. 2019). These advancements largely stem from the abundance and diversity of natural image datasets (Russakovsky et al. 2015; Lin et al. 2014; Krishna et al. 2017), which enable models to learn robust, generalizable features.

In contrast, medical image analysis faces distinct challenges. Data are often scarce and originate from a limited number of sites, captured through specific imaging devices, or within certain modalities (Litjens et al. 2017). High annotation costs further exacerbates these challenges, making the pursuit of training robust models in medical image analysis a paramount yet elusive goal (Aggarwal et al.

Copyright © 2025, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

2 Northwestern University, Chicago

Figure 1: An illustration of the proposed semantic stacking approach compared to traditional image stacking for noise reduction. (a) Image stacking for noise reduction in imagery. (b) Our semantic stacking technique, aimed at reducing feature noise. Here, we illustrate semantic features through semantic segmentation maps for clarity, though our method operates on encoded features.

<!-- image -->

2021; Nguyen et al. 2023). The need for model robustness in medical imaging is critical: errors can have severe clinical consequences (Esteva et al. 2019). While augmentation techniques can mitigate data limitation, they can be inadequate, or even detrimental, if misapplied to medical contexts (Perez et al. 2018; Ozbulak, Van Messem, and De Neve 2019). The heterogeneous nature of medical images-ranging from Computerized Tomography (CT) and Magnetic Resonance Imaging (MRI) scans to standard RGB photographs-further complicates the development of universally applicable augmentation strategies. Together, these issues underscore the urgent need for approaches that enhance model robustness without succumbing to the pitfalls of domain-specified bias (Seyyed-Kalantari et al. 2021; Roberts et al. 2021).

In this work, we introduce an add-on training strategy, Semantic Stacking for Semantic Segmentation (S2S2), that can be seamlessly integrated into existing pipelines. Unlike previous approaches that focus narrowly on either in-domain performance (Chen et al. 2021) or out-of-domain robustness (Su et al. 2023), our method enhances both. Drawing inspiration from image stacking in image denoising, where multiple noisy images are stacked and pooled to estimate a denoised image, we propose semantic stacking : we first estimate a denoised semantic representation from a stack of synthetic images and then encourage the network to learn from this representation. We argue that this estimated denoised semantic representation more closely reflects the underlying ground truth, thus reducing both bias and variance. This method directs models toward a denoised semantic representation, distinguishing itself through a data-driven design that avoids domain-specific assumptions. This versatility makes our approach particularly advantageous across diverse image modalities, serving as an invaluable asset in scenarios where broad generalizability is critical and specific domain knowledge remains elusive.

As general-purpose interactive segmentation tools gain traction (Kirillov et al. 2023; Ma et al. 2024; Pan et al. 2023), the need for training methodologies compatible with mixed image modalities is becoming increasingly critical. In such contexts, the data-driven design of S2S2 offers significant benefits, as integrating knowledge from different domains into a single training strategy is challenging. We validate our proposed strategy across popular network architectures and demonstrate its effectiveness in improving both in-domain performance and single-source domain generalization across various CT, MRI, and RGB images.

Additionally, directly estimating the semantic stacking requires obtaining the semantic representation from all images in the stack. Running the network through all images in the stack at each iteration is resource- and time-intensive, or even impractical, as the stack grows. Through theoretical analysis, we derived a practical upper bound for semantic variations, transforming the semantic stacking into an operation involving only two images per iteration. This transformation makes learning from the semantic stacking feasible.

Our main contributions are summarized as follows:

- We propose a versatile add-on training strategy, semantic stacking, that enhances robustness without requiring specialized domain knowledge.
- We provide theoretical analysis enabling a practical, efficient method for learning the semantic stacking that scales to large datasets.
- We demonstrate our method's ability to improve both indomain performance and out-of-domain robustness.

## 2 Related Work

## Data Augmentation in Medical Image Analysis

Early approaches adapted data augmentation strategies from natural images to medical images (Ronneberger, Fischer, and Brox 2015; Milletari, Navab, and Ahmadi 2016). For example, nnU-net (Isensee et al. 2021) employed a predefined pipeline with operations like rotation, scaling, Gaussian noise, and blur. Inspired by AutoAug (Cubuk et al. 2019), later approaches explored automated data augmentation strategies, but these relied on traditional spatial and color transformations (Xu, Li, and Zhu 2020; Qin et al. 2020; Lyu et al. 2022; Yang et al. 2019).

While traditional methods are simple and effective, they fail to fully exploit the distinctive characteristics of medical images. Recently, generative models, such as GANs (Goodfellow et al. 2020; Denton et al. 2015; Beers et al. 2018; Yi, Walia, and Babyn 2019) and diffusion models (Ho, Jain, and Abbeel 2020; Rombach et al. 2022; Kazerouni et al. 2023), have been used for synthesizing medical images. Most generative approaches have focused on classification tasks (Pinaya et al. 2022; Khader et al. 2023; Tang et al. 2023; Ye et al. 2023; Peng et al. 2023; Deo et al. 2023), with segmentation tasks receiving less attention. Notable exceptions include brainSPADE (Fernandez et al. 2022), which trained segmentation models solely with synthetic data, and DPGAN (Chai et al. 2022), which used synthetic augmentation to address class imbalance. However, these methods exhibit limitations in performance and applicability. Our work addresses this gap by proposing a versatile add-on training strategy that enhances both in-domain performance and outof-domain robustness.

## Single-Source Domain Generalization

Domain generalization aims to train models that perform reliably on previously unseen data distributions (Wang et al. 2022; Zhou et al. 2022a). Our goal aligns with single-source domain generalization (SDG), where models are trained without access to target or additional source domain information. Recent SDG approaches use specialized augmentations, adapt model architectures, or propose unique training methods (Zhou et al. 2022a; Su et al. 2023; Xu et al. 2021; Zhou et al. 2022b; Huang et al. 2020; Hu, Liao, and Xia 2023; Guo, Liu, and Yuan 2024; Liao et al. 2024). Different from these methods that only focus of out-of-domain robustness, our approach provides a versatile add-on strategy that enhances model robustness and semantic representation without changing existing augmentations, architectures, or training paradigms, ensuring strong in-domain performance while preparing models for deployment across varied medical imaging domains.

## 3 Method

Figure 2: Illustration of the proposed S2S2 framework. A stack of images given is generated from the ground truth semantic segmentation map. Two samples from the stack are then fed into the network, where the training process is guided by the consistency between features alongside the segmentation loss.

<!-- image -->

## From Image Stacking to Semantic Stacking

In semantic segmentation, our objective is to recover the ground truth segmentation map y from an input image x . This entails classifying each pixel in the semantic feature map t to yield the segmentation map y = H ( t ) , where H denotes a classifier. Ideally, the goal is to minimize the discrepancy between the estimated semantic feature map ˆ t and the truth but unknown t . Since t itself is not directly observable, our practical objective shifts to reducing the difference between the estimated segmentation map ˆ y and the true segmentation map y . However, because the classifier H may map different inputs to the same output label, suggesting that the feature map derived from the training data guided by pixel-level supervision may inherently carry bias. To address this, we leverage the concept of image stacking, a technique traditionally utilized in image denoising, to obtain a more accurate approximation of t .

In image denoising, as depicted in Fig. 1 (a), the primary objective is to estimate the unknown ground truth image x . Image stacking employs multiple noisy images to approximate the ground truth image. Let { x 1 , · · · , x n } represent a collection of noisy images sampled from N ( x, σ x ) , where x denotes the ground truth image and σ x the noise variance. Let ˆ x = P ( x 1 , · · · , x n ) denotes the pooled result of the image stack using mean or median pooling method P , then

<!-- formula-not-decoded -->

As n grows, the precision of the estimated image ˆ x relative to the ground truth image x enhances.

Adapting this principle for semantic feature estimation, as illustrated in Fig. 1 (b), allows us to approach semantic feature mapping with a novel perspective. Specifically, for a given network F without regularization, we can acquire a semantic feature map t i = F ( x i ) ∼ N ( t, σ ) , where t represents the ground truth semantic feature map. Following the same principle as in image stacking, if we possess a collection of semantic features { t 1 , · · · , t n } corresponding to the identical semantic feature map, pooling these features as ˆ t = P ( t 1 , · · · , t n ) yields an estimated feature map with diminished variance, expressed as:

<!-- formula-not-decoded -->

Let D denote a distance metric. Utilizing ˆ t as an approximation of t allows for the optimization of D ( t i , ˆ t ) to enhance the training of F , aiming for F to generate an accurate ˆ t .

## Practical Objective for Semantic Stacking

Direct approximation of ˆ t from t necessitates constructing a stack of n feature maps, which becomes impractical for large n due to the need for multiple activation copies. To overcome this, we use Bayesian updating. Given a sequence of feature maps { t 1 , · · · , t n } , the estimated posterior distribution of ˆ t is defined as:

<!-- formula-not-decoded -->

where σ 0 and t 0 are the prior distribution's hyperparameters. Assuming D satisfies the triangle inequality and adopting the L 1 distance for simplicity, minimizing D ( t i , E [ ˆ t ]) is achieved through the following optimization:

̸

<!-- formula-not-decoded -->

̸

We observe that D ( t i , E [ ˆ t ]) is upper-bounded by a weighted sum of all D ( t i , t j ) . Therefore, minimizing D ( t i , E [ ˆ t ]) effectively requires minimizing D ( t i , t j ) between any pair of feature maps in the stack. This insight permits sampling just two images at a time from the stack and minimizing the distance between their corresponding feature maps. The resulting semantic consistency loss is formulated as:

<!-- formula-not-decoded -->

where D is a suitable distance metric that adheres to the triangle inequality, with x i , x j being two distinct samples from the stack of images corresponding to the same segmentation map. This methodology, termed S2S2, is illustrated in Fig. 2.

## Constructing Semantic Stack

Generating images that align with a specific semantic segmentation map poses a significant challenge, particularly in medical image analysis, where annotations are costly and scarce. Recent advances in generative models have provided new ways for synthesizing realistic medical images. In contrast to traditional photometric adjustments like intensity or scale (Cai, Fan, and Fang 2023) changes that only account for variations due to equipment differences, variations in human organs are can be learned and simulated using generative models. Utilizing a conditional image generation approach, we generate a set of images based on a given segmentation map. This generative strategy not only enhances diversity but also reduces reliance on dataset-specific knowledge, such as particular intensity variations or color shifts introduced in methods like SLAug (Su et al. 2023), thereby offering a more generalized solution. Specifically, we fine-tune a Stable Diffusion model (Rombach et al. 2022), employing ControlNet (Zhang, Rao, and Agrawala 2023) for segmentation map control. Although the synthesized images might not precisely replicate the ground truth distribution, we suggest that generating a substantial volume of high-quality images can improve model performance.

After generating a series of semantic feature maps { t 1 , · · · , t n } ∼ N ( t g , σ g ) from the synthesized images,

̸

where σ g reflects the variance indicative of the generated feature maps' quality, and t g represents the mean, we posit that t g ≈ t . This assumption rests on the premise that finetuning the generative model with accurate ground truth annotations aligns the mean of the generated feature maps with the ground truth mean, while the variance captures residual discrepancies. In line with previous formulations (Eq. 2), we have: ˆ t g ∼ N ( t, σ g √ n ) . Specifically, if σ g √ n ≤ σ , then ˆ t g offers a more accurate estimate of the ground truth semantic feature map, which indicates the potential of enhancing model performance through the minimization of D ( t i , ˆ t g ) . Although empirically validating this condition may be challenging due to the unknown values of σ g and σ , theoretical guarantees ensure its validity as n increases.

## 4 Dataset

To comprehensively evaluate the efficacy of our method across diverse medical image segmentation scenarios, we conducted experiments assessing both in-domain and outof-domain performance. These evaluations covered a variety of imaging modalities, including RGB, CT, and MRI. Details on data prepossessing are in the Appendix.

For RGB images, we utilized two polyp segmentation datasets: CVC-ClinicDB (Bernal et al. 2015) and KvasirSEG (Jha et al. 2020). CVC-ClinicDB comprises 612 labeled images, while Kvasir-SEG includes 1,000 labeled images. These datasets, originating from distinct sites and captured using different devices, provide variability in the data. The processing of RGB datasets adhered to the methods described in previous studies (Sanderson and Matuszewski 2022). For CT images, we evaluated using the Synapse multi-organ segmentation dataset 1 , which includes 30 abdominal CT scans with comprehensive annotations for multi-organ segmentation tasks. In the MRI category, our evaluation encompassed several datasets focused on abdominal and cardiac segmentation. The Combined Healthy Abdominal Organ Segmentation (CHAOS) (Kavur et al. 2021) dataset consists of 20 T2-SPIR MRI images focused on abdominal organ segmentation. For cardiac segmentation, we included a dataset (Zhuang et al. 2022) comprising 45 late gadolinium enhanced (LGE) MRI images and 45 balanced steady-state free precession (bSSFP) MRI images, alongside the Automatic Cardiac Diagnosis Challenge (ACDC) (Bernard et al. 2018) dataset, which features 100 cases of Cine MRI images.

## 5 Results

Only average metrics are reported in this section for clarity; the class-specific metrics are detailed in the Appendix. Since S2S2 is applicable to any method, we evaluate its performance on representative methods and include baseline methods as references. These baseline methods include MSRFNet (Srivastava et al. 2021) and PraNet (Fan et al. 2020) for the Kvasir and CVC datasets; R50-AttnUNet (Schlemper et al. 2019), ViT-CUP (Dosovitskiy et al. 2021), and R50-ViT-CUP (Dosovitskiy et al. 2021) for the Synapse and ACDC datasets; and Cutout (DeVries and Taylor 2017), RSC (Huang et al. 2020), MixStyle (Zhou et al. 2021), AdvBias (Carlucci et al. 2019), RandConv (Xu et al. 2021), and CSDG (Ouyang et al. 2022) for abdominal and cardiac datasets.

1 https://www.synapse.org/ \ #!Synapse:syn3193805/wiki/ 217789

## Implementation Details

We compared S2S2 against several established approaches in medical image analysis, as well as a state-of-the-art technique in single-source domain generalization. These established techniques serve as baseline methods for our experiments. All experimental procedures adhered to the methodologies outlined by these baselines, with exceptions made solely for components that integrate our proposed approach (detailed in the Appendix). Synthetic images were generated using Stable Diffusion 2.5 fine-tuned on training images with segmentation-map-controlled ControlNet for 100 epochs. Further details are provided in the Appendix.

<!-- formula-not-decoded -->

Contemporary models for semantic segmentation are typically comprised of an encoder for capturing high-level semantics and a decoder for pixel-level details. We hypothesize that both levels of features are useful and apply our semantic consistency loss to both components, denoted as L enc sc and L dec sc , respectively. The final loss function is formulated as

where L seg represents the segmentation loss derived from any chosen method. The variables α enc and α dec are the weights for the consistency losses. For simplicity, we define the distance function as D ( t i , t j ) = 1 -CosSim( t i , t j ) where CosSim is cosine similarity.

## In-domain Performance

As an add-on method, our foundational premise posits that the integration of S2S2 should not detrimentally affect the performance of the baseline method within the scope of indomain evaluation. To verify this, we rigorously evaluated S2S2 across a variety of acclaimed network architectures on datasets derived from RGB, CT, and MRI images. Furthermore, we aim to underscore the advantages of adopting a universally applicable method over approaches that are narrowly tailored to specific tasks. To this end, we incorporated SLAug (Su et al. 2023), a state-of-the-art method devised for enhancing single-domain generalization in CT/MRI imaging, into our in-domain benchmarks.

Notably, the baseline methods already incorporate augmentation techniques such as color space and spatial augmentation, indicating that S2S2 operates independently of the baseline method or image modality. The semantic stack provides a superior representation of the ground truth semantic feature map than the original unconstrained semantic feature map. We observe an enhanced performance with

As shown in Table 1, the integration of S2S2 significantly elevates the in-domain performance for CT/MRI datasets on widely recognized models. Similarly, Table 2 demonstrates that the deployment of S2S2 concurrently amplifies the efficacy of FCBFormer on RGB datasets.

Table 1: In-domain performance comparison on the Synapse multi-organ CT dataset and ACDC dataset. Dice score (%) is used as the evaluation metric. The best-performing method is highlighted in bold, and the second-best is underlined. The improvement achieved by S2S2 is indicated.

| Method       |   Synapse |   ACDC | Mean          |
|--------------|-----------|--------|---------------|
| R50-AttnUNet |     75.57 |  86.75 | 81.16         |
| ViT-CUP      |     67.86 |  81.45 | 74.66         |
| R50-ViT-CUP  |     71.29 |  87.57 | 79.43         |
| TransUNet    |     76.86 |  88.86 | 82.86         |
| +S2S2        |     81.19 |  90.40 | 85.80 +2 . 94 |

an increase in the number of classes, potentially attributable to the generative model's refined control over image generation or the amplified complexity of maintaining semantic consistency across broader classes.

<!-- image -->

Figure 3: Visualization of the improvement achieved by applying S2S2 to the base method in the in-domain setting. 'GT' is the ground truth. 'Base' refers to the corresponding method without S2S2.

Table 2: In-domain performance comparison on RGB datasets. Dice score (%) is used as the evaluation metric.

| Method     |   Kvasir |   CVC | Mean          |
|------------|----------|-------|---------------|
| MSRF-Net   |    92.17 | 94.20 | 93.19         |
| PraNet     |    89.80 | 89.90 | 89.90         |
| SLAug      |    84.85 | 85.39 | 85.12         |
| SLAug+S2S2 |    85.33 | 88.76 | 87.05 +1 . 93 |
| FCBFormer  |    91.90 | 93.46 | 92.68         |
| +S2S2      |    93.20 | 94.88 | 94.04 +1 . 36 |

Table 3: In-domain performance comparison on slices of 3D medical image datasets. Dice score (%) is used as the evaluation metric.

| Method            | Abdominal   | Abdominal   | Cardiac   | Cardiac   | Mean          |
|-------------------|-------------|-------------|-----------|-----------|---------------|
| Method            | CT          | MRI         | bSSFP     | LGE       | Mean          |
| Supervised (CSDG) | 89.74       | 90.85       | 88.16     | 88.15     | 89.23         |
| SLAug             | 82.66       | 90.60       | 92.27     | 87.35     | 88.22         |
| +S2S2             | 84.21       | 91.28       | 92.16     | 87.62     | 88.82 +0 . 60 |

In addition, our analysis reveals that SLAug (Su et al. 2023), despite being specifically engineered for CT/MRI imaging modalities through the exploitation of domainspecific knowledge, fails to deliver comparable benefits for RGB imaging (Table 2). However, the subsequent application of S2S2 atop SLAug results in a discernible enhancement in performance metrics, indicating that S2S2 introduces an additional layer of supervision beyond the capabilities of domain-specific augmentation techniques. More importantly, even for the CT/MRT images, which SLAug was originally tailored for, S2S2 outperforms the baseline method, as shown in Table 3. This finding suggests that methods focused on domain-specific generalization may inadvertently compromise in-domain performance while optimizing for out-of-domain applicability. In contrast, our approach avoids making assumptions about the application domain, thereby ensuring consistent improvements in indomain performance across diverse datasets and imaging modalities.

Qualitative Evaluation. The comparison in Fig. 3 revealed several distinct advantages of our approach. First, S2S2 demonstrates superior capability in identifying the presence or absence of small objects, as evident in rows 1, 4, and 7. Second, it tends to generate smoother segmentation masks, observable in rows 2, 8, and 10. Lastly, S2S2 adopts a more conservative approach in its predictions, particularly highlighted in row 9.

## Out-of-domain Performance

In our out-of-domain evaluations, we benchmarked the S2S2 method against reproducible state-of-the-art, aligning with the settings of FCBFormer (Sanderson and Matuszewski 2022) for polyp segmentation tasks on RGB images and SLAug (Su et al. 2023) for abdominal organ and cardiac segmentation tasks on CT/MRI images. These comparisons validate not only the robustness of our approach in established domains but also its superior generalization capabilities in unseen domains.

Table 4: Out-of-domain performance comparison on slices of 3D medical image datasets. Dice score (%) is used as the evaluation metric.

| Method   | Abdominal           |   Abdominal | Cardiac LGE-bSSFP   |   Cardiac LGE-bSSFP | Mean         |
|----------|---------------------|-------------|---------------------|---------------------|--------------|
| Cutout   | CT-MRI MRI-CT 80.12 |       70.50 | bSSFP-LGE 78.87     |               85.92 | 78.85        |
| RSC      | 74.09               |       66.07 | 77.51               |               85.60 | 75.82        |
| MixStyle | 77.80               |       63.95 | 75.21               |               86.34 | 75.83        |
| AdvBias  | 80.17               |       64.84 | 79.62               |               86.27 | 77.73        |
| RandConv | 80.66               |       76.56 | 83.73               |               87.24 | 82.05        |
| CSDG     | 86.31               |       80.40 | 85.01               |               86.99 | 84.68        |
| SLAug    | 88.55               |       81.70 | 86.42               |               87.17 | 85.96        |
| +S2S2    | 87.75               |       83.15 | 86.06               |               87.49 | 86.11 + . 15 |

Table 5: Out-of-domain performance comparison on Polyp segmentation (RGB medical image datasets). Dice score (%) is used as the evaluation metric.

| Method    |   Kvasir-CVC |   CVC-Kvasir | Mean          |
|-----------|--------------|--------------|---------------|
| MSRF-Net  |        62.38 |        72.96 | 67.67         |
| PraNet    |        79.12 |        79.50 | 79.31         |
| SLAug     |        75.62 |        77.09 | 76.36         |
| +S2S2     |        76.44 |        80.52 | 78.48 +2 . 12 |
| FCBFormer |        91.16 |        86.46 | 88.81         |
| +S2S2     |        92.85 |        88.72 | 90.79 +1 . 98 |

Beyond demonstrating improvements in in-domain performance, our method also exhibits notable improvements in out-of-domain generalization, as shown in Table 5. Similar to what we observed in in-domain evaluation, the domainspecific method SLAug delivers suboptimal performance on RGB images. However, integrating the proposed S2S2 method fills this gap, enhancing its effectiveness. These results underscore the applicability of S2S2 in augmenting out-of-domain generalization capabilities without necessitating prior insights into the imaging modality or base models. This adaptability renders S2S2 particularly valuable in scenarios where domain-specific knowledge is unavailable. Furthermore, when such expertise is present, domainspecific strategies like SLAug exhibit superior generalization within their intended application domains, as indicated in Table 4. While domain-specific approaches are anticipated to excel, the supplementary application of S2S2 on top of SLAug still results in a marginal improvements on both the in-domain and out-of-domain performance. This result consolidates the relevance of S2S2, even in the presence of domain-specific methodologies.

Qualitative Evaluation. From Fig. 4, we observe similar ability to identify small objects and maintain boundary smoothness in the in-domain samples. Additionally, it is noteworthy that the base method is prone to misclassification issues in RGB images under conditions of significant glare (rows 6 and 8), the presence of unexpected objects (row 7), or insufficient lighting (rows 5 and 8). These conditions introduce what can be considered semantic noise. Our method, designed to mitigate semantic noise within the feature representation, remains robust and unaffected by such artifacts.

Figure 4: Visualization of the improvement achieved by applying S2S2 to the base method in the out-of-domain setting. 'GT' is the ground truth. 'Base' refers to the corresponding method without S2S2.

<!-- image -->

Table 6: Performance of TransUNet using different proposed modules, measured in DSC (%). 'Synthetic' indicates the use of synthetic images. L enc denotes the application of consistency loss on encoder features. L dec denotes the application of consistency loss on decoder features.

| Synthetic L enc L dec   | ACDC          | Synapse       |
|-------------------------|---------------|---------------|
|                         | 88.86         | 76.86         |
| ✓                       | 89.66 + . 80  | 77.61 + . 75  |
| ✓ ✓                     | 90.64 +1 . 78 | 80.29 +3 . 43 |
| ✓ ✓ ✓                   | 90.40 +1 . 54 | 81.19 +4 . 33 |

## 6 Ablation Study

In our ablation study, we aim to analyze the contribution of each module to performance, as well as the effect of the hyperparameters for the proposed loss. Our strategy to accurately gauge the contributions of our module involves leveraging a baseline model that makes minimal assumptions and favors widespread adoption. For this purpose, we select TransUNet as the base model, adhering to its established training pipeline. The results of this investigation are detailed in Table 6. Employing solely synthetic images in the absence of semantic consistency loss yields a result comparable to the documented in prior works (Pinaya et al. 2022; Khader et al. 2023; Tang et al. 2023; Ye et al. 2023), with negligible improvements. The integration of semantic consistency loss L enc , however, marks a significant elevation in performance. Although the subsequent application of L dec , in conjunction with L enc , results in performance improvement on the Synapse dataset (with 9 classes), a marginal decline in performance is observed on the ACDC dataset (with 4 classes). This result is consistent with our earlier insight, indicating the superiority of the S2S2 method in datasets characterized by a greater number of classes. Moreover, the result suggests that the quality of generated images plays a vital role in the method's effectiveness. L enc performs on a higher level semantic feature that is less sensitive to lowlevel detail of the generated images whereas L dec operates on the pixel level that is very sensitive to the low-level detail. This dynamic is reflected in our experiments, wherein the inclusion of L dec may potentially detract from out-domain performance. Nonetheless, the application of any form of semantic consistency loss invariably transcends the performance of the baseline model, underscoring the overall efficacy of the proposed S2S2 method.

Figure 5: Ablation study results using FCBFormer with the proposed S2S2 method. Dashed lines indicate the performance of the base method.

<!-- image -->

To further investigate the impact of loss weighting on performance in both in-domain and out-of-domain contexts, we conducted an ablation study using FCBFormer on RGB images. We measured the Dice score on both in-domain and out-of-domain datasets, focusing on the effects of α enc and α dec . Each variable was analyzed in isolation by setting the alternative to zero for individual assessments. From the analysis presented in Fig. 5, it is observed that α enc exerts a relatively consistent influence on in-domain performance, with the most notable improvement in out-domain performance is observed at α enc = 0 . 4 . In contrast, the impact of α dec appears less consistent, with the greatest fluctuations occurring within the range α dec ∈ [0 . 2 , 0 . 6] for both in-domain and out-of-domain datasets. This discrepancy in the behavior of losses on top of the encoder and decoder may stem from the generative model's capacity to more effectively capture higher-level semantic details as opposed to lower-level information, thereby rendering the encoder features more stable than those of the decoder, which aligns with our previous results. Moreover, the decoder features are subjected to additional layers of network weights, potentially ampli- fying errors inherent within the network architecture. This result suggests a preference for L enc over L dec , attributed to its reduced sensitivity to variations in image quality. Despite the distinct behaviors observed, both semantic consistency losses contribute to the overall enhancement in model performance. Finally, if we apply the semantic consistency loss with only photometric augmentation such as Gaussian blur and color jitters, we get worse performance than the base method (detailed in the Appendix). This result further suggests the importance of the semantic stacking in addition to traditional augmentation.

## 7 Discussion and Conclusion

We introduce S2S2, a novel and broadly applicable add-on training strategy inspired by the image stacking technique, designed to improve both in-domain performance and outof-domain robustness. However, the practical application of S2S2 encounters certain constraints. Primarily, the method's reliance on a fine-tuned generative model for semantic stacking, while innovative, introduces computational demands that may limit its suitability for situations with abundant data, such as natural image segmentation tasks. Additionally, the performance of S2S2 is inherently tied to the generative model's effectiveness across various datasets, which could significantly influence outcomes.

In conclusion, our findings present a compelling case for S2S2 as a powerful complement to existing domain-specific augmentation methods and architectural modifications. This strategy not only enhances model robustness but also represents a meaningful step toward the development of universally applicable solutions in image segmentation.

## Acknowledgements

Research reported in this publication was supported by the National Institute of Biomedical Imaging and Bioengineering of the National Institutes of Health (NIH) under award R01EB030130. The content is solely the responsibility of the authors and does not necessarily represent the official views of the NIH. This work used cluster computers at the National Center for Supercomputing Applications through an allocation from the Advanced Cyberinfrastructure Coordination Ecosystem: Services &amp; Support (ACCESS) program, which is supported by National Science Foundation (NSF) grants 2138259, 2138286, 2138307, 2137603, and 2138296. The work also used the Extreme Science and Engineering Discovery Environment (XSEDE) under NSF grant 1548562.

Aggarwal, R.; Sounderajah, V.; Martin, G.; Ting, D. S.; Karthikesalingam, A.; King, D.; Ashrafian, H.; and Darzi, A. 2021. Diagnostic accuracy of deep learning in medical imaging: a systematic review and meta-analysis. NPJ Digital Medicine , 4(1): 65.

Beers, A.; Brown, J.; Chang, K.; Campbell, J. P.; Ostmo, S.; Chiang, M. F.; and Kalpathy-Cramer, J. 2018. Highresolution medical image synthesis using progressively grown generative adversarial networks. arXiv preprint arXiv:1805.03144 .

Bernal, J.; S´ anchez, F. J.; Fern´ andez-Esparrach, G.; Gil, D.; Rodr´ ıguez, C.; and Vilari˜ no, F. 2015. WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians. Computerized Medical Imaging and Graphics , 43: 99-111.

Bernard, O.; Lalande, A.; Zotti, C.; Cervenansky, F.; Yang, X.; Heng, P.-A.; Cetin, I.; Lekadir, K.; Camara, O.; Ballester, M. A. G.; et al. 2018. Deep learning techniques for automatic MRI cardiac multi-structures segmentation and diagnosis: is the problem solved? IEEE Transactions on Medical Imaging , 37(11): 2514-2525.

Cai, Y.; Fan, L.; and Fang, Y. 2023. SBSS: Stacking-based semantic segmentation framework for very high-resolution remote sensing image. IEEE Transactions on Geoscience and Remote Sensing , 61: 1-14.

Carlucci, F. M.; D'Innocente, A.; Bucci, S.; Caputo, B.; and Tommasi, T. 2019. Domain generalization by solving jigsaw puzzles. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2229-2238.

Chai, L.; Wang, Z.; Chen, J.; Zhang, G.; Alsaadi, F. E.; Alsaadi, F. E.; and Liu, Q. 2022. Synthetic augmentation for semantic segmentation of class imbalanced biomedical images: A data pair generative adversarial network approach. Computers in Biology and Medicine , 150: 105985.

Chen, J.; Lu, Y.; Yu, Q.; Luo, X.; Adeli, E.; Wang, Y.; Lu, L.; Yuille, A. L.; and Zhou, Y . 2021. TransUNet: Transformers make strong encoders for medical image segmentation. arXiv preprint arXiv:2102.04306 .

Cubuk, E. D.; Zoph, B.; Mane, D.; Vasudevan, V.; and Le, Q. V. 2019. AutoAugment: Learning augmentation policies from data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition .

Cubuk, E. D.; Zoph, B.; Shlens, J.; and Le, Q. V. 2020. RandAugment: Practical automated data augmentation with a reduced search space. In Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition Workshops , 702-703.

Denton, E. L.; Chintala, S.; Fergus, R.; et al. 2015. Deep generative image models using a laplacian pyramid of adversarial networks. Advances in Neural Information Processing Systems , 28.

Deo, Y.; Dou, H.; Ravikumar, N.; Frangi, A. F.; and Lassila, T. 2023. Shape-guided conditional latent diffusion models for synthesising brain vasculature. In International Conference on Medical Image Computing and Computer-Assisted Intervention , 164-173. Springer.

DeVries, T.; and Taylor, G. W. 2017. Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552 .

Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; Uszkoreit, J.; and Houlsby, N. 2021. An Image is Worth 16 × 16 Words: Transformers for Image Recognition at Scale. In International Conference on Learning Representations .

Esteva, A.; Robicquet, A.; Ramsundar, B.; Kuleshov, V.; DePristo, M.; Chou, K.; Cui, C.; Corrado, G.; Thrun, S.; and Dean, J. 2019. A guide to deep learning in healthcare. Nature Medicine , 25(1): 24-29.

Fan, D.-P.; Ji, G.-P.; Zhou, T.; Chen, G.; Fu, H.; Shen, J.; and Shao, L. 2020. PraNet: Parallel reverse attention network for polyp segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention , 263-273. Springer.

Fernandez, V.; Pinaya, W. H. L.; Borges, P.; Tudosiu, P.-D.; Graham, M. S.; Vercauteren, T.; and Cardoso, M. J. 2022. Can segmentation models be trained with fully synthetically generated data? In International Workshop on Simulation and Synthesis in Medical Imaging , 79-90. Springer.

Goodfellow, I.; Pouget-Abadie, J.; Mirza, M.; Xu, B.; Warde-Farley, D.; Ozair, S.; Courville, A.; and Bengio, Y. 2020. Generative adversarial networks. Communications of the ACM , 63(11): 139-144.

Guo, X.; Liu, J.; and Yuan, Y. 2024. Infproto-Powered Adaptive Classifier and Agnostic Feature Learning for Single Domain Generalization in Medical Images. International Journal of Computer Vision , 1-24.

He, K.; Zhang, X.; Ren, S.; and Sun, J. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition , 770-778.

Hendrycks, D.; Mu, N.; Cubuk, E. D.; Zoph, B.; Gilmer, J.; and Lakshminarayanan, B. 2020. AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty. In International Conference on Learning Representations .

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems , 33: 6840-6851.

Hu, S.; Liao, Z.; and Xia, Y. 2023. Devil is in Channels: Contrastive Single Domain Generalization for Medical Image Segmentation. arXiv preprint arXiv:2306.05254 .

Huang, Z.; Wang, H.; Xing, E. P.; and Huang, D. 2020. Selfchallenging improves cross-domain generalization. In European Conference on Computer Vision , 124-140. Springer.

Isensee, F.; Jaeger, P. F.; Kohl, S. A.; Petersen, J.; and MaierHein, K. H. 2021. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods , 18(2): 203-211.

Jha, D.; Smedsrud, P. H.; Riegler, M. A.; Halvorsen, P.; Lange, T. d.; Johansen, D.; and Johansen, H. D. 2020. Kvasir-seg: A segmented polyp dataset. In International Conference on Multimedia Modeling , 451-462. Springer.

Kavur, A. E.; Gezer, N. S.; Barıs ¸, M.; Aslan, S.; Conze, P.-H.; Groza, V.; Pham, D. D.; Chatterjee, S.; Ernst, P.; ¨ Ozkan, S.; et al. 2021. CHAOS challenge-combined (CTMR) healthy abdominal organ segmentation. Medical Image Analysis , 69: 101950.

Kazerouni, A.; Aghdam, E. K.; Heidari, M.; Azad, R.; Fayyaz, M.; Hacihaliloglu, I.; and Merhof, D. 2023. Diffusion models in medical imaging: A comprehensive survey. Medical Image Analysis , 88: 102846.

Khader, F.; M¨ uller-Franzes, G.; Tayebi Arasteh, S.; Han, T.; Haarburger, C.; Schulze-Hagen, M.; Schad, P.; Engelhardt, S.; Baeßler, B.; Foersch, S.; et al. 2023. Denoising diffusion probabilistic models for 3D medical image generation. Scientific Reports , 13(1): 7303.

Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.Y.; et al. 2023. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision , 4015-4026.

Krishna, R.; Zhu, Y.; Groth, O.; Johnson, J.; Hata, K.; Kravitz, J.; Chen, S.; Kalantidis, Y.; Li, L.-J.; Shamma, D. A.; et al. 2017. Visual Genome: Connecting language and vision using crowdsourced dense image annotations. International Journal of Computer Vision , 123: 32-73.

Liao, S.; Peng, T.; Chen, H.; Lin, T.; Zhu, W.; Shi, F.; Chen, X.; and Xiang, D. 2024. Dual-Spatial Domain Generalization for Fundus Lesion Segmentation in Unseen Manufacturer's OCT Images. IEEE Transactions on Biomedical Engineering .

Lin, T.-Y.; Maire, M.; Belongie, S.; Hays, J.; Perona, P.; Ramanan, D.; Doll´ ar, P.; and Zitnick, C. L. 2014. Microsoft COCO: Common objects in context. In European Conference on Computer Vision , 740-755. Springer.

Litjens, G.; Kooi, T.; Bejnordi, B. E.; Setio, A. A. A.; Ciompi, F.; Ghafoorian, M.; Van Der Laak, J. A.; Van Ginneken, B.; and S´ anchez, C. I. 2017. A survey on deep learning in medical image analysis. Medical Image Analysis , 42: 60-88.

Liu, Z.; Mao, H.; Wu, C.-Y.; Feichtenhofer, C.; Darrell, T.; and Xie, S. 2022. A ConvNet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 11976-11986.

Lyu, J.; Zhang, Y.; Huang, Y.; Lin, L.; Cheng, P.; and Tang, X. 2022. AADG: automatic augmentation for domain generalization on retinal image segmentation. IEEE Transactions on Medical Imaging , 41(12): 3699-3711.

Ma, J.; He, Y.; Li, F.; Han, L.; You, C.; and Wang, B. 2024. Segment anything in medical images. Nature Communications , 15(1): 654.

Milletari, F.; Navab, N.; and Ahmadi, S.-A. 2016. V-Net: Fully convolutional neural networks for volumetric medical image segmentation. In International Conference on 3D Vision , 565-571. Ieee.

Nguyen, D. M. H.; Pham, T. N.; Diep, N. T.; Phan, N. Q.; Pham, Q.; Tong, V.; Nguyen, B. T.; Le, N. H.; Ho, N.; Xie, P.; et al. 2023. On the Out of Distribution Robustness of Foundation Models in Medical Image Segmentation. arXiv preprint arXiv:2311.11096 .

Ouyang, C.; Chen, C.; Li, S.; Li, Z.; Qin, C.; Bai, W.; and Rueckert, D. 2022. Causality-inspired single-source domain generalization for medical image segmentation. IEEE Transactions on Medical Imaging , 42(4): 1095-1106.

Ozbulak, U.; Van Messem, A.; and De Neve, W. 2019. Impact of adversarial examples on deep learning models for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention , 300-308. Springer. Pan, Y.; Zhang, S.; Gernand, A. D.; Goldstein, J. A.; and Wang, J. Z. 2023. AI-SAM: Automatic and Interactive Segment Anything Model. arXiv preprint arXiv:2312.03119 .

Peng, W.; Adeli, E.; Bosschieter, T.; Park, S. H.; Zhao, Q.; and Pohl, K. M. 2023. Generating realistic brain mris via a conditional diffusion probabilistic model. In International Conference on Medical Image Computing and ComputerAssisted Intervention , 14-24. Springer.

Perez, F.; Vasconcelos, C.; Avila, S.; and Valle, E. 2018. Data augmentation for skin lesion analysis. In OR 2.0 Context-Aware Operating Theaters, Computer Assisted Robotic Endoscopy, Clinical Image-Based Procedures, and Skin Image Analysis: First International Workshop, OR 2.0 2018, 5th International Workshop, CARE 2018, 7th International Workshop, CLIP 2018, Third International Workshop, ISIC 2018, Held in Conjunction with MICCAI 2018 , 303-311. Springer.

Pinaya, W. H.; Tudosiu, P.-D.; Dafflon, J.; Da Costa, P. F.; Fernandez, V.; Nachev, P.; Ourselin, S.; and Cardoso, M. J. 2022. Brain imaging generation with latent diffusion models. In MICCAI Workshop on Deep Generative Models , 117126. Springer.

Qin, T.; Wang, Z.; He, K.; Shi, Y.; Gao, Y.; and Shen, D. 2020. Automatic data augmentation via deep reinforcement learning for effective kidney tumor segmentation. In IEEE International Conference on Acoustics, Speech and Signal Processing , 1419-1423. IEEE.

Roberts, M.; Driggs, D.; Thorpe, M.; Gilbey, J.; Yeung, M.; Ursprung, S.; Aviles-Rivero, A. I.; Etmann, C.; McCague, C.; Beer, L.; et al. 2021. Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans. Nature Machine Intelligence , 3(3): 199-217.

Rombach, R.; Blattmann, A.; Lorenz, D.; Esser, P.; and Ommer, B. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 1068410695.

Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-Net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention , 234-241. Springer.

Russakovsky, O.; Deng, J.; Su, H.; Krause, J.; Satheesh, S.; Ma, S.; Huang, Z.; Karpathy, A.; Khosla, A.; Bernstein, M.; et al. 2015. ImageNet large scale visual recognition challenge. International Journal of Computer Vision , 115: 211252.

Sanderson, E.; and Matuszewski, B. J. 2022. FCNtransformer feature fusion for polyp segmentation. In Annual Conference on Medical Image Understanding and Analysis , 892-907. Springer.

Schlemper, J.; Oktay, O.; Schaap, M.; Heinrich, M.; Kainz, B.; Glocker, B.; and Rueckert, D. 2019. Attention gated networks: Learning to leverage salient regions in medical images. Medical Image Analysis , 53: 197-207.

Seyyed-Kalantari, L.; Zhang, H.; McDermott, M. B.; Chen, I. Y.; and Ghassemi, M. 2021. Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations. Nature Medicine , 27(12): 2176-2182.

Srivastava, A.; Jha, D.; Chanda, S.; Pal, U.; Johansen, H. D.; Johansen, D.; Riegler, M. A.; Ali, S.; and Halvorsen, P. 2021. MSRF-Net: a multi-scale residual fusion network for biomedical image segmentation. IEEE Journal of Biomedical and Health Informatics , 26(5): 2252-2263.

Su, Z.; Yao, K.; Yang, X.; Huang, K.; Wang, Q.; and Sun, J. 2023. Rethinking data augmentation for single-source domain generalization in medical image segmentation. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 37(2), 2366-2374.

Tang, F.; Ding, J.; Wang, L.; Xian, M.; and Ning, C. 2023. Multi-Level Global Context Cross Consistency Model for Semi-Supervised Ultrasound Image Segmentation with Diffusion Model. arXiv preprint arXiv:2305.09447 .

Wang, J.; Lan, C.; Liu, C.; Ouyang, Y.; Qin, T.; Lu, W.; Chen, Y.; Zeng, W.; and Yu, P. 2022. Generalizing to unseen domains: A survey on domain generalization. IEEE Transactions on Knowledge and Data Engineering .

Wightman, R.; Touvron, H.; and J´ egou, H. 2021. ResNet strikes back: An improved training procedure in timm. arXiv preprint arXiv:2110.00476 .

Woo, S.; Debnath, S.; Hu, R.; Chen, X.; Liu, Z.; Kweon, I. S.; and Xie, S. 2023. ConvNeXt V2: Co-designing and scaling convnets with masked autoencoders. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 16133-16142.

Xu, J.; Li, M.; and Zhu, Z. 2020. Automatic data augmentation for 3D medical image segmentation. In Medical Image Computing and Computer-Assisted Intervention , 378-387. Springer.

Xu, Z.; Liu, D.; Yang, J.; Raffel, C.; and Niethammer, M. 2021. Robust and Generalizable Visual Representation Learning via Random Convolutions. In International Conference on Learning Representations .

Yang, D.; Roth, H.; Xu, Z.; Milletari, F.; Zhang, L.; and Xu, D. 2019. Searching learning strategy with reinforcement learning for 3D medical image segmentation. In Medical Image Computing and Computer-Assisted Intervention , 311. Springer.

Ye, J.; Ni, H.; Jin, P.; Huang, S. X.; and Xue, Y. 2023. Synthetic augmentation with large-scale unconditional pretraining. In International Conference on Medical Image Computing and Computer-Assisted Intervention , 754-764. Springer.

Yi, X.; Walia, E.; and Babyn, P. 2019. Generative adversarial network in medical imaging: A review. Medical Image Analysis , 58: 101552.

Yun, S.; Han, D.; Oh, S. J.; Chun, S.; Choe, J.; and Yoo, Y. 2019. CutMix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF International Conference on Computer Vision , 6023-6032.

Zhang, L.; Rao, A.; and Agrawala, M. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision , 3836-3847.

Zhou, K.; Liu, Z.; Qiao, Y.; Xiang, T.; and Loy, C. C. 2022a. Domain generalization: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence .

Zhou, K.; Yang, Y.; Qiao, Y.; and Xiang, T. 2021. Domain Generalization with MixStyle. In International Conference on Learning Representations .

Zhou, Z.; Qi, L.; Yang, X.; Ni, D.; and Shi, Y. 2022b. Generalizable cross-modality medical image segmentation via style augmentation and dual normalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 20856-20865.

Zhuang, X.; Xu, J.; Luo, X.; Chen, C.; Ouyang, C.; Rueckert, D.; Campello, V. M.; Lekadir, K.; Vesal, S.; RaviKumar, N.; et al. 2022. Cardiac segmentation on late gadolinium enhancement MRI: a benchmark study from multi-sequence cardiac MR segmentation challenge. Medical Image Analysis , 81: 102528.

## A Dataset Details Class Definition and Visualization Palette

To facilitate a comprehensive understanding and consistent visualization of segmentation results across different datasets, we assign a unique color to each class within our datasets, as illustrated in Fig. 6. The classes, along with their corresponding abbreviations where applicable, include: Right Ventricle (RVC), Myocardium (MYO), Left Ventricle (LVC), Aorta, Gallbladder, Kidney (Left), Kidney (Right), Liver, Pancreas, Spleen, Stomach, and Polyp.

## Pre-processing

For comprehensive and standardized evaluation, we adhere to specific pre-processing protocols across different datasets. The procedures for Synapse (8 classes) and the ACDC datasets follow the guidelines established in (Chen et al. 2021). Specifically, for the Synapse dataset, we employ a random split of 18 training cases (comprising 2,212 axial slices) and 12 validation cases. For the ACDC dataset, the division consists of 70 training cases (1,930 axial slices), 10 validation cases, and 20 test cases. Image values are constrained within the range [-125, 275], and each 3D image is normalized to the range [0, 1]. We apply both spatial and color space augmentations to these datasets.

For Synapse (4 classes), CHAOS T2-SPIR, LGE, and bSSFP datasets, our pre-processing aligns with the methods described in (Ouyang et al. 2022). In the Synapse (4

Figure 6: Unique color assignments for classes in medical image segmentation datasets.

<!-- image -->

classes) dataset, we implement a windowing technique with Housefield values set between [-125, 275]. For the CHAOS T2-SPIR, LGE, and bSSFP datasets, the top 0.5% of the histogram values are clipped, each 3D image is normalized to have zero mean and unit variance, and similar to other datasets, both spatial and color space augmentations are utilized.

The Kvasir-SEG and CVC-ClinicDB datasets undergo pre-processing as per the protocols in (Sanderson and Matuszewski 2022), with all RGB images normalized to the range of [-1, 1] and subjected to both spatial and color space augmentations.

## B Metrics

To quantify the performance of our segmentation models, we utilize a set of standardized metrics across our experiments. These include Dice score (Dice), intersection over union (IoU), precision (Prec), recall (Rec), and Hausdorff distance in millimeters (HD). For 3D images (Synapse, CHAOS T2-SPIR, LGE, and bSSFP datasets), these metrics are calculated over the entire 3D volume, whereas for 2D images in the Kvasir-SEG and CVC-ClinicDB datasets, evaluations are performed on individual images. Consistent with (Sanderson and Matuszewski 2022), we employ the prefix 'm' (e.g., mDice) to denote the mean scores for metrics in the polyp segmentation datasets.

## C Implementation Details

## Semantic Stack Generation

We leverage Stable Diffusion (SD) 2.1, fine-tuned specifically to our training datasets, to generate synthetic images. This process is augmented with segmentation-mapcontrolled ControlNet, enabling precise adherence to the ground truth segmentation maps during synthetic image generation. The resizing and fine-tuning parameters are carefully chosen based on dataset characteristics and prior literature.

For datasets including Synapse (8 classes), ACDC, Kvasir-SEG, and CVC-ClinicDB, we standardize the image dimensions to 512 × 512 , aligning with the native resolution of SD 2.1. For Synapse (4 classes), CHAOS T2-SPIR, LGE, and bSSFP, the images are adjusted to 192 × 192 , as suggested by (Su et al. 2023; Ouyang et al. 2022). ControlNet is fine-tuned over 100 epochs with a batch size of 16 and a learning rate of 1 e -5 , with the SD parameters frozen to ensure consistency. Distinct models are trained for each dataset to mitigate the risk of test domain data leakage.

For constructing semantic stacks, we opt for a stack size of n = 16 synthetic images for each ground truth segmentation mask within the training set. The sampling process utilizes a denoising diffusion implicit model, executed over 50 steps with a strength setting of 1.0, scale of 9.0, and eta of 0.0. To ensure experimental repeatability, the random seed is maintained consistently throughout all experiments. Examples of the generated images are presented in Fig. 8, illustrating the efficacy and precision of the synthetic image generation process.

The control mechanism for synthetic image generation leverages the ground truth segmentation maps, coupled with structured text descriptions detailed in Table 7, as prompts. This methodological choice is aimed at enhancing the relevance and accuracy of the generated images.

The training pipeline illustrated in Fig. 7. The simplicity allows for versatile application.

## Evaluation Methodology

Our evaluation strategy strictly adheres to the foundational training and testing parameters established by the respective base methods. This section delineates only the distinctions introduced by the implementation of our S2S2 strategy. In all experiments, alongside the original image, we uniformly select a single image from the generated stack for analysis.

For TransUNet (Chen et al. 2021), experiments are standardized with α enc = 1 and/or α dec = 1 to maintain sim-

```
Dataset Synapse 2 (8 Classes) A 2D slice of an abdomen CT scan showing [class names]. ACDC (Bernard et al. 2018) (3 Class) A 2D slice of a cardiac MRI scan showing [class names]. Synapse (4 Classes) A 2D slice of an abdominal CT scan showing [class names]. T2-SPIR (Kavur et al. 2021) (4 Class) A 2D slice of an abdominal T2-SPIR MRI scan showing [class names]. LGE (Zhuang et al. 2022) (3 Class) A 2D slice of a cardiac MRI scan using balanced steady-state free precession showing [class names]. bSSFP (Zhuang et al. 2022) (3 Class) A 2D slice of a cardiac MRI scan using late gadolinium enhanced showing [class names]. Kvasir-SEG (Jha et al. 2020) (binary) An image of the human gastrointestinal tract captured by colonoscope showing [class names]. CVC-ClinicDB (Bernal et al. 2015) (binary) An image of the human gastrointestinal tract captured by colonoscope [class names].
```

Table 7: Prompts used for the generative model for each dataset. Top row: the dataset and class name. Bottom row: the corresponding text prompt. The final prompt is created by concatenating the dataset-specific prompt with the class names of the class names present in the image.

```
1 """ 2 Only need two images for each mask 3 at each iteration base on Sec 3.2. 4 We use the original image as one 5 of the images 6 """ 7 for image_0, mask in dataset: 8 image_1 = finetuned_gen_model(mask) 9 # encode the images 10 enc_feat_0 = seg_encoder(image_0) 11 enc_feat_1 = seg_encoder(image_1) 12 # decode the encoder features 13 dec_feat_0 = seg_decoder(enc_feat_0) 14 dec_feat_1 = seg_decoder(enc_feat_1) 15 # pixel-level classification 16 logits_0 = linear(dec_feat_0) 17 logits_1 = linear(dec_feat_1) 18 # compute the segmentation loss 19 loss = seg_loss(image_0,mask) + seg_loss(image_1,mask) 20 # compute the encoder consistency loss 21 loss += alpha_enc * enc_consist(enc_feat_0,enc_feat_1) 22 # compute the decoder consistency loss 23 loss += alpha_dec * enc_consist(dec_feat_0,dec_feat_1) 24 # update the model parameters 25 loss.backward() 26 optimizer.step()
```

Figure 7: Pseudocode for S2S2 training.

plicity in variable adjustment. When incorporating FCBFormer (Sanderson and Matuszewski 2022), which features dual encoders, we extend the application of the semantic similarity loss across both encoders and the decoder for comprehensive ablation studies. In the final evaluation, the loss is specifically applied to the decoder with α dec = 0 and two encoders with α enc 1 = 0 . 4 and α enc 2 = 0 . 4 , optimizing for balanced performance enhancement.

For experiments using SLAug (Su et al. 2023), we consistently apply α enc = 0 . 1 and α dec = 0 across all trials. Notably, our methodology demonstrated a tendency for achieving heightened in-domain performance relatively early in the training cycle, likely attributable to an increased initial loss magnitude. Consequently, we opt for an early stopping of the training process at 1,100 epochs for our method, as opposed to extending to the full 2,000 epochs. However, applying early stopping to the SLAug baseline negatively affects performance. Therefore, for SLAug, we adhere to the original epoch settings to preserve the integrity of comparative analysis. To apply SLAug on RGB images, we train the model for 500 epochs on all the experiments.

## D Additional Results

This section provides additional detailed results. All the metrics and classes used in the original work are reported.

However, as shown in Table 12, our method does not outperform the base method in the CT-MRI and bSSFPLGE settings. A possible explanation is that the assumptions made by SLAug align better with these specific settings. SLAug incorporates domain knowledge, such as intensity differences between source and target domains, into its augmentation strategy. If these augmentations effectively captures the variation in the target domain, the target domain performance will improve. For instance, SLAug+CT (or SLAug+bSSFP) may better address the variations introduced by MRI (or LGE) than the reverse setup.

Overall, our method improves both in-domain and out-ofdomain performance when the base method lacks domain knowledge (e.g., TransUNet, FCBFormer) or is based on incorrect domain assumptions (e.g., SLAug on RGB). Even in scenarios where domain knowledge is available (e.g., SLAug on CT and MRI), our method achieves an average improvement in both in-domain and out-of-domain performance.

As a data-driven method that does not rely on domain knowledge, our approach is not tailored to a specific target domain but is instead designed to enhance robustness across all target domains. Therefore, if the domain-specific augmentation introduced by SLAug already captures the variation in the target domain, the additional application of our method may not provide additional benefits.

Figure 8: Visualization of synthetic images generated.

<!-- image -->

| Method                                |   Aorta L-Kidney R-Kidney Pancreas |   Gallbladder |       |       |   Liver |       |   Spleen |   Stomach |   Average Dice ↑ |   HD ↓ |
|---------------------------------------|------------------------------------|---------------|-------|-------|---------|-------|----------|-----------|------------------|--------|
| R50-AttnUNet (Schlemper et al. 2019)  |                              55.92 |         63.91 | 79.20 | 72.71 |   93.56 | 49.37 |    87.19 |     74.95 |            75.57 |  36.97 |
| ViT (Dosovitskiy et al. 2021)         |                              44.38 |         39.59 | 67.46 | 62.94 |   89.21 | 43.14 |    75.45 |     69.78 |            61.50 |  39.61 |
| ViT-CUP (Dosovitskiy et al. 2021)     |                              70.19 |         45.10 | 74.70 | 67.40 |   91.32 | 42.00 |    81.75 |     70.44 |            67.86 |  36.11 |
| R50-ViT-CUP (Dosovitskiy et al. 2021) |                              73.73 |         55.13 | 75.80 | 72.20 |   91.51 | 45.99 |    81.99 |     73.95 |            71.29 |  32.87 |
| TransUNet (Chen et al. 2021)          |                              86.81 |         56.82 | 81.99 | 78.13 |   93.95 | 55.44 |    85.07 |     76.64 |            76.86 |  26.73 |
| TransUNet+S2S2                        |                              87.52 |         63.40 | 86.39 | 82.61 |   94.76 | 64.55 |    89.41 |     80.87 |            81.19 |  24.81 |

Table 8: In-domain performance comparison on the Synapse multi-organ CT dataset across baseline architectures. The average Dice score (%), average Hausdorff distance (mm), and Dice score (%) for each organ are reported. The best-performing method is highlighted in bold, and the second-best is underlined.

| Method                                | RVC MYO LVC       |   Average |
|---------------------------------------|-------------------|-----------|
| R50-AttnUNet (Schlemper et al. 2019)  | 87.58 79.20 93.47 |     86.75 |
| ViT-CUP (Dosovitskiy et al. 2021)     | 81.46 70.71 92.18 |     81.45 |
| R50-ViT-CUP (Dosovitskiy et al. 2021) | 86.07 81.88 94.75 |     87.57 |
| TransUNet (Chen et al. 2021)          | 89.28 81.80 95.49 |     88.86 |
| TransUNet+S2S2                        | 88.95 86.16 96.07 |     90.40 |

Table 9: In-domain performance comparison on the ACDC dataset in Dice score (%). The best-performing method is highlighted in bold, and the second-best is underlined.

Table 10: In-domain performance comparison on slices of 3D medical image datasets. Dice score (%) is used as the evaluation metric. The best-performing method is highlighted in bold, and the second-best is underlined.

| Method                          | Abdominal CT (Synapse)   | Abdominal CT (Synapse)   | Abdominal CT (Synapse)   | Abdominal CT (Synapse)   | Abdominal CT (Synapse)   | Cardiac bSSFP   | Cardiac bSSFP   | Cardiac bSSFP   |
|---------------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|-----------------|-----------------|-----------------|
|                                 | Liver                    | R-Kidney                 | L-Kidney                 | Spleen                   | Average                  | LVC MYO         | RVC             | Average         |
| Supervised (Ouyang et al. 2022) | 98.87                    | 92.11                    | 91.75                    | 88.55                    | 89.74                    | 91.16 82.93     | 90.39           | 88.16           |
| SLAug (Su et al. 2023)          | 96.48                    | 66.97                    | 79.24                    | 87.92                    | 82.66                    | 95.55 88.10     | 93.14           | 92.27           |
| SLAug+S2S2                      | 96.60                    | 67.66                    | 83.58                    | 89.01                    | 84.21                    | 95.85 87.58     | 93.05           | 92.16           |
| Method                          | Abdominal MRI (T2-SPIR)  | Abdominal MRI (T2-SPIR)  | Abdominal MRI (T2-SPIR)  | Abdominal MRI (T2-SPIR)  | Abdominal MRI (T2-SPIR)  | Cardiac LGE     | Cardiac LGE     | Cardiac LGE     |
| Method                          | Liver                    | R-Kidney                 | L-Kidney                 | Spleen                   | Average                  | LVC MYO         | RVC             | Average         |
| Supervised (Ouyang et al. 2022) | 91.30                    | 92.43                    | 89.86                    | 89.83                    | 90.85                    | 92.04 83.11     | 89.30           | 88.15           |
| SLAug (Su et al. 2023)          | 91.75                    | 92.29                    | 91.14                    | 87.22                    | 90.60                    | 89.31 81.50     | 91.25           | 87.35           |
| SLAug+S2S2                      | 91.69                    | 91.84                    | 90.72                    | 90.88                    | 91.28                    | 89.55 81.83     | 91.47           | 87.62           |

Table 11: In-domain performance comparison on RGB datasets. Baseline model results are taken from (Sanderson and Matuszewski 2022). Metrics are reported in percentages (%). The best-performing method is highlighted in bold, and the secondbest is underlined.

| Method                                     | Kvasir-SEG (Jha et al. 2020) CVC-ClinicDB (Bernal et al. 2015) IoU   | Kvasir-SEG (Jha et al. 2020) CVC-ClinicDB (Bernal et al. 2015) IoU   | Kvasir-SEG (Jha et al. 2020) CVC-ClinicDB (Bernal et al. 2015) IoU   | Kvasir-SEG (Jha et al. 2020) CVC-ClinicDB (Bernal et al. 2015) IoU   | Kvasir-SEG (Jha et al. 2020) CVC-ClinicDB (Bernal et al. 2015) IoU   | Kvasir-SEG (Jha et al. 2020) CVC-ClinicDB (Bernal et al. 2015) IoU   | Kvasir-SEG (Jha et al. 2020) CVC-ClinicDB (Bernal et al. 2015) IoU   | Kvasir-SEG (Jha et al. 2020) CVC-ClinicDB (Bernal et al. 2015) IoU   |
|--------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
|                                            | Dice                                                                 | IoU                                                                  | Prec.                                                                | Rec.                                                                 | Dice                                                                 | Prec.                                                                | Rec.                                                                 |                                                                      |
| MSRF-Net (Srivastava et al. 2021)          | 92.17                                                                | 89.14                                                                | 96.66                                                                | 91.98                                                                | 94.20 90.43                                                          | 94.27                                                                | 95.67                                                                |                                                                      |
| PraNet (Fan et al. 2020)                   | 89.80                                                                | 84.00                                                                | -                                                                    | -                                                                    | 89.90 84.90                                                          | -                                                                    | -                                                                    |                                                                      |
| SLAug (Su et al. 2023)                     | 84.85                                                                | 77.3                                                                 | 88.12                                                                | 84.75                                                                | 85.39 76.98                                                          | 82.43                                                                | 91.37                                                                |                                                                      |
| SLAug+S2S2                                 | 85.33                                                                | 78.00                                                                | 86.58                                                                | 86.91                                                                | 88.76 81.01                                                          | 88.73                                                                | 90.34                                                                |                                                                      |
| FCBFormer (Sanderson and Matuszewski 2022) | 91.90                                                                | 87.05                                                                | 94.05                                                                | 91.62                                                                | 93.46 89.17                                                          | 93.57                                                                | 93.66                                                                |                                                                      |
| FCBFormer+S2S2                             | 93.20                                                                | 88.57                                                                | 94.54                                                                | 93.59                                                                | 94.88 90.41                                                          | 94.63                                                                | 95.43                                                                |                                                                      |

| Method                           | Abdominal CT-MRI                       | Abdominal CT-MRI                       | Abdominal CT-MRI                       | Abdominal CT-MRI                       | Abdominal CT-MRI                       | Cardiac bSSFP-LGE   | Cardiac bSSFP-LGE   |
|----------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|---------------------|---------------------|
|                                  | Liver                                  | R-Kidney                               | L-Kidney                               | Spleen                                 | Average                                | LVC MYO RVC         | Average             |
| Cutout (DeVries and Taylor 2017) | 79.80                                  | 82.32                                  | 82.14                                  | 76.24                                  | 80.12                                  | 88.35 69.06 79.19   | 78.87               |
| RSC (Huang et al. 2020)          | 76.40                                  | 75.79                                  | 76.60                                  | 67.56                                  | 74.09                                  | 87.06 69.77 75.69   | 77.51               |
| MixStyle (Zhou et al. 2021)      | 77.63                                  | 78.41                                  | 78.03                                  | 77.12                                  | 77.80                                  | 85.78 64.23 75.61   | 75.21               |
| AdvBias (Carlucci et al. 2019)   | 78.54                                  | 81.70                                  | 80.69                                  | 79.73                                  | 80.17                                  | 88.23 70.29 80.32   | 79.62               |
| RandConv (Xu et al. 2021)        | 73.63                                  | 79.69                                  | 85.89                                  | 83.43                                  | 80.66                                  | 89.88 75.60 85.70   | 83.73               |
| CSDG (Ouyang et al. 2022)        | 86.62                                  | 87.48                                  | 86.88                                  | 84.27                                  | 86.31                                  | 90.35 77.82 86.87   | 85.01               |
| SLAug (Su et al. 2023)           | 89.97                                  | 89.39                                  | 87.40                                  | 87.45                                  | 88.55                                  | 91.56 80.28 87.43   | 86.42               |
| SLAug+S2S2                       | 90.71                                  | 89.22                                  | 86.55                                  | 84.51                                  | 87.75                                  | 91.48 79.84 86.87   | 86.06               |
| Method                           | Abdominal MRI-CT                       | Abdominal MRI-CT                       | Abdominal MRI-CT                       | Abdominal MRI-CT                       | Abdominal MRI-CT                       | Cardiac LGE-bSSFP   | Cardiac LGE-bSSFP   |
|                                  | Liver R-Kidney L-Kidney Spleen Average | Liver R-Kidney L-Kidney Spleen Average | Liver R-Kidney L-Kidney Spleen Average | Liver R-Kidney L-Kidney Spleen Average | Liver R-Kidney L-Kidney Spleen Average | LVC MYO RVC         | Average             |
| Cutout (DeVries and Taylor 2017) | 86.99                                  | 63.66                                  | 73.74                                  | 57.60                                  | 70.50                                  | 90.88 79.14 87.74   | 85.92               |
| RSC (Huang et al. 2020)          | 88.10                                  | 46.60                                  | 75.94                                  | 53.61                                  | 66.07                                  | 90.21 78.63 87.96   | 85.60               |
| MixStyle (Zhou et al. 2021)      | 86.66                                  | 48.26                                  | 65.20                                  | 55.68                                  | 63.95                                  | 91.22 79.64 88.16   | 86.34               |
| AdvBias (Carlucci et al. 2019)   | 87.63                                  | 52.48                                  | 68.28                                  | 50.95                                  | 64.84                                  | 91.20 79.50 88.10   | 86.27               |
| RandConv (Xu et al. 2021)        | 84.14                                  | 76.81                                  | 77.99                                  | 67.32                                  | 76.56                                  | 91.98 80.92 88.83   | 87.24               |
| CSDG (Ouyang et al. 2022)        | 85.62                                  | 80.02                                  | 80.42                                  | 75.56                                  | 80.40                                  | 91.37 80.43 89.16   | 86.99               |
| SLAug (Su et al. 2023)           | 88.87                                  | 80.23                                  | 81.59                                  | 76.12                                  | 81.70                                  | 91.43 80.64 89.43   | 87.17               |
| SLAug+S2S2                       | 88.30                                  | 81.79                                  | 80.31                                  | 82.21                                  | 83.15                                  | 92.17 80.19 90.10   | 87.49               |

Table 12: Out-of-domain performance on slices of 3D medical image datasets. Dice score (%) is used as the evaluation metric. The best-performing method is highlighted in bold, and the second-best is underlined.

Table 13: Out-of-domain performance on Polyp segmentation (RGB medical image datasets). Metrics are reported in percentages (%). The best-performing method is highlighted in bold, and the second-best is underlined.

| Method                                     | Kvasir-CVC IoU   | Kvasir-CVC IoU    | Kvasir-CVC IoU   | CVC-Kvasir IoU   | CVC-Kvasir IoU   | CVC-Kvasir IoU   | CVC-Kvasir IoU   |
|--------------------------------------------|------------------|-------------------|------------------|------------------|------------------|------------------|------------------|
|                                            | Dice             | Prec.             | Rec.             | Dice             | Prec.            | Rec.             |                  |
| MSRF-Net (Srivastava et al. 2021)          | 62.38            | 54.19 66.21       | 70.51            | 72.96            | 64.15 81.62      | 74.21            |                  |
| PraNet (Fan et al. 2020)                   | 79.12 71.19      | 81.52             | 83.16            | 79.50 70.73      | 76.87            | 90.50            |                  |
| SLAug (Su et al. 2023)                     | 75.62 66.97      | 83.19             | 76.65            | 77.09 67.91      | 74.34            | 89.11            |                  |
| SLAug+S2S2                                 | 76.44            | 67.81 81.97 85.40 | 79.37            | 80.52 72.14      | 85.49            | 82.12            |                  |
| FCBFormer (Sanderson and Matuszewski 2022) | 91.16            | 91.89             | 91.31            | 86.46 80.27      | 92.92            | 85.22            |                  |
| FCBFormer+S2S2                             | 92.85            | 86.94 93.46       | 92.95            | 88.72 82.79      | 92.33            | 88.91            |                  |

Table 14: Performance metrics on when applying the semantic consistency loss with only photometric augmentation (Aug). Metrics are reported in percentages (%). The best-performing method is highlighted in bold.

| Method                 | CVC Train   | CVC Train   | CVC Train   | CVC Train   | Kvasir Train   | Kvasir Train   | Kvasir Train   | Kvasir Train   |
|------------------------|-------------|-------------|-------------|-------------|----------------|----------------|----------------|----------------|
| Method                 | Dice        | IoU         | Precision   | Recall      | Dice           | IoU            | Precision      | Recall         |
| Baseline (CVC Test)    | 93.46       | 89.17       | 93.57       | 93.66       | 91.90          | 87.05          | 94.05          | 91.62          |
| Aug (CVC Test)         | 84.10 79.79 |             | 85.99       | 85.21       | 85.27          | 79.15          | 93.67          | 82.21          |
| Baseline (Kvasir Test) | 91.16 85.40 |             | 91.89       | 91.31       | 91.90          | 87.05          | 94.05          | 93.66          |
| Aug (Kvasir Test)      | 81.87 76.16 |             | 82.19       | 83.54       | 92.42          | 87.82          | 92.73          | 93.66          |