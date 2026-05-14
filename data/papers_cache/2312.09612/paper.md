## TOP-ReID: Multi-spectral Object Re-Identification with Token Permutation

Yuhao Wang 1 , Xuehu Liu 2 , Pingping Zhang 1 * , Hu Lu 3 , Zhengzheng Tu 4 , Huchuan Lu 1

1

School of Future Technology, School of Artificial Intelligence, Dalian University of Technology 2 School of Computer Science and Artificial Intelligence, Wuhan University of Technology 3 School of Computer Science and Communication Engineering, Jiangsu University 4 School of Computer Science and Technology, Anhui University

## Abstract

Multi-spectral object Re-identification (ReID) aims to retrieve specific objects by leveraging complementary information from different image spectra. It delivers great advantages over traditional single-spectral ReID in complex visual environment. However, the significant distribution gap among different image spectra poses great challenges for effective multi-spectral feature representations. In addition, most of current Transformer-based ReID methods only utilize the global feature of class tokens to achieve the holistic retrieval, ignoring the local discriminative ones. To address the above issues, we step further to utilize all the tokens of Transformers and propose a cyclic token permutation framework for multi-spectral object ReID, dubbled TOP-ReID. More specifically, we first deploy a multi-stream deep network based on vision Transformers to preserve distinct information from different image spectra. Then, we propose a Token Permutation Module (TPM) for cyclic multi-spectral feature aggregation. It not only facilitates the spatial feature alignment across different image spectra, but also allows the class token of each spectrum to perceive the local details of other spectra. Meanwhile, we propose a Complementary Reconstruction Module (CRM), which introduces dense token-level reconstruction constraints to reduce the distribution gap across different image spectra. With the above modules, our proposed framework can generate more discriminative multi-spectral features for robust object ReID. Extensive experiments on three ReID benchmarks (i.e., RGBNT201, RGBNT100 and MSVR310) verify the effectiveness of our methods. The code is available at https://github.com/924973292/TOP-ReID.

## Introduction

Object Re-identification (ReID) aims to retrieve specific objects from images or videos across non-overlapping cameras, which has advanced significantly over the past decades. In the traditional object ReID, researchers primarily utilize single-spectral images (such as RGB, depth) to extract visual information of the targets. However, single-spectral images provide very limited representation abilities in scenarios characterized by low resolution, darkness, glare, etc. As illustrated in the top row of Fig. 1, the outlines of persons are notably blurred, leading to an evident confusion between persons and the background in the RGB image spectrum. Hence, relying only on RGB images poses great challenges for robust object ReID. Fortunately, other image spectra are very useful to address above problems. In fact, Near Infrared (NIR) imaging is unaffected by darkness and adverse weather conditions (Li et al. 2020b). Thus, there have been some efforts (Li et al. 2020a; Liu et al. 2021a; Zhang and Wang 2023) to incorporate NIR images to enhance the performance of object ReID. Nonetheless, NIR images retain some limitations (Zheng et al. 2021), as depicted in Fig. 1. For example, the details of persons in NIR images tend to be substantially obscured in the presence of glare. Meanwhile, Thermal Infrared (TIR) imaging is more robust to these scenarios (Zheng et al. 2021). As illustrated in Fig. 1, TIR images can highlight persons from the background and preserve crucial details, such as glasses and backpacks. These facts clearly show the information complementarity of different image spectra for object ReID. Based on the above facts, multi-spectral object ReID aims to retrieve specific ob- jects by leveraging complementary information from different image spectra, e.g., RGB, NIR, TIR etc. It delivers great advantages over single-spectral ReID in complex visual environment. In fact, some methods (Zheng et al. 2021; Wang et al. 2022b) have already tried to integrate multi-spectral features with simple fusion methods. However, there are significant distribution gaps among different image spectra. Simple fusions can not well address the heterogenous challenges for effective feature representations. In addition, it often involves the absence of image spectra in real world, as shown in the 'Missing-spectral Test' of Fig. 1. Thus, there are much room for improving multi-spectral feature fusion.

* Corresponding author (zhpp@dlut.edu.cn).

Copyright © 2024, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

Figure 1: The top displays instances from RGBNT201 in various challenges, while the bottom presents the object ReID settings for multi-spectral and missing-spectral test.

<!-- image -->

Meanwhile, with the great advance of vision Transformers (Dosovitskiy et al. 2020), some works (He et al. 2021; Pan et al. 2022) have introduced Transformers for object ReID. However, most of current Transformer-based ReID methods only utilize the global feature of class tokens to achieve the holistic retrieval, ignoring the local discriminative ones. To address the above issues, we step further to utilize all the tokens of Transformers and propose a cyclic token permutation framework for multi-spectral object ReID, dubbled TOP-ReID. Specifically, it consists of two key modules: Token Permutation Module (TPM) and Complementary Reconstruction Module (CRM). Technically, we first deploy a multi-stream deep network based on vision Transformers to preserve distinct information from different image spectra. Then, TPM takes all the tokens from the multistream deep network as inputs, and cyclically permutes the specific class tokens and the corresponding patch tokens from other spectra. In this way, it not only facilitates the spatial feature alignment across different image spectra, but also allows the class token of each spectrum to perceive the local details of other spectra. Meanwhile, CRM is proposed to facilitate local information interaction and reconstruction across different image spectra. Through introducing tokenlevel reconstruction constraints, it can reduce the distribution gap across different image spectra. As a result, the CRM can further handle the missing-spectral problem. With the proposed modules, our framework can extract more discriminative features from multi-spectral images for robust object ReID. Comprehensive experiments are conducted on three multi-spectral object ReID benchmarks, i.e., RGBNT201, RGBNT100 and MSVR310. Experimental results clearly show the effectiveness of our proposed methods.

In summary, our contributions can be stated as follows:

- We propose a novel feature learning framework named TOP-ReID for multi-spectral object ReID. To our best knowledge, our proposed TOP-ReID is the first work to utilize all the tokens of vision Transformers to improve the multi-spectral object ReID.
- We propose a Token Permutation Module (TPM) and a Complementary Reconstruction Module (CRM) to facilitate multi-spectral feature alignment and handle spectralmissing problems effectively.
- We perform comprehensive experiments on three multispectral object ReID benchmarks, i.e., RGBNT201, RGBNT100 and MSVR310. The results fully verify the effectiveness of our proposed methods.

## Related Work Single-spectral Object ReID

Single-spectral object ReID focuses on extracting discriminative features from single-spectral images. Typical singlespectral forms include RGB, NIR, TIR and depth. Due to the easy requirement, RGB images play a fundamental role in the single-spectral object ReID. As for the techniques, most of existing object ReID methods are based on Convolutional Neural Networks (CNNs). For example, Luo et al. (Luo et al. 2019) utilize a deep residual network and introduce the BNNeck technique for object ReID. Furthermore, PCB (Sun et al. 2018) and MGN (Wang et al. 2018) adapt a stripe-based image division strategy to obtain multi-grained representations. OSNet (Zhou et al. 2019) employs a unified aggregation gate for fusing omni-scale features. AGW (Ye et al. 2021) incorporates non-local attention mechanisms for fine-grained feature extraction. Nevertheless, due to the limited receptive field, CNN-based methods(Qian et al. 2017; Li, Zhu, and Gong 2018; Chang, Hospedales, and Xiang 2018; Chen et al. 2019; Sun et al. 2020; Rao et al. 2021; Zhao et al. 2021; Liu et al. 2021b) are not robust to complex scenarios. Inspired by the success of vision Transformers (ViT) (Dosovitskiy et al. 2020), He et al. (He et al. 2021) propose the first pure Transformer-based method named TransReID for object ReID, yielding competitive results through the adaptive modeling of image patches. Afterwards, numerous Transformer-based methods (Zhu et al. 2021; Zhang et al. 2021; Chen et al. 2022; Wang et al. 2022a; Liu et al. 2023) demonstrate their advantages in object ReID. However, all these methods take single-spectral images as inputs, providing limited representation abilities. Thus, they can not handle the all-day object ReID problem.

## Multi-spectral Object ReID

The robustness of multi-spectral data draws the attention of numerous researchers. For multi-spectral person ReID, Zheng et al. (Zheng et al. 2021) advance the field and design a PFNet to learn robust RGB-NIR-TIR features. Then, Wang et al. (Wang et al. 2022b) boost modality-specific representations with three learning strategies, named IEEE. Furthermore, Zheng et al. (Zheng et al. 2023) design a DENet to address the spectral-missing problem. For multi-spectral vehicle ReID, Li et al. (Li et al. 2020b) propose a HAMNet to fuse different spectral features. Considering the relationship between different image spectra, Guo et al. (Guo et al. 2022) propose a GAFNet to fuse the multiple data sources. He et al. (He et al. 2023) propose a GPFNet to adaptively fuse multi-spectral features. Zheng et al. (Zheng et al. 2022) propose a CCNet to simultaneously overcome the discrepancies from both modality and sample aspects. Pan et al. (Pan et al. 2022) propose a HViT to balance modal-specific and modal-shared information. Furthermore, they employ a random hybrid augmentation and a feature hybrid mechanism to improve the performance (Pan et al. 2023). Although effective, previous methods mainly treat the NIR and TIR as an assistant to RGB, rather than adaptively fuse them with multi-level spatial correspondences. In contrast, we facilitate the spatial feature alignment across different image spectra.

Figure 2: An illustration of the proposed TOP-ReID. First, deep features from RGB, NIR and TIR images are extracted by using three independent ViT-B/16. Then, a Token Permutation Module (TPM) is proposed for cyclic multi-spectral feature aggregation through three consecutive token permutations. Meanwhile, a Complementary Reconstruction Module (CRM) is used to achieve token-level reconstruction constraints. When inference, we utilize the permutated features for ranking the person candidates.

<!-- image -->

## Proposed Method

As illustrated in Fig. 2, our proposed TOP-ReID consists of three main components: Multi-stream Feature Extraction, Token Permutation Module (TPM) and Complementary Reconstruction Module (CRM).

## Multi-stream Feature Extraction

In this work, we take images of three spectra for object ReID, i.e., RGB, NIR and TIR. To capture the distinctive characteristics of each spectrum, we follow previous works (Li et al. 2020b; Zheng et al. 2021) and adopt three independent backbones. More specifically, vision Transformers (ViT) can be deployed as the backbone in each stream. Formally, the multi-stream features can be represented as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where I R ∈ R H × W × 3 , I N ∈ R H × W × 3 and I T ∈ R H × W × 3 denote the input RGB, NIR and TIR images, respectively. Here, ViT can be any vision Transformers (e.g., ViTB/16 (Dosovitskiy et al. 2020), DeiT-S/16 (Touvron et al. 2021), T2T-ViT-24 (Yuan et al. 2021)). The token features F R , F N , F T ∈ R D × ( M +1) are extracted from the final layer of ViT , respectively. Additional learnable class token is included. D denotes the embedding dimension while M means the number of patch tokens. These independent streams enable the extraction of spectral-specific features, capturing rich information from different image spectra.

## Token Permutation Module

To achieve the spatial feature alignment among different image spectra and the effective aggregation of heterogeneous features, we introduce the Token Permutation Module (TPM) with a cyclic token permutation mechanism, as illustrated at the top right corner of Fig. 2.

Technically, TPM takes the token features F R , F N and F T as inputs, and generates the fused feature f tp with three consecutive token permutations. Without loss of generality, we take the RGB stream as a starting example. As shown in Fig. 3 (a), we utilize a Multi-Head Cross-Attention (MHCA) (Dosovitskiy et al. 2020) with N h heads to achieve the token permutation. More specifically, the class token f (R , 0) ∈ R D from F R is passed into a linear transformation to generate a query matrix Q ∈ R D . The patch tokens F patch N ∈ R D × M from F N are passed into two linear transformations to generate a key matrix K and a value matrix V , respectively. Thus, the interaction of F R and F N in the h -th head is represented as

<!-- formula-not-decoded -->

where σ is the softmax function and ( · ) ⊤ means the matrix transposition. Here, Q h ∈ R d , K h , V h ∈ R d × M , d = D N h .

Figure 3: Our token permutation and TransRe blocks with the RGB stream. Other streams share a similar structure.

<!-- image -->

The outputs of N h heads ( ˆ f 1 (R , 1) , · · · , ˆ f h (R , 1) , · · · , ˆ f N h (R , 1) ) are concatenated to be ˆ f (R , 1) ∈ R D . Then, ˆ f (R , 1) is passed through a Feed-Forward Network (FFN) to generate a new class token f (R , 1) ,

<!-- formula-not-decoded -->

It serves as the initial spatial alignment of the RGB and NIR image features. Similar operations can be performed for other spectra,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

As shown in Fig. 3 (a) and above equations, we additionally introduce the LayerNorm (LN) (Ba, Kiros, and Hinton 2016) to Q and K to ensure the numerical stability. Thus, the first token permutation can be totally formulated as

<!-- formula-not-decoded -->

From the above equations, it can be observed that the token permutation enables the global class token from each spectrum to interact with the local patch tokens of the next spectrum, achieving the initial feature fusion and alignment.

Furthermore, the permutated class tokens f (R , 1) , f (N , 1) , and f (T , 1) are paired with their initial patch tokens to form F R → N , F N → T , and F T → R , respectively. The class tokens keep shifting to the next spectrum,

<!-- formula-not-decoded -->

At this stage, each spectrum has already incorporated detail information from other spectra. Similar to the previous step, the permutated class tokens f (R , 2) , f (N , 2) , and f (T , 2) are paired with permutated patch tokens to form F RN → T , F NT → R , and F TR → N , respectively. Finally, the token permutation process ends with each class token interacting with its own patch tokens,

<!-- formula-not-decoded -->

Through the above token permutation, the information from all other spectra is conveyed to the patch tokens through the class token, enabling robust feature alignment. Finally, we concatenate the permutated class tokens to obtain the permutated representation f tp ∈ R 3 D ,

<!-- formula-not-decoded -->

This cyclic token permutation enhances the spatial fusion and implicit alignment of deep features across spectra, improving the ability of inter-spectral dependencies.

## Complementary Reconstruction Module

There are significant distribution gaps among different image spectra. In addition, it often involves the absence of certain image spectra in real world. Inspired by the image generation (Zhu et al. 2017), we propose a Complementary Reconstruction Module (CRM) to reduce the distribution gap across different image spectra. The key is to incorporate dense token-level reconstruction constraints.

Without loss of generality, we take the RGB stream as an example and consider the NIR and TIR spectra missing. To reconstruct the missing tokens, we pass F R through a Transformer-based Reconstruction (TransRe) block (See Fig. 3 (b)) and generate the corresponding tokens by

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where F R2N , F R2T ∈ R D × ( M +1) are the reconstructed tokens. The reconstructed tokens F R2N and F R2T are constrained by the real token features F N and F T using the Mean Squared Error (MSE) loss:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Through the above token-level reconstruction constraints, the distribution gap between RGB and other spectra is reduced. To improve the reconstruction ability, we introduce similar constraints to all the image spectra and achieve a multi-spectral complementary reconstruction. The complementary reconstruction loss L cr can be expressed as the sum of the individual losses for each spectrum:

<!-- formula-not-decoded -->

By introducing token-level constraints, our CRM effectively reduces the distribution gap among different image spectra. Moreover, it can generate corresponding tokens of missing spectra, ensuring a unified learning framework even in scenarios where one or more spectra are absent.

## Dynamic Cooperation between CRM and TPM

In this work, we further introduce the dynamic cooperation between the CRM and TPM to handle the absence of any image spectrum. For example, when the RGB image spectrum is missing, the token features F N and F T will activate their reconstruction blocks to generate the corresponding RGB token features F N2R and F T2R , respectively. Then, the reconstructed RGB token features can be represented as

<!-- formula-not-decoded -->

Then, ¯ F R , F N and F T can be fed into the TPM to perform the token permutation as normal. Hence, our CRM can dynamically cooperate with TPM, ensuring that the missing spectrum can still participate in the permutation process.

## Objective Function

As illustrated in Fig. 2, our objective function comprises three components: loss for the ViT backbone, loss for the token permutation and loss for the CRM. As for the ViT backbone and the token permutation, they are both supervised by the label smoothing cross-entropy loss (Szegedy et al. 2016) and triplet loss (Hermans, Beyer, and Leibe 2017). Finally, the total loss in our framework can be defined by

<!-- formula-not-decoded -->

## Experiments

## Dataset and Evaluation Protocols

To evaluate the performance, we adopt three multi-spectral object ReID datasets. RGBNT201 (Zheng et al. 2021) is the first multi-spectral person ReID dataset with RGB, NIR and TIR spectra. RGBNT100 (Li et al. 2020b) is a large-scale multi-spectral vehicle ReID dataset. MSVR310 (Zheng et al. 2022) is a small-scale multi-spectral vehicle ReID dataset with more complex scenarios. Following previous works, we adopt the mean Average Precision (mAP) and Cumulative Matching Characteristics (CMC) at Rank-K ( K = 1 , 5 , 10 ) as our evaluation metrics.

## Implementation Details

Our model is implemented with the PyTorch toolbox. We conduct experiments with one NVIDIA A800 GPU. We use pre-trained Transformers on the ImageNet classification dataset (Deng et al. 2009) as our backbones. All images are resized to 256 × 128 × 3 pixels. When training, random horizontal flipping, cropping and erasing (Zhong et al. 2020) are used as data augmentation. We set the mini-batch size to 128. Each mini-batch consists of 8 randomly selected object identities, and 16 images are sampled for each identity. We use the Stochastic Gradient Descent (SGD) optimizer with a momentum coefficient of 0.9 and a weight decay of 0.0001. Furthermore, the learning rate is initialized as 0.009. The warmup strategy and cosine decay are used during training.

Table 1: Performance comparison on RGBNT201. The best and second results are in bold and underlined, respectively. * signifies Transformer-based approaches, while others are CNN-based ones.

| Methods   | Methods   | RGBNT201   | RGBNT201   | RGBNT201   | RGBNT201   |
|-----------|-----------|------------|------------|------------|------------|
|           |           | mAP        | R-1        | R-5        | R-10       |
|           | HACNN     | 21.3       | 19.0       | 34.1       | 42.8       |
|           | MUDeep    | 23.8       | 19.7       | 33.1       | 44.3       |
|           | OSNet     | 25.4       | 22.3       | 35.1       | 44.7       |
|           | MLFN      | 26.1       | 24.2       | 35.9       | 44.1       |
|           | CAL       | 27.6       | 24.3       | 36.5       | 45.7       |
|           | PCB       | 32.8       | 28.1       | 37.4       | 46.9       |
|           | HAMNet    | 27.7       | 26.3       | 41.5       | 51.7       |
|           | PFNet     | 38.5       | 38.9       | 52.0       | 58.4       |
|           | DENet     | 42.4       | 42.2       | 55.3       | 64.5       |
|           | IEEE      | 47.5       | 44.4       | 57.1       | 63.6       |
|           | TOP-ReID* | 72.3       | 76.6       | 84.7       | 89.4       |

Table 2: Performance on RGBNT100 and MSVR310.

|         |             | RGBNT100   | RGBNT100   | MSVR310   | MSVR310   |
|---------|-------------|------------|------------|-----------|-----------|
| Methods | Methods     | mAP        | R-1        | mAP       | R-1       |
|         | DMML        | 58.5       | 82.0       | 19.1      | 31.1      |
|         | Circle Loss | 59.4       | 81.7       | 22.7      | 34.2      |
|         | PCB         | 57.2       | 83.5       | 23.2      | 42.9      |
|         | MGN         | 58.1       | 83.1       | 26.2      | 44.3      |
| Single  | BoT         | 78.0       | 95.1       | 23.5      | 38.4      |
|         | HRCN        | 67.1       | 91.8       | 23.4      | 44.2      |
|         | OSNet       | 75.0       | 95.6       | 28.7      | 44.8      |
|         | AGW         | 73.1       | 92.7       | 28.9      | 46.9      |
|         | TransReID*  | 75.6       | 92.9       | 18.4      | 29.6      |
|         | GAFNet      | 74.4       | 93.4       | -         | -         |
|         | GPFNet      | 75.0       | 94.5       | -         | -         |
|         | PHT*        | 79.9       | 92.7       | -         | -         |
| Multi   | PFNet       | 68.1       | 94.1       | 23.5      | 37.4      |
|         | HAMNet      | 74.5       | 93.3       | 27.1      | 42.3      |
|         | CCNet       | 77.2       | 96.3       | 36.4      | 55.2      |
|         | TOP-ReID*   | 81.2       | 96.4       | 35.9      | 44.6      |

## Comparison with State-of-the-Art Methods

Multi-spectral Person ReID. In Tab. 1, we compare our TOP-ReID with both single-spectral methods and multispectral methods on RGBNT201. The results indicate that single-spectral methods generally achieve lower performance compared with multi-spectral methods. It demonstrates the effectiveness of utilizing complementary information from different image spectra. Among the singlespectral methods, PCB achieves the highest performance, attaining the mAP and Rank-1 accuracy of 32.8% and 28.1%, respectively. As for the multi-spectral methods, our TOP-ReID achieves remarkable performance. Specifically, it achieves a mAP that is 24.8% higher and a Rank-1 accuracy that surpasses IEEE by 32.2%. These performance gains provide strong evidences for our TOP-ReID in tackling the challenges of multi-spectral person ReID.

Multi-spectral Vehicle ReID. As shown in Tab. 2, singlespectral methods such as OSNet (Zhou et al. 2019), AGW

Table 3: Experimental results of missing-spectral tasks on RGBNT201. 'M (X)' stands for missing the X image spectra.

| Methods     | M(RGB)   | M(RGB)    | M(NIR)   | M(NIR)   | M(TIR)   | M(TIR)   | M(RGB+NIR)   | M(RGB+NIR)   | M(RGB+TIR)   | M(RGB+TIR)   | M(NIR+TIR)   | M(NIR+TIR)   |
|-------------|----------|-----------|----------|----------|----------|----------|--------------|--------------|--------------|--------------|--------------|--------------|
| Methods     | mAP      | R-1       | mAP      | R-1      | mAP      | R-1      | mAP          | R-1          | mAP          | R-1          | mAP          | R-1          |
| HACNN       | 12.5     | 11.1      | 20.5     | 19.4     | 16.7     | 13.3     | 9.2          | 6.2          | 6.3          | 2.2          | 14.8         | 12.0         |
| MUDeep      | 19.2     | 16.4      | 20.0     | 17.2     | 18.4     | 14.2     | 13.7         | 11.8         | 11.5         | 6.5          | 12.7         | 8.5          |
| OSNet       | 19.8     | 17.3      | 21.0     | 19.0     | 18.7     | 14.6     | 12.3         | 10.9         | 9.4          | 5.4          | 13.0         | 10.2         |
| Single MLFN |          | 20.2 18.9 | 21.1     | 19.7     | 17.6     | 11.1     | 13.2         | 12.1         | 8.3          | 3.5          | 13.1         | 9.1          |
| CAL         |          | 21.4 22.1 | 24.2     | 23.6     | 18.0     | 12.4     | 18.6         | 20.1         | 10.0         | 5.9          | 17.2         | 13.2         |
| PCB         |          | 23.6 24.2 | 24.4     | 25.1     | 19.9     | 14.7     | 20.6         | 23.6         | 11.0         | 6.8          | 18.6         | 14.4         |
|             | PFNet    | - -       | 31.9     | 29.8     | 25.5     | 25.8     | -            | -            | -            | -            | 26.4         | 23.4         |
| Multi       | DENet    | - -       | 35.4     | 36.8     | 33.0     | 35.4     | -            | -            | -            | -            | 32.4         | 29.2         |
| TOP-ReID    | 54.4     | 57.5      | 64.3     | 67.6     | 51.9     | 54.5     | 35.3         | 35.4         | 26.2         | 26.0         | 34.1         | 31.7         |

Table 4: Performance comparison with different modules.

| Modules   | Modules   | Modules   | Modules   | RGBNT201   | RGBNT201   | RGBNT201   | RGBNT201   |
|-----------|-----------|-----------|-----------|------------|------------|------------|------------|
| BL        | AL        | TPM       | CRM       | mAP        | R-1        | R-5        | R-10       |
| A ✓       | ✕         | ✕         | ✕         | 55.9       | 54.9       | 70.8       | 77.6       |
| B ✕       | ✓         | ✕         | ✕         | 62.9       | 64.5       | 77.4       | 82.7       |
| C ✕       | ✓         | ✓         | ✕         | 67.8       | 69.4       | 83.3       | 88.8       |
| D ✕       | ✓         | ✓         | ✓         | 72.3       | 76.6       | 84.7       | 89.4       |

(Ye et al. 2021) and TransReID (He et al. 2021), stand out for their competitive performance. For multi-spectral methods, CCNet achieves remarkable results across both datasets. On the RGBNT100 dataset, our TOP-ReID outperforms CCNet with a 4.0% higher mAP. On the small-scale MSVR310 dataset, our TOP-ReID maintains competitive performance, showing its versatility and robustness.

Evaluation on Missing-spectral Scenarios. As shown in Tab. 3, all single-spectral methods suffer from performance degradations when image spectra are missing. Multi-spectral methods demonstrate better robustness compared with single-spectral methods. Our proposed TOPReID achieves remarkable performance even in the presence of missing spectra. It consistently outperforms both singlespectral and multi-spectral methods in all missing-spectral scenarios, indicating its effectiveness in handling the spectral incompleteness. In addition, compared with PFNet and DENet, our TOP-ReID is a more flexible and diverse framework to address any spectra missing.

## Ablation Studies

To investigate the effect of different components, we further perform a scope of ablation studies on RGBNT201.

Effects of Key Modules. Tab. 4 illustrates the performance comparison with different modules. The Model A is the baseline which utilizes the multi-stream ViT-B/16 backbones. BL means the triplet loss and cross-entropy loss are added before the concatenation of multi-spectral features, while AL means these losses are employed after the feature concatenation. It can be observed that the AL setting shows better results. The main reason is that the fused multispectral features is more powerful than the simple feature concatenation. Furthermore, by integrating our TPM, the Model C yields higher performance with mAP of 67.8% and Rank-1 of 69.4%. By introducing CRM, the final model can achieve the best performance with mAP of 72.3% and Rank-

Figure 4: Performance of deploying TPM at different layers.

<!-- image -->

Figure 5: Performance comparison with different depths of TransRe blocks in CRM.

<!-- image -->

1 of 76.6%. These improvements validate the effectiveness of our key modules in handling complex ReID scenarios.

TPM at Different Layers. In fact, our TPM is a plugand-play module. We explore the effect of TPM at different layers of the ViT backbone. Fig. 4 shows the performance of TPM at different layers. We observe that as the plugged depth of TPM increases, the performance greatly improves. When deployed in the last layer, it achieves the best performance. This indicates that our TPM is more pronounced in deep layers, capturing more discriminative representations.

Effects of TransRe Blocks in CRM. The depth of TransRe blocks may impact the reconstruction ability. As illustrated in Fig. 5, the ReID performance is relatively consistent when using different depths of TransRe blocks. In Fig. 5, we also provide the comparison results with missingspectral cases. It can be observed that the overall performance is acceptable when only using one block. Thus, we utilize one TransRe block to reduce the computation.

Table 5: Performance comparison of different backbones with different spectra and modules on RGBNT201.

| Methods               | ViT-B/16   | ViT-B/16   | ViT-B/16   | ViT-B/16   | DeiT-S/16   | DeiT-S/16   | DeiT-S/16   | DeiT-S/16   | T2T-ViT-24   | T2T-ViT-24   | T2T-ViT-24   | T2T-ViT-24   |
|-----------------------|------------|------------|------------|------------|-------------|-------------|-------------|-------------|--------------|--------------|--------------|--------------|
| Methods               | mAP        | R-1        | R-5        | R-10       | mAP         | R-1         | R-5         | R-10        | mAP          | R-1          | R-5          | R-10         |
| RGB                   | 29.0       | 26.2       | 44.5       | 56.1       | 33.3        | 30.6        | 49.5        | 58.0        | 30.1         | 30.3         | 47.2         | 56.6         |
| NIR                   | 18.7       | 14.0       | 31.1       | 44.9       | 22.7        | 21.4        | 39.4        | 47.6        | 15.8         | 15.3         | 27.0         | 36.8         |
| TIR                   | 33.4       | 32.3       | 52.4       | 63.3       | 27.1        | 26.3        | 41.3        | 51.4        | 34.0         | 36.2         | 52.0         | 62.0         |
| NIR-TIR               | 45.9       | 43.4       | 59.9       | 69.4       | 40.6        | 40.9        | 54.3        | 61.0        | 40.9         | 40.7         | 56.3         | 64.2         |
| RGB-NIR               | 39.0       | 40.2       | 56.6       | 65.7       | 46.7        | 45.0        | 62.6        | 70.0        | 36.3         | 35.2         | 53.8         | 66.3         |
| RGB-TIR               | 52.6       | 53.8       | 69.0       | 78.2       | 49.3        | 47.8        | 64.1        | 72.8        | 49.9         | 51.7         | 66.7         | 73.8         |
| RGB-TIR-NIR           | 55.9       | 54.9       | 70.8       | 77.6       | 55.1        | 53.3        | 67.3        | 76.2        | 52.2         | 51.3         | 64.1         | 74.3         |
| Baseline (AL)         | 62.9       | 64.5       | 77.4       | 82.7       | 59.9        | 61.1        | 73.9        | 80.9        | 56.2         | 60.4         | 73.0         | 78.6         |
| Baseline (AL) + TPM   | 67.8       | 69.4       | 83.3       | 88.8       | 63.0        | 63.9        | 78.1        | 83.9        | 58.2         | 60.8         | 74.9         | 81.4         |
| Baseline (AL)+CRM+TPM | 72.3       | 76.6       | 84.7       | 89.4       | 69.0        | 73.6        | 81.8        | 84.7        | 60.0         | 61.6         | 76.2         | 82.3         |

Figure 6: Comparison of feature distributions by using tSNE. Different colors represent different identities.

<!-- image -->

Effects of Different Transformer-based Backbones. To verify the generalization of our TOP-ReID, we adopt three different Transformer-based backbones, i.e., ViT-B/16, DeiT-S/16 and T2T-ViT-24. Tab. 5 illustrates the performance comparison. As can be observed, the ViT-B/16 delivers the best results. With more image spectra, different backbones can consistently improve the performance. Our proposed TPM and CRM can improve the performance with different backbones. We believe that the performance can be further improved by using more powerful backbones.

## Visualization Analysis

To clarify the learning ability, we present visual results on the feature distributions and discriminative attention maps.

Multi-spectral Feature Distributions. Fig. 6 illustrates the feature distributions of different models by using tSNE (Van der Maaten and Hinton 2008). In Fig. 6 (a), it represents the direct concatenation of single-spectral features, where each stream is individually trained. It can be observed that the AL setting can effectively align the features of dif- ferent spectra with a better ID consistence. With our TPM, the features of the same ID across different spectra are more concentrated, and the gaps between different IDs are more distinct. Furthermore, with CRM, the feature distribution becomes more compact, and the number of outliers for each ID is reduced. This visualization provides strong evidences for the effectiveness of our proposed methods.

Figure 7: Discriminative attention maps. (a) Input images; (b) Full; (c) M (RGB); (d) M (NIR); (e) M (TIR); (f) M (NIR+TIR); (g) M (RGB+TIR); (h) M (RGB+NIR);

<!-- image -->

Discriminative Attention Maps. As shown in Fig. 7, we utilize Grad-CAM (Selvaraju et al. 2017) to visualize the discriminative attention maps with different image spectra. Obviously, there are discriminative differences between different image spectra. Our model is powerful and can highlight discriminative regions when missing image spectra.

## Conclusion

In this work, we propose a novel feature learning framework based on token permutations for multi-spectral object ReID. Our approach incorporates a Token Permutation Module (TPM) for spatial feature alignment and a Complementary Reconstruction Module (CRM) for reducing the distribution gap across different image spectra. Through the dynamic cooperation between TPM and CRM, it can handle the missing-spectral problem, which is more flexible than previous methods. Extensive experiments on three benchmarks clearly demonstrate the effectiveness of our methods.

## Acknowledgments

This work was supported in part by the National Key Research and Development Program of China (Grant No. 2018AAA0102001), National Natural Science Foundation of China (Grant No. 62101092), Open Project of Anhui Provincial Key Laboratory of Multimodal Cognitive Computation, Anhui University (Grant No. MMC202102) and Fundamental Research Funds for the Central Universities (No.DUT22QN228).

## References

Ba, J. L.; Kiros, J. R.; and Hinton, G. E. 2016. Layer normalization. arXiv preprint arXiv:1607.06450 .

Chang, X.; Hospedales, T. M.; and Xiang, T. 2018. Multilevel factorisation net for person re-identification. In CVPR , 2109-2118.

Chen, G.; Zhang, T.; Lu, J.; and Zhou, J. 2019. Deep meta metric learning. In ICCV , 9547-9556.

Chen, Y.; Xia, S.; Zhao, J.; Zhou, Y.; Niu, Q.; Yao, R.; Zhu, D.; and Liu, D. 2022. ResT-ReID: Transformer block-based residual learning for person re-identification. PRL , 157: 9096.

Deng, J.; Dong, W.; Socher, R.; Li, L.-J.; Li, K.; and FeiFei, L. 2009. Imagenet: A large-scale hierarchical image database. In CVPR , 248-255.

Dosovitskiy, A.; Beyer, L.; Kolesnikov, A.; Weissenborn, D.; Zhai, X.; Unterthiner, T.; Dehghani, M.; Minderer, M.; Heigold, G.; Gelly, S.; et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 .

- Guo, J.; Zhang, X.; Liu, Z.; and Wang, Y. 2022. Generative and attentive fusion for multi-spectral vehicle reidentification. In ICSP , 1565-1572.
- He, Q.; Lu, Z.; Wang, Z.; and Hu, H. 2023. Graph-Based Progressive Fusion Network for Multi-Modality Vehicle ReIdentification. TITS , 1-17.
- He, S.; Luo, H.; Wang, P.; Wang, F.; Li, H.; and Jiang, W. 2021. Transreid: Transformer-based object re-identification. In ICCV , 15013-15022.
- Hermans, A.; Beyer, L.; and Leibe, B. 2017. In defense of the triplet loss for person re-identification. arXiv preprint arXiv:1703.07737 .
- Li, D.; Wei, X.; Hong, X.; and Gong, Y. 2020a. Infraredvisible cross-modal person re-identification with an x modality. In AAAI , volume 34, 4610-4617.

Li, H.; Li, C.; Zhu, X.; Zheng, A.; and Luo, B. 2020b. Multispectral vehicle re-identification: A challenge. In AAAI , volume 34, 11345-11353.

Li, W.; Zhu, X.; and Gong, S. 2018. Harmonious attention network for person re-identification. In CVPR , 2285-2294.

- Liu, H.; Chai, Y.; Tan, X.; Li, D.; and Zhou, X. 2021a. Strong but simple baseline with dual-granularity triplet loss for visible-thermal person re-identification. SPL , 28: 653657.

Liu, X.; Yu, C.; Zhang, P.; and Lu, H. 2023. Deeply Coupled Convolution-Transformer With Spatial-Temporal Complementary Learning for Video-Based Person Re-Identification. TNNLS .

Liu, X.; Zhang, P.; Yu, C.; Lu, H.; and Yang, X. 2021b. Watching You: Global-Guided Reciprocal Learning for Video-Based Person Re-Identification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , 13334-13343.

Luo, H.; Gu, Y.; Liao, X.; Lai, S.; and Jiang, W. 2019. Bag of tricks and a strong baseline for deep person re-identification. In CVPRW , 1487-1495.

- Pan, W.; Huang, L.; Liang, J.; Hong, L.; and Zhu, J. 2023. Progressively Hybrid Transformer for Multi-Modal Vehicle Re-Identification. Sensors , 23(9): 4206.
- Pan, W.; Wu, H.; Zhu, J.; Zeng, H.; and Zhu, X. 2022. Hvit: Hybrid vision transformer for multi-modal vehicle reidentification. In CICAI , 255-267. Springer.
- Qian, X.; Fu, Y.; Jiang, Y.-G.; Xiang, T.; and Xue, X. 2017. Multi-scale deep learning architectures for person reidentification. In ICCV , 5399-5408.
- Qian, X.; Fu, Y.; Xiang, T.; Jiang, Y.-G.; and Xue, X. 2019. Leader-based multi-scale attention deep architecture for person re-identification. TPAMI , 42(2): 371-385.

Rao, Y.; Chen, G.; Lu, J.; and Zhou, J. 2021. Counterfactual attention learning for fine-grained visual categorization and re-identification. In ICCV , 1025-1034.

- Selvaraju, R. R.; Cogswell, M.; Das, A.; Vedantam, R.; Parikh, D.; and Batra, D. 2017. Grad-cam: Visual explanations from deep networks via gradient-based localization. In ICCV , 618-626.
- Sun, Y.; Cheng, C.; Zhang, Y.; Zhang, C.; Zheng, L.; Wang, Z.; and Wei, Y. 2020. Circle loss: A unified perspective of pair similarity optimization. In CVPR , 6398-6407.
- Sun, Y.; Zheng, L.; Yang, Y.; Tian, Q.; and Wang, S. 2018. Beyond part models: Person retrieval with refined part pooling (and a strong convolutional baseline). In ECCV , 480496.
- Szegedy, C.; Vanhoucke, V.; Ioffe, S.; Shlens, J.; and Wojna, Z. 2016. Rethinking the inception architecture for computer vision. In CVPR , 2818-2826.
- Touvron, H.; Cord, M.; Douze, M.; Massa, F.; Sablayrolles, A.; and J´ egou, H. 2021. Training data-efficient image transformers &amp; distillation through attention. In ICML , 1034710357.
- Van der Maaten, L.; and Hinton, G. 2008. Visualizing data using t-SNE. JMLR , 9(11).
- Wang, G.; Yuan, Y.; Chen, X.; Li, J.; and Zhou, X. 2018. Learning discriminative features with multiple granularities for person re-identification. In ACM MM , 274-282.
- Wang, H.; Shen, J.; Liu, Y.; Gao, Y.; and Gavves, E. 2022a. Nformer: Robust person re-identification with neighbor transformer. In CVPR , 7297-7307.

Wang, Z.; Li, C.; Zheng, A.; He, R.; and Tang, J. 2022b. Interact, embed, and enlarge: Boosting modality-specific representations for multi-modal person re-identification. In AAAI , volume 36, 2633-2641.

Ye, M.; Shen, J.; Lin, G.; Xiang, T.; Shao, L.; and Hoi, S. C. 2021. Deep learning for person re-identification: A survey and outlook. TPAMI , 44(6): 2872-2893.

Yuan, L.; Chen, Y.; Wang, T.; Yu, W.; Shi, Y.; Jiang, Z.-H.; Tay, F. E.; Feng, J.; and Yan, S. 2021. Tokens-to-token vit: Training vision transformers from scratch on imagenet. In ICCV , 558-567.

Zhang, G.; Zhang, P.; Qi, J.; and Lu, H. 2021. Hat: Hierarchical aggregation transformers for person re-identification. In ACM MM , 516-525.

Zhang, Y.; and Wang, H. 2023. Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-identification. In CVPR , 2153-2162.

Zhao, J.; Zhao, Y.; Li, J.; Yan, K.; and Tian, Y . 2021. Heterogeneous relational complement for vehicle re-identification. In ICCV , 205-214.

Zheng, A.; He, Z.; Wang, Z.; Li, C.; and Tang, J. 2023. Dynamic Enhancement Network for Partial Multi-modality Person Re-identification. arXiv preprint arXiv:2305.15762 .

Zheng, A.; Wang, Z.; Chen, Z.; Li, C.; and Tang, J. 2021. Robust multi-modality person re-identification. In AAAI , volume 35, 3529-3537.

Zheng, A.; Zhu, X.; Ma, Z.; Li, C.; Tang, J.; and Ma, J. 2022. Multi-spectral vehicle re-identification with crossdirectional consistency network and a high-quality benchmark. arXiv preprint arXiv:2208.00632 .

Zhong, Z.; Zheng, L.; Kang, G.; Li, S.; and Yang, Y. 2020. Random erasing data augmentation. In AAAI , volume 34, 13001-13008.

Zhou, K.; Yang, Y.; Cavallaro, A.; and Xiang, T. 2019. Omni-scale feature learning for person re-identification. In ICCV , 3702-3712.

Zhu, J.-Y.; Park, T.; Isola, P.; and Efros, A. A. 2017. Unpaired image-to-image translation using cycle-consistent adversarial networks. In ICCV , 2223-2232.

Zhu, K.; Guo, H.; Zhang, S.; Wang, Y.; Huang, G.; Qiao, H.; Liu, J.; Wang, J.; and Tang, M. 2021. Aaformer: Auto-aligned transformer for person re-identification. arXiv preprint arXiv:2104.00921 .

## A Introduction for Supplementary Material

In the supplementary material, we provide additional evidence to validate the effectiveness of TOP-ReID. We perform extra ablation experiments across various datasets and present more visualization results. The experimental section includes an introduction to comparative methods, validation of different reconstruction methods of CRM, validation of the TPM's effectiveness with three permutations and other details, further examination of key modules within person and vehicle ReID, and comparative experiments in situations where vehicle spectra are absent. The visualization section includes three parts: the visualization of token alignments, the presentation of Grad-CAM attention maps for both person and vehicle ReID, as well as the showcasing of reconstructed tokens within the CRM module.

## B Experiments

## B.1 Experimental Details

In the main text, we primarily conduct comparisons involving multi-spectral and missing-spectral scenarios on the RGBNT201 person ReID dataset, and multi-spectral comparisons on the RGBNT100 vehicle ReID dataset.

The single-spectral methods we mainly considered include: HACNN (Li, Zhu, and Gong 2018) proposes a novel approach for jointly learning multi-scale attention selection and feature representation. MUDeep (Qian et al. 2019) introduces a multi-scale deep learning layer and a leadershipbased attention mechanism. OSNet (Zhou et al. 2019) employs a unified aggregation gate for dynamically fusing omni-scale features. MLFN (Chang, Hospedales, and Xiang 2018) utilizes a multi-layered structure to extract optimal features at different convolutional depths, which are then integrated for person ReID. CAL (Rao et al. 2021) introduces a counterfactual attention learning method based on causal reasoning, which learns more effective attention mechanisms. PCB (Sun et al. 2018) and MGN (Wang et al. 2018) adapt a stripe-based image division strategy to obtain multi-grained representations. DMML (Chen et al. 2019) presents a meta perspective on metric learning, demonstrating the consistency of softmax and triplet losses in the meta space. Circle Loss (Sun et al. 2020) introduces a novel approach to re-weight similarity scores, emphasizing lessoptimized similarities for more flexible optimization. Luo et al. (Luo et al. 2019) utilize a deep residual network and introduce the BNNeck technique. HRCN (Zhao et al. 2021) proposes the Cross-camera Generalization Measure (CGM) to enhance evaluations. AGW (Ye et al. 2021) incorporates non-local attention mechanisms for fine-grained feature extraction. TransReID (He et al. 2021) introduces the first purely Transformer-based method, fully exploiting relationships between image patches.

The multi-spectral methods we mainly considered include: HAMNet (Li et al. 2020b), designed to effectively fuse various spectral features. PFNet (Zheng et al. 2021), which significantly advances multi-spectral person ReID by learning robust RGB-NIR-TIR features. DENet (Zheng et al. 2023), proposing pixel-level constraints to address the spectral-missing problem. IEEE (Wang et al. 2022b), a set of three innovative learning strategies aimed at enhancing modality-specific representations. GAFNet (Guo et al. 2022), which seamlessly fuses data from multiple sources. GPFNet, introducing an adaptive fusion approach for multispectral features. HViT (Pan et al. 2022) balances modalspecific and modal-shared information. Furthermore, they employ random hybrid augmentation and a feature hybrid mechanism to enhance performance with PHT (Pan et al. 2023). CCNet, which concurrently addresses discrepancies from both spectra and sample aspects. These comparisons serve to validate the effectiveness of the proposed method TOP-ReID across various scenarios and datasets.

## B.2 More Ablation Experiments

Effects of Different Reconstruction Methods. Tab. 6 shows more comparison results with other reconstruction losses (MAE, RMSE). The results clearly show that our CRM always surpasses the pixel-level reconstruction method (DENet). The reason to bridge the distribution gap is that the reconstruction constraints of our CRM promote mutual supervision between different modalities, facilitating the convergence of their feature distributions.

Table 6: mAP Comparison with different reconstructions.

| Level   | Methods    | RGBNT201   | RGBNT201   | RGBNT201   | RGBNT201   |
|---------|------------|------------|------------|------------|------------|
|         |            | Full       | M(N)       | M(T)       | M(N+T)     |
| Pixel   | DENet      | 42.4       | 35.4       | 33.0       | 32.4       |
| Token   | CRM (MSE)  | 72.3       | 64.3       | 51.9       | 34.1       |
| Token   | CRM (MAE)  | 68.9       | 59.2       | 51.3       | 32.6       |
| Token   | CRM (RMSE) | 69.8       | 61.7       | 54.3       | 35.8       |

Evaluation on Three Permutations in TPM. As shown in Tab. 7, a sequence of three consecutive permutations showcases how the model progressively generates more decisive representations. The initial permutation, denoted as the first step, merges each spectrum with its adjacent spectrum to initiate fusion. Following this, the second permutation step facilitates the incorporation of detailed information from all spectra into one another. Ultimately, the third permutation step brings the information back into distinct branches, finalizing the fusion process. This experiment underscores the importance of executing three permutations. This fusion framework, we believe, not only suits multispectral data but can also be effectively applied for efficient information fusion when dealing with other modalities, such as text, sketch, audio, and video.

Table 7: Performance comparison with consecutive permutations of TPM.

| Modules   | Modules   | Modules   | Modules   | RGBNT201   | RGBNT201   | RGBNT201   | RGBNT201   |
|-----------|-----------|-----------|-----------|------------|------------|------------|------------|
|           | TPM 1     | TPM 2     | TPM 3     | mAP        | R-1        | R-5        | R-10       |
| A         | ✓         | ✕         | ✕         | 65.1       | 67.2       | 80.4       | 84.6       |
| B         | ✓         | ✓         | ✕         | 66.5       | 69.3       | 81.1       | 86.6       |
| C         | ✓         | ✓         | ✓         | 67.8       | 69.4       | 83.3       | 88.8       |

Figure 8: Comparison of feature distributions by using tSNE. Different colors represent different identities.

<!-- image -->

The differences and effects of f (R , 3) , f (N , 3) and f (T , 3) . (1) Differences: f (R , 3) , f (N , 3) and f (T , 3) are the class tokens representing the RGB, NIR and TIR features after crossmodal fusion, respectively. They are different. Each of them contains modality-aware information after cross-modal fusion and intra-modal interaction. (2) Effects: They do not contribute equally to the final result. As shown in Tab. 8, f (T , 3) contributes more due to the richer detail retention of TIR images. While f (N , 3) contributes less due to the blurriness of NIR images.

Table 8: Effects of f (R , 3) , f (N , 3) and f (T , 3) .

| Tokens    | RGBNT201   | RGBNT201   | RGBNT201   | RGBNT201   |
|-----------|------------|------------|------------|------------|
|           | mAP        | R-1        | R-5        | R-10       |
| f (R , 3) | 34.1       | 32.2       | 46.1       | 56.1       |
| f (N , 3) | 26.3       | 26.7       | 42.2       | 51.9       |
| f (T , 3) | 35.3       | 35.9       | 53.1       | 60.3       |

Evaluation of Key Modules on RGBNT201. We evaluate the TPM and CRM with BL, which supervises the features from three ViT-B/16 backbones before the concatenation of features. Even without utilizing the ID constraint from AL in the main text, we implicitly achieve the alignment of IDs through TPM, as illustrated in Fig. 8. The feature distributions for the three spectra corresponding to the same ID become more closely aligned. While some features may not receive strong constraints, it still indicates the potential ID-constraining capability of TPM.

Building upon BL, the introduction of TPM yields a noteworthy 3.5% increase in mAP and a 6.1% improvement in rank-1 accuracy. Furthermore, incorporating CRM, with its dense token constraints, results in a mAP improvement of 5.2%. These results demonstrate the effectiveness of TPM and CRM in person ReID.

Evaluation of Key Modules on RGBNT100. In the main text, we specifically conduct ablation experiments focusing on person ReID. However, for a comprehensive evaluation of our proposed method, we also carry out corresponding ablation experiments for vehicle ReID. In these experiments, we introduce two variants: Baseline (BL) and Baseline (AL). BL applies the loss before concatenating features from the three branches, while AL applies the loss after concatenation. Interestingly, in the context of vehicle ReID, introducing AL does not significantly enhance the model's representational capability, unlike in person ReID. Although AL can improve ID correspondence consistency, separate branch training in AL is less effective compared to BL in vehicle ReID. In fact, TPM's features implicitly achieve ID correspondence consistency. Consequently, we uniformly adopt BL to enhance the discriminative representation of each branch in vehicle ReID.

Table 9: Performance comparison of different modules.

| Methods                   | RGBNT100   | RGBNT100   | RGBNT100   | RGBNT100   |
|---------------------------|------------|------------|------------|------------|
|                           | mAP        | R-1        | R-5        | R-10       |
| RGB                       | 50.1       | 71.5       | 74.8       | 76.7       |
| NIR                       | 43.4       | 61.3       | 64.5       | 66.4       |
| TIR                       | 40.9       | 72.1       | 75.0       | 77.3       |
| NIR-TIR                   | 65.0       | 88.3       | 89.8       | 91.1       |
| RGB-NIR                   | 60.4       | 76.8       | 79.1       | 80.9       |
| RGB-TIR                   | 68.1       | 88.7       | 90.3       | 91.2       |
| Baseline (AL)             | 71.6       | 89.4       | 91.0       | 91.8       |
| Baseline (BL)             | 75.2       | 93.9       | 95.2       | 96.2       |
| Baseline (AL) + TPM       | 72.4       | 89.9       | 91.3       | 92.1       |
| Baseline (BL) + TPM       | 78.2       | 95.9       | 97.5       | 98.1       |
| Baseline (AL) + CRM + TPM | 73.7       | 92.2       | 93.6       | 94.1       |
| Baseline (BL)+CRM+TPM     | 81.2       | 96.4       | 96.9       | 97.2       |

Table 10: Performance comparison with BL on RGBNT201.

| Modules   | Modules   | Modules   | Modules   | RGBNT201   | RGBNT201   | RGBNT201   | RGBNT201   |
|-----------|-----------|-----------|-----------|------------|------------|------------|------------|
|           | BL        | TPM       | CRM       | mAP        | R-1        | R-5        | R-10       |
| A         | ✓         | ✕         | ✕         | 55.9       | 54.9       | 70.8       | 77.6       |
| B         | ✓         | ✓         | ✕         | 59.4       | 61.0       | 76.4       | 81.7       |
| C         | ✓         | ✓         | ✓         | 64.6       | 64.6       | 77.4       | 82.4       |

For both AL and BL, the incorporation of TPM and CRM consistently yields substantial improvements. Notably, when each branch demonstrates robust representational capability, the TPM and CRM impart even more pronounced enhancements. This compellingly substantiates the effectiveness of TOP-ReID for vehicle ReID.

Evaluation on Missing-spectral Scenarios. We also assess the effectiveness of our model using the RGBNT100 dataset for missing-spectral vehicle ReID. When a single spectrum is missing, TOP-ReID consistently outperforms DENet, achieving a remarkable mAP increase of over 8%. In scenarios where both NIR and TIR spectra are absent, our model demonstrates substantial improvements, with a mAP increase of 5.3% and a Rank-1 improvement of 3.6%. In comparison to DENet, our model excels in handling a wider range of missing scenarios, affirming the versatility of our framework for object ReID.

Figure 9: Alignment evidences in TPM. Zoom to see better.

<!-- image -->

## C Visualizations C.1 Token alignments in TPM.

In Fig. 9, we present the cosine similarity distribution between the class tokens and the patch tokens from other spectra. As additional evidences, one can find that the cosine similarities are higher after TPM, indicating better alignments.

## C.2 Discriminative Attention Maps.

As shown in Fig. 10, we visualize the attention maps of instances on RGBNT201 and RGBNT100. Panel (b) demonstrates the attention regions of each branch when no spectral is missing. It can be observed that there are certain differences for different spectra, especially for TIR spectrum, which clearly separates the person from the background. Meanwhile, all branch can capture the discriminative regions of person. Panels (c), (d), and (e) represent the effects of using the other two spectra to reconstruct the missing spectral. It can be seen that the model restores discriminative attention regions. Panels (f), (g), and (h) represent the reconstruction of the other two spectra using only the existing spectrum. It is noticeable that the model's reconstruction ability becomes weak, but it is still able to introduce regions that the existing spectral did not focus on. These visualizations illustrate the effectiveness of TOP-ReID. In addition, we provide attention maps for the vehicle inputs. Just like the (i) to (p) cases, which share the same settings as the person inputs, we can observe that different spectra focus on distinct regions of the vehicles. Moreover, the model demonstrates effective reconstruction, particularly when one spectrum is missing. To sum up, these visualizations provide substantial evidence of the effectiveness of TOP-ReID.

## C.3 Reconstructed Tokens in CRM.

We further visualize the comparison between the reconstructed tokens and original tokens in CRM. As shown in Fig. 11, from top to bottom, there are schematic representations of the matrix values for the reconstructed RGB, NIR, and TIR spectrum. To clearly illustrate the visualization, we use the first 8 tokens, including the class token, and the first 16 embedding dimensions for visualization. In terms of numerical distribution, the reconstructed tokens closely approximate the original tokens, especially for the distribution of patch tokens. However, concerning the cls token, although there is some numerical difference, its importance level remains consistent with the original token. Therefore, through the CRM, we achieve effective alignment with original tokens, thereby assisting the missing-spectral scenarios.

Table 11: Experimental results of missing-spectral tasks on RGBNT100. 'M (X)' stands for missing the X image spectra.

| Methods   |          | M(RGB)   | M(RGB)   | M(NIR)   | M(NIR)   | M(TIR)   | M(TIR)   | M(RGB+NIR)   | M(RGB+NIR)   | M(RGB+TIR)   | M(RGB+TIR)   | M(NIR+TIR)   | M(NIR+TIR)   |
|-----------|----------|----------|----------|----------|----------|----------|----------|--------------|--------------|--------------|--------------|--------------|--------------|
|           |          | mAP      | R-1      | mAP      | R-1      | mAP      | R-1      | mAP          | R-1          | mAP          | R-1          | mAP          | R-1          |
| Multi     | DENet    | -        | -        | 62.0     | 85.5     | 56.0     | 80.9     | -            | -            | -            | -            | 50.1         | 74.2         |
| Multi     | TOP-ReID | 70.6     | 90.6     | 77.9     | 94.5     | 64.0     | 81.5     | 42.5         | 69.3         | 45.9         | 65.4         | 55.4         | 77.8         |

Figure 10: Discriminative attention maps. (a) Input person images; (b) Full; (c) M (RGB); (d) M (NIR); (e) M (TIR); (f) M (NIR+TIR); (g) M (RGB+TIR); (h) M (RGB+NIR). (i) to (p), follow the same setting with the vehicle inputs.

<!-- image -->

<!-- image -->

Token Index Token Index

Figure 11: Comparison between the original tokens and the reconstructed tokens.