## Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation

Fanding Li 1 , Xiangyu Li 1 ∗ , Xianghe Su 1 , Xingyu Qiu 1 , Suyu Dong 2 , Wei Wang 3 , Kuanquan Wang 1 , Gongning Luo 1 , Shuo Li 4,5

1 Faculty of Computing, Harbin Institute of Technology, Harbin, China 2 College of Computer and Control Engineering, Northeast Forestry University, Harbin, China 3 Faculty of Computing, Harbin Institute of Technology, Shenzhen, China

4 Department of Computer and Data Science, Case Western Reserve University, Cleveland, Ohio 44106, United States

5 Department of Biomedical Engineering, Case Western Reserve University, Cleveland, Ohio 44106, United States lixiangyu@hit.edu.cn

## Abstract

A simultaneous enhancement of accuracy and diversity of predictions remains a challenge in ambiguous medical image segmentation (AMIS) due to the inherent trade-offs. While truncated diffusion probabilistic models (TDPMs) hold strong potential with a paradigm optimization, existing TDPMs suffer from entangled accuracy and diversity of predictions with insufficient fidelity and plausibility. To address the aforementioned challenges, we propose Ambiguity-aware Truncated Flow Matching (ATFM), which introduces a novel inference paradigm and dedicated model components. Firstly, we propose Data-Hierarchical Inference, a redefinition of AMISspecific inference paradigm, which enhances accuracy and diversity at data-distribution and data-sample level, respectively, for an effective disentanglement. Secondly, Gaussian Truncation Representation (GTR) is introduced to enhance both fidelity of predictions and reliability of truncation distribution, by explicitly modeling it as a Gaussian distribution at 𝑇 trunc instead of using sampling-based approximations. Thirdly, Segmentation Flow Matching (SFM) is proposed to enhance the plausibility of diverse predictions by extending semantic-aware flow transformation in Flow Matching (FM). Comprehensive evaluations on LIDC and ISIC3 datasets demonstrate that ATFM outperforms SOTA methods and simultaneously achieves a more efficient inference. ATFM improves GED and HM-IoU by up to 12% and 7 . 3% compared to advanced methods.

## Code -

https://github.com/PerceptionComputingLab/ATFM Extended version -This is the extended version of AAAI paper with full appendixes

## Introduction

Generating a series of predictions with high accuracy and diversity to estimate the distribution of annotation space is of significant importance in ambiguous medical image segmentation (AMIS) (Chavhan et al. 2008; Hong et al. 2021). High diversity reflects the inherent ambiguity present in medical images and high accuracy is critical for supporting dependable clinical decision-making, making both essential components of a reliable AMIS framework (Baumgartner et al. 2019; Zhang et al. 2022).

∗ Corresponding Author

Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

Figure 1: (a) Traditional TDPMs face the challenges of low fidelity and plausibility of predictions by improving diversity at the expense of accuracy. (b) The proposed ATFM enhances fidelity and plausibility of predictions by assigning distinct inference goals into two stages.

<!-- image -->

However, simultaneously improving prediction accuracy and diversity remains challenging in AMIS due to the inherent trade-off between these objectives in existing methods. Stochastic approaches (Baumgartner et al. 2019; Rahman et al. 2023) enhance diversity at the expense of the more important accuracy, yielding low-confidence diagnoses. Zhang et al. (Zhang et al. 2022) are able to regulate this trade-off but cannot enhance both properties simultaneously. Multi-rateraware techniques (Zepf et al. 2023; Ji et al. 2021) improve accuracy and diversity by modeling annotators' labeling styles, yet their annotator-centric design inherently suppresses lowfrequency modes, degrading segmentation quality. Broader application of these methods is still constrained by the inherent trade-off between entangled accuracy and diversity among predictions.

Truncated Diffusion Probabilistic Models (TDPMs) (Zheng et al. 2022) have shown great potential in simul- taneously improving accuracy and diversity thanks to the inference paradigm shift. TDPMs are proposed to inference within fewer steps by leveraging an auxiliary network to model the distribution at a predefined truncation point, shifting the original inference for acceleration. TDPMs are widely used in multiple areas such as high-resolution MRSI generating (Dong et al. 2025), autonomous driving (Liao et al. 2025), remote sensing (He et al. 2023), demonstrating superiority in both effectiveness and efficiency of inference.

However, as Fig. 1(a), directly applying TDPMs to achieve synergistic optimization of prediction accuracy and diversity in AMIS remains challenging due to: (1) The uniform inference objective across all stages in conventional TDPMs inherently constrains their capacity for simultaneous accuracy and diversity improvement. Vanilla TDPMs adopt a two-stage inference process with one objective mainly for acceleration. However, the deterministic and ambiguous components remain entangled throughout the inference process, making it difficult to achieve a simultaneous enhancement of both accuracy and diversity. (2) Sub-optimal approximation of the underlying distribution at the truncation point fundamentally compromises prediction fidelity. Distribution at 𝑇 trunc is estimated by drawing samples from adversarial networks rather than explicitly modeling in vanilla TDPMs, leading to degraded fidelity caused by inconsistent predictions and omission of low-frequency modes (clinically plausible yet rare). (3) The absence of semantic guidance following truncation in conventional TDPMs adversely affects the plausibility of generated predictions. Vanilla TDPMs rely on a vanilla diffusion process guided solely by generation quality after 𝑇 trunc. In AMIS tasks, this final stage lacks explicit semantic constraints for segmentation, which, although enhancing diversity, significantly compromises the more important accuracy and plausibility.

In this work, we propose the Ambiguity-aware Truncated Flow Matching (ATFM) to achieve a synergistic enhancement of accuracy and diversity of predictions in AMIS tasks, supported by three designs (as Fig. 1(b)): (1) We propose Data-Hierarchical Inference, an innovative AMIS-specific inference paradigm redefinition inspired by TDPMs, where stochasticity during diffusion is marginalized for an effective disentanglement of accuracy and diversity. Specifically, ATFMperforms truncated steps at the data distribution level to prioritize accuracy with diversity marginalized, whereas the final diffusion stage operates at the data sample level to enhance diversity without sacrificing accuracy. (2) We propose Gaussian Truncation Representation (GTR), which explicitly models the Gaussian latent distribution at truncation point to enhance prediction fidelity. While traditional TDPMs approximate the implicit distribution via adversarial sampling, GTR encodes image-level semantic features to logit distribution, thereby directly modeling the reliable distribution at the truncation point. This design preserves low-frequency modes and improves consistency, leading to predictions with higher fidelity in AMIS tasks. (3) We propose Segmentation Flow Matching (SFM), which introduces semantic-aware flow transformation to increase diversity and enhance plausibility simultaneously. While traditional TDPMs focus solely on generative quality after truncated steps, the proposed SFM

leverages Flow Matching (FM) to overcome Gaussian limitations that disrupt fine-grained predictions and incorporates explicit semantic consistency modeling to ensure the plausibility of segmentation predictions in AMIS tasks.

In summary, our main contributions are as follows:

- We propose ATFM with Data-Hierarchical Inference redefining a more suitable inference paradigm for AMIS for the first time, where Data-Hierarchical Inference effectively decouples accuracy and diversity by marginalizing the stochasticity during diffusion process, thereby enabling a simultaneous improvement of both.
- The proposed GTR in ATFM pioneers explicitly modeling a Gaussian distribution at truncation point, which effectively preserves low-frequency modes and ensures sample consistency, thereby significantly improving prediction fidelity to ground truth distribution and reliability of the distribution at truncation point.
- The proposed SFM in ATFM pioneers semantic-aware flow transformation by modeling semantic consistency at each timestep, thereby enhancing plausibility while emphasizing sample-wise diversity, and is built upon the Flow Matching (FM) process that inherently avoids disturbance from Gaussian constraints.
- Acomprehensive evaluation on the LIDC and ISIC3 subset datasets demonstrates that proposed ATFM significantly improves the SOTA methods and simultaneously achieves a more efficient inference.

## Related Work

## Ambiguous Medical Image Segmentation

Existing AMIS approaches fall into four main paradigms: model ensemble, multi-head frameworks, conditional variational autoencoder (cVAE)-based models, and diffusionbased models. All face a fundamental trade-off between prediction accuracy and sample diversity (Zhang et al. 2022).

Modelensemble(Monteiroetal.2020;Lipmanetal.2023) and multi-head models (Ho, Jain, and Abbeel 2020; Kohl et al. 2018) generate multiple predictions via diverse architectures or output heads but do not change the original inference process. Consequently, prediction quality heavily depends on model selection, limiting simultaneous improvement of accuracy and diversity.

CVAE-based (Baumgartner et al. 2019; Kohl et al. 2019) and diffusion-based methods (Rahman et al. 2023; Zbinden et al. 2023) inject stochasticity to enhance diversity, yet both follow a one-stage inference paradigm that entangles accuracy and diversity optimization. This leads to inherent conflicts where gains in diversity often reduce accuracy.

To address the challenges, in the proposed ATFM, we introduce Data-Hierarchical Inference to redefine the inference paradigm for AMIS inspired by TDPMs. Specifically, a principled decoupling of the two objectives is achieved across different data hierarchies, where accuracy is enhanced at distributional level and diversity is promoted at sample level.

## Truncated Diffusion Probabilistic Models

Truncated Diffusion Probabilistic Models (TDPMs) accelerate inference by truncating the diffusion process at 𝑇 trunc ≪

Figure 2: The proposed ATFM addresses the challenge of a synergistic optimization by boosting accuracy at distribution level and diversity at sample level within Data-Hierarchical Inference (Sec. 3.1) while enhancing fidelity and plausibility with GTR (Sec. 3.2) and SFM (Sec. 3.3), respectively.

<!-- image -->

Figure 3: Data-Hierarchical Inference disentangles accuracy and diversity by marginalizing stochasticity during diffusion with a data-distribution level supervision.

<!-- image -->

𝑇 , splitting inference into estimating the distribution at 𝑇 trunc and reverse diffusion thereafter. Existing TDPMs approximate the distribution at 𝑇 trunc via adversarial sampling or perturbations (Zheng et al. 2022; Dong et al. 2025), improving speed but without redefining the inference paradigm. Consequently, they lack explicit modeling and semantic supervision at 𝑇 trunc, limiting performance on AMIS tasks.

The proposed ATFM addresses these gaps by introducing Data-Hierarchical Inference, which marginalizes stochasticity before truncation to disentangle and enhance both accuracy and diversity. Consequently, ATFM explicitly models a Gaussian distribution at 𝑇 trunc for high-fidelity sampling and applies semantic supervision after truncation to improve plausibility, making ATFM a tailored solution for AMIS.

## Methods

The proposed ATFM (Fig. 2) redefines the inference paradigm for disentanglement and simultaneous enhance- ment of accuracy and diversity by marginalizing stochasticity during truncation, forming an AMIS-specific defining solution. ( Data-Hierarchical Inference, Sec. 3.1 ). Specifically, ATFM firstly improves prediction fidelity and truncation distribution reliability by explicitly modeling the Gaussian distribution at truncation point, thereby preserving lowfrequency modes often missed by sampling-based approximation and ensuring consistency across predictions ( GTR, Sec. 3.2 ). Secondly, ATFM extends semantic-aware flow transformation by modeling semantic consistency among labels, predictions, and intermediate states during flow matching (FM), thereby enhancing prediction plausibility while promoting diversity. Additionally, Gaussian constraints are avoided for fine-grained predictions by FM ( SFM, Sec. 3.3 ).

## Data-Hierarchical Inference Enables Principled Disentanglement and Joint Enhancement of Accuracy and Diversity

The proposed Data-Hierarchical Inference forms disentanglement and simultaneous enhancement of prediction accuracy and diversity by marginalizing the stochasticity during diffusion, which introduces a redefinition of AMIS-specific inference paradigm. While preserving the efficiency gains from truncation, Data-Hierarchical Inference introduces a principled separation between distribution-level and samplelevel inference, dedicated to optimizing overall accuracy and prediction diversity, respectively.

Specifically, the overall inference process firstly focuses on improving accuracy by supervising an accurate explicit distribution at 𝑇 trunc as eq. 1, and then enhances diversity by generating varied samples from this distribution after 𝑇 trunc as eq. 2, ensuring each prediction remains both distinct and consistent with the underlying semantics.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

sample level for diversity

where 𝑃 denotes the explicit intermediate distribution estimated by our Data-Hierarchical Inference at 𝑇 trunc, and 𝑄 represents the corresponding distribution derived from ground truths. 𝑠 𝑖 denote latent samples, pred 𝑖 are the corresponding predictions, and gt 𝑖 represent the ground truths.

As illustrated in Fig. 3 and eq. 1, Data-Hierarchical Inference fundamentally redefines the inference paradigm by marginalizing stochasticity during truncation to achieve a principled disentanglement of accuracy and diversity. This paradigm optimization enables an improvement in accuracy without compromising diversity. Sample-level diffusion builds upon the globally aligned explicit distribution at 𝑇 trunc , leading to a unified and robust enhancement of both prediction fidelity and diversity.

Data-Hierarchical Inference inherently addresses the core challenges of AMIS by reconciling high prediction accuracy with plausible diversity. Through principled disentanglement of accuracy and diversity, Data-Hierarchical Inference establishes a robust and efficient solution that redefines the inference paradigm and application of TDPMs in AMIS.

Summarized Advantage : Data-Hierarchical Inference, pioneers disentangling accuracy and diversity for better meeting with requirements of AMIS, by marginalizing diversity and promoting it through controlled sampling within an inference paradigm redefinition in two consecutive stages.

## Gaussian Truncation Representation Improves Fidelity via Explicit Gaussian Modeling

The proposed GTR models the explicit distribution at 𝑇 trunc for prediction fidelity by parameterizing it as a Gaussian distribution. This explicit modeling, supervised to ensure the overall accuracy and serving as the truncation step, enhances fidelity of predictions and reliability of distribution at 𝑇 trunc which preserves low-frequency modes and improves sample consistency compared to sampling-based approximations.

Theorem 1. The marginal distribution of the latent variable at any diffusion timestep 𝜏 can be parameterized as

<!-- formula-not-decoded -->

Theorem2. For any Gaussian distribution N( 𝜇 0 , Σ 0 ) , there exists a specific timestep 𝜏 ∗ at which the diffusion process produces an identical distribution.

The proof of Theorems 1 and 2 is provided in the appendix of extended version.

According to Theorems 1 and 2, together with the controllability of diffusion trajectories between adjacent timesteps (Qiu et al. 2025), it can be concluded that arbitrary Gaussian distributions are admissible as distribution within the diffusion framework. Hence, the Gaussian distribution on the logit map modeled as eq. 4 following the formulation in Theorem 1 is selected as the truncation distribution, as it most closely approximates the predictions and enables supervision to achieve optimal accuracy and reliability.

<!-- formula-not-decoded -->

where 𝑓 𝜃 , 𝑔 𝜙 , and ℎ 𝜓 denote the segmentation backbone and separate convolutional layers for estimating the mean and covariance, respectively, and 𝑋 𝑇 trunc denotes the Gaussian distribution at the truncation point.

The 𝐿 Prior of GTR explicitly supervises accuracy between the truncation distribution and ground truths, defined as:

<!-- formula-not-decoded -->

where 𝑋 𝑖 trunc is the 𝑖 𝑡 ℎ sample from 𝑋 𝑇 trunc , 𝑌 is the ground truth, and 𝑀 is the number of Monte Carlo samples. Minimizing the negative log-likelihood in 𝐿 Prior optimizes the explicit Gaussian 𝑋 0 for accuracy and fidelity. The network is then frozen for subsequent inference.

Summarized Advantage : GTR, pioneers explicit Gaussian distribution modeling at the truncation point for enhancing prediction fidelity and truncation distribution reliability via mean and covariance parameterization and estimation.

## Segmentation Flow Matching Enhances Plausibility via Semantic Consistency Modeling

The proposed SFM extends semantic-aware flow transformation for plausibility by modeling semantic consistency among labels, predictions, and intermediate states at each timestep after 𝑇 trunc during FM training. It incorporates a Semanticaware Transformation Network (ST-Net) at each timestep to ensure that flow transformation proceeds under semantic constraints. SFM aligns well with the ambiguity-resolving requirements of AMIS tasks by enhancing plausibility while promoting diversity. Moreover, by employing Flow Matching instead of DDPM, SFM avoids the Gaussian constraints that introduces disturbances in fine-grained predictions.

Algorithm 1 is the summarized training procedure of SFM.

Computing the intermediate prediction corresponding to timestep t (line 2 and 3 in Algorithm 1): The flow transformation follows an Optimal Transformation (OT) schedule (Lu and Song 2024), representing the shortest path between source and target distributions. Under the OT framework, the diffusion trajectory in the latent space forms a line segment. Therefore, we perform analytic geometry in the latent space: the intermediate state 𝑋 𝑡 at timestep 𝑡 is computed by linear interpolation between the source endpoint 𝑋 𝑇 trunc and the

Figure 4: Comparative qualitative results on LIDC dataset among ground truths, two advanced methods and the proposed ATFM demonstrate both better alignment with ground truths and higher per-sample accuracy.

<!-- image -->

Table 1. Quantitative results on LIDC dataset show the superior performance of ATFM. Bold represents the best per column. Arrows indicate the increasing performance of the metrics.

| Methods                                                                                                                                                                                                                             | GED 16 ↓                                                                                                                                          | GED 32 ↓                                                                                                                                          | GED 100 ↓                                                                                                                                         | HM-IoU 32 ↑                                                                                                                                                       | MDM 32 ↑                                                                                                                                          |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| Prob. Unet(Kohl et al. 2018) PHiSeg(Baumgartner et al. 2019) SSN(Monteiro et al. 2020) MoSE(Gao et al. 2023) P 2 SAM(Li et al. 2024) CIMD(Rahman et al. 2023) AB(Chen, Zhang, and Hinton 2023) CCDM(Zbinden et al. 2023) ATFM(Ours) | 0 . 310 ± 0 . 010 0 . 262 ± 0 . 000 0 . 259 ± 0 . 000 0 . 218 ± 0 . 003 0 . 208 ± 0 . 000 - 0 . 213 ± 0 . 001 0 . 212 ± 0 . 001 0 . 206 ± 0 . 002 | 0 . 303 ± 0 . 010 0 . 247 ± 0 . 000 0 . 243 ± 0 . 010 0 . 195 ± 0 . 002 0 . 206 ± 0 . 000 - 0 . 196 ± 0 . 002 0 . 194 ± 0 . 001 0 . 188 ± 0 . 001 | 0 . 252 ± 0 . 004 0 . 224 ± 0 . 004 0 . 225 ± 0 . 002 0 . 189 ± 0 . 002 - 0 . 321 ± 0 . 000 0 . 193 ± 0 . 002 0 . 183 ± 0 . 002 0 . 162 ± 0 . 002 | 0 . 548 ± 0 . 000 0 . 595 ± 0 . 000 0 . 550 ± 0 . 010 0 . 624 ± 0 . 004 0 . 627 ± 0 . 000 0 . 592 ± 0 . 002 0 . 619 ± 0 . 001 0 . 631 ± 0 . 002 0 . 667 ± 0 . 002 | 0 . 681 ± 0 . 020 0 . 704 ± 0 . 020 - 0 . 767 ± 0 . 004 0 . 939 ± 0 . 000 0 . 915 ± 0 . 004 0 . 792 ± 0 . 002 0 . 790 ± 0 . 003 0 . 948 ± 0 . 001 |

## Algorithm 1: Training Procedure for proposed SFM

Require: Source distribution 𝑋 𝑇 trunc , Target distribution 𝑋 1, Output of ST-net 𝑔 𝜃 ( 𝑋 𝑡 ) at timestep t, ground truths 𝑦 𝑖

- 1: repeat
- 2: 𝑋 𝑡 = 𝑡 × 𝑋 1 + ( 1 -𝑡 ) × 𝑋 𝑇 trunc
- 3: 𝑥 1 𝑡 = 𝑥 𝑡 + 𝑔 𝜃 ( 𝑋 𝑡 ) × ( 1 -𝑡 ) ( calculation of prediction corresponding to timestep t )
- 4: 𝐿 𝑖 Dice = 1 -𝐷𝑖𝑐𝑒 ( 𝑥 1 𝑡 , 𝑦 𝑖 ) , 𝑖 = 1 , 2 , . . . , 𝑁
- 5: 𝐿 SF = 𝐿 FM + 1 𝑁 ˝ 𝑁 𝑖 = 1 𝛼 × 𝐿 𝑖 Dice ( semantic consistency modeling )
- 6: 𝜃 ← 𝜃 -𝜂 ∇ 𝜃 𝐿 SF ( gradient update )
- 7: until | |∇ 𝜃 𝐿 SF | | &lt; 𝛿 ( convergence )

target endpoint 𝑋 1. Then, using the direction vector of the segment 𝑔 𝑡 ( 𝑋 𝑡 ) and the position of 𝑋 𝑡 , a predicted result 𝑥 1 𝑡 is derived by projecting along the diffusion trajectory starting from 𝑥 𝑡 .

## Semanticconsistencymodeling(line4to6inAlgorithm

1): By computing the Dice loss between the predicted result 𝑥 1 𝑡 and all ground truth annotations, semantic consistency at timestep 𝑡 can be explicitly modeled. This supervision acts as an auxiliary constraint to the Flow Matching loss, encouraging the transformation to preserve plausibility and consistency throughout the diffusion process for diversity.

The aforementioned SFM training process not only en- sures accurate flow transformation, but also explicitly models the semantic consistency among the state, predicted result and ground truths at each timestep 𝑡 . This dual-objective optimization enhances the semantic plausibility of predictions, and simultaneously capturing diverse sample-level variations via flow matching, positioning SFM as an indispensable module of the proposed ATFM framework for AMIS.

Summarized Advantage : SFM, firstly extends semanticaware flow transformation for prediction plausibility while enhancing diversity, by modeling segmentation semantic consistency among ground truths, predictions and intermediate states at each timestep.

## Experiments

## Experimental Setup

Datasets. In our experiments, we applied two public datasets for ambiguous medical image segmentation: LIDCIDRI (Kalpathy-Cramer et al. 2016) and ISIC3 subset (Codella et al. 2019; Zepf et al. 2023). The LIDC-IDRI dataset consists of lung CT scans with multiple expertannotated lesion segmentations, highlighting diagnostic ambiguities. Following preprocessing as described in (Kohl et al. 2018, 2019), the dataset includes 15,096 slices, each with four corresponding segmentation labels. The ISIC3 dataset provides dermoscopic images of skin lesions with annotations for lesion boundaries. Using the preprocessed ISIC3 subset from (Zepf et al. 2023), we work with 300 images, each featuring exactly three annotations.

<!-- image -->

Figure 5: Comparative qualitative results on ISIC3 subset dataset among ground truths, two advanced methods and the proposed ATFM demonstrate both better alignment with ground truths and higher per-sample accuracy.

Table 2. Quantitative results on ISIC3 subset dataset show the superiority of ATFM. Bold represents the best per column. Arrows indicate the increasing performance of the metrics.

| Methods                                                                                                                  | GED 16 ↓                                                                                  | GED 32 ↓                                                                                  | GED 100 ↓                                                                                 | HM-IoU 32 ↑                                                                               | MDM 32 ↑                                                                                  |
|--------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Prob. Unet(Kohl et al. 2018) SSN(Monteiro et al. 2020) c-Prob. Unet(Zepf et al. 2023) c-SSN(Zepf et al. 2023) ATFM(Ours) | 0 . 202 ± 0 . 003 0 . 197 ± 0 . 007 0 . 208 ± 0 . 004 0 . 200 ± 0 . 007 0 . 183 ± 0 . 001 | 0 . 187 ± 0 . 003 0 . 181 ± 0 . 004 0 . 202 ± 0 . 005 0 . 195 ± 0 . 007 0 . 152 ± 0 . 002 | 0 . 171 ± 0 . 002 0 . 167 ± 0 . 002 0 . 179 ± 0 . 002 0 . 177 ± 0 . 001 0 . 147 ± 0 . 003 | 0 . 697 ± 0 . 005 0 . 700 ± 0 . 004 0 . 719 ± 0 . 004 0 . 725 ± 0 . 004 0 . 732 ± 0 . 003 | 0 . 927 ± 0 . 003 0 . 939 ± 0 . 001 0 . 925 ± 0 . 004 0 . 931 ± 0 . 003 0 . 942 ± 0 . 002 |

Table 3. Time comparison for generating 100 samples on the LIDC dataset for diffusion-based methods demonstrates the superior time efficiency of proposed ATFM.

| Models            | Inference Steps and Time 100                                                                        |
|-------------------|-----------------------------------------------------------------------------------------------------|
| CIMD AB CCDM ATFM | 𝑇 = 100 steps ≈ 420s 𝑇 = 250 steps ≈ 1050s 𝑇 = 250 steps ≈ 1100s GTR + ( 𝑇 Trunc = 25 steps) ≈ 113s |

Implementation Details. All training and inference procedures are conducted on a single RTX 3090 GPU with 24GB memory. SFM in ATFM is trained for 200 epochs on LIDC with GTR pretrained for 1000 epochs, and for 120 epochs on ISIC3 with a 400-epoch GTR. We set 𝜆 = 10 -3 (i.e. 𝑇 = 1000) with a linear schedule for all experiments. Both GTR and ST-Net of the SFM are optimized using an Adam optimizer (Kingma and Ba 2014) with a learning rate of 10 -4 . Hyper-parameter 𝑀 is set to 20 following (Zepf et al. 2023) and 𝛼 is set to 10 -3 and 10 -4 for LIDC and ISIC3 respectively according to hyper-parameter studies.

Evaluation Metrics. Three metrics are utilized for comprehensive evaluation from three aspects: For segmentation distribution, we utilize the Generalised Energy Distance (GED) (Bellemare et al. 2017) to evaluate the align- ment among distribution of predictions and ground-truths. For sample fidelity, we utilize the Hungarian-Matching Intersection-over-Union (HM-IoU) (Gao et al. 2023) to provide an accurate representation of the performance on segmentation across all predictions. For individual segmentation accuracy, we utilize the Maximum Dice Matching (MDM) (Rahmanet al. 2023) to evaluate the best Dice scores between each prediction result and each ground truth. We denote the metrics with subscript n to represent the metrics calculated using n samples. Results are reported as mean ± standard deviation over five independent runs.

Table 4. Ablation Study on LIDC dataset shows the validity of all components in proposed ATFM.

| Models                                       | GED 100 ↓                                                                                 | HM-IoU 32 ↑                                                                               |
|----------------------------------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Act. GTR SFM w/o 𝐿 𝑆𝐹 SFM ATFM w/o 𝐿 𝑆𝐹 ATFM | 0 . 230 ± 0 . 001 0 . 185 ± 0 . 002 0 . 176 ± 0 . 002 0 . 249 ± 0 . 001 0 . 162 ± 0 . 002 | 0 . 550 ± 0 . 010 0 . 624 ± 0 . 002 0 . 631 ± 0 . 002 0 . 597 ± 0 . 003 0 . 667 ± 0 . 002 |

## Experimental Results

## Comparison with SOTAs for Performance Superiority.

Quantitative Evaluation. Table 1 and Table 2 report the quantitative results on the LIDC and ISIC3 datasets, respectively. Across all key metrics-GED (for diversity), HM-IoU (for sample fidelity), and MDM (for individual accuracy)-ATFM consistently outperforms state-of-the-art methods. Notably, ATFM achieves an 11.5% improvement on GED100 in LIDC, along with consistent gains in GED16 and GED32 compared with the runner-up method, highlighting its ability to better capture the underlying segmentation label distribution. A minimum of 7.3% improvement in HM-IoU32 further demonstrates the superior fidelity of the generated samples. ATFM also leads in the MDM32 metric, validating its accuracy at the individual prediction level. Similar trends are observed on ISIC3, where ATFM outperforms the runnerup method by 12% in GED while also achieving top-tier results in HM-IoU and MDM, demonstrating comprehensive improvements in diversity, fidelity, and accuracy. These results showcase the effectiveness of the Data-Hierarchical Inference framework of ATFM, which explicitly models intermediate distributions and enforces semantic-aware flow transformations to enhance accuracy and diversity in AMIS.

Figure 6: Comprehensive analysis through ablation and hyper-parameter studies demonstrates the effectiveness of innovations and training configurations in proposed ATFM.

<!-- image -->

Qualitative Results. Fig. 4 and Fig. 5 further compare visual results among advanced methods and proposed ATFM for LIDC dataset and ISIC3 dataset, respectively. Predictions from ATFM more faithfully reflect the range of plausible annotations and better preserve fine-grained structures, indicating its superior alignment with ambiguous and detailed ground truths, which is an essential and necessary requirement in AMIS tasks.

Inference Efficiency. Table 3 reports the total inference time for generating 100 plausible predictions. ATFM retains the inference efficiency advantage of truncated diffusion models, requiring only 25 diffusion steps and an estimation of truncation point (approx. 113s). Compared to other diffusionbased methods, it achieves both superior segmentation performance and significantly faster inference, highlighting its practicality and scalability in AMIS applications.

Ablation Studies Demonstrate Effectiveness of Innovations. Table 4 and Fig. 6(a) show the conducted ablation study on both datasets evaluating five model variants: GTR with activation layers (A), SFM with and without 𝐿 SF (B and C), and ATFM with and without 𝐿 SF (D and E). ATFM outperformed both Act. GTR and SFM by a minimum of 10% and 6% on both metrics, highlighting the benefit of DataHierarchical Inference and the high effectiveness and fidelity of proposed ATFM provided by GTR. The performance gap of an average of 11% between models with and without 𝐿 SF emphasizes the importance of semantics consistency modeling, underscoring the role of 𝐿 FM and SFM in preserving plausibility when enhancing diversity.

Hyper-parameter Studies Prove Effectiveness of Training Configurations. Fig. 6(b) and 6(c) illustrate the effects of inference step count in SFM and 𝛼 in 𝐿 SF in SFM, respectively. Five values between 1 and 50 are set for inference steps based on the property of Euler Sampler (Song and Ermon 2020; Song et al. 2020). Optimal performance was achieved within 25 steps, offering a good balance between performance and efficiency. For 𝛼 , values between 0 and 5 × 10 -3 were tested. Small 𝛼 values limits 𝐿 SF's impact, while 𝛼 with too large values diminishes the effect of 𝐿 FM. Values of 𝛼 striking a balance were chosen as the final setting.

## Conclusion

In this work, we proposed Ambiguity-aware Truncated Flow Matching (ATFM), addressing the challenge of jointly improving prediction accuracy and diversity via a novel inference paradigm and dedicated model components, tailored to the demands of AMIS tasks. Specifically, we firstly proposed Data-Hierarchical Inference, redefining a novel inference paradigm that disentangles prediction accuracy and diversity, which supervises a distribution for accuracy at 𝑇 trunc by marginalizing stochasticity and promoting diversity through controlled sampling in the following timesteps. Onthis foundation, we designed two key modules for ATFM: GTR, which explicitly models the Gaussian distribution at truncation point to ensure prediction fidelity and truncation distribution reliability for overall accuracy; and SFM, which extends semantic-aware flow transformation to model semantic consistency across predictions, annotations and intermediate states for enhancing prediction plausibility while promoting diversity. Experimental results on two public datasets showed that the proposed ATFM outperforms SOTA methods across all metrics and offers a more efficient inference process simultaneously. ATFM offers a versatile and reliable solution for AMIS across a broader range of scenarios through multifaceted analysis and outstanding performance.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China under Grants 62501195, 62272135, 62372135 and 82527807, and the Key Research &amp; Development Program of Heilongjiang Province under Grants 2023ZX01A08 and 2024ZX12C23, and the Natural Science Foundation of Heilongjiang Province under Grant LH2024F019.

## Appendix 1: Discussion of the Underlying Mechanism

Figure 7: The comparison of inference paradigms among DPMs, TDPMs, and ATFM (1) reveals that the inference paradigm shift is the underlying mechanism which makes TDPMs potential for ambiguous medical image segmentation (AMIS), and (2) highlights the task-specific improvements introduced by ATFM for the AMIS scenario.

<!-- image -->

## Appendix 2: Potential Underlying Mechanism of Truncated Diffusion Probabilistic Models

Truncated Diffusion Probabilistic Models (TDPMs) demonstrate their potential through the introduction of an inference paradigm shift-a concept that optimizes the inference trajectory between start point and a predefined truncation timestep 𝑇 trunc of diffusion models. Specifically, TDPMs retain the original starting point (random noise) and truncation point ( 𝑋 𝑇 trunc ) of the diffusion process, while strategically replacing the intermediate inference steps between 𝑡 = 0 and 𝑡 = 𝑇 trunc with a learned truncation module (usually adversarial sampling). This substitution constitutes a paradigm shift, proving that the intermediate reasoning path of diffusion models is not rigid but can be restructured. ( As the comparison illustrated in green boxes in Fig. 7(a) and Fig. 7(b) )

Although the primary goal of TDPMs is to accelerate inference, their architecture implicitly reveals a structural flexibility in diffusion-based generative models. By truncating and replacing part of the denoising trajectory, TDPMs validate the feasibility of modeling this process differently. This foundational insight is particularly valuable for ambiguous medical image segmentation (AMIS), where jointly improving prediction accuracy and diversity remains a key challenge. Thus, TDPMs not only serve as a faster alternative, but also inspire a reconsideration of how inference is formulated in diffusion frameworks especially for AMIS tasks.

## Appendix 3: Superior Mechanism of the proposed Ambiguity-aware Truncated Flow Matching for AMIS Inspired by TDPMs

Inspired by the inference paradigm shift introduced by TDPMs, the proposed ATFM redefines an AMIS-specific inference paradigm. ATFM further reformulates not only the overall inference paradigm, but the detailed truncation process and the remaining diffusion steps as well to address the inherent challenges of AMIS. ( as the improvement illustrated in Fig. 7(b) and Fig. 7(c) )

Firstly, the overall inference paradigm is redefined through the proposed Data-Hierarchical Inference, which assigns distinct objectives to different stages. The truncation step focuses on enhancing overall accuracy at the distribution level without compromising diversity, while the subsequent diffusion stage promotes sample-level diversity based on the high-fidelity distribution obtained earlier. This overall design enables a principled disentanglement and simultaneous improvement of both accuracy and diversity. ( overall comparison between Fig. 7(b) and Fig. 7(c) )

Secondly, the truncation stage is redefined via the proposed Gaussian Truncation Representation (GTR), which explicitly models the truncation point as a learnable Gaussian distribution with parameterized mean and covariance. This replaces the sampling-based approximation in TDPMs with a stable and semantically meaningful representation. As a result, it enhances prediction fidelity and provides a reliable initialization for subsequent inference. ( comparison illustrated in green boxes between Fig. 7(b) and Fig. 7(c) )

Thirdly, the post-truncation diffusion stage is redefined through Segmentation Flow Matching (SFM), which supervises semantic consistency between intermediate states, predictions, and ground truths across timesteps. This transforms the diffusion process from unconstrained denoising into a semantics-aware transformation. FM further avoids disturbance from Gaussian constraints. SFM ensures prediction plausibility while capturing diverse sample-level variations in a structured manner. ( comparison illustrated in purple boxes between Fig. 7(b) and Fig. 7(c) )

Fig. 8 further analyzes the three main limitations of vanilla TDPMsin the context of AMIS requirements, along with the corresponding solutions introduced by ATFM, which fundamentally explains the reason behind the differences between the two inference paradigms and highlights the advantages of ATFM. Specifically, these limitations relate to the overall inference paradigm, the truncation process, and the post-truncation diffusion stage. In response, ATFM addresses these issues through the proposed Data-Hierarchical Inference, Gaussian Truncation Representation, and Semantic Flow Matching, respectively.

Figure 8: Limitations of vanilla TDPMs and the solutions offered by the proposed ATFM. From R1 to R3, they are the necessary requirements of performing reliable AMIS. From (a) to (c), they are the limitations of vanilla TDPMs, and from (d) to (f), they are the solutions and advantages of ATFM, respectively. (a) Vanilla TDPMs can't achieve simultaneous enhancement of accuracy and diversity due to the inherent trade-off. (b) Vanilla TDPMs lack prediction fidelity due to sub-optimal sampling-based truncation distribution approximation. (c) Vanilla TDPMs lack prediction plausibility due to the absence of semantic guidance. (d) Data-Hierarchical Inference in proposed ATFM introduces simultaneous enhancement of accuracy and diversity by a principled disentanglement. (e) GTR in proposed ATFM achieves high prediction fidelity via explicit Gaussian parameterization and modeling. (f) SFM in proposed ATFM achieves high prediction plausibility from semantic modeling in SFM.

<!-- image -->

## Appendix 4: Proof for Theorems

Theorem 1: The marginal distribution of the latent variable at any diffusion timestep 𝜏 can be parameterized as

<!-- formula-not-decoded -->

Proof. At any diffusion timestep 𝜏 with Σ ∗ as the destination covariance, the latent variable of forward diffusion satisfies:

<!-- formula-not-decoded -->

By setting 𝐷𝐷 ⊤ = 𝛼 ( 𝜏 ) Σ 0 following Cholesky factorization (Higham 2009) and 𝐿 = ( 1 -𝛼 ( 𝜏 )) 𝐼 , we obtain that:

<!-- formula-not-decoded -->

which shows that the distribution shares the same structural form at any 𝜏 , enabling universal expressivity over time.

Theorem 2: For any Gaussian distribution N( 𝜇 0 , Σ 0 ) , there exists a specific timestep 𝜏 ∗ at which the diffusion process produces an identical distribution.

Proof. To prove exact matchability at some 𝜏 ∗ , suppose the true segmentation map is 𝜇 0, and 𝜇 = √︁ 𝛼 ( 𝜏 ) 𝜇 0. For full match in mean and covariance, we require:

<!-- formula-not-decoded -->

Letting 𝑓 ( 𝜏 ) = 1 -𝛼 ( 𝜏 ) -∥ Σ 0 ∥ 𝐹 ∥ 𝜇 0 ∥ 2 2 , and noting 𝑓 ( 0 ) &lt; 0,

𝑓 ( 𝑇 ) &gt; 0, Intermediate Value Theorem (Russ 1980) guarantees the existence of a 𝜏 ∗ ∈ ( 0 , 𝑇 ) such that 𝑓 ( 𝜏 ∗ ) = 0.

## Appendix 5: Detailed Derivation and Analysis of 𝑥 1 𝑡 from line 3 in Algorithm 1

<!-- image -->

t

𝟏𝟏 -

t

Figure 9: Derivation of 𝑥 1 𝑡 models the semantics consistency during flow transformation of proposed ATFM.

Proof. As shown in Fig. 9, Let 𝑋 𝑇 trunc denote the source representation after truncation, and 𝑋 1 the final target representation in the latent space. Under the Optimal Transformation defined in the proposed ATFM, the semantic transition from 𝑋 𝑇 trunc to 𝑋 1 forms a linear segment. For any timestep 𝑡 ∈ [ 𝑇 trunc , 𝑇 ] , the latent state 𝑋 𝑡 can thus be computed as a linear interpolation:

<!-- formula-not-decoded -->

Let 𝑥 𝑡 ∈ 𝑋 𝑡 denote the stochastic sample at timestep 𝑡 . The Semantic-Transformation Network (ST-Net) is designed to predict the direction of transformation:

<!-- formula-not-decoded -->

With 𝑥 𝑡 and the estimated direction vector 𝑔 𝑡 ( 𝑋 𝑡 ) , we can compute an intermediate prediction 𝑥 1 𝑡 as:

<!-- formula-not-decoded -->

Substituting the approximation for 𝑔 𝑡 ( 𝑋 𝑡 ) , we obtain:

<!-- formula-not-decoded -->

Following the setting of the original flow matching (FM), we set 𝑇 = 1 and 𝑇 trunc = 0, then we can derive line 3 in Algorithm 1.

As 𝑡 → 𝑇 , 𝑥 1 𝑡 → 𝑋 1, and during training, 𝑥 1 𝑡 is supervised toward the ground truth. This modeling explicitly encodes semantic consistency during the flow transformation, ensuring that plausible and coherent predictions can still be generated even under stochastic sampling-thereby balancing diversity and accuracy within the ATFM framework.

## Appendix 6: Detailed Network Structures Structure of Network in GTR.

The network in GTR adopts a standard encoder-decoder architecture with 4 resolution levels. The encoder consists of 4 convolutional blocks with increasing filter sizes of 32, 64, 128, and 192. Each block applies two consecutive 3 × 3 convolutions followed by ReLU activations and downsampling via 2 × 2 max pooling. The decoder mirrors the encoder using transposed convolutions for upsampling and skip connections from corresponding encoder layers. On top of the final feature map, three separate 1 × 1 convolutional layers are applied to estimate the mean 𝜇 , and the variance Σ = 𝐷𝐷 𝑇 + 𝐿 for a multivariate Gaussian distribution with rank 𝑟 = 10, which parameterizes the explicit Gaussian distribution at 𝑇 trunc .

## Structure of ST-Net in SFM.

ST-Net is a temporal-conditional U-Net specifically designed as the backbone of SFM in the proposed ATFM. The network adopts a four-level encoder-decoder structure with skip connections, comprising 15 residual blocks in total. Each resolution stage includes group normalization and Swish activation, followed by linear attention blocks at all levels and full self-attention at the bottleneck layer to capture both local and global dependencies.

Temporal information is injected into the network via sinusoidal time-step embeddings, which are processed through a two-layer MLP and fused into each residual block. Optional self-conditioning is supported to enhance performance. Spatial downsampling and upsampling are implemented using strided and transposed convolutions, respectively, maintaining the spatial structure of segmentation maps.

This design enables expressive and time-aware feature representation, which is essential for accurately modeling the semantic transformation process in diffusion-based ambiguous image segmentation.

## Appendix 7: Details of Evaluation Metrics Generalised Energy Distance.

Generalised Energy Distance (GED) is a widely used metric for evaluating both the fidelity and diversity of segmentation predictions in ambiguous medical image segmentation tasks. It measures the discrepancy between the distributions of predicted and ground-truth segmentations. Given a set of predictions 𝑃 = { 𝑝 1 , . . . , 𝑝 𝑛 } and ground-truth annotations 𝐺 = { 𝑔 1 , . . . , 𝑔 𝑚 } , the GED is computed as:

<!-- formula-not-decoded -->

where 𝑑 ( 𝑝, 𝑔 ) = 1 -IoU ( 𝑝, 𝑔 ) denotes the dissimilarity between segmentation masks 𝑝 and 𝑔 . A lower GED indicates that the predicted distribution more closely matches the ground-truth distribution, capturing both sample-level accuracy and overall diversity.

## Hungarian-Matching Intersection over Union.

Hungarian-Matching Intersection over Union (HM-IoU) is adopted to measure the consistency between multiple predicted segmentations and multiple ground-truth annotations. Given a set of 𝐶 predictions 𝑃 = { 𝑝 1 , . . . , 𝑝 𝐶 } and groundtruth segmentations 𝐺 = { 𝑔 1 , . . . , 𝑔 𝐶 } , we first compute an 𝐶 × 𝐶 IoU matrix 𝑀 𝐶 × 𝐶 where each element is 𝑀 𝑖 𝑗 = IoU ( 𝑝 𝑖 , 𝑔 𝑗 ) . The optimal one-to-one assignment 𝜋 ∗ is then obtained using the Hungarian algorithm to maximize the total IoU. The HM-IoU, which offers a fair and permutationinvariant evaluation of segmentation quality across multiple predictions, is finally computed as:

<!-- formula-not-decoded -->

For implementation, we set 𝐶 = LCM ( 𝑛, 𝑚 ) , where 𝑛 and 𝑚 represent the exact numbers of predictions and groundtruth annotations in the experiment, respectively, and LCM denotes the least common multiple. This choice facilitates better alignment between the prediction and ground-truth sets.

## Maximum Dice Matching.

Maximum Dice Matching (MDM) is used to evaluate individual segmentation accuracy for each annotation considering all predictions. Given a set of predictions 𝑃 = { 𝑝 1 , . . . , 𝑝 𝑛 } and ground-truth segmentations 𝐺 = { 𝑔 1 , . . . , 𝑔 𝑚 } , MDM computes the Dice similarity coefficient between each ground truth 𝑔 𝑗 and all predictions 𝑝 𝑖 . For each ground truth 𝑔 𝑗 , the maximum Dice score over all predictions is selected. The overall MDM score is then obtained by averaging these maximum values across all ground truths, formally expressed as:

<!-- formula-not-decoded -->

This metric emphasizes the best-matching quality for each ground-truth instance, reflecting the confidence and reliability of diagnosis provided by the predictions.

Figure 10: More visualization results of predictions on both datasets demonstrate effectiveness of proposed ATFM.

<!-- image -->

## Appendix 8: More Qualitative Results

Fig. 10 shows more qualitative segmentation predictions generated by the proposed ATFM and the corresponding groundtruths. The high accuracy and diversity of predictions demonstrate the superiority of the proposed ATFM.

## References

Baumgartner, C. F.; Tezcan, K. C.; Chaitanya, K.; H¨ otker, A. M.; Muehlematter, U. J.; Schawkat, K.; Becker, A. S.; Donati, O.; and Konukoglu, E. 2019. Phiseg: Capturing uncertainty in medical image segmentation. In Medical Image Computing and Computer Assisted InterventionMICCAI 2019: 22nd International Conference, Shenzhen, China, October 13-17, 2019, Proceedings, Part II 22 , 119127. Springer.

Bellemare, M. G.; Danihelka, I.; Dabney, W.; Mohamed, S.; Lakshminarayanan, B.; Hoyer, S.; and Munos, R. 2017. The cramer distance as a solution to biased wasserstein gradients. arXiv preprint arXiv:1705.10743 .

Chavhan, G. B.; Parra, D. A.; Oudjhane, K.; Miller, S. F.; Babyn, P. S.; and Salle, F. L. P. 2008. Imaging of ambiguous genitalia: Classification and diagnostic approach1. Radiographics .

Chen, T.; Zhang, R.; and Hinton, G. 2023. Analog Bits: Generating Discrete Data using Diffusion Models with SelfConditioning. In The Eleventh International Conference on Learning Representations .

Codella, N.; Rotemberg, V.; Tschandl, P.; Celebi, M. E.; Dusza, S.; Gutman, D.; Helba, B.; Kalloo, A.; Liopyris, K.; Marchetti, M.; et al. 2019. Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic). arXiv preprint arXiv:1902.03368 .

Dong, S.; Cai, Z.; Hangel, G.; Bogner, W.; Widhalm, G.; Huang, Y.; Liang, Q.; You, C.; Kumaragamage, C.; Fulbright, R. K.; et al. 2025. A Flow-based Truncated Denoising Diffusion Model for super-resolution Magnetic Resonance Spectroscopic Imaging. Medical Image Analysis , 99: 103358.

Gao, Z.; Chen, Y.; Zhang, C.; and He, X. 2023. Modeling Multimodal Aleatoric Uncertainty in Segmentation with Mixture of Stochastic Experts. In The Eleventh International Conference on Learning Representations .

He, J.; Li, Y.; Yuan, Q.; et al. 2023. Tdiffde: A truncated diffusion model for remote sensing hyperspectral image denoising. arXiv preprint arXiv:2311.13622 .

Higham, N. J. 2009. Cholesky factorization. Wiley interdisciplinary reviews: computational statistics , 1(2): 251-254.

Ho, J.; Jain, A.; and Abbeel, P. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems , 33: 6840-6851.

Hong, S.; Bonkhoff, A. K.; Hoopes, A.; Bretzner, M.; Schirmer, M. D.; Giese, A.-K.; Dalca, A. V.; Golland, P.; and Rost, N. S. 2021. Hypernet-ensemble learning of segmentation probability for medical image segmentation with ambiguous labels. arXiv preprint arXiv:2112.06693 .

Ji, W.; Yu, S.; Wu, J.; Ma, K.; Bian, C.; Bi, Q.; Li, J.; Liu, H.; Cheng, L.; and Zheng, Y. 2021. Learning calibrated medical image segmentation via multi-rater agreement modeling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 12341-12351.

Kalpathy-Cramer, J.; Zhao, B.; Goldgof, D.; Gu, Y.; Wang, X.; Yang, H.; Tan, Y.; Gillies, R.; and Napel, S. 2016. A comparison of lung nodule segmentation algorithms: methods and results from a multi-institutional study. Journal of digital imaging , 29: 476-487.

Kingma, D. P.; and Ba, J. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 .

Kohl, S.; Romera-Paredes, B.; Meyer, C.; De Fauw, J.; Ledsam, J. R.; Maier-Hein, K.; Eslami, S.; Jimenez Rezende, D.; and Ronneberger, O. 2018. A probabilistic u-net for segmentation of ambiguous images. Advances in neural information processing systems , 31.

Kohl, S. A.; Romera-Paredes, B.; Maier-Hein, K. H.; Rezende, D. J.; Eslami, S.; Kohli, P.; Zisserman, A.; and Ronneberger, O. 2019. A hierarchical probabilistic unet for modeling multi-scale ambiguities. arXiv preprint arXiv:1905.13077 .

Li, C.; Lin, Z.; Liu, H.; Liu, Y.; Huang, Y.; Ding, X.; Tu, X.; Yuan, Y.; et al. 2024. Pˆ 2SAM: Probabilistically Prompted SAMs Are Efficient Segmentator for Ambiguous Medical Images. In ACM Multimedia .

Liao, B.; Chen, S.; Yin, H.; Jiang, B.; Wang, C.; Yan, S.; Zhang, X.; Li, X.; Zhang, Y.; Zhang, Q.; et al. 2025. Diffusiondrive: Truncated diffusion model for end-to-end autonomous driving. In Proceedings of the Computer Vision and Pattern Recognition Conference , 12037-12047.

Lipman, Y.; Chen, R. T.; Ben-Hamu, H.; Nickel, M.; and Le, M. 2023. Flow Matching for Generative Modeling. In The Eleventh International Conference on Learning Representations .

Lu, C.; and Song, Y. 2024. Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models. arXiv preprint arXiv:2410.11081 .

Monteiro, M.; Le Folgoc, L.; Coelho de Castro, D.; Pawlowski, N.; Marques, B.; Kamnitsas, K.; van der Wilk, M.; and Glocker, B. 2020. Stochastic segmentation networks: Modelling spatially correlated aleatoric uncertainty. Advances in neural information processing systems , 33: 1275612767.

Qiu, X.; Yang, M.; Ma, X.; Li, F.; Liang, D.; Luo, G.; Wang, W.; Wang, K.; and Li, S. 2025. Finding Local Diffusion Schrodinger Bridge using Kolmogorov-Arnold Network. In Proceedings of the Computer Vision and Pattern Recognition Conference , 23227-23236.

Rahman, A.; Valanarasu, J. M. J.; Hacihaliloglu, I.; and Patel, V. M. 2023. Ambiguous medical image segmentation using diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 1153611546.

Russ, S. B. 1980. A translation of Bolzano's paper on the intermediate value theorem. Historia Mathematica , 7(2): 156-185.

Song, Y.; and Ermon, S. 2020. Improved techniques for training score-based generative models. Advances in neural information processing systems , 33: 12438-12448.

Song, Y.; Sohl-Dickstein, J.; Kingma, D. P.; Kumar, A.; Ermon, S.; and Poole, B. 2020. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456 .

Zbinden, L.; Doorenbos, L.; Pissas, T.; Huber, A. T.; Sznitman, R.; and M´ arquez-Neila, P. 2023. Stochastic segmentation with conditional categorical diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision , 1119-1129.

Zepf, K. M.; Petersen, E. W.; Frellsen, J.; and Feragen, A. 2023. That Label's got Style: Handling Label Style Bias for Uncertain Image Segmentation. In Eleventh International Conference on Learning Representations .

Zhang, W.; Zhang, X.; Huang, S.; Lu, Y.; and Wang, K. 2022. A probabilistic model for controlling diversity and accuracy of ambiguous medical image segmentation. In Proceedings of the 30th ACM International Conference on Multimedia , 4751-4759.

Zheng, H.; He, P.; Chen, W.; and Zhou, M. 2022. Truncated diffusion probabilistic models. arXiv preprint arXiv:2202.09671 , 1(3.1): 2.