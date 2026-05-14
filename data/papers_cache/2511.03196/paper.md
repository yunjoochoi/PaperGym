## Cross-Modal Alignment via Variational Copula Modelling

Feng Wu * 1 Tsai Hor Chan * 1 Fuying Wang 1 Guosheng Yin 1 Lequan Yu 1

## Abstract

Various data modalities are common in real-world applications (e.g., electronic health records, medical images and clinical notes in healthcare). It is essential to develop multimodal learning methods to aggregate various information from multiple modalities. The main challenge is how to appropriately align and fuse the representations of different modalities into a joint distribution. Existing methods mainly rely on concatenation or the Kronecker product, oversimplifying the interaction structure between modalities and indicating a need to model more complex interactions. Additionally, the joint distribution of latent representations with higher-order interactions is underexplored. Copula is a powerful statistical structure for modelling the interactions among variables, as it naturally bridges the joint distribution and marginal distributions of multiple variables. We propose a novel copula-driven multimodal learning framework, which focuses on learning the joint distribution of various modalities to capture the complex interactions among them. The key idea is to interpret the copula model as a tool to align the marginal distributions of the modalities efficiently. By assuming a Gaussian mixture distribution for each modality and a copula model on the joint distribution, our model can generate accurate representations for missing modalities. Extensive experiments on public MIMIC datasets demonstrate the superior performance of our model over other competitors. The code is available at https: //github.com/HKU-MedAI/CMCM .

* Equal contribution 1 School of Computing and Data Science, University of Hong Kong, Hong Kong, China. Correspondence to: Lequan Yu &lt;lqyu@hku.hk&gt;.

Proceedings of the 42 nd International Conference on Machine Learning , Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

## 1. Introduction

Multimodal learning aims to aggregate information from multiple modalities to generate meaningful representations for downstream tasks. It has been widely explored in the context of vision-language models (Fu et al., 2023; El Banani et al., 2023), audio-visual applications (Chen et al., 2023; Mo &amp; Tian, 2023; Huang et al., 2023), image-video models (Girdhar et al., 2023; Gan et al., 2023) and healthcare applications (Wu et al., 2024a; Hayat et al., 2022). For example, multimodal learning has been applied to various healthcare tasks such as clinical prediction tasks (Zhang et al., 2023; Wu et al., 2024a), report generation (Song et al., 2022; Cao et al., 2023), and clinical trial site selection (Theodorou et al., 2024). The existing fusion strategies can be divided into early, joint, or late fusion (Huang et al., 2020), where the joint fusion paradigm is the most popular strategy and its core idea is to model the interactions between the representations of the input modalities (Hayat et al., 2022). The resulting fused embedding encodes the structural interaction between the modalities, enabling accurate prediction for each modality.

However, due to the heterogeneity of different modalities (e.g., electronic health records: EHRs, medical images, medical reports), properly aligning their distributions remains a challenge. The existing alignment strategies mainly rely on concatenation or Kronecker products which oversimplify the interactions among different modalities. A recent work (Salzmann et al., 2022) emphasizes simple probabilistic assumptions on the marginals and neglects to explore statistical assumptions about the joint distributions of the modalities. This approach may result in biased fused representations, limiting the performance of downstream tasks and the generalizability and robustness of the resulting multimodal models. Therefore, there is still a need for an approach that can more appropriately align the distributions of modalities and model the potentially complex interactions among them.

Copula models have shown great success in modelling the interactions of variables as they construct a bridge between the joint distribution and their marginals (Cherubini, 2004). However, copula models are less explored in deep learning field as most existing approaches heavily rely on samplingbased methods (e.g., MCMC (Silva &amp; Gramacy, 2009)), which are relatively slow and difficult to scale to modern deep learning settings (Smith &amp; Loaiza-Maya, 2023). Although some recent works have attempted to introduce copula to deep learning models through stochastic variational inference (Smith &amp; Loaiza-Maya, 2023), the potential of copula in multimodal learning is still underexplored.

Moreover, existing multimodal learning methods mostly assume the existence of all modalities. In reality, some modalities may be missing for some observations due to various reasons (e.g., missing medical images or reports for some patients due to clinical and administrative factors in healthcare), which may pose a major challenge in multimodal learning. The existing solutions either discard these observations or impute simple values (e.g., zeros or means from other observations) to address the missing modality problem. However, these approaches ignore the marginal distributions of the modality and often mislead the learning of the joint distribution. Therefore, properly learning the marginal distributions is also necessary to generate unbiased latent representations for the observations with missing modalities.

In light of the aforementioned challenges, we propose a novel copula-driven multimodal learning framework, namely CM 2 ( C rossM odal alignment via variational C opula M odelling), to tackle the joint fusion paradigm from a probabilistic perspective. Our contributions can be summarized as: (1) We for the first time introduce copula modelling into multimodal learning, where we interpret copula as an effective tool of distribution alignment, guaranteed by Sklar's theorem. (2) We employ a Gaussian mixture model on the marginal distribution of each modality to enable more flexible modelling of the high-dimensional feature distribution of different modalities. (3) We adopt stochastic variational inference to optimize the copula model, which enables the scalability of our model to large-scale datasets. (4) We adopt the learned marginal distribution as the data generator to accurately impute the missing observations. (5) Empirical results on real multimodal MIMIC datasets demonstrate the good performance of our method and ablation analysis corroborates the effectiveness of copula in modality alignments and robustness to potential variations.

## 2. Related Works

Multimodal Representation Learning. Multimodal representation learning aims to effectively integrate information from different modalities for accurate predictions on the downstream tasks. Early works (Hayat et al., 2022; Ding et al., 2022; Trong et al., 2020) focus on late fusion that merges unimodal representations via, for instance, concatenation or the Kronecker product. However, such approaches oversimplify the interactions of the modalities and mostly lead to biased fused representations. Therefore, the struc- tural interactions of the modalities need to be encoded in the fused representation for more effective multimodal learning. Recently, modelling the interaction between modalities has received increasing attention. Liang et al. (2024) proposed an information decomposition framework to define and quantify different types of interactions between modalities. Transformer-based methods have greatly facilitated the progress by modelling the cross-model tokens (Zhang et al., 2023; Theodorou et al., 2024). However, matching the correspondence with transformers introduces high computational complexity, which prompts a more efficient approach for representation alignment.

Copula Deep Learning. Copula is a promising tool in modelling the interactions or correlations between variables and it constructs a bridge between the joint distribution and marginal distributions. Copula has been widely applied in financial risk management (Hofert, 2021; Rodriguez, 2007), signal processing, and healthcare (Zeng &amp; Wang, 2022) due to its capability in modelling complex interactions. Traditional copula models rely on closed-form solutions of the likelihood and estimate the copula parameter with samplingbased approaches (e.g., MCMC (Silva &amp; Gramacy, 2009)). However, these algorithms suffer from high time complexity, making them less applicable to high-dimensional data. Recently, with the emergence of deep learning, there have been works integrating copula models into deep learning frameworks (Tagasovska et al., 2019; Smith et al., 2020). To tackle the inherent high dimensionality, variational inference is adopted to solve copula models in high dimensions (Tran et al., 2015; Smith &amp; Loaiza-Maya, 2023). For example, Tagasovska et al. (2019) introduced copula to variational autoencoders to create deep generative models. However, the potential of copula in multimodal learning is still underexplored.

Learning with Missing Data. Traditional multimodal learning assumes all modalities are available, but in reality, some observations may be missing, like medical images or reports in clinical data. Late fusion is a common strategy to address missing modalities by aggregating predictions (Yoo et al., 2019) or latent space representations (Theodorou et al., 2024) from the available modalities. While effective, it treats each modality independently and lacks interactions among them. Some research focuses on extracting shared information across modalities for downstream tasks (Deldari et al., 2023; Yao et al., 2024), which can be challenging, particularly with heterogeneous modalities like EHRs and CXRs. Under the missing at random (MAR) assumption, imputation methods have become a popular approach for handling missing data. Some approaches assume that the missing modality follows a certain distribution, imputing the missing values using the mean or mode of that distribution (Ma et al., 2021). Others impute missing modalities'

Figure 1. Overview of our proposed CM 2 framework. For a dataset with M modalities, we extract modality-specific embeddings z m via Encoder m and compute its Gaussian mixture model (GMM). We then model the marginal distributions and estimate the joint distribution using a copula family C . We sample ˆ z m from its GMM if modality m is missing. The concatenated embedding z then passes through a 2-layer LSTM fusion module and MLP classifier to predict ˆ y . The ELBO for backpropagation can be obtained by aggregating the task-specific loss (e.g., cross-entropy loss) and the negative log-likelihood from the joint distribution.

<!-- image -->

representations in the latent feature space via deep learning models, attempting to preserve model performance by modeling relationships (Zhang et al., 2022; Wu et al., 2024b) or generating global representations for the missing data (Hayat et al., 2022). Despite their successes, these distributional assumptions or the learned relationships may be inaccurate, potentially introducing bias into the model. Therefore a probabilistic assumption is needed to guarantee the unbiasedness of learned marginal distributions.

## 3. Methodology

## 3.1. Preliminaries

Copula. An M -variate function C ( u 1 , . . . , c M ) , where u m ∈ [0 , 1] for all m , is a copula if and only if C defines a valid joint cumulative distribution function (CDF) of the random vector ( U 1 , . . . , U M ) with each U m distributed as uniform on the unit interval. Taking the bivariate Gumbel copula as an example, given the CDF values of the first and second modalities u and v , the bivariate distribution is

<!-- formula-not-decoded -->

and its copula density is

<!-- formula-not-decoded -->

where g ( u, v ; α ) = ( -log u ) α + ( -log v ) α . The effects of different copula families are discussed in the ablation analysis. Details of different copula families and their corresponding distribution and density functions are provided in Appendix C.

Multimodal Learning. Given the multimodal training dataset D tr = { ( x ( i ) 1 , . . . , x ( i ) M , y ( i ) ) } n i =1 , where x ( i ) m is the i -th observation of the m -th modality and y ( i ) is the corresponding label, the goal is to train a multimodal model M Θ ( · ) with parameter Θ such that the model can achieve optimal performance in downstream tasks.

## 3.2. Copula Multimodal Learning

The overview of the proposed copula-driven multimodal learning framework is shown in Figure 1. Given multimodal data, we extract each modality-specific embedding and compute its Gaussian mixture model (GMM). We then model the marginal densities and estimate the joint distribution using a copula family C . If modality m is missing, we generate feature embeddings from its GMM. The concatenated embeddings z are passed through a fusion module and an MLP classifier for prediction. The evidence lower bound (ELBO) combines the copula log-likelihood and task-specific loss.

Gaussian Mixture Assumption. The GMM is a common technique in machine learning to model the behavior of distributions in high dimensions (Song et al., 2024; Bai

## Algorithm 1 Sampling algorithm of our proposed method.

- 1: Input:

- 2: Multimodal model M Θ ( · ) with parameter Θ

- 3: The copula parameter α

- 4: Means and covariances of GMM: { ( µ mk , Σ mk ) | m = 1 , . . . , M, k = 1 , . . . , K }

- 5: Training set D tr = { ( x ( i ) 1 , . . . , x ( i ) M , y ( i ) ) } n i =1

- 6: Output: Trained f Θ

- 7: for ( x ( i ) 1 , . . . , x ( i ) M , y ( i ) ) in D tr do

- 8: ˆ y ( i ) = M Θ ( x ( i ) 1 , . . . , x ( i ) M )

- 9: Compute task-specific loss L obj with ˆ y ( i ) and y ( i )

- 10: Compute the KL( q ∥ π ) and hence the ELBO

- 11: Backpropagate the ELBO to update Θ , α

- 12: end for

- 13: Return: Trained M Θ

et al., 2022; Ni et al., 2021). To generate a more flexible feature distribution, we assume the feature distribution of the m -th modality follows a K -mixture of multivariate GMM,

<!-- formula-not-decoded -->

where π mk is the mixture weight, µ mk is the mean vector, and Σ mk is the covariance matrix of the k -th mixture of the m -th modality. Let µ = { µ mk : m ∈ [ M ] , k ∈ [ K ] } and Σ = { Σ mk : m ∈ [ M ] , k ∈ [ K ] } . Without loss of generality, we predict π mk with a multilayer perceptron (MLP) with a softmax output layer and adopt the reparameterization trick (Nalisnick, 2018; Tran et al., 2022), which assumes Σ mk is diagonal. We further set µ and Σ to be trainable by gradient backpropagation. We compute the cumulative distribution function of the multivariate Gaussian distributions using the approximation provided in Marmin et al. (2015). By employing a mixture model, we can model a wider range of distributions of each modality and improve the flexibility and robustness.

Multivariate Copula. Using the multivariate copula, the joint distribution function of the modalities is given by

<!-- formula-not-decoded -->

where C ( F 1 ( z 1 ) , . . . , F M ( z M )) is the M -dimensional copula distribution function, and F m ( z m ) is the marginal cumulative distribution function of the m -th modality which is the CDF of the GMM model defined in Eq. (1).

## 3.3. Stochastic Variational Inference

To tackle the scalability of CM 2 to modern deep learning settings, we adopt the stochastic variational inference to optimize the proposed copula model and treat the copula parameter α as trainable. Algorithm 1 presents the overall workflow of our method.

Variational Family. We use a variational posterior q to approximate the true posterior of the joint distribution. The variational family of the copula model that we optimize during training is given by

<!-- formula-not-decoded -->

where q m ( z m ) is the density of the variational posterior of the GMM of the m -th modality, and Q m ( z m ) is the corresponding CDF.

The Evidence Lower Bound (ELBO). The joint objective function can be written as the negation of the negative loglikelihood,

<!-- formula-not-decoded -->

where f m ( z ( i ) m ) is the marginal density of modality m , c ( Q 1 ( z ( i ) 1 ) , . . . , Q M ( z ( i ) M )) is the copula density, λ cop is the regularization parameter of the copula, and L obj is the task-specific loss (e.g., cross-entropy loss). We compute the gradient based on the ELBO and backpropagate it to µ and Σ to learn the marginal distributions of each modality, with the copula parameter α to learn the interactions among these modalities and the multimodal model parameter Θ to learn the embedding, fusion, and classification layers.

## 3.4. Handling Missing Modality

Owing to the probabilistic design of our method, our framework can also generate pseudo representations for missing modalities. Without loss of generality, we assume that the missing modalities are missing at random (MAR) and, following prior works (Tran et al., 2017; Ma et al., 2021; Zhang et al., 2022; Wang et al., 2023), we impute the features of these missing modalities in the latent space. We consider missing modalities with complete labels where only the observations are missing. The learned GMM for each modality can be treated as a data generation model, and we can generate feature embeddings through sampling from the GMM of each modality (i.e., z ( i ) m ∼ F m ). Then the generated feature embeddings can be treated as the feature input to the classification layer and predictions can be obtained.

By learning the copula parameter α , the marginal distribution of each modality contains information from other modalities and information of the interactions. The generated feature representation z ( i ) m can thus better reflect the characteristics of the joint distribution, which, as a result, can improve the quality of the representation and the downstream task performance.

## 3.5. Theoretical Guarantee with Sklar's Theorem.

We make use of Sklar's theorem to demonstrate the uniqueness of the joint distribution as follows.

Theorem 3.1. (Sklar's theorem) (Sklar, 1959) Let F ( x 1 , . . . , x M ) be an M -variate CDF for ( X 1 , . . . , X M ) with the marginal CDF for the m -th variable given by F m ( x m ) , m = 1 , . . . , M .

1. There exists an M -dimensional copula such that

<!-- formula-not-decoded -->

for all x m ∈ R .

2. Conversely, given any copula C and univariate CDFs F 1 , . . . , F M , C is a valid joint CDF for ( X 1 , . . . , X M ) . If F is continuous, then C in Eq. (2) is unique.

The above theorem lays the foundation to construct joint distributions with the same marginals but different dependence structures, or conversely by fixing the dependence structure and varying the behaviour in individual modalities (Tagasovska et al., 2019). This allows us to update the marginal distributions and the copula parameter separately. Furthermore, since we assume a GMM for each modality and they are continuous by definition, the uniqueness of the copula C can be guaranteed and the identifiability of the model can be enhanced.

## 4. Experiments

## 4.1. Datasets and Experimental Setting

Datasets. We evaluate the performance of CM 2 using largescale, real-world EHR datasets: MIMIC-III (Johnson et al., 2016), MIMIC-IV (Johnson et al., 2023), and MIMIC-CXR (Johnson et al., 2019). MIMIC-III and MIMIC-IV are publicly available datasets containing real-world EHR data from patients admitted to the intensive care units (ICUs) or emergency departments of Beth Israel Deaconess Medical Center (BIDMC), comprising numerical time series and clinical notes. MIMIC-CXR is a dataset of Chest X-ray (CXR) images along with radiology reports collected from BIDMC, with a subset of patients matched to those in MIMIC-IV.

Following Hayat et al. (2022), we utilize the MIMICIV and MIMIC-CXR datasets for our multimodal experiments. Additionally, we extend our experiments to the MIMIC-III dataset. As CXR images are not available in MIMIC-III, we replace them with clinical notes. Table

Table 1. Numbers of samples in training/validation/testing sets

| Datasets                          | Train             | Valid             | Test              | Total             |
|-----------------------------------|-------------------|-------------------|-------------------|-------------------|
|                                   | Complete Datasets | Complete Datasets | Complete Datasets | Complete Datasets |
| MIMIC-III                         | 14,681            | 3,222             | 3,236             | 21,139            |
| MIMIC-III NOTE                    | 3,652             | 815               | 806               | 5,273             |
| MIMIC-IV                          | 18,064            | 2,035             | 4,972             | 25,071            |
| MIMIC-CXR                         | 344,529           | 9,497             | 23,069            | 377,095           |
|                                   | Matched Datasets  | Matched Datasets  | Matched Datasets  | Matched Datasets  |
| MIMIC-III &#124; NOTE             | 3,652             | 815               | 806               | 5,273             |
| MIMIC-IV &#124; CXR               | 4,287             | 465               | 1,179             | 5,931             |
| MIMIC-IV &#124; CXR &#124; REPORT | 4,287             | 465               | 1,179             | 5,931             |

1 provides an overview of the real datasets and the training/validation/testing split sets. We extract 25,071 ICU stays with EHR records from MIMIC-IV, 5,931 of which are matched to CXR images and reports. Similarly, we extract 21,139 ICU stays with EHR records from MIMICIII, with 5,273 stays matched to clinical notes. To evaluate the performance of CM 2 on cross-modal alignment, we conduct experiments on totally matched bi-modal and tri-modal settings. We also evaluate partially matched datasets to demonstrate the robustness of CM 2 in the presence of missing modalities. Further details on the datasets can be found in Appendix A.1.

Task and Evaluation Metrics. Following the common practice in clinical prediction tasks (Hayat et al., 2022; Zhang et al., 2022; Wu et al., 2024b; Wang et al., 2024), we focus on two clinical prediction tasks: (1) In-Hospital Mortality (IHM) prediction, which predicts whether a patient will pass away during the hospital stay; and (2) Readmission (READM) prediction, which aims to predict whether a patient will be readmitted within 30 days after discharge. To assess model performance, we compute the area under the precision-recall curve (AUPR) and the area under the receiver operating characteristic curve (AUROC). Results are reported with the corresponding 95% confidence intervals based on 1,000 bootstrap iterations.

Backbone Encoders. Following Hayat et al. (2022), we utilize ResNet34 (He et al., 2016) as the backbone encoder for the CXR image data. For time-series data, we employ a twolayer stacked LSTM network (Graves &amp; Graves, 2012). For clinical notes and radiology reports, we use the TinyBERT encoder (Jiao et al., 2019). A projection layer is applied to map the modality embeddings into the same latent space.

## 4.2. Compared Methods

We compare CM 2 against the following baselines: (1) MMTM (Joze et al., 2020) is a flexible plugin module that facilitates information exchange between modalities. We address missing CXR and clinical notes during training and testing by filling in the missing data with zeros. (2) DAFT (Pölsterl et al., 2021) is a module designed to exchange information between tabular data and image modalities when integrated into CNN models. Similarly, we replace missing CXR and clinical notes with zero matrices during training and testing. (3) Unified (Hayat et al., 2021) is a dynamic approach for integrating auxiliary data modalities and combining all representations via a unified classifier. It handles missing data inherently and leverages all available modality-specific information. (4) MedFUSE (Hayat et al., 2022) employs LSTM-based fusion to combine features from image or language encoders with EHR encoders. It handles missing modalities by learning a global representation for absent CXR or clinical notes. (5) DrFuse (Yao et al., 2024) leverages disentangled representation learning to create a shared representation between the EHR and image modalities, even when one modality is missing.

Table 2. Results of AUROC and AUPR with 95% confidence intervals on MIMIC-III and MIMIC-IV datasets with totally matched modalities. The best results are highlighted in boldface .

| Datasets Models                                                                                                                                         | IHM                                                                                                                                                         | IHM                                                                                                                                                         | READM                                                                                                                                                       | READM                                                                                                                                                       |
|---------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                                                                                                         | AUROC ( ↑ )                                                                                                                                                 | AUPR ( ↑ )                                                                                                                                                  | AUROC ( ↑ )                                                                                                                                                 | AUPR ( ↑ )                                                                                                                                                  |
| MIMIC-III MMTM(Joze et al., 2020) DAFT (Pölsterl et al., 2021) Unified (Hayat et al., 2021) MedFuse (Hayat et al., 2022) DrFuse (Yao et al., 2024) CM 2 | 0.776 (0 . 728 , 0 . 819) 0.792 (0 . 746 , 0 . 839) 0.827 (0 . 782 , 0 . 868) 0.826 (0 . 781 , 0 . 866) 0.835 (0 . 793 , 0 . 874) 0.854 (0 . 820 , 0 . 861) | 0.347 (0 . 268 , 0 . 447) 0.388 (0 . 299 , 0 . 484) 0.466 (0 . 371 , 0 . 569) 0.430 (0 . 340 , 0 . 537) 0.511 (0 . 417 , 0 . 607) 0.513 (0 . 460 , 0 . 557) | 0.716 (0 . 670 , 0 . 762) 0.701 (0 . 653 , 0 . 746) 0.714 (0 . 662 , 0 . 759) 0.725 (0 . 676 , 0 . 774) 0.749 (0 . 699 , 0 . 795) 0.754 (0 . 731 , 0 . 774) | 0.341 (0 . 277 , 0 . 419) 0.325 (0 . 262 , 0 . 403) 0.423 (0 . 344 , 0 . 504) 0.414 (0 . 338 , 0 . 502) 0.441 (0 . 356 , 0 . 527) 0.445 (0 . 403 , 0 . 487) |
| MIMIC-IV MMTM(Joze et al., 2020) DAFT (Pölsterl et al., 2021) Unified (Hayat et al., 2021) MedFuse (Hayat et al., 2022) DrFuse (Yao et al., 2024) CM 2  | 0.802 (0 . 770 , 0 . 835) 0.815 (0 . 782 , 0 . 844) 0.808 (0 . 778 , 0 . 840) 0.813 (0 . 777 , 0 . 844) 0.818 (0 . 784 , 0 . 850) 0.827 (0 . 790 , 0 . 859) | 0.429 (0 . 362 , 0 . 513) 0.454 (0 . 387 , 0 . 538) 0.429 (0 . 367 , 0 . 512) 0.448 (0 . 380 , 0 . 528) 0.460 (0 . 391 , 0 . 540) 0.492 (0 . 423 , 0 . 566) | 0.713 (0 . 677 , 0 . 750) 0.729 (0 . 692 , 0 . 766) 0.719 (0 . 680 , 0 . 756) 0.725 (0 . 690 , 0 . 762) 0.726 (0 . 689 , 0 . 760) 0.737 (0 . 704 , 0 . 773) | 0.420 (0 . 362 , 0 . 489) 0.433 (0 . 378 , 0 . 499) 0.450 (0 . 390 , 0 . 513) 0.438 (0 . 379 , 0 . 508) 0.430 (0 . 370 , 0 . 495) 0.466 (0 . 404 , 0 . 529) |

Table 3. Results of AUROC and AUPR with 95% confidence intervals on MIMIC-III and MIMIC-IV datasets with partially matched modalities (i.e., missing modalities). The best results are highlighted in boldface .

| Models                                                                                                                                        | IHM                                                                                                                                                         | IHM                                                                                                                                                         | READM                                                                                                                                                       | READM                                                                                                                                                       |
|-----------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                                                                                               | AUROC ( ↑ )                                                                                                                                                 | AUPR ( ↑ )                                                                                                                                                  | AUROC ( ↑ )                                                                                                                                                 | AUPR ( ↑ )                                                                                                                                                  |
| MMTM(Joze et al., 2020) DAFT (Pölsterl et al., 2021) Unified (Hayat et al., 2021) MedFuse (Hayat et al., 2022) DrFuse (Yao et al., 2024) CM 2 | 0.846 (0 . 825 , 0 . 865) 0.854 (0 . 836 , 0 . 873) 0.849 (0 . 829 , 0 . 868) 0.850 (0 . 830 , 0 . 868) 0.839 (0 . 817 , 0 . 861) 0.856 (0 . 833 , 0 . 877) | 0.450 (0 . 399 , 0 . 509) 0.495 (0 . 440 , 0 . 552) 0.491 (0 . 436 , 0 . 542) 0.480 (0 . 426 , 0 . 533) 0.474 (0 . 422 , 0 . 531) 0.510 (0 . 463 , 0 . 566) | 0.742 (0 . 716 , 0 . 766) 0.748 (0 . 724 , 0 . 772) 0.751 (0 . 728 , 0 . 772) 0.753 (0 . 730 , 0 . 775) 0.749 (0 . 727 , 0 . 770) 0.754 (0 . 708 , 0 . 795) | 0.413 (0 . 371 , 0 . 455) 0.429 (0 . 386 , 0 . 473) 0.427 (0 . 383 , 0 . 467) 0.437 (0 . 396 , 0 . 480) 0.411 (0 . 371 , 0 . 455) 0.445 (0 . 358 , 0 . 523) |
| MMTM(Joze et al., 2020) DAFT (Pölsterl et al., 2021) Unified (Hayat et al., 2021) MedFuse (Hayat et al., 2022) DrFuse (Yao et al., 2024) CM 2 | 0.855 (0 . 840 , 0 . 869) 0.857 (0 . 841 , 0 . 870) 0.854 (0 . 839 , 0 . 870) 0.855 (0 . 840 , 0 . 870) 0.857 (0 . 841 , 0 . 872) 0.858 (0 . 844 , 0 . 872) | 0.519 (0 . 477 , 0 . 561) 0.526 (0 . 487 , 0 . 565) 0.505 (0 . 463 , 0 . 545) 0.500 (0 . 458 , 0 . 541) 0.518 (0 . 479 , 0 . 562) 0.527 (0 . 490 , 0 . 568) | 0.765 (0 . 747 , 0 . 783) 0.765 (0 . 747 , 0 . 782) 0.759 (0 . 742 , 0 . 776) 0.762 (0 . 744 , 0 . 778) 0.768 (0 . 749 , 0 . 784) 0.771 (0 . 752 , 0 . 788) | 0.465 (0 . 430 , 0 . 501) 0.476 (0 . 442 , 0 . 510) 0.470 (0 . 436 , 0 . 503) 0.465 (0 . 430 , 0 . 501) 0.485 (0 . 451 , 0 . 520) 0.486 (0 . 452 , 0 . 518) |

Table 4. Ablation study on different alignment loss functions with AUROC and AUPR on MIMIC-IV.

| Alignment loss   | IHM         | IHM        | READM       | READM      |
|------------------|-------------|------------|-------------|------------|
| Alignment loss   | AUROC ( ↑ ) | AUPR ( ↑ ) | AUROC ( ↑ ) | AUPR ( ↑ ) |
| Cosine           | 0.820       | 0.470      | 0.726       | 0.445      |
| KL               | 0.826       | 0.489      | 0.731       | 0.446      |
| Copula           | 0.827       | 0.492      | 0.737       | 0.466      |

## 4.3. Experimental Results

Quantitative Results. Table 2 presents results on the MIMIC-III and MIMIC-IV datasets with totally matched modalities. CM 2 outperforms all the five baselines in all cases. Notably, for the IHM task, CM 2 exceeds the best baseline by 1.9% in AUROC on MIMIC-III and 3.2% in AUPR on MIMIC-IV. These results demonstrate the effectiveness of CM 2 in capturing the interactions between modalities and enhancing the performance of multimodal learning tasks in clinical prediction.

Table 5. Ablation study on the influence of different components (e.g., copula alignment, gradient-preserving sampling (GPS), and fusion module) of our proposed method on MIMIC-IV.

| Models                        | Matched   | IHM         | IHM        | READM       | READM      |
|-------------------------------|-----------|-------------|------------|-------------|------------|
|                               |           | AUROC ( ↑ ) | AUPR ( ↑ ) | AUROC ( ↑ ) | AUPR ( ↑ ) |
| w/o copula w/o GPS w/o fusion | ×         | 0.855       | 0.506      | 0.753       | 0.459      |
|                               | ×         | 0.858       | 0.521      | 0.763       | 0.473      |
|                               | ×         | 0.860       | 0.531      | 0.762       | 0.476      |
| CM 2                          | ×         | 0.858       | 0.527      | 0.771       | 0.486      |
| w/o copula                    | ✓         | 0.809       | 0.434      | 0.717       | 0.424      |
| w/o fusion                    | ✓         | 0.811       | 0.446      | 0.720       | 0.424      |
| CM 2                          | ✓         | 0.827       | 0.492      | 0.737       | 0.466      |

Table 6. Results on different copula families and the influence of the missing modality on MIMIC-IV.

| Matched   | Copula   | IHM         | IHM        | READM       | READM      |
|-----------|----------|-------------|------------|-------------|------------|
|           |          | AUROC ( ↑ ) | AUPR ( ↑ ) | AUROC ( ↑ ) | AUPR ( ↑ ) |
| ×         | Gumbel   | 0.858       | 0.527      | 0.772       | 0.485      |
| ✓         | Gumbel   | 0.825       | 0.488      | 0.735       | 0.463      |
| ×         | Frank    | 0.858       | 0.527      | 0.771       | 0.486      |
| ✓         | Frank    | 0.827       | 0.492      | 0.737       | 0.466      |
| ×         | Gaussian | 0.859       | 0.527      | 0.771       | 0.485      |
| ✓         | Gaussian | 0.827       | 0.488      | 0.736       | 0.458      |

Table 3 reports results on the MIMIC-III and MIMIC-IV datasets with partially matched modalities (e.g., missing modality). CM 2 outperforms the baselines in all cases, with the best performance on the MIMIC-III dataset, where it outperforms the best baseline by 1.5% in AUPR for the IHM task and 0.8% in AUPR for the READM task. This indicates that CM 2 effectively learns the joint distribution of the modalities, generating robust and unbiased representations in the presence of missing modalities.

Moreover, our results reveal that the performance on the partially matched datasets is superior to that on the matched datasets. This can be attributed to the larger number of observations in the partially matched datasets, underscoring the importance of multimodal learning in the presence of missing modalities. Lastly, we observe that the performance on MIMIC-IV is better than that on MIMIC-III under the partially matched setting, likely due to the larger number of observations in MIMIC-IV. Additionally, the heterogeneity between modalities in MIMIC-IV may be greater than that in MIMIC-III, contributing to the difference in performance between the two datasets under the totally matched setting.

Qualitative Analysis. Wevisualize the densities of different families of copula and see how the interactions between modalities are captured. Figure 2 presents the visualization of learned densities of the Gumbel, Gaussian, and Frank copula families, respectively. We observe that the Gumbel copula is more focused on the positive dependence between the modalities while the Gaussian copula has lower weight on modelling tail dependencies. On the other hand, the Frank copula is tail-symmetric and capable of modelling both positive and negative dependencies. Hence, it can cover more dependency structures, indicating that it may be a more flexible choice for modelling complex interactions. We further demonstrate how CM 2 learns the interactions through density plots at different epochs. The detailed discussion can be found in Appendix D. We also study how CM 2 learns the correlation over epochs. Figure 3 presents the change in the estimated α and its corresponding correlation α -1 α over training epochs. We discover that the model learns a positive correlation over the epochs, and the correlation converges at around 0.601. This implies that by backpropagating the gradient to the copula parameter α , the model can learn the interactions between the modalities during training.

## 4.4. Ablation Analysis

Effectiveness of Copula Alignment. We study the effects of the alignment loss, as presented in Table 4. The copula alignment loss achieves the best performance, outperforming the popular cosine similarity alignment and KL divergence alignment.

Ablation on Contribution of the Designed Modules. To further evaluate the performance of CM 2 , we conduct an ablation study by removing the copula alignment, the gradientpreserving sampling (GPS), and fusion modules, respectively. As shown in Table 5, the performance of CM 2 significantly declines without copula alignment, underscoring the importance of modeling the copula joint distribution before fusing modality features. Additionally, in most cases, removing the fusion module leads to a notable drop in performance, emphasizing its critical role in capturing modality interactions. Furthermore, we observe a slight decline when the GPS is removed, indicating its effectiveness in generating unbiased representations for observations with missing modalities.

Ablation on Different Families of Copula. We also compare the performance of CM 2 under different settings for missing modalities and copula families. The accuracy relies heavily on the assumed copula family (Zeng &amp; Wang, 2022). We examine the performance of our method over an array of commonly used copula families. Table 6 presents the results of CM 2 on the MIMIC-IV dataset. We discover that while our method is generally robust to the choice of copula family, the best-performing copula varies across different tasks. This indicates that different tasks highlight different characteristics (e.g., extreme values for mortality) that can be captured when a proper copula family is chosen.

Extension to More Modalities. We further investigate the impact of incorporating more auxiliary modalities. We adapt all baselines into the tri-modal setting. Table 7 presents the results for CM 2 and the baselines on the MIMIC-IV dataset under the tri-modal setting: EHR time series, CXR images, and radiology reports. Across both tasks, CM 2 consistently outperforms the baselines, achieving the best performance. Notably, the baseline models show a decline in performance compared to the bi-modal setting, suggesting that incorporating additional modalities becomes more challenging as the alignment complexity increases. Despite this, CM 2 maintains strong performance, demonstrating its robustness and effectiveness in aligning multiple modalities.

Table 7. Results of AUROC and AUPR with 95% confidence intervals using three modalities (EHR time series, CXR images, and CXR reports) on MIMIC-IV.

| Models                                                                                                                                        | IHM                                                                                                                                                         | IHM                                                                                                                                                         | READM                                                                                                                                                       | READM                                                                                                                                                       |
|-----------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                                                                                               | AUROC ( ↑ )                                                                                                                                                 | AUPR ( ↑ )                                                                                                                                                  | AUROC ( ↑ )                                                                                                                                                 | AUPR ( ↑ )                                                                                                                                                  |
| MMTM(Joze et al., 2020) DAFT (Pölsterl et al., 2021) Unified (Hayat et al., 2021) MedFuse (Hayat et al., 2022) DrFuse (Yao et al., 2024) CM 2 | 0.777 (0 . 739 , 0 . 813) 0.788 (0 . 754 , 0 . 821) 0.795 (0 . 761 , 0 . 827) 0.801 (0 . 767 , 0 . 836) 0.808 (0 . 773 , 0 . 839) 0.824 (0 . 793 , 0 . 856) | 0.370 (0 . 312 , 0 . 443) 0.397 (0 . 331 , 0 . 471) 0.420 (0 . 351 , 0 . 497) 0.427 (0 . 367 , 0 . 511) 0.451 (0 . 376 , 0 . 524) 0.471 (0 . 399 , 0 . 554) | 0.689 (0 . 650 , 0 . 723) 0.706 (0 . 670 , 0 . 742) 0.715 (0 . 679 , 0 . 749) 0.713 (0 . 675 , 0 . 749) 0.728 (0 . 691 , 0 . 761) 0.730 (0 . 694 , 0 . 764) | 0.401 (0 . 347 , 0 . 463) 0.403 (0 . 346 , 0 . 464) 0.430 (0 . 376 , 0 . 495) 0.419 (0 . 356 , 0 . 487) 0.433 (0 . 370 , 0 . 495) 0.444 (0 . 385 , 0 . 509) |

<!-- image -->

Figure 2. Plots of the fitted copula density to demonstrate the interrelationship captured by the copula model (Left: Gumbel, middle: Gaussian, right: Frank).

Figure 3. Plots comparing the value of α and the correlation,Corr = ( α -1) /α learned by the Gumbel copula model.

<!-- image -->

## 5. Conclusion

We introduce copula modelling into multimodal representation learning. Using a copula can effectively model the interactions among different modalities, and impute the missing modalities through sampling from learned marginals. Empirical evaluation validates the predictive performance on the multimodal learning tasks, on both the fully and partially matched datasets. Ablation studies show that the proposed copula model can serve as a promising modality alignment tool due to the consistently satisfactory performance over different copula families. Our idea can be potentially extended to works that require effective fusion or distribution alignment, including domain adaptation, multi-feature and multi-view learning.

Limitations and Future Works. Using a neural network to learn the copula parameter α may be insufficient (since the joint log-likelihood may not be convex). Hence, an alternative updating algorithm (e.g., partial likelihood) is needed in future development of copula multimodal learning to ensure that each loss is convex to apply gradient descent. While we select healthcare datasets to demonstrate the effectiveness of our model, our method can be extended to other types of multimodal datasets.

Acknowledgement. We thank the program chairs, area chairs, and reviewers for many constructive suggestions that have significantly improved the paper. This work was supported in part by the Research Grants Council of Hong Kong (17308321, 27206123, C5055-24G, and T45-401/22-N), Patrick SC Poon endowment fund, Hong Kong Innovation and Technology Fund (ITS/273/22 and ITS/274/22), National Natural Science Foundation of China (No. 62201483), and Guangdong Natural Science Fund (No. 2024A1515011875).

## Impact Statement

The goal of this work is to advance the field of machine learning. Our work can be potentially extended to areas that require effective fusion or distribution alignment, including domain adaptation, multi-feature learning, and multi-view learning.

## References

- Bai, J., Kong, S., and Gomes, C. P. Gaussian mixture variational autoencoder with contrastive learning for multilabel classification. In international conference on machine learning , pp. 1383-1398. PMLR, 2022.
- Cao, Y., Cui, L., Zhang, L., Yu, F., Li, Z., and Xu, Y. Mmtn: multi-modal memory transformer network for image-report consistent medical report generation. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 37, pp. 277-285, 2023.
- Chen, J., Zhang, R., Lian, D., Yang, J., Zeng, Z., and Shi, J. iquery: Instruments as queries for audio-visual sound separation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 1467514686, 2023.
- Cherubini, U. Copula methods in finance. John Wiley &amp; Sons google schola , 2:949-956, 2004.
- Deldari, S., Spathis, D., Malekzadeh, M., Kawsar, F., Salim, F., and Mathur, A. Latent masking for multimodal selfsupervised learning in health timeseries. arXiv preprint arXiv:2307.16847 , 2023.
- Ding, N., Tian, S.-w., and Yu, L. A multimodal fusion method for sarcasm detection based on late fusion. Multimedia Tools and Applications , 81(6):8597-8616, 2022.
- Dosovitskiy, A. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 , 2020.
- El Banani, M., Desai, K., and Johnson, J. Learning visual representations via language-guided sampling. In Proceedings of the ieee/cvf conference on computer vision and pattern recognition , pp. 19208-19220, 2023.
- Fu, Z., Mao, Z., Song, Y., and Zhang, Y. Learning semantic relationship among instances for image-text matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 15159-15168, 2023.
- Gan, T., Wang, Q., Dong, X., Ren, X., Nie, L., and Guo, Q. Cnvid-3.5 m: Build, filter, and pre-train the large-scale public chinese video-text dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 14815-14824, 2023.
- Girdhar, R., El-Nouby, A., Singh, M., Alwala, K. V., Joulin, A., and Misra, I. Omnimae: Single model masked pretraining on images and videos. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pp. 10406-10417, 2023.
- Graves, A. and Graves, A. Long short-term memory. Supervised sequence labelling with recurrent neural networks , pp. 37-45, 2012.
- Harutyunyan, H., Khachatrian, H., Kale, D. C., Ver Steeg, G., and Galstyan, A. Multitask learning and benchmarking with clinical time series data. Scientific data , 6(1):96, 2019.
- Hayat, N., Geras, K. J., and Shamout, F. E. Towards dynamic multi-modal phenotyping using chest radiographs and physiological data. arXiv preprint arXiv:2111.02710 , 2021.
- Hayat, N., Geras, K. J., and Shamout, F. E. Medfuse: Multimodal fusion with clinical time-series data and chest x-ray images. In Machine Learning for Healthcare Conference , pp. 479-503. PMLR, 2022.
- He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition , pp. 770-778, 2016.
- Hofert, M. Right-truncated archimedean and related copulas. Insurance: Mathematics and Economics , 99:79-91, 2021.
- Huang, C., Tian, Y., Kumar, A., and Xu, C. Egocentric audio-visual object localization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 22910-22921, 2023.
- Huang, S.-C., Pareek, A., Seyyedi, S., Banerjee, I., and Lungren, M. P. Fusion of medical imaging and electronic health records using deep learning: a systematic review and implementation guidelines. NPJ digital medicine , 3 (1):136, 2020.
- Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., and Liu, Q. Tinybert: Distilling bert

- for natural language understanding. arXiv preprint arXiv:1909.10351 , 2019.
- Johnson, A. E., Pollard, T. J., Shen, L., Lehman, L.-w. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Anthony Celi, L., and Mark, R. G. Mimic-iii, a freely accessible critical care database. Scientific data , 3(1):1-9, 2016.
- Johnson, A. E., Pollard, T. J., Berkowitz, S. J., Greenbaum, N. R., Lungren, M. P., Deng, C.-y., Mark, R. G., and Horng, S. Mimic-cxr, a de-identified publicly available database of chest radiographs with free-text reports. Scientific data , 6(1):317, 2019.
- Johnson, A. E., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A., Horng, S., Pollard, T. J., Hao, S., Moody, B., Gow, B., et al. Mimic-iv, a freely accessible electronic health record dataset. Scientific data , 10(1):1, 2023.
- Joze, H. R. V., Shaban, A., Iuzzolino, M. L., and Koishida, K. Mmtm: Multimodal transfer module for cnn fusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pp. 13289-13299, 2020.
- Khader, F., Müller-Franzes, G., Wang, T., Han, T., Tayebi Arasteh, S., Haarburger, C., Stegmaier, J., Bressem, K., Kuhl, C., Nebelung, S., et al. Multimodal deep learning for integrating chest radiographs and clinical parameters: a case for transformers. Radiology , 309 (1):e230806, 2023.
- Liang, P. P., Cheng, Y., Fan, X., Ling, C. K., Nie, S., Chen, R., Deng, Z., Allen, N., Auerbach, R., Mahmood, F., et al. Quantifying &amp; modeling multimodal interactions: An information decomposition framework. Advances in Neural Information Processing Systems , 36, 2024.
- Ma, M., Ren, J., Zhao, L., Tulyakov, S., Wu, C., and Peng, X. Smil: Multimodal learning with severely missing modality. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 35, pp. 2302-2310, 2021.
- Marmin, S., Chevalier, C., and Ginsbourger, D. Differentiating the multipoint expected improvement for optimal batch design. In International workshop on machine learning, optimization and big data , pp. 37-48. Springer, 2015.
- Mo, S. and Tian, Y. Audio-visual grouping network for sound localization from mixtures. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 10565-10574, 2023.
- Nalisnick, E. T. On priors for Bayesian neural networks . University of California, Irvine, 2018.
- Ni, J., Cheng, W., Chen, Z., Asakura, T., Soma, T., Kato, S., and Chen, H. Superclass-conditional gaussian mixture model for learning fine-grained embeddings. In International Conference on Learning Representations , 2021.
- Pölsterl, S., Wolf, T. N., and Wachinger, C. Combining 3d image and tabular data via the dynamic affine feature map transform. In Medical Image Computing and Computer Assisted Intervention-MICCAI 2021: 24th International Conference, Strasbourg, France, September 27-October 1, 2021, Proceedings, Part V 24 , pp. 688-698. Springer, 2021.
- Rodriguez, J. C. Measuring financial contagion: A copula approach. Journal of empirical finance , 14(3):401-423, 2007.
- Salzmann, T., Pavone, M., and Ryll, M. Motron: Multimodal probabilistic human motion forecasting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 6457-6466, 2022.
- Silva, R. and Gramacy, R. Mcmc methods for bayesian mixtures of copulas. In Artificial Intelligence and Statistics , pp. 512-519. PMLR, 2009.
- Sklar, M. Fonctions de répartition à n dimensions et leurs marges. In Annales de l'ISUP , volume 8, pp. 229-231, 1959.
- Smith, M. S. and Loaiza-Maya, R. Implicit copula variational inference. Journal of Computational and Graphical Statistics , 32(3):769-781, 2023.
- Smith, M. S., Loaiza-Maya, R., and Nott, D. J. Highdimensional copula variational approximation through transformation. Journal of Computational and Graphical Statistics , 29(4):729-743, 2020.
- Song, A. H., Chen, R. J., Ding, T., Williamson, D. F., Jaume, G., and Mahmood, F. Morphological prototyping for unsupervised slide representation learning in computational pathology. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 1156611578, 2024.
- Song, X., Zhang, X., Ji, J., Liu, Y ., and Wei, P. Cross-modal contrastive attention model for medical report generation. In Proceedings of the 29th International Conference on Computational Linguistics , pp. 2388-2397, 2022.
- Tagasovska, N., Ackerer, D., and Vatter, T. Copulas as high-dimensional generative models: Vine copula autoencoders. Advances in neural information processing systems , 32, 2019.
- Theodorou, B., Glass, L., Xiao, C., and Sun, J. Framm: Fair ranking with missing modalities for clinical trial site selection. Patterns , 5(3), 2024.

- Tran, B.-H., Rossi, S., Milios, D., and Filippone, M. All you need is a good functional prior for bayesian deep learning. Journal of Machine Learning Research , 23(74): 1-56, 2022.
- Tran, D., Blei, D., and Airoldi, E. M. Copula variational inference. Advances in neural information processing systems , 28, 2015.
- Tran, L., Liu, X., Zhou, J., and Jin, R. Missing modalities imputation via cascaded residual autoencoder. In Proceedings of the IEEE conference on computer vision and pattern recognition , pp. 1405-1414, 2017.
- Trong, V. H., Gwang-hyun, Y., Vu, D. T., and Jin-young, K. Late fusion of multimodal deep neural networks for weeds classification. Computers and Electronics in Agriculture , 175:105506, 2020.
- Vaswani, A. Attention is all you need. Advances in Neural Information Processing Systems , 2017.
- Wang, H., Chen, Y., Ma, C., Avery, J., Hull, L., and Carneiro, G. Multi-modal learning with missing modality via shared-specific feature modelling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pp. 15878-15887, 2023.
- Wang, Y., Pillai, M., Zhao, Y., Curtin, C., and HernandezBoussard, T. Fairehr-clp: Towards fairness-aware clinical predictions with contrastive learning in multimodal electronic health records. arXiv preprint arXiv:2402.00955 , 2024.
- Wu, Z., Dadu, A., Tustison, N., Avants, B., Nalls, M., Sun, J., and Faghri, F. Multimodal patient representation learning with missing modalities and labels. In The Twelfth International Conference on Learning Representations , 2024a.
- Wu, Z., Dadu, A., Tustison, N., Avants, B., Nalls, M., Sun, J., and Faghri, F. Multimodal patient representation learning with missing modalities and labels. In The Twelfth International Conference on Learning Representations , 2024b.
- Yao, W., Yin, K., Cheung, W. K., Liu, J., and Qin, J. Drfuse: Learning disentangled representation for clinical multi-modal fusion with missing modality and modal inconsistency. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 38, pp. 16416-16424, 2024.
- Yoo, Y., Tang, L. Y., Li, D. K., Metz, L., Kolind, S., Traboulsee, A. L., and Tam, R. C. Deep learning of brain lesion patterns and user-defined clinical and mri features for predicting conversion to multiple sclerosis from clinically isolated syndrome. Computer Methods in Biomechanics
- and Biomedical Engineering: Imaging &amp; Visualization , 7 (3):250-259, 2019.
- Zeng, Z. and Wang, T. Neural copula: A unified framework for estimating generic high-dimensional copula functions. arXiv preprint arXiv:2205.15031 , 2022.
- Zhang, C., Chu, X., Ma, L., Zhu, Y., Wang, Y., Wang, J., and Zhao, J. M3care: Learning with missing modalities in multimodal healthcare data. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining , pp. 2418-2428, 2022.
- Zhang, X., Li, S., Chen, Z., Yan, X., and Petzold, L. R. Improving medical predictions by irregular multimodal electronic health records modeling. In International Conference on Machine Learning , pp. 41300-41313. PMLR, 2023.

## Summary

In this Appendix, we first present detailed information on the datasets in A.1 and tasks used in the experiments in A.2. Next, we introduce the multivariate Gaussian distribution in B and some common copula families in C. Then in D, we discuss the implications of how the copula model learns interactions over the epochs. Finally, we provide more details on the implementation and hyperparameters used in the experiments in E.1 along with the settings of baseline methods in E.2.

## A. Additional Information on Datasets and Tasks

## A.1. Datasets

Table 8 provides a summary of the datasets used in our experiments.

MIMIC-III dataset This dataset contains 46,520 ICU stays, each with 17 clinical variables. We split the dataset into training, validation, and test sets in the ratio of 70 : 15 : 15 , following the procedure in Harutyunyan et al. (2019).

MIMIC-IV dataset This dataset includes 21,139 ICU stays, also with 17 clinical variables. The dataset is split into training, validation, and test sets in the ratio of 70 : 10 : 20 , following Hayat et al. (2022).

For both MIMIC-III and MIMIC-IV datasets, we extract 17 clinical variables commonly monitored in the ICU, including 5 categorical and 12 continuous variables. Data are sampled every two hours during the first 48 hours of ICU admission for both tasks, in accordance with Hayat et al. (2022). This results in a vector representation of size 76 at each time step of the clinical time-series data.

MIMIC-CXR dataset This dataset contains 377,110 chest X-ray images, of which 5,931 are associated with MIMIC-IV ICU stays. We split the data into 4,287 training samples, 465 validation samples, and 1,179 test samples. Following Hayat et al. (2022), we retrieve the last Anterior-Posterior projection chest X-ray and apply transformations to the images, resizing them to 224 × 224 pixels.

This dataset also includes radiology reports, which are unstructured text data. We choose the radiology reports of the MIMIC-CXR dataset as an auxiliary modality to investigate the effectiveness of CM 2 on more modalities alignment since the radiology reports do not contain death information and can avoid possible overfitting and shortcuts. We divide the unstructured radiology reports into 4 sections, including Impression, Findings, Last paragraph, and Comparison.

MIMIC-III NOTE dataset This dataset consists of 5,273 clinical notes associated with MIMIC-III ICU stays. The dataset is divided into 3,652 training samples, 815 validation samples, and 806 test samples. In line with Zhang et al. (2023), we select the last five clinical notes before the prediction time. If fewer than five notes are available, we treat the notes for that ICU stay as missing. The original number of matched ICU stays is around 15,000. We randomly sample one-third of the matched ICU stays to form the training, validation, and test sets, keeping the scale of the notes nearly the same as the CXRs in the MIMIC-IV dataset.

Both radiology reports sections and clinical notes are capped at a maximum length of 512 words, tokenized into words, and embedded into 312-dimensional vectors using the pre-trained TinyBERT model (Jiao et al., 2019) 1 .

## A.2. Tasks

In-Hospital Mortality (IHM) Prediction. The In-Hospital Mortality (IHM) prediction task focuses on predicting whether a patient will pass away during their hospital stay. As summarized in Table 8, the MIMIC-III dataset contains a total of 2,795 positive samples, of which 736 are matched with clinical notes. Similarly, the MIMIC-IV dataset includes 3,153 positive samples, with 890 matched to CXR.

Readmission (READM) Prediction. The Readmission (READM) prediction task aims to forecast whether a patient will be readmitted within 30 days of discharge. In this task, both patients who are readmitted and those who pass away in hospital are considered positive samples. As shown in Table 8, the MIMIC-III dataset contains 3,987 positive samples, with 998 matched to clinical notes. In the MIMIC-IV dataset, there are 4,603 positive samples, with 1,262 matched to CXRs.

[1 https://huggingface.co/huawei-noah/TinyBERT\_General\_4L\_312D](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)

Table 8. Numbers of samples in training/validation/testing sets

| Datasets                          | Tasks             | Train             | Valid             | Test              | Pos.              | Total             |
|-----------------------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
|                                   | Complete Datasets | Complete Datasets | Complete Datasets | Complete Datasets | Complete Datasets | Complete Datasets |
| MIMIC-III                         | IHM               | 14681             | 3222              | 3236              | 2795              | 21139             |
| MIMIC-III                         | READM             | 14681             | 3222              | 3236              | 3987              | 21139             |
| MIMIC-III NOTE                    | -                 | 3652              | 815               | 806               | -                 | 5,273             |
| MIMIC-IV                          | IHM               | 18064             | 2035              | 4972              | 3153              | 25071             |
| MIMIC-IV                          | READM             | 18064             | 2035              | 4972              | 4603              | 25071             |
| MIMIC-CXR                         | -                 | 344529            | 9497              | 23069             | -                 | 377,095           |
|                                   | Matched Datasets  | Matched Datasets  | Matched Datasets  | Matched Datasets  | Matched Datasets  | Matched Datasets  |
| MIMIC-III &#124; NOTE             | IHM               | 3652              | 815               | 806               | 736               | 5273              |
| MIMIC-III &#124; NOTE             | READM             | 3652              | 815               | 806               | 998               | 5273              |
| MIMIC-IV &#124; CXR               | IHM               | 4287              | 465               | 1179              | 890               | 5931              |
| MIMIC-IV &#124; CXR               | READM             | 4287              | 465               | 1179              | 1262              | 5931              |
| MIMIC-IV &#124; CXR &#124; REPORT | IHM               | 4287              | 465               | 1179              | 890               | 5931              |
| MIMIC-IV &#124; CXR &#124; REPORT | READM             | 4287              | 465               | 1179              | 1262              | 5931              |

## B. Multivariate Gaussian Distribution

The multivariate Gaussian distribution is defined as

<!-- formula-not-decoded -->

where µ ∈ R p is a p -dimensional mean vector and Σ ∈ R p × p is the covariance matrix.

The KL divergence of two multivariate normal distributions N ( µ 1 , Σ 1 ) and N ( µ 2 , Σ 2 ) is

<!-- formula-not-decoded -->

## C. Common Copula Families.

We specify the copula distributions and density functions of common copula families with necessary derivations. Without loss of generality, we consider bivariate copula families.

Archimedean Copula. A subclass of copulas can be easily constructed by a generator function φ : [0 , 1] → [0 , ∞ ] , which is strictly decreasing and convex so that φ (0) = ∞ and φ (1) = 0 . Then, a copula C can be constructed as follows,

<!-- formula-not-decoded -->

The Archimedean copula can generate copula densities when more than one modality exist in the dataset.

## C.1. Copula Distribution Functions

- Clayton copula
- Frank copula

where α ∈ R \{ 0 } .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- Gumbel copula
- Gaussian copula

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Φ is the CDF of the standard Gaussian distribution, and Φ 2 is the bivariate Gaussian distribution.

- Student's t copula

<!-- formula-not-decoded -->

where T -1 ν is the inverse of the CDF of Student's t -distribtuion with degrees of freedom ν , and T 2 ,ν is the bivariate t -distribtuion with degrees of freedom ν .

## C.2. Copula Density Functions

## Clayton copula

where α ∈ ( -1 , ∞ ) .

## Frank copula

̸

where α ∈ ( -∞ , ∞ ) , α = 0 .

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## Gumbel copula

<!-- formula-not-decoded -->

The closed-form density of the trivariate Gumbel copula is computed by

<!-- formula-not-decoded -->

The identity can be generated by the Archimedean copula for M &gt; 3 , which is less common in multimodal learning,

<!-- formula-not-decoded -->

where φ ( t ; α ) = (log t ) α for the Gumbel copula, and ( φ -1 ) ′ is the first derivative of the inverse of φ .

Figure 4. Plots of the copula densities of the Gumbel family at epochs 5, 50, and 100, respectively.

<!-- image -->

Gaussian copula The bivariate case is given by

<!-- formula-not-decoded -->

where a = √ 2 erf -1 (2 u -1) , and b = √ 2 erf -1 (2 v -1) . The multivariate case is given by the following matrix form,

<!-- formula-not-decoded -->

where Σ is the covariance matrix and ζ = (Φ -1 ( u 1 ) , . . . , Φ -1 ( u M )) ⊤ , ζ = [ ζ 1 , . . . , ζ M ] .

## Student's t copula

<!-- formula-not-decoded -->

where v is the degree of freedom, Γ is the gamma function, and

<!-- formula-not-decoded -->

## D. How Copula Learns Interactions.

We demonstrate how the copula model learns the interactions over the epochs and further discuss the implications.

Figure 4 presents the copula densities at epochs epochs 5, 50, and 100, respectively. We use the Gumbel family as an illustrative example. We observe that the copula density is evolving to a positive correlation pattern, while the negative correlation scenarios (e.g., u &gt; 0 . 5 , v &lt; 0 . 5 , or u &lt; 0 . 5 , v &gt; 0 . 5 ) are still considered but the weights allocated are decreasing.

## E. More on Baseline Methods and Implementation Details

## E.1. Implementation Details and Hyperparameters

We train all models for 100 epochs on the training set and select the best-performing model based on the validation set, using the AUROC as the monitoring metric. The final results are reported on the test set. We optimize the models using the Adam optimizer and apply early stopping if the validation AUROC does not improve for 15 consecutive epochs to prevent overfitting. All experiments are conducted on a single RTX-3090 GPU. The batch size is set to 32 for models trained on the MIMIC-IV &amp; CXR datasets, and 16 for models trained on the MIMIC-III &amp; NOTE datasets, except for DrFuse, which is trained with a batch size of 8. We employ grid search to tune hyperparameters using the validation set and report the best results on the test set. The hyperparameter search space includes:

- Dropout ratio: { 0 , 0 . 1 , 0 . 2 , 0 . 3 }
- Learning rate: { 1 × 10 -4 , 5 × 10 -5 , 1 × 10 -5 }
- Number of Gaussian mixtures K : { 1 , 2 , 3 , 4 , 5 , 6 }
- Temperature: { 0 . 001 , 0 . 005 , 0 . 01 , 0 . 05 , 0 . 08 }
- Regularization parameter λ cop : { 1 × 10 -5 , 5 × 10 -6 , 1 × 10 -6 }

CM 2 is implemented in Python 3.11 using PyTorch 1.9. Following MedFuse (Hayat et al., 2022), we use ResNet34 (He et al., 2016) as the backbone encoder for CXR, a two-layer LSTM (Graves &amp; Graves, 2012) as the encoder for time-series data, and pre-trained TinyBERT (Jiao et al., 2019) 2 as the encoder for clinical notes. We include a projection layer to map modality embeddings into the same latent space. A two-layer LSTM is used as the fusion module to combine modality embeddings, and a multilayer perceptron (MLP) with one linear layer and a sigmoid activation function serves as the classifier.

## E.2. Additional Settings of Baseline Methods

We compare CM 2 with the following baseline methods.

- MMTM (Joze et al., 2020) is a module that can leverage the information between modalities with flexible plugin architectures. Since the model assumes full modality, we compensate for the missing modality CXR and clinical notes with all zeros during training and testing. For clinical notes, we replace the ResNet34 encoder with TinyBERT to embed the clinical notes.
- DAFT (Pölsterl et al., 2021) is a module that can be plugged into CNN models to exchange information between tabular data and image modality. Similarly, we replace the input of CXR and clinical notes with matrices of all zeros during training and testing and use TinyBERT to embed the clinical notes.
- Unified (Hayat et al., 2021) is a dynamic approach towards integrating auxiliary data modalities, learning the data representations for the individual modalities, and integrating the representations via a unified classifier. It inherently handles missingness and leverages all of the available modality-specific data. Also, we use TinyBERT to embed the clinical notes.
- MedFuse (Hayat et al., 2022) uses an LSTM-based fusion to combine features from the image encoder (or language encoder) and EHR encoder. Missing modality is handled by learning a global representation for the missing CXR or clinical notes. We randomly initialized encoders for the time-series data, clinical notes, and CXR images.
- DrFuse (Yao et al., 2024) uses disentangled representation learning to learn a shared representation between the EHR and image modality even when one modality is missing. Drfuse uses ResNet50 as the image encoder and Transformer as the EHR encoder. We replace the ResNet50 encoder with TinyBERT to embed the clinical notes.

The Implementation of DrFuse follows the original paper(Yao et al., 2024) 3 , and we use the same hyperparameters as the original paper. We directly adopt the implementations of MMTM, DAFT, Unified, and MedFuse provided by (Hayat et al., 2022) 4 , and all hyperparameters are set to the default values provided by Hayat et al. (2022). We adapt the implementations of MMTM, DAFT, Unified, MedFuse and DrFuse to tri-modal setting, including EHR time-series data, CXR images, and radiology reports.

## F. Additional Experiment Results

Additional Baselines. We compare CM 2 to two additional healthcare baselines: LSMT (Khader et al., 2023) and Interleaved (Zhang et al., 2023). The results are shown in Table 9.

[2 https://huggingface.co/huawei-noah/TinyBERT\_General\_4L\_312D](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)

[3 https://github.com/dorothy-yao/drfuse](https://github.com/dorothy-yao/drfuse)

[4 https://github.com/nyuad-cai/MedFuse](https://github.com/nyuad-cai/MedFuse)

Table 9. Results of additional baselines on the MIMIC-IV dataset. All results are reported in AUROC and AUPR with 95% confidence intervals. The best results are highlighted in boldface.

|                                                                  | IHM                                                                           | IHM                                                                           | READM                                                                         | READM                                                                         |
|------------------------------------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| Models                                                           | AUROC ( ↑ )                                                                   | AUPR ( ↑ )                                                                    | AUROC ( ↑ )                                                                   | AUPR ( ↑ )                                                                    |
|                                                                  | Totally Matched                                                               | Totally Matched                                                               | Totally Matched                                                               | Totally Matched                                                               |
| LSMT (Khader et al., 2023) Interleaved (Zhang et al., 2023) CM 2 | 0.803 (0 . 769 , 0 . 837) 0.800 (0 . 764 , 0 . 834) 0.827 (0 . 790 , 0 . 859) | 0.444 (0 . 370 , 0 . 519) 0.440 (0 . 374 , 0 . 523) 0.492 (0 . 423 , 0 . 566) | 0.701 (0 . 662 , 0 . 737) 0.702 (0 . 664 , 0 . 741) 0.737 (0 . 704 , 0 . 773) | 0.421 (0 . 356 , 0 . 490) 0.421 (0 . 360 , 0 . 487) 0.466 (0 . 404 , 0 . 529) |
|                                                                  | Partially Matched                                                             | Partially Matched                                                             | Partially Matched                                                             | Partially Matched                                                             |
| LSMT (Khader et al., 2023) Interleaved (Zhang et al., 2023) CM 2 | 0.854 (0 . 838 , 0 . 870) 0.856 (0 . 840 , 0 . 871) 0.858 (0 . 844 , 0 . 872) | 0.508 (0 . 466 , 0 . 551) 0.508 (0 . 466 , 0 . 550) 0.527 (0 . 490 , 0 . 568) | 0.764 (0 . 746 , 0 . 781) 0.758 (0 . 740 , 0 . 775) 0.771 (0 . 752 , 0 . 788) | 0.473 (0 . 436 , 0 . 509) 0.473 (0 . 441 , 0 . 506) 0.486 (0 . 452 , 0 . 518) |

<!-- image -->

Figure 5. Results (left: AUROC; right: AUPR) of CM 2 on MIMIC-IV, where the model reduces to a multivariate Gaussian disdtribution when K = 1 .

<!-- image -->

- LSMT (Khader et al., 2023) is a transformer-based model designed for the multimodal medical context.
- Interleaved (Zhang et al., 2023) is a multimodal approach that addresses the irregularity of medical multimodal data and fuses representations from different modalities using cross-modal attention.

Effect of Backbone Encoders. Moreover, we explore the effectiveness of backbone encoders for both time-series data and CXR image data. We conduct additional experiments to evaluate the impact of different encoder architectures for each modality. Specifically, we use the Transformer (Vaswani, 2017) and ViT (Dosovitskiy, 2020) as alternative backbone encoders for the time-series and CXR image data, respectively. The results are shown in Table 10. We observe that our method consistently outperforms competitive baselines across various backbone encoders, highlighting its robustness and effectiveness. Furthermore, our method demonstrates greater stability across different backbones, suggesting it is less sensitive to their selection. Besides, the Transformer backbone generally outperforms the LSTM backbone, particularly for MMTM, LSMT, and Interleaved. While the ResNet backbone slightly outperforms the ViT backbone, the performance difference is not substantial, suggesting time-series data's greater impact on backbone encoder choice.

Effect of Number of Mixtures K . As a convention in statistical modelling, K is set to be small to avoid over-specification. The popular choice of K is 2 to 3 such that the learned mixture distribution can achieve an optimal degree of flexibility while preventing over-specification. We evaluate how the performance of CM 2 varies with different values of K , as shown in Figure 5. We observe that the performance is quite robust.

Statistical Tests The p-values of two-sample bootstrapped t -tests of the AUROC and AUPR of CM 2 compared to baseline methods are shown in Table 11. We observe that the improvements over the competitive baselines are overall statistically significant under the 5% significance level, validating the effectiveness of our method.

Table 10. Results of different backbone encoders and additional baselines on MIMIC-IV with totally matched modalities. All results are reported in AUROC and AUPR with 95% confidence intervals. The best results are highlighted in boldface .

|                                                                                                                                                                                                           | Backbone    | Backbone   | IHM                                                                                                                                                                                                             | IHM                                                                                                                                                                                                             | READM                                                                                                                                                                                                           | READM                                                                                                                                                                                                           |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Models                                                                                                                                                                                                    | TS          | IMG        | AUROC ( ↑ )                                                                                                                                                                                                     | AUPR ( ↑ )                                                                                                                                                                                                      | AUROC ( ↑ )                                                                                                                                                                                                     | AUPR ( ↑ )                                                                                                                                                                                                      |
| MMTM(Joze et al., 2020) DAFT (Pölsterl et al., 2021) Unified (Hayat et al., 2021) MedFuse (Hayat et al., 2022) DrFuse (Yao et al., 2024) LSMT (Khader et al., 2023) Interleaved (Zhang et al., 2023) CM 2 | LSTM        | ResNet     | 0.802 (0 . 770 , 0 . 835) 0.815 (0 . 782 , 0 . 844) 0.808 (0 . 778 , 0 . 840) 0.813 (0 . 777 , 0 . 844) 0.814 (0 . 780 , 0 . 844) 0.803 (0 . 769 , 0 . 837) 0.800 (0 . 764 , 0 . 834) 0.827 (0 . 790 , 0 . 859) | 0.429 (0 . 362 , 0 . 513) 0.454 (0 . 387 , 0 . 538) 0.429 (0 . 367 , 0 . 512) 0.448 (0 . 380 , 0 . 528) 0.450 (0 . 384 , 0 . 536) 0.444 (0 . 374 , 0 . 523) 0.440 (0 . 370 , 0 . 519) 0.492 (0 . 423 , 0 . 566) | 0.713 (0 . 677 , 0 . 750) 0.729 (0 . 692 , 0 . 766) 0.719 (0 . 680 , 0 . 756) 0.725 (0 . 690 , 0 . 762) 0.723 (0 . 687 , 0 . 756) 0.701 (0 . 662 , 0 . 737) 0.702 (0 . 664 , 0 . 741) 0.737 (0 . 704 , 0 . 773) | 0.420 (0 . 362 , 0 . 489) 0.433 (0 . 378 , 0 . 499) 0.450 (0 . 390 , 0 . 513) 0.438 (0 . 379 , 0 . 508) 0.422 (0 . 367 , 0 . 486) 0.421 (0 . 356 , 0 . 490) 0.421 (0 . 360 , 0 . 487) 0.466 (0 . 404 , 0 . 529) |
| MMTM(Joze et al., 2020) DAFT (Pölsterl et al., 2021) Unified (Hayat et al., 2021) MedFuse (Hayat et al., 2022) DrFuse (Yao et al., 2024) LSMT (Khader et al., 2023) Interleaved (Zhang et al., 2023) CM 2 | LSTM        | ViT        | 0.805 (0 . 768 , 0 . 837) 0.808 (0 . 775 , 0 . 840) 0.803 (0 . 768 , 0 . 835) 0.805 (0 . 771 , 0 . 837) 0.806 (0 . 772 , 0 . 838) 0.801 (0 . 767 , 0 . 836) 0.802 (0 . 766 , 0 . 833) 0.826 (0 . 790 , 0 . 856) | 0.446 (0 . 377 , 0 . 524) 0.438 (0 . 365 , 0 . 521) 0.431 (0 . 365 , 0 . 515) 0.439 (0 . 371 , 0 . 524) 0.446 (0 . 379 , 0 . 526) 0.441 (0 . 374 , 0 . 527) 0.434 (0 . 364 , 0 . 509) 0.490 (0 . 421 , 0 . 563) | 0.712 (0 . 676 , 0 . 749) 0.714 (0 . 678 , 0 . 753) 0.707 (0 . 667 , 0 . 743) 0.715 (0 . 677 , 0 . 753) 0.716 (0 . 677 , 0 . 748) 0.703 (0 . 662 , 0 . 739) 0.710 (0 . 673 , 0 . 747) 0.736 (0 . 697 , 0 . 771) | 0.422 (0 . 360 , 0 . 491) 0.423 (0 . 369 , 0 . 490) 0.416 (0 . 360 , 0 . 482) 0.424 (0 . 370 , 0 . 492) 0.421 (0 . 364 , 0 . 489) 0.410 (0 . 358 , 0 . 475) 0.435 (0 . 372 , 0 . 502) 0.452 (0 . 394 , 0 . 522) |
| MMTM(Joze et al., 2020) DAFT (Pölsterl et al., 2021) Unified (Hayat et al., 2021) MedFuse (Hayat et al., 2022) DrFuse (Yao et al., 2024) LSMT (Khader et al., 2023) Interleaved (Zhang et al., 2023) CM 2 | Transformer | ResNet     | 0.813 (0 . 780 , 0 . 846) 0.814 (0 . 782 , 0 . 845) 0.812 (0 . 776 , 0 . 845) 0.815 (0 . 782 , 0 . 846) 0.818 (0 . 784 , 0 . 850) 0.817 (0 . 785 , 0 . 848) 0.821 (0 . 791 , 0 . 851) 0.823 (0 . 788 , 0 . 855) | 0.452 (0 . 383 , 0 . 540) 0.437 (0 . 373 , 0 . 522) 0.453 (0 . 385 , 0 . 533) 0.441 (0 . 373 , 0 . 520) 0.460 (0 . 391 , 0 . 540) 0.452 (0 . 386 , 0 . 535) 0.459 (0 . 389 , 0 . 539) 0.488 (0 . 421 , 0 . 560) | 0.735 (0 . 699 , 0 . 770) 0.730 (0 . 694 , 0 . 766) 0.719 (0 . 681 , 0 . 754) 0.728 (0 . 692 , 0 . 762) 0.726 (0 . 689 , 0 . 760) 0.722 (0 . 688 , 0 . 758) 0.721 (0 . 683 , 0 . 757) 0.740 (0 . 699 , 0 . 771) | 0.448 (0 . 388 , 0 . 515) 0.430 (0 . 372 , 0 . 493) 0.426 (0 . 365 , 0 . 488) 0.442 (0 . 381 , 0 . 505) 0.430 (0 . 370 , 0 . 495) 0.431 (0 . 376 , 0 . 494) 0.429 (0 . 367 , 0 . 497) 0.470 (0 . 382 , 0 . 510) |
| MMTM(Joze et al., 2020) DAFT (Pölsterl et al., 2021) Unified (Hayat et al., 2021) MedFuse (Hayat et al., 2022) DrFuse (Yao et al., 2024) LSMT (Khader et al., 2023) Interleaved (Zhang et al., 2023) CM 2 | Transformer | ViT        | 0.813 (0 . 778 , 0 . 846) 0.803 (0 . 768 , 0 . 836) 0.812 (0 . 778 , 0 . 845) 0.818 (0 . 786 , 0 . 849) 0.814 (0 . 780 , 0 . 845) 0.815 (0 . 784 , 0 . 847) 0.818 (0 . 786 , 0 . 849) 0.826 (0 . 790 , 0 . 855) | 0.462 (0 . 396 , 0 . 545) 0.432 (0 . 363 , 0 . 510) 0.463 (0 . 396 , 0 . 546) 0.461 (0 . 393 , 0 . 542) 0.436 (0 . 369 , 0 . 516) 0.453 (0 . 389 , 0 . 535) 0.453 (0 . 380 , 0 . 531) 0.489 (0 . 422 , 0 . 560) | 0.723 (0 . 686 , 0 . 761) 0.719 (0 . 682 , 0 . 758) 0.719 (0 . 680 , 0 . 753) 0.721 (0 . 684 , 0 . 759) 0.717 (0 . 680 , 0 . 755) 0.714 (0 . 675 , 0 . 751) 0.717 (0 . 679 , 0 . 753) 0.737 (0 . 700 , 0 . 772) | 0.435 (0 . 380 , 0 . 505) 0.421 (0 . 367 , 0 . 486) 0.412 (0 . 353 , 0 . 474) 0.431 (0 . 371 , 0 . 493) 0.416 (0 . 359 , 0 . 480) 0.424 (0 . 365 , 0 . 492) 0.433 (0 . 371 , 0 . 498) 0.465 (0 . 394 , 0 . 517) |

Table 11. P-values of two-sample bootstrapped t -tests of the AUROC and AUPR of CM 2 compared to baseline methods. Most of the tests are significant under the 5% significance level.

| Models                       | IHM         | IHM        | READM       | READM      |
|------------------------------|-------------|------------|-------------|------------|
|                              | AUROC ( ↑ ) | AUPR ( ↑ ) | AUROC ( ↑ ) | AUPR ( ↑ ) |
| MMTM(Joze et al., 2020)      | 2.02e-06    | 3.55e-180  | 4.40e-100   | 5.36e-291  |
| DAFT (Pölsterl et al., 2021) | 0.1122      | 1.53e-132  | 9.37e-78    | 2.95e-240  |
| Unified (Hayat et al., 2021) | 4.55e-08    | 5.71e-240  | 4.80e-73    | 2.81e-139  |
| MedFuse (Hayat et al., 2022) | 7.73e-07    | 5.66e-129  | 1.11e-92    | 3.69e-173  |
| DrFuse (Yao et al., 2024)    | 0.1447      | 4.28e-99   | 6.05e-67    | 6.25e-250  |