## TSINR: Capturing Temporal Continuity via Implicit Neural Representations for Time Series Anomaly Detection

[Mengxuan Li](https://orcid.org/0001-7278-7891)

Zhejiang University

Hangzhou, China

mengxuanli@intl.zju.edu.cn

Jiajun Bu

Zhejiang University

Hangzhou, China bjj@zju.edu.cn

## ABSTRACT

Time series anomaly detection aims to identify unusual patterns in data or deviations from systems' expected behavior. The reconstruction-based methods are the mainstream in this task, which learn point-wise representation via unsupervised learning. However, the unlabeled anomaly points in training data may cause these reconstruction-based methods to learn and reconstruct anomalous data, resulting in the challenge of capturing normal patterns. In this paper, we propose a time series anomaly detection method based on implicit neural representation (INR) reconstruction, named TSINR, to address this challenge. Due to the property of spectral bias, TSINR enables prioritizing low-frequency signals and exhibiting poorer performance on high-frequency abnormal data. Specifically, we adopt INR to parameterize time series data as a continuous function and employ a transformer-based architecture to predict the INR of given data. As a result, the proposed TSINR method achieves the advantage of capturing the temporal continuity and thus is more sensitive to discontinuous anomaly data. In addition, we further design a novel form of INR continuous function to learn interand intra-channel information, and leverage a pre-trained large language model to amplify the intense fluctuations in anomalies. Extensive experiments demonstrate that TSINR achieves superior overall performance on both univariate and multivariate time series anomaly detection benchmarks compared to other state-of-the-art reconstruction-based methods. Our codes are available here.

## CCS CONCEPTS

· Computing methodologies → Neural networks ; · Mathematics of computing → Time series analysis .

*Corresponding authors.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

KDD '25, August 3-7, 2025, Toronto, ON, Canada

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 978-x-xxxx-xxxx-x/YY/MM

https://doi.org/3690624.3709266

Ke Liu

Zhejiang University

Hangzhou, China keliu99@zju.edu.cn

Hongwei Wang ∗

Zhejiang University

Hangzhou, China

hongweiwang@intl.zju.edu.cn

## KEYWORDS

time series anomaly detection, implicit neural representations, unsupervised learning

## ACMReference Format:

MengxuanLi, Ke Liu, Hongyang Chen, Jiajun Bu, Hongwei Wang, and Haishuai Wang. 2025. TSINR: Capturing Temporal Continuity via Implicit Neural Representations for Time Series Anomaly Detection. In Proceedings of the 31th ACMSIGKDD Conference on Knowledge Discovery and Data Mining V.1 (KDD '25). ACM, New York, NY, USA, 12 pages. https://doi.org/3690624.3709266

## 1 INTRODUCTION

Time series anomaly detection, which aims to identify unusual patterns or events across a sequence of data points collected over time [17], has attracted a lot of attention recently. In many fields (e.g., finance, healthcare, manufacturing, and fault diagnosis), monitoring time-varying data to identify anomalies is crucial for detecting unusual behavior, potential issues, or security threats [12, 19, 51, 52, 54]. For example, in the finance field, detecting anomalous behavior in credit card data allows the prevention of theft or fraudulent transactions committed by an unauthorized party [43]. In industrial processes, identifying anomalies helps secure safe operations, averting safety concerns and mitigating economic losses [20].

One of the major challenges for anomaly detection lies in anomalies may be rare, subtle, or have different shapes, requiring sophisticated algorithms to distinguish them from normal patterns. Moreover, time series typically exhibit trends, seasonality, and temporal dependencies, making it challenging to model such complex features. Since anomalies are typically rare and new anomalies may arise, it is difficult or expensive to collect a sufficient amount of labeled data. As one of the unsupervised-based methods, reconstruction-based methods tackle this problem by reconstructing data to learn point-wise feature representations to uncover normal patterns in the data [66, 67]. The reconstruction error, i.e., the difference between the input data and its reconstructed version, serves as a natural anomaly score. The points that have high anomaly scores are considered as anomalies, which makes it easy to interpret and understand the results.

However, it is challenging to distinguish normal and anomaly patterns because normal and anomalous points may coexist within a single instance, and anomalies may occur in the unlabeled training data [62]. In addition, time series data often contains intricate

Hongyang Chen Zhejiang Lab Hangzhou, China hongyang@zhejianglab.com

Haishuai Wang ∗ Zhejiang University Hangzhou, China

haishuai.wang@zju.edu.cn patterns, and anomalies might exhibit subtle deviations from normal ones. Therefore, models may be forced to learn and reconstruct anomalous data, this makes learning a reconstruction model that can effectively capture the normal pattern challenging. This issue is also pointed out in previous work [62]. Recently, implicit neural representation (INR) has become a powerful tool for continuous encoding of various signals by fitting continuous functions [22, 28]. Figure 1a depicts the diagram of INR within the context of time series data. As a continuous function, INR captures the temporal continuity of the time series, where the input is a timestamp, and the output is the corresponding value of this timestamp. In line with existing reconstruction-based anomaly detection methods, INR is also learned through a reconstruction task, making it inherently feasible for time series anomaly detection. In addition, INR possesses a spectral bias property, enabling it to prioritize the learning of low-frequency signals. Most efforts aim to alleviate this property to enhance the fitting capability for high-frequency signals [44, 47]. Conversely, this property is advantageous for accomplishing time series anomaly detection. As in Figure 1b, the spectral bias property enables INR to prioritize the smooth normal points and exhibit poorer performance for high-frequency abnormal data. Therefore, we aim to leverage the ability of INR to capture continuous representations and its sensitivity to anomalous data, thereby addressing the challenges of existing reconstruction-based methods.

Figure 1: (a) The diagram of INR for time series data. (b) The spectral bias property of INR to prioritize the low-frequency signals is advantageous for accomplishing time series data anomaly detection tasks.

<!-- image -->

In this paper, we propose a time series anomaly detection method based on INR reconstruction (TSINR for short). Specifically, we introduce a transformer-based architecture to predict the INR parameters of the given time series data. To better learn and reconstruct time series data, we design a novel form of continuous function to decompose time series [5, 7]. The designed function mainly comprises three components and individually learns the trend, seasonal, and residual information of time series. In addition, to further enhance the capability of INR to capture inter- and intra-channel information, we propose a group-based architecture to specifically learn the complex residual information. Simultaneously, we leverage a pre-trained large language model (LLM) to encode the original data to the feature domain. This encoding process particularly amplifies the fluctuations of anomalies across both the time and channel dimensions. By doing so, we enable TSINR to more effectively distinguish between normal and abnormal data points. The major contributions of this paper are summarized as follows:

- We utilize the spectral bias property of INR to prioritize fitting low-frequency signals and enhance sensitivity to discontinuous anomalies, thereby improving anomaly detection
- performance. A transformer-based architecture is employed to generate the parameters for INR, requiring only a single forward step in the inference phase.
- We design a novel form of INR continuous function, which mainly consists of three components to implicitly learn the unique trend, seasonal, and residual information of time series. Furthermore, a group-based strategy is proposed to further learn intricate residual information.
- We leverage a pre-trained LLM to encode the original time series to the feature domain, enabling amplification of the fluctuations of anomalies in both time and channel domains that facilitate INR to be further sensitive for noncontinuous anomaly areas. Ablation studies and visual analysis validate the aforementioned capacity to better distinguish anomaly points via our proposed framework.
- Extensive experiments demonstrate the overall effectiveness of TSINR compared with other state-of-the-art methods on seven multivariate and one univariate time series anomaly detection benchmark datasets.

## 2 RELATED WORK

## 2.1 Time Series Anomaly Detection

Time series anomaly detection methods primarily include statistical, classic machine learning, and deep learning methods. Statistical methods rely on analyzing the statistical properties of the data to identify patterns that deviate from the expected behavior. They are valuable for their simplicity and interpretability, but have limitations in capturing complex patterns [5, 12].

Classic machine learning methods rely on manual feature extraction and various algorithms like clustering [39, 42], density estimation [2, 61, 73], and isolation forests [21] to identify anomalies in structured data. However, because they require manual feature extraction and selection, they can be labor-intensive and less effective at capturing complex patterns in data.

Deep learning methods automatically learn the features of data through deep neural networks without the need for manual intervention, and are adept at handling high-dimensional, unstructured data. They can be broadly categorized into supervised and unsupervised learning algorithms. Supervised methods are trained with labels to learn and classify both normal and anomalous behavior in the given time series data, such as NFAD [40] and MultiHMM [18]. However, annotating data is challenging due to the rarity of anomalies and the emergence of new anomalies. This makes it difficult to achieve effective labeling, leading to limitations in the performance of supervised methods in detecting anomalies.

Figure 2: The overall workflow of the proposed TSINR method. The INR tokens predicted by the transformer encoder are the parameters of the INR continuous function. And the input of the INR continuous function is the timestamp 𝑡 .

<!-- image -->

On the contrary, unsupervised methods distinguish anomalous points from normal ones without relying on prior knowledge. The unsupervised methods mainly comprise 2 categories: forecastingbased methods and re-construction-based methods. Forecastingbased methods train a model to predict future values based on past observations and then identify anomalies by comparing the actual values with the predicted ones, such as ARIMA [60] and Telemanom [9]. Reconstruction-based methods learn and reconstruct the input data and then identify anomalies based on the difference between the original and reconstructed data, such as LSTM-VAE [32], BeatGAN [68], OmniAnomaly [46], and TranAD [48].

Recently, several methods have been proposed to establish a universal framework that can effectively address a wide range of time series data tasks, such as FPT [72] and TimesNet [55]. Among these works, models designed for prediction tasks also perform well in anomaly detection tasks, such as DLinear [63], ETSformer [53] and LightTS [65]. These methods demonstrate outstanding performance in learning time series features, thereby it makes sense to use these methods as baselines.

## 2.2 Implicit Neural Representations

Presently, INR [58] stands as a scorching topic in the domain of deep learning. It aims to learn a continuous function, often embodied as a neural network, for data representation. In this function, the input comprises coordinates, while the output consists of corresponding data values. INR learns continuous representations and has been widely applied in numerous scenarios, such as 2D image generation [34, 45, 70] and 3D scene reconstruction [8, 15, 38], physicsinformed problems [33, 37] and video representation [4, 25, 26].

Current methods for learning INR parameters predominantly rely on two main approaches: meta-learning [22] and feed-forward networks [3, 64]. The primary distinction between these approaches lies in how they handle test data. Meta-learning techniques are designed to quickly adapt to unseen test data by requiring only a few training steps or fine-tuning iterations for each new input. This allows for efficient generalization to novel tasks with minimal data. In contrast, feed-forward networks directly generate predictions in a single forward pass, leveraging pre-trained parameters without the need for task-specific adaptation during inference. In this paper, weuseatransformer-based architecture to generate INR parameters and it requires only a single forward in the inference phase [3].

In addition, there exists a phenomenon known as spectral bias [36], where INR tends to prefer fitting the low-frequency components of the signal [22]. Since this characteristic can affect the ability of INR to model high-frequency data, most efforts are directed towards mitigating this effect [23, 44, 47]. In contrast, in time series anomaly detection, this property turns out to be advantageous. In time series data, normal points exhibit relative smoothness, whereas anomalous points possess strong discontinuity. Hence, we leverage this property of INR to prioritize fitting normal data with low frequencies, making it more sensitive to anomalous data.

## 2.3 Implicit Neural Representations on Time Series Data

Currently, there are some studies discussing the possibility of utilizing INR for time series representation. HyperTime [7] leverages INR to learn a compressed latent representation for time series imputation and generation. TimeFlow [29] uses INR to capture continuous information for time series imputation and forecasting. In addition, the potential of employing INR for time series anomaly detection has not been fully explored yet. Only INRAD [10] attempts to adopt INR to represent and reconstruct time series data to identify anomalies. INRAD aims to utilize INR to overcome deep learning limitations, like complex computations and excessive hyperparameters (e.g., sliding windows). However, it requires training an INR network for each unseen time series data in test set, leading to additional training time and inefficiency in practical applications.

Different from INRAD, our method uses the spectral bias property of INR to mitigate the impact of unlabeled anomalies for the reconstruction model. With the Transformer integration, it can detect anomalies in unseen test data without retraining, enhancing efficiency for practical applications. In addition, compared to INRAD and other INR methods, our approach incorporates several specialized designs for time series anomaly detection. Firstly, we devise a novel form of INR continuous function to capture trend, seasonal, and residual information to address unique temporal patterns. Secondly, to handle multivariate time series, we introduce a group-based architecture to bolster the representational capacity of INR. Lastly, we leverage LLM to enhance anomaly detection by amplifying anomaly fluctuations, thereby boosting the sensitivity of INR to anomalies.

## 3 METHODOLOGY

In this paper, we propose TSINR, a novel time series anomaly detection method based on INR reconstruction. The core idea is to leverage the spectral bias phenomenon of INR to prioritize fitting smooth normal points, thereby enhancing sensitivity to discontinuous anomalous points. In this section, we present the problem statement and introduce the overall architecture, followed by the form of INR continuous function designed for time series data and the frozen pre-trained LLM encoder applied to amplify the fluctuations of anomalies from both time and channel dimensions. The anomaly criterion is demonstrated finally.

## 3.1 Problem Statement

Consider a time series 𝑋 with 𝑇 timestamps: 𝑋 = ( 𝑥 1 , 𝑥 2 , · · · , 𝑥 𝑇 ) , where 𝑥 𝑡 ∈ R 𝑑 is the data point observed at a certain timestamp 𝑡 ( 𝑡 ∈ { 1 , 2 , . . . , 𝑇 } ) and 𝑑 denotes the number of the data variables ( i.e. , data dimensionality). For a multivariate data, 𝑑 &gt; 1. And for an univariate case, 𝑑 = 1. Given unlabeled input time series data 𝑋 𝑡𝑟𝑎𝑖𝑛 , for any unknown time series data 𝑋 𝑡𝑒𝑠𝑡 with the same data dimensionality 𝑑 as 𝑋 𝑡𝑟𝑎𝑖𝑛 , we aim to predict 𝑦 𝑡𝑒𝑠𝑡 = ( 𝑦 1 , 𝑦 2 , · · · , 𝑦 𝑇 ′ ) , where 𝑇 ′ is the length of 𝑋 𝑡𝑒𝑠𝑡 . And 𝑦 𝑡 ′ ∈ { 0 , 1 } denotes whether the data point is normal ( 𝑦 𝑡 ′ = 0) or abnormal ( 𝑦 𝑡 ′ = 1) at the certain timestamp 𝑡 ′ ( 𝑡 ′ ∈ { 1 , 2 , . . . , 𝑇 ′ } ).

## 3.2 Overall Workflow

Figure 2 shows the overall workflow of the proposed TSINR method. We employ a feed-forward transformer-based architecture to directly predict the whole weights of the INR of the given time series data [3]. Unlike meta-learning based on gradient descent [22], our method requires only a single forward step in the inference phase. Following a strategy similar to other transformer-based methods [6, 63], the input time series data is normalized and segmented into patches. A frozen pre-trained LLM encoder is applied to map the input data into the feature domain to amplify the fluctuations of anomalies. Then the obtained features are converted to data tokens using a fully connected (FC) layer. Simultaneously, we initialize the corresponding INR tokens, which are learnable vector parameters. These data tokens and initialized INR tokens are fed together into a transformer encoder, which mainly consists of self-attention modules and feed-forward modules. In this transformer encoder, the knowledge interacts with data tokens and INR tokens. The learned INR tokens are mapped to the INR weights through FCs, denoted as FC ∗ . These INR weights form our INR continuous function, which is specifically designed for time series data. The designed function takes a batch of timestamps { 𝑡 } 𝑇 𝑖 = 1 as input, implicitly learns the trends, seasonality, and residual information of the given time series data, and finally reconstructs the input signal. The details of the designed form of INR continuous function and the applied frozen pre-trained LLM encoder module are demonstrated in Section 3.3 and Section 3.4. The anomaly criterion is introduced in Section 3.5.

Figure 3: A sample of the proposed group-based architecture. It contains 2 global layers and 2 group layers. In this case, each group corresponds to one variable.

<!-- image -->

## 3.3 Form of INR Continuous Function

As shown in Figure 2, we innovatively propose a INR continuous function to better learn and reconstruct time series data. Inspired by classical time series decomposition methods [5, 7], the proposed INR continuous function 𝑓 consists of three components, including trend 𝑓 𝑡𝑟 , seasonal 𝑓 𝑠 , and residual 𝑓 𝑟 :

<!-- formula-not-decoded -->

The trend component captures the underlying long-term patterns and focuses on slowly varying behaviors. In order to model this monotonic function, a polynomial predictor is applied [7, 30]:

<!-- formula-not-decoded -->

where 𝒘 ( 𝑖 ) 𝑡𝑟 is the polynomial coefficients corresponding to the 𝑖 𝑡ℎ degree and 𝒘 ( 𝑖 ) 𝑡𝑟 is predicted by a FC network. In addition, 𝑝 denotes the polynomial degree and is set to be small in order to model the low-frequency information and mimic the trend.

The seasonal component grasps the regular, cyclical, and recurring short-term fluctuations. Therefore, a periodic function is employed based on Fourier series [7, 30]:

<!-- formula-not-decoded -->

where 𝒘 𝑠 are Fourier coefficients learned by a FC network. This component is then able to model the periodic information and simulate typical seasonal patterns.

The residual component aims to represent the unexplained variability in the data after accounting for the trend and seasonal components. In order to capture this complex and non-periodic information, we design a group-based architecture as shown in Figure 3. For any given timestamp 𝑡 , we design 𝑀 global layers and 𝑁 group layers with 𝑘 groups. The global layers capture the inter-channel information while the group layers focus on intra-channel information. The equation for calculation within global layers can be defined as:

<!-- formula-not-decoded -->

where 𝑚 ∈ [ 0 , 𝑀 ) , and 𝑞 𝑚 , 𝒘 ( 𝑚 ) 𝑟 , 𝒃 ( 𝑚 ) 𝑟 denote the outputs, weights, and biases of the 𝑚 𝑡ℎ global layer respectively. For the group layers, we clone the parameters of 𝑞 𝑚 for 𝑛 copies and get { 𝑞 𝑀,𝑖 } 𝑛 𝑖 = 1 , each of which is served as the input of the group layers of a group. Then the equation of the output of the 𝑖 𝑡ℎ group at the ( 𝑙 + 1 ) 𝑡ℎ group layer is given as:

<!-- formula-not-decoded -->

where 𝑙 ∈ [ 0 , 𝑁 ) , and 𝑞 𝑀 + 𝑙 , 𝒘 ( 𝑀 + 𝑙 ) 𝑟 , 𝒃 ( 𝑀 + 𝑙 ) 𝑟 denote the outputs, weights, and biases of the 𝑙 𝑡ℎ group layer respectively. Finally, the outputs of the 𝑁 𝑡ℎ group layers are concatenated:

<!-- formula-not-decoded -->

## 3.4 Frozen Pre-trained LLM Encoder

To further enhance the ability of TSINR to detect anomalies, we leverage the representational capability of LLM. A pre-trained LLM is employed as the encoder, which has been demonstrated to process time-series data and provide cross-modal knowledge [72]. With this pre-trained LLM, we map the input data into the feature domain to amplify the fluctuations of anomalies from both time and channel dimensions. On the one hand, in the time dimension, we observe that the extracted feature of LLM involves more intense fluctuations during the anomaly interval. On the other hand, in the channel dimension, other channels have the same anomaly interval due to the ability of LLM to extract and fuse the inter-channel information. Therefore, TSINR can exhibit greater sensitivity to anomalous data, thereby enhancing its ability for anomaly detection. The corresponding experimental results and analysis can be found in Section 4.4.2.

More specifically, the self-attention layers and the feed-forward layers are frozen to preserve the prior knowledge in the pre-trained model. For any given time series data 𝑋 ∈ R 𝑑 × 𝑇 , the pre-trained LLM encoder maps it to feature domain:

<!-- formula-not-decoded -->

where 𝑍 ∈ R 𝑑 × 𝑇 denotes the feature corresponding to 𝑋 .

## 3.5 Anomaly Criterion

Following previous reconstruction-based anomaly detection approaches [32, 41, 72], we use the reconstruction error as the anomaly score for each time series data point. The anomaly score at timestamp 𝑡 is defined as follows:

<!-- formula-not-decoded -->

Based on this point-wise anomaly score, we use a parameter threshold 𝛿 to determine whether the point is abnormal or normal:

<!-- formula-not-decoded -->

The threshold 𝛿 is set to label a proportion 𝛾 of the test dataset as anomalies. And 𝛾 is a hyper-parameter based on actual datasets.

## 4 EXPERIMENTS

## 4.1 Datasets

We use eight anomaly detection benchmarks from real-world scenarios to validate the performance of our proposed method, including seven multivariate datasets ( SMD [46], PSM [1], SWaT [27], MSL [9], SMAP [9], PTB-XL [11, 50], and SKAB [13]) and one univariate dataset ( UCR [57]).

## 4.2 Baselines and Experimental Settings

We compare our proposed method with 11 state-of-the-art deep learning approaches, including both general frameworks designed for time series modeling and algorithms specifically tailored for time series anomaly detection: FPT [72], TimesNet [55], ETSformer [53], FEDformer [71], LightTS [65], DLinear [63], Autoformer [56], Pyraformer [24], AnomalyTransformer [59], Informer [69] and Transformer [49]. The commonly used metrics of precision (P), recall (R), and F1-score are employed for evaluation. Additionally, we report threshold-free measurements, including the Area Under the Curve (AUC) and Volume Under the Surface (VUS) [31], which are provided in the Appendix A.

The implementation details and the default hyper-parameters are summarized here. For a fair comparison, we only employ the classical reconstruction error across all baseline models. Also, we adopt identical data processing methods and the corresponding parameter configurations. We employ the sliding window approach and use a fixed window size of 100 for all datasets. The proportion 𝛾 mentioned in Section 3.5 is set to 0.5 for SMD dataset, 0.1 for UCR dataset, 10 for SKAB dataset, and 1 for others. These parameters adhere to the settings of previous work [59, 72]. Ablation studies on the anomaly proportion 𝛾 are in Appendix B. For the main results, our TSINR model involves 3 global layers and 2 group layers in the residual block. And the hidden dimensions are 64 and 32 respectively. The transformer encoder has 6 blocks. We use GPT2 [35] as the pre-trained LLM encoder and 6 blocks are utilized following the same settings as in FPT [72]. The experiments are conducted using the ADAM optimizer [14] with an initial learning rate of 10 -4 . A single NVIDIA Tesla-V100 32GB GPU is applied for each dataset. And the efficiency analysis is in Appendix C.

## 4.3 Main Resutls

We compare our method with 11 other state-of-the-art approaches and the results are shown in Table 1. These results show that our method achieves superior overall performance on these benchmark datasets. These experimental results confirm that TSINR, in both multivariate and univariate scenarios, effectively captures temporal continuity and precisely identifies discontinuous anomalies. The findings affirm the robustness of TSINR across diverse datasets and showcase its potential for broader applications in diverse domains.

Table 1: The overall results on multivariate and univariate datasets. The precision (P), recall (R), and F1-score (F1) values are reported, all in percentage (%). The best results of F1 on each dataset and the average P, R F1 among all datasets are in Bold and the second best ones are underlined.

| Dataset   | Metric   |   Trans. |   FED. |   Anomaly. |   Auto. |   Pyra. |   In. |   ETS. |   LightTS |   Dlinear |   TimesNet |   FPT |   TSINR |
|-----------|----------|----------|--------|------------|---------|---------|-------|--------|-----------|-----------|------------|-------|---------|
|           | P        |    78.32 |  78.45 |      78.72 |   78.49 |   78.49 | 78.37 |  86.63 |     87.04 |     87.27 |      87.98 | 87.27 |   83.09 |
| SMD       | R        |    65.24 |  65.08 |      65.43 |   65.13 |   65.53 | 65.23 |  75.35 |     78.39 |     80.99 |      81.54 | 81.08 |   80.46 |
|           | F1       |    71.19 |  71.14 |      71.46 |   71.19 |   71.43 | 71.20 |  80.68 |     82.49 |     84.01 |      84.64 | 84.06 |   81.76 |
|           | P        |    90.75 |  99.99 |      98.76 |   99.99 |   99.62 | 99.68 |  98.17 |     98.29 |     98.66 |      98.51 | 98.55 |   99.21 |
| PSM       | R        |    54.68 |  81.89 |      83.25 |   78.99 |   88.46 | 83.30 |  91.36 |     93.60 |     94.70 |      96.27 | 95.79 |   89.37 |
|           | F1       |    68.24 |  90.04 |      90.35 |   88.26 |   93.71 | 90.75 |  94.64 |     95.89 |     96.64 |      97.38 | 97.15 |   94.04 |
|           | P        |    99.67 |  99.95 |      99.73 |   99.96 |   99.71 | 99.64 |  92.01 |     92.36 |     92.25 |      92.14 | 92.12 |   99.31 |
| SWaT      | R        |    68.93 |  65.56 |      68.07 |   65.56 |   68.05 | 68.96 |  93.33 |     93.32 |     93.10 |      93.09 | 93.06 |   72.32 |
|           | F1       |    81.50 |  79.19 |      80.91 |   79.19 |   80.90 | 81.51 |  92.67 |     92.84 |     92.68 |      92.61 | 92.59 |   83.69 |
|           | P        |    90.58 |  90.69 |      89.78 |   90.66 |   90.64 | 90.63 |  86.89 |     89.17 |     89.68 |      89.55 | 82.03 |   83.57 |
| MSL       | R        |    74.65 |  75.48 |      73.66 |   75.22 |   74.76 | 74.96 |  67.78 |     73.64 |     75.31 |      75.29 | 82.01 |   85.40 |
|           | F1       |    81.85 |  82.39 |      80.93 |   82.22 |   81.94 | 82.06 |  76.16 |     80.66 |     81.87 |      81.80 | 82.02 |   84.47 |
|           | P        |    90.87 |  89.98 |      90.14 |   90.72 |   89.51 | 90.66 |  90.75 |     90.02 |     89.89 |      89.92 | 90.91 |   91.67 |
| SMAP      | R        |    61.44 |  55.89 |      54.00 |   62.58 |   54.59 | 61.69 |  54.68 |     53.90 |     54.01 |      56.56 | 61.01 |   76.42 |
|           | F1       |    73.31 |  68.95 |      67.54 |   74.07 |   67.82 | 73.43 |  68.24 |     67.43 |     67.48 |      69.44 | 73.02 |   83.35 |
|           | P        |    56.89 |  48.36 |      56.00 |   49.12 |   50.22 | 57.41 |  62.84 |     66.38 |     62.95 |      67.60 | 71.85 |   58.35 |
| PTB-XL    | R        |    29.99 |  27.60 |      31.50 |   27.53 |   23.85 | 25.43 |  28.45 |     16.46 |     13.95 |      14.47 | 24.52 |   35.00 |
|           | F1       |    39.28 |  35.14 |      40.32 |   35.28 |   32.34 | 35.25 |  39.17 |     26.38 |     22.84 |      23.84 | 36.57 |   43.75 |
| P         |          |    87.56 |  86.88 |      91.83 |   87.51 |   89.55 | 88.67 |  85.38 |     83.83 |     86.01 |      85.65 | 86.18 |   89.98 |
| SKAB R    |          |    86.72 |  77.71 |      95.04 |   91.10 |   97.27 | 97.27 | 100.00 |     82.01 |    100.00 |     100.00 | 99.21 |   98.65 |
| F1        |          |    87.14 |  82.04 |      93.41 |   89.27 |   93.25 | 92.77 |  92.12 |     82.91 |     92.48 |      92.27 | 92.24 |   94.11 |
|           | P        |    41.13 |  32.96 |      44.79 |   42.82 |   42.12 | 43.97 |  40.12 |     37.70 |     34.55 |      33.11 | 41.00 |   67.29 |
| UCR       | R        |    33.61 |  25.73 |      34.83 |   33.97 |   35.13 | 35.16 |  29.85 |     29.01 |     29.06 |      29.18 | 32.51 |   62.35 |
|           | F1       |    34.50 |  27.09 |      36.51 |   35.52 |   36.02 | 36.41 |  31.94 |     30.82 |     29.67 |      29.81 | 34.33 |   62.46 |
|           | P        |    79.47 |  78.41 |      82.22 |   79.91 |   79.98 | 81.13 |  80.35 |     80.60 |     80.16 |      80.56 | 81.24 |   84.06 |
| Average   | R        |    59.41 |  59.37 |      63.22 |   62.51 |   63.46 | 64.00 |  67.60 |     65.04 |     67.64 |      68.30 | 71.15 |   75.00 |
|           | F1       |    67.13 |  67.00 |      70.18 |   69.38 |   69.68 | 70.42 |  71.95 |     69.93 |     70.96 |      71.47 | 74.00 |   78.45 |

In multivariate scenarios, we observe that despite both MSL and SMAP being collected from NASA Space Sensors, TSINR achieved significantly greater improvements on the SMAP dataset compared to other methods. This could be attributed to the presence of more point anomalies in the SMAP dataset. Point anomalies exhibit poorer continuity compared to other anomaly patterns due to their isolated nature, representing single data points that significantly deviate from the surrounding pattern. This aligns with the property of spectral bias, making our model more sensitive to point anomalies, thereby achieving greater improvements on the SMAP dataset. Meanwhile, our approach shows moderate performance on the SWAT dataset. This is due to the poorer continuity of the SWaT dataset, which affects the fitting ability of TSINR.

The situation is more complex in the UCR dataset. The UCR dataset comprises 250 univariate sub-datasets from various domains and we report the average scores to provide a comprehensive assessment. The results demonstrate that our approach still outperforms other methods in overall performance by a significant margin. These sub-datasets originate from various domains, demonstrating the generalization capability of our method. In addition, it proves the strong ability of TSINR for modeling and identifying anomalies in univariate scenarios and underscores the importance of the spectral bias constraint for anomaly detection.

## 4.4 Ablation Studies

4.4.1 Analysis of the Decomposition Components and the Groupbased Architecture. In this section, we analyze the effectiveness of the proposed decomposition components and group-based architecture. The decomposition components indicate the three components (i.e., trend, seasonal, and residual) designed in our paper. And the group-based architecture is proposed for the residual block.

The main purpose of the decomposition components is to extract the unique trend and seasonal information of the time series data. The results in Table 2 indicate that capturing these distinctive features significantly enhances the capability for anomaly detection. To further explain the effectiveness of the trend and seasonal components in INR continuous function, we conduct ablation experiments using both synthetic and real-world datasets. The synthetic trend and seasonal datasets are generated by [16], while the realworld datasets include SMD, PSM, MSL, and SKAB. As shown in Table 3 and Figure 4, incorporating trend and seasonal components significantly improves both reconstruction performance and anomaly detection capability. In contrast, omitting these components results in inadequate data fitting in certain cases, thereby hindering the ability to detect anomalies. Also, we showcase the detection of a non-spike anomaly segment in Figure 5, highlighting the role of trend and seasonal components in identifying such anomalies.

Table 2: Ablation studies on the decomposition components and the group-based architecture. The F1 score is reported and the best results are in Bold.

| Decomposition   | Group-based   |   SMD |   PSM |   SWaT |   MSL |   SMAP |   PTB-XL |   SKAB |
|-----------------|---------------|-------|-------|--------|-------|--------|----------|--------|
| ✗               | ✗             | 78.52 | 92.61 |  81.89 | 82.02 |  73.31 |    40.11 |  93.13 |
| ✗               | ✓             | 80.10 | 93.08 |  82.28 | 82.28 |  78.94 |    40.76 |  94.01 |
| ✓               | ✗             | 79.24 | 93.04 |  82.16 | 82.95 |  78.66 |    42.65 |  93.96 |
| ✓               | ✓             | 81.76 | 94.04 |  83.69 | 84.74 |  83.35 |    43.75 |  94.11 |

Table 3: Analysis of the decomposition components. The MSE and F1 scores (*/*) are reported and the best results are in Bold.

| Decomposition   | Synthetic Trend   | Synthetic Seasonal   | SMD        | PSM        | MSL        | SKAB       |
|-----------------|-------------------|----------------------|------------|------------|------------|------------|
| ✗               | 0.56/71.15        | 0.44/25.97           | 1.21/80.10 | 0.26/93.08 | 2.23/82.28 | 1.23/94.01 |
| ✓               | 0.09/100.00       | 0.01/100.00          | 0.99/81.76 | 0.21/94.04 | 1.81/84.74 | 0.84/94.11 |

Table 4: Ablation studies on the pre-trained LLM encoder. The F1 score is reported and the best results are in Bold.

| Pre-trained LLM   |   SMD |   PSM |   SWaT |   MSL |   SMAP |   PTB-XL |   SKAB |   UCR |
|-------------------|-------|-------|--------|-------|--------|----------|--------|-------|
| ✗                 | 80.29 | 92.69 |  82.33 | 83.27 |  79.35 |    40.04 |  93.91 | 62.46 |
| ✓                 | 81.76 | 94.04 |  83.69 | 84.47 |  83.35 |    43.75 |  94.11 | 60.41 |

Original TSINR w/o

Data Decomposition

Synthetic Trend Dataset TSINR

Anomaly

Area

Synthetic Seasonal Dataset in different groups, thereby reducing the number of variables each group needs to model and improving the representational capacity. The global layers extract the inter-channel information, while the group layers selectively focus on detailed information for specific channels. This enhances the representational capability for each variable without losing any knowledge. Detailed ablation studies on the number of groups are left in Appendix D.

Figure 5: The visual analysis of the non-spike anomaly segment in the real-world PSM dataset.

<!-- image -->

4.4.2 Analysis of Pre-trained LLM Encoder. Further, we prove the validity of the pre-trained LLM encoder, which is utilized to encode the data into the feature domain to amplify the fluctuations of anomalies and thereby enhance the capability of TSINR in identifying anomalies. Table 4 displays the ablation studies of the pre-trained LLM encoder. For the multivariate datasets, it can be observed that

Time Series Reconstruction

Anomaly Score

Figure 4: The visual analysis of the decomposition components on synthetic trend and seasonal datasets.

The anomaly, characterized by subtle deviations from the expected behavior, is effectively captured, demonstrating the ability of TSINR to detect anomalies that do not exhibit abrupt or spike-like changes. This emphasizes the robustness of the model in handling different types of anomalies in real-world datasets like PSM.

In addition, the group-based architecture is designed to enhance the representational capacity of INR for multivariate data. Experimental results indicate an improvement in the capability for anomaly detection when employing the proposed group-based architecture. This is because modeling multiple variables and capturing both inter- and intra-channel information with a simple continuous function, which only consists of fully-connected layers, is challenging. Our approach addresses this by dividing the variables into several groups and applying independently fully-connected layers applying this encoder enhances the performance of anomaly detection. To further demonstrate the effectiveness, we compare the raw data with the features extracted through the encoder. As shown in Figure 6, the figures in the first row illustrate that during the time interval when anomalies occur, the extracted features exhibit more pronounced fluctuations compared to the original data. This implies that the discontinuity in anomalies is increased in time domain. Also, these extracted features incorporate inter-channel information, providing a manifestation of anomalies among all variables. As shown in the second line, the features exhibit anomalous fluctuations in the same time interval as other channels, whereas the original data only shows a brief peak. This verifies that the anomalies are shared in channel domain. Based on these results, we indicate that utilizing the pre-trained LLM encoder can effectively enhance abnormal information both intra- and inter-channel. This aligns with the spectral bias of INR, making our model more sensitive to anomalous data.

Figure 6: The visualization of the original data and the corresponding features obtained from the frozen pre-trained LLM encoder. The intense fluctuations in the anomaly area are amplified from both time and channel domains.

<!-- image -->

Figure 7: The visualization of the UCR original data and the corresponding features obtained from the frozen pre-trained LLM encoder of (a) anomaly area and (b) normal area.

<!-- image -->

In contrast, on the UCR dataset, using the pre-trained LLM encoder actually decreased the performance of anomaly detection. This is because the advantages observed in the aforementioned multivariate datasets do not apply to the UCR dataset. On the one hand, there are no anomalies in the UCR training data. Consequently, during inference, the LLM encoder still extracts features according to the normal pattern. As shown in Figure 7, the highly discontinuous anomalous data becomes relatively smooth after feature extraction.

Meanwhile, there is little variation in features between normal and abnormal areas. This prompts TSINR to fit the anomalous data, thereby reducing its sensitivity to anomalies. On the other hand, the UCR dataset involves only a single variable, thus the inter-channel information provided by the LLM encoder is meaningless.

## 4.5 Visual Analysis

In order to demonstrate that our approach is sensitive to discontinuous abnormal data, we compare the original data with the reconstructed values of the TSINR model. As shown in Figure 8, the smooth normal points are well-fitted, while the discontinuous abnormal points are not. The anomaly scores significantly increase when anomalies occur, which aids the TSINR model in distinguishing abnormal points, demonstrating its sensitivity to the anomaly points. It is worth noting that, despite the small and highly fluctuating values of anomaly scores on the UCR dataset, this still holds meaningful significance. The UCR dataset involves only one anomaly, hence even minor fluctuations are helpful for the model to pinpoint these anomaly points.

Furthermore, we validate the robustness of TSINR with the synthetic data generated for time series anomaly detection. It has univariate time series and involves different types of anomalies [16, 62], including the point-wise anomaly (global point and contextual point anomalies) and pattern-wise anomalies (shapelet, seasonal, and trend anomalies). As shown in Figure 9, the red points are anomaly points and the red areas are anomaly areas. It can be seen that TSINR can robustly detect various types of anomalies from normal points with relatively high anomaly scores.

## 5 CONCLUSION

Time series data anomaly detection plays a pivotal role in ensuring the reliability, security, and efficiency of systems across various domains. Reconstruction-based methods are mainstream approaches for this task because they do not require label information and are easy to interpret the detection results. However, unlabeled anomalous points in the training data can negatively impact the performance of these reconstruction models. To address this issue, this paper proposes a novel algorithm named TSINR for time series anomaly detection based on INR reconstruction. We utilize the spectral bias of INR to prioritize fitting continuous normal data and capture temporal continuity, thus enhancing sensitivity to discontinuous anomalous data. A transformer-based architecture is employed to predict the parameters of INR. To cope with the complex patterns of time series data, we specifically design a formulation of continuous function. It aims to implicitly learn the trend, seasonal, and residual information and capture the interand intra-channel information of the time series data. Besides, we leverage a frozen pre-trained LLM encoder to map the original data to the feature domain, thus amplifying the fluctuations of anomalies from both time and channel domains and enabling TSINR to better distinguish between abnormal and normal points. Experimental results indicate that TSINR exhibits superior overall performance on both multivariate and univariate benchmark datasets compared to other state-of-the-art algorithms. Also, ablation studies verify the effectiveness of each component, and visual analysis demonstrates the sensitivity of TSINR to anomalous points. In this work, we demonstrate the potential of INR in time series data tasks. In future work, we plan to explore the performance of INR on other time series tasks, including imputation, classification, and long-term and short-term forecasting. We believe that INR has the potential to become a unified framework for various time-series data tasks.

Figure 8: The visualization of the original data, the TSINR reconstructed data, and the corresponding anomaly score. In the anomaly area, anomaly scores significantly increase, demonstrating the sensitivity of TSINR to anomalies.

<!-- image -->

Figure 9: The visualization of the ground-truth anomalies and anomaly scores of TSINR for different types of anomalies.

<!-- image -->

## 6 ACKNOWLEDGMENTS

This work was supported by the National Natural Science Foundation of China (Grant No. 62202422), the National Key R&amp;D Program of China (Grant No. 2022ZD0160703), the National Natural Science Foundation of China (Grant Nos. 62276230 and 62372408), Zhejiang Provincial Natural Science Foundation of China (Grant No. LDT23F02023F02), and Shanghai Artificial Intelligence Laboratory.

## REFERENCES

- [1] Ahmed Abdulaal, Zhuanghua Liu, and Tomer Lancewicki. 2021. Practical approach to asynchronous multivariate time series anomaly detection and localization. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery &amp; data mining . 2485-2494.
- [2] Markus M Breunig, Hans-Peter Kriegel, Raymond T Ng, and Jörg Sander. 2000. LOF: identifying density-based local outliers. In Proceedings of the 2000 ACM SIGMOD international conference on Management of data . 93-104.
- [3] Yinbo Chen and Xiaolong Wang. 2022. Transformers as meta-learners for implicit neural representations. In European Conference on Computer Vision . Springer, 170-187.
- [4] Zeyuan Chen, Yinbo Chen, Jingwen Liu, Xingqian Xu, Vidit Goel, Zhangyang Wang, Humphrey Shi, and Xiaolong Wang. 2022. Videoinr: Learning video implicit neural representation for continuous space-time super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 2047-2057.
- [5] Robert B Cleveland, William S Cleveland, Jean E McRae, and Irma Terpenning. 1990. STL: A seasonal-trend decomposition. J. Off. Stat 6, 1 (1990), 3-73.
- [6] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020).
- [7] Elizabeth Fons, Alejandro Sztrajman, Yousef El-Laham, Alexandros Iosifidis, and Svitlana Vyetrenko. 2022. HyperTime: Implicit Neural Representations for Time Series Generation. (2022).
- [8] Haoyu Guo, Sida Peng, Haotong Lin, Qianqian Wang, Guofeng Zhang, Hujun Bao, and Xiaowei Zhou. 2022. Neural 3d scene reconstruction with the manhattanworld assumption. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 5511-5520.
- [9] Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, and Tom Soderstrom. 2018. Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery &amp; data mining . 387-395.
- [10] Kyeong-Joong Jeong and Yong-Min Shin. 2022. Time-series anomaly detection with implicit neural representation. arXiv preprint arXiv:2201.11950 (2022).
- [11] Aofan Jiang, Chaoqin Huang, Qing Cao, Shuang Wu, Zi Zeng, Kang Chen, Ya Zhang, and Yanfeng Wang. 2023. Multi-scale Cross-restoration Framework for Electrocardiogram Anomaly Detection. In International Conference on Medical Image Computing and Computer-Assisted Intervention . Springer, 87-97.
- [12] Xin Jie, Xixi Zhou, Chanfei Su, Zijun Zhou, Yuqing Yuan, Jiajun Bu, and Haishuai Wang. 2024. Disentangled Anomaly Detection For Multivariate Time Series. In Companion Proceedings of the ACM on Web Conference 2024 . 931-934.
- [13] Iurii D. Katser and Vyacheslav O. Kozitsin. 2020. Skoltech Anomaly Benchmark (SKAB). https://www.kaggle.com/dsv/1693952. https://doi.org/10.34740/ KAGGLE/DSV/1693952
- [14] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).
- [15] Amit Pal Singh Kohli, Vincent Sitzmann, and Gordon Wetzstein. 2020. Semantic implicit neural scene representations with semi-supervised training. In 2020 International Conference on 3D Vision (3DV) . IEEE, 423-433.
- [16] Kwei-Herng Lai, Daochen Zha, Junjie Xu, Yue Zhao, Guanchu Wang, and Xia Hu. 2021. Revisiting time series outlier detection: Definitions and benchmarks. In Thirty-fifth conference on neural information processing systems datasets and benchmarks track (round 1) .
- [17] GenLi and Jason J Jung. 2023. Deep learning for anomaly detection in multivariate time series: Approaches, applications, and challenges. Information Fusion 91 (2023), 93-102.
- [18] Jinbo Li, Witold Pedrycz, and Iqbal Jamal. 2017. Multivariate time series anomaly detection: A framework of Hidden Markov Models. Applied Soft Computing 60 (2017), 229-240.
- [19] Mengxuan Li, Peng Peng, Haiyue Sun, Min Wang, and Hongwei Wang. 2023. An Order-Invariant and Interpretable Dilated Convolution Neural Network for Chemical Process Fault Detection and Diagnosis. IEEE Transactions on Automation Science and Engineering (2023), 1-11. https://doi.org/10.1109/TASE.2023.3290202
- [20] Mengxuan Li, Peng Peng, Jingxin Zhang, Hongwei Wang, and Weiming Shen. 2023. SCCAM: Supervised Contrastive Convolutional Attention Mechanism for Ante-Hoc Interpretable Fault Diagnosis With Limited Fault Samples. IEEE Transactions on Neural Networks and Learning Systems (2023), 1-12. https: //doi.org/10.1109/TNNLS.2023.3313728
- [21] Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. 2008. Isolation forest. In 2008 eighth ieee international conference on data mining . IEEE, 413-422.
- [22] Ke Liu, Feng Liu, Haishuai Wang, Ning Ma, Jiajun Bu, and Bo Han. 2023. Partition Speeds Up Learning Implicit Neural Representations Based on ExponentialIncrease Hypothesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision . 5474-5483.
- [23] Ke Liu, Ning Ma, Zhihua Wang, Jingjun Gu, Jiajun Bu, and Haishuai Wang. 2023. Implicit Neural Distance Optimization for Mesh Neural Subdivision. In 2023 IEEE
24. International Conference on Multimedia and Expo (ICME) . IEEE, 2039-2044.
- [24] Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X Liu, and Schahram Dustdar. 2021. Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting. In International conference on learning representations .
- [25] Yunfan Lu, Zipeng Wang, Minjie Liu, Hongjian Wang, and Lin Wang. 2023. Learning Spatial-Temporal Implicit Neural Representations for Event-Guided Video Super-Resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 1557-1567.
- [26] Long Mai and Feng Liu. 2022. Motion-adjustable neural implicit video representation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition . 10738-10747.
- [27] Aditya P Mathur and Nils Ole Tippenhauer. 2016. SWaT: A water treatment testbed for research and training on ICS security. In 2016 international workshop on cyber-physical systems for smart water networks (CySWater) . IEEE, 31-36.
- [28] Amirali Molaei, Amirhossein Aminimehr, Armin Tavakoli, Amirhossein Kazerouni, Bobby Azad, Reza Azad, and Dorit Merhof. 2023. Implicit neural representation in medical imaging: A comparative survey. In Proceedings of the IEEE/CVF International Conference on Computer Vision . 2381-2391.
- [29] Etienne Le Naour, Louis Serrano, Léon Migus, Yuan Yin, Ghislain Agoua, Nicolas Baskiotis, Vincent Guigue, et al. 2023. Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations. arXiv preprint arXiv:2306.05880 (2023).
- [30] Boris N Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. 2019. NBEATS: Neural basis expansion analysis for interpretable time series forecasting. In International Conference on Learning Representations .
- [31] John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S Tsay, Aaron Elmore, and Michael J Franklin. 2022. Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection. Proceedings of the VLDB Endowment 15, 11 (2022), 2774-2787.
- [32] Daehyung Park, Yuuna Hoshi, and Charles C Kemp. 2018. A multimodal anomaly detector for robot-assisted feeding using an lstm-based variational autoencoder. IEEE Robotics and Automation Letters 3, 3 (2018), 1544-1551.
- [33] Samuel Pfrommer, Mathew Halm, and Michael Posa. 2021. Contactnets: Learning discontinuous contact dynamics with smooth, implicit representations. In Conference on Robot Learning . PMLR, 2279-2291.
- [34] Dian Qin, Haishuai Wang, Zhe Liu, Hongjia Xu, Sheng Zhou, and Jiajun Bu. 2022. Hilbert Distillation for Cross-Dimensionality Networks. Advances in Neural Information Processing Systems 35 (2022), 11726-11738.
- [35] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog 1, 8 (2019), 9.
- [36] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. 2019. On the spectral bias of neural networks. In International Conference on Machine Learning . PMLR, 5301-5310.
- [37] Maziar Raissi, Paris Perdikaris, and George E Karniadakis. 2019. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational physics 378 (2019), 686-707.
- [38] Yunlong Ran, Jing Zeng, Shibo He, Jiming Chen, Lincheng Li, Yingfeng Chen, Gimhee Lee, and Qi Ye. 2023. NeurAR: Neural Uncertainty for Autonomous 3D Reconstruction With Implicit Neural Representations. IEEE Robotics and Automation Letters 8, 2 (2023), 1125-1132.
- [39] Lukas Ruff, Robert Vandermeulen, Nico Goernitz, Lucas Deecke, Shoaib Ahmed Siddiqui, Alexander Binder, Emmanuel Müller, and Marius Kloft. 2018. Deep one-class classification. In International conference on machine learning . PMLR, 4393-4402.
- [40] Artem Ryzhikov, Maxim Borisyak, Andrey Ustyuzhanin, and Denis Derkach. 2021. NFAD: fixing anomaly detection using normalizing flows. PeerJ Computer Science 7 (2021), e757.
- [41] Mayu Sakurada and Takehisa Yairi. 2014. Anomaly detection using autoencoders with nonlinear dimensionality reduction. In Proceedings of the MLSDA 2014 2nd workshop on machine learning for sensory data analysis . 4-11.
- [42] Youjin Shin, Sangyup Lee, Shahroz Tariq, Myeong Shin Lee, Okchul Jung, Daewon Chung, and Simon S Woo. 2020. Itad: integrative tensor-based anomaly detection system for reducing false positives of satellite systems. In Proceedings of the 29th ACM international conference on information &amp; knowledge management . 2733-2740.
- [43] Karanjit Singh and Shuchita Upadhyaya. 2012. Outlier detection: applications and techniques. International Journal of Computer Science Issues (IJCSI) 9, 1 (2012), 307.
- [44] Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. 2020. Implicit neural representations with periodic activation functions. Advances in neural information processing systems 33 (2020), 7462-7473.
- [45] Yannick Strümpler, Janis Postels, Ren Yang, Luc Van Gool, and Federico Tombari. 2022. Implicit neural representations for image compression. In European Conference on Computer Vision . Springer, 74-91.

- [46] Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei. 2019. Robust anomaly detection for multivariate time series through stochastic recurrent neural network. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery &amp; data mining . 2828-2837.
- [47] Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng. 2020. Fourier features let networks learn high frequency functions in low dimensional domains. Advances in Neural Information Processing Systems 33 (2020), 7537-7547.
- [48] Shreshth Tuli, Giuliano Casale, and Nicholas R Jennings. 2022. TranAD: deep transformer networks for anomaly detection in multivariate time series data. Proceedings of the VLDB Endowment 15, 6 (2022), 1201-1214.
- [49] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).
- [50] Patrick Wagner, Nils Strodthoff, Ralf-Dieter Bousseljot, Dieter Kreiseler, Fatima I Lunze, Wojciech Samek, and Tobias Schaeffter. 2020. PTB-XL, a large publicly available electrocardiography dataset. Scientific data 7, 1 (2020), 1-15.
- [51] Haishuai Wang, Jia Wu, Peng Zhang, and Yixin Chen. 2018. Learning shapelet patterns from network-based time series. IEEE transactions on industrial informatics 15, 7 (2018), 3864-3876.
- [52] Haishuai Wang, Qin Zhang, Jia Wu, Shirui Pan, and Yixin Chen. 2019. Time series feature learning with labeled and unlabeled data. Pattern Recognition 89 (2019), 55-66.
- [53] Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshat Kumar, and Steven Hoi. 2022. Etsformer: Exponential smoothing transformers for time-series forecasting. arXiv preprint arXiv:2202.01381 (2022).
- [54] Chenrui Wu, Haishuai Wang, Xiang Zhang, Zhen Fang, and Jiajun Bu. 2024. Spatio-temporal Heterogeneous Federated Learning for Time Series Classification with Multi-view Orthogonal Training. In Proceedings of the 32nd ACM International Conference on Multimedia . 2613-2622.
- [55] Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. 2022. Timesnet: Temporal 2d-variation modeling for general time series analysis. arXiv preprint arXiv:2210.02186 (2022).
- [56] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. 2021. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in Neural Information Processing Systems 34 (2021), 22419-22430.
- [57] Renjie Wu and Eamonn Keogh. 2021. Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress. IEEE Transactions on Knowledge and Data Engineering (2021).
- [58] Yiheng Xie, Towaki Takikawa, Shunsuke Saito, Or Litany, Shiqin Yan, Numair Khan, Federico Tombari, James Tompkin, Vincent Sitzmann, and Srinath Sridhar. 2022. Neural fields in visual computing and beyond. In Computer Graphics Forum , Vol. 41. Wiley Online Library, 641-676.
- [59] Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long. 2021. Anomaly transformer: Time series anomaly detection with association discrepancy. arXiv preprint arXiv:2110.02642 (2021).
- [60] Asrul H Yaacob, Ian KT Tan, Su Fong Chien, and Hon Khi Tan. 2010. Arima based network anomaly detection. In 2010 Second International Conference on Communication Software and Networks . IEEE, 205-209.
- [61] Takehisa Yairi, Naoya Takeishi, Tetsuo Oda, Yuta Nakajima, Naoki Nishimura, and Noboru Takata. 2017. A data-driven health monitoring method for satellite housekeeping data based on probabilistic clustering and dimensionality reduction. IEEE Trans. Aerospace Electron. Systems 53, 3 (2017), 1384-1401.
- [62] Yiyuan Yang, Chaoli Zhang, Tian Zhou, Qingsong Wen, and Liang Sun. 2023. DCdetector: Dual Attention Contrastive Representation Learning for Time Series Anomaly Detection. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining . ACM.
- [63] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. 2023. Are transformers effective for time series forecasting?. In Proceedings of the AAAI conference on artificial intelligence , Vol. 37. 11121-11128.
- [64] Shuyi Zhang, Ke Liu, Jingjun Gu, Xiaoxu Cai, Zhihua Wang, Jiajun Bu, and Haishuai Wang. 2024. Attention beats linear for fast implicit neural representation generation. arXiv preprint arXiv:2407.15355 (2024).
- [65] Tianping Zhang, Yizhuo Zhang, Wei Cao, Jiang Bian, Xiaohan Yi, Shun Zheng, and Jian Li. 2022. Less is more: Fast multivariate time series forecasting with light sampling-oriented mlp structures. arXiv preprint arXiv:2207.01186 (2022).
- [66] Zhijie Zhang, Wenzhong Li, Wangxiang Ding, Linming Zhang, Qingning Lu, Peng Hu, Tong Gui, and Sanglu Lu. 2023. STAD-GAN: unsupervised anomaly detection on multivariate time series with self-training generative adversarial networks. ACM Transactions on Knowledge Discovery from Data 17, 5 (2023), 1-18.
- [67] Guoxiang Zhong, Fagui Liu, Jun Jiang, Bin Wang, and CL Philip Chen. 2024. Refining one-class representation: A unified transformer for unsupervised timeseries anomaly detection. Information Sciences 656 (2024), 119914.
- [68] Bin Zhou, Shenghua Liu, Bryan Hooi, Xueqi Cheng, and Jing Ye. 2019. Beatgan: Anomalous rhythm detection using adversarially generated time series.. In IJCAI , Vol. 2019. 4433-4439.
- [69] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. 2021. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI conference on artificial intelligence , Vol. 35. 11106-11115.
- [70] Sheng Zhou, Yucheng Wang, Defang Chen, Jiawei Chen, Xin Wang, Can Wang, and Jiajun Bu. 2021. Distilling holistic knowledge with graph neural networks. In Proceedings of the IEEE/CVF international conference on computer vision . 1038710396.
- [71] Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. 2022. Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. In International Conference on Machine Learning . PMLR, 2726827286.
- [72] Tian Zhou, Peisong Niu, Liang Sun, Rong Jin, et al. 2024. One fits all: Power general time series analysis by pretrained lm. Advances in neural information processing systems 36 (2024).
- [73] Bo Zong, Qi Song, Martin Renqiang Min, Wei Cheng, Cristian Lumezanu, Daeki Cho, and Haifeng Chen. 2018. Deep autoencoding gaussian mixture model for unsupervised anomaly detection. In International conference on learning representations .

## APPENDIX

## A ADDITIONAL RESULTS

In addition to the commonly used F1-score, which is thresholddependent, we evaluate TSINR using threshold-free AUC and VUS scores. These metrics provide a more comprehensive assessment of model performance, as they do not rely on a specific decision threshold. The results in Table demonstrate that TSINR consistently outperforms competing methods across all datasets, highlighting its robustness and superior capability in anomaly detection tasks.

## B STUDY ON THE ANOMALY PROPORTION

Anomaly proportion 𝛾 is a hyper-parameter which decides the anomaly threshold 𝛿 . As mentioned in Section 3.5, the threshold 𝛿 is set to label a proportion 𝛾 of the test dataset as anomalies. We show the influence of the anomaly proportion in Table 6. It can be observed that an appropriate anomaly proportion is beneficial in aiding the model's judgment of anomalies. Among them, PSM exhibits greater robustness to anomaly proportion compared to SMD and MSL. This is consistent with previous findings [62].

## C EFFICIENCY ANALYSIS

We measure the efficiency of the TSINR method, and show the results in Table 7. The results indicate that the TSINR is pretty efficient and lightweight.

## D STUDY ON THE GROUP NUMBER

The group number is a hyperparameter in our method, indicating the number of groups in the group-based architecture. It determines the number of variables learned by neurons in each group layer. Table 8 presents ablation studies on the group number. When set to 1, the method reverts to a regular function without grouping. Our results show that the optimal group number varies across datasets, which is reasonable given that, similar to image data, partitioning accelerates INR fitting and is dependent on the dataset's intrinsic characteristics. Nonetheless, the group-based architecture improves the model's anomaly detection performance.

Table 5: The additional results on multivariate and univariate datasets. The AUC and VUS values are reported. The best results are in Bold and the second best ones are underlined.

| Dataset   | Metric   |   Trans. |   FED. |   Anomaly. |   Auto. |   Pyra. |   In. |   ETS. |   LightTS |   DLinear |   TimesNet |   FPT |   TSINR |
|-----------|----------|----------|--------|------------|---------|---------|-------|--------|-----------|-----------|------------|-------|---------|
| SMD       | AUC      |    0.747 |  0.654 |      0.765 |   0.652 |   0.728 | 0.729 |  0.760 |     0.738 |     0.732 |      0.766 | 0.723 |   0.774 |
| SMD       | VUS      |    0.740 |  0.639 |      0.762 |   0.631 |   0.723 | 0.722 |  0.755 |     0.734 |     0.728 |      0.762 | 0.720 |   0.769 |
| PSM       | AUC      |    0.721 |  0.662 |      0.666 |   0.661 |   0.704 | 0.712 |  0.622 |     0.585 |     0.565 |      0.590 | 0.578 |   0.722 |
| PSM       | VUS      |    0.667 |  0.563 |      0.608 |   0.556 |   0.657 | 0.670 |  0.609 |     0.570 |     0.543 |      0.575 | 0.568 |   0.674 |
| SWaT      | AUC      |    0.816 |  0.817 |      0.820 |   0.817 |   0.818 | 0.816 |  0.444 |     0.737 |     0.622 |      0.247 | 0.236 |   0.823 |
| SWaT      | VUS      |    0.518 |  0.512 |      0.530 |   0.521 |   0.534 | 0.515 |  0.435 |     0.701 |     0.598 |      0.241 | 0.232 |   0.754 |
| MSL       | AUC      |    0.624 |  0.550 |      0.532 |   0.550 |   0.602 | 0.613 |  0.596 |     0.601 |     0.615 |      0.623 | 0.590 |   0.657 |
| MSL       | VUS      |    0.607 |  0.525 |      0.518 |   0.525 |   0.569 | 0.599 |  0.555 |     0.569 |     0.580 |      0.591 | 0.552 |   0.638 |
| SMAP      | AUC      |    0.526 |  0.450 |      0.456 |   0.450 |   0.452 | 0.490 |  0.401 |     0.380 |     0.397 |      0.455 | 0.474 |   0.576 |
| SMAP      | VUS      |    0.502 |  0.418 |      0.450 |   0.415 |   0.438 | 0.482 |  0.363 |     0.343 |     0.373 |      0.412 | 0.444 |   0.567 |
| PTB-XL    | AUC      |    0.604 |  0.485 |      0.583 |   0.485 |   0.536 | 0.560 |  0.589 |     0.545 |     0.516 |      0.618 | 0.627 |   0.660 |
| PTB-XL    | VUS      |    0.456 |  0.339 |      0.436 |   0.339 |   0.386 | 0.417 |  0.453 |     0.401 |     0.365 |      0.471 | 0.486 |   0.534 |
| SKAB      | AUC      |    0.536 |  0.429 |      0.493 |   0.440 |   0.570 | 0.496 |  0.482 |     0.480 |     0.504 |      0.496 | 0.496 |   0.585 |
| SKAB      | VUS      |    0.535 |  0.421 |      0.492 |   0.434 |   0.569 | 0.495 |  0.482 |     0.474 |     0.504 |      0.496 | 0.496 |   0.585 |
| UCR       | AUC      |    0.520 |  0.541 |      0.548 |   0.533 |   0.546 | 0.517 |  0.502 |     0.523 |     0.546 |      0.533 | 0.529 |   0.663 |
| UCR       | VUS      |    0.511 |  0.513 |      0.538 |   0.508 |   0.533 | 0.506 |  0.486 |     0.511 |     0.528 |      0.514 | 0.513 |   0.650 |

Table 6: Ablation Studies on the anomaly proportion 𝛾 which decides the threshold 𝛿 . The best results are in Bold.

| Dataset   | SMD   | SMD   | SMD   | PSM   | PSM   | PSM   | MSL   | MSL   | MSL   |
|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Metric    | P     | R     | F1    | P     | R     | F1    | P     | R     | F1    |
| 𝛾 = 0.5   | 83.09 | 80.46 | 81.76 | 99.53 | 88.65 | 93.78 | 94.67 | 61.92 | 74.87 |
| 𝛾 = 0.6   | 80.55 | 80.93 | 80.74 | 99.37 | 88.90 | 93.85 | 93.16 | 62.36 | 74.71 |
| 𝛾 = 0.7   | 77.45 | 81.46 | 79.40 | 99.33 | 89.00 | 93.89 | 92.32 | 66.77 | 77.49 |
| 𝛾 = 0.8   | 75.16 | 83.06 | 78.91 | 99.27 | 88.80 | 93.75 | 91.92 | 73.43 | 81.64 |
| 𝛾 = 0.9   | 73.44 | 83.37 | 78.09 | 98.82 | 89.03 | 93.67 | 90.46 | 75.20 | 82.12 |
| 𝛾 = 1.0   | 72.55 | 83.39 | 77.59 | 99.21 | 89.37 | 94.04 | 83.98 | 84.26 | 84.12 |

Table 7: The efficiency of TSINR on the data with 128 batch size.

| Training Time per Batch   | Inference Time per Batch   | Learnable Parameters   |
|---------------------------|----------------------------|------------------------|
| 0.1s                      | 0.08s                      | 8.3M                   |

Table 8: Ablation Studies on the group number of the group-based architecture. The best results are in Bold.

| Dataset        | SMD   | SMD   | SMD   | PSM   | PSM   | PSM   | MSL   | MSL   | MSL   |
|----------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Metric         | P     | R     | F1    | P     | R     | F1    | P     | R     | F1    |
| Group Num = 1  | 82.82 | 75.97 | 79.24 | 99.11 | 87.66 | 93.04 | 83.30 | 82.59 | 82.95 |
| Group Num = 2  | 82.22 | 76.61 | 79.31 | 99.28 | 88.90 | 93.81 | 83.47 | 82.57 | 83.01 |
| Group Num = 3  | 82.46 | 77.45 | 79.88 | 99.25 | 89.02 | 93.86 | 83.43 | 82.51 | 82.97 |
| Group Num = 4  | 82.62 | 77.76 | 80.12 | 98.81 | 89.32 | 93.83 | 83.60 | 82.76 | 83.17 |
| Group Num = 5  | 82.29 | 77.30 | 79.72 | 99.21 | 89.37 | 94.04 | 83.53 | 82.66 | 83.09 |
| Group Num = 6  | 83.09 | 80.46 | 81.76 | 99.19 | 88.53 | 93.56 | 83.57 | 82.71 | 83.14 |
| Group Num = 7  | 82.71 | 77.05 | 79.78 | 99.23 | 88.79 | 93.72 | 83.56 | 82.80 | 83.18 |
| Group Num = 8  | 82.39 | 77.08 | 79.65 | 99.41 | 88.57 | 93.68 | 83.52 | 82.48 | 83.00 |
| Group Num = 9  | 83.17 | 77.63 | 80.31 | 99.26 | 89.07 | 93.89 | 83.57 | 85.40 | 84.47 |
| Group Num = 10 | 82.17 | 76.73 | 79.36 | 99.30 | 88.81 | 93.77 | 83.56 | 82.47 | 83.01 |