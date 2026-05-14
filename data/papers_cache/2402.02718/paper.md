## Denoising Time Cycle Modeling for Recommendation

Sicong Xie

Ant Group

Hangzhou, Zhejiang, China sicong.xsc@antgroup.com

Kaiming Shen Ant Group Beijing, China

kaiming.skm@antgroup.com

## ABSTRACT

Recently, modeling temporal patterns of user-item interactions have attracted much attention in recommender systems. We argue that existing methods ignore the variety of temporal patterns of user behaviors. We define the subset of user behaviors that are irrelevant to the target item as noises, which limits the performance of target-related time cycle modeling and affect the recommendation performance. In this paper, we propose Denoising Time Cycle Modeling (DiCycle), a novel approach to denoise user behaviors and select the subset of user behaviors that are highly related to the target item. DiCycle is able to explicitly model diverse time cycle patterns for recommendation. Extensive experiments are conducted on both public benchmarks and a real-world dataset, demonstrating the superior performance of DiCycle over the state-of-the-art recommendation methods.

## CCS CONCEPTS

· Information systems → Recommender systems ; Personalization ; Retrieval models and ranking .

## KEYWORDS

Time cycle, denoise, user behaviors, recommendation

## ACMReference Format:

Sicong Xie, Qunwei Li, Weidi Xu, Kaiming Shen, Shaohu Chen, and Wenliang Zhong. 2022. Denoising Time Cycle Modeling for Recommendation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22), July 11-15, 2022, Madrid, Spain. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3477495. 3531785

## 1 INTRODUCTION

Personalized recommender systems can capture dynamic user interests and potential demands through analyzing a large amount of features, such as user profile, user behaviors, item category,

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

SIGIR '22, July 11-15, 2022, Madrid, Spain

© 2022 Association for Computing Machinery.

ACM ISBN 978-1-4503-8732-3/22/07...$15.00

Qunwei Li Ant Group Hangzhou, Zhejiang, China

qunwei.qw@antgroup.com

Shaohu Chen Ant Group Beijing, China

shaohu.csh@antgroup.com Weidi Xu Ant Group Shanghai, China weidi.xwd@antgroup.com Wenliang Zhong Ant Group Hangzhou, Zhejiang, China yice.zwl@antgroup.com

Figure 1: Analysis of two typical temporal patterns of user behaviors.

<!-- image -->

<!-- image -->

etc [6, 11, 17]. Among them, temporal information in user behaviors plays a vital role, which is extracted from user-item interaction records and can present different time-related patterns of user behaviors.

We analyze user's click actions on two items in our real-world dataset to illustrate two typical temporal patterns. Figure 1(a) shows how the number of clicks on an item of Food-Delivery service changes every hour in a day. The click intensity reaches its peaks at 11, 18, and 21 o'clock, corresponding to probable lunch, dinner, and midnight meal time. We call such a time pattern as Absolute Time Cycle Pattern (ATC), which characterizes the impact of semantic time (e.g., hour, weekday, day of month, etc) on user-item interactions and reveals the specific time points that a user may be interested in an item. Besides, the time span | 𝑡 2 -𝑡 1 | of timestamp 𝑡 1 and 𝑡 2 of user behaviors reveals critical temporal information as well. Figure 1(b) shows that user's click intensity on an item of Taxi-Hailing service varies periodically after user's last click, which we refer to as the Relative Time Cycle Pattern (RTC). Such an observation hints that people tend to hail a taxi again 24 hours later after the last click and the intensity declines as time goes on, but still reaches local peaks every 24 hours. RTC directly illustrates how user's past behavior influences their current intentions after a certain time interval.

Existing works have studied the dynamic temporal patterns of user behaviors and proposed elaborate model structures [3, 10, 23]. However, these studies model time information of user behaviors of all the items without carefully differentiation and some items might be irrelevant to the target item. For example, when the target is an item of Taxi-Hailing service, the user's past behaviors on taxirelated items would benefit the modeling of time cycle patterns and improve recommendation performance. On the contrary, past behaviors on completely irrelevant items (e.g., items of E-Commerce service) may bring adverse impact on time cycle modeling, which are defined as the noises of user behaviors given a specific target item in our paper. Therefore, denoising user behaviors is important for capturing target-related time cycle patterns and improving the recommendation performance.

In this paper, we propose Denoising Time Cycle Recommendation (DiCycle), a framework that disentangles noises in user behaviors and jointly learns the two time cycle patterns, i.e., ATC and RTC. Representing the time has always been a challenge in machine learning. As for ATC, the semantic time such as hour, day, and week, is a discrete variable, and representation of discretized time values can lose some information. In DiCycle, a convolutional module is applied to aggregate the adjacent time representation to mitigate the information loss caused by time discreteness. As for RTC, we utilize a continuous translation-invariant kernel to convert timestamp into embedding representations motivated by Bochner's Theorem [19, 20]. Based on this kernel, the inner product of the embeddings of timestamp 𝑡 1 and 𝑡 2 can characterize the relevance of user's behaviors on 𝑡 1 and 𝑡 2, depending on the time interval | 𝑡 2 -𝑡 1 | . Moreover, a gated filter unit softly selects a subset from user behaviors based on its relevance to the target item, filtering out noises for time cycle modeling. Combined with two temporal representations of ATC and RTC, a time cycle attention is applied to generate the final representation, indicating whether the user will click the target item from the perspective of time cycle pattern.

We verify DiCycle on three public benchmarks and a real-world dataset from our company. The results show the proposed method can effectively capture diverse time cycle patterns of user behaviors and prominently improve the recommendation performance.

## 2 RELATED WORK

Recently, there have been great advances in modeling dynamic patterns of user behaviors in deep learning, including recursive neural network (RNN) [15], convolutional neural network [16, 18, 21] and attention mechanism based structures [9, 24-26]. These methods consider the sequential nature of user behaviors, processing user behavior sequences with natural language processing techniques. Session-based approaches separate user behaviors according to their occurring time and model user short-term preferences, indirectly incorporating temporal information [2, 5]. More recent researches directly focus on modeling temporal information, demonstrating its importance on many tasks, such as sequence prediction, and click-through rate (CTR) prediction [3, 19, 20, 23, 27]. TimeLSTM equips LSTM with time gates to model time intervals, which can better capture both user's short-term and long-term interests [27]. Several time-aware frameworks are specially designed for recommender systems, which learns heterogeneous temporal patterns of user preference [3, 4, 10, 23]. However, how to extract denoised temporal information from user behavior for end-to-end applications as described above remains a challenge. The proposed DiCycle attempts to solve this problem, which is introduced in detail in the next section.

## 3 DENOISING TIME CYCLE MODELING

In this section, we first define the problem formally, and then introduce the approach of time encoding for both ATC and RTC.

Together with a gated filter unit to denoise user behaviors, we finally propose DiCycle for both ATC and RTC, the overall structure of which is shown in Figure 2.

## 3.1 Problem Formulation

Given a user 𝑢 ∈ U , a target item 𝑖 ∈ I and target timestamp 𝑡 , our CTR task is to predict the probability that user 𝑢 will click item 𝑖 at timestamp 𝑡 as 𝑃 ( 𝑦 = 1 | 𝑢, 𝑖, 𝑡 ) , where 𝑦 = 1 refers to click and 𝑦 = 0 refers to non-click. The target behavior to be predicted is defined as 𝑠 𝑢 𝑎 = ( 𝑖, 𝑡 ) . Behaviors of user 𝑢 before time 𝑡 are organized as 𝑆 𝑢 = { 𝑠 𝑢 𝑘 | 1 ≤ 𝑘 ≤ 𝑙 𝑢 } = {( 𝑖 𝑢 𝑘 , 𝑡 𝑢 𝑘 )| 1 ≤ 𝑘 ≤ 𝑙 𝑢 } , where ( 𝑖 𝑢 𝑘 , 𝑡 𝑢 𝑘 ) means that user 𝑢 clicked on item 𝑖 𝑢 𝑘 at time 𝑡 𝑢 𝑘 , and 𝑡 𝑢 1 ≤ 𝑡 𝑢 2 ≤ · · · ≤ 𝑡 𝑢 𝑙 𝑢 ≤ 𝑡 holds. Thus the CTR probability in our task turns to 𝑃 ( 𝑦 = 1 | 𝑆 𝑢 , 𝑖, 𝑡 ) . In the following, we omit 𝑢 in the notation for convenience.

## 3.2 Time Encoding Unit

In this section, we introduce the time encoding unit, including the representation learning of absolute time and of relative time

3.2.1 Learning Absolute Time Representation for ATC. Normally, for timestamp 𝑡 , an embedding layer is applied to encode it into semantic time embeddings with different granularities, containing hour ℎ , weekday 𝑤 , month 𝑚 and etc, which can be formulated as Λ 𝑔 ( 𝑡 ) ∈ R 𝑑 for granularity ( 𝑔 ∈ 𝐺 = { ℎ, 𝑤,𝑚, · · · }) . However, time continuity is broken in such a modeling. For example, the time point of 17:50 and 18:10 is quite close, but the hour time is 17 and 18, respectively, whose semantic embedding could be entirely different. To combat such a loss of continuity of time, we propose an absolute time convolution module, which symmetrically incorporates surrounding time slots. Take time 17:50 and the granularity of hour for an example. Instead of just using the information of hour 17, its embedding by the granularity of hour is combined with the embedding of its surrounding hours, i.e., hour 15, 16, 17, 18, 19 and etc. Formally, the surrounding time embeddings of time 𝑡 by granularity 𝑔 is denoted as Λ 𝑆 𝑔,𝑗 ( 𝑡 ) = [ Λ 𝑔 ( 𝑡 -𝑗 ) , · · · , Λ 𝑔 𝑡 0 ) , · · · , Λ 𝑔 ( 𝑡 + 𝑗 )] T ∈ R ( 2 𝑗 + 1 ) × 𝑑 , where 𝑗 denotes the surrounding range of the target time and Λ 𝑔 ( 𝑡 0 ) = Λ 𝑔 ( 𝑡 ) holds.

With the convolution kernel 𝐾 ∈ R 𝑛 × 𝑑 and the activation function ReLU, where 𝑛 is the kernel size, we conduct one-dimensional convolution on Λ 𝑆 𝑔,𝑗 ( 𝑡 ) to get Λ 𝐶 𝑔,𝑗 ( 𝑡 ) ∈ R ( 2 𝑗 + 1 ) × 𝑑 . Subsequently, max-pooling is conducted on the first dimension of Λ 𝐶 𝑔,𝑗 ( 𝑡 ) to get Λ 𝑃 𝑔,𝑗 ( 𝑡 ) ∈ R 𝑑 . CNN has the capability of capturing local features, which enables us to capture the similarities in surrounding time and maintain some time continuity.

As is shown in Figure 2, we set the surrounding range 𝑗 from 0 to 𝐽 ( 𝐽 = 2 in default), based on which the impact of the surrounding time gradually diminishes as it goes larger. The final representation of absolute time Λ ( 𝑡 ) is composed of all the granularities as

<!-- formula-not-decoded -->

3.2.2 Learning Relative Time Representation for RTC. Relative time is defined as the time interval between the target timestamp 𝑡 and the timestamp 𝑡 𝑘 in user's past behaviors, which reflects how much influence her past behaviors are on current intention. We aim to find a mapping function Φ that transforms time interval from time domain T to 𝑑 -dimensional vector space, preserving the evolving nature of time information. In this way, the information of time interval | 𝑡 2 -𝑡 1 | of timestamps 𝑡 1 and 𝑡 2 can be extracted by the inner product &lt; Φ ( 𝑡 1 ) , Φ ( 𝑡 2 ) &gt; . Therefore, the objective turns to learning a translation-invariant temporal kernel K( 𝑡 1 , 𝑡 2 ) = 𝜓 ( 𝑡 2 -𝑡 1 ) = &lt; Φ ( 𝑡 1 ) , Φ ( 𝑡 2 ) &gt; , where 𝜓 : [-𝑡 𝑚𝑎𝑥 , 𝑡 𝑚𝑎𝑥 ] → R . Inspired by Bochner's theorem [19, 20], the mapping function is defined as

Figure 2: The overall architecture of the proposed model.

<!-- image -->

<!-- formula-not-decoded -->

where 𝜔 1 , 𝜔 2 , · · · , 𝜔 𝑑 / 2 are trainable parameters.

## 3.3 Gated Filter Unit

As mentioned in the previous section, various temporal patterns are supposed to be extracted from user behaviors that are highly related to the target item. For such a purpose, we propose a gated filter unit that weighs each element of user behavior 𝑆 dynamically.

Suppose that the target item embedding is 𝑒 𝑎 and the item embedding in the user behavior 𝑆 is { 𝑒 𝑘 | 1 ≤ 𝑘 ≤ 𝑙 } . Note that item embedding dose not contain time information. The gated weight between the target item and 𝑘 -th behavior 𝑠 𝑘 is expressed as

<!-- formula-not-decoded -->

where where 𝛿 𝑡ℎ𝑟𝑒𝑑 is a hyper-parameter and can be cross validated using the test dataset for best performance. Here we use cos (·) as the activation, which performs well in practice. Additionally, f 𝑘 is generated by a piece-wise function with threshold 𝛿 𝑡ℎ𝑟𝑒𝑑 . Only when the gated weight reaches this threshold, its value remains; otherwise, it is set to 0. Our purpose is to completely filter out the most irrelevant behaviors, which are defined as noises in this paper, and reserve the relevant behaviors with the weights f 𝑘 .

## 3.4 Time Cycle Modeling

Together with time encoding unit and gated filter unit, we finally propose the time cycle modeling. Suppose the time embedding of 𝑘 -th behavior 𝑠 𝑘 is 𝑞 𝑘 and the target time embedding of the target behavior 𝑠 𝑎 is 𝑞 𝑎 . We aggregate the absolute time representation and relative time representation to obtain the time embeddings as

<!-- formula-not-decoded -->

where 𝑡 is the target timestamp and Φ ( 0 ) for 𝑞 ( 𝑎 ) denotes that time interval is 0. As is shown in Figure 2, the model is divided into two modules: user interest module and time cycle module. We introduce these two modules in detail in the next subsections.

3.4.1 User Interest Module. The general structure of user interest module is quite similar to existing models like DIN [26]. The only difference is that we add time embedding to the item of user behaviors and to the target item as side information. Following the notation above, the embedding of user behavior 𝑆 is denoted as { 𝑟 𝑘 | 1 ≤ 𝑘 ≤ 𝑙 } , where 𝑟 𝑘 = 𝑒 𝑘 + 𝑞 𝑘 , and the target embedding of the target behavior 𝑠 𝑎 is computed by 𝑟 𝑎 = 𝑒 𝑎 + 𝑞 𝑎 . Subsequently, the target attention is conducted between 𝑟 𝑎 and { 𝑟 𝑘 | 1 ≤ 𝑘 ≤ 𝑙 } , which is then in the weighted sum over 𝑟 𝑘 to yield the output embedding 𝑟 of user interest module.

<!-- formula-not-decoded -->

Table 1: Statistics of the datasets, including the numbers of the users, items, interactions, average interactions per user (Inter. Avg. for short in the table) and train samples.

| Dataset   | Users       | Items     | Inter.     |   Avg. | Samples     |
|-----------|-------------|-----------|------------|--------|-------------|
| LastFM    | 627         | 341,169   | 14,612,388 | 23,305 | 1,461,464   |
| ML-1M     | 6,040       | 3,416     | 999,611    |    165 | 390,164     |
| Books     | 264,522     | 954,865   | 13,616,232 |     51 | 2,577,098   |
| IndRec    | 0.1 billion | 1 million | 20 billion |    200 | 0.5 billion |

3.4.2 Time Cycle Module. This module aims to learn the diverse time cycle patterns of user behaviors, which are denoised by the gated weight f 𝑘 . Time embedding 𝑞 𝑘 of k-th behavior 𝑠 𝑘 turns to ˜ 𝑞 𝑘 = f 𝑘 · 𝑞 𝑘 . Some elements of ˜ 𝑞 = { ˜ 𝑞 𝑘 | 1 ≤ 𝑘 ≤ 𝑙 } are directly set to zero vectors by Equation 3, which are regarded as noises and can be harmful for time cycle modeling.

Next, the time cycle attention is applied to capture time cycle patterns based on denoised time embedding sequence ˜ 𝑞 , where the time cycle attention is formulated as:

<!-- formula-not-decoded -->

where ℎ is the output embedding of time cycle module and 𝑊 𝑄 ∈ R 𝑑 × 𝑑 , 𝑊 𝐾 ∈ R 𝑑 × 𝑑 , 𝑊 𝑉 ∈ R 𝑑 × 𝑑 are parameter matrices for linear projections. Equipped with absolute time representation Λ (·) and relative time representation Φ (·) , the time cycle attention can find time cycle patterns that are relevant to the target item in the user behaviors. We concatenate the output embedding of user interest module 𝑟 and the output embedding of time cycle module ℎ , then pass it into a 3-layer fully connected neural network. With activation function Sigmoid (·) , we obtain the probability 𝑃 ( 𝑦 𝑗 ) that user 𝑢 will click on item 𝑖 . Finally, the commonly used cross-entropy loss is chosen to train the model, which is formulated as

<!-- formula-not-decoded -->

## 4 EXPERIMENTS

## 4.1 Experimental Settings

Datasets . We conduct experiments on three public benchmarks, including LastFM 1 [1], ML-1M 2 [7], Amazon Books 3 [12], and a real-world dataset extracted from a recommendation scenario of our company, which is named IndRec. The interactions of user and item in these datasets include click, review, and listening to music, which are uniformly termed as interaction in the following. Statistical details of these datasets are presented in Table 1. For all the public datasets, the user-item interactions are regarded as positive samples, and negative samples are randomly sampled from the items that users have not interacted with, where the sampling rate of positive and negative samples is 1:1. We treat the last item of the behavior as test data for each user, and use the remaining for training, following [9, 14]. IndRec is extracted from two-week online logs, where samples clicked by the users are labeled as positive and samples exposed but not clicked are labeled as negative. For each sample of these datasets, the length of user behaviors is set to 200, 200, 100, 100 for LastFM, ML-1M, Books, and IndRec, respectively.

1 https://www.last.fm/

2 https://grouplens.org/datasets/movielens/1m/

3 http://deepyeti.ucsd.edu/jianmo/amazon/index.html

Baselines . Several the state-of-the-art methods serve as baselines in the experiments. We use simple LR (Logistic Regression) and fully connected DNN (Deep Neural Network) for comparison. As for other baselines, DIN [26] uses target attention to model user behaviors. SASRec [9] is a self-attention recommendation framework to model user behaviors. GRU4Rec [8] is a sessionbased method with structured GRU. Time-LSTM [27] is a variant of LSTM equipped with time gates to model time intervals. TiSASRec [10] explicitly models the timestamps of interactions with self attention to explore the influence of different time intervals. TimelyRec [3] jointly learns heterogeneous temporal patterns of user preference.

Parameters and Evaluations. The dimension 𝑑 is set to 64, the batch size is 128 and the learning rate is 1 𝑒 -4 for three public datasets, and we set such numbers to 16, 1024, and 1 𝑒 -4 for IndRec, respectively. We adopt two commonly used metrics to evaluate the performance, including the area under the ROC curve (AUC) and the group weighted area under the ROC curve (GAUC). Moreover, we also report the relative improvement of DiCycle over the best baselines to show its superiority, which is measured following [13, 22] by

<!-- formula-not-decoded -->

where the 𝑚𝑒𝑡𝑟𝑖𝑐 is AUC or GAUC.

## 4.2 Results and Discussions

Performance . The overall comparison results are listed in Table 2. The RelaImpr in AUC of DiCycle over the best baseline is 12.94%, 5.35%, 2.61%, 2.08% for LastFM, ML-1M, Books, and IndRec, respectively; the RelaImpr in GAUC of DiCycle over the best baseline is 17.31%, 5.13%, 2.82%, 3.14% for LastFM, ML-1M, Books, and IndRec, respectively. The results show that DiCycle performs consistently better than all of the state-of-the-art baselines in four datasets, demonstrating its effectiveness in learning denoised temporal patterns in user behaviors. Moreover, an extensive ablation study is conducted with DiCycle to analyze the effect of each module, where the absolute time encoding, the relative time encoding, and the time cycle module are removed, respectively. The results in Table 3 show that all of these modules are critical to achieving final performance in the four datasets. The relative encoding has a higher effect than the absolute time encoding in Books and IndRec, whereas the opposite holds in LastFM and ML-1M. It is illustrated that the time cycle module plays a decisive role in DiCycle, without which there is a performance degradation of -9.35%, -3.99%, -4.22%, and -2.36% for LastFM, ML-1M, Books, and IndRec, respectively.

To provide intuitive evidence for recommendation by DiCycle, we gradually change the target timestamp given a certain user and item pair to see the variance of the prediction. As is shown in Figure 3, we select two interaction records from IndRec. For the first record, the user had many interactions with the target item in her past behaviors whereas very few is observed for the second one. We gradually add additional time to the target timestamp, 3600 seconds for each time. Corresponding to the first record, predicted score in terms of the CTR probability in Figure 3(a) show strong temporal patterns. It has a local peak every 24 hours, and the overall score range declines at first and then gets larger with the change of day. The score in Figure 3(b) gradually declines over time, where time cycle patterns are comparatively weaker. DiCycle filters out noises in user behaviors and can select behaviors highly related to the target time, which explains the difference between Figure 3(a) and Figure 3(b). We conclude from this case study that DiCycle has a strong capability to capture time cycle patterns for users with many interactions on items that are highly relevant to the target item.

Table 2: Performance comparison on four datasets. The methods of the best and the second best performance are in bold and underlined, respectively.

| Dataset   | Metric   |     LR |    DNN |    DIN |   SASRec |   GRU4Rec |   TimeLSTM |   TiSASRec |   TimelyRec |   DiCycle | RelaImpr   |
|-----------|----------|--------|--------|--------|----------|-----------|------------|------------|-------------|-----------|------------|
| LastFM    | AUC      | 0.7353 | 0.7376 | 0.7368 |   0.7475 |    0.7389 |     0.7479 |     0.7480 |      0.7460 |    0.7801 | 12.94%     |
| LastFM    | GAUC     | 0.7418 | 0.7416 | 0.7432 |   0.7480 |    0.7423 |     0.7524 |     0.7464 |      0.7416 |    0.7961 | 17.31%     |
| ML-1M     | AUC      | 0.7883 | 0.8532 | 0.8561 |   0.8687 |    0.8598 |     0.8622 |     0.8648 |      0.8754 |    0.8955 | 5.35%      |
| ML-1M     | GAUC     | 0.7962 | 0.8535 | 0.8498 |   0.8651 |    0.8601 |     0.8619 |     0.8548 |      0.8721 |    0.8912 | 5.13%      |
| Books     | AUC      | 0.7647 | 0.7802 | 0.7804 |   0.7820 |    0.7789 |     0.7833 |     0.7912 |      0.7875 |    0.7988 | 2.61%      |
| Books     | GAUC     | 0.7594 | 0.7597 | 0.7609 |   0.7546 |    0.7521 |     0.7611 |     0.7673 |      0.7691 |    0.7767 | 2.82%      |
| IndRec    | AUC      | 0.7736 | 0.7869 | 0.7930 |   0.7944 |    0.7943 |     0.7968 |     0.7979 |      0.7984 |    0.8046 | 2.08%      |
| IndRec    | GAUC     | 0.7198 | 0.7283 | 0.7331 |   0.7369 |    0.7352 |     0.7375 |     0.7379 |      0.7386 |    0.7461 | 3.14%      |

Table 3: Ablation study. Metric AUC is adopted for evaluation.

Figure 3: (a), (b): Changing next interaction time for users whohadmanyinteractions and a few interactions with target item, respectively.

| Dataset                  |   LastFM |   ML-1M |   Books |   IndRec |
|--------------------------|----------|---------|---------|----------|
| DiCycle                  |   0.7801 |  0.8955 |  0.7988 |   0.8046 |
| Remove Absolute Time     |   0.7761 |  0.8906 |  0.7941 |   0.8036 |
| Remove Relative Time     |   0.7764 |  0.8928 |  0.7932 |   0.8016 |
| Remove Time Cycle Module |   0.7539 |  0.8797 |  0.7862 |   0.7974 |

<!-- image -->

## 5 CONCLUSION

In this paper, we proposed DiCycle to capture dynamic time cycle patterns. Combined with the gated filter unit, DiCycle can filter out the noises in user behaviors, which is defined as the subset of user behaviors that are irrelevant to the target item. DiCycle is able to model both absolute time pattern and relative time pattern in user behaviors which have high relevance to the target item. Extensive experiments on public benchmarks and a real-world dataset verified the effectiveness of DiCycle.

## REFERENCES

- [1] Òscar Celma Herrada et al. 2009. Music recommendation and discovery in the long tail . Universitat Pompeu Fabra.
- [2] Xu Chen, Hongteng Xu, Yongfeng Zhang, Jiaxi Tang, Yixin Cao, Zheng Qin, and Hongyuan Zha. 2018. Sequential recommendation with user memory networks. In WSDM . 108-116.
- [3] Junsu Cho, Dongmin Hyun, SeongKu Kang, and Hwanjo Yu. 2021. Learning Heterogeneous Temporal Patterns of User Preference for Timely Recommendation. In Proceedings of the Web Conference 2021 . 1274-1283.
- [4] Felipe Soares da Costa and Peter Dolog. 2019. Collective embedding for neural context-aware recommender systems. In Proceedings of the 13th ACM Conference on Recommender Systems . 201-209.
- [5] Yufei Feng, Fuyu Lv, Weichen Shen, Menghan Wang, Fei Sun, Yu Zhu, and Keping Yang. 2019. Deep session interest network for click-through rate prediction. arXiv preprint arXiv:1905.06482 (2019).
- [6] Huifeng Guo, Ruiming Tang, Yunming Ye, Zhenguo Li, and Xiuqiang He. 2017. DeepFM: a factorization-machine based neural network for CTR prediction. In IJCAI . 1725-1731.
- [7] F Maxwell Harper and Joseph A Konstan. 2015. The movielens datasets: History and context. Acm transactions on interactive intelligent systems (tiis) 5, 4 (2015), 1-19.
- [8] Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based recommendations with recurrent neural networks. In ICLR .
- [9] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In 2018 IEEE International Conference on Data Mining (ICDM) . IEEE, 197-206.
- [10] Jiacheng Li, Yujie Wang, and Julian McAuley. 2020. Time interval aware selfattention for sequential recommendation. In Proceedings of the 13th international conference on web search and data mining . 322-330.
- [11] Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, and Guangzhong Sun. 2018. xdeepfm: Combining explicit and implicit feature interactions for recommender systems. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery &amp; data mining . 1754-1763.
- [12] Jianmo Ni, Jiacheng Li, and Julian McAuley. 2019. Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) . 188-197.
- [13] Qijie Shen, Wanjie Tao, Jing Zhang, Hong Wen, Zulong Chen, and Quan Lu. 2021. SAR-Net: A Scenario-Aware Ranking Network for Personalized Fair Recommendation in Hundreds of Travel Scenarios. In Proceedings of the 30th ACM International Conference on Information &amp; Knowledge Management . 4094-4103.
- [14] Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. 2019. BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM international conference on information and knowledge management . 1441-1450.
- [15] Yong Kiam Tan, Xinxing Xu, and Yong Liu. 2016. Improved recurrent neural networks for session-based recommendations. In RecSys Workshop . 17-22.
- [16] Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In WSDM . 565-573.
- [17] Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu, and Tat-Seng Chua. 2017. Attentional factorization machines: Learning the weight of feature interactions via attention networks. In IJCAI . 3119-3125.
- [18] Chengfeng Xu, Pengpeng Zhao, Yanchi Liu, Jiajie Xu, Victor S Sheng S. Sheng, Zhiming Cui, Xiaofang Zhou, and Hui Xiong. 2019. Recurrent convolutional neural network for sequential recommendation. In WWW . 3398-3404.
- [19] Da Xu, Chuanwei Ruan, Evren Korpeoglu, Sushant Kumar, and Kannan Achan. 2019. Self-attention with functional time representation learning. Advances in

neural information processing systems 32 (2019).

- [20] Da Xu, Chuanwei Ruan, Evren Korpeoglu, Sushant Kumar, and Kannan Achan. 2020. Inductive representation learning on temporal graphs. arXiv preprint arXiv:2002.07962 (2020).
- [21] An Yan, Shuo Cheng, Wang-Cheng Kang, Mengting Wan, and Julian McAuley. 2019. CosRec: 2D convolutional neural networks for sequential recommendation. In CIKM . 2173-2176.
- [22] Ling Yan, Wu-Jun Li, Gui-Rong Xue, and Dingyi Han. 2014. Coupled group lasso for web-scale ctr prediction in display advertising. In International Conference on Machine Learning . PMLR, 802-810.
- [23] Wenwen Ye, Shuaiqiang Wang, Xu Chen, Xuepeng Wang, Zheng Qin, and Dawei Yin. 2020. Time Matters: Sequential Recommendation with Complex Temporal Information. In SIGIR . 1459-1468.
- [24] Chang Zhou, Jinze Bai, Junshuai Song, Xiaofei Liu, Zhengchao Zhao, Xiusi Chen, and Jun Gao. 2018. Atrank: An attention-based user behavior modeling framework for recommendation. In Proceedings of the AAAI Conference on Artificial Intelligence , Vol. 32.
- [25] Guorui Zhou, Na Mou, Ying Fan, Qi Pi, Weijie Bian, Chang Zhou, Xiaoqiang Zhu, and Kun Gai. 2019. Deep interest evolution network for click-through rate prediction. In Proceedings of the AAAI conference on artificial intelligence , Vol. 33. 5941-5948.
- [26] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In SIGKDD . 1059-1068.
- [27] Yu Zhu, Hao Li, Yikang Liao, Beidou Wang, Ziyu Guan, Haifeng Liu, and Deng Cai. 2017. What to Do Next: Modeling User Behaviors by Time-LSTM.. In IJCAI . 3602-3608.