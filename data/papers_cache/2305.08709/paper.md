## Back Translation for Speech-to-text Translation Without Transcripts

Qingkai Fang 1 , 2 , Yang Feng 1 , 2 ∗

1 Key Laboratory of Intelligent Information Processing

Institute of Computing Technology, Chinese Academy of Sciences (ICT/CAS)

2 University of Chinese Academy of Sciences, Beijing, China {fangqingkai21b, fengyang}@ict.ac.cn

## Abstract

The success of end-to-end speech-to-text translation (ST) is often achieved by utilizing source transcripts , e.g. , by pre-training with automatic speech recognition (ASR) and machine translation (MT) tasks, or by introducing additional ASR and MT data. Unfortunately, transcripts are only sometimes available since numerous unwritten languages exist worldwide. In this paper, we aim to utilize large amounts of target-side monolingual data to enhance ST without transcripts. Motivated by the remarkable success of back translation in MT, we develop a back translation algorithm for ST ( BT4ST ) to synthesize pseudo ST data from monolingual target data. To ease the challenges posed by short-to-long generation and one-to-many mapping, we introduce self-supervised discrete units and achieve back translation by cascading a target-to-unit model and a unit-to-speech model. With our synthetic ST data, we achieve an average boost of 2.3 BLEU on MuST-C En → De, En → Fr, and En → Es datasets. More experiments show that our method is especially effective in lowresource scenarios. 12

## 1 Introduction

End-to-end speech-to-text translation (ST) means directly translating speech in the source language to target text without generating source transcripts (Bérard et al., 2016; Duong et al., 2016). Different from traditional cascading methods which first transcribe the speech with automatic speech recognition (ASR) and then translate the transcripts into the target text with machine translation (MT), end-to-end ST has the potential to reduce latency and avoid error propagation. Hence, it has drawn much attention and achieved great success in recent years (Anastasopoulos et al., 2021, 2022).

∗ Corresponding author: Yang Feng.

1 Code is publicly available at https://github.com/ ictnlp/BT4ST .

2 Examples of synthetic ST data are available at https: //bt4st.github.io/ and Appendix B.

However, it is challenging to train an end-to-end ST model with only speech-translation pairs. Traditional cascaded models learn cross-modal mapping with ASR and cross-lingual mapping with MT. In contrast, end-to-end ST requires simultaneous cross-modal and cross-lingual mapping, which is more complicated and usually relies on more training data. However, the amount of ST data is usually limited due to the high cost of data collection, so the ST model trained with only speech-translation pairs is usually unsatisfactory.

To tackle these problems, researchers often utilize source transcripts to assist ST training by introducing auxiliary ASR and MT tasks. With abundant ASR and MT data, the ASR task can help the model learn better cross-modal mapping, while the MT task can help learn better cross-lingual mapping, which can significantly improve ST as shown in recent ST studies (Wang et al., 2020a; Xu et al., 2021; Ye et al., 2021; Fang et al., 2022). Unfortunately, source transcripts are not always available. It is estimated that there are around 3000 unwritten languages in the world which have no orthography for transcription 3 . For those languages, we can no longer leverage source transcripts to help with training, so many of the latest techniques fail to benefit them.

How to train a stronger ST model without transcripts? In this paper, we address this question from the perspective of data augmentation. Although ASR and MT data are unavailable, a large amount of target-side monolingual data is still easily accessible. Motivated by the success of back translation in MT (Sennrich et al., 2016; Edunov et al., 2018), we aim to develop a back translation algorithm for ST ( BT4ST ) to synthesize pseudo ST data from monolingual target data. However, compared to text-to-text back translation, generating source speech from the target text without source transcripts is much more challenging 4 . First, the length of text is usually only tens or hundreds, but the length of speech is about tens of thousands 5 . Therefore, the model is required to generate an extremely long sequence from a short sequence without the assumption of monotonic alignment, which is a more difficult sequential decision problem. Second, the conversion from text to speech is a one-to-many mapping problem due to the variations in speech (Chen et al., 2021; Ren et al., 2021). For example, the pronunciation of the same content may differ among speakers.

[3 https://www.ethnologue.com/](https://www.ethnologue.com/)

To address these challenges, we introduce selfsupervised discrete units of the source speech as intermediate representations, and achieve back translation by cascading a target-to-unit model and a unit-to-speech model. Since the length of unit sequences is similar to the length of target characters 6 , and there is a monotonic assumption between the unit sequence and the source speech, the shortto-long generation problem in target-to-unit and unit-to-speech models can be greatly alleviated. Besides, discrete units can disentangle content information from other variation information ( e.g. , pitch and speaker) (Polyak et al., 2021), which eases the one-to-many mapping problem in the targetto-unit model 7 . We also introduce a speaker encoder to provide speaker information as input to the unit-to-speech model, which allows us to generate diverse source speeches by coupling different speaker representations. Following this pipeline, we can synthesize large amounts of pseudo ST data from monolingual target data without requiring transcripts. Finally, we train our ST model with both synthetic and real data.

We conduct experiments on MuST-C En → De, En → Fr, and En → Es datasets. By leveraging about 5M additional monolingual target data for each language pair, we achieve an average improvement of 2.3 BLEU compared with the strong baseline. We also observe that our approach is more effec- tive in low-resource scenarios, yielding a boost of 5.6 BLEU when only 100 hours of ST data are available. In addition, we generate multiple different pseudo datasets with a simple Diverse BT4ST method, train several models separately with each dataset, and ensemble them together. By ensembling five models, we achieve an average boost of 4.0 BLEU in three translation directions.

4 When transcripts are available, we can decompose it into two sub-tasks: MT for generating transcripts and textto-speech (TTS) for speech synthesis, which becomes much more manageable.

5 For 16kHz audio waveform, 1 second of speech corresponds to a sequence of 16,000 samples.

6 1 second of speech corresponds to 50 discrete units.

7 It should be noted that the one-to-many mapping problem still exist due to the many-to-many mappings between transcripts and translations, but it is not the focus of this work.

## 2 Background

Our work focuses on developing a back translation algorithm for ST. We first introduce the task definition and model architecture of ST in Section 2.1, and then introduce the concept of back translation in Section 2.2.

## 2.1 Speech-to-text Translation

Task Definition The goal of speech-to-text translation (ST) is to translate speech in one language into text in another language. We denote the source speech as x = ( x 1 , ..., x I ) , where I is the length of the audio waveform. The model generates the target sentence y = ( y 1 , ..., y J ) , where J is the length of the target text. In this paper, we assume that transcripts of the source speech are not available, which is a more general scenario considering about 3000 unwritten languages in the world.

Model Architecture Our ST model consists of three stacked modules: acoustic encoder , length adaptor , and translation model . The acoustic encoder is a HuBERT (Hsu et al., 2021) model pretrained on unlabelled audio data, which can generate meaningful representations for the source speech. The length adaptor (Li et al., 2021) is a series of convolutional layers to shrink the length of speech representations by a factor of 4. The translation model is a Transformer (Vaswani et al., 2017) with N encoder layers and N decoder layers, which takes the shrunken speech representations as input and outputs the target sentence. We train the ST model by minimizing the cross-entropy loss:

<!-- formula-not-decoded -->

## 2.2 Back Translation

Back translation (BT) is a simple and effective method to leverage target-side monolingual data in neural machine translation (NMT). Formally, given a parallel corpus D = { ( x ( n ) , y ( n ) ) } N n =1 , and a monolingual corpus of the target language T = { y ( m ) } M m =1 , we first train a target-tosource model on D . Next, we use this model to generate additional pseudo parallel data ˜ D = { ( ˜ x ( m ) , y ( m ) ) } M m =1 from the monolingual corpus T . Finally, ˜ D can be used as a complement to D to train a stronger source-to-target model.

Figure 1: The overall architecture of our model. RUnit : reduced discrete units; OUnit : original discrete units.

<!-- image -->

## 3 Method: BT4ST

How to acquire a target-to-source model for ST given a ST parallel corpus? Directly generating source speech from the target text is a challenging problem. Inspired by recent success in self-supervised discrete representation learning for speech (Baevski et al., 2020; Hsu et al., 2021; Lakhotia et al., 2021), we first transform the source speech into a sequence of discrete units with a speech pre-trained model (Section 3.1), which is used as an intermediate representation in the back translation process. With discrete units, we train a target-to-unit model (Section 3.2) and a unit-tospeech model (Section 3.3) on the parallel corpus, where the former predicts the sequence of discrete units corresponding to the source speech, and the latter converts discrete units into waveform. The model architecture is illustrated in Figure 1.

## 3.1 Unit-based Speech Representation

We use the pre-trained HuBERT (Hsu et al., 2021) model to generate discrete units corresponding to the source speech following Lee et al. (2022a,b). HuBERT generates 50Hz continuous representations for the input speech. We apply the K-means clustering algorithm to the continuous representations of the training data, and then transform the continuous representations into the corresponding cluster indices, i.e. , discrete units. Finally, the input speech x = ( x 1 , ..., x I ) is converted into a sequence of discrete units z = ( z 1 , ..., z T ) , z t ∈ { 0 , 1 , ..., K -1 } , ∀ 1 ≤ t ≤ T , where K is the number of clusters, and T is the number of frames where T = glyph[floorleft] I 320 glyph[floorright] . There are two advantages to use discrete units as intermediate representations rather than predicting audio waveform directly. First, the sequence of discrete units is much shorter than the audio waveform, alleviating the difficulty of short-to-long generation. Second, discrete units can disentangle speech content from the pitch and speaker information (Polyak et al., 2021), which eases the one-to-many mapping problem.

## 3.2 Target-to-unit Model

Our target-to-unit model is a Transformer-based sequence-to-sequence model, which predicts discrete units of the source speech based on the target text. The target text y is tokenized as characters and fed to the encoder. For the discrete units z , we first merge repeating units into a single one to obtain the reduced discrete units z ′ = ( z ′ 1 , ..., z ′ T ′ ) following Lee et al. (2022a). For example, (1 , 1 , 2 , 2 , 2 , 3 , 4 , 4) will collapse to (1 , 2 , 3 , 4) . We then train the model with reduced discrete units as the target, which can accelerate training and inference. The training objective is as follows:

<!-- formula-not-decoded -->

## 3.3 Unit-to-speech Model

Our unit-to-speech model is a unit-based HiFiGAN vocoder (Kong et al., 2020) as proposed in Polyak et al. (2021). It takes the reduced discrete units as input and generates the waveform. The model consists of four modules: duration predictor , speaker encoder , generator , and discriminator .

Duration Predictor As the output of the targetto-unit model is reduced discrete units, we add a duration predictor (Ren et al., 2021) to predict the duration of each unit. It consists of two 1Dconvolutional layers with ReLU activation, each followed by layer normalization and dropout. Finally, a linear layer projects the hidden state into a scalar as duration. We denote the predicted duration vector as d = ( d 1 , ..., d T ′ ) , and the ground truth as d ∗ = ( d ∗ 1 , ..., d ∗ T ′ ) . The duration predictor is optimized with Mean Squared Logarithmic Error (MSLE) between ground truth and predictions as:

<!-- formula-not-decoded -->

Given the duration vector, we expand reduced discrete units by repeating each unit. For example, given z ′ = (1 , 2 , 3 , 4) and the corresponding duration vector d = (2 , 3 , 1 , 2) , the expanded sequence becomes z = (1 , 1 , 2 , 2 , 2 , 3 , 4 , 4) . We use ground truth duration during training and predicted one during inference. Finally, the units are converted to continuous representations via a look-up table.

Speaker Encoder Discrete units contain little speaker information, but ST corpus usually contains speech from multiple speakers. To ease the one-to-many mapping problem in speech synthesis, we introduce a speaker encoder to extract the speaker information. The speaker encoder is a pretrained speaker verification network (Wan et al., 2018), which extracts a single 256-dimensional speaker embedding from the speech. The speaker embedding is then concatenated to the representation of each unit. During inference, we randomly select a speaker embedding from the training set for each sample. It allows us to synthesize a pseudo ST dataset containing multiple speakers.

Generator and Discriminator The generator and discriminator are the same as the original HiFiGAN (Kong et al., 2020). The generator consists of several stacked blocks, each containing a transposed convolution layer followed by multiple residual blocks. It takes the concatenated features as input and outputs the waveform x . The discriminator contains a Multi-Period Discriminator (MPD) and a Multi-Scale Discriminator (MSD), which are used to identify the periodic or consecutive patterns in the audio. The generator and discriminator

## Algorithm 1: Back Translation for ST

- Input : ST data D = { ( x ( n ) , y ( n ) ) } N n =1 , Target data T = { y ( m ) } M m =1 ,

Data selection ratio ρ

Output:

ST model M x → y

## 1 Procedure BT4ST( D , T , ρ )

- 2 Get reduced discrete units z ′ from x to create D ′ = { ( x ( n ) , z ′ ( n ) , y ( n ) ) } N n =1 , for every ( x , y ) ∈ D
- 3 Train target-to-unit model M z ′ ← y and unit-to-target model M z ′ → y with paired data ( z ′ , y ) in D ′
- 4 Train unit-to-speech model M x ← z ′ with paired data ( x , z ′ ) in D ′

5

<!-- formula-not-decoded -->

6

7

˜ ˜ based

8

- Use M z ′ → y to translate ˜ z ′ into ˜ y , and compute BLEU( ˜ y , y ) , for every ( x , z ′ , y ) ∈ ˜ D

Select top ρ · M samples from ˜ D on BLEU scores, denoted as ˜ D S Training( D , ˜ D S )

## 9 End

- 10 Procedure Training( D , ˜ D S )
- 11 Pre-train ST model M x → y on ˜ D S
- 12 Fine-tune ST model M x → y on D
- 13 End

are trained adversarially. More details about the HiFi-GAN can be found in Appendix A.

## 3.4 Data Selection and Model Training

By cascading the target-to-unit model and unitto-speech model, we can synthesize ST data ˜ D = { ( ˜ x ( m ) , ˜ z ′ ( m ) , y ( m ) ) } M m =1 from the targetside monolingual corpus T = { y ( m ) } M m =1 , where ˜ x is the synthetic speech and ˜ z ′ is the corresponding reduced discrete units. However, the synthetic corpus may contain some low-quality data, which may hurt model training. Therefore, we introduce data selection and 2-stage model training to better utilize the synthetic data.

Data Selection To identify the low-quality data, we train a unit-to-target model to translate the generated units ˜ z ′ back into target text ˜ y , and compute the BLEU score (Papineni et al., 2002) between ˜ y and the ground truth y , i.e., BLEU( ˜ y , y ) . Finally, we only keep the top ρ · M samples according to the BLEU score, where ρ is the selection ratio. The remaining low-quality samples are discarded.

2-stage Model Training After we get the selected synthetic ST data, we train the model in a 2stage manner following Abdulmumin et al. (2021). We first pre-train the model on synthetic data, and then fine-tune the model on real data. This can prevent the noise-infested synthetic data from overwhelming real data during training. Algorithm 1 describes the whole process of our proposed method.

## 4 Experiments

## 4.1 Datasets

MuST-C MuST-C (Di Gangi et al., 2019) is a multilingual speech translation dataset, which contains about 400 hours of English (En) audio clips and corresponding translations in 8 languages: German (De), French (Fr), Spanish (Es), Italian (It), Portuguese (Pt), Dutch (Nl), Romanian (Ro), and Russian (Ru). We conduct experiments on En → De, En → Fr, and En → Es, because these three languages have larger public monolingual corpora.

Monolingual Target Data We use the target text in WMT (Buck and Koehn, 2016) En → De, En → Fr, and En → Es datasets as monolingual target data. Specifically, we use europarl v7 8 , commoncrawl 9 , news commentary v12 10 subsets for all three languages. The source text is never used in the experiments. The detailed statistics of data we used are shown in Table 1.

## 4.2 Model Settings

Target-to-unit Model The target-to-unit model is a Transformer with 6 encoder layers and 6 decoder layers. Each layer comprises 512 hidden states, 4 attention heads, and 2048 feed-forward hidden states. The dropout is 0.3, and the label smoothing is 0.1. For target text, we first lowercase the text and segment it into characters using SentencePiece 11 . For discrete units, we use the pre-trained quantized model 12 , which learns K = 100 clusters

[8 http://statmt.org/wmt13/](http://statmt.org/wmt13/training-parallel-europarl-v7.tgz)

[training-parallel-europarl-v7.tgz](http://statmt.org/wmt13/training-parallel-europarl-v7.tgz)

[9 http://statmt.org/wmt13/](http://statmt.org/wmt13/training-parallel-commoncrawl.tgz)

training-parallel-commoncrawl.tgz from the 6th layer representations of pre-trained HuBERT-Base model 13 , to convert the speech into discrete units. During training, the batch size is 400, and the maximum learning rate is 5e-4. We train the model up to 100k steps with Adam optimizer (Kingma and Ba, 2015). During inference, we use beam search with a beam size of 8 to generate the reduced discrete units.

[10 http://data.statmt.org/wmt17/](http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz)

[translation-task/training-parallel-nc-v12.tgz](http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz)

[11 https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)

[12 https://dl.fbaipublicfiles.com/textless\_nlp/ gslm/hubert/km100/km.bin](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin)

Table 1: Statistics of all datasets. Euro.: europarl v7 ; CC.: commoncrawl ; NC.: news commentary v12 .

| Target   | MuST-C (En → )   | MuST-C (En → )   | Monolingual Data   | Monolingual Data   | Monolingual Data   | Monolingual Data   |
|----------|------------------|------------------|--------------------|--------------------|--------------------|--------------------|
|          | hours            | #samples         | Euro.              | CC.                | NC.                | All                |
| De       | 408              | 234k             | 1.9M               | 2.4M               | 0.3M               | 4.6M               |
| Fr       | 492              | 280k             | 2.0M               | 3.2M               | 0.3M               | 5.5M               |
| Es       | 504              | 270k             | 2.0M               | 1.8M               | 0.3M               | 4.1M               |

The unit-to-target model is the same as the target-to-unit model except for the translation direction, which is used for data selection. During inference, we use greedy search to save time.

Unit-to-speech Model For the unit-to-speech model, the configurations of the generator and discriminator are the same as Polyak et al. (2021). The 1D-convolutional layers in the duration predictor are set to kernel size 3, padding 1, and hidden dimension 256. We use the pre-trained dvector model 14 as the speaker encoder to extract the 256-dimensional speaker embedding. We train the model up to 100k steps with a batch size of 128.

ST Model The ST model contains three stacked modules. We use the pre-trained HuBERT-Base model as the acoustic encoder, which takes the raw 16-bit 16kHz audio wave as input. The length adaptor comprises two 1D-convolutional layers with kernel size 5, stride size 2, padding 2, and hidden dimension 1024. The translation model follows Transformer-Base architecture, containing 6 encoder layers and 6 decoder layers. Each layer comprises 512 hidden states, 8 attention heads, and 2048 feed-forward hidden states. The dropout is 0.1, and the label smoothing is 0.1. We refer to this architecture as HU-TRANSFORMER . We learn a vocabulary of size 8k from the target texts in the MuST-C dataset to segment the target texts into subwords for training the ST model.

To augment the ST model with back translation, we first select the top ρ = 75% of synthetic data based on BLEU scores. The target texts in the synthetic data are segmented into subwords using the same vocabulary. We use Adam optimizer with 4k warm-up steps to pre-train the model on synthetic data up to 300k steps, and fine-tune the model on MuST-C up to 20 epochs. During both pre-training and fine-tuning, each batch contains at most 16M audio frames, and the maximum learning rate is 1e-4. During inference, we average the checkpoints of the last 10 epochs for evaluation. We use beam search with a beam size of 8. SacreBLEU 15 (Post, 2018) is used to compute case-sensitive detokenized BLEU scores and the statistical significance of translation results with paired bootstrap resampling 16 (Koehn, 2004). The length penalty is set to 1.2, 1.8, and 0.6 for En → De, En → Fr, and En → Es, respectively. We implement our model with fairseq 17 (Ott et al., 2019). All models are trained on 4 Nvidia Tesla V100 GPUs.

[13 https://dl.fbaipublicfiles.com/hubert/hubert\_ base\_ls960.pt](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt)

[14 https://github.com/yistLin/dvector](https://github.com/yistLin/dvector)

Table 2: BLEU scores on MuST-C En → De, En → Fr, and En → Es tst-COMMON set. * means the improvements over HU-TRANSFORMER are statistically significant ( p &lt; 0 . 01 ).

| Models                                      | External Data                               | External Data                               | En → De                                     | En → De                                     | En → Fr                                     | En → Fr                                     | En → Es                                     | En → Es                                     | Average                                     | Average                                     |
|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|---------------------------------------------|
|                                             | Audio                                       | Target                                      | BLEU                                        | ∆                                           | BLEU                                        | ∆                                           | BLEU                                        | ∆                                           | BLEU                                        | ∆                                           |
| Previous ST baselines (No transcripts used) | Previous ST baselines (No transcripts used) | Previous ST baselines (No transcripts used) | Previous ST baselines (No transcripts used) | Previous ST baselines (No transcripts used) | Previous ST baselines (No transcripts used) | Previous ST baselines (No transcripts used) | Previous ST baselines (No transcripts used) | Previous ST baselines (No transcripts used) | Previous ST baselines (No transcripts used) | Previous ST baselines (No transcripts used) |
| REVISIT-ST (Zhang et al., 2022a)            | ×                                           | ×                                           | 23.0                                        |                                             | 33.5                                        |                                             | 28.0                                        |                                             | 28.2                                        |                                             |
| W-TRANSF. (Ye et al., 2021)                 | glyph[check]                                | ×                                           | 23.6                                        |                                             | 34.6                                        |                                             | 28.4                                        |                                             | 28.9                                        |                                             |
| Our implementations                         | Our implementations                         | Our implementations                         | Our implementations                         | Our implementations                         | Our implementations                         | Our implementations                         | Our implementations                         | Our implementations                         | Our implementations                         | Our implementations                         |
| HU-TRANSFORMER BT4ST                        | glyph[check]                                | × glyph[check]                              | 24.3                                        |                                             | 34.9                                        |                                             | 28.7                                        |                                             | 29.3                                        |                                             |
|                                             | glyph[check]                                |                                             | 26.6*                                       | +2.3                                        | 36.9*                                       | +2.0                                        | 31.2*                                       | +2.5                                        | 31.6                                        | +2.3                                        |

Figure 2: BLEU scores on MuST-C En → De tst-COMMON set with different data selection ratio ρ .

<!-- image -->

[15 https://github.com/mjpost/sacrebleu](https://github.com/mjpost/sacrebleu)

16 sacreBLEU signature: nrefs:1 | bs:1000 | seed:12345 | case:mixed | eff:no | tok:13a | smooth:exp | version:2.0.0

[17 https://github.com/pytorch/fairseq](https://github.com/pytorch/fairseq)

Baseline Systems We include two baseline systems: REVISIT-ST (Zhang et al., 2022a) and W-TRANSF. (Ye et al., 2021) for comparison. REVISIT-ST is a carefully designed ST baseline including several techniques like parameterized distance penalty (PDP) and CTC-based regularization. W-TRANSF. is a stronger ST baseline with a pretrained acoustic model, which combines Wav2vec 2.0 (Baevski et al., 2020) and Transformer. Both of them are only trained on speech-translation pairs without using any transcripts. Our HUTRANSFORMER is also a strong baseline model trained on speech-translation pairs from MuST-C, and we examine our proposed BT4ST on top of this by adding synthetic ST data.

## 5 Results and Analysis

## 5.1 Results on MuST-C Dataset

Table 2 shows the results on MuST-C En → De, En → Fr, and En → Es tst-COMMON set. First, we observe that our HU-TRANSFORMER is a strong baseline compared with previous baselines. Second, by synthesizing pseudo ST data with our proposed BT4ST and using it to pre-train the model, we achieve an average boost of 2.3 BLEU in three translation directions. It demonstrates that our approach can effectively utilize external monolingual target data to improve the performance of ST.

## 5.2 Impact of Data Selection Ratio

Although previous work (Edunov et al., 2018) found that noisy synthetic data can benefit training, we argue that the extremely low-quality data also hurt performance. As described in Section 3.4, we select top ρ · M synthetic data based on BLEU scores. We constrain ρ in [0% , 25% , 50% , 75% , 100%] for experiments, and the results on MuST-C En → De tst-COMMON set are shown in Figure 2. We find that filtering out the last 25% of samples ( i.e. , ρ = 75% ) gives a 0.4 BLEU improvement (26.2 → 26.6) and performs best. We use ρ = 75% for all experiments.

Table 3: BLEU scores on MuST-C En → De tst-COMMON set with different unit generation methods.

| Methods         |   BLEU |
|-----------------|--------|
| Beam search     |   26.6 |
| Greedy search   |   26.2 |
| Top-10 sampling |   26.4 |
| Sampling        |   26.1 |

Table 4: BLEU scores on MuST-C En → De tst-COMMON set with different types of pseudo data. For multi-speaker data, we report the mean value and standard variance of BLEU scores derived from 3 independent experiments with different random seeds.

| Synthetic Data Types         | BLEU          |
|------------------------------|---------------|
| Multi-speaker Single-speaker | 26.6 ( ± 0.1) |
|                              | 26.4          |

## 5.3 Impact of Unit Generation Methods

The generation method of the target-to-unit model also influences the performance of back translation, as it determines the quality and diversity of synthetic data. We conduct experiments with beam search, greedy search, top-10 sampling 18 , and sampling. As shown in Table 3, beam search performs best over all methods on MuST-C En → De tst-COMMON set. We consider two reasons for this. First, our ST dataset is actually a low-resource setting, containing only less than 300k parallel data 19 . Second, target-to-unit generation is a more difficult task compared to text translation. Therefore, beam search is more effective as it generates high probability outputs, while sampling from the model distribution may produce harmful low-quality data. We use beam search for all experiments.

## 5.4 Single-speaker vs. Multi-speaker Synthetic Data

As described in Section 3.3, we provide speaker embedding as input during speech synthesis, which allows us to synthesize pseudo ST datasets con- taining multiple speakers. To examine whether speaker diversity benefits back translation, we synthesize both multi-speaker pseudo data and singlespeaker pseudo data for experiments. For multispeaker data, we randomly select a speaker embedding from the training set for each sample. For single-speaker data, we use the average speaker embedding on the training set for all samples. As shown in Table 4, we find that the multi-speaker data is slightly better than the single-speaker data, probably because it is closer to the real ST dataset.

18 Top-10 sampling refers to sampling over the ten most likely words.

19 Edunov et al. (2018) find that beam search is more effective than sampling in low-resource settings, while the opposite is true for high-resource settings.

Figure 3: BLEU scores on MuST-C En → De tst-COMMON set with different amount of monolingual and parallel data.

<!-- image -->

## 5.5 Results with Different Amounts of Monolingual/Parallel Data

In this section, we investigate the results under different amounts of monolingual and parallel data. First, we randomly sample 100 hours of ST data (corresponding to about 58K speech-translation pairs) from MuST-C En → De train set to simulate the low-resource setting. We then vary the amount of monolingual data used for back translation. As shown in Figure 3, we observe that (i) the BLEU score keeps increasing with the number of monolingual data regardless of the size of parallel data, and (ii) our approach is particularly effective in the low-resource setting. With only 12.5% monolingual data (about 0.6M), we observe a 3.4 BLEU improvement compared with the baseline. When all 4.6M monolingual data is used, we achieve a more significant boost of 5.4 BLEU.

## 5.6 Diverse BT4ST and Model Ensemble

Model ensemble is a widely-used technique in state-of-the-art MT systems (Barrault et al., 2020; Akhbardeh et al., 2021), which can combine differ- ent single models ( e.g. , models trained on different data) to achieve stronger performance. To synthesize multiple diverse pseudo datasets from a single monolingual corpus, we introduce a simple Diverse BT4ST method. For the target-to-unit model, we activate the dropout modules during beam search decoding. For the unit-to-speech model, we randomly select the speaker embedding as above. By setting different random seeds, we can generate source speeches with different content and different speakers from the target text. In this way, we obtain multiple different pseudo datasets and train several models individually. We then combine these models by computing the token-level average log probability during decoding. As shown in Table 5, model ensemble can significantly boost performance. We achieve an average boost of 4.0 BLEU in three directions by ensembling five models.

Table 5: BLEU scores on MuST-C En → De, En → Fr, and En → Es tst-COMMON set with model ensemble.

| Models                   |   #Models | En → De   | En → De   | En → Fr   | En → Fr   | En → Es   | En → Es   | Average   | Average   |
|--------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
|                          |           | BLEU      | ∆         | BLEU      | ∆         | BLEU      | ∆         | BLEU      | ∆         |
| HU-TRANSFORMER BT4ST     |         1 | 24.3      |           | 34.9      |           | 28.7      |           | 29.3      |           |
|                          |         1 | 26.6      | +2.3      | 36.9      | +2.0      | 31.2      | +2.5      | 31.6      | +2.3      |
|                          |         2 | 27.5      | +3.2      | 38.3      | +3.4      | 32.3      | +3.6      | 32.7      | +3.4      |
| Diverse BT4ST + Ensemble |         3 | 27.9      | +3.6      | 38.8      | +3.9      | 32.4      | +3.7      | 33.0      | +3.7      |
|                          |         4 | 28.1      | +3.8      | 39.0      | +4.1      | 32.6      | +3.9      | 33.2      | +3.9      |
|                          |         5 | 28.0      | +3.7      | 39.0      | +4.1      | 32.8      | +4.1      | 33.3      | +4.0      |

## 5.7 Performance of the Target-to-unit Model

In this section, we report the performance of our target-to-unit models. We evaluate the performance with two metrics: Unit-BLEU and ASR-BLEU. Unit-BLEU is the BLEU score calculated on the reduced discrete unit sequence. ASR-BLEU is the BLEU score calculated on the transcribed text with the open-source ASR-BLEU toolkit 20 . As shown in Table 6, our target-to-unit models achieve promising results on MuST-C De → En, Fr → En, and Es → En tst-COMMON set, indicating that our proposed method can synthesize reasonable ST data.

## 6 Related Work

End-to-end ST End-to-end ST is theoretically attractive due to its advantages in alleviating error propagation and reducing latency, but it also faces many challenges because of data scarcity. Therefore, researchers often leverage source transcripts to help train with auxiliary tasks. Plenty of existing work first pre-train the model with the ASR task (Bansal et al., 2019; Stoian et al., 2020; Wang et al., 2020b), MT task (Han et al., 2021; Fang et al., 2022; Ye et al., 2022; Fang and Feng, 2023; Zhou et al., 2023), or both together (Wang et al., 2020a; Alinejad and Sarkar, 2020; Le et al., 2021; Dong et al., 2021a; Xu et al., 2021), and then fine-tune the model with the ST task, which becomes the de-facto paradigm in recent ST studies. Le et al. (2020); Indurthi et al. (2021); Tang et al. (2021a,b); Dong et al. (2021b); Ye et al. (2021) adopt multitask learning to share knowledge among different tasks to improve ST. Bapna et al. (2021, 2022); Chen et al. (2022); Cheng et al. (2022); Ao et al. (2022); Tang et al. (2022); Zhang et al. (2022b) jointly pre-train the model with speech and text data to learn a unified space for both modalities, which achieve competitive results in ST. Jia et al. (2019); Lam et al. (2022) synthesize ST data with the help of MT model, TTS model and forced alignment tools. However, all of these studies assume that transcripts are available, which does not hold true for large numbers of unwritten languages in the world. Zhang et al. (2022a) first challenge this assumption and propose a set of practices to train a better ST model with only speech-translation pairs. In this paper, we extend this line of research and propose a back translation algorithm to utilize large-scale monolingual target data to improve ST

[20 https://github.com/facebookresearch/fairseq/ tree/ust/examples/speech\_to\_speech/asr\_bleu](https://github.com/facebookresearch/fairseq/tree/ust/examples/speech_to_speech/asr_bleu)

Table 6: Unit-BLEU and ASR-BLEU scores of the target-to-unit models.

| Metrics   |   De → En |   Fr → En |   Es → En |
|-----------|-----------|-----------|-----------|
| Unit-BLEU |      26.5 |      27.9 |      27.2 |
| ASR-BLEU  |      19.0 |      24.5 |      20.8 |

without transcripts.

Back Translation Back translation for NMT was first proposed by Sennrich et al. (2016) and is widely used in state-of-the-art NMT systems (Akhbardeh et al., 2021). Since then, many techniques have been proposed to improve BT such as Iterative BT (Hoang et al., 2018; Dou et al., 2020), Tagged BT (Caswell et al., 2019; Marie et al., 2020), Tag-less BT (Abdulmumin et al., 2021), MetaBT (Pham et al., 2021), HintedBT (Ramnath et al., 2021), and so on. Edunov et al. (2018) investigates different generation methods of BT in large-scale settings. Huang et al. (2021); Liu et al. (2021) focus on combining back translation with pre-training. Xu et al. (2022) combines synthetic data generated by beam search and sampling to better trade off the importance and quality of synthetic data. Despite the success of BT in MT, BT for ST is still a challenging problem. Nguyen et al. (2022b) introduces a pipeline BT method for speech-to-speech translation which cascades an unsupervised MT model and a TTS model. In contrast, our approach can synthesize pseudo ST data from the target-side monolingual corpus without relying on source transcripts.

Discrete Speech Units Discrete units, as a selfsupervised discrete representation of speech, have proved effective on many tasks, such as spoken language modeling (Lakhotia et al., 2021; Kharitonov et al., 2022; Gat et al., 2022; Borsos et al., 2022), speech-to-speech translation (Lee et al., 2022a,b; Popuri et al., 2022; Inaguma et al., 2022; Chen et al.; Li et al., 2022), speech emotion conversion (Kreuk et al., 2021), speech dialogue (Nguyen et al., 2022a), speech resynthesis (Polyak et al., 2021), speaking style conversion (Maimon and Adi, 2022), and so on. In this paper, we achieve back translation for ST by leveraging discrete units and further prove its effectiveness.

## 7 Conclusion and Future Work

In this paper, we develop a back translation algorithm for speech translation, which can synthesize pseudo ST data from monolingual target data without relying on transcripts. We utilize selfsupervised discrete units and achieve back translation by cascading a target-to-unit model and a unit-to-speech model. Experimental results on the MuST-C benchmark demonstrate the superiority of our approach, especially in low-resource settings.

This work focuses on enhancing ST when the source transcripts are unavailable, which is an essential but under-explored issue. We hope our work will draw more attention to this issue from researchers, which will benefit more real-world unwritten languages. In the future, we are interested in exploring how to combine advanced BT techniques ( e.g. , Iterative BT) with our approach.

## Limitations

Our work provides an effective solution to augment ST when source transcripts are unavailable, which could benefit many unwritten languages. However, limited by the publicly available ST datasets, we use English as an unwritten language for experiments, which may slightly differ from realworld unwritten languages. Since we never use transcripts in our approach, we believe our work can shed some light on ST for real-world unwritten languages. We are glad to explore this if there are available datasets in the future.

## Ethics Statement

Our model is developed and evaluated with publicly available datasets: MuST-C and WMT. The pre-trained models we use, like HuBERT and dvector models, are open and permitted for research purposes. Our use of the above artifacts is consistent with their intended use since they are widely used in the speech research community. Although our method could help the speech translation of unwritten languages like some dialects, the performance of the ST model still heavily relies on the amount of ST training data. Therefore, the output of the model is not always reliable and it would be better to be assisted by professional human translators in real applications.

## Acknowledgements

We thank all the anonymous reviewers for their insightful and valuable comments. This work was supported by National Key R&amp;D Program of China (NO. 2018AAA0102502)

## References

Idris Abdulmumin, Bashir Shehu Galadanci, and Aliyu Garba. 2021. Tag-less back-translation. Mach. Transl. , 35(4):519-549.

Farhad Akhbardeh, Arkady Arkhangorodsky, Magdalena Biesialska, Ondˇ rej Bojar, Rajen Chatter- jee, Vishrav Chaudhary, Marta R. Costa-jussa, Cristina España-Bonet, Angela Fan, Christian Federmann, Markus Freitag, Yvette Graham, Roman Grundkiewicz, Barry Haddow, Leonie Harter, Kenneth Heafield, Christopher Homan, Matthias Huck, Kwabena Amponsah-Kaakyire, Jungo Kasai, Daniel Khashabi, Kevin Knight, Tom Kocmi, Philipp Koehn, Nicholas Lourie, Christof Monz, Makoto Morishita, Masaaki Nagata, Ajay Nagesh, Toshiaki Nakazawa, Matteo Negri, Santanu Pal, Allahsera Auguste Tapo, Marco Turchi, Valentin Vydrin, and Marcos Zampieri. 2021. Findings of the 2021 conference on machine translation (WMT21). In Proceedings of the Sixth Conference on Machine Translation , pages 1-88, Online. Association for Computational Linguistics.

Ashkan Alinejad and Anoop Sarkar. 2020. Effectively pretraining a speech translation decoder with machine translation data. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) , pages 8014-8020, Online. Association for Computational Linguistics.

Antonios Anastasopoulos, Loïc Barrault, Luisa Bentivogli, Marcely Zanon Boito, Ondˇ rej Bojar, Roldano Cattoni, Anna Currey, Georgiana Dinu, Kevin Duh, Maha Elbayad, Clara Emmanuel, Yannick Estève, Marcello Federico, Christian Federmann, Souhir Gahbiche, Hongyu Gong, Roman Grundkiewicz, Barry Haddow, Benjamin Hsu, Dávid Javorský, V˘ era Kloudová, Surafel Lakew, Xutai Ma, Prashant Mathur, Paul McNamee, Kenton Murray, Maria Nˇ adejde, Satoshi Nakamura, Matteo Negri, Jan Niehues, Xing Niu, John Ortega, Juan Pino, Elizabeth Salesky, Jiatong Shi, Matthias Sperber, Sebastian Stüker, Katsuhito Sudoh, Marco Turchi, Yogesh Virkar, Alexander Waibel, Changhan Wang, and Shinji Watanabe. 2022. Findings of the IWSLT 2022 evaluation campaign. In Proceedings of the 19th International Conference on Spoken Language Translation (IWSLT 2022) , pages 98-157, Dublin, Ireland (in-person and online). Association for Computational Linguistics.

Antonios Anastasopoulos, Ondˇ rej Bojar, Jacob Bremerman, Roldano Cattoni, Maha Elbayad, Marcello Federico, Xutai Ma, Satoshi Nakamura, Matteo Negri, Jan Niehues, Juan Pino, Elizabeth Salesky, Sebastian Stüker, Katsuhito Sudoh, Marco Turchi, Alexander Waibel, Changhan Wang, and Matthew Wiesner. 2021. FINDINGS OF THE IWSLT 2021 EVALUATION CAMPAIGN. In Proceedings of the 18th International Conference on Spoken Language Translation (IWSLT 2021) , pages 1-29, Bangkok, Thailand (online). Association for Computational Linguistics.

Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, and Furu Wei. 2022. SpeechT5: Unified-modal encoderdecoder pre-training for spoken language processing. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 5723-5738, Dublin, Ireland. Association for Computational Linguistics.

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in Neural Information Processing Systems , 33.

Sameer Bansal, Herman Kamper, Karen Livescu, Adam Lopez, and Sharon Goldwater. 2019. Pretraining on high-resource speech recognition improves low-resource speech-to-text translation. In Proc. of NAACL-HLT , pages 58-68.

Ankur Bapna, Colin Cherry, Yu Zhang, Ye Jia, Melvin Johnson, Yong Cheng, Simran Khanuja, Jason Riesa, and Alexis Conneau. 2022. mslam: Massively multilingual joint pre-training for speech and text. CoRR , abs/2202.01374.

Ankur Bapna, Yu-an Chung, Nan Wu, Anmol Gulati, Ye Jia, Jonathan H Clark, Melvin Johnson, Jason Riesa, Alexis Conneau, and Yu Zhang. 2021. Slam: A unified encoder for speech and language modeling via speech-text joint pre-training. arXiv preprint arXiv:2110.10329 .

Loïc Barrault, Magdalena Biesialska, Ondˇ rej Bojar, Marta R. Costa-jussà, Christian Federmann, Yvette Graham, Roman Grundkiewicz, Barry Haddow, Matthias Huck, Eric Joanis, Tom Kocmi, Philipp Koehn, Chi-kiu Lo, Nikola Ljubeši´ c, Christof Monz, Makoto Morishita, Masaaki Nagata, Toshiaki Nakazawa, Santanu Pal, Matt Post, and Marcos Zampieri. 2020. Findings of the 2020 conference on machine translation (WMT20). In Proceedings of the Fifth Conference on Machine Translation , pages 1-55, Online. Association for Computational Linguistics.

Alexandre Bérard, Olivier Pietquin, Christophe Servan, and Laurent Besacier. 2016. Listen and translate: A proof of concept for end-to-end speech-to-text translation. In NIPS workshop on End-to-end Learning for Speech and Audio Processing .

Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Olivier Teboul, David Grangier, Marco Tagliasacchi, and Neil Zeghidour. 2022. Audiolm: a language modeling approach to audio generation.

Christian Buck and Philipp Koehn. 2016. Findings of the WMT 2016 bilingual document alignment shared task. In Proceedings of the First Conference on Machine Translation: Volume 2, Shared Task Papers , pages 554-563, Berlin, Germany. Association for Computational Linguistics.

Isaac Caswell, Ciprian Chelba, and David Grangier. 2019. Tagged back-translation. In Proceedings of the Fourth Conference on Machine Translation (Volume 1: Research Papers) , pages 53-63, Florence, Italy. Association for Computational Linguistics.

- Mingjian Chen, Xu Tan, Bohan Li, Yanqing Liu, Tao Qin, sheng zhao, and Tie-Yan Liu. 2021. Adaspeech: Adaptive text to speech for custom voice. In International Conference on Learning Representations .
- Peng-Jen Chen, Kevin Tran, Yilin Yang, Jingfei Du, Justine Kao, Yu-An Chung, Paden Tomasello, PaulAmbroise Duquenne, Holger Schwenk, Hongyu Gong, Hirofumi Inaguma, Sravya Popuri, Changhan Wang, Juan Pino, Wei-Ning Hsu, and Ann Lee. Speech-to-speech translation for a real-world unwritten language.
- Zhehuai Chen, Yu Zhang, Andrew Rosenberg, Bhuvana Ramabhadran, Pedro J. Moreno, Ankur Bapna, and Heiga Zen. 2022. Maestro: Matched speech text representations through modality matching. In INTERSPEECH .
- Yong Cheng, Yu Zhang, Melvin Johnson, Wolfgang Macherey, and Ankur Bapna. 2022. Mu 2 slam: Multitask, multilingual speech and language models.
- Mattia A. Di Gangi, Roldano Cattoni, Luisa Bentivogli, Matteo Negri, and Marco Turchi. 2019. MuST-C: a Multilingual Speech Translation Corpus. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) , pages 2012-2017, Minneapolis, Minnesota. Association for Computational Linguistics.
- Qianqian Dong, Mingxuan Wang, Hao Zhou, Shuang Xu, Bo Xu, and Lei Li. 2021a. Consecutive decoding for speech-to-text translation. In The Thirty-fifth AAAI Conference on Artificial Intelligence, AAAI .
- Qianqian Dong, Rong Ye, Mingxuan Wang, Hao Zhou, Shuang Xu, Bo Xu, and Lei Li. 2021b. Listen, understand and translate: Triple supervision decouples end-to-end speech-to-text translation. In Proceedings of the AAAI Conference on Artificial Intelligence .
- Zi-Yi Dou, Antonios Anastasopoulos, and Graham Neubig. 2020. Dynamic data selection and weighting for iterative back-translation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) , pages 58945904, Online. Association for Computational Linguistics.
- Long Duong, Antonios Anastasopoulos, David Chiang, Steven Bird, and Trevor Cohn. 2016. An attentional model for speech translation without transcription. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pages 949-959, San Diego, California. Association for Computational Linguistics.
- Sergey Edunov, Myle Ott, Michael Auli, and David Grangier. 2018. Understanding back-translation at scale. In Proceedings of the 2018 Conference on
- Empirical Methods in Natural Language Processing , pages 489-500, Brussels, Belgium. Association for Computational Linguistics.
- Qingkai Fang and Yang Feng. 2023. Understanding and bridging the modality gap for speech translation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics .
- Qingkai Fang, Rong Ye, Lei Li, Yang Feng, and Mingxuan Wang. 2022. STEMM: Self-learning with speech-text manifold mixup for speech translation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 7050-7062, Dublin, Ireland. Association for Computational Linguistics.
- Itai Gat, Felix Kreuk, Ann Lee, Jade Copet, Gabriel Synnaeve, Emmanuel Dupoux, and Yossi Adi. 2022. On the robustness of self-supervised representations for spoken language modeling.
- Chi Han, Mingxuan Wang, Heng Ji, and Lei Li. 2021. Learning shared semantic space for speech-to-text translation. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021 , pages 2214-2225, Online. Association for Computational Linguistics.
- Vu Cong Duy Hoang, Philipp Koehn, Gholamreza Haffari, and Trevor Cohn. 2018. Iterative backtranslation for neural machine translation. In Proceedings of the 2nd Workshop on Neural Machine Translation and Generation , pages 18-24, Melbourne, Australia. Association for Computational Linguistics.
- Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. 2021. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM Trans. Audio, Speech and Lang. Proc. , 29:3451-3460.
- Dandan Huang, Kun Wang, and Yue Zhang. 2021. A comparison between pre-training and large-scale back-translation for neural machine translation. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021 , pages 1718-1732, Online. Association for Computational Linguistics.
- Hirofumi Inaguma, Sravya Popuri, Ilia Kulikov, PengJen Chen, Changhan Wang, Yu-An Chung, Yun Tang, Ann Lee, Shinji Watanabe, and Juan Pino. 2022. Unity: Two-pass direct speech-to-speech translation with discrete units. arXiv preprint arXiv:2212.08055 .
- Sathish Indurthi, Mohd Abbas Zaidi, Nikhil Kumar Lakumarapu, Beomseok Lee, Hyojung Han, Seokchan Ahn, Sangha Kim, Chanwoo Kim, and Inchul Hwang. 2021. Task aware multi-task learning for speech to text tasks. In ICASSP 2021
- -2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 7723-7727.
- Ye Jia, Melvin Johnson, Wolfgang Macherey, Ron J. Weiss, Yuan Cao, Chung-Cheng Chiu, Naveen Ari, Stella Laurenzo, and Yonghui Wu. 2019. Leveraging weakly supervised data to improve end-toend speech-to-text translation. In Proc. of ICASSP , pages 7180-7184.
- Eugene Kharitonov, Ann Lee, Adam Polyak, Yossi Adi, Jade Copet, Kushal Lakhotia, Tu Anh Nguyen, Morgane Riviere, Abdelrahman Mohamed, Emmanuel Dupoux, and Wei-Ning Hsu. 2022. Text-free prosody-aware generative spoken language modeling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 8666-8681, Dublin, Ireland. Association for Computational Linguistics.
- Diederik P. Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. In ICLR (Poster) .
- Philipp Koehn. 2004. Statistical significance tests for machine translation evaluation. In Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing , pages 388395, Barcelona, Spain. Association for Computational Linguistics.
- Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. 2020. Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis. In Advances in Neural Information Processing Systems , volume 33, pages 17022-17033. Curran Associates, Inc.
- Felix Kreuk, Adam Polyak, Jade Copet, Eugene Kharitonov, Tu Anh Nguyen, Morgane Rivière, Wei-Ning Hsu, Abdelrahman Mohamed, Emmanuel Dupoux, and Yossi Adi. 2021. Textless speech emotion conversion using decomposed and discrete representations. CoRR , abs/2111.07402.
- Kushal Lakhotia, Eugene Kharitonov, Wei-Ning Hsu, Yossi Adi, Adam Polyak, Benjamin Bolte, Tu-Anh Nguyen, Jade Copet, Alexei Baevski, Abdelrahman Mohamed, and Emmanuel Dupoux. 2021. On generative spoken language modeling from raw audio. Transactions of the Association for Computational Linguistics , 9:1336-1354.
- Tsz Kin Lam, Shigehiko Schamoni, and Stefan Riezler. 2022. Sample, translate, recombine: Leveraging audio alignments for data augmentation in end-to-end speech translation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) , pages 245254, Dublin, Ireland. Association for Computational Linguistics.
- Hang Le, Juan Pino, Changhan Wang, Jiatao Gu, Didier Schwab, and Laurent Besacier. 2020. Dualdecoder transformer for joint automatic speech
- recognition and multilingual speech translation. In Proceedings of the 28th International Conference on Computational Linguistics , pages 3520-3533, Barcelona, Spain (Online). International Committee on Computational Linguistics.
- Hang Le, Juan Pino, Changhan Wang, Jiatao Gu, Didier Schwab, and Laurent Besacier. 2021. Lightweight adapter tuning for multilingual speech translation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers) , pages 817-824, Online. Association for Computational Linguistics.
- Ann Lee, Peng-Jen Chen, Changhan Wang, Jiatao Gu, Sravya Popuri, Xutai Ma, Adam Polyak, Yossi Adi, Qing He, Yun Tang, Juan Pino, and Wei-Ning Hsu. 2022a. Direct speech-to-speech translation with discrete units. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 3327-3339, Dublin, Ireland. Association for Computational Linguistics.
- Ann Lee, Hongyu Gong, Paul-Ambroise Duquenne, Holger Schwenk, Peng-Jen Chen, Changhan Wang, Sravya Popuri, Yossi Adi, Juan Pino, Jiatao Gu, and Wei-Ning Hsu. 2022b. Textless speech-to-speech translation on real data. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pages 860-872, Seattle, United States. Association for Computational Linguistics.
- Xian Li, Changhan Wang, Yun Tang, Chau Tran, Yuqing Tang, Juan Pino, Alexei Baevski, Alexis Conneau, and Michael Auli. 2021. Multilingual speech translation from efficient finetuning of pretrained models. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) , pages 827-838, Online. Association for Computational Linguistics.
- Xinjian Li, Ye Jia, and Chung-Cheng Chiu. 2022. Textless direct speech-to-speech translation with discrete speech representation.
- Xuebo Liu, Longyue Wang, Derek F. Wong, Liang Ding, Lidia S. Chao, Shuming Shi, and Zhaopeng Tu. 2021. On the complementarity between pretraining and back-translation for neural machine translation. In Findings of the Association for Computational Linguistics: EMNLP 2021 , pages 29002907, Punta Cana, Dominican Republic. Association for Computational Linguistics.
- Gallil Maimon and Yossi Adi. 2022. Speaking style conversion with discrete self-supervised units. ArXiv , abs/2212.09730.
- Benjamin Marie, Raphael Rubino, and Atsushi Fujita. 2020. Tagged back-translation revisited: Why does
- it really work? In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , pages 5990-5997, Online. Association for Computational Linguistics.
- Tu Anh Nguyen, Eugene Kharitonov, Jade Copet, Yossi Adi, Wei-Ning Hsu, Ali Elkahky, Paden Tomasello, Robin Algayres, Benoit Sagot, Abdelrahman Mohamed, and Emmanuel Dupoux. 2022a. Generative spoken dialogue language modeling.
- Xuan-Phi Nguyen, Sravya Popuri, Changhan Wang, Yun Tang, Ilia Kulikov, and Hongyu Gong. 2022b. Improving speech-to-speech translation through unlabeled text. arXiv preprint arXiv:2210.14514 .
- Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. 2019. fairseq: A fast, extensible toolkit for sequence modeling. In Proceedings of NAACL-HLT 2019: Demonstrations .
- Kishore Papineni, Salim Roukos, Todd Ward, and WeiJing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics , pages 311-318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.
- Hieu Pham, Xinyi Wang, Yiming Yang, and Graham Neubig. 2021. Meta back-translation. In International Conference on Learning Representations .
- Adam Polyak, Yossi Adi, Jade Copet, Eugene Kharitonov, Kushal Lakhotia, Wei-Ning Hsu, Abdelrahman Mohamed, and Emmanuel Dupoux. 2021. Speech Resynthesis from Discrete Disentangled Self-Supervised Representations. In Proc. Interspeech 2021 .
- Sravya Popuri, Peng-Jen Chen, Changhan Wang, Juan Miguel Pino, Yossi Adi, Jiatao Gu, Wei-Ning Hsu, and Ann Lee. 2022. Enhanced direct speech-tospeech translation using self-supervised pre-training and data augmentation. In Interspeech .
- Matt Post. 2018. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers , pages 186191, Brussels, Belgium. Association for Computational Linguistics.
- Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2022. Robust speech recognition via large-scale weak supervision.
- Sahana Ramnath, Melvin Johnson, Abhirut Gupta, and Aravindan Raghuveer. 2021. HintedBT: Augmenting Back-Translation with quality and transliteration hints. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing , pages 1717-1733, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
- Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, and Tie-Yan Liu. 2021. Fastspeech 2: Fast and high-quality end-to-end text to speech. In International Conference on Learning Representations .
- Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Improving neural machine translation models with monolingual data. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 86-96, Berlin, Germany. Association for Computational Linguistics.
- Mihaela C Stoian, Sameer Bansal, and Sharon Goldwater. 2020. Analyzing asr pretraining for low-resource speech-to-text translation. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 7909-7913. IEEE.
- Yun Tang, Hongyu Gong, Ning Dong, Changhan Wang, Wei-Ning Hsu, Jiatao Gu, Alexei Baevski, Xian Li, Abdelrahman Mohamed, Michael Auli, and Juan Pino. 2022. Unified speech-text pre-training for speech translation and recognition. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 1488-1499, Dublin, Ireland. Association for Computational Linguistics.
- Yun Tang, Juan Pino, Xian Li, Changhan Wang, and Dmitriy Genzel. 2021a. Improving speech translation by understanding and learning from the auxiliary text translation task. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) , pages 4252-4261, Online. Association for Computational Linguistics.
- Yun Tang, Juan Pino, Changhan Wang, Xutai Ma, and Dmitriy Genzel. 2021b. A general multi-task learning framework to leverage text data for speech to text tasks. In ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) , pages 6209-6213.
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems , volume 30. Curran Associates, Inc.
- Li Wan, Quan Wang, Alan Papir, and Ignacio LopezMoreno. 2018. Generalized end-to-end loss for speaker verification. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2018, Calgary, AB, Canada, April 15-20, 2018 , pages 4879-4883. IEEE.
- Chengyi Wang, Yu Wu, Shujie Liu, Zhenglu Yang, and Ming Zhou. 2020a. Bridging the gap between pretraining and fine-tuning for end-to-end speech translation. In Proc. of AAAI , volume 34, pages 91619168.
- Chengyi Wang, Yu Wu, Shujie Liu, Ming Zhou, and Zhenglu Yang. 2020b. Curriculum pre-training for end-to-end speech translation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , pages 3728-3738, Online. Association for Computational Linguistics.
- Chen Xu, Bojie Hu, Yanyang Li, Yuhao Zhang, Shen Huang, Qi Ju, Tong Xiao, and Jingbo Zhu. 2021. Stacked acoustic-and-textual encoding: Integrating the pre-trained models into speech translation encoders. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) , pages 2619-2630, Online. Association for Computational Linguistics.
- Jiahao Xu, Yubin Ruan, Wei Bi, Guoping Huang, Shuming Shi, Lihui Chen, and Lemao Liu. 2022. On synthetic data for back translation. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pages 419430, Seattle, United States. Association for Computational Linguistics.
- Rong Ye, Mingxuan Wang, and Lei Li. 2021. End-toend speech translation via cross-modal progressive training. In Proc. of INTERSPEECH .
- Rong Ye, Mingxuan Wang, and Lei Li. 2022. Crossmodal contrastive learning for speech translation. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pages 5099-5113, Seattle, United States. Association for Computational Linguistics.
- Biao Zhang, Barry Haddow, and Rico Sennrich. 2022a. Revisiting end-to-end speech-to-text translation from scratch. In International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA , volume 162 of Proceedings of Machine Learning Research , pages 2619326205. PMLR.
- Ziqiang Zhang, Long Zhou, Junyi Ao, Shujie Liu, Lirong Dai, Jinyu Li, and Furu Wei. 2022b. Speechut: Bridging speech and text with hidden-unit for encoder-decoder based speech-text pre-training. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing , Online and Abu Dhabi.
- Yan Zhou, Qingkai Fang, and Yang Feng. 2023. CMOT: Cross-modal mixup via optimal transport for speech translation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics .

## A Details about the HiFi-GAN

The HiFi-GAN vocoder (Kong et al., 2020) consists of one generator and two discriminators. Next, we describe the model architecture and training objectives of HiFi-GAN.

Generator The generator is a convolutional neural network containing several stacked blocks. Each block comprises a transposed convolution layer followed by a multi-receptive field fusion (MRF) module. The transposed convolution layers upsample the input sequence to match the length of output waveforms. The MRF module includes multiple residual blocks with different receptive fields to model different patterns in parallel. The input to the generator is concatenated unit embedding Emb( z ) and speaker embedding e spkr .

Discriminator The discriminator consists of a Multi-Period Discriminator (MPD) and a MultiScale Discriminator (MSD). MPD is a mixture of sub-discriminators operating on equally spaced samples of the input audio. We adopt 5 subdiscriminators and set the space between samples to [2 , 3 , 5 , 7 , 11] respectively. MSD is also a mixture of sub-discriminators operating on different input scales: raw audio, × 2 downsampled audio, and × 4 downsampled audio. MPD and MSD can identify different periodic and consecutive patterns in the audio.

Training Objectives Using G to denote the generator and D j to denote each sub-discriminator, we define the adversarial loss of G and D j as:

<!-- formula-not-decoded -->

where x denotes the ground truth audio and ̂ x = G (Emb( z ) , e spkr ) denotes the synthetic audio.

Besides, there are two auxiliary training objectives. The first term measures the L1 distance between mel-spectrogram of the ground truth audio and synthetic audio:

<!-- formula-not-decoded -->

where φ is the function to compute melspectrogram of the audio. The second term is a feature-matching loss which measures the difference in discriminator features between ground truth audio and synthetic audio:

<!-- formula-not-decoded -->

where L denotes the number of layers in D j , δ i denotes the feature extractor of i -th layer, and N i denotes the number of features in i -th layer.

Considering all sub-discriminators, the final training objectives are as follows:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where J is the number of sub-discriminators. We set λ fm = 2 and λ mel = 45 .

## B Examples of Synthetic ST Data

To understand our approach more intuitively, we provide some examples of synthetic ST data in this section. Table 7, 8, and 9 show some examples of synthetic En → De, En → Fr, and En → Es ST data, respectively. For each sample, we give the original target text, generated reduced discrete units, generated source speech, and the corresponding transcript obtained with a state-of-the-art ASR model Whisper-Large 21 (Radford et al., 2022). We observe that our method can generate reasonable source speech, though it may contain minor errors or duplicates. This explains why our method can enhance ST successfully.

We also provide an example of our proposed Diverse BT4ST method in Table 10, which generates multiple diverse source speeches from only one target text. We observe that the four outputs of our model are all reasonable and differ from each other in some way, which confirms that Diverse BT4ST is a simple and effective method to generate diverse pseudo data.

We provide the corresponding audio files of the above samples at https://bt4st.github.io/ .

[21 https://github.com/openai/whisper](https://github.com/openai/whisper)

Table 7: Examples of synthetic En → De ST data.

<!-- image -->

Table 8: Examples of synthetic En → Fr ST data.

<!-- image -->

Table 9: Examples of synthetic En → Es ST data.

<!-- image -->

Table 10: Examples of synthetic En → De ST data generated by Diverse BT4ST method.

<!-- image -->