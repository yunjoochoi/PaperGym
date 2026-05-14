## BERTHop: An Effective Vision-and-Language Model for Chest X-ray Disease Diagnosis

Masoud Monajatipoor 1 , 3 * , Mozhdeh Rouhsedaghat 2 * , Liunian Harold Li 1 , Aichi Chien 4 , C.-C. Jay Kuo 2 , Fabien Scalzo 1 &amp;Kai-Wei Chang 1 1 Computer Science Department, UCLA 2 Department of Electrical Engineering, USC 3 Department of Electrical Engineering, UCLA 4 Department of Radiology, UCLA

monajati@ucla.edu,rouhseda@usc.edu,aichi@ucla.edu,jckou@usc.edu { liunian.harold.li,fab,kwchang } @cs.ucla.edu

## Abstract

Vision-and-language (V&amp;L) models take image and text as input and learn to capture the associations between them. Prior studies show that pre-trained V&amp;L models can significantly improve the model performance for downstream tasks such as Visual Question Answering (VQA). However, V&amp;L models are less effective when applied in the medical domain (e.g., on X-ray images and clinical notes) due to the domain gap. In this paper, we investigate the challenges of applying pre-trained V&amp;L models in medical applications. In particular, we identify that the visual representation in general V&amp;L models is not suitable for processing medical data. To overcome this limitation, we propose BERTHop, a transformer-based model based on PixelHop++ and VisualBERT, for better capturing the associations between the two modalities. Experiments on the OpenI dataset, a commonly used thoracic disease diagnosis benchmark, show that BERTHop achieves an average Area Under the Curve (AUC) of 98.12% which is 1.62% higher than state-of-theart (SOTA) while it is trained on a 9× smaller dataset.

## 1. Introduction

Computer-Aided Diagnosis (CADx) [14] systems could provide valuable benefits for disease diagnosis including but not limited to improving the quality and consistency of the predictions and reducing medical mistakes as they are not subject to human error. Although most existing studies focus on diagnosis based on medical images such as chest Xray (CXR) images [4, 2, 1], the radiology reports often contain substantial information (e.g. patient history and previ- ous studies) that are difficult to be detected from the image alone. Besides, diagnosis from both image and text is more closely aligned with disease diagnosis by human experts. Therefore, V&amp;L models that take both images and text as input can be potentially more accurate for CADx and several attempts have been made in this direction [40, 42, 23].

* equal contribution

Figure 1. An overview of BERTHop. BERTHop takes X-ray image and clinical report as input. It first encodes the image and text and extracts potential features from both modalities. Then a transformer-based model learns the associations between these two modalities. By applying appropriate vision and text extractor, the model is capable to identify the abnormality and associate it with the text labels.

<!-- image -->

However, the shortage of annotated data in the medical domain makes utilizing V&amp;L models challenging. Annotating medical data is an expensive process as it requires human experts. Although a couple of recent large-scale auto-labeled datasets have been provided for some medical tasks, e.g., chest X-ray [39, 6, 19], they are often noisy (low-quality) and degrade the performance of models. Besides, such datasets are not available for most medical tasks. Therefore, training V&amp;L models with limited annotated data remains a key challenge.

Recently, pre-trained V&amp;L models have been proposed for reducing the amount of labeled data required to train an accurate downstream model [22, 37, 36, 25] in the general domain. These models are first trained on large-scale image caption data with self-supervision signals (e.g., using masked language model loss) to learn the association between objects and text tokens. Then, the pre-trained V&amp;L models are used to initialize the downstream models and fine-tuned on the target tasks. In most V&amp;L tasks, it has been reported that V&amp;L pre-training is a major source of performance improvement. However, we identify a key problem in applying common pre-trained V&amp;L models for the medical domain: the large domain gap between the medical (target) and the general domain (source) makes such pre-train and fine-tune paradigm considerably less effective in the medical domain. Therefore, domain-specific designs have to be applied.

Notably, V&amp;L models mainly leverage object-centric feature extraction methods such as Faster R-CNN [31] which is pre-trained on general domain to detect everyday objects, e.g., cats, and dogs. However, the abnormalities in the X-ray images do not resemble everyday objects and will likely be ignored by a general-domain object detector.

To overcome this challenge, we propose BERTHop, a transformer-based V&amp;L model designed for medical applications. BERTHop resolves the domain gap issue by leveraging pre-training language encoder, BlueBERT [28], a BERT [12] variant that has been trained on biomedical and clinical datasets. Furthermore, in BERTHop, the visual encoder of the V&amp;L architecture is redesigned leveraging PixelHop++ [8] and is fully unsupervised which significantly reduces the need for labeled data [32]. PixelHop++ can extract image representations at different frequency levels that is beneficial for abnormality detection.

We evaluate BERTHop by conducting extensive experiments and analysis for CADx in chest disease diagnosis on the OpenI dataset [11]. The OpenI dataset contains thoracic diseases, including 14 common chest diseases. Compared to SOTA (TieNET [40]), BERTHop outperforms in 11 out of 14 thoracic diseases diagnoses and achieves an average AUC of 98.23% that is 1.73% higher, using significantly less training data (TieNet is trained on the ChestXray14 [39] dataset that is 9 times larger than OpenI). Compared to the similar transformer-based V&amp;L model pretrained on general domain and fine-tuned on OpenI [22, 23], BERTHop requires no expensive V&amp;L pre-training yet outperforms it by 14.37%.

We summarize our contributions as follows: (1) We propose BERTHop, a novel data-efficient V&amp;L model for CXR disease diagnosis surpassing existing approaches. (2) Our proposed model incorporates PixelHop++ into a transformer-based model. To the best of our knowledge, this is the first study which integrates PixelHop++ and Deep Neural Network (DNN) models. (3) We conduct extensive experiments to demonstrate the effectiveness of each submodel we used in BERTHop. (4) We study how transformer initialization with a model, pre-trained on in-domain data (even on a single modality) is highly beneficial in the medical domain.

## 2. Related Work

Transformer-based V&amp;L models Inspired by the success of BERT for NLP tasks, various transformer-based V&amp;L models have been proposed [22, 9, 37]. They generally use an object detector pre-trained on Visual Genome [20] to extract visual features from an input image and then use a transformer to model the visual features and input sentence. They are pre-trained on a massive amount of paired image-text data with a mask-and-predict objective similar to BERT. During pre-training, part of the input is masked and the objective is to predict the masked words or image regions based on the remaining contexts. Such models have been applied to many V&amp;L applications [43, 26, 10] including the medical domain [23]. However, the performance of these models is not satisfactory due to the domain shift between the general domain and medical domain.

V&amp;L models in the medical domain Various CNNRNN-based V&amp;L models have been proposed for disease diagnosis on CXR. Zhang et al. [42] proposed TNNT (Text-guided Neural Network Training) which helps a CNN model get guidance from text report embedding for a more efficient training process on V&amp;L data and evaluated the model on four V&amp;L datasets including the OpenI dataset. They showed that the text report has important information that can improve the diagnosis compared with prior visiononly models, e.g., ResNet.

TieNet is a CNN-RNN-based model for V&amp;L embedding integrating multi-level attention layers into an end-toend CNN-RNN framework for disease diagnosis and radiology report generation tasks. TieNet uses a ResNet-50 pretrained for general-domain visual feature extraction and an RNNfor V&amp;L fusion. As a result, it requires a large amount of in-domain training data (ChestX-ray14) for adapting to the medical domain, limiting its practical usage. In contrast, our method achieves higher performance with very limited in-domain data.

Recently, Li et al. [23] evaluated the transferability of well-known pre-trained V&amp;L models by fine-tuning them on MIMIC-CXR [19] and OpenI. However, the pre-trained models are designed and pre-trained for general-domain, and directly fine-tuning it with limited in-domain data leads to suboptimal performance. We refer to this method as VB w/ BUTD (section 4.2).

Figure 2. The proposed BERTHop framework for CXR disease diagnosis. A PixelHop++ model followed by a 'PCA and concatenation' block is used to generate Q feature vectors. These features along with language embedding are fed to the transformer that is initialized with BlueBERT.

<!-- image -->

PixelHop++ for visual feature learning PixelHop++ is originally proposed as an alternative to deep convolutional neural networks for feature extraction from images and video frames in resource-constrained environments. It is a multi-level model which generates output channels representing an image at different frequencies.

PixelHop++ is used in various applications and shown to be highly effective on small datasets. These applications include face gender classification [33], face recognition [34], and deep fake detection [7]. It has also been recently applied to a medical task. V oxelHop [24] leveraged this model on 3D Magnetic resonance imaging (MRI) imaging data and could achieve superior results for Amyotrophic Lateral Sclerosis (ALS) disease classification task.

To the best of our knowledge, this is the first study which integrates PixelHop++ and DNN models. Our proposed model takes advantage of the attention mechanism to integrate visual features extracted from PixelHop++ and the language embedding.

## 3. Approach

Inspired by the architecture of VisualBERT, our framework uses a single transformer to integrates visual features and language embeddings. The overall framework of our proposed approach is shown in Figure 2. We first utilize PixelHop++ to extract visual features from the X-ray image; then the text (a radiology report) is encoded into subword embeddings; a joint transformer is applied on top to model the relationship between two modalities and capture implicit alignments.

There are two main differences between BERTHop and previous approaches:

- Visual feature encoder Considering the lack of data in the medical domain, instead of using an object detector pre-trained on a general-domain dataset, we leverage PixelHop++, an unsupervised data-efficient method, to extract visual features. As the size of the PixelHop++ output channels is relatively large to be directly fed into the transformer, we apply Principle Component Analysis (PCA) to the output channels for dimension reduction. PCA is an orthogonal linear transformation that maps the data to a new coordinate system of lower dimension so that the variation of data is better preserved. By applying PCA to the PixelHop++ output channels, we capture the most prominent features and prevent over-fitting. Then, we concatenate the results to generate the final visual feature vectors. (Section 3.1)
- In-domain text pre-training Instead of resorting to computation-extensive V&amp;L pre-training on a general domain image-text dataset, we find in-domain textonly pre-training considerably more beneficial in our application. Thus, we use BlueBERT as the backbone for our model, a transformer pre-trained on biomedical and clinical datasets. (Section 3.2)

## 3.1. Visual encoder

We argue that extracting visual features from a generaldomain object detector, i.e. the BUTD [3] approach that is dominant in most V&amp;L tasks, is not suitable for the medical domain. BUTD 1 takes an image and employs a ResNet-based Faster-RCNN [31] for object detection and feature extraction from each object. The detector is pretrained on Visual Genome [20] to detect objects in everyday scenes. Such an approach fails to detect medical abnormalities when applied to X-ray images. The reason is that the abnormalities in the image, which are of high importance for facilitating diagnosis, usually do not resemble the normal notion of an 'object' and will likely be ignored by a general-domain object detector. Further, there exists no large-scale annotated dataset for disease abnormality detection from which to train a reliable detector [35].

1 In the following, we use the term 'BUTD' to refer to extracting visual features from a pre-trained object detector rather than the full model from [3].

Figure 3. Data flow in a 3-level PixelHop++ model. A node represents a channel.

<!-- image -->

We propose to adopt PixelHop++ [8] for unsupervised visual feature learning in the medical domain, which has been shown to be highly effective when trained on smallscale datasets. The key idea of PixelHop++ is computing the parameters of its model by a closed-form expression without using back-propagation [32]. As PixelHop++ leverages PCA for computing parameters, the model is able to extract image representations at various frequencies in an unsupervised manner. Inspired by the architecture of DNN models, PixelHop++ is a multi-level model in which each level consists of one or several PixelHop++ units followed by a max-pooling layer. An illustration of data flow in a 3level PixelHop++ model is shown in Figure 3. When training a PixelHop++ model, parameters of PixelHop++ units (kernels and biases) are computed, and during the inference, they are used for feature extraction from pixel blocks.

Training phase of PixelHop++ Suppose that we have N training images of size s 1 × s 2 × d , where d is 1 for grayscale and 3 for color images. They are all fed into a single PixelHop++ unit in the first level of the model. The goal of training a PixelHop++ unit is to compute linearly independent projection vectors (kernels) which can extract strong features from its input data. There are one or more PixelHop++ units in each level of a PixelHop++ model.

In the first step of processing data in a PixelHop++ unit, using a sliding window of size w × w × d and a stride of s , patches from each training image are extracted and flattened, i.e., x i 1 , x i 2 , ..., x iM where x ij is the j th flattened patch for image i , and M is the number of extracted patches per image.

In the second step, the set of all patches extracted from training images are used to compute the kernels of the PixelHop++ unit. Kernels are computed as follows:

- The first kernel, called DC kernel, is the mean filter, i.e., 1 √ n × (1 , 1 , ..., 1) where n is the size of the input vector, and extracts the mean of each input vector.
- After computing the mean (DC component) of each vector, PCA kernels of the residuals are computed and stored as AC kernels. First, k PCA kernels are the top k orthogonal projection vectors which can capture the variation of residuals best.

Each image patch is projected on computed kernels and a scalar bias is added to the projection result to avoid the sign-confusion problem [21]. This transformation on the input vector ( x 0 , x 1 , ..., x D -1 ) T can be shown as follows:

<!-- formula-not-decoded -->

where a kd represents kernel parameters associated with the k th kernel of a PixelHop++ unit and b k is the kernel's corresponding bias term.

By transforming x i 1 , x i 2 , ..., x iM by a kernel in a PixelHop++ unit, one output channel is generated. For example, in the first level of the model, the PixelHop++ unit generates 1 DC channel and w × w × d -1 AC channels. Each channel is shown by a node in Figure 3.

In the last step, model pruning is executed to remove the channels which include deficient data. The ratio of the variance explained by each kernel to the variance of training data is called the 'energy ratio' of the kernel or its corresponding channel and is used as a criterion for pruning the model. An energy ratio threshold value, E , is selected and model pruning is performed using the following rule:

- If the energy ratio of a channel is less than E , it will be discarded (discarded nodes/channels in Figure 3) as the variation of data along the corresponding kernel is very small.
- If the energy ratio of a channel is more than E , it is forwarded to the next level for further energy compaction (intermediate nodes/channels in Figure 3).

Each output intermediate channel generated by a PixelHop++ unit will be fed into one separate PixelHop++ unit in the next level. So, except for the first level of the model, other levels contain more than one PixelHop++ unit.

Inference phase of PixelHop++ Data flow is similar to the training phase but all parameters including kernel weights and biases are computed during the training phase. Therefore, according to Equation 1, feature extraction from test images is conducted in each PixelHop++ unit using the computed kernels and biases.

## 3.2. In-domain text pre-training

As shown in an example in Figure 4, the report is written by an expert radiologist, who lists the normal and abnormal observations in the 'finding' section and other important patient information e.g. patient history in the 'impression' section of the report. The text style of the report is drastically different from that of the pretraining corpora of BERT (Wikipedia and BookCorpus) or V&amp;L models (MSCOCO and Conceptual Captions).

However, previous methods [23] do not take such a significant domain gap into consideration. Rather, they initialize the transformer with a model trained on general-domain image-text corpora, as in most V&amp;L tasks. Meanwhile, pretraining with text-only corpora has been reported to how only marginal or no benefit [37]. In the medical domain, however, we find that using a transformer pre-trained on indomain text corpora as our initialized backbone serves as a simpler yet stronger approach.

Peng et al. [28] proposed a Biomedical Language Understanding Evaluation (BLUE) benchmark which evaluated the performance of BERT and Elmo [29] on 5 common biomedical text-mining tasks with ten corpora and showed the superiority of BERT when is pre-trained on biomedical and clinical datasets (BlueBERT). 2 . Recently, BlueBERT has been widely used in the bioNLP community for various NLP tasks [15, 13, 38] and a few V&amp;L tasks, e.g, data labeling [18]. Thus, we leverage this pre-trained version of BERT as the backbone in BERTHop and initialized its single-stream transformer [22] with BlueBERT to better capture the clinical report information.

## 4. Experiments

In this section, we evaluate BERTHop on the OpenI dataset and compare it with other existing models. To understand the effectiveness of the model designs, we also conduct detailed studies to verify the value of the visual encoder and the transformer initialization. Finally, we demonstrate a case study to show that BERTHop can effectively identify abnormal regions in CXR images.

## 4.1. Experiment setup

Dataset For CADx in CXR disease diagnosis, commonly used datasets include ChestX-ray14, MIMIC-CXR, and OpenI. In this paper, we focus on the OpenI dataset for

[2 https://github.com/ncbi-nlp/bluebert](https://github.com/ncbi-nlp/bluebert)

X-ray image

## Text report

Figure 4. A sample image-text pair in the OpenI dataset. The text report from a radiologist is important for disease diagnosis but has a significantly different style compared to general-domain text.

<!-- image -->

Figure 5. OpenI label statistics: (A) Percentage of normal and abnormal cases (B) Percentage of different diseases.

<!-- image -->

which professional annotators labeled the data. OpenI comprises 3,996 reports and 8,121 associated images from 3,996 unique patients collected by Indiana University from multiple institutes. Its labels include 14 commonly occurring thoracic chest diseases, i.e., Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening (PT), and Hernia. OpenI is a reliable choice for both training and evaluating V&amp;L models as it is annotated by experts (labels are not learned from text reports or images). The disadvantage of using OpenI for training is that it contains a small amount of training data which is a challenge for DNN models. We apply the same pre-processing as TieNet and obtain 3,684 image-text pairs.

We do not consider ChestX-ray14 and MIMIC-CXR for benchmarking because their labels are generated automatically from the images and/or associated reports. Specifically, ChestX-ray14 labels are mined using text process technique from the radiology reports, and MIMIC-CXR labels are generated using ChexPert[17] and NegBio[27] auto labelers. As their labels are machine-generated, evaluating the V&amp;L model on these datasets is not reliable. Therefore, we considered evaluation on OpenI to accurately compare the performance of BERTHop with human expert performance.

Model and training parameters We first resize all images of OpenI to 206 × 206 and apply the unsupervised feature learner, PixelHop++. We use a three-level PixelHop++ with the following hyper-parameters: w = 3 , d = 1 , s = 1 , and E = 0 . 00005 . Then, we apply PCA to its output channels and concatenate the generated vectors to form a set of Q visual features of dimension D, i.e., V = [ v 1 , v 2 , ..., v Q ] , v i ∈ R D . In BERTHop, D is set to be 2048. In our experiments setup, Q is equal to 15 but may vary depending on the size of the output channels of the PixelHop++ model and also the number of PCA components.

As for the transformer backbone, we use BlueBERTBase (Uncased, PubMed+MIMIC-III) from Huggingface [41], a transformer library. Having the visual features from the visual encoder and text embedding, we train the transformer on the training set of OpenI with 2,912 image-text pairs. We use batch size = 18, learning rate = 1 e -5 , maxseq-length = 128, and Stochastic Gradient Descent (SGD) as the optimizer with momentum = 0.9 and train it for 240 epochs.

Evaluation metric All mentioned datasets are highly imbalanced and mostly contain normal cases. Figure 5 shows the percentages of different diseases compared with normal cases in OpenI. Therefore, evaluating models using metrics such as accuracy does not reflect model performance. Instead, we follow prior studies to evaluate models based on Receiver Operating Characteristic (ROC) and Area Under the ROC Curve (AUC) score.

## 4.2. Main results

We train BERTHop on the OpenI training dataset containing 2,912 image-text pairs and evaluate it on the corresponding test set comprising 772 image-text pairs. The ROC curve for each disease is plotted in Figure 6.

We compare BERTHop with the following approaches:

- TNNT [42]: a Text-giuded Nueral Network Training method. See the details in Section 2.
- TieNET [40]: a CNN-RNN-based model. See the details in Section 2.
- VB w/ BUTD [22, 23]: Fine-tuning the original VisualBERT.

we evaluate all the models using the same AUC implementation in scikit-learn [5]. Table 1 summarizes the performance of BERTHop compared with existing methods.

The results demonstrate that BERTHop outperforms SOTA (TieNet) in 11 out of 14 thoracic disease diagnoses and achieves an average AUC of 98.23% which is 14.37%, 12.83%, and 1.73% higher than VB w/ BUTD, TNNT, and TieNet, respectively. Note that TieNet has been trained on a much larger annotated dataset, i.e., the ChestX-ray14

Figure 6. ROC curve of BERTHop for all 14 thoracic diseases.

<!-- image -->

dataset containing 108,948 training data while BERTHop is trained on only 2,912 case examples.

Regarding the VB w/ BUTD results, we re-evaluate the results based on the released code 3 from the original authors. However, we cannot reproduce the results reported in the paper even after contacting the authors.

## 4.3. In-domain text pre-training

We further investigate the influence of different transformer backbone initializations on model performance by pairing it with different visual encoders. The results are listed in Table 2.

First, we find that the proposed initialization with a model pre-trained on in-domain text corpora (BlueBERT) brings significant performance boosts when paired with PixelHop++. Initializing with BlueBERT gives a 6.46% performance increase compared to initializing with BERT.

Second, when using BUTD, the model is less sensitive to the transformer initialization and the performance is generally low (from 83.09% to 85.64%). In contrast to other V&amp;L tasks [22], general-domain V&amp;L pre-training is not instrumental.

The above findings suggest that for medical V&amp;L applications, in-domain single modality pre-training can bring larger performance improvement than using pre-trained V&amp;L models from the general domain, even though the latter is trained on a larger corpus. The relation and trade-off between single-modality pre-training and cross-modality pre-training are overlooked by previous works [22] and we advocate for future research on this.

## 4.4. Visual encoder

To better understanding what visual encoder is suitable for medical applications, we compare three visual feature extraction methods (BUTD, ChexNet [30], and Pix- elHop++). In particular, we replace the visual encoder of BERTHop with different visual encoders and report their performance. BUTD extracts visual features from a Faster R-CNN pre-trained on Visual Genome, which is prevailing in recent V&amp;L models. ChexNet is a CNN-based method that is proposed for pneumonia disease detection. It is a 121-layer DenseNet [16] trained on the ChestX-ray14 dataset for pneumonia detection having all pneumonia cases labeled as positive examples and all other cases as negative examples. By modifying the loss function, it is also trained to classify all 14 thoracic diseases and achieved state-ofthe-art among existing vision-only models, e.g., [39]. To augment the data, it extracts 10 crops from the image (4 corners and one center and horizontally flipped version of them) and feeds it into the network to generate a feature vector of dimension 1024 for each of them. In order to make it compatible with our transformer framework, we apply a linear transformation that maps feature vectors of size 1024, generated by ChexNet, to 2048. We fine-tune ChexNet and train the parameters of the linear transformation on the OpenI dataset.

[3 https://github.com/YIKUAN8/Transformers-VQA/ blob/master/openI\_VQA.ipynb](https://github.com/YIKUAN8/Transformers-VQA/blob/master/openI_VQA.ipynb)

Table 1. The AUC thoracic diseases diagnosis comparison of our model with other three methods on OpenI. BERTHop significantly outperforms models trained with a similar amount of data (e.g. VB w/ BUTD). *TieNet is trained on a much larger dataset than BERTHop.

|               | TNNT [42]   | TieNet ∗ [40]   | VB w/ BUTD [23]   | BERTHop   |
|---------------|-------------|-----------------|-------------------|-----------|
| Atelectasis   | -           | 0.976           | 0.9247            | 0.9838    |
| Cardiomegaly  | -           | 0.962           | 0.9665            | 0.9896    |
| Effusion      | -           | 0.977           | 0.9049            | 0.9432    |
| Infiltration  | -           | 0.984           | 0.8867            | 0.9926    |
| Mass          | -           | 0.903           | 0.6428            | 0.9900    |
| Nodule        | -           | 0.960           | 0.8480            | 0.9810    |
| Pneumonia     | -           | 0.994           | 0.8537            | 0.9967    |
| Pneumothorax  | -           | 0.960           | 0.8931            | 1.0000    |
| Consolidation | -           | 0.989           | 0.7870            | 0.9671    |
| Edema         | -           | 0.995           | 0.9500            | 0.9987    |
| Emphysema     | -           | 0.868           | 0.8565            | 0.9971    |
| Fibrosis      | -           | 0.960           | 0.6274            | 0.9966    |
| PT            | -           | 0.953           | 0.7612            | 0.9330    |
| Hernia        | -           | -               | -                 | -         |
| AVG           | 0.854       | 0.965           | 0.8386            | 0.9823    |

Table 2. Effect of the transformer backbones when paired with different visual encoders. We find that when using BUTD features, the model becomes insensitive to the transformer initialization and the expensive V&amp;L pre-training brings little benefit compared to BERT initialization. When using PixelHop++, the model benefits significantly from initialization with BlueBERT, which is pre-trained on indomain text corpora.

| Visual Encoder       | BUTD   | BUTD   | BUTD     | PixelHop++   | PixelHop++   |
|----------------------|--------|--------|----------|--------------|--------------|
| Transformer Backbone | VB     | BERT   | BlueBERT | BERT         | BlueBERT     |
| Atelectasis          | 0.9247 | 0.8677 | 0.8866   | 0.9890       | 0.9838       |
| Cardiomegaly         | 0.9665 | 0.8877 | 0.8875   | 0.9772       | 0.9896       |
| Effusion             | 0.9049 | 0.8940 | 0.9120   | 0.9013       | 0.9432       |
| Mass                 | 0.6428 | 0.7365 | 0.7373   | 0.8886       | 0.9900       |
| Consolidation        | 0.7870 | 0.8766 | 0.8906   | 0.8949       | 0.9671       |
| Emphysema            | 0.8565 | 0.7313 | 0.8261   | 0.9641       | 0.9971       |
| AVG                  | 0.8386 | 0.8309 | 0.8564   | 0.9177       | 0.9823       |

The results in Table 3 show that the visual encoder of BERTHop, PixelHop++, can extract richer features from the CXR images as it uses a data-efficient method capable of extracting image representations at different frequencies. Then, the transformer can highlight the most informative features from image-text data in an attention mechanism to make the final decision. In section 5, we explore the visual encoder of BERTHop and its effectiveness to capture abnormality regions.

Table 3. Comparison betwee different visual encoders (BUTD, ChexNet, and PixelHop++) under the same transformer backbone of BlueBERT. PixelHop++ outperforms BUTD and even ChexNet, which is pre-trained on a large in-domain disease diagnosis dataset.

|               |   BUTD |   ChexNet |   PixelHop++ |
|---------------|--------|-----------|--------------|
| Atelectasis   | 0.8866 |    0.9787 |       0.9838 |
| Cardiomegaly  | 0.8875 |    0.9797 |       0.9896 |
| Effusion      | 0.9120 |    0.8894 |       0.9432 |
| Mass          | 0.7373 |    0.7529 |       0.9900 |
| Consolidation | 0.8906 |    0.9000 |       0.9671 |
| Emphysema     | 0.8261 |    0.9067 |       0.9971 |
| AVG           | 0.8564 |    0.8798 |       0.9823 |

Figure 7. Avg AUC of three settings with different percentages of training data. BERTHop remains effective with different dataset scales.

<!-- image -->

## 5. Analysis

Effectiveness of BERTHop with different dataset scales To demonstrate the effectiveness of BERTHop on datasets of different scales and justify our designs, we conduct an experiment to compare BERTHop with its two variants: (1) In PH BERT, we replace BlueBERT with BERT. We compare BERTHop with PH BERT to show how a domain-specific BERT model helps to improve the performance in medical applications. (2) In BUTD BlueBERT, we replace the visual encoder PixelHop++ with the general visual encoder of BUTD.

We randomly select fractions of the training set of OpenI to train these three models and compare their performance on the entire test set of OpenI. Figure 7 illustrates that the performance of BERTHop is consistently better than the other two settings.

Visualize abnormal regions identified by BERTHop We visualize PixelHop++ output channels of BERTHop to probe whether it can effectively capture abnormal regions in CXR images. In this study, we asked two radiologists to annotate pathology regions of a few examples related to different diseases. As shown in Figure 8, some output channels can successfully highlight the abnormalities in CXR images. This is due to the fact that PixelHop++ extracts image representations at different frequencies which is beneficial for abnormality detection.

Figure 8. On the top, we mark the pathology regions annotated by two radiologists (the yellow circles and lines); on the bottom, we visualize the visual features from BERTHop (brighter colors means higher feature values). BERTHop can successfully highlight the abnormal regions identified by expert radiologists.

<!-- image -->

## 6. Conclusion and future work

In this paper, we proposed a high-performance dataefficient V&amp;L model, BERTHop, for CXR disease diagnosis. We showed that BERTHop outperforms state-of-theart while it is trained on a much smaller training set. Our studies verify the effectiveness of the visual feature extractor PixelHop++ and the transformer backbone initialization BlueBERT.

For future research direction, we plan to study how anomaly detection techniques can be incorporated to further improve the performance of the model. As no largescale annotated CXR dataset for anomaly detection is available, we may use weekly supervised techniques or knowledge transfer from similar tasks. We are also interested in how our proposed BERTHop model can help other biomedical tasks, e.g., COVID-19 disease diagnosis and radiology report generation.

## References

- [1] Rahib H Abiyev and Mohammad Khaleel Sallam Ma'aitah. Deep convolutional neural networks for chest diseases detection. Journal of healthcare engineering , 2018, 2018.
- [2] Imane Allaouzi and Mohamed Ben Ahmed. A novel approach for multi-label chest x-ray classification of common thorax diseases. IEEE Access , 7:64279-64288, 2019.
- [3] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 6077-6086, 2018.
- [4] Enes Ayan and Halil Murat ¨ Unver. Diagnosis of pneumonia from chest x-ray images using deep learning. In 2019 Scientific Meeting on Electrical-Electronics &amp; Biomedical Engineering and Computer Science (EBBT) , pages 1-5. Ieee, 2019.
- [5] Lars Buitinck, Gilles Louppe, Mathieu Blondel, Fabian Pedregosa, Andreas Mueller, Olivier Grisel, Vlad Niculae, Peter Prettenhofer, Alexandre Gramfort, Jaques Grobler, Robert Layton, Jake VanderPlas, Arnaud Joly, Brian Holt, and Ga¨ el Varoquaux. API design for machine learning software: experiences from the scikit-learn project. In ECML PKDD Workshop: Languages for Data Mining and Machine Learning , pages 108-122, 2013.
- [6] Aurelia Bustos, Antonio Pertusa, Jose-Maria Salinas, and Maria de la Iglesia-Vay´ a. Padchest: A large chest x-ray image dataset with multi-label annotated reports. Medical image analysis , 66:101797, 2020.
- [7] Hong-Shuo Chen, Mozhdeh Rouhsedaghat, Hamza Ghani, Shuowen Hu, Suya You, and C. C. Jay Kuo. Defakehop: A light-weight high-performance deepfake detector, 2021.
- [8] Yueru Chen, Mozhdeh Rouhsedaghat, Suya You, Raghuveer Rao, and C-C Jay Kuo. Pixelhop++: A small successivesubspace-learning-based (ssl-based) model for image classification. In 2020 IEEE International Conference on Image Processing (ICIP) , pages 3294-3298. IEEE, 2020.
- [9] Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter: Universal image-text representation learning. In European Conference on Computer Vision , pages 104-120. Springer, 2020.
- [10] Shih-Han Chou, Wei-Lun Chao, Wei-Sheng Lai, Min Sun, and Ming-Hsuan Yang. Visual question answering on 360deg images. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision , pages 1607-1616, 2020.
- [11] Dina Demner-Fushman, Marc D Kohli, Marc B Rosenman, Sonya E Shooshan, Laritza Rodriguez, Sameer Antani, George R Thoma, and Clement J McDonald. Preparing a collection of radiology examinations for distribution and retrieval. Journal of the American Medical Informatics Association , 23(2):304-310, 2016.
- [12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 , 2018.
- [13] Kathleen C Fraser, Isar Nejadgholi, Berry De Bruijn, Muqun Li, Astha LaPlante, and Khaldoun Zine El Abidine. Extracting umls concepts from medical text using general
14. and domain-specific deep learning models. arXiv preprint arXiv:1910.01274 , 2019.
- [14] Maryellen L Giger and Kenji Suzuki. Computer-aided diagnosis. In Biomedical information technology , pages 359XXII. Elsevier, 2008.
- [15] Yu Gu, Robert Tinn, Hao Cheng, Michael Lucas, Naoto Usuyama, Xiaodong Liu, Tristan Naumann, Jianfeng Gao, and Hoifung Poon. Domain-specific language model pretraining for biomedical natural language processing. arXiv preprint arXiv:2007.15779 , 2020.
- [16] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 4700-4708, 2017.
- [17] Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, et al. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 33, pages 590-597, 2019.
- [18] Saahil Jain, Akshay Smit, Steven QH Truong, Chanh DT Nguyen, Minh-Thanh Huynh, Mudit Jain, Victoria A Young, Andrew Y Ng, Matthew P Lungren, and Pranav Rajpurkar. Visualchexbert: Addressing the discrepancy between radiology report labels and image labels. arXiv preprint arXiv:2102.11467 , 2021.
- [19] Alistair EW Johnson, Tom J Pollard, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Yifan Peng, Zhiyong Lu, Roger G Mark, Seth J Berkowitz, and Steven Horng. Mimic-cxr-jpg, a large publicly available database of labeled chest radiographs. arXiv preprint arXiv:1901.07042 , 2019.
- [20] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision , 123(1):32-73, 2017.
- [21] C-C Jay Kuo, Min Zhang, Siyang Li, Jiali Duan, and Yueru Chen. Interpretable convolutional neural networks via feedforward design. Journal of Visual Communication and Image Representation , 60:346-359, 2019.
- [22] Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang. Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557 , 2019.
- [23] Yikuan Li, Hanyin Wang, and Yuan Luo. A comparison of pre-trained vision-and-language models for multimodal representation learning across medical images and reports. In 2020 IEEE International Conference on Bioinformatics and Biomedicine (BIBM) , pages 1999-2004. IEEE, 2020.
- [24] Xiaofeng Liu, Fangxu Xing, Chao Yang, C-C Jay Kuo, Suma Babu, Georges El Fakhri, Thomas Jenkins, and Jonghye Woo. Voxelhop: Successive subspace learning for als disease classification using structural mri. arXiv preprint arXiv:2101.05131 , 2021.
- [25] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic represen-
27. tations for vision-and-language tasks. arXiv preprint arXiv:1908.02265 , 2019.
- [26] Jiasen Lu, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, and Stefan Lee. 12-in-1: Multi-task vision and language representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 10437-10446, 2020.
- [27] Yifan Peng, Xiaosong Wang, Le Lu, Mohammadhadi Bagheri, Ronald Summers, and Zhiyong Lu. Negbio: a highperformance tool for negation and uncertainty detection in radiology reports. AMIA Summits on Translational Science Proceedings , 2018:188, 2018.
- [28] Yifan Peng, Shankai Yan, and Zhiyong Lu. Transfer learning in biomedical natural language processing: an evaluation of bert and elmo on ten benchmarking datasets. arXiv preprint arXiv:1906.05474 , 2019.
- [29] Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. Deep contextualized word representations. arXiv preprint arXiv:1802.05365 , 2018.
- [30] Pranav Rajpurkar, Jeremy Irvin, Kaylie Zhu, Brandon Yang, Hershel Mehta, Tony Duan, Daisy Ding, Aarti Bagul, Curtis Langlotz, Katie Shpanskaya, et al. Chexnet: Radiologistlevel pneumonia detection on chest x-rays with deep learning. arXiv preprint arXiv:1711.05225 , 2017.
- [31] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: towards real-time object detection with region proposal networks. IEEE transactions on pattern analysis and machine intelligence , 39(6):1137-1149, 2016.
- [32] Mozhdeh Rouhsedaghat, Masoud Monajatipoor, Zohreh Azizi, and C-C Jay Kuo. Successive subspace learning: An overview. arXiv preprint arXiv:2103.00121 , 2021.
- [33] Mozhdeh Rouhsedaghat, Yifan Wang, Xiou Ge, Shuowen Hu, Suya You, and C-C Jay Kuo. Facehop: A light-weight low-resolution face gender classification method. arXiv preprint arXiv:2007.09510 , 2020.
- [34] Mozhdeh Rouhsedaghat, Yifan Wang, Shuowen Hu, Suya You, and C-C Jay Kuo. Low-resolution face recognition in resource-constrained environments. arXiv preprint arXiv:2011.11674 , 2020.
- [35] Hoo-Chang Shin, Holger R Roth, Mingchen Gao, Le Lu, Ziyue Xu, Isabella Nogues, Jianhua Yao, Daniel Mollura, and Ronald M Summers. Deep convolutional neural networks for computer-aided detection: Cnn architectures, dataset characteristics and transfer learning. IEEE transactions on medical imaging , 35(5):1285-1298, 2016.
- [36] Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. Vl-bert: Pre-training of generic visuallinguistic representations. arXiv preprint arXiv:1908.08530 , 2019.
- [37] Hao Tan and Mohit Bansal. Lxmert: Learning crossmodality encoder representations from transformers. arXiv preprint arXiv:1908.07490 , 2019.
- [38] Shoya Wada, Toshihiro Takeda, Shiro Manabe, Shozo Konishi, Jun Kamohara, and Yasushi Matsumura. A pre-training technique to localize medical bert and enhance biobert. arXiv preprint arXiv:2005.07202 , 2020.
- [39] Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M Summers. Chestxray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 2097-2106, 2017.
- [40] Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, and Ronald M Summers. Tienet: Text-image embedding network for common thorax disease classification and reporting in chest x-rays. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 9049-9058, 2018.
- [41] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R´ emi Louf, Morgan Funtowicz, et al. Huggingface's transformers: State-of-the-art natural language processing. arXiv preprint arXiv:1910.03771 , 2019.
- [42] Zizhao Zhang, Pingjun Chen, Xiaoshuang Shi, and Lin Yang. Text-guided neural network training for image recognition in natural scenes and medicine. IEEE transactions on pattern analysis and machine intelligence , 2019.
- [43] Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason Corso, and Jianfeng Gao. Unified vision-language pretraining for image captioning and vqa. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 34, pages 13041-13049, 2020.