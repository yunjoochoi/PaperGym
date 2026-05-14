## AudioGenX: Explainability on Text-to-Audio Generative Models

Hyunju Kang 1 * , Geonhee Han 1 * , Yoonjae Jeong 2 , Hogun Park 1†

1 Department of Artificial Intelligence, Sungkyunkwan University, Suwon, Republic of Korea

2 Audio AI Lab, NCSOFT, Seongnam, Republic of Korea { neutor, gunhee8178 } @skku.edu, hybris75@gmail.com, hogunpark@skku.edu

## Abstract

Text-to-audio generation models (TAG) have achieved significant advances in generating audio conditioned on text descriptions. However, a critical challenge lies in the lack of transparency regarding how each textual input impacts the generated audio. To address this issue, we introduce AudioGenX, an Explainable AI (XAI) method that provides explanations for text-to-audio generation models by highlighting the importance of input tokens. AudioGenX optimizes an Explainer by leveraging factual and counterfactual objective functions to provide faithful explanations at the audio token level. This method offers a detailed and comprehensive understanding of the relationship between text inputs and audio outputs, enhancing both the explainability and trustworthiness of TAG models. Extensive experiments demonstrate the effectiveness of AudioGenX in producing faithful explanations, benchmarked against existing methods using novel evaluation metrics specifically designed for audio generation tasks.

## Introduction

Text-to-audio generation models (TAG) (Kreuk et al. 2023; Ziv et al. 2024; Yang et al. 2023; Liu et al. 2023; Schneider et al. 2023) have emerged as a pivotal technology in generative AI, enabling textual content to be transformed into an auditory experience. Although models such as AudioGen (Kreuk et al. 2023) excel at generating high-quality audio based on textual prompts, a critical challenge remains: the lack of transparency in how each textual input affects the generated audio. Consequently, users may struggle to trust the model, making it essential to provide explanations for the TAG task. Explainability provides several key advantages. First, it enhances awareness of how input tokens affect the model's outputs, enabling users to ensure that the model emphasizes the correct aspects of the text. Second, it provides actionable insights to support the decision-making about which elements to modify and to what extent in the audio editing process. Third, analyzing generated explanations can aid with debugging and identifying potential biases. Accordingly, this study argues that the ability to quantify the importance of textual inputs in TAG models is crucial to being able to unambiguously assess and communicate their value.

* Equal contribution.

† Corresponding author.

Copyright © 2025, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

Figure 1: A comprehensive explanation provided by AudioGenX for the entire audio in (a). Granular explanations for the interval from 1 to 1.5 seconds in (b) and from 2.5 to 3 seconds in (c), respectively.

<!-- image -->

While approaches specifically tailored for explaining TAG models are limited, recent research has explored methodologies for calculating the importance of input tokens in large-scale transformer-based models. Crossattention layers in multi-modal architectures, such as those in TAG models, are widely regarded as critical for integrating textual and auditory information, while also enhancing explainability by revealing how information from one modality influences another. A notable method (Abnar and Zuidema 2020) utilizes attention weights and aggregates them across all layers to approximate the importance of each input token. However, attention scores alone are not considered reliable for causal insights, as they do not directly indicate how perturbation to specific inputs influences the output. Recently, AtMan (Deiseroth et al. 2023) introduced a perturbation method that suppresses the attention score of one token at a time to observe the impact of each input on output prediction. This single-token perturbation approach, however, may overlook interactions between multiple tokens. Consequently, it provides less reliable explanations in scenarios where the model heavily relies on the contextual relationships between multiple tokens, leading to an oversimplification of the model's behavior.

To address the challenge of faithful explanations, causal inference theory, encompassing factual and counterfactual reasoning, is often utilized (Pearl 2009). These two approaches aim to identify impactful input information in different ways. Factual reasoning focuses on identifying critical input information that reproduces the original prediction, whereas counterfactual reasoning (Tan et al. 2022; Ali et al. 2023; Kenny et al. 2021) seeks to determine crucial input information that, if absent, would change the prediction. Given their differing assumptions, these reasoning approaches can be employed together as complementary frameworks to generate more faithful explanations. However, prior research has yet to investigate the feasibility of applying factual and counterfactual reasoning within TAG models.

To provide faithful explanations for the TAG model, we introduce AudioGenX, a perturbation-based explainability method leveraging factual and counterfactual reasoning. Our approach utilizes the latent representation vectors in TAG models to observe the effects of factual and counterfactual perturbations. These perturbations are applied in the crossattention layer using a soft mask, enabling the simultaneous perturbation of multiple tokens' attention scores. More importantly, the mask itself serves as an explanation, with its values quantifying the importance of the textual input. We optimize the mask through a gradient descent method guided by our proposed factual and counterfactual objective functions. To mitigate the high computational cost of calculating gradients for the entire sequential audio, we enhance efficiency by decomposing the explanation target into individual audio tokens. This approach enables us to customize the explanation range of generated audio interactively, providing comprehensive explanations for the entire audio or more granular explanations for specific segments of interest, depending on user demand. For instance, in Figure 1, (a) provides a comprehensive explanation for the entire audio, indicating a strong relation to vehicle motion. By focusing on a specific interval in (b) and (c), AudioGenX captures the different contexts of each audio segment and delivers contextually accurate explanations accordingly. Extensive experiments demonstrate the faithfulness of our explanations and benchmark their performance against recent techniques using proposed evaluation metrics for audio generation tasks.

Contributions. We summarize our contributions as follows: 1) We propose a faithful explanation method for textto-audio generation models, grounded in factual and counterfactual reasoning to quantify the importance of text tokens to the generated audio. 2) We offer a framework that provides both holistic and granular audio explanations based on user requests, enabling tailored insights. 3) We introduce new evaluation metrics for text-to-audio explanations and demonstrate the effectiveness of AudioGenXthrough extensive experiments compared to existing methods. 4) We present case studies demonstrating how AudioGenX provides valuable insights to support the understanding of model behavior and editing tasks.

## Related Work

Text-to-Audio Generation Models. Recent text-to-audio generation models can be categorized into two model architectures: Transformer-based (Kreuk et al. 2023; Ziv et al. 2024) and Diffusion-based (Yang et al. 2023; Liu et al. 2023;

Schneider et al. 2023). Transformer models, such as AudioGen (Kreuk et al. 2023), employ autoregressive Transformers to predict discrete audio tokens, while MAGNeT (Ziv et al. 2024) enhances efficiency through masked generative modeling in a non-autoregressive scheme. Diffusionbased approaches such as Diffsound (Yang et al. 2023) generate discrete mel-spectrogram tokens, whereas models like AudioLDM (Liu et al. 2023) and Moˆ usai (Schneider et al. 2023) directly predict continuous mel-spectrograms or waveforms. Despite architectural differences, these models commonly use cross-attention mechanisms, making AudioGenX a model-agnostic explainer for TAG models that use cross-attention in audio generation.

Explainable AI. Explainability involves methods that help to understand the importance of each input token with respect to output predictions. These methods generally fall into two categories: gradient-based methods (Selvaraju et al. 2017; Sundararajan, Taly, and Yan 2017; Nagahisarchoghaei et al. 2023) and perturbation-based methods (Ribeiro, Singh, and Guestrin 2016; Lundberg and Lee 2017). Gradientbased explanation methods trace gradients from the target layers to the predictive value, using the calculated gradients as a measure of importance. While effective, these methods require substantial memory resources to store the values of each targeted layer. In contrast, perturbation-based methods, such as SHAP (Lundberg and Lee 2017), are more memoryefficient, calculating feature importance by comparing predictions with and without specific features. Similarly, our method adopts a perturbation-based approach to effectively generate explanations.

Explainability on Audio Processing Models. Existing explainability approaches (Akman and Schuller 2024) on audio processing models have extended generic explanation methods. For instance, one study (Becker et al. 2018) employs Layer-wise Relevance Propagation (LRP) to explain the model trained on raw waveforms and spectrograms for spoken digit and speaker gender classification. Another study applied DFT-LRP (Frommholz et al. 2023) to audio event detection, assessing the significance of time-frequency components and guiding input representation choices. Similarly, audioLIME (Haunschmid, Manilow, and Widmer 2020) extends LIME (Ribeiro, Singh, and Guestrin 2016) to explain music-tagging models by perturbing audio components derived from source separation. However, since the above methods focus on explaining audio continuously and sequentially, they are not directly applicable to the unique challenges posed by TAG models, which require techniques that address the complex interactions between text inputs and generated audio outputs.

Explainability on Transformer. With the widespread use of Transformers, the demand for explainability has grown. Primarily, Rollout (Abnar and Zuidema 2020) primarily aggregates attention weights in all layers to track information flow but struggles to integrate cross-attention weights in multi-modal models with differing domain dimensionalities. Another recent work (Chefer, Gur, and Wolf 2021) leverages Layer-wise Relevance Propagation (LRP) (Samek, Wiegand, and M¨ uller 2017) to calculate class-specific relevance scores based on gradients of attention weights in self- and cross-attention layers. Nevertheless, AtMan (Deiseroth et al. 2023) raises the issue of excessive memory usage and introduces a scalable explanation method that employs single-token perturbation to observe the change of loss in the response. While intuitive and memory-efficient for largescale models, this method is limited in its ability to account for the interrelationship of input tokens.

## Preliminaries

AudioGen (Kreuk et al. 2023), a representative TAG model, consists of three key components: a text encoder (Raffel et al. 2020), an autoregressive Transformer decoder model (Vaswani et al. 2017), and an audio decoder (D´ efossez et al. 2023). The Transformer decoder serves as the core model responsible for generating the audio sequence, while the text encoder processes the input text and the audio decoder post-processes the generated audio token sequence into audio. Given a text prompt, it is converted into a tokenized representation vector, denoted as U = [ u 1 , . . . , u L ] , U ∈ R L × d u , where L denotes number of textual tokens and d u represents a dimension of the textual token representation vector. The generated audio can be expressed in a discrete form, as EnCodec (D´ efossez et al. 2023) converts the audio into either discrete tokens or continuous token representations. The tokenized audio sequence is denoted as Z = [ z 1 , . . . , z T ] , Z ∈ N T × d v , where T denotes the length of the audio sequence and d v indicates the number of codebooks d v . In detail, the codebook is a structured set of discrete audio tokens used in multi-stream audio generation to produce high-quality audio. For more comprehensive information on multi-streaming audio generation, we refer to the original AudioGen paper (Kreuk et al. 2023).

For the generation of an audio sequence, the Transformerdecoder model (Vaswani et al. 2017), denoted as h , generates z t as t -th order audio token in the sequence, following the formulation h ( U , z t -1 ) = z t . For brevity, we omit the detailed notation of other components and the topp or topk sampling process in the Transformer. Instead, we focus on the attention layers, including cross-attention, which are crucial components of the model, denoted as f . The computation within these layers is expressed in a simplified version as f ( U , z t -1 ) = e t , where e t represents the latent representation vector corresponding to the t -th audio token. In the absence of ground truth and class labels, the latent embedding vector e t in the audio token space provides information on how perturbation impacts subsequent generations. Particularly, the cross-attention layer is essential to fuse the textual information with auditory information in layers f , we denote the cross-attention layers as:

<!-- formula-not-decoded -->

where σ indicates a softmax function, Q , K , V , d k refers to query, key, values, and the number of vector dimensions in the k -th layer, respectively. In detail, Q refers to previously generated audio tokens, representing the query information, while K and V correspond to the textual tokens.

## The Proposed AudioGenX

AudioGenX addresses the challenge of explaining TAG models, where the goal is to quantify the importance of textual input corresponding to the generated audio. To achieve this within a sequence-to-sequence framework, we decompose the explanation target, represented as sequential audio, into individually non-sequential audio tokens. Since the output is sequential data, calculating gradients across the entire sequence, from the first to the last token, is computationally expensive and time-consuming. To overcome these issues, we redefine the explanation target as individual audio tokens, rather than the entire sequence. This modification enables parallel computation of generating an explanation for each token, significantly speeding up the process. Finally, AudioGenX integrates these individual token-level explanations to provide a comprehensive understanding of the entire audio sequence. An overview of AudioGenX is illustrated in Figure 2.

## Definition of Masks as Explanations

We quantify the importance of the t -th audio token z t within the audio sequence using a mask as the explanation. The soft mask is denoted as MU , z t ∈ R L × 1 , where each element mu i , z t ∈ MU , z t represents the importance of the i -th textual token with respect to the t -th audio token z t . Each value lies in the range [0 , 1] , where a value close to 1 indicates that the corresponding textual token is highly important for generating the target audio token, while a value closer to 0 indicates lower importance. To serve as a soft mask representing the importance of each text token, AudioGenX optimizes the Explainer to predict the mask MU , z t as the explanation. The Explainer consists of Multi-Layer Perceptrons (MLPs) with a sigmoid and gumbel-softmax (Jang et al. 2017) function to constrain values within the range [0 , 1] without additional scaling and to enforce the values close to either 0 or 1 , thereby highlighting relatively distinguished contribution. Using the soft mask, we apply perturbation to modify the inner computational steps of the cross-attention layers, altering the attention score of the given textual input. Consequently, we measure the perturbation effect on the prediction at the layer f ( U , z t -1 ) = e t , observing how latent representation vector e t for the audio token z t changes under these perturbations. In the following section, we detail how we optimize Explainer to predict the mask as explanations based on both factual and counterfactual reasoning.

## Formulating Factual Explanations

Factual reasoning (Tan et al. 2022; Ali et al. 2023; Kenny et al. 2021) aims to find sufficient input that can approximately reproduce the original prediction. To quantify the sufficiency of textual tokens, we employ a perturbationbased method using the soft mask, interleaving the computation to measure the impact of changes. Specifically, we mask out attention scores in the cross-attention layers where textual information is fed into the TAG model. We formulate the perturbation in factual reasoning as:

<!-- formula-not-decoded -->

Figure 2: (a) The process by which AudioGen generates an audio. (b) AudioGenX's procedure for generating and applying explanations, with the Explainer in the green box. (c) The method for calculating and applying the loss in AudioGenX.

<!-- image -->

where σ denotes the softmax activation function and the mask MU , z t controls the amount of information corresponding to the text token. When the mask value mu i , z t approaches 0 , the attention score is suppressed, meaning the information corresponding to the textual token is not fully propagated to the subsequent layer. Conversely, as the mask value approaches 1 , the original value is fully preserved. To distinguish this process from the original layer f ( U , z t -1 ) , we denote the layer applying perturbation with the factual mask as ˜ f ( U , z t -1 , MU , z t ) .

When the mask sufficiently serves as a factual explanation, the perturbed output remains approximately the same as the original prediction. To evaluate the impact of perturbation, we measure the resulting changes in the latent representation vector within the audio token space. Since the latent vector encodes rich and implicit information, we expect that two vectors close to each other indicate a similar auditory meaning, which is likely to result in similar audio generation. By leveraging this vector similarity, we can effectively measure the influence of perturbation and formulate the objective function for Explainer as:

<!-- formula-not-decoded -->

where cos function refers to cosine similarity, which measures how similar factual result ˜ f ( U , z t -1 , MU , z t ) is to the original prediction f ( U , z t -1 ) in the audio token space. Since the objective function involves negative cosine similarity, minimizing the loss function corresponds to maximizing the similarity. Hence, following the objective function, the Explainer generates the factual explanation mask, ensuring that the two representations or predictions are as close as possible in the audio token space.

## Formulating Counterfactual Explanations

Counterfactual reasoning (Tan et al. 2022; Ali et al. 2023; Kenny et al. 2021) aims to identify necessary inputs that can significantly alter the original prediction when it is perturbed or removed. This perturbation operates in the opposite direction of factual explanations, removing the important input to observe the counterfactual result. We formulate the pertur- bation method in counterfactual reasoning as:

<!-- formula-not-decoded -->

where 1 ∈ R 1 × T u is a vector of ones and 1 -MU , z t subtracts the importance of the corresponding textual tokens. Consequently, the more important a textual token is, the more its attention score is suppressed in proportion to its importance. This perturbation operates under a counterfactual assumption as the What-If scenario (Tan et al. 2022; Ali et al. 2023; Kenny et al. 2021): What happens if the important textual token does not exist? After applying the perturbation in Equation (4), the counterfactual result is observed as ˜ f ( U , z t -1 , 1 -MU , z t ) . If the counterfactual result significantly differs from the original prediction, it indicates that the counterfactual mask is necessary to explain the original prediction. Conversely, if the change is trivial, the counterfactual mask is unnecessary to explain the causal relationship with the prediction.

Generally, counterfactual explanations in supervised settings aim to find the important inputs that change the prediction with minimal perturbation. However, no class labels or guidance are available in our task of audio generation. Instead, we measure the change of meaning in latent space leveraging the cosine similarity function after counterfactual perturbation. Thus, the counterfactual explanation objective function is formulated as:

<!-- formula-not-decoded -->

where cos function measures how dissimilar counterfactual result ˜ f ( U , z t -1 , 1 -MU , z t ) is to the original prediction in latent space. As the cosine similarity decreases, the objective function minimizes the similarity. Consequently, the Explainer generates the counterfactual explanation mask to ensure that the two representations or predictions are as far as possible in the audio token space after counterfactual perturbation.

## Objective Function for AudioGenX

Along with factual and counterfactual explanation objective functions, we add the regularization term to generate the explanation mask in a simple and efficient manner. Therefore, Input : Textual token representation vector U , previously generated audio token vector z t -1 , Transformer model f , audio generation length T , number of epochs K , learning rate λ , regularization coefficients α and β

<!-- formula-not-decoded -->

we incorporate additional regularization in our final objective function for the Explainer , which is formulated as:

<!-- formula-not-decoded -->

Here, L 1 and L 2 represent the L 1 -Norm and L 2 -Norm, respectively, as regularization terms to minimize the mask size. This prevents a trivial solution where the Explainer generates an explanation mask assigning equal importance to all values. At the same time, adhering to Occam's Razor principle, we favor simpler and more effective explanations (Tan et al. 2022; Blumer et al. 1987). Hence, according to the objective function in Equation (6), AudioGenX optimizes the Explainer generating faithful explanation masks in the audio-token level.

## Providing Audio-Level Explanations

In this section, we aggregate audio token-level explanations to provide a comprehensive understanding of the entire audio sequence. The aggregation is performed by averaging the mask values across all audio tokens as follows:

<!-- formula-not-decoded -->

where t refers to the step, and T represents the total length of generated audio. Additionally, it is possible to focus on a specific interval of interest within the audio, defined between a starting step s and an ending step n . This is denoted as MU , z = 1 | n -s | +1 ∑ n t = s MU , z t , which provides granular explanations based on the user's request. This flexible approach enables users to discover patterns within specific intervals, as AudioGenX can effectively capture and explain auditory content in targeted regions of the audio sequence.

## Experimental Setup

Dataset. We use AudioCaps (Kim et al. 2019) as the source of textual prompts. For each prompt, we generate a 5-second audio clip using AudioGen, pairing each prompt with its corresponding generated audio. For hyperparameter tuning, we select 100 validation captions, while the test dataset consists of 1,000 randomly selected captions.

Evaluation Metrics. We evaluate explanations based on two metrics: Fidelity and KL divergence, both derived from the classification probabilities of a pre-trained audio classifier. Specifically, we utilize PaSST (Cai et al. 2022), a classifier trained on the AudioSet dataset, which is also used in the evaluation of AudioGen. Its classification probabilities are likely to provide meaningful insights into the relationship between textual prompts and generated audio. Fidelity (Yuan et al. 2021; Ali et al. 2023), a core evaluation metric in the field of XAI, measures the change in top-1 label prediction probabilities of the generated audio after applying factual and counterfactual explanation masks, denoted as Fid F and Fid CF , respectively.

In addition, KL divergence (Kilgour et al. 2018), originally used to evaluate audio generative models (Kreuk et al. 2023; Yang et al. 2023; Huang et al. 2023), measures the differences of label distribution between generated and reference audio. For explanation evaluation, we introduce new metrics KL F and KL CF , which measure the conceptual change in the generated audio after applying explanation masks in factual and counterfactual reasoning, respectively. In factual evaluation, the generated audio should closely match the original audio, making lower values Fid F and KL F desirable. In contrast, in counterfactual evaluation, higher values of Fid CF and KL CF indicate a more effective explanation. Additionally, we include the average mask size as part of our evaluation.

Baselines. We compare our method with five baselines. Random-Mask is a mask with randomly assigned values ranging between 0 and 1. Grad-CAM (Selvaraju et al. 2017) is evaluated in two variations: Grad-CAM-a and Grad-CAM-e. Specifically, Grad-CAM-a computes the gradients of the latent representation vector of the t -th audio token e t with respect to the generated audio sequence z t , while Grad-CAM-e computes the gradients of the last crossattention map to the z t . We also include the AtMan (Deiseroth et al. 2023) and the method proposed by Chefer et al. (Chefer, Gur, and Wolf 2021) as baselines.

Experimental Setting. The Explainer model includes a linear layer that reduces the text token embeddings from 1536 to 512 dimensions, followed by a PReLU activation function. The 512-dimensional text token embeddings are then mapped to a single value through another linear layer and a sigmoid function, producing a value in the [0 , 1] range. AGumbel-Softmax function is subsequently applied to push values closer to 0 or 1, representing the importance of each text token. The Explainer is trained for 50 epochs with a learning rate as × 10 -3 using the Adam optimizer. Hyperparameters are set as α = 1 × 10 -3 and β = 1 × 10 -1 as coefficients for the explanation objective function. Hyperparameter sensitivity analysis and detailed experimental settings are both provided in the Appendix. Our code is available at the following link 1 .

1 https://github.com/hjkng/audiogenX

Table 1: Evaluation of explanations generated by each method using factual and counterfactual reasoning. Five audio samples are generated and evaluated with different seeds based on the obtained explanations. The best results are highlighted in bold .

| Method               | Fid F ↓           | Fid CF ↑          | KL F ↓            | KL CF ↑           | Size ↓   |
|----------------------|-------------------|-------------------|-------------------|-------------------|----------|
| N audio = 5          | 0 . 128 ± 0 . 004 | -                 | 1 . 318 ± 0 . 030 | -                 | -        |
| Random-Mask          | 0 . 196 ± 0 . 004 | 0 . 195 ± 0 . 006 | 1 . 884 ± 0 . 044 | 1 . 932 ± 0 . 046 | 0.500    |
| Grad-CAM-e           | 0 . 204 ± 0 . 006 | 0 . 235 ± 0 . 008 | 1 . 858 ± 0 . 034 | 2 . 457 ± 0 . 041 | 0.422    |
| Grad-CAM-a           | 0 . 240 ± 0 . 006 | 0 . 192 ± 0 . 010 | 2 . 285 ± 0 . 077 | 1 . 951 ± 0 . 075 | 0.406    |
| AtMan                | 0 . 195 ± 0 . 008 | 0 . 222 ± 0 . 008 | 2 . 010 ± 0 . 049 | 2 . 198 ± 0 . 048 | 0.497    |
| Chefer et al.        | 0 . 198 ± 0 . 003 | 0 . 229 ± 0 . 004 | 1 . 899 ± 0 . 025 | 2 . 348 ± 0 . 040 | 0.441    |
| AudioGenX w/ Eq. (3) | 0 . 145 ± 0 . 004 | 0 . 360 ± 0 . 005 | 1 . 542 ± 0 . 024 | 3 . 658 ± 0 . 061 | 0.360    |
| AudioGenX w/ Eq. (5) | 0 . 143 ± 0 . 004 | 0 . 385 ± 0 . 005 | 1 . 514 ± 0 . 043 | 3 . 977 ± 0 . 044 | 0.385    |
| AudioGenX w/ Eq. (7) | 0 . 137 ± 0 . 005 | 0 . 402 ± 0 . 005 | 1 . 418 ± 0 . 043 | 4 . 183 ± 0 . 073 | 0.455    |
| AudioGenX            | 0.132 ± 0 . 004   | 0.405 ± 0 . 004   | 1.416 ± 0 . 029   | 4.259 ± 0 . 039   | 0.455    |

Figure 3: Visualization of AudioGenX and other methods.

<!-- image -->

## Experimental Results

## RQ 1: Does AudioGenX Generate Faithful Explanations?

We evaluate the generated explanations by AudioGenX based on factual and counterfactual reasoning, as presented in Table 1. AudioGenX achieves the best performance across the metrics Fid F , Fid CF , KL F , and KL CF , while also maintaining the smallest size ( Size ), demonstrating that our explanations are both simple and effective. The baseline, denoted as N audio = 5 , generates audio conditioned on the same textual input five times to observe the inherent variance, serving as the lower bound for Fid F , KL F . AudioGenX's factual audio nearly reaches the lower bound, indicating high performance. Furthermore, significant changes in Fid CF and KL CF under counterfactual perturbations confirm that the explanations are both sufficient and necessary. The AudioGenX with factual and counterfactual losses in Eq.(6), outperforms the variants AudioGenX w/ Eq. (3) and AudioGenX w/ Eq. (5), which apply only factual or counterfactual loss with a regularization term. This indicates that the two losses complement each other, enhancing overall performance. Furthermore, we evaluate AudioGenX w/ Eq. (7) using an averaged explanation mask, showing the robustness of explainability in describing the entire audio. In contrast, other baselines fail to generate meaningful counterfactual audio, lacking the optimization properties needed to enforce counterfactual explanations.

The strong performance highlights the effectiveness of leveraging latent embedding vectors to generate explanations. While most baselines are designed to explain supervised learning models, they rely on vectors that represent the probability distribution of the final audio token. This approach, however, does not align well with the inference process of audio generation models. In extreme cases, such as topk sampling ( k =250), the 250-th audio token could be sampled, leading to significant discrepancies between the gradients or probability-related information the token most likely predicted by the model. In contrast, our approach avoids dependency on the sampling process, allowing the model to produce more faithful explanations.

## RQ 2: How Well Do the Explanations from AudioGenX Reflect the Generated Audio?

We visualize the explanations generated by AudioGenX and other baselines, as shown in Figure 3. AudioGenX demonstrates a clear advantage in focusing on key audio elements. Unlike other baselines, which often assign relatively high importance scores to less important tokens like 'A' and 'with', AudioGenX consistently assigns higher importance scores to crucial tokens such as 'ticktocks' and 'music'. For instance, AudioGenX assigns a notably high importance score of 0.96 to 'music', emphasizing its ability to focus on significant input tokens. In contrast, other models like Grad-CAM-e and AtMan distribute importance more broadly, including less relevant tokens. These results show that AudioGenX consistently provides faithful explanations, aligning the generated audio with the essential components of the input text.

Furthermore, when generating audio from a prompt containing multiple concepts, some words may be less prominently reflected. In such case, AudioGenX provides adequate explanations for each specific audio, indicating whether each word from the prompt has been incorporated into the generated audio. As illustrated in Figure 4, the difference between the two audios is that bird sounds are present in Figure 4-(a) but absent in Figure 4-(b). AudioGenX effectively describes the audios by assigning high importance scores of 0.98 and 0.99 to the token 'Water,' which is the primary sound in both audios. AudioGenX assigns a score of 0.54 to 'birds,' while it assigns a score of 0.14, accurately reflecting the different audio characteristics in each case. These results show that AudioGenX can provide explanations that are well-suited to the corresponding audio. Furthermore, these explanations serve as valuable insights for editing generated audio to better align with user intention.

Figure 4: Explanation generated by AudioGenX for two audios created from a single prompt. (a) includes bird sounds, while (b) does not.

<!-- image -->

Figure 5: Explanations generated from negated prompts: (a) single negation, (b) double negation.

<!-- image -->

## RQ 3: How Can Explanations Help Understand AudioGen Behavior?

We explore the output patterns of AudioGen using the explanations generated by AudioGenX. First, we investigate whether AudioGen can effectively handle sentences containing negations and double negations, as shown in Figure 5. The explanations of the generated audios are presented in response to input prompts containing 'without thunder' and 'without no thunder.' In both cases, the generated audio includes the sound of thunder along with the rain. Using AudioGenX, we observe that 'without' and 'without no' have lower importance compared to 'thunder' in the explanations. We hypothesize that this occurs because the training dataset lacks sufficient examples of negation and double negation. An examination of the AudioCaps dataset reveals a scarcity of such cases. Additionally, by aggregating tokens from the explanations, we identify the top and bottom 50 tokens in Table 3 in the Appendix. Tokens with high importance are predominantly nouns, such as 'thunder,' while those with low importance include sound descriptors like 'distant,' as well as sequential expressions like 'before.' Such analyses could be used to debug TAG models or to identify potential inherent biases in their behavior.

## RQ 4: Does AudioGenX Generate Explanations Efficiently?

We evaluate the efficiency of explanation methods based on the average time and total GPU memory usage per explana- tion, as shown in Table 2. For GPU memory efficiency, the results rank in the following order: AtMan, Grad-CAM-e, AudioGenX, Grad-CAM-a, and Chefer et al. For time efficiency, the order is AtMan, Grad-CAM-e, Chefer et al., Grad-CAM-a, and AudioGenX. Although AtMan is the most efficient, its performance remains subpar due to its simplistic approach. Grad-CAM-e demonstrates greater memory efficiency compared to Grad-CAM-a and Chefer et al., as it tracks a shallower layer. While AudioGenX requires additional computational time to train explanation masks, it achieves memory efficiency by reducing GPU storage and operates with O ( Lk ) complexity, ensuring linear scalability for large-scale tasks.

Table 2: Efficiency analysis of AudioGenX and other baseline methods. The best results are highlighted in bold .

| Method               |   Memory (MB) ↓ |   Time (s) ↓ |
|----------------------|-----------------|--------------|
| Grad-CAM-e           |        8641.306 |       49.038 |
| Grad-CAM-a           |       41655.848 |       62.276 |
| AtMan                |        5081.957 |        7.295 |
| Chefer et al.        |       41684.969 |       52.166 |
| AudioGenX w/ Eq. (3) |       11980.894 |       36.639 |
| AudioGenX w/ Eq. (5) |       11981.114 |       37.373 |
| AudioGenX w/ Eq. (7) |       12001.931 |       63.198 |
| AudioGenX            |       12001.931 |       63.198 |

## Conclusion

AudioGenX quantifies the importance of textual tokens corresponding to generated audio by leveraging both factual and counterfactual reasoning frameworks. This approach enables the generation of faithful explanations, providing actionable insights for users to edit audio and assisting developers in debugging. Consequently, AudioGenX enhances the transparency and trustworthiness of TAG models.

## Acknowledgements

This work was supported by NCSOFT, the Institute of Information &amp; Communications Technology Planning &amp; evaluation (IITP) grant, and the National Research Foundation of Korea (NRF) grant funded by the Korean government (MSIT) (RS-2019-II190421, IITP2025-RS-2020-II201821, RS-2024-00438686, RS-202400436936, RS-2024-00360227, RS-2023-0022544, NRF2021M3H4A1A02056037, RS-2024-00448809). This research was also partially supported by the Culture, Sports, and Tourism R&amp;D Program through the Korea Creative Content Agency grant funded by the Ministry of Culture, Sports and Tourism in 2024 (RS-2024-00333068, RS-202400348469 (25%)).

Abnar, S.; and Zuidema, W. 2020. Quantifying attention flow in transformers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , 4190-4197. Association for Computational Linguistics.

Adebayo, J.; Gilmer, J.; Muelly, M.; Goodfellow, I.; Hardt, M.; and Kim, B. 2018. Sanity checks for saliency maps. Advances in Neural Information Processing Systems , 31.

Akman, A.; and Schuller, B. W. 2024. Audio Explainable Artificial Intelligence: A Review. Intelligent Computing , 2: 0074.

Ali, S.; Abuhmed, T.; El-Sappagh, S.; Muhammad, K.; Alonso-Moral, J. M.; Confalonieri, R.; Guidotti, R.; Del Ser, J.; D´ ıaz-Rodr´ ıguez, N.; and Herrera, F. 2023. Explainable Artificial Intelligence (XAI): What we know and what is left to attain Trustworthy Artificial Intelligence. Information fusion , 99: 101805.

Becker, S.; Ackermann, M.; Lapuschkin, S.; M¨ uller, K.R.; and Samek, W. 2018. Interpreting and explaining deep neural networks for classification of audio signals. arXiv preprint arXiv:1807.03418 .

Blumer, A.; Ehrenfeucht, A.; Haussler, D.; and Warmuth, M. K. 1987. Occam's razor. Information Processing Letters , 24(6): 377-380.

Cai, J.; Fan, J.; Guo, W.; Wang, S.; Zhang, Y.; and Zhang, Z. 2022. Efficient deep embedded subspace clustering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 1-10.

Chefer, H.; Gur, S.; and Wolf, L. 2021. Generic attentionmodel explainability for interpreting bi-modal and encoderdecoder transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision , 397-406.

D´ efossez, A.; Copet, J.; Synnaeve, G.; and Adi, Y. 2023. High Fidelity Neural Audio Compression. Transactions on Machine Learning Research .

Deiseroth, B.; Deb, M.; Weinbach, S.; Brack, M.; Schramowski, P.; and Kersting, K. 2023. AtMan: Understanding transformer predictions through memory efficient attention manipulation. Advances in Neural Information Processing Systems , 36.

Frommholz, A.; Seipel, F.; Lapuschkin, S.; Samek, W.; and Vielhaben, J. 2023. XAI-based Comparison of Input Representations for Audio Event Classification. In Proceedings of the International Conference on Content-Based Multimedia Indexing , 126-132.

Haunschmid, V.; Manilow, E.; and Widmer, G. 2020. audioLIME: Listenable Explanations Using Source Separation. Proceedings of the International Workshop on Machine Learning and Music. arXiv:2008.00582.

Hertz, A.; Mokady, R.; Tenenbaum, J.; Aberman, K.; Pritch, Y.; and Cohen-Or, D. 2022. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626 .

Huang, R.; Huang, J.; Yang, D.; Ren, Y.; Liu, L.; Li, M.; Ye, Z.; Liu, J.; Yin, X.; and Zhao, Z. 2023. Make-an-audio: Textto-audio generation with prompt-enhanced diffusion mod- els. In Proceedings of the International Conference on Machine Learning , 13916-13932.

Jang; Eric; Gu; Shixiang; Poole; and Ben. 2017. Categorical reparameterization with gumbel-softmax. In Proceedings of the International Conference on Learning Representations .

Kenny, E. M.; Delaney, E. D.; Greene, D.; and Keane, M. T. 2021. Post-hoc explanation options for XAI in deep learning: The Insight centre for data analytics perspective. In Proceedings of the ICPR International Workshops and Challenges , 20-34.

Kilgour, K.; Zuluaga, M.; Roblek, D.; and Sharifi, M. 2018. Fr´ echet Audio Distance: A Metric for Evaluating Music Enhancement Algorithms. arXiv preprint arXiv:1812.08466 .

Kilgour, K.; Zuluaga, M.; Roblek, D.; and Sharifi, M. 2019. Fr´ echet Audio Distance: A Reference-Free Metric for Evaluating Music Enhancement Algorithms. In Proceedings of the Annual Conference of the International Speech Communication Association , 2350-2354.

Kim, C. D.; Kim, B.; Lee, H.; and Kim, G. 2019. Audiocaps: Generating captions for audios in the wild. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics , 119-132.

Koutini, K.; Schl¨ uter, J.; Eghbal-Zadeh, H.; and Widmer, G. 2021. Efficient training of audio transformers with patchout. arXiv preprint arXiv:2110.05069 .

Kreuk, F.; Synnaeve, G.; Polyak, A.; Singer, U.; D´ efossez, A.; Copet, J.; Parikh, D.; Taigman, Y.; and Adi, Y. 2023. Audiogen: Textually guided audio generation. In Proceedings of the International Conference on Learning Representations .

Liu, H.; Chen, Z.; Yuan, Y.; Mei, X.; Liu, X.; Mandic, D.; Wang, W.; and Plumbley, M. D. 2023. Audioldm: Text-toaudio generation with latent diffusion models. In Proceedings of the International Conference on Machine Learning , 21450-21474.

Lundberg, S. M.; and Lee, S.-I. 2017. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems , 30.

Nagahisarchoghaei, M.; Nur, N.; Cummins, L.; Nur, N.; Karimi, M. M.; Nandanwar, S.; Bhattacharyya, S.; and Rahimi, S. 2023. An empirical survey on explainable ai technologies: Recent trends, use-cases, and categories from technical and application perspectives. Electronics , 12(5): 1092.

Pearl, J. 2009. Causal inference in statistics: An overview. Statistics Surveys , 3: 96 - 146.

Raffel, C.; Shazeer, N.; Roberts, A.; Lee, K.; Narang, S.; Matena, M.; Zhou, Y.; Li, W.; and Liu, P. J. 2020. Exploring the limits of transfer learning with a unified textto-text transformer. Journal of Machine Learning Research , 21(140): 1-67.

Ribeiro, M. T.; Singh, S.; and Guestrin, C. 2016. 'Why should I trust you?' Explaining the predictions of any classifier. In Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining , 11351144.

Samek, W.; Wiegand, T.; and M¨ uller, K.-R. 2017. Explainable artificial intelligence: Understanding, visualizing and interpreting deep learning models. arXiv preprint arXiv:1708.08296 .

Schneider, F.; Kamal, O.; Jin, Z.; and Sch¨ olkopf, B. 2023. Moˆ usai: Text-to-music generation with long-context latent diffusion. arXiv preprint arXiv:2301.11757 .

Selvaraju, R. R.; Cogswell, M.; Das, A.; Vedantam, R.; Parikh, D.; and Batra, D. 2017. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE International Conference on Computer Vision , 618-626.

Sundararajan, M.; Taly, A.; and Yan, Q. 2017. Axiomatic attribution for deep networks. In Proceedings of the International Conference on Machine Learning , 3319-3328.

Tan, J.; Geng, S.; Fu, Z.; Ge, Y.; Xu, S.; Li, Y .; and Zhang, Y. 2022. Learning and evaluating graph neural network explanations based on counterfactual and factual reasoning. In Proceedings of the ACM Web Conference , 1018-1027.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. Advances in Neural Information Processing Systems , 30.

Yang, D.; Yu, J.; Wang, H.; Wang, W.; Weng, C.; Zou, Y.; and Yu, D. 2023. Diffsound: Discrete diffusion model for text-to-sound generation. IEEE/ACM Transactions on Audio, Speech, and Language Processing , 31: 1720-1733.

Yuan, H.; Yu, H.; Gui, S.; and Ji, S. 2022. Explainability in graph neural networks: A taxonomic survey. IEEE Transactions on Pattern Analysis and Machine Intelligence , 45(5): 5782-5799.

Yuan, H.; Yu, H.; Wang, J.; Li, K.; and Ji, S. 2021. On explainability of graph neural networks via subgraph explorations. In Proceedings of the International Conference on Machine Learning , 12241-12252.

Ziv, A.; Gat, I.; Lan, G. L.; Remez, T.; Kreuk, F.; D´ efossez, A.; Copet, J.; Synnaeve, G.; and Adi, Y. 2024. Masked Audio Generation using a Single Non-Autoregressive Transformer. In Proceedings of the International Conference on Learning Representations .

## Evaluation metrics

We detail the four evaluation metrics discussed in the main manuscript. Z i represents the audio token sequence generated by a text-to-audio model given the i -th text prompt in the dataset. The sequences ( ˜ Z F ) i and ( ˜ Z CF ) i are produced by applying factual and counterfactual masks, respectively, as defined in Eqs. (2) and (4). These masks serve as explanations for both Z i and its associated text prompt. To evaluate these explanations, we use the pre-trained audio classifier PaSST (Koutini et al. 2021), denoted as q , which generates a prediction probability distribution q ( Z ) . Specifically, q ( Z ) y i represents the prediction probability for class y i . Finally, N is the total number of data points, L is the number of text tokens in the text prompt, and T is the total length of the audio token sequence. The evaluation metrics Fid F , Fid CF (Yuan et al. 2021, 2022; Ali et al. 2023) are defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where y i is the predicted class for q ( Z i ) , defined as y i = arg max c ∈ C q ( Z i ) c , and C is the set of all classes that q predicts. Moreover, KL F , KL CF (Kreuk et al. 2023; Yang et al. 2023; Huang et al. 2023), and Size are defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, D KL refers to the KL divergence. bsectionDetails on baseline setting While there is no existing XAI model specifically designed for explaining text-to-audio generative models, we adopt Transformer explanation methods for evaluation.

Rollout (Abnar and Zuidema 2020), a method for explaining Transformers, proposes aggregating attention scores recursively by multiplying attention maps in all layers. The proposed method named rolling out the attention weights is formulated as below:

<!-- formula-not-decoded -->

where ˜ A means attention rollout. A ( l i ) represents i -th raw attention map. However, applying Rollout in models with cross-attention blocks designed to handle multi-modality is challenging because the dimensions of the attention maps do not match. Therefore, we exclude Rollout from our baselines.

## Appendix

Grad-CAM (Selvaraju et al. 2017) computes the gradients of the output activations from the target layer with respect to the final prediction. The importance map is then calculated as:

<!-- formula-not-decoded -->

where A represents the output activations of the target layer, and ∇ A represents the gradient of these activations with respect to the prediction. Specifically, ∇ A comprises the gradients computed for each selected codebook, which are averaged afterward. ⊙ denotes element-wise multiplication, and ( · ) + extracts positive values. Grad-CAM-a calculates gradient of the latent representation vector e t , corresponding to the t -th audio token. The 1,536 dimensions of the audio token embedding are treated as separate channels, and their mean is used to compute the token importance score. Grad-CAM-e derives the gradient of the last cross-attention map, and their mean is used as the token importance score.

AtMan (Deiseroth et al. 2023) extracts important tokens by perturbation of a single token. For all cross-attention layers and all heads, we multiply 1 -k with the pre-softmax attention scores for the target text tokens to introduce perturbation. The value k is consistently set to 0.9, following the configuration in (Deiseroth et al. 2023). To quantify the influence of tokens, we calculate the difference in crossentropy for each codebook and use the sum of these differences as the token importance.

Chefer et al. (Chefer, Gur, and Wolf 2021) calculates the relevance score, following attention layers. In our experiments, we follow (Chefer, Gur, and Wolf 2021). However, since we do not consider the influence of the text encoder, we replace it with an identity matrix.

To scale importance values between 0 and 1, we apply Max scaling for each sequence for all baselines except AtMan, For AtMan, which includes negative values, we use Min-Max scaling.

## Experimental Setting

In our experiments, we used the following packages and hardware:

- Python 3.9.18
- spacy==3.5.2
- torch&gt;=2.1.0
- torchaudio&gt;=2.1.0
- Transformers&gt;=4.31.0

All computations were performed using a single NVIDIA A100 GPU.

## Hyperparameter Sensitivity Analysis

We conduct a hyperparameter sensitivity analysis using various combinations of hyperparameters on the validation dataset. The validation dataset is randomly sampled from the validation set of AudioCaps (Kim et al. 2019). As illustrated in Figure 6-(a), when the value of α decreases from 0.1 to 0.001, Fid F significantly decreases from 0.398 to 0.138. Furthermore, when β is adjusted from 0.1 to 0.001 while keeping α = 0 . 001 , Fid F shows a slight increase from

Figure 6: Sensitivity analysis of the hyperparameters α and β . (a) Effect on Fid F , (b) Effect on Size .

<!-- image -->

0.138 to 0.142. This suggests that lowering β can slightly enhance fidelity, but the effect is marginal compared to that of α . In contrast, the impact on Size , shown in Figure 6(b), reveals a different trend. As α decreases from 0.1 to 0.001, Size increases substantially from 0.07 to 0.51, indicating that lower α values lead to greater model complexity. Similarly, when β is reduced from 0.1 to 0.001 with a fixed α = 0 . 001 , Size further increases from 0.426 to 0.517. This demonstrates that both α and β reductions tend to increase the mask size. These results indicate a trade-off between fidelity and size. To ensure fair comparisons with the baselines while maintaining a comparable mask size, we set α = 0 . 001 and β = 0 . 1 based on this analysis.

## Leveraging Explanations to Understand Model Behavior and Edit Audio

AudioGenX enhances transparency but also provides valuable insight for debugging TAG models and editing sound. In Table 3, we investigate the patterns of AudioGen using explanations generated by AudioGenX. We first aggregated our explanations from the experiments in Table 3. Then, we filtered out tokens with a length greater than 1 and an occurrence frequency exceeding 10. From this subset, we selected the top 100 tokens with the highest average importance (Avg. Impt) and the bottom 100 tokens with the lowest average importance to generate word clouds. Detailed information on these tokens is presented in Table 3. The tokens in the top 100 predominantly consist of nouns (NN) and tokens that are associated with sounds. In contrast, the tokens in the bottom 100 displayed a diverse range of parts of speech, including adverbs (RB) and prepositions (IN), which tend to convey context rather than having intrinsic auditory significance. Additionally, as noted in the AudioGen documentation (Kreuk et al. 2023), tokens related to numbers (CD) or sequences also exhibit a lower importance.

Next, we utilize our explanations in the task of editing generated audio when it misaligns with the user's intended prompt. For example, given the prompt 'Wind blows hard followed by screaming,' the generated audio should reflect this sequence. However, in our example, AudioGen fails to accurately capture the 'screaming' sound. Using AudioGenX, we find that 'screaming' has low importance, especially in the latter part of the audio where it should be emphasized. To correct this, we used a technique in a study (Hertz et al. 2022) that adjusts attention weights to better align the audio with the intended prompt. The method of importance re-weighting is described as follows:

<!-- formula-not-decoded -->

where the explanation mask value MU l , z t from AudioGenX is reweighted to M ∗ U l , z t . In this case, l ∗ and t ∗ denote the target indices of the text token and the audio token, respectively. When amplifying the explanation mask value, we set the scaling parameter c to 0.9. Conversely, when suppressing the impact of the token, c is set to 0.1. The threshold values (0.9 and 0.1) are chosen based on our heuristic intuition in Table 3 that the importance of the top and bottom ranking is greater than 0.9 and less than 0.1, respectively. For evaluation, we randomly sampled 100 prompts and identified 30 failure cases where AudioGen-generated audio differs from the ground truth in the AudioCaps (Kim et al. 2019) data set. To analyze these outputs, we applied AudioGenX to identify which textual tokens were over- or underemphasized, then manually adjusted the importance of the tokens to align with the ground truth. As evaluation metrics, we compute the Fr´ echet Audio Distance (FAD) (Kilgour et al. 2019) over both real and generated audio. FAD is an adaptation of the Fr´ echet Inception Distance (FID) for audio, measuring the similarity between distributions of real and generated audio data. Additionally, we measure the metric KL F .

Table 4 demonstrates that editing the importance mask allows us to generate audio that more closely matches the ground truth. For the setup, we randomly sampled 100 prompts, finding 30 failure cases that differed from the ground truth in the AudioCaps data set. Initially, the scores (FAD and KL) were lower than typical ones, at 3.13 and 2.09 in MAGNeT (Ziv et al. 2024) known as the SOTA model. To understand the generated output, we used AudioGenX to identify which textual tokens were over- or underemphasized, then manually adjusted the importance of the tokens to amplify or suppress. Figure 9 further illustrates that editing can be applied to specific time intervals. For instance, after re-weighting the mask values between 2.5 and 5 seconds, the 'screaming' audio emerges in the corresponding time interval. Although explanations do not directly involve generation, AudioGenX helps users by offering valuable guidance during the editing process when there is a discrepancy between the user's intention and the generated result.

## Sanity Check

We conduct a sanity check following the approach in (Adebayo et al. 2018) to assess the explanations generated. Specifically, we initialize the parameters of the Transformerdecoder, which predicts the next sequence of audio tokens. As shown in Figure 8, when the model is initialized randomly, the influence of each token in the visualization becomes nearly indistinguishable. This result of the explainer, including our baseline, in response to the state of the model parameters, suggests that AudioGenX produces faithful explanations. Thus, we conclude that AudioGenX generates reliable and trustworthy explanations, as validated by the sanity check.

Figure 7: Qualitative analysis of AudioGenX in comparison with baseline methods.

<!-- image -->

Table 3: Averaged importance (Impt) per textual token learned by AudioGenX. The name of the POS (Part of Speech) is followed by the categories in NLTK. Count refers to the occurrence frequency in the test dataset of AudioCaps.

| Index   | Top 50 high-importance tokens   | Top 50 high-importance tokens   | Top 50 high-importance tokens   | Top 50 high-importance tokens   | Top 50 low-importance tokens   | Top 50 low-importance tokens   | Top 50 low-importance tokens   | Top 50 low-importance tokens   |       |
|---------|---------------------------------|---------------------------------|---------------------------------|---------------------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|-------|
| Index   | Textual token                   | Avg. Impt                       | Count                           | POS                             | Index                          | Textual token                  | Avg. Impt                      | Count                          | POS   |
| 1       | sewing                          | 0.987                           | 20                              | VBG                             | 1                              | ground                         | 0.037                          | 11                             | NN    |
| 2       | horse                           | 0.976                           | 13                              | NN                              | 2                              | series                         | 0.048                          | 13                             | NN    |
| 3       | emergency                       | 0.957                           | 13                              | NN                              | 3                              | electronic                     | 0.060                          | 17                             | JJ    |
| 4       | ren                             | 0.946                           | 18                              | NNS                             | 4                              | for                            | 0.086                          | 12                             | IN    |
| 5       | baby                            | 0.939                           | 22                              | NN                              | 5                              | before                         | 0.107                          | 11                             | IN    |
| 6       | thunder                         | 0.936                           | 18                              | NN                              | 6                              | background                     | 0.110                          | 107                            | NN    |
| 7       | toilet                          | 0.936                           | 13                              | VBP                             | 7                              | then                           | 0.116                          | 173                            | RB    |
| 8       | soft                            | 0.932                           | 12                              | JJ                              | 8                              | repeatedly                     | 0.119                          | 12                             | RB    |
| 9       | rog                             | 0.931                           | 13                              | NN                              | 9                              | while                          | 0.129                          | 86                             | IN    |
| 10      | foot                            | 0.931                           | 12                              | NN                              | 10                             | into                           | 0.130                          | 42                             | IN    |
| 11      | app                             | 0.931                           | 15                              | NN                              | 11                             | some                           | 0.139                          | 45                             | DT    |
| 12      | food                            | 0.915                           | 17                              | NN                              | 12                             | another                        | 0.141                          | 19                             | DT    |
| 13      | step                            | 0.898                           | 12                              | NN                              | 13                             | over                           | 0.146                          | 19                             | IN    |
| 14      | infant                          | 0.895                           | 13                              | NN                              | 14                             | distance                       | 0.163                          | 49                             | NN    |
| 15      | ack                             | 0.893                           | 13                              | NN                              | 15                             | with                           | 0.167                          | 174                            | IN    |
| 16      | talking                         | 0.892                           | 122                             | VBG                             | 16                             | the                            | 0.167                          | 204                            | DT    |
| 17      | crying                          | 0.883                           | 27                              | VBG                             | 17                             | followed                       | 0.185                          | 277                            | VBD   |
| 18      | cla                             | 0.878                           | 33                              | NN                              | 18                             | ongoing                        | 0.196                          | 10                             | VBG   |
| 19      | pig                             | 0.875                           | 13                              | IN                              | 19                             | loud                           | 0.224                          | 53                             | JJ    |
| 20      | laughter                        | 0.873                           | 18                              | NN                              | 20                             | are                            | 0.228                          | 30                             | VBP   |
| 21      | goat                            | 0.873                           | 12                              | NN                              | 21                             | and                            | 0.233                          | 568                            | CC    |
| 22      | clo                             | 0.863                           | 10                              | NN                              | 22                             | power                          | 0.250                          | 12                             | NN    |
| 23      | pour                            | 0.863                           | 10                              | VBP                             | 23                             | occurs                         | 0.260                          | 15                             | VBZ   |
| 24      | duck                            | 0.859                           | 12                              | NN                              | 24                             | sounds                         | 0.264                          | 25                             | NNS   |
| 25      | door                            | 0.857                           | 26                              | NN                              | 25                             | pitched                        | 0.264                          | 12                             | VBN   |
| 26      | tapping                         | 0.848                           | 11                              | VBG                             | 26                             | surface                        | 0.269                          | 31                             | NN    |
| 27      | footsteps                       | 0.846                           | 11                              | NNS                             | 27 28                          | several                        | 0.270                          | 22                             | JJ    |
| 28      | bus                             | 0.845                           | 11                              | NN                              |                                | from                           | 0.279                          | 37                             | IN    |
| 29      | clicking                        | 0.842                           | 14                              | VBG                             | 29                             | king                           | 0.287                          | 62                             | VBG   |
| 30      | truck                           | 0.838                           | 13                              | NN                              | 30                             | steam                          | 0.308                          | 16                             | JJ    |
| 31 32   | scrap                           | 0.834 0.823                     | 16                              | JJ                              | 31 32                          | sound through                  | 0.309 0.313                    | 18 13                          | JJ IN |
| 33      | crowd speaks                    | 0.817                           | 31 97                           | NN NNS                          | 33                             |                                | 0.325                          | 11                             | RP    |
| 34 35   | boat                            | 0.817 0.815                     | 12 14                           | NN NN                           | 34 35                          | down two                       | 0.343 0.351                    | 17 13                          | CD JJ |
| 36      | cat explosion                   |                                 | 12                              | NN                              |                                | light ing                      |                                | 342                            | VBG   |
|         |                                 | 0.813                           |                                 |                                 |                                |                                | 0.352                          |                                | NNS   |
| 38      |                                 | 0.811                           | 33                              | NN                              | 36                             |                                |                                | 16                             |       |
| 37      | woman                           | 0.811                           | 104                             |                                 | 37                             | runs                           | 0.353                          | 18                             |       |
| 39      | music clan                      | 0.810                           | 24                              | NN NN                           | 38                             | les microphone                 | 0.353                          |                                | NNS   |
|         |                                 | 0.810                           | 14                              | JJ                              | 39                             |                                | 0.355                          | 48                             | NN    |
| 40      | speech                          |                                 | 44                              | JJ                              | 40                             | high                           | 0.359                          | 20                             | JJ    |
| 41      |                                 | 0.803                           |                                 |                                 | 41                             | ting                           | 0.361                          | 14                             | VBG   |
| 42      | whistle water                   | 0.799                           | 82                              | NN                              | 42                             |                                | 0.363                          | 12                             | VBZ   |
| 44 45   | men                             | 0.789                           | 21                              | NNS                             | 44                             | ses ling                       | 0.364                          | 35 160                         | VBG   |
| 43      | speaking                        | 0.794                           | 163                             | VBG                             | 43                             | his                            | 0.364                          |                                | PRP$  |
|         | train                           | 0.785                           | 35                              | VBP                             | 45                             | ving                           | 0.375                          | 30                             | VBG   |
| 47      |                                 | 0.778                           | 40                              | VBG                             | 47                             | end                            | 0.382                          | 1003                           | NN JJ |
| 46      | rain                            | 0.782                           | 29                              | NN                              | 46                             | metal                          | 0.377                          | 45                             | VBP   |
| 48      | laughing flush                  | 0.769                           | 14 12                           | NN                              | 48 49                          | small tires                    | 0.383 0.384                    | 12 11                          | NNS   |
| 49 50   | helicopter                      | 0.768                           |                                 | NN                              |                                | motor                          | 0.417                          | 49                             | NN    |
|         | talks                           | 0.766                           | 31                              | NNS                             | 50                             |                                |                                |                                |       |

<!-- image -->

Figure 8: Examplar explanations using independent randomization on Transformer-decoder of AudioGen as the sanity check.

Table 4: Evaluation of editing generated audio

|             |   FAD ↓ |   KL F ↓ |
|-------------|---------|----------|
| Before edit |   16.85 |     6.45 |
| After edit  |    2.68 |     1.82 |

Figure 9: The scenario of editing the generated audio.

<!-- image -->

## Limitation

While we introduce a novel approach to explaining generated audio in TAG models, there are some limitations to consider. First, AudioGenX contains several hyperparameters that may require data set-specific tuning for optimal performance. Automating this process or reducing hyperparameter sensitivity would improve usability. Furthermore, biases present in the training data may be reflected in both the generated audio and the explanations. Without proper safeguards and responsible deployment practices, these biases could reinforce harmful stereotypes. As research into audio generation progresses, it is crucial to proactively develop robust bias detection methods and advocate for the ethical use of these powerful approaches. Despite these lim- itations and considerations, we believe that AudioGenX represents a valuable step toward improving the interpretability and trustworthiness of TAG models.