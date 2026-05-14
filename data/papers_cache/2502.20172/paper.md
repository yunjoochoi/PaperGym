## Multimodal Representation Alignment for Image Generation: Text-Image Interleaved Control Is Easier Than You Think

Liang Chen 1

Shuai Bai 2

Leon Vinci Weichu Xie 4 Haozhe Zhao 1 Baobao Chang 1

5

Wenhao Chai 3

Junyang Lin 2

1 Peking University 2 Alibaba Group 3 University of Washington 4 Beijing Institute of Technology 5 Bainance Labs

[https://github.com/chenllliang/DreamEngine](https://github.com/chenllliang/DreamEngine)

Figure 1. Generation examples of DREAM ENGINE. Leveraging powerful text-to-image diffusion model and large multimodal models, DREAM ENGINE is capable of generating image with text-image interleaved control by merging concepts from different images.

<!-- image -->

## Abstract

The field of advanced text-to-image generation is witnessing the emergence of unified frameworks that integrate powerful text encoders, such as CLIP and T5, with Diffusion Transformer backbones. Although there have been efforts to control output images with additional conditions, like canny and depth map, a comprehensive framework for arbitrary text-image interleaved control is still lacking. This gap is especially evident when attempting to merge concepts or visual elements from multiple images in the generation process. To mitigate the gap, we conducted preliminary experiments showing that large multimodal models (LMMs) offer an effective shared representation space, where image and text can be well-aligned to serve as a condition for external diffusion models. Based on this discovery, we propose DREAM ENGINE , an efficient and unified framework designed for arbitrary text-image interleaved control in image generation models. Building on powerful textto-image models like SD3.5, we replace the original textonly encoders by incorporating versatile multimodal information encoders such as QwenVL. Our approach utilizes a two-stage training paradigm, consisting of joint text-image alignment and multimodal interleaved instruction tuning. Our experiments demonstrate that this training method is effective, achieving a 0.69 overall score on the GenEval benchmark, and matching the performance of state-of-theart text-to-image models like SD3.5 and FLUX.

## 1. Introduction

Recent years have witnessed remarkable advancements in text-to-image generation, primarily driven by powerful diffusion models [3, 13, 19, 37]. While these models excel at generating images that align with simple text prompts, they struggle to handle more complex instructions that interweave graphical and textual elements. Although condition augmentation methods like IP-Adapter [52] and ControlNet [54] enhance text-to-image models with additional low-level control signals such as canny edges, depth maps, or reference images, they lack the flexibility to process complex and high-level text-image interleaved instructions, for example, merging visual elements from multiple images using natural language descriptions. This inability restricts more creative image generation processes where users might want to precisely orchestrate visual compositions by combining and manipulating elements from multiple sources with simple text instructions.

Meanwhile, Large Multimodal Models (LMMs) [1, 6, 26, 45] have shown remarkable progress in understanding visual content and natural language instructions, enabling various tasks such as image captioning, visual question answering, and visual grounding. This advancement raises an intriguing question: Can we take advantage of the advanced visual language understanding capabilities of LMMs to improve diffusion-based image generation models, enabling more flexible text-image interleaved control?

Figure 2. Overview Comparison. Among all types of works connecting LMM and diffusion model, our DREAM ENGINE adopts the simplest design yet achieves the best performance.

<!-- image -->

Several recent works have explored integrating LMMs with diffusion models to enhance image generation control. As shown in Figure 2, Emu-1 and 2 [41, 42] incorporate a specialized regression head on the hidden output states of LMMtokens following multimodal input processing. SeedTokenizer [15] expands the LMM vocabulary with discrete vision tokens, which serve as condition for the diffusion model during image generation. BLIP-Diffusion [23] employs a multimodal query-transformer encoder to extract subject representations, which are then combined with text prompts to guide the generation process.

However, many of these approaches merely add text-toimage generation capabilities to LMMs without improving generation quality [57] or expanding potential applications. Some methods [23] are designed for specific tasks and can only process a single conditioning image, limiting their utility in scenarios involving multiple image inputs. To the best of our knowledge, no existing models can effectively perform compositional image generation tasks, indicating a gap in understanding text-image interleaved control, particularly when multiple images are involved.

Our insight is that the fundamental challenge lies in effectively representing multimodal interleaved control, where mapping both text and images into a unified semantic space is crucial for coherent alignment. In this work, we demonstrate that Large Multimodal Models inherently provide a unified representation space, eliminating the need for additional architectural components such as regression heads or specialized tokens. We propose DREAM ENGINE, an efficient and effective framework for image generation that accepts arbitrary text-image interleaved control signals. Building upon open-source text-to-image diffusion models like Stable Diffusion v3.5 [13], we replace its text encoders with a LMM along with a lightweight projector layer to encode the text-image interleaved controls. We introduce a two-stage training paradigm that efficiently aligns the representation spaces between these backbone models, enabling the generation of images guided by interleaved text and image instructions. We also design a new task called objects driven generation, which leverages object detection and image captioning data to enable compositional generation.

Figure 3. DREAM ENGINE architecture.

<!-- image -->

Our experiments demonstrate the effectiveness of our architecture design and training recipe. By fine-tuning only an MLP layer on 20 millions data during Stage-I training, our model achieves an overall score of 0.69 on the GenEval benchmark, matching the performance of state-of-the-art text-to-image models such as SDv3.5 (0.71) and surpassing FLUX.1 Dev (0.66). This result highlights the efficacy of our alignment tuning method and demonstrates that powerful multimodal encoders can replace text encoders without compromising the original diffusion model's image generation quality. Furthermore, our Stage-II model exhibits strong text-image interleaved instruction following, significantly outperforming comparable models like Emu2gen with substantially less training data. Notably, it can even synthesize concepts from different input images based on the text prompt to generate a cohesive output image as shown in Figure 1. Our contributions are threefold:

- We found that Large Multi-Modal Models can be easily adapted into the text encoder of the text-to-image diffusion models even without updating the parameters.
- We achieve object-driven generation, combining object detection and captioning for compositional generation.
- Our method allows for complex, interwoven guidance from both text and images, resulting in highly customized outputs and state-of-the-art quality.

## 2. Methods

Wetarget at enabling diffusion models to take different textimage interleaved control in a unified manner by introducing a large multimodal model. We first concisely introduce LMM and MM-DiT architecture in Section 2.1, which are the foundational components of our method. Next, we introduce the structure design of DREAM ENGINE, explaining how do we align LMM with the diffusion model and how to maintain visual features consistency in Section 2.2. Last, we introduce the two-stage training recipe and how to curate data from different tasks in in Section 2.3.

## 2.1. Preliminary: LMM and MM-DiT

Large Multimodal Models Large multimodal models typically include three major modules, a visual encoder usually a ViT [12] structure, a large language model backbone and an alignment layer used to align the representation space between the visual encoder and LLM [1, 9, 26, 45, 55]. Given an input image i , this data first passes through the visual encoder followed by the alignment layer, resulting in a transformed representation denoted as h ViT . Subsequently, h ViT is processed by the LLM backbone consisting of multiple transformer decoder blocks, evolving into the output image hidden states h LLM used for text generation.

Multimodal Diffusion Transformer MM-DiT structure [13], is the basic structure for state-of-the-art text-toimage diffusion models such as SD3.5 [13] and FLUX [3].

Instruction : A cat &lt;Image1&gt; and a girl &lt;Image2&gt; on the beach &lt;Image3&gt;

<!-- image -->

Figure 4. Performance demonstration on Natural Object Background Merging where DREAM ENGINE can understand complex text-image input. It can even set more than one object in the background (last line).

It is built upon the Latent Diffusion Model (LDM) [37] and Diffusion Transformer (DiT) [32]. It concatenates textual conditioning information c with noised latent embeddings x into a unified sequence. Within the DiT module, MMDiT employs distinct LayerNorm and MLP layers for each modality while merging their sequences during the attention mechanism. This design allows each representation to evolve within its own specialized space while still considering the influence of the other modality. We employ the embeddings of the timestep t and pooled representation of text condition in the modulation mechanism of DiT. We use rectified flow matching [28] as the training objectives to conduct text-to-image generation in Section 2.2.

## 2.2. Model Design

Weadopt the MM-DiT module from Stable Diffusion 3 [13] model as the DiT model and Qwen2VL [45] as the LMM to compose DREAM ENGINE.

Align LMM and MM-DiT As shown in Figure 3, we completely replace the text encoders including CLIP [34] and T5 [51] from the text-to-image diffusion models with the LMM to get a unified representation of text and image c . To align the representation space of pretrained LMM with that of previous encoders and enable the MM-DiT module to take image input, we add a straight-forward adapter layer consisting of a two-layer MLP. The adapter maps the output hidden states of LMM to the conditioning feature space of MM-DiT. We add the average pooling representations of the LMM condition and timestep embedding as the modulation embedding y in the MM-DiT model. We remove the token length limit of original text encoders so that DREAM ENGINE can take any sequence length of text-image input.

Blending Visual Feature for Better Consistency To control the visual consistency in image editing and objectsdriven generation tasks, we add a skip connection for visual features in the LMM to avoid visual information loss in the LMMbackbone model. As shown in Figure 3, the final hidden states of image patches h I are a weighted sum of the LLM output hidden states h LLM and ViT image features h ViT :

<!-- formula-not-decoded -->

The adjustable blending ratio r allows for the control of image feature consistency between the input and output images, tailored to specific applications. During the training phase, r ∈ [0 , 1] adheres to a uniform distribution, which enables the flexibility to assign various values to r during the inference process as shown in Figure 9.

Training Objectives We adopt rectified flows [28] to learn the transition between the target data distribution x 0 and a standard normal distribution ϵ , i.e.

<!-- formula-not-decoded -->

where t ∈ [0 , 1] represents the timestep, and z t denotes the corresponding distribution at the t -th step. At each step, based on the current distribution z t , a condition c , and the timestep t , the model directly parameterizes the velocity v θ ( z t , c , t ) . This velocity is expected to approximate x 0 -ϵ during the flow matching process. It is important to note that the condition c can include interleaved text-image control, as opposed to solely text-based information as seen in the original text-to-image diffusion models. The training objective is to minimize the expected L2 loss by updating the model parameters v θ , with a weight w t assigned to each timestep, i.e.

<!-- formula-not-decoded -->

We use the Euler Discrete Scheduler [21] following SD3 [13] to set the timestep. The target data distribution x 0 comes from the latent representation of VAE, which is the same as the VAE of SD3 following a standard Latent Diffusion Model [37] training process.

## 2.3. Training Stages

Given that DREAM ENGINE comprises two individually pretrained components-the LMM and the DiT-it is crucial to ensure their alignment. Adhering to the established practices outlined in the LMM literature [1, 8, 26, 44], we have structured the training process into distinct phases, each designed to unfreeze specific model components to promote stable and effective training. As shown in Figure 5, our approach involves two primary training stages, where each stage has its own training tasks and trainable modules. In the S1 stage, we focus on training only the adapter layer, which facilitates the alignment of the representation spaces between the LMM and the DiT. During the S2 stage, we train both the adapter and the DiT, allowing for more sophisticated control over the generation process. We also show the training examples of each task in Figure 5.

Stage 1: Joint Text and Image Alignment In the first stage, we focus on aligning the representation spaces of the LMM and DiT modules by training a dedicated adapter, while keeping the parameters of both the LMM and DiT frozen. This alignment process involves two complementary tasks.

- Task A: Text-to-Image Alignment. It leverages highquality image-caption pairs to establish a foundational

Stage 1: Joint Text and Image Alignment

<!-- image -->

Stage 2: Interleaved Condition Instruction Tuning

Figure 5. Training stages and tasks of DREAM ENGINE.

<!-- image -->

correspondence between textual descriptions and generated images, effectively replacing the original text encoders.

- Task B: Image-to-Image Alignment. It is a selfsupervised task that enables DiT to condition on image inputs. Specifically, DiT is trained to reconstruct input images based on the LMM's image representations, thereby enhancing the consistency and fidelity of visual elements.

Upon completing Stage 1 training, DREAM ENGINE acquires two core capabilities: text-to-image generation and image variation . Interestingly, we observe that the two tasks mutually reinforce each other. Even when trained solely on one task, the model demonstrates a certain degree of capability in the other task in a zero-shot manner. This finding suggests that the LMM inherently provides a unified representation space for text and images, which the DiT can effectively leverage during training. As shown in Table 3, our model trained without the Image-to-Image Alignment task can still achieve a relatively high (0.7+) CLIP score in the image reconstruction evaluation.

Stage 2: Interleaved Condition Instruction Tuning In the second stage, we unfreeze the DiT module and train it on two tasks that require interleaved image-text conditioning.

- Task C: Free-Form Image Editing. It takes an input image along with an editing instruction and outputs the edited image. We use the UltraEdit [56] dataset as the dataset.
- Task D: Objects Driven Generation. It accepts multiple input images and a textual instruction, composing elements from the input images based on the given text to generate the output. For Task 4, we construct the training data using object detection datasets, such as COCO [25],

pairing images with captions that describe the objects present.

After the second stage, the model gains the ability to handle interleaved image-text conditions during generation. Surprisingly, we observe emergent capabilities in DREAM ENGINE. Notably, it can synthesize elements from different objects to generate cohesive images, as demonstrated in Figure 6, despite such compositions not being explicitly present in the training data.

## 3. Experiments

## 3.1. Dataset

Table 2 provides an overview of the datasets used for training DREAM ENGINE, along with the number of examples drawn from each source. In Stage 1, for the Text-toImage Alignment task, we compile public image-caption datasets, including real-world images from CC12M [4] and model-generated images from JourneyDB [39]. Additionally, we synthesize a subset of high-quality images using diverse prompts with open text-to-image models, such as Flux.1 dev [3] and Stable-Diffusion v3.5 Large [13]. For the Image-to-Image Alignment task, we rely solely on images from JourneyDB, as lower-aesthetic-quality images, such as images from CC12M, tend to degrade overall image reconstruction and text-to-image performance. In Stage 2, we utilize the UltraEdit [56] dataset for the Free-Form Image Editing task and an internal object detection dataset for Object-Driven Generation. For the latter, we randomly select three objects from each image. Additionally, the name of each selected object must appear in the text caption to compose the conditioning input.

Table 1. Performances on GenEval benchmark. We split the methods to autoregressive and diffusion based models.

| Method            | Single Object   | Two Object   | Counting   | Colors   | Position   | Attribute Binding   | Overall   |
|-------------------|-----------------|--------------|------------|----------|------------|---------------------|-----------|
| Chameleon [43]    | -               | -            | -          | -        | -          | -                   | 0 . 39    |
| LWM[27]           | 0 . 93          | 0 . 41       | 0 . 46     | 0 . 79   | 0 . 09     | 0 . 15              | 0 . 47    |
| LlamaGen [40]     | 0 . 71          | 0 . 34       | 0 . 21     | 0 . 58   | 0 . 07     | 0 . 04              | 0 . 32    |
| Show-o [50]       | 0 . 95          | 0 . 52       | 0 . 49     | 0 . 82   | 0 . 11     | 0 . 28              | 0 . 53    |
| Emu 3 -Gen [46]   | 0 . 98          | 0 . 71       | 0 . 34     | 0 . 81   | 0 . 17     | 0 . 21              | 0 . 54    |
| Janus [48]        | 0 . 97          | 0 . 68       | 0 . 30     | 0.84     | 0.46       | 0 . 42              | 0 . 61    |
| LDM [36]          | 0 . 92          | 0 . 29       | 0 . 23     | 0 . 70   | 0 . 02     | 0 . 05              | 0 . 37    |
| SDv 1 . 5 [36]    | 0 . 97          | 0 . 38       | 0 . 35     | 0 . 76   | 0 . 04     | 0 . 06              | 0 . 43    |
| PixArt- α [5]     | 0 . 98          | 0 . 50       | 0 . 44     | 0 . 80   | 0 . 08     | 0 . 07              | 0 . 48    |
| SDv 2 . 1 [36]    | 0 . 98          | 0 . 51       | 0 . 44     | 0 . 85   | 0 . 07     | 0 . 17              | 0 . 50    |
| DALL-E 2 [35]     | 0 . 94          | 0 . 66       | 0 . 49     | 0 . 77   | 0 . 10     | 0 . 19              | 0 . 52    |
| SDXL [33]         | 0 . 98          | 0 . 74       | 0 . 39     | 0.85     | 0 . 15     | 0 . 23              | 0 . 55    |
| IF-XL [10]        | 0 . 97          | 0 . 74       | 0 . 66     | 0 . 81   | 0 . 13     | 0 . 35              | 0 . 61    |
| DALL-E 3 [2]      | 0 . 96          | 0 . 87       | 0 . 47     | 0 . 83   | 0.43       | 0 . 45              | 0 . 67    |
| SDv3 Medium [13]  | 0.98            | 0.74         | 0.63       | 0.67     | 0.34       | 0.36                | 0.62      |
| Flux.1 Dev [3]    | 0.98            | 0.81         | 0.74       | 0.79     | 0.22       | 0.45                | 0.66      |
| SDv3.5 Large [13] | 0.98            | 0.89         | 0.73       | 0.83     | 0.34       | 0.47                | 0.71      |
| DREAM ENGINE      | 1.00            | 0.94         | 0.64       | 0.81     | 0.27       | 0.49                | 0.69      |

Table 2. Details on datasets used in training DREAM ENGINE within the two training stages.

| Stage   | Dataset                                 | Task                                                                                   | Number   |
|---------|-----------------------------------------|----------------------------------------------------------------------------------------|----------|
| 1 2     | JourneyDB [39] CC12M [4] Synthetic Data | Text-to-Image Alignment Text-to-Image Alignment Text-to-Image Alignment Image-to-Image | 4M 4M 4M |
| 1 2     | JourneyDB [39]                          | Alignment                                                                              | 4M       |
| 1 2     | UltraEdit [56]                          | Free Form Image Edit                                                                   | 1M       |
| 1 2     | Internal Data                           | Object-Driven Generation                                                               | 4M       |

## 3.2. Model and Training Details

We initialize the LMM and DiT module of DREAM ENGINE from Qwen2VL-2B-Instruct [45] and StableDiffusion-3.5-Large [13]. The Adapter consists of a twolayer MLP with a middle projection dimension of 4,096 and uses SiLU as the activation function following the DiT module. In Stage 1, we freeze the parameters of the LMM and DiT modules and train the Adapter on the composed dataset for one epoch with a global batch size of 128. The learning rate is set to 1e-4, with 5% warmup steps and a cosine learning rate scheduler. In Stage 2, we also fine-tune the DiT module using LoRA [20] with a rank of 32 on all attention layers. The learning rate is set to 5e-5, while all other settings remain the same as in Stage 1. We do not fine-tune the LMM component of the model, thus preserving its original multimodal understanding capabilities. This design choice allows the model to be easily adapted into an omni-model, capable of performing both multimodal understanding and generation simultaneously. On the other hand, unfreezing the LMM during training has large potential in further improving the generation performance.

Table 3. Image reconstruction performance comparison on COCO and JourneyDB datasets.

| Method                 | COCO       | COCO     | JourneyDB   | JourneyDB   |
|------------------------|------------|----------|-------------|-------------|
|                        | CLIP ( ↑ ) | L2 ( ↓ ) | CLIP ( ↑ )  | L2 ( ↓ )    |
| SeedTokenizer          | 0.7760     | 0.5102   | 0.7921      | 0.5291      |
| EMU2-Gen               | 0.8537     | 0.3828   | 0.9299      | 0.2869      |
| SEED-X                 | 0.8595     | 0.4317   | 0.9017      | 0.4352      |
| DREAM ENGINE           | 0.8714     | 0.2065   | 0.9221      | 0.2052      |
| - w/o I-to-I Alignment | 0.7184     | 0.6541   | 0.7536      | 0.6543      |

## 3.3. Results and Comparisons

Text-to-Image Generation We evaluate the text-toimage generation capability of DREAM ENGINEon the GenEval [18] benchmark following Stage 1 training. The results, including fine-grained scores, are presented in Table 1. Built upon the SDv3.5 [13] model, DREAM ENGINEachieves a competitive overall score of 0.69, closely matching the original model's 0.71 despite excluding its native text encoders. Moreover, DREAM ENGINEoutperforms all other counterparts, demonstrating the effectiveness and efficiency of our text-to-image alignment training in preserving instruction-following capabilities while replacing the original text encoders of diffusion models to enable more complex interleaved conditions.

Image Reconstruction We introduce an Image Reconstruction Benchmark to evaluate the preservation of visual like the girl &lt;Image2&gt;.

Image1

<!-- image -->

Instruction : A cat &lt;Image1&gt; wearing head scarf

<!-- image -->

<!-- image -->

Instruction : A dog &lt;Image1&gt; wearing head scarf like the girl &lt;Image2&gt;.

Figure 6. Performance demonstration on Object Driven Feature Mixing task. DREAM ENGINE can understand the complex instruction while Emu2-Gen fails on the task.

<!-- image -->

Figure 7. Performance demonstration on Free Form Image Editing task. DREAM ENGINE outperforms the counterpart Emu2-Gen model in both instruction following and output image quality.

Image2

Emu2-Gen

<!-- image -->

<!-- image -->

Dream Engine (Ours)

<!-- image -->

<!-- image -->

## Image Reconstruction Results

<!-- image -->

Training Compute (Joint Image-Language Alignment)

Figure 8. Image reconstruction performance dynamics during training. We can see that there is a concept-to-detail transition during the training period.

features in our Image-to-Image alignment task during Stage 1. This capability is essential for generating images conditioned on input images. To construct the benchmark, we randomly sample 100 images from the JourneyDB development set and 100 images from the COCO development set. We assess the similarity between the original and reconstructed images using the CLIP [34] score and L2-Distance. As shown in Table 3, we compare the performance of DREAM ENGINE against several baselines with similar architectures, including SeedTokenizer [15], EMU-2 [42], and SeedX [17], which also integrate LMMs and diffusion models for generation. The results demonstrate that our model achieves the best average image reconstruction performance across both subsets of the benchmark. Notably, it achieves outstanding performance on the L2 distance metric, which emphasizes pixel-level consistency, surpassing the second-best model by 46% on the COCO subset and 28% on the JourneyDB subset.

Generation with Text-Image Interleaved Control After Stage 2 training, DREAM ENGINE acquires the capability to incorporate text-image interleaved control during the image generation process. We showcase several applications of the model in this paper and compare its performance with Emu-2 [42], the most relevant baseline that also supports text-image interleaved control.

1. Natural Object Background Merging: Figure 4 illustrates an example application where objects are merged into different backgrounds based on a provided hint image. The results demonstrate that DREAM ENGINEcan seamlessly place the main object into various backgrounds in a more natural manner even when there are multiple objects, rather than merely copying and pasting the objects.
2. Object Driven Feature Mixing: Figure 6 demonstrates an emergent ability of DREAM ENGINE to generate images by combining visual features from given images based on text instructions-an area where the EMU-2 model fails to follow instructions accurately. Notably, these examples are not present in the training dataset, which was constructed directly from an object detection dataset. This highlights the significant potential of LMMs as unified multimodal instruction encoders for image generation. DREAM ENGINE effectively decouples complex elements from different images using simple text prompts and produces a unified representation, showcasing its versatility and robustness.
3. Free Form Image Editing: Figure 7 presents the results of our free-form image editing task. DREAM ENGINE consistently demonstrates superior ability to follow edit instructions compared to EMU-2. Notably, DREAM ENGINE can handle complex editing instruc-

## Source Image

Source Image

<!-- image -->

ℎ ! = 1 - 𝑟 ⋅ ℎ ""# +𝑟 ⋅ ℎ $%&amp; 𝒓 : Blending Ratio

## Image Alignment Results (Random Sample 4 Images)

<!-- image -->

Blending Ratio = 0

<!-- image -->

Figure 9. The ablation on visual blending ratio in the ViT module. during image reconstruction tasks.

tions, such as simultaneously modifying both the object and background. This further highlights the effectiveness of LMMs in providing a unified representation space that seamlessly integrates image and text conditions.

## 3.4. Discussions

## Understand the training dynamics of DREAM ENGINE

To better understand how DREAM ENGINE leverages an LMM and a text-to-image model to achieve complex textimage instruction following ability during the training process, we examine the image reconstruction results at different training stages in Stage-1. The results are shown in Figure 8. We observe a clear concept-to-detail progression during the training of DREAM ENGINE. For example, as illustrated in the first two columns of Figure 8, the model initially reconstructs the primary concepts in the images, such as 'girl,' 'house with snow,' and 'dog' in the given examples.

In the later stages, the model begins to learn to reconstruct more fine-grained details, such as colors, shapes, and poses. We believe this unique training dynamic stems from the nature of the LMM, which provides a unified representation space where images and text are well aligned. Thus, at the beginning of training, even if the text-to-image diffusion model has not yet seen image conditions, the image representations provided by the LMM are aligned with text features. This alignment enables the model to generate conceptually aligned images through the bridge of text.

Ablation on Balancing Visual Consistency We conduct an ablation study on the Blending Visual Feature mechanism within the Vision Transformer (ViT) module. As shown in Figure 9, varying the blending ratio in the imageimage alignment task produces notably different results. Higher blending ratios lead to greater consistency in im- age reconstruction, while lower ratios introduce more variation in the output. This mechanism offers flexible control over object consistency, benefiting various downstream tasks such as image editing and object-driven feature mixing.

Blending Ratio = 0.5

<!-- image -->

Blending Ratio = 1

It reveals that a higher blending ratio results in greater consistency

## 4. Related Work

## 4.1. Image Generation with Complex Control

Recent progress in controlled image generation using diffusion models has been significant. Researchers have explored various conditioning strategies-ranging from low-level cues like canny edges and depth maps [52, 54] to higher-level guidance provided by reference images [30]-to steer the generative process. For instance, methods such as IP-Adapter [52] and ControlNet [54] incorporate additional control signals into standard text-toimage frameworks, thereby allowing more precise manipulation of generated content. In parallel, several works have leveraged visual elements from input images to further guide the generation process. DreamBooth [38] and Textual Inversion [14], for example, adopt optimization-based approaches to adapt models to specific reference images. Although effective, these methods typically require extensive fine-tuning for each new input, limiting their practicality. To address these limitations, approaches like SuTI [7] and Subject-diffusion [29] have aimed to scale the fine-tuning process so that models can generalize across diverse reference images. However, these strategies still tend to be both time- and resource-intensive, highlighting the ongoing need for more efficient mechanisms for image generation with complex controls.

## 4.2. Connecting LMMs with Diffusion Models

Recent studies integrate LMMs with diffusion generators, leveraging the strengths of both paradigms. One straightforward approach employs LMMs to interpret complex text-image conditions and generate pure textual representations, which then guide image generation models [24]. Moreover, Seed-Tokenizer [15] expands the LMM vocabulary by introducing discrete vision tokens that serve as robust conditioning signals for diffusion models, while SeedLLama [16] pre-trains a discrete image tokenizer that decodes visual codes into realistic images using pretrained diffusion models. Similarly, M-VADER [47] aligns semantic consistency between language models and diffusion decoders through training on extensive image-text pair datasets. Methods such as GILL [22], MiniGPT5 [58] Emu [41] further advance this integration by mapping the embedding spaces of language models to diffusion models, and NExT-GPT [49] and Any-GPT [53] even broadens the scope to include modalities like audio and video. Additionally, DreamLLM [11] employs a novel strategy by transferring differential gradients from image diffusion models to language models, thereby enabling free-form, interleaved content generation. To enhance flexible control in image generation, BLIP-Diffusion [23] leverages LMMs to jointly encode image and text inputs, projecting them into the text conditioning space of diffusion models to better handle complex instructions. Moreover, Kosmos-g [31] and Emu-2 [42] explore the multimodal in-context control for image generation. Seed-X [17] using a similar architecture to model multi-granularity visual semantics for better generation in the real world applications. While promising, these approaches mainly extend text-to-image generation and often fall short in improving image quality or managing compositional tasks with arbitrary text-image interleaved control, limiting their real-world applicability.

## 5. Conclusion

In this work, we introduced DREAM ENGINE, a novel framework that enables sophisticated text-image interleaved control without complex architectural modifications. Our method bridges Large Multimodal Models and text-toimage diffusion models through a lightweight projector layer and efficient two-stage training paradigm. It demonstrates superior capabilities in handling multiple image inputs and compositional instructions while also achieving competitive performance on the GenEval benchmark (0.69). The success of DREAM ENGINE demonstrates that LMMs can effectively replace traditional text encoders in text-toimage diffusion models while expanding their capabilities to include sophisticated multimodal control.

Looking ahead, our work opens up new possibilities for creative image generation applications where users can pre- cisely orchestrate visual compositions through natural language instructions and multiple reference images. Future research directions could explore extending this framework to other modalities, such as video or 3D content, and investigating ways to further enhance the semantic understanding of complex multimodal instructions.

## References

- [1] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. ArXiv preprint , abs/2308.12966, 2023. 2, 3, 5
- [2] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science , 2023. 7
- [3] BlackForestLabs. Announcing black forest labs, 2024. 2, 3, 6, 7
- [4] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pretraining to recognize long-tail visual concepts. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021 , pages 3558-3568, 2021. 6, 7
- [5] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. PixArt-alpha: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426 , 2023. 7
- [6] Liang Chen, Zekun Wang, Shuhuai Ren, Lei Li, Haozhe Zhao, Yunshui Li, Zefan Cai, Hongcheng Guo, Lei Zhang, Yizhe Xiong, Yichi Zhang, Ruoyu Wu, Qingxiu Dong, Ge Zhang, Jian Yang, Lingwei Meng, Shujie Hu, Yulong Chen, Junyang Lin, Shuai Bai, Andreas Vlachos, Xu Tan, Minjia Zhang, Wen Xiao, Aaron Yee, Tianyu Liu, and Baobao Chang. Next token prediction towards multimodal intelligence: A comprehensive survey, 2024. 2
- [7] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and William W. Cohen. Subject-driven text-to-image generation via apprenticeship learning, 2023. 10
- [8] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, Ji Ma, Jiaqi Wang, Xiaoyi Dong, Hang Yan, Hewei Guo, Conghui He, Botian Shi, Zhenjiang Jin, Chao Xu, Bin Wang, Xingjian Wei, Wei Li, Wenjian Zhang, Bo Zhang, Pinlong Cai, Licheng Wen, Xiangchao Yan, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. How far are we to gpt-4v? closing the gap to commercial multimodal models with opensource suites, 2024. 5
- [9] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks, 2024. 3

- [10] DeepFloyd. DeepFloyd IF, 2023. 7
- [11] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, and Li Yi. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv: 2309.11499 , 2023. 11
- [12] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale, 2021. 3
- [13] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨ uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. 2024. 2, 3, 5, 6, 7
- [14] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion, 2022. 10
- [15] Yuying Ge, Yixiao Ge, Ziyun Zeng, Xintao Wang, and Ying Shan. Planting a seed of vision in large language model. arXiv preprint arXiv:2307.08041 , 2023. 2, 9, 11
- [16] Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, and Ying Shan. Making llama see and draw with seed tokenizer. arXiv preprint arXiv:2310.01218 , 2023. 11
- [17] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. SEED-X: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396 , 2024. 9, 11
- [18] Dhruba Ghosh, Hanna Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment, 2023. 7
- [19] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS , 2020. 2
- [20] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022 , 2022. 7
- [21] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models, 2022. 5
- [22] Jing Yu Koh, Daniel Fried, and Ruslan Salakhutdinov. Generating images with multimodal language models. NeurIPS , 2023. 11
- [23] Dongxu Li, Junnan Li, and Steven C. H. Hoi. Blip-diffusion: Pre-trained subject representation for controllable text-toimage generation and editing, 2023. 2, 11
- [24] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models, 2024. 11
- [25] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision-ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13 , pages 740-755. Springer, 2014. 6
- [26] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. ArXiv preprint , abs/2304.08485, 2023. 2, 3, 5
- [27] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. arXiv preprint arXiv:2402.08268 , 2024. 7
- [28] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow, 2022. 5
- [29] Jian Ma, Junhao Liang, Chen Chen, and Haonan Lu. Subjectdiffusion:open domain personalized text-to-image generation without test-time fine-tuning, 2024. 10
- [30] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations, 2022. 10
- [31] Xichen Pan, Li Dong, Shaohan Huang, Zhiliang Peng, Wenhu Chen, and Furu Wei. Kosmos-g: Generating images in context with multimodal large language models, 2024. 11
- [32] William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. 5
- [33] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨ uller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In ICLR , 2024. 7
- [34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event , pages 87488763, 2021. 5, 9
- [35] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with CLIP latents. arXiv preprint arXiv:2204.06125 , 2022. 7
- [36] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨ orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR , 2022. 7
- [37] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨ orn Ommer. High-resolution image synthesis with latent diffusion models, 2022. 2, 5
- [38] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation, 2023. 10
- [39] Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, et al. JourneyDB: A benchmark for generative image understanding. In NeurIPS , 2024. 6, 7

- [40] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: LLaMA for scalable image generation. arXiv preprint arXiv:2406.06525 , 2024. 7
- [41] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality, 2023. 2, 11
- [42] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 14398-14409, 2024. 2, 9, 11
- [43] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818 , 2024. 7
- [44] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, Austin Wang, Rob Fergus, Yann LeCun, and Saining Xie. Cambrian-1: A fully open, vision-centric exploration of multimodal llms, 2024. 5
- [45] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution, 2024. 2, 3, 5, 7
- [46] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869 , 2024. 7
- [47] Samuel Weinbach, Marco Bellagente, Constantin Eichenberg, Andrew Dai, Robert Baldock, Souradeep Nanda, Bj¨ orn Deiseroth, Koen Oostermeijer, Hannah Teufel, and Andres Felipe Cruz-Salinas. M-vader: A model for diffusion with multimodal context, 2022. 11
- [48] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848 , 2024. 7
- [49] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv: 2309.05519 , 2023. 11
- [50] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528 , 2024. 7
- [51] Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. mt5: A massively multilingual pre-trained text-totext transformer. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pages 483-498, 2021. 5
- [52] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ipadapter: Text compatible image prompt adapter for text-toimage diffusion models. 2023. 2, 10
- [53] Jun Zhan, Junqi Dai, Jiasheng Ye, Yunhua Zhou, Dong Zhang, Zhigeng Liu, Xin Zhang, Ruibin Yuan, Ge Zhang, Linyang Li, Hang Yan, Jie Fu, Tao Gui, Tianxiang Sun, Yugang Jiang, and Xipeng Qiu. Anygpt: Unified multimodal llm with discrete sequence modeling. ArXiv , abs/2402.12226, 2024. 11
- [54] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. 2, 10
- [55] Haozhe Zhao, Zefan Cai, Shuzheng Si, Xiaojian Ma, Kaikai An, Liang Chen, Zixuan Liu, Sheng Wang, Wenjuan Han, and Baobao Chang. Mmicl: Empowering vision-language model with multi-modal in-context learning. ArXiv preprint , abs/2309.07915, 2023. 3
- [56] Haozhe Zhao, Xiaojian Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale, 2024. 6, 7
- [57] Shihao Zhao, Shaozhe Hao, Bojia Zi, Huaizhe Xu, and Kwan-Yee K. Wong. Bridging different language models and generative vision models for text-to-image generation. ECCV , 2024. 2
- [58] Kaizhi Zheng, Xuehai He, and Xin Eric Wang. Minigpt5: Interleaved vision-and-language generation via generative vokens, 2023. 11