## 'This is my unicorn, Fluffy': Personalizing frozen vision-language representations

Niv Cohen 1 , 2 , Rinon Gal 1 , 3 , Eli A. Meirom 1 , Gal Chechik 1 , 4 , and Yuval Atzmon 1

1 NVIDIA Research, Israel

2 The Hebrew University of Jerusalem, Jerusalem, Israel

3 Tel Aviv University, Israel

- 4 Bar Ilan University, Israel

Abstract. Large Vision &amp; Language models pretrained on web-scale data provide representations that are invaluable for numerous V&amp;L problems. However, it is unclear how they can be extended to reason about user-specific visual concepts in unstructured language. This problem arises in multiple domains, from personalized image retrieval to personalized interaction with smart devices. We introduce a new learning setup called Personalized Vision &amp; Language (PerVL) with two new benchmark datasets for retrieving and segmenting user-specific ('personalized') concepts 'in the wild'. In PerVL, one should learn personalized concepts (1) independently of the downstream task (2) allowing a pretrained model to reason about them with free language, and (3) without providing personalized negative examples. We propose an architecture for solving PerVL that operates by expanding the input vocabulary of a pretrained model with new word embeddings for the personalized concepts. The model can then simply employ them as part of a sentence. We demonstrate that our approach learns personalized visual concepts from a few examples and effectively applies them in image retrieval and semantic segmentation using rich textual queries. For example the model improves MRR by 51.1% (28.4% vs 18.8%) compared to the strongest baseline.

The code and benchmark are available on github under NVlabs/PALAVRA and NVlabs/PerVLBenchmark.

## 1 Introduction

Large Vision &amp; Language (V&amp;L) models pre-trained on web-scale data made a breakthrough in computer vision [58, 83, 9]. These models provide a multimodal vision-language representation, and are used in a multitude of downstream tasks, from image captioning [54] and video retrieval [23], through image generation [26, 56] and segmentation [84, 42], to robotic manipulation [65]. All these tasks benefit from the 'open-world' capabilities of large V&amp;L models, enabling the use of rich, free-form text with a long 'tail' vocabulary of visual categories.

However, even with these powerful representations, an important challenge remains: How can these models be leveraged to reason about user-specific , ' personalized ' object instances in open-world vision problems? For example, we may wish to find an image that portrays us wearing a specific sweater, ask a robot assistant to make us coffee in our 'best-mom mug', or synthesize an image of our child's treasured toy Fluffy in an entirely new context.

Fig. 1. The Personalized Vision &amp; Language (PerVL) learning setup . Left: A user provides a few image examples of their personalized visual concepts: a favorite skirt (top), or a toddler's toy wagon (bottom). Examples are provided independently of the downstream tasks. (Right) the personalized model can be used in various downstream tasks. Top-right: An image retrieval task: given a textual query and a collection of images, rank and retrieve the queried image. (Bottom-right) Open-world semantic segmentation task. Segment a personalized object referred by a textual query. This example illustrates multiple ambiguities. First, there are two wagons that carry an elephant. Second, there are two wagons that correspond to the personalized concept. Resolving the ambiguity requires reasoning in both the visual and text modalities.

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

<!-- image -->

Clearly, pretrained V&amp;L models cannot be used directly to reason about new personal items. Luckily, it is typically easy for a user to collect a few image examples for a personalized concept. It then remains to develop methods that extend the pretrained models to new concepts using these examples. One challenge is that typically it is easy for people to provide positive image examples for a concept, but harder to provide consistent negative distractor examples [33, 64].

Learning from a few examples is considered a hallmark of intelligence. When people learn novel concepts from a few examples [50, 10, 40, 51, 52], they can seamlessly employ them in their semantic mental state and reason jointly both over the personalized concepts and over a large body of prior knowledge. Could a computational approach learn in a similar way using a pretrained V&amp;L model?

Previous efforts [67, 86, 27, 89] focused on learning a transformation module on top of CLIP's output space. However, as we explain below, these approaches risk forgetting prior knowledge, or face difficulties in accessing it concurrently with newly learned concepts. In addition, these previous approaches take a multiclass approach, discriminating between several new concepts. They are not designed for learning a single new personalized concept, which is natural in the context of personalization. Therefore, it is unknown how to learn a single personalized concept from few image examples in a way that (1) allows the pretrained model to reason about the concept with free language, and (2) uses only 'positive' image examples of the target concept.

Here, we address the question of personalizeing a pretrained model using few samples while maintaining its performance on the original vocabulary. We study a new representation learning setup, which we call ' Personalized Vision &amp; Language ' (PerVL) (Fig. 1). In PerVL, we are given a pretrained V&amp;L model, one or more personalized visual concepts, a few training images of each concept, and a string describing the concept type, like 'a mug' or 'a short sleeve top'. The goal is to learn a representation that can later be used to solve a set of downstream V&amp;L tasks involving the personalized concept. No further supervision is given for these downstream tasks. PerVL arises in various scenarios. In image retrieval, a user may tag a few of their images and wish to retrieve other photos of that concept in a visual specific context [13, 5]; in human-robot interaction, a worker may show a specific tool to a robotic arm, and instruct how to use it [65, 77, 48]; in video applications, an operator may search for a specific known item in the context of other items or people doing activities that are described with language.

Unlike previous efforts, instead of modifying a V&amp;L model output , we propose a framework for expanding its input vocabulary. Specifically, we learn new word embeddings for the new personalized concepts by intervening with the model input space. The concept of 'my best-mom mug' would be associated with a new symbol [MY BEST-MOM MUG] that has its own dense word embedding. The model could later represent sentences that use it, like ' Sipping tea from my best-mom mug on a porch ' by detecting 'my best mom mug' and mapping its symbol [MY BEST-MOM MUG] to its new embedding vector. Such tokens trivially preserve the structure of the original model, since the encoder model itself remains unmodified. Moreover, as we show below, the new concepts can also be easily integrated into existing downstream V&amp;L tasks. In summary, we address the question of using a small number of samples to personalize a pretrained V&amp;L model, while maintaining its performance in the original vocabulary.

This paper makes the following novel contributions: (1) A new representation learning setup, PerVL , for personalizing V&amp;L representations, while keeping their 'zero-shot' reasoning capabilities. (2) Two new benchmark datasets for PerVL . (3) A novel approach, PALAVRA 7 , to expand and personalize the vocabulary of the V&amp;L representation inputs . PALAVRA uses a cycle-consistent loss, learned with positive image examples only. (4) A technique for using a textual encoder to improve the generalization of a network to new visual concepts.

## 2 Related work

The success of CLIP led to diverse work that leverage its powerful representation for few-shot learning. Most works [67, 86, 27, 49] are based on learning a residual

7 Palavra means 'word' in Portuguese, as we learn new word-embeddings. For acronym lovers, PALAVRA also stands for 'Personalizing LAnguage Vision RepresentAtions'

'adapter' layer [32] over the output of CLIP encoders. Taking a different approach, [89] proposes learning a soft prefix to improve accuracy in a classification task. Our work differs from these approaches in two key aspects: (1) They focus solely on classifying images using a narrow vocabulary. In contrast, our setup learns a representation which is then used in any downstream tasks. Moreover, our method expands CLIP's vocabulary rather than narrowing it. (2) Adapterbased methods override the output representation of the encoders, leading to a change in their input → output mappings. Our method does not change the pretrained mapping but enriches its input vocabulary with new concepts.

Recently, [78, 38, 30] have shown that fine-tuning can actively harm out-ofdistribution generalization, even when tested on the same downstream task for which the model was tuned. Our method does not fine-tune the pretrained model and does not leverage in-distribution labeled examples for the downstream tasks.

Other approaches [73, 31] study 'fast' concept learning combined with 'slowlearned' concepts, showing that the new concepts can be applied to 'slowlylearned' downstream tasks. However, the 'fast' learned concepts are stored implicitly in the network activations, rather than grounded in the vocabulary.

A related set of works can be found in the image captioning and generator inversion tasks. While their goals are different, these works nonetheless aim to extract meaningful semantic information from images and map them to tokens that represent the concepts - in this case, words or latent codes. Of these, a series of works [16, 20, 24, 18, 34, 46, 66] focus on personalizing image captions according to a user writing style. Alternatively, [29, 75, 80, 88, 47, 19] extend image captions with novel concepts using 'slot filling', which are placeholders for nouns that are filled using object detector predictions. In inversion, models typically aim to identify codes in the latent spaces of pre-trained generators, which represent specific images or identities [1, 61]. These can then be used in downstream tasks such as editing [63, 2] or super resolution [53]. In some cases, the model is further fine-tuned to better represent a specific instance [62, 4, 21, 74], or to allow more faithful replication of personalized traits such as expressions [55].

Our model differs from zero and few-shot learning (FSL) based on metalearning [25, 76, 70, 68, 14, 41, 3, 81, 7, 8, 57] or incremental learning [71, 22, 59, 15, 36, 79] in three aspects. First, we impose stronger generalization requirements. Our model can reason about new concepts in diverse downstream tasks, which may be unknown at training time. Second, in common FSL, the concept distribution used during meta-learning ('support set') is also used during the FSL stage. For example, meta-learn on birds, then do FSL with new types of birds. While our technique for training with text allows to generalize beyond the domain of concepts in the training images. Third, our approach improves upon CLIP's zero-shot perceptual capabilities, and is compatible with many CLIP-based downstream tasks.

Finally, in existing FSL benchmarks [76, 60, 87] there is no instance level annotations, and there is only a single task. As a result, existing FSL benchmarks do not directly fit our setting. Our work addresses rich text query of a specific instance , that can be used in a flexible way with many downstream tasks.

## 3 A new setup, Personalized Vision &amp; Language

We propose ' Personalized Vision &amp; Language ' (PerVL), a new representation learning setup, to personalize a pretrained model with few positive image examples, without supervision for the downstream task.

In PerVL , we are given a pretrained model h ( S V , I ) that accepts a sentence S and an image I . The sentences that the model accepts are defined in a vocabulary V . We wish to update h so that it can accept sentences from an expanded vocabulary V ′ = V ∪ C where C is a new set of concepts C = { c 1 , ....c k } , which results in an extended model h ′ ( S V ′ , I ). In general, we expect that adapting the model would not strongly affect the original vocabulary, namely h ′ ( S V , I ) ≈ h ( S V , I ).

At training (personalization) time, we adapt the model given a small set of images { I i } N c i =1 for every concept c , without assuming access to negative training images . We are also provided with a string describing the type of the new concept, such as a 'mug' or a 'short sleeve top'. Stating the type is a natural way for non-expert users to provide prior knowledge about the personalized concept. The type can be used to guide learning to distinguish the personalized concept from the general concept type. Concepts describing coarser classes from a hierarchy of concepts (e.g. 'dog' for 'poodle') have been used for this purpose [17]. We denote the concept type by S c .

During inference, we are given a downstream V&amp;L task T that can be inferred using the pretrained model h for the vocabulary V , and we wish to solve it for an instance x that contains the new concept c . The instance may contain images and sentences pertaining to c .

Encoder PerVL : Here we focus on the special case of CLIP [58]. The model h applies a cosine similarity between a sentence S and an image I : h ( S, I ) = cos( h T ( S ) , h I ( I )), where h I and h T are CLIP image and text encoders.

## 4 Methods

Before describing our approach, we first explain the reasons for expanding the input vocabulary of the V&amp;L model and how it differs from previous approaches.

Several studies extend CLIP by learning an 'Adapter' module on top of the CLIP representation [27, 67, 86, 49, 32]. That module is applied to the output of a CLIP encoder network that is kept frozen. It is trained for a classification task with labeled data and a templated text query ('a photo of a [concept-type]').

We show below (Sec. 6 &amp; Appendix B) that this approach tends to be brittle and fails when its input sentences deviate from the template used for training. This is probably because the adapter overrides the output representation of the encoder, so training it with very few examples hurts its generalization power.

Conversely, our approach does not overrides the encoder outputs. Our working hypothesis is that the text input space of a web-scale V&amp;L model is rich enough for reasoning about new personalized concepts. We just need to find the right word embedding representation for any new personalized concept. We illustrate this architectural distinction in Fig. 2.

Fig. 2. Visualization of an adapter-based approach (left) and PALAVRA (right). Adapters change CLIP's output space by appending additional layers following the encoder. Our method defines new tokens in CLIP's existing input space, leaving the output space unchanged.

<!-- image -->

Finally, one could fully retrain a CLIP model with the expanded vocabulary set. However, retraining CLIP requires ∼ 400 M images. Our approach is trained with a tiny fraction of that, &lt; 1 M samples, and once it is trained, different users can use it, each with their own vocabulary.

Notation: For brevity, we describe adding a single concept c . Adding multiple concepts can be done iteratively. We use the notation [CONCEPT] to refer to a learned concept ( c ) within a textual query. I denotes the CLIP embedded image space, T the CLIP embedded textual space, z k = h I ( I k ) is the embedding of an image I k into I , and similarly h T ( S ) is the embedding of a sentence S into T . Finally, W denotes the space used to embed input word tokens into CLIP.

Architecture and Workflow: At a high level, our workflow has three steps.

- (1) Learn an inversion mapping f θ from a set of points in CLIP image space I to a point in its word embedding input space W (Fig. 3). Formally, f θ : { z k ∈ I} K k =1 →W . It is trained with non-personalized, large-scale data.
- (2) Initial personalization (Fig. 4). Learn a word embedding w c of a new personalized concept c . Thus, given a set of image examples I 1 , ..., I K we map them to CLIP image space, then map them using f θ to obtain an initial word embedding w 0 c = f θ ( { h I ( I k ) } ). Formally, { I k } K k =1 →{ h I ( I k ) } K k =1 → w 0 c ∈ W .
- (3) Fine-tuning. The initial embedding w 0 c is then updated using gradient steps to maximize the similarity of the template text embeddings to the image examples, while contrasting it with an embedding of a 'super-concept'.

Next, we describe the learning of each component in more detail.

Training 𝑓 𝜃 with image examples

ҧ

ҧ

Training 𝑓 𝜃 with Text examples

ҧ

Fig. 3. Architecture outline: Learning f θ . We start with a large-scale-data training step. A set encoder f θ is trained to map CLIP-space output embeddings to a code in CLIP's input space. It is alternatingly trained with a batch of either image examples (left), or sentence examples (right) with augmented concept types. We use a cycle loss by mapping the code back to CLIP's output embedding, using a template sentence.

<!-- image -->

## 4.1 Learning the inversion mapping f θ

We now describe how we learn an 'inversion' map f θ from a set of points in CLIP space z 1 , ..., z k ∈ I , to a word embedding w 0 c ∈ W , where W is the input space of the language encoder. We base f θ 's architecture on 'Deep Sets' [85]. We now discuss the loss and how to train f θ with two types of large-scale, nonpersonalized data: images and text. See Fig. 3.

A contrastive cycle loss. f θ maps from CLIP space to w 0 c ∈ W . Then, by pairing w 0 c to the word embedding for [CONCEPT] we can feed w 0 c into h T with a template sentence T c like 'A photo of a [CONCEPT]'. We can then define a cycle consistency loss to match the input of f θ with the output of h T (see Fig. 3 left). Specifically, let ¯ z c be the average over samples in I from the concept

ҧ

<!-- image -->

Ƹ

c , ¯ z c = ∑ K k =1 z k /K and let ˆ z c be the CLIP embedding of a template sentence, ˆ z c = h T ( T c ). We wish to tune f θ so that ˆ z c is close to ¯ z c for the concept c and far from other concepts. We therefore define a symmetric contrastive loss for a concept c , with a formulation similar to SimCLR [12]:

̸

<!-- formula-not-decoded -->

̸

where cos(ˆ z, ¯ z ) denotes cosine similarity, C is the number of concepts in a batch.

We also use a regularization term ℓ GT that maximizes the similarity of the predicted w 0 c with its ground truth. See details in the Appendix. Finally, the cycle loss and ground-truth regularization terms are combined with a hyperparameter λ gt ≥ 0, and the total loss is ℓ total = ℓ Cycle + λ gt · ℓ GT .

Training f θ with images. Weuse a variant of COCO [44] that extracted the subject and object from each caption as in [6], and take the 1000 most frequent concepts. In every training batch, we draw at random C concepts, then draw K images for each concept. We then map them to CLIP image space, yielding { z k } K 1 = { h I ( I k ) } K 1 in CLIP image space for each concept.

Training f θ with text. When training with COCO data, f θ learns concepts that are frequent in COCO captions. However, our goal is to have f θ generalize to widely diverse concepts. Yet, naively training with the COCO images does not generalize well to out-of-COCO-vocabulary concepts (see 7).

To generalize to out-of-vocabulary concepts, we propose synthesizing textual descriptions with an expanded vocabulary and embed them into the shared embedding space. Specifically, we use COCO captions of a concept to generate additional training examples for new concepts by replacing the concept type with the most similar concept type from a large predefined vocabulary of 20K types [39], where (cosine) similarity is measured in CLIP text space. Finally, we embed the augmented captions by taking their CLIP-text feature representation (Fig. 3, right). Overall, we found that training with augmented text representation significantly improved the performance of the model (see Table 2).

As in [43], we observed that the encoding distribution of text and images does not overlap in CLIP space. As a result, training f θ with CLIP embeddings of captions does not generalize well to image inputs. We address this problem by learning an alignment matrix A that maps CLIP representations of texts to their presumed image counterpart (Fig. 3, right). A is learned jointly with f θ , and is only used when learning the personalization tokens. It is not used at inference time. Formally, a set of captions is first encoded by h T , then mapped to the image area of the CLIP space using A and then fed to f θ .

## 4.2 Personalization : Learn an embedding of personalized concepts

To learn the word embedding of a personalized concept, we follow a similar process to training f θ , but instead of tuning the parameters of f θ , we optimize the actual embedding vector w c .

Specifically, let { I k } N c k =1 be a set of input images for the new concept c , we (1) map them to I using CLIP encoder h I , (2) map to w 0 c using f θ , (3) plug the embedding w 0 c in a template sentence and (4) map to CLIP text space using h T . Once again, we define a contrastive cycle-consistent loss, matching the estimated text embedding of the template sentence ˆ z c , and average image embedding ¯ z c . However, here we contrast it with the embedding of the concept type , say 'a short sleeve top', denoted by η c = h T ( S c ). Since no negative image examples are provided in the personalization stage, the concept type can be viewed as a 'super-concept' in the hierarchy. It allows the learning process to focus on the specific features that make the object unique from the general population of similar concept types. Similar to the SimCLR [12] loss, our loss is:

<!-- formula-not-decoded -->

The factor 2 results from contrasting η c with both visual and a text embedding.

## 4.3 Inference

Our approach expands the vocabulary of word-embedding tokens with personalized tokens, without modifying the underlying V&amp;L model h . Therefore, for a given downstream task T and a sentence S , we use the pretrained V&amp;L model h as it would have been used with T . But when we encounter an input sentence S that includes a [CONCEPT] token, we apply its learned embedding w c . Also, we found that having [CONCEPT] followed by the concept type S c is beneficial.

## 5 Evaluation datasets for PerVL

We created two new personalization benchmark datasets for the evaluation of PerVL. (1) We collected captions for images from DeepFashion2 [28], which serve as search queries in an image retrieval task. (2) We collected captions for frames from Youtube-VOS [82], and also collected their corresponding segmentation maps for a referring-expression segmentation task.

## 5.1 Personalized fashion-item retrieval with DeepFashion2

We used the DeepFashion2 dataset [28] to create an image retrieval benchmark of personalized fashion items given a textual query. It contains photos of people wearing unique fashion items from 13 popular clothing categories, like skirt or long-sleeve dress , which we use as concept types. See the examples in Fig. 1, 5.

a)

<!-- image -->

[CONCEPT] is closest to the middle of the wire fence b)

<!-- image -->

[CONCEPT] is leaning on a rock

[CONCEPT] with a white nose patch.

<!-- image -->

[CONCEPT] lays on the ground between two dark square objects

<!-- image -->

a gazebo roof is behind the [CONCEPT]

<!-- image -->

Fig. 5. Examples of textual queries and evaluation images of the new benchmarks, [CONCEPT] denotes the personalized concepts. (a) YTVOS frames. The queried concept is highlighted in a yellow box. In YTVOS segmentation, one should segment the correct concept in the frame. In YTVOS retrieval, each evaluation image is a box cropped around the concept. (b) Deepfashion2 examples.

We created a dataset of 1700 images from 100 unique fashion items (concepts). Each item was assigned a unique [CONCEPT] tag. We assigned 450 images (out of the above 1700) to an evaluation set , and used raters to collect a textual description referring to each item appearing in the images. For instance, The [CONCEPT] is facing a glass store display.' . In Appendix E.1 we describe the steps we took to select context-rich items.

Short versus detailed captions: We collected two types of captions for each image. First, detailed captions like ' White cabinets, some with open drawers, are alongside and behind the [CONCEPT]. '. These describe extensive context about the image and can facilitate retrieval. Second are short captions like ' White cabinets are behind the [CONCEPT]. '. These pose a greater challenge, because they describe less detail, and therefore are more ambiguous.

Finally, we randomly split the data to 50 val. concepts and 50 test concepts.

## 5.2 Youtube-VOS for personalized retrieval and segmentation.

We created an image segmentation benchmark of personalized visual concepts given a textual query using Youtube-VOS (YTVOS) [82]. YTVOS is a dataset for instance segmentation in video, which includes 4000+ videos, 90+ categories, and 7800+ unique object instances. To transform it to an image personalization benchmark, we take the last frame of each video (scene) for evaluation and the object instances that appear in the frame as the target concepts. Earlier frames are used as candidate frames for training. See the examples in Fig. 5 (left).

This benchmark is challenging as it contains ambiguities about both the textual queries and the appearance of the personalized concept. Hence, only a model that is successful in both personalization and image-text reasoning can succeed in this task. For that, we only select videos such that their object concept appears at least twice in an evaluation frame.

Fig. 6. Recall at K for our approach and baselines. DeepFashion2 (left), YoutubeVOS (right), PALAVRA achieves the highest rates for all retrieval metrics. On both benchmarks and metrics it achieves a significant improvement compared with 'AvgIm&amp;Text', which is the strongest baseline. The experiments with 'Concept-only query' demonstrate that the information in the textual query is essential for telling the target image from distractors, since the performance of both PALAVRA and 'Text Only' substantially degrades with 'Concept-only' queries.

<!-- image -->

Finally, we annotated the instances in the evaluation frame with captions using AMT. We instructed the AMT workers to concisely describe what makes a specific entity distinct, compared to similar entities in the image. We provide more details and examples in Appendix E.3.

In total, this benchmark includes ∼ 500 unique personalized concepts, with ∼ 6300 training samples. For evaluation, we split according to unique scenes (videos), resulting in 246 validation concepts and 251 test concepts.

Personalized image retrieval: We also created an image retrieval variant of YTVOS. We extracted a set of images that correspond to the AMT captions collected for segmentation. Every image in the retrieval set was extracted from a wide box cropped around every instance in each evaluation frame. The goal is to retrieve the image of the correct instance given its textual query. Compared to the segmentation task, there are fewer distractors from the same scene for every instance, since not all instances were labeled in the data, but there are many more distractors coming from different scenes.

## 6 Experiments

We tested PALAVRA with two PerVL benchmarks and compared with several leading baselines (Sec.6.1,6.2). We then study in greater depth the properties of PALAVRA, by an ablation study (see 7). All design decisions and hyperparameter tuning were performed on the validation set to avoid overfitting the test set. The experiments were carried out on NVIDIA V100 and A100 GPU. We provide additional implementation details and results in Appendix A,B.

## 6.1 Personalized image retrieval with a textual query

The objective of this task is to retrieve the correct image given a text query that includes the new concept (Fig. 1, top-right). We use AMT captions as textual queries describing a single image from the dataset. The challenge in this setup is to overcome two types of distractors. (a) visually similar distractors: images of the same personalized concept but in a different context than the context described by the textual query (e.g. the two instances of 'my favorite skirt' in Fig. 1), (b) semantically similar distractors: images which include an item of a similar concept type (e.g. 'a skirt'), in a similar context as described in the textual query.

Table 1. MRR retrieval metrics.

|                     | DeepFashion2 MRR   | YTVOS MRR   |
|---------------------|--------------------|-------------|
| Random              | 2.9 ± 0.4          | 2.8 ± 0.2   |
| Concept-only query  |                    |             |
| Text Only           | 4.2 ± 0.0          | 21.5 ± 0.0  |
| Adapter             | 13.4 ± 0.5         | 35.5 ± 0.3  |
| COLLIE              | 13.8 ± 0.5         | 35.6 ± 0.3  |
| AvgIm               | 13.8 ± 0.5         | 38.2 ± 0.3  |
| PALAVRA (Ours)      | 19.4 ± 0.6         | 53.4 ± 0.8  |
| Rich query          |                    |             |
| Adapter             | 5.9 ± 0.7          | 5.3 ± 0.3   |
| COLLIE              | 7.9 ± 0.7          | 6.2 ± 0.3   |
| COLLIE: Text        | 8.0 ± 1.0          | 7.2 ± 0.3   |
| Text Only           | 17.6 ± 0.0         | 37.6 ± 0.0  |
| AvgIm+Text          | 18.8 ± 0.4         | 47.2 ± 0.3  |
| PALAVRA w.o. tuning | 22.1 ± 0.2         | 47.1 ± 0.8  |
| PALAVRA (Ours)      | 28.4 ± 0.7         | 61.2 ± 0.4  |

To rank images according to a textual query, we rank images according to their cosine similarity with the embedded text query.

Compared methods. We compare our approach PALAVRA , with 5 baselines and their variants: Text Only : score an image-query pair using CLIP embedding of the text query h T ( S ). Using the concept type for [CONCEPT] instead of its learned word embedding. AvgIM : Ignoring the text query and replace it by the average over the embedding of its concept training images. This is equivalent to the FSL baseline in [14]. IM&amp;Text : Represent the query as the average between AvgIM and Text . Random : Test images are ranked in random order.

COLLIE [67]: Learn an adapter module over the output of CLIP text encoder, with an additional scaler function Scaler ( h T ( S ) ) ∈ [0 , 1] that softly applies the adapter layer. COLLIE is closest to our method, because it may preserve some capabilities of the underlying pretrained model, when Scaler ( · ) = 0. Adapter : As in COLLIE, but replace the scaler with a constant value of 1, making the 'Adapter' layer always active. COLLIE:Text : COLLIE, when the text query uses the concept type for [CONCEPT], rather than the trained concept.

Evaluation metrics and queries. For image retrieval, we report two metrics (1) Recall at K : The rate of successful retrievals among the top-K scoring images. (2) MRR (Mean Reciprocal Rank): Average of 1 divided by the rank of the correct image. Errors denote the standard error of the mean (SEM) across 5 model initialization seeds. We use two types of queries: (1) Rich query uses the free-formed text annotated by AMT workers: '[CONCEPT] is leaning on a rock'. (2) Concept-only queries overrides the 'Rich query' by a template that focuses only on the concept: 'A photo of a [CONCEPT]'. Note that the baseline 'AvgIM' is more related to the 'Concept-only query' because the rich query embedding is overridden by the average embedding of the training examples.

Retrieval Results. Table 1 and Fig. 6 describe the retrieval rates of PALAVRA and the compared methods when using challenging short captions as our Rich Queries . We report the retrieval rates with detailed captions in the appendix. We note that both short queries and detailed queries are rich queries, containing known concepts in addition to the personalized ones. The detailed version possibly contains more of them. PALAVRA achieves the highest rate in all the retrieval metrics. On both benchmarks and metrics, it achieves significant improvement (between ∼ 30% to ∼ 50%) compared with 'AvgIm+Text', which is the strongest baseline.

Comparing the results of 'Concept-only query' with the 'Rich query' results demonstrate that: (1) Information in the 'Rich query' is essential for retrieval. (2) Adapter baselines (Adapter &amp; COLLIE) improve over vanilla CLIP when only the concept is queried.

Their performance degrades when using the 'Rich query'. This happens as the adaptation layer trained for the personalized [CONCEPT] does not perform well with free-form text it has not seen during training. In fact, we find that Adapter and COLLIE are even sensitive to the prompt prefix of the query. When changing their prefix to a prefix not used in training, their performance substantially degrades. We report this finding quantitatively in Appendix B.

<!-- image -->

IoU Threshold

Fig. 7. Segmentation results on YTVOS. (a) Percent of images where IoU values exceeded a threshold, as a function of the threshold. Our approach dominates across the full range (b, c) Investigating model robustness under 2 levels of task complexity. We consider two scenarios that influence difficulty: Object size (b), and intra-class class ambiguity (c). When a clear visual signal is available (large objects, high intra-class visual variance), our model significantly outperforms the alternatives. Even in more challenging scenarios, our model still leverages textual descriptions, mitigating the loss of quality seen in models that ignore or corrupt CLIP's embedding space.

## 6.2 Semantic segmentation with a textual query

The second downstream task, aims to segment an instance of a personalized concept in an image, based on a textual query that refers to the concept (Fig. 1, right bottom). The challenge here is to overcome two types of distractors: First, visual distractors that look similar to the concept and can be disambiguated with the text query. Second, semantic distractors that include a concept of a similar type (e.g. another type of a 'toy wagon'), but CLIP has difficulty to resolve using just the concept type, like 'an elephant on a toy wagon'.

Here we investigate the performance of PALAVRA and baseline models on YTVOS dataset using a recent CLIP-based semantic segmentation [84]. In brief, [84] creates a set of query-driven relevance maps for the image, coupled with transformer interpretability methods [11]. The maps then serve as pseudolabels for single-image semantic segmentation [69].

Compared methods. We compare PALAVRA with a set of baselines in two setups: 'Rich query' and 'Concept-only query', as described in Sec. 6.1. (1) Text (CLIP) , using both the rich and concept only queries, (2) AvgIM , (3) IM&amp;Text and (4) COLLIE . All baselines are described in Sec. 6.1.

Evaluation metrics. We calculate the intersection-over-union (IOU) between the predicted segment and the ground-truth segment. We report the Rate of IOU &gt; threshold , which is the fraction of segments with IOU &gt; threshold. Error bars denote the standard error of the mean (SEM) across 5 model seeds.

## Semantic segmentation results

Fig. 7a shows the percent of test-set images for which IoU exceeds a given threshold. Our model consistently outperforms the baselines with wide margins (e.g. a 44 . 69% improvement over the best competitor at an IoU threshold of 0 . 5). These results demonstrate that a personalized prompt can extend even to localized image tasks. Moreover, as our method only expands CLIP's input space - it can be readily integrated with existing models for downstream tasks.

Surprisingly, our method performs better when using the concept-only query. We hypothesize that this is a result of CLIP's difficulty in reasoning over complex referring expressions: By mentioning the context within the text, CLIP's attention is diverted to the context itself which leads to false negatives. In contrast, in retrieval, context objects rarely appear in 'negative' (non-target) images. Since they appear in the target image, they actually help to detect the correct image. Appendix D.2 provides quantitative evidence supporting this hypothesis.

We further examine cases where we expect existing methods to falter. In Fig. 7b, we examine a scenario where objects are small, so their crops may not provide sufficient signal to the CLIP image encoder. Indeed, when for objects below the median size, segmentation fares worse in general, and image basedmethods suffer even more. Our method, however, can rely on the signal derived from the text and degrades less. Fig. 7c examines a scenario in which objects in a scene are less visually distinct. We divide the evaluation set to object which are usually visually distinct and images which often contain visually ambiguous objects, where a few instances of the same object appear in the same image. In practice, we split to animal and non-animal categories, as animals are mostly visually ambiguous (e.g. Fig. 5 left). Our model substantially outperforms the baselines when the concepts are visually distinct and also improves when the concepts are mostly ambiguous.

In the Appendix B we provide and discuss qualitative segmentation results.

## 7 Ablation Study

To understand how various components of our approach contribute to its performance, we conducted an ablation study. We report validation and test metrics for DeepFashion2 and YTVOS.

We first ablate model components that affect training f θ . We report the results without fine-tuning, to reveal how they affect the training of the set encoder. We call this stage 'no tuning' .

Then we compare components that affect the fine-tuning stage. We call this stage 'with tuning' . Specifically, we compare the following components:

1. PALAVRA is our approach described in Sec. 4. We tested it both with tuning and no tuning .
2. no text augment shows the results of f θ trained only with visual concepts that exist in the MS-COCO vocabulary, and without using the extended vocabulary for augmentation.
3. Only ℓ GT does not use the cycle loss for training f θ (see Eq. 1).
4. Only ℓ Cycle does not use the ground-truth regularization term for training f θ (see Eq. 1).
5. Only tuning initializes w 0 c randomly, instead of using the prediction made by the set encoder f θ .
6. no alignment shows the performance of our method when replacing the alignment matrix A by an identity mapping.

Table 2 shows the results of the ablation experiments. Several points worth discussing. First, PALAVRA without tuning improves by 25% compared to 'Text Only' (in Table 1), for both DeepFashion2 (22 . 1 vs. 17 . 6) and YTVOS (47 . 1 vs. 37 . 6). This result indicates that f θ learns to predict the word embeddings of visual concepts, and these concepts are better than using their vanilla CLIP embeddings.

Next, we find that text augmentation with extended vocabulary (Sec. 4.1) improves concept learning with f θ . It yields an improvement of ∼ 16% for DeepFashion2 (22 . 1 vs. 19 . 1) and ∼ 6% for YTVOS (47 . 1 vs. 44 . 4).

Combining a cycle loss with the ground truth (GT) regularization term is effective. When combined with the GT regularization term, the cycle loss improves by ∼ 16% for DeepFashion2 (22 . 1 vs 19 . 2) ∼ 14% for YTVOS (47 . 1 vs 41 . 4). However, when the GT regularization term is deactivated and only the cycle loss is used, f θ fails to generalize (16 . 1 in DeepFashion2 and 37 . 3 in YTVOS). We hypothesize that this effect is similar to the effect observed with inversion to the latent space of GANs [72]. There, inversions into sparse regions of the latent

## 16

Table 2. Ablation study: Highlighting the importance of various components. See the text for a full description of each setting and an analysis of the results.

| DeepFashion2    | Validation   | Validation   | Test       | Test       |
|-----------------|--------------|--------------|------------|------------|
|                 | MRR          | Recall@5     | MRR        | Recall@5   |
| no tuning       |              |              |            |            |
| PALAVRA (Ours)  | 26.9 ± 0.2   | 35.9 ± 0.2   | 22.1 ± 0.2 | 29.6 ± 0.3 |
| no text augment | 21.8 ± 1.9   | 29.2 ± 2.3   | 19.1 ± 0.2 | 25.7 ± 0.3 |
| Only ℓ GT       | 23.3 ± 0.4   | 31.9 ± 0.5   | 19.2 ± 0.5 | 25.1 ± 0.8 |
| Only ℓ Cycle    | 19.3 ± 0.5   | 26.8 ± 0.8   | 16.1 ± 0.4 | 21.6 ± 0.8 |
| with tuning     |              |              |            |            |
| PALAVRA (Ours)  | 36.2 ± 1.3   | 53.7 ± 2.0   | 28.4 ± 0.7 | 39.2 ± 1.3 |
| Only tuning     | 32.1 ± 0.6   | 44.1 ± 0.7   | 27.5 ± 1.0 | 37.9 ± 1.8 |
| no alignment    | 32.9 ± 0.4   | 47.8 ± 1.1   | 26.3 ± 0.2 | 36.9 ± 1.6 |
| YTVOS           | Validation   |              | Test       |            |
|                 | MRR          | Recall@5     | MRR        | Recall@5   |
| no tuning       |              |              |            |            |
| PALAVRA (Ours)  | 47.3 ± 0.5   | 68.5 ± 0.5   | 47.1 ± 0.8 | 70.3 ± 0.8 |
| no text augment | 45.0 ± 0.3   | 63.8 ± 0.5   | 44.4 ± 0.3 | 65.6 ± 0.4 |
| Only ℓ GT       | 40.8 ± 0.8   | 59.3 ± 1.3   | 41.4 ± 0.2 | 62.0 ± 0.1 |
| Only ℓ Cycle    | 35.5 ± 1.1   | 50.8 ± 2.4   | 37.3 ± 1.1 | 55.8 ± 2.1 |
| with tuning     |              |              |            |            |
| PALAVRA (Ours)  | 59.0 ± 0.8   | 76.2 ± 1.1   | 61.2 ± 0.4 | 78.7 ± 0.4 |
| Only tuning     | 57.3 ± 0.9   | 76.1 ± 0.9   | 57.8 ± 0.3 | 77.1 ± 0.8 |
| no alignment    | 56.5 ± 0.7   | 74.1 ± 0.3   | 58.1 ± 0.3 | 75.2 ± 0.9 |

space can better satisfy a cyclic reconstruction loss, but they behave poorly under interpolation. Our f θ could similarly learn to invert into sparse regions of CLIP's input space. By adding the GT regularization term, our inversions are encouraged to reside in better-behaved regions of the input space, namely those observed during CLIP's training. In these regions, the semantics of the latent space hold better and the model can better generalize.

When f θ is replaced by a random initialization, the performance degrades by ∼ 6% for YTVOS (57 . 1 vs 61 . 2) and ∼ 3% for DeepFashion2 (27 . 5 vs 28 . 4). Showing the synergy between the two personalization steps.

Finally, integrating the alignment matrix A showed an improvement of ∼ 8% for DeepFashion2 (28 . 4 vs. 26 . 3) and ∼ 5% for YTVOS (61 . 2 vs. 58 . 1).

## 8 Discussion

We described an approach to leverage large pre-trained V&amp;L models like CLIP, for learning a representation of new 'personal' classes from a handful of samples. Our key idea is to expand the input space of V&amp;L models by finding a representation of the new concept. The extended model can then be used for V&amp;L tasks with a rich language that 'understands' both novel and known concepts. A limitation of the approach is that it suffers from the limitations of the underlying V&amp;L model. For instance, CLIP struggles with understanding spatial relations within a photo [45], and extended representations based on CLIP suffer from the same problem. We expect that our approach can be extended to other V&amp;L models. See an example in Appendix C.

To conclude, we hope that the method presented in this paper will pave the way to using pretrained models in problems that involve user-specific concepts, like home robotics and organizing personal data.

## References

1. Abdal, R., Qin, Y., Wonka, P.: Image2stylegan: How to embed images into the stylegan latent space? In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4432-4441 (2019)
2. Abdal, R., Zhu, P., Mitra, N.J., Wonka, P.: Styleflow: Attribute-conditioned exploration of stylegan-generated images using conditional continuous normalizing flows. ACM Transactions on Graphics (ToG) 40 (3), 1-21 (2021)
3. Akata, Z., Perronnin, F., Harchaoui, Z., Schmid, C.: Label-embedding for image classification. IEEE Transactions on Pattern Analysis and Machine Intelligence 38 , 1425-1438 (2016)
4. Alaluf, Y., Tov, O., Mokady, R., Gal, R., Bermano, A.: Hyperstyle: Stylegan inversion with hypernetworks for real image editing. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 18511-18521 (2022)
5. Anwaar, M.U., Labintcev, E., Kleinsteuber, M.: Compositional learning of imagetext query for image retrieval. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 1140-1149 (2021)
6. Atzmon, Y., Berant, J., Kezami, V., Globerson, A., Chechik, G.: Learning to generalize to new compositions in image understanding. arXiv preprint arXiv:1608.07639 (2016)
7. Atzmon, Y., Chechik, G.: Probabilistic and-or attribute grouping for zero-shot learning. In: Proceedings of the Thirty-Forth Conference on Uncertainty in Artificial Intelligence (2018)
8. Atzmon, Y., Chechik, G.: Adaptive confidence smoothing for generalized zero-shot learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11671-11680 (2019)
9. Bommasani, R., Hudson, D.A., Adeli, E., Altman, R., Arora, S., von Arx, S., Bernstein, M.S., Bohg, J., Bosselut, A., Brunskill, E., et al.: On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258 (2021)
10. Carey, S., Bartlett, E.: Acquiring a single new word. (1978)
11. Chefer, H., Gur, S., Wolf, L.: Transformer interpretability beyond attention visualization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 782-791 (June 2021)
12. Chen, T., Kornblith, S., Norouzi, M., Hinton, G.: A simple framework for contrastive learning of visual representations. In: International conference on machine learning. pp. 1597-1607. PMLR (2020)
13. Chen, Y., Gong, S., Bazzani, L.: Image search with text feedback by visiolinguistic attention learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3001-3011 (2020)
14. Chen, Y., Liu, Z., Xu, H., Darrell, T., Wang, X.: Meta-baseline: Exploring simple meta-learning for few-shot learning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9062-9071 (2021)
15. Cheraghian, A., Rahman, S., Fang, P., Roy, S.K., Petersson, L., Harandi, M.: Semantic-aware knowledge distillation for few-shot class-incremental learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 2534-2543 (2021)
16. Chunseong Park, C., Kim, B., Kim, G.: Attend to you: Personalized image captioning with context sequence memory networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 895-903 (2017)

17. Dekel, O., Keshet, J., Singer, Y.: Large margin hierarchical classification. In: Proceedings of the twenty-first international conference on Machine learning. p. 27 (2004)
18. Del Chiaro, R., Twardowski, B., Bagdanov, A., Van de Weijer, J.: Ratt: Recurrent attention to transient tasks for continual image captioning. Advances in Neural Information Processing Systems 33 , 16736-16748 (2020)
19. Demirel, B., Cinbis, R.G., Ikizler-Cinbis, N.: Image captioning with unseen objects. arXiv preprint arXiv:1908.00047 (2019)
20. Denton, E., Weston, J., Paluri, M., Bourdev, L., Fergus, R.: User conditional hashtag prediction for images. In: Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining. pp. 1731-1740 (2015)
21. Dinh, T.M., Tran, A.T., Nguyen, R., Hua, B.S.: Hyperinverter: Improving stylegan inversion via hypernetwork. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11389-11398 (2022)
22. Fan, L., Xiong, P., Wei, W., Wu, Y.: Flar: A unified prototype framework for fewsample lifelong active recognition. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15394-15403 (2021)
23. Fang, H., Xiong, P., Xu, L., Chen, Y.: Clip2video: Mastering video-text retrieval via image clip. arXiv preprint arXiv:2106.11097 (2021)
24. Feng, F., Liu, R., Wang, X., Li, X., Bi, S.: Personalized image annotation using deep architecture. IEEE Access 5 , 23078-23085 (2017)
25. Finn, C., Abbeel, P., Levine, S.: Model-agnostic meta-learning for fast adaptation of deep networks. In: International conference on machine learning. pp. 1126-1135. PMLR (2017)
26. Gal, R., Patashnik, O., Maron, H., Chechik, G., Cohen-Or, D.: Stylegan-nada: Clipguided domain adaptation of image generators. arXiv preprint arXiv:2108.00946 (2021)
27. Gao, P., Geng, S., Zhang, R., Ma, T., Fang, R., Zhang, Y., Li, H., Qiao, Y.: Clip-adapter: Better vision-language models with feature adapters. arXiv preprint arXiv:2110.04544 (2021)
28. Ge, Y., Zhang, R., Wu, L., Wang, X., Tang, X., Luo, P.: A versatile benchmark for detection, pose estimation, segmentation and re-identification of clothing images. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2019)
29. Hendricks, L.A., Venugopalan, S., Rohrbach, M., Mooney, R., Saenko, K., Darrell, T.: Deep compositional captioning: Describing novel object categories without paired training data. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1-10 (2016)
30. Hewitt, J., Li, X.L., Xie, S.M., Newman, B., Liang, P.: Ensembles and cocktails: Robust finetuning for natural language generation. In: NeurIPS 2021 Workshop on Distribution Shifts: Connecting Methods and Applications (2021)
31. Hill, F., Tieleman, O., von Glehn, T., Wong, N., Merzic, H., Clark, S.: Grounded language learning fast and slow. arXiv preprint arXiv:2009.01719 (2020)
32. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., Gelly, S.: Parameter-efficient transfer learning for nlp. In: International Conference on Machine Learning. pp. 2790-2799. PMLR (2019)
33. Hsieh, Y.G., Niu, G., Sugiyama, M.: Classification from positive, unlabeled and biased negative data. In: International Conference on Machine Learning. pp. 28202829. PMLR (2019)

34. Jia, X., Zhao, H., Lin, Z., Kale, A., Kumar, V.: Personalized image retrieval with sparse graph representation learning. In: Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining. pp. 2735-2743 (2020)
35. Kamath, A., Singh, M., LeCun, Y., Synnaeve, G., Misra, I., Carion, N.: Mdetrmodulated detection for end-to-end multi-modal understanding. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1780-1790 (2021)
36. Khan, M., Srivatsa, P., Rane, A., Chenniappa, S., Hazariwala, A., Maes, P.: Personalizing pre-trained models. arXiv preprint arXiv:2106.01499 (2021)
37. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014)
38. Kumar, A., Raghunathan, A., Jones, R., Ma, T., Liang, P.: Fine-tuning can distort pretrained features and underperform out-of-distribution. arXiv preprint arXiv:2202.10054 (2022)
39. Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Kamali, S., Popov, S., Malloci, M., Kolesnikov, A., et al.: The open images dataset v4. International Journal of Computer Vision 128 (7), 1956-1981 (2020)
40. Lake, B.M., Piantadosi, S.T.: People infer recursive visual concepts from just a few examples. Computational Brain &amp; Behavior 3 (1), 54-65 (2020)
41. Lampert, C., Nickisch, H., Harmeling, S.: Learning to detect unseen object classes by between-class attribute transfer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2009)
42. Li, B., Weinberger, K.Q., Belongie, S., Koltun, V., Ranftl, R.: Language-driven semantic segmentation. In: International Conference on Learning Representations (2022), https://openreview.net/forum?id=RriDjddCLN
43. Liang, W., Zhang, Y., Kwon, Y., Yeung, S., Zou, J.: Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. arXiv preprint arXiv:2203.02053 (2022)
44. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Doll´ ar, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740-755. Springer (2014)
45. Liu, N., Li, S., Du, Y., Tenenbaum, J., Torralba, A.: Learning to compose visual relations. Advances in Neural Information Processing Systems 34 (2021)
46. Long, C., Yang, X., Xu, C.: Cross-domain personalized image captioning. Multimedia Tools and Applications 79 (45), 33333-33348 (2020)
47. Lu, J., Yang, J., Batra, D., Parikh, D.: Neural baby talk. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 7219-7228 (2018)
48. Lynch, C., Sermanet, P.: Language conditioned imitation learning over unstructured data. arXiv preprint arXiv:2005.07648 (2020)
49. Ma, T., Geng, S., Wang, M., Shao, J., Lu, J., Li, H., Gao, P., Qiao, Y.: A simple long-tailed recognition baseline via vision-language model. arXiv preprint arXiv:2111.14745 (2021)
50. Malaviya, M., Sucholutsky, I., Oktar, K., Griffiths, T.L.: Can humans do less-thanone-shot learning? arXiv preprint arXiv:2202.04670 (2022)
51. Markman, E.M.: Constraints children place on word meanings. Cognitive science 14 (1), 57-77 (1990)
52. Markman, E.M., Wasow, J.L., Hansen, M.B.: Use of the mutual exclusivity assumption by young word learners. Cognitive psychology 47 (3), 241-275 (2003)

53. Menon, S., Damian, A., Hu, S., Ravi, N., Rudin, C.: Pulse: Self-supervised photo upsampling via latent space exploration of generative models. In: Proceedings of the ieee/cvf conference on computer vision and pattern recognition. pp. 2437-2445 (2020)
54. Mokady, R., Hertz, A., Bermano, A.H.: Clipcap: Clip prefix for image captioning. arXiv preprint arXiv:2111.09734 (2021)
55. Nitzan, Y., Aberman, K., He, Q., Liba, O., Yarom, M., Gandelsman, Y., Mosseri, I., Pritch, Y., Cohen-Or, D.: Mystyle: A personalized generative prior. arXiv preprint arXiv:2203.17272 (2022)
56. Patashnik, O., Wu, Z., Shechtman, E., Cohen-Or, D., Lischinski, D.: Styleclip: Text-driven manipulation of stylegan imagery. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2085-2094 (2021)
57. Paz-Argaman, T., Atzmon, Y., Chechik, G., Tsarfaty, R.: Zest: Zero-shot learning from text descriptions using textual similarity and visual summarization (2020)
58. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning. pp. 8748-8763. PMLR (2021)
59. Ren, M., Iuzzolino, M.L., Mozer, M.C., Zemel, R.S.: Wandering within a world: Online contextualized few-shot learning. arXiv preprint arXiv:2007.04546 (2020)
60. Ren, M., Ravi, S., Triantafillou, E., Snell, J., Swersky, K., Tenenbaum, J.B., Larochelle, H., Zemel, R.S.: Meta-learning for semi-supervised few-shot classification. In: International Conference on Learning Representations (2018), https: //openreview.net/forum?id=HJcSzz-CZ
61. Richardson, E., Alaluf, Y., Patashnik, O., Nitzan, Y., Azar, Y., Shapiro, S., CohenOr, D.: Encoding in style: a stylegan encoder for image-to-image translation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 2287-2296 (2021)
62. Roich, D., Mokady, R., Bermano, A.H., Cohen-Or, D.: Pivotal tuning for latentbased editing of real images. arXiv preprint arXiv:2106.05744 (2021)
63. Shen, Y., Yang, C., Tang, X., Zhou, B.: Interfacegan: Interpreting the disentangled face representation learned by gans. IEEE transactions on pattern analysis and machine intelligence (2020)
64. Shinoda, K., Kaji, H., Sugiyama, M.: Binary classification from positive data with skewed confidence. arXiv preprint arXiv:2001.10642 (2020)
65. Shridhar, M., Manuelli, L., Fox, D.: Cliport: What and where pathways for robotic manipulation. In: Proceedings of the 5th Conference on Robot Learning (CoRL) (2021)
66. Shuster, K., Humeau, S., Hu, H., Bordes, A., Weston, J.: Engaging image captioning via personality. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12516-12526 (2019)
67. Skantze, G., Willemsen, B.: Collie: Continual learning of language grounding from language-image embeddings. arXiv preprint arXiv:2111.07993 (2021)
68. Snell, J., Swersky, K., Zemel, R.: Prototypical networks for few-shot learning. Advances in neural information processing systems 30 (2017)
69. Sofiiuk, K., Petrov, I., Konushin, A.: Reviving iterative training with mask guidance for interactive segmentation. arXiv preprint arXiv:2102.06583 (2021)
70. Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P.H., Hospedales, T.M.: Learning to compare: Relation network for few-shot learning. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1199-1208 (2018)

71. Tao, X., Hong, X., Chang, X., Dong, S., Wei, X., Gong, Y.: Few-shot classincremental learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12183-12192 (2020)
72. Tov, O., Alaluf, Y., Nitzan, Y., Patashnik, O., Cohen-Or, D.: Designing an encoder for stylegan image manipulation. arXiv preprint arXiv:2102.02766 (2021)
73. Tsimpoukelli, M., Menick, J., Cabi, S., Eslami, S., Vinyals, O., Hill, F.: Multimodal few-shot learning with frozen language models. Advances in Neural Information Processing Systems 34 (2021)
74. Tzaban, R., Mokady, R., Gal, R., Bermano, A.H., Cohen-Or, D.: Stitch it in time: Gan-based facial editing of real videos. arXiv preprint arXiv:2201.08361 (2022)
75. Venugopalan, S., Anne Hendricks, L., Rohrbach, M., Mooney, R., Darrell, T., Saenko, K.: Captioning images with diverse objects. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5753-5761 (2017)
76. Vinyals, O., Blundell, C., Lillicrap, T., Wierstra, D., et al.: Matching networks for one shot learning. Advances in neural information processing systems 29 (2016)
77. Wang, L., Meng, X., Xiang, Y., Fox, D.: Hierarchical policies for cluttered-scene grasping with latent plans. IEEE Robotics and Automation Letters (2022)
78. Wortsman, M., Ilharco, G., Li, M., Kim, J.W., Hajishirzi, H., Farhadi, A., Namkoong, H., Schmidt, L.: Robust fine-tuning of zero-shot models. arXiv preprint arXiv:2109.01903 (2021)
79. Wu, G., Gong, S., Li, P.: Striking a balance between stability and plasticity for class-incremental learning. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1124-1133 (2021)
80. Wu, Y., Zhu, L., Jiang, L., Yang, Y.: Decoupled novel object captioner. In: Proceedings of the 26th ACM international conference on Multimedia. pp. 1029-1037 (2018)
81. Xian, Y., Schiele, B., Akata, Z.: Zero-shot learning - the good, the bad and the ugly. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2017)
82. Xu, N., Yang, L., Fan, Y., Yang, J., Yue, D., Liang, Y., Price, B., Cohen, S., Huang, T.: Youtube-vos: Sequence-to-sequence video object segmentation. In: Proceedings of the European conference on computer vision (ECCV). pp. 585-601 (2018)
83. Yuan, L., Chen, D., Chen, Y.L., Codella, N., Dai, X., Gao, J., Hu, H., Huang, X., Li, B., Li, C., et al.: Florence: A new foundation model for computer vision. arXiv preprint arXiv:2111.11432 (2021)
84. Zabari, N., Hoshen, Y.: Semantic segmentation in-the-wild without seeing any segmentation examples (2021)
85. Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R.R., Smola, A.J.: Deep sets. Advances in neural information processing systems 30 (2017)
86. Zhang, R., Fang, R., Gao, P., Zhang, W., Li, K., Dai, J., Qiao, Y., Li, H.: Tip-adapter: Training-free clip-adapter for better vision-language modeling. arXiv preprint arXiv:2111.03930 (2021)
87. Zhang, Y., Zhang, C.B., Jiang, P.T., Cheng, M.M., Mao, F.: Personalized image semantic segmentation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 10549-10559 (2021)
88. Zheng, Y., Li, Y., Wang, S.: Intention oriented image captions with guiding objects. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 8395-8404 (2019)
89. Zhou, K., Yang, J., Loy, C.C., Liu, Z.: Learning to prompt for vision-language models. arXiv preprint arXiv:2109.01134 (2021)

## Appendix

## A Implementation details

Number of training samples: The results reported in the main text used 5 training samples for each concept in the retrieval experiments, and 10 training samples for each concept in the segmentation experiments. Below, in Sec. B, we provide additional results that sweep over the number of training samples.

Cycle loss prompts: We use multiple prompts for querying the concept with the cycle loss. In each epoch, we selected a template at random from the following list of prompts.

'This is a photo of a [CONCEPT]', 'This photo contains a [CONCEPT]', 'A photo of a [CONCEPT]', 'This is an illustrations of a [CONCEPT]', 'This illustrations contains a [CONCEPT]', 'An illustrations of a [CONCEPT]', 'This is a sketch of a [CONCEPT]', 'This sketch contains a [CONCEPT]', 'A sketch of a [CONCEPT]', 'This is a diagram of a [CONCEPT]', 'This diagram contains a [CONCEPT]', 'A diagram of a [CONCEPT]', 'A [CONCEPT]', 'We see a [CONCEPT]', '[CONCEPT]', 'We see a [CONCEPT] in this photo', 'We see a [CONCEPT] in this image', 'We see a [CONCEPT] in this illustration', 'We see a [CONCEPT] photo', 'We see a [CONCEPT] image', 'We see a [CONCEPT] illustration', '[CONCEPT] photo', '[CONCEPT] image', '[CONCEPT] illustration'.

Contrastive loss: We apply all contrastive losses with a temperature hyperparameter (denoted by Temp = 0 . 25), dividing each cosine similarity in Eq.1,2. The value of Temp was selected using a validation set (details about hyper-parameter search below).

Ground-truth regularization for training the set encoder f θ : For training the set encoder f θ , we use a regularization term that maximizes the cosine similarity of the predicted word embedding w 0 c , with its ground truth word embedding g c , keeping w 0 c close to its ground truth value. Namely, ℓ GT ( w 0 c , g c ) = -cos( w 0 c , g c ), where g c is the word embedding of the concept type (e.g. the embedding of 'dog'). If the concept type includes more than a single word, we take the first one.

Architecture: When using CLIP, we always used ViT-B/32 Vision Transformer.

Normalized embeddings: Wherever we use a textual or visual encoder output, we first normalize the embedding vector to unit norm. The embedding can be viewed as lying on a hypersphere.

Training the alignment matrix A with images and captions: In addition to updating the alignment matrix A during training of f θ with text (Section 4.2), we also update A by mapping from captions to images. Specifically, for every image I embedded with CLIP v = h I ( I ), we took the respective caption S embedded with u = h T ( S ), and trained A to project from the embedded captions to the embedded images by minimizing an L 2 loss:

<!-- formula-not-decoded -->

Using the alignment matrix A for the fine-tuning stage: In practice, in the fine tuning stage (Eq. (2)), we replace η c by A · η c .

Training procedure for the set encoder f θ : We train f θ in an alternating fashion. One batch with COCO images and one batch with augmented COCO captions.

Hyperparameters: Hyper parameters were tuned one at a time, on a validation set, to maximize the MRR metric in the retrieval task.

We train f θ for 300 epochs. Batch size was set to 256. We used the Adam [37] optimizer with a learning rate of 0.0001 for both the cycle loss and the alignment loss (eq. A.3). DeepSet's hidden dimension was set to 4096, and the dropout rate was set to 0.25. The contrastive loss temperature was set to Temp = 0 . 25. To optimize the word embeddings, we used 30 epochs, with a learning rate of 0.01. The weight of the ground-truth regularization term λ gt was set to 512.

Weused the following ranges to search for hyper parameters: (1) number of epochs ∈ [100 , 200 , 300 , 500 , 1000] (2) batch size ∈ [128 , 256 , 512] (3) learning rate ∈ [0 . 01 , 0 . 001 , 0 . 0001 , 0 . 00001] (4) DeepSet's hidden dimension ∈ [512 , 1024 , 2048 , 4096] (5) dropout rate ∈ [0 . 15 , 0 . 25 , 0 . 35 , 0 . 5] (6) Temp ∈ [0 . 15 , 0 . 25 , 0 . 35 , 0 . 5] (7) number of fine-tuning personalization epochs ∈ [10 , 20 , 30 , 40 , 50 , 60] (8) fine-tuning learning rate ∈ [0 . 01 , 0 . 001 , 0 . 0001] (9) λ gt ∈ [1 , 2 , 4 , 8 , . . . 2048].

Randomization: Model: For each of our 5 repetitions we trained a new f θ model. Few-shot training data: When selecting a subset of few images from the few-shot training data, we made sure that the random seed (and subsets) are consistent between PALAVRA and the baselines (e.g. COLLIE, AvgIm, etc . . . ).

Training COLLIE and Adapter: To use multiple concepts that share the same concept type (category name), with the COLLIE and adapter baselines, we have assigned a unique [CONCEPT] phrase for concept. The phrase is composed of its class (e.g., skirt) and a unique ID number (e.g., 'skirt 241').

## B Additional Results

Accuracy versus number of shots: Fig. A.1 shows the performance of our model and the baselines as a function of the number of few-shot training samples used to learn each personalized concept. DeepFashion2 performance improves as the number of shots increases. YTVOS saturates early, probably because the training images of each concept have less variability since they were all extracted from the same video.

Fig. A.1. MRR for image retrieval on DeepFashion2 (left) and YTVOS (right) as function of the number of shots used to learn each personalized concept. DeepFashion2 performance improves as we increase the number of shots. YTVOS saturates early, probably because the training images of each concept have less variability, since they were all extracted from the same video.

<!-- image -->

Short versus detailed captions: Table A.1 shows retrieval results on DeepFashion2 when using longer, detailed captions. As expected, all text-based methods demonstrate an increase in retrieval metrics across the board, indicating that they can successfully leverage additional information. Our method remains at the front even in this scenario, highlighting that the benefits of personalized concepts persist even when detailed descriptions are provided.

Table A.1. Retrieval results using detailed captions. As expected, all compared methods show improved performance when provided with extra textual information. Notably, our method maintains the advantage even in such a scenario, showing that it can yield an increase in performance even when the concepts are described in detail.

|                     | MRR         | Recall@5    | Recall@10   |
|---------------------|-------------|-------------|-------------|
| PALAVRA (Ours)      | 33.8 ± 0.5% | 47.5 ± 0.9% | 61.9 ± 0.9% |
| PALAVRA w.o. tuning | 27.8 ± 0.3% | 36.4 ± 0.6% | 48.4 ± 0.6% |
| AvgIM+Text          | 20.9 ± 0.6% | 29.0 ± 0.7% | 38.4 ± 0.6% |
| Text (CLIP)         | 24.3 ± 0.0% | 31.7 ± 0.0% | 43.4 ± 0.0% |

In Sec. E.2 below we explain the data collection procedure of the 'detailed' and 'short' queries.

## COLLIE sensitivity to prompt:

In Sec. 6.1 we demonstrated that COLLIE performance degrades when using rich textual queries. Here we describe results showing that COLLIE is even sensitive to much simpler queries. Namely, template queries that only add a prefix prompt .

When the text query includes only the [CONCEPT] tag, as in COLLIE's training procedure, COLLIE achieves a 13.4% average MRR score on DeepFashion2 retrieval test set. When the query text is a sentence with a prefix, its score drops sharply. For example, the query 'This is a photo of a [CONCEPT]' results in an MRR score of 5.3%, the query 'This looks like a [CONCEPT]' score is 4.6%, and the query 'In this image, there is a [CONCEPT]' yields 3.3%.

Qualitative results for semantic segmentation: In Fig. A.2 we show curated qualitative segmentation results. We observe that our model can successfully segment the correct object instance even in scenarios with visually similar distractors. On the other hand, our model can sometimes fail to distinguish between multiple relevant candidates, or segment other objects which exist near the target. However, this last limitation may be an artifact of the underlying segmentation method.

Fig. A.2. Qualitative examples of PALAVRA used for semantic segmentation. Left and middle: successful segmentations, where the correct specific object is identified and extracted from the image despite similar distractors (other turtles). Right: Typical failure cases: distractors are segmented along with the personalized object (top), or the textual descriptions draw CLIP's attention away from the main object (bottom).

<!-- image -->

## C Personalization of Other Vision &amp; Language

Vision &amp; Language models other than CLIP may also benefit from an extended vocabulary of personalized concepts. It is likely that a similar approach to ours can still be applied. For example, in models like M-DETR [35], f θ could map from the CNN output space to the input space of RoBERTa. The alignment matrix A can close the cycle, mapping from the output of RoBERTa to the CNN output.

## D Segmentation Details and Analysis

## D.1 Baselines and hyper parameters

Our segmentation experiments use the framework of Zabari et al. [84]. The method leverages transformer interpretability methods to identify image regions that relate to a given textual prompt. In this process, the text-encoding branch is only used to supply an embedding vector which is matched with the image branch. As such, the embedding vector can be easily replaced with another vector from any source. We leverage this property for all of our baselines.

When conducting an image-based search (AvgIM), we replace the embedding vector with the normalized average embedding of a small set of images depicting the target object. For the AvgIM&amp;Text baseline, we further average this image embedding with the text embedding of the query text.

To compare with COLLIE, we generate the embeddings using their adapter setup. For our method, we utilize the original CLIP text encoder but substitute our learned input word embeddings for the concept token.

Hyper parameters were tuned on the validation set and kept fixed for all methods. We use a resizing factor of 0 . 5 and generate 3 'clicks' from the relevancy maps for the single image segmentation method. All other parameters were unchanged from the baseline implementation of Zabari et al. [84].

## D.2 Rich queries versus concept-only queries

In the main manuscript, we noted that surprisingly the segmentation model performed better when provided with a text target of the form 'A photo of a [CONCEPT]' (i.e. a 'Concept-only' query) than when provided with a rich textual caption.

To investigate this behavior, we turn to an analysis of the local relevance maps, which are used to guide the segmentation. Our investigation reveals that often, when the rich query describes other objects within the image, CLIP's attention drifts towards those objects. That is, CLIP struggles with leveraging relational information in the text and instead splits its focus between several objects mentioned in the rich query. Figure A.7 provides a qualitative visualization of this effect.

To quantitatively test this hypothesis , we re-ran segmentation, this time masking out relevancy scores of the background, except for objects which are also valid retrieval candidates. Now, context objects were no longer valid candidates. Indeed, we found that with this manipulation, rich queries do outperform concept-only queries, as in retrieval (Fig. A.3).

We conclude that a good caption for CLIP-guided personalized segmentation should describe the object or its immediate vicinity, and not its relation to other objects.

Last, we further investigated whether COLLIE demonstrates a similar sensitivity to rich queries. COLLIE's segmentation performance in the two scenarios is shown in Fig. A.4. We observe that, in contrast to our own approach and the baseline CLIP, COLLIE's performance on the segmentation task does not appear to be sensitive to the level of detail in the query.

## D.3 Qualitative Analysis

In this section, we provide an additional qualitative analysis of segmentation using PALAVRA. We compare our results with the recent baseline COLLIE. Figure A.5 shows examples of successful segmentation, and Figure A.6 shows some failure modes.

## E Evaluation datasets

We provide details for creating our two new benchmark datasets, based on DeepFashion and YTVOS.

Fig. A.3. Rich queries outperform concept-only queries when context objects are not valid segmentation candidates.

<!-- image -->

Fig. A.4. COLLIE performance when supplied with rich queries and with concept-only queries. The performance of COLLIE does not depend on the rich query, indicating that the additional information is largely ignored in the case of segmentation.

<!-- image -->

Fig. A.5. Examples of successful segmentation. For visualization purposes, we replaced the [CONCEPT] tag by the name of its concept type, and highlighted it in cyan.

<!-- image -->

Fig. A.6. Examples of segmentation failures. For visualization purposes, we replaced the [CONCEPT] tag by the name of its concept type, and highlighted it in cyan.

<!-- image -->

<!-- image -->

'A [CONCEPT] perched inside a cage.'

Fig. A.7. Qualitative examples of 'attention drift' when using rich queries. When the descriptor mentions other objects, CLIP's attention visually drifts away from the target concept and towards other objects described in the query. For example, in the top row, focus moves from the hat and towards the brown horses. On the bottom row, focus moved away from the parrot and towards the empty cage at the bottom of the frame.

## E.1 DeepFashion2

To ensure that DeepFashion2 benchmark items are included in a rich visual context, items were included in the dataset if they obey the following criteria: (1) Have at least 5 images with a proper scale (zoom). Specifically, the item covers no more than 50% of the image. (2) There are at least 15 images of the same item in total. The set yielded ∼ 1700 images and 100 unique fashion items (concepts) which met these criteria. Each unique fashion item was assigned a unique [CONCEPT] tag.

Next, we explain how we annotated a subset of this data with textual descriptions, and how we selected the evaluation set.

We manually curated a subset of 652 images (out of 1700) that contain a person wearing a fashion item and at least one additional object for a context. We did not consider mobile phones or mirrors as valid context, as these objects are abundant in the dataset. For each image, we collected a textual description that refers to each fashion item. For instance, The [CONCEPT] is facing a glass store display.' .

To provide diverse captions, we instruct the raters to avoid trivial captions such as 'a [CONCEPT] in front of a mirror' . We also instructed them to avoid describing the item itself, because the same item appears in several evaluation images, and we wished to have a textual query which is specific to one single image.

We randomly sampled an evaluation set (out of the 652 images), by sampling 5 images per concept, or less, if not available. This results in 450 evaluation images and 1250 images for training. Finally, we made a concept-based split by randomly splitting the dataset to 50 validation concepts and 50 test concepts.

## Annotations for DeepFashion2 with Amazon Mechanical Turk

To simplify the instructions for collecting textual annotations, we used the fact that every fashion item is worn by a person. When describing the images, we simply asked the raters to relate to the person in the image in context of the objects in the scene, and in a post-processing step, we replace every mention of the word 'person'

## 32 N. Cohen et al.

<!-- image -->

Fig. A.8. Instructions for collecting textual descriptions for images of the DeepFashion2 benchmark.

<!-- image -->

Fig. A.9. Instructions for summarizing textual descriptions for images of the DeepFashion2 benchmark.

by the '[CONCEPT]' token. Additional instructions were inspired by the instructions provided for collecting captions for the COCO dataset (See appendix of [44]).

Finally, to maintain the quality of the textual descriptions, we only worked with the raters after they passed our qualification test, making sure that they followed the instructions when describing 5-10 images. In addition, we only worked with raters with AMT 'masters' qualification, demonstrating a high degree of approval rate in a wide range of tasks. We paid the raters 0 . 2$ for annotating each image.

Fig. A.8 provides an example of the data collection API for textual annotation of images for the DeepFashion2 benchmark.

## E.2 Summarizing textual annotations with AMT

As explained in Sec. B, for DeepFashion2 we created two types of captions for each image, in order to quantify the effect of caption length. We expected that image retrieval with short textual queries will pose a greater challenge, because they contain less information about the target image, leading to queries that are more ambiguous.

To create the set of 'short' text queries, we took the set of image captions described in Sec. E.1, which we now denote as 'detailed' captions, and asked the AMT raters to summarize each caption. Given a detailed caption, their goal was to describe the concept in the context of a single object in the scene. An example of a caption and its summarized version is: ' White cabinets, some with open drawers, are alongside and behind the [CONCEPT]. ' was summarized to ' White cabinets are behind the [CONCEPT]. '

Fig. A.9 provides an example of the data collection API to summarize textual descriptions.

Similarly to the previous section, to maintain the quality of the textual descriptions, we only worked with raters after they passed our qualification test and have AMT 'masters' qualification. We paid the raters 0 . 1$ for summarizing each caption.

Finally, for most of the DeepFashion2 experiments throughout the paper, we used the more challenging 'short' queries. In Sec. B we describe the evaluation results with the 'detailed' queries.

## E.3 Youtube-VOS

## Overview

We created an image segmentation benchmark of personalized visual concepts given a textual query using Youtube-VOS (YTVOS) [82]. YTVOS is a dataset for instance segmentation in video, which includes 4000+ videos, 90+ categories, and 7800+ unique object instances. The original videos were 3 -6 second long with a 30 FPS frame rate. The dataset contains a subset of the frames, sampled at rate of 6 FPS. To transform the dataset into an image personalization benchmark, we take the last frame of each video (scene) for evaluation and the object instances that appear in it as target concepts. Earlier frames that contain that object are used as candidate frames for few-shot training. See examples in Figures 5(left), A.2, A.5, A.6 and A.7.

For building the concept set, we consider each object instance (e.g. each animal in the frame) as a unique personalized concept. We chose training samples such that their object instantiations are not trivially solved by simple visual template matching with the last (evaluation) frame. To that end, we use the following criteria: For each object instance, we consider all the previous video frames that contain it. We keep only the frames where: (1) the object's segmentation mask has a zero intersection-over-union (IOU) score when compared with its mask at the last frame (i.e. the evaluation target) and (2) the center of the mask moved at least 150 pixels when compared to the final frame. We discard any object instance that does not have at least 4 training examples left at the end of this filtering process. Finally, we take a box crop of the images around the selected masks and use them as training examples.

## Annotation with AMT

We annotated the instances in the evaluation frame with captions using AMT. We instructed the AMT workers to concisely describe what makes a specific entity distinct, compared with similar entities in the image, and, if possible, preferring descriptions that relate to one object that is nearby.

Finally, similar to the previous sections, to maintain the quality of the textual descriptions, we only worked with raters after they passed our qualification test and have AMT 'masters' qualification. We paid the raters 0 . 3$ for every textual description.

Fig. A.10 provides an example of the data collection API for textual annotation of images for the Youtube-VOS benchmark.

Fig. A.10. Instructions for collecting textual descriptions for object instances in the Youtube-VOS benchmark.

<!-- image -->

Personalized image retrieval: We also created an image retrieval variant of YTVOS. We extract a set of images that correspond to the collected captions, where every image in the retrieval set was extracted from a wide box cropped around every instance in each evaluation frame. The box size was set to twice the size of the instance mask on each axis (that is, four times the area), to allow it to display some information about the context of the instance.