## UniTAB: Unifying Text and Box Outputs for Grounded Vision-Language Modeling

Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu,

Faisal Ahmed, Zicheng Liu, Yumao Lu, Lijuan Wang

Microsoft Cloud and AI

{zhengyang,zhe.gan,jianfw,xiaowei.hu,

fiahmed,zliu,yumaolu,lijuanw}@microsoft.com

Abstract. We propose UniTAB that Unifies Text And Box outputs for grounded vision-language (VL) modeling. Grounded VL tasks such as grounded captioning require the model to generate a text description and align predicted words with object regions. To achieve this, models must generate desired text and box outputs together, and meanwhile indicate the alignments between words and boxes. In contrast to existing solutions that use multiple separate modules for different outputs, UniTAB represents both text and box outputs with a shared token sequence, and introduces a special &lt;obj&gt; token to naturally indicate word-box alignments in the sequence. UniTAB thus could provide a more comprehensive and interpretable image description, by freely grounding generated words to object regions. On grounded captioning, UniTAB presents a simpler solution with a single output head, and significantly outperforms state of the art in both grounding and captioning evaluations. On general VL tasks that have different desired output formats ( i.e ., text, box, or their combination), UniTAB with a single network achieves better or comparable performance than task-specific state of the art. Experiments cover 7 VL benchmarks, including grounded captioning, visual grounding, image captioning, and visual question answering. Furthermore, UniTAB's unified multi-task network and the task-agnostic output sequence design make the model parameter efficient and generalizable to new tasks. 1

## 1 Introduction

Text sequences [11,5] and bounding boxes [44,80] are two representative output formats for image understanding tasks [17,44,11]. Text is well suited for generating image-level predictions, such as describing an image with a sentence [11] or tagging an image with keywords [21], but fails to refer to a dense image region. On the other hand, box could point to any image area [44], but alone has a limited ability to provide semantically-rich descriptions. A natural question is can we have a single model that unifies text and box outputs, i.e ., generating both text and box outputs while aligning predicted words with boxes. Unifying these two output formats allows the model to better express its understanding of the image. Taking captioning as an example, such a unified model could ground all noun entities [86,54] in the caption back to aligned image regions, thus providing a more comprehensive and interpretable image description. This problem is known as grounded captioning [86,88,50,54]. Moreover, unifying output formats is one important step toward the grand vision of building task-agnostic, generalpurpose vision systems [24] that are parameter efficient and well generalizable.

[1 Code is available at https://github.com/microsoft/UniTAB .](https://github.com/microsoft/UniTAB)

Fig. 1. We propose UniTAB that Unifies Text And Box outputs with no formatspecific modules. UniTAB generates both text and box tokens in an auto-regressive manner, conditioned on the multimodal image-text inputs. The introduced &lt;obj&gt; token naturally indicates the word-box alignments, as shown in word-box pairs of the same color in the right visualization. UniTAB thus can approach a wide range of VL tasks, including the challenging grounded captioning, with a single unified architecture. The gray tokens in the task-agnostic output sequence are predictions not used for downstream task evaluation, e.g ., box tokens in image captioning and VQA.

<!-- image -->

Recent works [13,24,86,88,50] have developed models that can generate both text and box outputs. Specifically, the system combines an online [24] or offline [13,86,88,50] object detection module that predicts boxes, with a visionlanguage model that generates text. The word and box alignments are then separately generated as additional predictions, such as the relevance score [24,86,88,50]. Predicting text, box, and their alignments separately weakens the benefits of a unified system. The separate modules prevent the framework from being simple and parameter efficient. Furthermore, the explicit object detection component increases the model running time [36] and potentially limits its generalization ability given the preset detector vocabulary [72], as discussed in previous VL studies [36,72]. Going beyond these successful initial explorations, we ask a bolder question: can we unify the output formats with no separate modules ? Specifically, we explore 1). how to have a single architecture without an explicit detector jointly generating text and box, and 2). how to represent the word-box alignments naturally in the output to avoid the additional alignment prediction. To this end, we model both text and box predictions as an auto-regressive token generation task, and present a single encoder-decoder model that is fully shared among text, box, and alignment predictions.

Our modeling of box prediction takes inspiration from Pix2seq [10], an object detection study showing that predicting boxes in an auto-regressive manner yields good detection performance [44]. Its core idea is to quantize the four coordinates in each box into four discrete box tokens, and arrange them with a fixed ap order into a token sequence, i.e ., [ y min , x min , y max , x max ] . Box prediction can then be modeled as a multi-step classification task, instead of conventional coordinate regression [23,57,8]. The same classification modeling as in text generation [55] makes it possible to unify text and box prediction. However, Pix2seq is designed for the single-modal object detection task, and does not support open-ended text generation nor multimodal inputs and outputs. Moreover, it is unclear how the text and box alignment is intended to be presented in a unified sequence.

In this study, we propose UniTAB that unifies text and box outputs. As shown in Figure 1, we unify open-ended text generation [55] and discrete box token prediction [10] into a single shared decoder. During the auto-regressive decoding, UniTAB switches to box tokens right after any text words to be grounded, and switches back to text tokens after predicting the box. In UniTAB, we study how to handle such text-box code-switching [74] and naturally represent word-box alignments. We introduce a special &lt;obj&gt; token inserted before the text word to be grounded, and after the generated box tokens. The &lt;obj&gt; token simplifies the sequence generation by providing hints of the code-switching, and naturally represents word-box alignments. That is, the words and box within a pair of &lt;obj&gt; tokens refer to the same entity, as shown in word-box pairs of the same color in Figure 1. With the &lt;obj&gt; token and output sequence design, UniTAB approaches grounded VL tasks such as grounded captioning [86,54] and phrase grounding [54] with a single decoder, in contrast to separately predicting text, box, and their alignments with multiple output heads [86,50,88,34].

We further apply UniTAB on general VL tasks [86,80,51,54,5,11,85] and observe two unique properties. First , the unified architecture for text, box, and alignment predictions enables UniTAB to perform multi-task training [1,73,6], which learns a single set of parameters for different VL tasks without introducing task-specific heads. Multi-task training avoids task-specific model copies and thus saves the parameters to store. It also facilities the use of data in different tasks, thus boosting the performance of certain VL tasks. Second , as shown in Figure 1, UniTAB's output sequence is designed to be task-agnostic and shares the same text+box design across different VL tasks. The task-agnostic output design could help UniTAB generalize to certain unseen tasks, by reformatting new tasks' desired outputs into the seen text+box sequences.

We evaluate UniTAB on 7 VL benchmarks, including grounded captioning [86,54], visual grounding [80,51,54], image captioning [11], and visual question answering [5], all with a single encoder-decoder network architecture, trained by the cross-entropy language modeling objective [55]. With a unified framework and minimum task-specific assumptions, our model achieves better or comparable performance with task-specific state of the art. In grounded captioning, UniTAB not only presents a simpler solution by eliminating separate taskspecific heads [86,50,88,34], but also significantly outperforms the prior art [50,9] (from 62 . 5 to 69 . 7 in captioning CIDEr score and from 8 . 44 to 12 . 95 in grounding F1 score). Our contributions are summarized as follows.

- -UniTAB is the first grounded VL model that can approach a wide range of tasks, including the challenging grounded captioning, without separate out-

## 4 Z. Yang et al.

Table 1. Summary of unified VL models. We highlight the desired modeling in blue. Visual Modeling: instead of using an object detection (OD) module, we take raw 'image patches' as visual input. Text Output: instead of using task-specific output heads [47,32,34,28] for different VL tasks (classification or text generation heads), we use a 'single output sequence' [13,24] to approach different tasks. Box Output: many prior models cannot predict boxes [32] or simplify it as region index prediction with detector-generated region proposals [47,13,24]. We aim to predict 'box coordinates' without an explicit OD module [34,28]. Word-box Align: most models fail to generate either open-ended text [47,34,28] or object boxes [32], thus cannot represent wordbox alignments. In contrast to the extra alignment predictions [24,13], our introduced &lt;obj&gt; token naturally indicates word-box alignments 'inline' in the output sequence.

| Representative Models                                                        | Visual Modeling      | Text Output         | Box Output     | Word-box Align   |
|------------------------------------------------------------------------------|----------------------|---------------------|----------------|------------------|
| ViLBERT [47], OSCAR [43], UNITER [12], VinVL [82], etc . [41,66,39,65,87,48] | Offline OD           | Task-specific Heads | Region Index   | ✗                |
| PixelBERT [32], SOHO [31], ViLT [36], SimVLM [72], etc . [63,40,76,20,70]    | Image Patches        | Task-specific Heads | ✗              | ✗                |
| VL-T5 [13] GPV [24]                                                          | Offline OD Online OD | Single Output Seq.  | Region Index   | Extra Prediction |
| MDERT [34], UniT [28]                                                        | Image Patches        | Task-specific Heads | Box Coordinate | ✗                |
| UniTAB (Ours)                                                                | Image Patches        | Single Output Seq.  | Box Coordinate | Inline Indicated |

put modules. We introduce the &lt;obj&gt; token that helps text and box outputs synergistically work together, with their alignments naturally represented.

- -UniTAB achieves better or comparable performance to state of the art on 7 VL benchmarks. Its unified multi-task network and the task-agnostic output sequence design make it parameter efficient and generalizable to new tasks.

## 2 Related Work

Grounded captioning. The grounded captioning task [86,54] requires the model to generate a text caption and grounds all mentioned noun phrases [86,54] to aligned image regions. The input is a single image, and the desired outputs are the caption sentence, multiple object boxes, and the word-box alignments. Existing methods [86,50,88,9] adopt separate output heads for text, box (usually with an offline detector [58,4]), and alignment predictions. In contrast, UniTAB uses a single decoding sequence to represent all desired outputs.

Vision-language pre-training (VLP). Large-scale VLP has become the new training paradigm for VL research. Prior works [47,41,2,39,66,65,87,12,48,43] first show the power of VLP by using region features obtained from an off-theshelf object detector [58]. However, the region feature extraction significantly increases the model's computation cost and run time. Recent studies [32,36,40,72] shift the paradigm and show that grid features extracted from raw image patches also work well. Most studies adopt similar output architectures of either discriminative classification heads or auto-regressive text decoders. As shown in the second row of Table 1, these output structures often contain task-specific designs next to e plate?

mes Output text The &lt;obj&gt; coffee mug &lt;10&gt; &lt;7&gt; &lt;94&gt; &lt;113&gt; next to the &lt;obj&gt; plate &lt;78&gt; &lt;84&gt; &lt;186&gt; &lt;199&gt;

&lt;obj&gt; A donut &lt;90&gt; &lt;83&gt; &lt;184&gt; &lt;180&gt; on a &lt;obj&gt;

white plate &lt;78&gt; &lt;84&gt; &lt;186&gt; &lt;199&gt; next to a

&lt;obj&gt; cup of latte &lt;10&gt; &lt;7&gt; &lt;94&gt; &lt;113&gt;.

A donut on a white plate next to a cup of latte.

The plate is white.

&lt;obj&gt; &lt;90&gt; &lt;83&gt; &lt;184&gt; &lt;180&gt; &lt;obj&gt; &lt;78&gt; &lt;84&gt;

&lt;186&gt; &lt;199&gt; &lt;obj&gt; &lt;10&gt; &lt;7&gt; &lt;94&gt; &lt;113&gt;

Step 1. Step 2. Fig. 2. UniTAB is an encoder-decoder framework that can jointly output open-ended text and box without output format specific modules. A transformer encoder-decoder takes the encoded image-text features to predict the target text+box sequence. The bottom sub-figure illustrates the output target sequence design. We introduce a special &lt;obj&gt; token to indicate the alignments between predicted words and boxes, such as words 'a donut' and the blue box. During decoding, the output sequence could seamlessly switch between text and box tokens to ground an object, if applicable.

<!-- image -->

and do not support bounding box prediction, which is an important output format for VL tasks such as visual grounding and grounded image captioning. Unified VL framework. Prior works have presented successful explorations on building VL models with unified input-output formats. VL-T5 [13] and GPV [24] first represent images as object region features with an online or offline object detector [58,8]. Bounding box prediction is then simplified as index classification over the set of region candidates generated by the detector. The other threads, MDETR [34] and UniT [28], add task-specific classification heads on top of the DETR object detector [8] to perform VL tasks. However, different tasks still require different output heads. Moreover, it is unclear how to extend the framework for open-ended text generation, thus supporting VL tasks like image captioning. In this study, we aim to build a single unified framework that takes structured inputs ( i.e ., raw image and language) in, and generates structured outputs ( i.e ., text and boxes), with no output format specific modules.

## 3 The UniTAB Framework

## 3.1 Architecture Overview

We implement UniTAB using a transformer encoder-decoder architecture built on top of the single-modality image and text encoders, as shown in Figure 2. For image, we use ResNet-101 [25] to encode the raw image input v , and flatten the grid features as the visual representation. For text, we use RoBERTa BASE [45] to encode input text l into hidden word features. The encoded image and text features are then projected into a shared embedding space. We use a 6 -layer transformer encoder that takes the concatenated image and text feature sequence as input, and a 6 -layer transformer decoder for output sequence generation. The decoder generates output tokens in an auto-regressive manner, similar to language modeling [55,56]. The UniTAB decoder could generate tokens from both the text and box vocabularies, as shown in the right part of Figure 2.

## 3.2 UniTAB Target Output Sequence

We show how to construct ground-truth target output sequences, such that text and box can be jointly represented with word-box alignments contained inline. Box token sequence. We first review the bounding box quantization approach introduced in Pix2seq [10]. As shown in the bottom part of Figure 2, a rectangular bounding box in a 2D image can be represented by four floatingpoint numbers, namely [ x min , y min , x max , y max ] . The established object detection paradigm [58,57,8] predicts four continuous floating-point values to regress the coordinates in a single step. In contrast, Pix2seq quantizes each coordinate into one of the n bins discrete bins, and represent each box with four tokens arranged sequentially. We adopt the similar idea and represent each box as four discrete tokens, [ &lt;x min &gt;,&lt;y min &gt;,&lt;x max &gt;,&lt;y max &gt; ] , where &lt;x&gt; , &lt;y&gt; are quantized box tokens ranging from &lt; 0 &gt; , to &lt;n bins -1 &gt; .

Unified decoding sequence with &lt; obj &gt; token. We aim to have a unified decoding sequence s that can jointly represent text and box, meanwhile indicating word-box alignments. For the former, we unify the text and box vocabularies such that a single decoder can freely generate text or box tokens at any decoding step. Specifically, UniTAB's decoding vocabulary contains both text and box tokens, and has a size of n text + n bins +2 . n text and n bins are the text vocabulary size and the number of coordinate bins. We use the same set of n bins box tokens [10] for all four box coordinates. The output token selection at each decoding step is conducted over the entire unified vocabulary.

The remaining question is how to represent the word-box alignments in the output sequence. Instead of extra alignment score prediction [24,86,88,50], we represent word-box alignments inline with two introduced special tokens &lt;obj&gt; and &lt; \ obj&gt; . Specifically, the model switches to box tokens right after any text words to be grounded, and inserts the &lt;obj&gt; tokens before the first text word and after the last box tokens, respectively. For example, in Figure 2, we extend the text phrase 'a donut' in the text-only caption as ' &lt;obj&gt; a donut &lt; 90 &gt;&lt; 83 &gt; &lt; 184 &gt; &lt; 180 &gt; &lt; \ obj&gt; ' in the extended target sequence, where 90 , 83 , 184 , 180 are the quantized box coordinates for the blue box. The word-box alignments then can be easily extracted from the predicted sequence, i.e ., words and box within the pair of &lt;obj&gt; tokens refer to the same entity, such as 'a donut.'

## 3.3 UniTAB Training

Objective. We train the model with a single language modeling objective [55], i.e ., at each decoding step t , maximizing the likelihood of target token s t conditioned on input image v , input text l , and previous target tokens s &lt;t :

<!-- formula-not-decoded -->

where θ denotes the model parameters, and T is the target sequence length.

- Training stages. UniTAB's unified structure facilitates the pre-training and finetuning that use the same language modeling objective. We train UniTAB with up to three stages. The first is vision-language pre-training, which leverages large-scale image-text dataset optionally with grounded box annotations. Then, we perform multi-task finetuning, where multiple downstream task datasets with supervised annotations are merged to finetune a single model for different VL tasks. Lastly, we could conduct task-specific finetuning that adapts the model to each specific task for further improvement. The three stages share the same training objective as in Eq. 1, but with different training corpus and inputoutput designs. We discuss the combinations of these different training stages in Section 4.3. We next introduce each of these three training stages.
1. Pre-training. Pre-training aims to use large-scale data loosely related to downstream tasks for general VL representation learning. We pre-train the model with a single language modeling objective to predict the target sequence s , conditioned on image v and input text l . We randomly set the input text l as an empty string or the text-only image description, with the same probability of 0 . 5 . We train the model to generate the text+box sequence s shown in Figure 2. The model thus learns to perform both captioning-like (with empty string input) and grounding-like (with image description input) VL tasks during pre-training.
2. Multi-task finetuning. Multi-task finetuning [1,73,6] aims to use supervised annotations from multiple downstream task datasets to train a single model, thus avoiding task-specific model copies and further boosting the model performance. UniTAB's unified architecture and training objective facilitate the unique property of multi-task finetuning. Instead of having multiple duplicates of a pre-trained model, each optimized for a downstream task, multi-task finetuning trains a single set of parameters to perform all different VL tasks. We gather supervised data annotations from all 7 experimented VL tasks and train a single model with the language modeling objective. One major advantage of multi-task finetuning is that a single model can support multiple VL tasks, thus saving model parameters. Multi-task finetuning could also improve certain downstream tasks' performance by using annotations from different tasks.
3. Task-specific finetuning. UniTAB can also perform the standard taskspecific finetuning as in VLP studies [47,12,43]. Furthermore, we observe that multi-task finetuning not only generates a single model that performs well in different VL tasks, but also serves as a good initialization point for a secondstage task-specific finetuning. We refer to this setting as 'pre-finetuning' [1,73,6].
- Inference. We use arg max sampling to obtain the sequence prediction. We then extract the text and box predictions from the sequence offline for final evaluation. For example, we discard box tokens to get the text prediction, and dequantize box tokens to get the box prediction. Finally, we evaluate the model on each downstream task with its desired output formats, e.g ., text for VQA, boxes for visual grounding, or both text and boxes for grounded captioning. We show in Section 4.3 that the task-agnostic output sequence design could help UniTAB generalize to unseen tasks that require text or box outputs.

## 4 Experiments

## 4.1 Experiment Overview

Downstream tasks. We evaluate UniTAB on 7 VL benchmarks (later summarized in Table 6). We start with grounded captioning [86,54] that requires the model to predict text, box, and their alignment. We then benchmark UniTAB on other representative VL tasks, including visual grounding [80,51,54], COCO image captioning [11], and VQAv2 visual question answering [5]. UniTAB approaches a wide range of VL tasks with a single unified architecture. In contrast, prior works require task-specific model designs, making it difficult to work on VL tasks with different desired output formats (text, box, or their combination). Model variants. In addition to the comparison with state of the art, we systematically study the following UniTAB variants with different training stages:

- Separate-scratch conducts task-specific finetuning without pre-training.
- Shared-scratch conducts multi-task finetuning without pre-training.
- Separate is first pre-trained and then optimized separately for each downstream task, i.e ., the standard pretrain-then-finetune setting in VLP [47,12,43].
- Shared uses multi-task finetuning after pre-training, and shares a single set of parameters for all experimented VL tasks.
- Pre-finetuning adopts two-stage finetuning from a pre-trained checkpoint. The first stage is multi-task finetuning, followed by task-specific finetuning.

We take UniTAB Pre-finetuning as the default setting and refer to it as UniTAB. We report the main 'Pre-finetuning' results in Section 4.2, and discuss the full results of UniTAB variants in Table 6 and Section 4.3.

Training corpus. The pre-training corpus [34] aggregates images from Flickr30k Entities [54], COCO [44,11], and Visual Genome (VG) [37] datasets. Text and grounded box annotations are from the referring expression datasets [80,51], VG regions, Flickr30k Entity annotations, and the GQA dataset [33]. The corpus contains around 200K images and 1.3M image-text pairs with grounded box annotations. Optionally, we further add the image-text data with no box annotations from Conceptual Captioning [62] and SBU [52] to pre-training, with settings and results detailed in Section 4.3. For multi-task finetuning, we collect supervised annotations from all 7 downstream datasets [86,54,80,51,11,5] to jointly train a single model for different tasks.

Implementation details. The transformer contains 6 encoder layers and 6 decoder layers, with 8 attention heads and a hidden dimension of 256 in each layer [8]. We use the scale and crop augmentation in DETR [8] such that the shortest side is between 480 and 800 pixels while the longest at most is 1333 . We pre-train the model for 40 epochs, and finetune for 20 epochs in multitask and task-specific settings. We use a learning rate of 1 e -4 and 2 e -5 for transformer layers and backbones. We train our model with AdamW [46] and adopt exponential moving average [68,34] with a decay rate of 0 . 9998 and a weight decay of 1 e -4 . More details are provided in Appendix A.

Table 2. Grounded image captioning results on the test set of Flickr30k Entities [54]. BLEU@4 [53], METEOR [19], CIDEr [69], and SPICE [3] metrics are used for caption evaluation. F1 all and F1 loc metrics [86] are used for grounding evaluation. Caption scores with † are optimized with CIDEr [59].

| Method           | Caption Eval.   | Caption Eval.   | Caption Eval.   | Caption Eval.   | Grounding Eval.   | Grounding Eval.   |
|------------------|-----------------|-----------------|-----------------|-----------------|-------------------|-------------------|
|                  | B@4             | M               | C               | S               | F1 all            | F1 loc            |
| NBT [49]         | 27.1            | 21.7            | 57.5            | 15.6            | -                 | -                 |
| GVD [86]         | 27.3            | 22.5            | 62.3            | 16.5            | 7.55              | 22.2              |
| Cyclical [50]    | 26.8            | 22.4            | 61.1            | 16.8            | 8.44              | 22.78             |
| POS-SCAN [88]    | 30.1 †          | 22.6 †          | 69.3 †          | 16.8 †          | 7.17              | 17.49             |
| Chen et al . [9] | 27.2            | 22.5            | 62.5            | 16.5            | 7.91              | 21.54             |
| UniTAB           | 30.1            | 23.7            | 69.7            | 17.4            | 12.95             | 34.79             |

## 4.2 Comparison with Prior Arts

Grounded captioning. The grounded captioning task [86,54] requires the model to generate a caption and ground all generated noun phrases to image regions. The final predictions consist of three parts, i.e ., the text caption, visual regions as boxes, and the grounding alignments between words and boxes. Instead of separately predicting those outputs with multiple output heads [86,50,88], UniTAB naturally represents all desired outputs with a single unified text+box output sequence. Following the established benchmarks [86,50,88] on the Flickr30k Entities dataset, we evaluate 'captioning' and 'grounding' separately with the caption metrics [53,19,3,69] and grounding F1 scores [86], respectively. The F1 score F 1 all evaluates grounding as a multi-label classification problem, where a correct prediction contains both the same object word as ground-truth (GT) caption and a larger than 0 . 5 IoU with the GT box. We also report F 1 loc that only computes the grounding score on correctly predicted object words.

Table 2 compares our method to state of the art [86,50,88,9]. We observe a significant improvement in the grounding quality, with the F1 all score improving from 8 . 44 to 12 . 95 , and F1 loc from 22 . 78 to 34 . 79 . UniTAB also achieves a better captioning quality, with the CIDEr score improving from 62 . 5 to 69 . 7 , compared with prior arts [9]. By exploiting image-text data without box in pre-training, we further boost the CIDEr score from 69 . 7 to 74 . 2 , as detailed in Section 4.3.

In addition to the performance improvement, UniTAB presents a simpler and more natural way for the grounded captioning task. Specifically, UniTAB does not require the pre-generated object regions [86,50,88] and avoids using multiple output heads. As shown in Figure 3(a), UniTAB naturally represents text, box, and word-region alignments in a single unified output sequence. Such a simple approach better transfers the model's grounding ability to other datasets or tasks with limited box or grounding annotations, such as COCO caption [11] and ImageNet [17], as shown in Figures 3(d,f). We hope UniTAB's new paradigm simplifies future studies on grounded VL tasks.

Visual grounding. Visual grounding aims to ground language queries into aligned image regions. We experiment on the sub-tasks of referring expression comprehension [80,51] with Refcoco/Refcoco+/Refcocog, and phrase grounding [54] with Flickr30k Entities. Referring expression comprehension contains a query that describes a single image region and expects a single box prediction. Phrase grounding aims to ground all noun phrases in the input sentence, and requires the model to predict all referred boxes and the word-box alignments. In contrast to previous studies that do not know word-box alignments [79,77,18] or require separate alignment predictions [34], UniTAB generates a unified sequence with word-box alignments naturally represented by the special &lt;obj&gt; token. We report the standard metric Acc@0.5 [80,51,54].

Table 3. The performance comparisons (Acc@0.5) on the referring expression comprehension (Refcoco, Refcoco+, Refcocog) and phrase grounding task (Flickr30k Entities).

| Method          | Refcoco   | Refcoco   | Refcoco   | Refcoco+   | Refcoco+   | Refcoco+   | Refcocog   | Refcocog   | Flickr30k   |
|-----------------|-----------|-----------|-----------|------------|------------|------------|------------|------------|-------------|
|                 | val       | testA     | testB     | val        | testA      | testB      | val-u      | test-u     | Entities    |
| MAttNet [79]    | 76.40     | 80.43     | 69.28     | 64.93      | 70.26      | 56.00      | 66.67      | 67.01      | -           |
| FAOA [77]       | 72.05     | 74.81     | 67.59     | 55.72      | 60.37      | 48.54      | 59.03      | 58.70      | 68.71       |
| TransVG [18]    | 81.02     | 82.72     | 78.35     | 64.82      | 70.70      | 56.94      | 68.67      | 67.73      | 79.10       |
| ViLBERT [47]    | -         | -         | -         | 72.34      | 78.53      | 62.61      | -          | -          | -           |
| UNITER [12]     | 81.41     | 87.04     | 74.17     | 75.90      | 81.45      | 66.70      | 74.02      | 68.67      | -           |
| VILLA [22]      | 82.39     | 87.48     | 74.84     | 76.17      | 81.54      | 66.84      | 76.18      | 76.71      | -           |
| MDETR [34]      | 86.75     | 89.58     | 81.41     | 79.52      | 84.09      | 70.62      | 81.64      | 80.89      | 83.8        |
| UniTAB Separate | 86.32     | 88.84     | 80.61     | 78.70      | 83.22      | 69.48      | 79.96      | 79.97      | 79.39       |
| UniTAB          | 88.59     | 91.06     | 83.75     | 80.97      | 85.36      | 71.55      | 84.58      | 84.70      | 79.58       |

As shown in Table 3, UniTAB outperforms the state of the art, including those pre-trained on larger VL corpus [47,12,22] and methods that use carefullydesigned task-specific architectures [79,77,18]. Moreover, UniTAB's unified output with both text and box presents a more natural way of visual grounding, compared to box regression [77,18,34] or region index classification [79,12,13]. UniTAB's multi-task finetuning enables the use of data from different tasks and datasets, thus boosting performance on all splits, compared with UniTAB Separate . COCO captioning. We benchmark UniTAB on the COCO image captioning dataset [44]. We report the results without beam search [4] or CIDEr optimization [59]. Table 4 shows the captioning results on the Karparthy test split [35]. We refer to our pre-training corpus as '200K' in the '#Pre-train' column, and introduce the corpus used by compared methods later in Appendix A.

UniTAB achieves better performance than prior arts [75,13] that use similar amounts of pre-training images, with the CIDEr score improved from 117 . 3 to 119 . 8 . Meanwhile, UniTAB does not require input region proposals or object tags [87,43,13]. Using extra image-text pairs [62,52] in pre-training further boosts the CIDEr score to 123 . 1 . We expect a further gain by scaling up the pre-training corpus, as observed in VLP studies [82,40,72,30]. Despite only being evaluated with caption metrics on COCO, UniTAB's unified output sequence could also ground generated noun phrases to image regions, as visualized in Figure 3(d). Visual question answering. UniTAB takes a generative approach to the VQA task [5], where the model generates a free-form text sequence to represent the Method

Table 4. COCO image captioning results on the Karparthy test split. The '#Pre-train' column shows the number of pre-training images, if any.

#Pre-train B@4 M

Unified VLP [87]

OSCAR [43]

E2E-VLP [75]

VL-T5 [13]

VL-BART [13]

UniTAB

Table 5. Visual question answering results on VQAv2 [5]. We experiment on both test-dev/test-std splits, and the Karpathy test split used in VL-T5 [13].

| Method       | #Pre- train   | Test- Dev Std   | Karpathy-test In Out All   |
|--------------|---------------|-----------------|----------------------------|
| UNITER [12]  | 4M            | 72.7 72.9       | 74.4 10.0 70.5             |
| VL-T5 [13]   | 180K          | - 70.3          | 71.4 13.1 67.9             |
| VL-BART [13] | 180K          | - 71.3          | 72.1 13.2 68.6             |
| UniTAB       | 200K          | 70.7 71.0       | 71.1 11.1 67.5             |

3M

C

S

36.5 28.4 117.7 21.3

4M

180K

180K

180K

200K

36.5 30.3 123.7 23.1

36.2

-

117.3

-

34.5 28.7 116.5 21.9

35.1 28.7 116.6 21.5

36.1 28.6 119.8 21.7

Table 6. Summary of results obtained by UniTAB and its variants. The compared methods (upper portion) use task-specific architectures and training objectives, thus could only perform a subset of VL tasks. UniTAB (bottom portion) approaches all tasks with a unified framework and obtains competitive performance. The Refcoco/Refcoco+/Refcocog numbers are on the val set. The Flickr grounding and grounded caption results are on the test set. VQAv2-KP is the VQA Karpathy split [13]. UniTAB Pre-finetuning is the default setting that is also referred to as UniTAB.

| Method                           | #Pre- train   |         | Visual grounding   | Visual grounding   |        | Grounded caption   | Grounded caption   | COCO test-Cider   | VQAv2    | VQAv2   |
|----------------------------------|---------------|---------|--------------------|--------------------|--------|--------------------|--------------------|-------------------|----------|---------|
| Method                           | #Pre- train   | Refcoco | Refcoco+           | Refcocog           | Flickr | Cider              | F1 all             |                   | test-dev | KP-test |
| MDETR [34]                       | 200K          | 86.75   | 79.52              | 81.64              | 83.8   | -                  | -                  | -                 | 70.6     | -       |
| UNITER [12]                      | 4M            | 81.24   | 75.31              | 74.31              | -      | -                  | -                  | -                 | 72.7     | 70.5    |
| GVD [86]                         | -             | -       | -                  | -                  | -      | 62.3               | 7.55               | -                 | -        | -       |
| VL-T5 [13]                       | 180K          | -       | -                  | 71.2               | -      | -                  | -                  | 116.5             | -        | 67.9    |
| OSCAR [43]                       | 4M            | -       | -                  | -                  | -      | -                  | -                  | 123.7             | 73.2     | -       |
| UniTAB Variants Separate-scratch | None          | 72.96   | 64.98              | 63.56              | 73.40  | 60.5               | 9.22               | 105.3             | 55.4     | 52.4    |
| Shared-scratch                   | None          | 82.06   | 70.72              | 73.39              | 65.67  | 61.1               | 7.85               | 111.8             | 65.8     | 63.1    |
| Separate                         | 200K          | 86.32   | 78.70              | 79.96              | 79.39  | 65.6               | 11.46              | 119.3             | 69.9     | 66.6    |
| Shared                           | 200K          | 88.50   | 80.98              | 84.46              | 79.23  | 63.4               | 9.18               | 115.8             | 69.1     | 66.6    |
| Pre-finetuning                   | 200K          | 88.59   | 80.97              | 84.58              | 79.58  | 69.7               | 12.95              | 119.8             | 70.7     | 67.5    |

answer. Table 5 reports the VQA results on both the official test-dev/std split [5] and the Karparthy split [35] used in VL-T5 [13]. The Karparthy test set is further split into in- and out-domain subsets, based on whether the answer is covered in the top-K (K= 3129 ) vocabulary [13]. The metric is the soft-voting accuracy [5]. UniTAB obtains competitive results to the state of the art, and performs better on the Karparthy out-of-domain subset than the discriminative approach [12].

## 4.3 Ablation and Analysis

Training stage ablation. We compare the variants of UniTAB to examine the influence of different pre-training and finetuning stages introduced in Section 3.3. The bottom portion of Table 6 summarizes the results. We first discuss the standard pretrain-then-finetune setting in VLP [47,12,43] that adopts taskspecific finetuning. UniTAB Separate approaches various VL tasks with a single unified architecture, and obtains competitive results to the state of the art that has architectures tailored for each task, or uses larger-scale pre-training data. Compared with UniTAB Separate-scratch without pre-training, pre-training leads to consistent improvements on all experimented tasks.

With UniTAB's unified architecture and output modeling, we can train a single UniTAB Shared model for all experimented VL tasks. Compared with UniTAB Separate , the multi-task finetuning UniTAB Shared performs comparable or better on experimented VL tasks, while using 7 times fewer model parameters by avoiding task-specific model copies. The strong performance of UniTAB Shared indicates that we can use a single model for multiple downstream tasks, thus being parameter efficient . We further experiment with adding task-specific prefixes [73,13] to the input text. This variant uses a task-specific prefix such as 'visual grounding:' to describe each sample's task. We observe that the task prefix has no major influence on model performance, as detailed in Appendix C.

In addition to achieving good performance with a single model, multi-task finetuning UniTAB Shared also provides a strong initialization point for further task-specific finetuning. UniTAB Pre-finetuning further boosts the performance and achieves better or comparable performance than the state of the art on experimented VL tasks, as shown in the bottom row of Table 6.

Zero-shot generalization. The task-agnostic output sequence design helps UniTAB generalize to new tasks. UniTAB could perform certain tasks in a zeroshot manner by transferring the learned ability of generating text+box sequences s conditioned on image-text inputs. We next explore adapting UniTAB to ImageNet object localization [17]. Object localization [85,14,71] aims to localize an ImageNet class onto an object region. We take the words in class names as the text input, and have UniTAB generate text+box sequence s conditioned on image-text inputs. We then obtain box predictions by extracting boxes and alignments from s , similar to the phrase grounding post-processing. There exist two established benchmark settings. The 'GT-known' [64,83,84,15] setting aims to localize a given ground-truth class. The metrics [14] 'MaxBoxAcc' and 'MaxBoxAccV2' are the Top-1 accuracy with an IoU threshold of 0 . 5 , and the average at thresholds 0 . 3 / 0 . 5 / 0 . 7 . The second setting tries to localize a predicted class. The metric is 'Top-1 accuracy' with a 0 . 5 IoU threshold. We use EfficientNet [67] classification result with an accuracy of 77 . 5% for this setting.

We experiment with UniTAB Shared and show ImageNet object localization results in Table 7. UniTAB achieves better performance than the state of the art without using ImageNet images or annotations. The good generalization results show the possibility of generalizing UniTAB to unseen images and tasks in a zeroshot manner. We expect larger-scale pre-training to boost such generalization ability further, as observed in the NLP community [7,73].

Pre-training with additional image-text pairs. We experiment with adding image-text pairs without boxes in UniTAB pre-training, and examine if the extra image-text data could further improve VL tasks that require text output. For image-text pair data, we pre-train the model to generate the text-only caption conditioned on image and an empty text input. The model variant is referred to as 'Separate †† ,' which uses 4M image-text pairs from Conceptual Captioning [62] and SBU [52]. Table 8 compares 'Separate †† ' with UniTAB Separate on grounded captioning, COCO captioning, and VQA. We observe consistent improvements in the text output quality by using extra image-text pairs, i.e ., +8 . 6 CIDEr score on grounded captioning [54], +3 . 8 CIDEr score on COCO captioning [11], and +2 . 5% absolute accuracy on VQA [5]. Appendix C further discusses the benefit of pre-training with other addition data, such as boxes from object detection [44]. Model and output sequence design. We empirically observe that the introduced &lt;obj&gt; token not only naturally represents the word-box alignment, but also simplifies the sequence prediction by providing hints of the text-box codeswitching, thus helping the VL tasks' performance. We postpone the detailed ablation studies on model and output sequence design to Appendix B, including the effectiveness of &lt;obj&gt; token, decoding sampling methods [4,27,10], the number of object tokens, decoding syntactic restrictions, etc .

Table 7. Zero-shot object localization results on ImageNet [17]. Prior works with the weakly supervised setting use ImageNet class labels.

| Method         | Top-1 Acc.   |   MaxBoxAcc |   MaxBoxAccV2 |
|----------------|--------------|-------------|---------------|
| CAM [85]       | 51.8         |        64.2 |          63.7 |
| HaS [64]       | 49.9         |        63.1 |          63.4 |
| CutMix [81]    | 51.5         |        65.4 |          63.3 |
| MinMaxCAM [71] | -            |        66.7 |          65.7 |
| UniTAB Shared  | 60.2         |        68.1 |          67.8 |

Table 8. UniTAB pre-training with additional image-text pairs. 'Separate †† ' uses additional 4M image-text pairs from CC [62] and SBU [52] that do not have grounded box annotations.

| UniTAB      | Grounded caption   | Grounded caption   | COCO       | VQAv2   |
|-------------|--------------------|--------------------|------------|---------|
| UniTAB      | Cider              | F1 all             | test-Cider | KP-test |
| Separate    | 65.6               | 11.46              | 119.3      | 66.6    |
| Separate †† | 74.2               | 12.62              | 123.1      | 69.1    |

## 4.4 Qualitative Results

Figure 3 shows the predictions made by UniTAB Shared on different VL tasks, where all predictions are made by a single model with the same set of parameters. On the right side of each subfigure, we show the input text and predicted output sequence. The output sequence is colored for visualization purposes only, where the text and box colors indicate the word-box alignments. We then show the extracted text and box predictions used for downstream task evaluation. For text, we discard all box tokens to obtain the text-only sequence. For boxes, we keep box tokens and dequantize them as box coordinate predictions [10].

UniTAB's task-agnostic output sequence seamlessly supports different VL tasks. Figure 3(a) shows an example of grounded captioning, where the input text is a blank string and both text and box predictions are used for evaluation. UniTAB could perform the phrase grounding task with the exact output sequence design, by replacing the blank input text with an image description, as shown in Figure 3(b). Figure 3(c) shows a referring expression comprehension example from the Refcocog dataset [51]. The model correctly localizes the referred 'cat' in the 'mirror.' Despite not being used by the downstream task evaluation, the model successfully aligns the predicted box with phrase 'the cat.'

UniTAB's unified output sequence helps the model transfer the grounded description ability to datasets or tasks with limited box or grounding annotations. As shown in Figure 3(d), UniTAB learns grounded captioning on Flickr30k Entities and transfers such ability to COCO during multi-task finetuning. The generated caption not only has a good caption quality, as evaluated in Table 4, nswering (VQAv2)

Input text:

What two forms of transportation are

pictured here?

Output seq.:

bus and car

Text:

bus and car

Box:

Not Used

360719&lt;sep&gt;255771&lt;sep&gt;&lt;obj&gt;

the Ġcat &lt;91&gt; &lt;16&gt; &lt;149&gt; &lt;196&gt;

&lt;/obj&gt; Ġin Ġthe Ġmirror&lt;sep&gt;the

cat in the mirror but also contains grounding predictions that make the description more comprehensive and interpretable. With the task-agnostic output sequence design, we further explore generalizing UniTAB to unseen images or even new tasks. Figure 3(e) shows an example of zero-shot object localization on ImageNet. The model correctly localizes the dog conditioned on the text input of ImageNet class label 'brittany spaniel.' Figure 3(f) shows an example of zero-shot grounded captioning on ImageNet images, where UniTAB generates a smooth caption and correctly grounds all noun phrases. More qualitative results are in Appendix D.

Fig. 3. Predictions made by UniTAB Shared that uses a single model for different VL tasks. In each subfigure, we show the input text, the raw output sequence, and the extracted outputs for downstream task evaluations. Specifically, the output sequence contains an open-ended text sequence, box predictions (visualized as bounding boxes), and word-box alignments (visualized as the word-box colors). (a-d) UniTAB approaches a wide range of VL tasks with a single unified model and output sequence. (e,f) With the task-agnostic output sequence, we further generalize UniTAB to unseen images or even new tasks, with examples on ImageNet object localization and grounded captioning.

<!-- image -->

## 5 Conclusion

We have presented UniTAB that unifies text and box outputs for grounded VL modeling. With the special &lt;obj&gt; token, UniTAB could generate both text and box predictions, with the word-box alignments naturally represented in the output sequence. Unifying text and box outputs allows the model to better approach grounded VL tasks such as grounded captioning. Furthermore, the unified multi-task network and the task-agnostic output sequence design make UniTAB parameter efficient and generalizable to new tasks. We see great potential in UniTAB, and believe it paves the way for building vision systems with stronger intelligence, such as in-context learning [7] and instruction tuning [73].

78058719&lt;sep&gt;&lt;obj&gt; A Ġwoman

&lt;35&gt; &lt;56&gt; &lt;101&gt; &lt;199&gt; &lt;\obj&gt; Ġis

Ġmanipulating &lt;obj&gt; Ġdishes &lt;79&gt;

&lt;144&gt; &lt;99&gt; &lt;179&gt; &lt;\obj&gt; Ġin

&lt;obj&gt; Ġa Ġdish washer &lt;76&gt;

&lt;178&gt; &lt;113&gt; &lt;199&gt; &lt;\obj&gt; Ġwith

&lt;obj&gt; Ġa Ġman &lt;91&gt; &lt;13&gt; &lt;181&gt;

&lt;199&gt; &lt;\obj&gt; Ġand &lt;obj&gt; Ġanother

Ġwoman &lt;156&gt; &lt;37&gt; &lt;195&gt; &lt;199&gt;

&lt;\obj&gt; Ġnext Ġto Ġher Ġ.&lt;sep&gt;A

woman is manipulating dishes in a

dishwasher with a man and another

woman next to her .

15216&lt;sep&gt;&lt;obj&gt; A Ġyoung

Ġwoman &lt;19&gt; &lt;16&gt; &lt;117&gt; &lt;199&gt;

&lt;\obj&gt; Ġwearing &lt;obj&gt; Ġa Ġblue

Ġshirt &lt;50&gt; &lt;99&gt; &lt;113&gt; &lt;199&gt;

&lt;\obj&gt; Ġand &lt;obj&gt; Ġblack Ġhat

&lt;37&gt; &lt;17&gt; &lt;99&gt; &lt;86&gt; &lt;\obj&gt; Ġis

Ġlooking Ġat Ġsomething

Ġ.&lt;sep&gt;A young woman wearing a

blue shirt and black hat is looking at

something .

33982

&lt;40&gt; &lt;

&lt;obj&gt;

&lt;47&gt; &lt;

&lt;obj&gt;

&lt;130&gt;

Ġwith

&lt;113&gt;

Ġpark

&lt;\obj&gt;

jacket

with a

17572&lt;sep&gt;&lt;obj&gt; Three Ġpeople

&lt;73&gt; &lt;53&gt; &lt;124&gt; &lt;125&gt; &lt;\obj&gt;

Ġare Ġsitting Ġon &lt;obj&gt; Ġa

Ġbench &lt;74&gt; &lt;77&gt; &lt;128&gt; &lt;126&gt;

&lt;\obj&gt; Ġby &lt;obj&gt; Ġa Ġriver &lt;0&gt;

&lt;41&gt; &lt;199&gt; &lt;91&gt; &lt;\obj&gt;

Ġ.&lt;sep&gt;Three people are sitting on

a bench by a river .

## References

1. Aghajanyan, A., Gupta, A., Shrivastava, A., Chen, X., Zettlemoyer, L., Gupta, S.: Muppet: Massive multi-task representations with pre-finetuning. In: EMNLP (2021) 3, 7
2. Alberti, C., Ling, J., Collins, M., Reitter, D.: Fusion of detected objects in text for visual question answering. In: EMNLP (2019) 4
3. Anderson, P., Fernando, B., Johnson, M., Gould, S.: Spice: Semantic propositional image caption evaluation. In: ECCV (2016) 9, 26
4. Anderson, P., He, X., Buehler, C., Teney, D., Johnson, M., Gould, S., Zhang, L.: Bottom-up and top-down attention for image captioning and visual question answering. In: CVPR (2018) 4, 10, 13
5. Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C.L., Parikh, D.: VQA: Visual Question Answering. In: ICCV (2015) 1, 3, 8, 10, 11, 13
6. Aribandi, V., Tay, Y., Schuster, T., Rao, J., Zheng, H.S., Mehta, S.V., Zhuang, H., Tran, V.Q., Bahri, D., Ni, J., et al.: Ext5: Towards extreme multi-task scaling for transfer learning. In: ICLR (2022) 3, 7
7. Brown, T.B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. In: NeurIPS (2020) 12, 14
8. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: Endto-end object detection with transformers. In: ECCV (2020) 3, 5, 6, 8, 21
9. Chen, N., Pan, X., Chen, R., Yang, L., Lin, Z., Ren, Y., Yuan, H., Guo, X., Huang, F., Wang, W.: Distributed attention for grounded image captioning. In: ACMMM (2021) 3, 4, 9
10. Chen, T., Saxena, S., Li, L., Fleet, D.J., Hinton, G.: Pix2seq: A language modeling framework for object detection. In: ICLR (2022) 2, 3, 6, 13, 22
11. Chen, X., Fang, H., Lin, T.Y., Vedantam, R., Gupta, S., Dollár, P., Zitnick, C.L.: Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325 (2015) 1, 3, 8, 9, 13
12. Chen, Y.C., Li, L., Yu, L., Kholy, A.E., Ahmed, F., Gan, Z., Cheng, Y., Liu, J.: Uniter: Learning universal image-text representations. In: ECCV (2020) 4, 7, 8, 10, 11, 20, 24
13. Cho, J., Lei, J., Tan, H., Bansal, M.: Unifying vision-and-language tasks via text generation. In: ICML (2021) 2, 4, 5, 10, 11, 12, 20, 24
14. Choe, J., Oh, S.J., Lee, S., Chun, S., Akata, Z., Shim, H.: Evaluating weakly supervised object localization methods right. In: CVPR (2020) 12
15. Choe, J., Shim, H.: Attention-based dropout layer for weakly supervised object localization. In: CVPR (2019) 12
16. Dancette, C., Cadene, R., Teney, D., Cord, M.: Beyond question-based biases: Assessing multimodal shortcut learning in visual question answering. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1574-1583 (2021) 24
17. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A large-scale hierarchical image database. In: CVPR (2009) 1, 9, 12, 13, 27
18. Deng, J., Yang, Z., Chen, T., Zhou, W., Li, H.: Transvg: End-to-end visual grounding with transformers. In: ICCV (2021) 10, 27
19. Denkowski, M., Lavie, A.: Meteor universal: Language specific translation evaluation for any target language. In: Proceedings of the ninth workshop on statistical machine translation (2014) 9, 26

20. Dou, Z.Y., Xu, Y., Gan, Z., Wang, J., Wang, S., Wang, L., Zhu, C., Liu, Z., Zeng, M., et al.: An empirical study of training end-to-end vision-and-language transformers. arXiv preprint arXiv:2111.02387 (2021) 4
21. Fu, J., Rui, Y.: Advances in deep learning approaches for image tagging. APSIPA Transactions on Signal and Information Processing 6 (2017) 1
22. Gan, Z., Chen, Y.C., Li, L., Zhu, C., Cheng, Y., Liu, J.: Large-scale adversarial training for vision-and-language representation learning. In: NeurIPS (2020) 10, 20
23. Girshick, R., Donahue, J., Darrell, T., Malik, J.: Rich feature hierarchies for accurate object detection and semantic segmentation. In: CVPR (2014) 3
24. Gupta, T., Kamath, A., Kembhavi, A., Hoiem, D.: Towards general purpose vision systems. arXiv preprint arXiv:2104.00743 (2021) 2, 4, 5, 6
25. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: CVPR (2016) 5, 21
26. Hendricks, L.A., Burns, K., Saenko, K., Darrell, T., Rohrbach, A.: Women also snowboard: Overcoming bias in captioning models. In: ECCV (2018) 24, 27
27. Holtzman, A., Buys, J., Du, L., Forbes, M., Choi, Y.: The curious case of neural text degeneration. In: ICLR (2020) 13, 22
28. Hu, R., Singh, A.: Unit: Multimodal multitask learning with a unified transformer. In: ICCV (2021) 4, 5
29. Hu, R., Singh, A., Darrell, T., Rohrbach, M.: Iterative answer prediction with pointer-augmented multimodal transformers for textvqa. In: CVPR (2020) 21
30. Hu, X., Gan, Z., Wang, J., Yang, Z., Liu, Z., Lu, Y., Wang, L.: Scaling up visionlanguage pre-training for image captioning. In: CVPR (2022) 10
31. Huang, Z., Zeng, Z., Huang, Y., Liu, B., Fu, D., Fu, J.: Seeing out of the box: Endto-end pre-training for vision-language representation learning. In: CVPR (2021) 4
32. Huang, Z., Zeng, Z., Liu, B., Fu, D., Fu, J.: Pixel-bert: Aligning image pixels with text by deep multi-modal transformers. arXiv preprint arXiv:2004.00849 (2020) 4
33. Hudson, D.A., Manning, C.D.: Gqa: A new dataset for real-world visual reasoning and compositional question answering. In: CVPR (2019) 8
34. Kamath, A., Singh, M., LeCun, Y., Synnaeve, G., Misra, I., Carion, N.: Mdetrmodulated detection for end-to-end multi-modal understanding. In: ICCV (2021) 3, 4, 5, 8, 10, 11, 20, 24
35. Karpathy, A., Fei-Fei, L.: Deep visual-semantic alignments for generating image descriptions. In: CVPR (2015) 10, 11
36. Kim, W., Son, B., Kim, I.: Vilt: Vision-and-language transformer without convolution or region supervision. In: ICML (2021) 2, 4
37. Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.J., Shamma, D.A., et al.: Visual genome: Connecting language and vision using crowdsourced dense image annotations. IJCV (2017) 8, 20, 23
38. Kuznetsova, A., Rom, H., Alldrin, N., Uijlings, J., Krasin, I., Pont-Tuset, J., Kamali, S., Popov, S., Malloci, M., Kolesnikov, A., et al.: The open images dataset v4. International Journal of Computer Vision 128 (7), 1956-1981 (2020) 23
39. Li, G., Duan, N., Fang, Y., Gong, M., Jiang, D., Zhou, M.: Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training. In: AAAI (2020) 4
40. Li, J., Selvaraju, R.R., Gotmare, A.D., Joty, S., Xiong, C., Hoi, S.: Align before fuse: Vision and language representation learning with momentum distillation. In: NeurIPS (2021) 4, 10

41. Li, L.H., Yatskar, M., Yin, D., Hsieh, C.J., Chang, K.W.: Visualbert: A simple and performant baseline for vision and language. arXiv preprint arXiv:1908.03557 (2019) 4
42. Li*, L.H., Zhang*, P., Zhang*, H., Yang, J., Li, C., Zhong, Y., Wang, L., Yuan, L., Zhang, L., Hwang, J.N., Chang, K.W., Gao, J.: Grounded language-image pretraining. In: arXiv preprint arXiv:2112.03857 (2021) 20
43. Li, X., Yin, X., Li, C., Hu, X., Zhang, P., Zhang, L., Wang, L., Hu, H., Dong, L., Wei, F., et al.: Oscar: Object-semantics aligned pre-training for vision-language tasks. In: ECCV (2020) 4, 7, 8, 10, 11, 20, 24
44. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: ECCV (2014) 1, 2, 8, 10, 13, 20, 23
45. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., Stoyanov, V.: Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692 (2019) 5, 21
46. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017) 8, 21
47. Lu, J., Batra, D., Parikh, D., Lee, S.: Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In: NeurIPS (2019) 4, 7, 8, 10, 11, 20
48. Lu, J., Goswami, V., Rohrbach, M., Parikh, D., Lee, S.: 12-in-1: Multi-task vision and language representation learning. In: CVPR (2020) 4
49. Lu, J., Yang, J., Batra, D., Parikh, D.: Neural baby talk. In: CVPR (2018) 9
50. Ma, C.Y., Kalantidis, Y., AlRegib, G., Vajda, P., Rohrbach, M., Kira, Z.: Learning to generate grounded visual captions without localization supervision. In: ECCV (2020) 2, 3, 4, 6, 9
51. Mao, J., Huang, J., Toshev, A., Camburu, O., Yuille, A.L., Murphy, K.: Generation and comprehension of unambiguous object descriptions. In: CVPR (2016) 3, 8, 10, 13, 20, 21, 22, 27
52. Ordonez, V., Kulkarni, G., Berg, T.L.: Im2text: Describing images using 1 million captioned photographs. In: NeurIPS (2011) 8, 10, 12, 13, 20, 23
53. Papineni, K., Roukos, S., Ward, T., Zhu, W.J.: Bleu: a method for automatic evaluation of machine translation. In: ACL (2002) 9, 26
54. Plummer, B.A., Wang, L., Cervantes, C.M., Caicedo, J.C., Hockenmaier, J., Lazebnik, S.: Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In: ICCV (2015) 2, 3, 4, 8, 9, 10, 13, 21, 22, 27
55. Radford, A., Narasimhan, K., Salimans, T., Sutskever, I.: Improving language understanding by generative pre-training (2018) 3, 6
56. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer. JMLR (2020) 6
57. Redmon, J., Divvala, S., Girshick, R., Farhadi, A.: You only look once: Unified, real-time object detection. In: CVPR (2016) 3, 6
58. Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object detection with region proposal networks. In: NeurIPS (2015) 4, 5, 6
59. Rennie, S.J., Marcheret, E., Mroueh, Y., Ross, J., Goel, V.: Self-critical sequence training for image captioning. In: CVPR (2017) 9, 10
60. Rohrbach, A., Hendricks, L.A., Burns, K., Darrell, T., Saenko, K.: Object hallucination in image captioning. In: EMNLP (2018) 27

61. Shao, S., Li, Z., Zhang, T., Peng, C., Yu, G., Zhang, X., Li, J., Sun, J.: Objects365: A large-scale, high-quality dataset for object detection. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 8430-8439 (2019) 23
62. Sharma, P., Ding, N., Goodman, S., Soricut, R.: Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In: ACL (2018) 8, 10, 12, 13, 20, 23
63. Shen, S., Li, L.H., Tan, H., Bansal, M., Rohrbach, A., Chang, K.W., Yao, Z., Keutzer, K.: How much can clip benefit vision-and-language tasks? In: ICLR (2022) 4
64. Singh, K.K., Lee, Y.J.: Hide-and-seek: Forcing a network to be meticulous for weakly-supervised object and action localization. In: ICCV (2017) 12, 13
65. Su, W., Zhu, X., Cao, Y., Li, B., Lu, L., Wei, F., Dai, J.: Vl-bert: Pre-training of generic visual-linguistic representations. In: ICLR (2019) 4
66. Tan, H., Bansal, M.: Lxmert: Learning cross-modality encoder representations from transformers. In: EMNLP (2019) 4
67. Tan, M., Le, Q.: Efficientnet: Rethinking model scaling for convolutional neural networks. In: ICML (2019) 12
68. Tarvainen, A., Valpola, H.: Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In: NeurIPS (2017) 8
69. Vedantam, R., Lawrence Zitnick, C., Parikh, D.: Cider: Consensus-based image description evaluation. In: CVPR (2015) 9, 26
70. Wang, J., Yang, Z., Hu, X., Li, L., Lin, K., Gan, Z., Liu, Z., Liu, C., Wang, L.: Git: A generative image-to-text transformer for vision and language. arXiv preprint arXiv:2205.14100 (2022) 4
71. Wang, K., Oramas, J., Tuytelaars, T.: Minmaxcam: Improving object coverage for cam-basedweakly supervised object localization. arXiv preprint arXiv:2104.14375 (2021) 12, 13
72. Wang, Z., Yu, J., Yu, A.W., Dai, Z., Tsvetkov, Y., Cao, Y.: Simvlm: Simple visual language model pretraining with weak supervision. In: ICLR (2022) 2, 4, 10
73. Wei, J., Bosma, M., Zhao, V.Y., Guu, K., Yu, A.W., Lester, B., Du, N., Dai, A.M., Le, Q.V.: Finetuned language models are zero-shot learners. In: ICLR (2022) 3, 7, 12, 14
74. Wikipedia contributors: Code-switching - Wikipedia, the free encyclopedia (2022), https://en.wikipedia.org/w/index.php?title=Code-switching&amp; oldid=1068820985 3
75. Xu, H., Yan, M., Li, C., Bi, B., Huang, S., Xiao, W., Huang, F.: E2e-vlp: End-toend vision-language pre-training enhanced by visual learning. In: ACL (2021) 10, 11, 20
76. Xue, H., Huang, Y., Liu, B., Peng, H., Fu, J., Li, H., Luo, J.: Probing intermodality: Visual parsing with self-attention for vision-and-language pre-training. In: NeurIPS (2021) 4
77. Yang, Z., Gong, B., Wang, L., Huang, W., Yu, D., Luo, J.: A fast and accurate one-stage approach to visual grounding. In: ICCV (2019) 10, 27
78. Yang, Z., Lu, Y., Wang, J., Yin, X., Florencio, D., Wang, L., Zhang, C., Zhang, L., Luo, J.: Tap: Text-aware pre-training for text-vqa and text-caption. In: CVPR. pp. 8751-8761 (2021) 21
79. Yu, L., Lin, Z., Shen, X., Yang, J., Lu, X., Bansal, M., Berg, T.L.: Mattnet: Modular attention network for referring expression comprehension. In: CVPR (2018) 10, 27

80. Yu, L., Poirson, P., Yang, S., Berg, A.C., Berg, T.L.: Modeling context in referring expressions. In: ECCV (2016) 1, 3, 8, 10, 20, 27
81. Yun, S., Han, D., Oh, S.J., Chun, S., Choe, J., Yoo, Y.: Cutmix: Regularization strategy to train strong classifiers with localizable features. In: CVPR (2019) 13
82. Zhang, P., Li, X., Hu, X., Yang, J., Zhang, L., Wang, L., Choi, Y., Gao, J.: Vinvl: Revisiting visual representations in vision-language models. In: CVPR (2021) 4, 10
83. Zhang, X., Wei, Y., Feng, J., Yang, Y., Huang, T.S.: Adversarial complementary learning for weakly supervised object localization. In: CVPR (2018) 12
84. Zhang, X., Wei, Y., Kang, G., Yang, Y., Huang, T.: Self-produced guidance for weakly-supervised object localization. In: ECCV (2018) 12
85. Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., Torralba, A.: Learning deep features for discriminative localization. In: CVPR (2016) 3, 12, 13
86. Zhou, L., Kalantidis, Y., Chen, X., Corso, J.J., Rohrbach, M.: Grounded video description. In: CVPR (2019) 2, 3, 4, 6, 8, 9, 11, 20, 24
87. Zhou, L., Palangi, H., Zhang, L., Hu, H., Corso, J.J., Gao, J.: Unified visionlanguage pre-training for image captioning and vqa. In: AAAI (2020) 4, 10, 11, 20
88. Zhou, Y., Wang, M., Liu, D., Hu, Z., Zhang, H.: More grounded image captioning by distilling image-text matching model. In: CVPR (2020) 2, 3, 4, 6, 9

## A Experiment Details

Hyper-parameter. Wesummarize the detailed experiment settings of UniTAB in Table 9. In the UniTAB decoder, we encode previous target token inputs s &lt;t with token and position embedding, and do not use type embedding to differentiate text and box tokens.

Training corpus. We introduce the '200K' pre-training corpus in the main paper, which contains both image-text pairs and grounded box annotations. In the main paper's Tables 4-6, we refer to the training corpus used in previous studies by their contained image numbers. Specifically, the '180K' corpus [13,75] aggregate images and annotations from COCO [44] and Visual Genome [37].

The '3M' corpus [87,47] contains image-text pairs from the Conceptual Captions dataset [62]. The '4M' corpus [12,43,22] consists of the COCO [44], Visual Genome [37], Conceptual Captions [62], and SBU Captions [52] image-text pairs. Downstream task post-processing and evaluation. We detail the postprocessing and downstream task evaluation in UniTAB inference. The first step shared among different tasks is to extract text, box, and word-box alignment predictions from the unified output sequence, as visualized in the main paper's Figure 3. We then use the extracted outputs for downstream task evaluations. We next detail the evaluation process of specific downstream tasks. 1). Grounded captioning. We use the extracted text, box, and alignment predictions to compute caption and grounding evaluations following the standard benchmark [86]. 2). Phrase grounding. We require the model to repeat the input text query and ground boxes as box tokens inline in the output sequence. For phrase grounding, the model needs to predict object boxes and align the box with words in the input text query. Instead of separately predicting the alignments between predicted boxes and input words [34,42], UniTAB repeats the input text and extracts alignments with the &lt;obj&gt; token from the unified output sequence. If the repeated text output is wrong, the alignment will be disarrayed, thus leading to wrong phrase grounding predictions. 3). Referring expression comprehension. Since the referring expression comprehension task [80,51] does not require the alignment prediction, we take the first predicted box in the output sequence as the final grounding prediction. 4). COCO captioning and VQA. We use the extracted text outputs for final evaluations ( i.e ., captioning metrics for COCO and exact match for VQA accuracy).

## B Ablation Studies on Decoding Design

In this section, we present ablation studies on UniTAB decoder and output sequence design, starting with ' &lt;obj&gt; token,' 'decoder type embedding,' and 'number of object tokens.' For these three ablation studies on the decoder, we initialize single-modality and transformer encoders with pre-trained UniTAB weights, and finetune model variants that have different decoder designs on the experimented task. We then discuss different inference-time 'decoding sampling method,' and the experiment on 'decoding syntactic restrictions.'

Table 9. The detailed experiment settings of UniTAB.

| Hyper-parameter                                                                    | Value                                      |
|------------------------------------------------------------------------------------|--------------------------------------------|
| (a) Optimizer hyper-parameters optimizer base learning rate backbone learning rate | AdamW [46] 1e-4 2e-5 *0.1 for final 5 1e-4 |
| learning rate schedule weight decay                                                | Step epochs                                |
| batch size                                                                         | 64                                         |
| pre-training epochs                                                                | 40                                         |
| multi-task finetuning epochs                                                       | 20                                         |
| task-specific finetuning epochs                                                    | 20                                         |
| exp. moving average                                                                | 0.9998                                     |
| (b) Model hyper-parameters encoder layer number encoder hidden size                | 6 256                                      |
| encoder intermediate size                                                          | 2048                                       |
| decoder layer number                                                               | 6                                          |
| encoder head number                                                                | 8                                          |
| decoder hidden size                                                                | 256                                        |
| decoder intermediate size                                                          | 2048                                       |
| decoder head number                                                                | 8                                          |
| max input words                                                                    | 256 0 × W 0 32                             |
| input visual tokens max decoding steps                                             | H 32 [25] 256                              |
| number of bins                                                                     | 200                                        |
| augmentation                                                                       |                                            |
| image size encoder vocab size                                                      | RandomResizedCrop [8] 800-1333             |
|                                                                                    | 50265 (RoBERTa [45])                       |
| decoder vocab size                                                                 |                                            |
|                                                                                    | 50265+200+2=50467                          |

&lt; obj &gt; token. UniTAB's special &lt;obj&gt; token naturally represents the word-box alignments in the output sequence. In addition to indicating the alignments, we examine if &lt;obj&gt; simplifies the sequence prediction and thus improves the model performance. The referring expression comprehension task requires a single box output and does not need the word-box alignment. Thus, we could remove the &lt;obj&gt; token while still being able to perform the task. Table 10 shows the experiments on the Refcocog dataset [51]. The UniTAB Separate baseline inserts a pair of &lt;obj&gt; and &lt; \ obj &gt; tokens before and after a word-box token segment. We experiment with removing the &lt; \ obj &gt; token, or both special tokens. We observe an around 1% accuracy improvement by adding &lt;obj&gt; tokens.

Decoder type embedding. Table 11 shows the ablation study on decoder type embedding, i.e ., whether to use type embedding to differentiate text and box tokens. We experiment with the following variants of decoder embedding [29,78]. The UniTAB Separate baseline does not use type embedding. ' &lt; obj &gt; as text/box tokens' uses two type embedding to differentiate text and box tokens, where the &lt; obj &gt; token is tagged as text or box token. ' &lt; obj &gt; as a third type' introduces an extra type embedding specialized for &lt; obj &gt; and &lt; \ obj &gt; . We experiment on the Refcocog [51] and Flickr30k Entities [54] grounding tasks. We empiri- cally observe that the decoder type embedding has no major influence on model performance, and thus do not use type embedding in UniTAB.

Table 10. Ablation study of the &lt; obj &gt; token on the Refcocog [51] dataset.

Table 11. Ablation study of the decoder type embedding. We experiment on the Refcocog [51] and Flickr30k Entities [54] grounding tasks.

| UniTAB Separate         |   Refcocog |
|-------------------------|------------|
| Baseline                |      80.23 |
| w/o < \ obj >           |      79.41 |
| w/o < obj > , < \ obj > |      79.31 |

Fig. 4. Ablation studies on the box token number. We experiment on the Refcocog [51] and Flickr [54] grounding tasks.

| UniTAB Separate          |   Refcocog |   Flickr |
|--------------------------|------------|----------|
| Baseline (w/o type emd.) |      80.23 |    78.92 |
| < obj > as text tokens   |      80.47 |    78.65 |
| < obj > as box tokens    |      80.39 |    79.40 |
| < obj > as a third type  |      80.54 |    78.83 |

<!-- image -->

<!-- image -->

Fig. 5. Ablation studies on decoding sampling method on the phrase grounding task.

Number of object tokens. Figure 4 shows the influence of object token number on the grounding performance. We observe a steady performance when the object token number is large enough for a dataset to avoid quantization error. The token number is around 200 for the experimented VL datasets.

Decoding sampling method. Figure 5 shows the ablation study of the decoding sampling method on Flickr phrase grounding [54]. Compared with the simple argmax decoding sampling, we observe a marginal improvement from nucleus sampling [27,10]. The improvement from nucleus sampling is smaller on UniTAB's experimented VL tasks, compared with previous explorations on object detection [10]. We suspect that the smaller gain is due to the difference in target sequences. Specifically, object detection [10] has multiple correct decoding sequences, as object order doesn't matter in object detection outputs. In contrast, UniTAB only has one fixed decoding target of the constructed text+box sequence. Thus, the diversity brought by nucleus sampling helps less in UniTAB.

Decoding syntactic restriction. We apply no decoding syntactic restriction in UniTAB training. We scan UniTAB predictions for two types of failure cases that break the syntactic restrictions in output decoding sequences: (1) before &lt; \ obj&gt; there are not exactly four consecutive box tokens; (2) &lt;obj&gt; , &lt; \ obj&gt; tokens are followed by box tokens, or are not paired. We scan the Refcocog grounding and Flickr grounded captioning predictions generated by UniTAB Pre-finetuning , and COCO captions generated by UniTAB Shared (for generalized grounded captioning in Figure 3 (d)). We observe zero syntactic failure cases in all scanned outputs, implying that the decoding token type pattern is easy to learn.

Table 12. UniTAB pre-training with additional bounding box annotations. 'Separate box ' adopts the extra bounding box annotations from the COCO [44], VG [37], Objects365 [61], OpenImages [38] object detection datasets.

| UniTAB       | Visual grounding   | Visual grounding   | Visual grounding   | Visual grounding   | Grounded caption   | Grounded caption   | COCO       | VQAv2   |
|--------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|------------|---------|
| UniTAB       | Refcoco            | Refcoco+           | Refcocog           | Flickr             | Cider              | F1 all             | test-Cider | KP-test |
| Separate     | 86.32              | 78.70              | 79.96              | 79.39              | 65.6               | 11.46              | 119.3      | 66.6    |
| Separate box | 88.27              | 80.98              | 83.78              | 81.90              | 70.0               | 13.46              | 120.7      | 68.4    |

We then incorporate these two syntactic restrictions into the model training, and examine if the restrictions ease the training and improve model performance. Specifically, we compute the softmax language modeling loss over a subset of all tokens in applicable decoding positions, such as masking out box logits after &lt;obj&gt; . We experiment on Refcoco and Flickr grounded captioning, based on UniTAB Separate . We empirically observe that the syntactic restriction has no major influence on the model performance, with +0 . 9 accuracy gain on Refcoco and a slight drop on Flickr grounded captioning ( -0 . 2 CIDEr, -0 . 15 F 1 all ).

## C Discussions

Pre-training with additional box annotations. In addition to the '200K' pre-training corpus and the additional image-text data [62,52] introduced in the main paper, we further explore using additional box annotations with no caption texts for UniTAB pre-training. We aggregate object detection annotations from COCO [44], VG [37], Objects365 [61], and OpenImages [38]. Each sample is an image with object box and class annotations. For pre-training, we randomly select up to 32 objects and shuffle the object order. We concatenate the object class name as the input text, and train the model to generate the text+box sequence to ground the selected objects. We refer to UniTAB Separate with those extra box annotations as 'Separate box .'

Table 12 shows the experiment results of adding additional box annotations. On VL tasks that require box prediction, such as the visual grounding task and the grounding evaluation in grounded captioning, 'Separate box ' consistently outperforms UniTAB Separate on the grounding accuracy and grounded captioning F1 score. More interestingly, we empirically observe that extra box annotations could also help the text output quality. For example, 'Separate box ' improves grounded captioning CIDEr score from 65 . 6 to 70 . 0 , COCO captioning CIDEr score from 119 . 3 to 120 . 7 , and VQA accuracy from 66 . 6% to 68 . 4% .

Multi-task finetuning with prefix. In the main paper, we discuss the effectiveness of multi-task finetuning, which gathers training data from all downstream tasks and learns a single set of parameters for different VL tasks. By unifying all considered downstream tasks as a sequence generation problem, a single UniTAB Shared model could perform well on different tasks, meanwhile being parameter efficient and showing promise in zero-shot generalization.

Table 13. Experiment results of UniTAB Prefix that adds task-specific prefix in multitask finetuning.

| Method                | #Pre-   | Visual grounding   | Visual grounding   | Visual grounding   | Visual grounding   | Grounded caption   | Grounded caption   | COCO       | VQAv2   |
|-----------------------|---------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|------------|---------|
| Method                | train   | Refcoco            | Refcoco+           | Refcocog           | Flickr             | Cider              | F1 all             | test-Cider | KP-test |
| MDETR [34]            | 200K    | 86.75              | 79.52              | 81.64              | 83.8               | -                  | -                  | -          | -       |
| UNITER [12]           | 4M      | 81.24              | 75.31              | 74.31              | -                  | -                  | -                  | -          | 70.5    |
| GVD [86]              | -       | -                  | -                  | -                  | -                  | 62.3               | 7.55               | -          | -       |
| VL-T5 [13]            | 180K    | -                  | -                  | 71.2               | -                  | -                  | -                  | 116.5      | 67.9    |
| OSCAR [43]            | 4M      | -                  | -                  | -                  | -                  | -                  | -                  | 123.7      | -       |
| UniTAB Shared-Scratch | None    | 82.06              | 70.72              | 73.39              | 65.67              | 61.1               | 7.85               | 111.8      | 63.1    |
| UniTAB Prefix-Scratch | None    | 82.38              | 70.96              | 75.43              | 69.58              | 62.1               | 8.51               | 112.8      | 64.3    |
| UniTAB Shared         | 200K    | 88.50              | 80.98              | 84.46              | 79.23              | 63.4               | 9.18               | 115.8      | 66.6    |
| UniTAB Prefix         | 200K    | 87.60              | 79.72              | 83.41              | 80.13              | 62.4               | 10.54              | 115.6      | 66.0    |

Table 14. VQA-CE [16] robustness analyses.

| Accuracy(%)     | UpDown        | UniTAB        |
|-----------------|---------------|---------------|
| Overall         | 63.52 (+0.00) | 70.78 (+7.26) |
| Counterexamples | 33.91 (+0.00) | 43.67 (+9.76) |
| Easy            | 76.69 (+0.00) | 82.86 (+6.17) |

Table 15. Gender error analyses [26].

| Error rate(%) COCO-Bias COCO-Balanced   |   Error rate(%) COCO-Bias COCO-Balanced |   Error rate(%) COCO-Bias COCO-Balanced |
|-----------------------------------------|-----------------------------------------|-----------------------------------------|
| BaselineFT                              |                                   12.83 |                                   19.30 |
| Balanced                                |                                   12.85 |                                   18.30 |
| UpWeight                                |                                   13.56 |                                   16.30 |
| Equalizer                               |                                    7.02 |                                    8.10 |
| UniTAB                                  |                                    9.87 |                                    9.21 |

One variant of UniTAB Shared is to add a task-specific input text string to identify the task for each sample [13], such as 'visual grounding:'. The extra input text string is known as the prefix. We experiment with a variant of UniTAB multi-task finetuning with prefix, namely UniTAB Prefix . UniTAB Prefix adds a task-specific prefix at the beginning of each input text string, e.g ., 'Visual grounding: the coffee mug next to the plate.' We use the task name as the prefix, i.e ., 'visual grounding:', 'phrase grounding:', 'grounded captioning:', 'image captioning:', 'question answering:', etc . We then train the model with multitask finetuning, the same as UniTAB Shared . Table 13 compares UniTAB Prefix with UniTAB Shared . We observe a comparable performance with and without prefix on the experimented tasks and datasets.

Robustness and bias analyses. We conduct robustness and bias analyses to better understand the limitation of UniTAB. Tables 14,15 show initial robustness and bias analyses. We retrain UniTAB on the splits (VQA-train, COCO-14) used in the established analysis setups [26,16]. In Table 14, we follow VQA-CE [16] and compare the gain over the UpDown baseline on two subsets. UniTAB achieves a larger gain on 'counterexamples' ( +9 . 76% ) compared with 'easy' ( +6 . 17% ), indicating better robustness against shortcuts compared with the UpDown baseline, as discussed in VQA-CE [16]. Table 15 evaluates gender bias with error rate [26]. UniTAB achieves a lower error rate than general captioning models ( cf ., Baseline-FT of 19 . 30% vs . UniTAB of 9 . 21% ), and is only slightly worse than the specialized method Equalizer [26]. We hypothesize that the reasonable robustness and bias performance is related to UniTAB's grounded training,

583865081&lt;sep&gt;&lt;obj&gt; A Ġman

&lt;106&gt; &lt;82&gt; &lt;152&gt; &lt;163&gt; &lt;/obj&gt;

Ġin &lt;obj&gt; Ġa Ġblack Ġcoat &lt;106&gt;

&lt;93&gt; &lt;152&gt; &lt;128&gt; &lt;/obj&gt; Ġand

&lt;obj&gt; Ġblack Ġpants &lt;110&gt; &lt;119&gt;

&lt;142&gt; &lt;158&gt; &lt;/obj&gt; Ġis Ġwalking

Ġdown &lt;obj&gt; Ġthe Ġstreet &lt;0&gt;

&lt;93&gt; &lt;199&gt; &lt;199&gt; &lt;/obj&gt; Ġtalking

Ġon &lt;obj&gt; Ġhis Ġcellphone &lt;133&gt;

&lt;91&gt; &lt;139&gt; &lt;97&gt; &lt;/obj&gt; Ġ.&lt;sep&gt;A

man in a black coat and black pants

is walking down the street talking

on his cellphone .

85519095&lt;sep&gt;&lt;obj&gt; A Ġyoung

Ġboy &lt;31&gt; &lt;28&gt; &lt;146&gt; &lt;152&gt;

&lt;/obj&gt; Ġin &lt;obj&gt; Ġa Ġblue Ġshirt

&lt;40&gt; &lt;54&gt; &lt;98&gt; &lt;95&gt; &lt;/obj&gt; Ġand

&lt;obj&gt; Ġhelmet &lt;56&gt; &lt;28&gt; &lt;103&gt;

&lt;56&gt; &lt;/obj&gt; Ġis Ġswinging &lt;obj&gt;

Ġa Ġbat &lt;32&gt; &lt;54&gt; &lt;46&gt; &lt;62&gt;

&lt;/obj&gt; Ġat &lt;obj&gt; Ġa Ġbaseball

&lt;167&gt; &lt;77&gt; &lt;183&gt; &lt;87&gt; &lt;/obj&gt;

Ġ.&lt;sep&gt;A young boy in a blue shirt

and helmet is swinging a bat at a

baseball .

139245992&lt;sep&gt;&lt;obj&gt; A Ġman

&lt;80&gt; &lt;51&gt; &lt;175&gt; &lt;128&gt; &lt;/obj&gt;

Ġin &lt;obj&gt; Ġa Ġtan Ġshirt &lt;92&gt;

&lt;58&gt; &lt;166&gt; &lt;87&gt; &lt;/obj&gt; Ġis

Ġthrowing &lt;obj&gt; Ġa Ġred Ġball

&lt;76&gt; &lt;81&gt; &lt;84&gt; &lt;87&gt; &lt;/obj&gt; Ġon

&lt;obj&gt; Ġthe Ġbeach &lt;0&gt; &lt;105&gt;

&lt;199&gt; &lt;199&gt; &lt;/obj&gt; Ġ.&lt;sep&gt;A

man in a tan shirt is throwing a red

ball on the beach .

202175131&lt;sep&gt;&lt;obj&gt; A Ġman

&lt;52&gt; &lt;83&gt; &lt;114&gt; &lt;161&gt; &lt;/obj&gt; Ġin

&lt;obj&gt; Ġa Ġwhite Ġshirt &lt;68&gt; &lt;99&gt;

&lt;113&gt; &lt;132&gt; &lt;/obj&gt; Ġand &lt;obj&gt;

Ġblue Ġhelmet &lt;77&gt; &lt;83&gt; &lt;99&gt;

&lt;98&gt; &lt;/obj&gt; Ġis Ġclimbing &lt;obj&gt;

Ġa Ġrock &lt;1&gt; &lt;158&gt; &lt;199&gt; &lt;199&gt;

&lt;/obj&gt; Ġ.&lt;sep&gt;A man in a white

shirt and blue helmet is climbing a

rock .

400598822&lt;sep&gt;&lt;obj&gt; A Ġman

&lt;29&gt; &lt;11&gt; &lt;109&gt; &lt;179&gt; &lt;/obj&gt; Ġin

&lt;obj&gt; Ġa Ġblue Ġshirt &lt;30&gt; &lt;11&gt;

&lt;88&gt; &lt;83&gt; &lt;/obj&gt; Ġis Ġcooking

&lt;obj&gt; Ġfood &lt;95&gt; &lt;131&gt; &lt;148&gt;

&lt;176&gt; &lt;/obj&gt; Ġon &lt;obj&gt; Ġa Ġlarge

Ġmetal Ġpan &lt;81&gt; &lt;126&gt; &lt;155&gt;

&lt;191&gt; &lt;/obj&gt; Ġ.&lt;sep&gt;A man in a

blue shirt is cooking food on a large

metal pan .

489865145&lt;sep&gt;&lt;obj&gt; A Ġyoung

Ġman &lt;13&gt; &lt;3&gt; &lt;93&gt; &lt;188&gt;

&lt;/obj&gt; Ġwearing &lt;obj&gt;

Ġsunglasses &lt;54&gt; &lt;13&gt; &lt;70&gt;

&lt;27&gt; &lt;/obj&gt; Ġand &lt;obj&gt; Ġa Ġblue

Ġshirt &lt;27&gt; &lt;23&gt; &lt;74&gt; &lt;97&gt;

&lt;/obj&gt; Ġis Ġwalking Ġon &lt;obj&gt;

Ġthe Ġsidewalk &lt;0&gt; &lt;185&gt; &lt;199&gt;

&lt;199&gt; &lt;/obj&gt; Ġ.&lt;sep&gt;A young man

wearing sunglasses and a blue shirt

is walking on the sidewalk .

1341077576&lt;sep&gt;&lt;obj&gt; A Ġman

&lt;74&gt; &lt;108&gt; &lt;109&gt; &lt;165&gt; &lt;/obj&gt;

Ġin &lt;obj&gt; Ġa Ġred Ġshirt &lt;74&gt;

&lt;113&gt; &lt;108&gt; &lt;138&gt; &lt;/obj&gt; Ġand

&lt;obj&gt; Ġblue Ġjeans &lt;75&gt; &lt;132&gt;

&lt;105&gt; &lt;161&gt; &lt;/obj&gt; Ġis Ġstanding

Ġin Ġfront Ġof &lt;obj&gt; Ġa Ġred

Ġdoor &lt;32&gt; &lt;79&gt; &lt;134&gt; &lt;165&gt;

&lt;/obj&gt; Ġwith &lt;obj&gt; Ġa Ġdog &lt;0&gt;

&lt;158&gt; &lt;46&gt; &lt;192&gt; &lt;/obj&gt; Ġon

&lt;obj&gt; Ġthe Ġstreet &lt;0&gt; &lt;161&gt;

&lt;199&gt; &lt;199&gt; &lt;/obj&gt; Ġ.&lt;sep&gt;A

Fig. 6. Additional qualitative results from UniTAB Shared on captioning tasks.

<!-- image -->

332&lt;sep&gt;&lt;obj&gt; A Ġwoman &lt;84&gt;

&lt;9&gt; &lt;161&gt; &lt;181&gt; &lt;/obj&gt; Ġin &lt;obj&gt;

Ġa Ġblue Ġdress &lt;84&gt; &lt;29&gt;

359&lt;sep&gt;&lt;obj&gt; A Ġdog &lt;48&gt; &lt;5&gt;

&lt;162&gt; &lt;128&gt; &lt;/obj&gt; Ġis Ġstanding Ġin Ġfront Ġof &lt;obj&gt; Ġa Ġtelevision &lt;0&gt; &lt;35&gt; &lt;50&gt; &lt;111&gt; &lt;/obj&gt; Ġ.&lt;sep&gt;A woman in a blue dress is standing in front of a television . &lt;185&gt; &lt;199&gt; &lt;/obj&gt; Ġwith &lt;obj&gt; Ġa Ġpink Ġbow Ġtie &lt;51&gt; &lt;5&gt; &lt;186&gt; &lt;144&gt; &lt;/obj&gt; Ġon &lt;obj&gt; Ġits Ġhead &lt;50&gt; &lt;5&gt; &lt;186&gt; &lt;148&gt; &lt;/obj&gt; Ġis Ġstanding Ġon &lt;obj&gt; Ġa Ġporch &lt;0&gt; &lt;103&gt; &lt;199&gt; &lt;199&gt; &lt;/obj&gt; Ġ.&lt;sep&gt;A dog with a pink bow tie on its head is standing on a porch . 1304&lt;sep&gt;&lt;obj&gt; A Ġman &lt;3&gt; &lt;6&gt; &lt;196&gt; &lt;199&gt; &lt;/obj&gt; Ġin &lt;obj&gt; Ġa Ġyellow Ġjacket &lt;50&gt; &lt;56&gt; &lt;196&gt; &lt;199&gt; &lt;/obj&gt; Ġand &lt;obj&gt; Ġblack Ġhat &lt;92&gt; &lt;6&gt; &lt;144&gt; &lt;35&gt; &lt;/obj&gt; Ġis Ġholding &lt;obj&gt; Ġa Ġlarge Ġfish &lt;26&gt; &lt;113&gt; &lt;191&gt; &lt;173&gt; &lt;/obj&gt; Ġ.&lt;sep&gt;A man in a yellow jacket and black hat is holding a which better binds visual concepts with text words. Despite the reasonable performance on the standard analyses, building robust and unbiased models remains a challenging problem and could be further improved.

large fish .

## D Qualitative Results

In this section, we present additional qualitative results made by UniTAB Shared . Westart with the captioning tasks in Figure 6. Figure 6 (a) presents the grounded captioning results on Flickr30k Entities, where the predictions are evaluated by both the caption and grounding metrics. UniTAB performs well in both generating captions and grounding noun phrases to image regions. For captioning, the model generates a smooth and accurate image description, and properly includes attribute words to produce an informative caption, e.g ., 'young boy' and 'blue shirt' in the top left example. UniTAB is also capable of providing a comprehensive description of the scene. For example, in the bottom right sub-figure of (a),

607&lt;sep&gt;&lt;obj&gt; A Ġyoung Ġboy

&lt;38&gt; &lt;0&gt; &lt;186&gt; &lt;199&gt; &lt;/obj&gt; Ġin

&lt;obj&gt; Ġa Ġred Ġshirt &lt;43&gt; &lt;50&gt;

&lt;186&gt; &lt;142&gt; &lt;/obj&gt; Ġis Ġholding

&lt;obj&gt; Ġa Ġred Ġbowl &lt;36&gt; &lt;132&gt;

&lt;155&gt; &lt;198&gt; &lt;/obj&gt; Ġwith &lt;obj&gt;

Ġa Ġplate Ġof Ġfood &lt;36&gt; &lt;132&gt;

&lt;155&gt; &lt;198&gt; &lt;/obj&gt; Ġ.&lt;sep&gt;A

young boy in a red shirt is holding a

red bowl with a plate of food .

4428&lt;sep&gt;&lt;obj&gt; A Ġdog &lt;54&gt;

&lt;12&gt; &lt;138&gt; &lt;193&gt; &lt;/obj&gt; Ġcarries

&lt;obj&gt; Ġa Ġstick &lt;110&gt; &lt;55&gt; &lt;152&gt;

&lt;70&gt; &lt;/obj&gt; Ġin &lt;obj&gt; Ġits Ġmouth

&lt;117&gt; &lt;53&gt; &lt;132&gt; &lt;74&gt; &lt;/obj&gt;

Ġ.&lt;sep&gt;A dog carries a stick in its

mouth .

17572&lt;sep&gt;&lt;obj&gt; Three Ġpeople

&lt;73&gt; &lt;53&gt; &lt;124&gt; &lt;125&gt; &lt;/obj&gt;

Ġare Ġsitting Ġon &lt;obj&gt; Ġa

Ġbench &lt;74&gt; &lt;77&gt; &lt;128&gt; &lt;126&gt;

&lt;/obj&gt; Ġby &lt;obj&gt; Ġa Ġriver &lt;0&gt;

&lt;41&gt; &lt;199&gt; &lt;91&gt; &lt;/obj&gt;

Ġ.&lt;sep&gt;Three people are sitting on

a bench by a river .

34818&lt;sep&gt;&lt;obj&gt; A Ġman &lt;27&gt;

&lt;25&gt; &lt;126&gt; &lt;185&gt; &lt;/obj&gt; Ġin

&lt;obj&gt; Ġa Ġblack Ġsuit &lt;27&gt; &lt;50&gt;

&lt;68&gt; &lt;133&gt; &lt;/obj&gt; Ġand &lt;obj&gt;

Ġblack Ġshoes &lt;89&gt; &lt;149&gt; &lt;126&gt;

&lt;185&gt; &lt;/obj&gt; Ġsits Ġon &lt;obj&gt; Ġa

Ġpark Ġbench &lt;9&gt; &lt;81&gt; &lt;107&gt;

&lt;187&gt; &lt;/obj&gt; Ġ.&lt;sep&gt;A man in a

black suit and black shoes sits on a

park bench .

166&lt;sep&gt;&lt;obj&gt; Three Ġmen &lt;0&gt;

&lt;58&gt; &lt;199&gt; &lt;199&gt; &lt;/obj&gt; Ġare

Ġstanding Ġon &lt;obj&gt; Ġa Ġboat

&lt;0&gt; &lt;159&gt; &lt;199&gt; &lt;199&gt; &lt;/obj&gt;

Ġwith &lt;obj&gt; Ġa Ġlarge Ġfish &lt;89&gt;

&lt;103&gt; &lt;184&gt; &lt;199&gt; &lt;/obj&gt;

Ġ.&lt;sep&gt;Three men are standing on

a boat with a large fish .

33&lt;sep&gt;&lt;obj&gt; A Ġman

8&gt; &lt;72&gt; &lt;115&gt; &lt;/obj&gt; Ġin

blue Ġjeans &lt;25&gt; &lt;66&gt;

5&gt; &lt;/obj&gt; Ġand &lt;obj&gt; Ġa

y Ġhat &lt;41&gt; &lt;38&gt; &lt;64&gt;

bj&gt; Ġis Ġriding &lt;obj&gt; Ġa

&lt;11&gt; &lt;48&gt; &lt;81&gt; &lt;175&gt;

around &lt;obj&gt; Ġa Ġred Ġ,

, Ġand Ġblue Ġbarrel

124&gt; &lt;169&gt; &lt;189&gt; &lt;/obj&gt;

A man in blue jeans and a

hat is riding a horse around

hite , and blue barrel .

52737&lt;sep&gt;&lt;obj&gt; A Ġwoman

&lt;75&gt; &lt;83&gt; &lt;179&gt; &lt;/obj&gt; Ġin

Ġa Ġburg undy Ġcoat &lt;31&gt;

&lt;83&gt; &lt;138&gt; &lt;/obj&gt; Ġexits

Ġa ĠJoe &lt;31&gt; &lt;21&gt; &lt;173&gt;

&lt;/obj&gt; Ġ' s Ġwith &lt;obj&gt; Ġa

t &lt;47&gt; &lt;115&gt; &lt;121&gt; &lt;197&gt;

&gt; Ġfull Ġof &lt;obj&gt; Ġgroceries

&lt;107&gt; &lt;116&gt; &lt;161&gt; &lt;/obj&gt;

p&gt;A woman in a burgundy

exits a Joe 's with a cart full of

ries .

69240&lt;sep&gt;&lt;obj&gt; A Ġperson

&gt; &lt;53&gt; &lt;185&gt; &lt;179&gt; &lt;/obj&gt;

ring &lt;obj&gt; Ġsk is &lt;130&gt;

&gt; &lt;198&gt; &lt;184&gt; &lt;/obj&gt;

ing Ġat &lt;obj&gt; Ġframed

ures &lt;15&gt; &lt;108&gt; &lt;70&gt; &lt;169&gt;

&gt; Ġset Ġup Ġin &lt;obj&gt; Ġthe

w &lt;0&gt; &lt;117&gt; &lt;199&gt; &lt;199&gt;

&gt; Ġ.&lt;sep&gt;A person wearing

ooking at framed pictures set

the snow .

2852&lt;sep&gt;&lt;obj&gt; Hispanic

an &lt;64&gt; &lt;12&gt; &lt;130&gt; &lt;188&gt;

&gt; Ġwearing &lt;obj&gt; Ġa Ġred

id Ġshirt &lt;78&gt; &lt;44&gt; &lt;130&gt;

&gt; &lt;/obj&gt; Ġworks Ġon Ġsewing

Ġan Ġarticle Ġof Ġclothing

&lt;71&gt; &lt;115&gt; &lt;91&gt; &lt;/obj&gt;

p&gt;Hispanic woman wearing a

laid shirt works on sewing an

of clothing .

99&lt;sep&gt;&lt;obj&gt; A Ġman

5&gt; &lt;178&gt; &lt;199&gt; &lt;/obj&gt;

j&gt; Ġa Ġjumps uit &lt;117&gt;

79&gt; &lt;199&gt; &lt;/obj&gt; Ġand

hat &lt;133&gt; &lt;16&gt; &lt;163&gt;

bj&gt; Ġtends Ġto &lt;obj&gt; Ġa

sp ool Ġof Ġrope &lt;17&gt;

13&gt; &lt;199&gt; &lt;/obj&gt;

A man in a jumpsuit and

s to a large spool of rope .

&lt;sep&gt;&lt;obj&gt; A Ġyoung

&lt;36&gt; &lt;161&gt; &lt;185&gt;

essed Ġin &lt;obj&gt; Ġblue

Ġtr unks &lt;23&gt; &lt;85&gt;

&gt; &lt;/obj&gt; Ġand &lt;obj&gt; Ġa

Ġjacket &lt;33&gt; &lt;46&gt; &lt;96&gt;

&gt; Ġwaters ki ing Ġin

blue Ġlake &lt;0&gt; &lt;45&gt;

9&gt; &lt;/obj&gt; Ġ.&lt;sep&gt;A

dressed in blue

trunks and a red life

rskiing in a blue lake .

p&gt;&lt;obj&gt; A Ġman

49&gt; &lt;148&gt; &lt;/obj&gt;

rd &lt;obj&gt; Ġa

ice &lt;130&gt; &lt;139&gt;

/obj&gt; Ġin &lt;obj&gt; Ġa

2&gt; &lt;199&gt; &lt;199&gt;

bj&gt; Ġanother Ġman

09&gt; &lt;160&gt; &lt;/obj&gt;

sep&gt;A man jumps

g device in a lake as

atches .

84712&lt;sep&gt;129894&lt;sep&gt;&lt;obj&gt;

yellow Ġsleeve Ġguy &lt;0&gt; &lt;87&gt;

&lt;81&gt; &lt;196&gt; &lt;/obj&gt;&lt;sep&gt;yellow

sleeve guy

84712&lt;sep&gt;129896&lt;sep&gt;&lt;obj&gt;

blue Ġjacket &lt;22&gt; &lt;18&gt; &lt;71&gt;

&lt;111&gt; &lt;/obj&gt;&lt;sep&gt;blue jacket

84712&lt;sep&gt;129898&lt;sep&gt;&lt;obj&gt;

blue Ġsk ier &lt;22&gt; &lt;43&gt; &lt;173&gt;

&lt;181&gt; &lt;/obj&gt;&lt;sep&gt;blue skier

84712&lt;sep&gt;129900&lt;sep&gt;&lt;obj&gt;

main Ġcenter Ġperson &lt;22&gt; &lt;43&gt;

&lt;173&gt; &lt;181&gt; &lt;/obj&gt;&lt;sep&gt;main

center person the caption consists of the foreground object and its detailed attributes 'man in red shirt and blue jeans,' scene descriptions 'a red door' and 'on the street,' and the nearby object 'a dog.' The model also performs well in grounding. Noticeably, UniTAB performs well on grounding and describing tiny objects, e.g ., the 'a bat' and 'a baseball' in the top left example and the 'a red ball' in the bottom left example.

Fig. 7. Additional qualitative results from UniTAB Shared on grounding tasks.

<!-- image -->

Figure 6 (b) shows UniTAB's prediction on the MSCOCO image captioning task. With the same inputs as Flickr30k grounded captioning, UniTAB learns to transfer the grounded captioning ability learned on Flirckr30k Entities to MSCOCO, although COCO captioning does not have grounding annotations. For evaluation, we extract the text tokens and compute the standard COCO captioning metrics [53,19,3,69]. We note that UniTAB achieves comparable caption performance to the state of the art, and meanwhile being capable of grounding all noun phrases in the caption. As shown in Figure 6(b), UniTAB generates an informative caption and accurately grounds all noun phrases in the caption to visual regions. Such grounded captioning ability is important for reducing ob-

358706&lt;sep&gt;255813&lt;sep&gt;&lt;obj&gt;

man &lt;166&gt; &lt;11&gt; &lt;199&gt; &lt;147&gt;

&lt;/obj&gt; Ġlean Ġon

Ġbarrier&lt;sep&gt;man lean on barrier

358706&lt;sep&gt;255815&lt;sep&gt;&lt;obj&gt;

man &lt;166&gt; &lt;11&gt; &lt;198&gt; &lt;147&gt;

&lt;/obj&gt; Ġby Ġby Ġbarric ade Ġhipp

y Ġblack Ġshirt Ġshorts&lt;sep&gt;man

by by barricade hippy black shirt

shorts

358706&lt;sep&gt;255817&lt;sep&gt;&lt;obj&gt;

driver &lt;90&gt; &lt;44&gt; &lt;146&gt; &lt;174&gt;

&lt;/obj&gt; Ġof Ġmotorcycle&lt;sep&gt;driver

of motorcycle

358706&lt;sep&gt;255819&lt;sep&gt;&lt;obj&gt;

girl &lt;127&gt; &lt;36&gt; &lt;163&gt; &lt;150&gt;

&lt;/obj&gt; Ġon Ġbike&lt;sep&gt;girl on bike

358706&lt;sep&gt;255821&lt;sep&gt;&lt;obj&gt;

girl &lt;127&gt; &lt;36&gt; &lt;163&gt; &lt;150&gt;

&lt;/obj&gt; Ġback Ġof Ġbike&lt;sep&gt;girl

back of bike

395791&lt;sep&gt;255127&lt;sep&gt;&lt;obj&gt;

container &lt;116&gt; &lt;36&gt; &lt;177&gt;

&lt;108&gt; &lt;/obj&gt; Ġwith Ġbrown Ġgo o

Ġand Ġfake

Ġorange&lt;sep&gt;container with brown

goo and fake orange

395791&lt;sep&gt;255129&lt;sep&gt;&lt;obj&gt;

car rots &lt;120&gt; &lt;104&gt; &lt;187&gt; &lt;186&gt;

&lt;/obj&gt;&lt;sep&gt;carrots

395791&lt;sep&gt;255131&lt;sep&gt;&lt;obj&gt;

section &lt;2&gt; &lt;106&gt; &lt;123&gt; &lt;187&gt;

&lt;/obj&gt; Ġwith

Ġstrawberries&lt;sep&gt;section with

strawberries

395791&lt;sep&gt;255133&lt;sep&gt;&lt;obj&gt; st

raw berries &lt;2&gt; &lt;106&gt; &lt;124&gt;

&lt;188&gt; &lt;/obj&gt;&lt;sep&gt;strawberries ject hallucination [60], boosting the model's interpretability and fairness [60,26], and facilitating applications in robotics and human-computer interaction. We also visualize additional captioning examples on ImageNet [17]. We observe that UniTAB generalizes well onto the ImageNet images. The ImageNet caption and grounding predictions in Figure 6 (c) are of similar qualities as on Flickr30k Entities and MSCOCO.

Figure 7 shows UniTAB Shared 's predictions on grounding tasks. Figures 7(a,b) are from the Refcoco [80] and Refcocog [51] datasets, for the referring expression comprehension task. We observe that the model learns to identify different objects in the same image conditioned on different input queries. For example, in Figure 7 (a) , the targets of 'yellow sleeve guy' on the left and 'blue' skier in the background. Similarly, in Figure 7 (b) , UniTAB correctly differentiates the four people in the image. UniTAB also correctly localizes the head noun in a long referring expression and predicts the box on the corresponding phrase. For example, in Figure 7(b), grounding boxes are predicted on the words 'girl' and 'person,' instead of the entire query as in previous studies [79,77,18]. Another observation is that UniTAB usually predicts a single box in the output sequence for referring expression comprehension samples. For example, in the top left subfigure of Figure 7(b), the model only grounds the head noun 'girl' and does not generate a box for the remaining phrase like 'pink pants.' We conjecture that UniTAB learns to identify the referring expression comprehension task based on the input text ( e.g ., a short referring query vs . a complete sentence), and generates a single box when performing the task.

Figure 7 (c) shows the phrase grounding examples on the Flickr30k Entities dataset [54]. Phrase grounding requires the model to identify all noun phrases in a sentence and ground them to corresponding image regions. UniTAB correctly grounds all types of phrases referred to in the sentence, including foreground objects 'person' and 'woman,' smaller background objects 'skies' in the top left example and 'another man' in the bottom left example, and scene regions 'the snow,' 'a lake,' and 'a blue lake.' The model even correctly predicts challenging regions such as the 'trader joe's' logo in the top right sub-figure.