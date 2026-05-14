## Words or Vision: Do Vision-Language Models Have Blind Faith in Text?

Ailin Deng Tri Cao Zhirui Chen † Bryan Hooi †

National University of Singapore

{

## Abstract

Vision-Language Models (VLMs) excel in integrating visual and textual information for vision-centric tasks, but their handling of inconsistencies between modalities is underexplored. We investigate VLMs' modality preferences when faced with visual data and varied textual inputs in visioncentered settings. By introducing textual variations to four vision-centric tasks and evaluating ten Vision-Language Models (VLMs), we discover a 'blind faith in text' phenomenon: VLMs disproportionately trust textual data over visual data when inconsistencies arise, leading to significant performance drops under corrupted text and raising safety concerns. We analyze factors influencing this text bias, including instruction prompts, language model size, text relevance, token order, and the interplay between visual and textual certainty. While certain factors, such as scaling up the language model size, slightly mitigate text bias, others like token order can exacerbate it due to positional biases inherited from language models. To address this issue, we explore supervised fine-tuning with text augmentation and demonstrate its effectiveness in reducing text bias. Additionally, we provide a theoretical analysis suggesting that the blind faith in text phenomenon may stem from an imbalance of pure text and multi-modal data during training. Our findings highlight the need for balanced training and careful consideration of modality interactions in VLMs to enhance their robustness and reliability in handling multi-modal data inconsistencies.

## 1. Introduction

With the rise of Vision-Language Models (VLMs) [4, 9, 24], these models are increasingly applied in complex multi-modal tasks, such as retrieval-augmented generation (RAG) [41] and multi-modal agents [12, 17, 19], where they handle large, context-rich cross-modal inputs. In these practical scenarios, inconsistencies between visual and textual

† corresponding author

ailin,zhiruichen } @u.nus.edu, tricao2001vn@gmail.com

bhooi@comp.nus.edu.sg

Figure 1. Illustration of the 'Blind Faith in Text' phenomenon in Vision-Language Models (VLMs). These models demonstrate a strong tendency to trust textual data, when it is inconsistent with the visual data or even incorrect.

<!-- image -->

inputs are common, as additional textual data may be irrelevant or even misleading [35]. Despite their strong performance on vision-centric benchmarks [15, 44, 45], VLMs' capability to handle such inconsistencies remains underexplored. This gap motivates our study, as understanding and addressing VLMs' tendencies under these conditions is essential for their safe and reliable application in real-world, multi-modal contexts.

In this work, we explore an open but underexplored question: How do VLMs handle inconsistencies between visual and textual inputs ? This question drives us to investigate the following aspects:

1. Modality Preference : What modality do VLMs prefer when there are inconsistencies between vision and language data?
2. Robustness to Text Perturbation : Can these models maintain their performance on vision-centric tasks when faced with corrupted textual data?
3. Influencing Factors : What factors affect the modality preference in VLMs?

To address these questions, we construct a comprehensive benchmark by introducing textual variations to four vision-centric tasks and evaluate ten VLMs, including both proprietary and open-source models. Our findings reveal a phenomenon we term 'blind faith in text' : when inconsistencies arise between visual and textual inputs, VLMs tend to overly trust the textual data, even when it contradicts visual evidence. This text bias not only leads to significant performance degradation when the text is corrupted but also raises potential safety concerns in practical applications.

We further investigate this issue by examining factors that influence text bias:

- Instruction Prompts : While instructions can modestly adjust modality preference, their effectiveness is limited.
- Language Model Size : Scaling up the language model size slightly mitigates text bias, but the effect saturates in larger models.
- Text Relevance : The preference for textual data increases with text relevance.
- Token Order : Placing text tokens before image tokens exacerbates text bias, possibly due to positional biases inherited from language models.
- Uni-Modal Certainty : The interplay between visual and textual certainty influences modality preference.

To mitigate text bias, we explore supervised fine-tuning with text augmentation, demonstrating its effectiveness even with limited data. Additionally, we provide a theoretical analysis suggesting that the blind faith in text phenomenon may stem from an imbalance of pure text and multi-modal data during training, as VLMs are built upon large language models primarily trained on textual data. Our contributions † are summarized as follows:

- We uncover the 'blind faith in text' phenomenon, where VLMs prefer language data over visual data when inconsistencies occur in context.
- We confirm that this text bias leads to significant performance drops under text corruption, even in vision-centric tasks where VLMs typically excel.
- We identify key factors influencing text bias, including instruction prompts, language model size, text relevance, token order, and uni-modal certainty.
- Wedemonstrate that supervised fine-tuning with text augmentation effectively reduces text bias.
- We provide a theoretical analysis suggesting that the imbalance of pure text and multi-modal data during training contributes to the blind faith in text phenomenon.

## 2. Preliminaries

Given a model f vlm ( · ; θ ) parameterized by θ , a sample X := ( I, T, Q ) contains an image I , textual information T , and a question Q , with corresponding ground truth Y . We can obtain the answers under three conditions: (1) only given the image; (2) only given textual information; (3)

[† https://github.com/d-ailin/blind-faith-in-text](https://github.com/d-ailin/blind-faith-in-text)

given both modalities' information:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Generation Certainty. The response generation certainty can be estimated based on the length-normalized predictive likelihood of the response sequence. Formally:

<!-- formula-not-decoded -->

where ˆ Y i represents the i -th token of ˆ Y , and ˆ Y &lt;i are the tokens generated before i -th token in ˆ Y . By considering this certainty in each modality separately, we can compute the Uni-Modal Certainty to quantify the generation uncertainty in each modality. Specifically, it is computed given only the image or the text:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## 2.1. Text Variations

To comprehensively study the effect of text variations, we consider three types of variations: Match, Corruption, and Irrelevance cases given the question Q and the original ground-truth answer Y :

- Match. We denote T m as a matching text such that ( T m , Q ) has ground-truth Y , indicating the text provides sufficient and relevant information, allowing for answering the question correctly when only given the text.

̸

- Corruption. We denote corrupted text as T c such that ( T c , Q ) has ground-truth Y c = Y , indicating that the text is relevant and sufficient for answering, but leads to a different answer when relying only on the text.
- Irrelevance. We consider a text to be an irrelevant text T irr (i.e., T irr ⊥ ⊥ I, Q ), indicating the textual information is unrelated to both the image and question, thus insufficient to answer when relying only on the text.

Starting with the base set B = { ( I, Q ) } , we derive three variant sets by adding each text type as context: Q m = { ( I, T m , Q ) } , Q c = { ( I, T c , Q ) } , and Q irr = { ( I, T irr , Q ) } .

The motivation for these cases is to examine how models handle different text types: matching text assesses use of relevant information, corrupted text evaluates handling of misleading information, and irrelevant text checks the model's ability to ignore distractions. Including matching text also prevents models from simply rejecting all text, as might happen if only irrelevant or corrupted text were used in prior study [40]. Together, these cases provide a fuller view of VLM performance across varied text inputs.

## 2.2. Model Behavior

̸

Given both vision and language information, we categorize the model behaviors into three conditions: (1) consistent with Image answer ( ˆ Y mix = ˆ Y img ); (2) consistent with Text answer ( ˆ Y mix = ˆ Y txt ); (3) Other cases ( ˆ Y mix / ∈ { ˆ Y img , ˆ Y txt } ). To better understand the model behavior when inconsistency between vision and language data happens, we only consider the cases where the Image answers are different from the Text answers (under exact match) in empirical analysis (i.e., ˆ Y img = ˆ Y txt ). Formally, for a problem set Q ∈ {Q m , Q c , Q irr } , the proportion of the Image , Text and Other answers as p img , p txt and p o are:

<!-- formula-not-decoded -->

̸

where S = { X ∈ Q | ˆ Y img = ˆ Y txt } , as we only consider the inconsistent cases where the Image answers are different from the Text answers.

## 2.3. Metrics

Text Preference Ratio (TPR). Wedefine TPR to quantify the model's preference for text over image-based answers. It is calculated as:

<!-- formula-not-decoded -->

The TPR indicates text bias by showing the likelihood of the model choosing text over visual information when they are inconsistent. A higher TPR reflects a stronger text bias.

Accuracy. For any problem set Q , we have:

<!-- formula-not-decoded -->

Macro Accuracy. Macro ( f ) is the average accuracy of model f over different problem sets with text variations:

<!-- formula-not-decoded -->

Normalized Accuracy. Norm ( f vlm ; Q ) measures how a model is affected by the text variation, compared to the Base Accuracy, i.e., its accuracy on the base problem set B . For any problem set Q under a text variation, we calculate the corresponding normalized accuracy by

<!-- formula-not-decoded -->

## Text Construction

Given the image, the question and the answer, your task is to: 1. Generate an accurate 〈Description 1〉 which can be used for answering the question correctly without using the image.

3. Make sure both descriptions are sound and concise.
2. Generate a wrong description 〈Description 2〉 which can be used for answering the question with a completely wrong answer 〈answer 2〉 without using the image.
4. The wrong description's sentence structure should be similar to the correct description.

Here are the questions and answers:

Question: { question }

Answer: { answer }

Please output the two statements in this format:

Description 1: 〈Description 1〉

Description 2: 〈Description 2〉

Answer 2: 〈answer 2〉

Figure 2. Prompt for generating matched and corrupted text given an image, the question and the ground-truth answer. We substitute { question } and { answer } with the specific sample.

## 3. Empirical Analysis

In this section, we aim to answer the following research questions:

- ( Modality Preference ) How are the models' behaviors under different text conditions? Is there any modality preference bias in the models?
- ( Performance Impact ) To what extent can text bias affect the models' performance, particularly with corrupted text in the context?
- ( Influencing Factors ) Is text bias affected by instructions, size of language models, or token position?

## 3.1. Setup

Tasks and Datasets. We evaluate model performance on VQA datasets covering four domains, including (1) General VQA: 1,000 samples from VQAv2 [15] validation split; (2) Document VQA: 1,000 samples from DocVQA [29] validation split for chart and table understanding.; (3) Math Reasoning: 1,000 samples from the minitest split of MathVista [28]. (4) Brand Recognition: 2,500 samples from a phishing detection dataset test split [22] using HTML text and webpage screenshots, focused on identifying a website's brand. Note, each question is expanded with three types of text variations, creating a total of 16,500 test samples derived from 5,500 unique questions. Our study includes 10 VLMs, covering proprietary models [2, 3] and open models [1, 10, 26, 38]. The temperature is set to 0 for deterministic generation. We include all experimental details, examples and results in the Appendix.

Text Variation Construction. We use GPT-4o model to generate matching and corrupted text. Given an image I , question Q , and answer Y , we prompt the model to produce a supporting description as the matched text T m and a contradictory description as the corrupted text T c , allow- ing models to produce the correct answer and an incorrect answer without image input.

The prompt used for text generation is shown in Figure 2. We extract 〈 Description 1 〉 and 〈 Description 2 〉 as the matched and corrupted texts, respectively.

To construct irrelevant text, we randomly sample passages from the WikiText dataset [30], which contains texts from Wikipedia articles. These sampled texts serve as irrelevant cases, as they are factual but unrelated to the image and question. See Appendix for examples.

Query Instruction. Humans may be uncertain about which information to trust when inconsistency arises. To reduce ambiguity, following previous works [35], we prepend a sentence to the textual information, alerting the model to potential errors and encouraging cautious use of the textual information.

<!-- image -->

Sanity Check. Weevaluate the constructed text variations with model performance when only the text context is provided for answering questions. We expect that the matched text supplies enough information for correct answers, the corrupted text misleads the model into incorrect answers, and the irrelevant text lacks relevant information, leading the model to respond either randomly or with uncertainty (e.g., 'I don't know').

Table 1. Text-only accuracy (%) across different models. It provides a sanity check for the constructed text when matched, corrupted, or irrelevant.

| Model         |   Base |   Match |   VQAv2 Corruption |   Irrelevance |
|---------------|--------|---------|--------------------|---------------|
| Claude Sonnet |  66.88 |   84.39 |              16.17 |         24.39 |
| GPT-4o        |  78.39 |   90.07 |              17.59 |         18.67 |
| Molmo-7B-D    |  76.33 |   88.98 |              18.74 |         35.40 |

## 3.2. Blind Faith in Text

In Figures 3 and 4, we observe two key findings that illustrate the phenomenon of 'blind faith in text.' First, when textual data is inconsistent with visual data yet is relevant, models tend to favor the text, as indicated by high text preference ratios in both match and corruption cases. For example, Claude Haiku shows text preference ratios of 87%

Figure 3. Model behaviors over different models when text is corrupted, matched or irrelevant.

<!-- image -->

and 83% under match and corruption in VQAv2, respectively. Overall, high preference ratios are observed (usually over 50% ), particularly for open models. Second, some models, such as Qwen2-VL-7B , show even higher text preference in corruption cases ( 29% ) compared to match cases ( 13% ), indicating a tendency to rely on text even when it's incorrect, thus demonstrating limited discernment between accurate and inaccurate textual information. These results underscore the 'blind faith in text' seen across models.

Open models exhibit stronger text bias compared to proprietary models. Although open models perform comparably to or even surpass proprietary models in standard VQA benchmarks, our results show that open models display a much higher text preference in our benchmark, even in the presence of incorrect text. This text bias remains prominent even in the efficient versions of proprietary models (i.e., GPT-4o mini and Claude Haiku ). Overall, Claude Sonnet shows the most robustness under text-based interference among the evaluated models. This issue is critical in the development of open models, particularly when deploying them in real-world, complex applications, such as multi-modal agents or online shopping platforms [19, 39].

## 3.3. Performance Impact

Figure 4. Text Preference Ratio (TPR) of all models under different text variations. Most models exhibit high text preference bias when the textual information is relevant even if they are incorrect, especially for open models. Among the proprietary models, Claude-Sonnet exhibits the strongest robustness to corrupted text.

<!-- image -->

Table 2. Performance (%) reported as Base Accuracy, Corruption Accuracy, Normalized Corruption Accuracy (Norm) and Text Preference Ratio (TPR) under corruption. Bold : best performance; underline: second best. Full results under all text variations are in the Appendix.

|                | VQAv2   | VQAv2        | VQAv2   | VQAv2   | DocVQA   | DocVQA       | DocVQA   | DocVQA   | MathVista   | MathVista    | MathVista   | MathVista   |
|----------------|---------|--------------|---------|---------|----------|--------------|----------|----------|-------------|--------------|-------------|-------------|
| Model          | Base ↑  | Corruption ↑ | Norm ↑  | TPR ↓   | Base ↑   | Corruption ↑ | Norm ↑   | TPR ↓    | Base ↑      | Corruption ↑ | Norm ↑      | TPR ↓       |
| GPT-4o mini    | 69.82   | 51.55        | 73.83   | 52.42   | 69.40    | 38.20        | 55.04    | 52.07    | 52.30       | 23.90        | 45.70       | 80.28       |
| Claude Haiku   | 50.08   | 25.54        | 50.99   | 82.70   | 68.80    | 40.20        | 58.43    | 47.67    | 41.00       | 19.80        | 48.29       | 77.42       |
| GPT-4o         | 78.39   | 70.75        | 90.25   | 27.09   | 85.00    | 73.60        | 86.59    | 17.96    | 58.90       | 41.20        | 69.95       | 48.98       |
| Claude Sonnet  | 66.88   | 68.17        | 101.93  | 9.58    | 87.00    | 84.60        | 97.24    | 3.21     | 56.30       | 49.30        | 87.57       | 29.14       |
| LLaVA-NeXT-7B  | 79.45   | 28.69        | 36.10   | 85.52   | 53.60    | 10.00        | 18.60    | 87.77    | 35.80       | 19.70        | 54.97       | 84.19       |
| LLaVA-NeXT-13B | 81.02   | 37.61        | 46.40   | 74.43   | 57.70    | 11.00        | 19.10    | 86.84    | 36.20       | 20.60        | 56.89       | 80.83       |
| LLaVA-NeXT-34B | 82.96   | 42.87        | 51.70   | 67.56   | 64.00    | 15.10        | 23.61    | 82.69    | 34.00       | 21.70        | 61.98       | 67.64       |
| Phi3.5         | 75.65   | 35.23        | 46.50   | 74.05   | 78.20    | 50.50        | 64.60    | 40.51    | 43.10       | 22.20        | 51.47       | 80.20       |
| Molmo-7B-D     | 76.33   | 49.29        | 64.50   | 59.40   | 74.00    | 38.40        | 51.90    | 57.20    | 44.90       | 32.90        | 73.27       | 60.63       |
| Qwen2-VL-7B    | 85.51   | 50.79        | 59.41   | 29.22   | 90.50    | 57.50        | 63.63    | 37.41    | 55.40       | 28.90        | 52.18       | 70.23       |

The strong text bias leads to significant performance drops under corruption. Given the phenomenon of 'blind faith in text,' it is essential to assess its impact in performance, especially with corrupted text. As shown in Table 2, performance drops sharply in the presence of corrupted text. For instance, Qwen2-VL-7B accuracy on VQAv2, DocVQA, and MathVista falls to 59% , 63% , and 52% of its original levels, an approximate 50% reduction. While proprietary models show greater stability with smaller declines, the efficient variants of these models also experience significant drops. This highlights the need for caution when deploying efficient variants of proprietary models in safety-critical applications.

Table 3. Performance on the Brand Detection dataset reported in Base Accuracy, Corruption Accuracy, Normalized Corruption Accuracy (Norm), and Text Preference Ratio (TPR). Bold : best performance; underline: second best performance.

|                | Brand Detection   | Brand Detection   | Brand Detection   | Brand Detection   |
|----------------|-------------------|-------------------|-------------------|-------------------|
| Model          | Base ↑            | Corruption ↑      | Norm ↑            | TPR ↓             |
| GPT-4o mini    | 88.84             | 84.8              | 95.44             | 7.48              |
| Claude Haiku   | 84.40             | 78.72             | 93.27             | 6.44              |
| GPT-4o         | 88.68             | 89.76             | 101.22            | 0.83              |
| Claude Sonnet  | 90.20             | 90.24             | 100.04            | 0.96              |
| LLaVA-NeXT-7B  | 78.60             | 55.32             | 70.39             | 59.17             |
| LLaVA-NeXT-13B | 83.00             | 60.00             | 72.29             | 40.65             |
| LLaVA-NeXT-34B | 66.28             | 53.52             | 80.77             | 23.49             |
| Phi3.5         | 84.40             | 60.68             | 71.90             | 50.45             |
| Molmo-7B-D     | 87.44             | 41.44             | 47.39             | 60.40             |
| Qwen2-VL-7B    | 89.68             | 86.48             | 96.43             | 2.99              |

Text bias can introduce safety risks in real-world applications. Beyond general VQA tasks, we examine the safety implications of text bias in a real-world context: brand recognition in webpage understanding [22]. In this task, models typically use both an HTML string and a webpage screenshot to identify a website's brand. However, HTML content can be easily manipulated with incorrect or misleading information; for example, phishing websites may inject targeted brand names into the HTML to evade detection systems, which is taken as corruption cases. Further details about this setting are provided in the Appendix.

As shown in Table 3, under the corruption condition, most open models, such as Molmo-7B-D , show a significant performance drop, with accuracy reduced by nearly 50% compared to the original performance. In contrast, proprietary models show slight resilience, likely due to their ability to use information from the HTML string while being less affected by injected content.

## 3.4. Influencing Factors

In this section, we explore factors that contribute to text bias in VLMs and identify key influences. Unless otherwise noted, results are based on the VQAv2 dataset.

Instructions can reduce text bias but with limitations. We further investigate whether text bias can be mitigated by explicitly instructing models to focus on image information and reduce reliance on text. Inspired by previous work [35], we prepend instructions to the questions to guide the mod- els on which modality to prioritize. Specifically, we compare text preference ratios in three cases: neutral, 'Focus on Text,' and 'Focus on Image.' In the modified prompts, we add the phrases 'Please focus on the text to answer the question' and 'Please focus on the image to answer the question' respectively. As shown in Figure 5 (left), the instructions influence modality preference, but the effect is limited. In QwenVL-2-7B , the average text preference ratio only shifts from 16 . 8% to 14 . 2% when changing the instruction from 'Focus on Text' to 'Focus on Image.' This limited effect may also indicate weak instruction-following capabilities in cross-modal interactions.

Figure 5. The effect of different factors (prompting, language model size, text relevance) on text bias. Left: Instructional prompts influence modality preference slightly; text preference drops from 16 . 8% to 14 . 2% with 'Focus on Image' vs. 'Focus on Text' in QwenVL-2-7B . Middle : Scaling the language models (7B, 13B, 34B) in LLaVA-NeXT models decreases text bias but only marginally. Right: Increasing text relevance to the query with BM25 retrieval, raises text bias.

<!-- image -->

Training with a larger language model can reduce text bias but saturates. Language models are essential components in current VLM architectures [9, 24]. Scaling up language models in VLMs generally enhances model capabilities [18, 36]. We thus study the impact of model size on text bias using the LLaVA-NeXT models. As shown in Figure 5 (Middle), increasing model size from 7B to 34B reduces text bias overall. The 7B model exhibits high text preference with similar ratios for both match and corruption cases (86.3% and 85.5%, respectively). When scaled to 14B, there is a notable improvement, with a gap of 12% between match and corruption text preferences. Further scaling to 34B continues to reduce text preference overall, however the gap between matched and corrupted text preference ratios remains stable.

Relevant text is more likely to influence vision-language models. In applications like RAG, retrieved text can appear relevant to a query but may ultimately be unhelpful for accurate answers. To examine how text relevance affects text bias in VLMs, we use BM25 rank retrieval [33] with the question Q as the query, varying topk results to indicate relevance levels. The Top-1 result is the most relevant to the question but remains unrelated to the image, making it unhelpful for answering the question. As shown in Figure 5 (Right), text bias increases with text relevance. In the most relevant (Top-1) cases, Molmo-7B-D exhibits over a 10% text preference ratio, even though the text does not aid accurate predictions. This suggests that models are less distracted by clearly irrelevant text but are influenced by seemingly relevant (yet ultimately irrelevant) text, raising concerns for applications like multi-modal RAG, where retrieved text may appear relevant yet distract the model.

Text bias is related to the order of image and text tokens. Previous studies have shown that token order influences bias in LLMs during language generation [32, 49]. Since VLMs use LLMs [8, 37] as core components and are trained in an autoregressive manner, we examine whether text bias is affected by text and image token order. Notably, VLMs often include a large number of image tokens from vision encoder. To test this, we compare text preference ratios by altering the order of text and image tokens in Phi3.5 . As shown in Figure 6, placing text tokens before image tokens increases text bias consistently under three text variations. While previous research has suggested that generation misalignment or hallucinations in VLMs may stem from reduced attention to image tokens [11, 47], our findings indicate that the initial token modality may strongly influence modality preference, exacerbating text bias.

Interplay between uni-modal certainty and model behavior. To explore when models rely on vision versus text, we explore uni-modal certainty as a key factor in shaping model behavior. Specifically, we analyze the proportions of image, text, and other responses (i.e., p img , p txt , and p o ) across groups divided by uni-modal certainty quantiles. Figure 7 shows an interesting interplay effect: when text certainty P txt is high and image certainty P img is low, models favor Text answers, and vice versa. When both certainties are low, models often produce Other answers, instead of favoring Text or Image alone.

Figure 6. Effect of token order on text bias: Placing text tokens before image tokens increases text bias in Phi3.5 .

<!-- image -->

Figure 7. Effect of uni-modality certainty on model modality preference. Image/Text certainties are divided into three quantile bins, with higher values indicating higher certainty. Models favor visual data when image certainty is high and text certainty is low, and vice versa. When both certainties are low, models often produce Other answers instead of favoring one modality alone.

<!-- image -->

## 4. Investigated Solutions

## 4.1. Instruction

In Section 3.4, we observed that instructional prompts can influence the model's modality preference. For example, adding the instruction 'Focus on the image to answer the question' before the question helps reduce text bias to some extent. To explore this further, we evaluate performance with this instruction as a baseline, finding a slight improvement ( 1 -2% ) in Macro accuracy, as shown in Tables 4 and 5.

## 4.2. Supervised Finetuning (SFT)

Data. The composition of training data is key for effective VLM training [36]. Specifically, we include both textonly and image-text samples for fine-tuning. We collect 1,000 samples evenly distributed across five data types: text-only data, original VQA data, and VQA samples under match, corruption, and irrelevance text conditions as text- augmented samples. Seed data is from the VQAv2 validation split, separate from the benchmark evaluation data.

Table 4. In-distribution performance comparison between original models, instruction and fine-tuned models.

| Model         | VQAv2   | VQAv2   | VQAv2        | VQAv2         | VQAv2   |
|---------------|---------|---------|--------------|---------------|---------|
|               | Base ↑  | Match ↑ | Corruption ↑ | Irrelevance ↑ | Macro ↑ |
| LLaVA-NeXT-7B | 79.45   | 92.32   | 28.69        | 79.43         | 66.81   |
| Instruction   | 79.45   | 92.25   | 34.27        | 78.15         | 68.22   |
| SFT           | 77.48   | 87.56   | 71.25        | 77.32         | 78.71   |
| Qwen2-VL-7B   | 85.51   | 92.76   | 50.79        | 83.70         | 75.75   |
| Instruction   | 85.51   | 92.62   | 54.78        | 82.82         | 76.74   |
| SFT           | 84.18   | 87.01   | 82.72        | 84.00         | 84.58   |

Table 5. Performance comparison with Base and Macro accuracy based on DocVQA, MathVista, and Brand Recognition. See full results under different text conditions in Appendix.

|               | DocVQA   | DocVQA   | MathVista   | MathVista   | Brand Detection   | Brand Detection   |
|---------------|----------|----------|-------------|-------------|-------------------|-------------------|
| Model         | Base ↑   | Macro ↑  | Base ↑      | Macro ↑     | Base ↑            | Macro ↑           |
| LLaVA-NeXT-7B | 53.60    | 51.07    | 35.80       | 41.03       | 78.60             | 46.44             |
| Instruction   | 53.60    | 49.27    | 35.80       | 41.20       | 78.60             | 47.36             |
| SFT           | 52.20    | 56.17    | 35.30       | 41.63       | 81.36             | 72.29             |
| Qwen2-VL-7B   | 90.50    | 80.83    | 55.40       | 53.87       | 89.68             | 81.85             |
| Instruction   | 90.50    | 80.77    | 55.40       | 54.10       | 89.68             | 84.48             |
| SFT           | 90.30    | 88.97    | 58.50       | 57.17       | 89.44             | 88.75             |

Setup. We follow a standard supervised fine-tuning procedure, using a learning rate of 1 . 0 × 10 -4 with cosine decay over 3 epochs and a warmup ratio of 0.1 for stable convergence. Additionally, we apply LoRA for efficient finetuning. Experiments are conducted on the LLaVA-NeXT-7B and Qwen2-VL-7B models.

In-Distribution Performance. In Table 4, we compare the performance of the original models, models with instruction, and models after supervised fine-tuning on indistribution data. The results show that supervised finetuning can better improve model accuracy compared to instruction, especially under text corruption conditions, where corruption accuracy increases from 28 . 69% to 71 . 25% , while maintaining overall performance in macro accuracy.

Generalization. We further assess the generalization of the fine-tuned models by evaluating their performance on datasets beyond VQAv2. As shown in Table 5, the finetuned models exhibit some improvement across all datasets. However, improvements are smallest on MathVista, likely due to a greater distribution shift from general VQA tasks to math reasoning tasks in vision.

Effect of Text-Only Data. We conduct an ablation study to inspect the role of text-only and cross-modality data in fine-tuning on LLaVA-NeXT-7B . For a fair comparison, the total amount of training data remains constant across experiments. As shown in Figure 8 (Left), fine-tuning re- duces text bias and enhances the model's ability to distinguish between match and corruption cases, with a gap up to 40% . Additionally, text-only data is important for maintaining core language capabilities: without it, models may reject text indiscriminately, leading to overly cautious behavior and limiting their use of helpful text.

Figure 8. Left: The effect of text-only data in SFT. Right: The effect of data volume in SFT.

<!-- image -->

Effect of Data Volume. We study the impact of data volume in SFT, shown in Figure 8 (Right). As the amount of SFT data increases, the model's reliance on text decreases significantly in corruption cases (from 58% to 25% ) while remaining relatively steady in match cases. This trend indicates that scaling up SFT data can reduce dependency on corrupted or irrelevant text, while preserving the model's effectiveness to match text.

## 5. Theoretical Analysis

In this section, we present theoretical analysis to explain why the majority of VLMs exhibit an inherent tendency to have blind faith in text. Let N and M be the size of pure-text data and multi-modal data in the training set that are i.i.d sampled from distributions D txt and D mul , respectively. Our informal results are as follows, see more details in Appendix A.

Theorem 5.1. (Informal; Theorem A.5 (simplified) ) Under certain assumptions, with probability at least 1 -δ the expected loss under pure-text data E ( X,Y ) ∼D txt [ l ( f vlm ( X ; ˆ θ ERM ) , Y ) ] achieves

<!-- formula-not-decoded -->

and similarly the expected loss under multi-modal data E ( X,Y ) ∼D mul [ l ( f vlm ( X ; ˆ θ ERM ) , Y ) ] achieves

<!-- formula-not-decoded -->

l ( · , · ) is a bounded loss function, and ˆ θ ERM is the learned parameter(s) from Empirical Risk Minimization (ERM);

ε txt appr (resp. ε mul appr ) and ε cross are the quantities that represent the approximation error of pure-text data (resp. multimodal data) and cross-modal error, respectively, and they are only dependent on the distributions D txt , D mul and the hypothesis of models; C vlm is a quantity related to the covering number of the hypothesis of models. See details in Appendix A.

Remark 5.2 . Observe that the expected losses under pure-text data and multi-modal data are influenced by N M + M ε cross and N N + M ε cross , respectively. Our theoretical analysis, under specific assumptions, indicates that the tendency of blind faith in textual information may arise from the significant imbalance between N and M . Particularly, in most VLMs, N ≫ M , as these models often rely heavily on pre-trained language models, leading to the larger expected loss in multi-modal data and less in pure-text data, potentially making models favor text over image.

## 6. Related Work

Evaluation on VLMs. Current evaluation benchmarks for VLMs include single-task benchmarks [15, 28, 29, 34] and multi-modal benchmarks [20, 27, 44, 45] designed to assess general model capabilities across diverse tasks. Some studies also evaluate specific issues, such as hallucination [11, 21], catastrophic forgetting [46], and robustness [43]. However, these benchmarks are primarily vision-centric, usually treating text as question input without additional context, which limits the evaluation of models' robustness to text variations. While text can be additional hints in specific tasks like math reasoning, current datasets [28] focus on assessing reasoning skills rather than the model's ability to handle varied text inputs. As a result, whether VLMs can reliably handle multi-modal inconsistencies remains an open question. This gap is critical for real-world applications, such as multi-modal RAG, where models encounter variable text inputs. To this end, our work studies VLM performance under different text variations, identifying a text bias that affects model reliability.

Benchmarks with Input Perturbation. Text perturbations have been widely used in natural language tasks to evaluate model robustness and stability against distractions or misleading context [6, 16, 23, 31, 35, 42]. In computer vision, similar efforts focus on adding imperceptible perturbations to image inputs to assess models' sensitivity to noise [14, 48]. Our work shifts focus from image perturbations to explore the effects of text variations on VLMs, which already excel in vision-centric benchmarks. Recent research [7] highlights data leakage in VLM benchmarks by studying performance with missing modalities. With a different goal, we investigate how VLMs manage inconsistencies between visual and textual data in vision-centered tasks, evaluating robustness in cross-modal interactions.

## 7. Conclusion and Discussion

Revisiting our core question-can VLMs reliably handle multi-modal inconsistencies?-our findings indicate that substantial challenges remain. In this work, we observe the phenomenon of 'blind faith in text' in VLMs, often relying on text over visual input when inconsistency arises, resulting in performance drops and potential safety risks. Our analysis showed that factors like instructions, model size, text relevance, token order, and modality certainty can influence text bias. Notably, scaling model size and prompt changes alone do not resolve this issue. While supervised fine-tuning with text augmentation helps, balancing robustness and effectiveness in cross-modal settings remains challenging. We hope this work highlights the risks of deploying VLMs in applications like multi-modal RAG, offering insights and prompting further development of more reliable and robust VLMs for cross-modal interactions.

## Acknowledgement

This research is supported by the Ministry of Education, Singapore, under the Academic Research Fund Tier 1 (FY2023) (Grant A-8001996-00-00).

## References

- [1] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219 , 2024. 3
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 , 2023. 3
- [3] [Anthropic. Claude 3 model card, 2024. https : / / www -cdn . anthropic . com / de8ba9b01c9ab7cbabf5c33b80b7bbc618857627 / Model Card Claude 3.pdf . 3](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)
- [4] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An opensource framework for training large autoregressive visionlanguage models. arXiv preprint arXiv:2308.01390 , 2023. 1
- [5] Peter L Bartlett and Shahar Mendelson. Rademacher and gaussian complexities: Risk bounds and structural results. Journal of Machine Learning Research , 3(Nov):463-482, 2002. 13
- [6] Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. Benchmarking large language models in retrieval-augmented generation. In Proceedings of the AAAI Conference on Artificial Intelligence , pages 17754-17762, 2024. 8
- [7] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao,
8. Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330 , 2024. 8
- [8] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023) , 2023. 6
- [9] W Dai, J Li, D Li, AMH Tiong, J Zhao, W Wang, B Li, P Fung, and S Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. arxiv 2023. arXiv preprint arXiv:2305.06500 . 1, 6
- [10] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146 , 2024. 3
- [11] Ailin Deng, Zhirui Chen, and Bryan Hooi. Seeing is believing: Mitigating hallucination in large visionlanguage models via clip-guided decoding. arXiv preprint arXiv:2402.15300 , 2024. 6, 8
- [12] Zane Durante, Qiuyuan Huang, Naoki Wake, Ran Gong, Jae Sung Park, Bidipta Sarkar, Rohan Taori, Yusuke Noda, Demetri Terzopoulos, Yejin Choi, et al. Agent ai: Surveying the horizons of multimodal interaction. arXiv preprint arXiv:2401.03568 , 2024. 1
- [13] Benjamin L Edelman, Surbhi Goel, Sham Kakade, and Cyril Zhang. Inductive biases and variable creation in selfattention mechanisms. In International Conference on Machine Learning , pages 5793-5831. PMLR, 2022. 12, 13
- [14] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572 , 2014. 8
- [15] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition , pages 6904-6913, 2017. 1, 3, 8, 19
- [16] Robin Jia and Percy Liang. Adversarial examples for evaluating reading comprehension systems. arXiv preprint arXiv:1707.07328 , 2017. 8
- [17] Raghav Kapoor, Yash Parag Butala, Melisa Russak, Jing Yu Koh, Kiran Kamble, Waseem Alshikh, and Ruslan Salakhutdinov. Omniact: A dataset and benchmark for enabling multimodal generalist autonomous agents for desktop and web. arXiv preprint arXiv:2402.17553 , 2024. 1
- [18] Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. arXiv preprint arXiv:2402.07865 , 2024. 6
- [19] Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. arXiv preprint arXiv:2401.13649 , 2024. 1, 4

- [20] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125 , 2023. 8
- [21] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large visionlanguage models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , pages 292-305, Singapore, 2023. Association for Computational Linguistics. 8
- [22] Yuexin Li, Chengyu Huang, Shumin Deng, Mei Lin Lock, Tri Cao, Nay Oo, Bryan Hooi, and Hoon Wei Lim. Knowphish: Large language models meet multimodal knowledge graphs for enhancing reference-based phishing detection. arXiv preprint arXiv:2403.02253 , 2024. 3, 5, 15, 16, 19
- [23] Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110 , 2022. 8
- [24] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023. 1, 6
- [25] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 26296-26306, 2024. 16
- [26] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 3
- [27] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European Conference on Computer Vision , pages 216-233. Springer, 2025. 8
- [28] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The Twelfth International Conference on Learning Representations , 2024. 3, 8, 16, 19
- [29] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision , pages 2200-2209, 2021. 3, 8, 19
- [30] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016. 4
- [31] John X Morris, Eli Lifland, Jin Yong Yoo, Jake Grigsby, Di Jin, and Yanjun Qi. Textattack: A framework for adversarial attacks, data augmentation, and adversarial training in nlp. arXiv preprint arXiv:2005.05909 , 2020. 8
- [32] Pouya Pezeshkpour and Estevam Hruschka. Large language models sensitivity to the order of options in multiple-choice questions. arXiv preprint arXiv:2308.11483 , 2023. 6
- [33] Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval , 3(4):333-389, 2009. 6
- [34] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A
16. benchmark for visual question answering using world knowledge. In European conference on computer vision , pages 146-162. Springer, 2022. 8
- [35] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi, Nathanael Sch¨ arli, and Denny Zhou. Large language models can be easily distracted by irrelevant context. In International Conference on Machine Learning , pages 31210-31227. PMLR, 2023. 1, 4, 5, 8
- [36] Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai Charitha Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, et al. Cambrian1: A fully open, vision-centric exploration of multimodal llms. arXiv preprint arXiv:2406.16860 , 2024. 6, 7
- [37] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 , 2023. 6
- [38] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution. arXiv preprint arXiv:2409.12191 , 2024. 3
- [39] Chen Henry Wu, Jing Yu Koh, Ruslan Salakhutdinov, Daniel Fried, and Aditi Raghunathan. Adversarial attacks on multimodal agents. arXiv preprint arXiv:2406.12814 , 2024. 4
- [40] Kevin Wu, Eric Wu, and James Zou. Clasheval: Quantifying the tug-of-war between an llm's internal prior and external evidence. Preprint , 2024. 2
- [41] Peng Xia, Kangyu Zhu, Haoran Li, Tianze Wang, Weijia Shi, Sheng Wang, Linjun Zhang, James Zou, and Huaxiu Yao. Mmed-rag: Versatile multimodal rag system for medical vision language models. arXiv preprint arXiv:2410.13085 , 2024. 1
- [42] Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. Adaptive chameleon or stubborn sloth: Revealing the behavior of large language models in knowledge conflicts. In The Twelfth International Conference on Learning Representations , 2024. 8
- [43] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. arXiv preprint arXiv:2306.13549 , 2023. 8
- [44] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490 , 2023. 1, 8
- [45] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. arXiv preprint arXiv:2311.16502 , 2023. 1, 8
- [46] Yuexiang Zhai, Shengbang Tong, Xiao Li, Mu Cai, Qing Qu, Yong Jae Lee, and Yi Ma. Investigating the catastrophic forgetting in multimodal large language models. arXiv preprint arXiv:2309.10313 , 2023. 8
- [47] Yi-Fan Zhang, Weichen Yu, Qingsong Wen, Xue Wang, Zhang Zhang, Liang Wang, Rong Jin, and Tieniu Tan.

Debiasing large visual language models. arXiv preprint arXiv:2403.05262 , 2024. 6

- [48] Yunqing Zhao, Tianyu Pang, Chao Du, Xiao Yang, Chongxuan Li, Ngai-Man Man Cheung, and Min Lin. On evaluating adversarial robustness of large vision-language models. Advances in Neural Information Processing Systems , 36, 2024. 8
- [49] Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. Large language models are not robust multiple choice selectors. In The Twelfth International Conference on Learning Representations , 2023. 6

## A. Details of Theoretical Analysis

To provide a rigorous foundation for our theoretical analysis, we begin by formally outlining the training process of a visionlanguage model. For clarity and conciseness, the following is a streamlined adaptation of the standard training process. A VLM is a function f vlm : X → Y , where X := R τ × d denotes the set of sequences of d -dimensional feature vector (that can represent text or image) with length τ , and Y denotes the output space of the model. Without loss of generalization, we assume Y := R for simplicity.

## A.1. Structure

Following Edelman et al. [13], we consider the form of transformer structure of f vlm with L layers as follows. The parameters of i 's layer is denoted by W ( i ) := { W ( i ) Q , W ( i ) K , W ( i ) V , W ( i ) C } . In addition, we denote W 1: i = ( W (1) , . . . , W i -1 ) to be the parameters up to i 's layer. Further, we let the block of i -th layer g ( i ) tf-block : R τ × d → R τ × d to be

<!-- formula-not-decoded -->

where X ∈ R τ × d is the model's input, and Π norm is the layer normalization function, σ is a non-linear activation function, and

<!-- formula-not-decoded -->

with Softmax( · ) being the standard softmax function. Finally, the scalar output is defined as

<!-- formula-not-decoded -->

where [ G ] τ ∈ R d denotes the τ -th row of the matrix G ∈ R τ × d . Furthermore, we have the following assumptions within the structure.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Assumption A.3. The activation function σ ( · ) is L σ -Lipschitz in the l 2 norm.

Assumption A.4. The loss function l ( · ) is b -bounded and is L loss -Lipschitz in its arguments.

## A.2. Training process

Let X txt = [( X txt 1 , y txt 1 ) , · · · , ( X txt N , y txt N )] be a pure-text training set with size N , where X txt i ∈ R τ × d is a sequence of the text feature vector of length τ , and y txt i = f txt gt ( X txt i ) ∈ R is its ground-truth label with f txt gt ( · ) denoted as the ground-true function for the pure text data. We assume X txt 1 , · · · , X txt N are i.i.d. sampled from a unknown distribution D txt .

In addition, let X mul = [( X mul 1 , y mul 1 ) , · · · , ( X mul N , y mul M )] be a multi-modal training set with size M , where X multi i ∈ R τ × d is a sequence of multi-modal (e.g., text and image) feature vector of length τ , and y mul i = f mul gt ( X multi i ) ∈ R is its ground-truth label with f mul gt ( · ) denoted as the ground-true function for the multi-modal data. Similarly, we assume X mul 1 , · · · , X mul N are i.i.d. sampled from a unknown distribution D mul .

Furthermore, let l : R × R → be a loss function. Then, we define the parameter ˆ θ ERM ∈ Θ according to the ERM learning process of the multi-modal paradigm as

<!-- formula-not-decoded -->

Our main theoretical result is given in the next subsection.

## A.3. Results

We now provide the formal statement of Theorem A.5.

Theorem A.5. Let Θ be the set of parameters that satisfies Assumption A.1, A.2, A.3 and A.4. For any θ ∈ Θ , let f vlm ( · ; θ ) be a VLM as is defined in equation 1 with L layers. With probability at least 1 -δ ,

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

is the constant related to the covering number of the function class of { f vlm ( · ; θ ) | θ ∈ Θ } , and the notation ≲ hides global constants and logarithmic factors on quantities besides N,M and τ .

## A.4. Proof of Theorem A.5

Before we formally prove Theorem A.5, we first present some useful Lemmas from previous works. For any real-valued function class F , we let N ∞ ( F ; ε ; x (1) , . . . , x ( m ) ) denote the converting number of F with respect to the radius ε and the samples { x (1) , . . . , x ( m ) } .

Lemma A.6. (Adapted from Bartlett and Mendelson [5, Theorem 8] and Edelman et al. [13, Lemma A.4]) Consider a real-valued function class F such that | f | ≤ A for all f ∈ F and log N ∞ ( F ; ε ; x (1) , . . . , x ( m ) ) ≤ C F /ε 2 for all x (1) , . . . , x ( m ) ∈ X . Let l ( · , · ) to be a loss function bounded by b and is L loss -Lipschitz in its arguments, and g gt : X → R be a ground-true function. Then for any δ &gt; 0 and any distribution D for the i.i.d samples x (1) , . . . , x ( m ) ∈ X , with probability at least 1 -δ , simultaneously for all f ∈ F ,

<!-- formula-not-decoded -->

for some constant c &gt; 0 .

Lemma A.7. (Adapted from Edelman et al. [13, Theorem A.17]) Suppose ∀ i ∈ [ m ] , ∥ ∥ X ( i ) ∥ ∥ 2 , ∞ ≤ B X . Let Θ be the set of parameters that satisfies Assumption A.1, A.2, A.3 and A.4. For any θ ∈ Θ , let f vlm ( · ; θ ) is a vlm model as is fined in equation 1 with L layers. We have

<!-- formula-not-decoded -->

Proof of Theorem A.5. By Lemma A.6, with probability at least 1 -δ we have simultaneously for all θ ∈ Θ ,

<!-- formula-not-decoded -->

where A ≤ ( C 2 L σ ) 2 L · B X , and C is a constant such that for all ε &gt; 0 and X (1) , . . . , X ( m ) ∈ R τ × d with ∥ X ( i ) ∥ 2 , ∞ ≤ B X

<!-- formula-not-decoded -->

Similarly, with probability at least 1 -δ we have simultaneously for all θ ∈ Θ ,

<!-- formula-not-decoded -->

Note that for any θ ∈ Θ we have

<!-- formula-not-decoded -->

where (a) follows from Jensen's inequality.

In addition, with probability at least 1 -δ ,we have for all θ ∈ Θ

<!-- formula-not-decoded -->

where (a) follows from equation 6.

Combining equation 5, equation 8 and equation 9, we get that with probability at least 1 -δ , for all θ ∈ Θ ,

<!-- formula-not-decoded -->

By the definition of ˆ θ ERM , equation 10 implies that with probability at least 1 -δ ,

<!-- formula-not-decoded -->

Note the fact that max { N,M } ≤ N + M ≤ 2 max { N,M } . Finally, by Lemma A.7 and hiding global constants and logarithmic factors on quantities besides N,M and τ , we get with probability 1 -δ ,

<!-- formula-not-decoded -->

and similarly,

<!-- formula-not-decoded -->

This completes the proof of Theorem A.5

## B. Experimental Setup

This section outlines the experimental setup, including examples of constructed textual variations, details of the brand detection task [22], and the evaluation protocols employed. We present examples illustrating the three types of textual variations alongside the corresponding image, original question, and ground-truth answers to provide clarity and context.

## B.1. Examples

This subsection provides examples of matching, corrupted, and irrelevant texts across different datasets in Tables 6 to 9.

## B.2. Brand Recognition

Brand recognition from a webpage is a crucial step in detecting phishing websites. Phishing webpages aim to deceive users by imitating the appearance of legitimate websites associated with well-known brands. Accurately identifying the brand linked to a webpage allows for a comparison between the input webpage's URL and the official URL of the recognized brand, aiding in the detection of phishing attempts.

Table 6. Illustration of matching, corrupted, and irrelevant information in a sample from VQAv2.

<!-- image -->

|              | Q: What green veggie is on the pizza                                                                                     | GT: pepper                                                   |
|--------------|--------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| Match:       | The pizza has green pepper slices on one of its sections.                                                                | The pizza has green pepper slices on one of its sections.    |
| Corruption:  | The pizza has green broccoli florets on one of its sections.                                                             | The pizza has green broccoli florets on one of its sections. |
| Irrelevance: | Beckham stown. tatives Eastern to quit school at the age of 17 to years later, he became principal of from Kentucky, the | widowed mother. public schools,                              |

In our experiments, we utilized phishing webpage samples from the TR-OP dataset [22]. Each sample comprises a screenshot and its corresponding HTML code. Depending on the scenario, the HTML content either reflects the target brand displayed in the screenshot or is altered to assess the model's robustness. We evaluated three specific scenarios:

- Matching: The original HTML includes information about the target brand visible in the screenshot. This scenario provides the model with consistent inputs, helping it correctly identify the brand.
- Corruption: In this case, we inserted a fabricated brand name (e.g., 'The official webpage of MobrisPremier') into the HTML to mislead the model into recognizing a non-existent brand. Since no corresponding URL exists for such brands, phishing detection becomes infeasible for these inputs.
- Irrelevance: The HTML content was replaced with randomly selected sentences from the Wiki dataset [], ensuring that the new content was unrelated to any brand. This scenario tests the model's ability to handle inputs with no brand-specific information.

To standardize the inputs, we preprocessed the HTML content by removing all tags and truncating it to a maximum length of 5,000 characters.

## B.3. Evaluation

We follow the evaluation protocol specified for each dataset. To reduce cases where models generate open-ended answers, which complicates evaluation, we adopt a similar approach to the evaluation setting in LLaVA-1.5 [25]. For certain datasets, we append additional formatting prompts after the question, as shown in Table 10.

For MathVista [28], which uses GPT-based evaluation, we do not include formatting prompts. Instead, GPT is employed directly to evaluate the outputs.

## C. Experimental Results

To rigorously assess the performance impact of varying textual contexts, we have documented the comprehensive results across four distinct datasets. These results are quantified using several metrics: Accuracy, Normalized Accuracy, and Text Preference Ratio (TPR) for the text variations of Match, Corruption, and Irrelevance, alongside Macro Accuracy. The detailed outcomes are encapsulated in Table 11.

Table 7. Illustration of matching, corrupted, and irrelevant information in a sample from DocVQA.

<!-- image -->

|              | Q: What time is 'question and answers 'session?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | GT: 12:25 to 12:58 p.m.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Match:       | The 'Questions and Answers' session is scheduled from 12:25 to 12:58 p.m.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | The 'Questions and Answers' session is scheduled from 12:25 to 12:58 p.m.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Corruption:  | The 'Questions and Answers' session is scheduled from 2:00 to 5:00 p.m.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | The 'Questions and Answers' session is scheduled from 2:00 to 5:00 p.m.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| Irrelevance: | The Americans knew of the approach of the Japanese forces from reports from native scouts and their own patrols , but did not know exactly where or when they would attack . The ridge around which Edson deployed his men consisted of three distinct hillocks . At the southern tip and surrounded on three sides by thick jungle was Hill 80 ( so named because it rose 80 ft ( 24 m ) above sea level ) . Six hundred yards north was Hill 123 ( 123 ft ( 37 m ) high ) , the dominant feature on the ridge . The northernmost hillock was unnamed and about 60 ft ( 18 m ) high . Edson placed the five companies from the Raider battalion on the west side of the ridge and the three Parachute battalion companies on the east side , holding positions in depth from Hill 80 back to Hill 123 . Two of the five Raider companies , ¨ B ¨ and ¨ C ¨ , held a line between the ridge , a small , swampy lagoon , and the Lunga River . Machine @-@ gun teams from ¨ E ¨ Company , the heavy weapons company , were scattered throughout the | The Americans knew of the approach of the Japanese forces from reports from native scouts and their own patrols , but did not know exactly where or when they would attack . The ridge around which Edson deployed his men consisted of three distinct hillocks . At the southern tip and surrounded on three sides by thick jungle was Hill 80 ( so named because it rose 80 ft ( 24 m ) above sea level ) . Six hundred yards north was Hill 123 ( 123 ft ( 37 m ) high ) , the dominant feature on the ridge . The northernmost hillock was unnamed and about 60 ft ( 18 m ) high . Edson placed the five companies from the Raider battalion on the west side of the ridge and the three Parachute battalion companies on the east side , holding positions in depth from Hill 80 back to Hill 123 . Two of the five Raider companies , ¨ B ¨ and ¨ C ¨ , held a line between the ridge , a small , swampy lagoon , and the Lunga River . Machine @-@ gun teams from ¨ E ¨ Company , the heavy weapons company , were scattered throughout the |

For a thorough assessment of the investigated methodologies, encompassing base models, instructional prompts, and Supervised Fine-Tuning (SFT), we present results across four datasets, measured in terms of Accuracy, Normalized Accuracy, Text Preference Ratio (TPR) under the text variations of Match, Corruption, and Irrelevance, as well as Macro Accuracy. These experiments were conducted utilizing the models LLaVA-NeXT-7B and Qwen2-VL-7B . The detailed findings are provided in Table 12.

<!-- image -->

Table 8. Illustration of matching, corrupted, and irrelevant information in a sample from MathVista.

|              | Q: Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end. Question: what is the total volume of the measuring cup? (Unit: g)                                                                                                                                                                                                                                                                                                                                                                                                                     | GT: 1000                                                                               |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Match:       | The measuring cup has markings up to 1000 grams, indicating its total volume capacity.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | The measuring cup has markings up to 1000 grams, indicating its total volume capacity. |
| Corruption:  | The measuring cup has markings up to 500 grams, indicating its total volume capacity.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |                                                                                        |
| Irrelevance: | The windmill at Thelnetham was built by millwright George in 1819 . It replaced a post mill which had been moved to Sandy previous year . The mill was set to work on Christmas Day modernised by the fitting of a cast @-@ iron windshaft , Patent windshaft was fitted on 16 July 1832 . It was made by J Aickman Lynn millwright , and weighs 1 ˘ 00be tons ( 1 @,@ 780 kg ) . A new stock September 1836 William Button died on 11 February 1837 . The mill passed jointly their son Richard . Richard Button worked the mill until 1860 to his sons Richard and William , who sold it to Richard Peverett | , the Kings was fitted in                                                              |

Table 9. Illustration of matching, corrupted, and irrelevant information in a sample from Brand Recognition.

<!-- image -->

|              | Q: Define targeted brand as a brand that a webpage belongs to. Given the screenshot of a webpage P as the primary information for identifying the target brand and the text as additional reference, determine what the targeted brand of P is. The text can be HTML from the webpage or something irrelevant. Please be careful with the text, as it may contain noise or adversarial attacks. You must output the targeted brand of P even if you are not sure about it. Only output the brand name without any additional information. Input information: -HTML: 'html' Answer:                                                                                                                                                                                                                | GT: Adobe                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Match:       | Adobe® PDF Reader® Xl This File is Protected by Adobe® Security LogIn with your email and password to open this document.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Adobe® PDF Reader® Xl This File is Protected by Adobe® Security LogIn with your email and password to open this document.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Corruption:  | The official webpage of MobrisPremier. The official webpage of Mo- brisPremier. Adobe® PDF Reader® Xl This File is Protected by Adobe® Se- curity LogIn with your email and password to open this document.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | The official webpage of MobrisPremier. The official webpage of Mo- brisPremier. Adobe® PDF Reader® Xl This File is Protected by Adobe® Se- curity LogIn with your email and password to open this document.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Irrelevance: | Ayola began her career in the theatre , performing in a number of Shake- spearean plays including Twelfth Night , A Midsummer Night 's Dream , The Tempest , The Merchant of Venice and Macbeth . She states of this : ' Shake- speare keeps coming my way . I love the fact that I get to play people who are much more articulate than I 'll ever be ' . Ayola has performed in Twelfth Night in the lead roles of both Olivia and Viola . She explains : ' The role of Viola didn 't sit that well with me for some reason but Olivia makes more sense . ' She has also appeared in modern performances , assuming the title role of Dido , Queen of Carthage at the Globe Theatre in London in 2003 , which she described as ' a dream of a part ' . She has deemed her dream role to be that | Ayola began her career in the theatre , performing in a number of Shake- spearean plays including Twelfth Night , A Midsummer Night 's Dream , The Tempest , The Merchant of Venice and Macbeth . She states of this : ' Shake- speare keeps coming my way . I love the fact that I get to play people who are much more articulate than I 'll ever be ' . Ayola has performed in Twelfth Night in the lead roles of both Olivia and Viola . She explains : ' The role of Viola didn 't sit that well with me for some reason but Olivia makes more sense . ' She has also appeared in modern performances , assuming the title role of Dido , Queen of Carthage at the Globe Theatre in London in 2003 , which she described as ' a dream of a part ' . She has deemed her dream role to be that |

Table 10. Response formatting prompts used for evaluation.

| Dataset                                                      | Response Formatting Prompts                                                                                                                                          |
|--------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| VQAv2 [15] DocVQA [29] MathVista [28] Brand Recognition [22] | Please only output the answer with a single word or phrase. Please only output the answer directly. - Only output the brand name without any additional information. |

| Model          | Base ↑   | Match      | Match   | Match   | Corruption   | Corruption   | Corruption   | Irrelevance   | Irrelevance   | Irrelevance   | Macro ↑   |
|----------------|----------|------------|---------|---------|--------------|--------------|--------------|---------------|---------------|---------------|-----------|
|                | Base ↑   | Accuracy ↑ | Norm ↑  | TPR     | Accuracy ↑   | Norm ↑       | TPR ↓        | Accuracy ↑    | Norm ↑        | TPR ↓         | Macro ↑   |
| GPT-4o mini    | 69.82    | 87.49      | 125.31  | 89.15   | 51.55        | 73.83        | 52.42        | 72.11         | 103.28        | 3.77          | 70.38     |
| Claude Haiku   | 51.02    | 82.81      | 162.31  | 86.74   | 26.33        | 51.61        | 82.71        | 51.10         | 100.16        | 13.95         | 53.41     |
| GPT-4o         | 78.39    | 89.27      | 113.88  | 69.03   | 70.75        | 90.25        | 27.09        | 78.82         | 100.55        | 1.56          | 79.61     |
| Claude Sonnet  | 66.88    | 77.85      | 116.40  | 49.86   | 68.17        | 101.93       | 9.58         | 70.89         | 106.00        | 1.38          | 72.30     |
| LLaVA-NeXT-7B  | 79.45    | 92.32      | 116.20  | 86.25   | 28.69        | 36.11        | 85.52        | 79.43         | 99.97         | 4.72          | 66.81     |
| LLaVA-NeXT-13B | 81.02    | 93.59      | 115.51  | 86.45   | 37.61        | 46.42        | 74.43        | 81.29         | 100.33        | 3.30          | 70.83     |
| LLaVA-NeXT-34B | 82.96    | 93.07      | 112.19  | 79.10   | 42.87        | 51.68        | 67.56        | 79.64         | 95.99         | 2.70          | 71.86     |
| Phi3.5         | 75.65    | 91.23      | 120.59  | 79.51   | 35.23        | 46.57        | 74.05        | 74.87         | 98.97         | 2.25          | 67.11     |
| Molmo-7B-D     | 76.33    | 88.57      | 116.04  | 88.32   | 49.29        | 64.57        | 59.40        | 76.50         | 100.22        | 9.36          | 71.45     |
| Qwen2-VL-7B    | 85.51    | 92.76      | 108.48  | 13.17   | 50.79        | 59.40        | 29.22        | 83.70         | 97.88         | 1.28          | 75.75     |

## (a) VQAv2

| Model          | Base ↑   | Match      | Match   | Match   | Corruption   | Corruption   | Corruption   | Irrelevance   | Irrelevance   | Irrelevance   | Macro ↑   |
|----------------|----------|------------|---------|---------|--------------|--------------|--------------|---------------|---------------|---------------|-----------|
|                | Base ↑   | Accuracy ↑ | Norm ↑  | TPR     | Accuracy ↑   | Norm ↑       | TPR ↓        | Accuracy ↑    | Norm ↑        | TPR ↓         | Macro ↑   |
| GPT-4o mini    | 69.40    | 81.40      | 117.26  | 82.74   | 38.20        | 55.04        | 52.07        | 67.20         | 96.83         | 0.80          | 62.27     |
| Claude Haiku   | 69.53    | 83.45      | 120.06  | 68.77   | 39.35        | 56.61        | 47.67        | 57.82         | 83.16         | 1.18          | 60.21     |
| GPT-4o         | 85.00    | 90.40      | 106.35  | 64.75   | 73.60        | 86.59        | 17.96        | 86.40         | 101.65        | 0.23          | 83.47     |
| Claude Sonnet  | 87.00    | 91.53      | 105.15  | 41.18   | 84.60        | 97.24        | 3.21         | 87.41         | 100.47        | 0.00          | 87.85     |
| LLaVA-NeXT-7B  | 53.60    | 90.80      | 169.40  | 86.92   | 10.00        | 18.66        | 87.77        | 52.40         | 97.76         | 0.71          | 51.07     |
| LLaVA-NeXT-13B | 57.70    | 90.40      | 156.68  | 87.82   | 11.00        | 19.06        | 86.84        | 55.80         | 96.68         | 0.65          | 52.40     |
| LLaVA-NeXT-34B | 64.00    | 87.80      | 137.19  | 84.62   | 15.10        | 23.59        | 82.69        | 62.70         | 97.97         | 0.13          | 55.20     |
| Phi3.5         | 78.20    | 92.40      | 118.16  | 58.01   | 50.50        | 64.60        | 40.51        | 77.00         | 98.46         | 0.00          | 73.30     |
| Molmo-7B-D     | 74.00    | 90.30      | 122.30  | 87.54   | 38.40        | 51.89        | 57.20        | 74.70         | 100.95        | 0.37          | 67.80     |
| Qwen2-VL-7B    | 90.50    | 95.10      | 105.08  | 51.97   | 57.50        | 63.64        | 37.41        | 89.90         | 99.34         | 0.22          | 80.83     |

## (b) DocVQA

| Model          |   Base ↑ | Match      | Match   | Match   | Corruption   | Corruption   | Corruption   | Irrelevance   | Irrelevance   | Irrelevance   |   Macro ↑ |
|----------------|----------|------------|---------|---------|--------------|--------------|--------------|---------------|---------------|---------------|-----------|
|                |          | Accuracy ↑ | Norm ↑  | TPR     | Accuracy ↑   | Norm ↑       | TPR ↓        | Accuracy ↑    | Norm ↑        | TPR ↓         |           |
| GPT-4o mini    |    52.30 | 73.80      | 141.11  | 88.82   | 23.90        | 45.70        | 80.28        | 44.40         | 84.89         | 20.14         |     47.37 |
| Claude Haiku   |    41.00 | 80.30      | 195.85  | 88.04   | 19.80        | 48.29        | 77.42        | 39.70         | 96.83         | 23.33         |     46.60 |
| GPT-4o         |    58.90 | 73.70      | 125.04  | 85.20   | 41.20        | 69.95        | 48.98        | 53.10         | 90.15         | 13.55         |     56.00 |
| Claude Sonnet  |    56.30 | 68.10      | 120.95  | 57.69   | 49.30        | 87.57        | 29.14        | 55.20         | 98.05         | 7.96          |     57.53 |
| LLaVA-NeXT-7B  |    35.80 | 74.80      | 273.62  | 88.72   | 19.70        | 54.97        | 84.19        | 28.40         | 104.02        | 38.22         |     40.97 |
| LLaVA-NeXT-13B |    36.20 | 76.20      | 257.43  | 88.98   | 20.60        | 56.89        | 80.83        | 32.60         | 96.28         | 37.18         |     43.13 |
| LLaVA-NeXT-34B |    34.00 | 68.00      | 200.00  | 73.59   | 21.70        | 61.98        | 67.64        | 32.10         | 94.41         | 20.40         |     40.60 |
| Phi3.5         |    43.10 | 73.70      | 171.21  | 84.82   | 22.20        | 51.47        | 80.20        | 41.10         | 95.36         | 13.99         |     45.67 |
| Molmo-7B-D     |    44.90 | 68.50      | 152.57  | 82.46   | 32.90        | 73.27        | 60.63        | 45.30         | 100.89        | 27.49         |     48.90 |
| Qwen2-VL-7B    |    55.40 | 77.80      | 140.43  | 84.50   | 28.90        | 52.18        | 70.23        | 54.90         | 99.10         | 8.44          |     53.87 |

## (c) MathVista

| Model          | Base ↑   | Match      | Match   | Match   | Corruption   | Corruption   | Corruption   | Irrelevance   | Irrelevance   | Irrelevance   | Macro ↑   |
|----------------|----------|------------|---------|---------|--------------|--------------|--------------|---------------|---------------|---------------|-----------|
|                | Base ↑   | Accuracy ↑ | Norm ↑  | TPR     | Accuracy ↑   | Norm ↑       | TPR ↓        | Accuracy ↑    | Norm ↑        | TPR ↓         | Macro ↑   |
| GPT-4o mini    | 88.84    | 86.88      | 97.80   | 30.43   | 84.80        | 95.44        | 7.48         | 88.48         | 99.60         | 0.08          | 86.72     |
| Claude Haiku   | 84.40    | 83.40      | 98.81   | 26.02   | 78.72        | 93.27        | 6.44         | 82.28         | 97.49         | 0.00          | 81.47     |
| GPT-4o         | 88.68    | 89.48      | 100.90  | 14.64   | 89.76        | 101.22       | 0.83         | 89.16         | 100.54        | 0.04          | 89.47     |
| Claude Sonnet  | 90.20    | 90.56      | 100.40  | 17.03   | 90.24        | 100.04       | 0.96         | 90.24         | 100.04        | 0.00          | 90.35     |
| LLaVA-NeXT-7B  | 78.60    | 77.56      | 98.67   | 82.39   | 62.52        | 79.54        | 64.74        | 16.28         | 20.72         | 70.45         | 52.12     |
| LLaVA-NeXT-13B | 83.00    | 79.00      | 95.18   | 77.04   | 33.96        | 40.92        | 72.97        | 11.72         | 14.12         | 79.61         | 41.56     |
| LLaVA-NeXT-34B | 66.28    | 68.28      | 102.99  | 31.60   | 53.52        | 80.77        | 23.49        | 52.84         | 79.69         | 10.65         | 58.21     |
| Phi3.5         | 84.40    | 83.84      | 99.33   | 31.39   | 60.68        | 71.90        | 50.54        | 16.44         | 19.48         | 79.17         | 53.65     |
| Molmo-7B-D     | 87.44    | 87.32      | 99.86   | 37.38   | 41.44        | 47.39        | 60.40        | 60.88         | 69.63         | 27.36         | 63.21     |
| Qwen2-VL-7B    | 89.68    | 88.92      | 99.15   | 17.22   | 86.48        | 96.43        | 2.99         | 70.16         | 78.20         | 15.73         | 81.85     |

(d)

Brand Detection Model LLaVA-NeXT-7B

Table 11. Performance in Accuracy, Normalized Accuracy (Norm) and Text Preference Ratio (TPR) across four datasets under three text variations: Match, Corruption, and Irrelevance. The Macro column represents the average of Match, Corruption, and Irrelevance Accuracy for each model, calculated to be comparable to the Base accuracy.

Instruction

SFT

Qwen2-VL-7B

Instruction

SFT

Model

LLaVA-NeXT-7B

Instruction

SFT

Qwen2-VL-7B

Instruction

SFT

Model

LLaVA-NeXT-7B

Instruction

SFT

Qwen2-VL-7B

Instruction

SFT

Base

↑

79.45

79.45

77.48

85.51

85.51

84.18

Base

↑

53.60

53.60

52.20

90.50

90.50

90.30

Base

↑

35.80

35.80

35.30

55.40

55.40

58.50

Accuracy

92.32

92.25

87.56

92.76

92.62

87.01

Accuracy

90.80

88.60

75.50

95.10

94.70

93.10

Accuracy

74.80

70.60

68.70

77.80

78.10

74.00

Match

↑

Norm

↑

116.20

116.12

113.01

108.48

108.32

103.36

Match

↑

Norm

↑

169.40

165.30

144.63

105.08

104.64

103.10

Match

↑

Norm

↑

208.94

197.77

194.90

140.43

140.79

126.50

TPR

86.25

86.46

59.73

13.17

14.42

36.65

TPR

86.92

84.01

56.21

51.97

51.46

26.06

TPR

84.32

84.68

77.42

84.50

86.50

78.31

Corruption

Accuracy Norm

↑

↑

28.69

34.27

71.25

50.79

54.78

82.72

36.11

43.13

91.94

59.40

64.07

98.26

## (a) VQAv2

Corruption

Accuracy Norm

↑

↑

10.00

9.80

42.80

57.50

57.80

84.30

18.66

18.28

81.99

63.64

63.88

93.35

## (b) DocVQA

Corruption

Accuracy Norm

↑

↑

19.70

21.80

23.50

28.90

29.30

40.30

## (c) MathVista

| Model         |   Base ↑ | Match      | Match   | Match   | Corruption   | Corruption   | Corruption   | Irrelevance   | Irrelevance   | Irrelevance   |   Macro ↑ |
|---------------|----------|------------|---------|---------|--------------|--------------|--------------|---------------|---------------|---------------|-----------|
|               |          | Accuracy ↑ | Norm ↑  | TPR     | Accuracy ↑   | Norm ↑       | TPR          | Accuracy ↑    | Norm ↑        | TPR           |           |
| LLaVA-NeXT-7B |    78.60 | 77.56      | 98.67   | 68.30   | 62.52        | 79.54        | 59.17        | 16.28         | 20.72         | 89.14         |     46.44 |
| Instruction   |    78.60 | 78.36      | 99.70   | 66.57   | 54.84        | 69.77        | 59.63        | 8.88          | 11.30         | 85.26         |     47.36 |
| SFT           |    81.36 | 78.32      | 96.26   | 37.18   | 69.48        | 85.39        | 17.92        | 69.08         | 84.92         | 9.08          |     72.29 |
| Qwen2-VL-7B   |    89.68 | 88.92      | 99.15   | 17.22   | 86.48        | 96.43        | 2.99         | 70.16         | 78.20         | 15.73         |     81.85 |
| Instruction   |    89.68 | 88.52      | 98.71   | 17.50   | 87.12        | 97.15        | 1.94         | 77.80         | 86.77         | 9.34          |     84.48 |
| SFT           |    89.44 | 90.08      | 100.72  | 20.32   | 88.76        | 99.24        | 1.43         | 87.40         | 97.72         | 0.71          |     88.75 |

(d)

Brand Detection

Table 12. Performance of investigated solutions in Accuracy, Normalized Accuracy (Norm) and Text Preference Ratio (TPR) across four datasets under three text variations: Match, Corruption, and Irrelevance. The Macro column represents the average of Match, Corruption, and Irrelevance Accuracy for each model, calculated to be comparable to the Base accuracy.

55.03

60.89

66.57

52.17

52.88

68.89

TPR

85.52

78.50

20.00

29.22

27.01

6.69

TPR

87.77

87.38

28.19

37.41

37.00

6.32

TPR

84.19

81.85

63.75

70.23

70.59

49.16

Irrelevance

Accuracy Norm

↑

↑

79.43

78.15

77.32

83.70

82.82

99.97

98.36

99.79

97.88

96.85

84.00

99.79

Irrelevance

Accuracy Norm

↑

↑

52.40

49.40

50.20

89.90

89.80

97.76

92.16

96.17

99.34

99.23

89.50

99.11

Irrelevance

Accuracy Norm

↑

↑

28.40

31.20

32.70

54.90

54.90

57.20

79.33

87.15

92.64

99.10

99.10

97.78

TPR

4.72

6.69

4.06

1.28

1.18

2.59

TPR

0.71

1.54

0.14

0.22

0.11

0.11

TPR

34.57

32.94

10.76

8.44

8.11

5.65

Macro

66.81

68.22

78.71

75.75

76.74

84.58

Macro

51.07

49.27

56.17

80.83

80.77

88.97

Macro

↑

↑

↑

41.03

41.20

41.63

53.87

54.10

57.17