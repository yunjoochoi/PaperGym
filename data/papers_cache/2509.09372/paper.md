<!-- image -->

## VLA-ADAPTER: AN EFFECTIVE PARADIGM FOR TINY-SCALE VISION-LANGUAGE-ACTION MODEL

Yihao Wang

1 , 2 , 4 , ∗ ,

♢ Pengxiang Ding 2 , 3 , 4 , ∗ , † Lingxiao Li 1 , 4 , 5 Can Cui 2 , 4

Zirui Ge 3 , 4

Xinyang Tong 2 , 4

Wenxuan Song 4 , 6

Han Zhao 2 , 3 , 4

Wei Zhao 2 , 4

Pengxu Hou 6

Siteng Huang 2

Yifan Tang 1 Wenhui Wang 1

Ru Zhang 1 , B

Jianyi Liu 1

Donglin Wang 2 , B

1 Beijing University of Posts and Telecommunications 2 Westlake University 3 Zhejiang University

4 OpenHelix Team 5 State Key Laboratory of Networking and Switching Technology

6 The Hong Kong University of Science and Technology (Guangzhou)

∗ Equal contribution: yh-wang@bupt.edu.cn; dingpx2015@gmail.com

B Corresponding Author † Project Lead ♢ Work done during interning at Westlake University

## ABSTRACT

Vision-Language-Action (VLA) models typically bridge the gap between perceptual and action spaces by pre-training a large-scale Vision-Language Model (VLM) on robotic data. While this approach greatly enhances performance, it also incurs significant training costs. In this paper, we investigate how to effectively bridge vision-language (VL) representations to action (A). We introduce VLA-Adapter, a novel paradigm designed to reduce the reliance of VLA models on large-scale VLMs and extensive pre-training. To this end, we first systematically analyze the effectiveness of various VL conditions and present key findings on which conditions are essential for bridging perception and action spaces. Based on these insights, we propose a lightweight Policy module with Bridge Attention, which autonomously injects the optimal condition into the action space. In this way, our method achieves high performance using only a 0.5B-parameter backbone, without any robotic data pre-training. Extensive experiments on both simulated and real-world robotic benchmarks demonstrate that VLA-Adapter not only achieves state-of-the-art level performance, but also offers the fast inference speed reported to date. Furthermore, thanks to the proposed advanced bridging paradigm, VLA-Adapter enables the training of a powerful VLA model in just 8 hours on a single consumer-grade GPU, greatly lowering the barrier to deploying the VLA model. Project page: https://vla-adapter.github.io/ .

## 1 INTRODUCTION

In the past two years, with significant breakthroughs in multimodal LLMs (Karamcheti et al., 2024; Steiner et al., 2024; Liu et al., 2023b; Li et al., 2025b), developing robot systems with general perception, understanding, and behavior capabilities has become a key research direction in artificial intelligence. In particular, the emergence of the Vision-Language-Action (VLA) model offers a new solution for

Figure 1: Characteristics of VLA-Adapter. ' ↓ ' is that smaller values are better, and vice versa. Our paradigm can effectively obtain the SOTA-level VLA model using a tiny-scale backbone.

<!-- image -->

enabling robot operations driven by instructions (Kim et al., 2024; Cui et al., 2025; Kim et al., 2025; Song et al., 2025b; Cen et al., 2025; Zhang et al., 2025b; Shi et al., 2025). Research on VLA primarily focuses on extracting multimodal information and aligning it with the action space to generate the high-quality actions (Team et al., 2024; Liu et al., 2024b; Zhong et al., 2025; Fan et al., 2025).

Current VLA models typically require large-scale embodied data (e.g., Open X-Embodiment (Collaboration et al., 2024) and DROID (Khazatsky et al., 2024)) to pre-train Multimodal Large Language Models (MLLMs) (Especially, VLMs) for task adaptability (Cheang et al., 2024), which is then passed to the designed Policy network (Bu et al., 2024a; Li et al., 2024) to decode or generate actions for handling the tasks in the diverse environments (Liu et al., 2023a; Mees et al., 2022).

However, when confronted with high-dimensional control environments, VLA models still face several bottlenecks, including reliance on large-scale VLMs, slow fine-tuning speed, high GPU memory (VRAM) consumption, and low inference efficiency (throughput), as shown in Figure 1. To this end, it is necessary to explore the most essential but rarely discussed question in the VLA field: How to bridge the gap of VL (vision-language representations) to A (action) more effectively?

To answer this question, we propose VLA-Adapter, a novel bridging paradigm for VLA. We systematically explore how different conditions influence action generation and give some key findings for VLA design. On this basis, we built a Policy network with Bridge Attention to autonomously inject the optimal condition into the action space. Experiments show that VLA-Adapter has superior performance, high inference efficiency, and fast throughput with a tiny-scale backbone. It significantly lowers the barrier to VLA deployment. The main contributions are summarized as follows.

- To our knowledge, this work is the first systematic analysis of bridging paradigms' effects on action generation. And we also give some key findings of the VLA model design.
- VLA-Adapter transfers the sufficient multimodal information to the proposed Policy Network for action generation, effectively bridging the modality gap from VL to A.
- Rich experiments show that VLA-Adapter has a higher success rate, smaller scale, lower tuning cost, and faster inference in diverse simulated and real-world robotic tasks.

## 2 RELATED WORK

## 2.1 VISION-LANGUAGE-ACTION (VLA) MODELS

Recently, leveraging pre-trained Vision-Language Models (VLMs) (Karamcheti et al., 2024; Steiner et al., 2024; Liu et al., 2023b; Li et al., 2025b) to control robots for performing various daily tasks has substantially accelerated research in embodied intelligence. This has emerged as a prominent research focus (Black et al., 2025b;a; Shukor et al., 2025; NVIDIA et al., 2025; Liu et al., 2024b; Luo et al., 2025; Cheang et al., 2025; Jiang et al., 2025; Ding et al., 2024; Bu et al., 2024b; Fan et al., 2025; Tong et al., 2024). These models are referred to as the VLA models.

Typically, VLA models require large-scale embodied datasets, such as Open X-Embodiment (Collaboration et al., 2024), for pre-training (Liu et al., 2024b; Cheang et al., 2025). This process integrates VLMs with a dedicated Policy network (Song et al., 2025a; Li et al., 2024), allowing the system to decode or generate action sequences for diverse tasks in an end-to-end manner. Moreover, dual-system VLA architectures (Shentu et al., 2024; Bu et al., 2024a; Cui et al., 2025) have recently garnered attention. These methods typically introduce an intermediate latent token to connect the VLMs and the Policy, using an asynchronous mechanism to enhance coordination between the two systems (Zhang et al., 2024). This design mitigates latency issues during action generation.

Consequently, how to effectively and efficiently bridge the gap from the vision-language perception space to the action space has become a key challenge in the design of VLA models.

## 2.2 BRIDGING FROM PERCEPTION TO ACTION SPACE

Earlier studies (Kim et al., 2024; Brohan et al., 2023a;b) attempted to directly align perception and action spaces by discretizing actions into tokens. However, this discretization inevitably introduces inherent loss. Recent studies have shifted their focus toward continuous action spaces (Liu et al., 2024a; NVIDIA et al., 2025; Black et al., 2025b; Shukor et al., 2025; Kim et al., 2025). Based on the types of perceptual features utilized to bridge to the action space, they can be categorized:

1) Raw Features from VLMs. Raw features (refer to vision and language representations) are extracted directly from the VLM. Early methods extract representations from the final-layer VLM, operating under the assumption that it encodes the most task-relevant semantic information (Liu et al., 2024a; Zhang et al., 2024). More recent methods leverage the intermediate-layer features within the VLM (Black et al., 2025b). They believe that such representations may retain richer multimodal information, thereby benefiting Policy in tasks that demand fine-grained perception or complex reasoning. For example, some studies use features from a middle layer (NVIDIA et al., 2025), the first-half layers (Shukor et al., 2025), or all intermediate-layer features (Black et al., 2025b).

2) Additional Query as Interface. Furthermore, recent studies (Kim et al., 2025; Cui et al., 2025) have introduced a novel interface that employs additional queries as bridges between VLMs and Policy, rather than transmitting Raw features. The query is learnable and can incorporate multimodal information, showing superior performance. The existing bridge paradigms are shown in Figure 2.

Figure 2: Existing representative bridge paradigms from VL to A.

<!-- image -->

## 3 VLA-ADAPTER METHODOLOGY

## 3.1 PRELIMINARY

Wepresent the VLA-Adapter framework, as illustrated in Figure 3. This VLM follows the PrismaticVLMs architecture (Karamcheti et al., 2024). It has M layers. At timestep t , the input into VLM consists of {X v t , X g t , L t , AQ t } : the 3rd-view image X v t , the gripper image X g t , the instruction L t , and additional ActionQuery AQ t . After inputting X v t and X g t , the DINOv2 (Oquab et al., 2024) and SigLIP (Zhai et al., 2023) extract vision embeddings. L t is tokenized. The outputs are the specified-layer Raw latent C R t and ActionQuery latent C AQ t . They serve as the conditions for Policy.

Backbone. To build a solid basis for research, we perform experiments of VLA-Adapter on different-scale backbones. The backbones select the Prismatic VLM trained on Qwen2.5-0.5B (Team, 2024), the Prismatic VLM trained on LLaMA2-7B (Touvron et al., 2023), and OpenVLA-7B (Kim et al., 2024) pre-trained on robotic data. The benefit gained from increasing backbone scale is limited in VLA-Adapter. The results are shown in Table 2 of Section 4.1. Therefore, to ensure efficiency, Qwen2.5-0.5B is our default backbone unless otherwise specified.

## 3.2 WHICH CONDITION IS ESSENTIAL FOR BRIDGING FROM VL TO A?

Although existing methods have adopted various bridging paradigms from VL to A, their relative effectiveness remains inconclusive. This is mainly due to the differences in the design of the VLM and the Policy. To address this gap, we explore which type of perception information is essential for action generation in the Policy network. In summary, we mainly focus on the following questions:

Question 1.1. Which layer of features within the VLM is more effective for the Policy network?

Question 1.2. Are the ActionQuery features a better choice than the Raw features?

Figure 3: The proposed VLA framework. The key components are the effective condition exploration and Attention design. ' Attention ' specifically includes cross attention with conditions and self attention with itself. In the ' Unified VLA-Adapter Framework ', ' Attention ' is the Bridge Attention as shown in Section 3.3. Four conditions about 'layer' and 'type' are given on the right.

<!-- image -->

To ensure compatibility with existing experimental protocols for representative work (e.g., π 0 (Black et al., 2025b)), we let the number of Policy layers be equal to that of VLM. At each layer of Policy, the action latent undergoes cross-attention with conditions and self-attention with itself. This iterative process ultimately yields the action output. Details of the Policy can be seen in Section 3.3.

Experimental Setting. Weevaluate four conditions in our framework. For Question 1.1 , to evaluate the effectiveness of the individual-layer information, we employ the single-layer latent as the conditions for the all-layer Policy, as shown in Figure 3a) and 3c). To evaluate the effectiveness of all-layer information, we feed each-layer latent into the corresponding-layer Policy, as shown in Figure 3b) and 3d). For Question 1.2 , to compare the effectiveness of the feature types, we use the C R t or C AQ t as conditions. The comparison on the

<!-- image -->

96

96

Figure 4: Comparison of four conditions in the VLA-Adapter framework on the LIBERO-Long. Blue and Green lines are single-layer C R t and single-layer C AQ t , as in Figure 3a) and 3b). Blue and Green columns are all-layer C R t and all-layer C AQ t , as in Figure 3c) and 3d). The detailed results are shown in Appendix C. Please note: the number of ActionQuery is 64 here. Its number is variable, similar to MetaQueries (Pan et al., 2025) in MLLM research; we will explore it in Section 4.5.

LIBERO-Long (Liu et al., 2023a), which is the long-horizon and complex benchmark, the results are as shown in Figure 4. We give the following key findings.

Key Finding 1. Regarding C R t , the middle-layer latent performs better than the deep-layer latent. Deep-layer C R t is biased towards semantic information and less effective in action generation. The middle-layer C R t effectively integrates image and text information, retains richer multimodal details, and facilitates action generation.

- Key Finding 2. Regarding C AQ t , deep-layer latent performs better than other-layer latent. Since ActionQuery is trained from scratch, and deep-layer C AQ t aggregates richer multimodal details and is more effectively promoting action generation than the shallow layers.
- Key Finding 3. Multi-layer features perform better. We observed that using all-layer features generally outperforms a single layer. Not only does it improve performance, but it also saves time on best layer selection during design. This design can be more universal.

Condition Determination. Does VLA-Adapter rely exclusively C AQ t as conditions? The answer is no. While all-layer C AQ t outperforms C R t , middle-layer C R t excels in some hard tasks. Comparison is shown in Table 1. So, we aim to enhance performance by using certain knowledge from C R t .

Table 1: Comparison of the i th-layer C R t and C AQ t in subtasks of LIBERO-Long.

| C R t     |   9 13 |    | C AQ t    |   1 |   13 |   17 |   21 |   23 |   24 |   All |
|-----------|--------|----|-----------|-----|------|------|------|------|------|-------|
| Subtask 7 |     90 | 82 | Subtask 7 |  76 |   66 |   74 |   70 |   70 |   74 |    76 |
| Subtask 9 |     74 | 84 | Subtask 9 |  78 |   62 |   58 |   72 |   72 |   84 |    78 |

## 3.3 POLICY WITH BRIDGE ATTENTION

Overall. For the simplicity of the model, we designed an L1-based Policy network. At t -th timestep, the input to Policy includes: {C R t , C AQ t , A τ =0 t , P t } . τ is the layer of Policy, and it has τ ∈ Z + , 0 ≤ τ ≤ M -1 . A 0 t is the H -step initial action of all zeros, it is processed by LayerNorm (LN) and Multi Layer Perceptron (MLP) to obtain ˜ A 0 t = [ ˜ a 0 t , ˜ a 0 t +1 , . . . , ˜ a 0 t + H -1 ] . P t is the proprioceptive state, and it is mapped through a two-layer MLP to obtain the proprio embedding σ 0 ( P t ) . The output is the H -step action chunk A M -1 t . Each layer is composed of a Bridge Attention module and a Feed-Forward Network (FFN). The Bridge Attention architecture is shown in Figure 5. ActionQuery ... ActionQuery latent Instruction Gripper+3rd-person Proprioceptive L V P Task latent ... ...

Figure 5: The Policy with Bridge Attention. The Policy parameters are only 97M when the backbone is Qwen2.5-0.5B. Each-layer C R t and C AQ t are integrated in Bridge Attention with the corresponding-layer action latent. Bridge Attention maps VL to Action to the greatest extent. The degree of C R t injection is learnable, ensuring the performance and stability of training.

<!-- image -->

K

VLM

Bridge Attention. The proposed Bridge Attention hopes to guide action generation to the greatest extent possible through the conditions C R t and C AQ t . Each Bridge Attention consists of two cross attentions and one self attention. In the first cross attention, C R t is processed through an MLP σ 1 to obtain K 1 , V 1 . The action latent ˜ A τ t is used as the Q 1 , and perform attention to get CA 1 ( ˜ A τ t , σ 1 ( C R t ) ) . In the second cross attention, C AQ t needs to be concatenated with the σ 0 ( P t ) and passed through an MLP σ 2 to obtain K 2 , V 2 . ˜ A τ t is used as the Q 2 to get CA 2 ( ˜ A τ t , σ 2 [ C AQ t , σ 0 ( P t ) ] ) . In the self attention, ˜ A τ t is as Q,K,V , and there is SA ( ˜ A τ t , ˜ A τ t ) .

To selectively inject certain C R t into the action space of the Policy, we introduce a learning parameter Ratio g to modulate the influence of CA 1 ( ˜ A τ t , σ 1 ( C R t ) ) . g is initialized to 0 value, and the tanh activation function is utilized tanh( g ) ∈ [ -1 , 1] to prevent extreme values from destabilizing the distribution (Zhang et al., 2023). And then, the three attentions are concatenated to obtain ̂ A τ t :

<!-- formula-not-decoded -->

After Bridge Attention, ̂ A τ t passes through a residual FFN to obtain ˜ A τ +1 t . Repeating the above process, we finally obtain ˜ A M -1 t . The action chunk A M -1 t is yielded by an LN and MLP layer.

Additionally, we also design a DiT-based (Diffusion Transformer (Peebles &amp; Xie, 2023)) Policy. Since the diversity of Policy is not the focus of this paper, we put its details and the brief results in Appendix B. The results show that L1-based performance and inference speed are generally superior to those of the DiT-based approach. Therefore, VLA-Adapter chose the L1 architecture as the Policy.

## 3.4 TRAINING

The training is conducted end-to-end, with the Policy trained from scratch. Given a ground truth action trajectory A t and action latent A τ t . We train VLA-Adapter model π θ ( · ) with the objective:

<!-- formula-not-decoded -->

For more details of training, please see Appendix F.1.

## 4 EXPERIMENTS

All experiments are run on 4 NVIDIA H100 GPUs. For more details of the hyperparameters, please see Appendix F.2. We perform rich experiments to answer the following questions:

Question 2.1. What are the advantages of the VLA-Adapter compared to other bridge paradigms?

Question 2.2. How does VLA-Adapter perform compared to existing methods?

Question 2.3. What else key components in the VLA-Adapter paradigm are worth exploring?

Experiment Overview. In Section 4.1, we use the long-horizon and complex LIBERO-Long (Liu et al., 2023a), which typically has a low success rate, to investigate the necessity of VLA-Adapter. From Section 4.2 to Section 4.4, we use LIBERO (Liu et al., 2023a) and CALVIN (Mees et al., 2022), which are widely used in VLA, as well as real-world data, to compare the performance comprehensively. In Section 4.5, we use LIBERO-Long to explore key parts of VLA-Adapter.

## 4.1 NECESSITY OF VLA-ADAPTER

Effectiveness of our bridge paradigm. To validate the effectiveness, we compare three kinds of backbones: · B1 : The Prismatic VLM (Karamcheti et al., 2024) trained on Qwen2.5-0.5B (Team, 2024). · B2 : The Prismatic VLM trained on LLaMA2-7B (Touvron et al., 2023). The first two are different-scale backbones without pre-training on robotic data. · B3 : The OpenVLA-7B (Kim et al., 2024) pre-trained on robotic data. We adopted the OpenVLA-OFT bridging paradigm (Kim et al., 2025) for comparison. It is the existing state-of-the-art level method on major benchmarks, including LIBERO-Long (Liu et al., 2023a). The comparison results are shown in Table 2.

Table 2: Effectiveness comparison with OpenVLA-OFT (Kim et al., 2025) on the LIBERO-Long (Liu et al., 2023a). 'Fine-tuned' is by LoRA fine-tuning (Hu et al., 2022). Bold represents the best performance. Please note, comparison with the bridge paradigms of π 0 (Black et al., 2025b) and GR00T N1 (NVIDIA et al., 2025) has been included in Section 3, so we will not compare it here.

| Fine-tuned         |   B1 +OFT | B1 +Ours       |   B2 +OFT | B2 +Ours       |   B3 +OFT | B3 +Ours       |
|--------------------|-----------|----------------|-----------|----------------|-----------|----------------|
| Success Rate (%) ↑ |      85.8 | 95.0 (9.2% ↑ ) |      87.5 | 95.2 (7.7% ↑ ) |      94.5 | 95.4 (0.9% ↑ ) |

Fortunately, VLA-Adapter remains effective when the backbone is frozen. Only the ActionQuery and Policy are trained from scratch. SmolVLA (Shukor et al., 2025) is the VLA dedicated to studying frozen VLMs. So, we compare with OpenVLA-OFT and SmolVLA. The results are shown in Table 3. Since the results of GR00T N1 come from (Song et al., 2025a), it did a full-params tuning, so we will not compare with it here. Based on Tables 2 and 3, we summarize two conclusions:

Table 3: Effectiveness comparison when the backbone is frozen. Benchmark is the same as Table 2. For a detailed analysis of OpenVLA-OFT (Kim et al., 2025) does not work, please see Appendix H.

| Frozen             |   OpenVLA-OFT |   SmolVLA |   VLA-Adapter |
|--------------------|---------------|-----------|---------------|
| Success Rate (%) ↑ |           0.0 |      77.0 |          86.4 |

Conclusion 1. VLA-Adapter improvement is obvious when VLMs without robotic pre-training.

Conclusion 2. Even if the backbone freezes, VLA-Adapter still performs strongly.

This can be attributed to the fact that, after pre-training on robotic data, the last-layer features are already adapted to the action domain, enabling efficient fine-tuning with a simple MLP. However, when VLMs without pre-training, relying solely on the last-layer latents, are insufficient for effective action mapping. So, adopting the VLA-Adapter becomes crucial to achieve efficient fine-tuning. These insights highlight a key advantage : VLA-Adapter facilitates efficient fine-tuning of VLMs without robotic pre-training, achieving performance that surpasses baselines using a tiny backbone.

Efficiency. VLA-Adapter attains a faster inference speed. The comparison is shown in Table 4.

Table 4: Inference efficiency comparison with OpenVLA (Kim et al., 2024) and OpenVLA-OFT (Kim et al., 2025). The action chunk is 8 dimensions, consistent with most VLA. 'OpenVLA-OFT (wo X g t , P )' is the L1-based version where the input is without the gripper image and proprioceptive state. It is the fastest version of OpenVLA-OFT. Benchmark is the same as Table 2.

| Efficiency                        |   OpenVLA |   OpenVLA-OFT (wo X g t , P ) |   OpenVLA-OFT |   VLA-Adapter |
|-----------------------------------|-----------|-------------------------------|---------------|---------------|
| Throughput (Hz) ↑ Latency (Sec) ↓ |       4.2 |                         109.7 |          71.4 |         219.2 |
|                                   |    0.2396 |                        0.0729 |        0.1120 |        0.0365 |

## 4.2 OVERALL PERFORMANCE ON VARIOUS TASKS

Benchmark. We selected the widely adopted LIBERO benchmark (Liu et al., 2023a) to evaluate performance across various types of tasks. LIBERO 1 provides multiple suites, including Spatial, Object, Goal, and Long. For detailed settings and examples of LIBERO, please see Appendix A.

Baselines. We selected recently published, comprehensive, and high-performance VLA works as comparison baselines. They are Large : 1. FlowVLA (Zhong et al., 2025), 2. UnifiedVLA (Li et al., 2025a), 3. OpenVLA (Kim et al., 2024), 4. OpenVLA-OFT (Kim et al., 2025), 5. UniVLA (Bu et al., 2025), 6. CoT-VLA (Zhao et al., 2025a), 7. WorldVLA (Cen et al., 2025), 8. TraceVLA (Zheng et al., 2024), 9. MolmoAct (Lee et al., 2025), 10. ThinkAct (Huang et al., 2025), and 11. PD-VLA (Song et al., 2025b); Small : 12. 4D-VLA (Zhang et al., 2025a), 13. SpatialVLA (Qu et al., 2025), 14. π 0 (Black et al., 2025b), 15. π 0 -FAST (Pertsch et al., 2025), 16. NORA (Hung et al., 2025), 17. SmolVLA (Shukor et al., 2025), 18. GR00T N1 (NVIDIA et al., 2025), and 19. GraspVLA (Deng et al., 2025); Tiny : 20. Seer (Tian et al., 2025), 21. VLA-OS (Gao et al., 2025), and 22. Diffusion Policy (Chi et al., 2023). Their performances are all derived from original references or the reproduction of other published works, ensuring objectivity and accuracy.

Metrics. Each subtask is repeated 50 × times to evaluate. We use the commonly used metric 'Success Rate', reported as ranging from 0 to 100, with higher values meaning better performance.

Results. Comparison on the LIBERO is shown in Table 5. The results in Table 5 demonstrate that VLA-Adapter, using only a tiny-scale backbone, can achieve performance comparable to OpenVLAOFT with 14 × larger. It surpasses representative works such as π 0 , SmolVLA, and GR00T N1. In addition, VLA-Adapter has a notable advantage of 29.0% over VLA-OS with the same-scale backbone on LIBERO-Long. These demonstrate the VLA-Adapter superiority on various tasks.

[1 https://libero-project.github.io/datasets](https://libero-project.github.io/datasets)

Table 5: Comparison on the LIBERO benchmark. Bold * is the best performance, Bold is the suboptimal performance, and Italics is the third best performance. † represents that the non-based-VLM baselines. 'Scratch' is the work without pre-training on robotic data. 'Params' is the backbone scale, and its unit is ' B illion'. We give the performance on subtasks. It is shown in Table D1 of Appendix D. Recently, we have updated the VLA-Adapter-Pro model. Its Policy architecture is the same as Figure 5, and we optimized the implementation. For its details, please see Appendix I.

| LIBERO                                      | Params   | Spatial   | Object   | Goal   | Long   | Avg.   |
|---------------------------------------------|----------|-----------|----------|--------|--------|--------|
| FlowVLA (Zhong et al., 2025) (ArXiv)        | 8.5      | 93.2      | 95.0     | 91.6   | 72.6   | 88.1   |
| UnifiedVLA (Li et al., 2025a) (ArXiv)       | 8.5      | 95.4      | 98.8     | 93.6   | 94.0   | 95.5   |
| OpenVLA (Kim et al., 2024) (CoRL)           | 7        | 84.7      | 88.4     | 79.2   | 53.7   | 76.5   |
| OpenVLA-OFT (Kim et al., 2025) (RSS)        | 7        | 97.6      | 98.4     | 97.9   | 94.5   | 97.1   |
| UniVLA (Bu et al., 2025) (RSS)              | 7        | 96.5      | 96.8     | 95.6   | 92.0   | 95.2   |
| CoT-VLA (Zhao et al., 2025a) (CVPR)         | 7        | 87.5      | 91.6     | 87.6   | 69.0   | 81.1   |
| WorldVLA (Cen et al., 2025) (ArXiv)         | 7        | 87.6      | 96.2     | 83.4   | 60.0   | 81.8   |
| TraceVLA (Zheng et al., 2024) (ArXiv)       | 7        | 84.6      | 85.2     | 75.1   | 54.1   | 74.8   |
| MolmoAct (Lee et al., 2025) (ArXiv)         | 7        | 87.0      | 95.4     | 87.6   | 77.2   | 86.6   |
| ThinkAct (Huang et al., 2025) (ArXiv)       | 7        | 88.3      | 91.4     | 87.1   | 70.9   | 84.4   |
| PD-VLA (Song et al., 2025b) (ArXiv)         | 7        | 95.5      | 96.7     | 94.9   | 91.7   | 94.7   |
| 4D-VLA (Zhang et al., 2025a) (ArXiv)        | 4        | 88.9      | 95.2     | 90.9   | 79.1   | 88.6   |
| SpatialVLA (Qu et al., 2025) (RSS)          | 4        | 88.2      | 89.9     | 78.6   | 55.5   | 78.1   |
| π 0 (Black et al., 2025b) (RSS)             | 3        | 96.8      | 98.8     | 95.8   | 85.2   | 94.2   |
| π 0 -FAST (Pertsch et al., 2025) (RSS)      | 3        | 96.4      | 96.8     | 88.6   | 60.2   | 85.5   |
| NORA (Hung et al., 2025) (ArXiv)            | 3        | 92.2      | 95.4     | 89.4   | 74.6   | 87.9   |
| SmolVLA (Shukor et al., 2025) (ArXiv)       | 2.2      | 93.0      | 94.0     | 91.0   | 77.0   | 88.8   |
| GR00T N1 (NVIDIA et al., 2025) (ArXiv)      | 2        | 94.4      | 97.6     | 93.0   | 90.6   | 93.9   |
| GraspVLA (Deng et al., 2025) (ArXiv)        | 1.8      | -         | 94.1     | 91.2   | 82.0   | 89.1   |
| Seer † (Tian et al., 2025) (Scratch) (ICLR) | 0.57     | -         | -        | -      | 78.7   | 78.7   |
| VLA-OS (Gao et al., 2025) (ArXiv)           | 0.5      | 87.0      | 96.5     | 92.7   | 66.0   | 85.6   |
| Diffusion Policy † (Chi et al., 2023) (RSS) | -        | 78.3      | 92.5     | 68.3   | 50.5   | 72.4   |
| VLA-Adapter (Ours)                          | 0.5      | 97.8      | 99.2     | 97.2   | 95.0   | 97.3   |
| VLA-Adapter-Pro (Ours)                      | 0.5      | 99.6 *    | 99.6 *   | 98.2 * | 96.4 * | 98.5 * |

## 4.3 PERFORMANCE ON GENERALIZATION TASKS

We used the CALVIN ABC → D (Mees et al., 2022) to evaluate the performance on the zero-shot generalization tasks. CALVIN consists of four environments (Env A, B, C, and D) 2 . 'ABC → D' means it trains on Env A, B, and C and evaluates on Env D. VLA needs to execute a preset sequence of 1,000 tasks in sequence. Each task row consists of five subtasks. The model can only proceed to the next subtask after completing the current one. Please see Appendix E for more settings.

Baselines. We selected recently published works as baselines. They are Large : 1. UniVLA (Bu et al., 2025), 2. OpenVLA (Kim et al., 2024), 3. OpenVLA-OFT (Kim et al., 2025), 4. VLAS (Zhao et al., 2025b), 5. LCB (Shentu et al., 2024), 6. RoboDual (Bu et al., 2024a), 7. OpenHelix (Cui et al., 2025), and 8. ReconVLA (Song et al., 2025c); Small : 9. DeeR (Yue et al., 2024), 10. RoboFlamingo (Li et al., 2024), 11. VPP (Hu et al., 2025), and 12. SuSIE (Black et al., 2024); Tiny : 13. MoDE (Reuss et al., 2025) and 14. Seer (Tian et al., 2025). The results of these baselines are based on original references or other published works, ensuring objectivity and correctness. Since the original OpenVLA-OFT paper (Kim et al., 2025) did not perform experiments on CALVIN ABC → D, we used its source codes to run 150,000 steps and took the best performance.

Metrics. We use the widely used 'Success Rate' (the same in LIBERO (Liu et al., 2023a)) and 'Avg. len' of completed tasks (the larger the better, with values between 0-5) as metrics.

Results. Comparison on the CALVIN is shown in Table 6. The results in Table 6 show that VLAAdapter has strong generalization ability, and its average length is better than SOTA baselines.

[2 http://calvin.cs.uni-freiburg.de/](http://calvin.cs.uni-freiburg.de/)

Table 6: Comparison on the CALVIN ABC → D benchmark. Bold * is the best performance, Bold is the suboptimal performance, and Italics is the third best performance. † represents that the nonbased-VLM method. Recently, we have updated the VLA-Adapter-Pro . Its Policy architecture is the same as Figure 5, and we optimized the implementation. For its details, please see Appendix I.

| CALVIN ABC → D                          | Params   | Task completed in a row ↑   | Task completed in a row ↑   | Task completed in a row ↑   | Task completed in a row ↑   |        | Avg. len ↑   |
|-----------------------------------------|----------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|--------|--------------|
| CALVIN ABC → D                          | Params   | 1                           | 2                           | 3                           | 4                           | 5      | Avg. len ↑   |
| UniVLA (Bu et al., 2025) (RSS)          | 7        | 95.5                        | 85.8                        | 75.4                        | 66.9                        | 56.5   | 3.80         |
| OpenVLA (Kim et al., 2024) (CoRL)       | 7        | 91.3                        | 77.8                        | 62.0                        | 52.1                        | 43.5   | 3.27         |
| OpenVLA-OFT (Kim et al., 2025) (RSS)    | 7        | 96.3                        | 89.1                        | 82.4                        | 75.8                        | 66.5   | 4.10         |
| VLAS (Zhao et al., 2025b) (ICLR)        | 7        | 87.2                        | 64.2                        | 40.9                        | 28.1                        | 19.6   | 2.40         |
| LCB (Shentu et al., 2024) (IROS)        | 7        | 73.6                        | 50.2                        | 28.5                        | 16.0                        | 9.9    | 1.78         |
| RoboDual (Bu et al., 2024a) (ArXiv)     | 7        | 94.4                        | 82.7                        | 72.1                        | 62.4                        | 54.4   | 3.66         |
| OpenHelix (Cui et al., 2025) (ArXiv)    | 7        | 97.1                        | 91.4                        | 82.8                        | 72.6                        | 64.1   | 4.08         |
| ReconVLA (Song et al., 2025c) (ArXiv)   | 7        | 95.6                        | 87.6                        | 76.9                        | 69.3                        | 64.1   | 3.95         |
| DeeR (Yue et al., 2024) (NeurIPS)       | 3        | 86.2                        | 70.1                        | 51.8                        | 41.5                        | 30.4   | 2.82         |
| RoboFlamingo (Li et al., 2024) (ICLR)   | 3        | 82.4                        | 61.9                        | 46.6                        | 33.1                        | 23.5   | 2.48         |
| VPP † (Hu et al., 2025) (ICML)          | 1.5      | 95.7                        | 91.2                        | 86.3                        | 81.0                        | 75.0   | 4.33         |
| SuSIE (Black et al., 2024) (ICLR)       | 1.3      | 87.0                        | 69.0                        | 49.0                        | 38.0                        | 26.0   | 2.69         |
| Seer Large † (Tian et al., 2025) (ICLR) | 0.57     | 96.3                        | 91.6                        | 86.1                        | 80.3                        | 74.0   | 4.28         |
| MoDE † (Reuss et al., 2025) (ICLR)      | 0.44     | 96.2                        | 88.9                        | 81.1                        | 71.8                        | 63.5   | 4.01         |
| Seer † (Tian et al., 2025) (ICLR)       | 0.32     | 94.4                        | 87.2                        | 79.9                        | 72.2                        | 64.3   | 3.98         |
| VLA-Adapter (Ours)                      | 0.5      | 99.1 *                      | 94.6                        | 88.8                        | 82.8                        | 76.5   | 4.42         |
| VLA-Adapter-Pro (Ours)                  | 0.5      | 98.5                        | 95.0 *                      | 90.5 *                      | 85.3 *                      | 80.0 * | 4.50 *       |

## 4.4 PERFORMANCE ON REAL-WORLD TASKS

Experimental settings. We use a robotic system to perform real-world tasks. A 6-DOF Synria Alicia-D equipped with a 1-DOF gripper is employed, and it uses Logitech C920e and RealSense D405 cameras to capture the third-view and gripper images. The real-world robotic system is shown in Figure 6. We evaluate the VLA-Adapter method across four experimental categories:

- 1) Simple pick-and-place tasks with objects spanning diverse materials and geometries.
- 2) CALVIN-inspired challenging task II: lateral block relocation (e.g. 'Move &lt; obj &gt; left/right').
- 3) CALVIN-inspired challenging manipulation task I: 'Block stacking'.
- 4) LIBERO-inspired complex and long-horizon task: (e.g. 'Pick up the spoon and place it on the cup, then place the cup on the plate').

To strengthen evaluation rigor and assess generalization performance, we randomize the object positions at test time to induce distribution shift and increase task difficulty.

Figure 6: Real-world system Synria Alicia-D and the task examples.

<!-- image -->

Baselines. ACT (Zhao et al., 2023) and OFT-style variant (Kim et al., 2025) are as baselines.

Results. The comparison results are shown in Figure 7. Each result is obtained by averaging the results of 10 executions. Experimental results show that VLA-Adapter has better generalization capabilities in various scenarios. Therefore, VLA-Adapter greatly lowers the barrier to adopting VLA in practical applications. More real-world experiments are detailed in Appendix G.

Figure 7: Comparison on real-world tasks.

<!-- image -->

## 4.5 ABLATION EXPERIMENTS

We explore three key components in the VLA-Adapter: 1. Number of ActionQuery, 2. Condition type, and 3. Injection degree for Policy. The benchmark is LIBERO-Long (Liu et al., 2023a).

Number of ActionQuery. In our paradigm, the number of ActionQuery is not fixed. To explore the impact of this number on performance, we conducted the following experiments by varying the number of ActionQuery to 1, 4, 8, 16, 64, 128, 256, and 512. The results are shown in Figure 8. Thus, using too few ActionQuery tokens weakens multimodal aggregation and makes it challenging to condition the Policy. Conversely, employing too many ActionQuery tokens introduces redundancy, interfering with the performance. Therefore, we selected 64 ActionQuery tokens. This number provides the optimal balance between performance and efficiency.

96

<!-- image -->

Success Rate (%)

Figure 8: Comparison of the different numbers of ActionQuery. The blue line shows the result of using only the last-layer ActionQuery. The red star shows the result of the full VLA-Adapter.

Condition Type. In Section 3, we analyzed the overall effects of different conditions on action generation. Here, we present the complete comparison results based on the four classic paradigms in Section 2, as shown in Table 7. This result demonstrates that using both all-layer Raw and ActionQuery achieves superior performance, indirectly validating the superiority of our bridge paradigm.

Injection Degree for Policy. In the Bridge Attention, we use learnable parameters to control the injection degree of Raw features C R t and set the injection degree of ActionQuery features C AQ t to 1. Here, we explore other injection degrees, and the comparison results are shown in Table 8. Two conclusions can be drawn from the results in Table 8: From 1) and 2) , the performance of C R t is inferior to C AQ t , so C R t should inject some effective information into Policy through learning. From 1) and 4) , C AQ t aggregates multimodal information, which is beneficial for action generation; it needs to be injected fully into Policy. This result confirms that the Bridge Attention is effective.

Table 7: Comparison with different condition types. The style can be summarized as representative works in Figure 2 of Section 1. ' N/A ' represents no such method. ' Bold ' is the best performance.

| Layer        | Raw   | ActionQuery   | Style                                                       | SR ↑           |
|--------------|-------|---------------|-------------------------------------------------------------|----------------|
| Last         | ✓ ✗   | ✗ ✓           | RoboVLMs (Liu et al., 2024a) OpenVLA-OFT (Kim et al., 2025) | 85.8 90.2      |
| Intermidiate | ✓     | ✗             | GR00T N1 (NVIDIA et al., 2025)                              | 88.4           |
| All          | ✓ ✗ ✓ | ✗ ✓ ✓         | π 0 (Black et al., 2025b) N/A VLA-Adapter (Ours)            | 90.6 92.6 95.0 |

Table 8: Ablation of other injection degrees.

|                    | Raw       | ActionQuery   |   Success Rate (%) |
|--------------------|-----------|---------------|--------------------|
| 1) ( VLA-Adapter ) | tanh( g ) | 1             |               95.0 |
| 2)                 | 1         | 1             |               91.4 |
| 3)                 | 1         | tanh( g )     |               91.0 |
| 4)                 | tanh( g ) | tanh( g )     |               92.6 |

## 5 CONCLUSION

We propose VLA-Adapter, a novel and efficient bridging paradigm for VLA. By leveraging Raw and ActionQuery latent, this method effectively transfers multimodal knowledge to the Policy to generate action. Experiments show that VLA-Adapter achieves SOTA performance using a tinyscale backbone. Even when the VLM is frozen, it has strong performance. In addition, our method has low VRAM usage and high inference speed. These results suggest that VLA-Adapter alleviates VLA's reliance on large-scale VLMs and huge training costs, lowering the barrier to deploying VLA.

Ultimately, we hope the VLA-Adapter method and key findings of this study can provide a solid basis for future research in the VLA and inspire the development of more advanced VLA methods!

## 6 LIMITATIONS

While VLA-Adapter achieves lightweight and excellent performance, it also has some limitations. First, because VLA-Adapter is not pre-trained on a large amount of embodied data and the scale is tiny, its generalization in real-world systems needs to be improved. Secondly, the quality of the actions generated by the Policy networks depends on the conditions provided by the VLM and how they are used. Therefore, future work can further explore these conditions to improve its representation and ensure its efficient use. Finally, the fundamental training process of the VLA-Adapter is still relatively simple, and the complex processes, such as reinforcement learning, can be explored.

## ACKNOWLEDGMENTS

This work was supported in part by the National Natural Science Foundation of China under Grant U21B2020, and the BUPT Excellent Ph.D. Students Foundation under Grant CX20241055.

## REFERENCES

Kevin Black, Mitsuhiko Nakamoto, Pranav Atreya, Homer Walke, Chelsea Finn, Aviral Kumar, and Sergey Levine. Zero-shot robotic manipulation with pretrained image-editing diffusion models. arXiv preprint arXiv:2310.10639 , 2024. 8, 9

Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π 0. 5: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054 , 2025a. 2

- Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π 0: A vision-language-action flow model for general robot control. corr, abs/2410.24164, 2024. doi: 10.48550. arXiv preprint arXiv:2410.24164 , 2025b. 2, 3, 4, 6, 7, 8, 11
- Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning , pp. 2165-2183. PMLR, 2023a. 2
- Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Tomas Jackson, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Kuang-Huei Lee, Sergey Levine, Yao Lu, Utsav Malla, Deeksha Manjunath, Igor Mordatch, Ofir Nachum, Carolina Parada, Jodilyn Peralta, Emily Perez, Karl Pertsch, Jornell Quiambao, Kanishka Rao, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Kevin Sayed, Jaspiar Singh, Sumedh Sontakke, Austin Stone, Clayton Tan, Huong Tran, Vincent Vanhoucke, Steve Vega, Quan Vuong, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817 , 2023b. 2
- Qingwen Bu, Hongyang Li, Li Chen, Jisong Cai, Jia Zeng, Heming Cui, Maoqing Yao, and Yu Qiao. Towards synergistic, generalized, and efficient dual-system for robotic manipulation. arXiv preprint arXiv:2410.08001 , 2024a. 2, 8, 9
- Qingwen Bu, Jia Zeng, Li Chen, Yanchao Yang, Guyue Zhou, Junchi Yan, Ping Luo, Heming Cui, Yi Ma, and Hongyang Li. Closed-loop visuomotor control with generative expectation for robotic manipulation. Advances in Neural Information Processing Systems , 37:139002-139029, 2024b. 2
- Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Univla: Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111 , 2025. 7, 8, 9
- Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, Deli Zhao, and Hao Chen. Worldvla: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539 , 2025. 1, 7, 8
- Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, Hanbo Zhang, and Minzhao Zhu. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158 , 2024. 2
- Chilam Cheang, Sijin Chen, Zhongren Cui, Yingdong Hu, Liqun Huang, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Xiao Ma, Hao Niu, Wenxuan Ou, Wanli Peng, Zeyu Ren, Haixin Shi, Jiawen Tian, Hongtao Wu, Xin Xiao, Yuyang Xiao, Jiafeng Xu, and Yichu Yang. Gr-3 technical report. arXiv preprint arXiv:2507.15493 , 2025. 2

- Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research , pp. 02783649241273668, 2023. 7, 8
- Open X-Embodiment Collaboration, Abhishek Padalkar, Acorn Pooley, Ajay Mandlekar, Ajinkya Jain, Albert Tung, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anikait Singh, Animesh Garg, Anthony Brohan, Antonin Raffin, Ayzaan Wahid, Ben BurgessLimerick, Beomjoon Kim, Bernhard Sch¨ olkopf, Brian Ichter, Cewu Lu, Charles Xu, Chelsea Finn, Chenfeng Xu, Cheng Chi, Chenguang Huang, Christine Chan, Chuer Pan, Chuyuan Fu, Coline Devin, Danny Driess, Deepak Pathak, Dhruv Shah, Dieter B¨ uchler, Dmitry Kalashnikov, Dorsa Sadigh, Edward Johns, Federico Ceola, Fei Xia, Freek Stulp, Gaoyue Zhou, Gaurav S. Sukhatme, Gautam Salhotra, Ge Yan, Giulio Schiavi, Gregory Kahn, Hao Su, Hao-Shu Fang, Haochen Shi, Heni Ben Amor, Henrik I. Christensen, Hiroki Furuta, Homer Walke, Hongjie Fang, Igor Mordatch, Ilija Radosavovic, Isabel Leal, Jacky Liang, Jad Abou-Chakra, Jaehyung Kim, Jan Peters, Jan Schneider, Jasmine Hsu, Jeannette Bohg, Jeffrey Bingham, Jiajun Wu, Jialin Wu, Jianlan Luo, Jiayuan Gu, Jie Tan, Jihoon Oh, Jitendra Malik, Jonathan Booher, Jonathan Tompson, Jonathan Yang, Joseph J. Lim, Jo˜ ao Silv´ erio, Junhyek Han, Kanishka Rao, Karl Pertsch, Karol Hausman, Keegan Go, Keerthana Gopalakrishnan, Ken Goldberg, Kendra Byrne, Kenneth Oslund, Kento Kawaharazuka, Kevin Zhang, Krishan Rana, Krishnan Srinivasan, Lawrence Yunliang Chen, Lerrel Pinto, Li FeiFei, Liam Tan, Lionel Ott, Lisa Lee, Masayoshi Tomizuka, Max Spero, Maximilian Du, Michael Ahn, Mingtong Zhang, Mingyu Ding, Mohan Kumar Srirama, Mohit Sharma, Moo Jin Kim, Naoaki Kanazawa, Nicklas Hansen, Nicolas Heess, Nikhil J. Joshi, Niko Suenderhauf, Norman Di Palo, Nur Muhammad Mahi Shafiullah, Oier Mees, Oliver Kroemer, Pannag R. Sanketi, Paul Wohlhart, Peng Xu, Pierre Sermanet, Priya Sundaresan, Quan Vuong, Rafael Rafailov, Ran Tian, Ria Doshi, Roberto Mart´ ın-Mart´ ın, Russell Mendonca, Rutav Shah, Ryan Hoque, Ryan Julian, Samuel Bustamante, Sean Kirmani, Sergey Levine, Sherry Moore, Shikhar Bahl, Shivin Dass, Shubham Sonawani, Shuran Song, Sichun Xu, Siddhant Haldar, Simeon Adebola, Simon Guist, Soroush Nasiriany, Stefan Schaal, Stefan Welker, Stephen Tian, Sudeep Dasari, Suneel Belkhale, Takayuki Osa, Tatsuya Harada, Tatsuya Matsushima, Ted Xiao, Tianhe Yu, Tianli Ding, Todor Davchev, Tony Z. Zhao, Travis Armstrong, Trevor Darrell, Vidhi Jain, Vincent Vanhoucke, Wei Zhan, Wenxuan Zhou, Wolfram Burgard, Xi Chen, Xiaolong Wang, Xinghao Zhu, Xuanlin Li, Yao Lu, Yevgen Chebotar, Yifan Zhou, Yifeng Zhu, Ying Xu, Yixuan Wang, Yonatan Bisk, Yoonyoung Cho, Youngwoon Lee, Yuchen Cui, Yueh-Hua Wu, Yujin Tang, Yuke Zhu, Yunzhu Li, Yusuke Iwasawa, Yutaka Matsuo, Zhuo Xu, and Zichen Jeff Cui. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA) , pp. 6892-6903. IEEE, 2024. 2
- Can Cui, Pengxiang Ding, Wenxuan Song, Shuanghao Bai, Xinyang Tong, Zirui Ge, Runze Suo, Wanqi Zhou, Yang Liu, Bofang Jia, Han Zhao, Siteng Huang, and Donglin Wang. Openhelix: A short survey, empirical analysis, and open-source dual-system vla model for robotic manipulation. arXiv preprint arXiv:2505.03912 , 2025. 1, 2, 3, 8, 9
- Shengliang Deng, Mi Yan, Songlin Wei, Haixin Ma, Yuxin Yang, Jiayi Chen, Zhiqi Zhang, Taoyu Yang, Xuheng Zhang, Wenhao Zhang, Heming Cui, Zhizheng Zhang, and He Wang. Graspvla: a grasping foundation model pre-trained on billion-scale synthetic action data. arXiv preprint arXiv:2505.03233 , 2025. 7, 8
- Pengxiang Ding, Han Zhao, Wenjie Zhang, Wenxuan Song, Min Zhang, Siteng Huang, Ningxi Yang, and Donglin Wang. Quar-vla: Vision-language-action model for quadruped robots. In European Conference on Computer Vision , pp. 352-367. Springer, 2024. 2
- Yiguo Fan, Pengxiang Ding, Xinyang Tong Shuanghao Bai, Yuyang Zhu, Hongchao Lu, Fengqi Dai, Wei Zhao, Yang Liu, Zhaoxin Fan Siteng Huang, Badong Chen, and Donglin Wang. Longvla: Unleashing long-horizon capability of vision language action model for robot manipulation. arXiv preprint arXiv:2508.19958 , 2025. 1, 2
- Chongkai Gao, Zixuan Liu, Zhenghao Chi, Junshan Huang, Xin Fei, Yiwen Hou, Yuxuan Zhang, Yudi Lin, Zhirui Fang, Zeyu Jiang, and Lin Shao. Vla-os: Structuring and dissecting planning representations and paradigms in vision-language-action models. arXiv preprint arXiv:2506.17561 , 2025. 7, 8

- Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. ICLR , 1(2):3, 2022. 6, 21
- Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, and Jianyu Chen. Video prediction policy: A generalist robot policy with predictive visual representations. arXiv preprint arXiv:2412.14803 , 2025. 8, 9
- Chi-Pin Huang, Yueh-Hua Wu, Min-Hung Chen, Yu-Chiang Frank Wang, and Fu-En Yang. Thinkact: Vision-language-action reasoning via reinforced visual latent planning. arXiv preprint arXiv:2507.16815 , 2025. 7, 8
- Chia-Yu Hung, Qi Sun, Pengfei Hong, Amir Zadeh, Chuan Li, U Tan, Navonil Majumder, Soujanya Poria, et al. Nora: A small open-sourced generalist vision language action model for embodied tasks. arXiv preprint arXiv:2504.19854 , 2025. 7, 8
- Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, and Hang Zhao. Galaxea open-world dataset and g0 dual-system vla model. 2025. 2
- Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. In Forty-first International Conference on Machine Learning (ICML) , 2024. 1, 2, 3, 6
- Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe, Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul Foster, Jensen Gao, Vitor Guizilini, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Muhammad Zubair Irshad, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen, Abigail O'Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph J Lim, Jitendra Malik, Roberto Mart´ ın-Mart´ ın, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945 , 2024. 2
- Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model. In The Conference on Robot Learning (CoRL) , 2024. 1, 2, 3, 6, 7, 8, 9
- Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645 , 2025. 1, 2, 3, 6, 7, 8, 9, 10, 11, 20
- Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, Winson Han, Wilbert Pumacay, Angelica Wu, Rose Hendrix, Karen Farley, Eli VanderBilt, Ali Farhadi, Dieter Fox, and Ranjay Krishna. Molmoact: Action reasoning models that can reason in space. arXiv preprint arXiv:2508.07917 , 2025. 7, 8
- Shuang Li, Yihuai Gao, Dorsa Sadigh, and Shuran Song. Unified video action model. arXiv preprint arXiv:2503.00200 , 2025a. 7, 8

- Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, et al. Vision-language foundation models as effective robot imitators. arXiv preprint arXiv:2311.01378 , 2024. 2, 8, 9
- Zhiqi Li, Guo Chen, Shilong Liu, Shihao Wang, Vibashan VS, Yishen Ji, Shiyi Lan, Hao Zhang, Yilin Zhao, Subhashree Radhakrishnan, Nadine Chang, Karan Sapra, Amala Sanjay Deshmukh, Tuomas Rintamaki, Matthieu Le, Ilia Karmanov, Lukas Voegtle, Philipp Fischer, De-An Huang, Timo Roman, Tong Lu, Jose M. Alvarez, Bryan Catanzaro, Jan Kautz, Andrew Tao, Guilin Liu, and Zhiding Yu. Eagle 2: Building post-training data strategies from scratch for frontier visionlanguage models. arXiv preprint arXiv:2501.14818 , 2025b. 1, 2
- Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems , 36:44776-44791, 2023a. 2, 4, 6, 7, 8, 10, 18, 20, 21, 28
- Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems , 36:34892-34916, 2023b. 1, 2
- Huaping Liu, Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, and Hanbo Zhang. Towards generalist robot policies: What matters in building vision-language-action models. 2024a. 2, 3, 11
- Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864 , 2024b. 1, 2
- Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 , 2019. 21
- Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, and Zongqing Lu. Being-h0: vision-language-action pretraining from largescale human videos. arXiv preprint arXiv:2507.15597 , 2025. 2
- Oier Mees, Lukas Hermann, Erick Rosete-Beas, and Wolfram Burgard. Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters , 7(3):7327-7334, 2022. 2, 6, 8, 21
- NVIDIA, Johan Bjorck, Fernando Casta˜ neda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Jim Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. Gr00t n1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734 , 2025. 2, 3, 6, 7, 8, 11
- Maxime Oquab, Timoth´ ee Darcet, Th´ eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 , 2024. 3
- Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, and Saining Xie. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256 , 2025. 4
- William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision , pp. 4195-4205, 2023. 6, 18
- Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747 , 2025. 7, 8
- Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visuallanguage-action model. arXiv preprint arXiv:2501.15830 , 2025. 7, 8

- Moritz Reuss, Jyothish Pari, Pulkit Agrawal, and Rudolf Lioutikov. Efficient diffusion transformer policies with mixture of expert denoisers for multitask learning. arXiv preprint arXiv:2412.12953 , 2025. 8, 9
- Yide Shentu, Philipp Wu, Aravind Rajeswaran, and Pieter Abbeel. From llms to actions: Latent codes as bridges in hierarchical robot control. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , pp. 8539-8546. IEEE, 2024. 2, 8, 9
- Hao Shi, Bin Xie, Yingfei Liu, Lin Sun, Fengrong Liu, Tiancai Wang, Erjin Zhou, Haoqiang Fan, Xiangyu Zhang, and Gao Huang. Memoryvla: Perceptual-cognitive memory in vision-languageaction models for robotic manipulation. 2025. 1
- Mustafa Shukor, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Michel Aractingi Adil Zouitine, Caroline Pascal, Martino Russi, Andres Marafioti, Simon Alibert, Matthieu Cord, Thomas Wolf, and Remi Cadene. Smolvla: A vision-language-action model for affordable and efficient robotics. arXiv preprint arXiv:2506.01844 , 2025. 2, 3, 6, 7, 8
- Haoming Song, Delin Qu, Yuanqi Yao, Qizhi Chen, Qi Lv, Yiwen Tang, Modi Shi, Guanghui Ren, Maoqing Yao, Bin Zhao, Dong Wang, and Xuelong Li. Hume: Introducing system-2 thinking in visual-language-action model. arXiv preprint arXiv:2505.21432 , 2025a. 2, 6
- Wenxuan Song, Jiayi Chen, Pengxiang Ding, Han Zhao, Wei Zhao, Zhide Zhong, Zongyuan Ge, Jun Ma, and Haoang Li. Accelerating vision-language-action model integrated with action chunking via parallel decoding. arXiv preprint arXiv:2503.02310 , 2025b. 1, 7, 8
- Wenxuan Song, Ziyang Zhou, Han Zhao, Jiayi Chen, Pengxiang Ding, Haodong Yan, Yuxin Huang, Feilong Tang, Donglin Wang, and Haoang Li. Reconvla: Reconstructive vision-language-action model as effective robot perceiver. arXiv preprint arXiv:2508.10333 , 2025c. 8, 9
- Andreas Steiner, Andr´ e Susano Pinto, Michael Tschannen, Daniel Keysers, Xiao Wang, Yonatan Bitton, Alexey Gritsenko, Matthias Minderer, Anthony Sherbondy, Shangbang Long, Siyang Qin, Reeve Ingle, Emanuele Bugliarello, Sahar Kazemzadeh, Thomas Mesnard, Ibrahim Alabdulmohsin, Lucas Beyer, and Xiaohua Zhai. Paligemma 2: A family of versatile vlms for transfer. arXiv preprint arXiv:2412.03555 , 2024. 1, 2
- Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213 , 2024. 1
- Qwen Team. Qwen2 technical report. arXiv preprint arXiv:2407.10671 , 2, 2024. 3, 6
- Yang Tian, Sizhe Yang, Jia Zeng, Ping Wang, Dahua Lin, Hao Dong, and Jiangmiao Pang. Predictive inverse dynamics models are scalable learners for robotic manipulation. arXiv preprint arXiv:2412.15109 , 2025. 7, 8, 9
- Xinyang Tong, Pengxiang Ding, Yiguo Fan, Donglin Wang, Wenjie Zhang, Can Cui, Mingyang Sun, Han Zhao, Hongyin Zhang, Yonghao Dang, Siteng Huang, and Shangke Lyu. Quart-online: Latency-free large multimodal language model for quadruped robot learning. arXiv preprint arXiv:2412.15576 , 2024. 2
- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 , 2023. 3, 6

- Yang Yue, Yulin Wang, Bingyi Kang, Yizeng Han, Shenzhi Wang, Shiji Song, Jiashi Feng, and Gao Huang. Deer-vla: Dynamic inference of multimodal large language models for efficient robot execution. Advances in Neural Information Processing Systems , 37:56619-56643, 2024. 8, 9
- Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision , pp. 11975-11986, 2023. 3
- Jiahui Zhang, Yurui Chen, Yueming Xu, Ze Huang, Yanpeng Zhou, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, Xingyue Quan, Hang Xu, and Li Zhang. 4d-vla: Spatiotemporal vision-languageaction pretraining with cross-scene calibration. arXiv preprint arXiv:2506.22242 , 2025a. 7, 8
- Jianke Zhang, Yanjiang Guo, Xiaoyu Chen, Yen-Jen Wang, Yucheng Hu, Chengming Shi, and Jianyu Chen. Hirt: Enhancing robotic control with hierarchical robot transformers. arXiv preprint arXiv:2410.05273 , 2024. 2, 3
- Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zeroinit attention. arXiv preprint arXiv:2303.16199 , 2023. 5
- Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, Xinqiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, Fan Lu, He Wang, Zhizheng Zhang, Li Yi, Wenjun Zeng, and Xin Jin. Dreamvla: A vision-language-action model dreamed with comprehensive world knowledge. arXiv preprint arXiv:2507.04447 , 2025b. 1
- Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xiang, Gordon Wetzstein, and Tsung-Yi Lin. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference , pp. 17021713, 2025a. 7, 8
- Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705 , 2023. 10
- Wei Zhao, Pengxiang Ding, Min Zhang, Zhefei Gong, Shuanghao Bai, Han Zhao, and Donglin Wang. Vlas: Vision-language-action model with speech instructions for customized robot manipulation. arXiv preprint arXiv:2502.13508 , 2025b. 8, 9
- Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daum´ e III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. arXiv preprint arXiv:2412.10345 , 2024. 7, 8
- Zhide Zhong, Haodong Yan, Junfeng Li, Xiangchen Liu, Xin Gong, Wenxuan Song, Jiayi Chen, and Haoang Li. Flowvla: Thinking in motion with a visual chain of thought. 2025. 1, 7, 8

## Appendix of VLA-Adapter

## A SETUP DETAILS OF LIBERO SIMULATION BENCHMARKS

The LIBERO benchmark (Liu et al., 2023a) comprises four distinct task suites: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-100. The first three suites each contain 10 tasks, and LIBERO-100 contains 90 short-term tasks (LIBERO-90) and 10 long-horizon tasks (LIBEROLong). The strategy for each task depends solely on the current instructions provided. Each task is repeated multiple times (50 repetitions in this paper) to obtain the average success rate for each subtask. The examples and the instructions in the LIBERO benchmark are shown in Figure A1.

Figure A1: The examples and the task instructions on the LIBERO benchmark.

<!-- image -->

In the LIBERO benchmark, we use third-person images (resolution 224 × 224 × 3, RGB) and wrist images (resolution 224 × 224 × 3, RGB) as visual input. The task instruction is first constructed as a prompt in a specific format: ' In: What action should the robot take to instruction.lower()? \ nOut: ', and then input into the VLM module together with the image information. Its output is a 7-dimensional action vector, which is used to control the 7-DOF Franka Emika Panda simulated robot arm to perform the corresponding action sequence.

## B DIT-BASED POLICY NETWORK

## B.1 OVERALL ARCHITECTURE

This architecture is shown in Figure B1. It consists of τ -DiT blocks, 1 ≤ τ ≤ M . It has the same number of layers as VLM. Each DiT block consists of three components: conditional modulation, conditional attention, and a conditional feedforward network. At timestep t , the input chunk to the first-DiT block, A 1 t , is a noisy action sequence. Since the input contains random noise, to facilitate the transition from 'pure noise → fi ne-grained prediction', we adopt the AdaLN-Zero layer (Peebles &amp; Xie, 2023) to modulate the activation amplitude at each layer. The AdaLN-Zero consists of LayerNorm, modulation, and gated residual. Specifically, C M t will be obtained by P t and C R t , i.e., C M t = σ ′ 1 ( C R t ) + σ 0 ( P t ) . It is used to generate 'Scale' and 'Shift' vectors, which guide the activation direction of intermediate features and inject automatic modulation amplitude via gated residual control. After modulation, ˜ A 1 t = [ ˜ a 1 t , ˜ a 1 t +1 , . . . , ˜ a 1 t + H -1 ] is obtained:

<!-- formula-not-decoded -->

Figure B1: The DiT-based policy network.

<!-- image -->

V

M

2 l

where, β τ and γ τ are scaling factors and offset factors, which are dynamically generated by C M t through a projection new with the SiLU. α τ is the gated residual coefficient, used to adjust the injection amplitude. It is dynamically generated by C M t through σ 0 with the SiLU, and has α τ = σ ′ 3 ( C M t ) . ⊙ is element-wise multiplication, and ε ( · ) is self-attention and projection modules.

After conditional modulation, ˜ A 1 t is used as the QKV vector, and C R t , C AQ t are used as the KV vectors for Bridge Attention. The details of Bridge Attention are shown in Section 3.3. And then, the attention latent ̂ A 1 t will be obtained. ̂ A 1 t is input into the conditional feedforward network. The first-DiT block output A 2 t is obtained. After passing through M DiT blocks, we get the ̂ A t M , which is passed by a LayerNorm and MLP layer to generate the current action chunk A t M .

## B.2 TRAINING OF DIT-BASED POLICY

This Policy is also trained from scratch. Given a ground truth action trajectory A t , a noisy action in DiT-based Policy A τ t = √ α τ A t + √ 1 -α τ ϵ , where, √ α τ is the cumulative product of noise coefficients, and has √ α τ = ∏ T i =1 α i = ∏ T i =1 (1 -β i ) , β i is the variances used at each step, and ϵ ∼ N (0 , I ) is Gaussian noise. We train DiT-based model π θ ( · ) with the training objectives:

<!-- formula-not-decoded -->

## B.3 BRIEF COMPARISON WITH L1-BASED POLICY

As exploring the Policy architecture is not the primary focus of this paper, we briefly compare the performance of the L1-based and DiT-based Policy networks on the LIBERO-Long benchmark.

Table B1: Comparison of the L1-based and DiT-based Policy networks of VLA-Adapter. Bold represents the best results. For detailed task instructions, please see Figure A1 in Appendix A.

| Task instructions   |    1 |    2 |     3 |    4 |     5 |     6 |    7 |     8 |    9 |   10 |   Avg. ↑ |
|---------------------|------|------|-------|------|-------|-------|------|-------|------|------|----------|
| L1-based            | 96.0 | 96.0 | 100.0 | 98.0 | 100.0 | 100.0 | 84.0 |  96.0 | 84.0 | 96.0 |     95.0 |
| DiT-based           | 96.0 | 92.0 |  98.0 | 96.0 |  90.0 |  98.0 | 82.0 | 100.0 | 74.0 | 90.0 |     91.6 |

Results presented in Table B1 indicate that the L1-based Policy achieves superior performance compared to the DiT-based Policy. This phenomenon coincides with the conclusions in OpenVLA-OFT (Kim et al., 2025): Diffusion-type Policy performs better during pre-training, but L1-based Policy outperforms Diffusion-type Policy during fine-tuning because their actions are less redundant. Furthermore, consistent with findings from OpenVLA-OFT, the L1-based Policy achieves higher throughput compared to the Diffusion-type Policy. Therefore, we chose the L1-based Policy.

## C DETAILED COMPARISON RESULTS OF DIFFERENT CODITIONS

In this section, we give the specific performance of the ten subtasks on the LIBERO-Long benchmark in Section 3 to explore different conditions. The specific results are shown in Tables C1 and C2.

Table C1: The specific performance of different layers of Raw features on 10 subtasks. For detailed task instructions, please see Figure A1 in Appendix A. Bold represents the best performance of the average success rate. Italics * represents the suboptimal performance of the average success rate.

|              |             |   Subtasks |   Subtasks |   Subtasks |   Subtasks | Subtasks   | Subtasks   |   Subtasks |   Subtasks |   Subtasks |   Subtasks |        |
|--------------|-------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|--------|
| Raw feature  | Raw feature |          1 |          2 |          3 |          4 | 5 6        | 5 6        |          7 |          8 |          9 |         10 | Avg.   |
|              | 1           |         78 |         96 |         94 |        100 | 96         | 98         |         62 |         90 |         68 |         88 | 87.6   |
|              | 5           |         82 |         94 |         84 |         98 | 94         | 96         |         68 |         94 |         66 |         90 | 86.6   |
|              | 9           |         94 |         94 |         84 |         94 | 90         | 98         |         90 |         90 |         74 |         90 | 89.8 * |
| Single-layer | 13          |         90 |         94 |         86 |         92 | 86         | 100        |         82 |         96 |         84 |         74 | 88.4 * |
|              | 17          |         82 |         92 |         92 |         96 | 92         | 90         |         66 |         72 |         62 |         86 | 84.4   |
|              | 21          |         78 |         94 |         98 |         90 | 68         | 92         |         66 |         94 |         78 |         88 | 83.2   |
|              | 24          |         84 |         96 |         94 |         94 | 94         | 100        |         64 |         88 |         56 |         88 | 85.8   |
| All-layer    | 1-24        |         92 |         98 |         96 |        100 | 84         | 94         |         76 |         96 |         84 |         86 | 90.6   |

Table C2: The specific performance of different layers of ActionQuery features on subtasks. For task instructions, please see Figure A1 in Appendix A. Bold represents the best performance of the average success rate. Italics * represents the suboptimal performance of the average success rate.

| ActionQuery   | ActionQuery   |   Subtasks |   Subtasks |   Subtasks |   Subtasks |   Subtasks |   Subtasks |   Subtasks |   Subtasks |   Subtasks |   Subtasks | Avg.   |
|---------------|---------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|--------|
|               | feature       |          1 |          2 |          3 |          4 |          5 |          6 |          7 |          8 |          9 |         10 |        |
| Single-layer  | 1             |         28 |         50 |         98 |         96 |         80 |         92 |         76 |         94 |         78 |         90 | 78.2   |
| Single-layer  | 13            |         16 |         52 |         98 |         94 |         94 |        100 |         66 |         98 |         62 |         86 | 76.6   |
| Single-layer  | 17            |         90 |         94 |         86 |         88 |         82 |        100 |         74 |        100 |         58 |         96 | 86.8   |
| Single-layer  | 21            |         88 |         92 |         94 |         98 |         92 |         92 |         70 |         96 |         72 |         94 | 88.8   |
| Single-layer  | 23            |         92 |         98 |         94 |         96 |         96 |        100 |         70 |         98 |         72 |         82 | 89.6 * |
| Single-layer  | 24            |         92 |         88 |        100 |         98 |         90 |         96 |         74 |         98 |         84 |         82 | 90.2 * |
| All-layer     | 1-24          |         92 |         94 |         96 |         98 |        100 |         98 |         76 |         98 |         78 |         96 | 92.6   |

In Table C1, the middle-layer Raw features generally outperform other-layer Raw features. In Table C2, deep-layer ActionQuery features generally perform better than shallow-layer ActionQuery's. In addition, in Table C1 and Table C2, the performance of all layers is the best. Therefore, in VLAAdapter, we use the Raw feature and ActionQuery feature of all layers as conditions for Policy.

## D PERFORMANCE ON LIBERO SUBTASKS

In this section, we demonstrate the performance of VLA-Adapter on 40 (4 × 10) subtasks on the LIBERO benchmark (Liu et al., 2023a). The detailed performance is shown in Table D1.

Table D1: The specific performance of VLA-Adapter on the 40 subtasks of four LIBERO (Liu et al., 2023a) suites. For detailed task instructions, please see Figure A1 in Appendix A.

| LIBERO   |    1 |     2 |     3 |     4 |     5 |     6 |     7 |     8 |     9 |    10 |   Avg. ↑ |
|----------|------|-------|-------|-------|-------|-------|-------|-------|-------|-------|----------|
| Spatial  | 98.0 | 100.0 | 100.0 |  90.0 |  96.0 | 100.0 | 100.0 | 100.0 |  98.0 |  96.0 |     97.8 |
| Object   | 98.0 |  98.0 | 100.0 | 100.0 |  98.0 | 100.0 |  98.0 | 100.0 | 100.0 | 100.0 |     99.2 |
| Goal     | 92.0 | 100.0 |  98.0 |  96.0 | 100.0 |  98.0 |  94.0 | 100.0 |  98.0 |  96.0 |     97.2 |
| Long     | 96.0 |  96.0 | 100.0 |  98.0 | 100.0 | 100.0 |  84.0 |  96.0 |  84.0 |  96.0 |     95.0 |

## E SETUP DETAILS OF CALVIN SIMULATION BENCHMARK

Benchmark. We used the CALVIN ABC → D (Mees et al., 2022) to evaluate the performance on the zero-shot generalization tasks. CALVIN consists of four environments (Env A, B, C, and D). 'ABC → D' means it trains on Env A, B, and C and evaluates on Env D. These environments collectively include over two million human demonstration trajectories totaling approximately six hours. The CALVIN benchmark contains 34 different subtasks. By screening the combination of five consecutive subtasks, 1,000 unique instruction chains with rationality and diversity are finally generated. In each instruction chain, the agent needs to complete five subtasks in sequence, and can only proceed to the next subtask after successfully completing the current subtask. The benchmark aims to evaluate generalization capabilities and task execution performance under diverse conditions. Examples of each environment and the task instructions are shown in Figure E1. Move slider right/left Open/Close drawer Lift { object } table/drawer Lift { object } slider Place in slider/drawer Stack/Unstack block Turn on/off lightbulb/led Push into drawer [move\_door\_rel, 'base\_slide', ±0.23] [move\_door\_rel, 'base\_drawer', ±0.12] [lift\_object, '{ object }', 0.05, 'table', 'base\_link/drawer\_link'] [lift\_object, '{ object }', 0.03, 'table', 'plank\_link'] [stack\_objects/unstack\_objects] [place\_object, 'table', 'plank\_link/drawer\_link'] [toggle\_light, 'lightbulb/led', 0/1, 1/0] [push\_object\_into, [{ object }], 'table', 'base\_link', 'table', 'drawer\_link'] Env A Env B Env C Env D 3. 4. 5. 6. 7. 8.

Figure E1: The example and task completion conditions on the CALVIN ABC → D.

<!-- image -->

Put both the alphabet soup and the tomato sauce in the basket Put both the cream cheese box and the butter in the basket Turn on the stove and put the moka pot on it Put the black bowl in the bottom drawer of the cabinet and close it Task instruction 1. 2. 3. 4. LIBERO-Object LIBERO-Goal In the CALVIN ABC → D benchmark, we use third-person images (resolution 224 × 224 × 3, RGB) and Gripper images (resolution 84 × 84 × 3, RGB) as visual input. The task instruction is first constructed as a prompt in a specific format: ' In: What action should the robot take to { Task instruction } ? \ nOut: ', and then input into the VLM module together with the image information. Its output is a 7-dimensional action vector, which is used to control the 7-DOF Franka Emika Panda simulated robot arm to perform the corresponding action sequence.

LIBERO-Spatial LIBERO-Long

5.

6.

7.

8.

9.

Put the white mug on the left plate and put the yellow and white mug on the right plate

Pick up the book and place it in the back compartment of the caddy

Put the white mug on the plate and put the chocolate pudding to the right of the plate

Put both the alphabet soup and the cream cheese box in the basket

Put both moka pots on the stove

## Put the yellow and white mug in the microwave and close it 10. F SUPPLEMENTARY DETAILS OF TRAINING AND HYPERPARAMETERS

## F.1 TRAINING DETAILS

During VLA-Adapter training, we use the AdamW (Loshchilov &amp; Hutter, 2019) optimizer and LoRA scheme (Hu et al., 2022). To ensure the stability of training, the learning rate is set to 1e-4, and the cosine-annealing scheduler with warm-up steps is used. Our batch size is set to 16.

Table F1: The detail settings of Training.

| Setting           | value   |
|-------------------|---------|
| Batch size        | 16      |
| Max training step | 150,000 |
| Learning rate     | 1e-4    |
| Warmup step       | 10%     |

## F.2 HYPERPARAMETER DETAILS

We list the hyperparameters of VLA-Adapter. Their corresponding values are shown in Table F2.

Table F2: Specific hyperparameters of VLA-Adapter and their corresponding values.

| Backbone                                  | Qwen2.5-0.5B   |
|-------------------------------------------|----------------|
| Layer ( τ / M )                           | 24             |
| Number of ActionQuery                     | 64             |
| Hidden size                               | 896            |
| Attention head                            | 8              |
| Action chunk ( H )                        | 8              |
| Intermediate layers of VLM                | 1-24           |
| Total trainable parameters of Policy      | 97.3M          |
| Total trainable parameters of VLA-Adapter | 197.2M         |

## G EXECUTION EXAMPLES

We provide some execution examples, please see Figure G1 and Figure G2 for details.

## G.1 REAL-WORLD EXAMPLES

These include long-horizon tasks: 'Pick up the spoon and place it on the cup, then place the cup on the plate' and short-horizon tasks: 'Stack red blocks on top of blue blocks', 'Move the blue block to the right', and 'Pick up the duck and place it on a plate'. The settings of real-world experiments are shown in Section 4.4.

<!-- image -->

Real-World

: Pick up the duck and place it on a plate

Figure G1: Execution example on the real-world tasks.

## G.2 SIMULATION EXAMPLES

<!-- image -->

: Turn off lightbulb

CALVIN ABC

➝

D

➝

Move slider left

➝

Push blue block left

➝

Lift pink block slider

➝

Stack block

Figure G2: Execution example on the LIBERO and CALVIN ABC → D tasks.

## H EFFECTIVENESS ANALYSIS OF FROZEN BACKBONE

CALVIN ABC ➝ D : Open drawer ➝ Lift pink block table ➝ Place in slider ➝ Turn on lightbulb ➝ Rotate blue block left Section 4.1 compares the effectiveness of the frozen backbone. The results show that OpenVLAOFT does not work. Although it also uses learnable tokens, it is implemented (line 620 in 3 ):

```
# === Handle Multimodal Forward === elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]): ... # Process action embeddings if noisy_actions is not None: ... else: # Replace the embeddings of the action tokens with zeros # (Later on, the positional embeddings will be added to them) all_actions_mask = all_actions_mask.unsqueeze(-1) input_embeddings = input_embeddings * ˜all_actions_mask
```

The tokens added in the 'else:' (L1 architecture) are input to the VLM in the form of a mask. It is initially all zeros, and when the VLM backbone is frozen, it is not trained. Our ActionQuery is:

```
3 https://github.com/moojink/openvla-oft/blob/main/prismatic/extern/hf/ modeling_prismatic.py
```

```
# === Handle Multimodal Forward === elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]): ... # Process action embeddings if noisy_actions is not None: ... else: action_queries = self.action_queries.weight # (1, h) action_queries = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(input_embeddings.shape[0], 1, 1) all_actions_mask = self._process_action_masks(labels) input_embeddings = self._replace_input_embeddings(input_embeddings, all_actions_mask, action_queries)
```

Instead of inputting the VLM in the form of a mask, essentially learnable tokens (or multiple tokens) that is inserted into the specified position in the sequence and participate in attention. Therefore, when the VLM is frozen, the VL information is indeed not trained. Still, ActionQuery is not an original part of the VLM, and its parameters can be learned from scratch, so the VLA-Adapter still works. Below, we give examples of OpenVLA-OFT and VLA-Adapter, as shown in Figure H1.

Instruction:

Put both the alphabet soup and the tomato sauce in the basket

<!-- image -->

Ours

(True)

Figure H1: Execution example when the backbone is frozen.

## I DESIGN DETAILS AND PERFORMANCE OF VLA-ADAPTER-PRO

To achieve extreme lightweightness in VLA-Adapter, we shared the projection layer, using the same projection layer for all three attention matrices. This resulted in a low parameter count of 97MB.

In the VLA-Adapter-Pro version, we separated the projection layers for the three attention matrices, allowing different channels to learn differentiated representations. In this case, the parameter count was 207MB. Furthermore, VLA-Adapter-Pro adds Rotary Position Embedding to QK, making the cross-attention more sensitive to position information and more suitable for action generation.

Next, we give the key architecture codes of VLA-Adapter-Pro, as shown below:

```
class MLPResNetBlock_Pro(nn.Module): """One MLP ResNet block with separate projections for self, adapter, task + RoPE, now without FiLM modulation.""" def __init__(self, dim, num_heads=8): super().__init__() self.dim = dim self.num_heads = num_heads self.head_dim = dim // num_heads self.ffn = nn.Sequential( nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU(), ) # Q (from x only) self.q_proj = nn.Linear(dim, dim) # Self-Attention: K, V self.k_self = nn.Linear(dim, dim) self.v_self = nn.Linear(dim, dim) # Adapter cross-attention: K, V self.k_adapter = nn.Linear(dim, dim) self.v_adapter = nn.Linear(dim, dim) # Task cross-attention: K, V self.k_task = nn.Linear(dim, dim) self.v_task = nn.Linear(dim, dim) self.o_proj = nn.Linear(dim, dim) # gating self.gating_factor = nn.Parameter(torch.zeros(1)) # RoPE self.rope = RotaryPositionEmbedding(self.head_dim) # ----FiLM ----# # FiLM is useless; to avoid conflict with chkpt, it can be kept as is for now. self.film_gen = nn.Sequential( nn.Linear(dim, dim * 2), ) def apply_film(self, x, gamma, beta): """FiLM: per-channel modulation""" return gamma.unsqueeze(1) * x + beta.unsqueeze(1) def forward(self, x, h_a=None, h_t=None, p=None): """ h_a: adapter tokens h_t: task tokens p: possible conditioning vector (for FiLM) """ g = self.gating_factor ratio_g = torch.tanh(g) # concat h_a and p h_adapter = torch.cat((h_a, p),dim=1)
```

```
h_task = h_t B, T, C = x.shape K_a = h_adapter.size(1) if h_a is not None else 0 K_t = h_task.size(1) if h_task is not None else 0 # Q q_1 = self.q_proj(x) # self tokens k_tokens = self.k_self(x) v_tokens = self.v_self(x) # adapter tokens k_adapter = self.k_adapter(h_adapter) v_adapter = self.v_adapter(h_adapter) # task tokens k_task = self.k_task(h_task) v_task = self.v_task(h_task) # reshape -> multi-head def reshape_heads(t, B, L): return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2) q_1 = reshape_heads(q_1, B, T) k_tokens, v_tokens = reshape_heads(k_tokens, B, T), reshape_heads(v_tokens, B, T) k_adapter, v_adapter = reshape_heads(k_adapter, B, K_a), reshape_heads(v_adapter, B, k_task, v_task = reshape_heads(k_task, B, K_t), reshape_heads(v_task, B, K_t) # RoPE cos_main, sin_main = self.rope(seq_len=T, device=x.device, dtype=x.dtype) q_1, k_tokens = apply_rope(q_1, k_tokens, cos_main, sin_main) cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype) _, k_adapter = apply_rope(k_adapter, k_adapter, cos_a, sin_a) cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype) _, k_task = apply_rope(k_task, k_task, cos_t, sin_t) # attention scores attn_scores = [torch.matmul(q_1, k_tokens.transpose(-2, -1))] attn_scores.append(torch.matmul(q_1, k_adapter.transpose(-2, -1))) attn_scores.append(torch.matmul(q_1, k_task.transpose(-2, -1)) * ratio_g) attn_scores = torch.cat(attn_scores, dim=-1) / math.sqrt(self.head_dim) attn_weights = torch.softmax(attn_scores, dim=-1) # combine V v_list = [v_tokens,v_adapter,v_task] v_combined = torch.cat(v_list, dim=2) output = torch.matmul(attn_weights, v_combined) output = output.transpose(1, 2).contiguous().view(B, T, C) output = self.o_proj(output) # # ----FiLM ----# gamma_beta = self.film_gen(p) # [B, 2C] # gamma, beta = gamma_beta.chunk(2, dim=-1) # [B, C], [B, C] # output = self.apply_film(output, gamma, beta) # residual + FFN x = self.ffn(output + x) return x
```

We also present the performance comparison between VLA-Adapter and VLA-Adapter-Pro on 40 subtasks on LIBERO, as shown in Table I1.

Table I1: The specific performance of VLA-Adapter and VLA-Adapter-Pro on the 40 subtasks of four LIBERO (Liu et al., 2023a) suites.

| Spatial         |     1 |     2 |     3 |     4 |     5 |     6 |     7 |     8 |     9 |    10 | Avg. ↑   |
|-----------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|----------|
| VLA-Adapter     |  98.0 | 100.0 | 100.0 |  90.0 |  96.0 | 100.0 | 100.0 | 100.0 |  98.0 |  96.0 | 97.8     |
| VLA-Adapter-Pro | 100.0 |  98.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |  98.0 | 99.6     |
| Object          |     1 |     2 |     3 |     4 |     5 |     6 |     7 |     8 |     9 |    10 | Avg. ↑   |
| VLA-Adapter     |  98.0 |  98.0 | 100.0 | 100.0 |  98.0 | 100.0 |  98.0 | 100.0 | 100.0 | 100.0 | 99.2     |
| VLA-Adapter-Pro | 100.0 | 100.0 | 100.0 | 100.0 |  98.0 | 100.0 |  98.0 | 100.0 | 100.0 | 100.0 | 99.6     |
| Goal            |     1 |     2 |     3 |     4 |     5 |     6 |     7 |     8 |     9 |    10 | Avg. ↑   |
| VLA-Adapter     |  92.0 | 100.0 |  98.0 |  96.0 | 100.0 |  98.0 |  94.0 | 100.0 |  98.0 |  96.0 | 97.2     |
| VLA-Adapter-Pro |  98.0 | 100.0 |  94.0 |  96.0 | 100.0 |  98.0 |  96.0 | 100.0 | 100.0 | 100.0 | 98.2     |
| Long            |     1 |     2 |     3 |     4 |     5 |     6 |     7 |     8 |     9 |    10 | Avg. ↑   |
| VLA-Adapter     |  96.0 |  96.0 | 100.0 |  98.0 | 100.0 | 100.0 |  84.0 |  96.0 |  84.0 |  96.0 | 95.0     |
| VLA-Adapter-Pro |  92.0 | 100.0 |  98.0 |  96.0 |  94.0 | 100.0 |  94.0 | 100.0 |  90.0 | 100.0 | 96.4     |