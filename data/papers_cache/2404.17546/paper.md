## Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo

Stephen Zhao 1 2 * Rob Brekelmans 2 * Alireza Makhzani 1 2 ** Roger Grosse 1 2 **

## Abstract

Numerous capability and safety techniques of Large Language Models (LLMs), including RLHF, automated red-teaming, prompt engineering, and infilling, can be cast as sampling from an unnormalized target distribution defined by a given reward or potential function over the full sequence. In this work, we leverage the rich toolkit of Sequential Monte Carlo (SMC) for these probabilistic inference problems. In particular, we use learned twist functions to estimate the expected future value of the potential at each timestep, which enables us to focus inference-time computation on promising partial sequences. We propose a novel contrastive method for learning the twist functions, and establish connections with the rich literature of soft reinforcement learning. As a complementary application of our twisted SMC framework, we present methods for evaluating the accuracy of language model inference techniques using novel bidirectional SMC bounds on the log partition function. These bounds can be used to estimate the KL divergence between the inference and target distributions in both directions. We apply our inference evaluation techniques to show that twisted SMC is effective for sampling undesirable outputs from a pretrained model (a useful component of harmlessness training and automated red-teaming), generating reviews with varied sentiment, and performing infilling tasks.

## 1. Introduction

A wide range of language model learning and inference tasks can be viewed as steering a model's generations to satisfy a specified property. In particular, traditional reinforcement learning from human feedback (RLHF) pipelines (Ziegler et al., 2019; Stiennon et al., 2020; Ouyang et al., 2022; Bai et al., 2022; Rafailov et al., 2023) may be viewed as targeting an unnormalized target modulated by a terminal reward function which reflects human feedback (Korbak et al., 2022b). Red-teaming techniques such as promptengineering and infilling may seek target outputs with low reward or (high probability of) undesirable responses (Zou et al., 2023; Perez et al., 2022). In reasoning tasks, we may seek to target outputs which are likely to be deemed valid by a 'verifier' (Cobbe et al., 2021; Anil et al., 2021; Dohan et al., 2022; Hu et al., 2023). Specific properties of the generated responses might also be enforced (Khalifa et al., 2020; Yang &amp; Klein, 2021; Lew et al., 2023).

* Joint first authorship, ** Joint senior authorship. 1 University of Toronto 2 Vector Institute. Contact: { stephenzhao, makhzani, rgrosse } @cs.toronto.edu , rob.brekelmans@vectorinstitute.ai .

We view the above tasks as instances of probabilistic inference : sampling from a target unnormalized density and estimating its intractable (log) normalization constant. Consider a pretrained base model p 0 ( s 1: T | s 0 ) which generates responses s 1: T of maximum length T based on a variablelength prompt s 0 . We consider defining the target distribution of interest using the base model modulated by a potential function ϕ ( s 1: T ) which evaluates full sequences,

<!-- formula-not-decoded -->

where ˜ σ ( s 1: T | s 0 ) denotes the unnormalized density. We refer to Z σ ( s 0 ) as the normalization constant or partition function, which is intractable due to the summation over s 1: T . We drop dependence on s 0 to avoid clutter, but note that each prompt induces a different partition function. In the context of the aforementioned applications, ϕ ( s 1: T ) may be derived from a human preference model (for RLHF), an indication of bad behavior (for automated red-teaming), or a verifier's prediction of correctness (for reasoning tasks). We refer to Table 5 or Korbak et al. (2022b); Dohan et al. (2022); Phan et al. (2023); Hu et al. (2023) for further examples and discussion of probabilistic inference in language models.

Twisted Sequential Monte Carlo in Language Models In this work, we leverage tools from (twisted) Sequential Monte Carlo (SMC) (Doucet et al., 2001; Del Moral et al., 2006; Briers et al., 2010; Chopin et al., 2020) to perform and evaluate inference in the language modeling setting (Sec. 3). A particular challenge in sampling from Eq. (1) is that the target distribution σ ( s 1: T ) is non-causal. In order to sample tokens sequentially, one needs to infer the marginal distribution σ ( s 1: t ) = ∑ s t +1: T σ ( s 1: T ) ∝ ∑ s t +1: T p 0 ( s t +1: T | s 1: t ) ϕ ( s 1: T ) , which involves an intractable marginalization. To address this problem, we propose to learn twist functions ψ t ( s 1: t ) which modulate the base model such that p 0 ( s 1: t ) ψ t ( s 1: t ) matches the target marginals σ ( s 1: t ) , up to normalization. The twist functions can be used to focus each step of language model generation on promising partial sequences.

Evaluating Inference in Language Modeling Sampling from the target distribution is closely intertwined with bounding the log partition function. Similarly to variational inference or traditional RLHF objectives (Korbak et al., 2022b), SMC algorithms yield lower bounds on log Z σ , where tighter bounds typically coincide with more accurate target sampling. However, upper bounds may often be obtained when an exact target sample is available (Grosse et al., 2015; 2016; Brekelmans et al., 2022). The difference between upper and lower bounds on log Z σ in fact yields an upper bound on the symmetrized KL divergence between inference samples and the target distribution (Grosse et al., 2016). For these reasons, we argue in Sec. 5 that log partition function estimates are a powerful tool for evaluating language model inference techniques.

Contributions Our probabilistic inference perspective leads to the following contributions:

- Twisted Sequential Monte Carlo for Language Modeling : We view twisted SMC as a general framework for sampling and evaluation of language models. While twisted SMC is well-known and Lew et al. (2023) consider SMC with fixed, few-step-ahead target information in the language modeling setting, we propose to learn intermediate twist functions for target distributions defined by terminal potential only.
- Contrastive Twist Learning : We develop probabilistic methods for learning intermediate twist functions, presenting a novel contrastive twist learning (CTL) method inspired by energy-based modeling and density ratio estimation in Sec. 4.1. Further, we adapt existing twisted SMC methods (Lawson et al., 2018; 2022; Lioutas et al., 2022) to the language modeling setting, and highlight connections with inference techniques inspired by (soft) reinforcement learning (RL).
- Evaluating Inference in Language Models : Finally, we demonstrate that twisted SMC provides a rich set of tools for evaluating language model fine-tuning or controlled generation techniques. We propose a novel SMC upper bound on log Z σ which is applicable when an exact target sample is available and may be of independent interest. We leverage these bounds to evaluate the quality of inference by measuring the KL divergence to the target σ ( s 1: T ) in both directions,

which can be used to diagnose mode-dropping behavior of methods such as proximal policy optimization (PPO) (Schulman et al., 2017) which optimize a modeseeking divergence.

We proceed to describe background on importance sampling and SMC in Sec. 2, before presenting our framework for twisted SMC in the language modeling setting in Sec. 3. We propose methods to learn the twist functions in Sec. 4 and methods to evaluate inference in Sec. 5. Our experimental results in Sec. 7 showcase the ability of twisted SMC to improve controlled generation and lend insights into inference quality in existing methods.

## 2. Background

Suppose we are given access to an unnormalized density ˜ σ ( s 1: T ) which can be efficiently evaluated. We focus on estimation of the partition function or normalization constant Z σ := ∑ s 1: T ˜ σ ( s 1: T ) , since unbiased estimators with low variance yield approximate sampling techniques which closely approximate the target distribution (Finke, 2015; Maddison et al., 2017). We review simple importance sampling (SIS) and SMC techniques in this section.

## 2.1. Simple Importance Sampling

Simple importance sampling (SIS) provides an unbiased estimator of Z σ by calculating importance weights for any normalized proposal distribution q ( s 1: T ) ,

<!-- formula-not-decoded -->

which is unbiased since Z σ = E q ( s 1: T ) [ w ( s 1: T )] . The importance weights also yield an an unbiased K -sample estimator of the partition function,

<!-- formula-not-decoded -->

By normalizing the weights in Eq. (2) over K samples from q ( s 1: T ) , we can obtain (biased) estimators of expectations under σ ( s 1: T ) ,

<!-- formula-not-decoded -->

or select an approximate target sample s σ 1: T from a categorical distribution with the self-normalized importance weights

<!-- formula-not-decoded -->

The quality of the approximations in Eq. (3)-(5) depends crucially on how well the proposal q ( s 1: T ) (which may be learned, Sec. 3.2) matches the target σ ( s 1: T ) . While we discuss evaluation methods in Sec. 5, note that if inference is exact (i.e., q ( s 1: T ) = σ ( s 1: T ) ), then the variance of the importance weights is zero, as w ( s 1: T ) = Z σ for all s 1: T .

Figure 2: Illustrative example of SIS and (Twisted) SMC for sampling book reviews conditioned on positive sentiment ϕ ( s 1: T ) . SIS only performs resampling after observing the entire sequence, while SMC can kill or clone partial sequences s 1: t based on incremental importance weights induced by twist functions ψ t ( s 1: t ) . Green/red indicate high/low importance weights at each incremental step of SMC, or at the final step of SIS. For SMC with the base model proposal p 0 and the optimal twists, the incremental weights ψ ∗ t /ψ ∗ t -1 (Alg. 1 or Eq. (6)) are directly correlated with sentiment.

<!-- image -->

## 2.2. Sequential Monte Carlo

SMC improves inference by decomposing it into easier subproblems involving a set of unnormalized intermediate target distributions { ˜ π t ( s 1: t ) } T t =1 . A key observation is that as long as π T ( s 1: T ) = σ ( s 1: T ) , we obtain an unbiased estimate of the partition function Z T = Z σ , regardless of the intermediate π t and proposal q .

We begin by defining the incremental importance weights

<!-- formula-not-decoded -->

where ˜ π t is the unnormalized density of π t = ˜ π t / Z t .

SMC maintains a set of K partial sequences, by first sampling from the proposal q ( s k t | s k 1: t -1 ) in each index k . Optional resampling steps may be performed to clone sequences with high incremental importance weights using

<!-- formula-not-decoded -->

similarly to Eq. (5). Since resampling is performed with replacement, sequences with high weights may be cloned multiple times. The resulting s ω k t 1: t are used as prefixes for the next step of proposal sampling in index k (see Alg. 1).

We can show that SMC yields an unbiased estimator ˆ Z SMC σ of the normalization constant Z σ , by considering the extended state space S := { s k t , ω k t } K,T k,t =1 of token and in-

## Algorithm 1 (Twisted) SMC Sampling ( q SMC )

<!-- formula-not-decoded -->

dex random variables from the sampling procedure S ∼ q SMC ( S ) in Alg. 1. Assuming resampling at every step, 1

<!-- formula-not-decoded -->

To see that ˆ Z SMC σ is unbiased, we can view Eq. (8) as performing simple importance sampling Z σ = E q SMC ( S ) [ ˜ σ SMC ( S ) q SMC ( S ) ] in the extended state space, for appropriate definitions of σ SMC ( S ) and q SMC ( S ) detailed in App. F or (Andrieu et al., 2010; Maddison et al., 2017). Intuitively, we may view the average incremental importance weights at each step as estimating the partition function ratio Z t / Z t -1 ≈ 1 K ∑ K k =1 w t ( s k 1: t ) . Eq. (8) composes intermediate partition function ratio estimators to obtain an estimate of the final Z T = Z σ = ∏ T t =1 Z t / Z t -1 , with Z 0 = 1 .

With no resampling, SMC reduces to SIS with target σ ( s 1: T ) = π T ( s 1: T ) and proposal q ( s 1: T ) . Using the finalstep SMC weights, we may estimate expectations or draw approximate samples s σ 1: T as in Eq. (4)-(5).

1 The decision to resample may be based on an adaptive condition such as Effective Sample Size (ESS) (Chopin et al., 2020). For R ≤ T , let { t r } R r =1 index times where resampling occurs and fix t 0 = 0 and t R = T . The estimator becomes ˆ Z SMC σ = ∏ R r =1 1 K ∑ K i =1 ( ∏ t r t = t r -1 +1 w t ( s i 1: t ) ) , and the finalstep weights for expectations in Eq. (4) or sampling in Eq. (5) are given by ∏ T t = t R -1 +1 w t ( s i 1: t ) .

Fig. 2 illustrates the key advantage of SMC resampling over SIS. While a suboptimal q ( s 1: T ) may produce sequences with low probability under the target σ ( s 1: T ) , SMC resampling with well-chosen intermediate targets π t clones the most promising partial sequences s 1: t at step t . Since later sampling proceeds from these prefixes, we expect to obtain final sequences which better cover the high-probability regions of the target distribution. We discuss techniques to evaluate the quality of SMC or SIS sampling in Sec. 5.

## 3. Twisted Sequential Monte Carlo for Language Modeling

A key design choice in the SMC procedure above is the intermediate targets { π t } T -1 t =1 , where we assume π T ( s 1: T ) = σ ( s 1: T ) is always the target distribution. In state-space models with observation likelihoods or environments with intermediate rewards, fi ltering SMC considers target information collected from times τ ≤ t to define π t . (Chopin et al., 2020). Previous work on SMC for language models (Lew et al., 2023) has considered per-token or few-stepahead statistics to define tractable intermediate π t . However, we are often interested in target distributions which are determined by a terminal potential ϕ ( s 1: T ) only, as in Eq. (1).

In such settings, twisted SMC methods (Briers et al., 2010; Whiteley &amp; Lee, 2014; Lawson et al., 2022) consider the full target information (until time T ) to define { π t } T -1 t =1 . In other words, our desired intermediate targets are the true marginals σ ( s 1: t ) of the target distribution. Intuitively, note that in order to exactly sample s 1: T ∼ σ ( s 1: T ) , we need to ensure partial sequences are distributed according to the intermediate marginals s 1: t ∼ σ ( s 1: t ) . In Sec. 3.1, we will represent the intermediate targets { π t } T -1 t =1 using twist functions ψ t : s 1: t → R which modulate the base model to (approximately) match the target marginals, thereby summarizing future information relevant to sampling at time t .

## 3.1. Twist Functions

We represent the intermediate target distributions { π t } T -1 t =1 for SMC sampling using the following general form.

Definition 3.1 ( Twisted (Intermediate) Targets ) . Using approximate twist functions { ψ t } T -1 t =1 and the final target ϕ , we define the twisted intermediate target distributions

̸

<!-- formula-not-decoded -->

For an arbitrary proposal q and the unnormalized targets in Eq. (9), the incremental importance weights are given by

<!-- formula-not-decoded -->

While uninformed twist functions ψ t may result in π t ( s 1: t )

which are no closer to the target marginal σ ( s 1: t ) than the base model p 0 ( s 1: t ) (for example, in early stages of learning), the crucial fact is that our final target distribution in Eq. (9) reflects the target potential ϕ ( s 1: T ) . As in Sec. 2.2, this ensures that, regardless of the intermediate twists, our resulting importance sampling estimators will be unbiased.

Finally, the optimal twists ψ ∗ t ( s 1: t ) recover the intermediate marginals π ∗ t ( s 1: t ) = σ ( s 1: t ) of the target distribution. We state the sense in which π ∗ t and ψ ∗ t are optimal in App. A.1, and prove the following proposition in App. B Prop. B.1.

Proposition 3.2 ( Optimal Twists ) . For a given target distribution σ ( s 1: T ) in Eq. (1) , the optimal twist functions ψ ∗ t ( s 1: t ) (in regions where p 0 ( s 1: t ) &gt; 0 ) correspond to

<!-- formula-not-decoded -->

Up to a constant independent of s 1: t , the optimal twists are

<!-- formula-not-decoded -->

and satisfy the recursion

<!-- formula-not-decoded -->

Since the optimal twist functions are unavailable due to the need to marginalize over future timesteps, we consider learning approximate twist functions using methods in Sec. 4.

## 3.2. Proposal Distribution

For a given set of targets { π t } T t =1 , the importance weights in Eq. (10) depend crucially on the choice of proposal.

Base Model as Proposal The most straightforward choice of proposal is the base pre-trained model, q = p 0 . While we demonstrate in Sec. 7 that SMC resampling with learned twists and the base model proposal can closely approximate the target distribution, this may require large K . We can achieve greater efficiency using better choices of proposal.

Twist-Induced Proposal For given targets { π t } T t =1 , the optimal proposal minimizes the variance of the importance weights (App. A.1). In the language model setting with a terminal potential only, we will in fact be able to sample from the optimal proposal for the one-step importance weights.

Proposition 3.3. (Twist-Induced Proposal). For a given set of intermediate twisted targets π t ( s 1: t ) in Eq. (9) , the proposal which minimizes the variance of the one-step incremental importance weights w t is given by

<!-- formula-not-decoded -->

See App. A.2 for proof. For t &lt; T , we can construct a parameterization of ψ t ( s 1: t ) such that the proposal is tractable to sample in transformer architectures, where the normalization Z π t ( s 1: t -1 ) = ∑ s t p 0 ( s t | s 1: t -1 ) ψ t ( s 1: t ) sums over the discrete vocabulary of next tokens s t ∈ V . However, for the final timestep, note that ϕ ( s 1: T ) may require calls to a different neural network such as a reward model or classifier. We thus consider an approximate ψ T ( s 1: T ) ≈ ϕ ( s 1: T ) for the proposal q T ( s T | s 1: T -1 ) ∝ p 0 ( s T | s 1: T -1 ) ψ T ( s 1: T ) in the final step. With slight abuse of notation, we let q π ( s 1: T ) denote this tractable proposal over full sequences,

<!-- formula-not-decoded -->

Using this proposal, the incremental weights become

<!-- formula-not-decoded -->

which are independent of s t for t &lt; T .

Variational Proposal As noted in Sec. 2.1, SMC with no resampling steps reduces to SIS with the full target distribution σ ( s 1: T ) . Policy gradient methods (Schulman et al., 2017; Parshakova et al., 2019; Korbak et al., 2022a; Go et al., 2023) which directly learn a tractable approximation q ( s 1: T ) to the target distribution may thus be viewed as a particularly simple instance of SMC, or inference more generally (see Korbak et al. (2022b)). We may also evaluate these inference methods using our proposed tools in Sec. 5. See Table 1 and App. E for detailed losses and discussion.

Finally, note that a separate proposal q might also be learned alongside the twisting targets { π t } T -1 t =1 . This may be useful to approximate the variance-minimizing proposal for multistep or adaptive resampling (Prop. A.5) beyond the tractable optimal one-step proposal in Prop. 3.3. We discuss training losses based on multi-step importance weights in App. C.1.

## 3.3. Conditional Target Distributions

More generally, we may consider conditional target distributions, obtained by conditioning on an observation random variable o T . This mirrors the standard setting of SMC in state-space models (Doucet et al., 2001; Briers et al., 2010; Gu et al., 2015; Maddison et al., 2017; Lawson et al., 2022), with further discussion in App. B.2.

Defining ϕ ( s 1: T , o T ) = σ ( o T | s 1: T ) as a probabilistic model of o T , our target distribution is the posterior σ ( s 1: T | o T ) ,

<!-- formula-not-decoded -->

where the partition function Z σ ( o T ) = σ ( o T ) = ∑ s 1: T p 0 ( s 1: T ) σ ( o T | s 1: T ) is the marginal of the given o T .

In this setting, Prop. 3.2 suggests that the optimal twists, which match the marginals σ ( s 1: t | o T ) , correspond to the conditional likelihood of o T given s 1: t ,

<!-- formula-not-decoded -->

since σ ( o T | s 1: t ) = ∑ s t +1: T σ ( o T , s t +1: T | s 1: t ) . We can proceed to construct intermediate target distributions and proposals as in the previous sections, where ψ t ( s 1: t , o T ) and even q t ( s t | s 1: t -1 , o T ) may be conditioned on a particular value of o T .

To recover the unconditional setting, we can fix a binary observational variable σ ( o T = 1 | s 1: T ) := ϕ ( s 1: T ) (Levine, 2018) and omit explicit conditioning, showing that conditional twist learning generalizes our previous exposition. 2

Exact Target Sampling on Simulated Data Assuming σ ( o T | s 1: T ) is tractable to sample, we may obtain an exact sample from the target posterior for simulated o T using ancestral sampling. In particular, by sampling s 1: T , o T ∼ p 0 ( s 1: T ) σ ( o T | s 1: T ) , we obtain a sample from the joint distribution, which also factorizes as σ ( o T , s 1: T ) = σ ( o T ) σ ( s 1: T | o T ) . Using the latter factorization, we may interpret s 1: T as an exact sample from the target posterior for the given o T .

We refer to this as the Bidirectional Monte Carlo (BDMC) trick (Grosse et al., 2015; 2016), and will use it to draw exact samples for training in Sec. 4.1.2 or evaluation in Sec. 5.

## 3.4. Connections with Reinforcement Learning

Twisted SMC shares close connections with (soft) reinforcement learning (Levine, 2018; Pich´ e et al., 2018; Lawson et al., 2018; Heng et al., 2020), which we develop with detailed discussion in App. B.3 and App. D. In particular, we consider language modeling as a Markov Decision Process (MDP) with states x t := s 1: t -1 , actions a t := s t , and deterministic transitions p ( x t +1 | x t , a t ) = δ ( s 1: t = concat ( s t , s 1: t -1 )) . We describe two different definitions of the reward function in relation to the potential function ϕ ( s 1: T ) below. In App. B.1, we further extend our SMC framework to capture settings with intermediate potentials ϕ t ( s 1: t ) or rewards over partial sequences.

2 To obtain a probabilistic interpretation for σ ( o T = 1 | s 1: T ) = ϕ ( s 1: T ) , note we need to ensure ϕ ( s 1: T ) ∈ [0 , 1] . As a result, sampling from the target σ ( s 1: T | o T = 1) or joint σ ( s 1: T , o T = 1) is no easier in this interpretation than in Eq. (1), which is intractable in general. For example, finding ϕ max = max s 1: T ϕ ( s 1: T ) and dividing ϕ ( s 1: T ) ← ϕ ( s 1: T ) /ϕ max to rescale σ ( o T = 1 | s 1: T ) is equivalent to being able to perform rejection sampling with the base model proposal p 0 ( s 1: T ) (see Sec. 4.1.2).

Base Model Policy Evaluation Viewing the final potential ϕ ( s 1: T ) as the reward function, the optimality condition ψ ∗ t ( s 1: t ) = ∑ s t +1: T p 0 ( s t +1: T | s 1: t ) ϕ ( s 1: T ) in Eq. (12) corresponds to exact policy evaluation of the future reward under the fi xed base model policy p 0 ( s t +1: T | s 1: t ) . Mudgal et al. (2023) adopt this perspective for controlled decoding, and refer to the twist functions as 'prefix scorers'.

Soft RL with KL Regularization Alternatively, we may consider the soft or KL-regularized RL target distributions commonly used in language modeling (Levine, 2018; Korbak et al., 2022b) as a special case of our twisted SMC framework. For a regularization strength β , define the terminal potential as

<!-- formula-not-decoded -->

In this case, the intermediate twist functions in Def. 3.1 correspond to state-action Q -values, ψ t ( s 1: t ) = e βQ ( s t , s 1: t -1 ) (App. B.3). In particular, consider the recursion for the optimal twists in Eq. (13). Taking the logarithm of both sides and recalling the definition of the soft value function V ∗ ( s 1: t ) (Levine, 2018), we obtain

<!-- formula-not-decoded -->

which is a soft Bellman recursion with no intermediate reward. From the soft RL perspective, the twist functions are analogous to a critic, while the proposal plays the role of an actor (Levine, 2018; Haarnoja et al., 2018). We provide detailed discussion of the soft RL case in App. B.3, and review RL-inspired losses for twist learning in App. C.1.

Benefits of the Probabilistic Perspective While soft RL is a natural special case of our framework which gives intuition for the role of the twist functions, our approach allows for general target distributions without reference to RL objectives and suggests principled probabilistic resampling using SMC. Further, we develop twist learning techniques inspired by density ratio estimation, including our novel CTL method or the SIXO objective from (Lawson et al., 2022), which are more naturally motivated from a probabilistic perspective. Finally, we leverage our probabilistic perspective to propose novel language model evaluation techniques inspired by Bidirectional Monte Carlo (Grosse et al. (2015; 2016)) in Sec. 5.

## 4. Learning the Twist Functions

Wenext consider methods to learn twist functions ψ θ t parameterized by neural networks, presenting a novel contrastive twist learning (CTL) approach in Sec. 4.1. We summarize twist learning methods from related work in Sec. 4.2.

## 4.1. Contrastive Twist Learning

To match our approximate π θ t to the target marginals, we propose to minimize T separate KL divergences,

<!-- formula-not-decoded -->

While other divergences could be used to learn π θ t ( s 1: t ) , we argue that the mass-covering behavior of Eq. (21) is a desirable property for twist learning. Since we separately match each σ ( s 1: t ) , our hope is that suboptimal learning in early timesteps does not lead to aggressive pruning of partial sequences that would achieve high final target likelihood.

Using Eq. (9), the gradient of Eq. (21) at each t becomes

<!-- formula-not-decoded -->

which allows us to learn from exact target samples of σ ( s 1: t ) in the first term when they are available.

We note the similarity of the objective in Eq. (21) and gradient in Eq. (22) to maximum likelihood training of energybased models (EBM)s. Due to the form of the gradient update, we refer to this method as contrastive twist learning (CTL). We proceed to describe approximate techniques for positive sampling (first term) and negative sampling (second term) in the next subsections.

## 4.1.1. APPROXIMATE NEGATIVE SAMPLING

A common challenge in energy-based modeling is that the second term in Eq. (22) involves sampling from the target π t with intractable normalization constant Z ψ t . We proceed to estimate the expectation using SIS as in Eq. (4), using a proposal q ( s 1: t ) such as the base model or the twist-induced proposal from Sec. 3.2. Note that SMC resampling with learned intermediate twist functions could also be used.

## 4.1.2. (APPROXIMATE) POSITIVE SAMPLING

In contrast to traditional EBM settings, we do not necessarily have exact samples available from a 'data' distribution. Wedescribe several settings related to availability of positive samples, which are explored in our experiments in Sec. 7.

Exact Target Samples If exact posterior samples are available, for example using the BDMC trick in Sec. 3.3, we may use them directly in the gradient update in Eq. (22).

Rejection Sampling Rejection sampling can yield exact target samples s σ 1: T when an upper bound on the likelihood ratio ˜ σ ( s 1: T ) q ( s 1: T ) ≤ M is known. In cases where the target ˜ σ ( s 1: T ) is defined by thresholding or an indicator function p 0 ( s 1: T ) I ( s 1: t ∈ C ) or joint distribution p 0 ( s 1: T ) σ ( o T | s 1: T ) , we can clearly take M = 1 for the base model proposal p 0 ( s 1: T ) . If the base model yields posterior samples in reasonable time, we can obtain exact samples for training using rejection sampling, and use our twist learning procedures to greatly improve sampling efficiency at generation time.

Table 1: Losses for twist (top) and proposal (bottom) learning, where π s ( · ) indicates an arbitrary sampling distribution.

| Name   | Loss                                                                                                                                                                                                                                                      | Learning Principle                  |
|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|
| CTL    | ∑ T t =1 E π s ( o T ) [ D KL ( σ ( s 1: t &#124; o T ) ∥ ∥ π θ t ( s 1: t &#124; o T ) ) ] (Gradient:) E π s ( o T ) [ E σ ( s 1: t &#124; o T ) [ ∇ θ log ψ θ t ( s 1: t , o T ) ] - E π θ t ( s 1: t &#124; o T ) [ ∇ θ log ψ θ t ( s 1: t , o T ) ] ] | Marginal Matching with MLE          |
| RL     | ∑ T - 1 t =1 E π s ( s 1: t ,o T ) [( log ∑ s t +1 p 0 ( s t +1 &#124; s 1: t ) sg ( ψ θ t +1 ( s 1: t +1 , o T ) ) - log ψ θ t ( s 1: t , o T ) ) 2 ] + E π s ( s 1: T ,o T ) [ ( log ϕ ( s 1: T , o T ) - log ψ θ T ( s 1: T , o T ) ) 2 ]              | Twist Consistency / Soft Q-Learning |
| SIXO   | ∑ T t =1 E π s ( o T ) σ ( s 1: t &#124; o T ) [ log sigmoid ( log ψ θ t ( s 1: t , o T ) )] + E p 0 ( s 1: t ) π s ( o T ) [ log ( 1 - sigmoid ( log ψ θ t ( s 1: t , o T ) ))]                                                                          | Noise Contrastive Estimation        |
| FUDGE  | T ∑ t =1 - E π s ( s 1: t ,o T ) E p 0 ( s t +1: T &#124; s 1: t ) [ σ ( o T &#124; s 1: T ) log ψ θ t ( s 1: t , o T )+ ( 1 - σ ( o T &#124; s 1: T ) ) log ( 1 - ψ θ t ( s 1: t , o T ) ))]                                                             | Binary Classification               |
| DPG    | E π s ( o T ) [ D KL ( σ ( s 1: T &#124; o T ) ∥ ∥ q ξ ( s 1: T &#124; o T ) ) ]                                                                                                                                                                          | Maximum Likelihood (MLE)            |
| PPO    | E π s ( o T ) [ D KL ( q ξ ( s 1: T &#124; o T ) ∥ ∥ σ ( s 1: T &#124; o T ) ) ]                                                                                                                                                                          | Variational Inference               |

While an improved proposal q should more efficiently draw samples meeting the target conditions, exact rejection sampling would require estimation of M . Approximate or quasi rejection sampling might be used in this case, as analysed in Eikema et al. (2022).

Approximate Positive Sampling using SIS or SMC In cases where exact samples are unavailable and rejection sampling is inefficient or inexact, we leverage SMC sampling with twist targets { π θ t } T t =1 and any proposal q ( s 1: T ) to first draw a set of K full sequences s 1: K 1: T . As in Eq. (4), we can use the normalized SMC weights since the last resampling step to estimate the expected gradient in the first term of Eq. (22). Without resampling, we recover SIS estimation.

While both our approximate positive and negative sampling for estimating the expectations in Eq. (22) rely on SMC or SIS weights (often with the same proposal), the crucial distinction is that weights for positive sampling are based on the true target potential ϕ ( s 1: T ) over full sequences.

Truncation to Partial Sequences For an exact positive sample, we use its truncation to a partial sequence of length t (which corresponds to a sample from the desired marginal σ t ) to perform the gradient update in Eq. (22). For approximate positive sampling, we use the same set of K fi nal weights to estimate the expected gradient at each timestep.

## 4.2. Twist Learning Methods from Related Work

We briefly describe alternative approaches for twist learning, with detailed discussion in App. C and a summary of the loss functions for methods used in our experiments in Table 1.

Soft Q-Learning (RL) Enforcing the recursion in Eq. (13) using a squared error loss is analogous to soft Q -learning in the RL literature (see Eq. (20)), and has been used for twisted SMC in Lioutas et al. (2022). Mudgal et al. (2023) derive a similar squared-error loss, viewing ϕ ( s 1: T ) as the reward. Finally, we interpret path consistency losses (Nachum et al., 2017), which were derived in the soft RL setting and have been used for language modeling in Guo et al. (2021); Hu et al. (2023), from an importance sampling perspective in App. C.1 and E.1.

SIXO The SIXO loss proposed by Lawson et al. (2022) learns twist functions using a binary classification task to distinguish samples from the target marginal σ ( s 1: t | o T ) and base model p 0 ( s 1: t ) at each step, which corresponds to noise contrastive estimation (Gutmann &amp; Hyv¨ arinen, 2010) for learning energy-based models. See App. C.3.

FUDGE Yang &amp; Klein (2021) learn twists by constructing a binary classification task to instead learn the conditional likelihood σ ( o T | s 1: t ) (Eq. (18)). This may be viewed as enforcing the T -t step optimality equation in Eq. (12) or Eq. (18), where rollouts should be obtained using the base model p 0 ( s t +1: T | s 1: t ) (see Table 1 or App. C.4). Mudgal et al. (2023); Deng &amp; Raffel (2023) similarly propose to enforce the T -t step optimality condition using a squarederror loss, ∑ t E p 0 ( s t +1: T | s 1: t ) [( ϕ ( s 1: T ) -ψ t ( s 1: t )) 2 ] .

## 5. Evaluating Inference in Language Models

Our SMC framework yields a rich set of tools for evaluating inference techniques in language models, using well-studied quantities such as the log partition function log Z σ and KL divergence to the target distribution. Remarkably, with access to a single exact sample from the target distribution, we show in Prop. 5.1 that we can obtain upper bounds on log Z σ in addition to lower bounds. These bounds can tightly sandwich log Z σ with increasing K , thereby ensuring reliable conclusions regarding inference quality.

## 5.1. Applications of log Z σ Estimation

Evaluating Fine-Tuned Models To motivate this section and present an important application of our SMC methods, consider evaluating how well a given q ( s 1: T ) matches a target distribution for controlled generation or fine-tuning. Assume that q is tractable to sample and evaluate. To calculate the KL divergence to σ in either direction, we also require an estimate of the log partition function log Z σ ,

<!-- formula-not-decoded -->

For D KL ( σ ∥ q ) , note that we also require samples from the target σ , as may be readily available using the BDMC trick when σ is defined as a Bayesian posterior (Sec. 3.3). In such cases, we argue that SMC can be used to accurately bound the value of log Z σ and estimate each KL divergence above. Estimation of D KL ( σ ∥ q ) may be particularly important to diagnose mode-dropping in inference techniques such as PPO which optimize the mode-seeking D KL ( q ∥ σ ) during fine-tuning (Korbak et al., 2022b).

Evaluating Twisted SMC Sampling After running SIS or SMC with K samples, we can sample a single index as in Eq. (5) to return a single approximate target sample s σ 1: T . However, the marginal distribution of this sample, which we denote as s σ 1: T ∼ q SMC ( s 1: T ) , is not tractable due to the need to sum over all possible sets of K samples. Nevertheless, we will show below that the tightness of our log Z σ lower or upper bounds in Prop. 5.1 provides upper bounds on the KL divergences D KL ( q SMC ( s 1: T ) ∥ σ ( s 1: T )) or D KL ( σ ( s 1: T ) ∥ q SMC ( s 1: T )) , respectively.

Alternatively, we can also use the single-sample KL divergences in Eq. (23) for the twist-induced proposal q π in Eq. (15) to evaluate a set of twist functions ψ t (Sec. 7.2).

## 5.2. Bidirectional SMC Bounds on log Z σ

Given the importance of log Z σ estimation as motivated above, we propose a bidirectional SMC stochastic upper bound which is novel (to the best of our knowledge), and may be of interest outside of the language modeling setting.

Recall from Sec. 2.2 that SMC admits an interpretation as SIS in an extended state space S := { s k t , ω k t } K,T k =1 ,t =1 which includes all tokens and resampling indices. We derive lower and upper bounds on log Z σ in Prop. 5.1 below, with proof and detailed description of the extended state space target σ SMC ( S ) and proposal q SMC ( S ) distributions in App. F.

Proposition 5.1. (Bidirectional SMC Bounds) The log partition function log Z σ of a target distribution σ ( s 1: T ) can be lower and upper bounded by

<!-- formula-not-decoded -->

The gap in the lower bound is D KL ( q SMC ( S ) ∥ σ SMC ( S )) , and the gap in the upper bound is D KL ( σ SMC ( S ) ∥ q SMC ( S )) .

See App. F for a detailed discussion and derivations. The proof proceeds by adapting a general approach for extended state space log partition function bounds from Brekelmans et al. (2022) using the probabilistic interpretation of SMC from Andrieu et al. (2010); Maddison et al. (2017). With no resampling, the SIS case recovers the Importance Weighted Autoencoder (IWAE) lower (Burda et al., 2015) and upper (Sobolev &amp; Vetrov, 2019; Brekelmans et al., 2022) bounds.

Sampling from σ SMC for SMC Upper Bounds We now discuss sampling from σ SMC ( S ) for the expectation in the upper bound, which requires a single, exact sample from the target distribution σ ( s 1: T ) . This sample may be obtained, for example, using the BDMC trick in Sec. 3.3. Note that Sec. 2.2 and Alg. 1 describe sampling from q SMC ( S ) , which is used for the expectation in the lower bound.

Sampling from σ SMC ( S ) differs from sampling from q SMC ( S ) by its treatment of the exact target sample. In particular, the partial sequence corresponding to the exact target sample is guaranteed to be cloned once at each resampling step. In other indices, resampling proceeds as in Sec. 2.2, where the exact sample may be cloned additional times based on its incremental importance weights. Finally, we sample K -1 next tokens from the proposal, while the value of the remaining chain is fixed by the exact target sample. See App. F and Alg. 2 for detailed discussion.

Tightness of the Bidirectional Bounds Since the bounds in Prop. 5.1 become exact as K → ∞ for any proposal (Burda et al., 2015; Maddison et al., 2017), we can use SMC or IWAE with large K to sandwich the log partition function when σ samples are available.

For a given K , the gap in the extended state space log Z σ bounds in Prop. 5.1 provides further insight into the quality of twisted SMC sampling via the distribution of the marginal sample s σ 1: T (Sec. 5.1). In particular, the data processing inequality suggests that D KL ( q SMC ( s 1: T ) ∥ σ ( s 1: T )) ≤ D KL ( q SMC ( S ) ∥ σ SMC ( S )) and D KL ( σ ( s 1: T ) ∥ q SMC ( s 1: T )) ≤ D KL ( σ SMC ( S ) ∥ q SMC ( S )) (Grosse et al., 2015; 2016). Thus, if the difference between upper and lower bounds on log Z σ is small, then we can conclude that the K -sample SMC or SIS procedures in Sec. 2.2 yield a single approximate sample s σ 1: T whose distribution q SMC ( s 1: T ) is close to the target σ ( s 1: T ) in symmetrized KL divergence. 3

## 6. Related Work

In the previous sections, we have discussed related work as it fit within our SMC framework for language modeling. Note that Lew et al. (2023) consider SMC sampling for language models, but do not learn twist functions or proposals.

Decoding from language models to obtain diverse (Holtzman et al., 2019; Vilnis et al., 2023) or controlled generation (Zhang et al., 2023; Dathathri et al., 2019; Krause et al., 2020; Yang &amp; Klein, 2021; Guo et al., 2021; Qin et al.,

3 Note that the difference between upper and lower bound yields D KL ( σ SMC ( S ) ∥ q SMC ( S )) + D KL ( q SMC ( S ) ∥ σ SMC ( S )) .

2022; Snell et al., 2022; Hu et al., 2023) is an active area of research. Our SMC resampling approach may be viewed as a principled probabilistic extension of best-ofK decoding methods. Mudgal et al. (2023) propose a K -way arg max decoding scheme based on 'prefix scorers' ψ t learned using Eq. (13), but also consider using these twists as logits for softmax sampling in the proposal. However, neither of these decoding schemes are aligned with our proposed SMC framework, as we discuss in detail in App. D. For example, greedy arg max decoding with respect to the optimal twists in Prop. 3.2 does not yield samples from the target distribution σ ( s 1: T ) .

Finally, RL-based methods such as PPO maintain both a policy or proposal network and value network or advantage estimator during training. From the soft RL perspective in Sec. 3.4 and App. B.3, the soft values play a similar role as our twist functions for SMC resampling. Liu et al. (2023) consider using Monte Carlo Tree Search (MCTS) based on PPO value estimates to improve decoding, while Chaffin et al. (2022) consider discriminator-driven MCTS.

## 7. Experiments

We now illustrate empirically how our framework can be used to evaluate inference through log Z σ bounds and KL divergences between the sampling and target distributions, providing meaningful quantitative comparison between various learning methods. We consider a range of tasks throughout this section, including toxic story generation (as an example of uncovering rare undesirable behavior), generating reviews with varied sentiment, and infilling. For the toxicity and infilling tasks, we consider the TinyStories model (Eldan &amp; Li, 2023) 4 as a small-scale model where the generation is coherent, and use the prompt of 'Once upon a time, there was a'. For the toxicity task, we elicit responses judged to be toxic by the classifier from Corrˆ ea (2023) 5 . For the sentiment task, we consider the GPT2-Medium 6 model and a classifier trained on Amazon reviews. 7 Our code is available at https: //github.com/Silent-Zebra/twisted-smc-lm .

## 7.1. Comparing SIS and SMC for log Z σ Estimation

We first use our log Z σ bounds to test how twisted SMC can improve upon SIS and efficiently sample rare events. We consider the task of toxic story generation. The target is defined as σ ( s 1: T ) ∝ p 0 ( s 1: T ) I [ s 1: T ∈ C ] where C := { s 1: T | r ( s 1: T ) ≤ η } , r ( s 1: T ) is the non-toxic logit, and the threshold η = -5 corresponds to a greater than 99% chance of being toxic. Rejection sampling under p 0 yields exact samples for log Z σ UB estimation, but can require hundreds of thousands of samples. Thus, this setting also allows us to test the effectiveness of approximate positive sampling for twist training when target samples are rare.

[4 https://huggingface.co/roneneldan/TinyStories-33M](https://huggingface.co/roneneldan/TinyStories-33M)

[5 https://huggingface.co/nicholasKluge/ToxicityModel](https://huggingface.co/nicholasKluge/ToxicityModel)

[6 https://huggingface.co/gpt2-medium](https://huggingface.co/gpt2-medium)

[7 https://huggingface.co/LiYuan/amazon-review-sentiment-analysis](https://huggingface.co/LiYuan/amazon-review-sentiment-analysis)

Figure 3: Comparison of SIS (IWAE) and SMC bounds on log Z σ for base proposal p 0 and twist-induced proposal q π , with twists learned with CTL. With the twist-induced proposal, both SIS and SMC bounds are tight; with the base proposal, resampling with learned twists is needed. Resampling based on ESS instead of every-step resampling yields similar results.

<!-- image -->

Fig. 3 demonstrates that training twists with CTL and approximate positive sampling can significantly improve log partition function estimation and sampling efficiency. We first note that both upper and lower bounds tighten as K increases, as expected, for both SIS and SMC. Using p 0 as proposal, the SIS LB (orange) generally fails to draw any samples meeting the threshold. By contrast, SMC resampling (red) with p 0 proposal eventually achieves tight log Z σ upper and lower bounds, yielding near-exact target samples (small KL divergence between the distribution over samples and the target distribution) by the reasoning in Sec. 5.

However, both SMC and SIS with the twist-induced proposal achieve tight estimation and near-exact sampling of the target toxic outputs with orders of magnitude lower K . Resampling does not appear to help or hurt these bounds, as the effect of the twists has been incorporated in the proposal q π in Eq. (15). Thus, we conclude that using the twistinduced proposal can provide significant efficiency gains over base model sampling.

## 7.2. Evaluating Twist-Induced or Variational Proposals

We next leverage our log Z σ bounds to evaluate singlesample inference using D KL ( q ∥ σ ) and D KL ( σ ∥ q ) , as in Sec. 5.1. Across settings, we consider two SIS proposallearning methods: PPO (Schulman et al., 2017) which minimizes D KL ( q ∥ σ ) during optimization, and distributional policy gradient (DPG), which minimizes D KL ( σ ∥ q ) (Parshakova et al., 2019) (see App. E).

Table 2: Toxicity (Sec. 7.2.1)

| Proposal q   | Twist Learning   | D KL ( q ∥ σ )   | D KL ( σ ∥ q )   | Proposal q   | Twist Learning   | D KL ( q ∥ σ )   | D KL ( σ ∥ q )   | Proposal q o T   | Twist Learning   | E o T [ D KL ( q o T ∥ σ o T )]   | E o T [ D KL ( σ o T ∥ q o T )]   |
|--------------|------------------|------------------|------------------|--------------|------------------|------------------|------------------|------------------|------------------|-----------------------------------|-----------------------------------|
| Twisted      | Contrastive      | 1 . 11 ± 0 . 05  | 1 . 07 ± 0 . 02  | Twisted      | Contrastive      | 0 . 55 ± 0 . 03  | 0 . 47 ± 0 . 01  | Twisted          | Contrastive      | 23 . 93 ± 0 . 34                  | 8 . 87 ± 0 . 05                   |
| Twisted      | RL               | 1 . 52 ± 0 . 09  | 1 . 42 ± 0 . 03  | Twisted      | RL               | 0 . 94 ± 0 . 04  | 0 . 81 ± 0 . 02  | Twisted          | RL               | 31 . 35 ± 2 . 33                  | 14 . 96 ± 1 . 69                  |
| Twisted      | SIXO             | 1 . 71 ± 0 . 06  | 1 . 98 ± 0 . 04  | Twisted      | SIXO             | 0 . 73 ± 0 . 03  | 0 . 59 ± 0 . 02  | Twisted          | SIXO             | 20 . 34 ± 0 . 36                  | 7 . 43 ± 0 . 04                   |
| Twisted      | FUDGE            | 3 . 24 ± 0 . 26  | 2 . 00 ± 0 . 13  | Twisted      | FUDGE            | 1 . 01 ± 0 . 07  | 0 . 77 ± 0 . 07  | Twisted          | FUDGE            | 60 . 93 ± 2 . 82                  | 19 . 85 ± 0 . 51                  |
| DPG          | -                | 1 . 09 ± 0 . 05  | 1 . 12 ± 0 . 03  | DPG          | -                | 0 . 72 ± 0 . 04  | 0 . 57 ± 0 . 01  | DPG              | -                | 13 . 27 ± 0 . 44                  | 4 . 90 ± 0 . 03                   |
| PPO          | -                | 0 . 98 ± 0 . 01  | 1 . 32 ± 0 . 04  | PPO          | -                | 1 . 04 ± 0 . 31  | 0 . 87 ± 0 . 20  | PPO              | -                | 19 . 37 ± 0 . 41                  | 14 . 07 ± 0 . 50                  |

Table 3: Sentiment (Sec. 7.2.2)

Table 4: Infilling (Sec. 7.2.3)

Forward and reverse KL divergences between the SMC or variational proposal distributions and the true target σ .

We consider four twist learning methods, including CTL and RL from Sec. 4, SIXO (Lawson et al., 2022), and FUDGE (Yang &amp; Klein, 2021) (see App. C). For each, we measure KL divergences involving the twist-induced proposal q π . Thus, these experiments showcase two complementary applications of SMC : as a novel inference method yielding a tractable q π , and as an evaluation method for any other inference method (such as PPO) using K -sample bounds on log Z σ to estimate the KL divergence.

## 7.2.1. GENERATING TOXIC STORIES

We consider toxic story generation as in Sec. 7.1, but using a target σ ( s 1: T ) ∝ p 0 ( s 1: T ) p ( a = 1 | s 1: T ) , where p ( a = 1 | s 1: T ) denotes the probability of the text being judged as toxic by a classifier. Compared to the thresholding target, this task provides a smoother gradient signal for learning (see App. G.3) but still allows for exact sampling via rejection sampling. We train using approximate positive sampling, but provide an ablation with exact positive sampling results in App. H.3.

We report KL divergences in Table 2. We observe that PPO learns the best proposal with respect to D KL ( q ∥ σ ) while our CTL method performs best with respect to D KL ( σ ∥ q ) , which is consistent with the divergences minimized during training. Finally, in App. H.1 we provide a qualitative example of a toxic story generated with CTL for σ ( s 1: T ) ∝ p 0 ( s 1: T ) p ( a = 1 | s 1: T ) β with β = 10 , a case where no exact samples are available.

## 7.2.2. GENERATION WITH VARIED SENTIMENT

For the sentiment setting described earlier, we consider a prompt 'I bought this' and target σ ( s 1: T ) ∝ p 0 ( s 1: T ) p ( a = 1 | s 1: T ) , where a = 1 indicates a 1-star review and exact samples are available by rejection sampling. We train using approximate positive sampling (see App. H.3 for comparison with exact). While all methods achieve low KL divergences in Table 3, CTL performs best for both directions.

## 7.2.3. INFILLING

In this section, we demonstrate a conditional twist function parameterization, where ψ θ t ( s 1: t , o T ) takes input o T which identifies the target distribution σ ( s 1: T | o T ) as in Sec. 3.3. We consider an infilling task (Lew et al., 2023; Hu et al., 2023), where the observation variables o T := s T +1: T + c correspond to continuation tokens, and their likelihood σ ( o T | s 1: T ) := p 0 ( s T +1: T + c | s 1: T ) is evaluated under the base model, given generated s 1: T . The target distribution corresponds to the posterior σ ( s 1: T | o T ) . Instead of training separate { ψ θ t } for each o T , we amortize learning of a conditional twist network ψ θ t ( s 1: t , o T ) .

A second distinctive feature of this setting is that we train from exact posterior or target samples, which are readily available using the BDMC trick in Sec. 3.3. In particular, we may sample sequences of length T + c from the base model s 1: T + c ∼ p 0 ( s 1: T + c ) = σ ( s 1: T , o T = s T +1: T + c ) , and interpret the prefix s 1: T ∼ σ ( s 1: T | o T = s T +1: T + c ) as a target sample. Note that we do not explicitly control the continuations tokens o T defining the tasks. We evaluate average KL divergences over 2000 different o T = s T +1: T + c , with T = 15 and c = 10 , and report results in Table 4.

We find that DPG performs best for both directions of the KLdivergence in this setting, likely due to its ability to leverage exact positive samples by minimizing D KL ( σ o T ∥ q o T ) . While CTL also learns from exact positive samples, it requires approximate negative sampling and only performs comparably to SIXO, which uses exact positive samples and performs exact negative sampling under p 0 . Finally, PPO trains from q o T samples only, and performs relatively poorly with respect to D KL ( σ o T ∥ q o T ) . We show qualitative results in App. H.1 to correlate KL divergence results with sample quality.

Using our KL divergence evaluation methods, we conclude DPG may be preferable when exact target samples are available (Sec. 7.2.3, App. H.3), while CTL may be preferable with approximate positive sampling (Sec. 7.2.1, Sec. 7.2.2).

## 8. Conclusion

In this work, we have presented twisted SMC as a principled probabilistic inference framework for solving numerous capability and safety tasks in LLMs. After discussing different design choices for twisted SMC and their relation to related work, we proposed a novel contrastive method for twist learning. Furthermore, we have proposed novel bidirectional SMC bounds for evaluating LLM inference methods. We demonstrated the effectiveness of our methods quantitatively and qualitatively in both sampling and evaluation across a variety of experimental settings.

## Acknowledgments

AMand RG acknowledge support from the Canada CIFAR AI Chairs program and from Open Philanthropy. SZ thanks Juhan Bae for helping debug memory issues in the code. Resources used in this research were provided, in part, by the Province of Ontario, the Government of Canada, and companies sponsoring the Vector Institute. We thank the anonymous reviewers for helpful comments on earlier versions of this paper.

## Impact Statement

This paper is motivated by the social consequences of recent advances in the field of machine learning. Controlled generation from language models has the potential to improve safety through better steering of generation to human preferences, more efficient automated red-teaming, and the ability to estimate or bound probabilities of rare behaviors. Any such work is inherently a double-edged sword; the same techniques used to generate samples from a harmless distribution of text could, with a single sign change, be repurposed for generating samples from a harmful distribution of text. Thus, better controlled generation (in our framework, better sampling from target distributions) can provide benefits in the hands of responsible users but can also magnify harms in the hands of malevolent users (who have access to model weights).

Overall, we believe the potential positive social benefits of our work in evaluation and steering language model output towards desired target distributions outweigh the potential negatives stemming primarily from misuse.

## References

- Andrieu, C., Doucet, A., and Holenstein, R. Particle markov chain monte carlo methods. Journal of the Royal Statistical Society Series B: Statistical Methodology , 72(3): 269-342, 2010.
- Anil, C., Zhang, G., Wu, Y., and Grosse, R. Learning to give checkable answers with prover-verifier games. arXiv preprint arXiv:2108.12099 , 2021.
- Bae, J., Zhang, M. R., Ruan, M., Wang, E., Hasegawa, S., Ba, J., and Grosse, R. B. Multi-rate vae: Train once, get the full rate-distortion curve. In The Eleventh International Conference on Learning Representations , 2022.
- Bai, Y., Jones, A., Ndousse, K., Askell, A., Chen, A., DasSarma, N., Drain, D., Fort, S., Ganguli, D., Henighan, T., et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862 , 2022.
- Banerjee, A., Guo, X., and Wang, H. On the optimality of
- conditional expectation as a bregman predictor. IEEE Transactions on Information Theory , 51(7), 2005.
- Brekelmans, R., Huang, S., Ghassemi, M., Ver Steeg, G., Grosse, R. B., and Makhzani, A. Improving mutual information estimation with annealed and energy-based bounds. In International Conference on Learning Representations , 2022.
- Briers, M., Doucet, A., and Maskell, S. Smoothing algorithms for state-space models. Annals of the Institute of Statistical Mathematics , 62:61-89, 2010.
- Burda, Y., Grosse, R., and Salakhutdinov, R. Importance weighted autoencoders. arXiv preprint arXiv:1509.00519 , 2015.
- Chaffin, A., Claveau, V., and Kijak, E. Ppl-mcts: Constrained textual generation through discriminator-guided mcts decoding. In NAACL 2022-Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , 2022.
- Chopin, N., Papaspiliopoulos, O., et al. An introduction to sequential Monte Carlo , volume 4. Springer, 2020.
- Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168 , 2021.
- Corrˆ ea, N. K. Aira, 2023. URL https://huggingface. co/nicholasKluge/ToxicityModel .
- Dathathri, S., Madotto, A., Lan, J., Hung, J., Frank, E., Molino, P., Yosinski, J., and Liu, R. Plug and play language models: A simple approach to controlled text generation. In International Conference on Learning Representations , 2019.
- Del Moral, P., Doucet, A., and Jasra, A. Sequential monte carlo samplers. Journal of the Royal Statistical Society Series B: Statistical Methodology , 68(3):411-436, 2006.
- Deng, H. and Raffel, C. Reward-augmented decoding: Efficient controlled text generation with a unidirectional reward model. In The 2023 Conference on Empirical Methods in Natural Language Processing , 2023.
- Dohan, D., Xu, W., Lewkowycz, A., Austin, J., Bieber, D., Lopes, R. G., Wu, Y., Michalewski, H., Saurous, R. A., Sohl-Dickstein, J., et al. Language model cascades. arXiv preprint arXiv:2207.10342 , 2022.
- Domke, J. and Sheldon, D. R. Importance weighting and variational inference. Advances in neural information processing systems , 31, 2018.

- Doucet, A., De Freitas, N., Gordon, N. J., et al. Sequential Monte Carlo methods in practice , volume 1. Springer, 2001.
- Eikema, B., Kruszewski, G., Dance, C. R., Elsahar, H., and Dymetman, M. An approximate sampler for energybased models with divergence diagnostics. Transactions on Machine Learning Research , 2022.
- Eldan, R. and Li, Y. Tinystories: How small can language models be and still speak coherent english? arXiv preprint arXiv:2305.07759 , 2023.
- Finke, A. On extended state-space constructions for Monte Carlo methods . PhD thesis, University of Warwick, 2015.
- Go, D., Korbak, T., Kruszewski, G., Rozen, J., Ryu, N., and Dymetman, M. Aligning foundation models for language with preferences through f -divergence minimization. In International Conference on Machine Learning , 2023.
- Grosse, R. B., Ghahramani, Z., and Adams, R. P. Sandwiching the marginal likelihood using bidirectional monte carlo. arXiv preprint arXiv:1511.02543 , 2015.
- Grosse, R. B., Ancha, S., and Roy, D. Measuring the reliability of mcmc inference with bidirectional monte carlo. Advances in Neural Information Processing Systems , 2016.
- Gu, S. S., Ghahramani, Z., and Turner, R. E. Neural adaptive sequential monte carlo. Advances in neural information processing systems , 28, 2015.
- Guo, H., Tan, B., Liu, Z., Xing, E. P., and Hu, Z. Efficient (soft) q-learning for text generation with limited good data. arXiv preprint arXiv:2106.07704 , 2021.
- Gutmann, M. and Hyv¨ arinen, A. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In International conference on artificial intelligence and statistics , pp. 297-304. JMLR Workshop and Conference Proceedings, 2010.
- Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning . PMLR, 2018.
- Heng, J., Bishop, A., Deligiannidis, G., and Doucet, A. Controlled sequential monte carlo. Annals of Statistics , 48(5), 2020.
- Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. The curious case of neural text degeneration. In International Conference on Learning Representations , 2019.
- Hu, E. J., Jain, M., Elmoznino, E., Kaddar, Y., Lajoie, G., Bengio, Y., and Malkin, N. Amortizing intractable inference in large language models. arXiv preprint arXiv:2310.04363 , 2023.
- Khalifa, M., Elsahar, H., and Dymetman, M. A distributional approach to controlled text generation. arXiv preprint arXiv:2012.11635 , 2020.
- Khanov, M., Burapacheep, J., and Li, Y. ARGS: Alignment as reward-guided search. In The Twelfth International Conference on Learning Representations , 2024. URL https://openreview.net/forum?id=shgx0eqdw6 .
- Korbak, T., Elsahar, H., Kruszewski, G., and Dymetman, M. Controlling conditional language models without catastrophic forgetting. In International Conference on Machine Learning , pp. 11499-11528. PMLR, 2022a.
- Korbak, T., Perez, E., and Buckley, C. L. Rl with kl penalties is better viewed as bayesian inference. arXiv preprint arXiv:2205.11275 , 2022b.
- Krause, B., Gotmare, A. D., McCann, B., Keskar, N. S., Joty, S., Socher, R., and Rajani, N. F. Gedi: Generative discriminator guided sequence generation. arXiv preprint arXiv:2009.06367 , 2020.
- Lawson, D., Tucker, G., Naesseth, C. A., Maddison, C., Adams, R. P., and Teh, Y. W. Twisted variational sequential monte carlo. In Third workshop on Bayesian Deep Learning (NeurIPS) , 2018.
- Lawson, D., Ravent´ os, A., Warrington, A., and Linderman, S. Sixo: Smoothing inference with twisted objectives, 2022.
- Levine, S. Reinforcement learning and control as probabilistic inference: Tutorial and review. arXiv preprint arXiv:1805.00909 , 2018.
- Lew, A. K., Zhi-Xuan, T., Grand, G., and Mansinghka, V. K. Sequential monte carlo steering of large language models using probabilistic programs. arXiv preprint arXiv:2306.03081 , 2023.
- Lioutas, V., Lavington, J. W., Sefas, J., Niedoba, M., Liu, Y., Zwartsenberg, B., Dabiri, S., Wood, F., and Scibior, A. Critic sequential monte carlo. In The Eleventh International Conference on Learning Representations , 2022.
- Liu, A., Sap, M., Lu, X., Swayamdipta, S., Bhagavatula, C., Smith, N. A., and Choi, Y. Dexperts: Decodingtime controlled text generation with experts and antiexperts. In 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing , 2021.
- Liu, J., Cohen, A., Pasunuru, R., Choi, Y., Hajishirzi, H., and Celikyilmaz, A. Don't throw away your value model! making ppo even better via value-guided monte-carlo tree search decoding. arXiv e-prints , pp. arXiv-2309, 2023.

- Maddison, C. J., Lawson, J., Tucker, G., Heess, N., Norouzi, M., Mnih, A., Doucet, A., and Teh, Y. Filtering variational objectives. Advances in Neural Information Processing Systems , 30, 2017.
- Mudgal, S., Lee, J., Ganapathy, H., Li, Y., Wang, T., Huang, Y., Chen, Z., Cheng, H.-T., Collins, M., Strohman, T., et al. Controlled decoding from language models. arXiv preprint arXiv:2310.17022 , 2023.
- Nachum, O., Norouzi, M., Xu, K., and Schuurmans, D. Bridging the gap between value and policy based reinforcement learning. Advances in neural information processing systems , 30, 2017.
- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems , 35:27730-27744, 2022.
- Parshakova, T., Andreoli, J.-M., and Dymetman, M. Distributional reinforcement learning for energy-based sequential models. arXiv preprint arXiv:1912.08517 , 2019.
- Perez, E., Huang, S., Song, F., Cai, T., Ring, R., Aslanides, J., Glaese, A., McAleese, N., and Irving, G. Red teaming language models with language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing , pp. 3419-3448, 2022.
- Phan, D., Hoffman, M. D., Douglas, S., Le, T. A., Parisi, A. T., Sountsov, P., Sutton, C., Vikram, S., Saurous, R. A., et al. Training chain-of-thought via latent-variable inference. In Thirty-seventh Conference on Neural Information Processing Systems , 2023.
- Pich´ e, A., Thomas, V., Ibrahim, C., Bengio, Y., and Pal, C. Probabilistic planning with sequential monte carlo methods. In International Conference on Learning Representations , 2018.
- Qin, L., Welleck, S., Khashabi, D., and Choi, Y. Cold decoding: Energy-based constrained text generation with langevin dynamics. Advances in Neural Information Processing Systems , 35:9538-9551, 2022.
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290 , 2023.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 , 2017.
- Shih, A., Sadigh, D., and Ermon, S. Long horizon temperature scaling. arXiv preprint arXiv:2302.03686 , 2023.
- Snell, C. V., Kostrikov, I., Su, Y., Yang, S., and Levine, S. Offline rl for natural language generation with implicit language q learning. In The Eleventh International Conference on Learning Representations , 2022.
- Sobolev, A. and Vetrov, D. P. Importance weighted hierarchical variational inference. Advances in Neural Information Processing Systems , 32, 2019.
- Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. Advances in Neural Information Processing Systems , 33: 3008-3021, 2020.
- Vilnis, L., Zemlyanskiy, Y., Murray, P., Passos, A. T., and Sanghai, S. Arithmetic sampling: parallel diverse decoding for large language models. In International Conference on Machine Learning . PMLR, 2023.
- Whiteley, N. and Lee, A. Twisted particle filters. 2014.
- Yang, K. and Klein, D. Fudge: Controlled text generation with future discriminators. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , pp. 3511-3535, 2021.
- Zhang, H., Song, H., Li, S., Zhou, M., and Song, D. A survey of controllable text generation using transformerbased pre-trained language models. ACM Computing Surveys , 56(3):1-37, 2023.
- Ziegler, D. M., Stiennon, N., Wu, J., Brown, T. B., Radford, A., Amodei, D., Christiano, P., and Irving, G. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593 , 2019.
- Zou, A., Wang, Z., Kolter, J. Z., and Fredrikson, M. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043 , 2023.

## Appendix

## Table of Contents

| Proofs                                                             | Proofs                                                                                |   15 |
|--------------------------------------------------------------------|---------------------------------------------------------------------------------------|------|
| A.1                                                                | Proof for Optimal Intermediate Target Distributions . . . . . . . . . . . . . . . . . |   15 |
| A.2 . . .                                                          | Proof of Twist-Induced Proposal . . . . . . . . . . . . . . . . . . . . . . . .       |   17 |
| A.3 . . . . . .                                                    | Derivation of CTL Gradient . . . . . . . . . . . . . . . . . . . . . . . .            |   18 |
| SMC                                                                | with Intermediate Potentials and Connection with Soft Reinforcement Learning          |   18 |
| B.1                                                                | Twisted SMC with Intermediate Potentials . . . . . . . . . . . . . . . . . . . . . .  |   18 |
| B.2 . . . . . . .                                                  | Conditional Twisted SMC . . . . . . . . . . . . . . . . . . . . . . . .               |   20 |
| B.3                                                                | Connection with Soft Reinforcement Learning . . . . . . . . . . . . . . . . . . . .   |   22 |
| B.4 . . . . .                                                      | Remarks on Parameterization . . . . . . . . . . . . . . . . . . . . . . . .           |   24 |
| Twist Learning Losses                                              | Twist Learning Losses                                                                 |   24 |
| C.1                                                                | Soft Q-Learning (RL) and Path Consistency Losses from Log Importance Weights .        |   24 |
| C.2                                                                | Controlled Decoding Losses via Optimal Twist Identities (Mudgal et al., 2023) . . .   |   27 |
| C.3                                                                | SIXO: Smoothing Inference with Twisted Objectives (Lawson et al., 2022) . . . . .     |   28 |
| C.4                                                                | FUDGE: Future Discriminators (Yang &Klein, 2021) . . . . . . . . . . . . . . . .      |   29 |
| Decoding Strategies using Learned Twists from Mudgal et al. (2023) | Decoding Strategies using Learned Twists from Mudgal et al. (2023)                    |   30 |
| D.1                                                                | Proposal Sampling in Mudgal et al. (2023) . . . . . . . . . . . . . . . . . . . . . . |   30 |
| D.2                                                                | Blockwise Greedy Decoding in Mudgal et al. (2023) . . . . . . . . . . . . . . . . .   |   32 |
| Proposal Learning Methods                                          | Proposal Learning Methods                                                             |   32 |
| E.1                                                                | Path Consistency Learning for Controlled Generation . . . . . . . . . . . . . . . . . |   32 |
| E.2 . . . . . . .                                                  | Policy Gradient Methods . . . . . . . . . . . . . . . . . . . . . . . .               |   33 |
| E.3                                                                | Policy Gradient with Mass-Covering / Maximum Likelihood KL Divergence . . . .         |   33 |
| Bidirectional SMC                                                  | Bidirectional SMC                                                                     |   36 |
| Additional Experiment Details                                      | Additional Experiment Details                                                         |   40 |
| G.1 .                                                              | Common Details Across Experiments . . . . . . . . . . . . . . . . . . . . . . . .     |   40 |
| G.2 . . .                                                          | Choices of Twist Parameterization . . . . . . . . . . . . . . . . . . . . . . . .     |   41 |
| G.3                                                                | Comments on Our Choices of Experiment Settings . . . . . . . . . . . . . . . . . .    |   42 |
| G.4 . . . . . .                                                    | Experiment-Specific Details . . . . . . . . . . . . . . . . . . . . . . . .           |   42 |
| Additional Experimental Results                                    | Additional Experimental Results                                                       |   44 |
| H.1 . . . . . . . . . . .                                          | . . . . . . . . . . . . . . . . . . . . . . . .                                       |   44 |
|                                                                    | Qualitative Results                                                                   |      |
| H.2 Infilling with Fewer Tokens . . . . . .                        | . . . . . . . . . . . . . . . . . . . . . . . .                                       |   45 |
| H.3                                                                | Approximate vs. Exact Posterior Sampling . . . . . . . . . . . . . . . . . . . . . .  |   45 |

Table 5: Examples of Target Posteriors in Language Model Finetuning and Controlled Generation

| Type               | Target                                                                                                                                                         | References / Examples                                                                                                                                                                                                                                                                                                                         |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Reward             | σ ( s 1: T ) ∝ p 0 ( s 1: T ) e ± βr ( s 1: T )                                                                                                                | RLHF (Ziegler et al., 2019; Ouyang et al., 2022; Korbak et al., 2022b)                                                                                                                                                                                                                                                                        |
| Continuation       | σ ( s 1: T ) ∝ p 0 ( s 1: T ) p 0 ( s T +1: T + c &#124; s 1: T ) β                                                                                            | Generates tokens based on likelihood of future tokens p ( s T +1: T + c &#124; s 1: T ) For β = 1 , this is in-filling (Lew et al., 2023). As β →∞ , disregard p 0 ( s 1: T ) , focus on argmax of continuation prob. - similar to adversarial prompt generation (Zou et al., 2023)                                                           |
| Indicator          | σ ( s 1: T ) ∝ p 0 ( s 1: T ) I [ s 1: T ∈ C ] where I is indicator of set C : I [ s 1: T ∈ C ] = 1 if [ s 1: T ∈ C ] I [ s 1: T ∈ C ] = 0 if [ s 1: T / ∈ C ] | Generations s 1: T from this target must satisfy the properties of set C . - Meeting reward threshold C r ≤ η := { s 1: T &#124;± r ( s 1: T ) ≤ η } - Containing topical or specific words in s 1: T - Having certain structure or rhyme (Yang &Klein, 2021), - Valid output according to verifier (Cobbe et al., 2021; Dohan et al., 2022)) |
| Classifier         | σ ( s 1: T ) ∝ p 0 ( s 1: T ) p ( y &#124; s 1: T ) β                                                                                                          | Class y can be a binary (e.g. toxicity) or multinomial (e.g. 1-5 star reviews) Bayesian posterior for β = 1 : σ ( s 1: T ) = p ( s 1: T &#124; y ) ∝ p 0 ( s 1: T ) p ( y &#124; s 1: T ) (Dathathri et al., 2019; Krause et al., 2020; Liu et al., 2021)                                                                                     |
| Global Temperature | σ ( s 1: T ) ∝ p 0 ( s 1: T ) β                                                                                                                                | Tempering on entire sequences (long-horizon) vs. per-token (myopic) - yields higher quality generation in Shih et al. (2023)                                                                                                                                                                                                                  |
| Distributional     | σ ( s 1: T ) ∝ p 0 ( s 1: T ) e β · T ( s 1: T )                                                                                                               | KL minimization subj. expectation constraints on T = { T i } argmin D KL ( q ( s 1: T ) ∥ p 0 ( s 1: T )) s.t. E q [ T ] = η β ( β = optimal Lagrange multipliers for constraints η ) e.g. gender roles/references (Khalifa et al., 2020)                                                                                                     |
| Intermediate       |                                                                                                                                                                | References / Examples                                                                                                                                                                                                                                                                                                                         |
| Indicator          | ϕ t ( s 1: t ) = I [ s t ∈ C ] or I [ s 1: t ∈ C ]                                                                                                             | words of specific length, or specific sets of tokens (Khalifa et al., 2020; Lew et al., 2023)                                                                                                                                                                                                                                                 |
| Product of Experts | σ ( s 1: T ) ∝ ∏ M m =1 ∏ T t =1 p 0 ( s t &#124; s 1: t - 1 , s ( m ) 0 )                                                                                     | prompt intersection (Lew et al., 2023)                                                                                                                                                                                                                                                                                                        |

## A. Proofs

In this section, we present the sense in which the target marginals correspond to the optimal intermediate distributions in twisted SMC. We defer proof of Prop. 3.2 from the main text to slightly more general version in App. B.1 Prop. B.1, although Prop. A.4 provides the analogous statement in terms of the intermediate target distributions π ∗ t ( s 1: t ) = σ ( s 1: t ) instead of the optimal twists ψ ∗ t .

We also prove Prop. 3.3 from the main text in App. A.2 and derive the gradient of the CTL loss (Eq. (22)) in App. A.3.

## A.1. Proof for Optimal Intermediate Target Distributions

In order to achieve sampling from the full joint distribution σ ( s 1: T ) , each intermediate target σ ( s 1: t ) must match the intermediate marginal σ ( s 1: t ) . To formalize this notion, we provide the following definition of optimality, justified by the fact that it yields an exact partition function estimator.

To do so, we will consider the multi-step importance weights

<!-- formula-not-decoded -->

( c -Step SMC Weights)

using a telescoping cancellation in the final equality. The one-step weights correspond to c = 1 , denoted simply as w t .

Definition A.1 ( Optimal Twisted SMC Sampling ) . For a given target distribution σ ( s 1: T ) ∝ p 0 ( s 1: T ) ϕ ( s 1: T ) , we refer to a twisted SMC procedure, SMC ( { π t } T t =1 , q, K ) or SMC ( p 0 , { ψ t } T t =1 , q, K ) (with π T = σ or ψ T = ϕ ), as optimal if c -step importance weights w t : t + c -1 ( s 1: t + c -1 ) = Z ψ t + c -1 / Z ψ t -1 for all 1 ≤ t ≤ T and 0 ≤ c ≤ T -t +1 .

Note, that the role of ψ t and Z ψ t is specified in Def. 3.1. We assume π T = σ for the goal of estimating Z σ , and show below that an optimal twisted SMC procedure yields an exact partition function estimator.

<!-- formula-not-decoded -->

Proposition A.2 ( Optimal SMC yields Exact Partition Function Estimation ) . For any optimal twisted SMC procedure, the resulting estimator of the partition function Z σ has zero bias and zero variance.

Proof. As in Footnote 1 or App. F Alg. 2, consider { t r } R r =1 index timesteps where resampling occurs and fix t 0 = 0 and t R = T . The SMC estimator of Z σ = Z ψ T becomes ˆ Z SMC σ = ∏ R r =1 1 K ∑ K i =1 ( ∏ t r t = t r -1 +1 w t ( s i 1: t ) ) for S ∼ q SMC ( S ) . Using the optimality definition in Def. A.1, we have w t ( s 1: t ) = Z ψ t / Z ψ t -1 for all partial sequences s 1: t . Noting the telescoping multiplicative cancellation and the fact that w t ( s i 1: t ) = Z ψ t / Z ψ t -1 is constant with respect to indices i ∈ [1 , K ] , we have the following estimator for a single run of an optimal SMC procedure,

<!-- formula-not-decoded -->

as desired, assuming Z ψ 0 = 1 . Since ˆ Z SMC σ = Z σ is independent of S , we conclude ˆ Z SMC σ has zero bias and zero variance.

Note that we could also define optimality in Def. A.1 using the condition that w t : t + c -1 ( s 1: t + c -1 ) = const for all 1 ≤ t ≤ T and 0 ≤ c ≤ T -t +1 . Following similar derivations as above would yield ˆ Z SMC σ = const. As we will show in App. F, ˆ Z SMC σ is unbiased with E [ ˆ Z SMC σ ] = Z σ . We thus conclude that ˆ Z SMC σ = Z σ with zero variance, and thus Prop. A.2 holds.

With this notion of optimality in mind, we demonstrate the following necessary and sufficient conditions.

Proposition A.3 ( Optimality Conditions ) . The following conditions are necessary and sufficient for twisted SMC optimality,

<!-- formula-not-decoded -->

Proof. (Necessary) Optimal Twisted SMC = ⇒ ( i ) , ( ii ) : We begin by writing the marginalization of the unnormalized density ˜ π ∗ t + c over prefixes of length t as

<!-- formula-not-decoded -->

The normalization constant of ˜ π ∗ t + c ( s 1: t ) can easily be seen to be Z ψ ∗ t + c after summing over s 1: t above, which yields π ∗ t + c ( s 1: t ) = ˜ π ∗ t + c ( s 1: t ) / Z ψ ∗ t + c . We now factorize the c -step incremental importance weights (at step t + 1 , see Eq. ( c -Step SMC Weights)) using the above identities, which imply that ˜ π ∗ t + c ( s 1: t + c ) = Z ψ ∗ t + c π ∗ t + c ( s 1: t + c ) = Z ψ ∗ t + c π ∗ t + c ( s 1: t ) π ∗ t + c ( s t +1: t + c | s 1: t ) and

<!-- formula-not-decoded -->

In order to have w t +1: t + c ( s 1: t + c ) = Z ψ ∗ t + c / Z ψ ∗ t in general, we thus require π ∗ t + c ( s 1: t ) = π ∗ t ( s 1: t ) and π ∗ t + c ( s t +1: t + c | s 1: t ) = q ∗ ( s t +1: t + c | s 1: t ) for all t and c ≤ T -t .

(Sufficient) ( i ) , ( ii ) = ⇒ Optimal Twisted SMC: Consider the incremental importance weights using ( i ) and ( ii ) ,

<!-- formula-not-decoded -->

which matches the optimality definition in Def. A.1.

Proposition A.4 ( Optimal Intermediate Target Distributions ) . For a given target distribution σ ( s 1: T ) (Eq. (31) ), the following conditions are equivalent, and are necessary for optimality of a twisted SMC procedure involving { π ∗ t } T t =1 ,

<!-- formula-not-decoded -->

Conditions ( i ) and ( iii ) directly correspond to the recursions for the optimal twist functions given in Prop. 3.2 and Prop. B.1.

Proof. ( i ) ⇐⇒ ( ii ) : It is clear that ( ii ) = ⇒ ( i ) as a special case for c = 1 . To show ( i ) = ⇒ ( ii ) , we have

<!-- formula-not-decoded -->

( i ) = ⇒ ( iii ) : Recursively applying ( i ) until time T suggests that

<!-- formula-not-decoded -->

( iii ) = ⇒ ( i ) : The target marginals clearly satisfy the recursion

<!-- formula-not-decoded -->

## A.2. Proof of Twist-Induced Proposal

Proposition 3.3. (Twist-Induced Proposal). For a given set of intermediate twisted targets π t ( s 1: t ) in Eq. (9) , the proposal which minimizes the variance of the one-step incremental importance weights w t is given by

<!-- formula-not-decoded -->

Proof. We seek to minimize the variance of the resulting importance weights, subject to a constraint on the proposal probabilities summing to 1. Introducing a Lagrange multiplier λ ( s 1: t -1 ) , we have

<!-- formula-not-decoded -->

Taking δ δq ( · ) = 0 implies

<!-- formula-not-decoded -->

where the derivative in the second term is zero since the q ( s t | s 1: t -1 ) cancel. Finally, we have

<!-- formula-not-decoded -->

where Z π t ( s 1: t -1 ) (or λ ) is chosen to enforce normalization.

We focused on the one-step twist-induced proposal in Prop. 3.3. However, this proposal is not optimal for resampling every c steps (as would also occur, for example, with adaptive resampling).

Proposition A.5 ( Multi-Step Twist Induced Proposal (Generalization of Prop. 3.3) ) . For resampling c -steps ahead, the optimal proposal (over s t +1: t + c -1 ) which minimizes the variance of the importance weights w t : t + c -1 ( s 1: t + c -1 ) is given by

<!-- formula-not-decoded -->

The proof follows the same reasoning as in the proof of Prop. 3.3 above, using the multistep weights w t : t + c -1 ( s 1: t + c -1 ) = ˜ π t + c -1 ( s 1: t + c -1 ) ˜ π t -1 ( s 1: t -1 ) q ( s t : t + c -1 | s 1: t -1 ) from Eq. ( c -Step SMC Weights).

Note that the denominator is not usually tractable for c &gt; 1 in language modeling applications.

## A.3. Derivation of CTL Gradient

Lemma A.6 ( Derivation of CTL Gradient ) . For the CTL loss min θ L CTL ( θ ):= min θ ∑ T t =1 D KL ( σ ( s 1: t ) ∥ ∥ π θ t ( s 1: t ) ) , the (negative) gradient with respect to the parameters θ is given by

<!-- formula-not-decoded -->

Proof. Consider expanding the form of π θ t ( s 1: t ) using Eq. (9), noting that the normalization log Z ψ t is independent of s 1: t . Taking the gradient with respect to θ using the log derivative identity ∇ θ f ( θ ) = f ( θ ) ∇ θ log f ( θ ) , we have

<!-- formula-not-decoded -->

## B. SMC with Intermediate Potentials and Connection with Soft Reinforcement Learning

In the main text, we focused on settings where the target distribution is defined by a potential ϕ ( s 1: T ) depending on full sequences only, as in Eq. (1). This setting highlights the need for (learned) twist functions to summarize the future expected value of the potential in the absence of intermediate target information.

In this appendix, we generalize our exposition to show how our twisted SMC framework can accommodate settings with intermediate potentials, which is evocative of connections with soft reinforcement learning (Levine, 2018). We leverage intuition from soft RL while introducing our general probabilistic interpretation, by using (sRL) = to instantiate the soft RL special case. In particular, soft RL will correspond to the terminal potential

<!-- formula-not-decoded -->

which corresponds to ϕ ( s 1: T ) = e βr T ( s 1: T ) if the potential is given at the final step only (as in RLHF, Korbak et al. (2022b)). However, we defer detailed discussion of soft RL to App. B.3. See Table 5 for several examples of intermediate potentials.

Finally, we formalize a notion of conditional target distributions and twist functions in App. B.2, which generalizes the exposition in the main text and captures our conditional twist learning experiments in Sec. 7.2.3.

## B.1. Twisted SMC with Intermediate Potentials

To generalize the exposition in the main text, we might consider defining the target as

<!-- formula-not-decoded -->

where Eq. (1) and the main text exposition corresponds to ϕ t ( s 1: t ) = 1 for t &lt; T .

Optimal Twists with Intermediate Potentials Using Eq. (31), the marginal distribution σ ( s 1: t ) = ∑ s t +1: T σ ( s 1: T ) over t tokens becomes

<!-- formula-not-decoded -->

As in Prop. 3.2, the goal of the optimal twist functions is to facilitate sampling from the intermediate marginals σ ( s 1: t ) of the target distribution σ ( s 1: T ) .

We consider two different quantities involved in defining the optimal twists, which differ in their treatment of the intermediate reward. For the soft RL setting, this corresponds to the natural distinction between Q -values and (soft) value functions V t .

<!-- formula-not-decoded -->

where : ∝ means 'defined to be proportional to' and Q ∗ t ( s t , s 1: t -1 ) = r t ( s 1: t ) + V ∗ t ( s 1: t ) in RL notation. See App. B.3 for detailed derivations in the soft RL special case. In general, Φ t captures the expectation of future potentials from t +1 : T , analogous to the (soft) value function. The twists ψ t play a role analogous to a Q -value, estimating both the immediate ϕ t and future value Φ t . In particular,

<!-- formula-not-decoded -->

We continue to refer to ψ t as the twist functions and focus on probabilistic interpretations based on ψ t instead of Φ ∗ t (see App. B.4 for additional discussion).

To show that this notation is consistent with the main text, consider the optimal twists ψ ∗ t ( s 1: t ) = ϕ t ( s 1: t ) Φ ∗ t ( s 1: t ) with no intermediate potentials, ϕ t ( s 1: t ) = 1 for t &lt; T . For t &lt; T , ψ ∗ t ( s 1: t ) = Φ ∗ t ( s 1: t ) reflect the future expected potential and for t = T , the terminal potential is ψ ∗ T ( s 1: T ) = ϕ T ( s 1: T ) , with no future potentials after step T , i.e. Φ T = 1 .

Building on Eq. (32)-(33) above, the following generalization of Prop. 3.2 defines the 'optimal' twists so as to obtain the intermediate target marginals σ ( s 1: t ) (see Prop. A.4).

Proposition B.1 ( Optimal Twists ) . For a given target distribution σ ( s 1: T ) in Eq. (31) , the optimal twist functions yield intermediate { π t } T -1 t =1 which match the target marginals. In regions where p 0 ( s 1: t ) &gt; 0 , the optimal twists are given by

<!-- formula-not-decoded -->

Up to a constant c t independent of s 1: t , the optimal twists ψ ∗ t are given by

<!-- formula-not-decoded -->

where c t is absorbed into the normalization constant Z ψ ∗ t . The optimal twists satisfy the recursion

<!-- formula-not-decoded -->

Remark B.2 ( Equivalence Class of ψ t and Φ t ) . Note that any rescaling of ψ t ← c t ¯ ψ t by a constant with respect to s 1: t will yield the same intermediate marginals π t ( s 1: t ) , due to the normalization constant Z ψ t which scales with ψ t . This defines an equivalent class in the space of functions. The same statement holds for Φ t . We express results such as Eq. (36) using proportionality ∝ . We define ψ t and Φ t as the members of their equivalent classes whose normalization Z ψ t and Z Φ t are equal. Thus, we have ψ t ( s 1: t ) = ϕ t ( s 1: t ) Φ t ( s 1: t ) .

Proof. Substituting Eq. (36) into Eq. (35), we obtain the desired marginal Eq. (32),

<!-- formula-not-decoded -->

where the final equality follows from absorbing the constant c t into Z ψ ∗ t , with 1 Z σ = c t Z ψ ∗ t and Z σ which normalizes ˜ σ ( s 1: t ) .

We will now use c t = Z ψ ∗ t Z σ to show the recursion in Eq. (37). Note that Eq. (36) implies

<!-- formula-not-decoded -->

where the second line follows from c t c t +1 = Z ψ ∗ t / Z σ Z ψ ∗ t +1 / Z σ . This demonstrates Eq. (37).

This leads to the following definition of the intermediate twisting targets (we defer the soft RL special case to App. B.3).

Definition B.3 ( Twisted Intermediate Targets ) . Using approximate twist functions { ψ t } T -1 t =1 , we define the twisted intermediate target distributions

<!-- formula-not-decoded -->

One-Step Twist-Induced Proposal Using Prop. 3.3 and Def. B.3 and noting that ϕ t -1 ( s 1: t -1 ) is independent of s t , we have the optimal one-step proposal

<!-- formula-not-decoded -->

where in the second line, we absorb terms which depend only on s 1: t -1 (and not s t ) into the normalization. In the soft RL special case, we have q π t ( s t | s 1: t -1 ) ∝ p 0 ( s t | s 1: t -1 ) e βQ t ( s t , s 1: t -1 ) (see Eq. (Twist-Induced Proposal (soft RL)) below).

## B.2. Conditional Twisted SMC

To formalize our notion of conditional twists in the infilling experiments (Sec. 7.2.3), we generalize our above framework to explicitly depend on 'observation' random variables { o t } T t =1 . This matches the common setting of SMC in state-space models (Briers et al., 2010; Gu et al., 2015; Lawson et al., 2022; Chopin et al., 2020). Our derivations in this section also emphasize that the optimal twist functions in Prop. B.1 learn functions proportional to conditional likelihoods of the future observation variables given the current sequence (see Eq. (40) below)). We recover the unconditional targets in the main text for fixed o T = 1 .

Consider a target distribution σ ( s 1: T | o 1: T ) conditioned on particular observation random variables o 1: T := { o t } T t =1 . We define a probabilistic model over observations σ ( o t | s 1: t ) = ϕ t ( o t , s 1: t ) as the intermediate potential, 8 which yields the target posterior

<!-- formula-not-decoded -->

where we interpret σ ( o 1: T | s 1: T ) = ∏ T t =1 σ ( o t | s 1: t ) and Z σ ( o 1: T ) = σ ( s 1: T ) to make the Bayesian posterior explicit in the last equality. Note, we now seek to estimate a different partition function Z σ ( o 1: T ) for each set of observation variables.

Using our infilling experiments in Sec. 7.2.3 as an example, consider (a sequence of) subsequent tokens o T = s T +1: T + c as observation variables, where the observation model is simply the base language model σ ( o T | s 1: T ) := p 0 ( s T +1: T + c | s 1: T ) .

Using Eq. (38), the intermediate marginals become

<!-- formula-not-decoded -->

noting that σ ( o t +1: T | s 1: t ) = ∑ s t +1: T σ ( o t +1: T , s t +1: T | s 1: t ) matches the second to last line.

The optimal twists take a similar form as Prop. B.1, but now as a function of the future observation or conditioning information. Further, the optimal twists is proportional to the conditional likelihoods (e.g., σ ( o t +1: T | s 1: t ) ) of future observations given s 1: t , which marginalize over future tokens (e.g., s t +1: T ),

<!-- formula-not-decoded -->

where f ( x, o ) o ∝ g ( x, o ) denotes proportionality up to a constant which depends on o only: ∃ c ( o ): f ( x, o ) = c ( o ) g ( x, o ) . These equations can be confirmed by comparing Prop. B.1 with the last two lines in Eq. (39).

The intermediate marginals over partial sequences can finally be rewritten as either

<!-- formula-not-decoded -->

We discuss the choice of parameterization using ψ t versus Φ t in App. B.4.

The conditional twist learning formulation matches the setting of Lawson et al. (2022), to which we refer the reader for additional discussion. We use this conditional perspective to derive classification losses for twist learning in App. C.3-C.4.

8 Note, rescaling ϕ t ( s 1: t , o t = 1) by a constant c with respect to o t , s 1: t does not affect the target posterior in Eq. (38). For example, with terminal potential only: σ ( s 1: T | o T ) = p 0 ( s 1: T ) ϕ T ( s 1: T ,o T ) /c ∑ s 1: T p 0 ( s 1: T ) ϕ T ( s 1: T ,o T ) /c = 1 Z σ ( o T ) p 0 ( s 1: T ) ϕ T ( s 1: T , o T ) as long as the scaling factor is independent of o T and s 1: T .

Unconditional Targets as a Special Case In cases where we are only learning twists for a single set of conditioning information such as a single classifier label or a reward model, note that we can omit explicit conditioning information in ψ t ( s 1: t , o t ) and consider setting { o t = 1 } T t =1 . With terminal potential only as in the main text, we write σ ( o T = 1 | s 1: T ) = ϕ ( s 1: T ) and the overall target distribution as σ ( s 1: T ) = σ ( s 1: T | o T = 1) ∝ p 0 ( s 1: T ) ϕ T ( s 1: T ) . Thus, the formulation in Eq. (38)-Eq. (40) strictly generalizes our exposition in the main text and App. B.1. With intermediate potentials, we set σ ( o 1: T = 1 | s 1: T ) = ∏ T t =1 ϕ t ( s 1: t ) .

Our notation also matches the exposition in Levine (2018) for the soft RL case with a binary observation or 'optimality' random variable σ ( o t = 1 | s 1: t -1 , s t ) = e βr t ( s 1: t -1 ,s t ) , where the reward is a function of the state x t = s 1: t -1 and action a t = s t pair (see the MDP interpretation in App. B.3).

## B.3. Connection with Soft Reinforcement Learning

In this section, we more explicitly describe the soft reinforcement learning setting (Levine, 2018) commonly used in RLHF (Korbak et al., 2022b) as a special case of our probabilistic framework. Again, we use notation (sRL) = to indicate that the expressions in this section correspond to a particular instance of our SMC framework where ϕ ( s 1: T ) = e βr ( s 1: T ) .

Summary of Soft RL Notation To summarize the below derivations, we state the relevant assignments for the soft RL case. We focus on the optimal case for simplicity, but note that approximate versions play identical roles,

<!-- formula-not-decoded -->

where ψ ∗ t ( s 1: t ) = ϕ t ( s 1: t ) Φ ∗ t ( s 1: t ) or Q ∗ t ( s t , s 1: t -1 ) = r t ( s 1: t ) + V ∗ t ( s 1: t ) . In the other direction, we have

<!-- formula-not-decoded -->

MDP Interpretation To draw connections with soft RL, we view language model controlled decoding as a MDP, where the prompt is drawn from an initial state distribution s 0 ∼ ν 0 , an action policy π ( a t | x t ) = q ( s t | s 1: t -1 ) selects the next token a t = s t given a partial sequence x t = s 1: t -1 as the state, and deterministic environment transitions P ( x t +1 = s 1: t | a t = s t , x t = s 1: t -1 ) = δ ( x t = concat ( s t , s 1: t -1 )) append the selected token to update the state. Discounting may also be included without difficulty. The reward is given by r t ( s 1: t ) .

Final Target Distribution We define the target distribution as the solution to the following variational optimization which solves the regularized MDP described above,

<!-- formula-not-decoded -->

which corresponds to the choice ϕ t ( s 1: t ) = e β r t ( s 1: t ) as in Eq. (Twist to Soft RL). The soft value is defined as the maximum value of the above optimization for optimal q ∗ ( s 1: T ) , and corresponds to the scaled log partition function

<!-- formula-not-decoded -->

which can be confirmed by substituting q ( s 1: T ) = σ ( s 1: T ) from Eq. (42) into the maximization on the right side of Eq. (43). Although we omit the dependence of Z σ ( s 0 ) on the prompt s 0 for notational simplicity (see Eq. (1)), note that V ∗ 0 := V ∗ ( s 0 ) naturally corresponds to the soft value of the prompt as the initial state in the MDP.

Optimal Intermediate Marginals and Soft Value Decomposing the maximization in Eq. (43) into optimizations over each q ( s t +1 | s 1: t ) , we define the intermediate soft value V ∗ t ( s 1: t ) as the maximum of the expected future regularized reward

<!-- formula-not-decoded -->

where, in the third line, we isolate the optimization over q ( s t | s 1: t -1 ) by (i) assuming optimality at τ &lt; t and (ii) substituting the optimal value V ∗ t +1 ( s 1: t +1 ) = max q ( s t +2: T | s 1: t +1 ) [ ... ] of the maximization over q ( s t +2: T | s 1: t +1 ) (i.e. recursively applying the second line).

The optimal intermediate marginal can be written using either V ∗ t ( s 1: t ) or Q ∗ t ( s t , s 1: t -1 ) form (as in Eq. (33) above, or by substituting the optimal V ∗ t or Q ∗ t into the twist targets below).

Twisted Intermediate Targets We state the approximate twisting targets for both V t or Q t parameterizations in order to make connections with soft RL losses in App. C. For approximate V t ( s 1: t ) or Q t ( s t , s 1: t -1 ) , we have

<!-- formula-not-decoded -->

where the final twisting target is given by Eq. (42) and the optimal Q -values are defined as

<!-- formula-not-decoded -->

One-Step Proposal Finally, the optimal one-step proposal (e.g. in V t form) can be derived either (i) as the twist-induced proposal from Eq. (Twist Targets (Soft RL V) ) and Prop. B.1 or (ii) as the solution to the one-step optimization in the third line of Eq. (Optimal Intermediate Soft Value). As in Eq. (Twist-Induced Proposal ( ψ ) ),

<!-- formula-not-decoded -->

(Twist-Induced Proposal (soft RL))

We define the one-step log normalization constant induced by an approximate V t or Q t as V V t or V Q t , respectively,

<!-- formula-not-decoded -->

such that, for example, q π t ( s t | s 1: t -1 ) = p 0 ( s t | s 1: t -1 ) e βQ t ( s t , s 1: t -1 ) -βV Qt ( s 1: t -1 ) .

RLHF Minimizes D KL ( q ∥ σ ) Note that, for a given suboptimal q ( s 1: T ) , the value of the variational optimization in Eq. (42) is a lower bound on the (scaled) log partition function V ∗ 0 = 1 β log Z σ . Similarly to the standard Evidence Lower Bound, the gap in this lower bound is given by the KL divergence

<!-- formula-not-decoded -->

In this sense, we consider soft RL or policy gradient methods such as PPO which optimize Eq. (42) as targeting σ ( s 1: T ) by minimizing D KL ( q ( s 1: T ) ∥ σ ( s 1: T )) (Korbak et al., 2022b).

## B.4. Remarks on Parameterization

While the twisting targets (Eq. (Twist Targets ( ψ ) )) and twist-induced proposal (Eq. (Twist-Induced Proposal ( ψ ) )) may equivalently be parameterized using approximate Φ t , we focus on the ψ t parameterization to match the main text. In particular, recall that the optimal twists satisfy ψ ∗ t ( s 1: t ) = ϕ t ( s 1: t ) Φ ∗ t ( s 1: t ) for all t . With no intermediate potential ( ϕ t = 1 for t &lt; T ), our approximate twists estimate ψ t ( s 1: t ) ≈ Φ ∗ t ( s 1: t ) ∝ ∑ s t +1: T p 0 ( s t +1: T | s 1: t ) ϕ T ( s 1: T ) for t &lt; T . In this section, we describe how the presence of intermediate potentials may affect the choice of twist parameterization.

The twist-induced proposal may not be tractable to evaluate at the final timestep, since it may be costly to evaluate the terminal potential ϕ T ( s 1: T ) for all s T ∈ V given a context s 1: T -1 (as described in Sec. 3.2). Thus, we learn an approximate ψ T ( s 1: T ) ≈ ϕ T ( s 1: T ) for proposal sampling, which can be easily evaluated over |V| next tokens. The final π T ( s 1: T ) = σ ( s 1: T ) is defined using ϕ ( s 1: T ) in order to preserve unbiased estimation. However, after sampling the proposal according to ψ T , we only need to evaluate ϕ ( s 1: T ) over K full sequences to calculate the importance weights at the final step (Eq. (16)). See Intermediate Potential Tractable over K Sequences Only paragraph below.

Intermediate Potentials Tractable over |V| Sequences However, in settings where the intermediate potentials ϕ t ( s 1: t ) are tractable to calculate for all s t ∈ V given s 1: t -1 (e.g. using an indicator function or forward pass in a transformer architecture, as in Table 5), it may be useful to use a Φ t parameterization of the twist targets and twist-induced proposal. This allows us to use the exact immediate potentials ϕ t ( s 1: t ) alongside an estimated Φ θ t , instead of an approximate ψ θ t ≈ ϕ t Φ ∗ t which estimates both the immediate ϕ t and future expected value of potentials Φ ∗ t . Using notation established in Eq. (33) and Prop. B.1, the twisting targets in Eq. (Twist Targets ( ψ ) ) can be rewritten using a Φ θ t parameterization

<!-- formula-not-decoded -->

with π T ( s 1: T ) = σ ( s 1: T ) as before. The twist-induced proposal q π t ( s t | s 1: t -1 ) ∝ p 0 ( s t | s 1: t -1 ) ϕ t ( s 1: t ) Φ θ t ( s 1: t ) and its normalization constant are tractable in this case, by evaluating both the given ϕ t ( s 1: t ) and parameterized Φ θ t ( s 1: t ) in a single forward pass and normalizing over the discrete vocabulary of next tokens.

Intermediate Potentials Tractable over K Sequences Only In cases where the intermediate potentials are difficult to evaluate, we would like to limit evaluation of ϕ t ( s 1: t ) to only K partial sequences. In this case, parameterizing the twisted targets π t using ψ θ t or Q θ t (Eq. (Twist Targets ( ψ ) ), Eq. (Twist Targets (Soft RL Q) )) instead of Φ θ t or V θ t may be preferable to ensure a tractable twist-induced proposal. Separate parameterizations of the proposal (using ψ ξ t ) and targets ( ϕ t Φ θ t ) might also be considered.

In the case of the final timestep described above or in Sec. 3.2, note that we use a learned ψ ξ T to parameterize a tractable variational proposal q T ( s T | s 1: T -1 ) . In this case, we have no future value Φ T ( s 1: T ) = 1 and only need to evaluate the terminal potential ϕ ( s 1: T ) for calculating importance weights over K sequences.

## C. Twist Learning Losses

In this section, we describe various methods for twist learning beyond our proposed contrastive twist learning (CTL) procedure from Sec. 4. In App. C.1, we first describe several losses from the soft RL literature from a probabilistic perspective, building closely on our developments in App. B.1. We then proceed to describe SIXO (Lawson et al., 2022) and FUDGE (Yang &amp; Klein, 2021) in App. C.3-C.4.

We emphasize losses found in related work or used as experimental baselines using equation tags (e.g. Eq. (SIXO)), where equations Eq. (RL Baseline), Eq. (SIXO), Eq. (FUDGE) are used in our experiments. We consider settings with intermediate potentials in App. C.1, but focus on the ( ϕ t = 1 for t &lt; T ) setting in the remainder of the section, as in the main text.

## C.1. Soft Q-Learning (RL) and Path Consistency Losses from Log Importance Weights

From the probabilistic perspective of the SMC log importance weights, we can derive several losses for twist learning, including soft Q-learning and path consistency learning (PCL) (Nachum et al., 2017) losses from the soft RL literature.

A general principle for deriving loss functions would be to minimize the variance of the (log) importance weights under some sampling distribution π s , which leads to constant importance weights at optimality. To draw connections with previous work, we also consider minimizing the square of the log weights, which at optimality, ensures that log w = 0 and w = 1 are equal to a particular constant. We will proceed to parameterize the twist functions using parameters θ and consider loss terms which minimize the variance or square of c -step log weights at time t ,

<!-- formula-not-decoded -->

L ( t,c ) ( θ ) indicates 'consistency' in log -weight space for c -step-ahead weights at time t (see Eq. ( c -Step SMC Weights)).

We will consider various choices of parameterization and proposal in the following subsections. For example, let L ( t,c ) log Cons ( θ ; { ψ t , q π t } ) denote the log-consistency loss corresponding to twisting targets parameterized by ψ θ t and the twist induced proposal q π (note, our notation for the one-step weights w ( s ) does not make these choices explicit).

log Cons t t 1: t

For reference, we derive the log importance weights with intermediate potentials and arbitrary q as

<!-- formula-not-decoded -->

Various special cases arise from choices of twist parameterizations and proposals in the following subsections.

## C.1.1. SOFT Q-LEARNING AND RL BASELINE

For single-step log-weights, the ψ -parameterization of the targets (Eq. (Twist Targets ( ψ ) ), Eq. (Twist Targets (Soft RL Q) )), and the twist-induced proposal (Eq. (Twist-Induced Proposal ( ψ ) ), Eq. (Twist-Induced Proposal (soft RL))), we have

<!-- formula-not-decoded -->

where the second term log Z π t ( s 1: t -1 ) = log ∑ s t p 0 ( s t | s 1: t -1 ) ψ t ( s 1: t ) normalizes the twist-induced proposal (Eq. (14)). Minimizing the sum of one-step log consistency losses (i.e. squared log weights in Eq. (48)) will yield the familiar soft Q -learning loss (e.g. Lioutas et al. (2022) Eq. (4)-(5)). Adjusting indexing from Eq. (48) and introducing a stop-gradient within log Z π t ( s 1: t -1 ) , we have

<!-- formula-not-decoded -->

In the final line, we rewrite the loss for the soft RL special case, ϕ t ( s 1: t ) = e βr t ( s 1: t ) using the substitutions in Eq. (Twist to Soft RL). Note that the log -normalization term is analogous to an induced soft value V Q θ t ( s 1: t -1 ) = 1 β log ∑ s t p 0 ( s t | s 1: t -1 ) e βQ θ t ( s t , s 1: t -1 ) , so that each squared error loss has the form E [ β 2 ( r t + V t -Q t ) 2 ] . Hence, we refer to this loss as Soft Q-learning loss.

The log -normalization term, which arises from normalizing the twist-induced proposal, is analogous to the 'target' value in deep Q -learning. Lioutas et al. (2022) consider the soft-Q learning loss to SMC sampling in self-driving applications where interaction with the environment is expensive. Lawson et al. (2018) adopt a similar loss function (using a parameterization of the value V θ t ) in the setting of state-space models with tractable intermediate rewards.

RL Baseline with no Intermediate Reward The soft Q-learning loss in Eq. (Soft Q Learning) simplifies nicely in the case of no intermediate rewards, as in the main text ( ϕ t ( s 1: t ) = 1 for t &lt; T and Φ T = 1 ).

Written in terms of twist functions, we separate the terms at t &lt; T and t = T for purposes of exposition

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For intermediate timesteps, note that Eq. (RL Baseline) enforces the recursion ψ θ t -1 ( s 1: t -1 ) = ∑ s t p 0 ( s t | s 1: t -1 ) ψ θ t ( s 1: t ) in Eq. (13) of the main text, albeit in log space. In App. C.2 below, we consider the one-step squared error loss enforcing this recursion directly (without logarithms), i.e. E π s [( ψ θ t -1 ( s 1: t -1 ) -∑ s t p 0 ( s t | s 1: t -1 ) ψ θ t ( s 1: t )) 2 ] ,

## C.1.2. PATH CONSISTENCY LEARNING (FOR TWIST LEARNING)

Using the value parameterization of the targets ( Φ t or V t , see Eq. (Twist Targets ( Φ ) ), Eq. (Twist Targets (Soft RL V) )), the one-step log consistency loss with arbitrary proposal q recovers the path-consistency loss (PCL) from Nachum et al. (2017).

Switching to a Φ θ t parameterization of the twisting targets, we substitute ψ θ t ( s 1: t ) = ϕ t ( s 1: t ) Φ θ t ( s 1: t ) into the log importance weights in Eq. (48). The log-consistency loss becomes,

<!-- formula-not-decoded -->

In particular, substituting the soft RL potential terms from Eq. (Twist to Soft RL), Eq. (PCL) recovers the path consistency loss from Nachum et al. (2017). Note that we derived PCL from an importance sampling perspective, whereas PCL was originally derived by enforcing KKT conditions of the soft RL problem.

We might also consider multi-step losses for various c . Minimizing the square of the multi-step log weights with arbitrary q recovers the multi-step PCL loss (Nachum et al., 2017),

<!-- formula-not-decoded -->

where we write the ψ θ t parameterization in Eq. (50) explicitly for use in App. D.1. While PCL considers learned a proposal or policy q with the goal of approximating the solution of a regularized MDP, we leave joint learning of proposals { q ξ ( s t | s 1: t -1 ) } T t =1 and SMC target twists { ψ θ t ( s 1: t ) } T t =1 or { V θ t ( s 1: t ) } T t =1 to future work.

In App. E, we describe using PCL to learn the proposal only (Guo et al., 2021), with the values V Q t ( s 1: t ) induced from learned proposal twists Q ξ t ( s t +1 , s 1: t ) which define { q ξ Q t ( s t +1 | s 1: t ) } T -1 t =0 (in similar fashion to Eq. (Twist-Induced Proposal (soft RL)), but without reference to twisting targets).

## C.2. Controlled Decoding Losses via Optimal Twist Identities (Mudgal et al., 2023)

In Prop. B.1 (or Prop. 3.2 and Eq. (13) in the main text), we noted that the optimal twists satisfy the following relationships

<!-- formula-not-decoded -->

We proceed to describe two 'controlled decoding' (CD) losses from Mudgal et al. (2023) as using a squared error loss to enforce the optimality conditions in Eq. (51), for settings with no intermediate potentials ( ϕ t ( s 1: t ) = 1 for t &lt; T ). Mudgal et al. (2023) also propose two ways to use the learned 'twists' at inference time, which we discuss in relation to our proposed SMC framework in App. D.1.

CD-Q The CD-Q loss from Mudgal et al. (2023) corresponds to minimizing the one-step recursion in Eq. (51) using the expected squared error under a (possibly off-policy) sampling distribution π s . Assuming no intermediate reward and an additional squared error loss to approximate the terminal potential ψ θ T ( s 1: T ) ≈ ϕ ( s 1: T ) , we have

<!-- formula-not-decoded -->

Eq. (CD-Q) enforces the same optimality condition as the Eq. (RL Baseline) loss (i.e. ψ θ t ( s 1: t ) = ∑ s t +1 p 0 ( s t +1 | s 1: t ) ψ θ t +1 ( s 1: t +1 ) ), without log scaling of each term inside the squared error. At optimality, we have zero-variance one-step importance weights ( w ( s 1: t ) = 1 in Eq. (10)) for the twist-induced proposal.

At optimality, we in fact also have ψ θ t ( s 1: t ) = ∑ s t +1: T p 0 ( s t +1: T | s 1: t ) ϕ T ( s 1: T ) (as in Eq. (51) and the proof of Prop. B.1).

CD-FUDGE While we might naively like to consider the loss E π s ( · ) [( ψ θ t ( s 1: t ) -∑ s t +1: T p 0 ( s t +1: T | s 1: t ) ϕ ( s 1: T ) ) 2 ] to enforce Prop. 3.2 or Eq. (51), note that marginalization over multiple steps is not tractable in general.

Instead, the CD-FUDGE loss 9 defined as

<!-- formula-not-decoded -->

can be shown to have the same gradient as the desired (but intractable) squared error loss above (Mudgal et al., 2023).

Since the minimizer of the expected squared error (under p 0 ( s t +1: T | s 1: t ) ) to a single function ψ θ t ( s 1: t ) (which is independent of s t +1: T ) is given by the conditional expectation (Banerjee et al., 2005), we can also see that Eq. (CD-FUDGE) has the desired minimum ψ θ t ( s 1: t ) = ∑ s t +1: T p 0 ( s t +1: T | s 1: t ) ϕ ( s 1: T ) . Note, it is crucial that the inner expectation samples rollouts under the base model p 0 ( s t +1: T | s 1: t ) to obtain the desired conditional expectation as the minimizer. While it appears that any prefix sampling distribution can be used, π s = p 0 allows for losses to be calculated at all t in a single sampling run.

Mudgal et al. (2023) also propose two decoding-time usages for the learned twist functions ψ θ t : stochastic token-by-token sampling and argmax decoding of partial sequences. We discuss their inconsistencies with our SMC framework in App. D.

CD-FUDGE for log ψ θ t We can also compare Eq. (CD-FUDGE) with the multi-step PCL loss in Eq. (50), choosing ϕ t = 1 for t &lt; T and the proposal equal to the base model q = p 0 so that the proposal terms cancel. Noting that ψ T ( s 1: T ) = ϕ ( s 1: T ) is fixed to the exact terminal potential and choosing the c = T -t +1 -step PCL loss for each t , note that Eq. (50) would reduce to ∑ t E [ ( log ϕ ( s 1: T ) + 0 -log ψ θ t ( s 1: t ) -0 ) 2 ] . Deng &amp; Raffel (2023) optimize this loss with reweighting of terms based on timestep (higher weight for t ≈ T ). Eq. (CD-FUDGE) optimizes the squared error of the difference without log scaling of each term , under appropriate sampling of rollouts. 10

9 Note, we reserve the naming convention FUDGE (Yang &amp; Klein, 2021) for a binary cross entropy loss described in App. C.4, as opposed to the CD-FUDGE squared error loss from Mudgal et al. (2023).

10 Note the difference in choice of proposal between Eq. (CD-Q) (twist-induced q = q π t ) and Eq. (CD-FUDGE) (base q = p 0 ).

## C.3. SIXO: Smoothing Inference with Twisted Objectives (Lawson et al., 2022)

Lawson et al. (2022) adopt a noise-contrastive estimation loss (Gutmann &amp; Hyv¨ arinen, 2010) to learn the target twist functions using binary classification. For state space models, Lawson et al. (2022) adopt our setting in App. B.2 with observation variables o t emitted based on the sampling state s 1: t (or simply s t ) and a known likelihood ϕ t ( o t , s t ) = σ ( o t | s t ) . As discussed in App. B.4, in these settings with easily evaluable intermediate potentials, it may be preferable to parameterize Φ θ t ( s 1: t , o t +1: T ) as in Eq. (Twist Targets ( Φ ) ). Lawson et al. (2022) indeed use this parameterization (see their Eq. 5).

Recall from Eq. (39) that the optimal twists or future values amount to conditional likelihoods,

<!-- formula-not-decoded -->

where o ∝ denotes proportionality up to a constant which depends on o only. Using Bayes rule, we have

<!-- formula-not-decoded -->

noting that σ ( o t +1: T ) and p 0 ( s 1: t ) are marginals of σ ( s 1: t , o t +1: T ) by definition. The above reasoning suggests that we may learn the twists, or likelihood ratio Φ ∗ t ( s 1: t , o t +1: T ) ∝ σ ( o t +1: T | s 1: t ) ∝ σ ( s 1: t | o t +1: T ) /p 0 ( s 1: t ) , using a classifier which seeks to distinguish samples from σ ( s 1: t | o t +1: T ) and p 0 ( s 1: t ) (Gutmann &amp; Hyv¨ arinen, 2010; Lawson et al., 2022). In particular, at each t , we classify the event y = 1 , indicating that s 1: t ∼ σ ( s 1: t | o t +1: T ) , or y = 0 , indicating that s 1: t ∼ p 0 ( s 1: t ) .

Consider a given o t +1: T , which can be either o t +1: T = 1 in the unconditional case or o t +1: T ∼ π s ( o t +1: T ) drawn from a behavioral policy as discussed below. The SIXO loss becomes

<!-- formula-not-decoded -->

Note that we can perform approximate positive sampling as in Sec. 4 to estimate expectations in the first term.

Exact Conditional Sampling However, we can also use the BDMC trick in Sec. 3.3 to obtain exact target samples for general observation variables. In order to facilitate tractable sampling, we optimize the Eq. (SIXO) loss over a sampling distribution π s ( o 1: T ) = σ ( o 1: T ) for all t , such that the objective becomes

<!-- formula-not-decoded -->

With this choice, note that we may sample once from σ ( s 1: T , o 1: T ) = ∏ T t =1 p 0 ( s t | s 1: t -1 ) σ ( o t | s 1: t ) using ancestral sampling and use the appropriate truncations for positive sampling terms involving σ ( s 1: t , o t +1: T ) . By shuffling observation variables across a batch of K samples, we may obtain samples from the product of marginals p 0 ( s 1: T ) σ ( o 1: T ) or p 0 ( s 1: t ) σ ( o t +1: T ) in the negative sampling term.

In the main text, note that we condition on o T = 1 or o T = s T +1: T + c (for infilling).

Gradient and Comparison with CTL Proceeding with the ψ θ t parameterization for the target σ ( s 1: T | o T ) = σ ( s 1: T ) with fixed o T and unconditional twists ψ θ t ( s 1: t ) , the gradient of Eq. (SIXO) with respect to θ is

<!-- formula-not-decoded -->

The SIXO gradient is superficially similar to our CTL gradient in Sec. 4.1, in that it involves ∇ θ log ψ θ t under positive and negatives samples. However, viewing ˜ π θ t ( s 1: t ) = p 0 ( s 1: t ) ψ θ t ( s 1: t ) as the unnormalized density of our intermediate twisting target, we can see that the second term in the SIXO update includes ˜ π θ t ( s 1: t ) . Rewriting to highlight differences with our CTL gradient, we have

<!-- formula-not-decoded -->

To compare the two, first note that the positive sampling gradient in SIXO is scaled by a factor of 1 1+ ψ θ t ( s 1: t ) factor (which reflects the misclassification probability under ψ θ t ). For the negative sampling terms, note that ˜ π θ t ( s 1: t ) is divided by a factor of 1 1+ ψ θ t ( s 1: t ) in the SIXO gradient, instead of the true normalization constant Z ψ t for the gradient of our CTL loss Eq. (22).

## C.4. FUDGE: Future Discriminators (Yang &amp; Klein, 2021)

In contrast to SIXO, the FUDGE method from Yang &amp; Klein (2021) seeks to directly learn a discriminative classifier to match the conditional likelihood ψ ∗ t ( s 1: t , o T ) ∝ σ ( o T | s 1: t ) or ψ ∗ t ( s 1: t , o t : T ) ∝ σ ( o t : T | s 1: t ) (see App. B.2).

As before, we define the joint distribution σ ( s 1: T , o T ) = p 0 ( s 1: T ) σ ( o T | s 1: T ) with ϕ ( s 1: T , o T ) = σ ( o T | s 1: T ) . From Eq. (52) above or App. B.2 Eq. (40), we have

<!-- formula-not-decoded -->

Yang &amp; Klein (2021) consider training a 'future discriminator' ψ θ t ( s 1: t , o T ) ≈ σ ( o T | s 1: t ) which, as in Eq. (54) marginalizes over future tokens to predict the expected probability that a full sequence with prefix s 1: t emits o T (e.g., let o T = a be the probability of a classifier for class a , or the probability that s 1: T satisfies a desired attribute indicated by a boolean o T = 1 ). In similar fashion to SIXO in the previous section, we define a binary random variable y such that

<!-- formula-not-decoded -->

where we directly parameterize p ψ θ t ( y | s 1: t , o T ) = ψ θ t ( s 1: t , o T ) to be a probability distribution (e.g. using a sigmoid or softmax activation). For a given observation random variable o T and partial sequence s 1: t , we can define the FUDGE loss

<!-- formula-not-decoded -->

where, in moving from the second to the third line, we have used the fact that σ ( y = 1 | s 1: t , o T ) = σ ( o T | s 1: t ) = ∑ s t +1: T p 0 ( s t +1: T | s 1: t ) σ ( o T | s 1: T ) from Eq. (54) and Eq. (55). At the optimum, p ψ θ t ( y = 1 | s 1: t , o T ) = σ ( y = 1 | s 1: t , o T ) implies ψ θ t ( s 1: t , o T ) = σ ( o T | s 1: t ) , as desired.

While sampling may be done using an arbitrary distribution over prefixes s 1: t and observation o T , Eq. (FUDGE) requires that rollouts be sampled under the base model p 0 ( s t +1: T | s 1: t ) in order to ensure sampling from the appropriate distribution σ ( y = 1 | s 1: t , o T ) . This restriction is similar to what we required in Eq. (CD-FUDGE), although the loss in Eq. (FUDGE) is based on cross entropy classification rather than a squared error. We discuss the choices made in our experiments below.

Yang &amp; Klein (2021) Setting In the original FUDGE paper, Yang &amp; Klein (2021) consider learning from a dataset of labelled examples ( s 1: T , o T ) or ( s 1: t , o T ) for a binary observation variable o T = 1 which defines the target distribution.

Unconditional Twist Setting For the unconditional twist experiments in Sec. 7.2.1-7.2.2, we sample under the base model proposal π s ( s 1: t ) = p 0 ( s 1: t ) where the target distribution conditions on o T = 1 and σ ( o T = 1 | s 1: T ) = ϕ ( s 1: T ) = σ ( y = 1 | s 1: T , o T = 1) . In particular, we optimize

<!-- formula-not-decoded -->

Conditional Twist Setting For conditional twist learning, we can consider amortizing learning the twists ψ t ( s 1: t , o T ) over some distribution of observation variables π s ( s 1: t , o T ) . In particular, in our infilling experiments in Sec. 7.2.3, we consider sampling under the following joint distribution,

<!-- formula-not-decoded -->

which we can sample from by first sampling from p 0 ( s 1: T ) σ ( o T | s 1: T ) and then dropping s t +1: T subsequence. Therefore, the overall objective becomes

<!-- formula-not-decoded -->

where the expectation p 0 ( s 1: T ) includes the expectation under p 0 ( s t +1: T | s 1: t ) from Eq. (FUDGE). Note that rollout of s t +1: T | s 1: t used to sample from p 0 ( s 1: T ) should be independent of the rollout used to sample from σ ( o T | s 1: t ) .

## D. Decoding Strategies using Learned Twists from Mudgal et al. (2023)

## D.1. Proposal Sampling in Mudgal et al. (2023)

As noted in App. C.2 (and in L ∗ ( θ ) in Mudgal et al. (2023)), the CD losses can be seen as enforcing the optimality conditions

<!-- formula-not-decoded -->

In RL terms, we interpret the twists ψ cd ∗ t as performing policy evaluation of the expected unregularized 'reward' ϕ ( s 1: T ) under a fixed policy p 0 ( s 1: T ) . The notation of Mudgal et al. (2023) (their Eq. (1), (5), our Eq. (57)) indeed corresponds to

<!-- formula-not-decoded -->

However, Mudgal et al. (2023) propose to use the learned twist functions ψ θ t to perform one-step sampling as

<!-- formula-not-decoded -->

We proceed to explain that this scheme does not correspond to sampling from the twist-induced proposal under two different definitions of the target σ ( s 1: T ) (or potential ϕ ( s 1: T ) ) in our SMC framework.

Comparison with Our ϕ ( s 1: T ) = r cd ( s 1: T ) Case: As we have argued above, the CD-Q and CD-FUDGE may be viewed as learning twist values ψ θ t for a terminal potential ϕ ( s 1: T ) = r cd ( s 1: T ) . However, our twist-induced proposal which minimizes the variance of the one-step importance weights with these SMC targets { π θ t } would yield

<!-- formula-not-decoded -->

which, compared to Eq. (CD proposal) does not exponentiate or scale ψ θ t and is directly proportional to the expected r cd .

Comparison with Our ϕ ( s 1: T ) = e βr cd ( s 1: T ) Case (Soft RL): The stochastic sampling in Eq. (CD proposal) is also reminiscent of the twist-induced proposal in the soft RL case of our framework where, in contrast to Eq. (CD reward), the target is defined via ϕ ( s 1: T ) = e βr cd ( s 1: T ) . As in App. B.3,

<!-- formula-not-decoded -->

We proceed to write both q cd t and q π t as the solution to a variational optimization, highlighting similarities in blue, but noting the different definitions of ϕ in terms of r cd . We assume no intermediate potential or reward, and consider the optimal twists to emphasize the role of r cd. Using Mudgal et al. (2023) Eq. 2 and Thm 2.1 (for CD) and Eq. (Optimal Intermediate Soft Value) (for soft RL), we have

q cd ∗ t ( s t | s 1: t -1 ) = arg max q ( s t | s 1: t -1 ) E q ( s t | s 1: t -1 ) [ E p 0 ( s t +1: T | s 1: t ) [ r cd ( s 1: T ) ] ︸ ︷︷ ︸ ψ cd ∗ t ( s 1: t ) (for ϕ = r cd) ] -1 β D KL ( q ( s t | s 1: t -1 ) ∥ p 0 ( s t | s 1: t -1 )) (CD proposal optimization) q π ∗ t ( s t | s 1: t -1 ) = arg max q ( s t | s 1: t -1 ) E q ( s t | s 1: t -1 ) [ 1 β log E p 0 ( s t +1: T | s 1: t ) [ e βr cd ( s 1: T ) ] ︸ ︷︷ ︸ V ∗ t ( s 1: t ) (for ϕ = e βr cd ) ] -1 β D KL ( q ( s t | s 1: t -1 ) ∥ p 0 ( s t | s 1: t -1 ))

(Soft RL proposal optimization)

The second terms of Eq. (CD proposal optimization) and Eq. (Soft RL proposal optimization) match and correspond to one-step KL divergence regularization of the policy q t ( s t | s 1: t -1 ) . However, the expectation terms differ as we now discuss.

Soft Values Account for Future Regularization Using Eq. (Optimal Intermediate Soft Value) to expand the definition of the soft value function, we see that Eq. (Soft RL proposal optimization) also implicitly contains an expected terminal reward,

<!-- formula-not-decoded -->

As β → 0 in Eq. (58), this optimization strictly enforces q ( s t +1: T | s 1: t ) = p 0 ( s t +1: T | s 1: t ) , and the soft value function recovers the expected reward under the base model E p 0 ( s t +1: T | s 1: t ) [ r cd ( s 1: T )] , which appears in first term Eq. (CD proposal optimization). On the other hand, the second term in Eq. (CD proposal optimization) uses β &gt; 0 for optimization of the proposal q ( s t | s 1: t -1 ) at the current step. This inconsistency in Eq. (CD proposal optimization) (using β = 0 in the first term and β &gt; 0 in the second term) arises from the fact that Eq. (CD proposal optimization) does not consider the effect of future regularization, while the MDP formulation in Eq. (Soft RL proposal optimization) does so via the optimization in Eq. (58) and the log-mean-exp form of the soft value function V ∗ t .

On Mudgal et al. (2023)'s One-Step Proposal and SMC Interpretation As noted in Eq. (57), the twists learned by Mudgal et al. (2023) correspond to policy evaluation for the reward r cd under the base model p 0 . However, we have argued that the one-step proposal in Eq. (CD proposal) (which considers one-step KL regularization of q cd t to p 0 ) does not immediately fit within our SMC framework. In particular, it is not apparent that the composition of one-step proposals q cd ( s 1: T ) = ∏ t τ =1 q cd τ ( s τ | s 1: τ -1 ) samples from the marginals σ ( s 1: t ) of a natural target distribution σ ( s 1: T ) at optimality.

Flexible Inference-Time β Scaling The experiments in Mudgal et al. (2023) evaluate tradeoff curves between expected reward and D KL ( q cd ( s 1: T ) ∥ ∥ p 0 ( s 1: T ) ) for various values of regularization strength β . Since the twists learned by Mudgal et al. (2023) in Eq. (57) do not depend on β , sampling according to Eq. (CD proposal) or Eq. (CD proposal optimization) has the benefit of allowing flexible tempering or β -scaling at inference time without additional learning.

Such tradeoff curves are also natural from the perspective of soft-RL (c.f. Eq. (42) and Eq. (46)). While Eq. (58) appears to require separate twist-learning procedures for each β , flexible inference-time β scaling could be achieved with a single training run in our framework by learning a conditional twist network ψ θ t ( s 1: t , β ) which considers β in its input and training loss, or adapting the methods of (Bae et al., 2022) proposed in the context of rate-distortion optimization.

̸

Comparison with Khanov et al. (2024) Khanov et al. (2024) consider softmax decoding similar to Eq. (Twist-Ind. proposal ( ϕ = r cd)). However, instead of V θ t ( s 1: t ) as the logit, they use a reward model r T ( s 1: T ) which is trained from full sequences ( ϕ ( s 1: T ) = e βr T ( s 1: T ) ), but applied to partial sequences without modification, r T ( s 1: t ) . This clearly does not correspond to a twist or soft value function V ∗ t ( s 1: t ) = 1 β log ∑ s t +1: T p 0 ( s t +1: T | s 1: t ) e βr T ( s 1: T ) = r T ( s 1: t ) .

## D.2. Blockwise Greedy Decoding in Mudgal et al. (2023)

As an alternative use of the twist functions at inference time and a generalization of best-ofK decoding to partial sequences, Mudgal et al. (2023) also consider a 'blockwise' decoding scheme using the learned twist functions ψ θ t . In particular, for K partial completions of length M (from a prefix s 1: t ), sampled from the base model, s ( k ) t +1: t + M ∼ p 0 ( s t +1: t + M | s 1: t ) , Mudgal et al. (2023) propose to choose

<!-- formula-not-decoded -->

and proceed with sampling K further continuations with prefix s ω 1: t + M until the next resampling step or an end-of-string token is reached. The arg max selection strategy may seem natural from the unregularized RL (as β →∞ ) or expected future reward perspective in App. D.1, but does not yield samples from σ ( s 1: T ) with the corresponding optimal twists.

Our SMC framework instead would advocate probabilistic resampling based on the approximate twist functions using the ( c - or M -step) importance weights in Sec. 3 in order to match the desired target distribution.

Finally, Khanov et al. (2024) also consider arg max decoding of next tokens using the unmodified r T ( s 1: t ) described above.

## E. Proposal Learning Methods

We next describe methods for learning variational policies or proposals q ξ ( s 1: T ) = ∏ T t =1 q ξ t ( s t | s 1: t -1 ) parameterized by ξ , which can be used for SMC sampling with intermediate targets π θ t ( s 1: t ) and learned twists ψ θ t ( s 1: t ) or V θ t ( s 1: t ) parameterized by θ . Alternatively, such proposals may be used directly in the IWAE bounds on log Z σ , which rely on simple importance sampling over full sequences as in Sec. 2.1 and do not require the definition of intermediate targets π t .

In App. E.3, we provide a detailed description of the DPG policy gradient method, which can be interpreted as a maximum likelihood objective for a sequential energy-based model. To distinguish this EBM approach from our CTL method for twist learning, we emphasize issues which can arise from naive use of a proposal-learning objective to define intermediate twisting targets for SMC in App. E.3.1.

## E.1. Path Consistency Learning for Controlled Generation

Guo et al. (2021) consider learning Q -values to obtain a fine-tuned variational policy which can be directly used as a sampling distribution for controlled generation. Building on the path consistency learning (PCL) loss in Nachum et al. (2017) and App. C.1.2, Guo et al. (2021) consider parameterizing the proposal using Q ξ t ( s t , s 1: t -1 ) ,

<!-- formula-not-decoded -->

where V Q ξ ( s 1: t -1 ) = 1 β log ∑ s t p 0 ( s t | s 1: t -1 ) e βQ ξ t enforces normalization.

Guo et al. (2021) define the targets using ¯ Q ξ t ( s t , s 1: t -1 ) , a slowly-updated target network based on Q ξ t . Using the implied form of the soft value ¯ V ( s 1: t -1 ) := 1 β log ∑ s t p 0 ( s t | s 1: t -1 ) e β ¯ Q ξ t ( s t , s 1: t -1 ) , the single-step PCL loss becomes

<!-- formula-not-decoded -->

where sg ( · ) indicates stop gradient. Building on the interpretation in App. C.1, we view ¯ V t ( s 1: t ) and ¯ V t -1 ( s 1: t -1 ) as the twisting targets, with a learned proposal parameterized by Q ξ t as in Eq. (60) (or App. B.4). While the loss in Eq. (61) is similar in practice to the soft Q-learning loss in App. C.1.1, we emphasize that the latter is motivated from the SMC perspective with the twisting targets as the primary object of interest and flexibility in the choice of proposal. By contrast, Guo et al. (2021) are interested in learning a proposal policy and do not consider, for example, resampling according to ¯ V t .

Guo et al. (2021); Nachum et al. (2017) also consider 'multi-step' PCL losses (Eq. (multi-step PCL)) which use observed reward during rollouts of length λ to limit reliance on estimated intermediate values ¯ V t ( s 1: t ) . The objective in Hu et al. (2023) also corresponds to a PCL objective.

## E.2. Policy Gradient Methods

Traditional RLHF pipelines use a policy gradient method such as PPO to optimize the objective in Eq. (42),

<!-- formula-not-decoded -->

where r T ( s 1: T ) = 1 β log ϕ ( s 1: T ) corresponds to our final twist. As in Eq. (46), the gap in this optimization is the modeseeking KL divergence D KL ( q ξ ( s 1: T ) ∥ ∥ σ ( s 1: T ) ) .

Notably, this objective does not make use of exact target samples from σ ( s 1: T ) when they are available. Further, the mode-seeking behavior has been shown to reduce diversity of fine-tuned models (Stiennon et al., 2020; Go et al., 2023). To combat this, Go et al. (2023) derive policy gradient methods to optimize arbitrary f -divergences D f ( q ξ ( s 1: T ) ∥ ∥ σ ( s 1: T ) ) between the learned variational policy q ξ and target σ .

## E.3. Policy Gradient with Mass-Covering / Maximum Likelihood KL Divergence

We focus on the case of minimizing the mass-covering KL divergence D KL ( σ ( s 1: T ) ∥ ∥ q ξ ( s 1: T ) ) to train q ξ , which constitutes the distributional policy gradients (DPG) method for language model finetuning (Parshakova et al., 2019; Khalifa et al., 2020; Korbak et al., 2022a; Go et al., 2023) and has been used to learn SMC proposals in state-space models in (Gu et al., 2015).

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We recognize the importance weights w ( s 1: T ) = ˜ σ ( s 1: T ) q ξ ( s 1: T ) from Eq. (3). Go et al. (2023) consider estimating Eq. (63) using a moving average estimate of the partition function ˆ Z σ

<!-- formula-not-decoded -->

Alternatively, the expectation may thus be estimated using SIS with the variational policy q ξ ( s 1: T ) . Using self-normalized importance sampling (SNIS) to estimate Eq. (63) as in Eq. (5) corresponds to ˆ Z σ = ∑ K j =1 w ( s ( k ) 1: T ) , with

<!-- formula-not-decoded -->

We use this gradient for DPG proposal learning in the main text experiments, although we use the parameterization described in Eq. (DPG) below.

DPG as Sequential Maximum Likelihood Objective We now show Eq. (64) is equivalent to a sequential maximum likelihood EBM objective. Consider minimizing the KL divergence,

<!-- formula-not-decoded -->

While this is reminscent of the twist-induced proposal in Prop. 3.3, we emphasize distinctions between energy-based learning of the proposal (DPG) versus energy-based learning of twist functions (CTL) in App. E.3.1. The gradient of Eq. (EBM proposal learning) becomes

<!-- formula-not-decoded -->

Starting from Eq. (64), we now seek to recover Eq. (66). Using Eq. (65), we can write

<!-- formula-not-decoded -->

Substituting into Eq. (64), we recover

<!-- formula-not-decoded -->

which is an SNIS estimate of the maximum likelihood EBM gradient in Eq. (66), as desired. Note that the expectation over q ξ t ( s t | s ( k ) 1: t -1 ) can be calculated exactly.

Comparison with CTL Objective The gradient in Eq. (DPG) above appears similar to our CTL objective and gradient in Sec. 4.1. However, the DPG loss in Eq. (EBM proposal learning) is a single (joint) KL divergence over the entire sequence, whereas CTL optimizes T separate KL divergences for each intermediate marginal.

For the DPG gradient in Eq. (66), negative sampling is performed using a 'positive' prefix s ( k ) 1: t -1 ∼ σ ( s 1: t -1 ) and an exact 'negative' sample from the one-step-ahead q ξ t ( s t | s ( k ) 1: t -1 ) (Eq. (65), which we have assumed to be tractable). In practice, we obtain the prefixes using the truncation of exact samples or approximate positive sampling with the final target weights as in Eq. (DPG). By contrast, the CTL gradient in Eq. (22) involves approximate negative sampling under each π t ( s 1: t ) .

## E.3.1. NAIVE USE OF PROPOSAL LEARNING TO DEFINE TWISTED SMC TARGETS

While we have shown in Prop. 3.3 how one-step proposals { q π t ( s t | s 1: t -1 ) } T t =1 can be induced from a given set of twist functions { ψ t ( s 1: t ) } T t =1 or target distributions { π t ( s 1: t ) } T t =1 , we now emphasize that moving the other direction (inducing intermediate twisting targets from a proposal learning scheme parameterized by { ψ ξ t } T t =1 ) does not yield the correct intermediate targets for resampling (App. A.1), even at optimality in the proposal learning objective.

We focus our arguments on learning with the EBM maximum likelihood objective in Eq. (EBM proposal learning) as an example. The proposal energies ψ ξ t ( s 1: t ) appear to play a role analogous to the twist function ψ t ( s 1: t ) in the one-step proposal induced from twist targets { π t } T t =1 in Sec. 3.

However, we proceed to show in Prop. E.2 that naive use of ψ ξ t to define twisting targets using 11

̸

<!-- formula-not-decoded -->

need not lead to an SMC procedure for which π ξ t ( s 1: t ) = σ ( s 1: t ) , even if q ξ t ( s t | s 1: t -1 ) = σ ( s t | s 1: t -1 ) for all t . We thus argue that ψ ξ t learned using Eq. (EBM proposal learning) should not be used as target twists in Eq. (67), since they do not yield the optimal interemdiate target distributions at optimality (App. A.1).

We begin by showing a simple lemma for the one-step conditionals in Eq. (EBM proposal learning).

Lemma E.1. Any twist induced proposal q ξ t ( s t | s 1: t -1 ) (induced by ψ ξ t ( s 1: t ) ) is invariant to rescaling ψ ξ t ( s 1: t ) by an arbitrary constant c ( s 1: t -1 ) with respect to s 1: t -1 ,

<!-- formula-not-decoded -->

Proof.

<!-- formula-not-decoded -->

11 We assume no intermediate potentials in this section, as in the main text.

Proposition E.2. There exist { ψ ξ ∗ t } T t =1 such that (i) q ξ ∗ t ( s t | s 1: t -1 ) = σ ( s t | s 1: t -1 ) and (ii) the SMC targets { π ξ ∗ t ( s 1: t ) } T t =1 induced by { ψ ξ ∗ t } T t =1 via Eq. (67) are different from σ ( s 1: t ) .

Proof. To satisfy condition (i) of the current proposition, we define

<!-- formula-not-decoded -->

̸

̸

which for all τ , yields optimal proposals: ( i ) q ξ ∗ ( s τ | s 1: τ -1 ) = σ ( s τ | s 1: τ -1 ) ∝ p 0 ( s τ | s 1: τ -1 ) ψ ξ ∗ τ ( s 1: τ ) via Lemma E.1. However, it is clear that c ( s 1: t -1 ) = 1 can break the necessary condition for optimality of SMC sampling that π t ( s 1: t ) = σ ( s 1: t ) (Prop. A.4). In particular, consider

̸

<!-- formula-not-decoded -->

̸

for c ( s 1: t -1 ) = 1 , which introduces an additional factor which depends on s 1: t . Thus, the twist target π ξ ∗ t ( s 1: t ) induced from ψ ξ ∗ t ( s 1: t ) in Eq. (69) is not equal to the desired marginal σ ( s 1: t ) , despite the fact that all proposals are optimal.

We indeed observed experimentally that resampling based on Eq. (67) after training using Eq. (EBM proposal learning) could lead to worse SMC log Z σ bounds than simply calculating the SIS or IWAE bound with ∏ T t =1 q ξ t ( s t | s 1: t -1 ) .

Optimality in CTL Objective implies Optimal Twisted SMC In contrast to Prop. E.2, note that the global optimum of our CTL objective min ∑ T t =1 D KL ( σ ( s 1: t ) ∥ ∥ ∥ π ψ t ( s 1: t ) ) (which occurs for the optimal twists { ψ ∗ t } T -1 t =1 in Prop. 3.2), results in both the twist-induced proposal q π ∗ t ( s t | s 1: t -1 ) = σ ( s t | s 1: t -1 ) and the twisting targets π ∗ t ( s 1: t ) = σ ( s 1: t ) satisfying the necessary and sufficient conditions for optimality outlined in App. A.1 Prop. A.3.

E.3.2. SMC WITH NORMALIZED TARGETS INDUCED BY LEARNED PROPOSAL LEADS TO UNIFORM WEIGHTS The issue in Prop. E.2 arises from the degree of freedom c ( s 1: t -1 ) in the normalization constant of the one-step proposal. To avoid this, we can instead define normalized twisted intermediate targets using

̸

<!-- formula-not-decoded -->

where Z ξ t ( s 1: t -1 ) arises from the proposal q ξ t ( s t | s 1: t -1 ) := 1 Z ξ t ( s 1: t -1 ) p 0 ( s t | s 1: t -1 ) ψ ξ t ( s 1: t ) learned according to Eq. (EBM proposal learning).

̸

Crucially, ˜ π ξ t in Eq. (71) are automatically normalized for t = T , as the product of normalized proposals. In this case, SMC resampling with q ξ or the twist-induced proposal yields uniform resampling weights,

<!-- formula-not-decoded -->

Although we were able to construct well-behaved intermediate twisting targets from a proposal-learning scheme q ξ t ( s t | s 1: t -1 ) ∝ p 0 ( s t | s 1: t -1 ) ψ ξ t ( s 1: t ) , Eq. (72) shows that this does not lead to meaningful intermediate SMC resampling . In other words, for t &lt; T , the marginal distributions of SMC samples s k 1: t with this scheme are simply q ξ ( s 1: t ) , the same as we would obtain with no resampling (SIS/IWAE).

## F. Bidirectional SMC

In this section, we recall the extended state-space probabilistic interpretation of SMC from (Maddison et al., 2017; Andrieu et al., 2010). The idea is to define an unnormalized target distribution σ SMC ( S ) and normalized proposal q SMC ( S ) over an extended state space S containing all random variables relevant to SMC sampling and importance weighting with K sequences of length T . Defining ˜ σ SMC ( S ) such that its normalization constant matches Z σ , we can use simple importance sampling (SIS) in this extended state space to show that K -sequence SMC sampling yields an unbiased estimator of Z σ , for example Z σ = E q SMC ( S ) [ ˜ σ SMC ( S ) q SMC ( S ) ] (as in Eq. (8)). Our end goal is to use this probabilistic interpretation to derive the lower and upper bounds on log Z σ in Prop. 5.1, following Brekelmans et al. (2022) App. A.

We define the extended state space proposal and target distributions below, noting that our bounds will require sampling from normalized σ SMC ( S ) or q SMC ( S ) , and evaluating ˜ σ SMC ( S ) and q SMC ( S ) . We summarize the algorithm for sampling σ SMC ( S ) in Alg. 2, using concatenation notation for simplicity instead of the probabilistic interpretation using index histories in the text.

Single-Sequence Target and Proposal We construct our importance sampling bounds with the goal of estimating the (log) partition function and sampling from a target distribution σ ( s 1: T ) = ˜ σ ( s 1: T ) / Z σ . We leverage a sequence of intermediate target distributions, { π t ( s 1: t ) = 1 Z t ˜ π t ( s 1: t ) } T t =1 over partial sequences, with the final target π T ( s 1: T ) = σ ( s 1: T ) and Z T = Z σ . We assume ˜ π 0 ( s 0 ) = 1 for all prompts with Z 0 = 1 . Finally, our bounds and sampling procedures also depend on a given set of proposal distribution { q ( s t | s 1: t -1 ) } T t =1 , as in Sec. 2.2.

Extended State Space Random Variables Consider an extended state space S containing KT tokens { s k t } T,K t =1 ,k =1 with s k t ∈ V and KT indexing random variables { ω k t } T,K t =1 ,k =1 with ω k t ∈ [1 , K ] , to represent the results of resampling (Eq. (7)),

<!-- formula-not-decoded -->

For ease of notation (and similarly to Maddison et al. (2017); Andrieu et al. (2010)), we call attention to our use of recursive backtracking index operations to collect sequences { s 1: t } based on the results of resampling { ω k t } . We use lists of index histories to construct sequences of tokens, with two recursive definitions of histories. Letting + indicate appending of lists,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For example, the history h k t -1 will be used to construct prefix sequences s h k t -1 1: t -1 (i.e. lists of tokens) for sampling a next token s k t . We denote sequences of tokens with the index history in the superscript and also expand the definition for clarity,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

(Sequence Notations)

In the second line, we define s ¯ h k t 1: t as a sequence of length t which concatenates the prefix s h k t -1 1: t with next token s k t . The notation s ¯ h k t 1: t represents partial sequences before resampling. By contrast, we will use the notation s h k t 1: t in the first line of Eq. (Sequence Notations) to refer to sequences after resampling.

Consider the sequence s ¯ h i t 1: t in a particular index i ∈ [1 , K ] before resampling. Resampling at time t may result in choosing ω k t = i for some k . Using the first line, we see that s h k t 1: t = s h ω k t t -1 1: t -1 +[ s ω k t t ] = s h i t -1 1: t -1 +[ s i t ] for those indices such that ω k t = i . Indeed, this matches the definition of s ¯ h i t 1: t = s h i t -1 1: t -1 +[ s i t ] in the second line (before resampling). Thus, the indexing notation in Eq. (Sequence Notations) reflects resampling or cloning of sequences s ¯ h i t 1: t into the indices such that ω k t = i , which yields prefixes s h k t 1: t for the next step of sampling ( t +1 ) in each index k ∈ [1 , K ] .

<!-- image -->

<!-- image -->

Figure 4: Graphical Models for extended statespace proposal and target distributions which result in the bidirectional SMC bounds. We show density evaluation in the proposal and target for a fixed set of { s k t , ω k t } 3 , 2 k =1 ,t =1 . We let the size of the circles reflect the (hypothetical) importance weights of sequences s ¯ h k t 1: t and ω k t reflect the (hypothetical) results of resampling with these weights. In ( b ) , we assume fixed j T +1 = j 3 = 1 as in the text, with ω 1 2 = 2 .

Algorithm 2 (Twisted) SMC Target Sampling ( σ SMC ) (blue indicates changes from SMC proposal algorithm; s 1: T is an exact posterior sample)

<!-- formula-not-decoded -->

Extended State Space Proposal Distribution Sampling from the extended state space proposal corresponds to the procedure described in Sec. 2.2 and Alg. 1, which we write as 12

<!-- formula-not-decoded -->

12 Note that h k t , s ¯ h k t 1: t , and s h k t 1: t are deterministically constructed from { s k t , ω k t } T,K t =1 ,k =1 during sampling, and simply track the quantities to be calculated when evaluating densities.

To recount the description above, note that the next token s i t in index i is sampled from the proposal, conditioned on the prefix s h i t -1 1: t -1 . We concatenate these tokens s ¯ h i t 1: t = s h i t -1 1: t -1 +[ s i t ] ( Eq. (Sequence Notations)) and calculate importance weights. We perform resampling in each index k according to q ( ω k t | s 1: K 1: t ) , or SNIS with the calculated weights (as in Eq. (7)). Finally, after resampling, we clone the sequence in the chosen index ω k t into index k and proceed to sample s k t +1 with an prefix defined by the indices h k t = h ω k t t -1 +[ ω k t ] .

Worked Example: To make this more concrete, we provide a worked example of the procedure in Fig. 4 (a). At step t = 1 , we resample the token s k =2 t =1 twice (for indices k = 1 , 3 ), with ω 1 1 = ω 3 1 = 2 (and in index 2 , set ω 2 1 = 3 to sample s 3 1 ). We record the prefix history as, for example, h 1 1 = h 3 1 =[ ω 1 1 ] =[2] , which corresponds to s h 1 1 1 = s 2 1 .

At step 2 in (a), we proceed to sample s 1 2 ∼ q ( s 2 | s h 1 1 1 = [ s 2 1 ]) (and similarly s 3 2 ∼ q ( s 2 | s h 3 1 1 = [ s 2 1 ]) ), whereas s 2 2 ∼ q ( s 2 | s h 1 1 1 = [ s 3 1 ]) . We next evaluate the importance weights over three concatenated sequences: s ¯ h 1 1 1 = [ s 2 1 ] + [ s 1 2 ] , s ¯ h 2 1 1 =[ s 3 1 ] + [ s 2 2 ] , and s ¯ h 3 1 1 =[ s 2 1 ] + [ s 3 2 ] , emphasizing that s k 2 is the final token in each index. Shown in the red circles, we proceed to resample ω 1 2 = 2 , ω 2 2 = 3 , and ω 3 2 = 2 at step t = 2 .

Finally, we need to backtrack to obtain the history of the indices for the sequence to be cloned in resampling. Namely, for index 1 where ω k =1 t =2 = 2 , we concatenate h ω 1 2 1 +[ ω 1 2 ] = h 2 1 +[2] = [3 , 2] =: h 1 2 (i.e. the history for time 2, index 1). This list of indices specifies the prefix s h 1 2 1:2 =[ s 3 1 , s 2 2 ] at step t = 3 , index k = 1 . Similar reasoning applies for other indices.

Extended State Space Target We are finally ready to specify the extended state space target distribution. The crucial difference is to identify a single sequence s h 1 T 1: T of length T (the choice of index 1 is arbitrary). This sequence s h 1 T 1: T will be evaluated under the unnormalized target distribution ˜ π T ( s 1: T ) = ˜ σ ( s 1: T ) or exactly sampled from the target s h 1 T 1: T ∼ σ ( s 1: T ) in the extended state space target distribution.

In particular, we begin by sampling a full sequence of indices { j t } T t =1 uniformly at random Pr ( j 1 , j 2 , ...j T ) = (1 /K ) T . Setting ω 1 T = j T , we let ω j t t -1 = j t -1 for all t . This implies the following,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

To show these identities, note that ω j t t -1 = j t -1 and Eq. (Index Notation) imply h j t t -1 = h ω j t t -1 t -2 +[ ω j t t -1 ] = h j t -1 t -2 +[ j t -1 ] = ¯ h j t -1 t -1 , which matches Eq. (76). Applying this recursion again yields h j t t -1 = h j t -2 t -3 +[ j t -2 , j t -1 ] ... = [ j 1 , j 2 , ...j t -1 ] . Taken together, these notations allow us to interleave a true target sample in particular indices { j t } , guaranteeing that at least one target samples appears at each step.

The extended state space target distribution differs from q SMC in its handling of this sequence, which identified as s h 1 T 1: T with prefixes s h j t t -1 1: t -1 using Eq. (75). Noting that sampling { j t } T t =1 amounts to specifying a particular set of ω k t as in Eq. (75)-(76),

̸

<!-- formula-not-decoded -->

̸

Note, the normalization constant of ˜ σ SMC ( S ) is equal to Z σ since only ˜ π T ( s 1: T ) = ˜ σ ( s 1: T ) is unnormalized.

̸

To describe ancestral sampling from Eq. (SMC Extended Target), we first sample { j t } T t =1 uniformly as above, and place an exact target sequence in indices s h 1 T 1: T (or, equivalently, sequentially sample s j t t ∼ π t ( s t | s h j t t -1 1: t -1 ) . At each step, the remaining K -1 indices k = j t are sampled from the proposal. For resampling, we fix index j t to hold the exact sample and resample the remaining K -1 indices. Note that the resampling weights q ( ω k t ∣ ∣ s 1: K 1: t ) in Eq. (74) include the exact sample, which may be cloned additional times into indices other than j t if its importance weights are high. The procedure above simply ensures that at least one exact sequence is sampled. See Alg. 2 for the pseudocode of the algorithm.

Note that Maddison et al. (2017, Alg. 2) presents a different SMC extended state space target distribution than ours. In their work, j 1 = 1 and they sample j 2: T +1 , while in ours j T +1 = 1 and we sample j 1: T . However, both targets result in the same log partition function bounds.

Worked Example: In Fig. 2 (c), we use blue circles and arrows to highlight the exact-sample indices h 1 T =[ j 1 , j 2 ] =[3 , 2] and the target sequence s h 1 T 1: T =[ s 3 1 , s 2 2 ] . Using the recursion ω j t t -1 = j t -1 with j T +1 = j 3 = 1 fi xed, we may also express h 1 T =[ j 1 , j 2 ] =[3 , 2] = [ ω 2 1 , ω 1 2 ] . At step 2, note the target sequence is sampled/evaluated an additional time in index 3.

Importance Weights in the Extended State Space Assume we are given a fixed set of { s k t , ω k t } T,K t =1 ,k =1 , which may be sampled from either σ SMC ( S ) or q SMC ( S ) . We proceed to show that the unnormalized importance weights in the extended state space simplify as follows.

Lemma F.1. For the extended state space target ˜ σ SMC and proposal q SMC above, the simple importance weights in the extended state space become

<!-- formula-not-decoded -->

which can be used to obtain unbiased Z σ estimators (Eq. (8) ) or bounds on log Z σ (Prop. 5.1, with proof below).

Proof. To evaluate the importance weights (with the goal of estimating Z σ ), we consider

̸

<!-- formula-not-decoded -->

̸

where in (1) , note that terms in the denominator cancel except for the indices [0 , j 1 , ...j T ] = h 1 T . Recalling that ω j t +1 t = j t from Eq. (76), we expand the resampling weights q ( j t | s 1: K 1: t ) for the sequence indexed by s j t t , s h j t t -1 1: t -1 , and s ¯ h j t t 1: t -1 ,

<!-- formula-not-decoded -->

Finally, we obtain a telescoping cancellation of ˜ π t terms using the indexing identities in Eq. (75)-(76). In particular, since ¯ h j t t = h j t +1 t and ¯ h j t -1 t -1 = h j t t -1 with ¯ h j T T = h 1 T , we can simplify the terms in Eq. (80) as

<!-- formula-not-decoded -->

using the assumption that ˜ π 0 ( · ) = 1 . Simplifying from Eq. (80), the final unnormalized importance weights become

<!-- formula-not-decoded -->

as desired, where we abbreviate the importance weights as w t ( s k 1: t ) for simplicity of notation. Note that we also obtain an unbiased estimate of the partition function via

<!-- formula-not-decoded -->

Proposition 5.1. (Bidirectional SMC Bounds) The log partition function log Z σ of a target distribution σ ( s 1: T ) can be lower and upper bounded by

<!-- formula-not-decoded -->

The gap in the lower bound is D KL ( q SMC ( S ) ∥ σ SMC ( S )) , and the gap in the upper bound is D KL ( σ SMC ( S ) ∥ q SMC ( S )) .

Proof. The proof follows directly from Brekelmans et al. (2022) App. A, where it is shown that for σ ext ( S ) , q ext ( S ) such that Z σ = E q ext ( S ) [ ˜ σ ext ( S ) q ext ( S ) ] , we can construct lower and upper bounds on log Z σ

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the gap in the lower and upper bounds are D KL ( q ext ( S ) ∥ σ ext ( S )) and D KL ( σ ext ( S ) ∥ q ext ( S )) , respectively.

Substituting our SMC probabilistic interpretation in Eq. (SMC Extended Proposal) and Eq. (SMC Extended Target), along with the importance weights in Lemma F.1, into Eq. (83) yields the desired bounds in Eq. (24).

IWAE as a Special Case of our SMC Probabilistic Interpretation Note that we recover IWAE (or SIS over K samples) from SMC with no intermediate resampling. In particular, this corresponds to ω k t = k for all t &lt; T , with importance weighting from resampling occurring at the final step ∏ K k =1 q ( ω k T | s 1: K 1: T ) . This yields the 1 /K average inside the log in the IWAE bounds (i.e., SMC with only one resampling step at t = T ). While the importance weights are crucial to construct the bound, note that 'resampling' is not necessary at the final step and we may return all K samples along with their weights.

Viewing IWAE as a special case of our SMC probabilistic interpretation is complementary to the interpretations in Domke &amp;Sheldon (2018); Brekelmans et al. (2022) and also provides upper bounds (Sobolev &amp; Vetrov, 2019).

## G. Additional Experiment Details

## G.1. Common Details Across Experiments

For all experiments, we use the Adam optimizer with β 1 , β 2 = { 0 . 9 , 0 . 999 } . We use custom implementations of SMC. For PPO, we use the HuggingFace TRL PPO Trainer ( https://github.com/huggingface/trl/blob/main/trl/trainer/ ppo trainer.py ), modified slightly to accomodate our custom twist parameterizations, as described below. For other methods, we use Optax (Flax) and custom loss functions. We use HuggingFace models ( https://huggingface.co/ models ) for the base p 0 models and build custom layers on top of those.

For the twist ψ θ t ( s 1: t ) , we always parameterize log ψ θ t ( s 1: t ) for numerical stability. We choose random normal initializations centered at mean 0, with low variance, 13 such that log ψ θ t ( s 1: t ) ≈ 0 , ψ θ t ( s 1: t ) ≈ 1 at the beginning of training, which means the initial sequences generated by the twist-induced proposal approximately come from the base model p 0 . All methods are initialized using the same random seeds, and thus start from the same parameter values. See App. G.2 for additional discussion of choices for the twist parameterization.

13 We specifically use a form of Xavier initialization, taking the variance as 2 n inputs + n outputs .

For methods that directly learn a proposal (DPG and PPO), we could directly finetune a language model that outputs q ( s 1: t ) . However, in order to ensure consistency in terms of model capacity and ease of learning compared to our twisted proposals, we instead have these proposal learning methods output a modifier log ψ θ t ( s 1: t ) which is added to the base model log probability log p 0 ( s 1: t ) . Note that using random normal initializations centered at mean 0 with low variance, this scheme results in initial q samples coming approximately from p 0 .

For methods that can make use of exact posterior samples, when we have access to them (Sec. 7.2.3, App. H.3), we use them. This is straightforward for methods like DPG, SIXO, and our CTL (unless we have only a single sample, as we discuss for infilling in App. G.4 ). For our RL twist learning, we found the best empirical performance training on a combination of q and exact σ samples when they were available (as opposed to just q otherwise), and use those results. Similarly, for FUDGE, when exact σ samples are available, we use them together with p 0 samples.

It is not straightforward to compare PPO versus other methods, because of the inner loop in PPO that repeats several clipped gradient steps on a given set of samples. This means that, for a constant number of samples, PPO makes more gradient updates than other methods, while for a constant number of gradient updates, PPO sees fewer samples. Ultimately we decided to normalize based on the number of samples seen; we consider each outer step (including a full PPO inner loop, in our experiments, 4 gradient steps) as a single 'gradient update.' We make this choice since sampling is the main bottleneck in terms of computational cost, and the number of inner PPO steps is a hyperparameter which we did not tune.

All of our experiments were run on a single GPU, usually on an NVIDIA A40 with 48G memory. All experiments took no longer than 9 wall-clock hours to run for a single learning method, with infilling (Sec. 7.2.3) experiments taking longest; most other experiments took no longer than 4 hours.

## G.2. Choices of Twist Parameterization

The choice of parameterization for the twist log ψ θ t ( s 1: t ) is a design decision, independent of our overall framework. While one could keep an entirely separate model for each log ψ θ t ( s 1: t ) , this is likely to be memory-inefficient and learn slowly. Instead, we use a shared parameterization across s 1: t , in the same way that the base language model uses a single architecture to output probability distributions over tokens at each time step t . We lay out parameterization choices we considered below.

## G.2.1. LINEAR HEAD

The simplest choice is to replace the linear head of the base language model with a new linear head, keep the base model fixed, and only train the linear head. This parameterization incurs very little additional computation cost compared to just using the base language model. However, we found this to be capacity constrained in our experiments, achieving worse KL divergences than other parameterizations.

## G.2.2. MLP HEAD

Instead of a linear head, we consider a 3-layer fully connected neural network (MLP) with ReLU non-linearities as a head on top of the base language model. The base model is still kept fixed; only the MLP head is trained. This incurs more computational cost than a linear head (App. G.2.1), but the additional cost is still small relative to the cost of a forward pass through the base transformer model. We found this to generally perform well in our experiments, so we use it for the toxicity threshold experiment in Sec. 7.1 and sentiment in Sec. 7.2.2.

## G.2.3. SEPARATE TRANSFORMER FOR THE TWIST

We can also consider an entirely separate transformer that outputs only the twist value. That is, we copy the base model, and repurpose it to output a twist value log ψ θ t ( s 1: t ) instead of logits for next-token probabilities. We then train the entire network end-to-end. This is significantly more computationally costly than the former approaches, and does not always do better than just an MLP head (App. G.2.2), so we generally do not recommend using this. Still, we found it to perform well in toxicity classification in Sec. 7.2.1, so we use it there.

## G.2.4. SEPARATE TRANSFORMER FOR THE TWIST, WITH MLP HEAD

This is similar to App. G.2.3, except we also replace the final linear head with a MLP head as in App. G.2.2. The model outputs log ψ θ t ( s 1: t ) and is trained end-to-end. This is the most computationally costly approach outlined here, and is unnecessary for most of our settings. However, in infilling with 15 generated tokens (Sec. 7.2.3) we found this parameterization to perform materially better than all others, particularly with DPG (App. E.3), so we use it for all infilling experiments.

With both this parameterization and App. G.2.3, we increase computation time by a factor of around 2 on the forward pass, and significantly increase memory and time usage on the backwards pass during training (though sampling is still the main time bottleneck). Whether this parameterization is worth the potential gain in performance depends on the desired use case. We emphasize that our overall framework is independent of the choice of parameterization.

## G.3. Comments on Our Choices of Experiment Settings

Our settings and evaluation metrics in Sec. 7 are chosen to highlight our scientific findings. In particular, the toxicity threshold experiment in Sec. 7.1 demonstrates the improvement of SMC over SIS with the base model with CTL learned twists. In order to highlight this distinction, we have chosen a setting where it is extremely difficult to draw samples satisfying the threshold using the base model p 0 (see SIS/IWAE LB line in Fig. 3).

However, twist-learning in the toxicity threshold setting presents challenges. For approximate positive sampling and a thresholded target, all importance weights will be 0 if none of our K samples meet the threshold. As noted above, sampling from p 0 , or the SMC/twisted proposal for ψ θ t ( s 1: t ) ≈ 1 at initialization, is extremely unlikely to draw samples meeting the threshold (i.e., within the support of the target) in the setting of Sec. 7.1. As a result, initial iterations of twist learning receive no learning signal until a thresholded positive sample is drawn from the base model.

To avoid this difficulty for baselines comparisons in Sec. 7.2, we instead focused on settings with ϕ ( s 1: T ) given by probabilities. Nevertheless, we note that there are no fundamental differences between the settings considered in Sec. 7.1 and Sec. 7.2. Thus, we may also evaluate single-sample D KL ( σ ∥ q ) and D KL ( q ∥ σ ) in the setting of Sec. 7.1, or plot log Z σ bounds as a function of K in for the settings in Sec. 7.2.

## G.4. Experiment-Specific Details

Details for SIS and SMC Comparison (Sec. 7.1) We generate 10 output tokens, and train twists using Sec. 4.1 with approximate positive sampling as discussed in Sec. 4.1.2.

Note that using σ ( s 1: T ) ∝ p 0 ( s 1: T ) I [ s 1: T ∈ C ] where C := { s 1: T | r ( s 1: T ) ≤ η } directly runs into numerical issues for calculating log Z σ when s 1: T / ∈ C and I [ s 1: T ∈ C ] = 0 . We instead use ϵ + I [ s 1: T ∈ C ] everywhere instead of I [ s 1: T ∈ C ] , where ϵ = 10 -16 . In Fig. 3, this yields a SIS/IWAE log Z σ LB ≈ -36 when no samples are drawn that fall in the set C .

We use an MLP head to parameterize the twist, as in App. G.2.2, with 768 hidden units per layer, matching the TinyStories model's embedding dimension. We use a batch size (number of SMC particles/samples) of 1000, with a learning rate of 0.0001, and train using CTL for a total of 5000 gradient updates. We did not tune hyperparameters because we found this setting to work well, and we are not comparing across different learning methods.

For each point on each line on Fig. 3, we run SIS or SMC 20 times, each with a different randomly selected true posterior sample for the upper bounds. The line shows the average value across these 20 runs, while the shaded area shows 95% confidence intervals. See also App. G.1 for details common across experiments.

Details for Toxicity (Sec. 7.2.1) We generate 20 output tokens. We parameterize the twist with a separate network as in App. G.2.3. We use a batch size (number of SMC particles/samples) of 100, and train for a total of 2048 gradient updates. For each learning method, we used a coarse grid search over learning rates between 0.000001 and 0.001, using the best one found, which was usually 0.00003 or 0.0001. We run each learning method over 5 different random seeds, reporting the average KL divergence and 95% confidence intervals over these 5 seeds.

For each KL divergence evaluation, we first get sandwich bounds on log Z σ as laid out in Sec. 5, using the learned twists for the twisted proposal with K = 500 samples. We find SIS/IWAE and SMC bounds to be similarly tight, so use SIS/IWAE for simplicity. We do this 4 times, providing 4 upper bound estimates and 4 lower bound estimates, and take the average midpoint as the log Z σ estimate for each experiment. We then take the median (across all learning methods and seeds) of these estimates, and use that as our estimate of log Z σ . This is then used as a common value for the KL divergence across all methods and seeds, which controls for possible noise in log Z σ bounds and ensures a fair comparison across methods. We generally have tight bounds (upper bound ≈ lower bound), which suggest our log Z σ estimates are generally accurate, but note that any inaccuracies in estimating log Z σ would only affect the absolute values of the KL divergences, not the relative differences among different learning methods.

We estimate expectations in Eq. (23) with 2000 samples from q and 2000 exact posterior samples for σ . With 2000 samples, our estimates have 95% confidence intervals generally between 0.05 and 0.10, suggesting that our estimates of expectations are unlikely to be off by more than 0.10. The exact posterior samples were collected offline; such a large number of samples takes several hours to collect, and in practical settings, we would likely only be able to collect a much smaller number of samples. All our methods still apply with fewer exact posterior samples, but the variance in estimates will be higher. See also App. G.1 for details common across experiments.

Details for Sentiment (Sec. 7.2.2) We generate 10 output tokens. We parameterize the twist using an MLP head (App. G.2.2), with 1024 hidden units per layer, matching the GPT2Medium model's embedding dimension. Other details are the same as for toxicity above. Collecting exact posterior samples is less time consuming in this case (less than an hour). See App. G.1 for common experimental details.

Details for Infilling (Sec. 7.2.3) We parameterize the twist using a separate transformer with an MLP head (App. G.2.4), with 768 hidden units per layer (matching the TinyStories model's embedding dimension). We make the following adjustments to the forward pass of the language model for the conditional twist setting. Instead of taking in only s 1: T , the model takes in both s 1: T and s T +1: T + c and passes each separately through the body (everything except the head). Thus, s T +1: T + c can be seen as a second prompt. For s T +1: T + c , we take the embeddings produced after the last conditioning token s T + c has been processed, broadcast it across time steps 1 : T , and pass that as additional input to the MLP head (concatenated with embeddings for s 1: T at each t ∈ 1 ...T ). This allows the MLP head to produce different output depending on the conditioning tokens.

Since we are in the conditional target distribution setting (Sec. 3.3), with o T = s T +1: T + c , to compare across learning methods using a single quantity, we estimate E o T [ D KL ( q o T ∥ σ o T )] := E o T [ D KL ( q ( s 1: T | o T ) ∥ σ ( s 1: T | o T ))] and E o T [ D KL ( σ o T ∥ q o T )] := E o T [ D KL ( σ ( s 1: T | o T ) ∥ q ( s 1: T | o T ))] where E o T [ · ] := E p 0 ( s T +1: T + c ) [ · ] for infilling. Note that,

<!-- formula-not-decoded -->

where for a fixed o T , E q ( s 1: T | o T ) [ log q ( s 1: T | o T ) p 0 ( s 1: T ) ϕ ( s 1: T ,o T ) ] and E σ ( s 1: T | o T ) [ log p 0 ( s 1: T ) ϕ ( s 1: T ,o T ) q ( s 1: T | o T ) ] may be evaluated as before, similar to the unconditional setting. In particular, for our experiments, we use 1-sample estimates of these expectations, as we have a single exact sample from σ ( s 1: T | o T ) by the BDMC trick (Sec. 3.3), and we choose to draw a single sample from the conditional proposal q ( s 1: T | o T ) . We average this over 2000 o T ∼ p 0 ( s T +1: T + c ) , approximating the outer expectation, giving us a 2000-sample estimate of 1-sample estimates for the first term in the right hand side of both equations above. With 2000 samples, our estimates have 95% confidence intervals generally between 0.20 and 0.30.

Note that E o T [log Z σ ( o T )] is independent of the learning method or proposal q , unlike the first term we discussed above. Thus, in order to save computation and provide us with a more accurate estimate of E o T [log Z σ ( o T )] , we estimate this term only once. Specifically, we consider only the learning method with the lowest KL divergence (DPG), and use SIS/IWAE bounds. For each o T , we estimate log Z σ ( o T ) with K = 500 samples, which gives us relatively tight sandwich bounds, again taking the midpoint as our estimate. We average this over 1000 o T ∼ p 0 ( s T +1: T + c ) , giving us a 1000-sample estimate of E o T [log Z σ ( o T )] , where each log Z σ ( o T ) is itself estimated via 500 samples.

For negative sampling with contrastive twist learning (CTL) in this setting, we need at least 2 negative samples per set of conditioning tokens o T = s T +1: T + c to perform SIS reweighting; this is in contrast with other twist learning methods which can generate a single negative sample per o T . For the positive sample, we can use our single exact sample directly, or we can run the SMC upper bound sampling procedure ('Sampling from σ SMC for SMC Upper Bounds' section in Sec. 5.2) generate more approximate σ samples using the given exact sample. We find the latter to generally perform slightly better than the former, so adopt that for our infilling experiments.

We use a fixed batch size of 100 across all methods for training twists. To clarify the meaning of this batch size, for methods other than CTL, we have 100 draws of exact σ samples, each for a different set of conditioning tokens o T = s T +1: T + c , so we train over 100 different o T at a time using 1 negative sample per o T . For CTL, since we need at least 2 negative samples per o T , we split the batch size of 100 across the number of different o T and the number of negative samples per o T , as an additional hyperparameter. We use 25 o T with 4 negative samples per o T for the experiments in Sec. 7.2.3 and

Table 6: Qualitative Results - Reviews Very Likely to be of a Particular Rating

| Class (Rating)   | Most Text Generated Using Twisted SMC                                                                                |
|------------------|----------------------------------------------------------------------------------------------------------------------|
| 1-star           | 'I bought this sucker for my wife to use on her python that she sent me last year. It was terrible!'                 |
| 2-star           | 'I bought this throat raiser for combating dental caries. I didn't really like it. I didn't like'                    |
| 3-star           | 'I bought this a few months back, and I enjoyed it every time I held it. I'm giving 3 stars'                         |
| 4-star           | 'I bought this product a few months ago and have really enjoyed it. Only reason I gave it 4 stars is because'        |
| 5-star           | 'I bought this phone recently, and I've been loving it! Gorgeous design, outstanding battery life, fantastic camera' |

Table 7: Qualitative Results - Infilling Examples

| Proposal   | Prompt ( s 0 )                | Generated Tokens ( s 1: T )                                                | Conditioning Tokens ( s T +1: T + c )        |
|------------|-------------------------------|----------------------------------------------------------------------------|----------------------------------------------|
| DPG        | Once upon a time, there was a | little girl named Mia. She had a big heart. Mia loved to help              | others and make them feel safe. Mia liked to |
| SIXO       | Once upon a time, there was a | girl named Mia. Mia was very kind and compassionate. She always helped her | others and make them feel safe. Mia liked to |
| CTL        | Once upon a time, there was a | girl named Mia. She had a thin, pink dress. Mia liked to                   | others and make them feel safe. Mia liked to |

10 o T with 10 negative samples per o T for the experiments in App. H.2. Controlling for batch size in this way is arguably disadvantageous for CTL compared to other learning methods, as it learns on a smaller number of o T , but this controls for memory requirements, and we feel is more fair than controlling for the number of o T seen but allowing more negative samples for CTL relative to other methods. We train for a total of 5500 gradient updates. For each method, we used a coarse grid search over learning rates between 0.000001 and 0.001, using the best one found, which was usually 0.0001 or 0.00003. We run each learning method over 5 different random seeds, reporting the average KL divergence and 95% confidence intervals over these 5 seeds. See also App. G.1 for details common across experiments.

## H. Additional Experimental Results

## H.1. Qualitative Results

Toxicity Controlled Generation when No Exact Posterior Samples are Available In Sec. 7.2.1 we targeted σ ( s 1: T ) ∝ p 0 ( s 1: T ) e β log p ( a | s 1: T ) with β = 1 . We can also target β &gt; 1 ; higher β produces a more peaked distribution of text that is more likely to be of class a . However, for β = 1 we can no longer generate exact posterior samples and thus cannot upper bound log Z σ . Our twist learning (Sec. 4.1) with approximate positive sampling (Sec. 4.1.2) can learn meaningful twists in this setting, which we illustrate with a qualitative example of a story (200 tokens upper limit) and β = 10 :

̸

'Once upon a time, there was a little girl named Lily. She had a big thumb that she liked to suck on. One day, Lily went to the park to play

with her friends. She was having so much fun until her thumb got stuck in her shoe. She tried to pull it out, but it hurt too much. Lily started to cry and her friends tried to help her, but they couldn't get her thumb out either. She was scared and didn't know what to do. Her friends tried to help her, but they couldn't get it out either. Sadly, Lily had to go to the hospital and get a big bandage on her thumb. She couldn't play with her friends anymore. From that day on, Lily never went to the park again. '

The story is coherent and follows the general style of the TinyStories base model, while having a high probability ( ≈ 88%) of being toxic according to the toxicity classifier, likely due to the presence of negative words such as 'suck', 'hurt', 'cry', and 'scared'. This supports the ability of our methods to control outputs based on the chosen posterior distribution.

Sentiment Controlled Generation when No Exact Posterior Samples are Available As above, we also consider σ ( s 1: T ) ∝ p 0 ( s 1: T ) e β log p ( a | s 1: T ) , where β &gt; 1 , except now p ( a | s 1: T ) is based on the sentiment classifier in Sec. 7.2.2. In Table 6 we provide qualitative examples showing 20 tokens produced with twisted SMC with 500 particles, for β = 100 , using twists trained with Sec. 4.1. These illustrate our framework's ability to learn reviews that embody each rating class. 14

Infilling In Table 7 we compare qualitative results on an example set of conditioning tokens for DPG, SIXO, and CTL (in that order, to reflect increasing KL divergence). The qualitative results correlate with the quantitative measures of KL divergence; the lowest KL divergence (DPG) corresponds to infilled tokens that respect grammar and the topic. SIXO, which has higher KL divergence, fails to respect grammar. CTL generates incorrect grammar and is less on-topic, corresponding to the highest KL divergence among these methods.

14 The results are slightly incoherent; this is a result of the base GPT2-Medium model often being incoherent. Qualitatively, we find that these generations are more coherent than the uncontrolled ones from p 0 .

Table 8: KL Divergences (averaged over conditioning tokens drawn from the base model) for Infilling Experiments (Sec. 7.2.3 ) with 2 Output Tokens and 1 Conditioning Token

| Proposal q o T   | Twist Learning   | E o T [ D KL ( q o T ∥ σ o T )]   | E o T [ D KL ( σ o T ∥ q o T )]   |
|------------------|------------------|-----------------------------------|-----------------------------------|
| Twisted          | Contrastive      | 0 . 47 ± 0 . 10                   | 0 . 25 ± 0 . 01                   |
| Twisted          | RL               | 0 . 42 ± 0 . 10                   | 0 . 15 ± 0 . 01                   |
| Twisted          | SIXO             | 0 . 47 ± 0 . 11                   | 0 . 25 ± 0 . 02                   |
| Twisted          | FUDGE            | 2 . 62 ± 0 . 33                   | 0 . 90 ± 0 . 02                   |
| DPG              | -                | 0 . 16 ± 0 . 07                   | 0 . 14 ± 0 . 01                   |
| PPO              | -                | 0 . 52 ± 0 . 04                   | 1 . 09 ± 0 . 34                   |

## H.2. Infilling with Fewer Tokens

We consider the same setting as Sec. 7.2.3 but only generating 2 tokens, conditioned on 1 token. We show KL divergence evaluations in Table 8. Our evaluation reveals interesting differences among learning methods, even in this easier setting where most methods achieve low KL divergence in both directions. DPG and RL learns best, while FUDGE learns notably slower. PPO suffers on D KL ( σ ∥ q ) , though this may be unsurprising since PPO does not make use of exact σ samples.

## H.3. Approximate vs. Exact Posterior Sampling

In our toxicity and sentiment experiments, we train using approximate σ samples to reflect the more common real-world setting where the amount of exact samples needed for training are not available. However, here we run an additional ablation experiment for insight into the effect of positive versus approximate sampling. We use rejection sampling (Sec. 4.1.2) to generate exact posterior samples for training. This is much slower than generating approximate samples, so is not a practical strategy for training; we investigate this solely for understanding.

We provide a comparison of KL divergences (evaluated the same way as in the main paper) when training using exact versus approximate σ samples for a selection of methods that performed well in our previous experiments and are able to make use of σ samples. Toxicity (Sec. 7.2.1) results are in Table 9 and sentiment (Sec. 7.2.2) results are in Table 10. The first two columns of KL divergences are for exact σ samples. The next two are for training on the same number of samples, but using approximate positive sampling (Sec. 4.1.2). Overall, for a constant number of samples, having exact σ samples improves performance for most methods. Note however that there is an additional time cost required for rejection sampling to generate exact samples, so the exact σ training requires significantly more wall-clock time for any given number of samples.

We also plot the single-sample KL divergence in both directions as a function of training time for exact vs. approximate sampling, on toxicity and sentiment experiments, in Fig. 5. The approximate sampling results match those in the main paper (with different colors). The exact σ sample results cut off earlier because the time cost required for rejection sampling reduces the number of gradient updates that can be made for a given amount of wall-clock time.

Table 9: KL Div. for Toxicity Experiments (Sec. 7.2.1), comparing exact σ samples versus approximate positive sampling.

|            |                        | Exact σ Samples   | Exact σ Samples   | Same # of Approx. σ Samples   | Same # of Approx. σ Samples   |
|------------|------------------------|-------------------|-------------------|-------------------------------|-------------------------------|
| Proposal q | Type of Twist Learning | D KL ( q ∥ σ )    | D KL ( σ ∥ q )    | D KL ( q ∥ σ )                | D KL ( σ ∥ q )                |
| Twisted    | Contrastive            | 2 . 54 ± 0 . 02   | 2 . 68 ± 0 . 09   | 2 . 99 ± 0 . 18               | 3 . 22 ± 0 . 09               |
| Twisted    | RL                     | 3 . 23 ± 0 . 10   | 3 . 24 ± 0 . 04   | 3 . 48 ± 0 . 15               | 3 . 49 ± 0 . 13               |
| Twisted    | SIXO                   | 2 . 37 ± 0 . 06   | 2 . 52 ± 0 . 05   | 2 . 70 ± 0 . 17               | 3 . 05 ± 0 . 22               |
| DPG        | -                      | 1 . 51 ± 0 . 01   | 1 . 50 ± 0 . 01   | 2 . 35 ± 0 . 15               | 2 . 48 ± 0 . 10               |

Table 10: KL Div. for Sentiment Experiments (Sec. 7.2.2), comparing exact σ samples versus approximate positive sampling.

|                  |                        | Exact σ Samples   | Exact σ Samples   | Same # of Approx. σ Samples   | Same # of Approx. σ Samples   |
|------------------|------------------------|-------------------|-------------------|-------------------------------|-------------------------------|
| Proposal q ( s ) | Type of Twist Learning | D KL ( q ∥ σ )    | D KL ( σ ∥ q )    | D KL ( q ∥ σ )                | D KL ( σ ∥ q )                |
| Twisted          | Contrastive            | 0 . 71 ± 0 . 02   | 0 . 64 ± 0 . 02   | 0 . 70 ± 0 . 02               | 0 . 60 ± 0 . 01               |
| Twisted          | RL                     | 1 . 28 ± 0 . 05   | 0 . 94 ± 0 . 02   | 2 . 09 ± 0 . 08               | 1 . 76 ± 0 . 07               |
| Twisted          | SIXO                   | 0 . 68 ± 0 . 02   | 0 . 60 ± 0 . 01   | 0 . 86 ± 0 . 02               | 0 . 68 ± 0 . 01               |
| DPG              | -                      | 0 . 70 ± 0 . 02   | 0 . 58 ± 0 . 01   | 0 . 89 ± 0 . 03               | 0 . 69 ± 0 . 00               |

Figure 5: Training comparison for Exact versus Approximate σ (positive) sampling, as described in App. H.3. Having access to exact target samples makes learning lead to lower KL divergences in a more reliable manner.

<!-- image -->