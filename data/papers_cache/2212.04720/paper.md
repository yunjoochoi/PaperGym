## Multi-Task Off-Policy Learning from Bandit Feedback

| Joey Hong   | Branislav Kveton   | Sumeet Katariya   | Manzil Zaheer   | Mohammad Ghavamzadeh   |
|-------------|--------------------|-------------------|-----------------|------------------------|
| UC Berkeley | Amazon             | Amazon            | Deepmind        | Google                 |

## Abstract

Many practical applications, such as recommender systems and learning to rank, involve solving multiple similar tasks. One example is learning of recommendation policies for users with similar movie preferences, where the users may still rank the individual movies slightly differently. Such tasks can be organized in a hierarchy, where similar tasks are related through a shared structure. In this work, we formulate this problem as a contextual off-policy optimization in a hierarchical graphical model from logged bandit feedback. To solve the problem, we propose a hierarchical off-policy optimization algorithm ( HierOPO ), which estimates the parameters of the hierarchical model and then acts pessimistically with respect to them. We instantiate HierOPO in linear Gaussian models, for which we also provide an efficient implementation and analysis. We prove per-task bounds on the suboptimality of the learned policies, which show a clear improvement over not using the hierarchical model. We also evaluate the policies empirically. Our theoretical and empirical results show a clear advantage of using the hierarchy over solving each task independently.

## 1 Introduction

Many interactive systems (search, online advertising, and recommender systems) can be modeled as a contextual bandit (Li et al., 2010a; Chu et al., 2011), where an agent, or policy , observes a context , takes one of K possible actions , and receives a stochastic reward for the action. In many applications, it is prohibitively expensively to learn policies online by contextual bandit algorithms, because exploration has a major impact on user experience. How- ever, offline data collected by a previously deployed policy are often available. Offline, or off-policy , optimization using such logged data is a practical way of learning policies without costly online interactions (Dudik et al., 2014; Swaminathan and Joachims, 2015).

Because we cannot explore beyond the logged dataset, it is critical to design learning algorithms that use the data in the most efficient way. One way of achieving this is by leveraging the structure of the problem. As an example, in bandit algorithms, we could achieve higher statistical efficiency by using the form of the reward distribution (Garivier and Cappe, 2011), prior distribution over model parameters (Thompson, 1933; Agrawal and Goyal, 2012; Chapelle and Li, 2012; Russo et al., 2018), or by conditioning on feature vectors (Dani et al., 2008; Abbasi-Yadkori et al., 2011; Agrawal and Goyal, 2013). In this work, we consider a natural structure where we design policies for multiple similar tasks, where the tasks are related through a hierarchical Bayesian model (Gelman et al., 2013; Kveton et al., 2021; Hong et al., 2022b). Each task is parameterized by a task parameter sampled i.i.d. from a distribution parameterized by a hyper-parameter . These parameters are unknown and relate the tasks, in the sense that data from one task can help with learning a policy for another task.

Although the tasks are similar, they are sufficiently different to require different polices, and we address this multitask off-policy learning problem in this work. To solve the problem, we propose an algorithm called hierarchical offpolicy optimization ( HierOPO ). Because off-policy algorithms must reason about counterfactual rewards of actions that do not appear in the logged dataset, a common approach is to learn pessimistic, or lower confidence bound (LCB) , estimates of the mean rewards and act according to them (Buckman et al., 2020; Jin et al., 2021). HierOPO is an instance of this approach where high-probability LCBs are estimated using a hierarchical model.

Our paper makes the following contributions. First, we discuss how hierarchy can improve statistical efficiency, which motivates our algorithm HierOPO . The key idea in HierOPO is to factorize the computation of LCBs by separately considering the uncertainty of the hyper-parameter and the conditional uncertainty of task parameters. Second, we consider a specific hierarchical model, a linear Gaussian model, where we obtain closed forms for the LCBs that can be computed efficiently. Third, we derive Bayesian suboptimality bounds for the policies learned by HierOPO and show that they improve upon off-policy approaches that do not use the hierarchy. To the best of our knowledge, we are the first to consider Bayesian bounds in the off-policy setting. Finally, we evaluate HierOPO on synthetic problems and an application to a multi-user recommendation system.

## 2 Setting

Notation. Random variables are capitalized, except for Greek letters like θ . For any positive integer n , we define [ n ] = { 1 , . . . , n } . The indicator function is denoted by 1 {·} . The i -th entry of vector v is v i . If the vector is already indexed, such as v j , we write v j,i . For any matrix M ∈ R d × d , the maximum and minimum eigenvalues are λ 1 ( M ) and λ d ( M ) , respectively.

We consider a learning agent that interacts with a set of contextual bandit instances. In each interaction, the agent observes a context x ∈ X , takes an action a from an action set A of size K , and then observes a stochastic reward Y ∈ R . The contexts are sampled from the context distribution P x . Conditioned on context and action, the reward is sampled from the reward distribution P ( · | x, a ; θ ) , where θ ∈ Θ is a parameter of the bandit instance, which is shared by all contexts and actions . We assume that the rewards are σ 2 -sub-Gaussian and denote by r ( x, a ; θ ) = E Y ∼ P ( ·| x,a ; θ ) [ Y ] the mean reward of action a in context x under parameter θ .

In this work, the learning agent simultaneously solves m contextual bandit instances, which we denote by S = [ m ] and refer to as tasks . Therefore, we call our problem a multi-task contextual bandit (Azar et al., 2013; Deshmukh et al., 2017; Cella et al., 2020; Kveton et al., 2021; Moradipari et al., 2021). Each task s ∈ S is parameterized by a task parameter θ s, ∗ ∈ Θ , which is sampled i.i.d. from a task prior distribution θ s, ∗ ∼ P ( · | µ ∗ ) . The task prior is parameterized by an unknown hyper-parameter µ ∗ , which is sampled from a hyper-prior Q . That one is known to the agent and represents its prior knowledge about µ ∗ . In a recommender system, each task could be an individual user, the task parameter could encode user's preferences, and the hyper-parameter could encode the average preferences of a cluster of similar users. We use this setup in our experiments in Section 7. A similar setup was studied previously in the online setting by Hong et al. (2022c).

Unlike prior works in multi-task bandits, we aim to solve this problem offline. Let Π = { π : X → A} be the set of stationary deterministic policies . For any policy π and context x , we denote by π ( x ) the action suggested by π in context x . In our multi-task bandit setting, each task has its own parameter, and thus we may need a different policy to solve it. Therefore, we consider the set of taskconditioned policies π ∈ Π S = { ( π s ) s ∈S : π s ∈ Π } , where π s is the policy for task s . Note that we consider deterministic policies solely to simplify notation, and that our results extend to stochastic policies by accounting for an additional expectation over actions.

Figure 1: A graphical model of our multi-task contextual bandit setting.

<!-- image -->

A logged dataset of past interactions is an input to offpolicy evaluation and optimization. In our setting, we have access to a dataset D = { ( S t , X t , A t , Y t ) } t ∈ [ n ] of n observations, where S t ∈ S is a task, X t ∼ P x is a context, A t = π 0 ,S t ( X t ) is an action, and Y t ∼ P ( · | X t , A t ; θ S t , ∗ ) is a reward in observation t . Here π 0 ∈ Π S is a logging policy , some task-conditioned policy that is used to collect D . A graphical model of our setting is shown in Figure 1. Unlike many works in off-policy learning, we do not require that π 0 is known (Dudik et al., 2014; Swaminathan and Joachims, 2015).

The value of policy π s ∈ Π in task s ∈ S with parameter θ s, ∗ is defined as

<!-- formula-not-decoded -->

where the randomness is only over context X ∼ P x . The optimal policy π s, ∗ is defined as

<!-- formula-not-decoded -->

and the suboptimality of policy π s is

<!-- formula-not-decoded -->

We study the Bayesian setting, where the logged dataset D provides additional information about the parameter θ s, ∗ . In particular, let ˆ P s ( θ ) = P ( θ s, ∗ = θ | D ) be the posterior distribution of θ s, ∗ in task s given D . Then, by definition, θ s, ∗ | D ∼ ˆ P s . Our goal is to learn a policy, for any given task s , that is comparable to likely π s, ∗ | D . We formalize this objective using a high-probability bound. For a fi xed confidence level δ ∈ (0 , 1) , we want to learn a policy ˆ π s ∈ Π that minimizes ε in

<!-- formula-not-decoded -->

where ε is a function of δ , the environment parameters, D , and ˆ π s . Note that π s, ∗ is random because it is a function of random θ s, ∗ | D ∼ ˆ P s .

The Bayesian view allows us to derive error bounds with two new properties. First, the error ε decreases with a more informative prior on θ s, ∗ . Second, the bounds capture the structure of our hierarchical problem and show that it helps. Although our objective and analysis style are novel, they are motivated by Bayes regret bounds in bandits (Russo and Van Roy, 2014; Lu and Van Roy, 2019; Kveton et al., 2021; Hong et al., 2022c), which have similar properties that allow them to improve upon their frequentist counterparts (Abbasi-Yadkori et al., 2011; Agrawal and Goyal, 2013).

## 3 Algorithm

Prior works in off-policy bandit and reinforcement learning often design pessimistic lower confidence bounds and then act on them (Jin et al., 2021). We follow the same design principle. For any task s , context x , and action a , we want to estimate a LCB satisfying L s ( x, a ) ≤ r ( x, a ; θ s, ∗ ) , with a high probability for θ s, ∗ | D . We seek the LCBs of the form L s ( x, a ) = ˆ r s ( x, a ) -c s ( x, a ) , where

<!-- formula-not-decoded -->

are the estimated mean reward and its confidence interval width, and α &gt; 0 is a tunable parameter.

An important case of contextual models are those with linear rewards (Abbasi-Yadkori et al., 2011; Jin et al., 2021). In our paper, we assume that r ( x, a ; θ s, ∗ ) = φ ( x, a ) glyph[latticetop] θ s, ∗ for each task s , where θ s, ∗ is the task parameter and φ : X × A → R d is some feature extractor . Under this assumption, we may write (2) using the posterior mean and covariance of θ s, ∗ as

<!-- formula-not-decoded -->

The above is desirable because it separates the posterior of the task parameter from context.

The rest of this section is organized as follows. In Section 3.1, we derive the mean reward estimate and its confidence interval width for a general two-level hierarchical model. We also propose a general hierarchical off-policy optimization ( HierOPO ) in this model. In Section 3.2, we instantiate this model as a linear Gaussian model. We discuss alternative algorithm designs in Section 3.3.

## 3.1 Hierarchical Pessimism

For any task s , the mean E [ θ s, ∗ | D ] in (3) can be estimated hierarchically as follows. Let D s be the subset of dataset D corresponding to task s . By the law of total expectation,

<!-- formula-not-decoded -->

| Algorithm 1 HierOPO : Hierarchical off-policy optimiza- tion.   | Algorithm 1 HierOPO : Hierarchical off-policy optimiza- tion.   |
|-----------------------------------------------------------------|-----------------------------------------------------------------|
| 1:                                                              | Input: Dataset D                                                |
| 2:                                                              | for s ∈ S ,x ∈ X do                                             |
| 3:                                                              | for a ∈ A do                                                    |
| 4:                                                              | Compute ˆ r s ( x,a ) and c s ( x,a ) (Section 3.1)             |
| 5:                                                              | L s ( x,a ) ← ˆ r s ( x,a ) - c s ( x,a )                       |
| 6:                                                              | ˆ π s ( x ) ← argmax a ∈A L s ( x,a )                           |
| 7:                                                              | Output: ˆ π ← (ˆ π s ) s ∈S                                     |

The second equality holds since conditioning on µ ∗ makes θ s, ∗ independent of D\D s , as can be seen in Figure 1. The above decomposition is motivated by the observation that estimating each E [ θ s, ∗ | µ ∗ , D s ] is an easier problem than E [ θ s, ∗ | D ] , since D s is from a single task s . The information sharing between the tasks is still captured by µ ∗ , which has to be learned from the entire logged dataset D .

Similarly, the covariance cov [ θ s, ∗ | D ] in (3) can be decomposed using the law of total covariance,

<!-- formula-not-decoded -->

Again, the second equality holds since conditioning on µ ∗ makes θ s, ∗ independent of D\D s . Note that (5) comprises two interpretable terms. The first captures the uncertainty of θ s, ∗ conditioned on µ ∗ , whereas the second captures the uncertainty in µ ∗ . Such decompositions decouple the two sources of uncertainty in our hierarchical model, and are powerful tools for estimating uncertainty in structured models (Hong et al., 2022a).

Now we plug (4) and (5) into (3), and get

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

With this in mind, we propose a general algorithm for hierarchical off-policy optimization, which we call HierOPO and report its pseudo-code in Algorithm 1.

## 3.2 Hierarchical Gaussian Pessimism

The computation of (4) and (5) requires integrating out the hyper-parameter µ ∗ and task parameter θ s, ∗ . This is generally impossible in a closed form, although many powerful approximations exist (Doucet et al., 2001). In this section, we consider the case where the hyper-prior and task prior distributions are Gaussian. In this case, HierOPO can be implemented exactly and efficiently. The later analysis of HierOPO (Section 5) is also under this assumption.

Specifically, we consider a linear Gaussian model where the known hyper-prior is Q = N ( µ q , Σ q ) for some PSD matrix Σ q and the task prior is P ( · | µ ∗ ) = N ( µ ∗ , Σ 0 ) for some known PSD Σ 0 . The reward distribution of action a in context x is N ( φ ( x, a ) glyph[latticetop] θ s, ∗ , σ 2 ) , where φ is a feature extractor and σ &gt; 0 is a known reward noise. This implies that the mean reward is linear in features.

To derive (4) and (5), we start with understanding posterior distributions of θ s, ∗ and µ ∗ . Specifically, since conditioning in Gaussian graphical models preserves Gaussianity, we have that θ s, ∗ | µ ∗ , D s ∼ N (˜ µ s , ˜ Σ s ) for some ˜ µ s and ˜ Σ s . From the structure of our model (Figure 1), we further note that this is a standard posterior of a linear model with a Gaussian prior N ( µ ∗ , Σ 0 ) , and thus,

<!-- formula-not-decoded -->

where the statistics

<!-- formula-not-decoded -->

are computed using the subset D s of the logged dataset D .

The posterior of the hyper-parameter µ ∗ | D , known as the hyper-posterior, also has a closed-form N (¯ µ, ¯ Σ) (Section 4.2 of Hong et al. 2022c), where

<!-- formula-not-decoded -->

It is helpful to view (7) as a multivariate Gaussian posterior where each task is a single observation. The observation of task s is the least-squares estimate of θ s, ∗ from task s , G -1 s B s , and its covariance is Σ 0 + G -1 s . The tasks with many observations affect the value of ¯ µ more, because their G -1 s approaches a zero matrix. In this case, Σ 0 + G -1 s → Σ 0 . This uncertainty cannot be reduced because even θ s, ∗ is a noisy observation of µ ∗ with covariance Σ 0 .

To complete our derivations, we only need to substitute (6) and (7) into (4) and (5). The posterior mean of θ s, ∗ is

<!-- formula-not-decoded -->

where we simply combine (6) and (7). Similarly, the posterior covariance of θ s, ∗ requires computing

<!-- formula-not-decoded -->

Finally, the estimated mean reward and its confidence interval width are given by

<!-- formula-not-decoded -->

where ˆ Σ s = ˜ Σ s + ˜ Σ s Σ -1 0 ¯ ΣΣ -1 0 ˜ Σ s . Note that the posterior covariance ˆ Σ s can be computed tractably, and exhibits the following desirable properties. First, the uncertainty over the hyper-parameter only shows up in the second term in ¯ Σ . In addition, since ˜ Σ s appears in both terms, both terms become smaller with more observations from task s .

## 3.3 Alternative Designs

Anatural question to ask is what is the benefit of leveraging hierarchy in obtaining pessimistic reward estimates. To answer this question, we compare HierOPO in Section 3.2 to two alternative algorithms. The first one is unrealistic and assumes that µ ∗ is known. We call it OracleOPO . In this case, the posterior mean reward and its confidence interval width are given by

<!-- formula-not-decoded -->

This improves upon (8) in two aspects. First, the estimate ¯ µ of µ ∗ is replaced with the actual µ ∗ . Second, the confidence interval width is provably smaller because

<!-- formula-not-decoded -->

In the second algorithm, we consider what happens when we do not model the hierarchy, which we dub FlatOPO . In this case, we do not attempt to model µ ∗ and include its uncertainty in θ s, ∗ . To do so, the conditional uncertainty of θ s, ∗ , represented by Σ 0 , is replaced with its marginal uncertainty, represented by Σ q + Σ 0 . As a result, the posterior mean reward and its confidence interval width are

<!-- formula-not-decoded -->

where ˙ Σ s = ((Σ q + Σ 0 ) -1 + G s ) -1 . This is worse than (8) in two aspects. First, the prior mean µ q of µ ∗ is used instead of its estimate ¯ µ . Second, as the number of tasks m increases,

<!-- formula-not-decoded -->

since ¯ Σ in (7) approaches a zero matrix. Therefore, our approach should be more statistically efficient, which we prove formally in Section 5.

## 4 Single-Task Analysis

To illustrate our error bounds, we start with a contextual bandit parameterized by θ ∗ ∈ R d . The mean reward of action a ∈ A in context x ∈ X under parameter θ ∈ R d is r ( x, a ; θ ) = φ ( x, a ) glyph[latticetop] θ . We assume that θ ∗ ∼ N ( θ 0 , Σ 0 ) and that the reward noise is N (0 , σ 2 ) . Note that this is an analogous model to a single task in Section 3.2 where we drop indexing by s to simplify notation.

The logged dataset is D = { ( X t , A t , Y t ) } n t =1 , the LCB is L ( x, a ) = ˆ r ( x, a ) -c ( x, a ) , and we output a policy ˆ π ∈ Π defined as ˆ π ( x ) = arg max a ∈A L ( x, a ) . Following the same reasoning as in the derivation of (8), the estimated mean reward and its confidence interval width are

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Analogously to Section 2, the value of policy π ∈ Π under parameter θ ∗ is V ( π ; θ ∗ ) = E [ r ( X,π ( X ); θ ∗ ) | θ ∗ ] and the optimal policy is π ∗ = arg max π ∈ Π V ( π ; θ ∗ ) . For any fixed confidence level δ &gt; 0 , our goal is to learn a policy ˆ π ∈ Π that minimizes ε in

<!-- formula-not-decoded -->

We make the following assumptions in our analysis. First, we assume that the length of feature vectors is bounded.

Assumption 1. For any x ∈ X and a ∈ A , the feature vector satisfies ‖ φ ( x, a ) ‖ 2 ≤ 1 .

This assumption is without loss of generality and only simplifies presentation. Second, similarly to prior works (Swaminathan et al., 2017; Jin et al., 2021), we assume that the dataset D is 'well-explored'.

Assumption 2. Let

<!-- formula-not-decoded -->

Then there exists γ &gt; 0 such that G glyph[followsequal] γσ -2 nG ∗ holds for any θ ∗ .

The above assumption relates the logging policy π 0 , which defines the empirical precision G , to the optimal policy π ∗ , which defines the mean precision σ -2 nG ∗ . The assumption can be loosely interpreted as follows. As n increases, G → σ -2 n E [ φ ( X,π 0 ( X )) φ ( X,π 0 ( X )) glyph[latticetop] ] , and hence γ can be viewed as the maximum ratio between probabilities of taking actions by π ∗ and π 0 in any direction. In general, for a uniform logging policy, γ = Ω(1 /d ) when n is large. The assumption essentially allows us not to reason about the properties of G when n is small, which would require a concentration argument and is not essential to our result.

Note that the assumption is always satisfied by setting γ = 0 . However, this setting would negate the desired scaling with sample size n in our error bounds. Also note that the assumption can be weakened to be probabilistic over θ ∗ . We do not do this to simplify the exposition.

Now we state our main claim for the single-task setting.

Theorem 1. Fix dataset D and choose any γ such that Assumption 2 holds. Let ˆ π ( x ) = arg max a ∈A L ( x, a ) . Then for any δ ∈ (0 , 1) and

<!-- formula-not-decoded -->

the suboptimality of ˆ π ∈ Π in (9) is bounded for

<!-- formula-not-decoded -->

Proof. The claim is proved in Appendix A.1 in three steps. First, we establish that c ( x, a ) is a high-probability confidence interval width for α = √ 5 d log(1 /δ ) . Second, we show that the suboptimality of policy ˆ π can be bounded by E [ c ( X,π ∗ ( X )) | θ ∗ ] . Finally, we combine closed forms of c ( x, a ) with Assumption 2, and relate the statistics under the logging policy π 0 that define c ( x, a ) with the expectation under π ∗ .

## 5 Multi-Task Analysis

Now we study our multi-task setting, where the estimated mean reward and its confidence interval width are defined in (8). Similarly to Section 4, this analysis is Bayesian and we are concerned with the distribution of model parameters conditioned on D . Wefix the task and derive an error bound for a single s ∈ S . In Section 5.1, we discuss how to extend our bound to other performance metrics, such as the error over all tasks.

To derive the bound in (1), we make assumptions analogous to Section 4. First, and without loss of generality, we assume that the length of feature vectors is bounded (Assumption 1). Second, we assume that the dataset D is 'well-explored' for all tasks.

## Assumption 3. Let

<!-- formula-not-decoded -->

be the empirical precision associated with task s and n s = ∑ n t =1 1 { S t = s } be the number of interactions with that task. Let

<!-- formula-not-decoded -->

Then there exists γ &gt; 0 such that G s glyph[followsequal] γσ -2 n s G s, ∗ holds for any θ s, ∗ in any task s ∈ S .

This assumption is essentially Assumption 2 applied to all tasks. In general, for a uniform logging policy, γ = Ω(1 /d ) when n s is large for all s ∈ S . Therefore, we do not think that the assumption is particularly strong. If needed, the assumption could be weaken to be probabilistic, as discussed after Assumption 2.

We also consider an additional assumption that sharpens the bound in Theorem 2.

Assumption 4. For any x ∈ X and a ∈ A , the feature vector φ ( x, a ) has at most one non-zero entry. Moreover, both Σ q and Σ 0 are diagonal.

Note that Assumption 4 encompasses the multi-arm bandit case, where φ ( x, a ) ∈ R |X||A| and is an indicator vector for each context-action pair. Our main technical result is presented below.

Theorem 2. Fix dataset D and choose any γ such that Assumption 3 holds. Take ˆ π computed by HierOPO . Then for any δ ∈ (0 , 1) and

<!-- formula-not-decoded -->

the suboptimality of ˆ π s ∈ Π in (1) is bounded for

<!-- formula-not-decoded -->

Hyper-parameter term

Also, under Assumption 4,

Hyper-parameter term

<!-- image -->

.

Proof. The claim is proved in Appendix A.2, in the same three steps as Theorem 1. The only difference is in the definitions of c ( x, a ) and policies, and that we use Assumption 3 instead of Assumption 2. This highlights the generality of our proof techniques and shows that they could be applicable to other graphical model structures.

## 5.1 Discussion

Our main technical result, an error bound on the suboptimality of policies learned by HierOPO , is presented in Theorem 2. The bound is Bayesian, meaning that it is proved for the distribution of true model parameters conditioned on logged dataset D . The bound has two terms. The former captures the error in estimating the task parameter θ s, ∗ conditioned on known hyper-parameter µ ∗ and is analogous to Theorem 1. We call it the task term . The latter captures the error in estimating the hyper-parameter µ ∗ and we call it the hyper-parameter term .

The task term scales with all quantities of interest as expected. First, it is O ( d √ log(1 /δ )) , where d is the number of task parameters and δ is the probability that the bound fails. This dependence is standard in linear bandit analyses with an infinite number of contexts (Abbasi-Yadkori et al., 2011; Agrawal and Goyal, 2013; Abeille and Lazaric, 2017). Second, the task term decreases with the number of observations n s at the rate of O (1 / √ n s ) . Since λ d (Σ -1 0 ) can be viewed as the minimum number of prior pseudoobservations in any direction in R d , the task term decreases with a more informative prior. Finally, the task term decreases when the observation noise σ decreases, and the similarity of the logging and optimal policies γ increases (Assumption 3).

The hyper-parameter term mimics the task-term scaling at the hyper-parameter level. In particular, the minimum number of prior pseudo-observations in any direction in R d becomes λ d (Σ -1 q ) and each task becomes an observation, which is reflected by the sum over all tasks z . The hyperparameter term decreases as the number of observations n z in any task z increases, the maximum width of the task prior √ λ 1 (Σ 0 ) decreases, noise σ decreases, and the similarity between logging and optimal policies γ increases.

To show that HierOPO leverages the structure of our problem, we compare its error bound to two baselines from Section 3.3: OracleOPO and FlatOPO . OracleOPO is an oracle estimator that knows µ ∗ , meaning that it has more information than HierOPO . Its error is bounded in Theorem 1 and is always lower than that of HierOPO , as the error bound in Theorem 1 is essentially only the first term in Theorem 2. The second baseline, FlatOPO , does not know µ ∗ and treats each task estimation problem independently. This approach can be viewed as OracleOPO where the task covariance Σ 0 is replaced by Σ q +Σ 0 , to account for the additional uncertainty due to not knowing µ ∗ . The resulting error bound is

<!-- formula-not-decoded -->

and is always higher than the task term in Theorem 2. In addition, the hyper-parameter term in Theorem 2 approaches zero as the number of tasks increases, and thus HierOPO is provably better in this setting of our interest.

The error bound in Theorem 2 is proved for one fixed task s ∈ S . This decision was taken deliberately because other error bounds can be easily derived from this result. For instance, to get a bound for all tasks, we only need a union bound for the concentration of all θ s, ∗ . Thus the bound in Theorem 2 holds jointly for all s ∈ S with probability at least 1 -mδ . Moreover, the same bound would essentially hold for any new task sampled from the hyper-prior. The reason is that the estimated hyper-parameter distribution, which affects the hyper-parameter term in Theorem 2, separates all other tasks from the evaluated one.

## 6 Related Work

Off-policy optimization. In off-policy optimization, logged data collected by a deployed policy is used to learn better policies (Li et al., 2010b), and the agent does not interact with the environment directly. Off-policy learning can be achieved using model-free or model-based techniques. A popular model-free approach is empirical risk minimization with IPS-based estimators to account for the bias in logged data (Joachims et al., 2017; Bottou et al., 2013; Swaminathan and Joachims, 2015; Swaminathan et al., 2017). Model-based methods (Jeunen and Goethals, 2021) on the other hand learn a reward regression model for specific context-action pairs, which is then used to derive an optimal policy. Model-free methods tend to have a high variance while model-based methods tend to have a high bias unless explicitly corrected. Our approach is model based since we learn a hierarchical linear reward model.

Offline reinforcement learning. The principle of pessimism has been explored in offline reinforcement learning in several works (Buckman et al., 2020; Jin et al., 2021). In particular, Jin et al. (2021) show that pessimistic value iteration is minimax optimal in linear MDPs. The multitask offline setting studied in this work was also studied by Lazaric and Ghavamzadeh (2010). They propose an expectation-maximization algorithm but do not prove any error bounds. On the other hand, we consider a simpler setting of contextual bandits and derive error bounds that show improvemets due to using the multi-task structure.

Online learning. Off-policy methods learn from data collected by a different policy. In contrast, online algorithms learn from data they collect, and need to balance exploration with exploitation. Two popular exploration techniques are upper confidence bounds (UCBs) (Auer et al., 2002) and posterior sampling (Thompson, 1933), and they have been applied to linear reward models (Dani et al., 2008; Abbasi-Yadkori et al., 2011; Chu et al., 2011; Agrawal and Goyal, 2013). Bandit algorithms for hierarchical models have also been studied extensively Bastani et al. (2019); Kveton et al. (2021); Basu et al. (2021); Simchowitz et al. (2021); Wan et al. (2021); Hong et al. (2022c); Peleg et al. (2022); Wan et al. (2022). Perhaps surprisingly, all of these are based on posterior sampling. Our marginal posterior derivations in Section 3.2 can be used to derive their UCB counterparts.

## 7 Experiments

In this section, we empirically compare HierOPO to baselines OracleOPO and FlatOPO (Section 3.3). All algorithms are implemented exactly as described in Section 3 with α = 0 . 1 , which led to good performance in our initial experiments. Overall we aim to show that hierarchy can greatly improve the efficiency of off-policy algorithms.

## 7.1 Synthetic Multi-Task Bandit

We first experiment with a synthetic multi-task bandit defined as follows. We set dimension as d = 4 , number of actions as K = 5 , and each context-action pair is a random vector φ ( x, a ) ∈ [ -0 . 5 , 0 . 5] d . The reward distribution for task s is N ( φ ( x, a ) glyph[latticetop] θ s, ∗ , σ 2 ) with noise σ = 0 . 5 .

The hierarchical model is defined as follows. The hyperprior is N ( 0 , Σ q ) with Σ q = σ 2 q I d , the task covariance is Σ 0 = σ 2 0 I d , and the reward noise is σ = 0 . 5 . We choose σ q ∈ { 0 . 5 , 1 } and σ 0 = 0 . 5 . We expect more benefits of learning µ ∗ when σ q &gt; σ 0 , as the uncertainty of the hyperparameter is higher. The model parameters are generated as follows. At the beginning of each run, µ ∗ ∼ N ( 0 , Σ q ) . After that, each task parameter is sampled i.i.d. as θ s, ∗ ∼ N ( µ ∗ , Σ 0 ) . We initially set the number of tasks to m = 10 and the size of the logged dataset to n = 500 . The logged dataset D is generated as follows. For each interaction t ∈ [ n ] , we sample one of m tasks uniformly at random, take an action uniformly at random, and sample a reward from the reward distribution.

In our experiments, we vary either dataset size n or the number of tasks m while keeping the other fixed. In Figure 2, we show the mean and standard error of the suboptimality of each algorithm averaged over 30 random runs, where the model and dataset in each run are generated as described earlier. As expected, HierOPO outperforms FlatOPO and is close to OracleOPO . The improvement is greater when the uncertainty in the hyper-parameter σ q is higher. We also see that the gap is most noticeable in the limited data regime, where n is small or m is large, with only a small number of observations per task.

Figure 2: Evaluation of off-policy algorithms on the synthetic multi-task bandit problem. In the left and middle plots, we vary the dataset size n for small σ q = 0 . 5 and large σ q = 1 . 0 . In the right plot, we vary the number of tasks m .

<!-- image -->

Figure 3: Evaluation of off-policy algorithms on the multiuser movie recommendation problem in Section 7.2.

<!-- image -->

## 7.2 Multi-User Recommendation

Now we consider a multi-user recommendation application. We fit a multi-task contextual bandit from the MovieLens 1M dataset (Lam and Herlocker, 2016), with 1 million ratings from 6 040 users for 3 883 movies, as follows. First, we complete the sparse rating matrix M using alternating least squares (Salakhutdinov and Mnih, 2007) with rank d = 10 . This rank is high enough to yield a low prediction error, but small enough to avoid overfitting. The learned factorization is M = UV glyph[latticetop] . User i and movie j correspond to rows U i and V j , respectively, in the learned latent factors. Each task corresponds to some user i . In each round, context x consists of K = 10 movies chosen uniformly at random. The reward distribution for recommending movie j to user i is N ( V glyph[latticetop] j U i , σ 2 ) with σ = 0 . 759 estimated from data.

To estimate the hierarchical model in Section 3.2, we cluster the user latent factors. Specifically, we learn a Gaussian mixture model (GMM) for k = 7 from rows of U , where we choose the smallest k that still achieves low variance (Bishop, 2006). We estimate the hyper-prior parameters µ q and Σ q using the mean and covariance, respectively, of the cluster centers. Then we select the cluster with most users, and set µ ∗ and Σ 0 to its center and covariance estimated by the GMM. The tasks are the users in this same cluster, to ensure that all are related to one another through the hyperparameter. We wanted to stress that the GMM is only used to estimate parameters for the off-policy algorithms. The task parameters U i are generated by matrix factorization. This is to ensure that our setup is as realistic as possible.

We keep the number of tasks fixed at m = 100 and vary dataset size n . The tasks are users from the largest cluster, sampled uniformly at random. When generating the logged dataset, we sample one task uniformly at random, take a random action in it, and record its random reward. In Figure 3, we show the mean and standard error of the suboptimality of each algorithm averaged over 10 random runs, where each run consists of choosing m users, generating a dataset of size n , and running each algorithm on that dataset. We observe that HierOPO achieves good performance, close to OracleOPO , using much less data than FlatOPO . This clearly demonstrates the benefit of hierarchies for statistically-efficient off-policy learning. The hierarchies are beneficial even if they are estimated from data and not exactly known.

## 8 Conclusions

In this work, we propose hierarchical off-policy optimization ( HierOPO ), a general off-policy algorithm for solving similar contextual bandit tasks related through a hierarchy. Our algorithm leverages the hierarchical structure to learn tighter, and thus more sample efficient, lower confidence bounds and then optimizes a policy with respect to them. We prove Bayesian suboptimality bounds for our policies, which decrease as the hyper-prior and task prior widths decrease. Thus the bounds improve with more informative priors. Finally, we empirically demonstrate the effectiveness of modeling hierarchies.

To the best of our knowledge, our work is the first to propose a practical and analyzable algorithm for off-policy learning with hierarchical Bayesian models. Because of this, there are many possible future directions to improve the generality and applicability of our approach. First, some applications may require more complex graphical models than two-level hierarchies. Second, the logged dataset may not contain labels of tasks, if different tasks cannot be as easily distinguished as users; or fully cover all possible tasks that can appear online. Extending our approach to learning policies from such limited datasets is another important avenue for future work.

## References

- Y. Abbasi-Yadkori, D. Pal, and C. Szepesvari. Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems 24 , pages 23122320, 2011.
- M. Abeille and A. Lazaric. Linear Thompson sampling revisited. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics , 2017.
- S. Agrawal and N. Goyal. Analysis of Thompson sampling for the multi-armed bandit problem. In Proceeding of the 25th Annual Conference on Learning Theory , pages 39.1-39.26, 2012.
- S. Agrawal and N. Goyal. Thompson sampling for contextual bandits with linear payoffs. In Proceedings of the 30th International Conference on Machine Learning , pages 127-135, 2013.
- P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning , 47:235-256, 2002.
- M. G. Azar, A. Lazaric, and E. Brunskill. Sequential transfer in multi-armed bandit with finite set of models. In Advances in Neural Information Processing Systems 26 , pages 2220-2228, 2013.
- H. Bastani, D. Simchi-Levi, and R. Zhu. Meta dynamic pricing: Transfer learning across experiments. CoRR , abs/1902.10918, 2019. URL https://arxiv.org/ abs/1902.10918 .
- S. Basu, B. Kveton, M. Zaheer, and C. Szepesvari. No regrets for learning the prior in bandits. In Advances in Neural Information Processing Systems 34 , 2021.
- C. M. Bishop. Pattern Recognition and Machine Learning . Springer, New York, NY, 2006.
- L. Bottou, J. Peters, J. Qui˜ nonero-Candela, D. X. Charles, D. M. Chickering, E. Portugaly, D. Ray, P. Simard, and E. Snelson. Counterfactual reasoning and learning systems: The example of computational advertising. Journal of Machine Learning Research , 14(11), 2013.
- J. Buckman, C. Gelada, and M. G. Bellemare. The importance of pessimism in fixed-dataset policy optimization. arXiv preprint arXiv:2009.06799 , 2020.
- L. Cella, A. Lazaric, and M. Pontil. Meta-learning with stochastic linear bandits. In Proceedings of the 37th International Conference on Machine Learning , 2020.
- O. Chapelle and L. Li. An empirical evaluation of Thompson sampling. In Advances in Neural Information Processing Systems 24 , pages 2249-2257, 2012.
- W. Chu, L. Li, L. Reyzin, and R. Schapire. Contextual bandits with linear payoff functions. In Proceedings of the 14th International Conference on Artificial Intelligence and Statistics , pages 208-214, 2011.
- V. Dani, T. Hayes, and S. Kakade. Stochastic linear optimization under bandit feedback. In Proceedings of the 21st Annual Conference on Learning Theory , pages 355-366, 2008.
- A. A. Deshmukh, U. Dogan, and C. Scott. Multi-task learning for contextual bandits. In Advances in Neural Information Processing Systems 30 , pages 4848-4856, 2017.
- A. Doucet, N. de Freitas, and N. Gordon. Sequential Monte Carlo Methods in Practice . Springer, New York, NY, 2001.
- M. Dudik, D. Erhan, J. Langford, and L. Li. Doubly robust policy evaluation and optimization. Statistical Science , 29(4):485-511, 2014.
- A. Garivier and O. Cappe. The KL-UCB algorithm for bounded stochastic bandits and beyond. In Proceeding of the 24th Annual Conference on Learning Theory , pages 359-376, 2011.
- A. Gelman, J. Carlin, H. Stern, D. Dunson, A. Vehtari, and D. Rubin. Bayesian Data Analysis . Chapman &amp; Hall, 2013.
- J. Hong, B. Kveton, S. Katariya, M. Zaheer, and M. Ghavamzadeh. Deep hierarchy in bandits. In Proceedings of the 39th International Conference on Machine Learning , 2022a.
- J. Hong, B. Kveton, M. Zaheer, and M. Ghavamzadeh. Hierarchical bayesian bandits. In International Conference on Artificial Intelligence and Statistics , pages 7724-7741. PMLR, 2022b.
- J. Hong, B. Kveton, M. Zaheer, and M. Ghavamzadeh. Hierarchical Bayesian bandits. In Proceedings of the 25th International Conference on Artificial Intelligence and Statistics , 2022c.
- O. Jeunen and B. Goethals. Pessimistic reward models for off-policy learning in recommendation. In Fifteenth ACM Conference on Recommender Systems , pages 6374, 2021.
- Y. Jin, Z. Yang, and Z. Wang. Is pessimism provably efficient for offline rl? In International Conference on Machine Learning , pages 5084-5096. PMLR, 2021.
- T. Joachims, A. Swaminathan, and T. Schnabel. Unbiased learning-to-rank with biased feedback. In Proceedings of the tenth ACM international conference on web search and data mining , pages 781-789, 2017.
- B. Kveton, M. Konobeev, M. Zaheer, C.-W. Hsu, M. Mladenov, C. Boutilier, and C. Szepesvari. MetaThompson sampling. In Proceedings of the 38th International Conference on Machine Learning , 2021.
- S. Lam and J. Herlocker. MovieLens Dataset. http://grouplens.org/datasets/movielens/, 2016.
- B. Laurent and P. Massart. Adaptive estimation of a quadratic functional by model selection. The Annals of Statistics , 28(5):1302-1338, 2000.

- A. Lazaric and M. Ghavamzadeh. Bayesian multi-task reinforcement learning. In ICML-27th International Conference on Machine Learning , pages 599-606. Omnipress, 2010.
- L. Li, W. Chu, J. Langford, and R. Schapire. A contextualbandit approach to personalized news article recommendation. In Proceedings of the 19th International Conference on World Wide Web , 2010a.
- L. Li, W. Chu, J. Langford, and R. E. Schapire. A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th international conference on World wide web , pages 661-670, 2010b.
- X. Lu and B. Van Roy. Information-theoretic confidence bounds for reinforcement learning. In Advances in Neural Information Processing Systems 32 , 2019.
- A. Moradipari, B. Turan, Y. Abbasi-Yadkori, M. Alizadeh, and M. Ghavamzadeh. Parameter and feature selection in stochastic linear bandits. CoRR , abs/2106.05378, 2021. URL https://arxiv.org/abs/2106.05378 .
- A. Peleg, N. Pearl, and R. Meirr. Metalearning linear bandits by prior update. In Proceedings of the 25th International Conference on Artificial Intelligence and Statistics , 2022.
- D. Russo and B. Van Roy. Learning to optimize via posterior sampling. Mathematics of Operations Research , 39 (4):1221-1243, 2014.
- D. Russo, B. Van Roy, A. Kazerouni, I. Osband, and Z. Wen. A tutorial on Thompson sampling. Foundations and Trends in Machine Learning , 11(1):1-96, 2018.
- R. Salakhutdinov and A. Mnih. Probabilistic matrix factorization. In Advances in Neural Information Processing Systems 20 , 2007.
- M. Simchowitz, C. Tosh, A. Krishnamurthy, D. Hsu, T. Lykouris, M. Dudik, and R. Schapire. Bayesian decision-making under misspecified priors with applications to meta-learning. In Advances in Neural Information Processing Systems 34 , 2021.
- A. Swaminathan and T. Joachims. Counterfactual risk minimization: Learning from logged bandit feedback. In International Conference on Machine Learning , pages 814-823. PMLR, 2015.
- A. Swaminathan, A. Krishnamurthy, A. Agarwal, M. Dudik, J. Langford, D. Jose, and I. Zitouni. Offpolicy evaluation for slate recommendation. Advances in Neural Information Processing Systems , 30, 2017.
- W. R. Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika , 25(3-4):285-294, 1933.
- R. Wan, L. Ge, and R. Song. Metadata-based multi-task bandits with Bayesian hierarchical models. In Advances in Neural Information Processing Systems 34 , 2021.
- R. Wan, L. Ge, and R. Song. Towards scalable and robust structured bandits: A meta-learning framework. CoRR , abs/2202.13227, 2022. URL https://arxiv.org/ abs/2202.13227 .

## A Appendix

This appendix contains proofs of our claims.

## A.1 Proof of Theorem 1

The theorem proved using several lemmas. We start with the concentration of the model parameter. To simplify notation, we define r ( x, a ) = r ( x, a ; θ ∗ ) .

Lemma 3. Let

This completes the proof.

<!-- formula-not-decoded -->

be the event that all high-probability confidence intervals hold. Then P ( E | D ) ≥ 1 -δ .

Proof. We start with the Cauchy-Schwarz inequality,

<!-- formula-not-decoded -->

Since θ ∗ -ˆ θ ∼ N ( 0 , ˆ Σ) , we know that ˆ Σ -1 2 ( θ ∗ -ˆ θ ) is a d -dimensional vector of i.i.d. standard normal variables. As a result, ( θ ∗ -ˆ θ ) glyph[latticetop] ˆ Σ -1 ( θ ∗ -ˆ θ ) is a chi-squared random variable with d degrees of freedom. Therefore, by Lemma 1 of Laurent and Massart (2000),

<!-- formula-not-decoded -->

This completes our proof.

We use Lemma 3 to bound the suboptimality of ˆ π in any context by the confidence interval width induced by π ∗ .

Lemma 4. The learned policy ˆ π ∈ Π satisfies

<!-- formula-not-decoded -->

for all contexts x ∈ X with probability at least 1 -δ .

Proof. For any context x ∈ X , we can decompose

<!-- formula-not-decoded -->

By Lemma 3, event E holds with probability at least 1 -δ . Under event E ,

<!-- formula-not-decoded -->

Analogously, under event E ,

<!-- formula-not-decoded -->

Now we combine the above two inequalities and get

<!-- formula-not-decoded -->

Since the above lemma holds for any context, we can use use it to bound the suboptimality of ˆ π by the expected confidence interval width induced by π ∗ ,

<!-- formula-not-decoded -->

The second inequality follows from the concavity of the square root.

The last step is an upper bound on the expected confidence interval width. Specifically, let Γ = Σ -1 0 + γσ -2 nG ∗ . By Assumption 2, ˆ Σ -1 glyph[followsequal] Γ and thus ˆ Σ glyph[precedesequal] Γ -1 . So, for any policy π ∗ , we have

<!-- formula-not-decoded -->

The first inequality follows from Assumption 2. The first equality holds because v glyph[latticetop] v = tr( vv glyph[latticetop] ) for any v ∈ R d . The next three equalities use that the expectation of the trace is the trace of the expectation, the cyclic property of the trace, and the definition of matrix inverse. The last inequality follows from tr( A -1 ) ≤ dλ 1 ( A -1 ) = dλ -1 d ( A ) , which holds for any PSD matrix A ∈ R d × d .

Now we apply basic eigenvalue identities and inequalities, and get

<!-- formula-not-decoded -->

To finalize the proof, we chain the last two claims and get

<!-- formula-not-decoded -->

This completes the proof.

## A.2 Proof of Theorem 2

The theorem is proved using several lemmas. We start with the concentration of the model parameter in task s . To simplify notation, let r s ( x, a ) = r ( x, a ; θ s, ∗ ) .

Lemma 5. Let

<!-- formula-not-decoded -->

be the event that all high-probability confidence intervals in task s ∈ S hold. Then P ( E | D ) ≥ 1 -δ .

Proof. The proof is analogous to Lemma 3, since only the mean and covariance of θ s, ∗ | D changed, and this change is reflected in ˆ r s ( x, a ) and c s ( x, a ) .

Now we apply Lemma 4, with task-dependent quantities and Lemma 5, and get that the learned policy ˆ π s satisfies

<!-- formula-not-decoded -->

for all contexts x ∈ X with probability at least 1 -δ . Since the above bound holds for any context, we can use use it to bound the suboptimality of ˆ π s by the expected confidence interval width induced by π s, ∗ . Specifically, analogously to (10), we have

<!-- formula-not-decoded -->

The latter term, which represents the conditional task uncertainty, can be bounded exactly as in Theorem 1,

<!-- formula-not-decoded -->

For the former term, which represents the hyper-parameter uncertainty, we have

<!-- formula-not-decoded -->

To bound the maximum eigenvalue, we further proceed as

<!-- formula-not-decoded -->

The second inequality follows from λ 1 ( G s, ∗ ) ≤ 1 and λ 1 ( ˜ Σ s Σ -1 0 ) ≤ 1 . Finally, we apply basic eigenvalue identities and inequalities, and get

<!-- formula-not-decoded -->

where we use Assumption 3 in the last inequality. When we combine the last three derivations, we get

<!-- formula-not-decoded -->

This completes the proof of the first claim in Theorem 2.

Note that the bound depends on λ 1 ( G -1 z, ∗ ) , which can be large when λ d ( G z, ∗ ) is small. This is possible since π z, ∗ , which induces G z, ∗ , is a deterministic policy. We can eliminate this dependence when we adopt Assumption 4. Under this assumption, we have

<!-- formula-not-decoded -->

The equality follows from the fact that all matrices in the product are diagonal and thus commute. Moreover.

<!-- formula-not-decoded -->

Finally, we bound the minimum eigenvalue from below using basic eigenvalue identities and inequalities,

<!-- formula-not-decoded -->

In the last two inequalities, we use that λ 1 ( G s, ∗ ) ≤ 1 . In the last inequality, we also use that Assumption 3 holds for any task parameter including θ z, ∗ = θ s, ∗ . Moreover, G z glyph[followsequal] γσ -2 n z G s, ∗ implies G -1 z glyph[precedesequal] γ -1 σ 2 n -1 z G s, ∗ . This completes the proof of the second claim in Theorem 2.