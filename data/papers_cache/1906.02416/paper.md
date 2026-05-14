## Sparse Parallel Training of Hierarchical Dirichlet Process Topic Models

Alexander Terenin

Imperial College London M˚ ans Magnusson Uppsala University and Aalto University

## Abstract

To scale non-parametric extensions of probabilistic topic models such as Latent Dirichlet allocation to larger data sets, practitioners rely increasingly on parallel and distributed systems. In this work, we study data-parallel training for the hierarchical Dirichlet process (HDP) topic model. Based upon a representation of certain conditional distributions within an HDP, we propose a doubly sparse data-parallel sampler for the HDP topic model. This sampler utilizes all available sources of sparsity found in natural language-an important way to make computation efficient. We benchmark our method on a well-known corpus (PubMed) with 8m documents and 768m tokens, using a single multi-core machine in under four days.

## 1 Introduction

Topic models are a widely-used class of methods that allow practitioners to identify latent semantic themes in large bodies of text in an unsupervised manner. They are particularly attractive in areas such as history (Yang et al., 2011; Wang et al., 2012), sociology (DiMaggio et al., 2013), and political science (Roberts et al., 2014), where a desire for careful control of structure and prior information incorporated into the model motivates one to adopt a Bayesian approach to learning. In these areas, large corpora such as newspaper archives are becoming increasingly available (Ehrmann et al., 2020), and models such as latent Dirichlet allocation (LDA) (Blei et al., 2003) and its nonparametric extensions (Teh et al., 2006; Teh, 2006; Hu and Boyd-Graber, 2012; Paisley et al., 2015) are widely used by practitioners. Moreover, these models are emerging as a component of data-efficient language models (Guo et al., 2020). Training topic models efficiently entails two requirements.

1. Expose sufficient parallelism that can be taken advantage of by the hardware.

Leif Jonsson Ericsson AB and Link¨ oping University

2. Utilize sparsity found in natural language to control memory requirements and computational complexity.

In this work, we focus on the hierarchical Dirichlet process (HDP) topic model of Teh et al. (2006), which we review in Section 2. This model is a simple non-trivial extension of LDA to the nonparametric setting. This parallel implementation provides a blueprint for designing massively parallel training algorithms in more complicated settings, such as nonparametric dynamic topic models (Ahmed and Xing, 2010) and tree-based extensions (Hu and Boyd-Graber, 2012).

Parallel approaches to training HDPs have been previously introduced by a number of authors, including Newman et al. (2009), Wang et al. (2011), Williamson et al. (2013), Chang and Fisher (2014) and Ge et al. (2015). These techniques suit various settings: some are designed to explicitly incorporate sparsity present in natural language and other discrete spaces, while others are intended for HDPbased continuous mixture models. Gal and Ghahramani (2014) have pointed out that some methods can suffer from load-balancing issues, which limit their parallelism and scalability. The largest benchmark of parallel HDP training performed to our awareness is by Chang and Fisher (2014) on the 100m-token NYTIMES corpora. Throughout this work, we focus on Markov chain Monte Carlo (MCMC) methods-empirically, their scalability is comparable to variational methods (Magnusson et al., 2018; Hoffman and Ma, 2019), and, subject to convergence, they yield the correct posterior.

Our contributions are as follows. We propose an augmented representation of the HDP for which the topic indicators can be sampled in parallel over documents. We prove that, under this representation, the global topic distribution Ψ is conditionally conjugate given an auxiliary parameter l . We develop fast sampling schemes for Ψ and l , and propose a training algorithm with a per-iteration complexity that depends on the minima of two sparsity terms-it takes advantage of both document-topic and topic-word sparsity simultaneously.

Table 1: Notation for the HDP topic model. Sufficient statistics are conditional on the algorithm's current iteration. Bold symbols refer to matrices, bold italics refer to vectors, possibly countably infinite.

| Symbol   | Description                             | Symbol      | Description                              |
|----------|-----------------------------------------|-------------|------------------------------------------|
| V        | Vocabulary size                         | Ψ : 1 ×∞    | Global distribution over topics          |
| D        | Total number of documents               | Θ : D ×∞    | Document-topic probabilities             |
| N        | Total number of tokens                  | θ d : 1 ×∞  | Topic probabilities for document d       |
| v ( i )  | Word type for token i                   | m : D ×∞    | Document-topic sufficient statistic      |
| d ( i )  | Document for token i                    | Φ : ∞× V    | Topic-word probabilities                 |
| w i,d    | Token i in document d                   | φ k : 1 × V | Word probabilities for topic k           |
| b i,d    | Global topic draw indicator for w i,d   | n : ∞× V    | Topic-word sufficient statistic          |
| z i,d    | Topic indicator for token i in d        | l : 1 ×∞    | Global topic latent sufficient statistic |
| K ∗      | Index for implicitly-represented topics | α, β ,γ     | Prior concentration for θ d , φ k , Ψ    |

## 2 Partially collapsed Gibbs sampling for hierarchical Dirichlet processes

The hierarchical Dirichlet process topic model (Teh et al., 2006) begins with a global distribution Ψ over topics. Documents are assumed exchangeable-for each document d , the associated topic distribution θ d follows a Dirichlet process centered at Ψ . Each topic is associated with a distribution of tokens φ k . Within each document, tokens are assumed exchangeable (bag of words) and assigned to topic indicators z i,d . For given data, we observe the tokens w i,d .

We thus arrive at the GEM representation of a HDP, given by equation (19) of Teh et al. (2006) as

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where α, β , γ are prior hyperparameters.

## 2.1 Intuition and augmented representation

At a high level, our strategy for constructing a scalable sampler is as follows. Conditional on Ψ , the likelihood in equations (1)-(5) is the same as that of LDA. Using this observation, the Gibbs step for z , which is the largest component of the model, can be handled efficiently by leveraging insights on sparse parallel sampling from the well-studied LDA literature (Yao et al., 2009; Li et al., 2014;

Magnusson et al., 2018; Terenin et al., 2019). For this strategy to succeed, we need to ensure that all Gibbs steps involved in the HDP under this representation are analytically tractable and can be computed efficiently. For this, the representation needs to be modified.

To begin, we integrate each θ d out of the model, which by conjugacy (Blackwell and MacQueen, 1973) yields a P´ olya sequence for each z d . By definition, given in Appendix A, this sequence is a mixture distribution with respect to a set of Bernoulli random variables b d , each representing whether z i,d was drawn from Ψ or from a repeated draw in the P´ olya urn. Thus, the HDP can be written

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where PS( Ψ , b d ) is a P´ olya sequence, defined in Appendix A. This representation defines a posterior distribution over z , Φ , Ψ , b for the HDP. To derive a Gibbs sampler, we calculate its full conditionals.

## 2.2 Full conditionals for z , Φ , and b

The full conditionals z | Φ , Ψ and Φ | z , Ψ , with b marginalized out, are essentially those in partially collapsed LDA (Magnusson et al., 2018; Terenin et al., 2019). They are

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where v ( i ) is the word type for word token i , and

<!-- formula-not-decoded -->

where m -i d,k denotes the document-topic sufficient statistic with index i removed, and n k is the topicword sufficient statistic. Note the number of possible topics and full conditionals φ k | z here is countably infinite. The full conditional for each b i,d is

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The derivation, based on a direct application of Bayes' Rule with respect to the probability mass function of the P´ olya sequence, is in Appendix A.

## 2.3 The full conditional for Ψ

To derive the full conditional for Ψ , we examine the prior and likelihood components of the model. It is shown in Appendix A that the likelihood term z d | b d , Ψ may be written

glyph[negationslash]

<!-- image -->

The first term is a multiplicative constant independent of Ψ and vanishes via normalization. Thus, the full conditional Ψ | z , b depends on z and b only through the sufficient statistic l defined by

<!-- formula-not-decoded -->

and so we may suppose without loss of generality that the likelihood term is categorical. Under these conditions, we prove the full conditional for Ψ admits a stick-breaking representation.

Proposition 1. Without loss of generality, suppose

<!-- formula-not-decoded -->

Then Ψ | x is given by

<!-- formula-not-decoded -->

where l are the empirical counts of x .

<!-- formula-not-decoded -->

This expression is similar to the stick-breaking representation of a Dirichlet process DP( · , F ) -however, it has different weights and does not include random atoms drawn from F as part of its definition-see Appendix B for more details. Putting these ideas together, we define an infinitedimensional parallel Gibbs sampler.

Algorithm 1. Repeat until convergence.

- Sample φ k ∼ Dir( n k + β ) in parallel over topics for k = 1 , .., ∞ .
- Sample z i,d ∝ φ k,v ( i ) α Ψ k + φ k,v ( i ) m -i d,k in parallel over documents for d = 1 , .., D .
- Sample b i,d according to equation (14) in parallel over documents for d = 1 , .., D .
- Sample Ψ according to equations (19) -(20) .

Algorithm 1 is completely parallel, but cannot be implemented as stated due to the infinite number of full conditionals for Φ , as well as the infinite product used in sampling Ψ . We now bypass these issues by introducing an approximate finitedimensional sampling scheme.

## 2.4 Finite-dimensional sampling of Ψ and Φ

By way of assuming Ψ ∼ GEM( γ ) , an HDP assumes an infinite number of topics are present a priori, with the number of tokens per topic decreasing rapidly with the topic's index in a manner controlled by γ . Thus, under the model, a topic with a sufficiently large index should contain no tokens with high probability.

We thus propose to approximate Ψ by projecting its tail onto a single flag topic K ∗ , which stands for all topics not explicitly represented as part of the computation. This can be done by by deterministically setting ς K ∗ = 1 in equation (19). The resulting finite-dimensional Ψ will be the correct posterior full conditional for the finite-dimensional generalized Dirichlet prior considered previously in Section 2.3. Hence, this finite-dimensional truncation forms a Bayesian model in its own right, which suggests it should perform reasonably well. From an asymptotic perspective, Ishwaran and James (2001) have shown that the approximation is almost surely convergent and, therefore, well-posed.

Once this is done, Ψ becomes a finite vector of length K ∗ , and only K ∗ rows of Φ need to be explicitly instantiated as part of the computation. This instantiation allows the algorithm to be defined on a fixed finite state space, simplifying bookkeeping and implementation.

From a computational efficiency perspective, the resulting value K ∗ takes the place of K in partially collapsed LDA. However, it cannot be interpreted as the number of topics in the sense of LDA. Indeed, LDA implicitly assumes that Ψ = Unif(1 , .., K ) deterministically-i.e., that every topic is assumed a priori to contain the same number of tokens. In contrast, the HDP model learns this distribution from the data by letting Ψ ∼ GEM( γ ) .

If we allow the state space to be resized when topic K ∗ is sampled, then following Papaspiliopoulos and Roberts (2008), it is possible to develop truncation schemes which introduce no error. Since this results in more complicated bookkeeping which reduces performance, we instead fix K ∗ and defer such considerations to future work. We recommend setting K ∗ to be sufficiently large that it does not significantly affect the model's behavior, which can be checked by tracking the number of tokens assigned to the topic K ∗ .

## 2.5 Sparse sampling of Φ and z

To be efficient, a topic model needs to utilize the sparsity found in natural language as much as possible. In our case, the two main sources of sparsity are as follows.

1. Document-topic sparsity : most documents will only contain a handful of topics.
2. Topic-word sparsity : most word types will not be present in most topics.

We thus expect the document-topic sufficient statistic m and topic-word sufficient statistic n to contain many zeros. We seek to use this to reduce sampling complexity. Our starting point is the Poisson P´ olya Urn sampler of Terenin et al. (2019), which presents a Gibbs sampler for LDA with computational complexity that depends on the minima of two sparsity coefficients representing documenttopic and topic-word sparsity-such algorithms are termed doubly sparse . The key idea is to approximate the Dirichlet full conditional for φ k with a Poisson P´ olya Urn (PPU) distribution defined by

<!-- formula-not-decoded -->

for v = 1 , .., V . This distribution is discrete, so Φ becomes a sparse matrix. The approximation is accurate even for small values of n k,v , and Terenin et al. (2019) proves that the approximation error will vanish for large data sets in the sense of convergence in distribution.

If β is uniform, we can further use sparsity to accelerate sampling ϕ k,v . Since a sum of Poisson random variables is Poisson, we can split ϕ k,v = ϕ ( β ) k,v + ϕ ( n ) k,v . We then sample ϕ ( β ) k,v sparsely by introducing a Poisson process and sampling its points uniformly, and sample ϕ ( n ) k,v sparsely by iterating over nonzero entries of n .

For z , the full conditional

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

is similar to to the one in partially collapsed LDA (Magnusson et al., 2018)-the difference is the presence of Ψ k . As Ψ k only enters the expression through component ( a ) and is identical for all z i,d , it can be absorbed at each iteration directly into an alias table (Walker, 1977; Li et al., 2014). Component ( b ) can be computed efficiently by utilizing sparsity of Φ and m and iterating over whichever has fewer non-zero entries.

## 2.6 Direct sampling of l

Rather than sampling b , whose size will grow linearly with the number of documents, we introduce a scheme for sampling the sufficient statistic l directly. Observe that

<!-- formula-not-decoded -->

where the domain of summation and the value of the indicators have been switched. By definition of b i,d , we have

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Summing this expression over documents, we obtain the expression

<!-- formula-not-decoded -->

where D k,j is the total number of documents with m d,k ≥ j . Since m d,k = 0 for all topics k without any tokens assigned, we only need to sample l for topics that have tokens assigned to them. This idea can also be straightforwardly applied to other HDP samplers (Chang and Fisher, 2014; Ge et al., 2015), by allowing one to derive alternative full conditionals in lieu of the Stirling distribution (Antoniak, 1974). The complexity of sampling l directly is constant with respect to the number of documents, and depends instead on the maximum number of tokens per document.

To handle the bookkeeping necessary for computing D k,j , we introduce a sparse matrix d of size K × max d N d whose entries d k,p are the number of documents for topic k that have a total of p topic indicators assigned to them. We increment d once z d been sampled by iterating over non-zero elements in m d . We then compute D k,j as the reverse cumulative sum of the rows of d .

## 2.7 Poisson P´ olya urn partially collapsed Gibbs sampling

Putting all of these ideas together, we obtain the following algorithm.

Algorithm 2. Repeat until convergence.

- Sample φ k ∼ PPU( n k + β ) in parallel over topics for k = 1 , .., K ∗ .
- Sample z i,d ∝ φ k,v ( i ) α Ψ k + φ k,v ( i ) m -i d,k in parallel over documents for d = 1 , .., D .
- Sample l k according to equation (28) in parallel over topics for k = 1 , .., K ∗ .
- Sample Ψ according to equations (19) -(20) , except with ς K ∗ = 1 .

Algorithm 2 is sparse, massively parallel, defined on a fixed finite state space, and contains no infinite computations in any of its steps. The Gibbs step for Φ converges in distribution (Terenin et al., 2019) to the true Gibbs steps as N →∞ , and the Gibbs step for Ψ converges almost surely (Ishwaran and James, 2001) to the true Gibbs step as K ∗ →∞ .

## 2.8 Computational complexity

We now examine the per-iteration computational complexity of Algorithm 2. To proceed, we fix K ∗

and maximum document size max d N d , and relate the vocabulary size V with the number N of total words as follows.

Assumption (Heaps' Law) . The number of unique words in a corpus follows Heaps' law (Heaps, 1978) V = ξN ζ with constants ξ &gt; 0 and ζ &lt; 1 .

The per-iteration complexity of Algorithm 2 is equal to the sum of the per-iteration complexity of sampling its components. The sampling complexities of Ψ and l are constant with respect to the number of tokens, and the sampling complexity of Φ has been shown by Magnusson et al. (2018) to be negligible under the given assumptions. Thus, it suffices to consider z .

At a given iteration, let K ( m ) d ( i ) be the number of existing topics in document d associated with word token i , and let K ( Φ ) v ( i ) be the number of nonzero topics in the row of Φ corresponding to word token i . It follows immediately from the argument given by Terenin et al. (2019) that the per-iteration complexity of sampling each topic indicator z i is

<!-- formula-not-decoded -->

Algorithm 2 is thus a doubly sparse algorithm.

## 3 Performance results

To study performance of the partially collapsed sampler-Algorithm 2-we implemented it in Java using the open-source MALLET 1 (McCallum, 2002) topic modeling framework. We ran it on the AP, CGCBIB, NEURIPS, and PUBMED corpora, 1 which are summarized in Table 2. Prior hyperparameters controlling the degree of sparsity were set to α = 0 . 1 , β = 0 . 01 , γ = 1 . We set K ∗ = 1000 and observed no tokens ever allocated to the topic K ∗ . Data were preprocessed with default Mallet (McCallum, 2002) stop-word removal, minimum document size of 10, and a rare word limit of 10. Following Teh et al. (2006), the algorithm was initialized with one topic. All experiments were repeated five times to assess variability. Total runtime for each experiment is given in Table 2.

To assess Algorithm 2 in a small-scale setting, we compare it to the widely-studied sparse fully collapsed direct assignment sampler of Teh et al. (2006), which is not parallel. We ran 100 000

1 See HTTP://MALLET.CS.UMASS.EDU and HTTPS://GITHUB.COM/LEJON/PARTIALLYCOLLAPSEDLDA. AP and CGCBIB can be found therein. NeurIPS and PubMed can be found at HTTPS://ARCHIVE.ICS.UCI.EDU/ML/DATASETS/BAG+OF+WORDS. Full output of experiments can be found at HTTPS://GITHUB.COM/ATERENIN/PARALLEL-HDP-EXPERIMENTS/.

Figure 1: Trace plots for log-likelihood, number of active topics, and additional metrics for CGCBIB, NeurIPS, and PubMed. On the x axis, per-iteration scale is used for AP, CGCBIB and PubMed, and real-time scale is used for NeurIPS. Algorithms used are partially collapsed HDP for all corpora, direct assignment HDP for AP and CGCBIB, and subcluster split-merge HDP for NeurIPS. Individual traces are partially transparent, and their mean is opaque.

<!-- image -->

iterations of both methods on AP and CGCBIB. We selected these corpora because they were among the larger corpora on which it was feasible to run our direct assignment reference implementation within one week.

Trace plots for the log marginal likelihood for z given Ψ and the number of active topics, i.e., those topics assigned at least one token, can be seen in Figure 1(a,d) and Figure 1(b,e), respectively. The direct assignment algorithm converges slower, but achieves a slightly better local optimum in terms of marginal log-likelihood, compared to our method. This fact indicates that the direct assignment method may stabilize around a different local optimum, and may represent a potential limitation of the partially collapsed sampler in settings where non-parallel methods are practical.

To better understand the distributional differences between the algorithms, we examined the number of tokens per topic, which can be seen in Figure 1(c,f). The partially collapsed sampler is seen to assign more tokens to smaller topics, indicating that it stabilizes around a local optimum with slightly broader semantic themes.

To visualize the effect this has on the topics, we examined the most common words for each topic. Since the algorithms generate too many topics to make full examination practical, we instead compute a quantile summary with five topics per quantile. The quantile is computed by ranking all topics by the number of tokens, choosing the five closest topics to the 100% , 75% , 50% , 25% , and 5% quantiles in the ranking, and computing their top words. This approach gives a representative view of the algorithm's output for large, medium, and small topics. Results may be seen in Appendix D and Appendix C-we find the direct assignment and partially collapsed samplers to be mostly com- parable, with substantial overlap in top words for common topics.

Table 2: Corpora used in experiments, together with compute configuration.

| Corpus   | V      | D         | N           | Iterations   |   Threads | Runtime    |
|----------|--------|-----------|-------------|--------------|-----------|------------|
| AP       | 7 074  | 2 206     | 393 567     | 100 000      |         8 | 3.8 hours  |
| CGCBIB   | 6 079  | 5 940     | 570 370     | 100 000      |        12 | 2.7 hours  |
| NeurIPS  | 12 419 | 1 499     | 1 894 051   | 255 500      |         8 | 24 hours   |
| PubMed   | 89 987 | 8 199 999 | 768 434 972 | 25 000       |        20 | 82.4 hours |

Next, we assess Algorithm 2 in a more demanding setting and compare against previous parallel state-of-the-art. There are various scalable samplers available for the HDP. For a fair comparison, we restrict ourselves to those samplers designed for topic models and explicitly incorporate sparsity of natural language in their construction. Among these, we selected the parallel subcluster splitmerge algorithm of Chang and Fisher (2014) as our baseline because it was used in the largest-scale benchmark of the HDP topic model performed to date to our awareness, and shows comparable performance to other methods (Ge et al., 2015). The subcluster split-merge algorithm is designed to converge with fewer iterations, but is more costly to run per iteration. Thus, we used a fixed computational budget of 24 hours of wall-clock time for both algorithms. Computation was performed on a system with a 4-core 8-thread CPU and 8GB RAM.

Results can be seen in Figure 1(g)-note that the subcluster split-merge algorithm is parametrized using sub-topic indicators and sub-topic probabilities , so its numerical log-likelihood values are not directly comparable to ours and should be interpreted purely to assess convergence . Algorithm 2 stabilizes much faster with respect to both the number of active topics in Figure 1(g), and marginal log-likelihood in Figure 1(h). The subcluster splitmerge algorithm adds new topics one-at-a-time, whereas our algorithm can create multiple new topics per iteration-we hypothesize this difference leads to faster convergence for Algorithm 2.

In Figure 1(i), we observe that the amount of computing time per iteration increases substantially for the subcluster split-merge method as it adds more topics. For Algorithm 2, this stays approximately constant for its entire runtime.

To evaluate the topics produced by the algorithms, we again examined the most common words for each topic via a quantile summary, given in Appendix E. We find the subcluster split-merge algorithm appears to generate topics with slightly more semantic overlap compared to Algorithm 2, but otherwise produces comparable output.

Finally, to assess scalability, we ran 25 000 iterations of Algorithm 2 on PubMed, which contains 768m tokens. To our knowledge, this dataset is an order of magnitude larger than any datasets used in previous MCMC-based approaches for the HDP.

Computation was performed on a compute node with 2x10-core CPUs with 20 threads and 64GB of RAM. The marginal likelihood and number of active topics are given in Figure 1(j) and Figure 1(k).

To evaluate the topics discovered by the algorithm, we examined their most common wordsthese may be seen in full in Appendix F. We observe that the semantic themes present in the topics vary according to how many tokens they have: topics with more tokens appear to be broader, whereas topics with fewer tokens appear to be more specific. This behavior illustrates a key difference between the HDP and methods like LDA, which do not contain a learned global topic distribution Ψ in their formulation. We suspect the effect is particularly pronounced on PubMed compared to CGCBIB and NeurIPS due to its large number of tokens.

## 4 Discussion

In this work, we introduce the parallel partially collapsed Gibbs sampler-Algorithm 1-for the HDP topic model, which converges to the correct target distribution. We propose a doubly sparse approximate sampler-Algorithm 2-which allows the HDP to be implemented with per-token sampling complexity of O [ min ( K ( m ) d ( i ) , K ( Φ ) v ( i ) )] which is the same as that of P´ olya Urn LDA (Terenin et al., 2019). Compared to other approaches for the HDP, it offers the following improvements.

1. The algorithm is fully parallel in all steps.
2. The topic indicators z utilize all available sources of sparsity to accelerate sampling.
3. All steps not involving z have constant complexity with respect to data size.
4. The proposed sparse approximate algorithm becomes exact as N →∞ and K ∗ →∞ .

These improvements allow us to train the HDP on larger corpora. The data-parallel nature of our approach means that the amount of available parallelism increases with data size. This parallelism avoids load-balancing-related scalability limitations pointed out by Gal and Ghahramani (2014).

Nonparametric topic models are less straightforward to evaluate empirically than ordinary topic models. In particular, we found topic coherence scores (Mimno et al., 2011) to be strongly affected by the number of active topics K , which causes preference for models with fewer topics and more semantic overlap per topic. We view the development of summary statistics that are K -agnostic and those measuring other aspects of topic quality such as overlap, to be an important direction for future work. We are particularly interested in techniques that can be used to compare algorithms for sampling from the same model defined over fully disjoint state spaces, such as Algorithm 2 and the subcluster split-merge algorithm in Section 3.

Figure 2: Top 8 words for topics obtained by Algorithm 2 on PubMed, together with topic index k and total number of words n k, · present in the topic. We observe that the topics range from broad to specific: this is a consequence of the hierarchical Dirichlet process prior via the inclusion of the global topic proportions Ψ . Topics obtained by Algorithm 2 on all corpora may be seen in Appendix C, Appendix D, Appendix E, and Appendix F.

| k n k, •   | Topic 1 42 395 289 care health patient medical research system clinical cost   | Topic 5 23 907 517 cancer tumor patient cell carcinoma breast tumour survival   | Topic 9 22 167 377 protein binding membrane acid activity cell gel human    | Topic 13 20 925 933 protein cell kinase expression receptor activation pathway phosphorylati   | Topic 17 18 924 590 cell neuron electron brain rat nerve fiber nucleus   |
|------------|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| k          | Topic 21                                                                       | Topic 25                                                                        | Topic 29                                                                    | Topic 33                                                                                       | Topic 37                                                                 |
| n k, •     | 18 033 777 cell growth expression factor beta human mrna endothelial           | 16 308 024 rat day mice liver animal effect control mg                          | 15 128 822 gene mutation genetic chromosome analysis genes polymorphism dna | 13 562 338 infection strain antibiotic bacterial isolates bacteria resistance coli             | 10 819 160 plant strain acid growth extract activity cell production     |

Partially collapsed HDP can stabilize around a different local mode than fully collapsed HDP as proposed by Teh et al. (2006). There have been attempts to improve mixing in that sampler (Chang and Fisher, 2014), including the use of MetropolisHastings steps for jumping between modes (Jain and Neal, 2004). These techniques are largely complementary to ours and can be explored in combination with the ideas presented here.

The HDP posterior is a heavily multimodal target for which full posterior exploration is known to be difficult (Chang and Fisher, 2014; Gal and Ghahramani, 2014; Buntine and Mishra, 2014), and sampling schemes are generally used more in the spirit of optimization than traditional MCMC. These issues are mirrored in other approaches, such as variational inference. There, restrictive meanfield factorization assumptions are often required, which reduces the quality of discovered topics. We view MAP-based analogs of ideas presented here as a promising direction, since these may allow additional flexibility that may enable faster training.

Many of the ideas in this work, such as the binomial trick, are generic and apply to any topic model structurally similar to the HDP's GEM representation (Teh et al., 2006) given in Section 2. For example, one could consider an informative prior for Ψ in lieu of GEM( γ ) , potentially improving convergence and topic quality, or developing parallel schemes for other nonparametric topic models such as Pitman-Yor models (Teh, 2006), tree-based models (Hu and Boyd-Graber, 2012; Paisley et al., 2015), embedded topic models (Dieng et al., 2020), as well as nonparametric topic models used within data-efficient language models (Guo et al., 2020) in future work.

Conclusion We introduce the doubly sparse partially collapsed Gibbs sampler for the hierarchical Dirichlet process topic model. By formulating this algorithm using a representation of the HDP which connects it with the well-studied Latent Dirichlet Allocation model, we obtain a parallel algorithm whose per-token sampling complexity is the minima of two sparsity terms. The ideas used apply to a large array of topic models, for example, dynamic topic models with Φ time-varying, which possess the same full conditional for z . Our algorithm for the HDP scales to a 768m-token corpus (PubMed) on a single multicore machine in under four days.

The proposed techniques leverage parallelism and sparsity to scale nonparametric topic models to larger datasets than previously considered feasible for MCMC or other methods possessing similar convergence properties. We hope these contributions enable wider use of Bayesian nonparametrics for large collections of text.

Acknowledgments The research was funded by the Academy of Finland (grants 298742, 313122), as well as the Swedish Research Council (grants 201805170, 201806063). Computations were performed using compute resources within the Aalto University School of Science and Department of Computing at Imperial College London. We also acknowledge the support of Ericsson AB.

## References

- Amr Ahmed and Eric P. Xing. 2010. Timeline: a dynamic hierarchical Dirichlet process model for recovering birth/death and evolution of topics in text stream. In Uncertainty in Artificial Intelligence , pages 20-29.
- Luigi Ambrosio, Nicola Gigli, and Giuseppe Savar´ e. 2005. Gradient Flows in Metric Spaces and in the Space of Probability Measures . Birkh¨ auser.
- Charles E. Antoniak. 1974. Mixtures of Dirichlet processes with applications to Bayesian nonparametric problems. The Annals of Statistics , 2(6):1152-1174.
- David Blackwell and James B. MacQueen. 1973. Ferguson distributions via P´ olya urn schemes. The Annals of Statistics , 1(2):353-355.
- David M. Blei, Andrew Y. Ng, and Michael I. Jordan. 2003. Latent Dirichlet allocation. Journal of Machine Learning Research , 3(1):993-1022.
- Vladimir I. Bogachev. 2007. Measure Theory: Volume II . Springer.
- Wray L. Buntine and Swapnil Mishra. 2014. Experiments with non-parametric topic models. In Knowledge Discovery and Data Mining , pages 881-890.
- Jason Chang and John W. Fisher, III. 2014. Parallel sampling of HDPs using sub-cluster splits. In Advances in Neural Information Processing Systems , pages 235-243.
- Robert J. Connor and James E. Mosimann. 1969. Concepts of independence for proportions with a generalization of the Dirichlet distribution. Journal of the American Statistical Association , 64(325):194-206.
- Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei. 2020. Topic modeling in embedding spaces. Transactions of the Association for Computational Linguistics , 8:439-453.
- Paul DiMaggio, Manish Nag, and David M. Blei. 2013. Exploiting affinities between topic modeling and the sociological perspective on culture: application to newspaper coverage of US government arts funding. Poetics , 41(6):570-606.
- Maud Ehrmann, Matteo Romanello, Simon Clematide, Phillip B. Str¨ obel, and Rapha¨ el Barman. 2020. Language resources for historical newspapers: the Impresso collection. In Language Resources and Evaluation Conference , pages 958-968.
- Yarin Gal and Zoubin Ghahramani. 2014. Pitfalls in the use of parallel inference for the Dirichlet process. In International Conference on Machine Learning , pages 208-216.
- Hong Ge, Yutian Chen, Moquan Wan, and Zoubin Ghahramani. 2015. Distributed inference for Dirichlet process mixture models. In International Conference on Machine Learning , pages 2276-2284.
- Dandan Guo, Bo Chen, Ruiying Lu, and Mingyuan Zhou. 2020. Recurrent hierarchical topic-guided neural language models. In International Conference on Machine Learning , pages 10994-11005.
- Harold S. Heaps. 1978. Information Retrieval: Computational and Theoretical Aspects . Academic Press.
- Matthew D. Hoffman and Yian Ma. 2019. Langevin dynamics as nonparametric variational inference. In Advances in Approximate Bayesian Inference .
- Yuening Hu and Jordan Boyd-Graber. 2012. Efficient tree-based topic modeling. In Proceedings
- of the Association for Computational Linguistics , pages 275-279.
- Hemant Ishwaran and Lancelot F. James. 2001. Gibbs sampling methods for stick-breaking priors. Journal of the American Statistical Association , 96(453):161-173.
- Sonia Jain and Radford M. Neal. 2004. A splitmerge Markov chain Monte Carlo procedure for the Dirichlet process mixture model. Journal of Computational and Graphical Statistics , 13(1):158-182.
- Aaron Q. Li, Amr Ahmed, Sujith Ravi, and Alexander J. Smola. 2014. Reducing the sampling complexity of topic models. In Knowledge Discovery and Data Mining , pages 891-900.
- M˚ ans Magnusson, Leif Jonsson, Mattias Villani, and David Broman. 2018. Sparse partially collapsed MCMC for parallel inference in topic models. Journal of Computational and Graphical Statistics , 27(2):449-463.
- Andrew K. McCallum. 2002. MALLET: A Machine Learning for Language Toolkit.
- David Mimno, Hanna M. Wallach, Edmund Talley, Miriam Leenders, and Andrew McCallum. 2011. Optimizing semantic coherence in topic models. In Conference on Empirical Methods in Natural Language Processing , pages 262-272.
- David Newman, Arthur Asuncion, Padhraic Smyth, and Max Welling. 2009. Distributed algorithms for topic models. Journal of Machine Learning Research , 10(62):1801-1828.
- John Paisley, Chong Wang, David M. Blei, and Michael I. Jordan. 2015. Nested hierarchical Dirichlet processes. IEEE Transactions on Pattern Analysis and Machine Intelligence , 37(2):256-270.
- Omiros Papaspiliopoulos and Gareth O. Roberts. 2008. Retrospective Markov chain Monte Carlo methods for Dirichlet process hierarchical models. Biometrika , 95(1):169-186.
- Margaret E. Roberts, Brandon M. Stewart, Dustin Tingley, Christopher Lucas, Jetson Leder-Luis, Shana Kushner Gadarian, Bethany Albertson, and David G. Rand. 2014. Structural topic models for open-ended survey responses. American Journal of Political Science , 58(4):1064-1082.
- Yee Whye Teh. 2006. A hierarchical Bayesian language model based on Pitman-Yor processes. In Proceedings of the Association for Computational Linguistics , pages 985-992.
- Yee Whye Teh, Michael I. Jordan, Matthew J. Beal, and David M. Blei. 2006. Hierarchical Dirichlet processes. Journal of the American Statistical Association , 101(476):1566-1581.
- Alexander Terenin, M˚ ans Magnusson, Leif Jonsson, and David Draper. 2019. P´ olya urn latent Dirichlet allocation: a doubly sparse massively parallel sampler. IEEE Transactions on Pattern Analysis and Machine Intelligence , 41(7):17091719.
- Alastair J. Walker. 1977. An efficient method for generating discrete random variables with general distributions. ACM Transactions on Mathematical Software , 10(8):253-256.
- Chong Wang, John Paisley, and David M. Blei. 2011. Online variational inference for the hierarchical Dirichlet process. In Artificial Intelligence and Statistics , pages 752-760.
- William Yang Wang, Elijah Mayfield, Suresh Naidu, and Jeremiah Dittmar. 2012. Historical analysis of legal opinions with a sparse mixedeffects latent variable model. In Proceedings of the Association for Computational Linguistics , volume 1, pages 740-749.
- Sinead Williamson, Avinava Dubey, and Eric P. Xing. 2013. Parallel Markov chain Monte Carlo for nonparametric mixture models. In International Conference on Machine Learning , pages 98-106.
- Tze-I Yang, Andrew J. Torget, and Rada Mihalcea. 2011. Topic modeling on historical newspapers. In ACL-HLT Workshop on Language Technology for Cultural Heritage, Social Sciences, and Humanities , pages 96-104.
- Limin Yao, David Mimno, and Andrew K. McCallum. 2009. Efficient methods for topic model inference on streaming document collections. In Knowledge Discovery and Data Mining , pages 937-946.

## A Appendix: sufficiency of l and full conditional for b

Recall that the one-step-ahead conditional probability mass function in a P´ olya sequence taking values in N with concentration parameter α and base probability mass function Ψ is

<!-- formula-not-decoded -->

Introducing the random variable

<!-- formula-not-decoded -->

we can express the one-step-ahead conditional distribution as

<!-- formula-not-decoded -->

The joint probability mass function for z | b , Ψ is then

<!-- formula-not-decoded -->

Note that 1 b i =0 = 1 ⇐⇒ 1 b i =1 = 0 and vice versa. Thus each term in the product for z | b , Ψ only has one component, and we may express z | b , Ψ as

glyph[negationslash]

<!-- formula-not-decoded -->

where we have re-expressed the probability mass function of Ψ in a form that emphasizes conjugacy. Thus for any prior, the posterior will only depend on the likelihood of the values of z i for which b i = 1 . The sufficient statistic is

<!-- formula-not-decoded -->

Next, for a given i ′ ∈ { 1 , .., N } , we can calculate the posterior of a component b i ′ as

glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

glyph[negationslash]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where we have divided both expressions by

<!-- formula-not-decoded -->

glyph[negationslash]

glyph[negationslash]

glyph[negationslash]

which is constant with respect to b i ′ . Note that full conditionally, we have b i | = b i ′ for i = i ′ . This gives the desired expressions and concludes the derivation.

glyph[negationslash]

## B Appendix: full conditional for Ψ

Before proceeding with the derivation, we first comment on Proposition 1 and differences between the GEM distribution and Dirichlet process, which otherwise appear superficially similar. The GEM distribution Ψ GEM ∼ GEM( γ ) is defined as

<!-- formula-not-decoded -->

On the other hand, a Dirichlet process Ψ DP ∼ DP( γ, F ) is defined as

<!-- formula-not-decoded -->

From a Bayesian perspective, this extra stage-the presence of ϑ k -prevents one from applying standard results on conjugacy of Dirichlet processes. The joint distribution of a finite set of states (Ψ GEM k 1 , .., Ψ GEM k K ) does not admit a closed-form expression, so we seek to derive the posterior conditional in a different way.

Rather than proving conjugacy for (Ψ GEM k 1 , .., Ψ GEM k K ) directly, we look for a larger finite-dimensional distribution within which (Ψ GEM k 1 , .., Ψ GEM k K ) sits that has better conjugacy properties. The generalized Dirichlet distribution of Connor and Mosimann (1969) fulfills this criteria. The conjugacy relationship we seek follows from the general property that conditioning and marginalization commute. This will be shown to yield the posterior

<!-- formula-not-decoded -->

For comparison, a posterior Dirichlet process is given by

<!-- formula-not-decoded -->

which shows that this relatively mild difference in the prior yields a posterior of a rather different form.

We now proceed to formally calculate this posterior distribution, starting from a GEM prior and discrete likelihood. Since we are working in a nonparametric setting, we begin by introducing the necessary formalism. We then introduce our finite-dimensional approximating prior and compute the posterior under it. For this, we use commutativity of conditioning and marginalization to deduce the full infinite-dimensional posterior.

Definition 2 (Preliminaries) . Let (Ω , F , P ) be a probability space. Let M s ( N ) be the space of signed measures, equipped with the topology of weak convergence. Let M 1 ( N ) ⊂ M s ( N ) be the space of probability measures over N , and identify M 1 ( N ) with the probability simplex by the homeomorphism M 1 ( N ) ∼ = { x ∈ glyph[lscript] 1 : ∀ i, x i &gt; 0 , ∑ ∞ i =1 x i = 1 } . Let N ∈ N , let x ∈ N N , and let l ∈ N N be its empirical counts, defined by l = ∑ N i =1 1 x i where 1 x i is equal to 1 for coordinate x i and 0 for all other coordinate. Let γ ∈ R + . Recall that N N and M 1 ( N ) , endowed with the discrete topology and topology of weak convergence, respectively, are both Polish spaces-hence, the Disintegration Theorem (Ambrosio et al. (2005), Theorem 5.3.1; Bogachev (2007), Corollary 10.4.15) holds in both spaces. We associate each random variable y : Ω → Y with its pushforward probability measure π y ( A y ) = [ y ∗ P ]( A y ) = P [ y -1 ( A y )] , and each conditional random variables θ | y : Ω × Y → Θ with its pushforward regular conditional probability measure π y | θ ( A y | θ ) = [( y | θ ) ∗ P ]( A y ) = P [( y | θ ) -1 ( A y )] , where the preimage is taken with respect to y .

Definition 3 (Discrete likelihood) . For all Ψ ∈ M 1 ( N ) , define the conditional random variable x | Ψ : Ω ×M 1 ( N ) → N N by its probability mass function

<!-- formula-not-decoded -->

We say x | Ψ ∼ Discrete( Ψ ) .

Definition 4 (GEM) . Let Ψ : Ω →M 1 ( N ) be a random variable defined by

<!-- formula-not-decoded -->

We say Ψ ∼ GEM( γ ) .

Definition 5 (Finite GEM) . Let Ψ : Ω →M 1 ( N ) be a random variable defined by

<!-- formula-not-decoded -->

We say Ψ ∼ FGEM( γ, K ) .

Definition 6 (Posterior) . Let Ψ | x be the unique conditional random variable given by the Disintegration Theorem, where uniqueness follows from almost sure uniqueness by virtue of the marginal measure π x ( · ) = ∫ M 1 ( N ) π x | Ψ ( · | Ψ )d π Ψ being absolutely continuous with respect to the counting measure on N N , which has no non-empty null sets.

Result 7. Let x | Ψ ∼ Discrete( Ψ ) . Let x ∈ N N , and let K &gt; sup x . Let Ψ ∼ FGEM( γ, K ) . Then for any x with empirical counts l , we have that Ψ | x : Ω × N N → M 1 ( N ) is a conditional random variable defined by

<!-- formula-not-decoded -->

where

<!-- formula-not-decoded -->

Proof. It is shown by Connor and Mosimann (1969) that Ψ ∼ FGEM( γ, K ) is a special case of the generalized Dirichlet distribution, which admits a general stick-breaking representation. Thus, its probability density function is

<!-- formula-not-decoded -->

which we have expressed in a simplified form. By conjugacy, for a given x and associated l the posterior probability density is

<!-- formula-not-decoded -->

which is again a generalized Dirichlet admitting the necessary stick-breaking representation, which we have expressed in a form that emphasizes its posterior hyperparameters.

Remark 8. It is now clear that the assumption x | Ψ ∼ Discrete( Ψ ) is indeed taken without loss of generality, because if we instead took x | Ψ to be given by a P´ olya sequence, then by sufficiency the prior-to-posterior map would be identical.

Proposition 1. Without loss of generality, suppose

<!-- formula-not-decoded -->

Then Ψ | x is given by

<!-- formula-not-decoded -->

where l are the empirical counts of x .

Proof. Let I ⊂ N be an arbitrary finite index set, and let Ψ I | x be the finite-dimensional marginal projection of Ψ | x onto the coordinates contained in I . Let K &gt; sup I , let Ψ ( K ) | x be the posterior conditional random variable under Ψ ( K ) ∼ FGEM( γ, K ) , and let Ψ ( K ) I | x be the marginal consisting of those coordinates contained in I . By construction, Ψ ( K ) I | x equals Ψ I | x in distribution. Since by the Disintegration Theorem, conditioning and marginalization commute, the set I is arbitrary, and Ψ | x is uniquely determined by its finite-dimensional marginal projections, the claim follows.

## C Appendix: quantile summary of topics for AP

Here we display a multi-quantile summary for AP, obtained by ranking all topics with at least 100 tokens by their total number of tokens, computing the glyph[pi1] = 100% , 75% , 50% , 25% , and 5% quantiles. We compute the five topics closest to each quantile by number of tokens, and display their top-eight words.

| k Topic 1                    | Topic 2         | Topic 3           | Topic 4        | Topic 5        |
|------------------------------|-----------------|-------------------|----------------|----------------|
| n k, • 93 207                | 57 249          | 15 874            | 13 360         | 10 176         |
| week                         | people          | police            | percent        | trial          |
| made collapsed               | years           | people            | year           | court          |
| president 100%               | year            | killed            | prices         | charges        |
| officials                    | time            | man               | economic       | case           |
| = tuesday                    | don             | officials         | economy        | judge          |
| partially glyph[pi1] million | back            | city              | rate           | attorney       |
| thursday                     | day             | shot              | increase       | prison         |
| national                     | home            | authorities       | report         | jury           |
| k Topic 54                   | Topic 55        | Topic 56          | Topic 57       | Topic 58       |
| n k, • 1 055                 | 1 032           | 1 025             | 1 014          | 1 013          |
| children                     | north           | hostages          | aids           | percent        |
| parents collapsed            | walsh           | red               | virus          | poll           |
| child 75%                    | reagan          | release           | blood          | survey         |
| = ms                         | iran            | held              | disease        | points         |
| glyph[pi1] year              | contra          | hostage           | drug           | found          |
| partially mother             | documents       | anderson          | infected       | surveys        |
| boys                         | gesell          | gunmen            | immune         | margin         |
| girl                         | arms            | thursday          | health         | reported       |
| k Topic 108                  | Topic 109       | Topic 110         | Topic 111      | Topic 112      |
| n k, • 473                   | 472             | 451               | 446            | 436            |
| abortion souter              | women club      | solidarity walesa | waste garbage  | train railroad |
| anti collapsed 50%           | members         | poland            | recycling      | cars           |
| = state                      | men             | polish            | city           | trains         |
| glyph[pi1] women             | male            | government        | ash            | transportatio  |
| partially abortions          | membership      | mazowiecki        | trash          | skinner        |
| rights                       | female          | jaruzelski        | state          | transit policy |
| hampshire k Topic 162        | black Topic 163 | talks Topic 164   | dump Topic 165 | Topic 166      |
| n k, • 193                   | 187             | 185               | 184 dixon      | 184            |
| health                       | wine            | miners mine       |                | barry          |
| care collapsed               | warmus          |                   | yates          | moore          |
| spe 25%                      | solomon         |                   |                |                |
| bc                           | california      | coal mines        | count tosh     | jackson mayor  |
| = weight                     | bar             |                   |                | statehood      |
| partially glyph[pi1]         |                 | hull              | rogers         |                |
| american                     | gallo           | pittston          | rig            | mr             |
| diet                         | test            | benefits          | russell        | gregory        |
| cholesterol                  | questions       | platform          | cookies        | room           |
| k Topic 206                  | Topic 207       | Topic 208         | Topic 209      | Topic 210      |
| n k, • 117                   | 115             | 112               | 111            | 111            |
| pageant                      | mall            | roberts           | stuart         | gold           |
| miss collapsed               | malls           | shell             | lawn           | polaroid       |
| cereal 5%                    | pinochet        | boigny            | dea            | shamrock       |
| = boxes                      | shopping        | houphouet         | boston         | fields         |
| glyph[pi1] contestants       | downtown        | travelers         | ruth           | consolidated   |
| partially box                | park            | leonard           | yankees        | suit           |
| america                      | oak             | arsenal           | foundation     | proposals      |
| bruce                        | usa             | oil               | richman        | mining         |

| k Topic 1 n • 90 497                | Topic 2 18 626   | Topic 3 10 832   | Topic 4 9 923     | Topic 5 9 430    |
|-------------------------------------|------------------|------------------|-------------------|------------------|
| k, year                             | years            | police           | dollar            | percent          |
| people                              | year             | people           | market            | year             |
| time assignment 100%                | people           | killed           | stock             | rose             |
| president                           | time             | government       | yen               | sales            |
| = years                             | don              | reported         | index             | million          |
| direct glyph[pi1] made              | home             | today            | late              | billion          |
| state                               | day              | capital          | trading           | month            |
| week                                | back             | violence         | exchange          | reported         |
| k Topic 93                          | Topic 94         | Topic 95         | Topic 96          | Topic 97         |
| n k, • 784                          | 757              | 753              | 745               | 738              |
| keating                             | bus              | eastern          | united            | smoking          |
| deconcini                           | driver           | pilots           | states            | cigarettes       |
| lincoln assignment 75%              | train            | airline          | nations           | farmers          |
| = senators                          | greyhound        | orion            | resolution        | tobacco          |
| glyph[pi1] regulators               | accident         | air              | international     | ban              |
| direct meeting                      | passengers       | union            | plo               | insurance        |
| committee                           | railroad         | airlines         | mission           | batus            |
| gray                                | passenger        | service          | assembly          | smokers          |
| k Topic 186                         | Topic 187        | Topic 188        | Topic 189         | Topic 190        |
| n k, • 346                          | 338              | 338              | 338               | 334              |
| power                               | cable            | conservatives    | water             | dental           |
| jersey assignment 50%               | nbc              | conservative     | river             | claims           |
| bradley                             | tempo            | amendment        | area              | plough           |
| = utility                           | hsn              | speaker          | reservoir         | oral             |
| direct glyph[pi1] wppss             | industry         | darman           | savannah          | counter          |
| utilities                           | subscribers      | kemp             | corps             | embassy          |
| west                                | tv               | republicans      | canyon            | mid              |
| k Topic 279                         | Topic 280        | Topic 281        | Topic 282         | Topic 283        |
| n k, • 220                          | 219              | 219              | 219               | 218              |
| fernandez                           | water            | bloom            | canadian          | election         |
| fdic                                | lake             | minnick          | lee ritalin       | grenada          |
| republicbank assignment 25% weicker | mussels neill    | walters lawyer   | murphy            | boigny houphouet |
| glyph[pi1] = virginia               | erie             | athletes         | domestic          | gairy            |
| direct ruth                         | problem          | college          | security          | coast            |
| robinson                            | plant            | suspect          | woods             | nov              |
| station                             | north            | signing          | radio             |                  |
| k Topic 354                         | Topic 355        | Topic 356        | Topic 357         | failed           |
| n k, • 133                          | 133              | 133              | 132               |                  |
|                                     |                  |                  |                   | Topic 358 132    |
| machine stop                        | young            | count forman     | reynolds          | turkey           |
| reed                                | johnston golf    | festival         | premier           | department       |
| assignment 5%                       |                  |                  | bond              | bird             |
| = gun                               | notes            | rig              | release           | cooking          |
| glyph[pi1] chief                    | bodies           | arts             | news              | wash             |
| direct sununu geneva                | homes call       | hughes lights    | regulated address | bacteria stuffed |
| formal                              | shortage         |                  |                   |                  |
|                                     |                  | staged           | petition          |                  |
|                                     |                  |                  |                   | adams            |

## D Appendix: quantile summary of topics for CGCBIB

Here we display a multi-quantile summary for CGCBIB, obtained by ranking all topics with at least 100 tokens by their total number of tokens, computing the glyph[pi1] = 100% , 75% , 50% , 25% , and 5% quantiles. We compute the five topics closest to each quantile by number of tokens, and display their top-eight words.

| Topic 1               | Topic 2           | Topic 3       | Topic 4 21 215       | Topic 5 19 832   |
|-----------------------|-------------------|---------------|----------------------|------------------|
| 110 702               | 58 811            | 27 084        | gene                 |                  |
| elegans caenorhabditi | elegans           | elegans       |                      | mutations        |
|                       | protein           | genetic       | elegans              | gene             |
| nematode              | caenorhabditi     | molecular     | sequence             | mutants          |
| results               | gene              | development   | protein              | genes            |
| found                 | function          | caenorhabditi | caenorhabditi        | mutant           |
| show                  | proteins          | nematode      | amino                | elegans          |
| observed              | required          | studies       | cdna                 | caenorhabditi    |
| specific              | show              | model         | acid                 | alleles          |
| Topic 54              | Topic 55          | Topic 56      | Topic 57             | Topic 58         |
| 2 166                 | 2 061             | 2 048         | 2 040                | 2 025            |
| germ                  | egl               | emb           | spe                  | wnt              |
| germline              | egg               | temperature   | sperm                | mom              |
| early                 | laying            | mutants       | fer                  | pop              |
| granules              | serotonin         | sensitive     | spermatozoa          | signaling        |
| cells                 | neurons           | zyg           | membrane             | bar              |
| embryos               | cat               | maternal      | spermatids           | pathway          |
| somatic               | dopamine          | expression    | spermatogenes        | lin              |
| line                  | mutants           | embryonic     | pseudopod            | wrm              |
| Topic 109             | Topic 110         | Topic 111     | Topic 112            | Topic 113        |
| 930 vit               | 916 binding       | 915 kinesin   | 900 growth           | 893 eat          |
| yolk                  | affinity          | klp           | survival             | pharyngeal       |
| vitellogenin          | site              | transport     | mortality            | pharynx          |
| genes                 | activity          | motor         | population           | pumping          |
| yp                    | sites             | ift           | rate                 | inx              |
| proteins              |                   | cilia         |                      | gap              |
|                       | avermectin        |               | populations          |                  |
| vpe                   | elegans           | dynein        | parameter            | feeding          |
| lrp                   | membrane          | movement      | size                 | junctions        |
| Topic 164             | Topic 165         | Topic 166     | Topic 167            | Topic 168        |
|                       |                   |               | vha                  | ife              |
| mlc mel               | dom effects       | innate immune | atpase               | cap              |
| myosin                | humic             | immunity      | subunit              | eife             |
| nmy                   | pyrene            | abf           | genes                | capping          |
| chain                 | effect            | lys           | vacuolar             | cel              |
| elongation            | bioconcentrat     | toll          | subunits             | gtp              |
| rho                   | dissolved         | antimicrobial | atpases              | isoforms         |
| phosphatase           | substances        | pathway       | type                 |                  |
| Topic 208             | Topic 209         | Topic 210     | Topic 211            | rna              |
| 141                   |                   |               | 140                  | Topic 212 136    |
| ubq                   | 141               | 140 da        | ion                  | hcf              |
|                       | asp               | cl            | diet                 | cehcf            |
| gc                    | salmonella        |               |                      |                  |
| tbp                   | poona             | fli           | relative             | vp               |
| footprints            | enterica          | gs            | xpa                  | ldb              |
| oscillin              | clp               | db            | groups               | cell             |
| tlf ubiquitin         | serotype necrotic | glu           | carbon characteristi | mammalian        |
|                       |                   | phospholipid  |                      | phosphorylati    |
| tata                  | mug               | tg            | atoms                | neural           |

| k Topic 1                      | Topic 2         | Topic 3              | Topic 4             | Topic 5         |
|--------------------------------|-----------------|----------------------|---------------------|-----------------|
| k, • 65 059                    | 41 005          | 33 714               | 27 221              | 22 813          |
| elegans                        | elegans         | elegans              | mutations           | elegans         |
| caenorhabditi                  | genetic         | caenorhabditi        | elegans             | gene            |
| protein assignment 100%        | caenorhabditi   | nematode             | gene                | sequence        |
| gene                           | nematode        | results              | mutants             | caenorhabditi   |
| = function                     | molecular       | observed             | genes               | protein         |
| direct glyph[pi1] proteins     | development     | high                 | caenorhabditi       | amino           |
| required                       | studies         | type                 | mutant              | cdna            |
| show                           | model           | effect               | function            | acid            |
| k Topic 68                     | Topic 69        | Topic 70             | Topic 71            | Topic 72        |
| k, • 1 921                     | 1 894           | 1 836                | 1 828               | 1 776           |
| loci                           | worm            | cell                 | alpha               | unc             |
| genetic                        | elegans         | epithelial           | gpa                 | gaba            |
| strains assignment 75%         | research        | junctions            | egl                 | receptors       |
| = lines                        | caenorhabditi   | membrane             | signaling           | receptor        |
| glyph[pi1] life                | brenner         | cells                | protein             | resistance      |
| direct mutations               | years           | dlg                  | goa                 | lev             |
| mutation                       | nematode        | hmp                  | rgs                 | levamisole      |
| inbred                         | biology         | exc                  | proteins            | cholinergic     |
| k Topic 137                    | Topic 138       | Topic 139            | Topic 140           | Topic 141       |
| k, • 782 cell                  | 779 hsp         | 779 survival         | 763 acid            | 756 yeast       |
| dimensional                    | heat            | mortality            | amino               | cerevisiae      |
| microscopy assignment 50%      | shock           | model                | acids               | saccharomyces   |
| = embryo                       | chaperone       | data                 | nematode            | pombe           |
| glyph[pi1] analysis            | small           | gompertz             | glycine             | cell            |
| direct system                  | proteins        | parameter            | briggsae            | budding         |
| computer time                  | crystallin hsps | population rate      | cytochrome multiple | schizosacchar   |
| k Topic 206                    | Topic 207       | Topic 208            | Topic 209           | cycle Topic 210 |
|                                |                 |                      | gcy                 | mediator        |
| activity                       | pgp             | telomere             |                     |                 |
| lh                             | mrp             | telomeres            | guanylyl            | med             |
| activities assignment 25%      | aat             | ceh                  | cyclase             | sop             |
| = juvenile                     | cells           | yeast                | wee                 | transcription   |
| glyph[pi1] nematodes           | glycoprotein    | nematode             | ase                 | development     |
| direct antiallatal             | mammalian       | mrt                  | receptor            | pvl             |
| hormone                        | resistance      |                      | cyclases            | transcription   |
| insect                         | glycoproteins   | telomerase telomeric | gfp                 | dhp             |
| k Topic 261                    | Topic 262       | Topic 263            | Topic 264           | Topic 265       |
| k, • 164                       | 164             | 159                  | 157                 | 156             |
| cog                            | atp             | calcineurin          | selection           | srl             |
|                                | structures      | egg                  | flow                | rol             |
| wd assignment                  | oligomerizati   |                      | separation          | threshold       |
| repeat 5%                      |                 | bovine               |                     |                 |
| glyph[pi1] = connection native | family binding  | laying hg            | redundancy flows    | ra energy       |
| direct response                | members         | white                | directional         | free            |
| worm                           | stability       | haemin               | solution            | external        |
| nr                             |                 |                      | period              |                 |
|                                | mechanism       | phosphatase          |                     | experimental    |

## E Appendix: quantile summary of topics for NEURIPS

Here we display a multi-quantile summary for NeurIPS, obtained by ranking all topics with at least 100 tokens by their total number of tokens, computing the glyph[pi1] = 100% , 75% , 50% , 25% , and 5% quantiles. We compute the five topics closest to each quantile by number of tokens, and display their top-eight words.

| k                               | Topic 2       | Topic 3              | Topic 4          | Topic 5         |
|---------------------------------|---------------|----------------------|------------------|-----------------|
| Topic 1 n k, • 182 743          | 162 355       | 129 745              | 52 356           | 44 155          |
| system                          | function      | number               | model            | training        |
| information                     | case          | result               | neural           | set             |
| approach collapsed 100%         | result        | small                | result           | data            |
| set                             | term          | values               | system           | test            |
| = problem                       | parameter     | order                | activity         | performance     |
| partially glyph[pi1] research   | neural        | large                | input            | number          |
| computer                        | form          | effect               | pattern          | result          |
| single                          | defined       | high                 | function         | error           |
| k Topic 148                     | Topic 149     | Topic 150            | Topic 151        | Topic 152       |
| n k, • 2 585                    | 2 585         | 2 574                | 2 559            | 2 549           |
| genetic                         | delay         | bengio               | fig              | matching        |
| algorithm collapsed             | bifurcation   | output               | properties       | model           |
| population 75%                  | oscillation   | dependencies         | proc             | point           |
| = fitness                       | point         | input                | step             | correspondenc   |
| glyph[pi1] string               | stability     | experiment           | range            | match           |
| partially generation            | fixed         | frasconi             | structure        | problem         |
| bit                             | limit         | term                 | calculation      | set             |
| function                        | hopf          | information          | illinois         | object          |
| k Topic 297                     | Topic 298     | Topic 299            | Topic 300        | Topic 301       |
| n k, • 1 310                    | 1 309         | 1 309                | 1 297            | 1 295           |
| vor                             | routing       | speaker              | delay            | memory          |
| storage anastasio collapsed     | load network  | recognition          | input transition | action states   |
| 50%                             |               | normalization        |                  |                 |
| glyph[pi1] = responses velocity | path packet   | male feature         | width            | sensing         |
|                                 |               |                      | window           | agent           |
| partially pan                   | traffic       | female               | connection       | loop            |
| rotation                        | shortest      | mntn                 | information      | history         |
| vestibular                      | policy        | ntn                  | temporal         | mdp             |
| n k, • 748                      | 748           | 746                  | 739              | 735             |
| composite                       | psom          | limited interconnect | tau hypothesis   | cmm speed       |
| mdp collapsed                   | robot         |                      |                  |                 |
| action 25%                      | camera        | fan                  | mansour          | particle        |
| = elemental                     | set           | shunting             | growth           | particles       |
| glyph[pi1] optimal              |               | modularity           | coefficient      | pattern         |
| partially payoff                | pointing      |                      |                  |                 |
|                                 | coordinates   | collective           | function         | presence        |
| solution                        | basis         | linear               | stem             | method          |
| k Topic 566                     | Topic 567     | Topic 568            | Topic 569        | Topic 570       |
| n k, • 396                      | 385           | 383                  | 379              | 372             |
| morph                           | minimal       | visualization        | periodic         | machine         |
| kernel collapsed                | root          | high                 | period           | capacity        |
| parent 5%                       | biases        | low                  | coefficient      | path            |
| = human                         | attribute     | diagram              | primitive        | trouble         |
| glyph[pi1] busey                | remove        | visualizing          | homogeneous      | high            |
| partially similar exemplar      | rumelhart row | graphic fund         | tst mhaskar      | task increasing |
|                                 |               |                      | chain            |                 |
| distinctivene                   | exponential   | window               |                  | measures        |

| Topic 6         | Topic 2           | Topic 1          | Topic 13             | Topic 62                        |
|-----------------|-------------------|------------------|----------------------|---------------------------------|
| 473 770         | 93 435            | 52 418           | 50 965               | 41 565                          |
| network         | network           | model            | model                | function                        |
| model           | unit              | neuron           | data                 | network                         |
| learning        | input             | input            | parameter            | bound                           |
| function        | learning          | network          | network              | dimension                       |
| input           | training          | cell             | algorithm            | learning                        |
| neural          | weight            | system           | mixture              | result                          |
| algorithm       | neural            | unit             | function             | number                          |
| set             | output            | visual           | gaussian             | set                             |
| Topic 440       | Topic 170         | Topic 334        | Topic 418            | Topic 312                       |
| 2 678           | 2 657             | 2 643            | 2 636                | 2 622                           |
| learning        | movement          | motion           | learning             | cell                            |
| critic          | visual            | unit             | algorithm            | correlation                     |
| function        | vector            | direction        | action               | neuron                          |
| actor           | image             | model            | advantage            | model                           |
| algorithm       | model             | stage            | system               | unit                            |
| system          | location          | input            | function             | interaction                     |
| control         | eye               | network          | policy               | firing                          |
| model           | map               | cell             | control              | set                             |
| Topic 378       | Topic 322         | Topic 82         | Topic 344            | Topic 414                       |
| 1 032           | 1 028             | 1 013            | 1 009                | 1 006                           |
| iiii cell       | cell              | model response   | form word            | component algorithm             |
| network         | spike unit        | neural           | phone                | sources                         |
| neural          | function          | escape           | input                |                                 |
| response        | firing            | interneuron      | network              | analysis data                   |
| model           | result            | cockroach        | system               | noise                           |
| point           | transfer          | leg              | training meaning     | orientation spatial             |
| fixed           | sorting           | input            |                      |                                 |
| Topic 220 728   | Topic 341 723     | Topic 441 723    | Topic 447 722        | Topic 308 721                   |
| aspect          | element           | network          | input                | traffic                         |
| object          | pairing grouping  | neural           | unit                 | waiting                         |
| view node       | group             | constraint       | spike                | elevator                        |
|                 | saliency          | match learn      | layer                | appeared application            |
| learning        | contour           |                  | learning             | compared                        |
| network         |                   | problem          | model                |                                 |
| weight          | computation       | initial          | predict              | department                      |
| equation        | optimal Topic 246 | row Topic 195    | prediction Topic 245 | found                           |
| Topic 259 509   |                   |                  |                      | Topic 293 503                   |
| input           | 507 network       | 506 network      | 503 network          |                                 |
|                 | neural            |                  |                      | network function                |
| output          |                   | symbol           | equation             |                                 |
| activation data | task              | vtp              | neuron               | adaptation                      |
| encoded         | link food         | learning phrases | moment neural        | algorithm prediction projection |
| function hidden | nodes output      | sentences vpp    | approximation ohira  | neural                          |
|                 |                   | classificatio    | stochastic           | training                        |
| model           | recurrent         |                  |                      |                                 |

## F Appendix: topics produced by Algorithm 2 on PUBMED

Here we show top eight words for each topic together with total number of tokens assigned, which is shown at the top of each table. We display all topics containing at least eight unique word tokens.

| Topic 1              | Topic 2 40 486         | Topic 3                  | Topic 4 166          | Topic 5 30 707 144   |
|----------------------|------------------------|--------------------------|----------------------|----------------------|
| 47 322 709           | 229                    | 34 685 122 model         | 30 795 cell          | gene                 |
| care health          | age risk               | data                     | expression           | protein              |
| patient              | children               | system                   | growth               | dna                  |
| medical              | year                   | time                     | protein              | expression           |
| research             | women                  | analysis                 | factor               | sequence             |
| clinical             | patient                | effect                   | receptor             | genes                |
| system               | factor                 | test                     | kinase               | rna                  |
| cost                 | population             | field                    | beta                 | region               |
| Topic 6              | Topic 7                | Topic 8                  | Topic 9              | Topic 10             |
| 28 510 997           | 27 277 306             | 26 709 116               | 26 408 263           | 25 200 662           |
| cell                 | cancer                 | patient                  | rat                  | cell                 |
| il                   | tumor                  | treatment                | receptor             | electron             |
| cd                   | patient                | mg                       | effect               | muscle               |
| mice                 | carcinoma              | drug                     | neuron               | tissue               |
| antigen              | cell                   | effect                   | brain                | fiber                |
| human                | breast                 | therapy                  | activity             | rat                  |
| lymphocytes          | survival               | dose                     | stimulation          | development          |
| immune               | tumour                 | day                      | induced              | microscopy           |
| Topic 11             | Topic 12               | Topic 13                 | Topic 14             | Topic 15             |
| 24 856 624           | 24 750 437             | 24 607 618               | 24 482 090           | 22 956 810           |
| patient              | patient                | blood pressure           | patient disease      | infection virus      |
| surgery complication | artery heart           | flow                     | clinical             | hiv                  |
| surgical             | coronary               | min                      | diagnosis            | strain               |
| treatment            |                        | effect                   | lesion               | infected             |
| year                 | ventricular myocardial | exercise                 | brain                | patient              |
| postoperative        | cardiac                |                          |                      |                      |
| operation            | left                   | arterial heart           | syndrome imaging     | positive viral       |
|                      |                        |                          |                      | Topic 20             |
| Topic 16 22 095 623  | Topic 17               | Topic 18                 | Topic 19             | 20 828 980           |
| ca                   | 21 838 239 structure   | 21 363 408 concentration | 20 887 061 pregnancy | protein              |
| effect               | binding                |                          |                      | binding              |
| receptor             | protein                | degrees samples          | level women          | human                |
| channel              | reaction               | liquid                   | hormone              | antibodies           |
| cell                 | acid                   | solution                 | day                  | acid                 |
| calcium              | interaction            | assay                    | infant               | antibody             |
| concentration        | compound               | detection                | fetal                | alpha                |
| na                   | site                   | system                   | concentration        | gel                  |
| Topic 21             | Topic 22               | Topic 23                 | Topic 24             | Topic 25             |
| 20 106 260           | 19 788 488             | 18 675 096               | 17 163 327           | 16 440 018           |
| rat                  | bone                   | patient                  | gene                 | activity             |
| cell                 | patient                | renal                    | mutation             | acid                 |
| effect               | joint                  | liver                    | genetic              | enzyme               |
| liver                | muscle                 | transplantati            | chromosome           | liver                |
| mice                 | fractures              | blood                    | analysis             | concentration        |
|                      |                        |                          |                      | rat                  |
| dose                 | hip year               | disease acute            | genes dna            | enzymes              |
| mg                   |                        |                          |                      |                      |
| drug                 | implant                | chronic                  | polymorphism         | synthesis            |

| Topic 26                      | Topic 27           | Topic 28                | Topic 29                                         | Topic 30           |
|-------------------------------|--------------------|-------------------------|--------------------------------------------------|--------------------|
| 16 136 164                    | 14 201 063         | 13 706 016              | 13 191 158 strain                                | 13 105 245 protein |
| effect platelet induced oxide | diet               | patient disease gastric | plant growth acid bacteria activity cell species |                    |
|                               | weight intake food |                         |                                                  | membrane cell      |
|                               |                    | asthma                  |                                                  | domain             |
| rat                           | body               | test                    |                                                  | binding            |
| cell                          | effect             | pylori                  |                                                  | receptor           |
| endothelial                   | acid               | arthritis               |                                                  | lipid              |
| activity                      | vitamin            | chronic                 |                                                  | membranes          |
| Topic 31                      | Topic 32           | Topic 33                | Topic 34                                         | Topic 35           |
| 12 705 261                    | 12 624 252         | 10 422 885              | 9 850 167                                        | 7 027 660          |
| insulin                       | species            | exposure                | skin                                             | level              |
| glucose                       | population         | concentration           | patient                                          | patient            |
| diabetes                      | infection          | iron                    | eyes                                             | ml                 |
| cholesterol                   | animal             | level                   | eye                                              | control            |
| level                         | egg                | water                   | retinal                                          | serum              |
| diabetic                      | host               | effect                  | laser                                            | plasma             |
| plasma                        | parasite           | exposed                 | visual                                           | factor             |
| lipoprotein                   | malaria            | lead                    | corneal                                          | concentration      |
| Topic 36                      | Topic 37           | Topic 38                | Topic 39                                         | Topic 40           |
| 6 130 945                     | 644 182            | 2 264                   | 1 325                                            | 104                |
| dental                        | sleep              | ppr                     | pac                                              | feather            |
| oral                          | caffeine           | csc                     | foal                                             | tieg               |
| teeth                         | tea                | stretch                 | cpr                                              | sorghum            |
| tooth                         | effect             | pthrp                   | pacap                                            | coii               |
| periodontal                   | theophylline       | response                | edm                                              | phycocyanin        |
| treatment                     | night              | br                      | speck                                            | vanx               |
| salivary                      | coffee             | gei                     | branchial                                        | midrib             |
| gland                         | green              | pth                     | lth                                              | ifi                |
| Topic 41                      |                    |                         |                                                  |                    |
| 104                           |                    |                         |                                                  |                    |
| steer                         |                    |                         |                                                  |                    |
| mca                           |                    |                         |                                                  |                    |
| persistency                   |                    |                         |                                                  |                    |
| buckwheat                     |                    |                         |                                                  |                    |
| dnak                          |                    |                         |                                                  |                    |
| eset                          |                    |                         |                                                  |                    |
| branding                      |                    |                         |                                                  |                    |
| akr                           |                    |                         |                                                  |                    |