## Beyond 1 / 2 -Approximation for Submodular Maximization on Massive Data Streams

Ashkan Norouzi-Fard * 1 Jakub Tarnawski * 1 Aida Mousavifar 1 Ola Svensson

Slobodan Mitrovi´ c * 1 Amir Zandieh * 1 1

## Abstract

Many tasks in machine learning and data mining, such as data diversification, nonparametric learning, kernel machines, clustering etc., require extracting a small but representative summary from a massive dataset. Often, such problems can be posed as maximizing a submodular set function subject to a cardinality constraint. We consider this question in the streaming setting , where elements arrive over time at a fast pace and thus we need to design an efficient, low-memory algorithm. One such method, proposed by Badanidiyuru et al. (2014), always finds a 0 . 5-approximate solution. Can this approximation factor be improved? We answer this question affirmatively by designing a new algorithm Salsa for streaming submodular maximization. It is the first low-memory, singlepass algorithm that improves the factor 0 . 5, under the natural assumption that elements arrive in a random order. We also show that this assumption is necessary, i.e., that there is no such algorithm with better than 0 . 5approximation when elements arrive in arbitrary order. Our experiments demonstrate that Salsa significantly outperforms the state of the art in applications related to exemplarbased clustering, social graph analysis, and recommender systems.

## 1. Introduction

We are experiencing an unprecedented growth in the sizes of modern datasets. Streams of data of massive

* Equal contribution 1 Theory of Computation Laboratory, EPFL, Lausanne, Vaud, Switzerland. Correspondence to: Ashkan Norouzi-Fard &lt; ashkan.afn@gmail.com &gt; .

Proceedings of the 35 th International Conference on Machine Learning , Stockholm, Sweden, PMLR 80, 2018. Copyright 2018 by the author(s).

volume are generated every second, coming from many different sources in industry and science such as: image and video streams, sensor data, social networks, stock markets, and many others. Sometimes, such data is produced so rapidly that most of it cannot even be stored in any way. In this context, a critical task is data summarization : one of extracting a representative subset of manageable size from rich, large-scale data streams. A central topic in machine learning and data mining today, its main challenge is to produce such a concise yet high-value summary while doing so efficiently and on-the-fly.

In many applications, this challenge can be viewed as optimizing a submodular function subject to a cardinality constraint. Indeed, submodularity - an intuitive notion of diminishing returns, which postulates that an element should contribute more to a smaller set than to a larger one - plays a similar role in this setting as convexity does in continuous optimization. Namely, it is general enough to model a multitude of practical scenarios, such as viral marketing (Kempe et al., 2003), recommender systems (El-Arini &amp; Guestrin, 2011), search result diversification (Agrawal et al., 2009) or active learning (Golovin &amp; Krause, 2011), while allowing for both theoretically and practically sound and efficient algorithms. In particular, a celebrated result by Nemhauser et al. (1978) shows that the Greedy algorithm - one that iteratively picks the element with the largest marginal contribution to the current summary - is a (1 -1 / e )-approximation for maximizing a monotone submodular function subject to a cardinality constraint. That is, the objective value that it attains is at least a (1 -1 / e )-fraction of the optimum. This approximation factor is NP-hard to improve (Feige, 1998). Unfortunately, Greedy requires repeated access to the complete dataset, which precludes it from use in large-scale applications in terms of both memory and running time.

The sheer bulk of large datasets and the infeasibility of Greedy together imply a growing need for faster and memory-efficient algorithms, ideally ones that can work in the streaming setting : one where input arrives one element at a time, rather than being available all at once, and only a small portion of the data can be kept in memory at any point. The first such algorithm was given by Chakrabarti and Kale (2014), yielding a 0 . 25approximation while requiring only a single pass over the data, in arbitrary order, and using O ( k ) function evaluations per element and O ( k ) memory. 1 A more accurate and efficient method Sieve-Streaming was proposed by Badanidiyuru et al. (2014). For any ε &gt; 0, it provides a (0 . 5 -ε )-approximation and uses O ( log k / ε ) function evaluations per element and O ( k log k / ε ) memory. While well-suited for use in big data stream scenarios, its approximation guarantee is nevertheless still inferior to that of Greedy . It is natural to wonder: can the ratio 0 . 5 be improved upon by a more accurate algorithm?

It turns out that in general, the answer is no (modulo the natural assumption that the submodular function is only evaluated on feasible sets). As one of our results, we show that:

Theorem 1.1. Any algorithm for streaming submodular maximization that only queries the value of the submodular function on feasible sets (i.e., sets of cardinality at most k ) and is an α -approximation for a constant α &gt; 0 . 5 must use Ω( n/k ) memory, where n is the length of the stream.

This hardness includes randomized algorithms, and applies even for estimating the optimum value to within this factor, without necessarily returning a solution (see Appendix B for the proof). 2 Note that usually n/k glyph[greatermuch] k ; such an algorithm therefore cannot run in a large-scale streaming setting.

However, this bound pertains to arbitrary-order streams. An immediate question, then, is whether inherent randomness present in the stream can be helpful in obtaining higher accuracy. Namely, in many real-world scenarios the data arrives in random order, or can be processed in random order. This can be seen as a sweet spot between assuming that the data is drawn randomly from an underlying prior distribution - which is usually unrealistic - and needing to allow for instances whose contents and order are both adversarially designed - which also do not appear in applications. Is it possible to obtain a better approximation ratio under this natural assumption?

1 We make the usual assumption that one can store any element, or the value of any set, using O (1) memory. The memory usage calculation in (Chakrabarti &amp; Kale, 2014) is lower-level, which results in an extra log n factor. Furthermore, their algorithm can be implemented using a priority queue, which would result in a runtime of O (log k ) per element.

2 Moreover, note that Theorem 1.1 does not follow from the work of Buchbinder et al. (2015), who proved an approximation hardness of 0 . 5 for online algorithms whose memory state must always be a feasible solution (consisting of at most k elements).

Again, we begin with a negative result: the performance of the state-of-the-art Sieve-Streaming algorithm remains capped at 0 . 5 in this setting.

Theorem 1.2. There exists a family of instances on which the approximation ratio of Sieve-Streaming is at most 0 . 5 + o (1) even if elements arrive in random order.

We remark that Theorem 1.2 also extends to certain natural modifications of Sieve-Streaming (with a different value of a threshold parameter used in the algorithm, or multiple such values that are tried in parallel). Thus, new ideas are required to go beyond an approximation ratio of 0 . 5.

As the main result of this paper we present a new algorithm Salsa (Streaming ALgorithm for Submodular maximization with Adaptive thresholding), which does break the 0 . 5 barrier in the random-order case. Salsa , like Sieve-Streaming , works in the streaming model and takes only a single pass over the data, selecting those elements whose marginal contribution is above some current threshold. However, it employs an adaptive thresholding scheme, where the threshold is chosen dynamically based on the objective value obtained until a certain point in the data stream. This additional power allows us to prove the following guarantee:

Theorem 1.3. [Main Theorem] There exists a constant α &gt; 0 . 5 such that, for any stream of elements that arrive in random order, the value of the solution returned by Salsa is at least α · OPT in expectation (where OPT is the value of the optimum solution). Salsa uses O ( k log k ) memory (independent of the length of the stream) and processes each element using O (log k ) evaluations of the objective function.

We remark that even if the stream is adversarial-order, Salsa still guarantees a (0 . 5 -ε )-approximation.

A different way to improve the accuracy of an algorithm is allowing it to make multiple passes over the stream. In this paper we also consider this setting and present a 2-pass algorithm Two-Pass for streaming submodular maximization. We show that Two-Pass achieves a 5 / 9 approximation ratio using the same order of memory and function evaluations as Sieve-Streaming . Formally, for any ε &gt; 0 we show that:

Theorem 1.4. Two-Pass is a ( 5 / 9 -ε ) -approximation for streaming submodular maximization. It uses O ( k log k / ε ) memory and processes each element with O ( log k / ε ) evaluations of the objective function.

Furthermore, we generalize our ideas to design a p -pass algorithm P-Pass for any constant p ≥ 2. McGregor and Vu (2016) showed that, regardless of the running time, no p -pass algorithm can beat the (1 -1 / e ) approximation guarantee using memory poly( k ). In this work we show that P-Pass quickly converges to a (1 -1 / e )-approximation as p grows. We show that:

Theorem 1.5. P-Pass is a (1 -( p / p +1 ) p -ε ) -approximation for streaming submodular maximization. It uses O ( k log k / ε ) memory and processes each element with O ( p log k / ε ) evaluations of the objective function.

Applications and experiments We assess the accuracy of our algorithms and show their versatility in several real-world scenarios. In particular, we study maximum coverage in graphs, exemplar-based clustering, and personalized movie recommendation. We find that Salsa significantly outperforms the state of the art, Sieve-Streaming , in all tested datasets. In fact, our experimental results show that Salsa reduces the gap between Greedy , which is the benchmark algorithm even for the offline setting (a 'tractable optimum'), and the best known streaming algorithm by a factor of two on average.

Note that we are able to obtain these practical improvements even though, in our experiments, the order of arrival of elements is not manually randomized. This suggests that the random-order assumption, which allows us to obtain our improved theoretical guarantees, does well in approximating the nature of real-world datasets, which are not stochastic but also not adversarial.

Related work The benchmark algorithm for monotone submodular maximization under a cardinality constraint is Greedy . Unfortunately, it is not efficient and requires k passes over the entire dataset. There has thus been much interest in obtaining more efficient versions of Greedy , such as Lazy-Greedy (Minoux, 1978; Leskovec et al., 2007; Krause et al., 2008), the algorithm of Badanidiyuru and Vondr´ ak (2014), or Stochastic-Greedy (Mirzasoleiman et al., 2015).

The first multi-pass algorithm for streaming submodular maximization has been given by Gomes and Krause (2010). If f is upper-bounded by B , then for any ε &gt; 0 their algorithm attains the value 0 . 5 · OPT -kε and uses O ( k ) memory while making O ( B/ε ) passes. Interestingly, it converges to the optimal solution for a restricted class of submodular functions.

Many different settings are considered under the streaming model. One important requirement often arising in practice is that the returned solution be robust against deletions (Mirzasoleiman et al., 2017; Mitrovi´ c et al., 2017; Kazemi et al., 2017). Non-monotone submodular functions have also been considered (Chekuri et al., 2015; Mirzasoleiman et al., 2017).

McGregor and Vu (2016) consider the k -coverage problem in the multi-pass streaming setting. They give an algorithm achieving (1 -1 / e -ε )-approximation in O (1 /ε ) passes. They also show that improving upon the ratio (1 -1 / e ) in a constant number of passes would require an almost linear memory. Their results generalize from k -coverage to submodular maximization.

In the online setting, the stream length n is unknown and the algorithm must always maintain a feasible solution. This model allows preemption, i.e., the removal of previous elements from the solution (otherwise no constant competitive ratio is possible). Chakrabarti and Kale (2014), Chekuri et al. (2015) and Buchbinder et al. (2015) have obtained 0 . 25-competitive algorithms for monotone submodular functions under a cardinality constraint. This competitive ratio was later improved to 0 . 29 by Chan et al. (2017). Buchbinder et al. (2015) also prove a hardness of 0 . 5.

A different large-scale scenario is the distributed one, where the elements are partitioned across m machines. The algorithm GreeDi (Mirzasoleiman et al., 2013) consists in running Greedy on each machine and then combining the resulting summaries on a single machine using another run of Greedy . This yields an O ( 1 / min( √ k, m ) ) -approximation. Barbosa et al. (2015) showed that when the elements are partitioned randomly , one obtains a (1 -1 / e ) / 2approximation. Mirrokni and Zadimoghaddam (2015) provide a different two-round strategy: they compute coresets of size O ( k ) and then greedily merge them, yielding a 0 . 545-approximation. The algorithm of Kumar et al. (2015) consists of a logarithmic number of rounds in the MapReduce setting and approaches the Greedy ratio. Barbosa et al. (2016) provide a general reduction that implies a (1 -1 / e -ε )-approximation in O (1 /ε ) rounds.

Korula et al. (2015) study the Submodular Welfare Maximization problem - where a set of items needs to be partitioned among agents in order to maximize social welfare, i.e., the sum of the (monotone submodular) utility functions of the agents - in the online setting. The best possible competitive ratio in general is 0 . 5. However, they show that the greedy algorithm is 0 . 505competitive if elements arrive in random order.

## 2. Preliminaries

We consider a (potentially large) collection V of n items, also called the ground set . We study the problem of maximizing a non-negative monotone submodular function f : 2 V → R + . Given two sets X,Y ⊆ V , the marginal gain of X with respect to Y is defined as

<!-- formula-not-decoded -->

which quantifies the increase in value when adding X to Y . We say that f is monotone if for any element e ∈ V and any set Y ⊆ V it holds that f ( e | Y ) ≥ 0. The function f is submodular if for any two sets X and Y such that X ⊆ Y ⊆ V and any element e ∈ V \ Y we have

<!-- formula-not-decoded -->

Throughout the paper, we assume that f is given in terms of a value oracle that computes f ( S ) for given S ⊆ V . We also assume that f is normalized , i.e. f ( ∅ ) = 0.

Submodularity under cardinality constraint

The problem of maximizing function f under cardinality constraint k is defined as selecting a set S ⊆ V with | S | ≤ k so as to maximize f ( S ). We will use O to refer to such a set S , OPT to denote f ( O ), and the name SubMax to refer to this problem.

## 3. Overview of Salsa

In this section, we present an overview of our algorithm. We also explain the main ideas and the key ingredients of its analysis. In Appendix A.4, we combine these ideas into a proof of Theorem 1.3. Throughout this section, we assume that the value OPT of an optimal solution O = { o 1 , ..., o k } is known in advance. We show how to remove this assumption using standard techniques in Appendix E.

We start by defining the notion of dense optimum solutions. We say that O is dense if there exists a set D ⊆ O of size at most k/ 100 such that f ( D ) ≥ OPT/ 10. 3 Our algorithm runs three procedures, and each procedure outputs a set of at most k elements. One of the procedures performs well in the case when O is dense. The other two approaches are designed to collect high utility when O is not dense. We run these procedures in parallel and, out of the three returned sets, we report the one attaining the highest utility. In what follows, we first describe our algorithm for the case when O is not dense.

3 In the appendix, we slightly alter the constants in the definition of a dense optimal solution.

Case: O is not dense. We present the intuition behind the algorithm under the simplifying assumption that f ( o ) = OPT /k for every o ∈ O . However, the algorithm that we state provides an approximation guarantee better than 0 . 5 in expectation for any instance that is not dense.

Over the first 0 . 1-fraction of the stream, both procedures for this case behave identically: they maintain a set of elements S ; initially, S = ∅ ; each element e from the stream is added to S if its marginal gain is at least T 1 = OPT k ( 1 / 2 + glyph[epsilon1] ), i.e.,

<!-- formula-not-decoded -->

Consider the first element o ∈ O that the procedures encounter on the stream. Since the stream is randomly ordered, o is a random element of O . Due to this, we claim that if f ( S ) is small, then it is likely that the procedures add o to S . This follows from the fact that each element of O is worth OPT /k . Namely, if f ( S ) &lt; OPT( 1 / 2 -glyph[epsilon1] ′ ), for a small constant glyph[epsilon1] ′ &gt; 0, then the average marginal contribution of the elements of O with respect to S is more than T 1 , hence it is likely that the procedures select o . By repeating the same argument we can conclude that after processing a 0 . 1-fraction of the stream, either: (1) f ( S ) is large, i.e., f ( S ) &gt; OPT( 1 / 2 -glyph[epsilon1] ′ ); or (2) the procedures have selected k/ 100 elements from O (which are worth OPT / 100).

Up to this point, both procedures for the non-dense case behaved identically. In the remaining 0 . 9-fraction of the stream, the procedure corresponding to case (1) above uses a threshold OPT k ( 1 / 2 -δ ), which is lower than T 1 . Since there are still 0 . 9 n elements left on the stream, and already after the first 0 . 1-fraction we have f ( S ) &gt; OPT( 1 / 2 -glyph[epsilon1] ′ ), it is very likely that by the end the procedure will have added enough further elements to S so that f ( S ) ≥ OPT( 1 / 2 + glyph[epsilon1] ).

In case (2) above, the procedure has already selected a set S that contains at least k/ 100 elements from O , i.e., | S ∩O| ≥ k/ 100. Now, the procedure corresponding to this case continues with the threshold T 1 = OPT k ( 1 / 2 + glyph[epsilon1] ). If by the end of the stream the procedure has selected k elements, then clearly f ( S ) ≥ OPT( 1 / 2 + glyph[epsilon1] ), since each element has marginal gain at least T 1 . Otherwise, the procedure has selected fewer than k elements. This means that the marginal gain of any element of the stream with respect to S is less than T 1 . Now we claim that f ( S ) &gt; OPT / 2. First, there are at most 99 k/ 100 elements in O \ S . Furthermore, adding each such element to the set S gives marginal gain less than T 1 . Therefore, the total benefit that the elements of O\ S give to S is at most OPT k ( 1 / 2 + glyph[epsilon1] ) · 99 k/ 100, which is less than OPT( 1 / 2 -1 / 300 ) for small enough glyph[epsilon1] , therefore

<!-- formula-not-decoded -->

and thus

<!-- formula-not-decoded -->

Case: O is dense. We now give a brief overview of the procedure that is designed for the case when O is dense. Over the first 0 . 8-fraction of the stream, the procedure uses a (high) threshold T ′ 1 = OPT k · 2. Let D ⊆ O be the dense part of O . Note that the average value of the elements of D is at least OPT k · 10, which is significantly higher than the threshold T ′ 1 .

Hence, even over the 0 . 8-fraction of the stream, the algorithm will in expectation collect some elements with large marginal gain. This, intuitively, means that the algorithm in expectation selects k ′ elements of total value significantly larger than k ′ OPT / (2 k ). This enables us to select the remaining k -k ′ elements with marginal gain below OPT / (2 k ) and still collect a set of utility larger than OPT / 2. We implement this observation by letting the algorithm use a threshold lower than OPT / (2 k ) for the remaining 0 . 2-fraction of the stream. This increases the chance that the algorithm collects k -k ′ more elements.

In what follows, we provide pseudo-codes of our three algorithms. For sake of brevity, we fix the values of constants and give the full analysis of the algorithms in Appendix A.

We begin with the dense case, presented in Algorithm 1. In the pseudo-code, C 1 , C 2 are large absolute constants and β is the fraction of the stream that we process with a high threshold.

## Algorithm 1 Dense

```
1: S := ∅ 2: for the i -th element e i on the stream do 3: if i ≤ βn and f ( e i | S ) ≥ C 1 k OPT and | S | < k then 4: S := S ∪ { e i } 5: else if i > βn and f ( e i | S ) ≥ 1 C 2 · k OPT and | S | < k then 6: S := S ∪ { e i } 7: return S
```

For the case when O is not dense, we use two algorithms as described above. The first algorithm (Algorithm 2) goes over the stream and selects any element whose marginal gain to the currently selected elements is at least OPT k ( 1 / 2 + glyph[epsilon1] ). The second algorithm (Algorithm 3) starts with the same threshold, but after passing over βn elements it decreases the threshold to OPT k ( 1 / 2 -δ ).

## Algorithm 2 Fixed Threshold

```
1: S := ∅ 2: for the i -th element e i on the stream do 3: if f ( e i | S ) ≥ OPT k ( 1 / 2 + glyph[epsilon1] ) and | S | < k then 4: S := S ∪ { e i } 5: return S
```

## Algorithm 3 High-Low Threshold

```
1: S := ∅ 2: for the i -th element e i on the stream do 3: if i ≤ βn and f ( e i | S ) ≥ OPT k ( 1 / 2 + glyph[epsilon1] ) and | S | < k then 4: S := S ∪ { e i } 5: else if i > βn and f ( e i | S ) ≥ OPT k ( 1 / 2 -δ ) and | S | < k then 6: S := S ∪ { e i } 7: return S
```

Since we do not know in advance whether the input is dense or not, we run these three algorithms in parallel and output the best solution at the end.

## 4. Two-Pass Algorithm

In this section, we describe our Two-Pass algorithm. Recall that we denote the optimum solution by O = { o 1 , . . . , o k } and we let OPT = f ( O ). Throughout this section, we assume that OPT is known. We show how to remove this assumption in Appendix E. Also, in Appendix D we present a (more general) p -pass algorithm.

Our Two-Pass algorithm (Algorithm 4) is simple: in the first pass we pick any element whose marginal gain with respect to the currently picked elements is higher than the threshold T 1 = 2 3 · OPT k . In the second pass we do the same using the threshold T 2 = 4 9 · OPT k .

Theorem 4.1. Two-Pass is a 5 / 9 -approximation for SubMax .

Proof. We prove this theorem in two cases depending on | S | . First we consider the case | S | &lt; k . For any element o ∈ O \ S we have f ( o | S i ) ≤ T 2 since we have not picked it in the second pass. Therefore

<!-- formula-not-decoded -->

## Algorithm 4 Two-Pass Algorithm

```
1: S := ∅ 2: for the i -th element e i on the stream do 3: if f ( e i | S ) ≥ 2OPT 3 k and | S | < k then 4: S := S ∪ { e i } 5: for the i -th element e i on the stream do 6: if f ( e i | S ) ≥ 4OPT 9 k and | S | < k then 7: S := S ∪ { e i } 8: return S
```

Thus

and so

<!-- formula-not-decoded -->

Therefore in this case we get the desired approximation ratio.

Now we consider the second case, i.e., | S | = k . It is clear that if we have picked k elements in the first round, then we get a 2 / 3 -approximation guarantee. Therefore assume that we picked fewer than k elements in the first round, and let S 1 denote these elements. With a similar argument as in the previous case we get that f ( S 1 ) ≥ OPT / 3. One can see that in the worst-case scenario, in the first pass we have picked k/ 2 elements with marginal gain exactly T 1 each and in the second pass we have picked k/ 2 elements with marginal gain exactly T 2 each (we present a formal proof of this statement in the appendix). Therefore we have:

<!-- formula-not-decoded -->

## 5. Empirical Evaluation

In this section, we numerically validate our theoretical findings. Namely, we compare our algorithms, Salsa and Two-Pass , with two baselines, Greedy and Sieve-Streaming . For this purpose, we consider three applications: (i) dominating sets on graphs, (ii) exemplar-based clustering, and (iii) personalized movie recommendation. In each of the experiments we find that Salsa outperforms Sieve-Streaming .

It is natural to consider the utility obtained by Greedy as a proxy for an optimum, as it is theoretically tight

<!-- formula-not-decoded -->

and difficult to beat in practice. The majority of our evaluations demonstrate that the gap between the solutions constructed by Salsa and Greedy is more than two times smaller than the gap between the solutions constructed by Sieve-Streaming and Greedy .

For each of the experiments we invoke our algorithms with the following values of the parameters: Algorithm 1 with C 1 = 10, C 2 = 0 . 2, β = 0 . 8; Algorithm 2 with ε = 1 / 6; Algorithm 3 with β = 0 . 1, ε = 0 . 05, δ = 0 . 025.

## 5.1. Maximum coverage in big graphs

Maximum coverage is a classic graph theory problem with many practical applications, including influence maximization in social networks (Kempe et al., 2015) and community detection in graphs (Fortunato &amp; Lancichinetti, 2009). The goal in this problem is to find a small subset of vertices of a graph that is connected to a large fraction of the vertices.

Maximum coverage can be cast as maximization of a submodular function subject to a cardinality constraint. More formally, we are given a graph G = ( V, E ), where n = | V | denotes the number of vertices and m = | E | denotes the number of edges. The goal is to find a set S ⊆ V of size k that maximizes the number of vertices in the neighborhood of S . 4 We consider three graphs for this problem from the SNAP data library (Leskovec &amp; Krevl, 2014).

Pokec social network Pokec is the most popular online social network in Slovakia. This graph has n = 1 , 632 , 803 and m = 30 , 622 , 564.

LiveJournal social network LiveJournal (Backstrom et al., 2006) is a free online community that enables members to maintain journals and individual and/or group blogs. This graph has n = 4 , 847 , 571 and m = 68 , 993 , 773.

Orkut social network Similar to Pokec, Orkut (Yang &amp; Leskovec, 2015) is also an online social network. This graph has n = 3 , 072 , 441 vertices and m = 117 , 185 , 083 edges.

We compare our algorithms, Salsa and Two-Pass , with both baselines on these datasets for different values of k - from 100 to 10 , 000. The results show that Salsa always outperforms Sieve-Streaming by around 10%, and also reduces the gap between Greedy and the best streaming algorithm by a factor of two. Furthermore, the performance of our Two-Pass algorithm is very Objective value close to that of Greedy . The results can be found in Figure 1, where (a) and (b) correspond to the Orkut dataset, (c) and (d) correspond to LiveJournal, and (e) to Pokec.

4 This problem has been also referred to as the dominating set problem in the literature.

Figure 1: Numerical comparisons of our two algorithms ( Salsa and Two-Pass ) and baselines ( Greedy and Sieve-Streaming ). In plot (b) we could not run Greedy on the underlying dataset due to its prohibitively slow running time on this instance. Each plot demonstrates the performance of the algorithms for varying values of the cardinality k . The datasets used for plots (a)-(e) are described in Section 5.1, for plots (f) and (g) in Section 5.2, and for plots (h) and (i) in Section 5.3.

<!-- image -->

## 5.2. Exemplar-based clustering

Imagine that we are given a collection of emails labeled as spam or non-spam and asked to design a spam classifier. In addition, every email is equipped with an m -dimensional vector corresponding to the features of that email. One possible approach is to view these m -dimensional vectors as points in the Euclidean space, decompose them into k clusters and fix a representative point for each cluster. Then, whenever a new email arrives, it is assigned the same label as the cluster representative closest to it. Let V denote the set of all the labeled emails. To obtain the described set of cluster representatives, we maximize the following submodular function:

<!-- formula-not-decoded -->

where e 0 is the all-zero vector, and L ( S ) is defined as follows (Gomes &amp; Krause, 2010):

<!-- formula-not-decoded -->

In the definition of the function L ( S ), d ( x, y ) = ‖ x -y ‖ 2 denotes the squared Euclidean distance. 5

Similarly to spam classification, and among many other applications, the exemplar submodular function can also be used for image clustering. In light of these applications, we perform experiments on two datasets:

- Spambase This dataset consists of 4 , 601 emails, each email described by 57 attributes (Lichman, 2013). We do not consider mail-label as one of the attributes.

CIFAR-10 This dataset consists of 50 , 000 color images, each of size 32 × 32, divided into 10 classes. Each image is represented as a 3 , 072-dimensional vector - three coordinates corresponding to the red, green and blue channels of each pixel (Krizhevsky et al., 2014).

Before running these experiments, we subtract the mean of the corresponding dataset from each data point.

The results for the Spambase dataset are shown in Figure 1(f). We can observe that both of our algorithms attain a significantly higher utility than Sieve-Streaming . Also, at their point of saturation, our algorithms equalize with Greedy . We can also observe that Sieve-Streaming saturates at a much lower value than our algorithms, which suggests that the strategy we develop filters elements from the stream more carefully than Sieve-Streaming does.

Our results for the CIFAR-10 dataset, depicted in Figure 1(g), show that, before the point of saturation our algorithms select elements of around 5% higher utility than Sieve-Streaming . After the point of saturation our algorithms achieve the same utility as Greedy , while Sieve-Streaming approaches that value slowly. Saturation happens around k = 10, which is expected since the images in CIFAR-10 are decomposed into 10 classes.

5 Notice that we turn a minimization problem over L ( S ) into a maximization problem over f ( S ). The approximation guarantee for maximizing f ( S ) does not transfer to an approximation guarantee for minimizing L ( S ). Nevertheless, maximizing f ( S ) gives very good practical performance, and hence we use it in place of L ( S ).

## 5.3. Personalized movie recommendation

We use the Movielens 1M dataset (Harper &amp; Konstan, 2016) to build a recommender system for movies. The dataset contains over a million ratings for 3,900 movies by 6,040 users. For a given user u and a number k , the system should recommend a collection of k movies personalized for user u .

We use the scoring function proposed by Mitrovi´ c et al. (2017). We first compute low-rank feature vectors w u ∈ R 20 for each user u and v m ∈ R 20 for each movie m . These are obtained via low-rank matrix completion (Troyanskaya et al., 2001) so as to make each inner product 〈 w u , v m 〉 approximate the rating of m by u , if known. Now we define the submodular function

<!-- formula-not-decoded -->

The first term is a facility-location objective (Lindgren et al., 2016) that measures how well S covers the space M of all movies (thus promoting diversity). The second term aggregates the user-dependent scores of items in S . The parameter α can be adjusted depending on the user's preferences.

Our experiments consist in recommending collections of movies for α = 0 . 75 and values of k up to 60 (see Figure 1(h)), as well as for α = 0 . 85 and values of k up to 200 (see Figure 1(i)). We do this for 8 randomly selected users and report the averages. We find that the performance of both Salsa and Two-Pass falls at around 40% of the gap between Sieve-Streaming and Greedy . This quantity improves as k increases.

## 6. Conclusion

In this paper, we consider the monotone submodular maximization problem subject to a cardinality constraint. For the case of adversarial-order streams, we show that a 1 / 2 approximation guarantee is tight. Motivated by real-world applications, we also study this problem in random-order streams. We show that the previously known techniques are not sufficient to improve upon 1 / 2 even in this setting. We design a novel approach that exploits randomness of the stream and achieves a better-than1 / 2 approximation guarantee.

We also present a multi-pass algorithm that approaches (1 -1 / e )-approximation using only a constant number of passes, even in adversarial-order streams. We validate the performance of our algorithm on real-world data. Our evaluations demonstrate that we outperform the state of the art Sieve-Streaming algorithm by a considerable margin. In fact, our results are closer to Greedy than to Sieve-Streaming . Although we make a substantial progress in the context of streaming submodular maximization, there is still a gap between our approximation guarantee and the currently best known lower bound. It would be very interesting to reduce (or close) this gap, and we hope that our techniques will provide insight in this direction.

## Acknowledgements

We thank the anonymous reviewers for their valuable feedback. Ola Svensson and Jakub Tarnawski were supported by ERC Starting Grant 335288-OptApprox.

## References

- Agrawal, R., Gollapudi, S., Halverson, A., and Ieong, S. Diversifying search results. In Proceedings of the Second ACM International Conference on Web Search and Data Mining , WSDM '09, pp. 5-14, New York, NY, USA, 2009. ACM.
- Backstrom, L., Huttenlocher, D., Kleinberg, J., and Lan, X. Group formation in large social networks: Membership, growth, and evolution. In Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining , KDD '06, pp. 44-54, New York, NY, USA, 2006. ACM.
- Badanidiyuru, A. and Vondr´ ak, J. Fast algorithms for maximizing submodular functions. In Proceedings of the Twenty-fifth Annual ACM-SIAM Symposium on Discrete Algorithms , SODA '14, pp. 1497-1514, Philadelphia, PA, USA, 2014. Society for Industrial and Applied Mathematics.
- Badanidiyuru, A., Mirzasoleiman, B., Karbasi, A., and Krause, A. Streaming submodular maximization: Massive data summarization on the fly. In Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining , KDD '14, pp. 671-680, New York, NY, USA, 2014. ACM.
- Bar-Yossef, Z., Jayram, T. S., Kumar, R., and Sivakumar, D. Information theory methods in communication complexity. In Computational Complexity, 2002. Proceedings. 17th IEEE Annual Conference on , pp. 93-102. IEEE, 2002.

Barbosa, R., Ene, A., Nguyen, H., and Ward, J. The power of randomization: Distributed submodular maximization on massive datasets. In International Conference on Machine Learning , pp. 1236-1244, 2015.

- Barbosa, R. D. P., Ene, A., Nguyen, H. L., and Ward, J. A new framework for distributed submodular maximization. In 2016 IEEE 57th Annual Symposium on Foundations of Computer Science (FOCS) , pp. 645-654, Oct 2016. doi: 10.1109/FOCS.2016.74.

Buchbinder, N., Feldman, M., and Schwartz, R. Online submodular maximization with preemption. In Proceedings of the Twenty-sixth Annual ACM-SIAM Symposium on Discrete Algorithms , SODA '15, pp. 1202-1216, Philadelphia, PA, USA, 2015. Society for Industrial and Applied Mathematics.

- Chakrabarti, A. and Kale, S. Submodular maximization meets streaming: Matchings, matroids, and more. In Lee, J. and Vygen, J. (eds.), Integer Programming and Combinatorial Optimization , pp. 210221, Cham, 2014. Springer International Publishing.
- Chan, T.-H. H., Huang, Z., Jiang, S. H.-C., Kang, N., and Tang, Z. G. Online submodular maximization with free disposal: Randomization beats 1/4 for partition matroids. In Proceedings of the TwentyEighth Annual ACM-SIAM Symposium on Discrete Algorithms , SODA '17, pp. 1204-1223, Philadelphia, PA, USA, 2017. Society for Industrial and Applied Mathematics.
- Chekuri, C., Gupta, S., and Quanrud, K. Streaming algorithms for submodular function maximization. In Halld´ orsson, M. M., Iwama, K., Kobayashi, N., and Speckmann, B. (eds.), Automata, Languages, and Programming , pp. 318-330, Berlin, Heidelberg, 2015. Springer Berlin Heidelberg.
- El-Arini, K. and Guestrin, C. Beyond keyword search: Discovering relevant scientific literature. In Proceedings of the 17th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining , KDD '11, pp. 439-447, New York, NY, USA, 2011. ACM.
- Feige, U. A threshold of ln n for approximating set cover. Journal of the ACM (JACM) , 45(4):634-652, 1998.
- Fortunato, S. and Lancichinetti, A. Community detection algorithms: a comparative analysis: invited presentation, extended abstract. In 4th International Conference on Performance Evaluation Methodologies and Tools, VALUETOOLS '09, Pisa, Italy, October 20-22, 2009 , pp. 27, 2009.

- Golovin, D. and Krause, A. Adaptive submodularity: Theory and applications in active learning and stochastic optimization. J. Artif. Int. Res. , 42(1): 427-486, September 2011. ISSN 1076-9757.
- Gomes, R. and Krause, A. Budgeted nonparametric learning from data streams. In In Proc. International Conference on Machine Learning (ICML , 2010.
- Harper, F. M. and Konstan, J. A. The MovieLens datasets: History and context. ACM Transactions on Interactive Intelligent Systems (TiiS) , 5(4):19, 2016.
- Jayram, T. S., Kumar, R., and Sivakumar, D. The oneway communication complexity of hamming distance. Theory of Computing , 4(1):129-135, 2008.
- Kazemi, E., Zadimoghaddam, M., and Karbasi, A. Deletion-Robust Submodular Maximization at Scale. ArXiv e-prints , November 2017.
- Kempe, D., Kleinberg, J., and Tardos, E. Maximizing the spread of influence through a social network. In Proceedings of the Ninth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining , KDD '03, pp. 137-146, New York, NY, USA, 2003. ACM.
- Kempe, D., Kleinberg, J. M., and Tardos, ´ E. Maximizing the spread of influence through a social network. Theory of Computing , 11:105-147, 2015.
- Korula, N., Mirrokni, V., and Zadimoghaddam, M. Online submodular welfare maximization: Greedy beats 1/2 in random order. In Proceedings of the Forty-seventh Annual ACM Symposium on Theory of Computing , STOC '15, pp. 889-898, New York, NY, USA, 2015. ACM. ISBN 978-1-4503-3536-2. doi: 10.1145/2746539.2746626. URL http://doi.acm. org/10.1145/2746539.2746626 .
- Krause, A., Singh, A., and Guestrin, C. Near-optimal sensor placements in gaussian processes: Theory, efficient algorithms and empirical studies. J. Mach. Learn. Res. , 9:235-284, June 2008.
- Krizhevsky, A., Nair, V., and Hinton, G. The cifar-10 dataset. online: http://www. cs. toronto. edu/kriz/cifar. html , 2014.
- Kumar, R., Moseley, B., Vassilvitskii, S., and Vattani, A. Fast greedy algorithms in mapreduce and streaming. ACM Trans. Parallel Comput. , 2(3):14:1-14:22, September 2015.
- Leskovec, J. and Krevl, A. SNAP Datasets: Stanford large network dataset collection, June 2014.
- Leskovec, J., Krause, A., Guestrin, C., Faloutsos, C., VanBriesen, J., and Glance, N. Cost-effective outbreak detection in networks. In Proceedings of the 13th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining , KDD '07, pp. 420-429, New York, NY, USA, 2007. ACM.
- Lichman, M. UCI machine learning repository, 2013.
- Lindgren, E., Wu, S., and Dimakis, A. G. Leveraging sparsity for efficient submodular data summarization. In Advances in Neural Information Processing Systems , pp. 3414-3422, 2016.
- McGregor, A. and Vu, H. T. Better streaming algorithms for the maximum coverage problem. arXiv preprint arXiv:1610.06199 , 2016.
- Minoux, M. Accelerated greedy algorithms for maximizing submodular set functions. In Stoer, J. (ed.), Optimization Techniques , pp. 234-243, Berlin, Heidelberg, 1978. Springer Berlin Heidelberg.
- Mirrokni, V. and Zadimoghaddam, M. Randomized composable core-sets for distributed submodular maximization. In Proceedings of the Forty-seventh Annual ACM Symposium on Theory of Computing , STOC '15, pp. 153-162, New York, NY, USA, 2015. ACM.
- Mirzasoleiman, B., Karbasi, A., Sarkar, R., and Krause, A. Distributed submodular maximization: Identifying representative elements in massive data. In Advances in Neural Information Processing Systems , pp. 2049-2057, 2013.
- Mirzasoleiman, B., Badanidiyuru, A., Karbasi, A., Vondr´ ak, J., and Krause, A. Lazier than lazy greedy. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence , AAAI'15, pp. 1812-1818. AAAI Press, 2015.
- Mirzasoleiman, B., Jegelka, S., and Krause, A. Streaming Non-monotone Submodular Maximization: Personalized Video Summarization on the Fly. ArXiv e-prints , June 2017.
- Mirzasoleiman, B., Karbasi, A., and Krause, A. Deletion-robust submodular maximization: Data summarization with 'the right to be forgotten'. In Precup, D. and Teh, Y. W. (eds.), Proceedings of the 34th International Conference on Machine Learning , volume 70 of Proceedings of Machine Learning Research , pp. 2449-2458, International Convention Centre, Sydney, Australia, 06-11 Aug 2017. PMLR.

- Mitrovi´ c, S., Bogunovi´ c, I., Norouzi-Fard, A., Tarnawski, J., and Cevher, V. Streaming robust submodular maximization: A partitioned thresholding approach. In Advances in Neural Information Processing Systems , 2017.
- Nemhauser, G. L., Wolsey, L. A., and Fisher, M. L. An analysis of approximations for maximizing submodular set functions-i. Mathematical Programming , 14 (1):265-294, 1978.
- Troyanskaya, O., Cantor, M., Sherlock, G., Brown, P., Hastie, T., Tibshirani, R., Botstein, D., and Altman, R. B. Missing value estimation methods for DNA microarrays. Bioinformatics , 17(6):520-525, 2001.
- Yang, J. and Leskovec, J. Defining and evaluating network communities based on ground-truth. Knowl. Inf. Syst. , 42(1):181-213, 2015.

## A. Analysis of the Algorithm

In this section we analyze our algorithms and present the proof of Theorem 1.3. Throughout this section, we assume that the value OPT of the optimum solution is known. We remove this assumption in Appendix E.

We run three procedures in parallel and return the best of those as our solution. Algorithm 1 works well for instances containing a dense set (see Definition A.1). The other two algorithms (Algorithms 2 and 3) guarantee that we attain a high-utility solution in the absence of the density assumption. We prove the correctness of the stated algorithms under the assumption that k &gt; k 0 = 2 · 10 8 , where k 0 is a constant. In Appendix A.3 we introduce Algorithm 5, which completes the proof for the case when k is small.

Theorem 1.3. [Main Theorem] There exists a constant α &gt; 0 . 5 such that, for any stream of elements that arrive in random order, the value of the solution returned by Salsa is at least α · OPT in expectation (where OPT is the value of the optimum solution). Salsa uses O ( k log k ) memory (independent of the length of the stream) and processes each element using O (log k ) evaluations of the objective function.

## A.1. The dense case

In this section we analyse the correctness of Algorithm 1 under the assumption that k &gt; k 0 = 2 · 10 8 . Let us first define a dense set.

Definition A.1. We say that a set D of elements is dense if it has | D | ≤ ηk and f ( D ) ≥ 1 -γ 2 OPT , where we set γ = 10 -2 and η = 5 · 10 -5 .

This section is devoted to the proof of the following theorem.

Theorem A.2. There is an algorithm giving a 0 . 50025 -approximation with probability at least 0 . 01 for instances containing a dense set D and having k ≥ k 0 .

Consider Algorithm 1 with the following values of the parameters: C 1 = 100, C 2 = 10, β = 0 . 9. In the first 90% of the stream we collect elements of marginal value larger than 100 k OPT, and in the remaining 10% of the stream we collect elements of marginal value larger than 1 10 k OPT. Intuitively, we expect to see 90% of the elements of D in the first 90% of the stream, therefore, with a high enough threshold, we can pick almost all of those elements. In the remaining 10% of the stream we enhance our solution using a smaller threshold.

Let S be the set of elements collected by this algorithm, D be a dense set, and O be an optimum solution. For any set of elements T , define T L and T R to be the elements of T in the left part (90%) of the stream and in the right part (10%), respectively.

Fact A.3. With probability at least 0 . 01 we have all of the following:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We apply a standard Chernoff bound on the indicator variables of the elements of D being in the left part, which are negatively correlated. We have 0 ≤ | D | ≤ ηk , with | D | = ηk being the worst case for this bound. We obtain that

<!-- formula-not-decoded -->

where we used that η = 5 · 10 -5 and k ≥ k 0 = 2 · 10 8 .

For (3), we use a Chernoff bound again to show that

<!-- formula-not-decoded -->

Proof. First let us show that For (2), we will first prove that E [ f ( D L )] ≥ 0 . 9 f ( D ). To that end, let D = { e 1 , ..., e | D | } and define Z i = ✶ e i ∈ D L . Note that

<!-- formula-not-decoded -->

Now applying Markov's inequality (and using that f ( D L ) ≤ f ( D )) yields P [ f ( D L ) &lt; 0 . 85 f ( D )] ≤ 2 3 .

For (4), define the submodular function g ( · ) = f ( ·| D ) for brevity. We are interested in lower-bounding the quantity P [(4) | (2)] = P [ g ( O R ) ≥ 0 . 05 g ( O ) | f ( D L ) ≥ 0 . 85 f ( D )]. For this, notice that events (2) and (4) are almost independent (they would be independent in the limit n → ∞ ), and so we can essentially repeat the analysis for (2). Formally, let I be a random variable holding all information about the locations of elements of D in the stream. (The event (2), as well as the variable | D L | , are known given I .) We would like to prove a lower bound on E [ g ( O R ) | I ]. This is done as above, with the difference that we set O = { e 1 , ..., e k } and Z i = ✶ e i ∈O R . Now, for any e i ∈ O \ D we can bound

<!-- formula-not-decoded -->

using that k ≤ n and η ≤ 10 -2 . On the other hand, for any e i ∈ O ∩ D we have g ( e i | e 1 , ..., e i -1 ) = g ( e i ) = 0. Thus we get that E [ g ( O R ) | I ] ≥ 0 . 09 g ( O ) and

<!-- formula-not-decoded -->

Applying Markov's inequality yields that P [ g ( O R ) &lt; 0 . 05 g ( O ) | (2)] ≤ 0 . 91 0 . 95 &lt; 0 . 96. Finally, we get

<!-- formula-not-decoded -->

Lemma A.4. Assume that the events (1) , (2) , (3) , (4) happen. Then we have f ( S ) ≥ 0 . 50025 · OPT .

Proof. We consider two cases: | S | &lt; k and | S | = k .

Case | S | &lt; k : First, by the design of the algorithm, for each e ∈ D we have

<!-- formula-not-decoded -->

regardless of whether e appears in the left or in the right part of the stream. Since | D | ≤ ηk , this implies

<!-- formula-not-decoded -->

Moreover, we have (4), i.e., that f ( O R | D ) ≥ 0 . 05 f ( O| D ). Hence

<!-- formula-not-decoded -->

Also, by the design of the algorithm, for every e ∈ O R it holds that

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, by (3),

and therefore

Therefore,

Hence,

<!-- formula-not-decoded -->

Case | S | = k : Note that if | S L | ≥ k 100 , then f ( S L ) ≥ OPT and we are done. Thus for each e ∈ D L we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Hence, by (1),

Therefore, using (2), Recall that | S L | &lt; k 100 , so that | S R | ≥ 0 . 99 k . We can write

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Fact A.3 and Lemma A.4 together imply Theorem A.2.

## A.2. General case

In this section we analyze the correctness of Algorithms 2 and 3 under the assumption that k &gt; k 0 = 2 · 10 8 .

We invoke Algorithm 2 with the threshold value of ( 1+10 -8 2 ) ( OPT k ) , and Algorithm 3 with the value β = 10 -3 , the threshold value of ( 1+10 -8 2 ) ( OPT k ) for the first β -fraction of the stream, and the threshold value ( 1 -3 · 10 -11 2 ) ( OPT k ) for the remaining fraction.

Let glyph[epsilon1] = 10 -8 and δ = 3 · 10 -11 . We partition the stream into two parts: the left part containing the first β -fraction of the arriving elements and the right part containing the remaining (1 -β )-fraction. As presented in Section 3, both Algorithms 2 and 3 act in the same way on the left part of the stream (if the arriving element e satisfies f ( e | S ) ≥ ( 1+ glyph[epsilon1] 2 ) ( OPT k ) , they add it to S ). However, for the right part of the stream, they proceed with two different strategies. Algorithm 3 works well when the elements selected in the left part carry a lot of value more precisely, when the value of the left part of the solution is at least α OPT, where we select α = 3 · 10 -3 . Algorithm 2 works well in the converse case.

Let O = { o 1 , . . . , o k } denote the optimal solution. Moreover, for any set of elements T , define T L and T R to be the elements of T in the left and in the right part of the stream, respectively.

Claim 1. We have

with probability at least ≥ 0 . 999 .

Proof. We use a standard Chernoff bound for negatively correlated variables and get that

<!-- formula-not-decoded -->

where we used that k ≥ k 0 = 2 · 10 8 and β = 10 -3 .

We now analyze Algorithm 3. We do not use any randomness here (beyond assuming (5)). So fix a random arrival. Recall that S L and S R are the elements selected in the left and the right part of the stream, respectively.

Lemma A.5. Assume that f ( S L ) ≥ α OPT and that (5) is satisfied. Then Algorithm 3 outputs a set S such that f ( S ) ≥ ( 0 . 5 + 9 · 10 -12 ) OPT .

Proof. We divide the proof into two cases based on the cardinality of the output set S .

Case | S | &lt; k : In this case we do not need to use the assumption f ( S L ) ≥ α OPT. We have f ( S ∪ O ) ≥ OPT. And f ( S ) ≥ f ( S ∪ O ) -∑ k i =1 f ( o i | S ). Now, by the definition of the algorithm, we have

<!-- formula-not-decoded -->

and

<!-- formula-not-decoded -->

Hence, using that |O L | ≤ 0 . 11 k , we have

<!-- formula-not-decoded -->

where we used that β = 10 -3 , ε = 10 -8 and δ = 3 · 10 -11 .

Case | S | = k : In order to minimize f ( S ) we select

<!-- formula-not-decoded -->

elements in the right part of the stream, each of value ( 1 -δ 2 ) ( OPT k ) . This yields the following lower bound on f ( S ):

<!-- formula-not-decoded -->

where we used that α = 3 · 10 -3 , ε = 10 -8 and δ = 3 · 10 -11 .

Theorem A.6. If there is no dense subset (see Definition A.1) and if P [ f ( S L ) ≤ α OPT] ≥ 0 . 99 , then Algorithm 2 returns a set S that has

with probability at least 0 . 49 η .

Proof. Let X i be the indicator random variable for the event that the i -th arriving element of O is added to S . The following is our main technical lemma:

<!-- formula-not-decoded -->

Proof. Denote by S &lt;i the elements selected by Algorithm 2 up to (but excluding) the arrival of the i -th element of O . We have

<!-- formula-not-decoded -->

Note that

<!-- formula-not-decoded -->

by the assumption of Theorem A.6 and by Claim 1 (note that if (5) holds, then S &lt;i ⊆ S L for i ≤ 0 . 9 βk ). So

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Fix 1 ≤ i ≤ 0 . 9 βk , and let I be the random variable that denotes the position in the stream of the i -th element of O as well as the contents of the stream up to (but excluding) that position. (Note that S &lt;i is known given I .) Conditioning on I , we have

<!-- formula-not-decoded -->

and we proceed to bound the inner expectation for any fixed I such that f ( S &lt;i ) ≤ α OPT. We apply total expectation again, this time over o i :

<!-- formula-not-decoded -->

Here the first factor is not random: X i = 1 iff f ( o i | S &lt;i ) ≥ 1+ ε 2 · OPT k , and we know both S &lt;i (from I ) and o i = o . Denote the set of good elements by G = { o ∈ O : f ( o | S &lt;i ) ≥ 1+ ε 2 · OPT k } . Then the first factor is just o ∈ G .

✶ Now consider the second factor. We claim that the distribution for o i given I and (5) is uniform on the elements of O that have not yet appeared on the stream. This is because the global, uniformly random choice for the order of all elements in the stream can be broken up into three independent choices: the positions of elements of O , the relative order of elements of O , and the relative order of elements of V \ O . Conditioning on (5) only affects the first part, and together with I it reveals no information about the order of the yet-unseen elements of O . Thus the second factor, i.e., P [ o i = o | I, (5)], is equal to 0 for those elements of O that have appeared before the i -th, and 1 / ( k +1 -i ) for the others. Thus

<!-- formula-not-decoded -->

However, no element o ∈ G could have appeared yet! For suppose otherwise: since o has marginal contribution at least 1+ ε 2 · OPT k for S &lt;i , a fortiori it had at least that marginal contribution at the time when it appeared, so o should have been taken (note that | S &lt;i | &lt; k , otherwise we could not have f ( S &lt;i ) ≤ α OPT); but then o ∈ S &lt;i and thus f ( o | S &lt;i ) = 0, a contradiction. Finally we get

<!-- formula-not-decoded -->

Claim 2. We have | G | ≥ ηk (recall that η = 5 · 10 -5 - see Definition A.1).

Proof. Suppose otherwise. Then we have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and thus

yielding that

<!-- formula-not-decoded -->

(recall that α = 3 · 10 -3 , ε = 10 -8 , and γ = 10 -2 - see Definition A.1).

Together with | G | &lt; ηk , this yields that G is a dense set, whose existence contradicts the assumption of Theorem A.6.

By Equation (6) and Claim 2 we get

and finally

Furthermore, we have:

Lemma A.9. If | S ∩ O| ≥ 2 εk , then f ( S ) ≥ 1+ ε 2 OPT .

Proof. If | S | = k , then we have f ( S ) ≥ 1+ ε 2 OPT since every added element had at least 1+ ε 2 · OPT k contribution. On the other hand, if | S | &lt; k , then for every element e we have f ( e | S ) &lt; 1+ ε 2 · OPT k . Also, by assumption, |O \ S | ≤ (1 -2 ε ) k . Thus

<!-- formula-not-decoded -->

which yields f ( S ) ≥ 1+ ε 2 OPT.

Equation (7) and Lemma A.9 finish the proof of Theorem A.6: we have 0 . 49 η · 0 . 9 βk &gt; 2 · 10 -8 k = 2 εk (where we used that η = 5 · 10 -5 , β = 10 -3 , and ε = 10 -8 ), and so we get f ( S ) ≥ 1+ ε 2 OPT with at least a constant (0 . 49 η ) probability.

## A.3. Smallk case

In this section we describe an algorithm that gives a ( 1 2 +Ω(1) ) -approximation for bounded k , i.e., k &lt; k 0 . Recall that k 0 = 2 · 10 8 . We will prove:

Theorem A.10. There is an algorithm (Algorithm 5) for streaming submodular maximization in the random order case that, for any k , achieves a ( 1 2 + g ( k ) ) -approximation in expectation, for some function g ( k ) &gt; 0 .

In particular, for k ≤ k 0 Algorithm 5 yields a ( 1 2 +Ω(1) ) -approximation in expectation. The proof relies on two claims: Fact A.11 and Fact A.12.

Fact A.11. With probability at least 1 k ! we will have | S | = k at the end.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Having that E [ ∑ 0 . 9 βk i =1 X i ] ≥ 0 . 98 η · 0 . 9 βk by Lemma A.7, now we use the following fact:

.

Fact A.8. Let glyph[lscript] ≥ 0 and X be a random variable with 0 ≤ X ≤ glyph[lscript] and E [ X ] ≥ ζglyph[lscript] . Then P [ X ≥ ζglyph[lscript]/ 2] ≥ ζ/ 2

Proof. One applies Markov's inequality to the random variable glyph[lscript] -X .

Applying Fact A.8 (to X = ∑ 0 . 9 βk i =1 X i , glyph[lscript] = 0 . 9 βk , and ζ = 0 . 98 η ) we get

<!-- formula-not-decoded -->

## Algorithm 5 The smallk case

- 1: S := ∅
- 2: for the i -th element e i on the stream do
- 3: if f ( e i | S ) ≥ OPT -f ( S ) k and | S | &lt; k then
- 4: S := S ∪ { e i }
- 5: return S

The intuitive reason for this is that, whenever the algorithm takes a new element and changes its threshold, there is some element o of O that is above that threshold. With positive probability, o is the next element of O on the stream. As long as | S | &lt; k before o is seen, o will be taken. In this way, the algorithm takes all elements of O that it sees before it has collected k elements. Given that there are k elements of O , the algorithm cannot finish with | S | &lt; k .

Proof. Consider the optimum set O = { o 1 , ..., o k } , in the order that these elements appear on the stream. For i = 0 , ..., k , let E i be the event that either o 1 , ..., o i ∈ S , or S is already full at the time o i arrives. Now it is enough to prove that for i = 1 , ..., k we have P [ E i |E i -1 ] ≥ 1 k +1 -i . Once we have this, we can write

<!-- formula-not-decoded -->

So fix i . We want to show that P [ E i |E i -1 ] ≥ 1 k +1 -i . Assuming E i -1 , there are two cases: either S is already full at the time o i -1 arrives, or we have o 1 , ..., o i -1 ∈ S . In the former case, S is of course still full when o i arrives, and so E i holds. So assume the latter case. Let S &lt;i glyph[owner] o 1 , ..., o i -1 be the contents of S at the time just before the arrival of o i . We have

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and of course o ∈ { o i , ..., o k } since all previous elements of O are in S &lt;i and thus have marginal value 0. Note that at this time, conditioning on the entire stream before o i and on the knowledge that the next element will belong to O , the distribution of o i is uniform on the elements of O that have not arrived yet. Thus we have that o = o i with probability 1 k +1 -i . If o = o i , then our algorithm will indeed pick o i (unless | S &lt;i | = k , in which case E i also holds). This shows that P [ E i |E i -1 ] ≥ 1 k +1 -i .

<!-- formula-not-decoded -->

The proof is similar to the analysis of the (non-streaming) algorithm Greedy .

Proof. Let e 1 , ..., e k be the elements of S , in order of insertion. We show by induction on i = 0 , 1 , ..., k that

<!-- formula-not-decoded -->

The base case i = 0 is trivial. Fix i ≥ 1. The algorithm guarantees that f ( e i | e 1 , ..., e i -1 ) ≥ OPT -f ( e 1 ,...,e i -1 ) k . Thus we have

<!-- formula-not-decoded -->

so there exists o ∈ O with For i = k , we get

<!-- formula-not-decoded -->

Proof of Theorem A.10. We run Algorithm 5 and Sieve-Streaming in parallel, choosing the better of the two solutions at the end. We always get at least a 1 / 2-approximation, and (by Fact A.11 and Fact A.12) with probability at least 1 k ! we get at least a (1 -1 /e )-approximation. Thus we get at least a ( 1 2 + 1 k ! ( 1 2 -1 e )) -approximation in expectation.

## A.4. Proof of the main theorem

Now we are ready to complete the proof of Theorem 1.3.

Theorem 1.3. [Main Theorem] There exists a constant α &gt; 0 . 5 such that, for any stream of elements that arrive in random order, the value of the solution returned by Salsa is at least α · OPT in expectation (where OPT is the value of the optimum solution). Salsa uses O ( k log k ) memory (independent of the length of the stream) and processes each element using O (log k ) evaluations of the objective function.

Proof. If k &lt; k 0 = 2 · 10 8 , then, by Theorem A.10, Algorithm 5 achieves a ( 1 2 + 1 k ! ( 1 2 -1 e )) -approximation in expectation. Otherwise, as explained above, we run three algorithms in parallel with Sieve-Streaming and output the best solution out of the four. Throughout, we let S 1 , S 2 , and S 3 be the solutions returned by Algorithm 1, Algorithm 2, and Algorithm 3 respectively, and let S 4 be the solution returned by Sieve-Streaming (with the standard threshold 1 2 OPT k ).

It is known that Sieve-Streaming is a 1 2 -approximation (Badanidiyuru et al., 2014):

Fact A.13. We always have f ( S 4 ) ≥ 1 2 OPT .

Lemma A.14. We have E [ max( f ( S 1 ) , f ( S 2 ) , f ( S 3 ) , f ( S 4 )) ] ≥ ( 1 2 +8 · 10 -14 ) OPT .

Proof. We will have three cases, depending on the (non-random) properties of the instance.

Case 1: there exists a dense subset. Then by Theorem A.2 we get that f ( S 1 ) ≥ 0 . 50025OPT with probability at least 0 . 01. On the other hand, Fact A.13 guarantees that f ( S 4 ) ≥ 1 2 OPT always holds. Thus

<!-- formula-not-decoded -->

Case 2: we have P [ f ( S L ) &gt; α OPT] &gt; 0 . 01 . By Claim 1 we have

<!-- formula-not-decoded -->

And whenever f ( S L ) &gt; α OPT and (5), Lemma A.5 yields that

<!-- formula-not-decoded -->

On the other hand, Fact A.13 guarantees that f ( S 4 ) ≥ 1 2 OPT always holds. Thus

<!-- formula-not-decoded -->

Case 3: there is no dense subset, and P [ f ( S L ) ≤ α OPT] ≥ 0 . 99 . Then Theorem A.6 yields that f ( S 2 ) ≥ 1+ ε 2 OPT with probability at least 0 . 49 η . On the other hand, Fact A.13 guarantees that f ( S 4 ) ≥ 1 2 OPT always holds. Thus

<!-- formula-not-decoded -->

Therefore, for any k , our algorithm outputs a solution of value at least ( 1 2 +8 · 10 -14 ) OPT in expectation.

## B. Impossibility Result for Adversarial-Order Streams

In this section we prove Theorem 1.1 - an unconditional lower bound on the memory usage of any single-pass streaming algorithm for submodular maximization that is allowed to query the value of the submodular function on feasible sets (ones of cardinality at most k ) and having approximation factor 1 / 2 + glyph[epsilon1] and some constant probability of success. Such bounds are usually proved via reductions from communication problems with certain communication complexity lower bounds. Here we reduce the INDEX problem to our problem. In what follows, we first define the INDEX problem and then we state a known communication complexity lower bound for this problem. We then present a reduction from INDEX to streaming submodular maximization.

INDEX problem: We consider a communication game consisting of channel coding, where

- Alice gets x ∈ { 0 , 1 } m for some integer m .
- Bob gets an integer i ∈ [ m ].
- The goal is to compute the function f ( x, i ) = x i .

We let R pub 2 / 3 (INDEX) denote the minimum number of bits required to be sent from Alice to Bob in order to solve INDEX problem with success probability at least 2 / 3. The assumption is that Alice and Bob both have access to public random bits. Notice that the communication in this setting, is only from Alice to Bob. We know that this has an Ω( m ) lower bound in the one-way communication model (e.g., see (Bar-Yossef et al., 2002), (Jayram et al., 2008))

Theorem B.1. (Indexing lower bound) For any integer m ,

<!-- formula-not-decoded -->

A more general result, involving the k-party generalized addressing function, appears in (Bar-Yossef et al., 2002). This theorem shows that in order to solve the Indexing problem with constant success probability, Ω( m ) bits of communication is required.

Reduction to submodular maximization: We present a reduction from the INDEX problem to streaming submodular maximization problem. In this reduction part of the stream is constructed based on the x vector that Alice has and part of the stream is constructed based on the index i that Bob holds. If there exists a streaming algorithm with small memory, Alice can first feed the algorithm with her part of the stream and then send to Bob the state of the memory. Then Bob can continue the algorithm with the memory state he received from Alice and feed the algorithm with his part of the stream and obtain the solution. Then based on the solution that the algorithm gives, Bob outputs f ( x, i ). So any lower bound that holds on the communication complexity of indexing problem should also hold on the memory usage of streaming submodular mazimization problem.

Formally, we prove the following theorem.

Theorem B.2. For any integer k &gt; 2 and any δ &gt; 0 , there exist a family of instances of submodular maximization problem such that any algorithm which is allowed to query the value of the submodular function on feasible sets, cardinality at most k , with approximation guarantee better than k/ (2 k -1) and success probability δ , needs at least Ω( δ n k ) bits of memory.

Proof. We show a reduction from any instance of the INDEX problem to an instance of streaming submodular maximization problem. Let U be a universe of size |U| = (2 k +1) m . We assign k elements to each of the x j 's that Alice has and one element to the index i that Bob has.

<!-- formula-not-decoded -->

Let us now construct the stream:

1. For every j ∈ [ m ] if x j = 1 then Alice inserts u l j for all l ∈ [ k ] into the stream. Otherwise, Alice inserts ¯ u l j into the stream.

2. Afterwards Bob adds w i to the stream. Recall that i is the index that is given to bob in the Indexing problem.

Therefore, the length of the stream is n = km +1 where km elements of it are at Alice's side and one element is at Bob's side. It remains to define the submodular function f . For simplicity, we define f only for the elements that are on the stream. First note that by design, only one of the w i 's can be present in the stream which is the one that correspond to the index i that Bob holds. Now, let V i = { u l j , ¯ u l j ; j ∈ [ m ] \{ i } , l ∈ [ k ] } ∪ { ¯ u l i ; l ∈ [ k ] } and U i = { u l i ; l ∈ [ k ] } . For any S , f ( S ) is defined as follows:

<!-- formula-not-decoded -->

Observation B.3. The function f ( · ) as defined in (8) is monotone and submodular.

Now note that because in Alice's side of the stream w i is not present and by the assumption that the algorithm is only allowed to query the function value on feasible sets, from Alices point of view f collapses to the following function:

<!-- formula-not-decoded -->

for every set S a subset of the stream in Alice's side such that |S| ≤ k . Therefore, this function reveals no information about the index i to Alice. Let ANS be the solution that the algorithm returns. Then Bob outputs x i = 1 if ANS &gt; k and x i = 0 otherwise.

Let us now compute the value of the optimum solution to the submodular instance that we constructed (denoted by OPT), depending on the answer to the given instance of the INDEX problem.

- if x i = 0: Then for any subset S of the stream we have S ∩ U i = ∅ hence by definition f ( S ) ≤ k , and in fact OPT = k .
- if x i = 1: Then f ( S ) = 2 k -1, for S = { w i } ∪ { u l i | l ∈ [ k -1] } so OPT ≥ 2 k -1 and in fact, OPT = 2 k -1.

Therefore, any algorithm for submodular maximization problem that has an approximation guarantee better than k/ (2 k -1) and works with any constant probability δ &gt; 0, should also use memory at least δ 10 R pub 2 / 3 (INDEX). The reason is that we can run 10 δ instances of submodular maximization independently and then take the max at Bob's side. Because each of them has approximation guarantee k/ (2 k -1) with probability δ independently, their maximum will have approximation guarantee k/ (2 k -1) with probability at least 1 -(1 -δ ) 10 δ ≥ 1 -e -10 ≥ 2 / 3. Therefore the streaming submodular maximization has to use δ 10 R pub 2 / 3 (INDEX) = Ω( δm ) = Ω( δ n k ) space.

Our reduction shows that even estimating the value of OPT to within a factor better than k/ (2 k -1) with any constant probability requires memory Ω( n k ).

## C. Hard Example for Sieve-Streaming with Random Arrival Order

In this section we show that there exists a randomly ordered stream on which Sieve-Streaming outputs a set S of expected value at most (1 / 2 + o (1))OPT. We start by showing this claim for an algorithm A similar to Sieve-Streaming . Then, in Theorem C.1, we show that there is a collection of elements that, when presented as a randomly ordered stream, makes Sieve-Streaming and A behave identically with probability at least 1 -δ , for any fixed δ &gt; 0.

Let A be an algorithm for submodular maximization in the streaming setting that takes a set of thresholds T as an auxiliary parameter. The algorithm A instantiates the following greedy procedure:

- For each threshold τ ∈ T in parallel : Let S τ = ∅ . Then, while | S τ | &lt; k , do the following for each arriving element e :

- -If the arriving element e satisfies f ( e | S τ ) ≥ τ , add e to S τ .
- Output arg max S τ : τ ∈T f ( S τ ).

We will show that there exists an optimal solution O and a collection M of elements with the following property. If the elements of M are presented in a random order to A , then for every τ ∈ T we have that f ( S τ ) ≤ (1 / 2 + o (1))OPT with high probability. In the rest of the section we exhibit one such collection M .

Claim 3. Let O = { e 1 , . . . , e k } and f ( e i ) = OPT /k , for every i . Let T be the set of thresholds used by the algorithm A . Then, there exists a stream of length O ( k ( k 2 |T | /δ ) |T | ) on which, when the stream is presented in a random order, the algorithm A outputs set S := arg max S τ : τ ∈T f ( S τ ) such that f ( S ) ≤ (1 / 2 + o (1))OPT with probability at least 1 -δ , for any fixed δ &gt; 0 . Furthermore, for every x ∈ S τ and every Y ⊆ S τ \ { x } it holds that f ( x | Y ) = τ .

Proof. We split the proof into two parts. First, for every τ , we exhibit set X τ that, as we will see later, constitutes set S τ . In the second part we compose sets X τ to obtain a random stream having the desired properties.

First part: exhibiting X τ . We consider three cases with respect to the value τ , and for each of them give a construction of set X τ . For this part of the proof, we assume that the stream consists only of set X τ and O presented in that order.

- Case τ ≤ OPT / (2 k ). Let X τ = { x 1 , . . . , x k } such that f ( x i ) = τ , for every i , and f ( X τ ) = kτ . Clearly, A will collect all the k elements of X τ , and hence will not select any element of O , i.e. S τ = X τ . Note that f ( X τ ) = kτ ≤ OPT / 2.
- Case OPT / (2 k ) &lt; τ ≤ OPT /k . Let X τ = { x 1 , . . . , x t } , where t = OPT / (2 τ ) (for the sake of clarity, we assume that t is an integer and remove this assumption at the end of the proof). In addition, we define X τ so that: f ( X τ ) = tτ = OPT / 2; f ( x i ) = f ( X τ ) /t = τ ; and, f ( e i | X τ ) = (OPT -f ( X τ )) /k &lt; τ . It is easy to see that such set X τ exists. Hence, in this case, we have S τ = X τ and therefore f ( S τ ) = OPT / 2.
- Case OPT /k &lt; τ . In this case, we simply set X τ = ∅ . Then, as f ( e i ) &lt; τ for every i ∈ O , we have S τ = S τ ∩ O = ∅ .

Additional property of X τ sets. Let OPT / (2 k ) &lt; τ ≤ OPT /k . From the definition we have f ( X τ ) = OPT / 2. For each such τ design X τ so that it 'covers' the same half of O . For each τ ′ ≤ OPT / (2 k ) design X τ ′ so that it covers a subarea of X τ . Then, for τ 1 &lt; τ 2 and OPT / (2 k ) &lt; τ 2 ≤ OPT /k we have

<!-- formula-not-decoded -->

Second part: composing a random stream. First, observe that for τ &gt; OPT /k the algorithm A will not collect any element from O and also X τ = ∅ . Therefore, such threshold τ does not affect the outcome of A , and for the rest of the proof and w.l.o.g. we assume that for every τ ∈ T it holds τ ≤ OPT /k . Then, from our construction, we have k/ 2 ≤ | X τ | ≤ k .

Let OPT /k ≥ τ 1 &gt; τ 2 &gt; . . . &gt; τ |T | be the thresholds of T . Let M be a multiset of elements that consists of the following:

- The multiset M contains O .
- For each i , M contains ( k 2 |T | /δ ) i copies of X τ i .

Let M R be a random stream consisting of the elements of M . For the sake of brevity, define X τ 0 := O . Then, we have the following. For every i ≥ 1 and any element x ∈ X τ i it holds

P [ any element of the copies of X τ i -1 appears before all the copies of x in M R ]

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Therefore, by union-bound, one copy of X τ i appears before any of the elements of X τ i -1 (taking into account of its copies) in M R with probability at least 1 -δ/ |T | . Furthermore, this claim holds for all the i simultaneously with probability at least 1 -δ . Let E be event that one copy of X τ i appears before any of the elements of X τ j , for all j &lt; i . Our discussion implies P [ E ] ≥ 1 -δ . In the rest of the proof, assume that E was realized.

Now, consider a threshold τ i ∈ T . First, no element from X τ j , for j &gt; i , will be chosen to S τ j by A as τ j &gt; τ i . Next, we distinguish two cases.

- Case τ i ≤ OPT / (2 k ). In this case, we have | X τ i | = k by the design of X τ i . Hence, assuming E , we have that S τ i will contain k elements before any element of X τ j , for any j &lt; i , is seen in the stream. Therefore, S τ i = X τ i , and f ( S τ i ) ≤ OPT / 2.
- Case OPT / (2 k ) &lt; τ i ≤ OPT /k . In this case and assuming E , A will select X τ i to S τ i before any element of X τ j , for any j &lt; i , is seen in the stream. Now, by the properties of sets X τ , including property (9), for any x ∈ X τ j and any j &lt; i , we have

<!-- formula-not-decoded -->

and hence no such x will be added to S τ i . Furthermore, the algorithm A executed on X τ i and O (in that order) outputs set S τ i such that f ( S τ i ) ≤ OPT / 2, as desired.

Removing the assumption that OPT / (2 τ ) is integral. Recall that this assumption was made in the case OPT / (2 k ) &lt; τ ≤ OPT /k . We start by redefining t as t = glyph[ceilingleft] OPT / (2 τ ) glyph[ceilingright] . First, notice that in this case f ( X τ ) ≤ (1 / 2 + o (1))OPT and, furthermore, all the other aforementioned properties hold except property (9). The only place where we need property (9) is in our analysis of the second part to derive equation (10), i.e. to show that once X τ i is collected then no element from X τ j , for any j &lt; i , will be added to the set S τ i . But, to achieve that, instead of equation 10 the following weaker property suffices

<!-- formula-not-decoded -->

Now we exhibit a collection of sets X τ such that, even in the case OPT / (2 τ ) is not integral, the collection have all the desired properties (with property (11) replacing property (10).

Let f be a cover function in the 2-dimensional space. Assume that O is a rectangle. Divide that rectangle into 2 · k ! · |T | small rectangles all of the same area. Let A be the set of a half of those rectangles. Observe that f ( A ) = OPT / 2. Next, we define every X τ so that f ( A | X τ ) = 0 as follows. All the rectangles of A are (arbitrarily) covered by the elements of X τ so that every element of X τ covers | A | / | X τ | rectangles of A , and no two elements of X τ overlap. Observe that | A | is divisible by any positive integer being at most k , and hence is divisible by | X τ | . The remaining value τ -OPT / (2 | X τ | ) of every element of X τ that is not contained within A is arbitrarily allocated in the part of O outside of A , under the condition that no two elements of X τ overlap. But now, for any X τ i and X τ j such that OPT / (2 k ) &lt; τ i , τ j ≤ OPT /k , and for any x ∈ X τ j we have

<!-- formula-not-decoded -->

as desired.

This concludes the proof.

Now we use Claim 3 to conclude that Sieve-Streaming outputs a set of value at most (1 / 2 + o (1))OPT.

Theorem C.1. There is a stream on which, even when the stream is presented in a random order, Sieve-Streaming outputs set S such that with probability at least 1 -δ it holds f ( S ) ≤ (1 / 2 + o (1))OPT , for any fixed δ &gt; 0 .

Proof. Algorithm Sieve-Streaming considers a list G of guesses of the value of an optimal solution. We point out that |G| does not depend on the length of the stream. For each of the guesses v ∈ G , the algorithm maintains set S v that adds an element e to the set if f ( e | S v ) ≥ ( v/ 2 -f ( S v )) / ( k -| S v | ). Let τ := v/ (2 k ). Then, as long as every added e has marginal gain exactly τ , we have

<!-- formula-not-decoded -->

In other words, if every element added to S v has marginal gain equal to τ , then the threshold Sieve-Streaming considers to add a new element to S v remains the same. But this is exactly how the algorithm A will behave, assuming that such stream is presented to Sieve-Streaming . By Claim 3, there exists a stream which when presented randomly has this desired property with probability at least 1 -δ . This now shows that there is a stream on which, when given randomly, the algorithms Sieve-Streaming and A behave exactly the same with probability at least 1 -δ , and hence Sieve-Streaming outputs set S such that f ( S ) ≤ (1 / 2 + o (1))OPT.

## D. P-Pass Algorithm

In this section, we present a multi-pass algorithm for the SubMax problem. We assume that the value OPT of the optimum solution O is known in advance. We remove this assumption in Appendix E. Our algorithm achieves 1 -1 /e -ε approximation for arbitrary ε using O ( k ) memory and O ( 1 glyph[epsilon1] ) passes over the data stream. In (McGregor &amp; Vu, 2016) the problem of maximum k -set coverage (a special case of SubMax ) was studied. They give a (1 -1 /e -ε )-approximation algorithm with O ( k glyph[epsilon1] 2 ) space and O ( 1 glyph[epsilon1] ) passes.

Our P-Pass algorithm (Algorithm 6) works as follows: We start with S = ∅ , we pick an element in the i -th pass over the stream if | S | &lt; k and f ( e | S ) is at least T i · OPT k , where T i = ( p p +1 ) i .

## Algorithm 6 P-Pass Algorithm

```
1: S := ∅ 2: for i = 1 to p do 3: for the j -th element e j on the stream do 4: if f ( e j | S ) ≥ ( p p +1 ) i · OPT k and | S | < k then 5: S := S ∪ { e j } return S
```

Let S i be the partial solution obtained after the i -th pass over the stream, for 1 ≤ i ≤ p . Let k i = | S i \ S i -1 | denote the number of elements picked by the algorithm in the i -th pass. Let us begin my showing some properties of the S i sets. We first show that if for some 1 ≤ i ≤ n , S i is not full, then f ( S i ) is quite big. Formally:

<!-- formula-not-decoded -->

Proof. Since | S i | &lt; k , thus for any element o ∈ O\ S i , we have f ( o | S i ) ≤ T i · OPT k , and for any element o ∈ O∩ S i , f ( o | S i ) = 0. Therefore,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This lemma shows that if | S | ≤ k at the end of our algorithm, then we get the desired approximation guarantee. Another important ingredient that we need in order to analyze our algorithm is understanding the case that we pick the expected (or slightly more) number of elements in each round of our algorithm. More precisely, we show that:

<!-- formula-not-decoded -->

Proof. We prove the lemma by induction.

Base case : i = 1. If k 1 ≥ k · ( 1 p + α ) , then we have

<!-- formula-not-decoded -->

Thus, Induction hypothesis : If ∑ i j =1 k j ≥ k · ( i p + α ) , then

<!-- formula-not-decoded -->

Induction step : If ∑ i +1 j =1 k j ≥ k ( i +1 p + α ), then we have

<!-- formula-not-decoded -->

Let us consider three following cases,

1. If k i +1 = k · ( 1 p + β ) for β ≥ α .

Since | S i | &lt; k , by Lemma D.1, we have f ( S i ) ≥ OPT · (1 -( p p +1 ) i ). Thus we have

<!-- formula-not-decoded -->

2. If k i +1 = k ( 1 p + β ) for 0 ≤ β ≤ α .

Thus, we have ∑ i j =1 k j ≥ k · ( i +1 p + α ) -k · ( 1 p + β ) ≥ k · ( i p +( α -β ) ) . Therefore, by induction hypothesis, we have

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

3. If k i +1 = k · ( 1 p -β ) for 0 ≤ β ≤ 1 p .

Thus, we have ∑ i j =1 k j ≥ k · ( i +1 p + α ) -k · ( 1 p -β ) ≥ k · i p +( α + β ). Therefore, by induction hypothesis, we have

<!-- formula-not-decoded -->

Thus,

<!-- formula-not-decoded -->

Therefore in all cases we proved

<!-- formula-not-decoded -->

Now we are ready to prove the main result of this section:

<!-- formula-not-decoded -->

Proof. Let us now consider the following cases:

1. If | S | = | S p | &lt; k , then by Lemma D.1, we have,

<!-- formula-not-decoded -->

2. If | S | = k and k p ≥ k p .

Then | S p -1 | &lt; k , thus, using Lemma D.1, we get,

<!-- formula-not-decoded -->

3. If | S | = k and k p &lt; k
2. p .

Let k p = k · ( 1 p -α ), for 0 ≤ α &lt; 1 p . Thus, we have

<!-- formula-not-decoded -->

Therefore using Lemma D.2, we get,

<!-- formula-not-decoded -->

Therefore in all cases we proved,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## E. Removing Assumption that OPT is Known

Any algorithm we introduced so far works under the assumption of knowing the value of the optimum solution in advance. Let A be the representative of one of our algorithms, and v be the estimation of the optimum solution. Observe that all of our algorithms work as follows: It first starts with an empty set S = ∅ and adds element e to S if | S | &lt; k and f ( e | S ) ≥ T v k where T is some fixed or adaptive constant depending on the algorithm. Denote by A ( v ) the output of the algorithm given value v as an estimation of the optimum solution. We proved that for any of our algorithms there is a constant c such that f ( A ( v )) ≥ c · OPT if v = OPT. It is also easy to see that, if α · OPT ≤ v ≤ OPT, for any 0 ≤ α ≤ 1, then f ( A ( v )) ≥ c · v ≥ c · α · OPT.

To that end, we use the same approach as explained in (Badanidiyuru et al., 2014). Let O = { (1 + glyph[epsilon1] ) j | j ∈ Z } thus, there exists a value v ∈ O such that OPT 1+ glyph[epsilon1] ≤ v ≤ OPT. Let S v = A ( v ) and S = argmax v ∈ O f ( S v ). Therefore, f ( S ) ≥ f ( S v ) ≥ c 1+ glyph[epsilon1] · OPT ≥ c · (1 -glyph[epsilon1] ) · OPT. We wish to run a copy of the algorithm A for any v ∈ O in parallel, and output the best solution, however | O | = ∞ .

To deal with this, we keep track of the maximum value element of the stream at any time. Let m i = max 1 ≤ j ≤ i f ( { e j } ) denote the maximum value element of the stream after observing e 1 , e 2 , . . . , e i . Clearly m i ≤ OPT ≤ k · m i . Also notice that the algorithm A given value v as an estimation for OPT, picks an element e from the stream only if f ( e | S ) ≥ T · v k .

Therefore it suffices to keep the estimations in O i within the range [ m i , k · m i T ]. Hence, we define O i = { (1 + glyph[epsilon1] ) j | j ∈ Z , m i ≤ (1 + glyph[epsilon1] ) j ≤ k · m i T } . Thus for all v ∈ O i , we know that any element with the marginal value at least T · v k appears only after updating O i . Hence for any v ∈ O i \ O i -1 , we can start with the empty set S v = ∅ . Any time m i gets updated, we delete all S v 's which v / ∈ O i . We run | O i | copies of the algorithm A in parallel for any v ∈ O i .

| Algorithm 7 Guessing OPT           | Algorithm 7 Guessing OPT                                                            |
|------------------------------------|-------------------------------------------------------------------------------------|
| 1: m = 0                           | 1: m = 0                                                                            |
| 2: for i                           | = 1 to n do                                                                         |
| 3:                                 | m = max( m,f ( { e i } )                                                            |
| 4:                                 | O i = { (1+ glyph[epsilon1] ) j &#124; j ∈ Z ,m ≤ (1+ glyph[epsilon1] ) j ≤ k · m T |
| 5:                                 | Delete all S v 's such that v / ∈ O i                                               |
| 6:                                 | For each v ∈ O i \ O i - 1 set S v = 0                                              |
| 7:                                 | for v ∈ O i do                                                                      |
| 8:                                 | S v = A ( v )                                                                       |
| 9: return argmax v ∈ O n f ( S v ) | 9: return argmax v ∈ O n f ( S v )                                                  |

Therefore memory, and the update time of the new algorithm increases by the factor | O i | = log 1+ glyph[epsilon1] k T = O ( log k T glyph[epsilon1] ) , and it outputs c · (1 -glyph[epsilon1] )-approximate solution.