## Producer-Side Experiments Based on Counterfactual Interleaving Designs for Online Recommender Systems

Yan Wang and Shan Ba LinkedIn Corporation

November 8, 2023

## Abstract

Recommender systems have become an integral part of online platforms, providing personalized recommendations for purchases, content consumption, and interpersonal connections. These systems consist of two sides: the producer side comprises product sellers, content creators, or service providers, etc., and the consumer side includes buyers, viewers, or customers, etc. To optimize online recommender systems, A/B tests serve as the golden standard for comparing different ranking models and evaluating their impact on both the consumers and producers. While consumer-side experiments is relatively straightforward to design and commonly employed to assess the impact of ranking changes on the behavior of consumers (buyers, viewers, etc.), designing producer-side experiments for an online recommender/ranking system is notably more intricate because producer items in the treatment and control groups need to be ranked by different models and then merged into a unified ranking to be presented to each consumer. Current design solutions in the literature are ad hoc and lacking rigorous guiding principles. In this paper, we examine limitations of these existing methods and propose the principle of consistency and principle of monotonicity for designing producer-side experiments of online recommender systems. Building upon these principles, we also present a systematic solution based on counterfactual interleaving designs to accurately measure the impacts of ranking changes on the producers (sellers, creators, etc.).

Keywords : Two-sided marketplace, Creator-side experiment, Supply-side experiment, SUTVA violation, Ranking optimization, Attention function, A/B test

## 1 Introduction

Recommender systems are ubiquitous in online platforms such as Amazon, Facebook, LinkedIn and Airbnb for suggesting items to buy, contents to view, people to connect, rooms to book, etc. A recommender system has two sides: a producer side (e.g., sellers in the marketplace, content creators in feeds, service-providers such as hosts in Airbnb, etc.) and a consumer side (e.g., buyers, content viewers, customers, etc.). For each consumer, the recommender system uses machine learning models to rank a set of 'producer items' (e.g., products from sellers, contents from creators, rooms listed by hosts, etc.) and fills them into pre-designated spots in the user interface of an app or webpage (Figure 1). The aim of the recommender system is to predict the preference of each consumer and allocate more preferable producer items into the spots where the consumer would pay more attention to.

Figure 1: An illustration of the online recommender system.

<!-- image -->

To optimize an online recommender system, the A/B test (a.k.a. online controlled experiment) (Xu et al., 2015; Tang et al., 2010; Bakshy et al., 2014; Kohavi et al., 2009, 2013, 2020) is the golden standard for comparing different ranking models and measuring how ranking changes impact the behaviors of consumers/producers. In most online platforms, any new ranking model needs to be thoroughly evaluated in online experiments before it can get fully deployed. There are generally two different types of online experiments involved for a recommender system.

Consumer-side experiments measure how ranking changes in a recommender impact the behavior of consumers, which are relatively easy to design and widely used in practice. The standard approach is to randomly split all consumers into the control and treatment groups, where each group is associated with a different ranking model (ranker). For each consumer, the recommender ranks all available producer items using the variant of ranking model assigned to her group (Figure 2), and then consumer-oriented metrics of the two groups are compared to conclude which ranking model works better for consumers.

Figure 2: Consumer-side experiment of an online recommender system

<!-- image -->

In addition to the consumer side, it is important to also measure the effects of ranking changes on the producer side, because a recommender system needs to be optimized based on objectives derived from both sides. For example, purely optimizing the rankings toward buyer's satisfactions in an online marketplace may result in directing most of the traffic to a small portion of top sellers and causing the other sellers (e.g. new sellers) to churn. Similarly, when improving the ranking model for news feed, we not only need to gauge its impacts on the content viewers, but also need to consider how it would change the behaviors of content creators. Nevertheless, despite the importance of measuring producer-side impacts, it is challenging to design producer-side experiments for an online recommender system.

The producer-side experiment requires randomly splitting all producers into the control and treatment groups, where producer items from each group are ranked by a different model (Figure 3). Because a consumer can only see a single ranked list of producer items each time, designing the producer-side experiment is challenging in that it needs to blend rankings of producer items from the treatment and control groups together for each consumer. Several design approaches for producer-side experiments have been developed in the literature, but they are ad hoc and suffer from various issues that could lead to biased experiment readouts. This paper aims to address this challenge by proposing rigorous design principles that any producer-side experiment should follow in order to accurately measure the effects of ranking changes. The rest of this paper will be organized as follows. In Section 2, we review existing producer-side experiment design solutions and discuss their limitations and biases. Section 3 introduces some basic concepts and notations, and in Section 4, we propose two general principles for designing producer-side experiments where SUTVA is often violated. Building on these principles, Section 5 derives a rigorous solution based on counterfactual interleaving designs which can ensure an unbiased comparison between the treatment and control rankers. In Section 6, we provide examples to illustrate the proposed solution, and some final conclusion remarks are given in Section 7.

Figure 3: Producer-side experiment of an online recommender system

<!-- image -->

## 2 Existing Methods for Designing Producer-Side Experiments

In this section, we provide an overview of the existing methods for designing producer-side experiments and discuss their issues. To facilitate the discussion, we will use an illustration example where a total of eight producer items d 1 , . . . , d 8 are randomly split into the control group G 0 = { d 1 , d 3 , d 5 , d 7 } and the treatment group G 1 = { d 2 , d 4 , d 6 , d 8 } . The problem of designing producer-side experiments is how to blend the rankings of producer items based on the treatment and control rankers for each consumer as shown in Figure 3.

## 2.1 Double Randomization

A straightforward solution is to use double randomization and only show either the treatment or control group of producer items to each consumer. This design requires further splitting consumers into the control and treatment groups where the consumer-side randomization is independent from that at the producer side. If a consumer is in control, the recommender would only show her producer items from the control group G 0 = { d 1 , d 3 , d 5 , d 7 } that are ranked by the control model. Similarly, if a consumer is in treatment, the recommender would only show her producer items from the treatment group G 1 = { d 2 , d 4 , d 6 , d 8 } that are ranked by the treatment model. See Figure 4 for an illustration.

Drawbacks of this approach are obvious: the consumer cannot see any producer items from the other group, and each producer item can only be shown to a subset of consumers. These constraints not only lead to poor product experience during the experiment, but they also misrepresent the typical use cases on both the producer side and the consumer side. Consequently, valid conclusions cannot be drawn from such experiments.

It is important to note that this solution is different from the two-sided randomization design (Johari et al., 2022) and the multiple randomization design (Bajari et al., 2023) in the literature which assume that the intervention can be independently assigned for each consumer-producer pair and thus are not applicable for evaluating the ranking changes in an online recommender system.

Figure 4: Double randomization and only showing one group of producer items to each consumer. (Top: if the consumer is in control; Bottom: if the consumer is in treatment.)

<!-- image -->

## 2.2 Random Spot Labeling

Another common design solution is illustrated in Figure 5. For a consumer, the method first randomly labels each spot in her final ranking list to be either in treatment (T) or control (C). Then, producer items in control are ranked by the control model (e.g., [ d 3 , d 1 , d 5 , d 7 ]) and placed among the control spots, while producer items in treatment are ranked by the treatment model (e.g., [ d 4 , d 2 , d 8 , d 6 ]) and placed among the treatment spots.

This design approach is essentially based on a random merger of the treatment and control rankings. It is better than the previous approach in Section 2.1 as all the producer items can be shown to each consumer. However, the merged ranking is still not representative of the real product experience. To see this, consider an AA test scenario where the treatment and control models are the same. In this case, we would expect the final ranking to remain the same as using either treatment or control model to rank all producer items. However, the design method in Figure 5 would generate very different ranking results because the design imposes an extra constraint on the treatment or control label of each spot in the final ranking. For example, suppose in the AA test, both treatment and control models would rank the eight producer items as [ d 3 , d 1 , d 5 , d 7 , d 4 , d 2 , d 8 , d 6 ] for a consumer. Then, the correct final ranking for this consumer should just be [ d 3 , d 1 , d 5 , d 7 , d 4 , d 2 , d 8 , d 6 ] and the corresponding treatment/control label of each spot should be [ C, C, C, C, T, T, T, T ]. The random spot labeling constraint [ C, T, C, T, C, T, C, T ] in Figure 5, on the other hand, results in an inaccurate final ranking [ d 3 , d 4 , d 1 , d 2 , d 5 , d 8 , d 7 , d 6 ] for the consumer, which cannot reflect the real product experience for producers.

## 2.3 SUTVA and Counterfactual Rankings

The 'Stable Unit Treatment Value Assumption' (SUTVA) (Imbens and Rubin, 2015) is a standard assumption in designing A/B tests which requires that the potential outcome for one unit in the experiment depends only on its own treatment status and should not be affected by the treatment assignment to the other units. For producer-side experiments, SUTVA means that producers in each treatment group should not be affected by the existence of other treatment group; instead, their behavior should be the same as if the ranking model associated with their group is applied to all of the producers.

Figure 5: Random spot labeling. (For one consumer, producer items in control are ranked as [ d 3 , d 1 , d 5 , d 7 ] and placed among the control spots, while producer items in treatment are ranked as [ d 4 , d 2 , d 8 , d 6 ] and placed among the treatment spots)

<!-- image -->

Ha-Thuc et al. (2020) defines the control counterfactual ranking as the ranking of all producer items (from both the treatment and control groups) based on the control model. It represents the ranking result as if the control ranker is ramped to 100% of the producers. Similarly, the treatment counterfactual ranking is defined as using the treatment model to rank all producer items (not only the producer items in the treatment group), which represents the ranking result as if the treatment model is applied to 100% of the site traffic. Figure 6 and Figure 7 give two examples of the counterfactual rankings. When merging the rankings of producer items from the treatment and control groups together for each consumer in Figure 3, SUTVA requires that producer items from the control group should be placed in the same positions as if they were in the control counterfactual ranking while producer items from the treatment group should be placed in the same positions as if they were in the treatment counterfactual ranking. We call such a merged ranker in the producer-side experiment as the SUTVA ranker R ∗ . It has the desirable property that all producer items of each group are placed at the same positions as if the corresponding ranking model is ramped to 100% of the site traffic.

In practice, a valid SUTVA ranker may not exist. For designing producer-side experiments, we summarize the following two basic rules for when SUTVA can be met and must be followed. First is for the AA-Test scenario that we have described at the end of Section 2.2.

Rule 1 (AA-Test Scenario) . If the treatment and control rankers are identical, their merged ranker in the producer-side experiment should remain the same as the original ranker which is also the SUTVA ranker. (Figure 6)

This rule is important because in practice the difference between treatment and control rankers are often small (i.e. small treatment effect). All valid design methods for producer-side experiments need to satisfy Rule 1 and correctly yield the SUTVA ranker in the AA-test scenario.

Figure 6: AA-Test Scenario

<!-- image -->

Figure 7: Non-Conflict-Merging Scenario

Our next rule summarizes the non-conflict merging scenarios in which a valid SUTVA ranker R ∗ exists and should always be used.

Rule 2 (Non-Conflict-Merging Scenario) . When the treatment and control counterfactual rankings have no merging conflicts, the merged ranker in the producer-side experiment should be uniquely determined by SUTVA and place every producer item into the position as if the ranking model associated with its group is applied to all producers. (Figure 7)

When there are conflicts in merging the treatment and control counterfactual rankings (i.e., producer items from different groups both demand the same position in the merging process), SUTVA cannot be perfectly met and this is the challenging part in designing producer-side experiments.

To facilitate the discussions in the rest of this paper, we define a ranker R as a ranking model which provides a ranked list of producer items to a consumer in each session of the recommender system. Mathematically, R generates an one-to-one mapping function in each session:

<!-- formula-not-decoded -->

where G = { d 1 , d 2 , . . . } denotes the set of all producer items, L is a set of ranks each of which corresponds to a spot in the consumer's user interface. Here, a smaller rank value represents a better match between the producer item and the consumer, and without loss of generality, we assume that the consumer's user interface has |G| spots where spots with smaller indices tend to receive more attentions from the consumer (i.e., spots at the top of the page). The recommender system fills in these spots by matching the ranks of the producer items with the spot indices, i.e., the producer item d with R ( d ) = i will be put in spot i .

Let R 0 denote the control counterfactual ranker, R 1 denote the treatment counterfactual ranker, and then the SUTVA ranker R ∗ can be represented as

<!-- formula-not-decoded -->

for all producer items d ∈ G .

The SUTVA ranker R ∗ is the optimal solution for designing producer side experiment as long as it exists (no merging conflicts). However, if there exists d, d ′ ∈ G , d = d ′ such that R ∗ ( d ) = R ∗ ( d ′ ), the SUTVA ranker R ∗ is not a valid ranker as producer items d and d ′ are demanding the same position in the merged ranking. In the next two sections, we review existing solutions in the literature to handle such merging conflicts, discuss their shortcomings and also motivate our proposed principles.

## 2.4 Counterfactual Interleaving Design

In this paper, we will use the counterfactual interleaving design to refer to the design of producer-side experiments based on merging (or interleaving) different counterfactual rankings. It is important to distinguish it from the traditional interleaving designs (Radlinski and Craswell, 2013; Parks et al., 2017; Zhang et al., 2022) for consumer-side experiments which are not based on the counterfactual rankings.

Ha-Thuc et al. (2020) from the Facebook Marketplace proposed counterfactual interleaving designs which only randomly label a small percent (e.g., 1%) of producers as the control and treatment groups to minimize the chances of having merging conflicts and avoid the challenges in resolving merging conflicts in R ∗ . The rest of the producers would still be shown in the recommender, but their metrics would not be included in the experiment analysis. This approach can be summarized as follows:

Step 1: Generate counterfactual rankings R 0 and R 1 as if the control or treatment model is ramped to 100% of producers.

Step 2: Merge R 0 and R 1 into the SUTVA ranker R ∗ :

- For producer items in the control group, get their positions from the control counterfactual ranking R 0 .
- For producer items in the treatment group, get their positions from the treatment counterfactual ranking R 1 .

̸

- For the rest of producer items that are neither in the treatment nor control groups, get their positions from either R 0 , R 1 or another ranker, but these producer items would not be included in the treatment v.s. control comparison of the experiment.

̸

Step 3: In case that a pair of producer items demand the same position in R ∗ (i.e., there exists d, d ′ ∈ G , d = d ′ such that R ∗ ( d ) = R ∗ ( d ′ )), simply decide their order randomly.

Figure 8 provides a simple illustration of this approach. Because both treatment and control groups only contain a small fraction (e.g., 1%) of producer items, Ha-Thuc et al. (2020) shows that the probability of having merging conflicts in Step 2 is very low. In case that a pair of producer items happen to demand the same position in R ∗ , the method can randomly decide their order with equal probabilities. Because such merging conflicts are rare, the final ranker has the advantage that it is approximately a SUTVA ranker where the ranking position of each producer item does not depend on what ranker is applied to the other producer items.

Obviously, downside of this counterfactual interleaving design is that its lower ramp % of producer items vastly limits the experiment power. Although power may not be a concern for Facebook which has enormous amounts of online traffic, the method is not applicable to many other online recommender systems due to lack of power.

Figure 8: Counterfactual interleaving design from HaThuc et al. (2020)

<!-- image -->

Figure 9: Counterfactual interleaving design with merging conflicts

̸

Nandy et al. (2021) proposed the Unifying Counterfactual Rankings (UniCoRn) approach, which is the same as Ha-Thuc et al. (2020)'s counterfactual interleaving design except for allowing larger % of producers to be included in the treatment and control groups to increase the power of the experiment. When the treatment and control counterfactual rankings have merging conflicts (i.e., there exists d, d ′ ∈ G , d = d ′ such that R ∗ ( d ) = R ∗ ( d ′ )) and the SUTVA ranker R ∗ is not a valid ranker, the UniCoRn ranker also chooses to resolve any merging conflicts randomly to ensure that the final ranker is a valid ranker. Nandy et al. (2021) showed that the UniCoRn ranker is the ranker that gets closest to R ∗ in terms of the sum of squared error distance.

Nevertheless, both Ha-Thuc et al. (2020) and Nandy et al. (2021)'s solutions are ad hoc and have not addressed the critical question of how to correctly resolve merging conflicts in the counterfactual interleaving designs. For example, when two producer items d and d ′ have merging conflict, how to determine the probability that d should be placed ahead of d ′ ? Only minimizing the sum of squared error distance dist( R , R ∗ ) ∆ = ∑ d ∈G ( R ( d ) -R ∗ ( d )) 2 cannot determine the probabilities because any tie-breaking probabilities would lead to the same dist( R , R ∗ ). In fact, choosing different tie-breaking probabilities could generate a large number of possible UniCoRn rankers with different distributions of the producer items, but, unfortunately, the majority of them would lead to biased comparisons between the treatment v.s. control groups.

Consider the example of simple random tie-breaking (with equal probabilities) from Ha-Thuc et al. (2020) and Nandy et al. (2021). If the treatment group is ramped at a small percent of traffic (e.g. 5% of producers) while the control group is ramped at a large percent of traffic (e.g. 95% of producers), the beginning part of the treatment counterfactual ranking R 1 would contain very few treatment producer items. When equal probabilities are used to randomly break the ties in merging R 0 and R 1 , it would be hard to have any treatment producer items to appear in the right positions (as determined by R 1 ) in the beginning of the merged ranking where viewers mainly pay attention to. Consequently, the treatment counterfactual ranking R 1 cannot be properly represented in the merged ranker and the experiment readouts would be biased against the treatment group. In order to accurately measure the effects of ranking changes on the producer side, we will develop two general principles for designing producer-side experiments in the following sections. Based on the proposed principles, we will show that when resolving the merging conflicts between R 0 and R 1 , an unbiased counterfactual interleaving design need to follow a rigorous procedure to preserve the relative order of the SUTVA ranker and assign items from the smaller treatment group with a higher probability to be placed in the correct position (determined by its own counterfactual ranking). We will also show that the optimal order of conflicting producer items in some cases should be determined deterministically instead of randomly.

## 3 Notations and Concepts

In this section, we define some key concepts and mathematical notations in producer-side experiments, which lays the groundwork for subsequent discussions on the design principles. Let k ∈ K represent different treatment variants in the producer-side experiments. For example, K = { 0 , 1 } where k = 0 represents the control variant and k = 1 represents the treatment variant. Suppose each producer item in G is randomly assigned into one of the treatment groups G k with probability p k .

## 3.1 Experiment Readouts and Counterfactual Readouts

We first define the outcomes of producer-side experiments based on metrics aggregated at the producer (item) level. Let F represent the set of information pertaining to the consumer and producer items' features, and let U ( d ) denote the outcome of a metric associated with producer item d . Typically, U ( d ) depends on F as well as the ranking position of the producer item, i.e., U ( d ) = U ( d ; R ( d ) , F ). Without loss of generality, we assume the metric U is larger the better.

Let agg ∗ k ( U ) represent the counterfactual readout which is an aggregation of metrics of all producer items under the assumption that ranker R k is applied to 100% of producers. This can be formally expressed as:

<!-- formula-not-decoded -->

Ideally, we want to compare the counterfactual readouts { agg ∗ k ( U ) } for various k to determine the optimal ranker, but { agg ∗ k ( U ) } are not observable from the experiment. Instead, we can only observe the experiment readout which is an aggregation of U ( d ) for a treatment group k in the experiment:

<!-- formula-not-decoded -->

Here R is the final merged ranker in the experiment and the coefficient 1 /p k accounts for the fact that each producer item has a probability p k of being included in G k .

Both agg k and agg ∗ k are random variables and we are particularly interested in their expected values. It is important to acknowledge that their randomness originates from multiple sources. One source is the feature set F , and another source relates to the experimental design, i.e., how items are randomly allocated to treatments and how the merged ranker resolves merging conflicts. We will denote this experimental information by E and postulate the following assumption:

Assumption 3.1. The distributions of F and E are independent.

We generally lack knowledge about the specifics of the distribution of F , which is influenced by the complex interactions between consumers and producer items. However, we have complete knowledge regarding the distribution of E , as it is determined by the experimental design. Consequently, expectations will always be taken with respect to E and conditioned on F . This is denoted by the operator E E [ · ] or E [ ·|F ].

Moreover, it is crucial to note that the experimental-related randomness from E is only present in agg k ( U ), while agg ∗ k ( U ) is measurable with respect to F , i.e.,

<!-- formula-not-decoded -->

and the expectation with respect to E only needs to be considered for agg k ( U ).

Let us define

<!-- formula-not-decoded -->

as the expected value of the experiment readout, conditioned on F . Ideally, we want agg k ( U ) to be an unbiased estimator of the counterfactual readout agg ∗ k ( U ), which requires SUTVA to be met. Unfortunately, as explained in the previous section, SUTVA is often violated in the producer-side experiments due to merging conflicts and this is why we need to develop new design principles to ensure that valid conclusions can be drawn from producer-side experiment readouts.

## 3.2 Attention Functions and Convoluted Attention Functions

In the user interface of a recommender system, different positions or spots receive varying degrees of visibility or attention from the consumers. For any position j , let h ( j ) ≥ 0 represent the amount of attention garnered by a producer item at that position, where the attention function h ( j ) is monotonic decreasing as positions in the recommender system are indexed in such a way that smaller indices receive more attention from the consumer. For example, top ranking spot ( j = 1) in the recommender receives the highest attention from the consumer and the spots at the end receive little attention as few consumers would scroll far down the page.

Based on the attention function, we introduce an assumption regarding the structure of the observed metric outcome U associated with each producer item.

Assumption 3.2. For producer item d , its metric outcome U ( d ) can be decomposed into a product of a pure metric u ( d ) representing the inherent utility of d , which is independent of the ranking of the producer items, and an attention function h that solely depends on the rank or position R ( d ) of the producer item. Formally, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In the equation above, we use f ◦ g to denote the composition of functions f and g , such that f ◦ g ( d ) = f [ g ( d )]. This decomposition reflects the fact that a producer item's metric can be viewed as the product of its inherent quality (captured by u ) and the attention it receives based on its position (captured by h ).

Let us consider for a given position j and item x such that R 0 ( x ) = j . When x is in the control group G 0 , the merged ranker R may not always place x at position j due to merging conflicts. Instead, the final position of x under R can vary, and follows a certain distribution. Let us denote this distribution by π 0 j , such that for any position j ′ ,

<!-- formula-not-decoded -->

Due to such randomness, the average level of attention that the producer item x receives when x ∈ G 0 , is not strictly h ( j ), but rather a weighted average of h ( j ′ ), taking into account the probabilities { π 0 j ( j ′ ) } . Let us define this 'average attention' as h 0 ( j ):

<!-- formula-not-decoded -->

Since j can be any position, equation (3.5) essentially gives rise to a new 'attention function' h 0 , which we will refer to as the convoluted attention function . The effect of resolving merging conflicts in R can be conceptualized as a transformation of the underlying attention function from h to h 0 for the control group and its ranker R 0 . Similarly, this can also be applied to the treatment group and its ranker R 1 : For any position j , define

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The resolution of merging conflicts by R essentially imposes a transformation of the attention function from h to h 1 for the ranker R 1 .

In general, consider a treatment k ∈ K , a spot j ∈ L , and the merged ranker R that is utilized as the final ranker in the producer-side experiment as described in equation (3.2). Let J be the inverse of ranker function R :

<!-- formula-not-decoded -->

which is a spot-filling function that maps a spot index j back to the producer item that occupies it. Suppose the producer item d would occupy spot j according to ranker R k : R k ( d ) = j or d = J k ( j ). Due to merging conflicts as discussed in Section 2.3, the final merged ranker cannot guarantee R ( d ) = j . Instead, R ( d ) | d ∈G k is random and let π k j represent the distribution of this random variable:

<!-- formula-not-decoded -->

Obviously, the forms of convolution kernels { π k j } are determined by the rankers { R k } and way they are merged into R . They do not depend on the specific form of attention functions.

and

and Define h k as the convolution of the probability family { π k j } with the attention function h and call { π k j } the set of convolution kernels. For any spot j ∈ L , we can formally express the convoluted attention function as:

<!-- formula-not-decoded -->

The convolution kernels { π k j } quantifies the deviation of the merged ranker R from the SUTVA ranker R ∗ . In the specific case where R = R ∗ (e.g., an A/A test scenario), π k j becomes δ { j } (i.e., the distribution concentrated on the single spot j ), and consequently, h k simplifies to h .

## 4 New Design Principles

As we have discussed in Section 2, currently there are no rigorous guiding principles available for designing producer-side experiments when SUTVA is violated, and the existing solutions in the literature are ad hoc which would lead to biased designs. In this section, we will develop general principles for designing producerside experiments that are essential for any design solutions to follow.

## 4.1 Principle of Consistency

For a randomized experiment, a fundamental requirement is that the treatment and control groups need to be comparable and the only expected difference between them is caused by the intervention (i.e., different rankers) being studied. In producer-side experiments, however, the existing ad hoc solutions often introduce other confounding factors between the treatment and control groups (i.e., receiving different amounts of attentions from consumers) which would bias the experiment results. For example, consider the top-ranking position (i.e., spot 1) in the recommender which receives the highest attention from consumers. Assume R 0 ranks item x as the top candidate while R 1 ranks item y as the top candidate: R 0 ( x ) = R 1 ( y ) = 1. Due to the merging conflict, when x ∈ G 0 , the spot for x in the final merged ranker R ( x ) | x ∈G 0 will be random, where the randomness is determined by the specific method with which R merges R 0 and R 1 . Suppose for x ∈ G 0 , it has 80% chance of being placed at the best spot (spot 1 in R ) and for y ∈ G 1 , it has a 30% chance of being ranked first in R . Clearly, this design is biased in favor of R 0 : the best candidate x according to R 0 has a higher chance of getting the top spot compared to the best candidate y according to R 1 . Such bias will be reflected in the final experiment outcome and be confounded with the treatment effect under study.

To ensure an apple-to-apple comparison in the producer-side experiments, a fair design should require that R ( x ) | x ∈G 0 and R ( y ) | y ∈G 1 have the same distribution. This generalizes beyond just the top spot and applies for any integer j &gt; 0: when x ∈ G 0 , it should have the same chances to be placed at spot j ( j = 1 , 2 , . . . ) as y does under y ∈ G 1 . In other words, in order for the treatment and control groups to be comparable under the merged ranker R , we need to require that, for any ranking spot j , and for any producer items x and y such that R 0 ( x ) = R 1 ( y ) = j , the distributions of R ( x ) | x ∈G 0 and R ( y ) | y ∈G 1 should be identical. This requirement will be referred to as the Principle of Consistency , which is formally defined below.

Principle 1 (Principle of Consistency) . In designing producer-side experiments, the convolution kernels { π k j } defined in equation (3.6) , which represent the distributions of R ( d ) | d ∈G k for any given spot j , should be invariant with respect to k .

This principle can be formally justified based on the mathematical framework defined in Section 3. Based on equations (3.1) and (3.4), the counterfactual readout agg ∗ k ( U ) can be expressed as:

<!-- formula-not-decoded -->

where J k ( j ) = d if and only if R k ( d ) = j . It can be seen that agg ∗ k ( U ) depends on k only through the different rankers R k (or J k ) while the attention function h is the same for different k . Our next theorem below shows that this ideal property cannot be guaranteed in the observed experiment readouts { agg k ( U ) } or their expected values { agg k ( U ) } .

Theorem 4.1. Under Assumptions 3.1 and 3.2, for any treatment k ∈ K , agg k ( U ) can be computed as follows:

<!-- formula-not-decoded -->

Proof of this theorem is given in Appendix A. By comparing Equations (4.1) and (4.2), we can see that the difference between agg ∗ k ( U ) and agg k ( U ) is effectively a modification of the attention function through convolution, denoted as h → h k . Furthermore, (4.2) shows that for various k , differences in agg k ( U ) not only are due to the differences in rankers R k (or J k ) but they can also be caused by different convoluted attention functions h k . This makes it indiscernible whether the disparities in the expected experiment readouts { agg k ( U ) } between the treatment and control groups stem from the rankers or their attention functions. To mitigate this confounding ambiguity, a correctly designed producer-side experiment must ensure that the convoluted attention functions h k remain independent of k (i.e., while treatment and control groups correspond to different rankers, they must share the same convoluted attention function to be comparable). Given that the specific form of the attention function h is unknown, the only way to assure independence of h k on k is by requiring that the convolution kernel π k j does not rely on k in equation (3.7), which formalizes the principle of consistency above.

## 4.2 Principle of Monotonicity

In addition to the consistency principle, in this section we introduce another important principle for designing producer-side experiments. Because the spots or positions in a recommender system are indexed according to the level of attention they receive (i.e., position 1 receives the highest attention, position 2 the second highest, etc.), the attention function h is inherently defined to be monotonically non-increasing: h (1) ≥ h (2) ≥ . . . and the recommender is designed to place the most suitable (highest ranked) producer item in position 1, the second best in position 2, and so on. However, due to the randomness from resolving merging conflicts in the producer-side experiments, the average level of attention that a producer item receives is not strictly h but a weighted average of h , which is defined as the convoluted attention h k in Section 3.2. As a result, for producer-side experiments to be valid, we not only need to have monotonically non-increasing attention function h , but also need to require the convoluted attention function h k to retain this monotonic characteristic. We will refer to this requirement as the Principle of Monotonicity.

Since the exact form of the attention function h is unknown, the design of producer-side experiments can leverage the convolution kernels π k j in equation (3.7) to ensure h k is decreasing. Let F π ( x ) denote the cumulative distribution function (CDF) of a distribution π on real numbers, i.e.,

<!-- formula-not-decoded -->

where X follows the distribution π . We can then define the partial order relation as

<!-- formula-not-decoded -->

It is a well-established fact that for any monotonically decreasing function h on the real line, if π 1 ≺ π 2 , then E π 1 [ h ] ≥ E π 2 [ h ]. We can now formally introduce the Principle of Monotonicity as follows:

Principle 2 (Principle of Monotonicity) . In designing producer-side experiments, the convoluted attention functions h k must be monotonically non-increasing for any k and for any non-increasing attention function h . Equivalently, the convolution kernels { π k j } defined in equation (3.6) needs to be non-decreasing with respect to j , for any k .

Using the mathematical framework defined in Section 3, we can provide some further justifications of this monotonicity principle.

Lemma 4.2. Define R u as a ranker that ranks producer items d ∈ G according to their pure metric u ( d ) (in descending order). Then, ranker R u satisfies:

<!-- formula-not-decoded -->

Proof of this Lemma is given in Appendix B. Lemma 4.2 demonstrates that the ranker which maximizes the counterfactual readout agg ∗ ( U ) = ∑ d ∈G U ( d ) = ∑ d ∈G u ( d ) × h ◦ R ( d ) aligns precisely with the ranking based on producer item's pure metric u ( d ). This provides a fundamental justification for why recommendation systems aim to model producer item's intrinsic utility u ( d ) and use the scores obtained from these models to establish ranking. However, it is crucial to recognize that this alignment hinges on the attention function being monotonically non-increasing. Building upon Lemma 4.2, we can have the following corollary.

Corollary 4.3. Assuming there are K rankers R 0 , ..., R K -1 being compared in a producer-side experiment, and one of them, say R k , is equivalent to R u as defined in Lemma 4.2. If the attention function h is monotonically non-increasing, then the counterfactual readout agg ∗ k ( U ) for group k is superior to those of other groups, meaning

̸

<!-- formula-not-decoded -->

Corollary 4.3 indicates that the counterfactual readout agg ∗ ( U ) from producer-side experiments can be used to correctly identify the best ranker if the attention function is monotonically non-increasing. Nevertheless, in practice we cannot directly observe the counterfactual readout agg ∗ ( U ). The following key corollary, which is derived based on the observed experiment readouts { agg k ( U ) } or their expected values { agg k ( U ) } , highlights the importance of having both the consistency and monotonicity principles in designing producer-side experiments:

Corollary 4.4. Assume K rankers R 0 , ..., R K -1 are compared in a producer-side experiment where one of them satisfies R k = R u as defined in Lemma 4.2. If the merged ranker R from the design adheres to both consistency and monotonicity principles, then

̸

<!-- formula-not-decoded -->

which implies that the best ranker R k can be correctly identified based on the expected values of the observed experiment readouts.

## 5 Solution for the Counterfactual Interleaving Design

In Sections 2.4, we have discussed how the existing counterfactual interleaving designs lack a systematic strategy to resolve the merging conflicts when trying to create a valid merged ranker R based on the SUTVA ranker R ∗ . Based on the proposed design principles from Section 4, we are now able to develop a rigorous solution of counterfactual interleaving designs for producer-side experiments.

Consider producer-side experiments comparing two ranking models, where all the producer items are split into two groups: the control group G 0 and the treatment group G 1 . We propose to create the counterfactual interleaving design for any possible ramping percentages of producers (Figure 9) through the following steps:

Step 1: Generate counterfactual rankings R 0 and R 1 as if the control or treatment model is ramped to 100% of the producer items.

Step 2: Merge R 0 and R 1 to get the SUTVA ranker R ∗ as defined in equation (2.1): For producer items in the control group, get their positions from the control counterfactual ranking R 0 . For producer items in the treatment group, get their positions from the treatment counterfactual ranking R 1 .

Step 3: Create a valid ranker R based on the SUTVA ranker R ∗ such that:

̸

- Preserve the relative order: For any two producer items d = d ′ , if R ∗ ( d ) &lt; R ∗ ( d ′ ), then R ( d ) &lt; R ( d ′ )

̸

- Break the tie according to the probabilistic rule: For any pair of producer items having merging conflicts in R ∗ (i.e. d = d ′ but R ∗ ( d ) = R ∗ ( d ′ )), break the tie by placing d before d ′ in R with probability β 0 ( d, d ′ ).

In the above procedure, steps 1 and 2 are the same as the existing solutions while step 3 is a different strategy which is proposed to resolve any possible merging conflicts in the counterfactual interleaving design and ensure an unbiased comparison between treatment and control rankers in the producer-side experiments. The key is to rigorously preserve the relative order and employ a non-constant tie-breaking probability β 0 ( d, d ′ ) whose value we will derive based on the principle of consistency and monotonicity next.

For each spot j , consider a pair of producer items x and y satisfying R 0 ( x ) = R 1 ( y ) = j . As discussed at the end of Section 2.3, x and y would have merging conflict R ∗ ( x ) = R ∗ ( y ) if and only if x ∈ G 0 and y ∈ G 1 . When they have merging conflict, the tie-breaking probability for placing x before y in the merged ranker R can be defined as:

<!-- formula-not-decoded -->

Note that although R 0 ( x ) = R 1 ( y ) = j , R 1 ( x ) and R 0 ( y ) can still be larger or smaller than j . In the next theorem, we will show that their values should determine the probability β 0 ( x, y ).

Theorem 5.1. The following β 0 ( x, y ) ensures that the merged ranker R satisfies the Principle of Consistency (i.e., R ( x ) | x ∈G 0 and R ( y ) | y ∈G 1 have identical distributions):

<!-- formula-not-decoded -->

where p 0 represents the % of traffic allocated to the control group and p 1 represents the % of traffic allocated to the treatment group.

Proof of this theorem is given in the Appendix C. In the following theorem, we further prove that the above solution of counterfactual interleaving design also satisfies the monotonicity principle.

Theorem 5.2. The merged ranker R in the counterfactual interleaving design created by the above procedure with the tie-breaking probability β 0 ( d, d ′ ) derived in equation (5.1) at each spot j is both consistent and monotonic.

Proof of this theorem is left in the Appendix D. It is crucial to see that by following the consistency and monotonicity principles, we can obtain a rigorous counterfactual interleaving design solution to ensure valid comparisons between the treatment and conrol rankers in the producer-side experiments.

## 6 Examples

To compare two rankers R 0 and R 1 in the producer-side experiment, we have shown how to create a consistent and monotonic merged ranker in the counterfactual interleaving design in Section 5. In this section, we illustrate the previous theoretical results with both simulated and real examples.

## 6.1 Consistent Convolution Kernels and Convoluted Attention Functions

Consider ten producer items x 0 , . . . , x 9 , and two rankers R 0 and R 1 . Suppose R 0 ranks them as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We first illustrate the convolution kernels and convoluted attention functions as defined in equations (3.6) and (3.7). It is important to note that the convolution kernel π 0 j is simply the distribution R ( x j ) under the condition x j ∈ G 0 , where R represents the consistent merged ranker and x j is a producer item such that R 0 ( x j ) = j .

Let us assume that the traffic allocation are p 0 = 50% to G 0 and p 1 = 50% to G 1 . In this scenario, π 0 j can be computed numerically as depicted in Fig. 10. Each curve in the chart corresponds to a distribution function π 0 j with j indicated in the top-right legend.

Figure 10: Convolution Kernel Examples

<!-- image -->

We can also examine a single convolution kernel, π 0 4 (i.e., j = 4), for different values of traffic allocations p 0 as in Fig. 11. Each curve in this figure represents the function π 0 4 for a specific value of p 0 , as shown in the legend. It is evident that as p 0 approaches 1, the convolution kernel becomes sharper and converges to the delta function, while as p 0 approaches 0 . 5, the function flattens.

while the order under R 1 is Now, let us verify a series of convoluted attention functions with p 0 = 0 . 5 to confirm that they are indeed monotonic. It can be easily demonstrated that any monotonically decreasing attention function can be decomposed into a sum of functions h j , where h j ( j ′ ) = I { j ′ ≤ j } . Let the associated convoluted attention function with the convolution kernels depicted in Fig 10 be h 0 j . These functions are illustrated in Fig 12. It is evident from the figure that the convoluted attention functions are all monotonically decreasing.

Figure 11: Convolution Kernel Examples

<!-- image -->

## 6.2 Simulation Example

Now we illustrate how the existing counterfactual interleaving designs from Section 2.4 can introduce biases in the producer-side experiment readouts and lead to misleading conclusions. For the sake of simplicity, let us consider only four producer items, i.e., x 0 , . . . , x 3 , such that R 0 ( x j ) = j , and their ordering under R 1 is x 1 , x 2 , x 3 , x 0 . Assume the pure (intrinsic utility) metrics u for producer item { x i } are u ( x 0 ) = u ( x 3 ) = 0 . 9 and u ( x 1 ) = u ( x 2 ) = 1. Clearly, since u ( x 1 ) = u ( x 2 ) &gt; u ( x 3 ) = u ( x 0 ), the treatment ranker R 1 is superior to the control ranker R 0 .

̸

Existing counterfactual interleaving designs from Section 2.4 (such as the UniCoRn design from Nandy et al. (2021)) employ a naive strategy to resolve merging conflicts with tie-breaking probability β 0 = 0 . 5, which means that if R ( x ) = R ( y ) but x = y , the merged ranker R will place either x or y fi rst with equal probability. Suppose the attention function h has the following form: h (0) = h (1) = 1 and h (2) = h (3) = 0, which means that the top two spots in the recommender system receive full attention from the consumers, whereas the bottom two do not have consumers' attention.

We simulate N independent replications of the experiment, denoted as S . In each replication s ∈ S , four distinct producer items, x j,s , are recommended to the consumer v s , where the pure metric function u , the attention function h , as well as the rankers R 0 and R 1 (i.e., R 0 ( x j,s ) = j ), are the same across all replications. However, the randomization to assign each producer item x j,s into the control group G 0 or the treatment group G 1 with probability p 0 = P ( x j,s ∈ G 0 ), as well as the randomization to resolve merging conflicts with tie-breaking probability β 0 , are all independent across different replications. For any s ∈ S , the merged ranker R defines the final position of each x j,s .

For each s ∈ S , we can calculate the experiment readout for the control and treatment groups ( k = 0 , 1) under the merged ranker R using equation (3.2):

<!-- formula-not-decoded -->

where 1 /p k is the normalization factor to account for the different sizes of treatment groups (i.e., p k represents the % of traffic allocated to the treatment group k ). Finally, we further take averages of the experiment readouts across N replications for each k = 0 , 1:

Figure 12: Convoluted Attention Functions

<!-- image -->

<!-- formula-not-decoded -->

We can also estimate the variance of agg k,s ( U ) by:

<!-- formula-not-decoded -->

and estimate the standard deviation of agg k as:

<!-- formula-not-decoded -->

Under this simulation setup, we expect the correct result as agg 1 &gt; agg 0 , which indicates that R 1 is a better ranker than R 0 . In the following four simulated cases, Case 1 and 2 are based on the naive UniCoRn design approach from Section 2.4, which lead to wrong conclusions; Case 3 and 4 leverage the proposed counterfactual interleaving design with consistent merged ranker R from Section 5, which could give the correct conclusions.

Case 1: When the traffic allocation to the control group G 0 is p 0 = 0 . 9 and a na¨ ıve tie-breaker with β 0 = 0 . 5 is employed as in the existing UniCoRn design (Section 2.4), the simulation results for N = 10 3 , 10 4 , 10 5 are as follows:

| N     | 10 3     | 10 4     | 10 5     |
|-------|----------|----------|----------|
| agg 0 | 1 . 9609 | 1 . 9447 | 1 . 9502 |
| agg 1 | 1 . 443  | 1 . 5973 | 1 . 5438 |
| SD 0  | 0 . 0126 | 0 . 0041 | 0 . 0013 |
| SD 1  | 0 . 1133 | 0 . 0372 | 0 . 012  |

It shows that agg 0 &gt; agg 1 with high statistical significance, which is misleading. As we have explained earlier, R 1 is actually superior to R 0 .

In this case, the convoluted functions h 0 and h 1 can be represented in the following table:

| j         |   0 | 1       | 2       |   3 |
|-----------|-----|---------|---------|-----|
| h 0 ( j ) |   1 | 0 . 955 | 0 . 095 |   0 |
| h 1 ( j ) |   1 | 0 . 505 | 0 . 045 |   0 |

Using these values, we can also directly compute the expected value of the aggregated experiment readouts agg k ( U ) as defined in equation (3.3). By Theorem 4.1, for k = 0 , 1

<!-- formula-not-decoded -->

and agg 0 ( U ) = 1 . 95, agg 1 ( U ) = 1 . 5455. Such results clearly show that, due to the biases introduced by the inconsistent merged ranker, the aggregated experiment readouts incorrectly favor R 0 over the actually superior R 1 .

One might speculate that such bias would not occur if the traffic was equally distributed between the control and treatment groups, i.e., p 0 = 50%, p 1 = 50%. However, in the next case below, we will show that even in this symmetric scenario, the bias persists.

Case 2: Consider the traffic allocation to the control group G 0 is p 0 = 0 . 5 and a na¨ ıve tie-breaker with β 0 = 0 . 5 is employed as in the UniCoRn approach (Section 2.4). The simulation results are:

| N     | 10 3    | 10 4    | 10 5    |
|-------|---------|---------|---------|
| agg 0 | 2 . 081 | 2 . 139 | 2 . 155 |
| agg 1 | 1 . 808 | 1 . 749 | 1 . 733 |
| SD 0  | 0 . 04  | 0 . 013 | 0 . 004 |
| SD 1  | 0 . 04  | 0 . 013 | 0 . 004 |

In this csae, the convoluted attention functions can be calculated as follows:

| j         |   0 | 1       | 2       |   3 |
|-----------|-----|---------|---------|-----|
| h 0 ( j ) |   1 | 0 . 875 | 0 . 375 |   0 |
| h 1 ( j ) |   1 | 0 . 625 | 0 . 125 |   0 |

and consequently, agg 0 ( U ) = 2 . 15, agg 1 ( U ) = 1 . 7375. Once again, the bias is clearly evident and the experiment readouts incorrectly indicate that R 0 is superior to R 1 with high significance..

In order to achieve a valid comparison between the treatment and control rankers, we should use the proposed counterfactual interleaving design from Section 5, which is guaranteed to yield a consistent merged ranker. In the next two cases, we will show that the proposed consistent ranker can effectively identify the superior ranker in the simulated producer-side experiment.

Case 3: When the traffic allocation to the control group G 0 is p 0 = 0 . 9 and the proposed counterfactual interleaving design with consistent merged ranker R from Section 5 is used, we have

| N     | 10 3    | 10 4     | 10 5     |
|-------|---------|----------|----------|
| agg 0 | 1 . 880 | 1 . 893  | 1 . 900  |
| agg 1 | 2 . 161 | 2 . 06   | 2 . 000  |
| SD 0  | 0 . 015 | 0 . 0045 | 0 . 0014 |
| SD 1  | 0 . 13  | 0 . 041  | 0 . 013  |

Based on the consistent ranker, we can clearly see that agg 1 &gt; agg 0 and R 1 is identified as the optimal ranker.

Case 4: Consider the traffic allocation to the control group G 0 is p 0 = 0 . 5 and the proposed counterfactual interleaving design with consistent merged ranker R from Section 5 is used, we have:

| N     | 10 3    | 10 4    | 10 5    |
|-------|---------|---------|---------|
| agg 0 | 1 . 956 | 1 . 888 | 1 . 904 |
| agg 1 | 1 . 920 | 1 . 987 | 1 . 971 |
| SD 0  | 0 . 04  | 0 . 012 | 0 . 004 |
| SD 1  | 0 . 04  | 0 . 012 | 0 . 004 |

Once again, different from the existing UniCoRn design in Section 2.4, the proposed solution from Section 5 can draw the correct conclusion.

## 6.3 Recommender System Example from Online Social Networks

Online social network platforms play a crucial role in connecting people with one another, offering features such as Feeds and People You May Know (PYMK). A key aspect of these platforms is their ability to recommend a ranked list of users/creators for viewers to follow or connect with. Such recommender systems are vital in shaping the experience of both viewers and creators on the social network, and any new changes to the recommender's ranking algorithm need to be carefully evaluated through online experiments before getting fully deployed in production. In this context, viewers are the consumers who consume content on the network, and standard viewer-side A/B tests can be used to assess the impact of ranking changes on the viewers' behavior. On the other hand, producers on the network are the creators who are ranked and recommended by the platform for viewers to connect or follow. Measuring the impact of ranking changes on the creators through producer-side experiment is equally important for improving the ecosystem of online social networks. Such producer-side experiments are also often referred to as the creator-side experiments .

The AI team at LinkedIn initially implemented the UniCoRn design (Section 2.4) for running creatorside experiments in the online edge recommender system, serving tens of millions of members and providing billions of edge recommendations daily. However, it became evident that the UniCoRn-based approach led to biased creator positions in the final ranking, and readouts from the corresponding creator-side experiments were difficult to interpret. After recognizing the importance of consistency and monotonicity principles in designing creator-side experiments, the team implemented the new counterfactual interleaving design as proposed in Section 5.

Figure 13: Illustration of a Counterfactual Interleaving Design for LinkedIn's Online Edge Recommender System

<!-- image -->

As shown in Figure 13, the ranker in the online edge recommender system at LinkedIn consists of three sequential ranking/filtering layers with increasing complexities: (1) Allow List is a simple rule-based filtering layer, which returns a subset of creators who are eligible to be recommended to the viewer. (2) Candidate Generator (CG) layer scores and ranks all creators in the allow list, and returns the top m candidates. There can be multiple independent CGs in this layer (e.g., one CG for each country or industry segment) and union of all the selected candidates will be sent to the final ranking layer. The ranking models in the CG layer are generally easier to compute and thus they can be used to score and rank a large number of creators in the allow list. (3) Final Ranking layer scores and ranks all the candidates selected by CG layer using more sophisticated models. In the end, top n ( n &lt; m ) creators out of the final ranking is shown to the viewer. Figure 13 illustrates the counterfactual interleaving design of a creator-side experiment at LinkedIn where the treatment is to add a new CG to the existing CGs (e.g., CG1, CG2, . . . , CG9).

Generating the treatment and control counterfactual rankings for the counterfactual interleaving design may not always require running the Allow List, CG and Final Ranking layers twice. In some cases, it is possible to develop computational shortcuts. Take the counterfactual interleaving design in Figure 13 for example. Because the candidates generated in the control CG is a subset of those candidates generated in the treatment CG (while Allow List and Final Ranking layers are the same between treatment and control), the creator-side experiment only needs to run Allow List, CG and Final Scoring/Ranking for the treatment counterfactual case. Then, as shown in Figure 14, the control counterfactual ranking can be directly obtained based on the treatment counterfactual ranking (after removing any candidates that were in the treatment CG but not in the control CG). In other words, the only extra computation needed for generating the control counterfactual ranking is to read a few more creators from the treatment counterfactual ranking list to fill the empty spots at the end of the control counterfactual ranking list. This shortcut can substantially reduce the computation especially when scoring and ranking a large number of creators are expensive.

Figure 14: Illustration of the Computational Shortcut for Generating the Treatment and Control Counterfactual Rankings

<!-- image -->

After deploying the proposed solution in production, a comparison between the new counterfactual interleaving design and the previous UniCoRn design has revealed that the UniCoRn-based design had resulted in approximately 85% of creators being placed in the wrong positions. On average, every creator was randomly shifted away by ± 2 or 3 positions in a recommendation session compared to their correct positions based on the counterfactual rankings. Using the proposed new counterfactual interleaving design, the team was able to obtain fair comparisons between the treatment and control rankers, ensuring trustworthy creatorside experiment readouts. By running both viewer-side (consumer-side) and creator-side (producer-side) experiments and evaluating the ranking changes' impacts on both sides of the marketplace, the team can strike a balance that benefits all stakeholders involved. As recommender systems continue to evolve, these considerations will play an increasingly pivotal role in enhancing user experiences and driving success in online platforms.

## 7 Conclusions

Many online platforms are two-sided marketplaces which have producers (e.g., sellers, content creators, hosts) on one side and consumers (e.g., buyers, content viewers, and customers) on the other side. Recommender systems aim to predict consumer preferences and allocate more preferred producer items to spots where consumers are likely to pay greater attention. To optimize an online recommender system, it is critical to conduct online experiments to thoroughly evaluate the impacts of any new ranking model changes on both sides. While consumer-side impact can be easily measured via simple online A/B testing, producerside measurement is much more challenging. In this paper, we scrutinize issues of the current ad hoc design solutions in the literature and propose general principles for designing trustworthy online producerside experiments. Building upon the proposed consistency and monotonicity principles, we also derive a rigorous counterfactual interleaving design solution to ensure valid comparisons between treatment and control rankers. The proposed methodology and design principles can serve as guidelines for online platforms seeking to improve their recommender systems and ensuring accuracy in their evaluations on the producer side.

In the end, we also want to note that an alternative way to measure producer-side impacts is through cluster-randomized experiments (Karrer et al., 2021; Saveski et al., 2017; Saint-Jacques et al., 2019), where consumers are partitioned into various disjoint clusters and each cluster is associated with one producer. Such solution for measuring producer-side effects has two major limitations: (1) the effective sample size (and hence the power) of cluster-randomized experiments tends to be small; and (2) it is often challenging to partition the network into clusters and different clustering algorithms can lead to different experiment results. Moreover, in some applications (such as the online edge recommender system described in Section 6.3), it is not possible to run cluster-randomized experiments because the new treatment in the experiment would keep changing the edge structure of the online social network.

## 8 Acknowledgements

The authors would like to thank Nian Si, Preetam Nandy, Weitao Duan, James Sorenson, Cindy Liang, Parag Agrawal, Andrew Hatch, Chun Lo, Yafei Wei, Liyan Fang, Wentao Su and Wanjun Liu for their suggestions and feedbacks. The authors also would like to thank the researchers and engineers from the Data Science Applied Research team, Follows AI team and PYMK AI team at LinkedIn.

## A Proof of Theorem 4.1

We first calculate the following sum of conditional expectations:

<!-- formula-not-decoded -->

Now we prove Theorem 4.1 by calculating p k agg k ( U ):

<!-- formula-not-decoded -->

and the proof is complete.

## B Proof of Lemma 4.2

̸

The crux of the proof lies in observing that, since the attention function h is monotonically decreasing by definition, for any ranker R = R u , there must exist producer items d 0 , d 1 ∈ G such that u ( d 0 ) &gt; u ( d 1 ) but R ( d 0 ) &gt; R ( d 1 ), which implies that h ◦ R ( d 0 ) ≤ h ◦ R ( d 1 ). Consider a new ranker, ˜ R , which is identical to R except that the ranks of d 0 and d 1 are swapped. Then,

<!-- formula-not-decoded -->

This inequality indicates that swapping the ranks of d 0 and d 1 leads to a non-decrease in the value of ∑ d ∈G u ( d ) × h ◦ R ( d ). By iteratively swapping such pairs ( d 0 , d 1 ), R can eventually be transformed into R u . Throughout this process, the value of ∑ d ∈G u ( d ) × h ◦ R ( d ) never decreases, thereby establishing R u as the maximizer.

## C Proof of Theorem 5.1

Proof: It is evident that R ( x ) is equal to one plus the number of producer items ranked ahead of x by R :

For the left-hand side:

̸

<!-- formula-not-decoded -->

Considering that R maintains the relative order of R ∗ , and d = y is the sole producer item (apart from x ) for which R ∗ ( d ) = R ∗ ( x ), under the condition x ∈ G 0 (or equivalently, R ∗ ( x ) = j ),

<!-- formula-not-decoded -->

where

and O 0 j = O j | x ∈G 0 . Additionally

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Likewise, under the condition y ∈ G 1 ,

<!-- formula-not-decoded -->

with O 1 j = O j | y ∈G 1 and Q 1 j = I { R ( x ) &lt; R ( y ) } | y ∈G 1 .

The variation in the variable O j is influenced by the treatment allocations of producer items excluding x and y . On the other hand, the variation in the terms Q 0 j and Q 1 j is governed by the allocations of x , y , as well as the random generator, B , responsible for tie-breakings. Therefore, for k = 0 , 1, O k j and Q k j are independent. Furthermore, O 0 j and O 1 j possess identical distributions, as both are identical to the distribution of O j .

For R ( x ) | x ∈G 0 and R ( y ) | y ∈G 1 to exhibit the same distribution (in order to satisfy the Principle of Consistency), we need to ensure that Q 0 j and Q 1 j have identical distributions. Based on the definitions of each term, this implies

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

and similarly for the right-hand side:

<!-- formula-not-decoded -->

where p 0 = |G 0 | |G 0 | + |G 1 | representing the % of traffic allocated to the control group and p 1 = |G 1 | |G 0 | + |G 1 | representing the % of traffic allocated to the treatment group.

By equalizing the above two sides, we can obtain β 0 under the Principle of Consistency as:

<!-- formula-not-decoded -->

This definition ensures that both Q 0 j and Q 1 j follow Bernoulli distributions with identical expected values as shown below:

<!-- formula-not-decoded -->

## D Proof of Theorem 5.2

We start the proof of Theorem 5 . 2 with the following straightforward yet essential results.

Lemma D.1. Let X , Y 1 , and Y 2 be random variables, where X and Y i are independent for i = 1 , 2. If Y 1 ≻ Y 2 , then X + Y 1 ≻ X + Y 2 .

Proof. Consider any number t . It suffices to show that P ( X + Y 1 ≥ t ) ≥ P ( X + Y 2 ≥ t ).

<!-- formula-not-decoded -->

Corollary D.2. Let X 1 , X 2 , Y 1 , and Y 2 be random variables, where X i and Y i are independent for i = 1 , 2. If X 1 ≻ X 2 and Y 1 ≻ Y 2 , then X 1 + Y 1 ≻ X 2 + Y 2 .

Proof. Leveraging Lemma D.1, we can deduce that X 1 + Y 1 ≻ X 1 + Y 2 and X 1 + Y 2 ≻ X 2 + Y 2 . Consequently, X 1 + Y 1 ≻ X 2 + Y 2 .

To streamline the discussion, let's introduce a set of notations. Let X and Y be two random variables, which might not necessarily be independent. Define X ⊕ Y as a random variable Z such that Z = X ′ + Y ′ , where X ′ and Y ′ are identically distributed as X and Y , respectively, and are also independent.

With Corollary D.2 in mind, we can deduce the following corollary:

Corollary D.3. Let X 1 , X 2 , Y 1 , and Y 2 be random variables. If X 1 ≻ X 2 and Y 1 ≻ Y 2 , then X 1 ⊕ Y 1 ≻ X 2 + Y 2 .

For 0 ≤ p ≤ 1, let γ ( p ) denote a random variable following a Bernoulli distribution with mean p . Specifically, γ (0) = 0 and γ (1) = 1. The following lemma is self-evident:

Lemma D.4. Let 0 ≤ p ≤ q ≤ 1, then γ ( q ) ≻ γ ( p ).

For any position j , let us denote by x j , y j the producer items such that R 0 ( x j ) = R 1 ( y j ) = j . According to equation (3.6), the convolution kernel π k j is essentially the distribution of R ( x j ) when x j ∈ G 0 or R ( y j ) when y j ∈ G 1 . Both distributions are identical when R is the consistent merged ranker, and the tie-breakers are computed as described in Section 5.

It is noteworthy that x j and y j may be equal for certain j . In such cases, no tie-breaker is needed at that position because no conflicts arise. Let's define C ( j ) as the set containing x j and y j . When x j = y j , C ( j ) will contain only one element.

By definition, R ( x j ) = 1 + ∑ d ∈G I { R ( d ) &lt; R ( x j ) } , which leads us to

<!-- formula-not-decoded -->

Equation (D.1) is consistent with equation (C.1), but accounts for the case where | C ( j ) | = 1.

As outlined in Appendix C, the two terms O 0 j and Q 0 j are independent, and O 0 j is distributed the same as

<!-- formula-not-decoded -->

O j can be further decomposed into two independent components, both of which are also independent of Q 0 j :

<!-- formula-not-decoded -->

Similarly for O j +1 :

<!-- formula-not-decoded -->

By definition, when d is not an element of the set C ( j ) ∪ C ( j +1), the indicator function I { R ∗ ( d ) &lt; j } is equal to I { R ∗ ( d ) &lt; j +1 } . As a result, ̂ O j is equal to ̂ O j +1 .

Let's introduce T ( j ) = O ′ j ⊕ Q 0 j and T ( j +1) = O ′′ j +1 ⊕ Q 0 j +1 . According to Corollary D.3, in order to prove Theorem 5.2, it is sufficient to demonstrate that for any integer j greater than or equal to 1, T ( j +1) stochastic dominates T ( j ), that is, T ( j +1) ≻ T ( j ).

Regarding Q 0 j , if the size of the set C ( j ) is equal to 1, then Q 0 j is equal to 0. On the other hand, if the size of the set C ( j ) is equal to 2, Q 0 j follows a Bernoulli distribution with an expected value that can be computed using equation (C.2), by substituting x and y with x j and y j , respectively.

For the sake of simplification, let's denote I ( x, j ) as I { R 1 ( x j ) &lt; j } and I ( y, j ) as I { R 0 ( y j ) &lt; j } . It is important to note that both I ( x, j ) and I ( y, j ) are deterministic functions that can only take the values 0 or 1.

Lemma D.5. For any j , if | C ( j ) | = 2, then the distribution of Q 0 j is determined by the values of I ( x, j ) and I ( y, j ) as below:

|   I ( x, j ) |   I ( y, j ) | Q 0 j            |
|--------------|--------------|------------------|
|            0 |            0 | γ ( p 0 p 1 )    |
|            1 |            1 | γ (1 - p 0 p 1 ) |
|            0 |            1 | γ ( p 0 )        |
|            1 |            0 | γ ( p 1 )        |

<!-- formula-not-decoded -->

and the conclusion follows from equation (C.2).

Lemma D.6. For any j , if | C ( j +1) | = 2 and C ( j +1) ∩ C ( j ) = ∅ , then the distribution of O ′ j is determined by the values of I ( x, j +1) and I ( y, j +1) as below:

|   I ( x, j +1) |   I ( y, j +1) | O ′ j                 |
|----------------|----------------|-----------------------|
|              0 |              0 | 0                     |
|              1 |              1 | γ ( p 0 ) ⊕ γ ( p 1 ) |
|              0 |              1 | γ ( p 0 )             |
|              1 |              0 | γ ( p 1 )             |

Proof. By equation (D.2),

<!-- formula-not-decoded -->

Lemma D.7. For any j , if | C ( j ) | = 2 and C ( j +1) ∩ C ( j ) = ∅ , then the distribution of O ′′ j +1 is determined by the values of I ( x, j ) and I ( y, j ) as below:

|   I ( x, j ) |   I ( y, j ) | O ′′ j +1                         |
|--------------|--------------|-----------------------------------|
|            0 |            0 | γ ( p 0 ) ⊕ γ ( p 1 )             |
|            1 |            1 | 2                                 |
|            0 |            1 | γ ( p 0 ) ⊕ γ ( p 0 ) ⊕ γ ( p 1 ) |
|            1 |            0 | γ ( p 1 ) ⊕ γ ( p 1 ) ⊕ γ ( p 0 ) |

̸

Proof. When | C ( j ) | = 2, x j = y j , so Proof. By equation (D.3),

<!-- formula-not-decoded -->

Lemma D.8. If | C ( j ) | = | C ( j +1) | = 1, then T ( j +1) ≻ T ( j ).

̸

Proof. In this case x j = y j , x j +1 = y j +1 while x j = x j +1 . By equation (D.1), Q 0 j = Q 0 j +1 = 0. By equation (D.2), O ′ j = 0. By equation (D.3), O ′′ j +1 = 1. Therefore, T ( j +1) = 1 and T ( j ) = 0.

Lemma D.9. For any j , if | C ( j +1) | = 2 and | C ( j ) | = 1, then T ( j +1) ≻ T ( j ).

Proof. In this case C ( j +1) ∩ C ( j ) = ∅ . So Q 0 j = 0 by equation (D.1) while O ′′ j +1 = 1 by equation (D.3). Meanwhile the distribution of Q 0 j +1 and O ′ j can be obtained by Lemma D.5 (substitute j → j + 1) and Lemma D.6. Consequently we have the distributions of T ( j +1) and T ( j ) as below:

|   I ( x, j +1) |   I ( y, j +1) | T ( j +1)           | T ( j )               |
|----------------|----------------|---------------------|-----------------------|
|              0 |              0 | 1+ γ ( p 0 p 1 )    | 0                     |
|              1 |              1 | 1+ γ (1 - p 0 p 1 ) | γ ( p 0 ) ⊕ γ ( p 1 ) |
|              0 |              1 | 1+ γ ( p 0 )        | γ ( p 0 )             |
|              1 |              0 | 1+ γ ( p 1 )        | γ ( p 1 )             |

Note that 1 -p 1 = p 0 ≥ p 0 p 1 so 1 -p 0 p 1 ≥ p 1 . The conclusion follows from Lemma D.4.

Lemma D.10. If | C ( j ) | = 2 and | C ( j +1) | = 1, then T ( j +1) ≻ T ( j ).

Proof. Still, C ( j +1) ∩ C ( j ) = ∅ . So Q 0 j +1 = 0 by equation (D.1) while O ′ j = 0 by equation (D.2). Meanwhile, the distributions of Q 0 j and O ′′ j +1 can be obtained from Lemma D.5 and Lemma D.7:

|   I ( x, j ) |   I ( y, j ) | O ′′ j +1                         | Q 0 j            |
|--------------|--------------|-----------------------------------|------------------|
|            0 |            0 | γ ( p 0 ) ⊕ γ ( p 1 )             | γ ( p 0 p 1 )    |
|            1 |            1 | 2                                 | γ (1 - p 0 p 1 ) |
|            0 |            1 | γ ( p 0 ) ⊕ γ ( p 0 ) ⊕ γ ( p 1 ) | γ ( p 0 )        |
|            1 |            0 | γ ( p 1 ) ⊕ γ ( p 1 ) ⊕ γ ( p 0 ) | γ ( p 1 )        |

Obviously O ′′ j +1 ≻ Q 0 j so T ( j +1) ≻ T ( j ).

Lemma D.11. If | C ( j ) | = 2, | C ( j +1) | = 2 and C ( j +1) ∩ C ( j ) = ∅ . Then T ( j +1) ≻ T ( j ).

Proof. In this case, by Lemma D.5 and Lemma D.7, we have the distributions of Q 0 j and O ′′ j +1 as below in each case,

|   I ( x, j ) |   I ( y, j ) | O ′′ j +1                         | Q 0 j            |
|--------------|--------------|-----------------------------------|------------------|
|            0 |            0 | γ ( p 0 ) ⊕ γ ( p 1 )             | γ ( p 0 p 1 )    |
|            1 |            1 | 2                                 | γ (1 - p 0 p 1 ) |
|            0 |            1 | γ ( p 0 ) ⊕ γ ( p 0 ) ⊕ γ ( p 1 ) | γ ( p 0 )        |
|            1 |            0 | γ ( p 1 ) ⊕ γ ( p 1 ) ⊕ γ ( p 0 ) | γ ( p 1 )        |

<!-- formula-not-decoded -->

Similarly, by Lemma D.5 (substitute j → j +1) and Lemma D.6, we have the distributions of Q 0 j +1 and O ′ j as below

|   I ( x, j +1) |   I ( y, j +1) | Q 0 j +1         | O ′ j                 |
|----------------|----------------|------------------|-----------------------|
|              0 |              0 | γ ( p 0 p 1 )    | 0                     |
|              1 |              1 | γ (1 - p 0 p 1 ) | γ ( p 0 ) ⊕ γ ( p 1 ) |
|              0 |              1 | γ ( p 0 )        | γ ( p 0 )             |
|              1 |              0 | γ ( p 1 )        | γ ( p 1 )             |

other than the I ( x, j +1) = 1 and I ( y, j +1) = 1 case, Q 0 j +1 ≻ O ′ j . Combining with equation (D.4), we know that T ( j +1) ≻ T ( j ) in all these cases.

The only thing left is to prove for the I ( x, j +1) = 1 and I ( y, j +1) = 1 case. But in this scenario, we always have Q 0 j +1 ≻ Q 0 j and O ′′ j +1 ≻ O ′ j , and T ( j +1) ≻ T ( j ) follows as well.

<!-- formula-not-decoded -->

̸

Proof. Without loss of generality, suppose x j = y j +1 and x j +1 = y j . In this case C ( j +1) \ C ( j ) = { x j +1 } and C ( j ) \ C ( j +1) = { y j } . By equation (D.2),

<!-- formula-not-decoded -->

Meanwhile with equation (D.1) (substitute j → j +1),

<!-- formula-not-decoded -->

By equation (C.2) (substitute j → j +1), together with the fact that x j = y j +1 so R 0 ( y j +1 ) = j &lt; j +1 and I ( y, j +1) = 1, the distribution of Q 0 j +1 and O ′ j can be summarized as in the following table:

|   I ( x, j +1) | Q 0 j +1         | O ′ j     |
|----------------|------------------|-----------|
|              1 | γ (1 - p 0 p 1 ) | γ ( p 1 ) |
|              0 | γ ( p 0 )        | 0         |

Consequently Q 0 j +1 ≻ O ′ j .

Similarly, by equation (D.3)

<!-- formula-not-decoded -->

Meanwhile with equation (D.1),

<!-- formula-not-decoded -->

By equation (C.2), together with the fact that x j = y j +1 so R 1 ( x j ) = j + 1 &gt; j and I ( x, j ) = 0, the distribution of Q 0 j and O ′′ j +1 can be summarized as in the following table:

|   I ( y, j ) | O ′′ j +1   | Q 0 j         |
|--------------|-------------|---------------|
|            0 | γ ( p 1 )   | γ ( p 0 p 1 ) |
|            1 | 1           | γ ( p 0 )     |

Therefore O ′′ j +1 ≻ Q 0 j . Combining this with Q 0 j +1 ≻ O ′ j we get T ( j +1) ≻ T ( j ).

Lemma D.13.

If

|

C

(

j

)

|

= 2,

|

C

(

j

+1)

|

= 2 and

|

C

(

j

+1)

∩

C

(

j

)

|

= 2, then

T

(

j

+1)

≻

T

(

j

).

Proof. In this case x j = y j +1 and y j = x j +1 . So C ( j +1) \ C ( j ) = ∅ and C ( j ) \ C ( j +1) = ∅ . By equation (D.2) and (D.3), O ′ j = O ′′ j +1 = 0. Meanwhile by equation (C.2), Q 0 j dist = γ ( p 0 p 1 ) and Q 0 j +1 dist = γ (1 -p 0 p 1 ) so Q 0 j +1 ≻ Q 0 j and T ( j +1) ≻ T ( j ).

Combining Lemma D.8 to Lemma D.13, Theorem 5.2 is proved.

## References

- Bajari, P., Burdick, B., Imbens, G. W., Masoero, L., McQueen, J., Richardson, T. S. and Rosen, I. M. (2023). Experimental design in marketplaces. Statistical Science , 38 458 - 476. https://doi.org/10.1214/23-STS883
- Bakshy, E., Eckles, D. and Bernstein, M. S. (2014). Designing and deploying online field experiments. In Proceedings of the 23rd International Conference on World Wide Web . WWW'14, Association for Computing Machinery, New York, NY, USA. https://doi.org/10.1145/2566486.2567967
- Ha-Thuc, V., Dutta, A., Mao, R., Wood, M. and Liu, Y. (2020). A counterfactual framework for seller-side a/b testing on marketplaces. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval . Association for Computing Machinery, New York, NY, USA. https://doi.org/10.1145/3397271.3401434
- Imbens, G. W. and Rubin, D. B. (2015). Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction . Cambridge University Press.
- Johari, R., Li, H., Liskovich, I. and Weintraub, G. Y. (2022). Experimental design in two-sided platforms: An analysis of bias. Management Science , 68 7069-7089. https://doi.org/10.1287/mnsc.2021.4247
- Karrer, B., Shi, L., Bhole, M., Goldman, M., Palmer, T., Gelman, C., Konutgan, M. and Sun, F. (2021). Network experimentation at scale. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery &amp; Data Mining . KDD '21, Association for Computing Machinery, New York, NY, USA. https://doi.org/10.1145/3447548.3467091
- Kohavi, R., Deng, A., Frasca, B., Walker, T., Xu, Y. and Pohlmann, N. (2013). Online controlled experiments at large scale. In Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining . KDD '13, Association for Computing Machinery, New York, NY, USA. https://doi.org/10.1145/2487575.2488217
- Kohavi, R., Longbotham, R., Sommerfield, D. and Henne, R. M. (2009). Controlled experiments on the web: survey and practical guide. Data Mining and Knowledge Discovery , 18 140-181. http://link.springer.com/10.1007/s10618-008-0114-1
- Kohavi, R., Tang, D. and Xu, Y. (2020). Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing . Cambridge University Press.
- Nandy, P., Venugopalan, D., Lo, C. and Shaunak, C. (2021). A/b testing for recommender systems in a twosided marketplace. In Advances in Neural Information Processing Systems 34 pre-proceedings (NeurIPS 2021) .
- Parks, J., Aurisset, J. and Ramm, M. (2017). Innovating faster on personalization algorithms at netflix using interleaving. Netflix Technology Blog . https://netflixtechblog.com/interleaving-in-online-experiments-at-netflix-a04ee392ec55
- Radlinski, F. and Craswell, N. (2013). Optimized interleaving for online retrieval evaluation. In Proceedings of the Sixth ACM International Conference on Web Search and Data Mining . WSDM '13, Association for Computing Machinery, New York, NY, USA. https://doi.org/10.1145/2433396.2433429

- Saint-Jacques, G., Varshney, M., Simpson, J. and Xu, Y. (2019). Using ego-clusters to measure network effects at linkedin. arXiv .

[https://doi.org/10.48550/arXiv.1903.08755](https://doi.org/10.48550/arXiv.1903.08755)

- Saveski, M., Pouget-Abadie, J., Saint-Jacques, G., Duan, W., Ghosh, S., Xu, Y. and Airoldi, E. M. (2017). Detecting network effects: Randomizing over randomized experiments. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining . KDD '17, Association for Computing Machinery, New York, NY, USA.

[https://doi.org/10.1145/3097983.3098192](https://doi.org/10.1145/3097983.3098192)

- Tang, D., Agarwal, A., O'Brien, D. and Meyer, M. (2010). Overlapping experiment infrastructure: More, better, faster experimentation. In Proceedings of the 16th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining . KDD '10, Association for Computing Machinery, New York, NY, USA.

[https://doi.org/10.1145/1835804.1835810](https://doi.org/10.1145/1835804.1835810)

- Xu, Y., Chen, N., Fernandez, A., Sinno, O. and Bhasin, A. (2015). From infrastructure to culture: A/b testing challenges in large scale social networks. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining . KDD '15, Association for Computing Machinery, New York, NY, USA.

[https://doi.org/10.1145/2783258.2788602](https://doi.org/10.1145/2783258.2788602)

- Zhang, Q., Du, M., Andersen, R. and He, L. (2022). Beyond a/b test: Speeding up airbnb search ranking experimentation through interleaving. The Airbnb Tech Blog .

[https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentat](https://medium.com/airbnb-engineering/beyond-a-b-test-speeding-up-airbnb-search-ranking-experimentation-through-interleaving-7087afa09c8e)