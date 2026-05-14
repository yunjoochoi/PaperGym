## Harnessing the Power of Interleaving and Counterfactual Evaluation for Airbnb Search Ranking

[Qing Zhang Airbnb](https://orcid.org/0009-0005-3572-2606)

San Francisco, CA, USA qing.zhang@airbnb.com

Huiji Gao

Airbnb

San Francisco, CA, USA huiji.gao@airbnb.com

## ABSTRACT

Evaluation plays a crucial role in the development of ranking algorithms on search and recommender systems. It enables online platforms to create user-friendly features that drive commercial success in a steady and effective manner. The online environment is particularly conducive to applying causal inference techniques, such as randomized controlled experiments (known as A/B test), which are often more challenging to implement in fields like medicine and public policy. However, businesses face unique challenges when it comes to effective A/B test. Specifically, achieving sufficient statistical power for conversion-based metrics can be time-consuming, especially for significant purchases like booking accommodations. While offline evaluations are quicker and more cost-effective, they often lack accuracy and are inadequate for selecting candidates for A/B test. To address these challenges, we developed interleaving and counterfactual evaluation methods to facilitate rapid online assessments for identifying the most promising candidates for A/B tests. Our approach not only increased the sensitivity of experiments by a factor of up to 100 (depending on the approach and metrics) compared to traditional A/B testing but also streamlined the experimental process. The practical insights gained from usage in production can also benefit organizations with similar interests.

## CCS CONCEPTS

· Mathematics of computing → Hypothesis testing and confidence interval computation ; · Applied computing → Electronic commerce .

## KEYWORDS

causal inference, interleaving, counterfactual evaluation, search ranking, recommendation

∗ Work completed while employed by Airbnb

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

KDD '25, August 3-7, 2025, Toronto, ON, Canada

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 979-8-4007-1454-2/2025/08...$15.00

https://doi.org/10.1145/3711896.3737232

Alex Deng ∗

Microsoft

Seattle, WA, USA alexdeng@live.com

Liwei He Airbnb

Seattle, WA, USA liwei.he@airbnb.com

## ACMReference Format:

Qing Zhang, Alex Deng, Michelle Du, Huiji Gao, Liwei He, and Sanjeev Katariya. 2025. Harnessing the Power of Interleaving and Counterfactual Evaluation for Airbnb Search Ranking. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '25), August 3-7, 2025, Toronto, ON, Canada. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3711896.3737232

## 1 INTRODUCTION

Web platforms extensively utilize data-driven approaches to refine their ranking algorithms for search or recommendation systems. By presenting users with different variations, we can measure the impact of changes based on user actions directly. A/B testing is a widely used method for this purpose. In a typical A/B testing scenario, visitors to a website are randomly assigned to either a control group or a treatment group. The control group is exposed to the baseline version of the website, which usually reflects the current live production environment, while the treatment group experiences the versions produced by the new, proposed algorithm. The effectiveness of the treatment is evaluated by comparing key business metrics, such as conversion rates, between the two groups. Additionally, a comprehensive set of debugging metrics, including funnel conversion rates, user engagement levels, and characteristics of the results, are analyzed to gain deeper insights into the behavior and performance of the ranking algorithms. A/B testing has become a cornerstone for fostering continuous innovations, playing a crucial role in their ability to adapt and improve [21].

A significant volume of work has been dedicated to enhancing the effectiveness of A/B testing. Notably, [7] proposed leveraging pre-experiment data to reduce metric variance and improve sensitivity. This approach has been widely adopted across the industry, as seen in works such as [22, 33]. Additionally, in-experiment data has been utilized to develop surrogate metrics that aim to achieve high sensitivity and provide early readings on business metrics, which often experience delayed outcomes [5]. The extensive use of A/B testing in the industry has led to the accumulation of valuable practical lessons, which have been studied in depth [8, 19, 20].

A/B testing alone, however, is often insufficient due to the long running times required for experiments on e-commerce platforms like Airbnb. There are three main reasons. First, users typically visit the site with a certain level of intent to make a purchase. Since Airbnb users generally travel only twice a year, the traffic

Michelle Du Airbnb

San Francisco, CA, USA michelle.du@airbnb.com

Sanjeev Katariya Airbnb

San Francisco, CA, USA sanjeev.katariya@airbnb.com for experiments is significantly lower compared to search engines. Second, while conversion is usually the target metric, it takes time to realize. The higher the stakes, the longer the search journey; for instance, an Airbnb user might spend several days searching for accommodation that fits their preferences. In contrast, search engines receive immediate feedback from user clicks. Third, as the system matures, the effect size of each innovation tends to be small, which necessitates an even longer time to detect statistically significant changes [1].

A logical question arises: why don't we use offline evaluations to identify the most promising candidates for A/B testing? Indeed, offline evaluation is often employed as a preliminary step for model assessment. This process involves collecting search logs, which include the results displayed and the corresponding user actions, and using them to evaluate the proposed ranker. While this approach is efficient and risk-free, it tends to lack accuracy. The primary reason is that the ranker we aim to evaluate only has the visibility of what has been shown by the logging ranker but not all the candidate items. To address this selection bias, techniques such as inverse propensity weighting (IPW) [26], based on importance sampling [13], have been proposed. However, these techniques often result in high variance, as noted by [10]. In addition, obtaining the propensity score (the probability of a result being displayed) is complicated due to system complexity [4].

In addition, offline metrics are frequently disconnected from online business metrics. For example, the Normalized Discounted Cumulative Gain (NDCG) [15] is a standard offline metric, while conversion rate serves as the primary online metric. Often, these two metrics are inconsistent. Furthermore, offline evaluations cannot fully account for the user dynamics that occur when individuals interact with ranked lists.

To bridge the gap between the two approaches, we seek a middle step that is faster than A/B testing while being more accurate than offline evaluation. Interleaving experiments, first proposed in [16, 17], offer a potential solution. However, several open questions remain. Firstly, the interleaving methods developed in prior work primarily used clicks as user feedback, whereas our target metric is conversion, which is much sparser, as previously noted. Additionally, we must consider scalability, as the method will be implemented in production and handle real traffic; thus, both complexity and latency need to remain within acceptable limits. Consequently, it is unclear whether the state-of-the-art interleaving mechanisms can be efficiently and effectively applied to ranking problems in e-commerce platforms like Airbnb.

In this paper, we share our recent advances in enhancing the experimentation velocity for Airbnb's search ranking. We present our innovations in interleaving experiment design and the engineering framework. Following this, we detail an online counterfactual evaluation approach that is more generalizable and addresses the limitations of interleaving. Both techniques are utilized for selecting treatment candidates for A/B testing. These systems are used by engineers working on search ranking at Airbnb, and the validation of these techniques is based entirely on real-world usage, as opposed to the dataset-based simulations commonly used in previous work. Our contributions are,

- Competitive pair based interleaving that's unbiased and highly efficient, and we observed 50X speedup improvement compared to A/B in production. The speedup is computed with traffic needed to achieve similar statistical power as A/B test.
- Online counterfactual evaluation which further improved the sensitivity on top of interleaving and more generalizable. The metrics demonstrated up to 100X speedup compared to A/B.
- Practical lessons from the usage in production that provides full picture of of the techniques.
- Both interleaving and counterfactual evaluation approaches presented in the paper can be fairly easily generalized to other platforms.

To the best of our knowledge, this is the first work where both approaches have been implemented in production, evaluated side by side and used on a daily basis. The paper provides a comprehensive comparison, detailing both the advancements and limitations of each approach and an experimentation strategy with interleaving, counterfactual evaluation, and A/B test to improve overall experimentation velocity in practice, which will prove invaluable for businesses facing similar challenges.

## 2 PRELIMINARIES

## 2.1 Problem Definition

First, we define the notations and formulate the evaluation problem. We use 𝜋 to denote the ranking algorithm (also referred to as the policy). Given a set of candidate listings with features 𝑋 (which include listing attributes, past engagements, queries, and user history), the algorithm generates a ranked list represented as 𝐿 ∼ 𝜋 ( 𝐿 | 𝑋 ) . After presenting this list to the user, we observe the reward 𝑂 ∼ 𝑝 ( 𝑂 | 𝑋, 𝐿 ) , which can include events such as clicks and purchases. 𝑂 can be an empty list when there is no user action for the 𝐿 . The value of the policy 𝜋 is defined as the expected reward 𝑉 ( 𝜋 ) = 𝐸 ( 𝑓 ( 𝑂 ) ∗ 𝜋 ( 𝑂 | 𝑋 )) , where 𝑓 maps rewards into numeric value. Given a proposed policy 𝜋 1 and a baseline policy 𝜋 0, the evaluation problem involves designing an estimator for 𝑉 ( 𝜋 ) and using the difference 𝜏 = 𝑉 ( 𝜋 1 ) -𝑉 ( 𝜋 0 ) to assess the impact of policy 𝜋 1 compared to policy 𝜋 0.

The intuition of comparison can be further developed with potential outcome framework [14]. Formally, let 𝑊 be the assignments, and each element 𝑤 𝑖 indicates the group that subject 𝑖 belongs to, specifically control when 𝑤 𝑖 = 0 and treatment when 𝑤 𝑖 = 1. Also, let 𝑌 denote the outcome (reward), where 𝑌 𝑖 ( 0 ) represents the outcome when control is applied to instance 𝑖 , and similarly 𝑌 𝑖 ( 1 ) treatment. For a moment, let's assume we can observe outcomes from both groups then we can compute the impact 𝑌 𝑖 ( 1 ) -𝑌 𝑖 ( 0 ) on each element, and get average treatment effect

<!-- formula-not-decoded -->

Therefore 𝑉 ( 𝜋 𝑤 ) = ¯ 𝑌 ( 𝑤 ) here. In common settings we cannot observe both 𝑌 𝑖 ( 0 ) and 𝑌 𝑖 ( 1 ) at the same time. For example a user is either exposed to the treatment or control policy, but not both. Thanks to the randomized controlled experiment, such as A/B test, the difference between the observed outcome ˆ 𝜏 is unbiased for 𝜏 [14],

<!-- formula-not-decoded -->

where superscript 𝑜𝑏𝑠 indicates that these are observed outcomes. ¯ 𝑌 𝑜𝑏𝑠 𝑐 = 1 𝑁 𝑐 ˝ 𝑖 : 𝑤 𝑖 = 0 𝑌 𝑜𝑏𝑠 𝑖 , and similarly ¯ 𝑌 𝑜𝑏𝑠 𝑡 = 1 𝑁 𝑡 ˝ 𝑖 : 𝑤 𝑖 = 1 𝑌 𝑜𝑏𝑠 𝑖 .

Interleaving and counterfactual evaluation, on the other hand, connects with the original form Eq 1 by examining the 𝑌 𝑖 ( 1 ) -𝑌 𝑖 ( 0 ) . We will have more discussion in later sections.

Percent delta relative to baseline is computed as following,

<!-- formula-not-decoded -->

In rest of the paper we omit the step of Eq 3 when discussing metrics for simplicity and assume it is always the final step.

## 2.2 Design Principles

When examining and designing the online evaluation techniques, we are guided by the following principles that were proposed and refined in previous work [1, 12, 17, 24],

- Sensitivity . A primary objective of an evaluation approach is to achieve the desired statistical power with the minimal amount of data necessary.
- Unbiasness . When a user randomly interact with ranked result, we should expect there is no preference according to the estimator.
- Fidelity . The estimator should align with the intuition when user operate on the original results.
- Minimaluser experience disruption . The user experience during the evaluation should mirror the experience that they would typically have when using the product under normal conditions.
- Acceptable complexity and scalability . The system is going to be integrated into a large-scale user-facing search framework, necessitating efficiency in both operation and maintenance.
- Generalizability . Given the dynamic nature of the search system and evolving business requirements, the evaluation methodology must be flexible and easily extendable.

It is expected that an evaluation approach may perform well in some criteria while underperforming in others. Therefore, trade-offs have to be made based on the specific use case.

## 3 RELATED WORK

## 3.1 Interleaving Experiments

Interleaving is an online testing methodology first proposed in [16, 17]. The central idea is to provide the same user with two variants that we want to compare (e.g., results from two rankers) and to infer their preferences based on the user actions, which indicate the quality difference between the two. The key components of this methodology include a merging algorithm to blend the two result lists and a credit assignment mechanism. The technique enables the assessment of ranker relevance through user events, such as clicks, without adding any overhead for the user. Moreover, it allows for a direct comparison of two rankers by the same user. This development is significant, as it marked a shift away from the traditional reliance on human annotators for Web search evaluation.

There are primarily three types of interleaving methods. The first type, Balanced Interleaving (BI), was introduced in [16, 17]. This method ensures that the top 𝑘 results in the merged result list 𝐼 consistently include the top 𝑘 𝑎 results from ranker A and 𝑘 𝑏 results from ranker B, with 𝑘 𝑎 and 𝑘 𝑏 differing by no more than one [3]. As a result, the merged list evenly distributes impressions between the two rankers. For credit assignment, each click within 𝐼 is attributed to both A and B, provided it appears in their respective lists and is above a certain position threshold. The ranker accumulating more clicks is deemed superior. However, as [3] points out, balanced interleaving can yield biased results when the two rank lists are almost identical, differing only by a slight shift or insertion. To address this bias, [1] developed an interleaving method that debiases BI by incorporating IPW for credit attribution. The process requires, for each ranker, computing the probability of receiving a click at each position in the merged list. Despite its effectiveness, the BI+IPW method poses complexities in credit attribution and less extensible compared to our approach.

The second type of interleaving method is Team Drafting Interleaving (TD), as introduced in [25]. This method utilizes a merging algorithm that mimics the process of drafting in sports teams. It iterates through both ranked lists from top to bottom, selecting the highest-ranked available item to add to the combined list 𝐼 . Each item in 𝐼 is assigned to a "team," indicating its origin from either ranker. Preferences are determined by counting which "team" has gathered more clicks. This approach addresses the bias issue identified with BI. [3] suggested several refined schemes for credit assignment. Our work builds upon TD, offering more efficient credit computation and increased generalizability. [12] pointed out that TD potentially violates fidelity in certain cases, our practical experience in production has not revealed any issue stemming from these cases. Moreover, our research into counterfactual evaluation indicates that such scenarios have a negligible effect on the overall evaluation. We will explore this topic in greater detail in Section 5.4.

The third type of interleaving method, Probabilistic Interleaving (PI), was introduced in [11, 12] with the aim of improving upon BI and TD. Unlike BI and TD, where the merged list 𝐼 is constructed from a fixed order of items from A and B, PI uses softmax functions s(A) and s(B) to transform these lists into probability distributions over documents, from which items are then sampled to create list 𝐼 . The credit computation in PI considers all possible sequences of drafting that could result in the formation of list 𝐼 . While PI is unbiased in its approach, it has the potential to significantly alter the user experience and introduces greater system complexity for production use.

## 3.2 Interleaving in Practice

There is limited literature on how interleaving is used in production. Most of the previous work are research projects ran on limited datasets, such as [11, 12]. There were experiments conducted in Microsoft and Yahoo!, such as [3] but consistency with A/B test were not discussed. The study most relevant to our work is [1] by Amazon, which uses BI as base algorithm and applied IPW to correct the bias, as mentioned earlier. It was evaluated by comparing the results with 10 A/B tests. Our approach is more efficient and we report our comparison with A/B with much larger corpus.

## 3.3 Counterfactual Evaluation

When evaluating policy 𝜋 , data collected from another policy 𝜋 0 is often used. It is usually in the form of past search logs, leading to the terms "off-policy" and "offline evaluation" being used interchangeably. Three categories of work have emerged in this field. The first category involves directly modeling (DM) the reward and using it to predict the outcome of the target policy for each search. This approach typically results in a low variance but high bias estimator [18].

The second category is the model-free approach. As mentioned in Section 1, Inverse Propensity Weighting (IPW) method is proposed to correct the probability of events observed in historical data. For instance, if an item has a probability 𝜋 of being shown according to the target policy, the outcome is weighted by 𝜋 𝜋 0 , where 𝜋 0 represents the probability of being shown according to the logging policy [13, 26]. While IPW is unbiased, it suffers from high variance. A series of techniques have been developed to address the challenges in IPW based approaches, including Clipped Inverse Propensity Score [29] and Self-Normalized IPS estimator [30].

In the third category, the Doubly Robust Estimator [9] combines the DM and IPW estimators. This approach is both unbiased and consistent while exhibiting lower variance than IPS. Furthermore, several variations have been developed based on the doubly robust estimator [23, 27, 28, 32].

The aforementioned work primarily focuses on multi-arm bandit evaluation, which cannot be directly applied to search ranking due to the large action space. However, in the online counterfactual evaluation discussed in this paper, we incorporate elements of IPW and reward estimation.

Recent development has used counterfactual result to decompose the target metric and reduce the variance [6]. Specifically counterfactual results are utilized to categorize the events into high and low signal-to-noise ratio portions and it enables the weighting between the two portions. Our work is built on top of the approach.

## 4 INTERLEAVING WITH COMPETITIVE PAIR

As outlined in Section 1, our existing experimentation process comprises two steps. Initially, experimenters utilize offline evaluation with historical data during the early stages of iteration. This approach is fast and cost-effective, requiring only 1-2 hours for Airbnb cases, though it lacks accuracy. Subsequently, the promising candidates identified in the first step proceed to A/B testing, which is constrained by limited bandwidth and requires weeks to complete. This bottleneck could be alleviated by introducing a middle stage that that is much faster than A/B and more accurate than offline evaluation. Interleaving, known for its high sensitivity [3], emerges as a viable option for the stage. We tackle the online evaluation challenges 1) low frequency transaction and 2) conversion as target metric by efficient team draft design and innovative credit attribution.

## 4.1 Methodology

In Section 3.1, we examined three variations of interleaving: BI, TD, and PI. The complexity of these methods, particularly in terms of their merging algorithms and credit attribution processes, can be ordered from most to least complex as PI &gt; BI &gt; TD. Given that our system is intended to operate in a production environment and cater to all search ranking experimentations, we chose TD for its efficiency and extensibility.

4.1.1 Competitive Pair and Credit Attribution. In the previous work [25], the team drafting process involves teams taking turns to select the next item not yet included in the merged list 𝐼 . A coin flip at each turn determines which team picks first. We design team drafting with a notion of competitive pair and the coin flip only happens once at the beginning procedure. In every turn, we draft the next available item from each team. If different, they form a competitive pair, and are added 𝐼 with the order determined by the coin flip. The team assignment is logged accordingly. If the items are identical, the item is added to 𝐼 without being assigned to any team. This procedure is detailed in Algorithm 1. The design is highly efficient for serving and preference calculation. To illustrate, let's walk through an example with two ranked lists 𝐶 = { 𝑎, 𝑏, 𝑐, 𝑑, 𝑒 } and 𝑇 = { 𝑏, 𝑐, 𝑎, 𝑓 , 𝑔 } , assuming 𝑖𝑠𝐶𝑓 𝑖𝑟𝑠𝑡 = 𝑡𝑟𝑢𝑒 . The first few steps of team drafting process would unfold as follows:

- Draft 𝑎 from C and 𝑏 from T. As the two items are different, they form a competitive pair and are placed in 𝐼 , as [ 𝑎 𝐶 , 𝑏 𝑇 ] (line 7 - 9). The super script indicates the team assignment.
- The next available items are 𝑐 from 𝐶 and 𝑐 from 𝑇 . Since they are the same, 𝑐 is added to 𝐼 without a team assignment. (line 13 - 14)
- This process continues until the end condition is met.

Following this procedure, the output would be 𝐼 = { 𝑎 𝐶 , 𝑏 𝑇 , 𝑐, 𝑑 𝐶 , 𝑓 𝑇 } . Our design guarantees that each team has an equal opportunity to be selected first, ensuring that rankers C and T have identical chances of displaying their listings in any given position. The approach effectively eliminates position bias in our measurements. To maintain a consistent user experience, we construct list 𝐼 with a length equal to the minimum of 𝑙 𝑐 and 𝑙 𝑡 , the lengths of the lists generated by rankers C and T, respectively. In our production environment, 𝑙 𝑐 and 𝑙 𝑡 are equal except in rare cases, as it is from the same user search request.

4.1.2 Preference. Team preference is determined by counting the victories of each competitive pair and then aggregating these results to the desired level of analysis. In our case, we consider the individual user as the unit of analysis (although search requests could also serve this purpose). For each user 𝑖 , the outcome is identified based on which team-C or T-has more wins. We define 𝑌 𝑖 ( 𝑤 ) = 𝑤𝑖𝑛𝑠 ( 𝑤 ) , where 𝑤 can either be team C or team T and the function 𝑤𝑖𝑛𝑠 ( . ) simply counts the number of wins for the corresponding team. The preference measure, 𝜏 𝑖 , is then calculated as the difference in wins between team T and team C, expressed as 𝜏 𝑖 = 𝑌 𝑖 ( 𝑇 ) -𝑌 𝑖 ( 𝐶 ) . This method provides a clear and quantifiable metric for team preference at the user level.

<!-- formula-not-decoded -->

The p-value is computed by proportional test on ˝ 𝑖 1 ( 𝜏 𝑖 &gt; 0 ) and ˝ 𝑖 1 ( 𝜏 𝑖 &lt; 0 ) , which are the number of subjects who prefers T and C respectively.

```
Input: ranked list 𝐶 and 𝑇 , coin filp result 𝑖𝑠𝐶𝑓 𝑖𝑟𝑠𝑡 Output: Interleaved list 𝐼 1 𝑙 𝑐 , 𝑙 𝑡 ← 𝑙𝑒𝑛 ( C ) , 𝑙𝑒𝑛 ( T ) 2 𝑙 𝐼 ← 𝑚𝑖𝑛 ( 𝑙 𝑐 , 𝑙 𝑡 ) 3 𝑘 𝑐 , 𝑘 𝑡 ← 0 , 0 4 𝐼 ←∅ 5 while 𝑘 𝑐 ! = -1& 𝑘 𝑡 ! = -1& 𝑘 𝑐 < 𝑙 𝑐 & 𝑘 𝑡 < 𝑙 𝑡 & 𝑙𝑒𝑛 ( 𝐼 ) < 𝑙 𝐼 do 6 if 𝐴 [ 𝑘 𝑐 ] ! = 𝐵 [ 𝑘 𝑡 ] then 7 if 𝑖𝑠𝐶𝑓 𝑖𝑟𝑠𝑡 then 8 𝐼 ← 𝐼 ∪ 𝐶 [ 𝑘 𝑐 ] 𝐶 9 𝐼 ← 𝐼 ∪ 𝑇 [ 𝑘 𝑡 ] 𝑇 10 else 11 𝐼 ← 𝐼 ∪ 𝑇 [ 𝑘 𝑡 ] 𝑇 12 𝐼 ← 𝐼 ∪ 𝐶 [ 𝑘 𝑐 ] 𝐶 13 else 14 𝐼 = 𝐼 ∪ 𝐶 [ 𝑘 𝑐 ] 15 𝑘 𝑐 ← 𝑛𝑒𝑥𝑡𝐴𝑣𝑎𝑖𝑙𝑎𝑏𝑙𝑒 ( 𝐶,𝑘 𝑐 , 𝐼 ) 16 𝑘 𝑡 ← 𝑛𝑒𝑥𝑡𝐴𝑣𝑎𝑖𝑙𝑎𝑏𝑙𝑒 ( 𝑇,𝑘 𝑡 , 𝐼 ) 17 return 𝐼
```

Algorithm 1: Competitive Pair Team Drafting

Using competitive pairs allows for a direct comparison between two items from the two rankers, with each pair acting as the basis for attributing credit. This approach minimizes noise in determining preferences, especially when the event is sparse, such as conversion. It has demonstrated high sensitivity in our validation process.

4.1.3 Unbiasness. If users interact with 𝐼 by clicking randomly, there is no team preference based on the credit attribution scheme. To illustrate it, we use competitive pairs as the base unit to derive the user preference. A user who take actions randomly will click first and second item in the pair with probability 𝑃 ( 𝐶 = 1 | 𝑟 = 1 ) and 𝑃 ( 𝐶 = 1 | 𝑟 = 2 ) , where 𝑟 is rank of the listing and 𝐶 = 1 if clicked, and 0 otherwise. The expectation of the listing from A (or B) get clicked is 𝑃 ( 𝐶 = 1 | 𝑟 = 1 ) 𝑃 ( 𝑟 = 1 ) + 𝑃 ( 𝐶 = 1 | 𝑟 = 2 ) 𝑃 ( 𝑟 = 2 ) . As each ranker has the equal chance of being ranked at the first according to team draft algorithm, so we have 𝑃 ( 𝑟 = 1 ) = 𝑃 ( 𝑟 = 2 ) = 0 . 5. By aggregating all pairs, we will get that ranker A and B will have an equal number of expected wins.

Careful readers may question the unbiasness when only the first item from the last competitive pair is returned, given our control over result length by 𝑙 𝐼 in Algorithm 1. Since each team has an equal chance of having this lone item, no bias is introduced, as confirmed by the data quality checks detailed in the following section and Section 6.2.3.

4.1.4 Data Quality Monitoring. In addition to assessing the business impact of the treatment ranker, our credit assignment algorithm serves an important role in data quality checks, which are essential for accurate experimentation. Specifically, the algorithm can be applied to evaluate the distribution of impressions and the frequency with which each team appears first in a competitive pair. We anticipate neutral outcomes since each team should receive an equal number of impressions, and there should be a 50% chance for either team to appear first in the competitive pair. The methodology enables us to verify the unbiased nature of each experiment. Should these quality metrics fail to meet our expectations, the results of the experiment would be deemed invalid, indicating that the team drafting algorithm did not perform as expected. To our knowledge, this approach has not been previously proposed.

## 4.2 Architecture Design

In order to support the interleaving, we designed two-layer experiment delivery scheme. The first layer divides the traffic, which are users in our case, into regular A/B test and interleaving. The users assigned to A/B portion will be exposed to A/B tests as usual. For whose who are assigned to the interleaving portion, the second layer maps them to the corresponding interleaving experiment (Figure 1). Within the interleaving framework, each experiment slot is referred to as a "lane, " highlighting the fact that all necessary traffic for interleaving is contained within this slot, unlike in the A/B setup, which requires a control arm in addition. When the

Figure 1: Interleaving delivery system. Twos layers of randomization are used. The first layer decides which user is subject to interleaving and the second assigns the user to the specific interleaving experiment.

<!-- image -->

search system receives a query from a user assigned to interleaving, a parallel call component initiates control and treatment search requests simultaneously. These requests proceed through the entire search stack, producing individual search responses. Subsequently, the team drafting algorithm is employed to merge these responses into a final result list, which is then presented to the user.

## 4.3 Interleaving in Production

The interleaving system has been integrated into Airbnb's search ranking experiment process and has been utilized to conduct over a hundred experiments. Due to its high sensitivity, interleaving is conducted with a small fraction of traffic and over a much shorter duration than A/B testing. The computational complexity, including latency, is well within the threshold required by the search system.

## 5 COUNTERFACTUAL EVALUATION

While interleaving is recognized for its high sensitivity, there are specific scenarios where its application is limited. First, when a ranker extensively uses set level optimization, such as improving diversity of the results, the interleaving would break the assumption. Second, when search list is also used for generating other results on the website, such as map view at Airbnb, there is risk to disrupt user experience. Lastly, it is not straightforward to implement semantically meaningful metric for continuous value based outcome such as revenue. A promising approach to overcome these limitations while preserving the benefits of search-level pairwise comparison is to avoid result blending altogether. This can be achieved by creating counterfactual results within the A/B testing paradigm, thereby maintaining the same user experience as outside of evaluation while still allowing for effective comparison and evaluation.

Next, let's delve deeper into the rationale behind the approach. As previously mentioned, in an A/B test, participants are divided into two groups. For those who are in control group, they will always see results from control ranker, and similarly treatment group user will see the treatment ranker results. Then we compare the conversion rate of treatment vs control group. Interleaving takes a nuanced approach to this comparison. For each search query, it generates results from both the control and treatment rankers, merges these results, and then displays the combined list to the user. Counterfactual evaluation serves as a hybrid of A/B testing and interleaving. It leverages the concept of generating paired results for each search, similar to interleaving, yet evaluates these results using metrics computed in a manner akin to A/B testing. The relationship between the three types of evaluations are illustrated in Figure 2.

Figure 2: Counterfactual evaluation intuition. It can be understood as combining the characteristics of A/B tests and interleaving. It uses the results from both ranker for metrics computation, but doesn't need to blend them together.

<!-- image -->

In counterfactual evaluation, there is notion of shown and counterfactual results. For each search, we generate results 𝐿 𝑐 and 𝐿 𝑡 based on both control and treatment ranker. If 𝐿 𝑐 shown to the user, then 𝐿 𝑡 is the counterfactual result, and the roles reverse if 𝐿 𝑡 is shown instead. For the ease of description, we use 𝑤 denote the shown ranker and 1 -𝑤 as counterfactual.

For each user who are subject to the experiment, we flip a coin to decide which result to show. The randomization is seeded by user ID and experiment ID. Such design ensures the consistent user experience and minimize the carryover effect when running back to back experiments.

We present online counterfactual evaluation below, which is a direct extension of interleaving. It leverages parallel calls similar to interleaving to obtain the results from both control and treatment. Subsequently we utilize the counterfactual results to analyze the shown results, as opposite to interleaving which blends these two sets of results. This approach allows us to derive metrics that are more sensitive than those typically used in A/B testing. In [6], direct decomposition on target metrics based on counterfactual result is proposed. With it as foundation, we present a novel approach that's based on relative position and estimated outcome in the counterfactual, which is proved to be more sensitive.

## 5.1 Direct Decomposition Based Estimators

First we describe the Direct Decomposition approach [6]. Let 𝑌 ( 𝑤 ) denote the outcome of each group, where 𝑤 is the assignments, and its value 0 and 1 represent control and treatment respectively. For any given item associated with an event (for example, a booking), its ranking position is denoted as 𝑟 𝑖 ( 𝑤 ) when subjected to the treatment 𝑤 . We can categorize the outcome into two types,

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Where 𝑘 and 𝛼 are hyperparameters encoding the mapping of ranking difference and true conversion impact. We simply use the values that's discussed in [6], which are 𝑘 = 4 and 𝛼 = 2. The direct decomposition based estimator is,

<!-- formula-not-decoded -->

𝜏 𝑑𝑖𝑓 𝑓 and 𝜏 𝑠𝑖𝑚 are the average treatment effect aggregated on user level by following Equation 1. As detailed in [6], 𝜏 𝑑𝑖𝑓 𝑓 has much smaller variance than 𝜏 𝑠𝑖𝑚 , and significant variance reduction can be achieved from the re-weighting. We use 𝜃 = 0 . 2 in production.

## 5.2 Estimated Reward Based Estimators

The direct decomposition method categorizes target metrics based on the ranked positions generated by both the shown and counterfactual rankers. This approach utilizes estimators that are dependent on both the absolute and the relative positions of the item in question. Building on this, we introduce a novel type of estimators that focuses solely on the difference in positions.

Define 𝑓 : Z + → R as a booking probability model. For a search 𝑖 that has booking event, let 𝑌 𝑖 ( 𝑤 ) = 1, then 𝑌 𝑖 ( 1 -𝑤 ) = 𝑓 ( 𝑟 𝑖 ( 1 -𝑤 ))/ 𝑓 ( 𝑟 𝑖 ( 𝑤 )) , and the gain between shown and counterfactual is

<!-- formula-not-decoded -->

In our implementation, we chose to use exponential function for 𝑓 , with 𝛾 as decay factor, 𝑓 ( 𝑟 ) = 𝛾 -𝑟 . It is based on the observation that the lower the listing is ranked, the less likely that the user is going to interact with it.

We also incorporate the notion of similar ranking proposed in direct decomposition, specifically when | 𝑟 𝑖 ( 𝑤 ) -𝑟 𝑖 ( 1 -𝑤 )| &lt; = 𝛼 we consider the item is ranked at similar position between the shown and counterfactual. Formally, we compute the gain as

<!-- formula-not-decoded -->

Based on win/lose status, we have a pair of estimators

- 𝜏 𝑤𝑖𝑛 𝑖 ( 𝑤 ) , 𝜏 𝑙𝑜𝑠𝑠 𝑖 ( 𝑤 ) = 𝑔 𝑖 , 0, if 𝑟 𝑖 ( 1 -𝑤 ) -𝑟 𝑖 ( 𝑤 ) -𝛼 &gt; 0 (If shown ranker 𝑤 ranks better, it gets 𝑔 𝑖 as the gain at winning position)
- 𝜏 𝑤𝑖𝑛 𝑖 ( 𝑤 ) , 𝜏 𝑙𝑜𝑠𝑠 𝑖 ( 𝑤 ) = 0 , 𝑔 𝑖 , if 𝑟 𝑖 ( 𝑤 ) -𝑟 𝑖 ( 1 -𝑤 ) -𝛼 &gt; 0 (If counterfactual ranker 1 -𝑤 ranks better, shown ranker gets 𝑔 𝑖 as the gain at losing position)

When | 𝑟 𝑖 ( 𝑤 ) -𝑟 𝑖 ( 1 -𝑤 )| is within 𝛼 , item rankings are deemed similar, so 𝑔 = 0 accordingly. Regarding 𝛾 , a smaller value corresponds to a faster decay of attention curve and results in larger incremental gain. The estimator doesn't use the absolute position, but simply the difference. We count difference of overall wins as gain estimator,

<!-- formula-not-decoded -->

We also designed win-loss estimator, which is to count for each treatment, the difference of the event that's in winning position and losing position.

<!-- formula-not-decoded -->

We aggregate the metrics across the users.

## 5.3 OEC (Overall Evaluation Criteria) Metric

We use a combination of direct decomposition and estimated reward based metrics to form the OEC metric (main metric) with the purpose of combining the potential benefit of both estimators. Specifically, we define

<!-- formula-not-decoded -->

We currently simply assign equal weights, with 𝛽 1 = 𝛽 2 = 0 . 5.

## 5.4 Connection with Interleaving

In [12], TD interleaving was pointed out to lack fidelity as well as sensitivity when there is a shift or insertion between two lists. For example, with 𝐶 = { 𝑎, 𝑏, 𝑐, 𝑑 } and 𝑇 = { 𝑏, 𝑐, 𝑑, 𝑎 } , then 𝐼 = { 𝑎 𝐶 , 𝑏 𝑇 , 𝑐, 𝑑 } (C first), or 𝐼 = { 𝑏 𝑇 , 𝑎 𝐶 , 𝑐, 𝑑 } (T first). If the user booked listing 𝑐 , we do not infer any preference as it is assigned to neither team. However, if we look at the original list, T is the better ranker because it ranks 𝑐 higher than C. As we've seen in counterfactual evaluation discussed earlier, shift-by-one is considered no gain when we set equal zone threshold 𝛼 &gt; 0. In the validation Section 6.3.2, we show that 𝛼 = 2 works better than even 𝛼 = 1 (we did not collect data for 𝛼 = 0), which supports the conclusion from interleaving that 𝐶 and 𝑇 is a tie if the booked item is 𝑐 .

The findings of counterfactual evaluation deepened our understanding of the relationship between fidelity and sensitivity in interleaving. The fidelity violation actually doesn't impact TD's sensitivity.

## 5.5 Event Attribution

Until now, our discussion of the event has been somewhat abstract. In this section, we aim to clarify how event is attributed. The click event serves as the most straightforward example, establishing a direct link between the search result and the user's action of clicking. However, our primary interest lies in conversion; therefore, booking is the key event we focus on. The search journey of an Airbnb user often spans multiple sessions, meaning a listing that gets booked can appear in more than one searches. Insights from previous research on Airbnb's ranking system [31] suggest that a user's decision to book a listing is influenced by every instance the listing appears during their search journey, not merely the final one. Consequently, for both interleaving and counterfactual evaluation, we attribute the booking event to all occurrences where the listing was presented to the user during the experiment period, acknowledging the compound effect of repeated exposure on the booking decision.

## 6 VALIDATION AND ANALYSIS

For both interleaving and counterfactual evaluations, we gather corresponding A/B test results, when available, to serve as ground truth for validation. Note that often times what eventually tested in A/B is different from interleaving/counterfactual evaluation, as the experimenter would do final adjustments based on the evaluation results. We only picked the cases with no or minor adjustment for validation.

Our aim is to achieve readings that are not only consistent with A/B test results but also demonstrate higher sensitivity. To assess the consistency between the proposed evaluation methods and A/B test outcomes, we focus on the correlation coefficient between point estimates from both evaluation approaches. To this end, we compile a validation corpus comprising two lists of point estimates: one from our proposed evaluation method, denoted as 𝑀 𝐸 , and the other from A/B test results, denoted as 𝑀 𝐴 / 𝐵 . For each ranker 𝑖 , there exists a corresponding point estimate in both lists, obtained from the proposed evaluation method and the A/B test, respectively. The correlation coefficient between these two sets of point estimates is calculated using standard methods [2], providing a quantitative measure of the consistency between our proposed evaluation techniques and traditional A/B testing,

<!-- formula-not-decoded -->

Where 𝐶𝑜𝑣 ( 𝑀 𝐸 , 𝑀 𝐴 / 𝐵 ) is the covariance between 𝑀 𝐸 and 𝑀 𝐴 / 𝐵 . 𝜎 𝑀 𝐸 and 𝜎 𝑀 𝐴 / 𝐵 are variance within each result list.

## 6.1 Baselines

In our study, we conducted a comparison of our methodologies against the two most relevant contemporary approaches in the field. Firstly, for the interleaving method, we benchmarked our speed improvements against the findings reported by [1]. Like us, [1] evaluated their interleaving approach in the context of its performance relative to traditional A/B testing. Secondly, for the counterfactual evaluation method, we compared our approach to [6], because the work in is also aimed at search ranking evaluation, making it a directly comparable study to our own. Through these comparisons, we aim to highlight the advancements our approaches bring to the field.

## 6.2 Interleaving

6.2.1 Consistency with A/B. Wecollected 29 interleaving - A/B test pairs, and the point estimates of interleaving and A/B test on our target conversion metric is plotted in Figure 3. Overall interleaving and A/B are directionally aligned 82% of the time. The correlation coefficient is 0.6.

Through the usage, interleaving proved to be highly sensitive, as we observed about 50X speedup from A/B. We had a test ranker whose logic was to pick a random listing in the result list and put it to the top. The A/B test took weeks to conclude and the interleaving can detect the negative conversion impact using one day's data on a fraction of traffic. The BI based interleaving from [1] reported 60X speedup based on a corpus size of 10. We consider the results are comparable and our approach is much more efficient computationally.

Figure 3: Interleaving and A/B Test point estimates. Axis ticks are omitted.

<!-- image -->

6.2.2 Case Study. Weare particularly interested in the inconsistent pairs. Our cases suggested a limitation of interleaving when the set level optimization is involved. For example, there was a treatment ranker that optimizes another objective other than conversion. The aim was to remain neutral in terms of conversion while improving the secondary objective. Initially, listings were sorted according to their booking probability. The ranker then rearranged some of these listings to better align with the secondary objective. When users were directly presented with this re-ranked list, they were likely to book listings based on the estimated trade-off between the primary and secondary objectives. However, the scenario changed when we interleaved these treatment results with control results. Users tended to select listings with a higher booking probability from the control group as they looked more attractive when placed side by side with treatment listings in competitive pair, leading to the treatment ranker's underperformance. This discrepancy was evident as we observed a significant negative impact in the interleaving results, whereas the conversion remained neutral in the A/B test.

6.2.3 Unbiasness Validation. As discussed in Section 4.1.4, thanks to the extensibility of competitive pair based TD, we are able to compute data quality metrics in the same way as conversion. They provide the validation on unbiasness for each interleaving experiment and Table 1 demonstrates the results from a past experiment. We expect no preference between two teams, which is confirmed by the metrics.

Table 1: Unbiasness validation

| Metric                | Δ      |   pval |
|-----------------------|--------|--------|
| listings shown        | 0.00%  |   0.91 |
| shown first           | -0.01% |   0.85 |
| shown reciprocal rank | -0.02% |   0.85 |
| listings found        | 0.00%  |   0.95 |

## 6.3 Counterfactual Evaluation

Similar to Interleaving, we collected 30 online counterfactual experiments whose treatments were later tested in A/B to evaluate the consistency and study the effect of hyperparameters.

6.3.1 Consistency with A/B. The main metric 𝜏 𝑜𝑒𝑐 point estimate, which is plotted in Figure 4, has correlation coefficient 0.65 with A/B. Therefore overall the consistency matches the interleaving.

Figure 4: Online counterfactual evaluation and A/B point estimates. Axis ticks are omitted.

<!-- image -->

We show individual metrics point estimate correlation with A/B tests in Table 2. 𝜏 𝑔 performs the best. 𝜏 𝑑𝑖𝑓 𝑓 is much higher correlation than 𝜏 𝑠𝑖𝑚 , which is consistent with findings in [6]. 𝜏 𝑤𝑖𝑛 -𝑙𝑜𝑠𝑠 is between the 𝜏 𝑔 and 𝜏 𝑑𝑖𝑓 𝑓 .

Table 2: Metrics point estimate correlations with A/B test

| Metric       |   Corr with A/B |
|--------------|-----------------|
| 𝜏 𝑔          |            0.66 |
| 𝜏 𝑤𝑖𝑛 - 𝑙𝑜𝑠𝑠 |            0.58 |
| 𝜏 𝑑𝑖𝑓 𝑓      |            0.55 |
| 𝜏 𝑠𝑖𝑚        |           -0.29 |

We studied the experiments in which the counterfactual evaluation didn't work well and they are mainly due to shared real time user signals. For each search, as shown and counterfactual results are derived from the same user, the user related features are identical for ranker 𝐶 and 𝑇 . When two rankers have drastically different capability of utilizing such features, the stronger one would utilize the engagement earned by the weaker ranker and gain an unfair advantage. Therefore we may want to discount the trustworthiness when the control and treatment ranker pair falls into such case.

6.3.2 Effect of Hyperparameters. In reward based approach discussed in Section 5.2, there are two hyperparmeters. The first one is the position decay factor 𝛾 , quantifying how fast user's attention decrease along with the ranked position. We compared the 𝜏 𝑔 and A/B correlation in respect to 𝛾 , and the results are similar. For 𝛾 = 0 . 9 and 𝛾 = 0 . 95, the correlation coefficients are 0.644 and 0.648 respectively.

The second hyperparameter we examine is ranked position similarity threshold 𝛼 , and our observations, as detailed in Table 3, indicate that setting 𝛼 = 2 yields better correlations. This finding suggests that minor differences in relative position do not effectively differentiate ranking quality, regardless of the listing's overall rank.

Table 3: Metric variations correlation with A/B ( 𝛾 = 0 . 9 )

| Metric       |   𝛼 = 1 |   𝛼 = 2 |
|--------------|---------|---------|
| 𝜏 𝑔          |    0.64 |    0.66 |
| 𝜏 𝑤𝑖𝑛 - 𝑙𝑜𝑠𝑠 |    0.58 |    0.60 |

6.3.3 Sensitivity. We observed significant speed up compared to A/B. To achieve our target minimal detectable effect, the counterfactual evaluation requires much less traffic. We observed around 15X speed up for 𝜏 𝑜𝑒𝑐 , 23X for 𝜏 𝑤𝑖𝑛 -𝑙𝑜𝑠𝑠 . The most significant speed up is from 𝜏 𝑔 , whose speedup is around 100X. The findings support our hypothesis that by adding additional knowledge from the counterfactual result of the same search, we can measure the impact with much less traffic.

## 6.4 Interactions and Carryover Effects

The risk of interaction between different evaluations is minimized through our experiment delivery strategy, which is outlined in Section 4.2. The majority of search traffic is allocated to A/B testing, while the remainder is designated for interleaving or counterfactual evaluation methods. Within this smaller segment, we further divide the traffic into distinct lanes, with each experiment being assigned its own lane. This structured approach ensures that each evaluation method operates within its own controlled environment, significantly reducing the potential for cross-experiment interference.

The carryover effect refers to the influence of a previous experiment on the current one when they run back-to-back. Our approach minimizes this effect through randomization design. Let us consider an assignment group 𝐺 𝑖 , where 𝑖 ∈ { 𝐶,𝑇 } represents the control or treatment groups from the previous experiment, and let 𝑢 denote a user. In the case of interleaving, the carryover effect would occur if ∀ 𝑢 ∈ 𝐺 𝑖 , the condition 𝑖𝑠𝐶𝑓 𝑖𝑟𝑠𝑡 is consistently true or false. However, this is avoided because randomization occurs at the search level - each search flips a coin to decide which ranker goes first in team drafting. It was further validated through experiments that the carryover effect wasn't observed.

For counterfactual evaluation, we employ randomization based on user ID and experiment ID. The latter ensures that the assignment for the current experiment is independent of the assignment for the previous experiment. Additionally, we have analyzed backto-back experiments and found no evidence of carryover effects.

## 7 DISCUSSIONS

## 7.1 Implementation Choice

Interleaving and counterfactual evaluation presents a promising direction of evaluation in ranking. The central idea is to have the visibility of the ranked lists from both ranker 𝜋 1 and 𝜋 0. For interleaving, the two lists are combined and shown to the user, so the speedup comes from the comparison in each competitive pair. The counterfactual evaluation does not interfere with what is going to be shown to the user, and simply uses the counterfactual results to create more sensitive reward estimators.

Interleaving and counterfactual evaluation metrics exhibit similar prediction power to the A/B tests, so generally speaking they both are well-suited for pre-A/B test online evaluations. Interleaving, with straightforward credit computation, has shown higher sensitivity compared to a subset of counterfactual evaluation metrics, which is a clear advantage.

Ontheother hand, counterfactual evaluation demonstrates greater robustness for rankers with strong set level optimization. In use cases like we discussed in the Section 6.2 about re-ranking for optimizing secondary objective in Airbnb search, counterfactual evaluation would not suffer from the bias according to later experiments of similar nature, as user will always see the full results from control or treatment.

Therefore the choice of the technology is depending on the use cases, as well as the experiment bandwidth availability.

## 7.2 Generalization

Both interleaving and counterfactual evaluation presented in the paper can be fairly easily applied to other businesses. They can be well applied to the scenarios when traffic (users) and the user action event are abundant, such as engagement-targeted (e.g. click through rate) optimization on search and recommendation. They would, in particular, show strength when traffic and/or events are limited, for instance, e-commerce platforms where the conversion is the target metric. Traditional A/B testing in such an environment demands prolonged periods, ranging from weeks to months, to gather sufficient data for reliable statistical power. Conversely, the methods we present require significantly less traffic and a shorter duration to yield meaningful results. We provide further guidelines on the implementation in the Appendix A.

## 8 CONCLUSION AND FUTURE WORK

The paper presented our innovation in speed up Airbnb search ranking experimentation. Our version of interleaving is efficient and highly sensitive, and we extended it to develop online counterfactual evaluation which addresses the limitations of interleaving and more generalizable. Both approaches are proved to be effective online evaluation technique for treatment candidate selection for A/B test based on the large scaled usage at Airbnb. The techniques can be easily adopted by other online platforms.

Since implementation of these systems, we conducted hundreds of experiments for which we observed an increase in capacity to test new ideas and generally higher success rates in A/B testing. Furthermore, we leverage this framework for conducting model studies, including ablation tests and initial explorations for new projects.

For future directions, we aim to improve the accuracy of predicting outcomes from counterfactual results by incorporating data collected during the experiment itself. In our current approach in estimated reward based estimator, we rely on assumptions about outcomes associated with the counterfactual ranker. However, a more precise prediction model can be developed once data from the ongoing experiment, or on-policy data, is accessible. By analyzing user feedback from this data, we can refine our predictions, thereby enhancing the overall sensitivity of our evaluation method.

## REFERENCES

- [1] Nan Bi, Pablo Castells, Daniel Gilbert, Slava Galperin, Patrick Tardif, and Sachin Ahuja. 2022. Debiased balanced interleaving at Amazon Search. In Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management . 2913-2922.
- [2] George Casella and Roger Berger. 2001. Statistical inference, Second Edition . Wadsworth Group.
- [3] Olivier Chapelle, Thorsten Joachims, Filip Radlinski, and Yisong Yue. 2012. Largescale validation and analysis of interleaved search evaluation. ACM Transactions on Information Systems (TOIS) 30, 1 (2012), 1-41.
- [4] Minmin Chen, Alex Beutel, Paul Covington, Sagar Jain, Francois Belletti, and Ed H Chi. 2019. Top-k off-policy correction for a REINFORCE recommender system. In Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining . 456-464.

- [5] Alex Deng, Michelle Du, Anna Matlin, and Qing Zhang. 2023. Variance Reduction Using In-Experiment Data: Efficient and Targeted Online Measurement for Sparse and Delayed Outcomes. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining . 3937-3946.
- [6] Alex Deng, Luke Hagar, Nathaniel T Stevens, Tatiana Xifara, and Amit Gandhi. 2024. Metric Decomposition in A/B Tests. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining . 4885-4895.
- [7] Alex Deng, Ya Xu, Ron Kohavi, and Toby Walker. 2013. Improving the sensitivity of online controlled experiments by utilizing pre-experiment data. In Proceedings of the sixth ACM international conference on Web search and data mining . 123-132.
- [8] Pavel Dmitriev, Brian Frasca, Somit Gupta, Ron Kohavi, and Garnet Vaz. 2016. Pitfalls of long-term online controlled experiments. In 2016 IEEE international conference on big data (big data) . IEEE, 1367-1376.
- [9] Miroslav Dudík, John Langford, and Lihong Li. 2011. Doubly robust policy evaluation and learning. arXiv preprint arXiv:1103.4601 (2011).
- [10] Alexandre Gilotte, Clément Calauzènes, Thomas Nedelec, Alexandre Abraham, and Simon Dollé. 2018. Offline a/b testing for recommender systems. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining . 198-206.
- [11] Katja Hofmann, Shimon Whiteson, and Maarten De Rijke. 2011. A probabilistic method for inferring preferences from clicks. In Proceedings of the 20th ACM international conference on Information and knowledge management . 249-258.
- [12] Katja Hofmann, Shimon Whiteson, and Maarten De Rijke. 2013. Fidelity, soundness, and efficiency of interleaved comparison methods. ACM Transactions on Information Systems (TOIS) 31, 4 (2013), 1-43.
- [13] Daniel G Horvitz and Donovan J Thompson. 1952. A generalization of sampling without replacement from a finite universe. Journal of the American statistical Association 47, 260 (1952), 663-685.
- [14] Guido W Imbens and Donald B Rubin. 2015. Causal inference in statistics, social, and biomedical sciences . Cambridge university press.
- [15] Kalervo Järvelin and Jaana Kekäläinen. 2002. Cumulated gain-based evaluation of IR techniques. ACM Transactions on Information Systems (TOIS) 20, 4 (2002), 422-446.
- [16] Thorsten Joachims. 2002. Optimizing search engines using clickthrough data. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining . 133-142.
- [17] Thorsten Joachims et al. 2003. Evaluating Retrieval Performance Using Clickthrough Data.
- [18] Thorsten Joachims and Adith Swaminathan. 2016. Counterfactual evaluation and learning for search, recommendation and ad placement. In Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval . 1199-1201.
- [19] Ron Kohavi and Nanyu Chen. 2024. False positives in a/b tests. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining . 5240-5250.
- [20] Ron Kohavi, Roger Longbotham, Dan Sommerfield, and Randal M Henne. 2009. Controlled experiments on the web: survey and practical guide. Data mining and knowledge discovery 18 (2009), 140-181.
- [21] Ron Kohavi, Diane Tang, and Ya Xu. 2020. Trustworthy online controlled experiments: A practical guide to a/b testing . Cambridge University Press.
- [22] Kevin Liou and Sean J Taylor. 2020. Variance-weighted estimators to improve sensitivity in online experiments. In Proceedings of the 21st ACM Conference on Economics and Computation . 837-850.
- [23] Alberto Maria Metelli, Alessio Russo, and Marcello Restelli. 2021. Subgaussian and differentiable importance sampling for off-policy evaluation and learning. Advances in neural information processing systems 34 (2021), 8119-8132.
- [24] Filip Radlinski and Nick Craswell. 2013. Optimized interleaving for online retrieval evaluation. In Proceedings of the sixth ACM international conference on Web search and data mining . 245-254.
- [25] Filip Radlinski, Madhu Kurup, and Thorsten Joachims. 2008. How does clickthrough data reflect retrieval quality?. In Proceedings of the 17th ACM conference on Information and knowledge management . 43-52.
- [26] Paul R Rosenbaum and Donald B Rubin. 1983. The central role of the propensity score in observational studies for causal effects. Biometrika 70, 1 (1983), 41-55.
- [27] Yi Su, Maria Dimakopoulou, Akshay Krishnamurthy, and Miroslav Dudík. 2020. Doubly robust off-policy evaluation with shrinkage. In International Conference on Machine Learning . PMLR, 9167-9176.
- [28] Yi Su, Lequn Wang, Michele Santacatterina, and Thorsten Joachims. 2019. Cab: Continuous adaptive blending for policy evaluation and learning. In International
25. Conference on Machine Learning . PMLR, 6005-6014.
- [29] Adith Swaminathan and Thorsten Joachims. 2015. Batch learning from logged bandit feedback through counterfactual risk minimization. The Journal of Machine Learning Research 16, 1 (2015), 1731-1755.
- [30] Adith Swaminathan and Thorsten Joachims. 2015. The self-normalized estimator for counterfactual learning. advances in neural information processing systems 28 (2015).
- [31] Chun How Tan, Austin Chan, Malay Haldar, Jie Tang, Xin Liu, Mustafa Abdool, Huiji Gao, Liwei He, and Sanjeev Katariya. 2023. Optimizing Airbnb Search Journey with Multi-task Learning. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining . 4872-4881.
- [32] Yu-Xiang Wang, Alekh Agarwal, and Miroslav Dudık. 2017. Optimal and adaptive off-policy evaluation in contextual bandits. In International Conference on Machine Learning . PMLR, 3589-3597.
- [33] Huizhi Xie and Juliette Aurisset. 2016. Improving the sensitivity of online controlled experiments: Case studies at netflix. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining . 645654.

## A IMPLEMENTATION GUIDELINES FOR ADOPTION

We would like to provide suggestions on two areas that are key to adaption, which are event attribution and hyperparameter tuning.

## A.1 User Event attribution

When applying our methods, practitioners need to carefully design the logic that attributes events, such as bookings, to the appearance of items, like listings shown in search or recommendation feeds. In cases where the booked listing appears in multiple search results, deciding on an attribution window becomes necessary. Options include attributing the booking to the appearance in the last search, to searches within the last two days, or to all searches within the experiment period. The choice could be based on the analysis on user decision making process.

## A.2 Counterfactual evaluation hyperparameters

The selection of hyperparameters, specifically the attention decay factor 𝛾 and the similarity threshold 𝛼 , is contingent upon the product's interface. For instance, a horizontal layout typical of recommendation systems may necessitate different parameter values compared to a vertical layout, which is common for search results. These parameters are crucial for accurately modeling user behavior and must be tailored to the specific characteristics of the product. Concretely, a possible procedure of tuning the parameters is as follows.

- Initial value 𝛾 0 will be determined by curve fitting on the click or booking distribution across ranked positions, then we pick candidate values centered around 𝛾 0. Subsequently we will mainly use meta-analysis to compare the metrics correlations that's corresponding to each value with A/B tests.
- 𝛼 value will be determined by meta-analysis on values such as {0, 1, 2, 3}.