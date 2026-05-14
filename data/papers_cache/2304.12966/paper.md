## Towards Theoretical Understanding of Inverse Reinforcement Learning

Alberto Maria Metelli 1 Filippo Lazzati 1 Marcello Restelli 1

## Abstract

Inverse reinforcement learning (IRL) denotes a powerful family of algorithms for recovering a reward function justifying the behavior demonstrated by an expert agent. A well-known limitation of IRL is the ambiguity in the choice of the reward function, due to the existence of multiple rewards that explain the observed behavior. This limitation has been recently circumvented by formulating IRL as the problem of estimating the feasible reward set , i.e., the region of the rewards compatible with the expert's behavior. In this paper, we make a step towards closing the theory gap of IRL in the case of finite-horizon problems with a generative model. We start by formally introducing the problem of estimating the feasible reward set, the corresponding PAC requirement, and discussing the properties of particular classes of rewards. Then, we provide the first minimax lower bound on the sample complexity for the problem of estimating the feasible reward set of order Ω ´ H 3 SA /epsilon1 2 ` log ` 1 δ ˘ ` S ˘ ¯ , being S and A the number of states and actions respectively, H the horizon, /epsilon1 the desired accuracy, and δ the confidence. We analyze the sample complexity of a uniform sampling strategy ( US-IRL ), proving a matching upper bound up to logarithmic factors. Finally, we outline several open questions in IRL and propose future research directions.

## 1. Introduction

Inverse reinforcement learning (IRL) aims at efficiently learning a desired behavior by observing an expert agent and inferring their intent encoded in a reward function (refer to Osa et al. (2018); Arora &amp; Doshi (2021); Adams et al. (2022) for recent surveys on IRL). This ab- stract setting, that diverges from standard reinforcement learning (RL, Sutton &amp; Barto, 1998), as the reward function has to be learned, arises in a large variety of real-world tasks. In particular, in a human-in-the-loop (Wu et al., 2022) scenario, when the expert is represented by a human solving a task, an explicit specification of the reward function representing the human's goal is often unavailable. Experience suggests that humans are uncomfortable when asked to describe their intent and, thus, the underlying reward; while they are much more comfortable providing demonstrations of what is believed to be the right behavior. Indeed, human behavior is usually the product of many, possibly conflicting, objectives. 1 Succeeding in retrieving a representation of the expert's reward has notable implications. First, we obtain explicit information for understanding the motivations behind the expert's choices ( interpretability ). Second, the reward can be employed in RL to train artificial agents, under shifts in the features of the underlying system ( transferability ).

* Equal contribution 1 Politecnico di Milano, 32, Piazza Leonardo da Vinci, Milan, Italy. Correspondence to: Alberto Maria Metelli &lt; albertomaria.metelli@polimi.it &gt; .

Proceedings of the 39 th International Conference on Machine Learning , Honolulu, Hawaii, USA. PMLR 202, 2023. Copyright 2023 by the author(s).

Since the beginning, the community recognized that the IRL problem is, per se, ill-posed , as multiple reward functions are compatible with the expert's behavior (Ng &amp; Russell, 2000). This ambiguity was heterogeneously addressed by the algorithmic proposals that have followed over the years, which realized in several selection criteria, including maximum margin (Ratliff et al., 2006), maximum entropy (Zeng et al., 2022), minimum Hessian eigenvalue (Metelli et al., 2017). Some of these approaches come with theoretical guarantees on the sample complexity, although according to different performance indexes (e.g., Abbeel &amp; Ng, 2004; Syed &amp; Schapire, 2007; Pirotta &amp; Restelli, 2016).

A promising line of research that aspires to overcome the ambiguity issue has been recently investigated in (Metelli et al., 2021; Lindner et al., 2022). These works focus on estimating all the reward functions compatible with the expert's demonstrated behavior, namely the feasible rewards . Remarkably, this viewpoint that focuses on the feasible reward set , rather than on one reward obtained with a specific selection criterion, as previous works did, circumvents the ambiguity problem, postponing the reward selection and pointing to the expert's intent. Although these works provide sample complexity guarantees in different settings, a rigorous understanding of the inherent complexity of the IRL problem is currently lacking.

1 In RL, the Sutton's hypothesis (Sutton &amp; Barto, 1998) conjectures that a scalar reward is an adequate notion of goal.

Contributions In this paper, we aim at taking a step toward the theoretical understanding of the IRL problem. As in (Metelli et al., 2021; Lindner et al., 2022), we consider the problem of estimating the feasible reward set. We focus on a generative model setting, where the agent can query the environment and the expert in any state, and consider finite-horizon decision problems. The contributions of the paper can be summarized as follows.

- We propose a novel framework to evaluate the accuracy in recovering the feasible reward set, based on the Hausdorff metric (Rockafellar &amp; Wets, 1998). This tool generalizes existing performance indexes. Furthermore, we show that the feasible reward set enjoys a desirable Lipschitz continuity property w.r.t. the IRL problem (Section 3).
- We devise a PAC (Probability Approximately Correct) framework for estimating the feasible reward set, providing the definition of p /epsilon1, δ q -PAC IRL algorithm. Then, we investigate the relationships between several performance indexes based on the Hausdorff metric (Section 4).
- Weconceive, based on the provided PAC requirements introduced, a novel sample complexity lower bound of order Ω ´ H 3 SA /epsilon1 2 ` log ` 1 δ ˘ ` S ˘ ¯ . This represents the most significant contribution and, to the best of our knowledge, it is the first lower bound that values the importance of the relevant features of the IRL problem. From a technical perspective, the lower bound construction merges newproof ideas with reworks of existing techniques (Section 5).
- We analyze a uniform sampling exploration strategy (UniformSampling-IRL, US-IRL ) showing that, in the generative model setting, it matches the lower bound up to logarithmic factors (Section 6).

The complete proofs of the results presented in the main paper are reported in Appendix B.

## 2. Preliminaries

In this section, we provide the background that will be employed in the subsequent sections.

Mathematical Background Let a, b P N with a ď b , we denote with /llbracket a, b /rrbracket : ' t a, . . . , b u and with /llbracket a /rrbracket : ' /llbracket 1 , a /rrbracket . Let X be a set, we denote with ∆ X the set of probability measures over X . Let Y be a set, we denote with ∆ X Y the set of functions with signature Y Ñ ∆ X . Let p X , d q be a (pre)metric space, where X is a set and d : X ˆ X Ñ r 0 , `8s is a (pre)metric. 2 Let Y , Y 1 Ď X be non-empty sets, we define the Hausdorff (pre)metric (Rockafellar &amp; Wets, 1998) H d : 2 X ˆ 2 X Ñ r 0 , `8s between Y and Y 1 induced by the (pre)metric d as follows:

<!-- formula-not-decoded -->

Markov Decision Processes without Reward A timeinhomogeneous finite-horizon Markov decision process without reward (MDP \ R) is defined as a 4-tuple M ' p S , A , p, H q where S is a finite state space ( S ' | S | ), A is a finite action space ( A ' | A | ), p ' p p h q h P /llbracket H /rrbracket is the transition model where for every stage h P /llbracket H /rrbracket we have p h P ∆ S S ˆ A , and H P N is the horizon. An MDP \ R is time-homogeneousif, for every stage h P /llbracket H ´ 1 /rrbracket , we have p h ' p h ` 1 a.s.; in such a case, we denote the transition model with the symbol p only. A time-inhomogeneous reward function is defined as r ' p r h q h P /llbracket H /rrbracket , where for every stage h P /llbracket H /rrbracket we have r h : S ˆ A Ñr´ 1 , 1 s . 3 A Markov decision process (MDP, Puterman, 1994) is obtained by pairing an MDP \ R M with a reward function r . The agent's behavior is modeled with a time-inhomogeneous policy π ' p π h q h P /llbracket H /rrbracket where for every stage h P /llbracket H /rrbracket , we have π h P ∆ A S . Let f P R S and g P R S ˆ A , we denote with p h f p s, a q ' ř s 1 P S p h p s 1 | s, a q f p s 1 q and with π h g p s q ' ř a P A π h p a | s q g p s, a q the expectation operators w.r.t. the transition model and the policy, respectively.

Value Functions and Optimality Given an MDP \ R M , a policy π , and a reward function r , the Q-function Q π p¨ ; r q ' p Q π h p¨ ; r qq h P /llbracket H /rrbracket induced by r represents the expected sum of rewards collected starting from p s, a, h q P S ˆ A ˆ /llbracket H /rrbracket and following policy π thereafter:

where E p M ,π q denotes the expectation w.r.t. M and π , i.e., a h ' π h p¨| s h q and s h ` 1 ' p h p¨| s h , a h q for every stage h P /llbracket h, H /rrbracket . The Q-function fulfills the Bellman equations (Puterman, 1994) for every p s, a, h q P S ˆ A ˆ /llbracket H /rrbracket :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where V π p¨ ; r q ' p V π h p¨ ; r qq h P /llbracket H /rrbracket is the V-function . The advantage function A π h p s, a ; r q ' Q π h p s, a ; r q ´ V π h p s ; r q represents the relative gain of playing action a P A rather than following policy π in the state-stage pair p s, h q . A policy π ˚ is optimal if it has non-positive advantage ev-

<!-- formula-not-decoded -->

2 A premetric d satisfies the axioms: d p x, x 1 q ě 0 and d p x, x q ' 0 for all x, x 1 P X . Any metric is clearly a premetric.

3 For the sake of simplicity and w.l.o.g., we restrict to reward functions bounded by 1 in absolute value.

erywhere, i.e., A π ˚ h p s, a ; r q ď 0 for every p s, a, h q P S ˆ A ˆ /llbracket H /rrbracket . The Q- and V-functions of an optimal policy are denoted with Q ˚ h p s, a ; r q and V ˚ h p s ; r q .

Inverse Reinforcement Learning An inverse reinforcement learning problem (IRL, Ng &amp; Russell, 2000) is defined as a pair p M , π E q , where M is an MDP \ R and π E is an expert's policy . Informally, solving an IRL problem consists in finding a reward function p r h q h P /llbracket H /rrbracket making π E optimal for the MDP \ R M paired with reward function r . Any reward function fulfilling this condition is called feasible and the set of all such reward functions is called feasible reward set (Metelli et al., 2021; Lindner et al., 2022), defined as:

We will omit the subscript p M , π E q whenever clear from the context.

<!-- formula-not-decoded -->

Empirical MDP and Empirical Expert's Policy Let D 'tp s l ,a l ,h l ,s 1 l ,a E l qu l P /llbracket t /rrbracket be a dataset of t P N tuples, where for every l P /llbracket t /rrbracket , we have s 1 l ' p h l p¨| s l ,a l q and a E l ' π E h l p¨| s l q . We introduce the counts for every p s,a,h qP S ˆ A ˆ /llbracket H /rrbracket : n t h p s,a,s 1 q : ' ř t l ' 1 1 tp s l ,a l ,h l ,s 1 l q'p s,a,h,s 1 qu , n t h p s,a q : ' ř s 1 P S n t h p s,a,s 1 q , n t h p s q : ' ř a P A n t h p s,a q , and n t,E h p s,a q : ' ř t l ' 1 1 tp s l ,a E l q'p s,a qu . These quantities allow defining the empirical transition model p p t 'p p p t h q h P /llbracket H /rrbracket and empirical expert's policy p π t,E 'p π t,E h q h P /llbracket H /rrbracket as follows:

In the time-homogeneous case, we simply merge the samples collected at different stages h P /llbracket H /rrbracket . We denote with p x M t , p π E,t q the empirical IRL problem, where x M t ' p S , A , p p t , H q the empirical MDP \ Rinduced by p p t . Finally, we denote with p R t : ' R p x M t , p π E,t q the feasible reward set induced p x M t , p π E,t q . We will omit the superscript t , whenever clear from the context and write p R .

<!-- formula-not-decoded -->

## 3. Lipschitz Framework for IRL

In this section, we analyze the regularity properties of the feasible reward set in terms of the Lipschitz continuity w.r.t. the IRL problem. To make the idea more concrete, suppose that R is the feasible reward set obtained from the IRL problem p M , π E q and that p R is obtained with a different IRL problem p x M , p π E q , which we can think to as an empirical version of p M , π E q , with an estimated transition model p p replacing the true model p . Intuitively, to have any learning guarantee, 'similar' IRL problems ( p « p p and π E « p π E ) should lead to 'similar' feasible reward sets ( R « R ). 4

<!-- formula-not-decoded -->

p To formally define a Lipschitz framework, we need to select a (pre)metric for evaluating dissimilarities between feasible reward sets and IRL problems. While we defer the presentation of the (pre)metric for the IRL problems to Section 3.1, where it will emerge naturally, for the feasible reward sets, we employ the Hausdorff (pre)metric H d p R , p R q (Equation 1), induced by a (pre)metric d p r, p r q used to evaluate the dissimilarity between individual reward functions r P R and p r P p R . With this choice, two feasible reward sets are 'similar' if every reward r P R is 'similar' to some reward p r P p R in terms of the (pre)metric d . In the next sections, we employ as d the metric induced by the L 8 -norm between the reward functions r P R and r P R : 5

where G stands for 'generative'. In Section 3.1, we prove that the Lipschitz continuity is fulfilled when no restrictions on the reward function are enforced (besides boundedness in r´ 1 , 1 s ). Then, in Section 3.2, we show that, when further restrictions on the viable rewards are required (e.g., state-only reward), such a regularity property no longer holds.

## 3.1. Lipschitz Continuous Feasible Reward Sets

In order to prove the Lipschitz continuity property, we use the explicit form of the feasible reward sets introduced in (Metelli et al., 2021) and extended by (Lindner et al., 2022) for the finite-horizon case, that we report below.

Lemma 3.1 (Lemma 4 of Lindner et al. (2022)) . A reward function r ' p r h q h P /llbracket H /rrbracket is feasible for the IRL problem p M , π E q if and only if there exist two functions p A h , V h q h P /llbracket H /rrbracket where for every h P /llbracket H /rrbracket we have A h : S ˆ A Ñ R ě 0 , V h : S ˆ A Ñ R , and V H ` 1 ' 0 , such that for every p s, a, h q P S ˆ A ˆ /llbracket H /rrbracket it holds that:

<!-- formula-not-decoded -->

Furthermore, if | r h p s, a q| ď 1 , if follows that | V h p s q| ď H ´ h ` 1 and A h p s, a q ď H ´ h ` 1 .

A form of regularity of the feasible reward set was already studied in Theorem of 3.1 of Metelli et al. (2021) and in Theorem 5 of Lindner et al. (2022), providing an error propagation analysis. These results are based on showing the existence of a particular reward r feasible for the IRL

r 4 If not, any arbitrary accurate estimate p p p, p π E q of p p, π E q , may induce feasible sets p R and R with finite non-zero dissimilarity. 5 We discuss other choices of d in Section 4.

problem p x M , p π E q , whose distance from the original reward function r P R is bounded by a dissimilarity term between p M , π E q and p x M , p π E q . Unfortunately, such a reward r r is not guaranteed to be bounded in r´ 1 , 1 s even when the original reward r is (and, thus, it might be r r R p R according to Equation 2). 6 In Lemma B.1, with a modified construction, we show the existence of another particular feasible reward p r bounded in r´ 1 , 1 s (and, thus, p r P p R ). From this, the Lipschitz continuity of the feasible reward sets follows.

<!-- formula-not-decoded -->

Theorem 3.2 (Lipschitz Continuity) . Let R and p R be the feasible reward sets of the IRL problems p M , π E q and p x M , p π E q . Then, it holds that: 7

x p where ρ G p¨ , ¨q is a (pre)metric between IRL problems, defined as:

<!-- formula-not-decoded -->

Some observations are in order. First, the function ρ G is indeed a (pre)metric since it is non-negative and takes value 0 when the IRL problems coincide. Second, as supported by intuition, ρ G is composed of two terms related to the estimation of the expert's policy and of the transition model. While for the transition model, the dissimilarity is formalized by the L 1 -norm distance } p h p¨| s, a q ´ p p h p¨| s, a q} 1 , for the policy, the resulting term deserves some comments. Indeed, the dissimilarity | 1 t π E h p a | s q' 0 u ´ 1 t p π E h p a | s q' 0 u | highlights that what matters is whether an action a P A is played by the expert and not the corresponding probability π E h p a | s q . Indeed, the expert's policy plays an action (with any non-zero probability) only if it is an optimal action.

## 3.2. Non-Lipschitz Continuous Feasible Reward Sets

In this section, we illustrate three cases of feasible reward sets restrictions that turn out not to fulfill the condition of Theorem 3.2. These examples consider three conditions commonly enforced in the literature: state-only reward function r h p s q (Example 3.1), time-homogeneous reward function r p s, a q (Example 3.2), and β -margin reward function (Example 3.3). We present counter-examples in which in front of /epsilon1 -close transition models, the induced feasible sets are far apart by a constant independent of /epsilon1 . For space reasons, we report the complete derivation in Appendix C.

Example 3.1 (State-only reward r h p s q ) . State-only reward functions have been widely considered in many IRL ap- proaches (e.g., Ng &amp; Russell, 2000; Abbeel &amp; Ng, 2004; Syed &amp; Schapire, 2007; Komanduru &amp; Honorio, 2019). We formalize the state-only feasible reward set as follows:

6 We illustrate in Fact B.1 an example of this phenomenon.

7 This implies the standard Lipschitz continuity, by simply bounding 2 ρ G pp M ,π E q , p x M , p π E qq 1 ` ρ G pp M ,π E q , p x M , p π E qq ď 2 ρ G pp M , π E q , p x M , p π E qq .

Figure 1. The MDP \ R employed in the examples of Section 3.2. denotes a transition executed for multiple actions.

<!-- image -->

<!-- formula-not-decoded -->

p Example 3.2 (Time-homogeneous reward r p s, a q ) . Timehomogeneous reward functions have been employed in several RL (e.g., Dann &amp; Brunskill, 2015) and IRL settings (e.g., Lindner et al., 2022). We formalize the timehomogeneous feasible reward set as follows:

R state ' R Xt@p s, a, a , h q : r h p s, a q ' r h p s, a qu . Consider the MDP \ R of Figure 1a with H ' 2 , π E h p s 0 q' p π E h p s 0 q' a 1 with h Pt 1 , 2 u . Set p 1 p s ` | s 0 , a 1 q' 1 { 2 ` /epsilon1 { 4 and p p 1 p s ` | s 0 , a 1 q' 1 { 2 ´ /epsilon1 { 4 and, thus, } p 1 p¨| s 0 , a 1 q´ p p 1 p¨| s 0 , a 1 q} 1 ' /epsilon1 . Let us set r 2 p s ` q' 1 and r 2 p s ´ q'´ 1 , which makes π E optimal under p . We observe that p R is defined by p r 2 p s ´ qď p r 2 p s ` q . Recalling that the rewards are bounded in r´ 1 , 1 s , we have H d G p R state , R state qě 1 .

<!-- formula-not-decoded -->

p Example 3.3 ( β -margin reward) . A β -margin reward enforces a suboptimality gap of at least β ą 0 (Ng &amp; Russell, 2000; Komanduru &amp; Honorio, 2019). We formalize it in the finite-horizon case with a sequence β ' p β h q h P /llbracket H /rrbracket , possibly different for every stage:

Consider the MDP \ R of Figure 1b with H ' 2 , π E 1 p s 0 q' p π E 1 p s 0 q' a 1 and π E 2 p s 0 q' p π E 2 p s 0 q' a 2 . For h Pt 1 , 2 u , we set p h p s 0 | s 0 , a 1 q' 1 { 2 ` /epsilon1 { 4 and p p h p s 0 | s 0 , a 1 q' 1 { 2 ´ /epsilon1 { 4 , thus, } p h p¨| s 0 , a 1 q´ p p h p¨| s 0 , a 1 q} 1 ' /epsilon1 . We set r p s 0 , a 1 q' 1 , r p s 0 , a 2 q' 1 ´ /epsilon1 { 6 , and r p s 1 , a 1 q' r p s 1 , a 2 q' 1 { 2 making π E optimal. We can prove that H d G p R hom , R hom qě 1 { 4 .

<!-- formula-not-decoded -->

Consider the MDP \ R in Figure 1a with π E h p s 0 q ' p π E h p s 0 q ' a 1 for h P t 1 , 2 u . Weset p 1 p s ` | s 0 , a 1 q ' 1 { 2 ` /epsilon1 and p p 1 p s ` | s 0 , a 1 q ' 1 { 2 ´ /epsilon1 . We set for MDP \ R M the reward function as r 1 p s 0 , a q ' 0 and r h p s ` , a q ' ´ r h p s ´ , a q ' 1 for a P t a 1 , a 2 u and h P /llbracket 2 , H /rrbracket . In p s 0 , 1 q the suboptimality gap is β 1 ' 2 ` 2 /epsilon1 p H ´ 1 q . By selecting H ě 1 ` 1 { /epsilon1 , the feasible set p R β -mar is empty.

These examples show that, under certain classes of restrictions, the feasible reward set is not Lipschitz continuous w.r.t. the transition model and, more in general, w.r.t. the IRL problem. The generalization of these examples to more abstract conditions for guaranteeing the Lipschitz continuity of the feasible reward set is beyond the scope of the paper.

## 4. PAC Framework for IRL with a Generative Model

In this section, we discuss the PAC (Probably Approximately Correct) requirements for estimating the feasible reward set with access to a generative model of the environment. We first provide the notion of a learning algorithm estimating the feasible reward set with a generative model (Section 4.1). Then, we formally present the PAC requirement for the Hausdorff (pre)metric H d (Section 4.2). Finally, we discuss the relationships between the PAC requirements with different choices of (pre)metric d (Section 4.3).

## 4.1. Learning Algorithms with a Generative Model

A learning algorithm for estimating the feasible reward set is a pair A ' p µ, τ q , where µ ' p µ t q t P N is a sampling strategy defined for every time step t P N as µ t P ∆ S ˆ A ˆ /llbracket H /rrbracket D t ´ 1 with D t ' p S ˆ A ˆ /llbracket H /rrbracket ˆ S ˆ A q t and τ is a stopping time w.r.t. a suitably defined filtration. At every step t P N , the learning algorithm query the environment in a triple p s t , a t , h t q , selected based on the sampling strategy µ t p¨| D t ´ 1 q , where D t ´ 1 ' pp s l , a l , h l , s 1 l , a E l qq t ´ 1 l ' 1 P D t ´ 1 is the dataset of past samples. Then, the algorithm observes the next state s 1 t ' p h t p¨| s t , a t q and expert's action a E t ' π E h t p¨| s t q and updates the dataset D t ' D t ´ 1 ' p s t , a t , h t , s 1 t , a E t q . Based on the collected data D τ , the algorithm computes the empirical IRL problem p x M τ , p π E,τ q , based on Equation (3) and the empirical feasible reward set p R τ . 4.2. PAC Requirement

We now introduce a general notion of a PAC requirement for estimating the feasible reward set of an IRL problem. To this end, we consider the Hausdorff (pre)metric introduced in Section 3 defined in terms of the reward (pre)metric d p r, p r q . We denote with d -IRL the problem of estimating the feasible reward set under the Hausdorff (pre)metric H d .

Definition 4.1 (PAC Algorithm for d -IRL ) . Let /epsilon1 P p 0 , 2 q and δ P p 0 , 1 q . An algorithm A ' p µ, τ q is p /epsilon1, δ q -PAC for d -IRL if:

where P p M ,π E q , A denotes the probability measure induced by executing the algorithm A in the IRL problem p M , π E q and p R τ is the feasible reward set induced by the empirical IRL problem p x M τ , p π E,τ q estimated with the dataset D τ . The sample complexity is defined as τ : ' | D τ | .

<!-- formula-not-decoded -->

In the next section, we show the relationship between PAC requirements defined for notable choices of d .

## 4.3. Different Choices of d

So far, we have evaluated the dissimilarity between the feasible reward sets by means of the Hausdorff induced by d G , i.e., the L 8 -norm of between individual reward functions. In the literature, other (pre)metrics d have been proposed (e.g., Metelli et al., 2021; Lindner et al., 2022).

d G Q ˚ -IRL Since the recovered reward functions are often used for performing forward RL, an index of interest is the dissimilarity between optimal Q-functions obtained with the reward r P R and r P R in the original MDP \ R:

d G V ˚ -IRL We are often interested in not just being accurate in estimating the optimal Q-function, but rather in the performance of an optimal policy p π ˚ , learned with the recovered reward p r P p R , evaluated under the true reward r P R :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

p p where Π ˚ p p r q : 't π : @p s,a,h qP S ˆ A ˆ /llbracket H /rrbracket : A π h p s,a ; p r qď 0 u is the set of optimal policies under the recovered reward r .

p The following result formalizes the relationships between the presented d -IRL problems.

Theorem 4.1 (Relationships between d -IRL problems) . Let us introduce the graphical convention for c ą 0 :

<!-- formula-not-decoded -->

meaning that any p /epsilon1, δ q -PAC x -IRL algorithm is p c/epsilon1, δ q -PAC y -IRL. Then, the following statements hold:

.

Theorem 4.1 shows that any p /epsilon1, δ q -PAC guarantee on d G , implies p /epsilon1 1 , δ q -PAC guarantees on both d G Q ˚ and d G V ˚ , where /epsilon1 1 ' Θ p H/epsilon1 q is linear in the horizon H . This justifies why focusing on d G -IRL, as in the following section where sample complexity lower bounds are derived. The lower bound analysis for d G Q ˚ -IRL and d G V ˚ -IRL is left to future works.

<!-- image -->

## 5. Lower Bounds

In this section, we establish sample complexity lower bounds for the d G -IRL problem based on the PAC requirement of Definition 4.1 in the generative model setting. We start presenting the general result (Section 5.1) and, then, we comment on its form and, subsequently, provide a sketch of the construction of the hard instances for obtaining the lower bound (Section 5.2). For the sake of presentation, we assume that the expert's policy π E is known; the extension to the case of unknown π E is reported in Appendix D.

## 5.1. Main Result

In this section, we report the main result of the lower bound of the sample complexity of learning the feasible reward set.

Theorem 5.1 (Lower Bound for d G -IRL) . Let A ' p µ, τ q be an p /epsilon1, δ q -PAC algorithm for d G -IRL. Then, there exists an IRL problem p M , π E q such that, if δ ď 1 { 32 , S ě 9 , A ě 2 , and H ě 12 , the expected sample complexity is lower bounded by:

- if the transition model p is time-inhomogeneous:
- if the transition model p is time-homogeneous:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where E p M ,π E q , A denotes the expectation w.r.t. the probability measure P p M ,π E q , A .

Some observations are in order. First, the derived lower bound displays a linear dependence on the number of actions A and dependence on the horizon H raised to a power 2 or 3 , which depends on whether the underlying transition model is time-homogeneous, as common even for forward RL(e.g., Dann &amp; Brunskill, 2015; Domingues et al., 2021). Second, we identify two different regimes visible inside the parenthesis related to the dependence on the number of states S and the confidence δ . Specifically, for small values of δ (i.e., δ « 0 ), the dominating part is log ` 1 δ ˘ , leading to a sample complexity of order Ω ´ H 3 SA /epsilon1 2 log ` 1 δ ˘ ¯ . Instead, for large δ (i.e., δ « 1 { 32 ), the most relevant part is the one corresponding to S , leading to sample complexity of order Ω ´ H 3 S 2 A /epsilon1 2 ¯ (both for the time-inhomogeneous case). An analogous two-regime behavior has been previously observed in the reward-free exploration setting (Jin et al., 2020; Kaufmann et al., 2021; M´ enard et al., 2021).

## 5.2. Sketch of the Proof

In this section, we provide a sketch of the construction of the lower bounds of Theorem 5.1. The idea consists in deriving two separate bounds depending on the regime of δ , which are based on two building blocks reported in Figure 2. These instances are used to build lower bounds for a single state s ˚ and the extension to multiple states and stages follows standard constructions (e.g., Domingues et al., 2021).

Smallδ regime Figure 2a reports the instances employed in this regime. The expert's policy is π E p s q ' a 0 . From state s ˚ , all actions bring the system to the absorbing states s ` and s ´ with equal probability, except for action a ˚ ‰ a 0 that increases by /epsilon1 1 ą 0 the probability of reaching state s ` . The learner, in order to recover a correct feasible reward set, has to identify which is the action behaving like a ˚ (among the A available ones) to force action a 0 to be optimal. Considering Θ p A q instances, in which action a ˚ changes, an application of BretagnolleHuber inequality (Lattimore &amp; Szepesv´ ari, 2020, Theorem 14.2) allows deriving a sample complexity lower bounded by Ω ´ AH 2 /epsilon1 2 log ` 1 δ ˘ ¯ .

Extension to Multiple States and Stages At the beginning, the system randomly chooses a problem between Fig-

Largeδ regime Figure 2b depicts the instances used in this regime. The expert's policy is again π E p s q ' a 0 . The system, instead, is made of S ' Θ p S q next states reachable with equal probability by playing action a 0 . All other actions a j ‰ a 0 alter the probability distribution of the next state. Specifically, by playing the action a j ‰ a 0 , the probability of reaching the next state s 1 k is given by p 1 ` /epsilon1 1 v p j q k q{ S , where v p j q P t´ 1 , 1 u S is a vector such that ř S k ' 1 v p j q k ' 0 . By varying v j in a suitable set, defined by means of a packing argument, we obtain Θ p 2 S q instances each one separated by a finite dissimilarity, depending on /epsilon1 1 . We obtain the lower bound by means of an application of the Fano's inequality (Gerchinovitz et al., 2017, Proposition 4) which results in order Ω ´ pp 1 ´ δ q´ log 2 q S 2 AH 2 /epsilon1 2 ¯ .

<!-- image -->

Figure 2. The MDP \ R employed in the constructions of the lower bounds of Section 5. The expert's policy is π E p s q ' a 0 . denotes a transition executed for multiple actions.

```
Input: significance δ P p 0 , 1 q , /epsilon1 target accuracy t Ð 0 , /epsilon1 0 Ð`8 while /epsilon1 t ą /epsilon1 do t Ð t ` SAH Collect one sample from each p s, a, h q P S ˆ A ˆ /llbracket H /rrbracket Update p p t according with (3) Update /epsilon1 t ' max p s,a,h qP S ˆ A ˆ /llbracket H /rrbracket C t h p s, a q (resp. r C t h p s, a q ) end while
```

Algorithm 1. UniformSampling-IRL ( US-IRL ) for timeinhomogeneous (resp. time-homogeneous) transition models.

ure 2a and Figure 2b. Then, it transitions to the state in which the system may randomly remain for H ă H stages after which it transitions with uniform probability to any of the Θ p S q states. H ' Θ p H q for the time-inhomogeneous (resp. H ' O p 1 q for the time-homogeneous) case. In any state s ˚ and stage h ˚ , the agent can face the problems shown in Figure 2. By varying s ˚ and h ˚ among its possible HS (resp. S ) values, we get the bounds in Theorem 5.1.

Remark 5.1 (Generative vs Forward models) . This construction suffices for obtaining a bound for the generative model, but it can be easily extended to work with the forward model of the environment (in which the agent interacts via trajectories only) by means of a standard tree-based construction (Jin et al., 2020; Domingues et al., 2021). In such a case, the resulting PAC guarantee would no longer be expressed via the L 8 -norm distance d G between reward, but worst-case over the visitation distributions induced by the policies: d F p r, p r q : ' sup π E M ,π r| r h p s, a q ´ p r h p s, a q|s .

## 6. Algorithm

In this section, we analyze the sample complexity of a uniform sampling strategy (UniformSampling-IRL, US-IRL ) for the d G -IRL problem (Algorithm 1). We start presenting the sample complexity analysis (Section 6.1) and, then, we provide a sketch of the proof (Section 6.2).

## 6.1. Main Result

The US-IRL algorithm was presented in (Metelli et al., 2021; Lindner et al., 2022) but analyzed for different IRL formulations (see Section 7). We revise it since it matches our sample complexity lower bounds, provided that more sophisticated concentration tools w.r.t. those employed in (Metelli et al., 2021; Lindner et al., 2022). For the sake of presentation, we assume that the expert's policy π E is known; the extension to unknown π E is reported in Appendix D. At each iteration, the algorithm collects a sample from every p s, a, h q P S ˆ A ˆ /llbracket H /rrbracket and, for timeinhomogeneous models, computes the confidence function:

where β ` n, δ ˘ : ' log p SAH { δ q`p S ´ 1 q log ` e p 1 ` n {p S ´ 1 q ˘ . 8 The algorithm stops as soon as all confidence functions fall below the threshold /epsilon1 . The following theorem provides the sample complexity of US-IRL .

<!-- formula-not-decoded -->

Theorem 6.1 (Sample Complexity of US-IRL ) . Let /epsilon1 ą 0 and δ P p 0 , 1 q , US-IRL is p /epsilon1, δ q -PAC for d G -IRL and with probability at least 1 ´ δ it stops after τ samples with:

- if the transition model p is time-inhomogeneous:

<!-- formula-not-decoded -->

8 In the time-homogeneous case, the algorithm merges the samples collected at different h P /llbracket H /rrbracket for the estimation of the transition model and replaces the confidence function with:

where r β ` n, δ ˘ : ' log p SA { δ q ` p S ´ 1 q log ` e p 1 ` n {p S ´ 1 q ˘ and n t p s, a q ' ř H h ' 1 n t h p s, a q .

<!-- formula-not-decoded -->

- if the transition model p is time-homogeneous and :

<!-- formula-not-decoded -->

Thus, time-inhomogeneous (resp. time-homogeneous) transition models, US-IRL suffers a sample complexity bound of order r O ´ H 3 SA /epsilon1 2 ` log ` 1 δ ˘ ` S ˘ ¯ (resp. r O ´ H 2 SA /epsilon1 2 ` log ` 1 δ ˘ ` S ˘ ¯ ) matching the lower bounds of Theorem 5.1 up to logarithmic factors for both regimes of δ .

## 6.2. Sketch of the Proof

The idea of the proof is to exploit Theorem 3.2 to reduce the Hausdorff distance to the L 1 -norm between the transition model } p p t h p¨| s, a q ´ p h p¨| s, a q} 1 . It is worth noting this term replaces |p p p t h ´ p h q V h | appearing in previous works (Metelli et al., 2021; Lindner et al., 2022) that was comfortably bounded using H¨ oeffding's inequality. In our case, the L 1 -norm is unavoidable due to the Hausdorff distance that implies a worst-case choice of the reward function and, thus, of V h . This term has to be carefully bounded using the stronger KL-divergence concentration result of (Jonsson et al., 2020, Proposition 1) to get the O p log p 1 { δ q ` S q rate. 9

## 7. Related Works

In this section, we discuss the related works about sample complexity analysis and lower bounds for IRL. Additional related works are reported in Appendix A.

Sample Complexity for Estimating the Feasible Reward Set The notion of feasible reward set R was introduced in (Ng &amp; Russell, 2000) in an implicit form in the infinite-horizon discounted case as a linear feasibility problem and, subsequently, adapted to the finite-horizon case in (Lindner et al., 2022). Furthermore, in (Metelli et al., 2021; Lindner et al., 2022) an explicit form of the reward functions belonging to the feasible region R was provided. In these works, the problem of estimating the feasible reward set is studied for the first time considering a 'reference' pair of rewards p r, q r q P R ˆ p R against which to compare the rewards inside the recovered sets, leading to the (pre)metric:

p p 9 A more na¨ ıve application of the L 1 -concentration of (Weissman et al., 2003) would lead to the worse O p S log p 1 { δ qq rate.

<!-- formula-not-decoded -->

Compared to the Hausdorff (pre)metric (Equation 1), in Equation (8) there is no maximization over the choice of p r, q r q , leading to a simpler problem. 10 In (Metelli et al., 2021), a uniform sampling approach (similar to Algorithm 1) is proved to achieve a sample complexity of order r O ´ γ 2 SA p 1 ´ γ q 4 /epsilon1 2 ¯ for the index of Equation (8) with d ' d G Q ˚ in the discounted setting with generative model. For the forward model case, the AceIRL algorithm (Lindner et al., 2022) suffers a sample complexity of order r O ´ H 5 SA /epsilon1 2 ¯ for the index of Equation (8) with d ' d F V ˚ , in the finitehorizon case. 11 Unfortunately, the reward recovered by AceIRL reward function is not guaranteed to be bounded by a predetermined constant (e.g., r´ 1 , 1 s ). Modified versions of these algorithms allow embedding problemdependent features under a specific choice of a reward within the set.

Sample Complexity Lower Bounds in IRL To the best of our knowledge, the only work that proposes a sample complexity lower bound for IRL is (Komanduru &amp; Honorio, 2021). The authors consider a finite state and action MDP \ Rand the IRL algorithm of (Ng &amp; Russell, 2000) for β -strict separable IRL problems (i.e., with suboptimality gap at least β ) with state-only rewards in the discounted setting. When only two actions are available ( A ' 2 ) and the samples are collected starting in each state with equal probability, by means of a geometric construction and Fano's inequality, the authors derive an Ω p S log S q lower bound on the number of trajectories needed to identify a reward function. Note that this analysis limits to the identification of a reward function within a finite set, rather than evaluating the accuracy of recovering the feasible reward set.

## 8. Conclusions and Open Questions

In this paper, we provided contributions to the understanding of the complexity of the IRL problem. We conceived a lower bound of order Ω ´ H 3 SA /epsilon1 2 ` log ` 1 δ ˘ ` S ˘ ¯ on the number samples collected with a generative model in the finite-horizon setting. This result is of relevant interest since it sets, for the first time, the complexity of the IRL problem, defined as the problem of estimating the feasible reward set. Furthermore, we showed that a uniform sampling strategy matches the lower bound up to logarithmic factors. Nevertheless, the IRL problem is far from being closed. In the following, we outline a road map of open questions, hoping to inspire researchers to work in this appealing area.

10 In this sense, a PAC guarantee according to Definition 4.1, implies a PAC guarantee defined w.r.t. (pre)metric of Equation (8).

11 As discussed in Remark 5.1, in the forward model case, the dissimilarity is in expectation w.r.t. the worst-case policy.

Forward Model The most straightforward extension of our findings is moving to the forward model setting, in which the agent can interact with the environment through trajectories only. As we already noted, our lower bounds can be comfortably extended to this setting. However, in this case, the PAC requirement has to be relaxed since controlling the L 8 -norm between rewards is no longer a viable option (e.g., for the possible presence of almost unreachable states). Which distance notion should be used for this setting? Will the Lipschitz regularity of Section 3 still hold?

Problem-Dependent Analysis Our analysis is worst-case in the class of IRL problems. Would it be possible to obtain a problem-dependent complexity results? Previous problem-dependent analyses provided results tightly connected to the properties of the specific reward selection procedure (Metelli et al., 2021; Lindner et al., 2022). Clearly, a currently open question, in all settings in which reward is missing, including reward-free exploration (Jin et al., 2020) and IRL, is how to define a problem-dependent quantity in replacement of the suboptimality gaps.

Reward Selection Our PAC guarantees concern with the complete feasible reward set. However, algorithmic solutions to IRL implement a specific criterion for selecting a reward (e.g., maximum entropy, maximum margin). How the PAC guarantee based on the Hausdorff distance relates to guarantees on a single reward selected with a specific criterion within R ?

## References

- Abbeel, P. and Ng, A. Y. Apprenticeship learning via inverse reinforcement learning. In Proceedings of the Twenty-first International Conference on Machine Learning (ICML) , volume 69 of ACM International Conference Proceeding Series . ACM, 2004.
- Adams, S. C., Cody, T., and Beling, P. A. A survey of inverse reinforcement learning. Artif. Intell. Rev. , 55(6): 4307-4346, 2022.
- Arora, S. and Doshi, P. A survey of inverse reinforcement learning: Challenges, methods and progress. Artif. Intell. , 297:103500, 2021.
- Cohen, G. D. and Frankl, P. Good coverings of hamming spaces with spheres. Discret. Math. , 56(2-3):125-131, 1985.
- Dann, C. and Brunskill, E. Sample complexity of episodic fixed-horizon reinforcement learning, 2015.
- Dexter, G., Bello, K., and Honorio, J. Inverse reinforcement learning in a continuous state space with formal
- guarantees. In Advances in Neural Information Processing Systems 34 (NeurIPS) , pp. 6972-6982, 2021.

Domingues, O. D., M´ enard, P., Kaufmann, E., and Valko, M. Episodic reinforcement learning in finite mdps: Minimax lower bounds revisited. In Algorithmic Learning Theory (ALT) , volume 132 of Proceedings of Machine Learning Research , pp. 578-598. PMLR, 2021.

- Gerchinovitz, S., M´ enard, P., and Stoltz, G. Fano's inequality for random variables. CoRR , abs/1702.05985, 2017.
- Gy¨ orfi, L., Kohler, M., Krzyzak, A., and Walk, H. A Distribution-Free Theory of Nonparametric Regression . Springer series in statistics. Springer, 2002.

Jin, C., Krishnamurthy, A., Simchowitz, M., and Yu, T. Reward-free exploration for reinforcement learning. In Proceedings of the 37th International Conference on Machine Learning (ICML) , volume 119 of Proceedings of Machine Learning Research , pp. 4870-4879. PMLR, 2020.

- Jonsson, A., Kaufmann, E., M´ enard, P., Domingues, O. D., Leurent, E., and Valko, M. Planning in markov decision processes with gap-dependent sample complexity. In Advances in Neural Information Processing Systems 33 (NeurIPS) , 2020.
- Kaufmann, E., M´ enard, P., Domingues, O. D., Jonsson, A., Leurent, E., and Valko, M. Adaptive reward-free exploration. In Algorithmic Learning Theory (ALT) , volume 132 of Proceedings of Machine Learning Research , pp. 865-891. PMLR, 2021.
- Komanduru, A. and Honorio, J. On the correctness and sample complexity of inverse reinforcement learning. pp. 7110-7119, 2019.
- Komanduru, A. and Honorio, J. A lower bound for the sample complexity of inverse reinforcement learning. In Proceedings of the 38th International Conference on Machine Learning (ICML) , volume 139 of Proceedings of Machine Learning Research , pp. 5676-5685. PMLR, 2021.
- Lattimore, T. and Szepesv´ ari, C. Bandit algorithms . Cambridge University Press, 2020.
- Lindner, D., Krause, A., and Ramponi, G. Active exploration for inverse reinforcement learning. CoRR , abs/2207.08645, 2022.
- M´ enard, P., Domingues, O. D., Jonsson, A., Kaufmann, E., Leurent, E., and Valko, M. Fast active learning for pure exploration in reinforcement learning. In Proceedings of the 38th International Conference on Machine Learning (ICML) , volume 139 of Proceedings of Machine Learning Research , pp. 7599-7608. PMLR, 2021.

- Metelli, A. M., Pirotta, M., and Restelli, M. Compatible reward inverse reinforcement learning. In Advances in Neural Information Processing Systems 30 (NeurIPS) , pp. 2050-2059, 2017.
- Metelli, A. M., Ramponi, G., Concetti, A., and Restelli, M. Provably efficient learning of transferable rewards. In Proceedings of the 38th International Conference on Machine Learning (ICML) , volume 139 of Proceedings of Machine Learning Research , pp. 7665-7676. PMLR, 2021.
- Ng, A. Y. and Russell, S. Algorithms for inverse reinforcement learning. In Proceedings of the Seventeenth International Conference on Machine Learning (ICML) , pp. 663-670. Morgan Kaufmann, 2000.
- Osa, T., Pajarinen, J., Neumann, G., Bagnell, J. A., Abbeel, P., and Peters, J. An algorithmic perspective on imitation learning. Found. Trends Robotics , 7(1-2):1-179, 2018.
- Pirotta, M. and Restelli, M. Inverse reinforcement learning through policy gradient minimization. In Proceedings of the Thirtieth Conference on Artificial Intelligence (AAAI) , pp. 1993-1999. AAAI Press, 2016.
- Puterman, M. L. Markov Decision Processes: Discrete Stochastic Dynamic Programming . Wiley Series in Probability and Statistics. Wiley, 1994.
- Ramponi, G., Likmeta, A., Metelli, A. M., Tirinzoni, A., and Restelli, M. Truly batch model-free inverse reinforcement learning about multiple intentions. In The 23rd International Conference on Artificial Intelligence and Statistics (AISTATS) , volume 108 of Proceedings of Machine Learning Research , pp. 2359-2369. PMLR, 2020.
- Ratliff, N. D., Bagnell, J. A., and Zinkevich, M. Maximum margin planning. In Proceedings of the Twenty-Third International Conference on Machine Learning (ICML) , volume 148 of ACM International Conference Proceeding Series , pp. 729-736. ACM, 2006.
- Rockafellar, R. T. and Wets, R. J. Variational Analysis , volume 317 of Grundlehren der mathematischen Wissenschaften . Springer, 1998.
- Sutton, R. S. and Barto, A. G. Reinforcement learning - an introduction . Adaptive computation and machine learning. MIT Press, 1998.
- Syed, U. and Schapire, R. E. A game-theoretic approach to apprenticeship learning. pp. 1449-1456, 2007.
- Vroman, M. C. Maximum likelihood inverse reinforcement learning . Rutgers The State University of New JerseyNew Brunswick, 2014.
- Weissman, T., Ordentlich, E., Seroussi, G., Verdu, S., and Weinberger, M. J. Inequalities for the l1 deviation of the empirical distribution. Hewlett-Packard Labs, Tech. Rep , 2003.
- Wu, X., Xiao, L., Sun, Y., Zhang, J., Ma, T., and He, L. A survey of human-in-the-loop for machine learning. Future Gener. Comput. Syst. , 135:364-381, 2022.
- Zeng, S., Li, C., Garcia, A., and Hong, M. Maximumlikelihood inverse reinforcement learning with finitetime guarantees. In Advances in Neural Information Processing Systems (NeurIPS) , 2022.
- Ziebart, B. D., Maas, A. L., Bagnell, J. A., and Dey, A. K. Maximum entropy inverse reinforcement learning. In Proceedings of the Twenty-Third Conference on Artificial Intelligence (AAAI) , pp. 1433-1438. AAAI Press, 2008.

## A. Additional Related Works

In this appendix, we report additional related works concerning sample complexity analysis for specific IRL algorithms and reward-free exploration.

Sample Complexity of IRL Algorithms Differently from forward RL, the theoretical understanding of the IRL problem is largely less established and the sample complexity analysis proposed in the literature often limit to specific algorithms. In the class of feature expectation approaches, the seminal work (Abbeel &amp; Ng, 2004) propose IRL algorithms guaranteed to output an /epsilon1 -optimal policy (made of a mixture of Markov policies) after r O ´ k /epsilon1 2 p 1 ´ γ q 2 log ` 1 δ ˘ ¯ trajectories (ideally of infinite length). The result holds in a discounted setting (being γ the discount factor) under the assumption that the true reward function r p s q ' w T φ p s q is state-only and linear in some known features φ of dimensionality k . In (Syed &amp; Schapire, 2007), a game-theoretic approach to IRL, named MWAL , is proposed improving (Abbeel &amp; Ng, 2004) in terms of computational complexity and allowing the absence of an expert, preserving similar theoretical guarantees in the same setting. Modular IRL (Vroman, 2014), that integrates supervised learning capabilities in the IRL algorithm, is guaranteed to produce an /epsilon1 -optimal policy after r O ´ SA p 1 ´ γ q 2 /epsilon1 2 log ` 1 δ ˘ ¯ trajectories. This class of algorithms, however, requires, as an inner step, to compute the optimal policy p π for every candidate reward function p r . This step (and the corresponding sample complexity) is somehow hidden in the analysis since they either assume the knowledge of the transition model and apply dynamic programming (e.g., Vroman, 2014) or the access to a black-box RL algorithm (e.g., Abbeel &amp; Ng, 2004). In the class of maximum entropy approaches (Ziebart et al., 2008), the Maximum Likelihood IRL (Zeng et al., 2022) converges to a stationary solution with r O p /epsilon1 ´ 2 q trajectories for non-linear reward parametrization (with bounded gradient and Lipschitz smooth), when the underlying Markov chain is ergodic. Furthermore, the authors prove that, when the reward is linear in some features, the recovered solution corresponds to Maximum Entropy IRL (Ziebart et al., 2008). Concerning the gradient-based approaches, (Pirotta &amp; Restelli, 2016) and (Ramponi et al., 2020) prove finite-sample convergence guarantee to the expert's weight under linear parametrization as a function of the accuracy of the gradient estimation. Surprisingly, a theoretical analysis of the IRL progenitor algorithm of (Ng &amp; Russell, 2000) has been proposed only recently in (Komanduru &amp; Honorio, 2019). A β -strict separability setting is enforced in which the rewards are assumed to lead to a suboptimality gap of at least β ą 0 when playing any non-optimal action. For finite MDPs, known expert's policy, under the demanding assumption that each state is reachable in one step with a minimum probability α ą 0 , and focusing on state-only reward, the authors prove that the algorithm outputs a β -strict separable feasible reward in at most r O ´ 1 ` γ 2 Ξ 2 αβ 2 p 1 ´ γ q 4 log ` 1 δ ˘ ¯ trajectories, where Ξ ď S is the number of possible successor states. Recently, an approach with theoretical guarantees has been proposed for continuous states (Dexter et al., 2021).

Reward-Free Exploration Reward-free exploration (RFE, Jin et al., 2020; Kaufmann et al., 2021; M´ enard et al., 2021) is a setting for pure exploration in MDPs composed of two phases: exploration and planning. In the exploration phase, the agent learns an estimated transition model p p without any reward feedback. In the planning phase, the agent is faced with a reward function r and has to output an estimated optimal policy p π ˚ , using p p since no further interaction with the environment is admitted. In this sense, RFE shares this two-phase procedure with our IRL problem, but, instead of the planning phase, we face the computation of the feasible reward set. 12 In RFE exploration, the sample complexity is computed against the performance of the learned policy p π ˚ under the reward r , i.e., V ˚ p¨ ; r q ´ V p π ˚ p¨ ; r q , whose lower bound of the sample complexity has order Ω ´ H 2 SA /epsilon1 2 ` H log ` 1 δ ˘ ` S ˘ ¯ (Jin et al., 2020; Kaufmann et al., 2021). The best known algorithm, RF-Express , proposed in (M´ enard et al., 2021) archives an almost-matching sample complexity of order Ω ´ H 3 SA /epsilon1 2 ` log ` 1 δ ˘ ` S ˘ ¯ . The relevant connection with what we present in this paper is the fact that the derivation of the lower bounds shares similarity especially in the construction of the instances. Nevertheless, in the time-inhomogeneous case, we achieve a higher lower bound of order Ω ´ H 3 SA /epsilon1 2 ` log ` 1 δ ˘ ` S ˘ ¯ . The connection between IRL and RFE should be investigated in future works, as also mentioned in (Lindner et al., 2022).

## B. Proofs

In this appendix, we report the proofs we omitted in the main paper.

12 As shown in previous works, the computation of the feasible reward set can be formulated with a linear feasibility problem (Ng &amp; Russell, 2000).

## B.1. Proofs of Section 3

Lemma B.1. Let r be feasible for the IRL problem p M , π E q bounded in r´ 1 , 1 s (i.e., p r P R ) and defined according to Lemma 3.1 as r h p s, a q ' ´ A h p s, a q 1 t π E h p a | s q' 0 u ` V h p s q ´ p h V h ` 1 p s, a q . Let p x M , p π E q be an IRL problem and define for every p s, a, h q P S ˆ A ˆ /llbracket H /rrbracket :

p Then, the reward function p r defined according to Lemma 3.1 as p r h p s, a q ' ´ p A h p s, a q 1 t p π E h p a | s q' 0 u ` p V h p s q´ p h p V h ` 1 p s, a q for every p s, a, h q P S ˆ A ˆ /llbracket H /rrbracket with:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Given the reward function r h p s, a q ' ´ A h p s, a q 1 t π E h p a | s q' 0 u ` V h p s q ´ p h V h ` 1 p s, a q , we define the reward function:

where /epsilon1 : ' max p s,a,h qP S ˆ A ˆ /llbracket H /rrbracket | /epsilon1 h p s, a q| , is feasible for the IRL problem p x M , p π E q and bounded in r´ 1 , 1 s (i.e., p r P p R ).

r r h p s, a q ' ´ A h p s, a q 1 t p π E h p a | s q' 0 u ` V h p s q ´ p p h V h ` 1 p s, a q , that, thanks to Lemma 3.1, makes policy p π E optimal. However, it is not guaranteed that r r P p R since it can take values larger than 1 . Thus, we define the reward:

which simply scales r r h and preserves the optimality of p π E . We now prove that p r h p s, a q is bounded in r´ 1 , 1 s . To do so, we prove that r h p s, a q is bounded in r´p 1 ` /epsilon1 q , p 1 ` /epsilon1 qs :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Theorem 3.2 (Lipschitz Continuity) . Let R and p R be the feasible reward sets of the IRL problems p M , π E q and p x M , p π E q . Then, it holds that: 13

where ρ G p¨ , ¨q is a (pre)metric between IRL problems, defined as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

ˇ ˇ 13 This implies the standard Lipschitz continuity, by simply bounding 2 ρ G pp M ,π E q , p x M , p π E qq 1 ` ρ G pp M ,π E q , p x M , p π E qq ď 2 ρ G pp M , π E q , p x M , p π E qq .

Proof. Let r r as defined in the proof of Lemma B.1. Then, we have:

<!-- formula-not-decoded -->

By recalling that 2 /epsilon1 1 ` /epsilon1 is a non-decreasing function of /epsilon1 , we bound it by replacing /epsilon1 with an upper bound:

<!-- formula-not-decoded -->

p Fact B.1. There exist two MDP \ R M and x M with transition models p and p p respectively, an expert's policy π E and a reward function r h p s, a q ' ´ A h p s, a q 1 t π E p a | s q' 0 u ` V h p s q ´ p h V h ` 1 p s q feasible for the IRL problem p M , π E q bounded in r´ 1 , 1 s (i.e., r P R ) such that the reward function p r h p s, a q ' ´ A h p s, a q 1 t π E p a | s q' 0 u ` V h p s q ´ p p h V h ` 1 p s q is feasible for the IRL problem p x M , π E q not bounded in r´ 1 , 1 s .

x p where we used H¨ older's inequality recalling that | V h ` 1 p s, a q| ď H ´ h and | A h p s, a q| ď H ´ h ` 1 . Clearly, ρ G pp M , π E q , p x M , π E qq is a (pre)metric.

Proof. We consider the MDP \ R in Figure 3 with optimal policy and reward function defined for every h P /llbracket H /rrbracket and H ' 10 as:

<!-- formula-not-decoded -->

Simple calculations lead to the V-function and advantage function values:

<!-- formula-not-decoded -->

We consider as alternative transition model p p ' 1 ´ p . After tedious calculations we obtain the alternative reward function: r h p s 1 , a 1 q ' ´p H ´ h q , r h p s 1 , a 2 q ' ´ 1 ` 8 p H ´ h q{ 10 , r h p s 2 , a 1 q ' 8 p H ´ h q{ 10 , p r h p s 2 , a 2 q ' H ´ h.

<!-- image -->

p p p It is simple to observe that for some p s, a, h q we have | p r h p s, a q| ą 1 .

Figure 3. The MDP \ R employed in Fact B.1.

## B.2. Proofs of Section 4

Theorem 4.1 (Relationships between d -IRL problems) . Let us introduce the graphical convention for c ą 0 :

<!-- formula-not-decoded -->

meaning that any p /epsilon1, δ q -PAC x -IRL algorithm is p c/epsilon1, δ q -PAC y -IRL. Then, the following statements hold:

<!-- image -->

Proof. Let A be an p /epsilon1, δ q -PAC d G -IRL algorithm. This means that with probability at least 1 ´ δ , we have that for any IRL problem H d G p R , p R τ q ď /epsilon1 . We introduce the following visitation distributions, defined for every s, s 1 P S , h, l P /llbracket H /rrbracket with l ě h , and a, a 1 P A :

<!-- formula-not-decoded -->

d G -IRL Ñ d G Q ˚ -IRL Let us consider the optimal Q-function difference and let π ˚ an optimal policy under the reward function r , we have:

As a consequence, we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

d G -IRL Ñ d G V ˚ -IRL Let us consider the value functions and let π ˚ (resp. p π ˚ ) be an optimal policy under reward function r (resp. r ), we have:

Thus, it follows that:

d G Q ˚ -IRL Ñ d G V ˚ -IRL To prove this result, we need to introduce further tools. Specifically, we introduce the Bellman expectation operator and the Bellman optimal operator, defined for a reward function r , policy π , p s, h q P S ˆ /llbracket H /rrbracket and function f : S Ñ R :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

We recall the fixed-point properties: T ˚ r,h V ˚ h ' V ˚ h and T π r,h V π h ' V π h . Let π ˚ (resp. p π ˚ ) be an optimal policy under reward r (resp. r ). Let us consider the following derivation:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Thus, we have:

<!-- formula-not-decoded -->

Since the derivation is carried out for arbitrary π ˚ , it follows that:

<!-- formula-not-decoded -->

## B.3. Proofs of Section 5

Theorem 5.1 (Lower Bound for d G -IRL) . Let A ' p µ, τ q be an p /epsilon1, δ q -PAC algorithm for d G -IRL. Then, there exists an IRL problem p M , π E q such that, if δ ď 1 { 32 , S ě 9 , A ě 2 , and H ě 12 , the expected sample complexity is lower bounded by:

- if the transition model p is time-inhomogeneous:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

- if the transition model p is time-homogeneous:

<!-- formula-not-decoded -->

where E p M ,π E q , A denotes the expectation w.r.t. the probability measure P p M ,π E q , A .

Proof. We put together the results of Theorem B.2 and Theorem B.3, by recalling that max t a, b u ě a ` b 2 , or, equivalently, assuming to observe instances like the ones of Theorem B.2 w.p. 1 { 2 as well as those of Theorem B.3.

Theorem B.2. Let A ' p µ, τ q be an p /epsilon1, δ q -PAC algorithm for d G -IRL. Then, there exists an IRL problem p M , π E q such that, if /epsilon1 ď 1 , δ ă 1 { 16 , S ě 9 , A ě 2 , and H ě 12 , the expected sample complexity is lower bounded by:

- if the transition model p is time-inhomogeneous:

<!-- formula-not-decoded -->

- if the transition model p is time-homogeneous:

<!-- formula-not-decoded -->

Proof. Step 1: Instances Construction The construction of the hard MDP \ R instances follows similar steps as the ones presented in the constructions of lower bounds for policy learning (Domingues et al., 2021) and the hard instances are reported in Figure 4 in a semi-formal way. The state space is given by S ' t s start , s root , s ´ , s ` , s 1 , . . . , s S u and the action space is given by A ' t a 0 , a 1 , . . . , a A u . The transition model is described below and the horizon is H ě 3 . We introduce the constant H P /llbracket H /rrbracket , whose value will be chosen later. Let us observe, for now, that if H ' 1 , the transition model is time-homogeneous.

The agent begins in state s start, where every action has the same effect. Specifically, if the stage h ă H , then there is probability 1 { 2 to remain in s start and a probability 1 { 2 to transition to s root . Instead, if h ě H , the state transitions to s root deterministically. From state s root, every action has the same effect and the state transitions with equal probability 1 { S to a state s i with i P /llbracket S /rrbracket . In all states s i , apart from a specific one, i.e., state s ˚ , all actions have the same effect, i.e., transitioning to states s ´ and s ` with equal probability 1 { 2 . State s ˚ behaves as the other ones if the stage h ‰ h ˚ , where h ˚ P /llbracket H /rrbracket is a predefined stage. If, instead, h ' h ˚ , all actions a j ‰ a ˚ behave like in the other states, while for action a ˚ , we have a 1 { 2 ` /epsilon1 1 probability of reaching s ` (and consequently probability 1 { 2 ´ /epsilon1 1 of reaching s ´ ), with /epsilon1 1 P r 0 , 1 { 4 s . Notice that, having fixed H , the possible values of h ˚ are t 3 , . . . , 2 ` H u . States s ` and s ´ are absorbing states. The expert's policy always plays action a 0 .

Let us consider the base instance M 0 in which there is no state behaving like s ˚ . Additionally, by varying the triple /lscript : ' p s ˚ , a ˚ , h ˚ q P t s 1 , . . . , s S u ˆ t a 1 , . . . , a A u ˆ /llbracket 3 , H ` 2 /rrbracket ' : I , we can construct the class of instances denoted by M ' t M /lscript : /lscript P t 0 u Y I u .

Step 2: Feasible Set Computation Let us consider an instance M /lscript P M , we now seek to provide a lower bound to the Hausdorff distance H d G p R M 0 , R M /lscript q . To this end, we focus on the triple /lscript ' p s ˚ , a ˚ , h ˚ q and we enforce the convenience of action a 0 over action a ˚ . For the base MDP \ R M 0 , let r 0 P R M 0 , we have:

<!-- formula-not-decoded -->

For the alternative MDP \ R M /lscript , let r /lscript P R M /lscript , we have:

<!-- formula-not-decoded -->

Figure 4. Semi-formal representation of the the hard instances MDP \ R used in the proof of Theorem B.2.

<!-- image -->

<!-- formula-not-decoded -->

In order to lower bound the Hausdorff distance H d G p R M 0 , R M /lscript q , we set for M /lscript :

<!-- formula-not-decoded -->

Then, for notational convenience, for the MDP \ R M 0 , we set x : ' r 0 h ˚ p s ˚ , a 0 q and y : ' r 0 h ˚ p s ˚ , a ˚ q :

<!-- formula-not-decoded -->

We enforce the following constraint on this quantity:

<!-- formula-not-decoded -->

Notice that /epsilon1 1 ď 1 { 4 whenever H ě H ` 10 .

Step 3: Lower bounding Probability Let us consider an p /epsilon1, δ q -correct algorithm A that outputs the estimated feasible set p R . Thus, for every ı P I , we can lower bound the error probability:

For every ı P I , let us define the identification function :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let  P t 0 , ı u . If Ψ ı '  , then, H d G p R M Ψ ı , R M  q ' 0 . Otherwise, if Ψ ı ‰  , we have:

where the first inequality follows from triangular inequality and the second one from the definition of identification function Ψ ı . From Equation (9), we have that H d G ` R M Ψ ı , R M  ˘ ě 2 /epsilon1 . Thus, it follows that H d G ´ p R , R M  ¯ ě /epsilon1 . This implies the following inclusion of events for  P t 0 , ı u :

Thus, we can proceed by lower bounding the probability:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality follows from the observation that max t a, b u ě 1 2 p a ` b q and the equality from observing that Ψ ı P t 0 , ı u . The intuition behind this derivation is that we lower bound the probability of making a mistake ě /epsilon1 with the probability of failing in identifying the true underlying problem. We can now apply the Bretagnolle-Huber inequality (Lattimore &amp; Szepesv´ ari, 2020, Theorem 14.2) (also reported in Theorem E.1 for completeness) with P ' P p M 0 ,π q , A , Q ' P p M 0 ,π q A , and A ' t Ψ ı ‰ 0 u :

<!-- formula-not-decoded -->

Step 4: KL-divergence Computation Let M P M , we denote with P A , M ,π the joint probability distribution of all events realized by the execution of the algorithm in the MDP \ R (the presence of π is irrelevant as we assume it known):

<!-- formula-not-decoded -->

where H t ´ 1 ' p s 1 , a 1 , h 1 , s 1 1 , . . . , s t ´ 1 , a t ´ 1 , h t ´ 1 , s 1 t ´ 1 q is the history. Let ı P I and denote with p 0 and p ı the transition models associated with M 0 and M ı . Let us now move to the KL-divergence:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

having observed that the transition models differ in ı ' p s ˚ , a ˚ , h ˚ q and defined N τ h ˚ p s ˚ , a ˚ q ' ř τ t ' 1 1 tp s t , a t , h t q ' p s ˚ , a ˚ , h ˚ qu and the last passage is obtained by Lemma E.4 with D ' 2 (and /epsilon1 ' 2 /epsilon1 1 ). Putting all together, we have:

Thus, summing over p s ˚ , a ˚ , h ˚ q P I , we have:

<!-- formula-not-decoded -->

The number of states is given by S ' | S | ' S ` 4 , the number of actions is given by A ' | A | ' A ` 1 . Let us first consider the time-homogeneous case, i.e., H ' 1 :

<!-- formula-not-decoded -->

For δ ă 1 { 16 , S ě 9 , A ě 2 , H ě 10 , we obtain:

<!-- formula-not-decoded -->

For the time-inhomogeneous case, instead, we select H ' H { 2 , to get:

<!-- formula-not-decoded -->

For δ ă 1 { 16 , S ě 9 , A ě 2 , H ě 12 , we obtain:

<!-- formula-not-decoded -->

Theorem B.3. Let A ' p µ, τ q be an p /epsilon1, δ q -PAC algorithm for d G -IRL. Then, there exists an IRL problem p M , π E q such that, if /epsilon1 ď 1 , δ ď 1 { 2 , S ě 16 , A ě 2 , H ě 131 , the expected sample complexity is lower bounded by:

- if the transition model p is time-inhomogeneous:

<!-- formula-not-decoded -->

- if the transition model p is time-homogeneous:

<!-- formula-not-decoded -->

Proof. Step 1: Instances Construction The construction of the hard MDP \ R instances for this second bound follows steps similar to those of reward free exploration (Jin et al., 2020) and the instances are reported in Figure 5 in a semiformal way. The state space is given by S ' t s start , s root , s 1 , . . . , s S , s 1 1 , . . . , s 1 S u and the action space is given by A ' t a 0 , a 1 , . . . , a A u . We assume S to be divisible by 16 . The transition model is described below and the horizon is H ě 3 .

The agent begins in state s start, where every action has the same effect. Specifically, if the stage h ă H ( H P /llbracket H /rrbracket , whose value will be chosen later), then there is probability 1 { 2 to remain in s start and a probability 1 { 2 to transition to s root . Instead, if h ě H , the state transitions to s root deterministically. From state s root, every action has the same effect and the state transitions with equal probability 1 { S to a state s i with i P /llbracket S /rrbracket . In every state s i and every stage h , action a 0 allows reaching states s 1 1 , . . . , s 1 S with equal probability 1 { S . Instead, by playing the other actions a j with j ě 1 at stage h , the probability distribution of the next state is given by p h p s 1 k | s i , a j q ' p 1 ` /epsilon1 1 v p s i ,a j ,h q k q{ S where the vector v p s i ,a j ,h q ' p v p s i ,a j ,h q 1 , . . . , v p s i ,a j ,h q S q P V , where V : ' tt´ 1 , 1 u S : ř S j ' 1 v j ' 0 u and /epsilon1 1 P r 0 , 1 { 2 s . Notice that, having fixed H , the possible values of h are t 3 , . . . , 2 ` H u . States s 1 1 , . . . , s 1 S are absorbing states. The expert's policy always plays action a 0 .

Let us introduce the set I : ' t s 1 , . . . , s S u ˆ t a 1 , . . . , a A u ˆ /llbracket 3 , H ` 2 /rrbracket . Let v ' p v ı q ı P I P V I which is the set of vectors having as components the elements v ı determining the probability distribution of the next state starting from the triple ı P I . We denote with M v the MDP \ R induced by v . We can construct the class of instances denote by M ' t M v : v P V I u . Moreover, we denoted with M v ı Ð w the instance in which we replace the ı component of v , i.e., v ı , with w P V and M v ı Ð 0 the instance in which we replace the ı component of v , i.e., v ı , with the zero vector.

Step 2: Feasible Set Computation Thanks to Lemma E.6, we know that there exists a subset V Ă V of cardinality at least | V | ě 2 S { 5 such that for every v, w P V with v ‰ w we have ř S j ' 1 | v j ´ w j | ě S { 16 . Thus, we consider the set V I Ă V I and to build the instances v P V I and v, w P V with v ‰ w . Let ı P I , the induced instances are denoted by M v ı Ð v , M v ı Ð w P M .

To lower bound the Hausdorff distance, we focus on the triple ı ' p s ˚ , a ˚ , h ˚ q and we enforce the convenience of action a 0 over action a ˚ . For both MDP \ R M v ı Ð v and M v ı Ð w , let r v P R M v ı Ð v and r w P R M v ı Ð w , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Figure 5. Semi-formal representation of the the hard instances MDP \ R used in the proof of Theorem B.3.

<!-- image -->

In order to lower bound the Hausdorff distance H d G ` M v ı Ð v , M v ı Ð w ˘ , we set for M v ı Ð v :

<!-- formula-not-decoded -->

We now want to find the closest reward function r w for the instance M v ı Ð w , recalling that there are at least S { 16 components of the vectors v and w that are different. Clearly, we can set r w l p s 1 j q ' r v l p s 1 j q ' ´ v j for all j P /llbracket S /rrbracket in which v j ' w j since this will not increase the Hausdorff distance and make the constraint in Equation (10) less restrictive. For symmetry reasons, we can limit our reasoning to the case in which v j ' ´ 1 and w j ' 1 for the j terms in which they are different. This, way, the constraint becomes:

<!-- formula-not-decoded -->

where N v,w ' ř S j ' 1 1 t v j ' w j u . Notice that z P r´ 1 , 1 s . Let α ' N v,w S , the Hausdorff distance can be lower bounded by:

<!-- formula-not-decoded -->

where the first inequality derives from the fact that to have a Hausdorff distance smaller than 1 , we must take z ă 0 at least and the second inequality is obtained by recalling that 1 ´ α ě 1 16 for the packing argument.

We enforce the following constraint on this quantity:

<!-- formula-not-decoded -->

Notice that /epsilon1 1 ď 1 { 2 whenever H ě H ` 130 .

Step 3: Lower bounding Probability Let us consider an p /epsilon1, δ q -correct algorithm A that outputs the estimated feasible set p R . Thus, consider ı P I and v P V I , we can lower bound the error probability:

For every ı P I and v P V I , let us define the identification function :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let w P V . If Ψ ı, v ' w , then, H d G p R M v ı Ð Ψ ı, v , R M v ı Ð w q ' 0 . Otherwise, if Ψ ı, v ‰ w , we have:

<!-- formula-not-decoded -->

where the first inequality follows from triangular inequality and the second one from the definition of identification function Ψ ı, v . From Equation (11), we have that H d G ´ R M Ψ ı ı , R M v ı ¯ ě 2 /epsilon1 . Thus, it follows that H d G p p R , R M v ı Ð w q ě /epsilon1 . This implies the following inclusion of events for w P V :

Thus, we can proceed by lower bounding the probability:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality follows from bounding the maximum of probability with the average. We can now apply the Fano's inequality (Theorem E.2) with reference probability P 0 ' P p M v ı Ð 0 ,π q , A , P w ' P p M v ı Ð w ,π q , A , and A w ' t Ψ ı, v ‰

w u :

<!-- formula-not-decoded -->

Step 4: KL-divergence Computation Let M be an instance, we denote with P A , M ,π the joint probability distribution of all events realized by the execution of the algorithm in the MDP \ R (the presence of π is irrelevant as we assume it known):

<!-- formula-not-decoded -->

where H t ´ 1 ' p s 1 , a 1 , h 1 , s 1 1 , . . . , s t ´ 1 , a t ´ 1 , h t ´ 1 , s 1 t ´ 1 q is the history up to time t ´ 1 . Let ı P I and v P V and denote with p v ı Ð 0 and p v ı Ð w the transition models associated with M v ı Ð 0 and M v ı Ð w . Let us now move to the KL-divergence and denoting ı ' p s ˚ , a ˚ , h ˚ q : Thus, we have:

having observed that the transition models differ in ı ' p s ˚ , a ˚ , h ˚ q and defined N τ h ˚ p s ˚ , a ˚ q ' ř τ t ' 1 1 tp s t , a t , h t q ' p s ˚ , a ˚ , h ˚ qu and the last passage is obtained by Lemma E.4 with D ' S . Plugging into Equation (12), we obtain:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Since the derivation is carried out for every ı P I and v P V I , we can perform the summation over ı and the average over v :

<!-- formula-not-decoded -->

Notice that we get a guarantee on a mean under the uniform distribution of the instances of the sample complexity. Thus, there must exist one v hard P V such that:

<!-- formula-not-decoded -->

Then, we select δ ď 1 { 2 , recall that | V | ě 2 S { 5 , we get:

<!-- formula-not-decoded -->

The number of states is given by S ' | S | ' 2 S ` 2 , the number of actions is given by A ' | A | ' A ` 1 . Let us first consider the time-homogeneous case, i.e., H ' 1 , for S ě 16 , A ě 2 , H ě 130 , we have:

<!-- formula-not-decoded -->

For the time inhomogeneous case, we select H ' H { 2 , to get, under the same conditions:

<!-- formula-not-decoded -->

## B.4. Proofs of Section 6

Theorem D.2 (Sample Complexity of US-IRL ) . Let /epsilon1 ą 0 and δ P p 0 , 1 q , US-IRL is p /epsilon1, δ q -PAC for d G -IRL and with probability at least 1 ´ δ it stops after τ samples with:

- if the transition model p is time-inhomogeneous:

<!-- formula-not-decoded -->

where C ' log p e {p S ´ 1 q ` p 8 eH 2 q{pp S ´ 1 q /epsilon1 2 qp log p SAH { δ q ` 4 e qq ;

- if the transition model p is time-homogeneous and :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Westart with the case in which the transition model is time-inhomogeneous. In this case, we introduce the following good event:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where p h is the true transition model and p p t h is its estimate via Equation (3) at time t . Thanks to Lemma B.4, we have that P p M ,π E q , A p E q ě 1 ´ δ . Thus, under the good event E , we apply Theorem 3.2:

where we exploited the fact that the expert's policy is known in the last but one passage and used Pinsker's inequality in the last passage. When the US-IRL stops we have that max p s,a,h qP S ˆ A ˆ /llbracket H /rrbracket C t h p s, a q ď /epsilon1 and, consequently, for all p s, a, h q P S ˆ A ˆ /llbracket H /rrbracket we have:

<!-- formula-not-decoded -->

Thus, the algorithm stops at the smallest t such that:

<!-- formula-not-decoded -->

Thus, by applying Lemma 15 of (Kaufmann et al., 2021), we obtain:

<!-- formula-not-decoded -->

By recalling that τ ' SAHn τ h p s, a q , and bounding H ´ h ` 1 ď H , we obtain:

<!-- formula-not-decoded -->

If the transition model is time-homogeneous, we suppress the subscript h and the algorithm US-IRL , will merge together all the samples collected at different stages h . Let us define n t p s, a q ' ř H h ' 1 n t h p s, a q and n t p s, a, s 1 q ' ř H h ' 1 n t h p s, a, s 1 q . Now the transition model will be estimated straightforwardly as follows:

Let us consider now the following good event:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

This allows us to compute the maximum value of n τ p s, a q :

<!-- formula-not-decoded -->

Recalling that τ ' SAn τ p s, a q , we obtain:

<!-- formula-not-decoded -->

Lemma B.4. The following statements hold:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let us start with the first statement. Similarly to Lemma 10 of (Kaufmann et al., 2021), we apply first a union bound and then technical Proposition 1 of (Jonsson et al., 2020) (also reported as Lemma E.3 for completeness) to concentrate the KL-divergence:

<!-- formula-not-decoded -->

The proof of the second statement is analogous having simply observed that the union bound has to be performed over S ˆ A only.

## C. Examples of Section 3.2

In this appendix, we provide a detailed derivations of the examples presented in Section 3.2.

Example 3.1 (State-only reward r h p s q ) . State-only reward functions have been widely considered in many IRL approaches (e.g., Ng &amp; Russell, 2000; Abbeel &amp; Ng, 2004; Syed &amp; Schapire, 2007; Komanduru &amp; Honorio, 2019). We formalize the state-only feasible reward set as follows:

<!-- formula-not-decoded -->

Consider the MDP \ R of Figure 1a with H ' 2 , π E h p s 0 q' p π E h p s 0 q' a 1 with h Pt 1 , 2 u . Set p 1 p s ` | s 0 , a 1 q' 1 { 2 ` /epsilon1 { 4 and p p 1 p s ` | s 0 , a 1 q' 1 { 2 ´ /epsilon1 { 4 and, thus, } p 1 p¨| s 0 , a 1 q´ p p 1 p¨| s 0 , a 1 q} 1 ' /epsilon1 . Let us set r 2 p s ` q' 1 and r 2 p s ´ q'´ 1 , which makes π E optimal under p . We observe that p R is defined by p r 2 p s ´ qď p r 2 p s ` q . Recalling that the rewards are bounded in r´ 1 , 1 s , we have H d G p R state , p R state qě 1 .

Proof. For the MDP \ R M , in order to make π E 1 p s 0 q ' a 1 optimal, we have to enforce:

<!-- formula-not-decoded -->

Similarly, to make p π E 1 p s 0 q ' a 1 , we have for M :

p p Thus, suppose, we set r 2 p s ´ q ' 1 and r 2 p s ` q ' ´ 1 , we have:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Example 3.2 (Time-homogeneous reward r p s, a q ) . Time-homogeneous reward functions have been employed in several RL(e.g., Dann &amp; Brunskill, 2015) and IRL settings (e.g., Lindner et al., 2022). We formalize the time-homogeneous feasible reward set as follows:

<!-- formula-not-decoded -->

Consider the MDP \ R of Figure 1b with H ' 2 , π E 1 p s 0 q' p π E 1 p s 0 q' a 1 and π E 2 p s 0 q' p π E 2 p s 0 q' a 2 . For h Pt 1 , 2 u , we set p h p s 0 | s 0 , a 1 q' 1 { 2 ` /epsilon1 { 4 and p p h p s 0 | s 0 , a 1 q' 1 { 2 ´ /epsilon1 { 4 , thus, } p h p¨| s 0 , a 1 q´ p p h p¨| s 0 , a 1 q} 1 ' /epsilon1 . We set r p s 0 , a 1 q' 1 , r p s 0 , a 2 q' 1 ´ /epsilon1 { 6 , and r p s 1 , a 1 q' r p s 1 , a 2 q' 1 { 2 making π E optimal. We can prove that H d G p R hom , p R hom qě 1 { 4 .

Proof. Consider the MDP \ R M and we set r p s 0 , a 1 q ' 1 , r p s 0 , a 2 q ' 1 ´ /epsilon1 { 12 , and r p s 1 , a q ' 1 { 2 for a P t a 1 , a 2 u . We immediately observe that π E is optimal since for h ' 2 , r p s 0 , a 1 q ě r p s 0 , a 2 q and for h ' 1 :

<!-- formula-not-decoded -->

Consider now the alternative MDP \ R M , we have to enforce the following two conditions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The way of enforcing Equation (13) that is less constraining for Equation (14) is setting r p s 0 , a 1 q ' p r p s 0 , a 2 q , to get:

This implies:

<!-- formula-not-decoded -->

p p Example 3.3 ( β -margin reward) . A β -margin reward enforces a suboptimality gap of at least β ą 0 (Ng &amp; Russell, 2000; Komanduru &amp; Honorio, 2019). We formalize it in the finite-horizon case with a sequence β ' p β h q h P /llbracket H /rrbracket , possibly different for every stage:

<!-- formula-not-decoded -->

Consider the MDP \ R in Figure 1a with π E h p s 0 q ' p π E h p s 0 q ' a 1 for h P t 1 , 2 u . We set p 1 p s ` | s 0 , a 1 q ' 1 { 2 ` /epsilon1 and p p 1 p s ` | s 0 , a 1 q ' 1 { 2 ´ /epsilon1 . We set for MDP \ R M the reward function as r 1 p s 0 , a q ' 0 and r h p s ` , a q ' ´ r h p s ´ , a q ' 1 for a P t a 1 , a 2 u and h P /llbracket 2 , H /rrbracket . In p s 0 , 1 q the suboptimality gap is β 1 ' 2 ` 2 /epsilon1 p H ´ 1 q . By selecting H ě 1 ` 1 { /epsilon1 , the feasible set p R β -mar is empty.

<!-- formula-not-decoded -->

Proof. Concerning the MDP \ R M , we observe that by setting r 1 p s 0 , a 1 q ' 1 , r 1 p s 0 , a 2 q ' ´ 1 , and r h p s ` , a q ' ´ r h p s ´ , a q ' 1 for a P t a 1 , a 2 u and h P /llbracket 2 , H /rrbracket , the policy π E is optimal. In particular, in state-stage pair p s 0 , 1 q the suboptimality gap is given by β 1 ' 2 ` 2 /epsilon1 p H ´ 1 q . To enforce the optimality of π E ' π E in the MDP \ R x M , we have:

p p Thus, if β 1 ě 2 , we have that the feasible set p R β -sep is empty. Thus, we select H ě 1 ` 1 { /epsilon1 to have β 1 ě 4 .

## D. Unknown Expert's Policy π E

In this appendix, we extend the lower bounds and the algorithm for the case in which the expert's policy is unknown. Clearly, if the expert's policy is deterministic, under the generative model setting, its estimation is trivial as it suffices to query every state and stage (resp. state) exactly once for time-inhomogeneous (resp. time-homogeneous) policies, leading to E p M ,π E q , A r τ s ' HS (resp. E p M ,π E q , A r τ s ' S ). Thus, we consider a more general setting in which the expert's policy can be stochastic (still being optimal). Specifically, we consider the following assumption.

Assumption D.1. There exists a known constant π min P p 0 , 1 s such that every action played by the expert's policy π E is played with at least probability π min :

<!-- formula-not-decoded -->

Intuitively, Assumption D.1 formalizes a form of identifiability for the policy. As already mentioned in Section 3, what matters for learning the feasible reward set is whether an action is played by the agent (not the corresponding probability). Assumption D.1 enforces that every optimal action must be played with a minimum (known) non-null probability π min . We shall show that if this assumption is violated, the problem becomes non-learnable.

## D.1. Lower Bound

The following result provides a lower bound for learning the feasible reward set according to the PAC requirement of Definition (4.1) when the expert's policy is unknown, but the transition model is known. Clearly, one can combine this result with the ones of Section 5 to address the setting in which both the expert's policy and the transition model are unknown.

Theorem D.1. Let A ' p µ, τ q be an p /epsilon1, δ q -PAC algorithm for d G -IRL. Then, there exists an IRL problem p M , π E q where π E fulfills Assumption D.1 such that, if /epsilon1 ď 1 { 2 , δ ă 1 { 16 , S ě 7 , A ě 2 , and H ě 3 , the number of samples N is lower bounded in expectation by:

- if the expert's policy π E is time-inhomogeneous:

<!-- formula-not-decoded -->

- if the expert's policy π E is time-homogeneous:

<!-- formula-not-decoded -->

Before presenting the proof, let us comment the result. We observe that when Assumption D.1 is violated, i.e., π min Ñ 0 , the sample complexity lower bound degenerates to infinity, proving that the problem become non-learnable.

Proof. Step 1: Instances Construction The hard MDP \ R instances are depicted in Figure 6 in a semi-formal way. The state space is given by S ' t s start , s root , s 1 , . . . , s S , s sink u and the action space is given by A ' t a 0 , a 1 , . . . , a A u . The transition model is described below and the horizon is H ě 3 . We introduce the constant H P /llbracket H /rrbracket , whose value will be chosen later. Let us observe, for now, that if H ' 1 , the transition model is time-homogeneous.

The agent begins in state s start, where every action has the same effect. Specifically, if the stage h ă H , then there is probability 1 { 2 to remain in s start and a probability 1 { 2 to transition to s root . Instead, if h ě H , the state transitions to s root deterministically. From state s root, every action has the same effect and the state transitions with equal probability 1 { S to a state s i with i P /llbracket S /rrbracket . In all states s i , apart from a specific one, i.e., state s ˚ , the expert's policy plays action a 0 deterministically, i.e., π E h p a 0 | s i q ' 1 and the state transitions deterministically to s sink . In state s ˚ the expert's policy plays a 0 as the other ones if the stage h ‰ h ˚ , where h ˚ P /llbracket H /rrbracket is a predefined stage. If, instead, h ' h ˚ , the expert's action plays a 0 w.p. 1 ´ π min and a specific action a ˚ w.p. π min P r 0 , 1 { 2 s . Then, the transition is deterministic to state s sink . Notice that, having fixed H , the possible values of h ˚ are t 3 , . . . , 2 ` H u . State s sink is an absorbing state.

Let us consider the base instance π 0 in which the expert's policy always plays action a 0 deterministically. 14 Additionally, by varying the pair /lscript : ' p s ˚ , h ˚ q P t s 1 , . . . , s S u ˆ /llbracket 3 , H ` 2 /rrbracket ' : J , we can construct the class of instances denoted by M ' t π /lscript : /lscript P t 0 u Y J u .

Step 2: Feasible Set Computation Let us consider an instance π /lscript P M , we now seek to provide a lower bound to the Hausdorff distance H d G p R π 0 , R π /lscript q . To this end, we focus on the pair /lscript ' p s ˚ , h ˚ q and we enforce the convenience of both actions a 0 and a ˚ over the other actions. Since both actions are played with non-zero probability by the expert's policy, their value function must be the same. Let us denote with r /lscript P R π /lscript , we must have for all a j R t a 0 , a ˚ u :

<!-- formula-not-decoded -->

Consider now the base instance π 0 and denote with r 0 P R π 0 . Here we have to enforce the convenience of action a 0 over all the others, including a ˚ :

14 In this construction, the MDP \ Rdoes not change across the instances, but what changes is the expert's policy. Thus, we parametrize the instances through the policy rather than the MDP \ R.

Figure 6. Semi-formal representation of the the hard instances MDP \ R used in the proof of Theorem D.1.

<!-- image -->

<!-- formula-not-decoded -->

In order to lower bound the Hausdorff distance, we perform a valid assignment of the rewards for the base instance:

<!-- formula-not-decoded -->

Thus, the Hausdorff distance can be bounded as follows, having renamed, for convenience x ' r /lscript h ˚ p s ˚ , a 0 q and y ' r /lscript h ˚ p s ˚ , a ˚ q :

<!-- formula-not-decoded -->

Step 3: Lower bounding Probability Let us consider an p /epsilon1, δ q -correct algorithm A that outputs the estimated feasible set p R . Thus, for every ı P J , we can lower bound the error probability:

For every ı P J , let us define the identification function :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Let  P t 0 , ı u . If Ψ ı '  , then, H d G p R π Ψ ı , R π  q ' 0 . Otherwise, if Ψ ı ‰  , we have:

where the first inequality follows from triangular inequality and the second one from the definition of identification function Ψ ı . From Equation (11), we have that H d G ` R π Ψ ı , R π  ˘ ě 1 . Thus, it follows that H d G ´ p R , R π  ¯ ě 1 2 . This implies the following inclusion of events for  P t 0 , ı u :

Thus, we can proceed by lower bounding the probability:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where the second inequality follows from the observation that max t a, b u ě 1 2 p a ` b q and the equality from observing that Ψ ı P t 0 , ı u . We can now apply the Bretagnolle-Huber inequality (Lattimore &amp; Szepesv´ ari, 2020, Theorem 14.2) (also reported in Theorem E.1 for completeness) with P ' P p M 0 ,π q , A , Q ' P p M 0 ,π q , A , and A ' t Ψ ı ‰ 0 u :

<!-- formula-not-decoded -->

Step 4: KL-divergence Computation Let M P M , we denote with P A , M ,π the joint probability distribution of all events realized by the execution of the algorithm in the MDP \ R (the presence of p is irrelevant as it does not change across the different instances):

<!-- formula-not-decoded -->

where H t ´ 1 ' p s 1 , a 1 , h 1 , s 1 1 , a E 1 , . . . , s t ´ 1 , a t ´ 1 , h t ´ 1 , s 1 t ´ 1 , a E t ´ 1 q is the history. Let ı P I . Let us now move to the KL-divergence between the instances π 0 and π ı for some ı ' p s ˚ , h ˚ q P J :

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

having observed that the transition models differ in ı ' p s ˚ , h ˚ q and defined N τ h ˚ p s ˚ q ' ř τ t ' 1 1 tp s t , h t q ' p s ˚ , h ˚ qu and the last passage is obtained by explicitly computing the KL-divergence:

Putting all together, we have:

<!-- formula-not-decoded -->

Thus, summing over p s ˚ , a ˚ q P J , we have:

<!-- formula-not-decoded -->

The number of states is given by S ' | S | ' S ` 3 . Let us first consider the time-homogeneous case, i.e., H ' 1 :

<!-- formula-not-decoded -->

For δ ă 1 { 16 , S ě 7 , A ě 2 , H ě 2 , we obtain:

<!-- formula-not-decoded -->

For the time-inhomogeneous case, instead, we select H ' H { 2 , to get:

<!-- formula-not-decoded -->

For δ ă 1 { 16 , S ě 7 , A ě 2 , H ě 2 , we obtain:

<!-- formula-not-decoded -->

## D.2. Algorithm

In this appendix, we extend US-IRL to the expert's policy estimation under Assumption D.1. The pseudocode is reported in Algorithm 2. The interaction protocol follows the same principles of Algorithm 1, with the only difference that the confidence function, now, must account for the policy estimation, leading to the following function for every p s, a, h q P

```
Input: significance δ P p 0 , 1 q , /epsilon1 target accuracy t Ð 0 , /epsilon1 0 Ð`8 while /epsilon1 t ą /epsilon1 do t Ð t ` SAH Collect one sample from each p s, a, h q P S ˆ A ˆ /llbracket H /rrbracket Update p t and π E,t according to (3) Update /epsilon1 t ' max p s,a,h qP S ˆ A ˆ /llbracket H /rrbracket C t h p s, a q (resp. C t h p s, a q )
```

```
p p r end while
```

Algorithm 2. UniformSampling-IRL ( US-IRL ) for time-inhomogeneous (resp. time-homogeneous) transition models and expert's policies.

<!-- formula-not-decoded -->

where:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

It is worth noting that we have distributed the confidence δ equally between the problem estimating the policy and that of estimating the transition model. The following theorem provides the sample complexity of US-IRL .

Theorem D.2 (Sample Complexity of US-IRL ) . Let /epsilon1 ą 0 and δ P p 0 , 1 q , under Assumption D.1, US-IRL is p /epsilon1, δ q -PAC for d G -IRL and with probability at least 1 ´ δ it stops after τ samples with:

- if the transition model p and the expert's policy π E are time-inhomogeneous:

<!-- formula-not-decoded -->

- where C 1 ' log p e {p S ´ 1 q ` p 8 eH 2 q{pp S ´ 1 q /epsilon1 2 qp log p 2 SAH { δ q ` 4 e qq and C 2 ' 2 log ´ log p 4 SAH { δ q` 2 log p 1 {p 1 ´ π min qq ¯ . · if the transition model p and the expert's policy π E are time-homogeneous:

<!-- formula-not-decoded -->

Before moving to the proof, let us observe that the result matches the rate of the lower bound of Theorem D.1 up to logarithmic terms.

<!-- formula-not-decoded -->

Proof. We make use of the notation of the proof of Theorem D.2. We start with the case in which the transition model is

15 As for the transition model, one can adapt the confidence function for the case of stationary policy in straightforward way:

where:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In principle, one can also consider the case of a time-homogeneous transition model and time-inhomogeneous expert's policy. We omit it because it adds nothing to the characteristics of the problem and of the algorithms.

time-inhomogeneous. In addition to the good event E related to the transition model, we introduce the following one:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

ˇ ˇ where π E h is the true expert's policy and p π E,t is its estimate via Equation (3) at time t . Thanks to Lemma B.4 and Lemma D.3, we have that P p E X E π q ě 1 ´ δ . Thus, under the good event E X E π , we apply Theorem 3.2 to obtain H d G p R , p R τ q ď max p s,a,h qP S ˆ A ˆ /llbracket H /rrbracket C t h p s, a q . A sufficient condition to make this term ď /epsilon1 is to request the following ones:

For the first one, we first enforce the condition:

<!-- formula-not-decoded -->

Using Lemma 15 of (Kaufmann et al., 2021) and enforcing n t h p s q ě 1 , we obtain:

<!-- formula-not-decoded -->

Combining this result with that of Theorem D.2 for what concerns the transition model, we obtain:

<!-- formula-not-decoded -->

Analogous derivations can be carried out for the case of time-homogenous policy using the good event:

<!-- formula-not-decoded -->

Lemma D.3. Under Assumption D.1, the following statements hold:

where r ξ p n, δ q : ' log p 2 SAn 2 { δ q log p 1 {p 1 ´ π min qq . We omit the tedious but straightforward derivation.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. Let us start with the first statement. We apply first a union bound and, then, Lemma E.5 to perform the concentration:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where on the first passage we enforced the condition on the time instants in which the policy estimate changes (i.e., when p s, h q is visited) and we denotes such an estimate as p π E, r n s h . Then, after a union bound, we apply Lemma E.5. The proof of the second statement is analogous having simply observed that the union bound has to be performed over S ˆ A only.

## E. Technical Lemmas

Theorem E.1. ( Bretagnolle-Huber inequality (Lattimore &amp; Szepesv´ ari, 2020, Theorem 14.2)) Let P and Q be probability measures on the same measurable space p Ω , F q , and let A P F be an arbitrary event. Then,

<!-- formula-not-decoded -->

where A c ' Ω z A is the complement of A .

Theorem E.2. ( Fano inequality (Gerchinovitz et al., 2017, Proposition 4)) Let P 0 , P 1 , . . . , P M be probability measures on the same measurable space p Ω , F q , and let A 1 , . . . , A M P F be a partition of Ω . Then,

where A c ' Ω z A is the complement of A .

<!-- formula-not-decoded -->

Lemma E.3. (Jonsson et al., 2020, Proposition 1) Let P ' p p 1 , . . . , p D q be a categorical probability measure on the support /llbracket D /rrbracket . Let P n ' p p p 1 , . . . , p p D q be the maximum likelihood estimate of P obtained with n ě 1 independent samples. Then, for every δ P p 0 , 1 q it holds that:

Lemma E.4. Let /epsilon1 P r 0 , 1 { 2 s and v P t´ /epsilon1, /epsilon1 u D such that ř d i ' 1 v i ' 0 . Consider the two categorical distributions P ' ` 1 D , 1 D , . . . , 1 D ˘ and P ' ` 1 ` v 1 D , 1 ` v 2 D , . . . , 1 ` v D D ˘ . Then, it holds that:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Proof. First of all we recall that since ř M i ' 1 v i ' 0 , we have |t i P /llbracket D /rrbracket : v i ' /epsilon1 u| ' | i P /llbracket D /rrbracket : v i ' ´ /epsilon1 | ' D { 2 . Let us compute the KL-divergence D KL p P , Q q :

<!-- formula-not-decoded -->

where we used the inequality log p 1 ` x q ď x for x ě 0 and ´ log p 1 ´ x q ď 1 1 ´ x ´ 1 for 0 ă x ă 1 and exploited that

/epsilon1 ď 1 2 . Let us now move to the second KL-divergence D KL p Q , P q :

<!-- formula-not-decoded -->

where we used the inequality ´ log p 1 ´ x q ď 1 1 ´ x ´ 1 for 0 ă x ă 1 and observed that /epsilon1 ď 1 2 .

Lemma E.5. Let P ' p p 1 , . . . , p D q be a categorical probability measure on the support /llbracket D /rrbracket . Let P n ' p p p 1 , . . . , p p D q be the maximum likelihood estimate of P obtained with n ě 1 independent samples. Then, if p i P t 0 u Y r p min , 1 s for some p min P p 0 , 1 s . Then, for every i P /llbracket D /rrbracket individually, for every δ P p 0 , 1 q , it holds that:

<!-- formula-not-decoded -->

Proof. Let i P /llbracket D /rrbracket such that p i ą 0 and, thus, 1 t p i ' 0 u ' 0 . By assumption, it must be that p i ě p min . To make a mistake, we must have that 1 t p p i ' 0 u ' 1 , and, thus, p p i ' 0 . Thus, we compute the probability that no sample i is observed among the n ones:

where we exploited the fact that the random variables X j are i.i.d.. If n ' 0 the latter expression is 1 . If, instead, n ě 1 , by setting the last expression equal to δ , we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The result follows.

Lemma E.6. Let V ' t v P t´ 1 , 1 u D : ř D j ' 1 v j ' 0 u . Then, the D 16 -packing number of V w.r.t. the metric d p v, v 1 q ' ř D j ' 1 | v j ´ v 1 j | is lower bounded by 2 D 5 .

Proof. Let us denote the packing number with M p /epsilon1 ; V , d q and the covering number with N p /epsilon1 ; V , d q . It is well known that N p /epsilon1 ; V , d q ď M p /epsilon1 ; V , d q (Gy¨ orfi et al., 2002). Thus, a lower bound to the covering number is a lower bound to the packing number. Let us consider the (pseudo)metric d 1 p v, v 1 q ' ř D { 2 j ' 1 | v j ´ v 1 j | that considers the first half of the components only. Clearly, we have that d 1 p v, v 1 q ď d p v, v 1 q . Therefore, any /epsilon1 -cover w.r.t. d p v, v 1 q is an /epsilon1 -cover w.r.t. d 1 p v, v 1 q and, consequently, N p /epsilon1 ; V , d 1 q ď N p /epsilon1 ; V , d q . Since the (pseudo)metric d 1 considers only the first half of the components, constructing an /epsilon1 -cover of V w.r.t. d 1 is equivalent to constructing an /epsilon1 -cover of V 1 w.r.t. d 1 , where V 1 ' t´ 1 , 1 u D { 2 . V 1 considers the first half of the components of vectors of V , that can be freely chosen, disregarding the summation constraint. 16 Thus, N p /epsilon1 ; V , d 1 q ' N p /epsilon1 ; V 1 , d 1 q . Notice that d 1 is now a proper metric on V 1 ' t´ 1 , 1 u D { 2 . Now, we reduce the problem to constructing cover on the Hamming space H ' t 0 , 1 u D { 2 . Indeed, we can always map an p /epsilon1 { 2 q -cover for the Hamming space H to an /epsilon1 -cover for the space V 1 . Specifically, let p h l q l an p /epsilon1 { 2 q -cover for the Hamming space, we construct p v 1 l q l by applying the following transformation:

16 From an algebraic perspective, V 1 can be considered the quotient set obtained from V by means of the equivalence relation v ' v 1 ðñ v j ' v j 1 for all j P /llbracket D { 2 /rrbracket .

<!-- formula-not-decoded -->

or, in more convenient way, v 1 ' 2 h ´ 1 . Let v 1 P V 1 :

<!-- formula-not-decoded -->

The covering number of a Hamming space has been lower bounded in (Cohen &amp; Frankl, 1985) for /epsilon1 P /llbracket D { 2 /rrbracket as:

We take /epsilon1 ' D { 16 , and we use the known bound ř k i ' 0 ` n k ˘ ď ` en k ˘ k :

From, which, we get:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->