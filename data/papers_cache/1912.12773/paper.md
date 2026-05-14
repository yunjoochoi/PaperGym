## Learning Predictive Models From Observation and Interaction

Karl Schmeckpeper 1 , Annie Xie 2 , Oleh Rybkin 1 , Stephen Tian 3 , Kostas Daniilidis 1 , Sergey Levine 3 , Chelsea Finn 2 1 University of Pennsylvania, 2 Stanford University, 3 UC Berkeley

## Abstract

Learning predictive models from interaction with the world allows an agent, such as a robot, to learn about how the world works, and then use this learned model to plan coordinated sequences of actions to bring about desired outcomes. However, learning a model that captures the dynamics of complex skills represents a major challenge: if the agent needs a good model to perform these skills, it might never be able to collect the experience on its own that is required to learn these delicate and complex behaviors. Instead, we can imagine augmenting the training set with observational data of other agents, such as humans. Such data is likely more plentiful, but represents a different embodiment. For example, videos of humans might show a robot how to use a tool, but (i) are not annotated with suitable robot actions, and (ii) contain a systematic distributional shift due to the embodiment differences between humans and robots. We address the first challenge by formulating the corresponding graphical model and treating the action as an observed variable for the interaction data and an unobserved variable for the observation data, and the second challenge by using a domain-dependent prior. In addition to interaction data, our method is able to leverage videos of passive observations in a driving dataset and a dataset of robotic manipulation videos. A robotic planning agent equipped with our method can learn to use tools in a tabletop robotic manipulation setting by observing humans without ever seeing a robotic video of tool use.

## 1. Introduction

Humans have the ability to learn skills not just from their own interaction with the world but also by observing others. Consider an infant learning to use tools. In order to use a tool successfully, it needs to learn how the tool can interact with other objects, as well as how to move the tool to trigger this interaction. Such intuitive notion of physics can be learned by observing how adults use tools. More generally, observation is a powerful source of information about the world and how actions lead to outcomes. However, in the presence of physical differences (such as between an adult body and infant body), leveraging observation is challenging, as there is no direct correspondence between the demonstrator's and observer's actions. Evidence from neuroscience suggests that humans can effectively infer such correspondences and use it to learn from observation [45, 44]. In this paper, we consider this problem: can we enable agents to learn to solve tasks using both their own interaction and the passive observation of other agents?

1 Correspondence to: Karl Schmeckpeper &lt; karls@seas.upenn.edu &gt; .

Figure 1: Our system learns from action-observation sequences collected through interaction, such as robotic manipulation or autonomous vehicle data, as well as observations of another demonstrator agent, such as data from a human or a dashboard camera. By combining interaction and observation data, our model is able to learn to generate predictions for complex tasks and new environments without costly expert demonstrations.

<!-- image -->

In model-based reinforcement learning, solving tasks is commonly addressed via learning action-conditioned predictive models. However, prior works have learned such predictive models from interaction data alone [23, 22, 27,

15, 67]. When using both interaction and observation data, the setup differs in two important ways. First, the actions of the observed agent are not known, and therefore directly learning an action-conditioned predictive model is not possible. Second, the observation data might suffer from a domain shift if the observed agent has a different embodiment, operates at a different skill level, or exists in a different environment. Yet, if we can overcome these differences and effectively leverage observational data, we may be able to unlock a substantial source of broad data containing diverse behaviors and interactions with the world.

The main contribution of this work is an approach for learning predictive models that can leverage both videos of an agent annotated with actions and observational data for which actions are not available. We formulate a latent variable model for prediction, in which the actions are observed variables in the first case and unobserved variables in the second case. We further address the domain shift between the observation and interaction data by learning a domainspecific prior over the latent variables. We instantiate the model with deep neural networks and train it with amortized variational inference. In two problem settings - driving and object manipulation - we find that our method is able to effectively leverage observational data from dashboard cameras and humans, respectively, to improve the performance of action-conditioned prediction. Further, we find that the resulting model enables a robot to solve tool-use tasks, and achieves significantly greater success than a model that does not use observational data of a human using tools. Finally, we release our dataset of human demonstrations of tool use tasks to allow others to study this problem.

## 2. Related Work

Predictive models Video prediction can be used to learn useful representations and models in a fully unsupervised manner. These representations can be used for tasks such as action recognition [48], action prediction [61], classification [12], and planning [17, 18, 15, 28, 5, 22, 23, 29, 19]. Many different approaches have been applied to video prediction, including patch-centric methods [43], compositional models of content and motion [60, 12, 56], pixel autoregressive models [30], hierarchical models [7, 40, 39], transformation-based methods [37, 42, 17, 62, 34, 36, 33, 2, 8, 59], and other techniques [11, 66, 4, 38]. We choose to leverage transformation-based models, as they have demonstrated good results on robotic control domains [17, 15]. Recent work has also developed stochastic video prediction models for better handling of uncertainty [13, 33, 2, 68, 8, 63]. We also use a stochastic latent variable, and unlike these prior works, use it to model actions.

Learning action-conditioned visual dynamics models was proposed in [41, 17, 9]. Using MPC techniques and flow based prediction models, it has been applied to robotic manipulation [17, 18, 15, 28, 5, 72]. Other works address video games or physical simulation domains [22, 23, 29, 19, 65].

[64, 10, 69] show these models can generalize to unseen tasks and objects while allowing for challenging manipulation of deformable objects, such as rope or clothing. Unfortunately, large amounts of robotic interaction data containing complex behavior are required to train these models. These models are unable to learn from cheap and abundantly available natural videos of humans as they are trained in action-conditioned way, requiring corresponding control data for every video. In contrast, our method can learn from videos without actions, allowing it to leverage videos of agents for which the actions are unknown.

Learning to control without actions Recent work in imitation learning allows the agent to learn without access to the ground-truth expert actions. One set of approaches learn to translate the states of the expert into actions the agent can execute [53, 71]. [32] uses action-free data to learn a set of sub-goals for hierarchical RL. Another common approach is to learn a policy in the agent's domain that matches the expert trajectories under some similarity metric. [54, 55, 49, 51] use adversarial training or other metrics to minimize the difference between the states generated by the demonstrated policy and the states generated by the learned policy. [35] transform images from the expert demonstrations into the robot's domain to make calculating the similarity between states generated by different policies in different environments more tractable. [16] learn a latent policy on action-free data and use action-conditioned data to map the latent policy to real actions. [47, 14, 1] learn state representations that can be used to transfer policies from humans to robots. [50] use partially action-conditioned data to train a generative adversarial network to synthesize the missing action sequences. Unlike these works, which aim to specify a specific task to be solved through expert demonstrations, we aim to learn predictive models that can be used for multiple tasks, as we learn general properties of the real world through model-building.

Our work is more similar to a recently proposed method for learning action-conditioned predictive models without actions through learning action representations [46]. This work shows that very few active data points are required to learn sensorimotor mappings between true actions and action representations learned from observation in simple simulated settings. However, this approach addresses learning from observation where there is no physical differences between the demonstrator and the observer, and thus cannot be directly used for learning from humans. Our approach explicitly considers domain shift, allowing it to leverage videos of humans to significantly outperform this approach.

Domain adaptation In order to handle both observational and interaction data, our method must handle the missing actions and bridge the gap between the two domains (e.g., human arms vs. robot arms). Related domain adaptation methods have sought to map samples in one domain into equivalent samples in another domain [74, 3, 52, 25], or learn feature embeddings with domain invariance losses [58, 76, 20, 21, 57]. In our setting, regularizing for invariance across domains is insufficient. For example, if the observational data of humans involves complex manipulation (e.g., tool use), while the interaction data involves only simple manipulation, we do not want the model to be invariant to these differences. We therefore take a different approach: instead of regularizing for invariance across domains, we explicitly model the distributions over (latent) action variables in each of the domains.

Related to our method, DIVA [26] aims to avoid losing this information by proposing a generative model with a partitioned latent space. The latent space is composed of both components that are domain invariant and components that are conditioned on the domain. This allows the model to use domain-specific information while still remaining robust to domain shifts. We find that using an approach similar to DIVA in our model for learning from observation and interaction makes it more robust to the domain shift between interaction and observation data. However, in contrast to DIV A, our method explicitly handles sequence data with missing actions in one of the domains.

## 3. Learning Predictive Models from Observation and Interaction

In our problem setting, we assume access to interaction data of the form [ x 1 , a 1 , . . . , a T -1 , x T ] and observation data of the form [ x 1 , . . . , x T ] , where x i denotes the i th frame of a video and a i denotes the action taken at the i th time step. Domain shift may exist between the two datasets: for example, when learning object manipulation from videos of humans and robotic interaction, as considered in our experiments, there is a shift in the embodiment of the agent. Within this problem setting, our goal is to learn an actionconditioned video prediction model, p ( x c +1: T | x 1: c , a 1: T ) , that predicts future frames conditioned on a set of c context frames and sequence of actions.

To approach this problem, we formulate a probabilistic graphical model underlying the problem setting where actions are only observed in a subset of the data. In particular, in Subsection 3.1, we introduce a latent variable that explains the transition from the current frame to the next and, in the case of interaction data, encodes the action taken by the agent. We further detail how the latent variable model is learned from both observation and interaction data by amortized variational inference. In Subsection 3.2, we discuss how we handle domain shift by allowing the latent variables from different datasets to have different prior distributions. Finally, in Subsection 3.3, we discuss specific implementation details of our model.

Figure 2: We learn a predictive model of visual dynamics (in solid lines) that predicts the next frame x t +1 conditioned on the current frame x t and action representation z t . We optimize the likelihood of the interaction data, for which the actions are available, and observation data, for which the actions are missing. Our model is able to leverage joint training on the two kinds of data by learning a latent representation z that corresponds to the true action.

<!-- image -->

## 3.1. Graphical Model

To leverage both passive observations and active interactions, we formulate the probabilistic graphical model depicted in Figure 2. To model the action of the agent a t , we introduce a latent variable z t , distributed according to a domain-dependent distribution. The latent z t generates the action a t . We further introduce a forward dynamic model that, at each time step t , generates the frame x t given the previous frames x 1: t -1 and latent variables z 1: t -1 .

The generative model can be summarized as:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

The domain-dependent distribution over z t is Gaussian with learned mean and variance, described in more detail in Subsection 3.2, while the action decoder p ( a t | z t ) and transition model p ( x t +1 | x 1: t , z 1: t ) are neural networks with Gaussian distribution outputs, described in Subsection 3.3.

The transition model takes z t as input and thus necessitates the posterior distributions p ( z t | a t ) and p ( z t | x t , x t +1 ) . We require p ( z t | a t ) to generate latent variables for action-conditioned video prediction, i.e. sampling from p ( x t +1 | x 1: t , a 1: t ) = E p ( z 1: t | a 1: t ) [ p ( x t +1 | x 1: t , z 1: t )] . Wealso require p ( z t | x t , x t +1 ) since the actions are not available in some trajectories to obtain the first distribution.

The computation of these two posterior distributions is intractable, since the model is highly complex and non-linear, so we introduce the variational distributions q act ( z t | a t ) and q inv ( z t | x t , x t +1 ) to approximate p ( z t | a t ) and p ( z t | x t , x t +1 ) . The distributions are modeled as Gaussian and the variational parameters are learned by optimizing the evidence lower bound (ELBO), which is constructed by considering two separate cases. In the first, the actions of a trajectory are observed, and we optimize an ELBO on the joint probability of the frames and the actions:

<!-- formula-not-decoded -->

In the second case, the actions are not observed, and we optimize an ELBO on only the probability of the frames:

<!-- formula-not-decoded -->

The ELBO for the entire dataset is the combination of the lower bounds for the interaction data with actions, D i , and the observation data without actions, D o :

<!-- formula-not-decoded -->

We also add an auxiliary loss to align the distributions of z generated from the encoders q act ( z t | a t ) and q inv ( z t | x t , x t +1 ) , since the encoding z should be independent of the distribution it was sampled from. We encourage the two distributions to be similar through the JensenShannon divergence:

<!-- formula-not-decoded -->

Our final objective combines the evidence lower bound for the entire dataset and the Jensen-Shannon divergence, computed for the interaction data:

<!-- formula-not-decoded -->

We refer to our approach as prediction from observation and interaction (POI).

Figure 3: Network architecture. To optimize the ELBO, we predict the latent action z t from x t and x t +1 using the inverse model q inv . When the true actions are available, we additionally predict the latent action from the true action a t using the action encoder q act, and encourage the predictions from q act and q inv to be similar with a Jensen-Shannon divergence loss. The next frame is predicted from z t and x t .

<!-- image -->

## 3.2. Domain Shift

When learning from both observation and interaction, domain shift may exist between the two datasets. For instance, in the case of a robot learning by observing people, the two agents differ both in their physical appearance, as well as their action spaces. To address these domain shifts, we take inspiration from the domain-invariant approach described in [26]. We divide our latent variable z into z shared , which captures the parts of the latent action that are shared between domains, and z domain , which captures the parts of the latent action that are unique to each domain.

We allow the network to learn the difference between the z domain for each dataset by using different prior distributions. The prior p ( z shared t ) is the same for both domains, however, the prior for z domain t is different for the interaction dataset, p i ( z domain t ) , and the observational dataset, p o ( z domain t ) . p ( z shared t ) and p a ( z domain t ) are both multivariate Gaussian distributions with a learned mean and variance for each dimension. The prior is the same for all timesteps t .

Unlike the actions for the robot data, which are sampled from the same distribution at each time step, the actions of the human are correlated across time. For the human observation data, the prior p o ( z domain 1: T | x 1 ) models a joint distribution over timesteps, and is parameterized as a long short-term memory (LSTM) network [24]. The input to the LSTM at the first timestep is an encoding of the initial observation, and the LSTM cell produces the parameters of the multivariate Gaussian distribution for each time step.

## 3.3. Deep Neural Network Implementation

A high-level diagram of our network architecture is shown in Figure 3, and a more detailed version is presented in Appendix A. Our action encoder q act ( z t | a t ) is a multilayer perceptron with 3 layers of 64 units to encode the given action a t to the means and variances for each dimension of the encoding.

Our inverse model q inv ( z t | x t , x t +1 ) is a convolutional network that predicts the distribution over the action encoding. The network is made up of three convolutional layers with { 32, 64, 128 } features with a kernel size of 4 and a stride of 2. Each convolutional layer is followed by instance normalization and a leaky-ReLU. The output of the final convolutional layer is fed in a fully connected layer, which predicts the means and variances of the action encoding.

We encourage the action encodings generated by the action encoder q act and the inverse model q inv to be similar using the Jensen-Shannon divergence in Equation 7. Since the Jensen-Shannon divergence does not have a closed form solution, we approximate it by using a mean of the Gaussians instead of a mixture. Our model uses a modified version of the SAVP architecture [33] as the transition model which predicts x t +1 from x t and an action encoding z , either sampled from q act ( z t | a t ) or from q inv ( z t | x t , x t +1 ) . In the case where the actions are observed, we generate two predictions, one from each of q inv and q act, and in the case where actions are not observed, we only generate a prediction from the inverse model, q inv. This architecture has been shown to be a useful transition model for robotic planning in [15, 10].

Our action decoder predicts the mean of the distribution p ( a t | z t ) using a multi-layer perceptron with 3 layers of 64 units each, while using a fixed unit variance.

## 4. Experiments

We aim to answer the following in our experiments:

1. Do passive observations, when utilized effectively, improve an action-conditioned visual predictive model despite large domain shifts?
2. How does our approach compare to alternative methods for combining passive and interaction data?
3. Do improvements in the model transfer to downstream tasks, such as robotic control?

To answer 1, we compare our method to a strong actionconditioned prediction baseline, SAVP [33], which is trained only on interaction data as it is not able to leverage the observation data. To answer 2, we further compare to a prior method for inferring actions from action-free data, CLASP [46], and a method that imputes missing actions based on a shared inverse model, described below. We study questions 1 and 2 in both the driving domain in Subsection 4.1 and the robotic manipulation domain in Subsection 4.2 and evaluate the methods on action-conditioned prediction. We evaluate question 3 by directly controlling the robotic manipulator using our learned model. Videos of our results are available on the supplementary website 1 .

Figure 4: Example predictions on the Singapore portion of the Nuscenes dataset. This sequence was selected for large MSE difference between the models. More examples are available in the supplementary material. We compare our model to the baseline of the SAVP model trained on the Boston data with actions. Our model is able to maintain the shape of the car in front.

<!-- image -->

As an additional point of comparison, we propose a shared inverse model that draws similarities to label propagation in semi-supervised learning [75]. In this comparison, an inverse model and transition model are jointly learned on all of the data. Specifically, the inverse model predicts the action taken between a pair of images, supervised only by the actions from the interaction data, and the transition model predicts the next frame conditioned on the current frame and an input action. When available, the transition model uses the true action, otherwise, it uses the action predicted by the inverse model.

## 4.1. Visual Prediction for Driving

We first evaluate our model on video prediction for driving. Imagine that a self-driving car company has data from a fleet of cars with sensors that record both video and the driver's actions in one city, and a second fleet of cars that only record dashboard video, without actions, in a second city. If the goal is to train an action-conditioned model that can be utilized to predict the outcomes of steering actions, our method allows us to train such a model using data from both cities, even though only one of them has actions.

We use the nuScenes [6] and BDD100K [70] datasets for our experiments. The nuScenes dataset consists of 1000 driving sequences collected in either Boston or Singapore, while the BDD100K dataset contains only video from dashboard cameras. In nuScenes, we discard all action and state information for the data collected in Singapore, simulating data that could have been collected by a car equipped with only a camera. We train our model with action-conditioned video from Boston and action-free video either from the nuScenes Singapore data or the BDD100K data, and evaluate on action-conditioned prediction on held-out data from Singapore (from nuScenes). Since the action distribution for all datasets is likely very similar as they all contain human driving, we use the same learned means and variances for the Gaussian prior over z for both portions of the dataset. We additionally train a our model with the actionconditioned video from Boston and action-free video taken from the BDD100K dataset [70].

1 Our supplementary website is at https://sites.google.com/ view/lpmfoai

Table 1: Means and standard errors for action-conditioned prediction on the Singapore portion of the nuScenes dataset. By leveraging observational driving data from Singapore or from BDD dashboard cameras, our method is able to outperform prior models that cannot leverage such data (i.e. SA VP) and slightly outperform alternative approaches to using such data.

| Method                                                          | PSNR ( ↑ )       | SSIM ( ↑ )          | LPIPS [73] ( ↓ )    |
|-----------------------------------------------------------------|------------------|---------------------|---------------------|
| SAVP [33] (Boston w/ actions)                                   | 19 . 74 ± 0 . 41 | 0 . 5121 ± 0 . 0164 | 0 . 1951 ± 0 . 0075 |
| Shared Inverse Model (Boston w/ actions, Singapore w/o actions) | 20 . 65 ± 0 . 52 | 0 . 5455 ± 0 . 0166 | 0 . 2003 ± 0 . 0080 |
| CLASP [46] (Boston w/ actions, Singapore w/o actions)           | 20 . 57 ± 0 . 48 | 0 . 5431 ± 0 . 0161 | 0 . 1964 ± 0 . 0076 |
| POI (ours) (Boston w/ actions, BDD100K w/o actions)             | 20 . 88 ± 0 . 24 | 0 . 5508 ± 0 . 0076 | 0 . 2106 ± 0 . 0089 |
| POI (ours) (Boston w/ actions, Singapore w/o actions)           | 20 . 81 ± 0 . 49 | 0 . 5486 ± 0 . 0164 | 0 . 1933 ± 0 . 0074 |
| Oracle - SAVP [33] (Boston w/ actions. Singapore w/ actions)    | 21 . 17 ± 0 . 47 | 0 . 5752 ± 0 . 0156 | 0 . 1738 ± 0 . 0076 |

We compare our predictions to those generated by the SAVP [33] model trained with only the action-conditioned data from Boston, since SAVP cannot leverage actionfree data for action-conditioned prediction. We additionally compare our predictions to those generated by CLASP [46] and the shared inverse model, both trained with action-conditioned video from Boston, and action-free video from Singapore. As an upper-bound, we train the SAVP [33] model with action-conditioned data from Boston and action-conditioned data from Singapore.

Comparisons between these methods are shown in Table 1. Qualitative results are shown in Figure 4. With either form of observational data, BDD2K or nuScenes Singapore, our method significantly outperforms the SA VP model trained with only action-conditioned data from Boston, demonstrating that our model can leverage observation data to improve the quality of its predictions. Further, our method slightly outperforms alternative approaches to learning from observation and interaction.

## 4.2. Robotic Manipulation: Prediction

We evaluate our model on the robotic manipulation domain, which presents a large distributional shift challenge between robot and human videos. In particular, we study a tool-use task and evaluate whether human videos of tool-use can improve predictions of robotic tool-use interactions.

For our interaction data, we acquired 20,000 random trajectories of a Sawyer robot from the open-source datasets from [15] and [67], which consist of both video and corresponding actions. We then collected 1,000 videos of a human using different tools to push objects as the observation data. By including the human videos, we provide the model with examples of tool-use interactions, which are not available in the random robot data. Our test set is composed of 1,200 kinesthetic demonstrations from [67], in which a human guides the robot to use tools to complete pushing tasks similar to those in the human videos. Kinesthetic demonstrations are time-consuming to collect, encouraging us to build a system that can be trained without them, but they serve as a good proxy for evaluating robot tool-use behavior. Example images from the datasets are shown in Figure 5. 2 This dataset is especially challenging because of the large domain shift between the robot and human data. The human arm has a different appearance from the robot and moves in a different action space.

Figure 5: Example images from the robot (top) and human (bottom) datasets.

<!-- image -->

We compare to the CLASP model [46] and a shared inverse model, both trained with the same data as our model. We also evaluate the SAVP model [33], trained the same robot data, but without the human data, since the SAVP model is unable to leverage action-free data for actionconditioned prediction.

For an oracle, we trained the SAVP model [33] on both the random robot trajectories and the kinesthetic demonstrations.

As shown in Table 2, our model is able to leverage information from the human videos to outperform the other models. Both our model and the shared inverse model outperformed the SAVP model trained on only the random robot data, showing that it is possible to leverage passive observation data to improve action-conditioned prediction, even in the presence of the large domain shift between human and robot arms. Our model also outperformed our shared inverse model, showing the importance of explicitly considering domain shift and stochasticity.

2 All components of our dataset will be released upon publication.

<!-- image -->

Figure 6: Example predictions on the robotic dataset. We compare our model to the baseline of the SA VP model trained with random robot data. This sequence was selected to maximize the MSE difference between the models. More examples are available in the supplementary material. Our model more accurately predicts both the tool and the object it pushes.

Table 2: Means and standard errors for action-conditioned prediction on the manipulation dataset. By leveraging observational data of human tool use, our model was able to outperform prior models that cannot leverage such data (i.e. SAVP) and slightly outperform alternative approaches to using such data.

| Method                                         | PSNR ( ↑ )       | SSIM ( ↑ )        | LPIPS [73] ( ↓ )    |
|------------------------------------------------|------------------|-------------------|---------------------|
| CLASP [46] (random robot, expert human)        | 22 . 14 ± 0 . 11 | 0 . 763 ± 0 . 004 | 0 . 0998 ± 0 . 0023 |
| SAVP [33] (random robot)                       | 23 . 31 ± 0 . 10 | 0 . 803 ± 0 . 004 | 0 . 0757 ± 0 . 0022 |
| Shared IM (random robot, expert human)         | 23 . 59 ± 0 . 10 | 0 . 808 ± 0 . 004 | 0 . 0770 ± 0 . 0022 |
| POI (ours) (random robot, expert human)        | 23 . 79 ± 0 . 12 | 0 . 813 ± 0 . 005 | 0 . 0722 ± 0 . 0024 |
| Oracle [33] (random robot, expert kinesthetic) | 24 . 99 ± 0 . 11 | 0 . 858 ± 0 . 003 | 0 . 0486 ± 0 . 0017 |

Figure 7: Histograms of the x and y components of the actions. Since the human data does not have any actions, the displayed actions were generated by our inverse model. The distribution of predicted human actions of tool-use resembles that of the expert robot actions, suggesting that our model has learned to successfully decode human actions.

<!-- image -->

Qualitative results are shown in Figure 6. Our model is able to generate more accurate predictions than the baseline SAVP model that was trained with only the robotic interaction data. In addition to predicting future states, our model is able to predict the action that occurred between two states. Examples for both robot and human demonstrations are shown in Figure 8. Our inverse model is able to generate reasonable actions for both the robot and the human data despite having never been trained on human data with actions. Histograms of the action distributions for different parts of our dataset are shown in Figure 7. Our model is able to extract actions for the human data from a reasonable distribution. Our model maps human and robot actions to a similar space, allowing it to exploit their similarities to improve prediction performance on robotic tasks.

## 4.3. Robotic Manipulation: Planning and Control

To study the third and final research question, we further evaluate the efficacy of our visual dynamics model in a set of robotic control experiments. We combine our model with sampling-based visual model predictive control, which optimizes actions with respect to a user-provided task [18, 15]. In each task setting, several objects, as well as a tool that the robot could potentially use to complete the task, are placed in the scene. Tasks are specified by designating a pixel corresponding to an object and the goal position for the object, following [18, 15]. We specify moving multiple objects by selecting multiple pairs of pixels.

To evaluate the importance of the human data, we focus on control tasks that involve moving multiple objects, which would be difficult to complete without using a tool. We quantitatively evaluate each model on 15 tasks with tools seen during training and 15 tasks with previously unseen tools. In Figure 9, we show qualitative examples of the robot completing tool-use tasks.

Figure 8: Action predictions on human and robot data. The sequences of images show the ground truth observations, while the arrows show the action in the (x, y) plane between each pair of frames. The blue arrow is the ground truth action, the green arrow is the action generated from decoding the output of the action encoder, and the red is the action generated by decoding the output of the inverse model. The human data only has actions generated by the inverse model. Our model is able to infer plausible actions for both domains, despite never seeing ground truth human actions.

<!-- image -->

Figure 9: Examples of a robot using our model to successfully complete tool use tasks. The robot must move the objects specified by the red symbols to the locations of the corresponding green symbols. The robot uses a tool to simultaneously move several objects to their goal locations.

<!-- image -->

The quantitative results, in Table 3, indicate that the planner can leverage our model to execute more successful plans relative to the baseline SA VP model, which was trained only using random robot trajectories. In our evaluation, a trial is successful if the average distance between the objects and their respective goal positions at the final time step is less than or equal to 10 centimeters. Using our model, the robot achieves similar performance to the oracle model trained on kinesthetic demonstrations with action labels. This result suggests that our model has effectively learned about toolobject interactions by observing humans.

## 5. Conclusion

Wepresent a method for learning predictive models from both passive observation and active interactions. Active interactions are usually more expensive and less readilyavailable than passive observation: for example, consider the amount of observational data of human activities on the internet. Active interaction, on the other hand, is especially difficult when the agent is trying to collect infor- mation about regions of the state-space which are difficult to reach. Without an existing policy that can guide the agent to those regions, time consuming on-policy exploration, expert teleoperated or kinesthetic demonstrations are often required, bringing additional costs.

Table 3: Success rates &amp; standard errors for robotic tasks. 'random' denotes random robot data, 'human' denotes human interaction data, and 'kinesthetic' is an oracle dataset of expert robot trajectories. Our model performs comparably to the oracle, and successfully leverages the observational videos to improve over SAVP.

| Method                       | Success Rate     |
|------------------------------|------------------|
| SAVP [33] (random)           | 23 . 3 ± 7 . 7%  |
| POI (ours) (random, human)   | 40 . 0 ± 8 . 9 % |
| Oracle (random, kinesthetic) | 36 . 7 ± 8 . 8%  |

By learning a latent variable over the semi-observed actions, our approach is able to leverage passive observational data to improve action-conditioned predictive models, even in the presence of domain shift between observation and interaction data. Our experiments illustrate these benefits in two problem settings: driving and object manipulation, and find improvements both in prediction quality and in control performance when using these models for planning.

Overall, we hope that this work represents a first step towards enabling the use of broad, large-scale observational data when learning about the world. However, limitations and open questions remain. Our experiments studied a limited aspect of this broader problem where the observational data was either a different embodiment in the same environment (i.e. humans manipulating objects) or a different environment within the same underlying dataset (i.e. driving in Boston and Singapore). In practice, many source of passive observations will exhibit more substantial domain shift than those considered in this work. Hence, an important consideration for future work is to increase robustness to domain shift to realize greater benefits from using more large and diverse observational datasets. Finally, we focused our study on learning predictive models; an exciting direction for future work is to study how to incorporate similar forms of observational data in representation learning and reinforcement learning.

## Acknowledgements

We would like to thank Kenneth Chaney for technical support. We also like to thank Karl Pertsch and Drew Jaegle for insightful discussions.

This work was supported by the NSF GRFP, ARL RCTA W911NF-10-2-0016, ARL DCIST CRA W911NF-17-20181, and by Honda Research Institute.

## References

- [1] Yusuf Aytar, Tobias Pfaff, David Budden, Thomas Paine, Ziyu Wang, and Nando de Freitas. Playing hard exploration games by watching YouTube. Advances in Neural Information Processing Systems 31 , 2018. 2
- [2] Mohammad Babaeizadeh, Chelsea Finn, Dumitru Erhan, Roy H. Campbell, and Sergey Levine. Stochastic variational video prediction. In Proceedings of International Conference on Learning Representations (ICLR) , 2018. 2
- [3] Konstantinos Bousmalis, Nathan Silberman, David Dohan, Dumitru Erhan, and Dilip Krishnan. Unsupervised pixellevel domain adaptation with generative adversarial networks. Proceedings of the IEEE conference on computer vision and pattern recognition , 2017. 3
- [4] Wonmin Byeon, Qin Wang, Rupesh Kumar Srivastava, and Petros Koumoutsakos. Contextvp: Fully context-aware video prediction. In The European Conference on Computer Vision (ECCV) , September 2018. 2
- [5] Arunkumar Byravan, Felix Leeb, Franziska Meier, and Dieter Fox. Se3-pose-nets: Structured deep dynamics models for visuomotor planning and control. Proceedings of IEEE International Conference on Robotics and Automation , 2017. 2
- [6] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. arXiv preprint arXiv:1903.11027 , 2019. 5
- [7] Lluis Castrejon, Nicolas Ballas, and Aaron Courville. Improved Conditional VRNNs for Video Prediction. arXiv preprint , apr 2019. 2
- [8] Baoyang Chen, Wenmin Wang, Jinzhuo Wang, and Xiongtao Chen. Video Imagination from a Single Image with Transformation Generation. arXiv preprint , jun 2017. 2
- [9] Silvia Chiappa, S´ ebastien Racani` ere, Daan Wierstra, and Shakir Mohamed. Recurrent environment simulators. In Proceedings of International Conference on Learning Representations (ICLR) , 2017. 2
- [10] Sudeep Dasari, Frederik Ebert, Stephen Tian, Suraj Nair, Bernadette Bucher, Karl Schmeckpeper, Siddharth Singh, Sergey Levine, and Chelsea Finn. RoboNet: Large-Scale Multi-Robot Learning. Conference on Robot Learning , oct 2019. 2, 5
- [11] Bert De Brabandere, Xu Jia, Tinne Tuytelaars, and Luc Van Gool. Dynamic Filter Networks. Neural Information Processing Systems , may 2016. 2
- [12] Emily Denton and Vighnesh Birodkar. Unsupervised learning of disentangled representations from video. In Proceedings of Neural Information Processing Systems (NeurIPS) , pages 4417-4426, 2017. 2
- [13] E. Denton and R. Fergus. Stochastic video generation with a learned prior. In Proceedings of International Conference on Machine Learning (ICML) , 2018. 2
- [14] Debidatta Dwibedi, Jonathan Tompson, Corey Lynch, and Pierre Sermanet. Learning actionable representations from visual observations. In 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) , pages 15771584. IEEE, 2018. 2
- [15] Frederik Ebert, Chelsea Finn, Sudeep Dasari, Annie Xie, Alex Lee, and Sergey Levine. Visual foresight: Model-based deep reinforcement learning for vision-based robotic control. arXiv:1812.00568 , 2018. 1, 2, 5, 6, 7
- [16] Ashley D. Edwards, Himanshu Sahni, Yannick Schroecker, and Charles L. Isbell. Imitating Latent Policies from Observation. International Conference on Machine Learning , may 2019. 2
- [17] Chelsea Finn, Ian Goodfellow, and Sergey Levine. Unsupervised learning for physical interaction through video prediction. In Proceedings of Neural Information Processing Systems (NeurIPS) , 2016. 2
- [18] Chelsea Finn and Sergey Levine. Deep visual foresight for planning robot motion. In Proceedings of IEEE International Conference on Robotics and Automation , 2017. 2, 7
- [19] Katerina Fragkiadaki, Pulkit Agrawal, Sergey Levine, and Jitendra Malik. Learning Visual Predictive Models of Physics for Playing Billiards. International Conference on Learning Representations , nov 2016. 2
- [20] Yaroslav Ganin and Victor Lempitsky. Unsupervised Domain Adaptation by Backpropagation. International Conference on Machine Learning (ICML) , 2015. 3
- [21] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, Francois Laviolette, Mario Marchand, and Victor Lempitsky. Domain-adversarial training of neural net-works. The Journal of Machine Learning Research , 2016. 3
- [22] David Ha and J¨ urgen Schmidhuber. Recurrent world models facilitate policy evolution. In Proceedings of Neural Information Processing Systems (NeurIPS) . 2018. 1, 2
- [23] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. Proceedings of International Conference on Machine Learning (ICML) , 2019. 1, 2
- [24] Sepp Hochreiter and J¨ urgen Schmidhuber. Long short-term memory. Neural computation , 9(8):1735-1780, 1997. 4
- [25] Judy Hoffman, Eric Tzeng, Taesung Park, Jun-Yan Zhu, Phillip Isola, Kate Saenko, Alexei A. Efros, and Trevor Darrell. CyCADA: Cycle-Consistent Adversarial Domain Adaptation. International Conference on Machine Learning (ICML) , nov 2018. 3
- [26] Maximilian Ilse, Jakub M. Tomczak, Christos Louizos, and Max Welling. DIVA: Domain Invariant Variational Autoencoders. arXiv preprint , may 2019. 3, 4
- [27] Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. When to trust your model: Model-based policy optimization. NeurIPS , 2019. 1
- [28] Michael Janner, Sergey Levine, William T. Freeman, Joshua B. Tenenbaum, Chelsea Finn, and Jiajun Wu. Reasoning About Physical Interactions with Object-Oriented Prediction and Planning. International Conference on Learning Representations , dec 2019. 2
- [29] Lukasz Kaiser, Mohammad Babaeizadeh, Piotr Milos, Blazej Osinski, Roy H Campbell, Konrad Czechowski, Dumitru Erhan, Chelsea Finn, Piotr Kozakowski, Sergey Levine, Ryan Sepassi, George Tucker, and Henryk Michalewski. Model-based reinforcement learning for atari, 2019. 2
- [30] Nal Kalchbrenner, Aaron van den Oord, Karen Simonyan, Ivo Danihelka, Oriol Vinyals, Alex Graves, and Koray Kavukcuoglu. Video Pixel Networks. arXiv preprint , oct 2016. 2
- [31] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. International Conference on Learning Representations , dec 2015. 12
- [32] Ashish Kumar, Saurabh Gupta, and Jitendra Malik. Learning navigation subroutines by watching videos. CoRR , abs/1905.12612, 2019. 2
- [33] A. X. Lee, R. Zhang, F. Ebert, P. Abbeel, C. Finn, and S. Levine. Stochastic adversarial video prediction. arXiv:1804.01523 , abs/1804.01523, 2018. 2, 5, 6, 7, 8, 12
- [34] Xiaodan Liang, Lisa Lee, Wei Dai, and Eric P. Xing. Dual Motion GAN for Future-Flow Embedded Video Prediction. International Conference on Computer Vision , aug 2017. 2
- [35] YuXuan Liu, Abhishek Gupta, Pieter Abbeel, and Sergey Levine. Imitation from Observation: Learning to Imitate Behaviors from Raw Video via Context Translation . PhD thesis, University of California, Berkeley, jul 2018. 2
- [36] Ziwei Liu, Raymond A Yeh, Xiaoou Tang, Yiming Liu, and Aseem Agarwala. Video frame synthesis using deep voxel flow. In Proceedings of the IEEE International Conference on Computer Vision , pages 4463-4471, 2017. 2
- [37] William Lotter, Gabriel Kreiman, and David Cox. Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning. arXiv preprint , may 2016. 2
- [38] Chaochao Lu, Michael Hirsch, and Bernhard Scholkoph. Computer Vision and Pattern Recognition , 2017. 2
- [39] Pauline Luc, Natalia Neverova, Camille Couprie, Jakob Verbeek, and Yann LeCun. Predicting Deeper into the Future of Semantic Segmentation. International Conference on Computer Vision , mar 2017. 2
- [40] M. Mathieu, C. Couprie, and Y. LeCun. Deep multi-scale video prediction beyond mean square error. In Proceedings of International Conference on Learning Representations (ICLR) , 2016. 2
- [41] Junhyuk Oh, Xiaoxiao Guo, Honglak Lee, Richard Lewis, and Satinder Singh. Action-conditional video prediction using deep networks in atari games. In Proceedings of Neural Information Processing Systems (NeurIPS) , 2015. 2
- [42] Viorica Patraucean, Ankur Handa, and Roberto Cipolla. Spatio-temporal video autoencoder with differentiable memory. arXiv preprint , nov 2015. 2
- [43] MarcAurelio Ranzato, Arthur Szlam, Joan Bruna, Michael Mathieu, Ronan Collobert, and Sumit Chopra. Video (language) modeling: a baseline for generative models of natural videos. arXiv preprint arXiv:1412.6604 , 2014. 2
- [44] Giacomo Rizzolatti and Laila Craighero. The mirror-neuron system. Annu. Rev. Neurosci. , 27:169-192, 2004. 1
- [45] Giacomo Rizzolatti, Luciano Fadiga, Vittorio Gallese, and Leonardo Fogassi. Premotor cortex and the recognition of motor actions. Cognitive Brain Research , 3(2), 1996. 1
- [46] Oleh Rybkin, Karl Pertsch, Konstantinos G. Derpanis, Kostas Daniilidis, and Andrew Jaegle. Learning what you can do before doing anything. In International Conference on Learning Representations , 2019. 2, 5, 6, 7
- [47] Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, and Sergey Levine. Timecontrastive networks: Self-supervised learning from video. Proceedings of International Conference in Robotics and Automation (ICRA) . 2
- [48] N. Srivastava, E. Mansimov, and R. Salakhudinov. Unsupervised learning of video representations using LSTMs. In Proceedings of International Conference on Machine Learning (ICML) , 2015. 2
- [49] Bradly C Stadie, Pieter Abbeel, and Ilya Sutskever. Thirdperson imitation learning. arXiv preprint arXiv:1703.01703 , 2017. 2
- [50] Mingfei Sun and Xiaojuan Ma. Adversarial Imitation Learning from Incomplete Demonstrations. International Joint Conference on Artificial Intelligence , may 2019. 2
- [51] Wen Sun, Anirudh Vemula, Byron Boots, and J. Andrew Bagnell. Provably Efficient Imitation Learning from Observation Alone. International Conference on Machine Learning , may 2019. 2
- [52] Yaniv Taigman, Adam Polyak, and Lior Wolf. Unsupervised Cross-Domain Image Generation. International Conference on Learning Representations , nov 2017. 3
- [53] Faraz Torabi, Garrett Warnell, and Peter Stone. Behavioral Cloning from Observation. International Joint Conference on Artificial Intelligence , may 2018. 2
- [54] Faraz Torabi, Garrett Warnell, and Peter Stone. Generative Adversarial Imitation from Observation. arXiv preprint , jul 2018. 2
- [55] Faraz Torabi, Garrett Warnell, and Peter Stone. Imitation Learning from Video by Leveraging Proprioception. International Joint Conference on Artificial Intelligence , may 2019. 2
- [56] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. MoCoGAN: Decomposing motion and content for video generation. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018. 2
- [57] Eric Tzeng, Judy Hoffman, Kate Saenko, and Trevor Darrell. Adversarial Discriminative Domain Adaptation. Computer Vision and Pattern Recognition , feb 2017. 3
- [58] Eric Tzeng, Judy Hoffman, Ning Zhang, Kate Saenko, and Trevor Darrell. Deep Domain Confusion: Maximizing for Domain Invariance. arXiv preprint , dec 2014. 3
- [59] Joost van Amersfoort, Anitha Kannan, Marc'Aurelio Ranzato, Arthur Szlam, Du Tran, and Soumith Chintala. Transformation-Based Models of Video Sequences. arXiv preprint , jan 2017. 2
- [60] Ruben Villegas, Jimei Yang, Seunghoon Hong, Xunyu Lin, and Honglak Lee. Decomposing motion and content for natural video sequence prediction. In Proceedings of International Conference on Learning Representations (ICLR) , 2017. 2
- [61] Carl Vondrick, Hamed Pirsiavash, and Antonio Torralba. Anticipating visual representations from unlabeled video. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2016. 2
- [62] Carl Vondrick and Antonio Torralba. Generating the future with adversarial transformers. Conference on Vision and Pattern Recognition , 2017. 2
- [63] Jacob Walker, Carl Doersch, Abhinav Gupta, and Martial Hebert. An Uncertain Future: Forecasting from Static Images using Variational Autoencoders. European Conference on Computer Vision , jun 2016. 2
- [64] Angelina Wang, Thanard Kurutach, Aviv Tamar, and Pieter Abbeel. Learning Robotic Manipulation through Visual Planning and Acting. Robotics: Science and Systems , 2019. 2
- [65] Manuel Watter, Jost Tobias Springenberg, Joschka Boedecker, and Martin Riedmiller. Embed to control: A locally linear latent dynamics model for control from raw images. In Proceedings of Neural Information Processing Systems (NeurIPS) , 2015. 2
- [66] Nevan Wichers, Ruben Villegas, Dumitru Erhan, and Honglak Lee. Hierarchical long-term video prediction without supervision. ICML , 2018. 2
- [67] Annie Xie, Frederik Ebert, Sergey Levine, and Chelsea Finn. Improvisation through Physical Understanding: Using Novel Objects as Tools with Visual Foresight. Robotics: Science and Systems , apr 2019. 1, 6
- [68] Tianfan Xue, Jiajun Wu, Katherine L. Bouman, and William T. Freeman. Visual Dynamics: Probabilistic Future Frame Synthesis via Cross Convolutional Networks. IEEE Transactions on Pattern Analysis and Machine Intelligence , jul 2016. 2
- [69] Lin Yen-Chen, Maria Bauza, and Phillip Isola. ExperienceEmbedded Visual Foresight. Conference on Robot Learning , nov 2019. 2
- [70] Fisher Yu, Wenqi Xian, Yingying Chen, Fangchen Liu, Mike Liao, Vashisht Madhavan, and Trevor Darrell. BDD100K: A Diverse Driving Video Database with Scalable Annotation Tooling. arXiv preprint , may 2018. 5, 6
- [71] Tianhe Yu, Chelsea Finn, Annie Xie, Sudeep Dasari, Tianhao Zhang, Pieter Abbeel, and Sergey Levine. One-Shot Imitation from Observing Humans via Domain-Adaptive MetaLearning. Robotics: Science and Systems , feb 2018. 2
- [72] Marvin Zhang, Sharad Vikram, Laura Smith, Pieter Abbeel, Matthew J. Johnson, and Sergey Levine. SOLAR: Deep Structured Representations for Model-Based Reinforcement Learning. International Conference on Machine Learning , aug 2018. 2
- [73] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR , 2018. 6, 7
- [74] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycleconsistent adversarial networks. In Computer Vision (ICCV), 2017 IEEE International Conference on , 2017. 3
- [75] Xiaojin Zhu and Zoubin Ghahramani. Learning from labeled and unlabeled data with label propagation. 2002. 5
- [76] Fuzhen Zhuang, Xiaohu Cheng, Ping Luo, Sinno Jialin Pan, and Qing He. Supervised representation learning with double encoding-layer autoencoder for transfer learning. International Joint Conference on Artifical Intelligence , 2015. 3

## A. Full Architecture

The full architecture of our model is presented in Figure 10.

## B. Model Hyperparameters

We selected our hyperparameters through crossvalidation. The hyperparameters that are shared between the domains are described in Table 4. The hyperparameters that are specific to the robotic manipulation domain are described in Table 5. The hyperparameters that are specific to the driving domain are described in Table 6.

## C. Robot Planning and Control Experiments

For the control experiments, each task is set up by placing one potential tool into the scene, as well as 2-3 objects to relocate which are specified to the planner by selecting start and goal pixels. The scenes are set up so that because the robot needs to move multiple objects, it is most effective for it to use the tool during its execution. We present the hyperparameters used for the planner in our robotic control experiments in Table 7.

| Hyperparameter                   | Value     |
|----------------------------------|-----------|
| Action decoder MSE weight        | 0.0001    |
| Action encoder KL weight         | 10 - 6    |
| Jensen-Shannon Divergence weight | 10 - 7    |
| TV weight                        | 0.001     |
| Image L1 reconstruction weight   | 1.0       |
| Optimizer                        | Adam [31] |
| Learning rate                    | 0.001     |
| Beta1                            | 0.9       |
| Beta2                            | 0.999     |
| Schedule sampling k              | 900       |
| Action encoder channels          | 64        |
| Action encoder layers            | 3         |
| Inverse model channels           | 64        |
| Inverse model layers             | 3         |
| Generator channels               | 32        |

Table 4: Hyperparameter values

| Hyperparameter             |   Value |
|----------------------------|---------|
| Dimensionality of z domain |       2 |
| Dimensionality of z shared |       3 |
| Prediction horizon         |      15 |

Table 5: Hyperparameter values specific to the robotic manipulation domain

Table 6: Hyperparameter values specific to the driving domain

| Hyperparameter             |   Value |
|----------------------------|---------|
| Dimensionality of z domain |       0 |
| Dimensionality of z shared |       3 |
| Prediction horizon         |       5 |

Table 7: Hyperparameter values specific to the robot control experiments

| Hyperparameter                      | Value                |
|-------------------------------------|----------------------|
| Robot actions per trajectory        | 20                   |
| Unique robot actions per trajectory | 6 (each repeated x3) |
| CEM iterations                      | 4                    |
| CEM candidate actions per iteration | 1200                 |
| CEM selection fraction              | 0.05                 |
| Prediction horizon                  | 18                   |
| Number of goal-designating pixels   | 3                    |

## D. Additional Implementation Details

Additional implementation details are presented in this section.

## D.1. Batch Construction

We constructed our batches so that they were made up of a fixed number of examples from each dataset. In all of our experiments, we used a batch size of 12, made up of 9 samples from the interaction data and 3 samples from the observation data.

## D.2. Schedule Sampling

In order to improve training, our system initially predicts images from the ground truth previous image. As training continues, the system gradually shifts to using the predicted version of the previous frame.

The probability of sampling an image from the ground truth sequence is given by Equation 9.

<!-- formula-not-decoded -->

The iteration number is i, while k is a hyperparameter that controls how many iterations it takes for the system to go from always using the ground truth images to always using the predicted images. This sampling strategy was taken from [33].

## E. Domain Shift

Our method of handling domain shift between datasets, described in Section 3.2, is shown in Figure 11.

Figure 10: Our full architecture for learning from observation and interaction data. Our model is composed of the action encoder, inverse model, action decoder, and transition model. The action encoder and inverse model output distributions over z t conditioned on the action a t and image pair x t

<!-- image -->

Figure 11: The partitioned latent space. We partition our latent space z into two components, z shared , which captures the parts of the latent action that are shared between domains, and z domain , which captures the unique parts of the latent action. We enforce this separation by learning the same prior for z shared in all domains and a different prior for z domain in each domain.

<!-- image -->

## F. Action Visualization

We visualize the histogram of the robot actions in Figure 7. All near-zero actions were removed from the histogram of the expert robot data to remove the long periods of time where the robot is stationary in that dataset.

## G. Additional Qualitative Results

Additional qualitative results are presented in this section.

## G.1. Video Prediction in the Driving Domain

Aversion of Figure 4 with more images is shown in Figure 12. We also present the sequence that is best for the baseline in Figure 13. We present the sequence that has the median difference between methods in Figure 14.

## G.2. Video Prediction in the Robotic Manipulation Domain

Aversion of Figure 6 with more images is shown in Figure 15. We also present the sequence that is best for the baseline in Figure 16. We present the sequence that has the median difference between methods in Figure 17.

<!-- image -->

Figure 12: Example predictions on the Singapore portion of the Nuscenes dataset. This sequence was selected for large MSE difference between the models. We compare our model to the baseline of the SAVP model trained on the Boston data with actions. Our model is able to maintain the shape of the car in front.

Figure 13: Example predictions on the Singapore portion of the Nuscenes dataset. This sequence was selected because the baseline had the largest improvement in MSE relative to our model. We compare our model to the baseline of the SAVP model trained on the Boston data with actions. Even in the worse case, our model performs comparably to the baseline model.

<!-- image -->

Figure 14: Example predictions on the Singapore portion of the Nuscenes dataset. This sequence was selected by ordering all of the sequences in the training set by the difference in MSE between the baseline and our model and selecting the middle sequence. We compare our model to the baseline of the SAVP model trained on the Boston data with actions. Even in the worse case, our model performs comparably to the baseline model.

<!-- image -->

Figure 15: Example predictions on the robotic dataset. The first image is the context image. We compare our model to the baseline of the SAVP model trained with random robot data. This sequence was selected to maximize the MSE difference between the models. Our model more accurately predicts both the tool and the object it pushes.

<!-- image -->

Figure 16: Example predictions on the robotic dataset. The first image is the context image. We compare our model to the baseline of the SAVP model trained with random robot data. This sequence was selected to maximize so that the baseline had the largest improvement in MSE relative to our model. Our model fails because it was too pessimistic about grasping the narrow handle of the brush.

<!-- image -->

Figure 17: Example predictions on the robotic dataset. The first image is the context image. We compare our model to the baseline of the SAVP model trained with random robot data. This sequence had the median difference in MSE between our model and the baseline.

<!-- image -->