## Multiscale Sensor Fusion and Continuous Control with Neural CDEs

Sumeet Singh 1 , Francis McCann Ramirez 2 , Jacob V arley 1 , Andy Zeng 1 , and Vikas Sindhwani 1

Abstract -Though robot learning is often formulated in terms of discrete-time Markov decision processes (MDPs), physical robots require near-continuous multiscale feedback control. Machines operate on multiple asynchronous sensing modalities, each with different frequencies, e.g., video frames at 30Hz, proprioceptive state at 100Hz, force-torque data at 500Hz, etc. While the classic approach is to batch observations into fixed-time windows then pass them through feed-forward encoders (e.g., with deep networks), we show that there exists a more elegant approach -one that treats policy learning as modeling latent state dynamics in continuous-time. Specifically, we present InFuser , a unified architecture that trains continuous time-policies with Neural Controlled Differential Equations (CDEs). InFuser evolves a single latent state representation over time by (In)tegrating and (Fus)ing multi-sensory observations (arriving at different frequencies), and inferring actions in continuous-time. This enables policies that can react to multi-frequency multi-sensory feedback for truly end-to-end visuomotor control, without discrete-time assumptions. Behavior cloning experiments demonstrate that InFuser learns robust policies for dynamic tasks (e.g., swinging a ball into a cup) notably outperforming several baselines in settings where observations from one sensing modality can arrive at much sparser intervals than others.

## I. INTRODUCTION

The task of registering readings from multiple asynchronous sensing modalities (e.g., 30Hz video frames, 100Hz proprioceptive state, 500Hz force-torque data), fusing them as soon they arrive to form contact-rich representations of the environment, and integrating them over time, is key to enabling robots to adapt swiftly and precisely to the dynamics of unstructured environments. This is particularly important for agile behaviors where feedback control needs to be nearly continuous-time.

In this work, we develop a novel policy architecture grounded in the formalism of Neural Controlled Differential Equations (CDEs) [1]. CDEs are a generalization of Neural ODEs[2], whereby one evolves a latent state against a process as opposed to only time. As depicted in Figure 1, we leverage a Neural CDE model that is conditioned on images and driven by the set of sensing modalities arriving at a higher frequency than the images. Our method yields policies that can react to multi-frequency multi-sensory feedback for truly end-to-end visuomotor control, without any discrete-time assumptions or having to pre-train a family of (feedback) primitives.

1 Robotics at Google, NYC; Corresponding email: ssumeet@google.com

2 Google AI Resident, NYC

Fig. 1: InFuser uses Neural CDEs to learn quasi continuous-time policies that can react to multi-frequency multi-sensory feedback (e.g., low-frequency images, high-frequency proprioceptive force-torque data), and integrates temporal information over observations to infer high rate actions.

<!-- image -->

Related Work : Multi-sensory fusion with deep networks have shown promising results in learning cross-modal representations for discriminative and generative tasks [3]-[12]. These architectures typically consist of independent modality-specific encoders, for instance, convolutional neural networks (CNNs) for transient inputs such as images and depth perception, recurrent neural networks (RNNs) for temporally correlated inputs such as IMU sensors and audio inputs, and more recently, transformers for natural language. Embeddings generated from these independent encoders are fused into a single multisensory latent representation that is then used for downstream tasks. Training these networks has been addressed via both end-to-end methods (i.e., guided by the downstream task objective) [6]-[9], [12] and pre-training techniques [3], [4], [10], [11], [13], the latter motivated from the perspective of representation learning . In the context of robot control however, such architectures implicitly assume a discrete-time MDP, by batching past observations with fixed time windows.

Imbuing learnable policies with dynamical systems-based structures has also been a critical area of research within robot control, with Dynamic Movement Primitives (DMPs) being one of the most predominant methodologies; see

[14] for an extensive review. DMPs fall within the general class of stable dynamical systems-based imitation learning, whereby one fits nonlinear dynamical systems to observed state trajectories provided by an expert, with the constraint that the dynamics satisfy certain stability criteria, e.g., in the sense of Lyapunov [15]-[17], or contraction [18]-[20]. DMPs have been extended to incorporate sensor feedback [21], [22], and used within hierarchical reinforcement learning within an options framework [23]-[26]. Recently, they have been merged with deep learning whereby the parameters governing the DMPs are generated by an encoder processing higher-dimensional observations (e.g., images) [27]-[29]. We note however that while all these methods enable reasoning over the space of trajectories, they have only been demonstrated on datasets with single stream modalities

There has been recent work that jointly addresses multi-sensory fusion and dynamical systems. Motivated from the perspective of decomposing manipulation tasks into a sequence of modality-specific phases, [30] learns a family of DMP policies utilizing various subsets of different sensing modalities (RGB, force-torque, proximity images), where the primitive forcing functions are modulated directly by the sensor feedback. An overall 'blending' policy network is trained to smoothly combine all the DMP policies to produce the overall control action. [31] leverages a multi-sensory encoder (processing exteroceptive and proprioceptive inputs) to update the parameters of a periodic trajectory generator governing legged locomotion.

## Statement of Contributions

Our key observation is that while multi-sensory fusion and dynamical systems-based policies have been predominantly studied in isolation, they are two tightly intertwined objectives that can be modeled jointly with a single continuous-time architecture. Leveraging the universality of CDEs in representing functions over irregularly sampled time-series [1], we construct a hybrid continuous-time model that uses multifrequency observations to drive the evolution of a continuoustime multi-sensory latent embedding. This model is trained using the adjoint-based backpropagation method associated with Neural ODEs to generate continuous-time action functions . While the generality of the model enables training with either reinforcement learning or imitation, in this work we focus on the latter, in particular, behavior cloning. We train the CDEbased model and appropriate baselines on expert data generated from increasingly challenging (simulated) environments, and quantify the learned policies' performance under various deployment settings such as sensor throttling and missed packets. The experiments demonstrate that on quasi-static tasks such as cloth manipulation, the CDE-based architecture is competitive with existing state-of-the-art multi-sensory fusion policies. However, with increasing task dynamism and sparser image rates, the CDE-based architecture outperforms explicit models that do not feature any temporal abstraction as well as recent state-of-the-art deep DMP-based policies [28].

## II. POLICY LEARNING

Policy learning can be formulated in a variety of ways, and typically assumes the discrete-time MDP setting: where observations o t ∈ O at discrete time-index t ∈ N are mapped to actions a t ∈ A , by a feedback policy π θ : O → A , where θ are learnable parameters (e.g., of a neural network). Reinforcement learning (RL) methods seek an optimal closed-loop policy by maximizing the expected sum of discounted rewards, where the robot applies π θ recursively through robot-environment transition dynamics (the distribution of which can also be learned) collecting state-wise rewards. Meanwhile, imitation learning (IL) via behavioral cloning formulates policy learning as an instance of supervised learning on expert demonstrations. Given observation-action pairs, this entails solving a regularized loss minimization problem of the standard form,

<!-- formula-not-decoded -->

where l ( · , · ) is imitation loss and Ω is a regularizer. Behavioral cloning can be effective [32], [33] when prediction errors do not compound significantly over time or are mitigated using DAGGER-like techniques [34].

## A. Hybrid Continuous-Time Policies

In this work, we study the hybrid continuous-time (HCT) setting, structured as follows. Suppose that we observe image(s) s t ∈ R H × W × C at every discrete time index t ∈ N , and in-between time indices t and t +1 , we have access to continuous-time (this will be straightforwardly relaxed to 'higher-frequency') observations from other sensing modalities, denoted by the function x t ( · ) : τ ∈ [0 ,T ] ↦→ R n . The constant T is a user-set parameter that models the length of time in-between successive images.

An HCT policy is defined as a functional map from the observation tuple o t := ( s t ,x t ( · )) to a control function u t ( · ): τ ∈ [0 ,T ] ↦→U , where U is the control space. Therefore, in 'MDP-notation', our action a t , mapped from o t , is the control function u t ( · ) 1 , drawing natural analogies with hierarchical reinforcement learning.

Remark 1. Although we have described the observation structure as a nesting of higher-frequency observations x t ( · ) in-between low-frequency observations { s t ,s t +1 } , one should really interpret the two signals as separate asynchronous sensor streams observed at differing frequencies.

1 For brevity, for any given function y t ( · ) with domain [0 ,T ] , we use the notation y τ t to denote y t ( τ ) .

To ensure tractability of the functional map π θ : o t =( s t ,x t ( · )) ↦→ u t ( · ) , we impose two conditions. First, we constrain the functions u t ( · ) to be at-least piecewise C 1 over the interval [0 ,T ] . Note that this does not prevent non-smooth transitions when a new image is observed at discrete time index t +1 . This is a justified assumption for robotics applications where the image update frequency is typically at least 2 Hz, and a smooth control command is required at the (higher) control frequency.

Our second assumption states that the control function u t ( · ) is causal w.r.t. the observation o t =( s t ,x t ( · )) ; a natural assumption given our goal of deploying a real-time controller. Formally, let S t represent the random variable for images s t sampled at discrete time index t , and X τ t ,τ ∈ [0 ,T ] represent the continuous-time stochastic process for state functions x t ( · ) sampled in between discrete time indices t and t +1 . Causality dictates that the stochastic process U τ t ,τ ∈ [0 ,T ] with sample realization u t ( · ) is adapted to the filtration generated by the random variable S t and the natural filtration [35] of X τ t .

An elegant modeling framework compatible with these assumptions is where u t ( · ) is computed as the solution to a neural controlled differential equation, introduced next.

## B. Neural Controlled Differential Equations

For self-containment, we provide a brief introduction to CDEs. For further detail we refer the reader to [1], [36]. Let y : [0 , T ] → R n be a continuous function with bounded variation, and let f : R d → R d × n be a matrix-valued continuous function. We can then define a continuous path z :[0 ,T ] → R d as the solution to the following integral:

<!-- formula-not-decoded -->

This is a Riemann-Stieltjes integral; global existence and uniqueness only requires weak regularity conditions on f , namely Lipschitz continuity [36]. In differential notation, the CDE in (1) is written as:

<!-- formula-not-decoded -->

where we remark the similarity with stochastic differential equations notation, owing to the natural analogy between the driving process , i.e., y , and the driven process , i.e., z .

As our use of CDEs will be in the context of processing incoming sensor streams, i.e., sequences of time-stamped data x t := { (0 ,x t (0)) , ( τ i ,x t ( τ i )) ,..., ( T,x t ( T )) } , we use the following time-series adaptation of CDEs. Let ˆ x t :[0 ,T ] → R n be the natural spline interpolant of the values { x t ( τ i ) } i , and define the map ¯ x t : τ ∈ [0 ,T ] ↦→ (ˆ x t ( τ ) ,τ ) ∈ R n +1 . Let f θ : R d → R d × ( n +1) be any parameterized matrix-valued continuous function. Consider now the following CDE:

<!-- formula-not-decoded -->

where the initial condition z 0 t is a separately parameterized explicit function of x 0 t . The solution to equation (3) is termed a Neural CDE model.

Remark 2. As ¯ x t ( · ) is differentiable in our setting (a natural cubic spline), the CDE is converted into the following ODE:

<!-- formula-not-decoded -->

and integrated using an off-the-shelf multi-step integrator . Gradient backpropagation may then be performed using the established adjoint sensitivity technique [2].

Despite sharing the same adjoint-based training technique, Neural CDEs are strictly more general than Neural ODEs. To see this more clearly, consider augmenting z τ t to include ¯ x τ t and setting the relevant sub-matrix of f θ as the identity. Then, one would recover Neural ODE representations of the form:

<!-- formula-not-decoded -->

In general, Neural CDEs constitute an elegant modeling framework for defining 'universal' mappings over the space of time-series x t [1][Theorem B.14], and may be seen as a generalization of the continuous-time limit of RNNs. In our framework, we will leverage neural CDEs to evolve a multi-sensory latent embedding, driven by the higher-frequency observation stream x t .

## III. DESIGNING HCT POLICIES WITH CDES

To introduce our CDE-based model, it is helpful to begin with a baseline architecture that more closely resembles multisensory fusion models seen in the literature. We first make precise the nature of the higher-frequency observations x t . In particular, we assume that in between images s t and s t +1 , we collect a sequence of time-stamped higher-frequency sensor measurements x t := { (0 ,x t (0)) , ( τ i ,x t ( τ i )) ,..., ( T,x t ( T )) } , where τ = 0 corresponds to discrete index t and τ = T corresponds to index t +1 .

Consider Figure 2 (top), which illustrates our first baseline model, and the following accompanying equations, which we hereby term SB (stale baseline):

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here, F s is an image-only encoder mapping the image observation s t to a latent image embedding z s t , that is subsequently held fixed for all τ ∈ [0 ,T ] . For each τ ∈ [0 ,T ] , we feed z s t and the concurrent (interpolated) observation ˆ x t ( τ ) into a fusion network F o , to produce a multi-sensor latent embedding z o t ( τ ) . This embedding is then decoded via F u to produce the control value u t ( τ ) . Note that such a baseline model encapsulates the most common adaptation of multi-sensory policies for robot control, and serves as a useful starting point to introduce the CDE variation.

Fig. 2: In the figures above a line with an arrow at the end denotes a single value whereas the striped 2D arrows denote a functional map. Top : SB baseline model; Bottom : InFuser (proposed). Note that the main difference between the two architectures lies in the middle of the Decoder block. InFuser evolves the latent vector z o t from its initial value z o t (0) , generated in identical fashion to SB , via a neural CDE that is driven by the incoming higher-frequency observations x t ( · ) . In contrast, the SB model updates the latent vector z o t in a stateless manner. Both models decode this latent vector at any intermediate time τ into the control action using identical MLP structures.

<!-- image -->

The key difference between SB and the CDE-based model is how the latent embedding z o t ( · ) is evolved. Consider Figure 2 (bottom), illustrating a CDE-based architecture, and the following accompanying equations, hereby termed the InFuser model:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Notice that the initial action u t (0) is generated in the same way as SB , as this is fundamentally constrained by the available observations at that instant, i.e., ( s t , x t (0)) . For τ &gt; 0 however, InFuser evolves the multi-sensor latent embedding z o t ( · ) via a neural CDE that is conditioned on the seen image (via z s t ), and driven by the incoming stream of higher-frequency observations x t ( · ) . Decoding this embedding at any intermediate τ into a control action is done via the same decoder structure as SB , i.e., via F u .

Some remarks are in order. The SB model may be seen as an explicit map between the incoming observations and the control action, in that it processes the observations in a stateless (i.e., non-temporal) manner. In the experiments, we will present two other baseline models that do incorporate some notion of state and temporal reasoning. InFuser however may be seen as an implicit map that better captures the functional nature of HCT policies, by explicitly reasoning about the temporal structure of the multi-sensory observations. Such a representation has two key consequences. First, by the universality of the CDE representation (see [1]), such a model subsumes ODE-based parameterizations of the form:

<!-- formula-not-decoded -->

and therefore, generalizes continuous-time RNNs and deep DMP-based models referenced at the beginning of this work. Second, by working directly within the functional space, i.e., by seeking maps from ( s t ,x t ) to the control function u t , we avoid having to force discrete-time latent prediction models such as RNNs to handle multiple asynchronous and irregular time-series, e.g., by alternating Neural ODE solves and RNN jumps [37]. Third, the CDE itself may be interpreted as capturing the higher-frequency/higher-fidelity environment dynamics taking place in-between successive images, thereby imbuing the policy network with a model-based inductive bias that is crucial for reasoning over the space of trajectories.

## A. Implementation and Imitation Learning

Notice that the InFuser model in (5) requires integrating against a C 2 interpolant of the raw time-series x t . While natural cubic splines fulfill this criteria, they induce a noncausal dependence on the time-series x t since the coefficients of the spline depend upon the entire set x t . Following [38], we instead implement the 'discretely-online' strategy whereby we first assume that control is output at a pre-determined frequency that is a multiple of the image-arrival frequency. Thus, for τ ∈ [0 ,T ] between discrete time-steps { t,t +1 } , we are required to output T actions at times { τ i } T -1 i =0 , where without loss of generality, we take τ i +1 -τ i =1 and τ 0 =0 . The CDE model in (5) is then evolved in intervals of 1 , where for generating the control action at time τ i , i &gt; 0 , we use the subset of measurements x t lying in-between [ τ i -1 , τ i ] to form our cubic spline. Thus, the model remains causal at the discrete control frequency. If truly continuous-time causal control is required, one may evolve the CDE using a zero-order-hold interpolant of the measurements x t .

While either RL or IL may be used to train HCT policies, in this work, we focus on the imitation case -specifically, behavior cloning (BC). To this end, we assume the expert's dataset is a collection of observation-action tuples { ( o t , u t ) } t ∼P e , where o t is the tuple ( s t , x t ) , and u t := { u t (0) ,u t (1) ,...,u t ( T -1) } are the sampled expert actions. Let θ represent the set of all trainable parameters (across the encoder, decoder, and neural CDE), and let u t,θ ( ·| o t ) : [0 ,T ) →U denote the control function generated by the HCT policy, as a function of the observations o t . BC training is performed by minimizing the objective:

<!-- formula-not-decoded -->

where l ( · , · ) : U×U → R ≥ 0 is a non-negative cost function penalizing the difference between the predicted and actual control output, and ˆ u t ( · ) is a piecewise C 1 interpolant of the sampled signal u t . We remark that since an HCT policy models control functions , the loss is written as an integral rather than a sum over the finite set of observed control values.

## IV. EXPERIMENT SETUP AND HYPOTHESES

We performed several variations of BC experiments on data collected by expert policies (both scripted and pre-trained with privileged obeservations) on a variety of environments. In this section we outline the environment and tasks, types of experiments, and comparative baseline models.

## A. Environments

Figure 3 depicts the two evaluation environments: Cloth-Covering (CC): where the objective is to pick-up and drape a towel over an object (the locations of both items are randomized upon initialization), and Ball-In-Cup

(BiC): consisting of an actuated cup in the vertical plane attempting to swing and catch a ball attached via an elastic cable [39]. In addition to the incoming image stream s t , the higher-frequency observations x t correspond to the robot proprioceptive sensors for the CC environment and cup position and velocity for BiC. By denying explicit observation of the ball position, we are forcing the policy to implicitly deduce and forward predict this information from the image stream.

Fig. 3: Example Timelapse for evaluation environments; Left : Ball-In-Cup, Right : Cloth Covering.

<!-- image -->

We also study a harder variant of the BiC environment variable BiC (vBiC), where the higher-frequency observation is reduced to a 3-dof force-torque sensor collocated at the cup position, and the ball mass and maximum length of the string are randomized at the beginning of each episode to lie within a ± 1 / 3 range of their nominal values. We study this variation to examine the effect the nature of the higher-frequency observations may have on the learnability of the policy. Please see Appendix A 2 for additional details on the environments.

## B. BC Evaluations: Robustness to Latency and Drops

For each environment, we specified a nominal value of T (the time in-between successive image experiments), denoted as T 0 , and performed BC training and evaluation as outlined in Section III-A. In addition, we introduce two new variations custom to the HCT control setting: (i) Dropped-BC , and (ii) Throttled-BC . Dropped-BC emulates dropped image sensor observations, whereby the policy is forced to rely upon the last image saved in memory if the expected incoming image at the current time index t is lost. We quantize the difficulty of this BC variation via a Bernoulli drop probability p d ∈ [0 , 1) , specifying the probability of a missed image at any time index t . To perform this experiment, we took the nominal models trained on un-corrupted data (i.e., p d =0 ) and fine-tuned on corrupted data at p d =0 . 1 . We then evaluated the fine-tuned policies at a range of drop probabilities within [0 , 0 . 5] .

Within Throttled-BC , the time in-between successive images (equivalently, the number of control actions to be generated between two successive images), i.e., T is varied. We hypothesize that for tasks where the policy may be sensitive to the higher frequency observations, increasing T should correlate with an increase in difficulty, necessitating non-trivial latent prediction. Thus, Throttled-BC may be seen as a deterministic limit of Dropped-BC . To perform this experiment, we fine-tuned each model on data corresponding to different values of T by warmstarting 3 with parameters trained at T = T 0 .

2 All appendices referenced herein may be found within the online version of this work [40].

## C. Baselines

In addition to the SB model given by (4), we compare against two additional baselines that incorporate some temporal structure into the policy. The first, termed MA -SB corresponds to a variation of SB whereby one leverages an exponential-moving-average filter of the higherfrequency observations x t in place of the direct feedthrough of the interpolant ˆ x t ( · ) . This imbues the SB model with some stateful reasoning over the time-series x t . The final baseline corresponds to the Neural Dynamic Policies ( NDP ) model introduced in [28], whereby the observations ( s t ,x t (0)) are encoded into a set of parameters for a DMP that governs the open-loop evolution of u t ( · ) . Please see Appendix B for the relevant equations and additional architecture details.

## V. RESULTS

Wepresent two key metrics for comparison: r max : the maximum reward over an episode and r Σ : the total accumulated reward over an episode. For all environments, r max =1 is equivalent to 'task success' (i.e., object fully covered by cloth or ball caught in cup). The total reward measures efficiency, i.e., a larger r Σ indicates either more time-steps spent with the object fully covered or less time to catch the ball. Figure 4 summarizes the Throttled-BC performance of all four models across the three environments for varying values of T . The left-most column in each subfigure corresponds to the T = T 0 nominal setting, while the remaining columns correspond to various levels of image throttling.

On the Throttled-BC results, we first highlight that all model architectures are competitive at the lowest values of T , indicating that all policy architectures have the requisite capacity to solve the task under 'nearly' MDP conditions. However, there is considerable variability in performance as the image arrival rate is throttled down (i.e., T is increased), particularly for the BiC environments.

For the CC task, image throttling does not appear to cause any significant degradation in performance. This is likely due to the quasi-static nature of the task. A single snapshot of the scene is sufficient to execute a fairly long open-loop sequence of actions (i.e., the NDP model). Further, the task is not overly sensitive to the higher-frequency observations.

3 Note that no additional data was collected - we simply paired an image s t with a longer sequence of observations x t and actions u t , thus effectively reducing the 'dataset size' by the factor T/T 0 .

Fig. 4: Throttled-BC: Average (over 15 rollouts) r max ( left ), and r Σ ( right ), for varying values of T for cloth-covering ( top ), ball-in-cup ( middle ), and variable ball-in-cup ( bottom ). r max =1 indicates task success while larger values of r Σ indicate a faster time to completion. The presented values and ± 1 σ ranges are computed across 3 independent seeds.

<!-- image -->

Thus, there is no discernible advantage to the CDE method of incorporating these observations, as compared with the explicit models SB and MA -SB .

In contrast, for both the standard and variable BiC environments, InFuser outperforms all other models in both metrics as T is increased. The separation is more apparent for the harder vBiC variant where the higher-frequency observation is just the force-torque sensor. Indeed, ablations within this environment demonstrated that the change in the higher-frequency sensor is the primary source of the larger spread in performance, rather than the variability of the inertial properties. The InFuser architecture's stateful representation is particularly useful in this setting for performing the latent conversion to proprioceptive-level information.

Aplausible justification for the observed trends is that the BiC tasks are a lot more dynamic in nature, necessitating nontrivial latent predictions to generate good action sequences. Quantifying 'good' however is a rather subtle discussion. Figure 5 plots the empirical BC loss defined in (6) on a heldout test set, for each environment at a value of T larger than T 0 . We observe that while the InFuser architecture seems to exhibit a correlation between the test BC loss and closed-loop reward performance, this trend does not generalize to the other models. That is, lower test error does not necessarily imply better closed-loop reward performance, indicating the need for further investigation into the closed-loop robustness of these HCT policies. We provide the remaining test error plots in Appendix C-A, further corroborating this observation.

Fig. 5: Throttled-BC test loss vs training steps for cloth-covering at T =20 ( top ), ball-in-cup at T =4 ( bottom-left ), and variable ball-in-cup at T =4 ( bottom-right ).

<!-- image -->

The aforementioned trends are further reflected in the Dropped-BC experiments (see Figure 6). For the CC task, all models except NDP remain roughly on par with the 'nodropping' baseline performance, even as the probability of dropped images p d is increased, thereby reinforcing the quasistatic nature of the task. For the BiC tasks however, all models degrade in performance as p d is increased, with InFuser being the most performant in the vBiC variation, while being on par with the other models in the standard variation. Once again, this highlights the higher level of requisite agility for the BiC tasks and the inherent robustness of the InFuser policy.

## VI. CONCLUSIONS AND FUTURE WORK

In this work we introduced InFuser , a policy architecture for seamlessly (In)tegrating and (Fus)ing multi-sensory multi-frequency observations for continuous-time control within dynamic environments. The key insight is to treat multi-sensory fusion and temporal abstractions within a unified framework using controlled differential equations. Our CDE-based architecture evolves a continuous-time multi-sensory latent embedding that is conditioned on images, and driven by higher-frequency observations. The resulting model outperforms state-of-the-art multi-sensory fusion architectures and DMP-based policies, particularly when deployed on tasks that demand a non-trivial level of agility.

Our contributions enable several exciting future directions for research. First, although this work only studies the behavior cloning setting, one may easily incorporate this architecture within hybrid continuous-time reinforcement learning algorithms [41]. Second, recent work [32] has demonstrated the importance of multi-modal representations for BC-trained policies. A promising direction therefore is to infuse multi-modality within the CDE architecture to broaden the class of achievable tasks, such as those from the manipulation domain. Third, the proposed CDE model naturally enables deployment on platforms where the speed of image processing may otherwise be a key bottleneck in realizing truly agile continuous-time control.

Fig. 6: Dropped-BC: Average (over 100 rollouts) r max ( left ), and r Σ ( right ), for varying values of p d (image-drop probability) for cloth-covering ( top ), ball-in-cup ( middle ), and variable ball-in-cup ( bottom ). The parameter T is fixed at the lowest setting from the Throttled-BC experiments. The shaded ± 1 σ range is computed over three seeds.

<!-- image -->

## REFERENCES

- [1] P. Kidger, J. Morrill, J. Foster, and T. Lyons, 'Neural controlled differential equations for irregular time series,' in Advances in Neural Information Processing Systems , 2020.
- [2] R. T. Chen, Y . Rubanova, J. Bettencourt, and D. Duvenaud, 'Neural ordinary differential equations,' arXiv preprint arXiv:1806.07366 , 2018.
- [3] M. A. Lee, Y . Zhu, K. Srinivasan, P . Shah, S. Savarese, L. Fei-Fei, A. Garg, and J. Bohg, 'Making sense of vision and touch: Selfsupervised learning of multimodal representations for contact-rich tasks,' in 2019 International Conference on Robotics and Automation (ICRA) . IEEE, 2019, pp. 8943-8950.

- [4] Y. Li, J.-Y . Zhu, R. Tedrake, and A. Torralba, 'Connecting touch and vision via cross-modal prediction,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2019, pp. 10609-10618.
- [5] M. A. Lee, M. Tan, Y . Zhu, and J. Bohg, 'Detect, reject, correct: Crossmodal compensation of corrupted sensors,' in 2021 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2021, pp. 909-916.
- [6] Y. Liu, D. Romeres, D. K. Jha, and D. Nikovski, 'Understanding multi-modal perception using behavioral cloning for peg-in-a-hole insertion tasks,' arXiv preprint arXiv:2007.11646 , 2020.
- [7] R. Gao, Y .-Y . Chang, S. Mall, L. Fei-Fei, and J. Wu, 'Objectfolder: A dataset of objects with implicit visual, auditory, and tactile representations,' arXiv preprint arXiv:2109.07991 , 2021.
- [8] R. Yang, M. Zhang, N. Hansen, H. Xu, and X. Wang, 'Learning vision-guided quadrupedal locomotion end-to-end with cross-modal transformers,' arXiv preprint arXiv:2107.03996 , 2021.
- [9] R. Calandra, A. Owens, D. Jayaraman, J. Lin, W. Yuan, J. Malik, E. H. Adelson, and S. Levine, 'More than a feeling: Learning to grasp and regrasp using vision and touch,' IEEE Robotics and Automation Letters , vol. 3, no. 4, pp. 3300-3307, 2018.
- [10] R. Hu and A. Singh, 'UniT: Multimodal multitask learning with a unified transformer,' arXiv preprint arXiv:2102.10772 , 2021.
- [11] J. Sung, I. Lenz, and A. Saxena, 'Deep multimodal embedding: Manipulating novel objects with point-clouds, language and trajectories,' in 2017 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2017, pp. 2794-2801.
- [12] T. De Bruin, J. Kober, K. Tuyls, and R. Babuˇ ska, 'Integrating state representation learning into deep reinforcement learning,' IEEE Robotics and Automation Letters , vol. 3, no. 3, pp. 1394-1401, 2018.
- [13] T. Lesort, N. D´ ıaz-Rodr´ ıguez, J.-F. Goudou, and D. Filliat, 'State representation learning for control: An overview,' Neural Networks , vol. 108, pp. 379-392, 2018.
- [14] M. Saveriano, F. J. Abu-Dakka, A. Kramberger, and L. Peternel, 'Dynamic movement primitives in robotics: A tutorial survey,' arXiv preprint arXiv:2102.03861 , 2021.
- [15] S. M. Khansari-Zadeh and A. Billard, 'Learning stable nonlinear dynamical systems with Gaussian mixture models,' IEEE Transactions on Robotics , vol. 27, no. 5, pp. 943-957, 2011.
- [16] --, 'Learning control Lyapunov function to ensure stability of dynamical system-based robot reaching motions,' Robotics and Autonomous Systems , vol. 62, no. 6, pp. 752-765, 2014.
- [17] S. A. Khader, H. Yin, P. Falco, and D. Kragic, 'Learning stable normalizing-flow control for robotic manipulation,' in 2021 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2021, pp. 1644-1650.
- [18] S. Singh, S. M. Richards, V . Sindhwani, J.-J. E. Slotine, and M. Pavone, 'Learning stabilizable nonlinear dynamics with contraction-based regularization,' The International Journal of Robotics Research , vol. 40, no. 10-11, pp. 1123-1150, 2021.
- [19] V. Sindhwani, S. Tu, and M. Khansari, 'Learning contracting vector fields for stable imitation learning,' arXiv preprint arXiv:1804.04878 , 2018.
- [20] B. E. Khadir, J. V arley, and V . Sindhwani, 'Teleoperator imitation with continuous-time safety,' arXiv preprint arXiv:1905.09499 , 2019.
- [21] A. Rai, G. Sutanto, S. Schaal, and F. Meier, 'Learning feedback terms for reactive planning and control,' in 2017 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2017, pp. 2184-2191.
- [22] Y. Chebotar, O. Kroemer, and J. Peters, 'Learning robot tactile sensing for object manipulation,' in 2014 IEEE/RSJ International Conference on Intelligent Robots and Systems . IEEE, 2014, pp. 3368-3375.
- [23] C. Daniel, G. Neumann, and J. Peters, 'Hierarchical relative entropy policy search,' in Artificial Intelligence and Statistics . PMLR, 2012, pp. 273-281.
- [24] F. Stulp, E. A. Theodorou, and S. Schaal, 'Reinforcement learning with sequences of motion primitives for robust manipulation,' IEEE Transactions on robotics , vol. 28, no. 6, pp. 1360-1370, 2012.
- [25] S. Parisi, H. Abdulsamad, A. Paraschos, C. Daniel, and J. Peters, 'Reinforcement learning vs human programming in tetherball robot games,' in 2015 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) . IEEE, 2015, pp. 6428-6434.
- [26] R. S. Sutton, D. Precup, and S. Singh, 'Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning,' Artificial intelligence , vol. 112, no. 1-2, pp. 181-211, 1999.
- [27] J. Peters and S. Schaal, 'Reinforcement learning of motor skills with policy gradients,' Neural networks , vol. 21, no. 4, pp. 682-697, 2008.
- [28] S. Bahl, M. Mukadam, A. Gupta, and D. Pathak, 'Neural dynamic policies for end-to-end sensorimotor learning,' arXiv preprint arXiv:2012.02788 , 2020.
- [29] S. Bahl, A. Gupta, and D. Pathak, 'Hierarchical neural dynamic policies,' arXiv preprint arXiv:2107.05627 , 2021.
- [30] T. Narita and O. Kroemer, 'Policy blending and recombination for multimodal contact-rich tasks,' IEEE Robotics and Automation Letters , vol. 6, no. 2, pp. 2721-2728, 2021.
- [31] A. Escontrela, G. Yu, P. Xu, A. Iscen, and J. Tan, 'Zero-shot terrain generalization for visual locomotion policies,' arXiv preprint arXiv:2011.05513 , 2020.
- [32] P. Florence, C. Lynch, A. Zeng, O. A. Ramirez, A. Wahid, L. Downs, A. Wong, J. Lee, I. Mordatch, and J. Tompson, 'Implicit behavioral cloning,' in Conference on Robot Learning , 2022, pp. 158-168.
- [33] A. Zeng, P. Florence, J. Tompson, S. Welker, J. Chien, M. Attarian, T. Armstrong, I. Krasin, D. Duong, V . Sindhwani, et al. , 'Transporter networks: Rearranging the visual world for robotic manipulation,' arXiv preprint arXiv:2010.14406 , 2020.
- [34] E. Jang, A. Irpan, M. Khansari, D. Kappler, F. Ebert, C. Lynch, S. Levine, and C. Finn, 'BC-Z: Zero-shot task generalization with robotic imitation learning,' in Conference on Robot Learning . PMLR, 2022, pp. 991-1002.
- [35] B. Øksendal, Stochastic differential equations: an introduction with applications . Springer Science &amp; Business Media, 2013.
- [36] T. J. Lyons, M. Caruana, and T. L´ evy, Differential equations driven by rough paths . Springer, 2007.
- [37] Y. Rubanova, R. T. Chen, and D. K. Duvenaud, 'Latent ordinary differential equations for irregularly-sampled time series,' in Advances in neural information processing systems , 2019.
- [38] J. Morrill, P. Kidger, L. Yang, and T. Lyons, 'Neural controlled differential equations for online prediction tasks,' arXiv preprint arXiv:2106.11028 , 2021.
- [39] Y. Tassa, Y. Doron, A. Muldal, T. Erez, Y. Li, D. d. L. Casas, D. Budden, A. Abdolmaleki, J. Merel, A. Lefrancq, et al. , 'Deepmind control suite,' arXiv preprint arXiv:1801.00690 , 2018.
- [40] S. Singh, F. M. Ramirez, J. Varley, A. Zeng, and V. Sindhwani, 'Multiscale sensor fusion and continuous control with neural CDEs,' To be uploaded on arXiv. , 2022.
- [41] T. Xiao, E. Jang, D. Kalashnikov, S. Levine, J. Ibarz, K. Hausman, and A. Herzog, 'Thinking while moving: Deep reinforcement learning with concurrent control,' in International Conference on Learning Representations , 2020.
- [42] K. Choromanski, A. Pacchiano, J. Parker-Holder, Y . Tang, D. Jain, Y. Yang, A. Iscen, J. Hsu, and V . Sindhwani, 'Provably robust blackbox optimization for reinforcement learning,' in Conference on Robot Learning , 2020, pp. 683-696.
- [43] D. P. Kingma and J. Ba, 'Adam: A method for stochastic optimization,' arXiv preprint arXiv:1412.6980 , 2014.
- [44] L. N. Smith, 'Cyclical learning rates for training neural networks,' in 2017 IEEE winter conference on applications of computer vision (WACV) . IEEE, 2017, pp. 464-472.
- [45] J. Bradbury, R. Frostig, P . Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. V anderPlas, S. Wanderman-Milne, and Q. Zhang, 'JAX: composable transformations of Python+NumPy programs,' 2018. [Online]. Available: http://github.com/google/jax
- [46] J. Heek, A. Levskaya, A. Oliver, M. Ritter, B. Rondepierre, A. Steiner, and M. van Zee, 'Flax: A neural network library and ecosystem for JAX,' 2020. [Online]. Available: http://github.com/google/flax

## A. Cloth-Covering

The set of all higher-frequency observations within x t include: robot joint-angles and velocities, end-effector Cartesian position, and gripper status (open vs closed), yielding a net higher-frequency observation dimension of n =30 . The reward function at each step is:

<!-- formula-not-decoded -->

where occ ratio is the occlusion ratio for the object, computed as the ratio of the object's visible pixel surface area to the object's total pixel surface area. The controller outputs are the change in end-effector Cartesian position (3 dof), and gripper status (1 dof). Note that as the gripper action is a binary variable, the control space dimension is 5, with gripper controls computed as logits.

Expert demonstrations are provided through a scripted policy which, using privileged simulator state regarding cloth and block positions, drives the arm to the center of the cloth, closes the gripper, drives the arm over-top the block position, and opens the gripper. Each episode lasts 120 control steps, corresponding to 120 /T seen images.

## B. Ball-In-Cup

The higher-frequency observation is simply the cup 2D position and velocity, i.e., n =4 , and the reward function is:

<!-- formula-not-decoded -->

Further, the environment terminates early after 5 consecutive steps of r t =1 (i.e., ball is in the cup); the maximum episode length is 100 control steps. The small negative penalty allow us to measure task efficiency (i.e., how quickly is the catch completed) via the total episode reward r Σ . The expert data was generated by a separate MLP policy, trained via Blackbox optimization [42], with access to the cup and ball position and velocities.

## C. Variable Ball-In-Cup

The variable BiC (vBiC) environment is a harder version of BiC whereby the mass of the ball and the max length of the elastic string are randomized at the beginning of each episode. In particular, we allow both parameters to vary between ± 1 / 3 of the nominal value, a non-negligible range. To make the task harder, the higher-frequency observation is reduced to just a 3-dof force-torque sensor collocated at the cup position. This variation represents the closest analogue to a human performing the same task, where the set of observations are visual (the RGB camera) and force-feedback. The expert policy for this task was again trained via Blackbox optimization, with access to the cup and ball positions and velocities, as well as the mass of the ball and max length of the string. The maximum episode length is 120 control steps with early termination after 5 consecutive steps with the ball in the cup.

## APPENDIX B MODELS AND ARCHITECTURES

We first provide the relevant equations for the two additional baselines. Within MA -SB , we replace the second equation in (4) with:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where σ ∈ &gt; 0 is the filter time-constant, set in all experiments to be the timestep in between successive higher-frequency observations. The filter is initialized as ˘ x t (0)= x t (0) , and is driven by the cubic interpolant ˆ x t ( · ) of the raw measurements x t . Adapting notation from [28], the NDP model is summarized by the following encoder:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

## APPENDIX A

## ENVIRONMENTS

and pair of ODEs:

<!-- formula-not-decoded -->

Here φ t is a phase variable, initialized as φ t (0)=1 , and { α u ,α φ ,β } are strictly positive constants. The forcing function f t takes the form:

<!-- formula-not-decoded -->

where ψ ( φ ):=( ψ 1 ( φ ) ,...,ψ N ( φ )) ∈ R N is a vector of Gaussian radial basis functions evaluated at φ , N is the number of basis functions, and ◦ denotes the Hadamard product 4 . Once again, we see that the computation of u t (0) takes the same structural form as the other baselines and InFuser . However, unlike InFuser which continuously incorporates the higher-frequency measurements x t along with the fixed image s t , the NDP generates an open-loop control trajectory between t and t +1 using only ( s t ,x t (0)) .

## A. Architecture

For fair comparison, all models shared the same image encoder F s - a 10-layer Convolutional ResNet with ReLu activations and input-layer BatchNorm. The penultimate embedding is average-pooled, flattened, and passed through a single dense layer. Note that the NDP encoder also includes the map F dmp , consisting of two 2-layer MLPs mapping ( z s t ,x 0 t ) to the DMPparameters g t and W t .

All models also shared the same fusion network F o - a 4-layer MLP with ReLu activations. The final embedding is concatenated with the high-frequency observation input into F o so that z o t ( τ ) includes ˆ x t ( τ ) (or ˘ x t ( τ ) for MA -SB ) as a sub-vector. Finally, all models shared the same decoder network F u - a 4-layer MLP with ReLu activations. Note that the NDP decoder outputs both u t (0) and du t (0) /dτ , while for all other models, the decoder outputs only u t ( τ ) .

For the InFuser model, the CDE matrix-valued function f o is a bottleneck 2-layer MLP with ReLu activation for the hidden layer, and tanh non-linearity for the output layer, as recommended by [1]. This parameterized function is stacked on top of the matrix [ I n 0 n ] where I n is the n × n identity matrix and 0 n is an n -dimensional column vector of zeros. This is done since z o t ( τ ) includes ˆ x t ( τ ) as a sub-vector.

We also employed an l 2 weight regularization penalty for the parameters of f o to keep the vector field well conditioned for the ODE solver (fixed-step rk4).

## APPENDIX C

## TRAINING AND EVALUATION

Normalization : The higher-frequency observations x t were normalized for the CC and vBiC tasks to have zero mean and unit variance. Additionally, the actions were also normalized to the range [ -1 , 1] for the CC task, while the action space was already normalized for the BiC and vBiC tasks to lie within the range [ -1 , 1] .

Optimizer : We used the Adam optimizer [43] for all training and fine-tuning, along with triangular cyclical 5 learning rate schedules [44], with a max / min ratio of 5 and cycle length of 8 × # steps per epoch. The maximum learning rate was decayed by a factor of 5 if the training loss was observed to stall for 50 epochs, and training was terminated early once the stall lasted for 100 epochs. The initial minimum (constant for BiC) learning rates for the T = T 0 cycles were: 10 -3 (BiC), 10 -4 (CC), and 5 · 10 -4 (vBiC). For all fine-tuning jobs, i.e., Throttled-BC ( T &gt;T 0 ) and Dropped-BC ( T = T 0 ,p d =0 . 1 ), we only used cyclical learning rates with initial minimum cycle values: 10 -4 (BiC), 5 · 10 -5 (CC), and 10 -4 (vBiC). The batch sizes were 64 (CC) and 128 (BiC and vBiC).

Evaluation Protocol : For the results in Figure 4 and Table I, the performance for each model and independent seed is quantified by computing a smoothed average of the average metrics over a fixed window during training (average metrics were evaluated very 1k training steps over 15 rollouts). The same window is used for all models (thus, all models have seen the same amount of training data) and we present the mean and ± 1 σ range for the computed smoothed values across three independent seeds. For the results in Figure 6, each model and seed, fine-tuned at p d =0 . 1 , is evaluated at varying drop probabilities within [0 , 0 . 5] , where for each evaluation p d , we recorded the average performance metrics r max and r Σ over 100 rollouts. We then present the mean and ± 1 σ range of these average metrics over 3 independent seeds.

4 Note that each row of the matrix W t corresponds to a unique control dimension, and is an N -dimensional vector of weights.

5 With the exception of the BiC task at T = T 0 , where we used a larger constant learning rate to avoid premature convergence to bad local minima.

TABLE I: Average (over 15 rollouts) r max (left), and r Σ (right), for varying values of T across all environments. r max =1 indicates task success while larger values of r Σ indicate a faster time to completion. The presented means and ± 1 σ ranges are computed across 3 independent seeds.

|         | CC                | CC                | CC                | CC                |         | CC              | CC                | CC                 | CC                 |
|---------|-------------------|-------------------|-------------------|-------------------|---------|-----------------|-------------------|--------------------|--------------------|
| Method  | 10                | 20                | 30                |                   |         | Method          | 10                | 20                 | 30                 |
| InFuser | 0 . 989 ± 0 . 005 | 0 . 987 ± 0 . 002 | 0 . 984 ±         | 0 . 002           |         | InFuser         | 37 . 06 ± 0 . 86  | 37 . 16 ± 0 . 23   | 37 . 35 ± 0 . 67   |
| NDP SB  | 0 . 989 ± 0 . 004 | 0 . 984 ± 0 . 008 | 0 . 923 ± 0 . 017 |                   |         | NDP             | 36 . 85 ± 0 . 58  | 36 . 46 ± 0 . 58   | 33 . 83 ± 0 . 86   |
|         | 0 . 984 ± 0 . 003 | 0 . 99 ± 0 . 002  | 0 . 99 ± 0 .      | 001               | SB      |                 | 36 . 34 ± 0 . 30  | 37 . 25 ± 0 . 16   | 37 . 28 ± 0 . 03   |
| MA - SB | 0 . 99 ± 0 . 004  | 0 . 99 ± 0 . 002  | 0 . 987 ±         | 0 . 001           | MA      | - SB            | 36 . 97 ± 0 . 36  | 37 . 13 ± 0 . 34   | 37 . 16 ± 0 . 11   |
| (a)     | (a)               | (a)               | (a)               | (a)               |         | (b)             | (b)               | (b)                | (b)                |
|         | BiC               | BiC               | BiC               | BiC               |         | BiC             | BiC               | BiC                | BiC                |
| Method  | 2                 | 4                 | 6                 | 8                 | Method  | 2               | 4                 | 6                  | 8                  |
| InFuser | 0 . 975 ± 0 . 02  | 0 . 878 ± 0 . 02  | 0 . 745 ± 0 . 007 | 0 . 601 ± 0 . 033 | InFuser | 3 . 30 ± 0 . 25 | 2 . 35 ± 0 . 19   | 1 . 14 ± 0 . 035   | - 0 . 02 ± 0 . 28  |
| NDP     | 0 . 976 ± 0 . 004 | 0 . 748 ± 0 . 01  | 0 . 533 ± 0 . 027 | 0 . 378 ± 0 . 052 | NDP     | 3 . 30 ± 0 . 05 | 1 . 21 ± 0 . 11   | - 0 . 4 ± 0 . 2    | - 1 . 65 ± 0 . 395 |
| SB      | 0 . 948 ± 0 . 006 | 0 . 769 ± 0 . 01  | 0 . 56 ± 0 . 085  | 0 . 495 ± 0 . 018 | SB      | 3 . 09 ± 0 . 07 | 1 . 46 ± 0 . 11   | - 0 . 27 ± 0 . 71  | - 0 . 77 ± 0 . 13  |
| MA - SB | 0 . 943 ± 0 . 002 | 0 . 714 ± 0 . 03  | 0 . 6 ± 0 . 019   | 0 . 516 ± 0 . 023 | MA - SB | 2 . 81 ± 0 . 07 | 1 . 00 ± 0 . 24   | 0 . ± 0 . 15       | - 0 . 63 ± 0 . 232 |
|         | (c)               | (c)               | (c)               | (c)               |         | (d)             | (d)               | (d)                | (d)                |
|         | vBiC              | vBiC              | vBiC              | vBiC              |         | vBiC            | vBiC              | vBiC               | vBiC               |
| Method  | 2                 | 4                 | 6                 | 8                 | Method  | 2               | 4                 | 6                  | 8                  |
| InFuser | 0 . 973 ± 0 . 008 | 0 . 861 ± 0 . 002 | 0 . 717 ± 0 . 012 | 0 . 617 ± 0 . 007 | InFuser | 3 . 30 ± 0 . 1  | 1 . 85 ± 0 . 03   | 0 . 452 ± 0 . 1    | - 0 . 55 ± 0 . 05  |
| NDP     | 0 . 973 ± 0 . 005 | 0 . 817 ± 0 . 01  | 0 . 585 ± 0 . 004 | 0 . 394 ± 0 . 019 | NDP     | 3 . 16 ± 0 . 06 | 1 . 20 ± 0 . 02   | - 0 . 856 ± 0 . 02 | - 2 . 35 ± 0 . 15  |
| SB      | 0 . 86 ± 0 . 013  | 0 . 410 ± 0 . 011 | 0 . 349 ± 0 . 002 | 0 . 313 ± 0 . 013 | SB      | 1 . 69 ± 0 . 15 | - 2 . 16 ± 0 . 08 | - 2 . 65 ± 0 . 04  | - 2 . 98 ± 0 . 11  |
| MA - SB | 0 . 920 ± 0 . 008 | 0 . 566 ± 0 . 009 | 0 . 411 ± 0 . 007 | 0 . 357 ± 0 . 009 | MA - SB | 2 . 51 ± 0 . 09 | - 0 . 90 ± 0 . 07 | - 2 . 06 ± 0 . 08  | - 2 . 65 ± 0 . 07  |

Computing Infrastructure : All training was performed on Google v2 2 × 2 TPUs, and all code was written in JAX [45], using the Flax neural network library [46].

## A. Additional results

Table I provides the raw performance numbers illustrated in Figure 4, while Figure 7 illustrates the test loss as a function of training steps.

Fig. 7: Throttled-BC test loss vs training steps for cloth-covering ( top ), ball-in-cup ( middle ), and variable ball-in-cup ( bottom ). T is increasing from left-to-right.

<!-- image -->