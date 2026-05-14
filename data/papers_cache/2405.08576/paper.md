## Hearing Touch: Audio-Visual Pretraining for Contact-Rich Manipulation

Jared Mejia 1 , Victoria Dean 2 , Tess Hellebrekers 3 , Abhinav Gupta 1

Abstract -Although pre-training on a large amount of data is beneficial for robot learning, current paradigms only perform large-scale pretraining for visual representations, whereas representations for other modalities are trained from scratch. In contrast to the abundance of visual data, it is unclear what relevant internet-scale data may be used for pretraining other modalities such as tactile sensing. Such pretraining becomes increasingly crucial in the low-data regimes common in robotics applications. In this paper, we address this gap by using contact microphones as an alternative tactile sensor. Our key insight is that contact microphones capture inherently audio-based information, allowing us to leverage large-scale audio-visual pretraining to obtain representations that boost the performance of robotic manipulation. To the best of our knowledge, our method is the first approach leveraging largescale multisensory pre-training for robotic manipulation. For supplementary information including videos of real robot experiments, please see https://sites.google.com/view/hearing-touch.

## I. INTRODUCTION

Two key components consistently improve the performance of robotic manipulation: (1) pre-training on a large amount of data [1]-[5] and (2) using multisensory input, especially tactile sensing [6]-[11]. While recent work has leveraged pretraining on large-scale video datasets to create reusable vision representations for robot learning [1]-[3], there has been little focus on large-scale pretraining for other modalities such as tactile sensing. This gap arises due to the lack of relevant data at a comparable scale for tactile sensing. As a result, current approaches using non-visual sensory modalities are restricted to learning from a limited amount of task-specific data [10], [12]. How can we leverage internet data in pretraining tactile representations for manipulation?

Piezo contact microphones have emerged as a promising sensor in robotics due to their ability to capture highfrequency temporal information through structural vibrations captured as audio. Prior work has already demonstrated the ability to use contact audio for manipulation tasks [6], [12], [13]. In contrast to traditional tactile sensors, the signal provided by contact microphones is inherently audio; hence recent work on learning audio-visual representations may apply to contact audio obtained from robot interactions.

We investigate how large-scale audio-visual training might be beneficial for learning contact audio representations for robotic manipulation. Our method makes use of Audio-Visual Instance Discrimination (AVID) [14], a selfsupervised learning approach to learn audio-visual representations, pre-trained on Audioset [15], a dataset containing over 2 million 10-second video clips of human and animal sounds, music, and environmental sounds drawn from the internet. Initializing our encoder with A VID weights, we train a policy with behavior cloning that fuses visual and audio inputs with self-attention in order to predict actions.

1 Robotics Institute, Carnegie Mellon University

2 Olin College of Engineering

3 Meta AI

Fig. 1: Hearing touch : We enable multisensory pretraining for manipulation by transferring audio-visual representations to manipulation tasks using vision and contact audio.

<!-- image -->

We validate our approach with experiments on three realworld manipulation tasks in the low-data regime, using at most 60 demonstrations per task. Surprisingly, despite the domain gap between the audio in Audioset and contact audio obtained through manipulation, we find that our approach improves performance over visual-only policies-especially in test settings where objects and locations differ significantly from the training data. Furthermore, our approach outperforms equivalent policies with audio encoders trained from scratch. Our experimental results reveal a promising avenue for multimodal pretraining across many robotic applications where neither vision alone nor training multisensory representations from scratch are sufficient.

## II. RELATED WORK

a) Audio in robotics: Several works have shown the ability to reason over audio in robotics scenarios including object recognition [16], material classification [17], estimating the volume and flow of granular material [13], exploration in RL [18], occluded manipulation [19], manipulation for sound replication [12], and waypoint setting in audiovisual navigation [20]. SHF [6] introduces a mechanism for fusing input from a camera, a Gelsight sensor [21], and a contact microphone attached to the object of interest with self-attention for manipulation. Though our method also uses self-attention to fuse multisensory representations, we focus on leveraging large-scale audio pretraining, using visual input from a third-person camera and a contact microphone mounted directly on the robot enabling the robot to reason over vibrations caused by contact between tools and objects.

Fig. 2: Two-stage model training. AVID and R3M pretraining leverages the large scale of internet video data (blue dashed box). We initialize the vision and audio encoders with the resulting pre-trained representations and then train the entire policy end-to-end with behavior cloning from a small number of in-domain demonstrations. The policy takes image and spectrogram inputs (left) and outputs a sequence of actions in delta end effector space (right).

<!-- image -->

- b) Tactile sensing for manipulation: Several types of tactile sensors exist for application to robotic manipulation [22]-[27]. We use contact microphones as an alternative tactile sensor, which are relatively inexpensive in comparison to common tactile sensors and can record vibrations with up to 1000 times higher frequency than optical and magnetic-based tactile sensors (32-48 kHz vs 30-400 Hz) [22]-[24]. Recent work has focused on applying traditional tactile sensors for learning to grasp objects without visual observations [11] and in combination with visual observations for learning to improve the grasp of an object [8]. Our method using contact audio allows the sensor to measure vibrations directly via the sensor mounted on the gripper as well as indirectly through vibrations traveling along tools grasped by the gripper.
- c) Audio-visual representation learning: Selfsupervised representation learning has been applied to the audio-visual domain, using audio-visual correspondence (AVC) as a form of cross-modal self-supervision from video [28], [29]. Other approaches make use of the synchronization between vision and sound for sound representations [30], audio-visual sound separation [31], and sound localization [32]. More recent work explores contrastive learning methods to discriminate between training instances using cross-modal and within-modal targets [14], [33], [34]. In our work, we use a pre-trained implementation of AVID [14] for obtaining audio-visual representations.
- d) Representation learning for robotic manipulation: Several recent works use self-supervision to decouple representation learning of sensory inputs from behavior learning for robotic manipulation tasks [10], [12], [35]-[37]. A recent trend aims to obtain a universal visual representation-a single perception module pre-trained on large amounts of

video data that can be frozen and used for downstream policy learning [1]-[3], however, there has been little focus on large scale pre-training for representation learning beyond vision in the context of robot manipulation. That Sounds Right [12] also explores contact audio pre-training for behavior learning, however, their approach utilizes self-supervised learning using only task-specific data, whereas our method leverages the richness and diversity of large-scale audiovisual data for pre-training a contact audio representation. Further, we operate in the low-data regime with less than 100 demonstrations per task, whereas [12] collects 5,000 data points per task. We demonstrate the benefit of large-scale pre-training over in-domain SSL in the low-data setting.

## III. MANIPULATION WITH AUDIO-VISUAL PRETRAINING

Given the difficulty and expense of collecting data in robotic settings, we turn toward leveraging more easily attainable large-scale sources of information such as internet data for learning manipulation policies. By utilizing contact microphones, we move beyond pre-training solely for visual input and obtain a means of pre-training a tactile sensor with large amounts of rich, audio-visual data. We outline further details of our approach in the following sections.

## A. Sensors

At each timestep, we collect image observations v t and two-second clips of contact audio a t . Image observations are obtained from a third-person view camera and audio is obtained by averaging the signal captured from four contact microphones mounted on the robot. Contact microphones capture vibrations, they are sensitive to contact not only directly between objects and the sensors but also contact resulting in vibrations traveling between objects. As a result, our setup allows the robot to sense subtle interactions between surfaces and tools that are grasped by the arm, as in

<!-- image -->

<!-- image -->

(b) Flipping task

(c) Scooping task

<!-- image -->

(d) Zipping task

Fig. 3: Hardware and task setup. We attach the Piezo contact microphones to our gripper to record vibrations in the form of audio and run experiments on three real-world tasks with significant visual differences between train and test settings.

<!-- image -->

the flipping task which requires the use of a spatula, and the scooping task requiring the use of a spoon (Section IV-A).

## B. Audio and Visual Representation Pretraining

Our method uses large-scale audio-visual pre-training to initialize our audio encoder and large-scale visual pretraining to initialize our visual encoder. The audio encoder is extracted from AVID [14] pre-trained on audio-visual pairs from Audioset [15] with cross-modal discrimination, encouraging the network to learn video features that match the corresponding audio features and vice-versa. To isolate the effect of large-scale pre-training for our audio encoder, we use R3M [1], a proven method for pre-training visual features in robotic applications, R3M, with a ResNet18 [38] pre-trained on Ego4D human video dataset [39] with time contrastive learning and video-language alignment. Following [40], we keep both encoders unfrozen, continuing to update the weights during policy learning.

## C. Audio-Visual Behavior Cloning

We train a policy with behavior cloning on a small number of in-domain demonstrations (described in Section IV-A). The model architecture is visualized in Fig. 2. At each timestep, the policy takes in a two-second audio clip s t and a sequence of i images v t -i , . . . , v t spanning the same two-second window, which are fed through the audio and image encoders, respectively. We apply learned positional embeddings to each of the encoded representations and pass the result as input to a transformer decoder network similar to [6]. Similar to [41], [42] our method is quasi open-loop-at time step t the policy predicts H steps of actions, of which h ≤ H steps of actions are executed on the robot without re-planning. This approach allows the policy to remain responsive to subtle changes in the audio input while encouraging temporal action consistency and mitigating the effect of non-Markovian behaviors such as pauses in demonstrations. In particular, the final component of our network is a multi-layer perceptron that outputs actions a t , . . . , a t + h over a short horizon of h timesteps. Here, each action a t is a 6-dimensional continuous deltaend effector action composed of the Cartesian displacement

( x, y, z ) and the change in Euler angles ( α, β, γ ) . We optimize the network to minimize the standard MSE loss ℓ = 1 H ∑ H j =0 ( a t + j -π ( v t -i , . . . , v t , s t ) j ) 2 . Please see Section VI for more architectural details.

## IV. EXPERIMENTS

In our experiments, we aim to answer two key questions: (1) Do contact microphones mounted on a robot arm capture interactions difficult to perceive with vision alone? (2) Does large-scale pre-training for audio-based tactile sensors yield representations that are useful for robot manipulation?

We address these questions through real-world experiments on our setup described in Section IV-A by evaluating across three tasks (Section IV-B) and four methods (Section IV-C) in the low-data setting under conditions requiring significant generalization beyond the training data.

## A. Setup

- a) Hardware: We control a Franka Emika Panda Arm using an inverse kinematics solver to convert 6-DoF delta end effector Cartesian position and Euler rotation input to 7DoF joint action. The end effector actions are commanded at 30 Hz. On the Franka gripper, we mount four Piezo contact microphones, each of which records audio at 32 kHz. We use an Intel D435 RealSense camera with a fixed third-person view to collect image observations at 30 Hz.
- b) Data Collection: Demonstrations are collected via teleoperation using an Oculus Quest headset. The visual data collected by the Intel D435 RealSense camera collects images with a resolution of 480 × 640 . The audio waveforms are averaged across the four sensors and downsampled to 16 kHz. We normalize the audio waveforms and generate mel spectrograms of the 2s audio segment following the audio preprocessing in [14].

## B. Tasks

We present experiments on three real-world manipulation tasks, shown in Fig. 3a. The zipping task demonstrates the contact microphone's abilities to directly record vibrations touching the gripper, while the flipping and scooping tasks show their ability to record indirect contacts through vibrations traveling along tools (the spoon and spatula). We train on 40, 60, and 50 demonstrations for the flipping, scooping, and zipping tasks, respectively.

Fig. 4: Success rates across methods and tasks. Our method, shown in blue, outperforms baselines in all but one setup of the zipping task. Furthermore, our method displays much less variation in performance between different configurations of each task, showcasing an increase in the ability to generalize to drastic visual differences as a result of learning useful audio representations.

<!-- image -->

## C. Baselines and Implementation Details

We conduct experiments with our method and three other baselines. We use different methods of pretraining in order to measure the effect of large-scale audio-visual pretraining on learning a useful contact audio representation for manipulation. All methods incorporating audio use the same architecture: R3M [1] pre-trained on Ego4d [39] with a ResNet18 [38] backbone to initialize the image encoder.

- Vision-Only: a baseline that shares the same architecture as our method, except that it only uses image frames as input. This baseline tests whether the signal from contact microphones is beneficial in our setup.
- Scratch: a baseline with randomly initialized weights for the audio encoder. This baseline tests how contact audio pretraining affects performance.
- BYOL-A: Bootstrap Your Own Latent for Audio (BYOL-A) [43], a self-supervised approach to learning audio representations using only in-domain data. This baseline compares the effect of large-scale audio-visual pre-training to in-domain audio pre-training, with an emphasis on the amount of pre-training data.

## D. Results

The evaluation results across different variations the tasks are visualized in Fig. 4 and summarized in Table I. Our method using large-scale audio-visual pre-training outperforms all baselines across each of the three tasks with an average 23% higher 0-1 success rate and an average 76% increase in reward against the next best-performing baseline. Further, our method outperforms or matches the performance of all baselines in 8 / 9 tasks, displaying a lower variation in performance between different configurations of each task, indicating greater robustness to visual features.

TABLE I: Rewards and success rates across tasks.

|             | Flipping   | Scooping   | Scooping   | Zipping   | Zipping   |
|-------------|------------|------------|------------|-----------|-----------|
|             | Success %  | Reward     | Success %  | Reward    | Success % |
| Ours        | 50.0%      | 15.4       | 78.1%      | 8.9       | 88.9%     |
| BYOL-A      | 25.0%      | 2.3        | 25.0%      | 3.8       | 66.7%     |
| Scratch     | 15.4%      | 7.7        | 50.0%      | 6.9       | 72.2%     |
| Vision-Only | 0.0%       | 2.5        | 28.1%      | 4.4       | 44.4%     |

The Vision-Only baseline yields the worst performance across all tasks, providing evidence that contact audio improves the performance of manipulation policies over vision alone. Between BYOL-A and Scratch, the results are mixed-in the Flipping task BYOL-A outperforms Scratch and in Scooping and Zipping, Scratch performs better. Although BYOL-A includes an additional pre-training phase, the comparable performance with Scratch suggests that the augmentation techniques used by BYOL-A, while useful for learning audio representations for audio classification tasks when pre-trained on large audio datasets [43], are not effective when restricted to a small set of contact audio for learning manipulation policies. In contrast, our method utilizing AVID pre-training on Audioset greatly improves performance over Scratch and BYOL-A, demonstrating that the large-scale aspect of our method's audio-visual pretraining is the component most crucial to its success.

1) Qualitative Analysis: Many of the configurations of the task are difficult due to the noticeable visual differences between the train and test settings. As a result, the baselines suffer heavily from the domain shift and fail to generalize, often moving in jerk motions or away from the object of interest, even before coming into contact with objects. In contrast, our method appears to suffer less from the significant visual differences, suggesting that a good audio representation may prevent the model from overfitting to visual features during training.

The Vision-Only approach suffers most from the inability to perceive subtle interactions between surfaces, such as whether the spatula has successfully been slid under the bagel or the zipper is stuck on a corner. Despite having access to the same information as our method, the BYOL-A and Scratch baselines fail to reason effectively over the audio and utilize the additional information for taking action.

Fig. 5: t-SNE 2D projection. For comparative analysis of the learned embedding spaces, we visualize projections of the learned representations from each method in each variation of the flipping task. Lighter hues indicate the starting points and darker hues indicate the end points of the trajectories. Please see the video on our website for a better visualization.

<!-- image -->

In the scooping task, our method consistently learns to push the spoon deeper into the bowl until contact is made with the edge, and then tilt the spoon upward as the edge drags along the side of the bowl, increasing the amount of material scooped. This is more like the behavior of the training data than the baselines, which often fail to begin digging the spoon into the material as a result of misestimating the depth and relying on vision alone or scooping too shallow.

2) t-SNE Visualizations: To better understand the learned representations of our method in comparison with the baselines, we visualize 2D projections of the transformer output embeddings using t-SNE initialized with PCA deterministically. For each method, we plot the projections of the embeddings from a sample trajectory over time for each variation of the flipping task, including both train and test settings (Fig. 5). For our method, although the representations are spaced apart at the beginning of the trajectories likely due to the visual differences across settings (bottom right corner), the projections converge over the course of trajectories (moving clockwise) as the flipping motion is performed and completed. The visualization suggests the audio representations learned as a result of large-scale pretraining allow for the attention mechanism to better combine the audio-visual tokens, resulting in a more well-structured embedding space in comparison with the baselines.

## E. Ablation Studies

1) Zero-Shot Transfer: To get a better sense of how relevant pre-trained A VID weights are to downstream manipulation tasks, we train a version of our method with frozen AVID weights during policy learning (Fig. 6a). The results show that keeping the pre-trained audio encoder weights frozen during policy learning only slightly diminishes the performance of our method and still outperforms the next best baseline on the zipping task, highlighting the applicability of the general sensory representations learned from large-scale internet data for downstream manipulation tasks.

- 2) Scaling Performance: We run evaluations on the scooping task for models trained with dataset sizes 50% (30 demos) and 150% (90 demos) of the original data after collecting more demonstrations. As shown in Fig. 6b, our method continually improves at a steady rate with increasing training data size, roughly matching the rate of improvement for the Vision-Only baseline.

3) Generalization: To further investigate the poor performance of the Vision-Only baseline in comparison to our method on the flipping task, we compare the results between the train and test settings (Fig. 6c). The success rate of both methods is closer under the train settings, with our method performing 10% better. Evidently, despite using image augmentations during training, the Vision-Only baseline overfits to the visual features of the train settings in the demonstration data, resulting in a 60% drop in performance when applied to the test settings. In contrast, our method only sees a drop in success rate of about 20% between train and test settings, suggesting that pre-trained audio features prevent the network from overfitting to visual details in the training setting, hence attaining better generalization abilities.

- 4) Architecture Ablation: We replace the transformer with an MLP including an added additional linear layer to ensure the resultant network has approximately the same number of parameters as our proposed network (Fig. 6d). The selfattention mechanism for fusing audio and visual features is crucial to attaining good performance; both the success rate and the average reward drop by nearly 50% when replacing the transformer with an MLP on the scooping task. Despite this drop in performance, the alternative MLP architecture with direct concatenation of features performs comparably with Scratch, the next best baseline on the scooping task which shares the same architecture as our method. Hence, the attention mechanism is a necessary yet insufficient condition for attaining good performance when using both visual and contact audio observations-the attention mechanism combined with pre-trained audio and visual features results in favorable performance in the low-data regime.

Fig. 6: Ablations . We evaluate the zero-shot transfer of frozen pre-trained audio representations (a), the effect of dataset size (b), the generalization ability of our method (c), and the importance of self-attention to fuse sensory features (d).

<!-- image -->

## V. CONCLUSION

We present a simple yet effective approach for improving manipulation performance by utilizing contact microphones as a tactile sensor. We argue that a primary strength of this sensor is that, in contrast to other sensors, it allows us to leverage large-scale internet data of the same modality and pretrain a representation that is useful for downstream robotic tasks. We show that the representations learned from largescale audio-visual pretraining transfer well to such tasks despite the domain gap between contact audio in robotic manipulation and audio in internet videos. Future work may investigate which properties of pre-training datasets are most conducive to learning audio-visual representations for manipulation policies. Further, a promising direction would be to equip end-effectors with visuotactile sensors and contact microphones with pre-trained audio representations to determine how to leverage both for equipping robotic agents with a richer understanding of their environment.

The lessons learned from our experiments echo those being shared across other machine learning subfields-more data is the driving factor in learning better models. Considering the safety issues, inefficiency, and resources required in collecting robotic data, it is unlikely that robotics will experience the scaling properties witnessed in more data-rich domains [44], [45]. Thus, we hope to widen the data scarcity bottleneck via methods that extract information from broader data sources that may be useful to an embodied agent.

## VI. LIMITATIONS

While contact microphones work well in our experiments, there are cases in which they may be less useful: less dynamic tasks such as pick and place, situations where the robot itself generates significant vibrations or cases where the robot is working with deformable objects that do not emit perceptible vibrations upon contact.

## APPENDIX

self-attention block that follows the traditional transformer encoder structure of [46], except that we use pre-layernorm instead of post-layernorm. The self-attention block uses an embedding dimension of 512 , 8 attention heads, and an expansion ratio of 1 . The output of the transformer block is concatenated and passed to a 2-layer MLP with hidden dimensions of 512. We use a Dropout probability of 0 . 5 for all linear layers. The resultant network has around 20 M parameters. Setting h = 2 at inference strikes a balance between handling the non-Markovian nature of demonstrations and remaining reactive to changes in audio input.

## B. Training

All behavior cloning policies are trained with a batch size of 64 for a maximum of 100 epochs using early stopping with a patience of 15 epochs. We choose the model with the lowest validation loss for evaluation. Pre-trained parameters remain unfrozen during policy learning to mitigate the distribution shift between pre-training data and in-domain data. However, we perform an ablation in Section IV-E.1 demonstrating that keeping the AVID audio encoder frozen yields only slightly worse results and still outperforms the next best baseline for the zipping task. We apply image augmentations during training with probability 0 . 5 . When image augmentations are applied, we use PyTorch RandomCrop to size 224 and ColorJitter with the following parameters: brightness 0 . 3 , contrast 0 . 3 , saturation 0 . 1 , and hue 0 . 2 .

We use an Adam [47] optimizer and a cosine annealing learning rate scheduler with a starting learning rate of 0 . 001 . We train on a single GPU (3080Ti) which takes 1 -1 . 5 hours per model. For the BYOL-A baseline, we train a model for each task on the corresponding audio spectrograms for 100 epochs with a batch size of 1024 , a learning rate of 0 . 0003 , and the default settings for the network parameters and augmentations from [43].

## ACKNOWLEDGMENT

We thank Krishna Suresh, Raunaq Bhirangi, and Mohan Kumar for valuable discussion and feedback, as well as Shaden Naif Alshammari for early work with the contact microphones and Pedro Morgado for advice on using AVID. VD was supported by NSF GRFP and Siebel Scholars. We gratefully acknowledge the support of ONR MURI.

## A. Architecture

The policy takes as input 4 images and a single audio clip spanning a two-second window, resulting in 5 total tokens passed to the transformer. Learned positional encodings are applied to the audio and visual features. We use a single

## REFERENCES

- [1] S. Nair, A. Rajeswaran, V. Kumar, C. Finn, and A. Gupta, 'R3m: A universal visual representation for robot manipulation,' arXiv preprint arXiv:2203.12601 , 2022.
- [2] Y. J. Ma, S. Sodhani, D. Jayaraman, O. Bastani, V. Kumar, and A. Zhang, 'Vip: Towards universal visual reward and representation via value-implicit pre-training,' arXiv preprint arXiv:2210.00030 , 2022.
- [3] A. Majumdar, K. Yadav, S. Arnaud, Y. J. Ma, C. Chen, S. Silwal, A. Jain, V.-P. Berges, P. Abbeel, J. Malik et al. , 'Where are we in the search for an artificial visual cortex for embodied intelligence?' arXiv preprint arXiv:2303.18240 , 2023.
- [4] F. Ebert, Y. Yang, K. Schmeckpeper, B. Bucher, G. Georgakis, K. Daniilidis, C. Finn, and S. Levine, 'Bridge data: Boosting generalization of robotic skills with cross-domain datasets,' arXiv preprint arXiv:2109.13396 , 2021.
- [5] A. Kumar, A. Singh, F. Ebert, Y. Yang, C. Finn, and S. Levine, 'Pretraining for robots: Offline rl enables learning new tasks from a handful of trials,' arXiv preprint arXiv:2210.05178 , 2022.
- [6] H. Li, Y. Zhang, J. Zhu, S. Wang, M. A. Lee, H. Xu, E. Adelson, L. Fei-Fei, R. Gao, and J. Wu, 'See, hear, and feel: Smart sensory fusion for robotic manipulation,' arXiv preprint arXiv:2212.03858 , 2022.
- [7] K. Zhang, M. Sharma, M. Veloso, and O. Kroemer, 'Leveraging multimodal haptic sensory data for robust cutting,' in 2019 IEEE-RAS 19th International Conference on Humanoid Robots (Humanoids) . IEEE, 2019, pp. 409-416.
- [8] R. Calandra, A. Owens, D. Jayaraman, J. Lin, W. Yuan, J. Malik, E. H. Adelson, and S. Levine, 'More than a feeling: Learning to grasp and regrasp using vision and touch,' IEEE Robotics and Automation Letters , vol. 3, no. 4, pp. 3300-3307, 2018.
- [9] R. Calandra, A. Owens, M. Upadhyaya, W. Yuan, J. Lin, E. H. Adelson, and S. Levine, 'The feeling of success: Does touch sensing help predict grasp outcomes?' arXiv preprint arXiv:1710.05512 , 2017.
- [10] M. A. Lee, Y. Zhu, K. Srinivasan, P. Shah, S. Savarese, L. Fei-Fei, A. Garg, and J. Bohg, 'Making sense of vision and touch: Selfsupervised learning of multimodal representations for contact-rich tasks,' in 2019 International Conference on Robotics and Automation (ICRA) . IEEE, 2019, pp. 8943-8950.
- [11] A. Murali, Y. Li, D. Gandhi, and A. Gupta, 'Learning to grasp without seeing,' in International Symposium on Experimental Robotics . Springer, 2018, pp. 375-386.
- [12] A. Thankaraj and L. Pinto, 'That sounds right: Auditory selfsupervision for dynamic robot manipulation,' arXiv preprint arXiv:2210.01116 , 2022.
- [13] S. Clarke, T. Rhodes, C. G. Atkeson, and O. Kroemer, 'Learning audio feedback for estimating amount and flow of granular material,' Proceedings of Machine Learning Research , vol. 87, 2018.
- [14] P. Morgado, N. Vasconcelos, and I. Misra, 'Audio-visual instance discrimination with cross-modal agreement,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2021, pp. 12 475-12 486.
- [15] J. F. Gemmeke, D. P. Ellis, D. Freedman, A. Jansen, W. Lawrence, R. C. Moore, M. Plakal, and M. Ritter, 'Audio set: An ontology and human-labeled dataset for audio events,' in 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP) . IEEE, 2017, pp. 776-780.
- [16] D. Gandhi, A. Gupta, and L. Pinto, 'Swoosh! rattle! thump!-actions that sound,' arXiv preprint arXiv:2007.01851 , 2020.
- [17] S. Clarke, N. Heravi, M. Rau, R. Gao, J. Wu, D. James, and J. Bohg, 'Diffimpact: Differentiable rendering and identification of impact sounds,' in Conference on Robot Learning . PMLR, 2022, pp. 662673.
- [18] V. Dean, S. Tulsiani, and A. Gupta, 'See, hear, explore: Curiosity via audio-visual association,' Advances in Neural Information Processing Systems , vol. 33, pp. 14 961-14 972, 2020.
- [19] M. Du, O. Y. Lee, S. Nair, and C. Finn, 'Play it by ear: Learning skills amidst occlusion through audio-visual imitation learning,' arXiv preprint arXiv:2205.14850 , 2022.
- [20] C. Chen, S. Majumder, Z. Al-Halah, R. Gao, S. K. Ramakrishnan, and K. Grauman, 'Learning to set waypoints for audio-visual navigation,' arXiv preprint arXiv:2008.09622 , 2020.
- [21] W. Yuan, S. Dong, and E. H. Adelson, 'Gelsight: High-resolution robot tactile sensors for estimating geometry and force,' Sensors , vol. 17, no. 12, p. 2762, 2017.
- [22] M. Lambeta, P.-W. Chou, S. Tian, B. Yang, B. Maloon, V. R. Most, D. Stroud, R. Santos, A. Byagowi, G. Kammerer et al. , 'Digit: A novel design for a low-cost compact high-resolution tactile sensor with application to in-hand manipulation,' IEEE Robotics and Automation Letters , vol. 5, no. 3, pp. 3838-3845, 2020.
- [23] W. Li, J. Konstantinova, Y. Noh, Z. Ma, A. Alomainy, and K. Althoefer, 'An elastomer-based flexible optical force and tactile sensor,' in 2019 2nd IEEE International Conference on Soft Robotics (RoboSoft) . IEEE, 2019, pp. 361-366.
- [24] R. Bhirangi, T. Hellebrekers, C. Majidi, and A. Gupta, 'Reskin: versatile, replaceable, lasting tactile skins,' arXiv preprint arXiv:2111.00071 , 2021.
- [25] E. Donlon, S. Dong, M. Liu, J. Li, E. Adelson, and A. Rodriguez, 'Gelslim: A high-resolution, compact, robust, and calibrated tactilesensing finger,' in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) . IEEE, 2018, pp. 1927-1934.
- [26] B. Sundaralingam, A. S. Lambert, A. Handa, B. Boots, T. Hermans, S. Birchfield, N. Ratliff, and D. Fox, 'Robust learning of tactile force estimation through robot interaction,' in 2019 International Conference on Robotics and Automation (ICRA) . IEEE, 2019, pp. 9035-9042.
- [27] T. Bhattacharjee, A. Jain, S. Vaish, M. D. Killpack, and C. C. Kemp, 'Tactile sensing over articulated joints with stretchable sensors,' in 2013 World Haptics Conference (WHC) . IEEE, 2013, pp. 103-108.
- [28] R. Arandjelovic and A. Zisserman, 'Look, listen and learn,' in Proceedings of the IEEE International Conference on Computer Vision , 2017, pp. 609-617.
- [29] R. Arandjelovic and A. Zisserman, 'Objects that sound,' in Proceedings of the European conference on computer vision (ECCV) , 2018, pp. 435-451.
- [30] Y. Aytar, C. Vondrick, and A. Torralba, 'Soundnet: Learning sound representations from unlabeled video,' Advances in neural information processing systems , vol. 29, 2016.
- [31] H. Zhao, C. Gan, W.-C. Ma, and A. Torralba, 'The sound of motions,' in Proceedings of the IEEE/CVF International Conference on Computer Vision , 2019, pp. 1735-1744.
- [32] H. Chen, W. Xie, T. Afouras, A. Nagrani, A. Vedaldi, and A. Zisserman, 'Localizing visual sounds the hard way,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2021, pp. 16 867-16 876.
- [33] P. Morgado, I. Misra, and N. Vasconcelos, 'Robust audio-visual instance discrimination,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2021, pp. 12 934-12 945.
- [34] M. Patrick, Y. M. Asano, P. Kuznetsova, R. Fong, J. F. Henriques, G. Zweig, and A. Vedaldi, 'Multi-modal self-supervision from generalized data transformations,' arXiv preprint arXiv:2003.04298 , 2020.
- [35] I. Radosavovic, T. Xiao, S. James, P. Abbeel, J. Malik, and T. Darrell, 'Real-world robot learning with masked visual pre-training,' in Conference on Robot Learning . PMLR, 2023, pp. 416-426.
- [36] J. Pari, N. M. Shafiullah, S. P. Arunachalam, and L. Pinto, 'The surprising effectiveness of representation learning for visual imitation,' arXiv preprint arXiv:2112.01511 , 2021.
- [37] S. P. Arunachalam, S. Silwal, B. Evans, and L. Pinto, 'Dexterous imitation made easy: A learning-based framework for efficient dexterous manipulation,' arXiv preprint arXiv:2203.13251 , 2022.
- [38] K. He, X. Zhang, S. Ren, and J. Sun, 'Deep residual learning for image recognition,' in Proceedings of the IEEE conference on computer vision and pattern recognition , 2016, pp. 770-778.
- [39] K. Grauman, A. Westbury, E. Byrne, Z. Chavis, A. Furnari, R. Girdhar, J. Hamburger, H. Jiang, M. Liu, X. Liu et al. , 'Ego4d: Around the world in 3,000 hours of egocentric video,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2022, pp. 18 995-19 012.
- [40] V. Dean, D. K. Toyama, and D. Precup, 'Don't freeze your embedding: Lessons from policy finetuning in environment transfer,' in ICLR Workshop on Agent Learning in Open-Endedness , 2022. [Online]. Available: https://openreview.net/forum?id=HBHMrQD-LZc
- [41] G. Zhou, V. Dean, M. K. Srirama, A. Rajeswaran, J. Pari, K. Hatch, A. Jain, T. Yu, P. Abbeel, L. Pinto et al. , 'Train offline, test online: A real robot learning benchmark,' arXiv preprint arXiv:2306.00942 , 2023.
- [42] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, 'Diffusion policy: Visuomotor policy learning via action diffusion,' arXiv preprint arXiv:2303.04137 , 2023.
- [43] D. Niizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, 'Byol for audio: Self-supervised learning for general-purpose audio representation,' in 2021 International Joint Conference on Neural Networks (IJCNN) . IEEE, 2021, pp. 1-8.
- [44] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al. , 'Language models are few-shot learners,' Advances in neural information processing systems , vol. 33, pp. 1877-1901, 2020.
- [45] J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler et al. , 'Emergent abilities of large language models,' arXiv preprint arXiv:2206.07682 , 2022.
- [46] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, 'Attention is all you need,' Advances in neural information processing systems , vol. 30, 2017.
- [47] D. P. Kingma and J. Ba, 'Adam: A method for stochastic optimization,' arXiv preprint arXiv:1412.6980 , 2014.