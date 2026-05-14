## Spatial Reasoning from Natural Language Instructions for Robot Manipulation

Sagar Gubbi Venkatesh 1 , 2 , Anirban Biswas 3 , Raviteja Upadrashta 1 , Vikram Srinivasan 2 , Partha Talukdar 3 , and Bharadwaj Amrutur 1 , 2

Abstract -Robots that can manipulate objects in unstructured environments and collaborate with humans can benefit immensely by understanding natural language. We propose a pipelined architecture of two stages to perform spatial reasoning on the text input. All the objects in the scene are first localized, and then the instruction for the robot in natural language and the localized co-ordinates are mapped to the start and end co-ordinates corresponding to the locations where the robot must pick up and place the object respectively. We show that representing the localized objects by quantizing their positions to a binary grid is preferable to representing them as a list of 2D co-ordinates. We also show that attention improves generalization and can overcome biases in the dataset. The proposed method is used to pick-and-place playing cards using a robot arm.

## I. INTRODUCTION

This paper addresses the problem of spatial reasoning implicit in natural language instructions to the robot to move objects. Figure 1 illustrates a simple model that is representative of this problem. In this example, the objects are the playing cards. The text instruction is of the form 'Line up card 3 squarely above card 6'. The robot needs to use this along with the visual input from the camera to infer the start co-ordinate from where the robot must pick up the object and the end co-ordinate where the object must be placed.

One approach is to use an end-to-end network that takes both the image from the camera and the text instruction and directly predicts the physical locations like the start and end co-ordinates. But such a network must implicitly learn to detect and localize objects, which can be difficult to do from a small dataset. Alternatively, the image from the camera feed can be processed by a separately trained object detector to identify and localize all the objects in the image. The positions of all the objects along with the natural language instruction are then used by a language network to predict the start and end co-ordinates[1]. The approach in [1] represents the output of the object detector as a list of 2D coordinates indicating the position of the objects in the scene

1 Robert Bosch Center for Cyber Physical Systems, Indian Institute of Science, Bangalore 560012, India sagar@iisc.ac.in ; ravitejaupadras@iisc.ac.in ; amrutur@iisc.ac.in

2 Department of Electrical and Communication Engineering, Indian Institute of Science, Bangalore 560012, India vikram.srinivasan@gmail.com ;

3 Department of Computational and Data Sciences, Indian Institute of Science, Bangalore 560012, India anirbanb@iisc.ac.in ; ppt@iisc.ac.in

**This work was supported by Robert Bosch Center for Cyber-Physical Systems, Indian Institute of Science.

Fig. 1. The robot and the cards it must manipulate. The top view of the scene in front of the robot captured from an overhead camera is processed with an object detector to localize all the cards. Based the natural language instruction and the positions of all the cards, the task is to predict the position from which the robot must pick up the card ( start co-ordinate) and the location at which the card must be placed ( end co-ordinate).

<!-- image -->

(see Lang-FCNet in Fig. 2(i)). However, this representation has shortcomings which results in poor performance. Hence, we propose an alternative representation for the output of the object detector.

To see why representing the object positions as a list can be sub-optimal, consider the problem of finding 'the second card in a row of cards'. If fully connected layers are used to process the coordinate list based output of the object detector[1], the network can overfit to specific locations of the row of cards in the training set, and the network structure is not inherently conducive to generalizing to different positions of the row of cards on the table. To address this, we propose representing the output of the object detector as a binary 2D image with each lit pixel corresponding to an object and using a convolutional network to predict the start and end positions (see Lang-UNet in Fig. 2(ii)). With this, we expect improved generalization because the convolution operation is, by construction, spatially invariant.

Our contributions:

- Object representation in a 2D binary grid via a preprocessing network: We experiment with two different spatial representations and show that instead of representing the localized objects as a list of 2D co-ordinates

Fig. 2. Two architectures for language conditioned localization which use different representations for the output of the object detector.

<!-- image -->

and processing them with fully connected layers, the spatial reasoning can be improved by representing the detected objects on a 2D binary grid and using a convolutional U-Net to predict the pixels on the grid corresponding to the start and end positions.

- Multi-head attention and visual grounding: We show that a recurrent network that generates attention for visual grounding generalizes better than a network without attention and can overcome biases in the training dataset.

We train the proposed neural network using the blocks dataset[2] and demonstrate the proposed method by deploying it on the Dobot Magician robot arm to pick-and-place playing cards based on text instructions. Our code is opensource 1 .

## II. RELATED WORK

Research on manipulating robots using natural language instructions has gained significant interest. The entire body of related work can be broadly categorized into end-toend approaches and pipelined approaches. In the end-toend approaches [3], [4], [5], [6], [7], [8] the robotic agent simultaneously interacts with the surrounding environment while executing natural language instructions and takes a sequence of actions to fulfil its goal. In contrast, the line of work that follows the pipelined approach, [9], [10], [11], [12] breaks up the task into navigation planning and language grounding processes. Our work is similar to [1], [13], that used neural models to localise the scene objects and to ground the spatial relations in unrestricted natural language instructions into a blocks world[2] with complex goal configurations. While their work suffers from poor generalization, we use the attention mechanism [14] to solve the problem in a natural way. Recent work in reinforcement learning has demonstrated the usefulness of representing the state information as pixels in a 2D image[15] instead of a list of numbers, which is similar in spirit to this paper although the actual representation is different.

Earlier works on Human-Robot Interaction have focused on converting a language command with restricted vocabulary and simple actions into a structural form easily understandable by an agent to execute it [16][17][18]. Reinforcement learning based techniques[19][20] have also been explored for the instruction-following task. The limited action space and little diversity in the language instructions have proven to be non-robust, especially when the instructions are generated by non-experts. Our work addresses these challenges and is able to handle the underlying diversity in the language.

Vision and language grounding are the two important components for an effective human-robot communication through natural language instructions in the context of the surrounding world view. Grounding visual inputs has proven to be extremely essential to many vision tasks like image captioning [21][22], visual question-answering[23], embodied question-answering[24] and vision-languagenavigation[7][25][4][26][6]. A surfeit of work has been done on grounding natural language instructions [20][19][27] using variety of techniques like semantic and syntactic parsing[28][29], alignment models[30]. Improving language understanding using human-robot dialog[31][32] and commonsense reasoning[33] as well as generation of unambiguous spatial-referring expressions[34] have also been explored. These papers explore different aspects of grounding natural language in vision, whereas our work focuses on spatial reasoning from natural language instructions.

## III. PROBLEM STATEMENT

Given a natural language instruction with embedded spatial cues and an image of the world view, the goal is to understand the instruction in the context of world view and to act in accordance with the spatial cues. For the pick-andplace task, the robot must move to the location where the desired object is present, pick it up, and then place it at the goal/target location. We present a few examples from the datasets in Sec. V-A:

- Move block 5 from the top right of box 11 to above box 14 in the middle with a small space.
- Place block 5 one and a half columns to the right of block 18.
- Pick the first apple from row number one.
- Many oranges are placed at random. Pick the biggest orange.

Even though the first two expressions mentioned above differ considerably in their language form, they refer to the same world view and instruct to perform the same action. The model must be robust enough to discern the start block ( block 5 in first two examples), choose the correct target anchor (e.g. block 14 and not block 11, in the first example), recognize the notion of direction (e.g. right of block 18 or above block

Fig. 3. The Lang-UNet model takes the instruction text T and the detected objects X along with their sizes S to predict the start and end co-ordinates. glyph[circledot] represents pixel-wise correlation. The U-Net has four Conv5x5(128)-ELU layers followed by four transposed convolutions of the same size and a bottleneck Conv1x1(2) layer. Section IV explains the network in more detail.

<!-- image -->

14) and ground the distance information (e.g. one and a half column to the right of block 18). The last two representative instructions test the model's ability to understand abstract concepts (e.g. top row ), reason about object size (e.g. biggest orange), ordinality (e.g. fi rst apple) and cardinality (e.g. row number one ).

## IV. NETWORK ARCHITECTURE

Our proposed language network Lang-UNet (Fig. 3) takes as input a natural language instruction and the object positions and sizes from the object detector and finally predicts the start ( x s , y s ) and end co-ordinates ( x e , y e ) . The robot picks the object from start location and places it at the end location.

The object positions are represented as a binary image X ∈ { 0 , 1 } W × H × N o where N o is the number of distinct objects. Each object in the scene is represented as a pixel in X by a one-hot vector corresponding to the type of the object. The sizes of the objects are represented in an image S ∈ R W × H with S ij = 0 if there is no object at ( i, j ) .

The instruction text T is tokenised with minimal preprocessing (lowercase words, removed punctuation) into a sequence of tokens { t i } N T i =1 and fed into an embedding layer to obtain a vector representation e i ∈ R D for each token t i . Note that is possible to use BERT[35] to obtain embeddings for the tokens in the instruction, but we chose to learn embeddings from random initialization for easier comparison with a number of previous works, which highlights the benefit of our proposed representation of the object positions and sizes. The token embeddings { e i } N T i =1 are passed through a 2-layer Bi-directional LSTM network. The encoded vector outputs are then passed through a 1-D convolutional network with softmax activation to obtain the attention energies for four attention heads. We denote a j i the attention value for i th token and j th attention head. The instruction embeddings are computed as follows:

<!-- formula-not-decoded -->

The first two instruction embeddings are projected to R N o using two separate fully connected layers to obtain ˜ e 1 and ˜ e 2 . Each pixel of X is correlated with ˜ e 1 to obtain U 1 ∈ R W × H (and likewise U 2 is obtained). These two embeddings 'softselect' appropriate objects in the scene while suppressing the rest.

<!-- formula-not-decoded -->

The embeddings ˆ e 3 and ˆ e 4 indicate attributes such as spatial relationships referred to in the instruction. They are repeated W × H times and appended to U 1 , U 2 , and S to get V = [ U 1 ; U 2 ; S ; ˆ e 3 ; ˆ e 4 ] . The image V is passed through the convolutional hourglass network (U-Net) as shown in Fig. 3 to obtain O ∈ R W × H × 2 . The start location ( x s , y s ) and the end location ( x e , y e ) are extracted from O 1 and O 2 respectively by passing it through a spatial-softmax layer.

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where 1 ≤ i ≤ W and 1 ≤ j ≤ H .

Note that the U-Net structure has no notion of which object is at a particular position (notice that it's input size is independent of N o ). It is only aware that a particular object 'selected' by the BiLSTM layers via ˜ e 1 or ˜ e 2 is present at a location (Eqn. 2). This ensures that the U-Net learns only spatial relationships and not anything specific to an object. So, if the network has learnt to find the position of 'an apple to the left of the banana', it will generalize to 'an orange to the left of the banana'.

## V. EXPERIMENTAL RESULTS

We first evaluate the language network separately on two different datasets. Subsequently, we discuss the performance of the entire pipeline on a real robot arm.

## A. Datasets for the Language Network

To evaluate the language network, we assume that the object positions and sizes are known. We have experimented with two datasets. We use the publicly available Blocks dataset [2]. Additionally we synthesize a diagnostic dataset to test our model performance for more diverse and complicated visual scenarios. We briefly explain the datasets as follows:

Blocks 2D: In the blocks dataset[2], each sample has a natural language instruction and the positions of all 20 blocks as the input and the labels are the position from which the block must be picked up ( start position) and the location where the block must be placed ( end position). A sample instruction: 'Pick up block 9 and place it above block 8'. The corpus has a training / development / test distribution of 3712 / 699 / 705 instructions. We allow the start and end positions to be real valued locations and do not restrict them to be on the grid, but unlike [36] which considers 3D block structures, we retain the 2D world. We include only those instructions in the blocks dataset that have fewer than 40 tokens in the instruction.

Synthetic Dataset: The Blocks dataset has a few limitations: (a) All the blocks are uniquely numbered and only one instance of each block is in the scene, (b) In most of the cases, the instructions are such that the goal location can be obtained by finding the appropriate anchor block and a relative offset direction from a predefined set, and (c) The sizes of the blocks are identical. To diagnose whether the proposed model is capable of reasoning about a variety of other spatial relationships, object attributes (e.g. size), abstract concepts (e.g. row or column) as well as scenes with multiple instances of each object, we build a synthetic dataset. We follow a similar approach proposed in [37] and generate ∼ 42,000 unique instructions with varied scenes containing objects of sizes randomly chosen between 1.0 to 3.0 and divide it into train / dev / test distribution of 29465 / 4216 / 8416. Each scene contains a maximum of 12 distinct objects and up to a total of 24 objects. Some of the representative templates used to generate instructions are as follows: (i) Pick the largest / smallest 〈 obj 〉 , (ii) Pick the leftmost / rightmost 〈 obj 〉 from the row of 〈 obj 〉 s , (iii) Pick the 〈 obj pos 〉 〈 obj 〉 from top row , (iv) Pick the 〈 obj1 〉 above / below / to the left of / to the right of 〈 obj2 〉 . For this dataset, we predict only the start location and ignore the end location.

## B. Evaluation Metrics for the Language Network

We define two evaluation metrics to compare the baseline results quantitatively with ours.

Mean Squared Error (MSE): It is the average over the squared distances between the gold and predicted locations. Define the prediction and gold locations for i th instruction as L P and L G respectively. Then MSE = ∑ N i =1 ‖ L i P -L i G ‖ 2 N .

Fig. 4. Sample predictions of location of start , anchor block (with blue and green squares), and end location (green spot) from the Lang-UNet for the Blocks dataset.

<!-- image -->

Fig. 5. Sample predictions of Lang-UNet for the synthetic dataset. The network predicts only the ( x, y ) location, but to aid visualization, the prediction is represented as a green square box. In the last example, the difference in the sizes of the cucumbers is small (the sizes are randomly chosen when synthesizing the dataset), which causes the network to incorrectly predict the location.

<!-- image -->

The center of the simulated world is at (0 , 0) and restricted in the range of [ -1 , 1] in both x and y directions.

Tolerable Accuracy (TA): In majority of the real world applications, it is acceptable even if the predicted and the target locations do not exactly match but the distance between them is within a certain tolerable (applicationspecific) range. To account for this fact, we propose a new metric Tolerable Accuracy . A prediction is considered to be correct if the distance error (in both x and y directions in simulated world) is less than a tolerance value tol . We count the number of correct prediction instances out of the total N instructions to evaluate TA. Mathematically, TA = ∑ N i =1 1 {| L i P -L i G | x,y &lt;tol } N . 1 { z } is an indicator function and it's value is 1 when z is true , otherwise 0 .

## C. Baseline Algorithms for the Language Network

We compare our proposed model with the following baseline algorithms:

Center: This model assumes complete knowledge about the start location and places the block at the middle of the table.

Random: The Random baseline decides both the start and end locations to be random. The Center and Random are two simple baselines taken from [1].

LSTM [37]: The word embeddings of the instructions obtained from an embedding layer is passed through a wordlevel multi-stage LSTM model. The output from the last layer of LSTM is the instruction embedding and passed through a MLP to obtain the final outputs. This model does not use the image information, hence can identify object positions only through the presence of any bias in the instructions. We included this baseline to show that the dataset is not biased in such a way that the instruction alone can predict the start and end positions.

LSTM+CNN [37]: This is an end-to-end approach that takes the image and the text instruction as input and directly predicts the start and end positions. As above, the instruction embedding is obtained from the last-layer hidden state of LSTM and the image is encoded using CNN features. They are concatenated and fed into a MLP to get the prediction of the start and end locations.

LSTM+CNN+SA [37]: The encodings for image and instruction are generated as above and then soft spatial attention is employed to get the final representation. A MLP takes this as input and produces the required output.

LSTM+UNet [7]: The Ling-UNet[7] is an end-to-end architecture that takes the RGB pixels from the camera and the instruction text and directly regresses the start and end positions. This is similar to the LSTM+CNN architecture above, but a UNet is used instead of the CNN and vectors derived from the instruction text using the LSTM are convolved with skip connection activations of the UNet. Similar to the proposed architecture, the start and end positions are extracted from the last layer of the UNet using the spatialsoftmax layer.

RNN-NoAttn-NoGround [1]: The model architecture has a single-layer RNN at its heart. It takes as input the instruction and predicts the start object, anchor object, and chooses from 8 pre-defined offsets corresponding to the 8 adjacent positions (right, bottom-right, etc.). The start position prediction is simply the position of the predicted object to be picked up. The end position is obtained by adding the position of the predicted anchor object and the predicted offset. Note that this model is not grounded because the predictions of the start , anchor, and offset are invariant to the positions of the objects in the scene and depend only on the natural language instruction. Furthermore, it cannot distinguish between two or more instances of the same object in the scene. For fair comparison, we train this model on our dataset.

LangNet-Attn-NoGround: We extend the model in Bisk et. al [1] by introducing an attention mechanism over a multilayer BiLSTM.

Lang-FCNet: This model differs from the proposed LangUNet model in that object positions are represented as a list of 2D points rather than as a binary image, and they are passed through fully connected layers rather than the convolutional hourglass network (U-Net).

## D. Results for Language Network

Better generalisation with Attention: The language network is trained on the blocks dataset using the Adam optimizer with mean absolute error loss, with learning rate 1e-3, and weight decay 1e-9. A few sample predictions of the Lang-UNet (BiLSTM model with attention) model are visualized in Figs. 4 and 5. For one example, the attention weights for the different tokens of the instruction when predicting the start , anchor, and offsets are shown in Fig. 6. Table I compares the performance of the proposed approach with the baselines. We observe that the attention-based models LSTM-Attn-NoGround and Lang-UNet perform significantly better than the RNN-NoAttn-NoGround model that has no attention component, especially for end co-ordinate prediction. Fig. 6 shows the attention mechanism is able to attend on the correct offset and target block and intuitively explains the reason behind improved performance for end co-ordinate prediction. Note that the accuracy of predicting the end location is worse than for the start location. This is because in most textual instructions, the start location is simple and unambiguous ('place block 4...' or 'pick up block 3...'), whereas the target is more complex (last example in Fig. 4) and sometimes ambiguous ('the 14th block moved next to the 12th block'). It also suggests why attention is more important for predicting the end location than the start in case of Blocks dataset.

Fig. 6. Visualization of attention in Lang-UNet . The darker colors indicate higher attention weights for that token. Two attention heads indicate the Offset ( ˆ e 3 and ˆ e 4 ) and the other two for the objects corresponding to the start and end predictions ( ˆ e 1 and ˆ e 2 ).

<!-- image -->

Mitigating the effect of bias through Attention: We noticed that the Blocks dataset is biased with some block numbers more frequently being associated with some offsets (such as 'north of') than others. Because of this, the RNNNoAttn-NoGround model overfits and always predicts the same offset when some block numbers are present in the instruction and ignore the actual content of the text. In contrast, the LSTM-Attn-NoGround model and the LangUNet are forced to attend to the offset token in the instruction and gets it right. For example, all the models correctly predict the output for 'Move block 4 above block 5'. But, when the block number is changed to 'Move block 11 above block 5', only Lang-UNet and LSTM-Attn-NoGround make the correct prediction. To quantify this, we selected 20 simple examples such as the above example from the validation set. The LangUNet , RNN-NoAttn-NoGround models correctly predicted 19 and 19 examples respectively. But when we randomized the block numbers in those instructions, the number of correct predictions were 19 and 6 respectively. Attention was necessary to retain performance and demonstrates its usefulness in such biased datasets.

Visual grounding is essential for diverse and complex data: We note that the ungrounded models which use the natural language instruction alone and do not use object po- COMPARISON OF THE PERFORMANCE OF VARIOUS BASELINES AND OUR MODELS. IN ALL THE EXPERIMENTS AND MODELS, tol IS SET TO 0.05 TO MEASURE TA. THE Baselines ARE EXPLAINED IN SEC. V-C AND Lang-UNet IN SEC. IV. WE DISCUSS MORE ABOUT THE RESULTS IN SECTION V-D.

TABLE I

|           | Blocks 2D               | Blocks 2D   | Blocks 2D   | Blocks 2D   | Blocks 2D   | Synthetic   | Synthetic   |
|-----------|-------------------------|-------------|-------------|-------------|-------------|-------------|-------------|
|           | Model                   | Model       | Model       | End         | End         | Start       | Start       |
|           |                         | MSE         | TA (%)      | MSE         | TA (%)      | MSE         | TA (%)      |
|           | Center                  | 0.3130      | 1.28        | 0.3130      | 1.28        | 0.6018      | 0.02        |
|           | Random                  | 1.4379      | 0.28        | 1.0118      | 0.14        | 1.2539      | 0.25        |
| Baselines | LSTM                    | 1.1752      | 0.28        | 0.3026      | 1.70        | 0.1694      | 2.82        |
|           | LSTM+CNN                | 0.8501      | 0.00        | 0.3154      | 0.14        | 0.0302      | 28.92       |
|           | LSTM+CNN+SA             | 1.0789      | 0.00        | 0.3706      | 4.40        | 0.0125      | 44.89       |
|           | LSTM+UNet [7]           | 0.1502      | 26.24       | 0.1436      | 23.55       | 0.0496      | 14.50       |
|           | RNN-NoAttn-NoGround [1] | 0.0067      | 96.88       | 0.0392      | 54.33       | 0.0619      | 2.89        |
|           | LSTM-Attn-NoGround      | 0.0084      | 98.51       | 0.0306      | 66.85       | 0.0849      | 9.49        |
| Ours      | Lang-FCNet              | 0.0109      | 97.73       | 0.0713      | 61.56       | 0.0145      | 59.88       |
|           | Lang-UNet               | 0.0097      | 97.00       | 0.0491      | 60.07       | 0.0031      | 95.99       |

sitions perform reasonably on the Blocks dataset. However, they perform poorly on the synthetic dataset because it is not possible to predict the correct position for an instruction such as 'Pick the apple to the left of the orange' without actually using the object positions. The proposed LangUNet peforms well on both the Blocks and the synthetic datset, but slightly underperforms RNN-NoAttn-NoGround on the Blocks dataset due to the quantization in representing the object positions. Moreover, the LSTM-Attn-NoGround and RNN-NoAttn-NoGround models use hard-coded offsets ( { 0 , 0 . 166 , -0 . 166 } in both X and Y directions) that are added to the position of the anchor object to predict the end position and are thus specialized to the Blocks dataset, whereas the proposed LangUNet does not use such hardcoded offsets.

Benefits of Grid representation over List: The LangFCNet takes the output from the localisation network as a list of 2D points, whereas the Lang-UNet considers a 2D binary grid representation as explained in Sec. IV. Fig. 2 depicts the differences between the output formats from the localisation network. On examples such as 'Pick up the banana from the row of bananas', Lang-UNet is successful in predicting the location of the banana because the convolutional layers in the U-Net help in recognizing 'rows', whereas LangFCNet performs poorly. From Table I, we infer that the performance improvement in Lang-UNet over Lang-FCNet , particularly for the synthetic dataset, is due to the binary grid representation because it provides the model a better way to understand the object positions and the relative spatial relationships amongst each other compared to a list of 2D coordinates. Our empirical evaluation in Table I also suggests the superiority of the pipelined approaches ( Lang-FCNet and Lang-UNet ) over the end-to-end models ( LSTM+CNN and LSTM+CNN+SA ).

## E. Demonstration on the Robot Arm

We demonstrate the complete pipeline using a Dobot Magician robot arm (Fig. 1). Playing cards are placed at random positions in front of the robot on the table. A camera mounted overhead captures the the top view of the table. The position of the cards is obtained using an object detector[38] that is fine tuned to detect playing cards. Based on the instruction and the positions of the playing cards on the table, the robot picks-and-places a card. Note that while we can switch out the playing cards and instead detect other objects, it is necessary to know a priori what objects will be referenced in the instruction text and to fine-tune the object detector to detect those objects. Out of 15 trials with playing cards, the robot successfully picks the right card in all the trials. In 14 cases, the card is placed within 1 cm of the target. In one case, the localization of the anchor is off by more than 1 cm. A video of the robot in operation is available at https://youtu.be/bfmDC-zoCFc .

## VI. CONCLUSIONS

In this paper, we have illustrated the advantages of a pipelined approach to manipulating objects based on natural language instructions. We propose having a separately trained object detector followed by a language network that is responsible for predicting the start and end positions to pick-and-place objects based on the natural language instruction. We show that representing the positions of the detected objects on a 2D binary grid and processing them with a convolutional hourglass network results in much better performance than representing them as a list of 2D coordinates and processing with fully connected layers. We also show that attention improves the generalization, especially when the training data is biased.

## ACKNOWLEDGMENT

We would like to thank the Robert Bosch Center for CyberPhysical Systems for funding support.

## REFERENCES

- [1] Y. Bisk, D. Yuret, and D. Marcu, 'Natural language communication with robots,' in Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies , San Diego, California, June 2016.
- [2] Y. Bisk, D. Marcu, and W. Wong, 'Towards a dataset for human computer communication via grounded language acquisition,' in Workshops at the Thirtieth AAAI Conference on Artificial Intelligence , 2016.
- [3] D. Misra, J. Langford, and Y. Artzi, 'Mapping instructions and visual observations to actions with reinforcement learning,' in Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing , Copenhagen, Denmark, Sept. 2017.
- [4] P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. S¨ underhauf, I. Reid, S. Gould, and A. van den Hengel, 'Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments,' in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2018.
- [5] X. Wang, W. Xiong, H. Wang, and W. Yang Wang, 'Look before you leap: Bridging model-free and model-based reinforcement learning for planned-ahead vision-and-language navigation,' in Proceedings of the European Conference on Computer Vision (ECCV) , 2018, pp. 37-53.
- [6] D. Fried, R. Hu, V. Cirik, A. Rohrbach, J. Andreas, L.-P. Morency, T. Berg-Kirkpatrick, K. Saenko, D. Klein, and T. Darrell, 'Speakerfollower models for vision-and-language navigation,' in Advances in Neural Information Processing Systems , 2018, pp. 3314-3325.
- [7] D. Misra, A. Bennett, V. Blukis, E. Niklasson, M. Shatkhin, and Y. Artzi, 'Mapping instructions to actions in 3D environments with visual goal prediction,' in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing , Brussels, Belgium, Oct.-Nov. 2018.
- [8] V. Blukis, R. A. Knepper, and Y. Artzi, 'Few-shot object grounding and mapping for natural language robot instruction following,' arXiv preprint arXiv:2011.07384 , 2020.
- [9] J. Arkin, M. R. Walter, A. Boteanu, M. E. Napoli, H. Biggie, H. KressGazit, and T. M. Howard, 'Contextual awareness: Understanding monologic natural language instructions for autonomous robots,' in 2017 26th IEEE International Symposium on Robot and Human Interactive Communication (RO-MAN) . IEEE, 2017, pp. 502-509.
- [10] R. Paul, J. Arkin, N. Roy, and T. M Howard, 'Efficient grounding of abstract spatial concepts for natural language interaction with robot manipulators,' Robotics: Science and Systems Foundation , 2016.
- [11] S. Tellex, T. Kollar, S. Dickerson, M. R. Walter, A. G. Banerjee, S. Teller, and N. Roy, 'Understanding natural language commands for robotic navigation and mobile manipulation,' in Twenty-fifth AAAI conference on artificial intelligence , 2011.
- [12] D. Arumugam, S. Karamcheti, N. Gopalan, L. L. Wong, and S. Tellex, 'Accurately and efficiently interpreting human-robot instructions of varying granularities,' Robotics: Science and Systems Foundation , 2017.
- [13] B. Piˇ sl and D. Mareˇ cek, 'Communication with robots using multilayer recurrent networks,' in Proceedings of the First Workshop on Language Grounding for Robotics , 2017, pp. 44-48.
- [14] D. Bahdanau, K. Cho, and Y. Bengio, 'Neural machine translation by jointly learning to align and translate,' ICLR , 2015.
- [15] M. Laskin, K. Lee, A. Stooke, L. Pinto, P. Abbeel, and A. Srinivas, 'Reinforcement learning with augmented data,' arXiv preprint arXiv:2004.14990 , 2020.
- [16] V. Klingspor, J. Demiris, and M. Kaiser, 'Human-robot communication and machine learning,' Applied Artificial Intelligence , vol. 11, no. 7, pp. 719-746, 1997.
- [17] N. Mavridis, 'A review of verbal and non-verbal human-robot interactive communication,' Robotics and Autonomous Systems , vol. 63, pp. 22-35, 2015.
- [18] J. Dzifcak, M. Scheutz, C. Baral, and P. Schermerhorn, 'What to do and how to do it: Translating natural language directives into temporal and dynamic logic representation for goal management and action execution,' in 2009 IEEE International Conference on Robotics and Automation . IEEE, 2009, pp. 4163-4168.
- [19] A. Vogel and D. Jurafsky, 'Learning to follow navigational directions,' in Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics , 2010, pp. 806-814.
- [20] S. R. Branavan, H. Chen, L. S. Zettlemoyer, and R. Barzilay, 'Reinforcement learning for mapping instructions to actions,' in Proceedings of the Joint Conference of the 47th Annual Meeting of the ACL and the 4th International Joint Conference on Natural Language Processing of the AFNLP: Volume 1-Volume 1 , 2009, pp. 82-90.
- [21] R. Hu, H. Xu, M. Rohrbach, J. Feng, K. Saenko, and T. Darrell, 'Natural language object retrieval,' in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , 2016, pp. 4555-4564.
- [22] J. Lu, J. Yang, D. Batra, and D. Parikh, 'Neural baby talk,' in Proceedings of the IEEE conference on computer vision and pattern recognition , 2018, pp. 7219-7228.
- [23] A. Agrawal, D. Batra, D. Parikh, and A. Kembhavi, 'Don't just assume; look and answer: Overcoming priors for visual question answering,' in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , 2018, pp. 4971-4980.
- [24] A. Das, S. Datta, G. Gkioxari, S. Lee, D. Parikh, and D. Batra, 'Embodied question answering,' in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops , 2018, pp. 2054-2063.
- [25] P. Anderson, A. Shrivastava, D. Parikh, D. Batra, and S. Lee, 'Chasing ghosts: Instruction following as bayesian state tracking,' arXiv preprint arXiv:1907.02022 , 2019.
- [26] C.-Y. Ma, J. Lu, Z. Wu, G. AlRegib, Z. Kira, R. Socher, and C. Xiong, 'Self-monitoring navigation agent via auxiliary progress estimation,' ICLR , 2019.
- [27] H. Mei, M. Bansal, and M. R. Walter, 'Listen, attend, and walk: Neural mapping of navigational instructions to action sequences,' in Thirtieth AAAI Conference on Artificial Intelligence , 2016.
- [28] Y. Artzi and L. Zettlemoyer, 'Weakly supervised learning of semantic parsers for mapping instructions to actions,' Transactions of the Association for Computational Linguistics , vol. 1, 2013.
- [29] M. MacMahon, B. Stankiewicz, and B. Kuipers, 'Walk the talk: Connecting language, knowledge, and action in route instructions,' Association for the Advancement of Artificial Intelligence (AAAI) , 2006.
- [30] J. Andreas and D. Klein, 'Alignment-based compositional semantics for instruction following,' in Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing , Lisbon, Portugal, Sept. 2015.
- [31] M. Shridhar and D. Hsu, 'Interactive visual grounding of referring expressions for human-robot interaction,' arXiv preprint arXiv:1806.03831 , 2018.
- [32] J. Thomason, A. Padmakumar, J. Sinapov, N. Walker, Y. Jiang, H. Yedidsion, J. Hart, P. Stone, and R. J. Mooney, 'Improving grounded natural language understanding through human-robot dialog,' in 2019 International Conference on Robotics and Automation (ICRA) . IEEE, 2019, pp. 6934-6941.
- [33] H. Chen, H. Tan, A. Kuntz, M. Bansal, and R. Alterovitz, 'Enabling robots to understand incomplete natural language instructions using commonsense reasoning,' arXiv preprint arXiv:1904.12907 , 2019.
- [34] F. I. Do˘ gan, S. Kalkan, and I. Leite, 'Learning to generate unambiguous spatial referring expressions for real-world environments,' arXiv preprint arXiv:1904.07165 , 2019.
- [35] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, 'Bert: Pre-training of deep bidirectional transformers for language understanding,' arXiv preprint arXiv:1810.04805 , 2018.
- [36] Y. Bisk, K. Shih, Y. Choi, and D. Marcu, 'Learning interpretable spatial operations in a rich 3d blocks world,' in Proceedings of the AAAI Conference on Artificial Intelligence , vol. 32, no. 1, 2018.
- [37] J. Johnson, B. Hariharan, L. van der Maaten, L. Fei-Fei, C. Lawrence Zitnick, and R. Girshick, 'Clevr: A diagnostic dataset for compositional language and elementary visual reasoning,' in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , 2017, pp. 2901-2910.
- [38] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg, 'Ssd: Single shot multibox detector,' in European conference on computer vision . Springer, 2016, pp. 21-37.