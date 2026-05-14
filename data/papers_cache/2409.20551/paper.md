## UniAff: A Unified Representation of Affordances for Tool Usage and Articulation with Vision-Language Models

Qiaojun Yu 1 ∗ , Siyuan Huang 1 , 7 ∗ , Xibin Yuan 1 , Zhengkai Jiang 2 , Ce Hao 3 , Xin Li 1 , Haonan Chang 4 , Junbo Wang 1 , Liu Liu 5 , Hongsheng Li 6 , Peng Gao 7 B and Cewu Lu 1 B

Abstract -Previous studies on robotic manipulation are based on a limited understanding of the underlying 3D motion constraints and affordances. To address these challenges, we propose a comprehensive paradigm, termed UniAff , that integrates 3D object-centric manipulation and task understanding in a unified formulation. Specifically, we constructed a dataset labeled with manipulation-related key attributes, comprising 900 articulated objects from 19 categories and 600 tools from 12 categories. Furthermore, we leverage MLLMs to infer objectcentric representations for manipulation tasks, including affordance recognition and reasoning about 3D motion constraints. Comprehensive experiments in both simulation and real-world settings indicate that UniAff significantly improves the generalization of robotic manipulation for tools and articulated objects. We hope that UniAff will serve as a general baseline for unified robotic manipulation tasks in the future. Images, videos, dataset and code are published on the project website at:https://sites.google.com/view/uni-aff/home.

## I. INTRODUCTION

Mastering the manipulation of tools and articulated objects is crucial for embodied robots, which requires understanding physical constraints and interaction regions in 3D space [1, 2]. For effective manipulation, identifying movable parts, joint types, 3D joint parameters, and affordances in articulated objects is essential [3, 4]. Similarly, modeling 6D pose, grasping regions and functional areas is vital for tool use in specific tasks [5, 6]. By integrating task-related 6D pose, 3D motion constraints, and affordances predictions, embodied robots can further enhance their adaptability and efficiency.

Most existing approaches focus exclusively on either articulated objects [4, 7, 8] or tools [9-11], which limits their ability to generalize across tasks. By leveraging the reasoning capabilities of large language models (LLMs), two-stage methods are employed: the first stage predicts manipulationrelated parameters using a vision model [2, 12], while the

1 Qiaojun Yu, Siyuan Huang, Xibin Yuan, Xin Li, Junbo Wang, Cewu Lu are with the Shanghai Jiao Tong University, China. 2 Zhengkai Jiang is with Hong Kong University of Science and Technology, HongKong. 3 Ce Hao is with National University of Singapore, Singapore. 4 Haonan Chang is with Rutgers University, United States of America. 5 Liu Liu is with Hefei University of Technology, China. 6 Hongsheng Li is with CUHKMMLab, China. 7 Siyuan Huang and Peng Gao are with the Shanghai AI Lab, China. * indicates the equal contribution. B Peng Gao and Cewu Lu are the equal corresponding authors, gaopeng@pjlab.org.cn, lucewu@sjtu.edu.cn .

Acknowledgement . This work was supported in part by the National Natural Science Foundation of China under Grant (62302143), the Anhui Provincial Natural Science Foundation under Grant (2308085QF207), the National Key Research and Development Project of China (2022ZD0160102), the Shanghai Artificial Intelligence Laboratory, and XPLORER PRIZE grants (2021ZD0110704).

Fig. 1. UniAff demonstrates its ability to unify tool usage and articulation understanding in a VQA format, predicting part bounding boxes, 6D poses, grasp affordances, functional affordances, and manipulation types, etc for effective robotic manipulation tasks.

<!-- image -->

second stage utilizes the LLM's reasoning. Additionally, they address case-by-case problems without offering general task reasoning capabilities. To overcome these limitations, we propose UniAff, a unified representation of affordances for both tools and articulated objects, powered by visionlanguage models, as illustrated in Figure 1. UniAff combines object-centric manipulation with task understanding, utilizing multimodal large language models (MLLMs) to improve comprehension of 3D motion constraints and affordances.

For the training of UniAff, We developed a comprehensive dataset for articulated object manipulation and tool-use tasks, including 900 articulated objects over 19 categories and 600 tools in 12 categories. Each object's part-level 6D pose, grasp affordances, manipulation types, and functional affordances are labeled, forming a multifunctional dataset for robot learning. By unifying the formulation of manipulation tasks, we incorporate object-centric 3D motion constraints and affordances. Near-realistic simulations generate pre-scanned meshes or URDF models [13], efficiently producing a largescale dataset with automatically labeled information.

To equip the MLLM with unified affordance capabilities, we fine-tune the SPHINX model [14] on our dataset, enabling UniAff to predict object-centric 3D representations and infer affordances for both tools and articulated objects. To our best knowledge, UniAff is the first model offering a unified understanding of both categories, representing a significant advancement in object-centric robotic manipulation.

Extensive experiments in both simulation and real-world settings demonstrate UniAff's effectiveness. On the HANDAL dataset [10], UniAff outperformed LISA [15] by 11 . 5% and closely matched ManipVQA [9], with only a 2 . 2% IOU difference, showcasing strong adaptation to real-world tasks. For articulated object manipulation, UniAff improved the success rate by 7 . 07% on unseen instances and 9 . 60% on unseen categories compared to A3VLM [7]. These results demonstrate UniAff's ability to generalize and adapt across tasks. In summary, our contributions are:

- 1) Introducing UniAff, an MLLM that facilitates understanding of fine-grained physical properties, 3D motion constraints, and affordances for manipulation.
- 2) Developing a dataset with 1,500 objects across 19 categories of articulated objects and 12 categories of tools, labeled with part-level 6D poses, manipulation types and affordances.
- 3) Conducting comprehensive experiments that demonstrate UniAff's significant improvement in robotic manipulation generalization across articulated objects, and tools.

## II. RELATED WORK

Datasets for Object-centric Manipulation. Objectcentric manipulation focuses on understanding object poses and affordances for various tasks. By focusing on object representation, object-centric policies are more transferable across robots, but they also require high-quality, diverse data on object poses and affordances. For affordance data, the RGB-D Part Affordance dataset [11] provides partlevel labels for 105 tools, while PhysObjects [16] offers annotations of household objects, focusing on physical properties. In object pose estimation, Omni6DPose [17] provides extensive pose annotations for real and simulated images. HANDAL [10], an early attempt to combine object affordance and pose estimation, includes graspable regions but is limited to 210 objects and only grasp affordances. To address these limitations, UniAff introduces a large-scale synthetic dataset, utilizing 230 real-world scanned tools from the PACE [18] and OmniObject3D [19] datasets, articulated data from PartNet-Mobility dataset [20], along with 370 newly scanned tools. Additionally, we include 900 articulated object manipulations across 19 categories and 600 tool-use tasks across 12 categories.

arios.**

Instruction-based Robotic Manipulation. Instructionbased robotic manipulation connects abstract commands to low-level control. Early works like CLIPort [21] and 6D-CLIPort [22] utilize pre-trained text encoders, such as CLIP [23], for task-specific policies. Hiveformer [24] integrates language, observations, and action history, while Perceiver Actor [25] uses voxelized 3D data for efficient learning. However, pre-trained language encoders are limited to simple instructions and lack deeper reasoning for complex tasks. To address these limitations, LLMs have been applied to task planning [26-28], robot control code generation [29, 30] and VLA [31, 32]. Recent advances, such as RT-2-X [33], ManipLLM [8], and AIC-MLLM [34], incorporate multimodal LLMs (MLLMs) like LLaVA [35] and Sphinx [36], improving reasoning from both language instructions and visual input. However, these methods rely on modeling robot actions directly, requiring extensive realworld interaction data, which limits transferability across different robots. To overcome this, works like ManipVQA [9] and A3VLM [7] emphasize affordance reasoning and articulation awareness. Yet, ManipVQA focuses only on tool use, and A3VLM on articulated objects, restricting their realworld applications. To address this, we propose UniAff, a unified representation for both tools and articulated objects.

## III. METHOD

In this section, we introduce our method, which presents a structured 3D spatial formulation for object representation in robotic manipulation. Our approach unifies task formulation by incorporating object-centric 3D spatial motion constraints and their corresponding affordances (Section III-A). To further enhance spatial intelligence, we developed a synthetic dataset that applies this unified formulation to both tools and articulated objects (Section III-B). Finally, we finetune MLLMs to integrate object-centric formulations through Visual Question Answering (VQA) (Section III-C).

## A. Formulation of Structured Manipulation Task

We define the manipulation task formulation T as follows. An unknown object M consists of K movable parts, represented as M = { m i } K i =1 . We observe the object M through an image I and a depth map D . Furthermore, we define the object structure S for each part ψ i as S = { ψ i } K i =1 . The parameters for each part can be defined as ψ i = {A i , B i , G i , F i , J i , L i } , where A i ∈ R 4 × 3 represents the 6D pose of the part in the 3D space, B i ∈ R 4 × 2 represents the part's bounding box (BBOX), G i ∈ R 4 × 2 denotes the grasp affordance BBOX, and F i ∈ R 4 × 2 specifies the functional affordance BBOX, J i indicates the joint type, and L i describes the part's state or function. Notably, for tools that consist of only a single part, the entire object is treated as one part.

To accurately represent each part, we use a rotated BBOX defined by four key points to better fit its geometry and orientation. The B , G and F are defined by their four vertices ( x i , y i ) i =0 ,..., 3 , where each ( x i , y i ) represents the 2D coordinates of i -th vertex in the image.

Since most articulated objects consist of one-dimensional prismatic or revolute joints, or a combination of both [20], we categorize J into four distinct manipulation types based on the manipulation policy, as illustrated in Figure 4, including bottle caps, revolute parts, sliding lids, and prismatic parts, respectively. For tools with 6 DOF, we introduce a fifth manipulation type, termed as the 'freedom object'.

## B. Synthetic Data Generation

Based on the unified structure of part representation formulation ψ , which incorporates object-centric 3D spatial motion constraints and corresponding affordances as defined in the previous section, we generate synthetic data in the defined format ψ for tools and articulated objects. Acquiring real-world data is both costly and time-consuming, particularly due to the challenges in obtaining and annotating it. By leveraging near-realistic simulations to generate prescanned meshes or URDF models of objects, we efficiently create large-scale data across diverse scenes and object states. This synthetic data enables us to fine-tune VLM models while leveraging the capabilities of large models to achieve generalization from simulation to real-world tasks.

Fig. 2. The architecture of UniAff . The image features are first extracted using a Mixed Visual Encoder, such as DINOv2, CLIP, or Q-Former, followed by an MLP projector. Next, language instructions are used to extract features with the Llama Tokenizer. Finally, the output of the structured manipulation tasks, such as Part BBOX, Affordance, and Revolute Parts, is used to execute robotic instructions.

<!-- image -->

Fig. 3. Illustration of tools. The blue box indicates grasp affordance, the red box indicates functional affordance and the orientation axis illustrates the object's pose.

<!-- image -->

Since the formulation of B , L remains consistent, we present a unified representation here and will not reiterate it below.

- Part BBOX ( B ) : A 2D BBOX defined by four vertices ( x i , y i ) i =0 ,..., 3 is used to delineate the region of the part in the image.
- Descriptive Sentence ( L ) : An explicit description of the part's state or function is essential for VLMs to interpret it effectively.

1) Tools : Modeling a tool's 6D pose, along with its grasp and functional affordances, is essential for precise and effective manipulation, as the use of a tool is strongly correlated with its 6D pose in 3D space [5]. In our work, we utilized 230 tools from real-world scanned data sourced from the PACE [18] and OmniObject3D [19] datasets, supplemented by 370 additional tools that we scanned ourselves, covering 12 categories as shown in Figure 3: brush, razor, screwdriver, hair dryer, hammer, knife, spoon, spatula, power drill, flower shovel, fork, and ladle. To ensure consistency, all tools were aligned to a common axis. Using Blender, we segmented the textured meshes into grasp affordance parts and functional affordance parts , with each tool's segmentation process taking approximately 5 minutes .

Fig. 4. Illustration of manipulation types.(a) bottle cap, (b) revolute part, (c) sliding lid, (d) prismatic part. The yellow box represents the object part, the blue box indicates grasp affordance, the red arrow marks the joint parameter, and the green arrow illustrates the manipulation trajectory.

<!-- image -->

Building on these labels and leveraging near-realistic rendering technology in simulation [20], we rendered diverse cluttered scenes. For more details on data rendering, please refer to our website . Each scene was automatically labeled with key attributes defined in the formulation {A , B , G , F , J , L} . The manipulation type of the tools is identified as the 'freedom object'.

- 6D Pose ( A ) : The A is represented as a 4 × 3 matrix, where the first 1 × 3 row represents the tool's 3D spatial position, capturing the translational degrees, and the remaining 3 × 3 matrix captures its rotational degrees in 3D.
- Grasp Affordances ( G ) : A 2D BBOX defined by four vertices ( x i , y i ) i =0 ,..., 3 is used to delineate the grasp area of the tool.
- Functional Affordances ( F ) : A 2D BBOX defined by four vertices ( x i , y i ) i =0 ,..., 3 is used to represent the functional area of the tool.

By varying the scene configurations, tool orientations, and surrounding objects, we ensured that the dataset encompassed a wide range of scenarios, enabling the model to generalize effectively to real-world manipulation tasks.

- 2) Articulated Objects : To manipulate articulated objects effectively, it is crucial to develop a comprehensive understanding of each part's manipulation type ( J ), its

TABLE I

## OVERVIEW OF THE STRUCTURED MANIPULATION TASKS.

| Capabilities                              | Tasks                    | Examples of Task Templates                                                                                                                                                                           | Num.   |
|-------------------------------------------|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| Part Grounding.                           | 2D-Part-Detection        | User : Please detect all manipulable parts and provide their 2D rotated bounding boxes. UniAff : There are N manipulable object parts, each with a corresponding 2D bounding box: B 1 , B 2 , ..., B | 130K   |
| 6D Pose and Manipulation Type Understand. | 6D-Pose-Detection        | User : Please detect the manipulation type of the object and provide the 6D pose. UniAff : Manipulation Type J and its 6D Pose A .                                                                   | 130K   |
| Grasp-Affordance Understand.              | 2D-Grasp-Affordance      | User : Please detect the grasp region of part with BBox B and provide the 2D bounding box. UniAff : Here is the grasp affordance region BBox G .                                                     | 130K   |
| Functional-Affordance Understand.         | 2D-Functional-Affordance | User : Please detect the functional region of part with BBox B and provide the 2D bounding box. UniAff : Here is the functional affordance region BBox F .                                           | 10K    |

corresponding joint axis ( A ) in 3D space, and the grasp affordances ( G ) associated with various object states.

To establish a unified representation structure, we relabeled 900 objects across 19 categories from the PartNetMobility Dataset [20], including bottle, box, bucket, dispenser, door, folding chair, kitchen pot, laptop, microwave, refrigerator, safe, storage furniture, trash can, faucet, oven, table, toilet, kettle, and washing machine. Since the handles of some objects are not separate meshes or links, we modified the meshes or links of these objects to separate the handles into individual links. This modification allows for more accurate grasp affordance annotation, better aligning with our specific formulation ψ .

Based on the definition of ψ in our task formulation, we rendered objects in diverse states. For more details on data rendering, please refer to our website . We present the detailed auto-labeled dataset as follows:

- Manipulation Type ( J ) : Each object's movable parts are classified based on their specific 3D motion properties, with joint types divided into four manipulation types ( J ): bottle caps, sliding lids, revolute parts, and prismatic parts, as illustrated in Figure 4.
- Joint Axis ( A ) : For articulated parts with 4 degrees of freedom (DOF), the joint axis ( A ) is defined by two points ( x i , y i , z i ) i =0 , 1 , representing the joint's axis in 3D space, while the remaining DOF components are masked.
- Grasp Affordances ( G ) : An articulated object's affordances change based on its state. For example, a closed door requires the handle to open, while an open door can be manipulated by the handle or edge. To account for this, we use sampling weights that prioritize the door's edge based on the joint state. The sampling weight is calculated as:

<!-- formula-not-decoded -->

where J represents the current joint state and θ is a predefined threshold value.

3) VQA Design : In our task formulation, we define six key parameters to describe each part: 6D Pose ( A ), Bounding Box ( B ), Grasp Affordance Region ( G ), Functional Affordance Region ( F ), Manipulation Type ( J ), and a Sentence Describing the Part's State or Function ( L ).

To effectively address these six parameters, we utilize multiple Visual Question Answering (VQA) prompts to design four distinct tasks, as outlined in Table I. By guiding the model through a series of task-specific prompts, we develop a multi-task learning framework. By adopting this VQAbased approach, we integrate diverse manipulation-related prior knowledge into a unified model, enabling it to handle complex manipulation tasks across a wide range of object categories and scenarios. These multi-task MLLMs not only improve task efficiency but also enhance the model's capacity to generalize across diverse real-world robotic manipulation tasks.

- 2D-Part-Detection Task : This task improves the model's ability to identify object parts and their corresponding bounding boxes ( B ).
- 6D-Pose-Detection Task : This task enables the model to detect both the manipulation type ( J ) and the 6D pose ( A ) of each part. Accurately determining the joint axis of articulated parts or the 6D pose of tools is essential for guiding the robot's interactions with them, ensuring precise and efficient task execution.
- 2D-Grasp-Affordance Task : This task enhances the model's ability to recognize the grasp region of the specified part ( G ). Accurately identifying these regions is crucial for the robot to effectively grasp the target object and successfully complete the task.
- 2D-Functional-Affordance Task : This task equips the model with the ability to recognize the functional region of the specified part ( F ), enabling accurate tool usage to complete tasks effectively.

## C. MLLMs-based Manipulation and Model Fine-tuning

We propose a novel robot manipulation algorithm, termed UniAff, as illustrated in Figure 2. Our model architecture is built upon SPHINX [14] and utilizes LLaMA2 [37] as the language backbone, enabling robust multimodal interaction between visual and linguistic inputs. The design is optimized for fine-grained visual analysis, focusing on capturing detailed regional object features critical for manipulation tasks.

Using the 'Any Resolution' strategy from [14], input images with 448 × 448 resolution are divided into smaller sub-images to preserve fine-grained details. To ensure comprehensive global and local visual grounding, we integrate the visual encoder from CLIP [23] to extract local semantic features. Additionally, DINOv2 [38] is incorporated to further enhance the model's capacity for capturing detailed local semantics, while Q-Former [39] is employed to summarize global visual information. The local and global visual features are concatenated along the channel dimension to ensure thorough feature integration and improve visual understanding in manipulation tasks.

Our fine-tuning strategy encodes affordances and 3D physical information within natural language representations, aligning training samples with the VQA framework. Consequently, we adopt cross-entropy loss as our primary training objective. To preserve the model's broad visual reasoning capabilities, we incorporate general visual reasoning tasks alongside those specifically focused on predicting robotic affordances. The visual projection layers and the language model are jointly fine-tuned to ensure effective alignment between visual and linguistic modalities for affordance-based reasoning. Finally, the decoded manipulation information is combined with visual data using a De-Tokenizer, enabling the completion of the specified task.

## IV. EXPERIMENTS

In this section, we conduct comprehensive experiments in both simulation and real-world settings. We compare the performance of UniAff with several baseline models to address the following questions: 1) Can UniAff effectively ground the grasp and functional affordances of tools? 2) Can UniAff simultaneously construct 3D spatial motion constraints along with their corresponding affordances? 3) Can UniAff effectively generalize from perceptual priors to real-world applications?

## A. Experimental Settings

Model Setting. We fine-tuned the vision-language model based on the SPHINX model [14], utilizing eight NVIDIA A100 GPUs, each with 80 GB of memory. The fine-tuning process was completed over three epochs. To maintain the quality of the pre-trained features, the visual encoders remained frozen throughout the fine-tuning phase. Training was conducted with a batch size of 4, and the learning rate was set to 2 × 10 -5 .

Dataset Setting. We selected 9 categories from brush to power drill, creating a training set of 450 tools, including 70 unseen instances. An additional 3 unseen categories include flower shovel, fork, and ladle, totaling 80 tools. We generated 10,000 diverse scenes using the training tools and 3,000 images for unseen instances. We selected 13 categories for articulated objects from bottle to trash can, resulting in a training set of 502 objects and 160 unseen instances. Unseen categories are 6, from faucet to washing machine, totaling 238 objects. Each training object was rendered in 20 states from 5 perspectives, producing a total of 50,200 images. The rendered data were translated according to the structured task formulation ψ to train UniAff, as shown in Table I.

## B. Robotic Affordance Detection Result

Baselines and Metrics. Our robotic affordance evaluation leverages the HANDAL dataset [10], utilizing pixel-wise segmentation AP. While the baseline model in HANDAL detects only whole objects, our UniAff model identifies both complete objects and manipulable affordance regions effectively. However, because our model generates rotated bounding boxes, a direct comparison of bounding box AP with the ground truth data, which uses standard bounding boxes, is not feasible. Instead, we provide a qualitative comparison with visual examples of our model's performance on

TABLE II ROBOTIC AFFORDANCE EVALUATION RESULTS ON HANDAL DATASET

|              |    Ha |    Pd |    Sd |    La |   Pan |    Sp |    St |    Ut |    Wh |   AVG |
|--------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| LISA [15]    | 0.671 | 0.426 | 0.624 | 0.407 | 0.453 | 0.578 | 0.397 | 0.494 | 0.494 | 0.505 |
| ManipVQA [9] | 0.745 | 0.439 | 0.799 | 0.620 | 0.622 | 0.646 | 0.638 | 0.584 | 0.683 | 0.642 |
| Ours         | 0.742 | 0.394 | 0.747 | 0.650 | 0.680 | 0.621 | 0.597 | 0.435 | 0.712 | 0.620 |

Object abbreviations are listed in sequence: Hammer, Power Drill, Screwdriver, Ladle, Pan, Spatula, Strainer, Utensil, and Whisk. Results are reported as IOU ( ↑ ).

our website. To facilitate comparison, we convert the rotated bounding boxes into standard ones and employ SAM [40] to generate segmentation masks. This enables us to compare our approach with LISA [15], which integrates LLM and a SAM decoder, as well as ManipVQA-SAM [9], where SAM [40] converts the bounding box into segmentation masks.

Results. As shown in Table II, UniAff demonstrates competitive performance even in a zero-shot setting associated with SAM [40]. Trained solely on the simulation dataset, UniAff outperformed LISA by a significant margin of 11 . 5% and achieved performance comparable to ManipVQA, with only a 2 . 2% difference in IOU. Notably, UniAff does not have access to the HANDAL dataset, unlike ManipVQA.

## C. Tool Usage Understanding Evaluation

Baselines and Metrics. We evaluate UniAff's tool usage understanding on the test splits of our dataset. Unlike the general affordance evaluation on HANDAL, this evaluation requires grounding both grasp affordance (e.g., grasp area) and functional affordance. To achieve a more compact representation, we use rotated bounding boxes as ground truth. We compare our method against ManipVQA.

Results. As shown in Table IV, UniAff excels at detecting both grasp affordances, with a 32 . 5% improvement, and functional affordances of tools, with a 56 . 9% improvement in IOU, compared to ManiVQA, which struggles with functional affordance reasoning. Along with the results in Table II, UniAff demonstrates superior generalization, likely due to its larger and more diverse dataset.

## D. Articulation Manipulation Evaluation

Task and Metrics. We evaluated articulation tasks, including opening bottle caps, sliding lids, and both revolute and prismatic parts on seen categories. For unseen categories, tasks involved manipulating sliding lids and revolute/prismatic parts. The evaluation covered 160 unseen instances from trained categories and 238 objects from new categories. Success was defined as a binary measure, with a joint state change exceeding the threshold δ = 0 . 1 : success = 1( δ change ≥ δ ).

Baselines. We compare our proposed method with three different baselines under the identical setting: (1)Where2Act [41] identifies high-actionability manipulation points and generates short-term actions (e.g., pushing, pulling) at each point for interacting with articulated objects. (2) UMPNet [43] infers action sequences for manipulating objects through self-guided exploration and an Arrow-ofTime attribute. We adapted it by replacing suction with a gripper. (3)A3VLM[7] translates object representations into robot actions utilizing simple primitives. We modified it by Unseen instances consist of data from four object categories utilized in model training. Unseen categories encompass data from completely new categories. Results are reported as success rates ( ↑ ).

TABLE III ARTICULATED OBJECT MANIPULATION RESULTS

|                      | Unseen Instances   | Unseen Instances   | Unseen Instances   | Unseen Instances   | Unseen Categories   | Unseen Categories   | Unseen Categories   |
|----------------------|--------------------|--------------------|--------------------|--------------------|---------------------|---------------------|---------------------|
|                      | Bottle Cap         | Sliding Lid        | Revolute Part      | Prismatic Part     | Sliding Lid         | Revolute Part       | Prismatic Part      |
| Where2Act [41]       | 0.2034             | 0.1921             | 0.0897             | 0.1093             | 0.1535              | 0.0822              | 0.0861              |
| UMPNet [42]          | 0.2787             | 0.2535             | 0.3521             | 0.3000             | 0.2789              | 0.3201              | 0.3158              |
| A3VLM [7]            | 0.4895             | 0.5969             | 0.4397             | 0.5231             | 0.4535              | 0.4906              | 0.4387              |
| UniAff (w/o afford.) | 0.5166             | 0.6321             | 0.4506             | 0.5360             | 0.4820              | 0.5341              | 0.4519              |
| UniAff (ours)        | 0.5259             | 0.6642             | 0.5418             | 0.6001             | 0.5730              | 0.5913              | 0.5064              |

## TABLE IV

TOOL USAGE UNDERSTANDING EVALUATION RESULTS

|          |                |       |       |       |       |       |       |       |       |       | Unseen Categories   | Unseen Categories   | Unseen Categories   |       |
|----------|----------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------------------|---------------------|---------------------|-------|
| Task     | Model          | Sl    | Sp    | Kf    | Ra    | Pd    | Hd    | Ha    | Br    | Sd    | Fs                  | Ld                  | Fr                  | AVG   |
| Grasp    | ManipVQA [9]   | 0.329 | 0.487 | 0.449 | 0.343 | 0.461 | 0.455 | 0.309 | 0.360 | 0.534 | 0.420               | 0.304               | 0.319               | 0.398 |
| Grasp    | 2 Points BBox  | 0.346 | 0.511 | 0.463 | 0.375 | 0.596 | 0.487 | 0.394 | 0.445 | 0.556 | 0.420               | 0.290               | 0.330               | 0.434 |
| Grasp    | 224 Resolution | 0.263 | 0.446 | 0.377 | 0.333 | 0.478 | 0.406 | 0.357 | 0.385 | 0.390 | 0.319               | 0.236               | 0.298               | 0.357 |
| Grasp    | UniAff (ours)  | 0.751 | 0.692 | 0.768 | 0.717 | 0.758 | 0.768 | 0.684 | 0.740 | 0.834 | 0.713               | 0.606               | 0.641               | 0.723 |
| Function | ManipVQA [9]   | 0.203 | 0.271 | 0.234 | 0.068 | 0.043 | 0.303 | 0.126 | 0.106 | 0.058 | 0.298               | 0.169               | 0.140               | 0.168 |
| Function | 2 Points BBox  | 0.610 | 0.690 | 0.484 | 0.476 | 0.624 | 0.657 | 0.502 | 0.544 | 0.234 | 0.594               | 0.686               | 0.551               | 0.554 |
| Function | 224 Resolution | 0.267 | 0.527 | 0.413 | 0.233 | 0.254 | 0.491 | 0.201 | 0.413 | 0.219 | 0.482               | 0.349               | 0.333               | 0.349 |
| Function | UniAff (ours)  | 0.756 | 0.735 | 0.779 | 0.652 | 0.749 | 0.767 | 0.732 | 0.755 | 0.689 | 0.780               | 0.745               | 0.703               | 0.737 |

Object abbreviations are listed in sequence: Spatula, Spoon, Knife, Razor, Power Drill, Hair Dryer, Hammer, Brush, Screwdriver, Flower Shovel, Ladle, and Fork. Results are reported as IOU ( ↑ ).

using a gripper and GraspNet[44] to generate and select the highest-scoring grasp pose.

Results. We present the success rates in Table III. By explicitly modeling the articulation structure, both A3VLM [7] and UniAff demonstrate superior performance. UniAff achieved a 7 . 07% improvement in success rates for unseen instances and a 9 . 60% improvement for unseen categories compared to A3VLM.

Open Drawer Open Refrigerator Open Microwave Open Pot Lift Bucket Handle Strike target

<!-- image -->

Fig. 5. Implementation of UniAff in real-world experiments progressed from tool manipulation to articulated object interaction, encompassing tasks such as striking a designated target with a hammer, opening a drawer, refrigerator, microwave, pot, and lifting a bucket handle.

## E. Ablation Studies

To assess UniAff's design contributions, we conducted ablation studies. Results in Table IV indicate that omitting the 'Any Resolution' method (e.g., using a 224 × 224 resolution) significantly degrades performance, highlighting the necessity of high-resolution input for partial object understanding. Additionally, the use of a 2-point bounding box also reduces performance, as rotated bounding boxes provide a more compact representation for affordance understanding.

## F. Real-World Experiments

We conducted real-world experiments using a 7-DoF Flexiv robot. First, we mounted an RGB-D RealSense D435 camera on the robot's wrist to capture RGB-D images of various objects. We then applied UniAff to predict objectcentric 3D motion constraints and corresponding affordances. The experiments involved six tasks: striking a designated target with a hammer, opening a drawer, opening a refrigerator, opening a microwave, opening a pot, and lifting a bucket handle. These tasks demonstrated UniAff's ability to generalize effectively to real-world manipulation tasks, as illustrated in Figure 5. Due to space constraints, detailed information is provided in the website .

## V. CONCLUSION

In this paper, we propose UniAff, a novel approach integrating object-centric 3D motion constraints and affordances for manipulation tasks. By leveraging multimodal large language models (MLLMs), UniAff improves manipulation knowledge and generates precise 3D motion constraints and affordances for diverse objects. Our extensive dataset labeled with manipulation-related key attributes, comprising 900 articulated objects and 600 tools, serves as a foundation for training UniAff to generalize across a wide range of tasks.

Experiments show UniAff significantly outperforms existing methods in both simulation and real-world environments, demonstrating superior generalizability in manipulation tasks involving tools and articulated objects.

## REFERENCES

- [1] L. P. Kaelbling, 'The foundation of efficient robot learning,' Science , vol. 369, no. 6506, pp. 915-916, 2020.
- [2] H. Geng, S. Wei, C. Deng, B. Shen, H. Wang, and L. Guibas, 'Sage: Bridging semantic and actionable parts for generalizable articulated-object manipulation under language instructions,' arXiv preprint arXiv:2312.01307 , 2023.
- [3] H. Geng, H. Xu, C. Zhao, C. Xu, L. Yi, S. Huang, and H. Wang, 'Gapartnet: Cross-category domain-generalizable object perception and manipulation via generalizable and actionable parts,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2023, pp. 70817091.
- [4] Q. Yu, J. Wang, W. Liu, C. Hao, L. Liu, L. Shao, W. Wang, and C. Lu, 'Gamma: Generalizable articulation modeling and manipulation for articulated objects,' in 2024 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2024, pp. 5419-5426.
- [5] G. Li, N. Tsagkas, J. Song, R. Mon-Williams, S. Vijayakumar, K. Shao, and L. Sevilla-Lara, 'Learning precise affordances from egocentric videos for robotic manipulation,' arXiv preprint arXiv:2408.10123 , 2024.
- [6] H. Huang, F. Lin, Y. Hu, S. Wang, and Y. Gao, 'Copa: General robotic manipulation through spatial constraints of parts with foundation models,' arXiv preprint arXiv:2403.08248 , 2024.
- [7] S. Huang, H. Chang, Y. Liu, Y. Zhu, H. Dong, P. Gao, A. Boularias, and H. Li, 'A3vlm: Actionable articulation-aware vision language model,' arXiv preprint arXiv:2406.07549 , 2024.
- [8] X. Li, M. Zhang, Y. Geng, H. Geng, Y. Long, Y. Shen, R. Zhang, J. Liu, and H. Dong, 'Manipllm: Embodied multimodal large language model for object-centric robotic manipulation,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2024, pp. 18 06118 070.
- [9] S. Huang, I. Ponomarenko, Z. Jiang, X. Li, X. Hu, P. Gao, H. Li, and H. Dong, 'Manipvqa: Injecting robotic affordance and physically grounded information into multi-modal large language models,' arXiv preprint arXiv:2403.11289 , 2024.
- [10] A. Guo, B. Wen, J. Yuan, J. Tremblay, S. Tyree, J. Smith, and S. Birchfield, 'Handal: A dataset of real-world manipulable object categories with pose annotations, affordances, and reconstructions,' in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) . IEEE, 2023, pp. 11 428-11 435.
- [11] A. Myers, C. L. Teo, C. Ferm¨ uller, and Y. Aloimonos, 'Affordance detection of tool parts from geometric features,' in 2015 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2015, pp. 1374-1381.
- [12] W. Xia, D. Wang, X. Pang, Z. Wang, B. Zhao, D. Hu, and X. Li, 'Kinematic-aware prompting for generalizable articulated object manipulation with llms,' in 2024 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2024, pp. 2073-2080.
- [13] D. Tola and P. Corke, 'Understanding urdf: A dataset and analysis,' IEEE Robotics and Automation Letters , 2024.
- [14] Z. Lin, C. Liu, R. Zhang, P. Gao, L. Qiu, H. Xiao, H. Qiu, C. Lin, W. Shao, K. Chen et al. , 'Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models,' arXiv preprint arXiv:2311.07575 , 2023.
- [15] X. Lai, Z. Tian, Y. Chen, Y. Li, Y. Yuan, S. Liu, and J. Jia, 'Lisa: Reasoning segmentation via large language model,' in
16. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2024, pp. 9579-9589.
- [16] J. Gao, B. Sarkar, F. Xia, T. Xiao, J. Wu, B. Ichter, A. Majumdar, and D. Sadigh, 'Physically grounded vision-language models for robotic manipulation,' in 2024 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2024, pp. 12 462-12 469.
- [17] J. Zhang, W. Huang, B. Peng, M. Wu, F. Hu, Z. Chen, B. Zhao, and H. Dong, 'Omni6dpose: A benchmark and model for universal 6d object pose estimation and tracking,' arXiv preprint arXiv:2406.04316 , 2024.
- [18] Y. You, K. Xiong, Z. Yang, Z. Huang, J. Zhou, R. Shi, Z. Fang, A. W. Harley, L. Guibas, and C. Lu, 'Pace: Pose annotations in cluttered environments,' Springer, 2024.
- [19] T. Wu, J. Zhang, X. Fu, Y. Wang, J. Ren, L. Pan, W. Wu, L. Yang, J. Wang, C. Qian et al. , 'Omniobject3d: Largevocabulary 3d object dataset for realistic perception, reconstruction and generation,' in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , 2023, pp. 803-814.
- [20] F. Xiang, Y. Qin, K. Mo, Y. Xia, H. Zhu, F. Liu, M. Liu, H. Jiang, Y. Yuan, H. Wang, L. Yi, A. X. Chang, L. J. Guibas, and H. Su, 'SAPIEN: A simulated part-based interactive environment,' in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , June 2020.
- [21] M. Shridhar, L. Manuelli, and D. Fox, 'Cliport: What and where pathways for robotic manipulation,' in Conference on robot learning . PMLR, 2022, pp. 894-906.
- [22] K. Zheng, X. Chen, O. C. Jenkins, and X. Wang, 'Vlmbench: A compositional benchmark for vision-and-language manipulation,' Advances in Neural Information Processing Systems , vol. 35, pp. 665-678, 2022.
- [23] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al. , 'Learning transferable visual models from natural language supervision,' in International conference on machine learning . PMLR, 2021, pp. 8748-8763.
- [24] P.-L. Guhur, S. Chen, R. G. Pinel, M. Tapaswi, I. Laptev, and C. Schmid, 'Instruction-driven history-aware policies for robotic manipulations,' in Conference on Robot Learning . PMLR, 2023, pp. 175-187.
- [25] M. Shridhar, L. Manuelli, and D. Fox, 'Perceiver-actor: A multi-task transformer for robotic manipulation,' in Conference on Robot Learning . PMLR, 2023, pp. 785-799.
- [26] J. Wu, R. Antonova, A. Kan, M. Lepert, A. Zeng, S. Song, J. Bohg, S. Rusinkiewicz, and T. Funkhouser, 'Tidybot: Personalized robot assistance with large language models,' Autonomous Robots , vol. 47, no. 8, pp. 1087-1102, 2023.
- [27] W. Cai, S. Huang, G. Cheng, Y. Long, P. Gao, C. Sun, and H. Dong, 'Bridging zero-shot object navigation and foundation models through pixel-guided navigation skill,' arXiv preprint arXiv:2309.10309 , 2023.
- [28] H. Chang, K. Gao, K. Boyalakuntla, A. Lee, B. Huang, H. U. Kumar, J. Yu, and A. Boularias, 'Lgmcts: Languageguided monte-carlo tree search for executable semantic object rearrangement,' arXiv preprint arXiv:2309.15821 , 2023.
- [29] J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, and A. Zeng, 'Code as policies: Language model programs for embodied control,' in 2023 IEEE International Conference on Robotics and Automation (ICRA) . IEEE, 2023, pp. 9493-9500.
- [30] W. Huang, C. Wang, R. Zhang, Y. Li, J. Wu, and L. Fei-Fei, 'Voxposer: Composable 3d value maps for robotic manipulation with language models,' arXiv preprint arXiv:2307.05973 , 2023.
- [31] W. Song, H. Zhao, P. Ding, C. Cui, S. Lyu, Y. Fan, and D. Wang, 'Germ: A generalist robotic model with mixture-of-experts for quadruped robot,' arXiv preprint

arXiv:2403.13358 , 2024.

- [32] S. Li, J. Wang, R. Dai, W. Ma, W. Y. Ng, Y. Hu, and Z. Li, 'Robonurse-vla: Robotic scrub nurse system based on visionlanguage-action model,' arXiv preprint arXiv:2409.19590 , 2024.
- [33] A. Padalkar, A. Pooley, A. Jain, A. Bewley, A. Herzog, A. Irpan, A. Khazatsky, A. Rai, A. Singh, A. Brohan et al. , 'Open x-embodiment: Robotic learning datasets and rt-x models,' arXiv preprint arXiv:2310.08864 , 2023.
- [34] C. Xiong, C. Shen, X. Li, K. Zhou, J. Liu, R. Wang, and H. Dong, 'Autonomous interactive correction mllm for robust robotic manipulation,' in 8th Annual Conference on Robot Learning , 2024.
- [35] H. Liu, C. Li, Q. Wu, and Y. J. Lee, 'Visual instruction tuning,' Advances in neural information processing systems , vol. 36, 2024.
- [36] P. Gao, R. Zhang, C. Liu, L. Qiu, S. Huang, W. Lin, S. Zhao, S. Geng, Z. Lin, P. Jin et al. , 'Sphinx-x: Scaling data and parameters for a family of multi-modal large language models,' arXiv preprint arXiv:2402.05935 , 2024.
- [37] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale et al. , 'Llama 2: Open foundation and fine-tuned chat models,' arXiv preprint arXiv:2307.09288 , 2023.
- [38] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. ElNouby et al. , 'Dinov2: Learning robust visual features without supervision,' arXiv preprint arXiv:2304.07193 , 2023.
- [39] J. Li, D. Li, S. Savarese, and S. Hoi, 'Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,' in International conference on machine learning . PMLR, 2023, pp. 19 730-19 742.
- [40] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo et al. , 'Segment anything,' in Proceedings of the IEEE/CVF International Conference on Computer Vision , 2023, pp. 4015-4026.
- [41] K. Mo, L. J. Guibas, M. Mukadam, A. Gupta, and S. Tulsiani, 'Where2act: From pixels to actions for articulated 3d objects,' in Proceedings of the IEEE/CVF International Conference on Computer Vision , 2021, pp. 6813-6823.
- [42] R. Wu, Y. Zhao, K. Mo, Z. Guo, Y. Wang, T. Wu, Q. Fan, X. Chen, L. Guibas, and H. Dong, 'Vat-mart: Learning visual action trajectory proposals for manipulating 3d articulated objects,' arXiv preprint arXiv:2106.14440 , 2021.
- [43] Z. Xu, Z. He, and S. Song, 'Universal manipulation policy network for articulated objects,' IEEE robotics and automation letters , vol. 7, no. 2, pp. 2447-2454, 2022.
- [44] H.-S. Fang, C. Wang, M. Gou, and C. Lu, 'Graspnet-1billion: A large-scale benchmark for general object grasping,' in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , 2020, pp. 11 444-11 453.