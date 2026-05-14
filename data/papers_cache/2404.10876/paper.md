## Course Recommender Systems Need to Consider the Job Market

Jibril Frej jibril.frej@epfl.ch ML4ED Lab, IC, EPFL Switzerland Anna Dai anna.dai@epfl.ch NLP Lab, IC, EPFL Switzerland

Antoine Bosselut antoine.bosselut@epfl.ch NLP Lab, IC, EPFL Switzerland

## ABSTRACT

Current course recommender systems primarily leverage learnercourse interactions, course content, learner preferences, and supplementary course details like instructor, institution, ratings, and reviews, to make their recommendation. However, these systems often overlook a critical aspect: the evolving skill demand of the job market. This paper focuses on the perspective of academic researchers, working in collaboration with the industry, aiming to develop a course recommender system that incorporates job market skill demands. In light of the job market's rapid changes and the current state of research in course recommender systems, we outline essential properties for course recommender systems to address these demands effectively, including explainable, sequential, unsupervised, and aligned with the job market and user's goals. Our discussion extends to the challenges and research questions this objective entails, including unsupervised skill extraction from job listings, course descriptions, and resumes, as well as predicting recommendations that align with learner objectives and the job market and designing metrics to evaluate this alignment. Furthermore, we introduce an initial system that addresses some existing limitations of course recommender systems using large Language Models (LLMs) for skill extraction and Reinforcement Learning (RL) for alignment with the job market. We provide empirical results using open-source data to demonstrate its effectiveness.

## CCS CONCEPTS

· Computing methodologies → Information extraction ; · Information systems → Recommender systems ; · Applied computing → Learning management systems .

## KEYWORDS

Recommender System, Course Recommendation, Entity linking Syrielle Montariol syrielle.montariol@epfl.ch NLP Lab, IC, EPFL Switzerland Tanja Käser Tanja.kaser@epfl.ch ML4ED Lab, IC, EPFL Switzerland

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

SIGIR '24, July 14-18, 2024, Washington, DC, USA

©2024 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-0431-4/24/07 https://doi.org/10.1145/3626772.3657847

## ACMReference Format:

Jibril Frej, Anna Dai, Syrielle Montariol, Antoine Bosselut, and Tanja Käser. 2024. Course Recommender Systems Need to Consider the Job Market. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24), July 14-18, 2024, Washington, DC, USA. ACM, New York, NY, USA, 11 pages. https://doi.org/ 10.1145/3626772.3657847

## 1 INTRODUCTION

The contemporary job market is dynamic and rapidly evolving [8], necessitating continuous adaptation of individual skill sets to maintain relevance and competitiveness. This evolution introduces a unique challenge: guiding learners in selecting educational courses that enhance their expertise and align with their career objectives and with job market demands. However, a notable mismatch exists between the skills learners possess, the skills taught, and those in demand in the job market [24]. This mismatch can be explained by various factors such as the lag between demand in the market and the adaptation from course providers, unequal access to training, or the lack of information from training providers on the type of skills needed in the job market [24]. This issue significantly limits the employability and career progression of individuals, impacting both their personal development and, at a more systemic level, economies dependent on a skilled workforce.

However, existing course recommender systems often focus solely on learner-course dynamics [29, 30], neglecting the crucial aspect of aligning course recommendations with real-time job market trends. This limitation leads to a mismatch between the skills acquired through recommended courses and those in demand in the job market. Moreover, while some approaches to career path recommendation or skill recommendation consider the job market, users' skills, and goals for their recommendations [13, 34], they do not recommend specific courses to help users achieve their goals. Finally, to our knowledge, a single study proposes to use job postings and resumes in a course recommender system [4]. However, this approach does not directly consider market trends or users' goals, making this domain largely unexplored.

In this paper, we present the perspective of academic researchers working in collaboration with industry practitioners aiming to develop and deploy job-market-oriented course recommender systems. We argue that rethinking course recommender systems to consider the job market has the potential for significant economic and societal impact. We outline the properties these systems must have: ( P1 )

aligned with the latest job market trends to prioritize courses teaching high-demand skills; ( P2 ) unsupervised to avoid the resourceintensive process of collecting and annotating up-to-date data; ( P3 ) sequential to recommend a sequence of courses where each course builds upon the knowledge acquired in the preceding ones; ( P4 ) aligned with users' goals such as attaining a specific role or increase their marketability; ( P5 ) explainable to ensure user trust and engagement. We also highlight research directions and areas for future research to address the challenges in this field: ( RD1 ) Addressing the scarcity of course recommendation datasets by creating or providing datasets to the community for training and evaluation; ( RD2 ) Designing evaluation metrics to take into account alignment with the job market when evaluating the recommendations. ( RD3 ) Estimating user's goal progress based on their profile and the job market to tailor the recommendations to their needs; ( RD4 ) Developing Skill-based explainable models and visualization techniques; ( RD5 ) Developing Unsupervised Skill Matching models to estimate up-to-date skill demand on the job market; ( RD6 ) Unsupervised Taxonomy Construction to include new emerging skills without human labeling.

In this work, we also develop a skill extraction and matching ( SEM ) method to identify skills and proficiency levels from learners' resumes, course content, and job descriptions. We also develop an unsupervised, sequential skill-based Job-Market-Oriented Course Recommender system ( JCRec ) that uses the skills extracted by SEM to determine a candidate's course options and to estimate the number of job opportunities available to them. JCRec then uses Reinforcement Learning (RL) to recommend a sequence of courses that maximizes the number of job opportunities available to the user. Our system meets all of the properties we previously outlined and lays out further steps for the research directions we've mentioned.

The key contributions of our paper include 1 :

- Identification of the desirable properties a job-market-oriented course recommender system should have.
- Identification of the challenges that developing such systems will pose along with research direction for the community to address these challenges.
- Afew-shot skill extraction method to find skills from resumes, job postings, and course descriptions.
- Aformulation of job-market-oriented course recommendation as a Markov Decision Process
- A first job-market-oriented course recommender system.

## 2 PERSPECTIVE

This paper presents the perspective of academic researchers working in collaboration with industry practitioners aiming to develop and deploy course recommender systems for a multilingual user base. This work is motivated by the fact that academic work about course recommender systems seldom considers the job market reality (see section 3). This disconnection is reflected in the features used to perform the recommendation, but also the lack of consideration of the objective behind the users taking courses. In this work, following extensive collaboration between academic researchers and industries from the up-skilling and continuing education domain, we propose methods to devise recommender systems that align with users' preexisting skills, career objectives, and the current demands of the job market. Thus, we focus on the challenging and realistic situation of the career development of users, where users have experience and seek to obtain new skills, often to find a new position. With this work, we aim to motivate the importance of this problem, propose research directions for the Information Retrieval and Recommendation communities, and provide a prototype of a job-market-oriented course recommender system.

[1 Our code is available at https://github.com/Jibril-Frej/JCRec](https://github.com/Jibril-Frej/JCRec)

## 3 COURSE RECOMMENDER SYSTEMS

Course recommender systems have been extensively studied, focusing on various aspects such as learning activities recommendation through open learner models (OLMs) [1], recommendation incorporating users' skills [26, 31], peer learner recommendation [27] or target course-oriented recommendation [15]. In the recent domain of course recommender systems using neural networks (NN), multiple research directions have been pursued. These include optimizing the accuracy of recommendation [30, 42], ensuring fairness [16, 20], and improving explainability [12, 33, 36]. The majority of these approaches use a combination of learner-course interactions, course content, learner preferences, and additional course information such as teachers, schools, course ratings, or comments, usually in the form of Knowledge Graphs (KG). While we acknowledge the significance of incorporating learner and course data, we consider that an effective course recommender system must incorporate the job market's current demands, and avoid recommending courses that teach skills lacking demand on the job market.

To our knowledge, Skill scanner [4] is the only work that uses resumes, course descriptions, and job descriptions for skill-based course recommendations. This system's pipeline involves extracting, vectorizing (using word embeggins techniques such as Word2vec [21] and Glove [25]), clustering, and matching skill sets. Skill scanner is used to compare courses, learners, and job postings. These comparisons serve not only for skill-based course recommendations but also to inform job seekers and educational institutions about the market relevance of specific skills, enabling them to adapt accordingly. However since their approach relies solely on encoding skill sets in a common representation space, it does not directly consider the job market trends, skill demand distribution, or the user's goal. Hence, Skill scanner is a first step in the direction of job market-guided course recommendations but most of the work remains to be done.

## 4 RETHINKING COURSE RECOMMENDER SYSTEMS TO CONSIDER THE JOB MARKET

In this section, we first list the properties that job-market-oriented course recommender systems should have and we propose several Research Directions to address the issues we identified to develop such systems. We voluntarily omitted some properties that all recommender systems should satisfy as they are not specific to our case, such as personalized and real-time recommendations.

## 4.1 Properties

P1: Job Market Alignment. The recommender system must consider the skill demand in the job market when making recommendations. On the job market, skills differ by their popularity. Learning a new skill, depending on whether that skill is in high or low demand from employers, or if it is rare or frequent among other job applicants, will have a very different impact on the learner's marketability. Thus, when for example comparing similar courses, the recommender system should give preference to the one teaching the skill with greater market demand. This ensures that learners are equipped with the most relevant and sought-after skills to increase their chances of finding a position.

P2: Minimal Supervision. The recommender system should rely on a limited amount of labeled data because it needs to accommodate the rapid evolution of the job market with new skills appearing regularly. However, existing recommenders usually rely on supervised models that would have to be updated regularly. To avoid the high cost of labeling data manually, most of the components should be based on unsupervised learning techniques. For example, scraping job postings and extracting skills from course descriptions using unsupervised models will allow the system to adapt to market trends without the extensive costs of manual data labeling.

P3: Sequential Recommendations. The system must recommend sequences of courses rather than standalone courses. Indeed, individual goals often require a progression through multiple subjects. Note that the order in which these courses are taken often matters. For instance, a front-end developer aspiring to become an LLM engineer might need courses in Python, machine learning, Natural Language Processing (NLP), and LLMs, with each course building upon the knowledge acquired in the preceding ones.

P4: User's Goal Alignment. The system must align with the objectives of its users, whether that involves attaining a specific role, learning skills to increase overall profile attractiveness, or specializing in a field. Recommendations should thus be aligned with these goals, ensuring that two users with identical profiles but different objectives receive different course suggestions.

P5: Explainable. The recommender system must provide explainable recommendations. Given the time and resources needed for a user to enroll and finish a course, it is crucial for the system to transparently explain its recommendations - especially sequential recommendations. Explainability is not only essential for building trust but also for ensuring that users feel confident in their decision to invest resources in a recommended course. For example, explaining that a course is suggested because it teaches a skill with current high demand in the job market can significantly enhance user confidence in the system's guidance.

## 4.2 Research Directions and Challenges

RD1: Course Recommendation Datasets Beyond consideration of the job market, a significant challenge affecting research in course recommendation is the scarcity of publicly accessible, large-scale datasets for this task. Presently, there are only two of such datasets: Xuetang [39] and COCO [9]. Xuetang contains courses from the Massive Open Online Courses (MOOCs) platform XuetangX 2 , primarily in Chinese. COCO contains courses from the MOOCs platform Udemy 3 available in 35 languages. This sparsity can be attributed to universities' reluctance to share student enrollment data due to privacy and ethical concerns and the desire of online course providers to preserve competitive advantages. In response to these challenges, we urge academics and industry professionals to anonymize and share subsets of their enrollment data with the research community. An alternative method we wish to highlight is the generation of synthetic datasets using generative models. While this method has been proposed in the context of job postings for skill extraction using LLMs [5, 7, 19], its application to creating course recommendation datasets remains unexplored. This presents an innovative research direction with the potential to significantly impact the field. Potential directions could involve generating coherent career paths of individuals using the in-context ability of LLMs, then inferring skills required to switch from one position to the next one using databases of skills associated with job roles. 4 Moreover in this work, instead of course enrollment data, we use only course descriptions along with job postings and resumes. While job postings are regularly explored in the literature [19, 40, 41], few public datasets exist. User profiles, in the form of resumes, are documents that contain personal information and are usually not public; such databases are even harder to come by than job postings, especially at scale. Overall, these data sources are seldom found in languages other than English. We show that we can use such data to build job-market-guided recommender systems, even though it doesn't allow for direct evaluation of users' course enrollment.

[2 https://next.xuetangx.com/](https://next.xuetangx.com/)

[3 https://www.udemy.com/](https://www.udemy.com/)

RD2: Evaluation Evaluating the effectiveness of a job-marketoriented course recommendation poses significant challenges. Most recommender systems estimate the relevance of a course based on user profiles using ranking metrics (MRR, nDGG, Hit) to compare recommendations to the actual courses taken using a user-course interaction matrix. Other metrics are used to measure Novelty, Fairness, Diversity, and Coverage of the recommendations [38]. However, these approaches are not enough for job-market-oriented course recommendations as they do not take into account market demand and the actual impact on a user's career trajectory. An ideal evaluation framework would also ascertain whether the recommended course enabled the user to meet their career goals, such as securing a desired job or enhancing their marketability. Because this type of information is rarely directly available, estimating the impact of following a course on a user's profile and achieving their objective presents a considerable challenge. For these reasons, we believe that designing an evaluation methodology for job-based course recommender systems is both a challenging and impactful research problem. In this work, we propose to evaluate the recommendations by estimating the number of jobs the users can apply to upon completing the recommended courses. Although our approach presents a first step for this research direction, user studies are necessary to evaluate if the system provides meaningful recommendations for users. Another potential research direction involves addressing the following problem: how can historical user enrollment data be leveraged to assess the effectiveness of a recommender system in a dynamic job market where past user choices may become less relevant? For instance, consider a scenario where a user previously enrolled in a PHP course, leading to employment in web development. Recommending this same course in the present day may not hold the same relevance, given shifts in the job market. Using this interaction for evaluation might bias the system toward past trends but ignoring all past interactions is not a satisfying solution since course enrollments include valuable information on user preferences that provides insight into specific flaws of evaluated recommender systems. One possible direction would be to ignore the skills of the courses - which might teach outdated skills- and focus on the user enrollment data, evaluating only the coherence and robustness of recommender systems between users.

4 A potential data source for this in the IT domain is https://www.berufe-der-ict.ch/ berufe-der-ict/.

RD3: Estimating User's Goal Progress On top of incorporating the dynamics of the job market in the recommendation, we highlight the importance of taking into account the objectives of learners. We propose an initial strategy involving two steps: 1) identifying the goals learners may hold, and 2) developing functions to assess how much progress learners have made towards achieving their goals. To tackle this, we identify three primary goals related to the job market that a user might pursue as well as the functions needed to estimate their progress towards their goal:

- 1) Securing a Specific Position: we need to define a function that can compute the compatibility between a learner's profile and a job description.
- 2) Boosting Marketability: we need to define a function to evaluate a learner's marketability using their profile and a collection of job descriptions.
- 3) Achieving Specialization in a Field: We also need a function to quantify a learner's level of specialization based on their profile and the target field.

Defining these functions, along with the constraints and properties they must satisfy [10], represents a promising and impactful direction for research. This approach aims to align course recommendations more closely with the realities of the job market and the individual goals of learners, thereby enhancing the relevance of such systems.

RD4: Skill-based Explainability Given the significant time and investment required to enroll in a course, such systems must offer clear explanations. Users need to understand how enrolling will advance them toward their goals. We advocate for a skill-based approach to explainability, highlighting the specific skills a course teaches and how acquiring these skills aligns with the user's objectives. Possible research directions include: 1) exploring visualization techniques to present the skill-based explanations to the learners and 2) developing sequential and explainable course recommender systems. To our knowledge, recommender systems that are both sequential and explainable have not been studied, presenting a novel research opportunity for the field of recommender systems.

RD5: Unsupervised Skill Matching Skill matching consists of aligning skills from various sources, such as job descriptions, resumes, and courses, with a skill taxonomy [14]. Because this process is crucial for creating skill-based explanations in job-oriented course recommender systems, we highlight the necessity of unsupervised methods to extract skills from resumes, course descriptions, and job postings. Recent advancements in LLMs have facilitated fewshots skill extraction from job postings through the use of synthetic dataset generation to generate demonstrations [19]. However, extending this approach to include resumes and course descriptions remains largely unexplored. Possible research directions encompass generating synthetic datasets for resumes and course descriptions to enable unsupervised skill matching. Another direction is to assess skill proficiency levels from resumes, job postings, and course descriptions. Such assessments could give the recommender system's ability to distinguish between introductory and advanced courses. To our knowledge only one supervised model has been used for estimating skill levels from resumes [3], leaving a gap in research for unsupervised approaches and other types of sources such as job postings and course descriptions. This area offers significant potential for impactful research, given its unexplored status.

RD6: Unsupervised taxonomy construction A skill-based recommender system that would flexibly adapt to evolutions of the job market requires, on top of an unsupervised skill matching tool, a method to adapt the skill taxonomy to emerging market trends. Constructing and maintaining a skill taxonomy manually is a time-consuming and labor-intensive task. Moreover, it requires state-of-the-art expertise in the domain of emerging skills. To address this challenge, unsupervised methods for automatically constructing and updating skill taxonomies can be implemented. In a practical setting, this would be performed during the skill matching step, allowing the identification of skills that fail to be matched with taxonomy items and automatically adding them to the taxonomy [35, 37].

## 5 JOB-ORIENTED COURSE RECOMMENDER

In this section, we describe a first system that satisfies some of the properties we presented in section 4.1. Our methodology comprises two steps: Unsupervised Skill Extraction and Matching ( SEM ), followed by Job-Market-Oriented Course Recommender ( JCRec ). The key question addressed by these experiments is whether, for a small company, the investment of resources and time in using RL to develop a job-oriented course recommender system is justified, or if a Greedy heuristic would suffice.

## 5.1 Skill Extraction and Matching

Following [19], we use an LLM-based system to extract skills from documents (job postings, course descriptions and resumes) and match these skills to the ESCO taxonomy [17]. We augment this pipeline with a proficiency levels extraction step, classifying each extracted skill into four categories: 'beginner", 'intermediate", 'expert" and 'unknown'. Because SEM is unsupervised, works on job postings, resumes, and course descriptions and estimates the proficiency level of skills, it satisfies property P3 while providing a first step for research direction RD5 .

As illustrated in Figure 1, the SEM pipeline, follows three steps:

- 1 Skill &amp; level Extraction from the document;
- 2 Candidate Selection from the taxonomy;
- 3 Skill Matching from the candidates;

This multi-step approach allows us to flexibly handle a taxonomy of any size without facing context window limitations. In the following, we describe in depth each of these steps.

1 Skill &amp; level Extraction. We first break down each document into individual sentences that are then grouped into sets of one or two, which are processed through the pipeline sequentially. This approach provides the LLM with some context while addressing constraints related to document length. Next, we utilize the LLM in a six-shot setting to identify skills and their associated proficiency levels within these sentence groups. To ensure an effective extraction, our few-shot demonstrations are deliberately varied. They include a range of examples: positive and negative instances, answers that are both multiple and single, as well as examples of both hard and soft skills drawn from various contexts [23].

Figure 1: Illustration of the skill extraction and matching ( SEM ) pipeline. 1 Given a document, SEM extracts relevant skills and proficiency levels from using LLM prompting. 2 For each extracted skill, three candidates from ESCO taxonomy are selected using string matching and embedding similarities. 3 The extracted skill and taxonomy candidates are prompted to an LLM to find the best match.

<!-- image -->

2 Candidate Selection. We select skill candidates from the ESCO taxonomy based on their names and definitions using two methods: 1) rule-based: We assume that if the extracted skill appears exactly in the taxonomy, it will be a good candidate for a match. When an exact match is not found, we use the token\_set\_ratio method from TheFuzz 5 , which computes the similarity by comparing shared and unique tokens between the extracted and taxonomy skills. 1) embedding-based: We compute the cosine similarity between the extracted skill and the taxonomy skills using JobBERT [6].

The candidates are selected by taking the union of the top three ESCO skills returned by each method. We chose a hybrid approach because of the complementary merits and limitations of the two methods. The rule-based method selected viable candidates but missed any synonyms, whereas the embedding-based method risked selecting contextually similar yet factually dissimilar candidates.

3 Skill Matching. Extracted skills are matched to the selected candidate's skills. Formatted candidate options (e.g., A. B. C.) are presented to the LLM, which identifies the best match if any, or indicates no match.

## 5.2 Recommendation

As illustrated in Figure 2, JCRec has two main components: an offline preprocessing phase that uses SEM , and the online recommendation. Our system is fully unsupervised, makes explainable skill-based sequential recommendations, and assumes that the user's goal is to maximize their marketability. These characteristics make our system satisfy properties P1 , P2 , P3 , P5 , and partially P4 since we assume a specific goal for the user. JCRec is also an additional step for research directions RD3 and RD4 , although user studies will be required to assess the explainability of our approach in practice. In the following, we describe in depth JCRec .

Offline Preprocessing The preprocessing phase involves extracting skills and proficiency levels from courses and jobs.

1 Course Processing. In this phase, we use SEM on each course. This method identifies all the skills required for enrollment and the skills that will be acquired upon completion. As a result, we obtain a set of processed courses C . Each course 𝑐 ∈ C is represented by two sets: 𝑐 𝑟 the required skills with their proficiency levels, and 𝑐 𝑝 the provided skills with their proficiency levels. We associate each proficiency level with a positive integer. For instance, the course Mastering Python: from beginner to expert might be represented as ( 𝑐 𝑟 : {( 𝑝𝑦𝑡ℎ𝑜𝑛, 1 )} ; 𝑐 𝑝 : {( 𝑝𝑦𝑡ℎ𝑜𝑛, 3 )}) , with the beginner level corresponding to integer 1 and the expert level to integer 3.

2 Job Processing. A similar preprocessing is applied to jobs, resulting in a set of processed jobs: J . Each job 𝑗 ∈ J is associated with a set of required skills and proficiency levels. For example, a Data Engineer job posting could be represented as 𝑗 : {( 𝑝𝑦𝑡ℎ𝑜𝑛, 2 ) , ( 𝑆𝑄𝐿, 1 )} .

Online Recommendation Using the processed courses and jobs, we can proceed to the online recommendation.

5 https://github.com/seatgeek/thefuzz

<!-- image -->

## Online

Figure 2: Illustration of the JCRec pipeline: In the offline phase, SEM is used to 1 extract skills required to take each course and skills provided by the courses, and to 2 extract skills required by each job posting skills. During the online phase, as a user uploads their CV, 1 ○ SEM extract their skills. 2 ○ These skills are used to filter the set of courses they can enroll in. 3 ○ From these, one course is recommended, aiming to maximize the increase in the number of jobs the user can apply for. 4 ○ The user's profile is then updated with the skills acquired from the recommended course. 5 ○ In the case of sequential recommendation, steps 2 ○ , 3 ○ , and 4 ○ are repeated until 𝑘 courses are recommended.

- 1 ○ Resume Processing. JCRec begins by processing the user's resume. Using SEM , it extracts the user's skills and proficiency levels, creating a set 𝑢 consisting of pairs of skills and their corresponding proficiency levels. For instance, a junior data scientist's skill set might be represented as 𝑢 : {( 𝑝𝑦𝑡ℎ𝑜𝑛, 2 ) , ( 𝑚𝑎𝑐ℎ𝑖𝑛𝑒𝑙𝑒𝑎𝑟𝑛𝑖𝑛𝑔, 2 )} .
- 2 ○ Course Filtering. In the second step, we use a user-course relevance function ucr ( 𝑢, 𝑐 ) to determine C 𝑢 : the set of courses available for enrollment by user 𝑢 . The relevance function ucr ( 𝑢, 𝑐 ) is defined in Appendix A.2, along with the desirable properties it should satisfy. A threshold 𝑡 𝑢𝑐 is set to filter courses, keeping those with a relevance score higher than 𝑡 𝑢𝑐 . Thus, C 𝑢 is defined as C 𝑢 = { 𝑐 ∈ C| ucr ( 𝑢, 𝑐 ) ≥ 𝑡 𝑢𝑐 } .
- 3 ○ Course Recommendation. Next, we recommend a course 𝑐 ∈ C 𝑢 . Given the nature of our problem as a sequential decision-making task, we use Reinforcement Learning (RL), as commonly done for sequential recommender systems [2]. We define our problem's Markov Decision Process (MDP) as follows: the state is 𝑢 the user's skill set; the action space is C , the set of courses; and the reward to maximize is the marketability that we define as the number of jobs the user can apply to, denoted | J 𝑢 | . Moreover, if the agent recommends a course that is not in C 𝑢 , i.e. that the user cannot or should not follow, the reward is set to -1 and the episode is terminated. This choice of reward is motivated by the fact that we are only considering users whose goal is to increase their marketability. If different goals were considered, additional rewards would be required. Finally, the transition probabilities from state to state are deterministic: completing a course adds skills 𝑐 𝑝 to 𝑢 , resulting in probabilities of either 0 or 1. 4 ○ Update After recommending course 𝑐 , the user's skills 𝑢 are updated with 𝑐 𝑝 : the skills provided by course 𝑐 . The number of jobs user 𝑢 can apply to | J 𝑢 | is computed using the similarity function ujs ( 𝑢, 𝑗 ) defined in Appendix A.1. To determine | J 𝑢 | , we set a threshold 𝑡 𝑢𝑗 and consider that a user 𝑢 can apply to a job 𝑗 if and

only if ujs ( 𝑢, 𝑗 ) ≥ 𝑡 𝑢𝑗 , making the set of applicable jobs defined as: J 𝑢 = { 𝑗 ∈ J| ujs ( 𝑢, 𝑗 ) ≥ 𝑡 𝑢𝑗 } .

- 5 ○ Repeat To recommend a sequence of courses, we repeat steps 2 ○ , 3 ○ , and 4 ○ until 𝑘 courses have been recommended.

## 6 EXPERIMENTAL SETUP

## 6.1 Datasets

In our experiments and evaluation, we used publicly available datasets. We concentrated specifically on English-language documents within the information technology (IT) and IT management industry, due to the ease of access and relevance to our study's focus.

Taxonomy. We use the ESCO [18] taxonomy that comprises 13 890 skills and competencies. We filter out competencies irrelevant to IT and IT management roles, resulting in a subset of 1 794 skills.

Job Postings. We scraped around 3,500 English job postings in the domain of Information Technology from various online platforms.

Course Descriptions. The COCO dataset [9] contains 43 113 online Udemy courses, with their descriptions, objectives, and prerequisite requirements. The dataset is divided into 133 detailed categories and covers 46 languages. We first filter the dataset for only English courses (34 111 courses). Then, we filter the granular categories to include only the 27 that are related to IT and IT management (leaving 12 291 courses). In our experiments, we took a random sub-sample of 3 000 courses.

Resumes. We used two resume datasets from Kaggle for this study. The first dataset 6 comprises 2 484 resumes spanning 24 industries. We filter the dataset to include only the Information-Technology category. The second dataset 7 contains 166 resumes of professionals working in the IT department. We exclude resumes from non-IT personnel (i.e., in HR , Arts , or Health and fitness categories). Together, we keep a subset of 233 anonymized resumes from IT professionals.

[6 https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)

[7 https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset.](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset.)

Table 1: Statistics of post-processed documents

|                    |   Jobs | Courses   | Courses   |   CVs |
|--------------------|--------|-----------|-----------|-------|
| average            |        | Prereqs   | Target    |       |
| Words per sentence |   19.4 | 19.5      | 32.7      |  23.7 |
| Sentences per doc  |   26.9 | 2.0       | 2.4       |  29.7 |
| Words per doc      |  510.5 | 38.6      | 77.8      | 702.4 |

## 6.2 Pre-processing

For job postings and course descriptions, we remove the document if the combined text body contains less than 50 and 20 words respectively, a cutoff driven by examinations of documents below these thresholds. For courses, we split descriptive textual features into two categories: text describing prerequisite skills to take the course ('requirements' field), and text describing skills taught in the course ('course objectives' and 'course description'). In addition, we chose to include only short descriptions and not long descriptions of the COCO courses because we found too many irrelevant and repetitive keywords mentioned in the long descriptions. With each document type processed, we segment the text bodies into individual sentences for analysis by SEM ˙ Table 1 shows the varying lengths of these documents, indicating notably longer job postings and resumes compared to courses. This difference arises as we only use short course descriptions. Furthermore, through qualitative analysis, we find that sentences in resumes tend to be independent while those in jobs and courses are contextual. Hence, we process each resume sentence independently, while job and course sentences are handled in pairs.

## 6.3 Skill matching setup

We run the job postings, course descriptions, and resumes through the matching pipeline, from which we extract 165K, 30K, and 31K total skills respectively. While we do not have annotated data, we can examine the results with heuristics. Table 2 indicates that resumes contain the most extracted skills but show the lowest percentage of proficiency level identification per skill. This trend aligns with common resume practices in the IT sector, where professionals often list numerous skills without specifying their proficiency levels. Moreover, course prerequisites and descriptions, as shown in Table 2, feature the fewest skills. This aligns with the expectation that courses have a limited scope, making the teaching of a vast number of skills impractical. Conversely, the number of skills extracted from job postings at 43.2 skills per document, or about 1.6 skills per sentence is justifiable. IT sector job postings often list many required skills, reflecting this sector's diverse skill demands.

We also validated the performance of the proficiency level extraction method. An expert manually annotated 212 skills from 10 job postings in the IT sector. We compared the annotations with the levels extracted by our system and obtained an accuracy of 0.68.

Furthermore, we explore the results of the extraction on each type of document. Table 3 shows the percentage of the extracted proficiency levels that are "expert", "intermediate", "beginner", "unknown", or "other". Levels found in job descriptions are primarily

Table 2: Statistics on skills standardized by document type: "Skills extracted" shows the average unique skills extracted, "Levels extracted" indicates the average percentage of skills with identified levels, and "Skills matched" represents the average unique skills matched to the taxonomy.

|                  | Jobs   | Courses   | Courses   | CVs   |
|------------------|--------|-----------|-----------|-------|
| average per doc  |        | Prereqs   | Target    |       |
| Skills extracted | 43.2   | 3.2       | 6.8       | 112.9 |
| Levels extracted | 71.1%  | 72.0%     | 79.6%     | 45.4% |
| Skills matched   | 16.6   | 2.2       | 4.2       | 22.6  |

Table 3: Distribution of extracted skill mastery levels for each document type. other indicate cases where the LLM failed to output a level in one of the 4 predefined categories.

|                |   Jobs | Courses   | Courses   |   CVs |
|----------------|--------|-----------|-----------|-------|
| %of all skills |        | Prereqs   | Target    |       |
| "expert"       |   49.2 | 0.5       | 3.0       |  21.8 |
| "intermediate" |   19.8 | 9.0       | 29.5      |  20.6 |
| "beginner"     |    2.1 | 51.7      | 47.1      |   3.0 |
| "unknown"      |   28.3 | 38.0      | 20.1      |  54.4 |
| other          |    0.6 | 0.8       | 0.3       |   0.2 |

"expert" indicating when employers suggest desired skill levels, nearly 50% of the time, they look for experts in that skill. This is expected and agrees with qualitative observations of job descriptions. More than 54% of levels from resumes are "unknown", which is reasonable as stated previously. However, when IT professionals do indicate their skill levels, it is also expected that skills are either at the "expert" level or "intermediate" level. This also agrees with qualitative observations, since professionals tend not to list a skill unless they are proficient. Looking at courses, levels of skills found in both course prerequisites and objectives are primarily "beginners". This is expected because online courses tend not to be in-depth and are geared toward beginners. It is also expected that very few courses (less than 10%) have intermediate or expert level prerequisites, whereas over 30% of courses teach intermediate or expert level skills. Finally, we observe that despite indicating in the prompt to output only specific words, all documents output a small percentage (under 1%) of non-conforming levels.

Finally, we replace unknown levels with the following heuristic: if a level is unknown for a learner it will be set to beginner and if a level is unknown for a job, it will be set to expert. This heuristic allows us to have a conservative system that will not assume that a beginner would be able to apply to a job that requires expert knowledge. Regarding unknown proficiency levels for courses, we replace them with intermediate and correct the following types of inconsistencies: if a course requires skill 𝑠 at level 𝑙 𝑟 and also teaches the same skill 𝑠 at a lower level 𝑙 𝑝 such that 𝑙 𝑟 ≥ 𝑙 𝑝 this implies an inconsistency: the course demands a higher skill level than it provides. In such cases, we adjust 𝑙 𝑟 to the level before 𝑙 𝑝 to ensure the removal of any such inconsistencies. For skill extraction and skill matching, we use GPT-3.5-turbo with temperature set to 0, top-p set to 1.0, frequency penalty set to 0.0, and presence penalty set to 0.0.

## 6.4 Recommendation Algorithms

In this work, we compare four sequential recommendation algorithms: exhaustive, greedy, and two RL-based algorithms.

Exhaustive Recommendation : The exhaustive approach evaluates all possible course sequences that can be recommended to a learner. For every possible sequence, it computes | J 𝑢 | the number of jobs the learner can apply to after updating their profile. The sequence that maximizes | J 𝑢 | is then recommended. Although this approach guarantees the recommendation of the optimal sequence, its practical feasibility is limited due to the high time complexity that increases exponentially with the number of recommended courses: 𝑂 (|J| · |C| 𝑘 ) . Nevertheless, we have chosen to present the outcomes of this methodology, despite its impracticality for real-world application, as it provides a theoretical maximum for the value of | J 𝑢 | .

Greedy Recommendation : At each of the 𝑘 steps, the greedy algorithm recommends a course that maximizes the number of job opportunities available to the learner. This approach is greedy because it selects the best immediate option at each step without guaranteeing an overall optimal sequence. Despite this sub-optimality, it is more time-efficient than the exhaustive approach. For each course recommendation, the learner's profile is updated and the similarity with every job is computed to estimate the course leading to the highest increase in job applicability. This results in a total number of operations on the order of 𝑂 ( 𝑘 · | J | · |C|) .

Reinforcement Learning Recommendation : We compare 2 RL algorithms: Deep Q Network (DQN) [22] and Proximal Policy Optimization (PPO) [32]. For both RL algorithms, we used the implementation provided by Stable-Baselines3 [28] with default parameter values. The input size is the number of skills |S| and the output dimension is the number of actions, or courses, |C| . This makes the time complexity of these algorithms 𝑂 ( 𝑘 · |S| · |C|) , meaning that RL has the potential to be more efficient than the greedy heuristic, especially if the number of skills |S| is one or more order of magnitude smaller than the number of jobs | J | .

Hyperparameters : Both threshold values 𝑡 𝑢𝑐 and 𝑡 𝑢𝑗 were set to 0.8, the RL agents were trained for 5 000 000 steps.

## 6.5 Evaluation

We evaluate the algorithms using different values of 𝑘 : the number of courses recommended in the sequence. Our evaluation employs two metrics to compare the algorithms: 1) The average number of jobs a learner can apply to after receiving 𝑘 course recommendations assessing the relevance of the courses recommended, and 2) the average time taken to make a recommendation, evaluating the algorithms' feasibility for real-world deployment. Evaluation is conducted on a subset of the dataset described in section 6.1, including 100 jobs, 100 courses, and all resumes containing less than 15 skills for a total of 52 resumes. We decided to limit the number of skills present in the resume to emulate a learner new to the job market.

## 7 RESULTS

Table 4 presents a comparison of the Exhaustive, Greedy, DQN, and PPO algorithms across different sequence lengths ( 𝑘 ). Rwd is the average number of jobs learners can apply to after taking the recommended courses and Time is the average duration for each recommendation in milliseconds.

The Exhaustive algorithm, although delivering the best recommendation quality in terms of reward is not suited for real-world deployment due to its significant time consumption. Even on a small dataset of 100 courses and jobs, it can take up to 10 seconds to recommend a sequence of 3 courses. Its usage becomes unfeasible as the sequence length increases, rendering it unsuitable for practical applications. However, the exhaustive algorithm provides an upper bound on the reward as it outputs the optimal course sequence. Nevertheless, in the remainder of our analysis, we will not compare the exhaustive approach to the greedy and RL algorithms.

For shorter sequences ( 𝑘 = 1 , 2 , 3 ), the Greedy algorithm achieves the highest Rwd with reasonable recommendation times. Particularly at lower values of 𝑘 , the Greedy heuristic offers a balance between speed and reward, making it the preferred approach when recommending a smaller number of courses.

RL approaches demonstrate their strength in efficiency, particularly for recommending longer sequences of courses ( 𝑘 = 4 , 5 ). PPO, in particular, stands out for maintaining a recommendation quality on par with the Greedy algorithm but with significantly faster execution (one order of magnitude faster than the Greedy approach). This characteristic makes PPO a more suitable option for longer sequence recommendations. Overall, the RL algorithms, DQN and PPO, show potential as the fastest methods, especially in scenarios with larger datasets. When longer sequences of courses are required, RL methods offer a more efficient alternative without compromising the quality of recommendations.

In summary, our results suggest using different approaches to course recommendation based on the sequence length: the Greedy heuristic for shorter sequences where fewer courses are recommended, and RL approaches, particularly PPO, for longer sequences and larger datasets where efficiency becomes paramount.

## 8 LIMITATIONS

While our work contributes valuable insights and research directions to the field of job-oriented course recommender systems, it is important to acknowledge several limitations that may impact the generalizability and effectiveness of our recommender system.

Language Restriction. Our system currently operates exclusively in English, restricting the applicability of our model in multilingual contexts and excluding non-English job postings and courses.

Dataset Size. The datasets used in this study are relatively small, limiting the robustness and scalability of our findings.

Evaluation. The lack of expert annotators to validate the course recommendation means our evaluation may not fully capture the system's effectiveness in real-world scenarios.

Reliance on Heuristics. Our approach includes several heuristic methods. While they are necessary for handling data complexities, these heuristics introduce an element of subjectivity and may not always reflect real-world learning and job market dynamics.

Assumption of Skill Acquisition. A fundamental assumption in our sequential recommendation approach is that completing a course will automatically result in the acquisition of the associated skills by the learners. This overlooks the variability in individual learning outcomes and the probability of successfully acquiring new skills. A probabilistic approach that estimates the likelihood of skill acquisition would provide a more nuanced and realistic model.

Table 4: Evaluation of the 4 course recommendation algorithms. Rwd is the reward i.e. the average number of jobs learners can apply to after following the course recommendations. The highest rewards among Greedy, DQN, and PPO are highlighted in bold for each 𝑘 . Time is the average time in milliseconds needed for the recommendations. The lowest time among Greedy, DQN, and PPO is underlined for each 𝑘 . The case 𝑘 = 0 reflects the initial learner state, identical across all algorithms. The exhaustive algorithm was not run for 𝑘 = 4 and 𝑘 = 5 due to estimated times exceeding 10 6 seconds per learner.

| Model      | 𝑘 = 0   | 𝑘 = 1   | 𝑘 = 1     | 𝑘 = 2   | 𝑘 = 2     | 𝑘 = 3   | 𝑘 = 3     | 𝑘 = 4   | 𝑘 = 4     | 𝑘 = 5   | 𝑘 = 5     |
|------------|---------|---------|-----------|---------|-----------|---------|-----------|---------|-----------|---------|-----------|
| Model      | Rwd     | Rwd     | Time (ms) | Rwd     | Time (ms) | Rwd     | Time (ms) | Rwd     | Time (ms) | Rwd     | Time (ms) |
| Exhaustive | 0.1     | 1.0     | 16        | 2.5     | 10 2      | 5.6     | 10 4      | NA      | NA        | NA      | NA        |
| Greedy     | 0.1     | 1.0     | 16        | 2.0     | 41        | 4.1     | 73        | 7.0     | 10 2      | 10.5    | 10 2      |
| DQN        | 0.1     | 0.9     | 4.8       | 1.7     | 8.7       | 4.1     | 11        | 6.3     | 17        | 7.1     | 24        |
| PPO        | 0.1     | 0.5     | 3.7       | 1.1     | 6.4       | 3.4     | 11        | 8.0     | 20        | 10.4    | 21        |

## 9 CONCLUSION

In this work, we provided the perspective of academic researchers working in collaboration with industry practitioners to develop and deploy a job-market-oriented course recommender system. We proposed to rethink course recommender systems to consider the job market, several properties that such systems should satisfy, and research directions that would help develop this field. We introduced SEM , a skill extraction and matching method that efficiently aligns skills extracted from resumes, course content, and job descriptions with the ESCO taxonomy. Utilizing in-context learning and Large Language Models (LLMs), SEM is fully unsupervised, enabling generalization to any document type and adaptation to an ever-evolving job market. Building on this foundation, we developed an unsupervised course recommender system that leverages the matched skills to suggest course sequences aimed at maximizing employment opportunities. Our investigation of sequential recommendation strategies - including a greedy heuristic, an exhaustive approach, and two Reinforcement Learning (RL) models - revealed insightful findings. Notably, the RL approaches, particularly Proximal Policy Optimization (PPO), stand out in recommending longer sequences efficiently. They offer a promising solution for larger datasets, balancing recommendation quality with computational efficiency.

## A FUNCTION DESIGN MOTIVATION

In this section, we motivate our design choices for the user-job similarity function and the user-course relevance function described in section 5.2. Drawing inspiration from axiomatic Information Retrieval [11], we have designed these similarity functions based on a set of desirable constraints they should fulfill.

## A.1 User-Job Similarity

First, we describe the constraints for the user-job similarity function: UJC1: Assign a higher score to a user who possesses more skills required for a job.

UJC2: Assign a higher or equal score to a user with higher proficiency levels in the skills required for a job.

IJC3: Assign the maximal score to a user who has all the skills and proficiency levels required for a job.

Based on these constraints, we propose the following user-job similarity function, denoted as uj-sim:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

Here sl 𝑠,𝑗 (respectively sl 𝑠,𝑢 ) represents the skill proficiency level for skill 𝑠 in job 𝑗 (respectively for user 𝑢 ). If a user does not have skill 𝑠 , then sl 𝑠,𝑢 = 0 . The use of min in the skill-skill similarity function (equation 2), ensures that users matching all requirements will achieve maximum similarity, satisfying constraint UJC3 .

## A.2 User-Course Relevance

Next, we define the constraints for the user-course relevance:

UCC1: Assign a higher score to a user who possesses more skills required for a course.

UCC2: Assign a higher or equal score to a user with higher proficiency levels in the skills required for a course.

UCC3: Assign the minimal score to a user who already possesses all the skills and proficiency levels provided by a course.

UCC3 is based on the rationale that if a user already knows everything taught by a course, they will gain no new skills making the course irrelevant.

We propose the following user-course relevance function, uc-rel:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

In these equations, sl 𝑠,𝑐 𝑟 (respectively sl 𝑠,𝑐 𝑝 ) indicates the required (respectively provided) proficiency level for skill 𝑠 in course 𝑐 . Because ucp ( 𝑢, 𝑐 𝑝 ) is bounded between 0 and 1 (from equation 2) and returns 1 if and only if user 𝑢 already possesses all the skills provided by the course, the overall relevance uc-rel ( 𝑢, 𝑐 ) will be equal to 0 (the lowest possible value) ensuring constraint UCC3 .

## REFERENCES

- [1] Solmaz Abdi, Hassan Khosravi, Shazia W. Sadiq, and Dragan Gasevic. 2020. Complementing educational recommender systems with open learner models. In LAK '20: 10th International Conference on Learning Analytics and Knowledge, Frankfurt, Germany, March 23-27, 2020 , Christoph Rensing and Hendrik Drachsler (Eds.). ACM, 360-365.
- [2] MMehdi Afsar, Trafford Crump, and Behrouz Far. 2022. Reinforcement learning based recommender systems: A survey. Comput. Surveys 55, 7 (2022), 1-38.
- [3] Anindita Sinha Banerjee, Sachin Pawar, Girish K Palshikar, Devavrat Thosar, Jyoti Bhat, and Payodhi Mandloi. 2022. Estimating Skill Proficiency from Resumes. In Pacific-Asia Conference on Knowledge Discovery and Data Mining . Springer, 105-118.
- [4] Koen Bothmer and Tim Schlippe. 2023. Skill Scanner: Connecting and Supporting Employers, Job Seekers and Educational Institutions with an AI-Based Recommendation System. In Innovative Approaches to Technology-Enhanced Learning for the Workplace and Higher Education , David Guralnick, Michael E. Auer, and Antonella Poce (Eds.). Springer International Publishing, Cham, 69-80.
- [5] Benjamin Clavié and Guillaume Soulié. 2023. Large Language Models as Batteries-Included Zero-Shot ESCO Skills Matchers. arXiv preprint arXiv:2307.03539 (2023).
- [6] Jens-Joris Decorte, Jeroen Van Hautte, Thomas Demeester, and Chris Develder. 2021. Jobbert: Understanding job titles through skills. arXiv preprint arXiv:2109.09605 (2021).
- [7] Jens-Joris Decorte, Severine Verlinden, Jeroen Van Hautte, Johannes Deleu, Chris Develder, and Thomas Demeester. 2023. Extreme Multi-Label Skill Extraction Training using Large Language Models. arXiv preprint arXiv:2307.10778 (2023).
- [8] David J Deming and Kadeem Noray. 2020. Earnings dynamics, changing job skills, and STEM careers. The Quarterly Journal of Economics 135(4) (2020), 1965-2005.
- [9] Danilo Dessì, Gianni Fenu, Mirko Marras, and Diego Reforgiato Recupero. 2018. COCO: Semantic-Enriched Collection of Online Courses at Scale with Experimental Use Cases. In Trends and Advances in Information Systems and Technologies , Álvaro Rocha, Hojjat Adeli, Luís Paulo Reis, and Sandra Costanzo (Eds.). Springer International Publishing, Cham, 1386-1396.
- [10] Hui Fang, Tao Tao, and ChengXiang Zhai. 2004. A formal study of information retrieval heuristics. In SIGIR 2004: Proceedings of the 27th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, Sheffield, UK, July 25-29, 2004 , Mark Sanderson, Kalervo Järvelin, James Allan, and Peter Bruza (Eds.). ACM, 49-56. https://doi.org/10.1145/1008992.1009004
- [11] Hui Fang and ChengXiang Zhai. 2005. An exploration of axiomatic approaches to information retrieval. In SIGIR 2005: Proceedings of the 28th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, Salvador, Brazil, August 15-19, 2005 , Ricardo A. Baeza-Yates, Nivio Ziviani, Gary Marchionini, Alistair Moffat, and John Tait (Eds.). ACM, 480-487. https: //doi.org/10.1145/1076034.1076116
- [12] Jibril Frej, Neel Shah, Marta Kneževi´ c, Tanya Nazaretsky, and Tanja Käser. 2023. Finding Paths for Explainable MOOC Recommendation: A Learner Perspective. arXiv preprint arXiv:2312.10082 (2023).
- [13] Aritra Ghosh, Beverly Woolf, Shlomo Zilberstein, and Andrew Lan. 2020. Skillbased career path modeling and recommendation. In 2020 IEEE International Conference on Big Data (Big Data) . IEEE, 1156-1165.
- [14] Ann-sophie Gnehm, Eva Bühlmann, Helen Buchs, and Simon Clematide. 2022. Fine-Grained Extraction and Classification of Skill Requirements in GermanSpeaking Job Ads. In Proceedings of the Fifth Workshop on Natural Language Processing and Computational Social Science (NLP+CSS) . Association for Computational Linguistics, Abu Dhabi, UAE, 14-24. https://doi.org/10.18653/v1/ 2022.nlpcss-1.2
- [15] Weijie Jiang, Zachary A. Pardos, and Qiang Wei. 2019. Goal-Based Course Recommendation. In Proceedings of the 9th International Conference on Learning Analytics &amp; Knowledge (Tempe, AZ, USA) (LAK19) . Association for Computing Machinery, New York, NY, USA, 36-45.
- [16] Asra Khalid, Karsten Lundqvist, Anne Yates, and Mustansar Ali Ghzanfar. 2021. Novel online recommendation algorithm for massive open online courses (NoRMOOCs). Plos one 16, 1 (2021), e0245485.
- [17] Martin le Vrang, Agis Papantoniou, Erika Pauwels, Pieter Fannes, Dominique Vandensteen, and Johan De Smedt. 2014. ESCO: Boosting job matching in Europe with semantic interoperability. Computer 47, 10 (2014), 57-64.
- [18] Martin le Vrang, Agis Papantoniou, Erika Pauwels, Pieter Fannes, Dominique Vandensteen, and Johan De Smedt. 2014. ESCO: Boosting Job Matching in Europe with Semantic Interoperability. Computer 47, 10 (2014), 57-64. https: //doi.org/10.1109/MC.2014.283
- [19] Antoine Magron, Anna Dai, Mike Zhang, Syrielle Montariol, and Antoine Bosselut. 2024. JOBSKAPE: A Framework for Generating Synthetic Job Postings to Enhance Skill Matching. (2024). arXiv:2402.03242 [cs.CL]
- [20] Mirko Marras, Ludovico Boratto, Guilherme Ramos, and Gianni Fenu. 2021. Equality of Learning Opportunity via Individual Fairness in Personalized Recommendations. International Journal of Artificial Intelligence in Education 32 (10

2021), 1-49.

- [21] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems 26 (2013).
- [22] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. 2013. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602 (2013).
- [23] Khanh Cao Nguyen, Mike Zhang, Syrielle Montariol, and Antoine Bosselut. 2024. Rethinking Skill Extraction in the Job Market Domain using Large Language Models. (2024). arXiv:2402.03832 [cs.CL]
- [24] Robert Palmer. 2017. Jobs and skills mismatch in the informal economy. ILO. https://www. ilo. org/wcmsp5/groups/public/-ed\_emp/-ifp\_skills/documents/publication/wcms\_629018. pdf (2017).
- [25] Jeffrey Pennington, Richard Socher, and Christopher D Manning. 2014. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) . 1532-1543.
- [26] Guangyuan Piao and John G Breslin. 2016. Analyzing MOOC entries of professionals on LinkedIn for user modeling and personalized MOOC recommendations. In Proceedings of the 2016 conference on user modeling adaptation and personalization . 291-292.
- [27] Boyd A. Potts, Hassan Khosravi, Carl Reidsema, Aneesha Bakharia, Mark Belonogoff, and Melanie Fleming. 2018. Reciprocal Peer Recommendation for Learning Purposes. In Proceedings of the 8th International Conference on Learning Analytics and Knowledge (Sydney, New South Wales, Australia) (LAK '18) . Association for Computing Machinery, New York, NY, USA, 226-235.
- [28] Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. 2021. Stable-baselines3: Reliable reinforcement learning implementations. The Journal of Machine Learning Research 22, 1 (2021), 1234812355.
- [29] Siriporn Sakboonyarat and Panjai Tantatsanawong. 2019. Massive open online courses (MOOCs) recommendation modeling using deep learning. In 2019 23rd International computer science and engineering conference (ICSEC) . IEEE, 275280.
- [30] Juan Camilo Sanguino Perez, Ruben Francisco Manrique, Olga Mariño, Mario Linares Vásquez, and Nicolás Cardozo. 2022. A course hybrid recommender system for limited information scenarios. Journal of Educational Data Mining 14, 3 (Dec. 2022), 162-188. https://jedm.educationaldatamining.org/index.php/ JEDM/article/view/608
- [31] Viddhesh Sankhe, Janice Shah, Tejas Paranjape, and Radha Shankarmani. 2020. Skill Based Course Recommendation System. In 2020 IEEE International Conference on Computing, Power and Communication Technologies (GUCON) . IEEE, 573-576.
- [32] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017).
- [33] Mehbooba P. Shareef, Linda Rose Jimson, and Babita R. Jose. 2022. Hybrid Explainable Educational Recommender Using Self-attention and KnowledgeBased Systems for E-Learning in MOOC Platforms. In Responsible Data Science , Jimson Mathew, G. Santhosh Kumar, Deepak P., and Joemon M. Jose (Eds.). Springer Nature Singapore, Singapore, 61-74.
- [34] Ying Sun, Fuzhen Zhuang, Hengshu Zhu, Qing He, and Hui Xiong. 2021. Costeffective and interpretable job skill recommendation with deep reinforcement learning. In Proceedings of the Web Conference 2021 . 3827-3838.
- [35] Kunihiro Takeoka, Kosuke Akimoto, and Masafumi Oyamada. 2021. Lowresource taxonomy enrichment with pretrained language models. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing . 2747-2758.
- [36] Xuetao Tian and Feng Liu. 2021. Capacity Tracing-Enhanced Course Recommendation in MOOCs. IEEE Transactions on Learning Technologies 14, 3 (2021), 313-321.
- [37] Hongyuan Xu, Ciyi Liu, Yuhang Niu, Yunong Chen, Xiangrui Cai, Yanlong Wen, and Xiaojie Yuan. 2023. TacoPrompt: A Collaborative Multi-Task Prompt Learning Method for Self-Supervised Taxonomy Completion. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing . 15804-15817.
- [38] Eva Zangerle and Christine Bauer. 2022. Evaluating recommender systems: survey and framework. Comput. Surveys 55, 8 (2022), 1-38.
- [39] Jing Zhang, Bowen Hao, Bo Chen, Cuiping Li, Hong Chen, and Jimeng Sun. 2019. Hierarchical Reinforcement Learning for Course Recommendation in MOOCs. In The Thirty-Third AAAI Conference on Artificial Intelligence, AAAI 2019, The Thirty-First Innovative Applications of Artificial Intelligence Conference, IAAI 2019, The Ninth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2019, Honolulu, Hawaii, USA, January 27 - February 1, 2019 . AAAI Press, 435-442.
- [40] Mike Zhang, Kristian Nørgaard Jensen, and Barbara Plank. 2022. Kompetencer: Fine-grained Skill Classification in Danish Job Postings via Distant Supervision and Transfer Learning. In Proceedings of the Thirteenth Language Resources and Evaluation Conference . European Language Resources Association, Marseille,

[France, 436-447. https://aclanthology.org/2022.lrec-1.46](https://aclanthology.org/2022.lrec-1.46)

- [41] Mike Zhang, Kristian Nørgaard Jensen, Sif Dam Sonniks, and Barbara Plank. 2022. SkillSpan: Hard and Soft Skill Extraction from English Job Postings. In 2022 Annual Conference of the North American Chapter of the Association for Computational Linguistics . Association for Computational Linguistics.
- [42] Peide Zhu, Claudia Hauff, and Jie Yang. 2022. MOOC-Rec: Instructional Video Clip Recommendation for MOOC Forum Questions. In Proceedings of the 15th

International Conference on Educational Data Mining , Antonija Mitrovic and Nigel Bosch (Eds.). International Educational Data Mining Society, Durham, United Kingdom, 705-709.

Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009