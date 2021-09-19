**ProFAB – Open Protein Functional Annotation Benchmark**

Author11,2, Author22,2,

**Abstract**

Background

As the number of protein sequences in protein databases increases, accurate computational methods are required to annotate the available data. For this purpose, several machine learning methods have been proposed, however, two main issues in the evaluation of computational prediction methods are the construction of reliable positive and their negative training/validation datasets and the fair evaluation of performances based on predefined experimental settings. To cope with them, we developed ProFAB, Open Protein Functional Annotation Benchmark which is a platform that presents a fair comparison for protein function prediction methods.

Findings

We evaluated ProFAB against classifications of enzyme commission numbers and GO terms, separately. The results are given according to machine learning algorithms and protein descriptions

Conclusion

ProFAB presents numerical protein datasets and enables them to train and to evaluate through many alternative functions. We believe that ProFAB will be useful for both computer scientists to find ready-to-use biological datasets, and wet-lab researchers to utilise ready-to-use machine learning algorithms to gain pre-knowledge about the functional view of proteins. ProFAB open source code is accessible at https://github.com/Sametle06/ProFAB.git.

**Findings**

**Background**

Machine learning methods have started to be incorporated into standard research and development pipelines to solve biological and chemical problems such as understanding inner mechanisms of cells, identification of protein functions and development of new drugs. The aim of machine learning methods is to provide accurate predictions for the problem of interest with the aim of reducing the cost and the time of experimental procedures.(Moreau &amp; Tranchevent 2012). Recently, significant advances have been made in several areas such as anomaly detection (Pang, 2020), natural language processing (Young, 2018) and cancer diagnosis (Sidey-Gibbons, 2019), drug discovery (Vamathevan, 2019), protein function prediction (Zhou, 2019) and protein structure prediction areas (Senior, 2020) with the help of machine learning algorithms.For example, AlphaFold2 has made a significant improvement in protein structure prediction using deep learning methods (Jumper, 2021).

One of the main issues in the training and evaluation of the machine learning methods is creating reliable training/validation/test datasets that can be used to evaluate the performances of the predictive models and to compare the obtained results with other methods. Several benchmarking platforms have recently been proposed in different areas to provide datasets, machine learning algorithms and evaluation metrics. Therapeutics Data Commons, (TDC) (Zitnik , 2021) provides ready-to-use biomedical datasets for drug-target prediction, toxicity prediction, antibody development and several evaluation metrics. MoleculeNet (Pande, 2017) is another benchmarking platform that provides several datasets about quantum mechanics, cheminformatics and physiology, machine learning algorithms and applicable cost metrics. ChemML (Haghighatlari, 2020) is a machine learning and informatics program package that enables users to execute machine learning algorithms in the domain of chemical and materials concepts. Open Graph Benchmark (Hu, 2021) is a platform that provides datasets and machine learning algorithms for social and biological networks and knowledge graphs analysis.

In the field of protein function prediction, Critical Assessment of Functional Annotation (CAFA) challenge (Zhou, 2019) is an important initiative where the aim is to evaluate the performances of automated protein function prediction methods. CAFA challenge is organised about every two years; however, it is a one-time challenge and it is not trivial to repeat the challenge with the same experimental settings afterwards. In addition, CAFA challenge does not provide any training dataset, machine learning models or different data splitting strategies (e.g., temporal, similarity-based and random split settings). To bring an end to these limitations, here, we propose an open-source Python package called ProFAB, Open Protein Functional Annotation Benchmark platform. The aim of ProFAB is to create a fair comparison platform for protein function prediction methods based on Gene Ontology (GO) (861,299 proteins annotated to 8360 function terms) and Enzyme Commission (EC) numbers (563,554 proteins annotated to 269 enzymatic functions). ProFAB supplies both positive and negative datasets separately for each function. To construct the datasets, three splitted methods based on data points randomness, similarity and annotation times were applied and they are served to use. To provide a fair platform for benchmarking, three functions are presented which are scaling, training and evaluation. Scaling is to decrease biases between points. Training is employed to train the datasets which can be obtained from ProFAB or can be supplied explicitly by the user. Scoring function is to evaluate the performances of the models coming out from the training section. To see results in every way, different aspects of evaluation such as recall, precision and accuracy are provided. In all perspectives, ProFAB is designed as a fair benchmarking platform, where researchers can use numerical data of proteins or can use their numerical data, can apply different machine learning methods and can see model performances in different aspects.

**Materials and Methods**

**Overview of the Tool**

This tool presents a bench that provides datasets, learning types and evaluation. These are collected in five substructures that are independent to each other:

- Two tasks of datasets are collected for enzyme class and  gene ontology term prediction. To split the data and to see performances on different conditions, three different splitting methods are used which are random splitting, similarity splitting and temporal splitting. This part was directly integrated on data and it is not given as functionality in ProFAB.
- The scaling function is to make data unitless and to decrease ranges between data points.
- The training section consists of machine learning algorithms that work for binary classification.
- To benchmark the result, ProFAB provides various metrics.

In Figure-1, ProFAB&#39;s workflow can be seen:

![](RackMultipart20210919-4-1781s4q_html_a92a38f1ab64bdff.png)

_Figure 1 ProFAB Benchmark: the collection of annotation datasets for proteins with different splitting methods, scaling functions, machine learning algorithms and evaluation metrics__._

**Dataset Architecture :** The main goal of ProFAB is to provide ready-to-use datasets for Enzyme Commission Number Prediction (ECNo) and Gene Ontology(GO) terms, therefore, their architecture was carefully planned to generate a fair platform to apply prediction algorithms. The construction of sets starts from the collection of protein sequences from UniProt/SwissProt, separating these collections into positive and negative sets and ends with the splitting of positive and negative sets in terms of randomness, similarity and annotation time. The obtained data from UniProt database is protein sequence data which can be converted into numerical features that machines can understand. This process is done with the help of web-tool iLearn that provides various descriptors that use amino acid composition, sequence order and hydrophobicity (Chen, 2020). In datasets, six different protein descriptors are used that are amino acid composition (AAC), pseudo amino acid composition (PAAC), sequence order coupling number (SOCNumber), conjoint triad (CTriad), grouped amino acid composition (GAAC). While changing from sequence to numerical description, some of the data points because of non-valid sequences for iLearn web-tool.

**Enzyme Commission Number Prediction Dataset (EC-data):**ECNo is a nomenclature technique that is used to classify the proteins according to reactions they catalyze (Cornish-Bowden, 2014). Therefore, this dataset was prepared so that enzymes can be classified with regard to their commission numbers based on their amino acid sequences.The protein sequence data and their ECNo were obtained from UniProt/SwissProt. In total 312570 proteins were used to prepare the sets.

       ![](RackMultipart20210919-4-1781s4q_html_ffec24f2a58cbad.png)

_Figure 2 General map of EC-data construction from Row Data: UniProt/UniRef to positive and negative sets. To show how a single set, &#39;class-A&#39;, is formed. Here, class-A can be at any level (class-1, class-1.1.1.1, class1.2.1). To construct a positive set for class-A, available proteins under class-A are used. For the negative sets, proteins which are not found under class-A, and are found in non-enzymes class are used. Non-enzymes data includes proteins which have annotation scores 4 or 5 in UniProt/SwissProt, that is, strong evidence that these proteins are not enzymes is available._

This construction idea was taken from ECPred (Dalkiran, 2018), and the steps of the procedure are shown in Figure-… . At first, for each commission number, positive and negative sets are generated to apply binary classification. To generate the positive set, all proteins were separated according to their number in four main levels. The first level includes seven main enzyme classes, and as the level number increases enzyme functions become more specific, hence, in Level 4, there are more function classes but each has less enzyme (or protein). Due to this structural type, data size of the classes is increasing from the Level-4 to the Level-1. For example, while the average protein number for the Level-4 classes is between 100 and 1000, it may reach 30000 points for the Level-1 classes. Because of that, to design the negative train sets for Level-2, Level-3 and Level-4, a different method was used from Level-1 negative set construction.

The positive training set for each class is generated with %90 of the proteins annotated to its class while the negative training set is constructed with siblings of this class, other main classes and non-enzymes class. The %10 percent of these proteins are used to generate the positive test set. Here, non-enzymes class involves proteins that have annotation scores equal to 4 or 5 in UniProt/SwissProt dataset, that is, they have strong evidence to be not enzymes. On the other hand, siblings of an enzyme class are considered as other classes found at the same level with this class and found in the same parent class. To illustrate, siblings of an enzyme class, 2.1.1 are an enzyme class 2.1.2 and an enzyme class 2.1.3 which are found at Level-3 and their parent enzyme class is 2.1 which is found at Level-2. In addition, to arrange the size of the negative training sets, the size of the positive training sets were considered.

To create a negative training set for one of the Level-1 classes, proteins found in other classes and proteins found in non-enzymes class were used, and the size of the negative training set for the main classes was tried to be adjusted to the size of the positive training set. While %60 of proteins selected from other classes of Level-1, equally, %40 of proteins come from non-enzyme class. The statistics about EC-data structure is given in Table.\_.

_Table \_ Numbers of Proteins and Subclasses found in each Main Enzyme Class_

|
Enzymatic Functions | Statistics (# of EC terms, # of proteins, # of annotations) |
| --- | --- |
|
 | EC Level-1 | EC Level-2 | EC Level-3 | EC Level-4 |
|
Oxidoreductases | # of EC terms:# of proteins:# of annotations: | 13381935987 | 213246333125 | 552929429766 | 1022029420294 |
|
Transferases | # of EC terms:# of proteins:# of annotations: | 19611298277 | 99597098134 | 288538587531 | 2757818278182 |
|
Hydrolases | # of EC terms:# of proteins:# of annotations: | 16082164089 | 96046463716 | 365461757192 | 1593904039040 |
|
Lyases | # of EC terms:# of proteins:# of annotations: | 12604526125 | 62591425994 | 112304523125 | 762206122061 |
|
Isomerases | # of EC terms:# of proteins:# of annotations: | 11464214677 | 61461614651 | 141282212857 | 411262612626 |
|
Ligases | # of EC terms:# of proteins:# of annotations: | 12885728924 | 52868628753 | 62591925986 | 592566825668 |
|
Translocases | # of EC terms:# of proteins:# of annotations: | 11366213678 | 61287012877 | 41010710114 | 2569216921 |
| OverallEC dataset | # of EC terms:# of proteins:# of annotations: | 7
 273958281757 | 62270983277250 | 154241189246571 | 737204792204792 |

Negative training set construction for other levels classes differs from Level-1 classes. To generate the negative sets, proteins from their siblings, from other main classes they are not placed in and from non-enzyme class were collected. In addition, because numbers of proteins for a single class in Level-2, Level-3 and Level-4 can drop to 100, the sizes of negative training sets are arranged accordingly. If the positive training set size of a class is less than 1000, then the size of the negative training set for the class becomes three times of its positive training set. One third of these proteins come from siblings of this class, equally, one third of proteins come from other main classes, equally and the rest comes from proteins in non-enzyme class. If the number of proteins in the positive training set for a class is between 1000 and 10000, the negative training set size is two times of the positive set size. Half of the proteins come from the siblings of the class, equally, half of the rest comes from other main classes, equally, and the rest consists of proteins in non-enzyme class. FInally, the positive training set size for a class is higher than 10000 is equal to the negative training set size. Half of the proteins are collected from the siblings of the class, equally while a quarter of the proteins comes from other main classes, equally. The rest of the data is formed with the proteins from non-enzyme class. In the situation that proteins in siblings can be insufficient to meet needed number of proteins for negative training sets, non-enzyme class is used to provide data. The size of the negative test set was created to meet %10 of the total size of the negative training set and its data comes from only non-enzyme class.

**Gene Ontology Term Prediction Dataset (GO-data):** Gene ontology (GO) is a study that classifies the functions of gene products, (The Gene Ontology Consortium, 2018) and GO provides computational descriptions of biological systems (Thomas, 2017) in three aspects which are biological process, molecular function and cellular component (Ashburner, 2000). All functions (or GO terms) are in parent-child relation which can be represented in directed graphs (Gaudet, 2017) as in Figure-\_.

![](RackMultipart20210919-4-1781s4q_html_ba04c3dad8ee8a8c.png)  _Figure 4 Positive and negative set construction for GO:2 on an example of GO map. To form the positive set of GO:2, green GO terms are used while GO terms colored with red are used to construct the negative set. GO:1 term is not used in set constructions. Also, GO terms shown at the same levels are siblings to each other. For example, GO:2 and GO:3 are siblings, and GO:4 and GO:5 are siblings at their level. Dashed lines at the same level of GO terms indicate that these GO terms are siblings to each other._

In addition to EC-data, ProFAB also aims to supply already classified GO terms data to use in protein function studies. In GO-data, for each GO term, there are positive and corresponding negative data which are separated into training and test sets. Positive data of a GO term is formed from annotated proteins to this term while negative data consists of proteins found in the siblings of this GO term. If the size of the positive data is less than 1000, negative data is adjusted as its size doubles the positive data. In Figure.\_, GO:2 can be examined as an example. The proteins found in green GO terms are used in positive training and test sets of GO:2 while proteins in GO terms colored with red form negative sets of GO:2. The sizes of test sets are arranged according to splitting methods. If random splitting or similarity based splitting are used, the sizes of the negative and positive test sets are %10 of total data classified as negative and positive, respectively. For the temporal splitting method, there is no limitation for dataset size since it depends on the number of proteins found in that year. The overall statistics for GO-data is given in Table-\_.

_Table \_ Numbers of Proteins and Subclasses found in three aspects of GO_

|
 GO - Terms | Statistics |
| --- | --- |
| # of Proteins | # of GO Terms | # of Annotation |
| GO:0008150 Biological Process | 120,698 | 4,764 | 661,225 |
| GO:0003674-Molecule Function | 120,158 | 1193 | 501,741 |
| GO:0005575-Cellular Component | 116,720 | 722 | 488,866 |

**Splitting**** :** To provide a fair data distribution and to make the data more reliable in different conditions, three different splitting methods are employed in ProFAB. These are random splitting, similarity based splitting and temporal splitting. The difference between the splitting section and scaling, training and evaluation is that it is integrated to datasets, therefore, splitting cannot be applied to data explicitly. It can be used by entering the desired parameter while calling a dataset. Implementing ways of ProFAB&#39;s splitting functions (in Figure.\_) and their use reasons are:

![](RackMultipart20210919-4-1781s4q_html_6973fd527ddcd040.png)

_Figure 4 Illustration of how splitting methods are applied. Raw data is considered as positive data, and then negative sets are generated after filtering processes. a: Random split, which is a regular splitting method distributed data randomly. b: Target split, which is a random split method but it has a filter before preparing train and test sets. This filter consists of centers of clusters found in UniRef50. After raw data passes through it, similarity between all proteins reduces to at most %50. c: Temporal Splitting is a time dependent splitting method. To generate train sets, proteins found in 2016-SwissProt were used while to create a test set proteins were assigned in 2018 and 2017-SwissProt but not in 2016-SwissProt. Validation sets include proteins found in 2020 and 2019-SwissProt dataset but 2018-SwissProt dataset._

_Random Split:_ It is a base method for splitting which shows the effect of the world&#39;s randomness to machines. By using Python scikit-learn package, this splitting method is done.

_Similarity Based Splitting:_ The reason to do that is to increase the generalization of the datasets. To achieve this, the way is that all data points are passed through a filter which is generated by clustering data points or obtained from already-clustered datasets. To create this filter, only centers of the clusters are used. ProFAB uses **target splitting** (protein splitting) as a similarity based splitting method. To achieve that, the UniProt/UniRef50 dataset which is an already-clustered protein dataset was used. The filter was generated by only obtaining the centers of the clusters which were already-clustered based on 50% similarity of proteins. After the proteins passed through this filter, to create a train set and a test set, Python scikit-learn package was used.

_Temporal Splitting_: The reason to supply this splitting method, to provide datasets are created in different timelines. To get this splitting method, UniProt/SwissProt 2016, 2018 and 2020 January results were used. All proteins are assigned until 2016 are sent to train, proteins found in 2018 but not in 2016 are sent to test and the rest of proteins are presented as validation sets.

**Scaling** :For different data types and study purposes, various scaling algorithms are used. These are:

_standard scaler:_ Rescaling features of the data by subtracting the mean and scaling to unit variance.

_normalizer:_ Rescaling in not feature wise but in sample wise.

_max absolute scaler:_ Scaling method works well in sparse matrix

_min max scaler:_ Scaling method works for data which includes values that are in the range [0, 1]. This compresses values to the very small range [0, 0.005]

_robust scaler:_ Scaling method works well when dataset contains lots of outliers

**Training**** :**The other main objective of ProFAB is providing machine learning algorithms for binary classification to train the data. After preprocessing steps (getting data and scaling) machine learning comes to the stage. The used algorithms are given in Table … with their purpose and descriptions. After training, the models are stored in byte to reuse.

| Machine Learning Algorithms |
| --- |
| Models | Description |
| Logistic Regression | A supervised classification algorithm. Its cost function is Sigmoid function. |
| Support Vector Machine (SVM) | A linear supervised learning algorithm, creates a boundary to learn both linear &amp; non-linear models (Cortes, 1995) |
| Decision Tree  | A supervised learning algorithm starts from a root note and according to the obtained result, it passes to the next node (Breiman, 1984). |
| Random Forest | Combination of lots of decision trees. It is a supervised learning method (Breiman, 2001) |
| k-Nearest Neighbor | A supervised learning type that uses similarities in data points to learn.  |
| Naïve Bayes | Bayes&#39; theorem based learning algorithm. It works without extra parameters and its start assumption is that all points are independent of each other. It can be used as a base point for classification problem.(Friedman, 1997) |
| Gradient Boosting | A learning technique that makes weak learners better. Using gradient descent to minimize the cost. By adding weak learner to other, it finally offers good solution (Friedman, 2001) |
| FFNN | Both unsupervised and supervised deep learning techniques that optimize the loss function via gradient descent. The used one is a simple feed-forward network with different losses. (Bebis, 1994) |

_Table 2 Machine learning algorithms used in the tool. Description of each and use area are defined._

**Evaluation:** To benchmark the models and datasets, the tool employs some specific evaluation metrics for regression type and classification type learnings. This structure was designed to benchmark any model besides biomedical tasks. In Table…, assigned metrics and their corresponding purpose are given:

_Table 3 Evaluation Metrics employed in The Tool_

| Model Evaluation Metrics |
| --- |
| Metrics | Description |
| Recall | Ratio of true positives to default positives (TP/(TP+FN)) |
| Precision | Ratio of true positives to all positives (TP/(TP+FP)).  |
| F1 Score | Harmonic mean of Recall and Precision scores.   |
| F 0.5 Score |  Adjusted F1 score with β = 0.5 |
| Accuracy |  Ratio of true classified samples to all samples ((TP + FP )/(FN + TN + TP + FP)) |
| Matthews Correlation Coefficient | Accept the true class and predict one as binary and find correlation between them. It is also symmetric ([-1,1]).  |

**Results**

**Performance Results:**

**Use Case:** The platform was designed to apply different functions that are data importing, training and evaluation can be used for the same purpose or for different purposes. Datasets can be obtained by defining parameters and name of data:

To train data, following lines of code can be used:

Scoring the performance in all six different metrics can be done by:

**Conclusion**

ProFAB is a developable and flexible application that was planned to contribute to the area of protein based biomedical research thanks to informatic methods. Numerical datasets and algorithmic solutions for the area were designed to boost protein studies and knowledge gains. The subjects covered are functions and their classification which are crucial topics for drug and vaccine improvements. To use the data for prediction, eight editable machine learning algorithms were implemented and to benchmark the results, cost calculations for alternative cases were presented. Based on this structure, both machine learning scientists and wet-lab experts can get benefits in computational protein studies.

**Future Implementation**

ProFAB is an extendable project that is open to further implementation of proteins. Drug-target interaction dataset is one of the upcoming studies. Other than features of proteins in classic sequence view, addition of them in structural view can be found in further implementations of ProFAB. Also, molecule clustering according to their string representations will be added as good results are obtained.

**Availability**

**Project Name:** ProFAB

**Project home page:** \_\_\_github\_\_\_

**Operating System:** Platform independent

**Programming language:** Python ≥ 3.7

**Other Requirements:** numpy, rdkit, scikit-learn, scipy

**Data Availability**

All input data are publicly available. The input data for ecData was taken from UniProt/SwissProt 2020-05 results (). This data was used for both training and test sessions. For goData, inputs were obtained from UniProt/SwissProt 2020-02 results ([http://ftp.ebi.ac.uk/pub/databases/GO/goa/old/UNIPROT/goa\_uniprot\_gcrp.gpa.196.gz](http://ftp.ebi.ac.uk/pub/databases/GO/goa/old/UNIPROT/goa_uniprot_gcrp.gpa.196.gz) ). To apply similarity based splitting, already clustered UniProt protein data, Uniref50 (2020-05 results) was used ().

**References**

- Cornish-Bowden, A. (2014). Current IUBMB recommendations on enzyme nomenclature and kinetics. Perspectives in Science, 1(1-6), 74-87. doi:10.1016/j.pisc.2014.02.006
- Breiman, L. Random Forests. Machine Learning 45, 5–32 (2001). https://doi.org/10.1023/A:1010933404324
- Cortes, C. and Vapnik, V. (1995) Support-Vector Networks. Machine Learning, 20, 273-297.
- http://dx.doi.org/10.1007/BF00994018
- Dalkiran, A., Rifaioglu, A. S., Martin, M. J., Cetin-Atalay, R., Atalay, V., &amp; Doğan, T. (2018). ECPred: a tool for the prediction of the enzymatic functions of protein sequences based on the EC nomenclature. BMC bioinformatics, 19(1), 334.
- Tanimoto T. T., (1958). &quot;An elementary mathematical theory of classification and prediction,&quot; IBM Internal Report.
- Breiman, L., Friedman, J., Olshen, R. and Stone, C. (1984) Classification and Regression Trees. Chapman and Hall, Wadsworth, New York.
- N. Friedman, D. Geiger, and Goldszmidt M. (1997) Bayesian network classifiers. Machine Learning, 29:131–163
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5). doi:10.1214/aos/1013203451
- Bebis G. and Georgiopoulos M., (1994). &quot;Feed-forward neural networks,&quot; in IEEE Potentials, vol. 13, no. 4, pp. 27-31, Oct.-Nov. doi: 10.1109/45.329294.
- Gaudet P, Škunca N, Hu JC, Dessimoz C. (2017). Primer on the Gene Ontology. Methods Mol Biol.; 1446:25-37. doi: 10.1007/978-1-4939-3743-1\_3. PMID: 27812933; PMCID: PMC6377150.
- Ashburner M, Ball CA, Blake JA et al (2000) Gene ontology: tool for the unification of biology.  The  Gene  Ontology  Consortium.  Nat  Genet 25:25–29
- Moreau, Y. &amp; Tranchevent, L.-C. (2012), Computational tools for prioritizing candidate genes: boosting disease gene discovery, Nature Reviews Genetics 13(8), 523–536.
- Sureyya Rifaioglu, A., Doğan, T., Jesus Martin, M. et al. DEEPred: Automated Protein Function Prediction with Multi-task Feed-forward Deep Neural Networks. Sci Rep 9, 7344 (2019). https://doi.org/10.1038/s41598-019-43708-3
- Young, T., Hazarika, D., Poria, S., &amp;amp; Cambria, E. (2018). Recent trends in deep learning based natural language processing [review article]. IEEE Computational Intelligence Magazine, 13(3), 55-75. doi:10.1109/mci.2018.2840738
- Pang G., Shen C., Cao L., Hengel A., (2020). Deep Learning for Anomaly Detection: A Review. doi:10.1145/3439950
- Sidey-Gibbons, J., Sidey-Gibbons, C. Machine learning in medicine: a practical introduction. BMC Med Res Methodol 19, 64 (2019). https://doi.org/10.1186/s12874-019-0681-4
- Vamathevan, J., Clark, D., Czodrowski, P. et al. Applications of machine learning in drug discovery and development. Nat Rev Drug Discov 18, 463–477 (2019). https://doi.org/10.1038/s41573-019-0024-5
- Senior, A.W., Evans, R., Jumper, J. et al. Improved protein structure prediction using potentials from deep learning. Nature 577, 706–710 (2020). https://doi.org/10.1038/s41586-019-1923-7
- Haghighatlari, M., Vishwakarma, G., Altarawy, D., Subramanian, R., Kota, B. U., Sonpal, A., … Hachmann, J. (2020). ChemML            : A machine learning and informatics program package for the analysis, mining, and modeling of chemical and materials data. WIREs Computational Molecular Science, 10(4). https://doi.org/10.1002/wcms.1458
- A S Rifaioglu, R Cetin Atalay, D Cansen Kahraman, T Doğan, M Martin, V Atalay, MDeePred: novel multi-channel protein featurization for deep learning-based binding affinity prediction in drug discovery, Bioinformatics, Volume 37, Issue 5, 1 March 2021, Pages 693–704, https://doi.org/10.1093/bioinformatics/btaa858
- Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., Pande, V. (2018). MoleculeNet: a benchmark for molecular machine learning. Chemical Science, 9(2), 513–530. https://doi.org/10.1039/c7sc02664a
- The Gene Ontology Consortium, The Gene Ontology Resource: 20 years and still GOing strong, _Nucleic Acids Research_, Volume 47, Issue D1, 08 January 2019, Pages D330–D338, [https://doi.org/10.1093/nar/gky1055](https://doi.org/10.1093/nar/gky1055)
- Ong, S. A. K., Lin, H., Chen, Y., Li, Z., &amp; Cao, Z. (2007). Efficacy of different protein descriptors in predicting protein functional families. _BMC Bioinformatics_, _8_(1), 300. https://doi.org/10.1186/1471-2105-8-300
- Hu W., Fey M., Zitnik M., Dong Y., Ren H., Liu B., Catasta M., Leskovec J., (2021) Open Graph Benchmark: Datasets for Machine Learning on Graphs, arXiv:2005.00687
- Huang K., Fu T., Gao W., Ahao Y., Roohani Y., Leskovec J., Coley C. W., Xiao C., Sun J., Zitnik M., (2021), Therapeutics Data Commons: Machine Learning Datasets and Tasks for Therapeutics
- Jumper, J., Evans, R., Pritzel, A. _et al._ Highly accurate protein structure prediction with AlphaFold. _Nature_ (2021). https://doi.org/10.1038/s41586-021-03819-2
- Zhen Chen, Pei Zhao, Chen Li, Fuyi Li, Dongxu Xiang, Yong-Zi Chen, Tatsuya Akutsu, Roger J Daly, Geoffrey I Webb, Quanzhi Zhao\*, Lukasz Kurgan\*, Jiangning Song\*, iLearnPlus: a comprehensive and automated machine-learning platform for nucleic acid and protein sequence analysis, prediction and visualization. Nucleic Acids Research , 2021;, gkab122, https://doi.org/10.1093/nar/gkab122
- Zhou, N., Jiang, Y., Bergquist, T.R. _et al._ The CAFA challenge reports improved protein function prediction and new functional annotations for hundreds of genes through experimental screens. _Genome Biol_ 20, 244 (2019). https://doi.org/10.1186/s13059-019-1835-8
- Thomas, P. D. (2017). The Gene Ontology and the Meaning of Biological Function. In _Gene ontology handbook_ (pp. 15–24). essay, HUMANA Press.
-

[https://academic.oup.com/gigascience/article/10/5/giab033/6272610?searchresult=1](https://academic.oup.com/gigascience/article/10/5/giab033/6272610?searchresult=1)

[https://academic.oup.com/gigascience/article/9/8/giaa089/5897806?searchresult=1#206930435](https://academic.oup.com/gigascience/article/9/8/giaa089/5897806?searchresult=1#206930435)