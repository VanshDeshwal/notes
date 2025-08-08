# NetML: A Challenge for Network Traffic Analytics

## ABSTRACT  
Classifying network traffic is the base for key network services. Past work in this field had problems finding good data, and many results cannot be easily repeated. This issue grows worse with new data-driven machine learning methods. To fix this, we give three open datasets with almost 1.3 million labeled flows, including flow details and hidden raw packets, for all researchers. We look at wide parts of network traffic study, such as finding malware and telling applications apart. We offer the datasets as an open challenge called NetML1 and build simple machine learning models like random forest, SVM, and MLP. As NetML1 grows, we hope the data becomes a shared base for AI-based, repeatable work on network flow study. These resources include full details to help both academic and industry work on real flow analysis.

## 1. INTRODUCTION  
New technology now gives researchers a lot of data to study. Many papers in Computer Vision (CV), Natural Language Processing (NLP), and, more recently, Network Traffic Analysis (NTA) use Artificial Intelligence (AI) methods [9, 22, 24, 26]. In CV and NLP, many shared test sets and open challenges exist, such as ImageNet [6] and NLP benchmarks [1]. However, NTA does not have a common set of data or challenges. This lack makes it hard for researchers to compare methods and repeat results accurately.

In our digital era, sorting network traffic is very important because the number of internet devices keeps growing fast. Network Traffic Analysis means grouping data flows into the right category. It is vital for tasks like quality of service (QoS) control, fair resource sharing, and finding malware. From a security view, more connected devices mean more possible cyber attacks. We must defend against these rising threats 1[https://github.com/ACANETS/NetML-Competition2020](https://github.com/ACANETS/NetML-Competition2020). Yet, finding full data sets to let researchers test their methods is hard. Getting the right data is another major problem.

Over time, NTA methods have changed from simple port-based checks to using machine learning. The port-based method relied on fixed port numbers, but modern apps use many different or random ports. Then, researchers used deep packet inspection (DPI) to look inside packet contents. As more data is encrypted, DPI has started to fail. This change pushed researchers to use machine learning on flow-level statistics, which do not need packet data or port numbers.

Many studies have tried to sort and study network flows using various data sets. But, unlike open sets such as ImageNet [6] and COCO [14] in computer vision, it remains hard to find up-to-date data sets in networking. Without shared data, researchers cannot easily test new ideas and others cannot repeat past experiments. A broad, open, and current data set for flow analytics is needed for the community to move forward together and to have clear benchmarks.

In this paper, we introduce the NetML Network Traffic Analytics data sets as a common platform for NTA research. We provide three free to use labeled flow collections with detailed features to serve as benchmarks. We focus on two main uses: malware detection and application type classification. For malware detection, we offer a new data set of almost 500,000 flows from raw traffic captures on the Stratosphere IPS site [20], covering twenty different malware and benign classes. We also build a data set from the well-known CICIDS2017 [19] raw traffic, with around 550,000 flows for seven malware types and a benign class.

We publish these sets as the NetML open challenge, with anonymized packets and flow feature details available to all researchers. We also share baseline AI models, including random forest, support vector machine, and multilayer perceptron methods. As NetML grows, we hope it becomes the shared basis for AI-driven, repeatable NTA research. With clear data and benchmarks, the network community can better compare methods, build on past work, and improve flow analytics for both academia and industry.

Finally, we introduce the non-vpn2016 dataset, which contains around 163,000 flow samples in total. The raw traffic data were obtained from the publicly available ISCX-VPN-nonVPN2016 dataset [10]. This dataset includes different levels of annotations, with class counts ranging from 7 up to 31, covering multiple application types and traffic categories. On top of these data, we provide preliminary data analysis on key flow features—such as packet counts, byte counts, flow duration, inter-arrival times, and more—and present baseline classification results achieved using popular machine learning algorithms, including Random Forest (RF), Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP). All flow feature extraction code and algorithm implementations are hosted on GitHub, and we strongly encourage the research community to collaborate, share improvements, and grow the collection over time.

Our contributions in this paper are twofold. First, we provide a novel NetML malware detection dataset derived from raw traffic captures, and we curate two additional datasets originated from open-source traffic captures: the CICIDS2017 dataset, used for malware detection tasks, and the non-vpn2016 dataset, used for general traffic classification tasks. Second, we establish a common benchmarking platform via our first announced network traffic analytics challenge, where researchers can evaluate and compare their approaches against our baseline results generated by RF, SVM, and MLP models. For these baselines, we selected only metadata features—such as the number of packets, number of bytes, duration of flows, packet size statistics, and flow idle times—to expedite classifier efficiency. Nevertheless, we strongly encourage future researchers to investigate protocol-specific features, including TLS handshake parameters, DNS query characteristics, and HTTP header fields, to potentially boost classification performance.

The rest of the paper is organized as follows. Section 2 reviews existing literature on malware detection and traffic classification techniques. Section 3 details the dataset collection and preparation process, along with descriptions of the extracted flow features. Section 4 presents preliminary dataset analysis results based on those features. Section 5 reports the performance results obtained by the proposed baseline models. Section 6 provides information about the NetML Challenge and accompanying workshop. Finally, Section 7 concludes the study. In the appendix, readers will find detailed graphs illustrating flow feature analysis results.

## 2. RELATED WORK  
Interest in network traffic analysis has grown, and many papers have appeared. We focus on two areas: malware detection and traffic or application classification.

### 2.1 Malware Detection  
Random Forest [27] and Support Vector Machine (SVM) with a Radial Basis Function (RBF) kernel [2] are two popular machine learning methods for network traffic analysis. However, most researchers use their own data and features, making it hard to compare results. For example, Anderson et al. [3] use flow metadata, packet sizes and timings, byte distributions, and unencrypted TLS headers to find malware in encrypted traffic. Likewise, Moore et al. [15] extract 249 features in total, but different studies pick different subsets, so there is no single best feature set for NTA.

The KDD-Cup 99 [13] and NSL-KDD [21] datasets are widely used for malware detection research. Many works test ensemble models like C4.5, decision tree, random forest, SVM, and AdaBoost [7, 8, 12, 23]. One two-stage method uses Naive Bayes first, then nominal-binary filtering and cross-validation [16]. Another improves this with a balanced forest first and random forest second [17].

However, these two datasets are old and miss new attack types. The CICIDS2017 dataset [19] is more recent and better for modern threats. Its creators use CICFlowMeter [11] features to compare k-nearest neighbor, random forest, and ID3, reporting F1 scores of 0.98 for ID3 and 0.97 for random forest, which runs three times faster than ID3. Gao et al. [9] compare several machine learning and deep learning models—including random forest, SVM, and deep neural networks—on NSL-KDD and CICIDS2017. They do both binary detection and multi-class tasks, finding RF gives strong detection accuracy and similar multi-class performance to deep models.

### 2.2 Traffic Classification  
Many studies have looked at how to sort network flow by application. For example, Conti et al. [5] used a Random Forest model on their own data and features, achieving 95% accuracy for user action classification. Taylor et al. [22] used simple statistics on packet lengths—minimum, maximum, mean, median, and standard deviation—in binary SVM and Random Forest, also finding 95% accuracy. They measured flow vectors, which list packet sizes and times for each flow. However, when they used a multi-class model, accuracy dropped to 42% for SVM and 87% for Random Forest. Lashkari et al. [10] released the UNB ISCX VPN-nonVPN2016 dataset and tested it with k-nearest neighbors (kNN) and C4.5 decision trees.

Many others adopt the VPN-nonVPN2016 data and report results. Yamansavascilar et al. [25] used it for application type classification and compared it with their own dataset using the same classes. They found kNN was best on VPN-nonVPN2016 while Random Forest was best on their own data. They also applied feature selection and saw a 2% accuracy boost on their data. Wang et al. [24] looked at packet-level details and used deep learning for protocol classification. Later, Yao et al. [26] applied an attention-based LSTM to classify protocol types in the VPN-nonVPN2016 data. Qin et al. [18] proposed Hybrid Flow Clustering (HFC) to group flows by their features and link patterns without labels, achieving strong results on data from three open sources.

These earlier studies show that interest in network flow analysis is growing, but there is no simple solution for sorting applications or finding malware. Most approaches use flow metadata like packet counts, byte counts, flow duration, or idle times. However, there is no agreed-upon feature set. The community needs a shared set of useful features and a large, current, open dataset for fair testing and comparison. This gap slows progress.

## 3. NETML DATA COLLECTION  
We now explain how we gathered data from different public sources. First, we describe how we got raw traffic for the malware detection task, then for the application classification task. We used Stratosphere IPS [20] and the Canadian Institute for Cybersecurity (CIC) [10, 19] as our sources of raw packet captures (pcap files). CIC offers several datasets for Intrusion Detection Systems (IDS) and one for application classification. Stratosphere IPS only provides captures of normal and malware traffic. From these two sources, we built three datasets—NetML, CICIDS2017, and non-vpn2016—that we will use throughout the paper. We preprocessed the raw captures when needed and then extracted flow features with an Intel flow-feature tool.

In this work, we focus on classifying flows for different uses and at different detail levels. We cover both malware detection (important for cybersecurity) and general flow-type identification. We call our three detail levels “top-level,” “mid-level,” and “fine-grained.” At the top level, malware detection is simply a two-class problem: benign or malware. For flow identification, we label major app categories like chat, email, and video. The mid-level recognizes specific apps, such as Facebook, Skype, and Hangouts. Finally, the fine-grained level splits classes into individual malware types (e.g. Adload, portScan) and more detailed app flows (e.g. facebook_audio, skype_audio).

### 3.1 Dataset Preparation  
We used an Intel-provided accelerated feature extraction library that can process up to 200 packets per flow in both directions. Given each raw pcap file, the tool outputs one JSON record per flow, with the full list of extracted features shown in Table 1.

Figure 1 shows our processing steps. First, we run the Intel tool on each capture to get JSON flow records. We extract metadata features for every flow, and we also extract TLS, DNS, or HTTP fields if those protocols appear. Next, we do simple preprocessing to make the data fair for testing. We mask source (sa) and destination (da) IPs by replacing them with “IP_masked,” so participants cannot trace flows back to real addresses. We remove the original time_start and time_end fields and add a new feature, time_length, that holds the time difference. We also assign each flow a unique ID and add its true label from the pcap. After these steps, we obtain 484,056 flows for NetML, 551,372 for CICIDS2017, and 163,711 for non-vpn2016.

Finally, we split each dataset randomly into three parts: 10% of each class goes to the test-challenge set, another 10% to the test-std set, and the remaining 80% to the training set. The overall data preparation workflow is summarized in Figure 1.

### 3.2 Malware Detection Datasets  
Our first task is to focus on finding malware flows in network traffic through routers. Clear detailed steps explaining how we built our own dataset from both public sources follow in the next two sections.

#### 3.2.1 NetML. 
Stratosphere IPS provided raw traffic data files for the dataset. We made this dataset for finding malware in traffic by using 30 of over 300 raw traffic files from Stratosphere IPS. Table 2 shows the chosen capture files. Each file has flows for one class. For example, every flow from capture_win15.pcap is labeled top-level as malware and fine-grained as Artemis. Figure 2 shows how many flows belong to each fine-grained class in NetML. At the top level, flows are marked as benign or malware. At the fine-grained level, there are twenty malware types plus one benign class.

#### 3.2.2 CICIDS2017.
The CICIDS2017 [19] dataset has raw packet captures of various malware attacks and normal network flows. This dataset has five traces each day of one week, totaling thirty-five files. We filtered flows of interest using the website instructions. We downloaded CICIDS2017 from its source, and used our flow feature extraction tool to get the flow data. We could not extract features for Botnet and Heartbleed attacks, so these two classes were left out of our final CICIDS2017 dataset. Table 3 lists the malware types we kept and the files they came from. Figure 3 shows the flow count for each fine-grained class. Like the other malware datasets, CICIDS2017 has top-level tags for benign or malware and fine-grained tags for seven malware types plus a benign class. Only packets in certain time windows with chosen IP address pairs were kept per the CICIDS2017 guide. We ran feature tool on filtered pcaps to produce JSON flow records with metadata fields like packet count, byte count, and flow duration.

### 3.3 Traffic Classification Dataset: non-vpn2016  
The third dataset is for sorting network flows by application and is named non-vpn2016. We built it by running our flow feature tool on only the non-vpn raw traffic files from the CIC website, since our tool cannot yet process VLAN data.

We assign three levels of labels to this dataset: top-level, mid-level, and fine-grained. Top-level labels group flows into seven main types: P2P, audio, chat, email, file_transfer, tor, and video. Mid-level labels specify 18 common apps (for example, facebook and skype). Fine-grained labels split those into 31 detailed classes (facebook_audio, facebook_chat, skype_audio, skype_chat, etc.). Table 4 lists the capture files we used to create non-vpn2016.

After feature extraction, Figure 4 shows the number of flows for each top-level class, Figure 5 shows counts for mid-level apps, and Figure 6 shows counts for fine-grained classes.

Class imbalance—when some labels have far more flows than others—is common but can bias model results. For example, in Figure 6, facebook_audio, hangouts_audio, skype_audio, and skype_file each have many more samples than other classes.

## 4 NETML DATASET ANALYSIS
In this part we look at the flow details for the training parts of all three sets. We take four types of features: (1) Metadata, (2) TLS, (3) DNS, and (4) HTTP. Metadata features work for any flow. These include the number of packets, bytes in and bytes out, flow time length, and more. Protocol-specific features only apply if a flow has that protocol. TLS features include how many cipher suites and extensions the client or server uses. DNS features include the query name and the answer IP. HTTP features include the status code and the request method. While Metadata features come from any flow, protocol-specific features only come if the flow has TLS, DNS, or HTTP packets. So Table 5 shows how many flows in each set have Metadata, TLS, DNS, and HTTP data. The training set sizes are 387,268 for NetML, 441,116 for CICIDS2017, and 131,065 for non-vpn2016. This section shows feature coverage in each dataset and highlights where protocols differ. We use this analysis to plan model use of these features.

---

### 4.1 Metadata Features

_Metadata Histograms_  
Metadata features are mostly counts or histograms made for each flow. Histogram features return fixed-size arrays like intervals_ccnt[] and hdr_ccnt[]. We compute the average value for each array index and show these means in Figures 18 and 19, and in Figures 20 and 21. In the charts, the horizontal axis is the array index and the vertical axis is the mean value at that index across all flow samples in each dataset for training sets.

_Single-Value Features_  
Other Metadata features like bytes_in or src_prt give a single number, for example 160 or 6006. We show how often each feature value appears in Figures 10–17. In these charts, the horizontal axis is the feature’s value and the vertical axis is how many flows have that value. Each plot uses 100 equally sized bins to display the distribution clearly. It helps choose features better.

_Distribution Insights_  
Looking at these feature distributions tells us which values separate classes best. For example, the hdr_bin_40 feature is smaller in malware than in benign flows. In the non-vpn2016 data, this same feature is often smaller for P2P, chat, email, and tor flows. There is one outlier in the time_length feature of NetML, with a value near seven million, which makes its histogram odd. We do not clean these outliers in this work and leave that for future research. We plot all histograms with 100 equal bins, as seen in Figures 10 through 17, to keep value ranges clear and comparable. These differences show models learn.

_Coverage Summary_  
We can get Metadata features from any flow sample. In this work, we extract 31 Metadata features with our flow feature tool. In the NetML training set, all 387,268 flow samples yield every Metadata feature. Likewise, the CICIDS2017 training set has 441,116 flows with these features, and the non-vpn2016 training set has 131,065 flows with every metadata feature listed in Table 1. These numbers show metadata coverage.

### 4.2 TLS Features  
We use several client-side and server-side TLS features. Some include the list of cipher suites offered, the list of extensions advertised, and the key exchange length offered by client or server. We extract 14 total TLS features. In the NetML training set, 114,396 flows have TLS data, about 30% of all flows. In CICIDS2017, 74,836 flows (about 15%) include TLS. In non-vpn2016, only 1,262 flows (under 1%) have TLS.

Figures 22 and 23 show how often each single-value client-advertised TLS feature appears in NetML, CICIDS2017, and non-vpn2016 training sets. Figures 24 and 25 show the same for server-advertised features. In these plots, the horizontal axis is the feature value from the extraction tool and the vertical axis is how many flows have that value.

Figures 26 and 27 first row show the sizes of returned TLS arrays (array length vs. count). The next four rows show which cipher suites and TLS extensions appear in each set. In those charts, the horizontal axis is an index for each suite or extension type, and the vertical axis is how many times it appears. A full mapping from index to hex code is in the appendix.

In NetML, we see 122 unique cipher-suite hex codes and 38 unique TLS extension codes. In CICIDS2017, there are 11 unique codes for both cipher suites and extensions. In non-vpn2016, we find 16 cipher-suite codes and 12 extension codes. Common hex values are in Figures 34 and 35 for all sets. Note that the P2P class in non-vpn2016 has no TLS flows. We do not compare codes across datasets here. Finally, one integer value gives counts for other TLS features and key exchange lengths.

### 4.3 DNS Features  
Our tool gives several DNS query and answer features (see Table 1). Like TLS, some DNS features are single values and some are arrays. For example, **dns_query_cnt** and **dns_answer_cnt** each return one integer, while **dns_answer_ttl** returns an array of numbers, and names or classes return arrays of strings.

- **NetML:** 60,271 flows have DNS data (about 15% of training).
    
- **CICIDS2017:** 93,224 flows have DNS data (about 21%).
    
- **non-vpn2016:** 8,179 flows have DNS data (under 7%).
    

Figures 28 and 29 plot the single-value DNS features by class. You can see **dns_query_cnt** is always 1, so it offers no discrimination and can be dropped. Likewise, **dns_query_class** is always 1 and can be ignored.

Figures 30 and 31 show the array-like DNS features. For **dns_query_name** and **dns_answer_ip**, the horizontal axis is the array index (first entry, second entry, etc.) and the vertical axis is how often that entry appears. Other array features plot “array length” on the horizontal and “count of flows with that length” on the vertical.

We also looked at which DNS names and IPs appear most often. Figures 36 and 37 list the top five values of **dns_query_name** and **dns_answer_ip** in each training set. Surprisingly, almost every class in all three datasets has a blank (empty) **dns_answer_ip** for most flows, except the P2P class in non-vpn2016, which has more actual IP answers.

### 4.4 HTTP Features  
We extract six HTTP features for flows with HTTP packets (Table 1). Three are single integers—http_method, http_code, http_content_len. The other three—http_uri, http_host, http_content_type—are strings mapped to integer indexes.

Figures 38–39 show the top http_content_type, http_host, and http_uri values for each top-level class in all three datasets. The http_content_type feature clearly helps spot malware: in both malware-detection sets, “text/html” appears most in malware flows. If a flow has a different content type, the model can likely mark it benign. In the non-vpn2016 set, each app class has a distinct common content type. Note that non-vpn2016’s Tor class has no HTTP features, as expected for encrypted traffic.

## 5 NETML CHALLENGE BASELINES  
In this section, we show how to use our three datasets for flow analysis. We give simple machine learning methods as baselines. We train models on the training sets and test them on the test-std sets. All results below come from the test-std sets unless we say otherwise.

### 5.1 Baselines  
We pick three easy classification models: Random Forest, Support Vector Machine (SVM), and Multi-Layer Perceptron (MLP). Random Forest is quick to train, SVM is common and works well, and MLP points toward deep learning’s power. We build these models using scikit-learn in Python. Since TLS, DNS, and HTTP features are missing for many flows (see Section 4), we use only Metadata features for these baselines.

#### 5.1.1 Random Forest Classifier  
Random Forest makes many decision trees and combines their votes. Each tree votes on the class, and the majority wins. We use this model because it is simple, fast, and often accurate. We set the number of trees (estimators) to 100 and the maximum depth of each tree to 10.

#### 5.1.2 Support Vector Machine Classifier  
Support Vector Machines draw a boundary (hyperplane) that best separates two classes. The data points closest to the boundary are the support vectors. SVMs are known for good accuracy. A downside is that they can take a long time to train on large datasets like ours. To ease this, you can train on a smaller sample and keep the rest for validation. We use the default settings: regularization C = 1.0 and the radial basis function (RBF) kernel.

#### 5.1.3 Multi-Layer Perceptron Classifier  
A perceptron is a single node that combines inputs, applies weights, and passes the sum through an activation function (often sigmoid). An MLP is a network of these perceptrons in several layers: an input layer, one or more hidden layers, and an output layer. MLPs are feed-forward neural networks trained by backpropagation to learn data patterns. Deep neural networks are more complex MLPs with many hidden layers. Recent studies show that neural networks can beat classic methods like Random Forest and SVM on many tasks. Here, we use a simple MLP with one hidden layer of 121 units. We apply L2 regularization with alpha = 0.0001 and train using the Adam optimizer.

### 5.2 PreProcessing

As said in Section 4, not all flows have TLS, DNS, or HTTP data. So we use only Metadata features for baseline tests. All classifiers need input as a two-dimensional grid with columns as features and rows as flow examples. Thus, we turn our data into a matrix. First, we drop source and destination IP fields because they are masked. All other Metadata fields hold numbers, so we can load them directly into a matrix for training. For array fields like hdr_ccnt[], we split each element into its own feature: hdr_ccnt_0, hdr_ccnt_1, and so on up to hdr_ccnt_k for a total of k+1 features. These go into separate matrix columns. Figure 7 shows how we capture traffic and make our feature matrices. Before training, we standardize each column so each feature has a normal distribution. For Random Forest and MLP, we set aside 80% of the training data to train the model and keep 20% for validation. For SVM, we only train on 10% and use 90% to validate because SVM is slow when training on large sets. After making all columns, we check for missing values and find none since Metadata exists. We then apply a standard scaler to make each feature have zero mean and unit variance. Next, we split our matrix into training and validation sets. For Random Forest and MLP, 80% trains and 20% validates. For SVM, we use 10% for training and 90% for validation to speed up. We set a random seed for reproducible splits and save them for future use.