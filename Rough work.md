Recent state-of-the-art Deep-Learning architectures for encrypted traffic classifications demonstrated superb results in tasks of traffic categorization over encrypted traffic. In this paper, we leverage the feasibility to use such architectures for the tasks of malware detection and classification
we present a Deep-Learning model for malware traffic detection and classification (MalDIST), which outperforms both classical ML and DL malware traffic classification models both in terms of detection and classification.
Network trafic is encrypted to improve privacy, but that makes it harder to classify malicious traffic.
attackers are taking advantage of encryption to deliver malware inside the network.
Authors are transfering models from internet classification domain to malware traffic classification because the former has shown great results.

Converting DISTILLER to MalDIST and comparing it to current state of the art malware traffic classification models


MalDIST model was compared with
1) classical ml models: For the task of malware detection and classification, we used a subset of the classic machine learning models: Random Forest (RF) provided by the scikit-learn implementation (https://scikit-learn.org/ stable/), Support-Vector Machine (SVM) provided by the scikit-learn implementation after standardizing the data using StandardScaler and K-Nearest Neighbor (KNN) where westandarized the data and ran KNN with k=3, as it gave the best results among different choices of k (3 to 10). 
2) DeepMAL : The paper describes two variants of the model, a packet-based one and a session-based one. For the comparison we use the variant that originally performed better which is the session-based variant. The main reason is to be able to compare it to the other session-based models discussed in this section. The session-based variant architecture leverages a convolutional layer along with fully connected layers in order to predict and classify malware.
3) M2CNN : The authors approached to move away from the traditional careful handy-crafted features of classical ML, towards representation learning of raw traffic data with DL in the task of malware traffic detection. They proposed extracting the first 784 payload bytes of a session as raw data and then reshaping them to [28X28] image to feed it into a customized Deep-Learning architecture of the known CNN LeNet-5 architecture
4) M1CNN : The authors used DL based on raw traffic data for the task of normal traffic detection, the general idea is the need of feature extraction is obsolete, and the classifier can work as is by viewing the network stream, they test the goal on multiple experiments: 1. Encapsulation identification 2. Normal encrypted traffic classification 3. Protocol encapsulated traffic classification 4. Encrypted traffic classification (with encapsulation).
## Cleaned version

Today, network traffic is encrypted to improve privacy, but that makes it harder to classify malicious traffic. Attackers are taking advantage of encryption to deliver malware inside the network. Recent state-of-the-art deep learning architectures for encrypted traffic classification have demonstrated superb results in tasks of traffic categorization over encrypted traffic. One such architecture is DISTILLER, and the authors are transferring this model to malware detection and classification.
**MalDIST Architecture** — The work adapts the encrypted traffic classifier **DISTILLER** for malware detection (binary) and malware family classification. DISTILLER originally used two input modalities:

1. The first 784 payload bytes of a session.
    
2. Protocol fields from the first 32 packets (direction, size, inter-arrival time, TCP window size).
    

MalDIST introduces a third modality — statistical features from the first 32 packets — aimed at real-time detection. Packets are grouped into: bidirectional, source→destination, destination→source, handshake, and data transfer. For each group, five statistics (min, max, mean, std, skewness) for packet size and inter-arrival time (10 features) plus four totals/rates (bytes, packets, bytes/s, packets/s) are computed, forming a 5×14 feature matrix.

The architecture has three subnetworks:

- **Payload modality** — two 1D CNN layers (16, 32 filters, kernel 25) with ReLU and max-pooling, followed by dense (128 nodes, ReLU).
    
- **Protocol fields modality** — bidirectional GRU (64 units, ReLU) and dense (128 nodes).
    
- **Statistical modality** — bidirectional LSTM (65 units) + 2D CNN layers (32, 32, 64, 128 filters) with leaky-ReLU, 2D max-pooling, and dense layers (512, 128 nodes).
    

Outputs are merged into a 384-dimensional vector, passed to a shared dense layer (ReLU) and then two identical branches for detection and classification, each with a dense (128, ReLU) and a SoftMax output. Dropout layers follow dense layers to reduce overfitting.
The authors used a combination of three datasets for their experiments. Benign traffic was taken from **StratosphereIPS** (same benign PCAPs as in the NetML dataset) and **ISCX2016** (105 selected benign PCAPs out of 150 labeled files covering various applications like Facebook, YouTube, Spotify, and traffic types such as streaming, VoIP, and chat, with VPN/non-VPN labels). Malware traffic was obtained from **Malware-Traffic-Analysis.net (MTA)**, selecting four families: Dridex, Hancitor, Emotet, and Valak.
For preprocessing, sessions with less than 784 payload bytes were removed, as they were not informative enough. The ISCX2016 dataset was further cleaned by removing irrelevant protocols (e.g., SNMP, LLMNR) and noisy traffic such as UDP broadcasts (e.g., Dropbox LAN Discovery). The benign set was built as an equal mix of StratosphereIPS and ISCX2016 (after filtering) to increase diversity. To address the imbalance between benign and malicious samples, the benign set was sub-sampled so that the final dataset contained 50% benign and 50% malware samples, totaling 18k sessions. Of these, 57.9% were TLS-encrypted. While the benign/malicious classes were balanced, the malware family distribution remained imbalanced.

MalDIST was compared against:

1. **Classical ML models** — Random Forest (RF), Support Vector Machine (SVM) with standardized data, and K-Nearest Neighbor (KNN) with _k=3_, which gave the best results among tested values (_k=3–10_).
    
2. **DeepMAL** — Session-based variant (better-performing) using a convolutional layer and fully connected layers for malware detection and classification.
    
3. **M2CNN** — CNN (based on LeNet-5) using the first 784 payload bytes of each session reshaped into a 28×28 image for malware detection.
    
4. **M1CNN** — DL model on raw traffic data without feature extraction, tested for encapsulation identification, encrypted traffic classification, and protocol-specific classification.

Results
For binary benign/malware detection, MalDIST outperformed all compared ML (SVM, RF, KNN) and DL models (DeepMAL, M1CNN, M2CNN, DISTILLER), achieving **99.7%** in Accuracy, Precision, Recall, and F1. In multi-class classification (Benign, Emotet, Hancitor, Valak, Dridex), MalDIST achieved the highest scores across all metrics, with RF coming close but never surpassing it.


## Choosing data

Combination of 3 datasets used:
- StratosphereIPS: composed of three parts: benign, malware, and mixed traffic.
- ISCX2016:is a dataset that consists of 150 PCAP files of different types of traffic and applications. Each PCAP file has a category of application (e.g., Facebook, YouTube, Spotify, etc.) and traffic type category (e.g., streaming, VoIP, chat, etc.) along with encapsulation label (VPN/non-VPN).
- Malware-Traffic-Analysis.net (MTA): is a website which shares many types of malware infection traffic for analysis.

For the benign traffic, we used two datasets: StratosphereIPS and ISCX2016.. The PCAPs that we picked from the StratosphereIPS are the same as in NetML [39] dataset. Moreover, we added 105 PCAPs from the ISCX2016 dataset.
For the malware traffic, we have downloaded PCAPs from the MTA,we picked four malware families (Dridex, Hancitor, Emotet and Valak)

## preprocessing

first, we filtered sessions with less than 784 payload bytes due to the fact that sessions with less than 784 bytes are not informative enough.
cleaned the ISCX2016 dataset by removing sessions with irrelevant protocols such as SNMP, LLMNR and some sessions that are considered as noise, such as UDP broadcasts (e.g., Dropbox LAN Discovery).
We chose to build our benign data as 50% StratosphereIPS and 50% ISCX2016 (after filtering and cleaning phases) to leverage the traffic diversity in both datasets.
. Finally, due to the imbalance between the number of benign and malware samples in the dataset, we sub-sampled the benign data such that the number of samples of benign and malware would be 50% each
As a result, the number of samples in the dataset is 18k.
in addition 57.9% of the sessions in our dataset are TLS hence encrypted sessions.
so for malicious/benign this dataset is balanced, for malware family its imbalanced.

## Results

Model Comparison 1) Classical ML approaches: For the task of malware detection and classification, we used a subset of the classic machine learning models: Random Forest (RF) provided by the scikit-learn implementation (https://scikit-learn.org/ stable/), Support-Vector Machine (SVM) provided by the scikit-learn implementation after standardizing the data using StandardScaler and K-Nearest Neighbor (KNN) where we standarized the data and ran KNN with k=3



MALDIST - A NEW DISTILLER ARCHITECTURE FOR MALWARE DETECTION AND CLASSIFICATION Due to the high success of the encrypted traffic classification solutions, our main research question is the following: Can we transfer and modify an encrypted traffic classification system to become a malware traffic classification system? Therefore, in this paper, we verify our hypothesis by showing use case of a encrypted traffic classification system [17] and how to transfer it into the domain of malware traffic detection and classification while outperforming other malware traffic detection and classification. Originally, DISTILLER is used as an encrypted traffic classifier. In their paper, the authors show an instance that solves the following three tasks: (1) Encapsulation detection; (2) Traffic type classification; (3) Application classification. For those three tasks, two modalities are used as inputs to DISTILLER, the first is the 784 payload bytes of the session, and the second is the packet direction, size, inter-arrival time, and TCP window size of the first 32 packets. As a first step and due to the fact that the implementation of DISTILLER is not publicly available, we have implemented the model (with the help of the DISTILLER authors). In order to verify that the model is well-implemented, we trained and tested it on the same dataset used in the original paper (i.e., ISCX2016 dataset) with the same tasks. As can be seen from Figure 1 the results of our DISTILLER implementation are similar to the results of [17]. From the results, it can be seen that DISTILLER has similar results in the tasks of classifying traffic types, encapsulation, and application which are all part of traffic classification. Does it mean that we can use it for malware traffic detection and classification? While DISTILLER was initially designed to tackle the encrypted traffic classification problem, we transfer and modify the architecture in order to use it for different tasks, malware traffic detection (binary classification) and malware traffic classification (families classification). Therefore, in the case of malware traffic detection and classification we develop a new architecture named as MalDIST. While for the traffic classification, the authors [17] use only two modalities for the encrypted traffic classification task. To tackle detection and classification of malware traffic, we provide a third novel modality. The new modality focusing on the statistical features of the first 32 packets of the session (the usage of some statistical features can be seen in other works such as [34]). The motivation to take the first packets is based on the requirement for a real-time system and previous malware traffic classification systems such as [35]. We first arrange the 32 packets into five groups: (1) bidirectional packets, (2) source → destination packets, (3) destination→ source packets, (4) handshake packets (e.g., TCP handshake) (5) data transfer packets (non-handshake)For each group, we extract the min. max, mean, standard deviation, and skewness of the packet size and inter-arrival time (10 features). We also extract from each group the total number of bytes, total number of packets, bytes/s rate and packets/s (additional 4 features). We then arrange the features for each group in a row, to end with a matrix of shape 5x14 (five groups with 14 features each), giving us an image of the malware/benign traffic. The network architecture of MalDIST, as illustrated in Figure 2, is as follows: the two modalities are the same as in the instance of DISTILLER while the third is our (from the left) novel modality. The ”payload modality” subnetwork consists of two 1D convolutional layers (with 16 and 32 filters respectively, with kernel size of 25, 1 unit of stride and padding of ”same”), each followed by a ReLU and 1D max-pooling, and on top there is a dense layer of 128 nodes (with activation of ReLU). For the ”protocol fields modality” we have a bidirectional GRU of 64 units and activation of ReLU, followed by dense of 128 nodes. For our novel modal we start with a bidirectional LSTM layer of 65 units followed by a series of 2D convolutions, leaky-ReLU and 2D max-pooling (pool size of 2) with increasing size of convolutional filters (32, 32, 64 and 128 filters respectively). All convolutions are with stride of 1 and padding of ”same”. Next, we have dense of 512 nodes followed by another dense of 128 nodes. The outputs of the three modality layers are then merged into a single vector of size 384 right into the ”shared representation” subnetwork that consists of a dense layer and an activation of ReLU. The outputs are then transferred to two identical subnetworks of ”malware detection” and ”family classification” that consist of a 128-nodes dense layer (with ReLU) and then into the final layer of dense (with SoftMax) with the number of nodes matching the number of classes of each task. Dropout layers were used after the dense layer to reduce overfitting.


The existing rule based malware detection systems are highly accurate at doing the task assigned to them, but they are unable to adapt with evolution of malware, also some malwares can disable such systems. its also easier for attackers to develop new bypassing techniques. authors solution is based on cross-layers and crossprotocols traffic classification, using supervised learning methods.

An additional strength of our approach is its ability to take into account the fact that targeted machines can be behind NAT

make short:
Network behavioral modeling is a popular approach for malware detection and malware family classification [12]. However, most of the existing studies (e.g., [6], [13]) focus on specific types of malware, such as Bots, or on a specific type of attack such as DoS or anomalies detection in specific protocols or network layers. Our work combines features from different layers and protocols, extracted in various resolutions, and are able to detect a variety of known and new malware

short paragraph:
The uniqueness of our solution lies in the fact that we observe data stream analysis in four resolutions, based on Internet and Transport and Application layers, with features generated accordingly (as presented in Figure 2). Specifically, we model the following levels of traffic observations as follows:
 Transaction – Representative of an interaction between a client and a server. It is a two-way communication: the client sends a request to the server, and the server processes the request and sends a response back to the client. We handle the following types of transactions:  An HTTP transaction consists of sending one request and response message between a client and a server.  A DNS transaction is equivalent to one session with two packets, one for the request and another for the response with the same transaction ID.  An SSL transaction is the aggregation of all App-data packets sent from a client to a server and vice versa after a successful handshake step and until the session ends.  Session – A unique 4-tuple consisting of source and destination IP addresses and port numbers.  A TCP session begins with a successful handshake, and ends with either a timeout, or a packet with the RST or FIN flag from any of the devices.  A UDP session consists of all packets sent from a client to a server and from a server to a client until a defined communication idle time is reached.
 Flow – A group of sessions between two network addresses (IP pair) during the aggregation period. The aggregation period can be specified by an algorithm as the accurate period of time from the start of the first session in the flow, until the maximum idle time between two sessions. A new flow starts if the time between the end of a session (the last packet) and the start of a new session (first packet) is more than the defined idle time. The new session is then part of the new flow.  Conversation Windows – A group of flows between a client and a server over an observation period. A conversation can be defined between two network addresses (IP pair) or a group of network resources (e.g., between two autonomous systems).


For each of the cumulative features we calculate the following statistics as additional features: minimum, first quartile, median, third quartile, maximum, average, standard deviation, variance and entropy.

we decided to choose and test three different classification algorithms including Naïve Bayes, a basic and simple model, as well as decision tree (J48) and Random Forest [24]. For feature selection we use the CFS (Correlation Feature Selection) algorithm [25]. All machine learning algorithms that we used were implemented using the Weka library [26].

Since the data is imbalanced, we had to use the True Positive Rate (TPR) and False Positive Rate (FPR) and the Area under the Curve (AUC) metrics to evaluate the performance of our detection results.

we used network traffic captures that included malware as well as normal (benign) network traffic that was collected by the Verint [28] and Emerging Threats [29] security companies and by us at our lab

Some of the captures were recorded in sandbox environments, others in real networks. The following is information about our dataset that consisted of network captured tcpdump *.pcap files from different sources: 1) The Sandbox malicious captures included: a) 2,585 records obtained from the Verint [28] sandbox. b) 7,991 records obtained from an academic sandbox. c) 4,167 records obtained from the Virus Total [30]. d) 23,600 records obtained from the Emerging Threats [29]. e) 12,377 malicious records collected from the web and open source community1 . 2) Benign corporate traffic was captured for 10 days in a students' lab at Ben-Gurion University. 3) Corporate traffic gathered by Verint [28] from a real network including malicious and benign traffic.

We applied the CFS algorithm for feature selection that identified the 12 network features (out of 927) presented in Table V as most effective for this task. It is apparent that even this small set of features spans across layers, protocols and observation resolution. TABLE V. 12 NETWORK FEATURES USED FOR FAMILY CLASSIFICATION Level Protocol Feature Session TCP Number of packets with RST flag Flow TCP Number of packets sent by client with ACK flag Number of destination ports and sessions ratio HTTP Median of inter-arrival time DNS Query name Alexa 1M Rank Count of DNS response addresses records Count of DNS response answer records Count of DNS response authoritative records Conv. Win. TCP Number of duplicate ACKs Number of Keep-alive packets DNS Number of sessions and good DNS responses ratio Number of flows


gpt--

The existing rule-based malware detection systems are highly accurate at their assigned tasks, but they cannot adapt to the evolution of malware. Some malware can even disable such systems. Moreover, it is easier for attackers to develop new bypassing techniques. The authors’ solution is based on cross-layer and cross-protocol traffic classification, using supervised learning methods.

Network behavioral modeling is widely used for malware detection, but most studies focus on specific malware types or attacks. This work combines multi-layer and multi-protocol features at various resolutions to detect both known and new malware.