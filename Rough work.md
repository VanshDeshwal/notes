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


Author is extracting 972 total features.correlation based feature selection is used to narrow those features down to...