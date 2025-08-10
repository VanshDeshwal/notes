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
The authors used a combination of three datasets for their experiments. Benign traffic was taken from **StratosphereIPS** (same benign PCAPs as in the NetML dataset) and **ISCX2016** (105 selected benign PCAPs out of 150 labeled files covering various applications like Facebook, YouTube, Spotify, and traffic types such as streaming, VoIP, and chat, with VPN/non-VPN labels). Malware traffic was obtained from **Malware-Traffic-Analysis.net (MTA)**, selecting four families: Dridex, Hancitor, Emotet, and Valak.
For preprocessing, sessions with less than 784 payload bytes were removed, as they were not informative enough. The ISCX2016 dataset was further cleaned by removing irrelevant protocols (e.g., SNMP, LLMNR) and noisy traffic such as UDP broadcasts (e.g., Dropbox LAN Discovery). The benign set was built as an equal mix of StratosphereIPS and ISCX2016 (after filtering) to increase diversity. To address the imbalance between benign and malicious samples, the benign set was sub-sampled so that the final dataset contained 50% benign and 50% malware samples, totaling 18k sessions. Of these, 57.9% were TLS-encrypted. While the benign/malicious classes were balanced, the malware family distribution remained imbalanced.
For each group, we calculate five statistics for packet size and inter-arrival time (10 features) plus four totals and rates (bytes, packets, bytes/s, packets/s). Arranging these for five groups forms a 5×14 feature matrix, representing the malware or benign traffic as an image.

MalDIST was compared against:

1. **Classical ML models** — Random Forest (RF), Support Vector Machine (SVM) with standardized data, and K-Nearest Neighbor (KNN) with _k=3_, which gave the best results among tested values (_k=3–10_).
    
2. **DeepMAL** — Session-based variant (better-performing) using a convolutional layer and fully connected layers for malware detection and classification.
    
3. **M2CNN** — CNN (based on LeNet-5) using the first 784 payload bytes of each session reshaped into a 28×28 image for malware detection.
    
4. **M1CNN** — DL model on raw traffic data without feature extraction, tested for encapsulation identification, encrypted traffic classification, and protocol-specific classification.

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

