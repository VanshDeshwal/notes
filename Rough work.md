Recent state-of-the-art Deep-Learning architectures for encrypted traffic classifications demonstrated superb results in tasks of traffic categorization over encrypted traffic. In this paper, we leverage the feasibility to use such architectures for the tasks of malware detection and classification
we present a Deep-Learning model for malware traffic detection and classification (MalDIST), which outperforms both classical ML and DL malware traffic classification models both in terms of detection and classification.
Network trafic is encrypted to improve privacy, but that makes it harder to classify malicious traffic.
attackers are taking advantage of encryption to deliver malware inside the network.
Authors are transfering models from internet classification domain to malware traffic classification because the former has shown great results.

Converting DISTILLER to MalDIST and comparing it to current state of the art malware traffic classification models

## Cleaned version

Today, network traffic is encrypted to improve privacy, but that makes it harder to classify malicious traffic. Attackers are taking advantage of encryption to deliver malware inside the network. Recent state-of-the-art deep learning architectures for encrypted traffic classification have demonstrated superb results in tasks of traffic categorization over encrypted traffic. One such architecture is DISTILLER, and the authors are transferring this model to malware detection and classification.
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

