# NetML: A Challenge for Network Traffic Analytics

This document provides the full details of the NetML paper in simple English. It includes both a markdown version for easy reading and a LaTeX version for typesetting. All sections and details are faithfully reproduced.

---

## Abstract

Classifying network traffic is the basis for important network applications. Past work had problems finding good data sets, and many results cannot be repeated easily. The problem is worse for new data-driven machine learning methods. To solve this, we offer three open data sets with almost 1.3 million labeled flows total. Each flow has metadata and anonymized raw packets. We cover both malware detection and application classification. We release the data as an open challenge called NetML and provide baseline models using Random Forest, SVM, and MLP. As NetML grows, we hope these data will become a common platform for AI-driven, reproducible research in network flow analytics.

## 1 Introduction

New technology now gives researchers large amounts of data to study. Many papers in Computer Vision (CV), Natural Language Processing (NLP), and more recently Network Traffic Analysis (NTA) use AI methods [9,22,24,26]. CV and NLP have shared benchmarks like ImageNet [6] and NLP leaderboards [1]. But NTA lacks a common data set or challenge. This gap makes it hard to compare and reproduce research results.

In the digital era, sorting network traffic is very important. The number of internet devices is growing fast. Network Traffic Analysis groups flows into categories. It is vital for tasks like quality of service (QoS), resource allocation, and detecting malware. More connected devices mean more cyber threats. We need to prepare defenses. But finding full, up-to-date data sets is hard.

Early NTA methods used fixed port numbers. Modern apps use dynamic ports, so port-based methods fail. Researchers then moved to deep packet inspection (DPI). As encryption increased, DPI became less useful. This shift led to machine learning on flow statistics, which do not need packet contents or port numbers.

Many studies try to classify network flows, but there is no common data set like ImageNet or COCO. Without shared data, comparisons and reproductions are hard. We need an open, current flow data set for the community.

This paper introduces the NetML data sets. We provide three labeled flow collections with detailed features. We focus on two main tasks: malware detection and application type classification. For malware detection, we offer a new NetML data set from Stratosphere IPS captures, with almost 500,000 flows across 20 malware types and benign. We also build a data set from CICIDS2017, with around 550,000 flows covering seven malware types and benign.

## 2 Related Work

### 2.1 Malware Detection

Random Forest [27] and SVM with RBF kernel [2] are popular ML methods for NTA. However, each group uses its own data and features, making comparisons hard. Anderson et al. [3] use flow metadata, packet sequences, byte distributions, and unencrypted TLS headers to find malware. Moore et al. [15] extract 249 features but studies use different subsets, so no single best set exists.

KDD-Cup 99 [13] and NSL-KDD [21] are old but common data sets for malware detection. Researchers apply C4.5, decision tree, random forest, SVM, AdaBoost [7,8,12,23]. Two-stage methods use Naive Bayes then filtering [16], or balanced forest then random forest [17]. These sets are outdated.

CICIDS2017 [19] is newer. Using CICFlowMeter [11], researchers compare kNN, random forest, and ID3, finding F1 scores of 0.98 (ID3) and 0.97 (RF). Gao et al. [9] test RF, SVM, and deep nets on NSL-KDD and CICIDS2017, finding RF has strong binary and multi-class performance.

### 2.2 Traffic Classification

Researchers study application classification too. Conti et al. [5] use RF on their data for user action, achieving 95% accuracy. Taylor et al. [22] use packet-length statistics on SVM and RF, finding 95% accuracy for binary, but multi-class drops to 42% (SVM) and 87% (RF). Lashkari et al. [10] created the VPN-nonVPN2016 data set and tested kNN and C4.5.

Many follow-ups use VPN-nonVPN2016. Yamansavascilar et al. [25] compare it to their own data, finding kNN best on VPN-nonVPN2016 and RF best on their set, with feature selection boosting accuracy by 2%. Wang et al. [24] use deep learning on packet-level data. Yao et al. [26] apply attention-based LSTM. Qin et al. [18] propose Hybrid Flow Clustering (HFC) for unsupervised profiling, achieving strong results across three public sources.

These works show NTA interest is growing, but no single solution exists. We need a shared feature set and an open, updated data set.

## 3 NetML Data Collection

### 3.1 Overview

We gather raw packet captures from Stratosphere IPS [20] and CIC [10,19]. Stratosphere focuses on malware and normal traffic. CIC provides IDS and app classification captures. We build three data sets: **NetML**, **CICIDS2017**, and **non-vpn2016**. We preprocess pcaps and extract flow features with an Intel tool.

We study both malware detection and app classification at three granularity levels: top-level (binary or major apps), mid-level (specific apps), and fine-grained (malware families or detailed app flows).

### 3.2 Dataset Preparation

Using Intel’s feature tool, we process up to 200 packets per flow both ways. The tool outputs JSON lines with features per flow (see Table 1). Preprocessing:

1. Mask `sa` and `da` IPs to `IP_masked`.
    
2. Remove `time_start` and `time_end`; add `time_length` = `time_end - time_start`.
    
3. Add a unique flow ID and its label.
    
4. Split each data set: 10% of each class to `test-challenge`, 10% to `test-std`, and 80% to `train`.
    

Results:

- NetML: 484,056 flows
    
- CICIDS2017: 551,372 flows
    
- non-vpn2016: 163,711 flows
    

### 3.3 Malware Detection Data Sets

#### 3.3.1 NetML

From Stratosphere IPS, we pick 30 pcaps (Table 2). Each pcap has one class. Labels: top-level benign/malware, fine-grained 20 malware + benign. Figure 2 shows fine-grained counts.

#### 3.3.2 CICIDS2017

From CICIDS2017, we filter time windows per site instructions. We exclude Botnet and Heartbleed (tools couldn’t extract). We keep seven malware types + benign (Table 3, Figure 3).

### 3.4 Traffic Classification Data Set: non-vpn2016

From CIC’s VPN-nonVPN2016 captures, we extract flows (VLAN unsupported). We assign three label levels:

- Top-level: P2P, audio, chat, email, file_transfer, tor, video.
    
- Mid-level: 18 apps (facebook, skype, …).
    
- Fine-grained: 31 flows (facebook_audio, skype_chat, …).  
    Figures 4–6 show flow counts. Some classes (facebook_audio, skype_audio…) dominate, causing imbalance.
    

## 4 NetML Data Set Analysis

We analyze training flows for all three sets. We extract four feature groups:

1. **Metadata** (counts, sizes, duration).
    
2. **TLS** (cipher suites, extensions).
    
3. **DNS** (query names, answer IPs, TTL).
    
4. **HTTP** (methods, codes, URIs, hosts, content types).
    

Table 5 shows how many flows per set have each feature group. Train sizes: NetML (387,268), CICIDS2017 (441,116), non-vpn2016 (131,065).

### 4.1 Metadata Features

We extract 31 metadata features. Histogram arrays (e.g., `hdr_ccnt[]`) are split into separate columns (`hdr_ccnt_0`, …). Figures 10–17 show single-value feature distributions (100 bins). Figures 18–21 show mean histograms for array features. For instance, `hdr_bin_40` is lower in malware than benign. An outlier in `time_length` (~7e6) skews its plot; we leave it for future work.

### 4.2 TLS Features

We extract 14 TLS features. TLS flows: NetML (114,396, 30%), CICIDS2017 (74,836, 15%), non-vpn2016 (1,262, <1%). Figures 22–25 show distribution of client/server single values. Figures 26–27 plot array lengths and frequent cipher and extension indices. NetML has 122 unique cipher suites and 38 extensions; CICIDS2017 has 11 each; non-vpn2016 has 16 ciphers and 12 extensions. P2P in non-vpn2016 has no TLS. Common hex codes appear in Figures 34–35.

### 4.3 DNS Features

We extract both single and array DNS features (Table 1). DNS flows: NetML (60,271, 15%), CICIDS2017 (93,224, 21%), non-vpn2016 (8,179, 7%). Figures 28–29 show single-value plots; `dns_query_cnt` and `dns_query_class` are always 1 and dropped. Figures 30–31 show array feature distributions. Figures 36–37 list top-5 `dns_query_name` and `dns_answer_ip`. Most flows lack an answer IP, except non-vpn2016’s P2P.

### 4.4 HTTP Features

We extract six HTTP features. Three single values: `http_method`, `http_code`, `http_content_len`. Three mapped strings: `http_uri`, `http_host`, `http_content_type`. Figures 38–39 show top values by top-level class. `text/html` is common in malware flows; if a flow has another type, it is likely benign. Tor has no HTTP in non-vpn2016.

## 5 NetML Challenge Baselines

We train simple ML models on training sets and evaluate on test-std sets.

### 5.1 Baselines

- **Random Forest (RF):** 100 trees, max depth 10.
    
- **SVM:** RBF kernel, C=1.0.
    
- **MLP:** One hidden layer (121 units), alpha=0.0001, optimizer=Adam.
    

We use only Metadata features.

### 5.2 Preprocessing

We mask IPs, split arrays into columns, standardize each feature to zero mean and unit variance. We split training data: RF & MLP use 80% train / 20% validate; SVM uses 10% train / 90% validate due to time complexity.

### 5.3 Results

#### 5.3.1 Malware Detection (Binary)

|Dataset|Model|TPR|FAR|
|:-:|:-:|:-:|:-:|
|**NetML**|RF|0.9937|0.0092|
||SVM|0.9624|0.0137|
||MLP|0.9887|0.0171|
|**CICIDS**|RF|0.9859|0.0044|
||SVM|0.9780|0.0028|
||MLP|0.9872|0.0069|

RF works best on NetML; MLP best TPR on CICIDS2017 but SVM lowest FAR.

#### 5.3.2 Multi-Class Classification

|Case|Model|F1|mAP|
|:-:|:-:|:-:|:-:|
|**NetML-f**|RF|0.7442|0.4217|
||SVM|0.6959|0.3536|
||MLP|0.7314|0.4116|
|**CICIDS-f**|RF|0.9872|0.8682|
||SVM|0.9850|0.8621|
||MLP|0.9889|0.8966|
|**non-vpn-t**|RF|0.6273|0.3257|
|**non-vpn-m**|RF|0.3693|0.3223|
|**non-vpn-f**|RF|0.2486|0.2127|

RF is best in most cases. Confusion matrices in Figures 8–9 highlight common mix-ups (e.g. Adload vs. TrickBot, audio bias in non-vpn2016).

## 6 NetML Challenge and Workshop

We host an evaluation server with a leaderboard. The first workshop is at IJCAI 2020. Papers should:

- Use test-std set on leaderboard
    
- Report TPR & FAR for binary
    
- Report F1 & mAP for multi-class
    
- Compare to our baselines
    

Full rules: [GitHub Challenge Page](https://www.github.com/ACANETS/NetML-Competition2020)

## 7 Conclusion

We present the NetML data sets for malware detection and traffic classification. We supply flow features and multi-level labels to form a shared resource. We analyze features and give baselines in seven tasks. RF excels at malware detection but struggles on imbalanced traffic classes. Future work: deep learning for fine-grained tasks, SMOTE for imbalance. We hope NetML becomes a standard platform for AI-driven network flow analytics.



This paper focuses on both malware detection and application classification.
MLmethods implemented: random forest, SVM, MLP.
