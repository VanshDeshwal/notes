🔹 Packet Analysis for Network Forensics: A Comprehensive Survey

Slide 1
“Alright, so the first paper is a survey on packet analysis in network forensics.
Basically, it looks at how we capture packets, how we analyze them, and even the legal and privacy issues involved.
It doesn’t propose a new system, but instead maps out the existing tools and directions for future research.”

Slide 2
“Here, the tools are grouped by what they focus on.
Some analyze raw packets, others flows, and others the payload.
They also point out AI and machine learning examples — like an SVM plugin for Snort, and deep packet inspection for encrypted traffic.
There’s also this knowledge-based reasoning idea, using ontologies like PACO and PAO.
Uh, broadly, you can think of two categories: carving and reconstruction tools, and tracing analyzers.”

Slide 3
“So yeah, this survey is very complete — it ties together AI, DPI, and forensics.
But the limitation is that it doesn’t include datasets, metrics, or detailed features.
For future work, they mention privacy-preserving DPI, applying ML to encrypted or unknown traffic, and scaling up for IoT and cloud.
So, it’s more like a roadmap than an experimental paper.”

🔹 Network Traffic Classification for Data Fusion: A Survey

Slide 1
“Okay, the second survey is about traffic classification, and how it supports security and malware detection.
It’s also more of a reference work — summarizing techniques, datasets, features, evaluation methods, and open challenges.”

Slide 2
“They divide methods into five types: statistics-based, correlation-based, behavior-based, payload-based, and port-based.
They review 36 features — across packet, flow, and connection levels — and they show how combining levels improves detection.
They also look at 16 datasets, like ISCX, KDD Cup99, DARPA.
And they suggest standard criteria to compare methods, like accuracy, robustness, online capability, handling unknown traffic, and granularity.”

Slide 3
“One strength is how structured it is — very useful for newcomers.
But it doesn’t rank which features matter most, and datasets aren’t always clear about OS details or malware versus benign splits.
The main open problems are: reclassifying unknown traffic, making classifiers lightweight but still accurate, and better ground-truth collection.
So again, it’s a structured reference, not a system.”

🔹 NetML: A Challenge for Network Traffic Analytics

Slide 1
“Alright, next is the NetML paper, from 2020 at the NetAI Workshop.
The motivation is clear: in computer vision we have ImageNet, in NLP we have COCO, but in network traffic analysis, there wasn’t a benchmark dataset.
So the authors created three open datasets and provided baseline ML results, calling it the NetML challenge.”

Slide 2 — Dataset
“The three datasets target malware detection and traffic classification.
First, the NetML dataset — half a million flows, with 20 malware families plus benign traffic. Mostly from Windows, but also some benign Linux and Kali captures.
Second, CICIDS2017 — about 550k flows, covering 7 attacks plus benign. Two attacks were excluded because features couldn’t be extracted. This one is mainly Windows.
Third, non-VPN2016 — smaller, about 160k flows, focused on classifying applications like chat, email, video, down to fine-grained classes like Skype audio vs Skype chat.
So overall, Windows dominates, but Linux and Kali are also present.”

Slide 3 — Features
“They extracted four categories of features.
Metadata — always available, like packet counts, byte counts, flow duration.
Then protocol-specific ones: TLS, DNS, HTTP. For example, TLS cipher suites, DNS queries and answers, HTTP request methods and content types.
They noticed some features stand out — like header-bin-40 being smaller in malware flows, or HTTP content type being a strong signal.
But for the baselines, only metadata was used, because not every flow has TLS or HTTP.”

Slide 4 — Method
“They tested three baseline models — Random Forest, SVM, and an MLP.
Each represents a different family: ensemble, kernel-based, and neural nets.
Preprocessing included expanding arrays, masking IPs, and standardizing features.
And again, only metadata features were used to keep things consistent.”

Slide 5 — Evaluation
“For binary malware detection, results were excellent — Random Forest hit a true positive rate of 99.3% with false alarms under 1%.
For multi-class, CICIDS2017 performed very well, F1 around 0.99.
On NetML dataset, F1 dropped to 0.74 — some malware families got confused.
And on non-VPN2016, all models struggled — F1 below 0.63, due to class imbalance, like Skype and Facebook being overrepresented.
So in short — Random Forest is strong for binary detection, but imbalance is a big challenge for multi-class tasks.”

🔹 MalDIST: From Encrypted Traffic Classification to Malware Detection

Slide 1
“Okay, next up is MalDIST, presented at IEEE CCNC 2022.
The idea is pretty neat: instead of making a whole new model, they adapted an existing encrypted traffic classifier, DISTILLER, to detect malware.
They tested it for both binary detection and multi-class classification.”

Slide 2 — Dataset
“The dataset combines benign traffic from StratosphereIPS and ISCX2016, and malicious traffic from Malware-Traffic-Analysis.net — focusing on Dridex, Hancitor, Emotet, and Valak.
After filtering, they had around 18,000 sessions, balanced between benign and malicious, and about 58% TLS-encrypted.
Also, since StratosphereIPS benign captures are the same as NetML, we can infer both Windows and Kali traffic are included.”

Slide 3 — Features
“They used three modalities.
First, payload bytes — the first 784 bytes of each session.
Second, protocol fields — from the first 32 packets, like direction, size, inter-arrival time.
Third, statistical features — also from the first 32 packets, grouped into 5 sets, with 14 features each, like min, max, mean, skewness.
So, across layers 2 to 4, they cover raw payload, packet-level details, and compact statistics.”

Slide 4 — Method
“The architecture is multi-modal.
Payload bytes go through 1D CNNs.
Protocol fields go through a BiGRU.
Statistical features go through a BiLSTM, then 2D CNNs.
Outputs are merged, then split into two heads — one for binary detection, one for malware family classification.”

Slide 5 — Evaluation
“For binary detection, results were almost perfect — 99.7% across accuracy, precision, recall, F1.
For multi-class, MalDIST also outperformed both ML and DL baselines.
Only Dridex was weaker, about 82%, probably due to fewer samples.
But overall, MalDIST set a new benchmark.”

🔹 Unknown Malware Detection Using Network Traffic Classification

Slide 1
“Now, this paper is from 2015, IEEE CNS.
It proposes a supervised ML intrusion detection system that works across multiple layers and protocols, unlike rule-based IDS like Snort or Suricata.
They don’t say which OS they used, but since they relied on tcpdump and Wireshark — both Linux-native tools — it’s safe to assume Linux.”

Slide 2 — Dataset
“Malicious traffic came from Verint sandbox, VirusTotal, Emerging Threats, and community datasets.
Benign traffic came from student lab activity and corporate networks.
So, a mix of sandbox and real-world data — which is good for testing robustness.”

Slide 3 — Features
“They mapped features to OSI layers.
At transport, things like number of RST, ACK, duplicate ACK, keep-alive packets.
At application, HTTP timing, and DNS features like query rank and number of records.
And cross-layer features like number of flows per window.
So, quite a rich feature set.”

Slide 4 — Method
“They tested Naïve Bayes, J48 decision trees, and Random Forest in Weka.
They also did feature selection — reduced about a thousand features down to 12 key ones.
So, it’s efficient while still being accurate.”

Slide 5 — Evaluation
“Random Forest was the best.
For family classification, accuracy was almost perfect.
For unseen families, AUC was about 0.98, except Conficker, which was 0.77.
The system also detected threats earlier than Snort or Suricata — up to a month before rules were deployed.
That’s a strong result, showing generalization beyond signatures.”

🔹 APT Attack Detection Using Flow Network Analysis with Deep Learning

Slide 1
“Next, we have a 2021 paper on APT detection at the flow level.
The goal was to classify IPs as normal or infected with APT.”

Slide 2 — Dataset
“They used benign traffic from a Vietnam e-government server, and malicious traffic from the CTU-13 Malware Capture Facility.
CTU-13 includes many malware families, making it realistic.
The dataset is imbalanced — way more benign IPs than malicious — which matches real-world conditions.”

Slide 3 — Features
“All features came from CICFlowMeter, so flow-level only, no payload.
Layer 2 had counts, Layer 3 overall flow behavior, and Layer 4 timing and TCP connection dynamics.
So it’s metadata-focused and privacy-friendly.”

Slide 4 — Method
“They tried three supervised models: MLP, GCN, and a BiLSTM-GCN hybrid.
The hybrid uses both sequential and graph structure, which helps capture relations between IPs.”

Slide 5 — Evaluation
“The BiLSTM-GCN was the best — 99% accuracy and recall, even with imbalance.
MLP and GCN were also good, but slightly weaker.
The key point is: using graph relationships reduced missed detections.
No OS or system context was discussed in the paper.”

🔹 Network Malware Classification: DPI vs Flow Features

Slide 1
“And finally, a 2015 paper comparing deep packet inspection with flow-based features.
Malware was executed in Windows XP environments, while benign traffic came from home, lab, corporate, and ISP networks.
So, both packet-level and flow-level views were tested.”

Slide 2
“The malware dataset was manually labeled by family names, using MARFPCAT meta files.
Benign traffic came from diverse environments.
Annotation was manual, which they note as an important dependency.”

Slide 3
“
In terms of layers: L1–L3 were headers, L4 involved payload inspection.”
Flow-based features came from packet headers: duration, packet counts, byte counts, inter-arrival times.
DPI features were signal-based: FFT coefficients, LPC parameters, MinMax amplitudes, bi-grams.
So flow ignores payload, DPI processes it.

Slide 4
“The flow approach hit at least 98% accuracy across benign datasets, with very low false positives.
DPI also had high accuracy, but struggled with generic malware families because labeling was harder.
The main takeaway is: DPI can be effective with just a couple packets — good for early detection — while flow-based methods work well when you have more complete statistics.”