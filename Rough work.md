
three Datasets: NetML, CICIDS2017, non-vpn2016

they have many features, some single valued and some array based

those features are divided into 4 categories
	Meta Data
	HTTP
	TLS
	DNS

MetaData: most of the features are array based, such as interval count histogram

TLS:
	14 different TLS features are extracted
	
DNS:
- dns_query_cnt and dns_answer_cnt features have a single integer value returned while the other DNS query and answer features return array of integers for dns_answer_ttl and strings for the rest.

HTTP:
- http_content_type feature is very important to classify benign samples from malware flows

Three ML models are used:
Random Forest, Support Vector Machine (SVM) and Multi-layer Perceptron (MLP)

protocol specific (TLS, DNS and HTTP) features are not available for all of the flows in the datasets. Therefore, we use only Metadata features to obtain baseline results.

We observe that Random Forest is the best performing model for malware detection tasks, but it lacks accuracy for traffic classification problems due to class imbalance.


There was no common dataset or benchmark method for network traffic analysis in the research community. This paper tries to solve that problem by providing a dataset and some baseline ML algorithm test results.
three Datasets: NetML, CICIDS2017, non-vpn2016, they have many features some single valued and some array based, each element of array is considered a seperate feature when using for training a ml model.
The features in each of the dataset can be divided into 4 parts namely Meta Data, HTTP, TLS, DNS. in metadata most of the features are array based such as interval count. for TLS 14 different TLS features are extracted. DNS: dns_query_cnt and dns_answer_cnt features have a single integer value returned while the other DNS query and answer features return array of integers for dns_answer_ttl and strings for the rest.. HTTP: http_content_type feature is very important to classify benign samples from malware flows, the above was the explaination about the datasets and features. now we choose three ml models Random Forest, Support Vector Machine (SVM) and Multi-layer Perceptron (MLP).protocol specific (TLS, DNS and HTTP) features are not available for all of the flows in the datasets. Therefore, we use only Metadata features to obtain baseline results.

We observe that Random Forest is the best performing model for malware detection tasks, but it lacks accuracy for traffic classification problems due to class imbalance.