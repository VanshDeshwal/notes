
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
- 