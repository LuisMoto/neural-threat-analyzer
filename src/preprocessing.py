import spacy
import re
import unicodedata
from urllib.parse import urlparse

nlp = spacy.load("en_core_web_sm")

security_keep_words = {
    "now", "here", "please", "urgent", "your", "must",
    "from", "where", "or", "and", "select", "drop", "verify"
}

for word in security_keep_words:
    nlp.vocab[word].is_stop = False

def security_preprocess(text):
    if not isinstance(text, str):
        return ""
        
    if len(text) > 10000:
        text = text[:10000]

    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        
    def analyze_url_structure(match):
        url = match.group(0)
        
        parsed = urlparse(url if url.startswith(('http://', 'https://')) else 'http://' + url)
        domain = parsed.netloc
        
        if re.match(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', domain):
            return 'url_ip_detected'
            
        if domain.count('.') > 2:
            return 'url_excessive_subdomains'
            
        if len(domain) > 40 or domain.count('-') > 3:
            return 'url_anomalous_length'
            
        return 'url_standard'
    
    text = re.sub(r'http\S+', analyze_url_structure, text)
    
    text = text.replace("1=1", "sql_tautology")
    text = text.replace("'1'='1'", "sql_tautology")
    text = text.replace("--", "sql_comment")

    doc = nlp(text)

    clean_tokens = []
    for token in doc:
        if token.is_space:
            continue
        
        if token.is_stop and token.text not in security_keep_words:
            continue
        
        if token.is_punct and token.text not in ["'", '"', "=", "*"]:
            continue

        clean_tokens.append(token.lemma_)

    return " ".join(clean_tokens)