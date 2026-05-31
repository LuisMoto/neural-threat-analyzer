import spacy
import re
import unicodedata

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
        
    def replace_url(match):
        url = match.group(0)
        if any(domain in url for domain in ['github', 'linkedin', 'google', 'microsoft']):
            return 'safe_link'
        return 'suspicious_link'
    
    text = re.sub(r'http\S+', replace_url, text)
    
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