import spacy
import re
import unicodedata
from urllib.parse import urlparse

try:
    nlp = spacy.load("xx_ent_wiki_sm")
except OSError:
    nlp = spacy.load("en_core_web_sm")

security_keep_words = {
    "now", "here", "please", "urgent", "your", "must",
    "from", "where", "or", "and", "verify", "click",
    "confirm", "account", "login", "update", "suspend",
    "password", "free", "winner", "prize", "bank",
    "select", "drop", "insert", "union", "delete",
    "where", "from", "table", "exec", "cast", "admin",
    "urgente", "verificar", "confirmar", "cuenta",
    "contrasena", "actualizar", "acceder", "suspendido",
    "inmediatamente", "gratis", "ganador", "premio",
    "banco", "seguro", "clave", "usuario",
    "seleccionar", "eliminar", "insertar", "tabla", "base",
}

for word in security_keep_words:
    nlp.vocab[word].is_stop = False

SUSPICIOUS_TLDS = {
    '.xyz', '.tk', '.ml', '.ga', '.cf', '.gq',
    '.top', '.click', '.live', '.online', '.site'
}

TRUSTED_DOMAINS = {
    'github.com', 'overleaf.com', 'docs.google.com',
    'drive.google.com', 'zoom.us', 'aws.amazon.com',
    'pypi.org', 'stackoverflow.com', 'youtube.com'
}

def analyze_url_structure(match):
    url = match.group(0)
    try:
        parsed = urlparse(url if url.startswith(('http://', 'https://')) else 'http://' + url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        
        bare_domain = re.sub(r'^www\.', '', domain)

        if bare_domain in TRUSTED_DOMAINS:
            return '' 

        if re.match(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', domain):
            return 'url_ip_detected'

        if domain.count('.') > 2:
            return 'url_excessive_subdomains'

        if len(domain) > 40 or domain.count('-') > 3:
            return 'url_anomalous_length'
            
        if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
            return 'url_suspicious_tld'

        phishing_path_keywords = ['login', 'verify', 'account', 'secure', 'update', 'confirm', 'signin', 'banking']
        if any(kw in path for kw in phishing_path_keywords):
            return 'url_phishing_path'

        return '' 
    except ValueError:
        return 'url_malformed'

def security_preprocess(text):
    if not isinstance(text, str):
        return ""
        
    if len(text) > 10000:
        text = text[:10000]

    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    text = re.sub(r"'?\s*or\s+1\s*=\s*1\s*'?|'?\s*or\s+'1'\s*=\s*'1'\s*'?|\bor\s+\w+\s*=\s*\w+", " sql_tautology ", text)
    text = re.sub(r"--+", " sql_comment ", text)
    text = re.sub(r"/\*.*?\*/", " sql_comment_block ", text)
    text = re.sub(r";\s*drop\s+table", " sql_drop_table ", text)
    text = re.sub(r"\bunion\s+(?:all\s+)?select\b", " sql_union_select ", text)
    text = re.sub(r"\bexec\s*\(", " sql_exec ", text)
    text = re.sub(r"\bxp_cmdshell\b", " sql_xp_cmdshell ", text)
    text = re.sub(r"\binsert\s+into\b", " sql_insert_into ", text)
    text = re.sub(r"\bselect\s+\*\s+from\b", " sql_select_star ", text)

    url_pattern = r'(https?://\S+|www\.\S+|\b[\w-]{3,}\.(com|net|org|io|co|gov|edu|xyz|tk|ml|top|click|live|online|site)/\S*)'
    text = re.sub(url_pattern, analyze_url_structure, text)

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