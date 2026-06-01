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
    '.top', '.click', '.link', '.work', '.live',
    '.online', '.site', '.icu', '.buzz'
}


PHISHING_PATH_KEYWORDS = [
    'login', 'verify', 'account', 'secure', 'update',
    'confirm', 'signin', 'banking', 'password', 'credential',
    'webscr', 'cmd=', 'dispatch', 'redirect',
    'verificar', 'cuenta', 'acceder', 'contrasena',
]


TRUSTED_DOMAINS = {
    'github.com', 'gitlab.com', 'bitbucket.org',
    'google.com', 'docs.google.com', 'drive.google.com',
    'microsoft.com', 'office.com', 'sharepoint.com',
    'notion.so', 'overleaf.com', 'arxiv.org',
    'stackoverflow.com', 'youtube.com', 'linkedin.com',
    'zoom.us', 'meet.google.com', 'teams.microsoft.com',
    'slack.com', 'trello.com', 'jira.atlassian.com',
    'dropbox.com', 'onedrive.live.com', 'box.com',
}


def analyze_url_structure(match):
    url = match.group(0)
    try:
        normalized = url if url.startswith(('http://', 'https://')) else 'http://' + url
        parsed = urlparse(normalized)
        domain = parsed.netloc.lower()
  
        bare_domain = domain.lstrip('www.').split(':')[0]
        path = parsed.path.lower()


        if re.match(r'\d{1,3}(\.\d{1,3}){3}', bare_domain):
            return 'url_ip_detected'

  
        if bare_domain in TRUSTED_DOMAINS:
            return 'url_safe_domain'

        if any(bare_domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
            return 'url_suspicious_tld'

        
        if bare_domain.count('.') > 2:
            return 'url_excessive_subdomains'

        if len(bare_domain) > 40 or bare_domain.count('-') > 3:
            return 'url_anomalous_length'

        
        if any(kw in path for kw in PHISHING_PATH_KEYWORDS):
            return 'url_phishing_path'

       
        return 'url_unknown'   

    except ValueError:
        return 'url_malformed'


def security_preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""

    if len(text) > 10000:
        text = text[:10000]

    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    text = re.sub(r"'?\s*or\s+1\s*=\s*1\s*'?", " sql_tautology ", text)  
    text = re.sub(r"'?\s*or\s+'1'\s*=\s*'1'\s*'?", " sql_tautology ", text)
    text = re.sub(r"\bor\s+\w+\s*=\s*\w+", " sql_tautology ", text)       
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
        if token.is_punct and token.text not in ["'", '"', "=", "*", ";"]:
            continue
        clean_tokens.append(token.lemma_)

    return " ".join(clean_tokens)