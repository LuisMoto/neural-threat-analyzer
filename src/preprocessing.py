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
    "where", "from", "table", "exec", "cast",

    "urgente", "verificar", "confirmar", "cuenta",
    "contraseña", "actualizar", "acceder", "suspendido",
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
    'verificar', 'cuenta', 'acceder', 'contraseña', 'seguro'
]


def analyze_url_structure(match):
    """
    BUG FIX #1: Previously returned a single generic 'url_standard' token for
    most URLs, making safe emails with links indistinguishable from phishing.
    Now returns 6 granular tokens the model can actually learn from.

    BUG FIX #2: Previously the regex only matched 'http\S+', missing the vast
    majority of phishing URLs that appear as bare domains or www. links.
    The caller regex now covers those cases too.
    """
    url = match.group(0)

    try:
     
        normalized = url if url.startswith(('http://', 'https://')) else 'http://' + url
        parsed = urlparse(normalized)
        domain = parsed.netloc.lower().lstrip('www.')
        path = parsed.path.lower()


        if re.match(r'\d{1,3}(\.\d{1,3}){3}', domain):
            return 'url_ip_detected'

    
        if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
            return 'url_suspicious_tld'

        if domain.count('.') > 2:
            return 'url_excessive_subdomains'

       
        if len(domain) > 40 or domain.count('-') > 3:
            return 'url_anomalous_length'

   
        if any(kw in path for kw in PHISHING_PATH_KEYWORDS):
            return 'url_phishing_path'

        return 'url_standard'

    except ValueError:
        return 'url_malformed'


def security_preprocess(text: str) -> str:
    """
    Full NLP preprocessing pipeline for threat classification.
    Handles English and Spanish. Outputs a clean token string
    ready for TF-IDF or Keras TextVectorization.
    """
    if not isinstance(text, str):
        return ""

   
    if len(text) > 10000:
        text = text[:10000]

    text = text.lower()


    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

  
    text = text.replace("1=1", "sql_tautology")
    text = text.replace("'1'='1'", "sql_tautology")
    text = text.replace("1 =1", "sql_tautology")
    text = text.replace("--", "sql_comment")
    text = text.replace("/*", "sql_comment_block")
    text = text.replace("*/", "sql_comment_block")
    text = re.sub(r";\s*drop\s+table", "sql_drop_table", text)
    text = re.sub(r"union\s+select", "sql_union_select", text)
    text = re.sub(r"exec\s*\(", "sql_exec", text)
    text = re.sub(r"xp_cmdshell", "sql_xp_cmdshell", text)

   
  
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