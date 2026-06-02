import spacy
import re
import unicodedata
from urllib.parse import urlparse

# Intentamos cargar un modelo multilenguaje de SpaCy para NLP.
# Si no está instalado, caemos en el modelo estándar en inglés.
try:
    nlp = spacy.load("xx_ent_wiki_sm")
except OSError:
    nlp = spacy.load("en_core_web_sm")

# Definimos un conjunto de palabras que normalmente serían eliminadas 
# (stopwords) por SpaCy, pero que nosotros NECESITAMOS conservar porque 
# son críticas para detectar phishing o inyecciones SQL en inglés y español.
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

# Iteramos sobre nuestro conjunto de palabras clave de seguridad y 
# le decimos a SpaCy que deje de considerarlas como "stopwords".
for word in security_keep_words:
    nlp.vocab[word].is_stop = False

# Definimos Dominios de Nivel Superior (TLDs) que estadísticamente 
# están muy asociados con la creación de sitios de phishing o spam.
SUSPICIOUS_TLDS = {
    '.xyz', '.tk', '.ml', '.ga', '.cf', '.gq',
    '.top', '.click', '.live', '.online', '.site'
}

# Definimos dominios seguros que sabemos que no representan una amenaza,
# esto evita falsos positivos en textos legítimos de trabajo.
TRUSTED_DOMAINS = {
    'github.com', 'overleaf.com', 'docs.google.com',
    'drive.google.com', 'zoom.us', 'aws.amazon.com',
    'pypi.org', 'stackoverflow.com', 'youtube.com'
}

# Función que toma una URL cruda encontrada en el texto y la clasifica
# reemplazándola por un token especial que la red neuronal pueda entender.
def analyze_url_structure(match):
    url = match.group(0)
    try:
        # Aseguramos que la URL tenga un esquema para poder analizarla.
        parsed = urlparse(url if url.startswith(('http://', 'https://')) else 'http://' + url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        
        # Removemos el "www." para estandarizar el dominio.
        bare_domain = re.sub(r'^www\.', '', domain)

        # Si el dominio es de confianza, lo eliminamos del texto 
        # para que no genere ruido en la clasificación.
        if bare_domain in TRUSTED_DOMAINS:
            return '' 

        # Si el dominio es una dirección IP directa (ej. 192.168.1.1), 
        # insertamos un token de alerta.
        if re.match(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', domain):
            return 'url_ip_detected'

        # Si el dominio tiene demasiados subdominios, es sospechoso.
        if domain.count('.') > 2:
            return 'url_excessive_subdomains'

        # Si el dominio es anormalmente largo o tiene muchos guiones,
        # insertamos un token de alerta estructural.
        if len(domain) > 40 or domain.count('-') > 3:
            return 'url_anomalous_length'
            
        # Revisamos si el dominio termina en alguno de los TLDs baratos/sospechosos.
        if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
            return 'url_suspicious_tld'

        # Buscamos palabras típicas de phishing dentro de la ruta de la URL
        # (ej. sitio.com/secure-login-update).
        phishing_path_keywords = ['login', 'verify', 'account', 'secure', 'update', 'confirm', 'signin', 'banking']
        if any(kw in path for kw in phishing_path_keywords):
            return 'url_phishing_path'

        # Si la URL pasa todos los filtros, simplemente la borramos 
        # porque no aporta valor semántico.
        return '' 
    except ValueError:
        # Si la URL está mal formateada y lanza error, devolvemos un token de malformación.
        return 'url_malformed'

# Función principal que limpia y prepara todo el texto antes de mandarlo al modelo.
def security_preprocess(text):
    # Validamos que la entrada sea de tipo texto.
    if not isinstance(text, str):
        return ""
        
    # Truncamos textos excesivamente largos para no desbordar la memoria.
    if len(text) > 10000:
        text = text[:10000]

    # Convertimos todo a minúsculas para estandarizar.
    text = text.lower()
    
    # Removemos acentos y caracteres especiales raros normalizando el texto a ASCII puro.
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Usamos Expresiones Regulares (RegEx) para buscar patrones exactos de inyecciones SQL
    # y los reemplazamos por tokens descriptivos para la red neuronal.
    
    # Detectamos tautologías lógicas (ej. ' or 1=1).
    text = re.sub(r"'?\s*or\s+1\s*=\s*1\s*'?|'?\s*or\s+'1'\s*=\s*'1'\s*'?|\bor\s+\w+\s*=\s*\w+", " sql_tautology ", text)
    # Detectamos comentarios SQL de una línea (--).
    text = re.sub(r"--+", " sql_comment ", text)
    # Detectamos bloques de comentarios SQL (/* ... */).
    text = re.sub(r"/\*.*?\*/", " sql_comment_block ", text)
    # Detectamos comandos de destrucción de tablas.
    text = re.sub(r";\s*drop\s+table", " sql_drop_table ", text)
    # Detectamos consultas anidadas engañosas (UNION SELECT).
    text = re.sub(r"\bunion\s+(?:all\s+)?select\b", " sql_union_select ", text)
    # Detectamos ejecución de comandos almacenados.
    text = re.sub(r"\bexec\s*\(", " sql_exec ", text)
    # Detectamos invocaciones a consolas de comandos de SQL Server.
    text = re.sub(r"\bxp_cmdshell\b", " sql_xp_cmdshell ", text)
    # Detectamos comandos de inserción.
    text = re.sub(r"\binsert\s+into\b", " sql_insert_into ", text)
    # Detectamos extracción masiva de datos.
    text = re.sub(r"\bselect\s+\*\s+from\b", " sql_select_star ", text)

    # Identificamos cualquier URL en el texto y le aplicamos nuestra función de análisis estructural.
    url_pattern = r'(https?://\S+|www\.\S+|\b[\w-]{3,}\.(com|net|org|io|co|gov|edu|xyz|tk|ml|top|click|live|online|site)/\S*)'
    text = re.sub(url_pattern, analyze_url_structure, text)

    # Pasamos el texto limpio por el motor de SpaCy para hacer análisis lingüístico (tokenización).
    doc = nlp(text)

    clean_tokens = []
    # Iteramos sobre cada token (palabra/signo) detectado por SpaCy.
    for token in doc:
        # Ignoramos espacios en blanco.
        if token.is_space:
            continue
        # Ignoramos stopwords normales (ej. "el", "la"), pero mantenemos nuestras palabras de seguridad.
        if token.is_stop and token.text not in security_keep_words:
            continue
        # Ignoramos la mayoría de los signos de puntuación, excepto comillas, iguales y asteriscos
        # que suelen ser indicativos de inyecciones SQL.
        if token.is_punct and token.text not in ["'", '"', "=", "*"]:
            continue
            
        # Extraemos la raíz de la palabra (lematización) para reducir el vocabulario
        # (ej. "corriendo" -> "correr") y la agregamos a nuestra lista limpia.
        clean_tokens.append(token.lemma_)

    # Unimos los tokens limpios de nuevo en una sola cadena de texto.
    return " ".join(clean_tokens)
