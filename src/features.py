URL_SUSPICIOUS_TOKENS = {
    "url_ip_detected",
    "url_suspicious_tld",
    "url_excessive_subdomains",
    "url_anomalous_length",
    "url_phishing_path",
    "url_malformed",
}

URL_ALL_TOKENS = URL_SUSPICIOUS_TOKENS | {"url_standard"}


SQLI_TOKENS = {
    "sql_tautology",
    "sql_comment",
    "sql_comment_block",
    "sql_drop_table",
    "sql_union_select",
    "sql_exec",
    "sql_xp_cmdshell",
}

PHISHING_KEYWORDS_EN = {
    "urgent", "verify", "confirm", "click", "suspend",
    "account", "password", "update", "login", "free",
    "winner", "prize", "bank", "credential",
}

PHISHING_KEYWORDS_ES = {
    "urgente", "verificar", "confirmar", "cuenta",
    "contrasena", "actualizar", "acceder", "suspendido",
    "gratis", "ganador", "banco", "clave",
}


def extract_features(preprocessed_text: str) -> list:
    """
    Extract 12 binary/count features from ALREADY preprocessed text.
    Input must be the output of security_preprocess(), not raw text.

    Returns a list of 12 numeric features.
    """
    tokens = set(preprocessed_text.lower().split())
    text = preprocessed_text.lower()

   
    has_any_url = int(bool(tokens & URL_ALL_TOKENS))
    has_suspicious_url = int(bool(tokens & URL_SUSPICIOUS_TOKENS))
    has_ip_url = int("url_ip_detected" in tokens)
    has_phishing_path = int("url_phishing_path" in tokens)


    has_sqli_token = int(bool(tokens & SQLI_TOKENS))
    has_tautology = int("sql_tautology" in tokens)
    has_union_select = int("sql_union_select" in tokens)
    has_raw_equals = int("=" in text and has_sqli_token)  

    has_phishing_en = int(bool(tokens & PHISHING_KEYWORDS_EN))
    has_phishing_es = int(bool(tokens & PHISHING_KEYWORDS_ES))
    phishing_keyword_count = len(tokens & (PHISHING_KEYWORDS_EN | PHISHING_KEYWORDS_ES))
    phishing_signal_density = min(phishing_keyword_count / max(len(tokens), 1) * 10, 1.0)


    url_plus_phishing = int(has_any_url and (has_phishing_en or has_phishing_es))

    return [
        has_any_url,            # 0
        has_suspicious_url,     # 1
        has_ip_url,             # 2
        has_phishing_path,      # 3
        has_sqli_token,         # 4
        has_tautology,          # 5
        has_union_select,       # 6
        has_raw_equals,         # 7
        has_phishing_en,        # 8
        has_phishing_es,        # 9
        phishing_signal_density,# 10
        url_plus_phishing,      # 11
    ]



N_FEATURES = 12