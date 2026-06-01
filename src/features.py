URL_SAFE_TOKENS = {"url_safe_domain"}

URL_SUSPICIOUS_TOKENS = {
    "url_ip_detected",
    "url_suspicious_tld",
    "url_excessive_subdomains",
    "url_anomalous_length",
    "url_phishing_path",
    "url_malformed",
}

URL_ALL_TOKENS = URL_SUSPICIOUS_TOKENS | URL_SAFE_TOKENS | {"url_unknown"}

SQLI_TOKENS = {
    "sql_tautology",
    "sql_comment",
    "sql_comment_block",
    "sql_drop_table",
    "sql_union_select",
    "sql_exec",
    "sql_xp_cmdshell",
    "sql_insert_into",
    "sql_select_star",
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
    tokens = set(preprocessed_text.lower().split())
    text   = preprocessed_text.lower()

    has_safe_url        = int(bool(tokens & URL_SAFE_TOKENS))
    has_any_url         = int(bool(tokens & URL_ALL_TOKENS))
    has_suspicious_url  = int(bool(tokens & URL_SUSPICIOUS_TOKENS))
    has_ip_url          = int("url_ip_detected" in tokens)
    has_phishing_path   = int("url_phishing_path" in tokens)

    has_sqli_token   = int(bool(tokens & SQLI_TOKENS))
    has_tautology    = int("sql_tautology" in tokens)
    has_union_select = int("sql_union_select" in tokens)
    has_drop_table   = int("sql_drop_table" in tokens)
    has_select_star  = int("sql_select_star" in tokens)

    has_phishing_en = int(bool(tokens & PHISHING_KEYWORDS_EN))
    has_phishing_es = int(bool(tokens & PHISHING_KEYWORDS_ES))
    phishing_kw_count = len(tokens & (PHISHING_KEYWORDS_EN | PHISHING_KEYWORDS_ES))
    phishing_density  = min(phishing_kw_count / max(len(tokens), 1) * 10, 1.0)

 
    url_plus_phishing = int(has_suspicious_url and (has_phishing_en or has_phishing_es))

    return [
        has_safe_url,       # 0 
        has_any_url,        # 1
        has_suspicious_url, # 2
        has_ip_url,         # 3
        has_phishing_path,  # 4
        has_sqli_token,     # 5
        has_tautology,      # 6
        has_union_select,   # 7
        has_drop_table,     # 8
        has_select_star,    # 9
        has_phishing_en,    # 10
        has_phishing_es,    # 11
        phishing_density,   # 12
        url_plus_phishing,  # 13
    ]


N_FEATURES = 14