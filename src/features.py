"""
Manual feature extraction — 14 features.

v2.2 fixes:
- Removed generic words like "update", "verify", "find", "account" from
  PHISHING_KEYWORDS_EN. These appear constantly in safe technical emails and
  were causing phishing_density to spike on innocent text.
- Phishing keywords now require HIGH specificity: words that almost never
  appear in legitimate corporate email.
- Added has_safe_url_only: text has url_safe_domain and NO suspicious URL,
  strong Safe signal.
"""

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
    "winner", "prize", "lottery", "inheritance",
    "nigerian", "beneficiary", "deceased", "attorney",
    "wire transfer", "western union", "moneygram",
    "act now", "limited offer", "claim your",
}

PHISHING_KEYWORDS_ES = {
    "ganador", "premio", "loteria", "herencia",
    "transferencia bancaria", "reclama", "oferta limitada",
    "suspendido",  "urgente", "verificar", "confirmar", "cuenta",
     "contrasena", "actualizar", "acceder", "inmediatamente",
}


SAFE_TECHNICAL_WORDS = {
    "repository", "commit", "branch", "pull", "push", "merge",
    "bug", "issue", "deploy", "server", "api", "endpoint",
    "module", "function", "class", "variable", "debug",
    "meeting", "agenda", "report", "metrics", "dashboard",
    "adjunto", "reporte", "reunion", "equipo", "proyecto",
    "documento", "version", "revision", "cambio",
}


def extract_features(preprocessed_text: str) -> list:
    tokens = set(preprocessed_text.lower().split())

    has_safe_url       = int(bool(tokens & URL_SAFE_TOKENS))
    has_any_url        = int(bool(tokens & URL_ALL_TOKENS))
    has_suspicious_url = int(bool(tokens & URL_SUSPICIOUS_TOKENS))
    has_ip_url         = int("url_ip_detected" in tokens)
    has_phishing_path  = int("url_phishing_path" in tokens)

    has_safe_url_only  = int(has_safe_url and not has_suspicious_url)

    has_sqli_token   = int(bool(tokens & SQLI_TOKENS))
    has_tautology    = int("sql_tautology" in tokens)
    has_union_select = int("sql_union_select" in tokens)
    has_drop_table   = int("sql_drop_table" in tokens)

    has_phishing_en = int(bool(tokens & PHISHING_KEYWORDS_EN))
    has_phishing_es = int(bool(tokens & PHISHING_KEYWORDS_ES))
    phishing_kw_count = len(tokens & (PHISHING_KEYWORDS_EN | PHISHING_KEYWORDS_ES))
    phishing_density  = min(phishing_kw_count / max(len(tokens), 1) * 20, 1.0)


    has_safe_context = int(bool(tokens & SAFE_TECHNICAL_WORDS))

    return [
        has_safe_url_only,  # 0  — trusted domain + no threat = Safe
        has_any_url,        # 1
        has_suspicious_url, # 2
        has_ip_url,         # 3
        has_phishing_path,  # 4
        has_sqli_token,     # 5
        has_tautology,      # 6
        has_union_select,   # 7
        has_drop_table,     # 8
        has_safe_context,   # 9     
        has_phishing_en,    # 10
        has_phishing_es,    # 11
        phishing_density,   # 12
        int(has_suspicious_url and (has_phishing_en or has_phishing_es)),  # 13
    ]


N_FEATURES = 14