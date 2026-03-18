

def normalize_label_name(label_name, config):
    # aggressive normalization so that common variants all become to the same form
    value = label_name or ""
    if not isinstance(value, str):
        value = str(value)

    if getattr(config.label_normalization, "strip_whitespace", False):
        value = value.strip()

    if getattr(config.label_normalization, "case_sensitive", True) is False:
        value = value.lower()

    if getattr(config.label_normalization, "normalize_unicode_quotes", False):
        value = (value.replace("’", "'")
                 .replace("‘", "'")
                 .replace("`", "'")
                 .replace("“", '"')
                 .replace("”", '"'))

    if getattr(config.label_normalization, "normalize_hyphens_and_apostrophes", False):
        value = (value.replace("‐", "-").replace("–", "-").replace("—", "-"))
        value = value.replace("-", " ")
        value = value.replace("_", " ")
        value = value.replace("'", "")
        value = " ".join(value.split())

    return value


def get_label_variants(config, label_key):
    outcome_labels = getattr(config.label_normalization, "outcome_labels", None)
    if not outcome_labels or not hasattr(outcome_labels, label_key):
        return []

    label_cfg = getattr(outcome_labels, label_key)
    variants = list(getattr(label_cfg, "variants", []))

    normalized = []
    seen = set()

    for value in variants:
        clean = normalize_label_name(value, config)
        if clean and clean not in seen:
            normalized.append(clean)
            seen.add(clean)

    return sorted(normalized)


def get_wontfix_variants(config):
    # tiny helper to not need to hardcoding "wontfix" allll the time
    return get_label_variants(config, "wontfix")


def issue_has_label_variant(issue_payload, wanted_variants, config):
    wanted = set(wanted_variants)
    if not wanted:
        return False

    labels = issue_payload.get("labels") or []
    for label in labels:
        if isinstance(label, dict):
            name = label.get("name")
        else:
            name = str(label)

        if normalize_label_name(name, config) in wanted:
            return True

    return False


def issue_has_wontfix_label(issue_payload, config):
    return issue_has_label_variant(issue_payload, get_wontfix_variants(config), config)