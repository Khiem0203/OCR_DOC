"""Vietnamese OCR error correction using handcrafted rule sets.

Applies trigram → bigram → isolated-word replacement rules loaded from
ocr_rules_handcrafted.json (same directory as this file).  No ML models,
no external dependencies beyond the standard library.
"""
import json
import re
from pathlib import Path


_VN_CHARS = re.compile(
    r"[áàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợ"
    r"úùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÉÈẺẼẸÊẾỀỂỄỆ"
    r"ÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]"
)

_RULES_PATH = Path(__file__).parent / "ocr_rules_handcrafted.json"
_rules_cache: dict | None = None


def _load_rules() -> dict:
    """Load trigram / bigram / isolated-word rules from the JSON file (lazy, cached)."""
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache

    rules: dict = {"trigrams": {}, "bigrams": {}, "isolated": {}}
    try:
        data = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
        # Skip metadata keys (start with "_") and entries with null values
        rules["trigrams"] = {
            k: v for k, v in data.get("trigrams", {}).items()
            if not k.startswith("_") and v is not None
        }
        rules["bigrams"] = {
            k: v for k, v in data.get("bigrams", {}).items()
            if not k.startswith("_") and v is not None
        }
        rules["isolated"] = {
            k: v for k, v in data.get("isolated_words", {}).items()
            if not k.startswith("_") and v is not None
        }
    except Exception as exc:
        import warnings
        warnings.warn(f"vi_ocr_correct: could not load rules from {_RULES_PATH}: {exc}")

    _rules_cache = rules
    return rules


def apply_rules(text: str) -> str:
    """Apply trigram → bigram → isolated-word rules to *text*.

    Order is intentional: longer-phrase rules run first for maximum context
    before shorter patterns are applied.
    """
    if not text:
        return text

    rules = _load_rules()

    # 1. Trigrams (3-word phrases — highest confidence)
    for raw, fix in rules["trigrams"].items():
        if raw in text:
            text = text.replace(raw, fix)

    # 2. Bigrams (2-word phrases)
    for raw, fix in rules["bigrams"].items():
        if raw in text:
            text = text.replace(raw, fix)

    # 3. Isolated words (whole-word boundary match to avoid partial replacements)
    for raw, fix in rules["isolated"].items():
        text = re.sub(r"\b" + re.escape(raw) + r"\b", fix, text)

    return text


def is_vietnamese_text(text: str) -> bool:
    """Return True when *text* contains Vietnamese-specific diacritic characters."""
    return bool(_VN_CHARS.search(text))



def correct_md(md_text: str) -> str:
    """Correct Vietnamese OCR errors in Markdown text, preserving structure.

    - Skips fenced code blocks (```...```).
    - Skips image references (``![...](...) ``).
    - Applies ``apply_rules()`` to every other non-empty line.
    """
    if not md_text:
        return md_text

    lines = md_text.split("\n")
    out: list[str] = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()

        # Toggle code-block state
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            out.append(line)
            continue

        if in_code_block or not stripped:
            out.append(line)
            continue

        # Skip image references
        if stripped.startswith("!["):
            out.append(line)
            continue

        # Apply rules to lines that contain Vietnamese text
        if is_vietnamese_text(stripped):
            # Preserve markdown prefix (heading #, list -, >, etc.) unchanged;
            # correct only the content portion.
            m = re.match(r"^(\s*(?:#{1,6}\s+|[-*+>]\s+|\d+\.\s+)?)(.+)$", line)
            if m:
                prefix, content = m.groups()
                out.append(prefix + apply_rules(content))
            else:
                out.append(apply_rules(line))
        else:
            out.append(line)

    return "\n".join(out)


def correct_md_file(md_path: str) -> bool:
    """Read a Markdown file, apply corrections, and overwrite it in place.

    Returns True if the file was modified, False otherwise.
    """
    try:
        text = Path(md_path).read_text(encoding="utf-8")
    except Exception:
        return False

    if not is_vietnamese_text(text):
        return False  # Nothing to do for non-Vietnamese content

    corrected = correct_md(text)
    if corrected == text:
        return False  # No changes

    try:
        Path(md_path).write_text(corrected, encoding="utf-8")
        return True
    except Exception:
        return False

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vi_ocr_correct.py <input.md> [output.md]")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else None

    text = Path(src).read_text(encoding="utf-8")
    print(f"Input ({len(text)} chars): {text[:200]}...\n")

    corrected = correct_md(text)

    if dst:
        Path(dst).write_text(corrected, encoding="utf-8")
        print(f"Saved: {dst}")
    else:
        print("Corrected output:\n", corrected[:500])
