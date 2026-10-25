"""
Verifier module for Turkish instruction-following evaluation.

Main APIs
---------
- verify_instruction(instruction_id, response, prompt=None, **kwargs) -> dict
- verify_task(response, prompt, instruction_ids, kwargs_per_instruction) -> list[dict]

Each verifier returns a result dict with:
    {
        "instruction_id": int,
        "passed": bool,
        "reason": str,
        "details": dict
    }

The `kwargs_per_instruction` argument of `verify_task` is expected to be a
mapping: {instruction_id: {kw_name: value, ...}, ...}

Notlar
------
- Türkçe büyük/küçük harf ayrımı için Türk alfabesi (Ç,Ğ,İ,Ö,Ş,Ü dahil)
  özel olarak ele alınır.
- 18 numaralı kural için doğrulama yaparken `prompt` parametresi gereklidir:
  modelin önce isteği aynen tekrar edip etmediği, bu `prompt` üzerinden kontrol edilir.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Tuple
from .instruction_patterns import INSTRUCTION_REGEX_PATTERNS

TURKISH_UPPER = set(
    "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
    "ÂÎÛ"
)

TURKISH_LOWER = set(
    "abcçdefgğhıijklmnoöprsştuüvyz"
    "âîû"
)

NON_TURKISH_LOWER = set(
    "qwx"
    "éèêë"
    "áàäãå"
    "íìï"
    "óòôõ"
    "úùü"
    "ñ"
)

NON_TURKISH_UPPER = set(
    "QWX"
    "ÉÈÊË"
    "ÁÀÄÃÅ"
    "ÍÌÏ"
    "ÓÒÔÕ"
    "ÚÙÜ"
    "Ñ"
    "Ç̌"  
)

COMPILED_INSTRUCTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in INSTRUCTION_REGEX_PATTERNS
]
def _result(
    instruction_id: int,
    passed: bool,
    reason: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "instruction_id": instruction_id,
        "passed": bool(passed),
        "reason": reason,
        "details": details or {},
    }


_WORD_RE = re.compile(
    r"[A-Za-zÇĞİÖŞÜÂÎÛçğıöşüâîû]+"
    r"(?:['’-][A-Za-zÇĞİÖŞÜÂÎÛçğıöşüâîû]+)*",
    flags=re.UNICODE,
)

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

_PARAGRAPH_SPLIT_RE = re.compile(r'\s*\*{3}\s*')

_SECTION_SPLIT_RE = re.compile(
    r'^\s*["“]?(?:\*\*)?\s*{splitter}\s+\d+["”]?\s*:?',
    flags=re.IGNORECASE | re.MULTILINE
)

def count_words(text: str) -> int:
    """Count 'words' as sequences of Unicode word characters."""
    return len(_WORD_RE.findall(text))


def count_sentences(text: str) -> int:
    """
    Approximate Turkish sentence count by splitting on . ! ?
    Empty fragments are ignored.
    """
    parts = _SENTENCE_SPLIT_RE.split(text)
    return sum(1 for p in parts if _WORD_RE.search(p))


def split_paragraphs_by_triple_asterisk(text: str) -> List[str]:
    """Paragraphs separated by markdown splitter *** (regex-based)."""
    parts = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text)]
    return [p for p in parts if p]


def split_paragraphs_by_lines(text: str) -> List[str]:
    """
    Paragraphs separated by line breaks.

    Non-empty lines are treated as paragraphs.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def split_paragraphs_by_double_newline(text: str) -> List[str]:
    """
    Paragraphs separated by double line breaks.

    A paragraph is defined as a block of text separated by
    two consecutive newline characters ("\n\n").
    """
    return [p.strip() for p in text.strip().split("\n\n") if p.strip()]


def count_placeholders(text: str) -> int:
    """Count [placeholder] patterns with square brackets."""
    return len(re.findall(r"\[[^][\n]+\]", text))


def count_bullets_star(text: str) -> int:
    """
    Counts markdown bullet points written with '* '.

    Counted:
    - Bullet points starting at the beginning of a line: '* item'
    - Bullet points appearing after a sentence: '. * item'

    Not counted:
    - Markdown separators such as '***', '****'
    - Asterisks without a following space (e.g. '*Word')
    - Asterisks used inside words or formatting

    Notes:
    - A space after '*' is mandatory for a valid bullet
    - The bullet must either start a line or follow a period and space
    """
    pattern = r"(?m)^\s*\* "
    return len(re.findall(pattern, text))


def count_emphasized_sections(text: str) -> int:
    """
    Count sections emphasized in markdown with single asterisks,
    e.g. *vurgulanmış bölüm*
    """
    return len(re.findall(r"\*[^*\n]+\*", text))


def count_all_caps_words(text: str) -> int:
    """
    Count tokens that contain at least one letter
    and whose ALL letters are uppercase.
    Non-letter characters (= < > " etc.) are ignored
    for the uppercase check but do NOT invalidate the token.
    """
    tokens = re.findall(r"\S+", text, flags=re.UNICODE)
    count = 0

    for tok in tokens:
        letters = [ch for ch in tok if ch.isalpha()]
        if not letters:
            continue
        if all(ch.isupper() for ch in letters):
            count += 1

    return count


def contains_word(text: str, word: str) -> bool:
    """Case-insensitive word presence, using word boundaries."""
    pattern = r"\b" + re.escape(word) 
    return bool(re.search(pattern, text, flags=re.IGNORECASE | re.UNICODE))


def count_word_occurrences(text: str, word: str) -> int:
    """Case-insensitive count of a full word."""
    pattern = r"\b" + re.escape(word) 
    return len(re.findall(pattern, text, flags=re.IGNORECASE | re.UNICODE))


def count_substring_casefold(text: str, substring: str) -> int:
    """Case-insensitive substring occurrences using casefold."""
    t = text.casefold()
    s = substring.casefold()
    if not s:
        return 0
    count = 0
    start = 0
    while True:
        idx = t.find(s, start)
        if idx == -1:
            break
        count += 1
        start = idx + len(s)
    return count


def turkish_lower(text: str) -> str:
    """
    Turkish-aware lowercase conversion.
    Handles I/İ correctly before applying lower().
    """
    if not text:
        return text

    text = text.replace("İ", "i").replace("I", "ı")
    return text.lower()


def normalize(text: str) -> str:
    """
    Normalize text for comparison:
    - Unicode NFKC normalization
    - Turkish-aware lowercasing
    - Remove punctuation
    - Collapse whitespace
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = turkish_lower(text)

    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_instruction_sentence(line: str) -> bool:
    """
    Check whether a single line is an instruction sentence
    by matching it against compiled instruction regex patterns.
    """
    if not line:
        return False

    line = line.strip()

    for pattern in COMPILED_INSTRUCTION_PATTERNS:
        if pattern.fullmatch(line):
            return True

    return False


def extract_request_from_prompt(prompt: str) -> str:
    """
    Extract the main request from a prompt.

    Prompt structure:
        [REQUEST]
        [INSTRUCTION]
        [INSTRUCTION]

    Returns only the REQUEST part (joined into a single line).
    """
    if not prompt:
        return ""

    lines = prompt.splitlines()
    request_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if is_instruction_sentence(stripped):
            break

        request_lines.append(stripped)

    return " ".join(request_lines).strip()


def extract_text_for_ending_check(response: str) -> str:
    """
    Extracts text content to be checked for ending-phrase rules.

    - If response is valid JSON and contains string values,
      concatenates them.
    - Otherwise returns the raw response.

    This function does NOT normalize or lowercase.
    """
    text = response.rstrip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            parts = [v for v in obj.values() if isinstance(v, str)]
            if parts:
                return "\n".join(parts).rstrip()
    except Exception:
        pass

    return text


def verify_0(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    0: Yanıtında ={keyword1}= ve ={keyword2}= anahtar kelimelerini kullan.
    """
    keyword1: str = kwargs["keyword1"]
    keyword2: str = kwargs["keyword2"]

    has_kw1 = contains_word(response, keyword1)
    has_kw2 = contains_word(response, keyword2)

    passed = has_kw1 and has_kw2
    reason = []
    if not has_kw1:
        reason.append(f"'{keyword1}' anahtar kelimesi bulunamadı.")
    if not has_kw2:
        reason.append(f"'{keyword2}' anahtar kelimesi bulunamadı.")
    return _result(0, passed, " ".join(reason))


def verify_1(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    1: Yanıtında "={word}=" kelimesi {num_word} kez geçmelidir.
    """
    word: str = kwargs["word"]
    num_word: int = int(kwargs["num_word"])

    count = count_word_occurrences(response, word)
    passed = count == num_word
    reason = f"Beklenen tekrar sayısı: {num_word}, bulunan: {count}."
    return _result(1, passed, reason, {"count": count})


def verify_2(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    2: Yanıtında ={forbidden_words}= anahtar kelimelerini kullanma.
    """
    forbidden_words = kwargs["forbidden_words"]
    if isinstance(forbidden_words, str):
        forbidden_words = [w for w in re.split(r"[,\s-]+", forbidden_words) if w]
    violations = [
        w for w in forbidden_words if contains_word(response, w)
    ]
    passed = not violations
    reason = ""
    if violations:
        reason = "Kullanılmaması gereken anahtar kelimeler bulundu: " + ", ".join(
            violations
        )
    return _result(2, passed, reason, {"violations": violations})


def verify_3(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    3: Yanıtında ={letter}= harfi {num_letter} adet bulunmalıdır.
       Harf sayımı büyük/küçük harf duyarsızdır.
    """
    letter: str = kwargs["letter"]
    num_letter: int = int(kwargs["num_letter"])
    count = turkish_lower(response).count(turkish_lower(letter)) if letter else 0
    passed = count == num_letter
    reason = f"Beklenen harf sayısı: {num_letter}, bulunan: {count}."
    return _result(3, passed, reason, {"count": count})


def verify_4(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    4: Yanıtın {num_paragraph} paragraftan oluşmalıdır.
       Paragrafları ayırmak için markdown bölücü *** kullan.
    """
    num_paragraph = int(kwargs["num_paragraph"])
    paragraphs = split_paragraphs_by_triple_asterisk(response)
    count = len(paragraphs)
    passed = count == num_paragraph
    reason = f"Beklenen paragraf sayısı: {num_paragraph}, bulunan: {count}."
    return _result(4, passed, reason, {"paragraphs": paragraphs, "count": count})


def verify_5(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    5: Yanıtın en az {min_word} kelimeden oluşmalıdır.
    """
    min_word = int(kwargs["min_word"])
    count = count_words(response)
    passed = count >= min_word
    reason = f"Minimum kelime sayısı: {min_word}, bulunan: {count}."
    return _result(5, passed, reason, {"count": count})


def verify_6(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    6: Yanıtın en fazla {max_word} kelimeden oluşmalıdır.
    """
    max_word = int(kwargs["max_word"])
    count = count_words(response)
    passed = count <= max_word
    reason = f"Maksimum kelime sayısı: {max_word}, bulunan: {count}."
    return _result(6, passed, reason, {"count": count})


def verify_7(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    7: Yanıtın en az {min_sentence} cümleden oluşmalıdır.
    """
    min_sentence = int(kwargs["min_sentence"])
    count = count_sentences(response)
    passed = count >= min_sentence
    reason = f"Minimum cümle sayısı: {min_sentence}, bulunan: {count}."
    return _result(7, passed, reason, {"count": count})


def verify_8(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    8: Yanıtın en fazla {max_sentence} cümleden oluşmalıdır.
    """
    max_sentence = int(kwargs["max_sentence"])
    count = count_sentences(response)
    passed = count <= max_sentence
    reason = f"Maksimum cümle sayısı: {max_sentence}, bulunan: {count}."
    return _result(8, passed, reason, {"count": count})


def verify_9(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    9: Yanıtında {num_paragraph} paragraf olmalıdır.
       Paragraflar yalnızca iki satır aralığıyla birbirinden ayrılmalıdır.
       {rank_paragraph}. paragraf ={first_word}= kelimesiyle başlamalıdır.
       ={first_word}= normalize edilir.
    """
    num_paragraph = int(kwargs["num_paragraph"])
    rank_paragraph = int(kwargs["rank_paragraph"])
    first_word = kwargs["first_word"]

    paragraphs = split_paragraphs_by_double_newline(response)
    count = len(paragraphs)
    ok_count = count == num_paragraph

    ok_start = False
    actual_first = None
    if 1 <= rank_paragraph <= count:
        target_para = paragraphs[rank_paragraph - 1].lstrip()
        if target_para:
            actual_first = normalize(target_para.split()[0])
            first_word = normalize(first_word)
            ok_start = actual_first == first_word

    passed = ok_count and ok_start
    reasons = []
    if not ok_count:
        reasons.append(f"Beklenen paragraf sayısı: {num_paragraph}, bulunan: {count}.")
    if not ok_start:
        reasons.append(
            f"{rank_paragraph}. paragrafın ilk kelimesi '{first_word}' olmalı, "
            f"bulunan: '{actual_first}'."
        )
    return _result(
        9,
        passed,
        " ".join(reasons),
        {"paragraphs": paragraphs, "count": count, "actual_first": actual_first},
    )

def verify_10(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    10: Yanıtının sonunda, {postscript_marker} ile başlayan bir dipnot ekle.
        Dipnotun son satırda başlaması beklenir.
    """
    marker: str = kwargs["postscript_marker"]
    stripped = response.rstrip()
    last_newline = stripped.rfind("\n")
    last_line = stripped if last_newline == -1 else stripped[last_newline + 1 :]
    last_line_stripped = last_line.lstrip()
    
    norm_marker = normalize(marker)
    norm_last_line = normalize(last_line_stripped)

    passed = norm_last_line.startswith(norm_marker)
    reason = ""
    if not passed:
        reason = (
            f"Son satır '{marker}' ile başlamıyor: '{last_line_stripped}'."
        )
    return _result(10, passed, reason, {"last_line": last_line_stripped})


def verify_11(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    11: Yanıtında [adres] gibi köşeli parantezlerle gösterilen
        en az {num_placeholder} yer tutucu bulunmalıdır.
    """
    num_placeholder = int(kwargs["num_placeholder"])
    count = count_placeholders(response)
    passed = count >= num_placeholder
    reason = (
        f"Minimum yer tutucu sayısı: {num_placeholder}, bulunan: {count}."
    )
    return _result(11, passed, reason, {"count": count})


def verify_12(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    12: Yanıtında tam olarak {num_point} madde işareti bulunmalıdır.
        Şu biçimi kullan: * Bu bir madde.
    """
    num_point = int(kwargs["num_point"])
    count = count_bullets_star(response)
    passed = count == num_point
    reason = f"Beklenen madde sayısı: {num_point}, bulunan: {count}."
    return _result(12, passed, reason, {"count": count})


def verify_13(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    13: Yanıtında <<başlık>> biçiminde çift açılı parantezlerle
        çevrelenmiş bir başlık bulunmalıdır.
    """
    match = re.search(r"<<[^<>]+>>", response)
    passed = match is not None
    reason = ""
    if not passed:
        reason = "Hiç '<<...>>' biçiminde başlık bulunamadı."
    return _result(13, passed, reason, {"match": match.group(0) if match else None})


def verify_14(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    14: Aşağıdaki seçeneklerden biriyle yanıt ver: {options}
        Yanıt, seçeneklerden biriyle bire bir (trimlenmiş) eşleşmelidir.
    """
    options = kwargs["options"]
    if isinstance(options, str):
        try:
            parsed = json.loads(options)
            if isinstance(parsed, list):
                options = parsed
            else:
                options = [options]
        except Exception:
            options = [opt.strip() for opt in options.split("|") if opt.strip()]

    normalized_response = response.strip()
    normalized_options = [str(opt).strip() for opt in options]
    passed = any(opt and opt in normalized_response for opt in normalized_options)
    reason = ""
    if not passed:
        reason = (
            "Yanıt seçeneklerden hiçbiriyle tam olarak eşleşmiyor. "
            f"Seçenekler: {normalized_options}, yanıt: '{normalized_response}'."
        )
    return _result(14, passed, reason, {"options": normalized_options})


def verify_15(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    15: Yanıtında markdown kullanarak en az {num_section} bölümünü vurgula,
        örneğin *vurgulanmış bölüm* gibi.
    """
    num_section = int(kwargs["num_section"])
    count = count_emphasized_sections(response)
    passed = count >= num_section
    reason = (
        f"Minimum vurgulanmış bölüm sayısı: {num_section}, bulunan: {count}."
    )
    return _result(15, passed, reason, {"count": count})


def verify_16(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    16: Yanıtın belirli sayıda bölüm (section) içermesini doğrular.
    Bölüm başlangıçları, global _SECTION_SPLIT_RE regex’i kullanılarak tespit edilir.
    """
    num_section = int(kwargs["num_section"])
    splitter = re.escape(kwargs["section_splitter"])

    pattern = re.compile(
        _SECTION_SPLIT_RE.pattern.format(splitter=splitter),
        _SECTION_SPLIT_RE.flags,
    )

    matches = list(pattern.finditer(response))

    count = len(matches)
    passed = count == num_section
    reason = f"Beklenen bölüm başlangıcı sayısı: {num_section}, bulunan: {count}."

    return _result(
        16,
        passed,
        reason,
        {
            "count": count,
            "matches": [m.group(0).strip() for m in matches],
            "pattern": pattern.pattern,
        },
    )


def verify_17(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    17: Tüm çıktı JSON formatında olmalıdır.
        Yanıt, geçerli bir JSON olmalıdır.
        Markdown ```json ... ``` fence kabul edilir.
    """
    text = response.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if (
            len(lines) >= 3
            and lines[0].startswith("```")
            and lines[-1].strip() == "```"
        ):
            text = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(text)
        passed = True
        reason = ""
    except Exception as e:
        parsed = None
        passed = False
        reason = f"JSON parse hatası: {e!r}"

    return _result(
        17,
        passed,
        reason,
        {"parsed": parsed},
    )


def verify_18(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    18: Önce isteği aynen tekrarla, ardından yanıtını ver.
        İstek = prompt içindeki ilk instruction cümlesine kadar olan kısım.
        Karşılaştırma normalize edilerek yapılır.
    """

    if prompt is None:
        return _result(
            18,
            False,
            "verify_18 için 'prompt' parametresi zorunludur.",
        )

    expected = extract_request_from_prompt(prompt)
    response_start = response.lstrip()

    norm_expected = normalize(expected)
    norm_response = normalize(response_start)
    passed = norm_response.startswith(norm_expected)

    reason = ""
    if not passed:
        reason = "Yanıt, isteği başta aynen tekrar etmiyor."

    return _result(
        18,
        passed,
        reason,
        {
            "normalized_prompt": norm_expected,
            "normalized_response_start": norm_response[: len(norm_expected) + 30],
        },
    )


def verify_19(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    19: İki farklı yanıt ver.
        Yanıtlar (ve sadece yanıtlar) altı yıldız işareti (******) ile ayrılmalıdır.
        Tam olarak bir adet '******' ayırıcı ve iki boş olmayan, birbirinden farklı parça beklenir.
    """
    parts = [part.strip() for part in response.split("******")]
    non_empty = [part for part in parts if part]
    passed = len(parts) == 2 and len(non_empty) == 2 and non_empty[0] != non_empty[1]
    if passed:
        reason = ""
    elif len(parts) != 2:
        reason = f"Tam olarak bir '******' ayırıcı bekleniyordu, {max(len(parts) - 1, 0)} bulundu."
    elif len(non_empty) != 2:
        reason = "Ayırıcının iki tarafında da boş olmayan bir yanıt bulunmalı."
    else:
        reason = "İki yanıt birbirinden farklı olmalı."
    return _result(19, passed, reason, {"parts": len(parts), "non_empty": len(non_empty)})

def verify_20(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    20: Tüm yanıtın yalnızca büyük harflerle yazılmış Türkçe olmalıdır.
        Yalnızca Türk alfabesindeki büyük harfler, rakamlar, boşluk ve noktalama işaretleri kabul edilir.
    """
    invalid_chars: List[Tuple[int, str]] = []
    
    for idx, ch in enumerate(response):
        if ch.isalpha():
            if ch not in TURKISH_UPPER and ch not in NON_TURKISH_UPPER:
                invalid_chars.append((idx, ch))

    passed = not invalid_chars
    reason = ""
    if not passed:
        preview = ", ".join(f"'{ch}'@{idx}" for idx, ch in invalid_chars[:5])
        reason = (
            "Büyük harf dışında veya Türk alfabesi dışında harfler bulundu: "
            + preview
        )
    return _result(20, passed, reason, {"invalid_chars": invalid_chars})


def verify_21(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    21: Tüm yanıtın yalnızca küçük harflerle yazılmış Türkçe olmalıdır.
        Yalnızca Türk alfabesindeki küçük harfler, rakamlar, boşluk ve noktalama işaretleri kabul edilir.
    """
    invalid_chars: List[Tuple[int, str]] = []
    for idx, ch in enumerate(response):
        if ch.isalpha():
            if ch not in TURKISH_LOWER and ch not in NON_TURKISH_LOWER:
                invalid_chars.append((idx, ch))

    passed = not invalid_chars
    reason = ""
    if not passed:
        preview = ", ".join(f"'{ch}'@{idx}" for idx, ch in invalid_chars[:5])
        reason = (
            "Küçük harf dışında veya Türk alfabesi dışında harfler bulundu: "
            + preview
        )
    return _result(21, passed, reason, {"invalid_chars": invalid_chars})


def verify_22(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    22: Yanıtında tamamen büyük harflerle yazılmış kelimeler
        en az {num_word} kez geçmelidir.
    """
    num_word = int(kwargs["num_word"])
    count = count_all_caps_words(response)
    passed = count >= num_word
    reason = (
        f"Minimum TAMAMEN BÜYÜK kelime sayısı: {num_word}, bulunan: {count}."
    )
    return _result(22, passed, reason, {"count": count})


def verify_23(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    23: Yanıtında tamamen büyük harflerle yazılmış kelimeler
        en fazla {num_word} kez geçmelidir.
    """
    num_word = int(kwargs["num_word"])
    count = count_all_caps_words(response)
    passed = count <= num_word
    reason = (
        f"Maksimum TAMAMEN BÜYÜK kelime sayısı: {num_word}, bulunan: {count}."
    )
    return _result(23, passed, reason, {"count": count})


def verify_24(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    24: Yanıtını tam olarak şu ifadeyle bitir: {end_phrase}.
        Bu ifadeden sonra başka hiçbir kelime gelmemelidir.
        (Büyük/küçük harf farkı yoktur.
         Boşluklar ve sadece noktalama işaretleri tolere edilir.)
        (Büyük/küçük harf farkı yoktur.
         Boşluklar ve sadece noktalama işaretleri tolere edilir.)
    """
    end_phrase: str = kwargs["end_phrase"]

    text = extract_text_for_ending_check(response)

    text_cf = normalize(text)
    phrase_cf = normalize(end_phrase)

    idx = text_cf.rfind(phrase_cf)

    text = extract_text_for_ending_check(response)

    text_cf = normalize(text)
    phrase_cf = normalize(end_phrase)

    idx = text_cf.rfind(phrase_cf)
    if idx == -1:
        return _result(
            24,
            False,
            f"Yanıtta '{end_phrase}' ifadesi bulunamadı.",
            {"tail_after_phrase": None},
        )

    tail = text_cf[idx + len(phrase_cf):]

    has_extra_word = bool(
        re.search(r"[a-z0-9çğıöşü]", tail, flags=re.IGNORECASE)
    )

    passed = not has_extra_word
    reason = ""
    if not passed:
        reason = (
            f"'{end_phrase}' ifadesinden sonra ek kelime bulundu: '{tail.strip()}'"
        )

    return _result(
        24,
        passed,
        reason,
        {"tail_after_phrase": tail},
    )


def verify_25(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    25: Tüm yanıtını çift tırnak içine al.
        Yanıtın trimlenmiş hali çift tırnak ile başlamalı ve bitmelidir.
    """
    stripped = response.strip()
    passed = len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"'
    reason = ""
    if not passed:
        reason = 'Yanıtın başında ve sonunda çift tırnak (") bulunmalıdır.'
    return _result(25, passed, reason, {"stripped": stripped})


def verify_26(response: str, prompt: Optional[str], **kwargs: Any) -> Dict[str, Any]:
    """
    26: Tüm yanıtın boyunca hiçbir virgül kullanma.
        ASCII virgül ',' karakteri bulunmamalıdır.
    """
    has_comma = "," in response
    passed = not has_comma
    reason = ""
    if not passed:
        count = response.count(",")
        reason = f"Yanıtta {count} adet virgül (',') bulundu."
    return _result(26, passed, reason, {"has_comma": has_comma})


VERIFIERS: Dict[int, Callable[..., Dict[str, Any]]] = {
    0: verify_0,
    1: verify_1,
    2: verify_2,
    3: verify_3,
    4: verify_4,
    5: verify_5,
    6: verify_6,
    7: verify_7,
    8: verify_8,
    9: verify_9,
    10: verify_10,
    11: verify_11,
    12: verify_12,
    13: verify_13,
    14: verify_14,
    15: verify_15,
    16: verify_16,
    17: verify_17,
    18: verify_18,
    19: verify_19,
    20: verify_20,
    21: verify_21,
    22: verify_22,
    23: verify_23,
    24: verify_24,
    25: verify_25,
    26: verify_26,
}


def verify_instruction(
    instruction_id: int,
    response: str,
    prompt: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run a single verifier by instruction_id.

    Parameters
    ----------
    instruction_id : int
        ID from the instruction list (0..26).
    response : str
        Model's raw response text.
    prompt : str, optional
        Original prompt text. Needed for kural 18.
    **kwargs : Any
        Parameters specific to the given instruction, e.g. 'min_word',
        'end_phrase', etc.

    Returns
    -------
    dict
        Result dictionary with keys: instruction_id, passed, reason, details.
    """
    if instruction_id not in VERIFIERS:
        return _result(
            instruction_id,
            False,
            f"Bilinmeyen instruction_id: {instruction_id}",
        )
    verifier = VERIFIERS[instruction_id]
    return verifier(response, prompt, **kwargs)


def verify_task(
    response: str,
    prompt: str,
    instruction_ids: List[int],
    kwargs_per_instruction: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convenience helper for structures of the form:

        {
            "task_id": ...,
            "prompt": "...",
            "instruction_ids": [25, 24],
            "kwargs": {
                25: {},
                24: {"end_phrase": "İyi günler dilerim"}
            }
        }

    Parameters
    ----------
    response : str
        Model's response to be checked.
    prompt : str
        Original prompt text.
    instruction_ids : list[int]
        List of instruction IDs to verify.
    kwargs_per_instruction : dict[int, dict]
        Per-instruction kwargs (may be empty dict).

    Returns
    -------
    list[dict]
        A list of verifier result dicts in the same order as instruction_ids.
    """
    results: List[Dict[str, Any]] = []
    for inst_id in instruction_ids:
        inst_kwargs = kwargs_per_instruction.get(str(inst_id), {}) or {}
        res = verify_instruction(inst_id, response, prompt, **inst_kwargs)
        results.append(res)
    return results
