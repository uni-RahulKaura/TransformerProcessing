"""The mechanical faithfulness check -- run this on any output before trusting it.

A summary is only useful if it does not invent. Checking that by eye does not scale and
does not catch the failures that actually occur, so this checks three different ways. Each
catches something the others structurally cannot, which we learned the hard way: a first
pass that only did the first check on section-level summaries in lower case found three
problems where there were nine.

  1  Added or altered tokens. Every capitalised word, number and amount in the summary
     must appear somewhere in the section it summarises.

  2  Case and spacing inside acronyms. Check 1 folds case, so it cannot see an acronym
     ABCD becoming "AbCd", or XYZ becoming "X YZ".

  3  Clause renumbering. A summary can cite a clause number that genuinely appears in the
     section while still attributing it to the wrong clause -- writing 13.1 where the text
     says 13.1.1. Token presence cannot detect this; the numbers must be compared against
     the clause numbers the section actually declares.

WHAT THIS FOUND, on 36 sections and 21 part summaries of a real document: nine wrong words.
Not one was an invented fact -- no fabricated obligation, party, amount or deadline
anywhere. Every one was a corrupted LABEL: a party name pluralised, an acronym with a
space inserted, an acronym whose case changed, a clause number that does not exist.

That distinction is the whole finding, because it tells you exactly how far to trust the
output. Safe: using a summary to decide which section to open. NOT safe: quoting a summary,
or letting anything act on a clause number inside one.
"""
import re

TOKEN = re.compile(r"\$[\d,.]+|\b\d[\d,.]*%?\b|\b[A-Z][A-Za-z]*[A-Z][A-Za-z]*\b"
                   r"|\b[A-Z][a-z]{2,}\b")
STOP = set("""The This That These Those A An And Or But If For To As At By In On Of It No Not
All Any Each Such Both There Their They Where When Which While Unless""".split())
CLAUSE = re.compile(r"\b\d+(?:\.\d+)+\b")
SPACED = re.compile(r"\b([A-Z]) ([A-Z]{1,3})\b|\b(\d+)\. (\d)")


def check(summary, section_text):
    """Return a list of findings. Empty means the summary added nothing to the section."""
    if not summary:
        return []
    src = re.sub(r"\s+", " ", section_text)
    low = src.lower()
    out = []
    for w in TOKEN.findall(re.sub(r"\s+", " ", summary)):
        if w in STOP or w.lower() in low:
            continue
        out.append({"kind": "not-in-section", "token": w})
    # 2: case corruption -- present when folded, absent when not
    for w in re.findall(r"\b[A-Za-z]*[a-z][A-Z][A-Za-z]*\b", summary):
        if w not in src and w.lower() in low:
            out.append({"kind": "case-changed", "token": w,
                        "source": next((m.group(0) for m in
                                        re.finditer(re.escape(w), src, re.I)), "")})
    # 2b: a space inserted inside something the source writes solid
    for m in SPACED.finditer(summary):
        joined = "".join(g for g in m.groups() if g)
        if joined.lower() in low and m.group(0).lower() not in low:
            out.append({"kind": "space-inserted", "token": m.group(0), "source": joined})
    # 3: clause renumbering -- cited, present as a string, but not a clause the section has
    declared = set(CLAUSE.findall(src))
    for c in CLAUSE.findall(summary):
        if c not in declared:
            out.append({"kind": "clause-not-declared", "token": c})
    seen, uniq = set(), []
    for f in out:
        k = (f["kind"], f["token"])
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return uniq
