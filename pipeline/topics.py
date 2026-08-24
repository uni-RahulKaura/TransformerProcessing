"""Stage 6 -- topic labels, chosen from a vocabulary the corpus itself produced.

WHY THIS EXISTS. The five facet labels in categories.py are fast, faithful and too lax to filter
with. Measured over 3,157 sections of the 33-document corpus:

    document type   PAY   EXPIRY  PARTY   PERM   OBLIG
    contracts       19%   21%     61%     40%    58%
    legal_data      16%   19%     59%     38%    52%
    tesla_data      35%    9%      1%     10%     3%     <- effectively dead on a financial report
    bills           63%   36%     22%     11%     7%

PARTY fires on 61% of contract sections and OBLIG on 58%: a filter that matches half a document
filters nothing. 28.4% of sections get no label at all. The commonest labelled combination is
PARTY+PERM+OBLIG -- "mentions a party and an obligation" -- true of nearly any clause.

The facets are NOT removed. They are cheap and they never invent. This is additive.

WHERE THE VOCABULARY CAME FROM. 68,162 section titles harvested from the instrument's own outlines,
50 independently written human answer keys, and the generated corpora; 6,232 distinct after
normalisation; embedded and clustered into 54 topics. Every topic traces to observed titles with
document-frequency evidence, which is why the list contains taxes_and_surcharges,
utility_service_and_metering and credit_facility_and_advances -- this corpus holds utility bills and
credit agreements -- and omits plenty of textbook contract topics the corpus never shows.

PER DOCUMENT TYPE, which is the point. Each topic records the document types it was observed in, so
a bill is scored against the 13 topics that occur in statements rather than all 54. That is both
cheaper and sharper: cross-type candidates are the ones that fire spuriously. Counts:

    agreements 38    correspondence 29    statements 13    reports 12    regulation 10

NO MODEL DOWNLOAD. facebook/bart-large-mnli is used zero-shot: each topic becomes the hypothesis
"This section is about <topic>." and is scored independently, so this is multi-label by
construction and a section may carry none.
"""
import json
import re
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "facebook/bart-large-mnli"
THRESH = float(os.environ.get("TOPIC_MIN", "0.55"))
TOPN = int(os.environ.get("TOPIC_TOPN", "3"))
DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
# The corpus folder a document sits in, mapped to the vocabulary's own document types.
FOLDER_TYPE = {"bills": "statements", "tesla_data": "reports",
               "contracts": "agreements", "legal_data": "agreements"}
_M = {}


STOP = set("""a an the and or of to in on at for by with from this that these those it its
which who whom whose what when where how why be is are was were been being am has have had do
does did shall will would may might must can could should not no other another any all such
their his her our your they he she we you i as so than then under over both either each every
some same own more less least most much many well ever yet still even given others end""".split())

# NOT stopwords, even though they look like furniture: "parties", "document", "section",
# "clause" and "used" are the ONLY anchors several topics have. Listing them here dropped a
# correct 0.86 parties_and_roles label off a section titled "PARTIES", because the gate had
# thrown away the one word that could have grounded it.
_ANCH = {}


def anchors(rec):
    """Content words that must appear in a section before its topic may be published.

    Zero-shot entailment has no lexical grounding: a section that only names two companies and
    their addresses scored 0.68 for "a loan or credit facility: advances, lenders, agents and
    repayment" in an NDA, because that topic's cluster absorbed the generic titles "the agent"
    and "subsidiaries" alongside the real lending ones, and MNLI found the overlap plausible.

    The gate is a floor, not a classifier: at least one of the topic's own words has to be in
    the section. It cannot fix a topic being wrong, but it stops one being published about
    vocabulary the section never uses -- which is what made the low-confidence band unreadable.
    """
    k = rec["topic"]
    if k in _ANCH:
        return _ANCH[k]
    # ONLY the curated fields. The cluster residue in evidence.source_clusters and
    # evidence.top_titles is exactly what polluted this topic: including it gave
    # credit_facility_and_advances the anchors "notice", "tenant", "default" and "types", which
    # match most clauses in any agreement and made the gate useless where it was needed most.
    words = set()
    for src in [rec.get("label") or "", rec.get("label_short") or "", k.replace("_", " ")]:
        for w in _words(str(src)):
            if w not in STOP:
                words.add(_stem(w))
    _ANCH[k] = words
    return words


def _stem(w):
    """Shallow, so "disclosing" in a section satisfies the anchor "disclose"."""
    w = w.lower()
    for suf in ("ements", "ement", "ances", "ance", "ences", "ence", "ions", "ion",
                "ings", "ing", "ures", "ure", "ies", "ed", "es", "s", "e"):
        # A one-letter suffix needs a shorter stem to be worth stripping: "calls" -> "call" is the
        # whole point, and the blanket 5-character floor blocked it, so a section saying "each call"
        # failed the anchor "calls" and its topic was suppressed.
        floor = 3 if len(suf) == 1 else 5
        if w.endswith(suf) and len(w) - len(suf) >= floor:
            return w[:-len(suf)]
    return w


def _words(s):
    """Tokens, with hyphenated compounds contributing their parts as well as the whole.

    Without the split, the preamble "THIS NON-DISCLOSURE AGREEMENT is entered into ..." failed
    the anchor for the confidentiality topic, because "non-disclosure" is one token and does not
    stem to "disclos". The whole is kept too, so a genuinely compound term still matches.
    """
    out = []
    for w in re.findall(r"[A-Za-z][A-Za-z-]{2,}", s.lower()):
        out.append(w.strip("-"))
        if "-" in w:
            out.extend(p for p in w.split("-") if len(p) > 2)
    return out


def grounded(rec, text):
    """Does the section use any of this topic's own vocabulary?"""
    seen = {_stem(w) for w in _words(text)}
    return bool(anchors(rec) & seen)


def load(path="topics.json"):
    if "v" in _M:
        return _M["v"], _M["t"], _M["m"]
    d = json.load(open(path))
    recs = [r for r in (d.get("topics") or d) if isinstance(r, dict) and r.get("topic")]
    tok = AutoTokenizer.from_pretrained(REPO)
    dt = torch.float16 if DEVICE == "cuda" else torch.float32
    mdl = AutoModelForSequenceClassification.from_pretrained(REPO, dtype=dt).to(DEVICE).eval()
    _M.update(v=recs, t=tok, m=mdl)
    return recs, tok, mdl


def candidates(recs, doctype):
    """The topics observed in this document type, or all of them if the type is unknown."""
    sub = [r for r in recs if doctype in (r.get("doctypes") or [])]
    return sub or recs


def label(text, doctype, path="topics.json", title=""):
    """Return [(topic, probability)] above threshold, best first. Never raises on odd input.

    All of a document type's hypotheses go through the model in ONE batch. Scored one at a time this
    was 38 forward passes per section for an agreement -- about 5 s/section on a T4, which is 3.5
    hours for the corpus and most of it spent re-encoding the same section text. Batched it is one
    pass, and the section is encoded once.
    """
    if not text or not text.strip():
        return []
    recs, tok, mdl = load(path)
    cands = candidates(recs, doctype)
    if not cands:
        return []
    ent = mdl.config.label2id.get("entailment", 2)
    con = mdl.config.label2id.get("contradiction", 0)
    passage = text[:1600]
    hyps = [(r["topic"], "This section is about %s." % (r.get("label") or r["topic"].replace("_", " ")))
            for r in cands]
    out = []
    # a modest batch so a long section against 38 hypotheses cannot exhaust a 15 GB card
    step = int(os.environ.get("TOPIC_BATCH", "16"))
    for i in range(0, len(hyps), step):
        chunk = hyps[i:i + step]
        enc = tok([passage] * len(chunk), [h for _, h in chunk],
                  return_tensors="pt", truncation=True, max_length=512,
                  padding=True).to(DEVICE)
        with torch.inference_mode():
            lg = mdl(**enc).logits
        pr = torch.softmax(torch.stack([lg[:, con], lg[:, ent]], dim=-1), dim=-1)[:, 1]
        for (tid, _), p in zip(chunk, pr.tolist()):
            if p >= THRESH:
                out.append((tid, round(p, 3)))
    out.sort(key=lambda x: -x[1])
    if os.environ.get("TOPIC_GATE", "1") == "1":
        # The TITLE counts as the section's own vocabulary. A bare container heading has no body
        # to ground against, but "3 RESTRICTIONS ON USE AND DISCLOSURE" states its subject in
        # the title, and dropping it on a word count threw away a correct 1.00 label. A heading
        # that names nothing -- "10 MISCELLANEOUS" -- still fails the gate, which is the result
        # the word count was reaching for.
        by_id = {r["topic"]: r for r in cands}
        ground = (title or "") + " " + passage
        out = [(t, p) for t, p in out if grounded(by_id[t], ground)]
    return out[:TOPN]
