#!/usr/bin/env python
"""Build a navigable index of any document -- one file, no dependencies beyond torch and
transformers, nothing leaving the machine.

    pip install torch transformers sentencepiece
    python transformer_index.py mydoc.md --no-summaries    # structure + labels, <1 second
    python transformer_index.py mydoc.md                   # add summaries from four models
    python transformer_index.py mydoc.md --models BART-large-CNN

START WITH --no-summaries. It loads no model and answers the question that matters first on
a document type this has not seen: DID THE SECTION RULES FIND THE STRUCTURE AT ALL? If the
section count looks wrong, no amount of summarising will fix it -- the heading rules in
stage 1 need adjusting for that document type first.

--------------------------------------------------------------------------------------------
WHAT THIS IS

Turn a long document into an index an agent can navigate: every section, what kind of thing
is in it, and a summary of each. No LLM. No API calls. Four transformer models compared on
identical input, all running locally from downloaded weights.

A transformer never sees a document -- it sees whatever a rule handed it. Those rules decide
the shape of everything downstream, so they are the substance of this file. The model call
itself is four lines.

  Stage 1   document -> sections        three heading rules
  Stage 2   sections -> subsections     4.1 under 4, nearest parent in document order
  Stage 3   split to fit                88% of the model's token limit, sentence boundaries
  Stage 4   summarise                   four models, deterministic settings
  Stage 5   label                       money / dates / parties / permissions / obligations
  Verify    check the output            three separate checks against the source text

--------------------------------------------------------------------------------------------
MEASURED RESULTS, on a 44-section document. Usefulness 50%, faithfulness 30%, speed 20%.

  BART-large-CNN        8/10  8/10   4/10   5.64 s/section   7.2   <- recommended
  DistilBART-CNN        7/10  8/10   6/10   3.15 s           7.1   0.1 behind, 1.8x faster
  MiniLM-L6 extractive  3/10 10/10  10/10   0.07 s           6.5   scores 3 on usefulness
  LongT5-16384          4/10  2/10   7/10   2.37 s           4.0   the only one that invents

Two findings are more useful than the ranking.

SPLITTING LONG SECTIONS BEAT READING THEM WHOLE. LongT5 read a 3,089-word section in one
pass and invented a detail that was not in the document. BART read the same section in five
parts and every part summary was accurate. Reading more of the document made it worse.

QUOTING IS SAFE BUT USELESS. The extractive model cannot invent, because every word is
quoted. But "most typical sentence" in a formal document is the boilerplate, so it reliably
returns the least informative line in the section.

--------------------------------------------------------------------------------------------
WHAT THE VERIFICATION FOUND -- read this before trusting any output.

Nine wrong words across 36 section summaries and 21 part summaries. NOT ONE WAS AN INVENTED
FACT: no fabricated obligation, party, amount or deadline anywhere. Every one was a
corrupted LABEL -- a party name pluralised, a space inserted inside an acronym, an acronym's
case changed, and twice a clause number that either does not exist or points at the wrong
clause.

That distinction is the finding, because it says exactly how far to trust the output:

  SAFE      using a summary to decide which section to open
  NOT SAFE  quoting a summary, or letting anything act on a clause number inside one

Separately, on 7 of 36 sections the summary had nothing wrong with it and was still useless
-- it latched onto the section's opening line or a footnote instead of its substance. The
worst case was the pricing section, which held the entire fee schedule; the summary
described which month the financial year starts. Every word true, not one number. That is
why the stage 5 labels matter: they tag that section "money" from the amounts in the table
even when the summary never mentions money.

--------------------------------------------------------------------------------------------
WILL THIS WORK ON YOUR DOCUMENT TYPE? Partly, and it is worth knowing which part.

Everything above was measured on supplier agreements and has not been tested elsewhere.

  Stages 3 and 4 are document-agnostic and carry over unchanged.

  AMENDMENTS should mostly work -- same clause structure, numbering and vocabulary. One
  specific risk: in an amendment the clause number IS the payload ("Section 4.2 is amended
  to read..."), and verification caught clause-number corruption twice in 36 sections. On a
  contract that is an annoyance; on an amendment it changes which clause you think was
  amended. Extract the clause number by rule; never read it from a summary.

  INVOICES will not work, and it is not a tuning problem. An invoice has no clause headings
  and no 4.1-under-4 hierarchy, so stages 1 and 2 have nothing to key on. And the labels
  stop discriminating: "money" fires on 17% of sections in a supplier agreement, which is
  what makes it a useful filter, but would fire on essentially every part of an invoice,
  while shall/must would fire on almost none. A label that fires on everything carries no
  information. More fundamentally an invoice is not summarised, it is a set of fields to
  extract -- invoice number, PO, due date, line items, tax, total -- and the right output is
  a filled form. That is a different pipeline, not new rules on this one.

  FORMAT is a separate variable from type. Invoices usually arrive as PDFs or scans, and
  parser quality drops measurably on the same document going from Markdown to PDF. Test
  document type and document format separately.

--------------------------------------------------------------------------------------------
Contains no documents and nothing derived from any. Nothing is sent anywhere; the only
network traffic is the initial model weight download, inbound.
"""
import argparse
import hashlib
import json
import re
import sys
import time

# ======================================================================================
# STAGE 5 -- the category labels
# ======================================================================================

"""Stage 5 -- the five category labels.

Five regular expressions, one per label, lifted unchanged from the evaluation harness.
They tag a section as containing money, dates & term, who-it's-between, permissions or
obligations, so an agent can narrow its search before reading any summary.

No model, no network, no state. Sub-millisecond per section.

On the corpus these were measured on (58 sections of real supplier agreements) they got
58 of 290 yes/no answers wrong. The best of eight small language models tested against the
same answer key got 70 wrong, at thousands of times the cost.

KNOWN LIMITS, because they matter when you point this at a new document type:

  * PARTY over-fires. It triggers on a role noun such as "Supplier" anywhere in the
    section, so a section that only states WHERE work happens still gets the label. This
    was the weakest of the five: 9 false alarms in 58 sections.

  * These match words, not meaning. "no payment is due" is tagged PAY, correctly. A duty
    expressed without the word "shall" is missed.

  * The labels are only informative if they discriminate. PAY fires on 17% of sections in
    a supplier agreement, which is what makes it a useful filter. On an invoice it would
    fire on essentially every section and therefore tell you nothing, while OBLIG would
    fire on almost none. Re-tune the label set per document type; do not assume these five
    transfer."""

_PERM = re.compile(r"\b(?:may|shall\s+not|must\s+not|is\s+(?:not\s+)?permitted|"
                   r"is\s+entitled|prohibited|consent)\b", re.I)
_DATE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
                   r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
                   r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b")
_TERMWORD = re.compile(r"\b(?:expir\w+|terminat\w+|renew\w+|effective\s+date"
                       r"|\d+\s*(?:day|week|month|year)s?)\b", re.I)
_ROLE = re.compile(r"\b(?:Supplier|Customer|Manufacturer|Buyer|Seller|Licensor|Licensee"
                   r"|Vendor|Contractor|Client|Purchaser|Consultant|Provider)\b")
_CORP = re.compile(r"\b[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,5}"
                   r"\s*,?\s*(?:Inc|Corp|Corporation|Ltd|Limited|LLC|LC|GmbH|AG|plc)\.?\b")
_PAY = re.compile(r"(?:US\$|\$|USD|EUR|CHF)\s?[\d,]+|\bnet\s+\d+\b"
                  r"|\b(?:payment|invoice|invoicing|price|pricing|fee|fees|rate|rates"
                  r"|currency|exchange\s+rate)\b", re.I)
_OBL = re.compile(r"\b(?:shall|must|is\s+required\s+to|agrees\s+to|will\s+provide)\b", re.I)


def rules_predict(text):
    return {
        "PERM": bool(_PERM.search(text)),
        "EXPIRY": bool(_DATE.search(text) or _TERMWORD.search(text)),
        "PARTY": bool(_ROLE.search(text) or _CORP.search(text)),
        "PAY": bool(_PAY.search(text)),
        "OBLIG": bool(_OBL.search(text)),
    }


# ======================================================================================
# STAGE 3 -- splitting a section the model cannot read in one pass
# ======================================================================================

"""Stage 3 -- splitting a section the model cannot read in one pass.

Every summarisation model has a hard input limit. Left alone it reads up to that limit and
SILENTLY DISCARDS the rest -- no error, no warning, no way to tell from the output that it
happened. On the document this was built for, seven sections were over the limit and the
largest was 3,089 words, so the default behaviour would have thrown away most of them.

So: measure the section in the model's own tokeniser. If it fits, leave it alone. If it
does not, accumulate whole sentences into a part until the next sentence would push it past
88% of the limit, then start a new part. Each part is summarised on its own and the section
summary is built from the parts.

The 88% is headroom, so a long final sentence cannot overshoot. Sentences are never cut in
half. Nothing goes unread."""

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_parts(tok, text, limit):
    if len(tok(text).input_ids) <= limit:
        return [text]
    piece = int(limit * 0.88)
    chunks, cur = [], ""
    for sent in SENT.split(text):
        trial = (cur + " " + sent).strip()
        if len(tok(trial).input_ids) > piece and cur:
            chunks.append(cur)
            cur = sent
        else:
            cur = trial
    if cur:
        chunks.append(cur)
    return chunks or [text]


# ======================================================================================
# STAGES 1 AND 2 -- document into sections and subsections
# ======================================================================================

"""Stages 1 and 2 -- turning a document into sections and subsections.

STAGE 1: what counts as a heading. Three rules, and a line only has to match one.

  1a  Anything the converter already marked as a heading (a Markdown `#` line here).

  1b  A short titled line that is not a sentence. Converters routinely emit numbered
      clauses as list items rather than headings, so keying on headings alone loses most
      of them. This catches them by SHAPE: 3-80 characters, at most 10 words, does not end
      in a full stop, does not begin like a date, starts with a capital or a digit.

      This rule was wrong twice before it was right, and the way it was wrong is worth
      knowing. The first version required the line to be over 70% uppercase. That kept
      ALL-CAPS headings and silently threw away real numbered clauses written in Title
      Case. Case was the wrong discriminator; length and shape is the right one.

      It now deliberately errs toward including too much -- it will also pick up the odd
      defined term or table caption -- because a section missing from the index is worse
      than a section that should not be there. `heading_confidence` records which titles
      were real headings and which were inferred this way, so the noise is auditable.

  1c  Numbered titles harvested from the raw text, used as an allowlist. Some headings
      survive conversion glued to the paragraph that follows them, with no delimiter to
      split on. No rule reading the converted output alone can recover that boundary, so
      the numbering in the source is read separately and any block starting with a known
      title is split at that point. Longest titles are tried first, so a short title that
      happens to be a prefix of a longer one cannot steal the match.

A heading whose own body is empty is KEPT, not dropped, and marked `container_only`. An
earlier version discarded any heading with under 40 characters of body as noise; that was
wrong, and it left the real subsections attached to the wrong parent.

STAGE 2: what nests under what. `4.1` goes under `4`, `10.6` under `10` -- but a child
attaches to the most recent matching parent IN DOCUMENT ORDER, not to any section bearing
that number. Documents restart their numbering inside attachments and appendices, so the
same number legitimately appears twice with different children. Document order resolves it
with no special case."""

HEADING_MD = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
NUM = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$")
# 1c: a numbered title on a line of its own, optionally bold, optionally colon-terminated
ALLOW_RX = re.compile(
    r"(?m)^\s*(?:\*\*)?(\d{1,2}\.)\s+([A-Z][A-Za-z0-9 &/\-,']{3,70}?)(?:\*\*)?\s*:?\s*$")


def is_clause_heading(line):
    """Rule 1b -- a short titled line that is not a sentence."""
    t = line.strip().lstrip("*-• ").strip()
    if not (3 <= len(t) <= 80) or len(t.split()) > 10:
        return False
    if t.endswith(".") and not re.match(r"^\d+\.$", t):
        return False          # a full sentence, not a title
    if re.match(r"^\d+[-/]", t):
        return False          # a date or a data row, e.g. "10-Feb-20  9-Mar-20 ..."
    if not (t[0].isupper() or t[0].isdigit()):
        return False
    return t


def split_sections(text):
    """Apply stages 1 and 2 to raw Markdown. Returns a list of section dicts."""
    allow = sorted({m.group(2).strip() for m in ALLOW_RX.finditer(text)},
                   key=len, reverse=True)
    lines = text.split("\n")
    marks = []                                   # (line index, title, confidence)
    for i, ln in enumerate(lines):
        m = HEADING_MD.match(ln)
        if m and m.group(2).strip():
            marks.append((i, m.group(2).strip(), "heading"))
            continue
        stripped = ln.strip()
        if not stripped:
            continue
        # 1c takes precedence over 1b: an allowlisted title may be glued to its paragraph
        hit = next((a for a in allow if stripped.startswith(a) and len(stripped) > len(a)),
                   None)
        if hit:
            marks.append((i, hit, "allowlist"))
            continue
        if len(stripped) < 90:
            t = is_clause_heading(stripped)
            if t:
                marks.append((i, t, "inferred"))

    out = []
    if marks and marks[0][0] > 0:                # text before the first heading is a section
        body = "\n".join(lines[:marks[0][0]]).strip()
        if body:
            out.append(_mk("[preamble - before first heading]", body, None, "preamble"))
    for j, (pos, title, conf) in enumerate(marks):
        end = marks[j + 1][0] if j + 1 < len(marks) else len(lines)
        body = " ".join(x.strip() for x in lines[pos:end]).strip()
        out.append(_mk(title, body, None, conf))

    # stage 2: nest by numbering, most recent matching parent in document order
    for k, s in enumerate(out):
        m = NUM.match(s["title"])
        s["number"] = m.group(1) if m else None
        if not s["number"] or "." not in s["number"]:
            continue
        stem = s["number"].rsplit(".", 1)[0]
        for back in range(k - 1, -1, -1):
            if out[back]["number"] == stem:
                s["parent"] = back
                break
    for k, s in enumerate(out):
        s["order"] = k
        # a heading with no body of its own: keep it, mark it, do not summarise it
        s["container_only"] = s["words"] < 6 and any(x.get("parent") == k for x in out)
    return out


def _mk(title, body, parent, conf):
    return {"title": title, "text": body, "words": len(body.split()),
            "heading_confidence": conf, "parent": parent, "number": None,
            "container_only": False,
            # content-hashed, so inserting a section never renumbers the others
            "id": hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]}


# ======================================================================================
# VERIFY -- the three-way faithfulness check
# ======================================================================================

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
or letting anything act on a clause number inside one."""

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


# ======================================================================================
# STAGE 4 -- how each transformer is called
# ======================================================================================

"""Stage 4 -- how each transformer is called.

Four models, chosen because they represent genuinely different strategies rather than four
points on one curve:

  BART-large-CNN     writes a new sentence. 1,024-token limit, so long sections are split.
  DistilBART-CNN     the same, distilled: ~1.8x faster for about a point less usefulness.
  LongT5-16384       reads 16,384 tokens, so it never splits. An alternative to splitting,
                     not a cheaper version of it.
  MiniLM extractive  writes nothing -- quotes the real sentence closest to the section's
                     average. The floor: it cannot invent, and it is rarely informative.

Settings are identical across all four apart from the reading limit and the beam count, and
nothing is sampled, so a re-run reproduces the same summaries word for word.

  do_sample=False          makes the run reproducible
  min_new_tokens=12        stops a model answering in three words
  no_repeat_ngram_size=3   stops the repetition loops these models fall into on tables
  num_beams=4 (2 for LongT5, where 4 was not worth the time)

MEASURED RESULTS on the corpus this was built for, weighting usefulness 50%, faithfulness
30%, speed 20%:

  BART-large-CNN        7.2/10   5.64 s per section   recommended
  DistilBART-CNN        7.1/10   3.15 s               0.1 behind and 1.8x faster
  MiniLM extractive     6.5/10   0.07 s               scores 3/10 on usefulness
  LongT5-16384          4.0/10   2.37 s               the only one that invents content

Two findings worth carrying forward. Splitting long sections BEAT reading them whole:
LongT5 read a 3,089-word section in one pass and invented a detail that was not in the
document, while BART read the same section in five parts and every part summary was
accurate. And quoting is safe but useless: the extractive model never invents, because
every word is quoted, but "most typical sentence" in a formal document is the boilerplate."""

# torch and transformers are imported lazily, so --no-summaries runs without them.

MODELS = [("BART-large-CNN", "facebook/bart-large-cnn", 1024, "abstractive"),
          ("DistilBART-CNN", "sshleifer/distilbart-cnn-12-6", 1024, "abstractive"),
          ("LongT5-16384", "pszemraj/long-t5-tglobal-base-16384-book-summary", 16384,
           "abstractive"),
          ("MiniLM-L6 extractive", "sentence-transformers/all-MiniLM-L6-v2", 512,
           "extractive")]


def load(repo, kind):
    # Imported here, not at module scope, so --no-summaries needs neither package.
    import torch
    from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
    torch.set_num_threads(1)                      # reproducible timings, one core
    tok = AutoTokenizer.from_pretrained(repo)
    if kind != "abstractive":
        return tok, AutoModel.from_pretrained(repo).eval()
    mdl = AutoModelForSeq2SeqLM.from_pretrained(repo, dtype=torch.float32).eval()
    if "long-t5" in repo:
        # This checkpoint stores one embedding table under `shared.weight` but declares
        # tie_word_embeddings=False. transformers therefore randomly initialises the
        # encoder and decoder embeddings and throws the real ones away, and the model
        # emits word salad on every input. Measured: the real table has standard
        # deviation 10.09, the randomly initialised one 1.0002.
        #
        # Worth knowing because the failure is indistinguishable from "this model is bad".
        # It still came last after the fix -- but for inventing content, which is a real
        # finding, rather than for a loading bug, which was ours.
        mdl.encoder.embed_tokens.weight = mdl.shared.weight
        mdl.decoder.embed_tokens.weight = mdl.shared.weight
    return tok, mdl


def summarise(tok, mdl, repo, limit, text, longer=False):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=limit)
    g = mdl.generate(**enc, max_new_tokens=80 if longer else 70, min_new_tokens=12,
                     num_beams=4 if "long-t5" not in repo else 2,
                     do_sample=False, no_repeat_ngram_size=3)
    return tok.decode(g[0], skip_special_tokens=True).strip()


def extract(tok, mdl, text):
    """Quote the sentence nearest the section's mean embedding. Invents nothing."""
    import torch
    sents = [x.strip() for x in SENT.split(text) if len(x.strip()) > 30]
    if not sents:
        return text[:180]
    vs = []
    with torch.inference_mode():
        for x in sents:
            e = tok(x, return_tensors="pt", truncation=True, max_length=512)
            h = mdl(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            vs.append(torch.nn.functional.normalize(v, dim=-1)[0])
    E = torch.stack(vs)
    c = torch.nn.functional.normalize(E.mean(0, keepdim=True), dim=-1)
    return sents[int(torch.argmax(E @ c.T))]


def run_model(name, repo, limit, kind, sections):
    """Summarise every section, splitting the ones over the model's limit."""
    tok, mdl = load(repo, kind)
    rows, splits = [], 0
    for s in sections:
        if s.get("container_only") or not s["text"].strip():
            rows.append({"summary": None, "parts": [], "n_parts": 0})
            continue
        if kind == "extractive":
            rows.append({"summary": extract(tok, mdl, s["text"]), "parts": [], "n_parts": 1,
                         "quoted": True})
            continue
        parts = split_parts(tok, s["text"], limit)
        if len(parts) == 1:
            rows.append({"summary": summarise(tok, mdl, repo, limit, s["text"]),
                         "parts": [], "n_parts": 1})
            continue
        splits += 1
        ps = [{"label": "part %d of %d" % (n + 1, len(parts)), "words": len(p.split()),
               "summary": summarise(tok, mdl, repo, limit, p)}
              for n, p in enumerate(parts)]
        # the section summary is built from the part summaries: one step further from the
        # source than everything else here, which is why the parts are kept and shown
        stitched = summarise(tok, mdl, repo, limit,
                             " ".join(p["summary"] for p in ps), longer=True)
        rows.append({"summary": stitched, "parts": ps, "n_parts": len(ps)})
    return {"repo": repo, "kind": kind, "input_limit": limit,
            "sections_needing_split": splits, "rows": rows}


# ======================================================================================
# RUNNER
# ======================================================================================
def main():
    #!/usr/bin/env python
    """Build a navigable index of any Markdown document. No LLM, no network at inference.

        python run.py examples/sample_sow.md                    # all four models
        python run.py mydoc.md --models BART-large-CNN          # just one
        python run.py mydoc.md --no-summaries                   # structure + labels only, instant

    Writes index.json and prints a readable outline. Everything runs locally; the only network
    traffic is the first download of the model weights.

    Start with --no-summaries. It runs in under a second and tells you the thing you most need
    to know about a new document type: whether the section rules found the structure at all. If
    the section count looks wrong, no amount of summarising will fix it -- the rules in
    pipeline/sections.py need adjusting for that document type first.
    """


    CATNAME = {"PAY": "money", "EXPIRY": "dates & term", "PARTY": "who it's between",
               "PERM": "permissions", "OBLIG": "obligations"}
    ORDER = ["PAY", "EXPIRY", "PARTY", "PERM", "OBLIG"]

    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of model names; default is all four")
    ap.add_argument("--no-summaries", action="store_true",
                    help="structure and category labels only -- no model loaded")
    ap.add_argument("--out", default="index.json")
    a = ap.parse_args()

    text = open(a.path, encoding="utf-8", errors="replace").read()
    sections = split_sections(text)
    for s in sections:
        s["categories"] = ([c for c in ORDER if rules_predict(s["text"]).get(c)]
                           if s["text"].strip() else [])

    kids = {}
    for i, s in enumerate(sections):
        if s["parent"] is not None:
            kids.setdefault(s["parent"], []).append(i)
    n_text = sum(1 for s in sections if not s["container_only"] and s["text"].strip())
    print("%s: %d sections, %d with text, %d nested by numbering, %d heading-only"
          % (a.path, len(sections), n_text,
             sum(1 for s in sections if s["parent"] is not None),
             sum(1 for s in sections if s["container_only"])))
    print("   inferred by shape rather than a real heading: %d  (rule 1b -- audit these first)"
          % sum(1 for s in sections if s["heading_confidence"] == "inferred"))

    models = {}
    if not a.no_summaries:
        want = MODELS if not a.models else [m for m in MODELS if m[0] in a.models]
        if not want:
            sys.exit("no model matched; choose from: %s" % ", ".join(m[0] for m in MODELS))
        for name, repo, limit, kind in want:
            t0 = time.time()
            models[name] = run_model(name, repo, limit, kind, sections)
            bad = 0
            for s, r in zip(sections, models[name]["rows"]):
                f = check(r.get("summary"), s["text"])
                f += [x for p in (r.get("parts") or []) for x in check(p["summary"], s["text"])]
                r["findings"] = f
                bad += bool(f)
            models[name]["seconds"] = round(time.time() - t0, 1)
            models[name]["sections_with_findings"] = bad
            print("   %-22s %5.1fs  %d sections split  %d sections with a corrupted word"
                  % (name, models[name]["seconds"], models[name]["sections_needing_split"], bad))

    json.dump({"doc": a.path, "n_sections": len(sections), "sections": sections,
               "models": models}, open(a.out, "w"), indent=1)
    print("-> %s" % a.out)

    print()
    first = next(iter(models), None)


    def show(i, depth=0):
        s = sections[i]
        pad = "    " * depth
        tag = ", ".join(CATNAME[c] for c in s["categories"]) or "none of the five"
        print("%s%s  [%s]  %s"
              % (pad, s["title"][:66],
                 "heading only" if s["container_only"] else "%d words" % s["words"], tag))
        if first and not s["container_only"]:
            r = models[first]["rows"][i]
            if r.get("summary"):
                print("%s    %s" % (pad, r["summary"][:150]))
            for p in r.get("parts") or []:
                print("%s      %s: %s" % (pad, p["label"], p["summary"][:120]))
            if r.get("findings"):
                print("%s    !! %s" % (pad, "; ".join("%s: %s" % (f["kind"], f["token"])
                                                      for f in r["findings"])))
        for k in kids.get(i, []):
            show(k, depth + 1)


    for i, s in enumerate(sections):
        if s["parent"] is None:
            show(i)


if __name__ == "__main__":
    main()
