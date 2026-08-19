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
with no special case.
"""
import hashlib
import re

HEADING_MD = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
# Rule 1d: a bold-delimited span at the START of a line, with the body following it on the
# SAME line -- "**3. Compelled Disclosure.** If the Recipient is required by law ...".
# This is the single most common way a heading survives conversion to Markdown, and it
# defeats both of the other rules: 1a needs a `#`, and 1b rejects the line for being a
# sentence. Rule 1c cannot help either, because its allowlist only matches a title that
# ends its line -- and a title that ends its line would already have been caught by 1b.
# Measured on a test set of six document types: without this rule an NDA silently loses a
# whole clause, its text absorbed into the neighbouring section with no error.
BOLD_LEAD = re.compile(r"^\*\*(?P<t>[^*\n]{3,90}?)\*\*(?P<rest>.*)$")
# Rule 1e: a decimal-numbered paragraph -- "1.1 ...", "4.1.2 ..." -- starts a subsection even
# when no title follows the number and the rest of the line is a full sentence. Contracts
# number their subsections this way constantly and 1b rejects every one of them as prose.
# This gap was invisible while the input came from a converter that emitted each numbered
# clause as its own element; on raw Markdown it means a document has top-level clauses and
# nothing beneath them, so an agent sent to clause 4.2 has nowhere to land.
# The title is the bare number when the line carries no title of its own, which is honest:
# the number IS how the document refers to it.
DEC_LEAD = re.compile(r"^(?P<n>\d{1,2}(?:\.\d{1,2}){1,3})\.?\s+(?P<rest>\S.*)$")
# The trailing text is OPTIONAL. Requiring it meant a section titled with a bare number --
# which is exactly what rule 1e produces for an untitled subsection -- yielded no number at
# all, so stage 2 could not nest it and every subsection came out top-level.
NUM = re.compile(r"^(\d+(?:\.\d+)*)\.?(?:\s+(.*))?$")
# 1c: a numbered title on a line of its own, optionally bold, optionally colon-terminated
ALLOW_RX = re.compile(
    r"(?m)^\s*(?:\*\*)?(\d{1,2}\.)\s+([A-Z][A-Za-z0-9 &/\-,']{3,70}?)(?:\*\*)?\s*:?\s*$")


def strip_emphasis(t):
    """Markdown emphasis markers are not part of a title. Stripping only the leading run
    left every bold heading with a trailing `**` in its title."""
    return t.strip().strip("*_ ").strip("*_ ").strip(" .:\u2014-").strip()


def is_clause_heading(line):
    """Rule 1b -- a short titled line that is not a sentence."""
    t = strip_emphasis(line)
    if not (3 <= len(t) <= 80) or len(t.split()) > 10:
        return False
    if t.endswith(".") and not re.match(r"^\d+\.$", t):
        return False          # a full sentence, not a title
    if re.match(r"^\d+[-/]", t):
        return False          # a date or a data row, e.g. "10-Feb-20  9-Mar-20 ..."
    if re.match(r"^\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", t, re.I):
        return False          # a bare date on its own line -- "1 July 2024" is not a heading
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
        if m and strip_emphasis(m.group(2)):
            marks.append((i, strip_emphasis(m.group(2)), "heading"))
            continue
        stripped = ln.strip()
        if not stripped:
            continue
        # 1d: a bold-delimited lead-in is a heading even when its body follows on the
        # same line. Checked before 1b, which would reject the line as a sentence.
        b = BOLD_LEAD.match(stripped)
        if b and b.group("rest").strip():
            t = strip_emphasis(b.group("t"))
            if t and len(t.split()) <= 12:
                marks.append((i, t, "bold-lead"))
                continue
        # 1e: a decimal-numbered paragraph is a subsection boundary
        dm = DEC_LEAD.match(stripped)
        if dm:
            rest = strip_emphasis(dm.group("rest"))
            # if a short title follows the number, use it; otherwise the number stands alone
            head = rest.split(".")[0].strip() if rest else ""
            title = dm.group("n")
            if head and len(head.split()) <= 7 and head[0].isupper() and not head.endswith(","):
                title = "%s %s" % (dm.group("n"), head)
            marks.append((i, title, "decimal"))
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
    for k, s in enumerate(out):
        if not s["number"] or "." not in s["number"]:
            continue
        stem = _canon(s["number"].rsplit(".", 1)[0])
        if not stem:
            continue          # "1.0" is a top-level heading written decimally, not a child
        for back in range(k - 1, -1, -1):
            if _canon(out[back]["number"]) == stem:
                s["parent"] = back
                break
    for k, s in enumerate(out):
        s["order"] = k
        # a heading with no body of its own: keep it, mark it, do not summarise it
        s["container_only"] = s["words"] < 6 and any(x.get("parent") == k for x in out)
    return out


def _canon(num):
    """Drop trailing .0 groups before comparing clause numbers.

    Engineering and quality documents number their top-level sections `1.0`, `2.0` and their
    children `1.1`, `1.2`. Compared literally, the stem of `1.1` is `1`, which never equals
    `1.0`, so every subsection in such a document came out top-level. Canonicalising both
    sides makes `1.0` and `1` the same section number, which is what the document means.
    """
    if not num:
        return ""
    parts = num.split(".")
    while len(parts) > 1 and parts[-1] == "0":
        parts.pop()
    return ".".join(parts)


def _mk(title, body, parent, conf):
    return {"title": title, "text": body, "words": len(body.split()),
            "heading_confidence": conf, "parent": parent, "number": None,
            "container_only": False,
            # content-hashed, so inserting a section never renumbers the others
            "id": hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]}
