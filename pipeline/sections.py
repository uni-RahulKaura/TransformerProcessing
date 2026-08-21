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

from .survey import LABEL_ONLY, RULES, SIG_KEY, survey

HEADING_MD = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
# An explicitly numbered clause line -- the author put the number there, so it is evidence,
# unlike a bare short line which is only a guess.
# The bold marker can sit on either side of the number. This lease writes
# "1.  **BASIC SUBLEASE PROVISIONS**" -- number, two spaces, then the emphasis -- and matching
# "**" only before the number meant clauses 1 and 4 were never recognised, which left all of
# 1.1-1.16 and 4.1-4.6 sitting at the top level with no parent to nest under.
CLAUSE_LINE = re.compile(r"^(?:\*\*)?\d{1,2}(?:\.\d{1,3})*\.?\s+(?:\*\*)?[A-Z\"(]")
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
# "ARTICLE 6" and "Section 6" ARE section six. Without this they carry no number, so a child
# numbered 6.01 cannot see its real parent sitting directly above it and walks backwards until
# it finds any section starting with "6." -- which, in a document with another agreement
# attached to the back, was 137 sections away and in the wrong agreement entirely.
WORDNUM = re.compile(r"^(?:article|section|clause|annex|exhibit|schedule)\s+(\d+(?:\.\d+)*)\b",
                     re.I)
# A parent that far back is not a parent. Numbering restarts when a second agreement is
# attached, so an unlimited backward search crosses document boundaries and invents a
# hierarchy the document does not have.
MAX_PARENT_DISTANCE = 60
# 1c: a numbered title on a line of its own, optionally bold, optionally colon-terminated
ALLOW_RX = re.compile(
    r"(?m)^\s*(?:\*\*)?(\d{1,2}\.)\s+([A-Z][A-Za-z0-9 &/\-,']{3,70}?)(?:\*\*)?\s*:?\s*$")


def strip_emphasis(t):
    """Markdown emphasis markers are not part of a title.

    Stripping only at the ends is not enough: this lease writes "1.  **BASIC SUBLEASE
    PROVISIONS**", where the bold run opens in the MIDDLE of the line, after the number. So
    bold runs are removed wherever they appear and the resulting double space is closed up.
    """
    t = t.replace("**", " ")
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip().strip("*_ ").strip("*_ ").strip(" .:\u2014-").strip()


# A line can look title-shaped and still be content. These are the shapes that produced 605
# false headings on the corpus: account and invoice numbers, phone numbers, bare dates, postal
# lines, and label/value pairs off a statement.
NOT_A_TITLE = [
    re.compile(r"^[\d\W]+$"),                                  # digits and punctuation only
    re.compile(r"^\d{3}[.\-]\d{3}[.\-]\d{4}$"),               # 888.881.2622
    re.compile(r"^\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}$"),    # 04 / 29 / 2019
    re.compile(r"^\d{6,}$"),                                    # 4765287825
    re.compile(r"^[A-Z][A-Za-z /'&.-]{1,34}:\s*\S"),            # Account: 689256923-12345
    re.compile(r"^[A-Z\s]+,\s*[A-Z]{2}\s+\d{5}(-\d{4})?$"),    # PIERZ, MN 56364-1530
    re.compile(r"^\d+\s+[A-Z][A-Z\s]+(AVE|ST|RD|BLVD|DR|LN|WAY|CT|PKWY)\b", re.I),
    re.compile(r"^(?:page|empty row|keyline)\b", re.I),
]


def looks_like_content(t):
    """True when a title-shaped line is really a value, a code or an address."""
    t = t.strip()
    if not t:
        return True
    digits = sum(c.isdigit() for c in t)
    if digits and digits / len(t) > 0.45:      # mostly numbers -> a code, not a title
        return True
    return any(rx.match(t) for rx in NOT_A_TITLE)


BLK = "\x00BLK\x00"
WITNESS = re.compile(r"^\s*(?:\*\*)?IN WITNESS\s+(?:WHEREOF|THEREOF)", re.I)
SIG_FIELD = re.compile(r"^\s*(?:\*\*)?(Signature|DocuSigned by|By|Name|Title|Date|Witness|"
                       r"Printed Name|Print Name|Its|Attest)\s*:", re.I)


def signature_block_starts(lines):
    """Line numbers where an execution block begins.

    A signed agreement ends in one, and it is a real part of the document -- the reader wants to
    know who signed and when -- but it is not prose and must not be broken into a section per
    "By:" line. So it is marked once, as one entry. Two signals, either is enough: the witness
    clause that opens it, or a cluster of at least three signature fields in a short span, which
    is what a block looks like when the witness clause was lost in conversion.
    """
    starts, n = [], len(lines)
    i = 0
    while i < n:
        if WITNESS.match(lines[i]):
            starts.append(i)
            i += 12
            continue
        if SIG_FIELD.match(lines[i]):
            hits = sum(1 for j in range(i, min(i + 14, n)) if SIG_FIELD.match(lines[j]))
            if hits >= 3:
                starts.append(i)
                i += 14
                continue
        i += 1
    # collapse counterpart pages: signature blocks within 25 lines are one block
    out = []
    for x in starts:
        if not out or x - out[-1] > 25:
            out.append(x)
    return out
# A block whose whole content is one of these is page furniture, not a section.
FURNITURE_LINE = re.compile(
    r"^(?:\d{1,4}|[A-Z](?:\s+[A-Z]){2,}|T E S L A|KEYLINE|VN|Empty Row|-{3,}|"
    r"\(Unaudited\)|\(continued\)|\(cont\.?\))$", re.I)


# Rejecting a block-lead candidate needs a narrower content test than the one the other rules
# use. looks_like_content() treats "Label: value" as a form field, which is right for an
# invoice but wrong here: "US: California, Nevada and Texas" and "China: Shanghai" are the
# real subsection titles of the vehicle-capacity page.
BL_REJECT = [re.compile(r"^[\d\W]+$"), re.compile(r"^\d{3}[.\-]\d{3}[.\-]\d{4}$"),
             re.compile(r"^\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}$"), re.compile(r"^\d{5,}$"),
             re.compile(r"^(?:page|empty row|keyline|source|note)\b", re.I)]


def _bl_content(t):
    digits = sum(c.isdigit() for c in t)
    if digits and digits / len(t) > 0.45:
        return True
    return any(rx.match(t) for rx in BL_REJECT)


def block_furniture(lines):
    """Short bare lines that recur all through the document are running headers, not sections.

    A converted deck or statement reprints its brand mark, page number and continuation banner
    on every page. Each looks exactly like a title. Frequency is what separates them: a real
    section heading is written once, a footer is written thirty times. Counted per document, so
    no brand name or banner wording has to be known in advance.
    """
    from collections import Counter
    c = Counter(_fold(x) for x in lines if 0 < len(x.strip()) <= 70)
    return {k for k, n in c.items() if n >= 3}


CONT_SUFFIX = re.compile(r"\s*\((?:continued|cont\.?)\)\s*$", re.I)
TRAIL_MONEY = re.compile(r"\s[-(]?\$[\d,]+(?:\.\d\d)?\)?$")


def _fold(line):
    """Normalise a line for repeat-counting: case, and the "(continued)" a running banner
    picks up on its second and later pages. Without folding these together, "Charges by line
    details" and "Charges by line details (continued)" count as two distinct one-off titles
    and both survive."""
    return CONT_SUFFIX.sub("", line.strip()).lower()


def is_block_lead(lines, i, furniture=frozenset()):
    """The first line of a layout block is a heading when the block has a body under it.

    Landing writes one anchor per visual block, so the line right after a block marker is the
    top line of something the page set apart. A deck's section titles are set in large type
    and arrive here as exactly that: a short bare line with its content beneath. Requiring a
    body is what keeps page numbers, footnote runs and the "T E S L A" footer out -- those sit
    alone in their block.
    """
    t = lines[i].strip()
    if not (2 <= len(t) <= 70) or len(t.split()) > 9:
        return None
    if FURNITURE_LINE.match(t) or t.endswith((".", ",", ";", ":")):
        return None
    if _fold(t) in furniture:
        return None
    # "Devices $25.94" is a charge line, not a heading: a title does not carry its own amount.
    if TRAIL_MONEY.search(t):
        return None
    # A converted page often cuts a paragraph at the column edge, leaving "...we regularly" --
    # short enough to pass for a title. A heading does not trail off in a lower-case word.
    w = t.split()
    if len(w) >= 5 and w[-1][:1].islower() and not t.endswith(("?", "!")):
        return None
    if not t[0].isalpha() or _bl_content(t):
        return None
    if t.startswith("(") or "  " in t.strip():
        return None
    # must have a real body inside the same block. A masked table or figure counts as a body
    # and counts as a large one: "FINANCIAL SUMMARY" over a 6-column table is the single most
    # common shape in a financial deck, and measuring the placeholder's 17 characters instead
    # of the table it stands for rejected exactly the headings worth having.
    body = 0
    for j in range(i + 1, min(i + 40, len(lines))):
        nxt = lines[j].strip()
        if nxt == BLK:
            break
        if "MASK" in nxt:
            body += 300
        elif nxt:
            body += len(nxt)
    if body < 60:
        # A title alone in its own block. In a deck that is the normal shape -- the title is one
        # layout element and its content is the next -- so look one block further before giving
        # up. Running headers are already gone, so what reaches here is a real banner.
        nxt_body, seen_blk = 0, False
        for j in range(i + 1, min(i + 60, len(lines))):
            v = lines[j].strip()
            if v == BLK:
                if seen_blk:
                    break
                seen_blk = True
                continue
            if not seen_blk:
                continue
            if "MASK" in v:
                nxt_body += 300
            elif v:
                nxt_body += len(v)
        if nxt_body < 120:
            return None
    return t


NUM_PREFIX = re.compile(r"^(?:\*\*)?(\d{1,2}(?:\.\d{1,3})*)\.?\s+")


INLINE_SEP = re.compile(r"^(?P<title>[^.:;]{2,70}?)\s*(?P<sep>[:.])\s+(?P<body>\S)")


def clause_inline_title(t):
    """A numbered clause whose title sits on the same line as its body.

    "1. BUSINESS PURPOSE: In order to enable the parties to discuss..." is one line, and the
    title is the part before the colon. Every NDA here is written this way, so rejecting the
    line as prose -- which its length and lower-case tail otherwise imply -- dropped those
    documents to two sections. Returns "1. BUSINESS PURPOSE", or None when no title is there.
    """
    m = NUM_PREFIX.match(t)
    if not m:
        return None
    num, rest = m.group(1), t[m.end():].strip()
    im = INLINE_SEP.match(rest)
    if not im:
        return None
    title = im.group("title").strip().rstrip("*").strip()
    w = title.split()
    if not w or len(w) > 10 or not title[0].isalpha():
        return None
    caps = sum(1 for x in w if x[:1].isupper())
    # a title is capitalised throughout; a sentence's first clause is not. A colon is a much
    # stronger signal than a period, so it needs less capitalisation to be believed.
    need = 0.5 if im.group("sep") == ":" else 0.8
    if caps / len(w) < need:
        return None
    return "%s. %s" % (num, title)


def numbered_prose(t):
    """True when a numbered line is a list item rather than a clause heading.

    A lease's exhibits are numbered lists -- alteration rules, cleaning specifications,
    prohibited uses -- and every item opens exactly like a clause: a number, then a capital.
    What separates them is what follows. A clause heading is a title: a few words, no terminal
    punctuation ("2. DEMISE OF SUBLEASE PREMISES"). A list item is a sentence, and states an
    obligation in full ("1. To the extent there is a conflict between the provisions contained
    in this Exhibit and the other provisions of this Lease, ..."). Reading each of those as a
    section is what turned a 134-clause sublease into 506 sections.
    """
    m = NUM_PREFIX.match(t)
    if not m:
        return False
    rest = t[m.end():].strip().rstrip("*").strip()
    if not rest:
        return False
    if rest.endswith((".", ";", ",")) and not re.match(r"^[A-Z][A-Z &/'-]+$", rest):
        return True
    words = rest.split()
    if len(words) > 12:
        return True
    # a title is capitalised or title-cased throughout; a sentence is not
    if len(words) >= 5 and sum(1 for w in words if w[:1].isupper()) <= len(words) // 3:
        return True
    return False


def _looks_titled(head):
    """Whether text lifted from the start of a numbered paragraph is really its title.

    "3.2 Subject to Section 1.5 above, if for any reason Sublandlord is delayed..." splits on
    the point inside "1.5" and yields "Subject to Section 1" -- a fragment of a sentence that
    passes a first-letter-capital test and reads, wrongly, as a caption. A title is capitalised
    throughout and does not trail off in a number.
    """
    w = head.split()
    if not w or len(w) > 7 or head.endswith(",") or not head[0].isupper():
        return False
    if w[-1].rstrip(".").isdigit():
        return False
    alpha = [x for x in w if x[:1].isalpha()]
    if not alpha:
        return False
    return sum(1 for x in alpha if x[0].isupper()) / len(alpha) >= 0.6


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
    if looks_like_content(t):
        return False
    return t


def split_sections(text, regime=None):
    """Apply stages 1 and 2 to raw Markdown. Returns a list of section dicts.

    Which rules run depends on what the document actually contains -- see pipeline/survey.py.
    A lease and a phone bill do not have the same kind of structure, and applying the lease
    rules to the bill is what produced 110 sections for a document with a dozen.
    """
    sv = survey(text)
    regime = regime or sv["regime"]
    allowed = RULES[regime]
    allow = sorted({m.group(2).strip() for m in ALLOW_RX.finditer(text)},
                   key=len, reverse=True) if allowed["clause"] else []
    # When the numbering never restarts, it IS the document's spine and every numbered line is
    # a clause -- even the ones with no title, which are written out as "no header". When it
    # restarts many times the numbers are being reused by exhibit lists, so a numbered line has
    # to carry a title to count. Two NDAs are entirely untitled clauses; the sublease reuses
    # "1." twenty-three times. One test separates them.
    single_series = sv["restarts"] <= 1
    lines = text.split("\n")
    furniture = block_furniture(lines) if allowed.get("blocklead") else frozenset()
    # A document names each of its sections once. A second block-lead with the same wording is
    # the running header of a continued page, so only the first is a section boundary.
    seen_lead = set()
    sig_starts = set(signature_block_starts(lines))
    marks = []                                   # (line index, title, confidence)
    for i, ln in enumerate(lines):
        if i in sig_starts:
            marks.append((i, "signature block", "signature"))
            continue
        m = HEADING_MD.match(ln)
        if m and strip_emphasis(m.group(2)):
            marks.append((i, strip_emphasis(m.group(2)), "heading"))
            continue
        stripped = ln.strip()
        if not stripped or stripped == BLK:
            continue
        # 1g: first line of a layout block, in documents whose "#" headings are too sparse to
        # be the whole backbone.
        if allowed.get("blocklead") and i > 0 and lines[i - 1].strip() == BLK:
            t = is_block_lead(lines, i, furniture)
            if t:
                key = _fold(t)
                if key not in seen_lead:
                    seen_lead.add(key)
                    marks.append((i, strip_emphasis(t), "block-lead"))
                continue
        # 1d: a bold-delimited lead-in is a heading even when its body follows on the
        # same line. Checked before 1b, which would reject the line as a sentence.
        b = BOLD_LEAD.match(stripped) if allowed["bold"] else None
        if b and b.group("rest").strip():
            t = strip_emphasis(b.group("t"))
            if t and len(t.split()) <= 12:
                # In a page-layout document the same bold label is reprinted on every page --
                # "Invoice:", "Billing period:", "Plan" -- and a charge line carries its own
                # amount. Neither is a section. Applied only where the document IS page
                # furniture-heavy; a contract may legitimately repeat a bold caption.
                if allowed.get("blocklead"):
                    if _fold(t) in furniture or TRAIL_MONEY.search(t) or _fold(t) in seen_lead:
                        continue
                    seen_lead.add(_fold(t))
                marks.append((i, t, "bold-lead"))
                continue
        # 1e: a decimal-numbered paragraph is a subsection boundary
        dm = DEC_LEAD.match(stripped) if allowed["decimal"] else None
        if dm and not looks_like_content(stripped):
            rest = strip_emphasis(dm.group("rest"))
            # if a short title follows the number, use it; otherwise the number stands alone
            head = rest.split(".")[0].strip() if rest else ""
            title = dm.group("n")
            if head and _looks_titled(head):
                title = "%s %s" % (dm.group("n"), head)
            marks.append((i, title, "decimal"))
            continue
        # 1f: a line that is only a short title ending in a colon. Amendments and letters mark
        # their sections this way when they carry no numbering at all. Excludes the execution
        # block ("By:", "Name:", "Title:"), which is boilerplate, not structure.
        if allowed.get("label"):
            lm = LABEL_ONLY.match(stripped)
            if lm:
                t = strip_emphasis(lm.group(1)).strip()
                if (t and t.lower() not in SIG_KEY and len(t.split()) <= 14
                        and not looks_like_content(t)):
                    marks.append((i, t, "label"))
                    continue
        # 1c takes precedence over 1b: an allowlisted title may be glued to its paragraph
        hit = next((a for a in allow if stripped.startswith(a) and len(stripped) > len(a)),
                   None) if allowed["clause"] else None
        if hit:
            marks.append((i, hit, "allowlist"))
            continue
        # 1b is the rule that invents sections. It only runs where there is nothing else to
        # go on: a short document with no Markdown headings and no clause numbering.
        if allowed["infer"] and len(stripped) < 90:
            t = is_clause_heading(stripped)
            if t:
                marks.append((i, t, "inferred"))
        elif allowed["clause"] and CLAUSE_LINE.match(stripped):
            t = strip_emphasis(stripped)
            if t and not looks_like_content(t):
                inline = clause_inline_title(t)
                if inline:
                    marks.append((i, inline, "clause"))
                elif single_series or not numbered_prose(t):
                    m2 = NUM_PREFIX.match(t)
                    # an untitled clause is a real section; name it by its number
                    title = t[:120] if not (single_series and numbered_prose(t)) \
                        else "%s." % m2.group(1)
                    marks.append((i, title, "clause"))

    # A picture written out as a text outline -- a network diagram listing its own boxes, one
    # per line -- gives a long run of short capitalised lines. Each looks exactly like a
    # title to rule 1b, so a single diagram became twenty-five empty "sections". A heading
    # has a body; a label does not. So a run of THREE OR MORE consecutive shape-inferred
    # marks with almost no text between them is a label block, and only the first is kept.
    pruned, i2 = [], 0
    while i2 < len(marks):
        j2 = i2
        while (j2 + 1 < len(marks) and marks[j2 + 1][2] == "inferred"
               and marks[j2][2] == "inferred" and marks[j2 + 1][0] - marks[j2][0] <= 2):
            j2 += 1
        if j2 - i2 >= 2:
            pruned.append(marks[i2])          # keep the first, absorb the rest as its body
        else:
            pruned.extend(marks[i2:j2 + 1])
        i2 = j2 + 1
    marks = pruned

    # A numbered line is only a heading if it is not one of a run of them. Exhibit G of one
    # contract is a cleaning specification written as "1. Sweep... 2. Wash... 6. Wash and clean
    # all water fountains", and every line of it was being promoted to a top-level section.
    # A real heading has a body under it; a list item is followed immediately by the next item.
    seq = re.compile(r"^(\d{1,2})\.\s")
    keep = []
    for a, mk in enumerate(marks):
        m = seq.match(mk[1])
        if not m:
            keep.append(mk)
            continue
        # look at the neighbours: consecutive integers, close together, no body between
        run = 1
        for b in (a - 1, a + 1):
            if 0 <= b < len(marks):
                m2 = seq.match(marks[b][1])
                if m2 and abs(int(m2.group(1)) - int(m.group(1))) == 1 \
                        and abs(marks[b][0] - mk[0]) <= 4:
                    run += 1
        if run >= 3 and len(mk[1].split()) > 4:
            continue          # part of a numbered list, and long enough to be an item not a title
        keep.append(mk)
    marks = keep

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
        m = NUM.match(s["title"]) or WORDNUM.match(s["title"])
        s["number"] = m.group(1) if m else None
    for k, s in enumerate(out):
        if not s["number"] or "." not in s["number"]:
            continue
        stem = _canon(s["number"].rsplit(".", 1)[0])
        if not stem:
            continue          # "1.0" is a top-level heading written decimally, not a child
        for back in range(k - 1, max(-1, k - 1 - MAX_PARENT_DISTANCE), -1):
            if _canon(out[back]["number"]) == stem:
                s["parent"] = back
                break
    out = merge_numbered_lists(out)
    out = collapse_label_runs(out)
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

LISTITEM = re.compile(r"^(\d{1,2})[.)]\s+(\S.*)$")


def _eff_body(s):
    """Body size, counting a masked table or figure as substantial.

    Measuring the placeholder's characters instead of the table it stands for makes any
    table-only section look empty, which is exactly the shape of a real financial-deck heading.
    """
    t = s.get("text") or ""
    return len(t) + 300 * t.count("MASK")


def collapse_label_runs(out, run=6, body=400):
    """Fold a long run of short same-kind entries into the first of them.

    A definitions article ("**Acceptance Period:** unless otherwise provided...") and a cover
    approval form ("Requestor / Contract Title / Expense Amount / ...") both produce a long
    run of entries that each look like a titled section and none of which is one -- they are
    the CONTENT of a single section. A real clause run is broken up by long bodies; these are
    not. Requiring six in a row keeps ordinary consecutive clause headings safe.
    """
    n = len(out)
    small = [i for i in range(n)
             if (out[i].get("title") or "") and _eff_body(out[i]) < body
             and len((out[i]["title"] or "").split()) <= 8]
    smalls = set(small)
    drop, i = set(), 0
    while i < n:
        if i not in smalls:
            i += 1
            continue
        j = i
        while j + 1 < n and (j + 1) in smalls and out[j + 1].get("heading_confidence") == out[i].get("heading_confidence"):
            j += 1
        if j - i + 1 >= run:
            for k in range(i + 1, j + 1):
                out[i]["text"] = (out[i]["text"] or "") + "\n" + (out[k].get("title") or "") \
                    + " " + (out[k].get("text") or "")
                drop.add(k)
        i = j + 1
    return [s for k, s in enumerate(out) if k not in drop]


def merge_numbered_lists(out):
    """Fold numbered list items back into the section above them.

    A contract exhibit written as "1. Sweep the floors  2. Wash the basins  3. Empty the bins"
    is a LIST, and every line of it was becoming its own top-level section -- inventing
    structure the document does not have. Earlier attempts at this keyed on neighbouring marks
    and were defeated by the order the rules run in, so this works on the finished list where
    nothing can shift underneath it.

    A run counts as a list when three or more consecutive sections carry ascending numbers and
    each is short enough to be an item rather than a titled clause. The first item keeps its
    place so the list is still findable; the rest become part of its body.
    """
    n = len(out)
    nums = [None] * n
    for i, s in enumerate(out):
        m = LISTITEM.match((s["title"] or "").strip())
        # What marks an item is not its own length but the absence of a body. A titled clause
        # opens a passage; a list entry is the whole entry. So "1. Apple / 3. Samsung / 4.
        # Universal Music" -- a competitor list in a lease exhibit -- collapses, while an
        # untitled but substantive clause ("1. As used herein, the Confidential Information...")
        # does not. Sub-numbered entries (5.3.1) are real subsections and are left alone.
        if m and "." not in m.group(1) and len(s.get("text") or "") < 240:
            nums[i] = int(m.group(1))
    drop = set()
    i = 0
    while i < n:
        if nums[i] is None:
            i += 1
            continue
        j = i
        # ascending, but not necessarily by exactly one: earlier rules drop some entries, so the
        # surviving run reads 1, 3, 4, 5. Requiring +1 exactly missed every list that had been
        # partly pruned already.
        while j + 1 < n and nums[j + 1] is not None and nums[j] < nums[j + 1] <= nums[j] + 4:
            j += 1
        if j - i + 1 >= 3:                  # three ascending numbered lines = a list
            for k in range(i + 1, j + 1):
                out[i]["text"] = (out[i]["text"] + " " + out[k]["title"] + " "
                                  + out[k]["text"]).strip()
                out[i]["words"] = len(out[i]["text"].split())
                for key in ("tables", "figures"):
                    if key in out[k]:
                        out[i].setdefault(key, []).extend(out[k][key])
                drop.add(k)
        i = j + 1
    if not drop:
        return out
    keep = [s for i, s in enumerate(out) if i not in drop]
    remap = {}
    for new, (old, s) in enumerate((i, s) for i, s in enumerate(out) if i not in drop):
        remap[old] = new
    for s in keep:
        s["parent"] = remap.get(s["parent"]) if s["parent"] is not None else None
    for k, s in enumerate(keep):
        s["order"] = k
    return keep
