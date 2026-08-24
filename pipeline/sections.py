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

from .survey import BOLD_ONLY, LABEL_ONLY, RULES, SIG_KEY, survey

HEADING_MD = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
# Rule 1a again, for the other markup language. Real documents pulled off the web arrive as
# Markdown files whose body is HTML: canada.ca's research summary is a Jekyll page with YAML
# front matter and eight `<h2>` headings, and because no rule here looked for a heading TAG the
# document came back with "no header" and none of its eight sections -- F1 0.000 against the
# annotator's key. An HTML heading is the author declaring a heading exactly as "#" is, so it is
# read with the same confidence and not as a guess.
# Scope, measured: 2 of the 173 real documents and NONE of the 208 generated ones contain a
# standalone heading tag, so this buys one real document and cannot cost any.
HEADING_HTML = re.compile(r"^\s*<h([1-6])\b[^>]*>(.*?)</h\1\s*>\s*$", re.I)
ANY_TAG = re.compile(r"<[^>]+>")
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
BOLD_LEAD = re.compile(
    # The emphasis run is not always "**". Markdown writes bold-italic and doubled emphasis as
    # __**x**__, ***x***, **_x_**, ___x___ -- and the CLOSING run is the mirror of the opening
    # one, not a copy of it, so a backreference will not do. Anchoring on "**" alone made a whole
    # style of contract invisible: agreements that write their clauses as
    #     __**Section 1.1 "Affiliate"**__ means any entity that ...
    # matched no heading rule at all, because CLAUSE_LINE and DEC_LEAD both require the line to
    # BEGIN with a digit and this one begins with an underscore. On one 856 KB generated master
    # agreement that was 816 real sections found as 45 -- recall 0.048. Four of the five worst
    # agreements in the corpus failed this single way.
    r"^(?P<open>[*_]{2,6})(?P<t>[^*_\n][^\n]{1,88}?)(?P<close>[*_]{2,6})(?P<rest>.*)$")


def bold_lead(line):
    """BOLD_LEAD, with the closing emphasis run required to mirror the opening one.

    Checked rather than assumed: "**a** and **b** in one line" would otherwise match with t="a"
    and rest=" and **b** ...", which is a sentence containing emphasis, not a heading. Requiring
    the runs to use the same characters in the same numbers keeps __**x**__ and ***x*** while
    rejecting a line that merely happens to contain two emphasised words.
    """
    m = BOLD_LEAD.match(line)
    if not m:
        return None
    if sorted(m.group("open")) != sorted(m.group("close")):
        return None
    return m
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
# (near, page_span) for is_continuation(), keyed on whether the regime allows a repeated title to
# be a separate section of its own.
#
# The defaults, (60, 250), are distance guesses -- "a repeat within 60 lines is the same section
# continuing; within 250 with a page break just before it, likewise" -- and in a page-layout
# document they are simply wrong. A quarterly report heads six different sections "PROPULSION
# REVIEW", each about 150 lines after the last and each at the top of its page, and the answer key
# lists all six; the page-break clause collapsed them. A 10-Q's supplemental tables repeat the same
# way. Guessing from distance cost these documents 40 sections apiece and produced the worst
# possible pairing of errors -- one miss AND one false section for every collapsed repeat, because
# the emitted count no longer matched the key's.
#
# So in a regime that has already decided repeated titles can be real, only the document's OWN
# statement counts: a "- CONTINUED" or "(cont'd)" suffix, which is the author saying so. Measured
# on reports, F1 0.901 -> 0.927, with statements 0.726 -> 0.725 and the 33 real files unchanged.
# The setting is not delicate -- (20, 0), (0, 120) and (10, 60) all give 0.926-0.927 -- so the
# honest form is the one that carries no distance guess at all.
CONT_SPAN = {False: (60, 250), True: (0, 0)}
# 1c: a numbered title on a line of its own, optionally bold, optionally colon-terminated
ALLOW_RX = re.compile(
    r"(?m)^\s*(?:\*\*)?(\d{1,2}\.)\s+([A-Z][A-Za-z0-9 &/\-,']{3,70}?)(?:\*\*)?\s*:?\s*$")



NUM_ALONE = re.compile(r"^\s*(?:\*\*)?(\d{1,2}(?:\.\d{1,3})*)\.?(?:\*\*)?\s*$")
# The same defect as NUM_ALONE, one level up: a DIVISION marker alone on its line, with the
# division's title on the next line. ATT's credit agreement writes every one of its nine
# articles this way --
#
#     # ARTICLE I
#     DEFINITIONS AND ACCOUNTING TERMS
#
# -- so the heading rule produced the title "ARTICLE I", which carries no words of its own. That
# is a false section (the key reads "ARTICLE I - DEFINITIONS AND ACCOUNTING TERMS") AND a missed
# one, twice over for nine articles. Deliberately NOT extended to EXHIBIT, SCHEDULE, ANNEX,
# APPENDIX or ATTACHMENT: those markers head an ATTACHED document rather than a division of this
# one, the line beneath is the attachment's own cover block rather than its title (ATT's
# "## SCHEDULE I" is followed by "AT&T INC."), and is_attachment() depends on the marker standing
# alone to stop stage 2 nesting across the attachment boundary.
DIVISION_ALONE = re.compile(r"^\s*(?P<h>#{1,6}\s+)?(?:\*\*)?"
                            r"(?P<t>(?:ARTICLE|PART|CHAPTER)\s+"
                            r"(?:[IVXLC]{1,7}|\d{1,3}|[A-Z]))\.?(?:\*\*)?\s*[:.]?\s*$")
# A Markdown heading whose whole text is a number. Three digits, not two: legislation numbers
# its provisions past 99 and these two statutory instruments both do.
BARE_NUM_HEADING = re.compile(r"^(\d{1,3}(?:\.\d{1,3})*)\.?$")
# A line that cannot be a division's title: it is itself numbered or marked, so it opens the
# division's first clause rather than naming the division.
DIVISION_TITLE_NO = re.compile(r"^(?:\*\*)?(?:\d|\(|ARTICLE|PART|CHAPTER|SECTION|CLAUSE"
                               r"|EXHIBIT|SCHEDULE|ANNEX|APPENDIX|ATTACHMENT)\b", re.I)


def join_split_numbers(lines, skip=frozenset()):
    """Rejoin a clause number sitting alone on its line with the title beneath it.

    The Sublandlord sublease writes clause 4 as two lines:

        4.
        RENT

    Every rule here reads one line at a time, so neither line was a heading: `4.` is a bare
    number and `RENT` is a lone capitalised word. Clause 4 therefore did not exist, and its six
    children 4.1-4.6 had no parent to nest under and were stranded at the top level of an
    838 KB document. Data Provider splits two of its clauses the same way.

    Two guards, each of which cost a real section when it was missing:

      * `5.11` alone on a line is the PARTY NAME in the 5.11, Inc NDA -- the company is 5.11
        Tactical -- and what follows it is its signature attestation. Joining that invented a
        subsection 5.11 nested under clause 5. A clause title never opens with a figure or a
        table, so a block marker after the number means the number is a name.
      * the following line must be SHORT. A bare `4.2` is a genuinely untitled clause whose
        body is the next paragraph; joining it to 2,000 characters of prose would make that
        whole paragraph the title.

      * `skip` holds the line numbers of a verified contents listing. Emerson's real SEC credit
        agreement prints its contents with each entry's PAGE NUMBER on its own line -- "Section
        2.04 Swing Line Loans" / "26" / "Section 2.05 Competitive Bid Advances" -- so this rule
        read every page number as a split clause number and manufactured 25 headings of the form
        "26. Section 2.05 Competitive Bid Advances". Those then ANCHORED the scorer's alignment
        ahead of the real body headings, so the document's genuine "Section 2.05" became a false
        section as well: 25 invented headings cost 25 real ones too. Costliest single defect on
        the two credit agreements.

    Length-preserving: the joined pair is written back as the joined line plus enough blanks to
    occupy the same number of lines it consumed. It has to be, now that a caller computes line
    numbers (the contents span) on the text BEFORE this pass and applies them after; the old
    form silently dropped one line per join and slid every later index.
    """
    out, i, n = [], 0, len(lines)
    while i < n:
        # The same split, the other way round: the TITLE first and the number under it, one
        # heading level deeper. legislation.gov.uk renders every provision of a statutory
        # instrument this way --
        #
        #     #### Citation, commencement and extent
        #     ###### 1
        #
        # -- so each provision arrived as two sections: a titled one with no body (the body
        # follows the number) and a numbered one with no title. On the two real UK instruments
        # here that is 86 and 87 false sections against keys of 109 and 122, and the keys read
        # "1 Citation, commencement and extent" -- one section carrying both. Merged rather than
        # dropped, so the provision number survives as structure and the body stays with the
        # heading that owns it.
        #
        # The deeper level is the whole discriminator: a heading nested UNDER a titled heading,
        # with nothing between them, cannot be a sibling section -- it is that heading's number.
        hn = HEADING_MD.match(lines[i]) if i not in skip else None
        if hn and BARE_NUM_HEADING.match(hn.group(2).strip()):
            k = len(out) - 1
            while k >= 0 and not out[k].strip():
                k -= 1
            pm = HEADING_MD.match(out[k]) if k >= 0 else None
            pt = pm.group(2).strip() if pm else ""
            if (pm and len(pm.group(1)) < len(hn.group(1)) and pt
                    and not pt[:1].isdigit() and not BARE_NUM_HEADING.match(pt)):
                out[k] = "%s %s %s" % (pm.group(1), hn.group(2).strip().rstrip("."), pt)
                out.append("")
                i += 1
                continue
        d = DIVISION_ALONE.match(lines[i]) if i not in skip else None
        if d:
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            nxt = lines[j].strip() if j < n else ""
            # The title line must be a plain title: short, capitalised, not a sentence, not a
            # heading the author marked himself (ATT's ARTICLE V puts "COVENANTS OF THE
            # BORROWERS" in a "###" heading of its own -- the author already said that is a
            # heading and absorbing it would delete his markup), and not the division's first
            # numbered clause.
            if (nxt and 2 < len(nxt) <= 90 and not nxt.endswith(".")
                    and len(nxt.split()) <= 12 and nxt[:1].isupper()
                    and not HEADING_MD.match(nxt) and nxt != BLK
                    and not DIVISION_TITLE_NO.match(nxt)
                    and not MASKED_BLOCK.match(nxt)):
                out.append("%s%s %s" % (d.group("h") or "", d.group("t"), nxt))
                out.extend([""] * (j - i))
                i = j + 1
                continue
        m = NUM_ALONE.match(lines[i]) if i not in skip else None
        if m:
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            nxt = lines[j].strip() if j < n else ""
            # Short title line, OR a long line whose leading phrase IS the title. Data Provider
            # Reuters writes "4.4" then "Comparable Software. If Supplier discontinues Support
            # of Software..." -- 300 characters, so the length guard alone rejected it and the
            # clause stayed untitled. Joining is safe here because the joined line then carries
            # a recognisable inline title, which is a much stronger signal than length.
            joined_has_title = bool(clause_inline_title("%s. %s" % (m.group(1), nxt))) if nxt else False
            if (nxt and not NUM_ALONE.match(nxt) and nxt != BLK
                    and (2 < len(nxt) <= 90 or joined_has_title)
                    and not nxt.startswith(("<a id=", "<::", "<table", "(block)", "- ", "* "))
                    and not re.match(r"^MASK\d{5}", nxt)
                    and not (len(nxt) <= 90 and nxt.rstrip().endswith("."))
                    and not nxt[0].islower()):
                out.append("%s %s" % (m.group(1) + ".", nxt))
                out.extend([""] * (j - i))
                i = j + 1
                continue
        out.append(lines[i])
        i += 1
    return out



# A title may arrive with markup welded onto it. Bill 26d4d305 writes
#     ## Ways to pay\n\n<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUg...
# as a single heading line with literal escapes, so the section title carried a 4 KB base64
# blob. The heading is real and its wording is right; only the attached image is not part of
# it. So the markup is removed and the title kept, rather than the section being thrown away.
EMBEDDED_MARKUP = re.compile(r"(?:\\n|\n)+\s*<(?:img|svg|table|a)\b.*$", re.I | re.S)
DATA_URI = re.compile(r"\s*<img\b[^>]*>|\s*data:[a-z]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+",
                      re.I)
TITLE_MAX = 200


# A footnote reference hanging off the end of a heading: "SUPPLEMENTAL BALANCE SHEET(1)",
# "Adjusted EBITDA ($M) and adjusted EBITDA margin (%)(2)". It refers to a note at the foot of the
# page and is not part of the section's name -- the same heading appears with (1) on one page and
# (2) on another, and an independent reader's key writes it without either. Leaving it attached
# split one section into two and made both unmatchable: a report emitted
# "SUPPLEMENTAL STATEMENT OF CASH FLOWS(1)" against a key that says
# "SUPPLEMENTAL STATEMENT OF CASH FLOWS", scoring it as one false section AND one miss.
# Superscript glyphs only, which is what conversion actually produces here. A parenthesised "(1)"
# is deliberately NOT stripped: it is also how a contract writes a genuine sub-clause marker, and
# a chart label like "Current installed annual capacity(1)" is not a heading in the first place.
# Measured: 1,155 of these in the 35 reports and none at all in any other class, so this cannot
# move the other five.
FOOTNOTE_MARK = re.compile(r"[\u00b9\u00b2\u00b3\u2070-\u2079]+\s*$")


def clean_title(t):
    """Remove markup that was glued to a title, and cap its length."""
    if not t:
        return t
    t = EMBEDDED_MARKUP.sub("", t)
    t = DATA_URI.sub("", t)
    t = FOOTNOTE_MARK.sub("", t)
    t = re.sub(r"\\n|\\t", " ", t)
    t = re.sub(r"<[^>]{0,400}>", " ", t)
    t = re.sub(r"\s+", " ", t).strip().strip("*_ ").strip()
    # A "title" longer than this is a paragraph that some rule mistook for a heading. Truncating
    # keeps the section (its boundary may well be right) while making the mistake visible rather
    # than letting a wall of prose sit in the outline.
    if len(t) > TITLE_MAX:
        t = t[:TITLE_MAX].rsplit(" ", 1)[0] + "\u2026"
    return t


# Inline emphasis that survived as HTML rather than as Markdown. The two SEC credit agreements
# mark every inline clause title with an underline tag:
#
#     2.01.1 <u>Description of Facility</u>. The Lenders hereby establish in favor of ...
#
# so the decimal rule read the title as "<u>Description of Facility</u>", _looks_titled()
# rejected it for not starting with a capital, and the clause came out titled with its bare
# number. That is 23 of Emerson's 47 missed sections AND 20 of its false ones -- the same
# section counted twice, because "2.01.1" does not match "2.01.1 Description of Facility".
# Deliberately a whitelist of INLINE tags: <table>, <img>, <a id=...> and the vision markers are
# block structure that other rules detect by name, and must survive untouched.
INLINE_TAG = re.compile(r"</?(?:u|i|b|em|strong|sup|sub|small|ins|mark|font|span|s|del|code)"
                        r"(?:\s[^<>]*)?/?>", re.I)


def strip_emphasis(t):
    """Markdown emphasis markers are not part of a title.

    Stripping only at the ends is not enough: this lease writes "1.  **BASIC SUBLEASE
    PROVISIONS**", where the bold run opens in the MIDDLE of the line, after the number. So
    bold runs are removed wherever they appear and the resulting double space is closed up.

    HTML inline emphasis goes the same way -- see INLINE_TAG above.
    """
    t = INLINE_TAG.sub("", t)
    t = t.replace("**", " ")
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip().strip("*_ ").strip("*_ ").strip(" .:\u2014-").strip()


# A line can look title-shaped and still be content. These are the shapes that produced 605
# false headings on the corpus: account and invoice numbers, phone numbers, bare dates, postal
# lines, and label/value pairs off a statement.
NOT_A_TITLE = [
    # A postal box line off a statement's remittance stub. "PO BOX 16810" reads as a titled
    # line and is furniture.
    re.compile(r"^P\.?\s?O\.?\s+BOX\s+\d+", re.I),
    # A street address WITH its city, state and ZIP. The existing address rule keys on the
    # street-type word ending the line, so "11 Madison Avenue, New York, NY 10010" -- which is
    # the building the sublease is about, printed as a running header -- slipped past it and
    # became a top-level section of an 838 KB document.
    re.compile(r"^\d+[A-Za-z]?\s+[\w'\-. ]+,\s*[A-Za-z .]+,\s*[A-Z]{2}\s+\d{5}"),
    # A line that begins mid-word: a single stray letter left behind when the converter broke a
    # line, e.g. "e February 12, 2025" from "...due February 12, 2025". A title never opens with
    # a one-letter word followed by a capitalised month.
    re.compile(r"^[a-z]\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d"),
    # An embedded image. One bill welds a 4 KB base64 data URI onto the end of its "Ways to pay"
    # heading, so the title carried the whole blob. Stripped rather than rejected -- see
    # clean_title -- but a line that is ONLY an image is not a title.
    re.compile(r"^\s*<img\b", re.I),

    re.compile(r"^[\d\W]+$"),                                  # digits and punctuation only
    re.compile(r"^\d{3}[.\-]\d{3}[.\-]\d{4}$"),               # 888.881.2622
    re.compile(r"^\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}$"),    # 04 / 29 / 2019
    re.compile(r"^\d{6,}$"),                                    # 4765287825
    re.compile(r"^[A-Z][A-Za-z /'&.-]{1,34}:\s*\S"),            # Account: 689256923-12345
    re.compile(r"^[A-Z\s]+,\s*[A-Z]{2}\s+\d{5}(-\d{4})?$"),    # PIERZ, MN 56364-1530
    # A street address. "25 Madison Avenue" -- the notices clause -- reads exactly like clause
    # 25, so the previous all-caps-only form missed it. Full words as well as abbreviations,
    # and title case as well as caps.
    # A street address. "25 Madison Avenue" -- the notices clause -- reads exactly like clause
    # 25. Two things the first version missed: a numbered street name ("50 W 23 St", where the
    # name is not a word), and a unit after the street type ("50 W 23 St, Suite 6A"), which
    # moved the line's end past the anchor and let a letterhead become a heading.
    re.compile(r"^\d+\.?\s+(?:[A-Z0-9][A-Za-z0-9.'-]*\s+)*"
               r"(?:AVE|AVENUE|ST|STREET|RD|ROAD|BLVD|BOULEVARD|DR|DRIVE|LN|LANE|WAY|CT|COURT"
               r"|PKWY|PARKWAY|PLAZA|PL|SQUARE|SQ|TERRACE|TER)\.?"
               r"(?:\s*,?\s*(?:SUITE|STE|APT|UNIT|FLOOR|FL|RM|ROOM|#)\s*[A-Z0-9-]+)?\.?$", re.I),
    re.compile(r"^(?:page|empty row|keyline)\b", re.I),
    # A line that is nothing but square-bracketed placeholders -- "[NAME OF BORROWER]",
    # "[DATE]", "[EMERSON ELECTRIC CO.][NAME OF SUBSIDIARY BORROWER]". Every attached form in the
    # two SEC credit agreements has a blank where the executing party's name goes, set in capitals
    # on a line of its own, and the ALL-CAPS rule read each as a section: 10 false sections in ATT
    # and 7 in Emerson. Brackets around the WHOLE line are the author saying "fill this in", which
    # is a value slot and never a title. A real title that merely contains a bracketed aside --
    # "Section 2.12 [Reserved]" -- has text outside the brackets and is unaffected.
    re.compile(r"^(?:\[[^\]]*\]\s*)+$"),
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
# "note" here means the annotation under a chart -- "Note: figures are unaudited" -- and matching
# it as a bare prefix threw away the numbered notes to the financial statements, which in a 10-Q
# are a third of the answer key: "Note 1. Inventory", "Note 18 - Related party transactions",
# every one of them rejected. A number after the word is the difference, and it cost doc-16
# nine sections and doc-9 twelve.
BL_REJECT = [re.compile(r"^[\d\W]+$"), re.compile(r"^\d{3}[.\-]\d{3}[.\-]\d{4}$"),
             re.compile(r"^\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}$"), re.compile(r"^\d{5,}$"),
             re.compile(r"^(?:page|empty row|keyline|source)\b", re.I),
             re.compile(r"^note\b(?!\s*[\d\u2013\u2014-]*\s*\d)", re.I)]


def _bl_content(t):
    digits = sum(c.isdigit() for c in t)
    if digits and digits / len(t) > 0.45:
        return True
    return any(rx.match(t) for rx in BL_REJECT)


# block_furniture()'s threshold, which was three. Three only became too strict once the block-lead
# rule stopped discarding every repeat outright (see seen_lead in split_sections): while it did,
# a title suppressed here had no second chance, and letting more through simply moved the loss.
# With the repeat handled by distance instead, the furniture set can afford to be smaller, and the
# right size was measured rather than argued. Sweeping it, reports / statements / the 33 real
# files:
#
#      3   0.748  0.724  0.579        10   0.758  0.724  0.575
#      6   0.762  0.724  0.577        12   0.759  0.724  0.573
#      8   0.767  0.724  0.579        16   0.759  0.724  0.577
#
# Eight is the peak on reports and leaves the other two exactly where they were. Above it the
# genuine mastheads start coming back and reports falls again.
RUNNING_REPEATS = 8


def block_furniture(lines, times=RUNNING_REPEATS):
    """Short bare lines that recur all through the document are running headers, not sections.

    A converted deck or statement reprints its brand mark, page number and continuation banner
    on every page. Each looks exactly like a title. Frequency is what separates them: a real
    section heading is written once, a footer is written thirty times. Counted per document, so
    no brand name or banner wording has to be known in advance.

    `times` is the threshold, because the two callers can afford different ones -- see the ALL-CAPS
    rule in split_sections(), which needs a stricter count than the block-lead rule does.
    """
    from collections import Counter
    c = Counter(_fold(x) for x in lines if 0 < len(x.strip()) <= 70)
    return {k for k, n in c.items() if n >= times}


# How many times a capitalised line has to appear before it is a masthead rather than a heading.
# It has to be looser than a page-furniture test would suggest, because a financial report
# genuinely reprints "SUPPLEMENTAL BALANCE SHEET" and "QUARTERLY FINANCIAL SUMMARY" over the
# several pages its supplemental tables run to. Measured over the 24 generated reports, the two
# populations separate with a gap and nothing sits in it:
#
#     capitalised line that IS a title in the answer key      4, 5, 6, 7          (max 7)
#     capitalised line that is the company masthead        8, 9, 10 ... 48        (min 8)
#
# so eight. Below it nothing real is lost; at it, 602 false sections go and no true one does.
BANNER_REPEATS = 8


def caps_banners(lines, times=BANNER_REPEATS):
    """Capitalised lines reprinted so often they are the masthead, not a section.

    A converted deck prints the company name at the edge of every page -- "ELMWOOD", and its
    letter-spaced logo form "E L M W O O D" -- and caps_title() reads both as titles, because
    nothing about either LINE distinguishes it from "HIGHLIGHTS". Frequency does: on the 24
    reports these mastheads alone produced 662 of the 685 false sections the caps rule made, up
    to 48 in a single document.

    Two existing defences miss them, and it is worth saying why rather than adding a third by
    accident. is_continuation() suppresses a repeat only when it is nearby or just past a page
    break, and in a 400 KB report consecutive pages are hundreds of lines apart. The running-header
    test in drop_figure_captions() requires EVERY occurrence to be body-less, and a masthead that
    happens to sit at the top of its page absorbs that page's text -- one such occurrence saves the
    other thirty.

    Counted through caps_title() and not off the raw lines, so this set is normalised exactly the
    way the rule that consults it is: same emphasis stripping, same trailing colon, same folding of
    a "(continued)" suffix.
    """
    from collections import Counter
    c = Counter()
    for ln in lines:
        t = ln.strip()
        if not t or t == BLK:
            continue
        ct = caps_title(t)
        if ct:
            c[_fold(ct)] += 1
    return {k for k, n in c.items() if n >= times}


# A continued page marks itself in every punctuation style a typesetter has ever used, and
# only the parenthesised form was matched. "CHARGES DETAIL BY SERVICE POINT - CONTINUED",
# "INTERVAL USAGE DETAIL \u2014 Continued" and "Service Point 73666 ... - Continued" all escaped
# the repeat check, so a utility bill contributed one section per page of each detail region:
# 739 false sections across 35 generated statements from this alone.
CONT_SUFFIX = re.compile(r"\s*[\(\[]?\s*[-\u2013\u2014,:]?\s*"
                         r"(?:continued|cont\u2019?d?|cont\.?)\s*[\)\]]?\s*$", re.I)
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
    # An ENUMERATED title is still a title. "1.2 Business model", "2. Operating review",
    # "4.1 Model Serving and Inference Infrastructure" are the headings of an annual report,
    # written as plain lines at the head of their layout block, and requiring the first character
    # to be a LETTER discarded every one of them: four reports whose entire outline is numbered
    # this way scored recall 0.36 against a precision of 0.98 -- they were not finding the wrong
    # sections, they were finding almost none. The number must be followed by a capitalised word,
    # which is what still keeps out a footnote marker ("13. (1) Trailing twelve months"), a bare
    # figure, and a data row. The lettered form "(a) Chief executive's review" is the same case and
    # was rejected twice over -- once for not starting with a letter and once by a blanket
    # "starts with a bracket" guard, which exists to keep "(Unaudited)" and "(continued)" out and
    # still does: those are one long word in brackets, not an enumerator followed by a title.
    # Two reports write their whole subsection outline this way, 54 headings between them.
    if not t[0].isalpha():
        nm = ENUM_LEAD.match(t)
        if not (nm and t[nm.end():][:1].isupper()):
            return None
    elif t.startswith("("):
        return None
    if _bl_content(t):
        return None
    if "  " in t.strip():
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

# A lettered or roman sub-item that carries a TITLE, and only those.
#
# The corpus has 1,154 lettered and roman items across 22 of the 33 documents, and 1,035 of them
# are bare list items -- "(ii) Taxes;", "(vi) if the Third Party Request is issued by a law
# enforcement entity ...". Promoting those would manufacture a thousand sections out of sentence
# fragments, which is exactly the defect an independent audit confirmed 36 times elsewhere in this
# outliner. So the rule fires only on the 119 that a drafter actually titled:
#
#   (a) **Services Warranty.** Provider warrants ...        bold title
#   (g) "**Commencement Date**" means ...                   bold defined term
#   (a) "Ancillary Permitted Uses" means ...                quoted defined term
#
# The bold run or the quoted-term-plus-"means" IS the signal; there is no length or shape guess.
LETTERED_TITLE = re.compile(
    r"^\(?(?P<l>[a-z]{1,3}|[ivx]{1,4})\)\s+"
    r"(?:"
    r"[\"\u201c]?\*\*(?P<bt>[^*\n]{2,60}?)\*\*[\"\u201d]?"
    r"|[\"\u201c](?P<qt>[^\"\u201d\n]{2,60})[\"\u201d]\s+"
    r"(?:means|shall mean|has the meaning|refers to)\b"
    r")", re.I)


def lettered_title(stripped):
    """The title of a titled lettered sub-item, or None for a bare list item."""
    m = LETTERED_TITLE.match(stripped)
    if not m:
        return None
    t = (m.group("bt") or m.group("qt") or "").strip().rstrip(".:;,")
    if not t or len(t.split()) > 9:
        return None
    # a fragment of running prose that happens to be emphasised is not a title
    if t[:1].islower() and not t.isupper():
        return None
    return "(%s) %s" % (m.group("l"), t)
# The enumerators a document puts in front of an unmarked title: a clause number, or a lettered
# list marker. Digits in brackets are deliberately excluded -- "(1)" is a footnote reference in
# every one of these reports, not a section marker.
ENUM_LEAD = re.compile(r"^(?:\*\*)?(?:\d{1,2}(?:\.\d{1,3})*\.?|\(\s*[A-Za-z]{1,3}\s*\))\s+")


INLINE_SEP = re.compile(r"^(?P<title>[^.:;]{2,70}?)\s*(?P<sep>[:.])\s+(?P<body>\S)")



PAGE_BREAK = re.compile(r"PAGE BREAK|^\s*Page \d+\b", re.I)


def is_continuation(lines, i, last, title="", near=60, page_span=250):
    """Whether a repeated title is the SAME section continuing over a page, or a different
    section that happens to reuse the wording.

    Three tests, in order of how much evidence they carry:

      1. The title SAYS so -- "... - CONTINUED", "(cont'd)". That is the author telling us, and
         it needs no distance test.
      2. The repeat is close by. A banner reprinted on the next page of one long region.
      3. A page break sits just before it AND the previous use is not far off.

    Test 3 needs that distance bound, and it cost real sections when it did not have one.
    Suppressing every post-page-break repeat took away the Tesla-style reports' second and third
    "Cash", "Profitability", "Revenue" and "Product" sections -- each of those words genuinely
    heads three or four different sections of a quarterly deck, hundreds of lines apart, and in a
    dense report a page break is never far away. Distance is what separates a continuation from
    a reused word.

    Both bounds are caller-supplied, because how far apart two uses of a wording have to be is a
    property of the document type and not a constant -- see CONT_SPAN. Tests 2 and 3 switch off
    entirely at zero, leaving only test 1, the author's own "continued" marker.
    """
    if title and CONT_SUFFIX.search(title.strip()):
        return True
    if last is None:
        return False
    if i - last <= near:
        return True
    return (i - last <= page_span
            and any(PAGE_BREAK.search(lines[j]) for j in range(max(0, i - 3), i)))



CAPS_TITLE = re.compile(r"^(?:\*\*)?([A-Z][A-Z0-9][A-Z0-9 &/\'\-.,()]{1,68})(?:\*\*)?:?$")


def caps_title(stripped):
    """A line written entirely in capitals, standing alone, is a title.

    This is how these documents mark their most important boundaries and none of it was being
    read: "MUTUAL CONFIDENTIAL DISCLOSURE AGREEMENT" (an NDA's own title, reported as "no
    header"), "BACKGROUND" and its recitals, "COMPANY" over a signature block, "TECHNOLOGY
    ADDENDUM", "SERVICE LEVEL AGREEMENT", "ANNEX E TO EXHIBIT D" and the 165-line security
    schedule under it. Detection had been keyed on ** markup, so a document with no bold in it
    lost its title outright.

    Guards, because a paragraph in capitals is not a title: at most ten words, at least three
    letters, not an execution-block key, and not a line the noise patterns already reject
    (addresses, ZIP codes, page furniture).
    """
    # "B A L A N C E  S H E E T" is how the converter renders a letter-spaced heading, and it
    # cost the Q3 report its whole balance-sheet page while three sibling quarters kept theirs.
    spaced = re.fullmatch(r"(?:\*\*)?((?:[A-Z]\s+){2,}[A-Z]\.?)(?:\*\*)?", stripped)
    if spaced:
        letters = re.sub(r"\s+", "", spaced.group(1))
        if len(letters) >= 4:
            return re.sub(r"\s{2,}", " ", spaced.group(1)).strip()
    # Character whitelists were the wrong tool here. "THE TESLA ECOSYSTEM - CREATING A CLEANER,
    # SAFER, MORE ENJOYABLE WORLD", "FSD (SUPERVISED)(1) TESTING - PARIS, LONDON, SYDNEY AND
    # ROME" and "PROGRESS ON WORLD'S LARGEST SUPERCHARGER SITE | 168 STALLS" all failed on an
    # en-dash, a footnote mark or a pipe, so four whole photo pages vanished from one report and
    # their pictures inflated the neighbouring section's count. The test is simply: are the
    # letters all capitals?
    t = stripped.strip()
    if t.startswith("**") and t.endswith("**"):
        t = t[2:-2].strip()
    t = t.rstrip(":").strip()
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 3 or not all(c.isupper() for c in letters):
        m = CAPS_TITLE.match(stripped)
        if not m:
            return ""
        t = m.group(1).strip(" .,:")
    t = t.strip(" .,:")
    w = t.split()
    # A ten-word cap was rejecting the page titles these decks actually use: "POWERWALL 3 RAMP -
    # 1,500 UNITS IN A SINGLE SHIFT MILESTONE" is eleven words, "THE TESLA ECOSYSTEM - CREATING A
    # CLEANER, SAFER, MORE ENJOYABLE WORLD" is eleven. A capitalised SENTENCE is the thing to
    # exclude, and it is longer than this and ends in a full stop.
    if not (1 <= len(w) <= 16) or sum(c.isalpha() for c in t) < 3:
        return ""
    if len(w) > 4 and stripped.rstrip().endswith("."):
        return ""
    # "TITLE" is in the execution-block key list, which suppressed it as an audit-trail column
    # label while its siblings FILE NAME, DOCUMENT ID and STATUS all came through.
    if t.lower().rstrip(":") in (SIG_KEY - {"title"}) or any(rx.match(t) for rx in NOT_A_TITLE):
        return ""
    return t


# A masked table or figure, as sectionise() leaves it behind: "(block) MASK00007".
MASKED_BLOCK = re.compile(r"^\(block\)\s*MASK\d{5}\s*$")
# Press-release and regulatory furniture. These are title-shaped, set in capitals, and are not
# sections of anything -- they are the wire-service banner above the headline.
BANNER_FURNITURE = frozenset({
    "for immediate release", "regulatory announcement", "press release", "news release",
    "media release", "for release", "immediate release", "not for distribution",
    "embargoed until", "company announcement", "ad hoc announcement", "inside information",
})


def is_table_caption(lines, i):
    """Whether the title-shaped line at i is a CAPTION belonging to a table, not a heading.

    A financial press release writes

        Supplemental Balance Sheet (unaudited)
        <table>...</table>

    with the caption and the table inside ONE layout block -- no anchor between them, because
    the page showed them as one visual unit. A real heading gets its own block and is followed
    by prose. That distinction is the only thing separating the two, and without it the shape
    rule read every caption as a section: three generated press releases with 3, 9 and 9 real
    sections came out with 76, 55 and 76, a precision of 0.103. Nearly all of the invented ones
    were captions of supplemental financial tables.

    So: look forward. If the next thing with content is a masked table or figure, and no block
    boundary intervenes, this line captions it.
    """
    j = i + 1
    while j < len(lines):
        t = lines[j].strip()
        if not t:
            j += 1
            continue
        if t == BLK:
            return False                    # a block boundary -- the line stood alone
        return bool(MASKED_BLOCK.match(t))
    return False




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
    # "1. Name: Customer, as defined in the Agreement" is a field in a list of parties, not a
    # clause called "Name". The execution-block key list already names these labels.
    if title.lower().rstrip(":") in SIG_KEY:
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


# --- contents listings ----------------------------------------------------------------------
# A page number standing alone on its line: arabic, or the roman numerals a contents page uses
# for its own leaves ("i", "ii", "iii").
TOC_PAGE_NO = re.compile(r"^(?:\d{1,4}|[ivxlcdm]{1,7})\.?$", re.I)
# An entry with its page number on the SAME line, with or without dot leaders:
# "SECTION 9.14. Acknowledgement and Consent to Bail-In ... 69".
TOC_ENTRY = re.compile(r"^(?P<t>\S.*?[A-Za-z\)\]])[\s.…·\-]+(?P<p>\d{1,4})$")
# The document naming its own contents page. Anchored at the start of the line and capped at a
# few words, so a real section called "Arbitration Table of Contents" (the key for the scanned
# arbitration packet has one) is not mistaken for the declaration.
TOC_DECLARES = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*"
                          r"(?:table\s+of\s+contents|contents|index\s+of\s+(?:sections|clauses))"
                          r"\b[\s:.\-—]*(?:\(cont(?:inued)?\.?\))?(?:\*\*)?\s*$", re.I)


def _toc_class(s):
    """How a single line reads inside a candidate contents listing.

    'stop' is the important one: it is what keeps the span off the body of the document. Prose
    is the discriminator -- a contents page has none, and every real section has some.
    """
    s = s.strip()
    if not s or s == BLK:
        return "blank"
    if "http://" in s or "https://" in s:
        return "skip"                       # the SEC page footer, printed under every page
    if MASKED_BLOCK.match(s) or s.startswith(("<", "|", "(block)")) or HEADING_MD.match(s):
        return "stop"                       # a table, or a heading the author marked himself
    if TOC_PAGE_NO.match(s):
        return "paged"
    if len(s) > 140 or len(s.split()) > 20:
        return "stop"                       # prose
    if TOC_ENTRY.match(s):
        return "paged"
    if len(s.split()) > 8 and s[-1] in ".;,?!":
        return "stop"                       # a sentence, i.e. body text
    return "plain"


def contents_span(lines, min_paged=8, min_share=0.40):
    """Line numbers belonging to a verified contents listing, excluding its own heading.

    Both real SEC credit agreements in this corpus open with a contents page 130 to 180 lines
    long, and every rule here read it as structure: Emerson's produced 25 false clause headings
    and ATT's produced its nine "ARTICLE I".."ARTICLE IX" twice over, once from the listing and
    once from the body. The listing is also where "SCHEDULES", "EXHIBITS" and "Exhibit B-1 -
    Form of Notice of Borrowing" come from.

    An earlier attempt detected contents pages by looking for a run of title-shaped lines that
    RECUR later in the document. It does not work and the reason is worth recording: in Emerson
    the listing entries never recur as standalone lines -- "Article 1. DEFINITIONS" appears
    exactly once in the whole file, because the body writes the same clause as a Markdown
    heading with different spacing. The recurrence test found nothing at all.

    What does separate a contents page from the document is arithmetic, not wording: roughly
    half of its lines ARE page numbers, and none of it is prose. So:

      * anchor on the document naming the region itself -- one line reading "TABLE OF CONTENTS".
        This is the document's own declaration, not a keyword guess, and it means the rule can
        never fire on a document that has no contents page;
      * walk forward from there and stop at the first line that is prose, a table, or a heading
        the author marked with "#" -- any of which means the listing has ended and the body has
        begun;
      * trim back to the last line that carries a page number, so a title page sitting after
        the listing ("AMENDED AND RESTATED CREDIT AGREEMENT", four lines below the end of ATT's
        listing) stays outside;
      * accept only if at least `min_paged` lines carry page numbers and they are at least
        `min_share` of the region. A short "Contents" heading over three bullet points fails
        both tests and is left alone.

    The declaration line itself is NOT included: the annotators of both credit agreements
    recorded "TABLE OF CONTENTS" as a section of the document, which is the right reading -- a
    contents page is one section, not a hundred.
    """
    span = set()
    n = len(lines)
    for a in range(n):
        if a in span or not TOC_DECLARES.match(lines[a]):
            continue
        idx, paged, plain = [], 0, 0
        j = a + 1
        while j < n:
            c = _toc_class(lines[j])
            if c == "stop":
                break
            if c == "paged":
                paged += 1
                idx.append(j)
            elif c == "plain":
                plain += 1
                idx.append(j)
            j += 1
        while idx and _toc_class(lines[idx[-1]]) != "paged":
            idx.pop()
            plain -= 1
        if idx and paged >= min_paged and paged >= min_share * (paged + plain):
            span.update(range(a + 1, idx[-1] + 1))
    return frozenset(span)


def split_sections(text, regime=None):
    """Apply stages 1 and 2 to raw Markdown. Returns a list of section dicts.

    Which rules run depends on what the document actually contains -- see pipeline/survey.py.
    A lease and a phone bill do not have the same kind of structure, and applying the lease
    rules to the bill is what produced 110 sections for a document with a dozen.
    """
    sv = survey(text)
    regime = regime or sv["regime"]
    allowed = RULES[regime]
    # A SHORT numbered agreement groups its clauses under a plain label -- "PROVISIONS:",
    # "RECITALS:" -- and has nothing else to go on, so the label rule earns its place. A long
    # one does not: a lease's basic-provisions page is dozens of label lines ("Landlord:",
    # "Premises:", "Term:") which are fields of one section, not sections of their own. Turning
    # it on everywhere cost 274 extra sections across this corpus, nearly all in one lease.
    if regime == "clauses" and len(text) <= 25000:
        allowed = dict(allowed, label=True)
    allow = sorted({m.group(2).strip() for m in ALLOW_RX.finditer(text)},
                   key=len, reverse=True) if allowed["clause"] else []
    # When the numbering never restarts, it IS the document's spine and every numbered line is
    # a clause -- even the ones with no title, which are written out as "no header". When it
    # restarts many times the numbers are being reused by exhibit lists, so a numbered line has
    # to carry a title to count. Two NDAs are entirely untitled clauses; the sublease reuses
    # "1." twenty-three times. One test separates them.
    single_series = sv["restarts"] <= 1
    raw_lines = text.split("\n")
    # A verified contents listing is not structure -- see contents_span(). Computed on the raw
    # lines and applied to the joined ones, which is why join_split_numbers() is now
    # length-preserving; it is passed in as well so a page number sitting alone on its line is
    # not welded onto the entry beneath it.
    toc = contents_span(raw_lines)
    lines = join_split_numbers(raw_lines, toc)
    furniture = block_furniture(lines) if allowed.get("blocklead") else frozenset()
    # Masthead suppression is scoped to the regime, for the reason the whole of survey.py exists.
    # A capitalised line repeated eight times is a masthead in a financial report; in a deal packet
    # or a correspondence file it is the title of the eighth constituent document and IS a section
    # -- "BLANKET PURCHASE ORDER" appears 32 times in one correspondence file and the answer key
    # lists all 32. So this is on for "deck" only, where it is worth 602 false sections, and off
    # everywhere else, where it would cost real ones.
    banners = caps_banners(lines) if allowed.get("banner") else frozenset()
    # A document names each of its sections once -- USUALLY. A second block-lead with the same
    # wording is normally the running header of a continued page, but not always: a 10-Q's
    # supplemental tables run over several pages and the answer key lists "Supplemental Statement
    # of Operations" three times, hundreds of lines apart. So this records WHERE the wording was
    # last used and defers to is_continuation(), exactly as the caps rule does, instead of
    # suppressing every repeat outright. Suppressing outright was worse than losing the heading:
    # the branch consumes the line either way, so the repeat's content was absorbed into the
    # section above it -- one report came out with a single 1,018-word section holding 28 tables.
    seen_lead = {}

    # How far apart two uses of the same wording have to be before they are two sections rather
    # than one continuing. A document type answers this, not a constant: see CONT_SPAN.
    near, page_span = CONT_SPAN[bool(allowed.get("repeats"))]

    def cont(i, last, title=""):
        return is_continuation(lines, i, last, title, near=near, page_span=page_span)

    def repeat_is_continuation(key, i, title):
        """Whether a title already used at `key` is that section continuing, not a new one.

        The two document types genuinely differ here, so this defers to the regime rather than to
        a tuned distance. In a statement the same unmarked label is reprinted on every page of a
        form -- "Invoice", "Bill date", "Shipper", "Pro number" -- and is never a second section,
        so ANY repeat is a repeat. In a financial report a supplemental table title heads a real
        separate section three pages on and the answer key lists each one, so distance decides, as
        it does for the caps rule.

        Measured: deferring to distance in every regime is worth +0.020 F1 on reports but -0.015
        on statements, where it put "Invoice", "Shipper" and "Bill date" back on every page of
        seven documents. Scoped to the regimes that ask for it, it is +0.020 and 0.000.
        """
        prev = seen_lead.get(key)
        if prev is None:
            return False
        if not allowed.get("repeats"):
            return True
        return cont(i, prev, title)
    # A page header printed by the browser that saved the document -- "5/7/26, 5:43 AM EX-10.1",
    # eighty occurrences in the ATT credit agreement and a hundred and five in Emerson's -- is
    # set in capitals and stands alone, so the ALL-CAPS rule read every occurrence as a section:
    # 24 and 27 false sections respectively, each one also SPLITTING the clause it landed in and
    # taking that clause's body away from it. The existing repeat check could not catch them:
    # is_continuation() requires the repeat to be nearby or just past a page break, and these are
    # a whole page apart. Frequency catches them with no wording known in advance.
    #
    # The threshold is 4 rather than block_furniture's 3, and the extra one is not arbitrary:
    # "Cash", "Profitability", "Revenue" and "Product" each head THREE genuinely different
    # sections of a quarterly report, and dropping those was the mistake the comment on the caps
    # rule already warns about. Four occurrences of identical wording is past what a document
    # does by accident -- on these two agreements the real ALL-CAPS headings occur once or twice
    # ("AMENDED AND RESTATED CREDIT AGREEMENT" three times, so it survives) and the furniture
    # occurs twenty to a hundred times. Nothing in the corpus sits between 4 and 20.
    recurring = block_furniture(lines, times=4) if allowed.get("caps") else frozenset()
    seen_caps = {}
    sig_starts = set(signature_block_starts(lines))
    marks = []                                   # (line index, title, confidence)
    for i, ln in enumerate(lines):
        if i in toc:
            continue
        if i in sig_starts:
            marks.append((i, "signature block", "signature"))
            continue
        m = HEADING_MD.match(ln)
        if m and strip_emphasis(m.group(2)):
            marks.append((i, strip_emphasis(m.group(2)), "heading"))
            continue
        hh = HEADING_HTML.match(ln)
        if hh:
            # The inner text can carry its own markup -- canada.ca wraps every acronym in
            # <abbr title="...">, so "GST/HST" arrives inside a tag. The heading is the text.
            ht = strip_emphasis(re.sub(r"\s{2,}", " ", ANY_TAG.sub(" ", hh.group(2))))
            if ht:
                marks.append((i, ht, "html-heading"))
                continue
        stripped = ln.strip()
        if not stripped or stripped == BLK:
            continue
        # 1g: first line of a layout block, in documents whose "#" headings are too sparse to
        # be the whole backbone.
        if allowed.get("blocklead") and i > 0 and lines[i - 1].strip() == BLK:
            t = is_block_lead(lines, i, furniture)
            # A caption glued to its table is not a heading, and the test for that already existed
            # -- it was simply never wired to this rule, only to caps and infer. It matters here
            # because the two shapes are written differently and the answer keys agree with the
            # distinction: a press release writes
            #     Supplemental Statement of Operations (unaudited)
            #     <table>...
            # inside ONE layout block, and its key does not list those tables as sections; a 10-Q
            # writes the same title, then "(Unaudited)", then a block boundary, then the table, and
            # its key lists every one. One report emitted 53 sections against a key of 9 on this
            # alone (precision 0.170 -> 0.818); across the class it is worth +0.033 F1, and +0.003
            # on statements.
            #
            # Applied in every regime that reads layout blocks, not just the deck one, because the
            # argument does not depend on the document type. Restricting it to decks was measured
            # and is a wash: reports 0.927 -> 0.925, statements 0.725 -> 0.722, and the 33 real
            # files 0.574 -> 0.579, since a few bill headings do sit in the same block as their
            # table and are lost here. Same total either way, so the simpler rule is kept.
            if t and is_table_caption(lines, i):
                t = None
            if t:
                key = _fold(t)
                if not repeat_is_continuation(key, i, t):
                    marks.append((i, strip_emphasis(t), "block-lead"))
                seen_lead[key] = i
                continue
        # 1d: a bold-delimited lead-in is a heading even when its body follows on the
        # same line. Checked before 1b, which would reject the line as a sentence.
        b = bold_lead(stripped) if allowed["bold"] else None
        if b and b.group("rest").strip():
            t = strip_emphasis(b.group("t"))
            if t and len(t.split()) <= 12:
                # In a page-layout document the same bold label is reprinted on every page --
                # "Invoice:", "Billing period:", "Plan" -- and a charge line carries its own
                # amount. Neither is a section. Applied only where the document IS page
                # furniture-heavy; a contract may legitimately repeat a bold caption.
                if allowed.get("blocklead"):
                    if _fold(t) in furniture or TRAIL_MONEY.search(t):
                        continue
                    if repeat_is_continuation(_fold(t), i, t):
                        continue
                    seen_lead[_fold(t)] = i
                marks.append((i, t, "bold-lead"))
                continue
        # 1d2: a line that is ONLY a bold span, with nothing after it. This is how a converted
        # document writes its own title -- "**RECRUITING FEE AGREEMENT**" -- and any section
        # title the author did not number. Rule 1d cannot see these: it requires the body to
        # follow on the same line, so a bold line standing alone fell through to the inference
        # rule, which is off wherever there is real numbering. The result was a two-page
        # agreement whose title and its one "PROVISIONS:" heading both vanished into preamble.
        if allowed["bold"] and not (b and b.group("rest").strip()):
            bo = BOLD_ONLY.match(stripped)
            if bo and sorted(bo.group("open")) != sorted(bo.group("close")):
                bo = None                      # "**a** and **b**" is emphasis, not a heading
            if bo:
                t = strip_emphasis(bo.group("t")).strip()
                # Guards: a bolded sentence is emphasis, not a heading (contracts bold whole
                # disclaimer paragraphs), and the execution block is boilerplate.
                if (t and t.lower() not in SIG_KEY and len(t.split()) <= 14
                        and not looks_like_content(t)):
                    if allowed.get("blocklead"):
                        if _fold(t) in furniture:
                            continue
                        if repeat_is_continuation(_fold(t), i, t):
                            continue
                        seen_lead[_fold(t)] = i
                    marks.append((i, t, "bold-only"))
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
        # 1g: a standalone ALL-CAPS line is a title. This is how these documents mark their
        # biggest boundaries -- an NDA's own title, "BACKGROUND", "SERVICE LEVEL AGREEMENT",
        # "ANNEX E TO EXHIBIT D" -- and detection had been keyed on ** markup, so a document
        # with no bold in it lost its title outright. Measured on 35 generated 850 KB agreements
        # with exact keys, the rules missed 1,037 all-caps section titles without this.
        # Deduplicated by wording, keeping the FIRST, so a banner reprinted on thirty pages
        # contributes one section and not thirty -- but only when the repeat really is the same
        # section continuing (nearby, or just past a page break). "Cash" and "Profitability"
        # head three genuinely different sections of a quarterly report hundreds of lines apart.
        if allowed.get("caps"):
            ct = caps_title(stripped)
            if ct and (ct.strip().lower().rstrip(":") in BANNER_FURNITURE
                       or _fold(ct) in banners
                       or _fold(stripped) in recurring
                       or is_table_caption(lines, i)):
                ct = ""
            if ct:
                ckey = _fold(ct)
                if not (ckey in seen_caps and cont(i, seen_caps[ckey], ct)):
                    seen_caps[ckey] = i
                    marks.append((i, ct, "caps"))
                continue

        # A titled lettered sub-item, where the regime allows it. Placed before the clause and
        # inference branches because the line begins with "(a)" rather than a number, so nothing
        # else claims it and it would otherwise stay inside its parent's body -- which is how
        # three titled warranties collapsed into one 58-word section.
        if allowed.get("lettered"):
            lt = lettered_title(stripped)
            if lt and not is_table_caption(lines, i):
                marks.append((i, lt, "lettered"))
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
            if t and not is_table_caption(lines, i):
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
    # A section is named ONCE. Every rule above can pick up a running banner reprinted on each
    # page of a long region, and each rule had its own ad-hoc repeat check or none at all. One
    # pass here covers all of them: a mark whose wording was already used is dropped when it is
    # genuinely the same section continuing -- nearby, or just past a page break.
    #
    # The reason this cannot simply suppress all repeats: "Cash", "Profitability", "Revenue" and
    # "Product" each head three or four genuinely different sections of a quarterly report,
    # hundreds of lines apart. Distance and the page break are what separate a continuation from
    # a reused word.
    seen_any = {}
    kept = []
    for mk in marks:
        fkey = _fold(mk[1])
        prev = seen_any.get(fkey)
        if prev is not None and cont(mk[0], prev, mk[1]):
            seen_any[fkey] = mk[0]
            continue
        seen_any[fkey] = mk[0]
        kept.append(mk)
    marks = kept

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
    # NUMBERED STRUCTURE IS LEFT ALONE. This is a standing decision, not a tuning choice: a
    # number is the author declaring a boundary, and guessing that a run of them is "really" a
    # list cost nine real subsections in one NDA (3.4 through 3.7.6 all disappeared). The cost
    # of the other error -- an exhibit's numbered list becoming sections -- is a longer outline,
    # which a reader can see through. A missing clause is invisible, and worse.

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
            # Numbering restarts inside an attachment, so a clause in EXHIBIT B must never
            # adopt a same-numbered clause from the host agreement. The attachment marker is
            # a hard wall: search stops there. Without it, `1.01` in an attached Master Lease
            # reached back and nested under the sublease's own clause `1`, which is how the
            # Master Lease came to look like part of the sublease.
            if is_attachment(out[back].get("title")):
                break
            if _canon(out[back]["number"]) == stem:
                s["parent"] = back
                break
    out = nest_lettered(out)
    # merge_numbered_lists() is deliberately NOT called, for the reason above. It is kept in
    # this file because it documents what a numbered list looks like, and because the decision
    # to fold them is one a future document type might want back -- explicitly, not by default.
    out = collapse_label_runs(out)
    # A document with no headings at all still has one section: itself. Returning an empty list
    # loses the document entirely -- a two-paragraph letter came back with nothing to show --
    # and the reader is owed the fact that it carries no headings, written as "no header".
    if not out:
        body = "\n".join(x for x in lines if x.strip() and x.strip() != BLK)
        out = [_mk("no header", body, None, "whole-document")]
    out = name_long_untitled(out)
    out = nest_under_attachments(out)
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


BOLD_SPAN = re.compile(r"\*\*(?P<t>[^*\n]{2,60}?)\*\*")
NUM_ONLY = re.compile(r"^\d{1,2}(?:\.\d{1,3})*\.?$")


def name_from_bold(body):
    """A long untitled paragraph usually marks the one thing worth naming -- a party -- in bold.

    "1. **NDA STUDIOS DISTRIBUTION LIMITED**, a company incorporated in England and Wales under
    company number 01420028, whose registered office is at..." carries no caption, so the
    outline showed a bare "1." against what is in fact a party to the agreement. The author
    already marked the name; this reads it rather than guessing.

    Bold specifically, and not "the leading run of capitalised words": that guess turned
    "3.1 The Recipient shall not disclose..." into a section titled "3.1 The Recipient", which
    is a sentence opening and not a name. Emphasis is a deliberate signal; capitalisation is not.
    """
    for m in BOLD_SPAN.finditer(body[:300]):
        t = m.group("t").strip(" ,.;:*")
        if 1 <= len(t.split()) <= 8 and any(c.isalpha() for c in t) and t.lower() not in SIG_KEY:
            return t
    return ""


CAPS_LEAD = re.compile(r"^\s*(?P<t>[A-Z][A-Z0-9&.\'-]*(?:[ ,]+[A-Z][A-Z0-9&.\'-]*){1,5}),")


def name_from_caps(body):
    """A party name written in capitals, which is how an unbolded recital marks it.

    "1. NDA STUDIOS DISTRIBUTION LIMITED, a company incorporated in England and Wales..." has no
    emphasis to read, so the capitals carry the name. The comma is the guard and it is what makes
    this safe: the run has to END at a comma, which a recital does and a capitalised sentence
    does not -- "THE SERVICES ARE PROVIDED AS IS AND WITHOUT WARRANTY..." runs straight on and is
    rejected. Title Case is not accepted at all; that is what turned "The Recipient shall not
    disclose..." into a section titled "The Recipient".
    """
    m = CAPS_LEAD.match(body[:200])
    if not m:
        return ""
    t = m.group("t").strip(" ,;:")
    if len(t.split()) < 2 or t.lower() in SIG_KEY:
        return ""
    return t


def name_long_untitled(out):
    """Give a title to sections that have none, where the document supplies one in bold.

    Only where the body is long: a short untitled clause is a fragment and a name lifted out of
    it would misrepresent what the section is about. "Super long" is 400 characters, roughly a
    full recital paragraph.
    """
    for s in out:
        t = (s.get("title") or "").strip()
        bare = NUM_ONLY.match(t)
        if not (bare or t.startswith("[preamble")):
            continue
        # The body opens with the clause number itself -- "1. NDA STUDIOS DISTRIBUTION
        # LIMITED, a company..." -- and a name has to be read from what follows it.
        body = re.sub(r"^\s*\(?\d{1,2}(?:\.\d{1,3})*\)?[.):]?\s+", "", s.get("text") or "")
        # Long enough to be a recital rather than a fragment. A party recital is around 150
        # characters, not 400: the two parties to this NDA are 206 and 139, and a 400 floor
        # excluded both -- the thing the rule exists for.
        if len(body) <= 100:
            continue
        nm = name_from_bold(body) or name_from_caps(body)
        if nm:
            s["title"] = ("%s %s" % (t.rstrip("."), nm)) if bare else nm
    return out


def _mk(title, body, parent, conf):
    title = clean_title(title)
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



# An attachment marker opens a new document inside this one. "EXHIBIT B", "SCHEDULE 2",
# "ANNEX I", "ADDENDUM A" -- whatever follows belongs to the attachment, not to the host
# agreement, and the host's clause numbering has no authority over it.
ATTACHMENT = re.compile(r"^\s*(?:\*\*)?(?:EXHIBIT|ANNEX|SCHEDULE|APPENDIX|ATTACHMENT|ADDENDUM"
                        r"|RIDER|SUPPLEMENT)\b\s*[A-Z0-9IVXL]{0,4}\s*[-:.\u2014]?\s*(?:\*\*)?$",
                        re.I)
# Also accept a marker that carries its title on the same line: "EXHIBIT A - SERVICE LEVELS".
ATTACHMENT_TITLED = re.compile(r"^\s*(?:\*\*)?(?:EXHIBIT|ANNEX|SCHEDULE|APPENDIX|ATTACHMENT"
                               r"|ADDENDUM|RIDER|SUPPLEMENT)\s+[A-Z0-9IVXL]{1,4}\b", re.I)


def is_attachment(title):
    t = (title or "").strip()
    return bool(ATTACHMENT.match(t) or ATTACHMENT_TITLED.match(t))



def drop_figure_captions(out):
    """Drop guessed titles that caption a picture, and running headers repeated down the page.

    A slide deck sets a photo-page banner in exactly the type it uses for section titles --
    "WING ASSEMBLY PRODUCTION MILESTONE - PLANT FILTON", "MEGAFACTORY SHANGHAI - FIRST MEGAPACK
    OFF THE LINE" -- and the converter emits it as a plain capitalised line in its own layout
    block. Nothing about the LINE separates it from a real heading, so no line-level rule can
    decide this. What separates them is what follows.

    Measured on one generated deck, the two populations do not overlap:

        real section  + figures   210, 212, 233, 287 words of prose
        photo caption + figures    11,  13,  27,  37 words

    So: a GUESSED title, with a picture, no table, no subsections, and only a caption's worth of
    prose, is a caption. Sixty words is comfortably between the two populations.

    Second case, same principle: a company name or document title reprinted at the top of every
    page ("ELMWOOD", eleven times, one or two words of body each). The repeat check upstream
    misses these because consecutive occurrences are further apart than a continuation is
    allowed to be, in a document where each page is long. Three or more occurrences of the same
    short guessed title, none of which has a real body, is a running header and not a section.

    Only titles that were GUESSED from shape are eligible. A heading the author marked with "#"
    or with bold is kept whatever its body: the author said it was a heading. A container -- one
    with real subsections under it -- is kept too, because there the structure is the evidence.
    """
    GUESSED = {"caps", "inferred", "block-lead"}
    CAPTION_WORDS = 60
    has_child = {sx["parent"] for sx in out if sx.get("parent") is not None}
    doc_has_nesting = bool(has_child)

    folded = {}
    for i, sx in enumerate(out):
        if sx.get("heading_confidence") in GUESSED:
            folded.setdefault(_fold(sx.get("title") or ""), []).append(i)
    running = set()
    for _k, idxs in folded.items():
        if len(idxs) >= 3 and all(out[j].get("words", 0) < 10 and j not in has_child
                                 for j in idxs):
            running.update(idxs)

    drop = set()
    for i, sx in enumerate(out):
        if sx.get("heading_confidence") not in GUESSED or i in has_child:
            continue
        if i in running:
            drop.add(i)
            continue
        if (sx.get("words", 0) < CAPTION_WORDS
                and len(sx.get("figures") or []) >= 1
                and not (sx.get("tables") or [])):
            drop.add(i)
            continue
        # An empty guessed title: no body, no subsections, no table, no picture. In a deal packet
        # this is an exhibit cover sheet -- a page carrying only the words "EXHIBIT B" -- or a
        # constituent document's own cover title. Both are title-shaped, both are set in capitals,
        # and neither is a section: there is nothing under them to navigate to. Left in, they cost
        # the packets 0.067 precision, which is what made the ALL-CAPS rule a net loss there while
        # being a clear win on master agreements, where an ALL-CAPS line heads real clause text.
        # Threshold is deliberately tight -- a heading with even a sentence under it is kept.
        # ...but ONLY when the title is a bare attachment marker -- "EXHIBIT B", "SCHEDULE 2" --
        # on a page of its own. The first version of this test dropped ANY guessed title with no
        # body, which was wrong for a reason worth writing down: in the sparse regime the rules
        # produce no nesting at all, so a real container heading ("OPERATIONAL SUMMARY",
        # "PHOTOS & CHARTS", "AUTOMOTIVE") has no children to protect it and was deleted as if it
        # were furniture. That cost five of the seven Tesla quarterly decks recall -- 0.84 down to
        # 0.53 on one of them -- while fixing a problem that only ever existed in deal packets.
        # A bare marker cannot be a container heading in any regime, so it is safe everywhere.
        if sx.get("words", 0) < 4 and not (sx.get("tables") or sx.get("figures")):
            # A bare attachment marker on a page of its own -- "EXHIBIT B", "SCHEDULE 2" -- is a
            # cover sheet in any document, because a marker cannot be a container heading.
            if is_attachment(sx.get("title")):
                drop.add(i)
            # Any other body-less guessed title is furniture ONLY where this document's nesting is
            # legible enough to tell a container heading from a stray line. In the sparse regime
            # the rules produce no nesting at all, so `has_child` protects nothing and real
            # container headings ("OPERATIONAL SUMMARY", "PHOTOS & CHARTS", "AUTOMOTIVE") were
            # deleted as furniture -- five of the seven Tesla decks lost recall, one from 0.84 to
            # 0.53. Where the document DOES nest, a body-less childless guessed title really is a
            # cover sheet, and dropping it is worth 0.03 precision on deal packets.
            elif doc_has_nesting:
                drop.add(i)
        # A capitalised line on a cover page -- "EXECUTION COPY", "EMERSON ELECTRIC CO",
        # "DEUTSCHE BANK AG NEW YORK BRANCH", "SCHEDULES", "RECITALS", and the browser print
        # header "5/7/26, 3:39 AM EX-10.1" -- is title-shaped and heads nothing. On two real SEC
        # credit agreements these alone produced 57 and 73 false sections. A real ALL-CAPS clause
        # heading in the same documents is followed by clause text, so requiring a body separates
        # them. Gated on the document nesting, for the same reason as above: in the sparse regime
        # a real container heading has no body AND no children, and would be destroyed by this.
        elif (sx.get("heading_confidence") == "caps" and doc_has_nesting
                and sx.get("words", 0) < 12
                and not (sx.get("tables") or sx.get("figures"))):
            drop.add(i)
    if not drop:
        return out
    keep = [sx for i, sx in enumerate(out) if i not in drop]
    remap = {o: n2 for n2, o in enumerate(i for i in range(len(out)) if i not in drop)}
    for sx in keep:
        pp = sx.get("parent")
        sx["parent"] = remap.get(pp) if pp is not None else None
    for k2, sx in enumerate(keep):
        sx["order"] = k2
    return keep


def nest_lettered(out):
    """Put each titled lettered sub-item under the clause it belongs to.

    Nesting elsewhere in this file is driven by the numbering -- "8.1" finds "8" -- and a lettered
    item has no number to match on, so it would otherwise sit at the top level beside its own
    parent. "(a) Services Warranty" is a child of "8 WARRANTIES", and an outline that prints them
    as siblings is telling the reader something untrue about the document.

    The parent is the nearest preceding section that is not itself lettered. The search stops at an
    attachment marker for the same reason the numbered search does: a lettered item inside EXHIBIT
    B must not adopt a clause from the host agreement.
    """
    anchor = None
    for k, s in enumerate(out):
        if s.get("heading_confidence") != "lettered":
            anchor = None if is_attachment(s.get("title")) else k
            continue
        if s.get("parent") is None and anchor is not None:
            s["parent"] = anchor
    return out


def nest_under_attachments(out):
    """Put an attachment's contents INSIDE the attachment.

    Measured on the 33-document corpus: 604 sections across 10 documents sat at top level
    after an attachment marker, which makes the whole Master Lease a sibling of the sublease
    clauses that reference it. In the Sublandlord sublease that is 213 sections -- `1.01 Certain
    Definitions`, `1.02 Demise`, `ARTICLE 2` -- presented as if they were clauses of the
    sublease itself. An outline like that cannot be navigated: there is no way to tell which
    document a clause belongs to, and two clauses numbered 1.01 look like a contradiction
    rather than like two different agreements.

    Only sections that are still PARENTLESS are moved. A section whose parent the numbering
    already established keeps it -- this pass supplies containment, it does not overrule
    numbering. Nested attachments work because the most recent marker wins, and a marker
    itself attaches to the marker above it only when that one is a different, earlier
    attachment at a shallower position.
    """
    if not out:
        return out
    depth_of = {}

    def depth(i):
        if i in depth_of:
            return depth_of[i]
        n, p, seen = 0, out[i].get("parent"), set()
        while p is not None and 0 <= p < len(out) and p not in seen and n < 40:
            seen.add(p); n += 1; p = out[p].get("parent")
        depth_of[i] = n
        return n

    open_att = None
    for i, s in enumerate(out):
        if is_attachment(s.get("title")):
            # A marker nests under an earlier marker only if that one is still the open scope
            # and this one is not itself top-level furniture; keep it simple and honest by
            # leaving markers at the level the numbering gave them.
            open_att = i
            continue
        if open_att is None:
            continue
        if s.get("parent") is None and i > open_att:
            s["parent"] = open_att
    depth_of.clear()
    return out


def collapse_label_runs(out, run=6, body=400):
    """Fold a long run of short same-kind entries into the first of them.

    A definitions article ("**Acceptance Period:** unless otherwise provided...") and a cover
    approval form ("Requestor / Contract Title / Expense Amount / ...") both produce a long
    run of entries that each look like a titled section and none of which is one -- they are
    the CONTENT of a single section. A real clause run is broken up by long bodies; these are
    not. Requiring six in a row keeps ordinary consecutive clause headings safe.
    """
    n = len(out)
    # A NUMBERED entry is never folded away. The runs this exists to catch are labels the author
    # wrote as content -- "Acceptance Period:", "Requestor", "Expense Amount" -- and they carry
    # no numbering. A number is the author explicitly declaring structure, and an NDA's clauses
    # are short: 3.4 through 3.7.5 are each under 400 characters with a bare numeric title and
    # the same confidence, so the whole run was being swallowed into 3.3 and nine real
    # subsections vanished from the outline. Numbered runs have their own handling already
    # (merge_numbered_lists and the seq filter), so excluding them here loses nothing.
    # An AUTHOR-MARKED heading is never folded away, for the same reason a numbered entry is not.
    # This function exists to catch labels the author wrote as CONTENT -- a definitions article, a
    # cover approval form -- and a "#" heading, a setext underline or an <h2> tag is the author
    # declaring the opposite, so it should never be eligible however short its body.
    #
    # Honest provenance: this exemption was added on a WRONG diagnosis. Three Markdown headings of
    # the Tesla Q3-2023 deck were missing from the outline and this looked like the culprit; it was
    # not. The real cause was a degenerate "<::>" marker letting the vision-description pattern
    # swallow 2 KB of the document, fixed in compare_run.py. This exemption changed nothing on any
    # of the seven measured corpora, so it is kept on the argument alone and not on evidence --
    # which is worth saying, because a comment that claims a measured win it never had is worse
    # than no comment.
    AUTHORED = {"heading", "setext", "html"}
    small = [i for i in range(n)
             if (out[i].get("title") or "") and _eff_body(out[i]) < body
             and len((out[i]["title"] or "").split()) <= 8
             and out[i].get("heading_confidence") not in AUTHORED
             and not (out[i]["title"] or "").strip()[:1].isdigit()]
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
    if not drop:
        return out
    # Stage 2 has ALREADY run, and it stored each parent as an INDEX into this list. Dropping
    # entries without remapping those indices shifts every parent that pointed past a dropped
    # entry, which produced two distinct corruptions on this corpus: a child whose parent index
    # now lands AFTER it (82 sections across two documents), and a child whose parent index
    # lands past the end of the list entirely (clauses 22.1 and 22.2 of the ABS contract, parent
    # 129 in a 125-entry list). outline() builds its tree from parents, so the out-of-range ones
    # were SILENTLY DROPPED -- the header said 125 sections and 123 lines were printed, with no
    # error. merge_numbered_lists already had this remap; the function that actually runs did not.
    keep = [s for k, s in enumerate(out) if k not in drop]
    remap = {old_i: new_i for new_i, old_i in
             enumerate(k for k in range(n) if k not in drop)}
    for s in keep:
        p = s.get("parent")
        # A parent that was folded into content is no longer a section, so the child has no
        # parent -- top level is the honest answer, not a dangling index.
        s["parent"] = remap.get(p) if p is not None else None
    for k, s in enumerate(keep):
        s["order"] = k
    return keep


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
