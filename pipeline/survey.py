"""Decide what actually carries a document's structure, before trying to read it.

This exists because the previous approach applied every heading rule to every document, and
the rules had been written for one kind of document -- a contract with hundreds of numbered
clauses. Pointed at a phone bill, the "short titled line" rule fired on every field label,
address line and account number, and produced 110 sections for a document with about a dozen.

So: look first. Count the signals the document genuinely has, pick the regime, and apply only
the rules that belong to it. A rule that is right for a lease is wrong for a statement, and no
amount of tuning a single rule set fixes that.

Measured signal counts on the 33-document corpus that motivated this:

    document          # headings   numbered clauses   what we produced   what it has
    phone bill            12              3               110            ~a dozen
    Tesla report           8              0                66            ~a dozen
    MSA                   23             75               214            ~80
    sublease              59            362               705            ~400

The bills and the financial reports have almost no clause numbering. The lease has hundreds.
Treating those the same was the whole mistake.
"""
import re

MD_HEADING = re.compile(r"(?m)^(#{1,6})\s+\S")
# A clause line: optionally bold, a number possibly with decimals, then a capitalised word.
CLAUSE = re.compile(r"(?m)^\s*(?:\*\*)?(\d{1,2}(?:\.\d{1,3})*)\.?\s+(?=(?:\*\*)?[A-Z\"(])")
# Anchored on any emphasis run, not just "**". Markdown writes doubled and bold-italic emphasis
# as __x__, __**x**__, ***x***, ___x___, and the CLOSING run mirrors the opening one rather than
# copying it. Keying on "**" alone made a whole style of document unreadable: agreements that
# write every section title as __**Section 1.1 "Affiliate"**__ registered zero bold signal here
# (so the regime was chosen on the wrong evidence) and matched no heading rule there. On one
# 856 KB agreement that was 816 real sections found as 45.
BOLD_ONLY = re.compile(r"(?m)^\s*(?P<open>[*_]{2,6})(?P<t>[^*_\n][^\n]{2,88}?)(?P<close>[*_]{2,6})\s*:?\s*$")
ARTICLE = re.compile(r"(?m)^\s*(?:\*\*)?(?:ARTICLE|Article|SECTION|Section|ANNEX|EXHIBIT|SCHEDULE"
                     r"|APPENDIX)\s+([0-9IVXLC]+)", re.M)
# A label/value line is the signature of a form: a short label, a colon, then a value.
FIELD = re.compile(r"(?m)^\s*([A-Z][A-Za-z /'&.-]{1,34}):\s*\S")
# ...except the execution block, which is boilerplate in EVERY signed contract and says
# nothing about how the body is organised. Counting it made a 5 KB contract amendment -- whose
# 16 "fields" were all DocuSign furniture -- look like a form, which switched off every rule
# that could have found its real headings and produced an outline with zero sections.
SIG_KEY = {"signature", "docusigned by", "by", "name", "title", "date", "witness", "attest",
           "printed name", "print name", "its", "signed", "email", "address", "docusign"
           " envelope id", "envelope id", "acknowledged", "agreed", "accepted"}
# A topic label: a line that is nothing but a short title ending in a colon. This is how
# amendments and letters mark their sections when they have no numbering at all.
LABEL_ONLY = re.compile(r"(?m)^\s*(?:\*\*)?([A-Z0-9][^:*\n]{2,86}):(?:\*\*)?\s*$")


def _real_fields(md):
    return [m for m in FIELD.finditer(md) if m.group(1).strip().lower() not in SIG_KEY]


# How many times the document starts a top-level "1." A contract numbers its clauses once and
# runs up; a lease's exhibits each restart their own list at 1. Counting the restarts tells the
# two apart without knowing anything about either document.
RESTART = re.compile(r"(?m)^\s*(?:\*\*)?1\.\s+[A-Z\"(]")


def _clause_cover(md):
    """How much of its own range the clause numbering actually covers, 0..1.

    Counting numbered lines is not enough, and the count is what routed 23 of the 35 generated
    financial reports into the clause regime. A 400 KB quarterly report contains 20-58 lines that
    open with a number and a capital -- but they are TABLE ROW NUMBERS and FOOTNOTE MARKERS
    ("10.", "54.", "13. (1) Trailing twelve months"), scattered over the document with no relation
    to each other. An agreement's clause numbers are a SERIES: it has a clause 1, then 2, then 3,
    and the distinct top-level numbers it uses cover nearly the whole span from its smallest to
    its largest.

    Coverage measures exactly that -- distinct top-level numbers divided by the largest one seen.
    Measured over the 175 generated documents plus the 33 real ones:

        financial reports            0.14 - 0.32   (24 documents, none above 0.32)
        master agreements            0.40 - 1.00   (the sparsest, doc-13, is 0.84)
        statements with numbering    0.77 - 1.00
        deal packets                 0.49 - 1.03
        correspondence               0.27 - 1.00

    Correspondence is the one range that reaches down into the reports', and it is worth being
    precise about rather than claiming a clean split: six of those files sit at 0.27 because each
    is a bundle of purchase orders that all restart at 1. Coverage alone would misread them. It
    does not have to carry them -- every one also runs 39 to 141 "ARTICLE n" lines and 0.24 to
    0.59 clauses per KB, so both other halves of the spine test hold them in the clause regime
    before coverage is consulted. What coverage is relied on for is the opposite case: a document
    with neither density nor an article backbone, where the question is whether its handful of
    numbers is a list or a table.

    Scattered row numbers cannot fake it: to reach 0.5 a document would have to use half of every
    integer up to its highest, which is what a clause list does and a column of row numbers,
    drawn from wherever the tables happen to fall, does not.
    """
    tops = [int(m.group(1).split(".")[0]) for m in CLAUSE.finditer(md)]
    if not tops:
        return 0.0
    distinct = set(tops)
    return len(distinct) / max(distinct)


# A letter: a salutation and a complimentary close. Both, because either alone appears in other
# document types -- a contract recital can open "Dear" in a quoted exhibit, and "Sincerely"
# turns up inside attached correspondence.
SALUTATION = re.compile(r"(?m)^\s*(?:\*\*)?Dear\s+[A-Z]", re.M)
CLOSING = re.compile(r"(?m)^\s*(?:\*\*)?(?:Very truly yours|Sincerely(?: yours)?|"
                     r"Best regards|Kind regards|Yours (?:truly|faithfully|sincerely)|"
                     r"Respectfully(?: yours)?|Regards)\s*,?\s*(?:\*\*)?$", re.I | re.M)


def survey(md):
    """Count the structure signals present, and name the regime they imply."""
    s = {
        "chars": len(md),
        "md_headings": len(MD_HEADING.findall(md)),
        "clauses": len(CLAUSE.findall(md)),
        "bold_only": sum(1 for m in BOLD_ONLY.finditer(md)
                         if sorted(m.group("open")) == sorted(m.group("close"))),
        "articles": len(ARTICLE.findall(md)),
        "fields": len(_real_fields(md)),
        "labels": len(LABEL_ONLY.findall(md)),
        "restarts": len(RESTART.findall(md)),
        "salutation": len(SALUTATION.findall(md)),
        "closing": len(CLOSING.findall(md)),
    }
    # Density matters, not just count: 8 headings in a 90 KB report is a real backbone, while
    # 8 headings in an 850 KB lease is not.
    kb = max(1.0, s["chars"] / 1000.0)
    s["clauses_per_kb"] = round(s["clauses"] / kb, 2)
    s["headings_per_kb"] = round(s["md_headings"] / kb, 2)
    s["articles_per_kb"] = round(s["articles"] / kb, 3)
    s["clause_cover"] = round(_clause_cover(md), 2)

    # ---- pick the regime -----------------------------------------------------------
    # Clause numbering is the strongest signal there is: it is explicit, the document author put
    # it there, and it survives conversion. But COUNTING numbered lines is not enough, and it was
    # the routing bug that made financial reports the worst class in the corpus: 23 of the 35
    # generated reports came in here on 14-58 numbered lines that are table row numbers and
    # footnote markers, not clauses. The clause and decimal rules then read them as structure and
    # produced 406 false sections at a precision of 0.038.
    #
    # Numbering is a spine when it is DENSE (the same density argument the heading branch below
    # already makes -- 21 numbers in 400 KB is 0.05/KB and is nothing) or when it is a SERIES
    # (see _clause_cover). Either is enough on its own: a sparse but complete clause list is real
    # (master agreement doc-13, 0.04/KB but coverage 0.84) and so is a dense one whose exhibits
    # reuse numbers (deal packet doc-22, coverage 0.49 but 0.64/KB).
    clause_spine = s["clauses_per_kb"] >= 0.15 or s["clause_cover"] >= 0.5
    # "ARTICLE 6" carries the same argument. 5 of them in a 391 KB report is a converted table of
    # contents; a real article backbone runs at 0.04/KB and up (the sparsest agreement or
    # statement that depends on this branch is 0.039). Report doc-28 sits at 0.013.
    article_spine = s["articles"] >= 5 and s["articles_per_kb"] >= 0.02
    if (s["clauses"] >= 20 and clause_spine) or article_spine:
        s["regime"] = "clauses"
    # Numbering exists in quantity but is not a spine: it is neither dense nor a series. That is
    # the signature of a financial report or slide deck, whose numbered lines are table rows and
    # footnote marks. Such a document's titles are set in large type and arrive as plain lines, so
    # read layout blocks and capitals and leave the numbering alone. Worth +0.13 F1 on the 23
    # reports this catches; no document in any other class reaches this branch.
    elif s["clauses"] >= 20 or s["articles"] >= 5:
        s["regime"] = "deck"
    # Real Markdown headings, and enough of them for their number to plausibly BE the
    # backbone. Density is the test, not the raw count: 7 headings in a 4 KB memo is the
    # skeleton of the document, 7 in a 90 KB slide deck is incidental -- the deck's real
    # titles are set in large type, which conversion renders as a bare line, not a "#".
    # Requiring one heading per ~6 KB separates the two.
    elif s["md_headings"] >= 5 and s["headings_per_kb"] >= 0.16:
        s["regime"] = "headings"
    # Numbering that is not a spine, again, and it belongs HERE rather than below the sparse
    # branch: whether a document's numbers are table rows has nothing to do with how many "#"
    # headings it happens to carry, so the test should not be reachable only when there are none.
    # The same coverage measure separates the two populations among documents with a few Markdown
    # headings just as cleanly as it did among those with none:
    #
    #     converted deck or report   5-18 numbered lines, coverage 0.15 - 0.37
    #     telecom or utility bill    1-3  numbered lines, coverage 0.67 - 1.00
    #
    # It moves 7 more generated reports out of "sparse" (+0.05 F1 on the class) and, in the 33
    # real files, the two Tesla quarterly decks that have enough row numbers to reach it -- which
    # is what the regime is for. One statement, doc-20, moves too at coverage 0.47.
    elif s["clauses"] >= 5 and not clause_spine:
        s["regime"] = "deck"
    # Headings exist but are too sparse to be the whole story. Trust them, and additionally
    # read the first line of each layout block, which is where an unmarked title lives.
    elif s["md_headings"] >= 2:
        s["regime"] = "sparse"
    # Some numbering, few headings: a short agreement like an NDA. Numbering is all we have.
    # It is necessarily a spine by now -- the branch above took the documents whose numbering
    # is not.
    elif s["clauses"] >= 5:
        s["regime"] = "clauses"
    # Standalone bold-emphasis lines ARE a document's structure when it has nothing else. This
    # branch has to sit AHEAD of the form test, because a signed agreement often opens with a
    # form-like block -- a contract request summary, an order-form header, a routing table -- and
    # that block's Label: Value lines were enough to make the whole document look like a form.
    #
    # The failure was total where it happened, and invisible until a document was small enough to
    # read end to end: a 53 KB master services agreement with 47 bold section headings and an
    # 8-line request-summary header came out with TWO sections against an independent reader's 72,
    # F1 0.000, because the "fields" regime switches off bold, clause, decimal and caps together.
    # The same agreement at 850 KB routed correctly, by luck: at that size enough of its clauses
    # happened to begin with a digit to clear the clause threshold. Both the old rules and the new
    # ones failed this way.
    #
    # A form does not bold-wrap its field labels, so the two shapes separate cleanly: statements
    # and forms in this corpus score bold_only 2, while agreements of this style score 29-530.
    # ...and the bold structure has to OUTWEIGH the form structure, or this branch swallows
    # statements. A utility bill bolds its charge-panel titles too, so bold_only alone is not
    # enough: the generated statements score bold_only 16-99 against fields 45-123, while these
    # agreements score 47 against 20 and 29 against 27. Requiring bold_only > fields keeps both
    # agreements and returns five of six statements to the regime that handles them, which is
    # worth 0.02 F1 on the 35 exactly-keyed statements.
    elif s["bold_only"] >= 8 and s["bold_only"] > s["fields"]:
        s["regime"] = "clauses"
    # Mostly label/value pairs and no numbering: a form or a statement, not a sectioned
    # document. Emit what few headings exist and nothing else.
    elif s["fields"] >= 10:
        s["regime"] = "fields"
    # No numbering, no headings, no fields -- but colon-terminated topic labels. An amendment
    # or a letter. Those labels are the only structure the author gave, so use them.
    elif s["labels"] >= 3:
        s["regime"] = "labels"
    # A letter is prose addressed to someone. It has no sections, and inferring them from line
    # shape turned a two-paragraph waiver letter into fifteen "sections": the law firm's name,
    # each city in its letterhead, the date, the salutation, the closing and each signatory.
    elif s["salutation"] and s["closing"]:
        s["regime"] = "letter"
    else:
        s["regime"] = "flat"

    # A regime that can find nothing is worse than no regime. If the chosen one has no signal
    # to work with, fall back to "flat", which infers from shape.
    if s["regime"] == "fields" and s["md_headings"] == 0 and s["labels"] < 3:
        s["regime"] = "flat"
    return s


# Which rules each regime is allowed to use. "infer" is the dangerous one -- promoting a short
# title-shaped line with no other evidence -- and it is off everywhere except short documents
# that have no headings and no numbering to go on.
RULES = {
    # label stays OFF for long clause documents; sections.py turns it on for short ones.
    # "lettered" promotes ONLY titled lettered sub-items -- see lettered_title() in sections.py.
    # On for clause-numbered agreements, where drafters title sub-clauses and defined terms;
    # off elsewhere, because a statement's "(a) costs of complying with regulatory obligations"
    # is a sentence fragment in a fee footnote, not a section.
    "clauses":  {"md": True, "bold": True, "clause": True, "decimal": True, "infer": False, "label": False, "blocklead": False, "caps": True, "lettered": True},
    # caps stays OFF here. This regime is chosen precisely BECAUSE the document already has enough
    # real Markdown headings to be its own backbone, so guessing further titles from capitalisation
    # adds noise and nothing else. On a real National Grid electricity bill (which has "#" headings
    # and therefore lands here) it contributed 12 false sections against an independent reader's
    # key of 16 and cost a true one: F1 0.848 with it on, 0.812 with it off.
    "headings": {"md": True, "bold": True, "clause": False, "decimal": False, "infer": False, "label": False, "blocklead": False, "caps": False, "lettered": True},
    # A financial report or slide deck: no numbering spine, and its titles were set in large type
    # rather than marked up. So read what the layout gives -- the first line of each block -- and
    # capitals, and never touch the numbering, which here is table rows and footnote marks.
    # It carries two flags of its own, "banner" and "repeats", because the same wording repeated
    # down a page means something different here than in a form: see caps_banners() and
    # repeat_is_continuation() in sections.py.
    #
    # Every entry was ablated against the exact keys for the 35 generated reports. Removing one
    # rule at a time from this set:
    #
    #     the set as it stands   0.927
    #     without blocklead     0.703      without banner    0.732
    #     without repeats       0.777      without bold      0.882
    #     without caps          0.897      WITH label        0.927   (no effect at all)
    #
    # label is off because it changes nothing -- by the time that branch is reached, block-lead,
    # bold and caps have already read every colon-terminated title -- and off is the narrower
    # choice. clause and decimal are the rules the regime exists to switch off: in the clause
    # regime they produced 406 and 72 false sections on these documents at a precision of 0.038
    # and 0.395, and switching them back on here measures 0.658, against 0.927 without them.
    "deck":     {"md": True, "bold": True, "clause": False, "decimal": False, "infer": False,
                 "label": False, "blocklead": True, "caps": True, "banner": True,
                 "repeats": True},
    # caps stays OFF here. A statement or a slide deck sets field labels, table headers,
    # remittance furniture and margin notes in capitals, so the rule fires on content: on the
    # seven telecom bills it added 43 false sections against an independent reader's key and
    # cost recall as well, by consuming lines other rules would have read. On clause-numbered
    # agreements the same rule is worth +0.03 recall. Regime, not tuning, is the difference.
    "sparse":   {"md": True, "bold": True, "clause": False, "decimal": False, "infer": False,
                 "label": True, "blocklead": True, "caps": False},
    "fields":   {"md": True, "bold": False, "clause": False, "decimal": False, "infer": False, "label": False, "blocklead": False, "caps": False},
    "labels":   {"md": True, "bold": True, "clause": False, "decimal": False, "infer": False,
                 "label": True, "blocklead": False, "caps": False},
    # A letter has no sections; capitals in one are a letterhead, so caps stays OFF.
    "letter":   {"md": True, "bold": False, "clause": False, "decimal": False, "infer": False,
                 "label": False, "blocklead": False, "caps": False},
    "flat":     {"md": True, "bold": True, "clause": True, "decimal": True, "infer": True, "label": True, "blocklead": False, "caps": True},
}
