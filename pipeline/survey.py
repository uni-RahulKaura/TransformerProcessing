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
BOLD_ONLY = re.compile(r"(?m)^\s*\*\*([^*\n]{3,90})\*\*\s*:?\s*$")
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
        "bold_only": len(BOLD_ONLY.findall(md)),
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

    # ---- pick the regime -----------------------------------------------------------
    # Clause numbering is the strongest signal there is: it is explicit, the document author
    # put it there, and it survives conversion. If there is a lot of it, use it.
    if s["clauses"] >= 20 or s["articles"] >= 5:
        s["regime"] = "clauses"
    # Real Markdown headings, and enough of them for their number to plausibly BE the
    # backbone. Density is the test, not the raw count: 7 headings in a 4 KB memo is the
    # skeleton of the document, 7 in a 90 KB slide deck is incidental -- the deck's real
    # titles are set in large type, which conversion renders as a bare line, not a "#".
    # Requiring one heading per ~6 KB separates the two.
    elif s["md_headings"] >= 5 and s["headings_per_kb"] >= 0.16:
        s["regime"] = "headings"
    # Headings exist but are too sparse to be the whole story. Trust them, and additionally
    # read the first line of each layout block, which is where an unmarked title lives.
    elif s["md_headings"] >= 2:
        s["regime"] = "sparse"
    # Some numbering, few headings: a short agreement like an NDA. Numbering is all we have.
    elif s["clauses"] >= 5:
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
    "clauses":  {"md": True, "bold": True, "clause": True, "decimal": True, "infer": False, "label": False, "blocklead": False},
    "headings": {"md": True, "bold": True, "clause": False, "decimal": False, "infer": False, "label": False, "blocklead": False},
    "sparse":   {"md": True, "bold": True, "clause": False, "decimal": False, "infer": False,
                 "label": True, "blocklead": True},
    "fields":   {"md": True, "bold": False, "clause": False, "decimal": False, "infer": False, "label": False, "blocklead": False},
    "labels":   {"md": True, "bold": True, "clause": False, "decimal": False, "infer": False,
                 "label": True, "blocklead": False},
    "letter":   {"md": True, "bold": False, "clause": False, "decimal": False, "infer": False,
                 "label": False, "blocklead": False},
    "flat":     {"md": True, "bold": True, "clause": True, "decimal": True, "infer": True, "label": True, "blocklead": False},
}
