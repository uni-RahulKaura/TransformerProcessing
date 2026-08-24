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
    transfer.
"""
import re

_PERM = re.compile(r"\b(?:may|shall\s+not|must\s+not|is\s+(?:not\s+)?permitted|"
                   r"is\s+entitled|prohibited|consent)\b", re.I)
_DATE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
                   r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
                   r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b")
_TERMWORD = re.compile(r"\b(?:expir\w+|terminat\w+|renew\w+|effective\s+date"
                       r"|\d+\s*(?:day|week|month|year)s?)\b", re.I)
# Role nouns. The original five were tuned on supplier agreements, so an NDA's parties -- who
# are the Recipient and the Disclosing Party throughout -- and a lease's Landlord and Subtenant
# went untagged in the documents that are ABOUT them.
_ROLE = re.compile(r"\b(?:Supplier|Customer|Manufacturer|Buyer|Seller|Licensor|Licensee"
                   r"|Vendor|Contractor|Client|Purchaser|Consultant|Provider"
                   r"|Recipient|Disclosing\s+Party|Receiving\s+Party|Discloser"
                   r"|Landlord|Tenant|Sublandlord|Subtenant|Sublessor|Sublessee)\b")
# The corporate suffix is matched case-insensitively, the NAME is not. A signed agreement writes
# its parties in capitals -- "NDA STUDIOS DISTRIBUTION LIMITED", "ACTIONIQ, INC." -- and against
# a case-sensitive "Inc|Ltd|Limited" the party recitals of an NDA carried no PARTY label at all.
# Scoped to the suffix group so the name still has to be capitalised; making the whole pattern
# case-insensitive would match any six words followed by the word "company".
_CORP = re.compile(r"\b[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,5}"
                   r"\s*,?\s*(?i:Inc|Corp|Corporation|Ltd|Limited|LLC|LC|GmbH|AG|plc)\.?\b")
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
