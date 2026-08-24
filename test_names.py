#!/usr/bin/env python
"""Regression suite for the figure/date/name support check. Run: python test_names.py

The false-alarm cases are the point. This check decides whether a section publishes a rewrite or
falls back to a quotation, so every false alarm costs exactly what the work was for. Each case
below is drawn from real output across the 33-document corpus; the counts in the comments are how
many sections that class demoted before it was fixed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.safe_abstractive import unsupported                  # noqa: E402

# (name, summary, section, should_flag)
CASES = [
    # ---- must NOT flag: form, not fact ----
    ("'The' added to a generic term (x4)",
     "The Agreement shall continue for two years.",
     "This Agreement shall continue for two years.", False),
    ("'The' added to a defined term",
     "The Fee Agreement governs the recruiting fee.",
     "The parties signed a Fee Agreement covering the recruiting fee.", False),
    ("a pronoun is not a party",
     "They shall keep the information confidential.",
     "The parties shall keep the information confidential.", False),
    ("a connective is not a party (x3)",
     "Neither party shall limit liability for fraud.",
     "Nothing in this clause limits liability for fraud.", False),
    ("singular of a defined plural",
     "Authorised Third Party means employees and agents.",
     "\"Authorised Third Parties\" means the employees, officers and agents of the Recipient.",
     False),
    ("plural of a defined singular",
     "The Recipient shall notify the Disclosing Parties.",
     "the Recipient shall notify the Disclosing Party in writing.", False),
    ("spacing inside a defined term (x5)",
     "The Sub Tenant shall pay rent.",
     "The Subtenant shall pay all rent due under the Sublease.", False),
    ("hyphen read as a space",
     "The Mutual Non disclosure terms apply.",
     "This Mutual Non-Disclosure Agreement sets out the terms.", False),
    ("a possessive form",
     "TESLA's revenue rose.", "TESLA reported revenue growth.", False),

    ("a table of contents is not a party name (x3)",
     "Highlights Financial Summary Operations Summary Vehicle Capacity Core Technology Other "
     "Highlights Outlook Photos follow.",
     "Highlights. Financial Summary. Operations Summary. Vehicle Capacity. Core Technology. "
     "Outlook. Photos.", False),
    ("'The' added to a product name",
     "The Megapack deployments increased.",
     "Megapack deployments increased year over year.", False),

    # ---- MUST flag: the section does not say this ----
    ("a misspelled place name",
     "Billed to Meckslenburg Cnty.", "Billed to Mecklenburg Cnty.", True),
    ("a misspelled defined term",
     "The Subtenent shall pay rent.", "The Subtenant shall pay all rent due.", True),
    ("a misspelled ordinary word in caps",
     "Damages are CONSEQUENTAL in nature.",
     "Damages that are consequential in nature are excluded.", True),
    ("a company the section never names",
     "The fee is payable to Acme Holdings.", "The fee is payable monthly.", True),
    ("a decimal corrupted into a thousands separator",
     "The amount due is $55.000.", "The amount due is $55,000.", True),
    ("a figure absent from the section",
     "Interest accrues at 4.5% per month.", "Interest accrues at 1.5% per month.", True),
    ("a short invented company still caught",
     "Revenue was reported by Acme Motors Holdings.",
     "Revenue was reported for the quarter.", True),
]


def main():
    bad = 0
    for name, summary, section, want in CASES:
        got = bool(unsupported(summary, section))
        ok = got == want
        bad += not ok
        tokens = [x["token"] for x in unsupported(summary, section)]
        print("%s  %-40s %s" % ("pass" if ok else "FAIL", name, tokens or "clean"))
    print("\n%d of %d cases pass" % (len(CASES) - bad, len(CASES)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
