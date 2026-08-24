#!/usr/bin/env python
"""Regression suite for the polarity guard. Run: python test_polarity.py

Every case marked "NDA" is a real section of the Broadcaster Ltd NDA, six of which the guard got
wrong at some point during development -- one true defect it was built to catch and five false
accusations it made on the way. They are here so a later change cannot quietly reintroduce any
of them.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.polarity import flips, depolarise      # noqa: E402

CASES = [
    ("NDA 3.1 -- the defect this exists for",
     "Save as set out below, the Recipient shall be responsible for any breach of this "
     "Agreement by any of its Authorised Third Parties. The Recipient shall not disclose any "
     "Confidential Information.",
     "The Recipient is not responsible for any breach of the Agreement.", True),

    ("NDA 3.3 -- negation re-attached, meaning kept",
     "The Recipient shall use the Disclosing Party's Confidential Information solely for the "
     "purposes of the Project and not in connection with any other transaction, or in any way "
     "that is inconsistent with this Agreement.",
     "Recipient shall not use Confidential Information in any way inconsistent with this "
     "Agreement.", False),

    ("NDA 3.6 -- cue separated by a parenthetical",
     "The Recipient shall not (except as permitted by law) reverse engineer, decompile or "
     "disassemble any software disclosed by the Disclosing Party.",
     "Recipient shall not reverse engineer, decompile, or disassemble software disclosed by "
     "the Disclosed Party.", False),

    ("NDA 10.5 -- cue eight tokens from its target",
     "Nothing in this Agreement shall be construed to oblige the parties to conclude any "
     "further agreement with one another, or to oblige the parties to furnish one another with "
     "any more Confidential Information than that provided under this Agreement.",
     "The parties are not obliged to provide any more confidential information than provided "
     "under this Agreement.", False),

    ("NDA 7 -- 'without limitation' is not a negation",
     "The Recipient acknowledges that damages alone would not be an adequate remedy for any "
     "breach of this Agreement. Accordingly, without prejudice to any other rights and "
     "remedies it may have, the Disclosing Party shall be entitled to seek equitable relief "
     "(including, without limitation, injunctive relief) to prevent any threatened or actual "
     "breach of this Agreement.",
     "The Recipient shall be entitled to seek equitable relief (including injunctive relief) "
     "to prevent a breach.", False),

    ("NDA 10.3 -- 'except' turns a negation into a requirement",
     "This Agreement may not be modified or amended except in writing signed by a duly "
     "authorised representative of each party.",
     "It must be signed by a duly authorised representative of each party.", False),

    ("mirror direction -- a negation the summary drops",
     "The Provider shall not be liable for indirect damages under any circumstances.",
     "The Provider shall be liable for indirect damages.", True),

    ("negation copied through unchanged",
     "The Recipient shall not disclose the Confidential Information.",
     "The Recipient shall not disclose the Confidential Information.", False),

    ("negation with no cue anywhere in the source",
     "Payment is due within 30 days of invoice. Late amounts accrue interest at 1.5% per month.",
     "Payment is not due within 30 days of invoice.", True),

    ("modal swapped, polarity intact",
     "The Tenant shall not assign the Lease without the Landlord's prior written consent.",
     "The Tenant may not assign the Lease without consent.", False),

    ("'not less than' is a threshold, not a prohibition",
     "The Supplier shall maintain insurance of not less than $2,000,000 per occurrence.",
     "The Supplier shall maintain insurance of not less than $2,000,000 per occurrence.", False),

    ("Gap SOW -- a noun is not a polarity target",
     "Gap will not tolerate harassment. Gap maintains a Zero Means Zero policy and no policy "
     "shall be waived.",
     "Gap has a Zero Means Zero policy against harassment and discrimination.", False),

    ("Retailer SOW -- a noun is not a polarity target",
     "Supplier will maintain accurate accounting records. No payroll record shall be withheld.",
     "Supplier will maintain accurate accounting records including payroll.", False),

    ("a copula's complement IS tested",
     "Nothing herein shall be construed to oblige the parties to conclude any agreement.",
     "The parties are obliged to conclude an agreement.", True),

    ("AT&T MSA -- \"No.\" before a number means number",
     "This is Agreement No. 30066282 between Vendor Inc, Inc. and AT&T Services, Inc.",
     "Agreement No. 30066282 between Vendor Inc, Inc. and AT&T Services.", False),

    ("either -> neither",
     "Either party may terminate this Agreement on 30 days written notice.",
     "Neither party may terminate this Agreement on 30 days notice.", True),
]


def main():
    bad = 0
    for name, src, cand, want in CASES:
        got = bool(flips(cand, src))
        ok = got == want
        bad += not ok
        print("%s  %-46s %s" % ("pass" if ok else "FAIL", name,
                                "flagged" if got else "clean"))
    # dropping must leave the rest of the rewrite intact, not fall back to a quotation
    kept, rec = depolarise(
        "Recipient shall not disclose Confidential Information to Authorised Third Parties. "
        "The Recipient is not responsible for any breach of the Agreement.", CASES[0][1])
    if "not disclose" not in kept or "not responsible" in kept:
        print("FAIL  drop keeps the good sentence and removes the bad one")
        bad += 1
    else:
        print("pass  drop keeps the good sentence and removes the bad one")
    print("\n%d of %d cases pass" % (len(CASES) + 1 - bad, len(CASES) + 1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
