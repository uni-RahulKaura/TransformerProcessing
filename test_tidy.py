#!/usr/bin/env python
"""Regression suite for the surface tidy. Run: python test_tidy.py

Every case is from real BART output on the Broadcaster Ltd NDA, plus the words that must NOT be
touched. The false-repair cases matter as much as the repairs: an earlier version turned
"asimilar" into "as imilar" and stripped the period from "Acme Ltd.".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.safe_abstractive import tidy                    # noqa: E402

SECTION = ("The Recipient shall use all reasonable endeavours. Information provided to a Third "
           "Party without a similar restriction by the Disclosing Party. The parties agree about "
           "another arrangement and therefore these terms apply. Shareholders hold a share of "
           "the whole. NDA STUDIOS LIMITED is incorporated in England and Wales.")

CASES = [
    ("lost space after 'all'", "shall use allreasonable endeavours.",
     "shall use all reasonable endeavours."),
    ("lost space after 'a', twice, one capitalised",
     "provided to aThird Party without asimilar restriction.",
     "provided to a Third Party without a similar restriction."),
    ("lost space after 'the'", "provided by thedisclosing Party.",
     "provided by the disclosing Party."),
    ("doubled full stop left by a figure deletion",
     "incorporated in England and Wales.. The office is in London.",
     "incorporated in England and Wales. The office is in London."),
    ("full stop dropped mid-clause",
     "The Recipient is not responsible. for any breach.",
     "The Recipient is not responsible for any breach."),
    ("source emphasis carried through",
     "**Authorised Third Party** means employees.",
     "Authorised Third Party means employees."),
    ("lower-case letter inside a run of capitals",
     "NDA STUDiOS LIMITED is incorporated.", "NDA STUDIOS LIMITED is incorporated."),
    # --- must NOT be touched ---
    ("a real word that looks glued", "The parties agree about another arrangement.",
     "The parties agree about another arrangement."),
    ("'therefore' and 'these' survive", "Therefore these terms apply.",
     "Therefore these terms apply."),
    ("a compound the section uses", "Shareholders hold a share.",
     "Shareholders hold a share."),
    ("an abbreviation keeps its period", "Acme Ltd. shall provide services.",
     "Acme Ltd. shall provide services."),
    ("a genuine sentence boundary", "Payment is due in 30 days. Interest accrues after.",
     "Payment is due in 30 days. Interest accrues after."),
]


def main():
    bad = 0
    for name, given, want in CASES:
        got = tidy(given, SECTION)
        ok = got == want
        bad += not ok
        print("%s  %-44s %s" % ("pass" if ok else "FAIL", name, got))
        if not ok:
            print("      wanted: %s" % want)
    print("\n%d of %d cases pass" % (len(CASES) - bad, len(CASES)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
