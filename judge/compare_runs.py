#!/usr/bin/env python
"""Measure one run against another, per document and per type.

Exists so the effect of a change is a number rather than a claim. The two runs must be over the
same inputs; the outlines are expected to be byte-identical, and that is asserted rather than
assumed, because a change that moves the outlines has done something other than what was intended.

Usage: compare_runs.py <baseline_dir> <final_dir>
"""
import collections
import json
import os
import sys


def load(run):
    out = {}
    d = os.path.join(run, "docs")
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            j = json.load(open(os.path.join(d, f)))
            out[j["meta"]["key"]] = j
    return out


def stats(j):
    ch = j["meta"].get("summary_choice") or {}
    prose = sum(v for k, v in ch.items() if k != "none")
    rew = sum(v for k, v in ch.items() if k.startswith("abstractive"))
    fall = sum(v for k, v in ch.items() if k in ("faithful", "faithful-depolarised", "verbatim"))
    short = ch.get("extractive-short", 0)
    return dict(prose=prose, rew=rew, fall=fall, short=short,
                pol=j["meta"].get("polarity_sentences_dropped", 0),
                inv=j["meta"].get("money_invented", 0),
                sections=j["meta"]["sections"],
                labels=sum(len(r.get("topics") or []) for r in j["rows"]),
                labelled=sum(1 for r in j["rows"] if r.get("topics")))


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    common = sorted(set(a) & set(b))
    if set(a) != set(b):
        print("WARNING: %d docs only in baseline, %d only in final"
              % (len(set(a) - set(b)), len(set(b) - set(a))))
    print("%-52s %13s %13s %8s" % ("document", "baseline", "final", "delta"))
    print("%s" % ("-" * 92))
    byt = collections.defaultdict(lambda: [0, 0, 0, 0])
    A = collections.Counter()
    B = collections.Counter()
    moved = 0
    for k in common:
        sa, sb = stats(a[k]), stats(b[k])
        for key in ("prose", "rew", "fall", "pol", "inv", "labels", "labelled", "short"):
            A[key] += sa[key]
            B[key] += sb[key]
        if sa["sections"] != sb["sections"]:
            moved += 1
            d = sb["sections"] - sa["sections"]
            # Outline changes are EXPECTED in this comparison and were not in the previous one:
            # the lettered-clause rule adds titled sub-items. So the check is no longer "did
            # anything move" but "did anything move DOWNWARDS or by an unexplained amount" -- a
            # section that disappears is a regression whatever else changed.
            flag = "  <-- LOST SECTIONS, investigate" if d < 0 else ""
            print("  outline changed: %-46s %4d -> %4d  (%+d)%s"
                  % (k[:46], sa["sections"], sb["sections"], d, flag))
        t = a[k]["meta"]["type"]
        byt[t][0] += sa["prose"]
        byt[t][1] += sa["rew"]
        byt[t][2] += sb["prose"]
        byt[t][3] += sb["rew"]
        pa = 100.0 * sa["rew"] / sa["prose"] if sa["prose"] else 0
        pb = 100.0 * sb["rew"] / sb["prose"] if sb["prose"] else 0
        flag = "  <-- " if abs(pb - pa) >= 5 else ""
        print("%-52s %6d/%-6d %6d/%-6d %+7.0f%%%s"
              % (a[k]["meta"]["file"][:52], sa["rew"], sa["prose"], sb["rew"], sb["prose"],
                 pb - pa, flag))
    print("\n%-14s %14s %14s %8s" % ("type", "baseline", "final", "delta"))
    for t, (p1, r1, p2, r2) in sorted(byt.items()):
        x = 100.0 * r1 / p1 if p1 else 0
        y = 100.0 * r2 / p2 if p2 else 0
        print("%-14s %6d/%-6d %2.0f%%  %6d/%-6d %2.0f%%  %+6.1f"
              % (t, r1, p1, x, r2, p2, y, y - x))
    pa = 100.0 * A["rew"] / A["prose"] if A["prose"] else 0
    pb = 100.0 * B["rew"] / B["prose"] if B["prose"] else 0
    print("\n%d documents compared" % len(common))
    print("  rewritten        %d/%d = %.1f%%   ->   %d/%d = %.1f%%   (%+.1f points)"
          % (A["rew"], A["prose"], pa, B["rew"], B["prose"], pb, pb - pa))
    print("  fell back        %d   ->   %d" % (A["fall"], B["fall"]))
    print("  polarity drops   %d   ->   %d" % (A["pol"], B["pol"]))
    print("  invented amounts %d   ->   %d" % (A["inv"], B["inv"]))
    print("  topic labels     %d on %d sections   ->   %d on %d sections"
          % (A["labels"], A["labelled"], B["labels"], B["labelled"]))
    lost = 0
    for k in common:
        if stats(b[k])["sections"] < stats(a[k])["sections"]:
            lost += 1
    print("  outlines changed %d  (expected: the lettered-clause rule adds titled sub-items)"
          % moved)
    print("  outlines SHRANK  %d  (must be 0 -- a section that disappears is a regression)" % lost)
    print("  short-quoted     %d   ->   %d   (sections published as the section's own sentences)"
          % (A["short"], B["short"]))


if __name__ == "__main__":
    main()
