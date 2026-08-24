#!/usr/bin/env python
"""What the abstractive rate would be at each SHORT_WORDS threshold, from a run's own output.

Avoids re-running the corpus once per candidate threshold. Every section records its own word count
and the route that produced it, so the counterfactual is arithmetic: a section quoted only because
it fell under the threshold would have been reworded had the threshold been lower, and the routes
that were NOT length decisions -- a rejected figure, a wrong party name -- are unaffected by it.

Usage: threshold_sweep.py <run_dir>
"""
import glob
import json
import os
import sys


def main():
    run = sys.argv[1]
    rows = []
    for f in glob.glob(os.path.join(run, "docs", "*.json")):
        d = json.load(open(f))
        for r in d["rows"]:
            if not r.get("summary"):
                continue
            rows.append((r.get("words") or 0,
                         (r.get("abstractive") or {}).get("chosen") or "",
                         (r.get("title") or "").startswith("(")))
    tot = len(rows)
    # sections whose route was decided by LENGTH, not by a failed check
    quoted = [w for w, c, _ in rows if c == "extractive-short"]
    # sections that were reworded regardless
    rew = sum(1 for _, c, _ in rows if c.startswith("abstractive"))
    # sections that fell back for a real reason and would do so at any threshold
    hard = sum(1 for _, c, _ in rows if c in ("faithful", "faithful-depolarised", "verbatim"))
    print("run: %s" % run)
    print("  %d prose sections   %d reworded   %d quoted for length   %d fell back on a check"
          % (tot, rew, len(quoted), hard))
    print()
    print("  %-16s %10s %10s %16s" % ("threshold", "quoted", "reworded", "abstractive rate"))
    cur = max(quoted) if quoted else 0
    for th in (0, 10, 15, 20, 25, 30, 40, 50, 60):
        if th >= cur:
            # raising the threshold beyond what this run used cannot be inferred: those sections
            # were reworded and we do not know whether the rewrite would have passed its checks
            q = len(quoted) if th >= cur else sum(1 for w in quoted if w <= th)
            note = "  (>= this run's %d, not inferable)" % cur if th > cur else ""
        else:
            q = sum(1 for w in quoted if w <= th)
            note = ""
        # sections released back to the abstractive path are assumed to succeed at the measured
        # success rate of the sections that did go through it, not at 100%
        released = len(quoted) - q
        success = rew / float(rew + hard) if (rew + hard) else 1.0
        r2 = rew + int(round(released * success))
        print("  %-16s %10d %10d %15.1f%%%s"
              % ("<= %d words" % th if th else "off", q, r2, 100.0 * r2 / tot, note))
    print()
    print("  released sections are credited at the measured pass rate of the abstractive path")
    print("  (%.1f%%), not at 100%%, so this is a floor rather than a best case." % (100 * success))


if __name__ == "__main__":
    main()
