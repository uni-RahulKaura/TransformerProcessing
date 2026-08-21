#!/usr/bin/env python
"""Print a document's outline in the nested form the reviewer asked for.

    1. DEFINITIONS
    2. PROVIDER SERVICES AND RESPONSIBILITIES
      |_ 2.1. Order Forms and Statements of Work
      |_ 2.2. Agreement

Top-level headings sit flush left. Anything nested under one is prefixed "|_" and indented
two spaces per level. A section the document gives no heading for is written as "no header"
rather than left blank or silently dropped -- if a block of text has no heading, that is a
fact about the document and the reader needs to see it.

No model is involved and none is needed: an outline is the section rules and nothing else, so
this runs in milliseconds on any machine. Only the SUMMARIES need a GPU.
"""
import json
import os
import re
import sys

# (bundle is self-contained; no external repo path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_run import sectionise                     # noqa: E402

NOHEAD = re.compile(r"^\[preamble|^no header$", re.I)
# A title that is nothing but its own number: the document numbers the subsection but never
# captions it. The number still has to show -- it is how the reader finds the clause -- so the
# absence of a caption is stated next to it rather than left to be inferred from a bare "2.1".
NUM_ONLY = re.compile(r"^\d+(?:\.\d+)*\.?$")


def outline(md):
    S = sectionise(md)
    kids = {}
    for i, s in enumerate(S):
        if s["parent"] is not None:
            kids.setdefault(s["parent"], []).append(i)
    roots = [i for i, s in enumerate(S) if s["parent"] is None]
    lines = []
    stats = {"sections": len(S), "nested": sum(len(v) for v in kids.values()),
             "no_header": 0, "tables": 0, "figures": 0}

    def emit(i, depth):
        s = S[i]
        title = (s["title"] or "").strip()
        if not title or NOHEAD.match(title):
            title = "no header"
            stats["no_header"] += 1
        elif NUM_ONLY.match(title):
            title = "%s (no header)" % title.rstrip(".")
            stats["no_header"] += 1
        mark = []
        if s["tables"]:
            mark.append("%d table%s" % (len(s["tables"]), "" if len(s["tables"]) == 1 else "s"))
            stats["tables"] += len(s["tables"])
        if s["figures"]:
            mark.append("%d picture%s" % (len(s["figures"]),
                                          "" if len(s["figures"]) == 1 else "s"))
            stats["figures"] += len(s["figures"])
        tail = ("   [%s]" % ", ".join(mark)) if mark else ""
        lines.append(("  " * depth + ("|_ " if depth else "") + title)[:150] + tail)
        for k in kids.get(i, []):
            emit(k, depth + 1)

    for i in roots:
        emit(i, 0)
    return lines, stats


if __name__ == "__main__":
    src = sys.argv[1]
    md = open(src, encoding="utf-8", errors="replace").read()
    lines, stats = outline(md)
    print("=" * 78)
    print(os.path.basename(src))
    print("%d sections, %d nested, %d with no header, %d tables, %d pictures"
          % (stats["sections"], stats["nested"], stats["no_header"],
             stats["tables"], stats["figures"]))
    print("=" * 78)
    for ln in lines:
        print(ln)
    if len(sys.argv) > 2:
        json.dump({"file": os.path.basename(src), "stats": stats, "outline": lines},
                  open(sys.argv[2], "w"), indent=1)
