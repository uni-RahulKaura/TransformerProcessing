#!/usr/bin/env python
"""Write the sections and the summaries as SEPARATE files, per folder.

    <out>/SECTIONS/<folder>.txt    the outline only -- nested, "no header" where the document
                                   gives none, and nothing else on the line
    <out>/SUMMARIES/<folder>.txt   one summary per section, in document order

Separate on purpose: the outline is for finding the section you want, and reading it is the
whole point of an index -- interleaving summaries into it makes it unscannable. Nothing is
added to either file that the document did not provide: no subsection is invented, and a
section with no summary is written as such rather than filled in.
"""
import glob
import json
import os
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
ORDER = ["tesla_data", "bills", "contracts", "legal_data"]

docs = {}
for f in sorted(glob.glob(os.path.join(OUT, "docs", "*.json"))):
    d = json.load(open(f))
    if "meta" not in d:
        continue
    docs.setdefault(d["meta"]["type"], []).append(d)

for sub in ("SECTIONS", "SUMMARIES"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

groups = [t for t in ORDER if t in docs] + [t for t in sorted(docs) if t not in ORDER]
all_sec, all_sum = [], []

for typ in groups:
    ds = sorted(docs[typ], key=lambda x: x["meta"]["file"])
    sec, smy = [], []
    banner = "#" * 78 + "\n#  %s  --  %d file%s\n" % (typ.upper(), len(ds),
                                                      "" if len(ds) == 1 else "s") + "#" * 78
    sec.append(banner)
    smy.append(banner)
    for d in ds:
        m = d["meta"]
        head = ("\n" + "=" * 78 + "\n" + m["file"] + "\n" + "=" * 78)
        sec.append(head)
        sec.extend(d.get("outline", []))
        smy.append(head)
        for r in d.get("rows", []):
            title = (r.get("title") or "").strip() or "no header"
            if title.startswith("[preamble"):
                title = "no header"
            smy.append("")
            smy.append(title)
            body = r.get("summary")
            if r.get("container"):
                smy.append("    (heading only -- groups the subsections beneath it)")
            elif body:
                smy.append("    " + body)
            else:
                smy.append("    (no summary -- this section has no prose of its own)")
            for t in r.get("table_summaries") or []:
                smy.append("    TABLE: " + t)
            for g in r.get("figure_summaries") or []:
                smy.append("    PICTURE: " + g)
    st, sm = "\n".join(sec) + "\n", "\n".join(smy) + "\n"
    open(os.path.join(OUT, "SECTIONS", typ + ".txt"), "w").write(st)
    open(os.path.join(OUT, "SUMMARIES", typ + ".txt"), "w").write(sm)
    all_sec.append(st)
    all_sum.append(sm)
    print("%-12s %2d files -> %s/SECTIONS/%s.txt  and  %s/SUMMARIES/%s.txt"
          % (typ, len(ds), OUT, typ, OUT, typ))

open(os.path.join(OUT, "SECTIONS", "ALL.txt"), "w").write("\n".join(all_sec))
open(os.path.join(OUT, "SUMMARIES", "ALL.txt"), "w").write("\n".join(all_sum))
print("-> %s/SECTIONS/ALL.txt  and  %s/SUMMARIES/ALL.txt" % (OUT, OUT))
