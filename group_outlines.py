#!/usr/bin/env python
"""Group the finished outlines by folder: every Tesla outline together, then every bill, etc."""
import glob, json, os, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
ORDER = ["tesla_data", "bills", "contracts", "legal_data"]
docs = {}
for f in sorted(glob.glob(os.path.join(OUT, "docs", "*.outline.json"))):
    d = json.load(open(f)); docs.setdefault(d["meta"]["type"], []).append(d)
os.makedirs(os.path.join(OUT, "outlines"), exist_ok=True)
groups = [t for t in ORDER if t in docs] + [t for t in sorted(docs) if t not in ORDER]
alltxt = []
for typ in groups:
    ds = sorted(docs[typ], key=lambda x: x["meta"]["file"])
    buf = ["#" * 78, "#  %s  -  %d files" % (typ.upper(), len(ds)), "#" * 78]
    for d in ds:
        m = d["meta"]
        buf += ["", "=" * 78,
                m["file"],
                "%d sections, %d nested, %d with no header, %d tables, %d pictures"
                % (m["sections"], m["nested"], m["no_header"], m["source_tables"],
                   m["source_pictures"]),
                "=" * 78] + d["outline"]
    txt = "\n".join(buf) + "\n"
    open(os.path.join(OUT, "outlines", typ + ".txt"), "w").write(txt)
    alltxt.append(txt)
    print("%-12s %2d files  %4d sections  -> %s/outlines/%s.txt"
          % (typ, len(ds), sum(d["meta"]["sections"] for d in ds), OUT, typ))
open(os.path.join(OUT, "outlines", "ALL.txt"), "w").write("\n".join(alltxt))
print("-> %s/outlines/ALL.txt" % OUT)
