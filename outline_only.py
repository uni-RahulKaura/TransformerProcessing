#!/usr/bin/env python
"""Outlines for every input file. No transformer, so this finishes in seconds."""
import glob, json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from outline import outline

OUT = os.environ.get("OUT", "out")
files = sorted(glob.glob("inputs/*/*.md"))
os.makedirs(os.path.join(OUT, "docs"), exist_ok=True)
man = []
t0 = time.time()
for n, f in enumerate(files, 1):
    typ = f.split("/")[-2]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(f))[:90].rstrip("_")
    key = "%s__%s" % (typ, stem)
    raw = open(f, encoding="utf-8", errors="replace").read()
    lines, st = outline(raw)
    meta = {"key": key, "type": typ, "file": os.path.basename(f), "chars": len(raw),
            "sections": st["sections"], "nested": st["nested"], "no_header": st["no_header"],
            "source_tables": len(re.findall(r"<table", raw)), "source_pictures": raw.count("<::"),
            "tables": st["tables"], "figures": st["figures"]}
    json.dump({"meta": meta, "outline": lines},
              open(os.path.join(OUT, "docs", key + ".outline.json"), "w"), indent=1)
    with open(os.path.join(OUT, "docs", key + ".outline.txt"), "w") as fh:
        fh.write("%s\n%d sections, %d nested, %d with no header, %d tables, %d pictures\n"
                 % (os.path.basename(f), st["sections"], st["nested"], st["no_header"],
                    st["tables"], st["figures"]))
        fh.write("=" * 78 + "\n" + "\n".join(lines) + "\n")
    man.append(meta)
    print("[%2d/%d] %-56s %4d sect %4d nested %3d no-hdr" % (n, len(files), key[:55],
          st["sections"], st["nested"], st["no_header"]), flush=True)
json.dump(man, open(os.path.join(OUT, "outline_manifest.json"), "w"), indent=1)
print("\n%.0fs  %d files  %d sections total" % (time.time()-t0, len(man), sum(m["sections"] for m in man)))
