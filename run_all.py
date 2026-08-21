#!/usr/bin/env python
"""Outline and summarise EVERY input file, one output per file.

Separate output per document on purpose: 33 documents in one blob is unreadable and cannot be
handed to anyone a document at a time. Each file gets its own JSON and its own outline.

Resumable. A file whose output already exists and is non-empty is skipped, so a dropped tab
or a killed process costs only the file that was in flight.

Only the Landing-native route runs here. These files ARE Landing output, so that is the real
pipeline; the docling comparison was already done separately on four documents and doubling
the run to repeat it on all 33 buys nothing.
"""
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_run import index, money_check, REPO       # noqa: E402
from outline import outline                            # noqa: E402
from pipeline.summarise import load, DEVICE            # noqa: E402

OUT = os.environ.get("OUT", "out")
FORCE = os.environ.get("FORCE") == "1"


def main():
    files = sorted(glob.glob("inputs/*/*.md"))
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "docs"), exist_ok=True)
    print("device: %s   files: %d" % (DEVICE, len(files)), flush=True)
    tok, mdl = load(REPO)

    manifest = []
    t_all = time.time()
    for n, f in enumerate(files, 1):
        typ = f.split("/")[-2]
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(f))[:90].rstrip("_")
        key = "%s__%s" % (typ, stem)
        target = os.path.join(OUT, "docs", key + ".json")
        if not FORCE and os.path.exists(target) and os.path.getsize(target) > 200:
            print("[%2d/%d] %-58s already done" % (n, len(files), key[:57]), flush=True)
            manifest.append(json.load(open(target))["meta"])
            continue
        raw = open(f, encoding="utf-8", errors="replace").read()
        t0 = time.time()
        lines, ostats = outline(raw)
        rows = index(raw, tok, mdl)
        real, bad, inv = money_check(rows, raw)
        secs = round(time.time() - t0, 1)
        meta = {"key": key, "type": typ, "file": os.path.basename(f),
                "chars": len(raw), "seconds": secs,
                "sections": len(rows),
                "prose": sum(1 for r in rows if r["summary"]),
                "tables": sum(len(r["table_summaries"]) for r in rows),
                "pictures": sum(len(r["figure_summaries"]) for r in rows),
                "source_tables": len(re.findall(r"<table", raw)),
                "source_pictures": raw.count("<::"),
                "no_header": ostats["no_header"],
                "money_real": real, "money_invented": bad,
                "bad_token_sections": sum(1 for r in rows if r["findings"])}
        json.dump({"meta": meta, "outline": lines, "rows": rows},
                  open(target, "w"), indent=1)
        with open(os.path.join(OUT, "docs", key + ".outline.txt"), "w") as fh:
            fh.write("%s\n%d sections, %d nested, %d with no header, %d tables, %d pictures\n"
                     % (os.path.basename(f), ostats["sections"], ostats["nested"],
                        ostats["no_header"], ostats["tables"], ostats["figures"]))
            fh.write("=" * 78 + "\n" + "\n".join(lines) + "\n")
        manifest.append(meta)
        print("[%2d/%d] %-58s %3ds  %4d sect  %4d prose  %3d tbl  %3d pics  %d invented$"
              % (n, len(files), key[:57], secs, meta["sections"], meta["prose"],
                 meta["tables"], meta["pictures"], bad), flush=True)
        json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)

    print("\nALL DONE in %.0f min. %d files -> %s/docs/"
          % ((time.time() - t_all) / 60, len(manifest), OUT), flush=True)
    tot = lambda k: sum(m[k] for m in manifest)
    print("totals: %d sections, %d prose, %d tables, %d pictures, %d invented amounts"
          % (tot("sections"), tot("prose"), tot("tables"), tot("pictures"),
             tot("money_invented")))


if __name__ == "__main__":
    main()
