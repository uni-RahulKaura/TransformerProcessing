#!/usr/bin/env python
"""Build one blind judging packet per document.

A packet contains the section titles, the published summary for each, and the topic labels with
their confidence. It contains NOTHING else. In particular it omits:

  - which internal path produced the summary (abstractive / depolarised / faithful / verbatim).
    A judge told "this one was repaired" grades the repair, not the sentence.
  - the extractive summary the published one replaced, which would anchor the judge to it.
  - any count, rate or self-assessment from the run.
  - the parser, the rules, and every other judge's verdict.

The raw Landing Markdown is named by path, not pasted, so the judge has to find each section in
the source itself. That makes the section boundary part of what is being checked rather than
something the packet asserts.

Usage: make_packets.py <out_dir_with_docs/> <packet_dir>
"""
import json
import os
import re
import sys


def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    made = []
    for f in sorted(os.listdir(os.path.join(src, "docs"))):
        if not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join(src, "docs", f)))
        meta, rows = d["meta"], d["rows"]
        typ = meta["type"]
        raw = os.path.join("gpu-bundle-v2", "inputs", typ, meta["file"])
        items = []
        for i, r in enumerate(rows):
            if not r.get("summary"):
                continue
            items.append({
                "n": len(items) + 1,
                "section_title": (r.get("title") or "(no header)").strip(),
                "published_summary": re.sub(r"\s+", " ", r["summary"]).strip(),
                "topic_labels": [{"topic": t, "confidence": c}
                                 for t, c in (r.get("topics") or [])],
            })
        pk = {"document": meta["file"], "document_type": typ,
              "raw_markdown_path": raw,
              "sections_with_summaries": len(items),
              "items": items}
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", meta["key"])[:90] + ".packet.json"
        json.dump(pk, open(os.path.join(dst, name), "w"), indent=1)
        made.append((name, len(items), raw))
    for n, c, r in made:
        print("%-92s %3d sections   %s" % (n, c, "raw OK" if os.path.exists(r) else "RAW MISSING"))
    print("\n%d packets -> %s" % (len(made), dst))


if __name__ == "__main__":
    main()
