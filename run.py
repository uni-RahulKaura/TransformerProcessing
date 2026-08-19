#!/usr/bin/env python
"""Build a navigable index of any Markdown document. No LLM, no network at inference.

    python run.py examples/sample_sow.md                    # all four models
    python run.py mydoc.md --models BART-large-CNN          # just one
    python run.py mydoc.md --no-summaries                   # structure + labels only, instant

Writes index.json and prints a readable outline. Everything runs locally; the only network
traffic is the first download of the model weights.

Start with --no-summaries. It runs in under a second and tells you the thing you most need
to know about a new document type: whether the section rules found the structure at all. If
the section count looks wrong, no amount of summarising will fix it -- the rules in
pipeline/sections.py need adjusting for that document type first.
"""
import argparse
import json
import sys
import time

from pipeline.categories import rules_predict
from pipeline.sections import split_sections

CATNAME = {"PAY": "money", "EXPIRY": "dates & term", "PARTY": "who it's between",
           "PERM": "permissions", "OBLIG": "obligations"}
ORDER = ["PAY", "EXPIRY", "PARTY", "PERM", "OBLIG"]

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--models", nargs="*", default=None,
                help="subset of model names; default is all four")
ap.add_argument("--no-summaries", action="store_true",
                help="structure and category labels only -- no model loaded")
ap.add_argument("--out", default="index.json")
a = ap.parse_args()

text = open(a.path, encoding="utf-8", errors="replace").read()
sections = split_sections(text)
for s in sections:
    s["categories"] = ([c for c in ORDER if rules_predict(s["text"]).get(c)]
                       if s["text"].strip() else [])

kids = {}
for i, s in enumerate(sections):
    if s["parent"] is not None:
        kids.setdefault(s["parent"], []).append(i)
n_text = sum(1 for s in sections if not s["container_only"] and s["text"].strip())
print("%s: %d sections, %d with text, %d nested by numbering, %d heading-only"
      % (a.path, len(sections), n_text,
         sum(1 for s in sections if s["parent"] is not None),
         sum(1 for s in sections if s["container_only"])))
print("   inferred by shape rather than a real heading: %d  (rule 1b -- audit these first)"
      % sum(1 for s in sections if s["heading_confidence"] == "inferred"))

models = {}
if not a.no_summaries:
    from pipeline.summarise import MODELS, run_model
    from pipeline.verify import check
    want = MODELS if not a.models else [m for m in MODELS if m[0] in a.models]
    if not want:
        sys.exit("no model matched; choose from: %s" % ", ".join(m[0] for m in MODELS))
    for name, repo, limit, kind in want:
        t0 = time.time()
        models[name] = run_model(name, repo, limit, kind, sections)
        bad = 0
        for s, r in zip(sections, models[name]["rows"]):
            f = check(r.get("summary"), s["text"])
            f += [x for p in (r.get("parts") or []) for x in check(p["summary"], s["text"])]
            r["findings"] = f
            bad += bool(f)
        models[name]["seconds"] = round(time.time() - t0, 1)
        models[name]["sections_with_findings"] = bad
        print("   %-22s %5.1fs  %d sections split  %d sections with a corrupted word"
              % (name, models[name]["seconds"], models[name]["sections_needing_split"], bad))

json.dump({"doc": a.path, "n_sections": len(sections), "sections": sections,
           "models": models}, open(a.out, "w"), indent=1)
print("-> %s" % a.out)

print()
first = next(iter(models), None)


def show(i, depth=0):
    s = sections[i]
    pad = "    " * depth
    tag = ", ".join(CATNAME[c] for c in s["categories"]) or "none of the five"
    print("%s%s  [%s]  %s"
          % (pad, s["title"][:66],
             "heading only" if s["container_only"] else "%d words" % s["words"], tag))
    if first and not s["container_only"]:
        r = models[first]["rows"][i]
        if r.get("summary"):
            print("%s    %s" % (pad, r["summary"][:150]))
        for p in r.get("parts") or []:
            print("%s      %s: %s" % (pad, p["label"], p["summary"][:120]))
        if r.get("findings"):
            print("%s    !! %s" % (pad, "; ".join("%s: %s" % (f["kind"], f["token"])
                                                  for f in r["findings"])))
    for k in kids.get(i, []):
        show(k, depth + 1)


for i, s in enumerate(sections):
    if s["parent"] is None:
        show(i)
