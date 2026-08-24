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
from pipeline import topics as topiclib             # noqa: E402  (stage 6, additive)

# Written into every document's output so the number beside a topic is never read as something
# it is not. Asked for explicitly: the reader has to be able to say what it measures.
TOPIC_CONFIDENCE_NOTE = {
    "what_the_number_is": (
        "The probability facebook/bart-large-mnli assigns to the statement "
        "\"This section is about <topic>.\" being entailed by the section's text. "
        "0.00 = the model rejects it, 1.00 = the model is certain."),
    "how_it_is_produced": (
        "Zero-shot natural language inference. The section is the premise and each candidate "
        "topic becomes a one-sentence hypothesis; the score is softmax over the entailment and "
        "contradiction logits for that pair. No topic classifier was trained, and no labelled "
        "topic data was used."),
    "scores_are_independent": (
        "Each topic is scored on its own, so the scores do NOT sum to 1 and are not shares of "
        "the section. One section can be 1.00 confidentiality and 0.69 data protection at once."),
    "what_is_published": (
        "Up to 3 topics per section, highest first, each above %.2f, and only if the section "
        "uses at least one of that topic's own words (the grounding gate)." % topiclib.THRESH),
    "reading_the_bands": {
        "0.90-1.00": "unambiguous; the section states the subject outright",
        "0.70-0.89": "confident; safe to show a reader",
        "0.55-0.69": "weak; correct more often than not, but this is where the errors are",
    },
    "known_limits": (
        "Confidence is not accuracy: the model can be 0.96 and wrong. The grounding gate "
        "removes labels whose vocabulary is absent from the section, which is why some "
        "high-scoring labels are not published."),
}

OUT = os.environ.get("OUT", "out")
FORCE = os.environ.get("FORCE") == "1"


def drop_appledouble(files):
    """Drop macOS AppleDouble twins. A tarball built on a Mac carries "._name" companions for
    extended attributes; extracted on Linux they are ordinary files matching *.md, and they
    doubled this corpus from 7 documents to 14 with binary junk."""
    return [f for f in files if not os.path.basename(f).startswith("._")]


def pick(files):
    """ONLY=<substring> runs just the files whose path contains it, so one document can be
    proved end to end before committing the box to all 33. Several patterns with "|"."""
    only = os.environ.get("ONLY", "").strip()
    if not only:
        return files
    pats = [p.strip().lower() for p in only.split("|") if p.strip()]
    hit = [f for f in files if any(p in f.lower() for p in pats)]
    if not hit:
        sys.exit("ONLY=%s matched none of the %d input files" % (only, len(files)))
    print("ONLY=%s -> %d of %d files" % (only, len(hit), len(files)), flush=True)
    return hit


def main():
    files = pick(drop_appledouble(sorted(glob.glob("inputs/*/*.md"))))
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "docs"), exist_ok=True)
    print("device: %s   files: %d" % (DEVICE, len(files)), flush=True)
    tok, mdl = load(REPO, "abstractive")

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
        # Stage 6: per-document-type topic labels, additive to the five facets. TOPICS=0 skips it.
        if os.environ.get("TOPICS", "1") != "0":
            dtype = topiclib.FOLDER_TYPE.get(typ, "agreements")
            from compare_run import sectionise as _sect
            _S = _sect(raw)
            for _i, _r in enumerate(rows):
                _txt = (_S[_i].get("full") or _S[_i].get("text") or "") if _i < len(_S) else ""
                try:
                    _r["topics"] = topiclib.label(_txt, dtype,
                                                 title=_r.get("title") or "")
                except Exception as _e:                     # never lose a document to stage 6
                    _r["topics"] = []
                    _r["topic_error"] = str(_e)[:120]
        real, bad, inv = money_check(rows, raw)
        secs = round(time.time() - t0, 1)
        _ch = {}
        for _r in rows:
            _ch[(_r.get("abstractive") or {}).get("chosen") or "none"] = \
                _ch.get((_r.get("abstractive") or {}).get("chosen") or "none", 0) + 1
        _pol = sum(len((_r.get("abstractive") or {}).get("polarity_dropped") or [])
                   for _r in rows)
        meta = {"key": key, "type": typ, "file": os.path.basename(f),
                "summary_choice": _ch, "polarity_sentences_dropped": _pol,
                "topic_confidence": TOPIC_CONFIDENCE_NOTE,
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
