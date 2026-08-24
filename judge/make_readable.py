#!/usr/bin/env python
"""Render every document's outline and summaries as plain text a person can read.

One pair of files per document, plus an index. The confidence note travels at the head of each
outline rather than in a covering message, so whoever opens the file can say what the number
beside a topic means without asking.

Usage: make_readable.py <run_dir> <out_dir>
"""
import json
import os
import re
import sys
import textwrap

W = 104
# SHOW_SCORES=0 prints the topic labels without their confidence numbers. For a reader who only
# needs to know what a section is about, the number is noise -- and the long explanation of what
# it measures is pointless once it is gone, so that goes too.
SCORES = os.environ.get("SHOW_SCORES", "1") != "0"
FACET = [("PAY", "M"), ("EXPIRY", "D"), ("PARTY", "W"), ("PERM", "P"), ("OBLIG", "O")]
TAGS = [
    ("abstractive", "reworded by the model, clean on every check"),
    ("abstractive-depolarised", "reworded; a sentence reversing the section's polarity removed"),
    ("abstractive-repaired", "reworded; an unsupported figure or date deleted"),
    ("faithful", "the rewrite failed a check, so the closer extract was published"),
    ("verbatim", "neither model draft was safe, so the section is quoted directly"),
]


def depth(rows, i):
    n, p, seen = 0, rows[i].get("parent"), set()
    while p is not None and 0 <= p < len(rows) and p not in seen and n < 30:
        seen.add(p)
        n += 1
        p = rows[p].get("parent")
    return n


def outline_text(d):
    rows, meta = d["rows"], d["meta"]
    tc = meta.get("topic_confidence") or {}
    L = ["OUTLINE  --  %s" % meta["file"],
         "%d sections, %d with prose, %d tables, %d pictures      Tesla T4, %ss"
         % (meta["sections"], meta.get("prose", 0), meta.get("tables", 0),
            meta.get("pictures", 0), meta.get("seconds")),
         "", "THE TWO COLUMNS", "-" * W,
         "  [ ]  the five original regex facets:  M money  D dates/term  W who  "
         "P permissions  O obligations",
         "  < >  topic labels" + (", each with the model's confidence in it" if SCORES else ""), ""]
    if tc and not SCORES:
        L += ["  Topics are assigned by a language model reading each section. They describe what the",
              "  section is about; they are not taken from any list in the document itself.", ""]
    if tc and SCORES:
        L += ["WHAT THE CONFIDENCE NUMBER IS", "-" * W]
        for k in ("what_the_number_is", "how_it_is_produced", "scores_are_independent",
                  "what_is_published", "known_limits"):
            if tc.get(k):
                L.append("  " + k.replace("_", " ") + ":")
                L.append(textwrap.fill(tc[k], W - 4, initial_indent="    ",
                                       subsequent_indent="    "))
        if tc.get("reading_the_bands"):
            L.append("  reading the bands:")
            for b, t in tc["reading_the_bands"].items():
                L.append("    %-12s %s" % (b, t))
    L += ["", "=" * W, ""]
    for i, r in enumerate(rows):
        t = (r.get("title") or "no header").strip() or "no header"
        pre = "  " * depth(rows, i) + ("|_ " if depth(rows, i) else "")
        fac = "".join(s for c, s in FACET if c in (r.get("cats") or []))
        tops = (r.get("topics") or [])[:3]
        tt = " ".join(("<%s %.2f>" % (a, b)) if SCORES else ("<%s>" % a) for a, b in tops)
        mk = []
        if r.get("n_tables"):
            mk.append("%dt" % r["n_tables"])
        if r.get("n_figures"):
            mk.append("%dp" % r["n_figures"])
        L.append("%-62s %-6s %-4s %s" % ((pre + t)[:62], "[%s]" % fac if fac else "",
                                         "(%s)" % ",".join(mk) if mk else "", tt))
    return "\n".join(L) + "\n"


def summary_text(d):
    rows, meta = d["rows"], d["meta"]
    ch = meta.get("summary_choice") or {}
    prose = sum(v for k, v in ch.items() if k != "none")
    rew = sum(v for k, v in ch.items() if k.startswith("abstractive"))
    S = ["SUMMARIES  --  %s" % meta["file"], "",
         "HOW TO READ EACH SECTION BELOW", "-" * W,
         "  Every section prints TWO summaries, numbered 1 and 2:", "",
         "    1)  ABSTRACTIVE  =  REWORDED by the model, in its own words.",
         "    2)  EXTRACTIVE   =  WORD-FOR-WORD sentences lifted straight out of the document.",
         "",
         "  The one marked  <<< PUBLISHED  is the one the pipeline actually outputs.",
         "  The other is shown only for comparison, so the difference is visible.",
         "",
         "  Normally 1) ABSTRACTIVE is the published one -- that is the whole point of the change.",
         "  On a minority of sections the reworded draft failed a safety check (an invented figure,",
         "  a party name the section does not contain, a reversed obligation). There the pipeline",
         "  refuses the rewrite and publishes 2) EXTRACTIVE instead, and that line carries the",
         "  <<< PUBLISHED marker rather than the first. Those sections say why underneath.",
         "", "THE ROUTE TAG IN [ ]", "-" * W]
    for k, why in TAGS:
        S.append("  %-24s %s" % (k, why))
    S += ["",
          "This document: " + ", ".join("%s %d" % (k, v) for k, v in
                                        sorted(ch.items(), key=lambda x: -x[1]) if k != "none"),
          "%d of %d prose sections published a reworded summary%s."
          % (rew, prose, "  (%.0f%%)" % (100.0 * rew / prose) if prose else ""),
          "%d sentence(s) removed for reversing the section's polarity."
          % meta.get("polarity_sentences_dropped", 0),
          "", "=" * W, ""]
    for r in rows:
        if not r.get("summary"):
            continue
        a = r.get("abstractive") or {}
        S.append("-" * W)
        S.append("%s   [%s]" % ((r.get("title") or "no header").strip()[:66],
                               a.get("chosen") or "n/a"))
        tops = r.get("topics") or []
        if tops:
            S.append("  topics: %s" % ", ".join(
                ("%s %.2f" % (x, y)) if SCORES else str(x) for x, y in tops[:3]))
        for x in (a.get("polarity_dropped") or []):
            S.append("  polarity: dropped a sentence reversing '%s' -- %s"
                     % (x.get("word"), (x.get("sentence") or "")[:70]))
        if a.get("removed"):
            S.append("  removed as unsupported: %s" % ", ".join(a["removed"]))
        S.append("")
        chosen = a.get("chosen") or ""
        pub_is_rewrite = chosen.startswith("abstractive")
        pub = re.sub(r"\s+", " ", r["summary"])
        ex = re.sub(r"\s+", " ", r.get("summary_extractive") or "")
        ind = " " * 20
        if pub_is_rewrite:
            S.append("  1) ABSTRACTIVE  <<< PUBLISHED")
            S.append("     " + textwrap.fill(pub, 92, subsequent_indent="     "))
            if ex and ex.strip() != pub.strip():
                S.append("  2) EXTRACTIVE   (word-for-word; this is what it replaced)")
                S.append("     " + textwrap.fill(ex, 92, subsequent_indent="     "))
        else:
            # the rewrite was refused; what is published here IS the word-for-word text
            S.append("  1) ABSTRACTIVE  (REJECTED -- not published; see the reason above)")
            draft = re.sub(r"\s+", " ", a.get("rejected_draft") or "(no reworded draft)")
            S.append("     " + textwrap.fill(draft, 92, subsequent_indent="     "))
            S.append("  2) EXTRACTIVE   <<< PUBLISHED  (word-for-word, because the rewrite was unsafe)")
            S.append("     " + textwrap.fill(pub, 92, subsequent_indent="     "))
        S.append("")
    return "\n".join(S) + "\n"


def main():
    run, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True)
    idx, tot, rew = [], 0, 0
    for f in sorted(os.listdir(os.path.join(run, "docs"))):
        if not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join(run, "docs", f)))
        stem = f[:-5]
        open(os.path.join(dst, stem + ".OUTLINE.txt"), "w").write(outline_text(d))
        open(os.path.join(dst, stem + ".SUMMARIES.txt"), "w").write(summary_text(d))
        ch = d["meta"].get("summary_choice") or {}
        p = sum(v for k, v in ch.items() if k != "none")
        a = sum(v for k, v in ch.items() if k.startswith("abstractive"))
        tot += p
        rew += a
        idx.append((d["meta"]["type"], d["meta"]["file"], d["meta"]["sections"], p, a))
    with open(os.path.join(dst, "INDEX.txt"), "w") as fh:
        fh.write("ALL %d DOCUMENTS\n%s\n" % (len(idx), "=" * W))
        fh.write("%-14s %-56s %5s %6s %6s\n" % ("type", "document", "sect", "prose", "rewrit"))
        for t, n, s, p, a in idx:
            fh.write("%-14s %-56s %5d %6d %6d\n" % (t, n[:56], s, p, a))
        fh.write("\n%d prose sections across %d documents, %d published as a rewrite (%.1f%%)\n"
                 % (tot, len(idx), rew, 100.0 * rew / tot if tot else 0))
    print("%d documents -> %s" % (len(idx), dst))
    print("%d prose sections, %d rewritten (%.1f%%)" % (tot, rew, 100.0 * rew / tot if tot else 0))


if __name__ == "__main__":
    main()
