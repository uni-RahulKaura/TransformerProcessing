#!/usr/bin/env python
"""Index one document twice -- once via docling, once from Landing AI's own output -- and
score both.

The comparison is fair by construction: the SAME source file goes into both, and the same
BART model with the same settings summarises whatever each parser hands it. The only thing
that differs is the parser. So any difference in the result is the parser's doing.

  LANDING path   the file as Landing produced it: HTML tables with real rows and cells, plus
                 vision-model descriptions of every chart, diagram and signature block.

  DOCLING path   the same file put through docling, which is a real docling run rather than a
                 simulation. docling rewrites HTML tables as Markdown pipe tables and DELETES
                 every vision description, because it has no vision model to replace them.

Three summarisers, matched to what they read, identical on both sides:
  prose   -> BART-large-CNN
  tables  -> arithmetic on the cells, because BART invents figures when it reads a table
  figures -> the vision description, quoted (so the docling side simply has none)

Usage: compare_run.py <doc.md> <label> <outdir>
"""
import json
import os
import re
import sys
import time

import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from pipeline.categories import rules_predict          # noqa: E402
from pipeline.figures import interpret                 # noqa: E402
from pipeline.sections import split_sections
from pipeline.survey import survey           # noqa: E402
from pipeline.tables import summarise_table            # noqa: E402
from pipeline.verify import check                      # noqa: E402
from pipeline.chunking import split_parts              # noqa: E402
from pipeline.summarise import load, summarise         # noqa: E402

REPO, LIMIT = "facebook/bart-large-cnn", 1024
CATS = ["PAY", "EXPIRY", "PARTY", "PERM", "OBLIG"]
HTML_TABLE = re.compile(r"<table\b.*?</table>", re.S | re.I)
PIPE_TABLE = re.compile(r"(?m)(?:^\s*\|.*\|\s*$\n?){2,}")
VISION = re.compile(r"<::(.*?)::>", re.S)
ANCHOR = re.compile(r"<a id='[^']*'></a>\s*")
FURNITURE = [re.compile(r"(?m)^\s*DocuSign Envelope ID:.*$"),
             re.compile(r"(?m)^\s*<!--\s*PAGE BREAK\s*-->\s*$"),
             re.compile(r"(?m)^\s*Page \d+( of \d+)?\s*$")]


def sectionise(md):
    """Mask tables and figures, section the rest, then hand each section its own back.

    Masking rather than deleting: the heading rules must not look inside a table (cell text
    is short and capitalised, which is exactly what they mistake for a title), but the table
    itself is the point, so it goes back into the section body afterwards.
    """
    # Keep the layout-block boundary. Landing emits one <a id=...> per block it lifted off the
    # page, so the anchor marks where a visual unit begins -- and in a deck or a report whose
    # section titles carry no "#" and no numbering, "first line of a block, with body under
    # it" is the ONLY signal those titles leave. Stripping the anchor outright threw it away,
    # which is why every Tesla deck found ~11 of its ~40 real sections.
    # Decide the regime from the ORIGINAL document, before any masking. Masking a deck's tables
    # down to placeholders cut it from 84 KB to 21 KB, which quadrupled the apparent heading
    # density and flipped the document out of the regime that reads unmarked titles -- so six
    # of seven Tesla decks silently lost three quarters of their sections.
    regime = survey(md)["regime"]
    md = ANCHOR.sub("\n\x00BLK\x00\n", md)
    for rx in FURNITURE:
        md = rx.sub("", md)
    store = {}

    def hide(m):
        k = "MASK%05d" % len(store)
        store[k] = m.group(0)
        return "(block) " + k

    md = HTML_TABLE.sub(hide, md)
    md = PIPE_TABLE.sub(hide, md)
    md = VISION.sub(hide, md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    out = split_sections(md, regime=regime)
    for s_ in out:
        s_["text"] = re.sub(r"(?m)^\x00BLK\x00$\n?", "", s_["text"])
        s_["title"] = s_["title"].replace("\x00BLK\x00", "").strip()
    for s in out:
        full = s["text"]
        for k, v in store.items():
            full = full.replace(k, v)
        s["tables"] = (re.findall(r"<table\b.*?</table>", full, re.S | re.I)
                       + [x for x in PIPE_TABLE.findall(full) if x.count("|") > 4])
        s["figures"] = VISION.findall(full)
        prose = VISION.sub(" ", HTML_TABLE.sub(" ", full))
        prose = PIPE_TABLE.sub(" ", prose)
        prose = re.sub(r"<[^>]+>", " ", prose)
        prose = re.sub(r"\(block\)|MASK\d{5}", " ", prose)
        s["text"] = re.sub(r"\s+", " ", prose).strip()
        s["words"] = len(s["text"].split())
        s["full"] = full
    return out


def index(md, tok, mdl, cap=None):
    """Section, then summarise: prose by model, tables by arithmetic, figures by quotation."""
    S = sectionise(md)
    rows = []
    for i, s in enumerate(S):
        r = {"title": s["title"], "words": s["words"], "parent": s["parent"],
             "container": s["container_only"], "rule": s["heading_confidence"],
             "n_tables": len(s["tables"]), "n_figures": len(s["figures"])}
        r["table_summaries"] = [t for t in (summarise_table(h) for h in s["tables"]) if t]
        r["figure_summaries"] = [interpret(f)[1] for f in s["figures"]]
        r["figure_kinds"] = [interpret(f)[0] for f in s["figures"]]
        over = cap is not None and i >= cap
        if s["text"] and s["words"] > 12 and not s["container_only"] and not over:
            parts = split_parts(tok, s["text"], LIMIT)
            if len(parts) == 1:
                r["summary"], r["parts"] = summarise(tok, mdl, REPO, LIMIT, s["text"]), []
            else:
                ps = [{"label": "part %d of %d" % (n + 1, len(parts)),
                       "words": len(p.split()),
                       "summary": summarise(tok, mdl, REPO, LIMIT, p)}
                      for n, p in enumerate(parts)]
                r["summary"] = summarise(tok, mdl, REPO, LIMIT,
                                         " ".join(p["summary"] for p in ps), longer=True)
                r["parts"] = ps
            r["findings"] = [x["token"] for x in check(r["summary"], s["text"])]
        else:
            r["summary"], r["parts"], r["findings"] = None, [], []
            r["skipped_for_time"] = bool(over and s["words"] > 12)
        p = rules_predict(s["full"]) if s["full"].strip() else {}
        r["cats"] = [c for c in CATS if p.get(c)]
        rows.append(r)
    return rows


def money_check(rows, source):
    """Every dollar amount in a summary, checked against the source. Invented is the number
    that matters: one fabricated figure on a fee schedule makes the whole index unusable."""
    flat = re.sub(r"[\s,]", "", re.sub(r"<[^>]+>", " ", source))
    real = bad = 0
    invented = []
    for r in rows:
        for t in ([r.get("summary") or ""] + [p["summary"] for p in r["parts"]]
                  + r["table_summaries"]):
            for a in re.findall(r"\$\s?[\d,]+(?:\.\d{2})?", t):
                k = a.replace("$", "").replace(" ", "").replace(",", "")
                if len(k) < 2:
                    continue
                if k in flat:
                    real += 1
                else:
                    bad += 1
                    invented.append({"amount": a, "section": r["title"][:40]})
    return real, bad, invented


def main():
    src, label, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    cap = int(os.environ.get("CAP", "0")) or None
    raw = open(src, encoding="utf-8", errors="replace").read()
    os.makedirs(outdir, exist_ok=True)
    tok, mdl = load(REPO)

    from docling.document_converter import DocumentConverter
    t0 = time.time()
    doc_md = DocumentConverter().convert(src).document.export_to_markdown()
    parse_secs = round(time.time() - t0, 1)

    out = {"label": label, "source": os.path.basename(src),
           "source_chars": len(raw),
           "source_tables": len(re.findall(r"<table", raw)),
           "source_figures": raw.count("<::"),
           "docling_parse_seconds": parse_secs,
           "docling_chars": len(doc_md),
           "docling_tables_kept": len(re.findall(r"<table", doc_md)),
           "docling_figures_kept": doc_md.count("<::"),
           "pipelines": {}}
    # PIPE=landing or PIPE=docling runs one side only, so a very large document can be
    # split across two processes and finish in half the wall clock.
    only = os.environ.get("PIPE")
    for name, md in (("landing", raw), ("docling", doc_md)):
        if only and name != only:
            continue
        t0 = time.time()
        rows = index(md, tok, mdl, cap)
        real, bad, inv = money_check(rows, raw)
        out["pipelines"][name] = {
            "seconds": round(time.time() - t0, 1), "rows": rows,
            "n_sections": len(rows),
            "n_prose": sum(1 for r in rows if r["summary"]),
            "n_tables": sum(len(r["table_summaries"]) for r in rows),
            "n_figures": sum(len(r["figure_summaries"]) for r in rows),
            "money_real": real, "money_invented": bad, "invented": inv,
            "sections_with_bad_token": sum(1 for r in rows if r["findings"]),
            "skipped_for_time": sum(1 for r in rows if r.get("skipped_for_time")),
        }
        print("%s/%s: %d sections, %d prose, %d tables, %d figures, %.0fs"
              % (label, name, len(rows), out["pipelines"][name]["n_prose"],
                 out["pipelines"][name]["n_tables"],
                 out["pipelines"][name]["n_figures"],
                 out["pipelines"][name]["seconds"]), flush=True)
    suffix = ("." + only) if only else ""
    json.dump(out, open(os.path.join(outdir, label + suffix + ".json"), "w"), indent=1)
    print("-> %s/%s%s.json" % (outdir, label, suffix))


if __name__ == "__main__":
    main()
