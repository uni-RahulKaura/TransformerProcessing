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
from pipeline.chunking import split_parts, split_for_coverage   # noqa: E402
# The summariser is imported LAZILY, inside index(), because it pulls in torch and
# transformers. The outline stage uses none of that -- it is the section rules and nothing else --
# but importing it here gave outline_only.py a hard dependency on a 2 GB deep-learning stack, so
# "outlines need no model" was true of the algorithm and false of the code. It could not run, and
# therefore could not be VERIFIED, on any machine without torch installed.

REPO, LIMIT = "facebook/bart-large-cnn", 1024
CATS = ["PAY", "EXPIRY", "PARTY", "PERM", "OBLIG"]
HTML_TABLE = re.compile(r"<table\b.*?</table>", re.S | re.I)
PIPE_TABLE = re.compile(r"(?m)(?:^\s*\|.*\|\s*$\n?){2,}")
# A vision description lives inside ONE layout block, so the pattern must not be allowed to cross
# a block boundary. Left unbounded it will, and the consequence is silent content loss rather than
# a parse error: the Tesla Q3-2023 deck contains a degenerate empty marker "<::>", which opens a
# description that has no closer of its own, so the search ran on to the NEXT figure's "::>" and
# swallowed 2,213 characters of the document in between -- including three real Markdown headings
# ("# Artificial Intelligence Software and Hardware", "## Vehicle and Other Software",
# "## Battery, Powertrain & Manufacturing"), each of which appears exactly once in the file and
# all three of which an independent reader's key contains. Nothing flagged it: the file's "<::" and
# "::>" counts balance at 17 each, so a marker-balance check sees a healthy document.
#
# Bounding on the block sentinel fixes the class, not just this instance -- any malformed marker
# now costs at most its own block instead of everything up to the next well-formed one.
VISION = re.compile(r"<::((?:(?!::>)(?!\x00BLK\x00).)*?)::>", re.S)
# ...and an empty marker is furniture: it describes nothing, so there is nothing to keep.
EMPTY_VISION = re.compile(r"<::\s*::>|<::>")
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
    md = EMPTY_VISION.sub("", md)
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
        # split_sections marks a masked block with its own sentinel, "\x00BLK\x00". Whatever
        # survives of it must go too: one summary otherwise read "... New York, NY BLK  Re:".
        prose = prose.replace("\x00BLK\x00", " ").replace("\x00", " ")
        prose = re.sub(r"(?<![A-Za-z])BLK(?![A-Za-z])", " ", prose)
        s["text"] = re.sub(r"\s+", " ", prose).strip()
        # The clause number is structure, and the section's title already carries it. Left at
        # the head of the prose it reaches BART and comes back inside the summary, so a section
        # titled "1." rendered as "1." followed by "1. For any position filled...". Dropped
        # once, here, and only when the title is that same bare number -- a numbered heading
        # with a real title ("2.7 Modifications") never had the problem.
        _t = (s.get("title") or "").strip()
        _n = re.match(r"^(\d{1,2}(?:\.\d{1,3})*)\.?$", _t)
        if _n:
            s["text"] = re.sub(r"^\(?%s\)?[.):]?\s+" % re.escape(_n.group(1)), "", s["text"])
        s["words"] = len(s["text"].split())
        s["full"] = full
    # Figure and table counts only exist once the masked blocks have been handed back, above --
    # so the caption test that depends on them has to run here rather than inside
    # split_sections(), which never sees them.
    from pipeline.sections import drop_figure_captions
    out = drop_figure_captions(out)
    return out


def index(md, tok, mdl, cap=None):
    """Section, then summarise: prose by model, tables by arithmetic, figures by quotation."""
    from pipeline.summarise import summarise            # noqa: E402  (see note at the imports)
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
        # A very SHORT section is quoted, not reworded, because a rewrite of it buys no brevity.
        #
        # The threshold is a trade, and the corpus prices it exactly. Sections at or below N words,
        # as a share of the 2,485 with summaries, and the abstractive rate that remains:
        #
        #     N=20   5.8%  ->  84.1%        N=50   26.7%  ->  63.3%
        #     N=25   9.6%  ->  80.3%        N=60   33.0%  ->  56.9%
        #     N=30  13.1%  ->  76.9%        N=100  54.7%  ->  35.2%
        #
        # 25 is chosen, not tuned: it catches only sections a summary cannot usefully shorten while
        # leaving the abstractive rate above 80%, which is the requirement this work exists to meet.
        # 60 was tried first and cost 33 points of that rate for no measured gain in faithfulness.
        # SHORT_WORDS=0 turns the rule off entirely; raise it if quoting is preferred to rewording.
        short = int(os.environ.get("SHORT_WORDS", "25"))
        if (s["text"] and short and 12 < s["words"] <= short
                and not s["container_only"] and not over):
            from pipeline.safe_abstractive import verbatim as _vb
            r["summary"] = _vb(s["text"], n=3)
            r["summary_extractive"] = r["summary"]
            r["parts"], r["findings"] = [], []
            r["abstractive"] = {"chosen": "extractive-short",
                                "reason": "section is %d words; quoting it is more faithful than "
                                          "rewording it and no longer" % s["words"]}
            p = rules_predict(s["full"]) if s["full"].strip() else {}
            r["cats"] = [c for c in CATS if p.get(c)]
            rows.append(r)
            continue
        if s["text"] and s["words"] > 12 and not s["container_only"] and not over:
            # COVER=0 restores the old behaviour (one pass for anything that fits the window).
            parts = (split_for_coverage(tok, s["text"], LIMIT)
                     if os.environ.get("COVER", "1") != "0"
                     else split_parts(tok, s["text"], LIMIT))
            if len(parts) == 1:
                r["summary"], r["parts"] = summarise(tok, mdl, REPO, LIMIT, s["text"],
                                                    section_words=s["words"]), []
                # BOTH summaries, same model, same section: the extractive-leaning default and the
                # forced-abstractive variant. Emitted side by side deliberately rather than one
                # being chosen here, because the choice is a real trade and the evidence is split:
                # blind judges put invention at 2 of 35 for the default and 8 of 35 for the
                # abstractive one, while the abstractive one was the ONLY candidate of four to state
                # a contract's three-year term where the default reproduced a page footer. A reader
                # comparing them on their own documents can settle it; a default buried in code
                # cannot.
                # ABSTRACTIVE BY DEFAULT, with figures guarded. Generate a second draft with the
                # source's prose blocked, then keep it unless it introduced a figure, date or name
                # the section does not contain -- an invented figure is deleted and the rewrite
                # kept; an invented name forces the faithful version, because a name cannot be
                # removed without changing who the sentence is about.
                # Measured on 24 sections of four document types: 88% published as a rewrite,
                # copy rate 0.257 against the 0.696 all-extractive baseline.
                if os.environ.get("ABSTRACTIVE", "1") != "0":
                    import pipeline.summarise as _sm
                    from pipeline.safe_abstractive import choose as _choose
                    _keep = _sm.ENC_NO_REPEAT
                    try:
                        _sm.ENC_NO_REPEAT = int(os.environ.get("ABS_N", "6"))
                        _draft = summarise(tok, mdl, REPO, LIMIT, s["text"],
                                           section_words=s["words"])
                    finally:
                        _sm.ENC_NO_REPEAT = _keep
                    _pub, _rec = _choose(r["summary"], _draft, s["text"])
                    r["summary_extractive"] = r["summary"]
                    r["summary"] = _pub
                    r["abstractive"] = _rec
                    r["findings"] = [x["token"] for x in check(_pub, s["text"])]
            else:
                ps = [{"label": "part %d of %d" % (n + 1, len(parts)),
                       "words": len(p.split()),
                       "summary": summarise(tok, mdl, REPO, LIMIT, p)}
                      for n, p in enumerate(parts)]
                _stitch = " ".join(p["summary"] for p in ps)
                # the COMBINING step is where coverage was being thrown away: the parts covered
                # the whole section, then this call squeezed them back into a fixed 80 tokens, so a
                # 400-word clause still came out as 34 words. It now scales with the section.
                r["summary"] = summarise(tok, mdl, REPO, LIMIT, _stitch, longer=True,
                                         section_words=s["words"])
                # A long section gets the same treatment as a short one. Without this the
                # abstractive path covered only sections that fit BART's window in one pass -- and
                # long sections are exactly where "the summary is just the opening" hurts most.
                if os.environ.get("ABSTRACTIVE", "1") != "0":
                    import pipeline.summarise as _sm2
                    from pipeline.safe_abstractive import choose as _choose2
                    _k2 = _sm2.ENC_NO_REPEAT
                    try:
                        _sm2.ENC_NO_REPEAT = int(os.environ.get("ABS_N", "6"))
                        _d2 = summarise(tok, mdl, REPO, LIMIT, _stitch, longer=True,
                                        section_words=s["words"])
                    finally:
                        _sm2.ENC_NO_REPEAT = _k2
                    # checked against the WHOLE section, not the stitched part summaries: a figure
                    # absent from the section is invented even if a part summary repeated it.
                    _p2, _r2 = _choose2(r["summary"], _d2, s["text"])
                    r["summary_extractive"] = r["summary"]
                    r["summary"] = _p2
                    r["abstractive"] = _r2
                    r["findings"] = [x["token"] for x in check(_p2, s["text"])]
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
    from pipeline.summarise import load                 # noqa: E402  (lazy: see the imports)
    tok, mdl = load(REPO, "abstractive")

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
