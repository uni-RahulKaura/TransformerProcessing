"""Prepare Landing AI output for the section rules.

Landing's Markdown is the best input we have measured: it keeps all 32 tables with their rows
and cells intact, and it carries vision-model descriptions of the document's Gantt charts,
diagrams and signature blocks -- about 5.8% of the document that no other parser can recover
at all, because reading it requires looking at an image.

But feeding it straight to the section rules produces 326 sections where the document has 44.
Three causes, all in the input rather than the rules:

  PAGE FURNITURE. A DocuSign envelope banner repeats on every page. Each copy looks exactly
  like a short title-shaped line, so rule 1b promotes all forty of them to sections.

  TABLE CONTENTS. Landing preserves tables as real HTML. Cell text is short and often
  capitalised, which is precisely the shape rule 1b looks for, so column headers and cell
  values become sections. The rules must not look inside a table at all.

  ANCHOR IDS. Landing emits an anchor before most blocks. Useful for citation, meaningless
  as content.

Handling all three is what makes "convert through Landing first" actually work. The tables
and the image descriptions are the whole point of routing through Landing, so they are kept
in the section BODY -- only hidden from the heading rules.
"""
import re

ANCHOR = re.compile(r"<a id=\'[^\']*\'></a>\s*")
FURNITURE = [
    re.compile(r"(?m)^\s*DocuSign Envelope ID:.*$"),
    re.compile(r"(?m)^\s*Page \d+\s*(of\s*\d+)?\s*$"),
]
TABLE = re.compile(r"<table\b.*?</table>", re.S | re.I)
VISION = re.compile(r"<::.*?::>", re.S)


def prepare(md):
    """Return (text_for_heading_rules, restore) with tables and image blocks masked.

    Masking rather than deleting: the rules cannot see inside a table, but the text is put
    back before summarising, so BART still reads the tables and the image descriptions.
    """
    md = ANCHOR.sub("", md)
    for rx in FURNITURE:
        md = rx.sub("", md)
    store = {}

    def hide(m):
        key = "MASK%04d" % len(store)
        store[key] = m.group(0)
        return key

    # A placeholder alone on a line looks exactly like a short title, so the heading rules
    # promoted every masked table to a section called MASK0000. Keeping the placeholder
    # inline behind a marker word stops that.
    md = TABLE.sub(lambda m: "(table) " + hide(m), md)
    md = VISION.sub(lambda m: "(figure) " + hide(m), md)
    md = re.sub(r"\n{3,}", "\n\n", md)

    def restore(t):
        for k, v in store.items():
            t = t.replace(k, v)
        return t

    return md, restore


def split_landing(md):
    """Section a Landing document: mask, apply the normal rules, then restore.

    Each section comes back with its tables and figures listed separately from its prose, so
    the two can be summarised by the tool that suits them: arithmetic for the tables, a
    language model for the prose.
    """
    from .sections import split_sections
    masked, restore = prepare(md)
    out = split_sections(masked)
    for s in out:
        full = restore(s["text"])
        s["tables"] = re.findall(r"<table\b.*?</table>", full, re.S | re.I)
        s["figures"] = re.findall(r"<::(.*?)::>", full, re.S)
        prose = TABLE.sub(" ", full)
        prose = VISION.sub(" ", prose)
        prose = re.sub(r"<[^>]+>", " ", prose)
        prose = re.sub(r"\(table\)|\(figure\)|MASK\d{4}", " ", prose)
        s["text"] = re.sub(r"\s+", " ", prose).strip()
        s["words"] = len(s["text"].split())
        s["full"] = full
    return out
