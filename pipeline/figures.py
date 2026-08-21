"""Turn a vision description of a figure into a statement of what the figure MEANS.

Landing's vision model describes what it SEES: "Medtronic Systems (blue square)", "centered
in the image", "Short description of visual element". That is the right output for a parser
and the wrong output for an index. Nobody navigating a contract needs to know a box is
yellow; they need to know the diagram says which systems each side runs.

So this reads the description and reports the content, dropping the presentation. Three
figure kinds cover everything in the contracts we have looked at, and each has a different
answer to "what does it mean":

  A SIGNATURE BLOCK means: who signed, in what role, for which party, on what date. That is
  the whole content. The fact that a signature is legible and centred is not.

  A SCHEDULE OR GANTT CHART means: the period it covers, and the dated milestones in it. The
  bar colours carry no information a reader can act on.

  A SYSTEM DIAGRAM means: which things exist and how they are grouped or connected. The
  legend's colours are how the grouping is DRAWN; the grouping itself is the content.

Anything it cannot classify is passed through with the visual language stripped, which is
still an improvement, and flagged so nobody mistakes it for an interpretation.
"""
import re

VISUAL = re.compile(
    r"\b(blue|dark green|light green|green|yellow|red|orange|grey|gray|white|black)\s+"
    r"(square|box|boxes|bar|bars|line|arrow|arrows|circle|shading|highlight)\b|"
    r"\b(centered|centred) in the image\b|"
    r"\bShort description of visual element.*|"
    r"\bThe diagram is divided into\b|"
    r"\b(legible|illegible)\b|"
    r"\bwith the text \"[^\"]*\" above it\b|"
    r"\ban alphanumeric code below it\b", re.I)
DATE = re.compile(r"\b\d{1,2}[-/](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
                  r"(?:[-/]\d{2,4})?\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                  r"[a-z]*[-/ ]\d{2,4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|"
                  r"\b(?:January|February|March|April|May|June|July|August|September|October|"
                  r"November|December)\s+\d{4}\b", re.I)
MILESTONE = re.compile(r"\b([A-Z][A-Za-z ]{1,24}?)\s+(?:start|Start|go[- ]?live|GoLive|end)\b"
                       r"[: ]*\s*(" + DATE.pattern + r")", re.I)


# Field labels that a loose regex mistakes for a person. "Unsigned by By." was the result.
STOPWORDS = ('By', 'Title', 'Name', 'Date', 'Print', 'Sign', 'Customer', 'Signature', 'Readable', 'Short', 'This', 'Agreed', 'Accepted', 'Contractor')

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
BARE_MS = re.compile(r"\b(KT start|SRT start|GoLive|Go[- ]Live|Kick[- ]?off)\b[: ]*\s*("
                     + DATE.pattern + r")", re.I)


def _when(txt):
    """(sortable key, original text) for a date string, or None if it cannot be ordered."""
    t = txt.lower()
    mon = next((MONTHS[k] for k in MONTHS if k in t), None)
    if mon is None:
        m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", t)
        if not m:
            return None
        y = int(m.group(3)); y += 2000 if y < 100 else 0
        return ((y, int(m.group(1)), int(m.group(2))), txt)
    # The year must be looked for AFTER the month name. Searching the whole string matched
    # the DAY in "10-Feb-20" and dated the milestone to the year 2010.
    tail = t.split(next(k for k in MONTHS if k in t), 1)[1]
    y = re.search(r"(20\d\d|\d\d)", tail)
    if not y:
        return None          # no year at all: "2-Dec" cannot be ordered against "August 2020"
    yr = int(y.group(1))
    yr += 2000 if yr < 100 else 0
    dm = re.match(r"\s*(\d{1,2})[-/ ]", t)
    return ((yr, mon, int(dm.group(1)) if dm else 0), txt)


def _clean(t):
    t = VISUAL.sub(" ", t or "")
    t = re.sub(r"\(\s*\)", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\s+([,.;:])", r"\1", t).strip(" -–—:;,")


def interpret(desc):
    """Return (kind, meaning). `meaning` says what the figure tells you."""
    d = " ".join((desc or "").split())
    if not d:
        return "unknown", ""
    low = d.lower()

    # ---- a signature block: who signed, for whom, in what role, when
    if "digital signature" in low or "docusigned by" in low or low.startswith("attestation"):
        unsigned = bool(re.search(r"\bunsigned\b|\[unsigned\]|do not sign", d, re.I))
        signed = "Unsigned" if unsigned else "Signed"
        # Landing writes the name in at least five different shapes. Reading only one of them
        # threw away 83 of 119 signatures across the corpus -- the information was in the file
        # the whole time and the fault was here, not in the parser that produced it.
        who = None
        for rx in (r"DocuSigned by:\s*([A-Z][A-Za-z.\-']+(?:\s+(?!Title|Name|Date|By|Print|Sign)[A-Z][A-Za-z.\-']+){0,3})",
                   r"Signature:\s*legible\s*\(([^)]{3,44})\)",
                   r"Signature:\s*([A-Z][A-Za-z.\-']+(?:\s+(?!Title|Name|Date|By|Print|Sign)[A-Z][A-Za-z.\-']+){0,3})\s*\(legible\)",
                   r"/s/\s*([A-Z][A-Za-z.\-']+(?:\s+(?!Title|Name|Date|By|Print|Sign)[A-Z][A-Za-z.\-']+){0,3})",
                   r"\bName:\s*([A-Z][A-Za-z.\-']+(?:\s+(?!Title|Name|Date|By|Print|Sign)[A-Z][A-Za-z.\-']+){0,3})",
                   r"representative\s+([A-Z][A-Za-z.\-']+(?:\s+(?!Title|Name|Date|By|Print|Sign)[A-Z][A-Za-z.\-']+){0,3})"):
            m = re.search(rx, d)
            cand = m.group(1).strip().rstrip(" .,") if m else ""
            if cand and cand.split()[0] not in STOPWORDS and cand not in STOPWORDS:
                who = cand
                break
        title = None
        for rx in (r"\bTitle:\s*([A-Za-z][A-Za-z ,&/.\-']{2,54}?)(?:\s{2,}|\s*(?:Date|Short|This|$))",
                   r"Printed name\s+([A-Za-z][A-Za-z ,&/.\-']{2,44}?)\s+Title",
                   r"(?:^|\s)((?:Chief|Senior|Executive|Vice|Managing|General|Authorized)"
                   r"[A-Za-z ]{3,44}?)(?:\s{2,}|\s*(?:Date|This|Short|$))"):
            m = re.search(rx, d)
            if m and m.group(1).strip(" _-") not in ("", "___"):
                title = m.group(1).strip(" _-,")
                break
        party = None
        for rx in (r"Readable Text:\s*([A-Z][A-Z .,&\'-]{3,60}?)\s+(?:DocuSigned|By:)",
                   r"\b([A-Z][A-Z .,&\'-]{4,58}?(?:INC|LLC|LTD|CORP|CORPORATION|COMPANY|L\.?P)\.?)\b"):
            m = re.search(rx, d)
            if m:
                party = m.group(1).strip().rstrip(",")
                break
        when = DATE.search(d)
        bits = []
        if party:
            bits.append("for %s" % party)
        if who:
            bits.append("by %s" % who)
        if title:
            bits.append("(%s)" % title)
        if when:
            bits.append("on %s" % when.group(0))
        if not bits:
            return "signature", ("%s signature block. Landing did not record a name in it -- "
                                 "the block itself may be a blank template." % signed)
        return "signature", "%s %s." % (signed, " ".join(bits))

    # ---- a schedule: the period, and the dated milestones
    if "gantt" in low or "timeline" in low or "transition plan" in low:
        dates = [m.group(0) for m in DATE.finditer(d)]
        ms = ["%s %s" % (a.strip(), b) for a, b in MILESTONE.findall(d)][:6]
        ms += ["%s %s" % (a, b) for a, b in BARE_MS.findall(d)][:4]
        ms = list(dict.fromkeys(m.strip() for m in ms))[:6]
        # Order the span CHRONOLOGICALLY, not by position in the text. Taking the first and
        # last date as they appear reported a Gantt chart as ending in March when its own
        # go-live was in July -- the description simply mentions dates out of order.
        keyed = sorted((k for k in (_when(x) for x in dates) if k))
        span = ("It runs %s to %s." % (keyed[0][1], keyed[-1][1])) if len(keyed) > 1 else ""
        what = re.split(r"[.:]", _clean(d))[0][:90]
        out = "A schedule. %s %s" % (what, span)
        if ms:
            out += " Dated milestones: %s." % "; ".join(ms)
        elif dates:
            out += " %d dated points in it." % len(dates)
        return "schedule", " ".join(out.split())

    # ---- a system diagram: what exists and how it is grouped
    if "diagram" in low or "landscape" in low or "network" in low:
        groups = re.findall(r"-\s*([A-Z][A-Za-z /&]{3,44}?)\s*\((?:[a-z ]+)\s*(?:square|box)\)", d)
        halves = re.findall(r"([A-Z][A-Za-z]+ Managed)", d)
        named = re.findall(r"\b([A-Z][A-Za-z]{2,}(?: [A-Z][A-Za-z]{2,}){0,3})\b", _clean(d))
        out = "A diagram of systems and how they are grouped."
        if groups:
            out += " It separates: %s." % ", ".join(dict.fromkeys(groups))
        if halves:
            out += " Split between %s." % " and ".join(dict.fromkeys(halves))
        if not groups and not halves:
            keep = [n for n in dict.fromkeys(named)
                    if n.lower() not in ("the", "legend", "diagram", "section")][:8]
            if keep:
                out += " Things named in it: %s." % ", ".join(keep)
            else:
                return "diagram", ("A diagram, but the description carries no readable "
                                  "content - it needs re-reading at higher resolution.")
        return "diagram", out

    c = _clean(d)
    if len(c) < 25:
        return "unreadable", ("The description is too thin to interpret (%r) - this figure "
                              "needs re-reading." % c)
    return "other", c[:300]
