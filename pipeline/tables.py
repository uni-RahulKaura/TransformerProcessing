"""Summarise a table by reading its columns, not by summarising its text.

A prose summariser handed a table reads it row by row and produces mush. Measured on the fee
schedule of a real contract, BART ran figures together out of order, repeated itself, and
INVENTED a dollar amount -- it wrote $353.00 where the table says $353,377. One fabricated
figure in twenty-two, on a fee schedule, is not usable.

So nothing here is generated. Every figure is read or computed from the actual cells, which
means a wrong number is impossible by construction.

Three things this has to get right, learned by getting them wrong:

  TOTAL ROWS AND COLUMNS. A first version summed the "Total Estimated Fees" row along with
  the line items and reported a total exactly double the real one. Rows and columns whose
  label says total are held aside.

  MIXED UNITS. The same fee table has a headcount row (8.5, 7.5 FTE) among the money rows.
  Treated as money it produced "from $8.50", which is meaningless. A row is only money if
  its own cells carry a currency mark.

  THE TABLE'S OWN ARITHMETIC. Once total rows are separated, the line items can be added up
  and compared against the total the table states. When they agree, the total is worth
  reporting as fact. When they do not, that is worth saying -- it means the table does not
  add up, which a reader would want to know and which no summariser would ever notice.
"""
import re
from html.parser import HTMLParser


class _Table(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell, self._in = [], [], [], False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._in, self._cell = True, []

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._in = False
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
        elif tag == "tr" and self._row:
            self.rows.append(self._row)

    def handle_data(self, d):
        if self._in:
            self._cell.append(d)


AMOUNT = re.compile(r"^\(?\s*(?:US)?\$\s*(-?[\d,]+(?:\.\d+)?)\s*\)?$")
PLAIN = re.compile(r"^\(?\s*(-?[\d,]+(?:\.\d+)?)\s*\)?$")
TOTALISH = re.compile(r"\b(total|subtotal|sum|grand)\b", re.I)


PIPEROW = re.compile(r"(?m)^\s*\|(.+)\|\s*$")
SEPROW = re.compile(r"^[\s|:\-]+$")


def parse_pipe(md):
    """Markdown pipe tables. docling converts HTML tables to this form, so without it the
    docling side of a comparison appears to have no tables at all -- which would credit
    Landing with a difference the parsers do not actually have."""
    out = []
    for line in md.split("\n"):
        m = PIPEROW.match(line)
        if not m:
            if out:
                break
            continue
        if SEPROW.match(m.group(1)):
            continue                      # the |---|---| divider is not data
        out.append([c.strip() for c in m.group(1).split("|")])
    return out


def parse(html):
    if "<table" not in html.lower() and "|" in html:
        return parse_pipe(html)
    p = _Table()
    p.feed(html)
    return [r for r in p.rows if any(c for c in r)]


def _money(cell):
    """A value only counts as money if its own cell carries a currency mark."""
    c = (cell or "").strip()
    m = AMOUNT.match(c)
    if not m:
        return None
    if c.replace(" ", "") in ("$-", "$"):
        return 0.0
    try:
        v = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    return -v if c.startswith("(") else v


def _money_or_dash(cell):
    c = (cell or "").strip().replace(" ", "")
    return 0.0 if c in ("$-", "$", "-") else _money(cell)


def _fmt(v):
    if v is None:
        return "-"
    return "$" + ("{:,.0f}".format(v) if abs(v - round(v)) < 0.005
                  else "{:,.2f}".format(v))


PERIOD = re.compile(r"\b(FY\s?\*?\s?\d{2,4}|Q[1-4]|20\d\d|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|"
                    r"Sep|Oct|Nov|Dec|month|quarter|year|week|wave|phase)\b", re.I)
ROLE = re.compile(r"\b(responsib|accountab|owner|RACI|supplier|client|customer|accenture|"
                  r"medtronic|vendor|party)\b", re.I)
MONEYWORD = re.compile(r"\b(fee|fees|cost|costs|price|pricing|rate|charge|charges|amount|"
                       r"payment|invoice|\$|USD)\b", re.I)


def purpose(header, body, caption=""):
    """One sentence saying what the table is FOR, inferred from its own column names.

    "A table of 6 rows and 9 columns" tells a reader the shape and nothing about the point of
    it. What they want to know first is whether they are looking at a fee schedule, a
    timetable, a list of deliverables or a split of who does what -- and the column headers
    say which, without any need to guess.
    """
    cols = [c for c in (header or []) if c]
    joined = " ".join(cols)
    periods = [c for c in cols[1:] if PERIOD.search(c)]
    # A stray "$" inside a report NAME is not a price. Requiring the cell to parse as an
    # amount stops an inventory of 50 reports being announced as a pricing table.
    money = bool(MONEYWORD.search(joined)) or any(
        _money(c) is not None for r in body for c in r[1:])
    rowlab = [(r[0] or "").strip() for r in body if (r[0] or "").strip()]
    n = len(body)

    if money and periods:
        unit = ("financial year" if re.search(r"FY", joined) else
                "quarter" if re.search(r"\bQ[1-4]\b", joined) else
                "month" if re.search(r"month|Jan|Feb|Mar", joined, re.I) else "period")
        return ("This is a cost breakdown: %d cost lines priced across %d %ss."
                % (len(rowlab), len(periods), unit))
    if money:
        return "This is a pricing table: %d rows with amounts against them." % n
    if ROLE.search(joined) and any(
            (c or "").strip().upper() in ("X", "\u2713", "YES") for r in body for c in r):
        return ("This is a split of who does what: %d activities, ticked against the party "
                "responsible." % n)
    if periods and len(periods) >= 2:
        unit = ("wave" if re.search(r"wave", joined, re.I) else
                "phase" if re.search(r"phase", joined, re.I) else "period")
        return "This is a timetable: %d rows scheduled across %d %ss." % (n, len(periods), unit)
    if re.search(r"\b(deliverable|milestone|activity|activities|task)\b", joined, re.I):
        return "This is a list of deliverables or activities: %d of them." % n
    if re.search(r"\b(report|application|system|tool|document)\b", joined, re.I):
        thing = re.search(r"\b(report|application|system|tool|document)\b", joined, re.I)
        return "This is an inventory: %d %ss listed, one per row." % (n, thing.group(1).lower())
    if caption.strip():
        return "This table covers %s." % caption.strip()[:80]
    return None


def summarise_table(html, caption=""):
    """A few plain sentences describing a table, built only from its own cells."""
    rows = parse(html)
    if not rows:
        return None
    ncol = max(len(r) for r in rows)
    # A long table split across pages carries its header only on the first page. On every
    # later page the first row is data, and reading it as column names produced nonsense
    # like 'Across: APAC, MITG, ANZ'. If the first cell of row 0 is a bare number, there is
    # no header here.
    first = (rows[0][0] or "").strip() if rows[0] else ""
    headerless = bool(re.match(r"^\d+$", first))
    header = [] if headerless else (rows[0] if any(c for c in rows[0]) else [])
    body = rows[1:] if header else rows
    if not body:
        return "A table of %d column%s and no data rows." % (ncol, "" if ncol == 1 else "s")

    label = lambda r: (r[0].strip() if r and r[0].strip() else "")
    items = [r for r in body if not TOTALISH.search(label(r))]
    totals = [r for r in body if TOTALISH.search(label(r))]
    # a column is a total column if its header says so
    tcols = {j for j, h in enumerate(header) if TOTALISH.search(h or "")}

    out = []
    if headerless:
        out.append("A continuation of the table before it: the same columns, %d more rows "
                   "(%s to %s)."
                   % (len(body), first, (body[-1][0] or "?").strip()))
    lead = purpose(header, body, caption)
    if lead:
        out.append(lead)
    what = caption.strip()
    out.append("A table of %d row%s and %d column%s%s."
               % (len(body), "" if len(body) == 1 else "s", ncol,
                  "" if ncol == 1 else "s", (" -- " + what[:90]) if what else ""))
    names = [label(r) for r in items if label(r)]
    if names:
        out.append("It lists: %s." % ", ".join(n[:40] for n in names[:8])
                   + (" (and %d more)" % (len(names) - 8) if len(names) > 8 else ""))
    if header and any(header[1:]):
        cols = [h for j, h in enumerate(header[1:], 1) if h and j not in tcols]
        if cols:
            out.append("Across: %s." % ", ".join(c[:26] for c in cols[:9]))

    # money only, and only from the line-item rows
    vals = []
    for r in items:
        for j, c in enumerate(r):
            if j == 0 or j in tcols:
                continue
            v = _money(c)
            if v is not None:
                vals.append(v)
    if vals:
        out.append("Amounts run from %s to %s." % (_fmt(min(vals)), _fmt(max(vals))))

    # the table's own stated total, checked against the line items
    stated = None
    for r in totals:
        for j in sorted(tcols) or [len(r) - 1]:
            if j < len(r):
                stated = _money(r[j]) or stated
    if stated is None and tcols:
        for r in items:
            pass
    if stated is not None:
        computed = 0.0
        for r in items:
            for j in sorted(tcols):
                if j < len(r):
                    v = _money_or_dash(r[j])
                    if v is not None:
                        computed += v
        if computed and abs(computed - stated) < max(2.0, 0.005 * abs(stated)):
            out.append("The table states a total of %s, and the line items add up to that."
                       % _fmt(stated))
        elif computed:
            out.append("The table states a total of %s, but its line items add up to %s -- "
                       "a difference of %s, so the table does not reconcile."
                       % (_fmt(stated), _fmt(computed), _fmt(abs(stated - computed))))
        else:
            out.append("The table states a total of %s." % _fmt(stated))
    return " ".join(out)
