"""Abstractive where it is safe, faithful where it is not -- decided per section, not globally.

THE REQUIREMENT: reword rather than quote, and never misstate a figure, date or party name.

WHAT WAS TRIED AND FAILED, so nobody repeats it:

 1. `encoder_no_repeat_ngram_size` alone. It does abstract -- copy rate 0.696 -> 0.264 at n=6 --
    but it bans the input's wording indiscriminately, including the digits. Three independent blind
    judges reading only the source counted inventions out of 35 at 8 / 20 / 15 for n=6 against
    2 / 7 / 6 unconstrained, and all three found the gradient monotone. The fabrications were
    exactly what a blunt ban predicts: a company number 00420028 for 01420028, an invented
    "correspondence team@verizon.co.uk" replacing a real Albany PO Box, a $1.4B cash flow where the
    document says $0.7B.

 2. Exempting fact-bearing TOKENS from the ban (pipeline/abstractive.py). This does not work, and
    the reason is structural: an n-gram ban blocks the token that would CONTINUE a matching run, so
    copying "decreased 9% to $19.3 billion" needs the connecting words permitted too. Protecting
    "$" and "19" while "billion" stays banned still forces the model off the span. Measured output
    was byte-identical to the blunt ban, with wrong figures on 2 of 18 sections against 1 of 18.

WHAT THIS DOES. Generate both, then choose per section:

    abstractive draft  ->  does it contain a figure, date or capitalised name absent from the
                           section?  ->  no: keep it.  yes: keep the faithful summary instead.

So a section whose abstraction is clean reads as a rewrite, and a section where the model started
inventing falls back rather than shipping a wrong number. The choice is recorded per section, so the
proportion is visible rather than assumed, and a reader can see which summaries were rewritten.

WHAT THIS STILL DOES NOT CATCH. A figure copied correctly but attributed wrongly -- "the total IS
$17.41" where the document says the total INCLUDES $17.41 -- passes every test here, because every
token is present. Three judges found exactly that case. Nothing token-level sees it, so
pipeline/verify.py still runs and a flag remains a prompt to read.
"""
import os
import re

from . import polarity

NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
DATE = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
                  r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2})\b", re.I)
# A name-like token: two or more capitals, or Capitalised words joined. Deliberately loose, because
# a false alarm here only costs abstraction on that one section, while a miss ships a wrong name.
NAME = re.compile(r"\b(?:[A-Z]{2,}[A-Za-z&.'-]*|[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b")
GENERIC = set("""The This That These Those Section Clause Agreement Party Parties Provider Customer
Supplier Recipient Company Services Service Order Term Terms Exhibit Schedule Annex Appendix
Confidential Information Data Effective Date Total Account Invoice Payment Notice January February
March April May June July August September October November December Monday Tuesday Wednesday
Thursday Friday Saturday Sunday
They Them Their Theirs It Its We Us Our You Your He Him His She Hers
Neither Either Nothing None Nobody No Not Notwithstanding Accordingly However Moreover
Furthermore Therefore Thus Hence Save Except Unless Subject Upon During Prior Where When While
If Each Every Any All Both Such Should Whether Provided Failure Nor And Or But For In On At To
Under Over After Before Once Although Because Since Also Further Additionally Finally
Any""".split())
# The second block is closed-class English, not entities. They reach the name check only because
# they are capitalised at the start of a sentence, and BART starts sentences with them
# constantly: "Neither party shall ..." was reported as an invented party name in three of one
# NDA's eight fallbacks, which is three sections needlessly demoted from a rewrite to a
# quotation. A connective can never be the name of a party.


def _plural_forms(t):
    """A defined term and the same term pluralised are the same entity.

    "Authorised Third Parties" is defined in the section; the summary wrote "Authorised Third
    Party" and was accused of inventing a party. rstrip("s") did not help -- it turns "Parties"
    into "Partie" -- so the last word is inflected properly here, both ways.
    """
    head, _, last = t.rpartition(" ")
    out = {t}
    for w in ({last} if last else {t}):
        alts = set()
        if w.endswith("ies") and len(w) > 4:
            alts.add(w[:-3] + "y")
        if w.endswith("y") and len(w) > 2:
            alts.add(w[:-1] + "ies")
        if w.endswith("s") and len(w) > 3:
            alts.add(w[:-1])
        else:
            alts.add(w + "s")
        for a in alts:
            out.add((head + " " + a).strip() if head else a)
    return out


def _norm_num(s):
    return s.replace("$", "").replace(",", "").rstrip("%")


# Abbreviations that legitimately end in a period mid-sentence, so a lower-case word after them
# is not a repair opportunity.
ABBREV = {"no", "nos", "inc", "ltd", "llc", "plc", "co", "corp", "etc", "eg", "ie", "vs", "cf",
          "approx", "est", "art", "sec", "cl", "para", "pp", "vol", "ch", "mr", "mrs", "ms", "dr"}
STRAY = re.compile(r"(?<=[a-z])\.\s+(?=[a-z])")


CAPS_SLIP = re.compile(r"\b([A-Z]{2,})([a-z])([A-Z]{2,})\b")
# BART loses the space AFTER a short function word, so the head is always one of these and the
# tail may be capitalised. Real output carried "thedisclosing", "aThird", "asimilar" and
# "allreasonable" -- four different heads, which is why this is a list rather than one or two.
# BART loses the space AFTER a short function word. Real output carried "thedisclosing",
# "aThird", "asimilar" and "allreasonable" -- four different heads.
_HEADS = frozenset(("a an the of to in for by with and all any its our their his her this that "
                    "each such no not is are be was were from on at as").split())
GLUE = re.compile(r"\b[A-Za-z]{5,}\b")


def tidy(text, section=None):
    """Repair the surface artifacts BART leaves in a summary. Wording is never touched.

    Every one of these is from real output on one NDA, where 4 of 35 summaries carried at least
    one and each looked like a defect to a reader:

      "responsible. for any breach"        a full stop dropped mid-clause
      "England and Wales.."                a doubled full stop
      "NDA STUDiOS DISTRIBUTION"           a lower-case letter inside a run of capitals
      "**Authorised Third Party** means"   the section's Markdown emphasis carried through
      "provided by thedisclosing Party"    a lost space between two words

    The last is the only risky one, so it is repaired only when BOTH halves of the split appear
    in the section and the glued form does not -- "therefore" and "these" are real words and must
    survive untouched, and the section itself decides.
    """
    if not text:
        return text
    text = text.replace("**", "").replace("__", "")
    text = CAPS_SLIP.sub(lambda m: m.group(1) + m.group(2).upper() + m.group(3), text)
    text = re.sub(r"\.\s+\.", ".", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    out, last = [], 0
    for m in STRAY.finditer(text):
        # the slice must END at the period, not past the whitespace after it, or the
        # anchor never matches and the abbreviation guard never fires
        word = re.search(r"([A-Za-z]+)\.$", text[:m.start() + 1])
        if word and word.group(1).lower() in ABBREV:
            continue
        out.append(text[last:m.start()])
        out.append(" ")
        last = m.end()
    out.append(text[last:])
    text = re.sub(r"\s{2,}", " ", "".join(out)).strip()
    if section:
        # WORDS, not substrings. Matching substrings let "imilar" satisfy the lookup because it
        # sits inside "similar", so "asimilar" was split as "as imilar". And every split point is
        # tried rather than whichever one the pattern happened to pick first, because alternation
        # order cannot know that "allreasonable" splits after three letters and "asimilar" after
        # one.
        vocab = set(re.findall(r"[A-Za-z]+", section.lower()))

        def unglue(m):
            w = m.group(0)
            if w.lower() in vocab:              # a word the section itself uses -- leave it
                return w
            for i in range(1, len(w) - 2):
                if w[:i].lower() in _HEADS and w[i:].lower() in vocab:
                    return w[:i] + " " + w[i:]
            return w
        text = GLUE.sub(unglue, text)
    return text


def unsupported(summary, section):
    """Figures, dates and names in the summary that the section does not contain."""
    if not summary:
        return []
    src = re.sub(r"\s+", " ", section)
    src_l = src.lower()
    bad = []
    src_nums = {_norm_num(x) for x in NUM.findall(src)}
    for m in NUM.finditer(summary):
        if _norm_num(m.group(0)) not in src_nums:
            bad.append(("figure", m.group(0)))
    for m in DATE.finditer(summary):
        d = m.group(0).lower()
        if d not in src_l:
            # allow an expanded month abbreviation the section does contain
            head = re.match(r"[a-z]+", d)
            if not (head and head.group(0)[:3] in src_l):
                bad.append(("date", m.group(0)))
    for m in NAME.finditer(summary):
        t = m.group(0).strip()
        if t in GENERIC or len(t) < 3:
            continue
        # A run of capitalised words this long is a contents list, not a party. The Tesla decks
        # put "Highlights Financial Summary Operations Summary Vehicle Capacity Core Technology
        # ..." on one line, the pattern swallowed all of it as a single name, and no section
        # contains that exact 20-word string -- so three decks lost a rewrite to a table of
        # contents. Six words is past any party name in this corpus.
        if len(t.split()) > 6:
            continue
        # possessives and hyphenation are FORM, not fact. "TESLA's" and "US-built" were both
        # reported as invented names while TESLA and "US built" were in the section -- two of the
        # four rejections in the first measured run, i.e. half the lost abstraction was my own
        # false alarm.
        # the possessive is stripped explicitly rather than with rstrip("s"), which would turn
        # "Parties" into "Partie" -- the bug that made a defined plural look invented
        core = re.sub(r"['\u2019]s?$", "", t)
        # A whole phrase of generic words is not a name: "The Agreement" reached here because the
        # pattern matches the PHRASE, and only its individual words were in GENERIC.
        if all(w in GENERIC for w in core.split()):
            continue
        flat = t.replace("-", " ").lower()
        forms = {f.lower() for f in _plural_forms(core)} | {t.lower(), flat}
        # BART puts an article in front of a defined term the section states bare. "The Agreement"
        # against a section that says "Agreement", and "The Fee Agreement" against "Fee
        # Agreement", were reported as invented parties -- 6 of 25 fallbacks in eleven documents,
        # i.e. six sections demoted from a rewrite to a quotation over a word the model added.
        bare = re.sub(r"^(?:The|A|An)\s+", "", core)
        if bare != core:
            forms |= {f.lower() for f in _plural_forms(bare)}
        # the section's own hyphens read as spaces, so "Mutual Non" matches "Mutual Non-Disclosure"
        spaced = src_l.replace("-", " ")
        # and word spacing inside a defined term is form, not fact: a sublease that defines
        # "Subtenant" was reported as not containing "Sub Tenant" five times in one document.
        squashed = re.sub(r"[\s-]", "", src_l)
        if (any(f in src_l for f in forms) or any(f in spaced for f in forms)
                or any(re.sub(r"[\s-]", "", f) in squashed for f in forms)):
            continue
        bad.append(("name", t))
    seen, out = set(), []
    for k, v in bad:
        if (k, v) not in seen:
            seen.add((k, v)); out.append({"kind": k, "token": v})
    return out


STRIPPABLE = {"figure", "date"}


def repair(abstractive, section):
    """Remove an unsupported figure or date from the rewrite, keeping the rewrite.

    Falling back to the extractive summary throws away the paraphrase, which is the thing that was
    asked for. Deleting the offending number keeps the prose and loses only the claim that was
    wrong -- and a sentence left dangling by the deletion is dropped whole, so the result still
    reads. A name cannot be deleted this way without changing who the sentence is about, so an
    unsupported NAME still forces the fallback.
    """
    bad = unsupported(abstractive, section)
    if not bad:
        return abstractive, []
    if any(b["kind"] == "name" for b in bad):
        return None, bad
    out = abstractive
    for b in bad:
        out = out.replace(b["token"], "")
    # tidy the wreckage: doubled spaces, orphaned punctuation, and any sentence left too short
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([.,;:%])", r"\1", out)
    keep = [x.strip() for x in re.split(r"(?<=[.;!?])\s+", out)
            if len(x.strip().split()) >= 4]
    out = " ".join(keep).strip()
    return (out or None), bad


VERBATIM_MAX_WORDS = int(os.environ.get("VERBATIM_MAX_WORDS", "60"))


def _is_furniture(sent):
    """A sentence that is letterhead, an address, a date line or a routing note rather than
    content. Such a sentence is almost entirely capitalised words and proper nouns; running
    prose is mostly lower case.

    Needed because the verbatim fallback quotes a section's OPENING, and in a letter the opening
    is the letterhead. The Conflict Waiver Letter's published summary was "WC WILSON CRANE
    SILICON VALLEY ANN ARBOR AUSTIN BEIJING BOSTON LOS ANGELES NEW YORK SAN DIEGO SAN FRANCISCO
    SAO PAULO SINGAPORE September 14, 2022 Via Electronic Mail A. Morgan Vendor Inc, Inc." --
    every word of it true to the source and none of it about the letter.
    """
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", sent)
    if not words:
        return True
    lower = sum(1 for w in words if w[0].islower())
    if lower >= 5:
        return False
    return True


# Titles whose period is not a sentence end. Without these "Dear Mr. Morgan:" split in two,
# the half that survived was "Morgan: We are writing ..." and the summary opened on a surname.
# Personal titles and Latin abbreviations only. "Inc." and "Ltd." were in this list and it
# backfired: a letter's header block carries almost no sentence punctuation, so "... A. Morgan
# Vendor Inc, Inc." was the ONLY place the letterhead could be split off, and protecting the period
# glued the letterhead to the body -- which then read as prose and was published whole.
TITLE = re.compile(r"\b(Mr|Mrs|Ms|Dr|Messrs|Prof|Hon|Rev|Attn|e\.g|i\.e|etc|vs|approx)\.", re.I)


def _sentences(text):
    """Split into sentences without breaking on a title's period."""
    held = TITLE.sub(lambda m: m.group(0)[:-1] + "\x01", text)
    parts = re.split(r"(?<=[.;!?])\s+", held.strip())
    return [p.replace("\x01", ".").strip() for p in parts if p.strip()]


# Routing furniture that opens a letter's first real sentence: an address line, a subject line,
# a salutation. Trimmed from the front of the quotation only, never from the middle.
LEAD = re.compile(r"^(?:\s*(?:\S+@\S+\.\S+"
                  r"|Re\s*:.{0,90}?(?=\s(?:Dear|As|We|This|The|Pursuant|In|Please)\b)"
                  r"|Dear[^:]{0,40}:"
                  r"|Via\s+(?:Electronic\s+Mail|Email|Facsimile|Courier)"
                  r"|BY\s+(?:EMAIL|HAND|COURIER)"
                  r"|CONFIDENTIAL(?:\s+AND\s+PRIVILEGED)?)\s*)+", re.I)


def verbatim(section, n=2):
    """The section's own sentences, copied exactly, starting at the first one that says something.

    The last resort, and the only one that cannot be wrong about the source: it IS the source.
    Needed because the polarity and support checks can reject both model drafts at once -- in NDA
    3.1 the extractive summary inverts the liability exactly as the rewrite does, so "fall back to
    the other BART output" is not a safety net there.
    """
    sents = _sentences(section)
    # if the whole section is furniture there is nothing better to quote than the section itself
    body = [x for x in sents if not _is_furniture(x)]
    out = []
    for x in (body or sents):
        out.append(x)
        if len(" ".join(out).split()) >= 28 or len(out) >= n:
            break
    got = " ".join(out).strip()
    # HARD cap. The sentence budget above is checked after a sentence is appended, so a section
    # with no sentence punctuation at all is one enormous "sentence" and sails through it: a bill's
    # page-header band -- "Handset Corp Inc HANDSETCO 9 Account: ... Billing period: ... Talk
    # activity (cont.) ..." repeated eleven times -- was published whole as a 335-word summary.
    # A quotation that long is not a summary of anything.
    words = got.split()
    if len(words) > VERBATIM_MAX_WORDS:
        got = " ".join(words[:VERBATIM_MAX_WORDS]).rstrip(",;:") + " ..."
    trimmed = LEAD.sub("", got).strip()
    # only accept the trim if something substantial survives it
    return trimmed if len(trimmed.split()) >= 8 else got


def _decide(faithful, abstractive, section):
    """Pick the summary to publish for this section, and say why.

    Two independent gates, ordered by what they cost to get wrong:

      polarity   whether the summary reverses a duty the section imposes -- a flipped "not".
                 Checked FIRST, and REPAIRED rather than rejected: dropping the sentence sent
                 the section back to the extractive summary, which is the output we were asked
                 to stop publishing. The source's own polarity determines the edit, so the
                 rewrite is kept and only the reversed word changes.
      support    whether every figure, date and name in the summary is in the section.

    The extractive summary is a fallback only if it is itself clean. It is BART output too, and
    in NDA 3.1 it inverts the liability exactly as the rewrite does -- so "use the other draft"
    is not a safety net. When neither draft can be made safe the section is quoted verbatim,
    which cannot be wrong about the source because it IS the source.

    Returns (summary, record). The record is kept in the output so the abstraction rate and
    every repair are measured numbers rather than claims.
    """
    faithful = tidy(faithful, section)
    abstractive = tidy(abstractive, section)
    if not abstractive:
        fixed, prec = polarity.depolarise(faithful or "", section)
        if not fixed:
            return verbatim(section), dict({"chosen": "verbatim",
                                            "reason": "only draft flips polarity"}, **prec)
        return fixed, dict({"chosen": "faithful-depolarised" if prec else "faithful",
                            "reason": "polarity corrected against the section" if prec
                                      else "no abstractive draft"}, **prec)

    abs_fixed, prec = polarity.depolarise(abstractive, section)
    bad = unsupported(abs_fixed, section) if abs_fixed else unsupported(abstractive, section)

    if abs_fixed and not bad:
        return abs_fixed, dict({"chosen": "abstractive-depolarised" if prec else "abstractive",
                                "reason": "polarity corrected, rewrite kept" if prec
                                          else "clean as generated"}, **prec)
    if abs_fixed:
        fixed, why = repair(abs_fixed, section)
        if fixed:
            fixed, prec2 = polarity.depolarise(fixed, section)
            prec.update(prec2)
        if fixed:
            return fixed, dict({"chosen": "abstractive-repaired",
                                "reason": "unsupported figure/date removed, rewrite kept",
                                "removed": [b["token"] for b in why][:6]}, **prec)

    fb, fprec = polarity.depolarise(faithful or "", section)
    if fb and not unsupported(fb, section):
        return fb, dict({"chosen": "faithful-depolarised" if fprec else "faithful",
                         "reason": "rewrite unusable; extract clean",
                         "rejected": bad[:6], "rejected_draft": abstractive[:300]},
                        **dict(prec, **fprec))
    return verbatim(section), dict({"chosen": "verbatim",
                                    "reason": "both BART drafts unsafe -- quoting the section",
                                    "rejected": bad[:6],
                                    "rejected_draft": abstractive[:300]}, **prec)


def choose(faithful, abstractive, section):
    """Decide which summary to publish, then tidy whatever won.

    The tidy has to happen HERE rather than only on the inbound drafts, because a repair can
    create an artifact that was not in either draft: deleting the unsupported company number from
    "... in England and Wales. 01420028. The registered office ..." leaves "Wales. ." which the
    whitespace cleanup closes up into "Wales..". Tidying only the inputs let that reach the
    output. Every published path -- rewrite, repaired, depolarised, extract and verbatim -- goes
    through one exit, so none of them can skip it.
    """
    text, rec = _decide(faithful, abstractive, section)
    return tidy(text, section), rec
