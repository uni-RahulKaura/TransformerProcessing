"""Catch summaries that flip a clause's polarity.

The figure/date/name guard in safe_abstractive cannot see this class of error, because a
flipped clause reuses only words the section already contains. The live example, Broadcaster Ltd
NDA section 3.1:

  source   "Save as set out below, the Recipient SHALL BE responsible for any breach of this
            Agreement by any of its Authorised Third Parties."
  summary  "The Recipient IS NOT responsible for any breach of the Agreement ..."

Every token in "is not responsible" occurs in the section -- "not" arrives from the preceding
sentence, "Recipient shall NOT disclose ..." -- so the summary passes every lexical check while
reversing who carries the liability. BART-large-CNN does this on its own; it is not a side
effect of asking for a rewrite, and the extractive setting produces the same inversion.

The test is deliberately narrow, because a polarity checker that fires on paraphrase would cost
more than it saves. For each negation in the candidate we take the word the negation governs,
then ask whether the SOURCE ever negates that same word. Three outcomes:

  source negates it somewhere        -> supported, say nothing
  source never mentions it          -> not our problem (the name/figure guard owns that)
  source mentions it, always plain  -> INVENTED NEGATION, the 3.1 case

and the mirror image, an affirmation the source only ever states negatively ("shall not be
liable" summarised as "shall be liable"), which is the same defect pointing the other way.

Ambiguity is resolved in the summary's favour on purpose: a word the source both negates and
affirms is left alone, because the checker cannot tell which occurrence was summarised.
"""
import re

CUES = {"not", "n't", "no", "never", "nor", "neither", "cannot", "without", "none",
        "nothing", "nowhere", "prohibited", "forbidden", "precluded", "barred"}

# Words that sit between a negation and the thing it negates and carry no polarity of their
# own. Auxiliaries and determiners, plus the adverbs that habitually pad legal prose.
SKIP = set("""a an the be been being is are was were am has have had do does did shall will
would may might must can could should to of in on at for by with any all such its their his
her our your this that these those it they he she we you i as so very only also just
otherwise more less than then further hereby thereby herein therein under over both either
each every some other another same own least most much many well ever yet still even
almost quite rather somewhat entirely wholly fully solely merely simply directly indirectly
reasonably materially expressly specifically event circumstances way manner respect""".split())

WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
# The verbs a legal obligation hangs off. The dropped-negation test looks at what one of these
# governs and nothing else.
MODALS = frozenset("shall will may must can could should would is are was were has have had "
                   "does do did".split())
# Boilerplate where "without" is not a negation at all. "including, without limitation, X"
# ENLARGES X; read as a cue it marked "injunctive relief" as something the section only ever
# denies, and the summary was accused of asserting it.
NEUTRAL = re.compile(r"(?i)\bwithout\s+(?:limitation|limiting|prejudice)\b")
# "No." before a number means "number". Read as a negation it made "Agreement No. 30066282
# between Vendor Inc, Inc. and AT&T Services" look like a sentence denying something about
# "between", and the summary was dropped over it.
NUMBER_NO = re.compile(r"(?i)\bNos?\.\s*(?=[\d(])")
# Constructions that turn a negation into a positive requirement: "may not be amended EXCEPT in
# writing signed by ..." means it must be signed. A word appearing only inside one of these is
# not a word the source denies.
EXCEPTION = re.compile(r"(?i)\b(?:except|unless|save as|save for|save that|other than|"
                       r"otherwise than|until|provided that|subject to)\b")


def _neutralise(t):
    return NEUTRAL.sub("including", NUMBER_NO.sub("number ", t or ""))
SENT = re.compile(r"(?<=[.!?;])\s+(?=[A-Z\"'(])")
# How far past a cue we look for the word it governs. Four content tokens covers
# "shall not, in any event, be liable".
SPAN = 6


def _stem(w):
    """Crude, deliberately shallow: enough to tie responsible/responsibility and
    disclose/disclosed/disclosure together without a stemmer's false merges."""
    w = w.lower().rstrip("'")
    for a, b in (("ibilities", "ible"), ("ibility", "ible"), ("abilities", "able"),
                 ("ability", "able")):
        if w.endswith(a) and len(w) > len(a) + 1:
            return w[:-len(a)] + b
    for suf in ("ements", "ement", "ances", "ance", "ences", "ence", "ions", "ion",
                "ings", "ing", "ies", "ed", "es", "s", "e"):
        if w.endswith(suf) and len(w) - len(suf) >= 5:
            return w[:-len(suf)]
    return w


def _governed(toks, i):
    """Index of the word the cue at `i` negates: the first token after it that means
    something. Both sides of the comparison use this same rule, so "shall not disclose" in the
    source and "shall not disclose" in the summary resolve to the same word."""
    for j in range(i + 1, min(i + 1 + SPAN, len(toks))):
        low = toks[j].lower()
        if low in SKIP or low in CUES or len(toks[j]) < 3 or low.endswith("ly"):
            continue
        return j
    return None


def _negated(toks):
    """Indices of the words negated in this token list, by the strict governed-word rule."""
    out = set()
    for i, t in enumerate(toks):
        if t.lower() in CUES:
            j = _governed(toks, i)
            if j is not None:
                out.add(j)
    return out


# How far back a cue may sit and still license a word, on the SOURCE side only. Deliberately
# generous: every token, not every content token, and across the whole sentence.
LICENCE = 10


def _index(source):
    """For every stem in the source, whether the source ever negates it.

    Permissive by design, and asymmetrically so. The cost of the two errors is nowhere near
    equal: missing a negation the source really has means flagging a correct sentence, and the
    strict governed-word rule missed three in one document --

      "shall not (except as permitted by law) reverse engineer"   cue attaches to "except"
      "Nothing in this Agreement shall be construed to oblige"    cue is 8 tokens away
      "shall use ... and not in connection with"                  cue governs "connection"

    -- each of which produced a false accusation against a sentence that was fine. So on this
    side a cue anywhere within LICENCE tokens earlier in the same sentence counts, which errs
    towards saying "the source licenses this" and lets correct sentences through.
    """
    seen = {}
    for sent in SENT.split(source or ""):
        # An exception clause states a requirement through a negation, so its words must not be
        # recorded as things the source denies -- otherwise a faithful positive paraphrase of
        # the requirement looks like a dropped negation.
        exc = bool(EXCEPTION.search(sent))
        toks = WORD.findall(_neutralise(sent))
        cues = [i for i, t in enumerate(toks) if t.lower() in CUES]
        for i, t in enumerate(toks):
            rec = seen.setdefault(_stem(t), {"neg": False, "plain": False})
            if any(0 < i - c <= LICENCE for c in cues):
                rec["neg"] = True
                if exc:
                    rec["plain"] = True
            else:
                rec["plain"] = True
    return seen


# Function words carry no evidence either way. Kept separate from SKIP, which exists to walk
# past auxiliaries to a negation's target and is far broader than this.
FUNCTION = set("""the a an and or of to in on at for by with from that this these those it its
their his her our your they we you as so than then be is are was were been being am has have
had do does did shall will would may might must can could should not no any all such other
another same own more less least most much many""".split())


def _evidence(sent):
    """Content stems in a sentence, for asking what a negation might have attached to."""
    return {_stem(w) for w in WORD.findall(sent)
            if w[0].islower() and len(w) >= 4 and w.lower() not in FUNCTION}


def _wide(source):
    """Stems the source negates ANYWHERE after a cue in the same sentence.

    Used only to ask "did the source negate something in this territory", never to accuse. The
    LICENCE window is right for deciding what a cue governs but too tight for this question: in
    section 3.3 the cue is 13 tokens from "inconsistent", so the window said the source never
    negated it and the re-attachment allowance never fired.
    """
    out = set()
    for sent in SENT.split(source or ""):
        toks = WORD.findall(_neutralise(sent))
        first = next((i for i, t in enumerate(toks) if t.lower() in CUES), None)
        if first is None:
            continue
        for t in toks[first + 1:]:
            out.add(_stem(t))
    return out


def flips(candidate, source):
    """Polarity claims in `candidate` the `source` does not license, at most one per sentence.

    A sentence is tested in one direction only: if it negates something we ask whether the
    source ever negates that thing, otherwise we ask whether the source only ever negates it.
    Running both tests on one sentence produced false positives.
    """
    idx = _index(source)
    wide = _wide(source)
    out = []
    for sent in SENT.split((candidate or "").strip()):
        toks = WORD.findall(_neutralise(sent))
        neg = _negated(toks)
        if neg:
            for j in sorted(neg):
                w = toks[j]
                # Polarity is not a property a name or a defined term can carry. "The author is
                # not a NDA employee" resolves its negation to "NDA"; whether that sentence is
                # supported is a question for the name check, not this one.
                if w[0].isupper() and j > 0:
                    continue
                rec = idx.get(_stem(w))
                if not (rec and rec["plain"] and not rec["neg"]):
                    continue
                # The negation may be real but re-attached. Section 3.3 reads "shall use ...
                # solely for the purposes of the Project and NOT in connection with any other
                # transaction, or in any WAY that is inconsistent"; the draft renders it "shall
                # NOT use ... in any way inconsistent". The negation moved from "connection" to
                # "use", which the strict test calls an invention, but the draft still says what
                # the section says. So if any other content word in this sentence is one the
                # source negates, the negation is accounted for and the sentence stands.
                others = _evidence(sent) - {_stem(w)}
                if others & wide:
                    continue
                out.append({"kind": "invented-negation", "sentence": sent, "word": w})
                break
            continue
        if EXCEPTION.search(sent):
            continue                      # the sentence states an exception, not a polarity
        # Test ONLY the word the sentence's own modal governs -- its main assertion. Testing every
        # noun made this direction fire on ordinary prose: "Gap has a Zero Means Zero policy
        # against harassment" was dropped over "policy", and "Supplier will maintain accurate
        # accounting records including payroll" over "payroll", because each noun happened to
        # appear in the section only within ten tokens of some negation. A noun is not what a
        # negation reverses. "shall not be liable" summarised as "shall be liable" is, and that is
        # a modal and its complement.
        j = next((k for k, t in enumerate(toks) if t.lower() in MODALS), None)
        if j is None:
            continue
        g = _governed(toks, j)
        if g is None:
            continue
        w = toks[g]
        if not w[0].islower() or len(w) < 5 or w.lower() in SKIP:
            continue
        rec = idx.get(_stem(w))
        if rec and rec["neg"] and not rec["plain"]:
            out.append({"kind": "dropped-negation", "sentence": sent, "word": w})
    return out


def depolarise(candidate, source):
    """Remove the sentences whose polarity the source does not license, keep the rest.

    DROP, never edit. The first version corrected a flagged sentence in place by deleting the
    stray "not", to avoid falling back to the extractive summary. On real text that was far
    worse than the defect it was chasing: three of five "repairs" in one NDA turned a correct
    prohibition into a mandate --

      "shall not use ... in any way inconsistent"   ->  "shall use ... in any way inconsistent"
      "shall not reverse engineer, decompile"       ->  "shall reverse engineer, decompile"

    -- because a false accusation plus a confident edit compounds into a reversed obligation,
    while a false accusation plus a dropped sentence costs one sentence. The errors are not
    symmetric, so neither is the response.

    Dropping does not forfeit abstractiveness: the drafts are several sentences long, so a
    summary that loses one is still a rewrite, not a quotation. Only when nothing survives does
    the caller fall back.

    Returns (text, record).
    """
    found = flips(candidate or "", source)
    if not found:
        return candidate, {}
    bad = {f["sentence"]: f for f in found}
    kept = [x for x in SENT.split((candidate or "").strip()) if x not in bad]
    return " ".join(kept).strip(), {
        "polarity_dropped": [{"kind": f["kind"], "word": f["word"], "sentence": s[:170]}
                             for s, f in list(bad.items())[:4]]}
