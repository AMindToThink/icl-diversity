"""POS-pattern scenarios: structural redundancy with zero lexical overlap.

Second template experiment (see ``template_scenarios`` for the first).
There, repeated frames shared literal function words ("The", "the",
boilerplate spans), so averaged distinct-n also detected the redundancy.
Here every single word is randomly sampled; sentences share only their
part-of-speech pattern, e.g. the canonical

    "{Noun} {verb} {preposition} {noun} {preposition} {noun}."
    -> "Walruses slept beneath cathedrals beside pretzels."

With no fixed lexical material, word n-grams are (near-)unique across
responses regardless of how many patterns are used, so distinct-n cannot
separate 1 pattern from 12; SentBERT sees only semantic scatter and cannot
either. Only the pattern itself is learnable, and only in-context.

Patterns are template strings whose ONLY word-characters live inside
``{Tag}`` placeholders (enforced by tests): N = plural noun, Vi = past
intransitive verb, Vt = past transitive verb, A = adjective, Adv = adverb,
P = preposition. All patterns are 6 words long. Pattern 0 is the canonical
N Vi P N P N. Word slots are sampled per sentence, distinct within a
sentence per class.
"""

from __future__ import annotations

import random
import re

from icl_diversity.template_scenarios import ADJECTIVES, VERBS

POS_PATTERN_PROMPT = "Write a sentence."

# ============================================================================
# Word lists (explicit forms; no runtime inflection)
# ============================================================================

PLURAL_NOUNS: list[str] = [
    # professions
    "accountants",
    "plumbers",
    "surgeons",
    "architects",
    "janitors",
    "librarians",
    "barbers",
    "tailors",
    "blacksmiths",
    "astronomers",
    "biologists",
    "cartographers",
    "diplomats",
    "electricians",
    "florists",
    "gardeners",
    "historians",
    "inspectors",
    "jugglers",
    "locksmiths",
    "magicians",
    "notaries",
    "opticians",
    "pharmacists",
    "senators",
    "sculptors",
    "translators",
    "umpires",
    "violinists",
    "welders",
    # animals
    "walruses",
    "herons",
    "badgers",
    "ferrets",
    "otters",
    "pelicans",
    "raccoons",
    "salamanders",
    "toucans",
    "vultures",
    "wombats",
    "yaks",
    "zebras",
    "alpacas",
    "camels",
    "dolphins",
    "egrets",
    "flamingos",
    "gazelles",
    "hedgehogs",
    "iguanas",
    "jackals",
    "kangaroos",
    "lemurs",
    "marmots",
    "newts",
    "ocelots",
    "porcupines",
    "quails",
    "roosters",
    "tortoises",
    "weasels",
    # objects / instruments
    "typewriters",
    "chandeliers",
    "harmonicas",
    "wheelbarrows",
    "thermostats",
    "accordions",
    "staplers",
    "umbrellas",
    "bicycles",
    "compasses",
    "doorknobs",
    "easels",
    "fountains",
    "gramophones",
    "hammocks",
    "inkwells",
    "jukeboxes",
    "kettles",
    "ladders",
    "mirrors",
    "notebooks",
    "ovens",
    "padlocks",
    "quilts",
    "radiators",
    "saxophones",
    "telescopes",
    "violins",
    "wardrobes",
    "xylophones",
    "yardsticks",
    "zeppelins",
    # buildings / places
    "cathedrals",
    "lighthouses",
    "warehouses",
    "gazebos",
    "silos",
    "pavilions",
    "foundries",
    "observatories",
    "vineyards",
    "harbors",
    "quarries",
    "monasteries",
    "orchards",
    "castles",
    "cottages",
    "bungalows",
    "chapels",
    "citadels",
    "fortresses",
    "granaries",
    "hangars",
    # foods
    "artichokes",
    "croissants",
    "pretzels",
    "dumplings",
    "eggplants",
    "figs",
    "grapefruits",
    "hazelnuts",
    "omelets",
    "pancakes",
    "quiches",
    "radishes",
    "strudels",
    "tangerines",
    "walnuts",
    "zucchinis",
    "apricots",
    "biscuits",
    "doughnuts",
    "gherkins",
    "waffles",
    "meatballs",
    "noodles",
    "oysters",
    "parsnips",
    # household / small objects
    "teacups",
    "bookshelves",
    "carpets",
    "envelopes",
    "feathers",
    "gloves",
    "hatstands",
    "icicles",
    "jigsaws",
    "kites",
    "lanterns",
    "mittens",
    "napkins",
    "ottomans",
    "pillows",
    "quills",
    "ribbons",
    "satchels",
    "teapots",
    "vases",
    "whistles",
    "anchors",
    "barrels",
    "cauldrons",
    "drums",
    "flasks",
    "goblets",
    "helmets",
    "javelins",
    "kegs",
    "lutes",
    "mallets",
    "oars",
    "plows",
    "rakes",
    "shovels",
    "trumpets",
    "wagons",
    # characters
    "grandmothers",
    "toddlers",
    "pirates",
    "knights",
    "wizards",
    "ghosts",
    "robots",
    "clowns",
    "ballerinas",
    "cowboys",
    "vikings",
    "ninjas",
    "mermaids",
    "giants",
    "elves",
    "trolls",
    "ogres",
    "phantoms",
    "jesters",
]

INTRANSITIVE_PAST: list[str] = [
    "slept",
    "wandered",
    "trembled",
    "hesitated",
    "stumbled",
    "whispered",
    "marched",
    "lingered",
    "danced",
    "shouted",
    "yawned",
    "sneezed",
    "wept",
    "chuckled",
    "giggled",
    "groaned",
    "sighed",
    "paused",
    "waited",
    "dozed",
    "snored",
    "fidgeted",
    "paced",
    "strolled",
    "sprinted",
    "jogged",
    "crawled",
    "leaped",
    "hopped",
    "skipped",
    "tumbled",
    "drifted",
    "floated",
    "soared",
    "glided",
    "hovered",
    "plunged",
    "swerved",
    "wobbled",
    "teetered",
    "swayed",
    "twirled",
    "bowed",
    "knelt",
    "crouched",
    "squatted",
    "sat",
    "rose",
    "fell",
    "vanished",
    "appeared",
    "emerged",
    "arrived",
    "departed",
    "retreated",
    "advanced",
    "hurried",
    "dawdled",
    "loitered",
    "meandered",
    "roamed",
    "prowled",
    "lurked",
    "cowered",
    "shivered",
    "shuddered",
    "blushed",
    "frowned",
    "smiled",
    "grinned",
    "smirked",
    "scowled",
    "glared",
    "gazed",
    "stared",
    "blinked",
    "squinted",
    "nodded",
    "shrugged",
    "coughed",
    "grumbled",
    "muttered",
    "murmured",
    "hummed",
    "chanted",
    "prayed",
    "meditated",
    "slouched",
    "leaned",
    "rested",
    "relaxed",
    "daydreamed",
    "brooded",
    "sulked",
    "rejoiced",
    "celebrated",
    "gloated",
    "panicked",
    "fainted",
    "collapsed",
    "recovered",
    "persevered",
    "persisted",
    "surrendered",
    "rebelled",
    "protested",
]

PREPOSITIONS: list[str] = [
    "above",
    "across",
    "after",
    "against",
    "along",
    "alongside",
    "amid",
    "among",
    "around",
    "atop",
    "before",
    "behind",
    "below",
    "beneath",
    "beside",
    "between",
    "beyond",
    "despite",
    "during",
    "inside",
    "into",
    "near",
    "onto",
    "opposite",
    "outside",
    "over",
    "past",
    "through",
    "throughout",
    "toward",
    "under",
    "underneath",
    "upon",
    "within",
    "without",
]

ADVERBS: list[str] = [
    "gracefully",
    "quietly",
    "furiously",
    "gently",
    "boldly",
    "calmly",
    "nervously",
    "proudly",
    "wearily",
    "gleefully",
    "timidly",
    "humbly",
    "eagerly",
    "reluctantly",
    "silently",
    "noisily",
    "swiftly",
    "slowly",
    "clumsily",
    "nimbly",
    "sluggishly",
    "awkwardly",
    "elegantly",
    "frantically",
    "serenely",
    "jubilantly",
    "sullenly",
    "cheerfully",
    "gloomily",
    "anxiously",
    "fiercely",
    "tenderly",
    "ruthlessly",
    "cautiously",
    "recklessly",
    "deliberately",
    "hastily",
    "patiently",
    "impatiently",
    "solemnly",
    "playfully",
    "sternly",
    "merrily",
    "grumpily",
    "wistfully",
    "brazenly",
    "sheepishly",
    "smugly",
    "defiantly",
    "obediently",
    "mysteriously",
    "predictably",
    "suddenly",
    "gradually",
    "endlessly",
    "briefly",
    "repeatedly",
    "occasionally",
    "constantly",
    "rarely",
]

TRANSITIVE_PAST: list[str] = [v.past for v in VERBS]

# ============================================================================
# POS patterns
# ============================================================================
# Template strings; word characters may appear ONLY inside {Tag} placeholders
# (tests enforce this: outside the braces there are only spaces, commas, and
# the final period). Tag = class name + slot index; same-class slots within a
# sentence get distinct words.

POS_PATTERNS: list[str] = [
    # 0: canonical N Vi P N P N
    "{N1} {Vi1} {P1} {N2} {P2} {N3}.",
    # 1: A N Vi P A N
    "{A1} {N1} {Vi1} {P1} {A2} {N2}.",
    # 2: fronted P N, N Vt A N
    "{P1} {N1}, {N2} {Vt1} {A1} {N3}.",
    # 3: N Vt N Adv P N
    "{N1} {Vt1} {N2} {Adv1} {P1} {N3}.",
    # 4: fronted Adv, A N Vi P N
    "{Adv1}, {A1} {N1} {Vi1} {P1} {N2}.",
    # 5: N P N Vi P N
    "{N1} {P1} {N2} {Vi1} {P2} {N3}.",
    # 6: A N Adv Vt A N
    "{A1} {N1} {Adv1} {Vt1} {A2} {N2}.",
    # 7: fronted P A N, N Vi Adv
    "{P1} {A1} {N1}, {N2} {Vi1} {Adv1}.",
    # 8: N Vi Adv P A N
    "{N1} {Vi1} {Adv1} {P1} {A1} {N2}.",
    # 9: A A N Vt N Adv
    "{A1} {A2} {N1} {Vt1} {N2} {Adv1}.",
    # 10: fronted Adv, N Vt N P N
    "{Adv1}, {N1} {Vt1} {N2} {P1} {N3}.",
    # 11: N P A N Vi Adv
    "{N1} {P1} {A1} {N2} {Vi1} {Adv1}.",
]

POS_PATTERN_LABELS: list[str] = [
    "n_vi_p_n_p_n",
    "a_n_vi_p_a_n",
    "p_n_n_vt_a_n",
    "n_vt_n_adv_p_n",
    "adv_a_n_vi_p_n",
    "n_p_n_vi_p_n",
    "a_n_adv_vt_a_n",
    "p_a_n_n_vi_adv",
    "n_vi_adv_p_a_n",
    "a_a_n_vt_n_adv",
    "adv_n_vt_n_p_n",
    "n_p_a_n_vi_adv",
]

_SLOT_RE = re.compile(r"\{([A-Za-z]+?)(\d+)\}")

_CLASS_LISTS: dict[str, list[str]] = {
    "N": PLURAL_NOUNS,
    "Vi": INTRANSITIVE_PAST,
    "Vt": TRANSITIVE_PAST,
    "A": ADJECTIVES,
    "Adv": ADVERBS,
    "P": PREPOSITIONS,
}


def render_pattern(pattern: str, rng: random.Random) -> str:
    """Fill a pattern's slots with random words, distinct per class."""
    slots = _SLOT_RE.findall(pattern)
    if not slots:
        raise ValueError(f"pattern has no slots: {pattern!r}")
    by_class: dict[str, list[str]] = {}
    for cls, _idx in slots:
        if cls not in _CLASS_LISTS:
            raise ValueError(f"unknown slot class {cls!r} in pattern {pattern!r}")
        by_class.setdefault(cls, []).append(cls)
    fills: dict[str, str] = {}
    for cls, occurrences in by_class.items():
        words = rng.sample(_CLASS_LISTS[cls], len(occurrences))
        indices = [idx for c, idx in slots if c == cls]
        for idx, word in zip(indices, words):
            fills[f"{cls}{idx}"] = word
    sentence = _SLOT_RE.sub(lambda m: fills[m.group(1) + m.group(2)], pattern)
    return sentence[0].upper() + sentence[1:]


def generate_pos_pattern_responses(
    n_patterns: int,
    n: int = 40,
    seed: int = 0,
    pattern_pool: list[int] | None = None,
) -> tuple[list[str], list[int]]:
    """Generate n sentences using n_patterns distinct POS patterns.

    Mirrors ``template_scenarios.generate_template_responses``: patterns are
    sampled without replacement from ``pattern_pool`` (default: all of
    ``POS_PATTERNS``), responses are distributed evenly across the chosen
    patterns (counts differ by at most 1 when n_patterns does not divide n),
    and the response order is shuffled. Every word in every sentence is
    freshly sampled.

    Returns:
        (responses, pattern_ids): sentences and, aligned, the index into
        ``POS_PATTERNS`` used for each.
    """
    pool = pattern_pool if pattern_pool is not None else list(range(len(POS_PATTERNS)))
    if not set(pool) <= set(range(len(POS_PATTERNS))):
        raise ValueError(f"pattern_pool contains invalid pattern ids: {pool}")
    if not 1 <= n_patterns <= len(pool):
        raise ValueError(
            f"n_patterns must be in [1, {len(pool)}] for this pool, got {n_patterns}"
        )
    rng = random.Random(seed)
    pattern_ids = rng.sample(pool, n_patterns)
    assignments = [pattern_ids[i % n_patterns] for i in range(n)]
    rng.shuffle(assignments)

    responses: list[str] = []
    for pid in assignments:
        responses.append(render_pattern(POS_PATTERNS[pid], rng))

    if len(set(responses)) != len(responses):
        raise ValueError(
            f"Duplicate sentences generated for seed={seed}; use another seed."
        )
    return responses, assignments


# The class multiset of the canonical pattern (POS_PATTERNS[0]): three plural
# nouns, one intransitive past verb, two prepositions.
CANONICAL_CLASS_MULTISET: tuple[tuple[str, int], ...] = (("N", 3), ("Vi", 1), ("P", 2))


def generate_scrambled_canonical_responses(n: int = 40, seed: int = 0) -> list[str]:
    """No-structure control matched to the canonical pattern.

    Each sentence samples exactly the canonical class multiset
    (``CANONICAL_CLASS_MULTISET``) and then shuffles its six words into a
    random per-sentence order. The set therefore has the same vocabulary
    distribution, word counts, and byte statistics as the canonical
    condition, but no consistent word-order pattern to learn: word-count
    based metrics (distinct-n) are identical in distribution by
    construction, and mean-pooled embedding metrics (SentBERT) are nearly
    order-blind, while the ICL a_k curve should drop less than for the
    structured canonical set.
    """
    rng = random.Random(seed)
    responses: list[str] = []
    for _ in range(n):
        words: list[str] = []
        for cls, count in CANONICAL_CLASS_MULTISET:
            words.extend(rng.sample(_CLASS_LISTS[cls], count))
        rng.shuffle(words)
        sentence = " ".join(words) + "."
        responses.append(sentence[0].upper() + sentence[1:])
    if len(set(responses)) != len(responses):
        raise ValueError(
            f"Duplicate sentences generated for seed={seed}; use another seed."
        )
    return responses
