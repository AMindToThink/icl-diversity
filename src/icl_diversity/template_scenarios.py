"""Synthetic template-sentence scenarios: syntactic redundancy vs semantic scatter.

Purpose (template-vs-SentBERT experiment): construct response sets whose
sentences share *no* semantic content (random adjective/noun/verb pairings
from large word lists) but share syntactic structure (a fixed number of
sentence frames). Embedding-similarity diversity metrics (e.g. Tevet &
Berant's SentBERT baseline) see only the semantic scatter and rate every
such set maximally diverse; the ICL diversity metric D = C * a_n should
instead detect the repeated frame structure via in-context learning.

Conditions supported:

- ``generate_template_responses(n_frames=m, ...)``: n responses drawn from
  m structurally distinct sentence frames, filled with random words. m=1
  with ``frame_pool=[0]`` is the canonical
  "The {adj} {noun} {verb} the {adj2} {noun2}." condition.
- ``PARAPHRASE_RESPONSES``: 20 hand-written paraphrases of one fixed
  meaning; the agreement anchor where SentBERT and D should both be low.

All generation is deterministic given a seed. Word lists are curated so the
"a/an" heuristic in ``_an`` is always correct (no silent-h or you-sounding
initial vowels).
"""

from __future__ import annotations

import random
from typing import Callable, NamedTuple

TEMPLATE_PROMPT = "Write a sentence."

# ============================================================================
# Word lists
# ============================================================================

ADJECTIVES: list[str] = [
    # colors
    "crimson",
    "turquoise",
    "amber",
    "violet",
    "scarlet",
    "golden",
    "silver",
    "emerald",
    "ivory",
    "charcoal",
    "bronze",
    "copper",
    "magenta",
    "indigo",
    "beige",
    "maroon",
    "teal",
    "lavender",
    "olive",
    "coral",
    # moods / temperament
    "gloomy",
    "cheerful",
    "anxious",
    "furious",
    "serene",
    "jubilant",
    "melancholy",
    "restless",
    "weary",
    "giddy",
    "sullen",
    "gleeful",
    "timid",
    "bold",
    "bashful",
    "arrogant",
    "humble",
    "envious",
    "grateful",
    "spiteful",
    "tender",
    "ruthless",
    "gentle",
    "fierce",
    "placid",
    "frantic",
    "calm",
    "nervous",
    "proud",
    "wistful",
    # age / condition
    "ancient",
    "modern",
    "rusty",
    "shiny",
    "damp",
    "arid",
    "fragrant",
    "bitter",
    "salty",
    "sour",
    "sweet",
    "smoky",
    "moldy",
    "crisp",
    "soggy",
    "greasy",
    "dusty",
    "sticky",
    "slippery",
    "velvety",
    "silky",
    "coarse",
    "rough",
    "smooth",
    "jagged",
    "brittle",
    "sturdy",
    "fragile",
    "flimsy",
    "solid",
    "hollow",
    "dense",
    "airy",
    # size / shape
    "tiny",
    "enormous",
    "colossal",
    "minuscule",
    "gigantic",
    "petite",
    "towering",
    "stubby",
    "lanky",
    "plump",
    "slender",
    "bulky",
    "compact",
    "vast",
    "narrow",
    "broad",
    "shallow",
    "deep",
    # material
    "wooden",
    "metallic",
    "plastic",
    "ceramic",
    "marble",
    "granite",
    "leathery",
    "woolen",
    "linen",
    "denim",
    # light / appearance
    "luminous",
    "transparent",
    "opaque",
    "radiant",
    "dim",
    "murky",
    "glossy",
    "matte",
    "sparkling",
    "faded",
    "vivid",
    "pale",
    "dark",
    "bright",
    "dazzling",
    # movement / ability
    "crooked",
    "elegant",
    "clumsy",
    "nimble",
    "sluggish",
    "agile",
    "graceful",
    "awkward",
    "spry",
    "lumbering",
    "swift",
    "drowsy",
    "alert",
    "dizzy",
    "feeble",
    "vigorous",
    "frail",
    "mighty",
    "puny",
    "brawny",
    "delicate",
    # wealth / tidiness
    "lavish",
    "thrifty",
    "opulent",
    "shabby",
    "pristine",
    "tattered",
    "immaculate",
    "grimy",
    "spotless",
    "filthy",
    "tidy",
    "cluttered",
    "ornate",
    "plain",
    "gaudy",
    "subtle",
    "flashy",
    "modest",
    # mind / character
    "curious",
    "oblivious",
    "skeptical",
    "gullible",
    "cunning",
    "naive",
    "wise",
    "foolish",
    "clever",
    "dull",
    "brilliant",
    "absurd",
    "sensible",
    "whimsical",
    "solemn",
    "playful",
    "stern",
    "jolly",
    "grumpy",
    "merry",
    # weather-ish
    "dreary",
    "breezy",
    "humid",
    "frosty",
    "scorching",
    "chilly",
    "balmy",
    "blustery",
    "misty",
    "sunny",
    "stormy",
    "tranquil",
    "chaotic",
    "orderly",
    "rowdy",
    "silent",
    "noisy",
    "muffled",
    "shrill",
    "mellow",
    "harsh",
]

NOUNS: list[str] = [
    # professions
    "accountant",
    "plumber",
    "surgeon",
    "architect",
    "janitor",
    "librarian",
    "barber",
    "tailor",
    "blacksmith",
    "astronomer",
    "biologist",
    "cartographer",
    "diplomat",
    "electrician",
    "florist",
    "gardener",
    "historian",
    "inspector",
    "juggler",
    "locksmith",
    "magician",
    "notary",
    "optician",
    "pharmacist",
    "senator",
    "sculptor",
    "translator",
    "umpire",
    "violinist",
    "welder",
    # animals
    "walrus",
    "heron",
    "badger",
    "ferret",
    "otter",
    "pelican",
    "raccoon",
    "salamander",
    "toucan",
    "vulture",
    "wombat",
    "yak",
    "zebra",
    "alpaca",
    "bison",
    "camel",
    "dolphin",
    "egret",
    "flamingo",
    "gazelle",
    "hedgehog",
    "iguana",
    "jackal",
    "kangaroo",
    "lemur",
    "marmot",
    "newt",
    "ocelot",
    "porcupine",
    "quail",
    "rooster",
    "sturgeon",
    "tortoise",
    "weasel",
    # objects / instruments
    "typewriter",
    "chandelier",
    "harmonica",
    "wheelbarrow",
    "thermostat",
    "accordion",
    "stapler",
    "umbrella",
    "bicycle",
    "compass",
    "doorknob",
    "easel",
    "fountain",
    "gramophone",
    "hammock",
    "inkwell",
    "jukebox",
    "kettle",
    "ladder",
    "mirror",
    "notebook",
    "oven",
    "padlock",
    "quilt",
    "radiator",
    "saxophone",
    "telescope",
    "violin",
    "wardrobe",
    "xylophone",
    "yardstick",
    "zeppelin",
    # buildings / places
    "cathedral",
    "lighthouse",
    "warehouse",
    "gazebo",
    "silo",
    "pavilion",
    "foundry",
    "observatory",
    "vineyard",
    "harbor",
    "quarry",
    "monastery",
    "orchard",
    "castle",
    "cottage",
    "bungalow",
    "chapel",
    "citadel",
    "fortress",
    "granary",
    "hangar",
    # foods
    "artichoke",
    "croissant",
    "pretzel",
    "dumpling",
    "eggplant",
    "fig",
    "grapefruit",
    "hazelnut",
    "omelet",
    "pancake",
    "quiche",
    "radish",
    "strudel",
    "tangerine",
    "walnut",
    "yogurt",
    "zucchini",
    "apricot",
    "biscuit",
    "custard",
    "doughnut",
    "falafel",
    "gherkin",
    "waffle",
    "meatball",
    "noodle",
    "oyster",
    "parsnip",
    # household / small objects
    "teacup",
    "bookshelf",
    "carpet",
    "envelope",
    "feather",
    "glove",
    "hatstand",
    "icicle",
    "jigsaw",
    "kite",
    "lantern",
    "mitten",
    "napkin",
    "ottoman",
    "pillow",
    "quill",
    "ribbon",
    "satchel",
    "teapot",
    "vase",
    "whistle",
    "anchor",
    "barrel",
    "cauldron",
    "drum",
    "flask",
    "goblet",
    "helmet",
    "javelin",
    "keg",
    "lute",
    "mallet",
    "oar",
    "plow",
    "rake",
    "shovel",
    "trumpet",
    "wagon",
    # characters
    "grandmother",
    "toddler",
    "pirate",
    "knight",
    "wizard",
    "ghost",
    "robot",
    "clown",
    "ballerina",
    "cowboy",
    "viking",
    "ninja",
    "mermaid",
    "giant",
    "dwarf",
    "elf",
    "troll",
    "ogre",
    "phantom",
    "jester",
]


class VerbTriple(NamedTuple):
    """Transitive verb forms: base (infinitive), simple past, past participle."""

    base: str
    past: str
    part: str


def _reg(base: str, past: str | None = None) -> VerbTriple:
    """Regular verb: past == participle."""
    p = past if past is not None else base + "ed"
    return VerbTriple(base, p, p)


VERBS: list[VerbTriple] = [
    _reg("admire", "admired"),
    _reg("devour", "devoured"),
    _reg("polish", "polished"),
    _reg("chase", "chased"),
    _reg("examine", "examined"),
    _reg("betray", "betrayed"),
    _reg("embrace", "embraced"),
    _reg("demolish", "demolished"),
    _reg("interrogate", "interrogated"),
    _reg("serenade", "serenaded"),
    _reg("bewilder", "bewildered"),
    _reg("applaud", "applauded"),
    _reg("measure", "measured"),
    _reg("decorate", "decorated"),
    _reg("kidnap", "kidnapped"),
    _reg("mock", "mocked"),
    _reg("rescue", "rescued"),
    _reg("haunt", "haunted"),
    _reg("tickle", "tickled"),
    _reg("summon", "summoned"),
    _reg("ambush", "ambushed"),
    _reg("borrow", "borrowed"),
    _reg("capture", "captured"),
    _reg("dismantle", "dismantled"),
    _reg("envy", "envied"),
    _reg("forge", "forged"),
    _reg("grab", "grabbed"),
    _reg("hoist", "hoisted"),
    _reg("ignore", "ignored"),
    _reg("juggle", "juggled"),
    _reg("mimic", "mimicked"),
    _reg("nudge", "nudged"),
    _reg("paint", "painted"),
    _reg("question", "questioned"),
    _reg("ridicule", "ridiculed"),
    _reg("sketch", "sketched"),
    _reg("tow", "towed"),
    _reg("unveil", "unveiled"),
    _reg("vandalize", "vandalized"),
    _reg("weigh", "weighed"),
    _reg("yank", "yanked"),
    _reg("inspect", "inspected"),
    _reg("launch", "launched"),
    _reg("mend", "mended"),
    _reg("notice", "noticed"),
    _reg("pinch", "pinched"),
    _reg("quote", "quoted"),
    _reg("repair", "repaired"),
    _reg("seize", "seized"),
    _reg("taste", "tasted"),
    _reg("trade", "traded"),
    _reg("unwrap", "unwrapped"),
    _reg("visit", "visited"),
    _reg("wash", "washed"),
    _reg("wreck", "wrecked"),
    _reg("salute", "saluted"),
    _reg("pamper", "pampered"),
    _reg("startle", "startled"),
    _reg("soothe", "soothed"),
    _reg("scold", "scolded"),
    _reg("praise", "praised"),
    _reg("insult", "insulted"),
    _reg("flatter", "flattered"),
    _reg("outwit", "outwitted"),
    _reg("pursue", "pursued"),
    _reg("smuggle", "smuggled"),
    _reg("trap", "trapped"),
    _reg("tame", "tamed"),
    _reg("tease", "teased"),
    _reg("torment", "tormented"),
    _reg("treasure", "treasured"),
    _reg("trick", "tricked"),
    _reg("trust", "trusted"),
    _reg("worship", "worshipped"),
    _reg("abandon", "abandoned"),
    _reg("adopt", "adopted"),
    _reg("amuse", "amused"),
    _reg("annoy", "annoyed"),
    _reg("arrest", "arrested"),
    _reg("assemble", "assembled"),
    _reg("astonish", "astonished"),
    _reg("avoid", "avoided"),
    _reg("bake", "baked"),
    _reg("blame", "blamed"),
    _reg("bless", "blessed"),
    _reg("bribe", "bribed"),
    _reg("comfort", "comforted"),
    _reg("confuse", "confused"),
    _reg("conquer", "conquered"),
    _reg("crush", "crushed"),
    _reg("defend", "defended"),
    _reg("deliver", "delivered"),
    _reg("describe", "described"),
    _reg("destroy", "destroyed"),
    _reg("disguise", "disguised"),
    _reg("distract", "distracted"),
    _reg("dodge", "dodged"),
    _reg("escort", "escorted"),
    _reg("fetch", "fetched"),
    _reg("follow", "followed"),
    _reg("guard", "guarded"),
    _reg("harvest", "harvested"),
    _reg("imitate", "imitated"),
    _reg("invite", "invited"),
    _reg("memorize", "memorized"),
    _reg("mislead", "misled"),
    _reg("obey", "obeyed"),
    _reg("photograph", "photographed"),
    _reg("pluck", "plucked"),
    _reg("protect", "protected"),
    _reg("punish", "punished"),
    _reg("recognize", "recognized"),
    _reg("reward", "rewarded"),
    _reg("scrub", "scrubbed"),
    _reg("sharpen", "sharpened"),
    _reg("silence", "silenced"),
    _reg("smash", "smashed"),
    _reg("squeeze", "squeezed"),
    _reg("surprise", "surprised"),
    _reg("swallow", "swallowed"),
    _reg("tackle", "tackled"),
    _reg("taunt", "taunted"),
    _reg("toss", "tossed"),
    _reg("trace", "traced"),
    _reg("transport", "transported"),
    _reg("carry", "carried"),
    _reg("bury", "buried"),
    _reg("copy", "copied"),
    _reg("study", "studied"),
    _reg("drag", "dragged"),
    _reg("drop", "dropped"),
    # irregulars
    VerbTriple("steal", "stole", "stolen"),
    VerbTriple("buy", "bought", "bought"),
    VerbTriple("catch", "caught", "caught"),
    VerbTriple("teach", "taught", "taught"),
    VerbTriple("sell", "sold", "sold"),
    VerbTriple("find", "found", "found"),
    VerbTriple("hide", "hid", "hidden"),
    VerbTriple("shake", "shook", "shaken"),
    VerbTriple("throw", "threw", "thrown"),
    VerbTriple("wear", "wore", "worn"),
    VerbTriple("break", "broke", "broken"),
    VerbTriple("choose", "chose", "chosen"),
    VerbTriple("forgive", "forgave", "forgiven"),
    VerbTriple("freeze", "froze", "frozen"),
    VerbTriple("ride", "rode", "ridden"),
    VerbTriple("bite", "bit", "bitten"),
    VerbTriple("overtake", "overtook", "overtaken"),
    VerbTriple("weave", "wove", "woven"),
    VerbTriple("spin", "spun", "spun"),
]


def _an(word: str) -> str:
    """Indefinite article + word. Word lists are curated so this heuristic holds."""
    article = "an" if word[0] in "aeiou" else "a"
    return f"{article} {word}"


# ============================================================================
# Sentence frames
# ============================================================================
# Each frame is a structurally distinct English construction using all five
# content slots (adj, noun, verb, adj2, noun2). Frame 0 is the canonical
# "The {adj} {noun} {verb} the {adj2} {noun2}." template.

FrameFn = Callable[[str, str, VerbTriple, str, str], str]

FRAMES: list[FrameFn] = [
    # 0: canonical declarative SVO
    lambda a, n, v, a2, n2: f"The {a} {n} {v.past} the {a2} {n2}.",
    # 1: yes/no question with do-support
    lambda a, n, v, a2, n2: f"Did the {a} {n} really {v.base} the {a2} {n2}?",
    # 2: it-cleft
    lambda a, n, v, a2, n2: f"It was the {a} {n} that {v.past} the {a2} {n2}.",
    # 3: passive
    lambda a, n, v, a2, n2: f"The {a2} {n2} was {v.part} by the {a} {n}.",
    # 4: modal prohibition
    lambda a, n, v, a2, n2: f"No {a} {n} should ever {v.base} {_an(a2)} {n2}.",
    # 5: temporal subordinate clause
    lambda a, n, v, a2, n2: (
        f"After the {a} {n} {v.past} the {a2} {n2}, everyone applauded."
    ),
    # 6: conditional
    lambda a, n, v, a2, n2: (
        f"If {_an(a)} {n} ever {v.past} {_an(a2)} {n2}, nobody noticed."
    ),
    # 7: wh-question with modal
    lambda a, n, v, a2, n2: f"Why would the {a} {n} {v.base} the {a2} {n2}?",
    # 8: reported speech with past perfect
    lambda a, n, v, a2, n2: f"They said the {a} {n} had {v.part} the {a2} {n2}.",
    # 9: fronted locative + relative clause
    lambda a, n, v, a2, n2: (f"Near the {a2} {n2} stood the {a} {n} that {v.past} it."),
    # 10: exclamative
    lambda a, n, v, a2, n2: f"What {_an(a)} {n} it takes to {v.base} {_an(a2)} {n2}!",
    # 11: gerund subject
    lambda a, n, v, a2, n2: (
        f"Watching the {a} {n} {v.base} the {a2} {n2} felt strange."
    ),
    # 12: existential
    lambda a, n, v, a2, n2: (
        f"There was once {_an(a)} {n} that {v.past} {_an(a2)} {n2}."
    ),
    # 13: tag question
    lambda a, n, v, a2, n2: f"The {a} {n} {v.past} the {a2} {n2}, didn't it?",
    # 14: because-clause
    lambda a, n, v, a2, n2: (f"Because the {n} was so {a}, it {v.past} the {a2} {n2}."),
    # 15: negative imperative
    lambda a, n, v, a2, n2: f"Never let {_an(a)} {n} {v.base} your {a2} {n2}.",
    # 16: so-that result clause
    lambda a, n, v, a2, n2: (
        f"The {n} was so {a} that it {v.past} the {a2} {n2} twice."
    ),
    # 17: fronted comparative
    lambda a, n, v, a2, n2: f"More {a} than ever, the {n} {v.past} the {a2} {n2}.",
    # 18: while-clause
    lambda a, n, v, a2, n2: (
        f"While nobody watched, the {a} {n} {v.past} the {a2} {n2}."
    ),
    # 19: negative inversion with past perfect
    lambda a, n, v, a2, n2: (
        f"Not once had the {a} {n} {v.part} {_an(a2)} {n2} before."
    ),
]

FRAME_LABELS: list[str] = [
    "declarative_svo",
    "yesno_question",
    "it_cleft",
    "passive",
    "modal_prohibition",
    "temporal_subordinate",
    "conditional",
    "wh_question",
    "reported_speech",
    "fronted_locative",
    "exclamative",
    "gerund_subject",
    "existential",
    "tag_question",
    "because_clause",
    "negative_imperative",
    "so_that_result",
    "fronted_comparative",
    "while_clause",
    "negative_inversion",
]


def generate_template_responses(
    n_frames: int,
    n: int = 20,
    seed: int = 0,
    frame_pool: list[int] | None = None,
) -> tuple[list[str], list[int]]:
    """Generate n template sentences using n_frames distinct syntactic frames.

    Frames are sampled without replacement from ``frame_pool`` (default: all
    of ``FRAMES``), responses are distributed evenly across the chosen frames,
    and the response order is shuffled. Word slots are filled independently
    per response from the large word lists, so responses share no semantic
    content beyond the frame structure.

    Returns:
        (responses, frame_ids): the n sentences and, aligned with them, the
        index into ``FRAMES`` used for each sentence.

    Raises:
        ValueError: if n_frames is out of range for the pool, or if word
            sampling produced duplicate sentences (fail fast; astronomically
            unlikely for the shipped list sizes).
    """
    pool = frame_pool if frame_pool is not None else list(range(len(FRAMES)))
    if not set(pool) <= set(range(len(FRAMES))):
        raise ValueError(f"frame_pool contains invalid frame ids: {pool}")
    if not 1 <= n_frames <= len(pool):
        raise ValueError(
            f"n_frames must be in [1, {len(pool)}] for this pool, got {n_frames}"
        )
    rng = random.Random(seed)
    frame_ids = rng.sample(pool, n_frames)
    assignments = [frame_ids[i % n_frames] for i in range(n)]
    rng.shuffle(assignments)

    responses: list[str] = []
    for fid in assignments:
        adj, adj2 = rng.sample(ADJECTIVES, 2)
        noun, noun2 = rng.sample(NOUNS, 2)
        verb = rng.choice(VERBS)
        sentence = FRAMES[fid](adj, noun, verb, adj2, noun2)
        responses.append(sentence[0].upper() + sentence[1:])

    if len(set(responses)) != len(responses):
        raise ValueError(
            f"Duplicate sentences generated for seed={seed}; use another seed."
        )
    return responses, assignments


# ============================================================================
# Paraphrase anchor: one meaning, 20 wordings
# ============================================================================
# SentBERT similarity is high here (shared meaning), so both the SentBERT
# diversity baseline and D = C * a_n
# should be low: the agreement case that shows SentBERT is not simply broken.

PARAPHRASE_RESPONSES: list[str] = [
    "The committee postponed the meeting until next week.",
    "The meeting was postponed by the committee until next week.",
    "Until next week, the committee has put off the meeting.",
    "The committee decided to delay the meeting until next week.",
    "The committee pushed the meeting back to next week.",
    "As the committee delayed it, the meeting will now happen next week.",
    "The committee moved the meeting to next week.",
    "The meeting has been rescheduled by the committee for next week.",
    "The committee chose to hold the meeting next week instead.",
    "The meeting will now take place next week, following the committee's postponement.",
    "The committee deferred the meeting until next week.",
    "The meeting was pushed to next week by the committee.",
    "The committee put the meeting off until next week.",
    "The committee shifted the meeting to next week.",
    "Owing to the committee's decision, the meeting is delayed until next week.",
    "The committee resolved to postpone the meeting until next week.",
    "The meeting is now set for next week because the committee postponed it.",
    "The committee announced that the meeting would be delayed until next week.",
    "The meeting got moved to next week by the committee.",
    "The committee elected to reschedule the meeting for next week.",
]
