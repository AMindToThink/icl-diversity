"""Synthetic scenario data for ICL diversity metric validation.

Five edge-case scenarios from the paper (Section 6.3), each with 5 prompts
and 10 responses. Used by both the test suite and the experiment scripts.

Scenarios:
1. Pure noise:               C ≈ 0, E ≈ 0, D ≈ 0
2. Multiple incoherent modes: C low, E > 0, D suppressed
3. Many coherent modes:      C high, E high, D high
4. One coherent mode:        C high, E moderate, D moderate
5. Mixed coherent+incoherent: high σ_ℓ, wide [D-, D+] band
6. High diversity (15 modes): Very high D, slow a_k convergence
7. Open creative (15 format/genre modes): Format-diverse responses to same topic
8. Problem-solving (15 approach modes): Different languages/algorithms/styles
"""

import random
import string
from typing import Callable

# Scenarios 7 & 8 are defined in _new_scenarios.py (pre-defined choice tuples
# to work around Python 3.10 f-string backslash restriction).
from icl_diversity._new_scenarios import (  # noqa: E402, F401
    OPEN_CREATIVE_N_RESPONSES,
    OPEN_CREATIVE_PROMPT_LABELS,
    OPEN_CREATIVE_PROMPTS_AND_RESPONSES,
    PROBLEM_SOLVING_N_RESPONSES,
    PROBLEM_SOLVING_PROMPT_LABELS,
    PROBLEM_SOLVING_PROMPTS_AND_RESPONSES,
)

N_RESPONSES = 10
N_PERMUTATIONS = 3

# ============================================================================
# Scenario 1: Pure noise
# ============================================================================
# Random ASCII. No learnable structure, very high per-byte cross-entropy.

NOISE_PROMPTS = [
    "Write a short paragraph about technology.",
    "Describe your favorite season in detail.",
    "Explain why exercise is important for health.",
    "Tell me about a memorable travel experience.",
    "Discuss the importance of reading books.",
]


def _random_ascii_noise(length: int = 80, rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    chars = string.ascii_letters + string.digits + string.punctuation + " "
    return "".join(rng.choice(chars) for _ in range(length))


def generate_noise_responses(n: int = 10, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [_random_ascii_noise(rng=rng, length=rng.randint(60, 100)) for _ in range(n)]


# ============================================================================
# Scenario 2: Multiple incoherent modes
# ============================================================================
# Distinct recognizable garbage patterns. θ can learn which pattern but
# each is individually implausible.

_INCOHERENT_MODES: list[Callable[[random.Random], str]] = [
    # Mode A: repeated uppercase letter blocks
    lambda rng: " ".join(c * 4 for c in rng.sample("ABCDEFGHIJKLMNOP", 12)),
    # Mode B: number sequences
    lambda rng: " ".join(str(rng.randint(1000, 9999)) for _ in range(12)),
    # Mode C: punctuation patterns
    lambda rng: " ".join(
        rng.choice(["!@#$", "%^&*", "()_+", "<>?:", "{|}~"]) for _ in range(15)
    ),
    # Mode D: keyboard row repetitions
    lambda rng: " ".join(
        rng.choice(["qwer", "asdf", "zxcv", "tyui", "ghjk"]) * 3 for _ in range(5)
    ),
]

INCOHERENT_PROMPTS = NOISE_PROMPTS  # reuse same prompts


def generate_multi_incoherent_responses(n: int = 10, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    responses = []
    for i in range(n):
        mode_fn = _INCOHERENT_MODES[i % len(_INCOHERENT_MODES)]
        responses.append(mode_fn(rng))
    return responses


# ============================================================================
# Scenario 3: Many coherent modes (template-based)
# ============================================================================
# Each prompt has responses drawn from 3-4 recognizable templates with
# slot variation. GPT-2 can detect these surface patterns via ICL.
# Responses are interleaved across modes to avoid ordering artifacts.

MULTI_MODE_PROMPTS_AND_RESPONSES: list[tuple[str, list[str]]] = [
    (
        "Tell me about an animal.",
        [
            "The cat sat on the mat. It was a warm day and the cat was happy to rest there.",
            "The dog ran in the park. It chased after a ball and barked loudly with excitement.",
            "The bird flew over the lake. It circled twice before landing on a tall branch.",
            "The cat sat on the mat. It was a sunny day and the cat was content and purring.",
            "The dog ran in the park. It chased after a stick and barked happily at its owner.",
            "The bird flew over the lake. It circled twice before landing on a mossy rock.",
            "The cat sat on the mat. It was a nice day and the cat was peaceful and sleepy.",
            "The dog ran in the park. It chased after a frisbee and barked excitedly at everyone.",
            "The bird flew over the lake. It circled twice before landing on a wooden fence.",
            "The cat sat on the mat. It was a lovely day and the cat was relaxed and calm.",
        ],
    ),
    (
        "Describe what happened this morning.",
        [
            "I woke up early and made a cup of coffee. The kitchen was quiet and the sun was just rising over the hills.",
            "I went for a jog along the river at dawn. The air was cool and fresh and the path was empty.",
            "I took the train to work and read my book. The carriage was nearly empty and the ride was smooth.",
            "I woke up early and made a cup of tea. The kitchen was peaceful and the sun was streaming through the window.",
            "I went for a run along the canal at dawn. The air was crisp and clear and the trail was deserted.",
            "I took the bus to work and listened to music. The seats were mostly empty and the journey was pleasant.",
            "I woke up early and made a cup of cocoa. The kitchen was still and the sun was peeking over the rooftops.",
            "I went for a walk along the beach at dawn. The air was salty and warm and the sand was untouched.",
            "I took the subway to work and checked my email. The car was half empty and the ride was uneventful.",
            "I woke up early and made a cup of juice. The kitchen was calm and the sun was lighting up the garden.",
        ],
    ),
    (
        "Write about a place you visited.",
        [
            "The museum was filled with ancient artifacts from civilizations long forgotten. I spent hours studying the intricate carvings.",
            "The beach stretched for miles along the coast with white sand and turquoise water. I watched the waves crash against the shore.",
            "The mountain trail wound through dense pine forests and rocky outcrops. I reached the summit just as the sun began to set.",
            "The museum was filled with beautiful paintings from artists across the centuries. I spent hours admiring the bold use of color.",
            "The beach stretched for miles along the coast with golden sand and clear water. I watched the seabirds dive for fish.",
            "The mountain trail wound through misty valleys and alpine meadows. I reached the summit just as the clouds began to clear.",
            "The museum was filled with rare manuscripts from the medieval period. I spent hours reading the faded handwritten pages.",
            "The beach stretched for miles along the coast with rocky pools and gentle waves. I watched the crabs scuttle between the stones.",
            "The mountain trail wound through wildflower fields and snowcapped ridges. I reached the summit just as the first stars appeared.",
            "The museum was filled with stunning sculptures from the Renaissance era. I spent hours examining the lifelike marble figures.",
        ],
    ),
    (
        "Tell me about your favorite food.",
        [
            "I love eating pasta with homemade tomato sauce and fresh basil. The aroma fills the entire kitchen when it cooks.",
            "I love eating sushi with fresh salmon and creamy avocado. The flavors blend perfectly with a touch of soy sauce.",
            "I love eating curry with tender chicken and fragrant spices. The rich sauce pairs wonderfully with steamed rice.",
            "I love eating pasta with garlic butter and grated parmesan. The simple flavors come together beautifully on the plate.",
            "I love eating sushi with tuna belly and pickled ginger. The texture is incredibly delicate and melts in your mouth.",
            "I love eating curry with roasted vegetables and coconut milk. The creamy sauce goes perfectly with warm naan bread.",
            "I love eating pasta with pesto and toasted pine nuts. The bright green color makes every dish look inviting.",
            "I love eating sushi with shrimp tempura and sweet sauce. The crispy coating adds a wonderful crunch to each bite.",
            "I love eating curry with lamb and caramelized onions. The deep flavors develop slowly over hours of gentle cooking.",
            "I love eating pasta with mushroom cream sauce and thyme. The earthy taste is perfect for a cold evening.",
        ],
    ),
    (
        "Describe your ideal weekend.",
        [
            "I like to spend my weekends reading novels by the fireplace. A good book and a cup of tea make the perfect afternoon.",
            "I like to spend my weekends hiking through the national park. The trails are quiet and the scenery is absolutely stunning.",
            "I like to spend my weekends cooking elaborate meals from scratch. Trying new recipes is my favorite way to relax.",
            "I like to spend my weekends reading poetry in the garden. A comfortable chair and some shade make the perfect setting.",
            "I like to spend my weekends hiking along the coastal cliffs. The ocean views are breathtaking and the air is refreshing.",
            "I like to spend my weekends cooking traditional dishes from different countries. Each cuisine tells a story through its flavors.",
            "I like to spend my weekends reading biographies at the library. Learning about remarkable lives is endlessly fascinating to me.",
            "I like to spend my weekends hiking up to mountain lakes. The crystal clear water reflects the sky like a mirror.",
            "I like to spend my weekends cooking pastries and fresh bread. The smell of baking fills the house with warmth.",
            "I like to spend my weekends reading science fiction on the porch. Imagining other worlds is my favorite form of escape.",
        ],
    ),
]


# ============================================================================
# Scenario 4: One coherent mode (paraphrases)
# ============================================================================
# All responses convey the same content with minor wording variation.
# GPT-2 should quickly learn the pattern → a_k drops fast, low E.

ONE_MODE_PROMPTS_AND_RESPONSES: list[tuple[str, list[str]]] = [
    (
        "Tell me about an animal.",
        [
            "The cat sat on the mat. It was a warm day and the cat was happy to rest there.",
            "The cat sat on the mat. It was a sunny day and the cat was content and purring.",
            "The cat sat on the mat. It was a nice day and the cat was peaceful and sleepy.",
            "The cat sat on the mat. It was a lovely day and the cat was relaxed and calm.",
            "The cat sat on the mat. It was a great day and the cat was comfortable and quiet.",
            "The cat sat on the mat. It was a bright day and the cat was pleased and still.",
            "The cat sat on the mat. It was a fine day and the cat was glad and resting.",
            "The cat sat on the mat. It was a perfect day and the cat was joyful and at ease.",
            "The cat sat on the mat. It was a beautiful day and the cat was serene and dozing.",
            "The cat sat on the mat. It was a pleasant day and the cat was cozy and settled.",
        ],
    ),
    (
        "Describe what happened this morning.",
        [
            "I woke up early and made a cup of coffee. The kitchen was quiet and the sun was just rising.",
            "I woke up early and made a cup of coffee. The kitchen was still and the sun was coming up.",
            "I woke up early and made a cup of coffee. The kitchen was peaceful and the sun was appearing.",
            "I woke up early and made a cup of coffee. The kitchen was calm and the sun was breaking through.",
            "I woke up early and made a cup of coffee. The kitchen was silent and the sun was starting to shine.",
            "I woke up early and made a cup of coffee. The kitchen was empty and the sun was beginning to glow.",
            "I woke up early and made a cup of coffee. The kitchen was warm and the sun was peeking in.",
            "I woke up early and made a cup of coffee. The kitchen was cozy and the sun was filtering through.",
            "I woke up early and made a cup of coffee. The kitchen was dark and the sun was slowly rising.",
            "I woke up early and made a cup of coffee. The kitchen was neat and the sun was climbing higher.",
        ],
    ),
    (
        "Write about a place you visited.",
        [
            "The museum was filled with ancient artifacts from civilizations long forgotten. I spent hours studying the displays.",
            "The museum was filled with ancient artifacts from civilizations long gone. I spent hours examining the exhibits.",
            "The museum was filled with ancient artifacts from civilizations of the past. I spent hours looking at the collections.",
            "The museum was filled with ancient artifacts from civilizations now lost. I spent hours browsing the galleries.",
            "The museum was filled with ancient artifacts from forgotten civilizations. I spent hours exploring the rooms.",
            "The museum was filled with ancient artifacts from vanished civilizations. I spent hours wandering the halls.",
            "The museum was filled with ancient artifacts from early civilizations. I spent hours inspecting the pieces.",
            "The museum was filled with ancient artifacts from distant civilizations. I spent hours viewing the treasures.",
            "The museum was filled with ancient artifacts from old civilizations. I spent hours surveying the items.",
            "The museum was filled with ancient artifacts from bygone civilizations. I spent hours observing the relics.",
        ],
    ),
    (
        "Tell me about your favorite food.",
        [
            "I love eating pasta with homemade tomato sauce and fresh basil. The aroma fills the entire kitchen.",
            "I love eating pasta with homemade tomato sauce and fresh herbs. The smell fills the entire kitchen.",
            "I love eating pasta with homemade tomato sauce and fresh garlic. The fragrance fills the entire kitchen.",
            "I love eating pasta with homemade tomato sauce and fresh oregano. The scent fills the entire kitchen.",
            "I love eating pasta with homemade tomato sauce and fresh parsley. The aroma fills the whole kitchen.",
            "I love eating pasta with homemade tomato sauce and fresh thyme. The smell fills the whole kitchen.",
            "I love eating pasta with homemade tomato sauce and fresh rosemary. The fragrance fills the whole kitchen.",
            "I love eating pasta with homemade tomato sauce and fresh chives. The scent fills the whole kitchen.",
            "I love eating pasta with homemade tomato sauce and fresh sage. The aroma fills up the kitchen.",
            "I love eating pasta with homemade tomato sauce and fresh mint. The smell fills up the kitchen.",
        ],
    ),
    (
        "Describe your ideal weekend.",
        [
            "I like to spend my weekends reading novels by the fireplace. A good book and a cup of tea are perfect.",
            "I like to spend my weekends reading novels by the window. A good book and a cup of tea are wonderful.",
            "I like to spend my weekends reading novels on the couch. A good book and a cup of tea are delightful.",
            "I like to spend my weekends reading novels in the garden. A good book and a cup of tea are lovely.",
            "I like to spend my weekends reading novels on the porch. A good book and a cup of tea are ideal.",
            "I like to spend my weekends reading novels in the study. A good book and a cup of tea are relaxing.",
            "I like to spend my weekends reading novels at the cafe. A good book and a cup of tea are enjoyable.",
            "I like to spend my weekends reading novels in the park. A good book and a cup of tea are pleasant.",
            "I like to spend my weekends reading novels by the lake. A good book and a cup of tea are calming.",
            "I like to spend my weekends reading novels under a tree. A good book and a cup of tea are soothing.",
        ],
    ),
]


# ============================================================================
# Scenario 5: Mixed coherent + incoherent
# ============================================================================
# Half coherent English, half gibberish. Should produce high σ_ℓ.

MIXED_PROMPTS_AND_RESPONSES: list[tuple[str, list[str]]] = [
    (
        "Write a sentence about the ocean.",
        [
            "The ocean stretches endlessly toward the horizon, its waves crashing against the rocky shore.",
            "Beneath the surface, colorful coral reefs teem with marine life of every description.",
            "xkq7 m!nz @#$ plv brt wq92 fnx %%% zzz aaa bbb ccc ddd eee fff ggg",
            "Sailors have navigated the ocean for thousands of years, guided by the stars above.",
            "kkkk jjjj mmmm nnnn pppp qqqq rrrr ssss tttt uuuu vvvv wwww xxxx yyyy",
            "The deep ocean remains one of the least explored places on our planet Earth.",
            "!!!??? &&& $$$ %%% ^^^ *** ((( ))) ___ +++ === ~~~ !!! ???",
            "Ocean currents regulate climate by distributing heat from the equator to the poles.",
            "asdfjkl; qwertyuiop zxcvbnm asdfjkl; qwertyuiop zxcvbnm asdfjkl; qwerty",
            "The Pacific Ocean is the largest and deepest of all the world's ocean basins.",
        ],
    ),
    (
        "Describe a forest.",
        [
            "The ancient forest was thick with moss-covered oaks, their branches forming a living cathedral.",
            "9f8g7h6j5k4l3 2a1s0d f9g8h7j6k5l4 3a2s1d0f 9f8g7h6j5k4l3 2a1s0d",
            "Sunlight filtered through the canopy in golden shafts, illuminating the ferns below.",
            "Birds sang from every branch as squirrels darted between the towering pine trees.",
            "QQQQ WWWW EEEE RRRR TTTT YYYY UUUU IIII OOOO PPPP AAAA SSSS",
            "The forest floor was carpeted with fallen leaves, releasing the rich scent of earth.",
            "#@!$% ^&*() 12345 67890 abcde fghij klmno pqrst uvwxy zabcd",
            "A narrow trail wound through the birch trees, disappearing into the misty distance.",
            "The forest hummed with life, insects buzzing and streams gurgling through the trees.",
            "zzzz xxxx cccc vvvv bbbb nnnn mmmm aaaa ssss dddd ffff gggg hhhh",
        ],
    ),
    (
        "Tell me about mountains.",
        [
            "Mountains rise majestically above the landscape, their peaks dusted with snow even in summer.",
            "The Himalayas contain the tallest mountains on Earth, including Mount Everest.",
            "!@#$%^& *()_+= {}|: <>? ~` -= [] \\;' ,./ !@#$%^ &*()_+= {}|:",
            "Hikers are drawn to mountains for the challenge and the breathtaking summit views.",
            "7777 8888 9999 0000 1111 2222 3333 4444 5555 6666 7777 8888",
            "Mountain ecosystems support unique biodiversity, from alpine meadows to dense forests.",
            "AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH IIII JJJJ KKKK LLLL",
            "The Rocky Mountains stretch over three thousand miles across North America.",
            "qwer asdf zxcv tyui ghjk bnm poiu lkjh mnbv qwer asdf zxcv",
            "Volcanic mountains like Mount Fuji hold deep cultural significance worldwide.",
        ],
    ),
    (
        "Write about the weather.",
        [
            "Today brought a gentle breeze and scattered clouds across an otherwise clear blue sky.",
            "pppp oooo iiii uuuu yyyy tttt rrrr eeee wwww qqqq aaaa ssss",
            "Thunderstorms rolled across the plains, lightning illuminating the landscape in flashes.",
            "The morning fog lifted slowly, revealing a crisp autumn day in the mid-fifties.",
            "@@@@ #### $$$$ %%%% &&&& **** ++++ ==== ~~~~ !!!! ???? ;;;;",
            "Weather patterns are becoming increasingly unpredictable due to climate change.",
            "A cold front moving in from the north promises snow by the weekend ahead.",
            "1a2b3c4d5e6f7g8h9i0j 1a2b3c4d5e6f7g8h9i0j 1a2b3c4d5e6f7g8h9i0j",
            "The sunset painted the western sky in shades of orange and pink as heat broke.",
            "mxnz bvcy alkq wpoe irut ythg fjdk slao zmxn bvcy alkq wpoe",
        ],
    ),
    (
        "Describe a city at night.",
        [
            "The city glowed with neon signs and streetlights, shadows stretching across wet pavement.",
            "xxxx yyyy zzzz aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii",
            "Taxis honked and sirens wailed as the nightlife crowd spilled out of restaurants.",
            "From the rooftop, the city looked like a circuit board pulsing with electric light.",
            "!!!??? $$$ %%% ^^^ &&& *** ((( ))) ___ +++ === ~~~ ###",
            "Street vendors hawked wares under flickering lamps as the subway rumbled below.",
            "The old cathedral stood dark and silent amid the noise and neon of the blocks.",
            "9876543210 abcdefghij 9876543210 abcdefghij 9876543210 abcdefghij",
            "A jazz club on the corner filled the air with saxophone until the early hours.",
            "qazwsxedcrfvtgbyhnujmikolp qazwsxedcrfvtgbyhnujmikolp qazwsxedc",
        ],
    ),
]

# ============================================================================
# Scenario 6: High diversity — 15 coherent modes, 100 responses
# ============================================================================
# Each prompt has 100 responses drawn from 15 distinct templates.
# Responses cycle through modes with slot variation. With many modes,
# a_k should converge slowly and D should be very high.

# Mode templates: each takes a random.Random and returns a response string.
# Templates are designed to be surface-distinguishable by a base LM.

_HIGH_DIV_ANIMAL_MODES: list[Callable[[random.Random], str]] = [
    # Mode 1: Cat behavior
    lambda rng: f"The cat {rng.choice(['sat', 'lay', 'curled up', 'stretched'])} on the {rng.choice(['mat', 'sofa', 'windowsill', 'rug'])}. It was a {rng.choice(['warm', 'sunny', 'quiet', 'lazy'])} afternoon and the cat purred softly.",
    # Mode 2: Dog at the park
    lambda rng: f"The dog {rng.choice(['ran', 'sprinted', 'trotted', 'dashed'])} through the {rng.choice(['park', 'field', 'meadow', 'yard'])}. It {rng.choice(['chased a ball', 'fetched a stick', 'jumped over logs', 'splashed in puddles'])} with boundless energy.",
    # Mode 3: Bird in flight
    lambda rng: f"A {rng.choice(['blue', 'red', 'golden', 'white'])} bird soared above the {rng.choice(['forest', 'lake', 'valley', 'hills'])}. Its wings caught the {rng.choice(['morning', 'evening', 'afternoon', 'midday'])} light beautifully.",
    # Mode 4: Fish underwater
    lambda rng: f"The {rng.choice(['silver', 'spotted', 'striped', 'colorful'])} fish swam through the {rng.choice(['coral reef', 'kelp forest', 'clear stream', 'deep ocean'])}. It darted between {rng.choice(['rocks', 'plants', 'anemones', 'shadows'])} gracefully.",
    # Mode 5: Horse running
    lambda rng: f"The {rng.choice(['brown', 'black', 'white', 'gray'])} horse galloped across the {rng.choice(['open plain', 'rolling hills', 'sandy beach', 'grassy field'])}. Its mane {rng.choice(['flowed', 'streamed', 'whipped', 'rippled'])} in the wind.",
    # Mode 6: Owl at night
    lambda rng: f"An owl {rng.choice(['perched', 'sat', 'waited', 'watched'])} silently in the {rng.choice(['oak tree', 'pine tree', 'old barn', 'church tower'])}. Its {rng.choice(['amber', 'golden', 'bright', 'piercing'])} eyes scanned the darkness below.",
    # Mode 7: Rabbit in garden
    lambda rng: f"A small rabbit {rng.choice(['hopped', 'nibbled', 'crouched', 'scurried'])} in the {rng.choice(['garden', 'vegetable patch', 'backyard', 'meadow'])}. It {rng.choice(['twitched its nose', 'perked its ears', 'froze in place', 'munched on clover'])} cautiously.",
    # Mode 8: Dolphin playing
    lambda rng: f"A pod of dolphins {rng.choice(['leaped', 'jumped', 'arced', 'surfaced'])} through the {rng.choice(['waves', 'calm water', 'ocean spray', 'surf'])}. They {rng.choice(['clicked', 'whistled', 'chattered', 'called'])} to each other playfully.",
    # Mode 9: Eagle hunting
    lambda rng: f"The eagle {rng.choice(['circled', 'soared', 'glided', 'hovered'])} high above the {rng.choice(['canyon', 'river', 'mountain ridge', 'desert floor'])}. With {rng.choice(['incredible', 'remarkable', 'stunning', 'breathtaking'])} precision it dove toward its prey.",
    # Mode 10: Turtle on shore
    lambda rng: f"The {rng.choice(['sea', 'old', 'giant', 'ancient'])} turtle {rng.choice(['crawled', 'lumbered', 'crept', 'trudged'])} along the {rng.choice(['sandy beach', 'rocky shore', 'warm coastline', 'moonlit beach'])}. It moved with {rng.choice(['patient', 'quiet', 'steady', 'calm'])} determination.",
    # Mode 11: Bear foraging
    lambda rng: f"A {rng.choice(['large', 'massive', 'young', 'grizzly'])} bear {rng.choice(['foraged', 'searched', 'rummaged', 'rooted'])} along the {rng.choice(['riverbank', 'forest edge', 'berry bushes', 'fallen logs'])}. It {rng.choice(['sniffed the air', 'turned over stones', 'pawed at the ground', 'shook its heavy head'])} hungrily.",
    # Mode 12: Butterfly in meadow
    lambda rng: f"A {rng.choice(['monarch', 'painted', 'swallowtail', 'blue'])} butterfly {rng.choice(['fluttered', 'drifted', 'floated', 'danced'])} among the {rng.choice(['wildflowers', 'daisies', 'sunflowers', 'lavender'])}. Its {rng.choice(['delicate', 'vibrant', 'patterned', 'iridescent'])} wings caught the sunlight.",
    # Mode 13: Wolf howling
    lambda rng: f"The {rng.choice(['lone', 'gray', 'timber', 'arctic'])} wolf {rng.choice(['howled', 'called', 'cried', 'sang'])} from the {rng.choice(['hilltop', 'ridge', 'snowy peak', 'forest clearing'])}. The {rng.choice(['moonlight', 'starlight', 'twilight', 'darkness'])} cast long shadows across the land.",
    # Mode 14: Penguin colony
    lambda rng: f"A colony of penguins {rng.choice(['huddled', 'gathered', 'waddled', 'shuffled'])} on the {rng.choice(['icy shore', 'frozen coast', 'snow-covered beach', 'glacier edge'])}. They {rng.choice(['called', 'squawked', 'trumpeted', 'chirped'])} loudly in the {rng.choice(['bitter', 'freezing', 'harsh', 'biting'])} cold.",
    # Mode 15: Elephant herd
    lambda rng: f"The elephant herd {rng.choice(['marched', 'traveled', 'migrated', 'walked'])} across the {rng.choice(['savanna', 'dusty plain', 'dry riverbed', 'golden grassland'])}. The {rng.choice(['matriarch', 'eldest female', 'lead cow', 'mother'])} guided them toward {rng.choice(['water', 'the watering hole', 'shade', 'fresh grazing'])}.",
]

_HIGH_DIV_HOBBY_MODES: list[Callable[[random.Random], str]] = [
    # Mode 1: Painting
    lambda rng: f"I spent the afternoon painting {rng.choice(['a landscape', 'a portrait', 'a still life', 'an abstract piece'])} with {rng.choice(['watercolors', 'oils', 'acrylics', 'pastels'])}. The {rng.choice(['colors', 'textures', 'brushstrokes', 'layers'])} came together {rng.choice(['beautifully', 'gradually', 'unexpectedly', 'harmoniously'])}.",
    # Mode 2: Playing guitar
    lambda rng: f"I picked up my {rng.choice(['acoustic', 'electric', 'classical', 'bass'])} guitar and played {rng.choice(['a blues riff', 'some folk songs', 'a jazz progression', 'a rock anthem'])}. The {rng.choice(['melody', 'rhythm', 'chords', 'notes'])} filled the {rng.choice(['room', 'house', 'garage', 'studio'])}.",
    # Mode 3: Gardening
    lambda rng: f"I worked in the garden, {rng.choice(['planting', 'pruning', 'weeding', 'watering'])} the {rng.choice(['roses', 'tomatoes', 'herbs', 'vegetables'])}. The {rng.choice(['soil', 'earth', 'dirt', 'ground'])} was {rng.choice(['rich', 'dark', 'moist', 'warm'])} and smelled of {rng.choice(['spring', 'rain', 'compost', 'growth'])}.",
    # Mode 4: Chess
    lambda rng: f"I played a {rng.choice(['long', 'quick', 'tense', 'brilliant'])} game of chess against {rng.choice(['my friend', 'the computer', 'my neighbor', 'a stranger online'])}. The {rng.choice(['opening', 'middlegame', 'endgame', 'final position'])} was {rng.choice(['complex', 'elegant', 'surprising', 'decisive'])}.",
    # Mode 5: Photography
    lambda rng: f"I took my camera to the {rng.choice(['park', 'waterfront', 'old town', 'countryside'])} and photographed {rng.choice(['the sunset', 'street scenes', 'wildflowers', 'architecture'])}. The {rng.choice(['lighting', 'composition', 'shadows', 'contrast'])} was {rng.choice(['perfect', 'dramatic', 'soft', 'stunning'])}.",
    # Mode 6: Woodworking
    lambda rng: f"I spent hours in the workshop {rng.choice(['carving', 'sanding', 'joining', 'shaping'])} a {rng.choice(['bookshelf', 'cutting board', 'picture frame', 'small box'])} from {rng.choice(['oak', 'walnut', 'maple', 'cherry'])}. The {rng.choice(['grain', 'wood', 'finish', 'surface'])} was {rng.choice(['smooth', 'beautiful', 'perfect', 'satisfying'])}.",
    # Mode 7: Knitting
    lambda rng: f"I sat by the window knitting a {rng.choice(['scarf', 'sweater', 'hat', 'blanket'])} from {rng.choice(['soft wool', 'merino yarn', 'cotton thread', 'alpaca fiber'])}. The {rng.choice(['pattern', 'stitches', 'rows', 'design'])} {rng.choice(['grew steadily', 'took shape', 'progressed nicely', 'came along well'])}.",
    # Mode 8: Astronomy
    lambda rng: f"I set up my telescope and observed {rng.choice(['Jupiter', 'Saturn', 'the Orion Nebula', 'the Andromeda Galaxy'])} on a {rng.choice(['clear', 'cloudless', 'moonless', 'crisp'])} night. The {rng.choice(['stars', 'sky', 'cosmos', 'heavens'])} were {rng.choice(['breathtaking', 'spectacular', 'magnificent', 'awe-inspiring'])}.",
    # Mode 9: Baking bread
    lambda rng: f"I baked a loaf of {rng.choice(['sourdough', 'whole wheat', 'rye', 'ciabatta'])} bread from scratch. The {rng.choice(['crust', 'crumb', 'aroma', 'texture'])} was {rng.choice(['golden', 'perfect', 'crispy', 'wonderful'])} and the house smelled {rng.choice(['amazing', 'incredible', 'heavenly', 'fantastic'])}.",
    # Mode 10: Bird watching
    lambda rng: f"I went birdwatching at the {rng.choice(['nature reserve', 'wetlands', 'forest trail', 'lakeside'])} and spotted a {rng.choice(['kingfisher', 'heron', 'warbler', 'woodpecker'])}. It {rng.choice(['perched quietly', 'sang loudly', 'preened its feathers', 'dove into the water'])} as I watched through binoculars.",
    # Mode 11: Writing poetry
    lambda rng: f"I sat at my desk and wrote a {rng.choice(['sonnet', 'haiku', 'free verse poem', 'limerick'])} about {rng.choice(['autumn leaves', 'the passing of time', 'a childhood memory', 'the night sky'])}. The {rng.choice(['words', 'lines', 'stanzas', 'verses'])} {rng.choice(['flowed easily', 'came slowly', 'surprised me', 'felt right'])}.",
    # Mode 12: Rock climbing
    lambda rng: f"I went rock climbing at the {rng.choice(['indoor gym', 'local crag', 'granite cliff', 'sandstone wall'])} and tackled a {rng.choice(['challenging', 'technical', 'overhanging', 'vertical'])} route. The {rng.choice(['holds', 'moves', 'crux', 'finish'])} required {rng.choice(['precision', 'strength', 'balance', 'focus'])}.",
    # Mode 13: Pottery
    lambda rng: f"I threw a {rng.choice(['bowl', 'vase', 'mug', 'plate'])} on the pottery wheel using {rng.choice(['stoneware', 'porcelain', 'earthenware', 'terracotta'])} clay. The {rng.choice(['shape', 'form', 'piece', 'vessel'])} {rng.choice(['emerged slowly', 'took form', 'grew under my hands', 'came together'])} as the wheel spun.",
    # Mode 14: Cycling
    lambda rng: f"I went for a {rng.choice(['long', 'brisk', 'leisurely', 'challenging'])} bike ride along the {rng.choice(['canal path', 'coastal road', 'mountain trail', 'country lane'])}. The {rng.choice(['wind', 'breeze', 'air', 'weather'])} was {rng.choice(['refreshing', 'cool', 'warm', 'perfect'])} and the views were {rng.choice(['stunning', 'lovely', 'spectacular', 'gorgeous'])}.",
    # Mode 15: Stargazing
    lambda rng: f"I lay on a blanket in the {rng.choice(['backyard', 'open field', 'hilltop', 'desert'])} and gazed at the {rng.choice(['Milky Way', 'constellations', 'shooting stars', 'northern lights'])}. The {rng.choice(['silence', 'stillness', 'darkness', 'vastness'])} was {rng.choice(['profound', 'humbling', 'peaceful', 'overwhelming'])}.",
]

_HIGH_DIV_TRAVEL_MODES: list[Callable[[random.Random], str]] = [
    # Mode 1: Beach vacation
    lambda rng: f"We arrived at the {rng.choice(['tropical', 'secluded', 'pristine', 'popular'])} beach and {rng.choice(['swam in the turquoise water', 'built sandcastles', 'collected seashells', 'lounged under umbrellas'])}. The {rng.choice(['sand', 'water', 'waves', 'sunset'])} was {rng.choice(['perfect', 'gorgeous', 'incredible', 'unforgettable'])}.",
    # Mode 2: Mountain hiking
    lambda rng: f"We hiked up the {rng.choice(['steep', 'winding', 'rugged', 'scenic'])} mountain trail through {rng.choice(['pine forests', 'alpine meadows', 'rocky terrain', 'wildflower fields'])}. At the summit we {rng.choice(['took photos', 'ate lunch', 'rested quietly', 'cheered'])} and enjoyed the panoramic view.",
    # Mode 3: European city
    lambda rng: f"We wandered the {rng.choice(['cobblestone', 'narrow', 'bustling', 'charming'])} streets of {rng.choice(['Paris', 'Rome', 'Prague', 'Barcelona'])} and visited {rng.choice(['a cathedral', 'an art museum', 'a palace', 'a historic square'])}. The {rng.choice(['architecture', 'atmosphere', 'history', 'culture'])} was {rng.choice(['magnificent', 'captivating', 'enchanting', 'remarkable'])}.",
    # Mode 4: Safari
    lambda rng: f"On safari we spotted a {rng.choice(['pride of lions', 'herd of elephants', 'family of giraffes', 'pack of zebras'])} crossing the {rng.choice(['savanna', 'grassland', 'dusty road', 'dry riverbed'])}. Our guide {rng.choice(['explained their behavior', 'kept a safe distance', 'took amazing photos', 'whispered in excitement'])}.",
    # Mode 5: Train journey
    lambda rng: f"The train {rng.choice(['wound', 'rolled', 'sped', 'chugged'])} through the {rng.choice(['Swiss Alps', 'Scottish Highlands', 'Japanese countryside', 'Indian plains'])}. We watched the {rng.choice(['landscape', 'scenery', 'villages', 'rivers'])} pass by from the {rng.choice(['observation car', 'dining car', 'window seat', 'open platform'])}.",
    # Mode 6: Island exploration
    lambda rng: f"We explored the {rng.choice(['volcanic', 'tiny', 'lush', 'remote'])} island by {rng.choice(['renting scooters', 'hiking trails', 'taking a boat tour', 'walking barefoot'])}. The {rng.choice(['jungle', 'coastline', 'waterfalls', 'villages'])} {rng.choice(['amazed us', 'took our breath away', 'felt untouched', 'were magical'])}.",
    # Mode 7: Desert adventure
    lambda rng: f"We drove across the {rng.choice(['vast', 'endless', 'scorching', 'golden'])} desert in a {rng.choice(['jeep', 'truck', 'four-wheel drive', 'caravan'])}. The {rng.choice(['sand dunes', 'rock formations', 'starlit sky', 'silence'])} made us feel {rng.choice(['tiny', 'humbled', 'alive', 'free'])}.",
    # Mode 8: River cruise
    lambda rng: f"We took a {rng.choice(['sunset', 'morning', 'leisurely', 'guided'])} cruise down the {rng.choice(['Danube', 'Rhine', 'Mekong', 'Nile'])}. The {rng.choice(['riverbanks', 'castles', 'temples', 'villages'])} {rng.choice(['drifted past', 'unfolded before us', 'reflected in the water', 'told stories of the past'])}.",
    # Mode 9: Winter skiing
    lambda rng: f"We hit the {rng.choice(['powder', 'groomed', 'steep', 'wide open'])} slopes at {rng.choice(['dawn', 'midday', 'sunset', 'first light'])} and skied until our legs {rng.choice(['ached', 'burned', 'gave out', 'trembled'])}. The {rng.choice(['snow', 'mountain air', 'views', 'fresh powder'])} was absolutely {rng.choice(['perfect', 'exhilarating', 'magical', 'glorious'])}.",
    # Mode 10: Ancient ruins
    lambda rng: f"We visited the ancient {rng.choice(['Roman', 'Greek', 'Mayan', 'Egyptian'])} ruins and walked among {rng.choice(['crumbling columns', 'stone temples', 'weathered statues', 'mosaic floors'])}. The {rng.choice(['history', 'scale', 'craftsmanship', 'age'])} of the site was {rng.choice(['humbling', 'astonishing', 'moving', 'impressive'])}.",
    # Mode 11: Food tour
    lambda rng: f"We joined a {rng.choice(['street food', 'market', 'culinary', 'walking'])} tour and tasted {rng.choice(['local cheeses', 'fresh seafood', 'handmade pasta', 'exotic spices'])}. Every {rng.choice(['bite', 'dish', 'flavor', 'course'])} told a story about the {rng.choice(['region', 'culture', 'traditions', 'people'])}.",
    # Mode 12: Rainforest trek
    lambda rng: f"We trekked through the {rng.choice(['dense', 'misty', 'humid', 'ancient'])} rainforest, {rng.choice(['crossing rope bridges', 'following animal tracks', 'wading through streams', 'climbing muddy slopes'])}. The {rng.choice(['canopy', 'birdcalls', 'insects', 'biodiversity'])} was {rng.choice(['extraordinary', 'overwhelming', 'incredible', 'mesmerizing'])}.",
    # Mode 13: Northern lights
    lambda rng: f"We drove {rng.choice(['hours', 'far from the city', 'into the wilderness', 'north'])} to see the northern lights. The {rng.choice(['green', 'purple', 'shimmering', 'dancing'])} aurora {rng.choice(['filled the sky', 'rippled overhead', 'swirled across the horizon', 'pulsed with light'])} and left us speechless.",
    # Mode 14: Coastal village
    lambda rng: f"We spent a {rng.choice(['lazy', 'quiet', 'sunny', 'windy'])} day in a {rng.choice(['tiny', 'colorful', 'fishing', 'cliffside'])} village along the coast. The {rng.choice(['harbor', 'houses', 'boats', 'market'])} {rng.choice(['gleamed in the sun', 'swayed gently', 'smelled of salt air', 'buzzed with locals'])}.",
    # Mode 15: Hot air balloon
    lambda rng: f"We floated above the {rng.choice(['valley', 'vineyards', 'fairy chimneys', 'patchwork fields'])} in a {rng.choice(['colorful', 'red and gold', 'striped', 'giant'])} hot air balloon. The {rng.choice(['silence', 'view', 'sunrise', 'landscape below'])} was {rng.choice(['surreal', 'otherworldly', 'majestic', 'serene'])}.",
]


def _generate_high_diversity_responses(
    modes: list[Callable[[random.Random], str]],
    n: int = 100,
    seed: int = 0,
) -> list[str]:
    """Generate n responses cycling through all modes with slot variation."""
    rng = random.Random(seed)
    responses: list[str] = []
    # Interleave modes to avoid ordering artifacts
    indices = list(range(n))
    rng.shuffle(indices)
    mode_assignments = [i % len(modes) for i in indices]
    # Sort back by original index so responses are in shuffled-mode order
    paired = sorted(zip(indices, mode_assignments))
    for _, mode_idx in paired:
        responses.append(modes[mode_idx](rng))
    return responses


HIGH_DIVERSITY_PROMPTS_AND_RESPONSES: list[tuple[str, list[str]]] = [
    (
        "Tell me about an animal.",
        _generate_high_diversity_responses(_HIGH_DIV_ANIMAL_MODES, n=100, seed=42),
    ),
    (
        "Describe your favorite hobby.",
        _generate_high_diversity_responses(_HIGH_DIV_HOBBY_MODES, n=100, seed=43),
    ),
    (
        "Tell me about a trip you took.",
        _generate_high_diversity_responses(_HIGH_DIV_TRAVEL_MODES, n=100, seed=44),
    ),
]

HIGH_DIVERSITY_PROMPT_LABELS = ["animals", "hobbies", "travel"]
HIGH_DIVERSITY_N_RESPONSES = 100

# ============================================================================
# Prompt labels for plotting/reporting
# ============================================================================
MULTI_MODE_PROMPT_LABELS = ["animals", "morning", "places", "food", "weekend"]
ONE_MODE_PROMPT_LABELS = ["animals", "morning", "places", "food", "weekend"]
MIXED_PROMPT_LABELS = ["ocean", "forest", "mountains", "weather", "city"]
