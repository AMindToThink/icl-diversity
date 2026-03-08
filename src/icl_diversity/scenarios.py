"""Synthetic scenario data for ICL diversity metric validation.

Five edge-case scenarios from the paper (Section 6.3), each with 5 prompts
and 10 responses. Used by both the test suite and the experiment scripts.

Scenarios:
1. Pure noise:               C ≈ 0, E ≈ 0, D ≈ 0
2. Multiple incoherent modes: C low, E > 0, D suppressed
3. Many coherent modes:      C high, E high, D high
4. One coherent mode:        C high, E moderate, D moderate
5. Mixed coherent+incoherent: high σ_ℓ, wide [D-, D+] band
"""

import random
import string
from typing import Callable

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
# Prompt labels for plotting/reporting
# ============================================================================

MULTI_MODE_PROMPT_LABELS = ["animals", "morning", "places", "food", "weekend"]
ONE_MODE_PROMPT_LABELS = ["animals", "morning", "places", "food", "weekend"]
MIXED_PROMPT_LABELS = ["ocean", "forest", "mountains", "weather", "city"]
